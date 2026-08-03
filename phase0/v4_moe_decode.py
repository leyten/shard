"""Sync-free MoE decode path for the V4 reference — the per-layer cost the ring actually pays.

MEASURED (RTX 5090, real weights, layer 7 of DeepSeek-V4-Flash, one token):
    whole layer   3.77 ms      of which  MoE 2.39 ms,  attention + hyper-connections 1.41 ms
The MoE half is 21 tiny fp4 GEMMs — a token through 6 routed experts plus the shared one. That is
microseconds of arithmetic. Nearly all of the 2.39 ms is the CPU getting there, and the reference's
own dispatch loop is why:

    counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()   # sync
    for i in range(self.experts_start_idx, self.experts_end_idx):                          # 256 iters
        if counts[i] == 0: continue
        idx, top = torch.where(indices == i)                                               # sync
        y[idx] += expert(x[idx], weights[idx, top, None])

`.tolist()` drains the device to get the host loop bound; each `torch.where` is a `nonzero()`, which
drains it again to learn its own output size. That is 1 + k drains per layer per token — 7 at the
shipped `n_activated_experts: 6` — and a drain does not just cost its own latency: it stalls the
queue, so the CPU can never run ahead and every subsequent launch pays full dispatch latency. On a
7-layer stage that is 49 device drains per decode token.

At b*s == 1 every answer those drains buy is already in the k routed ids. ONE `.tolist()` of a
6-element tensor yields the whole schedule, and the per-expert `indices == i` / `nonzero` / gather /
scatter kernels vanish with it.

BIT-EXACT, NOT APPROXIMATELY EQUAL, and by construction rather than by tolerance: the same experts
run, in the same ascending-expert-id order, into the same fp32 accumulator, on the same operands
(`x[idx]` at one row is `x`; `weights[idx, top, None]` at one row is `weights[:, k, None]`; and
`0.0 + a == a` exactly, so opening the accumulator at zero changes nothing). Anything that is not a
single-token step -- prefill, a speculation chunk verified at s > 1, a routing that picked the same
expert twice -- falls back to the reference's own code, untouched.

This does not edit the vendored reference; it rebinds `MoE.forward` after model.py is executed, the
way v4_sparse_attn_sm120.install_sm120() rebinds a kernel before it. `V4_MOE_DECODE=0` restores the
reference path for an A/B.
"""
import os

import torch

V4_MOE_DECODE = os.environ.get("V4_MOE_DECODE", "1") not in ("", "0")

_REF_FORWARD = None                     # the reference's own MoE.forward, kept for the fallbacks
_WORLD_SIZE = 1                         # model.py's, read at install (see the world_size fallback)


def decode_forward(self, x, input_ids):
    """MoE.forward with the decode-shape host syncs removed. See the module docstring."""
    shape = x.size()
    xv = x.view(-1, self.dim)
    if xv.size(0) != 1 or _WORLD_SIZE > 1:              # prefill / a verified chunk: reference path.
        return _REF_FORWARD(self, x, input_ids)         # world_size > 1 too: the reference all_reduces
                                                        # the routed sum across ranks before the shared
                                                        # expert, and skipping a rank's experts without
                                                        # that reduction silently drops them. A stage
                                                        # refuses world_size != 1 anyway, but load_ref()
                                                        # is a lower-level door with no such guard.
    weights, indices = self.gate(xv, input_ids.flatten())
    sel = indices[0].tolist()                           # the ONE host sync a decode step needs
    if len(set(sel)) != len(sel):                       # a repeated expert: `y[idx] +=` with duplicate
        return _REF_FORWARD(self, x, input_ids)         # indices is the reference's semantics, not ours
    y = torch.zeros_like(xv, dtype=torch.float32)
    for k in sorted(range(len(sel)), key=lambda j: sel[j]):     # ascending expert id, as the loop goes
        i = sel[k]
        if not (self.experts_start_idx <= i < self.experts_end_idx):
            continue                                    # another TP rank owns it; all_reduce would sum it
        y += self.experts[i](xv, weights[:, k, None])
    y += self.shared_experts(xv)
    return y.type_as(xv).view(shape)


def install(mod):
    """Rebind `mod.MoE.forward` to the decode fast path. Returns True if it took.

    Unlike install_sm120 this runs AFTER model.py is executed: it replaces a method on a class the
    exec created, so there is nothing to rebind before. Idempotent, and a no-op under
    V4_MOE_DECODE=0 or on a module already installed (load_ref memoizes, but a test may not).

    It also refuses when EITHER of the levers that install above it is already on top, which is not an
    idempotence check but a CYCLE guard: both capture `decode_forward` as their fallback, so
    re-installing over one of them would capture ITS forward here and make the two call each other
    forever -- a RecursionError on the first prefill, not a slow path. load_ref runs the three exactly
    once, so nothing reaches this today; it is the trap any second install path would fall into.
    `_v4_multi` was missing from this guard until the lever audit went looking for a chain that could
    eat itself and found that v4_moe_multi could close one."""
    global _REF_FORWARD, _WORLD_SIZE
    if not V4_MOE_DECODE or getattr(mod.MoE.forward, "_v4_decode_fast", False):
        return False
    if getattr(mod.MoE.forward, "_v4_grouped", False) or getattr(mod.MoE.forward, "_v4_multi", False):
        return False
    _REF_FORWARD = mod.MoE.forward
    _WORLD_SIZE = int(getattr(mod, "world_size", 1) or 1)
    decode_forward._v4_decode_fast = True
    mod.MoE.forward = decode_forward
    return True
