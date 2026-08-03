"""A CPU `fla` for Kimi-K3's KDA layers — so a layer range can be proven correct without a GPU.

Kimi's reference decoder (phase0/kimi_k3_ref/modeling_kimi_linear.py) hard-imports flash-linear-
attention at module scope:

    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda

Every one of those is Triton, i.e. GPU-only — `ShortConvolution.forward` dispatches into
`fla.modules.conv.triton`, and there is no torch fallback anywhere in the package. Importing the
reference file on a CPU box fails outright, and a stage whose parity can only be checked on rented
hardware is a stage whose parity is never checked. So this module supplies the same five names in
plain torch and registers them under `fla.*` in sys.modules BEFORE the reference is imported. The
reference file itself is untouched — it gets the API it asks for.

The math is not invented here. It is fla's own published reference, transcribed from
fla/ops/kda/naive.py (`naive_recurrent_kda`) and fla/ops/kda/gate.py (`naive_kda_gate`,
`naive_kda_lowerbound_gate`), both MIT (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li. The
preprocessing those two do not cover -- what fla folds into the kernel -- is read off
fla/ops/kda/fused_recurrent.py:149+:
  l2norm q,k as x/sqrt(sum(x*x)+1e-6)  ->  q *= K**-0.5  ->  gate  ->  sigmoid(beta)
  S *= exp(g);  S += (beta*k) (x) (v - k.S);  o = q.S
(the recurrence follows naive.py, which folds beta into k; the kernel folds it into v - k.S
instead. Same product, different association -- fp-identical here because both run in fp32.)
tests/test_k3_stage.py::test_cpu_kda_matches_flas_published_reference pins all of it.

DELIBERATE M1 SIMPLIFICATION: `chunk_kda` here IS `fused_recurrent_kda` — one sequential recurrence
for both prefill and decode. On a GPU the two are different kernels and are NOT bit-identical to each
other (chunked matmul reassociation), so this backend is the CPU reference, not an emulator of fla's
numerics. What it buys is exact: chunked prefill and token-by-token decode produce the same state, so
`tests/test_k3_stage.py::test_kda_state_continuity_across_chunked_prefill` measures the stage's state
threading rather than a kernel's chunk boundary. M2 swaps in real fla / FlashKDA (which builds sm_120
on CUDA >= 13) behind K3_KDA_BACKEND and re-proves parity against THIS on a GPU box.

State layouts are fla's, not ours, so the M2 swap is a drop-in: the conv cache is [N, D, W] holding
the last W inputs (newest last), and the recurrent state is [B, HV, K, V], or [B, HV, V, K] when the
caller passes transpose_state_layout=True (the reference always does).
"""
import importlib.util
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

# Which KDA kernels the reference gets. "fla" is the real package (Triton, needs a CUDA device),
# "cpu" is this module, "flashkda" is FlashKDA's CUDA recurrence with this module's conv/gated-norm
# around it (k3_kda_flash — the M2 GPU path, proven on sm_120), and "auto" picks a GPU backend only
# when both the package AND a GPU are present -- either package imports fine without one and then
# dies at the first kernel launch, so importability alone is the wrong test.
K3_KDA_BACKEND = os.environ.get("K3_KDA_BACKEND", "auto")
# Set by k3_stage when K3_STATIC_STATE (implied by K3_CUDA_GRAPH) is on: the conv cache is then
# updated IN PLACE and handed back, so `cache_params.conv_states[li] = ret` preserves the address a
# captured graph replays against. Off (the default, and every CPU parity test) a fresh tensor is
# returned, which is M1's behaviour exactly.
STATIC_STATE = False


# ── the ops the reference calls ──────────────────────────────────────────────────────────────────

def _l2norm(x):
    """fla's in-kernel l2 norm: the epsilon is INSIDE the sqrt, not added to the norm."""
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def kda_gate(g, A_log, dt_bias=None, lower_bound=None):
    """g -> the per-dimension log decay the recurrence consumes. fla/ops/kda/gate.py, both branches.

    `A_log` is indexed by VALUE head, and Kimi's checkpoint ships it wider than the head count (128
    entries for 96 heads); slicing it is the loader's job (k3_stage._load_kda_extras), because the
    reference declares the parameter at num_heads and would reject the wide tensor anyway."""
    H = g.shape[-2]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1).float()
    A = A_log.view(H, 1).float().exp()
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(A * g)
    return -A * F.softplus(g)


def fused_recurrent_kda(q, k, v, g, beta, A_log=None, dt_bias=None, initial_state=None,
                        output_final_state=False, scale=None, use_qk_l2norm_in_kernel=False,
                        use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
                        allow_neg_eigval=False, lower_bound=None, safe_gate=None,
                        transpose_state_layout=False, state_v_first=False,
                        cu_seqlens=None, **kwargs):
    """KDA's delta recurrence, token by token, in fp32. q,k: [B,T,H,K]  v,g: [B,T,HV,*]  beta: [B,T,HV].

    Returns (o [B,T,HV,V], final_state) with the state in fla's layout -- [B,HV,K,V], or [B,HV,V,K]
    V-first. fla spells that flag `state_v_first`; the reference decoder still calls it
    `transpose_state_layout`, so BOTH are accepted and mean the same thing. Swallowing the spelling
    we do not know into **kwargs would silently return a transposed state -- the same numbers, the
    wrong layout, no error -- which is exactly the handover M2 has to get right."""
    transpose_state_layout = transpose_state_layout or state_v_first
    if cu_seqlens is not None:
        raise NotImplementedError("k3_kda_cpu: varlen (cu_seqlens) is a batched-serving concern, "
                                  "deferred to M2 -- pass one unpadded sequence per call")
    dtype = v.dtype
    B, T, H, Kd = q.shape
    HV, Vd = v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = Kd ** -0.5

    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    if use_qk_l2norm_in_kernel:
        q, k = _l2norm(q), _l2norm(k)
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)
    if use_gate_in_kernel:
        g = kda_gate(g, A_log, dt_bias, lower_bound)
    if use_beta_sigmoid_in_kernel:
        beta = torch.sigmoid(beta)
        if allow_neg_eigval:
            beta = beta * 2

    S = q.new_zeros(B, HV, Kd, Vd)
    if initial_state is not None:
        init = initial_state.float()
        S = S + (init.transpose(-1, -2) if transpose_state_layout else init)
    o = torch.zeros_like(v)
    for i in range(T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum("bhk,bhv->bhkv", b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum("bhk,bhkv->bhv", q_i, S)

    if not output_final_state:
        return o.to(dtype), None
    final = (S.transpose(-1, -2) if transpose_state_layout else S).contiguous()
    if STATIC_STATE and initial_state is not None:
        initial_state.copy_(final)   # address-preserving update, same contract as k3_kda_flash's:
        return o.to(dtype), initial_state    # the cache slot a captured graph replays against
    return o.to(dtype), final


def chunk_kda(*args, **kwargs):
    """Prefill path. Deliberately the same sequential recurrence -- see this module's docstring."""
    return fused_recurrent_kda(*args, **kwargs)


def prepare_lens_from_mask(mask):
    return mask.sum(-1, dtype=torch.int32)


def prepare_cu_seqlens_from_mask(mask, out_dtype=torch.int32):
    return F.pad(prepare_lens_from_mask(mask).cumsum(0, dtype=out_dtype), (1, 0))


def tensor_cache(fn):
    """fla memoizes mask-derived indices; the CPU path just recomputes them."""
    return fn


# ── the nn.Modules the reference builds ──────────────────────────────────────────────────────────

class ShortConvolution(nn.Conv1d):
    """Depthwise causal short conv, parameterised exactly as fla's (weight [D,1,W], groups=D) so a
    checkpoint and a state_dict load into either implementation unchanged.

    The cache is fla's too: [N, D, W] = the last W inputs, newest last. A chunk therefore sees the
    previous chunk's final W-1 inputs as its left context, which is what makes chunked prefill agree
    with token-at-a-time decode -- the boundary this backend exists to keep honest."""

    def __init__(self, hidden_size, kernel_size, bias=False, activation="silu", backend=None,
                 device=None, dtype=None, **kwargs):
        # `padding` is declared to match fla's module exactly (repr, state_dict, any code that reads
        # it); forward() does NOT use it -- the causal left context comes from the cache instead, so
        # a chunk continues the previous one rather than restarting against zeros.
        super().__init__(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size,
                         groups=hidden_size, bias=bias, padding=kernel_size - 1,
                         device=device, dtype=dtype)
        self.hidden_size = hidden_size
        self.activation = activation
        if activation is not None and activation not in ("silu", "swish"):
            raise ValueError(f"k3_kda_cpu.ShortConvolution: unsupported activation {activation!r}")
        self.backend = "cpu"

    def forward(self, x, residual=None, mask=None, cache=None, output_final_state=False,
                cu_seqlens=None, chunk_indices=None, **kwargs):
        if cu_seqlens is not None:
            raise NotImplementedError("k3_kda_cpu: varlen (cu_seqlens) deferred to M2")
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        B, T, D = x.shape
        W = self.kernel_size[0]

        # left context: the newest W-1 cached inputs, or zeros on a cold sequence
        if cache is None:
            left = x.new_zeros(B, W - 1, D)
        else:
            left = cache[:, :, -(W - 1):].transpose(1, 2).to(x.dtype) if W > 1 else x.new_zeros(B, 0, D)
        xfull = torch.cat([left, x], dim=1)

        y = F.conv1d(xfull.transpose(1, 2), self.weight, self.bias, groups=self.groups)
        y = y.transpose(1, 2)
        if self.activation is not None:
            y = F.silu(y)
        if residual is not None:
            # fla adds the residual AFTER the activation (the Canon op). KDA never passes one, so
            # this line is unreached today -- it is here so it is not wrong when something does.
            y = y + residual

        if not output_final_state:
            return y, None
        # xfull is always >= W wide (W-1 of left context plus at least one new token), so the last
        # W columns are exactly the window the next chunk needs.
        window = xfull[:, -W:].transpose(1, 2)
        if STATIC_STATE and cache is not None:
            cache.copy_(window)          # address-preserving update; `left` was copied out by the
            return y, cache              # torch.cat above, so writing the cache now is safe
        return y, window.contiguous()


class FusedRMSNormGated(nn.Module):
    """RMSNorm with an output gate, in fp32: y = (x * rstd) * weight * act(g).

    fla/modules/fused_norm_gate.py:36 -- gating is applied AFTER the affine, and 'sigmoid' (what KDA
    asks for) is a plain sigmoid, not swish."""

    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, activation="swish",
                 device=None, dtype=None):
        super().__init__()
        if activation not in ("swish", "silu", "sigmoid"):
            raise ValueError(f"k3_kda_cpu.FusedRMSNormGated: unsupported activation {activation!r}")
        self.hidden_size, self.eps, self.activation = hidden_size, eps, activation
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def forward(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
        dtype = x.dtype
        h = x.float()
        if residual is not None:
            h = h + residual.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            h = h * self.weight.float()
        gf = g.float()
        h = h * (gf * torch.sigmoid(gf) if self.activation in ("swish", "silu") else torch.sigmoid(gf))
        return h.to(dtype)


# ── registration ─────────────────────────────────────────────────────────────────────────────────

_MODULES = {
    "fla": {},
    "fla.modules": {"FusedRMSNormGated": FusedRMSNormGated, "ShortConvolution": ShortConvolution},
    "fla.ops": {},
    "fla.ops.kda": {"chunk_kda": chunk_kda, "fused_recurrent_kda": fused_recurrent_kda},
    "fla.ops.utils": {},
    "fla.ops.utils.index": {"prepare_cu_seqlens_from_mask": prepare_cu_seqlens_from_mask,
                            "prepare_lens_from_mask": prepare_lens_from_mask},
    "fla.utils": {"tensor_cache": tensor_cache},
}


def _installed(name):
    """Is `name` a REAL importable module, as opposed to a bare directory that PEP-420 would turn
    into an empty namespace package? A stray `fla/` on sys.path -- a scratch checkout, an unpacked
    wheel next to a script -- imports fine and exports nothing, so `import fla` succeeding is not
    evidence that flash-linear-attention is installed. A namespace package has origin None."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return False
    return spec is not None and spec.origin is not None


def backend():
    """Which KDA backend this box resolves to under K3_KDA_BACKEND. See the constant's comment.

    `auto` prefers flashkda over fla on a GPU: FlashKDA is the recurrence M2 actually measured on
    sm_120, and fla's own KDA Triton kernels were never probed there."""
    if K3_KDA_BACKEND in ("cpu", "fla", "flashkda"):
        return K3_KDA_BACKEND
    if K3_KDA_BACKEND != "auto":
        raise RuntimeError(f"K3_KDA_BACKEND={K3_KDA_BACKEND!r} — expected auto, cpu, fla or flashkda")
    if not torch.cuda.is_available():
        return "cpu"
    import k3_kda_flash
    if k3_kda_flash.available():
        return "flashkda"
    return "fla" if _installed("fla.ops.kda") else "cpu"


# The names the reference imports that a non-`fla` backend has to supply. `_OVERRIDE` is the subset
# that gets written ONTO a real fla package when one is installed: the two KDA entry points and the
# two nn.Modules. `fla.ops.utils.index` and `fla.utils` are left alone there -- the real ones are
# correct, and fla's memoizing tensor_cache is better than the passthrough stub.
_OVERRIDE = ("fla.modules", "fla.ops.kda")


def _table(be):
    t = {k: dict(v) for k, v in _MODULES.items()}
    if be == "flashkda":
        import k3_kda_flash
        t["fla.ops.kda"] = {"chunk_kda": k3_kda_flash.chunk_kda,
                            "fused_recurrent_kda": k3_kda_flash.fused_recurrent_kda}
    return t


def install():
    """Give the reference its KDA ops, and report which backend it got.

    Three shapes, because `fla` is both a thing we replace and a thing we depend on:
      fla        nothing to do -- the real package serves every name.
      real fla   present but a different backend was resolved: the four names the reference imports
                 are written ON the real modules and the rest of the package is left intact.
                 Replacing it wholesale would strand `fla.ops.attnres` -- a stub `fla.ops` has no
                 __path__, so the submodule cannot be imported at all -- and the M2 box needs
                 fla-core for exactly that. It DOES mean the override is process-wide: anything else
                 importing fla.ops.kda gets this backend. That is what the knob asked for.
      no fla     build the stub tree under sys.modules, M1's original behaviour.

    M1 raised here instead of overriding, to avoid clobbering a real fla another module was using.
    That refusal is gone for two reasons: the surgical override is a far smaller clobber than the
    package replacement it was guarding against, and the check could not tell a genuine user of the
    package from `_installed()`'s own probe -- find_spec on a dotted name imports the parent, so
    merely ASKING whether fla existed made the guard fire. Idempotent either way."""
    be = backend()
    if be == "fla":
        return "fla"
    table = _table(be)
    if _installed("fla.ops.kda"):
        for name in _OVERRIDE:
            mod = importlib.import_module(name)
            for attr, value in table[name].items():
                setattr(mod, attr, value)
        return be
    for name, attrs in table.items():
        mod = sys.modules.get(name)
        if mod is None or not getattr(mod, "_k3_cpu_backend", False):
            mod = types.ModuleType(name)
            mod._k3_cpu_backend = True
            sys.modules[name] = mod
        for attr, value in attrs.items():
            setattr(mod, attr, value)
    for parent, child in (("fla", "modules"), ("fla", "ops"), ("fla", "utils"),
                          ("fla.ops", "kda"), ("fla.ops", "utils"), ("fla.ops.utils", "index")):
        setattr(sys.modules[parent], child, sys.modules[f"{parent}.{child}"])
    return be


def set_static_state(on):
    """Turn address-preserving state updates on for every backend at once (K3_STATIC_STATE).

    Both the conv cache (here) and the KDA recurrent state (k3_kda_flash) have to agree: a graph
    that captured one fixed buffer and one reallocating one replays against a stale half."""
    global STATIC_STATE
    STATIC_STATE = bool(on)
    import k3_kda_flash
    k3_kda_flash.STATIC_STATE = bool(on)


__all__ = ["backend", "install", "set_static_state", "chunk_kda", "fused_recurrent_kda", "kda_gate",
           "ShortConvolution", "FusedRMSNormGated"]
