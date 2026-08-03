"""Sync-free MoE dispatch at the DRAFTER's shape — the one MoE path on the tail that no lever claims.

WHY THIS FILE EXISTS, i.e. the hole it closes. `v4_moe_decode` and `v4_moe_grouped` both open with

    if xv.size(0) != 1: return _REF_FORWARD(self, x, input_ids)

so every fast MoE path in this engine is gated on ONE token. The DSpark drafter's MoEs never see one
token. `DSparkBlock.forward_embed` builds a `dspark_block_size`-wide draft block (model.py:854) and
every block in `RingDrafter`'s stack runs its MoE at `[b, block_size, dim]` — 5 rows at the shipped
config, 3 on the CPU harness, never 1. So the drafter's `n_mtp_layers` MoEs fall through BOTH gates on
their first line and execute the vendored reference's own dispatch loop:

    counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()   # 1 drain
    for i in range(self.experts_start_idx, self.experts_end_idx):                          # 256 iters
        if counts[i] == 0: continue
        idx, top = torch.where(indices == i)                                               # 1 drain each
        y[idx] += expert(x[idx], weights[idx, top, None])

At s == 1 that is 1 + 6 drains, which is what `v4_moe_decode` removes. At the drafter's s == 5 the
`k` routed ids of five different tokens are five different draws, so up to `s * k` = 30 DISTINCT
experts are active per block and the loop pays 1 + 30. Three DSparkBlocks make it up to 93 device
drains per drafted round, on the tail, every round — the busiest box in the ring, running the SLOWEST
of the three MoE paths this engine has. A drain does not merely cost its own latency: it stalls the
queue, so the CPU cannot run ahead and every launch behind it pays full dispatch. That is exactly the
regime `v4_dspark_fast` measured — `advance_and_draft` 224.55 ms with the GPU idle for nearly all of
it, ~40 ms once the wasted intermediate forwards are collapsed away, with the three surviving MoEs
still on this loop.

WHAT THE ADVERSARIAL PASS GOT WRONG, recorded so nobody re-derives it. The hole was first read as "the
drafter's MoEs never get `bank_layout`, so `_expert_bank`'s 2 GiB headroom check declines and the
grouped kernel does not fire." The first half is true (`v4_stage.Stage.__init__` passes `self.layers`
and nothing else, and `DSparkTail` builds its blocks separately and later). The second half is not:
`grouped_forward` returns on its `xv.size(0) != 1` line, BEFORE `_expert_bank`, so the drafter reaches
neither the bank nor the headroom check — measured zero calls to each over drafted rounds
(tests/test_v4_moe_multi.py::test_the_drafter_never_reaches_the_expert_bank). Two consequences:
  * Giving the drafter the bank layout would buy NOTHING. The grouped kernel is s == 1 by
    construction — grouped-and-padded MoE is not token-count invariant, which is why it declines s > 1
    rather than claiming it — so a banked drafter MoE would still fall through the same first line.
    That is a lever that provably does nothing, and it is not shipped here.
  * The tail is in no danger from the lazy stack either. `_expert_bank` is what would stack a second
    ~10.28 GiB copy of the drafter's experts after the graph pools are pinned, and it is never
    reached. The drafter's memory profile is one copy of `mtp.*` and nothing else.
AND BANKING THE DRAFTER WOULD BE THE WRONG SHAPE FOR THE TAIL EVEN IF IT WERE FREE, which is the
argument to have ready when an s > 1 grouped kernel eventually makes it look attractive. `_tail_drafter`
builds the drafter LAZILY, at the first dspark job's reset (v4_pipe), so a tail that served a greedy
job first is already holding its per-layer CUDA-graph pools when those 10.28 GiB are requested. Today
that request is 256 separate ~4 MiB per-expert blocks per weight kind, which a fragmented allocator
serves comfortably. The bank layout replaces them with SIX contiguous requests per block — three of
1024 MiB and three of 64 MiB — on the box with ~6-8 GiB of headroom and the most pinned pools in the
ring. So if the drafter is ever banked, it must be banked in `DSparkTail.__init__`, between the
ModuleList construction and `load()`, with `preserve=False` — the same window `Stage.__init__` uses,
where every routed-expert byte is still uninitialised `torch.empty` and the release can come first —
and the drafter should be built EAGERLY at stage load rather than at the first dspark reset, so the
1 GiB requests land before the graph pools rather than after them.

WHAT THIS DOES INSTEAD — the same trick `v4_moe_decode` plays at s == 1, generalised to a small block,
with NO change to the arithmetic. The whole schedule is already in `indices`: ONE `.tolist()` of the
[T, k] routing tensor yields, for every expert, exactly the (token row, slot column) pairs
`torch.where(indices == i)` would return. Bucket them on the host in row-major order, ship them to the
device as ONE index tensor, and run the reference's own body per expert:

    y[idx] += self.experts[i](xv[idx], weights[idx, top, None])

1 + E drains becomes 1, the 256-iteration Python scan becomes a sort over the active ids, and the per
expert `indices == i` + `nonzero` launches disappear. The expert GEMMs themselves are untouched — this
removes dispatch, not work.

BIT-EXACT BY CONSTRUCTION, NOT BY TOLERANCE, and the argument is short because nothing is re-derived:
  * `torch.where` on a 2-D boolean returns its True positions in ROW-MAJOR order. The host bucketing
    walks tokens ascending, then slots ascending, which IS row-major, so `idx`/`top` are
    element-for-element the tensors the reference would have built — same values, same order, same
    int64 dtype, same device.
  * The expert loop runs in ASCENDING expert id over the same active set (`counts[i] == 0` skips
    exactly the experts absent from `indices`, which are exactly the ids absent from the buckets), so
    the fp32 accumulator receives the same addends in the same order. fp32 add is commutative but not
    associative; the order is the whole of the proof.
  * `y` opens at `torch.zeros_like(xv, dtype=torch.float32)`, the shared expert is added last, and the
    cast and reshape on the way out are the reference's. Duplicate ids within a token row (which only
    hash routing can produce) need no special case: the op is `y[idx] += ...` on the SAME `idx`, so
    whatever the reference's duplicate semantics are, they are reproduced rather than reasoned about.
Nothing here changes a GEMM's shape, which is the failure mode this engine has been bitten by twice
(v4_dspark_fast's batched advance, the tail's per-position verify logits): every expert still sees the
row count the reference gave it.

DECLINES, untouched, for:
  T == 1                the s == 1 owners — install order puts this ON TOP of grouped/decode, so a
                        single-token step is handed straight down to them and this file is invisible
                        to the main decode path.
  world_size > 1        the reference all-reduces the routed sum across ranks before the shared
                        expert; skipping a rank's experts without that reduction silently drops them.
  T > V4_MOE_MULTI_MAX  a prefill. The bucketing is `T * k` Python iterations, and at a 4096-token
                        prefill that is 24576 of them per layer — real host time traded against 256
                        drains, a DIFFERENT trade with a different answer, and one that wants its own
                        measurement rather than a free ride on this one. 32 rows covers what this file
                        is for: the drafter's `block_size` block and a verify chunk of `g + 1`.

EXPECTED WIN — ARITHMETIC, NOT MEASUREMENT. There is no GPU on the box this was written on, and
nothing below has been run on a card. Two measured anchors, both from this repo's own headers
(RTX 5090, real DeepSeek-V4-Flash weights):
    A1  reference MoE at s == 1        2.39 ms, 1 + 6 drains, ~120 launches   (v4_moe_decode)
    A2  drafter's kept forward_spec    ~40 ms per call, GPU idle throughout   (v4_dspark_fast)
A1 splits into ~120 launches at ~5-10 us (0.6-1.2 ms) and 7 drains over the rest, so a drain plus the
dispatch it stalls is ~0.15-0.25 ms. At the shipped config a drafter MoE sees 5 rows x 6 routed
experts = 28.6 DISTINCT experts on average (simulated over the 256-expert draw), so per DSparkBlock
the dispatch is 1 + ~29 drains and ~26 launches per active expert (~750), and over 3 blocks:
    reference   3 x (29 x ~0.2 ms + 750 x ~7 us)  =  ~25-33 ms   of A2's ~40
    this file   3 x ( 1 x ~0.2 ms + 700 x ~7 us)  =  ~15 ms      launch-dispatch bound, the floor
i.e. the drafter's kept forward should land near 22-30 ms, ~10-18 ms saved per call. The 25-33 ms
estimate is consistent with A2's own observation that the GPU is idle for nearly all of the 40 ms.

WHAT THAT IS WORTH depends on which round shape the ring is running, and the shipped recipe is the
better one for this lever:
  * CHUNKED (coordinate_dspark): one kept forward per ROUND, against a tail whose main path is
    ~92 ms (3 layers x 15.4 ms/step graphed, x a 6-position verify chunk — v4_moe_grouped's
    stage[40:43) table). ~10-18 ms off a ~140 ms round = 7-12% of the tail.
  * PIPELINED (coordinate_dspark_pipelined, and V4_PIPELINED_SPEC=1 is in the ring recipe): the
    drafter drafts on EVERY COMMITTED FRAME, not once per round, so the tail pays a full kept
    forward per committed token beside ~15.4 ms of its own layers. The drafter is then ~2.6x the
    tail's layer work, and ~10-18 ms off a ~55 ms frame is 18-33% of the tail's per-token cost.
Neither is the grouped kernel's 1.72x. It is a single-digit-to-low-double-digit percentage on the one
box in the ring with no slack, for ~90 lines that change no arithmetic. MEASURE IT before believing
it, and measure the pipelined shape, which is the one that ships.

Opt-in, default OFF: `V4_MOE_MULTI=1`. With the env unset `install()` binds nothing and every MoE in
the process is byte-identical to today. Unlike the grouped kernel this is pure torch, so it runs (and
is proved) on CPU.

self-test:  V4_MOE_MULTI=1 python3 phase0/v4_moe_multi.py
"""
import os

import torch

V4_MOE_MULTI = os.environ.get("V4_MOE_MULTI", "0") not in ("", "0")

# Rows above which the host-side bucketing costs more than the drains it removes — a prefill, not a
# draft block. `T * k` Python iterations against `E` device drains: at the drafter's 5 rows that is 30
# iterations against 31 drains, at a 4096-token prefill it is 24576 against 256. 32 is comfortably
# above `dspark_block_size` (5) and a verify chunk (`g + 1`), and far below where the trade inverts.
V4_MOE_MULTI_MAX = int(os.environ.get("V4_MOE_MULTI_MAX", "32"))

_REF_FORWARD = None                     # whatever MoE.forward was bound at install — our fallback
_WORLD_SIZE = 1


def multi_forward(self, x, input_ids):
    """MoE.forward for a small multi-token block, with the per-expert host drains removed.

    The reference's body, addend for addend, reached through a routing schedule read off the host ONCE
    instead of re-derived per expert with a `nonzero`. See the module docstring for the bit-exactness
    argument and for what it declines."""
    shape = x.size()
    xv = x.view(-1, self.dim)
    T = xv.size(0)
    if T == 1 or T > V4_MOE_MULTI_MAX or _WORLD_SIZE > 1:
        return _REF_FORWARD(self, x, input_ids)

    weights, indices = self.gate(xv, input_ids.flatten())
    # THE ONE HOST SYNC. `indices` is [T, k]; its rows in order are exactly what `torch.where` scans.
    sel = indices.tolist()
    buckets = {}
    for t, row in enumerate(sel):
        for c, e in enumerate(row):
            b = buckets.get(e)
            if b is None:
                buckets[e] = b = ([], [])
            b[0].append(t)                                  # row-major: token ascending, then slot
            b[1].append(c)
    # Ascending expert id over the LOCAL range — the reference's `for i in range(start, end)` with its
    # `counts[i] == 0` skip, which is exactly "the ids that appear in indices".
    order = [i for i in sorted(buckets) if self.experts_start_idx <= i < self.experts_end_idx]

    # ONE host->device transfer for every expert's indices, sliced per expert with device-side views.
    # E separate `torch.tensor(..., device=cuda)` calls would each be a blocking copy, trading the
    # drains away for stalls of the same shape.
    flat = [t for i in order for t in buckets[i][0]] + [c for i in order for c in buckets[i][1]]
    pairs = len(flat) // 2
    both = torch.tensor(flat, dtype=torch.long, device=xv.device)
    idx_all, top_all = both[:pairs], both[pairs:]

    y = torch.zeros_like(xv, dtype=torch.float32)
    off = 0
    for i in order:
        n = len(buckets[i][0])
        idx, top = idx_all[off:off + n], top_all[off:off + n]
        off += n
        y[idx] += self.experts[i](xv[idx], weights[idx, top, None])
    y += self.shared_experts(xv)
    return y.type_as(xv).view(shape)


def install(mod):
    """Rebind `mod.MoE.forward` to the small-block path. Returns True if it took.

    ON TOP OF the s == 1 levers, and the order is the precedence: it captures whatever forward is
    bound (grouped -> decode -> reference, per v4_ref_cpu.load_ref) and hands every single-token step
    straight back to it, so the main decode path is untouched and only the shapes nothing else claimed
    change hands. Idempotent, and a no-op under `V4_MOE_MULTI=0`."""
    global _REF_FORWARD, _WORLD_SIZE
    if not V4_MOE_MULTI or getattr(mod.MoE.forward, "_v4_multi", False):
        return False
    _REF_FORWARD = mod.MoE.forward
    _WORLD_SIZE = int(getattr(mod, "world_size", 1) or 1)
    multi_forward._v4_multi = True
    mod.MoE.forward = multi_forward
    return True


def uninstall(mod):
    """Restore the forward this install captured. For an in-process A/B; not on the serving path."""
    global _REF_FORWARD
    if _REF_FORWARD is None:
        return False
    mod.MoE.forward = _REF_FORWARD
    _REF_FORWARD = None
    return True


def _selftest():
    """Bit-exact parity against the reference dispatch at the DRAFTER's own shape, on CPU.

    Drives v4_dspark_draft's toy harness down both paths from the same weights and the same prompt and
    requires every draft block, logit, confidence and mtp KV buffer back `torch.equal`, then counts the
    `torch.where` calls the lever removed from one drafted round."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import v4_ref_cpu
    import v4_stage
    import v4_dspark_draft as D
    # THE MODULE THAT IS ACTUALLY INSTALLED, which is not necessarily this one. Run as a script this
    # file is `__main__`, and `load_ref`'s `import v4_moe_multi` then loads a SECOND copy of it —
    # different globals, different `_REF_FORWARD`. Flipping the flag or calling install/uninstall on
    # `__main__` would configure the copy nobody is using while the other copy stays bound, which is
    # exactly the "lever that silently does nothing" this file exists to kill. So the A/B drives the
    # installed module by name; when imported normally it IS this module and nothing changes.
    import v4_moe_multi as MM

    args = v4_ref_cpu.cpu_args()
    M = v4_stage.ref()
    oracle = v4_ref_cpu.build_oracle(args, 0)

    def build():
        st = v4_stage.Stage(0, args.n_layers, args, head=True, tail=True, dspark=True, device="cpu")
        for li in range(args.n_layers):
            st.layers[li].load_state_dict(oracle.layers[li].state_dict(), strict=True)
        st.embed_tokens.load_state_dict(oracle.embed.state_dict(), strict=True)
        st.norm.load_state_dict(oracle.norm.state_dict(), strict=True)
        st.lm_head.load_state_dict(oracle.head.state_dict(), strict=True)
        with torch.no_grad():
            for n in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                getattr(st, n).data.copy_(getattr(oracle, n).data)
        st._dspark = True
        dr = D.DSparkTail(st)
        for k, blk in enumerate(dr.mtp):
            sd = {n: v for n, v in oracle.mtp[k].state_dict().items() if n not in D.ALIAS_KEYS}
            blk.load_state_dict(sd, strict=False)
        return st, dr

    def run():
        st, dr = build()
        torch.manual_seed(0)
        ids = torch.randint(0, args.vocab_size, (1, 13))
        tok = st.logits_all(st.forward(st.embed(ids), ids, 0), full_logits=False).argmax(-1)
        dr.prefill(tok, st.tail_main_hidden())
        out = []
        for i in range(13, 17):
            h = st.forward(st.embed(tok.unsqueeze(1)), tok.unsqueeze(1), i)
            tok = st.logits_all(h, full_logits=False).argmax(-1)
            blk, conf = dr.advance_and_draft(tok.unsqueeze(1), st.tail_main_hidden(), start_pos=i)
            out.append((blk.clone(), conf.clone(), dr.last_spec[1].clone()))
        return out, [b.attn.kv_cache.clone() for b in dr.mtp]

    # `cpu_args()` above already ran `load_ref()`, which installs when the env asks — and the env is
    # exactly how this self-test is invoked. So the BASELINE leg has to UNINSTALL first: `multi_forward`
    # dispatches on the row count alone, not on the flag, so leaving it bound would run both legs down
    # the same path and the comparison would prove nothing while passing.
    MM.uninstall(M)
    MM.V4_MOE_MULTI = False
    ref_out, ref_kv = run()
    assert not getattr(M.MoE.forward, "_v4_multi", False), "the baseline leg must not be the lever"
    MM.V4_MOE_MULTI = True
    assert MM.install(M), "install must take when V4_MOE_MULTI is on"
    try:
        got_out, got_kv = run()
    finally:
        MM.uninstall(M)
    for i, ((rb, rc, rl), (gb, gc, gl)) in enumerate(zip(ref_out, got_out)):
        assert torch.equal(rb, gb), f"draft ids diverged round {i}"
        assert torch.equal(rc, gc), f"confidence diverged round {i}"
        assert torch.equal(rl, gl), f"draft logits diverged round {i}"
    for i, (rk, gk) in enumerate(zip(ref_kv, got_kv)):
        assert torch.equal(rk, gk), f"mtp {i} kv_cache diverged"
    print(f"[v4] drafter MoE: bit-exact over {len(ref_out)} drafted rounds "
          f"(s={args.dspark_block_size}, {args.n_mtp_layers} mtp blocks)", flush=True)


if __name__ == "__main__":
    _selftest()
