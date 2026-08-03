"""The DRAFTER's MoE as one grouped fp4 launch per matrix kind — the tail's 9.16 ms draft, degrouped.

MEASURED (research/v4_profile_draft.py, shipped structure — 5 draft rows x 6 routed experts x 3 mtp
blocks over 256-expert score-routed MoEs — anchored to the live 9.16 ms/frame draft on the 07-31
ring, V4_MOE_MULTI leg, which is the recipe the ring ran):
    one draft = 1221 GPU launches + 3 host drains, and it solves to 7.0 us/launch — inside the
    4-10 us band this repo's own anchors give (v4_moe_decode's header). The split:
        MoE (3 blocks)        791 launches   ~6.1 ms   67%   <- expert GEMM chains 3.7, dispatch 2.0
        attention             210            ~1.5 ms   16%
        hyper-connections      96            ~0.7 ms
        head + embed + misc   125            ~0.9 ms
So the drafter's MoE is two thirds of the round that binds the whole ring, and nearly all of it is
the CPU issuing ~29 experts' worth of per-expert GEMM stacks (~26 launches each) that move
microseconds of fp4 arithmetic. This is the same disease `v4_moe_grouped` cured at s == 1; the
drafter runs at s == dspark_block_size and no fast path claims that shape (`v4_moe_multi` collapsed
the drains but explicitly not the launch count — its header says so).

WHY THE s == 1 GROUPED KERNEL COULD NOT SIMPLY CLAIM s > 1, and what this does instead. The M2.5
finding stands: grouped-AND-PADDED MoE is not token-count invariant — padding an expert's group to a
common M reassociates its K-reduction and moves bits. But the drafter's shape does not need padding.
A block routes T x k = `dspark_block_size * n_activated_experts` = 30 (row, expert) PAIRS at the
shipped config, and `v4_moe_grouped`'s kernel is not a padded-group kernel at all: its grid is one
block per (N-tile, slot), each slot g computing C[g] = A[g] @ W[g] as an INDEPENDENT single-row
GEMV — the A tile is shared, only row g is stored, and no output element ever reduces across rows.
So the pairs go in AS the slots: A holds the pair's quantized activation row, W holds the pair's
gathered expert, one launch fills every pair of a matrix kind, and 30 <= block_M = 32 means the
shipped shape fits the existing kernel with zero kernel changes. RENT-NOT-REWRITE: the kernel, the
bank layout, and the fp4 gather are all `v4_moe_grouped`'s, called, not copied.

THE DISPATCH, per drafter MoE — compare `v4_moe_grouped.grouped_forward`, which is this at T == 1:
    weights, indices = gate(xv, ids)                  # the reference's own routing, T x k
    order = argsort(indices, dim=1)                   # per-row ascending expert id (device, no sync)
    flat  = indices.gather(1, order).reshape(-1)      # pair -> expert, row-major
    xq,xs = act_quant(xv)                             # ONE quant of the T rows (per-row, so exact)
    A     = xq[row_of_pair]                           # gather: pair p reads row p // k
    both  = grouped_fp4_gemm(A, gather(w13, flat))    # ONE launch: every pair's w1 AND w3
    h     = weights * swiglu(both)                    # the reference's own torch ops, batched
    out   = grouped_fp4_gemm(quant(h), gather(w2, flat))   # ONE launch: every pair's w2
    y[r] += out[pairs of r] in per-row ascending id order, then + shared_experts(xv)
~50 launches and ZERO host syncs against the reference's ~29-expert loop — and sync-free matters
twice: it is also the property that makes the drafter's MoE CUDA-graph-capturable later, the same
precondition v4_whole_layer_graph just proved for the main layers (the drafter's lazy build order
is the remaining blocker there, not the routing).

BIT-EXACT — the drafter's proposals must not move, because a changed proposal changes acceptance
and therefore every measurement downstream. The bar is `torch.equal` on (output_ids, logits,
confidence) against the untouched reference, and the argument decomposes exactly like
`v4_moe_grouped`'s:
  * per-row ops are row-invariant: `act_quant` is a per-(row, 128-group) reduction (kernel.py's
    grid), so quantizing the T rows once and gathering equals the reference quantizing each
    expert's row subset; SwiGLU/clamp/routing-weight/casts are elementwise on the same values.
  * the accumulation ORDER is the reference's: `y[idx] +=` walking experts in ascending id gives
    each ROW its addends in ascending expert id; the per-row `argsort` and the k-slot fold add the
    same fp32 words in the same per-element sequence. The shared expert is the reference's own call
    at the reference's own shape, added last.
  * the GEMM M-shape: the reference runs expert e as ONE M=count_e GEMM; this runs count_e
    independent rows. For the tilelang kernel those are bit-identical BY CONSTRUCTION — every
    output element is a function of its own A row and W row with a fixed block_K reduction order —
    the same ROW-INVARIANCE `v4_moe_grouped` already stakes its hash-duplicate path on (its M=2
    duplicate runs as two 1-row slots, pinned bit-exact on hardware). CPU torch.matmul does NOT
    have that property, so OFF CUDA the pair GEMMs are computed at the REFERENCE'S OWN M-grouping
    (one `kernel.fp4_gemm` per distinct expert per matrix kind, w1 and w3 separate) — which makes
    the CPU suite's `torch.equal` proof of the whole dispatch DETERMINISTIC, and confines the
    kernel-numeric claim to the hardware tests, exactly where v4_moe_grouped's already lives.

INSTALLED PER INSTANCE, NEVER ON THE CLASS — the strongest possible losslessness statement for the
verifier. The three MoE levers rebind `MoE.forward` for every MoE in the process; this binds a
method on the DRAFTER'S three MoE instances only (`install_drafter`, called from `ring_drafter`
between DSparkTail construction and load). The main model's verify and decode paths run byte-for-
byte the code they ran yesterday — not "bit-exact by argument" but UNTOUCHED BY CONSTRUCTION — and
every shape this path declines falls through to the live class chain (multi -> grouped -> decode ->
reference), composing with whatever the ring recipe installed.

THE BANK, and the tail's memory. The pair gather needs `v4_moe_grouped`'s contiguous per-layer
bank. The drafter never got one (bug 4 in v4_levers' ledger: banking it alone was a no-op because
`grouped_forward` bails at s != 1 — v4_moe_multi's header records the whole post-mortem), and
v4_moe_multi's WRONG-SHAPE warning is honoured here: the bank is laid at `install_drafter` time —
between the ModuleList construction and `load()`, the same window `Stage.__init__` uses — with
`preserve=False`, so the release comes first, the checkpoint loads THROUGH the views into the bank,
and the tail holds ONE copy of `mtp.*` exactly as before (+0 bytes steady, +0 peak). The lazy
`_expert_bank` stack is deliberately NOT reachable from here: a drafter MoE with no bank DECLINES
to the class chain rather than requesting a 3.2 GiB duplicate on the most VRAM-pinned box in the
ring. If the three ~2 GiB bank requests themselves ever fail at the first dspark job (pools pinned
first — the drafter is built lazily), v4_pipe's dspark-unavailable guard serves the job greedily
and the fix is the one v4_moe_multi already names: build the drafter eagerly at stage load.

EXPECTED ON THE RING, and what falsifies it. Post-lever the draft is ~580 launches + 0 syncs
(profiler, V4_DSPARK_MOE leg), which at the anchored 7 us/launch is ~4.1 ms against today's 9.16 —
tail on_box 14.64 -> ~9.6 ms, under the next-slowest stage's 10.04, so the frame ceiling should
move OFF the tail (68.3 -> ~99 frames/s bound elsewhere). FALSIFIED IF: a post-lever draft still
> ~6 ms with `coverage()` showing every block grouped — that would mean the residue (attention +
hc + head, ~2.6 ms modeled) is bigger than modeled and the next lever is capturing the now
sync-free drafter in a graph, not this one. Opt-in, default OFF: `V4_DSPARK_MOE=1`, registered in
v4_levers, observed via the note `ring_drafter` records at build.

self-test:  V4_DSPARK_MOE=1 python3 phase0/v4_dspark_moe.py
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

V4_DSPARK_MOE = os.environ.get("V4_DSPARK_MOE", "0") not in ("", "0")

_MOD = None                  # the loaded dsv4_model — scale_fmt / scale_dtype / act_quant globals
_WORLD_SIZE = 1
_ROWS = {}                   # (T, k, device) -> the pair -> row index, built once


def _grouped():
    """v4_moe_grouped, imported on first use — the kernel, the bank layout and the fp4 gather are
    rented from it, and importing it must stay free of the CUDA toolchain (it is)."""
    import v4_moe_grouped
    return v4_moe_grouped


def _row_of_pair(T, k, device):
    """[T*k] long: pair p was routed for activation row p // k. Static per shape, so memoized —
    rebuilding an arange per call would be two launches paid every block of every round."""
    key = (T, k, str(device))
    r = _ROWS.get(key)
    if r is None:
        r = _ROWS[key] = torch.arange(T, device=device).repeat_interleave(k)
    return r


def _take_rows(t, idx):
    """t[idx] for any dtype, contiguous. fp8/e8m0 lack index_cuda (same hole `_gather_fp` plugs);
    the uint8 view is free, shape-preserving on dim 0, and always implemented."""
    u = t.view(torch.uint8)
    return u[idx].view(t.dtype).view(len(idx), *t.shape[1:])


def _pair_gemms_cuda(xq, xs, flat, bank, scale_dtype):
    """CUDA: both matrix kinds as one grouped launch each, every pair an independent slot.

    The gathers are `v4_moe_grouped._gather_fp` (device-side, no host sync) and the kernel is its
    `grouped_fp4_gemm` — slot g stores C[g] = A[g] @ W[g], which for slot = (row, expert) pair is
    exactly the reference's per-expert GEMM row (module docstring: ROW-INVARIANCE)."""
    G = _grouped()
    rows = _row_of_pair(xq.size(0), flat.numel() // xq.size(0), xq.device)
    a, s = _take_rows(xq, rows), _take_rows(xs, rows)
    both = G.grouped_fp4_gemm(a, s, G._gather_fp(bank["w13"], flat),
                              G._gather_fp(bank["w13_s"], flat), scale_dtype)

    def w2(hq, hs):
        return G.grouped_fp4_gemm(hq, hs, G._gather_fp(bank["w2"], flat),
                                  G._gather_fp(bank["w2_s"], flat), scale_dtype)
    return both, w2


def _pair_gemms_cpu(xq, xs, flat, bank, scale_dtype, inter):
    """Off CUDA: the SAME pairs at the REFERENCE'S OWN GEMM shapes — one `kernel.fp4_gemm` per
    distinct expert per matrix kind (w1 and w3 separate), rows gathered in ascending row order,
    results scattered back to their pair slots.

    This is what makes the CPU suite's parity proof deterministic: `torch.matmul` (inside the CPU
    stand-in) is not row-invariant, so running the pairs one row at a time would differ from the
    reference in last bits by BLAS reassociation, not by any property of this dispatch. Holding the
    reference's M per expert proves everything above the kernel — routing, gather, quantize-once,
    fold order — with the kernel numerics identical by identity. See the module docstring."""
    import kernel
    T = xq.size(0)
    k = flat.numel() // T
    by_expert = {}
    for p, e in enumerate(flat.tolist()):
        by_expert.setdefault(e, []).append(p)

    def run(w_key, lo, hi, aq, as_, row_of):
        """One matrix kind, the reference's shapes: per distinct expert, its count_e rows as ONE
        M=count_e GEMM (ascending pair order == the reference's ascending row order), scattered
        back to the pair slots. `row_of` maps a pair to its row in `aq` — the TOKEN row for the
        first GEMM (aq is the T quantized activations), the PAIR itself for w2 (aq is the P
        weighted intermediates)."""
        out = None
        for e in sorted(by_expert):
            ps = by_expert[e]
            rows = torch.tensor([row_of(p) for p in ps], dtype=torch.long)
            a, s = _take_rows(aq, rows), _take_rows(as_, rows)
            w = bank[w_key][e, lo:hi].contiguous()
            ws = bank[w_key + "_s"][e, lo:hi].contiguous()
            c = kernel.fp4_gemm(a, s, w, ws, scale_dtype)
            if out is None:
                out = c.new_empty(flat.numel(), c.size(-1))
            out[torch.tensor(ps, dtype=torch.long)] = c
        return out

    g1 = run("w13", 0, inter, xq, xs, lambda p: p // k)
    u3 = run("w13", inter, 2 * inter, xq, xs, lambda p: p // k)
    both = torch.cat([g1, u3], dim=-1)

    def w2(hq, hs):
        return run("w2", 0, bank["w2"].size(1), hq, hs, lambda p: p)
    return both, w2


def _decline(self, why, x, input_ids):
    """Record the reason per instance (the v4_moe_grouped pattern — a lever that silently does
    nothing is bug 4's whole genus) and hand the step to the live CLASS chain, so this composes
    with whatever multi/grouped/decode installed."""
    tally = getattr(self, "_draft_declined", None)
    if tally is None:
        tally = self._draft_declined = {}
    tally[why] = tally.get(why, 0) + 1
    return type(self).forward(self, x, input_ids)


def draft_forward(self, x, input_ids):
    """MoE.forward for the drafter's fixed T-row block — bound per INSTANCE by `install_drafter`.
    Everything it does not claim falls through to the class chain. See the module docstring."""
    shape = x.size()
    xv = x.view(-1, self.dim)
    T = xv.size(0)
    k = self.n_activated_experts
    if T <= 1:
        return _decline(self, "s<=1", x, input_ids)
    if _WORLD_SIZE > 1:
        return _decline(self, "world_size>1", x, input_ids)
    if self.gate.hash:
        return _decline(self, "hash-routed", x, input_ids)
    if T * k > _grouped()._BLOCK_M:
        return _decline(self, "pairs>block_M", x, input_ids)
    bank = getattr(self, "_grouped_bank", None)
    if not bank:
        # no bank, no lazy stack: the tail is the wrong box to duplicate 3.2 GiB on (module doc)
        return _decline(self, "no-bank", x, input_ids)
    inter = bank["w13"].size(1) // 2
    if xv.is_cuda and (bank["w13"].size(1) % _grouped()._BLOCK_N
                       or bank["w2"].size(1) % _grouped()._BLOCK_N):
        return _decline(self, "N%block_N", x, input_ids)

    weights, indices = self.gate(xv, input_ids.flatten())
    # Per-row ascending expert id — the order the reference's expert loop feeds each row in. topk
    # cannot repeat within a row on a score gate, so the order is strict and the fold below adds
    # each row's experts in exactly the reference's sequence.
    order = torch.argsort(indices, dim=1)
    flat = indices.gather(1, order).reshape(-1)
    wts = weights.gather(1, order).reshape(-1, 1)          # fp32, pair-aligned

    scale_fmt, scale_dtype = _MOD.scale_fmt, _MOD.scale_dtype
    act_quant = _MOD.act_quant

    # ONE quant of the T rows; per-row, so gathering its rows equals the reference quantizing each
    # expert's subset (module doc). The pair GEMMs differ per device — the dispatch does not.
    xq, xs = act_quant(xv, _MOD.block_size, scale_fmt, scale_dtype)
    pair = _pair_gemms_cuda if xv.is_cuda else _pair_gemms_cpu
    args = (xq, xs, flat, bank, scale_dtype) + (() if xv.is_cuda else (inter,))
    both, w2 = pair(*args)

    # SwiGLU + clamp + routing weight + cast: the reference's own torch ops (Expert.forward), one
    # row per pair, elementwise and therefore batch-invariant.
    g = both[:, :inter].float()
    u = both[:, inter:].float()
    lim = self.experts[self.experts_start_idx].swiglu_limit
    if lim > 0:
        u = torch.clamp(u, min=-lim, max=lim)
        g = torch.clamp(g, max=lim)
    h = torch.nn.functional.silu(g) * u
    h = wts * h
    h = h.to(xv.dtype)
    hq, hs = act_quant(h, _MOD.block_size, scale_fmt, scale_dtype)
    out = w2(hq, hs)

    # The fold: slot j of every row simultaneously, j ascending — per element that is the same fp32
    # add sequence the reference's ascending-id `y[idx] +=` performs (module doc), then the shared
    # expert, the reference's own call at the reference's own shape, last.
    y = torch.zeros_like(xv, dtype=torch.float32)
    outT = out.view(T, k, -1)
    for j in range(k):
        y += outT[:, j]
    y += self.shared_experts(xv)
    self._draft_steps = getattr(self, "_draft_steps", 0) + 1
    return y.type_as(xv).view(shape)


def install_drafter(dstail):
    """Bank + bind the drafter's MoEs, per instance. Returns how many took it.

    Called by `ring_drafter` between DSparkTail construction and `load()` — the release-first
    window where every routed-expert byte is still `torch.empty` garbage and `preserve=False` costs
    nothing (v4_moe_grouped._relayout_moe's contract; v4_moe_multi's warning says this is the ONLY
    shape a drafter bank may take on the tail). A harness whose weights are already real must lay
    the bank itself with `preserve=True` first — an existing bank is used as found. No-op unless
    V4_DSPARK_MOE; declines (fp4-only, score-gate-only) leave the block on the class chain."""
    if not V4_DSPARK_MOE:
        return 0
    global _MOD, _WORLD_SIZE
    import v4_stage
    _MOD = v4_stage.ref()
    _WORLD_SIZE = int(getattr(_MOD, "world_size", 1) or 1)
    G = _grouped()
    took = 0
    for blk in dstail.mtp:
        moe = getattr(blk, "ffn", None)
        if moe is None or moe.gate.hash:
            continue
        e0 = moe.experts[moe.experts_start_idx]
        if e0.w1.weight.dtype != torch.float4_e2m1fn_x2:
            continue
        if not getattr(moe, "_grouped_bank", None):
            G._relayout_moe(moe, preserve=False)
        if not getattr(moe, "_grouped_bank", None):
            continue
        moe.forward = types.MethodType(draft_forward, moe)
        took += 1
    return took


def coverage(dstail):
    """{layer_id: (pair-path steps, {reason: declines})} over the drafter's MoEs — the answer to
    "did the lever actually FIRE", per block, the question bug 4 was made of."""
    out = {}
    for blk in dstail.mtp:
        moe = blk.ffn
        out[moe.layer_id] = (getattr(moe, "_draft_steps", 0),
                             dict(getattr(moe, "_draft_declined", {})))
    return out


# ── the fp4 toy harness (tests + research/v4_profile_draft.py) ───────────────────────────────────

def swap_in_fp4_moes(dstail, moe_inter_dim=128, seed=0):
    """Give a CPU-toy DSparkTail REAL fp4 routed experts, so the pair path has something to claim.

    The bf16 toy harness (v4_ref_cpu.cpu_args) keeps experts unquantized, which the lever rightly
    declines — so parity tests would only ever prove the fallback. This swaps each mtp block's MoE
    for one with valid packed-fp4 + e8m0 weights (through the reference's own `fp4_act_quant`,
    the v4_moe_grouped harness's approach) at the drafter's own dim/topk/expert count, banks it
    with `preserve=True` (the weights are real and must survive), and leaves the gate score-routed.
    The shared expert stays bf16 — both paths call it identically, so it proves nothing quantized."""
    from kernel import fp4_act_quant
    import v4_stage
    mod = v4_stage.ref()
    a0 = dstail.args
    a = mod.ModelArgs(dim=a0.dim, moe_inter_dim=moe_inter_dim,
                      n_routed_experts=a0.n_routed_experts,
                      n_activated_experts=a0.n_activated_experts, n_shared_experts=1,
                      n_hash_layers=0, score_func=a0.score_func, route_scale=a0.route_scale,
                      swiglu_limit=a0.swiglu_limit, expert_dtype="fp4", dtype="bf16",
                      scale_fmt=None, scale_dtype="fp32", vocab_size=a0.vocab_size)
    g = torch.Generator().manual_seed(seed)
    for bi, blk in enumerate(dstail.mtp):
        with mod.set_dtype(torch.bfloat16):
            moe = mod.MoE(blk.layer_id, a)
        with torch.no_grad():
            for i in range(a.n_routed_experts):
                e = moe.experts[i]
                for lin, out_f, in_f in ((e.w1, a.moe_inter_dim, a.dim),
                                         (e.w3, a.moe_inter_dim, a.dim),
                                         (e.w2, a.dim, a.moe_inter_dim)):
                    w, s = fp4_act_quant(torch.randn(out_f, in_f, generator=g,
                                                     dtype=torch.bfloat16), mod.fp4_block_size)
                    lin.weight.data.copy_(w)
                    lin.scale.data.copy_(s)
            for lin in (moe.shared_experts.w1, moe.shared_experts.w2, moe.shared_experts.w3):
                lin.weight.data.normal_(0, 0.02, generator=g)
            moe.gate.weight.data.normal_(0, 0.02, generator=g)
            moe.gate.bias.data.normal_(0, 0.02, generator=g)
        # Bank HERE, preserve=True — the weights just written are real and must survive. This is
        # the harness's half of install_drafter's contract: an existing bank is used as found, so
        # a later install can never void a loaded toy with the release-first layout.
        assert _grouped()._relayout_moe(moe, preserve=True), "fp4 toy bank must take"
        blk.ffn = moe.eval()
    return dstail


def _selftest():
    """End to end, on CPU: the fp4-drafted toy down both paths, every proposal `torch.equal`.

    The v4_moe_multi selftest's shape: the same weights and the same recorded rounds through the
    reference dispatch and through the pair path, and every draft block, logit, confidence and mtp
    KV buffer has to come back bit-identical — plus coverage() proving the fast path actually
    served every block of every round rather than quietly declining (bug 4's test)."""
    import v4_ref_cpu
    import v4_stage
    import v4_dspark_draft as D
    import v4_dspark_moe as DM              # the module by its installed name (v4_moe_multi's trap)

    args = v4_ref_cpu.cpu_args(n_routed_experts=64, n_activated_experts=6, dspark_block_size=5,
                               n_mtp_layers=3,
                               compress_ratios=(0, 0, 4, 8, 4, 8, 4, 0, 0, 0, 0))
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
        DM.swap_in_fp4_moes(dr, moe_inter_dim=128, seed=7)
        return st, dr

    def run(dr_mutate=None):
        st, dr = build()
        if dr_mutate:
            dr_mutate(dr)
        torch.manual_seed(0)
        ids = torch.randint(0, args.vocab_size, (1, 13))
        tok = st.logits_all(st.forward(st.embed(ids), ids, 0), full_logits=False).argmax(-1)
        dr.prefill(tok, st.tail_main_hidden())
        out = []
        for i in range(13, 18):
            h = st.forward(st.embed(tok.unsqueeze(1)), tok.unsqueeze(1), i)
            tok = st.logits_all(h, full_logits=False).argmax(-1)
            dr.advance_and_draft(tok.unsqueeze(1), st.tail_main_hidden(), start_pos=i)
            out.append(tuple(t.clone() for t in dr.last_spec))
        return out, [b.attn.kv_cache.clone() for b in dr.mtp], dr

    DM.V4_DSPARK_MOE = False
    ref_out, ref_kv, _ = run()

    DM.V4_DSPARK_MOE = True
    banked = []

    def arm(dr):
        banked.append(DM.install_drafter(dr))
    got_out, got_kv, dr2 = run(arm)
    assert banked == [3], f"install must claim all 3 drafter MoEs, took {banked}"
    for i, (a, b) in enumerate(zip(ref_out, got_out)):
        for x, y, what in zip(a, b, ("output_ids", "logits", "confidence")):
            assert torch.equal(x, y), f"drafter {what} diverged at round {i}"
    for i, (rk, gk) in enumerate(zip(ref_kv, got_kv)):
        assert torch.equal(rk, gk), f"mtp {i} kv_cache diverged"
    cov = coverage(dr2)
    assert all(steps == len(ref_out) and not dec for steps, dec in cov.values()), \
        f"the pair path must serve EVERY round of EVERY block: {cov}"
    print(f"[v4 dspark moe] pair path bit-exact over {len(ref_out)} drafted rounds "
          f"(T={args.dspark_block_size} x k={args.n_activated_experts} pairs, "
          f"{args.n_mtp_layers} fp4 mtp blocks), coverage {cov}", flush=True)


if __name__ == "__main__":
    _selftest()
