"""The capture-safe decode transcription, graded on m25's two-tier bar.

WHAT IS BIT-EXACT AND WHAT IS NOT. A graph replays the same kernels on the same bytes, so
`graphed == the eager twin` is torch.equal and stays one. Against the VENDORED reference there are
TWO reasons the last bits can differ, and neither is a rounding drift in this code:

  1. TOP-K TIE ORDER. `index_score` is bf16 behind a `relu_()` that floors negatives to a hard 0.0,
     so ties at the k-th rank are routine (23 of 120 decode steps). `Tensor.topk` does not define its
     tie order, it differs between CPU and CUDA, and it is NOT invariant to array length -- so a
     fixed-width or bucketed read picked a DIFFERENT compressed slot than the reference's narrow one
     (pos 43, ~2e-3 on the hidden state: a different attention support set, not rounding).
     `_select_topk_width_invariant` imposes a total order (value DESC, index ASC), which makes the
     selection unique and width-invariant. It does not reproduce torch's arbitrary order and cannot.
  2. LANE ORDER. The reference's topk returns its picks in DESCENDING SCORE order; a width-invariant
     selection returns them in ascending index order. Same support set, but sparse_attn then gathers
     the rows in a different lane order and its per-block reduction regroups -- measured ~3e-4 at
     pos 55 with the sets identical.

So the reference is not the long-run bar. These are:

  test_read_width_is_a_cost_knob_over_a_long_run  TIER 1, the headline: max width vs bucketed widths
      give torch.equal hidden, logits and every KV buffer -- 120 steps x 3 seeds. This is what the
      capture actually has to guarantee, and it would have caught the original bug with no reference.
  test_read_width_is_a_cost_knob_only             the same invariant on the shipped selection
      function directly, over score vectors built to tie.
  test_capture_safe_block_matches_reference       TIER 2, the clean regime: where the Indexer keeps
      every valid column, hidden/logits/KV torch.equal the reference, 3 seeds.
  test_fixed_width_selection_matches_reference_selection  TIER 2, the selection: same SET when the
      boundary is tie-free, same selected SCORE MULTISET always (never a lower-scoring column).
  test_bucket_ladder_covers_and_never_truncates   the bucket may never be shorter than the position
      needs, and the rung count stays bounded.
  test_graph_output_is_fresh_across_positions     CUDA. TIER 1 per position + the FRESHNESS GATE: the
      same input at a DIFFERENT position must MOVE the output (a position baked in at capture is the
      worst silent failure here; m25 shipped it once as stale EAGLE aux).
  test_whole_layer_stage_is_bit_exact_to_eager    CUDA. A V4_CUDA_GRAPH=whole Stage vs the eager
      Stage, over the unambiguous regime.

Seeds matter: the first cut of this file failed on seeds 7 and 11 at different positions and PASSED
on 23, which is exactly how the bug survived a green run.

Run:  python3 -m pytest tests/test_v4_whole_layer.py -q         (CPU tests run anywhere)
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase0"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
REFCPU = pytest.importorskip("v4_ref_cpu")
V4 = pytest.importorskip("v4_stage")
WL = pytest.importorskip("v4_whole_layer_graph")
KERNELS = pytest.importorskip("v4_kernels_cpu")

# The block-level parity tests below drive CPU stages, so they need the CPU kernel stand-ins. That
# choice is frozen at IMPORT (model.py resolves `from kernel import ...` once), so a process started
# with V4_KERNELS=tilelang cannot be talked into it from inside a test -- it is skipped instead of
# failing on a device mismatch. Run them with V4_KERNELS=cpu (the default on a box with no GPU).
requires_cpu_kernels = pytest.mark.skipif(
    KERNELS.backend() != "cpu",
    reason=f"block parity drives CPU stages; this process bound the {KERNELS.backend()} kernels "
           f"(re-run with V4_KERNELS=cpu)")

SEED, PROMPT, STEPS = 7, 20, 120
# Seeds 7 and 11 both failed the first cut of this file at different positions and seed 23 PASSED,
# which is exactly how the bug survived a green run. Multi-seed is not decoration here.
SEEDS = (7, 11, 23)
# The Indexer only really SELECTS once end_pos//ratio exceeds index_topk; below that it keeps every
# valid column and the reference's tie-break cannot bite. At cpu_args() (ratio 4, index_topk 8) that
# is (pos+1)//4 <= 8, i.e. pos <= 34 -- 15 decode steps from PROMPT=20, which still wrap the 16-slot
# window and cross the ratio-4 (pos 23, 27, 31) and ratio-8 (pos 23, 31) compression boundaries.
STEPS_EXACT = 15


def _stage(oracle, args, device, graph=False):
    V4.V4_CUDA_GRAPH = graph
    st = V4.Stage(0, args.n_layers, args, head=True, tail=True, device=device)
    for li in range(args.n_layers):
        st.layers[li].load_state_dict(oracle.layers[li].state_dict(), strict=True)
    st.embed_tokens.load_state_dict(oracle.embed.state_dict(), strict=True)
    st.norm.load_state_dict(oracle.norm.state_dict(), strict=True)
    st.lm_head.load_state_dict(oracle.head.state_dict(), strict=True)
    with torch.no_grad():
        for n in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            getattr(st, n).data.copy_(getattr(oracle, n).data)
    return st


def _state_sig(st):
    sig = []
    for L in st.layers:
        sig.append(L.attn.kv_cache.clone())
        if L.attn.compress_ratio and L.attn.indexer is not None:
            sig.append(L.attn.indexer.kv_cache.clone())
    for c, _ in st._compressors():
        sig.append(c.kv_state.clone())
        sig.append(c.score_state.clone())
    return sig


def _cs_decode(st, R, M, h, ids, start_pos, bgs=None):
    """One capture-safe decode step through every layer of `st`, WITHOUT a graph.

    Delegates to `WholeBlockGraphs._eager` -- the same object the graph is graded against -- so the
    bucket decisions, the compress split and the selection are literally the shipped ones rather than
    a second copy that could drift from them."""
    if bgs is None:
        bgs = [WL.WholeBlockGraphs(L, st, moe_mode="eager") for L in st.layers]
    torch.set_grad_enabled(False)
    x = h
    for bg in bgs:
        x = bg._eager(x, ids, start_pos)
    return x


def _drive_both(steps, seed=SEED, **argov):
    """Prefill + `steps` decode steps through an eager and a capture-safe stage, in lockstep.

    Both stages are fed the SAME token every step (the eager side's argmax), so a divergence cannot
    compound into two different sequences and every comparison is at the same position on the same
    input. Yields (pos, h_eager, h_cs, logits_eager, logits_cs, eager, cs) per step."""
    os.environ["V4_KERNELS"] = "cpu"
    args = REFCPU.cpu_args(**argov)
    global SEED_IN_USE
    SEED_IN_USE = seed
    M = REFCPU.load_ref()
    R = WL._Ref(M)
    eager = _stage(REFCPU.build_oracle(args, seed), args, "cpu")
    cs = _stage(REFCPU.build_oracle(args, seed), args, "cpu")
    torch.manual_seed(seed)
    ids = torch.randint(0, args.vocab_size, (1, PROMPT))
    eager.forward(eager.embed(ids), ids, 0)
    cs.forward(cs.embed(ids), ids, 0)
    for a_, b_ in zip(_state_sig(eager), _state_sig(cs)):
        assert torch.equal(a_, b_), "prefill state diverged before decode even started"
    tok = torch.randint(0, args.vocab_size, (1, 1))
    bgs = [WL.WholeBlockGraphs(L, cs, moe_mode="eager") for L in cs.layers]
    for i in range(PROMPT, PROMPT + steps):
        h_e = eager.forward(eager.embed(tok), tok, i)
        lg_e = eager.logits_all(h_e, full_logits=False)
        h_c = _cs_decode(cs, R, M, cs.embed(tok), tok, i, bgs)
        cs._pos = i + 1                                  # keep the cs stage's bookkeeping honest
        lg_c = cs.logits_all(h_c, full_logits=False)
        yield i, h_e, h_c, lg_e, lg_c, eager, cs
        tok = lg_e.argmax(dim=-1, keepdim=True).view(1, 1)


@requires_cpu_kernels
def test_capture_safe_block_matches_reference():
    """CPU, no graph: where the Indexer's selection is unambiguous, the cs transcription IS the
    reference -- hidden, logits and every KV/compressor buffer, step for step.

    This is the headline correctness proof for the device-side position, the fixed max-width masked
    read, the -1 padding and the compress/no-compress split: over pos 20..34 the Indexer keeps every
    valid column (end_pos//ratio <= index_topk), so no tie-break can differ and the whole pipeline
    must agree BYTE FOR BYTE. The run still wraps the 16-slot window ring and crosses both
    compression boundaries."""
    for seed in SEEDS:
      for i, h_e, h_c, lg_e, lg_c, eager, cs in _drive_both(STEPS_EXACT, seed=seed):
        assert torch.equal(h_e, h_c), f"seed {seed}: hidden diverged at decode step {i} (pos {i})"
        assert torch.equal(lg_e, lg_c), f"logits diverged at decode step {i}"
        for a_, b_ in zip(_state_sig(eager), _state_sig(cs)):
            assert torch.equal(a_, b_), f"a KV/compressor buffer diverged at decode step {i}"


@requires_cpu_kernels
def test_read_width_is_a_cost_knob_over_a_long_run():
    """TIER 1, the long-run bar: driving the SAME decode at max width and at bucketed widths gives
    torch.equal hidden, logits and KV -- for 120 steps, on three seeds.

    This, not agreement with the reference, is what the whole-layer capture has to guarantee: the read
    width is a COST knob. It is also the bar that would have caught the original bug (a wide read
    picking a different compressed slot than a narrow one) without needing the reference at all, and
    it is the one that stays valid under the Tier-2 caveats below -- neither the reference's tie order
    NOR its lane order is part of the contract.

    Why the reference cannot be the long-run bar: `topk` returns its picks in DESCENDING SCORE order
    while a width-invariant selection returns them in ascending index order. Same support set, but
    sparse_attn then sums the gathered rows in a different lane order, and that regroups its per-block
    reduction -- a last-bit difference that has nothing to do with which slots were chosen. Measured
    here at pos 55 (~3e-4 on the hidden state) with the sets identical."""
    for seed in SEEDS:
        wide, bucketed = None, None
        for force_max in (True, False):
            orig = WL.bucket_width
            if force_max:
                WL.bucket_width = lambda need, maxw, floor=0: maxw
            try:
                run = [(h_c.clone(), lg_c.clone(), [t.clone() for t in _state_sig(cs)])
                       for _, _, h_c, _, lg_c, _, cs in _drive_both(STEPS, seed=seed)]
            finally:
                WL.bucket_width = orig
            if force_max:
                wide = run
            else:
                bucketed = run
        assert len(wide) == len(bucketed) == STEPS
        for i, ((h_w, lg_w, st_w), (h_b, lg_b, st_b)) in enumerate(zip(wide, bucketed)):
            pos = PROMPT + i
            assert torch.equal(h_w, h_b), f"seed {seed}: read width changed the hidden state at pos {pos}"
            assert torch.equal(lg_w, lg_b), f"seed {seed}: read width changed the logits at pos {pos}"
            for a_, b_ in zip(st_w, st_b):
                assert torch.equal(a_, b_), f"seed {seed}: read width changed a KV buffer at pos {pos}"



def test_fixed_width_selection_matches_reference_selection():
    """The #2 rewrite itself: fixed-width + mask + (-1) selects a maximal-scoring set, and exactly the
    reference's set whenever the boundary is tie-free.

    Drives the SHIPPED `select_compress_topk` against the reference's own expression
    (`score[:end].topk(min(k, end))[1] + offset`) over scores deliberately built to collide the way
    the real ones do (relu'd, coarsely quantised), at widths that straddle end_ratio < k, == k and > k.
    Two assertions, and the difference between them is exactly the documented tie-break ambiguity:
      * the selected SCORE MULTISET always matches -- neither side ever picks a lower-scoring column;
      * the selected SET matches whenever the k-th and (k+1)-th best valid scores differ."""
    torch.manual_seed(0)
    maxw, k, offset = 64, 8, 16
    arange = torch.arange(maxw)
    for end_ratio in (1, 3, 7, 8, 9, 11, 20, 64):
        for trial in range(25):
            # coarse quantisation + relu is what makes the real index_score tie so often
            s = (torch.randn(1, 1, maxw) * 3).round() / 4
            s = s.relu() if trial % 2 else s
            mine = WL.select_compress_topk(s.clone(), torch.tensor(end_ratio), k, offset, arange)
            ref = s[..., :end_ratio].topk(min(k, end_ratio), dim=-1)[1] + offset

            m_valid = sorted(v for v in mine[0, 0].tolist() if v != -1)
            r_valid = sorted(ref[0, 0].tolist())
            assert len(m_valid) == min(k, end_ratio), \
                f"end_ratio={end_ratio}: picked {len(m_valid)} valid, want {min(k, end_ratio)}"
            assert (mine == -1).sum().item() == k - min(k, end_ratio), \
                f"end_ratio={end_ratio}: wrong amount of -1 padding"
            assert all(offset <= v < offset + end_ratio for v in m_valid), \
                f"end_ratio={end_ratio}: a pick escaped the valid window after the offset"

            sm = sorted(s[0, 0, [v - offset for v in m_valid]].tolist())
            sr = sorted(s[0, 0, [v - offset for v in r_valid]].tolist())
            assert sm == sr, f"end_ratio={end_ratio}: selected a non-maximal set {sm} vs {sr}"

            vals = s[0, 0, :end_ratio].sort(descending=True)[0]
            tie_free = end_ratio <= k or vals[k - 1].item() != vals[k].item()
            if tie_free:
                assert m_valid == r_valid, \
                    f"end_ratio={end_ratio}: tie-free boundary but sets differ {m_valid} vs {r_valid}"


def test_read_width_is_a_cost_knob_only():
    """TIER 1, the property the first cut of this file did NOT have: the READ WIDTH cannot change the
    answer -- max width, a bucket, and the exact width all select the SAME slots.

    This is the direct regression test for the pos-43 failure. `torch.topk` fails it (its tie order is
    undefined and length-dependent, and `relu_()` manufactures ties by flooring negatives to a hard
    0.0), so a wide-masked read picked a different compressed slot than the reference's narrow one.
    `_select_topk_width_invariant`'s total order (value DESC, index ASC) makes the top-k unique, hence
    width-invariant, which is also what turns bucketing into a pure cost lever."""
    torch.manual_seed(0)
    k, offset = 8, 16
    for seed in SEEDS:
        torch.manual_seed(seed)
        for end_ratio in (5, 8, 9, 11, 20, 33):
            for trial in range(12):
                base = (torch.randn(1, 1, end_ratio) * 3).round() / 4      # collides on purpose
                if trial % 2:
                    base = base.relu()                                     # the real tie factory
                picks = []
                for width in (max(end_ratio, k), 64, 256):                 # exact-ish, bucket, max
                    sc = torch.zeros(1, 1, width)
                    sc[..., :end_ratio] = base
                    picks.append(WL.select_compress_topk(sc, torch.tensor(end_ratio), k, offset,
                                                        torch.arange(width))[0, 0].tolist())
                assert picks[0] == picks[1] == picks[2], \
                    f"seed {seed} end_ratio {end_ratio}: read width changed the selection {picks}"


def test_bucket_ladder_covers_and_never_truncates():
    """A bucket must never be SHORTER than what the position needs -- that would silently drop slots.

    The read is narrowed to the bucket, so `bucket >= end_ratio` is the invariant the whole scheme
    rests on (below it, valid compressed slots would fall outside the einsum and could never be
    selected). Also checks the ladder is monotone and clamps to the cache, and that a decode only
    crosses a rung a bounded number of times -- the reason for bucketing over one graph per length."""
    for maxw in (16, 64, 512, 4096, 16384):
        seen = set()
        for need in range(0, maxw + 1):
            b = WL.bucket_width(need, maxw)
            assert b >= min(need, maxw), f"bucket {b} < need {need} (maxw {maxw}) — would truncate the read"
            assert b <= maxw, f"bucket {b} exceeds the cache {maxw}"
            seen.add(b)
        assert len(seen) <= len(WL.INDEXER_BUCKETS), \
            f"maxw={maxw} produced {len(seen)} distinct buckets, more than the ladder has rungs"
    assert list(WL.INDEXER_BUCKETS) == sorted(WL.INDEXER_BUCKETS), "the ladder must be ascending"


def test_selection_is_deterministic_across_widths():
    """The fix's own property: the selection does NOT depend on the array width (what broke topk).

    The same valid scores, read at a fixed width of 64 and of 256, must select the same columns --
    which is what makes a graphed fixed-width read reproducible where `topk` was not."""
    torch.manual_seed(1)
    end_ratio, k, offset = 11, 8, 16
    base = (torch.randn(1, 1, end_ratio) * 3).round() / 4          # collides on purpose
    picks = []
    for maxw in (64, 256):
        s = torch.zeros(1, 1, maxw)
        s[..., :end_ratio] = base
        picks.append(WL.select_compress_topk(s, torch.tensor(end_ratio), k, offset,
                                             torch.arange(maxw))[0, 0].tolist())
    assert picks[0] == picks[1], f"width changed the selection: {picks[0]} vs {picks[1]}"


def _ensure_hadamard():
    """Register v4_kernels_cpu's pure-torch hadamard if the fast_hadamard_transform extension is absent.

    rotate_activation imports it lazily (model.py), so a GPU box without the CUDA extension can still run
    the Indexer's rotation while keeping the real sm120 sparse-attention / act_quant tilelang kernels."""
    import importlib.util
    import types
    import v4_kernels_cpu
    if "fast_hadamard_transform" in sys.modules:
        return                                  # already present (real, or this shim from an earlier test)
    try:
        if importlib.util.find_spec("fast_hadamard_transform") is not None:
            return
    except ValueError:                          # a live module with no __spec__
        return
    mod = types.ModuleType("fast_hadamard_transform")
    mod.hadamard_transform = v4_kernels_cpu.hadamard_transform
    mod._v4_cpu_backend = True
    sys.modules["fast_hadamard_transform"] = mod


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs are a GPU-only capture")
def test_graph_output_is_fresh_across_positions():
    """FRESHNESS GATE (research/graph_aux_check.py:93): the SAME input at a DIFFERENT position must
    MOVE the output, and must equal what the eager twin produces at that position.

    This is the cheap insurance against the worst silent failure class here -- a position baked in at
    capture. A graph that froze `start_pos`, or that replayed a stale static buffer, would return the
    capture-position answer forever: identical outputs across positions, plausible numbers, valid
    receipts. m25 shipped exactly that bug once (stale EAGLE aux). Positions are chosen to straddle a
    bucket rung so a rung crossing is exercised too."""
    _ensure_hadamard()
    torch.set_default_device("cuda")
    try:
        args = REFCPU.cpu_args()
        torch.manual_seed(SEED)
        ids = torch.randint(0, args.vocab_size, (1, PROMPT))
        st = _stage(REFCPU.build_oracle(args, SEED), args, "cuda", graph=False)
        st.forward(st.embed(ids), ids, 0)
        L = next(L for L in st.layers if getattr(L.attn, "indexer", None) is not None)
        bg = WL.WholeBlockGraphs(L, st, moe_mode="eager")
        torch.set_grad_enabled(False)
        x = st.embed(torch.randint(0, args.vocab_size, (1, 1)))
        tok = torch.randint(0, args.vocab_size, (1, 1))
        outs, buckets = [], []
        # 20/40 sit in the first rung (end_ratio <= 16); 71 has end_ratio 18 and lands in the next one,
        # so the last position is served by a graph captured at a DIFFERENT width.
        for pos in (PROMPT, PROMPT + 20, 71):
            snap = [b.clone() for b in WL._layer_state(L)]
            got = bg.run(x, tok, pos).clone()               # graphed
            for b, s in zip(WL._layer_state(L), snap):
                b.copy_(s)
            want = bg._eager(x, tok, pos).clone()           # the eager twin, same position
            for b, s in zip(WL._layer_state(L), snap):
                b.copy_(s)
            assert torch.equal(got, want), f"TIER-1: graphed != eager twin at pos {pos}"
            outs.append(got)
            buckets.append(bg._plan(pos)[0])
        for j in range(1, len(outs)):
            assert not torch.equal(outs[0], outs[j]), \
                f"output did not MOVE from pos {PROMPT} to the {j}th position — position looks baked in"
        assert len(set(buckets)) > 1, f"all three positions shared bucket {buckets} — rung never crossed"
    finally:
        torch.set_default_device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs are a GPU-only capture")
def test_whole_layer_stage_is_bit_exact_to_eager():
    """CUDA: a Stage with V4_CUDA_GRAPH=whole (attn core graphed, routed MoE eager) is bit-exact to the
    eager Stage -- hidden, logits, and every KV/compressor buffer, over a decode run that wraps the
    window and crosses the ratio-4 and ratio-8 compression boundaries. The real-serving graph path."""
    _ensure_hadamard()
    torch.set_default_device("cuda")
    try:
        args = REFCPU.cpu_args()
        torch.manual_seed(SEED)
        ids = torch.randint(0, args.vocab_size, (1, PROMPT))
        first = torch.randint(0, args.vocab_size, (1, 1))

        eager = _stage(REFCPU.build_oracle(args, SEED), args, "cuda", graph=False)
        graphed = _stage(REFCPU.build_oracle(args, SEED), args, "cuda", graph="whole")
        assert graphed._block_graphs is not None and graphed._graph_mode == "whole"

        eager.forward(eager.embed(ids), ids, 0)
        graphed.forward(graphed.embed(ids), ids, 0)
        tok = first
        # STEPS_EXACT, not STEPS: the eager Stage runs the VENDORED block, so past the point where the
        # Indexer really selects this becomes a Tier-2 comparison and the reference's tie-break is in
        # play. Inside the unambiguous regime it is a true end-to-end bit-exactness bar on the whole
        # stage; the strict Tier-1 gate (graphed == the capture-safe twin) is the freshness test above.
        for i in range(PROMPT, PROMPT + STEPS_EXACT):
            h_e = eager.forward(eager.embed(tok), tok, i)
            h_g = graphed.forward(graphed.embed(tok), tok, i)
            lg_e = eager.logits_all(h_e, full_logits=False)
            lg_g = graphed.logits_all(h_g, full_logits=False)
            assert torch.equal(h_e, h_g), f"hidden diverged at decode step {i}"
            assert torch.equal(lg_e, lg_g), f"logits diverged at decode step {i}"
            for a_, b_ in zip(_state_sig(eager), _state_sig(graphed)):
                assert torch.equal(a_, b_), f"a KV/compressor buffer diverged at decode step {i}"
            tok = lg_e.argmax(dim=-1, keepdim=True).view(1, 1)
    finally:
        V4.V4_CUDA_GRAPH = os.environ.get("V4_CUDA_GRAPH", "0")
        torch.set_default_device("cpu")
