"""What the capture-safe decode transcription is graded on, in two tiers, and why it is two.

The first cut of this file held ONE bar -- "byte-identical to the vendored reference" -- and it was
the wrong bar in a way that hid a real bug behind a green tick for exactly one seed. The reference's
Indexer ends its decode with `index_score.topk(min(index_topk, end_pos//ratio))`, and `Tensor.topk`
does not define its tie order: it is whatever the selection algorithm leaves behind, it differs
between the CPU and CUDA backends, and it is NOT invariant to the length of the array. `index_score`
is bf16 behind a `relu_()` that floors negatives to a hard 0.0, so exact ties at the k-th rank are
routine -- and a fixed-width masked topk therefore selected a DIFFERENT compressed KV slot than the
reference at decode pos 43, diverging the hidden state by ~2e-3. Demanding token-identity against an
arbitrary tie order is demanding that a capture reproduce an accident.

So:

  TIER 1, HARD (torch.equal), CPU -- the properties a CUDA graph actually needs:
    test_capture_safe_decode_is_width_invariant   the capture-safe path run with the shipped fixed
        MAX-WIDTH indexer read, with a BUCKETED read, and with the reference's own exact width must
        produce identical hidden/logits/KV across a full decode run. This is what licenses a fixed
        (or bucketed) capture at all, and it is the bar the old code silently failed.
    test_capture_safe_decode_is_position_fresh   the FRESHNESS gate: the same input at a different
        position must MOVE the output. Cheapest possible insurance against the worst silent failure
        a graph has -- a position baked in at capture time and replayed forever.

  TIER 2, NAMED AND BOUNDED, CPU:
    test_capture_safe_block_matches_reference_modulo_topk_ties   against the vendored `Block.forward`
        the path is bit-identical at every position where the reference's own top-k is well defined,
        and may differ only from the first step at which the index scores TIE at the k-th rank. The
        test locates that step, asserts bit-equality strictly before it, and reports the tie census
        so the deviation is a measured number rather than a hope.

  CUDA-only:
    test_whole_layer_stage_is_bit_exact_to_eager   a Stage with the whole-layer graph on replays
        byte-identically to the same Stage eager -- the capture machinery (static buffers,
        compress/no-compress variants, KV-rewind) composing on top of the CPU-proven attention math.

Run:  python3 -m pytest tests/test_v4_whole_layer.py -q         (the CPU tests run anywhere)
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

SEED, PROMPT = 7, 20
# 120 decode steps, not 40: the reduction-width bug this file now guards against first bit at step
# 122 of a 200-step run, and a 40-step bar walked straight past it (V4_WL_STEPS overrides).
STEPS = int(os.environ.get("V4_WL_STEPS", "120"))
# Toy stand-in for m25_stage.DECODE_BUCKETS -- every bucket must hold index_topk (8) slots.
READ_BUCKETS = (8, 16, 32, 64)


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


def _read_width(mode, end_ratio, index_topk, maxw):
    """The Indexer read width this step: the shipped fixed max, a bucket, or the reference's own.

    All three must give the same answer -- that is the whole point of the width-invariant selection,
    and it is what makes a bucketed capture (one graph per bucket, M2.5's DECODE_BUCKETS shape) a
    pure cost decision. A read still has to hold index_topk slots to select index_topk of them."""
    if mode == "max":
        return maxw
    if mode == "exact":
        return min(max(end_ratio, index_topk), maxw)
    for b in READ_BUCKETS:
        if b >= end_ratio:
            return min(max(b, index_topk), maxw)
    return maxw


def _cs_decode(st, R, M, h, ids, start_pos, read="max"):
    """One capture-safe decode step through every layer of `st` (the eager `forward`'s cs twin)."""
    a = st.args
    pos = torch.tensor([start_pos], dtype=torch.long, device=h.device)
    x = h
    torch.set_grad_enabled(False)
    for L in st.layers:
        A = L.attn
        ratio = A.compress_ratio
        win_topk = WL.build_win_topk(M, A.window_size, start_pos)
        if ratio:
            compress = (start_pos + 1) % ratio == 0
            if A.indexer is not None:
                comp_topk = None
                w = _read_width(read, (start_pos + 1) // ratio, A.indexer.index_topk,
                                A.indexer.kv_cache.size(1))
                arange = torch.arange(w, device=h.device)
            else:
                maxw = a.max_seq_len // ratio
                comp_topk = WL.build_comp_topk(M, ratio, start_pos, A.window_size, maxw)
                arange = None
        else:
            compress, comp_topk, arange = False, None, None
        x = WL.block_decode_cs(R, L, x, ids, pos, win_topk, comp_topk, arange, compress, WL.real_moe)
    return x


def _fresh_stage(args, M):
    """A stage prefilled with the standard prompt, plus the first decode token."""
    st = _stage(REFCPU.build_oracle(args, SEED), args, "cpu")
    torch.manual_seed(SEED)
    ids = torch.randint(0, args.vocab_size, (1, PROMPT))
    st.forward(st.embed(ids), ids, 0)
    return st, torch.randint(0, args.vocab_size, (1, 1))


def _tie_at_cut(score, valid, k):
    """Does the reference's own top-k have to break a tie here? (more tied candidates than slots)

    If the k-th and (k+1)-th valid scores are equal, `Tensor.topk` picks between them by an order it
    does not define -- so at this step the reference's selection is an artifact, not the model's
    intent, and the capture-safe path is entitled to a different (but deterministic) answer."""
    s = score.masked_fill(~valid, float("-inf"))
    kth = s.topk(k, dim=-1).values.narrow(-1, k - 1, 1)
    gt = int((s > kth).sum())
    eq = int(((s == kth) & valid).sum())
    return eq > 0 and gt + eq > k


def test_capture_safe_decode_is_width_invariant():
    """TIER 1. The Indexer read width is a COST knob and nothing else -- fixed-max, bucketed, and the
    reference's own exact width must give torch.equal hidden/logits/KV for a whole decode run.

    This is the property a fixed-shape capture actually rests on, and the bar the first cut of this
    file failed silently: a masked wide topk selected a different compressed slot than a narrow one
    the moment the scores tied at the k-th rank. Each read mode drives its OWN token stream, so a
    single differing slot compounds instead of being papered over by a shared teacher."""
    os.environ["V4_KERNELS"] = "cpu"
    args = REFCPU.cpu_args()
    M = REFCPU.load_ref()
    R = WL._Ref(M)

    modes = ("max", "bucket", "exact")
    runs = {}
    for mode in modes:
        st, tok = _fresh_stage(args, M)
        runs[mode] = [st, tok]
    for i in range(PROMPT, PROMPT + STEPS):
        out = {}
        for mode in modes:
            st, tok = runs[mode]
            h = _cs_decode(st, R, M, st.embed(tok), tok, i, read=mode)
            st._pos = i + 1
            lg = st.logits_all(h, full_logits=False)
            out[mode] = (h, lg, _state_sig(st))
            runs[mode][1] = lg.argmax(dim=-1, keepdim=True).view(1, 1)
        for mode in modes[1:]:
            assert torch.equal(out["max"][0], out[mode][0]), \
                f"read width changed the hidden state at decode step {i}: max vs {mode}"
            assert torch.equal(out["max"][1], out[mode][1]), \
                f"read width changed the logits at decode step {i}: max vs {mode}"
            for a_, b_ in zip(out["max"][2], out[mode][2]):
                assert torch.equal(a_, b_), \
                    f"read width changed a KV/compressor buffer at decode step {i}: max vs {mode}"


def test_capture_safe_decode_is_position_fresh():
    """TIER 1. The FRESHNESS gate: the same input at a different position must MOVE the output.

    The worst failure a decode graph has is silent -- a position baked in at capture time, replayed
    forever, every token computed against the wrong window slot and the wrong rope row while the
    numbers stay finite and plausible. Nothing else in this file would catch it: a stale graph is
    perfectly self-consistent. So drive one step at pos p, rewind the layer state, drive the SAME
    (h, ids) at p+1, and demand the answers differ -- and demand the same of the window wrap, where
    `pos % win` returns to a slot it has already used."""
    os.environ["V4_KERNELS"] = "cpu"
    args = REFCPU.cpu_args()
    M = REFCPU.load_ref()
    R = WL._Ref(M)
    st, tok = _fresh_stage(args, M)
    h = st.embed(tok)

    bufs = [b for L in st.layers for b in WL._layer_state(L)]
    snap = [b.clone() for b in bufs]

    def at(p):
        for b, s in zip(bufs, snap):
            b.copy_(s)
        return _cs_decode(st, R, M, h.clone(), tok, p, read="max").clone()

    win = st.layers[0].attn.window_size
    base = at(PROMPT)
    assert not torch.equal(base, at(PROMPT + 1)), \
        "the same input at pos+1 produced the SAME output -- the position is not being read"
    assert not torch.equal(base, at(PROMPT + win)), \
        f"the same input a whole window ({win}) later produced the SAME output -- pos % win is stale"
    assert not torch.equal(base, at(PROMPT + 4 * win)), \
        "the same input 4 windows later produced the SAME output -- the position is baked in"


def test_capture_safe_block_matches_reference_modulo_topk_ties():
    """TIER 2, NAMED AND BOUNDED: cs == Block.forward byte for byte until the reference's own top-k
    stops being well defined, and the census of where that is.

    `Tensor.topk` does not define its tie order, so at a step where the index scores tie at the k-th
    rank the reference's pick is an artifact of its selection algorithm (and differs between the CPU
    and CUDA backends). The capture-safe path resolves those ties by lowest index, deterministically
    and at any read width. Everything else -- the device-side position, the index_copy_ stores, the
    compressor overlap shift, the compress/no-compress split, the window wrap, the -1-padded compress
    list -- must be bit-identical, and this asserts that strictly, up to the first tied step. The tie
    census prints so the deviation stays a measured number."""
    os.environ["V4_KERNELS"] = "cpu"
    args = REFCPU.cpu_args()
    M = REFCPU.load_ref()
    R = WL._Ref(M)

    ties = []
    _sel = WL._select_topk_width_invariant

    def sel(score, valid, k, arange_w):
        if _tie_at_cut(score, valid, k):
            ties.append(True)
        return _sel(score, valid, k, arange_w)

    eager = _stage(REFCPU.build_oracle(args, SEED), args, "cpu")
    cs = _stage(REFCPU.build_oracle(args, SEED), args, "cpu")
    torch.manual_seed(SEED)
    ids = torch.randint(0, args.vocab_size, (1, PROMPT))
    eager.forward(eager.embed(ids), ids, 0)
    cs.forward(cs.embed(ids), ids, 0)
    for a_, b_ in zip(_state_sig(eager), _state_sig(cs)):
        assert torch.equal(a_, b_), "prefill state diverged before decode even started"

    WL._select_topk_width_invariant = sel
    try:
        tok = torch.randint(0, args.vocab_size, (1, 1))
        first_tie, n_tied_steps, matched = None, 0, 0
        for i in range(PROMPT, PROMPT + STEPS):
            h_e = eager.forward(eager.embed(tok), tok, i)
            lg_e = eager.logits_all(h_e, full_logits=False)
            ties.clear()
            h_c = _cs_decode(cs, R, M, cs.embed(tok), tok, i)
            cs._pos = i + 1
            lg_c = cs.logits_all(h_c, full_logits=False)
            if ties:
                n_tied_steps += 1
                first_tie = i if first_tie is None else first_tie
            if first_tie is None:
                assert torch.equal(h_e, h_c), f"hidden diverged at decode step {i} with NO top-k tie"
                assert torch.equal(lg_e, lg_c), f"logits diverged at decode step {i} with NO top-k tie"
                for a_, b_ in zip(_state_sig(eager), _state_sig(cs)):
                    assert torch.equal(a_, b_), \
                        f"a KV/compressor buffer diverged at decode step {i} with NO top-k tie"
                matched += 1
            tok = lg_e.argmax(dim=-1, keepdim=True).view(1, 1)
    finally:
        WL._select_topk_width_invariant = _sel
    print(f"\n  tier 2: bit-identical to the vendored reference for {matched} decode steps; "
          f"first k-th-rank top-k tie at pos {first_tie}; {n_tied_steps}/{STEPS} steps tied")


def _ensure_hadamard():
    """Register v4_kernels_cpu's pure-torch hadamard if the fast_hadamard_transform extension is absent.

    rotate_activation imports it lazily (model.py), so a GPU box without the CUDA extension can still run
    the Indexer's rotation while keeping the real sm120 sparse-attention / act_quant tilelang kernels."""
    import importlib.util
    import types
    import v4_kernels_cpu
    if importlib.util.find_spec("fast_hadamard_transform") is not None:
        return
    mod = types.ModuleType("fast_hadamard_transform")
    mod.hadamard_transform = v4_kernels_cpu.hadamard_transform
    mod._v4_cpu_backend = True
    sys.modules["fast_hadamard_transform"] = mod


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
        for i in range(PROMPT, PROMPT + STEPS):
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
