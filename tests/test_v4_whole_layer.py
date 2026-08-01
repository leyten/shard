"""The capture-safe decode transcription is BIT-EXACT to the reference `Block.forward`, and the
whole-layer graph is bit-exact to the same capture-safe path run eager.

Two independent bars, split so the hard one (the attention core) needs no GPU:

  test_capture_safe_block_matches_reference   CPU. `block_decode_cs(..., real_moe)` -- the device-side
      position, the fixed-width masked Indexer, the two-graph compress split -- reproduces
      `Block.forward` byte for byte, step for step, across a decode run long enough to wrap the window
      (window_size 16) MANY times and cross both compression boundaries (ratio 4 and 8). Hidden state,
      logits, AND every per-stage KV/compressor buffer must torch.equal the reference's. This is the
      real correctness proof; a graph only replays these same kernels.

  test_whole_layer_graph_is_bit_exact_to_eager   CUDA-only. The whole block (attn + islands + a
      graph-safe MoE stub) captured as one graph replays byte-identical to the same block run eager
      with the same stub -- proving the capture machinery (static buffers, compress/no-compress
      variants, KV-rewind) composes, on top of the CPU-proven attention math.

Run:  python3 -m pytest tests/test_v4_whole_layer.py -q         (first test runs anywhere)
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

SEED, PROMPT, STEPS = 7, 20, 40


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


def _cs_decode(st, R, M, h, ids, start_pos):
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
                maxw = A.indexer.kv_cache.size(1)
                arange = torch.arange(maxw, device=h.device)
            else:
                maxw = a.max_seq_len // ratio
                comp_topk = WL.build_comp_topk(M, ratio, start_pos, A.window_size, maxw)
                arange = None
        else:
            compress, comp_topk, arange = False, None, None
        x = WL.block_decode_cs(R, L, x, ids, pos, win_topk, comp_topk, arange, compress, WL.real_moe)
    return x


def test_capture_safe_block_matches_reference():
    """CPU, no graph: the cs transcription == Block.forward, every step, every buffer."""
    os.environ["V4_KERNELS"] = "cpu"
    args = REFCPU.cpu_args()
    M = REFCPU.load_ref()
    R = WL._Ref(M)

    eager = _stage(REFCPU.build_oracle(args, SEED), args, "cpu")
    cs = _stage(REFCPU.build_oracle(args, SEED), args, "cpu")

    torch.manual_seed(SEED)
    ids = torch.randint(0, args.vocab_size, (1, PROMPT))
    eager.forward(eager.embed(ids), ids, 0)
    cs.forward(cs.embed(ids), ids, 0)
    # prefill states must already agree (both ran the reference prefill)
    for a_, b_ in zip(_state_sig(eager), _state_sig(cs)):
        assert torch.equal(a_, b_), "prefill state diverged before decode even started"

    tok = torch.randint(0, args.vocab_size, (1, 1))
    for i in range(PROMPT, PROMPT + STEPS):
        h = eager.embed(tok)
        h_e = eager.forward(h, tok, i)
        lg_e = eager.logits_all(h_e, full_logits=False)

        h2 = cs.embed(tok)
        h_c = _cs_decode(cs, R, M, h2, tok, i)
        cs._pos = i + 1                                  # keep the cs stage's bookkeeping honest
        lg_c = cs.logits_all(h_c, full_logits=False)

        assert torch.equal(h_e, h_c), f"hidden diverged at decode step {i} (pos {i})"
        assert torch.equal(lg_e, lg_c), f"logits diverged at decode step {i}"
        for a_, b_ in zip(_state_sig(eager), _state_sig(cs)):
            assert torch.equal(a_, b_), f"a KV/compressor buffer diverged at decode step {i}"
        tok = lg_e.argmax(dim=-1, keepdim=True).view(1, 1)


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
