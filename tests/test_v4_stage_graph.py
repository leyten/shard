"""DeepSeek-V4-Flash decode CUDA graphs: the graphed step is BIT-IDENTICAL to the eager one.

V4_CUDA_GRAPH captures the position- and data-independent islands of a decode step (the two hc_pre,
the two hc_post, the attn/ffn RMSNorms) and leaves `attn` and `ffn` eager between them. Because a
graph replays the reference's OWN kernels on operands fed through static buffers, a correct capture is
the same arithmetic on the same bytes -- so the bar is torch.equal, not allclose (unlike a
reassociating fast path). This proves it: the same stage, run once eager and once graphed from the
same weights and the same scripted prefill+decode, must agree on the hidden state, the logits, AND
every per-stage KV/compressor buffer, step for step.

CUDA-only: a graph cannot be captured on CPU, so this is skipped without a device (the box parity ran
this same comparison on the real 158 GiB checkpoint, layers [40:43), 24 decode steps -- bit-exact).

Run (on a GPU box):  python3 -m pytest tests/test_v4_stage_graph.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
REFCPU = pytest.importorskip("v4_ref_cpu")
V4 = pytest.importorskip("v4_stage")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason="CUDA graphs are a GPU-only capture (torch.cuda.CUDAGraph)")

SEED, PROMPT, STEPS = 7, 13, 6


def _oracle_and_args():
    args = REFCPU.cpu_args()
    return REFCPU.build_oracle(args, SEED), args


def _stage(oracle, args, graph):
    """A [0:n_layers) head+tail stage on CUDA holding the oracle's own weights, graphed or not."""
    V4.V4_CUDA_GRAPH = graph
    st = V4.Stage(0, args.n_layers, args, head=True, tail=True, device="cuda")
    for li in range(args.n_layers):
        st.layers[li].load_state_dict(oracle.layers[li].state_dict(), strict=True)
    st.embed_tokens.load_state_dict(oracle.embed.state_dict(), strict=True)
    st.norm.load_state_dict(oracle.norm.state_dict(), strict=True)
    st.lm_head.load_state_dict(oracle.head.state_dict(), strict=True)
    with torch.no_grad():
        for n in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            getattr(st, n).data.copy_(getattr(oracle, n).data)
    return st


def _state(st):
    sig = []
    for L in st.layers:
        sig.append(L.attn.kv_cache.clone())
        if L.attn.compress_ratio and L.attn.indexer is not None:
            sig.append(L.attn.indexer.kv_cache.clone())
    for c, _ in st._compressors():
        sig.append(c.kv_state.clone())
        sig.append(c.score_state.clone())
    return sig


def _drive(st, ids, first):
    """Prefill `ids`, then STEPS greedy decode steps. -> (per-step hidden, per-step logits)."""
    hs, logits = [], []
    st.forward(st.embed(ids) if st.head else None, ids, 0)
    tok = first
    for i in range(PROMPT, PROMPT + STEPS):
        h = st.embed(tok.unsqueeze(1))
        h = st.forward(h, tok.unsqueeze(1), i)
        lg = st.logits_all(h, full_logits=False)
        hs.append(h.detach().clone())
        logits.append(lg.detach().clone())
        tok = lg.argmax(dim=-1)
    return hs, logits


def _restore_flag():
    V4.V4_CUDA_GRAPH = os.environ.get("V4_CUDA_GRAPH", "0") not in ("", "0")


@requires_cuda
def test_graphed_decode_is_bit_exact_to_eager():
    """The headline: V4_CUDA_GRAPH changes the launch profile, never a single bit of output."""
    torch.set_default_device("cuda")                    # the serving env (generate.py/v4_pipe do this)
    try:
        oracle, args = _oracle_and_args()
        torch.manual_seed(SEED)                         # ids on the default device (cuda here)
        ids = torch.randint(0, args.vocab_size, (1, PROMPT))
        first = torch.randint(0, args.vocab_size, (1,))

        eager = _stage(oracle, args, False)
        e_h, e_lg = _drive(eager, ids, first)
        e_state = _state(eager)

        graphed = _stage(REFCPU.build_oracle(args, SEED), args, True)
        assert graphed._block_graphs is not None, "graph should be armed on a CUDA stage"
        g_h, g_lg = _drive(graphed, ids, first)
        g_state = _state(graphed)
    finally:
        _restore_flag()
        torch.set_default_device("cpu")

    for i, (a, b) in enumerate(zip(e_h, g_h)):
        assert torch.equal(a, b), f"hidden state diverged at decode step {i}"
    for i, (a, b) in enumerate(zip(e_lg, g_lg)):
        assert torch.equal(a, b), f"logits diverged at decode step {i}"
    for a, b in zip(e_state, g_state):
        assert torch.equal(a, b), "a per-stage KV/compressor buffer diverged"
    assert len(e_state) == len(g_state)


def test_graph_refuses_on_cpu(monkeypatch):
    """A CPU stage never graphs -- it says why and stays eager (no silent half-capture)."""
    oracle, args = _oracle_and_args()
    monkeypatch.setattr(V4, "V4_CUDA_GRAPH", True)
    st = V4.Stage(0, 2, args, head=True, tail=False, device="cpu")
    assert st._block_graphs is None
    assert "device is cpu" in st._graph_refusal()
