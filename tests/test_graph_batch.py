"""CPU tests for the BATCHED-decode CUDA-graph lever (BatchGraphRunner / _BGraphState / _block_b):
the static-buffer math that carries per-stream starts into a captured run_block_decode_b graph must
equal the eager attn_decode_b computation EXACTLY (cp / RoPE gather / per-stream additive mask —
a wrong mask silently breaks stream isolation), and _block_b must mirror _block's routing: runner
cached per (B,s), M25_GRAPH_MAX bounding, over-budget/inactive/hatch -> eager fallback with the
HOST starts list turned into the device tensor run_block_decode_b expects. Everything GPU-real
(capture/replay bit-equality vs eager, aux freshness) is validated on-ring like solo's graph was.

Run: python3 -m pytest tests/test_graph_batch.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
fr = pytest.importorskip("fake_ring")               # bootstraps env (fake M25_DIR) + imports m25_pipe on CPU

MP = fr.MP
S = fr.S


# ---- 1. _BGraphState.set == the eager attn_decode_b start-math ------------------------------------

def eager_ref(starts, s, alen, cos, sin):
    """The exact cp/RoPE/mask computation from attn_decode_b's eager path (manual-matmul branch)."""
    B = len(starts)
    st = torch.as_tensor(starts, dtype=torch.long).view(B, 1)
    cp = st + torch.arange(s).view(1, s)
    cu = cos[cp].unsqueeze(1); su = sin[cp].unsqueeze(1)
    cols = torch.arange(alen).view(1, 1, alen)
    amask = torch.where(cols <= cp[:, :, None], 0.0, float("-inf")).to(torch.bfloat16)[:, None]
    return cp, cu, su, amask


@pytest.mark.parametrize("starts", [[0, 0, 0, 0], [17, 5, 900, 233], [2039, 1, 64, 2039]])
def test_bgraphstate_set_matches_eager_math(starts):
    B, s, alen, rd = len(starts), 9, 2048, 64
    g = torch.Generator().manual_seed(0)
    cos = (torch.randn(4096, rd, generator=g)).to(torch.bfloat16)
    sin = (torch.randn(4096, rd, generator=g)).to(torch.bfloat16)
    st = S._BGraphState(B, s, alen, rd, "cpu")
    st.set(starts, cos, sin)
    cp, cu, su, amask = eager_ref(starts, s, alen, cos, sin)
    assert torch.equal(st.cp, cp), "static cp != eager abs positions (KV scatter would corrupt rows)"
    assert torch.equal(st.cos, cu) and torch.equal(st.sin, su), "static RoPE rows != eager gather"
    assert torch.equal(st.mask, amask), "static mask != eager per-stream causal mask (isolation breaker)"
    # a second set() must fully overwrite the first (in-place refresh across replays)
    starts2 = [x + 3 for x in starts]
    st.set(starts2, cos, sin)
    cp2, cu2, su2, amask2 = eager_ref(starts2, s, alen, cos, sin)
    assert torch.equal(st.cp, cp2) and torch.equal(st.mask, amask2)
    assert torch.equal(st.cos, cu2) and torch.equal(st.sin, su2)


def test_bgraphstate_aux_buffers():
    st = S._BGraphState(4, 9, 2048, 64, "cpu", aux_ids=(30, 58))
    assert set(st.aux) == {30, 58}
    assert st.aux[30].shape == (4, 9, S.H) and st.aux[30].dtype == torch.bfloat16


# ---- 2. _block_b routing / bounding (FakeBGR mirrors FakeGR from test_graph_aux) -------------------

class FakeBGR:
    made = []

    def __init__(self, layers, vcfg, B, s):
        self.layers, self.vcfg, self.B, self.s = layers, vcfg, B, s
        self.graphs = {}
        self.eager = set()
        self.runs = []
        FakeBGR.made.append(self)

    def _bucket(self, total):                       # mirrors BatchGraphRunner._bucket exactly
        for b in S.DECODE_BUCKETS:
            if b >= total:
                return min(b, S.M25_KV_MAXLEN)
        return S.M25_KV_MAXLEN

    def run(self, starts, x):
        alen = self._bucket(max(starts) + self.s)
        if alen not in self.graphs:
            self.graphs[alen] = "g"
            S._GRAPH_COUNT += 1
        self.runs.append((list(starts), alen))
        return ("bgraph", self.B, self.s)


@pytest.fixture
def bgraph_env(monkeypatch):
    eager_calls = []

    def fake_run_block_decode_b(layers, starts, x, vcfg):
        assert torch.is_tensor(starts) and starts.dtype == torch.long, \
            "eager fallback must hand run_block_decode_b the device long tensor it expects"
        eager_calls.append((x.shape[0], x.shape[1], starts.tolist()))
        return ("eager", x.shape[0], x.shape[1])

    monkeypatch.setattr(S, "M25_CUDA_GRAPH_ACTIVE", True)
    monkeypatch.setattr(S, "M25_BATCH", 8)          # a batched ring (the guard: no runner without [B,...] KV rows)
    monkeypatch.setattr(S, "M25_GRAPH_MAX", 2)
    monkeypatch.setattr(S, "_GRAPH_COUNT", 0)
    monkeypatch.setattr(S, "_GRAPH_SKIPPED", 0)
    monkeypatch.setattr(S, "BatchGraphRunner", FakeBGR)
    monkeypatch.setattr(S, "run_block_decode_b", fake_run_block_decode_b)
    monkeypatch.setattr(MP, "M25_BATCH_GRAPH", True)
    monkeypatch.setattr(MP, "_GRAPH_CAP_LOGGED", set())
    monkeypatch.setattr(MP, "dev", "cpu")
    FakeBGR.made = []
    return eager_calls


def _x(B, s):
    return torch.zeros(B, s, 4)


def test_block_b_routes_and_caches_per_shape(bgraph_env):
    grs = {}
    out = MP._block_b(grs, [], [100, 60, 0, 3], _x(4, 9), None)     # capture #1 (4,9,2048)
    assert out == ("bgraph", 4, 9)
    assert list(grs) == [(4, 9)] and S._GRAPH_COUNT == 1
    MP._block_b(grs, [], [200, 10, 5, 80], _x(4, 9), None)          # same shape+bucket: replay
    assert S._GRAPH_COUNT == 1 and len(FakeBGR.made) == 1
    out = MP._block_b(grs, [], [50, 50], _x(2, 9), None)            # new B: its own runner, capture #2
    assert out == ("bgraph", 2, 9) and len(FakeBGR.made) == 2 and S._GRAPH_COUNT == 2
    assert bgraph_env == [] and S._GRAPH_SKIPPED == 0
    assert FakeBGR.made[0].runs[0][0] == [100, 60, 0, 3]            # host list reaches the runner as-is


def test_block_b_respects_graph_cap(bgraph_env, capsys):
    grs = {}
    MP._block_b(grs, [], [100] * 4, _x(4, 9), None)                 # (4,9,2048) capture #1
    MP._block_b(grs, [], [3000] * 4, _x(4, 9), None)                # (4,9,4096) capture #2 — budget spent
    assert S._GRAPH_COUNT == 2
    out = MP._block_b(grs, [], [100] * 8, _x(8, 9), None)           # NEW (8,9,2048) -> eager, counted, logged once
    assert out == ("eager", 8, 9)
    assert S._GRAPH_COUNT == 2 and S._GRAPH_SKIPPED == 1
    assert grs[(8, 9)].graphs == {}
    assert capsys.readouterr().out.count("[graph] cap:") == 1
    MP._block_b(grs, [], [120] * 8, _x(8, 9), None)                 # repeat skip: counted, NOT re-logged
    assert S._GRAPH_SKIPPED == 2 and "[graph] cap:" not in capsys.readouterr().out
    out = MP._block_b(grs, [], [150] * 4, _x(4, 9), None)           # captured shapes keep replaying over budget
    assert out == ("bgraph", 4, 9) and S._GRAPH_COUNT == 2


def test_block_b_inactive_or_hatch_is_eager(bgraph_env, monkeypatch):
    monkeypatch.setattr(S, "M25_CUDA_GRAPH_ACTIVE", False)
    assert MP._block_b({}, [], [10] * 4, _x(4, 9), None)[0] == "eager"
    monkeypatch.setattr(S, "M25_CUDA_GRAPH_ACTIVE", True)
    monkeypatch.setattr(MP, "M25_BATCH_GRAPH", False)               # escape hatch: batched eager, solo untouched
    assert MP._block_b({}, [], [10] * 4, _x(4, 9), None)[0] == "eager"
    assert FakeBGR.made == [] and S._GRAPH_COUNT == 0
    assert bgraph_env == [(4, 9, [10, 10, 10, 10]), (4, 9, [10, 10, 10, 10])]


def test_block_b_prefill_sized_stays_eager(bgraph_env):
    assert MP._block_b({}, [], [0] * 2, _x(2, 128), None)[0] == "eager"
    assert FakeBGR.made == [] and S._GRAPH_COUNT == 0


def test_block_b_solo_ring_never_makes_a_runner(bgraph_env, monkeypatch):
    """M25_BATCH=1 ring: no [B,...] KV rows exist, so the graph route must never construct a runner
    (the eager path's own failure mode is unchanged — no NEW crash class from graphs)."""
    monkeypatch.setattr(S, "M25_BATCH", 1)
    assert MP._block_b({}, [], [10], _x(1, 9), None)[0] == "eager"
    assert FakeBGR.made == [] and S._GRAPH_COUNT == 0


# ---- 3. BatchGraphRunner.run() host-side guards (no CUDA touched) ----------------------------------

def test_runner_bounds_check_and_manual_threshold(monkeypatch):
    monkeypatch.setattr(S, "M25_BATCH", 8)
    monkeypatch.setattr(S, "get_pe", lambda: (torch.zeros(64, 64, dtype=torch.bfloat16),
                                              torch.zeros(64, 64, dtype=torch.bfloat16)))
    eager = []
    monkeypatch.setattr(S, "run_block_decode_b",
                        lambda layers, starts, x, vcfg: eager.append(starts.tolist()) or "eager")
    r = S.BatchGraphRunner([], None, 8, 9, dv="cpu")
    with pytest.raises(RuntimeError, match="exceeds M25_KV_MAXLEN"):    # clean bound, never an OOB scatter
        r.run([S.M25_KV_MAXLEN] * 8, _x(8, 9))
    # a bucket past the manual-matmul threshold is NOT graphable -> eager fallback, no capture attempt
    big = next(b for b in S.DECODE_BUCKETS if not r._manual_ok(b))
    monkeypatch.setattr(S, "M25_KV_MAXLEN", max(S.M25_KV_MAXLEN, big))
    assert r.run([big - 9] * 8, _x(8, 9)) == "eager"
    assert r.graphs == {} and eager == [[big - 9] * 8]
