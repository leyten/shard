"""CPU tests for the CUDA-graph EAGLE-aux compatibility lever (perf/graph-aux): the M25_GRAPH_MAX
capture-set bounding + eager fallback in m25_pipe._block, the removed M25_CUDA_GRAPH+M25_EAGLE
SystemExit guard, the per-job runtime graph toggle (reset op -> S.set_graph -> _block routing), and
_merge_aux consuming S._AUX unchanged (the contract the graph path's aliased publish relies on).
Everything GPU-real — capture/replay bit-equality, aux freshness across start_pos, OOM fallback on a
real capture failure, launch-overhead timing — lives in research/graph_aux_check.py (on-box).

Run: python3 -m pytest tests/test_graph_aux.py -q
"""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
fr = pytest.importorskip("fake_ring")               # bootstraps env (fake M25_DIR) + imports m25_pipe on CPU

MP = fr.MP
S = fr.S

PHASE0 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase0")


class FakeGR:
    """The GraphRunner surface _block consults (graphs / _bucket) + the capture-accounting contract
    (_capture bumps S._GRAPH_COUNT once per new bucket). No CUDA anywhere — this fakes the RUNNER so
    the tests exercise _block's REAL bounding/routing logic."""
    made = []

    def __init__(self, layers, vcfg, s):
        self.layers, self.vcfg, self.s = layers, vcfg, s
        self.graphs = {}
        self.eager = set()
        self.runs = []
        FakeGR.made.append(self)

    def _bucket(self, total):                       # mirrors GraphRunner._bucket exactly
        for b in S.DECODE_BUCKETS:
            if b >= total:
                return min(b, S.M25_KV_MAXLEN)
        return S.M25_KV_MAXLEN

    def run(self, start, x):
        alen = self._bucket(start + self.s)
        if alen not in self.graphs:                 # "capture" — counted like the real _capture
            self.graphs[alen] = "g"
            S._GRAPH_COUNT += 1
        self.runs.append((start, alen))
        return ("graph", self.s, start)


@pytest.fixture
def graph_env(monkeypatch):
    """Graph route ACTIVE, budget of 2, fake runner, recording eager fallback. Counters zeroed."""
    eager_calls = []

    def fake_run_block(layers, start, x, vcfg):
        eager_calls.append((x.shape[1], start))
        return ("eager", x.shape[1], start)

    monkeypatch.setattr(S, "M25_CUDA_GRAPH_ACTIVE", True)
    monkeypatch.setattr(S, "M25_GRAPH_MAX", 2)
    monkeypatch.setattr(S, "_GRAPH_COUNT", 0)
    monkeypatch.setattr(S, "_GRAPH_SKIPPED", 0)
    monkeypatch.setattr(S, "GraphRunner", FakeGR)
    monkeypatch.setattr(S, "run_block", fake_run_block)
    monkeypatch.setattr(MP, "_GRAPH_CAP_LOGGED", set())
    FakeGR.made = []
    return eager_calls


def _x(s):
    return torch.zeros(1, s, 4)                     # _block only reads x.shape[1]


# ---- 1. M25_GRAPH_MAX bounding / fallback in _block ------------------------------------------------

def test_block_routes_small_blocks_and_caches_runner(graph_env):
    grs = {}
    out = MP._block(grs, [], 100, _x(9), None)      # start 100 + 9 -> bucket 2048, capture #1
    assert out == ("graph", 9, 100)
    assert list(grs) == [9] and S._GRAPH_COUNT == 1
    MP._block(grs, [], 200, _x(9), None)            # same (s, bucket): replay, no new capture
    assert S._GRAPH_COUNT == 1 and len(FakeGR.made) == 1
    assert graph_env == [] and S._GRAPH_SKIPPED == 0


def test_prefill_stays_eager(graph_env):
    out = MP._block({}, [], 0, _x(128), None)       # s > 64 = prefill -> never graphed
    assert out == ("eager", 128, 0)
    assert graph_env == [(128, 0)] and S._GRAPH_COUNT == 0


def test_graph_cap_bounds_capture_set(graph_env, capsys):
    grs = {}
    MP._block(grs, [], 100, _x(9), None)            # (9, 2048)  capture #1
    MP._block(grs, [], 3000, _x(9), None)           # (9, 4096)  capture #2 — budget (2) now spent
    assert S._GRAPH_COUNT == 2
    # NEW shape (12, 2048) would exceed the cap -> eager fallback, counted, never captured
    out = MP._block(grs, [], 100, _x(12), None)
    assert out == ("eager", 12, 100)
    assert S._GRAPH_COUNT == 2 and S._GRAPH_SKIPPED == 1
    assert grs[12].graphs == {}                     # the runner exists but captured nothing
    # NEW bucket for an EXISTING runner also respects the cap
    out = MP._block(grs, [], 10000, _x(9), None)    # (9, 16384) -> eager
    assert out == ("eager", 9, 10000) and S._GRAPH_SKIPPED == 2
    # already-captured shapes KEEP replaying over budget
    out = MP._block(grs, [], 150, _x(9), None)      # (9, 2048) replay
    assert out == ("graph", 9, 150) and S._GRAPH_COUNT == 2
    assert graph_env == [(12, 100), (9, 10000)]
    assert capsys.readouterr().out.count("[graph] cap:") == 2   # F3: logged once per skipped shape
    MP._block(grs, [], 120, _x(12), None)           # repeat (12, 2048) skip: counted, NOT re-logged
    assert S._GRAPH_SKIPPED == 3
    assert "[graph] cap:" not in capsys.readouterr().out


def test_graph_route_inactive_is_all_eager(graph_env, monkeypatch):
    monkeypatch.setattr(S, "M25_CUDA_GRAPH_ACTIVE", False)
    out = MP._block({}, [], 100, _x(9), None)
    assert out == ("eager", 9, 100)
    assert FakeGR.made == [] and S._GRAPH_COUNT == 0


# ---- 2. per-job runtime toggle (reset dict -> set_graph -> _block routing) -------------------------

def test_set_graph_flips_block_routing(graph_env, monkeypatch):
    monkeypatch.setattr(S, "M25_STATIC_KV", True)
    monkeypatch.setattr(S, "M25_SDPA", True)
    monkeypatch.setattr(S, "M25_CUDA_GRAPH_ACTIVE", False)
    grs = {}
    assert MP._block(grs, [], 100, _x(9), None)[0] == "eager"
    MP._reset_flags({"op": "reset", "graph": True})           # job N: graph arm
    assert S.M25_CUDA_GRAPH_ACTIVE is True
    assert MP._block(grs, [], 100, _x(9), None)[0] == "graph"
    MP._reset_flags({"op": "reset", "graph": False})          # job N+1: eager arm
    assert S.M25_CUDA_GRAPH_ACTIVE is False
    assert MP._block(grs, [], 100, _x(9), None)[0] == "eager"
    MP._reset_flags({"op": "reset"})                          # absent field = keep current setting
    assert S.M25_CUDA_GRAPH_ACTIVE is False


def test_set_graph_refused_without_static_kv(graph_env, monkeypatch, capsys):
    monkeypatch.setattr(S, "M25_STATIC_KV", False)
    monkeypatch.setattr(S, "M25_CUDA_GRAPH_ACTIVE", False)
    assert MP._reset_flags({"op": "reset", "graph": True}) is False   # the APPLIED value the tail acks
    assert S.M25_CUDA_GRAPH_ACTIVE is False                   # refused, ignored — never crash
    assert "GRAPH REFUSED" in capsys.readouterr().out         # ...but LOUD + grep-stable in the stage log
    assert MP._block({}, [], 100, _x(9), None)[0] == "eager"  # and never silently claims graphs
    assert MP._reset_flags({"op": "reset", "graph": False}) is False  # graph=false is always honored
    assert S.M25_CUDA_GRAPH_ACTIVE is False


def test_set_graph_refused_without_sdpa(graph_env, monkeypatch, capsys):
    """F9: M25_STATIC_KV on but M25_SDPA off — the graphed block still needs the SDPA-era routing
    (GraphRunner asserts both), so the toggle must refuse through the same loud path."""
    monkeypatch.setattr(S, "M25_STATIC_KV", True)
    monkeypatch.setattr(S, "M25_SDPA", False)
    monkeypatch.setattr(S, "M25_CUDA_GRAPH_ACTIVE", False)
    assert MP._reset_flags({"op": "reset", "graph": True}) is False
    assert S.M25_CUDA_GRAPH_ACTIVE is False
    assert "GRAPH REFUSED" in capsys.readouterr().out


def test_reset_op_stamps_graph_field(monkeypatch):
    assert "graph" not in MP._reset_op("s", "j")              # default: field absent (old stages untouched)
    monkeypatch.setattr(MP, "M25_GRAPH_JOB", True)
    assert MP._reset_op("s", "j")["graph"] is True
    monkeypatch.setattr(MP, "M25_GRAPH_JOB", False)
    assert MP._reset_op("s", "j")["graph"] is False
    o = MP._reset_op("sw", "jb")                              # rest of the frame unchanged
    assert o["op"] == "reset" and o["swarm_id"] == "sw" and o["job_id"] == "jb" and o["temp"] == 0.0


def test_check_reset_ack():
    """F2: a graph-stamped job must FAIL LOUDLY when the tail didn't apply the toggle — a silently
    eager 'graph' arm banks a lying A/B. Plain jobs accept the bare 'ok' unchanged."""
    assert MP._check_reset_ack({"op": "reset"}, "ok") is None           # no graph field: any ack passes
    ack = {"ok": 1, "graph": True, "graph_captured": 3, "graph_skipped": 1}
    assert MP._check_reset_ack({"op": "reset", "graph": True}, ack) is ack
    assert MP._check_reset_ack({"op": "reset", "graph": False},
                               {"ok": 1, "graph": False})["graph"] is False
    with pytest.raises(RuntimeError, match="GRAPH REFUSED"):            # old stage: bare "ok" to a graph job
        MP._check_reset_ack({"op": "reset", "graph": True}, "ok")
    with pytest.raises(RuntimeError, match="graph A/B poisoned"):       # refused toggle: applied != asked
        MP._check_reset_ack({"op": "reset", "graph": True}, {"ok": 1, "graph": False})


def test_coordinator_sends_graph_field_end_to_end(monkeypatch):
    """A real coordinate_pipe job over the fake ring: with M25_GRAPH_JOB set the ring must SEE the
    graph field on the job-opening reset, the dict ack must PASS _check_reset_ack, and the job still
    completes lossless. Without receipts there is no sweep, so graph_arm stays None."""
    from ngram_draft import NgramDrafter
    monkeypatch.setattr(MP, "M25_GRAPH_JOB", True)
    T = fr.repetitive_T(360)
    res, ring = fr.run_coordinator(T, 60, NgramDrafter(ng=3, min_match=1, margin=64),
                                   K=8, depth=4, max_new=80, eagle_ring=False)
    assert res["ok"] and res["output_ids"] == T[60:60 + len(res["output_ids"])]
    resets = [e for e in ring.log if e["op"] == "reset"]
    assert resets and all(e["graph"] is True for e in resets)
    assert res["graph_arm"] is None


def test_job_record_shows_graph_arm(monkeypatch):
    """F3: with receipts on, a graph-stamped job's receipt sweep returns the tail's graph counters and
    the coordinator surfaces them in the result — the 'how graphed was this arm ACTUALLY' record."""
    from ngram_draft import NgramDrafter
    monkeypatch.setattr(MP, "M25_GRAPH_JOB", True)
    monkeypatch.setattr(MP, "RECEIPTS", True)
    T = fr.repetitive_T(360)
    res, ring = fr.run_coordinator(T, 60, NgramDrafter(ng=3, min_match=1, margin=64),
                                   K=8, depth=4, max_new=80, eagle_ring=False)
    assert res["ok"] and res["output_ids"] == T[60:60 + len(res["output_ids"])]
    assert res["graph_arm"] == {"graph": True, "graph_captured": 0, "graph_skipped": 0}
    assert res["receipts"] == [] and res["receipts_ok"] is False        # fake ring signs nothing; fails closed


# ---- 3. guard removal: import with BOTH flags must not SystemExit ----------------------------------

def test_import_with_graph_and_eagle_does_not_exit():
    env = dict(os.environ, M25_CUDA_GRAPH="1", M25_EAGLE="1", SHARD_TRANSPORT="libp2p",
               M25_GRAPH_JOB="no",                            # F6: "no" must parse False (explicit truthy set)
               M25_DIR=os.environ["M25_DIR"])                 # fake model dir from the fake_ring bootstrap
    shard = os.path.join(os.path.dirname(PHASE0), "shard")
    code = ("import sys; sys.path.insert(0, {p!r}); sys.path.insert(0, {sh!r}); "
            "import m25_stage as S; "
            "assert S.M25_CUDA_GRAPH and S.M25_EAGLE and S.M25_STATIC_KV; "
            "assert S.M25_CUDA_GRAPH_ACTIVE is True and S.M25_GRAPH_MAX == 16; "
            "import m25_pipe as MP; assert MP.M25_GRAPH_JOB is False; "
            "print('IMPORT_OK')").format(p=PHASE0, sh=shard)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"import died (the old SystemExit guard?)\nstdout={r.stdout}\nstderr={r.stderr}"
    assert "IMPORT_OK" in r.stdout


# ---- 4. _merge_aux consumes _AUX unchanged ---------------------------------------------------------

def test_merge_aux_passes_aux_through_unchanged(monkeypatch):
    """The graph path publishes _AUX entries that ALIAS its static device buffers; the safety of that
    (no clone) rests on _merge_aux consuming the VALUES immediately and untouched. Pin that contract:
    values equal, upstream entries passed through, upstream dict not mutated."""
    monkeypatch.setattr(S, "M25_EAGLE", True)
    monkeypatch.setattr(MP, "M25_FP8_AUX", False)
    a30, a58 = torch.randn(9, 16), torch.randn(9, 16)
    monkeypatch.setattr(S, "_AUX", {30: a30, 58: a58})
    up = {"1": torch.randn(9, 16)}
    out = MP._merge_aux(up)
    assert set(out) == {"1", "30", "58"}
    assert torch.equal(out["30"], a30) and torch.equal(out["58"], a58)
    assert out["1"] is up["1"]                                # upstream packed-or-not entries untouched
    assert set(up) == {"1"}                                   # input dict not mutated
    monkeypatch.setattr(S, "M25_EAGLE", False)                # no-op unless M25_EAGLE
    assert MP._merge_aux(up) == up
