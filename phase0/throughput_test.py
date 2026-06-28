"""offline proof for throughput-aware, self-learning ring selection.

No GPUs, no network — drives shard/throughput.py, shard/perf_store.py, and the new
scheduler_svc tok_s plan / telemetry path directly. Proves the decisions that matter:

  1. est_tok_s math (round time -> tok/s, accept-cap, zero guards)
  2. a SLOW fat-VRAM GPU loses to a FAST leaner one — the thing pure-latency/VRAM can't see
  3. PerfStore EWMA learns ms/layer + accept_rate from runs, with GPU-class cold priors
  4. plan_tok_s end-to-end picks the higher-tok/s ring and is exact-coverage
  5. /telemetry -> /plan loop: a recorded slow run shifts the next plan's choice

run: python3 phase0/test_throughput.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shard.throughput import est_tok_s, round_ms, best_ring
from shard.perf_store import PerfStore, classify_gpu, GPU_MS_PER_LAYER_PRIOR
from shard.scheduler import Scheduler, JoinedNode

PASS = 0
FAIL = 0


def ok(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {label}")
    else:
        FAIL += 1
        print(f"FAIL  {label}")


# ---------- 1. est_tok_s math ----------
def test_est_tok_s():
    print("est_tok_s:")
    # 100ms round, accept 0.5, K=4 -> 2 tokens/round -> 20 tok/s
    ok(abs(est_tok_s(100.0, 0.5, 4) - 20.0) < 1e-9, "0.5*4 toks / 100ms = 20 tok/s")
    # accept caps at K+1: accept 1.0, K=4 -> min(4, 5) = 4 toks
    ok(abs(est_tok_s(100.0, 1.0, 4) - 40.0) < 1e-9, "accept-rate capped at K (4 toks)")
    # all-reject still commits the bonus token (never zero throughput)
    ok(est_tok_s(100.0, 0.0, 4) > 0, "all-reject still emits the 1 bonus token")
    # zero/negative round time -> 0, no div-by-zero
    ok(est_tok_s(0.0, 0.5, 4) == 0.0, "zero round_ms guarded")


# ---------- 2 & helpers: a fixed allocate_fn over a VRAM pool ----------
def make_allocate_fn(model, total_layers, vram, gb_per_layer, kv=0.0):
    def allocate_fn(subset_ids):
        sub = Scheduler(model, total_layers)
        for nid in subset_ids:
            sub.register(JoinedNode(node_id=nid, vram_gb=vram[nid], rtt_ms={}))
        try:
            alloc = sub.allocate(gb_per_layer, kv, 2.0, 1.0)
        except Exception:
            return None
        return {nid: (lr.end - lr.start) for nid, lr in alloc.items()}
    return allocate_fn


def test_slow_fat_loses_to_fast_lean():
    print("slow-fat vs fast-lean selection:")
    # Two ways to serve a 40-layer model that needs ~40GB total:
    #   FAT: one 48GB card that is SLOW (3090-ish, 0.70 ms/layer) holds all 40 -> 28ms compute
    #   LEAN: two 24GB FAST cards (4090, 0.42) hold 20 each -> 8.4ms each but +1 WAN hop
    # low edge latency makes the 2-fast-card ring win on tok/s despite the extra hop.
    model, total = "M", 40
    vram = {"FAT": 48.0, "F1": 24.0, "F2": 24.0}
    gb_per_layer = 1.0
    node_ids = ["FAT", "F1", "F2"]
    # cheap mesh (same-region fast links)
    L = [[0.0, 10.0, 10.0], [10.0, 0.0, 8.0], [10.0, 8.0, 0.0]]
    c0 = [0.0, 0.0, 0.0]
    ci = [0.0, 0.0, 0.0]
    ms = {"FAT": 0.70, "F1": 0.42, "F2": 0.42}
    af = make_allocate_fn(model, total, vram, gb_per_layer)
    best = best_ring(node_ids, vram, L, c0, ci, allocate_fn=af, ms_per_layer=ms,
                     draft_ms_by_node={n: 0.0 for n in node_ids}, accept_rate=0.62, K=4,
                     total_layers=total)
    ok(best is not None, "a feasible ring exists")
    ok(set(best["ring_order"]) == {"F1", "F2"}, "picked the two FAST cards, not the slow fat one")
    ok(best["coordinator"] in ("F1", "F2"), "coordinator is one of the fast cards")

    # now make the fat card FAST (an H100-class 0.18) and the leans slow (3090 0.70):
    # the single-fat ring should win — fewer hops, and now the compute is cheap.
    ms2 = {"FAT": 0.18, "F1": 0.70, "F2": 0.70}
    best2 = best_ring(node_ids, vram, L, c0, ci, allocate_fn=af, ms_per_layer=ms2,
                      draft_ms_by_node={n: 0.0 for n in node_ids}, accept_rate=0.62, K=4,
                      total_layers=total)
    ok(best2["ring_order"] == ["FAT"], "fast fat card wins as a single-stage ring")
    ok(best2["tok_s"] > best["tok_s"], "the all-fast-fat plan beats the split plan on tok/s")


def test_round_ms_compose():
    print("round_ms composition:")
    # 2 stages, 10 layers each; ms/layer 0.5 and 1.0; one 20ms hop; entry/return 0.
    L = [[0.0, 20.0], [20.0, 0.0]]
    order = [0, 1]
    layers = {"A": 10, "B": 10}
    ms = {"A": 0.5, "B": 1.0}
    rt = round_ms(order, L, [0.0, 0.0], [0.0, 0.0], layers, ms, ["A", "B"], draft_ms=2.0)
    # WAN 20 + compute (10*0.5 + 10*1.0 = 15) + draft 2 = 37
    ok(abs(rt - 37.0) < 1e-9, "20 wan + 15 compute + 2 draft = 37ms")


# ---------- 3. PerfStore learning ----------
def test_perf_store():
    print("PerfStore EWMA + priors:")
    ps = PerfStore()
    ok(classify_gpu("NVIDIA GeForce RTX 4090") == "4090", "classify 4090")
    ok(classify_gpu("Tesla H100 80GB") == "h100", "classify h100")
    ok(classify_gpu("some weird card") == "unknown", "unknown class")
    ps.seed_gpu("N", "RTX 4090")
    ok(abs(ps.ms_for("N") - GPU_MS_PER_LAYER_PRIOR["4090"]) < 1e-9, "cold node uses 4090 prior")
    # first real sample is authoritative over the (weak) class prior: 6.0ms / 20 layers = 0.30
    ps.observe_run({"model": "M", "stages": [{"node_id": "N", "n_layers": 20, "compute_ms": 6.0}]})
    ok(abs(ps.ms_for("N") - 0.30) < 1e-9, "first real sample replaces the cold prior (0.30)")
    # a SECOND, different sample (0.50 ms/layer) now EWMAs against 0.30 — damped, not replaced
    ps.observe_run({"model": "M", "stages": [{"node_id": "N", "n_layers": 20, "compute_ms": 10.0}]})
    blended = ps.ms_for("N")
    ok(0.30 < blended < 0.50, "second sample EWMA-blends (between 0.30 and 0.50, not a jump)")
    # accept-rate learning: first sample 64/20/K4 = 0.8 is authoritative
    ps.observe_run({"model": "M", "tokens": 64, "rounds": 20, "K": 4})
    ok(abs(ps.accept_for("M") - 0.8) < 1e-9, "first accept_rate sample = 0.8")
    # a worse run (40/20/K4 = 0.5) damps it down toward 0.5 but not all the way
    ps.observe_run({"model": "M", "tokens": 40, "rounds": 20, "K": 4})
    ok(0.5 < ps.accept_for("M") < 0.8, "second accept_rate sample EWMA-damps between 0.5 and 0.8")
    # rtt learning
    ps.observe_run({"model": "M", "edges": [{"from": "A", "to": "B", "rtt_ms": 33.0}]})
    ok(ps.rtt_for("A", "B") != ps.rtt_for("B", "A"), "rtt is directional")
    ok(ps.rtt_for("A", "A") == 0.0, "self-rtt is zero")


def test_perf_store_persist():
    print("PerfStore persistence:")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "perf.json")
        ps = PerfStore(p)
        ps.seed_gpu("N", "H100")
        ps.observe_run({"model": "M", "stages": [{"node_id": "N", "n_layers": 10, "compute_ms": 2.0}]})
        ok(os.path.exists(p), "store written to disk")
        ps2 = PerfStore(p)            # reload
        ok(abs(ps2.ms_for("N") - 0.2) < 1e-9, "learned ms/layer survived reload")
        ok(ps2.gpu_class.get("N") == "h100", "gpu class survived reload")


# ---------- 4 & 5. scheduler_svc tok_s plan + telemetry loop ----------
def test_plan_tok_s_and_telemetry():
    print("scheduler_svc tok_s plan + telemetry loop:")
    import importlib
    svc = importlib.import_module("phase0.scheduler_svc")
    # fresh in-memory perf store for a deterministic test
    svc.PERF = PerfStore()
    req = {
        "model": "M", "total_layers": 40, "gb_per_layer": 1.0,
        "objective": "tok_s", "K": 4,
        "nodes": [
            {"node_id": "FAT", "vram_gb": 48, "rtt_ms": {"F1": 10, "F2": 10}},
            {"node_id": "F1", "vram_gb": 24, "rtt_ms": {"FAT": 10, "F2": 8}},
            {"node_id": "F2", "vram_gb": 24, "rtt_ms": {"FAT": 10, "F1": 8}},
        ],
    }
    # cold: all use 'unknown' prior (0.80) equally -> fewer-hops single FAT wins on cold priors
    p_cold = svc.plan_tok_s(req)
    ok(p_cold["ok"] and sum(s["n_layers"] for s in p_cold["stages"]) == 40, "cold plan exact-covers 40 layers")
    ok("est_tok_s" in p_cold, "plan reports a predicted tok/s")

    # teach the store: FAT is SLOW (0.80 ms/layer measured over a full 40-layer hold),
    # the F-cards are FAST (0.42). Feed it as completed-run telemetry.
    svc.PERF.observe_run({"model": "M", "stages": [{"node_id": "FAT", "n_layers": 40, "compute_ms": 32.0}]})
    for _ in range(3):
        svc.PERF.observe_run({"model": "M", "stages": [
            {"node_id": "F1", "n_layers": 20, "compute_ms": 20 * 0.42},
            {"node_id": "F2", "n_layers": 20, "compute_ms": 20 * 0.42},
        ]})
    p_warm = svc.plan_tok_s(req)
    ok(set(p_warm["ring_order"]) == {"F1", "F2"},
       "after learning FAT is slow, plan switches to the two fast cards")
    ok(p_warm["est_tok_s"] >= p_cold["est_tok_s"] - 1e-6,
       "learned plan predicts >= the cold plan's tok/s")
    print(f"       cold ring={p_cold['ring_order']} ~{p_cold['est_tok_s']} tok/s  ->  "
          f"warm ring={p_warm['ring_order']} ~{p_warm['est_tok_s']} tok/s")


def test_backward_compat():
    print("backward-compat (no objective -> latency plan unchanged):")
    import importlib
    svc = importlib.import_module("phase0.scheduler_svc")
    req = {
        "model": "M", "total_layers": 40, "gb_per_layer": 1.0,
        "nodes": [
            {"node_id": "A", "vram_gb": 48, "rtt_ms": {"B": 30}},
            {"node_id": "B", "vram_gb": 24, "rtt_ms": {"A": 30}},
        ],
    }
    p = svc.plan(req)
    ok(p["ok"] and "objective" not in p, "default plan() unchanged (no objective key)")
    ok(sum(s["n_layers"] for s in p["stages"]) == 40, "default plan still exact-covers")


if __name__ == "__main__":
    test_est_tok_s()
    test_slow_fat_loses_to_fast_lean()
    test_round_ms_compose()
    test_perf_store()
    test_perf_store_persist()
    test_plan_tok_s_and_telemetry()
    test_backward_compat()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
