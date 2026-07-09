"""plan_ring (shard/plan.py) — the deployable-ring seam c0mpute calls to place a sharded swarm.

Covers the calibration + the JSON CLI contract:
  * a feasible pool -> a head-first ring whose per-stage blocks tile [0, n_layers) with no gap/overlap,
    on DISTINCT subnets (never co-located),
  * an infeasible pool -> None (not a bad ring),
  * the `python3 -m shard.plan` stdin/stdout round-trip a TypeScript orchestrator drives.

Run: python3 -m pytest tests/test_plan.py -q
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shard.plan import plan_ring  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pool(k, free_gb=30.0, rtt_ms=20.0, cpu_factor=1.0):
    """k nodes on distinct subnets, symmetric RTT, each fat enough for ~13 layers."""
    nodes = [{"id": f"node{i}", "free_vram_mb": free_gb * 1024, "subnet": f"10.{i}.0.0/24",
              "cpu_factor": cpu_factor} for i in range(k)]
    rtt = [[0.0 if i == j else rtt_ms for j in range(k)] for i in range(k)]
    return nodes, rtt


def test_feasible_pool_tiles_the_model():
    nodes, rtt = _pool(6)                      # 6 fat boxes, 62 layers -> ~5 stages of ~13
    plan = plan_ring(nodes, rtt)
    assert plan is not None
    _assert_tiles_simple(plan, 62)
    assert plan["k"] == len({s["id"] for s in plan["stages"]})
    assert plan["k"] >= 5                       # 62 layers / 13-per-box ceiling -> at least 5 stages
    # distinct subnets across the chosen ring (never co-located)
    chosen = {s["id"] for s in plan["stages"]}
    subs = [n["subnet"] for n in nodes if n["id"] in chosen]
    assert len(subs) == len(set(subs))


def _assert_tiles_simple(plan, n_layers):
    stages = sorted(plan["stages"], key=lambda s: s["index"])
    assert stages[0]["head"] and stages[0]["id"] == plan["head"] == plan["order"][0]
    assert stages[-1]["tail"]
    lo = 0
    for s in stages:
        assert s["lo"] == lo, f"gap/overlap at {s['index']}: {s['lo']} != {lo}"
        assert s["hi"] > s["lo"]
        assert s["layers"] == s["hi"] - s["lo"]
        lo = s["hi"]
    assert lo == n_layers


def test_infeasible_pool_returns_none():
    # two thin boxes can't hold a 62-layer model (each caps at 13) -> the pool genuinely can't serve.
    nodes, rtt = _pool(2)
    assert plan_ring(nodes, rtt) is None
    # and an empty pool
    assert plan_ring([], []) is None


def test_head_is_most_central():
    # give node2 the lowest total RTT to the rest -> it should be picked as head/coordinator.
    nodes, rtt = _pool(6)
    for i in range(6):
        for j in range(6):
            if i != j:
                rtt[i][j] = 10.0 if (i == 2 or j == 2) else 80.0
    plan = plan_ring(nodes, rtt)
    assert plan is not None
    assert plan["head"] == "node2"


def test_cli_roundtrip():
    """`python3 -m shard.plan` reads a JSON request on stdin and prints the plan on stdout."""
    nodes, rtt = _pool(6)
    req = json.dumps({"nodes": nodes, "rtt": rtt})
    r = subprocess.run([sys.executable, "-m", "shard.plan"], input=req, capture_output=True,
                       text=True, cwd=REPO, timeout=60)
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    assert plan is not None
    _assert_tiles_simple(plan, 62)


def test_cli_infeasible_prints_null():
    nodes, rtt = _pool(2)
    r = subprocess.run([sys.executable, "-m", "shard.plan"],
                       input=json.dumps({"nodes": nodes, "rtt": rtt}),
                       capture_output=True, text=True, cwd=REPO, timeout=60)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) is None        # a valid answer: this pool can't hold the model


def test_cli_bad_request_reports_error():
    r = subprocess.run([sys.executable, "-m", "shard.plan"], input="{not json",
                       capture_output=True, text=True, cwd=REPO, timeout=60)
    assert r.returncode == 2
    assert "error" in json.loads(r.stdout)


def test_tail_reserve_shrinks_or_moves_the_tail():
    """The tail stage also loads final norm + lm_head (measured 1.15 GiB on M2.5): a plan
    that packs the tail to its VRAM brim OOMs at load (live, 2026-07-09). The landed tail's
    block must leave tail_reserve_mb of headroom on top of its layers."""
    from shard.plan import M25_PROFILE
    m = M25_PROFILE
    per_layer = m["layer_vram_mb"] + m["kv_mb_per_layer"]
    # boxes whose calibrated free lands layers*per_layer + <1400 of slack: the layers fit,
    # the lm_head reserve does NOT — exactly the live-OOM shape.
    free_gb = (11 * per_layer + m["reserve_mb"] + 500.0) / 1024
    nodes, rtt = _pool(6, free_gb=free_gb)
    plan = plan_ring(nodes, rtt)
    assert plan is not None
    _assert_tiles_simple(plan, 62)
    tail = next(s for s in plan["stages"] if s["tail"])
    free = free_gb * 1024 - m["reserve_mb"]    # the tail is never the head in this pool shape
    assert tail["layers"] * per_layer + m["tail_reserve_mb"] <= free + 1e-6
    # and the reserve is a MODEL knob: zeroing it restores the old packing behavior
    plan0 = plan_ring(nodes, rtt, model={"tail_reserve_mb": 0.0})
    tail0 = next(s for s in plan0["stages"] if s["tail"])
    assert tail0["layers"] >= tail["layers"]
