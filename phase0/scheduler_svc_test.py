"""offline proof for scheduler_svc.plan — no socket, no GPU, $0.

asserts the ring plan the orchestrator will consume is correct: full layer coverage,
contiguous non-overlapping blocks in ring order, fat-node-first sizing, coordinator pick,
and clean failure when the pool can't hold the model.

  python3 phase0/scheduler_svc_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scheduler_svc import plan


def _check_contiguous_full(p, total):
    """blocks must tile [0:total] exactly: start at 0, no gaps, no overlaps, end at total."""
    stages = p["stages"]
    assert stages[0]["lo"] == 0, f"first block must start at 0, got {stages[0]['lo']}"
    cur = 0
    for s in stages:
        assert s["lo"] == cur, f"gap/overlap: stage {s['stage']} lo={s['lo']} expected {cur}"
        assert s["hi"] > s["lo"], f"empty block at stage {s['stage']}"
        assert s["n_layers"] == s["hi"] - s["lo"], "n_layers mismatch"
        cur = s["hi"]
    assert cur == total, f"coverage gap: tiled {cur} != {total}"
    # ring_order matches stage node order
    assert p["ring_order"] == [s["node_id"] for s in stages]


def test_even_homogeneous():
    """3 identical 24GB cards, 78-layer model -> ~26 each, full coverage."""
    req = {
        "model": "GLM-5.2", "total_layers": 78, "gb_per_layer": 0.5, "kv_gb_per_layer": 0.02,
        "nodes": [
            {"node_id": "A", "vram_gb": 24, "rtt_ms": {"B": 20, "C": 30}},
            {"node_id": "B", "vram_gb": 24, "rtt_ms": {"A": 20, "C": 25}},
            {"node_id": "C", "vram_gb": 24, "rtt_ms": {"A": 30, "B": 25}},
        ],
    }
    p = plan(req)
    assert p["ok"]
    _check_contiguous_full(p, 78)
    assert len(p["stages"]) == 3
    print("  OK even_homogeneous:", [(s["node_id"], s["lo"], s["hi"]) for s in p["stages"]])


def test_heterogeneous_fat_first():
    """48GB + 24GB + 24GB -> the 48GB node holds MORE layers than either 24GB node."""
    req = {
        "model": "GLM-5.2", "total_layers": 78, "gb_per_layer": 0.5,
        "nodes": [
            {"node_id": "fat", "vram_gb": 48, "rtt_ms": {"m": 30, "s": 40}},
            {"node_id": "m", "vram_gb": 24, "rtt_ms": {"fat": 30, "s": 25}},
            {"node_id": "s", "vram_gb": 24, "rtt_ms": {"fat": 40, "m": 25}},
        ],
    }
    p = plan(req)
    assert p["ok"]
    _check_contiguous_full(p, 78)
    counts = {s["node_id"]: s["n_layers"] for s in p["stages"]}
    assert counts["fat"] > counts["m"], f"fat node should hold more: {counts}"
    assert counts["fat"] > counts["s"], f"fat node should hold more: {counts}"
    print("  OK heterogeneous_fat_first:", counts)


def test_coordinator_pin():
    """explicit coordinator is honored."""
    req = {
        "model": "M", "total_layers": 40, "gb_per_layer": 0.5, "coordinator": "B",
        "nodes": [
            {"node_id": "A", "vram_gb": 24, "rtt_ms": {"B": 20, "C": 30}},
            {"node_id": "B", "vram_gb": 24, "rtt_ms": {"A": 20, "C": 25}},
            {"node_id": "C", "vram_gb": 24, "rtt_ms": {"A": 30, "B": 25}},
        ],
    }
    p = plan(req)
    assert p["coordinator"] == "B", p["coordinator"]
    _check_contiguous_full(p, 40)
    print("  OK coordinator_pin: coord=B")


def test_coordinator_auto_lowest_rtt():
    """no pin -> node with lowest mean rtt to the rest wins. C has mean (30+25)/2=27.5,
    B has (20+25)/2=22.5, A has (20+30)/2=25 -> B."""
    req = {
        "model": "M", "total_layers": 40, "gb_per_layer": 0.5,
        "nodes": [
            {"node_id": "A", "vram_gb": 24, "rtt_ms": {"B": 20, "C": 30}},
            {"node_id": "B", "vram_gb": 24, "rtt_ms": {"A": 20, "C": 25}},
            {"node_id": "C", "vram_gb": 24, "rtt_ms": {"A": 30, "B": 25}},
        ],
    }
    p = plan(req)
    assert p["coordinator"] == "B", f"expected B (lowest mean rtt), got {p['coordinator']}"
    print("  OK coordinator_auto_lowest_rtt: coord=B")


def test_insufficient_vram_raises():
    """pool can't hold the model -> ValueError (orchestrator turns this into a 400 + requeue)."""
    req = {
        "model": "huge", "total_layers": 200, "gb_per_layer": 2.0,
        "nodes": [
            {"node_id": "A", "vram_gb": 24, "rtt_ms": {"B": 20}},
            {"node_id": "B", "vram_gb": 24, "rtt_ms": {"A": 20}},
        ],
    }
    try:
        plan(req)
        assert False, "expected ValueError on insufficient VRAM"
    except ValueError as e:
        assert "insufficient" in str(e).lower()
        print("  OK insufficient_vram_raises:", str(e)[:60])


def test_zero_layer_node_dropped():
    """a tiny node the fit gives 0 layers is not emitted as a stage, coverage still full."""
    req = {
        "model": "M", "total_layers": 40, "gb_per_layer": 0.5,
        "nodes": [
            {"node_id": "big1", "vram_gb": 24, "rtt_ms": {"big2": 20, "tiny": 30}},
            {"node_id": "big2", "vram_gb": 24, "rtt_ms": {"big1": 20, "tiny": 25}},
            {"node_id": "tiny", "vram_gb": 3.0, "rtt_ms": {"big1": 30, "big2": 25}},  # < headroom+boundary
        ],
    }
    p = plan(req)
    _check_contiguous_full(p, 40)
    node_ids = {s["node_id"] for s in p["stages"]}
    assert "tiny" not in node_ids, "0-layer node must not be a stage"
    print("  OK zero_layer_node_dropped: stages =", sorted(node_ids))


if __name__ == "__main__":
    tests = [test_even_homogeneous, test_heterogeneous_fat_first, test_coordinator_pin,
             test_coordinator_auto_lowest_rtt, test_insufficient_vram_raises, test_zero_layer_node_dropped]
    print(f"scheduler_svc.plan — {len(tests)} offline tests")
    for t in tests:
        t()
    print(f"ALL {len(tests)} PASS")
