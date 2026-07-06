"""offline proof for plan_ring — vram parsing + node assembly + full pipeline, no fleet, $0.

stubs the ssh/vast helpers so plan_fleet runs end-to-end on fake boxes and we assert the
printed --layers string is a correct VRAM-fit that covers [0:total].

  python3 phase0/plan_ring_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan_ring


def test_parse_vram_single():
    assert plan_ring.parse_vram_gb("49140 MiB") == 48.0, plan_ring.parse_vram_gb("49140 MiB")
    assert plan_ring.parse_vram_gb("24564 MiB") == 24.0
    print("  OK parse_vram_single")


def test_parse_vram_multi_gpu():
    """two GPUs on one box -> summed."""
    out = "24564 MiB\n24564 MiB"
    assert plan_ring.parse_vram_gb(out) == 48.0, plan_ring.parse_vram_gb(out)
    print("  OK parse_vram_multi_gpu")


def test_parse_vram_units_and_garbage():
    assert plan_ring.parse_vram_gb("48 GiB") == 48.0
    assert plan_ring.parse_vram_gb("\n\n49140 MiB\n[some warning line]\n") == 48.0
    assert plan_ring.parse_vram_gb("") == 0.0
    print("  OK parse_vram_units_and_garbage")


def test_build_nodes():
    insts = [{"id": 1}, {"id": 2}, {"id": 3}]
    vram = {"1": 48.0, "2": 24.0, "3": 24.0}
    rtt = [[0, 30, 40], [30, 0, 25], [40, 25, 0]]
    nodes = plan_ring.build_nodes(insts, vram, rtt)
    assert nodes[0] == {"node_id": "1", "vram_gb": 48.0, "rtt_ms": {"2": 30, "3": 40}}, nodes[0]
    assert nodes[1]["rtt_ms"] == {"1": 30, "3": 25}
    print("  OK build_nodes")


def test_plan_fleet_full(monkeypatch_vram={"1": 48.0, "2": 24.0, "3": 24.0}):
    """end-to-end with stubbed vram + provided rtt: assert layers cover [0:78], fat node biggest."""
    insts = [{"id": 1, "geolocation": "TX"}, {"id": 2, "geolocation": "WA"}, {"id": 3, "geolocation": "CA"}]
    # stub query_vram so no ssh
    orig = plan_ring.query_vram
    plan_ring.query_vram = lambda _insts: monkeypatch_vram
    try:
        rtt = [[0, 30, 40], [30, 0, 25], [40, 25, 0]]
        p = plan_ring.plan_fleet(insts, "/root/models/gpt-oss-120b", 78, 1.05, 0.04,
                                 coordinator=None, rtt_matrix=rtt)
    finally:
        plan_ring.query_vram = orig
    assert p["ok"]
    # full coverage
    cur = 0
    for s in p["stages"]:
        assert s["lo"] == cur
        cur = s["hi"]
    assert cur == 78, f"coverage {cur} != 78"
    # fat node (id 1, 48GB) holds the most layers
    counts = {s["node_id"]: s["n_layers"] for s in p["stages"]}
    assert counts["1"] == max(counts.values()), f"fat node not biggest: {counts}"
    layers = ",".join(str(s["n_layers"]) for s in p["stages"])
    print(f"  OK plan_fleet_full: coord={p['coordinator']} ring={p['ring_order']} layers={layers}")


def test_plan_fleet_coord_pin():
    insts = [{"id": 1}, {"id": 2}, {"id": 3}]
    plan_ring.query_vram = lambda _i: {"1": 24.0, "2": 24.0, "3": 24.0}
    try:
        rtt = [[0, 30, 40], [30, 0, 25], [40, 25, 0]]
        p = plan_ring.plan_fleet(insts, "m", 60, 0.5, 0.0, coordinator="3", rtt_matrix=rtt)
    finally:
        pass
    assert p["coordinator"] == "3", p["coordinator"]
    assert p["ring_order"][0] == "3", "pinned coordinator must be the head"
    print("  OK plan_fleet_coord_pin")


if __name__ == "__main__":
    tests = [test_parse_vram_single, test_parse_vram_multi_gpu, test_parse_vram_units_and_garbage,
             test_build_nodes, test_plan_fleet_full, test_plan_fleet_coord_pin]
    print(f"plan_ring — {len(tests)} offline tests")
    for t in tests:
        t()
    print(f"ALL {len(tests)} PASS")
