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


def test_upload_aware_placement_prices_real_bytes():
    """M25_PROFILE never carried prefill_bytes/decode_bytes/decode_steps, so upload-aware placement
    (all nodes announcing up_mbps) optimized against ZERO byte cost — pure latency, uplinks ignored.
    With the measured payloads ((K+1)*H*2 decode, chunk*H*2 prefill) a 1 Mbps straggler must land
    the TAIL (the only upload-exempt prefill seat) and prefill_ms must reflect moving ~25 MB/hop."""
    from shard.plan import M25_PROFILE
    assert M25_PROFILE["decode_bytes"] == 9 * 3072 * 2.0      # (K+1)*H*2, K=8
    assert M25_PROFILE["prefill_bytes"] == 4096 * 3072 * 2.0  # chunk*H*2, prefill_chunk=4096
    assert M25_PROFILE["decode_steps"] >= 1
    nodes, rtt = _pool(6)
    for i, nd in enumerate(nodes):
        nd["up_mbps"] = 1.0 if i == 1 else 100.0   # node1: residential-cable-class uplink
    plan = plan_ring(nodes, rtt)
    assert plan is not None
    _assert_tiles_simple(plan, 62)
    assert plan["prefill_ms"] > 500                # ~2 s/hop at 100 Mbps; was ~121 ms (latency only)
    assert plan["order"][-1] == "node1"            # the slow uplink is spent on the exempt tail seat


def test_disconnected_node_never_heads_the_ring():
    """A node with NO usable path to anyone summed centrality 0 (unreachable edges were omitted,
    not penalized) and won mandatory head — an undeployable ring anchored on a partitioned box.
    Unreachable edges must count AGAINST a candidate, so the plan heads a connected node and
    drops the partitioned one entirely."""
    nodes, rtt = _pool(7)
    for j in range(7):
        if j != 4:
            rtt[4][j] = rtt[j][4] = 9000.0       # node4 is partitioned from the whole pool
    plan = plan_ring(nodes, rtt)
    assert plan is not None
    assert plan["head"] != "node4"
    assert "node4" not in plan["order"]
    _assert_tiles_simple(plan, 62)


def test_fully_partitioned_pool_returns_none():
    """Every pairwise path at the sentinel: no node can REACH enough capacity to serve, so the
    honest answer is None — not a 'ring' of unreachable hops with whoever scored centrality 0."""
    nodes, rtt = _pool(6)
    for i in range(6):
        for j in range(6):
            if i != j:
                rtt[i][j] = 9000.0
    assert plan_ring(nodes, rtt) is None


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


def test_fat_card_density_cap_shrinks_the_ring():
    """A 96 GB card announcing total_vram_mb gets the proven DENSITY scaled to its size
    (12 * 97887/32768 = 35 layers), not the flat 32 GB cap — so a fat-card pool plans a
    ring with FEWER hops than the all-5090 six-stage split."""
    nodes, rtt = _pool(6)
    nodes[0]["free_vram_mb"] = 97887.0
    nodes[0]["total_vram_mb"] = 97887.0
    plan = plan_ring(nodes, rtt)
    assert plan is not None
    _assert_tiles_simple(plan, 62)
    fat = next(s for s in plan["stages"] if s["id"] == "node0")
    assert fat["layers"] > 13                    # flat-cap behavior would pin it at <=13
    assert plan["k"] < 6                         # fewer, fatter stages: the fat card absorbed hops
    # a probe-verdict cap_layers wins outright over the density rule
    nodes[0]["cap_layers"] = 20
    plan20 = plan_ring(nodes, rtt)
    fat20 = next(s for s in plan20["stages"] if s["id"] == "node0")
    assert fat20["layers"] <= 20


def test_per_node_footprint_marlin_holds_fewer_layers():
    """A marlin-path card (~4060 MB/layer measured) in a cutlass pool must be planned at ITS
    footprint: same free VRAM, roughly half the layers of its cutlass twin."""
    nodes, rtt = _pool(7)
    nodes[6]["layer_vram_mb"] = 4060.0           # the H100/4090 dequant-path footprint class
    plan = plan_ring(nodes, rtt)
    assert plan is not None
    _assert_tiles_simple(plan, 62)
    by_id = {s["id"]: s for s in plan["stages"]}
    if "node6" in by_id:                         # if selected, its block must respect ITS footprint
        m = plan_ring.__globals__["M25_PROFILE"]
        free = 30.0 * 1024 - m["reserve_mb"]
        assert by_id["node6"]["layers"] * (4060.0 + m["kv_mb_per_layer"]) <= free + 1e-6
        assert by_id["node6"]["layers"] < 12     # cutlass twins hold 12-13; marlin can't


def test_measured_layer_ms_overrides_modeled_base():
    """A node announcing its probe-measured layer_ms (e.g. a box whose graph capture failed,
    running eager at ~4x) must be planned at that measurement — the planner balances stage
    time, so the slow box lands fewer layers than its VRAM twin."""
    nodes, rtt = _pool(7)
    nodes[5]["layer_ms"] = 2.6                   # eager-ish measured, vs base 0.65 modeled
    plan = plan_ring(nodes, rtt)
    assert plan is not None
    _assert_tiles_simple(plan, 62)
    by_id = {s["id"]: s for s in plan["stages"]}
    if "node5" in by_id:
        twin = max(s["layers"] for s in plan["stages"] if s["id"] not in ("node5", plan["head"]))
        assert by_id["node5"]["layers"] < twin


def test_load_peak_transient_gates_admission():
    """The measured load transient (marlin repack peaked 4.8 GB above resident, live 2026-07-09)
    subtracts from usable free — the admit-then-OOM fix, now honored per node in the plan."""
    nodes, rtt = _pool(6)
    plan_before = plan_ring(nodes, rtt)
    lay_before = {s["id"]: s["layers"] for s in plan_before["stages"]}
    nodes[3]["load_peak_extra_mb"] = 4824.0
    plan = plan_ring(nodes, rtt)
    assert plan is not None
    _assert_tiles_simple(plan, 62)
    by_id = {s["id"]: s for s in plan["stages"]}
    if "node3" in by_id and "node3" in lay_before:
        assert by_id["node3"]["layers"] < lay_before["node3"]


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


def test_tail_reserve_never_returns_an_unfittable_tail():
    """The old refinement loop ran 4 blind rounds and then returned the LAST spec even when its
    tail never fit: rotate the tail across three sacrificial nodes (each dies on its dock), and
    round 4 lands the tail FRESH on a node whose block + reserve exceeds its real budget — the
    26-on-20 shape: the returned tail held 6 MB of layers + a 20 MB lm_head reserve on a 20 MB
    budget (guaranteed OOM at load). A plan must fit, or the answer is None."""
    model = {"n_layers": 8, "layer_vram_mb": 1.0, "kv_mb_per_layer": 0.0, "layer_ms_base": 1.0,
             "reserve_mb": 0.0, "head_reserve_mb": 0.0, "tail_reserve_mb": 20.0,
             "cap_layers": 100, "head_layer_ms_mult": 1.0}
    frees = {"h": 2.0, "p": 6.9, "q": 6.9, "s": 6.9, "r": 20.0}
    ids = list(frees)
    nodes = [{"id": k, "free_vram_mb": frees[k], "subnet": f"10.{i}.0.0/24"}
             for i, k in enumerate(ids)]
    # h/p/q/s are mutually close (10 ms); r is far (40 ms) so it is only chosen when forced.
    rtt = [[0.0 if i == j else (40.0 if "r" in (ids[i], ids[j]) else 10.0)
            for j in range(5)] for i in range(5)]
    plan = plan_ring(nodes, rtt, model)
    if plan is not None:                         # a returned plan MUST fit its tail's real budget
        tail = next(s for s in plan["stages"] if s["tail"])
        assert tail["layers"] * 1.0 + 20.0 <= frees[tail["id"]] + 1e-6, \
            f"tail {tail['id']} needs {tail['layers'] + 20.0} MB on {frees[tail['id']]} MB"


def test_tail_reserve_docked_once_reappearing_tail_still_plans():
    """A node that reappears as tail must have the reserve modeled ONCE: the old loop docked it
    again on every reappearance (the check demanded the reserve on top of the already-docked
    budget) until the pool read as infeasible. This pool has an obvious valid plan — A holds 11
    layers, B tails 4 layers + the 6 MB reserve exactly filling its 10 MB budget."""
    model = {"n_layers": 15, "layer_vram_mb": 1.0, "kv_mb_per_layer": 0.0, "layer_ms_base": 1.0,
             "reserve_mb": 0.0, "head_reserve_mb": 0.0, "tail_reserve_mb": 6.0,
             "cap_layers": 100, "head_layer_ms_mult": 1.3}
    nodes = [{"id": "A", "free_vram_mb": 20.0, "subnet": "10.0.0.0/24"},
             {"id": "B", "free_vram_mb": 10.0, "subnet": "10.1.0.0/24"}]
    rtt = [[0.0, 10.0], [10.0, 0.0]]
    plan = plan_ring(nodes, rtt, model)
    assert plan is not None, "feasible pool rejected: the tail reserve was double-counted"
    _assert_tiles_simple(plan, 15)
    tail = next(s for s in plan["stages"] if s["tail"])
    assert tail["layers"] * 1.0 + 6.0 <= {"A": 20.0, "B": 10.0}[tail["id"]] + 1e-6
