"""Regression + unit tests for shard.topology.select_ring (the self-optimizer's pure core).

Covers: (1) the BYTE-IDENTICAL legacy path (decode-only objective, no upload info) against a golden
snapshot; (2) the two adversarially-found false-"infeasible" bug classes (RTT pre-trim before the
feasibility check; subnet-blind k_min); (3) the new UPLOAD-AWARE objective (tails/drops slow uplinks,
request_ms = prefill + D*decode, SUM<->MAX prefill regime); (4) role relegation (total coverage, the
five roles, never high-latency->hot-standby); (5) purity (deterministic, no input mutation).

Run: `python3 tests/test_topology.py`  (also collectable by pytest as test_*).
"""
import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shard.topology import (select_ring, predict_step_ms, predict_prefill_ms, node_capacity,
                            optimal_loop, assign_layers)

MODEL = dict(n_layers=62, layer_vram_mb=1700.0, kv_mb_per_layer=150.0)
H, DT = 3072, 2
PF16K = 16384 * H * DT          # ~100MB prefill activation @16k
DEC = 10 * H * DT               # ~60KB decode activation


def _scen(seed, n):
    """Deterministic pool generator (matches the golden capture)."""
    import random
    rng = random.Random(seed); nodes = list(range(n))
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                L[i][j] = round(10 + rng.random() * 40, 1)
    c_out = [round(8 + rng.random() * 30, 1) for _ in range(n)]
    c_in = [round(7 + rng.random() * 30, 1) for _ in range(n)]
    free = {i: rng.choice([16000, 24000, 32000, 48000, 80000]) for i in range(n)}
    lms = {i: round(8 + rng.random() * 6, 2) for i in range(n)}
    sub = {i: f"net{i}" for i in range(n)}
    return nodes, L, c_out, c_in, free, lms, sub


# ---- golden: the LEGACY (up_mbps=None) output, captured from the pre-upload-aware select_ring ----
GOLDEN = {
    "s1": {"order": [3, 5, 0], "step_ms": 725.2, "tok_s_per_g": 1.38, "k": 3,
           "layers": {3: 29, 5: 25, 0: 8}, "dropped": [1, 2, 4, 6, 7]},
    "s2": {"order": [2, 7, 0, 3], "step_ms": 694.9, "tok_s_per_g": 1.44, "k": 4,
           "layers": {2: 43, 7: 17, 0: 1, 3: 1}, "dropped": [1, 4, 5, 6]},
    "s3": {"order": [1, 0, 7, 9, 4, 2, 8, 5], "step_ms": 772.8, "k": 8, "dropped": [3, 6]},
    "s4": None,
}


def _legacy(seed, n, mut=None, free=None, slack=3):
    nodes, L, c_out, c_in, fv, lms, sub = _scen(seed, n)
    if mut:
        sub.update(mut)
    if free is not None:
        fv = free
    return select_ring(nodes, L, c_out, c_in, free_vram_mb=fv, layer_ms=lms, subnet=sub,
                       slack=slack, require=0, **MODEL)


def test_legacy_byte_identical():
    """up_mbps omitted => today's exact decode-step selection & dict (no prefill_ms/roles keys)."""
    r1 = _legacy(1, 8)
    r2 = _legacy(2, 8, mut={3: "net1"})                        # co-location: nodes 1,3 share a subnet
    r3 = _legacy(3, 10, free={i: 16000 for i in range(10)}, slack=4)   # widen-k (small cards)
    r4 = _legacy(4, 3, free={i: 16000 for i in range(3)}, slack=2)     # infeasible: pool can't hold model
    for key, r in [("s1", r1), ("s2", r2), ("s3", r3), ("s4", r4)]:
        g = GOLDEN[key]
        if g is None:
            assert r is None, f"{key}: expected None, got {r}"
            continue
        for f in ("order", "step_ms", "tok_s_per_g", "k", "dropped"):
            if f in g:
                assert r[f] == g[f], f"{key}.{f}: {r[f]} != golden {g[f]}"
        if "layers" in g:
            assert r["layers"] == g["layers"], f"{key}.layers drift"
        assert "prefill_ms" not in r and "roles" not in r, f"{key}: legacy dict grew upload-aware keys"
        # invariants that must always hold
        assert sum(r["layers"].values()) == MODEL["n_layers"]
        assert all(v >= 1 for v in r["layers"].values())      # no empty stages
        assert len(set(r["order"])) == len(r["order"])        # no repeats
    print("ok  legacy byte-identical (4 scenarios incl. co-location, widen-k, infeasible)")


def test_no_false_infeasible_rtt_trim():
    """Bug class 1: the >_TRIM latency funnel must NOT trim out nodes feasibility needs. Build a 16-node
    pool where the only VRAM big enough to hold the model sits on the HIGHEST-RTT nodes (the funnel would
    drop them if it ran before the feasibility check)."""
    n = 16
    nodes = list(range(n))
    L = [[0.0 if i == j else 15.0 for j in range(n)] for i in range(n)]
    # nodes 0..11 = low RTT but TINY VRAM (can't hold the model alone); 12..15 = high RTT, fat cards.
    c_out = [5.0] * 12 + [200.0] * 4
    c_in = [5.0] * 12 + [200.0] * 4
    free = {i: (2000 if i < 12 else 80000) for i in range(n)}  # <1850 => cap 0 for the low-RTT dozen
    lms = {i: 10.0 for i in range(n)}
    sub = {i: f"net{i}" for i in range(n)}
    r = select_ring(nodes, L, c_out, c_in, free_vram_mb=free, layer_ms=lms, subnet=sub,
                    slack=4, require=12, **MODEL)
    assert r is not None, "false infeasible: funnel trimmed the feasibility-critical fat nodes"
    assert sum(r["layers"].values()) == MODEL["n_layers"]     # actually holds the model
    assert r["order"].count(12) or 12 in r["order"]           # the pinned fat node is present
    assert any(x in {12, 13, 14, 15} for x in r["order"])     # >=1 fat node carries the bulk (feasible)
    print("ok  no false-infeasible (RTT funnel keeps feasibility-critical nodes)")


def test_no_false_infeasible_subnet_kmin():
    """Bug class 2: k_min must count DISTINCT subnets. If the fattest cards share one subnet, a subnet-blind
    k_min underestimates the ring size and could report infeasible before widening."""
    n = 8
    nodes = list(range(n))
    L = [[0.0 if i == j else 20.0 for j in range(n)] for i in range(n)]
    c_out = [10.0] * n; c_in = [10.0] * n
    free = {i: 80000 for i in range(n)}                        # every card fat (holds ~45 layers)
    lms = {i: 10.0 for i in range(n)}
    # the two fattest would be a 2-stage ring, but they're CO-LOCATED -> need >=2 distinct subnets
    sub = {0: "A", 1: "A", 2: "A", 3: "B", 4: "C", 5: "D", 6: "E", 7: "F"}
    r = select_ring(nodes, L, c_out, c_in, free_vram_mb=free, layer_ms=lms, subnet=sub,
                    slack=2, require=3, **MODEL)
    assert r is not None, "false infeasible: subnet-blind k_min"
    subs = [sub[x] for x in r["order"]]
    assert len(subs) == len(set(subs)), "co-located stages in the ring"
    print("ok  no false-infeasible (subnet-honest k_min) + never co-locates")


def test_no_false_infeasible_funnel_subnet_blind():
    """Bug class 1b (adversarial review 2026-07-01): with >_TRIM usable nodes the latency funnel's `must`
    set must keep a DISTINCT-SUBNET cover. If the fattest cards are CO-LOCATED, a subnet-blind
    `must=by_cap[:k_min+slack]` fills with same-subnet nodes and trims the feasibility-critical
    distinct-subnet cards (they're high-RTT, so also absent from `keep`) -> false None though a ring exists."""
    n = 14
    L = [[0.0 if i == j else 15.0 for j in range(n)] for i in range(n)]
    c_out = [10.0] * 12 + [300.0, 300.0]                       # nodes 12,13 high RTT
    c_in = [10.0] * 12 + [300.0, 300.0]
    free = {i: 74000 for i in range(8)}                        # 0-7 co-located subnet A, fat (cap 40)
    free.update({i: 2000 for i in range(8, 12)})              # 8-11 distinct subnets, cap 1
    free[12] = free[13] = 74000                               # 12,13 distinct subnets, fat, high RTT (feasibility-critical)
    lms = {i: 10.0 for i in range(n)}
    sub = {i: "A" for i in range(8)}
    sub.update({i: f"S{i}" for i in range(8, 14)})
    r = select_ring(range(n), L, c_out, c_in, free_vram_mb=free, layer_ms=lms, subnet=sub,
                    slack=2, require=0, **MODEL)
    assert r is not None, "false infeasible: subnet-blind funnel `must` trimmed the distinct-subnet cover"
    assert sum(r["layers"].values()) == MODEL["n_layers"]
    subs = [sub[x] for x in r["order"]]
    assert len(subs) == len(set(subs)) and 0 in r["order"]
    print("ok  no false-infeasible (funnel `must` keeps a distinct-subnet cover)")


def _upool():
    """6 fat single-stage-capable cards; node 5 fiber, nodes 3,4 slow cable, 4 is a subnet-twin of 0."""
    n = 6
    L = [[0.0 if i == j else 20.0 for j in range(n)] for i in range(n)]
    c_out = [10.0] * n; c_in = [10.0] * n
    free = {i: 80000 for i in range(n)}
    lms = {i: 5.0 for i in range(n)}
    sub = {i: f"net{i}" for i in range(n)}; sub[4] = "net0"    # 4 co-located with 0
    up = {0: 500, 1: 500, 2: 400, 3: 8, 4: 6, 5: 900}
    return n, L, c_out, c_in, free, lms, sub, up


def _aware(n, L, c_out, c_in, free, lms, sub, up, S=16384, D=256, chunks=4, require=0, **extra):
    return select_ring(range(n), L, c_out, c_in, free_vram_mb=free, layer_ms=lms, subnet=sub,
                       slack=4, require=require, up_mbps=up, prefill_bytes=S * H * DT,
                       decode_bytes=DEC, decode_steps=D, prefill_chunks=chunks, **MODEL, **extra)


def test_upload_aware_tails_and_drops():
    """Slow-upload nodes are dropped from the ring; the lowest-upload IN-ring node sits at the tail
    (which forwards nothing); the aware ring has a lower TRUE prefill cost than the blind one."""
    n, L, c_out, c_in, free, lms, sub, up = _upool()
    r = _aware(n, L, c_out, c_in, free, lms, sub, up)
    assert 3 not in r["order"] and 4 not in r["order"], "kept a slow-cable node on the critical path"
    # tail = the min-upload node among those in the ring
    tail = r["order"][-1]
    assert up[tail] == min(up[x] for x in r["order"]), "did not tail the lowest-upload in-ring node"
    assert "prefill_ms" in r and "request_ms" in r and "roles" in r
    # request_ms == prefill_ms + D * step_ms (decomposition holds on unrounded values; allow the
    # rounding budget: each of prefill/request rounds to 0.1, and D copies of step's 0.1 rounding).
    assert abs(r["request_ms"] - (r["prefill_ms"] + 256 * r["step_ms"])) <= 0.05 * (2 + 256) + 0.2
    print(f"ok  upload-aware drops slow uplinks {sorted(set(range(n))-set(r['order']))}, "
          f"tails node {tail} (up={up[tail]}), request=prefill+D*step")


def test_prefill_sum_vs_max_regime():
    """C=1 (single blob per hop) => SUM of forward uploads; large C (fine chunking) => MAX (bottleneck
    stage). A 3-forwarder ring with distinct uplinks must satisfy MAX <= blob-SUM, and interpolate."""
    order = [0, 1, 2, 3]                                       # 3 forwarders (0,1,2) + tail 3
    up = {0: 100.0, 1: 50.0, 2: 25.0, 3: 10.0}                 # tail 3 slow but exempt
    L = {a: {b: 0.0 for b in range(4)} for a in range(4)}      # zero latency -> isolate transport
    c_out = {a: 0.0 for a in range(4)}; c_in = {a: 0.0 for a in range(4)}
    layers = {i: 1 for i in range(4)}
    pf_sum = predict_prefill_ms(order, layers, L, c_out, c_in, up, PF16K, prefill_chunks=1)
    pf_max = predict_prefill_ms(order, layers, L, c_out, c_in, up, PF16K, prefill_chunks=10_000_000)
    pf_mid = predict_prefill_ms(order, layers, L, c_out, c_in, up, PF16K, prefill_chunks=4)
    us = [PF16K * 8 / (u * 1000.0) for u in (100.0, 50.0, 25.0)]  # forwarders only (tail exempt)
    assert abs(pf_sum - sum(us)) < 1e-6, "C=1 must be the SUM of forward uploads"
    assert abs(pf_max - max(us)) < 1.0, "C->inf must approach the MAX (bottleneck stage)"
    assert pf_max < pf_mid < pf_sum, "chunking must interpolate MAX < mid < SUM"
    print(f"ok  prefill regime interpolates: MAX {pf_max/1000:.0f}s < C=4 {pf_mid/1000:.0f}s < SUM {pf_sum/1000:.0f}s")


def test_roles_total_coverage_and_kinds():
    """Every dropped node gets a role; the five kinds are reachable from the right drop reasons."""
    # Build a pool exercising all roles: a cap==0 tiny node, a fiber aggregator, a subnet-twin, a
    # compute-fine slow-upload node, and a slow-COMPUTE node.
    n = 9
    L = [[0.0 if i == j else 20.0 for j in range(n)] for i in range(n)]
    c_out = [10.0] * n; c_in = [10.0] * n
    free = {0: 80000, 1: 80000, 2: 80000, 3: 80000, 4: 80000, 5: 80000, 6: 80000, 7: 80000, 8: 500}  # 8: cap 0
    lms = {i: 5.0 for i in range(n)}; lms[6] = 40.0            # 6: very slow compute
    sub = {i: f"net{i}" for i in range(n)}; sub[7] = "net0"    # 7: subnet-twin of 0
    up = {0: 500, 1: 500, 2: 500, 3: 900, 4: 8, 5: 6, 6: 500, 7: 500, 8: 500}   # 3 fiber; 4,5 slow cable
    r = _aware(n, L, c_out, c_in, free, lms, sub, up, require=0)
    roles = r["roles"]
    assert set(roles) == set(r["dropped"]), "role coverage is not total"
    assert roles.get(8) == "weight-seeder", "cap==0 node must be a weight-seeder"
    kinds = set(roles.values())
    assert "hot-standby" in kinds or 7 in r["order"], "subnet-twin should be a hot-standby if dropped"
    assert kinds <= {"weight-seeder", "aggregator", "hot-standby", "decode-only-replica",
                     "spot-check-verifier"}, f"unexpected role: {kinds}"
    # a fiber node (>= ring best up, subnet-distinct) that is dropped must be an aggregator
    for d in r["dropped"]:
        if up[d] >= max(up[x] for x in r["order"]) and sub[d] not in {sub[x] for x in r["order"]}:
            assert roles[d] == "aggregator", f"fiber node {d} not tagged aggregator"
    print(f"ok  role relegation total coverage; kinds seen: {sorted(kinds)}")


def test_never_high_latency_hot_standby():
    """A dropped node routed to hot-standby must be a subnet-twin of a ring node (a latency-close spare) —
    we never fail over to a high-latency node. Verify hot-standby => shares a subnet with a ring stage."""
    import random
    rng = random.Random(3)
    checked = 0
    for _ in range(200):
        n = 8
        L = [[0.0 if i == j else round(15 + rng.random() * 60, 1) for j in range(n)] for i in range(n)]
        c_out = [round(10 + rng.random() * 50, 1) for _ in range(n)]
        c_in = [round(10 + rng.random() * 50, 1) for _ in range(n)]
        free = {i: rng.choice([24000, 48000, 80000]) for i in range(n)}
        lms = {i: round(4 + rng.random() * 4, 2) for i in range(n)}
        sub = {i: f"net{rng.randint(0, 4)}" for i in range(n)}  # deliberate subnet collisions
        up = {i: rng.choice([6, 8, 20, 100, 500, 900]) for i in range(n)}
        r = select_ring(range(n), L, c_out, c_in, free_vram_mb=free, layer_ms=lms, subnet=sub,
                        slack=3, require=0, up_mbps=up, prefill_bytes=PF16K, decode_bytes=DEC,
                        decode_steps=256, prefill_chunks=4, **MODEL)
        if r is None:
            continue
        ring_subs = {sub[x] for x in r["order"]}
        for node, role in r.get("roles", {}).items():
            if role == "hot-standby":
                assert sub[node] in ring_subs, "hot-standby is not a subnet-twin of any ring stage"
                checked += 1
    assert checked > 0, "test never exercised a hot-standby assignment"
    print(f"ok  hot-standby is always a latency-close subnet-twin ({checked} checked)")


def test_purity_deterministic_and_no_mutation():
    """Pure function: identical inputs -> identical outputs; inputs are never mutated."""
    n, L, c_out, c_in, free, lms, sub, up = _upool()
    L0, co0, ci0 = copy.deepcopy(L), copy.deepcopy(c_out), copy.deepcopy(c_in)
    f0, l0, s0, u0 = copy.deepcopy(free), copy.deepcopy(lms), copy.deepcopy(sub), copy.deepcopy(up)
    r_a = _aware(n, L, c_out, c_in, free, lms, sub, up)
    r_b = _aware(n, L, c_out, c_in, free, lms, sub, up)
    assert r_a == r_b, "non-deterministic output"
    assert L == L0 and c_out == co0 and c_in == ci0, "mutated latency inputs"
    assert free == f0 and lms == l0 and sub == s0 and up == u0, "mutated capability inputs"
    print("ok  pure: deterministic + no input mutation")


def test_edge_cases():
    """Missing upload sample -> conservative floor (not a crash); decode_steps=0 -> prefill-only objective."""
    n, L, c_out, c_in, free, lms, sub, up = _upool()
    up_partial = {k: v for k, v in up.items() if k != 3}       # node 3 has no upload sample
    r = _aware(n, L, c_out, c_in, free, lms, sub, up_partial)  # must not raise
    assert r is not None and 3 not in r["order"]              # unmeasured -> treated as poor uplink -> off critical path
    r0 = _aware(n, L, c_out, c_in, free, lms, sub, up, D=0)    # decode_steps=0 -> request == prefill
    assert abs(r0["request_ms"] - r0["prefill_ms"]) <= 1.0
    assert r0["tok_s_per_g"] > 0                               # step_ms still the real decode step
    print("ok  edge cases: missing upload floors safely; decode_steps=0 -> prefill-only")


TESTS = [test_legacy_byte_identical, test_no_false_infeasible_rtt_trim, test_no_false_infeasible_subnet_kmin,
         test_no_false_infeasible_funnel_subnet_blind, test_upload_aware_tails_and_drops,
         test_prefill_sum_vs_max_regime, test_roles_total_coverage_and_kinds,
         test_never_high_latency_hot_standby, test_purity_deterministic_and_no_mutation, test_edge_cases]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"\nALL {len(TESTS)} topology tests passed.")
