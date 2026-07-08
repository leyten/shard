"""Boundary-layer pinning tests — the open-admission privacy rail in shard.topology / shard.plan.

The threat model (measured against the engine, not assumed): the HEAD is handed raw prompt token ids
to embed; the TAIL computes logits and returns argmax token ids (at prefill: the greedy next-token at
every prompt position — a near-copy of the prompt); and activations near either boundary invert back
toward the prompt far more easily than deep-middle ones. So with `trusted` given, select_ring must
(1) put trusted nodes at BOTH ends unconditionally, (2) keep every stage whose block intersects
[0, boundary_in) or [n_layers-boundary_out, n_layers) trusted, (3) never spill boundary layers onto
an untrusted neighbor when a trusted end can absorb them, (4) stay exactly the legacy objective when
`trusted=None`, and (5) never false-infeasible: an ends-constrained optimum must be SEARCHED for, not
post-hoc-rejected, and the >_TRIM funnel must keep a trusted cover.

Run: `python3 tests/test_boundary_pinning.py`  (also collectable by pytest as test_*).
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shard.plan import plan_ring
from shard.topology import select_ring

MODEL = dict(n_layers=62, layer_vram_mb=1700.0, kv_mb_per_layer=150.0)
H, DT = 3072, 2
PF16K = 16384 * H * DT
DEC = 10 * H * DT


def _flat_pool(n, free_mb=31000.0, lms=11.0):
    """symmetric flat mesh: every hop 20ms, identical fat-ish cards, distinct subnets."""
    L = [[0.0 if i == j else 20.0 for j in range(n)] for i in range(n)]
    c_out = [20.0] * n
    c_in = [20.0] * n
    free = {i: float(free_mb) for i in range(n)}
    layer_ms = {i: float(lms) for i in range(n)}
    sub = {i: f"net{i}" for i in range(n)}
    return L, c_out, c_in, free, layer_ms, sub


def _ring(n, trusted, b_in=0, b_out=0, require=0, free=None, lms=None, sub=None, L=None,
          c_out=None, c_in=None, slack=None, **extra):
    Ld, od, id_, fd, ld, sd = _flat_pool(n)
    return select_ring(range(n), L or Ld, c_out or od, c_in or id_,
                       free_vram_mb=free or fd, layer_ms=lms or ld, subnet=sub or sd,
                       slack=n if slack is None else slack, require=require,
                       trusted=trusted, boundary_in=b_in, boundary_out=b_out, **MODEL, **extra)


def _boundary_holders(r, b_in, b_out):
    """nodes whose block intersects a leaky layer range."""
    out = set()
    for n, (lo, hi) in r["blocks"].items():
        if lo < b_in or hi > MODEL["n_layers"] - b_out:
            out.add(n)
    return out


def test_legacy_unchanged_without_trusted():
    """trusted=None must be the exact legacy path: same ring, no `boundary` key in the spec."""
    n = 6
    L, c_out, c_in, free, lms, sub = _flat_pool(n)
    a = select_ring(range(n), L, c_out, c_in, free_vram_mb=free, layer_ms=lms, subnet=sub,
                    slack=n, require=0, **MODEL)
    b = _ring(n, trusted=None)
    assert a == b, "trusted=None drifted from the legacy result"
    assert "boundary" not in a
    print("ok  trusted=None is the exact legacy path (no spec drift, no boundary key)")


def test_ends_always_trusted_even_without_boundary_layers():
    """b_in=b_out=0 still forces trusted ends: the head embeds raw token ids, the tail argmaxes
    logits into token ids — those ROLES leak regardless of which layers they hold."""
    n = 6
    r = _ring(n, trusted={0, 3}, b_in=0, b_out=0)
    assert r is not None
    assert r["order"][0] == 0 and r["order"][-1] == 3, f"untrusted end in {r['order']}"
    assert set(r["boundary"]) == {0, 3}
    print("ok  head+tail trusted unconditionally (roles leak, not just layers)")


def test_untrusted_never_holds_boundary_layers():
    """with b_in/b_out set, no untrusted stage's block may intersect a leaky range."""
    n = 7
    trusted = {0, 5}
    r = _ring(n, trusted=trusted, b_in=8, b_out=4)
    assert r is not None
    holders = _boundary_holders(r, 8, 4)
    assert holders <= trusted, f"untrusted node holds boundary layers: {holders - trusted}"
    assert set(r["boundary"]) >= holders
    for x in r["order"]:
        if x not in trusted:
            lo, hi = r["blocks"][x]
            assert lo >= 8 and hi <= MODEL["n_layers"] - 4, f"stranger {x} holds [{lo},{hi})"
    print("ok  strangers hold only deep-middle layers (boundary ranges fully trusted)")


def test_trusted_end_absorbs_boundary_instead_of_spilling():
    """floors: a SLOW trusted tail would greedily get 1 layer, spilling the last layers onto its
    untrusted neighbor — the floor makes it absorb the whole b_out range instead."""
    n = 6
    lms = {i: 5.0 for i in range(n)}
    lms[3] = 40.0                                    # trusted tail is the slowest node in the pool
    r = _ring(n, trusted={0, 3}, b_in=1, b_out=6, lms=lms)
    assert r is not None
    assert r["order"][-1] == 3
    lo, hi = r["blocks"][3]
    assert hi == MODEL["n_layers"] and hi - lo >= 6, f"tail block [{lo},{hi}) doesn't cover b_out"
    assert _boundary_holders(r, 1, 6) <= {0, 3}
    print("ok  slow trusted end absorbs its boundary range (no spill onto a stranger)")


def test_boundary_spans_two_trusted_stages_when_end_is_thin():
    """a thin trusted head (cap < b_in) is legal IF the spill lands on another trusted stage."""
    n = 6
    free = {i: 31000.0 for i in range(n)}
    free[0] = 8000.0                                 # head cap ~3-4 layers < b_in
    trusted = {0, 1, 5}
    r = _ring(n, trusted=trusted, b_in=8, b_out=2, free=free)
    assert r is not None
    assert _boundary_holders(r, 8, 2) <= trusted
    assert r["order"][1] in trusted, "the spill stage after a thin trusted head must be trusted"
    print("ok  boundary may span multiple stages iff every holder is trusted")


def test_no_false_infeasible_boundary_spill_needs_order_search():
    """REGRESSION (adversarial fuzz 2026-07-08): when b_out exceeds the tail node's capacity the
    output boundary SPILLS onto the second-to-last stage, so it too must be trusted. The old code
    floored only the two ends, picked ONE min-latency order, and rejected the subset if that order's
    greedy fill spilled onto an untrusted neighbor — even when another order of the SAME subset seats
    a trusted node there. Here the only safe ring needs the order search to place the two small
    trusted tails (cap 8 each < b_out 9) adjacent at the back; a single-order search false-infeasibles."""
    n = 5
    # caps: node0->25 (fat, trusted head), node1->25 (fat, untrusted middle), node2,3->8 (thin trusted),
    # node4->12 (untrusted). Only trusted nodes {0,2,3}; tail+its neighbor must be trusted to cover b_out=9.
    free = {0: 48000.0, 1: 48000.0, 2: 16000.0, 3: 16000.0, 4: 24000.0}
    sub = {0: "s4", 1: "s3", 2: "s0", 3: "s1", 4: "sX"}
    L, c_out, c_in, _, _, _ = _flat_pool(n)
    lms = {i: 10.0 for i in range(n)}
    r = select_ring(range(n), L, c_out, c_in, free_vram_mb=free, layer_ms=lms, subnet=sub,
                    slack=n, require=0, trusted={0, 2, 3}, boundary_in=8, boundary_out=9, **MODEL)
    assert r is not None, "false infeasible: a boundary-spill ring exists but the order search missed it"
    assert r["order"][0] == 0
    holders = _boundary_holders(r, 8, 9)
    assert holders <= {0, 2, 3}, f"boundary spilled onto an untrusted node: {holders - {0,2,3}}"
    # both tail and its neighbor are the thin trusted pair covering the 9-layer output boundary
    assert r["order"][-1] in {2, 3} and r["order"][-2] in {0, 2, 3}
    print("ok  no false-infeasible when the boundary spills onto a trusted neighbor (order search finds it)")


def test_honest_infeasible_cases():
    """pinning with no trusted nodes, an untrusted require, or trusted nodes that can't cover the
    ends must return None — never a ring that leaks."""
    n = 5
    assert _ring(n, trusted=set()) is None, "empty trusted set must be infeasible"
    assert _ring(n, trusted={1, 2}, require=0) is None, "untrusted require/coord must be infeasible"
    # only ONE trusted node but k>=2 needed: head and tail can't both be trusted
    free = {i: 31000.0 for i in range(n)}            # ~17 layers max/node -> k_min 4
    assert _ring(n, trusted={0}, free=free) is None, "single trusted node can't hold both ends"
    print("ok  honest infeasible: no-trusted / untrusted-require / one-trusted-two-ends -> None")


def test_ends_constrained_search_not_post_hoc_rejection():
    """the unconstrained latency optimum ends at an UNTRUSTED node; a dearer trust-valid order
    exists. Post-hoc rejection of the optimum would falsely skip the subset — the ends-constrained
    search must find the valid order."""
    n = 4                                            # caps force all 4 into the ring (k_min=4)
    L = [[0.0] * n for _ in range(n)]
    cheap, dear = 5.0, 60.0
    for i in range(n):
        for j in range(n):
            if i != j:
                L[i][j] = dear
    # cheapest Hamiltonian path from 0: 0->2->3->1? no — craft: 0->2 cheap, 2->3 cheap, 3->1 dear,
    # 0->2->3 then ending at untrusted 3 is the optimum; trusted tail is 1.
    L[0][2] = L[2][3] = cheap                        # unconstrained optimum: [0,2,3] ends untrusted 3
    L[3][1] = dear                                   # valid order [0,2,3,1] exists but costs more
    c_out = [1.0, dear, cheap, dear]
    c_in = [1.0, dear, dear, cheap]                  # returning from 3 cheap -> optimum tails 3
    free = {i: 31000.0 for i in range(n)}            # ~17-layer caps: need all 4 nodes
    lms = {i: 10.0 for i in range(n)}
    sub = {i: f"net{i}" for i in range(n)}
    r = select_ring(range(n), L, c_out, c_in, free_vram_mb=free, layer_ms=lms, subnet=sub,
                    slack=0, require=0, trusted={0, 1}, boundary_in=2, boundary_out=2, **MODEL)
    assert r is not None, "false infeasible: trust-valid order exists but was never searched"
    assert r["order"][0] == 0 and r["order"][-1] == 1
    assert _boundary_holders(r, 2, 2) <= {0, 1}
    print("ok  ends-constrained order SEARCH (a rejected unconstrained optimum can't hide a valid ring)")


def test_funnel_keeps_trusted_cover():
    """>_TRIM pool where every trusted node is HIGH-RTT (outside the low-latency `keep`): the funnel
    must keep a trusted cover or pinning false-infeasibles on big pools."""
    n = 18
    L = [[0.0 if i == j else 15.0 for j in range(n)] for i in range(n)]
    c_out = [10.0] * 16 + [250.0, 250.0]
    c_in = [10.0] * 16 + [250.0, 250.0]
    free = {i: 31000.0 for i in range(n)}
    lms = {i: 10.0 for i in range(n)}
    sub = {i: f"net{i}" for i in range(n)}
    trusted = {0, 16, 17}                            # 16,17: high-RTT -> not in `keep`; 0 = require
    r = select_ring(range(n), L, c_out, c_in, free_vram_mb=free, layer_ms=lms, subnet=sub,
                    slack=2, require=0, trusted=trusted, boundary_in=4, boundary_out=4, **MODEL)
    assert r is not None, "false infeasible: funnel trimmed every trusted tail candidate"
    assert r["order"][0] == 0 and r["order"][-1] in {16, 17}
    print("ok  latency funnel keeps a trusted cover (no false-infeasible on big pools)")


def test_relegation_untrusted_twin_never_boundary_hot_standby():
    """an UNTRUSTED subnet-twin of a boundary stage must not be its hot-standby (failover would
    hand the leaky block to a stranger); a twin of a deep-middle stage may still standby."""
    n = 7
    L, c_out, c_in, free, lms, sub = _flat_pool(n)
    sub[5] = "net0"                                  # 5 = untrusted twin of the trusted head 0
    sub[6] = "net2"                                  # 6 = untrusted twin of (middle) node 2
    up = {i: 500.0 for i in range(n)}
    r = select_ring(range(n), L, c_out, c_in, free_vram_mb=free, layer_ms=lms, subnet=sub,
                    slack=n, require=0, trusted={0, 4}, boundary_in=4, boundary_out=4,
                    up_mbps=up, prefill_bytes=PF16K, decode_bytes=DEC, decode_steps=256,
                    prefill_chunks=4, **MODEL)
    assert r is not None
    roles = r["roles"]
    assert set(roles) == set(r["dropped"]), "role coverage must stay total under pinning"
    if 5 in roles:
        assert roles[5] != "hot-standby", "untrusted twin of a boundary stage became its failover"
    if 6 in roles and 2 in r["order"] and 2 not in set(r["boundary"]):
        assert roles[6] == "hot-standby", "trust-irrelevant middle twin lost its standby role"
    print(f"ok  relegation: boundary standby denied to strangers; middle standby kept ({roles})")


def test_all_trusted_matches_legacy_objective():
    """when EVERY node is trusted and boundaries are 0, the constraint is vacuous — the chosen ring
    must be as GOOD as the legacy optimum (trust is a constraint, never a score). On a symmetric mesh
    several orders tie at the optimum; the two order searches break the tie differently, so we assert
    cost-equivalence (same step_ms, same k, same layer multiset), not an identical tie-break. An
    asymmetric mesh with a unique optimum then pins the exact order too."""
    n = 6
    L, c_out, c_in, free, lms, sub = _flat_pool(n)
    legacy = select_ring(range(n), L, c_out, c_in, free_vram_mb=free, layer_ms=lms, subnet=sub,
                         slack=n, require=0, **MODEL)
    pinned = _ring(n, trusted=set(range(n)))
    assert pinned["step_ms"] == legacy["step_ms"] and pinned["k"] == legacy["k"]
    assert sorted(pinned["layers"].values()) == sorted(legacy["layers"].values())
    # asymmetric mesh: a unique cheapest loop -> the exact order must match too
    import random
    rng = random.Random(5)
    La = [[0.0 if i == j else round(rng.uniform(10, 60), 2) for j in range(n)] for i in range(n)]
    ca_out = [round(rng.uniform(8, 40), 2) for _ in range(n)]
    ca_in = [round(rng.uniform(8, 40), 2) for _ in range(n)]
    leg = select_ring(range(n), La, ca_out, ca_in, free_vram_mb=free, layer_ms=lms, subnet=sub,
                      slack=n, require=0, **MODEL)
    pin = select_ring(range(n), La, ca_out, ca_in, free_vram_mb=free, layer_ms=lms, subnet=sub,
                      slack=n, require=0, trusted=set(range(n)), boundary_in=0, boundary_out=0, **MODEL)
    assert pin["order"] == leg["order"], f"{pin['order']} != {leg['order']} on a unique-optimum mesh"
    assert pin["step_ms"] == leg["step_ms"]
    print("ok  all-trusted + b=0 == legacy optimum (cost-equivalent; exact order on a unique optimum)")


def test_purity_deterministic_no_mutation():
    n = 7
    L, c_out, c_in, free, lms, sub = _flat_pool(n)
    args = (copy.deepcopy(L), copy.deepcopy(c_out), copy.deepcopy(c_in),
            copy.deepcopy(free), copy.deepcopy(lms), copy.deepcopy(sub))
    trusted = {0, 4}
    a = select_ring(range(n), L, c_out, c_in, free_vram_mb=free, layer_ms=lms, subnet=sub,
                    slack=n, require=0, trusted=trusted, boundary_in=6, boundary_out=4, **MODEL)
    b = select_ring(range(n), L, c_out, c_in, free_vram_mb=free, layer_ms=lms, subnet=sub,
                    slack=n, require=0, trusted=trusted, boundary_in=6, boundary_out=4, **MODEL)
    assert a == b, "non-deterministic under pinning"
    assert (L, c_out, c_in, free, lms, sub) == args, "inputs mutated"
    assert trusted == {0, 4}, "trusted set mutated"
    print("ok  pure under pinning: deterministic, no input mutation")


def test_overlap_and_oversize_boundaries_not_false_infeasible():
    """REGRESSION (adversarial review 2026-07-08): when the two windows meet/overlap
    (b_in + b_out >= n_layers) the WHOLE model is boundary. The old code summed an independent front
    and back floor -> double-counted the shared layers -> assign_layers None even for an ALL-TRUSTED
    ring (it failed hardest exactly when the operator was most cautious). And an oversize single window
    (b_in > n_layers) overflowed the layer math. Both now: clamp to n_layers, and the overlap branch
    requires all-trusted + tiles freely."""
    n = 5
    L, c_out, c_in, free, lms, sub = _flat_pool(n, free_mb=80000.0)
    # all trusted, overlapping windows -> every layer boundary -> must FORM (all-trusted is always safe)
    r = select_ring(range(n), L, c_out, c_in, free_vram_mb=free, layer_ms=lms, subnet=sub,
                    slack=n, require=0, trusted=set(range(n)), boundary_in=40, boundary_out=40, **MODEL)
    assert r is not None, "false infeasible: all-trusted overlapping-boundary ring refused"
    assert set(r["order"]) <= set(range(n)) and sum(r["layers"].values()) == 62
    # oversize single window clamps to n_layers, still forms all-trusted
    r2 = select_ring(range(n), L, c_out, c_in, free_vram_mb=free, layer_ms=lms, subnet=sub,
                     slack=n, require=0, trusted=set(range(n)), boundary_in=1000, boundary_out=0, **MODEL)
    assert r2 is not None, "false infeasible: oversize b_in didn't clamp"
    # whole-model boundary with an untrusted node: it must be EXCLUDED (every layer is leaky) or None
    r3 = select_ring(range(n), L, c_out, c_in, free_vram_mb=free, layer_ms=lms, subnet=sub,
                     slack=n, require=0, trusted={0, 1, 2, 3}, boundary_in=40, boundary_out=40, **MODEL)
    assert r3 is None or 4 not in r3["order"], "LEAK: untrusted node in a whole-model-boundary ring"
    # only one trusted node but the model needs untrusted capacity -> fail closed
    tight = {0: 80000.0, 1: 16000.0, 2: 16000.0, 3: 16000.0, 4: 16000.0}
    r4 = select_ring(range(n), L, c_out, c_in, free_vram_mb=tight, layer_ms=lms, subnet=sub,
                     slack=n, require=0, trusted={0}, boundary_in=40, boundary_out=40, **MODEL)
    assert r4 is None, "whole-model boundary must fail closed when it needs an untrusted node"
    print("ok  overlap/oversize boundaries: no false-infeasible, whole-model stays all-trusted")


def test_seam_trusted_flag_is_fail_closed():
    """REGRESSION (adversarial review 2026-07-08): the plan.py `trusted` flag is the security boundary
    and must be read fail-CLOSED — only a genuine bool True marks a node trusted. A truthy STRING like
    "false" (a plausible mis-serialization) must NOT sneak a stranger into the trust set (that would
    fail OPEN — the one path to a stranger on a boundary while the plan claims to be pinned). Also:
    duplicate node ids are rejected (they'd collide in the output maps)."""
    n = 6
    rtt = [[0.0 if i == j else 20.0 for j in range(n)] for i in range(n)]
    # every node self-reports trusted:"false" (truthy string) -> NONE are trusted -> can't pin -> null
    bad = [{"id": f"x{i}", "free_vram_mb": 32000.0, "subnet": f"{i}.0/24", "trusted": "false"} for i in range(n)]
    assert plan_ring(bad, rtt, privacy={"boundary_in": 8, "boundary_out": 8}) is None, \
        "fail-OPEN: a truthy-string trusted flag admitted strangers"
    # genuine bool True still works
    good = [{"id": f"y{i}", "free_vram_mb": 32000.0, "subnet": f"{i}.0/24", "trusted": i < 3} for i in range(n)]
    p = plan_ring(good, rtt, privacy={"boundary_in": 8, "boundary_out": 8})
    assert p is not None and p["head"] in ("y0", "y1", "y2")
    # int 1 is NOT accepted (strict bool) -> those nodes untrusted -> too few trusted -> null
    inty = [{"id": f"z{i}", "free_vram_mb": 32000.0, "subnet": f"{i}.0/24", "trusted": 1} for i in range(n)]
    assert plan_ring(inty, rtt, privacy={"boundary_in": 8, "boundary_out": 8}) is None, \
        "int truthy must not mark trusted (strict bool)"
    # duplicate node ids rejected
    dup = [{"id": "same", "free_vram_mb": 32000.0, "subnet": f"{i}.0/24", "trusted": True} for i in range(3)]
    try:
        plan_ring(dup, [[0, 1, 1], [1, 0, 1], [1, 1, 0]], privacy={"boundary_in": 4, "boundary_out": 4})
        assert False, "duplicate node ids should raise"
    except ValueError:
        pass
    print("ok  seam trusted flag fail-closed (strict bool); duplicate ids rejected")


def test_plan_seam_threads_privacy():
    """plan_ring: trusted flags + privacy thread through; head = most central TRUSTED node; stages
    carry `boundary`; the privacy block lists the trust-critical stages; privacy=None == before."""
    n = 6
    nodes = [{"id": f"n{i}", "free_vram_mb": 32000.0, "subnet": f"net{i}",
              "trusted": i in (2, 4)} for i in range(n)]
    rtt = [[0.0 if i == j else (10.0 if 2 in (i, j) else 30.0) for j in range(n)] for i in range(n)]
    plan = plan_ring(nodes, rtt, privacy={"boundary_in": 6, "boundary_out": 4})
    assert plan is not None
    assert plan["head"] == "n2", "head must be the most central TRUSTED capable node"
    assert plan["order"][-1] == "n4", "tail must be trusted"
    for st in plan["stages"]:
        assert "boundary" in st
        if st["id"] not in ("n2", "n4"):
            assert not st["boundary"] and st["lo"] >= 6 and st["hi"] <= 62 - 4
    pv = plan["privacy"]
    assert pv["boundary_in"] == 6 and pv["boundary_out"] == 4
    assert set(pv["boundary_stages"]) == {st["id"] for st in plan["stages"] if st["boundary"]}
    # pinning with zero trusted nodes -> honestly unplannable
    bare = [dict(nd, trusted=False) for nd in nodes]
    assert plan_ring(bare, rtt, privacy={"boundary_in": 6, "boundary_out": 4}) is None
    # privacy=None -> no privacy/boundary keys anywhere (legacy shape)
    legacy = plan_ring(nodes, rtt)
    assert legacy is not None and "privacy" not in legacy
    assert all("boundary" not in st for st in legacy["stages"])
    print("ok  plan seam: trusted head/tail, boundary-flagged stages, privacy block, legacy shape intact")


TESTS = [test_legacy_unchanged_without_trusted, test_ends_always_trusted_even_without_boundary_layers,
         test_untrusted_never_holds_boundary_layers, test_trusted_end_absorbs_boundary_instead_of_spilling,
         test_boundary_spans_two_trusted_stages_when_end_is_thin,
         test_no_false_infeasible_boundary_spill_needs_order_search, test_honest_infeasible_cases,
         test_ends_constrained_search_not_post_hoc_rejection, test_funnel_keeps_trusted_cover,
         test_relegation_untrusted_twin_never_boundary_hot_standby, test_all_trusted_matches_legacy_objective,
         test_overlap_and_oversize_boundaries_not_false_infeasible, test_seam_trusted_flag_is_fail_closed,
         test_purity_deterministic_no_mutation, test_plan_seam_threads_privacy]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"\nALL {len(TESTS)} boundary-pinning tests passed.")
