"""shard.mesh.placement: capability-driven fit, roles, join priority, stickiness, repair, paths.

Fleets are MIXED on purpose (vision principle): every device family in tests/mesh_fleet.py is a
DeviceInfo record, never a card name placement could key on.
"""
import random

import pytest

from mesh_churn import SIM_PROFILE, _node as sim_node
from mesh_fleet import DEVICES, DISTILL_9B, GiB, node
from shard.mesh import placement as P
from shard.mesh.placement import (MIMO_V26_FLASH as FLASH, MIMO_V26_PRO as PRO, Cell, Fit, Holding, Node,
                                  Policy, SmallModel, candidate_paths, classify, stage_min_layers)
from shard.runtime.api import Capability

CATALOG = (SmallModel(DISTILL_9B["name"], DISTILL_9B["need_bytes"], tuple(DISTILL_9B["formats"])),)
POL = Policy(small_models=CATALOG)


def N(nid, kind, **kw):
    return Node.from_dict(node(nid, kind, **kw))


def spans_by_edge(n, prof=FLASH, pol=POL):
    sp = Fit(n, prof, pol).spans()
    L = prof.n_layers
    mx = lambda sel: max((h - l for l, h in sp if sel(l, h)), default=0)          # noqa: E731
    return mx(lambda l, h: l == 0), mx(lambda l, h: 0 < l and h < L), mx(lambda l, h: h == L)


def seat(cell, nid, lo, hi, layers=None, now=0.0):
    """White-box: put a sim-profile holder on [lo, hi) directly."""
    n = sim_node(nid, layers or (hi - lo)) if isinstance(nid, int) else nid
    cell.holdings[n.id] = Holding(n, classify(n, cell.profile, cell.policy), (lo, hi))
    cell.persisted[n.id] = (lo, hi)
    cell._refresh(now)
    return n


def sim_cell(**kw):
    base = dict(k_min=2, leg_ms=0.0, plan_sessions=0, unified_host_bytes=0, hole_repair_s=600.0,
                kmin_repair_s=1800.0, move_interval_s=600.0)
    return Cell(SIM_PROFILE, Policy(**{**base, **kw}))


# ── fit ───────────────────────────────────────────────────────────────────────────────────────

# design-runtime §5, Flash, N=8 x 32k column (card x 0.97 - reserve). Ours starts from the
# describe() free memory in mesh_fleet, so fat cards may land one layer higher.
@pytest.mark.parametrize("kind,table", [("4060-8g", 1), ("3060-12g", 2), ("4060ti-16g", 4), ("3090", 6),
                                        ("5090", 8), ("a6000-48g", 12), ("pro6000-96g", 26)])
def test_flash_mid_fit_matches_design_runtime_table(kind, table):
    _, mid, _ = spans_by_edge(N("X", kind))
    assert table <= mid <= table + 1


@pytest.mark.parametrize("kind,table", [("5090", 3), ("a6000-48g", 5), ("pro6000-96g", 11)])
def test_fit_comes_from_the_profile_not_the_card(kind, table):
    _, mid, _ = spans_by_edge(N("X", kind), PRO)
    assert table <= mid <= table + 1                       # Pro table, same devices


def test_edges_cost_the_tail_and_layer0_is_light():
    for kind in ("3090", "5090", "pro6000-96g", "spark-128g"):
        head, mid, tail = spans_by_edge(N("X", kind))
        assert tail < mid <= head, kind                    # lm_head + MTP + embed copy ~ 1 layer


def test_spark_holds_about_30_flash_layers_and_a_3090_six():
    assert spans_by_edge(N("S", "spark-128g"))[1] in (30, 31, 32)
    assert spans_by_edge(N("G", "3090"))[1] == 6


def test_unified_memory_pays_for_host_state():
    d = node("X", "pro6000-96g")
    d_uni = {**d, "device": {**d["device"], "unified_memory": True}}
    assert spans_by_edge(Node.from_dict(d_uni))[1] < spans_by_edge(Node.from_dict(d))[1]


def _cap(model, lo, hi, layer_bytes, service_ms=()):
    return Capability(device=Node.from_dict(node("X", "4090")).device, model=model, lo=lo, hi=hi,
                      is_head=False, is_tail=False, layer_kinds=FLASH.layer_kinds[lo:hi],
                      swa_window=128 if "swa" in FLASH.layer_kinds[lo:hi] else 0,
                      hidden_size=4096, vocab_size=152576, wire_dtype="bf16", codecs_in=("bf16",),
                      logits_dtype="f32", kv_dtype="bf16", kv_layout_id="kvl1-x", global_kv_bytes_per_pos=0,
                      swa_kv_bytes_per_pos=5120, kv_page_positions=1, kv_free_bytes=GiB,
                      layer_weight_bytes=tuple(layer_bytes), edge_weight_bytes=0, load_peak_extra_bytes=0,
                      max_sessions=32, max_ctx=32768, spec_window_max=64, max_tokens_per_step=2048,
                      tap_layers_held=(), tap_mode="raw", numerics_id="nid1-x", numerics_mode="invariant",
                      chunk_invariant=False, service_ms=tuple(service_ms)).to_dict()


def test_measured_footprint_from_describe_drives_the_fit():
    """design-runtime R3: MXFP4 may not stay packed on Ada (2.4x). The probe says so, placement
    believes the probe: the 4090 drops from a stage to a non-stage role."""
    packed = N("A", "4090")
    lb = [int(b * 2.4) for b in FLASH.layer_bytes[1:3]]
    fat = Node.from_dict(node("A", "4090", capability=_cap(FLASH.model, 1, 3, lb)))
    assert P.footprint_scale(fat, FLASH) == pytest.approx(2.4, rel=1e-6)
    assert classify(packed, FLASH, POL).role == "stage"
    v = classify(fat, FLASH, POL)
    assert spans_by_edge(fat)[1] == 2 and v.fit_layers < v.s_min and v.role == "whole-server"
    other = Node.from_dict(node("A", "4090", capability=_cap("some/other-model", 1, 3, lb)))
    assert P.footprint_scale(other, FLASH) == 1.0          # a probe of another model is ignored


def test_measured_service_ms_beats_the_bandwidth_estimate():
    slow = Node.from_dict(node("A", "5090", capability=_cap(
        FLASH.model, 1, 3, FLASH.layer_bytes[1:3], [(1, 1, 0, 2 * 6.0), (1, 1, 4096, 2 * 7.0), (4, 1, 0, 2 * 9.0)])))
    assert P.layer_cost_ms(slow, FLASH) == 6.0             # decode shape, smallest context, per layer
    assert classify(slow, FLASH, POL).role == "backup"     # holds a stage span, too slow to shorten chains


def test_probe_from_a_real_describe_record():
    from shard.runtime.api import StageSpec
    from shard.runtime.toy import ToyRuntime
    cap = ToyRuntime(StageSpec("toy", 0, 2), pattern="GS", window=2).describe()
    p = P.Probe.from_capability(Capability.from_dict(cap.to_dict()))    # JSON round trip, as pushed
    assert (p.model, p.lo, p.hi, len(p.layer_weight_bytes), p.step_ms, p.is_tail) == ("toy", 0, 2, 2, None, False)
    n = Node.from_dict({**node("T", "5090"), "capability": cap.to_dict()})
    assert n.probe == p and P.footprint_scale(n, FLASH) == 1.0          # another model's probe


# ── roles ─────────────────────────────────────────────────────────────────────────────────────

ROLE_TABLE_FLASH_H13 = {
    "4060-8g": "whole-server", "3060-12g": "whole-server", "4060ti-16g": "whole-server",
    "3090": "stage", "4090": "stage", "7900xtx": "whole-server", "5090": "stage", "a6000-48g": "stage",
    "pro6000-96g": "stage", "spark-128g": "stage", "mac-64g": "stage", "mac-192g": "stage",
    "mac-512g": "stage", "cpu-256g": "whole-server"}     # an unmeasured CPU holds nothing yet


def test_role_table_flash_default_cell():
    got = {k: classify(N("X", k), FLASH, POL).role for k in DEVICES}
    assert got == ROLE_TABLE_FLASH_H13
    assert set(got) == set(DEVICES)                        # every device family has a role


def test_small_cards_without_a_catalog_are_verifiers():
    for k in ("4060-8g", "3060-12g", "4060ti-16g"):
        v = classify(N("X", k), FLASH, Policy())
        assert (v.role, v.also) == ("verifier", ("seeder",))
    v = classify(N("X", "7900xtx"), FLASH, Policy())      # no MXFP4 kernel: cannot even verify
    assert (v.role, v.fit_layers) == ("seeder", 0)


def test_the_stage_bar_moves_with_the_cells_leg_latency():
    role = lambda k, leg: classify(N("X", k), FLASH, Policy(leg_ms=leg, small_models=CATALOG)).role  # noqa: E731
    assert role("4060ti-16g", 6.0) == "stage" and role("4060ti-16g", 13.0) == "whole-server"
    assert role("3090", 13.0) == "stage" and role("3090", 19.0) == "whole-server"
    assert role("5090", 19.0) == "stage"


def test_stage_min_layers_is_the_hop_law():
    # budget = (4.6 / 20 * 1000 - leg - 11.45) / 48 ; s_min = ceil(leg / (budget - t))
    assert [stage_min_layers(1.15, FLASH, Policy(leg_ms=h)) for h in (6.0, 13.0, 19.0)] == [2, 5, 7]
    assert stage_min_layers(4.3, FLASH, Policy()) is None          # t >= budget: no span length works
    assert stage_min_layers(float("inf"), FLASH, Policy()) is None


def cpu_measured(ms_per_layer):
    return Node.from_dict(node("C", "cpu-256g", capability=_cap(FLASH.model, 1, 3, FLASH.layer_bytes[1:3],
                                                                 [(1, 1, 0, 2 * ms_per_layer)])))


def test_big_but_slow_is_a_backup_holder_only_once_measured():
    v = classify(N("C", "cpu-256g"), FLASH, POL)
    assert v.role == "whole-server" and v.fit_layers == 48 and v.s_min is None   # unmeasured: no span
    slow = classify(cpu_measured(4.5), FLASH, POL)
    assert (slow.role, slow.s_min) == ("backup", None)     # holds 48 layers, too slow to be a hop
    assert classify(cpu_measured(2.0), FLASH, POL).role == "stage"   # 2 ms/layer x 48 = one-hop path


def test_device_names_are_irrelevant_only_numbers_matter():
    for k in DEVICES:
        d = node("X", k)
        anon = {**d, "device": {**d["device"], "device_name": "anon", "arch_class": "anon"}}
        assert classify(Node.from_dict(anon), FLASH, POL) == classify(Node.from_dict(d), FLASH, POL)


# ── join priority and stickiness ──────────────────────────────────────────────────────────────

def test_join_closes_holes_before_raising_k():
    c = sim_cell()
    seat(c, 1, 0, 24)
    seat(c, 2, 0, 24)
    seat(c, 3, 24, 40)                  # [40, 48) is a hole; [24, 40) sits at k = 1
    d = c.join(sim_node(9, 8), 0.0)
    assert d.span == (40, 48) and c.holes() == []


def test_join_raises_layers_below_kmin_before_capacity():
    c = sim_cell()
    for i, (lo, hi) in enumerate([(0, 24), (0, 24), (24, 48)]):
        seat(c, i, lo, hi)
    d = c.join(sim_node(9, 24), 0.0)
    assert d.span == (24, 48) and min(c.counts()) == 2


def test_join_prefers_spans_already_on_disk():
    c = sim_cell()
    for i, (lo, hi) in enumerate([(0, 24), (0, 24), (24, 48), (24, 48)]):
        seat(c, i, lo, hi)
    fresh = c.join(sim_node(8, 8), 0.0)
    n = sim_node(9, 8)
    warm = Node(**{**n.__dict__, "disk": ((SIM_PROFILE.model, 16, 24),)})
    d = c.join(warm, 0.0)
    assert fresh.download_layers == 8
    assert d.span == (16, 24) and d.download_layers == 0


def test_replicas_stack_on_the_same_boundaries():
    c = sim_cell()
    first = [c.join(sim_node(i, 6), 0.0).span for i in range(8)]
    second = [c.join(sim_node(10 + i, 6), 0.0).span for i in range(8)]
    assert sorted(first) == sorted(second) == [(6 * i, 6 * i + 6) for i in range(8)]


def test_a_returning_node_is_reseated_with_zero_download():
    c = sim_cell()
    seat(c, 1, 0, 8)
    seat(c, 2, 8, 48)
    c.persisted["n4"] = (0, 8)          # n4 held [0, 8) before; [8, 48) now sits at k = 1
    n = Node(**{**sim_node(4, 8).__dict__, "disk": ((SIM_PROFILE.model, 0, 8),)})
    d = c.join(n, 30.0)
    assert (d.reason, d.span, d.download_layers) == ("sticky", (0, 8), 0)   # sticky beats "raise k"


def test_a_returning_node_moves_to_a_hole_only_if_its_old_span_is_redundant():
    c = sim_cell()
    seat(c, 1, 0, 8)
    seat(c, 2, 0, 8)
    seat(c, 3, 8, 40)
    c.persisted["n4"] = (0, 8)                             # n4 used to hold [0, 8), now held twice
    d = c.join(sim_node(4, 8), 0.0)
    assert (d.reason, d.span) == ("hole", (40, 48))
    c2 = sim_cell()
    seat(c2, 1, 0, 8)                                      # old span held once: go home, repair the hole later
    seat(c2, 3, 8, 40)
    c2.persisted["n4"] = (0, 8)
    assert c2.join(sim_node(4, 8), 0.0).reason == "sticky"


def test_joins_and_leaves_never_move_a_healthy_node():
    rng = random.Random(7)
    kinds = sorted(DEVICES)
    c = Cell(FLASH, POL)
    live = {}
    for step in range(300):
        before = {h: x.span for h, x in c.holdings.items()}
        if live and rng.random() < 0.4:
            nid = rng.choice(sorted(live))
            del live[nid]
            c.leave(nid, float(step))
            before.pop(nid, None)
        else:
            nid = f"n{step}"
            live[nid] = True
            c.join(N(nid, rng.choice(kinds)), float(step))
        assert {h: c.holdings[h].span for h in before} == before


# ── repair ────────────────────────────────────────────────────────────────────────────────────

def _hole_cell(**kw):
    c = sim_cell(**kw)
    seat(c, 1, 0, 24)
    seat(c, 2, 0, 24)
    seat(c, 3, 24, 40)
    seat(c, 4, 24, 40)
    n5 = seat(c, 5, 40, 48)
    c.leave(n5.id, 100.0)                                  # [40, 48) becomes a hole at t = 100
    return c


def test_repair_waits_for_the_hole_timer_then_moves_a_redundant_node():
    c = _hole_cell()
    assert c.repair(699.0) == []
    (m,) = c.repair(700.0)
    assert m.reason == "hole" and m.to[0] <= 40 and m.to[1] == 48
    assert c.counts()[40:] == [0] * 8                      # make-before-break: old span still counts
    assert c.repair(701.0) == []                           # one move in flight per cell
    c.complete_move(m.id, 800.0)
    assert c.holes() == [] and min(c.counts()) >= 1


def test_repair_moves_disk_warm_nodes_first():
    c = _hole_cell()
    warm = Node(**{**sim_node(2, 24).__dict__, "disk": ((SIM_PROFILE.model, 24, 48),)})
    c.holdings["n2"].node = warm
    (m,) = c.repair(700.0)
    assert m.id == "n2" and m.download_layers == 0 and m.to == (24, 48)


def test_repair_never_takes_a_node_off_a_layer_it_alone_holds():
    c = sim_cell()
    seat(c, 1, 0, 24)
    seat(c, 2, 24, 40)                                     # nobody is held twice: no legal mover
    c._refresh(0.0)
    assert c.repair(10_000.0) == []


def test_repair_raises_k_after_the_kmin_timer_and_a_rejoin_preempts_it():
    c = sim_cell()
    for i, (lo, hi) in enumerate([(0, 24), (0, 24), (0, 24), (24, 48), (24, 48)]):
        seat(c, i, lo, hi)
    c.leave("n4", 0.0)                                     # [24, 48) at k = 1
    assert c.repair(1799.0) == []
    (m,) = c.repair(1800.0)
    assert m.reason == "k_min" and m.to[0] >= 24 and m.id in ("n0", "n1", "n2")
    c.complete_move(m.id, 1900.0)
    assert min(c.counts()) == 2                            # nobody fell below k_min to pay for it
    c2 = sim_cell()
    for i, (lo, hi) in enumerate([(0, 24), (0, 24), (0, 24), (24, 48), (24, 48)]):
        seat(c2, i, lo, hi)
    c2.leave("n4", 0.0)
    assert c2.join(sim_node(4, 24), 900.0).span == (24, 48)   # the disk-warm rejoin comes first
    assert c2.repair(1800.0) == []


def test_repair_is_damped():
    c = _hole_cell(hole_repair_s=0.0)
    (m,) = c.repair(700.0)
    assert m.id in ("n3", "n4")                            # the 16-layer movers download less
    c.complete_move(m.id, 700.0)
    other = "n4" if m.id == "n3" else "n3"
    c.leave(other, 750.0)                                  # a second hole opens at once
    assert c.holes() and c.repair(1299.0) == []            # < move_interval_s since the last move
    (m2,) = c.repair(1300.0)
    c.complete_move(m2.id, 1300.0)
    assert c.holes() == []


def test_no_moves_without_a_deficit():
    c = sim_cell()
    for i in range(12):
        c.join(sim_node(i, 8), 0.0)
    assert min(c.counts()) >= 2 and c.repair(1e9) == []


# ── mixed fleets ──────────────────────────────────────────────────────────────────────────────

def _fleet(kinds, pol=POL):
    c = Cell(FLASH, pol)
    for nid, k in kinds:
        c.join(N(nid, k), 0.0)
    return c


def _views(c):
    return [{"id": h, "span": list(x.span), "role": x.verdict.role, "layer_ms": x.verdict.layer_ms,
             "subnet": x.node.subnet, "reliability": x.node.reliability, "rho": 0.0, "sessions": 0,
             "kv_free": 10 ** 9, "uplink_mbps": x.node.uplink_mbps}
            for h, x in sorted(c.holdings.items()) if x.verdict.layer_ms != float("inf")]


def test_spark_5090_3090_is_three_layers_short_and_one_more_24gb_card_completes_it():
    c = _fleet([("Spark", "spark-128g"), ("R5090", "5090"), ("R3090", "3090")])
    assert c.holes() == [45, 46, 47]                       # 31 + 8 + 6 = 45 of 48 layers
    assert candidate_paths(_views(c), FLASH, POL) == []
    c.join(N("R3090b", "3090"), 1.0)
    (p, *_) = candidate_paths(_views(c), FLASH, POL)
    assert [h["id"] for h in p["hops"]] == ["Spark", "R5090", "R3090", "R3090b"]
    assert FLASH.g / p["step_ms"] * 1000 >= POL.bar_tok_s


def test_spark_plus_a_96gb_card_is_a_two_hop_path():
    c = _fleet([("Spark", "spark-128g"), ("Pro", "pro6000-96g")])
    (p,) = candidate_paths(_views(c), FLASH, POL)
    assert [(h["id"], h["role"]) for h in p["hops"]] == [("Spark", "head"), ("Pro", "tail")]


def test_96gb_and_24gb_mix_every_card_holds_and_fat_nodes_shorten_the_chain():
    c = _fleet([("P1", "pro6000-96g"), ("P2", "pro6000-96g")] + [(f"G{i}", "3090") for i in range(4)])
    assert all(x.verdict.role == "stage" for x in c.holdings.values()) and len(c.holdings) == 6
    paths = candidate_paths(_views(c), FLASH, POL)
    assert len(paths[0]["hops"]) == 2 and len(paths) > 1  # 24 GB cards add alternative chains


def test_fat_nodes_carry_sessions_even_when_only_small_chains_are_mirror_backed():
    """2 x 96 GB + 9 x 24 GB: the 24 GB chain is mirror-backed by the fat cards, the fat 2-hop path
    only by replay over the small cards. Mirror-first ranking would park the fat cards as backups
    at 23 tok/s; the 1.3x slack keeps them carrying sessions at 43 tok/s."""
    c = _fleet([("P1", "pro6000-96g"), ("P2", "pro6000-96g")] + [(f"G{i}", "3090") for i in range(9)])
    paths = candidate_paths(_views(c), FLASH, POL, k=40)
    assert [h["id"] for h in paths[0]["hops"]] == ["P1", "P2"] and paths[0]["failover"] == "replay"
    chain = [p for p in paths if p["failover"] == "mirror"]
    assert chain and all(p["rank_ms"] > POL.backed_slack * paths[0]["rank_ms"] for p in chain)


def test_a_fleet_of_only_24gb_cards_serves_backed_paths():
    c = _fleet([(f"G{i}", "3090" if i % 2 else "4090") for i in range(18)])
    (p, *_) = candidate_paths(_views(c), FLASH, POL)
    assert p["failover"] == "mirror" and len(p["hops"]) == 9      # the tail card holds 5, not 6
    assert FLASH.g / p["step_ms"] * 1000 >= POL.bar_tok_s


def test_14_5090s_reproduce_the_hop_law_6_hop_path():
    c = _fleet([(f"F{i}", "5090") for i in range(14)])
    (p, *_) = candidate_paths(_views(c), FLASH, POL)
    assert len(p["hops"]) == 6 and p["failover"] == "mirror"
    assert FLASH.g / p["step_ms"] * 1000 == pytest.approx(29.2, abs=0.05)   # design-mesh §1, h = 13


def test_every_device_class_gets_a_role_in_one_mixed_cell():
    c = _fleet([(f"X{i}", k) for i, k in enumerate(sorted(DEVICES))])
    c.join(cpu_measured(4.5), 1.0)                         # a measured-slow CPU box: a backup holder
    roles = {x.node.id + x.node.device.device_name: x.verdict.role for x in c.holdings.values()}
    roles.update({k: v.role for k, v in c.others.items()})
    assert len(roles) == len(DEVICES) + 1 and all(r in P.ROLES for r in roles.values())
    assert {r for r in roles.values()} >= {"stage", "backup", "whole-server"}


def test_a_device_holding_the_whole_model_is_a_solo_path():
    c = _fleet([("M", "mac-192g")])
    (p,) = candidate_paths(_views(c), FLASH, POL)
    assert p["hops"] == [{"id": "M", "lo": 0, "hi": 48, "role": "solo", "backup": None}]


# ── paths ─────────────────────────────────────────────────────────────────────────────────────

def test_backups_are_never_on_their_path_and_hold_the_whole_hop():
    rng = random.Random(3)
    kinds = [k for k, v in ROLE_TABLE_FLASH_H13.items() if v == "stage"]
    checked = backed = 0
    for trial in range(15):
        c = _fleet([(f"n{i}", rng.choice(kinds)) for i in range(rng.randint(3, 16))])
        spans = {h: x.span for h, x in c.holdings.items()}
        for p in candidate_paths(_views(c), FLASH, POL, k=5):
            checked += 1
            backed += sum(h["backup"] is not None for h in p["hops"])
            ids = [h["id"] for h in p["hops"]]
            assert len(set(ids)) == len(ids)
            for h in p["hops"]:
                if h["backup"] is not None:
                    assert h["backup"] not in ids
                    lo, hi = spans[h["backup"]]
                    assert lo <= h["lo"] and h["hi"] <= hi
            assert p["hops"][0]["lo"] == 0 and p["hops"][-1]["hi"] == 48
            assert all(a["hi"] == b["lo"] for a, b in zip(p["hops"], p["hops"][1:]))
    assert checked >= 30 and backed >= 30


def test_subnets_on_a_path_are_distinct():
    c = _fleet([("A", "pro6000-96g"), ("B", "pro6000-96g"), ("C", "pro6000-96g"), ("D", "pro6000-96g")])
    v = _views(c)
    together = [p for p in candidate_paths(v, FLASH, POL, k=10) if {"A", "B"} <= {h["id"] for h in p["hops"]}]
    assert together                                        # A and B do form a path on distinct subnets
    for x in v:
        x["subnet"] = "same" if x["id"] in ("A", "B") else x["id"]
    paths = candidate_paths(v, FLASH, POL, k=10)
    assert paths and all({"A", "B"} - {h["id"] for h in p["hops"]} for p in paths)


def test_load_and_admission_filters_steer_paths():
    c = _fleet([(f"F{i}", "5090") for i in range(12)])
    v = _views(c)
    best = candidate_paths(v, FLASH, POL)[0]
    head = best["hops"][0]["id"]
    for field, val in (("rho", 0.85), ("sessions", 16), ("kv_free", 100)):
        vv = [dict(x, **{field: val}) if x["id"] == head else x for x in v]
        paths = candidate_paths(vv, FLASH, POL, kv_tokens=4096)
        assert paths and all(head not in [h["id"] for h in p["hops"]] for p in paths)
        if field == "kv_free":
            assert all(h["backup"] != head for p in paths for h in p["hops"])
    vv = [dict(x, rho=0.5) if x["id"] == head else x for x in v]
    assert candidate_paths(vv, FLASH, POL)[0]["hops"][0]["id"] != head   # q(rho) prices the queue


def test_backup_role_holders_only_carry_a_path_when_nothing_else_can():
    cpu = {"id": "C", "span": [0, 48], "role": "backup", "layer_ms": 4.0, "subnet": "C", "reliability": 1.0,
           "rho": 0.0, "sessions": 0, "kv_free": 10 ** 9, "uplink_mbps": 100.0}
    (p,) = candidate_paths([cpu], FLASH, POL)
    assert p["degraded"] and p["hops"][0]["id"] == "C"
    mac = dict(cpu, id="M", role="stage", layer_ms=1.15, subnet="M")
    (p2, *_) = candidate_paths([cpu, mac], FLASH, POL)
    assert not p2["degraded"] and p2["hops"] == [{"id": "M", "lo": 0, "hi": 48, "role": "solo", "backup": "C"}]


def test_prefill_ranks_a_slow_uplink_down():
    c = _fleet([("A", "pro6000-96g"), ("B", "pro6000-96g"), ("C", "pro6000-96g"), ("D", "pro6000-96g")])
    v = _views(c)
    decode_best = candidate_paths(v, FLASH, POL)[0]
    slow = decode_best["hops"][0]["id"]
    vv = [dict(x, uplink_mbps=5.0) if x["id"] == slow else x for x in v]
    p = candidate_paths(vv, FLASH, POL, prompt_tokens=16384)[0]
    assert p["prefill_ms"] > 0 and p["hops"][0]["id"] != slow


def test_measured_rtt_overrides_the_cell_leg():
    c = _fleet([("A", "pro6000-96g"), ("B", "pro6000-96g"), ("C", "pro6000-96g"), ("D", "pro6000-96g")])
    v = _views(c)
    p0 = candidate_paths(v, FLASH, POL)[0]
    a, b = p0["hops"][0]["id"], p0["hops"][1]["id"]
    p1 = candidate_paths(v, FLASH, POL, rtt={a: {b: 90.0}})[0]
    assert (p1["hops"][0]["id"], p1["hops"][1]["id"]) != (a, b)


# ── adversarial-review regressions ────────────────────────────────────────────────────────────

def test_completing_a_move_never_opens_a_hole():
    c = _hole_cell(hole_repair_s=0.0)
    (m,) = c.repair(700.0)
    assert m.frm == (24, 40) and m.to[1] == 48
    other = "n4" if m.id == "n3" else "n3"
    c.leave(other, 750.0)                                  # the other holder of [24, 40) dies mid-download
    assert c.complete_move(m.id, 800.0) is False           # cancelled: the mover keeps [24, 40) warm
    assert c.holdings[m.id].span == (24, 40) and c.holdings[m.id].target is None
    assert all(v >= 1 for v in c.counts()[:40])


@pytest.mark.parametrize("leg", [16.0, 18.0])
def test_24gb_cells_hold_the_tail_even_when_it_is_one_layer_under_s_min(leg):
    pol = Policy(leg_ms=leg, small_models=CATALOG)
    c = _fleet([(f"G{i}", "3090") for i in range(16)], pol)
    assert classify(N("G", "3090"), FLASH, pol).s_min == 6 and spans_by_edge(N("G", "3090"), pol=pol)[2] == 5
    assert c.holes() == []
    (p, *_) = candidate_paths(_views(c), FLASH, pol)
    assert p["hops"][-1]["hi"] == 48


def _cap_tail(lo, hi, step_ms):
    d = _cap(FLASH.model, lo, hi, FLASH.layer_bytes[lo:hi], [(1, 1, 0, step_ms)])
    d["is_tail"] = hi == 48
    return d


def test_a_tail_probe_does_not_charge_the_logits_to_the_layers():
    ms = 6 * 1.15 + FLASH.logits_ms                        # six layers + lm_head, as warmup measures a tail
    tail = Node.from_dict(node("T", "5090", capability=_cap_tail(42, 48, ms)))
    assert P.layer_cost_ms(tail, FLASH) == pytest.approx(1.15)
    assert classify(tail, FLASH, POL).role == "stage"


def test_the_footprint_comes_from_moe_sized_probe_layers():
    dense_only = Node.from_dict(node("A", "4090", capability=_cap(FLASH.model, 0, 1, [FLASH.layer_bytes[0]])))
    assert P.footprint_scale(dense_only, FLASH) == 1.0    # layer 0 says nothing about MoE packing
    mixed = [FLASH.layer_bytes[0], int(FLASH.layer_bytes[1] * 2.4)]
    both = Node.from_dict(node("A", "4090", capability=_cap(FLASH.model, 0, 2, mixed)))
    assert P.footprint_scale(both, FLASH) == pytest.approx(2.4, rel=1e-6)   # not diluted by layer 0


def test_a_zero_or_negative_measurement_falls_back_to_the_estimate():
    n = Node.from_dict(node("A", "5090", capability=_cap(FLASH.model, 1, 3, FLASH.layer_bytes[1:3],
                                                          [(1, 1, 0, 0.0)])))
    assert P.layer_cost_ms(n, FLASH) == P.layer_cost_ms(N("A", "5090"), FLASH)
    Cell(FLASH, POL).join(n, 0.0)                          # no ZeroDivisionError


def test_a_slow_holder_hands_over_where_a_faster_span_starts():
    x = {"id": "X", "span": [0, 40], "role": "stage", "layer_ms": 3.0, "subnet": "X", "reliability": 1.0,
         "rho": 0.0, "sessions": 0, "kv_free": 10 ** 6, "uplink_mbps": 100.0}
    y = dict(x, id="Y", span=[20, 48], layer_ms=1.15, subnet="Y")
    (p, *_) = candidate_paths([x, y], FLASH, POL)
    assert [(h["id"], h["lo"], h["hi"]) for h in p["hops"]] == [("X", 0, 20), ("Y", 20, 48)]
    assert p["step_ms"] == pytest.approx(3 * 13.0 + 20 * 3.0 + 28 * 1.15 + FLASH.tail_ms)


def test_pruning_cannot_starve_the_subnet_filter():
    base = {"role": "stage", "layer_ms": 1.15, "reliability": 1.0, "rho": 0.0, "sessions": 0,
            "kv_free": 10 ** 6, "uplink_mbps": 100.0}
    cheap = [dict(base, id=f"C{i}", span=[0, 20], subnet="S") for i in range(13)]
    pricey = dict(base, id="Q", span=[0, 20], subnet="T", layer_ms=2.0)
    mid = dict(base, id="G", span=[20, 40], subnet="U")
    tail = dict(base, id="Tl", span=[40, 48], subnet="S")
    for k in (1, 3):
        (p, *_) = candidate_paths(cheap + [pricey, mid, tail], FLASH, POL, k=k)
        assert [h["id"] for h in p["hops"]] == ["Q", "G", "Tl"]


def test_an_unreachable_rtt_is_skipped_not_a_crash():
    c = _fleet([("A", "pro6000-96g"), ("B", "pro6000-96g"), ("C", "pro6000-96g")])
    v = _views(c)
    ps = candidate_paths(v, FLASH, POL, rtt={"A": {"B": float("inf")}, "B": {"A": -1.0}}, k=10)
    assert ps and all((a["id"], b["id"]) not in (("A", "B"), ("B", "A"))
                      for p in ps for a, b in zip(p["hops"], p["hops"][1:]))


def test_hole_repair_with_k_min_1_never_bounces_the_only_holder():
    c = sim_cell(k_min=1, hole_repair_s=0.0, move_interval_s=0.0)
    seat(c, 1, 0, 24)
    seat(c, 2, 24, 40)                                     # [40, 48) a hole; everyone is alone on its span
    assert c.repair(10.0) == []


@pytest.mark.parametrize("k", [1, 2, 3, 4, 8])
def test_co_located_heads_never_starve_the_only_valid_path(k):
    """Skeptic re-check: 3 cheap heads share the tail's subnet S, 1 pricier head R does not, and
    8 middles sit between. A label cap dropped every R label for k in 2..4."""
    base = {"role": "stage", "layer_ms": 1.15, "reliability": 1.0, "rho": 0.0, "sessions": 0,
            "kv_free": 10 ** 6, "uplink_mbps": 100.0}
    hs = [dict(base, id=f"S{i}", span=[0, 10], subnet="S") for i in range(3)]
    hs += [dict(base, id="R", span=[0, 10], subnet="T", layer_ms=2.0)]
    hs += [dict(base, id=f"Q{i}", span=[10, 20], subnet=f"u{i}") for i in range(8)]
    hs += [dict(base, id="G", span=[20, 40], subnet="V"), dict(base, id="Tl", span=[40, 48], subnet="S")]
    paths = candidate_paths(hs, FLASH, POL, k=k)
    assert len(paths) == min(k, 8) and all([h["id"] for h in p["hops"]][0] == "R" for p in paths)
