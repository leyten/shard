"""Churn replay of design-mesh's placement_sim.py through the real shard.mesh.placement code.

scratchpad/v2/placement_sim.py (untracked design sim) models one Flash cell: Poisson arrivals with
a VRAM-class mix, exponential lifetimes, one contiguous span per node sized by class (EST
3.6 GB/layer), and compares placement policies. `run()` below consumes the random stream in
exactly the sim's order (class draw, lifetime, next arrival), so the arrivals and deaths are
identical and only the placement decisions differ: they come from Cell.join / Cell.repair.

The profile and policy make the real code see the sim's world: uniform 3.6e9-byte layers, no KV,
no edge tensors, no reserve, every class a stage (leg_ms 0), repairs instant (timers 0) and applied
at once (the sim's documented caveat: no download delay).
"""
import math
import random

from shard.mesh.placement import Cell, ModelProfile, Node, Policy
from shard.runtime.api import DeviceInfo

L = 48
LAYER = 3_600_000_000
CLASSES = [("16GB", 3, 0.22), ("24GB", 5, 0.33), ("32GB", 8, 0.28), ("48GB", 12, 0.07),
           ("96GB", 25, 0.05), ("Mac128", 34, 0.03), ("Mac256", 48, 0.02)]

SIM_PROFILE = ModelProfile(
    model="sim/flash-48", n_layers=L, layer_bytes=(LAYER,) * L, layer_kinds=("swa",) * L,
    swa_window=128, kv_bytes_per_pos=(0, 0), head_bytes=0, tail_bytes=0, shared_edge_bytes=0,
    reserve_bytes=0, weight_formats=("bf16",), layer_ms=1.0, active_bytes_per_layer=0, tail_ms=0.0, logits_ms=0.0,
    g=4.6, lane=4, wire_bytes_per_pos=0)

# placement_sim.out (3 seeds, 20 days, 8 h node life): (cov>=1, cov>=2) per (mean nodes, policy).
SIM_OUT = {(8, "sticky"): (0.428, 0.070), (8, "repair"): (0.591, 0.081),
           (12, "sticky"): (0.688, 0.173), (12, "repair"): (0.859, 0.170),
           (20, "sticky"): (0.924, 0.576), (20, "repair"): (0.974, 0.522),
           (40, "sticky"): (0.997, 0.987), (40, "repair"): (0.997, 0.987)}


def _pick_class(rng):
    x, acc = rng.random(), 0.0
    for name, n, w in CLASSES:
        acc += w
        if x <= acc:
            return name, n
    return CLASSES[-1][0], CLASSES[-1][1]


def _node(nid: int, layers: int) -> Node:
    dev = DeviceInfo(backend="cuda", device_name="sim", arch_class="sim", host_arch="x86_64", n_devices=1,
                     total_mem_bytes=layers * LAYER, free_mem_bytes=layers * LAYER, unified_memory=False,
                     mem_bw_bytes_s=1e12, mem_bw_source="spec", formats_hw=("bf16",), formats_emulated=())
    return Node(id=f"n{nid}", cell="sim", subnet=f"s{nid}", device=dev)


def _min_hops(spans):
    spans = sorted(spans)
    pos, hops, i = 0, 0, 0
    while pos < L:
        reach = pos
        while i < len(spans) and spans[i][0] <= pos:
            reach = max(reach, spans[i][1])
            i += 1
        if reach == pos:
            return None
        pos, hops = reach, hops + 1
    return hops


def run(policy: str, mean_nodes: int, *, life_h: float = 8.0, days: int = 20, seed: int = 1,
        kmin_repair: bool = False) -> dict:
    """policy "sticky" (never move) or "repair" (Cell.repair after every event). kmin_repair also
    enables the k < k_min repair the sim never modelled (ARCHITECTURE C22)."""
    pol = Policy(k_min=2, leg_ms=0.0, plan_sessions=0, unified_host_bytes=0, hole_repair_s=0.0,
                 kmin_repair_s=0.0 if kmin_repair else math.inf, move_interval_s=0.0)
    cell = Cell(SIM_PROFILE, pol)
    rng = random.Random(seed)
    arr_rate = mean_nodes / life_h
    t, horizon = 0.0, days * 24.0
    dies, nid, moves = {}, 0, 0
    acc = {"cov1": 0.0, "cov2": 0.0, "hops": 0.0, "hops_t": 0.0}
    node_hours = 0.0
    next_arr = rng.expovariate(arr_rate)
    while t < horizon:
        next_dep = min(dies.values(), default=float("inf"))
        t_next = min(next_arr, next_dep, horizon)
        dt = t_next - t
        h = cell.counts()
        if min(h) >= 1:
            acc["cov1"] += dt
        if min(h) >= 2:
            acc["cov2"] += dt
        mh = _min_hops([x.span for x in cell.holdings.values()])
        if mh is not None:
            acc["hops"] += mh * dt
            acc["hops_t"] += dt
        node_hours += len(cell.holdings) * dt
        t = t_next
        if t >= horizon:
            break
        now = t * 3600.0
        if next_arr <= next_dep:
            _, span = _pick_class(rng)
            cell.join(_node(nid, span), now)
            dies[f"n{nid}"] = t + rng.expovariate(1.0 / life_h)
            nid += 1
            next_arr = t + rng.expovariate(arr_rate)
        else:
            dead = min(dies, key=dies.get)
            del dies[dead]
            cell.leave(dead, now)
        if policy == "repair" and cell.holdings:
            for m in cell.repair(now):
                cell.complete_move(m.id, now)
                moves += 1
    return {"cov>=1": acc["cov1"] / horizon, "cov>=2": acc["cov2"] / horizon,
            "hops": acc["hops"] / acc["hops_t"] if acc["hops_t"] else float("nan"),
            "moves/node/day": moves / (node_hours / 24.0) if node_hours else 0.0}


def averaged(policy: str, mean_nodes: int, seeds=(1, 2, 3), **kw) -> dict:
    rs = [run(policy, mean_nodes, seed=s, **kw) for s in seeds]
    return {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]}
