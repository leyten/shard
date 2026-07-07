"""Deployable-ring planner — the seam c0mpute's control plane calls to place a sharded swarm.

`select_ring` (topology.py) is the pure decision core: clean numbers in, a ring out. But turning what
a node ANNOUNCES (GPU, free VRAM, a CPU probe, its subnet) plus a measured RTT mesh into those clean
numbers takes engine calibration — VRAM reserves, the launch-bound per-layer time, and head placement.
That calibration lived inline in `scratchpad/ring_up.py` (throwaway SSH glue). `plan_ring` lifts it into
a tracked, tested function so the network layer can PLACE a swarm with one call instead of re-deriving
the engine's memory/compute model.

Boundary (docs/INTEGRATION.md): c0mpute owns MEASUREMENT (collecting `nodes` + `rtt`); shard owns the
ENGINE model (reserves, per-layer ms, the select_ring decision). Deps point one way: c0mpute -> shard.
`python3 -m shard.plan` reads `{nodes, rtt, model}` JSON on stdin and prints the plan on stdout, so a
TypeScript orchestrator drives the same proven planner as `ring_up` without porting its subtleties.
"""
import json
import sys

from .topology import select_ring

# M2.5-on-5090 anchors (docs/M25_ENGINE.md, mirrored from ring_up.py) — the DEFAULT model profile.
# The caller (c0mpute's model catalog) SHOULD pass an explicit `model` profile; these keep the seam
# runnable for M2.5 without one.
M25_PROFILE = {
    "n_layers": 62,
    "layer_vram_mb": 1700.0,     # NVFP4 experts + bf16 attn, per decoder layer
    "kv_mb_per_layer": 150.0,    # at the 40960 KV cap
    "layer_ms_base": 0.65,       # per-layer decode compute on an idle fast-CPU 5090 box
    "reserve_mb": 1500.0,        # CUDA context + allocator slack per box
    "head_reserve_mb": 4096.0,   # coordinator process on the head: embed + EAGLE head + its context
    "cap_layers": 13,            # proven warm per-box ceiling (16/box is OOM-adjacent, unproven)
    "head_layer_ms_mult": 1.3,   # the head box also runs the coordinator
}

_UNREACHABLE = 9000.0            # RTT sentinel: treat >= this as "no usable path" when ranking centrality


def plan_ring(nodes, rtt, model=None, *, slack=None):
    """Place a deployable sharded ring from announced capabilities + a measured RTT mesh.

    nodes: [{"id": <hashable>, "free_vram_mb": float, "subnet": str,
             "cpu_factor": float=1.0,          # >=1; pyloop/0.10 + load — a slow/loaded box drafts slower
             "up_mbps": float|None}]           # optional; present on ALL nodes -> upload-aware placement
    rtt:   NxN one-way ms matrix, row/col order aligned to `nodes` (rtt[i][i] ignored).
    model: profile dict (see M25_PROFILE); defaults to M2.5.
    slack: select_ring pool headroom; defaults to len(nodes) (let it drop any weak/co-located box).

    Returns a plan dict, or None if the pool genuinely can't hold the model:
      {"order":  [node_id, ...],                       # head-first, deployable
       "head":   node_id,
       "stages": [{"id", "index", "lo", "hi", "head", "tail", "layers"}...],
       "dropped":[node_id, ...],
       "roles":  {node_id: role},                      # only when every node carries up_mbps
       "step_ms", "tok_s_per_g", "k",
       "request_ms", "prefill_ms"}                     # only when upload-aware
    """
    m = {**M25_PROFILE, **(model or {})}
    n = len(nodes)
    if n == 0:
        return None
    ids = [nd["id"] for nd in nodes]
    layer_vram, kv = float(m["layer_vram_mb"]), float(m["kv_mb_per_layer"])
    cap_layers = int(m["cap_layers"])
    per_layer_mb = layer_vram + kv

    # 1) calibrate free VRAM: strip the per-box reserve, then cap at the proven per-box layer ceiling.
    free = {i: min(max(nodes[i]["free_vram_mb"] - float(m["reserve_mb"]), 0.0), cap_layers * per_layer_mb)
            for i in range(n)}
    cap_ok = [i for i in range(n) if free[i] >= per_layer_mb]
    if not cap_ok:
        return None                                          # no node can hold even one layer

    # 2) head = most central capable node (lowest total RTT to the rest); it runs the coordinator.
    def centrality(i):
        return sum(rtt[i][j] for j in range(n) if j != i and rtt[i][j] < _UNREACHABLE)
    head = min(cap_ok, key=centrality)
    free[head] = max(free[head] - float(m["head_reserve_mb"]), 0.0)

    # 3) launch-bound per-layer time: base * the node's cpu_factor; the head pays a coordinator penalty.
    layer_ms = {i: float(m["layer_ms_base"]) * float(nodes[i].get("cpu_factor", 1.0)) for i in range(n)}
    layer_ms[head] *= float(m["head_layer_ms_mult"])

    # 4) coordinator entry/return hops are measured relative to the chosen head.
    c_out = [rtt[head][i] if i != head else 1.0 for i in range(n)]
    c_in = [rtt[i][head] if i != head else 1.0 for i in range(n)]
    subnet = {i: nodes[i]["subnet"] for i in range(n)}

    # 5) upload-aware placement iff EVERY node announced an uplink (residential lever); else decode-only.
    ups = [nodes[i].get("up_mbps") for i in range(n)]
    aware = all(u is not None for u in ups)
    extra = {}
    if aware:
        extra = {"up_mbps": {i: float(ups[i]) for i in range(n)},
                 "prefill_bytes": float(m.get("prefill_bytes", 0.0)),
                 "decode_bytes": float(m.get("decode_bytes", 0.0)),
                 "decode_steps": int(m.get("decode_steps", 1)),
                 "prefill_chunks": int(m.get("prefill_chunks", 1))}

    spec = select_ring(range(n), rtt, c_out, c_in, free_vram_mb=free, layer_ms=layer_ms,
                       subnet=subnet, n_layers=int(m["n_layers"]), layer_vram_mb=layer_vram,
                       kv_mb_per_layer=kv, slack=n if slack is None else int(slack),
                       require=head, **extra)
    if spec is None:
        return None
    assert spec["order"][0] == head, "select_ring must return a head-first (deployable) order"

    order = [ids[i] for i in spec["order"]]
    last = len(spec["order"]) - 1
    stages = []
    for k, i in enumerate(spec["order"]):
        lo, hi = spec["blocks"][i]
        stages.append({"id": ids[i], "index": k, "lo": lo, "hi": hi,
                       "head": k == 0, "tail": k == last, "layers": hi - lo})
    out = {
        "order": order,
        "head": ids[head],
        "stages": stages,
        "dropped": [ids[i] for i in spec["dropped"]],
        "step_ms": spec["step_ms"],
        "tok_s_per_g": spec["tok_s_per_g"],
        "k": spec["k"],
    }
    if aware:
        out["request_ms"] = spec.get("request_ms")
        out["prefill_ms"] = spec.get("prefill_ms")
        out["roles"] = {ids[int(i)]: r for i, r in spec.get("roles", {}).items()}
    return out


def _main() -> int:
    """`python3 -m shard.plan` — JSON in ({nodes, rtt, model?, slack?}), JSON out (the plan, or null)."""
    try:
        req = json.load(sys.stdin)
    except Exception as e:  # noqa: BLE001 — a malformed request is a caller error, report it as JSON
        json.dump({"error": f"bad request json: {e}"}, sys.stdout)
        return 2
    try:
        plan = plan_ring(req["nodes"], req["rtt"], req.get("model"), slack=req.get("slack"))
    except KeyError as e:
        json.dump({"error": f"missing field: {e}"}, sys.stdout)
        return 2
    except Exception as e:  # noqa: BLE001
        json.dump({"error": f"plan failed: {e}"}, sys.stdout)
        return 1
    json.dump(plan, sys.stdout)          # `null` when the pool can't hold the model — a valid answer
    return 0


if __name__ == "__main__":
    sys.exit(_main())
