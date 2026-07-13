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
    "layer_vram_mb": 2330.0,     # NVFP4 experts + bf16 attn + norms, per decoder layer — MEASURED
                                 # 2026-07-09 (capability probe, one real layer resident: 2329.5 MB;
                                 # a warm 13-layer stage read 31.5/32.6 GiB — the old 1700 estimate
                                 # under-modeled by ~35% and packed stages one allocation from OOM)
    "kv_mb_per_layer": 150.0,    # at the 40960 KV cap (B=1; batched callers scale by B*maxlen/40960)
    "layer_ms_base": 0.65,       # per-layer decode compute on an idle fast-CPU 5090 box
    "reserve_mb": 1500.0,        # CUDA context + allocator slack per box
    "head_reserve_mb": 4096.0,   # coordinator process on the head: embed + EAGLE head + its context
    "tail_reserve_mb": 1400.0,   # tail stage: final norm + lm_head (measured 1.15 GiB bf16 + slack —
                                 # a 13-layer tail OOM'd on it live while 13-layer middles warmed fine)
    "cap_layers": 12,            # 32 GB ceiling by MEASURED footprint ((32768-1500)/2480 = 12.6);
                                 # 13 ran warm but at 31.5/32.6 GiB — brim-riding, not a plan target
    "head_layer_ms_mult": 1.3,   # the head box also runs the coordinator
}

_UNREACHABLE = 9000.0            # RTT sentinel: treat >= this as "no usable path" when ranking centrality

_PROVEN_CAP_VRAM_MB = 32768.0    # the card size cap_layers was proven on; bigger cards scale by density


def density_cap_layers(cap_layers, total_vram_mb):
    """The proven layer DENSITY scaled to the card size — a flat cap collapsed a 96 GB card
    to the 32 GB verdict (the spec's core distinction). ONE rule, shared with probe.derive_layers."""
    return int(int(cap_layers) * float(total_vram_mb) / _PROVEN_CAP_VRAM_MB)


def plan_ring(nodes, rtt, model=None, *, slack=None, privacy=None):
    """Place a deployable sharded ring from announced capabilities + a measured RTT mesh.

    nodes: [{"id": <hashable>, "free_vram_mb": float, "subnet": str,
             "cpu_factor": float=1.0,          # >=1; pyloop/0.10 + load — a slow/loaded box drafts slower
             "up_mbps": float|None,            # optional; present on ALL nodes -> upload-aware placement
             "trusted": bool=False,            # ASSIGNED by the control plane (stake/reputation), never
                                               # self-reported by the node
             # per-node MEASURED capability (the probe's cap vector; every field optional — absent
             # fields fall back to the model profile, so a homogeneous pool plans byte-identically):
             "layer_vram_mb": float|None,      # this node's per-layer footprint (arch/backend-specific:
                                               # cutlass ~2330, marlin ~4060 — a hetero ring needs both)
             "cap_layers": int|None,           # probe-verdict layer ceiling for THIS card (wins outright)
             "total_vram_mb": float|None,      # else: density-scale the proven cap to the card size
             "load_peak_extra_mb": float|None, # measured load/run transient above resident (peak gate)
             "layer_ms": float|None}]          # measured decode ms/layer (overrides base*cpu_factor)
    rtt:   NxN one-way ms matrix, row/col order aligned to `nodes` (rtt[i][i] ignored).
    model: profile dict (see M25_PROFILE); defaults to M2.5.
    slack: select_ring pool headroom; defaults to len(nodes) (let it drop any weak/co-located box).
    privacy: {"boundary_in": int, "boundary_out": int} — turn on BOUNDARY-LAYER PINNING: the ring's
             head/tail (they handle raw prompt / output tokens) and every stage holding a boundary
             layer must be `trusted` nodes; strangers hold only deep-middle layers. The head is the
             most central TRUSTED capable node under pinning (it runs the coordinator, which sees
             the raw prompt). None (default) = placement exactly as before.

    Returns a plan dict, or None if the pool genuinely can't hold the model (with pinning: can't
    hold it SAFELY — e.g. no trusted node for an end):
      {"order":  [node_id, ...],                       # head-first, deployable
       "head":   node_id,
       "stages": [{"id", "index", "lo", "hi", "head", "tail", "layers",
                   "boundary"}...],                    # "boundary" only when privacy pinning is on
       "dropped":[node_id, ...],
       "roles":  {node_id: role},                      # only when every node carries up_mbps
       "step_ms", "tok_s_per_g", "k",
       "request_ms", "prefill_ms",                     # only when upload-aware
       "privacy": {"boundary_in", "boundary_out", "boundary_stages"}}   # only when pinning is on
    """
    m = {**M25_PROFILE, **(model or {})}
    n = len(nodes)
    if n == 0:
        return None
    ids = [nd["id"] for nd in nodes]
    if len(set(ids)) != len(ids):                            # duplicate ids collide in the output maps
        raise ValueError("duplicate node id in `nodes`")     # (order/roles/boundary_stages) -> mis-deploy
    layer_vram, kv = float(m["layer_vram_mb"]), float(m["kv_mb_per_layer"])
    cap_layers = int(m["cap_layers"])

    # 1) calibrate free VRAM per node: strip the per-box reserve AND the node's measured load-peak
    #    transient (the admit-then-OOM gate), then cap at the proven layer ceiling — density-scaled
    #    to the card size when the node announces one (a flat cap collapsed a 96 GB card to the
    #    32 GB verdict). The footprint is per-NODE too: a marlin card (~4.1 GB/layer) and a cutlass
    #    card (~2.3 GB) hold very different blocks, and select_ring already takes the per-node dict.
    lv = {i: float(nodes[i].get("layer_vram_mb") or layer_vram) for i in range(n)}
    per_layer = {i: lv[i] + kv for i in range(n)}

    def _node_cap(i):
        if nodes[i].get("cap_layers") is not None:
            return int(nodes[i]["cap_layers"])                    # probe-verdict ceiling wins outright
        total = float(nodes[i].get("total_vram_mb") or 0.0)
        return density_cap_layers(cap_layers, total) if total > 0 else cap_layers

    free = {i: min(max(nodes[i]["free_vram_mb"] - float(m["reserve_mb"])
                       - float(nodes[i].get("load_peak_extra_mb") or 0.0), 0.0),
                   _node_cap(i) * per_layer[i])
            for i in range(n)}
    cap_ok = [i for i in range(n) if free[i] >= per_layer[i]]
    if not cap_ok:
        return None                                          # no node can hold even one layer

    # 2) head = most central capable node (lowest total RTT to the rest); it runs the coordinator.
    #    Under privacy pinning the coordinator sees the raw prompt, so the head must be TRUSTED —
    #    rank centrality over trusted capable nodes only.
    pin = privacy is not None
    # STRICT bool — trust is the security boundary, so read it fail-CLOSED: only a genuine `True`
    # (JSON `true`) marks a node trusted. A truthy string like "false"/"0" or an int must NOT sneak a
    # node into the trust set (a control plane that serialized the flag as a string would otherwise
    # fail OPEN — the one way a stranger could reach a boundary while the plan claims to be pinned).
    trusted = {i for i in range(n) if nodes[i].get("trusted") is True} if pin else None
    head_pool = [i for i in cap_ok if i in trusted] if pin else cap_ok
    if not head_pool:
        return None                                          # pinning on, but no trusted node can hold a block

    def centrality(i):
        # clamp each edge at the sentinel instead of OMITTING unreachable ones — omission summed a
        # fully-disconnected node to 0, which won min() and made it the mandatory head (undeployable)
        return sum(min(float(rtt[i][j]), _UNREACHABLE) for j in range(n) if j != i)

    def _connected_cap(i):
        # layers reachable from i: its own budget + every capable peer with a finite path BOTH ways
        return int(free[i] // per_layer[i]) + sum(
            int(free[j] // per_layer[j]) for j in cap_ok
            if j != i and rtt[i][j] < _UNREACHABLE and rtt[j][i] < _UNREACHABLE)
    head_pool = [i for i in head_pool if _connected_cap(i) >= int(m["n_layers"])]
    if not head_pool:
        return None                              # no candidate head can REACH enough capacity to serve
    head = min(head_pool, key=centrality)
    free[head] = max(free[head] - float(m["head_reserve_mb"]), 0.0)

    # 3) launch-bound per-layer time: base * the node's cpu_factor; the head pays a coordinator
    #    penalty. A node announcing a MEASURED layer_ms (the probe's graph-replayed decode number)
    #    is placed at that, not the modeled base — a box whose graph capture failed runs eager at
    #    ~4x and must be planned as what it measured, not what its GPU label suggests.
    layer_ms = {i: (float(nodes[i]["layer_ms"]) if nodes[i].get("layer_ms") is not None
                    else float(m["layer_ms_base"]) * float(nodes[i].get("cpu_factor", 1.0)))
                for i in range(n)}
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

    if pin:
        extra["trusted"] = trusted
        extra["boundary_in"] = int(privacy.get("boundary_in", 0))
        extra["boundary_out"] = int(privacy.get("boundary_out", 0))

    # 6) the TAIL stage also holds the final norm + lm_head (measured 1.15 GiB bf16 on
    #    M2.5 — a 13-layer tail OOM'd loading it on a 32 GB 5090, live 2026-07-09, while
    #    the same 13 layers warmed fine as a middle). The reserve applies to WHICHEVER
    #    node lands the tail, which select_ring decides — so plan, check the landed
    #    tail's block against the reserve, and if it doesn't fit, bake the reserve into
    #    that node's budget EXACTLY ONCE and re-plan (the tail may move). Convergence is
    #    checked against the ORIGINAL budget: the old loop compared against the already-
    #    docked value — re-demanding the reserve on top of itself — so a node that
    #    reappeared as tail was docked again each round (a feasible pool read as
    #    infeasible), and after 4 blind rounds the LAST spec was returned even when its
    #    tail never fit at all. The docked set is finite and only grows, so this
    #    converges in <= n rounds or honestly reports None.
    tail_reserve = float(m.get("tail_reserve_mb", 0.0))
    base_free = dict(free)                       # budgets to validate against (head reserve included)
    docked = set()                               # nodes whose budget already models the tail reserve
    spec = None
    for _ in range(n + 1):
        spec = select_ring(range(n), rtt, c_out, c_in, free_vram_mb=free, layer_ms=layer_ms,
                           subnet=subnet, n_layers=int(m["n_layers"]), layer_vram_mb=lv,
                           kv_mb_per_layer=kv, slack=n if slack is None else int(slack),
                           require=head, **extra)
        if spec is None:
            return None
        tail_i = spec["order"][-1]
        lo, hi = spec["blocks"][tail_i]
        if tail_reserve == 0.0 or base_free[tail_i] >= (hi - lo) * per_layer[tail_i] + tail_reserve:
            break                                # the landed tail fits block + reserve in its budget
        if tail_i in docked:
            return None                          # reserve already modeled and it STILL can't fit
        docked.add(tail_i)
        free[tail_i] = max(base_free[tail_i] - tail_reserve, 0.0)
    else:
        return None                              # no tail placement converged: the reserve fits nowhere
    assert spec["order"][0] == head, "select_ring must return a head-first (deployable) order"
    # a deployable ring never traverses an unreachable (sentinel) edge: the forward hops and the
    # tail -> head coordinator return must all be measured, finite paths — if feasibility forced
    # one in, there IS no usable ring, so say so instead of shipping a dead hop
    _o = spec["order"]
    if (any(rtt[a][b] >= _UNREACHABLE for a, b in zip(_o, _o[1:]))
            or (_o[-1] != head and rtt[_o[-1]][head] >= _UNREACHABLE)):
        return None
    # belt-and-braces: every stage's block must fit the node's ORIGINAL budget (the tail
    # including its reserve) — a violation here is a planner bug, never a deployable answer
    for i in spec["order"]:
        lo, hi = spec["blocks"][i]
        need = (hi - lo) * per_layer[i] + (tail_reserve if i == spec["order"][-1] else 0.0)
        if need > base_free[i] + 1e-6:
            raise RuntimeError(f"planned block [{lo}:{hi}) needs {need:.0f} MB on node {ids[i]!r} "
                               f"whose budget is {base_free[i]:.0f} MB")

    boundary = set(spec.get("boundary", []))
    order = [ids[i] for i in spec["order"]]
    last = len(spec["order"]) - 1
    stages = []
    for k, i in enumerate(spec["order"]):
        lo, hi = spec["blocks"][i]
        st = {"id": ids[i], "index": k, "lo": lo, "hi": hi,
              "head": k == 0, "tail": k == last, "layers": hi - lo}
        if pin:
            st["boundary"] = i in boundary
        stages.append(st)
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
    if pin:
        out["privacy"] = {"boundary_in": extra["boundary_in"], "boundary_out": extra["boundary_out"],
                          "boundary_stages": [ids[i] for i in spec["order"] if i in boundary]}
    return out


def _main() -> int:
    """`python3 -m shard.plan` — JSON in ({nodes, rtt, model?, slack?}), JSON out (the plan, or null)."""
    try:
        req = json.load(sys.stdin)
    except Exception as e:  # noqa: BLE001 — a malformed request is a caller error, report it as JSON
        json.dump({"error": f"bad request json: {e}"}, sys.stdout)
        return 2
    try:
        plan = plan_ring(req["nodes"], req["rtt"], req.get("model"), slack=req.get("slack"),
                         privacy=req.get("privacy"))
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
