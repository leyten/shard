"""plan_ring — turn a set of running fleet boxes into a scheduler-planned ring spec.

Today the libp2p demo is hand-tuned: the operator picks `--stages A,B,C` (a guessed
order) and hand-computes `--layers 18,9,9` (a guessed VRAM-fit). This reads each box's
REAL vram (nvidia-smi) + the measured RTT mesh and asks shard/scheduler.py for the
fit + min-latency ring, then prints the exact launch_libp2p.py invocation.

Non-invasive: the working launcher is untouched. This just computes the args the
operator currently guesses. Run it, paste the printed command.

  python3 phase0/plan_ring.py --ids 42330411,42330410,42330412 --model /root/models/gpt-oss-120b

  -> prints:
     [vram]   42330411  48.0 GB
     ...
     [rtt mesh] ...
     [plan] coordinator=42330411  ring: 42330411 -> 42330410 -> 42330412
     [plan] layers: 40,19,19  (covers [0:78])
     RUN:
       python3 phase0/launch_libp2p.py --stages 42330411,42330410,42330412 --layers 40,19,19 ...

Boundary law: pure control-plane. vram + rtt + layers in, ring spec out. No accounts/pay.
"""
import argparse, sys, os, concurrent.futures as cf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))            # phase0/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (shard pkg)
from scheduler_svc import plan                                            # local import, no HTTP
from shard import registry as reg                                        # M1: single signed registry
# NB: launch_oss / launch_swarm read ~/.shard_psk at import and pull vastai — they're the
# FLEET side. Import them lazily (inside the ssh-touching functions) so the pure planning
# logic (parse_vram_gb, build_nodes, plan_fleet-with-stub) stays importable + testable
# offline with no PSK and no vast creds.

# M1: layer counts (and bytes/layer, quant, engine path) now come from ONE signed registry
# (shard/registry.py -> registry/models.json), read by BOTH repos. The old MODEL_LAYERS dict
# lived here AND in c0mpute types.ts AND in getLayerCountForModel — three copies that drifted
# (gpt-oss 120-vs-36). Deleted. --total-layers stays an explicit override only.
# Resolution order: --total-layers > registry by model id > registry by enginePath basename.
DEFAULT_GB_PER_LAYER = 1.05
DEFAULT_KV_GB_PER_LAYER = 0.04


def _registry_lookup(model_arg: str):
    """Find a registry row for the --model arg, matching either its `id` or the basename of
    its `enginePath` (the CLI takes a path like /root/models/gpt-oss-120b OR an id). Returns
    (layerCount, gbPerLayer, kvGbPerLayer) or (None, None, None) if not found/registry absent.
    Fail-soft: a missing/invalid registry just means the caller must pass --total-layers."""
    try:
        registry = reg.load_registry(expected_pubkey=os.environ.get("SHARD_MODELS_PUBKEY") or None)
    except reg.RegistryError:
        return None, None, None
    base = model_arg.lower().rstrip("/").split("/")[-1]
    for m in registry.get("models", []):
        eng_base = str(m.get("enginePath", "")).lower().rstrip("/").split("/")[-1]
        if model_arg == m["id"] or base == m["id"] or base == eng_base:
            return m["layerCount"], m.get("gbPerLayer"), m.get("kvGbPerLayer")
    return None, None, None


def parse_vram_gb(nvidia_smi_out: str) -> float:
    """parse `nvidia-smi --query-gpu=memory.total --format=csv,noheader` -> GB (float).

    handles 'MiB'/'MB' lines, multi-GPU boxes (sum), blank/garbage lines (skipped).
    """
    total_mib = 0.0
    for ln in nvidia_smi_out.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        # e.g. "49140 MiB"
        parts = ln.split()
        try:
            val = float(parts[0])
        except (ValueError, IndexError):
            continue
        unit = (parts[1].lower() if len(parts) > 1 else "mib")
        if unit.startswith("gi") or unit.startswith("gb"):
            total_mib += val * 1024
        else:                                                              # MiB/MB default
            total_mib += val
    return round(total_mib / 1024, 1)


def query_vram(insts: list) -> dict:
    """node_id -> total vram GB, queried in parallel over ssh."""
    def one(inst):
        from launch_oss import rssh                                        # lazy: fleet-only
        r = rssh(inst, "nvidia-smi --query-gpu=memory.total --format=csv,noheader", 30)
        return str(inst["id"]), parse_vram_gb(r.stdout)
    with cf.ThreadPoolExecutor(max_workers=max(1, len(insts))) as ex:
        return dict(ex.map(one, insts))


def build_nodes(insts: list, vram: dict, rtt_matrix: list) -> list:
    """assemble the scheduler_svc /plan node list from vram + the NxN rtt mesh.

    rtt_matrix[i][j] = ms from insts[i] to insts[j] (launch_swarm.mesh_rtt order).
    """
    ids = [str(i["id"]) for i in insts]
    nodes = []
    for i, nid in enumerate(ids):
        rtt_ms = {ids[j]: rtt_matrix[i][j] for j in range(len(ids)) if j != i}
        nodes.append({"node_id": nid, "vram_gb": vram[nid], "rtt_ms": rtt_ms})
    return nodes


def plan_fleet(insts: list, model: str, total_layers: int, gb_per_layer: float,
               kv_gb_per_layer: float, coordinator: str | None, rtt_matrix=None) -> dict:
    """full pipeline: vram + rtt -> scheduler plan. rtt_matrix optional (else probe live)."""
    vram = query_vram(insts)
    if rtt_matrix is None:
        import launch_swarm                                                # lazy: fleet-only
        rtt_matrix = launch_swarm.mesh_rtt(insts)
    nodes = build_nodes(insts, vram, rtt_matrix)
    req = {"model": model, "total_layers": total_layers, "gb_per_layer": gb_per_layer,
           "kv_gb_per_layer": kv_gb_per_layer, "nodes": nodes}
    if coordinator:
        req["coordinator"] = str(coordinator)
    p = plan(req)
    p["_vram"] = vram
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="comma instance ids of the running boxes")
    ap.add_argument("--model", default="/root/models/gpt-oss-120b")
    ap.add_argument("--total-layers", type=int, default=None,
                   help="model layer count override (default: resolved from the signed registry)")
    ap.add_argument("--gb-per-layer", type=float, default=None,
                   help="model bytes/layer override (default: registry, else %.2f)" % DEFAULT_GB_PER_LAYER)
    ap.add_argument("--kv-gb-per-layer", type=float, default=None,
                   help="KV bytes/layer override (default: registry, else %.2f)" % DEFAULT_KV_GB_PER_LAYER)
    ap.add_argument("--coordinator", default="", help="pin coordinator instance id (else lowest-mean-rtt)")
    a = ap.parse_args()

    # M1: resolve layer count + fit bytes from the ONE signed registry, with explicit CLI
    # overrides winning. No more local MODEL_LAYERS copy to drift against c0mpute.
    reg_layers, reg_gb, reg_kv = _registry_lookup(a.model)
    total_layers = a.total_layers if a.total_layers is not None else reg_layers
    if total_layers is None:
        print(f"error: cannot determine layer count for model '{a.model}'. "
              f"Add it to the signed registry (registry/models.json) or pass --total-layers.",
              flush=True)
        sys.exit(1)
    gb_per_layer = (a.gb_per_layer if a.gb_per_layer is not None
                    else (reg_gb if reg_gb is not None else DEFAULT_GB_PER_LAYER))
    kv_gb_per_layer = (a.kv_gb_per_layer if a.kv_gb_per_layer is not None
                       else (reg_kv if reg_kv is not None else DEFAULT_KV_GB_PER_LAYER))

    ids = [int(x) for x in a.ids.split(",") if x.strip()]
    from launch_oss import instances                                       # lazy: fleet-only
    import launch_swarm                                                    # lazy: fleet-only
    allinsts = instances()
    insts = [allinsts[i] for i in ids]

    print("[vram] querying ...", flush=True)
    vram = query_vram(insts)
    for i in insts:
        print(f"  {i['id']}  {vram[str(i['id'])]:5.1f} GB  ({i.get('geolocation')})", flush=True)

    print("[rtt] probing mesh ...", flush=True)
    rtt_matrix = launch_swarm.mesh_rtt(insts)

    p = plan_fleet(insts, a.model, total_layers, gb_per_layer, kv_gb_per_layer,
                   a.coordinator or None, rtt_matrix=rtt_matrix)

    ring = p["ring_order"]
    layers = ",".join(str(s["n_layers"]) for s in p["stages"])
    print(f"[plan] coordinator={p['coordinator']}", flush=True)
    print(f"[plan] ring: {' -> '.join(ring)}", flush=True)
    print(f"[plan] layers: {layers}  (covers [0:{total_layers}])", flush=True)
    for s in p["stages"]:
        print(f"         stage{s['stage']} {s['node_id']}  [{s['lo']}:{s['hi']}]  {s['n_layers']}L", flush=True)
    print("\nRUN:", flush=True)
    print(f"  python3 phase0/launch_libp2p.py --stages {','.join(ring)} --layers {layers} \\", flush=True)
    print(f"      --model {a.model} --max-ctx 16384 --prompt-file /root/ft_prompt.txt \\", flush=True)
    print(f"      --K 4 --depth 2 --max-new 64 --receipts", flush=True)


if __name__ == "__main__":
    main()
