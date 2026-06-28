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
from scheduler_svc import plan                                            # local import, no HTTP
# NB: launch_oss / launch_swarm read ~/.shard_psk at import and pull vastai — they're the
# FLEET side. Import them lazily (inside the ssh-touching functions) so the pure planning
# logic (parse_vram_gb, build_nodes, plan_fleet-with-stub) stays importable + testable
# offline with no PSK and no vast creds.

# gpt-oss-120B: 78 transformer layers (matches launch_swarm.NLAYERS).
DEFAULT_TOTAL_LAYERS = 78
# NVFP4 120B: ~ model bytes per layer. Conservative so the fit never over-commits a box.
# (tune per quant; the scheduler subtracts headroom+boundary on top of this.)
DEFAULT_GB_PER_LAYER = 1.05
DEFAULT_KV_GB_PER_LAYER = 0.04


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
    ap.add_argument("--total-layers", type=int, default=DEFAULT_TOTAL_LAYERS)
    ap.add_argument("--gb-per-layer", type=float, default=DEFAULT_GB_PER_LAYER)
    ap.add_argument("--kv-gb-per-layer", type=float, default=DEFAULT_KV_GB_PER_LAYER)
    ap.add_argument("--coordinator", default="", help="pin coordinator instance id (else lowest-mean-rtt)")
    a = ap.parse_args()

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

    p = plan_fleet(insts, a.model, a.total_layers, a.gb_per_layer, a.kv_gb_per_layer,
                   a.coordinator or None, rtt_matrix=rtt_matrix)

    ring = p["ring_order"]
    layers = ",".join(str(s["n_layers"]) for s in p["stages"])
    print(f"[plan] coordinator={p['coordinator']}", flush=True)
    print(f"[plan] ring: {' -> '.join(ring)}", flush=True)
    print(f"[plan] layers: {layers}  (covers [0:{a.total_layers}])", flush=True)
    for s in p["stages"]:
        print(f"         stage{s['stage']} {s['node_id']}  [{s['lo']}:{s['hi']}]  {s['n_layers']}L", flush=True)
    print("\nRUN:", flush=True)
    print(f"  python3 phase0/launch_libp2p.py --stages {','.join(ring)} --layers {layers} \\", flush=True)
    print(f"      --model {a.model} --max-ctx 16384 --prompt-file /root/ft_prompt.txt \\", flush=True)
    print(f"      --K 4 --depth 2 --max-new 64 --receipts", flush=True)


if __name__ == "__main__":
    main()
