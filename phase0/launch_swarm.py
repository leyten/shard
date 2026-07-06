"""Parallel launcher for the GLM-5.2 NVFP4 scattered-5090 swarm.

Flow (run from the local orchestrator; needs vastai + ~/.ssh/vast_c0mpute + shard/topology.py):
  1. provision   — rent N offers (or reuse --use-running ids), poll to running, collect conn info
  2. bootstrap   — parallel: scp code, pip install vmoe env (idempotent), per node
  3. mesh        — temp echo listeners, measure all-pairs node->node RTT over the open internet
  4. topology    — shard/topology.optimal_loop on the measured RTT -> latency-optimal node order
  5. assign      — split 78 layers into contiguous blocks IN topology order; pick coord node
  6. download    — parallel: each node node_fetch.py its layer block; coord also --coord
  7. launch      — parallel: setsid stages chained via --next in topology order (last = tail)
  8. generate    — coord -> chain -> tok/s; emit proof receipt
  9. (teardown is manual: vastai stop/destroy)

Usage:
  python launch_swarm.py --use-running 41324652,41378912,... --coord-id 41324652 --prompt "..." --max-new 24
  python launch_swarm.py --rent 18 --prompt "..."        # auto-select + rent cuda>=13.2 rel>0.98
"""
import os, sys, json, time, subprocess, argparse, concurrent.futures as cf

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
KEY = os.path.expanduser("~/.ssh/vast_c0mpute")
IMAGE = "vastai/base-image:cuda-13.2.1-auto"
SSHO = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=25", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=20"]
# S1: layer count is model-specific, not a global 78 (that was GLM-5.2). gpt-oss-120B
# has 36 layers (gateway.py, ROADMAP.md, PROOF.md all agree). The CLI --total-layers
# overrides; this default matches the default --model (gpt-oss-120b).
NLAYERS, NDENSE, PER = 36, 3, 4   # 36 layers (gpt-oss-120B), first 3 dense, 4 NVFP4 layers/node
STAGE_PORT = 29600

def sh(cmd, timeout=120):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr

def vast(args, timeout=120):
    rc, out, err = sh(["vastai"] + args, timeout)
    return out + err

def vast_json(args, timeout=120):
    rc, out, err = sh(["vastai"] + args + ["--raw"], timeout)   # stdout ONLY (stderr warnings corrupt JSON)
    try: return json.loads(out)
    except Exception:
        # tolerate a leading non-JSON line (deprecation notice) before the JSON body
        for cut in ("[", "{"):
            k = out.find(cut)
            if k >= 0:
                try: return json.loads(out[k:])
                except Exception: pass
        return None

def instances():
    d = vast_json(["show", "instances-v1"]) or []
    return d if isinstance(d, list) else d.get("instances", [])

def _ep(inst):
    """direct SSH endpoint (public_ipaddr + mapped-22) — the --direct path; falls back to proxy."""
    port22 = (inst.get("ports") or {}).get("22/tcp")
    if port22 and inst.get("public_ipaddr"):
        return inst["public_ipaddr"], port22[0]["HostPort"]
    return inst["ssh_host"], inst["ssh_port"]

def node_ssh(inst):
    host, port = _ep(inst)
    return ["ssh", "-i", KEY, "-p", str(port)] + SSHO + [f"root@{host}"]

def node_scp(inst, files, dest="/root/"):
    host, port = _ep(inst)
    return ["scp", "-i", KEY, "-P", str(port)] + SSHO + files + [f"root@{host}:{dest}"]

def rssh(inst, remote_cmd, timeout=120):
    return sh(node_ssh(inst) + [remote_cmd], timeout)

def mapped_port(inst, cport=STAGE_PORT):
    p = (inst.get("ports") or {}).get(f"{cport}/tcp")
    return p[0]["HostPort"] if p else None

# ---------- 1. provision ----------
def wait_running(ids, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        insts = {i["id"]: i for i in instances() if i["id"] in ids}
        if len(insts) == len(ids) and all(i.get("actual_status") == "running" for i in insts.values()):
            # need ports mapped too
            if all(mapped_port(i) for i in insts.values()):
                return insts
        time.sleep(10)
    raise RuntimeError(f"nodes not all running+mapped in {timeout}s")

# ---------- 2. bootstrap ----------
def bootstrap(inst):
    files = [f"{HERE}/requirements_vmoe.txt", f"{HERE}/node_fetch.py",
             f"{REPO}/research/glm_swarm_nvfp4.py", f"{REPO}/research/glm_swarm_nvfp4_kv.py"]
    sh(node_scp(inst, files), 180)
    # idempotent: build venv only if missing; flashinfer compiles at first stage warmup (MAX_JOBS capped)
    cmd = ("test -x /root/vmoe/bin/python || (python3 -m venv /root/vmoe && "
           "/root/vmoe/bin/pip install --no-input -q -U pip && "
           "/root/vmoe/bin/pip install --no-input -q -r /root/requirements_vmoe.txt); "
           "/root/vmoe/bin/python -c 'import vllm,torch;print(\"OK\",torch.cuda.is_available())'")
    rc, out, err = rssh(inst, cmd, 1200)
    return inst["id"], ("OK True" in out), (out + err)[-300:]

# ---------- 3. mesh ----------
def mesh_rtt(insts_ordered):
    """temp echo listener on each node's STAGE_PORT, each node times TCP connects to all others."""
    targets = [(i["public_ipaddr"], mapped_port(i)) for i in insts_ordered]
    # start echo listeners
    echo = ("setsid python3 -c \"import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); "
            "s.bind(('0.0.0.0',%d)); s.listen(64); [c.close() for c in iter(lambda: s.accept()[0], None)]\" "
            "</dev/null >/dev/null 2>&1 &") % STAGE_PORT
    for i in insts_ordered:
        rssh(i, "fuser -k %d/tcp 2>/dev/null; sleep 1; " % STAGE_PORT + echo, 30)
    time.sleep(3)
    tgt_json = json.dumps(targets)
    probe = ("python3 - <<'PY'\nimport socket,time,json\nT=json.loads('''%s''')\nrow=[]\n"
             "for ip,port in T:\n  best=9999.0\n  for _ in range(3):\n    try:\n      t=time.time();s=socket.create_connection((ip,int(port)),timeout=5);s.close();best=min(best,(time.time()-t)*1000)\n    except Exception:best=min(best,9999.0)\n  row.append(round(best,1))\nprint('RTTROW',json.dumps(row))\nPY") % tgt_json
    M = []
    def probe_node(i):
        rc, out, err = rssh(i, probe, 120)
        for ln in out.splitlines():
            if ln.startswith("RTTROW"): return json.loads(ln[6:])
        return [9999.0] * len(targets)
    with cf.ThreadPoolExecutor(max_workers=len(insts_ordered)) as ex:
        M = list(ex.map(probe_node, insts_ordered))
    for i in insts_ordered:
        rssh(i, "fuser -k %d/tcp 2>/dev/null" % STAGE_PORT, 20)
    return M

# ---------- 4. topology ----------
def solve_order(rtt, coord_idx):
    """rtt = full NxN node->node ms. Returns ordered STAGE node-indices for the cheapest
    coord -> head -> ... -> tail -> coord loop (coord fixed as depot)."""
    from shard.topology import optimal_loop
    n = len(rtt)
    L = [[rtt[a][b] for b in range(n)] for a in range(n)]
    for a in range(n): L[a][a] = 0.0
    stages = [i for i in range(n) if i != coord_idx]
    c_out = [L[coord_idx][i] for i in range(n)]
    c_in = [L[i][coord_idx] for i in range(n)]
    return optimal_loop(stages, L, c_out, c_in)   # (order: stage node-indices, cost ms)

# ---------- 5. assign ----------
def assign_layers(order, nodes, total=NLAYERS):
    """order = ordered stage node-indices. Returns chain: list of (inst, [layers]) in chain order."""
    nstages = len(order)
    blocks = []; cur = 0
    base = total // nstages; rem = total % nstages
    for s in range(nstages):
        cnt = base + (1 if s < rem else 0)
        blocks.append(list(range(cur, cur + cnt))); cur += cnt
    return [(nodes[order[s]], blocks[s]) for s in range(nstages)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-running", default="")
    ap.add_argument("--coord-id", type=int, default=0)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--total-layers", type=int, default=NLAYERS)
    a = ap.parse_args()

    ids = [int(x) for x in a.use_running.split(",") if x.strip()]
    print(f"[1] using running nodes: {ids}", flush=True)
    insts = wait_running(ids)
    nodes = [insts[i] for i in ids]
    for n in nodes:
        print(f"    {n['id']} {n.get('geolocation')} ip={n['public_ipaddr']} :{mapped_port(n)}", flush=True)

    print("[2] bootstrap (parallel)...", flush=True)
    with cf.ThreadPoolExecutor(max_workers=len(nodes)) as ex:
        for nid, ok, tail in ex.map(bootstrap, nodes):
            print(f"    {nid}: {'OK' if ok else 'FAIL '+tail}", flush=True)

    print("[3] mesh RTT (node->node)...", flush=True)
    rtt = mesh_rtt(nodes)
    for r in rtt: print("    " + " ".join(f"{x:6.1f}" for x in r), flush=True)

    coord_idx = next((j for j, n in enumerate(nodes) if n["id"] == a.coord_id), 0)
    coord = nodes[coord_idx]
    print("[4] topology...", flush=True)
    order, cost = solve_order(rtt, coord_idx)
    print(f"    optimal chain cost={cost:.1f}ms; stage order (node ids): {[nodes[i]['id'] for i in order]}", flush=True)

    chain = assign_layers(order, nodes, a.total_layers)
    print(f"[5] coord={coord['id']} ({coord.get('geolocation')})", flush=True)
    for inst, blk in chain:
        print(f"    stage {inst['id']} ({inst.get('geolocation')}): layers {blk[0]}-{blk[-1]}", flush=True)

    print("[6] download layers (parallel)...", flush=True)
    def dl(item):
        inst, blk = item
        cmd = f". /root/vmoe/bin/activate && cd /root && python node_fetch.py --layers {' '.join(map(str,blk))}"
        rc, out, err = rssh(inst, cmd, 1200)
        return inst["id"], "NODE_FETCH_DONE" in out
    jobs = list(chain) + [(coord, None)]
    def dl_all(item):
        inst, blk = item
        if blk is None:
            cmd = ". /root/vmoe/bin/activate && cd /root && python node_fetch.py --coord"
        else:
            cmd = f". /root/vmoe/bin/activate && cd /root && python node_fetch.py --layers {' '.join(map(str,blk))}"
        rc, out, err = rssh(inst, cmd, 1800)
        return inst["id"], "NODE_FETCH_DONE" in out, (out+err)[-200:]
    with cf.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        for nid, ok, tail in ex.map(dl_all, jobs):
            print(f"    {nid}: {'downloaded' if ok else 'FAIL '+tail}", flush=True)

    print("[7] launch stages chained (parallel; compiles+warms)...", flush=True)
    # chain order: stage0 -> stage1 -> ... -> stageK (tail). coord connects to stage0.
    eps = [(inst, f"{inst['public_ipaddr']}:{mapped_port(inst)}") for inst, _ in chain]
    def launch(idx):
        inst, blk = chain[idx]
        nxt = eps[idx+1][1] if idx+1 < len(chain) else None
        nextarg = f" --next {nxt}" if nxt else ""
        # NOTE: kill via nvidia-smi only — NEVER pkill -f glm_swarm (matches this very command -> self-kill).
        # No MAX_JOBS / no compile: VLLM_CUTLASS is precompiled (moe_backend=cutlass in the driver).
        cmd = (f"nvidia-smi --query-compute-apps=pid --format=csv,noheader|xargs -r kill -9 2>/dev/null; fuser -k {STAGE_PORT}/tcp 2>/dev/null; sleep 2; "
               f"rm -f /root/stage.log; cd /root && . /root/vmoe/bin/activate && "
               f"setsid bash -c 'python glm_swarm_nvfp4_kv.py stage --layers {' '.join(map(str,blk))} --port {STAGE_PORT}{nextarg} > /root/stage.log 2>&1' </dev/null >/dev/null 2>&1 &")
        rssh(inst, cmd, 60)
        return inst["id"]
    for idx in range(len(chain)): launch(idx)
    # barrier: wait all stages 'listening'
    def warm(inst):
        for _ in range(60):
            rc, out, err = rssh(inst, "grep -c listening /root/stage.log 2>/dev/null; grep -ciE 'exit status|Traceback' /root/stage.log 2>/dev/null", 30)
            nums = [x for x in out.split() if x.isdigit()]
            if nums and nums[0] != "0": return inst["id"], True
            if len(nums) > 1 and nums[1] != "0": return inst["id"], False
            time.sleep(15)
        return inst["id"], False
    with cf.ThreadPoolExecutor(max_workers=len(chain)) as ex:
        warmed = list(ex.map(warm, [c[0] for c in chain]))
    print("    warmed:", warmed, flush=True)
    if not all(ok for _, ok in warmed):
        print("    !! not all stages warm; aborting", flush=True); return

    print("[8] coord -> generate -> tok/s...", flush=True)
    s0 = eps[0][1]
    ccmd = (f"cd /root && . /root/vmoe/bin/activate && "
            f"python glm_swarm_nvfp4_kv.py coord --stage {s0} --prompt \"{a.prompt}\" --max-new {a.max_new} 2>&1 | "
            f"grep -viE 'INFO |WARNING|modelopt|FutureWarning|warnings.warn'")
    rc, out, err = rssh(coord, ccmd, 1800)
    print(out[-1500:], flush=True)
    print("\n[done] tear down with: vastai stop/destroy instance <id>", flush=True)

if __name__ == "__main__":
    main()
