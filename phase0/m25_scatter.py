"""Launch the MiniMax-M2.5 ring across SCATTERED vast boxes over the libp2p sidecar (per-node
keys, NO PSK, NAT traversal) and run the coordinator. Relay-back ring (no return-tunnel).

Reuses the proven launch_libp2p sidecar pattern (fuser-not-pkill self-match footgun, setsid
detach, 'tunnel up|listening' health grep). Each node: sidecar (29600 libp2p, mapped public) +
m25_ring stage (29610 engine-in, 29611 -> sidecar forward to successor). Coord on the head box
dials the head engine locally; the tail's hidden relays back up the ring to it.

  python m25_scatter.py --order CA:42512149:0:10 WA:42512151:10:23 MA:...:23:36 ... --max-new 64 \
      --prompt-file /root/copy_prompt.txt
"""
import re, sys, json, time, shlex, subprocess, argparse

KEY = "/root/.ssh/vast_c0mpute"
SSHO = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=25", "-i", KEY]
LIBP2P, ENG_IN, FWD = 29600, 29610, 29611
REPO = "/root/.openclaw/workspace/shard"


def push_code(host, port):
    for src, dst in [("phase0/m25_ring.py", "/root/m25_ring.py"), ("phase0/m25_stage.py", "/root/m25_stage.py"),
                     ("phase0/ngram_draft.py", "/root/ngram_draft.py"), ("shard/transport.py", "/root/transport.py")]:
        subprocess.run(["scp", *SSHO, "-P", str(port), f"{REPO}/{src}", f"root@{host}:{dst}"], capture_output=True, text=True)


def vinst(iid):
    return json.loads(subprocess.check_output(["vastai", "show", "instance", str(iid), "--raw"], text=True))


def ssh_ep(j):
    return j["ssh_host"], int(j["ssh_port"])


def pub_ep(j):
    ip = (j.get("public_ipaddr") or "").strip()
    ports = j.get("ports") or {}
    m = ports.get(f"{LIBP2P}/tcp") or ports.get(f"{LIBP2P}/tcp ")
    hp = m[0]["HostPort"] if m else None
    return ip, hp


def sh(host, port, cmd, timeout=120):
    return subprocess.run(["ssh", *SSHO, "-p", str(port), f"root@{host}", cmd],
                          capture_output=True, text=True, timeout=timeout)


# A PeerId is REMOTE stdout interpolated into root shell commands on other boxes — validate
# strict base58btc before it can touch a command string (shlex-quoted again at point of use).
PEERID_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{40,64}$")


def check_peerid(pid):
    if not PEERID_RE.fullmatch(pid or ""):
        raise ValueError(f"invalid PeerId from node (not base58btc): {pid!r}")
    return pid


def peerid(host, port):
    r = sh(host, port, "/tmp/sidecar -key /root/node.key -prove ping 2>/dev/null | grep PEERID")
    for ln in r.stdout.splitlines():
        if ln.startswith("PEERID "):
            return check_peerid(ln.split()[1])
    raise RuntimeError(f"no PeerId from {host}:{port}: {r.stdout[-200:]} {r.stderr[-200:]}")


def sidecar_cmd(announce, inbound, forward):
    """Pure builder (unit-testable): remote-influenced values (multiaddrs carry PeerIds from remote
    stdout) are shlex-quoted, and the inner command is quoted ONCE for the bash -c level."""
    inb = f"-inbound {shlex.quote(inbound)}" if inbound else ""
    fw = f"-forward {shlex.quote(forward)}" if forward else ""
    inner = (f"/tmp/sidecar -key /root/node.key -listen /ip4/0.0.0.0/tcp/{LIBP2P} "
             f"-announce {shlex.quote(announce)} {inb} {fw} > /root/sidecar.log 2>&1")
    # free ALL sidecar ports (29600 listen + 29611/29612 forward listeners) — stale sidecars from a
    # prior run holding 29611 make the new forward listener fail to bind, silently breaking the ring.
    return (f"pkill -9 -x sidecar 2>/dev/null; fuser -k {LIBP2P}/tcp {FWD}/tcp 29612/tcp 2>/dev/null; sleep 2; rm -f /root/sidecar.log; "
            f"setsid bash -c {shlex.quote(inner)} </dev/null >/dev/null 2>&1 &")


def launch_sidecar(host, port, announce, inbound, forward):
    cmd = sidecar_cmd(announce, inbound, forward)
    for attempt in range(5):
        sh(host, port, cmd, 30)
        for _ in range(4):
            time.sleep(3)
            # require BOTH the tunnel up AND no forward-bind failure (the bug that broke the ring)
            r = sh(host, port, "grep -cE 'tunnel up|listening' /root/sidecar.log 2>/dev/null || echo 0", 20)
            last = (r.stdout.strip().splitlines() or ["0"])[-1].strip()
            bad = sh(host, port, "grep -c 'address already in use' /root/sidecar.log 2>/dev/null || echo 0", 20)
            badn = (bad.stdout.strip().splitlines() or ["0"])[-1].strip()
            if last not in ("", "0") and badn in ("", "0"):
                return True
        print(f"  sidecar {host} attempt {attempt+1} not up; relaunch", flush=True)
    return False


def launch_stage(host, port, lo, hi, is_tail):
    nxt = "" if is_tail else f"--next 127.0.0.1:{FWD}"
    cmd = (f"nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; "
           f"fuser -k {ENG_IN}/tcp 2>/dev/null; sleep 4; rm -f /root/stage.log; cd /root && "
           f"CUDA_VISIBLE_DEVICES=0 M25_DIR=/root/m25 setsid bash -c "
           f"'/root/venv/bin/python /root/m25_ring.py stage --lo {lo} --hi {hi} --port {ENG_IN} {nxt} "
           f"> /root/stage.log 2>&1' </dev/null >/dev/null 2>&1 &")
    try:
        sh(host, port, cmd, 25)            # fire-and-forget; the setsid stage keeps running, warm() confirms
    except subprocess.TimeoutExpired:
        pass


def warm(host, port, label, tries=70):
    for _ in range(tries):
        time.sleep(8)
        r = sh(host, port, "grep -c WARM /root/stage.log 2>/dev/null || echo 0", 20)
        if (r.stdout.strip().splitlines() or ["0"])[-1].strip() not in ("", "0"):
            return True
        r2 = sh(host, port, "grep -cE 'Traceback|Error|CUDA out' /root/stage.log 2>/dev/null || echo 0", 20)
        if (r2.stdout.strip().splitlines() or ["0"])[-1].strip() not in ("", "0"):
            print(f"  {label} ERROR:", sh(host, port, "tail -8 /root/stage.log", 20).stdout, flush=True)
            return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", nargs="+", required=True, help="REGION:instance_id:lo:hi, head first")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--prompt", default="Explain why WAN latency, not GPU FLOPs, is the bottleneck for a scattered inference swarm. 3 sentences.")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=0, help="spec-decode depth (0=plain greedy)")
    a = ap.parse_args()

    nodes = []
    for spec in a.order:
        region, iid, lo, hi = spec.split(":")
        j = vinst(iid)
        sh_host, sh_port = ssh_ep(j)
        pip, pport = pub_ep(j)
        nodes.append(dict(region=region, iid=iid, host=sh_host, port=sh_port, pip=pip, pport=pport, lo=int(lo), hi=int(hi)))
    n = len(nodes)
    print("[scatter] pushing fresh code to all nodes ...", flush=True)
    for nd in nodes:
        push_code(nd["host"], nd["port"])
    print("[scatter] collecting PeerIds ...", flush=True)
    for nd in nodes:
        nd["pid"] = peerid(nd["host"], nd["port"])
        nd["maddr"] = f"/ip4/{nd['pip']}/tcp/{nd['pport']}/p2p/{nd['pid']}"
        print(f"  {nd['region']} {nd['pip']}:{nd['pport']} layers[{nd['lo']},{nd['hi']}) {nd['pid'][:16]}...", flush=True)

    print("[scatter] launching sidecars (relay-back chain) ...", flush=True)
    for k, nd in enumerate(nodes):
        announce = f"/ip4/{nd['pip']}/tcp/{nd['pport']}"
        inbound = f"127.0.0.1:{ENG_IN}" if k > 0 else ""
        forward = f"127.0.0.1:{FWD}={nodes[k+1]['maddr']}" if k < n - 1 else ""
        ok = launch_sidecar(nd["host"], nd["port"], announce, inbound, forward)
        print(f"  {'OK' if ok else 'FAIL'} sidecar {nd['region']}", flush=True)
        if not ok:
            print(sh(nd["host"], nd["port"], "tail -4 /root/sidecar.log", 20).stdout, flush=True)
            return

    print("[scatter] launching stages tail-first ...", flush=True)
    for k in range(n - 1, -1, -1):
        nd = nodes[k]
        launch_stage(nd["host"], nd["port"], nd["lo"], nd["hi"], is_tail=(k == n - 1))
    for k in range(n - 1, -1, -1):
        nd = nodes[k]
        ok = warm(nd["host"], nd["port"], f"stage{k} {nd['region']}")
        print(f"  {'WARM' if ok else 'FAIL'} stage{k} {nd['region']}", flush=True)
        if not ok:
            return

    head = nodes[0]
    pf = f"--prompt-file {a.prompt_file}" if a.prompt_file else f'--prompt "{a.prompt}"'
    smpl = f"--temp {a.temp}" if a.temp > 0 else ""
    print("[scatter] running coordinator on head ...", flush=True)
    cmd = (f"cd /root && CUDA_VISIBLE_DEVICES=0 M25_DIR=/root/m25 /root/venv/bin/python /root/m25_ring.py coord "
           f"--stage 127.0.0.1:{ENG_IN} --max-new {a.max_new} --k {a.k} {pf} {smpl} 2>&1 | grep -vE 'INFO|WARNING|warn'")
    r = sh(head["host"], head["port"], cmd, timeout=900)
    print(r.stdout, flush=True)
    if r.stderr.strip():
        print("[stderr]", r.stderr[-600:], flush=True)
    print("\n[scatter] ring up; teardown: vastai destroy <ids>", flush=True)


if __name__ == "__main__":
    main()
