"""Bring up the gpt-oss-120B ring over the libp2p SIDECAR transport (per-node keys, NO SHARD_PSK) and
drive it with the n-gram coordinator — re-validating the perf path (incl. async inter-stage send) on the
REAL permissionless transport, not raw-TCP+PSK. Proves the perf wins carry to libp2p.

Each node runs: a sidecar (TCP<->libp2p tunnel, /tmp/sidecar) + the engine (specpipe with
SHARD_TRANSPORT=libp2p, which swaps wire.py for shard/transport.py). Ports per node:
  29600  libp2p listen (mapped to the box's public port; the dialable address)
  29610  engine listen  (sidecar -inbound delivers inbound libp2p streams here)
  29611  engine --next   (sidecar -forward carries it to the successor over libp2p)
  29612  coordinator ret (HEAD sidecar -forward carries it to the tail, for direct-return)

The coordinator runs ON the head box: --next 127.0.0.1:29610 (head engine, local) and --tail
127.0.0.1:29612 (head sidecar -> tail). serve_tail_fast still distinguishes predecessor vs
coordinator-return by content (hello_return), both arriving on the tail engine's 29610.

  SHARD_PSK=$(cat ~/.shard_psk) python3 launch_libp2p.py --stages A,B,C --max-ctx 16384 \
      --prompt-file /root/ft_prompt.txt --K 4 --depth 2 --max-new 64

Teardown is manual (vastai destroy)."""
import argparse, time

from launch_oss import ep, fire, instances, rssh, warm_stage, M120, PORT, PSK

LIBP2P = 29600          # sidecar libp2p listen (== the vast-mapped public port)
ENG_IN = 29610          # engine listen / sidecar inbound target
FWD_RING = 29611        # engine --next -> sidecar forward to successor
FWD_RET = 29612         # coordinator ret -> head sidecar forward to tail


def peerid(inst):
    """create-or-load the node key and print its PeerId (the sidecar embeds the pubkey in the id)."""
    r = rssh(inst, "/tmp/sidecar -key /root/node.key -prove ping 2>/dev/null | grep PEERID", 60)
    for line in r.stdout.splitlines():
        if line.startswith("PEERID "):
            return line.split()[1]
    raise RuntimeError(f"no PeerId from {inst['id']}: {r.stdout[-200:]} {r.stderr[-200:]}")


def maddr(inst, pid):
    ip, port = ep(inst)
    return f"/ip4/{ip}/tcp/{port}/p2p/{pid}"


def launch_sidecar(inst, announce, inbound, forwards):
    """forwards: list of 'localaddr=peer_multiaddr'. inbound: engine addr or '' (head has none).
    RETRIES: vast SSH is flaky (rc=255 'try again after a few seconds'), and a missed sidecar launch =
    the engine's forward connect gets refused. So launch + verify ('listening' in sidecar.log) up to 4×."""
    fw = " ".join(f"-forward {f}" for f in forwards)
    inb = f"-inbound {inbound}" if inbound else ""
    # proper detach (setsid bash -c '...' </dev/null >/dev/null 2>&1 &) — a bare setsid keeps the ssh
    # channel's fds and the daemon dies when ssh closes (the round-trip self-test only stayed up this way).
    # NB: do NOT `pkill -f /tmp/sidecar` here — this command STRING contains "/tmp/sidecar", so pkill -f
    # self-matches and kills the launching shell before the daemon starts (the documented specpipe footgun).
    # Free the libp2p port instead (kills any old sidecar bound there; never matches the launch shell).
    cmd = (f"fuser -k {LIBP2P}/tcp 2>/dev/null; sleep 1; rm -f /root/sidecar.log; "
           f"setsid bash -c '/tmp/sidecar -key /root/node.key -listen /ip4/0.0.0.0/tcp/{LIBP2P} "
           f"-announce {announce} {inb} {fw} > /root/sidecar.log 2>&1' </dev/null >/dev/null 2>&1 &")
    for attempt in range(6):
        fire(inst, cmd)
        for _ in range(4):                              # tolerant verify: vast ssh rc=255 can flake the CHECK too
            time.sleep(3)
            try:
                r = rssh(inst, "grep -cE 'tunnel up|listening' /root/sidecar.log 2>/dev/null || echo 0", 20)
                if r.returncode == 0:
                    last = r.stdout.strip().splitlines()[-1].strip() if r.stdout.strip() else "0"
                    if last not in ("", "0"):
                        return True
            except Exception:
                pass
        print(f"  sidecar {inst['id']} attempt {attempt+1} not up; relaunching", flush=True)
    print(f"  sidecar {inst['id']} FAILED to come up after retries", flush=True)
    return False


def launch_engine(inst, stage, nstages, served_head, max_ctx, timeout, sync_send, model):
    head = " --served-head" if served_head else ""
    nxt = f" --next 127.0.0.1:{FWD_RING}" if stage < nstages - 1 else ""
    env = f"SHARD_TRANSPORT=libp2p" + (" SHARD_SYNC_SEND=1" if sync_send else "")
    cmd = (f"nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; "
           f"fuser -k {ENG_IN}/tcp 2>/dev/null; sleep 6; rm -f /root/stage.log /root/.shard_next_*; cd /root && "
           f"{env} setsid bash -c 'python3 specpipe.py --stage {stage} --nstages {nstages} --model {model} "
           f"--listen-port {ENG_IN}{nxt}{head} --fast --direct-return --max-ctx {max_ctx} "
           f"--timeout {timeout} > /root/stage.log 2>&1' </dev/null >/dev/null 2>&1 &")
    fire(inst, cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", required=True, help="comma ids, head first")
    ap.add_argument("--model", default=M120, help="model path on the boxes (default the 120B; M20 for the 20B)")
    ap.add_argument("--max-ctx", type=int, default=16384)
    ap.add_argument("--prompt-file", default="/root/ft_prompt.txt")
    ap.add_argument("--prefill-chunk", type=int, default=4096)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--ngram-n", type=int, default=3)
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--edge-timeout", type=int, default=1200)
    ap.add_argument("--sync-send", action="store_true", help="SHARD_SYNC_SEND=1 baseline (A/B over libp2p)")
    ap.add_argument("--no-launch", action="store_true", help="ring already up; just run the coordinator")
    a = ap.parse_args()
    sids = [int(x) for x in a.stages.split(",")]
    insts = instances(); stages = [insts[i] for i in sids]; nstages = len(stages)
    head, tail = stages[0], stages[-1]

    if not a.no_launch:
        print("[libp2p] collecting PeerIds ...", flush=True)
        pids = [peerid(s) for s in stages]
        maddrs = [maddr(stages[k], pids[k]) for k in range(nstages)]
        for k, s in enumerate(stages):
            print(f"  stage{k} {s['id']} ({s.get('geolocation')}) {maddrs[k]}", flush=True)
        # sidecars: head forwards ring(->s1) + ret(->tail); middles forward ring; tail just inbound
        print("[libp2p] launching sidecars ...", flush=True)
        for k, s in enumerate(stages):
            ann = maddr(s, pids[k]).rsplit("/p2p/", 1)[0]      # announce = /ip4/ip/tcp/port (no /p2p)
            forwards, inbound = [], (f"127.0.0.1:{ENG_IN}" if k > 0 else "")
            if k < nstages - 1:
                forwards.append(f"127.0.0.1:{FWD_RING}={maddrs[k + 1]}")
            if k == 0:                                          # head also tunnels the coordinator's ret -> tail
                forwards.append(f"127.0.0.1:{FWD_RET}={maddrs[-1]}")
            launch_sidecar(s, ann, inbound, forwards)
        time.sleep(4)
        # engines tail-first (so a forward dial finds a listening successor); retry once for SSH flakiness
        print("[libp2p] launching engines tail-first (SHARD_TRANSPORT=libp2p, --fast --direct-return) ...", flush=True)
        for k in range(nstages - 1, -1, -1):
            ok = False
            for attempt in range(2):
                launch_engine(stages[k], k, nstages, served_head=(k == 0), max_ctx=a.max_ctx,
                              timeout=a.edge_timeout, sync_send=a.sync_send, model=a.model)
                _, ok = warm_stage(stages[k], f"stage{k} {stages[k]['id']}")
                if ok:
                    break
                print(f"  stage{k} warm attempt {attempt+1} failed; retrying", flush=True)
            print(f"  {'OK' if ok else 'FAIL'} stage{k}", flush=True)
            if not ok:
                print("[abort] engine failed to warm; sidecar.log + stage.log:", flush=True)
                print(rssh(stages[k], "tail -5 /root/sidecar.log; echo ---; tail -8 /root/stage.log", 30).stdout, flush=True)
                return

    # coordinator on the head: --next = head engine (local), --tail = head sidecar ret-forward
    print("[libp2p] running n-gram coordinator on head ...", flush=True)
    sync = " SHARD_SYNC_SEND=1" if a.sync_send else ""
    cmd = (f"cd /root && SHARD_TRANSPORT=libp2p{sync} python3 specpipe.py --coordinator --nstages {nstages} "
           f"--model {a.model} --ngram-draft --ngram-n {a.ngram_n} --pipe --depth {a.depth} --K {a.K} "
           f"--next 127.0.0.1:{ENG_IN} --direct-return --tail 127.0.0.1:{FWD_RET} --prompt-file {a.prompt_file} "
           f"--prefill-chunk {a.prefill_chunk} --max-ctx {a.max_ctx} --max-new {a.max_new} "
           f"--reasoning {a.reasoning} --timeout {a.edge_timeout} --dump /root/run.json 2>&1 | grep -viE 'INFO|WARNING|warn'")
    r = rssh(head, cmd, timeout=a.edge_timeout + 600)
    print(r.stdout[-3000:], flush=True)
    if r.stderr.strip():
        print("[stderr]", r.stderr[-800:], flush=True)
    print("\n[done] ring still up; teardown: vastai destroy instance <id>", flush=True)


if __name__ == "__main__":
    main()
