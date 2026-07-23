"""Launch the M2.5 PIPELINED direct-return ring across scattered vast boxes over libp2p, and run
the proven coordinate_pipe coordinator (depth chunks in flight = the GLM throughput lever).

Direct-return topology (per launch_libp2p): head sidecar forwards BOTH the ring (->s1) and the
coordinator return-channel (->tail); middle sidecars inbound+forward; tail inbound only. Stages run
m25_pipe (fire-forward); the coordinator (on the head box) dials the head engine locally and the tail
via the 29612 return tunnel.

  python m25_scatter_pipe.py --order CA:42545183:0:10 WA:..:10:23 MN:..:23:36 NJ:..:36:49 NC:..:49:62 \
      --K 6 --depth 4 --max-new 256 --prompt-file /root/copy_prompt.txt
"""
import os, re, sys, json, time, shlex, secrets, tempfile, subprocess, argparse

KEY = "/root/.ssh/vast_c0mpute"
SSHO = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=25", "-i", KEY]
REPO = os.environ.get("SHARD_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the checkout THIS launcher runs from (worktree-safe); override via SHARD_REPO
LIBP2P, ENG_IN, FWD_RING, FWD_RET = 29600, 29610, 29611, 29612

# Every engine flag the operator sets locally must reach ALL ring processes — the stages AND the
# coordinator/gateway. These are read per-process (S.M25_EAGLE etc.), so forwarding them only to the
# stages (the old behavior) silently disabled the feature on the coordinator side and poisoned the
# measurement (e.g. M25_EAGLE=1 warmed aux-capturing stages while the coordinator drafted n-gram-only).
ENG_ENV = ["M25_BATCH_MOE", "M25_KV_FP8", "M25_EAGLE", "M25_EAGLE_AUX", "M25_EAGLE_DIR",
           "M25_EAGLE_NEXT_HIDDEN", "M25_FP8_WIRE", "M25_FP8_AUX", "M25_NGRAM_MINMATCH",
           "M25_CONF_SCHED", "M25_SDPA", "M25_STATIC_KV", "M25_CUDA_GRAPH", "M25_GRAPH_MAX",
           "M25_GRAPH_JOB",                                               # per-job graph A/B: stages need the cap, the coordinator the reset stamp
           "M25_BATCH_GRAPH",                                             # batched-decode graph hatch: read STAGE-side (a hatch that doesn't reach the stages is dead)
           "M25_AUX_SLIM",                                                # accepted-prefix aux slimming hatch: read TAIL-side
           "M25_AUX_LOCAL",                                               # head-local aux lane: coordinator stamps the job, the HEAD stage arms
           "M25_DELOCKSTEP",                                              # per-stream async row frames (coordinator-side dispatch)
           "M25_MOE_BACKEND",
           "M25_DEFAULT_REASONING", "M25_MAX_POS",
           "M25_TREE", "M25_TREE_M", "M25_TREE_TOPB", "M25_TREE_DEPTH",   # tree-verify: stages need M25_TREE (tree kernel), the coordinator all four
           "M25_CWND_KEEPWARM_MS", "M25_KEEPWARM_JOB",                    # cwnd keep-warm: stage senders keep idle legs warm (default-ON for --serve interactive)
           "M25_STAGE_TIMING",                                            # per-stage [span,compute] stamps -> coordinator transport split
           "M25_FRAME_TIMEOUT",                                           # absolute per-frame recv deadline (H4); the sidecar gets the SAME number via -frame-timeout
           "M25_RET_STALL_S",                                             # P0-#5 M1: TAIL-side per-progress stall bound on the coordinator-return socket
           "M25_DRAFT_BUDGET_S", "M25_JOB_STALL_S",                       # P0-#5 L1/L3: coordinator-side draft-budget + job-stall backstop (the gateway coord reads them; forwarded so a --serve launch tunes them too)
           "M25_REPLY_TIMEOUT",                                           # F6 per-reply decode heartbeat (coordinator) — forwardable for the gateway path
           "SHARD_RECEIPT_DUMP"]                                          # coordinator exports the signed receipt set for the c0mpute settle seam


def eng_env():
    """The operator's engine flags as a shell env prefix (only the ones actually set — unset ones fall
    through to each process's own default, which is identical code on both sides)."""
    return "".join(f"{k}={os.environ[k]} " for k in ENG_ENV if k in os.environ)


def vinst(iid):
    return json.loads(subprocess.check_output(["vastai", "show", "instance", str(iid), "--raw"], text=True))


def sh(host, port, cmd, timeout=120):
    return subprocess.run(["ssh", *SSHO, "-p", str(port), f"root@{host}", cmd], capture_output=True, text=True, timeout=timeout)


def push_code(host, port):
    for f in ["phase0/m25_pipe.py", "phase0/m25_stage.py", "phase0/m25_tools.py", "phase0/ngram_draft.py", "phase0/eagle_draft.py",
              "phase0/tree_spec.py", "phase0/node_kv.py", "phase0/confidence.py", "phase0/m25_gateway.py",
              "phase0/safe_kill.sh",                       # every box gets the self-match-proof killer for ad-hoc ops (bash /root/safe_kill.sh PATTERN)
              "shard/transport.py", "shard/receipt.py", "shard/manifest.py"]:
        dst = "/root/" + f.split("/")[-1]
        for attempt in (1, 2):                       # fail LOUD: a silently-dropped scp launches a stale/mixed-version ring
            r = subprocess.run(["scp", *SSHO, "-P", str(port), f"{REPO}/{f}", f"root@{host}:{dst}"], capture_output=True, text=True)
            if r.returncode == 0:
                break
            if attempt == 2:
                raise RuntimeError(f"push_code {host}:{port} failed on {f}: {r.stderr.strip()[-200:]}")


# A PeerId arrives as a REMOTE box's stdout and gets interpolated into root shell commands on
# EVERY other box — validate strict base58btc (libp2p PeerIds; no 0OIl, no quotes/metachars)
# BEFORE it can touch a command string. shlex-quoted again at the point of use (defense in depth).
PEERID_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{40,64}$")


def check_peerid(pid):
    if not PEERID_RE.fullmatch(pid or ""):
        raise ValueError(f"invalid PeerId from node (not base58btc): {pid!r}")
    return pid


# a receipt pubkey is REMOTE stdout headed for a JSON config — keep it strictly base64-shaped
PUBKEY_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


def collect_assignments(nodes):
    """H2 fail-closed settlement: {receipt_pubkey_b64: [lo, hi]} straight from each box's SIGNING
    key (/root/.shard_node_key — NOT the sidecar's libp2p /root/node.key). The coordinator pins
    receipt verification to this map via SHARD_ASSIGNMENTS, so an interloper's validly-signed
    receipt for someone else's layers can never settle."""
    m = {}
    for nd in nodes:
        r = sh(nd["host"], nd["port"],
               "/root/venv/bin/python -c \"from receipt import load_or_make_node_key, pub_b64; "
               "print(pub_b64(load_or_make_node_key('/root/.shard_node_key')))\"", 60)
        pub = (r.stdout.strip().splitlines() or [""])[-1].strip()
        if not PUBKEY_RE.fullmatch(pub):
            raise RuntimeError(f"bad receipt pubkey from {nd['region']}: {pub!r} {r.stderr[-200:]}")
        if pub in m:
            raise RuntimeError(f"duplicate receipt pubkey across nodes ({nd['region']}) — two boxes share a signing key")
        m[pub] = [nd["lo"], nd["hi"]]
    return m


def push_assignments(head, assigns):
    """write the assignment map to the HEAD box as /root/assignments.json (the coordinator/gateway
    runs there and reads it via SHARD_ASSIGNMENTS)."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(assigns, f)
        path = f.name
    try:
        r = subprocess.run(["scp", *SSHO, "-P", str(head["port"]), path, f"root@{head['host']}:/root/assignments.json"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"assignments push failed: {r.stderr.strip()[-200:]}")
    finally:
        os.unlink(path)


def negotiated_max_ctx(operator_max_ctx, stage_kv_caps):
    """ONE context limit for the whole ring: min(operator ceiling, every stage's KV cap). Same
    pure function the gateway exposes (m25_gateway.negotiated_max_ctx) — the launcher computes it
    once and every downstream process receives it as explicit config, so no process ever trusts
    its own 131072 default over a 40960-capped ring."""
    return min([int(operator_max_ctx)] + [int(c) for c in stage_kv_caps if c and int(c) > 0])


def peerid(host, port):
    r = sh(host, port, "/tmp/sidecar -key /root/node.key -prove ping 2>/dev/null | grep PEERID")
    for ln in r.stdout.splitlines():
        if ln.startswith("PEERID "):
            return check_peerid(ln.split()[1])
    raise RuntimeError(f"no PeerId {host}:{port}: {r.stdout[-200:]}{r.stderr[-200:]}")


def fresh_count(host, port, nonce_file, nonce, log, pattern):
    """count PATTERN matches in LOG only when the box's nonce file matches THIS launch (M4): the
    launch command writes the nonce right after truncating the log, so a stale log left by a
    previous run — e.g. after a flaked/timed-out ssh that never ran the launch — can never satisfy
    a readiness check. Returns the count as a string ('0' when the nonce is absent/stale)."""
    q = (f"[ \"$(cat {nonce_file} 2>/dev/null)\" = \"{nonce}\" ] && "
         f"grep -cE {shlex.quote(pattern)} {log} 2>/dev/null || echo 0")
    r = sh(host, port, q, 20)
    return (r.stdout.strip().splitlines() or ["0"])[-1].strip()


def sidecar_cmd(announce, inbound, forwards, seed=None, dht_bootstrap=None, allow=None, nonce="", frame_timeout=60,
                quic=False, relay=False, relays=None):
    """Pure builder (unit-testable): every remote-influenced value (multiaddrs carry PeerIds from
    remote stdout) is shlex-quoted, and the whole inner command is quoted ONCE for the bash -c
    level — correct two-level quoting, byte-identical to the old literal form for legit values."""
    fw = " ".join(f"-forward {shlex.quote(f)}" for f in forwards)
    inb = f"-inbound {shlex.quote(inbound)}" if inbound else ""
    # seeding lifecycle (torrent): a stage that verified-pulled its layer range seeds it on the
    # shard DHT from the SAME tunnel daemon — 'manifest.json=modelDir' + neighbour bootstrap addrs.
    sd = f"-seed {shlex.quote(seed)}" if seed else ""
    bs = " ".join(f"-dht-bootstrap {shlex.quote(b)}" for b in (dht_bootstrap or []))
    # C2: cryptographic neighbour allowlist — the sidecar Reset()s inbound streams whose
    # Noise-authenticated RemotePeer isn't one of these PeerIds (empty = open, legacy).
    al = " ".join(f"-allow {shlex.quote(p)}" for p in (allow or []))
    # H4: absolute per-frame deadline for the tunnel pipe — 60s default clears the worst legit
    # frame (multi-MB bf16 prefill chunk on a ~5MB/s residential uplink ~5-10s) and sits far under
    # the 600s stage timeout; the engine keys off the same number (M25_FRAME_TIMEOUT). 0 disables.
    ft = f"-frame-timeout {int(frame_timeout)}"
    # NAT traversal (home/residential GPUs behind NAT/CGNAT): -quic ALSO listens on udp/quic-v1 derived
    # from the tcp listen port (hole-punches more reliably on lossy/CGNAT links); -relays <csv> reserves
    # a circuit on public relays so a NAT'd node stays dialable while DCUtR punches a direct hole; -relay
    # makes THIS (public) node a relay + AutoNAT server for others. All three are additive to the tcp path.
    qc = "-quic" if quic else ""
    rl = "-relay" if relay else ""
    rs = f"-relays {shlex.quote(relays)}" if relays else ""
    nn = f"echo {nonce} > /root/sidecar.nonce; " if nonce else ""
    inner = (f"/tmp/sidecar -key /root/node.key -listen /ip4/0.0.0.0/tcp/{LIBP2P} "
             f"-announce {shlex.quote(announce)} {inb} {fw} {sd} {bs} {al} {ft} {qc} {rl} {rs} > /root/sidecar.log 2>&1")
    return (f"pkill -9 -x sidecar 2>/dev/null; fuser -k {LIBP2P}/tcp {FWD_RING}/tcp {FWD_RET}/tcp 2>/dev/null; sleep 2; rm -f /root/sidecar.log; {nn}"
            f"setsid bash -c {shlex.quote(inner)} </dev/null >/dev/null 2>&1 &")


def launch_sidecar(host, port, announce, inbound, forwards, seed=None, dht_bootstrap=None, allow=None):
    nonce = secrets.token_hex(8)
    ft = int(os.environ.get("M25_FRAME_TIMEOUT", "60") or 0)   # operator override reaches BOTH the sidecar flag and (via ENG_ENV) the engine
    quic = os.environ.get("SHARD_QUIC", "") not in ("", "0")            # udp/quic-v1 listener for NAT'd / lossy residential links
    relay = os.environ.get("SHARD_RELAY", "") not in ("", "0")          # run THIS (public) node as a circuit-relay + AutoNAT server
    relays = os.environ.get("SHARD_RELAYS", "").strip() or None         # csv of relay multiaddrs a NAT'd node reserves on
    cmd = sidecar_cmd(announce, inbound, forwards, seed=seed, dht_bootstrap=dht_bootstrap, allow=allow,
                      nonce=nonce, frame_timeout=ft, quic=quic, relay=relay, relays=relays)
    for attempt in range(5):
        sh(host, port, cmd, 30)
        for _ in range(4):
            time.sleep(3)
            up = fresh_count(host, port, "/root/sidecar.nonce", nonce, "/root/sidecar.log", "tunnel up|listening")
            bad = fresh_count(host, port, "/root/sidecar.nonce", nonce, "/root/sidecar.log", "address already in use")
            if up not in ("", "0") and bad in ("", "0"):
                return True
        print(f"  sidecar {host} retry {attempt+1}", flush=True)
    return False


def stage_cmd(stage, nstages, lo, hi, is_tail, receipts=False, batch=1, kv_maxlen=0, graph_off=False, token=None, nonce=""):
    nxt = "" if is_tail else f"--next 127.0.0.1:{FWD_RING}"
    rc = "SHARD_RECEIPTS=1 " if receipts else ""
    kv = f"M25_KV_MAXLEN={kv_maxlen} " if kv_maxlen else ""   # cap batched-KV buffer (B*MAXLEN can OOM the tail at MAXLEN=40960)
    # per-stage graph-aux override: CUDA-graph capture of the NVFP4 MoE is proven on the sm_120 cutlass
    # path; a non-Blackwell (marlin) stage runs eager (this env assignment comes AFTER eng_env()'s, so
    # bash uses the last one). A marlin card holds few layers, so it barely benefits from graph anyway.
    goff = "M25_CUDA_GRAPH=0 " if graph_off else ""
    # C2: per-swarm epoch token — engine peers greet with it (hello_pred/hello_return) so a stage
    # never adopts a silent/foreign connection; never printed in any banner or log.
    tk = f"SHARD_SWARM_TOKEN={token} " if token else ""
    # C2: in libp2p mode the only legitimate dialer is the LOCAL sidecar (and the local coordinator
    # on the head) — bind the engine hop to loopback so raw TCP can't bypass the sidecar allowlist.
    nn = f"echo {nonce} > /root/stage.nonce; " if nonce else ""
    return (f"nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; "
            f"fuser -k {ENG_IN}/tcp 2>/dev/null; sleep 4; rm -f /root/stage.log; {nn}cd /root && "
            f"{rc}{tk}SHARD_TRANSPORT=libp2p M25_ENGINE_BIND=127.0.0.1 M25_BATCH={batch} {eng_env()}{goff}"
            f"{kv}CUDA_VISIBLE_DEVICES=0 M25_DIR=/root/m25 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid bash -c "
            f"'/root/venv/bin/python /root/m25_pipe.py stage --stage {stage} --nstages {nstages} --lo {lo} --hi {hi} "
            f"--port {ENG_IN} {nxt} > /root/stage.log 2>&1' </dev/null >/dev/null 2>&1 &")


def launch_stage(host, port, stage, nstages, lo, hi, is_tail, receipts=False, batch=1, kv_maxlen=0, graph_off=False, token=None):
    nonce = secrets.token_hex(8)
    cmd = stage_cmd(stage, nstages, lo, hi, is_tail, receipts=receipts, batch=batch,
                    kv_maxlen=kv_maxlen, graph_off=graph_off, token=token, nonce=nonce)
    try:
        sh(host, port, cmd, 25)
    except subprocess.TimeoutExpired:
        pass
    return nonce


def warm(host, port, label, nonce, tries=80):
    for _ in range(tries):
        time.sleep(8)
        if fresh_count(host, port, "/root/stage.nonce", nonce, "/root/stage.log", "WARM") not in ("", "0"):
            return True
        if fresh_count(host, port, "/root/stage.nonce", nonce, "/root/stage.log", "Traceback|Error|CUDA out") not in ("", "0"):
            print(f"  {label} ERROR:\n" + sh(host, port, "tail -12 /root/stage.log", 20).stdout, flush=True)
            return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", nargs="+", required=True)
    ap.add_argument("--K", type=int, default=8); ap.add_argument("--depth", type=int, default=4)   # K=8 = the measured sweet spot (2026-06-27 sweep)
    ap.add_argument("--max-new", type=int, default=256); ap.add_argument("--ngram-n", type=int, default=3)
    ap.add_argument("--prompt", default="Explain a decentralized inference swarm in 3 sentences.")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--sweep", default=None); ap.add_argument("--sweep-depth", default=None)  # pass through to coord
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--validate", action="store_true"); ap.add_argument("--receipts", action="store_true")
    ap.add_argument("--serve", action="store_true", help="deploy mode: after warm, start the OpenAI /v1 gateway on the head (persistent) instead of a one-shot coord job")
    ap.add_argument("--warm-only", action="store_true", help="warm stages+sidecars then STOP (no coord/gateway) so a measurement tool can run as the SOLE first coordinator on the head box")
    ap.add_argument("--batch", type=int, default=1, help="continuous batching: stages allocate [B,...] KV (M25_BATCH); warm the ring with --serve then drive coordinate_pipe_batch")
    ap.add_argument("--kv-maxlen", type=int, default=0, help="cap M25_KV_MAXLEN (batched KV is B*MAXLEN per layer; 40960 default OOMs the tail at B>=4)")
    ap.add_argument("--max-ctx", type=int, default=131072, dest="max_ctx",
                    help="operator context ceiling; the gateway gets min(this, every stage's KV cap)")
    ap.add_argument("--seed-shards", action="store_true",
                    help="torrent seeding lifecycle: every stage's sidecar also SEEDS its verified layer range "
                         "on the shard DHT (/root/m25_manifest.json=/root/m25, neighbours as bootstrap) so "
                         "joiners can pull from peers instead of the mirror")
    ap.add_argument("--external-tail", default=None,
                    help="attach a NON-SSH tail (home/NAT'd GPU the operator can't SSH) via its relay CIRCUIT "
                         "multiaddr: 'MADDR:LO:HI'. Its stage+sidecar are hand-launched on that box; the vast "
                         "head+predecessor forward to the circuit addr, everything else is unchanged.")
    ap.add_argument("--swarm-token", default=None,
                    help="use this pre-shared C2 epoch token (hex) instead of generating one, so a hand-launched "
                         "external-tail stage can carry the SAME SHARD_SWARM_TOKEN.")
    a = ap.parse_args()
    # Interactive deploy (--serve = the OpenAI gateway) defaults cwnd keep-warm ON: single-stream legs
    # idle between tokens long enough to trip TCP slow-start-after-idle (cwnd collapse -> the next frame
    # eats 2-4 extra RTTs), so tiny noops keep every leg hot. eng_env() forwards it to the stages AND the
    # gateway. Override via the env, set =0 to disable.
    # The "neutral on batched rings (legs never idle)" note above was WRONG for B>=4 (2026-07-11
    # research): B=1 rounds (~165ms) sit UNDER Linux RTO_min (200ms) so cwnd survives, but B>=4
    # lockstep rounds (450-900ms) idle every leg PAST the RTO -> cwnd collapses to IW10 EVERY round
    # -> +2-3.5 RTTs/leg = +180-500ms/round. Every batched receipt through 2026-07-11 carries that
    # handicap. Default keep-warm ON for --serve AND any batched launch; --warm-only solo measurement
    # stays OFF for A/B purity unless the operator sets it.
    if (a.serve or a.batch > 1) and "M25_CWND_KEEPWARM_MS" not in os.environ:
        os.environ["M25_CWND_KEEPWARM_MS"] = "150"
    nodes = []
    for spec in a.order:
        parts = spec.split(":")
        region, iid, lo, hi = parts[:4]
        # optional 5th field "eager": launch this stage graph-OFF (a probe-measured
        # graph-corrupt card serving at its eager speed — eager IS its reference numerics)
        forced_eager = len(parts) > 4 and parts[4] == "eager"
        j = vinst(iid)
        ports = j.get("ports") or {}
        m = ports.get(f"{LIBP2P}/tcp")
        host, port = j["ssh_host"], int(j["ssh_port"])
        # SHARD_SSH_OVERRIDE="iid=host:port,...": route around a BROKEN vast ssh-proxy mapping.
        # Proven failure (2026-07-23): the API returned ONE proxy endpoint for two different
        # instances, so every ssh op (bootstrap, code push, key collect) silently hit the same box
        # twice — caught only by the duplicate-receipt-pubkey guard. The direct '22/tcp' port map
        # on public_ipaddr keeps working; this pins it explicitly, per iid.
        ov = dict(kv.split("=") for kv in os.environ.get("SHARD_SSH_OVERRIDE", "").split(",") if "=" in kv)
        if iid in ov:
            host, port = ov[iid].rsplit(":", 1)
            port = int(port)
            print(f"[pipe] ssh override {region} {iid} -> {host}:{port}", flush=True)
        nodes.append(dict(region=region, iid=iid, host=host, port=port,
                          pip=(j.get("public_ipaddr") or "").strip(), pport=m[0]["HostPort"] if m else None,
                          lo=int(lo), hi=int(hi), forced_eager=forced_eager))
    if a.external_tail:                               # a NON-SSH tail (home GPU): pre-set its circuit maddr + pid,
        maddr, elo, ehi = a.external_tail.rsplit(":", 2)   # the vast head+predecessor forward to it, we never SSH it
        epid = maddr.rsplit("/p2p/", 1)[-1]           # the tail PeerId = last /p2p/ hop of the circuit addr
        nodes.append(dict(region="EXT", iid=None, host=None, port=None, pip=None, pport=None,
                          lo=int(elo), hi=int(ehi), forced_eager=True, external=True, maddr=maddr,
                          pid=epid, gpu="external-tail", graph_off=True))
    n = len(nodes)
    # H1: ONE negotiated context limit. Pin the stage KV cap explicitly (40960 = m25_stage.py's
    # M25_KV_MAXLEN default) so the negotiated ctx is REAL config, not a guessed default; the
    # gateway/coordinator then reject-or-clamp at min(operator ceiling, every stage's cap) instead
    # of silently running 131072 over 40960 stages (KV overflow mid-job).
    kv_eff = a.kv_maxlen or 40960
    eff_max_ctx = negotiated_max_ctx(a.max_ctx, [kv_eff] * n)
    print(f"[pipe] negotiated max_ctx {eff_max_ctx} (operator {a.max_ctx}, stage KV cap {kv_eff})", flush=True)
    # C2: one per-launch swarm/epoch token, injected as env into every stage + the gateway/coord
    # (never printed). --warm-only stays token-less: its whole point is an EXTERNAL measurement
    # coordinator on the head box, which predates the greeting protocol — legacy classification.
    swarm_token = a.swarm_token or (None if a.warm_only else secrets.token_hex(16))
    print("[pipe] push code + PeerIds ...", flush=True)
    for nd in nodes:
        if nd.get("external"):                        # hand-launched home tail: pid+maddr already set, never SSHed
            print(f"  EXT external-tail [{nd['lo']},{nd['hi']}) {nd['pid'][:14]}.. (via relay circuit)", flush=True)
            continue
        push_code(nd["host"], nd["port"])
        nd["pid"] = peerid(nd["host"], nd["port"])
        nd["maddr"] = f"/ip4/{nd['pip']}/tcp/{nd['pport']}/p2p/{nd['pid']}"
        g = sh(nd["host"], nd["port"], "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1", 20)
        nd["gpu"] = (g.stdout.strip().splitlines() or ["?"])[-1].strip()
        # graph-aux runs on EVERY arch by default: marlin (Ada 4090) is CUDA-graph-safe — proven live,
        # the 4090 stage dropped 32.65ms -> 8.02ms, receipts still valid — so a non-Blackwell card is a
        # FULL-speed ring stage, not an eager drag. M25_EAGER_NONBLACKWELL=1 restores eager for a card
        # whose backend turns out not to be graph-safe (the warm check would otherwise catch a crash).
        eager_nb = os.environ.get("M25_EAGER_NONBLACKWELL", "0") != "0"
        nd["graph_off"] = nd.get("forced_eager", False) or \
            (eager_nb and not any(b in nd["gpu"] for b in ("5090", "5080", "5070")))
        print(f"  {nd['region']} {nd['gpu']} {nd['pip']}:{nd['pport']} [{nd['lo']},{nd['hi']}) "
              f"{'eager' if nd['graph_off'] else 'graph'} {nd['pid'][:14]}..", flush=True)

    if a.receipts:                                    # H2: pin settlement to the assigned signer set
        print("[pipe] collecting receipt pubkeys -> head:/root/assignments.json ...", flush=True)
        push_assignments(nodes[0], collect_assignments(nodes))

    print("[pipe] sidecars (direct-return: head forwards ring+ret) ...", flush=True)
    for k, nd in enumerate(nodes):
        if nd.get("external"):                        # the home tail's sidecar is hand-launched; the vast
            continue                                  # head+predecessor already forward to nd['maddr'] (its circuit)
        announce = f"/ip4/{nd['pip']}/tcp/{nd['pport']}"
        inbound = f"127.0.0.1:{ENG_IN}" if k > 0 else ""           # head's predecessor is the local coord
        forwards = []
        if k < n - 1:
            forwards.append(f"127.0.0.1:{FWD_RING}={nodes[k+1]['maddr']}")
        if k == 0:
            forwards.append(f"127.0.0.1:{FWD_RET}={nodes[-1]['maddr']}")   # head also tunnels coord-return -> tail
        seed = "/root/m25_manifest.json=/root/m25" if a.seed_shards else None
        # bootstrap through the PREDECESSOR's sidecar only — it launched before us (k asc), so the
        # DHT link is up when we dial; the successor reciprocates when it launches. k=0 seeds solo.
        bsp = [nodes[k - 1]["maddr"]] if (a.seed_shards and k > 0) else None
        # C2: each inbound sidecar only admits its PREDECESSOR's PeerId; the TAIL also admits the
        # HEAD's — the coordinator-return tunnel is the head sidecar's -forward, so return streams
        # arrive at the tail with RemotePeer == head. Head (k==0) has no -inbound, no allowlist.
        allow = None
        if k > 0:
            allow = [nodes[k - 1]["pid"]]
            if k == n - 1:
                allow.append(nodes[0]["pid"])
        ok = launch_sidecar(nd["host"], nd["port"], announce, inbound, forwards, seed=seed, dht_bootstrap=bsp, allow=allow)
        print(f"  {'OK' if ok else 'FAIL'} {nd['region']}", flush=True)
        if not ok:
            print(sh(nd["host"], nd["port"], "tail -4 /root/sidecar.log", 20).stdout)
            sys.exit(1)                       # M4: a failed launch must FAIL the launcher

    print("[pipe] stages tail-first ...", flush=True)
    nonces = {}
    for k in range(n - 1, -1, -1):
        if nodes[k].get("external"):                  # the home tail's stage is hand-launched (no SSH); it must be
            continue                                  # already up so the coordinator's first job reaches it
        nonces[k] = launch_stage(nodes[k]["host"], nodes[k]["port"], k, n, nodes[k]["lo"], nodes[k]["hi"], k == n - 1, a.receipts, a.batch, kv_eff, graph_off=nodes[k].get("graph_off", False), token=swarm_token)
    for k in range(n - 1, -1, -1):
        if nodes[k].get("external"):
            continue
        ok = warm(nodes[k]["host"], nodes[k]["port"], f"s{k} {nodes[k]['region']}", nonces[k])
        print(f"  {'WARM' if ok else 'FAIL'} s{k} {nodes[k]['region']}", flush=True)
        if not ok:
            sys.exit(1)

    if a.external_tail:                           # the home tail is hand-launched — echo the exact commands + wiring
        et = nodes[-1]; pred = nodes[-2]; head0 = nodes[0]
        relay = et["maddr"].split("/p2p-circuit")[0]
        print("[pipe] EXTERNAL TAIL — the home box must ALREADY be running these (no SSH from here):", flush=True)
        print(f"  sidecar: ~/sidecar -key ~/.shard_reach.key -listen /ip4/0.0.0.0/tcp/{LIBP2P} -quic -relays {relay} -inbound 127.0.0.1:{ENG_IN}", flush=True)
        print(f"  stage:   cd ~/shard && PYTHONPATH=$HOME/shard/phase0:$HOME/shard/shard SHARD_TRANSPORT=libp2p "
              f"M25_ENGINE_BIND=127.0.0.1 M25_MOE_BACKEND=auto M25_CUDA_GRAPH=0 M25_DIR=$HOME/m25 "
              f"SHARD_SWARM_TOKEN={swarm_token} python phase0/m25_pipe.py stage --stage {n-1} --nstages {n} --lo {et['lo']} --hi {et['hi']} --port {ENG_IN}", flush=True)
        print(f"  wired -> predecessor s{n-2} ({pred['pid'][:12]}..) sends the ring, head s0 ({head0['pid'][:12]}..) sends the return, both to {et['maddr'][:44]}..", flush=True)
    head = nodes[0]
    if a.warm_only:                               # warm + STOP: run the measurement as the sole coordinator on the head (nxt_sock breaks if anything connects first)
        print(f"[pipe] WARM-ONLY — ring up. Drive it as the SOLE coordinator ON the head box:", flush=True)
        print(f"  ssh -i {KEY} -p {head['port']} root@{head['host']}", flush=True)
        print(f"  SHARD_TRANSPORT=libp2p HEAD_PORT={ENG_IN} TAIL_PORT={FWD_RET} M25_DIR=/root/m25 /root/venv/bin/python -u /root/m25_ctx_table.py", flush=True)
        print(f"HEAD_SSH {head['host']}:{head['port']}", flush=True)
        return
    if a.serve:                                   # DEPLOY: start the OpenAI /v1 gateway on the head over the warm ring
        GW = 18000
        rc = "SHARD_RECEIPTS=1 SHARD_ASSIGNMENTS=/root/assignments.json " if a.receipts else ""
        bt = f"M25_BATCH={a.batch} " if a.batch > 1 else ""   # gateway micro-batches up to the ring's KV rows
        tk = f"SHARD_SWARM_TOKEN={swarm_token} " if swarm_token else ""
        gnonce = secrets.token_hex(8)
        gw = (f"fuser -k {GW}/tcp 2>/dev/null; sleep 1; rm -f /root/gateway.log; echo {gnonce} > /root/gateway.nonce; "
              f"cd /root && {rc}{tk}{bt}SHARD_TRANSPORT=libp2p {eng_env()}M25_DIR=/root/m25 "
              f"setsid bash -c '/root/venv/bin/python /root/m25_gateway.py --head 127.0.0.1:{ENG_IN} --tail 127.0.0.1:{FWD_RET} "
              f"--port {GW} --K {a.K} --depth {a.depth} --ngram-n {a.ngram_n} --max-ctx {eff_max_ctx} > /root/gateway.log 2>&1' </dev/null >/dev/null 2>&1 & echo SERVING")
        # the setsid gateway detaches, but its outer `bash -c` stays on the ssh channel, so this ssh
        # doesn't return promptly (this is why the plain detach wasn't enough). Don't block or crash
        # on it — the gateway IS launched; real readiness is the nonce-gated poll below (M4).
        try:
            sh(head["host"], head["port"], gw, 8)
        except subprocess.TimeoutExpired:
            pass
        ok = False                            # M4: poll the REAL startup banner of THIS launch (nonce-gated), not any historical log line
        for _ in range(15):
            time.sleep(4)
            if fresh_count(head["host"], head["port"], "/root/gateway.nonce", gnonce, "/root/gateway.log", "m25-gateway") not in ("", "0"):
                ok = True; break
        print(f"[pipe] gateway {'UP' if ok else 'FAILED (see /root/gateway.log)'} on head, 127.0.0.1:{GW} (OpenAI /v1, single-stream)", flush=True)
        print(f"[pipe] reach it:  ssh -i {KEY} -p {head['port']} -L 8000:127.0.0.1:{GW} root@{head['host']}   then POST http://localhost:8000/v1/chat/completions", flush=True)
        if not ok:
            print(sh(head["host"], head["port"], "tail -5 /root/gateway.log", 20).stdout, flush=True)
            sys.exit(1)
        return
    pf = f"--prompt-file {a.prompt_file}" if a.prompt_file else f'--prompt "{a.prompt}"'
    sw = (f"--sweep {a.sweep} " if a.sweep else "") + (f"--sweep-depth {a.sweep_depth} " if a.sweep_depth else "") + ("--validate " if a.validate else "")
    rc = "SHARD_RECEIPTS=1 SHARD_ASSIGNMENTS=/root/assignments.json " if a.receipts else ""
    tk = f"SHARD_SWARM_TOKEN={swarm_token} " if swarm_token else ""
    print("[pipe] coordinator (pipelined) on head ...", flush=True)
    # M4: pipefail — without it the pipeline's rc is the trailing grep's, and a crashed coordinator
    # still exits 0 (an orchestrator above this launcher would read the run as green).
    cmd = (f"set -o pipefail; cd /root && {rc}{tk}SHARD_TRANSPORT=libp2p {eng_env()}CUDA_VISIBLE_DEVICES=0 M25_DIR=/root/m25 /root/venv/bin/python /root/m25_pipe.py coord "
           f"--head 127.0.0.1:{ENG_IN} --tail 127.0.0.1:{FWD_RET} --K {a.K} --depth {a.depth} --ngram-n {a.ngram_n} "
           f"--max-new {a.max_new} --prefill-chunk {a.prefill_chunk} --max-ctx {eff_max_ctx} {sw}{pf} 2>&1 | tee /root/coord.log | grep -vE 'INFO|WARNING|warn|instantiate'")
    r = sh(head["host"], head["port"], cmd, timeout=1800 if (a.sweep or a.sweep_depth or a.validate) else 1200)
    print(r.stdout, flush=True)
    if r.stderr.strip():
        print("[stderr]", r.stderr[-700:], flush=True)
    if r.returncode != 0:
        print(f"[pipe] coordinator FAILED (rc={r.returncode})", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
