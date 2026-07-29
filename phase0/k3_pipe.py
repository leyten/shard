"""Kimi-K3 PIPELINED ring — the full-ring launcher/coordinator, the K3 analogue of m25_pipe +
m25_scatter_pipe. This is what the mechanics harness (scratchpad/.../k3_ring.py, request/response)
did NOT do: a real fire-forward ring with a head that embeds, a tail that samples, a weightless
coordinator that drives greedy generation over the return tunnel, and the sidecar wiring for a
19-box deployment.

WHAT IS DIFFERENT FROM M2.5 — AND WHAT IS NOT
The M2.5 ring (m25_pipe / m25_scatter_pipe) moves ONE hidden tensor per hop. A K3 layer boundary
moves TWO: the running hidden state AND the AttnRes `block_residual` stack (phase0/k3_stage.py's
whole reason to exist). That is the ONLY delta. shard/transport.py's codec already encodes a dict
of tensors into one frame, so {"op":"step", "h":h, "br":br, "start_pos":n} rides the SAME sidecar
tunnel, byte for byte — proven over real WAN (HU->NO) in k3-mechanics-ring.md. So the topology,
the direct-return sidecar wiring, the reset/receipt sweep and the SHARD_JOB_* stdout contract are
mirrored from M2.5 unchanged; only the per-hop payload is a pair.

Phase 1 is GREEDY SEQUENTIAL decode (g=1): K3 ships no MTP head (num_nextn_predict_layers 0) and no
drafter is trained, so a traversal moves ONE token, not M2.5's (K+1)-token draft bundle. There is no
speculation, no depth/K, no tree — the coordinate loop is "send a token, get a token back, repeat".
That also means a stage never rewinds (k3_stage refuses a KDA rollback), which is exactly what greedy
sequential decode needs and nothing more (rollback is M3).

TOPOLOGY (direct-return, per m25_scatter_pipe)
  coordinator (on the head box, weightless) --pipe--> head engine ENG_IN
  head --FWD_RING--> s1 --FWD_RING--> ... --> tail          (fire-forward; each sidecar -forward)
  tail --ret--> coordinator, tunneled by the HEAD sidecar's FWD_RET -forward to the tail's inbound
The tail holds TWO inbound streams on its one engine port: the predecessor (job frames) and the
coordinator-return (identified by a hello_return greeting). Only the TAIL answers the coordinator;
head/middle stages forward frames down the ring and never reply on the pipe.

WHAT CROSSES THE RETURN TUNNEL: a TOKEN, not logits. The tail runs final-norm + output-AttnRes +
lm_head, then SAMPLES (greedy argmax by default, temperature optional) and returns the int id —
K3's vocab is ~160k, so streaming a per-token [vocab] logits vector down the WAN would dwarf the
~75 KiB/token boundary payload for no gain. This is m25's convention (the tail argmaxes; sampling is
pinned greedy in the reset).

RECEIPTS: each stage signs the activation hash-chain of its block over the job (shard/receipt.py).
Unlike M2.5's single-tensor chain, a K3 stage attests the FULL two-tensor boundary — it hashes
(h, block_residual) at input and output — so the receipt chain (out_root==in_root across adjacent
stages) proves the AttnRes stack crossed intact, not just the hidden state. The coordinator threads
the settlement nonce into the reset (every stage signs it), sweeps the ring once at job end, strips
the post-sign `stage` debug tag with wire_receipt() (C10) and verifies coverage — fail-closed.

  self-test (CPU, no GPU, no spend):  python3 phase0/k3_pipe.py selftest
  self-test, multi-GPU tail box:      python3 phase0/k3_pipe.py selftest-relay
  one stage:   python3 phase0/k3_pipe.py stage --stage 0 --nstages 3 --lo 0 --hi 31 --port 29610 --next 127.0.0.1:29611
  coordinator: python3 phase0/k3_pipe.py coord --head 127.0.0.1:29610 --tail 127.0.0.1:29612 --dir /root/k3
"""
import argparse
import json
import os
import select
import socket
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import torch  # noqa: E402
import k3_stage as K3  # noqa: E402

try:                                                    # flat box layout (files pushed to /root/) else package
    from transport import send_msg, recv_msg
except ImportError:
    from shard.transport import send_msg, recv_msg
try:
    from receipt import (ReceiptSigner, load_or_make_node_key, pub_b64, verify_coverage,
                         wire_receipt, ReceiptError)
except ImportError:
    from shard.receipt import (ReceiptSigner, load_or_make_node_key, pub_b64, verify_coverage,
                               wire_receipt, ReceiptError)

# The direct-return port map, identical to m25_scatter_pipe so a K3 ring reuses the SAME sidecar
# wiring: the sidecar is payload-agnostic (proven), so only the launched engine binary changes.
LIBP2P, ENG_IN, FWD_RING, FWD_RET = 29600, 29610, 29611, 29612
# Multi-GPU-per-box: one box runs G stage processes, each pinned to a distinct local GPU, chained
# over LOOPBACK. The box's FIRST local stage binds ENG_IN (so the sidecar -inbound and the head
# coordinator reach it); its 2nd..Gth bind ENG_LOCAL_BASE+idx, ports the sidecar never touches
# (they hand off loopback). Base > FWD_RET so a local port can't collide with the WAN legs.
ENG_LOCAL_BASE = 29620
# k3_moe_mxfp4.vllm_ctx binds its single-GPU vLLM group on MASTER_PORT (its K3_MOE_PORT default,
# 29557). Co-located stages each init their own group, so each needs a DISTINCT MASTER_PORT or the
# second stage's init_distributed_environment dies on "address already in use" — the launcher gives
# every local GPU K3_MOE_PORT_BASE+gpu.
K3_MOE_PORT_BASE = 29557
NODELAY = (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
# Forward-dial retry window (Bug 1, live K3 ring 2026-07-29): every stage dials its --next at boot,
# but in a real B-box x G-GPU ring all stages load their weights IN PARALLEL and a single stage's
# Marlin repack + whole-layer graph capture is 30-50s (minutes on a big shard). So a front stage can
# be up and dialing long before a downstream neighbour finishes binding. The window must therefore
# cover the SLOWEST stage's whole load, not a few seconds — the old fixed ~30s (120 tries x 0.25s)
# gave up mid-launch on a 14-stage ring and aborted it. Default derives from the stage `timeout`
# (the same 600s ceiling a stage already tolerates on a frame), overridable via K3_DIAL_RETRY_S for
# a very large ring; each connect is capped at K3_DIAL_CONNECT_TIMEOUT so one black-holing WAN box
# cannot burn the whole window in a single blocked connect.
DIAL_RETRY_S = float(os.environ.get("K3_DIAL_RETRY_S", "0") or 0)          # 0 => derive from timeout
DIAL_CONNECT_TIMEOUT = float(os.environ.get("K3_DIAL_CONNECT_TIMEOUT", "5") or 5)
NODE_KEY_PATH = os.environ.get("SHARD_NODE_KEY", "/root/.shard_node_key")
# C2: per-swarm epoch token — env-only (never argv/log), rides every greeting so a stage never
# adopts a silent/foreign connection. Optional; a bare localhost ring runs token-less.
SWARM_TOKEN = os.environ.get("SHARD_SWARM_TOKEN") or None
RECEIPTS = os.environ.get("SHARD_RECEIPTS", "") not in ("", "0")


# ── receipt payload digest: the FULL two-tensor boundary, not just the hidden state ────────────────

def _tbytes(t):
    return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _payload_bytes(h, br):
    """Deterministic bytes of a K3 boundary payload (hidden || block_residual), for the receipt
    hash-chain. Hashing BOTH tensors is what makes the chain attest the AttnRes stack: a hop that
    dropped `br` would break out_root==in_root, where M2.5's single-tensor chain could not see it.
    Identical bytes on both sides of a lossless hop (the wire is byte-faithful — k3-mechanics-ring),
    so adjacent stages' out_root/in_root match by construction."""
    return _tbytes(h) + _tbytes(br)


# ── sampling: the tail turns the last-position logits into a token ─────────────────────────────────

def sample_token(logits_row, temp=0.0, gen=None):
    """[vocab] -> int id. Greedy argmax at temp 0 (deployed default, bit-exact and the parity bar);
    temperature samples from softmax(logits/temp) with an optional torch.Generator for determinism.
    Ties break to the lowest index, exactly like the reference's argmax."""
    if temp and temp > 0:
        probs = (logits_row.float() / float(temp)).softmax(-1)
        return int(torch.multinomial(probs, 1, generator=gen).item())
    return int(logits_row.argmax().item())


# ── stage server: one K3 layer block, fire-forward ────────────────────────────────────────────────

def _dial_window(timeout):
    """The forward-dial retry WINDOW in seconds: K3_DIAL_RETRY_S when set, else the stage `timeout`
    itself. Keyed to an existing constant rather than a bare magic number so it scales with how long
    a stage is allowed to take, and comfortably covers a neighbour that is still loading weights."""
    return DIAL_RETRY_S or float(timeout)


def _dial(host, port, timeout, retry_s=None):
    """Dial host:port, retrying for a time WINDOW while the peer is still coming up (see DIAL_RETRY_S).
    A stage binds+listens BEFORE it dials, so a generous window lets the whole chain resolve whatever
    order the stages were launched in. Each individual connect is capped at DIAL_CONNECT_TIMEOUT so a
    black-holing WAN box cannot consume the window in one blocked connect."""
    connect_timeout = min(float(timeout), DIAL_CONNECT_TIMEOUT)
    window = float(retry_s) if retry_s is not None else _dial_window(timeout)
    deadline = time.time() + window
    last = None
    while True:
        try:
            s = socket.create_connection((host, int(port)), timeout=connect_timeout)
            s.setsockopt(*NODELAY)
            return s
        except OSError as e:
            last = e
            if time.time() >= deadline:
                break
            time.sleep(0.25)
    raise RuntimeError(f"k3 stage: could not connect to {host}:{port} within {window:.0f}s "
                       f"({type(last).__name__}: {last})")


def _is_return_hello(msg):
    return isinstance(msg, dict) and msg.get("op") == "hello_return"


def _is_pred_hello(msg):
    return isinstance(msg, dict) and msg.get("op") == "hello_pred"


def serve_stage(stage, nstages, lo, hi, port, nxt=None, *, ckpt_dir=None, cfg=None, device=None,
                receipts=None, key_path=None, timeout=600.0, bind="127.0.0.1", ready=None,
                ret_relay=None):
    """Serve one contiguous layer block [lo:hi) in the fire-forward ring.

    head (stage 0)      embeds token ids -> (h, br), runs its layers, forwards the pair.
    middle              runs its layers on the received pair, forwards it.
    tail (stage n-1)    runs its layers, then final-norm + output-AttnRes + lm_head + SAMPLE, and
                        returns the token id on the coordinator-return channel.

    `ret_relay` (a multi-GPU TAIL box only) makes this the box-ingress RELAY: it is a normal forward
    stage for its own layers, but the coordinator-return tunnel terminates at THIS box's ENG_IN while
    the token is produced on the box's last-GPU stage, so it also bridges the return — dialing the box
    tail's return channel at `ret_relay` (loopback) and pumping every frame the tail sends straight out
    to the coordinator. Only the box ingress of a G>1 tail box passes it; every other path is untouched.

    `ready` is an optional threading.Event set once the stage is listening (the selftest waits on it
    instead of sleeping). `cfg` (a KimiLinearConfig) skips the on-disk config for an in-process ring;
    `ckpt_dir` loads real weights."""
    head, tail = (stage == 0), (stage == nstages - 1)
    cfg = cfg if cfg is not None else K3.config(ckpt_dir)
    dev = device or K3.dev
    receipts = RECEIPTS if receipts is None else receipts
    key_path = key_path or NODE_KEY_PATH
    st = K3.Stage(lo, hi, cfg, head=head, tail=tail, device=dev)
    if ckpt_dir is not None:
        st.load(ckpt_dir)
    node_key = load_or_make_node_key(key_path) if receipts else None
    print(f"[s{stage}] {st}", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((bind, port))
    srv.listen(4)
    # non-tail stages dial the forward leg at startup (a dead --next at boot is a launcher bug); the
    # sidecar tunnels FWD_RING to the next stage's inbound. Kernel accepts the handshake as soon as
    # the peer is listening, so this completes before the peer calls accept().
    nxt_sock = _dial(*nxt.rsplit(":", 1), timeout=timeout) if (not tail and nxt) else None
    if nxt_sock is not None and SWARM_TOKEN is not None:
        send_msg(nxt_sock, {"op": "hello_pred", "token": SWARM_TOKEN})
    print(f"[s{stage}] listening {bind}:{port}"
          + (f" -> {nxt}" if nxt_sock is not None else " (tail)")
          + (f" pub={pub_b64(node_key)}" if node_key is not None else ""), flush=True)
    if ready is not None:
        ready.set()

    if tail:
        _serve_tail(st, srv, lo, hi, node_key, receipts, timeout)
    elif ret_relay is not None:
        _serve_relay_ingress(st, stage, srv, nxt_sock, head, lo, hi, node_key, receipts, timeout,
                             ret_relay)
    else:
        _serve_forward(st, stage, srv, nxt_sock, head, lo, hi, node_key, receipts, timeout)


def _accept_pred(srv, timeout):
    """Accept the predecessor and consume its greeting. In token mode the predecessor greets with
    hello_pred (consumed); token-less, its first frame is job data (returned as `queued`)."""
    conn, _ = srv.accept()
    conn.setsockopt(*NODELAY)
    conn.settimeout(timeout)
    if SWARM_TOKEN is not None:
        hello = recv_msg(conn)
        if not _is_pred_hello(hello):
            raise RuntimeError(f"s: predecessor greeting missing/invalid: {hello!r}")
        return conn, None
    return conn, None


def _serve_forward(st, stage, srv, nxt_sock, head, lo, hi, node_key, receipts, timeout):
    """Head/middle serve loop: process a frame, forward it down the ring, never answer the pipe."""
    conn, queued = _accept_pred(srv, timeout)
    print(f"[s{stage}] predecessor connected", flush=True)
    _forward_loop(st, stage, nxt_sock, head, lo, hi, node_key, receipts, conn, queued)


def _forward_loop(st, stage, nxt_sock, head, lo, hi, node_key, receipts, conn, queued):
    """The head/middle process-and-forward loop over an already-accepted predecessor `conn` (with an
    optional already-read first frame `queued`). Split out of _serve_forward so the box-ingress relay
    reuses it verbatim for the DOWN direction while a pump thread carries the return UP."""
    signer = None
    with torch.no_grad():
        while True:
            msg = queued if queued is not None else recv_msg(conn)
            queued = None
            op = msg.get("op")
            if op == "reset":
                st.reset()
                signer = (ReceiptSigner(node_key, msg.get("swarm_id", "swarm"),
                                        msg.get("job_id", "job"), lo, hi, nonce=msg.get("nonce"))
                          if receipts else None)
                send_msg(nxt_sock, msg)                       # propagate reset UNCHANGED down the ring
                continue
            if op == "receipt":                               # job done: append my receipt, pass on
                if signer is not None:
                    msg.setdefault("receipts", []).append({"stage": stage, **signer.finalize()})
                send_msg(nxt_sock, msg)
                continue
            if op == "step":
                if head:
                    h, br = st.embed(msg["ids"])              # token ids -> (h, seed block_residual)
                else:
                    h, br = msg["h"], msg["br"]
                start_pos = int(msg["start_pos"])
                in_b = _payload_bytes(h, br) if signer is not None else None
                h, br = st.forward(h, br, start_pos)
                if signer is not None:
                    signer.observe(in_b, _payload_bytes(h, br))
                send_msg(nxt_sock, {"op": "step", "h": h.contiguous(), "br": br.contiguous(),
                                    "start_pos": start_pos})
                continue
            if op == "stop":
                try:
                    send_msg(nxt_sock, msg)                   # tear the ring down front-to-back
                except OSError:
                    pass
                conn.close()
                return
            raise RuntimeError(f"s{stage}: unknown op {op!r}")


def _tail_bringup(srv, timeout):
    """Identify the two tail-inbound streams WITHOUT ever block-recv'ing a silent one. The
    predecessor connection is accepted early (the upstream stage dialed at launch) but stays silent
    until the first job frame flows the ring; the coordinator-return greets immediately. So select
    on the accepted set and only recv a connection that is READABLE, exactly like m25_pipe's
    _tail_accept — a plain accept-then-recv here deadlocks: the tail would block reading the silent
    predecessor while the coordinator blocks waiting for its ret_ok. Returns (ret, pred, queued)."""
    ret = pred = queued = None
    pending = []
    while ret is None or pred is None:
        ready, _, _ = select.select([srv] + pending, [], [])
        if srv in ready:
            conn, _ = srv.accept()
            conn.setsockopt(*NODELAY)
            conn.settimeout(timeout)
            pending.append(conn)
            continue
        conn = next(c for c in pending if c in ready)
        pending.remove(conn)
        first = recv_msg(conn)
        if _is_return_hello(first):
            ret = conn
            send_msg(ret, "ret_ok")                           # coordinate.py blocks on this ack
        elif _is_pred_hello(first):                           # token-mode predecessor greeting
            pred = conn
        else:                                                 # token-less predecessor: first frame is job data
            pred, queued = conn, first
    return ret, pred, queued


def _serve_tail(st, srv, lo, hi, node_key, receipts, timeout):
    """Tail serve loop. Accepts BOTH inbound streams (predecessor + coordinator-return) on the one
    engine port, classified by the hello_return greeting, then serves: run the block, sample, and
    send the token id back on the return channel."""
    ret, pred, queued = _tail_bringup(srv, timeout)
    print("[tail] predecessor + coord-return connected", flush=True)

    signer = None
    temp, gen = 0.0, None                                    # sampling arm, (re)set per job by the reset
    with torch.no_grad():
        while True:
            msg = queued if queued is not None else recv_msg(pred)
            queued = None
            op = msg.get("op")
            if op == "reset":
                st.reset()
                signer = (ReceiptSigner(node_key, msg.get("swarm_id", "swarm"),
                                        msg.get("job_id", "job"), lo, hi, nonce=msg.get("nonce"))
                          if receipts else None)
                temp = float(msg.get("temp", 0.0))
                gen = (torch.Generator(device="cpu").manual_seed(int(msg["seed"]))
                       if temp > 0 and msg.get("seed") is not None else None)
                send_msg(ret, "ok")
                continue
            if op == "receipt":
                if signer is not None:
                    msg.setdefault("receipts", []).append({"stage": "tail", **signer.finalize()})
                send_msg(ret, msg.get("receipts", []))
                continue
            if op == "step":
                h, br = msg["h"], msg["br"]
                start_pos = int(msg["start_pos"])
                in_b = _payload_bytes(h, br) if signer is not None else None
                h, br = st.forward(h, br, start_pos)
                if signer is not None:
                    signer.observe(in_b, _payload_bytes(h, br))
                logits = st.logits(h, br)                     # final-norm + output-AttnRes + lm_head
                tid = sample_token(logits[0, -1], temp, gen)
                send_msg(ret, {"token": tid})
                continue
            if op == "stop":
                try:
                    send_msg(ret, {"token": None})
                except OSError:
                    pass
                pred.close()
                if ret is not None:
                    ret.close()
                return
            raise RuntimeError(f"tail: unknown op {op!r}")


def _dial_return(addr, timeout):
    """Dial an intra-box return channel to the box TAIL and greet it exactly as the coordinator
    would — hello_return, then block on its ret_ok. Retries for the full dial window while the tail
    is still coming up (it loads its weights after the ingress binds — a box tail holding several
    KDA-MoE layers is minutes, not seconds). Returns the socket the tail replies on (reset ack,
    tokens, receipts)."""
    host, port = addr.rsplit(":", 1)
    s = _dial(host, port, timeout)
    s.settimeout(timeout)
    send_msg(s, {"op": "hello_return", "token": SWARM_TOKEN} if SWARM_TOKEN is not None
             else {"op": "hello_return"})
    recv_msg(s)                                                # the box tail acks (ret_ok)
    return s


def _serve_relay_ingress(st, stage, srv, nxt_sock, head, lo, hi, node_key, receipts, timeout,
                         ret_relay):
    """Box-ingress serve loop for a MULTI-GPU TAIL box. The sidecar delivers the coordinator-return
    tunnel to this box's ENG_IN (same one -inbound as the predecessor's job frames), but the token is
    produced on the box's LAST-GPU stage. So this stage:
      * runs its own layers and forwards frames down the intra-box loopback chain — byte-identical to
        a plain forward stage (it appends its receipt too),
      * bridges the return: it dials the box tail's return over loopback (`ret_relay`) and pumps every
        frame the tail sends (reset ack, tokens, the receipt list) straight out to the coordinator.
    The box tail's serve loop is UNCHANGED — it just gets its coordinator-return dialed from here
    instead of the WAN. Reached only when a G>1 box holds the global tail (a 1-GPU tail box has
    box ingress == box tail == the global tail and serves through _serve_tail)."""
    ret_up, pred, queued = _tail_bringup(srv, timeout)         # coordinator-return + predecessor at ENG_IN
    ret_down = _dial_return(ret_relay, timeout)                # our return leg to the box tail (loopback)

    def _pump():
        try:
            while True:
                send_msg(ret_up, recv_msg(ret_down))           # box tail -> coordinator, frame for frame
        except OSError:                                        # ConnectionError on stop / ring torn down
            pass
    threading.Thread(target=_pump, daemon=True).start()
    print(f"[s{stage}] relay ingress: coord-return <-> box tail {ret_relay}", flush=True)
    _forward_loop(st, stage, nxt_sock, head, lo, hi, node_key, receipts, pred, queued)


# ── coordinator: drive greedy generation over the ring ─────────────────────────────────────────────

def _hostport(s):
    h, _, p = s.rpartition(":")
    return h or "127.0.0.1", int(p)


def connect_ring(head, tail, timeout=600.0, token=None, retry_s=300):
    """Dial the head engine (pipe) + the return tunnel to the tail (ret), with retries — the daemon
    starts the coordinator while other stages may still be pulling weights. Mirrors
    shard.coordinate.connect_ring: hello_return classifies the tail-side stream, the token (env-only)
    rides both greetings when set."""
    deadline = time.time() + retry_s
    last = None
    while time.time() < deadline:
        pipe = ret = None
        try:
            pipe = socket.create_connection(_hostport(head), timeout=timeout)
            pipe.setsockopt(*NODELAY)
            ret = socket.create_connection(_hostport(tail), timeout=timeout)
            ret.setsockopt(*NODELAY)
            ret.settimeout(timeout)
            if token:
                send_msg(pipe, {"op": "hello_pred", "token": token})
                send_msg(ret, {"op": "hello_return", "token": token})
            else:
                send_msg(ret, {"op": "hello_return"})
            recv_msg(ret)                                     # tail acks the return channel (ret_ok)
            return pipe, ret
        except Exception as e:                                # noqa: BLE001 — any dial fault = retry
            last = e
            for s in (pipe, ret):
                if s is not None:
                    try: s.close()
                    except OSError: pass
            time.sleep(min(2.0, max(0.25, deadline - time.time())))
    raise ConnectionError(f"k3 ring not reachable after {retry_s}s: {type(last).__name__}: {last}")


def coordinate(pipe, ret, prompt_ids, max_new, *, eos_ids=(), nonce=None, swarm_id="swarm",
               job_id="job", layer_count=None, receipts=False, temp=0.0, seed=0, timeout=600.0,
               on_token=None):
    """Greedy sequential decode over the fire-forward ring. Weightless: the head embeds, the tail
    samples, and this loop only threads token ids and the settlement nonce over the sockets.

    Returns {ok, tokens, prompt_tokens, receipts, receipts_ok}. `receipts` sweeps the ring once at
    the end and verifies coverage against `layer_count` and the job nonce (fail-closed, C10)."""
    ret.settimeout(timeout)
    send_msg(pipe, {"op": "reset", "swarm_id": swarm_id, "job_id": job_id, "nonce": nonce,
                    "temp": float(temp), "seed": int(seed)})
    ack = recv_msg(ret)                                       # the tail acks the whole ring is reset
    if not (ack == "ok" or (isinstance(ack, dict) and ack.get("ok"))):
        raise RuntimeError(f"k3 ring reset not acked: {ack!r}")

    eos = set(eos_ids)
    ids = list(prompt_ids)
    pos = 0
    toks = []
    for _ in range(max_new):
        send_msg(pipe, {"op": "step", "ids": [ids], "start_pos": pos})
        pos += len(ids)
        rep = recv_msg(ret)
        tid = int(rep["token"]) if isinstance(rep, dict) else int(rep)
        toks.append(tid)
        if on_token is not None:
            on_token(tid)
        if tid in eos:
            break
        ids = [tid]

    recs, receipts_ok = [], None
    if receipts:
        send_msg(pipe, {"op": "receipt", "receipts": []})
        recs = recv_msg(ret) or []
        if recs and layer_count is not None:
            wired = [wire_receipt(r) for r in recs]
            try:
                verify_coverage(wired, int(layer_count), expected_nonce=nonce, check_chain=True)
                receipts_ok = True
            except ReceiptError:
                receipts_ok = False
    return {"ok": True, "tokens": toks, "prompt_tokens": len(prompt_ids),
            "receipts": recs, "receipts_ok": receipts_ok}


# ── layer tiling: consume K3_PROFILE via plan_ring ─────────────────────────────────────────────────

K3_MODEL_ID = "moonshotai/Kimi-K3-MXFP4"


def even_tiling(n_layers, nstages):
    """Contiguous, balanced tiling of [0:n_layers) across nstages — the front stages take the extra
    layer when it doesn't divide. The offline fallback (and the tiny-ring test split) when there is
    no measured pool to plan against."""
    if nstages < 1 or nstages > n_layers:
        raise ValueError(f"cannot tile {n_layers} layers over {nstages} stages")
    base, extra = divmod(n_layers, nstages)
    ranges, lo = [], 0
    for k in range(nstages):
        hi = lo + base + (1 if k < extra else 0)
        ranges.append((lo, hi))
        lo = hi
    return ranges


def plan_layer_ranges(nodes, rtt, model_id=K3_MODEL_ID, *, per_gpu=False, **kw):
    """Place the 93-layer ring across a measured pool. Delegates to shard.plan.plan_ring with the
    K3 engine profile (K3_PROFILE — VRAM/ms/boundary bytes measured on real weights), so k3_pipe
    never re-derives the memory model. Returns [{id, index, lo, hi, head, tail, layers}, ...] or
    None when the pool cannot hold the model.

    plan_ring places layers on NODES (a box announces its AGGREGATE free VRAM across its GPUs). With
    per_gpu=True each node's block is then split into one contiguous sub-block per local GPU (read
    from the node's "gpus" field, default 1) via split_stages_to_gpus — the per-(box, gpu) placement
    the multi-GPU launcher consumes. per_gpu=False (default) returns the node plan unchanged, so a
    single-GPU ring is byte-identical."""
    try:
        from plan import plan_ring
    except ImportError:
        from shard.plan import plan_ring
    plan = plan_ring(nodes, rtt, model=model_id, **kw)
    if not plan:
        return None
    if not per_gpu:
        return plan["stages"]
    return split_stages_to_gpus(plan["stages"], {nd["id"]: int(nd.get("gpus", 1)) for nd in nodes})


# ── sidecar wiring: the direct-return topology (payload-agnostic, mirrors m25_scatter_pipe) ─────────

def ring_sidecar_spec(k, n, maddrs, ret_maddr=None):
    """The per-stage sidecar forward/inbound/allow set for the direct-return topology, byte-identical
    in shape to m25_scatter_pipe's (the sidecar carries {h, br} exactly as it carries M2.5's single
    hidden — proven). Returns (inbound, forwards, allow):
      * every non-tail stage forwards FWD_RING -> the next stage's maddr,
      * the HEAD also forwards FWD_RET -> the tail (the coordinator-return tunnel),
      * every stage but the head takes an -inbound to its local engine, and admits only its
        predecessor's PeerId (the tail also admits the head's — the return stream arrives as head)."""
    inbound = f"127.0.0.1:{ENG_IN}" if k > 0 else ""
    forwards = []
    if k < n - 1:
        forwards.append(f"127.0.0.1:{FWD_RING}={maddrs[k + 1]}")
    if k == 0:
        forwards.append(f"127.0.0.1:{FWD_RET}={ret_maddr or maddrs[-1]}")
    allow = None
    if k > 0:
        allow = [_peerid_of(maddrs[k - 1])]
        if k == n - 1:
            allow.append(_peerid_of(maddrs[0]))
    return inbound, forwards, allow


def _peerid_of(maddr):
    return maddr.rsplit("/p2p/", 1)[-1].split("/p2p-circuit")[0]


def stage_launch_cmd(stage, nstages, lo, hi, *, model_dir="/root/k3", receipts=False,
                     device="cuda", token=None, extra_env="", gpu=None, port=None, nxt_addr=None,
                     ret_relay=None):
    """The shell command a launcher runs to start ONE k3 engine stage over the sidecar. Fully
    DETACHED in every shape (setsid + fd redirect), mirroring m25_scatter_pipe.stage_cmd — see the
    note by the return.

    Single-GPU box (gpu/port/nxt_addr all unset): the single-stage-per-box form — the stage binds
    ENG_IN and a non-tail stage forwards to the LOCAL sidecar forward leg (FWD_RING); the tail has
    none. The engine binds loopback (M25_ENGINE_BIND) so only the local sidecar can reach it.

    Multi-GPU box: pass this stage's local
      * `gpu`      — the local GPU index. Sets CUDA_VISIBLE_DEVICES=<gpu> so the process sees ONLY
                     that card, as cuda:0 — which is why k3_stage's K3_DEV=cuda and k3_moe_mxfp4's
                     torch.cuda.set_device(0) are already correct with NO engine change (the
                     least-invasive device pin). Also gives it a distinct K3_MOE_PORT so co-located
                     vLLM MoE groups don't collide on MASTER_PORT.
      * `port`     — its local engine port (ENG_IN for the box's first stage, else ENG_LOCAL_BASE+idx).
      * `nxt_addr` — where it forwards: a LOOPBACK 127.0.0.1:<next local port> to the next GPU on this
                     box, or the WAN 127.0.0.1:FWD_RING leg for the box's last stage.
      * `ret_relay` — set ONLY on the ingress of a G>1 TAIL box: the loopback 127.0.0.1:<box tail local
                     port> it bridges the coordinator-return to (--ret-relay). Unset everywhere else.
    """
    port = port or ENG_IN
    if nxt_addr is not None:
        nxt = f"--next {nxt_addr}"
    else:
        nxt = "" if stage == nstages - 1 else f"--next 127.0.0.1:{FWD_RING}"
    rr = f"--ret-relay {ret_relay} " if ret_relay else ""
    rc = "SHARD_RECEIPTS=1 " if receipts else ""
    tk = f"SHARD_SWARM_TOKEN={token} " if token else ""
    cvd = (f"CUDA_VISIBLE_DEVICES={int(gpu)} K3_MOE_PORT={K3_MOE_PORT_BASE + int(gpu)} "
           if gpu is not None else "")
    # DETACH like m25_scatter_pipe.stage_cmd (Bug 1, live K3 ring 2026-07-29): a bare `nohup <cmd> &`
    # over ssh kept the channel open on the engine's child fds and HUNG the launcher, so a parallel
    # launch of all B*G stages never returned the ssh session. setsid + a full fd redirect fully
    # detaches the engine and ssh returns instantly; its stdout goes to a PER-PORT log so co-located
    # stages on one box never clobber each other's.
    log = f"/root/k3_stage_{port}.log"
    inner = (f"python3 /root/k3_pipe.py stage --stage {stage} --nstages {nstages} --lo {lo} --hi {hi} "
             f"--port {port} {nxt} {rr}--dir {model_dir} > {log} 2>&1")
    return (f"{rc}{tk}{cvd}{extra_env}K3_DIR={model_dir} K3_DEV={device} M25_ENGINE_BIND=127.0.0.1 "
            f"setsid bash -c '{inner}' </dev/null >/dev/null 2>&1 &")


# ── multi-GPU per box: split node blocks to GPUs, wire loopback intra-box + WAN inter-box ───────────

def local_eng_port(local_index):
    """The engine port for the `local_index`-th stage on a box. Index 0 -> ENG_IN (the box ingress
    the sidecar -inbound and the head coordinator dial); 1.. -> ENG_LOCAL_BASE+idx (loopback only)."""
    return ENG_IN if local_index == 0 else ENG_LOCAL_BASE + local_index


def split_stages_to_gpus(node_stages, gpus):
    """Split a head-first NODE plan into one contiguous sub-block per local GPU.

    `node_stages` is plan_ring/plan_layer_ranges output (or any head-first [{id, lo, hi}...] whose
    blocks tile the model): each entry is a BOX holding a contiguous layer range. `gpus` gives a
    box's GPU count — an int applied to every box, a list aligned to node_stages, or a dict keyed by
    node id. A box's range is tiled EVENLY across its GPUs (front GPUs take the extra layer), so a
    G-GPU box becomes G contiguous sub-blocks. A 1-GPU box is returned unchanged (its one sub-block
    == the node block), so a single-GPU ring is byte-identical.

    Pure tiling: it does NOT enforce a per-GPU layer cap — VRAM feasibility is plan_ring's job
    upstream (the caller announces a box's aggregate VRAM). Returns a FLAT head-first list of
    per-GPU stage specs:
      {"id", "box_index", "gpu", "local_index", "nlocal", "global_index", "nstages",
       "lo", "hi", "layers", "head", "tail", "box_head", "box_tail"}
    where head/tail are the GLOBAL ring ends (the head embeds, the tail samples) and box_head/box_tail
    mark the first/last local stage on a box (its WAN-facing ends)."""
    def _g(k, nd):
        if isinstance(gpus, dict):
            return int(gpus.get(nd["id"], 1))
        if isinstance(gpus, (list, tuple)):
            return int(gpus[k])
        return int(gpus)
    subs = []
    for k, nd in enumerate(node_stages):
        G = _g(k, nd)
        if G < 1:
            raise ValueError(f"box {nd.get('id', k)!r} announced {G} GPUs")
        for j, (slo, shi) in enumerate(even_tiling(nd["hi"] - nd["lo"], G)):
            subs.append({"id": nd["id"], "box_index": k, "gpu": j, "local_index": j, "nlocal": G,
                         "lo": nd["lo"] + slo, "hi": nd["lo"] + shi, "layers": shi - slo,
                         "box_head": j == 0, "box_tail": j == G - 1})
    n = len(subs)
    for g, s in enumerate(subs):
        s["global_index"], s["nstages"] = g, n
        s["head"], s["tail"] = g == 0, g == n - 1
    return subs


def box_stage_wiring(sub):
    """(eng_port, next_addr, link) for one per-GPU stage from split_stages_to_gpus.

    eng_port: the box's first local stage binds ENG_IN (the sidecar -inbound / head coordinator reach
    it); the rest bind loopback-only local ports. next_addr / link:
      * global tail            -> (None, "tail")      no forward; it samples + returns to the coordinator
      * box_tail (not tail)    -> (FWD_RING, "wan")    the box's WAN egress -> next box via the sidecar
      * otherwise              -> (next local port, "loopback")   hand off to the next GPU on THIS box
    """
    eng_port = local_eng_port(sub["local_index"])
    if sub["tail"]:
        return eng_port, None, "tail"
    if sub["box_tail"]:
        return eng_port, f"127.0.0.1:{FWD_RING}", "wan"
    return eng_port, f"127.0.0.1:{local_eng_port(sub['local_index'] + 1)}", "loopback"


def box_return_relay(sub, tail_box_index):
    """The intra-box return-relay target for one per-GPU stage, or None. Only the box INGRESS
    (box_head) of a MULTI-GPU box that holds the global tail relays: the coordinator-return tunnel
    lands at that box's ENG_IN, but the token is produced on its last-GPU stage — so the ingress
    bridges them, dialing the box tail's return over loopback and pumping it out to the coordinator.
    Returns '127.0.0.1:<box tail local port>' for that one stage, None for every other. A 1-GPU tail
    box has box_head == box_tail == the global tail and needs no relay (returns None)."""
    if sub["box_head"] and sub["nlocal"] > 1 and sub["box_index"] == tail_box_index:
        return f"127.0.0.1:{local_eng_port(sub['nlocal'] - 1)}"
    return None


def box_ring_launch(node_stages, gpus, box_maddrs=None, *, model_dir="/root/k3", receipts=False,
                    token=None, ret_maddr=None):
    """The full launch plan for a ring of B boxes x G GPUs = B*G stages but only B WAN hops.

    node_stages: head-first NODE plan (plan_layer_ranges without per_gpu); gpus: per-box GPU count
    (see split_stages_to_gpus); box_maddrs: the B box libp2p multiaddrs (head-first) for the sidecar
    wiring — omit to skip it (topology-only). Returns:
      {"stages":   [{**sub, "eng_port", "next", "link", "ret_relay", "cmd"}...],  # one per (box, gpu)
       "sidecars": [(inbound, forwards, allow)...] | None}            # per box, ring_sidecar_spec at
                                                                      # BOX granularity

    The intra-box GPU stages hand off over loopback and NEVER touch the sidecar, so the sidecar sees
    only the B boxes: ring_sidecar_spec over B nodes gives B-1 forward hops + 1 head->tail return =
    B WAN hops, whatever G is.

    The RETURN path is closed for a G>1 TAIL box: the coordinator-return tunnel is delivered by the
    sidecar to the tail box's -inbound (ENG_IN = its FIRST local stage), but the token is produced on
    its LAST local GPU. That box's ingress carries a `ret_relay` (--ret-relay) and bridges the two over
    loopback (box_return_relay / _serve_relay_ingress) — so the sidecar wiring stays BOX-granular and
    identical to M2.5, and only that one engine gets the relay flag. A 1-GPU tail box (box ingress ==
    box tail == the global tail, ret_relay None) serves the coordinator-return directly, as before."""
    subs = split_stages_to_gpus(node_stages, gpus)
    tail_box = subs[-1]["box_index"]                           # the box holding the global tail
    stages = []
    for sub in subs:
        eng_port, nxt, link = box_stage_wiring(sub)
        ret_relay = box_return_relay(sub, tail_box)
        cmd = stage_launch_cmd(sub["global_index"], sub["nstages"], sub["lo"], sub["hi"],
                               model_dir=model_dir, receipts=receipts, token=token,
                               gpu=sub["gpu"], port=eng_port, nxt_addr=nxt, ret_relay=ret_relay)
        stages.append({**sub, "eng_port": eng_port, "next": nxt, "link": link,
                       "ret_relay": ret_relay, "cmd": cmd})
    B = len(node_stages)
    sidecars = ([ring_sidecar_spec(b, B, box_maddrs, ret_maddr=ret_maddr) for b in range(B)]
                if box_maddrs is not None else None)
    return {"stages": stages, "sidecars": sidecars}


# ── offline selftest: the whole ring on localhost, CPU, no spend ───────────────────────────────────

def _tiny_config(n_layers=6):
    """A small random-init K3 that exercises head + middles + tail with a MIXED KDA/MLA ring and an
    AttnRes append landing MID-ring. kda_layers is 1-BASED (is_kda_layer checks layer_idx+1). At
    n_layers=6: KDA at 0,1,3,4 and MLA at 2,5 (a ~3:1 interleave like the real 93), block_size 2 ->
    appends at layers 0,2,4 (three blocks by the end), first_k_dense_replace=1 -> layer 0 dense."""
    K3.ref()
    from kimi_k3_ref.configuration_kimi_k3 import KimiLinearConfig
    kda = [i + 1 for i in range(n_layers) if i % 3 != 2]     # 1-based; MLA every 3rd layer
    full = [i + 1 for i in range(n_layers) if i % 3 == 2]
    cfg = KimiLinearConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=n_layers,
        num_attention_heads=4, num_key_value_heads=4, hidden_act="situ",
        activation_situ_beta=4.0, activation_situ_linear_beta=25.0, rms_norm_eps=1e-5,
        moe_intermediate_size=24, num_experts=6, num_experts_per_token=2, num_shared_experts=1,
        first_k_dense_replace=1, moe_layer_freq=1, routed_expert_hidden_size=16,
        latent_moe_use_norm=True, moe_renormalize=True, routed_scaling_factor=1.0,
        moe_router_activation_func="sigmoid", num_expert_group=1, topk_group=1,
        q_lora_rank=16, kv_lora_rank=8, qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8,
        mla_use_nope=True, mla_use_output_gate=True, attn_res_block_size=2,
        linear_attn_config={"kda_layers": kda, "full_attn_layers": full, "head_dim": 8,
                            "num_heads": 4, "short_conv_kernel_size": 4,
                            "use_full_rank_gate": True, "gate_lower_bound": -5.0},
        max_position_embeddings=512, tie_word_embeddings=False, pad_token_id=0)
    cfg._attn_implementation = "eager"
    return cfg


def write_tiny_checkpoint(d, cfg, model=None):
    """A real tiny safetensors checkpoint in K3's OWN namespace (language_model.model.layers.N.),
    A_log padded the way the real checkpoint ships it — so every stage server exercises the real
    Stage.load() fetch namespace + A_log slice. Returns (dir, model) so the caller can build the
    same-weights reference baseline."""
    import safetensors.torch as ST
    from kimi_k3_ref import modeling_kimi_linear as M
    if model is None:
        torch.manual_seed(1234)
        model = M.KimiLinearForCausalLM(cfg).eval()
        _seed_unreached(model)
    os.makedirs(d, exist_ok=True)
    ns, outer = "language_model.model", "language_model"
    pad = cfg.linear_attn_config["num_heads"] + 32
    tensors = {}
    for k, v in model.state_dict().items():
        name = f"{outer}.{k}" if k.startswith("lm_head") else f"{ns}.{k[len('model.'):]}"
        if k.endswith("self_attn.A_log"):
            v = torch.cat([v, torch.full((pad - v.shape[0],), 99.0)])
        tensors[name] = v.clone().contiguous()
    ST.save_file(tensors, f"{d}/model.safetensors")
    json.dump({"weight_map": {k: "model.safetensors" for k in tensors}},
              open(f"{d}/model.safetensors.index.json", "w"))
    json.dump({"model_type": "kimi_k3", "text_config": cfg.to_dict()}, open(f"{d}/config.json", "w"))
    return d, model


def _seed_unreached(model):
    """Seed the parameters Moonshot's initializer doesn't reach (KimiDeltaAttention.dt_bias is a bare
    nn.Parameter) — else a random-init fixture is occasionally NaN. See tests/test_k3_stage.py."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("dt_bias"):
                p.normal_(0.0, 0.02)


def selftest(nstages=4, n_layers=6, prompt=(3, 9, 17, 2, 41), max_new=5, tail_box_g=1):
    """Offline, CPU, no spend: a full k3_pipe ring on localhost (head embed + middles + tail sample +
    weightless coordinator over real shard.transport sockets) decodes greedily, and the token stream
    is checked BIT-IDENTICAL against Moonshot's own KimiLinearForCausalLM whole-model decode.

    `tail_box_g` > 1 folds the last G stages into ONE multi-GPU tail box: the coordinator-return
    terminates at that box's ingress and is bridged over loopback to the box tail (the --ret-relay
    path a real G>1 tail box runs), proving the return relay against the same parity bar."""
    import tempfile
    os.environ["SHARD_RECEIPTS"] = "1"
    global RECEIPTS
    RECEIPTS = True
    cfg = _tiny_config(n_layers)
    d = tempfile.mkdtemp(prefix="k3pipe_")
    d, model = write_tiny_checkpoint(d, cfg)
    os.environ["K3_DIR"] = d

    ref_tokens = _reference_tokens(cfg, model, list(prompt), max_new)
    ranges = even_tiling(n_layers, nstages)
    base = 29661
    ports = [base + i for i in range(nstages)]
    relay_i = nstages - tail_box_g                             # ingress of the multi-GPU tail box (>1)
    events = [threading.Event() for _ in range(nstages)]
    threads = []
    for i, (lo, hi) in enumerate(ranges):
        nxt = None if i == nstages - 1 else f"127.0.0.1:{ports[i + 1]}"
        ret_relay = f"127.0.0.1:{ports[-1]}" if (tail_box_g > 1 and i == relay_i) else None
        t = threading.Thread(target=serve_stage, kwargs=dict(
            stage=i, nstages=nstages, lo=lo, hi=hi, port=ports[i], nxt=nxt, ckpt_dir=d,
            device="cpu", receipts=True, key_path=f"{d}/s{i}.key", ret_relay=ret_relay,
            ready=events[i]), daemon=True)
        t.start()
        threads.append(t)
    for e in events:
        e.wait(30)

    tail_port = ports[relay_i] if tail_box_g > 1 else ports[-1]   # return lands at the tail box ingress
    pipe, ret = connect_ring(f"127.0.0.1:{ports[0]}", f"127.0.0.1:{tail_port}", timeout=60)
    r = coordinate(pipe, ret, list(prompt), max_new, nonce="settle-nonce-0", receipts=True,
                   layer_count=n_layers)
    send_msg(pipe, {"op": "stop"})

    checks = {
        "full_ring_matches_reference": (r["tokens"] == ref_tokens),
        "receipts_settle": (r["receipts_ok"] is True),
        "coverage_tiles_all_layers": (sorted((c["layer_start"], c["layer_end"])
                                              for c in r["receipts"]) == _expected_cover(ranges)),
    }
    tag = f", tail box = {tail_box_g} GPUs (return relay)" if tail_box_g > 1 else ""
    print("\n=== K3 pipe offline selftest (CPU tiny config) ===")
    print(f"  ring: {nstages} stages over {n_layers} layers {ranges}{tag}")
    print(f"  tokens (ring) {r['tokens']}\n  tokens (ref)  {ref_tokens}")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(checks.values())
    print(f"\n  {'ALL PASS' if ok else 'FAILURES PRESENT'}", flush=True)
    os._exit(0 if ok else 1)


def _expected_cover(ranges):
    return sorted((lo, hi) for lo, hi in ranges)


def _reference_tokens(cfg, model, prompt, max_new):
    """Greedy token ids from Moonshot's own whole model — the parity baseline. Patches only the mask
    helper the reference borrows from transformers to the SAME architecture mask a Stage builds (see
    tests/test_k3_stage._adapt_reference_whole_model), so the comparison is pinned to the architecture
    rather than to a transformers release."""
    from kimi_k3_ref import modeling_kimi_linear as M

    def _mask(**kw):
        emb = kw["input_embeds"]
        start, q = int(kw["cache_position"][0]), emb.shape[1]
        return K3._causal_mask(q, start + q, start, emb.dtype, emb.device)
    M.create_causal_mask = _mask
    # KimiLinearModel.__init__ overwrote the requested attention with flash_attention_2 (right on an
    # 8xB300, fatal on CPU). The config is shared with every Stage, so pin it eager for both sides.
    cfg._attn_implementation = "eager"
    cache = M.KimiDynamicCache(config=cfg)
    ids, toks = list(prompt), []
    for _ in range(max_new):
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([ids]), past_key_values=cache, use_cache=True).logits
        toks.append(int(lg[0, -1].argmax()))
        ids = [toks[-1]]
    return toks


# ── CLI ────────────────────────────────────────────────────────────────────────────────────────────

def _emit(tag, **fields):
    sys.stdout.write(tag + " " + json.dumps(fields) + "\n")
    sys.stdout.flush()


def _coord_cli(a):
    """The node-daemon serving entrypoint (mirrors shard.coordinate): dial the ring, read jobs as
    JSON lines on stdin, drive greedy decode, and emit the SHARD_JOB_* stdout contract."""
    os.environ.setdefault("K3_DIR", a.dir)
    cfg = K3.config(a.dir)
    layer_count = cfg.num_hidden_layers
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.dir, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        _emit("SHARD_JOB_FATAL", error=f"tokenizer load failed: {e}")
        return 1
    eos = tok.eos_token_id
    eos_ids = tuple(eos) if isinstance(eos, (list, tuple)) else ((eos,) if eos is not None else ())
    pipe, ret = connect_ring(a.head, a.tail, timeout=a.timeout, token=SWARM_TOKEN,
                             retry_s=a.connect_retry)
    _emit("SHARD_COORD_READY", head=a.head, tail=a.tail, receipts=a.receipts)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
            job_id = job["jobId"]
        except (ValueError, KeyError) as e:
            _emit("SHARD_JOB_FATAL", error=f"unparseable job line: {e}")
            continue
        max_new = max(1, min(int(job.get("maxNew") or 512), 4096))
        _emit("SHARD_JOB_START", jobId=job_id, maxNew=job.get("maxNew"))
        state = {"n": 0}

        def _on_token(_tid, _job=job_id, _st=state):
            _st["n"] += 1
            _emit("SHARD_JOB_TOKEN", jobId=_job, delta=_st["n"])
        try:
            prompt_ids = tok.apply_chat_template(job["messages"], add_generation_prompt=True) \
                if hasattr(tok, "apply_chat_template") and job.get("messages") else job["promptIds"]
            r = coordinate(pipe, ret, prompt_ids, max_new, eos_ids=eos_ids,
                           nonce=job.get("nonce"), swarm_id=job.get("swarmId") or "swarm",
                           job_id=job_id, layer_count=layer_count, receipts=a.receipts,
                           temp=float(job.get("temperature", 0.0)), timeout=a.timeout,
                           on_token=_on_token)
            _emit("SHARD_JOB_DONE", jobId=job_id, ok=True,
                  response=tok.decode(r["tokens"], skip_special_tokens=True),
                  tokensGenerated=len(r["tokens"]),
                  receipts=[wire_receipt(rr) for rr in (r["receipts"] or [])],
                  receiptsOk=r["receipts_ok"], nonce=job.get("nonce"))
        except Exception as e:  # noqa: BLE001
            _emit("SHARD_JOB_FATAL", jobId=job_id, error=f"{type(e).__name__}: {e}")
            return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stage", help="serve one K3 layer block in the fire-forward ring")
    s.add_argument("--stage", type=int, required=True)
    s.add_argument("--nstages", type=int, required=True)
    s.add_argument("--lo", type=int, required=True)
    s.add_argument("--hi", type=int, required=True)
    s.add_argument("--port", type=int, default=ENG_IN)
    s.add_argument("--next", default=None, dest="next")
    s.add_argument("--ret-relay", default=None, dest="ret_relay",
                   help="multi-GPU tail box ingress only: loopback addr of the box tail's return "
                        "channel to bridge the coordinator-return to")
    s.add_argument("--dir", default=os.environ.get("K3_DIR", "/root/k3"))
    s.add_argument("--device", default=None)
    s.add_argument("--bind", default=os.environ.get("M25_ENGINE_BIND", "127.0.0.1"))
    s.add_argument("--receipts", action="store_true")

    c = sub.add_parser("coord", help="drive greedy jobs over a formed ring (stdin JSON lines)")
    c.add_argument("--head", default=f"127.0.0.1:{ENG_IN}")
    c.add_argument("--tail", default=f"127.0.0.1:{FWD_RET}")
    c.add_argument("--dir", default=os.environ.get("K3_DIR", "/root/k3"))
    c.add_argument("--receipts", action="store_true")
    c.add_argument("--timeout", type=int, default=600)
    c.add_argument("--connect-retry", type=int, default=300, dest="connect_retry")

    sub.add_parser("selftest", help="offline CPU full-ring parity proof (no GPU, no spend)")
    sub.add_parser("selftest-relay",
                   help="offline CPU parity proof with a MULTI-GPU tail box (return relay path)")
    a = ap.parse_args()

    if a.cmd == "stage":
        serve_stage(a.stage, a.nstages, a.lo, a.hi, a.port, nxt=a.next, ckpt_dir=a.dir,
                    device=a.device, receipts=(a.receipts or RECEIPTS), bind=a.bind,
                    ret_relay=a.ret_relay)
    elif a.cmd == "coord":
        sys.exit(_coord_cli(a))
    elif a.cmd == "selftest":
        selftest()
    elif a.cmd == "selftest-relay":
        selftest(nstages=3, tail_box_g=2)


if __name__ == "__main__":
    main()
