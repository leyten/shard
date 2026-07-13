"""shard: speculative decoding over the N-node pipeline (specdec.py x pipeline.py).

specdec.py proved draft-and-verify over a 2-node split. pipeline.py proved an
N-stage split big enough to hold a 120B target (gpt-oss: partial per-node loading,
sliding-window masks, MXFP4). this puts them together: a small draft proposes K
tokens locally on the head, and the *distributed* target verifies all K in ONE
traversal of the whole chain -- the same single WAN round-trip that plain decode
spends on a single token now commits several. greedy acceptance, so the output is
token-for-token identical to plain pipeline decode; the target is never made whole
on any node.

the draft runs on its own GPU on the head (a 120B stage already fills a 24GB card,
so the draft can't share it); every other node is unchanged from pipeline.py and
holds only its block of the target. the verify op carries a lazy `crop`: the prior
round's rejected tokens are rolled back from every node's cache on the next verify,
piggybacked, so a round costs exactly one round-trip end to end.

  # every node shares one swarm secret (same value on each box):
  export SHARD_PSK=$(openssl rand -hex 32)
  # tail (stage N-1)
  CUDA_VISIBLE_DEVICES=1 python specpipe.py --stage 3 --nstages 4 --model M --listen-port 29503
  # middle (stage i)
  CUDA_VISIBLE_DEVICES=0 python specpipe.py --stage 2 --nstages 4 --model M --listen-port 29502 --next H:29503
  # head (stage 0): stage block on one GPU, draft on another, drives generation
  CUDA_VISIBLE_DEVICES=0,2 python specpipe.py --stage 0 --nstages 4 --model M \
      --next 127.0.0.1:29501 --draft DRAFT --device cuda:0 --draft-device cuda:1 --adaptive
"""

import argparse, socket, time, threading, queue, os, json, hashlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
if os.environ.get("SHARD_TRANSPORT") == "libp2p":   # libp2p sidecar transport (no PSK; see node_kv)
    import transport as wire
else:
    import wire
from pipeline import load_stage, run_block
from node_kv import send_msg, recv_msg, EDGE_ERRORS, TransportError
from tree import accept_tree, gather_cache
from fastverify import FastVerify
from ngram_draft import NgramDrafter
from specsample import Sampler
import os
try:                                                    # PROVE: opt-in signed per-stage receipts
    from receipt import ReceiptSigner, load_or_make_node_key, verify_receipt, verify_coverage
except Exception:
    ReceiptSigner = None
RECEIPTS = bool(os.environ.get("SHARD_RECEIPTS")) and ReceiptSigner is not None
NODE_KEY_PATH = os.environ.get("SHARD_NODE_KEY", "/root/.shard_node_key")


def _act_digest(t):
    """A deterministic byte digest of an activation tensor for the receipt hash-chain (fp16 bytes)."""
    return t.detach().to(torch.float16).contiguous().cpu().numpy().tobytes()


# ---- fault-tolerance CHECKPOINT envelope (--ft-dump / --resume-file) ----
# The old dump was a bare {"output_ids": [...]}: nothing bound it to the generation it came from,
# so a stale/foreign checkpoint (other prompt, other model, other sampling settings) resumed
# silently and produced a HYBRID generation. The envelope binds every generation input plus a
# digest of the committed tokens; the resume side refuses anything that doesn't match THIS run.
CKPT_SCHEMA = "shard-ckpt-v1"


class CheckpointError(Exception):
    """A resume checkpoint is unversioned, corrupt, or bound to a different generation."""


def _sha256_text(s):
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()


def _ids_digest(ids):
    return hashlib.sha256(json.dumps([int(i) for i in ids]).encode()).hexdigest()


def checkpoint_env(*, prompt, model, tokenizer, settings):
    """The generation-input binding a checkpoint carries and a resume must match exactly.
    `job` is derived from the binding fields, so same-request heal+resume matches for free
    while any cross-job reuse (different prompt/model/tokenizer/settings) is a mismatch."""
    env = {"schema": CKPT_SCHEMA, "prompt_sha256": _sha256_text(prompt), "model": str(model),
           "tokenizer": str(tokenizer), "settings": dict(settings)}
    env["job"] = _sha256_text(json.dumps(env, sort_keys=True))
    return env


def write_checkpoint(path, env, output_ids, **extra):
    """Write the versioned checkpoint ATOMICALLY (tmp + rename): a healer polling the file can
    never read a torn dump, and a crash mid-write can't leave a truncated checkpoint behind."""
    ids = [int(i) for i in output_ids]
    d = {**env, "output_ids": ids, "ids_sha256": _ids_digest(ids), **extra}
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(d, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    return d


def load_checkpoint(path, env):
    """Load the committed tokens for a resume; REJECT (CheckpointError) anything not bound to
    this exact generation -- never splice another job's tokens into a fresh prefill."""
    d = json.load(open(path))
    if d.get("schema") != CKPT_SCHEMA:
        raise CheckpointError(f"checkpoint schema {d.get('schema')!r} != {CKPT_SCHEMA!r} "
                              f"(unversioned/foreign checkpoint refused)")
    for k in ("job", "prompt_sha256", "model", "tokenizer", "settings"):
        if d.get(k) != env[k]:
            raise CheckpointError(f"checkpoint {k} mismatch: resuming it would splice another "
                                  f"generation (checkpoint {str(d.get(k))[:60]!r}, this run {str(env[k])[:60]!r})")
    ids = d.get("output_ids")
    if not isinstance(ids, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        raise CheckpointError("checkpoint output_ids malformed (not a list of token ids)")
    if d.get("ids_sha256") != _ids_digest(ids):
        raise CheckpointError("checkpoint committed-token digest mismatch (corrupt or edited)")
    return ids


SOCK_BUF = 32 << 20   # 32MB SO_SNDBUF/SO_RCVBUF: a ~24MB prefill-chunk activation buffers in-kernel,
                      # so a stage's forward send returns without waiting for the next stage to drain it.
SYNC_SEND = bool(os.environ.get("SHARD_SYNC_SEND"))   # force the OLD synchronous forward send (A/B baseline)


class _AsyncSender:
    """Decouple a stage's compute from the synchronous inter-stage WAN send. At PREFILL the forward
    activation is ~24MB/chunk; with a plain send_msg the stage blocks until the next stage drains it
    (the socket buffer fills), so pipelined prefill collapses at long context -- one stage stalls the
    whole chain (the 1.17Ã—@110k handoff wall). This pushes the send to a background thread draining a
    FIFO queue: the compute thread enqueues and immediately processes the next chunk, so stages truly
    OVERLAP and TTFT approaches the m/(m+p-1) pipeline ceiling. FIFO order is preserved, so the tail's
    results still return to the coordinator in send order. DIRECT-return only (the stage never reads
    back on this socket). A send error is captured and re-raised on the next put(), so the existing
    `except EDGE_ERRORS` edge supervision resets the link exactly as before."""
    def __init__(self, sock):
        self.sock = sock
        self.q = queue.Queue(maxsize=64)
        self.error = None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while True:
            obj = self.q.get()
            if obj is None:
                return
            if self.error is not None:
                continue                       # link already failed; drain+discard while the loop tears down
            try:
                send_msg(self.sock, obj)
            except Exception as e:             # surfaced to the serve loop on the next put() -> edge reset
                self.error = e

    def put(self, obj):
        if self.error is not None:
            raise self.error
        self.q.put(obj)

    def close(self):
        try: self.q.put_nowait(None)
        except Exception: pass


def serve_spec(parts, stage, nstages, listen_port, nxt, timeout, dev, direct=False):
    """a non-head stage under speculative decoding. the op is `verify`: run this
    block on the K+1 proposed tokens, relay forward, and (at the tail) return the
    argmax for every position. `crop` rolls this node's cache back before running.
    direct=True: forward-only -- don't relay the result back up the chain (the tail
    sends it straight to the coordinator). only non-tail stages use this path; the
    direct tail is serve_tail_direct."""
    is_tail = stage == nstages - 1
    nxt_sock = None
    if not is_tail:
        host, port = nxt.split(":")
        nxt_sock = socket.socket(); nxt_sock.settimeout(timeout); nxt_sock.connect((host, int(port)))
        print(f"[s{stage}] connected forward to stage {stage+1} at {nxt}", flush=True)
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", listen_port)); srv.listen(1)
    print(f"[s{stage}] listening on :{listen_port} (edge timeout {timeout:.0f}s)", flush=True)
    while True:
        conn, addr = srv.accept(); conn.settimeout(timeout)
        print(f"[s{stage}] stage {stage-1} connected from {addr}", flush=True)
        cache = DynamicCache(); verifies = 0
        with torch.no_grad():
            while True:
                try:
                    msg = recv_msg(conn)
                    if msg["op"] == "reset":
                        cache = DynamicCache()
                        if nxt_sock:
                            send_msg(nxt_sock, msg)
                            if not direct: recv_msg(nxt_sock)     # wait downstream ack (relay only)
                        if not direct: send_msg(conn, "ok")       # ack predecessor (relay only)
                        continue
                    if msg.get("gather") is not None:      # tree: keep last round's accepted-path KV
                        gather_cache(cache, msg["gather"], dev)
                    elif msg.get("crop") is not None:      # linear: roll back the prior round's rejects
                        cache.crop(msg["crop"])
                    par, dep = msg.get("par"), msg.get("dep")
                    if "token_ids" in msg:                 # served head: coordinator sent token ids, embed here
                        x = parts["embed"](torch.tensor([msg["token_ids"]], device=dev))
                        h = run_block(x, parts, cache, msg["start"], par=par, dep=dep)
                    else:
                        h = run_block(msg["h"].to(dev), parts, cache, msg["start"], par=par, dep=dep)
                    if is_tail:                            # relay tail (non-direct)
                        h = parts["norm"](h)
                        toks = parts["lm_head"](h).argmax(-1)[0].tolist()
                        send_msg(conn, toks)
                    else:
                        send_msg(nxt_sock, {"op": "verify", "h": h.cpu(), "start": msg["start"],
                                            "crop": msg.get("crop"), "gather": msg.get("gather"),
                                            "par": par, "dep": dep})
                        if not direct:                     # relay the result back up the chain
                            send_msg(conn, recv_msg(nxt_sock))
                    verifies += 1
                except EDGE_ERRORS as e:
                    why = "stalled" if isinstance(e, socket.timeout) else "closed"
                    print(f"[s{stage}] edge {why} after {verifies} verifies ({type(e).__name__}); resetting", flush=True)
                    try: conn.close()
                    except OSError: pass
                    break


def serve_tail_direct(parts, listen_port, timeout, dev):
    """tail with DIRECT return: the result goes straight to the coordinator, not
    relayed up the chain. two connections arrive on the listen port -- the
    predecessor (activations) and the coordinator's return channel (which sends a
    {op:hello_return} on connect). select tells them apart (only the return channel
    has a message waiting; the predecessor is idle until driven). each verify's
    result is sent on the return channel."""
    import select
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", listen_port)); srv.listen(2)
    print(f"[tail] listening on :{listen_port} (predecessor + coordinator return, edge timeout {timeout:.0f}s)", flush=True)
    while True:
        c1, _ = srv.accept(); c2, _ = srv.accept()
        ready, _, _ = select.select([c1, c2], [], [], timeout)
        if not ready:
            print("[tail] no return-channel handshake; resetting", flush=True)
            c1.close(); c2.close(); continue
        ret_conn = ready[0]
        try:
            hello = recv_msg(ret_conn)
        except EDGE_ERRORS:
            c1.close(); c2.close(); continue
        if not (isinstance(hello, dict) and hello.get("op") == "hello_return"):
            print("[tail] unexpected handshake; resetting", flush=True)
            c1.close(); c2.close(); continue
        pred_conn = c2 if ret_conn is c1 else c1
        pred_conn.settimeout(timeout)
        print("[tail] predecessor + coordinator-return connected", flush=True)
        cache = DynamicCache(); verifies = 0
        with torch.no_grad():
            while True:
                try:
                    msg = recv_msg(pred_conn)
                    if msg["op"] == "reset":
                        cache = DynamicCache(); send_msg(ret_conn, "ok"); continue
                    if msg.get("gather") is not None:
                        gather_cache(cache, msg["gather"], dev)
                    elif msg.get("crop") is not None:
                        cache.crop(msg["crop"])
                    h = run_block(msg["h"].to(dev), parts, cache, msg["start"],
                                  par=msg.get("par"), dep=msg.get("dep"))
                    h = parts["norm"](h)
                    toks = parts["lm_head"](h).argmax(-1)[0].tolist()
                    send_msg(ret_conn, toks); verifies += 1
                except EDGE_ERRORS as e:
                    print(f"[tail] edge after {verifies} verifies ({type(e).__name__}); resetting", flush=True)
                    try: pred_conn.close(); ret_conn.close()
                    except OSError: pass
                    break


def serve_spec_fast(parts, stage, nstages, listen_port, nxt, timeout, dev, direct=False, max_ctx=2048):
    """serve_spec with the FAST verify: a static-cache CUDA-graph stage forward (~5x
    cheaper than eager). LINEAR spec only (the graph is a fixed K+1 shape; tree is
    variable). first verify after reset = prefill (eager, prompt-length); every later
    verify = a decode round (graphed). rollback is implicit -- a round writes at `start`
    (the committed length), overwriting the prior round's rejects."""
    is_tail = stage == nstages - 1
    nxt_sock = None; sender = None
    host = port = None
    if not is_tail:
        host, port = nxt.split(":")

    def mk_fwd():                                            # (re)build the forward link (+ async sender in direct mode)
        nonlocal nxt_sock, sender, host, port
        # HOT-HEAL: on a relink, repoint the forward link to a pre-warmed SPARE *without reloading weights*.
        # The control-plane healer writes "<host>:<port>" to /root/.shard_next_<stage> on the victim's
        # PREDECESSOR; this is re-read every relink attempt, so the dropped link reconnects to the spare
        # (warm, weights already in VRAM) instead of the dead victim. Absent file => use the launch --next.
        try:
            ov = open(f"/root/.shard_next_{stage}").read().strip()
            if ov:
                host, port = ov.split(":")
        except OSError:
            pass
        s = socket.socket(); s.settimeout(timeout)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SOCK_BUF)   # buffer a full ~24MB prefill chunk in-kernel
        s.connect((host, int(port)))
        nxt_sock = s
        # async send: decoupled forward (no read-back in direct mode). SHARD_SYNC_SEND=1 forces the old
        # synchronous path -> the clean A/B baseline (reproduces last session's handoff-bound 193s@110k).
        sender = _AsyncSender(s) if (direct and not SYNC_SEND) else None

    def fwd_send(o):                                         # async in direct mode, synchronous otherwise
        if direct and sender is not None:
            sender.put(o)
        else:
            send_msg(nxt_sock, o)

    if not is_tail:
        mk_fwd()
        print(f"[s{stage}] connected forward to stage {stage+1} at {nxt}", flush=True)
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCK_BUF)     # accepted conns inherit it: drain big chunks fast
    srv.bind(("0.0.0.0", listen_port)); srv.listen(1)
    fv = FastVerify(parts, maxlen=max_ctx, dev=dev)
    sampler = Sampler(device=dev)                         # greedy until a reset sets temp>0 (tail-relay path)
    node_key = load_or_make_node_key(NODE_KEY_PATH) if RECEIPTS else None
    signer = None
    print(f"[s{stage}] listening on :{listen_port} (FAST verify, max_ctx={max_ctx}, edge timeout {timeout:.0f}s"
          f"{', signed receipts ON' if RECEIPTS else ''})", flush=True)
    while True:
        if not is_tail and nxt_sock is None:                 # forward link dropped (coordinator churn) -> rebuild it;
            for _ in range(60):                              # closing it made the NEXT stage drop its link too, so the
                try:                                         # whole ring re-handshakes fresh and a new coordinator can drive it
                    mk_fwd(); break
                except OSError: time.sleep(0.5)
            print(f"[s{stage}] forward link rebuilt -> {nxt}" if nxt_sock else f"[s{stage}] relink FAILED", flush=True)
        conn, addr = srv.accept(); conn.settimeout(timeout)
        print(f"[s{stage}] stage {stage-1} connected from {addr}", flush=True)
        fv.reset(); first = True; verifies = 0
        with torch.no_grad():
            while True:
                msg = None                                    # bind before recv: a partial/corrupt frame (e.g. a
                try:                                          # peer resetting mid-message during a heal) must hit the
                    msg = recv_msg(conn)                      # 'bad msg -> reset' path below, not crash on unbound msg

                    if msg["op"] == "reset":
                        fv.reset(); first = True
                        sampler = Sampler(temp=msg.get("temp", 0.0), top_p=msg.get("top_p", 1.0),
                                          top_k=msg.get("top_k", 0), seed=msg.get("seed", 0), device=dev)
                        if RECEIPTS:                     # start this job's per-stage activation hash-chain
                            signer = ReceiptSigner(node_key, msg.get("swarm_id", "swarm"),
                                                   msg.get("job_id", "job"), parts["lo"], parts["hi"])
                        if nxt_sock:
                            fwd_send(msg)
                            if not direct: recv_msg(nxt_sock)
                        if not direct: send_msg(conn, "ok")
                        continue
                    if msg["op"] == "receipt":           # job done: sign + accumulate down the ring
                        if RECEIPTS and signer is not None:
                            msg.setdefault("receipts", []).append({"stage": stage, **signer.finalize()})
                        if nxt_sock:
                            fwd_send(msg)                 # tail returns the full list to the coordinator (direct)
                            if not direct: send_msg(conn, recv_msg(nxt_sock))
                        else:
                            send_msg(conn, msg.get("receipts", []))
                        continue
                    g = msg.get("gather")
                    if g:                                  # lazy: compact prev tree's accepted path KV
                        fv.tree_gather(g[0], g[1])
                    if "token_ids" in msg:                 # served head: embed ids here
                        x = parts["embed"](torch.tensor([msg["token_ids"]], device=dev))
                    else:
                        x = msg["h"].to(dev)
                    is_pf = ("par" not in msg) and (first or msg.get("prefill"))
                    draft = msg.get("draft")               # the K proposed tokens, for the tail's sampler
                    if draft is None and "token_ids" in msg and not is_pf and "par" not in msg:
                        draft = msg["token_ids"][1:]       # head derives them: chunk = [carry] + K drafts
                    if "par" in msg:                       # TREE verify (fixed-topology graph)
                        h = fv.tree_decode(x, msg["start"], msg["par"], msg["dep"])
                    else:                                  # LINEAR verify (prefill, then graphed decode)
                        h = fv.prefill(x, msg["start"]) if is_pf else fv.decode(x, msg["start"]); first = False
                        if RECEIPTS and signer is not None:   # attest this block's input->output transform
                            signer.observe(_act_digest(x), _act_digest(h))
                    if is_tail:
                        # prefill: only the last token's logit is consumed -> avoid a [chunk x vocab] OOM
                        h = parts["norm"](h[:, -1:] if is_pf else h)
                        logits = parts["lm_head"](h)
                        if is_pf:                          # prefill: the single first new token (greedy=argmax)
                            send_msg(conn, [sampler.sample_logits(logits[0, -1])])
                        elif draft is None:                # tree / no-draft decode: per-position (greedy=argmax)
                            send_msg(conn, logits.argmax(-1)[0].tolist() if sampler.greedy
                                     else [sampler.sample_logits(logits[0, i]) for i in range(logits.shape[1])])
                        else:                              # lossless speculative sampling over the K+1 logits
                            send_msg(conn, sampler.accept(logits[0], draft))
                    else:
                        fwd = {"op": "verify", "h": h.cpu(), "start": msg["start"], "prefill": msg.get("prefill")}
                        if "par" in msg: fwd["par"] = msg["par"]; fwd["dep"] = msg["dep"]
                        if draft is not None: fwd["draft"] = draft
                        if g: fwd["gather"] = g
                        fwd_send(fwd)
                        if not direct:
                            send_msg(conn, recv_msg(nxt_sock))
                    verifies += 1
                except EDGE_ERRORS as e:
                    why = "stalled" if isinstance(e, socket.timeout) else "closed"
                    print(f"[s{stage}] edge {why} after {verifies} verifies ({type(e).__name__}); resetting", flush=True)
                    try: conn.close()
                    except OSError: pass
                    if sender is not None:                    # stop the async sender thread before dropping the link
                        sender.close(); sender = None
                    if nxt_sock is not None:                  # drop forward link -> ring re-handshakes fresh on re-accept
                        try: nxt_sock.close()
                        except OSError: pass
                        nxt_sock = None
                    break
                except Exception as e:                       # survive a bad message instead of dying
                    k = list(msg.keys()) if isinstance(msg, dict) else "?"
                    print(f"[s{stage}] bad msg after {verifies} verifies ({type(e).__name__}: {str(e)[:80]} keys={k}); resetting", flush=True)
                    try: conn.close()
                    except OSError: pass
                    if nxt_sock is not None:
                        try: nxt_sock.close()
                        except OSError: pass
                        nxt_sock = None
                    break


def serve_tail_fast(parts, listen_port, timeout, dev, max_ctx=2048):
    """direct-return tail, RESILIENT to coordinator churn. the predecessor (the ring's
    forward chain) and the coordinator-return channel have INDEPENDENT lifecycles:
      - return channel drops (coordinator restart / gateway reconnect / retry) -> keep
        the predecessor + KV, re-accept a new return channel. the ring is NOT poisoned.
      - predecessor drops (an upstream stage reconnected) -> re-accept it, keep the return.
    a connection is identified by CONTENT: the coordinator sends {"op":"hello_return"} on
    its return channel; anything else is the predecessor (and its first message is queued).
    this is what lets the gateway connect, fail, retry, and restart without relaunching the
    swarm -- the c0mpute come-and-go property at the tail."""
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", listen_port)); srv.listen(8)
    fv = FastVerify(parts, maxlen=max_ctx, dev=dev)
    sampler = Sampler(device=dev)                         # greedy until a reset sets temp>0
    node_key = load_or_make_node_key(NODE_KEY_PATH) if RECEIPTS else None
    signer = None
    print(f"[tail] listening on :{listen_port} (FAST verify, max_ctx={max_ctx}, direct return, edge timeout {timeout:.0f}s"
          f"{', signed receipts ON' if RECEIPTS else ''})", flush=True)
    pred = ret = None; pending = None; first = True

    def fill():                                            # block until BOTH channels are present
        nonlocal pred, ret, pending
        while pred is None or ret is None:
            c, _ = srv.accept()
            try:
                m = recv_msg(c)
            except EDGE_ERRORS:
                try: c.close()
                except OSError: pass
                continue
            if isinstance(m, dict) and m.get("op") == "hello_return":
                if ret is not None:
                    try: ret.close()
                    except OSError: pass
                ret = c; print("[tail] coordinator-return (re)connected", flush=True)
            else:                                          # predecessor: its first msg is real, queue it
                if pred is not None:
                    try: pred.close()
                    except OSError: pass
                pred = c; pred.settimeout(timeout); pending = m
                print("[tail] predecessor (re)connected", flush=True)

    with torch.no_grad():
        while True:
            fill()
            try:
                msg = pending if pending is not None else recv_msg(pred)
                pending = None
            except EDGE_ERRORS as e:                       # predecessor gone -> usually a coordinator churn, which
                print(f"[tail] predecessor edge ({type(e).__name__}); re-accepting predecessor + return", flush=True)
                try: pred.close()
                except OSError: pass
                pred = None
                if ret is not None:                        # ALSO drop the return channel: the new coordinator brings a
                    try: ret.close()                       # fresh return channel, so the reset's 'ok' must not be sent to
                    except OSError: pass                   # the dead old one (the race that made churn recovery flaky)
                    ret = None
                continue
            try:
                if msg["op"] == "reset":
                    fv.reset(); first = True
                    sampler = Sampler(temp=msg.get("temp", 0.0), top_p=msg.get("top_p", 1.0),
                                      top_k=msg.get("top_k", 0), seed=msg.get("seed", 0), device=dev)
                    if RECEIPTS:
                        signer = ReceiptSigner(node_key, msg.get("swarm_id", "swarm"),
                                               msg.get("job_id", "job"), parts["lo"], parts["hi"])
                    send_msg(ret, "ok"); continue
                if msg["op"] == "receipt":                  # job done: sign + return the full ring's receipts
                    if RECEIPTS and signer is not None:
                        msg.setdefault("receipts", []).append({"stage": "tail", **signer.finalize()})
                    send_msg(ret, msg.get("receipts", []))
                    continue
                g = msg.get("gather")
                if g:
                    fv.tree_gather(g[0], g[1])
                x = msg["h"].to(dev)
                is_pf = ("par" not in msg) and (first or msg.get("prefill"))
                draft = msg.get("draft")                    # the K proposed tokens (for the sampler's accept)
                if "par" in msg:                           # TREE verify
                    h = fv.tree_decode(x, msg["start"], msg["par"], msg["dep"])
                else:
                    h = fv.prefill(x, msg["start"]) if is_pf else fv.decode(x, msg["start"]); first = False
                    if RECEIPTS and signer is not None:    # attest this block's input->output transform
                        signer.observe(_act_digest(x), _act_digest(h))
                    # prefill: the coordinator only consumes the LAST token (next-token after the chunk),
                    # so run lm_head on just that position -- a full [chunk x vocab] logit tensor is ~1.5GB
                    # at chunk 4096 and OOMs a 24GB tail at long context. decode needs all K+1 logits.
                    if is_pf:
                        h = h[:, -1:]
                logits = parts["lm_head"](parts["norm"](h))
                if is_pf:                                  # prefill: the single first new token (greedy=argmax)
                    send_msg(ret, [sampler.sample_logits(logits[0, -1])])
                elif draft is None:                        # tree / no-draft decode: per-position (greedy=argmax)
                    send_msg(ret, logits.argmax(-1)[0].tolist() if sampler.greedy
                             else [sampler.sample_logits(logits[0, i]) for i in range(logits.shape[1])])
                else:                                      # lossless speculative SAMPLING over the K+1 logits
                    send_msg(ret, sampler.accept(logits[0], draft))
            except EDGE_ERRORS as e:                        # return channel gone -> re-accept it, keep pred + KV
                print(f"[tail] return edge ({type(e).__name__}); dropping return channel, keeping predecessor+KV", flush=True)
                try: ret.close()
                except OSError: pass
                ret = None; continue
            except Exception as e:                          # bad message -> drop the return channel, survive
                k = list(msg.keys()) if isinstance(msg, dict) else "?"
                print(f"[tail] bad msg ({type(e).__name__}: {str(e)[:80]} keys={k}); dropping return channel", flush=True)
                try: ret.close()
                except OSError: pass
                ret = None; continue


def generate_spec(draft, parts, tok, sock, prompt, K, max_new, dev, draft_dev, timeout,
                  adaptive=False, k_min=1, k_max=12, draft_sock=None):
    """stage 0: draft proposes K tokens on its own GPU; the distributed target
    verifies [cur, d_1..d_K] in one chain traversal; greedy-accept the longest
    matching prefix. caches (draft + this node's block) crop locally; downstream
    nodes crop lazily on the next verify."""
    sock.settimeout(timeout)
    eos = tok.eos_token_id
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to(dev)
    prompt_ids = enc["input_ids"][0].tolist()   # for the in-house draft service (full-prefix queries)
    head_cache, draft_cache = DynamicCache(), DynamicCache()
    pos = 0
    out = []                                    # defined before any edge can fail (prefill incl.)

    def embed(tokens):                          # python list -> this node's block output
        x = torch.tensor([tokens], device=dev)
        return run_block(parts["embed"](x), parts, head_cache, pos)

    try:
        send_msg(sock, {"op": "reset"}); recv_msg(sock)
        # ---- prefill: target processes the whole prompt; draft fills its cache ----
        h = run_block(parts["embed"](ids), parts, head_cache, 0)
        preds = (send_msg(sock, {"op": "verify", "h": h.cpu(), "start": 0}), recv_msg(sock))[1]
        cur = preds[-1]                                     # target's token for position L
        pos = ids.shape[1]
        if draft_sock is None:                              # local transformers draft fills its cache
            with torch.no_grad():
                draft(input_ids=ids.to(draft_dev), past_key_values=draft_cache, use_cache=True)

        out = [cur]
        rounds, accepted_total = 0, 0
        kc, ema_n, k_hist = K, float(K), []
        tail_crop = None                                   # lazy downstream rollback, piggybacked
        t_draft = t_verify = 0.0                            # round-budget instrumentation
        t0 = time.time()
        with torch.no_grad():
            while len(out) < max_new and cur != eos:
                # 1. draft proposes kc tokens
                td = time.time()
                if draft_sock is not None:                  # in-house vLLM draft service (full prefix; prefix-cached)
                    send_msg(draft_sock, {"ids": prompt_ids + out, "k": kc})
                    drafts = recv_msg(draft_sock)
                else:                                       # local transformers draft (incremental cache)
                    drafts, dtok = [], cur
                    for i in range(kc + 1):
                        dl = draft(input_ids=torch.tensor([[dtok]], device=draft_dev),
                                   past_key_values=draft_cache, use_cache=True).logits
                        dtok = int(dl[0, -1].argmax())
                        if i < kc:
                            drafts.append(dtok)
                t_draft += time.time() - td
                # 2. verify [cur, d_1..d_kc] in one traversal (carry the prior round's rollback)
                tv = time.time()
                h = embed([cur] + drafts)
                send_msg(sock, {"op": "verify", "h": h.cpu(), "start": pos, "crop": tail_crop})
                r = recv_msg(sock)
                t_verify += time.time() - tv
                # 3. greedy acceptance: longest prefix with d_j == r_j
                n = 0
                for j in range(kc):
                    if drafts[j] == r[j]:
                        n += 1
                    else:
                        break
                committed = drafts[:n] + [r[n]]            # n accepted + 1 correction
                out.extend(committed)
                cur = r[n]
                pos += n + 1
                rounds += 1; accepted_total += n; k_hist.append(kc)
                # 4. roll caches back to the accepted length: this node's + draft's now,
                #    downstream nodes lazily on the next verify (no extra round-trip)
                head_cache.crop(pos)
                if draft_sock is None:
                    draft_cache.crop(pos)                   # vLLM service manages its own cache
                tail_crop = pos
                # 5. adaptive K: aim a couple beyond the running acceptance (EMA of n)
                ema_n = 0.7 * ema_n + 0.3 * n
                if adaptive:
                    kc = max(k_min, min(k_max, round(ema_n) + 2))
                if eos in committed:
                    break
    except EDGE_ERRORS as e:
        raise TransportError(f"pipeline edge failed at token {len(out)} ({type(e).__name__}: {e})") from e

    dt = time.time() - t0
    if eos in out:
        out = out[:out.index(eos)]
    return {
        "text": tok.decode(out, skip_special_tokens=True),
        "n_tokens": len(out), "rounds": rounds,
        "mean_accept": accepted_total / max(rounds, 1),
        "toks_per_traversal": (accepted_total + rounds) / max(rounds, 1),
        "tok_s": len(out) / max(dt, 1e-9),
        "mean_K": (sum(k_hist) / len(k_hist)) if k_hist else K,
        "k_lo": min(k_hist) if k_hist else K, "k_hi": max(k_hist) if k_hist else K,
        "draft_ms": t_draft / max(rounds, 1) * 1000, "verify_ms": t_verify / max(rounds, 1) * 1000,
        "output_ids": out,
    }


def coordinate(draft_sock, pipe_sock, tok, prompt, K, max_new, timeout,
               adaptive=False, k_min=1, k_max=12, ret_sock=None, prefill_chunk=0):
    """the in-house coordinator (c0mpute entry node): holds NO 120B layers. it
    tokenizes, queries the in-house draft for K tokens, sends token ids into the
    swarm's stage 0 (which embeds + runs), reads back the verify, greedy-accepts.
    the whole 120B lives on the scattered swarm nodes; this node is just the entry
    point plus the managed draft. lazy crop propagates to every swarm node.
    ret_sock set => DIRECT return: send forward to stage 0, receive the verify
    result straight from the tail (1 hop) instead of relayed back up the chain."""
    pipe_sock.settimeout(timeout)
    rx = ret_sock if ret_sock is not None else pipe_sock     # where results come back (direct => tail)
    eos = tok.eos_token_id
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    prompt_ids = enc["input_ids"][0].tolist()
    out = []
    try:
        send_msg(pipe_sock, {"op": "reset"}); recv_msg(rx)
        # prefill: one shot, or CHUNKED (long prompts) so each stage's per-chunk activation
        # stays bounded (the KV cache accumulates). flex_attention keeps each chunk O(n) on Ada.
        if prefill_chunk and len(prompt_ids) > prefill_chunk:
            r = None
            for i in range(0, len(prompt_ids), prefill_chunk):
                send_msg(pipe_sock, {"op": "verify", "token_ids": prompt_ids[i:i + prefill_chunk], "start": i})
                r = recv_msg(rx)
            cur = r[-1]
        else:
            send_msg(pipe_sock, {"op": "verify", "token_ids": prompt_ids, "start": 0})   # prefill
            cur = recv_msg(rx)[-1]
        pos = len(prompt_ids)
        out = [cur]
        rounds, accepted_total = 0, 0
        kc, ema_n, k_hist = K, float(K), []
        tail_crop = None
        t_draft = t_verify = 0.0
        t0 = time.time()
        while len(out) < max_new and cur != eos:
            td = time.time()
            send_msg(draft_sock, {"ids": prompt_ids + out, "k": kc}); drafts = recv_msg(draft_sock)
            t_draft += time.time() - td
            tv = time.time()
            send_msg(pipe_sock, {"op": "verify", "token_ids": [cur] + drafts, "start": pos, "crop": tail_crop})
            r = recv_msg(rx)
            t_verify += time.time() - tv
            n = 0
            for j in range(kc):
                if drafts[j] == r[j]: n += 1
                else: break
            committed = drafts[:n] + [r[n]]
            out.extend(committed); cur = r[n]; pos += n + 1
            rounds += 1; accepted_total += n; k_hist.append(kc); tail_crop = pos
            ema_n = 0.7 * ema_n + 0.3 * n
            if adaptive:
                kc = max(k_min, min(k_max, round(ema_n) + 2))
            if eos in committed:
                break
    except EDGE_ERRORS as e:
        raise TransportError(f"pipeline edge failed at token {len(out)} ({type(e).__name__}: {e})") from e
    dt = time.time() - t0
    if eos in out:
        out = out[:out.index(eos)]
    return {
        "text": tok.decode(out, skip_special_tokens=True), "n_tokens": len(out), "rounds": rounds,
        "mean_accept": accepted_total / max(rounds, 1),
        "toks_per_traversal": (accepted_total + rounds) / max(rounds, 1),
        "tok_s": len(out) / max(dt, 1e-9),
        "mean_K": (sum(k_hist) / len(k_hist)) if k_hist else K,
        "k_lo": min(k_hist) if k_hist else K, "k_hi": max(k_hist) if k_hist else K,
        "draft_ms": t_draft / max(rounds, 1) * 1000, "verify_ms": t_verify / max(rounds, 1) * 1000,
        "output_ids": out,
    }


def coordinate_pipe(draft_sock, pipe_sock, tok, prompt, K, max_new, timeout, depth, ret_sock=None,
                    ignore_eos=False, prefill_chunk=0, draft_ctx=0, on_commit=None, reasoning=None, system=None,
                    max_ctx=0, local_draft=None, temp=0.0, top_p=1.0, top_k=0, seed=0, prefill_depth=8,
                    resume_ids=None, resumable=False):
    """PIPELINED coordinator: keep `depth` verify chunks in flight over the ring, so
    throughput approaches the ring's per-chunk THROUGHPUT, not its full latency (the
    GLM-pipe lever, on the gpt-oss fast-verify path). Same K+1-token chunk as the
    synchronous coordinate() ([tail_tok]+K drafts) so the fixed-shape CUDA graph holds;
    the StaticKV writes at each chunk's `start`, so after a divergence the fresh chunk
    (sent next) overwrites the stale chunks' KV and the coordinator just discards the
    stale RESULTS. Greedy => output identical to the synchronous path. Needs the swarm
    in --direct-return mode (fire-forward stages, tail returns straight here).

    The draft source is pluggable: a vLLM draft over `draft_sock` (the request/fetch
    socket protocol), or an in-process `local_draft` (model-free n-gram drafter) that
    speaks the same request()/fetch() async shim. n-gram is the long-context path — it
    raises g with no model and no KV, so spec-decode survives past 100k where the 20b
    draft OOMs. Either way the swarm verifies every token, so output is identical."""
    pipe_sock.settimeout(timeout)
    rx = ret_sock if ret_sock is not None else pipe_sock
    use_local = local_draft is not None
    def d_request(ids, k):                       # issue a draft request (one outstanding)
        if use_local: local_draft.request(ids, k)
        else: send_msg(draft_sock, {"ids": ids, "k": k})
    def d_fetch():                               # collect the proposed k tokens
        return local_draft.fetch() if use_local else recv_msg(draft_sock)
    eos = tok.eos_token_id
    ct_kw = {"reasoning_effort": reasoning} if reasoning else {}   # gpt-oss: 'low' -> terse analysis, more budget for the answer
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True, **ct_kw)
    prompt_ids = enc["input_ids"][0].tolist()
    # FAULT-TOLERANCE RESUME: re-prefill prompt + the already-committed tokens onto a freshly-healed
    # ring, then continue decoding -- so a mid-request node death costs ONE re-prefill, not a restart,
    # and the user's committed output is preserved (continuation, not regeneration). gen_ids is the
    # full context to rebuild every stage's KV over; out is seeded with the recovered tokens.
    resume_ids = list(resume_ids or [])
    gen_ids = prompt_ids + resume_ids
    if max_ctx:                                                   # never let prompt+generation exceed the window
        max_new = max(len(resume_ids) + 16, min(max_new, max_ctx - len(gen_ids) - 16))
    out = []
    t_draft = t_recv = 0.0; prefill_s = 0.0
    try:
        send_msg(pipe_sock, {"op": "reset", "temp": temp, "top_p": top_p, "top_k": top_k, "seed": seed})
        recv_msg(rx)
        # prefill: one shot, or chunked (long prompts) with the prefill flag so stages run flex
        t_pf = time.time()
        if prefill_chunk and len(gen_ids) > prefill_chunk:
            # PIPELINED prefill: keep prefill_depth chunks in flight so the stages OVERLAP -- stage 0
            # prefills chunk c+1 while stage 1 prefills chunk c, etc. Each stage's chunk c+1 only needs
            # that stage's OWN chunk-c KV (already local, causal), so prefill is embarrassingly
            # pipelineable; the win is ~Nstage-fold over the old one-chunk-at-a-time path, which waited a
            # full ring traversal per chunk (zero overlap = the ~10-min/100k TTFT). Results come back in
            # FIFO order on rx; only the final chunk's last token (cur) is consumed. depth bounds how many
            # big inter-stage activation messages are in flight (backpressure self-regulates the ring).
            starts = list(range(0, len(gen_ids), prefill_chunk))
            def _send_pf(i):
                send_msg(pipe_sock, {"op": "verify", "token_ids": gen_ids[i:i + prefill_chunk],
                                     "start": i, "prefill": True})
            d = min(max(prefill_depth, 1), len(starts)); sent = 0; rr = None
            while sent < d:
                _send_pf(starts[sent]); sent += 1
            for _ in range(len(starts)):
                rr = recv_msg(rx)
                if sent < len(starts):
                    _send_pf(starts[sent]); sent += 1
            cur = rr[-1]
        else:
            send_msg(pipe_sock, {"op": "verify", "token_ids": gen_ids, "start": 0})   # prefill
            cur = recv_msg(rx)[-1]
        prefill_s = time.time() - t_pf
        pos = len(gen_ids)
        out = resume_ids + [cur]                    # preserve recovered tokens; cur = next after them
        if on_commit: on_commit({"phase": "prefilled", "prompt_tokens": len(prompt_ids),
                                 "resume_tokens": len(resume_ids), "prefill_s": prefill_s})
        inflight = []                              # FIFO of (start_pos, drafts) sent but not yet read
        discard = 0                                # stale post-divergence results still to drain
        send_pos = pos                             # absolute pos where the next chunk writes
        dprefix = gen_ids + [cur]                  # draft-server query prefix; dprefix[-1] == next tail_tok
        valid = accepted = wasted = 0
        t0 = time.time()
        done = False
        # ASYNC DRAFT: keep exactly one draft request outstanding so the draft server
        # computes the next chunk WHILE the current verify chunks cross the WAN. Each fill
        # collects the ready draft, sends the verify chunk, then issues the next draft request
        # (which then runs concurrently with the verify read below). draft latency is hidden.
        # the draft only needs RECENT context to predict the next tokens (next-token is dominated by
        # recent tokens); the full swarm still verifies with the complete context. Windowing the draft
        # query keeps the draft fast at long context (a full-95k draft is ~800ms/round and kills spec).
        dq = lambda: (dprefix[-draft_ctx:] if draft_ctx else dprefix)
        d_request(dq(), K)                                        # prime: one outstanding request
        while not done:
            while len(inflight) < depth and not done:                  # FILL the pipeline
                td = time.time(); ds = d_fetch(); t_draft += time.time() - td  # ready (overlapped)
                send_msg(pipe_sock, {"op": "verify", "token_ids": [dprefix[-1]] + ds, "start": send_pos})
                inflight.append((send_pos, ds)); dprefix = dprefix + ds; send_pos += K
                d_request(dq(), K)                                    # issue next -> runs during the read below
            tr = time.time(); r = recv_msg(rx); t_recv += time.time() - tr   # READ one result
            sp, ds = inflight.pop(0)
            if discard > 0:                                            # stale (post-divergence) -> skip
                discard -= 1; wasted += 1; continue
            n = 0
            for j in range(K):
                if ds[j] == r[j]: n += 1
                else: break
            valid += 1; accepted += n
            if n == K:
                out.extend(ds); pos += K; cur = ds[-1]
                committed = ds
            else:                                                      # divergence -> correct + flush in-flight
                committed = ds[:n] + [r[n]]
                out.extend(committed); cur = r[n]; pos += n + 1
                discard = len(inflight)                                # every chunk still in flight is stale
                d_fetch()                                             # outstanding draft is stale -> drop it
                dprefix = prompt_ids + out; send_pos = pos             # re-draft from the corrected prefix
                d_request(dq(), K)                                    # re-prime from the corrected prefix
            if on_commit: on_commit({"phase": "decode", "out": out, "dt": time.time() - t0})
            if len(out) >= max_new or (not ignore_eos and (cur == eos or eos in committed)):
                done = True
        d_fetch()                                                     # drain the outstanding draft request
        while inflight:                                               # drain unread results -> sockets clean for next gen
            recv_msg(rx); inflight.pop(0)
    except EDGE_ERRORS as e:
        if resumable:                          # a node died: hand the committed tokens back so the control
            committed = out if out else list(resume_ids)   # plane can heal the ring + resume (not restart)
            return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}",
                    "output_ids": committed, "n_tokens": len(committed),
                    "text": tok.decode(committed, skip_special_tokens=True)}
        raise TransportError(f"pipeline edge failed at token {len(out)} ({type(e).__name__}: {e})") from e
    dt = time.time() - t0
    if eos in out:
        out = out[:out.index(eos)]
    return {
        "ok": True,
        "text": tok.decode(out, skip_special_tokens=True), "n_tokens": len(out), "rounds": valid,
        "mean_accept": accepted / max(valid, 1),
        "toks_per_traversal": (accepted + valid) / max(valid, 1),
        "tok_s": len(out) / max(dt, 1e-9), "wasted": wasted, "depth": depth, "K": K,
        "draft_ms": t_draft / max(valid, 1) * 1000, "recv_ms": t_recv / max(valid, 1) * 1000,
        "prefill_s": prefill_s, "prompt_tokens": len(prompt_ids), "resume_tokens": len(resume_ids),
        "output_ids": out,
    }


def sample_distribution_test(pipe_sock, tok, prompt, timeout, ret_sock, local_draft, K, n_draws=2000,
                             temp=1.0, top_p=1.0, top_k=0, seed=0, prefill_chunk=0, reasoning=None,
                             stops=12, step=8, n_per=140):
    """On-swarm LOSSLESSNESS proof for speculative SAMPLING. At each of `stops` content positions, draw
    n_per iid samples of the next token THREE ways and compare the empirical distributions:
      PLAIN  — send a 1-token chunk (K=0 accept) -> a pure target sample from the temp/top-p dist.
      SPEC   — send [carry]+K n-gram drafts -> the speculative-sampling accept's committed first token.
      PLAIN2 — a second plain block -> calibrates the Monte-Carlo noise floor (two finite samples of the
               SAME distribution still differ by ~this much), so TV(spec,plain) is judged against it.
    Both PLAIN and SPEC must be distributed as p(next | context). Draws don't advance the KV (they
    overwrite scratch slots), so they're pipelined (depth chunks in flight) and fast; between stops we
    advance `step` tokens. We sweep positions so some land HIGH-ENTROPY (where the rejection/residual
    sampling actually does work and a distribution test is discriminating). Verdict aggregates over the
    high-entropy stops: if mean TV(spec,plain) ~ mean TV(plain,plain) noise floor, sampling is lossless
    on the real gpt-oss-120B distribution end-to-end through the WAN ring."""
    import collections, math
    pipe_sock.settimeout(timeout)
    rx = ret_sock if ret_sock is not None else pipe_sock
    ct_kw = {"reasoning_effort": reasoning} if reasoning else {}
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True, **ct_kw)
    prompt_ids = enc["input_ids"][0].tolist()
    send_msg(pipe_sock, {"op": "reset", "temp": temp, "top_p": top_p, "top_k": top_k, "seed": seed}); recv_msg(rx)
    if prefill_chunk and len(prompt_ids) > prefill_chunk:
        for i in range(0, len(prompt_ids), prefill_chunk):
            send_msg(pipe_sock, {"op": "verify", "token_ids": prompt_ids[i:i + prefill_chunk], "start": i, "prefill": True})
            cur = recv_msg(rx)[-1]
    else:
        send_msg(pipe_sock, {"op": "verify", "token_ids": prompt_ids, "start": 0, "prefill": True}); cur = recv_msg(rx)[-1]
    ctx = prompt_ids + [cur]; pos = len(prompt_ids) + 1

    def block(spec, ctx, start, depth=8):                       # n_per pipelined iid draws at a fixed position
        ds = local_draft.propose(ctx, K) if spec else []
        msg = {"op": "verify", "token_ids": ([ctx[-1]] + ds) if spec else [ctx[-1]], "start": start, "draft": ds}
        h = collections.Counter(); sent = got = 0
        while got < n_per:
            while sent < n_per and sent - got < depth:
                send_msg(pipe_sock, msg); sent += 1
            h[recv_msg(rx)[0]] += 1; got += 1
        return h

    def entropy(h):
        n = sum(h.values()); return -sum((c / n) * math.log2(c / n) for c in h.values() if c)

    def tvd(a, b):
        ks = set(a) | set(b); na, nb = sum(a.values()), sum(b.values())
        return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in ks)

    rows = []
    for _ in range(stops):
        start = pos - 1
        hp = block(False, ctx, start); hs = block(True, ctx, start); hp2 = block(False, ctx, start)
        rows.append({"entropy": round(entropy(hp), 3), "tv_spec_plain": tvd(hp, hs),
                     "tv_plain_plain": tvd(hp, hp2), "distinct": len(set(hp) | set(hs)),
                     "top": [{"id": int(k), "plain": round(hp[k] / n_per, 3), "spec": round(hs.get(k, 0) / n_per, 3),
                              "text": tok.decode([k])} for k in sorted(hp, key=lambda k: -hp[k])[:6]]})
        # advance `step` tokens (greedy-commit the plain mode's mode) to reach the next content position
        for _ in range(step):
            send_msg(pipe_sock, {"op": "verify", "token_ids": [ctx[-1]], "start": pos - 1, "draft": []})
            ctx.append(recv_msg(rx)[0]); pos += 1
    hi = [r for r in rows if r["entropy"] >= 1.0]               # discriminating (>=1 bit) positions
    agg = hi if hi else rows
    mean_spec = sum(r["tv_spec_plain"] for r in agg) / len(agg)
    mean_noise = sum(r["tv_plain_plain"] for r in agg) / len(agg)
    return {"temp": temp, "top_p": top_p, "n_per": n_per, "stops": stops, "high_entropy_stops": len(hi),
            "max_entropy": max(r["entropy"] for r in rows),
            "mean_tv_spec_vs_plain": mean_spec, "mean_tv_noise_floor": mean_noise,
            "per_stop": rows}


def coordinate_tree(draft_sock, pipe_sock, tok, prompt, tree_cfg, max_new, timeout, ret_sock=None):
    """TREE spec-decode coordinator. each round the draft returns a *tree* of
    candidate continuations rooted at cur; the swarm verifies the whole tree in one
    traversal (tree mask); accept_tree walks the target's argmaxes for the longest
    matching path. the accepted path's KV is kept on each node via a lazy gather
    (piggybacked on the next verify). exact greedy => identical to plain decode."""
    pipe_sock.settimeout(timeout)
    rx = ret_sock if ret_sock is not None else pipe_sock
    eos = tok.eos_token_id
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    prompt_ids = enc["input_ids"][0].tolist()
    out = []
    try:
        send_msg(pipe_sock, {"op": "reset"}); recv_msg(rx)
        send_msg(pipe_sock, {"op": "verify", "token_ids": prompt_ids, "start": 0})   # linear prefill
        cur = recv_msg(rx)[-1]
        pos = len(prompt_ids)
        out = [cur]
        rounds, accepted_total, m_total, gather_prev = 0, 0, 0, None
        t_draft = t_verify = 0.0
        t0 = time.time()
        while len(out) < max_new and cur != eos:
            td = time.time()
            send_msg(draft_sock, {"ids": prompt_ids + out, "tree": tree_cfg})   # ask for a tree
            tr = recv_msg(draft_sock)
            t_draft += time.time() - td
            tk, par, dep = tr["tok"], tr["par"], tr["dep"]
            children = [[] for _ in tk]
            for i, p in enumerate(par):
                if p != -1: children[p].append(i)
            tv = time.time()
            send_msg(pipe_sock, {"op": "verify", "token_ids": tk, "par": par, "dep": dep,
                                 "start": pos, "gather": gather_prev})
            targ = recv_msg(rx)                                # one argmax per tree node
            t_verify += time.time() - tv
            committed, kept = accept_tree(tk, par, {i: c for i, c in enumerate(children)}, targ)
            out.extend(committed); cur = committed[-1]
            gather_prev = list(range(pos)) + [pos + ki for ki in kept]   # keep prefix + accepted path
            pos += len(kept)
            rounds += 1; accepted_total += len(kept) - 1; m_total += len(tk)
            if eos in committed:
                break
    except EDGE_ERRORS as e:
        raise TransportError(f"pipeline edge failed at token {len(out)} ({type(e).__name__}: {e})") from e
    dt = time.time() - t0
    if eos in out:
        out = out[:out.index(eos)]
    return {
        "text": tok.decode(out, skip_special_tokens=True), "n_tokens": len(out), "rounds": rounds,
        "mean_accept": accepted_total / max(rounds, 1),
        "toks_per_traversal": len(out) / max(rounds, 1),
        "tree_nodes": m_total / max(rounds, 1),
        "tok_s": len(out) / max(dt, 1e-9),
        "draft_ms": t_draft / max(rounds, 1) * 1000, "verify_ms": t_verify / max(rounds, 1) * 1000,
    }


def coordinate_tree_fast(draft_sock, pipe_sock, tok, prompt, tree_cfg, max_new, timeout, ret_sock=None):
    """SYNC tree spec on the FAST (graphed) verify. Draft a fixed-topology tree, verify all
    its nodes in ONE traversal (FastVerify.tree_decode), accept the best root-to-leaf path,
    and compact the accepted KV (static tree_gather). gather payload is (start_prev, kept)
    for the static cache (vs the eager gather_cache's absolute-index list). exact greedy."""
    pipe_sock.settimeout(timeout)
    rx = ret_sock if ret_sock is not None else pipe_sock
    eos = tok.eos_token_id
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    prompt_ids = enc["input_ids"][0].tolist()
    out = []
    t_draft = t_verify = 0.0
    try:
        send_msg(pipe_sock, {"op": "reset"}); recv_msg(rx)
        send_msg(pipe_sock, {"op": "verify", "token_ids": prompt_ids, "start": 0})   # linear prefill
        cur = recv_msg(rx)[-1]
        pos = len(prompt_ids); out = [cur]
        rounds, accepted_total, tnodes = 0, 0, 0
        gather_prev = None
        t0 = time.time()
        while len(out) < max_new and cur != eos:
            td = time.time()
            send_msg(draft_sock, {"ids": prompt_ids + out, "tree": tree_cfg}); tr = recv_msg(draft_sock)
            t_draft += time.time() - td
            tk, par, dep = tr["tok"], tr["par"], tr["dep"]
            children = [[] for _ in tk]
            for i, p in enumerate(par):
                if p != -1: children[p].append(i)
            tv = time.time()
            send_msg(pipe_sock, {"op": "verify", "token_ids": tk, "par": par, "dep": dep,
                                 "start": pos, "gather": gather_prev})
            targ = recv_msg(rx)                                # one argmax per tree node
            t_verify += time.time() - tv
            committed, kept = accept_tree(tk, par, {i: c for i, c in enumerate(children)}, targ)
            out.extend(committed); cur = committed[-1]
            gather_prev = (pos, kept)                          # static-cache compaction for next round
            pos += len(kept)
            rounds += 1; accepted_total += len(kept) - 1; tnodes += len(tk)
            if eos in committed:
                break
    except EDGE_ERRORS as e:
        raise TransportError(f"tree edge failed at token {len(out)} ({type(e).__name__}: {e})") from e
    dt = time.time() - t0
    if eos in out:
        out = out[:out.index(eos)]
    return {"text": tok.decode(out, skip_special_tokens=True), "n_tokens": len(out), "rounds": rounds,
            "mean_accept": accepted_total / max(rounds, 1),
            "toks_per_traversal": len(out) / max(rounds, 1), "tree_nodes": tnodes / max(rounds, 1),
            "tok_s": len(out) / max(dt, 1e-9),
            "draft_ms": t_draft / max(rounds, 1) * 1000, "verify_ms": t_verify / max(rounds, 1) * 1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--nstages", type=int, required=True)
    ap.add_argument("--lo", type=int, default=-1, help="explicit layer-block start (uneven VRAM-aware split; -1 = even)")
    ap.add_argument("--hi", type=int, default=-1, help="explicit layer-block end (exclusive; -1 = even split)")
    ap.add_argument("--coordinator", action="store_true", help="in-house entry node: draft + drive, no 120B layers")
    ap.add_argument("--served-head", action="store_true", help="stage 0 runs as a swarm serve node (embeds token ids)")
    ap.add_argument("--direct-return", action="store_true", help="tail sends results straight to the coordinator (1 hop, not relayed)")
    ap.add_argument("--tail", default="", help="coordinator: host:port of the tail, for the direct return channel")
    ap.add_argument("--tree", default="", help="coordinator: tree spec 'width,depth' (e.g. 3,6) -> tree speculation")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")        # target
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--listen-port", type=int, default=29501)
    ap.add_argument("--next", default="")
    ap.add_argument("--device", default="cuda:0")           # this stage's block
    ap.add_argument("--draft-device", default="cuda:1")     # draft (head only; its own GPU)
    ap.add_argument("--draft-server", default="", help="host:port of the in-house vLLM draft service (else local draft)")
    ap.add_argument("--ngram-draft", action="store_true", help="coordinator: model-free n-gram/prompt-lookup drafter "
                    "(no draft server, no KV) — the long-context spec-decode path; survives past 100k where the 20b draft OOMs")
    ap.add_argument("--ngram-n", type=int, default=3, help="n-gram suffix length for --ngram-draft (tune 2-4)")
    ap.add_argument("--reasoning", default="", help="gpt-oss reasoning_effort (low|medium|high); low cuts the "
                    "analysis channel so the answer starts sooner (key for copy/retrieval-heavy long-ctx tasks)")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--adaptive", action="store_true", help="tune K live from the running acceptance rate")
    ap.add_argument("--fast", action="store_true", help="serve node: static-cache CUDA-graph verify (~5x, fixed-K linear)")
    ap.add_argument("--attn", default="eager", help="attention impl for stages: eager | flex_attention "
                    "(flex = O(n) prefill on Ada, needed for long context with gpt-oss sinks)")
    ap.add_argument("--prompt-file", default="", help="read the prompt from this file (for 100k-token prompts)")
    ap.add_argument("--draft-ctx", type=int, default=0, help="window the draft query to the last N tokens "
                    "(0=full); keeps the draft fast at long context (a full-95k draft is ~800ms/round)")
    ap.add_argument("--prefill-chunk", type=int, default=0, help="chunk the prefill into this many tokens "
                    "(0=one shot); long prompts need chunking so per-chunk activations stay bounded")
    ap.add_argument("--max-ctx", type=int, default=2048, help="fast-verify static cache size (prompt+gen ceiling); "
                    "sized to the request, not a hardware limit — the KV cache is ~tens of KB/token, so a 4090 "
                    "stage holds far more than the old 2048 default. overflow fails clean (ContextOverflow), never corrupts")
    ap.add_argument("--sweep", default="", help="comma K list to measure on one load, 0=adaptive (e.g. 2,3,4,0)")
    ap.add_argument("--pipe", action="store_true", help="coordinator: PIPELINED spec-decode (depth chunks in flight; needs --direct-return)")
    ap.add_argument("--depth", type=int, default=4, help="pipelined coordinator: verify chunks in flight")
    ap.add_argument("--prefill-depth", type=int, default=8, help="pipelined coordinator: PREFILL chunks in flight "
                    "(overlap prefill across stages; >=nstages fills the pipe -> ~Nstage-fold faster TTFT)")
    ap.add_argument("--temp", type=float, default=0.0, help="sampling temperature (0=greedy/argmax, exact legacy path); "
                    ">0 enables LOSSLESS speculative sampling at the tail (temp/top-p/top-k)")
    ap.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling cutoff (with --temp>0)")
    ap.add_argument("--top-k", type=int, default=0, help="top-k sampling cutoff (0=off; with --temp>0)")
    ap.add_argument("--seed", type=int, default=0, help="tail sampler seed (reproducible sampled runs / receipts)")
    ap.add_argument("--n-layers", type=int, default=0, help="coordinator: the model's TRUE layer count for the "
                    "receipt coverage check (e.g. 36 for gpt-oss-120b); 0 derives it from the receipts "
                    "themselves, which a layer-omitting ring can game (warned loudly)")
    ap.add_argument("--sample-test", type=int, default=0, help="coordinator: draw N iid next-tokens via PLAIN vs "
                    "SPECULATIVE sampling and report TV distance (on-swarm losslessness proof); needs --ngram-draft")
    ap.add_argument("--resume-file", default="", help="coordinator: versioned checkpoint (as written by --ft-dump) "
                    "to RESUME from (re-prefill prompt+committed on a healed ring, continue) — fault tolerance; "
                    "a checkpoint from another prompt/model/settings is REFUSED (CheckpointError)")
    ap.add_argument("--ft-dump", default="", help="coordinator: run resumable + atomically write the versioned "
                    "checkpoint {schema,job,prompt_sha256,...,output_ids,ids_sha256} here on completion OR "
                    "mid-request node death (exit 3 if a node died), so the control plane can heal+resume")
    ap.add_argument("--compare", action="store_true", help="coordinator: SYNC then PIPE (cold+warm) in ONE process for a clean A/B")
    ap.add_argument("--depths", default="2,4,8", help="--compare: pipe depths to sweep (one process)")
    ap.add_argument("--ks", default="4", help="--compare: K values to sweep (one process; graph recaptures per K)")
    ap.add_argument("--tree-fast", default="", help="coordinator: FAST graphed tree spec 'w,d' (cold+warm)")
    ap.add_argument("--dump", default="", help="--pipe: write {prompt, output_ids, tok_s} JSON here (for the receipt)")
    ap.add_argument("--prompt", default="Explain decentralized computing in two sentences.")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()
    wire.key_from_env()                 # shared swarm key (SHARD_PSK); fail fast before the model load
    if args.prompt_file:                # long (100k) prompts can't fit on the CLI
        args.prompt = open(args.prompt_file).read()

    if args.coordinator:                                    # in-house entry node: no 120B, just tokenizer + draft + swarm
        tok = AutoTokenizer.from_pretrained(args.model)     # 20b tokenizer == 120b tokenizer
        local_draft = None
        if args.ngram_draft:                                # model-free drafter: no draft server/socket needed
            local_draft = NgramDrafter(ng=args.ngram_n)
            draft_sock = None
            print(f"[coord] n-gram drafter (ng={args.ngram_n}); no draft server", flush=True)
        else:
            dh, dp = args.draft_server.split(":")
            draft_sock = socket.socket(); draft_sock.connect((dh, int(dp)))
        host, port = args.next.split(":")
        pipe_sock = socket.socket(); pipe_sock.settimeout(args.timeout); pipe_sock.connect((host, int(port)))
        ret_sock = None
        if args.direct_return:                              # open the return channel to the tail (once)
            th, tp = args.tail.split(":")
            ret_sock = socket.socket(); ret_sock.settimeout(args.timeout); ret_sock.connect((th, int(tp)))
            send_msg(ret_sock, {"op": "hello_return"})
            print(f"[coord] direct-return channel to tail at {args.tail}", flush=True)
        print(f"[coord] in-house draft {args.draft_server} + swarm stage 0 at {args.next}; generating ...", flush=True)
        if args.tree:                                       # TREE speculation
            w, d = (int(x) for x in args.tree.split(","))
            cfg = {"width": w, "depth": d}
            for _ in range(2):                              # cold + warm
                r = coordinate_tree(draft_sock, pipe_sock, tok, args.prompt, cfg, args.max_new, args.timeout, ret_sock=ret_sock)
                print(f"[TREE w={w},d={d}] {r['tok_s']:.2f} tok/s | {r['toks_per_traversal']:.2f} tok/traversal | "
                      f"accept {r['mean_accept']:.2f}/round | {r['tree_nodes']:.0f} tree nodes | "
                      f"draft {r['draft_ms']:.0f}ms + verify {r['verify_ms']:.0f}ms/round", flush=True)
            print(f"\n[coord] === OUTPUT ===\n{r['text']}\n", flush=True)
            return
        if args.tree_fast:                                 # FAST graphed tree spec (cold + warm), sweep 'w,d;w,d'
            for spec in args.tree_fast.split(";"):
                w, d = (int(x) for x in spec.split(","))
                cfg = {"width": w, "depth": d}
                for i in range(2):
                    r = coordinate_tree_fast(draft_sock, pipe_sock, tok, args.prompt, cfg, args.max_new,
                                             args.timeout, ret_sock=ret_sock)
                    print(f"[TREE-FAST w={w},d={d} {'warm' if i else 'cold'}] {r['tok_s']:.2f} tok/s | "
                          f"{r['toks_per_traversal']:.2f} tok/trav | accept {r['mean_accept']:.2f} | "
                          f"{r['tree_nodes']:.0f} nodes | draft {r['draft_ms']:.0f}ms verify {r['verify_ms']:.0f}ms", flush=True)
            print(f"\n[coord] === OUTPUT ===\n{r['text'][:400]}\n", flush=True)
            return
        if args.compare:                                   # SYNC then PIPE in ONE process (clean warm A/B)
            depths = [int(x) for x in args.depths.split(",")]
            sync_warm = pipe_warm = None
            for K in [int(x) for x in args.ks.split(",")]:
                for i in range(2):                         # sync: cold (captures K+1 graph), then warm
                    r = coordinate(draft_sock, pipe_sock, tok, args.prompt, K, args.max_new, args.timeout, ret_sock=ret_sock)
                    if i: sync_warm = r
                    print(f"[SYNC K={K} {'warm' if i else 'cold'}] {r['tok_s']:.2f} tok/s | "
                          f"{r['toks_per_traversal']:.2f} tok/trav | accept {r['mean_accept']:.2f} | "
                          f"draft {r['draft_ms']:.0f}ms verify {r['verify_ms']:.0f}ms", flush=True)
                for d in depths:
                    for i in range(2):                     # pipe: cold, then warm, at each depth
                        r = coordinate_pipe(draft_sock, pipe_sock, tok, args.prompt, K, args.max_new,
                                            args.timeout, d, ret_sock=ret_sock)
                        if i: pipe_warm = r
                        print(f"[PIPE K={K} depth={d} {'warm' if i else 'cold'}] {r['tok_s']:.2f} tok/s | "
                              f"{r['toks_per_traversal']:.2f} tok/trav | accept {r['mean_accept']:.2f} | "
                              f"+{r['wasted']} stale | draft {r['draft_ms']:.0f}ms recv {r['recv_ms']:.0f}ms", flush=True)
            if args.dump and sync_warm and pipe_warm:      # receipt: pipe ids + the sync-vs-pipe lossless check
                import json, hashlib
                sids, pids = sync_warm["output_ids"], pipe_warm["output_ids"]
                json.dump({"prompt": args.prompt, "model": args.model,
                           "tok_s_warm": round(pipe_warm["tok_s"], 2), "n_tokens": pipe_warm["n_tokens"],
                           "output_ids": pids, "output_text": pipe_warm["text"],
                           "output_sha256": hashlib.sha256(json.dumps(pids).encode()).hexdigest(),
                           "tokens_match_sync": (sids == pids)}, open(args.dump, "w"))
                print(f"[coord] dumped receipt run -> {args.dump} | tokens_match_sync={sids == pids}", flush=True)
            print(f"\n[coord] === sample output ===\n{r['text'][:400]}\n", flush=True)
            return
        if args.sample_test:                               # on-swarm losslessness proof for sampling
            import json as _json
            r = sample_distribution_test(pipe_sock, tok, args.prompt, args.timeout, ret_sock, local_draft,
                                         args.K, temp=(args.temp if args.temp > 0 else 1.0),
                                         top_p=args.top_p, top_k=args.top_k, seed=args.seed,
                                         prefill_chunk=args.prefill_chunk, reasoning=(args.reasoning or None),
                                         n_per=args.sample_test)
            print(f"\n[SAMPLE-TEST] temp={r['temp']} top_p={r['top_p']} n_per={r['n_per']} stops={r['stops']} "
                  f"(high-entropy>=1bit: {r['high_entropy_stops']}, max_entropy={r['max_entropy']:.2f} bits)", flush=True)
            print(f"  mean TV(spec, plain)  over high-entropy stops = {r['mean_tv_spec_vs_plain']:.4f}", flush=True)
            print(f"  mean TV(plain, plain) noise floor             = {r['mean_tv_noise_floor']:.4f}", flush=True)
            verdict = "LOSSLESS — spec-sampling distribution == plain sampling within MC noise" \
                if r['mean_tv_spec_vs_plain'] <= 1.5 * r['mean_tv_noise_floor'] + 0.01 else "MISMATCH"
            print(f"  VERDICT: {verdict}", flush=True)
            he = sorted([s for s in r["per_stop"] if s["entropy"] >= 1.0], key=lambda s: -s["entropy"])[:3]
            for s in he:
                print(f"  -- stop entropy={s['entropy']:.2f}b tv(spec,plain)={s['tv_spec_plain']:.3f} "
                      f"tv(plain,plain)={s['tv_plain_plain']:.3f}", flush=True)
                for t in s["top"]:
                    print(f"       {repr(t['text'])[:14]:14} plain={t['plain']:.3f} spec={t['spec']:.3f}", flush=True)
            if args.dump:
                _json.dump({"test": "spec-sampling-losslessness", "model": args.model, "verdict": verdict, **r},
                           open(args.dump, "w"))
                print(f"[coord] dumped sample-test -> {args.dump}", flush=True)
            return
        if args.ft_dump:                                   # FAULT-TOLERANT run: resumable, dump partial on node death
            import sys as _sys
            ck_env = checkpoint_env(prompt=args.prompt, model=args.model,
                                    tokenizer=getattr(tok, "name_or_path", args.model),
                                    settings={"temp": args.temp, "top_p": args.top_p, "top_k": args.top_k,
                                              "seed": args.seed, "reasoning": args.reasoning or None})
            resume_ids = load_checkpoint(args.resume_file, ck_env) if args.resume_file else None
            r = coordinate_pipe(draft_sock, pipe_sock, tok, args.prompt, args.K, args.max_new, args.timeout,
                                args.depth, ret_sock=ret_sock, prefill_chunk=args.prefill_chunk,
                                draft_ctx=args.draft_ctx, local_draft=local_draft, reasoning=(args.reasoning or None),
                                temp=args.temp, top_p=args.top_p, top_k=args.top_k, seed=args.seed,
                                prefill_depth=args.prefill_depth, resume_ids=resume_ids, resumable=True)
            write_checkpoint(args.ft_dump, ck_env, r.get("output_ids", []),
                             ok=r.get("ok", False), n_tokens=r.get("n_tokens", 0), text=r.get("text", ""),
                             error=r.get("error"), tok_s=round(r.get("tok_s", 0.0), 2),
                             prefill_s=round(r.get("prefill_s", 0.0), 2),
                             resume_tokens=(len(resume_ids) if resume_ids else 0))
            status = "OK" if r.get("ok") else f"NODE-DEATH (committed {r.get('n_tokens', 0)} tok)"
            print(f"[coord] FT run {status} -> {args.ft_dump} | {r.get('n_tokens',0)} tok"
                  f"{' | '+r['error'] if r.get('error') else ''}", flush=True)
            print(f"\n[coord] === OUTPUT ({r.get('n_tokens',0)} tok) ===\n{r.get('text','')[:600]}\n", flush=True)
            _sys.exit(0 if r.get("ok") else 3)
        if args.pipe:                                      # PIPELINED coordinator (depth chunks in flight)
            ks = [int(x) for x in args.sweep.split(",")] if args.sweep else [args.K]
            for kv in ks:
                r = coordinate_pipe(draft_sock, pipe_sock, tok, args.prompt, kv, args.max_new,
                                    args.timeout, args.depth, ret_sock=ret_sock, prefill_chunk=args.prefill_chunk,
                                    draft_ctx=args.draft_ctx, local_draft=local_draft,
                                    reasoning=(args.reasoning or None), temp=args.temp, top_p=args.top_p,
                                    top_k=args.top_k, seed=args.seed, prefill_depth=args.prefill_depth)
                print(f"[PIPE K={kv} depth={args.depth} temp={args.temp}] {r['tok_s']:.2f} tok/s | "
                      f"{r['toks_per_traversal']:.2f} tok/traversal | "
                      f"accept {r['mean_accept']:.2f} | +{r['wasted']} stale | prefill {r['prefill_s']:.1f}s | "
                      f"draft {r['draft_ms']:.0f}ms recv {r['recv_ms']:.0f}ms/round", flush=True)
            if args.dump:
                import json, hashlib
                ids = r["output_ids"]
                rec = {"prompt": args.prompt, "model": args.model, "K": ks[-1], "depth": args.depth,
                       "tok_s_warm": round(r["tok_s"], 2), "n_tokens": r["n_tokens"],
                       "prompt_tokens": r.get("prompt_tokens"), "prefill_s": round(r.get("prefill_s", 0.0), 2),
                       "prefill_depth": args.prefill_depth, "prefill_chunk": args.prefill_chunk,
                       "temp": args.temp, "top_p": args.top_p, "top_k": args.top_k, "seed": args.seed,
                       "mean_accept": round(r["mean_accept"], 3), "toks_per_traversal": round(r["toks_per_traversal"], 3),
                       "decode": ("greedy (exact)" if args.temp <= 0 else f"sampling temp={args.temp} top_p={args.top_p}"),
                       "output_ids": ids, "output_text": r["text"],
                       "output_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
                json.dump(rec, open(args.dump, "w"))
                print(f"[coord] dumped run -> {args.dump} (sha256 {rec['output_sha256'][:16]}..)", flush=True)
            if RECEIPTS:                                   # PROVE: sweep the ring once for signed per-stage receipts
                rx = ret_sock if ret_sock is not None else pipe_sock
                send_msg(pipe_sock, {"op": "receipt", "receipts": []})
                recs = recv_msg(rx)
                print(f"\n[coord] === PROVE: {len(recs)} signed per-stage receipts ===", flush=True)
                ok = True
                for rr in recs:
                    body = {k: v for k, v in rr.items() if k != "stage"}
                    try:
                        verify_receipt(body)
                        print(f"  stage {rr['stage']}: layers[{rr['layer_start']}:{rr['layer_end']}] "
                              f"n={rr['n_chunks']} in_root {rr['in_root'][:12]} out_root {rr['out_root'][:12]} "
                              f"pub {rr['pubkey'][:12]} — sig VALID", flush=True)
                    except Exception as e:
                        ok = False; print(f"  stage {rr['stage']}: sig FAILED ({e})", flush=True)
                try:
                    total = args.n_layers or max(rr["layer_end"] for rr in recs)
                    if not args.n_layers:                  # derived from the receipts under test = self-referential:
                        print("  ⚠ coverage target derived from the receipts themselves (pass --n-layers to pin "
                              "the model's true depth — a layer-omitting ring passes without it)", flush=True)
                    verify_coverage([{k: v for k, v in rr.items() if k != "stage"} for rr in recs], total)
                    print(f"  coverage: blocks tile [0:{total}] no gap/overlap — every layer attested by a distinct signed node", flush=True)
                except Exception as e:
                    ok = False; print(f"  coverage FAILED: {e}", flush=True)
                print(f"[coord] PROVE verdict: {'ALL receipts valid + full layer coverage — coordinator cannot fabricate, no node paid without proving its block' if ok else 'FAILED'}", flush=True)
            print(f"\n[coord] === OUTPUT ===\n{r['text']}\n", flush=True)
            return
        ks = [int(x) for x in args.sweep.split(",")] if args.sweep else [args.K]
        for kv in ks:
            adaptive = (kv == 0) or (not args.sweep and args.adaptive)
            r = coordinate(draft_sock, pipe_sock, tok, args.prompt, (6 if kv == 0 else kv),
                           args.max_new, args.timeout, adaptive=adaptive, ret_sock=ret_sock,
                           prefill_chunk=args.prefill_chunk)
            if args.sweep:
                print(f"[SWEEP K={kv}] {r['tok_s']:.2f} tok/s | {r['toks_per_traversal']:.2f} tok/traversal | "
                      f"accept {r['mean_accept']:.2f} | draft {r['draft_ms']:.0f}ms + verify {r['verify_ms']:.0f}ms/round", flush=True)
            else:
                print(f"\n[coord] === OUTPUT ===\n{r['text']}\n", flush=True)
                print(f"[coord] {r['n_tokens']} tok | {r['tok_s']:.2f} tok/s | {r['toks_per_traversal']:.2f} tok/traversal | "
                      f"accept {r['mean_accept']:.2f} | draft {r['draft_ms']:.0f}ms + verify {r['verify_ms']:.0f}ms/round", flush=True)
        if args.dump:                                       # sync output ids + hash (for the receipt / a transport A/B)
            import json, hashlib
            ids = r["output_ids"]
            json.dump({"prompt": args.prompt, "model": args.model, "K": ks[-1], "mode": "sync",
                       "tok_s": round(r["tok_s"], 2), "n_tokens": r["n_tokens"], "output_ids": ids,
                       "output_text": r["text"],
                       "output_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}, open(args.dump, "w"))
            print(f"[coord] dumped run -> {args.dump} (sha256 {hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]}..)", flush=True)
        return

    _lo = args.lo if args.lo >= 0 else None
    _hi = args.hi if args.hi >= 0 else None
    parts = load_stage(args.model, args.stage, args.nstages, device=args.device, attn=args.attn, lo=_lo, hi=_hi)

    if args.stage != 0 or args.served_head:                 # swarm serve node (stage 0 embeds token ids)
        is_tail = args.stage == args.nstages - 1
        if args.direct_return and is_tail:
            if args.fast:
                serve_tail_fast(parts, args.listen_port, args.timeout, args.device, max_ctx=args.max_ctx)
            else:
                serve_tail_direct(parts, args.listen_port, args.timeout, args.device)
        elif args.fast:
            serve_spec_fast(parts, args.stage, args.nstages, args.listen_port, args.next, args.timeout,
                            args.device, direct=args.direct_return, max_ctx=args.max_ctx)
        else:
            serve_spec(parts, args.stage, args.nstages, args.listen_port, args.next, args.timeout,
                       args.device, direct=args.direct_return)
        return

    draft, draft_sock = None, None
    if args.draft_server:                                   # in-house vLLM draft service
        dh, dp = args.draft_server.split(":")
        draft_sock = socket.socket(); draft_sock.connect((dh, int(dp)))
        print(f"[s0] using in-house draft service at {args.draft_server}", flush=True)
    else:
        print(f"[s0] loading draft {args.draft} on {args.draft_device} ...", flush=True)
        draft = AutoModelForCausalLM.from_pretrained(args.draft, dtype="auto",
                                                     device_map={"": args.draft_device},
                                                     attn_implementation="eager").eval()
        print(f"[s0] draft loaded, draft_mem={torch.cuda.memory_allocated(args.draft_device)/1e9:.1f}GB", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    host, port = args.next.split(":")
    sock = socket.socket(); sock.settimeout(args.timeout); sock.connect((host, int(port)))
    print(f"[s0] connected forward to stage 1 at {args.next}; K={args.K}; generating ...", flush=True)
    if args.sweep:                          # load once, measure several K on one connection
        try:
            for kv in [int(x) for x in args.sweep.split(",")]:
                adaptive = (kv == 0)
                rr = generate_spec(draft, parts, tok, sock, args.prompt, (6 if adaptive else kv),
                                   args.max_new, args.device, args.draft_device, args.timeout,
                                   adaptive=adaptive, draft_sock=draft_sock)
                tag = f"adaptive(mean {rr['mean_K']:.1f})" if adaptive else f"K={kv}"
                print(f"[SWEEP {tag}] {rr['tok_s']:.2f} tok/s | {rr['toks_per_traversal']:.2f} tok/traversal | "
                      f"accept {rr['mean_accept']:.2f} | draft {rr['draft_ms']:.0f}ms + verify {rr['verify_ms']:.0f}ms/round "
                      f"-> async ceiling {rr['toks_per_traversal']/(max(rr['draft_ms'],rr['verify_ms'])/1000):.1f} tok/s", flush=True)
        finally:
            sock.close()
        return
    try:
        r = generate_spec(draft, parts, tok, sock, args.prompt, args.K, args.max_new,
                          args.device, args.draft_device, args.timeout, adaptive=args.adaptive,
                          draft_sock=draft_sock)
    except TransportError as e:
        print(f"\n[s0] TRANSPORT FAILURE: {e}", flush=True); raise SystemExit(2)
    finally:
        sock.close()
    kdesc = f"adaptive (mean {r['mean_K']:.1f}, {r['k_lo']}-{r['k_hi']})" if args.adaptive else f"{args.K}"
    print(f"\n[s0] === OUTPUT ===\n{r['text']}\n", flush=True)
    print(f"[s0] {r['n_tokens']} tokens in {r['rounds']} verify traversals | "
          f"mean accepted/round {r['mean_accept']:.2f} | {r['toks_per_traversal']:.2f} tokens/traversal "
          f"(vs 1.0 plain) | {r['tok_s']:.2f} tok/s | draft={args.draft.split('/')[-1]} K={kdesc} "
          f"({args.nstages}-stage pipeline)", flush=True)


if __name__ == "__main__":
    main()
