"""MiniMax-M2.5 PIPELINED ring — direct-return stages + the PROVEN coordinate_pipe coordinator.

The lean m25_ring driver was synchronous (one ring traversal at a time), so every token paid the
full ~280ms ring latency. specpipe.coordinate_pipe keeps `depth` verify chunks IN FLIGHT (the GLM
2.9->16.6 lever) and is model-agnostic — it only orchestrates token-ids + argmax over sockets, with
a pluggable n-gram drafter. So we reuse it UNCHANGED and only provide M2.5-native stage serve loops
that speak its wire protocol: reset / verify(token_ids|h, start). The KV is purely start-based — a
fresh chunk at an earlier `start` overwrites stale speculative KV — which is EXACTLY m25_stage's
crop-to-start behaviour, so rollback needs no extra bookkeeping. Direct-return: middle stages
fire-forward, the tail returns straight to the coordinator (serve_tail_direct's 2-connection model).

  stage:  SHARD_TRANSPORT=libp2p M25_DIR=/root/m25 python m25_pipe.py stage --stage 0 --nstages 5 \
              --lo 0 --hi 10 --port 29610 --next 127.0.0.1:29611
  coord:  SHARD_TRANSPORT=libp2p M25_DIR=/root/m25 python m25_pipe.py coord --head 127.0.0.1:29610 \
              --tail 127.0.0.1:29612 --K 6 --depth 4 --max-new 256 --prompt-file p.txt
"""
import os, sys, socket, select, time, threading, argparse, hashlib, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("M25_DIR", "/root/m25")
import json
import m25_stage as S
from m25_tools import render_ids, parse_completion          # tool-calling: chat-template render + output parse
from node_kv import send_msg, recv_msg, EDGE_ERRORS, TransportError   # libp2p codec (SHARD_TRANSPORT=libp2p)
try:                                                    # opt-in confidence-scheduled depth (M25_CONF_SCHED=1)
    from confidence import ConfidenceScheduler
except Exception:
    ConfidenceScheduler = None

dev = "cuda"
NODELAY = (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def _keepalive(s):
    """Bound silent-peer death (no FIN: box power-loss, NAT idle expiry) to ~2min instead of NEVER: the
    churn-resilient tail waits in select(), which cannot see a half-open predecessor, and the old 600s
    recv-timeout teardown is gone (that timeout WAS the idle-wedge). Keepalive makes the kernel probe the
    peer and error the socket, which wakes select and routes through the normal death paths. Matters on
    direct-TCP rings; libp2p conns are loopback-to-sidecar (never half-open)."""
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 20)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
    except (OSError, AttributeError):                    # non-Linux: at least the plain keepalive attempt
        pass

try:                                                    # PROVE: opt-in signed per-stage receipts (trustless verify)
    from receipt import ReceiptSigner, load_or_make_node_key, verify_receipt, verify_coverage
except Exception:
    ReceiptSigner = None
RECEIPTS = bool(os.environ.get("SHARD_RECEIPTS")) and ReceiptSigner is not None
NODE_KEY_PATH = os.environ.get("SHARD_NODE_KEY", "/root/.shard_node_key")

# opt-in fp8 activations on the wire: the MEASURED per-hop bottleneck is moving the bf16 activation, so
# transporting it as fp8 (e4m3) halves bytes/hop (~2x tok/s) at a small, MEASURED precision cost. Lossy but
# still deterministic+verifiable (receipts hash the fp8-engine's activations). bf16 path is unchanged when off.
M25_FP8_WIRE = bool(int(os.environ.get("M25_FP8_WIRE", "0")))
# fp8 EAGLE-aux on the wire: with M25_EAGLE the aux hidden states (3 layers x [K+1,H] bf16 ~ 166KB/hop)
# are ~3x the activation payload itself and ride EVERY hop + the return. They only feed the DRAFTER
# (losslessness untouched — the ring still greedy-verifies), so fp8 is safe by construction; only accept
# could move, and per-tensor-scale e4m3 on O(1) hidden states is well inside the head's tolerance.
# Defaults to the M25_FP8_WIRE setting (one "fp8 on the wire" concept); override with M25_FP8_AUX=0/1.
M25_FP8_AUX = bool(int(os.environ.get("M25_FP8_AUX", "1" if M25_FP8_WIRE else "0")))
# Per-job CUDA-graph arm for the interleaved A/B: when not None the coordinator stamps {"graph": bool}
# on its reset op and every stage flips its runtime graph route for that job (S.set_graph via
# _reset_flags; the ring launches with M25_STATIC_KV=1, M25_CUDA_GRAPH unset). None (default) = field
# absent, stages keep their current route — old coordinators and old stages are mutually unaffected.
# Set via env, or flipped in-process between jobs by a bench (m25_pipe.M25_GRAPH_JOB = True/False).
M25_GRAPH_JOB = os.environ.get("M25_GRAPH_JOB")
if M25_GRAPH_JOB is not None:
    M25_GRAPH_JOB = M25_GRAPH_JOB.strip().lower() in ("1", "true", "yes", "on")   # explicit truthy set: "no"/"off"/"0" = False
# Batched-decode graph route (M25_BATCH_GRAPH=0 = escape hatch): with graphs ACTIVE, verify_batch
# blocks go through a BatchGraphRunner (batched stages ran EAGER — the measured ~220ms B=4 ring+eager
# round floor). 0 pins batched decode eager WITHOUT touching the solo graph route, so a batched-graph
# regression can be isolated per-launch on a warm ring (the M25_EAGER-style per-lever hatch).
M25_BATCH_GRAPH = os.environ.get("M25_BATCH_GRAPH", "1") != "0"
# Accepted-prefix aux slimming (M25_AUX_SLIM=0 = escape hatch): batched rounds are TRANSPORT-bound
# (batched-levers-sweep-20260710) and the EAGLE aux return is a big slice of the round's bytes. The
# tail recomputes the coordinator's accept rule from the drafted rows (the head forwards them as
# 'tids') and returns aux SLICED to each stream's committed prefix — lossless (unsent rows were never
# read) and compat-gated (no tids on the frame -> full aux, so old/new builds mix cleanly). This
# trims the RETURN leg only (~the tail's 3 aux layers); the forward-leg aux is the head-local-aux
# follow-up. Solo path untouched.
M25_AUX_SLIM = os.environ.get("M25_AUX_SLIM", "1") != "0"
# Head-local aux (M25_AUX_LOCAL=1, BATCHED jobs, default OFF until live-validated): the head stage's
# own aux layers (L1 at the standard shape) are produced ON THE COORDINATOR'S BOX yet ride every WAN
# leg — ~40% of a batched round's bytes (aux-layer-legs L1=6/L30=3/L58=1 vs 6 activation legs; same %
# at every B and wire mode). With the flag, a batched job opts in per job (reset_batch 'aux_local' ->
# the head acks 'aux_local_ok' on the pipe socket BEFORE forwarding the reset), then the head sends
# {op:aux_local, job, seq, aux} on the pipe AFTER each ring forward (forward-first: a big local frame
# must never block the ring send) and OMITS its own aux from the forward frames. The coordinator
# pulls exactly one local frame per aux-producing frame it sent, paired by (job, seq): any skew
# aborts LOUD — a silently mispaired aux row would degrade g invisibly, the worst drafter bug class.
# Old head (no ack) -> degrade to the ridden-ring path. Solo jobs never stamp the field.
# Design + failure matrix: .claude/plans/head-local-aux.md. Default ON since 2026-07-11: live-proven
# (receipt batched-viability-20260711 - 3 armed passes, all receipts valid, zero pairing aborts,
# ~20-30% round-time cut at fp8; the A/B knob remains for measurement).
M25_AUX_LOCAL = os.environ.get("M25_AUX_LOCAL", "1") != "0"
# De-lockstep (M25_DELOCKSTEP): batched EAGLE jobs run per-stream ASYNC solo-style frames that
# interleave on the ring instead of one lockstep [B,...] frame per round — the streams ARE the
# pipeline (each stream's WAN wait hides the others' compute+drafting; zero staleness, per-stream
# commit cadence, done streams stop sending, decode MoE back on the solo token-count-invariant
# path). The per-stream-20 plan's build #1 (.claude/plans/per-stream-20-plan.md). Default ON since
# 2026-07-11: live-proven (receipt perstream-delockstep-20260711 — 3 reps, receipts valid, B=4
# per-stream 1.6-2.7x over lockstep same-ring); the env stays as the lockstep A/B hatch.
M25_DELOCKSTEP = os.environ.get("M25_DELOCKSTEP", "1") != "0"

def _pack_h(h):
    """fp8 (e4m3) per-tensor quantize a hidden-state activation for transport. A per-tensor scale keeps the
    residual-stream OUTLIER channels inside e4m3's ±448 range. Returns (fp8_cpu_tensor, float_scale)."""
    scale = (h.detach().abs().amax() / 448.0).clamp(min=1e-8)
    return (h / scale).to(torch.float8_e4m3fn).cpu(), float(scale)

def _unpack_h(q, scale):
    return q.to(torch.bfloat16) * scale                 # cpu bf16; caller moves .to(dev)

def _hsend(msg):
    """Outgoing stage message: fp8-pack the activation 'h' when M25_FP8_WIRE (half the bytes/hop), else just
    move it to cpu as before. Only touches 'h'/'h8' — aux/other fields are untouched (aux stays bf16)."""
    h = msg.get("h")
    if torch.is_tensor(h):
        if M25_FP8_WIRE and h.dtype != torch.float8_e4m3fn:
            msg["h"], msg["h8"] = _pack_h(h)
        else:
            msg["h"] = h.cpu()
    return msg

def _hrecv(msg):
    """Dequantize an fp8-packed activation back to bf16 in place (no-op on a bf16 message)."""
    if isinstance(msg, dict) and "h8" in msg:
        msg["h"] = _unpack_h(msg["h"], msg.pop("h8"))
    return msg


class _KeepWarm:
    """cwnd keep-warm for one SENDING socket. TCP slow-start-after-idle collapses cwnd whenever a
    connection idles >RTO, and our serial decode idles every ring leg for a full traversal between
    frames — measured on a 40ms-RTT vast leg (2026-07-05 leg probe): idle<=300ms keeps 30KB-1.6MB
    frames at ~1 RTT, idle=900ms costs 2-4 RTTs on the SAME frames (cwnd_p50 167->94). The kernel
    knob (tcp_slow_start_after_idle) is read-only in vast containers, so the engine keeps its own
    sockets warm: a daemon thread posts a tiny {"op":"noop"} frame whenever the socket has idled
    past `interval_ms` (idle=100ms measured to preserve cwnd fully). Noops are LEG-LOCAL: receivers
    skip them; they are never forwarded, answered, attested, or counted in metrics.

    EVERY real send on a wrapped socket must go through send() — it takes the same lock as the noop
    thread; two threads calling sendall() on one socket interleave partial frames and corrupt the
    stream. Interval from M25_CWND_KEEPWARM_MS (default 0 = OFF, master behavior unchanged) or
    set_interval() at runtime (the reset-op toggle).

    CHURN-SAFE: a stuck noop send must NEVER be able to block a reconnect/teardown (the paid-ring
    fault is a GPU/driver hang whose TCP stays alive, so sendall blocks up to the socket's PRODUCTION
    timeout — 1800s on the gateway, forever on the tail's untimed ret). Three guards compose: (1) the
    noop thread acquires the send lock NON-BLOCKING — if a real send holds it the leg isn't idle, so
    it skips the tick rather than queueing; (2) the noop send is time-bounded (<=2s) independent of
    the socket's production timeout, so it releases the lock in ~2s even against a full-buffer peer;
    (3) attach()/set_interval()/stop()/close() NEVER take the send lock (attach = atomic ref swap;
    lifecycle uses a separate _life lock + a lockless _stop flag), so teardown is never blockable by
    an in-flight noop. attach() swaps the socket on churn WITHOUT closing anything — lifecycle stays
    the serve loop's; a dead socket just makes the noop send fail silently until the loop replaces it."""

    def __init__(self, sock=None, interval_ms=None):
        self.sock = sock
        self.lock = threading.Lock()          # serializes sendall() on the wrapped socket ONLY; a stuck
        self._life = threading.Lock()         # noop send holds THIS — no teardown path takes it
        self._last = time.monotonic()         # (runner spawn/exit) — never held across a send
        self._interval = 0.0
        self._stop = False                    # lockless kill flag: a stuck send can't block stop()
        self._runner = None
        self.set_interval(os.environ.get("M25_CWND_KEEPWARM_MS", "0")
                          if interval_ms is None else interval_ms)

    def attach(self, sock):
        self.sock = sock                      # atomic ref swap under the GIL; NO send lock, so a stuck
        self._last = time.monotonic()         # noop send can never block a churn reconnect/teardown

    def set_interval(self, ms):
        with self._life:                      # _life (not the send lock): can't block on a stuck send
            self._interval = max(int(ms or 0), 0) / 1e3
            if self._interval > 0 and self._runner is None and not self._stop:   # 0 exits; >0 respawns
                self._runner = t = threading.Thread(target=self._run, daemon=True, name="cwnd-keepwarm")
                t.start()

    def send(self, obj):
        with self.lock:
            n = send_msg(self.sock, obj)
            self._last = time.monotonic()
            return n

    def _noop_once(self):
        """Send one noop, called holding self.lock. Bounds the lock-hold to <=~2s WITHOUT touching the
        socket's timeout: the old settimeout(2.0)-then-restore dance mutated shared socket state from
        a background thread, racing the job thread's own recv/settimeout on the same socket (a recv
        entered inside the window ran with the noop's 2s deadline -> spurious mid-decode timeouts) and,
        on a reused gateway socket, restoring a STALE timeout over the next job's. Instead wait <=2s
        for send-buffer space via select and skip when the peer is backed up — a full buffer means the
        leg isn't idle-cold anyway, and a tiny noop frame fits whatever space select just reported, so
        sendall completes without blocking. Re-reads self.sock ONCE into a local — attach() may swap it
        concurrently, and a noop racing the just-detached old socket is harmless. Swallows all errors:
        a dead socket is the serve loop's problem, never the noop thread's to crash on."""
        sock = self.sock
        if sock is None:
            return
        try:
            if not select.select([], [sock], [], 2.0)[1]:
                return                        # no buffer space: peer backed up -> skip this tick
            send_msg(sock, {"op": "noop"})
        except Exception:
            pass
        finally:
            self._last = time.monotonic()      # reset cadence even on failure — retry at interval, not spin

    def _run(self):
        me = threading.current_thread()
        while not self._stop and self._runner is me:
            iv = self._interval
            if iv <= 0:                                     # toggled off
                break
            try:
                if self.sock is not None and time.monotonic() - self._last > iv:
                    if self.lock.acquire(blocking=False):   # a real send in progress => not idle => skip,
                        try:                                # never queue behind (and get pinned by) a send
                            self._noop_once()
                        finally:
                            self.lock.release()
            except Exception:
                pass                                        # bulletproof: the noop thread never dies on error
            time.sleep(iv / 3.0)
        with self._life:                                    # clear identity so set_interval can respawn; if
            if self._runner is me:                          # a set_interval raced our exit, respawn now so a
                self._runner = None                         # toggle-on can't be lost to the exit window
                if self._interval > 0 and not self._stop:
                    self._runner = t = threading.Thread(target=self._run, daemon=True, name="cwnd-keepwarm")
                    t.start()

    def stop(self):
        """Kill the noop thread AND WAIT for it (M1). stop() is the job's finally — without the join a
        runner mid-noop outlived the job on the REUSED gateway socket, and its sendall interleaved with
        the NEXT job's frames (two _KeepWarm instances = two locks on one socket = corrupted stream).
        The join is bounded: a noop send holds the lock <=~2s (select-bounded, see _noop_once) and the
        runner re-checks _stop every iv/3 — so after stop() returns there is ONE lifetime sender per
        socket, ever. No lock is held across the join (_life is only taken by the runner's exit path)."""
        self._stop = True                                   # lockless: teardown must not wait on a stuck send
        t = self._runner
        if t is not None and t is not threading.current_thread():
            t.join(timeout=5.0)

    def close(self):
        self.stop()
        s, self.sock = self.sock, None
        if s is not None:
            try: s.close()
            except OSError: pass


def recv_data(sock):
    """recv_msg that skips keep-warm noop frames — a noop must never pop `inflight`, count as a
    reply/ack, or be receipts. Blocking sites only (the coordinator's rx and coord()'s ret_ok); the
    stages' select-multiplexed loops skip inline instead, so a noop-only idle peer can't starve the
    select set. A peer whose noop thread is alive but whose compute is wedged would reset the socket
    timeout on every noop, so the caller's recv timeout is enforced as an overall deadline here."""
    try:
        to = sock.gettimeout()
    except (OSError, AttributeError):
        to = None
    deadline = time.monotonic() + to if to else None
    while True:
        m = recv_msg(sock)
        if isinstance(m, dict) and m.get("op") == "noop":
            if deadline is not None and time.monotonic() > deadline:
                raise socket.timeout("recv timed out (peer sent only keepwarm noops)")
            continue
        return m


class JobRejected(Exception):
    """A stage rejected THIS job with a structured per-job error (e.g. KV overflow) — the ring is
    healthy, the JOB is dead. Deliberately NOT an OSError (same rationale as the gateway's
    ClientGone): EDGE_ERRORS recovery must never eat it, and retrying the identical request would
    only be rejected again. The gateway maps it to a 400, never a reconnect-retry."""


def _reply_ok(resp):
    """Gate a ring reply on the coordinator's return channel: a dict carrying 'error' is a stage's
    structured per-job rejection (the H1 backstop — the stage stayed alive, the job died) ->
    JobRejected. Everything else passes through untouched."""
    if isinstance(resp, dict) and "error" in resp:
        raise JobRejected(json.dumps(resp["error"]))
    return resp


def _reply_timeout(timeout):
    """Per-reply recv deadline for the coordinator's DECODE loop (F6). An internal-ring leg can blip
    while the coordinator is alive — the steady state on a permissionless ring — and the tail then
    holds the job stale until the coordinator's next reset (the PR #26 return-channel fix), dropping
    the in-flight replies meanwhile. So a blocked recv would otherwise wait the full production
    `timeout` (1800s on the gateway) before EDGE_ERRORS fires the resume/retry. Bound each decode
    round-trip to a few seconds instead, so blip failover is seconds not up-to-timeout. PREFILL keeps
    the full timeout (a big activation over a slow uplink is legitimately slow), as does the batched
    coordinator (throughput tier, not latency-failover-sensitive). Env M25_REPLY_TIMEOUT (seconds);
    0/empty disables (falls back to the full timeout); never longer than the production timeout."""
    try:
        hb = float(os.environ.get("M25_REPLY_TIMEOUT", "20") or "0")
    except ValueError:
        hb = 20.0
    return min(timeout, hb) if hb > 0 else timeout


def _act_digest(t):
    """Deterministic byte digest of an activation tensor for the receipt hash-chain (fp16 bytes)."""
    return t.detach().to(torch.float16).contiguous().cpu().numpy().tobytes()


def _verify_receipts(receipts, layer_count, expected_nonce=None, check_chain=False):
    """Coordinator-side PROVE: every per-stage receipt's signature must verify AND the blocks must
    tile [0:layer_count] with no gap/overlap — so no node is paid without proving its own block and
    the coordinator cannot fabricate one. layer_count is the model's TRUE depth (config), never
    derived from the receipts under test: a ring that omits layers must FAIL coverage, not shrink
    the target to whatever it did attest. `expected_nonce` binds the set to THIS job (rejects a
    replayed receipt); `check_chain` asserts adjacent out_root==in_root (bind to one real ring pass —
    lossless wire only). Returns True/False (fails closed). Prints a per-stage line."""
    bodies = [{k: v for k, v in rr.items() if k != "stage"} for rr in receipts]
    ok = True
    for rr, body in zip(receipts, bodies):
        try:
            verify_receipt(body)
            print(f"  stage {rr.get('stage')}: layers[{body['layer_start']}:{body['layer_end']}] "
                  f"n={body['n_chunks']} in_root {body['in_root'][:12]} out_root {body['out_root'][:12]} "
                  f"pub {body['pubkey'][:12]} — sig VALID", flush=True)
        except Exception as e:
            ok = False; print(f"  stage {rr.get('stage')}: sig FAILED ({e})", flush=True)
    try:
        verify_coverage(bodies, layer_count, expected_nonce=expected_nonce, check_chain=check_chain)
    except Exception as e:
        ok = False; print(f"  coverage FAILED: {e}", flush=True)
    return ok


def _unpack(resp):
    """EAGLE verify return is {"toks":[ids], "aux":{li:[s,H]}}; the plain path returns just [ids].
    fp8-packed aux entries ([fp8_tensor, scale] pairs, M25_FP8_AUX) are dequantized here once, so every
    downstream consumer (_eagle_seed/_eagle_aux_range/debug) sees plain bf16 [s,H] tensors."""
    if isinstance(resp, dict):
        aux = resp.get("aux")
        if aux:
            for k, v in aux.items():
                if isinstance(v, list):                 # [fp8_cpu_tensor, float_scale] from _merge_aux
                    aux[k] = _unpack_h(v[0], v[1])
        return resp.get("toks"), aux
    return resp, None


def _acc_stage_dt(resp, per_stage):
    """Coordinator-side fold of one reply's stage_dt rows ([stage, span_ms, compute_ms], stages launched with
    M25_STAGE_TIMING=1) into running per-stage sums; returns this traversal's (span_s, compute_s) totals."""
    sd = resp.get("stage_dt") if isinstance(resp, dict) else None
    if not sd:
        return 0.0, 0.0
    sp = cp = 0.0
    for row in sd:
        sp += row[1] / 1e3; cp += row[2] / 1e3
        a = per_stage.setdefault(str(row[0]), [0.0, 0.0, 0]); a[0] += row[1]; a[1] += row[2]; a[2] += 1
    return sp, cp


def _timing_fields(t_trav, t_stage, t_comp, per_stage):
    """Traversal/transport split for the coordinator return dict. traversal_s = Σ per-chunk send->recv
    latency (under pipelining depth>1 chunks overlap, so this can EXCEED wall decode_s — it is per-chunk
    latency, not wall). transport_s = traversal - Σ stage spans = wire + sidecar hops + codec serialize;
    None unless stages ran M25_STAGE_TIMING=1. per_stage_ms = {stage: [mean_span_ms, mean_compute_ms]}."""
    return {"traversal_s": round(t_trav, 3),
            "stage_s": round(t_stage, 3) if t_stage else None,
            "stage_compute_s": round(t_comp, 3) if t_stage else None,
            "transport_s": round(t_trav - t_stage, 3) if t_stage else None,
            "per_stage_ms": ({k: [round(v[0] / v[2], 2), round(v[1] / v[2], 2)] for k, v in per_stage.items()}
                             if per_stage else None)}


def _eagle_seed(aux, pos):
    """Stack the 3 aux hidden states at chunk position `pos` -> [3,H] for EagleDrafter.set_hidden()."""
    import torch as _t
    return _t.stack([aux[str(li)][pos] for li in S.EAGLE_AUX_LAYER_IDS], 0)


def _eagle_aux_range(aux, lo, hi):
    """Stack the 3 aux hidden states for chunk positions [lo,hi) -> [hi-lo,3,H] for EagleDrafter.extend()
    (one slice+stack per layer — not a per-position Python loop, which cost seconds on long prefills)."""
    import torch as _t
    return _t.stack([aux[str(li)][lo:hi] for li in S.EAGLE_AUX_LAYER_IDS], 1)


def _eagle_aux_range_b(aux, b, lo, hi):
    """Batched _eagle_aux_range: stack the 3 aux hidden states for STREAM b's chunk positions [lo,hi)
    -> [hi-lo,3,H]. verify_batch aux entries are [B,s,H] (every row kept); the coordinator slices its
    stream's row for that stream's drafter."""
    import torch as _t
    return _t.stack([aux[str(li)][b, lo:hi] for li in S.EAGLE_AUX_LAYER_IDS], 1)


def _aux_local_handshake(sock, token, wait_s=2.0):
    """Confirm the head's per-job aux_local opt-in on the (localhost) pipe socket. `token` is the
    job's UNIQUE nonce (minted per job — job_id defaults to "job" everywhere, so it can NOT
    disambiguate a dead job's residue from this job's; the review's F2): the head echoes it in the
    ok and every frame. The head sends aux_local_ok BEFORE forwarding the reset, and the tail's
    reset ack only arrives after the full ring traversal — so an armed head's ok is already buffered
    when this runs and the read returns instantly. The wait only elapses against an OLD head that
    never acks: return False and run the job on the ridden-ring path. (NOTE that degrade is sound
    only for genuinely OLD heads — a NEW armed head whose ok was somehow lost would omit its aux
    from the ring and the job would crash LOUD on the missing layer; acceptable: loud, and
    near-unreachable on a loopback lane.) Also DRAINS a dead job's stale frames off the reused
    gateway socket — foreign-token oks and frames are dropped, never terminate the handshake."""
    old = sock.gettimeout()
    try:
        sock.settimeout(wait_s)
        while True:
            try:
                m = recv_data(sock)
            except (socket.timeout, TransportError, OSError):   # old head / nothing buffered
                return False
            if isinstance(m, dict) and m.get("op") == "aux_local_ok" and m.get("job") == token:
                return True
            # a dead job's ok or stale aux frame: drop and keep draining
    finally:
        sock.settimeout(old)


def _pull_aux_local(sock, token, want_seq):
    """Pull the head's aux_local frame for the round the coordinator JUST received the ring reply
    for. Causality guarantees it is already buffered (the head sent it right after forwarding the
    frame whose reply completed the traversal), so this is a local read, not a wait. The (token,
    seq) pair MUST match: two FIFOs driven 1:1 by the same single-threaded sender can only skew if
    a frame was lost or a build is mixed wrong — and a silently mispaired aux row would degrade g
    invisibly, so any mismatch aborts the job LOUDLY instead."""
    m = recv_data(sock)
    if not (isinstance(m, dict) and m.get("op") == "aux_local"
            and m.get("job") == token and m.get("seq") == want_seq):
        raise TransportError(f"aux_local pairing broken: wanted token={token} seq={want_seq}, got "
                             f"{str(m)[:100]} — aborting (a mispaired aux would silently degrade g)")
    return m.get("aux") or {}


def _drain_aux_local(sock, n, wait_s=1.0):
    """Best-effort post-ABORT drain: consume up to the n outstanding local frames an aborted armed
    job left in flight. Without this the single-threaded head can sit BLOCKED mid-send_msg on a full
    localhost buffer (a prefill chunk's aux is tens of MB) — it then can't read the NEXT job's
    reset, and the only other drain point (the next handshake) sits BEHIND that reset's tail ack:
    a full-ring wedge on any reused pipe socket (the review's F1). A quiet lane (frame not yet
    sent / head gone) stops the drain early; errors are absorbed — the caller is already aborting."""
    try:
        old = sock.gettimeout()
        sock.settimeout(wait_s)
        try:
            for _ in range(n):
                recv_data(sock)
        finally:
            sock.settimeout(old)
    except Exception:
        pass


def _aux_keep_lens(tids, rows):
    """Per-stream count of aux rows the coordinator will CONSUME this round — EXACTLY its accept rule
    (the batch loop): n = longest prefix where draft ds[j] == argmax r[j]; it commits n+1 rows on a
    divergence (accepted prefix + the correction) or the full K+1 on a full accept (bonus token
    possible at depth 1). tids rows are [anchor]+ds (length K+1); rows are the tail's argmax [B][K+1].
    The TAIL runs this so it can slice the returned aux to what will actually be consumed — the aux
    payload is the dominant batched-round transport term. Lossless: rows past the commit are never
    read by the coordinator (an extra unsent row could only ever have sat unread)."""
    lens = []
    for b, row in enumerate(rows):
        ds = tids[b][1:]
        K = len(ds)
        n = 0
        for j in range(K):
            if ds[j] == row[j]:
                n += 1
            else:
                break
        lens.append(n + 1 if n < K else K + 1)
    return lens


def _slim_aux_b(aux, lens):
    """Slice each merged aux entry ([B,s,H] bf16 tensor, or fp8 [q, [scales]]) to per-stream
    accepted-prefix lengths, ragged-packed as ["slim", cat, lens, scales|None] (cat = kept rows
    concatenated over streams; fp8 rides as a uint8 byte view — slice/cat-safe on every torch build).
    Unknown entry shapes pass through FULL (fail-open: worst case is the old payload, never a broken
    drafter)."""
    out = {}
    for k, v in aux.items():
        if torch.is_tensor(v) and v.dim() == 3:                        # bf16 [B,s,H]
            out[k] = ["slim", torch.cat([v[b, :l] for b, l in enumerate(lens)], 0), list(lens), None]
        elif (isinstance(v, list) and len(v) == 2 and torch.is_tensor(v[0]) and v[0].dim() == 3
              and isinstance(v[1], list)):                             # fp8 [q [B,s,H], per-stream scales]
            q = v[0].view(torch.uint8)
            out[k] = ["slim", torch.cat([q[b, :l] for b, l in enumerate(lens)], 0), list(lens), v[1]]
        else:
            out[k] = v
    return out


def _unpack_b(resp):
    """verify_batch reply: bare [B][K+1] rows (plain), or {"toks": rows, "aux": {li: [B,s,H]}} under
    M25_EAGLE — fp8-packed aux entries dequantized once, mirroring _unpack. Batched entries carry a
    PER-STREAM scale list ([q, [s0..sB-1]]); solo-shaped entries keep the [q, float] pair. SLIM
    entries (["slim", cat, lens, scales|None] — the tail sliced aux to accepted prefixes) reconstruct
    to a zero-PADDED [B, max_len, H]: the coordinator only ever reads [b, :len(committed)) with
    len(committed) <= lens[b], so padding is never consumed. Dequant matches the unslimmed path
    bit-for-bit (bf16 q times bf16 per-stream scale)."""
    if isinstance(resp, dict):
        aux = resp.get("aux")
        if aux:
            for k, v in aux.items():
                if isinstance(v, list):
                    if v and v[0] == "slim":            # ragged accepted-prefix aux -> padded [B,max,H]
                        _, cat, lens, scales = v
                        mx = max(lens) if lens else 0
                        if scales is not None:
                            cat = cat.view(torch.float8_e4m3fn).to(torch.bfloat16)
                            sc = torch.tensor(scales, dtype=torch.bfloat16)
                        o = torch.zeros(len(lens), mx, cat.shape[-1], dtype=torch.bfloat16)
                        for b, p in enumerate(torch.split(cat, lens, 0)):
                            o[b, :lens[b]] = p * sc[b] if scales is not None else p
                        aux[k] = o
                    elif isinstance(v[1], list):        # per-row scales for [B,s,H]
                        aux[k] = v[0].to(torch.bfloat16) * torch.tensor(v[1], dtype=torch.bfloat16).view(-1, 1, 1)
                    else:
                        aux[k] = _unpack_h(v[0], v[1])
        return resp.get("toks"), aux
    return resp, None


def _eagle_aux_nodes(aux, node_indices):
    """Gather the 3 aux hidden states at arbitrary FLAT verify-node indices -> [len,3,H]. The tree walk
    indexes the verify's nodes (anchor + accepted path), not contiguous chunk positions, so EagleDrafter.extend()
    needs the predicting-aux gathered by node index, not by range."""
    import torch as _t
    return _t.stack([_eagle_seed(aux, i) for i in node_indices], 0)


def _build_tree_msg(trunk, tree, vbase):
    """Tree-verify wire payload: a causal TRUNK (the last committed path, re-fed at absolute positions vbase+i)
    followed by the M draft-tree nodes off its last slot (the anchor). Returns (token_ids, parents, pos_ids) for
    {op:verify,tree:True,start:vbase}: the ring writes KV cropped-to-start=vbase, and build_tree_mask makes every
    node attend the committed prefix [0:vbase] + the trunk + its root->node ancestors (siblings never see each
    other). parents index the flat node set (-1 = attend committed prefix only); siblings share a depth-RoPE pos."""
    L = len(trunk); M = len(tree["tokens"]); token_ids = list(trunk) + list(tree["tokens"])
    parents = [i - 1 for i in range(L)]; pos_ids = [vbase + i for i in range(L)]      # trunk = causal chain
    for j in range(M):
        pj = tree["parents"][j]
        parents.append(L - 1 if pj == -1 else L + pj)                                 # anchor(-1)->trunk's last node
        pos_ids.append(vbase + (L - 1) + tree["depths"][j])                           # depth-based RoPE pos
    return token_ids, parents, pos_ids


_EAGLE = None


def _eagle_singleton():
    """Build the EagleDrafter ONCE (load the EAGLE-3 head + M2.5 embed_tokens onto the coordinator GPU)
    and reuse it across jobs — its only per-job state (_aux/_pending) is overwritten by set_hidden()/
    request() each prefill, so back-to-back jobs stay clean. The embed (200064x3072 bf16 ~1.2GB) + the
    0.2B head fit alongside whatever else shares the coordinator GPU. M25_EAGLE_DIR = the head checkpoint."""
    global _EAGLE
    if _EAGLE is None:
        from eagle_draft import EagleDrafter
        eagle_dir = os.environ.get("M25_EAGLE_DIR", "/root/m25-eagle")
        embed = S.raw("model.embed_tokens.weight").to(torch.bfloat16).to(dev)
        # next_hidden = which hidden the autoregressive draft chain carries forward. "prenorm" = the residual
        # stream (correct: the final norm is a readout-only transform for the lm_head); "final" = the
        # post-final-norm vector (collapses the chain to token-repetition — observed on-engine). Tunable A/B.
        nh = os.environ.get("M25_EAGLE_NEXT_HIDDEN", "prenorm")
        _EAGLE = EagleDrafter(eagle_dir, embed, device=dev, next_hidden=nh)
        print(f"[eagle] head loaded from {eagle_dir} + M2.5 embed on {dev}", flush=True)
    return _EAGLE


def make_drafter(ngram_n=3):
    """Coordinator-side drafter factory — ONE place so coord/_validate/sweep, the gateway, and the honest
    benchmark all build the same thing from the same env. Plain NgramDrafter by default; when M25_EAGLE=1,
    a HybridDrafter (n-gram-first on draftable text -> EAGLE-3 on novel/reasoning misses). The EagleDrafter
    is the shared singleton; the n-gram half is fresh per job (clean index)."""
    from ngram_draft import NgramDrafter
    ng = NgramDrafter(ng=ngram_n, min_match=int(os.environ.get("M25_NGRAM_MINMATCH", "1")))
    if not S.M25_EAGLE:
        return ng
    from eagle_draft import HybridDrafter
    return HybridDrafter(ng, _eagle_singleton())


def make_drafters_b(B, ngram_n=3):
    """Per-stream drafter factory for the batched coordinator — the SAME env logic as make_drafter,
    B times. Each stream gets a fresh n-gram index; under M25_EAGLE each also gets a fork() of the
    shared EAGLE head (shared read-only weights/RoPE, own committed-context state)."""
    from ngram_draft import NgramDrafter
    mm = int(os.environ.get("M25_NGRAM_MINMATCH", "1"))
    if not S.M25_EAGLE:
        return [NgramDrafter(ng=ngram_n, min_match=mm) for _ in range(B)]
    from eagle_draft import HybridDrafter
    base = _eagle_singleton()
    return [HybridDrafter(NgramDrafter(ng=ngram_n, min_match=mm), base.fork()) for _ in range(B)]


def _reset_op(swarm_id, job_id, nonce=None):
    """The job-opening reset frame (sampling pinned greedy). M25_GRAPH_JOB (per-job A/B) optionally
    stamps the runtime graph toggle the stages apply — via _reset_flags — before ack'ing. `nonce` is
    the coordinator's per-job receipt freshness challenge: it rides the reset to every stage so each
    stage signs it into its receipt (a replayed old receipt then carries a stale nonce)."""
    o = {"op": "reset", "temp": 0.0, "top_p": 1.0, "top_k": 0, "seed": 0,
         "swarm_id": swarm_id, "job_id": job_id}
    if nonce is not None:
        o["nonce"] = nonce
    if M25_GRAPH_JOB is not None:
        o["graph"] = M25_GRAPH_JOB
    return o


def _check_reset_ack(op, ack):
    """Job-open ack verification (measurement validity): when the reset carried a 'graph' field the
    tail acks {"ok":1, "graph": <applied>, ...counters}, and a mismatch means the toggle was REFUSED
    (M25_STATIC_KV off on a stage) or the ring runs a pre-toggle build — either way the "graph" arm
    would silently run eager and the paid A/B would bank a lie, so raise LOUDLY instead of measuring.
    Only the TAIL's applied value is visible here; head/middle refusals stay invisible (fine when the
    ring shares one launch env) — the bench runbook must grep every stage log for 'GRAPH REFUSED'
    before trusting an arm. Plain resets ack a bare "ok" (old builds mutually compatible); returns the
    ack dict (carrying the tail's graph_captured/graph_skipped snapshot) or None."""
    if "graph" not in op:
        return None
    applied = ack.get("graph") if isinstance(ack, dict) else None
    if applied != op["graph"]:
        raise RuntimeError(f"graph A/B poisoned: reset asked graph={op['graph']} but the tail applied "
                           f"{applied} (M25_STATIC_KV off on the ring, or an old stage build) — grep "
                           f"stage logs for 'GRAPH REFUSED'")
    return ack


def coordinate_pipe(pipe_sock, tok, messages, K, max_new, timeout, depth, ret_sock, local_draft,
                    tools=None, prefill_chunk=4096, max_ctx=0, prefill_depth=8, on_commit=None,
                    swarm_id="swarm", job_id="job", resume_ids=None, resumable=False, reasoning=True):
    """PIPELINED coordinator copied verbatim from specpipe.coordinate_pipe (n-gram local_draft path,
    greedy, direct-return) — keep `depth` verify chunks in flight so throughput approaches the ring's
    per-chunk THROUGHPUT, not its full latency (the GLM 2.9->16.6 lever). Self-contained: only sockets
    + the drafter + tokenizer. eos handled as int-or-list for M2.5."""
    if S.M25_TREE:                                          # EAGLE tree-verify (M25_TREE): one draft TREE per traversal
        return coordinate_pipe_tree(pipe_sock, tok, messages, K, max_new, timeout, depth, ret_sock, local_draft,
                                    tools=tools, prefill_chunk=prefill_chunk, max_ctx=max_ctx, prefill_depth=prefill_depth,
                                    on_commit=on_commit, swarm_id=swarm_id, job_id=job_id, resume_ids=resume_ids,
                                    resumable=resumable, reasoning=reasoning)
    pipe_sock.settimeout(timeout)
    rx = ret_sock if ret_sock is not None else pipe_sock
    rx.settimeout(timeout)                       # full budget for reset-ack + prefill (a reused gateway socket may carry a prior job's decode heartbeat)
    # cwnd keep-warm on the coord->head leg, wrapped HERE (not at the call boundary) so every caller
    # (coord CLI, gateway, sweep) gets one lock per socket with zero call-site changes; the socket is
    # only ever driven by one job at a time, so a per-job wrapper owns all sends for its lifetime.
    kw_job = os.environ.get("M25_KEEPWARM_JOB")
    kw = _KeepWarm(pipe_sock, interval_ms=kw_job if kw_job not in (None, "") else None)
    def d_request(ids, k): local_draft.request(ids, k)
    def d_fetch(): return local_draft.fetch()
    # discard a stale pending request WITHOUT computing it — fetch() runs the whole proposal (on the
    # EAGLE path a K-step serial chain) just to throw it away on every divergence + at drain.
    d_cancel = getattr(local_draft, "cancel", d_fetch)
    _eos = tok.eos_token_id
    eos_set = set(_eos) if isinstance(_eos, (list, tuple)) else {_eos}
    prompt_ids = render_ids(tok, messages, tools=tools, reasoning=reasoning)   # chat-template + tools; reasoning=False closes <think> -> direct answer (fast)
    resume_ids = list(resume_ids or [])                     # FT resume: re-prefill prompt+committed onto a healed ring, continue (not restart)
    gen_ids = list(prompt_ids) + resume_ids
    if max_ctx:
        max_new = max(len(resume_ids) + 16, min(max_new, max_ctx - len(gen_ids) - 16))
    out = []; t_draft = t_recv = 0.0; prefill_s = 0.0; receipts = []; graph_arm = None
    t_trav = t_stage = t_stage_comp = 0.0; per_stage = {}   # traversal/transport split (see _timing_fields)
    try:
        job_nonce = os.urandom(16).hex() if RECEIPTS else None   # per-job receipt freshness challenge (anti-replay)
        rop = _reset_op(swarm_id, job_id, nonce=job_nonce)  # graph toggle field (if M25_GRAPH_JOB set)
        if kw_job not in (None, ""):                        # + keepwarm toggle (interleaved A/B), one reset
            rop["keepwarm_ms"] = int(kw_job)
        kw.send(rop); _check_reset_ack(rop, recv_data(rx))  # kw lock; recv_data skips noops; raises on refused graph
        t_pf = time.time()
        eagle_on = S.M25_EAGLE and hasattr(local_draft, "extend")
        if eagle_on:
            from eagle_draft import prefill_pair_tokens
            local_draft.reset()                                       # fresh EAGLE context per job (drafter is a shared singleton)
        def _pf_extend(start_i, toks_i, aux_i):
            """Feed ONE prefill chunk's aux into the EAGLE context as it arrives, so the drafter sees the
            WHOLE prompt (the old code kept only the LAST chunk -> the drafter attended ~prefill_chunk
            tokens of context on long prompts) and the extend compute hides in the next chunk's WAN wait."""
            if eagle_on and aux_i is not None:
                local_draft.extend(prefill_pair_tokens(gen_ids, start_i, toks_i),
                                   _eagle_aux_range(aux_i, 0, len(toks_i)), base_pos=start_i)
        if prefill_chunk and len(gen_ids) > prefill_chunk:
            starts = list(range(0, len(gen_ids), prefill_chunk))
            def _send_pf(i): kw.send({"op": "verify", "token_ids": gen_ids[i:i + prefill_chunk], "start": i, "prefill": True})
            d = min(max(prefill_depth, 1), len(starts)); sent = 0; toks = None
            while sent < d: _send_pf(starts[sent]); sent += 1
            for j in range(len(starts)):
                toks, aux = _unpack(_reply_ok(recv_data(rx)))
                if sent < len(starts): _send_pf(starts[sent]); sent += 1
                _pf_extend(starts[j], toks, aux)              # after the refill send: extend overlaps the ring
            cur = toks[-1]
        else:
            kw.send({"op": "verify", "token_ids": gen_ids, "start": 0}); toks, aux = _unpack(_reply_ok(recv_data(rx))); cur = toks[-1]
            _pf_extend(0, toks, aux)
        prefill_s = time.time() - t_pf
        rx.settimeout(_reply_timeout(timeout))              # F6: tighten the per-reply deadline for decode — a mid-decode ring blip fails over in seconds, not up to the full timeout
        pos = len(gen_ids); out = resume_ids + [cur]        # preserve recovered tokens; cur = next after them
        if on_commit: on_commit(out, 0.0)               # stream: first token from prefill
        inflight = []; discard = 0; send_pos = pos; dprefix = gen_ids + [cur]
        valid = accepted = wasted = 0; t0 = time.time(); done = False
        conf = (ConfidenceScheduler(1, depth, lo=0.3, hi=0.7)               # opt-in DSpark depth throttle (M25_CONF_SCHED)
                if (ConfidenceScheduler and os.environ.get("M25_CONF_SCHED")) else None)  # K fixed (graph-safe); only in-flight depth adapts
        d_request(dprefix, K)
        while not done:
            cur_depth = 1 if S.M25_EAGLE else (conf.value() if conf else depth)   # EAGLE needs the verified hidden -> can't pipeline (v1: depth 1); else full depth
            while len(inflight) < cur_depth and not done:
                td = time.time(); ds = d_fetch(); t_draft += time.time() - td
                t_sent = time.monotonic()                       # traversal origin: includes the outbound serialize
                kw.send({"op": "verify", "token_ids": [dprefix[-1]] + ds, "start": send_pos})
                inflight.append((send_pos, ds, t_sent)); dprefix = dprefix + ds; send_pos += K
                d_request(dprefix, K)
            tr = time.time(); resp = _reply_ok(recv_data(rx)); t_recv += time.time() - tr
            r, aux = _unpack(resp)
            sp, ds, t_sent = inflight.pop(0)
            t_trav += time.monotonic() - t_sent                 # count discarded chunks too — they traversed
            s_, c_ = _acc_stage_dt(resp, per_stage); t_stage += s_; t_stage_comp += c_
            if discard > 0: discard -= 1; wasted += 1; continue
            n = 0
            for j in range(K):
                if ds[j] == r[j]: n += 1
                else: break
            valid += 1; accepted += n
            if os.environ.get("M25_EAGLE_DEBUG") and S.M25_EAGLE and valid <= 3:   # diagnostic: is aux arriving + is EAGLE drafting?
                _mt = getattr(local_draft, "matched", None)
                if aux is None:
                    print(f"[eagle-dbg] r{valid}: aux=None (ring returned plain toks -> EAGLE degrades to repeat) matched={_mt} ds={ds[:5]} r={r[:5]} acc={n}", flush=True)
                else:
                    _ok = all(str(li) in aux for li in S.EAGLE_AUX_LAYER_IDS)
                    _sd = _eagle_seed(aux, n) if _ok else None
                    _nm = float(_sd.float().norm()) if _ok else -1.0
                    _eg = getattr(local_draft, "eagle", local_draft)            # decisive probe: does fc(aux) VARY across prompts?
                    _fcs = "n/a"
                    if _ok and hasattr(_eg, "fc"):
                        _fc = torch.nn.functional.linear(_sd.reshape(-1).to(_eg.fc.dtype).to(_eg.fc.device).unsqueeze(0), _eg.fc)
                        _fcs = f"norm={float(_fc.norm()):.2f} v3={[round(x,3) for x in _fc[0,:3].float().tolist()]}"
                    print(f"[eagle-dbg] r{valid}: seednorm={_nm:.1f} fc(aux):{_fcs} matched={_mt} ds={ds[:5]} r={r[:5]} acc={n}", flush=True)
            if conf: conf.observe(n, K)                                     # acceptance EMA (free, from the verify result)
            if n == K:
                out.extend(ds); pos += K; cur = ds[-1]; committed = ds
                # FULL-ACCEPT BONUS: r[K] is the target's greedy token at the position just past the last
                # accepted draft (lossless — the whole draft matched, so the prefix IS the greedy sequence).
                # When nothing is in flight (depth-1: EAGLE/tree) that position is NOT covered by a queued
                # chunk, so committing it now advances the frontier K+1 not K -> ~1/K fewer WAN round-trips on
                # draftable text, and makes toks_per_traversal honest. Under pipelining (depth>1) the next
                # in-flight chunk already re-derives it at no extra RTT, so we leave it (committing would
                # collide with that chunk's start).
                if not inflight and len(r) > K:
                    out.append(r[K]); committed = ds + [r[K]]; cur = r[K]
                    pos += 1; send_pos += 1; dprefix = dprefix + [r[K]]
            else:
                committed = ds[:n] + [r[n]]; out.extend(committed); cur = r[n]; pos += n + 1
                discard = len(inflight); d_cancel(); dprefix = prompt_ids + out; send_pos = pos; d_request(dprefix, K)
            if eagle_on and aux is not None:                 # grow the EAGLE context with the newly committed positions
                local_draft.extend(committed, _eagle_aux_range(aux, 0, len(committed)), base_pos=sp)   # committed[i] predicted by aux[i] (target hidden one pos earlier)
            if on_commit: on_commit(out, time.time() - t0)   # stream: this commit's running output
            if len(out) >= max_new or (cur in eos_set) or (eos_set & set(committed)): done = True
        d_cancel()
        while inflight: recv_data(rx); inflight.pop(0)
        if RECEIPTS:                                        # PROVE: sweep the ring once for signed per-stage receipts
            kw.send({"op": "receipt", "receipts": []}); receipts = recv_data(rx)   # kw lock + noop-skip
            if isinstance(receipts, dict):              # graph-A/B job: tail promoted the reply with counters
                graph_arm = {k: receipts.get(k) for k in ("graph", "graph_captured", "graph_skipped")}
                receipts = receipts.get("receipts", [])
    except EDGE_ERRORS as e:
        if resumable:                                       # a node died: hand committed tokens back so the control plane heals + resumes (not restart)
            committed = out if out else list(resume_ids)
            return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}", "resumable": True,
                    "output_ids": committed, "n_tokens": len(committed),
                    "text": tok.decode(committed, skip_special_tokens=True)}
        raise TransportError(f"pipeline edge failed at token {len(out)} ({type(e).__name__}: {e})") from e
    finally:
        kw.stop()                                           # job over: never leak a noop thread onto the reused socket
    dt = time.time() - t0
    for ee in eos_set:
        if ee in out: out = out[:out.index(ee)]; break
    # True depth from the model config — never from the receipts themselves (self-referential coverage
    # let a layer-omitting ring pass). Fail CLOSED when receipts were requested but none came back.
    # expected_nonce binds the set to THIS job (anti-replay); check_chain (lossless wire only, fp8 is
    # intentionally lossy so out_root!=in_root) binds them to one real ring pass.
    receipts_ok = (_verify_receipts(receipts, S.cfg.num_hidden_layers, expected_nonce=job_nonce,
                                    check_chain=not M25_FP8_WIRE) if receipts
                   else (False if RECEIPTS else None))
    return {"ok": True, "text": tok.decode(out, skip_special_tokens=True), "n_tokens": len(out), "rounds": valid,
            # HONEST g: committed tokens per verify round = frontier advance (pos) / rounds. NOT the old
            # (accepted+valid)/valid, which counted a bonus token on EVERY full-accept round even when the
            # pipelined path dropped it -> up to +1 inflation, not comparable to the tree arm's exact g.
            "mean_accept": accepted / max(valid, 1), "toks_per_traversal": (pos - len(gen_ids)) / max(valid, 1),
            "tok_s": len(out) / max(dt, 1e-9), "wasted": wasted, "prefill_s": prefill_s, "output_ids": out,
            "prompt_tokens": len(prompt_ids), "resume_tokens": len(resume_ids),
            "receipts": receipts, "receipts_ok": receipts_ok,
            "graph_arm": graph_arm,   # graph-A/B jobs: tail's {graph, graph_captured, graph_skipped} at job end
            # decode-loop breakdown: draft_s = serial drafter compute, ring_wait_s = blocked on the ring's
            # return channel, decode_s = the whole decode wall. What's NOT wait or draft is coordinator-side
            # commit/extend/serialize overhead — the profile that ranks the next serial-path fix.
            "decode_s": round(dt, 3), "draft_s": round(t_draft, 3), "ring_wait_s": round(t_recv, 3),
            **_timing_fields(t_trav, t_stage, t_stage_comp, per_stage),
            "final_confidence": conf.confidence() if conf else None}


def coordinate_pipe_tree(pipe_sock, tok, messages, K, max_new, timeout, depth, ret_sock, local_draft,
                         tools=None, prefill_chunk=4096, max_ctx=0, prefill_depth=8, on_commit=None,
                         swarm_id="swarm", job_id="job", resume_ids=None, resumable=False, reasoning=True):
    """DEPTH-AWARE HYBRID coordinator (M25_TREE=1): per round, route by what the text is doing.
    * n-gram MATCHED (verbatim/draftable) -> a PLAIN pipelined chain frame, up to `depth` in flight —
      the flash kernel + the small payload + pipelining, exactly the regime that wins those cells.
      (The 2026-07-03 good-ring split showed routing matched rounds as 1-wide TREES paid the manual
      off-flash kernel + trunk re-feed + wider aux for zero accept gain: 199-303ms/round vs 139ms,
      ctx-8k-quote 5.1 vs 11.8 tok/s. And both arms ran depth-1, leaving the α≈0.97 streaks unpipelined.)
    * n-gram MISS (novel/reasoning) -> ONE synchronous EAGLE tree round: top-M best-first tree
      (M25_TREE_M nodes, M25_TREE_TOPB children, M25_TREE_DEPTH cap) verified in one forward under an
      ancestor-only mask (run_block_tree); tree_greedy_walk commits the longest path + 1 correction/bonus.
      Tree rounds stay depth-1 structurally (EAGLE needs the verified hidden to draft the next round).

    KV DIRTY-FRONTIER CONTRACT (the cross-mode bookkeeping): a tree round leaves the newly committed
    path's KV rows dirty (they were tree nodes at scattered slots), tracked as `pending_path` @ vbase.
    The NEXT frame must re-feed them: a tree round re-feeds them as its causal trunk (as before); the
    FIRST chain frame of a burst prepends them as a causal prefix (start=vbase, accept offset L-1).
    After any chain commit the only dirty token is `cur` (a correction/bonus is never an input until
    re-sent), so pending_path collapses to [cur] — which makes the burst's first frame IDENTICAL to
    coordinate_pipe's standard [anchor]+draft frame. The fake-ring harness models this frontier and
    asserts no frame ever writes past a KV gap.

    GREEDY / LOSSLESS by construction on BOTH routes (the ring greedy-verifies every proposed token;
    routing only changes WHICH tokens are proposed and how many frames are in flight). Prefill, the
    receipts sweep and the return-dict shape are identical to coordinate_pipe."""
    from tree_spec import tree_greedy_walk
    pipe_sock.settimeout(timeout)
    rx = ret_sock if ret_sock is not None else pipe_sock
    rx.settimeout(timeout)                                  # full budget for reset-ack + prefill (see coordinate_pipe); decode tightens below
    kw_job = os.environ.get("M25_KEEPWARM_JOB")             # cwnd keep-warm, same design as coordinate_pipe
    kw = _KeepWarm(pipe_sock, interval_ms=kw_job if kw_job not in (None, "") else None)
    _eos = tok.eos_token_id
    eos_set = set(_eos) if isinstance(_eos, (list, tuple)) else {_eos}
    prompt_ids = render_ids(tok, messages, tools=tools, reasoning=reasoning)
    resume_ids = list(resume_ids or [])
    gen_ids = list(prompt_ids) + resume_ids
    if max_ctx:
        max_new = max(len(resume_ids) + 16, min(max_new, max_ctx - len(gen_ids) - 16))
    out = []; prefill_s = 0.0; t_draft = t_recv = 0.0; receipts = []; graph_arm = None
    t_trav = t_stage = t_stage_comp = 0.0; per_stage = {}   # traversal/transport split (see _timing_fields)
    tree_m = int(os.environ.get("M25_TREE_M", "12"))
    tree_topb = int(os.environ.get("M25_TREE_TOPB", "3"))
    tree_depth = int(os.environ.get("M25_TREE_DEPTH", "8"))
    eg = getattr(local_draft, "eagle", local_draft)         # the EagleDrafter (HybridDrafter.eagle, or itself)
    if not hasattr(eg, "propose_tree"):
        raise RuntimeError("M25_TREE=1 needs the EAGLE drafter (set M25_EAGLE=1 + M25_EAGLE_DIR on the coordinator)")
    try:
        job_nonce = os.urandom(16).hex() if RECEIPTS else None   # per-job receipt freshness challenge (anti-replay)
        rop = _reset_op(swarm_id, job_id, nonce=job_nonce)  # graph toggle field (if M25_GRAPH_JOB set)
        if kw_job not in (None, ""):                        # + keepwarm toggle (interleaved A/B), one reset
            rop["keepwarm_ms"] = int(kw_job)
        kw.send(rop); _check_reset_ack(rop, recv_data(rx))  # kw lock; recv_data skips noops; raises on refused graph
        t_pf = time.time()                                  # ---- prefill: IDENTICAL to coordinate_pipe ----
        eg.reset()                                          # fresh EAGLE context per job (drafter is a shared singleton)
        from eagle_draft import prefill_pair_tokens
        def _pf_extend(start_i, toks_i, aux_i):
            """Feed ONE prefill chunk's aux into the EAGLE context as it arrives (whole-prompt drafter
            context — the accept lever the serial-path A/B proved on rag-quote, 13->44%)."""
            if aux_i is not None:
                eg.extend(prefill_pair_tokens(gen_ids, start_i, toks_i),
                          _eagle_aux_range(aux_i, 0, len(toks_i)), base_pos=start_i)
        if prefill_chunk and len(gen_ids) > prefill_chunk:
            starts = list(range(0, len(gen_ids), prefill_chunk))
            def _send_pf(i): kw.send({"op": "verify", "token_ids": gen_ids[i:i + prefill_chunk], "start": i, "prefill": True})
            d = min(max(prefill_depth, 1), len(starts)); sent = 0; toks = None
            while sent < d: _send_pf(starts[sent]); sent += 1
            for j in range(len(starts)):
                toks, aux = _unpack(_reply_ok(recv_data(rx)))
                if sent < len(starts): _send_pf(starts[sent]); sent += 1
                _pf_extend(starts[j], toks, aux)            # after the refill send: extend overlaps the ring
            cur = toks[-1]
        else:
            kw.send({"op": "verify", "token_ids": gen_ids, "start": 0}); toks, aux = _unpack(_reply_ok(recv_data(rx))); cur = toks[-1]
            _pf_extend(0, toks, aux)
        if aux is None:                                     # fail loud, not a mid-job TypeError: stages must run M25_EAGLE
            raise TransportError("tree-verify got no aux from the ring — launch stages with M25_TREE=1/M25_EAGLE=1")
        prefill_s = time.time() - t_pf
        rx.settimeout(_reply_timeout(timeout))                      # F6: per-reply decode deadline (see coordinate_pipe) — fast blip failover
        out = [cur]; pending_path = [cur]; vbase = len(gen_ids)      # cur = first gen token at abs pos vbase
        if on_commit: on_commit(out, 0.0)                            # stream: first token from prefill
        rounds = 0; total_committed = 0; accepted = 0; wasted = 0; t0 = time.time(); done = False
        ng = getattr(local_draft, "ngram", None)            # HybridDrafter's n-gram half (None on a bare EagleDrafter)
        inflight = []                                       # chain-burst frames: (off, start, ds, t_sent)
        discard = 0                                         # stale in-flight replies after a mid-burst divergence
        dprefix = None                                      # committed + speculative tokens (slot i == dprefix[i]); None = rebuild
        send_pos = 0                                        # slot of the NEXT plain frame's anchor (dprefix[-1])
        refeed = True                                       # pending_path KV is dirty (post-prefill / post-tree round)
        while not done:
            # ---- FILL: keep plain chain frames in flight while the n-gram matches -------------------
            while len(inflight) < depth and not done:
                if dprefix is None:
                    dprefix = list(gen_ids) + out
                d = None
                if ng is not None:
                    td = time.time()
                    ng.request(dprefix, K)
                    d0 = ng.fetch()
                    t_draft += time.time() - td
                    if d0 and getattr(ng, "matched", False):
                        d = list(d0)
                if d is None:
                    break                                   # novel text -> tree round (or drain what's in flight)
                if refeed:                                  # first frame after prefill/tree: re-feed the dirty path
                    ids = list(pending_path) + d; start = vbase; off = len(pending_path) - 1
                    refeed = False
                else:
                    ids = [dprefix[-1]] + d; start = send_pos; off = 0
                t_sent = time.monotonic()                   # traversal origin: includes the outbound serialize
                kw.send({"op": "verify", "token_ids": ids, "start": start})
                inflight.append((off, start, d, t_sent))
                dprefix = dprefix + d; send_pos = start + off + K   # next anchor = dprefix[-1]'s slot
            if not inflight:
                # ---- NOVEL: one synchronous EAGLE tree round ---------------------------------------
                L = len(pending_path)
                td = time.time()
                tree = eg.propose_tree(tree_m, topb=tree_topb, max_depth=tree_depth)
                t_draft += time.time() - td
                token_ids, parents, pos_ids = _build_tree_msg(pending_path, tree, vbase)
                t_sent = time.monotonic()
                kw.send({"op": "verify", "tree": True, "token_ids": token_ids,
                         "parents": parents, "pos_ids": pos_ids, "start": vbase})
                tr = time.time(); resp = _reply_ok(recv_data(rx)); t_recv += time.time() - tr
                t_trav += time.monotonic() - t_sent         # depth-1: this IS the clean per-round T_traversal
                s_, c_ = _acc_stage_dt(resp, per_stage); t_stage += s_; t_stage_comp += c_
                r, aux = _unpack(resp)
                path_idx, committed = tree_greedy_walk(tree["tokens"], tree["parents"], r[L:], r[L - 1])
                out.extend(committed); vbase += L; pending_path = committed; cur = committed[-1]
                # EAGLE extend: committed[0] predicted by the anchor (flat node L-1); committed[k>0] by the (k-1)-th
                # accepted path node (flat node L+path_idx[k-1]). Slice to len(committed) (== 1+len(path_idx)).
                pred_idx = ([L - 1] + [L + pi for pi in path_idx])[:len(committed)]
                eg.extend(committed, _eagle_aux_nodes(aux, pred_idx), base_pos=vbase - 1)   # base_pos = anchor's abs pos
                rounds += 1; total_committed += len(committed)
                accepted += len(committed) - 1              # DRAFT tokens accepted (the +1 is correction/bonus)
                refeed = True; dprefix = None               # the committed path's KV rows are tree nodes -> dirty
                if on_commit: on_commit(out, time.time() - t0)
                if len(out) >= max_new or (cur in eos_set) or (eos_set & set(committed)): done = True
                continue
            # ---- BURST REPLY: same accept/divergence bookkeeping as coordinate_pipe ----------------
            tr = time.time(); resp = _reply_ok(recv_data(rx)); t_recv += time.time() - tr
            r, aux = _unpack(resp)
            off, start, ds, t_sent = inflight.pop(0)
            t_trav += time.monotonic() - t_sent             # count discarded chunks too — they traversed
            s_, c_ = _acc_stage_dt(resp, per_stage); t_stage += s_; t_stage_comp += c_
            if discard > 0:
                discard -= 1; wasted += 1; continue
            n = 0
            for j in range(K):
                if ds[j] == r[off + j]: n += 1
                else: break
            rounds += 1; accepted += n                      # chain semantics: draft tokens accepted, exactly
                                                            # (a no-bonus full accept is n=K accepted, not K-1)
            if n == K:
                committed = list(ds)
                if not inflight and len(r) > off + K:       # full-accept bonus, same rule as coordinate_pipe:
                    committed.append(r[off + K])            # only when nothing queued re-derives that position
                    dprefix = dprefix + [r[off + K]]; send_pos += 1
            else:
                committed = ds[:n] + [r[off + n]]
                discard = len(inflight)                     # everything in flight speculated past the divergence
            out.extend(committed); cur = committed[-1]
            total_committed += len(committed)
            # the only dirty token after a chain commit is cur (a correction/bonus was never an input);
            # the next plain frame re-sends it as its anchor, so bursts continue with standard framing.
            vbase = start + off + len(committed); pending_path = [cur]
            if n < K:
                dprefix = list(gen_ids) + out; send_pos = vbase   # next anchor = cur @ its own slot
            if aux is not None:                             # burst frames carry aux rows per frame position:
                eg.extend(committed, _eagle_aux_range(aux, off, off + len(committed)), base_pos=start + off)
            if on_commit: on_commit(out, time.time() - t0)
            if len(out) >= max_new or (cur in eos_set) or (eos_set & set(committed)): done = True
        while inflight:                                     # drain replies for frames sent past the finish
            recv_data(rx); inflight.pop(0)
        if RECEIPTS:                                        # PROVE: sweep the ring once for signed per-stage receipts
            kw.send({"op": "receipt", "receipts": []}); receipts = recv_data(rx)   # kw lock + noop-skip
            if isinstance(receipts, dict):              # graph-A/B job: tail promoted the reply with counters
                graph_arm = {k: receipts.get(k) for k in ("graph", "graph_captured", "graph_skipped")}
                receipts = receipts.get("receipts", [])
    except EDGE_ERRORS as e:
        if resumable:                                       # a node died: hand committed tokens back so the control plane heals + resumes
            committed = out if out else list(resume_ids)
            return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}", "resumable": True,
                    "output_ids": committed, "n_tokens": len(committed),
                    "text": tok.decode(committed, skip_special_tokens=True)}
        raise TransportError(f"tree pipeline edge failed at token {len(out)} ({type(e).__name__}: {e})") from e
    finally:
        kw.stop()                                           # job over: never leak a noop thread onto the reused socket
    dt = time.time() - t0
    for ee in eos_set:
        if ee in out: out = out[:out.index(ee)]; break
    receipts_ok = (_verify_receipts(receipts, S.cfg.num_hidden_layers, expected_nonce=job_nonce,
                                    check_chain=not M25_FP8_WIRE) if receipts
                   else (False if RECEIPTS else None))
    # mean_accept counts DRAFT tokens accepted per round, same as coordinate_pipe — NOT committed-1:
    # a pipelined full-accept round commits K with no bonus, and deriving accept from committed under-
    # reported it by 1 (a 12.5pp phantom accept deficit vs the chain arm at K=8; found adversarially).
    return {"ok": True, "text": tok.decode(out, skip_special_tokens=True), "n_tokens": len(out), "rounds": rounds,
            "mean_accept": accepted / max(rounds, 1), "toks_per_traversal": total_committed / max(rounds, 1),
            "tok_s": len(out) / max(dt, 1e-9), "wasted": wasted, "prefill_s": prefill_s, "output_ids": out,
            "prompt_tokens": len(prompt_ids), "resume_tokens": len(resume_ids),
            "receipts": receipts, "receipts_ok": receipts_ok,
            "graph_arm": graph_arm,   # graph-A/B jobs: tail's {graph, graph_captured, graph_skipped} at job end
            "decode_s": round(dt, 3), "draft_s": round(t_draft, 3), "ring_wait_s": round(t_recv, 3),
            **_timing_fields(t_trav, t_stage, t_stage_comp, per_stage),
            "final_confidence": None}


def _sdpa_backend_probe(stage):
    """Fail loud at warm-up (not mid-prefill OOM) if no FUSED SDPA backend serves the prefill shape on this
    GPU. A fused backend (flash/cudnn/efficient) does online softmax = O(s) memory; the MATH fallback
    materializes the [1,NH,s,total] score matrix = the very OOM the SDPA fix removes. Reports which engage."""
    avail = []
    qd = torch.randn(1, S.NH, 64, S.HD, dtype=torch.bfloat16, device=dev)
    kd = torch.randn(1, S.NKV, 256, S.HD, dtype=torch.bfloat16, device=dev)
    mask = S.causal_lower_right(64, 256)
    for name, be in [("flash", S.SDPBackend.FLASH_ATTENTION), ("cudnn", S.SDPBackend.CUDNN_ATTENTION),
                     ("efficient", S.SDPBackend.EFFICIENT_ATTENTION)]:
        try:
            with S.sdpa_kernel([be]):
                torch.nn.functional.scaled_dot_product_attention(qd, kd, kd, attn_mask=mask,
                                                                 scale=S.SCALING, enable_gqa=True)
            avail.append(name)
        except Exception:
            pass
    if avail:
        print(f"[s{stage}] SDPA fused backends available on sm_120: {avail}", flush=True)
    else:
        print(f"[s{stage}] WARN SDPA: NO fused backend serves the prefill shape — falls back to MATH "
              f"(materializes scores; long-ctx will OOM). Lower prefill_chunk or set M25_SDPA=0.", flush=True)


def coordinate_pipe_batch(pipe_sock, tok, messages_list, K, max_new, timeout, ret_sock, drafters,
                          depth=4, tools=None, prefill_chunk=4096, max_ctx=0, reasoning=True,
                          on_commits=None, tools_b=None, swarm_id="swarm", job_id="job"):
    if M25_DELOCKSTEP and S.M25_EAGLE and all(hasattr(d, "extend") for d in drafters):
        return coordinate_pipe_rows(pipe_sock, tok, messages_list, K, max_new, timeout, ret_sock,
                                    drafters, tools=tools, prefill_chunk=prefill_chunk,
                                    max_ctx=max_ctx, reasoning=reasoning, on_commits=on_commits,
                                    tools_b=tools_b, swarm_id=swarm_id, job_id=job_id)
    """CONTINUOUS-BATCHING coordinator: B independent spec-decode streams share ONE ring traversal per
    round, so the WAN round-trip is amortized across all B (aggregate-throughput lever). Each stream's
    output is byte-identical to a solo coordinate_pipe run (per-stream KV row + per-stream causal mask +
    per-stream MoE on the stage side guarantee it). Prefill is PER-STREAM (variable length) into
    batch-row b; only the fixed-shape K+1 decode is batched. Greedy.

    DRAFTING = the FULL solo stack per stream: `drafters[b]` may be n-gram, EAGLE, or Hybrid — when the
    drafter has extend() and stages run M25_EAGLE, verify_batch returns per-stream aux rows ([B,s,H] per
    aux layer) and each stream's committed positions grow ITS drafter's context, exactly like solo. The
    depth rule mirrors solo too: EAGLE needs the verified hidden before the next draft, so the in-flight
    window is 1 under M25_EAGLE (batching itself is the amortizer); n-gram-only keeps depth pipelining.

    Protocol: reset_batch -> prefill each stream (op=verify, stream=b) -> per round, op=verify_batch
    with token_ids_b/start_b for the ACTIVE streams; the ring returns B argmax rows (+ aux under EAGLE).
    RECEIPTS (SHARD_RECEIPTS=1) sweep at job end like solo — batched rounds are attested on-stage."""
    B = len(messages_list)
    rx = ret_sock if ret_sock is not None else pipe_sock
    pipe_sock.settimeout(timeout)
    rx.settimeout(timeout)                               # NEVER inherit a prior solo job's tightened 20s decode
                                                         # deadline (gateway reuses the ret socket across jobs;
                                                         # a 4096-token batched prefill hop can exceed 20s)
    kw_job = os.environ.get("M25_KEEPWARM_JOB")          # cwnd keep-warm, same design as coordinate_pipe;
    kw = _KeepWarm(pipe_sock, interval_ms=kw_job if kw_job not in (None, "") else None)   # NOTE: reset_batch
    _eos = tok.eos_token_id                              # carries no keepwarm_ms — stage toggling is reset-only
    eos_set = set(_eos) if isinstance(_eos, (list, tuple)) else {_eos}
    # reasoning/max_new accept a scalar (every stream alike — the bench shape) or a per-stream list;
    # per-stream tool specs go in `tools_b` (a toolspec is itself a list, so `tools` stays scalar-only).
    # on_commits: per-stream streaming callbacks, called with (out[b], dt) after each commit — solo's
    # on_commit, per row (the gateway shape: independent requests riding one batch).
    tb = tools_b if tools_b is not None else [tools] * B
    reas_b = reasoning if isinstance(reasoning, list) else [reasoning] * B
    mxnew_b = max_new if isinstance(max_new, list) else [max_new] * B
    prompts = [render_ids(tok, m, tools=tb[b], reasoning=reas_b[b]) for b, m in enumerate(messages_list)]
    mx = [max(16, min(mxnew_b[b], max_ctx - len(p) - 16)) if max_ctx else mxnew_b[b]
          for b, p in enumerate(prompts)]
    out = [[] for _ in range(B)]; pos = [0] * B; cur = [0] * B; done = [False] * B
    acc = [0] * B; vrounds = [0] * B                     # per-stream accepted tokens / verify rounds (g telemetry)
    t_recv = 0.0; t_pf = time.time(); receipts = []; graph_arm = None; per_stage = {}
    eagle_on = S.M25_EAGLE and all(hasattr(d, "extend") for d in drafters)
    if eagle_on:
        from eagle_draft import prefill_pair_tokens, fetch_b
        for d in drafters:
            d.reset()                                    # fresh per-stream EAGLE context per job
    job_nonce = os.urandom(16).hex() if RECEIPTS else None   # per-job receipt freshness challenge (anti-replay)
    # aux_local token = a per-job NONCE, not job_id: job_id defaults to "job" for every sweep/gateway
    # job, so it cannot disambiguate a dead job's lane residue from this job's (review F2). Counters
    # live OUTSIDE the try so the finally-drain (review F1) is always well-defined.
    aux_token = os.urandom(8).hex() if (M25_AUX_LOCAL and eagle_on) else None
    aux_local = False; lseq_tx = lseq_rx = 0
    try:
        rb_op = {"op": "reset_batch", "B": B, "swarm_id": swarm_id, "job_id": job_id}
        if job_nonce is not None:
            rb_op["nonce"] = job_nonce
        if M25_GRAPH_JOB is not None:                        # per-job graph arm, exactly solo's stamp: a batched
            rb_op["graph"] = M25_GRAPH_JOB                   # job must never SILENTLY inherit a prior solo job's
                                                             # runtime route (the review's poisoned-arm scenario)
        if aux_token:
            rb_op["aux_local"] = aux_token                   # per-job opt-in; the head echoes the token on the lane
        kw.send(rb_op); ack = recv_data(rx)
        if isinstance(ack, dict) and ack.get("error"):       # e.g. B wider than the ring's launch-time KV rows:
            raise TransportError(f"reset_batch refused: {ack['error']}")   # abort BEFORE any batched
                                                                           # op can kill a warm stage
        _check_reset_ack(rb_op, ack)                         # graph-stamped job + refused/old-stage toggle = fail LOUD
        if aux_token:
            aux_local = _aux_local_handshake(pipe_sock, aux_token)   # also drains a dead job's stale frames
            if not aux_local:
                print("[batch] aux_local asked but the head never acked — ridden-ring aux (old head?)", flush=True)
        for b in range(B):                                   # PER-STREAM prefill into row b (variable length)
            gen = prompts[b]
            starts_b = range(0, len(gen), prefill_chunk) if prefill_chunk else [0]
            for i in starts_b:
                chunk = gen[i:i + prefill_chunk] if prefill_chunk else gen
                pf = {"op": "verify", "stream": b, "token_ids": chunk, "start": i, "prefill": True}
                if aux_local:
                    pf["seq"] = lseq_tx; lseq_tx += 1
                kw.send(pf)
                rr, aux = _unpack(_reply_ok(recv_data(rx)))   # stream-b prefill reply is solo-shaped ({toks, aux})
                if aux_local:                            # the head's own aux layers arrive on the local lane
                    loc = _pull_aux_local(pipe_sock, aux_token, lseq_rx); lseq_rx += 1
                    if loc:
                        aux = {**(aux or {}), **loc}     # all-empty stays None -> the no-aux path fails loud
                if eagle_on and aux is not None:         # whole-prompt drafter context, chunk by chunk
                    drafters[b].extend(prefill_pair_tokens(gen, i, rr),
                                       _eagle_aux_range(aux, 0, len(rr)), base_pos=i)
            cur[b] = rr[-1]
            pos[b] = len(gen); out[b] = [cur[b]]
            if cur[b] in eos_set or len(out[b]) >= mx[b]: done[b] = True
            drafters[b].request(prompts[b] + [cur[b]], K)
            if on_commits and on_commits[b]: on_commits[b](out[b], 0.0)   # stream: first token from prefill
        prefill_s = time.time() - t_pf; t0 = time.time()        # start the DECODE-rate timer after prefill (matches coordinate_pipe; agg_tok_s is steady-state decode, not TTFT-polluted)
        # PIPELINED: keep `depth` batched verify-rounds in flight so the WAN is HIDDEN (the synchronous depth=1 path
        # paid full ring latency L every round -> B/L; this restores depth-pipelining -> aggregate ~ B x single-stream).
        # Each round speculatively advances ALL B streams; on a stream's divergence we drop that stream's stale
        # in-flight chunks (per-row discard) and re-draft. Each stream stays data-isolated (output depends on B, not
        # on batch-mates) and byte-faithful to solo up to the batched-matmul tiling. Mirrors coordinate_pipe per row.
        cur_depth = 1 if eagle_on else depth                    # solo's rule: EAGLE drafts from the verified hidden
        rounds = 0; wasted = 0
        dprefix = [prompts[b] + [cur[b]] for b in range(B)]     # speculative continuation per stream (prefill already requested)
        spos = list(pos)                                        # send position per stream (advances K per drafted chunk)
        discard = [0] * B                                       # stale-chunk skip counter per stream after a divergence
        inflight = []                                           # FIFO of rounds; each = [(spos_b, ds_b) | None] over b
        while not all(done) or inflight:
            while len(inflight) < cur_depth and not all(done):  # fill the in-flight window (speculative per stream)
                tids = []; row = []; sb = []
                # ONE batched drafter pass per fill (eagle_draft.fetch_b): n-gram per stream (free), all
                # EAGLE misses as a single [n,...] chain forward. The per-b serial fetch() loop was the
                # measured drafting tax (~0.25s/stream/round at B=4 — the round went DRAFTING-bound);
                # each row stays byte-identical to drafters[b].fetch() (research/m25_draft_batch_test.py).
                if eagle_on:
                    act = [b for b in range(B) if not done[b]]
                    ds_act = dict(zip(act, fetch_b([drafters[b] for b in act])))
                for b in range(B):
                    if done[b]:
                        tids.append([cur[b]] * (K + 1)); row.append(None); sb.append(pos[b]); continue
                    ds = ds_act[b] if eagle_on else drafters[b].fetch()
                    tids.append([dprefix[b][-1]] + ds); row.append((spos[b], ds)); sb.append(spos[b])
                    dprefix[b] = dprefix[b] + ds; spos[b] += K; drafters[b].request(dprefix[b], K)
                vb = {"op": "verify_batch", "token_ids_b": tids, "start_b": sb}
                if aux_local:
                    vb["seq"] = lseq_tx; lseq_tx += 1
                kw.send(vb)
                inflight.append(row)
            if not inflight:
                break
            tr = time.time(); resp = _reply_ok(recv_data(rx)); t_recv += time.time() - tr
            if isinstance(resp, dict) and resp.get("stage_dt"):   # per-stage [span, compute] stamps (M25_STAGE_TIMING)
                _acc_stage_dt(resp, per_stage)
            rb, aux_b = _unpack_b(resp)                         # rb: [B][K+1] per-stream argmax; aux rows under EAGLE
            if aux_local:                                       # one local frame per round, ALWAYS consumed (even for
                loc = _pull_aux_local(pipe_sock, aux_token, lseq_rx); lseq_rx += 1   # rounds a divergence will discard —
                if loc:                                                           # the FIFO pairing must never skew)
                    aux_b = {**(aux_b or {}), **loc}
            if eagle_on and aux_b is None and rounds == 0:      # fail LOUD, not a silently-poisoned measurement:
                raise TransportError("EAGLE drafters but the ring returned no aux — launch stages "
                                     "with M25_EAGLE=1 (a depth-1 no-context run measures WORSE than n-gram)")
            row = inflight.pop(0); rounds += 1
            for b in range(B):
                if row[b] is None or done[b]:
                    continue
                if discard[b] > 0:                              # stale chunk from before this stream's last divergence
                    discard[b] -= 1; wasted += 1; continue
                sp_b, ds = row[b]; r = rb[b]; n = 0
                for j in range(K):
                    if ds[j] == r[j]: n += 1
                    else: break
                vrounds[b] += 1; acc[b] += n
                if n == K:
                    out[b].extend(ds); pos[b] += K; cur[b] = ds[-1]; committed = ds
                    # FULL-ACCEPT BONUS, exactly solo's rule: only when nothing is in flight (depth 1 —
                    # the EAGLE regime) is r[K]'s position uncovered; commit it to advance K+1 per round.
                    if not inflight and len(r) > K:
                        out[b].append(r[K]); committed = ds + [r[K]]; cur[b] = r[K]
                        pos[b] += 1; spos[b] += 1; dprefix[b] = dprefix[b] + [r[K]]
                        acc[b] += 1
                else:                                           # divergence: commit prefix, drop this stream's stale in-flight, re-draft
                    committed = ds[:n] + [r[n]]; out[b].extend(committed); cur[b] = r[n]; pos[b] += n + 1
                    discard[b] = sum(1 for rr in inflight if rr[b] is not None)
                    getattr(drafters[b], "cancel", drafters[b].fetch)()   # drop the stale pending draft WITHOUT computing it
                    dprefix[b] = prompts[b] + out[b]; spos[b] = pos[b]; drafters[b].request(dprefix[b], K)
                if eagle_on and aux_b is not None:              # grow stream b's EAGLE context with its commit
                    drafters[b].extend(committed, _eagle_aux_range_b(aux_b, b, 0, len(committed)), base_pos=sp_b)
                if on_commits and on_commits[b]: on_commits[b](out[b], time.time() - t0)   # stream this commit
                if len(out[b]) >= mx[b] or (cur[b] in eos_set) or (eos_set & set(committed)):
                    done[b] = True
        for b in range(B):
            getattr(drafters[b], "cancel", lambda: None)()      # drop any standing request at drain
        if RECEIPTS:                                            # PROVE: sweep the ring once, like solo
            kw.send({"op": "receipt", "receipts": []}); receipts = recv_data(rx)
            if isinstance(receipts, dict):              # graph-A/B job: tail promoted the reply with counters
                graph_arm = {k: receipts.get(k) for k in ("graph", "graph_captured", "graph_skipped")}
                receipts = receipts.get("receipts", [])
    finally:
        if aux_local and lseq_rx < lseq_tx:                 # an ABORTED armed job must never leave the head blocked
            _drain_aux_local(pipe_sock, lseq_tx - lseq_rx)  # mid-send on a full lane / the lane dirty (review F1)
        kw.stop()                                           # job over: never leak a noop thread onto the reused socket
    receipts_ok = (_verify_receipts(receipts, S.cfg.num_hidden_layers, expected_nonce=job_nonce,
                                    check_chain=not M25_FP8_WIRE) if receipts   # fp8 wire is intentionally lossy
                   else (False if RECEIPTS else None))                          # fail closed, like solo
    dt = time.time() - t0
    res = []
    for b in range(B):                                  # trim at first eos, per stream
        o = out[b]
        for ee in eos_set:
            if ee in o: o = o[:o.index(ee)]; break
        res.append({"ok": True, "output_ids": o, "n_tokens": len(o), "prompt_tokens": len(prompts[b]),
                    "g": round(acc[b] / max(vrounds[b], 1), 3),   # accepted/verify-round — the drafting telemetry
                    "text": tok.decode(o, skip_special_tokens=True)})
    return {"streams": res, "B": B, "rounds": rounds, "depth": cur_depth,
            "wasted": wasted, "dt": dt, "prefill_s": prefill_s, "receipts": receipts,
            "receipts_ok": receipts_ok, "eagle": eagle_on,
            "graph_arm": graph_arm,   # graph-A/B jobs: tail's {graph, graph_captured, graph_skipped} at job end
            "aux_local": bool(aux_local),   # armed lane vs ridden-ring aux — arms must never be conflated
            "per_stage": {k: [round(v[0], 1), round(v[1], 1), v[2]] for k, v in per_stage.items()},   # stage -> [span_ms_sum, compute_ms_sum, n] under M25_STAGE_TIMING
            "agg_tok_s": sum(len(r["output_ids"]) for r in res) / max(dt, 1e-9)}


def coordinate_pipe_rows(pipe_sock, tok, messages_list, K, max_new, timeout, ret_sock, drafters,
                         tools=None, prefill_chunk=4096, max_ctx=0, reasoning=True,
                         on_commits=None, tools_b=None, swarm_id="swarm", job_id="job"):
    """DE-LOCKSTEP coordinator (M25_DELOCKSTEP): B independent per-stream spec-decode chains whose
    [1,K+1] solo-style frames INTERLEAVE on the ring — no shared round, no lockstep barrier. Each
    stream is exactly solo depth-1 EAGLE (draft from the verified hidden -> send -> commit on reply,
    full-accept bonus always available since a stream never has a second frame in flight); the WAN
    time of one stream hides the compute/drafting of the others, which is where the per-stream
    speedup over lockstep comes from (the streams ARE the pipeline). Replies arrive in global FIFO
    order (one pipe path, single-threaded stages); each carries a 'stream' tag the coordinator
    asserts against its own FIFO — any skew aborts LOUD. Job opening, prefill, receipts, graph-arm
    stamping and the aux_local lane are the lockstep path's, verbatim.

    PER-STREAM TREES (M25_TREE=1, the g lever for latency-floor rounds): each stream routes per
    round by coordinate_pipe_tree's exact rule — n-gram MATCHED -> the plain chain frame above;
    n-gram MISS -> one EAGLE tree frame (top-M best-first, run_block_tree_row against KV row b,
    tree_greedy_walk commit). Tree draws batch across ready streams via propose_tree_b (serial
    propose_tree x B would re-open the drafting tax). KV dirty-frontier contract per stream, solo's:
    a tree round leaves the committed path's KV rows dirty (they were tree nodes at scattered
    slots), tracked as pending_path[b] @ pos[b]; the NEXT frame re-feeds them — a tree as its causal
    trunk, a chain frame as its causal prefix (accept offset len-1). After any chain commit the only
    dirty token is cur (correction/bonus was never an input), so pending_path collapses to [cur] and
    the frame IS the standard [anchor]+draft. GREEDY/LOSSLESS by construction on both routes."""
    from eagle_draft import prefill_pair_tokens, fetch_b, fetch_tree_b, HybridDrafter
    from collections import deque
    B = len(messages_list)
    rx = ret_sock if ret_sock is not None else pipe_sock
    pipe_sock.settimeout(timeout)
    rx.settimeout(timeout)
    kw_job = os.environ.get("M25_KEEPWARM_JOB")
    kw = _KeepWarm(pipe_sock, interval_ms=kw_job if kw_job not in (None, "") else None)
    _eos = tok.eos_token_id
    eos_set = set(_eos) if isinstance(_eos, (list, tuple)) else {_eos}
    tb = tools_b if tools_b is not None else [tools] * B
    reas_b = reasoning if isinstance(reasoning, list) else [reasoning] * B
    mxnew_b = max_new if isinstance(max_new, list) else [max_new] * B
    prompts = [render_ids(tok, m, tools=tb[b], reasoning=reas_b[b]) for b, m in enumerate(messages_list)]
    tree_on = S.M25_TREE
    hd = 16                                              # ctx headroom past the stop: worst frame extent
    if tree_on:
        from tree_spec import tree_greedy_walk
        if not all(isinstance(d, HybridDrafter) or hasattr(d, "propose_tree") for d in drafters):
            # EXACTLY fetch_tree_b's tree-routing condition — a drafter that would silently chain
            # (e.g. a wrapped Hybrid: passes a looser getattr check, never trees) is a poisoned
            # tree-arm measurement, not a degrade (review F2)
            raise RuntimeError("M25_TREE=1 needs tree-capable drafters (HybridDrafter or "
                               "propose_tree; set M25_EAGLE=1 + M25_EAGLE_DIR)")
        tree_m = int(os.environ.get("M25_TREE_M", "12"))
        tree_topb = int(os.environ.get("M25_TREE_TOPB", "3"))
        tree_depth = int(os.environ.get("M25_TREE_DEPTH", "8"))
        # a tree frame spans trunk (<= tree_depth+1 re-fed committed) + tree_m nodes — the fixed
        # 16 headroom only covers K<=15 chains, and attn_tree_row's MAXLEN bound is a stage-killing
        # RuntimeError, not an EDGE error (review: M25_TREE_M>=17 near the cap kills a warm stage)
        hd = max(16, tree_m + tree_depth + 1)
    mx = [max(16, min(mxnew_b[b], max_ctx - len(p) - hd)) if max_ctx else mxnew_b[b]
          for b, p in enumerate(prompts)]
    out = [[] for _ in range(B)]; pos = [0] * B; cur = [0] * B; done = [False] * B
    acc = [0] * B; vrounds = [0] * B
    t_recv = 0.0; t_pf = time.time(); receipts = []; graph_arm = None; per_stage = {}
    for d in drafters:
        d.reset()                                        # fresh per-stream EAGLE context per job
    job_nonce = os.urandom(16).hex() if RECEIPTS else None
    aux_token = os.urandom(8).hex() if M25_AUX_LOCAL else None
    aux_local = False; lseq_tx = lseq_rx = 0
    rounds = 0
    try:
        rb_op = {"op": "reset_batch", "B": B, "swarm_id": swarm_id, "job_id": job_id}
        if job_nonce is not None:
            rb_op["nonce"] = job_nonce
        if M25_GRAPH_JOB is not None:
            rb_op["graph"] = M25_GRAPH_JOB
        if aux_token:
            rb_op["aux_local"] = aux_token
        kw.send(rb_op); ack = recv_data(rx)
        if isinstance(ack, dict) and ack.get("error"):
            raise TransportError(f"reset_batch refused: {ack['error']}")
        _check_reset_ack(rb_op, ack)
        if aux_token:
            aux_local = _aux_local_handshake(pipe_sock, aux_token)
            if not aux_local:
                print("[rows] aux_local asked but the head never acked — ridden-ring aux (old head?)", flush=True)
        for b in range(B):                               # per-stream prefill, exactly the lockstep path
            gen = prompts[b]
            starts_b = range(0, len(gen), prefill_chunk) if prefill_chunk else [0]
            for i in starts_b:
                chunk = gen[i:i + prefill_chunk] if prefill_chunk else gen
                pf = {"op": "verify", "stream": b, "token_ids": chunk, "start": i, "prefill": True}
                if aux_local:
                    pf["seq"] = lseq_tx; lseq_tx += 1
                kw.send(pf)
                rr, aux = _unpack(_reply_ok(recv_data(rx)))
                if aux_local:
                    loc = _pull_aux_local(pipe_sock, aux_token, lseq_rx); lseq_rx += 1
                    if loc:
                        aux = {**(aux or {}), **loc}
                if aux is not None:
                    drafters[b].extend(prefill_pair_tokens(gen, i, rr),
                                       _eagle_aux_range(aux, 0, len(rr)), base_pos=i)
            cur[b] = rr[-1]
            pos[b] = len(gen); out[b] = [cur[b]]
            if cur[b] in eos_set or len(out[b]) >= mx[b]: done[b] = True
            drafters[b].request(prompts[b] + [cur[b]], K)
            if on_commits and on_commits[b]: on_commits[b](out[b], 0.0)
        prefill_s = time.time() - t_pf; t0 = time.time()
        rx.settimeout(_reply_timeout(timeout))           # decode replies fail over in seconds, like solo
        dprefix = [prompts[b] + [cur[b]] for b in range(B)]
        pending_path = [[cur[b]] for b in range(B)]      # per-stream KV dirty frontier @ pos[b] (post-
        inflight_b = {}                                  # prefill: cur's row is unwritten, like solo).
        pend = deque()                                   # inflight_b: stream -> (kind, send_pos, payload,
                                                         # off|L) — at most ONE per stream, ever.

        def _fire(b, route):
            nonlocal lseq_tx
            kind, payload = route
            pp = pending_path[b]                         # dirty tokens re-fed as this frame's causal prefix
            if kind == "tree":
                token_ids, parents, pids = _build_tree_msg(pp, payload, pos[b])
                fr = {"op": "verify", "stream": b, "tree": True, "token_ids": token_ids,
                      "parents": parents, "pos_ids": pids, "start": pos[b]}
                inflight_b[b] = ("tree", pos[b], payload, len(pp))
            else:                                        # chain: pp==[cur] after chain commits -> the
                off = len(pp) - 1                        # standard [anchor]+draft frame; longer only on
                fr = {"op": "verify", "stream": b,       # the first frame after a tree round (refeed)
                      "token_ids": pp + payload, "start": pos[b]}
                inflight_b[b] = ("chain", pos[b], payload, off)
                dprefix[b] = dprefix[b] + payload
            if aux_local:
                fr["seq"] = lseq_tx; lseq_tx += 1
            kw.send(fr); pend.append(b)

        def _draw(bs):
            """Route + draft for streams bs (drafters already request()ed): chain on n-gram hit, tree
            on miss when tree_on — misses expand in ONE propose_tree_b lockstep batch."""
            if tree_on:
                return fetch_tree_b([drafters[b] for b in bs], tree_m, topb=tree_topb, max_depth=tree_depth)
            return [("chain", ds) for ds in fetch_b([drafters[b] for b in bs])]

        act = [b for b in range(B) if not done[b]]
        for b, route in zip(act, _draw(act)):            # first draws batched in ONE pass
            _fire(b, route)
        while pend:
            tr = time.time(); resp = _reply_ok(recv_data(rx)); t_recv += time.time() - tr
            if isinstance(resp, dict) and resp.get("stage_dt"):
                _acc_stage_dt(resp, per_stage)
            b = pend.popleft()
            if not (isinstance(resp, dict) and resp.get("stream") == b):
                raise TransportError(f"row reply skewed or UNTAGGED: expected stream {b}, got "
                                     f"{str(resp)[:80]} — aborting (an old stage routing row frames "
                                     f"down the solo path replies untagged = wrong KV)")
            r, aux = _unpack(resp)
            if aux_local:
                loc = _pull_aux_local(pipe_sock, aux_token, lseq_rx); lseq_rx += 1
                if loc:
                    aux = {**(aux or {}), **loc}
            if aux is None and rounds == 0:              # fail LOUD, like lockstep: no aux = a poisoned run
                raise TransportError("EAGLE drafters but the ring returned no aux — launch stages "
                                     "with M25_EAGLE=1")
            kind, sp, payload, ext = inflight_b.pop(b)
            if kind == "tree":
                if not resp.get("tree"):                 # version-mix guard: an OLD stage (no tree-row
                    raise TransportError(                # branch) runs a tree frame as CHAIN math and
                        f"tree frame for stream {b} came back without the tree echo — an old stage "
                        f"ran it down the chain row path (corrupted row KV w/ valid receipts); "
                        f"relaunch the ring on current code")
                tree, L = payload, ext                   # trunk of L re-fed dirty tokens + M tree nodes
                path_idx, committed = tree_greedy_walk(tree["tokens"], tree["parents"], r[L:], r[L - 1])
                vrounds[b] += 1; acc[b] += len(committed); rounds += 1
                out[b].extend(committed); cur[b] = committed[-1]
                pos[b] = sp + L                          # trunk KV is clean; the committed path is the
                pending_path[b] = list(committed)        # new dirty frontier (nodes at scattered slots)
                dprefix[b] = prompts[b] + out[b]
                if aux is not None:                      # committed[0] predicted by the anchor (node L-1);
                    pred_idx = ([L - 1] + [L + pi for pi in path_idx])[:len(committed)]   # committed[k>0]
                    drafters[b].extend(committed, _eagle_aux_nodes(aux, pred_idx),        # by path node k-1
                                       base_pos=pos[b] - 1)
            else:
                ds, off = payload, ext
                n = 0
                for j in range(K):
                    if ds[j] == r[off + j]: n += 1
                    else: break
                vrounds[b] += 1; rounds += 1
                if n == K:                               # full accept + bonus (nothing else in flight
                    committed = ds + [r[off + K]]        # for THIS stream, ever — solo depth-1 rule)
                    out[b].extend(committed); cur[b] = r[off + K]
                    dprefix[b] = dprefix[b] + [r[off + K]]   # the NEXT frame's anchor IS the bonus (review
                                                         # MAJOR-1: omitting this fed ds[-1] at the
                                                         # bonus's position -> corrupted KV, valid receipts)
                else:
                    committed = ds[:n] + [r[off + n]]
                    out[b].extend(committed); cur[b] = r[off + n]
                    dprefix[b] = prompts[b] + out[b]     # divergence: rebase the prefix (no stale frames exist)
                acc[b] += len(committed)                 # g = COMMITTED per verify round, UNIFORM across
                                                         # chain and tree rounds (review F1: the old mixed
                                                         # accept semantics — chain counted the bonus but
                                                         # not the correction, tree neither — understated
                                                         # tree arms by ~1/round and biased the A/B)
                pos[b] = sp + off + len(committed)       # off=0 collapses to the old += n+1 / += K+1
                pending_path[b] = [cur[b]]               # only cur is dirty after a chain commit
                if aux is not None:                      # aux rows [off, off+len(committed)) — the frame
                    drafters[b].extend(committed, _eagle_aux_range(aux, off, off + len(committed)),
                                       base_pos=sp + off)   # positions the committed tokens landed on
            if on_commits and on_commits[b]: on_commits[b](out[b], time.time() - t0)
            if len(out[b]) >= mx[b] or (cur[b] in eos_set) or (eos_set & set(committed)):
                done[b] = True
            else:
                drafters[b].request(dprefix[b], K)
                _fire(b, _draw([b])[0])                  # one-stream draw runs the sync-light batched path;
                                                         # other streams' frames are on the WAN meanwhile
        for b in range(B):
            getattr(drafters[b], "cancel", lambda: None)()
        if RECEIPTS:
            kw.send({"op": "receipt", "receipts": []}); receipts = recv_data(rx)
            if isinstance(receipts, dict):
                graph_arm = {k: receipts.get(k) for k in ("graph", "graph_captured", "graph_skipped")}
                receipts = receipts.get("receipts", [])
    finally:
        if aux_local and lseq_rx < lseq_tx:
            _drain_aux_local(pipe_sock, lseq_tx - lseq_rx)
        kw.stop()
    receipts_ok = (_verify_receipts(receipts, S.cfg.num_hidden_layers, expected_nonce=job_nonce,
                                    check_chain=not M25_FP8_WIRE) if receipts
                   else (False if RECEIPTS else None))
    dt = time.time() - t0
    res = []
    for b in range(B):
        o = out[b]
        for ee in eos_set:
            if ee in o: o = o[:o.index(ee)]; break
        res.append({"ok": True, "output_ids": o, "n_tokens": len(o), "prompt_tokens": len(prompts[b]),
                    "g": round(acc[b] / max(vrounds[b], 1), 3),   # committed/round (uniform chain+tree
                    "text": tok.decode(o, skip_special_tokens=True)})   # since #86; pre-#86 receipts
                                                                        # quoted accept-only g, ~1 lower
                                                                        # on divergence-heavy content
    return {"streams": res, "B": B, "rounds": rounds, "depth": 1, "wasted": 0,
            "dt": dt, "prefill_s": prefill_s, "receipts": receipts,
            "receipts_ok": receipts_ok, "eagle": True, "delockstep": True, "tree": bool(tree_on),
            "graph_arm": graph_arm, "aux_local": bool(aux_local),
            "per_stage": {k: [round(v[0], 1), round(v[1], 1), v[2]] for k, v in per_stage.items()},
            "agg_tok_s": sum(len(r["output_ids"]) for r in res) / max(dt, 1e-9)}


def _load(stage, nstages, lo, hi):
    S.vllm_ctx()
    layers = [S.Layer(i) for i in range(lo, hi)]
    parts = {"layers": layers, "head": stage == 0, "tail": stage == nstages - 1}
    if parts["head"]:
        parts["embed_w"] = S.raw("model.embed_tokens.weight").to(torch.bfloat16).to(dev)
    if parts["tail"]:
        parts["norm_w"] = S.raw("model.norm.weight").float().to(dev)
        parts["lm_head_w"] = S.raw("lm_head.weight").to(torch.bfloat16).to(dev)
    print(f"[s{stage}] loaded layers [{lo}:{hi}] ({torch.cuda.memory_allocated()/1e9:.1f}GB) — warming", flush=True)
    with torch.no_grad():
        S.run_block(layers, 0, torch.randn(1, 4, S.H, dtype=torch.bfloat16, device=dev) * 0.1, S._CTX[1])
        for L in layers:
            L.reset()
    torch.cuda.synchronize()
    if S.M25_SDPA:
        _sdpa_backend_probe(stage)
    return parts


def _tail_logits(h, parts):
    x = h.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + S.EPS) * parts["norm_w"]
    return (x.to(torch.bfloat16) @ parts["lm_head_w"].t())   # [1, s, vocab]


_GRAPH_CAP_LOGGED = set()                           # (s, bucket) shapes already reported as cap-skipped


def _block(grs, layers, start, x, vcfg):
    """Run one block. Route fixed-shape verify/decode blocks (small s = K+1) through a lazily-captured
    CUDA graph when the graph route is ACTIVE (env M25_CUDA_GRAPH, or per job via the reset op's
    {"graph": true/false} -> S.set_graph; recovers per-kernel launch overhead — the 35-50ms/block
    slow-CPU-stage lever); prefill (large s) stays eager. grs caches one GraphRunner per block size; each (s, bucket)
    pair is ONE captured graph and hybrid refeed frames make s variable, so a NEW pair is captured only
    while the process-wide S.M25_GRAPH_MAX budget lasts — past it (and after a failed capture, inside
    run()) the block runs EAGER: silent, counted, never fatal. The graphed path is bit-equivalent to
    eager-MANUAL attention (proven), so receipts + spec-decode losslessness are preserved; vs the eager
    SDPA-flash path it is the same accepted-kernel-numerics class as fp8 wire (the ring A/B judges
    accept/g)."""
    if S.M25_CUDA_GRAPH_ACTIVE and x.shape[1] <= 64:
        s = x.shape[1]
        gr = grs.get(s)
        if gr is None:
            grs[s] = gr = S.GraphRunner(layers, vcfg, s)
        alen = gr._bucket(start + s)
        if alen in gr.graphs or S._GRAPH_COUNT < S.M25_GRAPH_MAX:
            return gr.run(start, x)                 # replay, or capture within budget (OOM-safe in run())
        S._GRAPH_SKIPPED += 1                       # budget spent: new (s,bucket) shapes run eager
        if (s, alen) not in _GRAPH_CAP_LOGGED:      # log ONCE per skipped shape (count every block)
            _GRAPH_CAP_LOGGED.add((s, alen))
            print(f"[graph] cap: s={s} bucket={alen} -> eager "
                  f"({S._GRAPH_COUNT}/{S.M25_GRAPH_MAX} graphs captured)", flush=True)
    return S.run_block(layers, start, x, vcfg)


def _block_b(grs_b, layers, starts, x, vcfg):
    """run_block_decode_b's analog of _block: route the fixed-shape [B,K+1] verify_batch block through
    a lazily captured BatchGraphRunner when the graph route is ACTIVE (same runtime toggle as solo,
    plus the M25_BATCH_GRAPH escape hatch). `starts` is the HOST list off the wire — the runner's
    bounds check + static-buffer refresh must never touch a device tensor (sync). One runner per
    (B, s) shape; each of its buckets is ONE captured graph against the process-wide S.M25_GRAPH_MAX
    budget, with the same over-budget/failed-capture eager fallback semantics as _block."""
    if S.M25_CUDA_GRAPH_ACTIVE and M25_BATCH_GRAPH and x.shape[1] <= 64 and S.M25_BATCH > 1:
        B, s = x.shape[0], x.shape[1]
        gr = grs_b.get((B, s))
        if gr is None:
            grs_b[(B, s)] = gr = S.BatchGraphRunner(layers, vcfg, B, s)
        alen = gr._bucket(max(starts) + s)
        if alen in gr.graphs or S._GRAPH_COUNT < S.M25_GRAPH_MAX:
            return gr.run(starts, x)                # replay, or capture within budget (OOM-safe in run())
        S._GRAPH_SKIPPED += 1                       # budget spent: new (B,s,bucket) shapes run eager
        if (B, s, alen) not in _GRAPH_CAP_LOGGED:
            _GRAPH_CAP_LOGGED.add((B, s, alen))
            print(f"[graph] cap: B={B} s={s} bucket={alen} -> eager "
                  f"({S._GRAPH_COUNT}/{S.M25_GRAPH_MAX} graphs captured)", flush=True)
    return S.run_block_decode_b(layers, torch.as_tensor(starts, dtype=torch.long, device=dev), x, vcfg)


def _block_row(grs_r, layers, row, start, x, vcfg):
    """De-lockstep row-decode router (mirrors _block/_block_b): route the fixed-shape [1,K+1] row
    frame through the shared RowGraphRunner (ONE graph serves every KV row) when graphs are active."""
    if S.M25_CUDA_GRAPH_ACTIVE and M25_BATCH_GRAPH and x.shape[1] <= 64 and S.M25_BATCH > 1:
        s_ = x.shape[1]
        gr = grs_r.get(s_)
        if gr is None:
            grs_r[s_] = gr = S.RowGraphRunner(layers, vcfg, s_)
        alen = gr._bucket(start + s_)
        if alen in gr.graphs or S._GRAPH_COUNT < S.M25_GRAPH_MAX:
            return gr.run(row, start, x)
        S._GRAPH_SKIPPED += 1
        if ("row", s_, alen) not in _GRAPH_CAP_LOGGED:
            _GRAPH_CAP_LOGGED.add(("row", s_, alen))
            print(f"[graph] cap: row s={s_} bucket={alen} -> eager "
                  f"({S._GRAPH_COUNT}/{S.M25_GRAPH_MAX} graphs captured)", flush=True)
    return S.run_block_decode_row(layers, row, start, x, vcfg)


def _reset_flags(msg):
    """Stage-side per-job flags off a reset frame, applied BEFORE the reset is ack'd/propagated:
    'graph' flips the runtime CUDA-graph route (S.set_graph — refused loudly, 'GRAPH REFUSED' in the
    stage log, if the M25_STATIC_KV prereq is off). Field absent = keep the current setting, so
    pre-toggle coordinators change nothing. Head/middle stages forward the reset msg unchanged, which
    carries the field down the ring — every stage of the warm ring flips together, per job (the
    interleaved A/B lever). Returns the APPLIED route (bool) when the frame carried 'graph', else
    None — the tail acks it back so the coordinator can catch a refused toggle (_check_reset_ack)."""
    if "graph" in msg:
        return S.set_graph(msg["graph"])
    return None


def _merge_aux(upstream):
    """EAGLE: accumulate this stage's captured aux hidden states (S._AUX, only the aux layers in [lo,hi))
    onto whatever upstream stages already collected, so the tail returns all of [1,30,58] to the coordinator.
    Keys are str(layer_id); values are [s,H] bf16 cpu tensors — or [fp8_tensor, scale] pairs under
    M25_FP8_AUX (halves the dominant EAGLE decode payload; upstream entries are already packed and pass
    through untouched; the coordinator dequantizes once in _unpack). No-op unless M25_EAGLE."""
    acc = dict(upstream or {})
    if S.M25_EAGLE:
        for li, h in S._AUX.items():
            if M25_FP8_AUX:
                if h.dim() == 3:                       # batched [B,s,H]: per-STREAM scale — one stream's
                    sc = (h.detach().abs().amax(dim=(1, 2)) / 448.0).clamp(min=1e-8)   # outlier must not
                    q = (h / sc.view(-1, 1, 1)).to(torch.float8_e4m3fn).cpu()          # degrade its batch-
                    acc[str(li)] = [q, [float(x) for x in sc]]                         # mates' drafter aux
                else:
                    q, sc = _pack_h(h)
                    acc[str(li)] = [q, sc]
            else:
                acc[str(li)] = h.cpu()
    return acc


def _dt_sync():
    """Stage-timing compute stamp: force the async block forward to finish so the delta is real GPU time,
    not launch time (the very next op — fp8 pack / logits .tolist() — syncs anyway, so this costs nothing)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.monotonic()


def _dt_row(msg, stage, t_rx, t_comp):
    """[stage, span_ms, compute_ms] appended to the frame's accumulated stage_dt (the _merge_aux pattern).
    span = post-dequant recv -> just-before-send (block forward + fp8 pack); compute = recv -> synced block
    forward. Traversal-minus-span on the coordinator = wire + sidecar + codec — the transport bucket."""
    return (msg.get("stage_dt") or []) + [[stage, round((time.monotonic() - t_rx) * 1e3, 2),
                                           round((t_comp - t_rx) * 1e3, 2)]]


def _recv_pred(conn):
    """Head/middle predecessor recv (H4). Pre-frame idle is UNLIMITED — a warm ring parks between
    jobs indefinitely, so the blocking wait happens in select, which ignores the socket timeout —
    and the socket's timeout (set at accept, mirroring the tail's pred.settimeout) then only bounds
    a MID-frame stall: a peer that wedges half-way through a frame surfaces as socket.timeout ->
    EDGE_ERRORS -> the existing edge recovery, instead of holding the stage's only loop forever
    (the accepted socket used to have NO timeout at all)."""
    select.select([conn], [], [])
    return _hrecv(recv_msg(conn))


def _tail_accept(srv, pending=None, ret=None, timeout=None):
    """Tail bring-up handshake. TWO connections land on the tail: the coordinator-RETURN channel (greets
    with {op:'hello_return'} the instant it connects, because the coordinator sends data immediately) and
    the PREDECESSOR ring stream (silent until the first job byte flows). With the libp2p sidecars the
    predecessor's downstream connect to our ENG_IN is established LAZILY — only when the upstream stage
    first forwards data — and that only happens after the coordinator gets `ret_ok`. So requiring BOTH
    connections before acking the return channel (the old `c1=accept(); c2=accept()`) is a circular
    deadlock: no `ret_ok` until the predecessor connects, no predecessor data until `ret_ok`. The tail
    then wedges on the 2nd accept and the coordinator hangs forever on recv(ret_ok) with EMPTY output.

    Fix: accept connections one at a time and ack the return channel the INSTANT we identify it — do not
    wait for the predecessor. Whichever connection greets with hello_return is the return channel; a
    connection that speaks a JOB frame first is a (re-dialing direct-TCP) predecessor and its frame is
    handed back as `first_msg` (specpipe fill()'s semantic — closing it kill-looped stage replacement);
    a SILENT connection is adopted as the predecessor once the return channel exists (the libp2p
    predecessor connects lazily and never speaks first). Returns (ret, pred, first_msg); first_msg is
    None on the silent-predecessor path. Blocks until both channels exist.

    `pending` seeds the accepted-but-unidentified pool with leftovers from a torn-down session (a new
    coordinator's hello_return may already have been accepted when the old predecessor died) — closing
    them instead would EOF the reconnecting peer and re-wedge. `ret` seeds an already-live return
    channel (kept across a pred-death that raced a fresh coordinator) — then only the predecessor is
    awaited. `timeout` bounds the greeting read so a half-sent frame can't hang bring-up."""
    pred, first, pending = None, None, list(pending or [])
    while ret is None or pred is None:
        if ret is not None and pred is None and pending:   # silent conn + live ret -> it's the predecessor
            pred = pending.pop()
            for extra in pending:                          # 2-conn ring never leaves extras, but stay clean
                try: extra.close()
                except OSError: pass
            pending = []
            break
        ready, _, _ = select.select([srv] + pending, [], [])   # wake on a new conn OR a pending conn speaking
        for s in ready:
            if s is srv:
                c, _ = srv.accept(); c.setsockopt(*NODELAY); _keepalive(c); pending.append(c); continue
            pending.remove(s)                              # it spoke -> classify by content
            if timeout:
                s.settimeout(timeout)
            try:
                hello = recv_msg(s)
            except EDGE_ERRORS:
                try: s.close()
                except OSError: pass
                continue
            if isinstance(hello, dict) and hello.get("op") == "hello_return":
                if ret is not None:                        # newer coordinator wins (the old one is dead/stale)
                    try: ret.close()
                    except OSError: pass
                ret = s
                try:
                    send_msg(ret, "ret_ok")                # ACK NOW so the coordinator proceeds — pred can connect after
                except EDGE_ERRORS:                        # greeter died between hello and ack: not a session event
                    try: ret.close()
                    except OSError: pass
                    ret = None
            elif isinstance(hello, dict) and "op" in hello:
                if pred is not None:                       # a speaking predecessor replaces a stale one
                    try: pred.close()
                    except OSError: pass
                pred = s; first = hello                    # its first frame is real job data — hand it back
            else:
                try: s.close()                             # unexpected greeter (junk/probe) -> drop, keep waiting
                except OSError: pass
    return ret, pred, first


def serve(stage, nstages, lo, hi, port, nxt, timeout):
    parts = _load(stage, nstages, lo, hi)
    layers = parts["layers"]
    vcfg = S._CTX[1]
    graph_runners = {}                                # opt-in CUDA-graph cache (M25_CUDA_GRAPH); persists across jobs
    graph_runners_b = {}                              # batched-decode graph cache, keyed (B, s) — same lifetime
    graph_runners_r = {}                              # de-lockstep row-decode graph cache, keyed s
    def _dial_fwd():
        host, p = nxt.rsplit(":", 1)
        s = socket.socket(); s.settimeout(timeout); s.connect((host, int(p))); s.setsockopt(*NODELAY)
        _keepalive(s)
        return s
    nxt_sock = None
    nxt_kw = _KeepWarm()                              # cwnd keep-warm on the forward ring leg (idle a full
    if not parts["tail"]:                             # traversal between frames); attach() tracks re-dials
        nxt_sock = _dial_fwd()                        # launch-time dial stays strict: a dead --next at boot is a launcher bug
        nxt_kw.attach(nxt_sock)
        print(f"[s{stage}] forward connected -> {nxt}", flush=True)
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port)); srv.listen(2)
    print(f"[s{stage}] WARM, listening :{port}", flush=True)

    if parts["tail"]:
        node_key = load_or_make_node_key(NODE_KEY_PATH) if RECEIPTS else None
        # serve_tail_direct: ack the coordinator-return as soon as we identify it, THEN take the (lazily
        # connecting) predecessor — see _tail_accept for why requiring both up front deadlocks bring-up.
        # CHURN-RESILIENT (specpipe.serve_tail_fast's model): pred and ret have INDEPENDENT lifecycles.
        # ret dies alone (coordinator exit/crash, gateway restart) -> keep the predecessor + warm KV,
        # accept the next hello_return MID-SESSION, and drop the dead job's in-flight replies until the
        # next reset (a stale reply poisons the new coordinator's handshake). Only a pred death tears the
        # session down — and takes ret with it, so a reset-ok can never go to a dead coordinator's channel.
        # This tail was the wedge's first domino: it closed pred whenever ret died, which EOF'd every
        # upstream stage in turn and left the whole warm ring needing a relaunch per coordinator.
        ret = pred = None; pending = []; stale = False
        ret_kw = _KeepWarm()                     # cwnd keep-warm on the tail->coordinator return leg;
                                                 # attach() tracks every ret (re)adoption below

        def _ret_send(o):
            # Deliver a reply to the coordinator-return. On failure the RETURN channel is dead, not the
            # session: drop only ret (the next coordinator brings a fresh one) and mark the job stale.
            nonlocal ret, stale
            if ret is None:
                return
            try:
                ret_kw.send(o)
            except EDGE_ERRORS as e:
                print(f"[tail] return edge died on send ({type(e).__name__}); keeping predecessor+KV", flush=True)
                try: ret.close()
                except OSError: pass
                ret = None; stale = True; ret_kw.attach(None)

        while True:
            if pred is None:
                # re-accept the predecessor (and the return channel unless a live one was carried over).
                # A carried-over live ret means a coordinator that may still push in-flight frames for the
                # OLD (now KV-reset) job, so STAY stale until its next reset clears the session below; only a
                # fresh ret with no prior session starts un-stale (its coordinator hellos + resets anyway).
                carried = ret is not None
                ret, pred, queued = _tail_accept(srv, pending, ret=ret, timeout=timeout)
                ret_kw.attach(ret)               # after ret_ok went out — a noop can never precede the ack
                pending = []                     # consumed (became ret/pred or were closed) — don't double-select
                pred.settimeout(timeout)         # bounds a mid-frame stall; idle waiting happens in select below
                stale = carried                  # carried live ret -> drop its stale in-flight until reset
                print("[tail] predecessor + coord-return connected", flush=True)
            signer = None; job_graph = None          # job_graph: the reset's APPLIED graph arm (None = plain job)
            with torch.no_grad():
                try:
                    while True:
                        # Multiplex: pred carries job data; srv carries a reconnecting coordinator's
                        # hello_return, which MUST be accepted mid-session or coordinator churn wedges the
                        # warm ring; pending holds accepted-but-silent conns (never block-recv a silent
                        # conn — the _tail_accept bring-up deadlock, same reasoning). `queued` is a job
                        # frame already read off a newly-adopted predecessor — process it before selecting.
                        if queued is not None:
                            msg, queued = _hrecv(queued), None
                        else:
                            ready, _, _ = select.select([srv, pred] + pending, [], [])
                            if srv in ready:
                                c, _ = srv.accept(); c.setsockopt(*NODELAY); _keepalive(c); pending.append(c)
                                if len(pending) > 8:       # reap silent junk before it grows the select set
                                    old = pending.pop(0)
                                    try: old.close()
                                    except OSError: pass
                                continue
                            spoke = next((s for s in pending if s in ready), None)
                            if spoke is not None:
                                pending.remove(spoke)
                                spoke.settimeout(timeout)  # a half-sent greeting must not hang the live session
                                try:
                                    hello = recv_msg(spoke)
                                except EDGE_ERRORS:
                                    try: spoke.close()
                                    except OSError: pass
                                    continue
                                if isinstance(hello, dict) and hello.get("op") == "hello_return":
                                    if ret is not None:    # coordinator churn: the old return channel is dead
                                        try: ret.close()   # even if this write-only socket never told us
                                        except OSError: pass
                                        stale = True       # in-flight traffic belongs to the dead job
                                    ret = spoke
                                    try:
                                        send_msg(ret, "ret_ok"); ret.settimeout(None)   # ret is untimed, like bring-up
                                        ret_kw.attach(ret)                   # after ret_ok: noop never precedes the ack
                                        print("[tail] coord-return (re)connected mid-session", flush=True)
                                    except EDGE_ERRORS:    # reconnector died between hello and ack: not a pred event
                                        try: ret.close()
                                        except OSError: pass
                                        ret = None; ret_kw.attach(None)
                                elif isinstance(hello, dict) and "op" in hello:
                                    # a NEW predecessor speaking its first job frame (direct-TCP stage
                                    # replacement after a silent pred death) — adopt it, keep the frame
                                    try: pred.close()
                                    except OSError: pass
                                    pred = spoke; queued = hello; stale = False
                                    print("[tail] predecessor REPLACED mid-session", flush=True)
                                else:
                                    try: spoke.close()     # unexpected greeter (junk/probe) -> drop
                                    except OSError: pass
                                continue
                            if pred not in ready:
                                continue
                            msg = _hrecv(recv_msg(pred))
                        if msg.get("op") == "noop":           # predecessor keep-warm frame: leg-local, never
                            continue                          # answered/forwarded/attested/timed (back to select)
                        t_rx = time.monotonic()               # stage-timing origin (cheap; used only under M25_STAGE_TIMING)
                        if stale or ret is None:
                            # These messages belong to a job whose coordinator died: don't compute them,
                            # never answer them. The next job boundary (reset) re-arms the session — but
                            # only once a live return channel exists to ack it (a reset with ret=None is
                            # the dead coordinator's own; its successor always hellos before sending).
                            if ret is None or msg.get("op") not in ("reset", "reset_batch"):
                                continue
                            stale = False
                        if msg.get("op") == "job_error":   # an upstream stage rejected THIS job (H1
                            _ret_send({"error": msg["error"]})   # backstop): relay the structured error;
                            stale = True; continue               # job dead, KV+process alive — the next
                                                                 # reset re-arms via the stale machinery
                        if msg["op"] == "reset":
                            job_graph = _reset_flags(msg)   # per-job runtime flags (graph A/B toggle)
                            for L in layers:
                                L.reset()
                            if "keepwarm_ms" in msg:        # coordinator toggle rode the reset (interleaved A/B)
                                ret_kw.set_interval(msg["keepwarm_ms"])
                            if RECEIPTS:                    # start this job's per-stage activation hash-chain
                                signer = ReceiptSigner(node_key, msg.get("swarm_id", "swarm"),
                                                       msg.get("job_id", "job"), lo, hi,
                                                       nonce=msg.get("nonce"))   # sign the job freshness challenge in
                            # Graph-stamped resets ack the APPLIED route + counters so the coordinator
                            # can catch a refused toggle before it poisons the A/B (_check_reset_ack).
                            # Plain resets keep the bare "ok" (old-coordinator compat). Only the TAIL's
                            # applied value rides this ack — a head/middle refusal is only visible as
                            # 'GRAPH REFUSED' in that stage's own log (the runbook greps for it).
                            _ret_send("ok" if job_graph is None else
                                      {"ok": 1, "graph": job_graph, "graph_captured": S._GRAPH_COUNT,
                                       "graph_skipped": S._GRAPH_SKIPPED}); continue
                        if msg["op"] == "receipt":          # job done: sign + return the full ring's receipts
                            if RECEIPTS and signer is not None:
                                msg.setdefault("receipts", []).append({"stage": "tail", **signer.finalize()})
                            # graph-A/B jobs: promote the reply to a dict carrying the tail's graph
                            # counters, so the job record shows how graphed the arm ACTUALLY was;
                            # plain jobs keep the bare receipts list (old-coordinator compat)
                            rec = msg.get("receipts", [])
                            _ret_send(rec if job_graph is None else
                                      {"receipts": rec, "graph": job_graph,
                                       "graph_captured": S._GRAPH_COUNT,
                                       "graph_skipped": S._GRAPH_SKIPPED}); continue
                        if msg["op"] == "reset_batch":      # continuous batching: logical reset of all rows
                            if int(msg.get("B", 1)) > S.M25_BATCH:   # nack a job wider than the launch-time KV
                                _ret_send({"error": f"B={msg.get('B')} > ring M25_BATCH={S.M25_BATCH}"}); continue
                            job_graph = _reset_flags(msg)   # per-job graph arm, like solo's reset (also CLEARS a
                            for L in layers: L.reset()      # stale solo job_graph when the field is absent)
                            if RECEIPTS:                    # batched jobs get a FRESH signer (a solo job's
                                signer = ReceiptSigner(node_key, msg.get("swarm_id", "swarm"),   # stale signer
                                                       msg.get("job_id", "job"), lo, hi,        # must never
                                                       nonce=msg.get("nonce"))                  # bleed in)
                            _ret_send("ok" if job_graph is None else    # graph-stamped batched jobs ack the APPLIED
                                      {"ok": 1, "graph": job_graph,     # route + counters (_check_reset_ack's food);
                                       "graph_captured": S._GRAPH_COUNT,   # plain jobs keep the bare ok (compat)
                                       "graph_skipped": S._GRAPH_SKIPPED}); continue
                        try:
                            if msg["op"] == "verify_batch":     # batched decode: [B,K+1,H] -> per-stream argmax [B][K+1]
                                x = msg["h"].to(dev)
                                h = _block_b(graph_runners_b, layers, msg["start_b"], x, vcfg)
                                t_comp = _dt_sync() if S.M25_STAGE_TIMING else 0.0
                                if RECEIPTS and signer is not None:   # batched rounds are attested like solo — the
                                    signer.observe(_act_digest(x), _act_digest(h))   # standard path must stay receipt-covered
                                toks = _tail_logits(h, parts).argmax(-1).tolist()
                                if S.M25_EAGLE:
                                    aux = _merge_aux(msg.get("aux"))
                                    if M25_AUX_SLIM and msg.get("tids") is not None:   # slice the return-leg aux to
                                        aux = _slim_aux_b(aux, _aux_keep_lens(msg["tids"], toks))   # accepted prefixes
                                    o = {"toks": toks, "aux": aux}
                                else:
                                    o = {"toks": toks}                # timing promotes bare rows to a dict (like solo)
                                if S.M25_STAGE_TIMING:                # batched rounds get the same per-stage [span,
                                    o["stage_dt"] = _dt_row(msg, "tail", t_rx, t_comp)   # compute] stamps as solo — the
                                                                      # round-decomposition experiment's food
                                _ret_send(o if (S.M25_EAGLE or S.M25_STAGE_TIMING) else toks)
                                continue
                            if msg.get("prefill") and "stream" in msg:  # BATCHED prefill into row b (single-stream prefill has no 'stream' -> falls through to the normal path)
                                x = msg["h"].to(dev)
                                h = S.run_block_prefill_b(layers, msg["stream"], msg["start"], x, vcfg)
                                if RECEIPTS and signer is not None:
                                    signer.observe(_act_digest(x), _act_digest(h))
                                toks = _tail_logits(h, parts).argmax(-1)[0].tolist()
                                _ret_send({"toks": toks, "aux": _merge_aux(msg.get("aux"))} if S.M25_EAGLE else toks); continue
                            if msg.get("tree") and "stream" in msg:          # DE-LOCKSTEP tree-verify: one stream's
                                b = msg["stream"]                            # tree-masked block against KV row b —
                                x = msg["h"].to(dev)                         # MUST route before the row branch (a
                                h = S.run_block_tree_row(layers, b, msg["start"], x, vcfg,   # chain-math tree frame
                                                         msg["parents"], msg["pos_ids"])     # = silent KV corruption
                                t_comp = _dt_sync() if S.M25_STAGE_TIMING else 0.0           # with valid receipts)
                                if RECEIPTS and signer is not None:
                                    signer.observe(_act_digest(x), _act_digest(h))
                                toks = _tail_logits(h, parts).argmax(-1)[0].tolist()
                                o = {"toks": toks, "stream": b, "tree": True}   # tree echo: an OLD stage that ran
                                if S.M25_EAGLE:                              # this frame as chain math replies
                                    o["aux"] = _merge_aux(msg.get("aux"))    # without it -> the coordinator
                                if S.M25_STAGE_TIMING:                       # aborts LOUD (version-mix guard)
                                    o["stage_dt"] = _dt_row(msg, "tail", t_rx, t_comp)
                                _ret_send(o); continue
                            if "stream" in msg and msg["op"] == "verify":   # DE-LOCKSTEP row decode: one stream's
                                b = msg["stream"]                            # solo-style frame against KV row b
                                x = msg["h"].to(dev)
                                h = _block_row(graph_runners_r, layers, b, msg["start"], x, vcfg)
                                t_comp = _dt_sync() if S.M25_STAGE_TIMING else 0.0
                                if RECEIPTS and signer is not None:          # row frames are attested like every op
                                    signer.observe(_act_digest(x), _act_digest(h))
                                toks = _tail_logits(h, parts).argmax(-1)[0].tolist()
                                o = {"toks": toks, "stream": b}              # stream tag = the coordinator's LOUD
                                if S.M25_EAGLE:                              # FIFO-pairing guard
                                    o["aux"] = _merge_aux(msg.get("aux"))
                                if S.M25_STAGE_TIMING:
                                    o["stage_dt"] = _dt_row(msg, "tail", t_rx, t_comp)
                                _ret_send(o); continue
                            if msg.get("tree"):                 # EAGLE tree-verify: per-node argmax over the tree-masked block
                                x = msg["h"].to(dev)
                                h = S.run_block_tree(layers, msg["start"], x, vcfg, msg["parents"], msg["pos_ids"])
                                t_comp = _dt_sync() if S.M25_STAGE_TIMING else 0.0
                                if RECEIPTS and signer is not None:   # attest tree blocks too — verification must not silently turn off under M25_TREE
                                    signer.observe(_act_digest(x), _act_digest(h))
                                toks = _tail_logits(h, parts).argmax(-1)[0].tolist()
                                o = {"toks": toks, "aux": _merge_aux(msg.get("aux"))} if S.M25_EAGLE else toks
                                if S.M25_STAGE_TIMING:            # timing promotes a bare-list reply to a dict (coordinator _unpack handles both)
                                    o = o if isinstance(o, dict) else {"toks": o}
                                    o["stage_dt"] = _dt_row(msg, stage, t_rx, t_comp)
                                _ret_send(o); continue
                            x = msg["h"].to(dev)
                            h = _block(graph_runners, layers, msg["start"], x, vcfg)
                            t_comp = _dt_sync() if S.M25_STAGE_TIMING else 0.0
                            if RECEIPTS and signer is not None:   # attest this block's input->output transform
                                signer.observe(_act_digest(x), _act_digest(h))
                            toks = _tail_logits(h, parts).argmax(-1)[0].tolist()
                            o = {"toks": toks, "aux": _merge_aux(msg.get("aux"))} if S.M25_EAGLE else toks
                            if S.M25_STAGE_TIMING:                # timing promotes a bare-list reply to a dict (coordinator _unpack handles both)
                                o = o if isinstance(o, dict) else {"toks": o}
                                o["stage_dt"] = _dt_row(msg, stage, t_rx, t_comp)
                            _ret_send(o)
                        except RuntimeError as e:
                            # H1 stage backstop: a KV overflow (or any per-job compute failure) is a JOB
                            # error, never a process exit — serve()'s entrypoint is unwrapped, and a
                            # RuntimeError is NOT in EDGE_ERRORS, so it used to escape the edge recovery
                            # and KILL the warm tail. Reply the structured error, hold the session stale;
                            # the coordinator raises JobRejected (never retried as an edge fault).
                            _ret_send({"error": {"code": "kv_overflow", "stage": "tail",
                                                 "message": str(e)[:300]}})
                            stale = True
                            continue
                except EDGE_ERRORS as e:
                    # PREDECESSOR edge died (ret failures are absorbed in _ret_send, never here). This is
                    # either the coordinator dying (cascade) OR just an INTERNAL ring leg blipping while the
                    # coordinator is alive — and the tail cannot tell which from a bare EOF. So ALWAYS KEEP
                    # the coordinator return channel across it: reset the KV, re-accept the predecessor (the
                    # upstream stage re-dials), and hold the session stale so every in-flight reply is dropped
                    # until the coordinator's next RESET re-arms it (the `stale` gate below). If the
                    # coordinator really died, the kept ret is dead-harmless — the next _ret_send drops it, or
                    # the reconnecting coordinator's hello_return replaces it mid-session. Closing a LIVE
                    # coordinator's ret here (the old behaviour) forced it into a full reconnect that raced the
                    # return-tunnel recovery and WEDGED — fatal on a permissionless ring where internal-leg
                    # blips are the steady state. Symmetric to _ret_send's "ret dies -> keep predecessor + KV".
                    print(f"[tail] predecessor edge closed ({type(e).__name__}); keeping coord-return, "
                          f"re-accepting predecessor (job stale until next reset)", flush=True)
                    for L in layers:
                        L.reset()
                    try:
                        if pred is not None: pred.close()
                    except OSError: pass
                    pred = None; stale = True             # ret KEPT (ret_kw stays attached); `stale` drops in-flight until the reset
        return

    # head / middle: single predecessor connection, FIRE-FORWARD (direct mode, no relay-back)
    node_key = load_or_make_node_key(NODE_KEY_PATH) if RECEIPTS else None
    while True:
        tries = 0
        while nxt_sock is None:                       # forward link dropped (churn cascade): rebuild it BEFORE
            try:                                      # accepting a new predecessor, so the ring re-handshakes
                nxt_sock = _dial_fwd()                # front-to-back onto WARM stages — no relaunch, no reload
                nxt_kw.attach(nxt_sock)
                print(f"[s{stage}] forward link rebuilt -> {nxt}", flush=True)
            except OSError:
                tries += 1                            # dial forever: a stage holding warm weights is worth more
                if tries % 60 == 0:                   # waiting than dead (downstream may be mid-restart)
                    print(f"[s{stage}] forward re-dial {nxt} still failing ({tries} tries)", flush=True)
                time.sleep(0.5)
        conn, _ = srv.accept(); conn.setsockopt(*NODELAY); _keepalive(conn)
        conn.settimeout(timeout)                  # bounds a MID-frame stall (H4); idle waits in _recv_pred's select
        print(f"[s{stage}] predecessor connected", flush=True)
        signer = None; aux_local_job = None           # aux_local_job: this batched job's head-local aux lane
        with torch.no_grad():
            try:
                while True:
                    msg = _recv_pred(conn)
                    if msg.get("op") == "noop":                 # predecessor keep-warm frame: leg-local,
                        continue                                # never forwarded/answered/attested/timed
                    t_rx = time.monotonic()               # stage-timing origin (cheap; used only under M25_STAGE_TIMING)
                    if msg.get("op") == "job_error":     # an upstream stage rejected the job (H1
                        nxt_kw.send(msg); continue          # backstop): relay it down to the tail untouched
                    if msg["op"] == "reset":
                        _reset_flags(msg)                       # per-job runtime flags (graph A/B toggle)
                        aux_local_job = None                    # a solo job takes over: the local lane closes
                        for L in layers:
                            L.reset()
                        if "keepwarm_ms" in msg:                # coordinator toggle rode the reset (interleaved A/B)
                            nxt_kw.set_interval(msg["keepwarm_ms"])
                        if RECEIPTS:                            # start this job's per-stage activation hash-chain
                            signer = ReceiptSigner(node_key, msg.get("swarm_id", "swarm"),
                                                   msg.get("job_id", "job"), lo, hi,
                                                   nonce=msg.get("nonce"))   # sign the job freshness challenge in
                        nxt_kw.send(msg); continue              # propagate reset down chain UNCHANGED (carries 'graph' + 'keepwarm_ms')
                    if msg["op"] == "receipt":                  # job done: sign + accumulate forward to the tail
                        if RECEIPTS and signer is not None:
                            msg.setdefault("receipts", []).append({"stage": stage, **signer.finalize()})
                        nxt_kw.send(msg); continue
                    if msg["op"] == "reset_batch":              # continuous batching: propagate logical reset
                        _reset_flags(msg)                       # per-job graph arm applies on EVERY stage (tail acks)
                        aux_local_job = (msg["aux_local"]   # the lane TOKEN: a per-job nonce (job_id is "job"
                                         if parts["head"] and msg.get("aux_local") and S.M25_EAGLE else None)   # everywhere — it can't disambiguate residue)
                        if aux_local_job is not None:           # ack the local lane on the pipe BEFORE forwarding the
                            send_msg(conn, {"op": "aux_local_ok", "job": aux_local_job})   # reset — the coordinator's
                                                                # handshake read is then causally satisfied
                        for L in layers: L.reset()
                        if RECEIPTS:                            # fresh per-job signer, same as tail (never let a
                            signer = ReceiptSigner(node_key, msg.get("swarm_id", "swarm"),   # solo job's signer
                                                   msg.get("job_id", "job"), lo, hi,         # bleed into batched)
                                                   nonce=msg.get("nonce"))
                        nxt_kw.send(msg); continue
                    try:
                        if msg["op"] == "verify_batch":             # batched decode: head embeds [B,K+1], else fwd [B,K+1,H]
                            if parts["head"]:
                                h = torch.nn.functional.embedding(torch.tensor(msg["token_ids_b"], device=dev), parts["embed_w"])
                            else:
                                h = msg["h"].to(dev)
                            x = h
                            h = _block_b(graph_runners_b, layers, msg["start_b"], h, vcfg)
                            if RECEIPTS and signer is not None:     # attest batched rounds like solo
                                signer.observe(_act_digest(x), _act_digest(h))
                            fwd = {"op": "verify_batch", "h": h, "start_b": msg["start_b"]}
                            if S.M25_EAGLE:                          # per-stream aux ([B,K+1,H] rows) to the tail
                                if aux_local_job is None:            # armed head: its own aux goes on the LOCAL lane
                                    fwd["aux"] = _merge_aux(msg.get("aux"))   # (below), never onto the WAN legs
                                tb = msg.get("token_ids_b") or msg.get("tids")   # drafted rows ride to the TAIL (tiny
                                if tb is not None:                               # ints) so it can slice the return-leg
                                    fwd["tids"] = tb                             # aux to accepted prefixes (M25_AUX_SLIM)
                            if S.M25_STAGE_TIMING:
                                fwd["stage_dt"] = _dt_row(msg, stage, t_rx, _dt_sync())
                            nxt_kw.send(_hsend(fwd))
                            if aux_local_job is not None:            # forward-FIRST, then the local lane: a big local
                                send_msg(conn, {"op": "aux_local", "job": aux_local_job,   # frame must never block
                                                "seq": msg.get("seq"),                      # the ring send (localhost
                                                "aux": {str(li): v.detach().to(torch.bfloat16).cpu()   # buffer deadlock)
                                                        for li, v in S._AUX.items()}})
                            continue
                        if msg.get("tree") and "stream" in msg:     # DE-LOCKSTEP tree-verify: one stream's tree-
                            b = msg["stream"]                       # masked block against KV row b — MUST route
                            if parts["head"]:                       # before the row branch (a tree frame down the
                                h = torch.nn.functional.embedding(  # chain-math row path = silent KV corruption
                                    torch.tensor([msg["token_ids"]], device=dev), parts["embed_w"])   # w/ valid receipts)
                            else:
                                h = msg["h"].to(dev)
                            x = h
                            h = S.run_block_tree_row(layers, b, msg["start"], h, vcfg, msg["parents"], msg["pos_ids"])
                            t_comp = _dt_sync() if S.M25_STAGE_TIMING else 0.0
                            if RECEIPTS and signer is not None:
                                signer.observe(_act_digest(x), _act_digest(h))
                            fwd = {"op": "verify", "tree": True, "stream": b, "h": h, "start": msg["start"],
                                   "parents": msg["parents"], "pos_ids": msg["pos_ids"]}
                            if S.M25_EAGLE:
                                if aux_local_job is None:                    # armed head: own aux on the local lane
                                    fwd["aux"] = _merge_aux(msg.get("aux"))
                            if S.M25_STAGE_TIMING:
                                fwd["stage_dt"] = _dt_row(msg, stage, t_rx, t_comp)
                            nxt_kw.send(_hsend(fwd))
                            if aux_local_job is not None:                    # forward-first, like every armed op
                                send_msg(conn, {"op": "aux_local", "job": aux_local_job, "seq": msg.get("seq"),
                                                "aux": {str(li): v.detach().to(torch.bfloat16).cpu()
                                                        for li, v in S._AUX.items()}})
                            continue
                        if "stream" in msg and msg["op"] == "verify" and not msg.get("prefill"):   # DE-LOCKSTEP row decode
                            b = msg["stream"]
                            if parts["head"]:
                                h = torch.nn.functional.embedding(torch.tensor([msg["token_ids"]], device=dev), parts["embed_w"])
                            else:
                                h = msg["h"].to(dev)
                            x = h
                            h = _block_row(graph_runners_r, layers, b, msg["start"], h, vcfg)
                            if RECEIPTS and signer is not None:
                                signer.observe(_act_digest(x), _act_digest(h))
                            fwd = {"op": "verify", "stream": b, "h": h, "start": msg["start"]}
                            if S.M25_EAGLE:
                                if aux_local_job is None:                    # armed head: own aux on the local lane
                                    fwd["aux"] = _merge_aux(msg.get("aux"))
                            if S.M25_STAGE_TIMING:
                                fwd["stage_dt"] = _dt_row(msg, stage, t_rx, _dt_sync())
                            nxt_kw.send(_hsend(fwd))
                            if aux_local_job is not None:                    # forward-first, like every armed op
                                send_msg(conn, {"op": "aux_local", "job": aux_local_job, "seq": msg.get("seq"),
                                                "aux": {str(li): v.detach().to(torch.bfloat16).cpu()
                                                        for li, v in S._AUX.items()}})
                            continue
                        if msg.get("prefill") and "stream" in msg:  # BATCHED prefill into row b (single-stream prefill has no 'stream' -> normal path)
                            if parts["head"]:
                                h = torch.nn.functional.embedding(torch.tensor([msg["token_ids"]], device=dev), parts["embed_w"])
                            else:
                                h = msg["h"].to(dev)
                            x = h
                            h = S.run_block_prefill_b(layers, msg["stream"], msg["start"], h, vcfg)
                            if RECEIPTS and signer is not None:
                                signer.observe(_act_digest(x), _act_digest(h))
                            fwd = {"op": "verify", "stream": msg["stream"], "h": h, "start": msg["start"], "prefill": True}
                            if S.M25_EAGLE:                          # stream-b prefill aux feeds that stream's drafter context
                                if aux_local_job is None:            # armed head: own aux on the local lane (below)
                                    fwd["aux"] = _merge_aux(msg.get("aux"))
                            nxt_kw.send(_hsend(fwd))
                            if aux_local_job is not None:            # forward-first, exactly like verify_batch
                                send_msg(conn, {"op": "aux_local", "job": aux_local_job, "seq": msg.get("seq"),
                                                "aux": {str(li): v.detach().to(torch.bfloat16).cpu()
                                                        for li, v in S._AUX.items()}})
                            continue
                        if msg.get("tree"):                         # EAGLE tree-verify: tree-masked block, thread the tree forward
                            if parts["head"]:
                                h = torch.nn.functional.embedding(torch.tensor([msg["token_ids"]], device=dev), parts["embed_w"])
                            else:
                                h = msg["h"].to(dev)
                            x = h
                            h = S.run_block_tree(layers, msg["start"], h, vcfg, msg["parents"], msg["pos_ids"])
                            t_comp = _dt_sync() if S.M25_STAGE_TIMING else 0.0
                            if RECEIPTS and signer is not None:     # attest tree blocks too — verification must not silently turn off under M25_TREE
                                signer.observe(_act_digest(x), _act_digest(h))
                            fwd = {"op": "verify", "tree": True, "h": h, "start": msg["start"],
                                   "parents": msg["parents"], "pos_ids": msg["pos_ids"]}
                            if S.M25_EAGLE:
                                fwd["aux"] = _merge_aux(msg.get("aux"))
                            fwd = _hsend(fwd)
                            if S.M25_STAGE_TIMING:
                                fwd["stage_dt"] = _dt_row(msg, stage, t_rx, t_comp)
                            nxt_kw.send(fwd); continue
                        if "token_ids" in msg:                      # head: embed the coordinator's token ids
                            h = torch.nn.functional.embedding(torch.tensor([msg["token_ids"]], device=dev), parts["embed_w"])
                        else:
                            h = msg["h"].to(dev)
                        x = h
                        h = _block(graph_runners, layers, msg["start"], h, vcfg)
                        t_comp = _dt_sync() if S.M25_STAGE_TIMING else 0.0
                        if RECEIPTS and signer is not None:         # attest this block's input->output transform
                            signer.observe(_act_digest(x), _act_digest(h))
                        fwd = {"op": "verify", "h": h, "start": msg["start"]}
                        if S.M25_EAGLE:                              # carry aux hidden states forward to the tail (EAGLE)
                            fwd["aux"] = _merge_aux(msg.get("aux"))
                        fwd = _hsend(fwd)
                        if S.M25_STAGE_TIMING:
                            fwd["stage_dt"] = _dt_row(msg, stage, t_rx, t_comp)
                        nxt_kw.send(fwd)
                    except RuntimeError as e:
                        # H1 stage backstop, head/middle half: a per-job compute failure (KV overflow)
                        # must never escape serve()'s unwrapped entrypoint and kill the warm stage.
                        # Signal a structured job_error down the ring — the tail relays it to the
                        # coordinator, which raises JobRejected. Ring alive, job dead.
                        nxt_kw.send({"op": "job_error", "stage": stage,
                                     "error": {"code": "kv_overflow", "message": str(e)[:300]}})
                        continue
            except EDGE_ERRORS as e:
                print(f"[s{stage}] edge closed ({type(e).__name__}); reset + drop forward link", flush=True)
                for L in layers:
                    L.reset()
                for s in (conn, nxt_sock):            # deliberately drop the forward link too: the next stage
                    if s is not None:                 # sees EOF and cascades, so the WHOLE ring re-handshakes
                        try: s.close()                # fresh (warm weights intact) and a new coordinator can
                        except OSError: pass          # drive it — specpipe's proven recovery choreography
                nxt_sock = None; nxt_kw.attach(None)


def _sweep_summary(rows):
    """Pure: format a K/depth sweep into an aligned table + the best-throughput row. No torch/model
    deps so it unit-tests standalone (research/m25_sweep_test.py). `h_kb` is the per-traversal
    inter-stage hidden-state payload (K+1)*H*fp16 — the bandwidth term that caps how far K pays off
    once GPU compute is flat in token count, so it's printed next to tok/s to read the sweep."""
    hdr = f"{'K':>3} {'depth':>5} {'tok/s':>7} {'g':>6} {'accept':>7} {'prefill':>8} {'ntok':>5} {'h/trav':>8}"
    lines = ["", "=== M2.5 swarm sweep (decode tok/s, warm over libp2p) ===", hdr, "-" * len(hdr)]
    for r in rows:
        flag = "" if r.get("ok") else "  <-- FAIL"
        lines.append(f"{r['K']:>3} {r['depth']:>5} {r['tok_s']:>7.2f} {r['g']:>6.2f} "
                     f"{r['accept'] * 100:>6.0f}% {r['prefill_s']:>7.2f}s {r['ntok']:>5} {r['h_kb']:>6.1f}K{flag}")
    ok = [r for r in rows if r.get("ok") and r["tok_s"] > 0]
    best = max(ok, key=lambda r: r["tok_s"]) if ok else None
    if best:
        lines.append("-" * len(hdr))
        lines.append(f"BEST: K={best['K']} depth={best['depth']} -> {best['tok_s']:.2f} tok/s "
                     f"(g={best['g']:.2f}, accept={best['accept'] * 100:.0f}%)")
    return "\n".join(lines), best


def _run_job(pipe, ret, tok, messages, k, max_new, timeout, d, ngram_n, prefill_chunk, tools=None,
             max_ctx=131072):
    """One coordinate_pipe job with a FRESH drafter (clean n-gram state per config). Sockets are
    reused across jobs — coordinate_pipe drains in-flight + opens each job with `reset`, which clears
    every stage's KV, so back-to-back jobs on the same ring are clean. make_drafter adds the EAGLE
    hybrid when M25_EAGLE=1. `max_ctx` is the launcher-negotiated ring limit (--max-ctx, H1) — the
    old hardcoded 131072 silently overran a 40960-KV ring."""
    drafter = make_drafter(ngram_n)
    return coordinate_pipe(pipe, tok, messages, k, max_new, timeout, d, ret_sock=ret,
                           local_draft=drafter, tools=tools, prefill_chunk=prefill_chunk, max_ctx=max_ctx)


def _validate(pipe, ret, tok, K, depth, ngram_n, prefill_chunk, timeout, longctx_path):
    """FULL usability pass on ONE warm ring (jobs reuse the socket like the sweep). Exercises every
    deploy-ready capability end-to-end over libp2p and prints a PASS/FAIL per capability. Receipts are
    proven on every job when the ring was launched with SHARD_RECEIPTS=1."""
    WEATHER = [{"type": "function", "function": {"name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "city name"}},
                               "required": ["city"]}}}]

    print("\n[validate] === FULL USABILITY PASS (warm, libp2p) ===", flush=True)

    # 1) TOOL CALLING — model must emit a structured get_weather call
    m = [{"role": "user", "content": "Use the get_weather tool to check the weather in Paris."}]
    r = _run_job(pipe, ret, tok, m, K, 256, timeout, depth, ngram_n, prefill_chunk, tools=WEATHER)
    p = parse_completion(r["text"]); tc = p["tool_calls"]
    print(f"[validate] 1.TOOLS      {'PASS' if tc else 'FAIL'}  tool_calls={json.dumps(tc, ensure_ascii=False)[:220]}  "
          f"receipts_ok={r.get('receipts_ok')}  {r['tok_s']:.1f}tok/s", flush=True)

    # 2) EXTENDED CONVO — a real ~9-turn back-and-forth; final turn must RECALL a fact stated in turn 1
    #    (tests render_ids threading a long history + cross-turn recall, the "long convo" usability dimension)
    m = [{"role": "user", "content": "Hey, I'm setting up a decentralized inference swarm. My node ID is SWARM-NODE-4417 and I'm running 5 RTX 5090s scattered across Europe."},
         {"role": "assistant", "content": "Nice — 5x5090 scattered across Europe is a solid ring. What model are you serving on it?"},
         {"role": "user", "content": "MiniMax-M2.5, sharded across the nodes over libp2p. I'm getting about 20 tokens per second warm."},
         {"role": "assistant", "content": "That's a healthy warm number for a 5-stage pipeline over WAN. Are you using speculative decoding to hide the per-hop latency?"},
         {"role": "user", "content": "Yeah, n-gram drafting — works great on copy and retrieval tasks. I also need tool calling and long context to work."},
         {"role": "assistant", "content": "Both are supported: the coordinator threads tools through the chat template, and chunked prefill handles long context without OOM."},
         {"role": "user", "content": "Good. I'm also worried about trusting the nodes I don't control."},
         {"role": "assistant", "content": "Each node signs a per-stage receipt with its own key and the coordinator verifies full layer coverage, so no node is paid without proving its block."},
         {"role": "user", "content": "Perfect. Now remind me — what was the node ID I gave you at the very start, and how many GPUs did I say I'm running?"}]
    r = _run_job(pipe, ret, tok, m, K, 96, timeout, depth, ngram_n, prefill_chunk, tools=None)
    p = parse_completion(r["text"]); ans = (p["content"] or "").strip()
    recall = ("SWARM-NODE-4417" in ans) and ("5" in ans)
    print(f"[validate] 2.CONVO(9-turn) {'PASS (recalled turn-1 facts)' if recall else 'PARTIAL/FAIL'}  "
          f"answer={ans[:200]!r}  receipts_ok={r.get('receipts_ok')}", flush=True)

    # 3) LONG CONTEXT — needle retrieval far past the old 8192 RoPE cap (proves the rope fix + chunked prefill)
    lc = open(longctx_path).read()
    m = [{"role": "user", "content": lc}]
    r = _run_job(pipe, ret, tok, m, K, 96, timeout, depth, ngram_n, prefill_chunk, tools=None)  # M2.5 reasons before answering; 24 only covered the restate (false FAIL on 2026-06-28)
    p = parse_completion(r["text"]); ans = (p["content"] or r["text"]).strip()
    hit = "ZX-PAYLOAD-7731" in r["text"]   # needle anywhere in the output (model surfaces it via reasoning, then answers)
    print(f"[validate] 3.LONG-CTX   {'PASS (needle found)' if hit else 'FAIL'}  prompt_tokens={r['prompt_tokens']}  "
          f"prefill={r['prefill_s']:.1f}s  answer={ans[:80]!r}  receipts_ok={r.get('receipts_ok')}", flush=True)

    print("[validate] === END ===", flush=True)


def coord(head_ep, tail_ep, prompt, K, max_new, depth, ngram_n, timeout, sweep=None, sweep_depth=None, prefill_chunk=512, validate=False, max_ctx=131072):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S.DIR, trust_remote_code=True)
    hh, hp = head_ep.rsplit(":", 1); th, tp = tail_ep.rsplit(":", 1)
    pipe = socket.create_connection((hh, int(hp)), timeout=timeout); pipe.setsockopt(*NODELAY)
    ret = socket.create_connection((th, int(tp)), timeout=timeout); ret.setsockopt(*NODELAY); ret.settimeout(timeout)
    send_msg(ret, {"op": "hello_return"})                       # identify the return channel to the tail
    recv_data(ret)                                              # wait ret_ok: tail confirmed ret before any reset flows
    messages = [{"role": "user", "content": prompt}]

    if validate:                                               # full usability pass (tools+multi-turn+long-ctx+receipts)
        _validate(pipe, ret, tok, K, depth, ngram_n, prefill_chunk, timeout, "/root/longctx_prompt.txt")
        return

    if sweep or sweep_depth:                                    # K/depth throughput sweep -> tok/s table
        Ks = sweep or [K]; Ds = sweep_depth or [depth]
        print(f"[coord] SWEEP K={Ks} depth={Ds} ngram={ngram_n} -> head {head_ep}, ret {tail_ep}", flush=True)
        rows = []
        for d in Ds:
            for k in Ks:
                row = {"K": k, "depth": d, "tok_s": 0.0, "g": 0.0, "accept": 0.0,
                       "prefill_s": 0.0, "ntok": 0, "h_kb": (k + 1) * S.H * 2 / 1024, "ok": False, "text": ""}
                try:
                    r = _run_job(pipe, ret, tok, messages, k, max_new, timeout, d, ngram_n, prefill_chunk, max_ctx=max_ctx)
                    row.update(tok_s=r["tok_s"], g=r["toks_per_traversal"], accept=r["mean_accept"] / max(k, 1),
                               prefill_s=r["prefill_s"], ntok=r["n_tokens"], ok=r.get("ok", False), text=r.get("text", ""))
                except Exception as e:
                    row["err"] = f"{type(e).__name__}: {e}"
                rows.append(row)
                print(f"[sweep] K={k:>2} depth={d}: {row['tok_s']:>6.2f} tok/s  g={row['g']:.2f}  "
                      f"accept={row['accept'] * 100:.0f}%  ({'ok' if row['ok'] else row.get('err', 'FAIL')})", flush=True)
        table, best = _sweep_summary(rows)
        print(table, flush=True)
        if best:
            print("\n[sweep] best output:\n" + (parse_completion(best["text"])["content"] or best["text"])[:800], flush=True)
        return

    print(f"[coord] pipelined (K={K} depth={depth} ngram={ngram_n}) -> head {head_ep}, ret {tail_ep}", flush=True)
    r = _run_job(pipe, ret, tok, messages, K, max_new, timeout, depth, ngram_n, prefill_chunk, max_ctx=max_ctx)
    if r.get("ok"):
        parsed = parse_completion(r["text"])
        print(f"\n[coord] {r['n_tokens']}tok  {r['tok_s']:.2f} tok/s  g={r['toks_per_traversal']:.2f}  "
              f"mean_accept={r['mean_accept']:.2f}/{K}  prefill={r['prefill_s']:.2f}s  depth={depth}", flush=True)
        if r.get("decode_s"):                                # where the decode wall went (serial-path profile)
            other = r["decode_s"] - r.get("draft_s", 0) - r.get("ring_wait_s", 0)
            print(f"[coord] decode {r['decode_s']:.1f}s = ring-wait {r.get('ring_wait_s', 0):.1f}s "
                  f"+ draft {r.get('draft_s', 0):.1f}s + coord-other {other:.1f}s", flush=True)
        if r.get("transport_s") is not None:                 # stage-timing split (stages ran M25_STAGE_TIMING=1)
            tv = r["traversal_s"]
            print(f"[coord] traversal {tv:.1f}s = transport {r['transport_s']:.1f}s "
                  f"({100 * r['transport_s'] / max(tv, 1e-9):.0f}%) + stage-span {r['stage_s']:.1f}s "
                  f"(compute {r['stage_compute_s']:.1f}s)  per-stage ms[span,comp]: "
                  + " ".join(f"s{k}={v}" for k, v in sorted(r["per_stage_ms"].items())), flush=True)
        if parsed["reasoning_content"]:
            print("[coord] THINK:\n" + parsed["reasoning_content"][:600], flush=True)
        print("[coord] OUTPUT:\n" + (parsed["content"] or "")[:1200], flush=True)
        if parsed["tool_calls"]:
            print("[coord] TOOL_CALLS: " + json.dumps(parsed["tool_calls"], ensure_ascii=False)[:800], flush=True)
        if r.get("receipts"):
            print(f"[coord] === PROVE: {len(r['receipts'])} signed per-stage receipts ===", flush=True)
            print(f"[coord] PROVE verdict: {'ALL receipts valid + full layer coverage' if r.get('receipts_ok') else 'FAILED'}", flush=True)
            dump = os.environ.get("SHARD_RECEIPT_DUMP")     # export the signed bodies for the c0mpute settle seam
            if dump:                                        # (strip the unsigned 'stage' tag -> exactly the signed dict)
                bodies = [{k: v for k, v in rr.items() if k != "stage"} for rr in r["receipts"]]
                json.dump({"receipts": bodies, "receipts_ok": r.get("receipts_ok"),
                           "n_tokens": r.get("n_tokens")}, open(dump, "w"))
                print(f"[coord] receipts exported -> {dump}", flush=True)
        print("SHA:", hashlib.sha256(r["text"].encode()).hexdigest()[:12], flush=True)
    else:
        print("[coord] FAILED:", r, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="role", required=True)
    ps = sub.add_parser("stage")
    ps.add_argument("--stage", type=int, required=True); ps.add_argument("--nstages", type=int, required=True)
    ps.add_argument("--lo", type=int, required=True); ps.add_argument("--hi", type=int, required=True)
    ps.add_argument("--port", type=int, default=29610); ps.add_argument("--next", default=None)
    ps.add_argument("--timeout", type=int, default=600)
    pc = sub.add_parser("coord")
    pc.add_argument("--head", required=True); pc.add_argument("--tail", required=True)
    pc.add_argument("--prompt", default="Explain a decentralized inference swarm in 3 sentences.")
    pc.add_argument("--prompt-file", default=None); pc.add_argument("--K", type=int, default=8)   # K=8 = the measured sweet spot (2026-06-27 sweep; K=6 left ~2x on the table)
    pc.add_argument("--depth", type=int, default=4); pc.add_argument("--max-new", type=int, default=256)
    pc.add_argument("--ngram-n", type=int, default=3); pc.add_argument("--timeout", type=int, default=600)
    pc.add_argument("--sweep", default=None, help="comma K list, e.g. 4,6,8,12,16 (drafter margin is safe to K<=16)")
    pc.add_argument("--sweep-depth", default=None, help="comma depth list, e.g. 2,4,8 (default: --depth)")
    pc.add_argument("--prefill-chunk", type=int, default=512, help="prefill tokens per ring traversal; under M25_SDPA (default) attn is O(chunk) not O(chunk*ctx), so this is now a TTFT/bandwidth knob, not the OOM guard")
    pc.add_argument("--validate", action="store_true", help="full usability pass: tools + multi-turn + long-ctx (needle) + receipts, one warm ring")
    pc.add_argument("--max-ctx", type=int, default=131072, help="negotiated ring context limit (the launcher passes min(operator ceiling, every stage's KV cap)); jobs clamp max_new against it")
    a = ap.parse_args()

    def _ilist(s): return [int(x) for x in s.split(",") if x.strip()] if s else None

    if os.environ.get("SHARD_TRANSPORT") != "libp2p":   # raw-wire mode: load the PSK before any send/recv (libp2p sidecar self-seals)
        import wire; wire.key_from_env()

    if a.role == "stage":
        serve(a.stage, a.nstages, a.lo, a.hi, a.port, a.next, a.timeout)
    else:
        prompt = open(a.prompt_file).read() if a.prompt_file else a.prompt
        coord(a.head, a.tail, prompt, a.K, a.max_new, a.depth, a.ngram_n, a.timeout,
              sweep=_ilist(a.sweep), sweep_depth=_ilist(a.sweep_depth), prefill_chunk=a.prefill_chunk,
              validate=a.validate, max_ctx=a.max_ctx)
