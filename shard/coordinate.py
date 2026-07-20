"""python -m shard.coordinate — the coordinator entrypoint the node daemon execs (leg 8, node half).

Promotes the gateway's coordinate_pipe driving (phase0/m25_gateway.py) into a socket-drivable
CLI: runs ON the head box beside the head stage (phase0/m25_scatter_pipe.py layout), dials the
LOCAL head engine (--head) and the head sidecar's return -forward to the tail (--tail), and
serves jobs read as JSON lines on stdin. A supervisor waits on the stdout contract:

    SHARD_COORD_OK    {...}   --check preflight passed (engine imports, model dir sane), exit 0
    SHARD_COORD_READY {...}   pipe + return channel connected; jobs accepted on stdin
    SHARD_JOB_TOKEN   {jobId, delta}   one committed decode delta (streamed per ring round)
    SHARD_JOB_DONE    {jobId, ok, response, tokensGenerated, receipts, receiptsOk, nonce}
    SHARD_JOB_FATAL   {jobId?, error}  job failed (process continues) or boot failed (exit 1)

Job line: {"jobId", "swarmId", "nonce", "messages", "maxNew"?, "reasoning"?, "tools"?}.
The settlement nonce threads into coordinate_pipe(job_nonce=...) so every stage signs THE
SERVER'S nonce into its receipt — settlement verifies receipts against exactly that nonce
(fail-closed without it). Jobs run one at a time (the engine coordinator is single-job);
batching windows are the gateway's trick and a follow-up here.

Secrets (SHARD_SWARM_TOKEN, SHARD_NODE_KEY) stay env-only, never argv (world-readable via ps).

  serve:     python -m shard.coordinate --head 127.0.0.1:29610 --tail 127.0.0.1:29612 --dir ~/m25 --receipts
  preflight: python -m shard.coordinate --check --dir ~/m25
"""
import argparse
import json
import os
import socket
import sys
import threading
import time
import traceback

_EMIT_LOCK = threading.Lock()
_hard_exit = os._exit          # injectable for tests; the stall watchdog runs on a THREAD, where
                               # sys.exit only raises in that thread — os._exit is the real kill


def _emit(tag, **fields):
    line = tag + " " + json.dumps(fields)
    with _EMIT_LOCK:           # one atomic write: a watchdog FATAL racing a TOKEN emit must never
        sys.stdout.write(line + "\n")   # splice mid-line (the complete-lines-only NDJSON contract)
        sys.stdout.flush()


def _fatal(msg, **fields):
    _emit("SHARD_JOB_FATAL", error=msg, **fields)
    return 1


def _bootstrap_path():
    """Make the phase0 engine modules importable from a repo checkout; on the flat box layout
    (every file in one dir, sys.path[0] = that dir) they already are."""
    p0 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase0")
    if os.path.isdir(p0) and p0 not in sys.path:
        sys.path.insert(0, p0)


def _apply_env(a):
    """Flags → engine env BEFORE the engine import (m25_stage reads env at module level)."""
    if a.dir:
        os.environ["M25_DIR"] = os.path.abspath(os.path.expanduser(a.dir))
    os.environ.setdefault("SHARD_TRANSPORT", "libp2p")
    if a.receipts:
        os.environ.setdefault("SHARD_RECEIPTS", "1")


def _hostport(s):
    h, _, p = s.rpartition(":")
    return h or "127.0.0.1", int(p)


NODELAY = (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def _stall_budget(timeout):
    """L3 stall budget (seconds). Unset/empty M25_JOB_STALL_S = AUTO: one full production recv
    timeout + 60s slack — by construction no legitimate single wait (reset ack, a prefill chunk on
    a thin uplink, a decode heartbeat) outlasts one recv timeout without producing a progress tick,
    so auto never false-kills a slow-but-moving job. 0 disables."""
    raw = os.environ.get("M25_JOB_STALL_S")
    try:
        v = float(raw) if raw not in (None, "") else float(timeout) + 60.0
    except ValueError:
        v = float(timeout) + 60.0
    return v if v > 0 else None


class _StallWatchdog:
    """P0-#5 L3 backstop: a RUNNING job that makes no observable progress (no reply received, no
    commit, no redial) for the stall budget gets a loud SHARD_JOB_FATAL and a hard process exit —
    the daemon restarts the coordinator and fail-closed-completes the job. This is the only guard
    for the classes nothing else bounds: a drafter wedged inside torch/CUDA, a send stuck against
    a full buffer. Never a silent freeze."""

    def __init__(self, stall_s, emit):
        self.stall_s = stall_s
        self.emit = emit
        self._last = time.monotonic()
        self._job = None
        if stall_s:
            threading.Thread(target=self._run, daemon=True, name="job-stall-watchdog").start()

    def arm(self, job_id):
        self._last = time.monotonic()
        self._job = job_id

    def tick(self):
        self._last = time.monotonic()

    def disarm(self):
        self._job = None

    def _run(self):
        while True:
            time.sleep(min(max(self.stall_s / 4.0, 0.05), 2.0))
            job = self._job
            if job is not None and time.monotonic() - self._last > self.stall_s:
                # The kill must be UNCONDITIONAL — this is the last line of defense. The FATAL emit
                # is best-effort: a bounded lock wait (the main thread may be wedged INSIDE a locked
                # stdout write — the very stuck-write class this backstop covers), stderr fallback,
                # and every exception swallowed so a dead stdout can never block or kill the exit.
                err = (f"stall-watchdog: no progress in {self.stall_s:.0f}s "
                       f"(ring or drafter wedged) — exiting so the daemon restarts us")
                try:
                    if self.emit is not _emit:               # injected emit (tests/collectors): no stdout lock
                        self.emit("SHARD_JOB_FATAL", jobId=job, error=err)
                    else:
                        line = "SHARD_JOB_FATAL " + json.dumps({"jobId": job, "error": err}) + "\n"
                        if _EMIT_LOCK.acquire(timeout=5.0):
                            try:
                                sys.stdout.write(line)
                                sys.stdout.flush()
                            finally:
                                _EMIT_LOCK.release()
                        else:                                # lock held by a wedged writer: bypass it
                            os.write(2, line.encode())
                except Exception:
                    pass
                finally:
                    _hard_exit(1)


def connect_ring(MP, head, tail, timeout, retry_s=300):
    """Dial the head engine (pipe) + the return tunnel to the tail (ret), with retries — the
    daemon starts us on head-stage READY while other stages may still be pulling weights, so
    the return leg can take minutes to come up. hello_return classifies the tail-side stream
    (m25_pipe._tail_accept acks immediately); SHARD_SWARM_TOKEN (env-only) rides both greetings
    when set, exactly like the gateway."""
    token = os.environ.get("SHARD_SWARM_TOKEN")
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
                MP.send_msg(pipe, {"op": "hello_pred", "token": token})
                MP.send_msg(ret, {"op": "hello_return", "token": token})
            else:
                MP.send_msg(ret, {"op": "hello_return"})
            MP.recv_data(ret)                       # tail acks the return channel (ret_ok)
            return pipe, ret
        except Exception as e:                      # noqa: BLE001 — every dial fault = retry until deadline
            last = e
            for s in (pipe, ret):
                if s is not None:
                    try: s.close()
                    except OSError: pass
            time.sleep(min(5.0, max(0.5, deadline - time.time())))
    raise ConnectionError(f"ring not reachable after {retry_s}s: {type(last).__name__}: {last}")


def run_job(MP, tok, eos_set, chans, a, job, emit=_emit, watchdog=None, redial=None):
    """One job through coordinate_pipe, with the P0-#5 mitigation ladder: attempt 1 runs as
    configured; if it EDGE-faults while EAGLE was armed (the residential-tail wedge class), ONE
    degraded retry on a FRESH ring dial — EAGLE off process-wide (sticky: this ring just proved
    EAGLE-hostile; a daemon restart re-arms from env) so the retry's reset stamps eagle:0 and the
    stages silence aux, resume_ids = the committed partial, the SAME nonce and the SAME delta
    state so the token stream continues with no dup/gap. The re-dial is mandatory: a plain reset
    on the old sockets would eat a late in-flight reply as its ack (the tail only goes stale on a
    fresh hello_return). A retry that also fails returns its resumable-failure dict — serve_jobs
    bails for a clean daemon restart. Deltas are capped at the first EOS so joined deltas == the
    final response text."""
    job_id = job["jobId"]
    max_new = max(1, min(int(job.get("maxNew") or 512), 4096))
    state = {"text": "", "eos_at": None}
    tick = watchdog.tick if watchdog is not None else None

    def on_commit(out, _dt):
        if tick:
            tick()
        # cap at the first EOS: a round can commit [tok, eos, tok2] but the final text ends at eos
        if state["eos_at"] is None:
            for i in range(len(out)):
                if out[i] in eos_set:
                    state["eos_at"] = i
                    break
        ids = out[: state["eos_at"]] if state["eos_at"] is not None else out
        text = tok.decode(ids, skip_special_tokens=True)
        if len(text) > len(state["text"]):
            delta = text[len(state["text"]):]
            state["text"] = text
            emit("SHARD_JOB_TOKEN", jobId=job_id, delta=delta)

    def _attempt(resume_ids=None):
        return MP.coordinate_pipe(
            chans["pipe"], tok, job["messages"], a.K, max_new, a.timeout, a.depth,
            ret_sock=chans["ret"], local_draft=MP.make_drafter(a.ngram_n), tools=job.get("tools"),
            prefill_chunk=a.prefill_chunk, max_ctx=a.max_ctx, on_commit=on_commit,
            swarm_id=job.get("swarmId") or "swarm", job_id=job_id,
            reasoning=bool(job.get("reasoning", True)),
            job_nonce=job.get("nonce") or None,
            resume_ids=resume_ids, resumable=True, on_progress=tick)

    eagle_arm = bool(MP.S.M25_EAGLE)
    r = _attempt()
    if r.get("ok") or not r.get("resumable") or not eagle_arm or redial is None:
        return r
    committed = list(r.get("output_ids") or [])
    emit("SHARD_JOB_RETRY", jobId=job_id, reason=str(r.get("error", ""))[:200], committed=len(committed))
    print(f"[coordinate] EAGLE-implicated edge fault on job {job_id} -> degraded retry "
          f"(plain decode, eagle:0 on the wire, {len(committed)} tokens resumed)", file=sys.stderr, flush=True)
    MP.S.M25_EAGLE = False
    if getattr(MP.S, "M25_TREE", False):     # tree mode implies EAGLE: both off or the tree
        MP.S.M25_TREE = False                # coordinator would reject the n-gram drafter
    for s in (chans["pipe"], chans["ret"]):
        try:
            s.close()
        except OSError:
            pass
    chans["pipe"], chans["ret"] = redial()   # fresh hello_return = the tail's stale gate arms
    if tick:
        tick()                               # the successful redial is progress
    r2 = _attempt(resume_ids=committed)
    r2["degraded_retry"] = True
    return r2


def serve_jobs(MP, tok, pipe, ret, a, lines, emit=_emit, redial=None):
    """The stdin job loop, factored for tests: `lines` is any iterator of JSON job lines. `chans`
    holds the ring channels so a mid-job degraded re-dial carries into the next job. The stall
    watchdog (L3) is armed per job and ticked by every reply recv / commit / redial."""
    eos = tok.eos_token_id
    eos_set = set(eos) if isinstance(eos, (list, tuple)) else {eos}
    chans = {"pipe": pipe, "ret": ret}
    watchdog = _StallWatchdog(_stall_budget(a.timeout), emit)
    if redial is None and getattr(a, "head", None) and getattr(a, "tail", None):
        def redial():
            # mid-job re-dial: cap the retry window well under the stall budget — a dead ring
            # should fail the job to the daemon in about a minute, not park it for 5
            return connect_ring(MP, a.head, a.tail, a.timeout,
                                retry_s=min(getattr(a, "connect_retry", 60) or 60, 60))
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
            job_id = job["jobId"]
        except (ValueError, KeyError) as e:
            emit("SHARD_JOB_FATAL", error=f"unparseable job line: {e}")
            continue
        watchdog.arm(job_id)
        try:
            r = run_job(MP, tok, eos_set, chans, a, job, emit=emit, watchdog=watchdog, redial=redial)
            if r.get("ok"):
                emit("SHARD_JOB_DONE", jobId=job_id, ok=True,
                     response=r.get("text", ""), tokensGenerated=int(r.get("n_tokens", 0)),
                     receipts=r.get("receipts") or [], receiptsOk=r.get("receipts_ok"),
                     nonce=job.get("nonce"),
                     degraded=bool(r.get("eagle_degraded") or r.get("degraded_retry")))
            else:
                # an edge fault that survived the mitigation ladder (or the plain path's): the
                # channels are poisoned — report the job dead and bail so the daemon restarts us
                # clean (the control plane fail-closed-completes the job)
                emit("SHARD_JOB_FATAL", jobId=job_id,
                     error=str(r.get("error") or "ring edge fault")[:300])
                return 1
        except Exception as e:                      # noqa: BLE001 — a job fault must not kill the loop
            traceback.print_exc(file=sys.stderr)
            emit("SHARD_JOB_FATAL", jobId=job_id, error=f"{type(e).__name__}: {e}")
            # socket faults poison the ring channels — bail so the daemon restarts us clean
            if isinstance(e, MP.EDGE_ERRORS) or isinstance(e, MP.TransportError):
                return 1
        finally:
            watchdog.disarm()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m shard.coordinate",
        description="drive generation jobs over a formed ring (the node daemon's serving exec target)")
    ap.add_argument("--head", default="127.0.0.1:29610", help="head engine host:port (local pipe)")
    ap.add_argument("--tail", default="127.0.0.1:29612", help="return-tunnel host:port (head sidecar -forward to the tail)")
    ap.add_argument("--dir", default=None, help="model dir (wins over M25_DIR)")
    ap.add_argument("--K", type=int, default=8, help="draft chunk length per round")
    ap.add_argument("--depth", type=int, default=4, help="pipelined verify chunks in flight")
    ap.add_argument("--ngram-n", type=int, default=3, help="n-gram drafter anchor length")
    ap.add_argument("--max-ctx", type=int, default=131072, help="context cap")
    ap.add_argument("--prefill-chunk", type=int, default=4096, dest="prefill_chunk",
                    help="prompt prefill chunk length")
    ap.add_argument("--timeout", type=int, default=600, help="per-frame recv deadline, seconds")
    ap.add_argument("--connect-retry", type=int, default=300,
                    help="seconds to keep retrying the ring dial at boot (stages may still be pulling)")
    ap.add_argument("--receipts", action="store_true", help="sweep + surface signed receipts (SHARD_RECEIPTS=1)")
    ap.add_argument("--check", action="store_true",
                    help="preflight only: import the engine against the model dir, print SHARD_COORD_OK, exit")
    a = ap.parse_args(argv)

    _apply_env(a)
    _bootstrap_path()

    mdir = os.environ.get("M25_DIR")
    if not mdir:
        return _fatal("no model dir: pass --dir or set M25_DIR")
    missing = [f for f in ("config.json", "model.safetensors.index.json")
               if not os.path.isfile(os.path.join(mdir, f))]
    if missing:
        return _fatal(f"model dir is missing {missing}", dir=mdir)

    try:
        import m25_pipe as MP             # heavy: torch + m25_stage's module-level M25_DIR init
    except ImportError as e:
        return _fatal(f"engine import failed: {e}",
                      hint="run from a shard checkout (phase0/ beside shard/) or the flat box layout")

    if a.check:
        _emit("SHARD_COORD_OK", dir=mdir, transport=os.environ["SHARD_TRANSPORT"],
              eagle=bool(os.environ.get("M25_EAGLE")))
        return 0

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)

    try:
        pipe, ret = connect_ring(MP, a.head, a.tail, a.timeout, retry_s=a.connect_retry)
    except Exception as e:                # noqa: BLE001
        return _fatal(f"ring connect failed: {type(e).__name__}: {e}", head=a.head, tail=a.tail)

    _emit("SHARD_COORD_READY", head=a.head, tail=a.tail,
          eagle=bool(os.environ.get("M25_EAGLE")), receipts=bool(os.environ.get("SHARD_RECEIPTS")))
    try:
        return serve_jobs(MP, tok, pipe, ret, a, sys.stdin)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
