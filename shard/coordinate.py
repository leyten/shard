"""python -m shard.coordinate — the coordinator entrypoint the node daemon execs (leg 8, node half).

Promotes the gateway's coordinate_pipe driving (phase0/m25_gateway.py) into a socket-drivable
CLI: runs ON the head box beside the head stage (phase0/m25_scatter_pipe.py layout), dials the
LOCAL head engine (--head) and the head sidecar's return -forward to the tail (--tail), and
serves jobs read as JSON lines on stdin. A supervisor waits on the stdout contract:

    SHARD_COORD_OK    {...}   --check preflight passed (engine imports, model dir sane), exit 0
    SHARD_COORD_READY {...}   pipe + return channel connected; jobs accepted on stdin
    SHARD_JOB_START   {jobId, maxNew}  the job line was READ off stdin and handed to the ring
    SHARD_JOB_TOKEN   {jobId, delta}   one committed VISIBLE-content delta (streamed per ring round;
                                       with reasoning on, the think block is withheld — an API user
                                       must never receive chain-of-thought as message content)
    SHARD_JOB_DONE    {jobId, ok, response, reasoning?, tokensGenerated, receipts, receiptsOk, nonce}
                                       response = visible content only; reasoning = the think text
                                       (present only when a think block was produced and split)
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

from shard.receipt import wire_receipt   # strip the post-sign `stage` debug tag before settlement

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
    """Make the engine modules importable from a repo checkout; on the flat box layout (every file
    in one dir, sys.path[0] = that dir) they already are.

    Engines live under engines/<model>/ and shared tooling under phase0/, so a checkout needs both;
    the box needs neither, which is why every candidate is probed rather than assumed."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands = [os.path.join(root, "phase0")]
    eng = os.path.join(root, "engines")
    if os.path.isdir(eng):
        cands = [os.path.join(eng, d) for d in sorted(os.listdir(eng))] + cands
    for c in cands:
        if os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)


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


def _firstack_budget(stall_s):
    """Budget (seconds) for a job's FIRST sign of ring life — the reset ack — kept separate from the
    steady-state stall budget.

    They are different failure classes. Mid-job, one legitimate wait can be a whole prefill chunk on
    a thin residential uplink, so the stall budget is deliberately one full recv timeout wide (660s
    at the daemon's defaults). The reset has no such case: it is one tiny control frame traversing
    an already-WARM ring once (~160ms round-ring on the measured 6-hop EU ring). Bounding both by
    the same number is what made the 2026-07-28 stranger-serve ring look like a coordinator that
    "accepts jobs it never executes": it HAD sent the reset and was parked in recv_data for 11
    minutes with nothing on stdout, so the operator saw an accepted job, silence, and 0% GPU.

    This budget covers EXACTLY that leg, no further: coordinate_pipe ticks progress the moment the
    reset is acked, so the prefill that follows is already back on the wide budget and a slow first
    chunk can never be false-killed. The un-ticked window is that RTT plus whatever runs before it
    on a job — render_ids over the prompt, and on a coordinator's FIRST job the drafter build
    (make_drafter is evaluated as an argument, so an EAGLE head load lands inside it).

    M25_JOB_FIRSTACK_S overrides; 0 disables. stall_s None means the operator turned the watchdog
    OFF with M25_JOB_STALL_S=0, and that hatch has to keep meaning off — a second budget that
    silently re-armed it would be a nasty surprise for whoever is debugging with it."""
    if not stall_s:
        return None
    raw = os.environ.get("M25_JOB_FIRSTACK_S")
    try:
        v = float(raw) if raw not in (None, "") else 90.0
    except ValueError:
        v = 90.0
    return min(v, stall_s) if v > 0 else None


class _StallWatchdog:
    """P0-#5 L3 backstop: a RUNNING job that makes no observable progress (no reply received, no
    commit, no redial) for the stall budget gets a loud SHARD_JOB_FATAL and a hard process exit —
    the daemon restarts the coordinator and fail-closed-completes the job. This is the only guard
    for the classes nothing else bounds: a drafter wedged inside torch/CUDA, a send stuck against
    a full buffer. Never a silent freeze."""

    def __init__(self, stall_s, emit, first_s=None):
        self.stall_s = stall_s
        self.first_s = first_s
        self.emit = emit
        self._last = time.monotonic()
        self._job = None
        self._moved = False              # has the ring answered anything at all for this job?
        budgets = [b for b in (stall_s, first_s) if b]
        if budgets:
            self._poll = min(max(min(budgets) / 4.0, 0.05), 2.0)
            threading.Thread(target=self._run, daemon=True, name="job-stall-watchdog").start()

    def _budget(self):
        """The pre-first-reply wait and the mid-job wait are different failure classes (see
        _firstack_budget) — this job is in exactly one of them."""
        if self._moved or not self.first_s:
            return self.stall_s
        return self.first_s

    def arm(self, job_id):
        self._last = time.monotonic()
        self._moved = False
        self._job = job_id

    def tick(self):
        self._last = time.monotonic()
        self._moved = True

    def disarm(self):
        self._job = None

    def _run(self):
        while True:
            time.sleep(self._poll)
            job = self._job
            budget = self._budget()
            if job is not None and budget and time.monotonic() - self._last > budget:
                # The kill must be UNCONDITIONAL — this is the last line of defense. The FATAL emit
                # is best-effort: a bounded lock wait (the main thread may be wedged INSIDE a locked
                # stdout write — the very stuck-write class this backstop covers), stderr fallback,
                # and every exception swallowed so a dead stdout can never block or kill the exit.
                # Name the PHASE: "the ring never answered the job at all" and "the ring stopped
                # answering mid-job" have different suspects, and the operator reading the daemon
                # log is the one who has to tell them apart.
                err = (f"stall-watchdog: no ring reply in {budget:.0f}s — the ring never acked this "
                       f"job's reset (frames sent, nothing came back on the return channel)"
                       if not self._moved else
                       f"stall-watchdog: no progress in {budget:.0f}s (ring or drafter wedged)")
                err += " — exiting so the daemon restarts us"
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
    final VISIBLE response text.

    Reasoning split: with reasoning on, M2.5's template hardwires "<think>\n" into the generation
    prompt, so the raw decode stream is `...chain-of-thought...</think>\n\n visible-content`. The
    daemon path was relaying that stream verbatim — every OpenAI-endpoint user got the model's
    thinking as message content (and a short-budget job burned its whole budget before the answer).
    The marker comes from the per-model tools seam (`MP._TOOLS.THINK_END`); models without one, and
    reasoning:false jobs (whose completion contains no think block), stream unchanged."""
    job_id = job["jobId"]
    max_new = max(1, min(int(job.get("maxNew") or 512), 4096))
    state = {"text": "", "eos_at": None, "marker_end": None, "vis_sent": 0}
    tick = watchdog.tick if watchdog is not None else None
    think_end = getattr(getattr(MP, "_TOOLS", None), "THINK_END", None)
    split = bool(job.get("reasoning", True)) and bool(think_end)

    def _visible(text):
        """The user-facing slice of the raw decode stream (think block + boundary newlines gone)."""
        if not split:
            return text
        if state["marker_end"] is None:
            j = text.find(think_end)
            if j < 0:
                return ""                        # still thinking — nothing visible yet
            state["marker_end"] = j + len(think_end)
        k = state["marker_end"]
        while k < len(text) and text[k] == "\n":  # boundary "\n\n" may straddle commits
            k += 1
        return text[k:]

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
            state["text"] = text
        visible = _visible(state["text"])
        if len(visible) > state["vis_sent"]:
            delta = visible[state["vis_sent"]:]
            state["vis_sent"] = len(visible)
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

    def _finish(r):
        """Split the FINAL text the same way the stream was split, so DONE.response == joined
        deltas. A completion that never closed its think block (budget exhausted mid-reasoning)
        honestly returns response '' with the whole text under `reasoning`."""
        if split and r.get("ok"):
            raw = r.get("text", "")
            j = raw.find(think_end)
            r["reasoning_text"] = raw[:j] if j >= 0 else raw
            r["text"] = _visible(raw)
        return r

    eagle_arm = MP.eagle_armed()
    r = _attempt()
    if r.get("ok") or not r.get("resumable") or not eagle_arm or redial is None:
        return _finish(r)
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
    return _finish(r2)


def serve_jobs(MP, tok, pipe, ret, a, lines, emit=_emit, redial=None):
    """The stdin job loop, factored for tests: `lines` is any iterator of JSON job lines. `chans`
    holds the ring channels so a mid-job degraded re-dial carries into the next job. The stall
    watchdog (L3) is armed per job and ticked by every reply recv / commit / redial."""
    eos = tok.eos_token_id
    eos_set = set(eos) if isinstance(eos, (list, tuple)) else {eos}
    chans = {"pipe": pipe, "ret": ret}
    stall_s = _stall_budget(a.timeout)
    watchdog = _StallWatchdog(stall_s, emit, first_s=_firstack_budget(stall_s))
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
        # The job line was READ off stdin and is going to the ring NOW. Without this, the whole
        # window between the daemon's "swarm:job accepted" and a DONE/FATAL is silent, and a
        # coordinator parked in recv_data is indistinguishable from one that never read stdin —
        # the misread that cost the 2026-07-28 stranger-serve ring its root cause.
        emit("SHARD_JOB_START", jobId=job_id, maxNew=job.get("maxNew"))
        try:
            r = run_job(MP, tok, eos_set, chans, a, job, emit=emit, watchdog=watchdog, redial=redial)
            if r.get("ok"):
                # Surface the PERF fields the engine computes (they were being dropped — the
                # daemon serve path was unmeasurable: no tok/s, g, or transport split). Additive
                # line; consumers that don't know it ignore it (the daemon logs unknown contract
                # lines). Only the keys the run actually produced are emitted.
                metrics = {k: r[k] for k in (
                    "tok_s", "mean_accept", "toks_per_traversal", "prefill_s", "decode_s",
                    "draft_s", "ring_wait_s", "n_tokens", "prompt_tokens", "graph_arm",
                    "traversal_s", "stage_s", "stage_compute_s", "transport_s", "per_stage_ms",
                ) if k in r}
                if metrics:
                    emit("SHARD_JOB_METRICS", jobId=job_id, **metrics)
                emit("SHARD_JOB_DONE", jobId=job_id, ok=True,
                     response=r.get("text", ""), tokensGenerated=int(r.get("n_tokens", 0)),
                     # additive field: only present when a think block was split off (deployed
                     # daemons ignore unknown keys — the proven contract-evolution path)
                     **({"reasoning": r["reasoning_text"]} if r.get("reasoning_text") is not None else {}),
                     # Hand settlement the SIGNED body: each stage tags its receipt with a `stage`
                     # debug label after signing, and shard.verify signs-all-but-sig (stays strict),
                     # so an un-stripped tag is InvalidSignature at the one call site that pays.
                     receipts=[wire_receipt(rr) for rr in (r.get("receipts") or [])],
                     receiptsOk=r.get("receipts_ok"),
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
              eagle=MP.eagle_armed())
        return 0

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)

    try:
        pipe, ret = connect_ring(MP, a.head, a.tail, a.timeout, retry_s=a.connect_retry)
    except Exception as e:                # noqa: BLE001
        return _fatal(f"ring connect failed: {type(e).__name__}: {e}", head=a.head, tail=a.tail)

    _emit("SHARD_COORD_READY", head=a.head, tail=a.tail,
          eagle=MP.eagle_armed(), receipts=bool(os.environ.get("SHARD_RECEIPTS")))
    try:
        return serve_jobs(MP, tok, pipe, ret, a, sys.stdin)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
