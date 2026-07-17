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
import time
import traceback


def _emit(tag, **fields):
    print(tag + " " + json.dumps(fields), flush=True)


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


def run_job(MP, tok, eos_set, pipe, ret, a, job, emit=_emit):
    """One job through coordinate_pipe: stream SHARD_JOB_TOKEN deltas per commit, return the
    result dict. The delta stream is capped at the first EOS so joined deltas == the final
    response text (coordinate_pipe trims post-EOS commits from the final text the same way)."""
    job_id = job["jobId"]
    max_new = max(1, min(int(job.get("maxNew") or 512), 4096))
    state = {"text": "", "eos_at": None}

    def on_commit(out, _dt):
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

    drafter = MP.make_drafter(a.ngram_n)
    r = MP.coordinate_pipe(
        pipe, tok, job["messages"], a.K, max_new, a.timeout, a.depth,
        ret_sock=ret, local_draft=drafter, tools=job.get("tools"),
        prefill_chunk=4096, max_ctx=a.max_ctx, on_commit=on_commit,
        swarm_id=job.get("swarmId") or "swarm", job_id=job_id,
        reasoning=bool(job.get("reasoning", True)),
        job_nonce=job.get("nonce") or None)
    return r


def serve_jobs(MP, tok, pipe, ret, a, lines, emit=_emit):
    """The stdin job loop, factored for tests: `lines` is any iterator of JSON job lines."""
    eos = tok.eos_token_id
    eos_set = set(eos) if isinstance(eos, (list, tuple)) else {eos}
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
        try:
            r = run_job(MP, tok, eos_set, pipe, ret, a, job, emit=emit)
            emit("SHARD_JOB_DONE", jobId=job_id, ok=bool(r.get("ok")),
                 response=r.get("text", ""), tokensGenerated=int(r.get("n_tokens", 0)),
                 receipts=r.get("receipts") or [], receiptsOk=r.get("receipts_ok"),
                 nonce=job.get("nonce"))
        except Exception as e:                      # noqa: BLE001 — a job fault must not kill the loop
            traceback.print_exc(file=sys.stderr)
            emit("SHARD_JOB_FATAL", jobId=job_id, error=f"{type(e).__name__}: {e}")
            # socket faults poison the ring channels — bail so the daemon restarts us clean
            if isinstance(e, MP.EDGE_ERRORS) or isinstance(e, MP.TransportError):
                return 1
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
