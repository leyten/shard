"""shard OpenAI-compatible gateway for MiniMax-M2.5 — the c0mpute integration seam.

Exposes /v1/chat/completions (messages + tools + tool_choice + streaming) and /v1/models over the
scattered libp2p ring. This is the PROGRAMMATIC api c0mpute calls — distinct from gateway.py's shared
public demo terminal.

CONTINUOUS BATCHING is the standard concurrency path: requests queue to a dispatcher; a burst that
arrives within M25_GW_WINDOW_MS rides ONE coordinate_pipe_batch job (per-stream drafting = the full
solo stack: n-gram -> EAGLE per stream; per-stream streaming; per-stream tools/reasoning/max_new).
A lone request takes the tuned solo path (identical to the old behavior). M25_GW_BATCH caps the batch
width and MUST be <= the ring's M25_BATCH (stage KV rows are allocated at launch) — it defaults to
M25_BATCH so an un-batched ring never sees a batched op.

  SHARD_TRANSPORT=libp2p M25_DIR=/root/m25 python m25_gateway.py --head H:P --tail H:P --port 29600
  M25_GATEWAY_MOCK=1 python m25_gateway.py --head x --tail x --port 29600   # local api/shape test, no GPU

Beta notes: decoding is greedy (speculative verify); non-greedy `temperature`/`top_p`/`top_k`
values are REJECTED with a 400 (silently ignoring them misrepresents the sampling the caller asked
for — lossless sampling is a separate engine lever; the tail argmaxes today).
"""
import argparse, json, os, socket, sys, threading, time, itertools
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m25_tools import parse_completion, render_ids, to_openai_message, TOOLCALL_BEGIN, THINK_BEGIN, THINK_END

MOCK = bool(os.environ.get("M25_GATEWAY_MOCK"))
MODEL_ID = os.environ.get("M25_MODEL_ID", "minimax-m2.5")
# default reasoning mode for the gateway: M2.5 hardwires a <think> block, which is novel (0% n-gram
# accept) so it runs at the WAN floor and dominates latency. M25_DEFAULT_REASONING=0 makes the gateway
# answer DIRECTLY by default (fast, for latency-sensitive/high-overlap normal usage); a request can
# override per-call with {"reasoning": true/false} or {"reasoning_effort": "none"|...}. Default ON (quality).
DEFAULT_REASONING = os.environ.get("M25_DEFAULT_REASONING", "1") != "0"
# Per-swarm/epoch token (C2 activation authorization): when the launcher sets SHARD_SWARM_TOKEN,
# every ring connection opens with an explicit identity-bound greeting so the tail/head adopt a
# socket by GREETING, never by silence-inference. Unset = exact legacy wire behavior. The token is
# validated ring-side and must never appear in receipts, replies, or logs.
SWARM_TOKEN = os.environ.get("SHARD_SWARM_TOKEN")
NODELAY = (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
# A stalled streaming client must not pin the single-stream ring: bound each chunk write so a client
# that stops reading is dropped in seconds, not up to the ring timeout (was ~30 min). Env-tunable.
STREAM_WRITE_TIMEOUT = float(os.environ.get("M25_STREAM_WRITE_TIMEOUT", "30"))
_ids = itertools.count(1)
RING_LOCK = threading.Lock()   # one ring job at a time (a job may carry up to M25_GW_BATCH streams)
GW_BATCH = int(os.environ.get("M25_GW_BATCH", os.environ.get("M25_BATCH", "1")))
GW_WINDOW_MS = float(os.environ.get("M25_GW_WINDOW_MS", "40"))
# ---- intake bounds (H5): the stdlib ThreadingHTTPServer spawns a thread per connection with no
# body cap and no read deadline, and every handler then parks on the dispatcher queue — so a flood
# (or one slow-loris) accumulates unbounded threads/bytes while the ring grinds. Bound all three:
# body size (413), header/body read deadline (H.timeout -> socket timeout), in-flight requests (429).
MAX_BODY = int(os.environ.get("M25_GW_MAX_BODY", str(16 << 20)))
REQ_TIMEOUT = float(os.environ.get("M25_GW_REQ_TIMEOUT", "30"))
MAX_INFLIGHT = int(os.environ.get("M25_GW_MAX_INFLIGHT", str(max(GW_BATCH * 4, 16))))
_INFLIGHT = {"n": 0}
_INFLIGHT_LOCK = threading.Lock()
# ---- content routing (per-stream-20 work): like-with-like batching + per-(content, B) K ----------
# The de-lockstep receipts split per-stream speed by CONTENT regime, and the 2026-07-12 K-reference
# pass (receipt perstream-trees-ab-20260712) pinned the K physics: at B<=2 the payload is amortized
# and g dominates, so full K wins (mix-B1 22.5 at K=8 vs 17.1 at K=6); at B>=4 a K=8 frame's dead
# draft slots bind on the payload-priced ring and K=6 lifted EVERY g-bound arm 20-100% (reasoning
# 14->29, mix 14.2->17.9, summarize ->18.2, qa ->21.1) — only tools' g~5 still earns K=8. Route by
# cheap request features (no model call): tools -> "tools"; reasoning-on -> "reasoning";
# reasoning-off + long prompt -> "context"; else "novel". The dispatcher batches LIKE WITH LIKE
# (single-class jobs: one K per job, and a job's round cadence fits its regime), FIFO-fair — the
# queue head always defines the next job's class, so no class starves. K then keys on (class, B).
# Env-tunable; M25_GW_CONTENT_K=0 disables routing (every job rides args.K). NOTE: distinct K
# values mean distinct [*, K+1] frame shapes stage-side -> more CUDA-graph captures per bucket
# (bounded by M25_GRAPH_MAX + the free-VRAM capture guard; excess shapes run eager).
CONTENT_K = os.environ.get("M25_GW_CONTENT_K", "1") != "0"
CTX_CHARS = int(os.environ.get("M25_GW_CTX_CHARS", "6000"))    # prompt chars >= this = "context" class
K_CTX = int(os.environ.get("M25_GW_K_CTX", "6"))               # context AND reasoning at wide B
K_NOVEL = int(os.environ.get("M25_GW_K_NOVEL", "5"))
K_WIDE_B = int(os.environ.get("M25_GW_K_WIDE_B", "3"))         # batch width where dead slots start binding


def _content_class(messages, tools, reasoning):
    if tools:
        return "tools"
    if reasoning:
        return "reasoning"
    n = sum(len(str(m.get("content") or "")) for m in (messages or []))
    return "context" if n >= CTX_CHARS else "novel"


def _class_k(cls, B=1):
    """The batched job's K for (content class, batch width). Solo and narrow batches keep args.K
    (the measured g-dominates regime); wide batches drop to the class K — the dead-slot payload
    argument is B>=4 physics (receipt perstream-trees-ab-20260712). Tools' g~5 earns full K at
    every width."""
    if not CONTENT_K or cls == "tools" or B < K_WIDE_B:
        return A.K
    if cls == "novel":
        return K_NOVEL
    return K_CTX                                       # context + reasoning at wide B


def negotiated_max_ctx(operator_max_ctx: int, stage_kv_caps: list[int]) -> int:
    """H1 single source of truth: the ring's effective context limit is the MIN of the operator's
    ceiling and every stage's KV cap (zero/absent caps are unknown, not infinite — filtered out).
    The launcher computes this once and passes it as --max-ctx; the gateway rejects above it."""
    return min([int(operator_max_ctx)] + [int(c) for c in stage_kv_caps if c and int(c) > 0])


def spec_headroom(k_max: int, tree: bool = False, tree_m: int = 0) -> int:
    """Speculative decode may run the KV up to K (or the tree's M) slots past the committed frontier;
    +16 matches coordinate_pipe's existing safety margin (m25_pipe reserves gen+16)."""
    return (max(int(k_max), int(tree_m)) if tree else int(k_max)) + 16


def _cap_output(ids, max_new, eos_set):
    """OpenAI semantics, enforced gateway-side: stop at the EARLIEST eos (speculative commits can
    land tokens PAST the eos before the coordinator trims — post-eos text must never leak) and treat
    max_tokens as a STRICT cap (a K-token round can overshoot max_new by up to K). Returns
    (capped_ids, finish) with finish "stop" (eos), "length" (cap hit) or None (untouched)."""
    for i, t in enumerate(ids):
        if t in eos_set:
            ids = ids[:i]                              # eos itself is never content
            return (ids[:max_new], "length") if len(ids) > max_new else (ids, "stop")
    if len(ids) >= max_new:
        return ids[:max_new], "length"
    return ids, None


def _enforce_cap(r, max_new):
    """Apply _cap_output to a finished result dict IN PLACE (text/n_tokens/output_ids); returns the
    finish override ("length"/"stop") or None. MOCK results carry no ids — cap in the same chars//4
    currency the mock counts tokens in, so the strict-cap path is testable without a GPU."""
    ids = r.get("output_ids")
    if not MOCK and ids:
        capped, fin = _cap_output(ids, max_new, EOS_SET)
        if len(capped) != len(ids):
            r["output_ids"] = capped
            r["text"] = tok.decode(capped, skip_special_tokens=True)
            r["n_tokens"] = len(capped)
        return fin
    if r.get("n_tokens", 0) > max_new:
        r["text"] = (r.get("text") or "")[:max_new * 4]
        r["n_tokens"] = max_new
        return "length"
    return None


class JobRejected(Exception):
    """A stage refused the job with a structured per-job error (e.g. kv_overflow) — the job is dead
    but the ring is healthy. Deliberately NOT an OSError so neither coordinate_pipe's EDGE_ERRORS
    recovery nor generate()'s reconnect-retry ever eats it: a rejected job must never be re-run.
    Rebound to m25_pipe.JobRejected by _engine_init so gateway excepts match the coordinator's."""


def _admit():
    """Non-blocking in-flight slot: False = the caller must fast-fail 429 instead of parking another
    handler thread on the dispatcher queue."""
    with _INFLIGHT_LOCK:
        if _INFLIGHT["n"] >= MAX_INFLIGHT:
            return False
        _INFLIGHT["n"] += 1
        return True


def _release():
    with _INFLIGHT_LOCK:
        _INFLIGHT["n"] -= 1


class ClientGone(Exception):
    """The HTTP client's socket died or stalled mid-stream (write failed / timed out). NOT a ring
    fault — so generate() must NEVER retry or re-run the generation (the old behaviour: a client
    disconnect surfaced as an OSError, which coordinate_pipe absorbed as a ring EDGE_ERROR, so the
    gateway reconnected and ran the ENTIRE generation a second time). A plain Exception, deliberately
    NOT an OSError, so coordinate_pipe's `except EDGE_ERRORS` lets it propagate untouched."""

A = None
tok = None
EOS_SET = frozenset()          # real eos ids bound by _engine_init; MOCK carries text, not ids
# speculative headroom defaults; _configure() derives the real values from the parsed args (K_MAX
# spans every K content routing may pick — a routed job must never out-run the reserved margin).
K_MAX = 8
TREE = False
TREE_M = 0
HEADROOM = spec_headroom(K_MAX)
coordinate_pipe = None
coordinate_pipe_batch = None
make_drafter = None
make_drafters_b = None
SOCKS = {}


def _configure(args):
    """Bind the parsed args + derived speculative headroom. A.max_ctx IS the negotiated effective
    limit at runtime — the launcher (m25_scatter_pipe) computes negotiated_max_ctx over the stage
    KV caps and passes it as --max-ctx; a hand-launched gateway keeps the operator default."""
    global A, K_MAX, TREE, TREE_M, HEADROOM
    A = args
    K_MAX = max(A.K, K_CTX, K_NOVEL)
    TREE = bool(int(os.environ.get("M25_TREE", "0") or 0))
    TREE_M = int(os.environ.get("M25_TREE_M", "0") or 0)
    HEADROOM = spec_headroom(K_MAX, TREE, TREE_M)


def _engine_init():
    """Import the M2.5 engine + tokenizer and resolve head/tail endpoints (real mode only)."""
    global tok, coordinate_pipe, coordinate_pipe_batch, make_drafter, make_drafters_b, JobRejected, EOS_SET
    import m25_stage as S
    from m25_pipe import (coordinate_pipe as cp, coordinate_pipe_batch as cpb,
                          make_drafter as md, make_drafters_b as mdb)
    from transformers import AutoTokenizer
    coordinate_pipe = cp; coordinate_pipe_batch = cpb; make_drafter = md; make_drafters_b = mdb
    try:                                                # coordinator's class wins so excepts match
        from m25_pipe import JobRejected as _jr
        JobRejected = _jr
    except ImportError:
        pass
    tok = AutoTokenizer.from_pretrained(S.DIR, trust_remote_code=True)
    _eos = tok.eos_token_id
    EOS_SET = set(_eos) if isinstance(_eos, (list, tuple)) else {_eos}   # same int-or-list handling as m25_pipe


def _drop_socks():
    """Close + forget the ring sockets so the next request reconnects fresh. Closing (not just
    clearing the dict) matters on an aborted job: the head sees EOF and the tail sees its return
    channel die, both handled by the churn recovery (PR #26 keeps pred+KV), so a fresh reset re-arms
    the warm ring. The old `SOCKS.clear()`-without-close also leaked the fds until GC."""
    for s in SOCKS.values():
        try: s.close()
        except OSError: pass
    SOCKS.clear()


def _connect(timeout):
    _drop_socks()
    hh, hp = A.head.rsplit(":", 1); th, tp = A.tail.rsplit(":", 1)
    pipe = socket.create_connection((hh, int(hp)), timeout=timeout); pipe.setsockopt(*NODELAY)
    ret = socket.create_connection((th, int(tp)), timeout=timeout); ret.setsockopt(*NODELAY); ret.settimeout(timeout)
    from node_kv import send_msg, recv_msg
    if SWARM_TOKEN:
        # C2 greetings: hello_return (with the swarm token) on the return channel, and hello_pred as
        # the FIRST frame on the head socket — the tail/head classify by these, never by silence.
        send_msg(ret, {"op": "hello_return", "token": SWARM_TOKEN}); recv_msg(ret)   # wait ret_ok
        send_msg(pipe, {"op": "hello_pred", "token": SWARM_TOKEN})
    else:
        send_msg(ret, {"op": "hello_return"}); recv_msg(ret)   # wait ret_ok before any reset flows
    SOCKS.update(pipe=pipe, ret=ret)


def generate(messages, tools, max_new, on_commit, timeout=1800, reasoning=True):
    """Run one chat completion through the ring (or a canned reply in MOCK). Returns the
    coordinate_pipe result dict ({text, n_tokens, prompt_tokens, tok_s, mean_accept, ...})."""
    if MOCK:
        return _mock_generate(messages, tools, max_new, on_commit, reasoning)
    for attempt in (1, 2):
        try:
            if "pipe" not in SOCKS or attempt == 2:
                _connect(timeout)
            drafter = make_drafter(A.ngram_n)
            return coordinate_pipe(SOCKS["pipe"], tok, messages, A.K, max_new, timeout, A.depth,
                                   ret_sock=SOCKS["ret"], local_draft=drafter, tools=tools,
                                   prefill_chunk=4096, max_ctx=A.max_ctx, on_commit=on_commit, reasoning=reasoning)
        except ClientGone:
            _drop_socks()   # the client died mid-decode -> stale in-flight replies on the ring; drop the
            raise           # (now desynced) sockets and abort. NEVER retry: re-running wastes a full ring pass
        except JobRejected:
            _drop_socks()   # the stage refused THIS job (kv_overflow etc): the ring is fine but the
            raise           # request is invalid — never burn a retry re-submitting a rejected job
        except Exception:
            _drop_socks()
            if attempt == 2:
                raise


# ---------- batched dispatcher: concurrent requests ride ONE ring job ----------

class _Req:
    __slots__ = ("messages", "tools", "max_new", "reasoning", "on_commit", "event",
                 "result", "error", "dead", "cls")

    def __init__(self, messages, tools, max_new, reasoning, on_commit):
        self.messages = messages; self.tools = tools; self.max_new = max_new
        self.reasoning = reasoning; self.on_commit = on_commit
        self.cls = _content_class(messages, tools, reasoning)
        self.event = threading.Event(); self.result = None; self.error = None; self.dead = False


_QUEUE = []
_QCOND = threading.Condition()


def run_request(messages, tools, max_new, reasoning, on_commit=None, timeout=1800):
    """The handler-side entry: enqueue and wait. The dispatcher owns the ring; a burst becomes one
    batched job, a lone request the solo path. MOCK short-circuits (no ring, no queue)."""
    if MOCK:
        return _mock_generate(messages, tools, max_new, on_commit, reasoning)
    rq = _Req(messages, tools, max_new, reasoning, on_commit)
    with _QCOND:
        _QUEUE.append(rq); _QCOND.notify()
    if not rq.event.wait(timeout + 120):
        rq.dead = True                                  # dispatcher will drop this client's writes
        raise TimeoutError("gateway dispatch timed out")
    if rq.error:
        raise rq.error
    return rq.result


def _stream_cb(rq):
    """Per-stream commit callback: a dead/stalled CLIENT must never abort its batch-mates — mark the
    stream dead and let the batch finish (solo keeps its abort-the-job behavior; there the client IS
    the job)."""
    if rq.on_commit is None:
        return None
    def cb(out, dt):
        if rq.dead:
            return
        try:
            rq.on_commit(out, dt)
        except (ClientGone, OSError):
            rq.dead = True
    return cb


def _run_solo(rq):
    try:
        rq.result = generate(rq.messages, rq.tools, rq.max_new, rq.on_commit, reasoning=rq.reasoning)
    except Exception as e:  # noqa: BLE001 — hand the handler thread whatever happened
        rq.error = e


def _run_batch(reqs):
    k = _class_k(reqs[0].cls, B=len(reqs))             # single-class job (the dispatcher groups), one K
    for attempt in (1, 2):
        try:
            if "pipe" not in SOCKS or attempt == 2:
                _connect(1800)
            drafters = make_drafters_b(len(reqs), A.ngram_n)
            r = coordinate_pipe_batch(
                SOCKS["pipe"], tok, [rq.messages for rq in reqs], k,
                [rq.max_new for rq in reqs], 1800, SOCKS["ret"], drafters, depth=A.depth,
                tools_b=[rq.tools for rq in reqs], prefill_chunk=4096, max_ctx=A.max_ctx,
                reasoning=[rq.reasoning for rq in reqs], on_commits=[_stream_cb(rq) for rq in reqs])
            dt = max(r["dt"], 1e-9)
            for b, rq in enumerate(reqs):
                s = r["streams"][b]
                rq.result = {"ok": True, "text": s["text"], "n_tokens": s["n_tokens"],
                             "prompt_tokens": s["prompt_tokens"], "tok_s": s["n_tokens"] / dt,
                             "mean_accept": s["g"], "output_ids": s["output_ids"],
                             "receipts": r.get("receipts"), "receipts_ok": r.get("receipts_ok"),
                             "batched_B": r["B"]}
                if rq.dead:
                    rq.error = ClientGone("client left mid-stream (batch completed without it)")
            return
        except JobRejected:
            _drop_socks()                               # rejected job: report, never retry (see generate)
            raise
        except Exception as e:  # noqa: BLE001 — one retry with fresh sockets, then report to every caller
            _drop_socks()
            if attempt == 2:
                for rq in reqs:
                    rq.error = rq.error or e


def _dispatcher():
    """The one ring writer. Pop a burst (first request + whatever lands inside GW_WINDOW_MS, up to
    GW_BATCH), run it as one job, wake every caller. GW_BATCH=1 (an un-batched ring) degenerates to
    the exact old serialize-through-a-lock behavior. CONTENT ROUTING: a job carries only requests of
    the queue HEAD's content class (like with like — one K per job, homogeneous round regime);
    other-class requests keep their queue order and lead a following job, so nothing starves."""
    while True:
        with _QCOND:
            while not _QUEUE:
                _QCOND.wait()
        if GW_BATCH > 1:
            time.sleep(GW_WINDOW_MS / 1000.0)           # let a burst gather
        with _QCOND:
            if CONTENT_K and GW_BATCH > 1 and _QUEUE:
                head_cls = _QUEUE[0].cls
                batch, rest = [], []
                for rq in _QUEUE:
                    (batch if (rq.cls == head_cls and len(batch) < GW_BATCH) else rest).append(rq)
                _QUEUE[:] = rest
            else:
                batch = _QUEUE[:GW_BATCH]
                del _QUEUE[:GW_BATCH]
        # a caller that timed out waiting marked itself dead — never run its job (solo would hand its
        # raw callback to generate() and a write to the gone client would abort the ring pass)
        for rq in [r for r in batch if r.dead]:
            rq.event.set()
        batch = [r for r in batch if not r.dead]
        if not batch:
            continue
        try:
            with RING_LOCK:
                if len(batch) == 1:
                    _run_solo(batch[0])
                else:
                    _run_batch(batch)
        except Exception as e:  # noqa: BLE001 — the dispatcher must NEVER die (a dead dispatcher
            for rq in batch:    # hangs every future request silently); report to the batch instead
                rq.error = rq.error or e
        finally:
            for rq in batch:
                rq.event.set()


def _mock_generate(messages, tools, max_new, on_commit, reasoning=True):
    """No-GPU canned completion that exercises the real parse/stream/assembly path. If tools are
    offered, emits a tool call; else a short answer. Streams in slices so on_commit/diff is tested.
    Honours `reasoning`: with it off the output carries NO <think> block (mirrors render_ids closing
    it in the prompt), so the stream path is exercised in both modes."""
    last = messages[-1]["content"] if messages else ""
    think = f"\nThinking about it.\n{THINK_END}" if reasoning else ""
    if tools:
        name = tools[0]["function"]["name"]
        text = (f"{f'{chr(10)}The user asked: {last[:40]}. I will call {name}.{chr(10)}{THINK_END}' if reasoning else ''}\n\n"
                f"Let me look that up.{TOOLCALL_BEGIN}\n<invoke name=\"{name}\">\n"
                f"<parameter name=\"query\">{last[:30]}</parameter>\n</invoke>\n</minimax:tool_call>")
    else:
        text = f"{think}\n\nHere is a concise answer to: {last[:60]}."
    if on_commit:
        for i in range(8, len(text) + 8, 8):
            on_commit_text = text[:i]
            on_commit([("T", on_commit_text)], 0.0)   # mock carries text directly (see stream handler)
    return {"ok": True, "text": text, "n_tokens": max(1, len(text) // 4), "prompt_tokens": len(last) // 4,
            "tok_s": 17.0, "mean_accept": 4.0, "toks_per_traversal": 5.0, "rounds": 1, "output_ids": []}


# ---------- OpenAI request handling ----------

def _split_stream(text, reasoning_on=True):
    """Monotonic split of the running generation into (reasoning, content).

    reasoning ON: generation starts inside the forced <think>, so reasoning is everything up to
    </think> and content is after it (up to any tool-call block — never leak XML).

    reasoning OFF: render_ids already CLOSED <think> in the prompt, so the OUTPUT has no think block
    and is pure content. Without the flag, a </think>-less output fell through to the last return and
    streamed the WHOLE answer as reasoning_content — then the end-of-stream flush re-emitted it as
    content, DUPLICATING the answer. So when reasoning is off, everything is content."""
    if not reasoning_on:
        content = text.split(THINK_END)[-1]            # defensive: drop an echoed think-close if any
        return "", content.split(TOOLCALL_BEGIN)[0] if TOOLCALL_BEGIN in content else content
    if THINK_END in text:
        head, _, tail = text.partition(THINK_END)
        reasoning = head.split(THINK_BEGIN)[-1]
        content = tail.split(TOOLCALL_BEGIN)[0] if TOOLCALL_BEGIN in tail else tail
        return reasoning, content
    return text.split(THINK_BEGIN)[-1], ""


def _decode_running(out, handler):
    """on_commit payload -> decoded text. MOCK carries text in the payload; real mode carries token
    ids that the tokenizer decodes (skip_special_tokens keeps the tool-call/think markers)."""
    if MOCK:
        return out[0][1]
    return tok.decode(out, skip_special_tokens=True)


class H(BaseHTTPRequestHandler):
    server_version = "shard-m25-gateway"
    timeout = REQ_TIMEOUT          # socket deadline on header/body reads (slow-loris guard, H5)
    def log_message(self, *a): pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/v1/models", "/models"):
            return self._json({"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "shard"}]})
        if self.path in ("/health", "/"):
            return self._json({"status": "ok", "model": MODEL_ID, "engine": "mock" if MOCK else "ring",
                               "max_ctx": A.max_ctx if A else None})
        self._json({"error": {"message": "not found", "type": "invalid_request_error"}}, 404)

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            return self._json({"error": {"message": "not found", "type": "invalid_request_error"}}, 404)
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_BODY:
            self.close_connection = True               # body left unread -> the connection is unusable
            return self._json({"error": {"message": f"request body is {n} bytes; this gateway caps "
                                                    f"bodies at {MAX_BODY}",
                                         "type": "invalid_request_error",
                                         "code": "request_too_large"}}, 413)
        try:
            raw = self.rfile.read(n)
        except OSError:                                # client stalled past REQ_TIMEOUT or died mid-body
            self.close_connection = True
            return
        try:
            body = json.loads(raw or b"{}")
        except Exception:
            return self._json({"error": {"message": "invalid JSON body", "type": "invalid_request_error"}}, 400)
        messages = body.get("messages")
        if not messages:
            return self._json({"error": {"message": "messages is required", "type": "invalid_request_error"}}, 400)
        if not _admit():                               # fast-fail: never park an unbounded thread pile
            self.close_connection = True               # on the dispatcher while the ring is saturated
            return self._json({"error": {"message": f"gateway at capacity ({MAX_INFLIGHT} requests "
                                                    f"in flight); retry with backoff",
                                         "type": "rate_limit_error",
                                         "code": "gateway_overloaded"}}, 429)
        try:
            self._do_completion(body, messages)
        finally:
            _release()

    def _do_completion(self, body, messages):
        # decoding is greedy (speculative verify): a non-greedy sampling request must be REJECTED,
        # not silently executed greedily — the caller asked for a distribution we don't produce.
        for knob, allowed in (("temperature", {0.0}), ("top_p", {1.0}), ("top_k", {0.0, 1.0})):
            v = body.get(knob)
            if v is None:
                continue
            try:
                greedy = float(v) in allowed
            except (TypeError, ValueError):
                greedy = False
            if not greedy:
                return self._json({"error": {
                    "message": f"{knob}={v!r} is unsupported: decoding is greedy (speculative "
                               f"verify); omit {knob} or pass a greedy value",
                    "type": "invalid_request_error", "code": "unsupported_sampling"}}, 400)
        tools = body.get("tools") or None
        tc = body.get("tool_choice")
        require_tool = False                           # 'required'/named: prose-only output = error
        if tc == "none":
            tools = None
        elif isinstance(tc, dict):                     # named function: offer ONLY that tool
            name = (tc.get("function") or {}).get("name")
            named = [t for t in (tools or []) if (t.get("function") or {}).get("name") == name]
            if not named:
                return self._json({"error": {"message": f"tool_choice names unknown tool {name!r}",
                                             "type": "invalid_request_error"}}, 400)
            tools = named
            require_tool = True
        elif tc == "required":
            if not tools:
                return self._json({"error": {"message": "tool_choice 'required' needs a non-empty "
                                                        "tools list",
                                             "type": "invalid_request_error"}}, 400)
            require_tool = True
        elif tc not in (None, "auto"):
            return self._json({"error": {"message": f"unsupported tool_choice {tc!r}",
                                         "type": "invalid_request_error"}}, 400)
        max_new = max(1, int(body.get("max_tokens") or body.get("max_completion_tokens") or 512))
        stream = bool(body.get("stream"))
        if body.get("reasoning") is not None:                    # explicit bool override
            reasoning = bool(body.get("reasoning"))
        elif body.get("reasoning_effort") is not None:           # OpenAI-style: "none" -> off
            reasoning = body.get("reasoning_effort") != "none"
        else:
            reasoning = DEFAULT_REASONING
        # H1: the request must fit the NEGOTIATED context (A.max_ctx = min of the operator ceiling
        # and every stage's KV cap, computed by the launcher) plus speculative headroom — reject
        # BEFORE anything is enqueued, so an over-limit prompt never reaches the ring. MOCK keeps
        # the rejection path testable without a GPU (chars//4 estimate).
        if MOCK:
            n_prompt = sum(len(str(m.get("content") or "")) for m in messages) // 4
        else:
            n_prompt = len(render_ids(tok, messages, tools=tools, reasoning=reasoning))
        if n_prompt + HEADROOM >= A.max_ctx:
            return self._json({"error": {
                "message": f"prompt is {n_prompt} tokens; negotiated max_ctx for this ring is "
                           f"{A.max_ctx} (min stage KV cap) with {HEADROOM} tokens speculative headroom",
                "type": "invalid_request_error", "code": "context_length_exceeded",
                "prompt_tokens": n_prompt, "max_ctx": A.max_ctx, "headroom": HEADROOM}}, 400)
        max_new = min(max_new, A.max_ctx - n_prompt - HEADROOM)   # silent clamp (OpenAI convention)
        cid = f"chatcmpl-{next(_ids)}"; created = int(time.time())
        try:
            # no ring lock here: the dispatcher owns the ring; handlers enqueue and wait, so a burst
            # of concurrent requests rides ONE batched job instead of serializing.
            if stream:
                self._stream(cid, created, messages, tools, max_new, reasoning, require_tool)
            else:
                self._complete(cid, created, messages, tools, max_new, reasoning, require_tool)
        except (BrokenPipeError, ClientGone):
            pass                                          # client is gone — nothing to send, ring already released
        except JobRejected as e:
            if getattr(self, "_sse_started", False):
                return                                    # _stream already reported in-band (M5)
            try: self._json({"error": {"message": str(e), "type": "invalid_request_error",
                                       "code": "job_rejected"}}, 400)
            except Exception: pass
        except Exception as e:
            if getattr(self, "_sse_started", False):
                return                                    # _stream already emitted the SSE error + [DONE]
            err = {"error": {"message": f"{type(e).__name__}: {str(e)[:200]}", "type": "engine_error"}}
            try: self._json(err, 500)
            except Exception: pass

    def _complete(self, cid, created, messages, tools, max_new, reasoning=True, require_tool=False):
        r = run_request(messages, tools, max_new, reasoning, on_commit=None)
        fin_cap = _enforce_cap(r, max_new)
        parsed = parse_completion(r["text"])
        if require_tool and not parsed["tool_calls"]:
            return self._json({"error": {"message": "tool_choice required a tool call but the model "
                                                    "produced none",
                                         "type": "engine_error", "code": "tool_choice_unfulfilled"}}, 502)
        msg, finish = to_openai_message(parsed)
        if not (tools and parsed["tool_calls"]) and finish == "tool_calls":
            finish = "stop"
        if fin_cap == "length":
            finish = "length"
        self._json({
            "id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": r.get("prompt_tokens", 0), "completion_tokens": r["n_tokens"],
                      "total_tokens": r.get("prompt_tokens", 0) + r["n_tokens"]},
            "x_shard": {"tok_s": round(r.get("tok_s", 0), 2), "mean_accept": round(r.get("mean_accept", 0), 2),
                        "toks_per_traversal": round(r.get("toks_per_traversal", 0), 2),
                        "receipts_ok": r.get("receipts_ok"), "n_receipts": len(r.get("receipts") or [])},
        })

    def _stream(self, cid, created, messages, tools, max_new, reasoning=True, require_tool=False):
        self.close_connection = True   # no chunked framing -> close at end so clients get clean EOF after [DONE]
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
        self._sse_started = True   # stream committed: from here ANY report goes in-band, never a 2nd status line
        self.connection.settimeout(STREAM_WRITE_TIMEOUT)   # a stalled client write must not pin the ring

        def raw(b):
            try:
                self.wfile.write(b); self.wfile.flush()
            except OSError as e:                           # client disconnected OR stalled past the timeout:
                raise ClientGone(f"{type(e).__name__}: {e}") from e   # abort the job, never retry (see generate)

        def chunk(delta, finish=None):
            o = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            raw(f"data: {json.dumps(o)}\n\n".encode())

        try:
            chunk({"role": "assistant"})
            state = {"r": 0, "c": 0}
            def on_commit(out, _dt):
                if not MOCK:                               # strict max_tokens + earliest-eos: never stream
                    out, _ = _cap_output(out, max_new, EOS_SET)   # tokens the final message won't carry
                text = _decode_running(out, self)
                if MOCK:
                    text = text[:max_new * 4]              # mock counts tokens as chars//4
                reasoning_txt, content = _split_stream(text, reasoning)   # `reasoning` = the request's bool (closure)
                if len(reasoning_txt) > state["r"]:
                    chunk({"reasoning_content": reasoning_txt[state["r"]:]}); state["r"] = len(reasoning_txt)
                if len(content) > state["c"]:
                    chunk({"content": content[state["c"]:]}); state["c"] = len(content)

            r = run_request(messages, tools, max_new, reasoning, on_commit=on_commit)
            fin_cap = _enforce_cap(r, max_new)
            parsed = parse_completion(r["text"])
            if require_tool and not parsed["tool_calls"]:
                raise RuntimeError("tool_choice required a tool call but the model produced none")
            # flush any tail not yet streamed (final trimmed text), then tool calls
            _, fcontent = _split_stream(r["text"], reasoning)
            final_content = parsed["content"] or ""
            if len(final_content) > state["c"]:
                chunk({"content": final_content[state["c"]:]})
            finish = "stop"
            if tools and parsed["tool_calls"]:
                msg, _ = to_openai_message(parsed)
                chunk({"tool_calls": msg["tool_calls"]}); finish = "tool_calls"
            if fin_cap == "length":
                finish = "length"
            chunk({}, finish=finish)
            usage = {"prompt_tokens": r.get("prompt_tokens", 0), "completion_tokens": r["n_tokens"],
                     "total_tokens": r.get("prompt_tokens", 0) + r["n_tokens"]}
            raw(f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [], 'usage': usage})}\n\n".encode())
            raw(b"data: [DONE]\n\n")
        except ClientGone:
            raise                                          # client is gone: nothing can be sent to it
        except Exception as e:                             # noqa: BLE001 — engine/ring failure mid-stream:
            # the 200 + SSE framing is already on the wire, so a JSON 500 here would be a SECOND
            # status line spliced into the event stream. Emit exactly ONE in-band error event, then
            # terminate the stream properly with [DONE].
            err = {"error": {"message": f"{type(e).__name__}: {str(e)[:200]}", "type": "engine_error"}}
            try:
                raw(f"data: {json.dumps(err)}\n\n".encode())
                raw(b"data: [DONE]\n\n")
            except ClientGone:
                pass                                       # client died while we reported — nothing left


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True); ap.add_argument("--tail", required=True)
    ap.add_argument("--port", type=int, default=29600)
    ap.add_argument("--K", type=int, default=8); ap.add_argument("--depth", type=int, default=4)   # K=8 = the measured sweet spot (2026-06-27 sweep)
    ap.add_argument("--ngram-n", type=int, default=3, dest="ngram_n")
    ap.add_argument("--max-ctx", type=int, default=131072, dest="max_ctx",
                    help="negotiated context ceiling; the launcher passes min(operator, stage KV caps)")
    _configure(ap.parse_args())
    if not MOCK:
        _engine_init()
        threading.Thread(target=_dispatcher, daemon=True).start()
    print(f"[m25-gateway] :{A.port}  model={MODEL_ID}  engine={'MOCK' if MOCK else f'head={A.head} tail={A.tail}'}  "
          f"(OpenAI /v1/chat/completions, batch<= {GW_BATCH} window={GW_WINDOW_MS:.0f}ms)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", A.port), H).serve_forever()
