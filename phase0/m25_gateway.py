"""shard OpenAI-compatible gateway for MiniMax-M2.5 — the c0mpute integration seam.

Exposes /v1/chat/completions (messages + tools + tool_choice + streaming) and /v1/models over the
scattered libp2p ring. This is the PROGRAMMATIC api c0mpute calls — distinct from gateway.py's shared
public demo terminal. The shard engine is single-stream (one ring), so requests are serialized
through a lock; concurrent callers queue.

  SHARD_TRANSPORT=libp2p M25_DIR=/root/m25 python m25_gateway.py --head H:P --tail H:P --port 29600
  M25_GATEWAY_MOCK=1 python m25_gateway.py --head x --tail x --port 29600   # local api/shape test, no GPU

Beta notes: decoding is greedy (n-gram speculative verify); `temperature`/`top_p`/`top_k` are accepted
but not yet applied (lossless sampling is a separate engine lever — the tail argmaxes today). One
in-flight request at a time.
"""
import argparse, json, os, socket, sys, threading, time, itertools
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m25_tools import parse_completion, to_openai_message, TOOLCALL_BEGIN, THINK_BEGIN, THINK_END

MOCK = bool(os.environ.get("M25_GATEWAY_MOCK"))
MODEL_ID = os.environ.get("M25_MODEL_ID", "minimax-m2.5")
# default reasoning mode for the gateway: M2.5 hardwires a <think> block, which is novel (0% n-gram
# accept) so it runs at the WAN floor and dominates latency. M25_DEFAULT_REASONING=0 makes the gateway
# answer DIRECTLY by default (fast, for latency-sensitive/high-overlap normal usage); a request can
# override per-call with {"reasoning": true/false} or {"reasoning_effort": "none"|...}. Default ON (quality).
DEFAULT_REASONING = os.environ.get("M25_DEFAULT_REASONING", "1") != "0"
NODELAY = (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
_ids = itertools.count(1)
RING_LOCK = threading.Lock()   # the ring is single-stream: one generation at a time

A = None
tok = None
coordinate_pipe = None
make_drafter = None
SOCKS = {}


def _engine_init():
    """Import the M2.5 engine + tokenizer and resolve head/tail endpoints (real mode only)."""
    global tok, coordinate_pipe, make_drafter
    import m25_stage as S
    from m25_pipe import coordinate_pipe as cp, make_drafter as md
    from transformers import AutoTokenizer
    coordinate_pipe = cp; make_drafter = md
    tok = AutoTokenizer.from_pretrained(S.DIR, trust_remote_code=True)


def _connect(timeout):
    for s in SOCKS.values():
        try: s.close()
        except OSError: pass
    SOCKS.clear()
    hh, hp = A.head.rsplit(":", 1); th, tp = A.tail.rsplit(":", 1)
    pipe = socket.create_connection((hh, int(hp)), timeout=timeout); pipe.setsockopt(*NODELAY)
    ret = socket.create_connection((th, int(tp)), timeout=timeout); ret.setsockopt(*NODELAY); ret.settimeout(timeout)
    from node_kv import send_msg, recv_msg
    send_msg(ret, {"op": "hello_return"}); recv_msg(ret)   # wait ret_ok before any reset flows
    SOCKS.update(pipe=pipe, ret=ret)


def generate(messages, tools, max_new, on_commit, timeout=1800, reasoning=True):
    """Run one chat completion through the ring (or a canned reply in MOCK). Returns the
    coordinate_pipe result dict ({text, n_tokens, prompt_tokens, tok_s, mean_accept, ...})."""
    if MOCK:
        return _mock_generate(messages, tools, max_new, on_commit)
    for attempt in (1, 2):
        try:
            if "pipe" not in SOCKS or attempt == 2:
                _connect(timeout)
            drafter = make_drafter(A.ngram_n)
            return coordinate_pipe(SOCKS["pipe"], tok, messages, A.K, max_new, timeout, A.depth,
                                   ret_sock=SOCKS["ret"], local_draft=drafter, tools=tools,
                                   prefill_chunk=4096, max_ctx=A.max_ctx, on_commit=on_commit, reasoning=reasoning)
        except Exception:
            SOCKS.clear()
            if attempt == 2:
                raise


def _mock_generate(messages, tools, max_new, on_commit):
    """No-GPU canned completion that exercises the real parse/stream/assembly path. If tools are
    offered, emits a tool call; else a short answer. Streams in slices so on_commit/diff is tested."""
    last = messages[-1]["content"] if messages else ""
    if tools:
        name = tools[0]["function"]["name"]
        text = (f"\nThe user asked: {last[:40]}. I'll call {name}.\n{THINK_END}\n\n"
                f"Let me look that up.{TOOLCALL_BEGIN}\n<invoke name=\"{name}\">\n"
                f"<parameter name=\"query\">{last[:30]}</parameter>\n</invoke>\n</minimax:tool_call>")
    else:
        text = f"\nThinking about it.\n{THINK_END}\n\nHere is a concise answer to: {last[:60]}."
    if on_commit:
        for i in range(8, len(text) + 8, 8):
            on_commit_text = text[:i]
            on_commit([("T", on_commit_text)], 0.0)   # mock carries text directly (see stream handler)
    return {"ok": True, "text": text, "n_tokens": max(1, len(text) // 4), "prompt_tokens": len(last) // 4,
            "tok_s": 17.0, "mean_accept": 4.0, "toks_per_traversal": 5.0, "rounds": 1, "output_ids": []}


# ---------- OpenAI request handling ----------

def _split_stream(text):
    """Monotonic split for streaming: generation starts inside the forced <think>, so reasoning is
    everything up to </think>; content is after it, up to any tool-call block (never leak XML)."""
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
            return self._json({"status": "ok", "model": MODEL_ID, "engine": "mock" if MOCK else "ring"})
        self._json({"error": {"message": "not found", "type": "invalid_request_error"}}, 404)

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            return self._json({"error": {"message": "not found", "type": "invalid_request_error"}}, 404)
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json({"error": {"message": "invalid JSON body", "type": "invalid_request_error"}}, 400)
        messages = body.get("messages")
        if not messages:
            return self._json({"error": {"message": "messages is required", "type": "invalid_request_error"}}, 400)
        tools = body.get("tools") or None
        if body.get("tool_choice") == "none":
            tools = None
        max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 512)
        stream = bool(body.get("stream"))
        if body.get("reasoning") is not None:                    # explicit bool override
            reasoning = bool(body.get("reasoning"))
        elif body.get("reasoning_effort") is not None:           # OpenAI-style: "none" -> off
            reasoning = body.get("reasoning_effort") != "none"
        else:
            reasoning = DEFAULT_REASONING
        cid = f"chatcmpl-{next(_ids)}"; created = int(time.time())
        try:
            with RING_LOCK:
                if stream:
                    self._stream(cid, created, messages, tools, max_new, reasoning)
                else:
                    self._complete(cid, created, messages, tools, max_new, reasoning)
        except BrokenPipeError:
            pass
        except Exception as e:
            err = {"error": {"message": f"{type(e).__name__}: {str(e)[:200]}", "type": "engine_error"}}
            try: self._json(err, 500)
            except Exception: pass

    def _complete(self, cid, created, messages, tools, max_new, reasoning=True):
        r = generate(messages, tools, max_new, on_commit=None, reasoning=reasoning)
        parsed = parse_completion(r["text"])
        msg, finish = to_openai_message(parsed)
        if not (tools and parsed["tool_calls"]) and finish == "tool_calls":
            finish = "stop"
        self._json({
            "id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": r.get("prompt_tokens", 0), "completion_tokens": r["n_tokens"],
                      "total_tokens": r.get("prompt_tokens", 0) + r["n_tokens"]},
            "x_shard": {"tok_s": round(r.get("tok_s", 0), 2), "mean_accept": round(r.get("mean_accept", 0), 2),
                        "toks_per_traversal": round(r.get("toks_per_traversal", 0), 2),
                        "receipts_ok": r.get("receipts_ok"), "n_receipts": len(r.get("receipts") or [])},
        })

    def _stream(self, cid, created, messages, tools, max_new, reasoning=True):
        self.close_connection = True   # no chunked framing -> close at end so clients get clean EOF after [DONE]
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()

        def chunk(delta, finish=None):
            o = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            self.wfile.write(f"data: {json.dumps(o)}\n\n".encode()); self.wfile.flush()

        chunk({"role": "assistant"})
        state = {"r": 0, "c": 0}
        def on_commit(out, _dt):
            text = _decode_running(out, self)
            reasoning, content = _split_stream(text)
            if len(reasoning) > state["r"]:
                chunk({"reasoning_content": reasoning[state["r"]:]}); state["r"] = len(reasoning)
            if len(content) > state["c"]:
                chunk({"content": content[state["c"]:]}); state["c"] = len(content)

        r = generate(messages, tools, max_new, on_commit=on_commit, reasoning=reasoning)
        parsed = parse_completion(r["text"])
        # flush any tail not yet streamed (final trimmed text), then tool calls
        _, fcontent = _split_stream(r["text"])
        final_content = parsed["content"] or ""
        if len(final_content) > state["c"]:
            chunk({"content": final_content[state["c"]:]})
        finish = "stop"
        if tools and parsed["tool_calls"]:
            msg, _ = to_openai_message(parsed)
            chunk({"tool_calls": msg["tool_calls"]}); finish = "tool_calls"
        chunk({}, finish=finish)
        usage = {"prompt_tokens": r.get("prompt_tokens", 0), "completion_tokens": r["n_tokens"],
                 "total_tokens": r.get("prompt_tokens", 0) + r["n_tokens"]}
        self.wfile.write(f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [], 'usage': usage})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True); ap.add_argument("--tail", required=True)
    ap.add_argument("--port", type=int, default=29600)
    ap.add_argument("--K", type=int, default=8); ap.add_argument("--depth", type=int, default=4)   # K=8 = the measured sweet spot (2026-06-27 sweep)
    ap.add_argument("--ngram-n", type=int, default=3, dest="ngram_n")
    ap.add_argument("--max-ctx", type=int, default=131072, dest="max_ctx")
    A = ap.parse_args()
    if not MOCK:
        _engine_init()
    print(f"[m25-gateway] :{A.port}  model={MODEL_ID}  engine={'MOCK' if MOCK else f'head={A.head} tail={A.tail}'}  "
          f"(OpenAI /v1/chat/completions, single-stream)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", A.port), H).serve_forever()
