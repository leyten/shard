"""Gateway audit hardening (phase0/m25_gateway.py) — context negotiation, intake limits, OpenAI
semantics, health/readiness, streaming error framing, incremental detokenize, ring greetings.

All CPU-only: the MOCK engine + fake handler sockets (same harness as test_gateway_streaming.py).
Run: M25_GATEWAY_MOCK=1 python3 -m pytest tests/test_gateway_hardening.py -q
"""
import io
import json
import os
import sys
import types

import pytest

os.environ["M25_GATEWAY_MOCK"] = "1"                    # MOCK is read at import -> set before importing
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase0"))

import m25_gateway as gw                                # noqa: E402


# ---- harness (mirrors test_gateway_streaming.py, extended to drive do_POST/do_GET) -----------------

class _FakeWfile:
    def __init__(self, fail_after=None):
        self.chunks = []
        self.fail_after = fail_after
        self.writes = 0

    def write(self, b):
        self.writes += 1
        if self.fail_after is not None and self.writes > self.fail_after:
            raise BrokenPipeError("client disconnected")
        self.chunks.append(b)

    def flush(self):
        pass


class _FakeConn:
    def settimeout(self, t):
        self.timeout = t


def _handler(wfile=None, body=None, path="/v1/chat/completions"):
    """An H instance wired to fake sockets, bypassing BaseHTTPRequestHandler.__init__. Records every
    send_response status so tests can assert EXACTLY ONE status line per exchange."""
    h = gw.H.__new__(gw.H)
    h.wfile = wfile or _FakeWfile()
    h.connection = _FakeConn()
    h.close_connection = False
    h.path = path
    raw = json.dumps(body).encode() if body is not None else b""
    h.rfile = io.BytesIO(raw)
    h.headers = {"Content-Length": str(len(raw))}
    h.statuses = []
    h.send_response = lambda code, *a, **k: h.statuses.append(code)
    h.send_header = lambda *a, **k: None
    h.end_headers = lambda: None
    return h


def _json_reply(h):
    return json.loads(h.wfile.chunks[-1].decode())


def _sse_frames(wfile):
    out = []
    for raw in wfile.chunks:
        for line in raw.decode().splitlines():
            if line.startswith("data: "):
                out.append(line[len("data: "):].strip())
    return out


def _sse_deltas(wfile):
    out = []
    for payload in _sse_frames(wfile):
        if payload == "[DONE]":
            continue
        obj = json.loads(payload)
        if obj.get("choices"):
            out.append(obj["choices"][0])
    return out


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    """Default gateway config for every test: MOCK engine, a roomy max_ctx, clean intake state."""
    monkeypatch.setattr(gw, "A", types.SimpleNamespace(
        K=8, depth=4, ngram_n=3, max_ctx=8192, head="h:1", tail="t:2", port=0))
    monkeypatch.setattr(gw, "K_MAX", 8, raising=False)
    monkeypatch.setattr(gw, "HEADROOM", 24, raising=False)
    if hasattr(gw, "_INFLIGHT"):
        gw._INFLIGHT["n"] = 0
    gw.SOCKS.clear()
    yield


# ---- H1: negotiated max_ctx (pure functions + rejection before dispatch) ----------------------------

def test_negotiated_max_ctx_takes_stage_min():
    assert gw.negotiated_max_ctx(131072, [40960, 40960]) == 40960


def test_negotiated_max_ctx_operator_can_be_lower():
    assert gw.negotiated_max_ctx(16384, [40960, 65536]) == 16384


def test_negotiated_max_ctx_ignores_zero_caps():
    assert gw.negotiated_max_ctx(131072, [0, 40960, 0]) == 40960
    assert gw.negotiated_max_ctx(131072, []) == 131072


def test_spec_headroom_values():
    assert gw.spec_headroom(8) == 24
    assert gw.spec_headroom(6, tree=True, tree_m=16) == 32


def test_over_limit_prompt_rejected_before_dispatch(monkeypatch):
    """MOCK estimates chars//4; a prompt past A.max_ctx - HEADROOM gets a 400 with
    code=context_length_exceeded and never produces a completion."""
    monkeypatch.setattr(gw.A, "max_ctx", 256)
    called = []
    monkeypatch.setattr(gw, "run_request", lambda *a, **k: called.append(1) or {})
    h = _handler(body={"messages": [{"role": "user", "content": "x" * 4096}]})
    h.do_POST()
    r = _json_reply(h)
    assert h.statuses == [400]
    assert r["error"]["code"] == "context_length_exceeded"
    assert r["error"]["max_ctx"] == 256
    assert called == [], "over-limit request must never reach the ring"


def test_under_limit_max_new_clamped_to_headroom(monkeypatch):
    """A fitting prompt passes, but max_tokens is silently clamped so prompt+new+headroom <= max_ctx."""
    monkeypatch.setattr(gw.A, "max_ctx", 300)
    seen = {}
    def fake_run(messages, tools, max_new, reasoning, on_commit=None, timeout=1800):
        seen["max_new"] = max_new
        return gw._mock_generate(messages, tools, max_new, on_commit, reasoning)
    monkeypatch.setattr(gw, "run_request", fake_run)
    prompt = "x" * 400                                   # ~100 estimated tokens
    h = _handler(body={"messages": [{"role": "user", "content": prompt}], "max_tokens": 10000})
    h.do_POST()
    assert h.statuses == [200]
    assert seen["max_new"] == 300 - 100 - gw.HEADROOM


def test_job_rejected_maps_to_400_and_never_retries(monkeypatch):
    calls = []
    def cp(*a, **k):
        calls.append(1)
        raise gw.JobRejected('{"code": "kv_overflow"}')
    monkeypatch.setattr(gw, "MOCK", False)
    monkeypatch.setattr(gw, "coordinate_pipe", cp)
    monkeypatch.setattr(gw, "make_drafter", lambda n: object())
    fake_sock = lambda: types.SimpleNamespace(close=lambda: None)  # noqa: E731
    monkeypatch.setattr(gw, "_connect", lambda t: gw.SOCKS.update(pipe=fake_sock(), ret=fake_sock()))
    with pytest.raises(gw.JobRejected):
        gw.generate([{"role": "user", "content": "hi"}], None, 32, on_commit=None)
    assert calls == [1], "a rejected job must never be retried"
    assert not gw.SOCKS, "rejected job must drop the ring sockets"


def test_do_post_maps_job_rejected_to_400(monkeypatch):
    def boom(*a, **k):
        raise gw.JobRejected("kv overflow at stage 2")
    monkeypatch.setattr(gw, "run_request", boom)
    h = _handler(body={"messages": [{"role": "user", "content": "hi"}]})
    h.do_POST()
    r = _json_reply(h)
    assert h.statuses == [400]
    assert r["error"]["code"] == "job_rejected"


def test_health_reports_negotiated_max_ctx():
    h = _handler(path="/health")
    h.do_GET()
    assert _json_reply(h)["max_ctx"] == gw.A.max_ctx


# ---- H5: bounded intake (body cap, in-flight cap, read deadline) ------------------------------------

def test_oversized_body_rejected_413(monkeypatch):
    monkeypatch.setattr(gw, "MAX_BODY", 100)
    h = _handler(body={"messages": [{"role": "user", "content": "y" * 500}]})
    h.do_POST()
    r = _json_reply(h)
    assert h.statuses == [413]
    assert r["error"]["code"] == "request_too_large"


def test_over_capacity_rejected_429(monkeypatch):
    monkeypatch.setattr(gw, "MAX_INFLIGHT", 1)
    gw._INFLIGHT["n"] = 1                                # someone already in flight
    h = _handler(body={"messages": [{"role": "user", "content": "hi"}]})
    h.do_POST()
    r = _json_reply(h)
    gw._INFLIGHT["n"] = 0
    assert h.statuses == [429]
    assert r["error"]["code"] == "gateway_overloaded"


def test_inflight_released_after_request():
    h = _handler(body={"messages": [{"role": "user", "content": "hi"}]})
    h.do_POST()
    assert h.statuses == [200]
    assert gw._INFLIGHT["n"] == 0, "in-flight slot must be released"


def test_inflight_released_on_engine_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("ring fell over")
    monkeypatch.setattr(gw, "run_request", boom)
    h = _handler(body={"messages": [{"role": "user", "content": "hi"}]})
    h.do_POST()
    assert gw._INFLIGHT["n"] == 0, "slot must be released on failure too"


def test_handler_read_deadline_armed():
    assert gw.H.timeout == gw.REQ_TIMEOUT and gw.REQ_TIMEOUT > 0, \
        "header/body reads must carry a socket deadline (slow-loris guard)"


def test_body_read_timeout_drops_connection():
    h = _handler()
    h.rfile = types.SimpleNamespace(read=lambda n: (_ for _ in ()).throw(TimeoutError("stalled")))
    h.headers = {"Content-Length": "10"}
    h.do_POST()
    assert h.statuses == [], "a stalled body read must drop, not crash the handler thread"
    assert h.close_connection


# ---- M2: OpenAI semantics ----------------------------------------------------------------------------

def test_cap_output_strict_max_tokens():
    ids, fin = gw._cap_output([1, 2, 3, 4, 5], 3, set())
    assert ids == [1, 2, 3] and fin == "length"


def test_cap_output_earliest_eos_wins():
    ids, fin = gw._cap_output([1, 2, 9, 3, 9, 4], 10, {9})
    assert ids == [1, 2] and fin == "stop"


def test_cap_output_under_cap_untouched():
    ids, fin = gw._cap_output([1, 2], 10, {9})
    assert ids == [1, 2] and fin is None


def test_max_tokens_one_returns_exactly_one():
    h = _handler(body={"messages": [{"role": "user", "content": "hello"}], "max_tokens": 1})
    h.do_POST()
    r = _json_reply(h)
    assert h.statuses == [200]
    assert r["usage"]["completion_tokens"] == 1
    assert r["choices"][0]["finish_reason"] == "length"


def test_nonzero_temperature_rejected_400():
    h = _handler(body={"messages": [{"role": "user", "content": "hi"}], "temperature": 0.7})
    h.do_POST()
    r = _json_reply(h)
    assert h.statuses == [400]
    assert "temperature" in r["error"]["message"]


def test_greedy_compatible_sampling_accepted():
    h = _handler(body={"messages": [{"role": "user", "content": "hi"}],
                       "temperature": 0, "top_p": 1, "top_k": 1})
    h.do_POST()
    assert h.statuses == [200]


def test_top_p_rejected_400():
    h = _handler(body={"messages": [{"role": "user", "content": "hi"}], "top_p": 0.9})
    h.do_POST()
    assert h.statuses == [400]


_TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]


def test_tool_choice_named_unknown_tool_400():
    h = _handler(body={"messages": [{"role": "user", "content": "hi"}], "tools": _TOOLS,
                       "tool_choice": {"type": "function", "function": {"name": "nope"}}})
    h.do_POST()
    assert h.statuses == [400]


def test_tool_choice_named_filters_tools(monkeypatch):
    tools2 = _TOOLS + [{"type": "function", "function": {"name": "other", "parameters": {}}}]
    seen = {}
    def fake_run(messages, tools, max_new, reasoning, on_commit=None, timeout=1800):
        seen["tools"] = tools
        return gw._mock_generate(messages, tools, max_new, on_commit, reasoning)
    monkeypatch.setattr(gw, "run_request", fake_run)
    h = _handler(body={"messages": [{"role": "user", "content": "hi"}], "tools": tools2,
                       "tool_choice": {"type": "function", "function": {"name": "other"}}})
    h.do_POST()
    assert h.statuses == [200]
    assert [t["function"]["name"] for t in seen["tools"]] == ["other"]


def test_tool_choice_required_without_tools_400():
    h = _handler(body={"messages": [{"role": "user", "content": "hi"}], "tool_choice": "required"})
    h.do_POST()
    assert h.statuses == [400]


def test_tool_choice_required_unfulfilled_is_error(monkeypatch):
    """The engine returned prose where a tool call was required: never silently hand prose back."""
    monkeypatch.setattr(gw, "run_request",
                        lambda *a, **k: {"ok": True, "text": "just prose", "n_tokens": 2,
                                         "prompt_tokens": 1, "output_ids": []})
    h = _handler(body={"messages": [{"role": "user", "content": "hi"}], "tools": _TOOLS,
                       "tool_choice": "required"})
    h.do_POST()
    r = _json_reply(h)
    assert h.statuses == [502]
    assert "tool" in r["error"]["message"]


def test_tool_choice_required_fulfilled_ok():
    h = _handler(body={"messages": [{"role": "user", "content": "hi"}], "tools": _TOOLS,
                       "tool_choice": "required"})
    h.do_POST()
    r = _json_reply(h)
    assert h.statuses == [200]
    assert r["choices"][0]["finish_reason"] == "tool_calls"
    assert r["choices"][0]["message"]["tool_calls"]


def test_stream_max_tokens_one_usage_is_one():
    h = _handler(body={"messages": [{"role": "user", "content": "hello"}], "stream": True,
                       "max_tokens": 1})
    h.do_POST()
    usage = [json.loads(f)["usage"] for f in _sse_frames(h.wfile)
             if f != "[DONE]" and "usage" in json.loads(f)]
    assert usage and usage[-1]["completion_tokens"] == 1


# ---- M5: one valid SSE error+end sequence, one status line ---------------------------------------------

def test_stream_engine_failure_emits_sse_error_and_done(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("ring fell over")
    monkeypatch.setattr(gw, "run_request", boom)
    h = _handler(body={"messages": [{"role": "user", "content": "hi"}], "stream": True})
    h.do_POST()
    frames = _sse_frames(h.wfile)
    assert h.statuses == [200], f"exactly one status line, got {h.statuses}"
    errs = [f for f in frames if f != "[DONE]" and "error" in json.loads(f)]
    assert len(errs) == 1, f"exactly one SSE error frame, got {frames}"
    assert "ring fell over" in errs[0]
    assert frames[-1] == "[DONE]", "stream must terminate with [DONE]"


def test_stream_client_gone_stays_silent():
    """A dead client gets nothing appended — no JSON 500, no half SSE error into a closed pipe."""
    wf = _FakeWfile(fail_after=1)                        # role chunk ok, first delta write dies
    h = _handler(wfile=wf, body={"messages": [{"role": "user", "content": "hi"}], "stream": True})
    h.do_POST()
    assert h.statuses == [200]
    assert all(b"engine_error" not in c for c in wf.chunks)


def test_stream_happy_path_single_done():
    h = _handler(body={"messages": [{"role": "user", "content": "hello"}], "stream": True})
    h.do_POST()
    frames = _sse_frames(h.wfile)
    assert h.statuses == [200]
    assert frames.count("[DONE]") == 1 and frames[-1] == "[DONE]"
    content = "".join(d["delta"].get("content", "") for d in _sse_deltas(h.wfile) if d.get("delta"))
    assert "concise answer" in content and content.count("concise answer") == 1


# ---- M4: liveness vs readiness ------------------------------------------------------------------------

def test_health_is_liveness_with_queue_state():
    h = _handler(path="/health")
    h.do_GET()
    r = _json_reply(h)
    assert h.statuses == [200]
    for key in ("queue", "inflight", "ring_connected", "max_ctx"):
        assert key in r, f"/health must expose {key}"


def test_ready_endpoint_exists_and_mock_is_ready():
    h = _handler(path="/ready")
    h.do_GET()
    r = _json_reply(h)
    assert h.statuses == [200]
    assert r["ready"] is True


def test_ready_503_when_ring_unreachable(monkeypatch):
    monkeypatch.setattr(gw, "MOCK", False)
    gw._LAST_OK["t"] = 0.0
    gw._READY_CACHE.update(t=0.0, ok=None, why="")
    def refuse(addr, timeout=None):
        raise ConnectionRefusedError("no ring")
    monkeypatch.setattr(gw.socket, "create_connection", refuse)
    h = _handler(path="/ready")
    h.do_GET()
    r = _json_reply(h)
    gw._READY_CACHE.update(t=0.0, ok=None, why="")
    assert h.statuses == [503]
    assert r["ready"] is False and r["reason"]


def test_ready_uses_recent_job_instead_of_probe(monkeypatch):
    monkeypatch.setattr(gw, "MOCK", False)
    gw._LAST_OK["t"] = gw.time.monotonic()
    def boom(addr, timeout=None):
        raise AssertionError("must not probe when a job just succeeded")
    monkeypatch.setattr(gw.socket, "create_connection", boom)
    h = _handler(path="/ready")
    h.do_GET()
    gw._LAST_OK["t"] = 0.0
    assert h.statuses == [200]


# ---- detokenize-perf: incremental decode + incremental split --------------------------------------------

class _CountingTok:
    """Counts how many ids each decode call touches — the O(n^2) regression detector."""
    def __init__(self):
        self.decoded = 0

    def decode(self, ids, skip_special_tokens=True):
        self.decoded += len(ids)
        return "".join(chr(97 + (i % 26)) for i in ids)


def test_incr_detok_is_linear_not_quadratic():
    t = _CountingTok()
    d = gw._IncrDetok(t)
    ids = []
    n = 200
    for i in range(n):
        ids.append(i % 26)
        d.feed(list(ids))
    assert t.decoded <= 6 * n, f"decode touched {t.decoded} ids over {n} commits — O(n^2) re-decode"
    # exactness: the accumulated text equals a one-shot full decode
    assert d.text == "".join(chr(97 + (i % 26)) for i in ids)


def test_incr_detok_holds_partial_char():
    class T:
        def decode(self, ids, skip_special_tokens=True):
            return "".join("x" if i else "�" for i in ids)
    d = gw._IncrDetok(T())
    assert d.feed([1, 0]) == "", "partial UTF-8 must be held back, never streamed"
    assert d.feed([1, 0, 1]) == "x�x", "held bytes flush once the char completes"


def test_splitter_matches_split_stream_reasoning_on():
    from m25_tools import THINK_END, TOOLCALL_BEGIN
    text = f"deep thought{THINK_END}the answer{TOOLCALL_BEGIN}<invoke/>"
    sp = gw._SSESplitter(reasoning_on=True)
    r = c = ""
    for i in range(0, len(text), 3):                      # tiny slices: markers straddle commits
        rd, cd = sp.feed(text[i:i + 3])
        r += rd; c += cd
    rd, cd = sp.end()
    r += rd; c += cd
    want_r, want_c = gw._split_stream(text, reasoning_on=True)
    assert r == want_r and c == want_c


def test_splitter_matches_split_stream_reasoning_off():
    text = "Here is the direct answer."
    sp = gw._SSESplitter(reasoning_on=False)
    r = c = ""
    for ch in text:
        rd, cd = sp.feed(ch)
        r += rd; c += cd
    rd, cd = sp.end()
    r += rd; c += cd
    assert r == "" and c == text


def test_splitter_never_leaks_partial_marker():
    from m25_tools import THINK_END, TOOLCALL_BEGIN
    sp = gw._SSESplitter(reasoning_on=True)
    sp.feed(f"t{THINK_END}answer")
    _, cd = sp.feed(TOOLCALL_BEGIN[:8])                   # half a tool-call marker
    assert "<" not in cd, "partial tool-call marker leaked into content"


def test_stream_on_commit_decodes_incrementally(monkeypatch):
    """End-to-end: the stream path must never re-decode the full output ids on every commit."""
    t = _CountingTok()
    monkeypatch.setattr(gw, "tok", t)
    n = 100
    def fake_run(messages, tools, max_new, reasoning, on_commit=None, timeout=1800):
        ids = []
        for i in range(n):
            ids.append(i % 26)
            on_commit(list(ids), 0.0)
        return {"ok": True, "text": t.decode(ids), "n_tokens": len(ids), "prompt_tokens": 1,
                "output_ids": ids}
    monkeypatch.setattr(gw, "run_request", fake_run)
    monkeypatch.setattr(gw, "MOCK", False)
    wf = _FakeWfile()
    h = _handler(wfile=wf)
    h._stream("cid", 0, [{"role": "user", "content": "hi"}], None, 4096, reasoning=False)
    assert t.decoded <= 8 * n, f"decode touched {t.decoded} ids over {n} commits — O(n^2) stream decode"
    content = "".join(d["delta"].get("content", "") for d in _sse_deltas(wf) if d.get("delta"))
    assert content == t.decode(list(range(n))) == "".join(chr(97 + (i % 26)) for i in range(n))


# ---- H6: identity-bound greetings from _connect ---------------------------------------------------------

class _FS:
    def __init__(self, name):
        self.name = name

    def setsockopt(self, *a):
        pass

    def settimeout(self, t):
        pass

    def close(self):
        pass


def _fake_ring(monkeypatch, sent):
    fake = types.ModuleType("node_kv")
    fake.send_msg = lambda s, m: sent.append((s.name, m))
    fake.recv_msg = lambda s: {"op": "ret_ok"}
    monkeypatch.setitem(sys.modules, "node_kv", fake)
    conns = iter([_FS("pipe"), _FS("ret")])
    monkeypatch.setattr(gw.socket, "create_connection", lambda addr, timeout=None: next(conns))
    gw.SOCKS.clear()


def test_connect_sends_token_greetings(monkeypatch):
    sent = []
    _fake_ring(monkeypatch, sent)
    monkeypatch.setattr(gw, "SWARM_TOKEN", "aabbcc", raising=False)
    gw._connect(5)
    assert ("ret", {"op": "hello_return", "token": "aabbcc"}) in sent
    assert ("pipe", {"op": "hello_pred", "token": "aabbcc"}) in sent


def test_connect_legacy_without_token(monkeypatch):
    sent = []
    _fake_ring(monkeypatch, sent)
    monkeypatch.setattr(gw, "SWARM_TOKEN", None, raising=False)
    gw._connect(5)
    assert sent == [("ret", {"op": "hello_return"})], "unset token must be byte-identical legacy"
