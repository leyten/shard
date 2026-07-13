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
