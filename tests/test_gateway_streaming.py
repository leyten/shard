"""TIER-3 gateway hardening (phase0/m25_gateway.py) — streaming + client-failure robustness.

Three fixes under test:
  - reasoning=False no longer DUPLICATES the answer (whole thing as reasoning_content, then re-flushed
    as content): _split_stream is reasoning-aware.
  - a client disconnect / stall mid-stream raises ClientGone (a non-OSError), so generate() ABORTS
    instead of the old behaviour where the OSError was absorbed as a ring fault and the ENTIRE
    generation re-ran.
  - a stalled streaming client is bounded by a write timeout (set on the connection) instead of
    pinning the single-stream ring.

Driven with the built-in MOCK engine (no GPU) + a fake request socket. Run:
    M25_GATEWAY_MOCK=1 python3 -m pytest tests/test_gateway_streaming.py -q
"""
import json
import os
import sys
import types

import pytest

os.environ["M25_GATEWAY_MOCK"] = "1"                    # MOCK is read at import -> set before importing
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase0"))

import m25_gateway as gw                                # noqa: E402
from m25_tools import THINK_END, TOOLCALL_BEGIN         # noqa: E402


# ---- 1. _split_stream: the reasoning=False duplication fix ------------------------------------------

def test_split_reasoning_on_with_think():
    r, c = gw._split_stream(f"pondering{THINK_END}the answer", reasoning_on=True)
    assert r == "pondering" and c == "the answer"


def test_split_reasoning_on_no_think_is_all_reasoning():
    r, c = gw._split_stream("still thinking, no close yet", reasoning_on=True)
    assert r == "still thinking, no close yet" and c == ""


def test_split_reasoning_off_no_think_is_all_content():
    """THE fix: with reasoning off the output has no </think>; it must be content, not reasoning
    (else it streams as reasoning_content AND gets re-flushed as content = duplicate answer)."""
    r, c = gw._split_stream("Here is the direct answer.", reasoning_on=False)
    assert r == "" and c == "Here is the direct answer."


def test_split_reasoning_off_strips_echoed_think_and_toolcall():
    r, c = gw._split_stream(f"junk{THINK_END}real answer{TOOLCALL_BEGIN}<invoke.../>", reasoning_on=False)
    assert r == "" and c == "real answer"


def test_split_content_stops_before_toolcall_when_reasoning_on():
    r, c = gw._split_stream(f"think{THINK_END}answer{TOOLCALL_BEGIN}<xml>", reasoning_on=True)
    assert r == "think" and c == "answer"


# ---- 2. streaming end to end via MOCK + a fake request socket --------------------------------------

class _FakeWfile:
    """Captures SSE bytes; optionally fails after N writes to model a client disconnect."""
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


def _handler(wfile, conn=None):
    """An H instance wired to a fake socket, bypassing BaseHTTPRequestHandler.__init__ (which would
    run the whole request cycle). Header writes are stubbed; only the SSE body is captured."""
    h = gw.H.__new__(gw.H)
    h.wfile = wfile
    h.connection = conn or _FakeConn()
    h.close_connection = False
    h.send_response = lambda *a, **k: None
    h.send_header = lambda *a, **k: None
    h.end_headers = lambda: None
    return h


def _sse_deltas(wfile):
    """Parse the captured SSE stream into the list of choice deltas (skips [DONE] + usage frames)."""
    out = []
    for raw in wfile.chunks:
        line = raw.decode()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):].strip()
        if payload == "[DONE]":
            continue
        obj = json.loads(payload)
        if obj.get("choices"):
            out.append(obj["choices"][0].get("delta", {}))
    return out


def test_stream_reasoning_off_has_no_duplicate():
    """reasoning=False: the answer streams once as content, and NOTHING as reasoning_content."""
    wf = _FakeWfile()
    h = _handler(wf)
    h._stream("cid", 0, [{"role": "user", "content": "hello"}], None, 64, reasoning=False)
    deltas = _sse_deltas(wf)
    content = "".join(d.get("content", "") for d in deltas)
    reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
    assert reasoning == "", f"reasoning leaked with reasoning off: {reasoning!r}"
    assert "concise answer" in content
    assert content.count("concise answer") == 1, "answer duplicated"


def test_stream_reasoning_on_splits_reasoning_and_content():
    wf = _FakeWfile()
    h = _handler(wf)
    h._stream("cid", 0, [{"role": "user", "content": "hello"}], None, 64, reasoning=True)
    deltas = _sse_deltas(wf)
    reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
    content = "".join(d.get("content", "") for d in deltas)
    assert "Thinking about it" in reasoning
    assert "concise answer" in content and "Thinking about it" not in content


def test_stream_sets_write_timeout():
    wf = _FakeWfile()
    conn = _FakeConn()
    h = _handler(wf, conn)
    h._stream("cid", 0, [{"role": "user", "content": "hi"}], None, 64, reasoning=False)
    assert conn.timeout == gw.STREAM_WRITE_TIMEOUT, "stalled-client write timeout not armed"


def test_stream_client_disconnect_raises_client_gone():
    """A mid-stream write failure surfaces as ClientGone (not a bare OSError), so the caller aborts
    rather than the ring treating it as an edge fault and re-running."""
    wf = _FakeWfile(fail_after=1)                      # role chunk ok, first real delta fails
    h = _handler(wf)
    with pytest.raises(gw.ClientGone):
        h._stream("cid", 0, [{"role": "user", "content": "hello"}], None, 64, reasoning=False)


# ---- 3. generate(): ClientGone is never retried; a ring error IS retried ---------------------------

def _fake_engine(monkeypatch, cp):
    monkeypatch.setattr(gw, "MOCK", False)
    monkeypatch.setattr(gw, "coordinate_pipe", cp)
    monkeypatch.setattr(gw, "make_drafter", lambda n: object())
    monkeypatch.setattr(gw, "_connect", lambda t: gw.SOCKS.update(pipe=object(), ret=object()))
    dropped = {"n": 0}
    monkeypatch.setattr(gw, "_drop_socks", lambda: (gw.SOCKS.clear(), dropped.__setitem__("n", dropped["n"] + 1)))
    monkeypatch.setattr(gw, "A", types.SimpleNamespace(ngram_n=3, K=8, depth=4, max_ctx=1000))
    gw.SOCKS.clear()
    return dropped


def test_generate_does_not_retry_on_client_gone(monkeypatch):
    calls = []
    def cp(*a, **k):
        calls.append(1)
        raise gw.ClientGone("client vanished mid-decode")
    dropped = _fake_engine(monkeypatch, cp)
    with pytest.raises(gw.ClientGone):
        gw.generate([{"role": "user", "content": "hi"}], None, 32, on_commit=None)
    assert calls == [1], "ClientGone must abort — never re-run the generation"
    assert dropped["n"] >= 1, "aborted job must drop the desynced ring sockets"


def test_generate_retries_once_on_ring_error(monkeypatch):
    calls = []
    def cp(*a, **k):
        calls.append(1)
        raise RuntimeError("ring edge died")
    _fake_engine(monkeypatch, cp)
    with pytest.raises(RuntimeError):
        gw.generate([{"role": "user", "content": "hi"}], None, 32, on_commit=None)
    assert len(calls) == 2, "a genuine ring error should reconnect + retry once"
