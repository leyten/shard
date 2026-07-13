"""C2/H6 — identity-bound role greetings. With SHARD_SWARM_TOKEN set (minted per launch by the
launcher), the tail classifies connections ONLY by an explicit token-bearing greeting
({op: hello_return|hello_pred, token}) — the old rules ("a silent connection is the predecessor",
"any op-carrying first frame is a new predecessor", incl. the H6 mid-session variant) let any peer
that could reach the port be adopted into the ring. Head/middle stages likewise require a
hello_pred greeting before any job frame, and every sender (stage _dial_fwd, coordinator) opens
with one. Token unset = exact legacy behavior (scenario v).

Run: python3 -m pytest tests/test_swarm_token_auth.py -q
"""
import os
import socket
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
fr = pytest.importorskip("fake_ring")               # bootstraps env + m25_pipe on CPU

MP = fr.MP
send_msg, recv_msg = fr.send_msg, fr.recv_msg

TOK = "a" * 32


def _srv():
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    s.listen(8)
    return s, s.getsockname()[1]


def _accept_thread(srv, **kw):
    out = {}

    def _run():
        out["res"] = MP._tail_accept(srv, timeout=3, **kw)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t, out


def _eof(sock, window=3.0):
    sock.settimeout(window)
    with pytest.raises((OSError, EOFError)):
        recv_msg(sock)


# ---- _tail_accept, token mode ------------------------------------------------------------------------

def test_greetings_classify_ret_and_pred(monkeypatch):
    """(i)+(ii): hello_return+token -> ret (ret_ok acked); hello_pred+token -> pred, first_msg=None."""
    monkeypatch.setattr(MP, "SWARM_TOKEN", TOK)
    srv, port = _srv()
    t, out = _accept_thread(srv)
    c_ret = socket.create_connection(("127.0.0.1", port), timeout=3)
    send_msg(c_ret, {"op": "hello_return", "token": TOK})
    c_ret.settimeout(3)
    assert recv_msg(c_ret) == "ret_ok"
    c_pred = socket.create_connection(("127.0.0.1", port), timeout=3)
    send_msg(c_pred, {"op": "hello_pred", "token": TOK})
    t.join(timeout=5)
    assert not t.is_alive()
    ret, pred, first = out["res"]
    assert first is None                             # the greeting was consumed, not handed back as job data
    assert ret.getpeername() == c_ret.getsockname()
    assert pred.getpeername() == c_pred.getsockname()
    for s in (ret, pred, c_ret, c_pred, srv):
        s.close()


def test_wrong_token_rejected(monkeypatch):
    """(iii): a valid-looking hello_return with the WRONG token is closed and never adopted."""
    monkeypatch.setattr(MP, "SWARM_TOKEN", TOK)
    srv, port = _srv()
    t, out = _accept_thread(srv)
    bad = socket.create_connection(("127.0.0.1", port), timeout=3)
    send_msg(bad, {"op": "hello_return", "token": "b" * 32})
    _eof(bad)                                        # AUTH REJECT: closed
    # satisfy the accept with genuine peers; the impostor's socket must not be either role
    c_ret = socket.create_connection(("127.0.0.1", port), timeout=3)
    send_msg(c_ret, {"op": "hello_return", "token": TOK})
    c_pred = socket.create_connection(("127.0.0.1", port), timeout=3)
    send_msg(c_pred, {"op": "hello_pred", "token": TOK})
    t.join(timeout=5)
    assert not t.is_alive()
    ret, pred, _ = out["res"]
    assert ret.getpeername() == c_ret.getsockname()
    assert pred.getpeername() == c_pred.getsockname()
    for s in (ret, pred, c_ret, c_pred, bad, srv):
        s.close()


def test_bare_job_frame_rejected(monkeypatch):
    """Token mode kills the 'first job frame IS the greeting' hand-back: a bare verify frame is an
    AUTH REJECT, not a predecessor adoption."""
    monkeypatch.setattr(MP, "SWARM_TOKEN", TOK)
    srv, port = _srv()
    t, out = _accept_thread(srv)
    intruder = socket.create_connection(("127.0.0.1", port), timeout=3)
    send_msg(intruder, {"op": "verify", "token_ids": [1, 2, 3], "start": 0})
    _eof(intruder)
    c_ret = socket.create_connection(("127.0.0.1", port), timeout=3)
    send_msg(c_ret, {"op": "hello_return", "token": TOK})
    c_pred = socket.create_connection(("127.0.0.1", port), timeout=3)
    send_msg(c_pred, {"op": "hello_pred", "token": TOK})
    t.join(timeout=5)
    assert not t.is_alive()
    for s in (*out["res"][:2], c_ret, c_pred, intruder, srv):
        s.close()


def test_silent_conn_never_adopted_reaped(monkeypatch):
    """(iv): a silent connection is NEVER adopted as predecessor and is closed once it outlives
    M25_GREET_TIMEOUT — the silence-adoption rule is dead in token mode."""
    monkeypatch.setattr(MP, "SWARM_TOKEN", TOK)
    monkeypatch.setattr(MP, "GREET_TIMEOUT", 0.4)
    srv, port = _srv()
    t, out = _accept_thread(srv)
    c_ret = socket.create_connection(("127.0.0.1", port), timeout=3)
    send_msg(c_ret, {"op": "hello_return", "token": TOK})
    c_ret.settimeout(3)
    assert recv_msg(c_ret) == "ret_ok"
    lurker = socket.create_connection(("127.0.0.1", port), timeout=3)   # ret live + silent conn: the
    _eof(lurker, window=4.0)                                            # legacy rule would adopt it
    assert t.is_alive()                              # still waiting for a REAL (greeting) predecessor
    c_pred = socket.create_connection(("127.0.0.1", port), timeout=3)
    send_msg(c_pred, {"op": "hello_pred", "token": TOK})
    t.join(timeout=5)
    assert not t.is_alive()
    _, pred, _ = out["res"]
    assert pred.getpeername() == c_pred.getsockname()
    for s in (*out["res"][:2], c_ret, c_pred, lurker, srv):
        s.close()


def test_legacy_classification_preserved_without_token():
    """(v): token unset -> byte-for-byte legacy: hello_return -> ret, a job-frame-first conn -> pred
    with the frame handed back, and a silent conn + live ret -> pred."""
    assert MP.SWARM_TOKEN is None                    # test env never sets SHARD_SWARM_TOKEN
    # (v-a) job-frame-first -> pred + first (the frame must land BEFORE a ret exists, else the
    # legacy silent-adopt rule wins the race — as it always has on the live tail)
    srv, port = _srv()
    t, out = _accept_thread(srv)
    c_pred = socket.create_connection(("127.0.0.1", port), timeout=3)
    frame = {"op": "verify", "token_ids": [7], "start": 0}
    send_msg(c_pred, frame)
    time.sleep(0.3)                                  # let the frame be read + classified
    c_ret = socket.create_connection(("127.0.0.1", port), timeout=3)
    send_msg(c_ret, {"op": "hello_return"})
    c_ret.settimeout(3)
    assert recv_msg(c_ret) == "ret_ok"
    t.join(timeout=5)
    assert not t.is_alive()
    ret, pred, first = out["res"]
    assert first == frame and pred.getpeername() == c_pred.getsockname()
    for s in (ret, pred, c_ret, c_pred, srv):
        s.close()
    # (v-b) silent conn + live ret -> pred
    srv, port = _srv()
    t, out = _accept_thread(srv)
    c_ret = socket.create_connection(("127.0.0.1", port), timeout=3)
    send_msg(c_ret, {"op": "hello_return"})
    c_ret.settimeout(3)
    assert recv_msg(c_ret) == "ret_ok"
    c_silent = socket.create_connection(("127.0.0.1", port), timeout=3)
    t.join(timeout=5)
    assert not t.is_alive()
    ret, pred, first = out["res"]
    assert first is None and pred.getpeername() == c_silent.getsockname()
    for s in (ret, pred, c_ret, c_silent, srv):
        s.close()


# ---- real serve(): head/middle greeting gate + _dial_fwd sends the greeting -------------------------

class _FakeLayer:
    def reset(self):
        pass


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _dial(port, timeout=3):
    deadline = time.monotonic() + timeout
    while True:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
            s.settimeout(timeout)
            return s
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.02)


def test_middle_stage_gate_and_dial_greeting(monkeypatch):
    """A middle stage (token mode) greets DOWNSTREAM on dial (_dial_fwd's first frame is hello_pred)
    and requires the same greeting from its own predecessor before any job frame."""
    monkeypatch.setattr(MP, "SWARM_TOKEN", TOK)
    monkeypatch.setattr(MP, "dev", "cpu")
    monkeypatch.setattr(MP, "RECEIPTS", False)
    monkeypatch.setattr(MP, "_load",
                        lambda stage, nstages, lo, hi: {"layers": [_FakeLayer()], "head": False, "tail": False})
    monkeypatch.setattr(MP, "_block", lambda grs, layers, start, x, vcfg: x)
    monkeypatch.setattr(MP.S, "_CTX", (None, None), raising=False)
    monkeypatch.setattr(MP.S, "M25_EAGLE", False, raising=False)
    monkeypatch.setattr(MP.S, "M25_STAGE_TIMING", False, raising=False)
    nxt_srv, nxt_port = _srv()
    port = _free_port()
    threading.Thread(target=MP.serve, args=(1, 3, 0, 1, port, f"127.0.0.1:{nxt_port}", 3),
                     daemon=True).start()
    fwd, _ = nxt_srv.accept()
    fwd.settimeout(5)
    assert recv_msg(fwd) == {"op": "hello_pred", "token": TOK}   # _dial_fwd greeted downstream

    intruder = _dial(port)                           # no greeting -> the job frame is an AUTH REJECT
    send_msg(intruder, {"op": "verify", "h": torch.zeros(1, 2, 4), "start": 0})
    _eof(intruder)

    pred = _dial(port)                               # greeting first -> served
    send_msg(pred, {"op": "hello_pred", "token": TOK})
    send_msg(pred, {"op": "verify", "h": torch.zeros(1, 2, 4), "start": 0})
    out = recv_msg(fwd)
    assert out["op"] == "verify" and out["start"] == 0
    assert "token" not in out                        # the token never rides a forwarded frame
    for s in (intruder, pred, fwd, nxt_srv):
        try:
            s.close()
        except OSError:
            pass


def test_tail_token_session_end_to_end(monkeypatch):
    """The real serve() tail in token mode: greeted ret+pred run a job; an unauthenticated
    mid-session hello_return is rejected and the live session is untouched."""
    monkeypatch.setattr(MP, "SWARM_TOKEN", TOK)
    monkeypatch.setattr(MP, "dev", "cpu")
    monkeypatch.setattr(MP, "RECEIPTS", False)
    monkeypatch.setattr(MP, "_load",
                        lambda stage, nstages, lo, hi: {"layers": [_FakeLayer()], "head": False, "tail": True})
    monkeypatch.setattr(MP, "_block", lambda grs, layers, start, x, vcfg: x)
    monkeypatch.setattr(MP, "_tail_logits", lambda h, parts: h)
    monkeypatch.setattr(MP.S, "_CTX", (None, None), raising=False)
    monkeypatch.setattr(MP.S, "M25_EAGLE", False, raising=False)
    monkeypatch.setattr(MP.S, "M25_STAGE_TIMING", False, raising=False)
    port = _free_port()
    threading.Thread(target=MP.serve, args=(1, 2, 0, 1, port, "127.0.0.1:1", 3), daemon=True).start()
    ret = _dial(port)
    send_msg(ret, {"op": "hello_return", "token": TOK})
    assert recv_msg(ret) == "ret_ok"
    pred = _dial(port)
    send_msg(pred, {"op": "hello_pred", "token": TOK})
    send_msg(pred, {"op": "reset"})
    assert recv_msg(ret) == "ok"
    send_msg(pred, {"op": "verify", "h": torch.zeros(1, 3, 4), "start": 0})
    assert isinstance(recv_msg(ret), list)

    hijack = _dial(port)                             # tokenless coordinator-churn attempt (the H6 hole)
    send_msg(hijack, {"op": "hello_return"})
    _eof(hijack)                                     # rejected — and the REAL session still works:
    send_msg(pred, {"op": "verify", "h": torch.zeros(1, 3, 4), "start": 3})
    assert isinstance(recv_msg(ret), list)
    for s in (ret, pred, hijack):
        s.close()
