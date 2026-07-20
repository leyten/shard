"""P0-#5 / M1 — the tail's coordinator-return send stall guard (m25_pipe._ret_stall_s).

The ret socket was UNTIMED: a coordinator-side wedge (dead conntrack on a relay path change, a
receive buffer that never drains) left the tail blocked inside _ret_send's send — unable to
select, accept a fresh hello_return, or see the next reset. One stuck send wedged the whole warm
ring: the strongest candidate for the 2026-07-14 residential-tail EAGLE hang (EAGLE's aux frames
are the big return payload; the CPU fake ring's infinite-bandwidth loopback is exactly why it
never reproduced there).

The guard: ret gets settimeout(M25_RET_STALL_S). With the libp2p transport the send loop is
per-sendmsg-call (shard/transport._sendall_vectored), so the timeout means "ZERO bytes accepted
for this long" — a slow-but-draining uplink resets the clock with every accepted chunk and never
trips, however big the frame. On trip: socket.timeout -> _ret_send's existing EDGE absorb drops
ret, keeps predecessor+KV, marks the job stale; the next coordinator re-adopts mid-session.

Run: python3 -m pytest tests/test_ret_stallguard.py -q
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

from shard.transport import _sendall_vectored       # noqa: E402  (the per-progress send loop under test)


# ---- 1. the transport semantics the guard relies on ------------------------------------------------

def _slow_reader(sock, chunk=65536, every=0.1, stop=None):
    """Drain `chunk` bytes every `every` seconds — a slow but LIVE peer."""
    def run():
        sock.settimeout(5)
        while stop is None or not stop.is_set():
            try:
                if not sock.recv(chunk):
                    return
            except OSError:
                return
            time.sleep(every)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def test_vectored_send_timeout_is_per_progress():
    """A 4MB frame to a slow-but-draining reader under a 0.3s timeout COMPLETES: every accepted
    chunk resets the clock. This is the property that lets a 180s stall bound coexist with
    multi-minute (but moving) residential aux sends."""
    a, b = socket.socketpair()
    a.settimeout(0.3)
    a.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
    b.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
    t = _slow_reader(b, chunk=262144, every=0.1)    # ~2.6MB/s: total send time >> the 0.3s timeout
    _sendall_vectored(a, [b"x" * (4 * 1024 * 1024)])
    a.close(); b.close(); t.join(1)


def test_vectored_send_timeout_trips_on_zero_progress():
    """The same frame to a reader that NEVER drains trips socket.timeout in ~the stall bound —
    not the multi-minute production timeout, and not never."""
    a, b = socket.socketpair()
    a.settimeout(0.3)
    a.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
    b.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
    t0 = time.monotonic()
    with pytest.raises(socket.timeout):
        _sendall_vectored(a, [b"x" * (4 * 1024 * 1024)])
    assert time.monotonic() - t0 < 2.0
    a.close(); b.close()


# ---- 2. the REAL serve() tail survives a wedged ret -------------------------------------------------

class _FakeLayer:
    def reset(self):
        pass


def _patch_cpu_tail(monkeypatch, eagle=False):
    monkeypatch.setattr(MP, "dev", "cpu")
    monkeypatch.setattr(MP, "RECEIPTS", False)
    monkeypatch.setattr(MP, "_load",
                        lambda stage, nstages, lo, hi: {"layers": [_FakeLayer()], "head": False, "tail": True})
    monkeypatch.setattr(MP, "_block", lambda grs, layers, start, x, vcfg: x)
    monkeypatch.setattr(MP, "_tail_logits", lambda h, parts: h)
    monkeypatch.setattr(MP.S, "_CTX", (None, None), raising=False)
    monkeypatch.setattr(MP.S, "_AUX", {}, raising=False)     # _merge_aux passes upstream aux through
    monkeypatch.setattr(MP.S, "M25_EAGLE", eagle, raising=False)
    monkeypatch.setattr(MP.S, "M25_STAGE_TIMING", False, raising=False)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_tail(monkeypatch, eagle=False, timeout=5):
    _patch_cpu_tail(monkeypatch, eagle=eagle)
    port = _free_port()
    t = threading.Thread(target=MP.serve, args=(1, 2, 0, 1, port, "127.0.0.1:1", timeout), daemon=True)
    t.start()
    return port


def _dial(port, timeout=3):
    deadline = time.monotonic() + timeout
    while True:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
            s.settimeout(timeout)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return s
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.02)


def _connect_ret(port, rcvbuf=None):
    r = _dial(port)
    if rcvbuf:
        r.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
    send_msg(r, {"op": "hello_return"})
    assert recv_msg(r) == "ret_ok"
    return r


def test_tail_survives_wedged_ret_and_readopts(monkeypatch):
    """A coordinator that STOPS READING mid-job (the M1 wedge: EAGLE's fat aux reply against a
    dead/blocked return path) must not hold the tail's loop forever: the stall guard trips inside
    _ret_send, the tail drops ret (keeps predecessor+KV, job stale), and a FRESH hello_return +
    reset re-arms the session and serves again. Before the guard the ret was untimed — this test
    hangs its 8s timeout inside the first big send."""
    monkeypatch.setenv("M25_RET_STALL_S", "0.4")
    # stage timeout 30s >> the 8s aliveness deadline: only the 0.4s stall guard can explain a fast
    # recovery (with the guard off, _tail_accept's greeting timeout would otherwise mask it)
    port = _start_tail(monkeypatch, eagle=True, timeout=30)
    ret = _connect_ret(port, rcvbuf=32768)          # tiny receive buffer: the wedge fills it fast
    pred = _dial(port)
    send_msg(pred, {"op": "reset"})
    assert recv_msg(ret) == "ok"

    # wedge: stop reading ret entirely, then drive verifies whose EAGLE aux replies (~4MB each)
    # overflow the socket buffers — the tail's send must STALL, trip, and drop ret
    big_aux = {"1": torch.zeros(1, 512 * 1024, dtype=torch.float32)}   # ~2MB upstream aux, passed through
    t0 = time.monotonic()
    for start in (0, 3, 6, 9):
        send_msg(pred, {"op": "verify", "token_ids": [1, 2, 3], "start": start,
                        "h": torch.zeros(1, 3, 4), "aux": big_aux})
    # the tail is alive iff it can re-adopt a NEW return channel within a few stall periods
    deadline = time.monotonic() + 8
    ret2 = None
    while ret2 is None:
        assert time.monotonic() < deadline, "tail never re-accepted a return channel — still wedged in send"
        try:
            ret2 = _connect_ret(port)
        except (OSError, AssertionError, EOFError, ConnectionError):
            time.sleep(0.1)
    trip_s = time.monotonic() - t0
    assert trip_s < 8, f"re-adoption took {trip_s:.1f}s"

    # stale until the next reset: pre-reset verifies are dropped, a reset re-arms and serves
    send_msg(pred, {"op": "reset"})
    assert recv_msg(ret2) == "ok"
    send_msg(pred, {"op": "verify", "token_ids": [1, 2, 3], "start": 0, "h": torch.zeros(1, 3, 4)})
    r = recv_msg(ret2)
    assert isinstance(r, dict) and len(r["toks"]) == 3   # M25_EAGLE tail replies {toks, aux}
    for s in (ret, ret2, pred):
        try:
            s.close()
        except OSError:
            pass


def test_ret_stall_disabled_keeps_untimed_behavior(monkeypatch):
    """M25_RET_STALL_S=0 = the escape hatch: no stall bound is installed and a healthy job flows
    exactly as before (pins the 0-disables convention; the wedge case is then master behavior)."""
    monkeypatch.setenv("M25_RET_STALL_S", "0")
    port = _start_tail(monkeypatch, eagle=False)
    ret = _connect_ret(port)
    pred = _dial(port)
    send_msg(pred, {"op": "reset"})
    assert recv_msg(ret) == "ok"
    for start in (0, 3):
        send_msg(pred, {"op": "verify", "token_ids": [1, 2, 3], "start": start, "h": torch.zeros(1, 3, 4)})
        r = recv_msg(ret)
        assert isinstance(r, list) and len(r) == 3
    for s in (ret, pred):
        s.close()
