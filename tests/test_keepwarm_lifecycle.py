"""M1 — _KeepWarm lifecycle: stop() must JOIN the noop runner (one lifetime sender per socket) and
the noop thread must never mutate the socket's timeout from the background.

The bug pair: (1) stop() only set a flag, so a runner mid-noop outlived the job on the REUSED
gateway socket and its sendall could interleave with the next job's frames (two _KeepWarm locks on
one socket); (2) _noop_once did settimeout(2.0)-then-restore on the shared socket, racing the job
thread's own recv/settimeout — a recv entered inside the window ran with the noop's 2s deadline.

Run: python3 -m pytest tests/test_keepwarm_lifecycle.py -q
"""
import os
import socket
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
fr = pytest.importorskip("fake_ring")               # bootstraps env (SHARD_TRANSPORT, M25_DIR) + m25_pipe

MP = fr.MP


def _full_pair():
    """A socketpair whose write side is FULL: the peer never reads and both buffers are minimized,
    so any blocking send would block — the stuck-noop scenario."""
    a, b = socket.socketpair()
    a.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    b.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    a.setblocking(False)
    try:
        while True:
            a.send(b"\x00" * 4096)
    except BlockingIOError:
        pass
    a.setblocking(True)
    return a, b


def test_stop_joins_runner():
    """After stop() returns, the noop runner is DEAD — the next job on the same socket can never
    race an in-flight noop send. Before the fix stop() returned with the thread still alive."""
    a, b = _full_pair()
    a.settimeout(30.0)
    kw = MP._KeepWarm(a, interval_ms=20)
    t = kw._runner
    assert t is not None and t.is_alive()
    time.sleep(0.15)                                 # let it enter a noop tick against the full buffer
    kw.stop()
    assert not t.is_alive(), "stop() returned with the noop runner still alive (unjoined sender)"
    a.close(); b.close()


def test_noop_never_mutates_socket_timeout():
    """The job thread owns the socket's timeout. Sample it while a noop tick runs against a FULL
    buffer: the old code set it to 2.0 for the whole blocked-send window; the new code never
    touches it (select-bounded skip)."""
    a, b = _full_pair()
    a.settimeout(37.0)
    seen = set()
    done = threading.Event()

    def _sample():
        while not done.is_set():
            seen.add(a.gettimeout())
            time.sleep(0.005)

    s = threading.Thread(target=_sample, daemon=True)
    s.start()
    kw = MP._KeepWarm(a)
    kw.lock.acquire()
    try:
        kw._noop_once()                              # full buffer: must skip within ~2s, timeout untouched
    finally:
        kw.lock.release()
    done.set(); s.join(timeout=2)
    assert seen == {37.0}, f"noop thread mutated the socket timeout: observed {seen}"
    kw.stop()
    a.close(); b.close()


def test_noop_still_sends_when_writable():
    """The keep-warm still does its job on a healthy leg: a noop frame lands on the peer."""
    a, b = socket.socketpair()
    kw = MP._KeepWarm(a)
    kw.lock.acquire()
    try:
        kw._noop_once()
    finally:
        kw.lock.release()
    b.settimeout(2.0)
    from node_kv import recv_msg
    assert recv_msg(b) == {"op": "noop"}
    kw.stop()
    a.close(); b.close()
