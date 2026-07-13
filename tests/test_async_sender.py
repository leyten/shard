"""_AsyncSender shutdown (specpipe): the close sentinel must survive a FULL queue.

The old close() was `q.put_nowait(None)` with the exception swallowed: when the send queue was
full (exactly the wedged-forward-link case close() runs in during an edge reset), the sentinel
was silently dropped and the daemon worker was stranded forever on q.get(). The fix is an
explicit close event + bounded join; these tests drive a worker wedged mid-send with a full
queue and assert the thread actually exits.

Run: pytest tests/test_async_sender.py -q   (CPU-only, no sockets — send_msg is monkeypatched)
"""
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase0"))

import specpipe


def _wedged_sender(monkeypatch):
    """An _AsyncSender whose worker is blocked inside send_msg until `release` is set,
    with the queue filled to maxsize behind it (the wedged-link state at edge reset)."""
    release = threading.Event()
    in_send = threading.Event()

    def slow_send(sock, obj):
        in_send.set()
        release.wait(10.0)

    monkeypatch.setattr(specpipe, "send_msg", slow_send)
    s = specpipe._AsyncSender(sock=None)
    s.put("wedge")                                   # worker takes it and blocks in send_msg
    assert in_send.wait(5.0)
    for i in range(s.q.maxsize):                     # fill the queue completely behind it
        s.q.put(i)
    assert s.q.full()
    return s, release


def test_close_on_full_queue_does_not_strand_worker(monkeypatch):
    """THE regression: queue full at close() -> old code dropped the sentinel and the worker
    lived forever after the wedge cleared. The worker must exit once the in-flight send returns."""
    s, release = _wedged_sender(monkeypatch)
    t0 = time.monotonic()
    s.close(timeout=0.2)                             # bounded: returns even though the send is wedged
    assert time.monotonic() - t0 < 2.0
    release.set()                                    # the wedge clears (peer drained / socket died)
    s.t.join(5.0)
    assert not s.t.is_alive(), "worker stranded after close() with a full queue"


def test_close_bounded_while_send_still_wedged(monkeypatch):
    """close() must never hang the serve loop's reset path behind a stalled send."""
    s, release = _wedged_sender(monkeypatch)
    t0 = time.monotonic()
    s.close(timeout=0.3)
    assert time.monotonic() - t0 < 2.0               # returned while the worker is still in send_msg
    release.set()
    s.t.join(5.0)
    assert not s.t.is_alive()


def test_put_after_close_raises(monkeypatch):
    monkeypatch.setattr(specpipe, "send_msg", lambda sock, obj: None)
    s = specpipe._AsyncSender(sock=None)
    s.close()
    with pytest.raises(RuntimeError):
        s.put("late")


def test_close_idempotent_and_clean_exit(monkeypatch):
    sent = []
    monkeypatch.setattr(specpipe, "send_msg", lambda sock, obj: sent.append(obj))
    s = specpipe._AsyncSender(sock=None)
    s.put("a")
    time.sleep(0.1)
    s.close()
    s.close()                                        # second close is a no-op, never raises
    assert not s.t.is_alive()
    assert sent == ["a"]


def test_send_error_still_surfaces_on_put(monkeypatch):
    """Regression guard: the existing error-propagation contract is preserved."""
    def boom(sock, obj):
        raise ConnectionResetError("peer died")

    monkeypatch.setattr(specpipe, "send_msg", boom)
    s = specpipe._AsyncSender(sock=None)
    s.put("x")
    for _ in range(100):
        if s.error is not None:
            break
        time.sleep(0.02)
    with pytest.raises(ConnectionResetError):
        s.put("y")
    s.close()
    assert not s.t.is_alive()
