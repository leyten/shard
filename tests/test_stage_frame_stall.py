"""H4 (python side) — a head/middle stage's accepted predecessor socket had NO timeout, so a peer
that stalled MID-frame (sent a partial length prefix / partial body, then nothing) held the stage's
only loop forever. The fix mirrors the tail: the accepted socket gets settimeout(timeout) which only
bounds a mid-frame stall (surfacing as socket.timeout -> EDGE_ERRORS -> the existing edge recovery),
while PRE-frame idle stays unlimited via a select wait (_recv_pred) — a warm ring parks between jobs
indefinitely and must never be torn down for being idle.

Run: python3 -m pytest tests/test_stage_frame_stall.py -q
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


# ---- helper level: _recv_pred semantics -------------------------------------------------------------

def test_recv_pred_mid_frame_stall_raises():
    """A partial frame followed by silence raises socket.timeout (an EDGE error) within ~timeout."""
    a, b = socket.socketpair()
    b.settimeout(0.3)
    a.sendall(b"\x00\x00\x00")                       # 3 of the 8 length-prefix bytes, then nothing
    t0 = time.monotonic()
    with pytest.raises(socket.timeout):
        MP._recv_pred(b)
    assert time.monotonic() - t0 < 2.0
    a.close(); b.close()


def test_recv_pred_pre_frame_idle_unlimited():
    """Idle far past the socket timeout, THEN a complete frame: must be delivered, not timed out —
    the select wait (not the socket timeout) governs pre-frame idle."""
    a, b = socket.socketpair()
    b.settimeout(0.3)

    def _late_send():
        time.sleep(1.0)                              # > 3x the socket timeout
        send_msg(a, {"op": "verify", "token_ids": [1, 2], "start": 0})

    t = threading.Thread(target=_late_send, daemon=True)
    t.start()
    msg = MP._recv_pred(b)
    assert msg["op"] == "verify" and msg["token_ids"] == [1, 2]
    t.join(); a.close(); b.close()


# ---- serve() level: the REAL middle stage recovers from a stalled predecessor -----------------------

class _FakeLayer:
    def reset(self):
        pass


def _start_middle(monkeypatch, timeout=1):
    """Real serve() middle stage on CPU (compute stubbed to identity), forward link to a local
    listener we play. Returns (pred_port, nxt_srv)."""
    monkeypatch.setattr(MP, "dev", "cpu")
    monkeypatch.setattr(MP, "RECEIPTS", False)
    monkeypatch.setattr(MP, "_load",
                        lambda stage, nstages, lo, hi: {"layers": [_FakeLayer()], "head": False, "tail": False})
    monkeypatch.setattr(MP, "_block", lambda grs, layers, start, x, vcfg: x)
    monkeypatch.setattr(MP.S, "_CTX", (None, None), raising=False)
    monkeypatch.setattr(MP.S, "M25_EAGLE", False, raising=False)
    monkeypatch.setattr(MP.S, "M25_STAGE_TIMING", False, raising=False)
    nxt_srv = socket.socket()
    nxt_srv.bind(("127.0.0.1", 0)); nxt_srv.listen(2)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0)); port = srv.getsockname()[1]; srv.close()
    t = threading.Thread(target=MP.serve,
                         args=(1, 3, 0, 1, port, f"127.0.0.1:{nxt_srv.getsockname()[1]}", timeout),
                         daemon=True)
    t.start()
    return port, nxt_srv


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


def test_middle_stage_survives_mid_frame_stall(monkeypatch):
    """A predecessor that wedges half-way through a frame must NOT hold the stage forever: the stage
    times the stall out, drops the edge (we see EOF), and re-accepts a fresh predecessor that can
    drive it again. Before the fix the accepted socket had no timeout and this hung."""
    port, nxt_srv = _start_middle(monkeypatch, timeout=1)
    pred = _dial(port)
    fwd, _ = nxt_srv.accept()                        # serve() dialed its forward link at boot
    fwd.settimeout(5)

    pred.sendall(b"\x00\x00\x00\x00")                # half a length prefix, then wedge
    pred.settimeout(4)
    with pytest.raises((OSError, EOFError)):         # stage recovery closes the edge -> we see EOF
        recv_msg(pred)                               # (no timeout on the stage side = this hangs 4s and fails)

    # the stage re-accepts and serves a fresh predecessor on a rebuilt forward link
    pred2 = _dial(port)
    fwd2, _ = nxt_srv.accept()
    fwd2.settimeout(5)
    send_msg(pred2, {"op": "verify", "h": torch.zeros(1, 2, 4), "start": 0})
    out = recv_msg(fwd2)
    assert out["op"] == "verify" and out["start"] == 0
    for s in (pred, pred2, fwd, fwd2, nxt_srv):
        try:
            s.close()
        except OSError:
            pass
