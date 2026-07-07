"""F8 — churn coverage for the REAL serve() tail (m25_pipe.serve, tail branch).

PR #26 hardened the tail so the PREDECESSOR and the coordinator-RETURN channel have INDEPENDENT
lifecycles:
  - an internal-ring leg blip (predecessor edge dies) while the coordinator is ALIVE must KEEP the
    return channel — closing it forced the coordinator into a full reconnect that raced the
    return-tunnel recovery and WEDGED the ring (fatal on a permissionless ring where internal-leg
    blips are the steady state); and
  - a coordinator that reconnects mid-session (a fresh hello_return) must be adopted on the SAME
    warm tail, closing the stale return channel, while the predecessor + warm KV survive.
In both cases the interrupted job's in-flight replies are dropped (the `stale` gate) until the next
reset re-arms the session.

That fix had NO CI coverage: the fake_ring harness MOCKS the tail to exercise the coordinator. This
test drives the REAL serve() tail on CPU (model compute stubbed to identity) over loopback TCP,
playing the coordinator's return + predecessor channels, and asserts the churn state machine.

Run: python3 -m pytest tests/test_tail_churn.py -q
"""
import os
import socket
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
fr = pytest.importorskip("fake_ring")               # bootstraps env (SHARD_TRANSPORT=libp2p, M25_DIR) + m25_pipe

MP = fr.MP
send_msg, recv_msg = fr.send_msg, fr.recv_msg
EDGE = (OSError, EOFError)                            # socket.timeout + peer-closed both land here


class _FakeLayer:
    def reset(self):
        pass


def _patch_cpu_tail(monkeypatch):
    """Stub the tail's model compute so serve()'s tail branch runs on CPU. Only _load/_block/
    _tail_logits are dummied — the CHURN state machine (connection lifecycles, the stale gate,
    _tail_accept bring-up, _ret_send) is all the real code under test."""
    monkeypatch.setattr(MP, "dev", "cpu")
    monkeypatch.setattr(MP, "RECEIPTS", False)
    monkeypatch.setattr(MP, "_load",
                        lambda stage, nstages, lo, hi: {"layers": [_FakeLayer()], "head": False, "tail": True})
    monkeypatch.setattr(MP, "_block", lambda grs, layers, start, x, vcfg: x)   # identity
    monkeypatch.setattr(MP, "_tail_logits", lambda h, parts: h)                # argmax over last dim
    monkeypatch.setattr(MP.S, "_CTX", (None, None), raising=False)             # vcfg = S._CTX[1] = None (unused)
    monkeypatch.setattr(MP.S, "M25_EAGLE", False, raising=False)
    monkeypatch.setattr(MP.S, "M25_STAGE_TIMING", False, raising=False)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_tail(monkeypatch, timeout=5):
    _patch_cpu_tail(monkeypatch)
    port = _free_port()
    t = threading.Thread(target=MP.serve, args=(1, 2, 0, 1, port, "127.0.0.1:1", timeout), daemon=True)
    t.start()
    return port


def _dial(port, timeout=3):
    """Connect with a short retry — serve() binds+listens a hair after the thread starts."""
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


def _connect_ret(port):
    """The coordinator's RETURN channel: greets with hello_return, expects ret_ok."""
    r = _dial(port)
    send_msg(r, {"op": "hello_return"})
    assert recv_msg(r) == "ret_ok"
    return r


def _connect_pred(port):
    """The PREDECESSOR ring stream: silent until its first job byte (adopted once ret exists)."""
    return _dial(port)


def _h(n):
    return torch.zeros(1, n, 4)                       # [1, s, V]; identity _block + argmax -> n-length reply


def _verify(pred, start, n=3):
    send_msg(pred, {"op": "verify", "token_ids": list(range(n)), "start": start, "h": _h(n)})


def _reply(ret, n=3):
    r = recv_msg(ret)
    assert isinstance(r, list) and len(r) == n, f"expected an n={n} token list, got {r!r}"
    return r


def _assert_no_reply(ret, window=0.5):
    ret.settimeout(window)
    with pytest.raises(socket.timeout):              # channel ALIVE but SILENT (stale drop) — NOT closed (EOF).
        recv_msg(ret)                                # a pre-#26 close-ret bug would EOF here (ConnectionError), not time out


# ---- 1. baseline: the harness drives the REAL tail --------------------------------------------------

def test_clean_job_drives_real_tail(monkeypatch):
    """Reset -> ok, several verifies -> per-frame token replies, receipt -> [] : proves the CPU
    harness actually exercises serve()'s tail loop end to end (not a mock)."""
    port = _start_tail(monkeypatch)
    ret = _connect_ret(port)
    pred = _connect_pred(port)
    send_msg(pred, {"op": "reset"})
    assert recv_msg(ret) == "ok"
    for start in (0, 3, 6):
        _verify(pred, start)
        _reply(ret)
    send_msg(pred, {"op": "receipt", "receipts": []})
    assert recv_msg(ret) == []                        # RECEIPTS off -> empty receipts list
    ret.close()
    pred.close()


# ---- 2. THE PR #26 FIX: a predecessor blip must KEEP the return channel + hold the job stale --------

def test_pred_blip_keeps_ret_and_stale_gate(monkeypatch):
    """Internal-ring leg blip while the coordinator is alive: the tail must NOT close the return
    channel (the wedge). After the blip the session is STALE — a verify before the next reset is
    dropped — and the coordinator's next reset re-arms the job on the SAME return channel."""
    port = _start_tail(monkeypatch)
    ret = _connect_ret(port)
    pred = _connect_pred(port)
    send_msg(pred, {"op": "reset"})
    assert recv_msg(ret) == "ok"
    _verify(pred, 0)
    _reply(ret)

    # BLIP: the predecessor edge drops (upstream stage restart / leg hiccup)
    pred.close()
    time.sleep(0.2)                                  # let the tail see EOF, reset KV, re-enter _tail_accept

    # the tail re-accepts a predecessor; the return channel must have SURVIVED
    pred2 = _connect_pred(port)
    time.sleep(0.1)

    # STALE GATE: in-flight traffic before the re-arming reset belongs to the dead job -> dropped
    _verify(pred2, 30)
    _assert_no_reply(ret)

    # the next reset re-arms the session on the SAME (never-closed) return channel
    ret.settimeout(3)
    send_msg(pred2, {"op": "reset"})
    assert recv_msg(ret) == "ok", "return channel did NOT survive the predecessor blip (wedge regression)"
    _verify(pred2, 0)
    _reply(ret)
    ret.close()
    pred2.close()


# ---- 3. coordinator reconnect: a mid-session hello_return adopts a new ret, keeps pred + KV ---------

def test_coordinator_reconnect_adopts_new_ret_keeps_pred(monkeypatch):
    """A reconnecting coordinator greets the warm tail with a fresh hello_return mid-session. The tail
    must adopt it, CLOSE the stale return channel, and keep the predecessor + warm KV — the next reset
    re-arms on the new channel."""
    port = _start_tail(monkeypatch)
    ret = _connect_ret(port)
    pred = _connect_pred(port)
    send_msg(pred, {"op": "reset"})
    assert recv_msg(ret) == "ok"
    _verify(pred, 0)
    _reply(ret)

    # coordinator churn: a NEW return channel connects and greets mid-session
    ret2 = _connect_ret(port)                        # sends hello_return, gets ret_ok on the new channel

    # the OLD return channel must be closed by the tail
    ret.settimeout(1)
    with pytest.raises(EDGE):
        recv_msg(ret)                                # EOF: the stale return channel was closed

    # predecessor + warm KV survive the swap; the next reset re-arms on ret2
    send_msg(pred, {"op": "reset"})
    assert recv_msg(ret2) == "ok", "predecessor did not survive the coordinator (ret) swap"
    _verify(pred, 0)
    _reply(ret2)
    ret2.close()
    pred.close()
