"""P0-#5 — the reset op's `eagle` field and the stages' session honor.

A degraded coordinator (EAGLE off after the watchdog / the L2 retry) stamps eagle:0 on its reset;
every stage then silences its aux payload for that session — the degraded arm equals the proven
plain ring ON THE WIRE (aux is the dominant EAGLE return payload and the prime root-cause suspect
for the 2026-07-14 wedge), not just coordinator-side. Field absent = old coordinator = the stage's
launch env decides (bit-compat with every deployed build).

Run: python3 -m pytest tests/test_reset_eagle_flag.py -q
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


def test_reset_op_stamps_the_coordinator_eagle_arm(monkeypatch, tmp_path):
    # eagle_armed() = requested AND the head checkpoint present; a present head makes the coordinator
    # EFFECTIVELY armed, so the reset stamps eagle:1. (A headless M25_EAGLE=1 coordinator degrades to
    # n-gram and stamps eagle:0 — covered by test_coordinate_watchdog's fail-safe test.)
    head = tmp_path / "m25-eagle"
    head.mkdir()
    (head / "config.json").write_text("{}")
    monkeypatch.setenv("M25_EAGLE_DIR", str(head))
    monkeypatch.setattr(MP, "_EAGLE_DISABLED", False)
    monkeypatch.setattr(fr.S, "M25_EAGLE", True)
    assert MP._reset_op("s", "j")["eagle"] == 1
    monkeypatch.setattr(fr.S, "M25_EAGLE", False)
    assert MP._reset_op("s", "j")["eagle"] == 0


# ---- real serve() stages on CPU (the test_tail_churn harness pattern) -------------------------------

class _FakeLayer:
    def reset(self):
        pass


def _patch_cpu(monkeypatch, tail):
    monkeypatch.setattr(MP, "dev", "cpu")
    monkeypatch.setattr(MP, "RECEIPTS", False)
    monkeypatch.setattr(MP, "_load",
                        lambda stage, nstages, lo, hi: {"layers": [_FakeLayer()], "head": False, "tail": tail})
    monkeypatch.setattr(MP, "_block", lambda grs, layers, start, x, vcfg: x)
    monkeypatch.setattr(MP, "_tail_logits", lambda h, parts: h)
    monkeypatch.setattr(MP.S, "_CTX", (None, None), raising=False)
    monkeypatch.setattr(MP.S, "_AUX", {}, raising=False)
    monkeypatch.setattr(MP.S, "M25_EAGLE", True, raising=False)   # the stage LAUNCHED eagle-armed
    monkeypatch.setattr(MP.S, "M25_STAGE_TIMING", False, raising=False)


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


def _verify(sock, start=0, aux=True, token_ids=True):
    # token_ids=False for the MIDDLE stage: a token_ids-bearing frame routes to the head's embed
    # branch (head/middle share serve()'s non-tail body); a middle stage receives h-only frames
    m = {"op": "verify", "start": start, "h": torch.zeros(1, 3, 4)}
    if token_ids:
        m["token_ids"] = [1, 2, 3]
    if aux:
        m["aux"] = {"1": torch.zeros(3, 8)}
    send_msg(sock, m)


def test_tail_honors_eagle0_per_session(monkeypatch):
    """An eagle-launched TAIL: a reset with eagle:0 -> plain token-list replies (no aux) for that
    session; the next plain reset (field absent, old coordinator) -> aux returns. Per-session, not
    sticky."""
    _patch_cpu(monkeypatch, tail=True)
    port = _free_port()
    threading.Thread(target=MP.serve, args=(1, 2, 0, 1, port, "127.0.0.1:1", 5), daemon=True).start()
    ret = _dial(port)
    send_msg(ret, {"op": "hello_return"})
    assert recv_msg(ret) == "ret_ok"
    pred = _dial(port)

    send_msg(pred, {"op": "reset", "eagle": 0})
    assert recv_msg(ret) == "ok"
    _verify(pred)
    r = recv_msg(ret)
    assert isinstance(r, list) and len(r) == 3, f"eagle:0 session must reply plain toks, got {r!r}"

    send_msg(pred, {"op": "reset"})                  # old-coordinator reset: launch env decides again
    assert recv_msg(ret) == "ok"
    _verify(pred)
    r = recv_msg(ret)
    assert isinstance(r, dict) and "aux" in r, f"flag-less reset must restore the launch arm, got {type(r)}"
    for s in (ret, pred):
        s.close()


def test_middle_honors_eagle0_and_propagates_the_field(monkeypatch):
    """An eagle-launched MIDDLE stage under an eagle:0 session: the forwarded reset CARRIES the
    field down the ring (stages forward resets unchanged), and forwarded verify frames drop the
    aux payload — every leg goes quiet, not just the tail's return."""
    _patch_cpu(monkeypatch, tail=False)
    nxt_srv = socket.socket()
    nxt_srv.bind(("127.0.0.1", 0)); nxt_srv.listen(2)
    port = _free_port()
    threading.Thread(target=MP.serve,
                     args=(1, 3, 0, 1, port, f"127.0.0.1:{nxt_srv.getsockname()[1]}", 5),
                     daemon=True).start()
    pred = _dial(port)
    fwd, _ = nxt_srv.accept()
    fwd.settimeout(5)

    send_msg(pred, {"op": "reset", "eagle": 0})
    m = recv_msg(fwd)
    assert m["op"] == "reset" and m.get("eagle") == 0, "reset must carry eagle:0 down the ring"
    _verify(pred, aux=True, token_ids=False)         # upstream aux arrives; the stage must NOT forward it
    m = recv_msg(fwd)
    assert m["op"] == "verify" and "aux" not in m, f"eagle:0 session forwarded aux: {list(m)}"

    send_msg(pred, {"op": "reset"})                  # flag-less reset restores the launch arm
    m = recv_msg(fwd)
    assert m["op"] == "reset" and "eagle" not in m
    _verify(pred, aux=True, token_ids=False)
    m = recv_msg(fwd)
    assert m["op"] == "verify" and "aux" in m
    for s in (pred, fwd, nxt_srv):
        try:
            s.close()
        except OSError:
            pass
