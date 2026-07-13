"""H1 — the negotiated-context backstop, m25_pipe half.

A KV overflow used to raise RuntimeError inside serve()'s dispatch, which is NOT in EDGE_ERRORS,
escaped the edge recovery, and KILLED the unwrapped warm stage process. Now it is a structured
PER-JOB error: the stage replies/forwards {"error"/"op":"job_error"} and stays alive (KV + stale
machinery re-arm on the next reset), and the coordinator raises JobRejected — deliberately not an
OSError, so EDGE_ERRORS recovery/retry can never eat it.

Also covers the coord CLI's --max-ctx threading into _run_job (the old hardcoded 131072 silently
overran smaller-KV rings).

Run: python3 -m pytest tests/test_job_rejected.py -q
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

from ngram_draft import NgramDrafter                # noqa: E402

MP = fr.MP
send_msg, recv_msg = fr.send_msg, fr.recv_msg

ERR = {"code": "kv_overflow", "stage": "tail", "message": "boom"}


# ---- coordinator: an {"error": ...} reply raises JobRejected (never retried as an edge fault) -------

def _run_coord(resumable):
    pipe_a, pipe_b = socket.socketpair()
    ret_a, ret_b = socket.socketpair()

    def _ring():
        while True:
            try:
                m = recv_msg(pipe_b)
            except (OSError, EOFError):
                return
            if m.get("op") == "reset":
                send_msg(ret_b, "ok")
            elif m.get("op") == "verify":
                send_msg(ret_b, {"error": dict(ERR)})

    t = threading.Thread(target=_ring, daemon=True)
    t.start()
    tok = fr.FakeTok(list(range(20)))
    try:
        return MP.coordinate_pipe(pipe_a, tok, [{"role": "user", "content": "x"}], K=4, max_new=16,
                                  timeout=5, depth=2, ret_sock=ret_a,
                                  local_draft=NgramDrafter(ng=3, min_match=1),
                                  resumable=resumable)
    finally:
        for s in (pipe_a, pipe_b, ret_a, ret_b):
            try:
                s.close()
            except OSError:
                pass


def test_error_reply_raises_job_rejected():
    with pytest.raises(MP.JobRejected) as ei:
        _run_coord(resumable=False)
    assert "kv_overflow" in str(ei.value)


def test_job_rejected_not_eaten_by_resumable_recovery():
    """resumable=True catches EDGE_ERRORS to hand tokens back — a rejection must NOT take that path
    (the ring is healthy; a resume of the same oversized job would only be rejected again)."""
    with pytest.raises(MP.JobRejected):
        _run_coord(resumable=True)


def test_job_rejected_is_not_an_edge_error():
    from node_kv import EDGE_ERRORS
    assert not issubclass(MP.JobRejected, EDGE_ERRORS)


# ---- real serve() tail: RuntimeError -> structured error reply, stage ALIVE -------------------------

class _FakeLayer:
    def reset(self):
        pass


def _block_overflow(grs, layers, start, x, vcfg):
    if start >= 100:
        raise RuntimeError("start exceeds KV maxlen (simulated overflow)")
    return x


def _patch_cpu_tail(monkeypatch):
    monkeypatch.setattr(MP, "dev", "cpu")
    monkeypatch.setattr(MP, "RECEIPTS", False)
    monkeypatch.setattr(MP, "_load",
                        lambda stage, nstages, lo, hi: {"layers": [_FakeLayer()], "head": False, "tail": True})
    monkeypatch.setattr(MP, "_block", _block_overflow)
    monkeypatch.setattr(MP, "_tail_logits", lambda h, parts: h)
    monkeypatch.setattr(MP.S, "_CTX", (None, None), raising=False)
    monkeypatch.setattr(MP.S, "M25_EAGLE", False, raising=False)
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


def test_tail_kv_overflow_is_job_error_not_process_exit(monkeypatch):
    _patch_cpu_tail(monkeypatch)
    port = _free_port()
    threading.Thread(target=MP.serve, args=(1, 2, 0, 1, port, "127.0.0.1:1", 5), daemon=True).start()
    ret = _dial(port)
    send_msg(ret, {"op": "hello_return"})
    assert recv_msg(ret) == "ret_ok"
    pred = _dial(port)

    send_msg(pred, {"op": "reset"})
    assert recv_msg(ret) == "ok"
    send_msg(pred, {"op": "verify", "h": torch.zeros(1, 3, 4), "start": 0})
    assert isinstance(recv_msg(ret), list)

    # the poisoned frame: before the fix this RuntimeError KILLED the serve loop
    send_msg(pred, {"op": "verify", "h": torch.zeros(1, 3, 4), "start": 999})
    r = recv_msg(ret)
    assert isinstance(r, dict) and r["error"]["code"] == "kv_overflow" and r["error"]["stage"] == "tail"

    # job is dead (stale) — pre-reset frames are dropped, not answered
    send_msg(pred, {"op": "verify", "h": torch.zeros(1, 3, 4), "start": 0})
    ret.settimeout(0.5)
    with pytest.raises(socket.timeout):
        recv_msg(ret)

    # ...and the stage is ALIVE: the next reset re-arms the session on the same sockets
    ret.settimeout(3)
    send_msg(pred, {"op": "reset"})
    assert recv_msg(ret) == "ok", "tail died on the KV overflow (process-exit regression)"
    send_msg(pred, {"op": "verify", "h": torch.zeros(1, 3, 4), "start": 0})
    assert isinstance(recv_msg(ret), list)
    ret.close(); pred.close()


# ---- real serve() middle: RuntimeError -> job_error forwarded; job_error frames are relayed ---------

def test_middle_kv_overflow_forwards_job_error(monkeypatch):
    monkeypatch.setattr(MP, "dev", "cpu")
    monkeypatch.setattr(MP, "RECEIPTS", False)
    monkeypatch.setattr(MP, "_load",
                        lambda stage, nstages, lo, hi: {"layers": [_FakeLayer()], "head": False, "tail": False})
    monkeypatch.setattr(MP, "_block", _block_overflow)
    monkeypatch.setattr(MP.S, "_CTX", (None, None), raising=False)
    monkeypatch.setattr(MP.S, "M25_EAGLE", False, raising=False)
    monkeypatch.setattr(MP.S, "M25_STAGE_TIMING", False, raising=False)
    nxt_srv = socket.socket()
    nxt_srv.bind(("127.0.0.1", 0)); nxt_srv.listen(2)
    port = _free_port()
    threading.Thread(target=MP.serve,
                     args=(1, 3, 0, 1, port, f"127.0.0.1:{nxt_srv.getsockname()[1]}", 5),
                     daemon=True).start()
    pred = _dial(port)
    fwd, _ = nxt_srv.accept()
    fwd.settimeout(5)

    # overflow on the middle stage -> a structured job_error rides the ring instead of a process exit
    send_msg(pred, {"op": "verify", "h": torch.zeros(1, 3, 4), "start": 999})
    m = recv_msg(fwd)
    assert m["op"] == "job_error" and m["error"]["code"] == "kv_overflow" and m["stage"] == 1

    # an upstream stage's job_error is relayed untouched (the old fall-through KeyError'd on msg['h'])
    send_msg(pred, {"op": "job_error", "stage": 0, "error": {"code": "kv_overflow", "message": "up"}})
    m = recv_msg(fwd)
    assert m["op"] == "job_error" and m["stage"] == 0

    # the stage survived both
    send_msg(pred, {"op": "verify", "h": torch.zeros(1, 3, 4), "start": 0})
    assert recv_msg(fwd)["op"] == "verify"
    for s in (pred, fwd, nxt_srv):
        try:
            s.close()
        except OSError:
            pass


# ---- coord CLI: --max-ctx threads into the job (no more hardcoded 131072) ---------------------------

def test_run_job_threads_max_ctx(monkeypatch):
    seen = {}

    def _fake_coordinate_pipe(*a, **k):
        seen.update(k)
        return {"ok": True}

    monkeypatch.setattr(MP, "coordinate_pipe", _fake_coordinate_pipe)
    monkeypatch.setattr(MP, "make_drafter", lambda n: object())
    MP._run_job(None, None, None, [], 6, 64, 5, 4, 3, 512, max_ctx=40960)
    assert seen["max_ctx"] == 40960
