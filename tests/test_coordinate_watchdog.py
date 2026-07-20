"""P0-#5 L2+L3 — shard.coordinate's degraded retry + job stall backstop.

L2: an EAGLE-implicated EDGE fault (the deliberately-stalled tail) triggers ONE degraded retry on
a FRESH ring dial — EAGLE off process-wide (sticky), eagle:0 on the wire, committed tokens resumed
under the SAME settlement nonce, the delta stream continuing with no dup/gap. A dead ring fails
the job cleanly (daemon restart), never freezes it. JobRejected never retries.

L3: a job that stops progressing entirely (a drafter wedged in torch, a stuck send — the classes
no socket timeout bounds) trips the stall watchdog: SHARD_JOB_FATAL + hard exit, daemon restart,
fail-closed complete. Prefill replies count as progress (a big chunk on a thin uplink is
legitimately slow) and idle-between-jobs never trips.

Run: python3 -m pytest tests/test_coordinate_watchdog.py -q
"""
import json
import argparse
import os
import socket
import subprocess
import sys
import time
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
import fake_ring as FR                              # noqa: E402  (bootstraps env + sys.path)
from fake_ring import FakeRing, FakeTok, novel_T    # noqa: E402

import shard.coordinate as C                        # noqa: E402
from ngram_draft import NgramDrafter                # noqa: E402
from eagle_draft import EagleDrafter, HybridDrafter  # noqa: E402
from test_eagle_draft import _make_head             # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = 60


def _args(**kw):
    d = dict(K=8, depth=4, ngram_n=3, max_ctx=0, timeout=30, prefill_chunk=24)
    d.update(kw)
    return argparse.Namespace(**d)


def _factory(monkeypatch, flags):
    """Replace MP.make_drafter: records the EAGLE arm at build time, returns a synthetic-head
    hybrid when armed (the real make_drafter would hit _eagle_singleton -> S.raw() -> boom on the
    fake model dir)."""
    def make_drafter(ngram_n=3):
        flags.append(bool(FR.MP.S.M25_EAGLE))
        ng = NgramDrafter(ng=3, min_match=1, margin=64)
        if not FR.MP.S.M25_EAGLE:
            return ng
        d, embed = _make_head(0)
        return HybridDrafter(ng, EagleDrafter(d, embed, device="cpu", next_hidden="prenorm"))
    monkeypatch.setattr(FR.MP, "make_drafter", make_drafter)
    return flags


def _mk_ring(T, ring_kw, timeout=30):
    c_pipe, r_pipe = socket.socketpair()
    c_ret, r_ret = socket.socketpair()
    c_ret.settimeout(timeout)
    ring = FakeRing(r_pipe, r_ret, T, **ring_kw)
    ring.tail_slack = 8                             # the muted/aborted trailing frames are exempt junk
    ring.start()
    return c_pipe, c_ret, ring


def _serve(T, lines, ring_kw=None, retry_ring_kw=None, args_kw=None):
    """serve_jobs against ring #1; redial() hands out a FRESH ring (the re-adopted tail model —
    the real tail's stale gate arms on a fresh hello_return). Returns (rc, emits, rings)."""
    rings = []
    pipe, ret, ring0 = _mk_ring(T, ring_kw or {})
    rings.append(ring0)
    socks = [pipe, ret]

    def redial():
        p, r, rg = _mk_ring(T, retry_ring_kw if retry_ring_kw is not None else {"eagle": True})
        rings.append(rg)
        socks.extend([p, r])
        return p, r

    emits = []

    def emit(tag, **fields):
        emits.append((tag, fields))

    try:
        rc = C.serve_jobs(FR.MP, FakeTok(T[:P]), pipe, ret, _args(**(args_kw or {})),
                          iter(lines), emit=emit, redial=redial)
    finally:
        for s in socks:
            try:
                s.close()
            except OSError:
                pass
        for rg in rings:
            rg.join(2)
    return rc, emits, rings


def _job(job_id="j-1", max_new=48, nonce="ab" * 16):
    return json.dumps({"jobId": job_id, "swarmId": "sw-1", "nonce": nonce, "maxNew": max_new,
                       "messages": [{"role": "user", "content": "fake"}]})


# ---- L2: the degraded retry ------------------------------------------------------------------------

def test_l2_degraded_retry_completes_lossless(monkeypatch):
    """Tail wedges mid-decode on attempt 1 (mutes forever) -> heartbeat trips in ~0.3s -> ONE
    degraded retry on a fresh dial resumes the committed tokens and completes LOSSLESS: joined
    deltas == response == the oracle continuation, same nonce on both resets, EAGLE built [on, off],
    sticky off after, DONE carries degraded:true."""
    monkeypatch.setattr(FR.MP.S, "M25_EAGLE", True)
    monkeypatch.setenv("M25_REPLY_TIMEOUT", "0.3")
    flags = _factory(monkeypatch, [])
    T = novel_T(400)
    t0 = time.monotonic()
    rc, emits, rings = _serve(T, [_job()], ring_kw={"eagle": True, "mute_after_decode": 3})
    assert time.monotonic() - t0 < 15
    assert rc == 0
    done = [f for t, f in emits if t == "SHARD_JOB_DONE"]
    assert len(done) == 1 and done[0]["ok"] and done[0]["degraded"] is True
    assert done[0]["tokensGenerated"] == 48 and done[0]["nonce"] == "ab" * 16
    want = FakeTok(T).decode(T[P:P + 48])
    assert done[0]["response"] == want, "losslessness broke across the retry"
    deltas = [f["delta"] for t, f in emits if t == "SHARD_JOB_TOKEN"]
    assert "".join(deltas) == want, "delta stream dup/gap across the retry (state not carried)"
    retries = [f for t, f in emits if t == "SHARD_JOB_RETRY"]
    assert len(retries) == 1 and retries[0]["jobId"] == "j-1"
    assert flags == [True, False], f"drafter arms {flags}"
    assert FR.MP.S.M25_EAGLE is False, "sticky degrade: the process must stay plain after an EAGLE-hostile fault"
    assert rings[0].muted >= 1, "the wedge never fired (vacuous)"
    assert len(rings) == 2
    # the retry re-prefilled prompt + committed under the SAME nonce (resume, not restart)
    r2_resets = [e for e in rings[1].log if e["op"] == "reset"]
    assert r2_resets and r2_resets[0]["nonce"] == "ab" * 16
    r2_first_pf = next(e for e in rings[1].log if e["op"] == "verify")
    assert r2_first_pf["prefill"] and r2_first_pf["start"] == 0
    assert P + retries[0]["committed"] >= r2_first_pf["n"] > 0


def test_l2_tree_mode_retry_falls_back_to_chain(monkeypatch):
    """M25_TREE implies EAGLE — the retry must flip BOTH off or coordinate_pipe_tree rejects the
    n-gram drafter. Retry ring sees chain frames only; both flags sticky off after."""
    monkeypatch.setattr(FR.MP.S, "M25_EAGLE", True)
    monkeypatch.setattr(FR.MP.S, "M25_TREE", True)
    monkeypatch.setenv("M25_TREE_TOPB", "2")
    monkeypatch.setenv("M25_TREE_DEPTH", "4")
    monkeypatch.setenv("M25_REPLY_TIMEOUT", "0.3")
    _factory(monkeypatch, [])
    T = novel_T(400)
    rc, emits, rings = _serve(T, [_job(max_new=24)], ring_kw={"eagle": True, "mute_after_decode": 2})
    assert rc == 0
    done = [f for t, f in emits if t == "SHARD_JOB_DONE"]
    assert len(done) == 1 and done[0]["ok"]
    assert done[0]["response"] == FakeTok(T).decode(T[P:P + 24])
    assert FR.MP.S.M25_EAGLE is False and FR.MP.S.M25_TREE is False
    assert all(not e.get("tree") for e in rings[1].log if e["op"] == "verify"), \
        "retry ring saw a TREE frame — M25_TREE not flipped with M25_EAGLE"


def test_l2_single_retry_then_bail_on_dead_ring(monkeypatch):
    """Both attempts wedge -> exactly one retry, then a clean bail: FATAL for the job, rc=1 (the
    daemon-restart path), no DONE — and it all happens in seconds, never a freeze."""
    monkeypatch.setattr(FR.MP.S, "M25_EAGLE", True)
    monkeypatch.setenv("M25_REPLY_TIMEOUT", "0.3")
    _factory(monkeypatch, [])
    T = novel_T(400)
    t0 = time.monotonic()
    rc, emits, rings = _serve(T, [_job()],
                              ring_kw={"eagle": True, "mute_after_decode": 1},
                              retry_ring_kw={"eagle": True, "mute_after_decode": 1})
    assert time.monotonic() - t0 < 15
    assert rc == 1
    tags = [t for t, _ in emits]
    assert tags.count("SHARD_JOB_RETRY") == 1
    assert not [f for t, f in emits if t == "SHARD_JOB_DONE"]
    fatal = [f for t, f in emits if t == "SHARD_JOB_FATAL"]
    assert fatal and fatal[-1]["jobId"] == "j-1"
    assert FR.MP.S.M25_EAGLE is False


def test_l2_no_retry_on_job_rejected(monkeypatch):
    """A structured job error (H1: the ring REPLIED, it didn't die) must never burn the retry:
    FATAL for job 1, no RETRY, EAGLE stays armed, and job 2 serves on the SAME channels."""
    monkeypatch.setattr(FR.MP.S, "M25_EAGLE", True)
    _factory(monkeypatch, [])
    T = novel_T(400)
    rc, emits, rings = _serve(T, [_job("j-1"), _job("j-2", max_new=8)],
                              ring_kw={"eagle": True, "reject_decode": 1})
    assert rc == 0
    tags = [t for t, _ in emits]
    assert "SHARD_JOB_RETRY" not in tags
    assert FR.MP.S.M25_EAGLE is True, "JobRejected must not degrade the process"
    fatal = [f for t, f in emits if t == "SHARD_JOB_FATAL"]
    assert len(fatal) == 1 and fatal[0]["jobId"] == "j-1"
    done = [f for t, f in emits if t == "SHARD_JOB_DONE"]
    assert len(done) == 1 and done[0]["jobId"] == "j-2" and done[0]["ok"]
    assert len(rings) == 1, "no redial may happen on a JobRejected"


def test_l2_no_retry_on_healthy_job(monkeypatch):
    monkeypatch.setattr(FR.MP.S, "M25_EAGLE", True)
    flags = _factory(monkeypatch, [])
    T = novel_T(400)
    rc, emits, rings = _serve(T, [_job(max_new=16)], ring_kw={"eagle": True})
    assert rc == 0
    done = [f for t, f in emits if t == "SHARD_JOB_DONE"]
    assert len(done) == 1 and done[0]["ok"] and done[0]["degraded"] is False
    assert not [1 for t, _ in emits if t == "SHARD_JOB_RETRY"]
    assert flags == [True] and len(rings) == 1
    assert FR.MP.S.M25_EAGLE is True


def test_l2_retry_receipt_swept_once_same_nonce(monkeypatch):
    """Receipts across the retry: attempt 1 aborts BEFORE its sweep, so exactly one receipt op
    total (on the retry ring), both resets carry the job's settlement nonce, and DONE returns it."""
    monkeypatch.setattr(FR.MP.S, "M25_EAGLE", True)
    monkeypatch.setattr(FR.MP, "RECEIPTS", True)
    monkeypatch.setenv("M25_REPLY_TIMEOUT", "0.3")
    _factory(monkeypatch, [])
    T = novel_T(400)
    rc, emits, rings = _serve(T, [_job(nonce="beef" * 8)], ring_kw={"eagle": True, "mute_after_decode": 2})
    assert rc == 0
    done = [f for t, f in emits if t == "SHARD_JOB_DONE"]
    assert len(done) == 1 and done[0]["ok"] and done[0]["nonce"] == "beef" * 8
    sweeps = [e for rg in rings for e in rg.log if e["op"] == "receipt"]
    assert len(sweeps) == 1, f"receipt swept {len(sweeps)}x across the retry"
    assert not any(e["op"] == "receipt" for e in rings[0].log), "aborted attempt must not sweep"
    for rg in rings:
        for e in rg.log:
            if e["op"] == "reset":
                assert e["nonce"] == "beef" * 8


# ---- L3: the stall backstop --------------------------------------------------------------------------

def test_l3_watchdog_fires_on_wedged_coordinate_pipe(monkeypatch):
    """A coordinate_pipe that never returns (drafter wedged in torch — nothing socket-level can
    bound it) trips the watchdog: SHARD_JOB_FATAL with the jobId, then the hard exit. Order pinned
    via a shared event list."""
    import threading
    unblock = threading.Event()
    events = []

    def fake_exit(code):
        events.append(("exit", code))
        unblock.set()                                # let the wedged main thread finish the test

    monkeypatch.setattr(C, "_hard_exit", fake_exit)
    monkeypatch.setenv("M25_JOB_STALL_S", "0.5")

    def wedged_pipe(*a_, **kw):
        unblock.wait(20)
        return {"ok": False, "resumable": False, "error": "wedged"}

    MP = types.SimpleNamespace(S=types.SimpleNamespace(M25_EAGLE=False, M25_TREE=False),
                               EDGE_ERRORS=(OSError, EOFError), TransportError=ConnectionError,
                               make_drafter=lambda n=3: object(), coordinate_pipe=wedged_pipe)
    emits = []

    def emit(tag, **fields):
        events.append((tag, fields))
        emits.append((tag, fields))

    t0 = time.monotonic()
    C.serve_jobs(MP, FakeTok([1, 2, 3]), None, None, _args(), iter([_job("j-wedge")]), emit=emit)
    dt = time.monotonic() - t0
    assert dt < 5, f"watchdog did not fire fast ({dt:.1f}s)"
    exits = [e for e in events if e[0] == "exit"]
    assert exits == [("exit", 1)]
    stall = [i for i, e in enumerate(events)
             if e[0] == "SHARD_JOB_FATAL" and "stall-watchdog" in e[1].get("error", "")]
    assert stall and events[stall[0]][1]["jobId"] == "j-wedge"
    assert stall[0] < events.index(("exit", 1)), "FATAL must be emitted BEFORE the hard exit"


def test_l3_slow_prefill_is_progress_not_stall(monkeypatch):
    """Chunked prefill with per-chunk stalls: total prefill time exceeds the stall budget but each
    reply arrives inside it — prefill replies MUST count as progress (a big chunk over a thin
    uplink is legitimately slow) so the job completes with no watchdog kill."""
    monkeypatch.setattr(FR.MP.S, "M25_EAGLE", False)
    monkeypatch.setenv("M25_JOB_STALL_S", "1.5")
    monkeypatch.setenv("M25_REPLY_TIMEOUT", "0.3")   # double-pins the prefill heartbeat exemption here
    exits = []
    monkeypatch.setattr(C, "_hard_exit", lambda code: exits.append(code))
    T = novel_T(400)
    prompt_len = 80                                  # 4 chunks at prefill_chunk=24 -> all carry prefill=True
    rings = []
    pipe, ret, ring = _mk_ring(T, {"stall_prefill": (4, 0.5)})
    rings.append(ring)
    emits = []
    try:
        rc = C.serve_jobs(FR.MP, FakeTok(T[:prompt_len]), pipe, ret, _args(),
                          iter([_job(max_new=8)]), emit=lambda tag, **f: emits.append((tag, f)))
    finally:
        for s in (pipe, ret):
            try:
                s.close()
            except OSError:
                pass
        ring.join(2)
    assert rc == 0
    done = [f for t, f in emits if t == "SHARD_JOB_DONE"]
    assert len(done) == 1 and done[0]["ok"]
    assert not exits, "watchdog killed a legitimately slow prefill"
    assert ring.stalled_pf == 4, "the prefill stalls never fired (vacuous)"


def test_l3_idle_between_jobs_is_not_a_stall(monkeypatch):
    monkeypatch.setattr(FR.MP.S, "M25_EAGLE", False)
    monkeypatch.setenv("M25_JOB_STALL_S", "0.3")
    exits = []
    monkeypatch.setattr(C, "_hard_exit", lambda code: exits.append(code))
    T = novel_T(400)
    pipe, ret, ring = _mk_ring(T, {})

    def lines():
        yield _job("j-1", max_new=8)
        time.sleep(1.0)                              # idle >> budget: the watchdog must be disarmed
        yield _job("j-2", max_new=8)

    try:
        rc = C.serve_jobs(FR.MP, FakeTok(T[:P]), pipe, ret, _args(), lines(),
                          emit=lambda tag, **f: None if tag else None)
    finally:
        for s in (pipe, ret):
            try:
                s.close()
            except OSError:
                pass
        ring.join(2)
    assert rc == 0
    assert not exits, "watchdog fired while idle between jobs"


def test_l3_real_exit_subprocess():
    """The one real os._exit test: a child drives serve_jobs against a muting ring with the
    heartbeat DISABLED (only the watchdog can save it) — the process must die fast with the
    stall-watchdog FATAL on stdout, not ride the 60s recv timeout."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("M25_", "SHARD_"))}
    env.update({"M25_REPLY_TIMEOUT": "0", "M25_JOB_STALL_S": "1",
                "PYTHONDONTWRITEBYTECODE": "1"})
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, os.path.join(REPO, "tests", "_watchdog_exit_child.py")],
                       capture_output=True, text=True, cwd=REPO, env=env, timeout=60)
    dt = time.monotonic() - t0
    assert r.returncode == 1, f"rc={r.returncode}\n{r.stdout}\n{r.stderr}"
    assert "stall-watchdog" in r.stdout, r.stdout
    assert dt < 45, f"child took {dt:.0f}s — it rode a recv timeout, not the watchdog"
