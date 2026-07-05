"""CPU tests for the cwnd keep-warm lever (phase0/m25_pipe.py _KeepWarm / recv_data / noop protocol).

WHY: TCP slow-start-after-idle collapses cwnd on every ring leg that idles >RTO between frames —
measured 2026-07-05 on a 40ms-RTT vast leg: idle<=300ms keeps 30KB-1.6MB frames at ~1 RTT, idle=900ms
costs 2-4 RTTs on the same frames. The kernel knob is read-only in vast containers, so the engine
sends {"op":"noop"} on idle legs. These tests pin: the noop cadence (and 0 = fully OFF), the
one-lock-per-socket send discipline (interleaved partial frames corrupt the stream), that receivers
skip noops without popping `inflight` or breaking losslessness, the reset-op runtime toggle, and the
_tail_accept noop-as-first-frame classification chain.

Run: python3 -m pytest tests/test_cwnd_keepwarm.py -q
"""
import os
import select
import socket
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
fr = pytest.importorskip("fake_ring")               # bootstraps env + imports m25_pipe on CPU

from ngram_draft import NgramDrafter                # noqa: E402
from node_kv import send_msg, recv_msg              # noqa: E402  (same codec the wrapper uses)

MP = fr.MP

P = 60                                              # prompt length for fake-ring runs


def _ngram():
    return NgramDrafter(ng=3, min_match=1, margin=64)


def _pair(timeout=5):
    a, b = socket.socketpair()
    b.settimeout(timeout)
    return a, b


def _drain(sock, seconds):
    """Collect every frame arriving on sock for `seconds` (select-polled, never blocks past the end)."""
    out, end = [], time.monotonic() + seconds
    while time.monotonic() < end:
        if select.select([sock], [], [], 0.01)[0]:
            out.append(recv_msg(sock))
    return out


# ---- 1. NOOP CADENCE -------------------------------------------------------------------------------

def test_noop_cadence_short_interval():
    """An idle wrapped socket at interval=30ms must emit >=2 noops in 150ms (wake ~interval/3)."""
    a, b = _pair()
    kw = MP._KeepWarm(a, interval_ms=30)
    try:
        frames = _drain(b, 0.15)
    finally:
        kw.stop(); a.close(); b.close()
    noops = [f for f in frames if f == {"op": "noop"}]
    assert len(noops) >= 2, f"expected >=2 noops in 150ms at 30ms interval, got {len(noops)}"
    assert noops == frames, "keep-warm thread sent something other than noops"


def test_interval_zero_is_off():
    """interval=0 (the default: M25_CWND_KEEPWARM_MS unset) = master behavior: no thread, no bytes."""
    a, b = _pair()
    for kw in (MP._KeepWarm(a), MP._KeepWarm(a, interval_ms=0)):
        assert kw._runner is None, "interval=0 must not spawn a keep-warm thread"
    assert _drain(b, 0.12) == []
    a.close(); b.close()


def test_env_interval_picked_up(monkeypatch):
    """M25_CWND_KEEPWARM_MS is the stage-launch default when no explicit interval is given."""
    monkeypatch.setenv("M25_CWND_KEEPWARM_MS", "20")
    a, b = _pair()
    kw = MP._KeepWarm(a)
    try:
        frames = _drain(b, 0.12)
    finally:
        kw.stop(); a.close(); b.close()
    assert sum(1 for f in frames if f == {"op": "noop"}) >= 2


def test_runtime_toggle_on_off():
    """set_interval() flips warming at runtime (the reset-op path): 0->on->0, thread exits cleanly."""
    a, b = _pair()
    kw = MP._KeepWarm(a, interval_ms=0)
    assert _drain(b, 0.08) == []
    kw.set_interval(20)
    assert sum(1 for f in _drain(b, 0.12) if f == {"op": "noop"}) >= 2
    kw.set_interval(0)
    t0 = time.monotonic()                           # runner exits on its next wake; then silence
    while kw._runner is not None and time.monotonic() - t0 < 1.0:
        time.sleep(0.005)
    assert kw._runner is None, "runner thread did not exit after set_interval(0)"
    _drain(b, 0.03)                                 # flush any noop sent before the exit
    assert _drain(b, 0.1) == [], "noops still flowing after toggle-off"
    a.close(); b.close()


def test_dead_socket_never_crashes():
    """The noop thread must swallow send errors (dead socket = the serve loop's problem)."""
    a, b = _pair()
    kw = MP._KeepWarm(a, interval_ms=5)
    time.sleep(0.03)
    b.close(); a.close()                            # kill the pipe under the running thread
    time.sleep(0.05)                                # several failing wakes: must not raise/exit process
    assert kw._runner is not None and kw._runner.is_alive()   # still alive, still swallowing
    kw.stop()


# ---- 2. LOCK SAFETY (no interleaved partial frames) ------------------------------------------------

def test_send_lock_interleave_safety():
    """Hammer real sends while the noop thread runs at 1ms over the same socket: the receiver must
    decode every frame intact and in order. Without the shared lock, two threads' sendall() calls
    interleave partial frames and the codec explodes / payloads scramble."""
    N = 300
    a, b = _pair(timeout=10)
    got, noops, err = [], [0], []

    def rx():
        try:
            while len(got) < N:
                f = recv_msg(b)
                if f == {"op": "noop"}:
                    noops[0] += 1
                else:
                    got.append(f)
        except Exception as e:                      # decode error == corruption == the bug
            err.append(e)

    t = threading.Thread(target=rx, daemon=True)
    t.start()
    kw = MP._KeepWarm(a, interval_ms=1)
    try:
        for i in range(N):                          # header+tensor-blob frames: multi-part on the wire
            kw.send({"op": "data", "i": i, "h": torch.full((4,), float(i), dtype=torch.float32)})
            if i % 25 == 0:
                time.sleep(0.003)                   # yield so the noop thread interleaves for real
    finally:
        kw.stop()
    t.join(10)
    assert not err, f"receiver hit a decode error (stream corrupted): {err[0]}"
    assert len(got) == N
    assert [f["i"] for f in got] == list(range(N)), "data frames reordered/lost"
    for f in got:
        assert torch.equal(f["h"], torch.full((4,), float(f["i"]), dtype=torch.float32)), \
            f"tensor payload scrambled in frame {f['i']}"
    assert noops[0] >= 1, "noop thread never interleaved — test exercised nothing"
    a.close(); b.close()


# ---- 3. RECV SKIP ----------------------------------------------------------------------------------

def test_recv_data_skips_noops():
    a, b = _pair()
    send_msg(a, {"op": "noop"})
    send_msg(a, {"op": "noop"})
    send_msg(a, {"op": "reply", "v": 1})
    assert MP.recv_data(b) == {"op": "reply", "v": 1}
    a.close(); b.close()


def test_recv_data_noop_flood_still_times_out():
    """A peer whose noop thread is alive but whose compute is wedged must still trip the recv
    timeout: each noop resets the socket timer, so recv_data enforces an overall deadline."""
    a, b = _pair(timeout=0.3)
    stop = threading.Event()

    def flood():
        while not stop.is_set():
            try:
                send_msg(a, {"op": "noop"})
            except OSError:
                return
            time.sleep(0.05)

    t = threading.Thread(target=flood, daemon=True)
    t.start()
    t0 = time.monotonic()
    with pytest.raises(OSError):                    # socket.timeout is an EDGE_ERROR -> normal supervision
        MP.recv_data(b)
    assert time.monotonic() - t0 < 2.0, "deadline not enforced under a noop flood"
    stop.set(); t.join(2); a.close(); b.close()


@pytest.mark.parametrize("flavor", ["repetitive", "trap"])
def test_inflight_survives_noops_between_replies(flavor, monkeypatch):
    """Inject a noop before EVERY ring reply (reset-ok, prefill, decode, receipts) on the return leg:
    the pipelined coordinator (depth=4, inflight nonempty) must stay byte-lossless — a noop that
    popped `inflight` or was counted as a reply would desync accept bookkeeping immediately. The trap
    flavor exercises the divergence/discard path with noops interleaved."""
    orig = fr.send_msg

    def noopy(sock, obj):
        orig(sock, {"op": "noop"})
        return orig(sock, obj)

    monkeypatch.setattr(fr, "send_msg", noopy)      # FakeRing.run resolves send_msg via the module global
    T = fr.repetitive_T(560) if flavor == "repetitive" else fr.trap_T(560)
    res, ring = fr.run_coordinator(T, P, _ngram(), K=8, depth=4, max_new=160,
                                   prefill_chunk=24, eagle_ring=False)
    assert res["ok"], res
    out = res["output_ids"]
    assert len(out) >= 160
    assert out == T[P:P + len(out)], "losslessness broke with noops interleaved on the return leg"
    if flavor == "trap":
        assert res["wasted"] > 0, "trap never exercised the in-flight discard path"


# ---- 4. RESET TOGGLE PLUMBING ----------------------------------------------------------------------

def test_keepwarm_job_toggles_reset_field_and_coord_sender(monkeypatch):
    """M25_KEEPWARM_JOB=25 must (a) ride the job's reset op as keepwarm_ms=25 — the ring-wide toggle
    stages apply to their wrapped senders — and (b) arm the coordinator's own coord->head wrapper:
    the ring stalls its first replies 250ms, so the idle pipe must carry noops (the ring logs+skips
    them exactly like serve() does)."""
    monkeypatch.setenv("M25_KEEPWARM_JOB", "25")
    T = fr.repetitive_T(560)
    res, ring = fr.run_coordinator(T, P, _ngram(), K=8, depth=4, max_new=120,
                                   prefill_chunk=4096, eagle_ring=False, stall_decode=(2, 0.25))
    assert res["ok"], res
    assert res["output_ids"] == T[P:P + len(res["output_ids"])]
    resets = [e for e in ring.log if e["op"] == "reset"]
    assert resets and resets[0]["keepwarm_ms"] == 25, f"reset did not carry the toggle: {resets}"
    assert any(e["op"] == "noop" for e in ring.log), \
        "coordinator wrapper sent no noop during a 250ms ring stall at interval=25ms"


def test_no_job_env_means_no_field_no_noops(monkeypatch):
    """Absent M25_KEEPWARM_JOB = master behavior: no keepwarm_ms on the reset (stages keep their
    current setting) and no noops from the coordinator, even across a ring stall."""
    monkeypatch.delenv("M25_KEEPWARM_JOB", raising=False)
    monkeypatch.delenv("M25_CWND_KEEPWARM_MS", raising=False)
    T = fr.repetitive_T(560)
    res, ring = fr.run_coordinator(T, P, _ngram(), K=8, depth=4, max_new=120,
                                   prefill_chunk=4096, eagle_ring=False, stall_decode=(2, 0.25))
    assert res["ok"], res
    resets = [e for e in ring.log if e["op"] == "reset"]
    assert resets and resets[0]["keepwarm_ms"] is None, \
        f"reset grew a keepwarm field with no env set: {resets}"
    assert not any(e["op"] == "noop" for e in ring.log), "noops sent with keep-warm OFF"


# ---- 5. _tail_accept: A NOOP CAN BE A NEW PREDECESSOR'S FIRST FRAME --------------------------------

def test_tail_accept_classifies_noop_as_speaking_predecessor():
    """A replaced predecessor's keep-warm thread may speak before the first job frame, so its
    greeting can be {"op":"noop"}. _tail_accept must classify it as a SPEAKING predecessor (any op
    frame) and hand the noop back as first_msg — where the serve loop's skip (op=="noop" ->
    continue) discards it. Closing the conn instead would kill-loop stage replacement."""
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0)); srv.listen(4)
    port = srv.getsockname()[1]
    pred = socket.create_connection(("127.0.0.1", port))
    send_msg(pred, {"op": "noop"})                  # buffered BEFORE _tail_accept runs -> deterministic
    time.sleep(0.05)                                # classification via the speaking path, not silent-adopt
    coord = [None]

    def late_coord():
        coord[0] = socket.create_connection(("127.0.0.1", port))
        send_msg(coord[0], {"op": "hello_return"})

    threading.Timer(0.3, late_coord).start()
    ret, pd, first = MP._tail_accept(srv, timeout=5)
    assert first == {"op": "noop"}, f"noop not handed back as first_msg: {first!r}"
    assert first.get("op") == "noop"                # the serve loop's skip guard catches exactly this
    assert recv_msg(coord[0]) == "ret_ok"           # return channel acked normally
    send_msg(pred, {"op": "reset"})                 # the adopted pred socket is live for real frames
    assert recv_msg(pd) == {"op": "reset"}
    for s in (pred, coord[0], ret, pd, srv):
        s.close()
