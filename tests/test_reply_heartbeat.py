"""F6 — the coordinator's per-reply DECODE heartbeat (m25_pipe._reply_timeout).

On a permissionless ring an internal leg blips while the coordinator is alive; the tail then holds
the job stale until the coordinator's next reset (the PR #26 return-channel fix) and drops the
in-flight replies. Without a tight per-reply deadline the coordinator would block on recv up to the
full production timeout (1800s) before EDGE_ERRORS fires the resume/retry. The heartbeat bounds each
DECODE round-trip to a few seconds so blip failover is seconds, not up-to-timeout — while PREFILL
keeps the full budget (a big activation over a slow uplink is legitimately slow).

Driven on the CPU fake ring (tests/fake_ring.py): stall_decode / stall_prefill delay the ring's
first replies to model a blip vs. legitimate slowness. No GPU, no model, no network.

Run: python3 -m pytest tests/test_reply_heartbeat.py -q
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
fr = pytest.importorskip("fake_ring")               # bootstraps env + imports m25_pipe on CPU

from ngram_draft import NgramDrafter                # noqa: E402

MP = fr.MP
P = 60                                              # prompt length (a T prefix, via FakeTok)


def _ngram():
    return NgramDrafter(ng=3, min_match=1, margin=64)


# ---- 1. _reply_timeout helper ---------------------------------------------------------------------

def test_reply_timeout_default_20(monkeypatch):
    monkeypatch.delenv("M25_REPLY_TIMEOUT", raising=False)
    assert MP._reply_timeout(1800) == 20.0
    assert MP._reply_timeout(1800.0) == 20.0


def test_reply_timeout_env_override(monkeypatch):
    monkeypatch.setenv("M25_REPLY_TIMEOUT", "30")
    assert MP._reply_timeout(1800) == 30.0
    monkeypatch.setenv("M25_REPLY_TIMEOUT", "2.5")
    assert MP._reply_timeout(1800) == 2.5


def test_reply_timeout_never_exceeds_production_timeout(monkeypatch):
    monkeypatch.setenv("M25_REPLY_TIMEOUT", "20")
    assert MP._reply_timeout(3) == 3           # clamp: a 20s heartbeat can't outlast a 3s job timeout


def test_reply_timeout_zero_or_empty_disables(monkeypatch):
    monkeypatch.setenv("M25_REPLY_TIMEOUT", "0")
    assert MP._reply_timeout(1800) == 1800     # 0 = escape hatch -> full timeout, no heartbeat
    monkeypatch.setenv("M25_REPLY_TIMEOUT", "")
    assert MP._reply_timeout(1800) == 1800


def test_reply_timeout_garbage_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("M25_REPLY_TIMEOUT", "not-a-number")
    assert MP._reply_timeout(1800) == 20.0


# ---- 2. the heartbeat FIRES on a mid-decode blip --------------------------------------------------

def test_heartbeat_trips_fast_on_decode_stall(monkeypatch):
    """A decode reply that never comes within the heartbeat trips EDGE_ERRORS -> the job fails over in
    seconds. The ring stalls the first decode reply 1.2s while the heartbeat is 0.3s (and the job
    timeout is a full 30s): the coordinator must give up on the heartbeat, NOT wait the 30s. Without
    F6 the 1.2s stall is under the timeout, so the job would just complete slowly and NOT raise —
    which is exactly the regression this guards."""
    monkeypatch.setenv("M25_REPLY_TIMEOUT", "0.3")
    T = fr.novel_T(200)
    t0 = time.monotonic()
    with pytest.raises(Exception):                  # coordinate_pipe raises TransportError on the blip
        # prefill_chunk<P so prefill frames carry prefill=True and the stall lands on the first DECODE frame
        fr.run_coordinator(T, P, _ngram(), K=8, depth=1, max_new=160,
                           prefill_chunk=24, eagle_ring=False, timeout=30, stall_decode=(1, 1.2))
    dt = time.monotonic() - t0
    # ~0.3s heartbeat + ~2s of run_coordinator's post-mortem ring.join(2) — the point is it's seconds,
    # nowhere near the 30s production timeout it would otherwise have blocked on.
    assert dt < 6.0, f"heartbeat did not fire fast: {dt:.1f}s (should be a few s, not the 30s timeout)"


def test_heartbeat_trips_on_tree_path(monkeypatch):
    """Same guard on the EAGLE tree coordinator (coordinate_pipe_tree has its own recv loop)."""
    pytest.importorskip("eagle_draft")
    from eagle_draft import EagleDrafter, HybridDrafter
    from test_eagle_draft import _make_head
    monkeypatch.setattr(fr.S, "M25_EAGLE", True)
    monkeypatch.setattr(fr.S, "M25_TREE", True)
    monkeypatch.setenv("M25_REPLY_TIMEOUT", "0.3")
    d, embed = _make_head(0)
    hyb = HybridDrafter(_ngram(), EagleDrafter(d, embed, device="cpu", next_hidden="prenorm"))
    T = fr.novel_T(200)
    t0 = time.monotonic()
    with pytest.raises(Exception):
        fr.run_coordinator(T, P, hyb, K=8, depth=4, max_new=160, prefill_chunk=24,
                           eagle_ring=True, timeout=30, stall_decode=(1, 1.2))
    assert time.monotonic() - t0 < 6.0


# ---- 3. the heartbeat does NOT false-trip ---------------------------------------------------------

def test_heartbeat_tolerates_normal_decode_jitter(monkeypatch):
    """A slow-but-fine traversal (0.3s) under a 2.0s heartbeat must NOT trip — the job completes
    lossless. Guards against a heartbeat set so tight it kills healthy jittery rings."""
    monkeypatch.setenv("M25_REPLY_TIMEOUT", "2.0")
    T = fr.novel_T(200)
    res, ring = fr.run_coordinator(T, P, _ngram(), K=8, depth=1, max_new=40,
                                   prefill_chunk=24, eagle_ring=False, timeout=30,
                                   stall_decode=(1, 0.3))
    assert res["ok"], res
    assert res["output_ids"] == T[P:P + len(res["output_ids"])], "losslessness broke under tolerated jitter"
    assert ring.stalled == 1, "the decode stall never fired (test would be vacuous)"


def test_prefill_is_exempt_from_heartbeat(monkeypatch):
    """PREFILL must keep the full timeout: a 1.5s prefill reply under a 0.3s decode heartbeat must
    still succeed (a big activation over a slow uplink is legitimately slow). If prefill were wrongly
    under the heartbeat, this 1.5s stall would trip and the job would raise instead of completing."""
    monkeypatch.setenv("M25_REPLY_TIMEOUT", "0.3")
    T = fr.novel_T(200)
    # prefill_chunk<P forces multi-chunk prefill so the frames carry prefill=True (the single-chunk
    # path omits the flag) — the stall then lands on a real prefill reply.
    res, ring = fr.run_coordinator(T, P, _ngram(), K=8, depth=1, max_new=40,
                                   prefill_chunk=24, eagle_ring=False, timeout=30,
                                   stall_prefill=(1, 1.5))
    assert res["ok"], res
    assert res["output_ids"] == T[P:P + len(res["output_ids"])]
    assert ring.stalled_pf == 1, "the prefill stall never actually fired (test would be vacuous)"
