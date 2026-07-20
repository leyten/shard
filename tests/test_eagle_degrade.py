"""P0-#5 L1 — the coordinator's EAGLE draft-budget watchdog (m25_pipe._draft_budget + the
coordinate_pipe degrade).

The serial EAGLE draft chain is coordinator-LOCAL compute — the one leg of a decode round no
socket timeout can see (F6 bounds the ring recv, the job timeout bounds prefill). A wedged or
pathologically slow drafter therefore crawls the job forever while still streaming: the
2026-07-14 residential-tail "silent hang" shape. The watchdog: an eagle-routed draft step over
M25_DRAFT_BUDGET_S twice consecutively (job's first eagle step exempt — warmup) flips the job to
n-gram-only IN PLACE — drafter latches via HybridDrafter.disable_eagle(), cur_depth returns to
pipelined `depth`, extend() stops. No wire change, job completes lossless: worst case slower,
never dead.

Driven on the CPU fake ring (tests/fake_ring.py). No GPU, no model, no network.

Run: python3 -m pytest tests/test_eagle_degrade.py -q
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
fr = pytest.importorskip("fake_ring")               # bootstraps env + imports m25_pipe on CPU

from ngram_draft import NgramDrafter                # noqa: E402
from eagle_draft import EagleDrafter, HybridDrafter  # noqa: E402
from test_eagle_draft import _make_head             # noqa: E402

MP = fr.MP
P = 60                                              # prompt length (a T prefix, via FakeTok)


def _ngram():
    return NgramDrafter(ng=3, min_match=1, margin=64)


class CountingEagle:
    """Wraps the EAGLE half: counts propose() calls and sleeps on the first `n_slow` of them (the
    deliberately-slow drafter). Everything else delegates — reset/extend/set_hidden/fork all reach
    the real synthetic-head EagleDrafter."""

    def __init__(self, inner, n_slow=0, sleep_s=0.0):
        self.inner = inner
        self.n_slow = n_slow
        self.sleep_s = sleep_s
        self.calls = 0

    def propose(self, ids, k):
        self.calls += 1
        if self.calls <= self.n_slow:
            time.sleep(self.sleep_s)
        return self.inner.propose(ids, k)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _hybrid(n_slow=0, sleep_s=0.0, seed=0):
    d, embed = _make_head(seed)
    eg = CountingEagle(EagleDrafter(d, embed, device="cpu", next_hidden="prenorm"),
                       n_slow=n_slow, sleep_s=sleep_s)
    return HybridDrafter(_ngram(), eg), eg


def test_l1_degrade_completes_lossless_and_restores_pipelining(monkeypatch):
    """An EAGLE half that is slow FOREVER (every propose sleeps past the budget) must not crawl the
    job to the end: the watchdog trips after 2 consecutive breaches (first step exempt), the drafter
    is never consulted again (calls stays tiny — a broken degrade fails this fast, not slow), the
    job completes LOSSLESS, and depth-pipelining resumes (ring.backlog > 0 = >1 frame in flight,
    impossible while EAGLE pins depth 1)."""
    monkeypatch.setattr(fr.S, "M25_EAGLE", True)
    monkeypatch.setenv("M25_DRAFT_BUDGET_S", "0.05")
    hyb, eg = _hybrid(n_slow=999, sleep_s=0.2)
    T = fr.novel_T(460)                             # novel -> every pre-trip fetch is eagle-routed
    rec = fr.RecordingDrafter(hyb)
    res, ring = fr.run_coordinator(T, P, rec, K=8, depth=4, max_new=100,
                                   eagle_ring=True, timeout=30, stall_decode=(3, 0.15))
    assert res["ok"], res
    assert res["output_ids"] == T[P:P + len(res["output_ids"])], "losslessness broke across the degrade"
    assert len(res["output_ids"]) >= 100
    assert res["eagle_degraded"] is True
    assert res["eagle_degraded_at"] is not None and res["eagle_degraded_at"] <= 3 * 8 + 2
    # trip = first (exempt) + 2 breaching steps; nothing after — the slow head is OUT of the loop
    assert eg.calls == 3, f"EAGLE consulted {eg.calls}x — degrade did not latch"
    assert ring.backlog >= 1, "no pipelined frames in flight after the degrade (depth stuck at 1)"
    assert ring.stalled == 3                        # the decode stalls actually fired (backlog not vacuous)


def test_l1_no_false_trip_healthy_eagle(monkeypatch):
    """A healthy fast drafter under the default-scale budget never trips: EAGLE is consulted every
    novel round to the end, no degrade is recorded, and depth stays pinned at 1 (no backlog)."""
    monkeypatch.setattr(fr.S, "M25_EAGLE", True)
    monkeypatch.setenv("M25_DRAFT_BUDGET_S", "5")
    hyb, eg = _hybrid()
    T = fr.novel_T(300)
    res, ring = fr.run_coordinator(T, P, hyb, K=8, depth=4, max_new=60,
                                   eagle_ring=True, timeout=30)
    assert res["ok"], res
    assert res["output_ids"] == T[P:P + len(res["output_ids"])]
    assert not res["eagle_degraded"]
    assert res["eagle_degraded_at"] is None
    assert eg.calls == res["rounds"], "EAGLE not consulted every round (cancel never computes)"
    assert ring.backlog == 0, "depth>1 in flight while EAGLE active — the depth-1 pin broke"


def test_l1_budget_zero_disables(monkeypatch):
    """M25_DRAFT_BUDGET_S=0 is the escape hatch (the _reply_timeout convention): a slow drafter is
    tolerated, the job completes lossless with EAGLE still armed."""
    monkeypatch.setattr(fr.S, "M25_EAGLE", True)
    monkeypatch.setenv("M25_DRAFT_BUDGET_S", "0")
    hyb, eg = _hybrid(n_slow=2, sleep_s=0.2)
    T = fr.novel_T(300)
    res, _ = fr.run_coordinator(T, P, hyb, K=8, depth=4, max_new=40,
                                eagle_ring=True, timeout=30)
    assert res["ok"], res
    assert res["output_ids"] == T[P:P + len(res["output_ids"])]
    assert not res["eagle_degraded"]
    assert eg.calls >= 2, "the slow steps never ran (test would be vacuous)"


def test_l1_inert_without_eagle(monkeypatch):
    """Plain n-gram jobs (no EAGLE) never arm the budget: an absurdly tight M25_DRAFT_BUDGET_S must
    change nothing — lossless, pipelined, no degrade fields set."""
    monkeypatch.setattr(fr.S, "M25_EAGLE", False)
    monkeypatch.setenv("M25_DRAFT_BUDGET_S", "0.0001")
    T = fr.repetitive_T(400)
    res, ring = fr.run_coordinator(T, P, _ngram(), K=8, depth=4, max_new=120,
                                   eagle_ring=False, timeout=30, stall_decode=(2, 0.1))
    assert res["ok"], res
    assert res["output_ids"] == T[P:P + len(res["output_ids"])]
    assert not res["eagle_degraded"]
    assert ring.backlog >= 1, "plain-path pipelining regressed"
