"""The lazy-draft licence (pspec.coord LAZY LICENCE, BUILD-PLAN D1 gate "lazy licence sound at
every F"): the entry hints `dnxt` / `lic`, the tail's TailDrafterHost skips the blocks the hint
proves unused and answers `lz`, and the entry raises if it ever needs the block of an `lz` reply.

  1. sound at every refill floor F x W x block length x acceptance x frame policy: lossless and
     no unsound skip (the raise), over the real TailDrafterHost with scripted blocks
  2. not vacuous: it skips drafts at floor 1 (`lic` + `dnxt`) and at floor 2 (`dnxt` alone)
  3. mutant licences (lic at floor 2, lic at W = 2, dnxt one row too far, the host dereferencing
     `lic` through a 1-draft block, V4's rule) are caught by the raise
  4. end to end with a real (tiny) MTP-3 / DFlash on the tail, lazy
"""
import inspect
import types

import pytest

from scripted_pipe import BlocksDrafter, alpha_miss, run_pspec, tail_blocks, toy_greedy, truth
from test_pspec_drafter import test_end_to_end_with_a_real_drafter as e2e

from shard.pspec import coord
from shard.pspec.drafter import TailDrafterHost

SID = "00000000000000a1"

# ── the licence ────────────────────────────────────────────────────────────────────────────────

T = truth(600)
P, K = 7, 6
ACCEPT = {"full": None, "zero": alpha_miss(0.0), "partial": lambda p, d: d >= 3,
          "random": alpha_miss(0.75, seed=11)}


def _lazy_run(policy, accept, W, floor, k, mod=coord, depth=3, max_new=45):
    host = TailDrafterHost(BlocksDrafter(tail_blocks(T, k, ACCEPT[accept]), k))
    r, pipe, s = run_pspec(T, P, max_new, depth=depth, W=W, floor=floor, policy=policy, host=host,
                           lazy=True, mod=mod)
    return r, host


@pytest.mark.parametrize("k", [1, 2, K])
@pytest.mark.parametrize("floor", [1, 2, K])
@pytest.mark.parametrize("W", [1, 2, 3, K + 1, 16])
@pytest.mark.parametrize("accept", sorted(ACCEPT))
@pytest.mark.parametrize("policy", coord.POLICIES)
def test_lazy_licence_sound_at_every_floor(policy, accept, W, floor, k):
    """Lossless, and the entry never needs the block of a reply the tail skipped (it would raise)."""
    r, host = _lazy_run(policy, accept, W, floor, k)
    assert r["tokens"] == toy_greedy(T, P, 45)
    assert r["lazy_skips"] <= host.skipped
    if policy == "chunk":
        assert host.skipped == 0, "a chunk burst is one frame: nothing to hint"


@pytest.mark.parametrize("floor", [1, 2])
@pytest.mark.parametrize("W", [2, 3, K + 1])
@pytest.mark.parametrize("off", range(1, 14))
def test_lazy_licence_sound_with_eos_anywhere(off, W, floor):
    """EOS committed at any depth of a run: the stop drains every later reply, skipped or not."""
    eos = (T[P + off],)
    for k in (1, 2, K):
        host = TailDrafterHost(BlocksDrafter(tail_blocks(T, k, ACCEPT["random"]), k))
        r, *_ = run_pspec(T, P, 45, W=W, floor=floor, policy="pertoken", host=host, lazy=True,
                          eos=eos)
        assert r["tokens"] == toy_greedy(T, P, 45, eos=eos)


def test_lazy_licence_saves_drafts():
    """The licence is not vacuous: at floor 1 with runs in flight it skips most drafts."""
    eager = TailDrafterHost(BlocksDrafter(tail_blocks(T, K), K))
    run_pspec(T, P, 90, W=K + 1, policy="pertoken", host=eager)
    r, lazy = _lazy_run("pertoken", "full", K + 1, 1, K, max_new=90)
    assert r["tokens"] == toy_greedy(T, P, 90)
    assert eager.skipped == 0 and lazy.skipped > 0 and lazy.drafted < eager.drafted
    r2, lazy2 = _lazy_run("pertoken", "full", 16, 2, K, max_new=90)
    assert lazy2.skipped > 0, "dnxt alone (floor > 1, no lic) still skips"


def _mutant(old, new):
    src = inspect.getsource(coord)
    assert src.count(old) == 1, old
    m = types.ModuleType("shard.pspec.coord_mutant")
    m.__package__, m.__file__ = coord.__package__, coord.__file__
    exec(compile(src.replace(old, new), coord.__file__, "exec"), m.__dict__)
    return m


@pytest.mark.parametrize("old,new,W,floor,k", [
    ("self.floor == 1 and self.W >= 3 and", "self.W >= 3 and", K + 1, 2, K),     # lic at floor 2
    ("self.floor == 1 and self.W >= 3 and", "self.floor == 1 and", 2, 1, K),      # lic at W = 2
    ("H - r > self.floor", "H - r >= self.floor", K + 1, 1, K),                    # dnxt one too far
])
def test_an_unsound_licence_is_caught(old, new, W, floor, k):
    bad = _mutant(old, new)
    with pytest.raises(RuntimeError, match="lazy licence"):
        _lazy_run("pertoken", "full", W, floor, k, mod=bad, max_new=60)


def test_the_two_draft_dereference_guard_is_load_bearing(monkeypatch):
    """V4's `wants_block` dereferenced any non-empty block. With a 1-draft block the refill streams
    a single position and the `lic` frame's own reply is the refill: V4's rule is unsound here."""
    from shard.pspec import accept as ACC
    from shard.pspec import drafter as DR
    src = inspect.getsource(ACC)
    old = "if blk and len(blk) >= 2 and at == int(pos) - 1:"
    assert src.count(old) == 1
    m = types.ModuleType("shard.pspec.accept_mutant")
    m.__package__, m.__file__ = ACC.__package__, ACC.__file__
    exec(compile(src.replace(old, "if blk and at == int(pos) - 1:"), ACC.__file__, "exec"), m.__dict__)
    monkeypatch.setattr(DR, "wants_block", m.wants_block)
    host = TailDrafterHost(BlocksDrafter(tail_blocks(T, 1), 1))
    with pytest.raises(RuntimeError, match="lazy licence"):
        run_pspec(T, P, 60, W=K + 1, floor=1, policy="pertoken", host=host, lazy=True)


def test_lazy_refuses_an_entry_drafter_and_an_ack_cap():
    with pytest.raises(ValueError):
        coord.Session(T[:P], 5, sid=SID, drafter=lambda ids: [], lazy=True)
    with pytest.raises(ValueError):
        coord.Session(T[:P], 5, sid=SID, acked=P, lazy=True)


@pytest.mark.parametrize("policy", coord.POLICIES)
@pytest.mark.parametrize("kind", ["mtp", "dflash"])
def test_end_to_end_lazy_with_a_real_drafter(kind, policy):
    e2e(kind, policy, lazy=True)
