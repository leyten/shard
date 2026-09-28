"""shard/pspec/accept.py: the one accept rule, both halves, against the V4 functions it was lifted from.

The lifted pure functions are pinned EQUAL to the V4 originals (v4_dspark_draft.plan_verify_round,
wants_block) over exhaustive small inputs, so "as-is" is a checked claim, not a comment. The
frame-generic rule is pinned to V4's per-token rule at n = 1 and to plan_verify_round (M2.5's
chunk round) at n = K+1. Stdlib only, except the parity tests, which skip without torch.
"""
import itertools
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "engines", "deepseek_v4"))

from shard.pspec.accept import OPEN, REJECT, RUN, Frontier, accept, plan_verify_round, wants_block


def _v4():
    pytest.importorskip("torch")
    return pytest.importorskip("v4_dspark_draft")


# ── the lifted functions ARE the V4 functions ──────────────────────────────────────────────────

def test_plan_verify_round_is_v4s_over_every_small_input():
    v4 = _v4()
    for g in range(0, 4):
        for drafts in itertools.product(range(3), repeat=g):
            for replies in itertools.product(range(3), repeat=g + 1):
                assert plan_verify_round(drafts, replies) == v4.plan_verify_round(drafts, replies)
    for bad in ([], [1], [1, 2, 3]):                               # g+1 replies or it raises, both
        with pytest.raises(RuntimeError):
            plan_verify_round([1], bad)
        with pytest.raises(RuntimeError):
            v4.plan_verify_round([1], bad)


def test_wants_block_is_v4s_over_every_hint_shape():
    v4 = _v4()
    lasts = [None, (4, []), (4, [7]), (4, [8, 1]), (5, [7]), (3, [7])]
    for pos, m, dnxt, lic, last in itertools.product([5], [7, 8], [None, 7, 8], [False, True], lasts):
        msg = {"start_pos": pos}
        if dnxt is not None:
            msg["dnxt"] = dnxt
        if lic:
            msg["dprev"] = True
        assert wants_block(pos, m, dnxt=dnxt, lic=lic, last=last) == v4.wants_block(msg, m, last), \
            (pos, m, dnxt, lic, last)


# ── one rule for every frame length ────────────────────────────────────────────────────────────

def test_n1_is_v4s_per_token_rule():
    """V4 (v4_pipe.py:2112-2180): commit m; RUN if the token streamed at pos+1 is m, REJECT if it is
    something else, OPEN (need a block) if nothing is streamed there."""
    for m, fed in itertools.product(range(3), [None, 0, 1, 2]):
        committed, verdict = accept([m], [fed])
        assert committed == [m]
        assert verdict == (OPEN if fed is None else RUN if fed == m else REJECT)


def test_chunk_with_nothing_behind_is_plan_verify_round():
    """n = K+1 with nothing in flight past the frame: exactly M2.5's chunk round (full-accept bonus
    included), because accept runs plan_verify_round over the frame's own rows."""
    rng = random.Random(0)
    for _ in range(3000):
        K = rng.randrange(0, 7)
        drafts = [rng.randrange(3) for _ in range(K)]
        toks = [rng.randrange(3) for _ in range(K + 1)]
        committed, verdict = accept(toks, drafts + [None])
        k, want = plan_verify_round(drafts, toks)
        assert committed == want
        assert verdict == (OPEN if k == K else REJECT)


def test_a_chunk_runs_into_the_next_frame_only_when_its_last_token_matches():
    toks = [5, 6, 7]
    assert accept(toks, [5, 6, 7]) == ([5, 6, 7], RUN)
    assert accept(toks, [5, 6, 9]) == ([5, 6, 7], REJECT)
    assert accept(toks, [5, 9, 7]) == ([5, 6], REJECT), "the first mismatch ends it, whatever follows"
    assert accept(toks, [9, 6, 7]) == ([5], REJECT)


def test_accept_refuses_a_malformed_frame():
    with pytest.raises(ValueError):
        accept([1, 2], [1])
    with pytest.raises(ValueError):
        accept([1, 2, 3], [1, None, 3])                          # a hole inside the frame's own rows
    with pytest.raises(ValueError):
        accept([], [])


def test_every_committed_token_is_the_models_never_the_draft():
    """Lossless by construction: committed is always a prefix of toks."""
    rng = random.Random(1)
    for _ in range(2000):
        n = rng.randrange(1, 6)
        toks = [rng.randrange(3) for _ in range(n)]
        fed = [rng.randrange(3) for _ in range(n - 1)] + [rng.choice([None, 0, 1, 2])]
        committed, _ = accept(toks, fed)
        assert committed == toks[:len(committed)] and committed


# ── the tail half ──────────────────────────────────────────────────────────────────────────────

def test_frontier_at_n1_is_v4s_incremental_rule():
    """V4's _on_chunk_pipelined: acc iff start_pos == cfront+1 and the fed id == mfront; an accepted
    frame moves (cfront, mfront) to (pos, m); a rejected one moves nothing."""
    rng = random.Random(2)
    fr = Frontier()
    fr.prefill(3, 10)
    cf, mf = 2, 10
    for _ in range(500):
        pos = cf + 1 + rng.choice([0, 0, 0, 1, -1])
        tok = mf if rng.random() < 0.7 else mf + 1
        m = rng.randrange(20)
        acc, k = fr.judge(pos, [tok], [m])
        want = pos == cf + 1 and tok == mf
        assert acc == want
        if want:
            cf, mf = pos, m
            assert k == 0
        assert (fr.cfront, fr.mfront) == (cf, mf)


def test_frontier_advances_over_a_chunks_accepted_prefix():
    fr = Frontier()
    fr.prefill(4, 50)                                           # rows 0..3, the model's token at 4 = 50
    assert fr.judge(4, [50, 51, 99, 53], [51, 52, 53, 54]) == (True, 1)
    assert (fr.cfront, fr.mfront) == (5, 52), "diverged at row 6: frontier at 5, correction 52 at 6"
    assert fr.judge(6, [99], [0]) == (False, -1), "the discarded future is off the path"
    assert fr.judge(6, [52, 53], [53, 54]) == (True, 1)
    assert (fr.cfront, fr.mfront) == (7, 54)


def test_frontier_refuses_a_frame_before_the_prefill():
    with pytest.raises(RuntimeError, match="before the prefill"):
        Frontier().judge(0, [1], [2])


def test_both_halves_agree_on_every_random_frame_stream():
    """The entry commits with `accept`, the tail advances with `Frontier`; over any stream of frames
    that the entry would actually send, their frontiers stay equal (this is what `acc` asserts)."""
    rng = random.Random(3)
    for _ in range(300):
        truth = [rng.randrange(4) for _ in range(80)]
        fr = Frontier()
        fr.prefill(1, truth[1])
        ids = truth[:2]
        while len(ids) < 60:
            f = len(ids) - 1
            n = rng.randrange(1, 5)
            x = [ids[f]] + [truth[f + 1 + j] if rng.random() < 0.6 else 9 for j in range(n - 1)]
            toks = [truth[f + j + 1] if x[:j + 1] == truth[f:f + j + 1] else 9 for j in range(n)]
            acc, _ = fr.judge(f, x, toks)
            committed, _ = accept(toks, x[1:] + [None])
            assert acc
            ids += committed
            assert fr.cfront + 1 == len(ids) - 1 and fr.mfront == ids[-1]
            assert ids == truth[:len(ids)]
