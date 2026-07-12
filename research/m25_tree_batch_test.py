"""Offline byte-identity gate for the BATCHED tree expansion (eagle_draft.propose_tree_b).

The de-lockstep coordinator drafts a speculative TREE per stream on n-gram-miss rounds; serial
propose_tree x B is the drafting tax draft_batch killed, reborn (~100 host syncs per tree: topk +
d2t reads per expansion). propose_tree_b runs the per-fork best-first heaps in lockstep, batching
every pending expansion into one [n,...] forward with ONE host sync per round. That is only
shippable if it is a pure wall-clock move: row j's tree must be BYTE-IDENTICAL to
eagles[j].propose_tree(m, topb, max_depth) — same tokens, same parents, same depths, same
expansion order — because the rows coordinator treats the two as interchangeable. Pinned here on
CPU with a synthetic tiny head (m25_draft_batch_test's pattern):

  1. propose_tree_b rows == per-fork propose_tree across ragged contexts + a degenerate (no-ctx) fork
  2. committed drafter state untouched by a batched expansion (scratch-tail discipline)
  3. multi-round: extend (commit growth) between draws keeps rows identical
  4. knob sweep: m/topb/max_depth combos incl. m=1, topb=1 (the chain degeneration) and depth-capped
  5. n==1 (the per-reply draw the event loop makes) + both next_hidden carries

NO GPU, NO model download: random weights in the real checkpoint format.

  python research/m25_tree_batch_test.py
"""
import os
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase0"))

from eagle_draft import EagleDrafter, propose_tree_b                        # noqa: E402
from m25_draft_batch_test import make_head, grow, snap, check_state, H, VOCAB  # noqa: E402


def _tree_eq(a, b, tag):
    assert a == b, (f"{tag}: batched tree != serial tree\n  serial ={a}\n  batched={b}")


def test_tree_batch_identity():
    tmp = tempfile.mkdtemp()
    make_head(tmp)
    gen = torch.Generator().manual_seed(42)
    embed = (torch.randn(VOCAB, H, generator=gen) * 0.3).to(torch.bfloat16)
    for carry in ("prenorm", "final"):
        base = EagleDrafter(tmp, embed, device="cpu", max_pos=2048, next_hidden=carry)
        forks = [base.fork() for _ in range(5)]
        for f, n in zip(forks, (6, 1, 17, 9, 0)):              # ragged contexts; last fork DEGENERATE
            if n:
                grow(f, n, gen, 0)
        serial = [f.propose_tree(12, topb=3, max_depth=8) for f in forks]
        states = [snap(f) for f in forks]
        batched = propose_tree_b(forks, 12, topb=3, max_depth=8)
        for j, (s, b) in enumerate(zip(serial, batched)):
            _tree_eq(s, b, f"[{carry}] fork {j}")
            check_state(states[j], forks[j], f"[{carry}] fork {j}")
        assert batched[4] == {"tokens": [], "parents": [], "depths": []}, "degenerate fork must be empty"
        # multi-round: commit growth between draws (the event-loop shape), rows must stay identical
        for rnd in range(3):
            for j, f in enumerate(forks):
                grow(f, 1 + (j + rnd) % 4, gen, f._last_pos + 1 if f.ctx_len else 0)
            serial = [f.propose_tree(12, topb=3, max_depth=8) for f in forks]
            batched = propose_tree_b(forks, 12, topb=3, max_depth=8)
            for j, (s, b) in enumerate(zip(serial, batched)):
                _tree_eq(s, b, f"[{carry}] round {rnd} fork {j}")
        print(f"  [{carry}] 5 ragged forks (incl. degenerate) x 4 rounds: batched tree == serial")


def test_knob_sweep():
    tmp = tempfile.mkdtemp()
    make_head(tmp, seed=7)
    gen = torch.Generator().manual_seed(99)
    embed = (torch.randn(VOCAB, H, generator=gen) * 0.3).to(torch.bfloat16)
    base = EagleDrafter(tmp, embed, device="cpu", max_pos=2048, next_hidden="prenorm")
    forks = [base.fork() for _ in range(4)]
    for f, n in zip(forks, (8, 24, 3, 11)):
        grow(f, n, gen, 0)
    for m, topb, depth in ((1, 3, 8), (12, 1, 8), (10, 2, 3), (16, 4, 2), (12, 3, 12)):
        serial = [f.propose_tree(m, topb=topb, max_depth=depth) for f in forks]
        batched = propose_tree_b(forks, m, topb=topb, max_depth=depth)
        for j, (s, b) in enumerate(zip(serial, batched)):
            _tree_eq(s, b, f"m={m} topb={topb} depth={depth} fork {j}")
        assert all(max(t["depths"], default=0) <= depth for t in batched), "depth cap violated"
        assert all(len(t["tokens"]) <= m for t in batched), "m cap violated"
    # topb=1 degenerates to the _draft chain (propose_tree's own losslessness gate) — batched too
    chain = forks[0]._draft(6)
    tb1 = propose_tree_b([forks[0]], 6, topb=1, max_depth=6)[0]
    assert tb1["tokens"] == chain, f"topb=1 batched tree != chain draft\n  {tb1['tokens']}\n  {chain}"
    print("  knob sweep (m/topb/depth incl. caps + chain degeneration): batched == serial")


def test_single_fork_path():
    """n==1 is the event loop's per-reply draw — must run the same batched code sync-light and stay
    byte-identical (draft_batch pins the same property for chains at m==1)."""
    tmp = tempfile.mkdtemp()
    make_head(tmp, seed=3)
    gen = torch.Generator().manual_seed(5)
    embed = (torch.randn(VOCAB, H, generator=gen) * 0.3).to(torch.bfloat16)
    base = EagleDrafter(tmp, embed, device="cpu", max_pos=2048, next_hidden="prenorm")
    f = base.fork()
    grow(f, 20, gen, 0)
    for rnd in range(4):
        s = f.propose_tree(12, topb=3, max_depth=8)
        b = propose_tree_b([f], 12, topb=3, max_depth=8)[0]
        _tree_eq(s, b, f"single-fork round {rnd}")
        grow(f, 2, gen, f._last_pos + 1)
    print("  single-fork (per-reply draw) x 4 rounds: batched == serial")


def bench():
    tmp = tempfile.mkdtemp()
    make_head(tmp, seed=3)
    gen = torch.Generator().manual_seed(1)
    embed = (torch.randn(VOCAB, H, generator=gen) * 0.3).to(torch.bfloat16)
    base = EagleDrafter(tmp, embed, device="cpu", max_pos=2048)
    forks = [base.fork() for _ in range(8)]
    for f in forks:
        grow(f, 64, gen, 0)
    t0 = time.time()
    for _ in range(10):
        for f in forks:
            f.propose_tree(12, topb=3, max_depth=8)
    ts = time.time() - t0
    t0 = time.time()
    for _ in range(10):
        propose_tree_b(forks, 12, topb=3, max_depth=8)
    tb = time.time() - t0
    print(f"  [bench, CPU-indicative] B=8 m=12: serial {ts*100:.1f}ms/round vs batched {tb*100:.1f}ms/round "
          f"({ts/tb:.1f}x)")


if __name__ == "__main__":
    test_tree_batch_identity()
    test_knob_sweep()
    test_single_fork_path()
    bench()
    print("[tree-batch] ALL PASS — batched tree expansion byte-identical to serial propose_tree per fork")
