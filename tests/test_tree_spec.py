"""Offline certification of the tree-verify primitives — the load-bearing math the ring executes under
M25_TREE, none of which previously had tracked tests (2026-07-02 fleet finding).

Covers: (1) build_tree_mask's ancestor-only property vs a brute-force ancestor walk on random trees;
(2) tree_greedy_walk == exhaustive longest-matching-path (duplicate-free trees) + the committed tokens are
always a true greedy-continuation prefix; (3) chain degeneracy (a 1-wide tree == linear accept);
(4) _gqa_masked_attend vs the repeat_interleave reference; (5) _rope_gather vs the contiguous-slice RoPE;
(6) EagleDrafter.propose_tree: topb=1 == propose() exactly (the losslessness gate's offline half),
tree well-formedness (parents before children, depth = parent depth + 1, node budget), and committed-cache
isolation (only extend() mutates).

Run: pytest tests/test_tree_spec.py -q   (CPU-only, no GPU / no model dir needed)
"""
import os
import random
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase0"))
sys.path.insert(0, os.path.dirname(__file__))
from tree_spec import build_tree_mask, tree_greedy_walk, _gqa_masked_attend, _rope_gather, _rotate_half  # noqa: E402
from test_eagle_draft import _make_head, _upcast_fp32  # noqa: E402
from eagle_draft import EagleDrafter  # noqa: E402


def _random_tree(rng, n, max_children=3):
    """Random parents-before-children tree over n nodes (parent -1 = anchor)."""
    parents, depths = [], []
    for i in range(n):
        p = -1 if i == 0 or rng.random() < 0.25 else rng.randrange(i)
        parents.append(p)
        depths.append(1 if p == -1 else depths[p] + 1)
    return parents, depths


def _ancestors(parents, i):
    out = set()
    while i >= 0:
        out.add(i)
        i = parents[i]
    return out


def test_mask_ancestor_only_property():
    rng = random.Random(7)
    for _ in range(20):
        n, start = rng.randrange(1, 12), rng.randrange(1, 30)
        parents, depths = _random_tree(rng, n)
        bias, positions = build_tree_mask(parents, depths, start, n)
        assert bias.shape == (1, 1, n, start + n)
        assert (bias[0, 0, :, :start] == 0).all()            # whole committed prefix visible to every node
        for i in range(n):
            anc = _ancestors(parents, i)
            for j in range(n):
                blocked = bias[0, 0, i, start + j].item() == float("-inf")
                assert blocked == (j not in anc), (i, j, parents)
        assert positions.tolist() == [(start - 1) + d for d in depths]


def _exhaustive_walk(tokens, parents, target_argmax, anchor_target):
    """Longest valid greedy path by brute force over every root->node chain."""
    best = []
    for leaf in range(len(tokens)):
        chain = []
        i = leaf
        while i >= 0:
            chain.append(i)
            i = parents[i]
        chain.reverse()
        want = anchor_target
        ok = 0
        for node in chain:
            if tokens[node] != want:
                break
            ok += 1
            want = target_argmax[node]
        if ok == len(chain) and len(chain) > len(best):
            best = chain
    return best


def test_walk_matches_exhaustive_on_duplicate_free_trees():
    rng = random.Random(11)
    for _ in range(50):
        n = rng.randrange(1, 14)
        parents, depths = _random_tree(rng, n)
        # duplicate-free siblings: token ids unique per node (first-match == longest-match territory)
        tokens = rng.sample(range(1000), n)
        target_argmax = [rng.choice(tokens + [rng.randrange(1000)]) for _ in range(n)]
        anchor_target = rng.choice(tokens + [rng.randrange(1000)])
        path, committed = tree_greedy_walk(tokens, parents, target_argmax, anchor_target)
        assert path == _exhaustive_walk(tokens, parents, target_argmax, anchor_target)
        # committed is always the accepted tokens + the target's next token at the stopping node
        want = anchor_target
        for k, node in enumerate(path):
            assert committed[k] == tokens[node] == want
            want = target_argmax[node]
        assert committed[-1] == want and len(committed) == len(path) + 1


def test_walk_chain_degenerate_equals_linear_accept():
    tokens = [5, 6, 7, 8]
    parents = [-1, 0, 1, 2]
    # target agrees on 5,6 then wants 99 at node 1 -> accept 2, correction 99
    path, committed = tree_greedy_walk(tokens, parents, [6, 99, 0, 0], 5)
    assert path == [0, 1] and committed == [5, 6, 99]
    # full accept: bonus = target's token after the leaf
    path, committed = tree_greedy_walk(tokens, parents, [6, 7, 8, 42], 5)
    assert path == [0, 1, 2, 3] and committed == [5, 6, 7, 8, 42]


def test_gqa_masked_attend_matches_repeat_interleave_reference():
    g = torch.Generator().manual_seed(3)
    NH, NKV, N, T, HD = 8, 2, 5, 17, 16
    q = torch.randn(1, NH, N, HD, generator=g)
    k = torch.randn(1, NKV, T, HD, generator=g)
    v = torch.randn(1, NKV, T, HD, generator=g)
    mask = torch.zeros(1, 1, N, T)
    mask[0, 0, torch.randint(0, N, (6,), generator=g), torch.randint(0, T, (6,), generator=g)] = float("-inf")
    kk = k.repeat_interleave(NH // NKV, dim=1)
    vv = v.repeat_interleave(NH // NKV, dim=1)
    ref = torch.matmul(torch.softmax((torch.matmul(q, kk.transpose(-1, -2)) * (HD ** -0.5) + mask).float(), -1).to(vv.dtype), vv)
    out = _gqa_masked_attend(q, k, v, mask, NH // NKV)
    assert torch.allclose(out, ref, atol=1e-6), (out - ref).abs().max()


def test_rope_gather_matches_contiguous_slice():
    g = torch.Generator().manual_seed(4)
    heads, N, HD, rd, start = 3, 6, 16, 8, 9
    maxpos = 64
    inv = 1.0 / (10000.0 ** (torch.arange(0, rd, 2).float() / rd))
    fr = torch.outer(torch.arange(maxpos).float(), inv)
    e = torch.cat([fr, fr], -1)
    cos, sin = e.cos(), e.sin()
    t = torch.randn(1, heads, N, HD, generator=g)
    tr, tp = t[..., :rd], t[..., rd:]
    cu = cos[start:start + N].unsqueeze(0).unsqueeze(0); su = sin[start:start + N].unsqueeze(0).unsqueeze(0)
    ref = torch.cat([tr * cu + _rotate_half(tr) * su, tp], -1)
    out = _rope_gather(t, cos, sin, torch.arange(start, start + N), rd)
    assert torch.allclose(out, ref, atol=1e-6)


# ---- propose_tree on the real (synthetic-weight) drafter ------------------------------------

def _ctx_drafter(seed=0, n_ctx=24):
    d, embed = _make_head(seed)
    eg = _upcast_fp32(EagleDrafter(d, embed, device="cpu", next_hidden="prenorm"))
    g = torch.Generator().manual_seed(seed + 100)
    toks = torch.randint(0, 100, (n_ctx,), generator=g).tolist()
    aux = torch.randn(n_ctx, 3, 32, generator=g)          # H=32 in the synthetic head
    eg.extend(toks, aux, base_pos=0)
    return eg


def test_propose_tree_topb1_equals_chain_propose():
    """The losslessness gate's offline half: a 1-wide tree IS the chain."""
    for seed in (0, 1, 2):
        eg = _ctx_drafter(seed)
        chain = eg.propose(6)
        tree = eg.propose_tree(6, topb=1, max_depth=6)
        assert tree["tokens"] == chain, (seed, tree["tokens"], chain)
        assert tree["parents"] == [-1, 0, 1, 2, 3, 4]
        assert tree["depths"] == [1, 2, 3, 4, 5, 6]


def test_propose_tree_well_formed():
    eg = _ctx_drafter(1)
    for m, topb, md in ((12, 3, 8), (8, 2, 4), (1, 3, 8), (16, 4, 2)):
        t = eg.propose_tree(m, topb=topb, max_depth=md)
        n = len(t["tokens"])
        assert n <= m and len(t["parents"]) == n and len(t["depths"]) == n
        for i, (p, d) in enumerate(zip(t["parents"], t["depths"])):
            assert -1 <= p < i                              # parents strictly before children
            assert d == (1 if p == -1 else t["depths"][p] + 1)
            assert d <= md
        # depth-1 nodes exist and the anchor has at most topb direct children
        assert t["depths"][0] == 1
        assert sum(1 for p in t["parents"] if p == -1) <= topb


def test_propose_tree_does_not_mutate_committed_cache():
    eg = _ctx_drafter(2)
    ctx_len = eg.ctx_len
    kc = eg.kbuf[:ctx_len].clone(); vc = eg.vbuf[:ctx_len].clone()
    lh = eg._last_h.clone(); lt, lp = eg._last_tok, eg._last_pos
    eg.propose_tree(12, topb=3, max_depth=6)
    assert eg.ctx_len == ctx_len and eg._last_tok == lt and eg._last_pos == lp
    assert torch.equal(eg.kbuf[:ctx_len], kc) and torch.equal(eg.vbuf[:ctx_len], vc)
    assert torch.equal(eg._last_h, lh)
    # and drafting again from the same state is deterministic
    a = eg.propose_tree(8, topb=2, max_depth=5)
    b = eg.propose_tree(8, topb=2, max_depth=5)
    assert a == b


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  {name} PASS")
    print("ALL PASS")
