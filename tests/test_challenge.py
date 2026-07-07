"""Layer-block challenge — the compute-honesty spot-check (shard/challenge.py).

This is the primitive that actually catches a node getting PAID WITHOUT running its block's real
matmuls (the gap the receipt hash-chain and any token "endpoint binding" cannot close — a receipt
chain only proves byte-continuity, never `out == block(in)`). A verifier draws a seeded input both
sides derive identically, the suspect and a trusted replica each run the same block, and their
outputs are compared by cosine similarity + relative norm with a tolerance (heterogeneous hardware
drifts a few ULPs → cosine ~1.0; garbage or a wrong/skipped block → cosine ~0). It had ZERO tests.

CPU, no model weights: the pure discriminators are tested directly, and challenge_block's orchestration
with synthetic blocks (block_forward stubbed) — proving an honest recompute PASSES and a lazy /
constant / wrong block FAILS.

Run: python3 -m pytest tests/test_challenge.py -q
"""
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

torch = pytest.importorskip("torch")
from shard import challenge as ch                    # noqa: E402


# ---- 1. derive_challenge: deterministic + host-independent ------------------------------------------

def test_derive_challenge_deterministic_same_seed():
    a = ch.derive_challenge("seed-A", 5, 32, device="cpu", dtype=torch.float32)
    b = ch.derive_challenge("seed-A", 5, 32, device="cpu", dtype=torch.float32)
    assert torch.equal(a, b)                          # same seed -> identical bytes (verifier == node)
    assert a.shape == (1, 5, 32)


def test_derive_challenge_differs_by_seed():
    a = ch.derive_challenge("seed-A", 5, 32, device="cpu", dtype=torch.float32)
    b = ch.derive_challenge("seed-B", 5, 32, device="cpu", dtype=torch.float32)
    assert not torch.equal(a, b)                      # a node can't precompute for an unknown seed


# ---- 2. sketch: deterministic, compact -------------------------------------------------------------

def test_sketch_is_deterministic_and_compact():
    h = torch.randn(1, 8, 64)
    s1, s2 = ch.sketch(h), ch.sketch(h)
    assert s1 == s2                                   # fixed projection seed -> verifier & node project identically
    assert s1["n"] == 8 * 64 and len(s1["proj"]) == min(256, 8 * 64)
    assert abs(s1["norm"] - float(h.flatten().norm())) < 1e-2


# ---- 3. compare: the discriminator (honest passes, cheat fails) ------------------------------------

def test_compare_identical_passes():
    h = torch.randn(1, 8, 64)
    r = ch.compare(h, h.clone())
    assert r["passed"] and r["cosine"] > 0.9999


def test_compare_honest_ulp_drift_passes():
    """Honest recompute on different hardware drifts a few ULPs — must PASS (else every honest node
    on heterogeneous GPUs false-fails)."""
    h = torch.randn(1, 16, 128)
    drift = h + 1e-4 * torch.randn_like(h)            # ~1e-4 relative — well inside the 0.99 threshold
    r = ch.compare(h, drift)
    assert r["passed"] and r["cosine"] > 0.99 and r["rel_norm"] < 0.05


def test_compare_garbage_fails():
    """A lazy node returning independent garbage -> cosine ~0 -> caught."""
    torch.manual_seed(0)
    h = torch.randn(1, 16, 128)
    garbage = torch.randn(1, 16, 128)
    r = ch.compare(h, garbage)
    assert not r["passed"] and r["cosine"] < 0.5


def test_compare_scaled_output_fails_on_rel_norm():
    """Same DIRECTION but wrong magnitude (a node that scales/half-runs the block): cosine ~1 but the
    relative-norm guard still fails it."""
    h = torch.randn(1, 8, 64)
    r = ch.compare(h, 2.0 * h)
    assert r["cosine"] > 0.9999 and r["rel_norm"] > 0.4 and not r["passed"]


def test_compare_partial_direction_fails():
    """A ~45-degree output (cosine ~0.7) is well below threshold -> fail."""
    v = torch.randn(4096)
    ortho = torch.randn(4096)
    ortho = ortho - (ortho @ v) / (v @ v) * v         # make it orthogonal to v
    mixed = v / v.norm() + ortho / ortho.norm()       # ~45 degrees from v (cosine ~0.707)
    r = ch.compare(v, mixed)
    assert 0.6 < r["cosine"] < 0.8 and not r["passed"]


def test_compare_works_on_sketches():
    h = torch.randn(1, 16, 128)
    assert ch.compare(ch.sketch(h), ch.sketch(h.clone()))["passed"]
    assert not ch.compare(ch.sketch(h), ch.sketch(torch.randn(1, 16, 128)))["passed"]


# ---- 4. challenge_block: honest recompute passes, a skipped/wrong block fails ----------------------

def _fake_blocks(monkeypatch):
    """Stub block_forward so challenge_block runs on CPU with synthetic 'blocks' (callables applying a
    transform to x) — no model weights. Returns nothing; the transforms are passed as `parts`."""
    monkeypatch.setattr(ch, "block_forward", lambda parts, x, start=0: parts(x))


def _linear(seed, hidden):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(hidden, hidden, generator=g).to(torch.bfloat16)   # match derive_challenge's bf16 input
    return lambda x: x @ w


def test_challenge_block_honest_passes(monkeypatch):
    _fake_blocks(monkeypatch)
    block = _linear(1, 32)                             # suspect and trusted run the SAME real block
    r = ch.challenge_block(block, block, "seed-x", 6, 32, device="cpu")
    assert r["passed"] and r["cosine"] > 0.9999


def test_challenge_block_lazy_constant_fails(monkeypatch):
    _fake_blocks(monkeypatch)
    trusted = _linear(1, 32)
    lazy = lambda x: torch.ones_like(x)               # node skips the matmuls, returns a cheap constant
    r = ch.challenge_block(lazy, trusted, "seed-x", 6, 32, device="cpu")
    assert not r["passed"]


def test_challenge_block_wrong_block_fails(monkeypatch):
    _fake_blocks(monkeypatch)
    trusted = _linear(1, 32)
    wrong = _linear(2, 32)                             # node ran a DIFFERENT block's weights
    r = ch.challenge_block(wrong, trusted, "seed-x", 6, 32, device="cpu")
    assert not r["passed"]


def test_challenge_block_uses_same_seeded_input_both_sides(monkeypatch):
    """Both sides must be fed the identical seeded input (only the transform is under test)."""
    seen = []
    monkeypatch.setattr(ch, "block_forward", lambda parts, x, start=0: seen.append(x.clone()) or parts(x))
    block = _linear(1, 32)
    ch.challenge_block(block, block, "seed-x", 6, 32, device="cpu")
    assert len(seen) == 2 and torch.equal(seen[0], seen[1])
