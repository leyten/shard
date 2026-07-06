"""Regression tests for the receipt coverage check (shard.receipt.verify_coverage) — the TRUST fix.

The fleet-confirmed CRITICAL hole: callers derived `layer_count` FROM the receipts being verified
(max(layer_end)), so a ring that OMITS layers still "tiled fully" and passed — a node could skip its
block and still be paid. The check itself was always sound; the caller's target was self-referential.
These tests pin the fixed semantics: coverage is verified against the model's TRUE depth, and a
truncated / gapped / overlapping / mis-assigned set of receipts fails CLOSED (raises ReceiptError).

Run: `python3 tests/test_receipt_coverage.py`  (also collectable by pytest as test_*).
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shard"))

import pytest
from shard.receipt import ReceiptError, ReceiptSigner, gen_key, pub_b64, verify_coverage

N_LAYERS = 62  # MiniMax-M2.5's true depth — what the coordinator must pass, never max(layer_end)


def _receipt(key, lo, hi, n=4):
    """One signed receipt attesting layers [lo:hi) with a tiny synthetic activation chain."""
    s = ReceiptSigner(key, "swarm-test", "job-test", lo, hi)
    for i in range(n):
        s.observe(f"in{i}".encode(), f"out{i}".encode())
    return s.finalize()


def _ring(spans):
    """A full set of receipts, one distinct signer per stage."""
    return [_receipt(gen_key(), lo, hi) for lo, hi in spans]


def test_full_tiling_passes():
    receipts = _ring([(0, 20), (20, 41), (41, 62)])
    verify_coverage(receipts, N_LAYERS)  # must not raise


def test_truncated_ring_fails_against_true_depth():
    """THE hole: a ring that never attested layers 40..62. Against the self-referential target
    (max(layer_end)=40) it tiles 'fully'; against the true depth it must fail."""
    receipts = _ring([(0, 20), (20, 40)])
    verify_coverage(receipts, 40)  # documents the old bug: self-derived target passes
    with pytest.raises(ReceiptError):
        verify_coverage(receipts, N_LAYERS)  # the fix: true depth fails closed


def test_gap_fails():
    with pytest.raises(ReceiptError):
        verify_coverage(_ring([(0, 20), (30, 62)]), N_LAYERS)


def test_overlap_fails():
    with pytest.raises(ReceiptError):
        verify_coverage(_ring([(0, 30), (20, 62)]), N_LAYERS)


def test_zero_length_block_fails():
    """A zero-length block [lo:lo) attests nothing but tiles cleanly (its neighbours meet at lo),
    so it slips past the gap/overlap cursor. The per-block range check must reject it (lo < hi),
    or a signer is in the paid tiling for a block it never held."""
    with pytest.raises(ReceiptError):
        verify_coverage(_ring([(0, 31), (31, 31), (31, 62)]), N_LAYERS)


def test_block_outside_model_fails():
    """A receipt attesting layers beyond the model's depth is as fake as a missing one."""
    with pytest.raises(ReceiptError):
        verify_coverage(_ring([(0, 62), (62, 70)]), N_LAYERS)


def test_signer_pinning_mismatch_fails():
    """expected_by_signer: a node attesting a block it was not assigned must fail."""
    key_a, key_b = gen_key(), gen_key()
    receipts = [_receipt(key_a, 0, 31), _receipt(key_b, 31, 62)]
    assigned = {pub_b64(key_a): (0, 31), pub_b64(key_b): (31, 62)}
    verify_coverage(receipts, N_LAYERS, expected_by_signer=assigned)  # correct assignment passes
    swapped = {pub_b64(key_a): (31, 62), pub_b64(key_b): (0, 31)}
    with pytest.raises(ReceiptError):
        verify_coverage(receipts, N_LAYERS, expected_by_signer=swapped)


def test_tampered_receipt_fails():
    receipts = _ring([(0, 31), (31, 62)])
    receipts[0]["layer_end"] = 62  # widen the attested block without re-signing
    with pytest.raises(ReceiptError):
        verify_coverage(receipts, N_LAYERS)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  {name} PASS")
    print("ALL PASS")
