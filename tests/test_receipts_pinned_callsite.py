"""H2 (call-site half) — the live coordinator verified receipts with NO assignment map: every
_verify_receipts call passed nothing to verify_coverage's expected_by_signer, so an interloper's
validly-signed receipt for someone else's layers settled (fail-open). Now _verify_receipts takes
`assignments` ({pubkey_b64: [lo, hi]}) and threads it as expected_by_signer, and every coordinator
call site loads it from SHARD_ASSIGNMENTS (the launcher-written map) via the cached
_load_assignments — env set = PINNED verdict, unset = today's unpinned verdict, announced once.

The fail-closed verify_coverage semantics themselves (unknown signer, set equality, zero-work
receipts) live in shard/receipt.py and are covered by its own tests; this file pins the PLUMBING —
the map actually reaching verify_coverage from the live path.

Run: python3 -m pytest tests/test_receipts_pinned_callsite.py -q
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
fr = pytest.importorskip("fake_ring")               # bootstraps env + m25_pipe on CPU

from receipt import ReceiptSigner, gen_key, pub_b64  # noqa: E402  (shard/ on path via fake_ring)

MP = fr.MP
N_LAYERS = 62


def _receipt(key, lo, hi, stage="s", n=4):
    s = ReceiptSigner(key, "swarm-test", "job-test", lo, hi)
    for i in range(n):
        s.observe(f"in{i}".encode(), f"out{i}".encode())
    return {"stage": stage, **s.finalize()}


def test_verify_receipts_threads_assignments(monkeypatch):
    """The map must reach verify_coverage as expected_by_signer with tuple blocks."""
    seen = {}

    def _fake_coverage(bodies, layer_count, expected_by_signer=None, expected_nonce=None, check_chain=False):
        seen["expected_by_signer"] = expected_by_signer
        seen["layer_count"] = layer_count

    monkeypatch.setattr(MP, "verify_coverage", _fake_coverage)
    assert MP._verify_receipts([], N_LAYERS, assignments={"PK": [0, 31], "PK2": [31, 62]}) is True
    assert seen["expected_by_signer"] == {"PK": (0, 31), "PK2": (31, 62)}
    seen.clear()
    assert MP._verify_receipts([], N_LAYERS) is True
    assert seen["expected_by_signer"] is None        # no map -> unpinned, exactly the old behavior


def test_pinned_verdict_rejects_misassigned_signer():
    """End to end with REAL signed receipts: a signer attesting a block it was never assigned must
    flip receipts_ok to False once the map is passed (it settled without one)."""
    k1, k2 = gen_key(), gen_key()
    receipts = [_receipt(k1, 0, 31, "0"), _receipt(k2, 31, 62, "tail")]
    assert MP._verify_receipts(receipts, N_LAYERS) is True   # unpinned: tiles + signatures pass
    good = {pub_b64(k1): [0, 31], pub_b64(k2): [31, 62]}
    assert MP._verify_receipts(receipts, N_LAYERS, assignments=good) is True
    swapped = {pub_b64(k1): [31, 62], pub_b64(k2): [0, 31]}  # same tiling, wrong owners
    assert MP._verify_receipts(receipts, N_LAYERS, assignments=swapped) is False


def test_load_assignments_env(tmp_path, monkeypatch):
    amap = {"PK": [0, 62]}
    f = tmp_path / "assignments.json"
    f.write_text(json.dumps(amap))

    monkeypatch.setattr(MP, "_ASSIGNMENTS", "unset")
    monkeypatch.setenv("SHARD_ASSIGNMENTS", str(f))
    assert MP._load_assignments() == amap
    assert MP._load_assignments() == amap            # cached

    monkeypatch.setattr(MP, "_ASSIGNMENTS", "unset")
    monkeypatch.delenv("SHARD_ASSIGNMENTS", raising=False)
    assert MP._load_assignments() is None            # unset -> unpinned (announced once)

    monkeypatch.setattr(MP, "_ASSIGNMENTS", "unset")
    monkeypatch.setenv("SHARD_ASSIGNMENTS", str(tmp_path / "missing.json"))
    assert MP._load_assignments() is None            # unreadable -> warned, unpinned
