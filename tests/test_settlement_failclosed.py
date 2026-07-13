"""Fail-closed settlement (audit H2) — payment NEVER settles on a receipt set the swarm didn't assign.

The confirmed fail-open trio this pins shut:
  * UNKNOWN SIGNER: verify_coverage skipped the pinning check for any pubkey absent from
    expected_by_signer (`want is None` -> pass), so an interloper's validly-signed receipt for
    someone else's layers settled — and the assigned node's receipt could be swapped out entirely.
  * ZERO-WORK: n_chunks was written into every receipt but never validated — a receipt attesting
    0 chunks (no activations ever observed) settled like real work.
  * EMPTY MAP: settle() treated assignments={} as "pinning off", so the payment path silently
    degraded to unpinned verification.

Fixed semantics: expected_by_signer is not None IS payment mode — unknown signer rejected,
assigned set == received set, duplicate signers and zero-work receipts rejected in BOTH modes;
shard.verify.settle hard-requires a non-empty map (settle_unpinned is the named diagnostics path).

Run: python3 -m pytest tests/test_settlement_failclosed.py -q
"""
import base64
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from shard.receipt import ReceiptError, ReceiptSigner, _canonical, gen_key, pub_b64, verify_coverage
from shard.verify import settle, settle_unpinned

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_LAYERS = 62


def _receipt(key, lo, hi, n=4):
    s = ReceiptSigner(key, "swarm-test", "job-test", lo, hi)
    for i in range(n):
        s.observe(f"in{i}".encode(), f"out{i}".encode())
    return s.finalize()


def _ring(spans):
    """(receipts, assignments) — one distinct signer per stage, map matching the attested blocks."""
    keys = [gen_key() for _ in spans]
    receipts = [_receipt(k, lo, hi) for k, (lo, hi) in zip(keys, spans)]
    assignments = {pub_b64(k): [lo, hi] for k, (lo, hi) in zip(keys, spans)}
    return receipts, assignments


def test_happy_pinned_path():
    receipts, assignments = _ring([(0, 31), (31, 62)])
    out = settle(receipts, N_LAYERS, assignments=assignments)
    assert out["ok"] is True and out["pinned"] is True
    assert [s["layers"] for s in out["stages"]] == [31, 31]


def test_unknown_signer_rejected():
    """THE exploit: an interloper's validly-signed receipt for the same span replaces the assigned
    node's — the set still tiles, every signature verifies, and the old None-pass settled it."""
    receipts, assignments = _ring([(0, 31), (31, 62)])
    receipts[1] = _receipt(gen_key(), 31, 62)        # valid sig, signer NOT in the map
    with pytest.raises(ReceiptError, match="not in the assignment map"):
        settle(receipts, N_LAYERS, assignments=assignments)


def test_assigned_signer_missing_rejected():
    """Set equality: the map says 3 nodes ran the job; only 2 produced receipts (that still tile)."""
    receipts, assignments = _ring([(0, 31), (31, 62)])
    assignments[pub_b64(gen_key())] = [31, 62]       # a third assigned signer, silent
    with pytest.raises(ReceiptError, match="produced no receipt"):
        settle(receipts, N_LAYERS, assignments=assignments)


def test_duplicate_signer_rejected_both_modes():
    key = gen_key()
    receipts = [_receipt(key, 0, 31), _receipt(key, 31, 62)]  # one key credits itself twice
    with pytest.raises(ReceiptError, match="duplicate signer"):
        verify_coverage(receipts, N_LAYERS)                   # unpinned
    with pytest.raises(ReceiptError, match="duplicate signer"):
        settle(receipts, N_LAYERS,
               assignments={pub_b64(key): [0, 31]})           # pinned


def test_zero_work_receipt_rejected_both_modes():
    receipts, assignments = _ring([(0, 31)])
    receipts.append(_receipt(gen_key(), 31, 62, n=0))         # signed, tiles, attests ZERO chunks
    assignments[receipts[1]["pubkey"]] = [31, 62]
    with pytest.raises(ReceiptError, match="zero-work"):
        verify_coverage(receipts, N_LAYERS)
    with pytest.raises(ReceiptError, match="zero-work"):
        settle(receipts, N_LAYERS, assignments=assignments)


def test_missing_n_chunks_rejected():
    key = gen_key()
    r = _receipt(key, 0, 62)
    body = {k: v for k, v in r.items() if k not in ("sig", "n_chunks")}  # re-sign without n_chunks
    body["sig"] = base64.b64encode(key.sign(_canonical(body))).decode()
    with pytest.raises(ReceiptError, match="zero-work"):
        verify_coverage([body], N_LAYERS)


def test_settle_requires_assignment_map():
    receipts, _ = _ring([(0, 62)])
    with pytest.raises(TypeError):                            # positional-only call: map is mandatory
        settle(receipts, N_LAYERS)
    with pytest.raises(ReceiptError, match="non-empty assignment map"):
        settle(receipts, N_LAYERS, assignments={})            # the old {} -> pinning-off hole


def test_settle_unpinned_is_the_named_diagnostics_path():
    receipts, _ = _ring([(0, 31), (31, 62)])
    out = settle_unpinned(receipts, N_LAYERS)
    assert out["ok"] is True and out["pinned"] is False
    assert sum(s["layers"] for s in out["stages"]) == N_LAYERS


def _cli(req):
    r = subprocess.run([sys.executable, "-m", "shard.verify"], input=json.dumps(req),
                       capture_output=True, text=True, cwd=REPO, timeout=60)
    return r.returncode, json.loads(r.stdout)


def test_cli_payment_mode_requires_assignments():
    receipts, assignments = _ring([(0, 31), (31, 62)])
    rc, out = _cli({"receipts": receipts, "layer_count": N_LAYERS})     # default mode = payment
    assert rc == 2 and out["ok"] is False and "assignments" in out["error"]

    rc, out = _cli({"receipts": receipts, "layer_count": N_LAYERS, "assignments": assignments})
    assert rc == 0 and out["ok"] is True and out["pinned"] is True

    rc, out = _cli({"receipts": receipts, "layer_count": N_LAYERS, "mode": "unpinned"})
    assert rc == 0 and out["ok"] is True and out["pinned"] is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  {name} PASS")
    print("ALL PASS")
