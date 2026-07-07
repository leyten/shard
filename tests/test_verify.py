"""shard.verify — the settlement seam c0mpute runs before per-shard-per-token pay.

Builds a real signed, chained ring of receipts (one per stage) and checks that `settle` and the
`python3 -m shard.verify` CLI accept an honest set (returning the per-stage split for the metering
fan-out) and REJECT the ways a node tries to get paid dishonestly: a replayed (stale-nonce) receipt,
a coverage gap, and a signer attesting a block it wasn't assigned.

Run: python3 -m pytest tests/test_verify.py -q
"""
import base64
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shard.receipt import ReceiptSigner  # noqa: E402
from shard.verify import settle  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pub(priv):
    return base64.b64encode(priv.public_key().public_bytes_raw()).decode()


def _ring(nstages=3, layer_count=9, nonce="job-nonce-1", chunks=2):
    """A chained ring: stage i's out activation per chunk == stage i+1's in, so out_root[i] ==
    in_root[i+1] (the lossless-wire chain holds). Returns (receipts, assignments, keys)."""
    from cryptography.hazmat.primitives.asymmetric import ed25519
    keys = [ed25519.Ed25519PrivateKey.generate() for _ in range(nstages)]
    base, bounds, lo = layer_count // nstages, [], 0
    for i in range(nstages):
        hi = layer_count if i == nstages - 1 else lo + base
        bounds.append((lo, hi)); lo = hi
    signers = [ReceiptSigner(keys[i], "swarm-1", "job-1", bounds[i][0], bounds[i][1], nonce=nonce)
               for i in range(nstages)]
    for c in range(chunks):
        prev = f"prompt-{c}".encode()
        for i in range(nstages):
            out = f"act-{i}-{c}".encode()        # the activation stage i outputs == stage i+1's input
            signers[i].observe(prev, out)
            prev = out
    receipts = [s.finalize() for s in signers]
    assignments = {_pub(keys[i]): [bounds[i][0], bounds[i][1]] for i in range(nstages)}
    return receipts, assignments, keys


def test_honest_set_settles_with_per_stage_split():
    receipts, assignments, _ = _ring(nstages=3, layer_count=9)
    out = settle(receipts, 9, expected_nonce="job-nonce-1", check_chain=True, assignments=assignments)
    assert out["ok"] is True
    assert [s["lo"] for s in out["stages"]] == [0, 3, 6]     # sorted, tiling [0:9)
    assert [s["layers"] for s in out["stages"]] == [3, 3, 3]
    assert sum(s["layers"] for s in out["stages"]) == 9      # the split the metering fan-out consumes


def test_replayed_nonce_is_rejected():
    receipts, assignments, _ = _ring(nonce="OLD-job-nonce")   # receipts from a previous job
    import pytest
    from shard.receipt import ReceiptError
    with pytest.raises(ReceiptError):
        settle(receipts, 9, expected_nonce="THIS-job-nonce", check_chain=True, assignments=assignments)


def test_coverage_gap_is_rejected():
    receipts, assignments, _ = _ring(nstages=3, layer_count=9)
    import pytest
    from shard.receipt import ReceiptError
    with pytest.raises(ReceiptError):                          # drop the middle stage -> a hole in [0:9)
        settle([receipts[0], receipts[2]], 9, expected_nonce="job-nonce-1", assignments=assignments)


def test_signer_attesting_unassigned_block_is_rejected():
    receipts, assignments, keys = _ring(nstages=3, layer_count=9)
    # swap the assignment for stage 0's signer to a block it did NOT attest
    assignments[_pub(keys[0])] = [3, 6]
    import pytest
    from shard.receipt import ReceiptError
    with pytest.raises(ReceiptError):
        settle(receipts, 9, expected_nonce="job-nonce-1", assignments=assignments)


def test_cli_roundtrip_accepts_and_rejects():
    receipts, assignments, _ = _ring(nstages=4, layer_count=12)
    req = {"receipts": receipts, "layer_count": 12, "expected_nonce": "job-nonce-1",
           "check_chain": True, "assignments": assignments}
    r = subprocess.run([sys.executable, "-m", "shard.verify"], input=json.dumps(req),
                       capture_output=True, text=True, cwd=REPO, timeout=60)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["ok"] is True and len(out["stages"]) == 4

    req["expected_nonce"] = "WRONG"                            # a rejected set is a verdict, exit 0 ok=false
    r = subprocess.run([sys.executable, "-m", "shard.verify"], input=json.dumps(req),
                       capture_output=True, text=True, cwd=REPO, timeout=60)
    assert r.returncode == 0
    out = json.loads(r.stdout)
    assert out["ok"] is False and "nonce" in out["error"]
