"""C10 regression — the settlement authority must verify the receipts the engine actually emits.

Every stage tags its receipt with a coordinator-side `stage` debug label AFTER signing
({"stage": ..., **finalize()} in m25_pipe / specpipe). `receipt._canonical` signs every key but
`sig`, so that tag enters the verify preimage and breaks the signature. The engine's own verify
sites strip it; the SETTLEMENT path (shard.verify, run over the wire from coordinate.py) did NOT —
so a job settled `receiptsOk: true` at the engine and `InvalidSignature` at the payment authority,
over the SAME receipts, and nobody was paid.

These tests live in the quadrant no existing receipt test covered: a coordinator-EMITTED wire-shape
receipt fed to shard.verify. The fix strips the tag at the coordinator's SHARD_JOB_DONE boundary
(receipt.wire_receipt) so the settled dict is byte-identical to the signed body, while shard.verify
stays STRICT (any OTHER unexpected key is still rejected as tamper).

Run: python3 -m pytest tests/test_settlement_wire_shape.py -q
"""
import base64
import json
import os
import subprocess
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shard.receipt import ReceiptSigner, ReceiptError, _canonical, verify_receipt, wire_receipt  # noqa: E402
from shard.verify import settle  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pub(priv):
    return base64.b64encode(priv.public_key().public_bytes_raw()).decode()


def _wire_ring(nstages=3, layer_count=9, nonce="job-nonce-1", chunks=2):
    """A chained ring of receipts in the shape the ENGINE emits on the wire: each carries the
    coordinator-side `stage` tag added AFTER signing (middle stages an int, the tail "tail"),
    exactly like m25_pipe.py / specpipe.py. Returns (emitted, assignments, keys, nonce)."""
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
            out = f"act-{i}-{c}".encode()        # stage i's output == stage i+1's input (chain holds)
            signers[i].observe(prev, out)
            prev = out
    labels = list(range(nstages - 1)) + ["tail"]   # middle stages tag an int; the tail tags "tail"
    emitted = [{"stage": labels[i], **signers[i].finalize()} for i in range(nstages)]
    assignments = {_pub(keys[i]): [bounds[i][0], bounds[i][1]] for i in range(nstages)}
    return emitted, assignments, keys, nonce


# 1. RED: the receipt exactly as the engine emits it fails settlement (InvalidSignature); GREEN
#    once the coordinator strips the tag to the signed body via wire_receipt.
def test_engine_emitted_receipt_is_invalidsig_until_stage_stripped():
    emitted, assignments, _, nonce = _wire_ring(nstages=3, layer_count=9)

    # AS EMITTED (with `stage`) -> the settlement authority. Reproduces C10 AND proves shard.verify
    # stays strict: the extra key is in the signed preimage, so the signature does not verify.
    with pytest.raises(ReceiptError, match="InvalidSignature"):
        settle(emitted, 9, expected_nonce=nonce, check_chain=True, assignments=assignments)

    # THE FIX: the coordinator hands settlement wire_receipt(r) == the signed body -> it verifies.
    out = settle([wire_receipt(r) for r in emitted], 9,
                 expected_nonce=nonce, check_chain=True, assignments=assignments)
    assert out["ok"] is True
    assert sum(s["layers"] for s in out["stages"]) == 9


def test_engine_emitted_receipt_via_cli():
    """Same asymmetry over the real SubprocessSeam path (`python3 -m shard.verify`)."""
    emitted, assignments, _, nonce = _wire_ring(nstages=4, layer_count=12)

    def _run(receipts):
        req = {"receipts": receipts, "layer_count": 12, "expected_nonce": nonce,
               "check_chain": True, "assignments": assignments}
        r = subprocess.run([sys.executable, "-m", "shard.verify"], input=json.dumps(req),
                           capture_output=True, text=True, cwd=REPO, timeout=60)
        assert r.returncode == 0, r.stderr
        return json.loads(r.stdout)

    raw = _run(emitted)                                    # as the engine emits -> rejected
    assert raw["ok"] is False and "InvalidSignature" in raw["error"]
    wired = _run([wire_receipt(r) for r in emitted])       # coordinator-stripped -> settles
    assert wired["ok"] is True and len(wired["stages"]) == 4


# 2. The invariant: signed preimage == verified preimage for the wire shape, at every emit-site
#    shape (int stage and the tail's "stage": "tail").
@pytest.mark.parametrize("stage", [0, 3, "tail"])
def test_signed_preimage_equals_verified_preimage(stage):
    from cryptography.hazmat.primitives.asymmetric import ed25519
    key = ed25519.Ed25519PrivateKey.generate()
    s = ReceiptSigner(key, "swarm-1", "job-1", 0, 4, nonce="n1")
    s.observe(b"prompt", b"act")
    signed = s.finalize()                                  # the dict that was actually signed
    emitted = {"stage": stage, **signed}                   # what the engine puts on the wire
    wired = wire_receipt(emitted)

    assert wired == signed                                 # wire form == signed body, byte for byte
    assert _canonical(wired) == _canonical(signed)         # the preimage the verifier hashes == signed
    assert _canonical(emitted) != _canonical(signed)       # ...and the tag DID pollute it (the C10 root)
    verify_receipt(wired)                                  # signature valid over the wire preimage
    with pytest.raises(ReceiptError, match="InvalidSignature"):
        verify_receipt(emitted)                            # unstripped -> the observed failure


# 3. The fix did not make the verifier permissive: a genuinely tampered receipt STILL fails, and an
#    unexpected key that is NOT the known debug label is STILL rejected (tamper signal preserved).
def test_tampered_receipt_still_fails_after_fix():
    emitted, assignments, _, nonce = _wire_ring(nstages=3, layer_count=9)

    # Mutate a real, signed field on one receipt, then strip `stage` as the coordinator now does.
    tampered = [dict(r) for r in emitted]
    tampered[1]["out_root"] = "0" * 64                     # a real field the signature covers
    with pytest.raises(ReceiptError, match="InvalidSignature"):
        settle([wire_receipt(r) for r in tampered], 9,
               expected_nonce=nonce, check_chain=True, assignments=assignments)

    # A DIFFERENT unexpected key (not `stage`) is left in place and still breaks the signature —
    # wire_receipt strips only the known debug tag, so future tampering is not silently accepted.
    sneaky = [dict(r) for r in emitted]
    sneaky[0]["evil"] = "injected"
    with pytest.raises(ReceiptError, match="InvalidSignature"):
        settle([wire_receipt(r) for r in sneaky], 9,
               expected_nonce=nonce, check_chain=True, assignments=assignments)


# 4. Teeth on the actual chokepoint: drive coordinate.serve_jobs and assert its SHARD_JOB_DONE
#    carries stage-free receipts that settle. Reverting the coordinate.py strip fails this test.
def test_coordinator_emit_strips_stage_and_settles(monkeypatch):
    import shard.coordinate as C
    emitted, assignments, _, nonce = _wire_ring(nstages=3, layer_count=9)

    monkeypatch.setattr(C, "_StallWatchdog",
                        lambda *a, **k: types.SimpleNamespace(
                            arm=lambda *a, **k: None, tick=lambda *a, **k: None,
                            disarm=lambda *a, **k: None))
    monkeypatch.setattr(C, "run_job", lambda *a, **k: {
        "ok": True, "text": "hi", "n_tokens": 2, "receipts": emitted, "receipts_ok": True})

    captured = []
    def emit(tag, **fields):
        captured.append((tag, fields))

    MP = types.SimpleNamespace(EDGE_ERRORS=(), TransportError=RuntimeError)
    tok = types.SimpleNamespace(eos_token_id=0)
    a = types.SimpleNamespace(timeout=600, head=None, tail=None, connect_retry=60)
    line = json.dumps({"jobId": "j1", "nonce": nonce})

    C.serve_jobs(MP, tok, None, None, a, [line], emit=emit, redial=lambda: (None, None))

    done = [f for (t, f) in captured if t == "SHARD_JOB_DONE"]
    assert len(done) == 1
    receipts = done[0]["receipts"]
    assert all("stage" not in r for r in receipts)         # the debug tag never reached the wire
    out = settle(receipts, 9, expected_nonce=nonce, check_chain=True, assignments=assignments)
    assert out["ok"] is True                               # ...and what did settles at the authority
