"""Receipt freshness + chain binding (shard.receipt) — the TRUST moat, TIER 2.2.

A signed receipt was bound only to (swarm_id, job_id, layer span, activation roots), and the
coordinator's verify checked signatures + coverage tiling. Two gaps, both let a dishonest node get
paid without doing THIS job's work:
  - REPLAY: a receipt signed for an earlier job is still valid, so a node can re-submit it. Closed by
    a per-job random NONCE the coordinator issues on the reset frame; every stage signs it in, and the
    coordinator rejects a set whose nonce isn't this job's.
  - FABRICATED / SPLICED ROOTS: a node can sign self-consistent but bogus in_root/out_root. Closed by
    CHAINING — each block's out_root must equal the next block's in_root (the activation a node attests
    it output is what the next node attests it received). Holds by construction on the lossless wire.

Run: python3 -m pytest tests/test_receipt_binding.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shard.receipt import ReceiptError, ReceiptSigner, gen_key, verify_coverage, verify_receipt

N_LAYERS = 62
SPANS = [(0, 20), (20, 41), (41, 62)]


def _chained_ring(spans=SPANS, nonce=None, n=4, break_at=None):
    """A full receipt set over one synthetic job. There are len(spans)+1 activation streams a0..aS;
    stage i attests (in=a_i, out=a_{i+1}), so out_root[i] == in_root[i+1] by construction — a real
    lossless ring pass. `break_at=i` makes stage i attest a forged output (an individually well-signed
    receipt that does NOT chain), modelling a node with fabricated roots. `spans` must be sorted."""
    S = len(spans)
    streams = [[f"a{i}-c{c}".encode() for c in range(n)] for i in range(S + 1)]
    receipts = []
    for i, (lo, hi) in enumerate(spans):
        s = ReceiptSigner(gen_key(), "swarm", "job", lo, hi, nonce=nonce)
        out_stream = streams[i + 1]
        if break_at == i:
            out_stream = [b"forged-" + o for o in out_stream]
        for c in range(n):
            s.observe(streams[i][c], out_stream[c])
        receipts.append(s.finalize())
    return receipts


# ---- 1. the nonce is signed into the receipt -------------------------------------------------------

def test_nonce_is_in_signed_body():
    r = ReceiptSigner(gen_key(), "s", "j", 0, 62, nonce="deadbeef").finalize()
    assert r["nonce"] == "deadbeef"
    verify_receipt(r)                                    # signature covers the nonce
    r["nonce"] = "tampered"                              # flipping it must break the signature
    with pytest.raises(ReceiptError):
        verify_receipt(r)


def test_no_nonce_field_when_not_issued():
    r = ReceiptSigner(gen_key(), "s", "j", 0, 62).finalize()
    assert "nonce" not in r                               # back-compat: no nonce -> no field, old receipts verify


# ---- 2. freshness: the coordinator's per-job nonce rejects a replayed receipt ----------------------

def test_matching_nonce_passes():
    verify_coverage(_chained_ring(nonce="n-123"), N_LAYERS, expected_nonce="n-123")


def test_wrong_nonce_rejected_replay():
    """A receipt set from job A (nonce n-A) submitted for job B (nonce n-B) — the replay case."""
    ring = _chained_ring(nonce="n-A")
    with pytest.raises(ReceiptError):
        verify_coverage(ring, N_LAYERS, expected_nonce="n-B")


def test_absent_nonce_rejected_when_expected():
    ring = _chained_ring(nonce=None)                     # a pre-nonce (or nonce-stripped) receipt set
    with pytest.raises(ReceiptError):
        verify_coverage(ring, N_LAYERS, expected_nonce="n-live")


def test_no_nonce_check_when_none_expected():
    verify_coverage(_chained_ring(nonce=None), N_LAYERS)  # expected_nonce=None -> legacy behavior, passes


# ---- 3. chain binding: adjacent out_root == in_root ------------------------------------------------

def test_chain_intact_passes():
    verify_coverage(_chained_ring(), N_LAYERS, check_chain=True)


def test_chain_intact_passes_when_shuffled():
    ring = _chained_ring()
    ring = [ring[2], ring[0], ring[1]]                    # coordinator can't rely on receipt order
    verify_coverage(ring, N_LAYERS, check_chain=True)


@pytest.mark.parametrize("break_at", [0, 1])            # the two INTERIOR out->in edges (3 stages)
def test_chain_break_rejected(break_at):
    """Every receipt is individually well-signed (verify_receipt passes), but a node attested an
    output its neighbour never received — check_chain must reject the spliced/fabricated set."""
    ring = _chained_ring(break_at=break_at)
    for r in ring:
        verify_receipt(r)                                # each sig is valid on its own
    with pytest.raises(ReceiptError):
        verify_coverage(ring, N_LAYERS, check_chain=True)


def test_tail_output_forge_is_out_of_chain_scope():
    """HONEST BOUNDARY: forging the LAST stage's output has no downstream in_root to contradict, so
    the chain guard cannot catch it. The tail's final output is bound instead by the coordinator
    observing the actual reply tokens (lm_head argmax) — a separate binding, not this check. This
    documents the scope so it isn't mistaken for full end-to-end proof-of-compute."""
    verify_coverage(_chained_ring(break_at=2), N_LAYERS, check_chain=True)  # not raised — by design


def test_chain_break_ignored_when_check_off():
    """check_chain=False (the fp8-wire / legacy path) tiles fine despite a broken chain — documents
    that the chain guard is opt-in because fp8 activation transport is intentionally lossy."""
    verify_coverage(_chained_ring(break_at=1), N_LAYERS, check_chain=False)


# ---- 4. freshness + chain compose ------------------------------------------------------------------

def test_nonce_and_chain_together():
    ring = _chained_ring(nonce="live-nonce")
    verify_coverage(ring, N_LAYERS, expected_nonce="live-nonce", check_chain=True)
    with pytest.raises(ReceiptError):                    # a chained-but-stale-nonce replay is still caught
        verify_coverage(ring, N_LAYERS, expected_nonce="other", check_chain=True)
