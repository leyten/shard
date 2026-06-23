"""Signed per-stage receipts (PROVE) — the engine-side attestation that a node ran its block.

Over a job, each stage chains the sha256 of every (input, output) activation it processes and
signs the pair of running roots with its node key. The coordinator collects one receipt per stage
and hands the set to c0mpute on job:complete. Two properties fall out, both about *trustworthy
output and honest pay*:

  * a node cannot be PAID without producing a receipt signed by ITS key, and
  * the coordinator cannot FABRICATE a node's receipt (it lacks the key) — this kills
    coordinator-takes-all: pay is attributed per signed receipt, not by the coordinator's word.

The (in_root, out_root) slot is exactly where a cheap cryptographic proof-of-compute drops in
later — the economic-now -> crypto-later seam (docs/INTEGRATION.md §6). Today the roots are an
audit trail + the binding a layer-block challenge (shard/challenge.py) checks against; tomorrow
they carry a succinct proof that out = block(in) without re-execution.

Pure engine (boundary law): knows activations, hashes, node keys — nothing about c0mpute
accounts, $ZERO, or payment. shard PRODUCES the receipt; c0mpute CONSUMES it.

The node key is the node's ed25519 identity (the same key behind its libp2p PeerId, bound to a
c0mpute account in step 2.3). Receipt pubkey -> PeerId -> account is how c0mpute attributes pay.
"""
import base64
import hashlib
import json

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519

from .manifest import gen_key, load_key, pub_b64, save_key  # noqa: F401  (shared ed25519 key helpers)

SCHEMA = "shard-receipt/1"


class ReceiptError(Exception):
    """A receipt failed to verify — bad signature, wrong signer, or malformed. Always raised
    (never a silent False) so a caller attributing pay fails closed."""


def _h(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _canonical(receipt: dict) -> bytes:
    """Deterministic bytes signed over: the receipt minus its signature, sorted keys, no
    incidental whitespace. pubkey IS covered, so a bare pubkey swap breaks the signature."""
    m = {k: v for k, v in receipt.items() if k != "sig"}
    return json.dumps(m, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class ReceiptSigner:
    """Accumulates the activation hash-chain for ONE stage over ONE job, then signs it.

    observe() is on the hot path: it hashes only the small per-chunk activation tensor (tens of
    KB for a decode chunk; a few MB for a prefill chunk), so the cost is negligible vs the WAN
    ring. Chaining sha256(in)/sha256(out) per chunk yields an order-sensitive root over the whole
    job — a node that skipped or altered any chunk produces a different root and is caught."""

    def __init__(self, priv: ed25519.Ed25519PrivateKey, swarm_id: str, job_id: str,
                 layer_start: int, layer_end: int):
        self.priv = priv
        self.meta = {"swarm_id": swarm_id, "job_id": job_id,
                     "layer_start": layer_start, "layer_end": layer_end}
        self._in = hashlib.sha256()
        self._out = hashlib.sha256()
        self.n = 0

    def observe(self, in_bytes: bytes, out_bytes: bytes) -> None:
        self._in.update(_h(in_bytes))
        self._out.update(_h(out_bytes))
        self.n += 1

    def finalize(self) -> dict:
        """Stamp pubkey + signature into a signed receipt dict and return it."""
        body = dict(self.meta, schema=SCHEMA, n_chunks=self.n,
                    in_root=self._in.hexdigest(), out_root=self._out.hexdigest(),
                    pubkey=base64.b64encode(self.priv.public_key().public_bytes_raw()).decode())
        body["sig"] = base64.b64encode(self.priv.sign(_canonical(body))).decode()
        return body


def verify_receipt(receipt: dict, expected_pubkey: str | None = None) -> None:
    """Fail closed: raise ReceiptError unless the signature is valid AND (if given) the signer's
    pubkey equals expected_pubkey (the key c0mpute bound to the node assigned this block)."""
    if receipt.get("schema") != SCHEMA:
        raise ReceiptError(f"unknown receipt schema {receipt.get('schema')!r}")
    pub_b64 = receipt.get("pubkey")
    sig_b64 = receipt.get("sig")
    if not pub_b64 or not sig_b64:
        raise ReceiptError("receipt is unsigned")
    if expected_pubkey is not None and pub_b64 != expected_pubkey:
        raise ReceiptError("receipt signer is not the node assigned this block")
    try:
        pub = ed25519.Ed25519PublicKey.from_public_bytes(base64.b64decode(pub_b64))
        pub.verify(base64.b64decode(sig_b64), _canonical(receipt))
    except (InvalidSignature, ValueError, Exception) as e:  # noqa: B014 — fail closed on anything
        raise ReceiptError(f"signature verification failed: {type(e).__name__}") from e


def verify_coverage(receipts: list[dict], layer_count: int,
                    expected_by_signer: dict | None = None) -> None:
    """The job-level check c0mpute runs before paying: the set of per-stage receipts must
    (1) each verify, (2) tile [0, layer_count) with NO gap or overlap (every layer was attested
    by exactly one node), and (3) chain — each stage's in_root continuity is the coordinator's to
    assert across the ring; here we enforce the coverage tiling, which is what stops a node from
    being paid for a block it didn't hold. expected_by_signer maps pubkey -> the block c0mpute
    assigned it (optional pinning)."""
    spans = []
    for r in receipts:
        verify_receipt(r, None)
        lo, hi = r["layer_start"], r["layer_end"]
        if not (0 <= lo < hi <= layer_count):
            raise ReceiptError(f"receipt block [{lo}:{hi}] outside [0:{layer_count}]")
        if expected_by_signer is not None:
            want = expected_by_signer.get(r["pubkey"])
            if want is not None and tuple(want) != (lo, hi):
                raise ReceiptError(f"signer {r['pubkey'][:12]}.. attested [{lo}:{hi}], assigned {tuple(want)}")
        spans.append((lo, hi))
    spans.sort()
    cursor = 0
    for lo, hi in spans:
        if lo != cursor:
            raise ReceiptError(f"layer coverage broken at {cursor}: next block starts {lo} (gap or overlap)")
        cursor = hi
    if cursor != layer_count:
        raise ReceiptError(f"layer coverage ends at {cursor}, expected {layer_count}")
