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

try:                                                          # package import (engine repo)
    from .manifest import gen_key, load_key, pub_b64, save_key  # noqa: F401
except ImportError:                                           # flat import (deployed next to the node code)
    from manifest import gen_key, load_key, pub_b64, save_key  # noqa: F401

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
                 layer_start: int, layer_end: int, nonce: str | None = None):
        self.priv = priv
        self.meta = {"swarm_id": swarm_id, "job_id": job_id,
                     "layer_start": layer_start, "layer_end": layer_end}
        if nonce is not None:                        # per-JOB freshness challenge (coordinator-issued,
            self.meta["nonce"] = nonce               # random): signed into the receipt so a replayed
        self._in = hashlib.sha256()                  # receipt from an earlier job carries a stale nonce
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


def load_or_make_node_key(path: str) -> ed25519.Ed25519PrivateKey:
    """The node's stable signing identity. Loaded from `path`, or generated + persisted on first
    use (0600). In production this is the same ed25519 key behind the node's libp2p PeerId (bound
    to a c0mpute account in step 2.3); here a per-node key file stands in for the demo."""
    import os
    if os.path.exists(path):
        return load_key(path)
    key = gen_key()
    save_key(key, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return key


def verify_coverage(receipts: list[dict], layer_count: int,
                    expected_by_signer: dict | None = None,
                    expected_nonce: str | None = None,
                    check_chain: bool = False) -> None:
    """The job-level check c0mpute runs before paying: the set of per-stage receipts must
    (1) each verify, (2) tile [0, layer_count) with NO gap or overlap (every layer was attested
    by exactly one node), (3) — with expected_nonce — carry the coordinator's per-JOB freshness
    challenge, so a receipt replayed from an earlier job (stale/absent nonce) is rejected, and
    (4) — with check_chain — CHAIN: each block's out_root equals the next block's in_root, i.e. the
    activation a node attests it output is byte-identical to what the next node attests it received.
    Chaining catches fabricated roots and binds the receipt set to one real ring pass; it holds by
    construction only on the LOSSLESS wire (the caller passes check_chain=not fp8_wire, since fp8
    activation transport is intentionally lossy). expected_by_signer maps pubkey -> the block the
    swarm assigned it; passing it IS payment (pinned) mode and fails CLOSED: a signer absent from
    the map is rejected, and every assigned signer must produce exactly one receipt (set equality).
    In both modes a duplicate signer or a zero-work receipt (n_chunks missing/<=0) is rejected."""
    entries = []
    seen_pubkeys = set()
    for r in receipts:
        verify_receipt(r, None)
        if expected_nonce is not None and r.get("nonce") != expected_nonce:
            raise ReceiptError(f"receipt nonce {r.get('nonce')!r} != job nonce (stale or replayed receipt)")
        lo, hi = r["layer_start"], r["layer_end"]
        if not (0 <= lo < hi <= layer_count):
            raise ReceiptError(f"receipt block [{lo}:{hi}] outside [0:{layer_count}]")
        if not isinstance(r.get("n_chunks"), int) or r["n_chunks"] <= 0:
            raise ReceiptError(f"receipt for [{lo}:{hi}] attests {r.get('n_chunks')!r} chunks (zero-work receipt)")
        pub = r["pubkey"]
        if pub in seen_pubkeys:                      # one receipt per identity — a key signing two
            raise ReceiptError(f"duplicate signer {pub[:12]}..")  # blocks is double-crediting
        seen_pubkeys.add(pub)
        if expected_by_signer is not None:
            want = expected_by_signer.get(pub)
            if want is None:                         # fail CLOSED: an interloper's validly-signed
                raise ReceiptError(f"signer {pub[:12]}.. is not in the assignment map")  # receipt never settles
            if tuple(want) != (lo, hi):
                raise ReceiptError(f"signer {pub[:12]}.. attested [{lo}:{hi}], assigned {tuple(want)}")
        entries.append((lo, hi, r))
    if expected_by_signer is not None:               # assigned set == received set (extras died above)
        missing = set(expected_by_signer) - seen_pubkeys
        if missing:
            raise ReceiptError(f"assigned signer(s) produced no receipt: {sorted(p[:12] for p in missing)}")
    entries.sort(key=lambda e: e[0])
    cursor = 0
    for lo, hi, _ in entries:
        if lo != cursor:
            raise ReceiptError(f"layer coverage broken at {cursor}: next block starts {lo} (gap or overlap)")
        cursor = hi
    if cursor != layer_count:
        raise ReceiptError(f"layer coverage ends at {cursor}, expected {layer_count}")
    if check_chain:                                  # entries are now sorted + provably contiguous
        for (lo_a, hi_a, ra), (lo_b, hi_b, rb) in zip(entries, entries[1:]):
            if ra["out_root"] != rb["in_root"]:
                raise ReceiptError(
                    f"chain break: block [{lo_a}:{hi_a}] out_root {ra['out_root'][:12]} != "
                    f"block [{lo_b}:{hi_b}] in_root {rb['in_root'][:12]} — an attested output is not "
                    f"what the next stage attests it received (fabricated roots or a spliced receipt)")
