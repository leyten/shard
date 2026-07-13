"""shard-manifest/1 — the signed, content-addressed weight manifest (step 3, JOIN).

A manifest is the trust root for a model's weights. A node that holds only the
manifest (and the publisher's pubkey, which c0mpute's catalog pins) can fetch its
layer block's shards from *any* provider and verify every byte before it touches
VRAM. A malicious mirror or peer physically cannot feed corrupted weights — the
sha256 won't match and the load fails closed.

Per the boundary law (docs/INTEGRATION.md §4) this is pure engine. It knows nothing
about c0mpute's catalog, accounts, or $ZERO: c0mpute stores a *pointer* to a manifest
plus the trusted publisher pubkey; shard defines, signs, and verifies the manifest.

Schema (`schema: "shard-manifest/1"`):

    {
      "schema": "shard-manifest/1",
      "model_id":   "openai/gpt-oss-120b",
      "arch":       "GptOssForCausalLM",
      "layer_count": 36,
      "tokenizer":  "openai/gpt-oss-120b",
      "weight_map": { "<weight name>": "<filename>", ... },  # signed: layer->file is authoritative
      "shards": [ {"shard_id": "<cidv1>", "path": "<filename>",
                   "sha256": "<hex>", "size": <int>,
                   "kind": "weights" | "config" | "tokenizer"}, ... ],
      "publisher_pubkey": "<base64 raw ed25519>",
      "signature":        "<base64 raw ed25519 over canonical(manifest \\ signature)>"
    }

`shard_id` is the CIDv1 (raw codec, sha2-256) of the file content — both a stable
content address and the exact key libp2p content routing will use when P2P
propagation takes over (step 8). It is derived from sha256, so verifying the sha256
verifies the CID. The same value is the contract the Go sidecar's `-fetch-cid`
already validates.
"""
import base64
import copy
import hashlib
import json
import os
import stat

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519

SCHEMA = "shard-manifest/1"
_CHUNK = 1 << 20  # 1 MiB hashing stride


class ManifestError(Exception):
    """A manifest failed to verify — bad signature, wrong publisher, or malformed.
    Always raised (never a silent False) so callers fail closed."""


# ── content addressing ────────────────────────────────────────────────────────
def cidv1_raw(sha256_hex: str) -> str:
    """CIDv1, raw codec (0x55), multihash sha2-256 (0x12 0x20 || digest), base32-lower.
    A self-contained implementation (no multiformats dep) — produces the canonical
    `bafkrei…` content id libp2p uses, so step-8 P2P routing keys by this directly."""
    digest = bytes.fromhex(sha256_hex)
    if len(digest) != 32:
        raise ValueError("sha256 digest must be 32 bytes")
    cid_bytes = b"\x01\x55\x12\x20" + digest  # version, codec(raw), mh(sha2-256, len 32)
    return "b" + base64.b32encode(cid_bytes).decode().lower().rstrip("=")


def sha256_file(path: str) -> tuple[str, int]:
    """(hex digest, byte size) of a file, streamed so multi-GB shards don't blow RAM."""
    h, size = hashlib.sha256(), 0
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


# ── signing / verifying ───────────────────────────────────────────────────────
def canonical(manifest: dict) -> bytes:
    """Deterministic bytes signed over: the manifest minus its signature field, JSON
    with sorted keys and no incidental whitespace. publisher_pubkey IS covered, so a
    bare pubkey swap breaks the signature (a full forge is stopped by the catalog pin)."""
    m = {k: v for k, v in manifest.items() if k != "signature"}
    return json.dumps(m, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign_manifest(manifest: dict, priv: ed25519.Ed25519PrivateKey) -> dict:
    """Stamp publisher_pubkey + signature into a copy of `manifest` and return it."""
    m = copy.deepcopy(manifest)
    m.pop("signature", None)
    m["publisher_pubkey"] = base64.b64encode(
        priv.public_key().public_bytes_raw()).decode()
    m["signature"] = base64.b64encode(priv.sign(canonical(m))).decode()
    return m


def verify_manifest(manifest: dict, expected_pubkey: str | None = None) -> None:
    """Fail closed: raise ManifestError unless the signature is valid AND (if given)
    publisher_pubkey equals the catalog-pinned `expected_pubkey`. The pin is what stops
    a mirror swapping in its own validly-signed manifest — c0mpute supplies it."""
    if manifest.get("schema") != SCHEMA:
        raise ManifestError(f"unknown manifest schema {manifest.get('schema')!r}")
    pub_b64 = manifest.get("publisher_pubkey")
    sig_b64 = manifest.get("signature")
    if not pub_b64 or not sig_b64:
        raise ManifestError("manifest is unsigned")
    if expected_pubkey is not None and pub_b64 != expected_pubkey:
        raise ManifestError("publisher pubkey does not match the pinned key")
    try:
        pub = ed25519.Ed25519PublicKey.from_public_bytes(base64.b64decode(pub_b64))
        pub.verify(base64.b64decode(sig_b64), canonical(manifest))
    except (InvalidSignature, ValueError, Exception) as e:  # noqa: B014 — fail closed on anything
        raise ManifestError(f"signature verification failed: {type(e).__name__}") from e


# ── publisher key helpers ─────────────────────────────────────────────────────
def gen_key() -> ed25519.Ed25519PrivateKey:
    return ed25519.Ed25519PrivateKey.generate()


def save_key(priv: ed25519.Ed25519PrivateKey, path: str) -> None:
    """Persist the raw 32-byte private key (base64). Keep this file secret — anyone
    holding it can publish manifests that nodes will trust for the pinned pubkey.
    Created EXCLUSIVELY at 0600: a plain open('w') under umask 022 lands 0644, exposing
    the trust root to every local user, and silently clobbers an existing key."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(base64.b64encode(priv.private_bytes_raw()).decode())


def load_key(path: str) -> ed25519.Ed25519PrivateKey:
    st = os.stat(path)
    if stat.S_ISREG(st.st_mode) and st.st_mode & 0o077:
        raise ManifestError(
            f"publisher key {path} is group/world accessible (mode {stat.S_IMODE(st.st_mode):o}) "
            "— chmod 600 it; anyone reading it can sign manifests nodes trust")
    with open(path) as f:
        return ed25519.Ed25519PrivateKey.from_private_bytes(base64.b64decode(f.read().strip()))


def pub_b64(priv: ed25519.Ed25519PrivateKey) -> str:
    return base64.b64encode(priv.public_key().public_bytes_raw()).decode()
