"""Content-addressed weight fetch (step 3, JOIN) — selective and byte-verified.

A node fetches ONLY its layer block's shards, from a pluggable provider, and verifies
every byte against the signed manifest before it lands in the model dir. Two properties
make this the trust root for weights:

  * selective  — a stage holding layers [lo:hi) downloads only the safetensors files
                 those layers (plus its boundary weights) live in, not the whole model.
                 NB byte savings track the checkpoint's shard packing, not 1/N: gpt-oss's
                 MXFP4 weight_map scatters layers across files (some straddle 3), so a
                 4-way split has stage 0 pull ~60% of bytes, not 25%. A future
                 re-shard-on-publish (contiguous layer order) recovers the full saving;
                 correctness + the trust property are unaffected either way.
  * verified   — fetch_block re-hashes every file itself, so a provider is never trusted.
                 A wrong byte → sha256 mismatch → the file is deleted and the load fails
                 closed. Corrupted weights cannot reach VRAM.

The **provider is a seam** (docs/INTEGRATION.md §4). A mirror (HTTP/HF) is the first
provider; libp2p content routing takes over at step 8 — additive, because the fetch was
content-verified from day one, so swapping the source changes nothing about trust.

Per the boundary law: pure engine. Knows about manifests, shards, and bytes — nothing
about c0mpute's catalog or accounts (the caller passes the pinned publisher pubkey in).
"""
import os
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from urllib.request import Request, urlopen

from . import manifest as mf


class FetchError(Exception):
    """A shard could not be fetched or failed verification. Fail closed."""


class ProviderUnavailable(Exception):
    """This provider can't serve right now — caller may fall back to another."""


def _log(msg: str) -> None:
    print(f"[fetch] {msg}", flush=True)


# ── providers (the source seam) ───────────────────────────────────────────────
class Provider(ABC):
    """Delivers a shard's bytes to `dest`. Verification is NOT the provider's job —
    fetch_block always re-hashes — so a buggy or hostile provider cannot bypass it."""

    @abstractmethod
    def fetch(self, shard: dict, dest: str) -> None: ...


class MirrorProvider(Provider):
    """The first provider: a plain HTTP mirror. For Hugging Face, base_url is
    `https://huggingface.co/<repo>/resolve/main/`. Resumable (HTTP Range) so a dropped
    5 GB download continues instead of restarting; the full-file hash in fetch_block
    catches any bad resume."""

    def __init__(self, base_url: str, headers: dict | None = None, retries: int = 6):
        self.base = base_url.rstrip("/") + "/"
        self.headers = headers or {}
        self.retries = retries

    def fetch(self, shard: dict, dest: str) -> None:
        url = self.base + shard["path"]
        part = dest + ".part"
        for attempt in range(self.retries):
            try:
                self._download(url, part, shard["size"])
                os.replace(part, dest)
                return
            except Exception as e:
                wait = min(60, 5 * (attempt + 1))
                _log(f"  retry {shard['path']} ({attempt + 1}/{self.retries}) "
                     f"after {wait}s: {str(e)[:80]}")
                time.sleep(wait)
        raise FetchError(f"mirror could not fetch {shard['path']} after {self.retries} tries")

    def _download(self, url: str, part: str, total: int) -> None:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > total:  # stale/corrupt partial — start over
            os.remove(part)
            have = 0
        req = Request(url, headers={"User-Agent": "shard/1", **self.headers})
        if have:
            req.add_header("Range", f"bytes={have}-")
        with urlopen(req, timeout=120) as r:
            resumed = have > 0 and getattr(r, "status", 200) == 206
            with open(part, "ab" if resumed else "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)


class LocalDirProvider(Provider):
    """A local directory acts as the mirror — used by the self-test, and as a
    same-host seed source. Same verification path; the source just happens to be a copy."""

    def __init__(self, root: str):
        self.root = root

    def fetch(self, shard: dict, dest: str) -> None:
        src = os.path.join(self.root, shard["path"])
        if not os.path.exists(src):
            raise FetchError(f"local source missing {shard['path']}")
        shutil.copyfile(src, dest)


class Libp2pProvider(Provider):
    """Fetch a shard by its CID over libp2p content routing, via the Go sidecar.

    The *contract* — shard_id is a CIDv1(raw, sha2-256) the sidecar validates — is live
    now (`sidecar -fetch-cid`). The *transfer* (DHT find-providers + block exchange)
    lands in step 8; until then fetch() raises ProviderUnavailable so the caller falls
    back to the mirror. This is the seam, wired end-to-end and stubbed at the edge — not
    a rewrite waiting to happen."""

    def __init__(self, sidecar_bin: str = "sidecar", key: str | None = None):
        self.bin = sidecar_bin
        self.key = key

    def fetch(self, shard: dict, dest: str) -> None:
        cmd = [self.bin, "-fetch-cid", shard["shard_id"]]
        if self.key:
            cmd += ["-key", self.key]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            detail = (r.stdout or r.stderr).strip()
        except FileNotFoundError:
            detail = f"sidecar binary {self.bin!r} not found"
        raise ProviderUnavailable(f"libp2p content routing not enabled (step 8): {detail}")


# ── block resolution (mirrors pipeline.load_stage) ────────────────────────────
def block_for_stage(n_layers: int, stage: int, nstages: int) -> tuple[int, int]:
    """[lo, hi) layer range for a stage — identical split math to pipeline.load_stage,
    so the bytes a node fetches are exactly the bytes load_stage will materialize."""
    lo = stage * n_layers // nstages
    hi = (stage + 1) * n_layers // nstages
    return lo, hi


def shards_for_block(manifest: dict, lo: int, hi: int, *, is_head: bool,
                     is_tail: bool, tied: bool, want_tokenizer: bool) -> list[dict]:
    """Resolve the shards a node needs. weights shards are chosen via the signed
    weight_map so only the files holding layers [lo:hi) (plus boundary weights for the
    head/tail, matching load_stage's device_map) are pulled. config shards (config.json,
    the index) go to everyone; tokenizer shards only to the coordinator/head. All non-
    weights files are KB–MB, so the multi-GB selectivity is entirely in the safetensors."""
    wm = manifest["weight_map"]
    need_files: set[str] = set()

    def add(prefixes):
        for w, fn in wm.items():
            if any(w.startswith(p) for p in prefixes):
                need_files.add(fn)

    add(tuple(f"model.layers.{j}." for j in range(lo, hi)))
    if is_head or (is_tail and tied):
        add(("model.embed_tokens",))
    if is_tail:
        add(("model.norm", "lm_head"))

    out = []
    for s in manifest["shards"]:
        kind = s.get("kind", "weights")
        if kind == "weights":
            if s["path"] in need_files:
                out.append(s)
        elif kind == "tokenizer":
            if want_tokenizer:
                out.append(s)
        else:  # config — every node loads config + the index
            out.append(s)
    return out


# ── the verified fetch ────────────────────────────────────────────────────────
def _safe_rel(model_dir: str, rel: str) -> str:
    """Resolve a shard's manifest `path` under model_dir, refusing any absolute path or
    `..` escape. The path is a SIGNED field (a mirror can't inject it — the signature
    would break), so this only bites a malicious/compromised publisher or a node run
    without a pinned pubkey — but a trust primitive that writes publisher-controlled
    names to disk must fail closed on a traversal, not trust it. Returns the safe dest."""
    if os.path.isabs(rel) or os.path.splitdrive(rel)[0]:
        raise FetchError(f"unsafe shard path (absolute): {rel!r}")
    dest = os.path.normpath(os.path.join(model_dir, rel))
    root = os.path.normpath(model_dir)
    if dest != root and not dest.startswith(root + os.sep):
        raise FetchError(f"unsafe shard path (escapes model_dir): {rel!r}")
    return dest


def _verify(path: str, shard: dict) -> None:
    """Re-hash a file and fail closed on any mismatch (size, sha256, or CID)."""
    sha, size = mf.sha256_file(path)
    if size != shard["size"]:
        os.remove(path)
        raise FetchError(f"{shard['path']}: size {size} != manifest {shard['size']}")
    if sha != shard["sha256"]:
        os.remove(path)
        raise FetchError(f"{shard['path']}: sha256 mismatch (corrupt or tampered)")
    if mf.cidv1_raw(sha) != shard["shard_id"]:
        os.remove(path)
        raise FetchError(f"{shard['path']}: CID mismatch")


def _cached(path: str, shard: dict) -> bool:
    """A file already present and matching its hash needs no re-fetch (fail-closed:
    a size match alone is not enough — we re-hash)."""
    if not os.path.exists(path) or os.path.getsize(path) != shard["size"]:
        return False
    try:
        _verify(path, shard)
        return True
    except FetchError:
        return False


def fetch_block_range(manifest: dict, model_dir: str, lo: int, hi: int, *,
                      is_head: bool, is_tail: bool, role: str, provider: Provider,
                      expected_pubkey: str | None = None, tied: bool = False) -> list[str]:
    """Fetch + verify exactly the shards covering layers [lo:hi) (plus the head/tail boundary weights
    and, for a coordinator/head, the tokenizer) into model_dir. Verifies the manifest signature (and
    the catalog-pinned publisher, if given) first; re-hashes every byte of every shard on arrival.
    Returns the local file paths; raises on any failure — nothing half-verified is left for the loader.

    This is the EXPLICIT-RANGE entry point: the scattered deploy splits layers UNEVENLY across stages
    (the self-optimizer's `select_ring` picks variable per-stage blocks, e.g. 10/13/13/13/13, not an
    even n/nstages split), so the puller passes the stage's actual [lo:hi] rather than a stage index."""
    mf.verify_manifest(manifest, expected_pubkey)
    os.makedirs(model_dir, exist_ok=True)
    want_tok = role == "coordinator" or is_head
    shards = shards_for_block(manifest, lo, hi, is_head=is_head, is_tail=is_tail,
                              tied=tied, want_tokenizer=want_tok)
    weights = [s for s in shards if s.get("kind", "weights") == "weights"]
    total = sum(s["size"] for s in shards)
    _log(f"layers [{lo}:{hi}] role={role} head={is_head} tail={is_tail}: "
         f"{len(shards)} shards ({len(weights)} weights), {total / 1e9:.2f} GB")

    paths = []
    for s in shards:
        dest = _safe_rel(model_dir, s["path"])     # fail closed on an absolute / `..` path
        os.makedirs(os.path.dirname(dest) or model_dir, exist_ok=True)
        if _cached(dest, s):
            _log(f"  have {s['path']}")
            paths.append(dest)
            continue
        _log(f"  fetch {s['path']} ({s['size'] / 1e9:.2f} GB)")
        provider.fetch(s, dest)
        _verify(dest, s)
        paths.append(dest)
    _log(f"block verified: {len(paths)} files in {model_dir}")
    return paths


def fetch_block(manifest: dict, model_dir: str, *, stage: int, nstages: int,
                role: str, provider: Provider, expected_pubkey: str | None = None,
                tied: bool = False) -> list[str]:
    """Fetch + verify this node's block by stage index (EVEN n/nstages split) into model_dir, ready for
    pipeline.load_stage(model_dir, stage, nstages). Thin wrapper over fetch_block_range; the scattered
    deploy uses fetch_block_range directly with select_ring's uneven per-stage [lo:hi]. role:
    "coordinator" | "stage" — the coordinator (and head stage 0) also pull the tokenizer."""
    lo, hi = block_for_stage(manifest["layer_count"], stage, nstages)
    return fetch_block_range(manifest, model_dir, lo, hi, is_head=stage == 0,
                             is_tail=stage == nstages - 1, role=role, provider=provider,
                             expected_pubkey=expected_pubkey, tied=tied)


if __name__ == "__main__":  # tiny smoke check of the pure logic, no network
    print("shard.fetch loaded; providers:",
          [c.__name__ for c in (MirrorProvider, LocalDirProvider, Libp2pProvider)],
          file=sys.stderr)
