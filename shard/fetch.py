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
                 correctness + the trust property are unaffected either way. For a
                 checkpoint packed by expert group (MiMo-V2.6: every layer spans every
                 file) the tensor-granular ranged path below (fetch_layer_tensors) pulls
                 exactly the stage's tensors instead.
  * verified   — fetch_block re-hashes every file itself, so a provider is never trusted.
                 A wrong byte → sha256 mismatch → the file is deleted and the load fails
                 closed. Corrupted weights cannot reach VRAM.

The **provider is a seam** (docs/INTEGRATION.md §4). A mirror (HTTP/HF) is the first
provider; libp2p content routing takes over at step 8 — additive, because the fetch was
content-verified from day one, so swapping the source changes nothing about trust.

Per the boundary law: pure engine. Knows about manifests, shards, and bytes — nothing
about c0mpute's catalog or accounts (the caller passes the pinned publisher pubkey in).
"""
import concurrent.futures as cf
import contextlib
import glob
import hashlib
import http.client
import json
import math
import os
import posixpath
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from urllib.parse import urlsplit
from urllib.request import Request

from . import manifest as mf
# the stage-key classification, re-exported here: the loaders resolve through the SAME
# functions, so the files a node pulls are the tensors it materializes
from .weightkeys import LAYER_PATTERN, boundary_of, layer_of, namespace


class FetchError(Exception):
    """A shard could not be fetched or failed verification. Fail closed."""


class ProviderUnavailable(Exception):
    """This provider can't serve right now — caller may fall back to another."""


def _log(msg: str) -> None:
    print(f"[fetch] {msg}", flush=True)


def _resume_offset(resp) -> int:
    """The byte offset this response body starts at. A 206 declares it in Content-Range
    ('bytes 19922944-4998528135/4998528136'); a 200 — or a 206 whose Range a redirect/CDN
    dropped, so the body is the WHOLE file — starts at 0. Trusting the bare 206 status and
    appending a full body is the live-ring bug that grew a shard to have+total bytes."""
    if getattr(resp, "status", 200) != 206:
        return 0
    cr = (getattr(resp, "headers", None) or {}).get("Content-Range", "") or ""
    m = re.match(r"\s*bytes\s+(\d+)-", cr)
    return int(m.group(1)) if m else 0  # a 206 with no parseable range: treat as a full body


def _copy_capped(src, dst, limit: int, bufsize: int = 1 << 20) -> None:
    """Copy at most `limit` bytes src->dst. A guard so a broken or hostile mirror that streams
    past the manifest size cannot flood the disk — an overshoot is bounded and then caught by the
    size check. A short read (dropped connection) just stops early; the retry loop resumes it."""
    remaining = limit
    while remaining > 0:
        chunk = src.read(min(bufsize, remaining))
        if not chunk:
            break
        dst.write(chunk)
        remaining -= len(chunk)


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """urllib's default redirect handler carries EVERY header — including Authorization —
    to the redirect target, so a mirror 302ing to another origin exfiltrates the bearer
    token (e.g. an HF_TOKEN). Rules: a redirect target must be https; crossing an origin
    (scheme or authority change) strips Authorization; if SHARD_REDIRECT_ALLOW is set
    (comma-separated hostnames), a cross-origin target must be on it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is None:
            return None
        old_u, new_u = urlsplit(req.full_url), urlsplit(new.full_url)
        if new_u.scheme != "https":
            raise FetchError(f"insecure redirect to {new.full_url!r} refused")
        if (old_u.scheme.lower(), old_u.netloc.lower()) != (new_u.scheme.lower(), new_u.netloc.lower()):
            allow = [h.strip() for h in os.environ.get("SHARD_REDIRECT_ALLOW", "").split(",") if h.strip()]
            if allow and new_u.hostname not in allow:
                raise FetchError(
                    f"cross-origin redirect to {new_u.netloc!r} not in SHARD_REDIRECT_ALLOW")
            new.remove_header("Authorization")  # never leak the bearer to another origin
        return new


_OPENER = urllib.request.build_opener(_SafeRedirectHandler())


def urlopen(req, timeout=None):
    """All engine HTTP goes through the redirect-hardened opener above (never the module
    default, whose redirect handler forwards Authorization cross-origin)."""
    return _OPENER.open(req, timeout=timeout)


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

    _RESOLVE_RE = re.compile(r"^(?P<origin>https?://[^/]+)/(?P<repo>.+?)/resolve/(?P<rev>[^/]+)/$")

    def __init__(self, base_url: str, headers: dict | None = None, retries: int = 6):
        self.base = base_url.rstrip("/") + "/"
        self.headers = headers or {}
        self.retries = retries

    def _url(self, shard: dict) -> str:
        """Route by the shard's SIGNED source repo/revision when the mirror is an
        HF-style `<origin>/<repo>/resolve/<rev>/` base: a manifest may declare its
        tokenizer shards from a different repo, and each shard pins the immutable
        revision it was hashed at — resolving them against the base repo's branch
        fetches the wrong (or a moved) file."""
        m = self._RESOLVE_RE.match(self.base)
        if m and shard.get("repo") and shard.get("revision"):
            return f"{m.group('origin')}/{shard['repo']}/resolve/{shard['revision']}/{shard['path']}"
        return self.base + shard["path"]

    def fetch(self, shard: dict, dest: str) -> None:
        url = self._url(shard)
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
        name = url.rsplit("/", 1)[-1]
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > total:  # stale/corrupt partial — start over
            os.remove(part)
            have = 0
        if have == total:  # a prior attempt already fetched the whole body
            return
        req = Request(url, headers={"User-Agent": "shard/1", **self.headers})
        if have:
            req.add_header("Range", f"bytes={have}-")
        with urlopen(req, timeout=120) as r:
            # Place the body by its Content-Range, NOT the bare 206 status. A mirror/CDN can answer
            # a Range request with the WHOLE file (a 200, or a 206 starting at 0 when a redirect
            # drops the Range header). Appending that to our partial overshoots to have+total bytes —
            # the live-ring bug. Only append when the server actually resumed at our offset.
            start = _resume_offset(r)
            if start == have and have:      # a genuine resume: append the tail
                mode, cap = "ab", total - have
            elif start == 0:                # the whole body (200, or a 206 whose Range was dropped)
                mode, cap = "wb", total
            else:                           # a range we can't place — drop the partial, restart clean
                if os.path.exists(part):    # (may be absent: a hostile 206 on a fresh, no-Range GET)
                    os.remove(part)
                raise FetchError(f"{name}: server resumed at {start}, expected {have} — restarting")
            with open(part, mode) as f:
                _copy_capped(r, f, cap)     # never write past `total`, whatever the mirror streams
        got = os.path.getsize(part)
        if got != total:  # dropped connection => a partial body with NO exception; keep the partial
            raise FetchError(  # and RAISE so fetch()'s retry loop resumes it via Range (the size-only
                f"incomplete download {got}/{total} bytes for {name} — resuming")  # bug that let a
                # truncated shard reach _verify, which then hard-failed the whole block pull


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
    """Fetch a shard by its CID from PEERS over libp2p content routing (the torrent path).

    Spawns the Go sidecar one-shot: `-fetch-cid` finds providers for the shard's CIDv1
    on the shard DHT (kad, /shard prefix) and block-exchanges the bytes from the first
    peer that serves them, resuming a partial across providers by offset. The peer is
    UNTRUSTED by design — fetch_block re-hashes every byte against the signed manifest,
    so a hostile seeder can waste time, never poison weights (the same property the
    mirror path has). Anything short of a complete transfer raises ProviderUnavailable
    so a ChainProvider can hand the shard to the mirror/origin."""

    # A peer must sustain at least this much throughput or it is abandoned for the mirror.
    #
    # Peers-before-mirror is only a win when the peer is COMPETITIVE. Peer quality spans two
    # orders of magnitude — a datacentre seeder saturates a link, a home seeder on a 20 Mbit
    # uplink gives ~2.5 MB/s — and a pure preference order has no way to tell them apart. Live on
    # 2026-07-25 a 5090 with a measured 48.9 MB/s path to the mirror chose a residential peer for a
    # 5 GB shard instead: ~2.5 MB/s, ~33 minutes, against a 30-minute deadline it would have blown
    # only to restart from the mirror anyway. The retry is progress-gated, so the tarpit looked
    # healthy the whole time. At launch every node is residential AND every node seeds, so without
    # a floor the network gets SLOWER as it grows — the inverse of the torrent property.
    #
    # The floor is deliberately far below a healthy peer and far above a residential uplink, so it
    # abandons tarpits without punishing a genuinely useful seeder.
    MIN_PEER_MBPS = float(os.environ.get("SHARD_MIN_PEER_MBPS", "8"))
    _RAMP_S = 25.0        # grace before judging: DHT lookup + dial + first blocks
    _POLL_S = 5.0

    def __init__(self, bootstrap: list[str] | None = None, sidecar_bin: str | None = None,
                 key: str | None = None, timeout: int = 1800):
        self.bin = sidecar_bin or os.environ.get("SHARD_SIDECAR", "/tmp/sidecar")
        env_bs = [b for b in os.environ.get("SHARD_DHT_BOOTSTRAP", "").split(",") if b]
        self.bootstrap = list(bootstrap) if bootstrap is not None else env_bs
        self.key = key
        self.timeout = timeout

    @staticmethod
    def _progress(dest: str) -> int:
        """Bytes delivered so far: the sidecar streams into '<dest>.p2p.<pid>.<peer>.part'."""
        n = 0
        for p in glob.glob(f"{dest}.p2p.*.part") + ([dest] if os.path.exists(dest) else []):
            try:
                n += os.path.getsize(p)
            except OSError:
                pass
        return n

    def fetch(self, shard: dict, dest: str) -> None:
        cmd = [self.bin, "-fetch-cid", shard["shard_id"], "-fetch-out", dest,
               "-fetch-size", str(shard["size"]), "-fetch-timeout", str(self.timeout)]
        for b in self.bootstrap:
            cmd += ["-dht-bootstrap", b]
        if self.key:
            cmd += ["-key", self.key]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except FileNotFoundError:
            raise ProviderUnavailable(f"sidecar binary {self.bin!r} not found")

        started = time.monotonic()
        base = self._progress(dest)
        deadline = started + self.timeout + 60
        slow = None
        while proc.poll() is None:
            time.sleep(self._POLL_S)
            now = time.monotonic()
            elapsed = now - started
            if now > deadline:
                slow = f"timed out ({self.timeout}s)"
                break
            if elapsed >= self._RAMP_S and self.MIN_PEER_MBPS > 0:
                rate = (self._progress(dest) - base) / elapsed / 1e6
                if rate < self.MIN_PEER_MBPS:
                    slow = (f"peer too slow: {rate:.1f} MB/s over {elapsed:.0f}s "
                            f"(floor {self.MIN_PEER_MBPS:.0f} MB/s) — falling back to the mirror")
                    break
        if slow:
            proc.kill()
            proc.wait(timeout=30)
            # drop the abandoned partial so it cannot be mistaken for progress or leak disk
            for p in glob.glob(f"{dest}.p2p.*.part"):
                try:
                    os.remove(p)
                except OSError:
                    pass
            raise ProviderUnavailable(f"libp2p fetch of {shard['path']}: {slow}")
        out, err = proc.communicate()
        if proc.returncode != 0:
            tail = ((err or out or "").strip().splitlines() or ["no output"])[-1]
            raise ProviderUnavailable(f"libp2p fetch failed: {tail[:160]}")


class ChainProvider(Provider):
    """Try providers in order — peers first, mirror/origin last. A provider that raises
    ProviderUnavailable/FetchError hands the shard down the chain; so does one whose
    delivered bytes fail the manifest hash (a hostile seeder must not be able to wedge
    the pull when an honest source remains — its garbage is deleted and the next source
    tried). fetch_block's own re-hash stays as the fail-closed backstop either way.
    Only when every provider failed does the fetch fail closed."""

    verifies = True   # fetch_block_range skips its backstop re-hash: we already re-hash per source

    def __init__(self, providers: list[Provider]):
        if not providers:
            raise ValueError("ChainProvider needs at least one provider")
        self.providers = list(providers)

    def fetch(self, shard: dict, dest: str) -> None:
        errs = []
        for p in self.providers:
            try:
                p.fetch(shard, dest)
                _verify(dest, shard)        # deletes dest on mismatch — a bad source is not fatal
                return
            except (ProviderUnavailable, FetchError) as e:
                errs.append(f"{type(p).__name__}: {str(e)[:120]}")
                _log(f"  {type(p).__name__} could not serve {shard['path']} -> next provider")
        raise FetchError(f"all providers failed for {shard['path']}:\n    " + "\n    ".join(errs))


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
    weights files are KB–MB, so the multi-GB selectivity is entirely in the safetensors.

    The layer->files mapping comes from the signed map's OWN tensor names (shard.weightkeys),
    so a checkpoint that namespaces its tensors differently still resolves; a range that
    selects nothing raises rather than pulling an empty block. The loaders it feeds resolve
    through the SAME classification (pipeline.load_stage's device_map,
    mlx_runtime.select_weight_keys), so the files a node pulls are the tensors it loads."""
    wm = manifest["weight_map"]
    need_files: set[str] = set()
    layer_files: set[str] = set()
    for w, fn in wm.items():
        j = layer_of(w)
        if j is not None:
            if lo <= j < hi:                 # a second numbered tower (e.g. a vision encoder) can
                layer_files.add(fn)          # add files to a range: bytes, never a missing weight
            continue
        b = boundary_of(w)
        if b == "embed" and (is_head or (is_tail and tied)):
            need_files.add(fn)               # the head's embedding (a tied tail reads logits from it)
        elif is_tail and b in ("norm", "lm_head"):
            need_files.add(fn)               # the tail's final norm + output head
        # anything else belongs to no stage block and is not pulled, as before
    # A range resolving to NO file is the silent-empty-pull: fail closed and say what the map holds.
    # (An empty weight_map — a config/tokenizer-only manifest — and an empty range select nothing
    # by construction, not by a namespace miss.)
    if wm and lo < hi and not layer_files:
        seen = sorted({j for w in wm if (j := layer_of(w)) is not None})
        raise FetchError(
            f"weight_map selects no weight file for layers [{lo}:{hi}]: " + (
                f"the map only carries layers {seen[0]}-{seen[-1]}" if seen else
                f"no tensor name matches the layer pattern {LAYER_PATTERN!r} "
                f"(e.g. {sorted(wm)[:3]})") + " — refusing an empty weight pull")
    need_files |= layer_files

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
    root = os.path.realpath(model_dir)
    dest = os.path.normpath(os.path.join(root, rel))
    if dest != root and not dest.startswith(root + os.sep):
        raise FetchError(f"unsafe shard path (escapes model_dir): {rel!r}")
    # Lexical containment is NOT enough: a pre-existing symlink — a directory component
    # or the destination file itself — redirects the write outside the model root even
    # though the string checks pass. Resolve what exists and require it stays inside.
    if os.path.islink(dest):
        raise FetchError(f"unsafe shard path (destination is a symlink): {rel!r}")
    real = os.path.realpath(dest)
    if real != root and not real.startswith(root + os.sep):
        raise FetchError(f"unsafe shard path (symlink escapes model_dir): {rel!r}")
    return dest


def _verify(path: str, shard: dict) -> None:
    """Re-hash a file and fail closed on any mismatch (size, sha256, or CID).
    A missing/unreadable file is a FETCH failure (FetchError), not a raw OSError:
    ChainProvider catches only provider exceptions, so a provider that claims success
    without leaving readable bytes must hand the shard to the next source, not
    crash the whole pull."""
    try:
        sha, size = mf.sha256_file(path)
    except OSError as e:
        try:
            os.remove(path)
        except OSError:
            pass
        raise FetchError(f"{shard['path']}: unreadable after fetch ({e})") from e
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


def check_manifest_cid(path: str, ref: str) -> None:
    """Assert the manifest FILE BYTES hash to the content id the assignment carried
    (`mf1:<name>@<cid>` — the CID part is normative, the name advisory; a bare CIDv1 is
    accepted too). Runs BEFORE the JSON is parsed, so substitution among validly-signed
    manifests, rollback to an older release, and a tampered local cache all die without
    attacker bytes being trusted for anything. Both trust layers stay required: this CID
    pins WHICH manifest, the pinned signature (verify_manifest) pins WHO published it."""
    cid = ref.rsplit("@", 1)[-1] if ref.startswith("mf1:") else ref
    if not cid.startswith("bafk"):
        raise mf.ManifestError(f"manifest ref carries no content id: {ref!r}")
    sha, _ = mf.sha256_file(path)
    got = mf.cidv1_raw(sha)
    if got != cid:
        raise mf.ManifestError(f"manifest content id mismatch: ref pins {cid}, file hashes to {got}")


def check_expected(manifest: dict, model_id: str | None = None,
                   layer_count: int | None = None) -> None:
    """Cross-check the resolved manifest against what the ASSIGNMENT said this node pulls
    (the ref/manifest-mismatch defense): a validly-signed, correctly-addressed manifest for
    the WRONG model must still refuse to drive a pull. Fail closed."""
    if model_id is not None and manifest.get("model_id") != model_id:
        raise mf.ManifestError(
            f"manifest model_id {manifest.get('model_id')!r} != assignment's {model_id!r}")
    if layer_count is not None and manifest.get("layer_count") != layer_count:
        raise mf.ManifestError(
            f"manifest layer_count {manifest.get('layer_count')!r} != assignment's {layer_count}")


def _check_map_coverage(manifest: dict) -> None:
    """Every file the signed weight_map references must have a shard entry, or the map
    points at bytes the manifest can never verify."""
    have = {s["path"] for s in manifest["shards"]}
    missing = sorted(set(manifest["weight_map"].values()) - have)
    if missing:
        raise FetchError(f"manifest weight_map references files with no shard entry: {missing[:5]}")


def _check_index_matches_map(manifest: dict, paths: list[str]) -> None:
    """The runtime loader trusts model.safetensors.index.json on disk while the fetch
    selects via the SIGNED weight_map. If the two maps differ, a weight file only the
    index knows about would load without ever being verified — refuse the pair."""
    for p in paths:
        if not p.endswith(".index.json"):
            continue
        try:
            with open(p) as f:
                idx = json.load(f)
        except (OSError, ValueError) as e:
            os.remove(p)
            raise FetchError(f"unparseable weight index {os.path.basename(p)}: {e}")
        if idx.get("weight_map") != manifest["weight_map"]:
            os.remove(p)
            raise FetchError("downloaded index weight_map differs from the signed manifest weight_map")


def _quarantine_stray_weights(manifest: dict, model_dir: str) -> None:
    """Rename any *.safetensors under model_dir that the signed manifest does not list:
    the runtime's index-driven loader would otherwise pick up a stale, never-verified
    file left by an earlier manifest or a hostile writer."""
    known = {s["path"] for s in manifest["shards"]}
    for r, _, files in os.walk(model_dir):
        for name in files:
            if not name.endswith(".safetensors"):
                continue
            full = os.path.join(r, name)
            rel = os.path.relpath(full, model_dir).replace(os.sep, "/")
            if rel in known:
                continue
            os.replace(full, full + ".quarantined")
            _log(f"  QUARANTINED stray weight file {rel} (not in the signed manifest)")


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
    _check_map_coverage(manifest)
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
        if not getattr(provider, "verifies", False):   # a self-verifying provider (ChainProvider)
            _verify(dest, s)                            # already re-hashed; don't hash 100+ GB twice
        paths.append(dest)
    _check_index_matches_map(manifest, paths)
    _quarantine_stray_weights(manifest, model_dir)
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


def build_chain_provider(*, mirror: str | None, bootstrap: list[str] | None,
                         sidecar_bin: str | None, key: str | None,
                         local_dir: str | None = None) -> Provider:
    """The daemon's default source chain: PEERS first (libp2p content routing), then the
    HTTP mirror — the torrent property with a guaranteed fallback. Every source is re-hashed
    against the signed manifest per shard, so ordering is a speed/cost choice, never a trust
    one. A lone provider is returned bare; an empty chain is a caller error."""
    chain: list[Provider] = []
    if local_dir:
        chain.append(LocalDirProvider(local_dir))
    if bootstrap:
        chain.append(Libp2pProvider(bootstrap=bootstrap, sidecar_bin=sidecar_bin, key=key))
    if mirror:
        chain.append(MirrorProvider(mirror))
    if not chain:
        raise ValueError("no providers: give --mirror and/or --bootstrap")
    return chain[0] if len(chain) == 1 else ChainProvider(chain)


# ── tensor-granular ranged fetch (shard v2, R1) ─────────────────────────────────────────────────
#
# File-granular selection only saves bytes when a checkpoint is packed layer by layer. MiMo-V2.6 is
# packed by EXPERT GROUP (model_pp0_ep{i}: experts [4i, 4i+4) of every layer), so every layer spans
# all 64 files and a file-granular pull of ANY layer range is the whole ~161 GiB model. A device
# holds exactly the layers its memory fits, so it must fetch exactly their bytes: read each file's
# safetensors header (8-byte LE length + JSON), and pull only the byte ranges of the wanted tensors
# with HTTP Range requests against one pinned revision.
#
# Integrity root (until the signed per-tensor manifest v2, R1b): TLS to the pinned-revision origin.
# What is enforced on every byte regardless:
#   * a source is trusted by default only if it is https AND its URL pins a 40-hex commit
#     (`.../resolve/<sha>/`; RangeSource.huggingface refuses a branch/tag), so two ranges of one
#     file can never come from two revisions; responses must not be content-encoded;
#   * every response must be a 206 whose Content-Range is EXACTLY the requested span and whose
#     total is the file size seen first (a server that ignores Range, shifts it, or swaps the file
#     mid-fetch is refused, never appended); bodies are capped at the requested length;
#   * each header is validated: dtype x shape == byte span, spans in bounds and non-overlapping,
#     header + data == file size, no duplicate names; each wanted tensor must be in the header of
#     the file the index maps it to;
#   * sha256 is recorded per tensor, per fetched range (journal) and per output file; a resume
#     re-hashes journaled ranges on disk before trusting them.
# What TLS alone cannot catch: an origin serving wrong bytes of the right length. That is what
# `expected` ({tensor: {sha256, dtype, shape}}, R1b's signed manifest) closes: bytes AND the
# header's dtype/shape claims are checked. An untrusted source is refused unless every wanted
# tensor has an expected entry.
#
# The layer -> tensor mapping is read off the checkpoint's own names (shard.weightkeys.namespace:
# the decoder stack the embedding/final norm hang off), so M2.5 (`model.layers.N`), K3
# (`language_model.model.layers.N`), MiMo (`model.layers.N`, with `model.mtp.layers.N` and an audio
# tower excluded) and DeepSeek-V4 (`layers.N`, with `layout=` naming its `embed`/`head`) all resolve
# without offsets or per-model code. The peers-first libp2p path stays file-granular (CID per file)
# until manifest v2 gives per-tensor content ids.

RANGED_SCHEMA = "shard-ranged/1"
RANGED_RECORD = "ranged.json"
LOCAL_INDEX = "model.safetensors.index.json"
MAX_HEADER_BYTES = 100_000_000      # the safetensors library's own header cap
MAX_AUX_BYTES = 64 << 20            # config / index / tokenizer: small whole files
DEFAULT_MAX_JOB = 64 << 20          # one Range request; 8 workers -> <= 512 MiB in flight

# bits per element of every safetensors dtype (sub-byte ones pack along the last dim)
DTYPE_BITS = {"BOOL": 8, "U8": 8, "I8": 8, "F8_E4M3": 8, "F8_E5M2": 8, "F8_E4M3FNUZ": 8,
              "F8_E5M2FNUZ": 8, "F8_E8M0": 8, "I16": 16, "U16": 16, "F16": 16, "BF16": 16,
              "I32": 32, "U32": 32, "F32": 32, "I64": 64, "U64": 64, "F64": 64, "C64": 64,
              "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6}

_REMOTE_PATH_RE = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*")
_CONTENT_RANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
_DIGITS_RE = re.compile(r"[0-9]+")


def _check_remote_path(path: str) -> str:
    """A file name from the (publisher-controlled) index becomes part of a URL and of a local cache
    path: plain relative names only, so `../main/x` cannot step off the pinned revision."""
    if not isinstance(path, str) or not _REMOTE_PATH_RE.fullmatch(path) or \
            any(p in (".", "..") for p in path.split("/")):
        raise FetchError(f"unsafe remote file name {path!r}")
    return path


_PINNED_BASE_RE = re.compile(r"/resolve/[0-9a-f]{40}/$")


class RangeSource:
    """Byte ranges of the files under ONE base URL, through the hardened opener. `trusted` = the
    transport vouches for the bytes: by default only an https base whose path pins a 40-hex commit
    (HF `.../resolve/<sha>/`), since TLS to a moving branch would let a resume mix revisions. Pass
    trusted=True only when the content is pinned some other way. Headers (auth) are never logged
    or repr'd."""

    def __init__(self, base_url: str, headers: dict | None = None, *, trusted: bool | None = None,
                 retries: int = 6, timeout: float = 120):
        self.base = base_url.rstrip("/") + "/"
        self.headers = dict(headers or {})
        u = urlsplit(self.base)
        self.trusted = (u.scheme == "https" and bool(_PINNED_BASE_RE.search(u.path))) \
            if trusted is None else bool(trusted)
        self.retries, self.timeout = retries, timeout

    @classmethod
    def huggingface(cls, repo: str, revision: str, headers: dict | None = None, **kw) -> "RangeSource":
        """An HF repo at one COMMIT. A branch or tag can move between two ranges of the same file."""
        if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
            raise FetchError(f"revision {revision!r} is not a full commit sha; a ranged fetch pins "
                             "one immutable revision")
        return cls(f"https://huggingface.co/{repo}/resolve/{revision}/", headers, **kw)

    def __repr__(self) -> str:
        return f"RangeSource({self.base!r}, trusted={self.trusted})"

    def read(self, path: str, start: int, end: int, sink, total: int | None = None) -> int:
        """Stream bytes [start, end) of `path` into sink(offset, chunk), in order. A dropped
        connection resumes at the byte it stopped (not at `start`); `retries` attempts in a row
        without progress give up. Returns the file size from Content-Range. Anything but an exact
        206 for the asked span (or a size different from `total`) is a FetchError, not a retry."""
        url = self.base + _check_remote_path(path)
        pos, stalls, attempts = start, 0, 0
        while pos < end:
            before, err = pos, None
            attempts += 1
            req = Request(url, headers={"User-Agent": "shard/1", **self.headers,
                                        "Accept-Encoding": "identity",
                                        "Range": f"bytes={pos}-{end - 1}"})
            try:
                resp = urlopen(req, timeout=self.timeout)
            except urllib.error.HTTPError as e:
                if e.code in (400, 401, 403, 404, 410, 416):
                    raise FetchError(f"{path}: HTTP {e.code} for bytes {pos}-{end - 1}") from e
                err = e
            except (OSError, http.client.HTTPException) as e:
                err = e
            else:
                with resp:
                    total = _check_range_response(resp, path, pos, end, total)
                    while pos < end:
                        try:
                            chunk = resp.read(min(1 << 20, end - pos))
                        except (OSError, http.client.HTTPException) as e:
                            err = e
                            break
                        if not chunk:
                            break
                        sink(pos, chunk)
                        pos += len(chunk)
            if pos >= end:
                break
            stalls = 0 if pos > before else stalls + 1
            if stalls >= self.retries:
                raise FetchError(f"{path}: gave up on bytes {pos}-{end - 1} after {self.retries} "
                                 f"attempts without progress ({err or 'short body'})")
            if attempts >= 8 * self.retries:        # a server trickling a few bytes per response
                raise FetchError(f"{path}: {attempts} requests for one range, still at byte {pos}")
            _log(f"  resume {path} at byte {pos} ({str(err or 'short body')[:80]})")
            if stalls:
                time.sleep(min(30, 2 ** stalls))
        return total


def _check_range_response(resp, path: str, pos: int, end: int, total: int | None) -> int:
    status = getattr(resp, "status", None)
    if status != 206:
        raise FetchError(f"{path}: HTTP {status} to a Range request (a whole-file body is not a "
                         "tensor range) - refusing")
    cr = (resp.headers.get("Content-Range") or "").strip()
    m = _CONTENT_RANGE_RE.fullmatch(cr)
    if not m:
        raise FetchError(f"{path}: 206 without a usable Content-Range ({cr!r})")
    a, b, tot = map(int, m.groups())
    if (a, b) != (pos, end - 1):
        raise FetchError(f"{path}: asked for bytes {pos}-{end - 1}, the server sent {a}-{b}")
    if total is not None and tot != total:
        raise FetchError(f"{path}: file is {tot} B now, {total} B before (a different file or "
                         "revision) - refusing to mix")
    ce = (resp.headers.get("Content-Encoding") or "identity").strip().lower()
    if ce != "identity":
        raise FetchError(f"{path}: Content-Encoding {ce!r} on a byte range - refusing")
    cl = resp.headers.get("Content-Length")
    if cl is not None and cl.strip() != str(end - pos):
        raise FetchError(f"{path}: Content-Length {cl!r} for a {end - pos} B range")
    return tot


def tensor_nbytes(dtype: str, shape) -> int:
    """Bytes a safetensors tensor of this dtype x shape occupies."""
    bits = DTYPE_BITS.get(dtype) if isinstance(dtype, str) else None
    if bits is None:
        raise FetchError(f"unknown safetensors dtype {dtype!r}")
    if not isinstance(shape, list) or not all(
            isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape):
        raise FetchError(f"bad tensor shape {shape!r}")
    n = math.prod(shape) * bits
    if n % 8:
        raise FetchError(f"{dtype}x{shape} is not a whole number of bytes")
    return n // 8


def _no_dupes(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise FetchError(f"duplicate key {k!r} in a safetensors header")
        d[k] = v
    return d


def parse_safetensors_header(raw: bytes, file_size: int, where: str = "") -> dict:
    """Validate a safetensors header (the file's first 8+n bytes) against the file size and return
    {tensor: {"dtype", "shape", "start" (absolute file offset), "nbytes"}}. Refuses anything a
    correct writer never produces: a span that is not dtype x shape, overlapping or out-of-bounds
    spans, header + data != file size, duplicate names, an oversized or non-JSON header."""
    if len(raw) < 8:
        raise FetchError(f"{where}header shorter than its 8-byte length prefix")
    n = struct.unpack("<Q", raw[:8])[0]
    if n > MAX_HEADER_BYTES or len(raw) != 8 + n:
        raise FetchError(f"{where}header length {n} is implausible or truncated")
    try:
        hdr = json.loads(raw[8:].decode("utf-8"), object_pairs_hook=_no_dupes)
    except (ValueError, UnicodeDecodeError, RecursionError) as e:
        raise FetchError(f"{where}header is not JSON: {e}") from e
    if not isinstance(hdr, dict):
        raise FetchError(f"{where}header is not a JSON object")
    base, out, spans = 8 + n, {}, []
    for name, t in hdr.items():
        if name == "__metadata__":
            continue
        try:
            dtype, shape, (b, e) = t["dtype"], t["shape"], t["data_offsets"]
        except (TypeError, KeyError, ValueError) as ex:
            raise FetchError(f"{where}{name}: malformed header entry") from ex
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in (b, e)) or not 0 <= b <= e:
            raise FetchError(f"{where}{name}: bad data_offsets {[b, e]}")
        want = tensor_nbytes(dtype, shape)
        if e - b != want:
            raise FetchError(f"{where}{name}: header span {e - b} B != {dtype}x{shape} = {want} B")
        spans.append((b, e, name))
        out[name] = {"dtype": dtype, "shape": list(shape), "start": base + b, "nbytes": want}
    end = 0
    for b, e, name in sorted(spans):
        if b < end:
            raise FetchError(f"{where}{name}: overlaps the tensor before it")
        end = e
    if base + end != file_size:
        raise FetchError(f"{where}header + data = {base + end} B, but the file is {file_size} B")
    return out


def _read_header(src: RangeSource, path: str) -> tuple[bytes, int]:
    buf = bytearray()
    total = src.read(path, 0, 8, lambda _o, c: buf.extend(c))
    n = struct.unpack("<Q", bytes(buf))[0]
    if n > MAX_HEADER_BYTES or 8 + n > total:
        raise FetchError(f"{path}: header length {n} is implausible for a {total} B file")
    src.read(path, 8, 8 + n, lambda _o, c: buf.extend(c), total=total)
    return bytes(buf), total


def select_tensors(names, lo: int, hi: int, *, head: bool = False, tail: bool = False,
                   tied: bool = False, layout: dict | None = None, extra=()) -> dict:
    """Exactly the tensors a stage holding decoder layers [lo:hi) needs, grouped for output:
    {"layer-00012": [...], ..., "embed": [...], "norm": [...], "lm_head": [...], "extra-<p>": [...]}.

    The decoder stack and the boundary modules come from the checkpoint's own names
    (weightkeys.namespace); `layout` overrides any of its keys ("layers", "embed", "norm",
    "lm_head") for a checkpoint that names them unconventionally (DeepSeek-V4: embed="embed",
    lm_head="head"). Unlike the file-granular union, a second numbered tower (MiMo's
    `model.mtp.layers.N`, an audio encoder's `layers.N`) is NOT pulled into a decoder range: those
    bytes are real on a tensor-granular fetch. `extra` adds whole modules by name prefix (the
    MTP drafter "model.mtp", V4's "hc_head_fn"...). Fails loud on a layer with no tensor, a
    boundary with no tensor, or a tensor selected twice."""
    names = list(names)
    try:
        ns = namespace(names)
    except ValueError as e:
        raise FetchError(str(e)) from e
    ns.update(layout or {})
    stack = ns["layers"] + "."
    by_layer: dict[int, list] = {}
    for n in names:
        if n.startswith(stack):
            j, sep, _ = n[len(stack):].partition(".")
            if sep and _DIGITS_RE.fullmatch(j) and lo <= int(j) < hi:
                by_layer.setdefault(int(j), []).append(n)
    missing = [j for j in range(lo, hi) if j not in by_layer]
    if missing:
        raise FetchError(f"no tensor under {stack!r} for layer(s) {missing[:8]} of [{lo}:{hi}] - "
                         f"the decoder stack resolved to {ns['layers']!r} (pass layout= if wrong)")
    groups = {f"layer-{j:05d}": sorted(v) for j, v in sorted(by_layer.items())}

    def module(prefix, what):
        sel = sorted(n for n in names if n == prefix or n.startswith(prefix + "."))
        if not sel:
            raise FetchError(f"{what}: no tensor under {prefix!r} - this checkpoint names it "
                             "differently; pass layout={...}")
        return sel
    if head or (tail and tied):
        groups["embed"] = module(ns["embed"], "embedding")
    if tail:
        groups["norm"] = module(ns["norm"], "final norm")
        if not tied:
            groups["lm_head"] = module(ns["lm_head"], "lm_head")
    for p in extra:
        groups["extra-" + re.sub(r"[^A-Za-z0-9_.-]", "_", p)] = module(p, f"extra {p!r}")
    if not groups:
        raise FetchError(f"nothing selected for [{lo}:{hi}] head={head} tail={tail}")
    owner: dict[str, str] = {}
    for g, v in groups.items():
        for n in v:
            if n in owner:
                raise FetchError(f"{n} selected twice ({owner[n]} and {g})")
            owner[n] = g
    return groups


def _atomic_write(path: str, data: bytes) -> None:
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


@contextlib.contextmanager
def _dir_lock(out_dir: str):
    """One ranged fetch per out_dir at a time (the record and the local index are read-modify-
    write); a second process waits, then finds the first one's groups already there."""
    import fcntl
    os.makedirs(out_dir, exist_ok=True)
    fd = os.open(os.path.join(out_dir, ".lock"), os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _load_record(out_dir: str, src: RangeSource) -> dict:
    p = os.path.join(out_dir, RANGED_RECORD)
    if not os.path.exists(p):
        return {"schema": RANGED_SCHEMA, "source": src.base, "files": {}, "aux": {}, "groups": {}}
    try:
        with open(p) as f:
            rec = json.load(f)
    except (OSError, ValueError) as e:
        raise FetchError(f"unreadable {RANGED_RECORD} in {out_dir}: {e}") from e
    if rec.get("schema") != RANGED_SCHEMA or rec.get("source") != src.base:
        raise FetchError(f"{out_dir} holds tensors from {rec.get('source')!r}; refusing to mix in "
                         f"{src.base!r}")
    return rec


def _save_record(out_dir: str, rec: dict) -> None:
    _atomic_write(os.path.join(out_dir, RANGED_RECORD), json.dumps(rec, indent=1).encode())


def _fetch_small(src: RangeSource, path: str, out_dir: str, rec: dict, dest_rel: str) -> bytes:
    """A small whole file (config, index, tokenizer) into out_dir/dest_rel, reused if its recorded
    sha256 still matches. TLS is its only integrity root, so an untrusted source is refused."""
    dest = _safe_rel(out_dir, dest_rel)
    known = rec["aux"].get(path)
    if known and os.path.exists(dest):
        with open(dest, "rb") as f:
            data = f.read()
        if hashlib.sha256(data).hexdigest() == known["sha256"]:
            return data
    if not src.trusted:
        raise FetchError(f"{path}: whole-file fetch needs a trusted source (https pinned to a "
                         "commit) until manifest v2 signs it")
    buf = bytearray()
    total = src.read(path, 0, 1, lambda _o, c: buf.extend(c))
    if total > MAX_AUX_BYTES:
        raise FetchError(f"{path}: {total} B is not a small file")
    if total > 1:
        src.read(path, 1, total, lambda _o, c: buf.extend(c), total=total)
    data = bytes(buf)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    _atomic_write(dest, data)
    rec["aux"][path] = {"path": dest_rel, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return data


def _source_headers(src: RangeSource, files: list, out_dir: str, rec: dict, workers: int) -> dict:
    """Validated headers of `files` ({file: {"size", "tensors"}}), cached under out_dir/.source/
    and reused only while the cached bytes hash to the recorded header sha256."""
    def one(fn):
        cache = _safe_rel(out_dir, f".source/{fn}.header")
        known = rec["files"].get(fn)
        if known and os.path.exists(cache):
            with open(cache, "rb") as f:
                raw = f.read()
            if hashlib.sha256(raw).hexdigest() == known["header_sha256"]:
                return fn, raw, known["size"]
        raw, total = _read_header(src, fn)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        _atomic_write(cache, raw)
        return fn, raw, total
    out = {}
    with cf.ThreadPoolExecutor(max(1, min(workers, len(files) or 1))) as ex:
        for fn, raw, total in ex.map(one, files):
            out[fn] = {"size": total, "tensors": parse_safetensors_header(raw, total, f"{fn}: ")}
            rec["files"][fn] = {"size": total, "header_sha256": hashlib.sha256(raw).hexdigest()}
    return out


def _plan_group(names, weight_map: dict, headers: dict, max_job: int, expected: dict | None = None):
    """Source tensors in (file, offset) order laid back to back in one output file, and the Range
    jobs that fill it: contiguous source tensors coalesce (<= max_job), big ones split. A header
    whose dtype/shape disagrees with `expected` is refused before any byte moves."""
    items = []
    for nm in names:
        fn = weight_map[nm]
        t = headers[fn]["tensors"].get(nm)
        if t is None:
            raise FetchError(f"{nm}: the index maps it to {fn}, whose header has no such tensor")
        want = (expected or {}).get(nm)
        if want and (want["dtype"], list(want["shape"])) != (t["dtype"], t["shape"]):
            raise FetchError(f"{nm}: header says {t['dtype']}x{t['shape']}, expected "
                             f"{want['dtype']}x{list(want['shape'])}")
        items.append((fn, t["start"], t["nbytes"], nm, t["dtype"], t["shape"]))
    items.sort(key=lambda x: (x[0], x[1], x[3]))
    out_hdr, off = {}, 0
    for fn, s, nb, nm, dt, sh in items:
        out_hdr[nm] = {"dtype": dt, "shape": sh, "data_offsets": [off, off + nb]}
        off += nb
    jobs, dst = [], 0
    for fn, s, nb, *_ in items:
        if nb and jobs and jobs[-1][0] == fn and jobs[-1][1] + jobs[-1][2] == s \
                and jobs[-1][2] + nb <= max_job:
            jobs[-1][2] += nb
        elif nb:
            jobs.append([fn, s, nb, dst])
        dst += nb
    split = [(fn, s + o, min(max_job, nb - o), d + o)
             for fn, s, nb, d in jobs for o in range(0, nb, max_job)]
    return items, out_hdr, off, split


def _pwrite_all(fd: int, data, off: int) -> None:
    mv = memoryview(data)
    while mv:
        n = os.pwrite(fd, mv, off)
        mv, off = mv[n:], off + n


def _hash_region(fd: int, off: int, n: int) -> str:
    h = hashlib.sha256()
    end = off + n
    while off < end:
        chunk = os.pread(fd, min(1 << 22, end - off), off)
        if not chunk:
            break
        h.update(chunk)
        off += len(chunk)
    return h.hexdigest()


def _read_journal(path: str, plan_id: str) -> dict:
    """{job index: sha256} of the ranges a previous run completed for THIS plan (else {})."""
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError:
        return {}
    if not lines or lines[0] != f"plan {plan_id}":
        return {}
    done = {}
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) == 2 and _DIGITS_RE.fullmatch(parts[0]) and len(parts[1]) == 64:
            done[int(parts[0])] = parts[1]
    return done


def _write_journal(path: str, plan_id: str, done: dict) -> None:
    _atomic_write(path, "".join([f"plan {plan_id}\n"] + [f"{i} {h}\n" for i, h in sorted(done.items())])
                  .encode())


def _run_parallel(fn, items, workers: int) -> None:
    """fn over items on a thread pool; the first failure cancels what has not started and is
    re-raised once the running ones finish (their completed ranges are journaled)."""
    if not items:
        return
    first = None
    with cf.ThreadPoolExecutor(max(1, min(workers, len(items)))) as ex:
        futs = [ex.submit(fn, i) for i in items]
        for f in cf.as_completed(futs):
            if f.cancelled() or f.exception() is None or first is not None:
                continue
            first = f.exception()
            for x in futs:
                x.cancel()
    if first is not None:
        raise first


def _fetch_group(src: RangeSource, g: str, path: str, items, out_hdr: dict, data_len: int, jobs,
                 headers: dict, expected: dict | None, workers: int) -> dict:
    """Fill one output safetensors file from its Range jobs, resumably: a `.part` file plus a
    `.journal` of completed job hashes, both keyed by the plan (so a different plan never reuses
    them). On resume, journaled ranges are re-hashed on disk and refetched if they no longer
    match. The file is promoted only after every tensor is hashed and, if `expected` names it,
    matches; a mismatch drops exactly the ranges holding the bad tensors and raises."""
    hdr = dict(out_hdr)
    hdr["__metadata__"] = {"format": "pt", "shard.group": g, "shard.source": src.base}
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    head = struct.pack("<Q", len(hb)) + hb
    base = len(head)
    plan_id = hashlib.sha256(head + json.dumps(jobs).encode()).hexdigest()
    part, journal = path + ".part", path + ".journal"
    done = _read_journal(journal, plan_id)
    try:
        fd = os.open(part, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o644)
    except OSError as e:
        raise FetchError(f"{g}: cannot open {part} ({e})") from e
    try:
        if os.fstat(fd).st_size != base + data_len or os.pread(fd, base, 0) != head:
            done = {}
            os.ftruncate(fd, 0)
            _pwrite_all(fd, head, 0)
            os.ftruncate(fd, base + data_len)
        good = {i: h for i, h in done.items()
                if i < len(jobs) and _hash_region(fd, base + jobs[i][3], jobs[i][2]) == h}
        if len(good) < len(done):
            _log(f"  {g}: {len(done) - len(good)} journaled range(s) no longer match on disk - refetching")
        if good:
            _log(f"  resume {g}: {len(good)}/{len(jobs)} ranges already on disk")
        _write_journal(journal, plan_id, good)
        lock = threading.Lock()
        with open(journal, "a") as jf:
            def run(i):
                fn, s, nb, d = jobs[i]
                h = hashlib.sha256()

                def sink(off, chunk):
                    _pwrite_all(fd, chunk, base + d + (off - s))
                    h.update(chunk)
                src.read(fn, s, s + nb, sink, total=headers[fn]["size"])
                with lock:
                    jf.write(f"{i} {h.hexdigest()}\n")
                    jf.flush()
            _run_parallel(run, [i for i in range(len(jobs)) if i not in good], workers)

        fh = hashlib.sha256(os.pread(fd, base, 0))
        tsha = {}
        for nm, t in sorted(out_hdr.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            b, e = t["data_offsets"]
            th = hashlib.sha256()
            pos = base + b
            while pos < base + e:
                chunk = os.pread(fd, min(1 << 22, base + e - pos), pos)
                if not chunk:
                    raise FetchError(f"{g}: short read of {part}")
                th.update(chunk)
                fh.update(chunk)
                pos += len(chunk)
            tsha[nm] = th.hexdigest()
        bad = sorted(n for n in tsha if expected and n in expected and expected[n]["sha256"] != tsha[n])
        if bad:
            spans = [out_hdr[n]["data_offsets"] for n in bad]
            keep = {i: h for i, h in _read_journal(journal, plan_id).items()
                    if not any(jobs[i][3] < e and b < jobs[i][3] + jobs[i][2] for b, e in spans)}
            _write_journal(journal, plan_id, keep)
            raise FetchError(f"{g}: sha256 mismatch on {len(bad)} tensor(s), first {bad[0]} - "
                             "corrupt or tampered range, not promoted")
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(part, path)
    os.remove(journal)
    return {"path": os.path.basename(path), "size": base + data_len, "sha256": fh.hexdigest(),
            "tensors": {nm: {"dtype": dt, "shape": sh, "src": fn, "src_offset": s, "nbytes": nb,
                             "sha256": tsha[nm]} for fn, s, nb, nm, dt, sh in items}}


def _group_intact(path: str, prev: dict, names, expected: dict | None) -> bool:
    """A finished group is reused only if it holds the same tensors, re-hashes to its recorded
    sha256 (size alone is never trusted) and agrees with any expected per-tensor hash."""
    if set(prev.get("tensors", {})) != set(names) or not os.path.exists(path) \
            or os.path.getsize(path) != prev["size"]:
        return False
    if expected and any((expected[n]["sha256"], expected[n]["dtype"], list(expected[n]["shape"]))
                        != (t["sha256"], t["dtype"], t["shape"])
                        for n, t in prev["tensors"].items() if n in expected):
        return False
    return mf.sha256_file(path)[0] == prev["sha256"]


def fetch_tensors(src: RangeSource, weight_map: dict, groups: dict, out_dir: str, *,
                  expected: dict | None = None, workers: int = 8,
                  max_job: int = DEFAULT_MAX_JOB) -> dict:
    """Range-fetch exactly the tensors in `groups` ({group: [tensor names]}, see select_tensors)
    into out_dir/<group>.safetensors (original names, loadable by safetensors), and keep
    out_dir/ranged.json (source, per-file header sha256, per-group file sha256, per-tensor source
    offset + sha256) and out_dir/model.safetensors.index.json (every fetched tensor -> its local
    file) current. Incremental: finished groups are re-hashed and kept, other groups in the dir
    stay. `expected` ({tensor: {"sha256", "dtype", "shape"}}) is the R1b hook: any mismatch fails
    closed. Returns the record. Use a dedicated dir: the file-granular fetch_block_range
    quarantines unknown files."""
    with _dir_lock(out_dir):
        return _fetch_tensors(src, weight_map, groups, out_dir, expected, workers, max_job)


def _fetch_tensors(src, weight_map, groups, out_dir, expected, workers, max_job) -> dict:
    wanted = [n for v in groups.values() for n in v]
    for n, e in (expected or {}).items():
        if not isinstance(e, dict) or not {"sha256", "dtype", "shape"} <= set(e):
            raise FetchError(f"expected[{n!r}] must carry sha256, dtype and shape")
    if not src.trusted and (not expected or any(n not in expected for n in wanted)):
        raise FetchError(f"{src.base} is not a trusted source (https pinned to a commit) and not "
                         "every tensor has an expected hash - nothing would vouch for the bytes")
    for n in wanted:
        if n not in weight_map:
            raise FetchError(f"{n} is not in the index weight_map")
        _check_remote_path(weight_map[n])
    rec = _load_record(out_dir, src)
    headers = _source_headers(src, sorted({weight_map[n] for n in wanted}), out_dir, rec, workers)
    _save_record(out_dir, rec)
    for g, names in groups.items():
        items, out_hdr, data_len, jobs = _plan_group(names, weight_map, headers, max_job, expected)
        path = _safe_rel(out_dir, f"{g}.safetensors")
        prev = rec["groups"].get(g)
        if prev and _group_intact(path, prev, names, expected):
            _log(f"  have {g}")
            continue
        _log(f"  fetch {g}: {len(names)} tensors, {data_len / 2**30:.3f} GiB in {len(jobs)} ranges "
             f"from {len({j[0] for j in jobs})} files")
        rec["groups"][g] = _fetch_group(src, g, path, items, out_hdr, data_len, jobs, headers,
                                        expected, workers)
        _save_record(out_dir, rec)
    wm = {n: grp["path"] for grp in rec["groups"].values() for n in grp["tensors"]}
    total = sum(t["nbytes"] for grp in rec["groups"].values() for t in grp["tensors"].values())
    _atomic_write(os.path.join(out_dir, LOCAL_INDEX), json.dumps(
        {"metadata": {"total_size": total, "source": rec["source"]},
         "weight_map": dict(sorted(wm.items()))}, indent=1).encode())
    return rec


def fetch_layer_tensors(src: RangeSource, out_dir: str, lo: int, hi: int, *, head: bool = False,
                        tail: bool = False, tied: bool | None = None, layout: dict | None = None,
                        extra=(), aux=("config.json",), index: str = LOCAL_INDEX,
                        expected: dict | None = None, workers: int = 8,
                        max_job: int = DEFAULT_MAX_JOB) -> dict:
    """The ranged counterpart of fetch_block_range: fetch the pinned index (kept under .source/)
    and the small `aux` files, select layers [lo:hi) (+ embedding / final norm + lm_head /
    `extra`) off the index's own tensor names, and range-fetch exactly those tensors. File names in
    the index and the `aux` names resolve relative to the index's directory (a drafter under
    `dflash/`). `tied` defaults to config.json's tie_word_embeddings (never inferred from the map)."""
    root = posixpath.dirname(_check_remote_path(index))

    def rel(p):
        return posixpath.join(root, p) if root else p
    with _dir_lock(out_dir):
        rec = _load_record(out_dir, src)
        try:
            raw = json.loads(_fetch_small(src, index, out_dir, rec, f".source/{index}"))["weight_map"]
            weight_map = {n: rel(fn) for n, fn in raw.items()}
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise FetchError(f"{index}: not a safetensors index ({e})") from e
        cfg = {}
        for a in aux:
            data = _fetch_small(src, rel(a), out_dir, rec, a)
            if a == "config.json":
                try:
                    cfg = json.loads(data)
                except ValueError as e:
                    raise FetchError(f"config.json is not JSON ({e})") from e
        _save_record(out_dir, rec)
        if tied is None:
            tied = bool(cfg.get("tie_word_embeddings", False))
        groups = select_tensors(weight_map, lo, hi, head=head, tail=tail, tied=tied, layout=layout,
                                extra=extra)
        return _fetch_tensors(src, weight_map, groups, out_dir, expected, workers, max_job)


def _hf_headers() -> dict:
    """HF auth, portable (same order as m25_pull_range): env HF_TOKEN, else ~/.hf_token,
    else anonymous (public repos need none)."""
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        p = os.path.expanduser("~/.hf_token")
        if os.path.isfile(p):
            tok = open(p).read().strip()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _default_mirror(manifest: dict) -> str | None:
    """An HF-style resolve base for the manifest's source repo, so `--mirror` is optional
    when the shards pin an HF repo/revision (MirrorProvider routes per-shard from there)."""
    for s in manifest.get("shards", []):
        if s.get("repo") and s.get("revision"):
            return f"https://huggingface.co/{s['repo']}/resolve/{s['revision']}/"
    return None


def _main_ranged(argv: list) -> int:
    """`python -m shard.fetch ranged ...`: tensor-granular fetch of layers [lo:hi) from an HF repo
    pinned to a commit (the promoted scratch fetch_layers.py). Same stdout contract as _main:
        SHARD_FETCH_DONE {groups, bytes, dir}   / SHARD_FETCH_FATAL {error}."""
    import argparse
    ap = argparse.ArgumentParser(prog="python -m shard.fetch ranged",
                                 description="range-fetch exactly a layer range's tensors")
    ap.add_argument("--repo", help="HF repo id (with --revision)")
    ap.add_argument("--revision", help="full 40-hex commit sha the fetch pins")
    ap.add_argument("--base", help="instead of --repo/--revision: a base URL (trusted only if https)")
    ap.add_argument("--dir", required=True, help="dedicated output dir")
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--head", action="store_true", help="also the embedding")
    ap.add_argument("--tail", action="store_true", help="also the final norm + lm_head")
    ap.add_argument("--extra", action="append", default=[], help="extra module prefix (repeatable)")
    ap.add_argument("--layout", action="append", default=[],
                    help="KEY=PREFIX override of layers/embed/norm/lm_head (repeatable; V4: "
                         "embed=embed, lm_head=head)")
    ap.add_argument("--index", default=LOCAL_INDEX, help="index path in the repo (e.g. dflash/...)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    try:
        layout = dict(kv.split("=", 1) for kv in a.layout)
        if a.base:              # never hand the HF token to an arbitrary base URL
            src = RangeSource(a.base)
        elif a.repo and a.revision:
            src = RangeSource.huggingface(a.repo, a.revision, _hf_headers())
        else:
            raise FetchError("give --repo and --revision, or --base")
        rec = fetch_layer_tensors(src, a.dir, a.lo, a.hi, head=a.head, tail=a.tail, layout=layout,
                                  extra=a.extra, index=a.index, workers=a.workers)
        print("SHARD_FETCH_DONE " + json.dumps({
            "groups": len(rec["groups"]), "dir": a.dir,
            "bytes": sum(g["size"] for g in rec["groups"].values())}), flush=True)
        return 0
    except Exception as e:  # noqa: BLE001 — the daemon reads this as the failure signal
        print("SHARD_FETCH_FATAL " + json.dumps({"error": f"{type(e).__name__}: {e}"}), flush=True)
        return 1


def _main() -> int:
    """`python -m shard.stage`'s sibling: the daemon's verified weight-pull entrypoint. Reads a
    signed manifest, fetches layers [lo:hi) PEERS-FIRST then mirror, re-hashing every byte, into
    --dir. Machine-readable stdout contract so the supervising daemon can wait on it:
        SHARD_FETCH_DONE {files, bytes, dir}   / SHARD_FETCH_FATAL {error}.
    `python -m shard.fetch ranged ...` is the tensor-granular path (_main_ranged)."""
    if sys.argv[1:2] == ["ranged"]:
        return _main_ranged(sys.argv[2:])
    import argparse
    ap = argparse.ArgumentParser(prog="python -m shard.fetch",
                                 description="verified peers-first weight fetch for a layer range")
    ap.add_argument("--manifest", required=True, help="signed shard-manifest/1 JSON (path)")
    ap.add_argument("--dir", required=True, help="model dir to fetch into")
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--head", action="store_true", help="also pull embed + tokenizer (stage 0 / coord)")
    ap.add_argument("--tail", action="store_true", help="also pull norm + lm_head (last stage)")
    ap.add_argument("--role", default="stage", choices=["stage", "coordinator"])
    ap.add_argument("--bootstrap", default="", help="comma-separated ringmate sidecar multiaddrs (peers-first)")
    ap.add_argument("--mirror", default=None, help="HTTP mirror base (default: the manifest's HF repo)")
    ap.add_argument("--local-dir", default=None, help="a same-host seed dir tried before peers (self-test)")
    ap.add_argument("--sidecar", default=None, help="sidecar binary for the libp2p provider")
    ap.add_argument("--key", default=None, help="node key for the libp2p provider")
    ap.add_argument("--pubkey", default=None, help="pin the manifest publisher pubkey (base64)")
    ap.add_argument("--manifest-cid", default=None,
                    help="content id the manifest file must hash to (bare CIDv1, or the "
                         "assignment's mf1:<name>@<cid> ref verbatim)")
    ap.add_argument("--expect-model-id", default=None,
                    help="refuse a manifest whose model_id differs (assignment cross-check)")
    ap.add_argument("--expect-layer-count", type=int, default=None,
                    help="refuse a manifest whose layer_count differs (assignment cross-check)")
    a = ap.parse_args()
    try:
        if a.manifest_cid:
            check_manifest_cid(a.manifest, a.manifest_cid)
        manifest = json.load(open(a.manifest))
        check_expected(manifest, a.expect_model_id, a.expect_layer_count)
        provider = build_chain_provider(
            mirror=a.mirror or _default_mirror(manifest),
            bootstrap=[b.strip() for b in a.bootstrap.split(",") if b.strip()],
            sidecar_bin=a.sidecar, key=a.key, local_dir=a.local_dir)
        if isinstance(provider, MirrorProvider) or (
                isinstance(provider, ChainProvider)
                and any(isinstance(p, MirrorProvider) for p in provider.providers)):
            # MirrorProvider doesn't carry auth headers itself; inject HF auth via the env it reads
            for p in ([provider] if isinstance(provider, MirrorProvider) else provider.providers):
                if isinstance(p, MirrorProvider):
                    p.headers = {**_hf_headers(), **p.headers}
        paths = fetch_block_range(manifest, a.dir, a.lo, a.hi, is_head=a.head, is_tail=a.tail,
                                  role=a.role, provider=provider, expected_pubkey=a.pubkey)
        total = sum(os.path.getsize(p) for p in paths)
        print("SHARD_FETCH_DONE " + json.dumps({"files": len(paths), "bytes": total, "dir": a.dir}),
              flush=True)
        return 0
    except Exception as e:  # noqa: BLE001 — the daemon reads this as the failure signal
        print("SHARD_FETCH_FATAL " + json.dumps({"error": f"{type(e).__name__}: {e}"}), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(_main())
