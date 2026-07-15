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
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from abc import ABC, abstractmethod
from urllib.parse import urlsplit
from urllib.request import Request

from . import manifest as mf


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

    def __init__(self, bootstrap: list[str] | None = None, sidecar_bin: str | None = None,
                 key: str | None = None, timeout: int = 1800):
        self.bin = sidecar_bin or os.environ.get("SHARD_SIDECAR", "/tmp/sidecar")
        env_bs = [b for b in os.environ.get("SHARD_DHT_BOOTSTRAP", "").split(",") if b]
        self.bootstrap = list(bootstrap) if bootstrap is not None else env_bs
        self.key = key
        self.timeout = timeout

    def fetch(self, shard: dict, dest: str) -> None:
        cmd = [self.bin, "-fetch-cid", shard["shard_id"], "-fetch-out", dest,
               "-fetch-size", str(shard["size"]), "-fetch-timeout", str(self.timeout)]
        for b in self.bootstrap:
            cmd += ["-dht-bootstrap", b]
        if self.key:
            cmd += ["-key", self.key]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout + 60)
        except FileNotFoundError:
            raise ProviderUnavailable(f"sidecar binary {self.bin!r} not found")
        except subprocess.TimeoutExpired:
            raise ProviderUnavailable(f"libp2p fetch of {shard['path']} timed out ({self.timeout}s)")
        if r.returncode != 0:
            tail = ((r.stderr or r.stdout).strip().splitlines() or ["no output"])[-1]
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


def _main() -> int:
    """`python -m shard.stage`'s sibling: the daemon's verified weight-pull entrypoint. Reads a
    signed manifest, fetches layers [lo:hi) PEERS-FIRST then mirror, re-hashing every byte, into
    --dir. Machine-readable stdout contract so the supervising daemon can wait on it:
        SHARD_FETCH_DONE {files, bytes, dir}   / SHARD_FETCH_FATAL {error}."""
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
    a = ap.parse_args()
    try:
        manifest = json.load(open(a.manifest))
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
