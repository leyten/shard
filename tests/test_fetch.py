"""Verified weight-fetch trust root (shard/fetch.py + shard/manifest.py).

This is the content-addressed "torrent-for-weights" primitive: a node fetches only its block's
shards from an untrusted provider and re-hashes every byte against a SIGNED manifest before the
loader sees it. It had ZERO test coverage. These are the adversarial guarantees, exercised on CPU
with a synthetic signed manifest + LocalDirProvider (no network):

  - a tampered / wrong-size / wrong-CID shard is REJECTED and the bad file DELETED (nothing
    half-verified is left for the loader);
  - a shard path that is absolute or escapes the model dir is REFUSED (a malicious publisher can't
    write outside model_dir);
  - a manifest with a bad signature, or not matching the catalog-pinned publisher, is REJECTED
    before any byte is fetched;
  - a node fetches ONLY its block's shards (selectivity), and a correct cached file is not re-fetched
    while a same-size-wrong-content file IS re-fetched (size match alone is never trusted).

Run: python3 -m pytest tests/test_fetch.py -q
"""
import io
import json
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from shard import manifest as mf                    # noqa: E402
from shard.fetch import (FetchError, LocalDirProvider, Provider, block_for_stage,  # noqa: E402
                         fetch_block, fetch_block_range, shards_for_block)
from shard.manifest import ManifestError            # noqa: E402


def _shard(src_dir, path, data, kind="weights"):
    """Write a file into the source 'mirror' and return its manifest shard dict (real hash)."""
    full = os.path.join(src_dir, path)
    os.makedirs(os.path.dirname(full) or src_dir, exist_ok=True)
    with open(full, "wb") as f:
        f.write(data)
    sha, size = mf.sha256_file(full)
    return {"path": path, "size": size, "sha256": sha, "shard_id": mf.cidv1_raw(sha), "kind": kind}


def _repo(tmp_path, priv, layer_count=4, mutate=None):
    """A synthetic 2-file M2.5-shaped repo (layers 0-1+embed in file1; 2-3+norm+lm_head in file2)
    plus config + tokenizer, and a SIGNED manifest over it. `mutate(manifest)` is applied BEFORE
    signing, so a bad claimed size/CID rides a VALID signature (exercising _verify, not the sig check)."""
    src = str(tmp_path / "src")
    os.makedirs(src, exist_ok=True)
    weight_map = {}
    for j in (0, 1):
        weight_map[f"model.layers.{j}.self_attn.q_proj.weight"] = "model-00001.safetensors"
    for j in (2, 3):
        weight_map[f"model.layers.{j}.self_attn.q_proj.weight"] = "model-00002.safetensors"
    weight_map["model.embed_tokens.weight"] = "model-00001.safetensors"
    weight_map["model.norm.weight"] = "model-00002.safetensors"
    weight_map["lm_head.weight"] = "model-00002.safetensors"
    shards = [
        _shard(src, "model-00001.safetensors", b"weights-file-one-" + b"A" * 200),
        _shard(src, "model-00002.safetensors", b"weights-file-two-" + b"B" * 200),
        _shard(src, "config.json", b'{"model_type":"minimax_m2"}', kind="config"),
        # the on-disk index must carry the SAME weight_map as the manifest — the fetch
        # refuses a mismatched pair (the runtime loader trusts the downloaded index)
        _shard(src, "model.safetensors.index.json",
               json.dumps({"weight_map": weight_map}).encode(), kind="config"),
        _shard(src, "tokenizer.json", b'{"tok":1}', kind="tokenizer"),
    ]
    manifest = {"schema": mf.SCHEMA, "model_id": "test/m2.5", "layer_count": layer_count,
                "weight_map": weight_map, "shards": shards}
    if mutate is not None:
        mutate(manifest)
    return src, mf.sign_manifest(manifest, priv)


def _mutate_shard(path, **fields):
    def _m(manifest):
        for s in manifest["shards"]:
            if s["path"] == path:
                s.update(fields)
    return _m


@pytest.fixture
def repo(tmp_path):
    priv = mf.gen_key()
    src, manifest = _repo(tmp_path, priv)
    return {"src": src, "manifest": manifest, "pub": mf.pub_b64(priv),
            "dest": str(tmp_path / "model"), "priv": priv}


class _CountingProvider(Provider):
    def __init__(self, root):
        self.inner = LocalDirProvider(root)
        self.fetched = []

    def fetch(self, shard, dest):
        self.fetched.append(shard["path"])
        self.inner.fetch(shard, dest)


# ---- 1. happy path + selectivity -------------------------------------------------------------------

def test_head_fetch_verifies_and_selects_its_shards(repo):
    prov = _CountingProvider(repo["src"])
    paths = fetch_block(repo["manifest"], repo["dest"], stage=0, nstages=2, role="coordinator",
                        provider=prov, expected_pubkey=repo["pub"])
    got = {os.path.basename(p) for p in paths}
    assert "model-00001.safetensors" in got               # head's layers 0-1 + embed
    assert "model-00002.safetensors" not in got           # NOT the tail's file
    assert "tokenizer.json" in got                        # coordinator pulls the tokenizer
    assert "config.json" in got and "model.safetensors.index.json" in got
    for p in paths:                                       # every file present + hash-verified
        assert os.path.exists(p)


def test_tail_fetch_selects_norm_lmhead(repo):
    paths = fetch_block(repo["manifest"], repo["dest"], stage=1, nstages=2, role="stage",
                        provider=LocalDirProvider(repo["src"]), expected_pubkey=repo["pub"])
    got = {os.path.basename(p) for p in paths}
    assert "model-00002.safetensors" in got and "model-00001.safetensors" not in got
    assert "tokenizer.json" not in got                    # a plain stage does not pull the tokenizer


def test_block_for_stage_tiles_gaplessly():
    n = 62
    spans = [block_for_stage(n, s, 5) for s in range(5)]
    assert spans[0][0] == 0 and spans[-1][1] == n
    for (a, b), (c, d) in zip(spans, spans[1:]):
        assert b == c                                     # no gap / overlap


# ---- 2. byte integrity: tampered / wrong size / wrong CID are rejected AND deleted -----------------

def test_tampered_shard_rejected_and_deleted(repo):
    # corrupt the SOURCE file to the SAME size but different bytes, AFTER the manifest was signed
    bad = os.path.join(repo["src"], "model-00001.safetensors")
    n = os.path.getsize(bad)
    with open(bad, "wb") as f:
        f.write(b"X" * n)
    with pytest.raises(FetchError, match="sha256 mismatch"):
        fetch_block(repo["manifest"], repo["dest"], stage=0, nstages=2, role="stage",
                    provider=LocalDirProvider(repo["src"]), expected_pubkey=repo["pub"])
    # the corrupt file must NOT be left in the model dir for the loader to pick up
    assert not os.path.exists(os.path.join(repo["dest"], "model-00001.safetensors"))


def test_size_mismatch_rejected(tmp_path):
    """A VALIDLY-SIGNED manifest whose claimed size != the real bytes (the sig covers the wrong
    size, so this reaches _verify, not the manifest check)."""
    priv = mf.gen_key()
    src, manifest = _repo(tmp_path, priv,
                          mutate=_mutate_shard("model-00001.safetensors", size=99999))
    with pytest.raises(FetchError, match="size"):
        fetch_block(manifest, str(tmp_path / "model"), stage=0, nstages=2, role="stage",
                    provider=LocalDirProvider(src), expected_pubkey=mf.pub_b64(priv))


def test_cid_mismatch_rejected(tmp_path):
    """sha256 + size match the bytes, but the CID doesn't derive from that sha — signed in, so it
    reaches the _verify CID check."""
    priv = mf.gen_key()
    src, manifest = _repo(tmp_path, priv,
                          mutate=_mutate_shard("model-00001.safetensors", shard_id=mf.cidv1_raw("00" * 32)))
    with pytest.raises(FetchError, match="CID"):
        fetch_block(manifest, str(tmp_path / "model"), stage=0, nstages=2, role="stage",
                    provider=LocalDirProvider(src), expected_pubkey=mf.pub_b64(priv))


# ---- 3. path traversal: a malicious/compromised publisher can't escape model_dir -------------------

@pytest.mark.parametrize("evil", ["../escape.bin", "/etc/evil.bin", "a/../../escape.bin"])
def test_path_traversal_refused(tmp_path, evil):
    priv = mf.gen_key()
    src = str(tmp_path / "src")
    os.makedirs(src, exist_ok=True)
    # a validly-SIGNED manifest whose shard path escapes model_dir (the path is a signed field, so
    # this models a malicious publisher, not a mirror injection)
    s = _shard(src, "ok.bin", b"data", kind="config")
    s["path"] = evil
    manifest = mf.sign_manifest(
        {"schema": mf.SCHEMA, "model_id": "x", "layer_count": 2, "weight_map": {}, "shards": [s]}, priv)
    dest = str(tmp_path / "model")
    with pytest.raises(FetchError, match="unsafe shard path"):
        fetch_block(manifest, dest, stage=0, nstages=1, role="stage",
                    provider=LocalDirProvider(src), expected_pubkey=mf.pub_b64(priv))
    assert not os.path.exists(os.path.join(tmp_path, "escape.bin"))   # nothing written outside


# ---- 4. manifest signature + catalog pin -----------------------------------------------------------

def test_tampered_manifest_rejected(repo):
    repo["manifest"]["layer_count"] = 999                 # mutate a signed field after signing
    with pytest.raises(ManifestError):
        fetch_block(repo["manifest"], repo["dest"], stage=0, nstages=2, role="stage",
                    provider=LocalDirProvider(repo["src"]), expected_pubkey=repo["pub"])


def test_wrong_pinned_publisher_rejected(repo):
    other = mf.pub_b64(mf.gen_key())                      # a different, catalog-pinned publisher
    with pytest.raises(ManifestError, match="pinned key"):
        fetch_block(repo["manifest"], repo["dest"], stage=0, nstages=2, role="stage",
                    provider=LocalDirProvider(repo["src"]), expected_pubkey=other)


def test_unsigned_manifest_rejected(repo):
    repo["manifest"].pop("signature")
    with pytest.raises(ManifestError, match="unsigned"):
        fetch_block(repo["manifest"], repo["dest"], stage=0, nstages=2, role="stage",
                    provider=LocalDirProvider(repo["src"]), expected_pubkey=None)


# ---- 5. cache: correct file skipped; same-size-wrong-content re-fetched ----------------------------

def test_correct_cache_is_not_refetched(repo):
    prov = _CountingProvider(repo["src"])
    fetch_block(repo["manifest"], repo["dest"], stage=0, nstages=2, role="stage",
                provider=prov, expected_pubkey=repo["pub"])
    first = len(prov.fetched)
    assert first > 0
    prov2 = _CountingProvider(repo["src"])                # second run: everything cached + hash-matches
    fetch_block(repo["manifest"], repo["dest"], stage=0, nstages=2, role="stage",
                provider=prov2, expected_pubkey=repo["pub"])
    assert prov2.fetched == [], "a correct cached block must not be re-fetched"


def test_wrong_content_cache_is_refetched(repo):
    # pre-place a same-SIZE but wrong-content file in the dest: size match alone must NOT be trusted
    os.makedirs(repo["dest"], exist_ok=True)
    good = os.path.join(repo["src"], "model-00001.safetensors")
    stale = os.path.join(repo["dest"], "model-00001.safetensors")
    with open(stale, "wb") as f:
        f.write(b"Z" * os.path.getsize(good))
    prov = _CountingProvider(repo["src"])
    fetch_block(repo["manifest"], repo["dest"], stage=0, nstages=2, role="stage",
                provider=prov, expected_pubkey=repo["pub"])
    assert "model-00001.safetensors" in prov.fetched, "stale same-size cache must be re-fetched"
    sha, _ = mf.sha256_file(stale)                        # and now it holds the correct verified bytes
    assert sha == next(s["sha256"] for s in repo["manifest"]["shards"]
                       if s["path"] == "model-00001.safetensors")


# ---- 6. fetch_block_range: the explicit-range (uneven-split) deploy entry point --------------------

def test_fetch_block_range_uneven_middle_spans_files(repo):
    """A middle stage assigned layers [1:3] spans the file boundary (layer 1 in file1, layer 2 in
    file2) — the deploy's uneven select_ring splits aren't stage-aligned. Both files must come."""
    paths = fetch_block_range(repo["manifest"], repo["dest"], 1, 3, is_head=False, is_tail=False,
                              role="stage", provider=LocalDirProvider(repo["src"]),
                              expected_pubkey=repo["pub"])
    got = {os.path.basename(p) for p in paths}
    assert "model-00001.safetensors" in got and "model-00002.safetensors" in got  # spans both
    assert "tokenizer.json" not in got                    # a middle stage pulls no tokenizer


def test_fetch_block_matches_range_on_even_split(repo):
    """fetch_block(stage=0,nstages=2) is just fetch_block_range over block_for_stage's [0:2]."""
    a = fetch_block(repo["manifest"], str(repo["dest"] + "_a"), stage=0, nstages=2, role="coordinator",
                    provider=LocalDirProvider(repo["src"]), expected_pubkey=repo["pub"])
    b = fetch_block_range(repo["manifest"], str(repo["dest"] + "_b"), 0, 2, is_head=True, is_tail=False,
                          role="coordinator", provider=LocalDirProvider(repo["src"]),
                          expected_pubkey=repo["pub"])
    assert {os.path.basename(p) for p in a} == {os.path.basename(p) for p in b}


def test_fetch_block_range_verifies_bytes(repo):
    """The range path re-hashes too: a tampered shard is rejected + deleted."""
    bad = os.path.join(repo["src"], "model-00002.safetensors")
    n = os.path.getsize(bad)                              # size BEFORE opening 'wb' truncates it
    with open(bad, "wb") as f:
        f.write(b"X" * n)                                 # same size, different bytes -> sha mismatch
    with pytest.raises(FetchError, match="sha256 mismatch"):
        fetch_block_range(repo["manifest"], repo["dest"], 2, 4, is_head=False, is_tail=True,
                          role="stage", provider=LocalDirProvider(repo["src"]), expected_pubkey=repo["pub"])
    assert not os.path.exists(os.path.join(repo["dest"], "model-00002.safetensors"))


# ---- 7. MirrorProvider survives a truncated (dropped-connection) download --------------------------

class _FakeResp:
    def __init__(self, payload, status=200, content_range=None):
        self._buf, self.status = io.BytesIO(payload), status
        # a real 206 always carries Content-Range; _download trusts it to place the body
        self.headers = {"Content-Range": content_range} if content_range else {}

    def read(self, n=-1):
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_mirror_provider_resumes_truncated_download(tmp_path, monkeypatch):
    """A dropped HTTP connection returns a PARTIAL body with no exception. The old code os.replace'd
    the short file and only _verify (later) caught the size mismatch — hard-failing the whole pull.
    Now _download rejects the short body and fetch()'s retry loop RESUMES it via Range to completion."""
    from shard import fetch as F
    full = bytes((i * 37) % 256 for i in range(5000))
    calls = {"n": 0}

    def fake_open(req, timeout=None):
        rng = req.get_header("Range")
        start = int(rng[len("bytes="):].rstrip("-")) if rng else 0
        calls["n"] += 1
        body = full[start:]
        if calls["n"] == 1:                              # first attempt drops at ~40%
            body = body[:2000]
        cr = f"bytes {start}-{len(full) - 1}/{len(full)}" if start else None
        return _FakeResp(body, status=206 if start else 200, content_range=cr)

    monkeypatch.setattr(F, "urlopen", fake_open)
    monkeypatch.setattr(F.time, "sleep", lambda s: None)   # don't wait between retries
    dest = str(tmp_path / "shard.bin")
    F.MirrorProvider("http://mirror/", retries=6).fetch({"path": "shard.bin", "size": len(full)}, dest)
    assert os.path.getsize(dest) == len(full)
    with open(dest, "rb") as f:
        assert f.read() == full                          # resumed to the EXACT bytes, not corrupted
    assert calls["n"] >= 2, "should have taken at least one resume"


def test_mirror_provider_no_overshoot_when_206_returns_full_body(tmp_path, monkeypatch):
    """The live-ring oversize bug (4,998,528,136 -> 5,018,451,080 = have + total). A mirror answered
    a Range request with the WHOLE body but a 206 status (Content-Range from 0 — a redirect/CDN
    dropped the Range). The old code trusted the 206 and APPENDED, growing the shard to have+total.
    Now the body is placed by its Content-Range, so a full body OVERWRITES — exact size, no overshoot."""
    from shard import fetch as F
    full = bytes((i * 53) % 256 for i in range(6000))
    calls = {"n": 0}

    def fake_open(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:                              # first attempt lands a partial, then drops
            return _FakeResp(full[:1500], status=200)
        # resume attempt: server IGNORES the Range and streams the whole file, tagged 206-from-0
        return _FakeResp(full, status=206, content_range=f"bytes 0-{len(full) - 1}/{len(full)}")

    monkeypatch.setattr(F, "urlopen", fake_open)
    monkeypatch.setattr(F.time, "sleep", lambda s: None)
    dest = str(tmp_path / "shard.bin")
    F.MirrorProvider("http://mirror/", retries=6).fetch({"path": "shard.bin", "size": len(full)}, dest)
    assert os.path.getsize(dest) == len(full)            # NOT 1500 + 6000
    with open(dest, "rb") as f:
        assert f.read() == full


def test_mirror_provider_caps_a_body_that_streams_past_size(tmp_path, monkeypatch):
    """A mirror that streams MORE than the manifest size can't overshoot onto disk: the copy is
    capped at the shard size (the sha256 in fetch_block still guards the bytes' correctness)."""
    from shard import fetch as F
    full = bytes((i * 17) % 256 for i in range(4000))
    monkeypatch.setattr(F, "urlopen",
                        lambda req, timeout=None: _FakeResp(full + b"\xff" * 500, status=200))
    monkeypatch.setattr(F.time, "sleep", lambda s: None)
    dest = str(tmp_path / "x.bin")
    F.MirrorProvider("http://mirror/", retries=2).fetch({"path": "x.bin", "size": len(full)}, dest)
    assert os.path.getsize(dest) == len(full)            # capped, not 4500
    with open(dest, "rb") as f:
        assert f.read() == full


def test_mirror_provider_unplaceable_range_fails_closed(tmp_path, monkeypatch):
    """A hostile/broken server answers a FRESH GET (no Range sent, no .part on disk) with a 206
    claiming a non-zero start. There is nothing to place it against, so we fail closed with a
    FetchError (not a stray FileNotFoundError from removing an absent .part)."""
    from shard import fetch as F
    monkeypatch.setattr(F, "urlopen", lambda req, timeout=None:
                        _FakeResp(b"x" * 100, status=206, content_range="bytes 500-599/9999"))
    monkeypatch.setattr(F.time, "sleep", lambda s: None)
    with pytest.raises(FetchError):
        F.MirrorProvider("http://mirror/", retries=2).fetch({"path": "x", "size": 9999}, str(tmp_path / "x"))


def test_mirror_provider_gives_up_after_retries(tmp_path, monkeypatch):
    """If the stream never completes, fetch raises (not a silent short file for _verify to catch late)."""
    from shard import fetch as F
    monkeypatch.setattr(F, "urlopen", lambda req, timeout=None: _FakeResp(b"tiny"))  # always short
    monkeypatch.setattr(F.time, "sleep", lambda s: None)
    with pytest.raises(FetchError):
        F.MirrorProvider("http://mirror/", retries=2).fetch({"path": "x", "size": 9999}, str(tmp_path / "x"))
