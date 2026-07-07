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
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from shard import manifest as mf                    # noqa: E402
from shard.fetch import (FetchError, LocalDirProvider, Provider, block_for_stage,  # noqa: E402
                         fetch_block, shards_for_block)
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
    shards = [
        _shard(src, "model-00001.safetensors", b"weights-file-one-" + b"A" * 200),
        _shard(src, "model-00002.safetensors", b"weights-file-two-" + b"B" * 200),
        _shard(src, "config.json", b'{"model_type":"minimax_m2"}', kind="config"),
        _shard(src, "model.safetensors.index.json", b'{"weight_map":{}}', kind="config"),
        _shard(src, "tokenizer.json", b'{"tok":1}', kind="tokenizer"),
    ]
    weight_map = {}
    for j in (0, 1):
        weight_map[f"model.layers.{j}.self_attn.q_proj.weight"] = "model-00001.safetensors"
    for j in (2, 3):
        weight_map[f"model.layers.{j}.self_attn.q_proj.weight"] = "model-00002.safetensors"
    weight_map["model.embed_tokens.weight"] = "model-00001.safetensors"
    weight_map["model.norm.weight"] = "model-00002.safetensors"
    weight_map["lm_head.weight"] = "model-00002.safetensors"
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
