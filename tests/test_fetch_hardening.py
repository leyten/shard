"""Audit hardening of the weight trust root (shard/fetch.py + shard/manifest.py +
phase0/publish_manifest.py). Companion to test_fetch.py / test_publish_manifest.py —
these are the adversarial regressions from the external audit:

  M8  — _safe_rel's lexical check let a pre-existing SYMLINK component (or a symlink at
        the destination itself) redirect a shard write outside the model root; the
        publisher key was written world-readable (0644 under umask 022).
  M9  — python's default redirect handler forwards Authorization to the redirect target,
        so a mirror 302ing cross-origin exfiltrates the bearer (e.g. an HF_TOKEN).
  L1  — a provider claiming success without leaving a readable file raised a raw
        FileNotFoundError from _verify, which ChainProvider does not catch — one broken
        peer blocked a valid later mirror instead of falling back.
  M7  — publish fetched the mutable `main` ref TWICE (weight_map parsed from one fetch,
        the signed index hash taken from another), the index a manifest references could
        thus disagree with its own weight_map, and a declared separate tokenizer repo was
        not encoded for fetch routing.

Run: python3 -m pytest tests/test_fetch_hardening.py -q
"""
import hashlib
import io
import json
import os
import sys
import urllib.request

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "phase0"))

from shard import fetch as F                         # noqa: E402
from shard import manifest as mf                     # noqa: E402
from shard.fetch import FetchError, _safe_rel        # noqa: E402


# ---- M8: symlinks must not redirect shard writes outside model_dir ---------------------------------

def test_symlinked_dir_component_refused(tmp_path):
    """A lexically-clean path like 'sub/shard.bin' whose 'sub' is a pre-existing symlink
    pointing OUTSIDE the model root must be refused — the old check compared strings only."""
    outside = tmp_path / "outside"
    outside.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    os.symlink(str(outside), str(model / "sub"))
    with pytest.raises(FetchError, match="symlink"):
        _safe_rel(str(model), "sub/shard.bin")
    assert not os.path.exists(outside / "shard.bin")


def test_symlinked_destination_refused(tmp_path):
    """A symlink AT the destination filename redirects the open() itself."""
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"precious")
    model = tmp_path / "model"
    model.mkdir()
    os.symlink(str(victim), str(model / "model-00001.safetensors"))
    with pytest.raises(FetchError, match="symlink"):
        _safe_rel(str(model), "model-00001.safetensors")
    assert victim.read_bytes() == b"precious"


def test_symlink_inside_model_dir_still_refused(tmp_path):
    """Even an inside-root symlink is refused at the destination: it would alias one
    verified shard's bytes over another's."""
    model = tmp_path / "model"
    model.mkdir()
    (model / "real.bin").write_bytes(b"a")
    os.symlink(str(model / "real.bin"), str(model / "alias.bin"))
    with pytest.raises(FetchError, match="symlink"):
        _safe_rel(str(model), "alias.bin")


def test_plain_nested_path_still_allowed(tmp_path):
    """No symlinks involved -> unchanged behavior (nested dirs are fine)."""
    model = tmp_path / "model"
    model.mkdir()
    dest = _safe_rel(str(model), "nested/dir/shard.bin")
    assert dest.startswith(os.path.realpath(str(model)) + os.sep)


def test_model_dir_itself_a_symlink_is_fine(tmp_path):
    """Operators legitimately symlink the model dir to a big disk; that must keep working."""
    real = tmp_path / "disk"
    real.mkdir()
    link = tmp_path / "model"
    os.symlink(str(real), str(link))
    dest = _safe_rel(str(link), "shard.bin")
    assert dest == os.path.join(os.path.realpath(str(link)), "shard.bin")


# ---- M8: publisher key hygiene ----------------------------------------------------------------------

def test_save_key_is_0600(tmp_path):
    path = str(tmp_path / "publisher.key")
    old_umask = os.umask(0o022)                       # the deployment default that exposed 0644
    try:
        mf.save_key(mf.gen_key(), path)
    finally:
        os.umask(old_umask)
    assert os.stat(path).st_mode & 0o777 == 0o600
    mf.load_key(path)                                 # round-trips


def test_save_key_never_clobbers_existing(tmp_path):
    path = str(tmp_path / "publisher.key")
    mf.save_key(mf.gen_key(), path)
    with pytest.raises(FileExistsError):
        mf.save_key(mf.gen_key(), path)               # exclusive create: a second write is a bug


def test_load_key_rejects_exposed_perms(tmp_path):
    path = str(tmp_path / "publisher.key")
    mf.save_key(mf.gen_key(), path)
    os.chmod(path, 0o644)
    with pytest.raises(mf.ManifestError, match="group/world"):
        mf.load_key(path)


# ---- M9: Authorization must never cross an origin on redirect ---------------------------------------

def _redirect(from_url, to_url, headers=None):
    req = urllib.request.Request(from_url, headers=headers or {})
    h = F._SafeRedirectHandler()
    return h.redirect_request(req, None, 302, "Found", {}, to_url)


def test_cross_origin_redirect_strips_authorization():
    new = _redirect("https://huggingface.co/org/model/resolve/main/f.safetensors",
                    "https://evil.example.com/f.safetensors",
                    {"Authorization": "Bearer hf_secret", "User-Agent": "shard/1"})
    assert not new.has_header("Authorization"), "bearer leaked across origins"
    assert new.has_header("User-agent")                    # benign headers still carried


def test_same_origin_redirect_keeps_authorization():
    new = _redirect("https://huggingface.co/a", "https://huggingface.co/b",
                    {"Authorization": "Bearer hf_secret"})
    assert new.has_header("Authorization")                 # same origin: auth is fine


def test_insecure_redirect_refused():
    with pytest.raises(FetchError, match="insecure"):
        _redirect("https://huggingface.co/a", "http://huggingface.co/a")


def test_redirect_allowlist_enforced_when_set(monkeypatch):
    monkeypatch.setenv("SHARD_REDIRECT_ALLOW", "cdn-lfs.huggingface.co")
    new = _redirect("https://huggingface.co/a", "https://cdn-lfs.huggingface.co/b",
                    {"Authorization": "Bearer x"})
    assert not new.has_header("Authorization")             # allowed host, auth still stripped
    with pytest.raises(FetchError, match="SHARD_REDIRECT_ALLOW"):
        _redirect("https://huggingface.co/a", "https://elsewhere.example.com/b")


def test_engine_http_uses_hardened_opener():
    """MirrorProvider and publish_manifest both resolve HTTP through shard.fetch.urlopen,
    whose opener carries the auth-stripping handler — not urllib's module default."""
    assert any(isinstance(h, F._SafeRedirectHandler) for h in F._OPENER.handlers)
    import publish_manifest as pub
    assert pub._fetch is F


# ---- L1: a provider that lies about success must not wedge the chain --------------------------------

class _LyingProvider(F.Provider):
    """Returns 'success' without writing any file — e.g. a hostile/buggy seeder."""

    def fetch(self, shard, dest):
        return                                             # no file, no exception


def test_chain_falls_back_past_provider_with_no_file(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "shard.bin").write_bytes(b"good-bytes")
    sha, size = mf.sha256_file(str(src / "shard.bin"))
    shard = {"path": "shard.bin", "sha256": sha, "size": size, "shard_id": mf.cidv1_raw(sha)}
    chain = F.ChainProvider([_LyingProvider(), F.LocalDirProvider(str(src))])
    dest = str(tmp_path / "shard.bin")
    chain.fetch(shard, dest)                               # old code: FileNotFoundError escapes here
    assert os.path.exists(dest)
    got, _ = mf.sha256_file(dest)
    assert got == sha                                      # the honest mirror served it


def test_verify_missing_file_is_fetcherror(tmp_path):
    shard = {"path": "x.bin", "sha256": "00" * 32, "size": 4, "shard_id": mf.cidv1_raw("00" * 32)}
    with pytest.raises(FetchError, match="unreadable"):
        F._verify(str(tmp_path / "nope.bin"), shard)


# ---- M7 (publish): one immutable revision, metadata fetched once, coverage enforced -----------------

def _fake_hf(monkeypatch, repos):
    """Serve a fake HF: repos = {repo: {"sha": rev, "tree": [...], "files": {path: blob}}}.
    Any URL outside the pinned revisions (e.g. the old mutable /main fetches) is an error.
    Returns a log of index fetch counts per repo."""
    counts = {}

    def fake(req, timeout=None):
        url = req.full_url
        for repo, r in repos.items():
            if url == f"https://huggingface.co/api/models/{repo}":
                return io.BytesIO(json.dumps({"sha": r["sha"]}).encode())
            if url == f"https://huggingface.co/api/models/{repo}/tree/{r['sha']}?recursive=1":
                return io.BytesIO(json.dumps(r["tree"]).encode())
            prefix = f"https://huggingface.co/{repo}/resolve/{r['sha']}/"
            if url.startswith(prefix):
                path = url[len(prefix):]
                counts[(repo, path)] = counts.get((repo, path), 0) + 1
                blobs = r["files"][path]
                blob = blobs[min(counts[(repo, path)], len(blobs)) - 1] if isinstance(blobs, list) else blobs
                return io.BytesIO(blob)
        raise AssertionError(f"unexpected URL (mutable ref or wrong repo?): {url}")

    monkeypatch.setattr(F, "urlopen", fake)
    return counts


def test_publish_hf_pins_revision_and_fetches_index_once(monkeypatch):
    """The weight_map and the SIGNED index hash must come from the SAME single fetch at a
    pinned revision. The old code hit mutable /main twice: an upstream push in between
    signed index bytes whose weight_map differs from the one the manifest carries."""
    import publish_manifest as pub
    repo = "org/model"
    wm = {"model.layers.0.q.weight": "model-00001.safetensors"}
    idx_v1 = json.dumps({"weight_map": wm}).encode()
    idx_v2 = json.dumps({"weight_map": {"model.layers.0.q.weight": "model-00002.safetensors"}}).encode()
    cfg = json.dumps({"num_hidden_layers": 1, "architectures": ["X"]}).encode()
    counts = _fake_hf(monkeypatch, {repo: {"sha": "rev1", "tree": [
        {"type": "file", "path": "config.json", "size": len(cfg)},
        {"type": "file", "path": "model.safetensors.index.json", "size": len(idx_v1)},
        {"type": "file", "path": "model-00001.safetensors", "size": 8,
         "lfs": {"oid": "ab" * 32, "size": 8}},
        {"type": "file", "path": "tokenizer.json", "size": 2},
    ], "files": {"config.json": cfg,
                 "model.safetensors.index.json": [idx_v1, idx_v2],  # a 2nd fetch sees v2
                 "tokenizer.json": b"{}"}}})
    _, weight_map, shards = pub.build_from_hf(repo, repo)
    assert weight_map == wm
    idx = next(s for s in shards if s["path"] == "model.safetensors.index.json")
    assert idx["sha256"] == hashlib.sha256(idx_v1).hexdigest(), \
        "signed index bytes differ from the bytes the weight_map was parsed from"
    assert counts[(repo, "model.safetensors.index.json")] == 1, "index fetched more than once"
    for s in shards:                                       # per-shard immutable source
        assert s["repo"] == repo and s["revision"] == "rev1"


def test_publish_tokenizer_repo_encoded_per_shard(monkeypatch):
    import publish_manifest as pub
    repo, tok = "org/model", "org/tok"
    wm = {"model.layers.0.q.weight": "model-00001.safetensors"}
    idx = json.dumps({"weight_map": wm}).encode()
    cfg = json.dumps({"num_hidden_layers": 1, "architectures": ["X"]}).encode()
    _fake_hf(monkeypatch, {
        repo: {"sha": "rev1", "tree": [
            {"type": "file", "path": "config.json", "size": len(cfg)},
            {"type": "file", "path": "model.safetensors.index.json", "size": len(idx)},
            {"type": "file", "path": "model-00001.safetensors", "size": 8,
             "lfs": {"oid": "cd" * 32, "size": 8}},
            {"type": "file", "path": "tokenizer.json", "size": 4},   # base repo's own copy
        ], "files": {"config.json": cfg, "model.safetensors.index.json": idx,
                     "tokenizer.json": b"base"}},
        tok: {"sha": "trev", "tree": [
            {"type": "file", "path": "tokenizer.json", "size": 10},
        ], "files": {"tokenizer.json": b'{"tok": 42}'}},
    })
    _, _, shards = pub.build_from_hf(repo, tok)
    toks = [s for s in shards if s["kind"] == "tokenizer"]
    assert len(toks) == 1, "tokenizer override must not duplicate the base repo's copy"
    assert toks[0]["repo"] == tok and toks[0]["revision"] == "trev"
    assert toks[0]["sha256"] == hashlib.sha256(b'{"tok": 42}').hexdigest()


def test_publish_dir_refuses_index_referencing_missing_files(tmp_path):
    import publish_manifest as pub
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "model-00001.safetensors").write_bytes(b"w1")
    json.dump({"num_hidden_layers": 1, "architectures": ["X"]}, open(d / "config.json", "w"))
    json.dump({"weight_map": {"model.layers.0.q.weight": "model-00001.safetensors",
                              "model.layers.0.k.weight": "model-00099.safetensors"}},  # absent
              open(d / "model.safetensors.index.json", "w"))
    with pytest.raises(ValueError, match="no shard entry"):
        pub.build_from_dir(str(d))


# ---- M7 (fetch): source routing, map==index equality, stray-weight quarantine -----------------------

def _mk(src, path, data, kind="weights"):
    full = os.path.join(src, path)
    with open(full, "wb") as f:
        f.write(data)
    sha, size = mf.sha256_file(full)
    return {"path": path, "size": size, "sha256": sha, "shard_id": mf.cidv1_raw(sha), "kind": kind}


def _mini_repo(tmp_path, priv, index_map=None, weight_map=None):
    """A 1-file signed repo; index_map lets the on-disk index disagree with the manifest."""
    src = str(tmp_path / "src")
    os.makedirs(src, exist_ok=True)
    wm = weight_map if weight_map is not None else {
        "model.layers.0.q.weight": "model-00001.safetensors",
        "model.layers.1.q.weight": "model-00001.safetensors",
        "model.embed_tokens.weight": "model-00001.safetensors",
        "model.norm.weight": "model-00001.safetensors",
        "lm_head.weight": "model-00001.safetensors",
    }
    shards = [
        _mk(src, "model-00001.safetensors", b"W" * 64),
        _mk(src, "config.json", b"{}", "config"),
        _mk(src, "model.safetensors.index.json",
            json.dumps({"weight_map": wm if index_map is None else index_map}).encode(), "config"),
    ]
    manifest = {"schema": mf.SCHEMA, "model_id": "t", "layer_count": 2,
                "weight_map": wm, "shards": shards}
    return src, mf.sign_manifest(manifest, priv)


def test_mirror_routes_by_shard_source(tmp_path, monkeypatch):
    """A shard declaring repo/revision (e.g. a separate tokenizer repo) must resolve
    against ITS source, not the mirror's base repo — the old code always used base+path."""
    urls = []

    def fake(req, timeout=None):
        urls.append(req.full_url)
        return io.BytesIO(b"tokdata")

    monkeypatch.setattr(F, "urlopen", fake)
    prov = F.MirrorProvider("https://huggingface.co/org/model/resolve/rev1/")
    prov.fetch({"path": "tokenizer.json", "size": 7, "repo": "other/tok", "revision": "trev"},
               str(tmp_path / "tokenizer.json"))
    assert urls == ["https://huggingface.co/other/tok/resolve/trev/tokenizer.json"]
    urls.clear()                                           # no declared source -> legacy base+path
    prov.fetch({"path": "tokenizer.json", "size": 7}, str(tmp_path / "t2.json"))
    assert urls == ["https://huggingface.co/org/model/resolve/rev1/tokenizer.json"]


def test_fetch_rejects_index_disagreeing_with_signed_map(tmp_path):
    """A (validly signed) manifest whose bundled index carries a DIFFERENT weight_map than
    the manifest's own: the runtime loader trusts the on-disk index, so a file only the
    index references would load unverified. The pair must be refused."""
    from shard.fetch import LocalDirProvider, fetch_block
    priv = mf.gen_key()
    src, manifest = _mini_repo(tmp_path, priv,
                               index_map={"model.layers.0.q.weight": "model-00099.safetensors"})
    dest = str(tmp_path / "model")
    with pytest.raises(FetchError, match="weight_map"):
        fetch_block(manifest, dest, stage=0, nstages=1, role="stage",
                    provider=LocalDirProvider(src), expected_pubkey=mf.pub_b64(priv))
    assert not os.path.exists(os.path.join(dest, "model.safetensors.index.json"))


def test_fetch_rejects_map_referencing_unlisted_files(tmp_path):
    from shard.fetch import LocalDirProvider, fetch_block
    priv = mf.gen_key()
    wm = {"model.layers.0.q.weight": "model-00001.safetensors",
          "model.layers.1.q.weight": "ghost.safetensors"}          # no shard entry for this
    src, manifest = _mini_repo(tmp_path, priv, weight_map=wm, index_map=wm)
    with pytest.raises(FetchError, match="no shard entry"):
        fetch_block(manifest, str(tmp_path / "model"), stage=0, nstages=1, role="stage",
                    provider=LocalDirProvider(src), expected_pubkey=mf.pub_b64(priv))


def test_stray_weight_file_is_quarantined(tmp_path):
    """A stale safetensors already in the model dir but absent from the signed manifest
    must not be loadable by the runtime's index-driven loader."""
    from shard.fetch import LocalDirProvider, fetch_block
    priv = mf.gen_key()
    src, manifest = _mini_repo(tmp_path, priv)
    dest = str(tmp_path / "model")
    os.makedirs(dest)
    stale = os.path.join(dest, "model-00099.safetensors")
    with open(stale, "wb") as f:
        f.write(b"never-verified")
    fetch_block(manifest, dest, stage=0, nstages=1, role="stage",
                provider=LocalDirProvider(src), expected_pubkey=mf.pub_b64(priv))
    assert not os.path.exists(stale), "stray weight file left loadable"
    assert os.path.exists(stale + ".quarantined")
    assert os.path.exists(os.path.join(dest, "model-00001.safetensors"))   # the real one intact
