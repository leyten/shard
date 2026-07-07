"""Manifest publisher (phase0/publish_manifest.py) — the generate side of the weight trust root.

fetch.py VERIFIES a signed manifest (tested in test_fetch.py); publish_manifest GENERATES one by
hashing a checkpoint + resolving the weight_map. The two must agree: a manifest the publisher emits
must fetch+verify cleanly, and only the loaded files (canonical safetensors in the index, config,
tokenizer) go in it — no README/dupes/original-fork files, no traversal paths. build_from_dir is
CPU-testable directly; build_from_hf needs the HF API so it's exercised by the CLI, not here.

Run: python3 -m pytest tests/test_publish_manifest.py -q
"""
import json
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "phase0"))

from shard import manifest as mf                    # noqa: E402
from shard.fetch import LocalDirProvider, fetch_block  # noqa: E402
import publish_manifest as pub                       # noqa: E402


def _checkpoint(tmp_path, layers=4):
    """A synthetic 2-file checkpoint: layers 0-1+embed in file1, 2-3+norm+lm_head in file2, plus
    config/index/tokenizer and some files that MUST be skipped (README, a dupe safetensors, an
    original/ fork)."""
    d = str(tmp_path / "ckpt")
    os.makedirs(os.path.join(d, "original"), exist_ok=True)
    open(os.path.join(d, "model-00001.safetensors"), "wb").write(b"w1" + b"A" * 300)
    open(os.path.join(d, "model-00002.safetensors"), "wb").write(b"w2" + b"B" * 300)
    open(os.path.join(d, "dupe.safetensors"), "wb").write(b"dupe-not-in-index")   # stray -> skipped
    open(os.path.join(d, "original", "consolidated.safetensors"), "wb").write(b"fork")  # skipped dir
    open(os.path.join(d, "README.md"), "wb").write(b"# hi")                        # kind None -> skipped
    json.dump({"num_hidden_layers": layers, "architectures": ["MiniMaxText01ForCausalLM"],
               "tie_word_embeddings": False}, open(os.path.join(d, "config.json"), "w"))
    wm = {}
    for j in (0, 1):
        wm[f"model.layers.{j}.q.weight"] = "model-00001.safetensors"
    for j in (2, 3):
        wm[f"model.layers.{j}.q.weight"] = "model-00002.safetensors"
    wm["model.embed_tokens.weight"] = "model-00001.safetensors"
    wm["model.norm.weight"] = "model-00002.safetensors"
    wm["lm_head.weight"] = "model-00002.safetensors"
    json.dump({"weight_map": wm}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    json.dump({"tok": 1}, open(os.path.join(d, "tokenizer.json"), "w"))
    return d


def _manifest_from(cfg, weight_map, shards, priv, model_id="test/m2.5"):
    m = {"schema": mf.SCHEMA, "model_id": model_id, "arch": (cfg.get("architectures") or ["x"])[0],
         "layer_count": cfg["num_hidden_layers"], "tied_embeddings": bool(cfg.get("tie_word_embeddings")),
         "tokenizer": model_id, "weight_map": weight_map, "shards": shards}
    return mf.sign_manifest(m, priv)


# ---- 1. build_from_dir selects exactly the loaded files, with real hashes --------------------------

def test_build_from_dir_selects_only_loaded_files(tmp_path):
    d = _checkpoint(tmp_path)
    cfg, weight_map, shards = pub.build_from_dir(d)
    paths = {s["path"] for s in shards}
    assert "model-00001.safetensors" in paths and "model-00002.safetensors" in paths
    assert "config.json" in paths and "model.safetensors.index.json" in paths and "tokenizer.json" in paths
    assert "README.md" not in paths                          # not a loaded file
    assert "dupe.safetensors" not in paths                   # safetensors not in the index
    assert not any("original" in p for p in paths)           # fork dir skipped
    kinds = {s["path"]: s["kind"] for s in shards}
    assert kinds["model-00001.safetensors"] == "weights"
    assert kinds["config.json"] == "config" and kinds["model.safetensors.index.json"] == "config"
    assert kinds["tokenizer.json"] == "tokenizer"


def test_build_from_dir_hashes_match_content(tmp_path):
    d = _checkpoint(tmp_path)
    _, _, shards = pub.build_from_dir(d)
    for s in shards:
        sha, size = mf.sha256_file(os.path.join(d, s["path"]))
        assert s["sha256"] == sha and s["size"] == size and s["shard_id"] == mf.cidv1_raw(sha)


# ---- 2. the generated manifest fetch+verifies cleanly (generate <-> verify agree) ------------------

def test_generated_manifest_roundtrips_through_fetch(tmp_path):
    d = _checkpoint(tmp_path)
    priv = mf.gen_key()
    manifest = _manifest_from(*pub.build_from_dir(d), priv)
    mf.verify_manifest(manifest, expected_pubkey=mf.pub_b64(priv))    # signs + self-verifies
    # a node fetches its block from the checkpoint-as-mirror and every byte re-verifies
    for stage in (0, 1):
        paths = fetch_block(manifest, str(tmp_path / f"node{stage}"), stage=stage, nstages=2,
                            role="stage", provider=LocalDirProvider(d), expected_pubkey=mf.pub_b64(priv))
        assert paths and all(os.path.exists(p) for p in paths)
    # head pulls its file, tail pulls the other — the weight_map selection is honored end to end
    head = {os.path.basename(p) for p in fetch_block(manifest, str(tmp_path / "h"), stage=0, nstages=2,
            role="stage", provider=LocalDirProvider(d), expected_pubkey=mf.pub_b64(priv))}
    assert "model-00001.safetensors" in head and "model-00002.safetensors" not in head


def test_kind_classification():
    assert pub._kind("model-00001.safetensors") == "weights"
    assert pub._kind("tokenizer.json") == "tokenizer" and pub._kind("config.json") == "config"
    assert pub._kind("model.safetensors.index.json") == "config"
    assert pub._kind("README.md") is None and pub._kind(".gitattributes") is None
