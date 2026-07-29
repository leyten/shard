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
    dec = pub.decoder_config(cfg)                      # same resolution main() uses
    m = {"schema": mf.SCHEMA, "model_id": model_id, "arch": (cfg.get("architectures") or ["x"])[0],
         "layer_count": dec["num_hidden_layers"], "tied_embeddings": bool(dec.get("tie_word_embeddings")),
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
    # a tiktoken vocab IS the tokenizer for the models that ship one (Kimi-K3 has no tokenizer.json)
    assert pub._kind("tiktoken.model") == "tokenizer"
    # K3's tokenizer is custom remote code: the two chained tokenizer .py files travel with the
    # checkpoint (the coordinator loads them via trust_remote_code), so they are tokenizer-kind ...
    assert pub._kind("tokenization_kimi.py") == "tokenizer"
    assert pub._kind("encoding_k3.py") == "tokenizer"
    # ... but the MODELING/config remote code is loaded from the vendored kimi_k3_ref, never fetched
    assert pub._kind("modeling_kimi_linear.py") is None
    assert pub._kind("configuration_kimi_k3.py") is None


# ---- 3. a multimodal checkpoint: nested depth, namespaced decoder, a second tower -----------------

def _multimodal_checkpoint(tmp_path, layers=4):
    """A Kimi-K3-shaped checkpoint at toy scale: the decoder is namespaced under
    `language_model.model.*`, its depth lives ONLY in config.json's `text_config`, the boundary
    tensors share one file, and a vision tower numbers its own `blocks.<n>.`."""
    d = str(tmp_path / "mm")
    os.makedirs(d, exist_ok=True)
    for i in (1, 2, 3, 4):
        open(os.path.join(d, f"model-0000{i}.safetensors"), "wb").write(b"f%d" % i + b"Z" * 200)
    json.dump({"architectures": ["KimiK3ForConditionalGeneration"], "model_type": "kimi_k3",
               "tie_word_embeddings": False, "vision_config": {"patch_size": 14},
               "text_config": {"architectures": ["KimiLinearForCausalLM"], "model_type": "kimi_linear",
                               "num_hidden_layers": layers, "tie_word_embeddings": False}},
              open(os.path.join(d, "config.json"), "w"))
    wm = {}
    for j in range(layers):                                  # one file per half of the decoder
        wm[f"language_model.model.layers.{j}.q.weight"] = f"model-0000{1 if j < layers // 2 else 2}.safetensors"
    for t in ("model.embed_tokens", "model.norm", "lm_head", "model.output_attn_res_proj"):
        wm[f"language_model.{t}.weight"] = "model-00003.safetensors"
    wm["vision_tower.encoder.blocks.0.wo.weight"] = "model-00004.safetensors"
    wm["mm_projector.proj.0.weight"] = "model-00004.safetensors"
    json.dump({"weight_map": wm}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    open(os.path.join(d, "tiktoken.model"), "wb").write(b"vocab")
    # K3's tokenizer is CUSTOM remote code, not a tokenizer.json: tokenizer_config's auto_map ->
    # tokenization_kimi (TikTokenTokenizer), which imports encoding_k3. All of these must ride in the
    # manifest or the coordinator cannot tokenize (live K3 ring 2026-07-29). configuration_k3.py /
    # special_tokens_map.json / tokenizer.json do NOT exist for K3, so they are absent here on purpose.
    json.dump({"tokenizer_class": "TikTokenTokenizer",
               "auto_map": {"AutoTokenizer": ["tokenization_kimi.TikTokenTokenizer", None]}},
              open(os.path.join(d, "tokenizer_config.json"), "w"))
    open(os.path.join(d, "tokenization_kimi.py"), "wb").write(b"from .encoding_k3 import build_chat_segments\n")
    open(os.path.join(d, "encoding_k3.py"), "wb").write(b"OPEN_TOKEN = '<|open|>'\n")
    # NON-tokenizer remote code that must NOT be pulled (loaded from the vendored kimi_k3_ref instead)
    open(os.path.join(d, "modeling_kimi_linear.py"), "wb").write(b"class KimiLinearForCausalLM: pass\n")
    return d


def test_layer_count_comes_from_the_nested_text_config():
    """A multimodal config.json has no top-level `num_hidden_layers` — reading it there raised
    KeyError and no such checkpoint could be published at all. layer_count is what receipt
    coverage tiles against, so it must be the DECODER's depth."""
    cfg = {"architectures": ["KimiK3ForConditionalGeneration"], "tie_word_embeddings": False,
           "text_config": {"num_hidden_layers": 93, "tie_word_embeddings": False}}
    assert "num_hidden_layers" not in cfg
    assert pub.decoder_config(cfg)["num_hidden_layers"] == 93
    # a plain single-stack config resolves to itself, unchanged ...
    flat = {"num_hidden_layers": 62}
    assert pub.decoder_config(flat) is flat
    # ... and so does one whose text_config is a partial override carrying no depth.
    partial = {"num_hidden_layers": 62, "text_config": {"rope_theta": 1e6}}
    assert pub.decoder_config(partial) is partial


def test_multimodal_checkpoint_publishes_and_fetches_the_text_stack(tmp_path):
    """End to end on the K3 shape: the published manifest carries the decoder's depth, the fetch
    resolves the `language_model.model.layers.<n>.` namespace, head/tail pull the shared boundary
    file — and the vision tower is pulled by NOBODY."""
    d = _multimodal_checkpoint(tmp_path)
    priv = mf.gen_key()
    cfg, weight_map, shards = pub.build_from_dir(d)
    manifest = _manifest_from(cfg, weight_map, shards, priv, model_id="test/k3")
    assert manifest["layer_count"] == 4                       # from text_config, not the wrapper
    # ALL of K3's tokenizer files — the tiktoken vocab AND the two chained remote-code .py files —
    # are selected as tokenizer shards; the modeling remote code is not in the manifest at all.
    assert {s["path"] for s in shards if s["kind"] == "tokenizer"} == {
        "tiktoken.model", "tokenizer_config.json", "tokenization_kimi.py", "encoding_k3.py"}
    assert "modeling_kimi_linear.py" not in {s["path"] for s in shards}

    def files(stage, nstages, role="stage"):
        return {os.path.basename(p) for p in fetch_block(
            manifest, str(tmp_path / f"n{stage}{nstages}{role}"), stage=stage, nstages=nstages,
            role=role, provider=LocalDirProvider(d), expected_pubkey=mf.pub_b64(priv))}

    head, tail = files(0, 2), files(1, 2)
    # the HEAD pull carries the whole custom tokenizer (it hosts the coordinator); a middle/tail does not
    assert {"tiktoken.model", "tokenizer_config.json", "tokenization_kimi.py", "encoding_k3.py"} <= head
    assert not ({"tokenization_kimi.py", "encoding_k3.py"} & tail)
    assert "model-00001.safetensors" in head and "model-00002.safetensors" not in head
    assert "model-00002.safetensors" in tail and "model-00001.safetensors" not in tail
    assert "model-00003.safetensors" in head and "model-00003.safetensors" in tail  # embed | norm+head
    # The vision tower and the projector are in no stage's block. That holds because this
    # checkpoint numbers them `blocks.<n>.`: a second tower numbered `layers.<n>.` would be
    # attributed to the decoder's ranges and over-pulled (the documented shard.weightkeys tradeoff
    # — extra bytes are survivable, a missing weight behind a valid receipt is not).
    assert all("model-00004.safetensors" not in f for f in (head, tail))
