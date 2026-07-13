"""Publish a signed, content-addressed weight manifest (step 3, JOIN — publisher side).

Builds a `shard-manifest/1` (see shard/manifest.py) over a model checkpoint and signs it
with a publisher ed25519 key. Two sources:

  --hf REPO    build from Hugging Face metadata WITHOUT downloading the weights. HF stores
               every LFS file's oid == sha256 of its content, so the multi-GB safetensors
               are hashed for free; only the small config/index/tokenizer files are
               fetched (KB–MB) to hash. This makes a real 120B manifest in seconds.
  --dir PATH   build from a local checkpoint by hashing each file on disk.

c0mpute's MODEL_CATALOG stores a pointer to the emitted manifest + the publisher pubkey
this prints; nodes pin that pubkey when they verify (shard/manifest.verify_manifest).

  python publish_manifest.py --hf openai/gpt-oss-120b --key keys/publisher.key --out manifest.json
  python publish_manifest.py --dir /root/models/gpt-oss-120b --key keys/publisher.key --out manifest.json
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shard import manifest as mf  # noqa: E402

TOKENIZER_FILES = {
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json",
    "merges.txt", "special_tokens_map.json", "chat_template.jinja", "added_tokens.json",
}
CONFIG_FILES = {"config.json", "generation_config.json"}
SKIP_DIRS = ("original/", "metal/", "onnx/", "gguf/")


def _hdr():
    tok = os.environ.get("HF_TOKEN", "")
    h = {"User-Agent": "shard-publish/1"}
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _hf_get(repo, path):
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=_hdr()), timeout=60).read()


def _hf_tree(repo):
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=1"
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=_hdr()), timeout=60))


def _kind(path):
    base = os.path.basename(path)
    if path.endswith(".safetensors"):
        return "weights"
    if base in TOKENIZER_FILES:
        return "tokenizer"
    if base in CONFIG_FILES or base.endswith(".index.json"):
        return "config"
    return None  # README, .gitattributes, bf16 dupes, etc. — not loaded


def build_from_hf(repo, tokenizer_repo):
    entries = {e["path"]: e for e in _hf_tree(repo)
               if e.get("type") == "file" and not any(e["path"].startswith(d) for d in SKIP_DIRS)}
    cfg = json.loads(_hf_get(repo, "config.json"))
    index = json.loads(_hf_get(repo, "model.safetensors.index.json"))
    weight_map = index["weight_map"]
    wanted_weights = set(weight_map.values())  # only canonical safetensors, not dupes

    # gpt-oss ships its own tokenizer; some quant repos borrow the base repo's — allow override.
    tok_entries = {}
    if tokenizer_repo and tokenizer_repo != repo:
        for e in _hf_tree(tokenizer_repo):
            if e.get("type") == "file" and os.path.basename(e["path"]) in TOKENIZER_FILES:
                tok_entries[e["path"]] = ("__TOK__", e)

    shards = []
    for path, e in sorted(entries.items()):
        kind = _kind(path)
        if kind is None:
            continue
        if kind == "weights" and path not in wanted_weights:
            continue  # a stray safetensors not in the index (e.g. a duplicate) — skip
        lfs = e.get("lfs")
        if lfs and lfs.get("oid"):
            sha, size = lfs["oid"], int(lfs.get("size", e["size"]))  # oid == sha256, free
        else:
            blob = _hf_get(repo, path)  # small file — hash it (KB–MB)
            sha, size = hashlib.sha256(blob).hexdigest(), len(blob)
        shards.append({"shard_id": mf.cidv1_raw(sha), "path": path,
                       "sha256": sha, "size": size, "kind": kind})

    for path, (_, e) in sorted(tok_entries.items()):
        blob = urllib.request.urlopen(
            urllib.request.Request(f"https://huggingface.co/{tokenizer_repo}/resolve/main/{path}",
                                   headers=_hdr()), timeout=60).read()
        sha = hashlib.sha256(blob).hexdigest()
        shards.append({"shard_id": mf.cidv1_raw(sha), "path": path,
                       "sha256": sha, "size": len(blob), "kind": "tokenizer"})

    return cfg, weight_map, shards


def build_from_dir(path):
    cfg = json.load(open(os.path.join(path, "config.json")))
    index = json.load(open(os.path.join(path, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    wanted_weights = set(weight_map.values())
    shards = []
    for root, _, files in os.walk(path):
        if any(seg in root[len(path):] for seg in ("original", "metal", "onnx", "gguf")):
            continue
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, path)
            kind = _kind(rel)
            if kind is None or (kind == "weights" and rel not in wanted_weights):
                continue
            sha, size = mf.sha256_file(full)
            shards.append({"shard_id": mf.cidv1_raw(sha), "path": rel,
                           "sha256": sha, "size": size, "kind": kind})
    return cfg, weight_map, shards


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", metavar="REPO", help="build from HF metadata (no weight download)")
    src.add_argument("--dir", metavar="PATH", help="build from a local checkpoint dir")
    ap.add_argument("--model-id", help="manifest model_id (default: the repo/dir name)")
    ap.add_argument("--tokenizer-repo", help="repo holding tokenizer files (default: same as --hf)")
    ap.add_argument("--key", required=True, help="publisher ed25519 key (created if absent)")
    ap.add_argument("--out", required=True, help="where to write the signed manifest json")
    a = ap.parse_args()

    repo = a.hf or a.dir
    model_id = a.model_id or (a.hf or os.path.basename(os.path.normpath(a.dir)))
    if a.hf:
        cfg, weight_map, shards = build_from_hf(a.hf, a.tokenizer_repo or a.hf)
    else:
        cfg, weight_map, shards = build_from_dir(a.dir)

    for s in shards:                                  # never sign a traversal/absolute path
        p = s["path"]
        if os.path.isabs(p) or os.path.splitdrive(p)[0] or ".." in p.replace("\\", "/").split("/"):
            sys.exit(f"refusing to sign unsafe shard path: {p!r}")

    arch = (cfg.get("architectures") or ["unknown"])[0]
    manifest = {
        "schema": mf.SCHEMA,
        "model_id": model_id,
        "arch": arch,
        "layer_count": cfg["num_hidden_layers"],
        "tied_embeddings": bool(cfg.get("tie_word_embeddings", False)),
        "tokenizer": a.tokenizer_repo or repo,
        "weight_map": weight_map,
        "shards": shards,
    }

    if os.path.exists(a.key):
        priv = mf.load_key(a.key)
    else:
        os.makedirs(os.path.dirname(a.key) or ".", mode=0o700, exist_ok=True)
        priv = mf.gen_key()
        mf.save_key(priv, a.key)
        print(f"generated new publisher key -> {a.key}", file=sys.stderr)
    signed = mf.sign_manifest(manifest, priv)

    with open(a.out, "w") as f:
        json.dump(signed, f, indent=0)

    weights = [s for s in shards if s["kind"] == "weights"]
    total = sum(s["size"] for s in shards)
    mf.verify_manifest(signed, expected_pubkey=mf.pub_b64(priv))  # self-check, fail closed
    print(json.dumps({
        "model_id": model_id, "arch": arch, "layer_count": cfg["num_hidden_layers"],
        "shards": len(shards), "weights_shards": len(weights),
        "total_gb": round(total / 1e9, 2),
        "publisher_pubkey": mf.pub_b64(priv), "out": a.out,
    }, indent=2))


if __name__ == "__main__":
    main()
