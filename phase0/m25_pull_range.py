"""Selective per-node weight pull for the scattered ring: a stage node pulls ONLY the shards
covering its layer range [lo,hi); the head/coord node additionally pulls the boundary tensors
(embed_tokens, final norm, lm_head) + tokenizer/chat-template. ~30GB/node instead of 144GB.

  stage node:  python m25_pull_range.py --lo 0  --hi 13
  head/coord:  python m25_pull_range.py --lo 0  --hi 13 --coord
"""
import os, re, json, argparse


def _hf_token_env():
    """HF auth, portable: an already-set HF_TOKEN wins; else ~/.hf_token (the box convention —
    /root/.hf_token on vast, which this used to hardcode); else leave auth to huggingface_hub's
    own login chain (public/cached repos need none)."""
    if not os.environ.get("HF_TOKEN"):
        p = os.path.expanduser("~/.hf_token")
        if os.path.isfile(p):
            os.environ["HF_TOKEN"] = open(p).read().strip()


def main(lo, hi, head, tail, repo, DIR):
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    _hf_token_env()
    from huggingface_hub import snapshot_download, hf_hub_download
    for f in ["config.json", "hf_quant_config.json", "model.safetensors.index.json", "generation_config.json"]:
        hf_hub_download(repo, f, local_dir=DIR)
    idx = json.load(open(f"{DIR}/model.safetensors.index.json"))["weight_map"]
    need = set(range(lo, hi))
    shards = set()
    for k, s in idx.items():
        m = re.search(r"model\.layers\.(\d+)\.", k)
        if m and int(m.group(1)) in need:
            shards.add(s)
        if head and "embed_tokens" in k:                       # m25_pipe head embeds
            shards.add(s)
        if tail and (k == "model.norm.weight" or k.startswith("lm_head")):  # m25_pipe tail does norm+lm_head
            shards.add(s)
    patterns = sorted(shards)
    if head:                                                   # coord (on head box) needs the tokenizer/chat template
        patterns += ["tokenizer*", "*.jinja", "special_tokens*", "chat_template*"]
    print(f"pulling {len(shards)} shards for layers [{lo},{hi}){' +head' if head else ''}{' +tail' if tail else ''}", flush=True)
    snapshot_download(repo, local_dir=DIR, allow_patterns=patterns)
    print(f"RANGE_PULL_DONE {len(shards)} shards", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--head", action="store_true")            # pull embed + tokenizer (m25_pipe stage 0 + coord)
    ap.add_argument("--tail", action="store_true")            # pull norm + lm_head (m25_pipe last stage)
    ap.add_argument("--repo", default="nvidia/MiniMax-M2.5-NVFP4")
    ap.add_argument("--dir", default="/root/m25")
    a = ap.parse_args()
    main(a.lo, a.hi, a.head, a.tail, a.repo, a.dir)
