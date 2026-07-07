"""Verified per-node weight pull for the scattered ring — the trust-root deploy path.

Same selective per-stage pull as m25_pull_range (a node fetches ONLY the shards covering its layer
range [lo,hi), plus the head/tail boundary weights + tokenizer for the coordinator), but every byte
is re-hashed against a SIGNED, content-addressed manifest before it touches disk
(shard.fetch.fetch_block_range). A malicious mirror or peer physically cannot feed corrupt weights —
the sha256 won't match the manifest and the load fails closed. Drop-in for m25_pull_range: same
--lo/--hi/--head/--tail, plus --manifest and the catalog-pinned --pubkey.

  stage: python m25_pull_verified.py --lo 13 --hi 26 --manifest m25_manifest.json --pubkey <b64>
  head:  python m25_pull_verified.py --lo 0  --hi 13 --head --manifest ... --pubkey ...
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shard.fetch import MirrorProvider, fetch_block_range  # noqa: E402


def main(a):
    token = ""
    if os.path.exists("/root/.hf_token"):
        token = open("/root/.hf_token").read().strip()
    token = token or os.environ.get("HF_TOKEN", "")
    manifest = json.load(open(a.manifest))
    # the manifest's model_id is the source of truth for the repo; --repo only overrides the mirror URL
    repo = a.repo or manifest.get("model_id")
    provider = MirrorProvider(f"https://huggingface.co/{repo}/resolve/main/",
                              headers={"Authorization": f"Bearer {token}"} if token else None)
    role = "coordinator" if a.head else "stage"
    paths = fetch_block_range(manifest, a.dir, a.lo, a.hi, is_head=a.head, is_tail=a.tail, role=role,
                              provider=provider, expected_pubkey=a.pubkey,
                              tied=bool(manifest.get("tied_embeddings", False)))
    print(f"VERIFIED_PULL_DONE {len(paths)} files in {a.dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--head", action="store_true")           # pull embed + tokenizer (stage 0 + coord)
    ap.add_argument("--tail", action="store_true")           # pull norm + lm_head (last stage)
    ap.add_argument("--repo", default=None, help="mirror repo (default: manifest model_id)")
    ap.add_argument("--manifest", required=True, help="signed shard-manifest/1 json")
    ap.add_argument("--pubkey", default=None,
                    help="catalog-pinned publisher pubkey (b64); omit to skip the pin (sig still checked)")
    ap.add_argument("--dir", default="/root/m25")
    main(ap.parse_args())
