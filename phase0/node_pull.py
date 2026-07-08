"""Pull this node's verified weight block (step 3, JOIN — node side).

A node runs this right before pipeline.load_stage. Given a manifest reference, the
catalog-pinned publisher pubkey, and its stage/role, it fetches ONLY its layer block's
shards from the mirror, verifies every byte, and writes them into model_dir — which is
then exactly what load_stage(model_dir, stage, nstages) expects. This is the verified,
selective replacement for the trust-the-download paths (get_model.py / node_fetch.py).

  # tail stage (loads layers + norm + lm_head)
  python node_pull.py --manifest manifest.json --pubkey <b64> \
      --model-dir /root/models/gpt-oss-120b --stage 3 --nstages 4 --role stage
  # coordinator/head (also pulls the tokenizer)
  python node_pull.py --manifest manifest.json --pubkey <b64> \
      --model-dir /root/models/gpt-oss-120b --stage 0 --nstages 4 --role coordinator

--manifest takes a URL or a local path. --pubkey is the pinned key (or @file); omit only
in a trusted lab — without it a mirror could serve its own validly-signed manifest.
"""
import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shard import fetch  # noqa: E402


def load_manifest(ref):
    if ref.startswith(("http://", "https://")):
        req = urllib.request.Request(ref, headers={"User-Agent": "shard/1"})
        return json.load(urllib.request.urlopen(req, timeout=60))
    with open(ref) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest URL or local path")
    ap.add_argument("--pubkey", help="pinned publisher pubkey (base64, or @file)")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--nstages", type=int, required=True)
    ap.add_argument("--role", choices=["stage", "coordinator"], default="stage")
    ap.add_argument("--base-url", help="mirror base (default: HF resolve URL for model_id)")
    ap.add_argument("--source", choices=["mirror", "libp2p"], default="mirror",
                    help="libp2p exercises the step-8 seam (falls back to mirror)")
    a = ap.parse_args()

    manifest = load_manifest(a.manifest)
    pubkey = a.pubkey
    if pubkey and pubkey.startswith("@"):
        pubkey = open(pubkey[1:]).read().strip()
    if not pubkey:
        print("[node_pull] WARNING: no --pubkey pin; trusting the manifest's own key", file=sys.stderr)

    base = a.base_url or f"https://huggingface.co/{manifest['model_id']}/resolve/main/"
    provider = fetch.MirrorProvider(base)
    if a.source == "libp2p":
        # torrent-first: peers on the shard DHT (SHARD_DHT_BOOTSTRAP / SHARD_SIDECAR env),
        # mirror as the origin fallback. Verification is identical either way.
        provider = fetch.ChainProvider([fetch.Libp2pProvider(), provider])

    paths = fetch.fetch_block(
        manifest, a.model_dir, stage=a.stage, nstages=a.nstages, role=a.role,
        provider=provider, expected_pubkey=pubkey,
        tied=bool(manifest.get("tied_embeddings", False)),
    )
    print(f"[node_pull] ready: {len(paths)} verified files in {a.model_dir}")


if __name__ == "__main__":
    main()
