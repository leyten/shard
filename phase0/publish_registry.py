"""Publish (sign) the single model registry — shard-models/1 (M1, publisher side).

Reads an UNSIGNED registry json (the human-edited source: schema + version + models[]),
signs it with the publisher ed25519 key (reusing shard/manifest.py's key path — the SAME
key that signs weight manifests), and writes the signed registry that both repos verify.

  # sign the source registry, creating the key if absent
  python phase0/publish_registry.py --in models/models.src.json --key keys/publisher.key --out models/models.json

  # verify an already-signed registry against the pinned pubkey
  python phase0/publish_registry.py --verify models/models.json --pubkey <base64>

The emitted pubkey is what c0mpute pins (lib/orchestrator/modelRegistry.ts MODELS_PUBKEY)
and what shard consumers pass as expected_pubkey. Keep the .key file secret.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shard import manifest as mf       # noqa: E402  — key gen/load + pub_b64
from shard import registry as reg      # noqa: E402  — schema + validate + verify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", help="unsigned source registry json to sign")
    ap.add_argument("--key", help="publisher ed25519 key (created if absent)")
    ap.add_argument("--out", help="where to write the signed registry json")
    ap.add_argument("--verify", help="instead of signing: verify this signed registry")
    ap.add_argument("--pubkey", help="expected publisher pubkey (base64) for --verify")
    a = ap.parse_args()

    if a.verify:
        r = reg.load_registry(a.verify, expected_pubkey=a.pubkey, verify=True)
        ids = [m["id"] for m in r["models"]]
        print(json.dumps({"ok": True, "schema": r["schema"], "version": r.get("version"),
                          "models": ids, "publisher_pubkey": r["publisher_pubkey"]}, indent=2))
        return

    if not (a.src and a.key and a.out):
        ap.error("signing requires --in, --key, --out")

    with open(a.src) as f:
        registry = json.load(f)

    # structural check BEFORE signing — never sign a malformed registry (fail closed early).
    registry.pop("signature", None)
    registry.pop("publisher_pubkey", None)
    if registry.get("schema") != reg.SCHEMA:
        sys.exit(f"source schema must be {reg.SCHEMA!r}, got {registry.get('schema')!r}")
    reg._validate_rows(registry)

    if os.path.exists(a.key):
        priv = mf.load_key(a.key)
    else:
        os.makedirs(os.path.dirname(a.key) or ".", exist_ok=True)
        priv = mf.gen_key()
        mf.save_key(priv, a.key)
        print(f"generated new publisher key -> {a.key}", file=sys.stderr)

    signed = reg.sign_registry(registry, priv)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(signed, f, indent=2)

    # self-check: re-verify what we just wrote, fail closed.
    reg.verify_registry(signed, expected_pubkey=mf.pub_b64(priv))
    print(json.dumps({
        "ok": True, "out": a.out, "version": signed.get("version"),
        "models": [m["id"] for m in signed["models"]],
        "publisher_pubkey": mf.pub_b64(priv),
    }, indent=2))


if __name__ == "__main__":
    main()
