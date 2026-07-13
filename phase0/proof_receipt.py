"""Run-receipt generator + verifier for a Shard swarm run (see docs/PROOF.md).

Produces a JSON receipt of the run's claims — distinct distributed nodes (ip/geo/gpu), real WAN
edge latencies, output token hash, and the commit to re-run — and verifies an existing receipt
against the skeptic checklist.

ATTESTATION HONESTY (M6): every input here is SELF-REPORTED by whoever ran the swarm — node
identity, topology, WAN latencies, model, commit and perf are all unsigned. verify() therefore
attests INTERNAL CONSISTENCY only, never that the run was distributed/correct/reproducible.
The v2 receipt labels itself accordingly and carries an `envelope` binding the raw artifact
hashes, layer assignments, code/model identity and stage receipts; until that envelope is
SIGNED (signature is None today), the verdict must not claim more.

  build:  python proof_receipt.py build --nodes nodes.json --edges edges.json \
                 --run run.json --model gpt-oss-120b --quant mxfp4 --out docs/receipts/<id>.json
  verify: python proof_receipt.py verify docs/receipts/<id>.json
          (--ref-tokens ref.json to confirm the output matches a reference decode bit-for-bit)

inputs (collected during a real run):
  nodes.json  [{role, layer_range, public_ip, geo, gpu_uuid, gpu_name}, ...]
  edges.json  [{from, to, rtt_ms}, ...]      (from phase0/mesh.py over the live transport)
  run.json    {prompt, output_text, output_token_ids, tok_s_warm}
"""
import argparse, hashlib, json, subprocess, sys


def _sha(token_ids):
    return hashlib.sha256(json.dumps(list(token_ids)).encode()).hexdigest()


def _commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(args):
    nodes = json.load(open(args.nodes))
    edges = json.load(open(args.edges)) if args.edges else []
    run = json.load(open(args.run))
    receipt = {
        "format": "shard-proof-receipt/v2",
        # nothing in this record is signed — verify() checks internal consistency only
        "attestation": "self-reported",
        "run_id": args.run_id, "utc": args.utc, "shard_commit": _commit(),
        "model": args.model, "quant": args.quant,
        "prompt": run["prompt"], "output_text": run.get("output_text", ""),
        "output_token_ids": run["output_token_ids"],
        "output_sha256": _sha(run["output_token_ids"]),
        "tok_s_warm": run.get("tok_s_warm"), "decode": "greedy (exact)",
        "nodes": nodes, "edges": edges,
        "reference": {"source": run.get("reference_source", "single-node decode"),
                      "tokens_match": run.get("tokens_match")},
        # what a SIGNED receipt must bind before the verdict may claim more than
        # self-consistency: raw artifacts, layer assignments, code+model identity,
        # fresh signed stage receipts, an independently-supplied reference output.
        "envelope": {
            "artifact_sha256": {"nodes": _file_sha(args.nodes),
                                "edges": _file_sha(args.edges) if args.edges else None,
                                "run": _file_sha(args.run)},
            "assignments": json.load(open(args.assignments)) if args.assignments else None,
            "stage_receipts": json.load(open(args.stage_receipts)) if args.stage_receipts else None,
            "code_identity": {"shard_commit": _commit()},
            "model_identity": {"model": args.model, "quant": args.quant},
            "reference_source": run.get("reference_source", "single-node decode"),
            "signature": None,           # unsigned -> attestation stays "self-reported"
        },
    }
    json.dump(receipt, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}  ({len(nodes)} nodes, {len(edges)} edges, sha {receipt['output_sha256'][:12]}, self-reported)")


def verify(args):
    r = json.load(open(args.receipt))
    nodes, edges = r["nodes"], r.get("edges", [])
    checks = []
    # 1. distinct distributed machines
    ips = [n["public_ip"] for n in nodes]
    gpus = [n["gpu_uuid"] for n in nodes]
    checks.append(("distinct public IPs", len(set(ips)) == len(ips) and len(ips) > 1, f"{len(set(ips))}/{len(ips)} unique"))
    checks.append(("distinct GPU UUIDs", len(set(gpus)) == len(gpus) and len(gpus) > 1, f"{len(set(gpus))}/{len(gpus)} unique"))
    checks.append(("multiple regions", len({n["geo"] for n in nodes}) > 1, f"{sorted({n['geo'] for n in nodes})}"))
    # 2. real WAN, not localhost
    rtts = [e["rtt_ms"] for e in edges]
    checks.append(("edges are WAN-scale (>1ms)", bool(rtts) and min(rtts) > 1.0, f"min {min(rtts):.1f}ms max {max(rtts):.1f}ms" if rtts else "no edges"))
    # 3. output hash integrity
    checks.append(("output hash matches token ids", _sha(r["output_token_ids"]) == r["output_sha256"], r["output_sha256"][:12]))
    # 4. (optional) reference match — bit-for-bit reproducibility
    if args.ref_tokens:
        ref = json.load(open(args.ref_tokens))
        checks.append(("matches reference decode", list(ref) == list(r["output_token_ids"]), f"{len(ref)} ref tokens"))
    elif r.get("reference", {}).get("tokens_match") is not None:
        checks.append(("reference match (claimed)", r["reference"]["tokens_match"] is True, "pass --ref-tokens to re-verify"))

    attestation = r.get("attestation", "self-reported")   # v1 receipts carry no label -> self-reported
    print(f"=== receipt {r['run_id']} | {r['model']} {r.get('quant')} | {r.get('tok_s_warm')} tok/s"
          f" | commit {r['shard_commit'][:12]} | attestation: {attestation} ===")
    ok = True
    for name, passed, detail in checks:
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:32s} {detail}")
    if not ok:
        print("VERDICT: RECEIPT FAILED a check — see above.")
    else:
        # never overclaim: node identity/topology/WAN/model/commit/perf are all unsigned, so a
        # passing checklist proves the record agrees with ITSELF — nothing about the world.
        ref = ("output matches the independently-supplied reference tokens; everything else is "
               if args.ref_tokens else "every claim is ")
        print("VERDICT: RECEIPT SELF-CONSISTENT — " + ref + "SELF-REPORTED (unsigned). "
              "This attests internal consistency only, NOT that the run was distributed, "
              "correct, or reproducible; a signed envelope is required for that claim.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--nodes", required=True); b.add_argument("--edges"); b.add_argument("--run", required=True)
    b.add_argument("--model", required=True); b.add_argument("--quant", default=""); b.add_argument("--out", required=True)
    b.add_argument("--run-id", default="run"); b.add_argument("--utc", default="")
    b.add_argument("--assignments", help="JSON {pubkey_b64: [lo, hi]} layer-assignment map to bind")
    b.add_argument("--stage-receipts", help="JSON list of fresh signed stage receipts to bind")
    v = sub.add_parser("verify"); v.add_argument("receipt"); v.add_argument("--ref-tokens")
    a = ap.parse_args()
    (build if a.cmd == "build" else verify)(a)
