"""Settlement seam — the check c0mpute runs before per-shard-per-token pay.

The coordinator hands c0mpute one signed receipt per stage on job:complete. Before crediting any
node, c0mpute must confirm the set is trustworthy: every receipt verifies against its node key, the
blocks tile the whole model with no gap/overlap, each carries the coordinator's per-job freshness
nonce (no replay), and — on the lossless wire — the activation chain is unbroken. That is exactly
`shard.receipt.verify_coverage`; this exposes it as a JSON CLI so the control plane runs the ONE
crypto implementation instead of re-porting ed25519 + coverage into TypeScript.

On success it returns the per-stage split — `{pubkey, lo, hi, layers}` — the metering fan-out needs
to attribute the job's tokens across the stages (each node earns for the layers its shard produced).
Pure engine (boundary law): receipts + keys in, a verdict out; nothing about accounts or $. Deps
point one way (c0mpute -> shard) over stdio, same as shard.plan.
"""
import json
import sys

from .receipt import ReceiptError, verify_coverage


def _stages(receipts):
    return sorted(
        ({"pubkey": r["pubkey"], "lo": r["layer_start"], "hi": r["layer_end"],
          "layers": r["layer_end"] - r["layer_start"]} for r in receipts),
        key=lambda s: s["lo"])


def settle(receipts, layer_count, *, assignments, expected_nonce=None, check_chain=False):
    """Verify the receipt set PINNED to the swarm's assignment map and return the per-stage
    split, or raise ReceiptError. This is the PAYMENT path: the map is mandatory, a signer
    absent from it is rejected, and every assigned signer must have produced a receipt.

    receipts:    [signed receipt dict, ...] (one per stage, from the coordinator)
    layer_count: the model's true depth — coverage must tile [0, layer_count)
    assignments: {pubkey: [lo, hi]} the swarm assigned each node — REQUIRED, non-empty
    expected_nonce: the coordinator's per-job nonce (rejects a replayed receipt); None to skip
    check_chain: True on the lossless wire (out_root[i] == in_root[i+1]); False for fp8 transport

    Returns {"ok": True, "pinned": True, "layer_count": n,
             "stages": [{"pubkey","lo","hi","layers"}...]} sorted by lo.
    """
    if not assignments:
        raise ReceiptError("payment settlement requires a non-empty assignment map "
                           "(use settle_unpinned for diagnostics)")
    verify_coverage(receipts, layer_count,
                    expected_by_signer={k: tuple(v) for k, v in assignments.items()},
                    expected_nonce=expected_nonce, check_chain=check_chain)
    return {"ok": True, "pinned": True, "layer_count": layer_count, "stages": _stages(receipts)}


def settle_unpinned(receipts, layer_count, *, expected_nonce=None, check_chain=False):
    """Verify the receipt set WITHOUT an assignment map — diagnostics/audit only, NEVER a
    payment path: any validly-signed set that tiles passes, so an unknown signer's receipt
    is not rejected. Same return shape as settle with "pinned": False."""
    verify_coverage(receipts, layer_count, expected_by_signer=None,
                    expected_nonce=expected_nonce, check_chain=check_chain)
    return {"ok": True, "pinned": False, "layer_count": layer_count, "stages": _stages(receipts)}


def settle_v2(req: dict) -> dict:
    """Mesh jobs (shard-receipt/2): verify the entry's path log and the hops' checkpoints against
    the orchestrator's leases (fail closed), then split `pool` over the attested work
    (shard.settle2). Weights ride as decimal strings: they exceed 2^53 (a JS Number)."""
    from .settle2 import REBUILD_CAP, settle_v2 as _settle_v2
    out = _settle_v2(req["path_log"], req["checkpoints"], req["pool"], leases=req["leases"],
                     entry_pubkey=req["entry_pubkey"], nonce=req["expected_nonce"], job_id=req["job_id"],
                     manifest_cid=req["manifest_cid"], layer_kinds=req["layer_kinds"],
                     window=req["window"], pay=req.get("pay"),
                     cap=tuple(req.get("rebuild_cap", REBUILD_CAP)))
    out["weights"] = {k: str(v) for k, v in out["weights"].items()}
    return out


def _main() -> int:
    """`python3 -m shard.verify` — JSON in ({receipts, layer_count, mode?, expected_nonce?,
    check_chain?, assignments?}), JSON out ({ok, pinned, stages} on success; {ok:false, error} on
    a rejected set). mode "payment" (the default) REQUIRES assignments — missing/empty is a config
    bug (exit 2), not a verdict; mode "unpinned" routes to settle_unpinned and ignores them.
    mode "v2" is a mesh job: {path_log, checkpoints, leases, entry_pubkey, expected_nonce, job_id,
    manifest_cid, layer_kinds, window, pool, pay?, rebuild_cap?} -> settle_v2's {ok, work,
    weights, payouts, ...}."""
    try:
        req = json.load(sys.stdin)
    except Exception as e:  # noqa: BLE001
        json.dump({"ok": False, "error": f"bad request json: {e}"}, sys.stdout)
        return 2
    mode = req.get("mode", "payment")
    if mode == "payment" and not req.get("assignments"):
        json.dump({"ok": False, "error": "payment settlement requires assignments"}, sys.stdout)
        return 2
    try:
        if mode == "v2":
            out = settle_v2(req)
        elif mode == "unpinned":
            out = settle_unpinned(req["receipts"], req["layer_count"],
                                  expected_nonce=req.get("expected_nonce"),
                                  check_chain=bool(req.get("check_chain", False)))
        else:
            out = settle(req["receipts"], req["layer_count"],
                         expected_nonce=req.get("expected_nonce"),
                         check_chain=bool(req.get("check_chain", False)),
                         assignments=req["assignments"])
    except KeyError as e:
        json.dump({"ok": False, "error": f"missing field: {e}"}, sys.stdout)
        return 2
    except ReceiptError as e:            # a rejected set is a verdict, not a crash: exit 0, ok=false
        json.dump({"ok": False, "error": str(e)}, sys.stdout)
        return 0
    except Exception as e:  # noqa: BLE001
        json.dump({"ok": False, "error": f"verify failed: {e}"}, sys.stdout)
        return 1
    json.dump(out, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
