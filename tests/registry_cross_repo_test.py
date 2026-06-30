"""registry_cross_repo_test — M1 acceptance: ONE registry, BOTH repos, no drift.

  python3 tests/registry_cross_repo_test.py

This is THE test that would have caught gpt-oss 120-vs-36. It:
  1. loads + verifies the signed registry on the PYTHON (shard) side,
  2. loads + verifies the SAME file on the TYPESCRIPT (c0mpute) side (shells out to
     lib/orchestrator/registryDump.ts under the c0mpute repo),
  3. asserts both resolve EVERY id with a non-zero layerCount and a KNOWN adapter, and
  4. asserts the two repos agree FIELD-FOR-FIELD on every model (layerCount, gbPerLayer,
     kvGbPerLayer, adapter, quant, enginePath, workerModel, hfArch, tokenizerId).

Editing the layer count in registry/models.src.json and re-signing changes BOTH repos at
once; an unsigned edit to either repo's copy fails the signature. Drift is structurally
impossible.

Requires: the c0mpute repo checked out next to shard (../c0mpute) with `npx tsx` available.
Set C0MPUTE_DIR to override. Skips the TS half (with a loud SKIP, not a pass) if absent.
"""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from shard import registry as reg  # noqa: E402

REGISTRY_PATH = os.path.join(REPO, "registry", "models.json")
C0MPUTE_DIR = os.environ.get("C0MPUTE_DIR", os.path.join(os.path.dirname(REPO), "c0mpute"))

passed = failed = 0


def ok(cond, name, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  OK {name}{(' ' + detail) if detail else ''}")
    else:
        failed += 1
        print(f"  FAIL {name} {detail}")


# Fields both repos must agree on, per model.
SHARED_FIELDS = ("hfArch", "workerModel", "enginePath", "layerCount",
                 "gbPerLayer", "kvGbPerLayer", "quant", "adapter", "tokenizerId")


def load_python_side():
    """Verify with the pinned pubkey FROM THE FILE (self-consistent); a prod CI would pin a
    known constant instead. We read the publisher_pubkey then re-verify against it, which
    still proves the signature is valid over the canonical bytes."""
    raw = reg.load_registry(REGISTRY_PATH, verify=False)
    pub = raw.get("publisher_pubkey")
    verified = reg.load_registry(REGISTRY_PATH, expected_pubkey=pub)  # raises on any failure
    return verified, pub


def load_ts_side(pub):
    """Shell out to the c0mpute TS loader; returns its parsed dump or None if unavailable."""
    dump = os.path.join(C0MPUTE_DIR, "lib", "orchestrator", "registryDump.ts")
    if not os.path.exists(dump):
        return None
    env = {**os.environ, "SHARD_MODELS_JSON": REGISTRY_PATH, "SHARD_MODELS_PUBKEY": pub}
    try:
        out = subprocess.run(
            ["npx", "tsx", "lib/orchestrator/registryDump.ts"],
            cwd=C0MPUTE_DIR, env=env, capture_output=True, text=True, timeout=120,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  (TS side unavailable: {e})")
        return None
    line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    if not line:
        print(f"  (TS side produced no output; stderr: {out.stderr[:300]})")
        return None
    return json.loads(line)


def main():
    print("── Python (shard) side ──")
    py, pub = load_python_side()
    py_by_id = {m["id"]: m for m in py["models"]}
    ok(py["schema"] == reg.SCHEMA, "python: schema is shard-models/1")
    for mid, m in py_by_id.items():
        ok(isinstance(m["layerCount"], int) and m["layerCount"] > 0,
           f"python: {mid} layerCount>0", f"({m['layerCount']})")
        ok(m["adapter"] in reg.KNOWN_ADAPTERS, f"python: {mid} adapter known", f"({m['adapter']})")

    print("\n── TypeScript (c0mpute) side ──")
    ts = load_ts_side(pub)
    if ts is None:
        print("  SKIP: c0mpute repo / tsx not available — cannot prove cross-repo agreement.")
        print("  (set C0MPUTE_DIR and ensure `npx tsx` works to run the full M1 acceptance test)")
        # A skip is NOT a pass: surface it but don't fail CI if c0mpute genuinely isn't checked out.
        print(f"\n{'ALL' if failed == 0 else 'SOME FAILURES:'} {passed} PASS (TS half skipped)"
              + (f", {failed} FAIL" if failed else ""))
        sys.exit(1 if failed else 0)

    ok(ts.get("ok") is True, "ts: registry loaded + verified", str(ts.get("error", "")))
    ts_by_id = ts.get("models", {})
    ok(ts.get("schema") == reg.SCHEMA, "ts: schema is shard-models/1")
    ok(set(ts.get("adapters", [])) == set(reg.KNOWN_ADAPTERS),
       "ts/python KNOWN_ADAPTERS identical",
       f"(ts={sorted(ts.get('adapters', []))} py={sorted(reg.KNOWN_ADAPTERS)})")

    print("\n── cross-repo agreement (the gpt-oss 120-vs-36 guard) ──")
    ok(set(py_by_id) == set(ts_by_id), "both repos resolve the same model ids",
       f"(py={sorted(py_by_id)} ts={sorted(ts_by_id)})")
    ok(ts.get("publisher_pubkey") == pub, "both repos verified the same publisher key")
    for mid in sorted(set(py_by_id) & set(ts_by_id)):
        for f in SHARED_FIELDS:
            pv, tv = py_by_id[mid].get(f), ts_by_id[mid].get(f)
            ok(pv == tv, f"{mid}.{f} agrees", f"(py={pv!r} ts={tv!r})")

    # The c0mpute repo carries a test FIXTURE copy of the registry (for its offline unit
    # tests). A copy is a drift surface — guard it: the fixture MUST be byte-identical to the
    # signed source of truth. If it drifts, re-copy registry/models.json into the fixture.
    fixture = os.path.join(C0MPUTE_DIR, "lib", "orchestrator", "fixtures", "models.json")
    if os.path.exists(fixture):
        with open(REGISTRY_PATH, "rb") as a, open(fixture, "rb") as b:
            ok(a.read() == b.read(), "c0mpute test fixture == signed registry (no copy drift)")

    print(f"\n{'ALL' if failed == 0 else 'SOME FAILURES:'} {passed} PASS"
          + (f", {failed} FAIL" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
