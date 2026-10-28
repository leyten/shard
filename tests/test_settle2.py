"""Settlement v2 (shard/settle2.py) and its shared Python <-> TypeScript vectors.

tests/vectors/settle_v2.json is the contract c0mpute's TS settlement must reproduce bit for bit:
each case gives a pool, pay weights, a rebuild cap and verify_coverage_v2's `work` list, and the
expected integer weights (decimal strings: they pass 2^53) and payouts. Some cases are the work
lists of real scripted mesh jobs (tests/receipt_mesh.py), some are synthetic edge cases (the cap
binding, remainder ties, an empty pool). Regenerate after a deliberate rule change with
`OMP_NUM_THREADS=2 python3 tests/test_settle2.py --write` and review the diff.

Run: OMP_NUM_THREADS=2 python3 -m pytest tests/test_settle2.py -q
"""
import json
import os
import random
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from shard import settle2 as S  # noqa: E402
from shard.receipt import ReceiptError  # noqa: E402

VECTORS = os.path.join(HERE, "vectors", "settle_v2.json")
POOL = 1_000_000_000          # e.g. micro-credits of one job's worker pool


def _pk(name):
    import receipt_mesh as M
    return M.pub(name)


def _synthetic():
    a, b, c, d = (f"pk-{x}" for x in "abcd")
    serve = lambda pk, lo, hi, pre, dec: {"pubkey": pk, "rte": 0, "lo": lo, "hi": hi, "kind": "serve",   # noqa: E731
                                          "prefill": pre, "decode": dec}
    return [
        {"name": "rebuild cap binds", "pool": POOL,
         "work": [serve(a, 0, 6, 100, 50), serve(b, 6, 12, 100, 50),
                  {"pubkey": c, "rte": 1, "lo": 0, "hi": 6, "kind": "rebuild", "positions": 4000},
                  {"pubkey": d, "rte": 1, "lo": 6, "hi": 12, "kind": "rebuild", "positions": 1000}]},
        {"name": "rebuild under the cap", "pool": POOL,
         "work": [serve(a, 0, 6, 100, 900), serve(b, 6, 12, 100, 900),
                  {"pubkey": c, "rte": 1, "lo": 6, "hi": 12, "kind": "rebuild", "positions": 200}]},
        {"name": "remainder ties go to the smaller pubkey", "pool": 10,
         "work": [serve(c, 0, 4, 0, 1), serve(a, 4, 8, 0, 1), serve(b, 8, 12, 0, 1)]},
        {"name": "one key in two rtes and a mirror", "pool": 999_999_937,
         "work": [serve(a, 0, 5, 30, 70), serve(b, 5, 12, 30, 70),
                  {"pubkey": a, "rte": 1, "lo": 0, "hi": 5, "kind": "serve", "prefill": 0, "decode": 44},
                  {"pubkey": c, "rte": 1, "lo": 5, "hi": 12, "kind": "serve", "prefill": 0, "decode": 44},
                  {"pubkey": d, "rte": 0, "lo": 5, "hi": 12, "kind": "mirror", "positions": 100}]},
        {"name": "custom pay and cap", "pool": 123_456_789, "pay": {"prefill": 250, "rebuild": 400},
         "cap": [1, 10],
         "work": [serve(a, 0, 12, 500, 100),
                  {"pubkey": b, "rte": 1, "lo": 0, "hi": 12, "kind": "rebuild", "positions": 600}]},
        {"name": "empty pool", "pool": 0, "work": [serve(a, 0, 12, 1, 1)]},
        {"name": "48-layer long context, capped (weights past 2^53)", "pool": 2 ** 52 + 7,
         "work": [serve(a, 0, 24, 200_000, 30_000), serve(b, 24, 48, 200_000, 30_000),
                  {"pubkey": c, "rte": 1, "lo": 24, "hi": 48, "kind": "rebuild", "positions": 400_000}]},
    ]


def _expect(case):
    r = S.settle_work(case["work"], case["pool"], pay=case.get("pay"), cap=tuple(case.get("cap", S.REBUILD_CAP)))
    return {"weights": {k: str(v) for k, v in sorted(r["weights"].items())}, "capped": r["capped"],
            "payouts": dict(sorted(r["payouts"].items()))}


def build_vectors():
    import receipt_mesh as M
    cases = []
    for name in ("plain", "takeover", "dead", "reroute"):
        b = M.scenario(name)
        cases.append({"name": f"mesh scenario: {name}", "pool": POOL, "work": M.verify(b)["work"]})
    cases += _synthetic()
    for c in cases:
        c["expect"] = _expect(c)
    return {"schema": "shard-settle-vectors/2", "pay": S.PAY, "rebuild_cap": list(S.REBUILD_CAP),
            "notes": "weights are decimal strings (BigInt); payouts sum to pool; ties -> smaller pubkey",
            "cases": cases}


def _load():
    with open(VECTORS) as f:
        return json.load(f)


def test_vectors_reproduce():
    v = _load()
    assert v["pay"] == S.PAY and tuple(v["rebuild_cap"]) == S.REBUILD_CAP
    for c in v["cases"]:
        assert _expect(c) == c["expect"], c["name"]
        assert sum(c["expect"]["payouts"].values()) == c["pool"], c["name"]


def test_vectors_cover_the_rules():
    cases = {c["name"]: c for c in _load()["cases"]}
    assert cases["rebuild cap binds"]["expect"]["capped"]
    assert not cases["rebuild under the cap"]["expect"]["capped"]
    assert cases["remainder ties go to the smaller pubkey"]["expect"]["payouts"] == {"pk-a": 4, "pk-b": 3, "pk-c": 3}
    big = cases["48-layer long context, capped (weights past 2^53)"]["expect"]["weights"]
    assert any(int(w) > 2 ** 53 for w in big.values())
    assert any(c["name"].startswith("mesh scenario") and any(w["kind"] == "rebuild" for w in c["work"])
               for c in cases.values())


def test_cap_holds_exactly_on_weights():
    for c in _load()["cases"]:
        units = S.unit_weights(c["work"], c.get("pay"))
        w, capped = S.capped_weights(units, tuple(c.get("cap", S.REBUILD_CAP)))
        num, den = c.get("cap", S.REBUILD_CAP)
        reb = sum(units[pk]["rebuild"] for pk in units)
        base = sum(units[pk]["base"] for pk in units)
        if capped:
            fr = base * num                                   # every rebuild weight x base*num
            assert sum(units[pk]["rebuild"] * fr for pk in units) * den == sum(w.values()) * num
        else:
            assert reb * den <= (reb + base) * num


def test_random_jobs_pay_exactly_the_pool():
    rng = random.Random(1234)
    for _ in range(2000):
        n = rng.randint(1, 9)
        pks = [f"k{rng.randint(0, 6)}" for _ in range(n)]
        work = []
        for pk in pks:
            lo = rng.randint(0, 40)
            hi = lo + rng.randint(1, 8)
            kind = rng.choice(["serve", "serve", "rebuild", "mirror"])
            if kind == "serve":
                work.append({"pubkey": pk, "rte": 0, "lo": lo, "hi": hi, "kind": kind,
                             "prefill": rng.randint(0, 5000), "decode": rng.randint(0, 5000)})
            else:
                work.append({"pubkey": pk, "rte": 1, "lo": lo, "hi": hi, "kind": kind,
                             "positions": rng.randint(0, 9000)})
        pool = rng.choice([0, 1, 7, 10 ** 6, rng.randint(0, 10 ** 15)])
        try:
            r = S.settle_work(work, pool)
        except ReceiptError as e:
            assert "no attested work" in str(e) or "no serve work" in str(e)
            continue
        assert sum(r["payouts"].values()) == pool
        assert all(v >= 0 for v in r["payouts"].values())
        assert set(r["payouts"]) == {w["pubkey"] for w in work}       # one payout per key: no double pay
        total = sum(r["weights"].values())
        for pk, v in r["payouts"].items():                             # largest remainder: within 1
            assert abs(v * total - pool * r["weights"][pk]) < total


def test_settle_rejects_bad_input():
    with pytest.raises(ReceiptError, match="no attested work"):
        S.settle_work([], 10)
    with pytest.raises(ReceiptError, match="pool"):
        S.settle_work([{"pubkey": "a", "lo": 0, "hi": 1, "kind": "serve", "prefill": 0, "decode": 1}], -1)
    with pytest.raises(ReceiptError, match="no serve work"):
        S.settle_work([{"pubkey": "a", "lo": 0, "hi": 1, "kind": "rebuild", "positions": 9}], 10)
    with pytest.raises(ReceiptError, match="unknown work kind"):
        S.settle_work([{"pubkey": "a", "lo": 0, "hi": 1, "kind": "tip", "positions": 1}], 10)


# ── end to end: scenario -> verify -> settle, in-process and over the CLI seam ────────────────
@pytest.mark.parametrize("name", ["spec", "takeover", "dead"])
def test_scenarios_settle_to_the_pool(name):
    import receipt_mesh as M
    b = M.scenario(name)
    r = S.settle_v2(b["path_log"], b["checkpoints"], POOL, leases=b["leases"], entry_pubkey=b["entry_pubkey"],
                    nonce=b["nonce"], job_id=b["job_id"], **M.MODEL_INFO)
    assert sum(r["payouts"].values()) == POOL
    pubs = {h["pubkey"] for e in b["path_log"]["rtes"] for h in e["hops"]}
    assert pubs <= set(r["payouts"])
    if name == "dead":
        dead = b["path_log"]["rtes"][0]["dead"][0]
        alive = b["path_log"]["rtes"][0]["hops"][0]["pubkey"]
        assert 0 < r["payouts"][dead] < r["payouts"][alive], "a dead hop is paid, to its last checkpoint"


def _cli(req):
    p = subprocess.run([sys.executable, "-m", "shard.verify"], input=json.dumps(req), capture_output=True,
                       text=True, cwd=REPO, env=dict(os.environ, OMP_NUM_THREADS="2"), timeout=120)
    return p.returncode, json.loads(p.stdout)


def _req(b):
    import receipt_mesh as M
    return {"mode": "v2", "path_log": b["path_log"], "checkpoints": b["checkpoints"], "leases": b["leases"],
            "entry_pubkey": b["entry_pubkey"], "expected_nonce": b["nonce"], "job_id": b["job_id"],
            "pool": POOL, **M.MODEL_INFO}


def test_cli_v2_settles_and_rejects():
    import receipt_mesh as M
    b = M.scenario("takeover")
    code, out = _cli(_req(b))
    assert code == 0 and out["ok"], out
    assert sum(out["payouts"].values()) == POOL
    assert all(isinstance(w, str) for w in out["weights"].values())
    bad = _req(b)
    bad["expected_nonce"] = "00" * 16
    code, out = _cli(bad)
    assert code == 0 and not out["ok"] and "nonce" in out["error"]
    del bad["pool"]
    code, out = _cli(bad)
    assert code == 2 and "missing field" in out["error"]


if __name__ == "__main__":
    if "--write" in sys.argv:
        os.makedirs(os.path.dirname(VECTORS), exist_ok=True)
        with open(VECTORS, "w") as f:
            json.dump(build_vectors(), f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"wrote {VECTORS}")
