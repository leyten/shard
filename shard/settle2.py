"""Settlement v2: split a job's pool over the work `verify_coverage_v2` attested (ARCHITECTURE-v2 §7.3.6).

The Python reference. c0mpute's TypeScript settlement mirrors it through the shared vectors in
tests/vectors/settle_v2.json, so every rule here is integer arithmetic with a total order: no
floats, no locale, no iteration-order dependence. TS must use BigInt (weights x pool exceed 2^53).

Weights, in units of one layer x one decode position = PAY["decode"]:
    serve    layers x (PAY["decode"] x decode positions + PAY["prefill"] x prompt positions)
    rebuild  layers x PAY["rebuild"] x rebuild positions   (a replacement re-deriving its state)
    mirror   layers x PAY["mirror"]  x acknowledged positions (a backup holding verified KV)
Each (layer, position) of serve work belongs to exactly one hop: rtes cover disjoint positions and
a dead hop is paid only to its last delivered checkpoint, so nothing is paid twice. The rebuild
share of the pool is capped (REBUILD_CAP = 1/5): above it, every rebuild weight is scaled down
by one common factor. The pool is then split by largest remainder (ties to the smaller pubkey),
so payouts are integers that sum to the pool exactly. The defaults are estimates (design-mesh
§7.8: rebuild priced at prefill cost, mirror tiny); c0mpute sets the live values.
"""
from __future__ import annotations

try:
    from .receipt import ReceiptError
except ImportError:
    from receipt import ReceiptError

PAY = {"decode": 1000, "prefill": 100, "rebuild": 100, "mirror": 10}
REBUILD_CAP = (1, 5)


def unit_weights(work, pay=None) -> dict:
    """{pubkey: {"base": int, "rebuild": int}} from verify_coverage_v2's `work` list."""
    pay = dict(PAY, **(pay or {}))
    out: dict[str, dict[str, int]] = {}
    for w in work:
        layers = w["hi"] - w["lo"]
        if w["kind"] == "serve":
            units, bucket = pay["decode"] * w["decode"] + pay["prefill"] * w["prefill"], "base"
        elif w["kind"] == "rebuild":
            units, bucket = pay["rebuild"] * w["positions"], "rebuild"
        elif w["kind"] == "mirror":
            units, bucket = pay["mirror"] * w["positions"], "base"
        else:
            raise ReceiptError(f"unknown work kind {w['kind']!r}")
        if layers < 1 or units < 0:
            raise ReceiptError(f"bad work item {w}")
        d = out.setdefault(w["pubkey"], {"base": 0, "rebuild": 0})
        d[bucket] += layers * units
    return out


def capped_weights(units: dict, cap=REBUILD_CAP) -> tuple[dict, bool]:
    """One integer weight per pubkey with the rebuild share <= cap. Returns (weights, capped)."""
    num, den = cap
    base = sum(u["base"] for u in units.values())
    reb = sum(u["rebuild"] for u in units.values())
    if base > 0 and reb * den > (reb + base) * num:
        # rebuild total R scaled to R' = base*num/(den-num): multiply base by R*(den-num), rebuild by base*num
        fb, fr = reb * (den - num), base * num
        return {pk: u["base"] * fb + u["rebuild"] * fr for pk, u in units.items()}, True
    return {pk: u["base"] + u["rebuild"] for pk, u in units.items()}, False


def split_pool(pool: int, weights: dict) -> dict:
    """Largest remainder: floor shares, then +1 to the largest remainders (ties: smaller pubkey)."""
    if type(pool) is not int or pool < 0:
        raise ReceiptError(f"pool must be a non-negative integer, got {pool!r}")
    total = sum(weights.values())
    if total <= 0:
        raise ReceiptError("no attested work to pay")
    out = {pk: pool * w // total for pk, w in weights.items()}
    rem = pool - sum(out.values())
    order = sorted(weights, key=lambda pk: (-(pool * weights[pk] % total), pk))
    for pk in order[:rem]:
        out[pk] += 1
    return out


def settle_work(work, pool: int, *, pay=None, cap=REBUILD_CAP) -> dict:
    """Payouts for one job: {payouts: {pubkey: int}, weights: {pubkey: int}, capped: bool}."""
    units = unit_weights(work, pay)
    weights, capped = capped_weights(units, cap)
    return {"payouts": split_pool(pool, weights), "weights": weights, "capped": capped}


def settle_v2(path_log, checkpoints, pool: int, *, leases, entry_pubkey, nonce, job_id, layer_kinds,
              window, pay=None, cap=REBUILD_CAP) -> dict:
    """Verify (fail closed) and split the pool. Returns the coverage verdict plus the payouts."""
    try:
        from .receipt2 import verify_coverage_v2
    except ImportError:
        from receipt2 import verify_coverage_v2
    cov = verify_coverage_v2(path_log, checkpoints, leases=leases, entry_pubkey=entry_pubkey,
                             nonce=nonce, job_id=job_id, layer_kinds=layer_kinds, window=window)
    return dict(cov, **settle_work(cov["work"], pool, pay=pay, cap=cap))
