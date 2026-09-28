"""StageRuntime conformance suite: the executable spec of ARCHITECTURE-v2 §4.1 (BUILD-PLAN F3).

A library, not a test file (pytest does not collect it). Every backend (ToyRuntime, the CPU
MiMo reference, CUDA, MLX, ROCm) plugs in a `RuntimeFactory` and must pass every check in
`CHECKS`:

    from runtime_conformance import CHECKS, Skip
    @pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.__name__)
    def test_conformance(check):
        try:
            check(MyFactory())
        except Skip as e:
            pytest.skip(str(e))

tests/test_runtime_conformance.py runs it on ToyRuntime in both numeric modes and on mutant
runtimes, each of which must fail at least one check (anti-vacuity).

Everything is compared as BYTES. A float backend widens only `close_enough` (the oracle
parity and the fused-tap reference, where the architecture allows tolerance); batch
invariance, row independence, replay, partition invariance and the rebuild stay byte-exact,
because receipts and takeovers rely on them. Checks that need the "invariant" numerics mode
skip on a "fast" runtime; checks that need chunk invariance skip unless describe() claims it
and fall back to replay-equality otherwise.

Terms: W = the spec window a session is opened with; w = the SWA window; F = the committed
(final-row) frontier; "twin" = a fresh runtime driven through the same history.
"""
from __future__ import annotations

import json
import random
from typing import Callable, Mapping, Protocol

import numpy as np

from shard.runtime.api import (BACKENDS, BadRequest, Capability, KVBlob, KVFull, LoadRefused,
                               PositionGap, RollbackPastCommit, Rows, Segment, SessionExists,
                               SpecWindowExceeded, StageRuntime, StageRuntimeError, StepIn,
                               StepTooLarge, UnknownSession, WTooDeep, rebuild_start)

W_SPEC = 4          # spec window sessions are opened with
MAX_CTX = 256       # max_pos sessions are opened with
REBUILD_PATTERNS = ("SSG", "SGS", "SSSSSG", "SGSGS", "GSS", "SSS", "GG")


class Skip(Exception):
    """The check does not apply to this backend (the message says why)."""


class RuntimeFactory(Protocol):
    """What a backend provides to be tested. `make` returns a LOADED runtime whose StageSpec
    carries `taps` = self.taps. Inputs from hidden()/ids() must be valid for the backend
    (finite, in range); perturb() must change EVERY row to other valid values."""
    name: str
    layer_kinds: tuple[str, ...]          # the whole model, per absolute layer
    window: int                           # SWA window w
    taps: tuple[int, ...]                 # drafter tap layers (empty: tap checks skip)

    def make(self, lo: int, hi: int, *, is_head: bool = False, is_tail: bool = False,
             tap_mode: str = "raw", spec_window: int = W_SPEC, max_sessions: int = 8,
             max_ctx: int = MAX_CTX, kv_budget_bytes: int = 0) -> StageRuntime: ...

    def hidden(self, n: int, seed: int) -> Rows: ...

    def ids(self, n: int, seed: int) -> Rows: ...

    def perturb(self, rows: Rows, seed: int) -> Rows: ...

    def fuse_taps(self, raw: Mapping[int, Rows]) -> Rows:
        """Reference fc(concat(taps)) over self.taps, in the stage's tap wire dtype."""

    def close_enough(self, a: Rows, b: Rows, what: str) -> bool:
        """what in ("oracle", "tap_fusion"); exact for an integer/invariant backend."""

    def oracle(self, lo: int, hi: int, *, is_head: bool, is_tail: bool, x: Rows,
               schedule: list) -> dict | None:
        """Fault-free reference {"hidden", "logits", "taps"} or None if there is none."""

    def make_broken(self, lo: int, hi: int) -> None:
        """Build + load a runtime whose weight source lacks one tensor (must raise LoadRefused)."""


# ── small helpers ─────────────────────────────────────────────────────────────────────────────

def _raises(exc, fn: Callable, *args, **kw):
    try:
        fn(*args, **kw)
    except exc as e:
        return e
    except Exception as e:   # noqa: BLE001 -- a wrong refusal is a contract failure too
        raise AssertionError(f"{getattr(fn, '__name__', fn)}{args} raised {type(e).__name__}: {e}, "
                             f"expected {exc}") from e
    raise AssertionError(f"{getattr(fn, '__name__', fn)}{args} did not raise {exc}")


def _pat(f) -> str:
    return "".join("G" if k == "global" else "S" for k in f.layer_kinds)


def _find(f, pat: str):
    i = _pat(f).find(pat)
    return None if i < 0 else (i, i + len(pat))


def _roles(f):
    """(lo, hi, is_head, is_tail) spans every general check runs on: a head span, a middle span
    and a tail span (each holding both layer kinds where the model has both) and the whole
    model. A factory may override with a `roles` attribute."""
    if getattr(f, "roles", None):
        return list(f.roles)
    L = len(f.layer_kinds)

    def both(lo, hi):
        return len(set(f.layer_kinds[lo:hi])) == len(set(f.layer_kinds))

    k = next(k for k in range(1, L + 1) if both(0, k))
    j = next(j for j in range(L - 1, -1, -1) if both(j, L))
    m_lo, m_hi = max(1, L // 3), min(L - 1, max(L // 3 + 2, 2 * L // 3 + 1))
    out = [(0, k, True, False), (j, L, False, True), (0, L, True, True)]
    if m_lo < m_hi:
        out.insert(1, (m_lo, m_hi, False, False))
    return out


def _inp(f, head: bool, n: int, seed: int) -> Rows:
    return f.ids(n, seed) if head else f.hidden(n, seed)


def _one(rt, seg: Segment):
    return rt.step(StepIn((seg,))).outs[0]


def _seg(sid, start, x, kind=None, **kw) -> Segment:
    return Segment(sid, start, x.n, x, kind=kind or ("decode" if x.n == 1 else "prefill"), **kw)


def _mk(f, lo, hi, head, tail, **kw):
    return f.make(lo, hi, is_head=head, is_tail=tail, **kw)


def _play(rt, sid, ops, logits="none"):
    """Apply a session script; returns the SegOut of every "seg" op (with its keep count).
    ops: ("commit", pos) | ("truncate", pos) | ("seg", start, rows, kind, keep)."""
    outs = []
    for op in ops:
        if op[0] == "commit":
            rt.commit(sid, op[1])
        elif op[0] == "truncate":
            rt.truncate(sid, op[1])
        else:
            _, start, x, kind, keep = op
            outs.append((start, keep, _one(rt, _seg(sid, start, x, kind, logits=logits))))
    return outs


def _row_sig(o, r: int) -> tuple:
    lg = None
    if o.logits is not None and o.logits.n == o.hidden.n:
        lg = o.logits.row_bytes(r)
    taps = tuple((t, o.taps[t].row_bytes(r)) for t in sorted(o.taps))
    acc = None if o.tap_acc is None else o.tap_acc.row_bytes(r)
    return (o.hidden.row_bytes(r), lg, taps, acc)


def _accepted(outs) -> dict:
    """position -> row signature, for the rows each seg op KEPT (accepted/committed)."""
    sig = {}
    for start, keep, o in outs:
        for r in range(keep):
            sig[start + r] = _row_sig(o, r)
    return sig


def _spec_script(f, head, *, seed, prefix, rounds, W, twin=False):
    """A pipelined-speculation history: a committed prefill of `prefix` rows, then rounds of
    a k-row chunk at the frontier c (k <= W) whose first j rows are accepted: commit(c+j), and
    the next round re-feeds row c+j (implicit truncate via start_pos, or an explicit truncate
    as lease_update does). The plan starts with the deepest cases (a full W rejected, a full W
    accepted). `twin=True` gives the same script with every REJECTED row perturbed: shapes and
    accepted values are identical, so a correct runtime's kept rows must be byte-identical.
    Returns (ops, committed_stream, c_end)."""
    rng = random.Random(seed)
    fixed = [(W, 0), (W, W), (1, 0), (W, 1), (1, 1), (2, 0), (W, W - 1)]
    plan = fixed[:rounds] + [(k, rng.randint(0, k)) for k in (rng.randint(1, W) for _ in range(max(0, rounds - len(fixed))))]
    true = _inp(f, head, prefix + W * len(plan) + 4 * W, seed)
    ops = [("commit", prefix), ("seg", 0, true.slice(0, prefix), "prefill", prefix)]
    c = prefix
    for i, (k, j) in enumerate(plan):
        rej = _inp(f, head, k, seed * 1009 + i + 1)
        if twin:
            rej = f.perturb(rej, seed * 7 + i)
        x = true.slice(c, c + k) if j == k else Rows.concat(
            ([true.slice(c, c + j)] if j else []) + [rej.slice(j, k)])
        ops.append(("seg", c, x, "decode" if k == 1 else "verify", j))
        ops.append(("commit", c + j))
        if j < k and (i + seed) % 2:
            ops.append(("truncate", c + j))
        c += j
    return ops, true, c


def _continue(rt, sid, true, c, rounds=3, W=W_SPEC, logits="none"):
    """Feed committed rows [c, c + rounds*W) in W-row verify chunks, committing each."""
    outs = []
    for i in range(rounds):
        p = c + i * W
        outs.append((p, W, _one(rt, _seg(sid, p, true.slice(p, p + W), "verify", logits=logits))))
        rt.commit(sid, p + W)
    return _accepted(outs)


def _requires_invariant(rt):
    if rt.describe().numerics_mode != "invariant":
        raise Skip("needs numerics_mode='invariant' (fast mode promises run-to-run determinism only)")


def _cat(outs, attr):
    parts = [getattr(o, attr) for o in outs]
    return None if any(p is None for p in parts) else Rows.concat(parts)


# ── the checks ────────────────────────────────────────────────────────────────────────────────

def check_describe(f):
    """describe() is a valid, JSON-portable, device-neutral record that matches the stage;
    warmup() measures a cost curve, self-probes and leaves no session behind."""
    for lo, hi, head, tail in _roles(f):
        rt = _mk(f, lo, hi, head, tail)
        cap = rt.describe()
        assert isinstance(cap, Capability)
        assert (cap.lo, cap.hi, cap.is_head, cap.is_tail) == (lo, hi, head, tail)
        assert cap.layer_kinds == tuple(f.layer_kinds[lo:hi])
        assert cap.swa_window == (f.window if "swa" in cap.layer_kinds else 0)
        assert cap.rebuild_positions == (f.window - 1) * cap.n_swa
        assert cap.device.backend in BACKENDS and cap.device.total_mem_bytes >= 0
        assert Capability.from_dict(json.loads(json.dumps(cap.to_dict()))) == cap
        assert cap.tap_layers_held == tuple(sorted(t for t in set(f.taps) if lo <= t < hi))
        assert cap.spec_window_max >= W_SPEC and cap.max_tokens_per_step >= 1
        got = [cap.sessions_at(c, W_SPEC) for c in (16, 64, MAX_CTX)]
        assert got == sorted(got, reverse=True) and got[-1] >= 1, got
        rt.open(1, MAX_CTX, W_SPEC)
        o = _one(rt, _seg(1, 0, _inp(f, head, 3, 5)))
        assert o.hidden.dtype == cap.wire_dtype and o.hidden.width == cap.hidden_size and o.hidden.n == 3
        rt.close(1)
        wcap = rt.warmup()
        assert wcap.service_ms and all(ms > 0 for _, ms in wcap.service_ms), wcap.service_ms
        assert wcap.self_probe_ok is True
        assert wcap.open_sessions == 0 and wcap.kv_free_bytes == cap.kv_free_bytes
        assert wcap.numerics_id == cap.numerics_id


def check_admission(f):
    """open() reserves exactly what describe().sessions_at() predicts; KVFull, WTooDeep,
    SessionExists and UnknownSession fire; close() frees and is idempotent."""
    lo, hi, head, tail = _roles(f)[1]
    per = _mk(f, lo, hi, head, tail).describe().session_kv_bytes(64, W_SPEC)
    rt = _mk(f, lo, hi, head, tail, kv_budget_bytes=3 * per + per // 2, max_sessions=8)
    want = rt.describe().sessions_at(64, W_SPEC)
    assert want >= 1
    opened = 0
    for sid in range(1, 50):
        try:
            rt.open(sid, 64, W_SPEC)
        except KVFull:
            break
        opened += 1
    assert opened == want, f"sessions_at predicted {want}, open() admitted {opened}"
    assert rt.describe().sessions_at(64, W_SPEC) == 0
    rt.close(1)
    rt.close(1)                                                   # idempotent
    assert rt.describe().sessions_at(64, W_SPEC) == 1
    rt.open(1, 64, W_SPEC)
    rt2 = _mk(f, lo, hi, head, tail, max_sessions=2)
    cap2 = rt2.describe()
    rt2.open(1, 64, W_SPEC)
    rt2.open(2, 64, W_SPEC)
    _raises(KVFull, rt2.open, 3, 64, W_SPEC)
    _raises(SessionExists, rt2.open, 1, 64, W_SPEC)
    rt2.close(2)
    _raises(WTooDeep, rt2.open, 2, 64, cap2.spec_window_max + 1)
    _raises(KVFull, rt2.open, 2, cap2.max_ctx + 1, W_SPEC)
    rt2.close(999)
    for call in (lambda: rt2.length(9), lambda: rt2.info(9), lambda: rt2.truncate(9, 0),
                 lambda: rt2.commit(9, 0), lambda: rt2.export_state(9, 0, 0),
                 lambda: rt2.rebuild_from(9, 0), lambda: _one(rt2, _seg(9, 0, _inp(f, head, 1, 1)))):
        _raises(UnknownSession, call)


def check_validation_atomic(f):
    """Every malformed step is refused with the right error and changes NO session: a step
    mixing one valid and one invalid segment computes nothing."""
    for lo, hi, head, tail in (_roles(f)[1], _roles(f)[2]):
        rt = _mk(f, lo, hi, head, tail)
        cap = rt.describe()
        xa, xb = _inp(f, head, 24, 11), _inp(f, head, 24, 12)
        hist = {1: [("commit", 6), ("seg", 0, xa.slice(0, 8), "prefill", 8)],
                2: [("commit", 5), ("seg", 0, xb.slice(0, 5), "prefill", 5)]}
        for sid in hist:
            rt.open(sid, MAX_CTX, W_SPEC)
            _play(rt, sid, hist[sid])
        good = _seg(2, 5, xb.slice(5, 6))
        bad = [
            (PositionGap, [_seg(1, 9, xa.slice(9, 10))]),
            (BadRequest, [_seg(1, 8, xa.slice(8, 9)), _seg(1, 8, xa.slice(8, 9))]),
            (BadRequest, [_seg(1, 8, xa.slice(8, 10), "decode")]),
            (SpecWindowExceeded, [_seg(1, 8, xa.slice(8, 8 + W_SPEC))]),
            (RollbackPastCommit, [_seg(1, 5, xa.slice(5, 6))]),
            (BadRequest, [_seg(1, 8, xa.slice(8, 9), "rebuild")]),
            (BadRequest, [_seg(1, 8, xa.slice(8, 9), "bogus")]),
            (PositionGap, [good, _seg(1, 9, xa.slice(9, 10))]),     # atomic: the good one must not run
        ]
        foreign = next(d for d in ("f32", "i64", "i32", "bf16") if d not in cap.codecs_in)
        shape = (1,) if head else (1, cap.hidden_size)
        bad.append((BadRequest, [_seg(1, 8, Rows(foreign, np.zeros(shape, dtype=np.dtype(
            {"f32": np.float32, "i64": np.int64, "i32": np.int32, "bf16": np.uint16}[foreign]))))]))
        if not head:
            h = f.hidden(1, 3)
            wide = Rows(h.dtype, np.concatenate([h.data, h.data[:, :1]], axis=1))
            bad.append((BadRequest, [_seg(1, 8, wide)]))
        if not tail:
            bad.append((BadRequest, [_seg(1, 8, xa.slice(8, 9), logits="last")]))
        if cap.max_tokens_per_step < MAX_CTX - 8:
            n = cap.max_tokens_per_step + 1
            rt.open(3, MAX_CTX, W_SPEC)
            rt.commit(3, n)
            bad.append((StepTooLarge, [_seg(3, 0, _inp(f, head, n, 13))]))
        before = {s: rt.info(s) for s in (1, 2)}
        for exc, segs in bad:
            _raises(exc, rt.step, StepIn(tuple(segs)))
            assert {s: rt.info(s) for s in (1, 2)} == before, f"refused step {exc.__name__} changed state"
        twin = _mk(f, lo, hi, head, tail)
        for sid in hist:
            twin.open(sid, MAX_CTX, W_SPEC)
            _play(twin, sid, hist[sid])
        for sid, x, st in ((1, xa, 8), (2, xb, 5)):
            a = _one(rt, _seg(sid, st, x.slice(st, st + 1)))
            b = _one(twin, _seg(sid, st, x.slice(st, st + 1)))
            assert a.hidden == b.hidden, "a refused step left a trace in session state"


def check_mixed_kinds(f):
    """One step mixing a prefill chunk, a verify chunk that rolls back, a decode row and a
    rebuild segment of four sessions equals each segment run alone; logits none/last/all
    shapes; the sessions continue identically."""
    lo_hi = [r for r in _roles(f) if r[0] == 0 or r[3]][:2]
    for lo, hi, head, tail in lo_hi:
        kinds = f.layer_kinds[lo:hi]
        xs = {s: _inp(f, head, 32, 20 + s) for s in (1, 2, 3, 4)}
        lg = (lambda m: m) if tail else (lambda m: "none")
        hist = {1: [("commit", 6)],
                2: [("commit", 5), ("seg", 0, xs[2].slice(0, 5), "prefill", 5),
                    ("seg", 5, xs[2].slice(5, 8), "verify", 3), ("commit", 6)],
                3: [("commit", 5), ("seg", 0, xs[3].slice(0, 5), "prefill", 5)]}
        final = {1: _seg(1, 0, xs[1].slice(0, 6), "prefill", logits=lg("none")),
                 2: _seg(2, 6, xs[2].slice(6, 9), "verify", logits=lg("all")),
                 3: _seg(3, 5, xs[3].slice(5, 6), "decode", logits=lg("last"))}
        F, g = 12, 7
        prim = _mk(f, lo, hi, head, tail)
        prim.open(4, MAX_CTX, W_SPEC)
        _play(prim, 4, [("commit", F), ("seg", 0, xs[4].slice(0, F), "prefill", F)])
        blob = prim.export_state(4, 0, g) if "global" in kinds else None

        def prepare(rt, sids):
            a = None
            for s in sids:
                rt.open(s, MAX_CTX, W_SPEC)
                if s == 4:
                    if blob is not None:
                        rt.import_state(4, blob, g)
                    a = rt.rebuild_from(4, F)
                else:
                    _play(rt, s, hist[s])
            return a

        mixed = _mk(f, lo, hi, head, tail)
        a = prepare(mixed, (1, 2, 3, 4))
        assert a < F, "rebuild segment needed for this check"
        final[4] = _seg(4, a, xs[4].slice(a, F), "rebuild")
        outs = mixed.step(StepIn(tuple(final[s] for s in (1, 2, 3, 4)))).outs
        assert [o.sid for o in outs] == [1, 2, 3, 4]
        for s, o in zip((1, 2, 3, 4), outs):
            solo = _mk(f, lo, hi, head, tail)
            prepare(solo, (s,))
            ref = _one(solo, final[s])
            assert o.hidden == ref.hidden and o.logits == ref.logits, f"session {s} differs in a mixed step"
            assert o.hidden.n == final[s].n
            want_lg = {"none": None, "last": 1, "all": final[s].n}[final[s].logits]
            assert (o.logits is None) if want_lg is None else (o.logits.n == want_lg)
            assert dict(o.taps).keys() == dict(ref.taps).keys()
            assert all(o.taps[t] == ref.taps[t] for t in o.taps)
            nxt = {1: 6, 2: 9, 3: 6, 4: F}[s]
            c1 = _one(mixed, _seg(s, nxt, xs[s].slice(nxt, nxt + 1)))
            c2 = _one(solo, _seg(s, nxt, xs[s].slice(nxt, nxt + 1)))
            assert c1.hidden == c2.hidden, f"session {s} diverges after the mixed step"


def check_frontier(f):
    """truncate/commit at F: commit is monotone and may run ahead of length; truncate below F
    or past length is refused; after truncate(F) the old length is a gap; export stops at the
    frontier; re-feeding F equals a twin that never saw the speculative rows."""
    for lo, hi, head, tail in _roles(f)[:3]:
        x = _inp(f, head, 40, 31)
        spec = _inp(f, head, 4, 32)
        rt = _mk(f, lo, hi, head, tail)
        rt.open(1, MAX_CTX, W_SPEC)
        rt.commit(1, 9)                                  # the prompt rows are final up front
        _one(rt, _seg(1, 0, x.slice(0, 10)))
        rt.commit(1, 5)                                  # a lower commit is a no-op
        assert rt.info(1).committed == 9
        _one(rt, _seg(1, 9, spec, "verify"))              # implicit truncate to 9, rows 9..12 speculative
        assert rt.length(1) == 13
        _raises(RollbackPastCommit, rt.truncate, 1, 8)
        _raises(PositionGap, rt.truncate, 1, 14)
        _raises(BadRequest, rt.export_state, 1, 0, 10)   # row 9 is past the frontier
        rt.truncate(1, 9)
        assert rt.length(1) == 9, "truncate(F) did not move the length"
        _raises(PositionGap, rt.step, StepIn((_seg(1, 13, x.slice(13, 14)),)))   # stale frame
        _raises(PositionGap, rt.step, StepIn((_seg(1, 10, x.slice(10, 11)),)))
        a = _one(rt, _seg(1, 9, x.slice(9, 11), "verify"))
        # Same segment shapes, every row at or past F with other values: nothing may differ.
        twin = _mk(f, lo, hi, head, tail)
        twin.open(1, MAX_CTX, W_SPEC)
        twin.commit(1, 9)
        _one(twin, _seg(1, 0, Rows.concat([x.slice(0, 9), f.perturb(x.slice(9, 10), 33)])))
        _one(twin, _seg(1, 9, f.perturb(spec, 34), "verify"))
        twin.truncate(1, 9)
        b = _one(twin, _seg(1, 9, x.slice(9, 11), "verify"))
        assert a.hidden == b.hidden, "rows after truncate(F) remember the speculative rows"
        if "global" in f.layer_kinds[lo:hi]:
            assert rt.export_state(1, 0, 9) == twin.export_state(1, 0, 9)
        if rt.describe().chunk_invariant:                # and a twin that never speculated
            clean = _mk(f, lo, hi, head, tail)
            clean.open(1, MAX_CTX, W_SPEC)
            clean.commit(1, 9)
            _one(clean, _seg(1, 0, x.slice(0, 9)))
            assert _one(clean, _seg(1, 9, x.slice(9, 11), "verify")).hidden == a.hidden, \
                "rows after truncate(F) differ from a run that never speculated"
        rt.commit(1, 40)                                 # commit ahead: a long chunk is legal
        _one(rt, _seg(1, 11, x.slice(11, 40)))
        _raises(SpecWindowExceeded, rt.step, StepIn((_seg(1, 40, _inp(f, head, W_SPEC + 1, 3)),)))


def check_swa_retention(f):
    """SWA window + W slack: a pipelined-spec history full of rejections (a whole W rejected,
    rollback to the frontier, re-feed) keeps every row it reads. Kept rows and the continuation
    are byte-identical to a twin whose rejected rows had other values; with chunk invariance,
    also to a clean run that never saw a rejected row."""
    spans = [r for r in _roles(f) if "swa" in f.layer_kinds[r[0]:r[1]]]
    if not spans:
        raise Skip("model has no SWA layers")
    for lo, hi, head, tail in spans:
        rt0 = _mk(f, lo, hi, head, tail)
        lg = "all" if tail else "none"
        prefix = 2 * f.window + 3
        res = []
        for twin in (False, True):
            ops, true, c = _spec_script(f, head, seed=41, prefix=prefix, rounds=14, W=W_SPEC, twin=twin)
            rt = _mk(f, lo, hi, head, tail)
            rt.open(1, MAX_CTX, W_SPEC)
            kept = _accepted(_play(rt, 1, ops, logits=lg))
            kept.update(_continue(rt, 1, true, c, logits=lg))
            res.append(kept)
        assert res[0] == res[1], "a rejected row's value leaked into kept rows or the continuation"
        if rt0.describe().chunk_invariant:
            clean = _mk(f, lo, hi, head, tail)
            clean.open(1, MAX_CTX, W_SPEC)
            end = max(res[0]) + 1
            clean.commit(1, end)
            outs, p, rng = [], 0, random.Random(3)
            while p < end:
                n = min(rng.randint(1, 9), end - p)
                outs.append((p, n, _one(clean, _seg(1, p, true.slice(p, p + n), logits=lg))))
                p += n
            assert _accepted(outs) == res[0], "speculation history changed committed bytes"


def check_row_independence(f):
    """A row's bytes (hidden, logits, taps) never depend on the VALUES of later rows in its
    segment (finite perturbations, not NaN), and truncating those rows leaves no trace."""
    for lo, hi, head, tail in _roles(f):
        rt0 = _mk(f, lo, hi, head, tail)
        _requires_invariant(rt0)
        lg = "all" if tail else "none"
        x = _inp(f, head, 30, 51)
        P0, K = 9, W_SPEC

        def run(chunk, j):
            rt = _mk(f, lo, hi, head, tail)
            rt.open(1, MAX_CTX, W_SPEC)
            rt.commit(1, P0)
            _one(rt, _seg(1, 0, x.slice(0, P0)))
            o = _one(rt, _seg(1, P0, chunk, "verify", logits=lg))
            rt.truncate(1, P0 + j + 1)
            rt.commit(1, P0 + j + 1)
            c = _one(rt, _seg(1, P0 + j + 1, x.slice(P0 + j + 1, P0 + j + 3), "verify", logits=lg))
            return o, c

        base = x.slice(P0, P0 + K)
        for j in range(K - 1):
            pert = Rows.concat([base.slice(0, j + 1), f.perturb(base.slice(j + 1, K), 60 + j)])
            (o1, c1), (o2, c2) = run(base, j), run(pert, j)
            for r in range(j + 1):
                assert _row_sig(o1, r) == _row_sig(o2, r), f"row {r} depends on later rows (j={j})"
            assert c1.hidden == c2.hidden and c1.logits == c2.logits, f"rejected rows left a trace (j={j})"


def _histories(f, head, sids, seed):
    out = {}
    for s in sids:
        ops, true, c = _spec_script(f, head, seed=seed + s, prefix=5 + s, rounds=4 + s, W=W_SPEC)
        out[s] = (ops, true, c)
    return out


def check_batch_invariance(f):
    """A segment's bytes do not depend on which segments share its step, their sizes or their
    order; every batch-mate also equals its solo run."""
    for lo, hi, head, tail in (_roles(f)[1], _roles(f)[2]):
        rt0 = _mk(f, lo, hi, head, tail)
        _requires_invariant(rt0)
        lg = "all" if tail else "none"
        hs = _histories(f, head, (1, 2, 3), seed=70)
        big = 4
        comps = [(1,), (1, 2), (2, 1), (3, 1, 2), (big, 1), (2, big, 3, 1)]
        seen = {}
        for comp in comps:
            rt = _mk(f, lo, hi, head, tail)
            segs = []
            for s in comp:
                rt.open(s, MAX_CTX, W_SPEC)
                if s == big:
                    n = min(40, rt.describe().max_tokens_per_step - 3 * W_SPEC)
                    rt.commit(s, n)
                    segs.append(_seg(s, 0, _inp(f, head, n, 99), "prefill", logits=lg))
                else:
                    ops, true, c = hs[s]
                    _play(rt, s, ops)
                    segs.append(_seg(s, c, true.slice(c, c + 3), "verify", logits=lg))
            for s, o in zip(comp, rt.step(StepIn(tuple(segs))).outs):
                sig = [_row_sig(o, r) for r in range(o.hidden.n)]
                if s in seen:
                    assert seen[s] == sig, f"session {s} changes with its batch-mates {comp}"
                seen[s] = sig


def check_replay(f):
    """Replaying the recorded schedule reproduces every byte: two fresh runtimes driven through
    the same randomly batched multi-session history agree, and so does a third that runs each
    session alone (batch invariance under random mixes)."""
    for lo, hi, head, tail in (_roles(f)[0], _roles(f)[2]):
        rt0 = _mk(f, lo, hi, head, tail)
        _requires_invariant(rt0)
        lg = "all" if tail else "none"
        hs = _histories(f, head, (1, 2, 3), seed=80)
        rng = random.Random(81)
        order = []                                    # batches of (sid, op index)
        idx = {s: 0 for s in hs}
        while any(idx[s] < len(hs[s][0]) for s in hs):
            live = [s for s in hs if idx[s] < len(hs[s][0])]
            batch = []
            for s in rng.sample(live, rng.randint(1, len(live))):
                while idx[s] < len(hs[s][0]) and hs[s][0][idx[s]][0] != "seg":
                    batch.append((s, idx[s]))
                    idx[s] += 1
                if idx[s] < len(hs[s][0]):
                    batch.append((s, idx[s]))
                    idx[s] += 1
            order.append(batch)

        def batched():
            rt = _mk(f, lo, hi, head, tail)
            for s in hs:
                rt.open(s, MAX_CTX, W_SPEC)
            res = {s: [] for s in hs}
            for batch in order:
                segs = []
                for s, i in batch:
                    op = hs[s][0][i]
                    if op[0] == "commit":
                        rt.commit(s, op[1])
                    elif op[0] == "truncate":
                        rt.truncate(s, op[1])
                    else:
                        segs.append((s, op, _seg(s, op[1], op[2], op[3], logits=lg)))
                if segs:
                    for (s, op, _), o in zip(segs, rt.step(StepIn(tuple(g for *_, g in segs))).outs):
                        res[s].append([_row_sig(o, r) for r in range(o.hidden.n)])
            return res

        r1, r2 = batched(), batched()
        assert r1 == r2, "the same batched schedule gave different bytes on a fresh runtime"
        solo = _mk(f, lo, hi, head, tail)
        for s in hs:
            solo.open(s, MAX_CTX, W_SPEC)
            got = [[_row_sig(o, r) for r in range(o.hidden.n)] for _, _, o in _play(solo, s, hs[s][0], lg)]
            assert got == r1[s], f"session {s} differs between a random batched schedule and solo"


def check_chunk_invariance(f):
    """Only if describe() claims chunk invariance: per-token, one-chunk and random chunkings of
    the same rows give identical hidden/logits/taps and identical exported KV."""
    for lo, hi, head, tail in _roles(f):
        rt0 = _mk(f, lo, hi, head, tail)
        if not rt0.describe().chunk_invariant:
            raise Skip("runtime does not claim chunk invariance (rebuild checks fall back to replay-equality)")
        lg = "all" if tail else "none"
        N = 22
        x = _inp(f, head, N, 91)
        rng = random.Random(92)
        rnd, p = [], 0
        while p < N:
            n = min(rng.randint(1, 7), N - p)
            rnd.append((p, n))
            p += n
        res = []
        for sched in ([(p, 1) for p in range(N)], [(0, N)], rnd):
            rt = _mk(f, lo, hi, head, tail)
            rt.open(1, MAX_CTX, W_SPEC)
            rt.commit(1, N)
            outs = [(s, n, _one(rt, _seg(1, s, x.slice(s, s + n), logits=lg))) for s, n in sched]
            blob = rt.export_state(1, 0, N) if "global" in f.layer_kinds[lo:hi] else None
            res.append((_accepted(outs), blob))
        assert res[0][0] == res[1][0] == res[2][0], "bytes depend on chunking despite chunk_invariant"
        assert res[0][1] == res[1][1] == res[2][1], "KV depends on chunking despite chunk_invariant"


def check_oracle(f):
    """Fault-free outputs match the factory's reference model (hidden, logits, raw taps)."""
    for lo, hi, head, tail in _roles(f):
        rt = _mk(f, lo, hi, head, tail)
        N = 20
        x = _inp(f, head, N, 101)
        sched = [(0, 7), (7, 1), (8, 5), (13, 7)]
        ref = f.oracle(lo, hi, is_head=head, is_tail=tail, x=x, schedule=sched)
        if ref is None:
            raise Skip("factory has no oracle")
        rt.open(1, MAX_CTX, W_SPEC)
        rt.commit(1, N)
        outs = [_one(rt, _seg(1, s, x.slice(s, s + n), logits="all" if tail else "none")) for s, n in sched]
        assert f.close_enough(_cat(outs, "hidden"), ref["hidden"], "oracle"), "hidden != oracle"
        if tail:
            assert f.close_enough(_cat(outs, "logits"), ref["logits"], "oracle"), "logits != oracle"
        for t, want in ref["taps"].items():
            got = Rows.concat([o.taps[t] for o in outs])
            assert f.close_enough(got, want, "oracle"), f"tap {t} != oracle"


def _chain(f, cuts, x, sched, tap_mode="raw"):
    """Run [0, L) as a chain of stages split at `cuts`, segment by segment."""
    L = len(f.layer_kinds)
    bounds = [0, *cuts, L]
    rts = []
    for i in range(len(bounds) - 1):
        rt = _mk(f, bounds[i], bounds[i + 1], i == 0, i == len(bounds) - 2, tap_mode=tap_mode)
        rt.open(1, MAX_CTX, W_SPEC)
        rt.commit(1, x.n)
        rts.append(rt)
    finals, taps = [], {}
    for s, n in sched:
        h, acc = x.slice(s, s + n), None
        for i, rt in enumerate(rts):
            last = i == len(rts) - 1
            o = _one(rt, Segment(1, s, n, h, kind="prefill", tap_in=acc, logits="all" if last else "none"))
            h, acc = o.hidden, o.tap_acc
            for t, r in o.taps.items():
                taps.setdefault(t, []).append(r)
        finals.append(o)
    return finals, {t: Rows.concat(v) for t, v in taps.items()}


def check_partition_invariance(f):
    """Where the stage cuts fall never changes the bytes: a chain of stages equals one stage
    holding every layer (hidden, logits, raw taps)."""
    L = len(f.layer_kinds)
    if L < 3:
        raise Skip("model too small to partition")
    _requires_invariant(_mk(f, 0, L, True, True))
    x = f.ids(18, 111)
    sched = [(0, 6), (6, 1), (6 + 1, 4), (11, 7)]
    whole, wt = _chain(f, [], x, sched)
    for cuts in ([L // 3, 2 * L // 3], [1, L - 1], [L // 2]):
        part, pt = _chain(f, cuts, x, sched)
        assert all(a.hidden == b.hidden and a.logits == b.logits for a, b in zip(whole, part)), cuts
        assert wt.keys() == pt.keys() and all(wt[t] == pt[t] for t in wt), cuts


def check_taps_raw(f):
    """Raw taps: exactly the held tap layers, tap L == the residual stream after layer L (the
    hidden output of a stage cut after L); taps=False captures nothing and never changes the
    hidden bytes."""
    if not f.taps:
        raise Skip("model has no drafter taps")
    for lo, hi, head, tail in _roles(f):
        held = sorted(t for t in set(f.taps) if lo <= t < hi)
        x = _inp(f, head, 12, 121)
        sched = [(0, 5), (5, 1), (6, 6)]

        def run(rt, taps=True):
            rt.open(1, MAX_CTX, W_SPEC)
            rt.commit(1, x.n)
            return [_one(rt, Segment(1, s, n, x.slice(s, s + n), kind="prefill", taps=taps)) for s, n in sched]

        outs = run(_mk(f, lo, hi, head, tail))
        for o in outs:
            assert sorted(o.taps) == held, f"taps {sorted(o.taps)} != held {held}"
            assert o.tap_acc is None
        off = run(_mk(f, lo, hi, head, tail), taps=False)
        assert all(not o.taps for o in off)
        assert _cat(off, "hidden") == _cat(outs, "hidden"), "capturing taps changed the hidden bytes"
        for t in held:
            cut = run(_mk(f, lo, t + 1, head, False))
            assert Rows.concat([o.taps[t] for o in outs]) == _cat(cut, "hidden"), \
                f"tap {t} is not the output of layer {t}"


def check_taps_fc_partial(f):
    """fc_partial: the tail's accumulator equals the reference fc(concat(raw taps)); it does not
    depend on where the chain is cut; a stage holding no tap passes tap_in through unchanged;
    the tap lane never changes the hidden bytes."""
    if not f.taps:
        raise Skip("model has no drafter taps")
    L = len(f.layer_kinds)
    x = f.ids(14, 131)
    sched = [(0, 6), (6, 1), (7, 7)]
    raw, rtaps = _chain(f, [], x, sched)
    ref = f.fuse_taps(rtaps)
    for cuts in ([], [L // 3, 2 * L // 3], [1, L // 2, L - 1]):
        fused, _ = _chain(f, cuts, x, sched, tap_mode="fc_partial")
        acc = _cat(fused, "tap_acc")
        assert acc is not None and f.close_enough(acc, ref, "tap_fusion"), f"fused taps != fc(concat) at cuts {cuts}"
        assert _cat(fused, "hidden") == _cat(raw, "hidden") and _cat(fused, "logits") == _cat(raw, "logits")
    quiet = next(((lo, lo + 2) for lo in range(1, L - 2) if not set(f.taps) & {lo, lo + 1}), None)
    if quiet:
        rt = _mk(f, quiet[0], quiet[1], False, False, tap_mode="fc_partial")
        rt.open(1, MAX_CTX, W_SPEC)
        h, t = f.hidden(3, 132), f.hidden(3, 133)
        o = _one(rt, Segment(1, 0, 3, h, kind="prefill", tap_in=t))
        assert o.tap_acc == t, "a stage without taps altered the tap accumulator"


def check_export_import(f):
    """Mirroring: exports stop at the frontier and only cover global layers; a blob survives
    the wire; contiguous chunked imports reproduce the primary's rows exactly and a rebuild
    never alters them; wrong layout / order / target / timing are refused."""
    spans = [r for r in _roles(f) if "global" in f.layer_kinds[r[0]:r[1]]]
    if not spans:
        raise Skip("model has no global-attention layers to mirror")
    for lo, hi, head, tail in spans[:3]:
        prim = _mk(f, lo, hi, head, tail)
        prim.open(1, MAX_CTX, W_SPEC)
        ops, true, c = _spec_script(f, head, seed=141, prefix=6, rounds=8, W=W_SPEC)
        _play(prim, 1, ops)
        _one(prim, _seg(1, c, _inp(f, head, 2, 142), "verify"))       # speculative rows past F
        _raises(BadRequest, prim.export_state, 1, 0, c + 1)
        _raises(BadRequest, prim.export_state, 1, 0, c, kinds=("swa",))
        _raises(BadRequest, prim.export_state, 1, 3, 2)
        blob = prim.export_state(1, 0, c)
        assert KVBlob.from_bytes(blob.to_bytes()) == blob
        _raises(BadRequest, KVBlob.from_bytes, blob.to_bytes()[:-1])
        assert all(blob.pos_bytes(p) == b"".join(r.row_bytes(p) for r in blob.rows) for p in range(c))
        k = c // 2
        b1, b2 = prim.export_state(1, 0, k), prim.export_state(1, k, c)
        bk = _mk(f, lo, hi, head, tail)
        bk.open(1, MAX_CTX, W_SPEC)
        _raises(BadRequest, bk.import_state, 1, b2, c)                # must start at 0
        bk.import_state(1, b1, k)
        _raises(BadRequest, bk.step, StepIn((_seg(1, 0, true.slice(0, 1)),)))   # rebuild_from first
        _raises(BadRequest, bk.export_state, 1, 0, 1)
        bk.import_state(1, b2, c)
        a = bk.rebuild_from(1, c)
        p = a
        while p < c:
            n = min(5, c - p)
            _one(bk, _seg(1, p, true.slice(p, p + n), "rebuild"))
            p += n
        assert bk.export_state(1, 0, c) == blob, "import + rebuild did not reproduce the mirrored rows"
        used = _mk(f, lo, hi, head, tail)
        used.open(1, MAX_CTX, W_SPEC)
        used.commit(1, 2)
        _one(used, _seg(1, 0, true.slice(0, 2)))
        _raises(BadRequest, used.import_state, 1, blob, c)            # not a fresh session
        fresh = _mk(f, lo, hi, head, tail)
        fresh.open(1, MAX_CTX, W_SPEC)
        _raises(BadRequest, fresh.import_state, 1, blob, c + 1)       # upto past the blob
        other = next(((l2, h2, hd, tl) for l2, h2, hd, tl in spans
                      if [i for i in range(l2, h2) if f.layer_kinds[i] == "global"]
                      != [i for i in range(lo, hi) if f.layer_kinds[i] == "global"]), None)
        if other:
            o = _mk(f, other[0], other[1], other[2], other[3])
            o.open(1, MAX_CTX, W_SPEC)
            _raises(BadRequest, o.import_state, 1, blob, c)          # another span's layout


def _rebuild_case(f, lo, hi, head, tail, g_of, seed):
    """Primary with a speculative history commits F and mirrors [0, g); backups import, rebuild
    [a, F) from the predecessor log (the committed input rows) and continue. Returns per-g
    results for the caller's equality policy."""
    kinds = tuple(f.layer_kinds[lo:hi])
    n_swa = kinds.count("swa")
    lg = "all" if tail else "none"
    prefix = 2 * (f.window - 1) * max(n_swa, 1) + 6
    ops, true, F = _spec_script(f, head, seed=seed, prefix=prefix, rounds=9, W=W_SPEC)
    prim = _mk(f, lo, hi, head, tail)
    prim.open(1, MAX_CTX, W_SPEC)
    _play(prim, 1, ops)
    _one(prim, _seg(1, F, _inp(f, head, 3, seed + 1), "verify"))      # uncommitted rows past F
    has_g = "global" in kinds
    ref_kv = prim.export_state(1, 0, F) if has_g else None
    ref_cont = _continue(prim, 1, true, F, logits=lg)
    out = []
    for g in sorted({g for g in g_of(F) if 0 <= g <= F}) if has_g else [0]:
        blob = _blob_prefix(ref_kv, g) if has_g else None     # what the primary had mirrored
        runs = []
        for chunk in (5, 5, 3):
            bk = _mk(f, lo, hi, head, tail)
            bk.open(1, MAX_CTX, W_SPEC)
            if has_g:
                bk.import_state(1, blob, g)
            a = bk.rebuild_from(1, F)
            info = bk.info(1)
            assert info.length == a and info.committed >= F and info.rebuild_hi == (F if a < F else None)
            p = a
            while p < F:
                n = min(chunk, F - p)
                _one(bk, _seg(1, p, true.slice(p, p + n), "rebuild"))
                p += n
            assert bk.length(1) == F and bk.info(1).rebuild_hi is None
            mirror_ok = (not has_g) or g == 0 or bk.export_state(1, 0, g) == blob
            kv = bk.export_state(1, 0, F) if has_g else None
            runs.append((mirror_ok, kv, _continue(bk, 1, true, F, logits=lg), a))
        out.append((g, F, runs, rebuild_start(kinds, f.window, F, g)))
    return ref_kv, ref_cont, out


def _blob_prefix(blob: KVBlob, g: int) -> KVBlob:
    return KVBlob(0, g, blob.layers, tuple(r.slice(0, g) for r in blob.rows), blob.layout_id, blob.numerics_id)


def check_rebuild(f):
    """The takeover rebuild (§4.1 rule) for span patterns SSG, SGS, SSSSSG, SGSGS, GSS, SSS,
    GG, with g_eff = F, just below F, well below F and 0: mirrored rows are never
    overwritten; with chunk invariance the continuation and the KV are bit-equal to the
    fault-free primary (whatever the rebuild chunking); otherwise two backups with the same
    schedule are replay-equal."""
    L = len(f.layer_kinds)
    cases = [(pat, _find(f, pat)) for pat in REBUILD_PATTERNS]
    cases = [(pat, sp) for pat, sp in cases if sp] + [("whole", (0, L))]
    ci = _mk(f, 0, L, True, True).describe().chunk_invariant
    w1 = f.window - 1

    def g_of(F):
        return [F, F - 1, F - w1 - 2, F - 3 * w1 - 5, w1 + 1, 0]

    for i, (pat, (lo, hi)) in enumerate(cases):
        head, tail = lo == 0, hi == L
        ref_kv, ref_cont, out = _rebuild_case(f, lo, hi, head, tail, g_of, seed=150 + i)
        for g, F, runs, rule in out:
            where = f"span {pat} [{lo},{hi}) F={F} g={g}"
            (_, kv_a, c_a, _), (_, kv_b, c_b, _), (_, kv_c, c_c, _) = runs
            # Behaviour first: a wrong rebuild must show up as wrong state, not only as a formula.
            if ci:
                assert c_a == ref_cont and c_c == ref_cont, f"continuation after rebuild != fault-free ({where})"
                assert kv_a == ref_kv and kv_c == ref_kv, f"KV after rebuild != fault-free ({where})"
            assert kv_a == kv_b and c_a == c_b, f"same rebuild schedule, different bytes ({where})"
            for mirror_ok, _, _, a in runs:
                assert mirror_ok, f"rebuild overwrote mirrored global rows ({where})"
                # Replay-equality would replay a late start too, so the rule is checked directly.
                assert 0 <= a <= rule, f"rebuild_from returned a={a}, the rule needs a <= {rule} ({where})"


def check_rebuild_protocol(f):
    """rebuild_from only on a fresh/imported session and with upto <= F; while rebuilding only
    in-order rebuild segments up to F (no logits, no truncate, no export); afterwards normal
    segments resume at F."""
    lo, hi = next((sp for sp in (_find(f, p) for p in ("SGS", "SSG", "GSS")) if sp),
                  _roles(f)[1][:2])
    L = len(f.layer_kinds)
    head, tail = lo == 0, hi == L
    kinds = f.layer_kinds[lo:hi]
    x = _inp(f, head, 40, 161)
    prim = _mk(f, lo, hi, head, tail)
    prim.open(1, MAX_CTX, W_SPEC)
    prim.commit(1, 30)
    _one(prim, _seg(1, 0, x.slice(0, 30)))
    _raises(BadRequest, prim.rebuild_from, 1, 30)                     # not fresh
    has_g = "global" in kinds
    g = 20
    rt = _mk(f, lo, hi, head, tail)
    rt.open(1, MAX_CTX, W_SPEC)
    if has_g:
        rt.import_state(1, prim.export_state(1, 0, g), g)
        _raises(BadRequest, rt.rebuild_from, 1, g - 1)                # F below the imported rows
    a = rt.rebuild_from(1, 30)
    assert a < 30
    _raises(BadRequest, rt.step, StepIn((_seg(1, a, x.slice(a, a + 1), "decode"),)))
    if a + 1 < 30:
        _raises((BadRequest, PositionGap), rt.step, StepIn((_seg(1, a + 1, x.slice(a + 1, a + 2), "rebuild"),)))
    _raises(BadRequest, rt.step, StepIn((_seg(1, a, x.slice(a, 31), "rebuild"),)))
    _raises((RollbackPastCommit, PositionGap), rt.truncate, 1, a)
    _raises(BadRequest, rt.export_state, 1, 0, 1)
    _one(rt, _seg(1, a, x.slice(a, 30), "rebuild"))
    _raises(BadRequest, rt.step, StepIn((_seg(1, 30, x.slice(30, 31), "rebuild"),)))
    _one(rt, _seg(1, 30, x.slice(30, 31)))
    if not tail:
        t = next((r for r in _roles(f) if r[3]), None)
        lo, hi, head, tail = t
        x = _inp(f, head, 40, 162)
    rt2 = _mk(f, lo, hi, head, tail)
    rt2.open(1, MAX_CTX, W_SPEC)
    a2 = rt2.rebuild_from(1, 30)
    if a2 < 30:
        _raises(BadRequest, rt2.step, StepIn((_seg(1, a2, x.slice(a2, 30), "rebuild", logits="last"),)))


def check_close_isolation(f):
    """close() drops one session: its batch-mate continues byte-identically to a runtime that
    never had it, and a reopened sid starts fresh."""
    lo, hi, head, tail = _roles(f)[1]
    hs = _histories(f, head, (1, 2), seed=170)
    rt = _mk(f, lo, hi, head, tail)
    for s in hs:
        rt.open(s, MAX_CTX, W_SPEC)
    ops1, ops2 = hs[1][0], hs[2][0]
    _play(rt, 1, ops1[: len(ops1) // 2])
    rt.close(1)
    got = _accepted(_play(rt, 2, ops2))
    alone = _mk(f, lo, hi, head, tail)
    alone.open(2, MAX_CTX, W_SPEC)
    assert got == _accepted(_play(alone, 2, ops2)), "closing a session disturbed another"
    rt.open(1, MAX_CTX, W_SPEC)
    assert rt.length(1) == 0 and rt.info(1).committed == 0
    fresh = _mk(f, lo, hi, head, tail)
    fresh.open(1, MAX_CTX, W_SPEC)
    assert _accepted(_play(rt, 1, ops1)) == _accepted(_play(fresh, 1, ops1)), "a reopened sid kept old state"


def check_load_refusal(f):
    """load() refuses a weight source missing a tensor of the span (never a random init)."""
    lo, hi, _, _ = _roles(f)[1]
    make_broken = getattr(f, "make_broken", None)
    if make_broken is None:
        raise Skip("factory cannot build a broken weight source")
    _raises(LoadRefused, make_broken, lo, hi)


CHECKS = (check_describe, check_admission, check_validation_atomic, check_mixed_kinds,
          check_frontier, check_swa_retention, check_row_independence, check_batch_invariance,
          check_replay, check_chunk_invariance, check_oracle, check_partition_invariance,
          check_taps_raw, check_taps_fc_partial, check_export_import, check_rebuild,
          check_rebuild_protocol, check_close_isolation, check_load_refusal)


def run_suite(f) -> dict:
    """{check name: "pass" | "skip" | "FAIL: <error>"} (used for the mutant matrix)."""
    res = {}
    for chk in CHECKS:
        try:
            chk(f)
            res[chk.__name__] = "pass"
        except Skip:
            res[chk.__name__] = "skip"
        except (AssertionError, StageRuntimeError, Exception) as e:   # noqa: BLE001
            res[chk.__name__] = f"FAIL: {type(e).__name__}: {e}"
    return res
