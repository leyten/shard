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

tests/test_runtime_conformance.py runs it on ToyRuntime in several geometries and both numeric
modes, and on mutant runtimes, each of which must fail the checks designed to catch it.

Two kinds of comparison:
  * Self-consistency, always BYTES: batch invariance, row independence, replay, partition
    invariance, twins that differ only in rejected rows. Receipts and takeovers rely on them.
  * Against the factory's REFERENCE runtime (an independent implementation of the same model,
    e.g. the CPU MiMo oracle for a CUDA backend), driven through the same calls, including
    rejections, mirror import and rebuild: every output row, every token and every exported
    KV byte. `close_enough(a, b, "reference")` may allow a tolerance for a float backend.
    This is what catches a wrong rebuild or a wrong KV export in a backend whose bytes depend
    on chunking (replay-equality alone would replay the bug).

The factory DECLARES the backend's numerics mode and whether it is chunk invariant; the runtime
must report the same in describe() (a runtime cannot exempt itself from checks). Checks that
need invariant numerics skip for a factory that declares "fast"; bit-equality with the
fault-free primary after a rebuild is required when the factory declares chunk invariance.
The suite needs max_tokens_per_step >= 32 and splits every long prefill into 16-row chunks.

Terms: W = the spec window a session is opened with (factory.W; some checks also use 1 and
16); w = the SWA window; F = the committed (final-row) frontier; "twin" = a fresh runtime
driven through the same history.
"""
from __future__ import annotations

import json
import random
import time
from typing import Callable, Mapping, Protocol

import numpy as np

from shard.runtime.api import (BACKENDS, BadRequest, Capability, KVBlob, KVFull, LoadRefused,
                               PositionGap, RollbackPastCommit, Rows, Sampling, Segment,
                               SessionExists, SpecWindowExceeded, StageRuntime,
                               StageRuntimeError, StepIn, StepTooLarge, UnknownSession, WTooDeep,
                               rebuild_exact_from, rebuild_start)

REBUILD_PATTERNS = ("SSG", "SGS", "SSSSSG", "SGSGS", "GSS", "SSS", "GG")
PREFILL_CHUNK = 16
# Session ids are opaque u64s. 9 and 2**33 + 9 collide modulo every power of two up to 2**33
# (a slot table, a 32-bit key); both are open together wherever SIDS[:3] are.
SIDS = (0, 9, (1 << 33) + 9, (1 << 64) - 1)
SAMPLE = Sampling(seed=1234, temperature=1.0, top_k=4)


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
    W: int                                # spec window sessions are opened with
    max_ctx: int                          # max_pos sessions are opened with
    numerics_mode: str                    # declared: "invariant" | "fast"
    chunk_invariant: bool                 # declared

    def make(self, lo: int, hi: int, *, is_head: bool = False, is_tail: bool = False,
             tap_mode: str = "raw", spec_window: int | None = None, max_sessions: int = 8,
             max_ctx: int | None = None, kv_budget_bytes: int = 0) -> StageRuntime: ...

    def reference(self, lo: int, hi: int, *, is_head: bool = False, is_tail: bool = False,
                  tap_mode: str = "raw"):
        """An independent implementation of the same stage (open, close, length, commit,
        truncate, step, export_state, import_state, rebuild_from, embed, project), or None."""

    def hidden(self, n: int, seed: int) -> Rows: ...

    def ids(self, n: int, seed: int) -> Rows: ...

    def perturb(self, rows: Rows, seed: int) -> Rows: ...

    def fuse_taps(self, raw: Mapping[int, Rows]) -> Rows:
        """Reference fc(concat(taps)) over self.taps, in the stage's tap wire dtype."""

    def close_enough(self, a: Rows, b: Rows, what: str) -> bool:
        """what in ("reference", "tap_fusion"); exact for an integer backend."""

    def make_broken(self, lo: int, hi: int) -> None:
        """Build + load a runtime whose weight source lacks one tensor (must raise LoadRefused)."""

    # Optional: make_unloaded(lo, hi, *, is_head, is_tail) -> (runtime, WeightSource), so
    # describe() can be checked before load(); kv_bytes_per_pos(lo, hi) -> (global, swa) bytes
    # of one position from the model's geometry, so describe() cannot under-report KV.


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


def _role(f, which):
    r = _roles(f)
    return {"head": r[0], "mid": r[1] if len(r) == 4 else r[0], "tail": r[-2], "whole": r[-1]}[which]


def _inp(f, head: bool, n: int, seed: int) -> Rows:
    return f.ids(n, seed) if head else f.hidden(n, seed)


def _one(rt, seg: Segment):
    return rt.step(StepIn((seg,))).outs[0]


def _seg(sid, start, x, kind=None, **kw) -> Segment:
    return Segment(sid, start, x.n, x, kind=kind or ("decode" if x.n == 1 else "prefill"), **kw)


def _mk(f, lo, hi, head, tail, **kw):
    return f.make(lo, hi, is_head=head, is_tail=tail, **kw)


def _ref(f, lo, hi, head, tail, **kw):
    return f.reference(lo, hi, is_head=head, is_tail=tail, **kw)


def _tailkw(tail: bool) -> dict:
    return dict(logits="all", tokens="all", sampling=SAMPLE) if tail else {}


def _open(rt, sid, f, **kw):
    rt.open(sid, kw.get("max_pos", f.max_ctx), kw.get("W", f.W))


def _prefill_ops(x: Rows, lo: int, hi: int) -> list:
    """Committed rows [lo, hi) of stream x as prefill ops of <= PREFILL_CHUNK rows."""
    return [("seg", p, x.slice(p, min(p + PREFILL_CHUNK, hi)), "prefill", min(PREFILL_CHUNK, hi - p))
            for p in range(lo, hi, PREFILL_CHUNK)]


def _play(rt, sid, ops, **kw):
    """Apply a session script; returns (start, keep, SegOut) for every "seg" op.
    ops: ("commit", pos) | ("truncate", pos) | ("seg", start, rows, kind, keep)."""
    outs = []
    for op in ops:
        if op[0] == "commit":
            rt.commit(sid, op[1])
        elif op[0] == "truncate":
            rt.truncate(sid, op[1])
        else:
            _, start, x, kind, keep = op
            outs.append((start, keep, _one(rt, _seg(sid, start, x, kind, **kw))))
    return outs


def _per_row(o, r: int) -> list:
    """(name, Rows of one row) for every per-row output of SegOut o at row r."""
    n = o.hidden.n
    out = [("hidden", o.hidden.slice(r, r + 1))]
    if o.logits is not None and o.logits.n == n:
        out.append(("logits", o.logits.slice(r, r + 1)))
    if o.tokens is not None and o.tokens.n == n:
        out.append(("tokens", o.tokens.slice(r, r + 1)))
    out += [(f"tap{t}", o.taps[t].slice(r, r + 1)) for t in sorted(o.taps)]
    if o.tap_acc is not None:
        out.append(("tap_acc", o.tap_acc.slice(r, r + 1)))
    return out


def _row_sig(o, r: int) -> tuple:
    return tuple((name, x.row_bytes(0)) for name, x in _per_row(o, r))


def _accepted(outs) -> dict:
    """position -> row signature, for the rows each seg op KEPT (accepted/committed)."""
    return {start + r: _row_sig(o, r) for start, keep, o in outs for r in range(keep)}


def _rows_of(outs, keep_only=True) -> dict:
    """key -> (SegOut, row): accepted rows by position, or every row by (op#, row)."""
    if keep_only:
        return {start + r: (o, r) for start, keep, o in outs for r in range(keep)}
    return {(i, r): (o, r) for i, (_, _, o) in enumerate(outs) for r in range(o.hidden.n)}


def _match_ref(f, got: dict, want: dict, where: str):
    assert got.keys() == want.keys(), f"{where}: different rows"
    for k in got:
        a, b = _per_row(*got[k]), _per_row(*want[k])
        assert [n for n, _ in a] == [n for n, _ in b], f"{where}: output kinds differ at {k}"
        for (n, x), (_, y) in zip(a, b):
            assert f.close_enough(x, y, "reference"), f"{where}: {n} at {k} != reference"


def _blob_close(f, a: KVBlob, b: KVBlob) -> bool:
    return ((a.lo_pos, a.hi_pos, a.layers) == (b.lo_pos, b.hi_pos, b.layers)
            and all(f.close_enough(x, y, "reference") for x, y in zip(a.rows, b.rows)))


def _spec_script(f, head, *, seed, prefix, rounds, W, twin=False):
    """A pipelined-speculation history: a committed prefill of `prefix` rows, then rounds of
    a k-row chunk at the frontier c (k <= W) whose first j rows are accepted: commit(c+j), and
    the next round re-feeds row c+j (implicit truncate via start_pos, or an explicit truncate
    as lease_update does). The plan starts with the deepest cases (a whole W rejected, a whole
    W accepted). `twin=True` gives the same script with every REJECTED row perturbed: shapes
    and accepted values are identical, so a correct runtime's kept rows must be byte-identical.
    Returns (ops, committed_stream, c_end)."""
    rng = random.Random(seed)
    fixed = [(W, 0), (W, W), (1, 0), (W, 1), (1, 1), (min(2, W), 0), (W, W - 1)]
    plan = fixed[:rounds] + [(k, rng.randint(0, k)) for k in
                             (rng.randint(1, W) for _ in range(max(0, rounds - len(fixed))))]
    true = _inp(f, head, prefix + W * len(plan) + 4 * W + 4, seed)
    ops = [("commit", prefix)] + _prefill_ops(true, 0, prefix)
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


def _continue(rt, sid, true, c, W, rounds=3, **kw):
    """Feed committed rows [c, c + rounds*W) in W-row verify chunks, committing each."""
    outs = []
    for i in range(rounds):
        p = c + i * W
        outs.append((p, W, _one(rt, _seg(sid, p, true.slice(p, p + W), "verify" if W > 1 else "decode", **kw))))
        rt.commit(sid, p + W)
    return outs


def _requires_invariant(f):
    if f.numerics_mode != "invariant":
        raise Skip("needs numerics_mode='invariant' (fast mode promises run-to-run determinism only)")


def _cat(outs, attr):
    parts = [getattr(o, attr) for o in outs]
    return None if any(p is None for p in parts) else Rows.concat(parts)


def _as_float(r: Rows) -> np.ndarray:
    if r.dtype == "bf16":
        return (r.data.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return r.data.astype(np.float64)


# ── the checks ────────────────────────────────────────────────────────────────────────────────

def check_describe(f):
    """describe() is a valid, JSON-portable, device-neutral record that matches the stage and
    the factory's declared numerics; warmup() measures a (tokens, context) cost grid,
    self-probes and leaves no session behind."""
    for lo, hi, head, tail in _roles(f):
        rt = _mk(f, lo, hi, head, tail)
        cap = rt.describe()
        assert isinstance(cap, Capability)
        assert cap.numerics_mode == f.numerics_mode, \
            f"describe() says numerics_mode={cap.numerics_mode!r}, the factory declares {f.numerics_mode!r}"
        assert cap.chunk_invariant == f.chunk_invariant, \
            f"describe() says chunk_invariant={cap.chunk_invariant}, the factory declares {f.chunk_invariant}"
        assert (cap.lo, cap.hi, cap.is_head, cap.is_tail) == (lo, hi, head, tail)
        assert cap.layer_kinds == tuple(f.layer_kinds[lo:hi])
        assert cap.swa_window == (f.window if "swa" in cap.layer_kinds else 0)
        assert cap.rebuild_positions == (f.window - 1) * cap.n_swa
        assert cap.device.backend in BACKENDS and cap.device.total_mem_bytes >= 0
        assert Capability.from_dict(json.loads(json.dumps(cap.to_dict()))) == cap
        assert cap.tap_layers_held == tuple(sorted(t for t in set(f.taps) if lo <= t < hi))
        assert cap.spec_window_max >= f.W and cap.max_tokens_per_step >= 32, "suite needs a 32-token step"
        assert cap.weight_bytes > 0 and cap.vocab_size > 0 and cap.hidden_size > 0
        assert len(cap.layer_weight_bytes) == hi - lo and min(cap.layer_weight_bytes) > 0
        geo = getattr(f, "kv_bytes_per_pos", None)
        if geo:
            g_b, s_b = geo(lo, hi)
            assert cap.global_kv_bytes_per_pos >= g_b and cap.swa_kv_bytes_per_pos >= s_b, \
                f"describe() under-reports KV: ({cap.global_kv_bytes_per_pos}, {cap.swa_kv_bytes_per_pos}) < ({g_b}, {s_b})"
        got = [cap.sessions_at(c, f.W) for c in (16, 64, f.max_ctx)]
        assert got == sorted(got, reverse=True) and got[-1] >= 1, got
        sid = SIDS[3]
        _open(rt, sid, f)
        rt.commit(sid, 3)
        o = _one(rt, _seg(sid, 0, _inp(f, head, 3, 5), **_tailkw(tail)))
        assert o.hidden.dtype == cap.wire_dtype and o.hidden.width == cap.hidden_size and o.hidden.n == 3
        if tail:
            assert o.logits.dtype == cap.logits_dtype and o.logits.width == cap.vocab_size
            assert o.tokens.dtype == "i32" and o.tokens.n == 3
            assert ((o.tokens.data >= 0) & (o.tokens.data < cap.vocab_size)).all()
        if "global" in cap.layer_kinds:
            blob = rt.export_state(sid, 0, 3)
            assert blob.layout_id == cap.kv_layout_id and all(r.dtype == cap.kv_dtype for r in blob.rows)
            assert cap.global_kv_bytes_per_pos >= len(blob.pos_bytes(0)), \
                "describe() reports fewer global KV bytes per position than one exported position holds"
        rt.close(sid)
        wcap = rt.warmup(buckets=(1, 4, 32), contexts=(0, 8), sessions=(1, 3))
        shapes = {(n, b, c) for n, b, c, _ in wcap.service_ms}
        assert shapes == {(n, b, c) for n in (1, 3) for b in (1, 4, 32) for c in (0, 8)
                          if n * b <= cap.max_tokens_per_step}, wcap.service_ms
        assert all(ms > 0 for *_, ms in wcap.service_ms)
        assert wcap.self_probe_ok is True and wcap.open_sessions == 0
        assert wcap.numerics_id == cap.numerics_id
        claimed = dict(((n, b, c), ms) for n, b, c, ms in wcap.service_ms)[(1, 32, 0)]
        best = float("inf")
        for i in range(3):                               # the curve must be a real measurement
            _open(rt, 100 + i, f)
            rt.commit(100 + i, 32)
            seg = _seg(100 + i, 0, _inp(f, head, 32, 6), logits="last" if tail else "none")
            t0 = time.perf_counter()
            _one(rt, seg)
            best = min(best, (time.perf_counter() - t0) * 1000)
            rt.close(100 + i)
        assert best / 10 <= claimed <= best * 10, f"service_ms says {claimed:.3f} ms, a real step takes {best:.3f} ms"
    unloaded = getattr(f, "make_unloaded", None)
    if unloaded:                                         # placement sizes spans before loading
        lo, hi, head, tail = _role(f, "tail")
        rt, src = unloaded(lo, hi, is_head=head, is_tail=tail)
        c0 = rt.describe()
        rt.load(src)
        c1 = rt.describe()
        assert (c0.layer_kinds, c0.kv_layout_id, c0.global_kv_bytes_per_pos) == \
            (c1.layer_kinds, c1.kv_layout_id, c1.global_kv_bytes_per_pos)
        assert abs(c0.weight_bytes - c1.weight_bytes) <= 0.1 * c1.weight_bytes, (c0.weight_bytes, c1.weight_bytes)


def check_admission(f):
    """open() reserves exactly what describe().sessions_at() predicts, one session at a time;
    KVFull, WTooDeep, SessionExists and UnknownSession fire; close() frees and is idempotent."""
    lo, hi, head, tail = _role(f, "mid")
    per = _mk(f, lo, hi, head, tail).describe().session_kv_bytes(64, f.W)
    rt = _mk(f, lo, hi, head, tail, kv_budget_bytes=3 * per + per // 2, max_sessions=8)
    want = rt.describe().sessions_at(64, f.W)
    assert want == 3, f"a pool of 3.5 sessions predicts {want}"
    sids = [SIDS[0], SIDS[1], SIDS[2], SIDS[3], 5]
    for i, sid in enumerate(sids[:want]):
        rt.open(sid, 64, f.W)
        cap = rt.describe()
        assert cap.open_sessions == i + 1 and cap.sessions_at(64, f.W) == want - i - 1, \
            f"after {i + 1} opens describe() says {cap.open_sessions} open, {cap.sessions_at(64, f.W)} free"
    _raises(KVFull, rt.open, sids[want], 64, f.W)
    rt.close(sids[0])
    rt.close(sids[0])                                             # idempotent
    assert rt.describe().sessions_at(64, f.W) == 1
    rt.open(sids[0], 64, f.W)
    rt2 = _mk(f, lo, hi, head, tail, max_sessions=2)
    cap2 = rt2.describe()
    rt2.open(SIDS[1], 64, f.W)
    rt2.open(SIDS[2], 64, f.W)
    _raises(KVFull, rt2.open, SIDS[3], 64, f.W)
    _raises(SessionExists, rt2.open, SIDS[1], 64, f.W)
    rt2.close(SIDS[2])
    _raises(WTooDeep, rt2.open, SIDS[2], 64, cap2.spec_window_max + 1)
    _raises(KVFull, rt2.open, SIDS[2], cap2.max_ctx + 1, f.W)
    rt2.close(999)
    u = SIDS[3]
    for call in (lambda: rt2.length(u), lambda: rt2.info(u), lambda: rt2.truncate(u, 0),
                 lambda: rt2.commit(u, 0), lambda: rt2.export_state(u, 0, 0),
                 lambda: rt2.rebuild_from(u, 0), lambda: _one(rt2, _seg(u, 0, _inp(f, head, 1, 1)))):
        _raises(UnknownSession, call)


def check_validation_atomic(f):
    """Every malformed segment is refused, for every segment kind, with the right error and the
    offending sid; a step mixing a valid segment with an invalid one changes NO session."""
    W = f.W
    for lo, hi, head, tail in (_role(f, "mid"), _role(f, "tail"), _role(f, "head")):
        rt = _mk(f, lo, hi, head, tail)
        cap = rt.describe()
        A, B, D, E, G = SIDS[1], SIDS[2], SIDS[3], SIDS[0], 77
        x = {s: _inp(f, head, 40, 11 + i) for i, s in enumerate((A, B, D, E, G))}
        vA = min(2, W)
        hist = {A: [("commit", 6), ("seg", 0, x[A].slice(0, 6), "prefill", 6),
                    ("seg", 6, x[A].slice(6, 6 + vA), "verify" if vA > 1 else "decode", vA)],
                B: [("commit", 5), ("seg", 0, x[B].slice(0, 5), "prefill", 5)],
                D: [("commit", 6), ("seg", 0, x[D].slice(0, 6), "prefill", 6),        # exactly at the edge:
                    ("seg", 6, x[D].slice(6, 6 + W), "verify" if W > 1 else "decode", W)],  # end == committed + W
                E: [("commit", 12)] + _prefill_ops(x[E], 0, 12),
                G: [("commit", 20)] + _prefill_ops(x[G], 0, 5)}              # committed ahead of length
        mp = {E: 12}

        def build(r):
            for s, ops in hist.items():
                r.open(s, mp.get(s, f.max_ctx), W)
                _play(r, s, ops)

        build(rt)
        LA = 6 + vA
        good = _seg(B, 5, x[B].slice(5, 6))
        seg_bad = []
        for kind in ("decode", "verify", "prefill"):
            n = 1 if kind == "decode" else 2
            seg_bad += [(PositionGap, _seg(A, LA + 1, x[A].slice(LA + 1, LA + 1 + n), kind)),
                        (RollbackPastCommit, _seg(A, 5, x[A].slice(5, 5 + n), kind))]
            nn = 1 if kind == "decode" else 6 + W - LA + 1
            if kind == "decode":
                seg_bad.append((SpecWindowExceeded, _seg(D, 6 + W, x[D].slice(6 + W, 7 + W), kind)))
            else:
                seg_bad.append((SpecWindowExceeded, _seg(A, LA, x[A].slice(LA, LA + nn), kind)))
        seg_bad += [
            (BadRequest, _seg(E, 12, x[E].slice(12, 13))),                          # past max_pos
            (BadRequest, _seg(E, 12, x[E].slice(12, 14), "prefill")),
            (BadRequest, _seg(A, LA, x[A].slice(LA, LA + 2), "decode")),            # decode is 1 row
            (BadRequest, _seg(A, LA, x[A].slice(LA, LA + 1), "rebuild")),
            (BadRequest, _seg(A, LA, x[A].slice(LA, LA + 1), "bogus")),
            (BadRequest, Segment(A, LA, 1, x[A].slice(LA, LA + 1), "decode", tap_in=f.hidden(1, 4))),
        ]
        foreign = next(d for d in ("f32", "i64", "i32", "bf16") if d not in cap.codecs_in)
        shape = (1,) if head else (1, cap.hidden_size)
        np_dt = {"f32": np.float32, "i64": np.int64, "i32": np.int32, "bf16": np.uint16}[foreign]
        seg_bad.append((BadRequest, _seg(A, LA, Rows(foreign, np.zeros(shape, dtype=np_dt)))))
        if head:
            for bad_id in (-1, cap.vocab_size):
                ids = Rows("i64", np.array([bad_id], dtype=np.int64))
                seg_bad.append((BadRequest, _seg(A, LA, ids)))
        else:
            h = f.hidden(1, 3)
            seg_bad.append((BadRequest, _seg(A, LA, Rows(h.dtype, np.concatenate([h.data, h.data[:, :1]], axis=1)))))
        if not tail:
            seg_bad += [(BadRequest, _seg(A, LA, x[A].slice(LA, LA + 1), logits="last")),
                        (BadRequest, _seg(A, LA, x[A].slice(LA, LA + 1), tokens="last"))]
        infos = lambda r: {s: r.info(s) for s in hist}                     # noqa: E731
        before = infos(rt)
        for exc, bad in seg_bad:
            e = _raises(exc, rt.step, StepIn((good, bad)))
            assert e.sid == bad.sid, f"{exc.__name__} for session {bad.sid} blames sid {e.sid}"
            assert infos(rt) == before, f"refused step ({exc.__name__}) changed state"
        e = _raises(BadRequest, rt.step, StepIn((good, _seg(A, LA, x[A].slice(LA, LA + 1)),
                                                 _seg(A, LA, x[A].slice(LA, LA + 1)))))
        assert e.sid == A and infos(rt) == before
        _raises(BadRequest, rt.commit, E, 13)                                  # commit past max_pos
        n = cap.max_tokens_per_step + 1
        if n <= f.max_ctx - 8:
            rt.open(99, f.max_ctx, W)
            rt.commit(99, n)
            _raises(StepTooLarge, rt.step, StepIn((good, _seg(99, 0, _inp(f, head, n, 13)))))
            rt.close(99)
            assert infos(rt) == before
        if "global" in f.layer_kinds[lo:hi]:
            _raises(BadRequest, rt.export_state, G, 0, 6)                     # committed 20 > length 5
            assert rt.export_state(G, 0, 5).hi_pos == 5
        if head:                                                              # i64 ids are an accepted codec
            o64 = _one(rt, _seg(B, 5, Rows("i64", x[B].data[5:6].astype(np.int64))))
            rt.truncate(B, 5)
        twin = _mk(f, lo, hi, head, tail)
        build(twin)
        for r in (rt, twin):
            r.commit(D, 6 + W)
        for s, st in ((A, LA), (B, 5), (D, 6 + W), (G, 5)):
            a = _one(rt, _seg(s, st, x[s].slice(st, st + 1)))
            b = _one(twin, _seg(s, st, x[s].slice(st, st + 1)))
            assert a.hidden == b.hidden, "a refused step left a trace in session state"
            if head and s == B:
                assert o64.hidden == a.hidden, "i64 token ids read differently from i32"
    rt = _mk(f, *_role(f, "mid")[:2], False, False, tap_mode="fc_partial")
    _open(rt, 1, f)
    _raises(BadRequest, rt.step, StepIn((Segment(1, 0, 1, f.hidden(1, 5), "decode", taps=False,
                                                 tap_in=f.hidden(1, 6)),)))


def check_mixed_kinds(f):
    """One step mixing a prefill chunk, a verify chunk that rolls back, a decode row and a
    rebuild segment of four sessions (sparse u64 sids) equals each segment run alone and the
    reference; logits/tokens none/last/all shapes; the sessions continue identically."""
    W = f.W
    for lo, hi, head, tail in (_role(f, "head"), _role(f, "tail")):
        kinds = f.layer_kinds[lo:hi]
        s1, s2, s3, s4 = SIDS
        xs = {s: _inp(f, head, 40, 20 + i) for i, s in enumerate(SIDS)}
        lg = (lambda m: dict(logits=m, tokens=m, sampling=SAMPLE)) if tail else (lambda m: {})
        v = min(3, W)
        hist = {s1: [("commit", 6)],
                s2: [("commit", 5), ("seg", 0, xs[s2].slice(0, 5), "prefill", 5),
                     ("seg", 5, xs[s2].slice(5, 5 + v), "verify" if v > 1 else "decode", v), ("commit", 6)],
                s3: [("commit", 5), ("seg", 0, xs[s3].slice(0, 5), "prefill", 5)]}
        vn = min(3, W)                              # rolls back to 6 when v > 1
        final = {s1: _seg(s1, 0, xs[s1].slice(0, 6), "prefill", **lg("none")),
                 s2: _seg(s2, 6, xs[s2].slice(6, 6 + vn), "verify" if vn > 1 else "decode", **lg("last")),
                 s3: _seg(s3, 5, xs[s3].slice(5, 6), "decode", **lg("all"))}
        F, g = 12, 7
        prim = _mk(f, lo, hi, head, tail)
        _open(prim, s4, f)
        _play(prim, s4, [("commit", F), ("seg", 0, xs[s4].slice(0, F), "prefill", F)])
        blob = prim.export_state(s4, 0, g) if "global" in kinds else None

        def prepare(rt, sids):
            a = None
            for s in sids:
                _open(rt, s, f)
                if s == s4:
                    if blob is not None:
                        rt.import_state(s4, blob, g)
                    a = rt.rebuild_from(s4, F)
                else:
                    _play(rt, s, hist[s])
            return a

        mixed = _mk(f, lo, hi, head, tail)
        a = prepare(mixed, SIDS)
        assert a < F, "a rebuild segment is needed for this check"
        final[s4] = _seg(s4, a, xs[s4].slice(a, F), "rebuild")
        outs = mixed.step(StepIn(tuple(final[s] for s in SIDS))).outs
        assert [o.sid for o in outs] == list(SIDS)
        ref = _ref(f, lo, hi, head, tail)
        if ref is not None:
            prepare(ref, SIDS)
            routs = ref.step(StepIn(tuple(final[s] for s in SIDS))).outs
        inv = f.numerics_mode == "invariant"          # fast mode: batch-mates may move bytes

        def same(a, b):
            if a is None or b is None:
                return a is b
            return a == b if inv else f.close_enough(a, b, "reference")

        for i, (s, o) in enumerate(zip(SIDS, outs)):
            solo = _mk(f, lo, hi, head, tail)
            prepare(solo, (s,))
            want = _one(solo, final[s])
            assert same(o.hidden, want.hidden) and same(o.logits, want.logits) and same(o.tokens, want.tokens), \
                f"session {s} differs in a mixed step"
            assert dict(o.taps).keys() == dict(want.taps).keys() and all(same(o.taps[t], want.taps[t]) for t in o.taps)
            assert o.hidden.n == final[s].n
            nrows = {"none": None, "last": 1, "all": final[s].n}
            for field in ("logits", "tokens"):
                m = getattr(final[s], field)
                got = getattr(o, field)
                assert (got is None) if nrows[m] is None else (got is not None and got.n == nrows[m]), (field, m)
            if ref is not None:
                r = routs[i]
                exact_lo = rebuild_exact_from(kinds, f.window, a) - a if s == s4 else 0
                for fld in ("logits", "tokens"):
                    x1, x2 = getattr(o, fld), getattr(r, fld)
                    assert (x1 is None) == (x2 is None) and (x1 is None or f.close_enough(x1, x2, "reference")), \
                        f"session {s}: {fld} != reference"
                for row in range(exact_lo, o.hidden.n):
                    for (nm, x1), (_, x2) in zip(_per_row(o, row), _per_row(r, row)):
                        assert f.close_enough(x1, x2, "reference"), f"session {s}: {nm} row {row} != reference"
            nxt = {s1: 6, s2: 6 + vn, s3: 6, s4: F}[s]
            for r in (mixed, solo):
                r.commit(s, nxt)
            c1 = _one(mixed, _seg(s, nxt, xs[s].slice(nxt, nxt + 1)))
            c2 = _one(solo, _seg(s, nxt, xs[s].slice(nxt, nxt + 1)))
            assert same(c1.hidden, c2.hidden), f"session {s} diverges after the mixed step"


def check_frontier(f):
    """truncate/commit at F: commit is monotone and may run ahead of length; truncate below F
    or past length is refused; after truncate(F) the old length is a gap (a stale frame);
    export stops at the frontier; the spec window ends exactly at committed + W; re-feeding F
    equals a twin whose speculative rows had other values (and, with chunk invariance, a twin
    that never speculated)."""
    W = f.W
    for lo, hi, head, tail in _roles(f)[:3]:
        x = _inp(f, head, 60, 31)
        spec = _inp(f, head, W, 32)
        rt = _mk(f, lo, hi, head, tail)
        _open(rt, 1, f)
        rt.commit(1, 9)                                  # the prompt rows are final up front
        _one(rt, _seg(1, 0, x.slice(0, 10)))
        rt.commit(1, 5)                                  # a lower commit is a no-op
        assert rt.info(1).committed == 9
        sk = "verify" if W > 1 else "decode"
        _one(rt, _seg(1, 9, spec, sk))                    # implicit truncate to 9; rows 9.. speculative
        assert rt.length(1) == 9 + W
        _raises(RollbackPastCommit, rt.truncate, 1, 8)
        _raises(PositionGap, rt.truncate, 1, 10 + W)
        _raises(BadRequest, rt.export_state, 1, 0, 10)   # row 9 is past the frontier
        rt.truncate(1, 9)
        assert rt.length(1) == 9, "truncate(F) did not move the length"
        _raises(PositionGap, rt.step, StepIn((_seg(1, 9 + W, x.slice(9 + W, 10 + W)),)))   # stale frame
        _raises(PositionGap, rt.step, StepIn((_seg(1, 10, x.slice(10, 11)),)))
        m = min(2, W)
        a = _one(rt, _seg(1, 9, x.slice(9, 9 + m), "verify" if m > 1 else "decode"))
        twin = _mk(f, lo, hi, head, tail)
        _open(twin, 1, f)
        twin.commit(1, 9)
        _one(twin, _seg(1, 0, Rows.concat([x.slice(0, 9), f.perturb(x.slice(9, 10), 33)])))
        _one(twin, _seg(1, 9, f.perturb(spec, 34), sk))
        twin.truncate(1, 9)
        b = _one(twin, _seg(1, 9, x.slice(9, 9 + m), "verify" if m > 1 else "decode"))
        assert a.hidden == b.hidden, "rows after truncate(F) remember the speculative rows"
        if "global" in f.layer_kinds[lo:hi]:
            assert rt.export_state(1, 0, 9) == twin.export_state(1, 0, 9)
        if f.chunk_invariant:                            # and a twin that never speculated
            clean = _mk(f, lo, hi, head, tail)
            _open(clean, 1, f)
            clean.commit(1, 9)
            _one(clean, _seg(1, 0, x.slice(0, 9)))
            assert _one(clean, _seg(1, 9, x.slice(9, 9 + m), "verify" if m > 1 else "decode")).hidden == a.hidden, \
                "rows after truncate(F) differ from a run that never speculated"
        rt.commit(1, 40)                                 # commit ahead: long chunks are legal
        _play(rt, 1, _prefill_ops(x, 9 + m, 40))
        _raises(SpecWindowExceeded, rt.step, StepIn((_seg(1, 40, x.slice(40, 41 + W), "prefill"),)))
        _one(rt, _seg(1, 40, x.slice(40, 40 + W), "prefill"))    # ends exactly at committed + W


def check_swa_retention(f):
    """SWA window + W slack: a pipelined-spec history full of rejections (a whole W rejected,
    rollback to the frontier, re-feed) keeps every row it reads. Kept rows and the continuation
    are byte-identical to a twin whose rejected rows had other values; with chunk invariance,
    also to a clean run that never saw a rejected row."""
    spans = [r for r in _roles(f) if "swa" in f.layer_kinds[r[0]:r[1]]]
    if not spans:
        raise Skip("model has no SWA layers")
    W = f.W
    for lo, hi, head, tail in spans:
        kw = _tailkw(tail)
        prefix = 2 * f.window + 3
        res = []
        for twin in (False, True):
            ops, true, c = _spec_script(f, head, seed=41, prefix=prefix, rounds=14, W=W, twin=twin)
            rt = _mk(f, lo, hi, head, tail)
            _open(rt, SIDS[2], f)
            kept = _accepted(_play(rt, SIDS[2], ops, **kw))
            kept.update(_accepted(_continue(rt, SIDS[2], true, c, W, **kw)))
            res.append(kept)
        assert res[0] == res[1], "a rejected row's value leaked into kept rows or the continuation"
        if f.chunk_invariant:
            clean = _mk(f, lo, hi, head, tail)
            _open(clean, 1, f)
            end = max(res[0]) + 1
            clean.commit(1, end)
            outs, p, rng = [], 0, random.Random(3)
            while p < end:
                n = min(rng.randint(1, 9), end - p)
                outs.append((p, n, _one(clean, _seg(1, p, true.slice(p, p + n), **kw))))
                p += n
            assert _accepted(outs) == res[0], "speculation history changed committed bytes"


def check_row_independence(f):
    """A row's bytes (hidden, logits, tokens, taps) never depend on the VALUES of later rows in
    its segment (finite perturbations, not NaN), for verify chunks of W, 16 and the deepest
    spec window the stage accepts, and truncating those rows leaves no trace."""
    _requires_invariant(f)
    c0 = _mk(f, *_role(f, "mid")[:2], False, False).describe()
    k_max = min(c0.spec_window_max, c0.max_tokens_per_step - 8, f.max_ctx - 48)
    for K in sorted({f.W, 16, k_max}):
        for lo, hi, head, tail in _roles(f):
            kw = _tailkw(tail)
            x = _inp(f, head, 30 + K, 51)
            P0 = 9

            def run(chunk, j):
                rt = _mk(f, lo, hi, head, tail, spec_window=max(K, f.W))
                rt.open(1, f.max_ctx, K)
                rt.commit(1, P0)
                _one(rt, _seg(1, 0, x.slice(0, P0)))
                o = _one(rt, _seg(1, P0, chunk, "verify" if K > 1 else "decode", **kw))
                rt.truncate(1, P0 + j + 1)
                rt.commit(1, P0 + j + 1)
                c = _one(rt, _seg(1, P0 + j + 1, x.slice(P0 + j + 1, P0 + j + 3), "verify", **kw))
                return o, c

            base = x.slice(P0, P0 + K)
            for j in sorted({0, 1, K // 2, K - 2} & set(range(K - 1))):
                pert = Rows.concat([base.slice(0, j + 1), f.perturb(base.slice(j + 1, K), 60 + j)])
                (o1, c1), (o2, c2) = run(base, j), run(pert, j)
                for r in range(j + 1):
                    assert _row_sig(o1, r) == _row_sig(o2, r), f"row {r} depends on later rows (K={K}, j={j})"
                assert [_row_sig(c1, r) for r in range(2)] == [_row_sig(c2, r) for r in range(2)], \
                    f"rejected rows left a trace (K={K}, j={j})"


def _histories(f, head, sids, seed):
    return {s: _spec_script(f, head, seed=seed + i, prefix=5 + i, rounds=4 + i, W=f.W)
            for i, s in enumerate(sids)}


def check_batch_invariance(f):
    """A segment's bytes do not depend on which segments share its step, their sizes (up to a
    full token budget) or their order; every batch-mate also equals its solo run."""
    _requires_invariant(f)
    for lo, hi, head, tail in (_role(f, "mid"), _role(f, "tail")):
        kw = _tailkw(tail)
        a, b, c, big = SIDS
        hs = _histories(f, head, (a, b, c), seed=70)
        cap = _mk(f, lo, hi, head, tail).describe().max_tokens_per_step
        m = min(3, f.W)
        comps = [(a,), (a, b), (b, a), (c, a, b), (big, a), (b, big, c, a)]
        seen = {}
        for comp in comps:
            rt = _mk(f, lo, hi, head, tail)
            segs, used = [], 0
            for s in comp:
                if s == big:
                    continue
                _open(rt, s, f)
                ops, true, cc = hs[s]
                _play(rt, s, ops)
                segs.append(_seg(s, cc, true.slice(cc, cc + m), "verify" if m > 1 else "decode", **kw))
                used += m
            if big in comp:
                n = min(cap - used, f.max_ctx - 1)                  # fill the whole token budget
                _open(rt, big, f)
                rt.commit(big, n)
                segs.insert(comp.index(big), _seg(big, 0, _inp(f, head, n, 99), "prefill", **kw))
            for s, o, sg in zip(comp, rt.step(StepIn(tuple(segs))).outs, segs):
                sig = [_row_sig(o, r) for r in range(o.hidden.n)]
                if s in seen and s != big:
                    assert seen[s] == sig, f"session {s} changes with its batch-mates {comp}"
                seen[s] = sig
                ref = _ref(f, lo, hi, head, tail) if s == big else None
                if ref is not None:                      # the full-budget segment itself is right
                    _open(ref, big, f)
                    ref.commit(big, sg.n)
                    _match_ref(f, _rows_of([(0, sg.n, o)]), _rows_of([(0, sg.n, _one(ref, sg))]),
                               f"a {sg.n}-row segment in a full step")


def check_replay(f):
    """Replaying the recorded schedule reproduces every byte: two fresh runtimes driven through
    the same randomly batched multi-session history (sparse u64 sids) agree, and so does a
    third that runs each session alone."""
    _requires_invariant(f)
    for lo, hi, head, tail in (_role(f, "head"), _role(f, "tail")):
        kw = _tailkw(tail)
        hs = _histories(f, head, SIDS[:3], seed=80)
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
                _open(rt, s, f)
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
                        segs.append((s, _seg(s, op[1], op[2], op[3], **kw)))
                if segs:
                    for (s, _), o in zip(segs, rt.step(StepIn(tuple(g for _, g in segs))).outs):
                        res[s].append([_row_sig(o, r) for r in range(o.hidden.n)])
            return res

        r1, r2 = batched(), batched()
        assert r1 == r2, "the same batched schedule gave different bytes on a fresh runtime"
        solo = _mk(f, lo, hi, head, tail)
        for s in hs:
            _open(solo, s, f)
            got = [[_row_sig(o, r) for r in range(o.hidden.n)] for _, _, o in _play(solo, s, hs[s][0], **kw)]
            assert got == r1[s], f"session {s} differs between a random batched schedule and solo"


def check_chunk_invariance(f):
    """Only for a factory that declares chunk invariance: per-token, one-chunk and random
    chunkings of the same rows give identical hidden/logits/tokens/taps and exported KV."""
    if not f.chunk_invariant:
        raise Skip("factory does not declare chunk invariance (rebuild checks use the reference)")
    for lo, hi, head, tail in _roles(f):
        kw = _tailkw(tail)
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
            _open(rt, 1, f)
            rt.commit(1, N)
            outs = [(s, n, _one(rt, _seg(1, s, x.slice(s, s + n), **kw))) for s, n in sched]
            blob = rt.export_state(1, 0, N) if "global" in f.layer_kinds[lo:hi] else None
            res.append((_accepted(outs), blob))
        assert res[0][0] == res[1][0] == res[2][0], "bytes (or sampled tokens) depend on chunking"
        assert res[0][1] == res[1][1] == res[2][1], "KV depends on chunking despite chunk_invariant"


def check_reference(f):
    """Every output row (kept or rejected), sampled token and exported KV byte of a
    speculative history matches the factory's reference implementation."""
    for lo, hi, head, tail in _roles(f):
        ref = _ref(f, lo, hi, head, tail)
        if ref is None:
            raise Skip("factory has no reference runtime")
        kw = _tailkw(tail)
        ops, true, c = _spec_script(f, head, seed=101, prefix=f.window + 5, rounds=10, W=f.W)
        got = {}
        for name, r in (("rt", _mk(f, lo, hi, head, tail)), ("ref", ref)):
            _open(r, SIDS[3], f)
            outs = _play(r, SIDS[3], ops, **kw) + _continue(r, SIDS[3], true, c, f.W, **kw)
            blob = r.export_state(SIDS[3], 0, c) if "global" in f.layer_kinds[lo:hi] else None
            got[name] = (_rows_of(outs, keep_only=False), blob)
        _match_ref(f, got["rt"][0], got["ref"][0], f"span [{lo},{hi})")
        if got["rt"][1] is not None:
            assert _blob_close(f, got["rt"][1], got["ref"][1]), "exported KV != reference"


def _chain(f, cuts, x, sched, tap_mode="raw", reference=False):
    """Run [0, L) as a chain of stages split at `cuts`, segment by segment."""
    L = len(f.layer_kinds)
    bounds = [0, *cuts, L]
    rts = []
    for i in range(len(bounds) - 1):
        args = (f, bounds[i], bounds[i + 1], i == 0, i == len(bounds) - 2)
        rt = _ref(*args, tap_mode=tap_mode) if reference else _mk(*args, tap_mode=tap_mode)
        _open(rt, 1, f)
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
    holding every layer (hidden, logits, raw taps); i64 and i32 token ids read the same."""
    L = len(f.layer_kinds)
    if L < 3:
        raise Skip("model too small to partition")
    _requires_invariant(f)
    x = f.ids(18, 111)
    sched = [(0, 6), (6, 1), (7, 4), (11, 7)]
    whole, wt = _chain(f, [], x, sched)
    w64, _ = _chain(f, [], Rows("i64", x.data.astype(np.int64)), sched)
    assert all(a.hidden == b.hidden and a.logits == b.logits for a, b in zip(whole, w64)), "i64 ids differ"
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
            _open(rt, 1, f)
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
    """fc_partial: the tail's accumulator equals the reference fc(concat(raw taps)) and the
    reference runtime's; it does not depend on where the chain is cut; a stage holding no tap
    passes tap_in through unchanged; the tap lane never changes the hidden bytes."""
    if not f.taps:
        raise Skip("model has no drafter taps")
    L = len(f.layer_kinds)
    x = f.ids(14, 131)
    sched = [(0, 6), (6, 1), (7, 7)]
    raw, rtaps = _chain(f, [], x, sched)
    want = f.fuse_taps(rtaps)
    if f.reference(0, L, is_head=True, is_tail=True, tap_mode="fc_partial") is not None:
        rf, _ = _chain(f, [], x, sched, tap_mode="fc_partial", reference=True)
        assert f.close_enough(_cat(rf, "tap_acc"), want, "tap_fusion"), "reference disagrees with fuse_taps"
    for cuts in ([], [L // 3, 2 * L // 3], [1, L // 2, L - 1]):
        fused, _ = _chain(f, cuts, x, sched, tap_mode="fc_partial")
        acc = _cat(fused, "tap_acc")
        assert acc is not None and f.close_enough(acc, want, "tap_fusion"), f"fused taps != fc(concat) at cuts {cuts}"
        assert _cat(fused, "hidden") == _cat(raw, "hidden") and _cat(fused, "logits") == _cat(raw, "logits")
    quiet = next(((lo, lo + 2) for lo in range(1, L - 2) if not set(f.taps) & {lo, lo + 1}), None)
    if quiet:
        rt = _mk(f, quiet[0], quiet[1], False, False, tap_mode="fc_partial")
        _open(rt, 1, f)
        h, t = f.hidden(3, 132), f.hidden(3, 133)
        o = _one(rt, Segment(1, 0, 3, h, kind="prefill", tap_in=t))
        assert o.tap_acc == t, "a stage without taps altered the tap accumulator"


def check_export_import(f):
    """Mirroring: exports stop at the frontier, cover only global layers and equal the
    reference's bytes; a blob survives the wire; contiguous chunked imports (the last cut
    short at g_eff) and imports of the reference's blob reproduce the right rows; a rebuild
    never alters them; wrong layout / order / target / timing are refused."""
    spans = [r for r in _roles(f) if "global" in f.layer_kinds[r[0]:r[1]]]
    if not spans:
        raise Skip("model has no global-attention layers to mirror")
    for lo, hi, head, tail in spans[:3]:
        prim = _mk(f, lo, hi, head, tail)
        _open(prim, SIDS[1], f)
        ops, true, c = _spec_script(f, head, seed=141, prefix=6, rounds=8, W=f.W)
        _play(prim, SIDS[1], ops)
        _one(prim, _seg(SIDS[1], c, _inp(f, head, min(2, f.W), 142), "verify" if f.W > 1 else "decode"))
        _raises(BadRequest, prim.export_state, SIDS[1], 0, c + 1)          # speculative rows
        _raises(BadRequest, prim.export_state, SIDS[1], 0, c, kinds=("swa",))
        _raises(BadRequest, prim.export_state, SIDS[1], 3, 2)
        blob = prim.export_state(SIDS[1], 0, c)
        ref = _ref(f, lo, hi, head, tail)
        rblob = None
        if ref is not None:
            _open(ref, SIDS[1], f)
            _play(ref, SIDS[1], ops)
            rblob = ref.export_state(SIDS[1], 0, c)
            assert _blob_close(f, blob, rblob), "exported KV bytes != reference"
        assert KVBlob.from_bytes(blob.to_bytes()) == blob
        _raises(BadRequest, KVBlob.from_bytes, blob.to_bytes()[:-1])
        k, g = c // 2, c - 2
        b1, b2 = prim.export_state(SIDS[1], 0, k), prim.export_state(SIDS[1], k, c)
        bk = _mk(f, lo, hi, head, tail)
        _open(bk, SIDS[2], f)
        _raises(BadRequest, bk.import_state, SIDS[2], b2, c)                # must start at 0
        bk.import_state(SIDS[2], b1, k)
        _raises(BadRequest, bk.step, StepIn((_seg(SIDS[2], 0, true.slice(0, 1)),)))   # rebuild_from first
        _raises(BadRequest, bk.export_state, SIDS[2], 0, 1)
        bk.import_state(SIDS[2], b2, g)                                     # g_eff cuts the blob short
        assert bk.info(SIDS[2]).imported_upto == g
        a = bk.rebuild_from(SIDS[2], c)
        p = a
        while p < c:
            n = min(5, c - p)
            _one(bk, _seg(SIDS[2], p, true.slice(p, p + n), "rebuild"))
            p += n
        assert bk.export_state(SIDS[2], 0, g) == _blob_prefix(blob, g), "a rebuild altered mirrored rows"
        if rblob is not None:                                               # import the ORACLE's blob
            cont = {}
            for name, r in (("rt", _mk(f, lo, hi, head, tail)), ("ref", _ref(f, lo, hi, head, tail))):
                _open(r, SIDS[3], f)
                r.import_state(SIDS[3], rblob, c)
                a2 = r.rebuild_from(SIDS[3], c)
                q = a2
                while q < c:
                    n = min(5, c - q)
                    _one(r, _seg(SIDS[3], q, true.slice(q, q + n), "rebuild"))
                    q += n
                cont[name] = _rows_of(_continue(r, SIDS[3], true, c, f.W, **_tailkw(tail)))
            _match_ref(f, cont["rt"], cont["ref"], "continuation after importing the reference's blob")
        used = _mk(f, lo, hi, head, tail)
        _open(used, 1, f)
        used.commit(1, 2)
        _one(used, _seg(1, 0, true.slice(0, 2)))
        _raises(BadRequest, used.import_state, 1, blob, c)                  # not a fresh session
        fresh = _mk(f, lo, hi, head, tail)
        _open(fresh, 1, f)
        _raises(BadRequest, fresh.import_state, 1, blob, c + 1)             # upto past the blob
        other = next(((l2, h2, hd, tl) for l2, h2, hd, tl in spans
                      if [i for i in range(l2, h2) if f.layer_kinds[i] == "global"]
                      != [i for i in range(lo, hi) if f.layer_kinds[i] == "global"]), None)
        if other:
            o = _mk(f, *other)
            _open(o, 1, f)
            _raises(BadRequest, o.import_state, 1, blob, c)                # another span's layout


def _blob_prefix(blob: KVBlob, g: int) -> KVBlob:
    return KVBlob(0, g, blob.layers, tuple(r.slice(0, g) for r in blob.rows), blob.layout_id, blob.numerics_id)


def _backup(r, sid, f, blob, g, F, true, chunk, exact_from, kinds, tail, floor=None):
    """Import [0, g), rebuild [a, F) in `chunk`-row segments, then (with a floor) rewind to the
    floor as a re-adopting entry would, and continue 3 W-rounds from the rollback floor."""
    _open(r, sid, f)
    if blob is not None:
        r.import_state(sid, blob, g)
    a = r.rebuild_from(sid, F, exact_from, floor)
    fl = F if floor is None else floor
    rb = []
    p = a
    while p < F:
        n = min(chunk, F - p)
        rb.append((p, n, _one(r, _seg(sid, p, true.slice(p, p + n), "rebuild"))))
        p += n
    lo_exact = max(rebuild_exact_from(kinds, f.window, a), exact_from or 0)
    exact = {p + i: (o, i) for p, n, o in rb for i in range(n) if p + i >= lo_exact}
    kv = r.export_state(sid, 0, fl) if blob is not None else None
    gm = min(g, fl)
    mirror_ok = blob is None or gm == 0 or r.export_state(sid, 0, gm) == _blob_prefix(blob, gm)
    if fl < F:
        r.truncate(sid, fl)
    cont = _rows_of(_continue(r, sid, true, fl, f.W, **_tailkw(tail)))
    return a, exact, kv, mirror_ok, cont


def check_rebuild(f):
    """The takeover rebuild (§4.1 rule) for span patterns SSG, SGS, SSSSSG, SGSGS, GSS, SSS, GG
    and the whole model, with g_eff = F, just below F, well below F and 0, with an exact_from
    request, and with a rollback floor below F that the continuation then rewinds to (entry
    re-adoption after a takeover): a is exactly the rule; the continuation, the rebuilt KV and the
    rebuild's output rows from rebuild_exact_from on match the reference (every numerics);
    mirrored rows are never overwritten; the same schedule replays byte-equal; with chunk
    invariance everything is also bit-equal to the fault-free primary, for any chunking."""
    L = len(f.layer_kinds)
    cases = [(pat, _find(f, pat)) for pat in REBUILD_PATTERNS]
    cases = [(pat, sp) for pat, sp in cases if sp] + [("whole", (0, L))]
    w1 = f.window - 1
    sid = SIDS[3]
    for i, (pat, (lo, hi)) in enumerate(cases):
        head, tail = lo == 0, hi == L
        kinds = tuple(f.layer_kinds[lo:hi])
        n_swa, has_g = kinds.count("swa"), "global" in kinds
        prefix = 2 * w1 * max(n_swa, 1) + 6
        ops, true, F = _spec_script(f, head, seed=150 + i, prefix=prefix, rounds=9, W=f.W)
        prim = _mk(f, lo, hi, head, tail)
        _open(prim, sid, f)
        pouts = _rows_of(_play(prim, sid, ops, **_tailkw(tail)))
        _one(prim, _seg(sid, F, _inp(f, head, min(3, f.W), 151 + i), "verify" if f.W > 1 else "decode"))
        full = prim.export_state(sid, 0, F) if has_g else None
        ref_cont = _rows_of(_continue(prim, sid, true, F, f.W, **_tailkw(tail)))
        gs = sorted({g for g in (F, F - 1, F - w1 - 2, F - 3 * w1 - 5, w1 + 1, 0) if 0 <= g <= F}) if has_g else [0]
        fl_cfg = max(F - min(f.W, 3), 0)
        cfgs = ([(g, None, None) for g in gs] + [(gs[-1], max(0, F - 3 * w1 - 2), None)]
                + [(g, None, fl_cfg) for g in (gs[-1], gs[len(gs) // 2])])
        for g, ex, fl in cfgs:
            where = f"span {pat} [{lo},{hi}) F={F} g={g} exact_from={ex} floor={fl}"
            blob = _blob_prefix(full, g) if has_g else None
            fe = F if fl is None else fl
            rule = rebuild_start(kinds, f.window, F, g if has_g else 0, fe if ex is None else min(ex, fe))
            runs = [_backup(_mk(f, lo, hi, head, tail), sid, f, blob, g, F, true, ch, ex, kinds, tail, fl)
                    for ch in (5, 5, 3)]
            ref = _ref(f, lo, hi, head, tail)
            # Behaviour first: a wrong rebuild must show up as wrong state, not only as a formula.
            if ref is not None:
                ra, rexact, rkv, _, rcont = _backup(ref, sid, f, blob, g, F, true, 5, ex, kinds, tail, fl)
                a, exact, kv, _, cont = runs[0]
                _match_ref(f, cont, rcont, f"continuation after rebuild ({where})")
                _match_ref(f, exact, rexact, f"rebuild outputs from rebuild_exact_from on ({where})")
                assert kv is None or _blob_close(f, kv, rkv), f"KV after rebuild != reference ({where})"
            if f.chunk_invariant:
                want = {p: _row_sig(*ref_cont[p]) for p in ref_cont}
                for a, exact, kv, _, cont in (runs[0], runs[2]):
                    if fl is None:
                        assert {p: _row_sig(*cont[p]) for p in cont} == want, \
                            f"continuation after rebuild != fault-free ({where})"
                    for p, (o, r) in cont.items():                 # from the floor: hidden bytes
                        fault = pouts.get(p) or ref_cont.get(p)
                        if fault is not None:
                            assert o.hidden.row_bytes(r) == fault[0].hidden.row_bytes(fault[1]), \
                                f"row {p} after rebuild + rewind != fault-free ({where})"
                    assert kv is None or kv == _blob_prefix(full, kv.hi_pos), f"KV after rebuild != fault-free ({where})"
                    for p, (o, r) in exact.items():
                        assert o.hidden.row_bytes(r) == pouts[p][0].hidden.row_bytes(pouts[p][1]), \
                            f"rebuilt row {p} != fault-free ({where})"
            (_, e_a, kv_a, _, c_a), (_, e_b, kv_b, _, c_b) = runs[0], runs[1]
            assert kv_a == kv_b and {p: _row_sig(*c_a[p]) for p in c_a} == {p: _row_sig(*c_b[p]) for p in c_b}, \
                f"same rebuild schedule, different bytes ({where})"
            for a, _, _, mirror_ok, _ in runs:
                assert mirror_ok, f"rebuild overwrote mirrored global rows ({where})"
                # Replay-equality would replay a wrong start too, so the rule is checked directly.
                assert a == rule, f"rebuild_from returned a={a}, the rule gives {rule} ({where})"


def check_rebuild_protocol(f):
    """rebuild_from only on a fresh/imported session, with upto <= F and exact_from <= F; while
    rebuilding only in-order rebuild segments up to F (no logits/tokens, no truncate, no
    export); afterwards normal segments resume at F."""
    lo, hi = next((sp for sp in (_find(f, p) for p in ("SGS", "SSG", "GSS")) if sp), _role(f, "mid")[:2])
    L = len(f.layer_kinds)
    head, tail = lo == 0, hi == L
    kinds = f.layer_kinds[lo:hi]
    x = _inp(f, head, 40, 161)
    prim = _mk(f, lo, hi, head, tail)
    _open(prim, 1, f)
    prim.commit(1, 30)
    _play(prim, 1, _prefill_ops(x, 0, 30))
    _raises(BadRequest, prim.rebuild_from, 1, 30)                     # not fresh
    has_g = "global" in kinds
    g = 20
    rt = _mk(f, lo, hi, head, tail)
    _open(rt, 1, f)
    if has_g:
        rt.import_state(1, prim.export_state(1, 0, g), g)
        _raises(BadRequest, rt.rebuild_from, 1, g - 1)                # F below the imported rows
    _raises(BadRequest, rt.rebuild_from, 1, 30, 31)                   # exact_from past F
    _raises(BadRequest, rt.rebuild_from, 1, 30, None, 31)             # floor past F
    _raises(BadRequest, rt.rebuild_from, 1, 30, None, 29 - f.W)       # floor deeper than W
    a = rt.rebuild_from(1, 30)
    assert a < 30
    _raises(BadRequest, rt.step, StepIn((_seg(1, a, x.slice(a, a + 1), "decode"),)))
    if a + 1 < 30:
        _raises((BadRequest, PositionGap), rt.step, StepIn((_seg(1, a + 1, x.slice(a + 1, a + 2), "rebuild"),)))
    _raises(BadRequest, rt.step, StepIn((_seg(1, a, x.slice(a, 31), "rebuild"),)))
    _raises((RollbackPastCommit, PositionGap, BadRequest), rt.truncate, 1, a)
    _raises(BadRequest, rt.export_state, 1, 0, 1)
    p = a
    while p < 30:
        n = min(PREFILL_CHUNK, 30 - p)
        _one(rt, _seg(1, p, x.slice(p, p + n), "rebuild"))
        p += n
    _raises(BadRequest, rt.step, StepIn((_seg(1, 30, x.slice(30, 31), "rebuild"),)))
    _one(rt, _seg(1, 30, x.slice(30, 31)))
    lo, hi, head, tail = _role(f, "tail")
    x = _inp(f, head, 40, 162)
    rt2 = _mk(f, lo, hi, head, tail)
    _open(rt2, 1, f)
    a2 = rt2.rebuild_from(1, 30)
    for kw in (dict(logits="last"), dict(tokens="last")):
        if a2 < 30:
            _raises(BadRequest, rt2.step, StepIn((_seg(1, a2, x.slice(a2, min(30, a2 + 8)), "rebuild", **kw),)))


def check_long_context(f):
    """A speculative history that crosses position 300 (past every small page table) matches
    the reference row for row, and so does its exported KV."""
    lo, hi, head, tail = _role(f, "mid")
    if f.max_ctx < 360:
        raise Skip("factory max_ctx < 360")
    ref = _ref(f, lo, hi, head, tail)
    if ref is None:
        raise Skip("factory has no reference runtime")
    ops, true, c = _spec_script(f, head, seed=201, prefix=300, rounds=8, W=f.W)
    got = {}
    for name, r in (("rt", _mk(f, lo, hi, head, tail)), ("ref", ref)):
        _open(r, SIDS[2], f)
        outs = _play(r, SIDS[2], ops) + _continue(r, SIDS[2], true, c, f.W)
        got[name] = (_rows_of(outs), r.export_state(SIDS[2], 0, c) if "global" in f.layer_kinds[lo:hi] else None)
    _match_ref(f, got["rt"][0], got["ref"][0], "long context")
    if got["rt"][1] is not None:
        assert _blob_close(f, got["rt"][1], got["ref"][1]), "long-context KV != reference"


def check_close_isolation(f):
    """close() drops one session: its batch-mate continues byte-identically to a runtime that
    never had it, and a reopened sid starts fresh."""
    lo, hi, head, tail = _role(f, "mid")
    s1, s2 = SIDS[1], SIDS[2]
    hs = _histories(f, head, (s1, s2), seed=170)
    rt = _mk(f, lo, hi, head, tail)
    for s in hs:
        _open(rt, s, f)
    ops1, ops2 = hs[s1][0], hs[s2][0]
    _play(rt, s1, ops1[: len(ops1) // 2])
    rt.close(s1)
    got = _accepted(_play(rt, s2, ops2))
    alone = _mk(f, lo, hi, head, tail)
    _open(alone, s2, f)
    assert got == _accepted(_play(alone, s2, ops2)), "closing a session disturbed another"
    _open(rt, s1, f)
    assert rt.length(s1) == 0 and rt.info(s1).committed == 0
    fresh = _mk(f, lo, hi, head, tail)
    _open(fresh, s1, f)
    assert _accepted(_play(rt, s1, ops1)) == _accepted(_play(fresh, s1, ops1)), "a reopened sid kept old state"


def check_tokens(f):
    """Tail tokens: a greedy token's logit is a maximum and top_k=1 is greedy; "last" is the last
    row of "all" for logits and tokens; sampled draws depend on the seed, match the reference
    for several (temperature, top_k, top_p) settings, equal project(tokens=True), and (with
    chunk invariance) are keyed by absolute position, not by row."""
    lo, hi, head, tail = _role(f, "tail")
    x = _inp(f, head, 64, 181)
    N = 24

    def run(sched, **kw):
        rt = _mk(f, lo, hi, head, tail)
        _open(rt, 1, f)
        rt.commit(1, N)
        return [_one(rt, _seg(1, s, x.slice(s, s + n), **kw)) for s, n in sched]

    chunks = [(p, min(8, N - p)) for p in range(0, N, 8)]
    greedy = run(chunks, logits="all", tokens="all")
    for o in greedy:                                    # the chosen logit is a maximum (the backend
        lg = _as_float(o.logits)                        # may argmax in finer precision than it returns)
        assert (lg[np.arange(o.tokens.n), o.tokens.data] == lg.max(axis=1)).all(), "greedy token is not a max"
    g_all = _cat(greedy, "tokens")
    assert _cat(run(chunks, tokens="all", sampling=Sampling(seed=5, temperature=1.0, top_k=1)), "tokens") == g_all, \
        "top_k=1 is not greedy"
    last = run(chunks, logits="last", tokens="last")
    for a, o in zip(greedy, last):
        assert o.logits.n == 1 and o.tokens.n == 1
        assert o.logits == a.logits.slice(a.logits.n - 1, a.logits.n), "logits='last' is not the last row"
        assert o.tokens == a.tokens.slice(a.tokens.n - 1, a.tokens.n), "tokens='last' is not the last row"
    s1 = _cat(run(chunks, tokens="all", sampling=SAMPLE), "tokens")
    s2 = _cat(run(chunks, tokens="all", sampling=Sampling(seed=SAMPLE.seed + 1, temperature=1.0, top_k=4)), "tokens")
    assert s1 != s2, "the sampling seed does not change any of 24 draws"
    ref = _ref(f, lo, hi, head, tail)
    if ref is not None:
        configs = (SAMPLE, Sampling(seed=7, temperature=1.0, top_k=0),
                   Sampling(seed=8, temperature=0.7, top_k=0, top_p=0.5), Sampling(seed=9, temperature=1.3, top_k=3))
        for i, sp in enumerate(configs):
            r = _ref(f, lo, hi, head, tail)
            _open(r, 1, f)
            r.commit(1, N)
            want = _cat([_one(r, _seg(1, s, x.slice(s, s + n), tokens="all", sampling=sp)) for s, n in chunks], "tokens")
            got = s1 if i == 0 else _cat(run(chunks, tokens="all", sampling=sp), "tokens")
            assert f.close_enough(got, want, "reference"), f"sampled tokens != reference for {sp}"
    tl = _mk(f, lo, hi, head, tail)
    _open(tl, 1, f)
    tl.commit(1, N)
    o = _one(tl, _seg(1, 0, x.slice(0, 8), tokens="all", sampling=SAMPLE))
    assert tl.project(o.hidden, tokens=True, start_pos=0, sampling=SAMPLE) == o.tokens, \
        "project(tokens=True) draws differently from the step"
    if f.chunk_invariant:
        per = _cat(run([(p, 1) for p in range(N)], tokens="all", sampling=SAMPLE), "tokens")
        assert per == s1, "sampled tokens depend on the segment shape (keyed by row, not position)"


def check_embed_project(f):
    """embed() (head) feeds a non-head stage at layer 0 exactly like the head's own step;
    project() (tail) equals the step's logits; both refuse the wrong role."""
    lo, hi, head, _ = _role(f, "head")
    x = f.ids(7, 191)
    hd = _mk(f, lo, hi, True, False)
    _open(hd, 1, f)
    hd.commit(1, 7)
    want = _one(hd, _seg(1, 0, x))
    body = _mk(f, lo, hi, False, False)
    _open(body, 1, f)
    body.commit(1, 7)
    emb = hd.embed(x)
    assert emb.n == 7 and emb.width == hd.describe().hidden_size
    assert _one(body, _seg(1, 0, emb)).hidden == want.hidden, "embed() != the head's own embedding"
    _raises(BadRequest, body.embed, x)
    lo, hi, head, tail = _role(f, "tail")
    tl = _mk(f, lo, hi, head, True)
    _open(tl, 1, f)
    tl.commit(1, 5)
    o = _one(tl, _seg(1, 0, _inp(f, head, 5, 192), logits="all"))
    assert tl.project(o.hidden) == o.logits, "project() != the step's logits"
    _raises(BadRequest, _mk(f, *_role(f, "mid")[:2], False, False).project, o.hidden)
    ref = _ref(f, lo, hi, head, True)
    if ref is not None:
        assert f.close_enough(tl.project(o.hidden), ref.project(o.hidden), "reference")


def check_load_refusal(f):
    """load() refuses a weight source missing a tensor of the span (never a random init)."""
    lo, hi, _, _ = _role(f, "mid")
    make_broken = getattr(f, "make_broken", None)
    if make_broken is None:
        raise Skip("factory cannot build a broken weight source")
    _raises(LoadRefused, make_broken, lo, hi)


CHECKS = (check_describe, check_admission, check_validation_atomic, check_mixed_kinds,
          check_frontier, check_swa_retention, check_row_independence, check_batch_invariance,
          check_replay, check_chunk_invariance, check_reference, check_partition_invariance,
          check_taps_raw, check_taps_fc_partial, check_export_import, check_rebuild,
          check_rebuild_protocol, check_long_context, check_close_isolation, check_tokens,
          check_embed_project, check_load_refusal)


def run_suite(f, checks=CHECKS) -> dict:
    """{check name: "pass" | "skip" | "FAIL: <error>"} (the mutant matrix uses it)."""
    res = {}
    for chk in checks:
        try:
            chk(f)
            res[chk.__name__] = "pass"
        except Skip:
            res[chk.__name__] = "skip"
        except (AssertionError, StageRuntimeError, Exception) as e:   # noqa: BLE001
            res[chk.__name__] = f"FAIL: {type(e).__name__}: {e}"
    return res
