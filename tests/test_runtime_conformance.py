"""ToyRuntime passes the StageRuntime conformance suite; wrong runtimes do not (BUILD-PLAN F3).

Two parts:
  1. tests/runtime_conformance.py's CHECKS on ToyRuntime in three geometries (w == W, w > W,
     W >> w, so a swapped window cannot hide) and both numeric modes: "chunk_invariant"
     (integer, schedule-free bytes: the rebuild must be bit-equal to the fault-free state) and
     "adversarial" (bytes depend on each segment's (start_pos, n), like real float kernels:
     everything must still hold except chunk invariance, and takeovers are judged against
     the independent ToyReference instead).
  2. Anti-vacuity: mutant runtimes, each one realistic bug on top of the toy, must FAIL the
     checks designed to catch them, in both modes. A suite a wrong runtime can pass proves
     nothing. The later mutants are the ones an adversarial review first got past the suite.
"""
import dataclasses

import numpy as np
import pytest

from runtime_conformance import (CHECKS, REBUILD_PATTERNS, Skip, _find, _seg, run_suite)
from shard.runtime.api import (ID_DTYPES, BadRequest, KVBlob, Rows, SegOut, StageSpec, StepIn,
                               rebuild_start)
from shard.runtime.toy import P, ToyReference, ToyRuntime, ToyWeights, toy_draw

PATTERN = "SSSSSGSGSSGG"      # holds every gate span: SSG, SGS, SSSSSG, SGSGS (+ GSS, SSS, GG)
TAPS = (0, 4, 7, 11)          # DFlash-like: first layer, spread, and the last layer (on the tail)
HIDDEN, VOCAB = 4, 16
GEOMS = {"w4W4": (4, 4), "w6W3": (6, 3), "w3W16": (3, 16)}   # (SWA window w, spec window W)


class ToyFactory:
    """RuntimeFactory for ToyRuntime (or a subclass of it: the mutants)."""
    layer_kinds = tuple("swa" if c == "S" else "global" for c in PATTERN)
    taps = TAPS
    numerics_mode = "invariant"
    max_ctx = 512

    def __init__(self, mode="chunk_invariant", cls=ToyRuntime, geom="w4W4"):
        self.mode, self.cls, self.geom = mode, cls, geom
        self.window, self.W = GEOMS[geom]
        self.chunk_invariant = mode == "chunk_invariant"
        self.name = f"{cls.__name__}[{mode},{geom}]"
        self.weights = ToyWeights(PATTERN, hidden=HIDDEN, vocab=VOCAB, taps=TAPS)

    def _spec(self, lo, hi, is_head, is_tail, tap_mode="raw", spec_window=64, max_sessions=8,
              max_ctx=None, kv_budget_bytes=0):
        # Stages accept speculation up to 64 deep (a production default); sessions open with W.
        return StageSpec("toy", lo, hi, is_head=is_head, is_tail=is_tail, taps=TAPS,
                         tap_mode=tap_mode, max_sessions=max_sessions,
                         max_ctx=max_ctx or self.max_ctx, spec_window=max(spec_window or 64, self.W),
                         kv_budget_bytes=kv_budget_bytes)

    def _build(self, lo, hi, is_head, is_tail, **kw):
        return self.cls(self._spec(lo, hi, is_head, is_tail, **kw), pattern=PATTERN,
                        window=self.window, hidden=HIDDEN, vocab=VOCAB, mode=self.mode)

    def make(self, lo, hi, *, is_head=False, is_tail=False, **kw):
        rt = self._build(lo, hi, is_head, is_tail, **kw)
        rt.load(self.weights)
        return rt

    def make_unloaded(self, lo, hi, *, is_head=False, is_tail=False):
        return self._build(lo, hi, is_head, is_tail), self.weights

    def kv_bytes_per_pos(self, lo, hi):
        kinds = self.layer_kinds[lo:hi]
        return kinds.count("global") * HIDDEN * 4, kinds.count("swa") * HIDDEN * 4

    def reference(self, lo, hi, *, is_head=False, is_tail=False, tap_mode="raw"):
        return ToyReference(self.weights, self._spec(lo, hi, is_head, is_tail, tap_mode=tap_mode),
                            pattern=PATTERN, window=self.window, mode=self.mode)

    def hidden(self, n, seed):
        rng = np.random.default_rng(seed)
        return Rows("i32", rng.integers(0, P, (n, HIDDEN)).astype(np.int32))

    def ids(self, n, seed):
        rng = np.random.default_rng(seed)
        return Rows("i32", rng.integers(0, VOCAB, n).astype(np.int32))

    def perturb(self, rows, seed):
        rng = np.random.default_rng(seed)
        m = VOCAB if rows.data.ndim == 1 else P
        d = rows.data.astype(np.int64)
        return Rows(rows.dtype, ((d + 1 + rng.integers(0, m - 1, d.shape)) % m).astype(rows.data.dtype))

    def fuse_taps(self, raw):
        acc = 0
        for t in TAPS:
            acc = (acc + raw[t].data.astype(np.int64) @ self.weights.read(f"tap_fc.{t}").T) % P
        return Rows("i32", acc.astype(np.int32))

    def close_enough(self, a, b, what):
        return a == b

    def make_broken(self, lo, hi):
        rt = self._build(lo, hi, lo == 0, hi == len(PATTERN))
        rt.load(self.weights.without(f"layers.{lo}.b"))


FACTORIES = [ToyFactory(mode, geom=g) for g in GEOMS for mode in ("chunk_invariant", "adversarial")]


@pytest.mark.parametrize("factory", FACTORIES, ids=lambda f: f"{f.mode}-{f.geom}")
@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.__name__)
def test_toy_conforms(check, factory):
    try:
        check(factory)
    except Skip as e:
        if (factory.mode, check.__name__) != ("adversarial", "check_chunk_invariance"):
            raise AssertionError(f"unexpected skip on the toy: {e}") from e
        pytest.skip(str(e))


def test_toy_model_holds_every_gate_span():
    """The rebuild gate names SSG, SGS, SSSSSG, SGSGS: the toy must contain them all, or
    check_rebuild would silently test fewer patterns."""
    f = FACTORIES[0]
    assert all(_find(f, p) for p in REBUILD_PATTERNS), [p for p in REBUILD_PATTERNS if not _find(f, p)]


def test_rebuild_cases_are_not_trivial():
    """The rebuild check must exercise partial SWA windows (0 < a) and global layers above SWA
    layers reading mirrored rows below g (a < g): otherwise 'bit-equal' is vacuous."""
    kinds = lambda pat: ["swa" if c == "S" else "global" for c in pat]   # noqa: E731
    F, w = 50, 4
    assert 0 < rebuild_start(kinds("SSSSSG"), w, F, F - 5) < F - 5 - 3
    assert rebuild_start(kinds("SGSGS"), w, F, F - 1) == F - 3 * (w - 1)      # F term binds
    assert rebuild_start(kinds("SGSGS"), w, F, F - 10) == F - 10 - 2 * (w - 1)  # upper G binds
    assert rebuild_start(kinds("SSS"), w, F, 0) == F - 3 * (w - 1)
    assert rebuild_start(kinds("SSS"), w, F, 0, exact_from=F - 20) == F - 20 - 3 * (w - 1)


def test_adversarial_mode_really_depends_on_shape():
    """In adversarial mode, chunking the same rows differently changes bytes (so the suite's
    adversarial pass shows it never leans on chunk invariance), while the reference agrees for
    a given schedule."""
    f = ToyFactory("adversarial")
    x = f.ids(12, 1)
    outs = []
    for sched in ([(0, 12)], [(0, 5), (5, 7)]):
        got = []
        for rt in (f.make(0, len(PATTERN), is_head=True, is_tail=True),
                   f.reference(0, len(PATTERN), is_head=True, is_tail=True)):
            rt.open(1, 64, 4)
            rt.commit(1, 12)
            got.append(Rows.concat([rt.step(StepIn((_seg(1, s, x.slice(s, s + n)),))).outs[0].hidden
                                    for s, n in sched]))
        assert got[0] == got[1]
        outs.append(got[0])
    assert outs[0] != outs[1]


# ── mutants: each one realistic bug; each must fail the checks designed to catch it ──────────

class MirrorOverwrite(ToyRuntime):
    """The rebuild rewrites global rows below the mirror frontier from partial-window inputs
    (ARCHITECTURE-v2 Appendix B1, the first-draft rebuild)."""
    def _write_kv(self, s, li, p, row, rebuilding):
        super()._write_kv(s, li, p, row, False)


class IgnoresTruncate(ToyRuntime):
    """An explicit truncate (lease_update truncate_to=F) is validated, then forgotten: the
    session keeps its old length and accepts a stale frame there."""
    def truncate(self, sid, pos):
        old = self._get(sid).length
        super().truncate(sid, pos)
        self._get(sid).length = old


class BatchCoupled(ToyRuntime):
    """A row's bytes depend on the step's total token count (a batch-size-dependent split-K)."""
    def _row_term(self, ctx, seg, li, r, x):
        return super()._row_term(ctx, seg, li, r, x) + ctx.total_tokens


class WindowOffByOne(ToyRuntime):
    """SWA attends to w + 1 rows ([p-w, p] instead of [p-w+1, p])."""
    _strict = False

    def _window_lo(self, s, p):
        return max(s.swa_lo, p - self.window, 0)


class LeaksRejectedRows(ToyRuntime):
    """Global attention through a running prefix accumulator that a rollback never rewinds:
    rejected speculative rows stay mixed into every later row."""
    def _attend_global(self, s, li, p):
        run = s.__dict__.setdefault("run", {})
        g = (run.get(li, np.zeros(self.H, np.int64)) * 31 + s.glob[li][p] * (p % 7 + 1)) % P
        run[li] = g
        return g


class SwaNoSlack(ToyRuntime):
    """An SWA ring of exactly `window` slots: fine without speculation, but rejected rows
    overwrite rows that a rollback to the frontier still reads."""
    _strict = False

    def _swa_capacity(self, W):
        return self.window


class RowCoupled(ToyRuntime):
    """Rows of a segment share a statistic (a per-tensor wire scale): kept rows depend on the
    values of rejected rows."""
    def _row_term(self, ctx, seg, li, r, x):
        return super()._row_term(ctx, seg, li, r, x) + int(x.sum() % 1009)


class TapOffByOne(ToyRuntime):
    """Captures tap L from the layer's INPUT (the output of layer L-1)."""
    def _tap_value(self, li, x_in, out):
        return x_in


class FirstDraftRebuildRule(ToyRuntime):
    """Rebuild starts at the first-draft a = min(upto, F - (w-1)*n_swa), ignoring the SWA
    layers below each global layer (it keeps mirrored rows, unlike MirrorOverwrite)."""
    def rebuild_from(self, sid, F, exact_from=None):
        s = self._get(sid)
        upto = s.upto
        a = super().rebuild_from(sid, F, exact_from)
        if any(k == "global" for k in self.kinds):
            late = max(0, min(upto, F - (self.window - 1) * sum(k == "swa" for k in self.kinds)))
            if late > a:
                s.length = s.swa_lo = a = late
                s.rebuild_hi = F if a < F else None
        return a


class NoopRebuild(ToyRuntime):
    """Rebuild segments only advance the length: SWA windows and global rows past g stay empty."""
    _strict = False

    def _run(self, s, seg, ctx):
        if seg.kind != "rebuild":
            return super()._run(s, seg, ctx)
        s.length = seg.start_pos + seg.n
        if s.length >= s.rebuild_hi:
            s.rebuild_hi = None
        return SegOut(seg.sid, Rows("i32", np.zeros((seg.n, self.H), np.int32)))


def _xor(blob):
    return KVBlob(blob.lo_pos, blob.hi_pos, blob.layers,
                  tuple(Rows("i32", r.data ^ np.int32(0x55)) for r in blob.rows), blob.layout_id, blob.numerics_id)


class BlobRowsMasked(ToyRuntime):
    """Export scrambles the rows and import unscrambles them: round trips through the same
    runtime pass, but the KV receipt leaves and a cross-backend takeover are wrong."""
    def export_state(self, sid, lo_pos, hi_pos, kinds=("global",)):
        return _xor(super().export_state(sid, lo_pos, hi_pos, kinds))

    def import_state(self, sid, blob, upto):
        return super().import_state(sid, _xor(blob), upto)


class SpecCheckUsesWindow(ToyRuntime):
    """Bounds speculation by the SWA window w instead of the spec window W."""
    _strict = False

    def _validate(self, seg):
        s = self._get(seg.sid)
        W, s.W = s.W, self.window
        try:
            super()._validate(seg)
        finally:
            s.W = W


class RebuildUsesW(ToyRuntime):
    """The rebuild rule counts (W - 1) rows per SWA layer instead of (w - 1)."""
    def rebuild_from(self, sid, F, exact_from=None):
        w, self.window = self.window, self._get(sid).W
        try:
            return super().rebuild_from(sid, F, exact_from)
        finally:
            self.window = w


class SidSlotAlias(ToyRuntime):
    """KV lives in table slot sid % max_sessions, so two open sids can share their KV."""
    _strict = False

    def open(self, sid, max_pos, spec_window):
        super().open(sid, max_pos, spec_window)
        me = self._sess[sid]
        for other, s in self._sess.items():
            if (other != sid and other % self.spec.max_sessions == sid % self.spec.max_sessions
                    and (s.max_pos, s.W) == (max_pos, spec_window)):
                me.glob, me.swa, me.tag = s.glob, s.swa, s.tag
                break


class LogitsLastIsFirst(ToyRuntime):
    """logits/tokens "last" return the FIRST row of the segment."""
    def _last_rows(self, x, mode):
        return x[:1] if mode == "last" else x


class BatchCoupledOverBudget(ToyRuntime):
    """Rows depend on the step size only above 40 tokens (a large-batch kernel path)."""
    def _row_term(self, ctx, seg, li, r, x):
        return super()._row_term(ctx, seg, li, r, x) + (ctx.total_tokens if ctx.total_tokens > 40 else 0)


class ClaimsFast(BatchCoupledOverBudget):
    """Hides its batch coupling by reporting numerics_mode='fast' (self-exemption)."""
    def describe(self):
        return dataclasses.replace(super().describe(), numerics_mode="fast")


class RowCoupledLarge(ToyRuntime):
    """Couples rows only in segments of 11+ rows (a large-chunk kernel path)."""
    def _row_term(self, ctx, seg, li, r, x):
        return super()._row_term(ctx, seg, li, r, x) + (int(x.sum() % 1009) if seg.n >= 11 else 0)


class SamplerKeyedByRow(ToyRuntime):
    """Keys the sampling RNG by the row's index in its segment instead of its position."""
    def _draw(self, logits_row, sampling, p, r):
        return toy_draw(logits_row, sampling, r)


class PrefillSkipsRollbackCheck(ToyRuntime):
    """Prefill segments may rewind below the committed frontier (final rows get rewritten)."""
    def _validate(self, seg):
        s = self._get(seg.sid)
        c = s.committed
        if seg.kind == "prefill" and seg.start_pos < s.length:
            s.committed = min(c, seg.start_pos)
        try:
            super()._validate(seg)
        finally:
            s.committed = c


class NoTokenRangeCheck(ToyRuntime):
    """Checks id dtype and shape but not range: -1 reads the last row, V fails mid-step."""
    def _check_ids(self, x, n, sid):
        if not isinstance(x, Rows) or x.dtype not in ID_DTYPES or x.data.ndim != 1 or x.n != n:
            raise BadRequest("token ids must be Rows i32/i64 [n]", sid=sid)


class RowCoupledN17(ToyRuntime):
    """Couples rows only in segments of 17+ rows: invisible to verify chunks of <= 16."""
    def _row_term(self, ctx, seg, li, r, x):
        return super()._row_term(ctx, seg, li, r, x) + (int(x.sum() % 1009) if seg.n >= 17 else 0)


class SidAlias32(ToyRuntime):
    """Keys KV storage by the low 32 bits of the sid."""
    _strict = False

    def open(self, sid, max_pos, spec_window):
        super().open(sid, max_pos, spec_window)
        me = self._sess[sid]
        for other, s in self._sess.items():
            if other != sid and other & 0xFFFFFFFF == sid & 0xFFFFFFFF and (s.max_pos, s.W) == (max_pos, spec_window):
                me.glob, me.swa, me.tag = s.glob, s.swa, s.tag
                break


def _page(p):
    return p if p < 192 else 128 + (p - 128) % 64       # a 3-page table that wraps its last page


class PosPageBug(ToyRuntime):
    """Global KV past position 191 wraps onto an earlier page."""
    def _write_kv(self, s, li, p, row, rebuilding):
        if self.kinds_all[li] == "global":
            if not (rebuilding and p < s.upto):
                s.glob[li][_page(p)] = row
        else:
            super()._write_kv(s, li, p, row, rebuilding)

    def _attend_global(self, s, li, p):
        coef = self._qw[:p + 1] * self._pow[p::-1] % P
        return (s.glob[li][[_page(q) for q in range(p + 1)]] * coef[:, None]).sum(0) % P


class KVBytesUnderreported(ToyRuntime):
    """describe() halves the global KV bytes and reports no SWA KV: admission over-promises."""
    def describe(self):
        c = super().describe()
        return dataclasses.replace(c, global_kv_bytes_per_pos=c.global_kv_bytes_per_pos // 2, swa_kv_bytes_per_pos=0)


class TopK0MeansOne(ToyRuntime):
    """Reads top_k=0 ("no cut") as k=1, i.e. greedy."""
    def _draw(self, logits_row, sampling, p, r):
        if sampling is not None and sampling.top_k == 0:
            sampling = dataclasses.replace(sampling, top_k=1)
        return toy_draw(logits_row, sampling, p)


class FakeServiceMs(ToyRuntime):
    """warmup() reports a flat 0.001 ms cost curve instead of measuring."""
    def warmup(self, buckets=(1, 8, 32), contexts=(0,), sessions=(1,)):
        super().warmup(buckets, contexts, sessions)
        self._service = tuple((n, b, c, 0.001) for n, b, c, _ in self._service)
        return self.describe()


class IgnoresRebuildFloor(ToyRuntime):
    """rebuild_from ignores the rollback floor: committed jumps to F, re-adoption cannot rewind."""
    def rebuild_from(self, sid, F, exact_from=None, floor=None):
        return super().rebuild_from(sid, F, exact_from)


class FloorWithoutEarlyStart(ToyRuntime):
    """Honours the floor as the commit frontier but rebuilds from the F rule: a rewind to the
    floor then reads SWA rows computed on partial windows."""
    def rebuild_from(self, sid, F, exact_from=None, floor=None):
        a = super().rebuild_from(sid, F, exact_from)
        if floor is not None:
            self._get(sid).committed = floor
        return a


class ExactFromOverridesFloor(ToyRuntime):
    """With both exact_from and a floor, starts the rebuild from exact_from alone."""
    def rebuild_from(self, sid, F, exact_from=None, floor=None):
        if exact_from is None or floor is None:
            return super().rebuild_from(sid, F, exact_from, floor)
        a = super().rebuild_from(sid, F, exact_from)
        self._get(sid).committed = floor
        return a


class FloorDepthCappedAt3(ToyRuntime):
    """Honours at most 3 rows of rollback floor below F."""
    def rebuild_from(self, sid, F, exact_from=None, floor=None):
        a = super().rebuild_from(sid, F, exact_from, floor)
        if floor is not None:
            self._get(sid).committed = max(floor, F - 3)
        return a


class ExportPastFrontier(ToyRuntime):
    """Mirrors rows up to the length, speculative or not (e.g. past a takeover's floor)."""
    def export_state(self, sid, lo_pos, hi_pos, kinds=("global",)):
        s = self._get(sid)
        c = s.committed
        s.committed = max(c, min(hi_pos, s.length))
        try:
            return super().export_state(sid, lo_pos, hi_pos, kinds)
        finally:
            s.committed = c


class CostTrueOnlyAtProbe(ToyRuntime):
    """Reports the (1 session, 32 tokens, context 0) cost honestly and every other shape 100x cheap."""
    def warmup(self, buckets=(1, 8, 32), contexts=(0,), sessions=(1,)):
        super().warmup(buckets, contexts, sessions)
        self._service = tuple((n, b, c, ms if (n, b, c) == (1, 32, 0) else ms / 100) for n, b, c, ms in self._service)
        return self.describe()


class SamplerPosMod256(ToyRuntime):
    """Keys the sampling RNG by an 8-bit position."""
    def _draw(self, logits_row, sampling, p, r):
        return toy_draw(logits_row, sampling, p & 0xFF)


class SeedTrunc32(ToyRuntime):
    """Keys the sampling RNG by the low 32 bits of the seed."""
    def _draw(self, logits_row, sampling, p, r):
        if sampling is not None:
            sampling = dataclasses.replace(sampling, seed=sampling.seed & 0xFFFFFFFF)
        return toy_draw(logits_row, sampling, p)


class TopKDroppedWithTopP(ToyRuntime):
    """Ignores top_k whenever top_p < 1."""
    def _draw(self, logits_row, sampling, p, r):
        if sampling is not None and sampling.top_p < 1:
            sampling = dataclasses.replace(sampling, top_k=0)
        return toy_draw(logits_row, sampling, p)


# mutant -> (geometry, the checks DESIGNED to catch it; each must fail in both numeric modes)
MUTANTS = {
    MirrorOverwrite: ("w4W4", {"check_rebuild", "check_export_import"}),
    IgnoresTruncate: ("w4W4", {"check_frontier"}),
    BatchCoupled: ("w4W4", {"check_batch_invariance", "check_mixed_kinds", "check_replay"}),
    WindowOffByOne: ("w4W4", {"check_reference", "check_swa_retention"}),
    LeaksRejectedRows: ("w4W4", {"check_frontier", "check_swa_retention", "check_row_independence"}),
    SwaNoSlack: ("w4W4", {"check_swa_retention"}),
    RowCoupled: ("w4W4", {"check_row_independence"}),
    TapOffByOne: ("w4W4", {"check_taps_raw", "check_reference"}),
    FirstDraftRebuildRule: ("w4W4", {"check_rebuild"}),
    NoopRebuild: ("w4W4", {"check_rebuild", "check_mixed_kinds"}),
    BlobRowsMasked: ("w4W4", {"check_export_import", "check_reference", "check_rebuild"}),
    SpecCheckUsesWindow: ("w6W3", {"check_frontier", "check_validation_atomic"}),
    RebuildUsesW: ("w6W3", {"check_rebuild"}),
    SidSlotAlias: ("w4W4", {"check_replay", "check_mixed_kinds"}),
    LogitsLastIsFirst: ("w4W4", {"check_tokens", "check_mixed_kinds"}),
    BatchCoupledOverBudget: ("w4W4", {"check_batch_invariance"}),
    ClaimsFast: ("w4W4", {"check_describe", "check_batch_invariance"}),
    RowCoupledLarge: ("w4W4", {"check_row_independence"}),
    SamplerKeyedByRow: ("w4W4", {"check_tokens", "check_reference"}),
    PrefillSkipsRollbackCheck: ("w4W4", {"check_validation_atomic"}),
    NoTokenRangeCheck: ("w4W4", {"check_validation_atomic"}),
    RowCoupledN17: ("w4W4", {"check_row_independence", "check_batch_invariance"}),
    SidAlias32: ("w4W4", {"check_replay", "check_mixed_kinds"}),
    PosPageBug: ("w4W4", {"check_long_context"}),
    KVBytesUnderreported: ("w4W4", {"check_describe"}),
    TopK0MeansOne: ("w4W4", {"check_tokens"}),
    FakeServiceMs: ("w4W4", {"check_describe"}),
    IgnoresRebuildFloor: ("w4W4", {"check_rebuild"}),
    FloorWithoutEarlyStart: ("w4W4", {"check_rebuild"}),
    ExactFromOverridesFloor: ("w4W4", {"check_rebuild"}),
    FloorDepthCappedAt3: ("w4W4", {"check_rebuild"}),
    ExportPastFrontier: ("w4W4", {"check_rebuild", "check_frontier", "check_export_import"}),
    CostTrueOnlyAtProbe: ("w4W4", {"check_describe"}),
    SamplerPosMod256: ("w4W4", {"check_long_context"}),
    SeedTrunc32: ("w4W4", {"check_tokens"}),
    TopKDroppedWithTopP: ("w4W4", {"check_tokens", "check_long_context"}),
}


@pytest.mark.parametrize("mode", ["chunk_invariant", "adversarial"])
@pytest.mark.parametrize("mutant", list(MUTANTS), ids=lambda m: m.__name__)
def test_mutant_is_caught(mutant, mode):
    geom, designed = MUTANTS[mutant]
    res = run_suite(ToyFactory(mode, cls=mutant, geom=geom), [c for c in CHECKS if c.__name__ in designed])
    escaped = {k: v for k, v in res.items() if not v.startswith("FAIL")}
    assert not escaped, f"{mutant.__name__} ({mode}, {geom}) escaped {escaped}"
