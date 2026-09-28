"""ToyRuntime passes the StageRuntime conformance suite; wrong runtimes do not (BUILD-PLAN F3).

Two parts:
  1. tests/runtime_conformance.py's CHECKS on ToyRuntime in both numeric modes:
     "chunk_invariant" (integer, schedule-free bytes: the rebuild must be bit-equal to the
     fault-free state) and "adversarial" (bytes depend on each segment's (start_pos, n), like
     real float kernels: everything must still hold except chunk invariance, and the rebuild
     falls back to replay-equality).
  2. Anti-vacuity: mutant runtimes, each one realistic bug on top of the toy, must FAIL the
     checks listed for them. A suite a wrong runtime can pass proves nothing.
"""
import numpy as np
import pytest

from runtime_conformance import (CHECKS, MAX_CTX, REBUILD_PATTERNS, W_SPEC, Skip, _find,
                                 _seg, run_suite)
from shard.runtime.api import Rows, StageSpec, StepIn, rebuild_start
from shard.runtime.toy import P, ToyRuntime, ToyWeights, reference_forward

PATTERN = "SSSSSGSGSSGG"      # holds every gate span: SSG, SGS, SSSSSG, SGSGS (+ GSS, SSS, GG)
WINDOW = 4
TAPS = (0, 4, 7, 11)          # DFlash-like: first layer, spread, and the last layer (on the tail)
HIDDEN, VOCAB = 4, 16


class ToyFactory:
    """RuntimeFactory for ToyRuntime (or a subclass of it: the mutants)."""
    layer_kinds = tuple("swa" if c == "S" else "global" for c in PATTERN)
    window = WINDOW
    taps = TAPS

    def __init__(self, mode="chunk_invariant", cls=ToyRuntime):
        self.mode, self.cls = mode, cls
        self.name = f"{cls.__name__}[{mode}]"
        self.weights = ToyWeights(PATTERN, hidden=HIDDEN, vocab=VOCAB, taps=TAPS)

    def _build(self, lo, hi, is_head, is_tail, tap_mode="raw", spec_window=W_SPEC,
               max_sessions=8, max_ctx=MAX_CTX, kv_budget_bytes=0):
        spec = StageSpec("toy", lo, hi, is_head=is_head, is_tail=is_tail, taps=TAPS,
                         tap_mode=tap_mode, max_sessions=max_sessions, max_ctx=max_ctx,
                         spec_window=spec_window, kv_budget_bytes=kv_budget_bytes)
        return self.cls(spec, pattern=PATTERN, window=WINDOW, hidden=HIDDEN, vocab=VOCAB,
                        mode=self.mode)

    def make(self, lo, hi, *, is_head=False, is_tail=False, **kw):
        rt = self._build(lo, hi, is_head, is_tail, **kw)
        rt.load(self.weights)
        return rt

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

    def oracle(self, lo, hi, *, is_head, is_tail, x, schedule):
        return reference_forward(self.weights, PATTERN, WINDOW, lo, hi, is_head=is_head,
                                 is_tail=is_tail, x=x, schedule=schedule, mode=self.mode, taps=TAPS)

    def make_broken(self, lo, hi):
        rt = self._build(lo, hi, lo == 0, hi == len(PATTERN))
        rt.load(self.weights.without(f"layers.{lo}.b"))


FACTORIES = [ToyFactory("chunk_invariant"), ToyFactory("adversarial")]


@pytest.mark.parametrize("factory", FACTORIES, ids=lambda f: f.mode)
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
    F = 50
    assert 0 < rebuild_start(kinds("SSSSSG"), WINDOW, F, F - 5) < F - 5 - 3
    assert rebuild_start(kinds("SGSGS"), WINDOW, F, F - 1) == F - 3 * (WINDOW - 1)      # F term binds
    assert rebuild_start(kinds("SGSGS"), WINDOW, F, F - 10) == F - 10 - 2 * (WINDOW - 1)  # upper G binds
    assert rebuild_start(kinds("SSS"), WINDOW, F, 0) == F - 3 * (WINDOW - 1)


def test_adversarial_mode_really_depends_on_shape():
    """In adversarial mode, chunking the same rows differently changes bytes (so the suite's
    adversarial pass shows it never leans on chunk invariance), while the oracle agrees for a
    given schedule."""
    f = FACTORIES[1]
    x = f.ids(12, 1)
    outs = []
    for sched in ([(0, 12)], [(0, 5), (5, 7)]):
        rt = f.make(0, len(PATTERN), is_head=True, is_tail=True)
        rt.open(1, MAX_CTX, W_SPEC)
        rt.commit(1, 12)
        o = [rt.step(StepIn((_seg(1, s, x.slice(s, s + n)),))).outs[0] for s, n in sched]
        outs.append(Rows.concat([q.hidden for q in o]))
        ref = f.oracle(0, len(PATTERN), is_head=True, is_tail=True, x=x, schedule=sched)
        assert outs[-1] == ref["hidden"]
    assert outs[0] != outs[1]


# ── mutants: each one realistic bug; each must fail the checks listed for it ──────────────────

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
    def rebuild_from(self, sid, F):
        s = self._get(sid)
        upto = s.upto
        a = super().rebuild_from(sid, F)
        if any(k == "global" for k in self.kinds):
            late = max(0, min(upto, F - (self.window - 1) * sum(k == "swa" for k in self.kinds)))
            if late > a:
                s.length = s.swa_lo = a = late
                s.rebuild_hi = F if a < F else None
        return a


# The checks each mutant is DESIGNED to trip (it may trip more).
MUTANTS = {
    MirrorOverwrite: {"check_rebuild"},
    IgnoresTruncate: {"check_frontier"},
    BatchCoupled: {"check_batch_invariance", "check_mixed_kinds", "check_replay"},
    WindowOffByOne: {"check_oracle", "check_swa_retention"},
    LeaksRejectedRows: {"check_frontier", "check_swa_retention", "check_row_independence"},
    SwaNoSlack: {"check_swa_retention"},
    RowCoupled: {"check_row_independence"},
    TapOffByOne: {"check_taps_raw", "check_oracle"},
    FirstDraftRebuildRule: {"check_rebuild"},
}


@pytest.mark.parametrize("mode", ["chunk_invariant", "adversarial"])
@pytest.mark.parametrize("mutant", list(MUTANTS), ids=lambda m: m.__name__)
def test_mutant_is_caught(mutant, mode):
    res = run_suite(ToyFactory(mode, cls=mutant))
    failed = {k for k, v in res.items() if v.startswith("FAIL")}
    assert failed, f"{mutant.__name__} passes the whole suite in {mode} mode: {res}"
    if mode == "chunk_invariant":
        missing = MUTANTS[mutant] - failed
        assert not missing, f"{mutant.__name__} escaped {sorted(missing)}: {res}"
