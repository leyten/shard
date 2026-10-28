"""shard/pspec/drafter.py: the drafter plug-in (BUILD-PLAN D1 gate).

  1. fused taps: fc_partial == concat-fc within the bf16 wire tolerance (exact in f32), and the
     fused bytes of a row are independent of the frame, the batch and the stage placement
  2. the drafter cache is never speculative: `propose` writes nothing, `advance` writes only rows
     whose inputs are committed (V4's test_cache_never_speculative, tests/test_v4_dspark.py:267)
  3. rejected state never leaks: a host that also judged off-path frames holds the same drafter
     state and proposes the same blocks as one fed only the committed stream
  4. the drafter rebuild is exact: DFlash from its last 1023 fused rows (window 1024), MTP-3 from
     its last 382 positions of ids + last hidden (3*127 + 1); one row fewer is not
  5. end to end: S1's coordinator over tests/scripted_pipe.py with a real (tiny) MTP-3 / DFlash on
     the tail: committed tokens == toy greedy, with drafts accepted and rejected
The lazy licence (the tail skipping blocks the entry proved it will not consume) is
tests/test_pspec_lazy.py; the reply's `lz` field is tests/test_frame.py.
"""
import numpy as np
import pytest

from scripted_pipe import SID, FusedTaps, LastHidden, run_pspec, toy_greedy

from shard.pspec import coord
from shard.pspec.drafter import (DFlashDrafter, Drafter, MTPDrafter, TailDrafterHost, TapSpec,
                                 bf16_round)

H = 16
MIMO_TAPS = (0, 11, 23, 35, 47)


def _rows(rng, n, h=H):
    return rng.standard_normal((n, h)).astype(np.float32)


def _ids(rng, n):
    return [int(t) for t in rng.integers(2, 100, size=n)]


# ── 1. fused taps ───────────────────────────────────────────────────────────────────────────────

def test_fused_equals_concat_within_tolerance():
    rng = np.random.default_rng(3)
    spec = TapSpec(MIMO_TAPS, H)
    fc = _rows(rng, H, len(MIMO_TAPS) * H)
    taps = {l: _rows(rng, 40) for l in spec.layers}
    W = spec.split(fc)
    ref = spec.concat(taps, fc)
    f32 = spec.fuse(taps, W, wire="f32")
    assert np.allclose(f32, ref, rtol=1e-5, atol=1e-5), "fc(concat) = sum_i W_i . h_i must hold"
    fused = spec.fuse(taps, W)                               # bf16 on the wire after every add
    assert np.array_equal(fused, bf16_round(fused)), "the tap lane carries bf16 values"
    bound = len(spec.layers) * 2.0 ** -8 * np.abs(ref).max(axis=1, keepdims=True) * 2
    assert (np.abs(fused - ref) <= bound).all(), "bf16 accumulation drifted past its rounding bound"


def test_fused_bytes_are_row_local_and_placement_invariant():
    """Receipts hash the tap lane: a row's bytes must not depend on the frame or batch that carried
    it, nor on how the tap layers are spread over hops (one rounding per tap layer, path order)."""
    rng = np.random.default_rng(4)
    spec = TapSpec(MIMO_TAPS, H)
    W = spec.split(_rows(rng, H, len(MIMO_TAPS) * H))
    taps = {l: _rows(rng, 30) for l in spec.layers}
    whole = spec.fuse(taps, W)
    for cuts in ([1] * 30, [7, 1, 22], [29, 1], [3] * 10):
        parts, lo = [], 0
        for c in cuts:
            parts.append(spec.fuse({l: t[lo:lo + c] for l, t in taps.items()}, W))
            lo += c
        assert np.concatenate(parts).tobytes() == whole.tobytes(), f"frame split {cuts} changed bytes"
    for D in (1, 2, 3, 5, 7, 48):
        spans = [(k * 48 // D, (k + 1) * 48 // D) for k in range(D)]
        assert spec.fuse(taps, W, spans=spans).tobytes() == whole.tobytes(), f"{D} hops changed bytes"
    # the order IS fixed: accumulating the same taps in another order is a different wire value
    rev = None
    for l in reversed(spec.layers):
        rev = spec.partial(rev, W[l], taps[l])
    assert rev.tobytes() != whole.tobytes(), "this test cannot see an order change"


def test_bf16_round_is_the_wire_cast():
    """The tap lane's rounding is torch's float32 -> bfloat16 cast, bit for bit on every finite
    value and +-inf. NaN stays a NaN but its payload is not canonical across devices, which is
    why `partial` refuses non-finite rows (next test)."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(9)
    x = (rng.standard_normal(50000) * 10.0 ** rng.uniform(-44, 38, 50000)).astype(np.float32)
    x = np.concatenate([x, np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 3.3895314e38, 1e-45],
                                    np.float32), np.array([0x7F800001], np.uint32).view(np.float32)])
    got = bf16_round(x)
    want = torch.from_numpy(x.copy()).to(torch.bfloat16).float().numpy()
    nan = np.isnan(want)
    assert np.array_equal(np.isnan(got), nan)
    assert np.array_equal(got[~nan].view(np.uint32), want[~nan].view(np.uint32))


def test_tap_lane_refuses_non_finite_rows():
    spec = TapSpec(MIMO_TAPS, H)
    W = np.eye(H, dtype=np.float32)
    ok = np.ones((2, H), np.float32)
    for bad in (np.nan, np.inf, 3.4e38):
        h = ok.copy()
        h[1, 3] = bad
        acc = None if bad != 3.4e38 else ok * 3.4e38          # overflow to inf in the add
        with pytest.raises(ValueError, match="non-finite"), np.errstate(over="ignore", invalid="ignore"):
            spec.partial(acc, W, h)


def test_tapspec_rejects_bad_specs():
    with pytest.raises(ValueError):
        TapSpec((11, 0), H)
    with pytest.raises(ValueError):
        TapSpec(MIMO_TAPS, H).split(np.zeros((H, H), np.float32))


def test_reference_drafters_satisfy_the_protocol():
    for d in (MTPDrafter(), DFlashDrafter()):
        assert isinstance(d, Drafter) and d.location == "tail"
    assert MTPDrafter().block == 3 and MTPDrafter().taps is None
    assert DFlashDrafter().block == 7 and DFlashDrafter().taps.layers == MIMO_TAPS
    assert MTPDrafter(window=128).rebuild_rows == 382 and DFlashDrafter(window=1024).rebuild_rows == 1023


# ── 2. never speculative ────────────────────────────────────────────────────────────────────────

def _cached_positions(d, sid):
    s = d._s[sid]
    return [sorted(k) for k in s["K"]]


@pytest.mark.parametrize("make", [lambda: MTPDrafter(window=16), lambda: DFlashDrafter(window=32)])
def test_cache_never_speculative(make):
    rng = np.random.default_rng(5)
    d = make()
    P = 12
    d.open("s", _ids(rng, P + 1), _rows(rng, P))
    c = P + 1
    for step in range(6):
        before = d.state_digest("s")
        blk = d.propose("s", d.frontier("s"))
        blk2 = d.propose("s", d.frontier("s"))
        assert blk == blk2 and d.state_digest("s") == before, "propose wrote drafter state"
        for k, ps in enumerate(_cached_positions(d, "s")):
            lag = k if isinstance(d, MTPDrafter) else 0
            assert ps and ps[-1] == c - 2 - lag, \
                f"layer/depth {k}: cached through {ps[-1]}, committed rows end at {c - 2 - lag}"
        m = 1 + step % 3
        d.advance("s", c - 1, _ids(rng, m), _rows(rng, m))
        c += m


def test_position_discipline_refuses_loudly():
    rng = np.random.default_rng(6)
    d = MTPDrafter(window=8)
    d.open("s", _ids(rng, 5), _rows(rng, 4))
    with pytest.raises(RuntimeError):
        d.advance("s", 5, [1], _rows(rng, 1))                   # a gap: the newest id's row is 4
    with pytest.raises(RuntimeError):
        d.propose("s", 4)                                        # frontier is 3
    with pytest.raises(RuntimeError):
        d.open("s", _ids(rng, 5), _rows(rng, 4))                 # already open
    with pytest.raises(RuntimeError):
        d.open("t", _ids(rng, 5), _rows(rng, 5))                 # ctx rows != ids - 1


@pytest.mark.parametrize("make", [lambda: MTPDrafter(window=16), lambda: DFlashDrafter(window=32)])
def test_a_history_shorter_than_the_block_drafts(make):
    """A 1-token prompt, or a rebuild from a single row: fewer positions than MTP's depth. Rows
    before the session's start do not exist; the drafter must draft from what it has."""
    rng = np.random.default_rng(10)
    d = make()
    d.open("s", _ids(rng, 2), _rows(rng, 1))
    assert len(d.propose("s", 0)) == d.block
    d.advance("s", 1, [5], _rows(rng, 1))
    assert len(d.propose("s", 1)) == d.block
    r = make()
    r.rebuild("s", 37, _ids(rng, 2), _rows(rng, 1))
    assert len(r.propose("s", 37)) == r.block


# ── 3. rejected state never leaks ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("make", [lambda: MTPDrafter(window=16), lambda: DFlashDrafter(window=32)])
def test_rejected_frames_never_reach_the_drafter(make):
    """Two hosts see the same committed stream; one also judges off-path frames (stale futures of
    a cancelled speculation, a wrong correction) carrying their own ctx rows. Same state, same
    blocks, every step."""
    rng = np.random.default_rng(7)
    clean, noisy = TailDrafterHost(make()), TailDrafterHost(make())
    P = 10
    ids, ctx = _ids(rng, P), _rows(rng, P)
    b1, b2 = clean.prefill(SID, ids, ctx, 42), noisy.prefill(SID, ids, ctx, 42)
    assert b1 == b2
    f = clean.front[SID]
    for step in range(12):
        q, m = f.cfront + 1, f.mfront
        n = 1 + step % 3
        fed = [m] + _ids(rng, n - 1)
        toks = _ids(rng, n)
        rows = _rows(rng, n)
        # off-path noise first: a frame past the frontier, and one at the frontier with a wrong id
        for bad_pos, bad_fed in ((q + 1, fed), (q, [m + 1] + fed[1:])):
            acc, blk, lz = noisy.on_frame(SID, bad_pos, bad_fed, toks, _rows(rng, n) * 9)
            assert (acc, blk, lz) == (False, None, False)
        a = clean.on_frame(SID, q, fed, toks, rows)
        b = noisy.on_frame(SID, q, fed, toks, rows)
        assert a == b and a[0] is True
        assert clean.d.state_digest(SID) == noisy.d.state_digest(SID), "rejected state leaked"


# ── 4. rebuild exact ────────────────────────────────────────────────────────────────────────────

def _history(d, rng, total, P=6):
    ids, ctx = _ids(rng, P + 1), _rows(rng, P)
    d.open("full", ids, ctx)
    while len(ids) < total:
        m = int(rng.integers(1, 9))
        new, rows = _ids(rng, m), _rows(rng, m)
        d.advance("full", len(ids) - 1, new, rows)
        ids += new
        ctx = np.concatenate([ctx, rows])
    return ids, ctx


@pytest.mark.parametrize("kind", ["dflash", "mtp"])
def test_drafter_rebuild_exact(kind):
    """A takeover tail re-opens the drafter from the log's last rows only (ARCHITECTURE-v2 §4.5):
    DFlash re-projects the last window-1 = 1023 fused rows; MTP-3 recomputes its 3 SWA depths over
    3*127+1 = 382 positions of ids + last hidden. Bitwise equal to the drafter that saw it all, now
    and after further commits; one row fewer is not."""
    rng = np.random.default_rng(8)
    make = (lambda: DFlashDrafter(window=1024)) if kind == "dflash" else (lambda: MTPDrafter(window=128))
    d = make()
    ids, ctx = _history(d, rng, d.rebuild_rows + 60)
    need = d.rebuild_rows
    c = len(ids)
    for rows, exact in ((need - 1, False), (need, True)):
        r = make()
        s = c - 1 - rows                                         # ctx rows s..c-2, ids s..c-1
        r.rebuild("full", s, ids[s:], ctx[s:])
        same = np.array_equal(r.draft_logits("full"), d.draft_logits("full"))
        assert same == exact, f"rebuild from {rows} rows: exact={same}, want {exact}"
        if exact:
            for _ in range(3):
                m = int(rng.integers(1, 5))
                new, nrows = _ids(rng, m), _rows(rng, m)
                for x in (d, r):
                    x.advance("full", len(ids) - 1, new, nrows)
                ids += new
                assert r.propose("full", r.frontier("full")) == d.propose("full", d.frontier("full"))
                assert np.array_equal(r.draft_logits("full"), d.draft_logits("full"))


# ── 5. end to end with a real drafter ───────────────────────────────────────────────────────────

def _ctx_rows(ctx, D, hist, pos, n):
    acc = None
    for k in range(D):
        acc = ctx.stage(k, D, hist, pos, n, acc)
    return ctx.tail_rows(acc, hist, pos, n)


def _truth_for(make, ctx, D, n, P=9, alpha=0.6, seed=2):
    """A greedy stream the drafter partly predicts: at each position the model's token is the
    drafter's first proposal (from the committed history, exactly as the tail will see it) with
    probability alpha, else a random token. Drafts deeper than 1 hit only by coincidence."""
    rng = np.random.default_rng(seed)
    Tt = _ids(rng, P + 1)
    d = make()
    d.open("x", Tt, _ctx_rows(ctx, D, Tt, 0, P))
    while len(Tt) < n:
        first = d.propose("x", len(Tt) - 2)[0]
        t = int(first) if rng.random() < alpha else int(rng.integers(2, 100))
        Tt.append(t)
        d.advance("x", len(Tt) - 2, [t], _ctx_rows(ctx, D, Tt, len(Tt) - 2, 1))
    return Tt


@pytest.mark.parametrize("policy", coord.POLICIES)
@pytest.mark.parametrize("kind", ["mtp", "dflash"])
def test_end_to_end_with_a_real_drafter(kind, policy, lazy=False):
    D = 4
    if kind == "mtp":
        make, ctx = (lambda: MTPDrafter(window=16)), LastHidden(H)
    else:
        spec = TapSpec(MIMO_TAPS, H)
        make = lambda: DFlashDrafter(window=32)                   # noqa: E731
        ctx = FusedTaps(spec, make().fc)
    Tt = _truth_for(make, ctx, D, 160)
    P, n = 9, 60
    host = TailDrafterHost(make())
    r, pipe, _ = run_pspec(Tt, P, n, depth=D, W=8, policy=policy, host=host, ctx=ctx, lazy=lazy)
    assert r["tokens"] == toy_greedy(Tt, P, n), "the committed stream left the greedy stream"
    assert r["accepted"] > 0 and r["cancels"] > 0, (r["accepted"], r["cancels"])
    assert host.drafted > 0


@pytest.mark.parametrize("kind", ["mtp", "dflash"])
def test_end_to_end_one_token_prompt(kind):
    ctx = LastHidden(H) if kind == "mtp" else FusedTaps(TapSpec(MIMO_TAPS, H), DFlashDrafter().fc)
    make = MTPDrafter if kind == "mtp" else DFlashDrafter
    Tt = _truth_for(make, ctx, 3, 60, P=1)
    r, *_ = run_pspec(Tt, 1, 30, W=8, policy="pertoken", host=TailDrafterHost(make()), ctx=ctx)
    assert r["tokens"] == toy_greedy(Tt, 1, 30)
