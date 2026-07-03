"""Offline integration tests for the m25_pipe coordinators over the CPU fake ring (tests/fake_ring.py):
losslessness, the EAGLE extend-pairing (left-shift) contract, chain-vs-tree accounting on identical
synthetic text (the rag-quote g-gap probe), the stage-timing transport breakdown, and mid-stream
divergence bookkeeping. No GPU, no model, no network — the ring is a teacher-forced oracle, so
`output == T-prefix` is exact.

Run: python3 -m pytest tests/test_fake_ring.py -q     (-s to see the accounting-comparison table)
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
fr = pytest.importorskip("fake_ring")               # bootstraps env + imports m25_pipe on CPU

from ngram_draft import NgramDrafter                # noqa: E402
from eagle_draft import EagleDrafter, HybridDrafter  # noqa: E402
from test_eagle_draft import _make_head             # noqa: E402  (synthetic EAGLE-3 head, H=32/vocab=100)

MP = fr.MP
S = fr.S

P = 60                                              # prompt length (a T prefix, via FakeTok)


def _ngram():
    # margin=64 (not the 256 default): tests commit a few hundred tokens, and 64 still clears the
    # speculative lookahead (2*depth*K = 64 at depth=4, K=8) the margin exists to exclude.
    return NgramDrafter(ng=3, min_match=1, margin=64)


def _eagle(seed=0):
    d, embed = _make_head(seed)
    return EagleDrafter(d, embed, device="cpu", next_hidden="prenorm")


def _hybrid(seed=0, record=False):
    """HybridDrafter with the REAL EagleDrafter (synthetic weights). record=True wraps the EAGLE half
    in a RecordingDrafter so every extend() on BOTH paths (hybrid delegates; the tree path calls
    local_draft.eagle directly) is captured."""
    eg = fr.RecordingDrafter(_eagle(seed)) if record else _eagle(seed)
    return HybridDrafter(_ngram(), eg), eg


def _T(flavor, n):
    return fr.novel_T(n) if flavor == "novel" else fr.repetitive_T(n)


def _assert_lossless(res, T, prompt_len, max_new):
    assert res["ok"], res
    out = res["output_ids"]
    assert res["n_tokens"] == len(out)
    assert len(out) >= max_new, f"stopped early: {len(out)} < {max_new} (eos is not in T)"
    assert out == T[prompt_len:prompt_len + len(out)], (
        f"LOSSLESSNESS BROKEN: committed output diverges from the oracle continuation at "
        f"index {next(i for i, (a, b) in enumerate(zip(out, T[prompt_len:])) if a != b)}")


def _tree_env(monkeypatch, m=12, topb=3, depth=8):
    monkeypatch.setattr(S, "M25_EAGLE", True)
    monkeypatch.setattr(S, "M25_TREE", True)
    monkeypatch.setenv("M25_TREE_M", str(m))
    monkeypatch.setenv("M25_TREE_TOPB", str(topb))
    monkeypatch.setenv("M25_TREE_DEPTH", str(depth))


# ---- 1. LOSSLESSNESS ------------------------------------------------------------------------------

@pytest.mark.parametrize("prefill_chunk", [24, 4096])       # chunked (60-token prompt -> 3 chunks) / whole
@pytest.mark.parametrize("flavor", ["novel", "repetitive"])
def test_lossless_chain_ngram(flavor, prefill_chunk):
    """Plain chain path (M25_EAGLE off), pure NgramDrafter, depth-pipelined."""
    T = _T(flavor, 460)
    res, ring = fr.run_coordinator(T, P, _ngram(), K=8, depth=4, max_new=160,
                                   prefill_chunk=prefill_chunk, eagle_ring=False)
    _assert_lossless(res, T, P, 160)
    n_pf = sum(1 for e in ring.log if e["op"] == "verify" and e["prefill"])
    assert n_pf == (3 if prefill_chunk == 24 else 0)        # unchunked prefill carries no prefill flag


@pytest.mark.parametrize("prefill_chunk", [24, 4096])
@pytest.mark.parametrize("flavor", ["novel", "repetitive"])
def test_lossless_chain_hybrid_eagle(flavor, prefill_chunk, monkeypatch):
    """Chain path under M25_EAGLE (HybridDrafter, depth forced to 1, aux riding every reply)."""
    monkeypatch.setattr(S, "M25_EAGLE", True)
    T = _T(flavor, 420)
    hyb, _ = _hybrid()
    res, ring = fr.run_coordinator(T, P, hyb, K=8, depth=4, max_new=120,
                                   prefill_chunk=prefill_chunk, eagle_ring=True)
    _assert_lossless(res, T, P, 120)


@pytest.mark.parametrize("prefill_chunk", [24, 4096])
@pytest.mark.parametrize("flavor", ["novel", "repetitive"])
def test_lossless_tree(flavor, prefill_chunk, monkeypatch):
    """Tree path (S.M25_TREE routes coordinate_pipe -> coordinate_pipe_tree), hybrid n-gram/EAGLE-tree."""
    _tree_env(monkeypatch, depth=4)                         # small tree keeps CPU runtime tight
    T = _T(flavor, 420)
    hyb, _ = _hybrid()
    res, ring = fr.run_coordinator(T, P, hyb, K=8, depth=4, max_new=120,
                                   prefill_chunk=prefill_chunk, eagle_ring=True)
    _assert_lossless(res, T, P, 120)
    assert any(e["tree"] for e in ring.log if e["op"] == "verify"), "tree path never sent a tree verify"


# ---- 2. EXTEND-PAIRING (the EAGLE left-shift contract, ±1 arithmetic) ------------------------------

@pytest.mark.parametrize("path", ["chain", "tree"])
@pytest.mark.parametrize("flavor", ["novel", "repetitive"])
def test_extend_pairing_invariant(path, flavor, monkeypatch):
    """For EVERY extend(tokens, auxes, base_pos) — prefill chunks AND decode commits, both paths:
      * auxes[i] is the target hidden at absolute position base_pos+i (the fake ring encodes the
        position as the aux VALUE, so this is exact);
      * tokens[i] == T[base_pos+i+1]  (each token pairs with the hidden ONE position earlier — the
        hidden that predicted it);
      * successive extends tile positions contiguously from 0 with no gap/overlap, and the
        concatenated tokens are exactly T[1:1+N] (the global left-shift of the committed stream).
    This is the test that catches the historical off-by-one bugs in prefill seeding, extend-on-commit
    pairing, divergence base_pos, and the tree's trunk/vbase/pred_idx arithmetic."""
    monkeypatch.setattr(S, "M25_EAGLE", True)
    if path == "tree":
        _tree_env(monkeypatch, depth=4)
    T = _T(flavor, 420)
    hyb, rec = _hybrid(record=True)
    res, ring = fr.run_coordinator(T, P, hyb, K=8, depth=4, max_new=100,
                                   prefill_chunk=24, eagle_ring=True)
    _assert_lossless(res, T, P, 100)
    assert rec.extends, "no extend() calls recorded"
    for toks, aux, base in rec.extends:
        assert aux.shape[0] == len(toks) and aux.shape[1] == 3, aux.shape
        for i, t in enumerate(toks):
            want = base + i
            got = aux[i].flatten()
            assert bool(torch.all(got == float(want))), (
                f"aux[{i}] encodes position {got[0].item():.0f}, expected base_pos+{i}={want} "
                f"(extend base_pos={base}, path={path})")
            assert t == T[want + 1], (
                f"tokens[{i}]={t} != T[base_pos+{i}+1]=T[{want + 1}]={T[want + 1]} — "
                f"the EAGLE left-shift is broken (path={path})")
    assert rec.extends[0][2] == 0, "first (prefill) extend must start at position 0"
    cur, all_toks = 0, []
    for toks, aux, base in rec.extends:
        assert base == cur, f"extend base_pos {base} != expected {cur} — position gap/overlap"
        cur += len(toks)
        all_toks += toks
    assert all_toks == T[1:1 + len(all_toks)], "concatenated extend tokens != T left-shifted by one"


# ---- 3. ACCOUNTING COMPARISON (the rag-quote g-gap probe) ------------------------------------------

def _run_cell(T, prompt_len, monkeypatch, *, tree, tree_depth=8, K=8, max_new=200, seed=0):
    if tree:
        _tree_env(monkeypatch, m=12, topb=3, depth=tree_depth)
    else:
        monkeypatch.setattr(S, "M25_EAGLE", True)
        monkeypatch.setattr(S, "M25_TREE", False)
    commits = []
    hyb, _ = _hybrid(seed)
    res, ring = fr.run_coordinator(T, prompt_len, hyb, K=K, depth=4, max_new=max_new,
                                   prefill_chunk=4096, eagle_ring=True,
                                   on_commit=lambda out, dt: commits.append(len(out)))
    per_round = [b - a for a, b in zip(commits, commits[1:])]           # commits[0] = the prefill token
    decode = [e for e in ring.log if e["op"] == "verify" and not e["prefill"] and e["start"] > 0]
    tree_rounds = [e for e in decode if e["tree"]]
    return res, per_round, decode, tree_rounds


def test_accounting_chain_vs_tree_repetitive(monkeypatch, capsys):
    """Chain-hybrid vs tree-hybrid on IDENTICAL repetitive T with the SAME NgramDrafter config —
    the mechanical half of the warm rag-quote gap (chain g=5.4 vs tree g=3.8). Reports rounds,
    committed, g, real committed-per-traversal, and per-round commit distribution for:
      chain (K=8) | tree TREE_DEPTH=8 (equal budget) | tree TREE_DEPTH=4 (the budget-coupling cell).
    Losslessness is hard-asserted; the comparison itself is reported + sanity-checked loosely."""
    T = fr.repetitive_T(760)
    PL, MAXNEW = 128, 200                       # longer prompt: n-gram index warm from round 1
    cells = {}
    for name, tree, td in (("chain_K8", False, None), ("tree_d8", True, 8), ("tree_d4", True, 4)):
        res, per_round, decode, tree_rounds = _run_cell(T, PL, monkeypatch, tree=tree,
                                                        tree_depth=td or 8, max_new=MAXNEW)
        _assert_lossless(res, T, PL, MAXNEW)
        real_g = (res["n_tokens"] - 1) / max(res["rounds"], 1)          # -1: the prefill token isn't a round's work
        cells[name] = {"res": res, "per_round": per_round, "real_g": real_g,
                       "wire_tokens_per_round": sum(e["n"] for e in decode) / max(len(decode), 1)}
    lines = ["", "=== ACCOUNTING: chain vs tree on identical repetitive T (same NgramDrafter) ==="]
    lines.append(f"{'cell':>10} {'rounds':>6} {'ntok':>5} {'reported_g':>10} {'real_g':>7} "
                 f"{'wire_tok/rnd':>12}  per-round commits (first 12)")
    for name, c in cells.items():
        r = c["res"]
        lines.append(f"{name:>10} {r['rounds']:>6} {r['n_tokens']:>5} {r['toks_per_traversal']:>10.2f} "
                     f"{c['real_g']:>7.2f} {c['wire_tokens_per_round']:>12.1f}  {c['per_round'][:12]}")
    print("\n".join(lines))
    ch, t8, t4 = cells["chain_K8"], cells["tree_d8"], cells["tree_d4"]
    # sanity (loose, report-style): at EQUAL budget the tree path must not be mechanically worse —
    # it commits the bonus token the chain path discards on full accepts.
    assert t8["real_g"] >= ch["real_g"] - 0.25, (
        f"tree_d8 real g {t8['real_g']:.2f} mechanically below chain {ch['real_g']:.2f} — "
        f"a bookkeeping cause for the warm rag gap DOES exist; inspect per-round commits above")
    # DEPTH/K coupling is now FIXED: the tree hybrid requests K n-gram draft tokens regardless of
    # TREE_DEPTH, so a shallow tree no longer caps verbatim g. tree_d4 must now match chain on
    # repetitive text (the n-gram path is identical; only novel rounds differ by tree depth).
    assert t4["real_g"] >= ch["real_g"] - 0.25, (
        f"tree_d4 real g {t4['real_g']:.2f} still capped below chain {ch['real_g']:.2f} — "
        f"the n-gram draft is being limited by TREE_DEPTH again")
    # both paths' reported g is now exact (committed frontier / rounds) — the full-accept bonus is
    # committed under depth-1 and counted, so there is no chain/tree reporting asymmetry left.
    skew_chain = ch["res"]["toks_per_traversal"] - ch["real_g"]
    skew_tree = t8["res"]["toks_per_traversal"] - t8["real_g"]
    print(f"reported-g minus real-g: chain {skew_chain:+.2f}, tree {skew_tree:+.2f} (both ~0 = honest)")
    assert abs(skew_tree) < 0.05, "tree reported g should equal committed/rounds exactly"
    assert abs(skew_chain) < 0.05, "chain reported g should now equal committed/rounds (bonus committed + honest metric)"


def test_accounting_same_drafter_novel(monkeypatch, capsys):
    """Same probe on NOVEL text (EAGLE-only rounds): chain-EAGLE K=8 vs tree M=12/topb=3/depth=8.
    Report-only — proposal quality differs by geometry (chain vs best-first tree), the harness pins
    the ACCOUNTING: both must be lossless and g must equal committed/rounds."""
    T = fr.novel_T(560)
    cells = {}
    for name, tree in (("chain_K8", False), ("tree_m12", True)):
        res, per_round, decode, _ = _run_cell(T, P, monkeypatch, tree=tree, max_new=120)
        _assert_lossless(res, T, P, 120)
        real_g = (res["n_tokens"] - 1) / max(res["rounds"], 1)
        cells[name] = (res, real_g)
        assert abs(sum(per_round) - (res["n_tokens"] - 1)) <= 8, "per-round commits don't sum to output"
    print("\n=== ACCOUNTING novel text ===")
    for name, (res, real_g) in cells.items():
        print(f"{name:>10}: rounds={res['rounds']} ntok={res['n_tokens']} "
              f"reported_g={res['toks_per_traversal']:.2f} real_g={real_g:.2f}")


# ---- 4. DEPTH-AWARE HYBRID (matched rounds = pipelined plain frames; novel rounds = sync tree) -----

def test_hybrid_routes_matched_rounds_as_plain_frames(monkeypatch):
    """On repetitive text the M25_TREE coordinator must send matched n-gram rounds as PLAIN chain
    frames (flash kernel, small payload — the 2026-07-03 receipt showed 1-wide TREE framing paid
    199-303ms/round vs 139ms for zero accept gain), reserving tree frames for novel rounds."""
    _tree_env(monkeypatch, depth=8)
    T = fr.repetitive_T(760)
    hyb, _ = _hybrid()
    res, ring = fr.run_coordinator(T, 128, hyb, K=8, depth=4, max_new=200,
                                   prefill_chunk=4096, eagle_ring=True)
    _assert_lossless(res, T, 128, 200)
    decode = [e for e in ring.log if e["op"] == "verify" and not e["prefill"]]
    plain = [e for e in decode if not e["tree"]]
    trees = [e for e in decode if e["tree"]]
    assert len(plain) > len(trees), (
        f"matched rounds still ride tree frames: {len(plain)} plain vs {len(trees)} tree")


def test_hybrid_pipelines_matched_bursts(monkeypatch):
    """Matched streaks must run depth>1: the ring STALLS its first 3 decode replies (200ms) — a
    pipelining coordinator fills its depth window during the stall, so the ring finds more frames
    already buffered when it wakes ('backlog'). The old synchronous one-round-in-flight tree loop
    cannot send frame N+1 before reply N and scores backlog == 0, deterministically."""
    _tree_env(monkeypatch, depth=8)
    T = fr.repetitive_T(760)
    hyb, _ = _hybrid()
    res, ring = fr.run_coordinator(T, 128, hyb, K=8, depth=4, max_new=200,
                                   prefill_chunk=4096, eagle_ring=True, stall_decode=(3, 0.2))
    _assert_lossless(res, T, 128, 200)
    assert ring.backlog >= 2, f"no pipelining evidence on a pure verbatim run (backlog={ring.backlog})"


def test_hybrid_novel_text_stays_sync_tree(monkeypatch):
    """On novel text the hybrid must remain the synchronous tree: tree frames dominate and there is
    no burst pipelining (EAGLE needs the verified hidden — depth-1 is structural, not a bug)."""
    _tree_env(monkeypatch, depth=4)
    T = fr.novel_T(560)
    hyb, _ = _hybrid()
    res, ring = fr.run_coordinator(T, P, hyb, K=8, depth=4, max_new=120,
                                   prefill_chunk=4096, eagle_ring=True)
    _assert_lossless(res, T, P, 120)
    decode = [e for e in ring.log if e["op"] == "verify" and not e["prefill"]]
    trees = [e for e in decode if e["tree"]]
    assert len(trees) >= len(decode) - len(trees), "novel text routed away from tree rounds"


# ---- 5. STAGE-TIMING / TRANSPORT BREAKDOWN ---------------------------------------------------------

SPANS = [[0, 5.0, 4.0], [1, 3.25, 2.5], [2, 1.5, 1.0]]     # [stage, span_ms, comp_ms]; binary-exact floats


@pytest.mark.parametrize("path", ["ngram-pipelined", "chain-eagle", "tree"])
def test_transport_breakdown_adds_up(path, monkeypatch):
    """Stages under M25_STAGE_TIMING return [stage, span_ms, comp_ms] rows on every verify reply; the
    coordinator's split must tile exactly: every decode reply it folds (rounds + wasted — drained
    chunks are not folded) contributes one full row set, transport_s == traversal_s - stage_s, and
    per_stage_ms means reproduce the injected constants. All three decode loops must carry it."""
    T = fr.repetitive_T(760)
    if path == "tree":
        _tree_env(monkeypatch, depth=8)
        drafter, eagle_ring = _hybrid()[0], True
    elif path == "chain-eagle":
        monkeypatch.setattr(S, "M25_EAGLE", True)
        drafter, eagle_ring = _hybrid()[0], True
    else:                                                   # plain pipelined n-gram: bare-list reply promoted to dict
        drafter, eagle_ring = _ngram(), False
    res, ring = fr.run_coordinator(T, P, drafter, K=8, depth=4, max_new=160,
                                   prefill_chunk=4096, eagle_ring=eagle_ring, stage_dt=SPANS)
    _assert_lossless(res, T, P, 160)
    folded = res["rounds"] + res["wasted"]
    span_s = sum(r[1] for r in SPANS) / 1e3
    comp_s = sum(r[2] for r in SPANS) / 1e3
    assert res["stage_s"] == pytest.approx(folded * span_s, abs=5e-3)
    assert res["stage_compute_s"] == pytest.approx(folded * comp_s, abs=5e-3)
    assert res["transport_s"] == pytest.approx(res["traversal_s"] - res["stage_s"], abs=2e-3)
    assert res["traversal_s"] > 0
    assert res["per_stage_ms"] == {"0": [5.0, 4.0], "1": [3.25, 2.5], "2": [1.5, 1.0]}


def test_no_stage_timing_means_none_fields():
    """Without M25_STAGE_TIMING rows the split must be absent (None), never a fabricated zero — the
    coordinator still reports traversal_s (its own clock), which needs no stage cooperation."""
    T = fr.repetitive_T(560)
    res, _ = fr.run_coordinator(T, P, _ngram(), K=8, depth=4, max_new=120, eagle_ring=False)
    assert res["traversal_s"] > 0
    for k in ("stage_s", "stage_compute_s", "transport_s", "per_stage_ms"):
        assert res[k] is None


# ---- 5. DIVERGENCE / IN-FLIGHT BOOKKEEPING ---------------------------------------------------------

def test_divergence_trap_chain_bookkeeping():
    """An n-gram trap (repetition that breaks once, then a different repetition): the pipelined chain
    path (depth=4, no EAGLE) speculates across the break with chunks in flight -> divergence must
    discard the stale in-flight chunks WITHOUT losing or duplicating tokens."""
    T = fr.trap_T(560)
    res, ring = fr.run_coordinator(T, P, _ngram(), K=8, depth=4, max_new=200,
                                   prefill_chunk=24, eagle_ring=False)
    _assert_lossless(res, T, P, 200)
    assert res["n_tokens"] == len(res["output_ids"])
    assert res["wasted"] > 0, "the trap never exercised the in-flight discard path"
    # ledger: every decode verify the ring saw is a counted round, a discarded stale chunk, or a
    # drained-at-exit chunk; the drain is bounded by the in-flight window (depth-1)
    decode = [e for e in ring.log if e["op"] == "verify" and not e["prefill"]]
    drained = len(decode) - res["rounds"] - res["wasted"]
    assert 0 <= drained < 4, (len(decode), res["rounds"], res["wasted"])


def test_divergence_trap_tree(monkeypatch):
    """Same trap through the tree path (synchronous: nothing in flight to discard, but the trunk /
    vbase bookkeeping must survive the n-gram divergence exactly)."""
    _tree_env(monkeypatch, depth=8)
    T = fr.trap_T(560)
    hyb, _ = _hybrid()
    res, ring = fr.run_coordinator(T, P, hyb, K=8, depth=4, max_new=200,
                                   prefill_chunk=24, eagle_ring=True)
    _assert_lossless(res, T, P, 200)
    assert res["n_tokens"] == len(res["output_ids"])
