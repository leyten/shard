"""Offline integration tests for the DE-LOCKSTEP rows coordinator with PER-STREAM TREES
(coordinate_pipe_rows + M25_TREE) over the CPU fake rings.

THE EQUIVALENCE GATE: a rows stream is structurally solo depth-1 — same drafter fork, same routing
rule (n-gram hit -> chain frame, miss -> tree frame), same dirty-frontier refeed, same accept/bonus
arithmetic. So rows-with-trees at B=4 must reproduce, PER STREAM and EXACTLY, what
coordinate_pipe_tree(depth=1) commits against the same oracle with an identically-seeded drafter:
identical output_ids AND an identical decode-frame wire sequence (start/tree-flag/token_ids/pos,
frame for frame). Any bookkeeping drift between the paths — off-by-one in the refeed offset, a
wrong tree base_pos, a missed bonus — breaks the frame-sequence equality immediately.

The FakeRingB harness also enforces the per-stream KV dirty-frontier invariant on every frame (a
tree round's nodes stay dirty until re-fed) and the post-hoc stale-read healing rule — the
silent-corruption-with-valid-receipts class the adversarial reviews keep catching.

Run: python3 -m pytest tests/test_fake_ring_rows.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
fr = pytest.importorskip("fake_ring")

from ngram_draft import NgramDrafter                 # noqa: E402
from eagle_draft import EagleDrafter, HybridDrafter  # noqa: E402
from test_eagle_draft import _make_head              # noqa: E402

MP = fr.MP
S = fr.S

K = 8


def _ngram():
    return NgramDrafter(ng=3, min_match=1, margin=64)


def _base(seed=0):
    d, embed = _make_head(seed)
    return EagleDrafter(d, embed, device="cpu", next_hidden="prenorm")


def _tree_env(monkeypatch, m=12, topb=3, depth=4):
    monkeypatch.setattr(S, "M25_EAGLE", True)
    monkeypatch.setattr(S, "M25_TREE", True)
    monkeypatch.setenv("M25_TREE_M", str(m))
    monkeypatch.setenv("M25_TREE_TOPB", str(topb))
    monkeypatch.setenv("M25_TREE_DEPTH", str(depth))


def _mixed_Ts():
    """One stream per content regime: novel (tree-heavy), repetitive (chain-heavy), trap
    (mid-stream divergence + chain<->tree flips), novel with another seed."""
    return [fr.novel_T(460), fr.repetitive_T(460), fr.trap_T(460), fr.novel_T(460, seed=77)], [60, 60, 60, 48]


def _decode_frames(log, stream=None):
    return [(e["start"], e["tree"], tuple(e["token_ids"]), tuple(e["pos"]))
            for e in log if e["op"] == "verify" and not e["prefill"] and e["start"] > 0
            and (stream is None or e.get("stream") == stream)]


def test_rows_tree_equals_solo_tree_per_stream(monkeypatch):
    """THE gate: rows+trees B=4 == 4x coordinate_pipe_tree(depth=1), exact — output_ids and the
    decode wire sequence per stream. Shared EAGLE base; forks are byte-identical clean states."""
    _tree_env(monkeypatch)
    Ts, PLs = _mixed_Ts()
    MAX_NEW = 120
    base = _base()
    rows_drafters = [HybridDrafter(_ngram(), base.fork()) for _ in Ts]
    res, ring = fr.run_rows_coordinator(Ts, PLs, rows_drafters, K=K, max_new=MAX_NEW, prefill_chunk=24)
    assert res["tree"], "tree_on must be armed under M25_TREE"
    assert any(e["tree"] for e in ring.log if e["op"] == "verify"), "no tree frame ever sent"
    for b, (T, PL) in enumerate(zip(Ts, PLs)):
        solo_drafter = HybridDrafter(_ngram(), base.fork())
        solo, solo_ring = fr.run_coordinator(T, PL, solo_drafter, K=K, depth=1, max_new=MAX_NEW,
                                             prefill_chunk=24, eagle_ring=True)
        rout = res["streams"][b]["output_ids"]
        assert rout == solo["output_ids"], (
            f"stream {b}: rows+tree output != solo-tree depth-1 output\n"
            f"  rows={rout[:24]}...\n  solo={solo['output_ids'][:24]}...")
        assert rout == T[PL:PL + len(rout)], f"stream {b}: LOSSLESSNESS BROKEN vs oracle"
        assert len(rout) >= MAX_NEW, f"stream {b} stopped early ({len(rout)} < {MAX_NEW})"
        rows_frames = _decode_frames(ring.log, stream=b)
        solo_frames = _decode_frames(solo_ring.log)
        assert rows_frames == solo_frames, (
            f"stream {b}: decode wire sequence diverged at frame "
            f"{next(i for i, (a, c) in enumerate(zip(rows_frames, solo_frames)) if a != c)}"
            f" of {len(rows_frames)}/{len(solo_frames)}")


def test_rows_prefill_depth_invariant(monkeypatch):
    """PIPELINED prefill (prefill_depth>1) is byte-identical to the serial send->recv path
    (prefill_depth=1). The single-threaded stage applies verify(prefill) ops in SEND order no matter
    how many are in flight, so each stream's chunk i still reads its own 0..i-1 KV — firing a window
    ahead only hides WAN latency. Deeply multi-chunk (prefill_chunk=24 over 460-tok prompts) x B=4,
    so a bug in the (stream,chunk) FIFO / drafter-extend ordering would diverge immediately."""
    _tree_env(monkeypatch)
    Ts, PLs = _mixed_Ts()
    base = _base()                                          # base.fork() = independent clean states, so the
    serial, _ = fr.run_rows_coordinator(                    # two runs' drafters start byte-identical and the
        Ts, PLs, [HybridDrafter(_ngram(), base.fork()) for _ in Ts],   # ONLY difference is prefill_depth
        K=K, max_new=80, prefill_chunk=24, prefill_depth=1)
    piped, _ = fr.run_rows_coordinator(
        Ts, PLs, [HybridDrafter(_ngram(), base.fork()) for _ in Ts],
        K=K, max_new=80, prefill_chunk=24, prefill_depth=8)
    assert [s["output_ids"] for s in serial["streams"]] == [s["output_ids"] for s in piped["streams"]]


def test_rows_tree_interleaves_streams(monkeypatch):
    """The de-lockstep property survives the tree route: frames from different streams interleave on
    the wire (the streams ARE the pipeline) — never B sequential solo runs."""
    _tree_env(monkeypatch)
    Ts, PLs = _mixed_Ts()
    base = _base()
    drafters = [HybridDrafter(_ngram(), base.fork()) for _ in Ts]
    res, ring = fr.run_rows_coordinator(Ts, PLs, drafters, K=K, max_new=80, prefill_chunk=24)
    decode = [e for e in ring.log if e["op"] == "verify" and not e["prefill"]]
    first_by_stream = {}
    for i, e in enumerate(decode):
        first_by_stream.setdefault(e["stream"], i)
    assert len(first_by_stream) == len(Ts)
    # every stream fires its first decode frame before any stream's 6th — lockstep-free scheduling
    assert max(first_by_stream.values()) < 6 * len(Ts), first_by_stream


def test_rows_tree_dirty_frontier(monkeypatch):
    """Per-stream KV-frontier structure on the wire (belt-and-braces over FakeRingB's in-ring
    asserts): a stream's frames advance monotonically with NO gap — every frame starts at or before
    the previous frame's write frontier — and every tree frame's trunk re-feeds committed text
    (oracle-checked at its causal positions)."""
    _tree_env(monkeypatch)
    Ts, PLs = _mixed_Ts()
    base = _base()
    drafters = [HybridDrafter(_ngram(), base.fork()) for _ in Ts]
    res, ring = fr.run_rows_coordinator(Ts, PLs, drafters, K=K, max_new=100, prefill_chunk=24)
    for b in range(len(Ts)):
        frames = [e for e in ring.log
                  if e["op"] == "verify" and not e["prefill"] and e.get("stream") == b and e["start"] > 0]
        assert frames, f"stream {b} sent no decode frames"
        for prev, nxt in zip(frames, frames[1:]):
            assert prev["start"] <= nxt["start"] <= prev["start"] + prev["n"], (
                f"stream {b}: frame start {nxt['start']} outside the written region "
                f"[{prev['start']}, {prev['start'] + prev['n']}] — refeed gap/regression")
        for e in frames:
            # ANCHOR INVARIANT, tree and chain alike: a frame's first token is the committed token
            # at its start slot (pending_path[0]) — a wrong anchor is corrupted row KV with valid
            # receipts (the #84 review's MAJOR-1 class). The tree nodes/draft tail past it are
            # speculative by design and pinned by losslessness instead.
            assert e["token_ids"][0] == Ts[b][e["start"]], (
                f"stream {b}: frame at start={e['start']} fed anchor {e['token_ids'][0]}, "
                f"committed token there is {Ts[b][e['start']]}")


def test_rows_chain_only_unchanged_by_tree_flag_off(monkeypatch):
    """M25_TREE off: the rows path must behave exactly as before this build (pure chain frames,
    no tree flags on the wire) — the flag-off regression pin."""
    monkeypatch.setattr(S, "M25_EAGLE", True)
    monkeypatch.setattr(S, "M25_TREE", False)
    Ts, PLs = _mixed_Ts()
    base = _base()
    drafters = [HybridDrafter(_ngram(), base.fork()) for _ in Ts]
    res, ring = fr.run_rows_coordinator(Ts, PLs, drafters, K=K, max_new=80, prefill_chunk=24)
    assert not res["tree"]
    assert not any(e.get("tree") for e in ring.log if e["op"] == "verify"), "tree frame with M25_TREE off"
    for b, (T, PL) in enumerate(zip(Ts, PLs)):
        rout = res["streams"][b]["output_ids"]
        assert rout == T[PL:PL + len(rout)], f"stream {b}: losslessness broken (chain-only rows)"


def test_rows_tree_trap_divergence(monkeypatch):
    """All four streams on trap_T (repetition that breaks once): chain bursts speculate across the
    break, diverge, flip to tree rounds through the novel run, then re-lock — per-stream refeed and
    extend bookkeeping must survive; outputs stay oracle-exact."""
    _tree_env(monkeypatch)
    Ts = [fr.trap_T(560, seed=s) for s in (7, 11, 23, 5)]
    PLs = [60, 60, 48, 60]
    base = _base()
    drafters = [HybridDrafter(_ngram(), base.fork()) for _ in Ts]
    res, ring = fr.run_rows_coordinator(Ts, PLs, drafters, K=K, max_new=160, prefill_chunk=24)
    for b, (T, PL) in enumerate(zip(Ts, PLs)):
        rout = res["streams"][b]["output_ids"]
        assert rout == T[PL:PL + len(rout)], f"stream {b}: losslessness broken through the trap"
        assert len(rout) >= 160, f"stream {b} stopped early"
    kinds = {(e["stream"], e["tree"]) for e in ring.log if e["op"] == "verify" and not e["prefill"]}
    for b in range(len(Ts)):
        assert (b, True) in kinds and (b, False) in kinds, f"stream {b} never flipped chain<->tree"


def test_version_mix_guards_abort_loud(monkeypatch):
    """The two version-mix guards' FAILURE paths (review adoption — both fakes echoing correctly
    would let a guard regress to a no-op unnoticed): an old stage that chain-mathed a tree frame
    replies without the tree echo -> LOUD abort; a pre-#84 stage replying untagged -> LOUD abort.
    In both mixes row KV is already corrupted with valid receipts — dying before commit is the
    whole defense."""
    _tree_env(monkeypatch)
    Ts, PLs = [fr.novel_T(300)], [60]
    base = _base()
    with pytest.raises(Exception, match="tree echo"):
        fr.run_rows_coordinator(Ts, PLs, [HybridDrafter(_ngram(), base.fork())], K=K, max_new=40,
                                strip_tree_echo=True)
    with pytest.raises(Exception, match="UNTAGGED"):
        fr.run_rows_coordinator(Ts, PLs, [HybridDrafter(_ngram(), base.fork())], K=K, max_new=40,
                                strip_stream_tag=True)


def test_rows_tree_extend_pairing(monkeypatch):
    """The EAGLE extend contract on the rows tree path (position-encoded aux): every
    extend(tokens, auxes, base_pos) pairs auxes[i] with absolute position base_pos+i and
    tokens[i] == T[base_pos+i+1]; contiguous from 0 — chain commits, tree commits and prefill
    chunks all tile. The rows analogue of test_extend_pairing_invariant."""
    _tree_env(monkeypatch)
    Ts, PLs = _mixed_Ts()
    base = _base()
    recs = [fr.RecordingDrafter(base.fork()) for _ in Ts]
    drafters = [HybridDrafter(_ngram(), r) for r in recs]
    res, ring = fr.run_rows_coordinator(Ts, PLs, drafters, K=K, max_new=100, prefill_chunk=24)
    for b, rec in enumerate(recs):
        assert rec.extends, f"stream {b}: no extend() calls recorded"
        T = Ts[b]
        cur = 0
        for toks, aux, bp in rec.extends:
            assert bp == cur, f"stream {b}: extend base_pos {bp} != expected {cur} (gap/overlap)"
            for i, t in enumerate(toks):
                want = bp + i
                assert bool(torch.all(aux[i].flatten() == float(want))), (
                    f"stream {b}: aux[{i}] encodes {aux[i].flatten()[0].item():.0f}, want {want}")
                assert t == T[want + 1], (
                    f"stream {b}: tokens[{i}]={t} != T[{want + 1}]={T[want + 1]} — left-shift broken")
            cur += len(toks)
