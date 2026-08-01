"""v4_dspark_fast: the cache-advance-only drafter, graded byte-for-byte against the reference loop.

v4_dspark_fast rebinds `DSparkTail.advance_and_draft` to skip the wasted intermediate forwards -- for
the n-1 committed positions whose draft blocks the ring discards it runs ONLY the KV-slot write, then
one full forward for the kept block. The claim is that this is BIT-EXACT to the reference's
per-position loop (v4_dspark_draft.advance_and_draft), because the only state a `DSparkAttention`
intermediate call persists is that one slot. This file is the CPU proof of that claim -- no GPU, no
158 GiB checkpoint -- plus the opt-in/fallback plumbing.

The bar is torch.equal, not allclose: a drafted token that gets ACCEPTED is committed output, so an
off-by-one-ulp in the intermediate advance would silently corrupt the stream (and only sometimes, at
an argmax near-tie), which is exactly the failure a spec-decode verify path must not have.

The head CUDA-graph (lever 2) is CUDA-only and cannot run here; its eligibility gate is tested to be
correctly OFF on CPU, and its capture/replay is proven on the GPU box, not in this suite.

Run: python3 -m pytest tests/test_v4_dspark_fast.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
REFCPU = pytest.importorskip("v4_ref_cpu")
V4 = pytest.importorskip("v4_stage")
D = pytest.importorskip("v4_dspark_draft")
FAST = pytest.importorskip("v4_dspark_fast")

SEED = 7
PROMPT = 13
BATCH = 1


@pytest.fixture
def args():
    return REFCPU.cpu_args()


@pytest.fixture(autouse=True)
def _clean_install():
    """Every test starts and ends on the reference path, whatever it did in between.

    install() rebinds a CLASS method and sets a module global; a test that left the fast path armed
    would silently change the reference-baseline of the next. Snapshot both, restore both."""
    ref_method = D.DSparkTail.advance_and_draft
    fast, graph = FAST.V4_DSPARK_FAST, FAST.V4_DSPARK_GRAPH
    ref_advance = FAST._REF_ADVANCE
    yield
    D.DSparkTail.advance_and_draft = ref_method
    FAST.V4_DSPARK_FAST, FAST.V4_DSPARK_GRAPH = fast, graph
    FAST._REF_ADVANCE = ref_advance


# ── the oracle -> stage + drafter transfer (test_v4_dspark's, self-contained) ─────────────────────

def _build(oracle, args):
    st = V4.Stage(0, args.n_layers, args, head=True, tail=True, dspark=True, device="cpu")
    for li in range(args.n_layers):
        st.layers[li].load_state_dict(oracle.layers[li].state_dict(), strict=True)
    st.embed_tokens.load_state_dict(oracle.embed.state_dict(), strict=True)
    st.norm.load_state_dict(oracle.norm.state_dict(), strict=True)
    st.lm_head.load_state_dict(oracle.head.state_dict(), strict=True)
    with torch.no_grad():
        for n in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            getattr(st, n).data.copy_(getattr(oracle, n).data)
    st._dspark = True
    dr = D.DSparkTail(st)
    for k, blk in enumerate(dr.mtp):
        sd = {n: v for n, v in oracle.mtp[k].state_dict().items() if n not in D.ALIAS_KEYS}
        report = blk.load_state_dict(sd, strict=False)
        assert set(report.missing_keys) == set(D.ALIAS_KEYS) and not report.unexpected_keys
    return st, dr


def _record(args, run_lengths, seed):
    """Prefill + FULL-accept rounds of the given run lengths, driven monotonically so the stage never
    rewinds. Returns (prefill_tok, prefill_main, [(committed, main, start_pos)]) — inputs the drafter
    consumes without touching the stage, so both a reference and a fast drafter can replay them."""
    oracle = REFCPU.build_oracle(args, SEED)
    st, _ = _build(oracle, args)
    ids = torch.randint(0, args.vocab_size, (BATCH, PROMPT),
                        generator=torch.Generator().manual_seed(seed))
    tok = st.logits_all(st.forward(st.embed(ids), ids, 0), full_logits=False).argmax(-1)
    prefill_main = st.tail_main_hidden().clone()
    g = torch.Generator().manual_seed(seed + 100)
    seq = [int(tok)] + torch.randint(0, args.vocab_size, (sum(run_lengths),), generator=g).tolist()
    rounds, pos, base = [], PROMPT, 0
    for rl in run_lengths:
        chunk = torch.tensor([seq[base:base + rl]], dtype=torch.long)
        st.forward(st.embed(chunk), chunk, pos)
        committed = torch.tensor([seq[base + 1:base + rl + 1]], dtype=torch.long)
        rounds.append((committed, st.tail_main_hidden().clone(), pos))
        pos += rl
        base += rl
    return tok, prefill_main, rounds


def _replay(dr, tok, prefill_main, rounds):
    dr.prefill(tok, prefill_main)
    out = []
    for committed, main, start_pos in rounds:
        blk, conf = dr.advance_and_draft(committed, main, start_pos=start_pos)
        out.append((blk.clone(), conf.clone(), dr.last_spec[1].clone()))
    return out, [b.attn.kv_cache.clone() for b in dr.mtp]


# ── 1. imports + the opt-in switch ────────────────────────────────────────────────────────────────

def test_imports_clean_and_expose_the_switches():
    """The module imports without a GPU and exposes both env flags and the install seam."""
    assert hasattr(FAST, "V4_DSPARK_FAST") and hasattr(FAST, "V4_DSPARK_GRAPH")
    assert callable(FAST.install) and callable(FAST.uninstall)
    assert callable(FAST.fast_advance_and_draft)


def test_default_off_is_a_noop(args):
    """V4_DSPARK_FAST off -> install() does nothing and advance_and_draft is the reference method."""
    FAST.V4_DSPARK_FAST = False
    before = D.DSparkTail.advance_and_draft
    assert FAST.install(D) is False
    assert D.DSparkTail.advance_and_draft is before, "the reference method must be untouched when off"


def test_install_is_idempotent_and_reversible(args):
    """On -> install rebinds once (a second call is a no-op), uninstall restores the reference."""
    FAST.V4_DSPARK_FAST = True
    ref = D.DSparkTail.advance_and_draft
    assert FAST.install(D) is True
    fast = D.DSparkTail.advance_and_draft
    assert fast is not ref and getattr(fast, "_v4_dspark_fast", False)
    assert FAST.install(D) is False, "a second install must be a no-op"
    assert FAST.uninstall(D) is True
    assert D.DSparkTail.advance_and_draft is ref, "uninstall must restore the reference method"


# ── 2. the headline: cache-advance-only is bit-exact ──────────────────────────────────────────────

@pytest.mark.parametrize("run_lengths", [
    (1, 1, 1),          # every round a bare g=1 advance (no intermediate positions)
    (1, 2, 3, 4),       # n = 1..block_size+1: the full range of committed-run lengths
    (4, 3, 2, 1),       # descending, so a long intermediate run precedes short ones
    (2, 4, 1, 3),       # mixed
])
@pytest.mark.parametrize("seed", [3, 5, 11])
def test_cache_advance_only_is_bit_exact(args, run_lengths, seed):
    """The fast drafter's drafts, logits, confidence AND every mtp KV buffer are torch.equal to the
    reference loop's, over committed runs of length 1..block_size+1 and several routings (seeds).

    This is the whole correctness claim of lever 1: skipping the intermediate forwards and writing
    only their KV slot changes nothing a downstream round can observe."""
    assert max(run_lengths) <= args.dspark_block_size + 1
    tok, prefill_main, rounds = _record(args, run_lengths, seed)

    FAST.V4_DSPARK_FAST = False
    ref_dr = _build(REFCPU.build_oracle(args, SEED), args)[1]
    ref_blocks, ref_caches = _replay(ref_dr, tok, prefill_main, rounds)

    FAST.V4_DSPARK_FAST = True
    assert FAST.install(D)
    fast_dr = _build(REFCPU.build_oracle(args, SEED), args)[1]
    assert getattr(fast_dr.advance_and_draft, "_v4_dspark_fast", False), "the fast path must be armed"
    fast_blocks, fast_caches = _replay(fast_dr, tok, prefill_main, rounds)

    for i, ((rb, rc, rl), (fb, fc, fl)) in enumerate(zip(ref_blocks, fast_blocks)):
        assert torch.equal(rb, fb), f"round {i}: draft ids diverged"
        assert torch.equal(rc, fc), f"round {i}: confidence diverged"
        assert torch.equal(rl, fl), f"round {i}: draft logits diverged"
    for k, (rk, fk) in enumerate(zip(ref_caches, fast_caches)):
        assert torch.equal(rk, fk), f"mtp {k}: kv_cache diverged"


# ── 3. state isolation: intermediate advance persists ONLY the committed KV slot ──────────────────

def test_intermediate_advance_ignores_input_ids(args):
    """The proof that an intermediate committed position leaves ONLY its KV slot behind.

    `_advance_cache_only` writes `main_kv`, which is `main_norm(main_proj(main_hidden))` projected
    through each layer's wkv (model.py:759,853) — a function of the TAP alone. The committed token id
    only ever reaches the draft block `x`, which the intermediate positions discard. So advancing the
    same taps with DIFFERENT intermediate token ids must land byte-identical KV, and only the KEPT
    position (whose block is real) may see the ids at all. Run two full-accept rounds that differ only
    in their non-final committed tokens; the mtp caches after must be equal."""
    FAST.V4_DSPARK_FAST = True
    assert FAST.install(D)
    tok, prefill_main, rounds = _record(args, (3,), seed=5)
    committed, main, start_pos = rounds[0]

    a = _build(REFCPU.build_oracle(args, SEED), args)[1]
    a.prefill(tok, prefill_main)
    a.advance_and_draft(committed, main, start_pos=start_pos)

    # same taps, same LAST (kept) token, different intermediate tokens
    other = committed.clone()
    other[0, :-1] = (other[0, :-1] + 1) % args.vocab_size
    assert not torch.equal(other, committed), "the intermediate tokens must actually differ"
    b = _build(REFCPU.build_oracle(args, SEED), args)[1]
    b.prefill(tok, prefill_main)
    b.advance_and_draft(other, main, start_pos=start_pos)

    for k, (ca, cb) in enumerate(zip(a.mtp, b.mtp)):
        assert torch.equal(ca.attn.kv_cache, cb.attn.kv_cache), \
            f"mtp {k}: an intermediate token id changed the KV — the advance is not state-isolated"


def test_fast_path_never_writes_a_speculative_slot(args):
    """The fast path keeps test_cache_never_speculative's invariant: the draft block's own positions
    stay zero in the window, so a rejected draft needs no rollback."""
    win = args.window_size
    assert PROMPT + 2 < win
    FAST.V4_DSPARK_FAST = True
    assert FAST.install(D)
    tok, prefill_main, rounds = _record(args, (1,), seed=3)
    committed, main, start_pos = rounds[0]
    dr = _build(REFCPU.build_oracle(args, SEED), args)[1]
    dr.prefill(tok, prefill_main)
    dr.advance_and_draft(committed, main, start_pos=start_pos)
    for k, blk in enumerate(dr.mtp):
        c = blk.attn.kv_cache
        assert not torch.equal(c[:, PROMPT], torch.zeros_like(c[:, PROMPT])), \
            f"mtp {k}: the advance wrote nothing to the committed slot"
        assert torch.equal(c[:, PROMPT + 1:], torch.zeros_like(c[:, PROMPT + 1:])), \
            f"mtp {k}: a draft-block position landed in the cache — the fast path needs a rollback"


# ── 4. the guards still fire, and the head graph stays off on CPU ─────────────────────────────────

def test_fast_path_keeps_the_reference_guards(args):
    """The fast advance is a drop-in: the position-discipline and range guards raise as before."""
    FAST.V4_DSPARK_FAST = True
    assert FAST.install(D)
    tok, prefill_main, rounds = _record(args, (1,), seed=3)
    committed, main, start_pos = rounds[0]
    dr = _build(REFCPU.build_oracle(args, SEED), args)[1]
    dr.prefill(tok, prefill_main)

    with pytest.raises(RuntimeError, match="gap"):
        dr.advance_and_draft(committed, main, start_pos=start_pos + 1)
    with pytest.raises(RuntimeError, match="overlap"):
        dr.advance_and_draft(committed, main, start_pos=start_pos - 1)
    assert dr.pos == PROMPT - 1, "a refused advance must not move the cursor"

    fresh = D.DSparkTail(dr.stage)
    with pytest.raises(RuntimeError, match="advance before prefill"):
        fresh.advance_and_draft(committed, main, start_pos=start_pos)

    n = args.dspark_block_size + 2
    with pytest.raises(RuntimeError, match="one round can commit at most"):
        dr.advance_and_draft(committed[:, :1].repeat(1, n), main[:, :1].repeat(1, n, 1),
                             start_pos=start_pos)


def test_head_graph_is_off_without_cuda(args):
    """On CPU the head-graph gate is False whatever the env asks — there is no graph off-device."""
    FAST.V4_DSPARK_FAST = True
    FAST.V4_DSPARK_GRAPH = True
    assert FAST.install(D)
    dr = _build(REFCPU.build_oracle(args, SEED), args)[1]
    h = torch.zeros(1, args.dspark_block_size, args.hc_mult, args.dim)
    assert h.is_cuda is False
    assert FAST._graph_eligible(dr, h) is False, "the head graph must not arm on CPU"


def test_graph_eligibility_predicate(args):
    """The gate is greedy + CUDA + b==1 + block_size (forward_head's input width) — each condition
    independently closes it."""
    FAST.V4_DSPARK_GRAPH = True
    dr = _build(REFCPU.build_oracle(args, SEED), args)[1]
    s = args.dspark_block_size                     # forward_head sees the block_size draft rows

    class _Cuda:                                   # a stand-in tensor that only claims to be on cuda
        is_cuda = True
        shape = (1, s, args.hc_mult, args.dim)
    assert FAST._graph_eligible(dr, _Cuda()) is True, "greedy b=1 block_size on cuda must be eligible"

    _Cuda.shape = (2, s, args.hc_mult, args.dim)   # b != 1
    assert FAST._graph_eligible(dr, _Cuda()) is False
    _Cuda.shape = (1, s + 1, args.hc_mult, args.dim)   # wrong width
    assert FAST._graph_eligible(dr, _Cuda()) is False
    _Cuda.shape = (1, s, args.hc_mult, args.dim)
    dr.temperature = 1.0                           # sampling drafter
    assert FAST._graph_eligible(dr, _Cuda()) is False
    FAST.V4_DSPARK_GRAPH = False                   # env off
    dr.temperature = 0.0
    assert FAST._graph_eligible(dr, _Cuda()) is False
