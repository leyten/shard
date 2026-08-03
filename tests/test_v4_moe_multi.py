"""The DSpark drafter's MoEs get a fast path, or this file says so out loud.

THE BUG CLASS THIS EXISTS FOR is "a lever that silently does nothing". The grouped fp4 MoE kernel
measured 1.72x on the ring's main path and, in the same process, on the same box, the drafter's
`n_mtp_layers` MoEs ran the vendored reference's dispatch loop — because every fast MoE path in this
engine opens with `if xv.size(0) != 1: return _REF_FORWARD(...)` and the drafter's blocks are
`dspark_block_size` rows wide by construction (`DSparkBlock.forward_embed`, model.py:854). Nothing
errored, nothing warned, and no test could tell.

So the first half of this file is the DIAGNOSIS, written as executable assertions rather than prose,
including the two parts of it that turned out to be WRONG:

  test_bank_layout_never_sees_the_drafters_moes    TRUE: `Stage.__init__` passes `self.layers` and
      only `self.layers`, and `DSparkTail` builds its blocks elsewhere and later, so the load-time
      bank layout reaches zero drafter MoEs.
  test_the_drafter_never_reaches_the_expert_bank   FALSE was the claim that `_expert_bank`'s 2 GiB
      headroom check DECLINES for the drafter. It is never called at all: `grouped_forward` returns
      on its `xv.size(0) != 1` line, above the bank. Which also means giving the drafter the bank
      layout would buy nothing (the grouped kernel would still decline the shape) and that the tail
      is in no danger of a lazy ~10.28 GiB expert stack appearing after its graph pools are pinned.
  test_the_single_token_levers_decline_the_drafters_shape   the gate that makes both of the above
      true. If someone ever teaches the grouped kernel s > 1, THIS test fails first and its message
      says what else has to change.

The second half is the GATE. `phase0/v4_moe_multi` claims the small-block shape and removes the
per-expert host drains bit-exactly; `test_the_drafter_actually_takes_the_fast_path` fails if a future
change routes the drafter off it again, and `test_no_per_expert_device_drain_in_a_drafted_round`
counts the syncs so "it took the path" cannot pass while the path stopped being fast.

Everything runs on CPU against v4_ref_cpu's toy V4 (2 MTP stages, block_size 3, 8 routed experts) —
the lever is pure torch, so unlike the grouped kernel its parity proof does NOT need a GPU.

Run: python3 -m pytest tests/test_v4_moe_multi.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
REFCPU = pytest.importorskip("v4_ref_cpu")
V4 = pytest.importorskip("v4_stage")
DS = pytest.importorskip("v4_dspark_draft")
DECODE = pytest.importorskip("v4_moe_decode")
GROUPED = pytest.importorskip("v4_moe_grouped")
MULTI = pytest.importorskip("v4_moe_multi")

from test_v4_dspark import step, tail_from_oracle  # noqa: E402  (the shared oracle->stage harness)

SEED = 7
PROMPT = 13
ROUNDS = 4


@pytest.fixture(scope="module")
def args():
    return REFCPU.cpu_args()


@pytest.fixture
def oracle(args):
    return REFCPU.build_oracle(args, SEED)


@pytest.fixture
def model():
    """The exec'd reference module, with `MoE.forward` restored after the test whatever it did."""
    mod = V4.ref()
    before = mod.MoE.forward
    try:
        yield mod
    finally:
        mod.MoE.forward = before
        MULTI._REF_FORWARD = None


def armed(monkeypatch, mod, max_rows=None):
    """Install the lever over whatever forward is currently bound. Returns the previous forward.

    Uninstalls first, because `load_ref` installs this module itself when `V4_MOE_MULTI` is exported
    — and `multi_forward` dispatches on the row count, not on the flag, so a suite run under that env
    would otherwise compare the fast path against itself and pass while proving nothing."""
    MULTI.uninstall(mod)
    monkeypatch.setattr(MULTI, "V4_MOE_MULTI", True)
    if max_rows is not None:
        monkeypatch.setattr(MULTI, "V4_MOE_MULTI_MAX", max_rows)
    before = mod.MoE.forward
    assert MULTI.install(mod), "install must take when V4_MOE_MULTI is on"
    return before


def drafted_rounds(oracle, args, n=ROUNDS, seed=0):
    """Prefill + `n` drafted rounds. -> [(drafts, confidence, draft logits)] + the mtp caches.

    Single-token main-model steps, so the STAGE's own MoEs stay on the s == 1 decode path and every
    difference this file can see belongs to the drafter."""
    st, dr = tail_from_oracle(oracle, args)
    ids = torch.randint(0, args.vocab_size, (1, PROMPT),
                        generator=torch.Generator().manual_seed(seed))
    tok, main = step(st, ids, 0)
    dr.prefill(tok, main)
    out = []
    for i in range(PROMPT, PROMPT + n):
        tok, main = step(st, tok, i)
        blk, conf = dr.advance_and_draft(tok.unsqueeze(1), main, start_pos=i)
        out.append((blk.clone(), conf.clone(), dr.last_spec[1].clone()))
    return out, [b.attn.kv_cache.clone() for b in dr.mtp], st, dr


def moe_spy(mod, monkeypatch, stage_moes, draft_moes):
    """Log (owner, rows, forward-name) for every MoE call. Wraps whatever forward is bound."""
    inner = mod.MoE.forward
    log = []

    def spy(self, x, input_ids):
        owner = "drafter" if any(self is m for m in draft_moes) else (
            "stage" if any(self is m for m in stage_moes) else "?")
        log.append((owner, x.view(-1, self.dim).size(0), inner.__name__))
        return inner(self, x, input_ids)

    monkeypatch.setattr(mod.MoE, "forward", spy)
    return log


def moes_under(module):
    return [m for m in module.modules()
            if hasattr(m, "experts") and hasattr(m, "experts_start_idx")]


# ── the diagnosis, as assertions ─────────────────────────────────────────────────────────────────

def test_bank_layout_never_sees_the_drafters_moes(oracle, args):
    """`Stage.__init__` banks `self.layers`; the drafter's blocks are built elsewhere, later.

    Not a source-grep: the two module trees are asked directly. The stage's MoEs are the ones
    `bank_layout` is handed, the drafter's are reachable from neither `Stage.layers` nor anything
    else the loader passes, and no drafter MoE ends up carrying a `_grouped_bank`."""
    st, dr = tail_from_oracle(oracle, args)
    stage_moes, draft_moes = moes_under(st.layers), moes_under(dr.mtp)
    assert len(stage_moes) == args.n_layers
    assert len(draft_moes) == args.n_mtp_layers
    assert not ({id(m) for m in draft_moes} & {id(m) for m in stage_moes}), \
        "the drafter's MoEs are its own modules — banking the stage's cannot reach them"
    assert all(getattr(m, "_grouped_bank", None) is None for m in draft_moes), \
        "a drafter MoE carries no bank: bank_layout is only ever called on Stage.layers"


def test_the_drafters_moe_never_runs_at_one_token(oracle, args, model, monkeypatch):
    """Every drafter MoE call is `dspark_block_size` rows wide — the shape no fast path claims.

    This is the whole hole in one assertion. `forward_embed` builds a block_size-wide draft block and
    the MoE sees `b * block_size` rows on every one of the `n_mtp_layers` blocks, every round, while
    the main stage's single-token steps sit at 1 row beside it."""
    st, dr = tail_from_oracle(oracle, args)
    log = moe_spy(model, monkeypatch, moes_under(st.layers), moes_under(dr.mtp))
    ids = torch.randint(0, args.vocab_size, (1, PROMPT), generator=torch.Generator().manual_seed(0))
    tok, main = step(st, ids, 0)
    dr.prefill(tok, main)
    assert not [r for r in log if r[0] == "drafter"], \
        "the drafter's prefill runs attention only (DSparkBlock.forward at start_pos == 0)"
    n0 = len(log)                                        # everything before this is the PROMPT-wide
    for i in range(PROMPT, PROMPT + 2):                  # ring prefill, whose rows are the prompt's
        tok, main = step(st, tok, i)
        dr.advance_and_draft(tok.unsqueeze(1), main, start_pos=i)
    draft = [r for r in log[n0:] if r[0] == "drafter"]
    stage = [r for r in log[n0:] if r[0] == "stage"]
    assert draft, "no drafter MoE calls were observed — the harness stopped exercising the drafter"
    assert {r[1] for r in draft} == {args.dspark_block_size}, \
        f"drafter MoE row counts {sorted({r[1] for r in draft})} != block_size {args.dspark_block_size}"
    assert {r[1] for r in stage} == {1}, "the main stage's decode steps are the s == 1 shape"


def test_the_single_token_levers_decline_the_drafters_shape(monkeypatch, args, oracle):
    """Both fast MoE paths return on their first line at the drafter's row count.

    THE COUPLING GATE. If the grouped kernel is ever taught s > 1, this test fails — and when it
    does, the drafter's MoEs must be given the bank layout at the same time, because `bank_layout` is
    called on `Stage.layers` alone (test_bank_layout_never_sees_the_drafters_moes) and a grouped
    kernel that claims s > 1 would then hit `_expert_bank`'s lazy stack on the tail: a second
    ~10.28 GiB copy of the drafter's experts, allocated on the first dspark job, after the graph
    pools are pinned, on the box with the least headroom in the ring."""
    moe = oracle.layers[args.n_hash_layers].ffn          # a score-routed MoE
    x = torch.randn(1, args.dspark_block_size, args.dim).to(moe.gate.weight.dtype)
    ids = torch.randint(0, args.vocab_size, (1, args.dspark_block_size))
    for mod_, name in ((DECODE, "v4_moe_decode"), (GROUPED, "v4_moe_grouped")):
        calls = []
        monkeypatch.setattr(mod_, "_REF_FORWARD",
                            lambda s, xx, ii: (calls.append(1), torch.zeros_like(xx))[1])
        monkeypatch.setattr(mod_, "_WORLD_SIZE", 1)
        fwd = mod_.decode_forward if mod_ is DECODE else mod_.grouped_forward
        fwd(moe, x, ids)
        assert calls == [1], (
            f"{name} claimed a {args.dspark_block_size}-row block. If that is deliberate, the "
            f"drafter's MoEs now need bank_layout() — v4_stage.Stage.__init__ banks self.layers "
            f"only, so they would fall to _expert_bank's lazy stack on the tail.")


def test_the_drafter_never_reaches_the_expert_bank(oracle, args, monkeypatch):
    """`_expert_bank` and its 2 GiB headroom check are called ZERO times during drafted rounds.

    The adversarial reading of this hole was "the headroom check declines for the drafter". It does
    not decline; it never runs, because the row-count gate is above it. Recorded as a test so the
    refutation is checkable and so a future change that DOES route the drafter into the lazy stack —
    the one path that could put a second copy of 3 x 256 experts on the tail — is caught here."""
    seen = {"bank": 0, "fits": 0}
    real_bank, real_fits = GROUPED._expert_bank, GROUPED._bank_fits
    monkeypatch.setattr(GROUPED, "_expert_bank",
                        lambda m: (seen.__setitem__("bank", seen["bank"] + 1), real_bank(m))[1])
    monkeypatch.setattr(GROUPED, "_bank_fits",
                        lambda e: (seen.__setitem__("fits", seen["fits"] + 1), real_fits(e))[1])
    drafted_rounds(oracle, args, n=2)
    assert seen == {"bank": 0, "fits": 0}, \
        f"the drafter reached the lazy expert bank {seen} times — it must not stack a second copy"


# ── the lever's envelope ─────────────────────────────────────────────────────────────────────────

def test_default_off_installs_nothing(model):
    """Env unset -> inert. `install()` no-ops and the forward the stack already had stays bound, so
    a ring launched without the flag is byte-identical to today."""
    assert MULTI.V4_MOE_MULTI is False
    before = model.MoE.forward
    assert MULTI.install(model) is False
    assert model.MoE.forward is before


def test_install_is_idempotent_and_sits_on_top(model, monkeypatch):
    """It captures whatever is bound (grouped -> decode -> reference) and never installs twice."""
    before = armed(monkeypatch, model)
    assert model.MoE.forward is MULTI.multi_forward
    assert MULTI._REF_FORWARD is before
    assert MULTI.install(model) is False
    assert MULTI.uninstall(model) is True
    assert model.MoE.forward is before


@pytest.mark.parametrize("rows", [1, 33])
def test_declines_one_token_and_anything_prefill_sized(model, monkeypatch, args, oracle, rows):
    """s == 1 goes DOWN to the single-token levers; a prefill-sized block goes to the reference.

    The first keeps the main decode path untouched (this file must not shadow the grouped kernel);
    the second keeps the host-side bucketing — `T * k` python iterations — out of a shape where it
    would cost more than the drains it removes."""
    armed(monkeypatch, model, max_rows=32)
    calls = []
    monkeypatch.setattr(MULTI, "_REF_FORWARD",
                        lambda s, xx, ii: (calls.append(1), torch.zeros_like(xx))[1])
    moe = oracle.layers[args.n_hash_layers].ffn
    x = torch.randn(1, rows, args.dim).to(moe.gate.weight.dtype)
    MULTI.multi_forward(moe, x, torch.randint(0, args.vocab_size, (1, rows)))
    assert calls == [1], f"{rows} rows must fall through, not be claimed"


def test_declines_tensor_parallel(model, monkeypatch, args, oracle):
    """world_size > 1: the reference all-reduces the routed sum before the shared expert, and a rank
    that skips that reduction silently drops the other ranks' experts."""
    armed(monkeypatch, model)
    calls = []
    monkeypatch.setattr(MULTI, "_REF_FORWARD",
                        lambda s, xx, ii: (calls.append(1), torch.zeros_like(xx))[1])
    monkeypatch.setattr(MULTI, "_WORLD_SIZE", 2)
    moe = oracle.layers[args.n_hash_layers].ffn
    x = torch.randn(1, args.dspark_block_size, args.dim).to(moe.gate.weight.dtype)
    MULTI.multi_forward(moe, x, torch.randint(0, args.vocab_size, (1, args.dspark_block_size)))
    assert calls == [1]


# ── bit-exactness ────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rows", [2, 3, 5, 8, 32])
def test_bit_exact_against_the_reference_dispatch(model, monkeypatch, args, oracle, rows):
    """One MoE, one input, both paths, `torch.equal`.

    Swept over row counts because the bucketing is where an ordering bug would live: at 2 rows every
    expert is likely unique, at 32 rows over 8 toy experts every expert holds several tokens and the
    per-expert `idx`/`top` vectors are long enough for a row-major slip to change the fp32 sum."""
    ref = model.MoE.forward
    moe = oracle.layers[args.n_hash_layers].ffn
    g = torch.Generator().manual_seed(rows)
    x = torch.randn(1, rows, args.dim, generator=g).to(moe.gate.weight.dtype)
    ids = torch.randint(0, args.vocab_size, (1, rows), generator=g)
    with torch.no_grad():
        want = ref(moe, x, ids)
        armed(monkeypatch, model, max_rows=32)
        got = MULTI.multi_forward(moe, x, ids)
    assert torch.equal(want, got), f"{rows} rows: {(want - got).abs().max().item():.3e}"


def test_bit_exact_on_a_hash_routed_layer(model, monkeypatch, args, oracle):
    """Hash routing can name the SAME expert twice in one token's row, and `y[idx] += ...` with a
    duplicate index is a last-write, not a sum. Nothing here special-cases that: the op and its
    operands are the reference's, so whatever the semantics are, they are reproduced."""
    ref = model.MoE.forward
    moe = oracle.layers[0].ffn
    assert moe.gate.hash, "layer 0 of the CPU config must be hash-routed for this test to mean anything"
    g = torch.Generator().manual_seed(11)
    x = torch.randn(1, 6, args.dim, generator=g).to(moe.gate.weight.dtype)
    ids = torch.randint(0, args.vocab_size, (1, 6), generator=g)
    with torch.no_grad():
        want = ref(moe, x, ids)
        armed(monkeypatch, model)
        got = MULTI.multi_forward(moe, x, ids)
    assert torch.equal(want, got)


def test_bit_exact_drafts_logits_and_confidence(model, monkeypatch, args, oracle):
    """The claim that matters on the ring: the drafter's proposed tokens, its logits, its confidence
    and its mtp KV cache are unchanged by the lever, over several drafted rounds.

    A drafter that is merely CLOSE is not lossless — a near-tie argmax flip in the draft block
    changes what the ring verifies, and the acceptance rate moves without any error being raised."""
    ref_out, ref_kv, _, _ = drafted_rounds(oracle, args)
    armed(monkeypatch, model)
    got_out, got_kv, _, _ = drafted_rounds(REFCPU.build_oracle(args, SEED), args)
    assert len(ref_out) == ROUNDS
    for i, ((rb, rc, rl), (gb, gc, gl)) in enumerate(zip(ref_out, got_out)):
        assert torch.equal(rb, gb), f"draft ids diverged round {i}"
        assert torch.equal(rc, gc), f"confidence diverged round {i}"
        assert torch.equal(rl, gl), f"draft logits diverged round {i}"
    for i, (rk, gk) in enumerate(zip(ref_kv, got_kv)):
        assert torch.equal(rk, gk), f"mtp {i} kv_cache diverged"


# ── THE GATE ─────────────────────────────────────────────────────────────────────────────────────

def test_the_lever_adds_no_persistent_state_to_a_drafter_moe(model, monkeypatch, args, oracle):
    """THE TAIL'S MEMORY BAR. Steady-state VRAM delta must be zero, and the reason must be structural.

    The tail is the box with the least headroom in the ring — layers 40-42 plus the drafter's three
    256-expert blocks plus the fp32 head plus the embedding, ~24.7 of 32.6 GiB — so anything that
    allocates per-MoE state there is a different decision from the same change on a middle stage.
    This lever allocates none: it builds no bank, keeps no cache, and hangs nothing off the module.
    Every expert parameter is the same storage before and after, so there is no second copy and no
    relayout, and its per-call temporaries are strictly FEWER than the reference's (which materialises
    a `counts` histogram, a boolean mask per expert and two `nonzero` outputs per expert, against one
    index tensor here) — so peak is bounded by the reference's peak too.

    Asserted on the module rather than on a byte counter because a CPU box has no VRAM meter, and
    because the property that matters is the structural one: if a future version starts caching a
    bank on the drafter's MoEs, this fails before it reaches a card."""
    armed(monkeypatch, model)
    st, dr = tail_from_oracle(oracle, args)
    draft_moes = moes_under(dr.mtp)
    before_attrs = [set(vars(m)) for m in draft_moes]
    before_store = [[p.data_ptr() for p in m.parameters()] for m in draft_moes]
    ids = torch.randint(0, args.vocab_size, (1, PROMPT), generator=torch.Generator().manual_seed(0))
    tok, main = step(st, ids, 0)
    dr.prefill(tok, main)
    for i in range(PROMPT, PROMPT + 3):
        tok, main = step(st, tok, i)
        dr.advance_and_draft(tok.unsqueeze(1), main, start_pos=i)
    for m, attrs, ptrs in zip(draft_moes, before_attrs, before_store):
        assert set(vars(m)) == attrs, \
            f"the lever hung {sorted(set(vars(m)) - attrs)} off a drafter MoE — that is tail VRAM"
        assert getattr(m, "_grouped_bank", None) is None, "no bank is built, lazily or otherwise"
        assert [p.data_ptr() for p in m.parameters()] == ptrs, \
            "an expert parameter moved — the lever must not relay out or copy the weights"


def test_the_drafter_actually_takes_the_fast_path(model, monkeypatch, args, oracle):
    """THE GATE. With the lever armed, EVERY drafter MoE call goes through `multi_forward`.

    This is the test the original bug would have failed. It does not check that the lever exists or
    that it is bit-exact — both were true of the grouped kernel while the drafter ran the reference
    loop — it checks that the drafter's own MoEs are the ones taking it. Change the drafter's block
    shape, add a gate to `multi_forward`, install in the wrong order, or route the drafter around
    `MoE.forward`, and this fails."""
    armed(monkeypatch, model)
    st, dr = tail_from_oracle(oracle, args)
    log = moe_spy(model, monkeypatch, moes_under(st.layers), moes_under(dr.mtp))
    ids = torch.randint(0, args.vocab_size, (1, PROMPT), generator=torch.Generator().manual_seed(0))
    tok, main = step(st, ids, 0)
    dr.prefill(tok, main)
    for i in range(PROMPT, PROMPT + 2):
        tok, main = step(st, tok, i)
        dr.advance_and_draft(tok.unsqueeze(1), main, start_pos=i)
    draft = [r for r in log if r[0] == "drafter"]
    assert len(draft) == 2 * args.n_mtp_layers, \
        f"expected one MoE call per mtp block per round, got {len(draft)}"
    assert {r[2] for r in draft} == {"multi_forward"}, \
        (f"the drafter's MoEs are on {sorted({r[2] for r in draft})}, not the fast path — a lever "
         f"that silently does nothing is the bug this file exists for")


def test_no_per_expert_device_drain_in_a_drafted_round(model, monkeypatch, args, oracle):
    """Counts the syncs, because "it took the path" can be true while the path stopped being fast.

    The reference dispatch costs one `bincount(...).tolist()` plus one `torch.where` PER ACTIVE
    EXPERT, and on a GPU each of those is a device drain that also stalls every launch behind it —
    the drain is the cost, not the arithmetic. The lever's whole claim is that the same schedule is
    read off `indices` ONCE. So: the reference path shows per-expert `torch.where` inside a drafted
    round, and the lever shows exactly zero.

    Counted only WHILE INSIDE an MoE forward, so the drafter's attention (which has its own uses of
    both ops) cannot mask a regression in the dispatch."""
    n = {"where": 0, "bincount": 0, "depth": 0}
    real = {"where": torch.where, "bincount": torch.bincount}

    def counter(name):
        def f(*a, **k):
            if n["depth"]:
                n[name] += 1
            return real[name](*a, **k)
        return f

    monkeypatch.setattr(torch, "where", counter("where"))
    monkeypatch.setattr(torch, "bincount", counter("bincount"))

    def drains_in_one_round(o):
        st, dr = tail_from_oracle(o, args)
        inner = model.MoE.forward

        def depth_marked(self, x, input_ids):
            n["depth"] += 1
            try:
                return inner(self, x, input_ids)
            finally:
                n["depth"] -= 1

        ids = torch.randint(0, args.vocab_size, (1, PROMPT),
                            generator=torch.Generator().manual_seed(0))
        tok, main = step(st, ids, 0)
        dr.prefill(tok, main)
        tok, main = step(st, tok, PROMPT)
        n["where"] = n["bincount"] = 0
        monkeypatch.setattr(model.MoE, "forward", depth_marked)
        try:
            dr.advance_and_draft(tok.unsqueeze(1), main, start_pos=PROMPT)
        finally:
            monkeypatch.setattr(model.MoE, "forward", inner)
        return n["where"], n["bincount"]

    ref_where, ref_bincount = drains_in_one_round(oracle)
    armed(monkeypatch, model)
    got_where, got_bincount = drains_in_one_round(REFCPU.build_oracle(args, SEED))
    assert ref_where >= args.n_mtp_layers and ref_bincount == args.n_mtp_layers, (
        f"the reference dispatch should cost one bincount + one torch.where per active expert "
        f"per mtp block; saw where={ref_where} bincount={ref_bincount}")
    assert (got_where, got_bincount) == (0, 0), \
        f"the fast path still costs where={got_where} bincount={got_bincount} host syncs"
