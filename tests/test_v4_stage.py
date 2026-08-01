"""DeepSeek-V4-Flash stage: a layer range is bit-identical whether it runs whole, split, or chunked.

Three headlines, and each is a different way a V4 pipeline can be silently wrong:

  test_three_stage_split_matches_single   V4's boundary payload is `h [b, s, hc_mult=4, dim]` plus
      the token ids -- four hyper-connection streams that persist through every layer and collapse
      only at hc_head, and the ids the first `n_hash_layers` MoE gates route on. Carry three streams,
      or forget the ids, and the ring still serves plausible-looking logits.

  test_chunk_loop_equals_stepwise         the reference's decode branch is hard `seqlen == 1`
      (`kv_cache[:, start_pos % win] = kv.squeeze(1)`), but speculative verify sends a chunk at
      `start_pos > 0`. v4_stage loops internally; this is the proof that the loop is exactly
      sequential decode and not an approximation of it.

  test_ids_reach_the_hash_gate            the red the greens are measured against: same payload,
      different ids, output MUST move. If it does not, the ids are being dropped somewhere.

Everything runs on CPU against v4_ref_cpu's toy 8-layer V4 (2 hash-routed MoE layers, sliding-window
+ ratio-4-indexed + ratio-8-compressed attention, 4 HC streams, 3 DSpark taps) with the whole
single-process Transformer as the oracle -- no GPU, no network, no 158 GiB checkpoint.

Run: python3 -m pytest tests/test_v4_stage.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
REFCPU = pytest.importorskip("v4_ref_cpu")
V4 = pytest.importorskip("v4_stage")

SEED = 7
PROMPT = 13
STEPS = 5
BATCH = 1


# ── fixtures ─────────────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def args():
    return REFCPU.cpu_args()


@pytest.fixture
def oracle(args):
    """A FRESH reference Transformer per test.

    Deliberately not module-scoped: the Compressor's kv_state/score_state are non-persistent buffers
    that a prefill only partially rewrites, so a second sequence on the same object starts from the
    first one's tail (v4_ref_cpu's module docstring). Every test that compares against the oracle
    needs it in its constructor state."""
    return REFCPU.build_oracle(args, SEED)


def _ids(n, args, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, args.vocab_size, (BATCH, n), generator=g)


# ── oracle -> stage weight transfer ──────────────────────────────────────────────────────────────

def stage_from_oracle(oracle, args, lo, hi, *, head=None, tail=None, dspark=False, spec_depth=None):
    """A Stage over [lo, hi) holding the oracle's OWN weights, transferred in process.

    No checkpoint round-trip and no name mapping: `Stage.layers[i]` is the same `Block` class the
    oracle built from the same ModelArgs, so `load_state_dict(oracle.layers[li].state_dict())` is
    exact and strict=True proves the two agree on every parameter. The boundary tensors go the same
    way, except the three `hc_head_*`, which are bare Transformer-level Parameters with no
    state_dict of their own and are copied by hand.

    `head`/`tail` default to the range's position in the stack, which is what a pipe would pass.
    Buffers (kv_cache, kv_state, score_state, freqs_cis) are all persistent=False and therefore NOT
    in any state_dict here -- each Stage allocates its own, in its constructor state."""
    head = (lo == 0) if head is None else head
    tail = (hi == args.n_layers) if tail is None else tail
    st = V4.Stage(lo, hi, args, head=head, tail=tail, dspark=dspark, device="cpu",
                  spec_depth=spec_depth)
    for li in range(lo, hi):
        st.layers[li - lo].load_state_dict(oracle.layers[li].state_dict(), strict=True)
    if st.embed_tokens is not None:
        st.embed_tokens.load_state_dict(oracle.embed.state_dict(), strict=True)
    if tail:
        st.norm.load_state_dict(oracle.norm.state_dict(), strict=True)
        st.lm_head.load_state_dict(oracle.head.state_dict(), strict=True)
        with torch.no_grad():
            for n in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                getattr(st, n).data.copy_(getattr(oracle, n).data)
    return st


def chain(oracle, args, splits, *, dspark=False):
    return [stage_from_oracle(oracle, args, lo, hi, dspark=dspark) for lo, hi in splits]


def run(stages, ids, start_pos):
    """Drive a chunk through the whole chain, exactly as a pipe would. -> (h, last-position logits)."""
    h = stages[0].embed(ids)
    for st in stages:
        h = st.forward(h, ids, start_pos)
    return h, stages[-1].logits_all(h, full_logits=False)


def greedy(stages, first, start_pos, steps):
    """`steps` greedy decode steps from token `first` at `start_pos`. -> list of [b] tensors."""
    out, ids, pos = [], first, start_pos
    for _ in range(steps):
        _, logits = run(stages, ids, pos)
        tok = logits.argmax(dim=-1)
        out.append(tok)
        ids, pos = tok.unsqueeze(1), pos + 1
    return out


# ── 1. one stage == the whole model ──────────────────────────────────────────────────────────────

def test_single_stage_matches_oracle_prefill_and_decode(oracle, args):
    """[0, n_layers) as one head+tail stage reproduces the oracle bit for bit, prefill and decode.

    The oracle's `Transformer.forward` returns LAST-POSITION logits (`ParallelHead.forward` slices
    before the vocab projection unless full_logits), so the comparison goes through the stage's own
    full_logits=False path -- the same GEMM shape, hence torch.equal rather than allclose."""
    M = V4.ref()
    before = (M.world_size, M.rank, M.default_dtype, M.scale_fmt, M.scale_dtype)
    st = stage_from_oracle(oracle, args, 0, args.n_layers)
    assert st.globals == before, "a Stage must derive model.py's globals exactly as Transformer does"

    ids = _ids(PROMPT, args)
    o_tok, o_logits, _ = oracle(ids)
    _, logits = run([st], ids, 0)
    assert torch.equal(logits, o_logits)
    assert torch.equal(logits.argmax(dim=-1), o_tok)
    assert st._pos == PROMPT

    tok = o_tok
    for i in range(PROMPT, PROMPT + STEPS):
        o_tok, o_logits, _ = oracle(tok.unsqueeze(1), i)
        _, logits = run([st], tok.unsqueeze(1), i)
        assert torch.equal(logits, o_logits), f"logits diverged at decode step {i}"
        tok = logits.argmax(dim=-1)
        assert torch.equal(tok, o_tok), f"token stream diverged at decode step {i}"
    assert st._pos == PROMPT + STEPS


# ── 2. the split ─────────────────────────────────────────────────────────────────────────────────

SPLITS = [(0, 3), (3, 6), (6, 8)]


def test_three_stage_split_matches_single(oracle, args):
    """Threading (h, ids) across two boundaries changes nothing. The headline.

    Both halves of the payload are under test at once: `h` is [b, s, 4, dim] because the four
    hyper-connection streams do not collapse until hc_head, and `ids` because layers 0 and 1 route
    their MoE by tid2eid[input_ids]. Compared against BOTH the single stage and the oracle, so a
    split that agrees with a broken single stage cannot pass."""
    assert SPLITS[0][0] == 0 and SPLITS[-1][1] == args.n_layers
    single = stage_from_oracle(oracle, args, 0, args.n_layers)
    split = chain(REFCPU.build_oracle(args, SEED), args, SPLITS)

    ids = _ids(PROMPT, args)
    o_tok, o_logits, _ = oracle(ids)
    h_one, l_one = run([single], ids, 0)
    h_many, l_many = run(split, ids, 0)
    assert torch.equal(h_one, h_many), "the 4-stream payload did not survive the boundaries"
    assert torch.equal(l_many, l_one) and torch.equal(l_many, o_logits)
    assert torch.equal(l_many.argmax(dim=-1), o_tok)

    got_one = greedy([single], o_tok.unsqueeze(1), PROMPT, STEPS)
    got_many = greedy(split, o_tok.unsqueeze(1), PROMPT, STEPS)
    want = []
    tok = o_tok
    for i in range(PROMPT, PROMPT + STEPS):
        tok = oracle(tok.unsqueeze(1), i)[0]
        want.append(tok)
    assert [t.tolist() for t in got_many] == [t.tolist() for t in want]
    assert [t.tolist() for t in got_many] == [t.tolist() for t in got_one]
    assert [st._pos for st in split] == [PROMPT + STEPS] * len(SPLITS)


# ── 3. the verify path ───────────────────────────────────────────────────────────────────────────

def test_chunk_loop_equals_stepwise(args):
    """A 4-token chunk at start_pos > 0 == the same 4 tokens fed one at a time. Step 4's whole bet.

    The reference cannot do this itself: its decode branch writes `kv_cache[:, start_pos % win]` and
    `kv_state[:, start_pos % ratio]` from a `squeeze(1)`, so a chunk would land only its last token
    and answer the other three off a stale window. v4_stage loops per position instead; if that loop
    ever stops matching sequential decode, every speculated token the ring verifies is wrong."""
    chunk = 4
    ids = _ids(PROMPT + chunk, args)
    prompt, extra = ids[:, :PROMPT], ids[:, PROMPT:]

    a = chain(REFCPU.build_oracle(args, SEED), args, SPLITS)
    b = chain(REFCPU.build_oracle(args, SEED), args, SPLITS)
    run(a, prompt, 0)
    run(b, prompt, 0)

    h_chunk, l_chunk = run(a, extra, PROMPT)
    steps = [run(b, extra[:, i:i + 1], PROMPT + i) for i in range(chunk)]
    h_step = torch.cat([h for h, _ in steps], dim=1)

    assert torch.equal(h_chunk, h_step), "the internal per-token loop is not sequential decode"
    assert torch.equal(l_chunk, steps[-1][1])
    assert [st._pos for st in a] == [st._pos for st in b] == [PROMPT + chunk] * len(SPLITS)


# ── 4. the DSpark tap ────────────────────────────────────────────────────────────────────────────

def test_tap_main_hidden(oracle, args):
    """An armed tail records `h.mean(dim=2)` after each target layer; unarmed it records nothing.

    This is `Transformer.forward:920-925`: the drafter's input is the mean over the four HC streams
    after layers dspark_target_layer_ids, concatenated in LAYER order."""
    n_t = len(args.dspark_target_layer_ids)
    st = stage_from_oracle(oracle, args, 0, args.n_layers, dspark=True)
    st._dspark = True

    ids = _ids(PROMPT, args)
    _, _, o_main = oracle(ids)
    run([st], ids, 0)
    main = st.tail_main_hidden()
    assert main.shape == (BATCH, PROMPT, n_t * args.dim)
    assert torch.equal(main, o_main)

    tok = _ids(1, args, seed=1)
    _, _, o_main = oracle(tok, PROMPT)
    run([st], tok, PROMPT)
    main = st.tail_main_hidden()
    assert main.shape == (BATCH, 1, n_t * args.dim)
    assert torch.equal(main, o_main)

    greedy_st = stage_from_oracle(oracle, args, 0, args.n_layers)
    run([greedy_st], _ids(PROMPT, args), 0)
    assert greedy_st._last_tap == {}, "the greedy path must clone nothing"
    with pytest.raises(RuntimeError, match="_dspark off"):
        greedy_st.tail_main_hidden()


def test_tap_refuses_a_split_target_range(oracle, args):
    """Targets spread over two stages is a config error, not a silent short concatenation."""
    st = stage_from_oracle(oracle, args, 6, args.n_layers, dspark=True)   # owns 6,7 — not 5
    st._dspark = True
    assert st._tap_ids == (6, 7)
    with pytest.raises(RuntimeError, match=r"\[5\] are not in this stage's range"):
        st.tail_main_hidden()


# ── 5. state ─────────────────────────────────────────────────────────────────────────────────────

def test_reset_clears_state(oracle, args):
    """Prefill, reset, prefill again -> bit-identical. Everything a job touched must be gone.

    The buffer that makes this a real test is `Compressor.score_state`: it is initialised to -inf,
    not zero (it is a softmax logit accumulator), and a prefill only rewrites the rows past the last
    full compression block. A reset that merely zeroed it would give the second prefill a uniform
    weight on slots the first one left behind, and the divergence would be small enough to look like
    rounding."""
    st = stage_from_oracle(oracle, args, 0, args.n_layers)
    ids = _ids(PROMPT, args)
    h1, l1 = run([st], ids, 0)
    _, l1b = run([st], _ids(3, args, seed=2), PROMPT)      # dirty the decode-side state too
    st.reset()
    assert st._pos == 0
    h2, l2 = run([st], ids, 0)
    assert torch.equal(h1, h2) and torch.equal(l1, l2)
    assert torch.equal(l2, oracle(ids)[1])


def test_snapshot_restore_round_trips(oracle, args):
    """The step-5 seam: snapshot -> mutate -> restore leaves the stage answering as it did.

    Not a rollback test (that needs _seek, which is step 5) -- it checks that `_snapshot` covers
    everything a chunk at start_pos > 0 disturbs EXCEPT the compressed regions, whose staleness is
    argued away in `Stage._snapshot`'s docstring. Restoring and replaying the same chunk must
    reproduce it exactly."""
    st = stage_from_oracle(oracle, args, 0, args.n_layers)
    st._spec = True
    ids = _ids(PROMPT + 4, args)
    run([st], ids[:, :PROMPT], 0)

    snap = st._snapshot()
    h1, l1 = run([st], ids[:, PROMPT:], PROMPT)
    assert st._spec_ckpt["start_pos"] == PROMPT and st._spec_ckpt["s"] == 4
    st._restore(snap)
    st._pos = PROMPT
    h2, l2 = run([st], ids[:, PROMPT:], PROMPT)
    assert torch.equal(h1, h2) and torch.equal(l1, l2)


# ── 6. the real load path ────────────────────────────────────────────────────────────────────────

def _write_converted(oracle, d, drop=()):
    """The oracle's weights in convert.py's OUTPUT format. -> the dir.

    convert.py's names ARE `Transformer.state_dict()`'s names: generate.py:91 feeds the file it
    writes straight into `load_model(model, ...)`, so `embed.weight`, `layers.0.attn.wq_a.weight`,
    `hc_head_fn`, ... round-trip unchanged. The one filter is convert.py's own (convert.py:91): the
    DSpark stages ALIAS the main embedding and head (`self.mtp[-1].embed = self.embed`), so those
    keys are the same storage under a second name and safetensors refuses to write both."""
    os.makedirs(d, exist_ok=True)
    sd = {k: v.detach().clone().contiguous() for k, v in oracle.state_dict().items()
          if k not in drop
          and not (k.startswith("mtp.") and ("emb" in k or k.endswith("head.weight")))}
    assert any(k.startswith("mtp.") for k in sd), "mtp.* must be in the file and ignored by the tail"
    safetensors_torch.save_file(sd, os.path.join(d, "model0-mp1.safetensors"))
    return d


def test_load_roundtrip_converted_format(oracle, args, tmp_path):
    """Stage.load off a converted-format file reproduces the in-process transfer, split and all.

    Everything but DeepSeek's own HF->converted step, which is `convert.py` run per box at ring time.
    Also pins the two things a stage must not do: it must ignore `mtp.*` (step 4's drafter owns it),
    and it must not expect the non-persistent buffers, which are absent from any state_dict."""
    assert not any(k.endswith(("kv_cache", "kv_state", "score_state", "freqs_cis"))
                   for k in oracle.state_dict()), "buffers are persistent=False and must stay out"
    d = _write_converted(oracle, str(tmp_path / "ckpt"))

    loaded = [V4.Stage(lo, hi, args, head=(lo == 0), tail=(hi == args.n_layers),
                       device="cpu").load(d) for lo, hi in SPLITS]
    want = chain(REFCPU.build_oracle(args, SEED), args, SPLITS)

    ids = _ids(PROMPT, args)
    o_tok, o_logits, _ = oracle(ids)
    _, l_loaded = run(loaded, ids, 0)
    _, l_want = run(want, ids, 0)
    assert torch.equal(l_loaded, l_want) and torch.equal(l_loaded, o_logits)
    assert greedy(loaded, o_tok.unsqueeze(1), PROMPT, STEPS) == \
           greedy(want, o_tok.unsqueeze(1), PROMPT, STEPS)


def test_load_is_strict_about_a_missing_tensor(oracle, args, tmp_path):
    """One tensor short is a hard failure, not a random-init layer behind a valid receipt."""
    d = _write_converted(oracle, str(tmp_path / "holed"), drop=("layers.7.attn.wq_a.weight",))
    with pytest.raises(RuntimeError, match="Missing key"):
        V4.Stage(6, 8, args, tail=True, device="cpu").load(d)

    empty = tmp_path / "empty"
    _write_converted(oracle, str(empty), drop=tuple(k for k in oracle.state_dict()
                                                    if k.startswith("layers.7.")))
    with pytest.raises(RuntimeError, match="no tensor under 'layers.7.'"):
        V4.Stage(6, 8, args, tail=True, device="cpu").load(str(empty))


# ── 7. the seek guards ───────────────────────────────────────────────────────────────────────────

def test_seek_guards(oracle, args):
    """A gap is a bug; a rewind with nothing to rewind INTO is a bug. Both refuse loudly rather than
    answering off stale state — an unarmed stage has no checkpoint, so its only way back is reset()."""
    st = stage_from_oracle(oracle, args, 0, args.n_layers)
    ids = _ids(5, args)
    run([st], ids, 0)
    tok = _ids(1, args, seed=3)
    with pytest.raises(RuntimeError, match="is ahead of the 5 tokens"):
        st.forward(st.embed(tok), tok, 7)
    with pytest.raises(RuntimeError, match="checkpoint covers none"):
        st.forward(st.embed(tok), tok, 3)
    assert st._pos == 5, "a refused seek must not move the stage"


# ── 7b. the rollback ─────────────────────────────────────────────────────────────────────────────

CHUNK = 4


def stream(st, first, start_pos, steps):
    """`steps` greedy steps from `first`, keeping the PAYLOAD as well as the logits at each one.

    greedy() above compares token ids, which is the user-visible bar; a rollback has to clear a
    higher one. The hidden state is what the next hop consumes, so two stages whose `h` differs by an
    ulp can still emit the same tokens for a while and then diverge — comparing `h` every step is how
    a partially-restored window ring is caught in the round it happens rather than ten later."""
    out, ids, pos = [], first, start_pos
    for _ in range(steps):
        h, logits = run([st], ids, pos)
        out.append((h, logits))
        ids, pos = logits.argmax(dim=-1).unsqueeze(1), pos + 1
    return out


@pytest.mark.parametrize("k", range(CHUNK + 1))
def test_rollback_is_bit_exact(oracle, args, k):
    """Spec chunk of 4 at PROMPT, rewind to PROMPT+k, decode on: identical to never having speculated.

    THE headline of the rollback. `k` is how many of the chunk's positions the round committed (k=0 is
    a drafter that got nothing right, k=CHUNK is a full accept), so the reference side is a stage fed
    the prompt plus exactly those k tokens ONE AT A TIME — the sequential decode the ring claims to be
    reproducing. Bit-exact on the payload AND the logits at every subsequent step: anything the
    rejected tail left in the window ring or in a compressor accumulator shows up here, and nowhere
    else, because a wrong slot is still a plausible number."""
    ids = _ids(PROMPT + CHUNK, args)
    prompt, chunk = ids[:, :PROMPT], ids[:, PROMPT:]

    st = stage_from_oracle(oracle, args, 0, args.n_layers)
    st._spec = True                                        # the reset's `spec` flag, on the wire
    run([st], prompt, 0)
    run([st], chunk, PROMPT)                               # the speculated chunk, most of it doomed
    assert st._pos == PROMPT + CHUNK

    st._seek(PROMPT + k)
    assert st._pos == PROMPT + k
    if k == CHUNK:                                         # FULL ACCEPT: the next chunk opens exactly
        assert st._spec_ckpt is not None, \
            "a full accept must be the no-op path — nothing restored, nothing replayed, nothing spent"
    else:
        assert st._spec_ckpt is None, "the checkpoint is spent by the rewind that used it"

    want = stage_from_oracle(REFCPU.build_oracle(args, SEED), args, 0, args.n_layers)
    run([want], prompt, 0)
    for j in range(k):
        run([want], chunk[:, j:j + 1], PROMPT + j)

    first = _ids(1, args, seed=5)
    for i, ((h_got, l_got), (h_want, l_want)) in enumerate(
            zip(stream(st, first, PROMPT + k, STEPS), stream(want, first, PROMPT + k, STEPS))):
        assert torch.equal(h_got, h_want), f"payload diverged {i} steps after a rewind to +{k}"
        assert torch.equal(l_got, l_want), f"logits diverged {i} steps after a rewind to +{k}"


def _poison_stale_compressed(st):
    """NaN every compressed slot `Stage._snapshot` calls safe-to-be-stale, at the stage's position.

    Slot j of a ratio-R region is written at position (j+1)*R-1, so after `_pos` positions the slots
    the committed history legitimately wrote are [0, _pos // R) and everything from there up is
    whatever a rejected speculation happened to leave behind. Those are exactly the slots the
    snapshot deliberately does not cover."""
    win = st.args.window_size
    with torch.no_grad():
        for L in st.layers:
            r = L.attn.compress_ratio
            if not r:
                continue
            first = st._pos // r
            L.attn.kv_cache[:, win + first:] = float("nan")
            if L.attn.indexer is not None:
                L.attn.indexer.kv_cache[:, first:] = float("nan")


@pytest.mark.parametrize("prompt,s,k", [(13, 6, 2), (11, 6, 1), (13, 4, 0), (15, 5, 3)])
def test_rollback_survives_a_poisoned_compressed_region(oracle, args, prompt, s, k):
    """The compressed regions are NOT snapshotted, and the argument for that has ZERO margin.

    `_snapshot`'s docstring works out that a compressed slot is first read AT the position that
    writes it — not after — so a slot a rejected chunk poisoned survives only because the reference
    calls `self.compressor(x, start_pos)` BEFORE it reads `kv_cache` inside that same position
    (model.py:537-538, and :423-426 for the Indexer). Reorder either pair — a re-vendored model.py,
    a fused attention kernel that captures the compressed region at entry — and speculation starts
    reading rejected tokens' compression, silently, in numbers that still look like numbers.

    So this does not trust the rejected chunk to have written something harmless: after the rewind it
    fills every slot the argument calls safe with NaN. If any of them is ever read before being
    rewritten, NaN propagates and torch.equal fails. Every case here rewinds with a COMPRESSION
    BOUNDARY inside the rejected tail (positions where (p+1) % 4 == 0 or % 8 == 0 for cpu_args'
    ratios), which is the only place the zero margin is even exercised."""
    ratios = [r for r in args.compress_ratios if r]
    assert any((p + 1) % r == 0 for r in ratios for p in range(prompt + k, prompt + s)), \
        "this case rejects no compression boundary — it would not exercise the argument at all"

    ids = _ids(prompt + s, args)
    head, chunk = ids[:, :prompt], ids[:, prompt:]
    st = stage_from_oracle(oracle, args, 0, args.n_layers)
    st._spec = True
    run([st], head, 0)
    run([st], chunk, prompt)
    st._seek(prompt + k)
    _poison_stale_compressed(st)

    want = stage_from_oracle(REFCPU.build_oracle(args, SEED), args, 0, args.n_layers)
    run([want], head, 0)
    for j in range(k):
        run([want], chunk[:, j:j + 1], prompt + j)

    first = _ids(1, args, seed=5)
    for i, ((h_got, l_got), (h_want, l_want)) in enumerate(
            zip(stream(st, first, prompt + k, STEPS + 3), stream(want, first, prompt + k, STEPS + 3))):
        assert torch.isfinite(h_got).all(), \
            f"step {i}: a poisoned compressed slot was READ before being rewritten — the " \
            f"write-before-read ordering in model.py's Attention/Indexer no longer holds"
        assert torch.equal(h_got, h_want) and torch.equal(l_got, l_want), \
            f"step {i}: rollback + poisoned compressed region diverged from sequential decode"


def test_rewind_outside_the_checkpoint_refuses(oracle, args):
    """One chunk deep. A rewind past the checkpointed interval has no state to restore FROM, and
    guessing would be worse than failing: the error names the interval it does cover."""
    st = stage_from_oracle(oracle, args, 0, args.n_layers)
    st._spec = True
    ids = _ids(PROMPT + CHUNK, args)
    run([st], ids[:, :PROMPT], 0)
    run([st], ids[:, PROMPT:], PROMPT)

    with pytest.raises(RuntimeError, match=rf"checkpoint covers \[{PROMPT}, {PROMPT + CHUNK}\]"):
        st._seek(PROMPT - 1)
    assert st._pos == PROMPT + CHUNK and st._spec_ckpt is not None, \
        "a refused rewind must leave the stage and its checkpoint exactly as they were"


def test_replay_leaves_the_taps_alone(oracle, args):
    """A rollback must not re-record `_last_tap`. The drafter already consumed the verify chunk's
    taps — that is how it knew what to draft — and it is advanced over COMMITTED positions only, so a
    replay that rewrote them would hand the tail a second, shorter copy of its own history."""
    st = stage_from_oracle(oracle, args, 0, args.n_layers, dspark=True)
    st._dspark = True
    st._spec = True
    ids = _ids(PROMPT + CHUNK, args)
    run([st], ids[:, :PROMPT], 0)
    run([st], ids[:, PROMPT:], PROMPT)
    taps = st.tail_main_hidden().clone()
    assert taps.shape[1] == CHUNK, "the chunk's taps are one row per chunk position"

    st._seek(PROMPT + 2)
    assert torch.equal(st.tail_main_hidden(), taps), "the replay overwrote the chunk's taps"


def test_forward_refuses_a_payload_without_ids(oracle, args):
    st = stage_from_oracle(oracle, args, 0, args.n_layers)
    ids = _ids(5, args)
    with pytest.raises(RuntimeError, match="needs the token ids"):
        st.forward(st.embed(ids), None, 0)


def test_a_refused_frame_does_not_spend_the_rollback(oracle, args):
    """A malformed chunk must be refused BEFORE `_seek` touches anything.

    `_seek` is the one mutating validator: a rewind restores the snapshot, replays the accepted
    prefix and SPENDS the checkpoint. If a frame that is going to be rejected gets to run it first,
    the round's retry finds the stage already rewound with nothing left to rewind into — a refused
    frame would have narrowed what the ring can still roll back to."""
    st = stage_from_oracle(oracle, args, 0, args.n_layers)
    st._spec = True
    ids = _ids(PROMPT + CHUNK, args)
    run([st], ids[:, :PROMPT], 0)
    run([st], ids[:, PROMPT:], PROMPT)

    mismatched = _ids(2, args, seed=9)                     # 2 ids against a 3-position payload
    with pytest.raises(RuntimeError, match="do not match"):
        st.forward(st.embed(_ids(3, args, seed=9)), mismatched, PROMPT + 1)
    assert st._pos == PROMPT + CHUNK and st._spec_ckpt is not None, \
        "a rejected frame rewound the stage before noticing it was malformed"


# ── 7c. the W-deep rollback — pipelined speculation ──────────────────────────────────────────────

def stream_spec(st, ids, start_pos):
    """Feed `ids` through a single stage one position at a time with _spec armed — a sender that
    streams s=1 speculative frames back-to-back, which is what fills the 5-of-6 idle pipeline stages.

    Each forward pushes its own pre-frame checkpoint, so the ring ends holding one per position: the
    W un-judged snapshots a rejection W frames downstream has to rewind through. The one-shot s=W
    chunk the existing tests use collapses all of that into a single checkpoint and never exercises
    the ring."""
    for i in range(ids.shape[1]):
        st.forward(st.embed(ids[:, i:i + 1]), ids[:, i:i + 1], start_pos + i)


@pytest.mark.parametrize("prompt,W,k", [
    (13, 12, 0),    # reject all 12 — rewind across ratio-4 boundaries 15,19,23 and ratio-8 15,23
    (13, 12, 3),    # commit 3, reject a boundary-crossing tail
    (11, 14, 1),    # a 13-frame rejected tail, several ratio blocks wide
    (17, 10, 5),    # tail [22,27) crosses the ratio-4 AND ratio-8 boundary at p=23
    (40, 12, 3),    # past index_topk*ratio=32: the Indexer now DISCRIMINATES, so its own compressor
    (40, 12, 0),    # accumulators are load-bearing — a regime the short-prompt tests never reach
    (13, 12, 12),   # full accept at depth W — still the no-op path
])
def test_multi_deep_rollback_across_boundaries(oracle, args, prompt, W, k):
    """Stream W s=1 frames, rewind up to W deep across several compression boundaries, NaN-poison the
    stale compressed region, and match sequential decode bit-for-bit.

    test_rollback_is_bit_exact + test_rollback_survives_a_poisoned_compressed_region taken to the
    depth pipelined speculation runs at: not one chunk, but W frames streamed before the first reply,
    so the rejection unwinds a ring of W checkpoints crossing ratio-4 (overlap) and ratio-8 (plain)
    boundaries at once. The compressed regions are STILL not snapshotted; the poison proves the
    depth-invariant write-before-read argument in Stage._snapshot's docstring holds W deep, not one."""
    ratios = [r for r in args.compress_ratios if r]
    assert k == W or any((p + 1) % r == 0 for r in ratios for p in range(prompt + k, prompt + W)), \
        "this case rejects no compression boundary — it would not exercise the argument"

    ids = _ids(prompt + W, args)
    head, chunk = ids[:, :prompt], ids[:, prompt:]

    st = stage_from_oracle(oracle, args, 0, args.n_layers)
    st._spec = True
    run([st], head, 0)
    stream_spec(st, chunk, prompt)
    assert st._pos == prompt + W
    assert len(st._spec_ckpts) == W, "one pre-frame checkpoint per streamed s=1 frame"

    st._seek(prompt + k)
    assert st._pos == prompt + k
    if k == W:
        assert len(st._spec_ckpts) == W, "full accept is the no-op path — nothing restored or spent"
    else:
        assert all(c["start_pos"] < prompt + k for c in st._spec_ckpts), \
            "every checkpoint at or after the rewind target is spent; earlier ones survive"
        _poison_stale_compressed(st)

    want = stage_from_oracle(REFCPU.build_oracle(args, SEED), args, 0, args.n_layers)
    run([want], head, 0)
    for j in range(k):
        run([want], chunk[:, j:j + 1], prompt + j)

    first = _ids(1, args, seed=5)
    for i, ((h_got, l_got), (h_want, l_want)) in enumerate(
            zip(stream(st, first, prompt + k, STEPS + 3),
                stream(want, first, prompt + k, STEPS + 3))):
        assert torch.isfinite(h_got).all(), \
            f"step {i}: a poisoned compressed slot was READ before being rewritten — the " \
            f"write-before-read ordering broke {W - k} frames deep"
        assert torch.equal(h_got, h_want) and torch.equal(l_got, l_want), \
            f"step {i}: W-deep rollback diverged from sequential decode (rewound {W - k} frames)"


def _snapshot_dropping(st, region):
    """A `_snapshot` that fails to preserve exactly ONE region the real one covers — the mutation the
    green must catch. Restore it and the W-deep rollback has to diverge, or the poison test is
    vacuous (a snapshot that copied nothing would pass it just as well)."""
    real = V4.Stage._snapshot

    def broken():
        snap = real(st)
        n = len(st.layers)
        if region == "window":
            for e in snap[:n]:
                e["win"].zero_()
        else:
            for (c, is_idx), e in zip(st._compressors(), snap[n:]):
                if region == "indexer" and not is_idx:
                    continue
                if region in ("kv_state", "indexer"):
                    e["kv_state"].zero_()
                if region in ("score_state", "indexer"):
                    e["score_state"].zero_()
        return snap

    return broken


@pytest.mark.parametrize("region", ["window", "kv_state", "score_state", "indexer"])
def test_multi_deep_rollback_mutation_check(oracle, args, region):
    """Every region the W-deep snapshot preserves is load-bearing: drop it and the rollback DIVERGES.

    Long prompt on purpose — past index_topk*ratio the Indexer discriminates, so dropping its own
    compressor's accumulators actually moves the selected slots and therefore the output; under 32
    tokens it picks everything and the drop would be silently harmless (which is itself the reason the
    short-prompt suite could never have pinned the Indexer's state)."""
    prompt, W, k = 40, 12, 2
    ids = _ids(prompt + W, args)
    head, chunk = ids[:, :prompt], ids[:, prompt:]

    st = stage_from_oracle(oracle, args, 0, args.n_layers)
    st._spec = True
    st._snapshot = _snapshot_dropping(st, region)
    run([st], head, 0)
    stream_spec(st, chunk, prompt)
    st._seek(prompt + k)

    want = stage_from_oracle(REFCPU.build_oracle(args, SEED), args, 0, args.n_layers)
    run([want], head, 0)
    for j in range(k):
        run([want], chunk[:, j:j + 1], prompt + j)

    first = _ids(1, args, seed=5)
    diverged = any(
        not torch.equal(g[0], w[0]) or not torch.equal(g[1], w[1])
        for g, w in zip(stream(st, first, prompt + k, STEPS + 5),
                        stream(want, first, prompt + k, STEPS + 5)))
    assert diverged, \
        f"dropping {region!r} from the snapshot changed nothing — the W-deep poison test is vacuous"


def test_multi_deep_rollback_exceeds_window(oracle, args):
    """A rewind farther back than `window_size` positions is still exact. The window snapshot is a
    FULL copy of the ring, not its last few slots, so the ring's depth is bounded by W (the checkpoint
    count) and never by the window — the one buffer whose size might have looked like the real cap."""
    win = args.window_size
    prompt, W, k = 20, win + 6, 0                          # 22 frames, well past the 16-slot ring
    assert W > win
    st = stage_from_oracle(oracle, args, 0, args.n_layers, spec_depth=W + 2)
    st._spec = True
    ids = _ids(prompt + W, args)
    head, chunk = ids[:, :prompt], ids[:, prompt:]
    run([st], head, 0)
    stream_spec(st, chunk, prompt)
    st._seek(prompt + k)
    _poison_stale_compressed(st)

    want = stage_from_oracle(REFCPU.build_oracle(args, SEED), args, 0, args.n_layers)
    run([want], head, 0)

    first = _ids(1, args, seed=5)
    for i, ((h_got, l_got), (h_want, l_want)) in enumerate(
            zip(stream(st, first, prompt + k, STEPS + 3),
                stream(want, first, prompt + k, STEPS + 3))):
        assert torch.isfinite(h_got).all() and torch.equal(h_got, h_want) and torch.equal(l_got, l_want), \
            f"step {i}: rewind {W} deep (> window {win}) diverged from sequential decode"


def test_multi_deep_rollback_on_a_split_chain(oracle, args):
    """The ring is PER-STAGE: the coordinator rewinds every stage to the same position, and each one
    restores its own window ring and compressor accumulators. A three-stage split streamed and rewound
    W deep must decode exactly what a fresh chain fed the accepted prefix one token at a time does — so
    nothing about the rollback depends on a stage owning the whole stack."""
    prompt, W, k = 13, 12, 3
    ids = _ids(prompt + W, args)
    head, chunk = ids[:, :prompt], ids[:, prompt:]

    split = chain(REFCPU.build_oracle(args, SEED), args, SPLITS)
    for st in split:
        st._spec = True
    run(split, head, 0)
    for i in range(W):
        tok = chunk[:, i:i + 1]
        h = split[0].embed(tok)
        for st in split:
            h = st.forward(h, tok, prompt + i)
    for st in split:
        st._seek(prompt + k)
    assert [st._pos for st in split] == [prompt + k] * len(SPLITS)

    want = chain(REFCPU.build_oracle(args, SEED), args, SPLITS)
    run(want, head, 0)
    for j in range(k):
        run(want, chunk[:, j:j + 1], prompt + j)

    first = _ids(1, args, seed=5)
    assert [t.tolist() for t in greedy(split, first, prompt + k, STEPS)] == \
           [t.tolist() for t in greedy(want, first, prompt + k, STEPS)], \
        "the split chain's rollback diverged from sequential decode"


def test_rewind_deeper_than_W_refuses(oracle, args):
    """The ring is W deep and no deeper. Stream W+2 frames into a W-cap ring: the two oldest are
    evicted, so a rewind to their positions refuses (loudly, naming the interval still covered) rather
    than serving off a checkpoint that no longer exists — while a rewind INSIDE the ring still works."""
    W, prompt = 4, 13
    st = stage_from_oracle(oracle, args, 0, args.n_layers, spec_depth=W)
    st._spec = True
    ids = _ids(prompt + W + 2, args)
    run([st], ids[:, :prompt], 0)
    stream_spec(st, ids[:, prompt:], prompt)
    assert len(st._spec_ckpts) == W, "maxlen caps the ring at W live checkpoints"
    assert st._spec_ckpts[0]["start_pos"] == prompt + 2, "the two oldest frames were evicted"

    with pytest.raises(RuntimeError, match="cannot rewind"):
        st._seek(prompt + 1)                               # below the ring — evicted
    assert st._pos == prompt + W + 2, "a refused rewind must not move the stage"

    st._seek(prompt + 3)                                   # inside the ring — fine
    assert st._pos == prompt + 3


def test_commit_drops_settled_checkpoints(oracle, args):
    """commit(pos) frees the checkpoints the ring has settled irrevocably past — the memory the
    W-deep ring costs — and leaves everything still in flight rewindable and bit-exact."""
    prompt, W = 13, 8
    st = stage_from_oracle(oracle, args, 0, args.n_layers)
    st._spec = True
    ids = _ids(prompt + W, args)
    run([st], ids[:, :prompt], 0)
    stream_spec(st, ids[:, prompt:], prompt)
    assert len(st._spec_ckpts) == W

    st.commit(prompt + 3)
    assert all(c["start_pos"] + c["s"] > prompt + 3 for c in st._spec_ckpts)
    assert st._spec_ckpts[0]["start_pos"] == prompt + 3, "s=1 frames ending at/below the ack are gone"

    with pytest.raises(RuntimeError, match="cannot rewind"):
        st._seek(prompt + 2)                               # settled past — refuse
    st._seek(prompt + 5)                                   # still in flight — bit-exact rewind
    assert st._pos == prompt + 5


# ── 8. the red test ──────────────────────────────────────────────────────────────────────────────

def test_ids_reach_the_hash_gate(oracle, args):
    """Same payload, different ids -> different output. Guards against silently dropping the ids.

    `Gate.forward` uses `input_ids` only when `layer_id < n_hash_layers`, and only to index
    `tid2eid`. So this feeds an IDENTICAL `h` with two different id chunks: any difference can only
    have come through the gate. The tid2eid rows are asserted distinct first, so a green here is
    never an accident of two ids happening to route the same way."""
    assert args.n_hash_layers > 0
    gate = oracle.layers[0].ffn.gate
    assert gate.hash
    a_id, b_id = 0, 0
    for i in range(args.vocab_size):
        if not torch.equal(gate.tid2eid[i], gate.tid2eid[0]):
            b_id = i
            break
    assert b_id, "no two token ids in this random tid2eid route differently"

    st_a = stage_from_oracle(oracle, args, 0, args.n_layers)
    st_b = stage_from_oracle(oracle, args, 0, args.n_layers)
    prompt = _ids(PROMPT, args)
    run([st_a], prompt, 0)
    run([st_b], prompt, 0)

    tok_a = torch.full((BATCH, 1), a_id, dtype=torch.long)
    tok_b = torch.full((BATCH, 1), b_id, dtype=torch.long)
    h = st_a.embed(tok_a)
    h_a = st_a.forward(h.clone(), tok_a, PROMPT)
    h_b = st_b.forward(h.clone(), tok_b, PROMPT)
    assert not torch.equal(h_a, h_b), "the ids never reached the MoE gate"

    st_c = stage_from_oracle(oracle, args, 0, args.n_layers)
    run([st_c], prompt, 0)
    assert torch.equal(st_c.forward(h.clone(), tok_a, PROMPT), h_a), "control: same ids, same output"
