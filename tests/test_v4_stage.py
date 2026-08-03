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

Run: OMP_NUM_THREADS=1 python3 -m pytest tests/test_v4_stage.py -q

Pin the threads. At cpu_args' toy shape (dim 256) torch's intra-op threading is pure contention --
measured 1.78 s per decode step at 4 threads against 0.079 s at 1, a 22x slowdown, and this suite
decodes thousands of single positions. It changes no numerics (every comparison here is
stage-vs-oracle inside ONE process at the same thread count).
"""
import contextlib
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

def stage_from_oracle(oracle, args, lo, hi, *, head=None, tail=None, dspark=False,
                      spec_depth=None, fast=False):
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
                  spec_depth=spec_depth, fast_verify=fast)
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


def test_multi_deep_rollback_at_a_larger_ratio(args):
    """The depth-invariance argument is RATIO-AGNOSTIC, so prove it at a ratio far from 4 and 8.

    The shipped config compresses at 4 and 128; cpu_args uses 4 and 8 so a short prompt still crosses a
    boundary. That leaves "does a big ratio behave differently" untested, and the whole reason a toy
    oracle licenses a claim about the 158 GiB model is that the read set [0,(P+1)//ratio) has the same
    shape at every ratio. Ratio 16 (plain, no overlap — only 4 overlaps) with a rejected tail crossing
    p=31 is that check."""
    a = REFCPU.cpu_args(compress_ratios=(0, 0, 4, 16, 4, 16, 4, 0, 0, 0))
    prompt, W, k = 20, 14, 2
    assert any((p + 1) % 16 == 0 for p in range(prompt + k, prompt + W)), "must cross a ratio-16 bound"
    o = REFCPU.build_oracle(a, SEED)
    ids = _ids(prompt + W, a)
    head, chunk = ids[:, :prompt], ids[:, prompt:]

    st = stage_from_oracle(o, a, 0, a.n_layers)
    st._spec = True
    run([st], head, 0)
    stream_spec(st, chunk, prompt)
    st._seek(prompt + k)
    _poison_stale_compressed(st)

    want = stage_from_oracle(REFCPU.build_oracle(a, SEED), a, 0, a.n_layers)
    run([want], head, 0)
    for j in range(k):
        run([want], chunk[:, j:j + 1], prompt + j)

    first = _ids(1, a, seed=5)
    for i, ((h_got, l_got), (h_want, l_want)) in enumerate(
            zip(stream(st, first, prompt + k, STEPS), stream(want, first, prompt + k, STEPS))):
        assert torch.isfinite(h_got).all() and torch.equal(h_got, h_want) and torch.equal(l_got, l_want), \
            f"step {i}: W-deep rollback diverged at compress ratio 16"


def test_multi_deep_rollback_at_the_shipped_depth_and_ratio_alternation(args):
    """The SHIPPED shape of the problem, end to end: compress_ratios alternating 4/128 (the real
    config's pattern — config.json ships [0,0,4,128,4,128,...]), a full V4_SPEC_DEPTH=16 frames in
    flight, and a rejection at depth 16 rewinding across the ratio-128 boundary at p=127 (which is a
    ratio-4 boundary too, 128 % 4 == 0 — both compressor kinds cross at once).

    Two zero-margin facts are exercised together and neither is covered by the smaller cases: the
    checkpoint ring holds exactly W=16 snapshots, so a rewind to the oldest streamed frame lands on
    the ring's LAST live checkpoint; and a ratio-128 accumulator folds 128 positions before emitting
    a slot, so the rejected tail poisons an accumulator that will not emit again for ~128 positions —
    the restore has to put back the running state, not just the window."""
    a = REFCPU.cpu_args(compress_ratios=(0, 0, 4, 128, 4, 128, 4, 0, 0, 0))
    prompt, W, k = 115, 16, 0
    assert W == int(os.environ.get("V4_SPEC_DEPTH", 16)), "W must be the shipped rollback depth"
    assert any((p + 1) % 128 == 0 for p in range(prompt + k, prompt + W)), \
        "the rejected tail must cross the ratio-128 boundary at p=127"
    o = REFCPU.build_oracle(a, SEED)
    ids = _ids(prompt + W, a)
    head, chunk = ids[:, :prompt], ids[:, prompt:]

    st = stage_from_oracle(o, a, 0, a.n_layers, spec_depth=W)
    st._spec = True
    run([st], head, 0)
    stream_spec(st, chunk, prompt)
    assert len(st._spec_ckpts) == W, "exactly W live checkpoints — the rewind has zero margin"
    st._seek(prompt + k)
    _poison_stale_compressed(st)

    want = stage_from_oracle(REFCPU.build_oracle(a, SEED), a, 0, a.n_layers)
    run([want], head, 0)
    for j in range(k):
        run([want], chunk[:, j:j + 1], prompt + j)

    first = _ids(1, a, seed=5)
    for i, ((h_got, l_got), (h_want, l_want)) in enumerate(
            zip(stream(st, first, prompt + k, STEPS), stream(want, first, prompt + k, STEPS))):
        assert torch.isfinite(h_got).all() and torch.equal(h_got, h_want) and torch.equal(l_got, l_want), \
            f"step {i}: a depth-16 rollback across the 4/128 boundary diverged from sequential decode"


def test_multi_deep_rollback_batched(args):
    """b=2. `_snapshot` clones whole buffers (all `max_batch_size` rows), not `[:bsz]`, and the
    reference indexes every write `[:bsz, ...]` — so a rollback must restore BOTH rows and the two
    sequences must stay independent. A snapshot that silently covered only row 0 passes every b=1 test
    in this file."""
    prompt, W, k = 13, 10, 3
    b = 2
    assert args.max_batch_size >= b
    g = torch.Generator().manual_seed(11)
    ids = torch.randint(0, args.vocab_size, (b, prompt + W), generator=g)
    head, chunk = ids[:, :prompt], ids[:, prompt:]

    st = stage_from_oracle(REFCPU.build_oracle(args, SEED), args, 0, args.n_layers)
    st._spec = True
    run([st], head, 0)
    stream_spec(st, chunk, prompt)
    st._seek(prompt + k)
    _poison_stale_compressed(st)

    want = stage_from_oracle(REFCPU.build_oracle(args, SEED), args, 0, args.n_layers)
    run([want], head, 0)
    for j in range(k):
        run([want], chunk[:, j:j + 1], prompt + j)

    nxt = torch.randint(0, args.vocab_size, (b, 1), generator=g)
    pos = prompt + k
    for i in range(STEPS):
        h_got, l_got = run([st], nxt, pos + i)
        h_want, l_want = run([want], nxt, pos + i)
        assert h_got.shape[0] == b
        assert torch.isfinite(h_got).all() and torch.equal(h_got, h_want) and torch.equal(l_got, l_want), \
            f"step {i}: batched W-deep rollback diverged from sequential decode"
        nxt = l_got.argmax(dim=-1).unsqueeze(1)


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


# ── 9. the chunked verify path (V4_FAST_VERIFY) ──────────────────────────────────────────────────
#
# WHAT IS PROVED HERE, AND WHY IT RUNS UNDER A BATCH-INVARIANT HARNESS
#
# The fast path runs an s-token chunk in ONE pass per layer instead of s. Its MECHANICS -- which kv
# each position attends, in which order, which compressed slots exist by then, what the window ring
# and the compressor accumulators end up holding -- must be EXACTLY the per-token loop's. Its
# ARITHMETIC cannot be, and no implementation could make it so: a batched pass hands torch bigger
# tensors, and torch reassociates its reductions by SIZE. A GEMM at M = s tiles its K-reduction
# differently from M = 1 (cuBLAS, MKL, and -- the one that first bit this path in CI -- the einsum
# inside `sparse_attn`, whose d-contraction over a [b, s, ...] query reorders relative to [b, 1, ...]).
# So "chunk == loop, bit for bit" is a statement about the ALGORITHM that is only well-posed once that
# one freedom is removed, and removing it is exactly what `batch_invariant()` below does: it swaps
# every linear and every einsum for a broadcast-multiply-then-fp32-sum, whose reduction order is fixed
# per output element and independent of s. Under it BOTH paths compute the identical (lower-precision,
# deterministic) numbers, so a surviving difference is a real mechanical error and nothing else.
#
# This is not a workaround for a flaky test -- it is the only honest form of the claim. An earlier
# cut asserted torch.equal on raw arithmetic and passed only because THIS box happened to reassociate
# a small toy identically at s <= 6; CI's build did not, and the "failure" was reassociation wearing a
# mechanical error's error message. A batch-invariant sweep over every window-straddling position
# (15,16,31,32,63,127 x s in 2,4,6, every layer kind) is bit-exact, which is the real proof the wrap
# is correct; what follows pins it permanently.
#
#   attends_exactly / leaves_the_same_state / rows_are_the_greedy_rows   EXACT, under batch_invariant.
#       The mechanics tests. They fail if the ring is pre-written (a chunk position reads a FUTURE
#       token, a different POSITION, which no reassociation can mask), if a compressed row is one slot
#       long or short, or if the indexer picks its top-k off the wrong end_pos.
#   drift_is_reassociation_sized              REAL arithmetic, on purpose: the size of what batching
#       costs when the reassociation is left in, bounded so a structural regression still stands out.

# (prompt, s) geometries. Every one is here for a reason cpu_args() makes reachable: window_size 16,
# compress ratios 4 and 8, index_topk 8.
FAST_CASES = [
    (2, 2),      # before ANY compression: ratio-8 rows are empty, ratio-4 crosses its first boundary
    (5, 4),      # still inside the first window: the un-wrapped `[0..p]` index row, padded
    (16, 2),     # crosses no compression boundary at all — the control
    (15, 2),     # the window-wrap seam: p=15 is the last un-wrapped row, p=16 the first wrapped one
    (18, 2),     # a ratio-4 boundary lands INSIDE the chunk
    (13, 4),     # ratio-4 and ratio-8 boundaries both inside, plus the wrap
    (20, 4),     # ratio-8 boundary inside, ring already full
    (14, 6),     # TWO ratio-4 boundaries in one chunk -> three indexer read lengths
    (37, 6),     # deep in the ring: every chunk position overwrites a slot an earlier one still reads
    (12, 8),     # wider than a g=5 round sends — the mechanics must hold there too
]
# Straddle EVERY window multiple, at every real chunk width. win=16, so 15/31 are the last un-wrapped
# ring rows (get_window_topk_idxs's `elif` branch) and 16/32 the first wrapped ones (its `if` branch)
# -- the seam a chunk crosses when start_pos ≡ -1 mod window. CI caught the original cut here: a chunk
# at 15 spans 15,16 and 16 % 16 wraps the ring slot to 0. Permanent coverage, per the fix.
WRAP_CASES = [(p, s) for p in (15, 16, 31, 32) for s in (2, 4, 6)]
# The mechanics tests run over both — the compression-boundary geometries above and every wrap seam.
MECH_CASES = FAST_CASES + [c for c in WRAP_CASES if c not in FAST_CASES]


@contextlib.contextmanager
def batch_invariant():
    """Force every contraction to a broadcast-multiply-then-fp32-sum, so no BLAS/einsum kernel runs
    and reduction order is fixed per output element -- identical whether a pass carries 1 position or
    s. This is what makes "the chunk equals the loop, bit for bit" a well-posed claim: it removes the
    ONE thing that legitimately differs between them (size-dependent reassociation) and nothing else,
    so anything still divergent is a mechanical error. See this section's header.

    fp32 accumulation mirrors torch's own mixed-precision matmul; the result is cast back to the first
    operand's dtype. Covers the four einsum signatures the V4 forward uses (sparse_attn's two, the
    grouped o-projection, the indexer score) and every Linear."""
    real_linear, real_einsum = torch.nn.functional.linear, torch.einsum

    def linear(x, weight, bias=None):
        y = (x.unsqueeze(-2).float() * weight.float()).sum(-1).to(x.dtype)
        return y if bias is None else y + bias

    patterns = {
        "bshd,bstd->bsht": lambda q, g: (q.unsqueeze(-2).float() * g.unsqueeze(-3).float()).sum(-1),
        "bsht,bstd->bshd": lambda p, g: (p.unsqueeze(-1).float() * g.unsqueeze(-3).float()).sum(-2),
        "bshd,btd->bsht": lambda q, kv: (q.unsqueeze(-2).float() * kv[:, None, None].float()).sum(-1),
        "bsgd,grd->bsgr": lambda o, w: (o.unsqueeze(-2).float() * w.float()).sum(-1).to(o.dtype),
    }

    def einsum(eq, *ops):
        fn = patterns.get(eq.replace(" ", ""))
        return fn(*ops) if fn is not None else real_einsum(eq, *ops)

    torch.nn.functional.linear = linear
    torch.einsum = einsum
    try:
        yield
    finally:
        torch.nn.functional.linear = real_linear
        torch.einsum = real_einsum


def _capture_sparse_attn(monkeypatch):
    """Record every (topk_idxs, kv) the reference's attention kernel is called with. -> the list."""
    M = V4.ref()
    calls, real = [], M.sparse_attn

    def spy(q, kv, attn_sink, topk_idxs, softmax_scale):
        calls.append((topk_idxs.clone(), kv.clone()))
        return real(q, kv, attn_sink, topk_idxs, softmax_scale)

    monkeypatch.setattr(M, "sparse_attn", spy)
    return calls


def _meaning(row, win, pos, base=None):
    """One sparse-attention index row -> what it MEANS, independent of where the bytes live.

    An index is only ever a place: a window slot holding some absolute position, a compressed slot,
    the fast path's scratch row for a chunk position, or -1 for nothing. Resolving all of them to
    ('p', absolute position) / ('c', compressed slot) / ('-',) is what lets the fast path's rows be
    compared against the loop's AT ALL -- they are deliberately different integers (that is the whole
    mechanism) and have to be the same MEANING.

    `pos` is the position the row belongs to, and the ring is read as it stood when the row was used:
    slot v last held `pos - ((pos - v) % win)`. `base` is the fast path's scratch offset, whose row i
    holds chunk position `start` + i."""
    out = []
    for v in (int(x) for x in row):
        if v < 0:
            out.append(("-",))
        elif base is not None and v >= base[0]:
            out.append(("p", base[1] + (v - base[0])))
        elif v < win:
            out.append(("p", pos - ((pos - v) % win)))
        else:
            out.append(("c", v - win))
    return out


def _drive(st, prompt_ids, chunk_ids, hp, hc, start_pos):
    """Prefill a layer-range stage with `hp`, then feed it `hc` at `start_pos`. -> the chunk's output.

    Payloads rather than token ids because these tests isolate ONE layer at a time, and a middle
    layer has no embedding: what matters is that both stages see the SAME [b, s, hc_mult, dim] in."""
    st.forward(hp, prompt_ids, 0)
    return st.forward(hc, chunk_ids, start_pos)


def _payloads(args, prompt, s, seed, b=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, args.vocab_size, (b, prompt + s), generator=g)
    hp = torch.randn(b, prompt, args.hc_mult, args.dim, generator=g).bfloat16() * 0.3
    hc = torch.randn(b, s, args.hc_mult, args.dim, generator=g).bfloat16() * 0.3
    return ids, hp, hc


def _state(st):
    """Every per-stage buffer of a one-layer stage, as (name, tensor). The fast stage's kv_cache is
    truncated to the reference's length -- the scratch region past it is the fast path's own."""
    a, out = st.layers[0].attn, []
    real = st.args.window_size + (st.args.max_seq_len // a.compress_ratio if a.compress_ratio else 0)
    out.append(("kv_cache", a.kv_cache[:, :real]))
    if a.compress_ratio:
        out += [("kv_state", a.compressor.kv_state), ("score_state", a.compressor.score_state)]
        if a.indexer is not None:
            out += [("idx.kv_cache", a.indexer.kv_cache),
                    ("idx.kv_state", a.indexer.compressor.kv_state),
                    ("idx.score_state", a.indexer.compressor.score_state)]
    return out


@pytest.mark.parametrize("prompt,s", MECH_CASES)
@pytest.mark.parametrize("li", (0, 2, 3))          # sliding-window / ratio-4 + indexer / ratio-8
def test_fast_verify_attends_exactly_what_the_loop_attends(args, li, prompt, s):
    """THE headline. Every chunk position attends the same places, in the same order, as the loop.

    Compared as MEANINGS, not as integers, because the integers are deliberately different: the fast
    path's rows point at the scratch copies of the chunk's own kv, precisely so that the window ring
    can go on holding the PRE-CHUNK tokens while the whole chunk is answered in one call. That is the
    part with no margin. Chunk position p+i writes ring slot (p+i) % win, which is exactly the slot
    holding the oldest token of position p's own window -- so a pass that writes the ring before it
    attends (the obvious implementation, and the one the reference's own decode branch does per
    token) answers p's oldest window slots with tokens from p's FUTURE. It runs, and it is wrong, and
    nothing but this comparison notices. The WRAP_CASES (15,16,31,32) are where it bites: a chunk at
    15 spans 15,16 and 16 % 16 wraps back to slot 0. (37, 6) is where every position does it.

    RUNS UNDER batch_invariant so the byte check below is a statement about POSITION, not float order.
    Both stages are ONE layer and handed the same payload, so this is the chunk mechanics alone.

    The one licensed difference is the right-hand -1 padding: a chunk's positions have different
    compressed read lengths and different indexer top-k widths, and one [b, s, k] tensor has to hold
    all of them, so the short rows are padded with the kernel's own "no position". Asserted to be
    exactly that -- trailing, and nothing but."""
    monkey = pytest.MonkeyPatch()
    try:
        calls = _capture_sparse_attn(monkey)
        oracle = REFCPU.build_oracle(args, SEED)
        fast = stage_from_oracle(oracle, args, li, li + 1, head=False, tail=False, fast=True)
        slow = stage_from_oracle(oracle, args, li, li + 1, head=False, tail=False)
        ids, hp, hc = _payloads(args, prompt, s, seed=prompt * 31 + s)

        with batch_invariant():
            fast.forward(hp, ids[:, :prompt], 0)              # prefill attends too — not under test
            calls.clear()
            fast.forward(hc, ids[:, prompt:], prompt)
            assert len(calls) == 1, "the fast path must answer the whole chunk in ONE attention call"
            f_rows, f_kv = calls[0]
            slow.forward(hp, ids[:, :prompt], 0)
            calls.clear()
            slow.forward(hc, ids[:, prompt:], prompt)
        assert len(calls) == s, "the reference path is one call per position — the control"
        win = args.window_size
        base = (f_kv.size(1) - V4.V4_FAST_VERIFY_MAX, prompt)

        for j in range(s):
            got = _meaning(f_rows[0, j].tolist(), win, prompt + j, base)
            want = _meaning(calls[j][0][0, 0].tolist(), win, prompt + j)
            assert got[:len(want)] == want, (
                f"chunk position {prompt + j} attends elsewhere than the loop does\n"
                f"  chunked {got}\n  loop    {want}")
            assert all(e == ("-",) for e in got[len(want):]), \
                f"the chunked row's extra entries must be -1 padding, got {got[len(want):]}"
            # ...and the places must still HOLD what the loop found in them. Comparing meanings alone
            # would pass an implementation that writes the ring before it attends: the row still says
            # "slot 3, which is position 35", and slot 3 now holds position 39. Same claim, different
            # POSITION -- so the gathered operands differ under batch_invariant, where the only thing
            # that could make two equal-position kv differ (reassociation) has been removed.
            s_kv, s_row = calls[j][1], calls[j][0][0, 0].tolist()
            for k, (v_f, v_s) in enumerate(zip(f_rows[0, j].tolist(), s_row)):
                if int(v_s) < 0:
                    continue
                assert torch.equal(f_kv[0, int(v_f)], s_kv[0, int(v_s)]), (
                    f"chunk position {prompt + j} entry {k} points at {got[k]} in both, but the kv "
                    f"there is not the kv the loop read — the ring was overwritten before it was read")
    finally:
        monkey.undo()


@pytest.mark.parametrize("prompt,s", MECH_CASES)
@pytest.mark.parametrize("li", (0, 2, 3))
def test_fast_verify_leaves_the_same_state(args, li, prompt, s):
    """Bit-identical window ring, compressed region and compressor accumulators after the chunk.

    The other half of the mechanics: the chunk must not only READ what sequential decode reads, it
    must LEAVE what sequential decode leaves -- the next round's window, the compressed slots this
    chunk's boundaries emitted, and both fp32 accumulators (the layer's and, on a ratio-4 layer, the
    indexer's). torch.equal under batch_invariant, per layer, on the same input: the compressor is
    driven one position at a time through the reference's OWN decode branch so that even the emitted
    compressed slot is the loop's, and the harness removes the only other source of difference.

    A rejected chunk's rollback rests on this (`Stage._snapshot` restores the window and the
    accumulators and argues the compressed region is always rewritten before it is read), so a fast
    chunk that left a subtly different accumulator would make every rewind after it wrong. The
    WRAP_CASES pin the ring commit specifically: a wrong slot at start_pos ≡ -1 mod window shows up
    here as a window row that no longer matches the loop's."""
    oracle = REFCPU.build_oracle(args, SEED)
    fast = stage_from_oracle(oracle, args, li, li + 1, head=False, tail=False, fast=True)
    slow = stage_from_oracle(oracle, args, li, li + 1, head=False, tail=False)
    ids, hp, hc = _payloads(args, prompt, s, seed=prompt * 31 + s)

    with batch_invariant():
        _drive(fast, ids[:, :prompt], ids[:, prompt:], hp, hc, prompt)
        _drive(slow, ids[:, :prompt], ids[:, prompt:], hp, hc, prompt)
    for (name, got), (_, want) in zip(_state(fast), _state(slow)):
        assert torch.equal(got, want), f"layer {li}: {name} diverged over a {s}-token chunk at {prompt}"
    assert fast._pos == slow._pos == prompt + s


@pytest.mark.parametrize("prompt,s", MECH_CASES)
def test_fast_verify_rows_are_the_greedy_rows(oracle, args, prompt, s):
    """Whole 8-layer stage: the verify chunk's per-position greedy tokens are the loop's. Losslessness.

    This is the bar the RING is settled against -- `_serve_tail` answers a spec chunk with
    `[int(r.argmax()) for r in rows]`, one row per position at the same GEMM shape greedy decode uses
    (v4_pipe.py:698), and a draft is accepted exactly when that token matches. So this compares the
    thing acceptance is decided on, over a chunk that has been through all eight layers -- including
    every WRAP seam and every s up to the scratch width.

    UNDER batch_invariant, this is exact and it is the ALGORITHM's losslessness: chunked verify picks
    the same tokens sequential decode would. On REAL arithmetic the eight layers' reassociation can
    flip a near-tie (the toy's tiny dims make ties common; measured 90/90 at s <= 6 on one build, 9/18
    at s = 8, and CI's build flipped one at s = 2) -- which is the documented, off-by-default cost of
    batching, and is what test_fast_verify_drift_is_reassociation_sized bounds on real arithmetic. At
    V4's real shape near-ties are astronomically rarer and the GPU's sparse_attn is itself
    batch-invariant, so real-ring losslessness is far better than this toy's real-arithmetic run.

    The three decode steps after the chunk are compared too: a chunk that committed a different
    window or a different compressed slot would not necessarily move THIS round's tokens, but it
    moves the next ones."""
    fast = stage_from_oracle(oracle, args, 0, args.n_layers, fast=True)
    slow = stage_from_oracle(REFCPU.build_oracle(args, SEED), args, 0, args.n_layers)
    ids = _ids(prompt + s, args, seed=prompt * 7 + s)

    with batch_invariant():
        rows = []
        for st in (fast, slow):
            st.forward(st.embed(ids[:, :prompt]), ids[:, :prompt], 0)
            h = st.forward(st.embed(ids[:, prompt:]), ids[:, prompt:], prompt)
            rows.append([st.logits_all(h[:, j:j + 1], full_logits=False) for j in range(s)])
        assert [int(r.argmax()) for r in rows[0]] == [int(r.argmax()) for r in rows[1]], \
            f"a {s}-token chunk at {prompt} verified different tokens than sequential decode"

        tok = rows[1][-1].argmax(dim=-1).unsqueeze(1)
        for i in range(3):
            nxt = []
            for st in (fast, slow):
                h = st.forward(st.embed(tok), tok, prompt + s + i)
                nxt.append(st.logits_all(h, full_logits=False))
            assert int(nxt[0].argmax()) == int(nxt[1].argmax()), f"streams parted {i} steps after chunk"
            tok = nxt[1].argmax(dim=-1).unsqueeze(1)


@pytest.mark.parametrize("prompt,s", FAST_CASES)
def test_fast_verify_drift_is_reassociation_sized(oracle, args, prompt, s):
    """What batching costs, measured rather than asserted away: |Δh| after a whole 8-layer chunk.

    bf16 carries 8 mantissa bits, so one ulp of the payload's largest element is |h|max / 256. A
    batched pass differs from the loop by a handful of those, amplified layer over layer: measured at
    <= 24 for s <= 6 and <= 53 at s = 8. They enter wherever torch swaps kernel for size -- MKL's
    M = 1 sgemm against its M = s one, and bf16 `rsqrt` taking its vectorized path once a tensor is
    big enough -- and no implementation of a batched pass avoids them.

    A TRIPWIRE, NOT A PRECISION CLAIM. The bound is two and a half times the worst measured value, so
    it passes reassociation and fails structure: a mechanical regression (a stale window slot, a
    compressed row one short) moves the payload by a factor, not by ulps, and lands orders of
    magnitude past this."""
    fast = stage_from_oracle(oracle, args, 0, args.n_layers, fast=True)
    slow = stage_from_oracle(REFCPU.build_oracle(args, SEED), args, 0, args.n_layers)
    ids = _ids(prompt + s, args, seed=prompt * 7 + s)
    out = []
    for st in (fast, slow):
        st.forward(st.embed(ids[:, :prompt]), ids[:, :prompt], 0)
        out.append(st.forward(st.embed(ids[:, prompt:]), ids[:, prompt:], prompt))
    ulp = out[1].abs().float().max().item() / 256.0                       # one bf16 ulp at full scale
    drift = (out[0].float() - out[1].float()).abs().max().item()
    assert drift <= 128 * ulp, f"chunk drift {drift / ulp:.1f} ulps is structural, not reassociation"


def test_fast_verify_is_off_by_default(args):
    """OFF unless asked for, and OFF means the reference's own `Block` with the reference's buffers.

    Not a style point. The whole argument for the fast path is that the per-token loop stays the
    thing everything else is graded against, so a stage nobody opted in must be structurally unable
    to take the branch -- not merely disinclined to."""
    M = V4.ref()
    plain = V4.Stage(0, 2, args, device="cpu")
    assert type(plain.layers[0]) is M.Block, "the default stage must be the reference's own Block"
    assert not plain._chunk_ok(4) and plain._chunk_cap == 0
    win = args.window_size
    assert plain.layers[0].attn.kv_cache.size(1) == win, "no scratch on a stage that cannot use it"

    fast = V4.Stage(0, 2, args, device="cpu", fast_verify=True)
    assert type(fast.layers[0]) is not M.Block and isinstance(fast.layers[0], M.Block)
    assert fast.layers[0].attn.kv_cache.size(1) == win + V4.V4_FAST_VERIFY_MAX
    assert fast._chunk_ok(4) and not fast._chunk_ok(1), "s=1 is the reference path, not a chunk"
    assert not fast._chunk_ok(V4.V4_FAST_VERIFY_MAX + 1), "an oversized chunk falls back to the loop"


def test_fast_verify_falls_back_over_the_scratch_width(oracle, args):
    """A chunk wider than the scratch region is answered by the LOOP, bit-identically, not refused.

    The cap is a VRAM decision (scratch rows per layer), not a contract, so meeting a wider chunk has
    to be slow rather than fatal -- a coordinator that raises its speculation depth past the stage's
    cap gets correct tokens at the old speed and a knob to turn, instead of a dead ring. Bit-identical
    because it IS the loop: same code path, same shapes, same arithmetic."""
    s = V4.V4_FAST_VERIFY_MAX + 2
    fast = stage_from_oracle(oracle, args, 0, args.n_layers, fast=True)
    slow = stage_from_oracle(REFCPU.build_oracle(args, SEED), args, 0, args.n_layers)
    ids = _ids(PROMPT + s, args, seed=11)
    out = []
    for st in (fast, slow):
        st.forward(st.embed(ids[:, :PROMPT]), ids[:, :PROMPT], 0)
        out.append(st.forward(st.embed(ids[:, PROMPT:]), ids[:, PROMPT:], PROMPT))
    assert torch.equal(out[0], out[1]), "the fallback must be the per-token loop itself"


@pytest.mark.parametrize("k", (0, 2, 4))
def test_fast_verify_rolls_back(oracle, args, k):
    """A rejected fast chunk rewinds like any other: same checkpoint contract, same committed stream.

    `_spec_ckpt` is taken in `forward` BEFORE the branch, so the fast path inherits it untouched --
    this is the test that keeps it that way. The replay deliberately stays on the per-token loop
    (correctness over speed on a path that only runs when speculation already lost), and what has to
    hold afterwards is that the stage answers as if the rejected tail had never been fed: compared
    against a stage that was fed the committed prefix one token at a time and never speculated."""
    s = 4
    fast = stage_from_oracle(oracle, args, 0, args.n_layers, fast=True)
    fast._spec = True
    ids = _ids(PROMPT + s, args, seed=13)
    fast.forward(fast.embed(ids[:, :PROMPT]), ids[:, :PROMPT], 0)
    fast.forward(fast.embed(ids[:, PROMPT:]), ids[:, PROMPT:], PROMPT)
    assert fast._spec_ckpt["start_pos"] == PROMPT and fast._spec_ckpt["s"] == s
    fast._seek(PROMPT + k)
    assert fast._pos == PROMPT + k

    want = stage_from_oracle(REFCPU.build_oracle(args, SEED), args, 0, args.n_layers)
    want.forward(want.embed(ids[:, :PROMPT]), ids[:, :PROMPT], 0)
    for j in range(k):
        want.forward(want.embed(ids[:, PROMPT + j:PROMPT + j + 1]), ids[:, PROMPT + j:PROMPT + j + 1],
                     PROMPT + j)
    for i, ((h_got, l_got), (h_want, l_want)) in enumerate(
            zip(stream(fast, _ids(1, args, seed=5), PROMPT + k, STEPS),
                stream(want, _ids(1, args, seed=5), PROMPT + k, STEPS))):
        assert int(l_got.argmax()) == int(l_want.argmax()), \
            f"token stream diverged {i} steps after rewinding a fast chunk to +{k}"
