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

def stage_from_oracle(oracle, args, lo, hi, *, head=None, tail=None, dspark=False):
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
    st = V4.Stage(lo, hi, args, head=head, tail=tail, dspark=dspark, device="cpu")
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
    """A gap is a bug; a rewind is step 5. Both refuse loudly rather than answering off stale state."""
    st = stage_from_oracle(oracle, args, 0, args.n_layers)
    ids = _ids(5, args)
    run([st], ids, 0)
    tok = _ids(1, args, seed=3)
    with pytest.raises(RuntimeError, match="is ahead of the 5 tokens"):
        st.forward(st.embed(tok), tok, 7)
    with pytest.raises(NotImplementedError, match="step 5"):
        st.forward(st.embed(tok), tok, 3)
    assert st._pos == 5, "a refused seek must not move the stage"


def test_forward_refuses_a_payload_without_ids(oracle, args):
    st = stage_from_oracle(oracle, args, 0, args.n_layers)
    ids = _ids(5, args)
    with pytest.raises(RuntimeError, match="needs the token ids"):
        st.forward(st.embed(ids), None, 0)


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
