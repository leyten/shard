"""The routed MoE inside the whole-layer graph: the premise audit, and the ways it could go stale.

A CUDA graph replays a FIXED sequence of kernels over FIXED addresses. Capturing a region whose
program depends on data therefore does not fail — it succeeds and serves the CAPTURE step's answer
forever, behind plausible tokens and valid receipts. The routed MoE is the worst possible candidate
for that: the vendored dispatch reads `bincount(...).tolist()` and loops over the experts it found,
and `v4_moe_decode`'s s==1 path still opens with `indices[0].tolist()`. Capture either and every token
after the first runs the first token's experts.

So this file does not test that the capture "works". It tests the PREMISE that makes capturing legal,
mechanically, on CPU, and it tests the mutants that would break it:

  test_the_moe_program_is_identical_at_every_routing        THE PREMISE. The whole capture-safe block
      — attention core, hyper-connections and the real routed MoE — is recorded as an aten TAPE (op
      sequence, every tensor shape and dtype, every non-tensor argument) at six DIFFERENT routings, on
      a score-routed layer and on a hash-routed layer whose ids REPEAT, at a compress step and a
      non-compress step. Every tape must be IDENTICAL and no device drain may appear. Identical
      program + identical shapes is exactly what a replay re-executes, so a graph captured at one
      routing is the program every other routing needs.
  test_the_audit_rejects_the_dispatch_it_exists_to_reject   THE NEGATIVE CONTROL. The same audit over
      `v4_moe_decode.decode_forward` — the path a stage takes with V4_MOE_GROUPED unset — must FAIL:
      it drains the device with `.tolist()` and its program changes with the routing. An audit that
      cannot fail proves nothing, and this is the exact configuration a mis-set flag would leave.
  test_routing_reaches_the_block_only_through_the_static_buffers   THE REPLAY CONTRACT. Driving the
      capture target `_block()` off the static buffers must equal the eager twin at that routing, and
      a different token must MOVE the answer.
  test_a_block_that_stopped_feeding_ids_serves_stale_routing       the mutant: skip the `ids_buf`
      copy and the same test goes green-looking (identical output at a new routing) — which is what
      "would catch it" means.
  test_the_static_buffer_path_is_bit_exact_to_the_reference_block  the capture target vs the vendored
      `Block.forward`, torch.equal, over routings including hash duplicates.
  test_graphed_and_eager_agree_across_a_compression_boundary       THE ROLLBACK COMPOSITION. A
      speculative rejection replays EAGER what the graphed path wrote, so the two must leave the same
      state and the same output at every position — including the compress step a rewind crosses.
  the refusal tests                                        every way `_moe_refusal` must decline, and
      each is a way a capture could be wrong rather than merely slow.

Everything above runs on CPU, on real fp4 experts through `v4_kernels_cpu`'s stand-ins, because the
claim under test is DISPATCH — which program runs, on which shapes, from which buffers — and that is
exactly what a CPU box can settle. The two CUDA tests at the bottom re-prove it against a real capture.

Run:  python3 -m pytest tests/test_v4_moe_in_graph.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase0"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
REFCPU = pytest.importorskip("v4_ref_cpu")
V4 = pytest.importorskip("v4_stage")
WL = pytest.importorskip("v4_whole_layer_graph")
GROUPED = pytest.importorskip("v4_moe_grouped")
DECODE = pytest.importorskip("v4_moe_decode")
LEVERS = pytest.importorskip("v4_levers")
KERNELS = pytest.importorskip("v4_kernels_cpu")

from torch.overrides import TorchFunctionMode                       # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode          # noqa: E402

SEED = 5
PROMPT = 20
# `_fp4_args()` is cpu_args with n_hash_layers=3 against compress_ratios (0, 0, 4, 8, 4, 8, 4, 0), the
# shipped model's shape at toy scale. That gives one layer of every combination the audit must hold
# for, and `LAYER_KINDS` is used verbatim as the parametrisation so a layer can never be silently
# dropped from the sweep. `pos` picks a compress step wherever there is one: (pos+1) % ratio == 0.
HASH_WINDOW_LAYER = 0       # hash-routed (ids can REPEAT), no compressor at all
HASH_INDEX_LAYER = 2        # hash-routed AND ratio 4 -> Indexer + Compressor. The shipped hard case.
SCORE_COMPRESS_LAYER = 3    # score-routed, ratio 8 -> a plain Compressor, no Indexer
SCORE_INDEX_LAYER = 4       # score-routed, ratio 4 -> Indexer + Compressor
# (layer, position). 27 is a compress step for ratio 4, 31 for ratio 8; 25 is neither.
LAYER_KINDS = [(HASH_WINDOW_LAYER, 25), (HASH_INDEX_LAYER, 25), (HASH_INDEX_LAYER, 27),
               (SCORE_COMPRESS_LAYER, 25), (SCORE_COMPRESS_LAYER, 31),
               (SCORE_INDEX_LAYER, 25), (SCORE_INDEX_LAYER, 27)]

requires_cpu_kernels = pytest.mark.skipif(
    KERNELS.backend() != "cpu",
    reason=f"drives fp4 CPU stages; this process bound the {KERNELS.backend()} kernels")


# ── the audit instruments ────────────────────────────────────────────────────────────────────────

# Every torch API that moves a value from the device to the host. `.tolist()` and `.item()` are the
# two the MoE dispatch actually used; the rest are here because the audit has to be TOTAL — a future
# edit that reaches for `.cpu()` or a bare `if tensor:` must fail this file, not be discovered on a
# ring. They are caught at the TorchFunction level rather than the dispatch level on purpose: on a CPU
# tensor `.tolist()` reads memory directly and emits no aten op at all, so a dispatch-only audit would
# pass the very thing this test exists to reject.
DRAIN_OPS = frozenset({"tolist", "item", "__bool__", "__int__", "__float__", "__index__",
                       "cpu", "numpy", "nonzero", "bincount", "unique", "unique_consecutive",
                       "masked_select", "local_scalar_dense", "argwhere"})


class Drains(TorchFunctionMode):
    """Records every device->host drain attempted inside the `with`. See DRAIN_OPS."""

    def __init__(self):
        self.hits = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", str(func))
        if name in DRAIN_OPS:
            self.hits.append(name)
        return func(*args, **(kwargs or {}))


class Tape(TorchDispatchMode):
    """The aten program: for every op, its name, the shape/dtype of each tensor argument, every
    non-tensor argument, and the shape/dtype of every output.

    This is the thing a CUDA graph replays. Two runs with the same tape differ ONLY in tensor
    contents, which is what replay recomputes; two runs with different tapes cannot share a graph.
    Non-tensor arguments are compared as-is because that is where a data-dependent value would have to
    land to become a baked constant — a Python-side `int(idx)` folded into a `narrow` shows up here."""

    def __init__(self):
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        self.ops.append((str(func), _sig(args), tuple(sorted((k, _sig(v)) for k, v in kwargs.items())),
                         _sig(out)))
        return out


def _sig(t):
    if isinstance(t, torch.Tensor):
        return (tuple(t.shape), str(t.dtype))
    if isinstance(t, (list, tuple)):
        return tuple(_sig(v) for v in t)
    return t


def _record(fn):
    """(aten tape, drains) for one call of `fn`."""
    drains, tape = Drains(), Tape()
    with torch.no_grad(), drains, tape:
        fn()
    return tape.ops, drains.hits


def _first_difference(tapes):
    """(index, the op name each tape had there) for the first op the tapes disagree on."""
    for i in range(min(len(t) for t in tapes)):
        if any(t[i] != tapes[0][i] for t in tapes):
            return i, [t[i][0] for t in tapes]
    return min(len(t) for t in tapes), ["<ran out>"] * len(tapes)


# ── an fp4 CPU stage with the grouped MoE bound ──────────────────────────────────────────────────

def _fill(st, args, M, seed=SEED):
    """Deterministic weights for a banked fp4 stage.

    `v4_ref_cpu.init_random` cannot do this one: `normal_` has no kernel for float4_e2m1fn_x2, and the
    routed experts have to be VALID packed fp4 + e8m0 (reinterpreted noise dequantizes to inf and
    makes every comparison vacuously equal). So the routed experts go through the reference's own
    `fp4_act_quant`, written THROUGH the bank views the layout left, and everything else is filled the
    way the oracle fills it."""
    from kernel import fp4_act_quant
    g = torch.Generator().manual_seed(seed)
    norms = {id(m.weight) for mod in st._owned_modules() for m in mod.modules()
             if isinstance(m, M.RMSNorm)}
    with torch.no_grad():
        for mod in st._owned_modules():
            for _name, p in mod.named_parameters():
                if p.dtype in (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu):
                    continue                                    # written below, quantized
                if p.dtype == torch.int32:                      # Gate.tid2eid: an expert-id table
                    p.random_(0, args.n_routed_experts, generator=g)
                elif id(p) in norms:
                    p.normal_(1.0, 0.02, generator=g)
                else:
                    p.normal_(0.0, 0.02, generator=g)
        for n in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            getattr(st, n).data.normal_(0.0, 0.02, generator=g)
        for L in st.layers:
            for e in L.ffn.experts:
                if e is None:
                    continue
                for k in ("w1", "w2", "w3"):
                    lin = getattr(e, k)
                    w, s = fp4_act_quant(torch.randn(lin.out_features, lin.in_features,
                                                     generator=g, dtype=torch.bfloat16),
                                         M.fp4_block_size)
                    lin.weight.data.copy_(w)
                    lin.scale.data.copy_(s)
    return st


def _fp4_args():
    """cpu_args with fp4 routed experts, widened so the CPU kernels can run them.

    `moe_inter_dim` has to be a multiple of `act_quant`'s 128-wide block, because an Expert quantizes
    its INTERMEDIATE on the way into w2 — the default 64 builds fine and cannot take a step.

    `n_hash_layers=3` rather than cpu_args' 2, to reproduce the SHIPPED layer shape: config.json has
    n_hash_layers 3 against compress_ratios (0, 0, 4, 128, ...), so layer 2 of the real model is
    hash-routed AND carries an Indexer. At n_hash_layers=2 that combination does not exist on this toy
    and the hardest case — routing that depends on the TOKEN, in a layer with a compress branch —
    would go untested."""
    return REFCPU.cpu_args(expert_dtype="fp4", moe_inter_dim=128, n_hash_layers=3)


def _build_stage(args, M, seed=SEED):
    st = V4.Stage(0, args.n_layers, args, head=True, tail=True, device="cpu")
    assert st._moe_banked == args.n_layers, "every fp4 layer must come out of __init__ banked"
    return _fill(st, args, M, seed)


@pytest.fixture(scope="module")
def fp4stage():
    """A prefilled fp4 CPU stage with `grouped_forward` bound — the ONLY chain the capture is legal on.

    `install()` refuses off CUDA (the tilelang kernel is CUDA-only), so the binding is done by hand the
    way tests/test_v4_moe_grouped.py's `_runnable` does it; `grouped_fp4_gemm` off CUDA is a per-slot
    loop over the same `kernel.fp4_gemm`, so the DISPATCH under test here is the shipped one. The
    marker is what `v4_levers.moe_chain` reads to identify the live link, and `install()` is what
    normally sets it.

    Module-scoped, and every process-wide thing it touches is restored: `MoE.forward` is a class
    attribute and the rest of the v4 suite runs in this same interpreter."""
    if KERNELS.backend() != "cpu":
        pytest.skip("drives fp4 CPU stages through the CPU stand-in kernels")
    args = _fp4_args()
    M = REFCPU.load_ref()
    saved = (GROUPED.V4_MOE_GROUPED, M.MoE.forward,
             GROUPED._MOD, GROUPED._WORLD_SIZE, GROUPED._REF_FORWARD)
    GROUPED.V4_MOE_GROUPED = True                      # read by Stage.__init__'s bank_layout
    try:
        st = _build_stage(args, M)
        GROUPED._MOD, GROUPED._WORLD_SIZE, GROUPED._REF_FORWARD = M, 1, M.MoE.forward
        GROUPED.grouped_forward._v4_grouped = True
        M.MoE.forward = GROUPED.grouped_forward
        torch.manual_seed(SEED)
        ids = torch.randint(0, args.vocab_size, (1, PROMPT))
        with torch.no_grad():
            st.forward(st.embed(ids), ids, 0)
        yield st, args, M
    finally:
        (GROUPED.V4_MOE_GROUPED, M.MoE.forward,
         GROUPED._MOD, GROUPED._WORLD_SIZE, GROUPED._REF_FORWARD) = saved


def _block(bg, compress, bucket):
    """The capture target, run under no_grad -- which is what the serve path and a capture both do."""
    with torch.no_grad():
        return bg._block(compress, bucket)


def _eager(bg, h, tok, pos):
    with torch.no_grad():
        return bg._eager(h, tok, pos)


def _snap(L):
    return [b.clone() for b in WL._layer_state(L)]


def _restore(L, snap):
    for b, s in zip(WL._layer_state(L), snap):
        b.copy_(s)


def _h(args, scale=0.5):
    return torch.randn(1, 1, args.hc_mult, args.dim, dtype=torch.bfloat16) * scale


def _routing_of(bg, L, h, tok, pos, bucket, compress):
    """The expert ids this layer's gate REALLY picked for (h, tok) — read off the gate, not guessed.

    Hooked rather than recomputed: a score-routed gate sees the ffn_norm of the hc_pre mix, not `h`, so
    a re-derivation would be a second implementation that could agree with itself while disagreeing
    with the block. Run OUTSIDE the tape, in its own snapshot/restore pass, so the measurement does not
    appear in the program it is used to describe."""
    got = []
    hook = L.ffn.gate.register_forward_hook(lambda _m, _i, o: got.append(o[1][0].clone()))
    snap = _snap(L)
    try:
        bg._feed(h, tok, pos, bucket)
        _block(bg, compress, bucket)
    finally:
        hook.remove()
        _restore(L, snap)
    assert len(got) == 1, f"the gate fired {len(got)} times in one block"
    return got[0].tolist()


# ── THE PREMISE ──────────────────────────────────────────────────────────────────────────────────

@requires_cpu_kernels
@pytest.mark.parametrize("layer_id,pos", LAYER_KINDS)
def test_the_moe_program_is_identical_at_every_routing(fp4stage, layer_id, pos):
    """THE PREMISE, audited rather than argued: the capture target is the SAME PROGRAM at any routing.

    Records the aten tape of `WholeBlockGraphs._block` — the exact function `torch.cuda.graph` wraps —
    at six different (hidden state, token) draws, and requires every op, every shape, every dtype and
    every non-tensor argument to match, with a `TorchFunctionMode` watching for device drains. That is
    the whole correctness argument for capturing the MoE: a replay re-executes this program, so if the
    program does not depend on the routing, a graph captured at one routing serves every other.

    Both gates are covered because they fail differently. A score-routed layer picks its experts by
    `topk`, which cannot repeat, so its risk is only a data-dependent host branch. A HASH-routed layer
    reads `tid2eid[input_ids]`, whose ids CAN repeat — and the reference's duplicate semantics are a
    last-write-wins drop that `v4_moe_grouped` reproduces with a device-side `[G, G]` compare. If that
    compare were ever done on the host, the program would change with the number of duplicates and
    this test is what says so. Compress steps are in the sweep too (`pos` 27 for ratio 4, 31 for
    ratio 8), so the Compressor's write path is inside the taped region.

    The FIRST call is discarded: torch memoizes (kernel caches, lazily built bucket buffers) and a
    capture is preceded by three warm-up runs for exactly that reason (`_warm_and_capture`)."""
    st, args, _M = fp4stage
    L = st.layers[layer_id]
    bg = WL.WholeBlockGraphs(L, st, moe_mode="graph")
    bucket, compress = bg._plan(pos)
    assert compress == ((pos + 1) % L.attn.compress_ratio == 0 if L.attn.compress_ratio else False)
    torch.manual_seed(SEED + layer_id + pos)
    draws = [(_h(args), torch.randint(0, args.vocab_size, (1, 1))) for _ in range(6)]

    tapes, drains = [], []
    for i, (h, tok) in enumerate([draws[0]] + draws):          # [0] again = the discarded warm-up
        snap = _snap(L)
        bg._feed(h, tok, pos, bucket)
        ops, hits = _record(lambda: bg._block(compress, bucket))
        _restore(L, snap)
        if i:
            tapes.append(ops)
            drains.append(hits)
    routes = [_routing_of(bg, L, h, tok, pos, bucket, compress) for h, tok in draws]

    assert not any(drains), (
        f"layer {layer_id}: the capture target drained the device {sorted(set(sum(drains, [])))} — a "
        f"host sync cannot be captured, and a graph taken over one replays the capture step's answer")
    assert len({len(t) for t in tapes}) == 1, (
        f"layer {layer_id}: the aten program LENGTH depends on the routing "
        f"({[len(t) for t in tapes]} ops for routings {routes}) — no single graph can serve them")
    if any(t != tapes[0] for t in tapes):
        i, names = _first_difference(tapes)
        pytest.fail(f"layer {layer_id}: the aten program depends on the routing. First difference at "
                    f"op {i}: {names} (routings {routes})")
    assert len({tuple(r) for r in routes}) > 1, (
        f"layer {layer_id}: all six draws routed to {routes[0]} — the test never varied the thing it "
        f"claims to hold invariant")


@requires_cpu_kernels
@pytest.mark.parametrize("layer_id", [HASH_WINDOW_LAYER, HASH_INDEX_LAYER])
def test_a_hash_layer_with_duplicate_expert_ids_is_the_same_program(fp4stage, layer_id):
    """The duplicate case FORCED, not waited for — the one the grouped path handles with a mask.

    A random `tid2eid` repeats often and "often" is not a test. This pins one token's row to a
    triple-named expert and another's to all-distinct ids, so the two draws differ in exactly the
    property that could have been resolved on the host, and requires the same program for both. A
    `keep` mask computed with a `.tolist()` (or an expert loop that skipped the discarded slots) would
    change the op count here and be invisible everywhere else."""
    st, args, _M = fp4stage
    L = st.layers[layer_id]
    pos, k = 27, args.n_activated_experts          # a compress step on the ratio-4 layer
    bg = WL.WholeBlockGraphs(L, st, moe_mode="graph")
    bucket, compress = bg._plan(pos)
    torch.manual_seed(SEED + 3)
    h = _h(args)
    rows = {3: [1] * k,                                   # every slot the SAME expert
            4: list(range(k)),                            # all distinct
            5: [0] + [2] * (k - 1)}                       # one duplicate run
    tapes = []
    with torch.no_grad():
        for tok_id, row in rows.items():
            L.ffn.gate.tid2eid[tok_id].copy_(torch.tensor(row, dtype=torch.int32))
    for i, tok_id in enumerate([3] + list(rows)):         # [3] again = the discarded warm-up
        tok = torch.tensor([[tok_id]])
        snap = _snap(L)
        bg._feed(h, tok, pos, bucket)
        ops, hits = _record(lambda: bg._block(compress, bucket))
        _restore(L, snap)
        assert not hits, f"token {tok_id} drained the device: {hits}"
        if i:
            tapes.append(ops)
    if any(t != tapes[0] for t in tapes):
        i, names = _first_difference(tapes)
        pytest.fail(f"duplicate hash ids changed the program at op {i}: {names}")


@requires_cpu_kernels
def test_the_audit_rejects_the_dispatch_it_exists_to_reject(fp4stage, monkeypatch):
    """THE NEGATIVE CONTROL. The same audit, over the MoE a stage runs with V4_MOE_GROUPED unset.

    `v4_moe_decode.decode_forward` is the DEFAULT s==1 path, and it opens with `indices[0].tolist()`
    and a Python loop over the ids it read — then hands a duplicated routing to the reference, whose
    dispatch is a `bincount().tolist()` and a `nonzero` per active expert. Capturing that would bake
    one token's expert set into the replay. The audit above has to REJECT it, on both counts, or it is
    not evidence: a check that only ever passes is a check nobody has tested.

    This is also exactly what a mis-set flag leaves behind, which is why `_moe_refusal` inspects the
    LIVE chain instead of the env — see test_it_refuses_a_layer_whose_moe_still_syncs."""
    st, args, M = fp4stage
    L = st.layers[HASH_WINDOW_LAYER]
    pos = 25
    bg = WL.WholeBlockGraphs(L, st, moe_mode="graph")
    bucket, compress = bg._plan(pos)
    monkeypatch.setattr(M.MoE, "forward", DECODE.decode_forward)
    torch.manual_seed(SEED + 5)
    k = args.n_activated_experts
    with torch.no_grad():                                  # token 6 duplicates, token 7 does not
        L.ffn.gate.tid2eid[6].copy_(torch.tensor([2] * k, dtype=torch.int32))
        L.ffn.gate.tid2eid[7].copy_(torch.tensor(list(range(k)), dtype=torch.int32))
    h = _h(args)
    tapes, drains = [], []
    for i, tok_id in enumerate([6, 6, 7]):                 # first is the discarded warm-up
        snap = _snap(L)
        bg._feed(h, torch.tensor([[tok_id]]), pos, bucket)
        ops, hits = _record(lambda: bg._block(compress, bucket))
        _restore(L, snap)
        if i:
            tapes.append(ops)
            drains.append(hits)
    assert any(drains), (
        "the drain detector did not see decode_forward's `.tolist()` — it cannot catch the bug it "
        "exists for, so a clean audit above means nothing")
    assert tapes[0] != tapes[1], (
        "the decode dispatch produced the same aten program for a duplicated and a distinct routing — "
        "the tape comparison is not sensitive to the thing it is measuring")


# ── THE REPLAY CONTRACT ──────────────────────────────────────────────────────────────────────────

@requires_cpu_kernels
@pytest.mark.parametrize("layer_id", [HASH_WINDOW_LAYER, HASH_INDEX_LAYER, SCORE_INDEX_LAYER])
def test_routing_reaches_the_block_only_through_the_static_buffers(fp4stage, layer_id):
    """A replay reads its inputs from the SAME addresses every time, so everything that varies per
    token must arrive through those buffers and nothing may be remembered between calls.

    Drives the capture target off `_feed`ed static buffers and requires (a) the answer to equal the
    eager twin at that routing — nothing is stale — and (b) the answer to MOVE when the token moves —
    nothing is frozen. On a hash layer (b) is the whole ballgame: routing depends on the TOKEN ID and
    on nothing else in the block, so a token that reached the buffers is the only reason the output
    could change. This is the freshness gate of tests/test_v4_whole_layer.py applied to the axis the
    MoE added."""
    st, args, _M = fp4stage
    L = st.layers[layer_id]
    pos = 25
    bg = WL.WholeBlockGraphs(L, st, moe_mode="graph")
    bucket, compress = bg._plan(pos)
    torch.manual_seed(SEED + 7)
    h = _h(args)
    if L.ffn.gate.hash:
        # Pin two rows that cannot route the same way, and hold `h` FIXED: on a hash layer the token
        # id is the only thing that can move the routing, so this isolates the axis the MoE added.
        k = args.n_activated_experts
        with torch.no_grad():
            L.ffn.gate.tid2eid[8].copy_(torch.tensor([0] * k, dtype=torch.int32))
            L.ffn.gate.tid2eid[9].copy_(torch.tensor([1] * k, dtype=torch.int32))
        draws = [(h, torch.tensor([[8]])), (h, torch.tensor([[9]]))]
    else:
        # A score gate ignores the token entirely — it routes on the ACTIVATION — so that is what has
        # to move here, and the token is held fixed for the same isolating reason.
        draws = [(h, torch.tensor([[3]])), (_h(args, scale=1.5), torch.tensor([[3]]))]
    outs = []
    for hh, tok in draws:
        snap = _snap(L)
        bg._feed(hh, tok, pos, bucket)
        got = _block(bg, compress, bucket).clone()
        _restore(L, snap)
        want = _eager(bg, hh, tok, pos).clone()
        _restore(L, snap)
        assert torch.equal(got, want), (
            f"layer {layer_id}: the static-buffer path disagreed with the eager twin at token "
            f"{tok.item()} — an input did not reach the buffers, or one is stale")
        outs.append(got)
    assert not torch.equal(outs[0], outs[1]), (
        f"layer {layer_id}: a different routing produced a bit-identical block output — the routing "
        f"looks baked in rather than read per step")


@requires_cpu_kernels
def test_a_block_that_stopped_feeding_ids_serves_stale_routing(fp4stage, monkeypatch):
    """THE MUTANT THE TEST ABOVE CATCHES, made explicit.

    Break exactly one thing — `_feed` no longer copies the token into `ids_buf` — and a hash-routed
    layer replays the previous token's experts. The output stops moving, and stays a perfectly
    plausible hidden state. This is what "replays stale routing" looks like from the outside, and this
    is the assertion that would fail; it is also why `run()` refuses a `None` ids under moe_mode
    "graph" instead of quietly leaving the buffer as it found it."""
    st, args, _M = fp4stage
    L = st.layers[HASH_WINDOW_LAYER]
    pos = 25
    bg = WL.WholeBlockGraphs(L, st, moe_mode="graph")
    bucket, compress = bg._plan(pos)
    k = args.n_activated_experts
    with torch.no_grad():
        L.ffn.gate.tid2eid[10].copy_(torch.tensor([0] * k, dtype=torch.int32))
        L.ffn.gate.tid2eid[11].copy_(torch.tensor([5] * k, dtype=torch.int32))
    torch.manual_seed(SEED + 8)
    h = _h(args)

    def blind_feed(self, hh, ids, start_pos, bkt):      # the bug: ids never reach the buffer
        self.h_buf.copy_(hh)
        self.pos_buf.fill_(start_pos)
        self.win_topk_buf.copy_(WL.build_win_topk(self.R.M, self.win, start_pos))

    outs = []
    for tok_id in (10, 11):
        snap = _snap(L)
        blind_feed(bg, h, torch.tensor([[tok_id]]), pos, bucket)
        outs.append(_block(bg, compress, bucket).clone())
        _restore(L, snap)
    assert torch.equal(outs[0], outs[1]), (
        "the mutant did not actually go stale, so it does not demonstrate what the real test catches")
    # and the shipped feed does not
    outs = []
    for tok_id in (10, 11):
        snap = _snap(L)
        bg._feed(h, torch.tensor([[tok_id]]), pos, bucket)
        outs.append(_block(bg, compress, bucket).clone())
        _restore(L, snap)
    assert not torch.equal(outs[0], outs[1]), "the shipped _feed is the mutant"


@requires_cpu_kernels
@pytest.mark.parametrize("layer_id,pos", LAYER_KINDS)
def test_the_static_buffer_path_is_bit_exact_to_the_reference_block(fp4stage, layer_id, pos):
    """torch.equal against the VENDORED `Block.forward`, over several routings.

    The capture target reads its position, its token and its index lists out of static buffers instead
    of taking them as arguments; this is the statement that doing so changes no bit of the answer and
    no byte of the KV/compressor state it leaves behind. `pos` 25 and 27 straddle the ratio-4 layer's
    compression boundary, so the compress branch is compared too. (Inside pos <= 34 the Indexer keeps
    every valid column at cpu_args' index_topk=8, so the reference's undefined topk tie order — the
    documented Tier-2 divergence of tests/test_v4_whole_layer.py — cannot be in play.)"""
    st, args, _M = fp4stage
    L = st.layers[layer_id]
    bg = WL.WholeBlockGraphs(L, st, moe_mode="graph")
    bucket, compress = bg._plan(pos)
    torch.manual_seed(SEED + 13)
    for draw in range(4):
        h, tok = _h(args), torch.randint(0, args.vocab_size, (1, 1))
        snap = _snap(L)
        bg._feed(h, tok, pos, bucket)
        got = _block(bg, compress, bucket).clone()
        got_state = _snap(L)
        _restore(L, snap)
        with torch.no_grad():
            want = L(h, pos, tok).clone()
        want_state = _snap(L)
        _restore(L, snap)
        assert torch.equal(got, want), (
            f"layer {layer_id} pos {pos} draw {draw}: max|d| = "
            f"{(got.float() - want.float()).abs().max().item():.3e}")
        for a, b in zip(got_state, want_state):
            assert torch.equal(a, b), f"layer {layer_id} pos {pos} draw {draw}: a KV buffer diverged"


# ── THE ROLLBACK COMPOSITION ─────────────────────────────────────────────────────────────────────

@requires_cpu_kernels
@pytest.mark.parametrize("layer_id", [HASH_INDEX_LAYER, SCORE_INDEX_LAYER])
def test_graphed_and_eager_agree_across_a_compression_boundary(fp4stage, layer_id):
    """A speculative REJECTION replays EAGER the positions the graphed path already wrote.

    `v4_stage._run` sends a rollback replay (`_replaying`) down the vendored `Block.forward` while the
    frames it is undoing went through the graph, so the two paths have to leave IDENTICAL state at
    every position or a rewind serves a state neither path would have produced. Under moe_mode="graph"
    the routed MoE moves from one side of that seam to the other, which is why it is re-proven here.

    Walks positions 24..29 on the ratio-4 Indexer layer — 27 is a compress step ((27+1) % 4 == 0), the
    boundary `_snapshot` reasons about — running each position BOTH ways from the same state and
    requiring the output and every KV/compressor buffer to agree. Then it does the interleave itself:
    graphed for three positions, restore, eager replay of the accepted prefix, and the result must
    equal the run that was eager throughout."""
    st, args, _M = fp4stage
    L = st.layers[layer_id]
    bg = WL.WholeBlockGraphs(L, st, moe_mode="graph")
    torch.manual_seed(SEED + 17 + layer_id)
    steps = [(p, _h(args), torch.randint(0, args.vocab_size, (1, 1))) for p in range(24, 30)]
    assert any((p + 1) % L.attn.compress_ratio == 0 for p, _, _ in steps), "no compress step in range"

    base = _snap(L)
    for pos, h, tok in steps:                       # position by position, both ways, same state
        snap = _snap(L)
        bucket, compress = bg._plan(pos)
        bg._feed(h, tok, pos, bucket)
        g_out, g_state = _block(bg, compress, bucket).clone(), None
        g_state = _snap(L)
        _restore(L, snap)
        with torch.no_grad():
            e_out = L(h, pos, tok).clone()
        e_state = _snap(L)
        assert torch.equal(g_out, e_out), f"pos {pos}: graphed and eager outputs differ"
        for a, b in zip(g_state, e_state):
            assert torch.equal(a, b), f"pos {pos}: graphed and eager left different state"

    # the interleave: graphed writes, a rewind, then an EAGER replay of the accepted prefix
    _restore(L, base)
    for pos, h, tok in steps[:4]:
        bucket, compress = bg._plan(pos)
        bg._feed(h, tok, pos, bucket)
        _block(bg, compress, bucket)
    _restore(L, base)                                # the rejection: back to the checkpoint
    with torch.no_grad():
        for pos, h, tok in steps[:2]:                # replay the ACCEPTED prefix, eagerly
            L(h, pos, tok)
    mixed_out, mixed_state = None, _snap(L)
    for pos, h, tok in steps[2:4]:                   # and carry on graphed across the boundary
        bucket, compress = bg._plan(pos)
        bg._feed(h, tok, pos, bucket)
        mixed_out = _block(bg, compress, bucket).clone()
    mixed_state = _snap(L)

    _restore(L, base)
    with torch.no_grad():
        for pos, h, tok in steps[:4]:
            pure_out = L(h, pos, tok).clone()
    pure_state = _snap(L)
    assert torch.equal(mixed_out, pure_out), (
        "a rewind that replayed eagerly and then continued graphed did not match the all-eager run")
    for a, b in zip(mixed_state, pure_state):
        assert torch.equal(a, b), "the interleaved run left different KV/compressor state"


# ── THE REFUSALS ─────────────────────────────────────────────────────────────────────────────────

@requires_cpu_kernels
def test_it_clears_a_layer_whose_moe_really_is_grouped(fp4stage):
    """The gate has to SAY YES on the configuration it was built for, or nothing else here matters."""
    st, args, M = fp4stage
    for layer_id in (HASH_WINDOW_LAYER, SCORE_INDEX_LAYER):
        L = st.layers[layer_id]
        why = WL._moe_refusal(L, "cpu", st.dtype, torch.tensor([[3]]), M)
        assert why is None, f"layer {layer_id} was refused: {why}"


@requires_cpu_kernels
def test_the_probe_leaves_the_coverage_counters_where_it_found_them(fp4stage):
    """The gate's last check RUNS the MoE, and `_grouped_steps` is what `coverage()` reports.

    A probe that showed up in that number would be a lever audit reporting activity nobody asked for —
    small, but it is the same class of untruth this whole registry exists to stop."""
    st, _args, M = fp4stage
    L = st.layers[SCORE_INDEX_LAYER]
    before = (getattr(L.ffn, "_grouped_steps", 0), dict(getattr(L.ffn, "_grouped_declined", {}) or {}))
    assert WL.moe_probe(L, "cpu", st.dtype, torch.tensor([[3]])) is None
    after = (getattr(L.ffn, "_grouped_steps", 0), dict(getattr(L.ffn, "_grouped_declined", {}) or {}))
    assert before == after, f"the probe moved the counters: {before} -> {after}"


@requires_cpu_kernels
def test_it_refuses_a_layer_whose_moe_still_syncs(fp4stage, monkeypatch):
    """THE DEFAULT CONFIGURATION, and the one that must never be captured.

    With `V4_MOE_GROUPED` unset the live chain is decode -> ref, and a decode step is dispatched by
    `indices[0].tolist()`. The gate reads the LIVE chain rather than the flag, because the flag is
    exactly what has been wrong six times on this engine (phase0/v4_levers.py)."""
    st, _args, M = fp4stage
    L = st.layers[SCORE_INDEX_LAYER]
    monkeypatch.setattr(M.MoE, "forward", DECODE.decode_forward)
    why = WL._moe_refusal(L, "cpu", st.dtype, torch.tensor([[3]]), M)
    assert why and "grouped" in why, why
    assert not WL.grouped_at_one_token(M)


@requires_cpu_kernels
def test_a_multi_link_on_top_is_transparent_at_one_token(fp4stage, monkeypatch):
    """`v4_moe_multi` declines T == 1 on its first line and hands the step straight down, so a chain
    topped by it still reaches the grouped kernel and must NOT be refused. Getting this wrong would
    silently disarm the lever on every ring that runs the drafter's MoE path — which is the recipe."""
    st, _args, M = fp4stage
    MULTI = pytest.importorskip("v4_moe_multi")
    saved = MULTI._REF_FORWARD
    MULTI._REF_FORWARD = M.MoE.forward
    MULTI.multi_forward._v4_multi = True
    monkeypatch.setattr(M.MoE, "forward", MULTI.multi_forward)
    try:
        assert LEVERS.moe_chain(M)[:2] == ["multi", "grouped"], LEVERS.moe_chain(M)
        assert WL.grouped_at_one_token(M)
        L = st.layers[SCORE_INDEX_LAYER]
        assert WL._moe_refusal(L, "cpu", st.dtype, torch.tensor([[3]]), M) is None
    finally:
        MULTI._REF_FORWARD = saved


@requires_cpu_kernels
def test_it_refuses_a_layer_with_no_expert_bank(fp4stage, monkeypatch):
    """No load-time bank means the FIRST grouped step stacks one — an allocation that would happen
    INSIDE the capture, out of the graph's private memory pool, while the module keeps pointing at it
    afterwards. A later capture sharing that pool may hand those bytes to something else.

    Declining is not conservatism here: `_expert_bank`'s lazy stack is a correct eager path and a
    corrupt graphed one."""
    st, _args, M = fp4stage
    L = st.layers[SCORE_INDEX_LAYER]
    monkeypatch.delattr(L.ffn, "_grouped_bank", raising=True)
    why = WL._moe_refusal(L, "cpu", st.dtype, torch.tensor([[3]]), M)
    assert why and "bank" in why, why


@requires_cpu_kernels
def test_it_refuses_when_the_probe_step_declines(fp4stage, monkeypatch):
    """STRUCTURE IS NOT OBSERVATION. Everything can be bound and banked and the step can still land on
    the fallback — three levers in a row on this engine did exactly that. The probe is the only check
    that watches the MoE actually take the step, so it has to be the one that catches a decline no
    structural check could see."""
    st, _args, M = fp4stage
    L = st.layers[SCORE_INDEX_LAYER]
    monkeypatch.setattr(GROUPED, "_WORLD_SIZE", 1)         # pass the structural world_size check
    real = GROUPED.grouped_forward

    def declines(self, x, input_ids):                      # a bank that "would not fit", say
        return GROUPED._decline(self, "bank-would-not-fit", x, input_ids)
    declines._v4_grouped = True
    monkeypatch.setattr(GROUPED, "grouped_forward", declines)
    monkeypatch.setattr(M.MoE, "forward", declines)
    try:
        why = WL._moe_refusal(L, "cpu", st.dtype, torch.tensor([[3]]), M)
        assert why and "DECLINED" in why, why
    finally:
        GROUPED.grouped_forward = real


@requires_cpu_kernels
def test_it_refuses_world_size_above_one(fp4stage, monkeypatch):
    """The reference all-reduces the routed sum across ranks before the shared expert; the grouped
    path has no all_reduce, so it declines — and a capture must decline for the same reason rather
    than capture a fallback."""
    st, _args, M = fp4stage
    monkeypatch.setattr(GROUPED, "_WORLD_SIZE", 2)
    why = WL._moe_refusal(st.layers[SCORE_INDEX_LAYER], "cpu", st.dtype, torch.tensor([[3]]), M)
    assert why and "world_size" in why, why


@requires_cpu_kernels
def test_it_refuses_a_step_with_no_token_ids(fp4stage):
    """`ids` is optional on the eager-MoE path (the MoE gets it as an argument) and LOAD-BEARING here:
    the first `n_hash_layers` route on `tid2eid[input_ids]`, so a graph replayed without the step's
    token serves the capture step's experts. Refused at the gate, and refused again at `run()`."""
    st, _args, M = fp4stage
    L = st.layers[HASH_WINDOW_LAYER]
    why = WL._moe_refusal(L, "cpu", st.dtype, None, M)
    assert why and "token ids" in why, why
    bg = WL.WholeBlockGraphs(L, st, moe_mode="graph")
    bg.moe_in_graph = True                                  # judged clear; now feed it nothing
    with pytest.raises(RuntimeError, match="no token ids"):
        bg.run(_h(_fp4_args()), None, 25)


# ── THE LEVER ────────────────────────────────────────────────────────────────────────────────────

def test_default_off_leaves_the_eager_moe_split():
    """Default OFF has to mean the shipped whole-layer path is untouched: the routed MoE runs eager
    between two graphs, nothing asked for a capture, and the audit has nothing to judge."""
    assert WL.V4_MOE_IN_GRAPH is False
    assert "V4_MOE_IN_GRAPH" in LEVERS.LEVERS_BY_ENV


def test_the_lever_reports_armed_captured_and_refused_apart():
    """`armed` (capture is LAZY, nothing judged yet) must not read as OK, and `on/0-of-N` must read as
    a MISMATCH — that is the state a ring lands in when V4_MOE_IN_GRAPH is set and V4_MOE_GROUPED is
    not, which is the single likeliest way to run this lever and measure nothing."""
    class _BG:
        def __init__(self, requested, got):
            self.moe_requested, self.moe_in_graph = requested, got

    class _St:
        def __init__(self, bgs):
            self._block_graphs = bgs

    saved = WL.V4_MOE_IN_GRAPH
    WL.V4_MOE_IN_GRAPH = True
    try:
        ctx = LEVERS.Ctx(LEVERS.STAGE, _St([_BG(True, None), _BG(True, None)]))
        assert LEVERS._check_moe_in_graph(ctx) == ("on", "armed/2", None)
        ctx = LEVERS.Ctx(LEVERS.STAGE, _St([_BG(True, True), _BG(True, False)]))
        assert LEVERS._check_moe_in_graph(ctx) == ("on", "on/1-of-2", True)
        ctx = LEVERS.Ctx(LEVERS.STAGE, _St([_BG(True, False), _BG(True, False)]))
        assert LEVERS._check_moe_in_graph(ctx) == ("on", "on/0-of-2", False)
        # whole-layer graphs never built at all (V4_CUDA_GRAPH off, or the stage refused to capture)
        ctx = LEVERS.Ctx(LEVERS.STAGE, _St(None))
        assert LEVERS._check_moe_in_graph(ctx)[1:] == ("off", False)
        # ISLAND mode: graphs exist but none of them is a WholeBlockGraphs, so nothing asked. The
        # operator set a lever that configures nothing, and that has to read as a MISMATCH — it is the
        # `V4_MOE_IN_GRAPH=1` beside `V4_CUDA_GRAPH=1` typo, and it would otherwise measure the
        # baseline while the banner claimed the lever.
        ctx = LEVERS.Ctx(LEVERS.STAGE, _St([_BG(False, None)]))
        assert LEVERS._check_moe_in_graph(ctx)[1:] == ("off", False)
        assert WL.moe_graph_coverage(_St([_BG(True, True), _BG(True, False), _BG(False, None)])) \
            == (1, 1, 0)
    finally:
        WL.V4_MOE_IN_GRAPH = saved


def test_a_refused_layer_is_demoted_and_counted(monkeypatch):
    """A refusal must leave the layer on the PROVEN path and be visible: moe_mode back to "eager",
    `moe_in_graph` False, and `moe_requested` still True so the audit can tell "asked and refused"
    from "never asked"."""
    class _BG(WL.WholeBlockGraphs):
        def __init__(self):
            self.moe_requested, self.moe_in_graph = True, None
            self.moe_mode, self.moe = "graph", WL.real_moe
            self.L = type("L", (), {"layer_id": 3})()
            self.dev, self.dt, self.R = "cpu", torch.bfloat16, type("R", (), {"M": None})()

    monkeypatch.setattr(WL, "_moe_refusal", lambda *a: "because")
    bg = _BG()
    bg._resolve_moe_mode(torch.tensor([[1]]))
    assert (bg.moe_in_graph, bg.moe_mode, bg.moe_requested) == (False, "eager", True)


# ── CUDA: the real capture ───────────────────────────────────────────────────────────────────────

@pytest.mark.hardware
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs are a GPU-only capture")
def test_a_captured_moe_replays_a_new_routing():
    """THE GPU BAR: capture at one routing, replay at another, and demand the eager twin's answer.

    Everything above says the program does not depend on the routing; this says the real capture
    agrees. Runs on the fp4 stage this file builds, at the shipped `V4_MOE_IN_GRAPH` mode:
      * every replayed token must equal `_eager` at the same token (TIER 1, torch.equal),
      * and two tokens that route differently must give different answers (the freshness gate — a
        capture that froze the routing returns the capture step's answer forever).
    Skipped on a CPU box, which is where this file was written; the ring is where it is judged."""
    args = _fp4_args()
    M = REFCPU.load_ref()
    torch.set_default_device("cuda")
    saved = (GROUPED.V4_MOE_GROUPED, M.MoE.forward, GROUPED._MOD, GROUPED._WORLD_SIZE,
             GROUPED._REF_FORWARD, WL.V4_MOE_IN_GRAPH)
    try:
        GROUPED.V4_MOE_GROUPED = WL.V4_MOE_IN_GRAPH = True
        st = V4.Stage(0, args.n_layers, args, head=True, tail=True, device="cuda")
        _fill(st, args, M)
        assert GROUPED.install(M), "the grouped kernel must install on a CUDA box"
        torch.manual_seed(SEED)
        ids = torch.randint(0, args.vocab_size, (1, PROMPT), device="cuda")
        with torch.no_grad():
            st.forward(st.embed(ids), ids, 0)
        L = st.layers[HASH_WINDOW_LAYER]
        k = args.n_activated_experts
        with torch.no_grad():
            L.ffn.gate.tid2eid[8].copy_(torch.tensor([0] * k, dtype=torch.int32, device="cuda"))
            L.ffn.gate.tid2eid[9].copy_(torch.tensor([5] * k, dtype=torch.int32, device="cuda"))
        bg = WL.WholeBlockGraphs(L, st, moe_mode="graph")
        h = torch.randn(1, 1, args.hc_mult, args.dim, dtype=st.dtype, device="cuda") * 0.5
        outs = []
        for tok_id in (8, 9, 8):
            tok = torch.tensor([[tok_id]], device="cuda")
            snap = _snap(L)
            got = bg.run(h, tok, 25).clone()
            _restore(L, snap)
            want = bg._eager(h, tok, 25).clone()
            _restore(L, snap)
            assert torch.equal(got, want), f"TIER-1: graphed != eager twin at token {tok_id}"
            outs.append(got)
        assert bg.moe_in_graph is True, "the MoE was refused, so this proved the eager path"
        assert not torch.equal(outs[0], outs[1]), "the capture replayed one routing for both tokens"
        assert torch.equal(outs[0], outs[2]), "the same token gave two answers"
    finally:
        (GROUPED.V4_MOE_GROUPED, M.MoE.forward, GROUPED._MOD, GROUPED._WORLD_SIZE,
         GROUPED._REF_FORWARD, WL.V4_MOE_IN_GRAPH) = saved
        torch.set_default_device("cpu")
