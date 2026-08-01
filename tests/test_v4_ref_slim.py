"""The reference-compute "slim" overrides are correct, or they are a silent regression.

v4_ref_slim removes per-layer work the ring pays for nothing. Both items are default-OFF and gated
separately, so a mistake here fails nothing loudly — it just quietly stops the ring being the model.
This suite is the gate that decides whether either ships. It proves, all GPU-less against the CPU
oracle (phase0/v4_ref_cpu):

  install       env off => a no-op, the reference byte-identical; idempotent; captures the reference.
  item 1        in the select-all regime the fixed compressed index is the SAME SET the indexer's
                top-k picks (order-only difference => the sm120-class gather-order ULP, and 0 at this
                scale); past the crossover the indexer RE-ENGAGES bit-exact IF the compressor was kept
                advanced; a run that under-declares its horizon (compressor skipped) DIVERGES.
  item 2        the inplace QAT round-trip becomes a no-op (KV stays full bf16) while the real fp8/fp4
                GEMM quantization is untouched; the reference really was reducing precision; the
                decode logits move within a measured, documented bound.

The toy config (v4_ref_cpu.cpu_args) has index_topk=8, ratio=4, so the select-all crossover is
end_pos <= 35 (35//4 == 8; 36//4 == 9 is the first discriminating step) — the same regime the shipped
config is in at 2051/2052, reachable in tens of CPU decode steps instead of thousands.

Run: python3 -m pytest tests/test_v4_ref_slim.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase0"))

torch = pytest.importorskip("torch")
REFCPU = pytest.importorskip("v4_ref_cpu")
SLIM = pytest.importorskip("v4_ref_slim")

SEED = 3
WIN = None  # filled from args


@pytest.fixture(scope="module")
def mod():
    return REFCPU.load_ref()


@pytest.fixture(scope="module")
def args():
    return REFCPU.cpu_args()


@pytest.fixture(scope="module")
def installed(mod):
    """Force both items onto the class once; per-test toggles pick which is active.

    Torn down with `uninstall`, because `mod` is the process-wide reference every other v4 test
    module shares: leaving the Indexer rebound made test_v4_stage.py's chunked-verify parity test
    compare a slim loop against a reference chunk (same key SET, different gather order) and fail,
    but only when the two files ran in the same pytest process."""
    SLIM.install(mod, item1=True, item2=True)
    assert SLIM._REF_INDEXER_FORWARD is not None
    assert SLIM._GET_COMPRESS_TOPK_IDXS is not None
    assert SLIM._REF_ACT_QUANT is not None and SLIM._REF_FP4_ACT_QUANT is not None
    yield mod
    assert SLIM.uninstall(mod) == {"indexer_skip": True, "noqat": True}
    assert not getattr(mod.Indexer.forward, "_v4_ref_slim", False)
    assert not getattr(mod.act_quant, "_v4_ref_slim", False)


@pytest.fixture(autouse=True)
def _reset_toggles():
    """Every test starts from a known slim state and leaks none of its own."""
    SLIM.set_active(True)
    SLIM.set_active_noqat(True)
    SLIM.set_job_max_pos(None)
    yield
    SLIM.set_active(True)
    SLIM.set_active_noqat(True)
    SLIM.set_job_max_pos(None)


def _ratio4_indexer(args, seed):
    """A fresh ratio-4 Indexer with identical weights per seed and independent state, freqs_cis bound
    the way Attention.forward binds it (model.py:499-500)."""
    o = REFCPU.build_oracle(args, seed)
    li = next(i for i, L in enumerate(o.layers)
              if L.attn.compress_ratio == 4 and L.attn.indexer is not None)
    attn = o.layers[li].attn
    idx = attn.indexer
    idx.freqs_cis = attn.freqs_cis
    return idx


def _seq(args, n, seed=7):
    g = torch.Generator().manual_seed(seed)
    xs = [torch.randn(1, 1, args.dim, generator=g, dtype=torch.bfloat16) for _ in range(n)]
    qrs = [torch.randn(1, 1, args.q_lora_rank, generator=g, dtype=torch.bfloat16) for _ in range(n)]
    return xs, qrs


def _bf16_ulp(a, b):
    """Distance in representable bf16 steps between a and b, on a monotone bit ordering."""
    def key(t):
        i = t.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        return torch.where(i < 0, torch.tensor(-32768, dtype=torch.int32) - i, i)
    return (key(a) - key(b)).abs()


# ── install / wiring ────────────────────────────────────────────────────────────────────────────

def test_env_off_install_is_a_noop(mod):
    """The whole point of the default: with neither flag set, install rebinds nothing. Order-
    independent — it asserts the env-gated decision, not the current class state."""
    assert SLIM.V4_REF_SLIM is False and SLIM.V4_REF_SLIM_NOQAT is False
    assert SLIM.install(mod) == {"indexer_skip": False, "noqat": False}


def test_install_is_idempotent(installed):
    """A second install must not chain a slim path onto itself and lose the reference it falls back to."""
    ref_before = SLIM._REF_INDEXER_FORWARD
    aq_before = SLIM._REF_ACT_QUANT
    took = SLIM.install(installed, item1=True, item2=True)
    assert took == {"indexer_skip": False, "noqat": False}
    assert SLIM._REF_INDEXER_FORWARD is ref_before and SLIM._REF_ACT_QUANT is aq_before
    assert installed.Indexer.forward is SLIM.slim_indexer_forward
    assert installed.act_quant is SLIM.slim_act_quant


def test_toggle_off_delegates_to_reference(installed, args):
    """_ACTIVE False => the installed slim forward is the reference forward, bit-for-bit. Two fresh
    same-seed indexers so the reference's state mutation does not contaminate the comparison."""
    ia, ib = _ratio4_indexer(args, 1), _ratio4_indexer(args, 1)
    xs, qrs = _seq(args, 6)
    for sp in range(6):
        SLIM.set_active(False)
        a = installed.Indexer.forward(ia, xs[sp], qrs[sp], sp, args.window_size)
        b = SLIM._REF_INDEXER_FORWARD(ib, xs[sp], qrs[sp], sp, args.window_size)
        assert torch.equal(a, b), f"active-off must equal the reference at step {sp}"


# ── item 1: the context gate ──────────────────────────────────────────────────────────────────────

def test_context_gate_keep_vs_skip(installed, args):
    """_keep_compressor is the whole safety argument: keep advancing unless the job is GUARANTEED to
    stay select-all. Crossover for the toy config is end_pos 35 (index_topk 8 * ratio 4 boundary)."""
    idx = _ratio4_indexer(args, 1)
    SLIM.set_job_max_pos(None)
    assert SLIM._keep_compressor(idx, 4) is True         # unknown horizon -> always keep
    SLIM.set_job_max_pos(30)
    assert SLIM._keep_compressor(idx, 4) is False         # 30//4=7 <= 8 -> guaranteed short
    SLIM.set_job_max_pos(35)
    assert SLIM._keep_compressor(idx, 4) is False         # 35//4=8 <= 8 -> still short
    SLIM.set_job_max_pos(36)
    assert SLIM._keep_compressor(idx, 4) is True          # 36//4=9 > 8 -> will re-engage -> keep


def test_select_all_set_identical_order_differs(installed, args):
    """In the select-all regime the slim fixed index and the reference top-k pick the SAME SET of KV
    slots — the correctness of WHICH keys are attended is exact. They differ only in order, which is
    the sub-ULP gather-order approximation and nothing more."""
    ir, is_ = _ratio4_indexer(args, 1), _ratio4_indexer(args, 1)
    xs, qrs = _seq(args, 30)                              # end_pos up to 30 < crossover 35
    saw_reorder = False
    for sp in range(30):
        SLIM.set_active(False)
        ref = installed.Indexer.forward(ir, xs[sp], qrs[sp], sp, args.window_size)
        SLIM.set_active(True)
        SLIM.set_job_max_pos(None)
        slim = installed.Indexer.forward(is_, xs[sp], qrs[sp], sp, args.window_size)
        assert sorted(ref.flatten().tolist()) == sorted(slim.flatten().tolist()), \
            f"attended SET differs at step {sp}"
        if ref.flatten().tolist() != slim.flatten().tolist():
            saw_reorder = True
    assert saw_reorder, "expected the order to differ somewhere (that IS the ULP source)"


def test_short_ctx_logits_within_2_bf16_ulp(installed, args):
    """The full model over many short-ctx decode steps: slim vs reference logits within <=2 bf16 ULP.
    Measures 0 at this scale — the gather-order reordering washes out at small top-k widths — which is
    STRONGER than the >=1 ULP the real top-k-512 shape would show and the sm120 retile already ships."""
    def run(active):
        torch.manual_seed(0)
        prompt, steps = 16, 15                            # max end_pos 31, all select-all
        x = torch.randint(0, args.vocab_size, (1, prompt + steps))
        m = REFCPU.build_oracle(args, SEED)
        SLIM.set_active(active)
        SLIM.set_job_max_pos(None)
        out = []
        with torch.no_grad():
            m(x[:, :prompt])
            for i in range(prompt, prompt + steps):
                _, logits, _ = m(x[:, i:i + 1], i)
                out.append(logits.clone())
        return torch.cat(out, 0)

    ref, slim = run(False), run(True)
    assert _bf16_ulp(ref, slim).max().item() <= 2
    assert (ref.argmax(-1) == slim.argmax(-1)).all(), "greedy token must be unchanged"


def test_reengage_past_crossover_is_bit_exact_when_compressor_kept(installed, args):
    """The correctness gate, proved: keep the indexer's Compressor advanced through the select-all
    phase and, the moment the top-k has to discriminate (end_pos > 35), the re-engaged selection is
    bit-exact to a pure-reference run — because the cache it scores against was filled identically."""
    ir, is_ = _ratio4_indexer(args, 1), _ratio4_indexer(args, 1)
    xs, qrs = _seq(args, 42)                              # crosses 35
    saw_discriminating = False
    for sp in range(42):
        SLIM.set_active(False)
        ref = installed.Indexer.forward(ir, xs[sp], qrs[sp], sp, args.window_size)
        SLIM.set_active(True)
        SLIM.set_job_max_pos(None)                        # unknown -> keep compressor
        slim = installed.Indexer.forward(is_, xs[sp], qrs[sp], sp, args.window_size)
        if (sp + 1) // 4 > args.index_topk:               # discriminating regime
            saw_discriminating = True
            assert torch.equal(ref, slim), f"re-engage not bit-exact at step {sp}"
    assert saw_discriminating, "test never reached the discriminating regime"


def test_mutation_skipping_compressor_diverges_past_crossover(installed, args):
    """The mutation-check the coordinator asked for: a version that does NOT keep the compressor
    advanced (here: a job that under-declares its horizon as short, then runs long) re-engages against
    a half-empty cache and picks a DIFFERENT set of keys past the crossover. If this ever stops
    diverging, the compressor-advance became dead code and the skip is silently unsafe."""
    ir, im = _ratio4_indexer(args, 1), _ratio4_indexer(args, 1)
    xs, qrs = _seq(args, 42)
    diverged = 0
    checked = 0
    for sp in range(42):
        SLIM.set_active(False)
        ref = installed.Indexer.forward(ir, xs[sp], qrs[sp], sp, args.window_size)
        SLIM.set_active(True)
        SLIM.set_job_max_pos(30)                          # LIE: claim short, but we run to 42
        mut = installed.Indexer.forward(im, xs[sp], qrs[sp], sp, args.window_size)
        if (sp + 1) // 4 > args.index_topk:
            checked += 1
            if sorted(ref.flatten().tolist()) != sorted(mut.flatten().tolist()):
                diverged += 1
    assert checked > 0
    assert diverged > 0, "skipping the compressor must corrupt the re-engaged selection"


# ── item 2: the QAT-sim skip ──────────────────────────────────────────────────────────────────────

def test_noqat_inplace_is_a_noop(installed):
    """inplace act_quant/fp4_act_quant leave the tensor at full bf16 — the fp8/fp4 round-trip is gone."""
    SLIM.set_active_noqat(True)
    x = torch.randn(2, 128, dtype=torch.bfloat16)
    x0 = x.clone()
    r = installed.act_quant(x, 64, None, torch.float32, True)
    assert r is x and torch.equal(x, x0), "inplace act_quant must not touch the tensor"
    q = torch.randn(2, 64, dtype=torch.bfloat16)
    q0 = q.clone()
    installed.fp4_act_quant(q, 32, True)
    assert torch.equal(q, q0), "inplace fp4_act_quant must not touch the tensor"


def test_noqat_reference_really_reduced_precision(installed):
    """Prove the thing being removed is real work: with the item off, the reference inplace path DOES
    change the tensor (that is the precision reduction a bf16-KV deployment does not want)."""
    SLIM.set_active_noqat(False)
    x = torch.randn(2, 128, dtype=torch.bfloat16)
    x0 = x.clone()
    installed.act_quant(x, 64, None, torch.float32, True)
    assert not torch.equal(x, x0), "reference inplace act_quant is supposed to quantize"


def test_noqat_noninplace_still_quantizes(installed):
    """The real fp8/fp4 GEMM quantization (Linear.forward, inplace defaulted False) must be untouched."""
    SLIM.set_active_noqat(True)
    x = torch.randn(4, 128, dtype=torch.bfloat16)
    packed, scales = installed.act_quant(x, 128, None, torch.float32, False)
    assert packed.dtype == torch.float8_e4m3fn and scales.dtype == torch.float32
    q = torch.randn(4, 64, dtype=torch.bfloat16)
    p4, s4 = installed.fp4_act_quant(q, 32, False)
    assert p4.dtype == torch.float4_e2m1fn_x2 and s4.dtype == torch.float8_e8m0fnu


def test_noqat_logits_move_within_a_bounded_amount(installed, args):
    """Removing the QAT sim changes the model (it is more precise, not bit-exact) — bound the change.
    On the toy random config the decode logits move by O(the KV's fp8 quantization error): documented,
    small relative to the logit magnitude, and gated OFF for any deployment that stores an fp8 KV."""
    def run(active):
        torch.manual_seed(0)
        prompt, steps = 16, 20
        x = torch.randint(0, args.vocab_size, (1, prompt + steps))
        m = REFCPU.build_oracle(args, SEED)
        SLIM.set_active_noqat(active)
        out = []
        with torch.no_grad():
            m(x[:, :prompt])
            for i in range(prompt, prompt + steps):
                _, logits, _ = m(x[:, i:i + 1], i)
                out.append(logits.clone())
        return torch.cat(out, 0)

    ref, slim = run(False), run(True)
    d = (ref - slim).abs()
    assert d.max().item() > 0, "NOQAT must actually change something (else it is not installed)"
    # bounded: the perturbation is the fp8 KV quantization scale, well under the logit magnitude
    assert d.max().item() < 0.5 * ref.abs().max().item()


# ── the install is REVERSIBLE, and the chunk path never sees it ──────────────────────────────────
# `mod` is a process singleton (v4_ref_cpu.load_ref caches it), so an install that cannot be undone
# is an install every later job in the same interpreter inherits. Both of these are regressions that
# actually fired: the first as a cross-file test failure, the second as its cause.

def test_uninstall_restores_the_reference_and_is_idempotent(installed):
    """After uninstall the reference's OWN function objects are back — identity, not behaviour, so a
    later `set_active(True)` cannot silently re-arm anything.

    Takes `installed` (armed by the module fixture), unwinds it, proves the round trip, and re-arms,
    so it neither depends on nor disturbs the module's install state."""
    mod = installed
    assert SLIM.uninstall(mod) == {"indexer_skip": True, "noqat": True}
    ref_fwd, ref_aq, ref_fp4 = mod.Indexer.forward, mod.act_quant, mod.fp4_act_quant
    assert not getattr(ref_fwd, "_v4_ref_slim", False), "uninstall left a slim function bound"
    assert not getattr(ref_aq, "_v4_ref_slim", False)
    assert SLIM.uninstall(mod) == {"indexer_skip": False, "noqat": False}, "uninstall is idempotent"
    assert mod.Indexer.forward is ref_fwd, "the idempotent call must not re-bind anything"

    assert SLIM.install(mod, item1=True, item2=True) == {"indexer_skip": True, "noqat": True}
    assert mod.Indexer.forward is SLIM.slim_indexer_forward
    assert SLIM.uninstall(mod) == {"indexer_skip": True, "noqat": True}
    assert mod.Indexer.forward is ref_fwd, "a second round trip must land on the SAME reference"
    assert mod.act_quant is ref_aq and mod.fp4_act_quant is ref_fp4
    SLIM.install(mod, item1=True, item2=True)          # hand the module fixture back what it armed


def test_indexer_skip_does_not_reach_the_chunked_verify_path():
    """V4_REF_SLIM's item 1 and V4_FAST_VERIFY are MUTUALLY EXCLUSIVE per position, and this pins why.

    v4_stage._chunk_attention calls `_chunk_indexer(...)` — its own s-position reimplementation — not
    `self.indexer.forward(...)`, so rebinding `Indexer.forward` cannot reach a chunk. Item 2 (noqat)
    is a MODULE-LEVEL name the chunk path does read (`M.act_quant`), so that one composes. Documented
    in docs/V4_FULL_STACK.md; asserted here so a refactor that routes the chunk through the class
    method (and would then silently apply the slim index to an s>1 pass) fails loudly instead."""
    import inspect
    V4 = pytest.importorskip("v4_stage")
    src = inspect.getsource(V4._chunk_attention)
    assert "_chunk_indexer(" in src, "the chunk path no longer owns its indexer — recheck exclusivity"
    assert "self.indexer.forward" not in src and "self.indexer(" not in src
    assert "M.act_quant(" in src, "item 2 (noqat) rebinds a module global the chunk path DOES read"


def test_indexer_skip_does_not_reach_a_WHOLE_LAYER_GRAPHED_decode_step():
    """The same bypass, on the lever the ring recipe actually turns on — and this one is easy to miss.

    v4_whole_layer_graph never calls `Indexer.forward`. It passes the Indexer MODULE as data to
    `_indexer_decode_cs`, its own capture-safe reimplementation that reads the module's weights and
    buffers directly, because the reference's version bakes a growing read width a graph cannot hold.
    So under `V4_CUDA_GRAPH=whole`, rebinding `Indexer.forward` cannot reach a graphed decode step.

    V4_REF_SLIM is still worth setting with `whole`, for a reason that is NOT the one on the tin:
      * PREFILL is `start_pos == 0`, the graph gate requires `start_pos > 0`, so prefill runs the
        reference `Attention.forward` -> `self.indexer(...)` and DOES take the slim path.
      * a layer that could not capture (V4_GRAPH_MAX, or a failed capture) falls back to eager.
    What it does NOT buy under `whole` is the decode-step launch saving the flag is advertised for —
    the graph collapses those launches anyway. Documented in docs/V4_FULL_STACK.md so the ring's A/B
    is not read as "ref-slim did nothing".

    Item 2 (noqat) is different again: `_Ref` snapshots the module-level `act_quant`/`fp4_act_quant`
    at WholeBlockGraphs construction, which is after load_ref() installed them, so that one DOES reach
    a graphed step."""
    import inspect
    WL = pytest.importorskip("v4_whole_layer_graph")
    attn = inspect.getsource(WL.attn_decode_cs)
    assert "_indexer_decode_cs(" in attn, "the capture-safe path no longer owns its indexer"
    assert "indexer.forward" not in attn and "A.indexer(" not in attn
    assert "R.act_quant(" in attn, "item 2 rebinds a name the capture-safe path DOES read"
    idx = inspect.getsource(WL._indexer_decode_cs)
    assert "R.fp4_act_quant(" in idx and "I.forward" not in idx
    # and _Ref binds the module's OWN functions, so an install before Stage construction is picked up
    ref_src = inspect.getsource(WL._Ref.__init__)
    assert "M.act_quant" in ref_src and "M.fp4_act_quant" in ref_src
