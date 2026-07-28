"""Kimi-K3 stage, milestone 2: the GPU seams, proven as far as a box with no GPU can prove them.

M2 puts four kernels behind k3_stage -- FlashKDA for the delta recurrence, fla-core for the AttnRes
collapse, vLLM Marlin for the MXFP4 routed experts, and a whole-layer CUDA graph over the lot. None
of those run here. What DOES run here is everything that decides whether they run and how, and that
is where the expensive mistakes live: a format mis-read that loads the wrong packing, a graph that
re-captures every step, a static-state path that quietly changes the answer, an activation stand-in
that survives into production. Each of those is a shape-level or dict-level fact.

The kernels themselves are pinned on rented hardware -- scratchpad/k3-research-20260727/m2-results.md
records the tolerances and the measurements, and tests/test_k3_stage.py's parity suite is what the
GPU path is measured against.

Run: python3 -m pytest tests/test_k3_stage_m2.py -q
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
pytest.importorskip("einops")
K3 = pytest.importorskip("k3_stage")
MOE = pytest.importorskip("k3_moe_mxfp4")
ATTNRES = pytest.importorskip("k3_attnres")
FLASH = pytest.importorskip("k3_kda_flash")
KDA = pytest.importorskip("k3_kda_cpu")

# K3's real block, trimmed to the keys this repo reads. Kept as a literal rather than derived from a
# fixture so a checkpoint respin that changes the packing shows up as a test diff, not a silent pass.
K3_QUANT = {
    "format": "mxfp4-pack-quantized",
    "quant_method": "compressed-tensors",
    "ignore": ["re:.*self_attn.*", "re:.*shared_experts.*", "re:.*mlp\\..*_proj", "lm_head"],
    "config_groups": {
        "group_0": {
            "targets": ["re:.*block_sparse_moe.experts.*"],
            "weights": {"num_bits": 4, "group_size": 32, "strategy": "group", "symmetric": True,
                        "type": "float"},
        },
    },
}


# ── the checkpoint's quantization block ──────────────────────────────────────────────────────────

def test_spec_reads_k3s_own_quantization_block():
    sp = MOE.spec(K3_QUANT)
    assert (sp["format"], sp["group_size"], sp["num_bits"]) == ("mxfp4-pack-quantized", 32, 4)
    assert sp["symmetric"] is True
    assert sp["ignore"] == sorted(K3_QUANT["ignore"])
    assert MOE.routed_experts_only(sp), "only the routed experts are 4-bit in K3"


@pytest.mark.parametrize("mutate, match", [
    (lambda q: q.update(format="nvfp4-pack-quantized"), "quantization format"),
    (lambda q: q["config_groups"]["group_0"]["weights"].update(group_size=16), "group_size 16"),
    (lambda q: q["config_groups"]["group_0"]["weights"].update(num_bits=8), "num_bits 8"),
    (lambda q: q["config_groups"]["group_0"]["weights"].update(strategy="tensor"), "strategy"),
    (lambda q: q.update(config_groups={}), "no config_groups"),
])
def test_spec_refuses_a_packing_this_backend_has_not_run(mutate, match):
    """Every one of these would otherwise load bytes under the wrong interpretation and decode
    fluently. There is no bf16 upstream to fall back to, so a wrong read has no safety net."""
    q = json.loads(json.dumps(K3_QUANT))
    mutate(q)
    with pytest.raises((NotImplementedError, ValueError), match=match):
        MOE.spec(q)


def test_spec_refuses_config_groups_that_disagree():
    """A mixed-precision respin flattened to one group's numbers would 4-bit half the experts at the
    wrong group size. Two groups, two specs, one loud refusal."""
    q = json.loads(json.dumps(K3_QUANT))
    q["config_groups"]["group_1"] = {"targets": ["re:.*shared_experts.*"],
                                     "weights": {"num_bits": 8, "group_size": 32,
                                                 "strategy": "group", "symmetric": True}}
    with pytest.raises(NotImplementedError, match="disagree"):
        MOE.spec(q)


def test_routed_experts_only_is_false_when_something_else_is_quantized():
    q = json.loads(json.dumps(K3_QUANT))
    q["config_groups"]["group_0"]["targets"] = ["re:.*self_attn.*"]
    assert not MOE.routed_experts_only(MOE.spec(q))


def test_backend_is_none_without_a_quant_block_and_refuses_when_forced(monkeypatch):
    monkeypatch.setattr(MOE, "K3_MOE_BACKEND", "auto")
    assert MOE.backend(None) == "none"
    monkeypatch.setattr(MOE, "K3_MOE_BACKEND", "marlin")
    with pytest.raises(RuntimeError, match="no quantization_config"):
        MOE.backend(None)


def test_backend_off_keeps_m1s_bf16_path(monkeypatch):
    """K3_MOE_BACKEND=none must not even look at the block — that is the escape hatch when the
    4-bit path is what is broken."""
    monkeypatch.setattr(MOE, "K3_MOE_BACKEND", "none")
    assert MOE.backend(K3_QUANT) == "none"
    assert MOE.backend({"format": "something-we-never-ran"}) == "none"


def test_backend_auto_needs_a_gpu(monkeypatch):
    monkeypatch.setattr(MOE, "K3_MOE_BACKEND", "auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert MOE.backend(K3_QUANT) == "none"


def test_unknown_backend_name_is_refused(monkeypatch):
    monkeypatch.setattr(MOE, "K3_MOE_BACKEND", "cutlass")
    with pytest.raises(RuntimeError, match="expected auto, marlin or none"):
        MOE.backend(K3_QUANT)


# ── the activation G0 stood in for ───────────────────────────────────────────────────────────────

def test_situ_stand_in_matches_the_references_own_activation():
    """G0 substituted SiLU for `situ` to get timings, and said so. This is the assertion that keeps
    that substitution out of the serving path: vLLM's fused MoE knows silu/gelu/swigluoai and
    nothing else, so `situ` is installed as a callback and has to be the reference's function."""
    M = K3.ref()
    beta, linear_beta = 4.0, 25.0
    act = M.SituAndMul(beta=beta, linear_beta=linear_beta)
    torch.manual_seed(11)
    x = torch.randn(7, 64) * 3.0
    want = act(x)
    got = torch.zeros(7, 32)
    MOE._situ_and_mul(got, x, beta, linear_beta)
    assert torch.allclose(got, want, atol=0, rtol=0), \
        f"max |diff| {(got - want).abs().max().item():.3e}"
    # and it is NOT silu_and_mul, which is what vLLM would apply if the install silently failed
    d = x.shape[-1] // 2
    silu = torch.nn.functional.silu(x[:, :d]) * x[:, d:]
    assert not torch.allclose(got, silu, atol=1e-3)


def test_situ_stand_in_honours_a_config_without_linear_beta():
    M = K3.ref()
    x = torch.randn(4, 8)
    got = torch.zeros(4, 4)
    MOE._situ_and_mul(got, x, 2.0, None)
    assert torch.allclose(got, M.SituAndMul(beta=2.0, linear_beta=None)(x))


def test_situ_install_replaces_vllms_activation_with_the_reference_math():
    """After the install, vLLM's own activation must never run and the seat must compute situ."""
    class Modular:
        def activation(self, activation, output, input, **kw):
            raise AssertionError("vLLM's own activation must not run once situ is installed")

    obj = Modular()
    assert MOE._install_situ(obj, 4.0, 25.0)
    probe = torch.tensor([[1.0, -2.0, 3.0, 0.5]])
    out, want = torch.zeros(1, 2), torch.zeros(1, 2)
    obj.activation("silu", out, probe)                     # would raise if vLLM's were still bound
    MOE._situ_and_mul(want, probe, 4.0, 25.0)
    assert torch.equal(out, want)


@pytest.mark.parametrize("cls", [
    type("NoHook", (), {}),
    type("NotCallable", (), {"activation": "a string, not a method"}),
    type("ReadOnly", (), {"activation": property(lambda self: None)}),
])
def test_situ_install_reports_failure_rather_than_leaving_silu_in_place(cls):
    """No seat, or a seat that cannot be assigned, has to come back False — `arm()` turns that into
    a refusal, because a silent SiLU substitution is a different model, not a slower one."""
    assert not MOE._install_situ(cls(), 4.0, 25.0)


def test_fused_experts_is_resolved_through_the_kernels_own_path():
    """vLLM 0.26 parks the MarlinExperts instance on quant_method.moe_kernel.impl.fused_experts, and
    only after process_weights_after_loading. Resolving None is what makes `arm()` refuse."""
    class Impl:
        fused_experts = "marlin-experts"

    class Kernel:
        impl = Impl()

    class QM:
        moe_kernel = Kernel()

    assert MOE.fused_experts_of(QM()) == "marlin-experts"
    assert MOE.fused_experts_of(object()) is None


# ── graph routing ────────────────────────────────────────────────────────────────────────────────

class _StubStage:
    """Enough of a Stage for _StageGraph.plan, which touches nothing else. plan() is deliberately
    pure so this stub is honest rather than a mock of the code under test."""
    lo, hi = 0, 4


def _graph():
    return K3._StageGraph(_StubStage())


def _pair(tokens, blocks, hidden=8, batch=1):
    return (torch.zeros(batch, tokens, hidden),
            torch.zeros(batch * tokens, blocks, hidden))


def test_graph_key_is_tokens_and_attnres_depth():
    """Not a context bucket: a KDA-only stage reads no position, so only the token count and the
    depth of the AttnRes stack change its shapes."""
    g = _graph()
    assert g.plan(*_pair(1, 3)) == ("capture", (1, 3))
    g.graphs[(1, 3)] = object()
    assert g.plan(*_pair(1, 3)) == ("replay", (1, 3))
    assert g.plan(*_pair(1, 4)) == ("capture", (1, 4)), "a deeper stack is a different graph"
    assert g.plan(*_pair(5, 3)) == ("capture", (5, 3)), "a prefill chunk is a different graph"


def test_a_stage_settles_on_one_graph():
    """The payoff of that key: decode is always (1, nb) and nb is fixed for a given layer range, so
    a warm stage captures once and replays forever."""
    g = _graph()
    g.graphs[(1, 8)] = object()
    for _ in range(50):
        assert g.plan(*_pair(1, 8)) == ("replay", (1, 8))
    assert len(g.graphs) == 1


def test_batched_and_mismatched_payloads_route_eager_without_being_remembered():
    """block_residual's leading dim is B*S; a pair that does not satisfy that is not a graph shape
    and is not a shape worth caching a refusal for either."""
    g = _graph()
    assert g.plan(*_pair(1, 3, batch=2)) == ("eager", None)
    h, br = _pair(2, 3)
    assert g.plan(h, br[:1]) == ("eager", None)
    assert g.graphs == {} and g.eager == set()


def test_a_failed_shape_is_remembered_and_never_retried():
    g = _graph()
    g.eager.add((1, 3))
    assert g.plan(*_pair(1, 3)) == ("eager", (1, 3))


def test_the_graph_budget_is_process_wide(monkeypatch):
    """Each captured graph pins its own workspace pool, so the cap is on the process, not the stage
    — two stages in one worker must not each get the full budget."""
    monkeypatch.setattr(K3, "K3_GRAPH_MAX", 2)
    monkeypatch.setattr(K3, "_GRAPH_COUNT", 0)
    a, b = _graph(), _graph()
    assert a.plan(*_pair(1, 1)) == ("capture", (1, 1))
    monkeypatch.setattr(K3, "_GRAPH_COUNT", 2)
    assert a.plan(*_pair(1, 1)) == ("budget", (1, 1))
    assert b.plan(*_pair(1, 7)) == ("budget", (1, 7))


def test_a_captured_shape_still_replays_after_the_budget_is_spent(monkeypatch):
    """Hitting the cap must not turn a warm stage cold — already-captured shapes keep replaying."""
    monkeypatch.setattr(K3, "K3_GRAPH_MAX", 1)
    monkeypatch.setattr(K3, "_GRAPH_COUNT", 99)
    g = _graph()
    g.graphs[(1, 8)] = object()
    assert g.plan(*_pair(1, 8)) == ("replay", (1, 8))


# ── backend resolution on a box with no GPU ──────────────────────────────────────────────────────

def test_every_backend_defaults_to_the_cpu_path_without_cuda(monkeypatch):
    """The M1 contract: knobs off (or no device) and nothing about the stage changes."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(KDA, "K3_KDA_BACKEND", "auto")
    monkeypatch.setattr(ATTNRES, "K3_ATTNRES_BACKEND", "auto")
    monkeypatch.setattr(MOE, "K3_MOE_BACKEND", "auto")
    assert KDA.backend() == "cpu"
    assert ATTNRES.backend() == "ref"
    assert MOE.backend(K3_QUANT) == "none"


@pytest.mark.parametrize("name", ["cpu", "fla", "flashkda"])
def test_kda_backend_names_are_accepted(monkeypatch, name):
    monkeypatch.setattr(KDA, "K3_KDA_BACKEND", name)
    assert KDA.backend() == name


def test_unknown_kda_backend_name_is_refused(monkeypatch):
    monkeypatch.setattr(KDA, "K3_KDA_BACKEND", "triton")
    with pytest.raises(RuntimeError, match="expected auto, cpu, fla or flashkda"):
        KDA.backend()


def test_attnres_install_is_idempotent_and_restores_the_reference(monkeypatch):
    """install() rebinds a name ON the reference module, so a process that flips the knob between
    Stages must not end up half-swapped."""
    M = K3.ref()
    monkeypatch.setattr(ATTNRES, "K3_ATTNRES_BACKEND", "ref")
    assert ATTNRES.install(M) == "ref"
    original = M._apply_attn_res_ref
    assert M._apply_attn_res is original
    assert ATTNRES.install(M) == "ref"
    assert M._apply_attn_res_ref is original, "the original must be stashed once, not re-stashed"


def test_attnres_fla_is_refused_when_the_package_is_absent(monkeypatch):
    monkeypatch.setattr(ATTNRES, "K3_ATTNRES_BACKEND", "fla")
    monkeypatch.setattr(ATTNRES, "_FUSED", False)          # memoized "not importable"
    with pytest.raises(RuntimeError, match="fla-core==0.5.2"):
        ATTNRES.install(K3.ref())


# ── the FlashKDA contract, as far as it can be checked without the kernel ────────────────────────

def _kda_args(head_dim=128, heads=2, v_heads=None, **over):
    v_heads = v_heads or heads
    a = dict(q=torch.zeros(1, 1, heads, head_dim), k=torch.zeros(1, 1, heads, head_dim),
             v=torch.zeros(1, 1, v_heads, head_dim), g=torch.zeros(1, 1, v_heads, head_dim),
             beta=torch.zeros(1, 1, v_heads), A_log=torch.zeros(v_heads),
             dt_bias=torch.zeros(v_heads * head_dim), lower_bound=-5.0,
             use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True)
    a.update(over)
    return a


@pytest.mark.parametrize("over, match", [
    ({"use_qk_l2norm_in_kernel": False}, "UNCONDITIONALLY"),
    ({"use_gate_in_kernel": False}, "UNCONDITIONALLY"),
    ({"use_beta_sigmoid_in_kernel": False}, "UNCONDITIONALLY"),
    ({"allow_neg_eigval": True}, "allow_neg_eigval"),
    ({"cu_seqlens": torch.zeros(2)}, "varlen"),
])
def test_flashkda_refuses_a_call_shape_the_kernel_cannot_serve(over, match):
    """FlashKDA folds the l2-norm, the gate and the beta sigmoid in unconditionally. A caller that
    pre-applied any of them would be double-applying, silently -- so the wrapper refuses instead of
    dropping the flag into **kwargs, which is the trap k3_kda_cpu's docstring names for M2."""
    with pytest.raises(NotImplementedError, match=match):
        FLASH.fused_recurrent_kda(**_kda_args(**over))


def test_flashkda_refuses_a_head_dim_it_was_never_built_for():
    """csrc TORCH_CHECKs D == 128. Saying so here means a toy parity config fails with the reason
    rather than a bare check from inside the extension."""
    assert FLASH.KERNEL_HEAD_DIM == 128
    with pytest.raises(NotImplementedError, match="head_dim-128 kernel"):
        FLASH.fused_recurrent_kda(**_kda_args(head_dim=8))


def test_flashkda_refuses_gqa_shapes():
    with pytest.raises(NotImplementedError, match="no GQA"):
        FLASH.fused_recurrent_kda(**_kda_args(heads=2, v_heads=4))


def test_flashkda_refuses_an_unsliced_a_log():
    """K3 ships A_log padded to head_dim; k3_stage._fixup slices it to num_heads. A stage that
    skipped that slice would decay the wrong heads, and the kernel indexes A_log by head."""
    with pytest.raises(RuntimeError, match="A_log is"):
        FLASH.fused_recurrent_kda(**_kda_args(A_log=torch.zeros(128)))


# ── static state, which is testable on CPU precisely because it changes no numbers ───────────────

@pytest.fixture
def static_state():
    """Turn K3_STATIC_STATE on for a Stage built inside the fixture, and put everything back."""
    import importlib
    os.environ["K3_STATIC_STATE"] = "1"
    importlib.reload(K3)
    K3.ref()
    yield K3
    del os.environ["K3_STATIC_STATE"]
    importlib.reload(K3)
    KDA.set_static_state(False)
    K3.ref()


def _tiny_stage(mod, **kw):
    from test_k3_stage import _tiny_config
    cfg = _tiny_config()
    torch.manual_seed(4242)
    st = mod.Stage(0, 3, cfg, head=True, device="cpu", **kw)      # 3 KDA layers, no MLA
    with torch.no_grad():
        for p in st.layers.parameters():
            p.normal_(0.0, 0.02)
    return st


def test_static_state_gives_the_kda_buffers_fixed_addresses(static_state):
    """The whole point: `cache.recurrent_states[li] = ret` has to stop being a reallocation, or a
    captured graph replays against a freed pointer. Both halves -- the recurrent summary AND the
    three conv windows -- or the graph reads one live and one dead."""
    st = _tiny_stage(static_state)
    before = [(st.cache.recurrent_states[li].data_ptr(),
               tuple(c.data_ptr() for c in st.cache.conv_states[li])) for li in range(3)]
    h, br = st.embed([[1, 2, 3]])
    h, br = st.forward(h, br, 0)
    for step in range(3):
        h, br = st.forward(h[:, -1:], br[-1:], 3 + step)
    after = [(st.cache.recurrent_states[li].data_ptr(),
              tuple(c.data_ptr() for c in st.cache.conv_states[li])) for li in range(3)]
    assert after == before, "a KDA buffer moved — every captured graph would now be stale"


def _decode(stage):
    h, br = stage.embed([[1, 2, 3]])
    h, br = stage.forward(h, br, 0)
    outs = []
    for step in range(3):
        h, br = stage.forward(h[:, -1:], br[-1:], 3 + step)
        outs.append(h.clone())
    return torch.cat(outs)


def test_static_state_changes_no_numbers(static_state):
    """It is an allocation policy, not a numerics change: the in-place write copies exactly the
    tensor the allocating path would have returned. If this ever goes red, the graph path is not
    slower, it is wrong.

    Both arms have to be built in THIS process, so the knob is flipped around the second build --
    re-importing k3_stage would hand back the same already-reloaded module and quietly compare the
    static path against itself."""
    mod = static_state
    st = _tiny_stage(mod)
    with_static = _decode(st)
    weights = {"layers": st.layers.state_dict(), "embed": st.embed_tokens.state_dict()}

    KDA.set_static_state(False)
    mod.K3_STATIC_STATE = False
    try:
        plain = _tiny_stage(mod)
        plain.layers.load_state_dict(weights["layers"])
        plain.embed_tokens.load_state_dict(weights["embed"])
        assert plain.cache.recurrent_states[0] is None, "the knob-off arm must not preallocate"
        without = _decode(plain)
    finally:
        mod.K3_STATIC_STATE = True
        KDA.set_static_state(True)
    assert torch.equal(with_static, without)


def test_reset_under_static_state_keeps_the_addresses_and_still_forgets_the_job(static_state):
    """A logical reset, m25's shape: zero in place rather than rebuild, so a warm stage's graphs
    survive a job boundary -- and a second job still cannot see the first one's state."""
    st = _tiny_stage(static_state)
    ptrs = lambda: [st.cache.recurrent_states[li].data_ptr() for li in range(3)]   # noqa: E731
    before = ptrs()
    h, br = st.embed([[5, 6]])
    st.forward(h, br, 0)
    assert st.cache.recurrent_states[0].abs().sum() > 0
    st.reset()
    assert st._pos == 0
    assert ptrs() == before
    assert st.cache.recurrent_states[0].abs().sum() == 0
    assert all(c.abs().sum() == 0 for c in st.cache.conv_states[0])


# ── the two KDA gate parameters ──────────────────────────────────────────────────────────────────

def test_gate_parameters_survive_a_low_precision_stage():
    """A_log and dt_bias parameterize the decay inside a sigmoid and every backend consumes them
    fp32 (FlashKDA TORCH_CHECKs it). A blanket .to(bfloat16) would round them to ~3 digits for no
    saving -- 96 and 96x128 numbers per layer."""
    from test_k3_stage import _tiny_config
    st = K3.Stage(0, 2, _tiny_config(), device="cpu", dtype=torch.bfloat16)
    for L in st.layers:
        assert L.self_attn.A_log.dtype == torch.float32
        assert L.self_attn.dt_bias.dtype == torch.float32
        assert L.self_attn.q_proj.weight.dtype == torch.bfloat16, "everything else still casts"


def test_graph_is_refused_with_a_reason_rather_than_half_captured():
    """A CPU stage, an MLA stage and a reference-moe_infer stage each have a specific reason they
    cannot be captured; the operator gets that reason, not a mysterious eager run."""
    from test_k3_stage import _tiny_config
    cfg = _tiny_config()
    assert "device is cpu" in K3.Stage(0, 3, cfg, device="cpu")._graph_refusal()
    st = K3.Stage(3, 4, cfg, device="cpu")                        # the MLA layer
    st.device = "cuda:0"                                          # pretend past the device check
    assert "MLA" in st._graph_refusal()
