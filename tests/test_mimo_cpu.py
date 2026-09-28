"""MiMo-V2 CPU reference backend (BUILD-PLAN R2), CI gates on a tiny random MiMo-shaped model.

What each part proves (the model and its independent weight decoding: tests/mimo_tiny.py):
  1. Vendor math. The adapter's canonical layer (shard/runtime/adapters/mimo_v2.py) against the
     vendor's own MiMoV2DecoderLayer, vendored minimally at vendor/mimo_v2_ref/ (math lines
     verbatim, transformers plumbing removed), both fed the same weights: the adapter decodes the
     checkpoint BYTES through shard/formats, the vendor module gets values decoded independently
     by the generator. fp32 must agree to ~1e-7 of scale on a global dense, an SWA MoE and a
     global MoE layer with the window binding; bf16 must sit in the vendor's own rounding band;
     routing sets must be equal. Negative controls show the fp32 check catches a window off by
     one, a dropped sink, a dropped v pre-scale and the vendor HF file's plain qkv split.
  2. Loader guards: nothing half-loaded, no down-cast router bias, no unknown layer tensor.
  3. The StageRuntime conformance suite on CpuRuntime in two geometries (w > W as on the real
     model, and W > w), so a swapped window/spec-window cannot hide.
  4. The production geometry (w = 128, W = 64): pipelined-speculation rollback across the window
     edge is bit-identical to a clean run and to the reference; a 2-runtime chain equals one
     runtime (partition invariance) with a sequence longer than the window.
"""
import dataclasses
import math
import types

import numpy as np
import pytest
import torch

import mimo_v2_ref.modeling_mimo_v2_layer as vm          # vendor/ is on sys.path (root conftest.py)
from mimo_tiny import MimoFactory, bf16_values, tiny_config
from runtime_conformance import CHECKS, Skip, _accepted, _continue, _play, _spec_script, check_swa_retention
from shard.formats.quant import MX_BLOCK
from shard.runtime.adapters import mimo_v2 as adapter
from shard.runtime.adapters.mimo_v2 import MimoConfig
from shard.runtime.cpu.mimo_v2 import MimoLayer, MimoOracle, WeightError, read_tensor
from shard.runtime.api import BadRequest, LoadRefused, Rows, Sampling, Segment, StageSpec, StepIn
from shard.runtime.cpu.runtime import CpuRuntime, draw, uniform01

torch.set_grad_enabled(False)

T_TOK = 24                                  # > the tiny window (8): SWA masks bind at T > 8
FP32_TOL = 2e-6                             # x max(1, output scale); measured ~2e-7


@pytest.fixture(scope="module")
def tiny():
    return MimoFactory(window=8, W=4)


def _vendor_layer(f, i, dtype):
    """The vendor MiMoV2DecoderLayer for tiny layer i with the generator's logical weights."""
    c = types.SimpleNamespace(**f.config)
    c._attn_implementation = "eager"
    mod = vm.MiMoV2DecoderLayer(c, i, attention_projection_layout="fused_qkv").to(dtype).eval()
    lg = f.ckpt.logical[i]
    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))    # noqa: E731
    mod.input_layernorm.weight.copy_(t(lg["ln1"]))
    mod.post_attention_layernorm.weight.copy_(t(lg["ln2"]))
    mod.self_attn.qkv_proj.weight.copy_(t(lg["qkv"]))
    mod.self_attn.o_proj.weight.copy_(t(lg["o_proj"]))
    if "sink" in lg:
        mod.self_attn.attention_sink_bias.copy_(t(lg["sink"]))
    if "router" in lg:                      # the router stays fp32, as the checkpoint and vLLM keep it
        mod.mlp.gate.weight = torch.nn.Parameter(t(lg["router"]), requires_grad=False)
        mod.mlp.gate.e_score_correction_bias = torch.nn.Parameter(t(lg["router_bias"]), requires_grad=False)
        for e, mats in lg["experts"].items():
            for proj, m in zip(("gate_proj", "up_proj", "down_proj"), mats):
                getattr(mod.mlp.experts[e], proj).weight.copy_(t(m))
    else:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            getattr(mod.mlp, proj).weight.copy_(t(lg[proj]))
    rot = vm.MiMoV2RotaryEmbedding(c, is_swa=f.config["hybrid_layer_pattern"][i] == 1)
    window = f.config["sliding_window"] if f.config["hybrid_layer_pattern"][i] == 1 else None

    def run(h):
        x = h.to(dtype)[None]
        pos = torch.arange(h.shape[0])[None]
        return mod(x, attention_mask=vm.mimo_mask(h.shape[0], window, dtype), position_ids=pos,
                   position_embeddings=rot(x, pos))[0]
    return mod, run


def _h(seed, T=T_TOK, H=256):
    return torch.randn(T, H, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)


def _stats(a, b):
    d = (a.float() - b.float()).abs()
    return d.max().item(), d.mean().item(), b.float().abs().max().item()


def _ulp(x):
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7)


# ── 1. vendor math ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("layer", [0, 1, 5], ids=["L0-global-dense", "L1-swa-moe", "L5-global-moe"])
def test_layer_matches_vendor_fp32(tiny, layer):
    ref = MimoOracle.from_source(tiny.config, tiny.ckpt, [layer], torch.float32)
    h = _h(layer)
    tr = {}
    ours = ref.forward(h.float(), 0, ref.new_cache(), traces=tr)
    mod, run = _vendor_layer(tiny, layer, torch.float32)
    mx, mean, scale = _stats(ours, run(h.float()))
    assert mx <= FP32_TOL * max(1.0, scale), (mx, mean, scale)
    if "topk_idx" in tr[layer]:
        vidx, _ = mod.mlp.gate(tr[layer]["ffn_in"][None])
        assert [set(r.tolist()) for r in vidx] == [set(r.tolist()) for r in tr[layer]["topk_idx"]]


@pytest.mark.parametrize("layer", [0, 1, 5], ids=["L0-global-dense", "L1-swa-moe", "L5-global-moe"])
def test_layer_bf16_in_the_vendor_band(tiny, layer):
    """Canonical bf16 rounds once per op; the vendor's bf16 path rounds more often (bf16 cos/sin,
    probabilities, RMSNorm before the weight). Both are judged against the exact math (the vendor
    module in fp32 on the same input): ours may not be farther from it than the vendor's own bf16,
    and the two bf16 paths agree to within a few bf16 ulps at the output scale."""
    ref = MimoOracle.from_source(tiny.config, tiny.ckpt, [layer], torch.bfloat16)
    h = _h(10 + layer)
    tr = {}
    ours = ref.forward(h, 0, ref.new_cache(), traces=tr)
    mod_b, run_b = _vendor_layer(tiny, layer, torch.bfloat16)
    truth = _vendor_layer(tiny, layer, torch.float32)[1](h.float())
    vend = run_b(h)
    o_mx, o_mean, scale = _stats(ours, truth)
    v_mx, v_mean, _ = _stats(vend, truth)
    assert o_mean <= 1.1 * v_mean and o_mx <= 1.5 * v_mx, ((o_mx, o_mean), (v_mx, v_mean))
    assert _stats(ours, vend)[0] <= 4 * _ulp(scale)             # measured <= 2 over 9 draws
    if "topk_idx" in tr[layer]:
        vidx, _ = mod_b.mlp.gate(tr[layer]["ffn_in"][None])
        assert [set(r.tolist()) for r in vidx] == [set(r.tolist()) for r in tr[layer]["topk_idx"]]


def test_fp32_check_catches_the_known_landmines(tiny):
    """Each deliberate error moves the fp32 output far past FP32_TOL, so the parity test above
    would fail on it: an SWA window of w-1 or w+1, a dropped sink, a dropped v pre-scale, and the
    vendor HF file's plain split([q, k, v]) of the interleaved checkpoint qkv."""
    cfg = MimoConfig.from_hf(tiny.config)
    L1 = MimoLayer.load(cfg, 1, tiny.ckpt)
    h = _h(21).float()
    base = MimoOracle(cfg, {1: L1}, torch.float32).forward(h, 0, MimoOracle(cfg, {1: L1}, torch.float32).new_cache())
    tol = FP32_TOL * max(1.0, base.abs().max().item())

    def run(layer, spec=None):
        o = MimoOracle(cfg, {1: layer}, torch.float32)
        if spec is not None:
            o.specs[1] = spec
        return o.forward(h, 0, o.new_cache())

    wrong = {
        "window-1": run(L1, dataclasses.replace(cfg.layers[1], window=cfg.window - 1)),
        "window+1": run(L1, dataclasses.replace(cfg.layers[1], window=cfg.window + 1)),
        "no sink": run(MimoLayer(cfg, cfg.layers[1], {**L1.w, "sink": None}, L1.experts)),
        "no v-scale": run(MimoLayer(dataclasses.replace(cfg, v_scale=None), cfg.layers[1], L1.w, L1.experts)),
    }
    s, tp = cfg.layers[1], cfg.ckpt_tp      # the SWA qkv: its per-chunk grid equals the plain one,
    q_n, k_n = s.n_q * s.qk_dim, s.n_kv * s.qk_dim    # so a naive split dequantizes but mis-splits
    v_n = s.n_kv * s.v_dim
    inter = torch.cat([torch.cat([L1.w["wq"][c * q_n // tp:(c + 1) * q_n // tp],      # the checkpoint's
                                  L1.w["wk"][c * k_n // tp:(c + 1) * k_n // tp],      # [Q_c|K_c|V_c]
                                  L1.w["wv"][c * v_n // tp:(c + 1) * v_n // tp]]) for c in range(tp)])
    naive = {"wq": inter[:q_n], "wk": inter[q_n:q_n + k_n], "wv": inter[q_n + k_n:]}
    wrong["plain qkv split"] = run(MimoLayer(cfg, s, {**L1.w, **naive}, L1.experts))
    for name, out in wrong.items():
        assert (out - base).abs().max().item() > 10 * tol, name
    assert (wrong["window-1"][:cfg.window - 1] == base[:cfg.window - 1]).all()   # only past the window


def test_runtime_equals_oracle_and_decode_equals_prefill(tiny):
    """The StageRuntime path (ring, fresh-row gather) reproduces the promoted oracle's bytes, and
    token-by-token decode reproduces a single prefill (chunk invariance by construction)."""
    L = len(tiny.layer_kinds)
    ids = tiny.ids(3 * tiny.window + 5, 4)
    o = MimoOracle.from_source(tiny.config, tiny.ckpt, range(L), torch.bfloat16, head=True, tail=True)
    want = o.forward(o.embed(torch.from_numpy(ids.data.astype(np.int64))), 0, o.new_cache())
    for sched in ([(0, ids.n)], [(p, 1) for p in range(ids.n)], [(0, 7), (7, 1), (8, 13), (21, ids.n - 21)]):
        rt = tiny.make(0, L, is_head=True, is_tail=True)
        rt.open(1, 64, tiny.W)
        rt.commit(1, ids.n)
        got = Rows.concat([rt.step(StepIn((Segment(1, s, n, ids.slice(s, s + n), kind="prefill"),))).outs[0].hidden
                           for s, n in sched])
        assert got == Rows("bf16", want.view(torch.int16).numpy().view(np.uint16)), sched


def test_routing_matches_the_vendor_gate_on_many_tokens(tiny):
    """Routing sets on 3000 decisions per run (vs 48 in the layer tests), fp32 router math on
    identical inputs: must be identical. A router bias rounded to bf16 (the landmine the formats
    guard names: ulp 0.0156 at 2.0) flips some of them, so this catches it in the shared math too."""
    cfg = MimoConfig.from_hf(tiny.config)
    c = types.SimpleNamespace(**tiny.config)
    x = torch.randn(500, cfg.hidden, generator=torch.Generator().manual_seed(7)).to(torch.bfloat16)
    flips = 0
    for i in [j for j, s in enumerate(cfg.layers) if s.is_moe][:3]:
        L = MimoLayer.load(cfg, i, tiny.ckpt)
        gate = vm.MiMoV2MoEGate(c).eval()
        gate.weight = torch.nn.Parameter(torch.from_numpy(tiny.ckpt.logical[i]["router"]), requires_grad=False)
        gate.e_score_correction_bias = torch.nn.Parameter(torch.from_numpy(tiny.ckpt.logical[i]["router_bias"]),
                                                          requires_grad=False)
        vidx, vw = gate(x[None])
        ours = [L.route(x[r:r + 1]) for r in range(x.shape[0])]
        assert [set(r.tolist()) for r in vidx] == [set(o[1][0].tolist()) for o in ours]
        for r, (_, idx, wt) in enumerate(ours):         # same weights too, in the vendor's order
            order = [idx[0].tolist().index(e) for e in vidx[r].tolist()]
            assert torch.allclose(wt[0, order], vw[r], rtol=0, atol=1e-6)     # fp32 sum order only
        rounded = MimoLayer(cfg, cfg.layers[i], {**L.w, "router_bias": L.w["router_bias"].to(torch.bfloat16).float()},
                            L.experts)
        flips += sum(set(rounded.route(x[r:r + 1])[1][0].tolist()) != set(vidx[r].tolist()) for r in range(x.shape[0]))
    assert flips > 0, "a bf16-rounded router bias must be visible here"


def test_bytes_do_not_depend_on_the_ambient_thread_count(tiny):
    """Every compute call runs under the runtime's pinned intra-op thread count (part of the
    numerics_id), whatever the process has set, and the process setting is restored."""
    ids = tiny.ids(40, 5)
    old = torch.get_num_threads()
    outs = []
    try:
        for ambient in (1, 8):
            torch.set_num_threads(ambient)
            rt = tiny.make(0, 8, is_head=True, is_tail=True)
            rt.open(1, 64, tiny.W)
            rt.commit(1, 40)
            o = rt.step(StepIn((Segment(1, 0, 40, ids, kind="prefill", logits="all"),))).outs[0]
            assert torch.get_num_threads() == ambient
            outs.append((o.hidden, o.logits, rt.describe().numerics_id))
    finally:
        torch.set_num_threads(old)
    assert outs[0] == outs[1]
    four = CpuRuntime(StageSpec("mimo-tiny", 0, 8, is_head=True, is_tail=True), tiny.config, threads=4)
    four.load(tiny.ckpt)
    assert four.describe().numerics_id != outs[0][2]


def test_ids_bind_the_config_and_the_loaded_bytes(tiny):
    """numerics_id and kv_layout_id change with anything that changes what a byte means: a config
    value (the v pre-scale), or one weight byte behind identical names and headers; KV from a
    stage with another config is refused on import."""
    base = tiny.make(0, 2, is_head=True)
    alt_cfg = {**tiny.config, "attention_value_scale": 0.5}
    alt = CpuRuntime(StageSpec("mimo-tiny", 0, 2, is_head=True), alt_cfg)
    alt.load(tiny.ckpt)
    assert alt.describe().numerics_id != base.describe().numerics_id
    assert alt.describe().kv_layout_id != base.describe().kv_layout_id
    base.open(1, 64, tiny.W)
    base.commit(1, 5)
    base.step(StepIn((Segment(1, 0, 5, tiny.ids(5, 1), kind="prefill"),)))
    alt.open(1, 64, tiny.W)
    with pytest.raises(BadRequest, match="layout"):
        alt.import_state(1, base.export_state(1, 0, 5), 5)
    name = "model.layers.1.self_attn.o_proj.weight"
    w = tiny.ckpt.read(name)
    w[0, 0] ^= 1                                                   # one mantissa bit
    other = CpuRuntime(StageSpec("mimo-tiny", 0, 2, is_head=True), tiny.config)
    other.load(tiny.ckpt.replaced(name, w, "BF16"))
    assert other.describe().numerics_id != base.describe().numerics_id


def test_truncate_is_refused_mid_rebuild(tiny):
    """With a < floor <= length < F, a truncate would cut a rebuild short: refused until length == F."""
    rt = tiny.make(1, 5)
    rt.open(1, 128, tiny.W)
    a = rt.rebuild_from(1, 40, floor=38)
    rt.step(StepIn((Segment(1, a, 39 - a, tiny.hidden(39 - a, 3), kind="rebuild"),)))
    assert a < 38 <= rt.length(1) < 40
    with pytest.raises(BadRequest, match="mid-takeover"):
        rt.truncate(1, 38)
    rt.step(StepIn((Segment(1, 39, 1, tiny.hidden(1, 4), kind="rebuild"),)))
    rt.truncate(1, 38)


def test_expert_scale_payload_refused_at_load(tiny):
    """A NaN E8M0 scale or an overflowing MXFP4 block is refused by load(), not by the first step
    that routes to it (which would already have written KV)."""
    name = "model.layers.1.mlp.experts.3.down_proj.weight_scale"
    sc = tiny.ckpt.read(name)
    for bad in (255, 254):
        s2 = sc.copy()
        s2[0, 0] = bad
        with pytest.raises(LoadRefused):
            _rt(tiny, 1, 2).load(tiny.ckpt.replaced(name, s2, "U8"))


def test_describe_counts_the_expert_cache(tiny):
    cfg = MimoConfig.from_hf(tiny.config)
    one = CpuRuntime(StageSpec("mimo-tiny", 1, 2), tiny.config, expert_cache=1).describe()
    eight = CpuRuntime(StageSpec("mimo-tiny", 1, 2), tiny.config, expert_cache=8).describe()
    assert eight.layer_weight_bytes[0] - one.layer_weight_bytes[0] == 7 * 3 * cfg.moe_intermediate * cfg.hidden * 2


# ── 2. loader guards ─────────────────────────────────────────────────────────────────────────

def _rt(f, lo, hi, **kw):
    return CpuRuntime(StageSpec("mimo-tiny", lo, hi, **kw), f.config)


def test_router_bias_must_arrive_fp32(tiny):
    name = "model.layers.1.mlp.gate.e_score_correction_bias"
    b = tiny.ckpt.read(name)
    bf = (b.view(np.uint32) >> 16).astype(np.uint16)
    with pytest.raises(LoadRefused, match="float32"):
        _rt(tiny, 1, 3).load(tiny.ckpt.replaced(name, bf, "BF16"))
    with pytest.raises(WeightError, match="float32"):   # a source without dtype_of: the storage says it
        read_tensor(types.SimpleNamespace(read=lambda n: bf), name, "F32", (b.shape[0],))


def test_load_refuses_unknown_missing_and_mistyped(tiny):
    with pytest.raises(LoadRefused, match="not part of the MiMo-V2"):
        _rt(tiny, 1, 3).load(tiny.ckpt.replaced("model.layers.2.self_attn.q_norm.weight",
                                                np.zeros(48, np.uint16), "BF16"))
    _rt(tiny, 3, 5).load(                                    # layer 2's extra tensor is outside [3, 5)
        tiny.ckpt.replaced("model.layers.2.self_attn.q_norm.weight", np.zeros(48, np.uint16), "BF16"))
    with pytest.raises(LoadRefused, match="lacks"):
        _rt(tiny, 0, 2, is_head=True).load(tiny.ckpt.without("model.embed_tokens.weight"))
    qkv = "model.layers.1.self_attn.qkv_proj.weight"
    with pytest.raises(LoadRefused, match="F8_E4M3"):
        _rt(tiny, 1, 2).load(tiny.ckpt.replaced(qkv, tiny.ckpt.read(qkv), "U8"))
    scale = "model.layers.0.self_attn.qkv_proj.weight_scale_inv"      # the 6-row per-chunk grid, cut to
    with pytest.raises(LoadRefused):                                   # the naive plain grid's 5 rows
        _rt(tiny, 0, 1).load(tiny.ckpt.replaced(scale, tiny.ckpt.read(scale)[:5], "F32"))
    fc = CpuRuntime(StageSpec("mimo-tiny", 0, 4, is_head=True, taps=(0, 3, 7), tap_mode="fc_partial"), tiny.config)
    with pytest.raises(LoadRefused, match="fc.weight"):
        fc.load(tiny.ckpt.without("fc.weight"))


def test_config_refusals(tiny):
    for patch in ({"n_group": 2}, {"n_shared_experts": 1}, {"attention_projection_layout": "split"},
                  {"tie_word_embeddings": True}, {"hidden_act": "gelu"},
                  {"quantization_config": {**tiny.config["quantization_config"], "store_dtype": "nvfp4"}}):
        with pytest.raises(BadRequest):
            CpuRuntime(StageSpec("mimo-tiny", 0, 2), {**tiny.config, **patch})
    with pytest.raises(BadRequest):
        CpuRuntime(StageSpec("mimo-tiny", 0, 2, numerics="fast"), tiny.config)


def test_adapter_is_device_neutral():
    """The config + tensor plan import and run with torch and mlx blocked (an MLX or CUDA backend
    shares them), and its MXFP4 block constant is the formats module's."""
    assert adapter.MX_BLOCK == MX_BLOCK
    import pathlib
    import subprocess
    import sys
    root = pathlib.Path(__file__).resolve().parent.parent
    code = ("import sys; sys.modules['torch'] = None; sys.modules['mlx'] = None\n"
            f"sys.path[:0] = [{str(root)!r}, {str(root / 'tests')!r}]\n"
            "from shard.runtime.adapters.mimo_v2 import MimoConfig, layer_plan, edge_plan\n"
            "from shard.runtime.weights import SafetensorsSource\n"
            "import json; c = json.loads(sys.stdin.read()); m = MimoConfig.from_hf(c)\n"
            "print(len(layer_plan(m, 1)), len(edge_plan(m, head=True, tail=True)))\n")
    r = subprocess.run([sys.executable, "-c", code], input=__import__("json").dumps(tiny_config()),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split() == ["56", "3"]   # 5 attn/norm + sink + router w/bias + 8 experts x 3 x 2


def test_describe_is_device_neutral(tiny):
    rt = tiny.make(0, 8, is_head=True, is_tail=True)
    cap = rt.describe()
    assert cap.device.backend == "cpu" and cap.device.format_support("mxfp4") == "emulated"
    assert cap.device.format_support("fp8_e4m3") == "emulated" and cap.device.format_support("fp32") == "hw"
    assert cap.wire_dtype == "bf16" and cap.kv_dtype == "bf16" and cap.logits_dtype == "f32"
    assert cap.layer_kinds == ("global", "swa", "swa", "swa", "swa", "global", "swa", "swa")
    assert cap.rebuild_positions == (tiny.window - 1) * 6
    # KV geometry: global 2 kv heads x (48 + 32), SWA 4 x (48 + 32), bf16
    assert cap.global_kv_bytes_per_pos == 2 * 2 * 2 * 80 and cap.swa_kv_bytes_per_pos == 6 * 2 * 4 * 80
    other = tiny.make(0, 8, is_head=True, is_tail=True)
    assert other.describe().numerics_id == cap.numerics_id
    tiny2 = MimoFactory(window=8, W=4, seed=1)
    assert tiny2.make(0, 8, is_head=True, is_tail=True).describe().numerics_id != cap.numerics_id


def test_sampler_is_a_function_of_seed_and_position():
    lg = np.linspace(0, 1, 50, dtype=np.float32)
    sp = Sampling(seed=7, temperature=1.0, top_k=10)
    assert draw(lg, sp, 5) == draw(lg.copy(), sp, 5)
    assert draw(lg, None, 0) == 49 and draw(np.zeros(4, np.float32), None, 0) == 0   # lowest id on ties
    got = {draw(lg, sp, p) for p in range(300)}
    assert len(got) > 5 and got <= set(range(40, 50))                                  # the top_k cut
    got = {draw(lg, Sampling(seed=7, temperature=1.0, top_p=0.05), p) for p in range(300)}
    assert 1 < len(got) and got <= set(range(40, 50))                                  # the top_p cut
    assert {draw(lg, Sampling(seed=s, temperature=1.0), 3) for s in range(40)} != {draw(lg, sp, 3)}
    assert 0 <= uniform01(1 << 70, -1) < 1


# ── 3. the conformance suite ─────────────────────────────────────────────────────────────────

FACTORIES = {"w6W3": (6, 3), "w3W8": (3, 8)}


@pytest.fixture(scope="module", params=list(FACTORIES))
def factory(request):
    w, W = FACTORIES[request.param]
    return MimoFactory(window=w, W=W)


@pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.__name__)
def test_cpu_runtime_conforms(check, factory):
    try:
        check(factory)
    except Skip as e:
        raise AssertionError(f"unexpected skip on the MiMo CPU runtime: {e}") from e


# ── 4. the production geometry: w = 128, W = 64 ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def prod():
    f = MimoFactory(window=128, W=64, pattern="GSSSSG", taps=(0, 5), max_ctx=1200)
    f.roles = [(0, 3, True, False), (2, 6, False, True)]          # G S S | S S S G: both kinds per span
    return f


def test_production_window_rollback_is_bit_identical(prod):
    """check_swa_retention at w=128, W=64: a whole W rejected and re-fed across the window edge,
    kept rows and the continuation equal a twin with other rejected values AND a clean run."""
    check_swa_retention(prod)


def test_production_window_spec_history_matches_reference(prod):
    """A speculative history of 300+ rows (every rejection pattern of _spec_script) through the
    span G S S: every output row and the exported global KV equal the independent reference."""
    for lo, hi, head, tail in prod.roles:
        ops, true, c = _spec_script(prod, head, seed=5, prefix=2 * prod.window + 20, rounds=9, W=prod.W)
        got = {}
        for name, r in (("rt", prod.make(lo, hi, is_head=head, is_tail=tail)),
                        ("ref", prod.reference(lo, hi, is_head=head, is_tail=tail))):
            r.open(1, prod.max_ctx, prod.W)
            outs = _play(r, 1, ops) + _continue(r, 1, true, c, prod.W, rounds=1)
            got[name] = (_accepted(outs), r.export_state(1, 0, c))
        assert got["rt"][0] == got["ref"][0]
        assert got["rt"][1] == dataclasses.replace(got["ref"][1], numerics_id=got["rt"][1].numerics_id)


def test_production_window_partition_invariance(prod):
    """Layers [0,6) as one runtime == [0,2) + [2,6) == [0,4) + [4,6), 160 tokens (past the window)
    fed in uneven chunks: identical bytes (the canonical residual rule at the cut)."""
    ids = prod.ids(160, 9)
    sched = [(0, 64), (64, 1), (65, 63), (128, 31), (159, 1)]

    def chain(cuts):
        b = [0, *cuts, 6]
        rts = []
        for i in range(len(b) - 1):
            r = prod.make(b[i], b[i + 1], is_head=i == 0, is_tail=i == len(b) - 2)
            r.open(1, 256, prod.W)
            r.commit(1, 160)
            rts.append(r)
        outs = []
        for s, n in sched:
            x = ids.slice(s, s + n)
            for r in rts:
                o = r.step(StepIn((Segment(1, s, n, x, kind="prefill", logits="all" if r is rts[-1] else "none"),))).outs[0]
                x = o.hidden
            outs.append((o.hidden, o.logits))
        return outs

    whole = chain([])
    for cuts in ([2], [4], [1, 3, 5]):
        assert chain(cuts) == whole, cuts


def test_bf16_values_helper():
    r = Rows("bf16", np.array([[0x3F80, 0xBF80]], dtype=np.uint16))
    assert bf16_values(r).tolist() == [[1.0, -1.0]]
