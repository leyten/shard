"""Kimi-K3 stage, milestone 1: a layer range is bit-identical whether it runs whole or split.

The headline is test_split_across_stages_is_bit_identical. K3's decoder passes TWO tensors between
layers -- a running prefix sum and an AttnRes `block_residual` stack -- so a stage boundary that
carries only the hidden state still runs, still emits plausible logits, and is silently wrong. That
is not a hypothetical: test_dropping_block_residual_at_a_boundary_diverges is the same split with the
stack dropped at the hop, and it is the red the green is measured against.

Everything runs on CPU against a 4-layer random-init K3 (3 KDA + 1 MLA, 2 AttnRes blocks, one dense
layer and three latent-MoE layers) -- no GPU, no network, no 15.8 GiB layer downloads.

Run: python3 -m pytest tests/test_k3_stage.py -q
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
pytest.importorskip("einops")                      # the vendored reference imports it at module scope
safetensors_torch = pytest.importorskip("safetensors.torch")
K3 = pytest.importorskip("k3_stage")               # installs the CPU KDA backend on first ref() call

PROMPT = [3, 9, 17, 2, 41]
NEW = 4


def _tiny_config():
    """A 4-layer K3 that exercises every structure the real 93-layer one has, at toy dims.

    Layer roles follow the real config's own rules: `kda_layers` is 1-BASED (is_kda_layer checks
    `layer_idx + 1`), so [1,2,3] makes layers 0-2 KDA and layer 3 MLA; first_k_dense_replace=1 makes
    layer 0 a dense MLP and 1-3 latent MoE; attn_res_block_size=2 appends an AttnRes block at layers
    0 and 2, i.e. the payload ends at 2 blocks + the prefix sum, the shape the real one reaches at 8.
    `gate_lower_bound` is set because K3 sets it (-5.0) -- that selects the sigmoid gate branch, not
    the softplus one, so the default path here is the path the checkpoint actually takes."""
    K3.ref()
    from kimi_k3_ref.configuration_kimi_k3 import KimiLinearConfig
    return KimiLinearConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4, hidden_act="situ",
        activation_situ_beta=4.0, activation_situ_linear_beta=25.0, rms_norm_eps=1e-5,
        moe_intermediate_size=24, num_experts=6, num_experts_per_token=2, num_shared_experts=1,
        first_k_dense_replace=1, moe_layer_freq=1, routed_expert_hidden_size=16,
        latent_moe_use_norm=True, moe_renormalize=True, routed_scaling_factor=1.0,
        moe_router_activation_func="sigmoid", num_expert_group=1, topk_group=1,
        q_lora_rank=16, kv_lora_rank=8, qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8,
        mla_use_nope=True, mla_use_output_gate=True, attn_res_block_size=2,
        linear_attn_config={"kda_layers": [1, 2, 3], "full_attn_layers": [4], "head_dim": 8,
                            "num_heads": 4, "short_conv_kernel_size": 4,
                            "use_full_rank_gate": True, "gate_lower_bound": -5.0},
        max_position_embeddings=512, tie_word_embeddings=False, pad_token_id=0,
    )


def _adapt_reference_whole_model():
    """Make the reference's WHOLE-MODEL path runnable on whatever transformers is installed.

    A Stage never touches KimiLinearModel -- it drives KimiDecoderLayer directly and builds its own
    mask -- so this adapter exists only so the tests have an official model to measure against, and
    it deliberately lives here rather than in k3_stage.

    It replaces the ONE thing the reference borrows from transformers on that path: the mask helper.
    Doing it by signature-patching was a losing game -- create_causal_mask renamed a kwarg in 5.12
    and by 5.14 its Cache protocol wanted a get_query_offset the vendored KimiDynamicCache has never
    heard of, so the reference model broke on a transformers bump that touched nothing we own.
    Building the mask here instead pins the comparison to the architecture rather than to a release.

    It does mean the reference model is handed the STAGE's mask, so this comparison cannot also
    prove the mask right. That is pinned separately and better, by properties a serving path
    actually depends on: test_causal_mask_is_bottom_right_aligned on the tensor itself, and
    test_kda_state_continuity_across_chunked_prefill, which a top-left-aligned or off-by-one mask
    cannot survive (every decode step would attend to the wrong span)."""
    M = K3.ref()

    def _mask(**kw):
        emb = kw["input_embeds"]
        start, q = int(kw["cache_position"][0]), emb.shape[1]
        return K3._causal_mask(q, start + q, start, emb.dtype, emb.device)
    M.create_causal_mask = _mask
    return M


def _seed_unreached_parameters(model):
    """Seed the parameters Moonshot's own initializer does not reach.

    KimiPreTrainedModel._init_weights handles nn.Linear and nn.Embedding only, but
    KimiDeltaAttention declares `dt_bias` as a bare nn.Parameter(torch.empty(...)) and never
    initializes it. torch.manual_seed cannot reach raw heap, so a random-init fixture gets whatever
    was in memory: across ten builds at the SAME seed the absmax ran 0.0, 0.1, 3.4e11, ... 1.9e36,
    and one build contained a NaN, which took the whole suite red (torch.equal(nan, nan) is False)
    about 3% of runs. A real checkpoint always overwrites dt_bias with a loaded tensor, so this bites
    fixtures only -- but a headline parity test that is randomly red proves nothing, so seed it.

    A_log needs nothing: the reference builds it as log(empty.uniform_(1, 16)), which IS seeded."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("dt_bias"):
                p.normal_(0.0, 0.02)


@pytest.fixture(scope="module")
def fixture():
    """(reference module, config, reference model). Random-init, seeded, built once."""
    M = _adapt_reference_whole_model()
    cfg = _tiny_config()
    torch.manual_seed(1234)
    model = M.KimiLinearForCausalLM(cfg).eval()
    _seed_unreached_parameters(model)
    assert all(p.isfinite().all() for p in model.parameters()), "fixture built a non-finite weight"
    # KimiLinearModel.__init__ overwrites the requested attention with flash_attention_2 (right on an
    # 8xB300, fatal on CPU). One config object is shared by the model and every Stage below, so
    # pinning it back here pins it for both sides of the parity comparison.
    cfg._attn_implementation = "eager"
    return M, cfg, model


def _reference_decode(M, cfg, model, prompt, n_new):
    """Greedy decode through the official whole model. Returns [n_new+1, vocab] last-token logits."""
    cache = M.KimiDynamicCache(config=cfg)
    ids, out = list(prompt), []
    for _ in range(n_new + 1):
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([ids]), past_key_values=cache, use_cache=True).logits
        out.append(lg[0, -1].clone())
        ids = [int(out[-1].argmax())]
    return torch.stack(out)


def _stages(cfg, model, ranges):
    """Build the stage split, loading each layer out of the reference model's own state dict."""
    sd = model.state_dict()
    stages = []
    for i, (lo, hi) in enumerate(ranges):
        st = K3.Stage(lo, hi, cfg, head=(i == 0), tail=(i == len(ranges) - 1), device="cpu")
        for li in range(lo, hi):
            pre = f"model.layers.{li}."
            st.layers[li - lo].load_state_dict(
                {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, strict=True)
        if st.head:
            st.embed_tokens.load_state_dict({"weight": sd["model.embed_tokens.weight"]})
        if st.tail:
            st.norm.load_state_dict({"weight": sd["model.norm.weight"]})
            st.lm_head.load_state_dict({"weight": sd["lm_head.weight"]})
            st.output_attn_res_norm.load_state_dict(
                {"weight": sd["model.output_attn_res_norm.weight"]})
            st.output_attn_res_proj.load_state_dict(
                {"weight": sd["model.output_attn_res_proj.weight"]})
        stages.append(st)
    return stages


def _stage_decode(stages, prompt, n_new, drop_block_residual_at=None):
    """Greedy decode across the split, moving (h, block_residual) over every hop.

    `drop_block_residual_at` re-seeds the stack before stage N, i.e. simulates a boundary that
    carried only the hidden state -- the bug this whole file exists to catch."""
    pos, ids, out = 0, list(prompt), []
    for _ in range(n_new + 1):
        h, br = stages[0].embed([ids])
        for i, st in enumerate(stages):
            if i == drop_block_residual_at:
                br = K3.seed_block_residual(h)
            h, br = st.forward(h, br, pos)
        lg = stages[-1].logits(h, br)
        pos += len(ids)
        out.append(lg[0, -1].clone())
        ids = [int(out[-1].argmax())]
    return torch.stack(out)


# ── the contract ─────────────────────────────────────────────────────────────────────────────────

def test_causal_mask_is_bottom_right_aligned():
    """A decode step's query sits at the END of the kv, not the top-left.

    Top-left alignment is the classic version of this bug and it is silent on a cold prefill (where
    the two agree) -- it only corrupts the warm path, which is every token after the first."""
    m = K3._causal_mask(1, 5, 4, torch.float32, "cpu")            # 1 new token, 4 already cached
    assert m.shape == (1, 1, 1, 5)
    assert (m[0, 0, 0] == 0).all(), "a decode step must see the whole kv"

    m = K3._causal_mask(3, 5, 2, torch.float32, "cpu")            # a 3-token chunk at offset 2
    allowed = (m[0, 0] == 0)
    assert allowed.tolist() == [[True, True, True, False, False],
                                [True, True, True, True, False],
                                [True, True, True, True, True]]
    assert m.min() == torch.finfo(torch.float32).min              # finfo.min, never -inf


def test_boundary_payload_is_the_hidden_state_and_the_attnres_stack(fixture):
    """A stage's payload is (h, block_residual), and the stack GROWS as blocks are appended."""
    _, cfg, model = fixture
    st = _stages(cfg, model, [(0, 4)])[0]
    h, br = st.embed([PROMPT])
    assert h.shape == (1, len(PROMPT), cfg.hidden_size)
    assert br.shape == (len(PROMPT), 0, cfg.hidden_size)      # seeded empty, per pass
    h, br = st.forward(h, br, 0)
    # attn_res_block_size 2 over 4 layers -> blocks appended at layer 0 and layer 2
    assert br.shape == (len(PROMPT), 2, cfg.hidden_size)
    assert br.dtype == h.dtype and br.device == h.device      # transport encodes both as-is
    payload = (1 + br.shape[1]) * cfg.hidden_size
    assert payload == 3 * cfg.hidden_size                     # NOT 1x, which is what a plain stage sends


def test_forward_refuses_a_missing_block_residual(fixture):
    """Calling a K3 stage like a plain-transformer stage must say so, not crash somewhere inside."""
    _, cfg, model = fixture
    st = _stages(cfg, model, [(0, 4)])[0]
    h, _ = st.embed([PROMPT])
    with pytest.raises(RuntimeError, match="needs the AttnRes block_residual"):
        st.forward(h, None, 0)


def test_stage_block_matches_the_reference_whole_model(fixture):
    """One stage over every layer reproduces Moonshot's own model, bit for bit.

    This is the anchor: it pins the parts a Stage does NOT rent -- its causal mask, the empty stack it
    seeds, and the order of output-AttnRes / final norm / lm_head -- against the official forward."""
    M, cfg, model = fixture
    ref = _reference_decode(M, cfg, model, PROMPT, NEW)
    got = _stage_decode(_stages(cfg, model, [(0, 4)]), PROMPT, NEW)
    assert torch.equal(got, ref), f"max |diff| {(got - ref).abs().max().item():.3e}"


@pytest.mark.parametrize("ranges", [
    [(0, 2), (2, 4)],                 # split inside a KDA run, and on an AttnRes append boundary
    [(0, 1), (1, 3), (3, 4)],         # 3 stages: dense head, KDA middle, MLA tail
    [(0, 3), (3, 4)],                 # split exactly at the KDA -> MLA transition
])
def test_split_across_stages_is_bit_identical(fixture, ranges):
    """THE milestone: the same layers, split, decoding the same tokens -- bit-identical logits.

    Every hop moves (h, block_residual) as tensors; the MLA KV and the KDA recurrent + conv state
    stay resident in the stage that owns those layers and never cross."""
    M, cfg, model = fixture
    ref = _reference_decode(M, cfg, model, PROMPT, NEW)
    got = _stage_decode(_stages(cfg, model, ranges), PROMPT, NEW)
    assert torch.equal(got, ref), f"{ranges}: max |diff| {(got - ref).abs().max().item():.3e}"


def test_dropping_block_residual_at_a_boundary_diverges(fixture):
    """The red the green is measured against: forget the stack at one hop and the answer changes.

    Note what does NOT happen -- no shape error, no exception. The tail's _apply_attn_res softmaxes
    over whatever blocks it was given, so a stage boundary that carries only `h` produces a fluent,
    wrong model. Nothing but a parity assertion catches it."""
    M, cfg, model = fixture
    ref = _reference_decode(M, cfg, model, PROMPT, NEW)
    ranges = [(0, 2), (2, 4)]
    dropped = _stage_decode(_stages(cfg, model, ranges), PROMPT, NEW, drop_block_residual_at=1)
    assert not torch.equal(dropped, ref)
    assert not torch.equal(dropped.argmax(-1), ref.argmax(-1)), \
        "the dropped-stack run picked the same tokens — this test is not proving anything"


# ── state: reset, start_pos, chunked prefill ─────────────────────────────────────────────────────

def test_warm_stage_after_reset_matches_a_cold_one(fixture):
    """A second job on a warm stage must equal the same job on a fresh one -- reset() drops it all."""
    M, cfg, model = fixture
    ref = _reference_decode(M, cfg, model, PROMPT, NEW)
    stages = _stages(cfg, model, [(0, 2), (2, 4)])
    _stage_decode(stages, [7, 7, 7, 1], NEW)                  # a first, different job
    for st in stages:
        st.reset()
    assert all(st._pos == 0 for st in stages)
    assert torch.equal(_stage_decode(stages, PROMPT, NEW), ref)


def test_kda_state_continuity_across_chunked_prefill(fixture):
    """Prefill split into chunks must leave the same state as one shot -- conv window included.

    The KDA short conv is kernel 4, so a chunk boundary lands inside the window: get the cache
    layout wrong and the first tokens of chunk 2 convolve against zeros instead of the tail of
    chunk 1. Chunk sizes below are chosen to straddle that (2 + 3 with a 4-wide kernel).

    NOT bit-exact, and that is arithmetic rather than a defect: chunking changes the SHAPE of every
    projection GEMM and conv, so the same multiply-adds accumulate in a different order. Re-run in
    float64 and the gap falls from ~1e-7 to 2.8e-17, i.e. it tracks the machine epsilon, not the
    state. The split-stage tests above ARE bit-exact because splitting changes no shape at all --
    only which process owns which layer -- which is precisely why bit-equality is the right bar
    there and the wrong one here. A genuinely dropped conv window moves logits by O(1e-1)."""
    _, cfg, model = fixture
    whole, chunked = _stages(cfg, model, [(0, 4)])[0], _stages(cfg, model, [(0, 4)])[0]
    toks = [5, 11, 3, 60, 2]

    h, br = whole.embed([toks])
    h, br = whole.forward(h, br, 0)
    one_shot = whole.logits(h, br)

    pos = 0
    for chunk in ([5, 11], [3, 60, 2]):
        h, br = chunked.embed([chunk])
        h, br = chunked.forward(h, br, pos)
        pos += len(chunk)
    in_chunks = chunked.logits(h, br)

    assert chunked._pos == whole._pos == len(toks)
    gap = (in_chunks[0, -1] - one_shot[0, -1]).abs().max().item()
    assert gap < 1e-5, f"max |diff| {gap:.3e} — too large to be accumulation order"
    assert int(in_chunks[0, -1].argmax()) == int(one_shot[0, -1].argmax())


def test_rewind_is_refused_when_the_stage_holds_kda(fixture):
    """A KDA recurrent state cannot be cropped, so the stage refuses rather than answering wrong."""
    _, cfg, model = fixture
    st = _stages(cfg, model, [(0, 4)])[0]
    h, br = st.embed([PROMPT])
    st.forward(h, br, 0)
    with pytest.raises(RuntimeError, match="cannot rewind"):
        st.forward(h[:, :1], K3.seed_block_residual(h[:, :1]), 2)


def test_rewind_crops_mla_kv(fixture):
    """An MLA-only stage CAN rewind: cropping its KV to start_pos reproduces the shorter history.

    Same accumulation-order caveat as the chunked-prefill test -- the warm run's first 3 KV rows came
    out of a 5-token projection and the cold run's out of a 3-token one. Exactly 0.0 in float64."""
    _, cfg, model = fixture
    assert not _stages(cfg, model, [(3, 4)])[0].has_kda
    torch.manual_seed(7)
    h = torch.randn(1, 5, cfg.hidden_size)
    br = K3.seed_block_residual(h)

    cold = _stages(cfg, model, [(3, 4)])[0]
    cold.forward(h[:, :3], br[:3], 0)
    want, _ = cold.forward(h[:, 3:], br[3:], 3)

    warm = _stages(cfg, model, [(3, 4)])[0]
    warm.forward(h, br, 0)                                    # consumes all 5
    got, _ = warm.forward(h[:, 3:], br[3:], 3)                # rewind to 3, re-feed the last 2
    assert warm._pos == 5
    assert torch.allclose(got, want, atol=1e-6, rtol=1e-5), \
        f"max |diff| {(got - want).abs().max().item():.3e}"


def test_forward_refuses_a_gap_in_start_pos(fixture):
    """start_pos ahead of what the stage consumed means tokens skipped this block's layers."""
    _, cfg, model = fixture
    st = _stages(cfg, model, [(0, 4)])[0]
    h, br = st.embed([PROMPT])
    with pytest.raises(RuntimeError, match="ahead of"):
        st.forward(h, br, 3)


# ── loading a layer range off a real checkpoint ──────────────────────────────────────────────────

def _write_checkpoint(d, cfg, model, namespace="language_model.model", pack_experts=False):
    """A real (tiny) safetensors checkpoint in K3's own shape: the decoder namespaced under a
    multimodal wrapper, and `A_log` PADDED the way the real one ships it."""
    os.makedirs(d, exist_ok=True)
    outer = namespace.rsplit(".", 1)[0]
    tensors, pad = {}, cfg.linear_attn_config["num_heads"] + 32
    for k, v in model.state_dict().items():
        name = f"{outer}.{k}" if k.startswith("lm_head") else f"{namespace}.{k[len('model.'):]}"
        if k.endswith("self_attn.A_log"):
            v = torch.cat([v, torch.full((pad - v.shape[0],), 99.0)])   # junk past num_heads
        tensors[name] = v.clone().contiguous()
    if pack_experts:
        victim = f"{namespace}.layers.1.block_sparse_moe.experts.0.w1.weight"
        tensors[victim.replace(".weight", ".weight_packed")] = tensors.pop(victim).to(torch.uint8)
    safetensors_torch.save_file(tensors, f"{d}/model.safetensors")
    json.dump({"weight_map": {k: "model.safetensors" for k in tensors}},
              open(f"{d}/model.safetensors.index.json", "w"))
    json.dump({"model_type": "kimi_k3", "text_config": cfg.to_dict()}, open(f"{d}/config.json", "w"))
    return d


def test_load_a_layer_range_from_a_namespaced_checkpoint(tmp_path, fixture):
    """The loader reads the checkpoint's OWN names (shard/weightkeys), and slices the padded A_log.

    K3's decoder lives under `language_model.model.layers.N.` -- no per-model key table here, and a
    stage that loaded correctly must then DECODE identically to the reference."""
    M, cfg, model = fixture
    d = _write_checkpoint(str(tmp_path / "ckpt"), cfg, model)
    ref = _reference_decode(M, cfg, model, PROMPT, NEW)

    stages = [K3.Stage(0, 2, K3.config(d), head=True, device="cpu").load(d),
              K3.Stage(2, 4, K3.config(d), tail=True, device="cpu").load(d)]
    kda = stages[0].layers[0].self_attn
    assert kda.A_log.shape == (cfg.linear_attn_config["num_heads"],)
    assert torch.equal(kda.A_log, model.model.layers[0].self_attn.A_log)   # the junk tail is gone
    assert torch.equal(_stage_decode(stages, PROMPT, NEW), ref)


def test_packed_expert_tensors_are_refused(tmp_path, fixture):
    """M1 materializes bf16; a packed MXFP4 expert must fail loudly, never load as a random init."""
    _, cfg, model = fixture
    d = _write_checkpoint(str(tmp_path / "packed"), cfg, model, pack_experts=True)
    with pytest.raises(NotImplementedError, match="MXFP4"):
        K3.Stage(0, 4, K3.config(d), device="cpu").load(d)


def test_quant_config_reads_the_checkpoints_own_block(tmp_path, fixture):
    """The seam a 4-bit MoE backend resolves through exists and reads config.json."""
    _, cfg, model = fixture
    d = _write_checkpoint(str(tmp_path / "q"), cfg, model)
    assert K3.quant_config(d) is None
    cfgj = json.load(open(f"{d}/config.json"))
    cfgj["quantization_config"] = {"format": "mxfp4-pack-quantized"}
    json.dump(cfgj, open(f"{d}/config.json", "w"))
    assert K3.quant_config(d)["format"] == "mxfp4-pack-quantized"


# ── the CPU backend itself ───────────────────────────────────────────────────────────────────────

def _fla_published_kda(q, k, v, g, beta, A_log, dt_bias, lower_bound, initial_state=None):
    """flash-linear-attention's OWN published reference, transcribed here as the golden formulation.

    Everything above this line runs the reference model and the stage through the same KDA backend
    -- k3_stage.ref() installs k3_kda_cpu as `fla` before importing the reference -- so those tests
    prove state THREADING and are blind to the recurrence itself. Verified by mutation: deleting the
    delta subtraction entirely, dropping the q scale, or flipping the gate sign all survive every
    other test in this file. This is what pins the math.

    Sources, MIT (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li:
      fla/ops/kda/naive.py     naive_recurrent_kda
      fla/ops/kda/gate.py      naive_kda_lowerbound_gate
    plus the preprocessing fla does inside the kernel rather than in naive.py (l2 norm, beta
    sigmoid), read off fla/ops/kda/fused_recurrent.py's kernel body."""
    B, T, H, Kd = q.shape
    V = v.shape[-1]
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q * (Kd ** -0.5)
    g = lower_bound * torch.sigmoid(A_log.view(H, 1).float().exp() * (g + dt_bias.view(H, -1)))
    beta = torch.sigmoid(beta)

    S = q.new_zeros(B, H, Kd, V)
    if initial_state is not None:
        S = S + initial_state
    o = torch.zeros_like(v)
    for i in range(T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum("bhk,bhv->bhkv", b_i[..., None] * k_i,
                             v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum("bhk,bhkv->bhv", q_i, S)
    return o, S


def test_cpu_kda_matches_flas_published_reference():
    """The CPU KDA recurrence, gate and in-kernel preprocessing against fla's own formulation."""
    import k3_kda_cpu
    torch.manual_seed(3)
    B, T, H, Kd, V = 2, 6, 3, 8, 8
    q, k = torch.randn(B, T, H, Kd), torch.randn(B, T, H, Kd)
    v, g = torch.randn(B, T, H, V), torch.randn(B, T, H, Kd)
    beta, A_log, dt_bias = torch.randn(B, T, H), torch.randn(H), torch.randn(H * Kd)

    o, state = k3_kda_cpu.fused_recurrent_kda(
        q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias, output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
        lower_bound=-5.0, transpose_state_layout=True)
    o_ref, s_ref = _fla_published_kda(q, k, v, g, beta, A_log, dt_bias, -5.0)

    assert torch.allclose(o, o_ref, atol=1e-6), f"max |diff| {(o - o_ref).abs().max().item():.3e}"
    # transpose_state_layout is fla's state_v_first: the state comes back [B,H,V,K], not [B,H,K,V]
    assert state.shape == (B, H, V, Kd)
    assert torch.allclose(state.transpose(-1, -2), s_ref, atol=1e-6)


def test_state_v_first_is_accepted_under_both_fla_spellings():
    """fla renamed transpose_state_layout -> state_v_first. A shim that swallowed the new name into
    **kwargs would return [B,H,K,V] where the caller wants [B,H,V,K] -- silently transposed state,
    which is the worst possible M2 handover bug. Both spellings must mean the same thing."""
    import k3_kda_cpu
    torch.manual_seed(5)
    a = dict(q=torch.randn(1, 3, 2, 4), k=torch.randn(1, 3, 2, 4), v=torch.randn(1, 3, 2, 4),
             g=torch.randn(1, 3, 2, 4), beta=torch.randn(1, 3, 2), output_final_state=True)
    _, plain = k3_kda_cpu.fused_recurrent_kda(**a)
    _, old = k3_kda_cpu.fused_recurrent_kda(**a, transpose_state_layout=True)
    _, new = k3_kda_cpu.fused_recurrent_kda(**a, state_v_first=True)
    assert torch.equal(old, new)
    assert torch.equal(old, plain.transpose(-1, -2))


def test_seed_block_residual_flattens_batch_and_sequence():
    """(num_tokens, 0, hidden) -- num_tokens is B*S. At B=1 a bug here is invisible, which is
    exactly why the parity tests above cannot be trusted to catch it."""
    assert K3.seed_block_residual(torch.zeros(3, 5, 7)).shape == (15, 0, 7)



def test_cpu_kda_backend_is_what_a_gpuless_box_resolves_to():
    """No GPU -> the CPU backend, so none of the above needs rented hardware."""
    import k3_kda_cpu
    assert k3_kda_cpu.backend() == ("fla" if torch.cuda.is_available()
                                    and k3_kda_cpu._installed("fla.ops.kda") else "cpu")
    assert k3_kda_cpu.install() in ("cpu", "fla")


def test_a_stray_fla_directory_is_not_mistaken_for_the_package(tmp_path, monkeypatch):
    """PEP-420 turns any `fla/` on sys.path into an importable, empty namespace package -- an
    unpacked wheel next to a script would otherwise shadow the CPU backend with nothing."""
    import k3_kda_cpu
    (tmp_path / "fla").mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("fla", None)
    try:
        assert not k3_kda_cpu._installed("fla.ops.kda")
    finally:
        sys.modules.pop("fla", None)
        k3_kda_cpu.install()


def test_importing_the_stage_needs_no_checkpoint_on_disk(monkeypatch):
    """`import k3_stage` must work on a box with no model -- m25_stage's import-time AutoConfig is
    exactly what forced its --dir pre-parse."""
    monkeypatch.setenv("K3_DIR", "/nonexistent")
    import importlib
    importlib.reload(K3)
    assert K3.K3_DIR == "/nonexistent"
    importlib.reload(K3)
