"""K3's MXFP4 routed experts on a GPU, through vLLM's Marlin path -- the M1 `quant_config()` TODO.

K3 ships PACKED: `mxfp4-pack-quantized`, group_size 32, num_bits 4, E8M0 scales, and there is no
bf16 upstream to fall back to. M1 therefore refused a packed expert outright (a stage that skipped
them would serve random-init experts behind a valid receipt). This module is the other half: it
hands the routed experts to vLLM, which on sm_120 resolves the format to
CompressedTensorsW4A4Mxfp4MoEMethod -> MarlinExperts -- a REPACK, not a bf16 dequant, so the 4x
memory blow-up cannot happen by accident (the EMULATION backend that would materialize bf16 every
forward is not in the CUDA priority list at all).

WHAT IS AND IS NOT RENTED
Only `KimiSparseMoeBlock.moe_infer` is replaced. The router (`KimiMoEGate`, incl. the grouped
noaux_tc branch and the renormalize/scaling), the latent down/up projections, `routed_expert_norm`
and the shared experts all stay the reference's own code on their own bf16 weights. That is
deliberate: G0's probe stood in a plain sigmoid+bias top-16 router and measured shapes, which is
fine for timing and wrong for output. Here the reference decides WHICH experts and with WHAT
weights; vLLM only executes the two 4-bit GEMMs.

THE ACTIVATION IS NOT NEGOTIABLE
K3's `hidden_act` is `situ` -- beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta) -- and
vLLM's fused MoE knows silu/gelu/swigluoai and nothing else. Running the experts under SiLU is the
same class of bug as dropping `block_residual` at a hop: it runs, it is fast, and it is silently a
different model. So `situ` is installed into the ONE place the Marlin path applies an activation
(see `_install_situ`), and if that install cannot be verified this module REFUSES to hand back a
backend rather than quietly substituting SiLU. G0 substituted SiLU for timing; that substitution
does not survive into this file.

Knobs: K3_MOE_BACKEND (auto|marlin|none), K3_MOE_TP (informational; 1 only today).
"""
import contextlib
import os

import torch

# auto -> marlin when the checkpoint is packed AND vLLM + a CUDA device are present, else none.
# "none" is M1's behaviour verbatim (bf16 experts, packed tensors refused by Stage.load), which is
# what a CPU box and every existing parity test resolve to.
K3_MOE_BACKEND = os.environ.get("K3_MOE_BACKEND", "auto")

_SUPPORTED_FORMAT = "mxfp4-pack-quantized"
# The per-expert tensor suffixes this loader consumes. Anything else under an expert is a format we
# have not read -- raise rather than load a partial expert (m25's silent-skip is the failure mode
# M1's packed-tensor refusal exists to prevent).
_EXPERT_SUFFIXES = ("weight_packed", "weight_scale")

_CTX = None


# ── the checkpoint's quantization block ──────────────────────────────────────────────────────────

def spec(qc):
    """Validate + normalize a checkpoint `quantization_config` into the facts this module needs.

    Pure dict work: no torch, no vLLM, no GPU -- so the format contract is testable in CI. Raises on
    anything we have not actually run, because the alternative is a backend that silently mis-reads
    a future K3 respin's packing and produces fluent garbage.

    Returns {"format", "group_size", "num_bits", "symmetric", "ignore", "targets"}."""
    if not qc:
        raise ValueError("k3 moe: no quantization_config in the checkpoint — nothing to resolve")
    fmt = qc.get("format")
    if fmt != _SUPPORTED_FORMAT:
        raise NotImplementedError(
            f"k3 moe: quantization format {fmt!r} — this backend was proven against "
            f"{_SUPPORTED_FORMAT!r} only (G0 2026-07-28, sm_120)")
    groups = qc.get("config_groups") or {}
    if not groups:
        raise ValueError("k3 moe: mxfp4 quantization_config carries no config_groups")
    # K3 ships ONE group covering the routed experts. Take its weight spec and require every group
    # to agree, so a mixed-precision respin cannot be flattened into one wrong number.
    seen = []
    for g in groups.values():
        w = g.get("weights") or {}
        seen.append((w.get("group_size"), w.get("num_bits"), bool(w.get("symmetric", True)),
                     w.get("strategy")))
    if len(set(seen)) != 1:
        raise NotImplementedError(f"k3 moe: config_groups disagree on the weight spec: {sorted(set(seen))}")
    group_size, num_bits, symmetric, strategy = seen[0]
    if (group_size, num_bits) != (32, 4):
        raise NotImplementedError(
            f"k3 moe: group_size {group_size} num_bits {num_bits} — proven at (32, 4) only")
    if strategy not in (None, "group"):
        raise NotImplementedError(f"k3 moe: quantization strategy {strategy!r} — proven at 'group' only")
    targets = sorted({t for g in groups.values() for t in (g.get("targets") or [])})
    return {"format": fmt, "group_size": group_size, "num_bits": num_bits,
            "symmetric": symmetric, "ignore": sorted(qc.get("ignore") or []), "targets": targets}


def routed_experts_only(sp):
    """Does this spec quantize ONLY the routed experts?

    K3's ignore list excludes `self_attn`, `shared_experts`, `mlp.*_proj`, `lm_head` and the vision
    tower, so everything outside `block_sparse_moe.experts` stays bf16 and loads through the
    reference's own modules. If a respin ever 4-bits something else, Stage.load's packed-tensor
    refusal fires on it and this returns False -- both loud, neither silent."""
    return all(("experts" in t) or ("block_sparse_moe" in t) for t in sp["targets"])


def backend(qc):
    """Which routed-expert backend this box + checkpoint resolve to under K3_MOE_BACKEND."""
    if K3_MOE_BACKEND == "none":
        return "none"
    if K3_MOE_BACKEND not in ("auto", "marlin"):
        raise RuntimeError(f"K3_MOE_BACKEND={K3_MOE_BACKEND!r} — expected auto, marlin or none")
    if not qc:
        if K3_MOE_BACKEND == "marlin":
            raise RuntimeError("K3_MOE_BACKEND=marlin but the checkpoint carries no quantization_config")
        return "none"
    spec(qc)                                     # raises on a format we have not run
    if K3_MOE_BACKEND == "marlin":
        return "marlin"
    if not torch.cuda.is_available():
        return "none"
    try:
        import vllm                              # noqa: F401
    except ImportError:
        return "none"
    return "marlin"


# ── keeping 896 bf16 experts from ever being allocated ───────────────────────────────────────────

@contextlib.contextmanager
def hollow_experts(M):
    """Build `KimiBlockSparseMLP` on the META device for the duration of a layer construction.

    K3 has 896 routed experts per layer at moe_intermediate 3072 over a 3584 latent. Letting the
    reference materialize them costs 58.6 GiB of bf16 per layer -- an instant OOM on any card, and
    on a CPU box an instant OOM of a different colour. The MXFP4 tensors are what we actually load,
    so the reference's expert modules are never used at all; this makes their construction free and
    `Stage` drops the ModuleList immediately afterwards (meta parameters cannot be `.to()`d, so they
    must not survive construction).

    Scoped to the `with` block and restored on the way out, so a Stage built without the backend --
    every CPU parity test -- constructs the real experts exactly as M1 did."""
    orig = M.KimiBlockSparseMLP.__init__

    def _init(self, config, hidden_size=None, intermediate_size=None):
        with torch.device("meta"):
            orig(self, config, hidden_size=hidden_size, intermediate_size=intermediate_size)

    M.KimiBlockSparseMLP.__init__ = _init
    try:
        yield
    finally:
        M.KimiBlockSparseMLP.__init__ = orig


# ── the vLLM side ────────────────────────────────────────────────────────────────────────────────

def vllm_ctx():
    """Process-wide vLLM config + distributed + workspace, entered once (m25_stage.vllm_ctx's shape).

    vLLM's standalone MoE needs all four of set_current_vllm_config -> init_distributed_environment
    -> initialize_model_parallel -> init_workspace_manager before a RoutedExperts can be built; the
    fourth is the one that is easy to miss and fails deep inside the kernel launch."""
    global _CTX
    if _CTX is not None:
        return _CTX[1]
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.v1.worker.workspace import init_workspace_manager
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", os.environ.get("K3_MOE_PORT", "29557"))
    torch.cuda.set_device(0)
    vcfg = VllmConfig()
    ctx = set_current_vllm_config(vcfg)
    ctx.__enter__()                              # never exited: the experts read it for the process's life
    init_distributed_environment(world_size=1, rank=0, distributed_init_method="env://",
                                 local_rank=0, backend="nccl")
    initialize_model_parallel(1, 1)
    init_workspace_manager(torch.device("cuda"))
    _CTX = (ctx, vcfg)
    return vcfg


def _situ_and_mul(out, x, beta, linear_beta):
    """K3's `situ`, gate-and-mul shaped: x is [..., 2N] = cat(gate, up), out is [..., N].

    Transcribed from the reference's SituAndMul.forward (kimi_k3_ref/modeling_kimi_linear.py:75) --
    fp32 internally, cast back at the end, `linear_beta=None` leaves `up` alone. This is the exact
    function the reference applies inside every routed expert; the whole point of installing it is
    that vLLM would otherwise apply SiLU."""
    d = x.shape[-1] // 2
    gate = x[..., :d].float()
    up = x[..., d:].float()
    a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    out.copy_((a * up).to(out.dtype))


def fused_experts_of(quant_method):
    """The experts object the Marlin kernels actually dispatch through, or None.

    vLLM 0.26 builds it lazily: CompressedTensorsW4A4Mxfp4MoEMethod picks `experts_cls` at __init__
    (MarlinExperts on sm_120, once the two SM100+ MXFP4 backends fail is_supported_config) but does
    not instantiate until process_weights_after_loading -> make_mxfp4_moe_kernel -> FusedMoEKernel,
    which parks it on `.impl.fused_experts`. `.moe_kernel.fused_experts` is a read-only property over
    the same object, so the assignable path is through `.impl`. The other spellings are older/newer
    layouts -- probed in order rather than pinned, because the caller REFUSES on None and a wrong
    guess here cannot end in a silent SiLU."""
    for path in (("moe_kernel", "impl", "fused_experts"),
                 ("moe_kernel", "fused_experts"),
                 ("fused_experts",)):
        obj = quant_method
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def _install_situ(experts, beta, linear_beta):
    """Make the resolved fused-expert object apply K3's `situ` instead of SiLU.

    vLLM's modular MoE applies the activation through ONE overridable method on the experts object,
    `FusedMoEExpertsModular.activation(self, activation, output, input, *, clamp_limit, alpha, beta,
    topk_ids, expert_map)`, and MarlinExperts.apply passes `activation_func=self.activation` down to
    `_fused_marlin_moe`, which calls it in PYTHON between the two 4-bit GEMMs. Binding a replacement
    onto the instance is therefore the whole install -- no vLLM fork, no kernel patch, and it depends
    on nothing but the attribute name (stable from 0.20 through main).

    The MoEActivation enum stays SILU on purpose. Every structural decision downstream reads it --
    is_gated -> w13 holds two shards, adjust_N_for_activation -> N//2, _supports_activation -> True,
    and the backend oracle still picks MARLIN. Registering a new enum member would have to be
    threaded through four more places for no gain, since the value is never used once the callback
    is ours. `input` is [M*topk, 2N] packed gate-then-up, which is exactly SituAndMul's layout.

    Returns True once the bound method is in place AND computes situ on a probe tensor. What that
    CANNOT prove is that vLLM still calls it -- a version that moved the activation into the kernel,
    or onto a module-level function, would leave this install correct and inert, and SiLU would run.
    `MarlinRoutedExperts.arm` closes that gap by counting the callback through one real forward;
    this half only has to be sure the seat exists."""
    if not callable(getattr(experts, "activation", None)):
        return False

    def activation(self, activation_name, output, input, **kw):   # noqa: A002 (vLLM's parameter name)
        _situ_and_mul(output, input, beta, linear_beta)

    try:
        experts.activation = activation.__get__(experts, type(experts))
    except AttributeError:                                        # __slots__, or a read-only property
        return False
    x = torch.tensor([[1.0, -2.0, 3.0, 0.5]])
    out = torch.zeros(1, 2)
    try:
        experts.activation("silu", out, x)
    except TypeError:
        return False
    want = torch.zeros(1, 2)
    _situ_and_mul(want, x, beta, linear_beta)
    return torch.allclose(out, want)


class MarlinRoutedExperts:
    """One layer's 896 MXFP4 routed experts, executed by vLLM's Marlin kernels.

    Owns nothing else: the caller (Stage) keeps the reference's gate, latent projections, norm and
    shared experts, and calls this only where `moe_infer` used to be."""

    def __init__(self, cfg, qc, device="cuda"):
        from vllm.model_executor.layers.fused_moe import FusedMoEConfig, RoutedExperts
        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEParallelConfig, MoEActivation, RoutingMethodType)
        from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig)
        from kimi_k3_ref.modeling_kimi_linear import _get_situ_activation_params

        vcfg = vllm_ctx()
        self.cfg = cfg
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_token
        self.max_tokens = int(os.environ.get("K3_MOE_MAX_TOKENS", "64"))
        latent = self.latent = getattr(cfg, "routed_expert_hidden_size", None) or cfg.hidden_size
        ct = CompressedTensorsConfig.from_config(qc)
        pc = FusedMoEParallelConfig.make(tp_size_=1, dp_size_=1, pcp_size_=1, sp_size_=1,
                                         vllm_parallel_config=vcfg.parallel_config)
        # The routing fields below (renormalize / grouped topk / scoring_func / scaling) are INERT:
        # forward_modular() is handed topk weights+ids that the reference's own KimiMoEGate already
        # produced, so vLLM never runs select_experts. They are set to the checkpoint's values
        # anyway so a future non-modular call cannot silently route differently. MoEActivation.SILU
        # is likewise inert -- _install_situ overrides the activation the kernels actually apply,
        # and __init__ refuses to return without that install succeeding.
        mc = FusedMoEConfig(
            num_experts=self.num_experts, experts_per_token=self.top_k, hidden_dim=latent,
            intermediate_size=cfg.moe_intermediate_size, num_local_experts=self.num_experts,
            num_logical_experts=self.num_experts, activation=MoEActivation.SILU,
            device=torch.device(device), moe_parallel_config=pc, in_dtype=torch.bfloat16,
            routing_method=RoutingMethodType.DeepSeekV3,
            intermediate_size_per_partition=cfg.moe_intermediate_size,
            max_num_tokens=self.max_tokens)
        emm = ExpertMapManager(max_num_batched_tokens=self.max_tokens, top_k=self.top_k,
                               global_num_experts=self.num_experts, num_redundant_experts=0,
                               num_expert_group=getattr(cfg, "num_expert_group", 1) or 1,
                               moe_parallel_config=pc, placement_strategy="linear", enable_eplb=False)
        with torch.device(device):
            self.experts = RoutedExperts(
                layer_name="k3.routed_experts", params_dtype=torch.bfloat16, moe_config=mc,
                quant_config=ct, expert_map_manager=emm,
                ckpt_gate_proj_name="w1", ckpt_down_proj_name="w2", ckpt_up_proj_name="w3",
                renormalize=bool(getattr(cfg, "moe_renormalize", True)), use_grouped_topk=True,
                num_expert_group=getattr(cfg, "num_expert_group", 1) or 1,
                topk_group=getattr(cfg, "topk_group", 1) or 1, scoring_func="sigmoid",
                routed_scaling_factor=float(getattr(cfg, "routed_scaling_factor", 1.0)))
        self.quant_method = (getattr(self.experts, "quant_method", None)
                             or getattr(self.experts, "_quant_method", None))
        self._loaded = False
        self._situ = (_get_situ_activation_params(cfg) if cfg.hidden_act == "situ" else None)

    # ---- weights ----

    def load(self, get_tensor, prefix):
        """Stream this layer's expert tensors into the vLLM parameter buffers, then repack.

        `get_tensor(name) -> cpu tensor`, `prefix` = the checkpoint path of the MoE block, e.g.
        `language_model.model.layers.46.block_sparse_moe`. Streamed one expert at a time and
        repacked at the END of this layer -- G0 measured the repack as PER-TENSOR, so the peak is
        (layer resident + the largest single tensor) = 25.04 GiB, not 2x the layer; loading every
        layer first and repacking after would break that and cost a card."""
        params = dict(self.experts.named_parameters())
        w13, w2 = params["w13_weight_packed"], params["w2_weight_packed"]
        w13s, w2s = params["w13_weight_scale"], params["w2_weight_scale"]
        dv = w13.device
        with torch.no_grad():
            for e in range(self.num_experts):
                b = f"{prefix}.experts.{e}"
                w1p = get_tensor(f"{b}.w1.weight_packed")
                inter = w1p.shape[0]                       # the w1 half of the fused w13 buffer
                w13[e, :inter].copy_(w1p.to(dv))
                w13[e, inter:].copy_(get_tensor(f"{b}.w3.weight_packed").to(dv))
                w2[e].copy_(get_tensor(f"{b}.w2.weight_packed").to(dv))
                w13s[e, :inter].copy_(get_tensor(f"{b}.w1.weight_scale").to(dv))
                w13s[e, inter:].copy_(get_tensor(f"{b}.w3.weight_scale").to(dv))
                w2s[e].copy_(get_tensor(f"{b}.w2.weight_scale").to(dv))
        del params, w13, w2, w13s, w2s      # drop OUR references before the repack: a live handle on
        import gc                           # a pre-repack tensor keeps it resident and adds ~5.7 GiB
        gc.collect()                        # to the peak (the G0 harness's own 30.7-GiB false reading)
        torch.cuda.empty_cache()
        self.quant_method.process_weights_after_loading(self.experts)
        self._loaded = True
        return self

    def arm(self):
        """Install `situ` on the RESOLVED fused-expert object and refuse if it did not take.

        Has to run after process_weights_after_loading: the experts object the kernels dispatch
        through is chosen during weight processing (the SM100+ MXFP4 backends fail is_supported_config
        on sm_120 and it falls through to MarlinExperts), so before that there is nothing to bind."""
        if self._situ is None:                             # a config whose hidden_act is not situ
            return self
        beta, linear_beta = self._situ
        experts = fused_experts_of(self.quant_method)
        self.fused_experts = experts
        if experts is None or not _install_situ(experts, beta, linear_beta):
            raise RuntimeError(
                "k3 moe: could not install K3's `situ` activation on vLLM's fused experts "
                f"(resolved {type(experts).__name__ if experts is not None else 'nothing'}). "
                "Refusing the marlin backend: running these experts under vLLM's SiLU is a DIFFERENT "
                "MODEL, not a slower one. Re-check FusedMoEExpertsModular.activation in the installed "
                "vLLM, or set K3_MOE_BACKEND=none and serve bf16 experts.")
        self._prove_the_route()
        return self

    def _prove_the_route(self):
        """Run one token through the real experts and count the activation callback.

        The install can be perfectly correct and completely inert: if a vLLM version stopped calling
        `self.activation` -- fused it into the kernel, moved it to a module function -- the bound
        method sits there unused and every expert quietly runs SiLU. Nothing about the output shape,
        the speed or the memory would change. So the callback is wrapped in a counter for exactly
        one forward_modular, and a zero count refuses the backend.

        The probe costs one MoE forward (G0: 0.223 ms at B=1) and runs on uninitialised-but-loaded
        weights, so it asserts nothing about the numbers -- only about the call graph."""
        experts = self.fused_experts
        inner = experts.activation
        calls = []

        def counting(*a, **kw):
            calls.append(1)
            return inner(*a, **kw)

        experts.activation = counting
        try:
            x = torch.zeros(1, self.latent, dtype=torch.bfloat16, device="cuda")
            tw = torch.full((1, self.top_k), 1.0 / self.top_k, dtype=torch.float32, device="cuda")
            ti = torch.arange(self.top_k, dtype=torch.int32, device="cuda").view(1, -1)
            self.experts.forward_modular(x, tw, ti)
        finally:
            experts.activation = inner
        if not calls:
            raise RuntimeError(
                "k3 moe: `situ` was installed on "
                f"{type(experts).__name__}.activation but vLLM never called it during a real "
                "forward — the experts are running vLLM's own activation (SiLU), which is a "
                "DIFFERENT MODEL. Refusing the marlin backend; set K3_MOE_BACKEND=none to serve "
                "bf16 experts, or re-find the activation hook in the installed vLLM.")

    # ---- the seam ----

    def moe_infer(self, x, topk_ids, topk_weight):
        """Drop-in for KimiSparseMoeBlock.moe_infer: [T, latent] -> [T, latent].

        The reference's version sorts tokens by expert and loops in Python (with a `.cpu().numpy()`
        host sync that alone makes a graph capture impossible). vLLM's modular entry point takes the
        router's decision as-is: fp32 weights, int32 ids, both contiguous."""
        return self.experts.forward_modular(x, topk_weight.float().contiguous(),
                                            topk_ids.int().contiguous())
