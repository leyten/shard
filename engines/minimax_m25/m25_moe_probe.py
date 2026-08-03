"""S1 EXISTENTIAL GATE: does MiniMax-M2.5's 256-expert / top-8 sigmoid NVFP4 FusedMoE
EXECUTE on an RTX 5090 (sm_120) without the vllm#35566 illegal-memory crash?

Standalone, one layer's real NVFP4 experts, finite-output check. Mirrors the GLM-proven
path (research/glm_nvfp4_moe.py) but for M2.5: plain top-8 (NO grouped-topk), sigmoid +
per-layer e_score_correction_bias, experts named block_sparse_moe.experts.{e}.w1/w2/w3.

Tries the modelopt loader first (the GLM-proven sm_120 cutlass-fp4 path), falls back to
compressed-tensors. Prints the ACTUAL tensor + param names so the layout is verified.

  python m25_moe_probe.py --dir /root/m25 --layer 30
"""
import os, json, argparse, torch
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29577")
os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("LOCAL_RANK", "0")
from safetensors import safe_open

_CTX = None


def vllm_ctx():
    global _CTX
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.config import VllmConfig, set_current_vllm_config, get_current_vllm_config
    from vllm.v1.worker.workspace import init_workspace_manager
    torch.cuda.set_device(0)
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://", backend="nccl")
    vcfg = VllmConfig()
    try:
        vcfg.kernel_config.moe_backend = "cutlass"  # precompiled cutlass fp4, no flashinfer JIT
    except Exception as e:
        print("warn moe_backend:", e, flush=True)
    _CTX = set_current_vllm_config(vcfg); _CTX.__enter__()  # keep ref alive (GC kills the ctx otherwise)
    try:
        print("[cfg] moe_backend =", get_current_vllm_config().kernel_config.moe_backend, flush=True)
    except Exception:
        pass
    initialize_model_parallel(1); init_workspace_manager(torch.device("cuda"))
    return vcfg


def build_quant_config(DIR):
    """Try modelopt (GLM-proven sm_120 path) then compressed-tensors. Return (method, cfg)."""
    cfgj = json.load(open(f"{DIR}/config.json"))
    qc = cfgj.get("quantization_config")
    hfq = json.load(open(f"{DIR}/hf_quant_config.json")) if os.path.exists(f"{DIR}/hf_quant_config.json") else None
    errs = []
    # 1) modelopt NVFP4 (proven to fire cutlass fp4 on sm_120 in the GLM work)
    try:
        from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
        src = None
        if hfq and "quantization" in hfq and "quant_algo" in hfq["quantization"]:
            src = hfq["quantization"]
        elif qc and "quant_algo" in qc:
            src = qc
        if src is not None:
            return "modelopt", ModelOptNvFp4Config.from_config(src)
        errs.append("modelopt: no quant_algo schema in hf_quant_config/config")
    except Exception as e:
        errs.append(f"modelopt: {type(e).__name__}: {e}")
    # 2) compressed-tensors W4A4-FP4
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
        if qc and "config_groups" in qc:
            return "compressed-tensors", CompressedTensorsConfig.from_config(qc)
        errs.append("compressed-tensors: no config_groups in config.json")
    except Exception as e:
        errs.append(f"compressed-tensors: {type(e).__name__}: {e}")
    raise RuntimeError("no quant config built:\n  " + "\n  ".join(errs))


def main(DIR, L):
    vcfg = vllm_ctx()
    cfgj = json.load(open(f"{DIR}/config.json"))
    H = cfgj["hidden_size"]
    E = cfgj.get("num_local_experts", cfgj.get("num_experts"))
    K = cfgj["num_experts_per_tok"]
    I = cfgj.get("moe_intermediate_size") or cfgj.get("intermediate_size")
    norm = cfgj.get("norm_topk_prob", True)
    scale = cfgj.get("routed_scaling_factor", 1.0)
    print(f"M2.5 MoE dims: H={H} E={E} K={K} I={I} norm_topk_prob={norm} routed_scale={scale}", flush=True)

    method, qcfg = build_quant_config(DIR)
    print(f"quant config: {method} -> {type(qcfg).__name__}", flush=True)

    idx = json.load(open(f"{DIR}/model.safetensors.index.json"))["weight_map"]
    _HD = {}
    def raw(n):
        s = idx[n]
        if s not in _HD:
            _HD[s] = safe_open(f"{DIR}/{s}", "pt", device="cpu")
        return _HD[s].get_tensor(n)

    Pmoe = f"model.layers.{L}.block_sparse_moe."
    Pexp = Pmoe + "experts."
    suffixes = sorted({k.split(f"{Pexp}0.w1.")[1] for k in idx if k.startswith(f"{Pexp}0.w1.")})
    print(f"expert tensor suffixes (experts.0.w1.*): {suffixes}", flush=True)
    if not suffixes:
        sample = [k for k in idx if k.startswith(f"model.layers.{L}.")][:8]
        raise RuntimeError(f"no expert tensors under {Pexp} — sample layer-{L} keys: {sample}")

    eb = None
    ebn = Pmoe + "e_score_correction_bias"
    if ebn in idx:
        eb = raw(ebn).float().cuda()
    print(f"correction bias: present={eb is not None} shape={tuple(eb.shape) if eb is not None else None}", flush=True)

    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    kw = dict(num_experts=E, top_k=K, hidden_size=H, intermediate_size=I,
              params_dtype=torch.bfloat16, renormalize=norm, use_grouped_topk=False,
              scoring_func="sigmoid", routed_scaling_factor=scale, quant_config=qcfg, prefix=Pexp[:-1])
    if eb is not None:
        kw["e_score_correction_bias"] = eb
    try:
        moe = FusedMoE(**kw).cuda()
    except TypeError as e:
        print(f"FusedMoE kwargs rejected ({e}); retrying minimal", flush=True)
        for k in ("e_score_correction_bias", "routed_scaling_factor", "scoring_func"):
            kw.pop(k, None)
        moe = FusedMoE(**kw).cuda()
    params = dict(moe.named_parameters())
    print("moe params:", [k for k in params if "weight" in k or "scale" in k][:12], flush=True)

    loaded = 0
    for e in range(E):
        for proj, shard in [("w1", "w1"), ("w3", "w3"), ("w2", "w2")]:
            grp = "w2" if shard == "w2" else "w13"
            for suf in suffixes:
                name = f"{Pexp}{e}.{proj}.{suf}"
                pname = f"{grp}_{suf}"
                if name in idx and pname in params:
                    moe.weight_loader(params[pname], raw(name).cuda(), name, shard, e)
                    loaded += 1
    print(f"loaded {loaded} expert tensors", flush=True)
    moe.quant_method.process_weights_after_loading(moe)
    print("process_weights_after_loading OK -- nvfp4 kernel set up", flush=True)

    from vllm.forward_context import set_forward_context
    torch.manual_seed(0)
    T = 6
    x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.1
    gate_w = raw(Pmoe + "gate.weight").to(torch.bfloat16).cuda()  # [E, H]
    router_logits = torch.nn.functional.linear(x, gate_w)         # [T, E]
    with torch.no_grad(), set_forward_context(None, vcfg):
        out = moe(x, router_logits)
    fin = torch.isfinite(out).all().item()
    print(f"\nFORWARD: out {tuple(out.shape)} finite={fin} mean|out|={out.abs().mean().item():.4f}", flush=True)
    print("VERDICT:", "M2.5 NVFP4 FusedMoE EXECUTES on sm_120 — vllm#35566 dodged; the PP stage path is viable."
          if fin else "ran but output NOT finite — inspect routing/scales.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/root/m25")
    ap.add_argument("--layer", type=int, default=30)
    a = ap.parse_args()
    main(a.dir, a.layer)
