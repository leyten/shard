"""Heterogeneous-arch existential gate: does MiniMax-M2.5's NVFP4 FusedMoE run on a
NON-Blackwell card (4090 sm_89 / 3090 sm_86), on which M25_MOE_BACKEND, at what VRAM
cost and per-layer latency?

Same load path as m25_moe_probe.py (the sm_120-proven probe) but backend-parametrized
and measuring: for each backend in --backends, build one layer's MoE, load real NVFP4
experts, forward, then time T=1 and T=8 decode-shaped calls. Prints a PER-BACKEND
verdict line the tier table is built from:

  BACKEND cutlass    VERDICT=... vram_gb=... t1_ms=... t8_ms=...

  python hetero_moe_probe.py --dir /root/m25 --layer 30 --backends cutlass,marlin,emulation
"""
import os, json, argparse, time, traceback
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29577")
os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "0" and "0" or "1"); os.environ.setdefault("LOCAL_RANK", "0")
import torch
from safetensors import safe_open

_CTX = None


def vllm_ctx(backend):
    global _CTX
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.v1.worker.workspace import init_workspace_manager
    if _CTX is None:
        torch.cuda.set_device(0)
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                     distributed_init_method="env://", backend="nccl")
        vcfg = VllmConfig()
        ctx = set_current_vllm_config(vcfg); ctx.__enter__()
        initialize_model_parallel(1); init_workspace_manager(torch.device("cuda"))
        _CTX = (ctx, vcfg)
    try:
        _CTX[1].kernel_config.moe_backend = backend
    except Exception as e:
        print(f"warn moe_backend set: {e}", flush=True)
    return _CTX[1]


def build_quant_config(DIR):
    cfgj = json.load(open(f"{DIR}/config.json"))
    qc = cfgj.get("quantization_config")
    hfq = json.load(open(f"{DIR}/hf_quant_config.json")) if os.path.exists(f"{DIR}/hf_quant_config.json") else None
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
    src = hfq["quantization"] if (hfq and "quantization" in hfq) else qc
    return ModelOptNvFp4Config.from_config(src)


def run_backend(DIR, L, backend):
    vcfg = vllm_ctx(backend)
    cfgj = json.load(open(f"{DIR}/config.json"))
    H = cfgj["hidden_size"]; E = cfgj.get("num_local_experts", cfgj.get("num_experts"))
    K = cfgj["num_experts_per_tok"]
    I = cfgj.get("moe_intermediate_size") or cfgj.get("intermediate_size")
    norm = cfgj.get("norm_topk_prob", True); scale = cfgj.get("routed_scaling_factor", 1.0)
    qcfg = build_quant_config(DIR)

    idx = json.load(open(f"{DIR}/model.safetensors.index.json"))["weight_map"]
    _HD = {}
    def raw(n):
        s = idx[n]
        if s not in _HD:
            _HD[s] = safe_open(f"{DIR}/{s}", "pt", device="cpu")
        return _HD[s].get_tensor(n)

    Pmoe = f"model.layers.{L}.block_sparse_moe."; Pexp = Pmoe + "experts."
    suffixes = sorted({k.split(f"{Pexp}0.w1.")[1] for k in idx if k.startswith(f"{Pexp}0.w1.")})
    eb = raw(Pmoe + "e_score_correction_bias").float().cuda() if Pmoe + "e_score_correction_bias" in idx else None

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    kw = dict(num_experts=E, top_k=K, hidden_size=H, intermediate_size=I,
              params_dtype=torch.bfloat16, renormalize=norm, use_grouped_topk=False,
              scoring_func="sigmoid", routed_scaling_factor=scale, quant_config=qcfg, prefix=Pexp[:-1])
    if eb is not None:
        kw["e_score_correction_bias"] = eb
    try:
        moe = FusedMoE(**kw).cuda()
    except TypeError:
        for k in ("e_score_correction_bias", "routed_scaling_factor", "scoring_func"):
            kw.pop(k, None)
        moe = FusedMoE(**kw).cuda()
    params = dict(moe.named_parameters())
    loaded = 0
    for e in range(E):
        for proj, shard in [("w1", "w1"), ("w3", "w3"), ("w2", "w2")]:
            grp = "w2" if shard == "w2" else "w13"
            for suf in suffixes:
                name = f"{Pexp}{e}.{proj}.{suf}"; pname = f"{grp}_{suf}"
                if name in idx and pname in params:
                    moe.weight_loader(params[pname], raw(name).cuda(), name, shard, e)
                    loaded += 1
    moe.quant_method.process_weights_after_loading(moe)
    vram_gb = (torch.cuda.memory_allocated() - base) / 1e9
    print(f"[{backend}] loaded {loaded} tensors, moe VRAM {vram_gb:.2f} GB "
          f"(kernel={type(moe.quant_method).__name__})", flush=True)

    from vllm.forward_context import set_forward_context
    torch.manual_seed(0)
    gate_w = raw(Pmoe + "gate.weight").to(torch.bfloat16).cuda()

    def fwd(T):
        x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.1
        rl = torch.nn.functional.linear(x, gate_w)
        with torch.no_grad(), set_forward_context(None, vcfg):
            return moe(x, rl)

    out = fwd(6)
    fin = torch.isfinite(out).all().item()
    ref = out.float().mean().item()
    if not fin:
        return dict(ok=False, why="non-finite output", vram_gb=vram_gb)

    times = {}
    for T in (1, 8):
        for _ in range(10):
            fwd(T)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        N = 50
        for _ in range(N):
            fwd(T)
        torch.cuda.synchronize()
        times[T] = (time.perf_counter() - t0) / N * 1000
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    return dict(ok=True, vram_gb=vram_gb, peak_gb=peak_gb, t1_ms=times[1], t8_ms=times[8],
                mean_out=ref, kernel=type(moe.quant_method).__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/root/m25")
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--backends", default="cutlass,marlin,emulation")
    a = ap.parse_args()
    name = torch.cuda.get_device_name(0); cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU {name} sm_{cap[0]}{cap[1]} {total:.0f} GB  torch {torch.__version__}", flush=True)
    results = {}
    for b in a.backends.split(","):
        b = b.strip()
        print(f"\n===== backend {b} =====", flush=True)
        try:
            results[b] = run_backend(a.dir, a.layer, b)
        except Exception as e:
            traceback.print_exc()
            results[b] = dict(ok=False, why=f"{type(e).__name__}: {str(e)[:200]}")
    print("\n===== SUMMARY =====", flush=True)
    for b, r in results.items():
        if r.get("ok"):
            print(f"BACKEND {b:10s} VERDICT=RUNS vram_gb={r['vram_gb']:.2f} peak_gb={r['peak_gb']:.2f} "
                  f"t1_ms={r['t1_ms']:.2f} t8_ms={r['t8_ms']:.2f} kernel={r['kernel']} mean_out={r['mean_out']:.5f}", flush=True)
        else:
            print(f"BACKEND {b:10s} VERDICT=FAILS why={r.get('why')}", flush=True)


if __name__ == "__main__":
    main()
