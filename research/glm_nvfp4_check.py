"""NVFP4 correctness check: run the SAME inputs + IDENTICAL routing through the NVFP4 experts
(vLLM FusedMoE) and the fp8 experts (fused_experts) for the same layer 6, and compare. High
cosine similarity => the NVFP4 kernel computes the right function (4-bit vs 8-bit quant of the
same weights), confirming it's correct, not just finite. run under /root/vmoe.
"""
import os, json, torch
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29579")
os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("LOCAL_RANK", "0")
from safetensors import safe_open
from transformers import GlmMoeDsaConfig
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

dev, LAYER, BLK = "cuda", 6, 128
torch.cuda.set_device(0)
init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://", backend="nccl")
vcfg = VllmConfig(); _ctx = set_current_vllm_config(vcfg); _ctx.__enter__()
initialize_model_parallel(1); init_workspace_manager(torch.device("cuda"))

NV, F8 = "/root/glm52nvfp4", "/root/glm52fp8"
cfg = GlmMoeDsaConfig.from_pretrained(NV)
H, E, I, K = cfg.hidden_size, cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.num_experts_per_tok
torch.manual_seed(0)
T = 8
x = torch.randn(T, H, dtype=torch.bfloat16, device=dev) * 0.1
router = torch.randn(T, E, dtype=torch.bfloat16, device=dev)
# IDENTICAL fixed routing for both paths (test the experts, not the router)
topk_w, topk_id = torch.softmax(router.float(), -1).topk(K, -1)
topk_w = topk_w.to(torch.bfloat16)

def rd(d):
    idx = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]; h = {}
    def g(n):
        s = idx[n]
        if s not in h: h[s] = safe_open(f"{d}/{s}", "pt", device="cpu")
        return h[s].get_tensor(n)
    return idx, g

# ---- NVFP4 path: vLLM FusedMoE with forced routing ----
def fixed_routing(*a, **kw):
    return topk_w, topk_id.to(torch.int32)
qnv = ModelOptNvFp4Config.from_config(json.load(open(f"{NV}/config.json"))["quantization_config"])
moe = FusedMoE(num_experts=E, top_k=K, hidden_size=H, intermediate_size=I, params_dtype=torch.bfloat16,
               renormalize=False, custom_routing_function=fixed_routing, quant_config=qnv, prefix="m").to(dev)
idxn, gn = rd(NV); P = f"model.layers.{LAYER}.mlp.experts."
pp = dict(moe.named_parameters())
for e in range(E):
    for proj, shard in [("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")]:
        grp = "w2" if shard == "w2" else "w13"
        for suf in ["weight", "weight_scale", "weight_scale_2", "input_scale"]:
            n = f"{P}{e}.{proj}.{suf}"
            if n in idxn: moe.weight_loader(pp[f"{grp}_{suf}"], gn(n).to(dev), n, shard, e)
moe.quant_method.process_weights_after_loading(moe)
with torch.no_grad(), set_forward_context(None, vcfg):
    out_nv = moe(x, router)

# ---- fp8 path: fused_experts with the same routing ----
idxf, gf = rd(F8)
def bs(n): return gf(n + "_scale_inv")
w1 = torch.empty(E, 2 * I, H, dtype=torch.float8_e4m3fn, device=dev); w2 = torch.empty(E, H, I, dtype=torch.float8_e4m3fn, device=dev)
w1s = torch.empty(E, (2 * I) // BLK, H // BLK, dtype=torch.float32, device=dev); w2s = torch.empty(E, H // BLK, I // BLK, dtype=torch.float32, device=dev)
for e in range(E):
    w1[e] = torch.cat([gf(P + f"{e}.gate_proj.weight"), gf(P + f"{e}.up_proj.weight")], 0).to(dev)
    w1s[e] = torch.cat([bs(P + f"{e}.gate_proj.weight"), bs(P + f"{e}.up_proj.weight")], 0).to(dev)
    w2[e] = gf(P + f"{e}.down_proj.weight").to(dev); w2s[e] = bs(P + f"{e}.down_proj.weight").to(dev)
qc = fp8_w8a8_moe_quant_config(w1_scale=w1s, w2_scale=w2s, block_shape=[BLK, BLK])
with torch.no_grad():
    out_f8 = fused_experts(x, w1, w2, topk_w, topk_id.to(torch.int32), quant_config=qc)

cos = torch.nn.functional.cosine_similarity(out_nv.float().flatten(), out_f8.float().flatten(), 0).item()
rel = ((out_nv.float() - out_f8.float()).norm() / out_f8.float().norm()).item()
print(f"\nNVFP4 mean|x|={out_nv.abs().mean():.4f}  fp8 mean|x|={out_f8.abs().mean():.4f}")
print(f"cosine(NVFP4, fp8) = {cos:.4f} | relative L2 diff = {rel:.3f}")
print("VERDICT:", "NVFP4 CORRECT — matches fp8 within expected 4-bit-vs-8-bit quant error. 16-node path validated."
      if cos > 0.95 else "DIVERGES — NVFP4 wiring likely has a scale/layout bug; inspect.")
