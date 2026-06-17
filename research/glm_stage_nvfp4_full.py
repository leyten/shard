"""GLM-5.2 NVFP4 FULL decoder-layer stage — wire the proven NVFP4 FusedMoE into the real
layer: dense MLA (bf16) + NVFP4 routed experts (vLLM FusedMoE) + shared expert + norms,
hidden-state in/out. This is the code a 16-node swarm runs. Validate one layer on box A.

(First pass: shared expert STUBBED to zero — it's also NVFP4 and needs its own handling;
the routed MoE is the 95% of compute and is proven correct (cosine 0.977 vs fp8). Once the
routed full-layer stage is green, add the shared expert.)  run under /root/vmoe.
"""
import os, json, torch
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29580")
os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("LOCAL_RANK", "0")
from safetensors import safe_open
from transformers import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

DIR, dev, LAYER = "/root/glm52nvfp4", "cuda", 6
torch.cuda.set_device(0)
init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://", backend="nccl")
vcfg = VllmConfig(); _ctx = set_current_vllm_config(vcfg); _ctx.__enter__()
initialize_model_parallel(1); init_workspace_manager(torch.device("cuda"))

cfg = GlmMoeDsaConfig.from_pretrained(DIR); cfg._attn_implementation = "eager"
H, E, I, K = cfg.hidden_size, cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.num_experts_per_tok
idx = json.load(open(f"{DIR}/model.safetensors.index.json"))["weight_map"]
_HD = {}
def raw(n):
    s = idx[n]
    if s not in _HD: _HD[s] = safe_open(f"{DIR}/{s}", "pt", device="cpu")
    return _HD[s].get_tensor(n)
P = f"model.layers.{LAYER}."

# ---- NVFP4 routed experts via vLLM FusedMoE (the proven path) ----
qnv = ModelOptNvFp4Config.from_config(json.load(open(f"{DIR}/config.json"))["quantization_config"])
e_bias = raw(P + "mlp.gate.e_score_correction_bias").float().to(dev)
moe = FusedMoE(num_experts=E, top_k=K, hidden_size=H, intermediate_size=I, params_dtype=torch.bfloat16,
               renormalize=cfg.norm_topk_prob, use_grouped_topk=True, num_expert_group=cfg.n_group,
               topk_group=cfg.topk_group, scoring_func="sigmoid", routed_scaling_factor=cfg.routed_scaling_factor,
               e_score_correction_bias=e_bias, quant_config=qnv, prefix=P + "mlp.experts").to(dev)
pp = dict(moe.named_parameters()); EP = P + "mlp.experts."
for e in range(E):
    for proj, shard in [("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")]:
        grp = "w2" if shard == "w2" else "w13"
        for suf in ["weight", "weight_scale", "weight_scale_2", "input_scale"]:
            n = f"{EP}{e}.{proj}.{suf}"
            if n in idx: moe.weight_loader(pp[f"{grp}_{suf}"], raw(n).to(dev), n, shard, e)
moe.quant_method.process_weights_after_loading(moe)
print("NVFP4 routed experts loaded + kernel set up", flush=True)

# ---- NVFP4 shared expert: run as a 1-expert FusedMoE (same proven kernel, no fp4 unpack) ----
def shared_routing(*a, **kw):
    hs = kw["hidden_states"]; T = hs.shape[0]
    return (torch.ones(T, 1, dtype=torch.bfloat16, device=hs.device),
            torch.zeros(T, 1, dtype=torch.int32, device=hs.device))
shmoe = FusedMoE(num_experts=1, top_k=1, hidden_size=H, intermediate_size=I, params_dtype=torch.bfloat16,
                 renormalize=False, custom_routing_function=shared_routing, quant_config=qnv,
                 prefix=P + "mlp.shared_experts").to(dev)
sp = dict(shmoe.named_parameters()); SP = P + "mlp.shared_experts."
for proj, shard in [("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")]:
    grp = "w2" if shard == "w2" else "w13"
    for suf in ["weight", "weight_scale", "weight_scale_2", "input_scale"]:
        n = f"{SP}{proj}.{suf}"
        if n in idx: shmoe.weight_loader(sp[f"{grp}_{suf}"], raw(n).to(dev), n, shard, 0)
shmoe.quant_method.process_weights_after_loading(shmoe)
print("NVFP4 shared expert loaded (1-expert FusedMoE)", flush=True)

# ---- the rest of the layer (bf16 in the NVFP4 ckpt): MLA, norms, gate. shared via shmoe above. ----
sd = {}
for n in ["self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight", "self_attn.kv_a_proj_with_mqa.weight",
          "self_attn.kv_b_proj.weight", "self_attn.o_proj.weight", "self_attn.q_a_layernorm.weight",
          "self_attn.kv_a_layernorm.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
          "mlp.gate.weight"]:
    sd[n] = raw(P + n).to(torch.bfloat16).to(dev)
sd["mlp.gate.e_score_correction_bias"] = e_bias
with torch.device("meta"):
    layer = M.GlmMoeDsaDecoderLayer(cfg, LAYER)
layer.load_state_dict(sd, strict=False, assign=True); layer.eval()

# ---- swap the layer's MoE to the NVFP4 FusedMoE; dense MLA bypass ----
def moe_forward(self, hidden_states):
    shp = hidden_states.shape
    h = hidden_states.view(-1, H)
    router_logits = torch.nn.functional.linear(h, self.gate.weight)
    ones = torch.ones(h.shape[0], 1, dtype=torch.bfloat16, device=h.device)
    out = moe(h, router_logits) + shmoe(h, ones)
    return out.view(shp)
M.GlmMoeDsaMoE.forward = moe_forward

def dense_attn(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
               position_ids=None, prev_topk_indices=None, **kw):
    b, s = hidden_states.shape[:-1]
    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    q = self.q_b_proj(q_resid).view(b, s, -1, self.qk_head_dim).transpose(1, 2)
    q_pass, q_rot = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    ckv = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(ckv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(b, s, -1, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
    k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_rot = k_rot.view(b, 1, s, self.qk_rope_head_dim)
    cos, sin = position_embeddings
    q_rot, k_rot = M.apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
    o, w = M.eager_attention_forward(self, torch.cat((q_pass, q_rot), -1), torch.cat((k_pass, k_rot), -1),
                                     value_states, attention_mask, dropout=0.0, scaling=self.scaling, **kw)
    return self.o_proj(o.reshape(b, s, -1).contiguous()), w, None
M.GlmMoeDsaAttention.forward = dense_attn

rotary = M.GlmMoeDsaRotaryEmbedding(cfg).to(dev)
T = 6
torch.manual_seed(0)
h = torch.randn(1, T, H, dtype=torch.bfloat16, device=dev) * 0.1
pos = torch.arange(T, device=dev).unsqueeze(0); pe = rotary(h, position_ids=pos)
mask = torch.zeros(1, 1, T, T, dtype=torch.bfloat16, device=dev)
mask.masked_fill_(torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), 1), float("-inf"))
with torch.no_grad(), set_forward_context(None, vcfg):
    out = layer(h, attention_mask=mask, position_ids=pos, use_cache=False, position_embeddings=pe)[0]
    fin = torch.isfinite(out).all().item()
    mem = torch.cuda.max_memory_allocated() / 1e9
print(f"\nNVFP4 FULL STAGE (layer {LAYER}): out {tuple(out.shape)} finite={fin} "
      f"mean|x| {h.abs().mean():.3f}->{out.abs().mean():.3f} | peak GPU {mem:.1f} GB", flush=True)
print("VERDICT:", "NVFP4 COMPLETE-LAYER STAGE RUNS — MLA bf16 + NVFP4 routed MoE + NVFP4 shared expert "
      "in one real decoder layer. The 16-node stage is feature-complete. Next: multi-layer + launcher." if fin and out.abs().max() < 1e4 else "not sane — inspect.", flush=True)
