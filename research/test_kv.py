"""KV-cache correctness unit test: the incremental decode hidden must equal the full-recompute
hidden for the same position (within bf16 tolerance). Independent of argmax, so it isn't fooled
by gibberish-logit fragility. run under /root/vmoe on a node holding layers 6-9."""
import torch
import glm_swarm_nvfp4_kv as G
G._vllm_ctx()
vcfg = G._VC
LIDS = [6, 7, 8, 9]
layers = [G.Layer(i) for i in LIDS]
torch.manual_seed(0)
N = 6
h_prompt = torch.randn(1, N, G.H, dtype=torch.bfloat16, device="cuda") * 0.1
T0 = torch.randn(1, 1, G.H, dtype=torch.bfloat16, device="cuda") * 0.1

# (1) full recompute of [prompt, T0] -> reference last-position hidden
for L in layers: L.reset()
h_full = G.run_block(layers, 0, torch.cat([h_prompt, T0], 1), vcfg)
ref = h_full[0, -1].float()

# (2) incremental: prefill prompt at pos 0, then decode T0 at pos N
for L in layers: L.reset()
_ = G.run_block(layers, 0, h_prompt, vcfg)
h_dec = G.run_block(layers, N, T0, vcfg)
cached = h_dec[0, -1].float()

md = (ref - cached).abs().max().item()
cos = torch.nn.functional.cosine_similarity(ref, cached, 0).item()
rel = ((ref - cached).norm() / ref.norm()).item()
print(f"\nKV-CACHE CHECK: max|diff|={md:.5f}  cosine={cos:.6f}  relL2={rel:.5f}", flush=True)
print("VERDICT:", "KV CACHE CORRECT (incremental == full recompute within bf16)" if cos > 0.9995 and rel < 0.02
      else "KV CACHE BUG — incremental decode diverges from full recompute", flush=True)
