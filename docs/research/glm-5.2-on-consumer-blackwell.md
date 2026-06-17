# GLM-5.2 on consumer Blackwell (RTX 5090)

*Research record. Status: feasibility de-risked end to end; serving path identified
(quantized pipeline-parallel). 2026-06-17.*

## Why GLM-5.2

744B MoE (40B active), MLA + DeepSeek Sparse Attention (DSA), a **native MTP draft**, 1M
context, MIT license. Beats GPT-5.5 on long-horizon coding at ~1/6 the cost. Serving it
across scattered consumer 5090s would prove the engine generalizes beyond gpt-oss — a
frontier open model on the swarm.

It fits the WAN spec-decode thesis *better* than gpt-oss:
- **Native MTP head** = a free trained draft (no separate 20B model; the EAGLE endpoint,
  shipped in the weights).
- **MLA** = a tiny compressed KV latent (kv_lora_rank 512) → small activations on the wire
  and a cheap fast-verify static cache.
- **MoE sparsity** (40B active) = light per-node verify compute.

## Fast-verify de-risk (real RTX 5090, sm_120)

- `glm_moe_dsa` runs eager on Blackwell; the layer block has no kernel gaps.
- **The gpt-oss CUDA-graph fast-verify lever does NOT transfer.** GLM's verify is
  memory-bandwidth-bound (reading selected-expert weights), not launch-overhead-bound, so
  graphs give ~1.0×. Dense-attn bypass (the DSA indexer is a no-op at verify lengths
  ≪ index_topk) + a sparse/grouped MoE are bit-exact, but the graph doesn't help.
- **The real levers are quantization + fused kernels.** vLLM `fused_experts` at GLM dims:
  **2.16 ms bf16 / 1.10 ms fp8** per MoE layer (NVFP4 available) → ~55–100 ms full-model
  verify projection — comparable to gpt-oss-120B's 135 ms despite 6× the params.
  (`research/glm_probe_*.py`, `research/bench_fused_moe.py`.)

## Topology — the scattered-swarm wedge, validated on real WAN

`shard/topology.py`: minimum-latency Hamiltonian loop over the **measured asymmetric** RTT
mesh (exact Held-Karp ≤16 nodes, NN + 2-opt above) + best-k-of-pool node selection, fed by
`phase0/mesh.py`. Validated on 5 cheap scattered US nodes: latencies are genuinely
asymmetric, and **node selection dropped a 210–242 ms outlier → 114 ms vs 517 ms loop =
4.53×**. The scheduler pillar works on real internet, not just synthetic data.

## 2-box WAN pipeline — our PP engine

`research/glm_stage_node.py`: a real GLM-5.2 layer block per machine with hidden-state I/O
over our TCP transport. Layer 6 (Washington 5090) ↔ layer 7 (Florida 5090), warm round
~110 ms, output sane. Multi-machine pipeline integration works.

## The full-model correctness run, and the wall

Rented 16×5090 (512 GB), `Mapika/GLM-5.2-NVFP4` (410 GB), TP=16 under vLLM 0.23. **Every
*software* wall was surmountable, in sequence:**

1. DSA sparse-MLA has **no sm_120 kernel** (`FLASHMLA_SPARSE` is Hopper / datacenter-Blackwell
   only) → patched `is_v32=False` → dense MLA (`TRITON_MLA`). Valid because dense ≡ sparse at
   decode lengths ≪ `index_topk` (2048).
2. Indexer weights have no home in dense → patched `load_weights` to skip `"indexer"` weights.
   **Model fully loaded across all 16 GPUs.**
3. flashinfer MoE workspace OOM (1.54 GB, ~200 MB short) → capped `max_num_batched_tokens`.

**Then the *hardware* wall:** `No available memory for the cache blocks`. Rank 0 holds
~29 GB (embeddings + lm_head + its weight share) on a 32 GB card; after weights + MoE
workspace there is **zero room for KV cache**. High util → workspace OOMs; low util → no KV.
No setting fits — a hard per-GPU 32 GB overflow.

**Conclusion: GLM-5.2-NVFP4 does not fit on 16×32 GB RTX 5090 under vLLM tensor-parallel.**
The overflow is a TP artifact; the model's own configs all use ≥80 GB/GPU.

## The insight, and the path

The rank-0 overflow is a **tensor-parallel** artifact — TP piles embeddings + lm_head onto
rank 0. **Pipeline parallel — our engine — spreads embeddings (stage 0), lm_head (tail), and
layers evenly, so it sidesteps the wall.** Our PP engine already runs GLM-5.2 dense + correct
on a single 5090.

The remaining work: our PP stage runner currently **dequantizes fp8→bf16** (full model
~1.5 TB — far too heavy). The unlock is **quantized PP stage execution**: run NVFP4/fp8
weights directly — `fused_experts` for the MoE at quant, dense MLA in bf16 (the NVFP4
checkpoints keep MLA/indexer in BF16 anyway) — with no bf16 dequant. That drops per-stage
memory ~2–4× so the full model fits a feasible 5090 swarm, and PP avoids the TP rank-0
overflow entirely. (Separately, stock vLLM/SGLang sm_120 DSA support is maturing — community
`vllm-sm120` builds exist — which may give a turnkey path later.)

## Status / log

- **2026-06-17** — Full de-risk above. **Decision: build quantized pipeline-parallel stage
  execution** as the c0mpute-native serving path for GLM-5.2 on consumer Blackwell; table
  stock-vLLM-TP serving (doesn't fit per-GPU on 32 GB cards) and watch the sm_120 DSA
  ecosystem. gpt-oss-120B (18–25 tok/s over WAN) remains the shipped headline meanwhile.
