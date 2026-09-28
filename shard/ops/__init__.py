"""shard.ops: receipt-grade (batch-invariant) Triton kernels for the v2 stage runtime (BUILD-PLAN R4).

  invariant_moe   MXFP4 W4A16 grouped MoE GEMM, weights packed 4-bit, dequant in-kernel, SPLIT_K=1
  invariant_attn  fixed-split diff-kv attention (QK 192 / V 128), sinks, window 128 / global, paged KV

Device families
  NVIDIA sm_80+ via Triton: Ampere 3090/A100 (sm_80/86: bf16 MMA, no FP8/FP4 needed -- the weights are
  decoded to bf16 in registers), Ada 4090 (sm_89), Blackwell 5090 / RTX 6000 Pro (sm_120), GB10 Spark
  (sm_121, ARM host). AMD ROCm via Triton's AMD backend: expected to compile, untested. Apple: NOT covered
  (Triton has no Metal backend); the MLX path needs the same contract as Metal kernels
  (mx.fast.metal_kernel): fixed threadgroup tiles, full-K sequential accumulators, split boundaries at
  absolute KV positions, fixed-order merge -- mx.quantized_matmul / gather_qmm / fast SDPA pick kernels by
  shape and are not invariant as-is. Each (backend, arch) is its own numerics_id class: bytes are promised
  equal only within a class (ARCHITECTURE-v2 §6); the pins are invariant_moe.NUMERICS_PIN and
  invariant_attn.NUMERICS_PIN.
"""
