"""Checkpoint weight formats: pure torch-CPU decoders, correct first (shard v2, R1).

These define what a checkpoint's bytes MEAN. Every runtime (CPU oracle, CUDA kernels, MLX) must
decode to the same values, so the canonical definitions live here, each pinned against independent
implementations and the real data (evidence in each docstring):

  quant    MXFP4 (OCP MX: e2m1 low nibble first, E8M0 scales), FP8-e4m3 128x128 block dequant,
           the bf16 RNE rounding that is part of the canonical weight, and the keep-fp32 rule for
           router correction biases.
  mimo_v2  the MiMo-V2 fused qkv layout: pre-sharded per checkpoint-TP chunk with a per-chunk
           FP8 scale grid (the vendor HF split is wrong on the released checkpoint).
"""
