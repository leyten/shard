"""MiMo-V2 checkpoint layout: the fused qkv is pre-sharded, and its FP8 scales are tiled per chunk.

Facts from the released checkpoints' safetensors headers (XiaomiMiMo/MiMo-V2.6-Flash-RL @ 5711b268,
MiMo-V2.6-Pro-RL @ 73875d00), read by HTTP Range, no weights needed:

  `self_attn.qkv_proj.weight` is FP8-e4m3 [rows, H] stored for ckpt_tp = config
  `num_key_value_heads` chunks (4 Flash, 8 Pro; for EVERY layer, SWA and MTP included), interleaved
  [Q_0|K_0|V_0|Q_1|K_1|V_1|...], and its fp32 `weight_scale_inv` is tiled PER CHUNK:
  ceil(rows_per_chunk / 128) scale rows each.

    Flash global  qkv [13568, 4096]  scale [108, 32]  (chunk 3392 rows -> 27 x 4; plain grid = 106)
    Flash SWA/MTP qkv [14848, 4096]  scale [116, 32]  (chunk 3712 = 29 x 128: the per-chunk grid
                                                       happens to equal the plain one, but the rows
                                                       are still interleaved)
    Pro (all 73)  qkv [27136, 6144]  scale [216, 48]  (chunk 3392 rows -> 27 x 8; plain grid = 212)

So the vendor HF modeling file's `qkv.split([q, k, v])` is wrong on the released FP8 checkpoint, and
a plain `r // 128` scale lookup mis-scales every chunk after the first. transformers' own FP8 path
allocates a [106, 32] scale for Flash global layers.

Evidence for `qkv_rechunk` (scratchpad oracle, 2026-09-28): bit-identical to vLLM v0.30.0
`models/mimo_v2.py::_shard_fp8_qkv_proj`'s row gather (same per-chunk `scale_index` arithmetic) and
to llama.cpp `conversion/mimo.py::_tp_aware_qkv_dequant` on the real layer-0 and layer-1 tensors; data
check on real layer 0: with s = amax/448, every per-chunk 128-row block holds a 448 code. We do NOT
import vLLM's function: after the gather it RE-QUANTIZES the regrouped rows to fp8 for its own TP
degree (lossy whenever TP != ckpt_tp). Here the rows are dequantized with their own chunk's scales
and handed back in natural head order; callers round to bf16 (quant.to_bf16_values) as the
canonical weight.
"""
from __future__ import annotations

import math

import torch

from .quant import FP8_BLOCK, dequant_fp8_block


def qkv_rechunk(w: torch.Tensor, s: torch.Tensor, n_q: int, n_kv: int, qk_dim: int, v_dim: int,
                ckpt_tp: int, block: int = FP8_BLOCK):
    """De-interleave MiMo-V2's pre-sharded fused qkv and dequantize each row with its own chunk's
    scales. Returns fp32 (Wq [n_q*qk_dim, H], Wk [n_kv*qk_dim, H], Wv [n_kv*v_dim, H]), heads in
    natural order: chunk c holds Q heads [c*n_q/tp, (c+1)*n_q/tp) and KV heads
    [c*n_kv/tp, (c+1)*n_kv/tp). `ckpt_tp` is the config's global `num_key_value_heads`."""
    rows = w.shape[0]
    if n_q % ckpt_tp or n_kv % ckpt_tp or rows % ckpt_tp:
        raise ValueError(f"qkv rows {rows} / heads {n_q}q {n_kv}kv do not split into {ckpt_tp} chunks")
    rpc = rows // ckpt_tp
    q_pc, k_pc, v_pc = n_q // ckpt_tp * qk_dim, n_kv // ckpt_tp * qk_dim, n_kv // ckpt_tp * v_dim
    if q_pc + k_pc + v_pc != rpc:
        raise ValueError(f"qkv chunk of {rpc} rows != q {q_pc} + k {k_pc} + v {v_pc}")
    cs = math.ceil(rpc / block)
    if s.shape[0] != ckpt_tp * cs:
        raise ValueError(f"qkv scale rows {s.shape[0]} != {ckpt_tp} chunks x {cs} (per-chunk grid)")
    r = torch.arange(rows)
    scale_row = (r // rpc) * cs + (r % rpc) // block
    d = dequant_fp8_block(w, s, scale_row, block).view(ckpt_tp, rpc, -1)
    hid = d.shape[-1]
    return (d[:, :q_pc].reshape(-1, hid), d[:, q_pc:q_pc + k_pc].reshape(-1, hid),
            d[:, q_pc + k_pc:].reshape(-1, hid))
