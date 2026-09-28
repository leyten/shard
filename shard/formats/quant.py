"""Generic quantized-weight formats, decoded on the CPU with torch. Correct first, not fast.

MXFP4 (OCP Microscaling v1.0, as MiMo-V2 / gpt-oss / Kimi-K3 routed experts ship it)
  packed  U8 [..., N, K/2]: element 2i in the LOW nibble, 2i+1 in the HIGH nibble.
  codes   e2m1, bit 3 = sign: {0, .5, 1, 1.5, 2, 3, 4, 6} and their negatives (0x8 = -0.0).
  scale   U8 E8M0 [..., N, K/32]: one shared exponent per 32 consecutive elements along K,
          value 2^(s-127); 0xFF is NaN.
  Evidence: bit-identical to transformers' gpt-oss decoder (`integrations.mxfp4.
  convert_moe_packed_tensors`) on real MiMo-V2.6-Flash layer-1 experts 0/131/255 x gate/up/down
  (scratchpad oracle, 2026-09-28); the same OCP layout vLLM 0.30.0's Mxfp4MoEMethod consumes and
  llama.cpp's conversion/mimo.py reads; DeepSeek-V4's own converter (vendor/deepseek_v4_ref/
  inference/convert.py, FP4_TABLE) unpacks low nibble first too.
  Exactness: every finite MXFP4 value is a code with <= 2 significant bits times a power of two
  >= 2^-128, so it is EXACTLY representable in bf16 (same exponent range as fp32, subnormals down
  to 2^-133). W4A16 therefore means "bf16 GEMM on exact weights". A result that overflows (a scale
  >= 253 with a large code) is refused, not silently turned into inf.

FP8-e4m3 128x128 block scales (DeepSeek-style `weight_scale_inv`, as MiMo-V2 qkv / dense MLP / MTP
and DeepSeek-V3/V4 ship them)
  weight  F8_E4M3 [N, K]; scale F32 [ceil(N/128), ceil(K/128)] for the plain grid.
  The scale is a MULTIPLIER despite the `_inv` name: dequant = w * s. (Data check on real MiMo
  layer 0: every true 128x128 block holds a 448 code, i.e. s = amax / 448.)
  Canonical rounding: the fp32 product is one IEEE fp32 multiply (RNE), then `to_bf16_values`
  rounds it RNE to bf16 once. That is TWO roundings, and deliberately so: an e4m3 significand has
  4 bits and an fp32 scale 24, so the exact product can need 28 bits. The definition is "fp32
  multiply, then bf16 RNE", which every IEEE implementation reproduces bit for bit; it is NOT the
  single correctly-rounded bf16 of the exact product (they differ on rare near-ties).

keep-fp32 tensors
  Router correction biases (`e_score_correction_bias`, noaux_tc routing: MiMo-V2, MiniMax-M2.5,
  DeepSeek-V3/V4, Kimi-K3) select the top-k experts by `sigmoid(logits) + bias`. They are stored F32
  and MUST stay F32: on MiMo-V2.6-Flash the values sit at 1.7-2.2, where a bf16 ulp is 0.0078, and
  rounding them to bf16 flips 12 of 256 expert choices on the 32-token golden prompt (scratchpad
  golden/README.md). The vendor HF module declares no `_keep_in_fp32_modules`, so a stock bf16 load
  does exactly that; vLLM keeps them fp32. `check_keep_fp32` is the load-time guard.
"""
from __future__ import annotations

import math

import torch

MX_BLOCK = 32          # OCP MX: elements per shared E8M0 scale
FP8_BLOCK = 128        # DeepSeek-style block-scale tile edge

FP4_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)
_B = torch.arange(256)
_BYTE_LUT = torch.stack([FP4_E2M1[_B & 0x0F], FP4_E2M1[_B >> 4]], dim=-1)  # byte -> (elem 2i, elem 2i+1)
del _B

# Tensors a loader must never down-cast (matched on the last name component).
KEEP_FP32_SUFFIXES = ("e_score_correction_bias",)


def e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """E8M0 shared scales -> fp32 2^(s-127). Built from the fp32 bit pattern (s << 23; s = 0 is the
    subnormal 2^-127), so it is exact for every code and independent of any exp2 implementation.
    0xFF (NaN) is refused. Note: torch.set_flush_denormal(True) would flush s = 0 to zero."""
    if scale.dtype != torch.uint8:
        raise ValueError(f"E8M0 scale must be uint8, got {scale.dtype}")
    if (scale == 255).any():
        raise ValueError("E8M0 NaN scale (0xFF)")
    s = scale.to(torch.int32)
    bits = torch.where(s == 0, torch.full_like(s, 0x00400000), s << 23)
    return bits.view(torch.float32)


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """MXFP4 -> fp32 [..., N, K]. packed U8 [..., N, K/2] (low nibble first), scale U8 E8M0
    [..., N, K/32]. Every value is exactly bf16-representable (see the module docstring)."""
    if packed.dtype != torch.uint8:
        raise ValueError(f"MXFP4 packed tensor must be uint8, got {packed.dtype}")
    *lead, kh = packed.shape
    k = 2 * kh
    if k % MX_BLOCK or tuple(scale.shape) != (*lead, k // MX_BLOCK):
        raise ValueError(f"MXFP4 shapes: packed {tuple(packed.shape)} needs scale "
                         f"{(*lead, k // MX_BLOCK)}, got {tuple(scale.shape)}")
    vals = _BYTE_LUT[packed.long()]                                   # [..., K/2, 2]
    out = vals.reshape(*lead, k // MX_BLOCK, MX_BLOCK) * e8m0_to_fp32(scale).unsqueeze(-1)
    out = out.reshape(*lead, k)
    if not torch.isfinite(out).all():
        raise ValueError("MXFP4 value overflows (E8M0 scale >= 253 with a large code)")
    return out


def dequant_fp8_block(w: torch.Tensor, s: torch.Tensor, scale_row: torch.Tensor | None = None,
                      block: int = FP8_BLOCK) -> torch.Tensor:
    """FP8 e4m3 [N, K] x fp32 block scale -> fp32 [N, K]: w[r, c] * s[scale_row[r], c // block].

    `scale_row[r]` names the scale row covering weight row r; the default is the plain grid
    r // block (and then s must have exactly ceil(N/block) rows). A checkpoint that tiles its scales
    differently (MiMo-V2's per-chunk qkv grid) passes its own map: see mimo_v2.qkv_rechunk. The
    product is one IEEE fp32 multiply (see the module docstring for why that is not exact)."""
    if w.dtype != torch.float8_e4m3fn or s.dtype != torch.float32:
        raise ValueError(f"FP8 block dequant wants float8_e4m3fn x float32, got {w.dtype} x {s.dtype}")
    n, k = w.shape
    if s.dim() != 2 or s.shape[1] != math.ceil(k / block):
        raise ValueError(f"FP8 scale {tuple(s.shape)} does not tile K={k} in blocks of {block}")
    if scale_row is None:
        if s.shape[0] != math.ceil(n / block):
            raise ValueError(f"FP8 scale has {s.shape[0]} rows, the plain grid for N={n} needs "
                             f"{math.ceil(n / block)} (a per-chunk grid must pass scale_row)")
        scale_row = torch.arange(n) // block
    col_block = torch.arange(k) // block
    out = w.to(torch.float32) * s[scale_row][:, col_block]
    if not torch.isfinite(out).all():
        raise ValueError("FP8 block dequant produced a non-finite value (NaN code or bad scale)")
    return out


def to_bf16_values(x: torch.Tensor) -> torch.Tensor:
    """Round to bf16 (RNE) but keep an fp32 container, so every value is exactly bf16."""
    return x.to(torch.bfloat16).to(torch.float32)


def keep_fp32(name: str) -> bool:
    """True for a tensor a loader must keep in fp32 (router correction biases)."""
    return name.rsplit(".", 1)[-1] in KEEP_FP32_SUFFIXES


def check_keep_fp32(name: str, t: torch.Tensor) -> torch.Tensor:
    """Load-time guard: refuse a keep-fp32 tensor that is not fp32 (someone down-cast it)."""
    if keep_fp32(name) and t.dtype != torch.float32:
        raise ValueError(f"{name} must stay float32 (router selection bias), got {t.dtype}")
    return t
