"""Receipt-grade MXFP4 W4A16 grouped (MoE) GEMM: batch-invariant by construction.

What "batch-invariant" means here: the output bytes of one (token, expert) row depend only on that
token's activation row, the expert's weights and this module's pinned config -- never on how many
other rows share the call, which experts they route to, where the row lands inside an M tile, or how
many tiles the grid has. Guaranteed by:

  * ONE fixed tile config (BLOCK_M/N/K below) for every M. No autotune, no M-dependent heuristic.
  * SPLIT_K = 1: every output element is ONE program's fp32 accumulator, walking K in ascending
    BLOCK_K steps (acc = tl.dot(a, w, acc)). No split-K, no stream-K, no atomics, no cross-program
    reduction -- so there is nothing whose order a batch change could permute.
  * Rows never interact: a tile's rows are independent dot products; padding rows (sentinel ids) load
    zeros and are never stored.
  * The expert combine (outside the GEMM) is an elementwise fp32 sum over a token's top-k slots in
    ascending expert id -- the oracle's order (mimo_ref.Layer.ffn).

Weights stay 4-bit in memory: packed U8 [E, N, K/2] (element 2i = low nibble, 2i+1 = high nibble)
plus U8 E8M0 scales [E, N, K/32]; each BLOCK_K x BLOCK_N tile is decoded in registers
(e2m1 -> fp32 by integer arithmetic, E8M0 -> fp32 from the exponent bits, product exact, cast to
bf16 exact) right before the tl.dot. No FP4/FP8 tensor-core instruction is used, so this is the path
for Ampere (sm_80/86, no FP4/FP8 hardware) as well as Ada/Blackwell/GB10.

Device families (Triton backends): NVIDIA sm_80+ (Ampere 3090/A100, Ada 4090, Hopper, Blackwell
5090/RTX 6000 Pro sm_120, GB10 Spark sm_121) and AMD ROCm (CDNA/RDNA3 -- untested). NOT Apple: Triton
has no Metal backend. The MLX path needs the same contract re-stated in MLX: a custom Metal kernel
(mx.fast.metal_kernel) with a fixed threadgroup tile and a full-K sequential per-output accumulator
(mx.quantized_matmul / gather_qmm pick kernels by M, so they are NOT invariant as-is); receipts from
Metal are their own numerics_id class.

The config is part of `numerics_id` (NUMERICS_PIN). Changing any constant changes the bytes.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

MX_BLOCK = 32
BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 64      # pinned; BLOCK_K % 32 == 0 so a K tile never splits a scale
NUM_WARPS, NUM_STAGES = 4, 2
# Triton 3.6's CPU interpreter multiplies the raw uint16 bit patterns in a bf16 x bf16 tl.dot (verified:
# 1.0 * 1.0 -> 2.6e8). Under TRITON_INTERPRET the operands are upcast to fp32 (exact for bf16 values;
# products exact in fp32; fp32 accumulate) -- the same math the bf16 MMA does, so it tests the TILING
# and ORDER contract; the on-GPU bytes of the bf16 MMA are the HW2 probe's job (hw2_invariant_cost.py).
INTERPRET = os.environ.get("TRITON_INTERPRET", "0") not in ("", "0")
NUMERICS_PIN = f"mxfp4-w4a16-grouped/v1/bm{BLOCK_M}-bn{BLOCK_N}-bk{BLOCK_K}-sk1-w{NUM_WARPS}-s{NUM_STAGES}"


@triton.jit
def _e2m1_to_f32(code):
    """4-bit e2m1 code (int32, 0..15) -> fp32 exactly: {0,.5,1,1.5,2,3,4,6}, bit 3 = sign."""
    mag = code & 7
    e = mag >> 1
    m = (mag & 1).to(tl.float32)
    pow2 = (1 << tl.maximum(e - 1, 0)).to(tl.float32)            # 1,1,2,4 -- integer shift, exact
    v = tl.where(e == 0, m * 0.5, (1.0 + 0.5 * m) * pow2)
    return tl.where((code & 8) != 0, -v, v)


@triton.jit
def f32_to_bf16_rne(x):
    """fp32 -> bf16, round-to-nearest-even, by integer bit arithmetic (finite inputs). Identical on
    every backend; Triton 3.6's interpreter implements `.to(tl.bfloat16)` as TRUNCATION (verified),
    so the explicit form is what makes CPU-interpreted bytes mean anything."""
    b = x.to(tl.uint32, bitcast=True)
    b = b + 0x7FFF + ((b >> 16) & 1)
    return (b >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _mxfp4_grouped_gemm_kernel(
    a_ptr, w_ptr, s_ptr, out_ptr, sorted_ids_ptr, expert_ids_ptr,
    N, K, num_rows,
    stride_am, stride_we, stride_wn, stride_se, stride_sn, stride_om, stride_os,
    A_DIV: tl.constexpr,          # a row = sorted_id // A_DIV (top_k for gate/up, 1 for down)
    SPLIT_K: tl.constexpr,        # 1 in receipt-grade mode; >1 only for the naive control
    K_ITERS: tl.constexpr,        # BLOCK_K steps per program (= K / BLOCK_K when SPLIT_K == 1)
    DOT_F32: tl.constexpr,        # interpreter workaround only (see INTERPRET)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    e = tl.load(expert_ids_ptr + pid_m)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ids = tl.load(sorted_ids_ptr + offs_m)
    row_ok = ids < num_rows
    a_rows = tl.where(row_ok, ids // A_DIV, 0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_ok = offs_n < N
    half = tl.arange(0, BLOCK_K // 2)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, K_ITERS):                                  # ascending K, fixed step
        k0 = (pid_k * K_ITERS + i) * BLOCK_K
        kk = k0 + offs_k
        a = tl.load(a_ptr + a_rows[:, None] * stride_am + kk[None, :],
                    mask=row_ok[:, None] & (kk[None, :] < K), other=0.0)
        # packed bytes for this tile, laid out [BLOCK_K/2, BLOCK_N]
        kb = k0 // 2 + half
        wb = tl.load(w_ptr + e * stride_we + offs_n[None, :] * stride_wn + kb[:, None],
                     mask=n_ok[None, :] & (kb[:, None] < K // 2), other=0).to(tl.int32)
        lo = _e2m1_to_f32(wb & 15)
        hi = _e2m1_to_f32((wb >> 4) & 15)
        w = tl.reshape(tl.permute(tl.join(lo, hi), (0, 2, 1)), (BLOCK_K, BLOCK_N))  # rows 2i, 2i+1
        sc = tl.load(s_ptr + e * stride_se + offs_n[None, :] * stride_sn + (kk[:, None] // 32),
                     mask=n_ok[None, :] & (kk[:, None] < K), other=127).to(tl.int32)
        scale = tl.where(sc == 0, 0x00400000, sc << 23).to(tl.float32, bitcast=True)   # 2^(s-127), exact
        wbf = f32_to_bf16_rne(w * scale)                         # exact: e2m1 x 2^k fits bf16
        if DOT_F32:
            acc = tl.dot(a.to(tl.float32), wbf.to(tl.float32), acc, input_precision="ieee")
        else:
            acc = tl.dot(a, wbf, acc)
    out_off = pid_k * stride_os + ids[:, None] * stride_om + offs_n[None, :]
    omask = row_ok[:, None] & n_ok[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptr + out_off, f32_to_bf16_rne(acc), mask=omask)
    else:   # the Triton split-K recipe: partials land in the OUTPUT dtype (atomic_add of acc.to(c_dtype))
        tl.store(out_ptr + out_off, f32_to_bf16_rne(acc).to(tl.float32), mask=omask)


def align_by_expert(topk_ids: torch.Tensor, n_experts: int, block_m: int = BLOCK_M):
    """moe_align_block_size equivalent (host-side, deterministic). Flat row r = token*top_k + slot.
    Returns (sorted_ids [n_blocks*block_m] padded with the sentinel T*top_k, expert_ids [n_blocks]).
    Row order within an expert is ascending r; it does not affect any row's bytes (rows independent)."""
    flat = topk_ids.reshape(-1).to(torch.int64)
    sentinel = flat.numel()
    ids, experts = [], []
    for ex in range(n_experts):
        rows = torch.nonzero(flat == ex).flatten()
        if rows.numel() == 0:
            continue
        nb = -(-rows.numel() // block_m)
        pad = torch.full((nb * block_m - rows.numel(),), sentinel, dtype=torch.int64)
        ids.append(torch.cat([rows, pad]))
        experts += [ex] * nb
    dev = topk_ids.device
    if not ids:
        return torch.empty(0, dtype=torch.int32, device=dev), torch.empty(0, dtype=torch.int32, device=dev)
    return (torch.cat(ids).to(torch.int32).to(dev), torch.tensor(experts, dtype=torch.int32, device=dev))


def mxfp4_grouped_gemm(a: torch.Tensor, w_packed: torch.Tensor, w_scale: torch.Tensor,
                       topk_ids: torch.Tensor, a_div: int, split_k: int = 1) -> torch.Tensor:
    """out[r] = bf16( a[r // a_div] @ dequant(W[topk_ids.flat[r]]).T ), r over T*top_k rows.

    a: bf16 [rows_a, K]; w_packed U8 [E, N, K/2]; w_scale U8 [E, N, K/32]. split_k != 1 is the
    NON-invariant control (partials summed afterwards) -- never use it for receipts."""
    assert a.dtype == torch.bfloat16 and a.is_contiguous()
    E, N, kh = w_packed.shape
    K = 2 * kh
    assert a.shape[1] == K and K % BLOCK_K == 0 and w_scale.shape == (E, N, K // MX_BLOCK)
    w_packed, w_scale = w_packed.contiguous(), w_scale.contiguous()
    num_rows = topk_ids.numel()
    sorted_ids, expert_ids = align_by_expert(topk_ids, E)
    if split_k == 1:
        out = torch.empty(num_rows, N, dtype=torch.bfloat16, device=a.device)
        stride_os = 0
    else:
        out = torch.zeros(split_k, num_rows, N, dtype=torch.float32, device=a.device)
        stride_os = out.stride(0)
    if expert_ids.numel():
        grid = (expert_ids.numel(), triton.cdiv(N, BLOCK_N), split_k)
        _mxfp4_grouped_gemm_kernel[grid](
            a, w_packed, w_scale, out, sorted_ids, expert_ids, N, K, num_rows,
            a.stride(0), w_packed.stride(0), w_packed.stride(1), w_scale.stride(0), w_scale.stride(1),
            N, stride_os, A_DIV=a_div, SPLIT_K=split_k, K_ITERS=triton.cdiv(K // BLOCK_K, split_k),
            DOT_F32=INTERPRET,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    if split_k == 1:
        return out
    while out.shape[0] > 1:                  # split-K partials reduced as a pairwise tree (cuBLAS-style)
        out = out[0::2] + out[1::2]
    return out[0].to(torch.bfloat16)


def naive_split_k(num_tokens: int) -> int:
    """The kind of heuristic fast kernels use (more K-parallelism when M is small). Batch-DEPENDENT:
    used only by the anti-vacuity control."""
    return 4 if num_tokens <= 4 else 1


def moe_experts_forward(x: torch.Tensor, topk_ids: torch.Tensor, topk_w: torch.Tensor,
                        w13_packed: torch.Tensor, w13_scale: torch.Tensor,
                        w2_packed: torch.Tensor, w2_scale: torch.Tensor,
                        naive: bool = False) -> torch.Tensor:
    """Routed-expert FFN with the oracle's numerics (mimo_ref.Layer.ffn/_mlp):
    g,u = bf16(x @ Wg.T), bf16(x @ Wu.T); h = bf16(silu(g) * u) in fp32; y = bf16(h @ Wd.T);
    out = bf16( sum over the token's experts in ascending id of fp32(y) * w ).

    x bf16 [T, H]; topk_ids [T, k] (distinct per token); topk_w fp32 [T, k];
    w13 = gate rows then up rows: U8 [E, 2I, H/2] / [E, 2I, H/32]; w2 U8 [E, H, I/2] / [E, H, I/32]."""
    T, top_k = topk_ids.shape
    inter = w13_packed.shape[1] // 2
    sk = naive_split_k(T) if naive else 1
    gu = mxfp4_grouped_gemm(x, w13_packed, w13_scale, topk_ids, a_div=top_k, split_k=sk)   # [T*k, 2I]
    g = gu[:, :inter].float()
    u = gu[:, inter:].float()
    h = (torch.nn.functional.silu(g) * u).to(torch.bfloat16).contiguous()
    y = mxfp4_grouped_gemm(h, w2_packed, w2_scale, topk_ids, a_div=1, split_k=sk)          # [T*k, H]
    y = y.view(T, top_k, -1)
    order = torch.argsort(topk_ids, dim=1)                       # ascending expert id per token
    acc = torch.zeros(T, y.shape[-1], dtype=torch.float32, device=x.device)
    for j in range(top_k):                                       # elementwise, fixed order
        sl = order[:, j]
        yj = y[torch.arange(T, device=x.device), sl].float()
        wj = topk_w.gather(1, sl[:, None])
        acc = acc + yj * wj
    return acc.to(torch.bfloat16)
