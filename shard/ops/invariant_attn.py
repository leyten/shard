"""Receipt-grade fixed-split diff-kv attention (QK head dim 192, V head dim 128): batch-invariant.

Semantics (= scratchpad mimo_ref.Layer.attention inner loop): query token t of session `seq` at absolute
position p attends to exactly the keys at positions [lo, p], lo = max(0, p - window + 1) for a sliding-window
layer, 0 for a global one; an optional learned per-q-head sink logit joins the softmax with a zero value
vector. q/k arrive post-RoPE (partial RoPE on the first 64 dims is the caller's elementwise op), v already
scaled. Prefill, verify and decode segments are all just runs of query tokens with (seq, pos): the kernel
does not know or care which kind a token came from.

Why the bytes of token t cannot depend on the batch or on how the KV was written:
  * One program per (query token, kv head, split): a token's work is a function of (its q row, its seq's
    KV bytes, lo, p) only. No other token's data or count enters its arithmetic.
  * FIXED split size along the KV axis at ABSOLUTE positions: split s = [s*SPLIT, (s+1)*SPLIT). A token
    uses splits lo//SPLIT .. p//SPLIT. Inside a split, BLOCK_KV tiles at fixed absolute offsets, online
    softmax in ascending position order (m starts at a finite -1e30 so a fully masked tile is an exact
    no-op). The page size / page placement only changes WHERE a key is loaded from, never the order.
  * Merge of the splits: a separate program per (token, kv head) walks the splits in ascending order; the
    padding iterations (the loop bound is the batch's max split count, rounded up to a power of two) are
    discarded with a select (tl.where), not added, so they are exact no-ops (even for -0.0).
  * No atomics, no split count chosen from the batch, no cross-program reduction other than the fixed merge.
  * QK over 192 dims = two dots (dims 0..127 then 128..191) in that fixed order; P is cast to bf16 for the
    PV dot (FlashAttention practice; the tolerance vs the fp32-PV oracle absorbs it).

Chunk invariance is a bonus here (the runtime contract does not promise it): a token decoded alone and the
same token inside a prefill/verify segment produce the same bytes, because the per-token program never sees
its neighbours.

Device families: Triton NVIDIA sm_80+ (bf16 mma; Ampere has it) and AMD ROCm (untested); not Apple (no Triton
Metal backend -- MLX needs a Metal kernel with the same per-token/fixed-split structure;
mx.fast.scaled_dot_product_attention picks kernels by shape and has no sink/192-128 contract).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .invariant_moe import INTERPRET, f32_to_bf16_rne

D_QK, D_V = 192, 128
D_A, D_B = 128, 64                   # QK dims split into two power-of-two dots, fixed order
BLOCK_H = 16                         # q heads per kv head, padded (tl.dot needs M >= 16)
BLOCK_KV = 16                        # keys per inner tile
SPLIT_KV = 256                       # positions per split (absolute boundaries); pinned
NEG_INIT = -1.0e30
NUMERICS_PIN = f"diffkv-attn/v1/split{SPLIT_KV}-bkv{BLOCK_KV}-bh{BLOCK_H}-qk128+64-pbf16"


@triton.jit
def _attn_split_kernel(
    q_ptr, k_ptr, v_ptr, bt_ptr, seq_ptr, pos_ptr, m_ptr, l_ptr, acc_ptr,
    scaling,
    stride_qt, stride_qh, stride_kp, stride_ks, stride_kh, stride_vp, stride_vs, stride_vh,
    stride_bt, stride_mt, stride_mh, stride_ms, stride_at, stride_ah, stride_as, stride_ai,
    GRP: tl.constexpr, WINDOW: tl.constexpr, PAGE: tl.constexpr, SPLIT: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_KV: tl.constexpr, D_A: tl.constexpr, D_B: tl.constexpr,
    D_V: tl.constexpr, DOT_F32: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    sidx = tl.program_id(2)
    seq = tl.load(seq_ptr + t)
    pos = tl.load(pos_ptr + t)
    if WINDOW > 0:
        lo = tl.maximum(pos - WINDOW + 1, 0)
    else:
        lo = pos * 0
    s = lo // SPLIT + sidx
    split_ok = s <= pos // SPLIT
    hi_ = tl.arange(0, BLOCK_H)
    h_ok = hi_ < GRP
    qh = h * GRP + hi_
    da = tl.arange(0, D_A)
    db = tl.arange(0, D_B)
    dv = tl.arange(0, D_V)
    qbase = q_ptr + t * stride_qt + qh[:, None] * stride_qh
    qa = tl.load(qbase + da[None, :], mask=h_ok[:, None], other=0.0)
    qb = tl.load(qbase + D_A + db[None, :], mask=h_ok[:, None], other=0.0)
    m = tl.full((BLOCK_H,), NEG_INIT, tl.float32)
    lsum = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, D_V), tl.float32)
    for j in range(SPLIT // BLOCK_KV):                           # ascending absolute positions
        kp = s * SPLIT + j * BLOCK_KV + tl.arange(0, BLOCK_KV)
        vis = (kp >= lo) & (kp <= pos) & split_ok
        page = tl.load(bt_ptr + seq * stride_bt + kp // PAGE, mask=vis, other=0)
        slot = kp % PAGE
        kbase = k_ptr + page[:, None] * stride_kp + slot[:, None] * stride_ks + h * stride_kh
        ka = tl.load(kbase + da[None, :], mask=vis[:, None], other=0.0)
        kb = tl.load(kbase + D_A + db[None, :], mask=vis[:, None], other=0.0)
        vv = tl.load(v_ptr + page[:, None] * stride_vp + slot[:, None] * stride_vs + h * stride_vh
                     + dv[None, :], mask=vis[:, None], other=0.0)
        sc = tl.zeros((BLOCK_H, BLOCK_KV), tl.float32)
        if DOT_F32:
            sc = tl.dot(qa.to(tl.float32), tl.trans(ka.to(tl.float32)), sc, input_precision="ieee")
            sc = tl.dot(qb.to(tl.float32), tl.trans(kb.to(tl.float32)), sc, input_precision="ieee")
        else:
            sc = tl.dot(qa, tl.trans(ka), sc)
            sc = tl.dot(qb, tl.trans(kb), sc)
        sc = tl.where(vis[None, :], sc * scaling, float("-inf"))
        m_new = tl.maximum(m, tl.max(sc, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(sc - m_new[:, None])
        lsum = lsum * alpha + tl.sum(p, axis=1)
        pb = f32_to_bf16_rne(p)
        if DOT_F32:
            acc = tl.dot(pb.to(tl.float32), vv.to(tl.float32), acc * alpha[:, None], input_precision="ieee")
        else:
            acc = tl.dot(pb, vv, acc * alpha[:, None])
        m = m_new
    ok = h_ok & split_ok
    moff = t * stride_mt + h * stride_mh + sidx * stride_ms + hi_
    tl.store(m_ptr + moff, m, mask=ok)
    tl.store(l_ptr + moff, lsum, mask=ok)
    aoff = t * stride_at + h * stride_ah + sidx * stride_as + hi_[:, None] * stride_ai + dv[None, :]
    tl.store(acc_ptr + aoff, acc, mask=ok[:, None])


@triton.jit
def _attn_merge_kernel(
    pos_ptr, m_ptr, l_ptr, acc_ptr, sink_ptr, out_ptr,
    stride_mt, stride_mh, stride_ms, stride_at, stride_ah, stride_as, stride_ai, stride_ot, stride_oh,
    GRP: tl.constexpr, WINDOW: tl.constexpr, SPLIT: tl.constexpr, HAS_SINK: tl.constexpr,
    MERGE_ITERS: tl.constexpr, BLOCK_H: tl.constexpr, D_V: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    pos = tl.load(pos_ptr + t)
    if WINDOW > 0:
        lo = tl.maximum(pos - WINDOW + 1, 0)
    else:
        lo = pos * 0
    n_splits = pos // SPLIT - lo // SPLIT + 1
    hi_ = tl.arange(0, BLOCK_H)
    h_ok = hi_ < GRP
    dv = tl.arange(0, D_V)
    mbase = t * stride_mt + h * stride_mh + hi_
    abase = t * stride_at + h * stride_ah + hi_[:, None] * stride_ai + dv[None, :]
    if HAS_SINK:
        sink = tl.load(sink_ptr + h * GRP + hi_, mask=h_ok, other=0.0)
        m_fin = sink
    else:
        m_fin = tl.full((BLOCK_H,), float("-inf"), tl.float32)
    for i in range(MERGE_ITERS):                                 # max is exact and order-free
        ok = (i < n_splits) & h_ok
        m_i = tl.load(m_ptr + mbase + i * stride_ms, mask=ok, other=float("-inf"))
        m_fin = tl.maximum(m_fin, m_i)
    if HAS_SINK:
        l_fin = tl.exp(sink - m_fin)
    else:
        l_fin = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, D_V), tl.float32)
    for i in range(MERGE_ITERS):                                 # ascending split order, fixed
        ok = (i < n_splits) & h_ok
        m_i = tl.load(m_ptr + mbase + i * stride_ms, mask=ok, other=0.0)
        l_i = tl.load(l_ptr + mbase + i * stride_ms, mask=ok, other=0.0)
        a_i = tl.load(acc_ptr + abase + i * stride_as, mask=ok[:, None], other=0.0)
        w = tl.exp(m_i - m_fin)
        l_fin = tl.where(ok, l_fin + l_i * w, l_fin)             # select, not add: padding = exact no-op
        acc = tl.where(ok[:, None], acc + a_i * w[:, None], acc)
    o = acc / l_fin[:, None]
    tl.store(out_ptr + t * stride_ot + (h * GRP + hi_[:, None]) * stride_oh + dv[None, :],
             f32_to_bf16_rne(o), mask=h_ok[:, None])


def paged_diffkv_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                           block_table: torch.Tensor, tok_seq: torch.Tensor, tok_pos: torch.Tensor,
                           window: int = 0, sinks: torch.Tensor | None = None,
                           scaling: float = D_QK ** -0.5, split: int = SPLIT_KV) -> torch.Tensor:
    """q bf16 [T, Hq, 192]; k_cache bf16 [P, PAGE, Hkv, 192]; v_cache bf16 [P, PAGE, Hkv, 128];
    block_table int32 [S, max_pages]; tok_seq/tok_pos int32 [T]; window 0 = global; sinks fp32 [Hq].
    Returns bf16 [T, Hq, 128]. `split` other than SPLIT_KV is a different numerics_id (tests use a small
    one to exercise several splits at interpreter-sized contexts; the naive control varies it per batch)."""
    T, Hq, dqk = q.shape
    _, page, Hkv, _ = k_cache.shape
    assert dqk == D_QK and v_cache.shape[-1] == D_V and Hq % Hkv == 0
    grp = Hq // Hkv
    assert grp <= BLOCK_H and split % BLOCK_KV == 0
    assert q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16
    q, k_cache, v_cache = q.contiguous(), k_cache.contiguous(), v_cache.contiguous()
    block_table = block_table.to(torch.int32).contiguous()
    tok_seq, tok_pos = tok_seq.to(torch.int32).contiguous(), tok_pos.to(torch.int32).contiguous()
    out = torch.empty(T, Hq, D_V, dtype=torch.bfloat16, device=q.device)
    if T == 0:
        return out
    lo = (tok_pos - window + 1).clamp(min=0) if window > 0 else torch.zeros_like(tok_pos)
    n_spl = int((tok_pos // split - lo // split + 1).max())
    m = torch.empty(T, Hkv, n_spl, BLOCK_H, dtype=torch.float32, device=q.device)
    lsum = torch.empty_like(m)
    acc = torch.empty(T, Hkv, n_spl, BLOCK_H, D_V, dtype=torch.float32, device=q.device)
    _attn_split_kernel[(T, Hkv, n_spl)](
        q, k_cache, v_cache, block_table, tok_seq, tok_pos, m, lsum, acc, scaling,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), block_table.stride(0),
        m.stride(0), m.stride(1), m.stride(2), acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
        GRP=grp, WINDOW=window, PAGE=page, SPLIT=split, BLOCK_H=BLOCK_H, BLOCK_KV=BLOCK_KV,
        D_A=D_A, D_B=D_B, D_V=D_V, DOT_F32=INTERPRET, num_warps=4)
    sink_arg = sinks.to(torch.float32).contiguous() if sinks is not None else m
    _attn_merge_kernel[(T, Hkv)](
        tok_pos, m, lsum, acc, sink_arg, out,
        m.stride(0), m.stride(1), m.stride(2), acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
        out.stride(0), out.stride(1),
        GRP=grp, WINDOW=window, SPLIT=split, HAS_SINK=sinks is not None,
        MERGE_ITERS=triton.next_power_of_2(n_spl), BLOCK_H=BLOCK_H, D_V=D_V, num_warps=4)
    return out


def naive_split(num_tokens: int) -> int:
    """FlashDecoding-style heuristic: split the KV finer when the batch is small. Batch-DEPENDENT --
    used only by the anti-vacuity control."""
    return 32 if num_tokens <= 2 else 128
