"""Known-answer tests for the checkpoint weight formats (shard/formats), CPU only, no weights.

Ported from the MiMo CPU oracle's suite (scratchpad/v2/mimo/test_mimo_ref.py, which also ran these
primitives against the REAL MiMo-V2.6-Flash layer-0/1 tensors). Here every check is synthetic, and
the independent opinions come along as verbatim ports: vLLM v0.30.0's qkv row gather and llama.cpp's
TP-aware qkv dequant (both from scratchpad/v2/mimo/vendor_oracle.py), and transformers' gpt-oss MXFP4
decoder when it is importable.

Run: python3 -m pytest tests/test_formats.py -q
"""
import math
import os
import sys
from fractions import Fraction

import pytest
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from shard.formats.mimo_v2 import qkv_rechunk          # noqa: E402
from shard.formats.quant import (FP4_E2M1, check_keep_fp32, dequant_fp8_block,  # noqa: E402
                                 dequant_mxfp4, e8m0_to_fp32, keep_fp32, to_bf16_values)

LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


# ── MXFP4 + E8M0 ─────────────────────────────────────────────────────────────────────────────────

def test_mxfp4_known_answer():
    """One row per case, K=64 (two 32-element blocks). Byte = (hi << 4) | lo; element 2i = lo."""
    packed = torch.zeros(2, 32, dtype=torch.uint8)
    packed[0, 0] = 0x21   # lo 1 -> +0.5, hi 2 -> +1.0
    packed[0, 1] = 0x9F   # lo F -> -6.0, hi 9 -> -0.5
    packed[0, 2] = 0x07   # lo 7 -> +6.0, hi 0 -> +0.0
    packed[0, 16] = 0xC3  # first byte of block 1: lo 3 -> +1.5, hi C -> -2.0
    packed[1, 0] = 0x5E   # lo E -> -4.0, hi 5 -> +3.0
    scale = torch.tensor([[127, 125], [128, 127]], dtype=torch.uint8)  # 2^0, 2^-2 | 2^1, 2^0
    w = dequant_mxfp4(packed, scale)
    assert w.shape == (2, 64) and w.dtype == torch.float32
    assert w[0, :6].tolist() == [0.5, 1.0, -6.0, -0.5, 6.0, 0.0]
    assert w[0, 32:34].tolist() == [1.5 * 0.25, -2.0 * 0.25]
    assert w[1, :2].tolist() == [-4.0 * 2, 3.0 * 2]
    assert torch.count_nonzero(w) == 9                    # everything else decodes to (+/-)0


def test_mxfp4_full_code_table_both_nibbles():
    b = torch.arange(256, dtype=torch.uint8).reshape(8, 32)
    full = dequant_mxfp4(b, torch.full((8, 2), 127, dtype=torch.uint8)).reshape(-1)
    assert full[0::2].tolist() == [LUT[i & 15] for i in range(256)]
    assert full[1::2].tolist() == [LUT[i >> 4] for i in range(256)]
    assert torch.equal(torch.signbit(full[0::2]), torch.tensor([bool(i & 8) for i in range(256)]))


def test_e8m0_is_exact_at_both_ends_and_refuses_nan():
    s = torch.tensor([0, 1, 126, 127, 128, 254], dtype=torch.uint8)
    got = e8m0_to_fp32(s).tolist()
    assert got == [2.0 ** -127, 2.0 ** -126, 0.5, 1.0, 2.0, 2.0 ** 127]
    with pytest.raises(ValueError, match="NaN"):
        e8m0_to_fp32(torch.tensor([255], dtype=torch.uint8))
    with pytest.raises(ValueError, match="NaN"):
        dequant_mxfp4(torch.zeros(2, 16, dtype=torch.uint8), torch.full((2, 1), 255, dtype=torch.uint8))


def test_mxfp4_overflow_refused_not_inf():
    """6 x 2^(254-127) overflows fp32 (and bf16): refuse, never hand a loader an inf weight."""
    packed = torch.zeros(1, 16, dtype=torch.uint8)
    packed[0, 0] = 0x07                                   # +6.0
    with pytest.raises(ValueError, match="overflow"):
        dequant_mxfp4(packed, torch.full((1, 1), 254, dtype=torch.uint8))
    packed[0, 0] = 0x06                                   # +4.0 x 2^126 = 2^128: overflows too
    with pytest.raises(ValueError, match="overflow"):
        dequant_mxfp4(packed, torch.full((1, 1), 253, dtype=torch.uint8))
    packed[0, 0] = 0x05                                   # +3.0 x 2^126 fits
    assert dequant_mxfp4(packed, torch.full((1, 1), 253, dtype=torch.uint8))[0, 0].item() == 3.0 * 2.0 ** 126


def test_mxfp4_every_value_is_bf16_exact_and_leading_dims_work():
    g = torch.Generator().manual_seed(0)
    packed = torch.randint(0, 256, (3, 5, 64), dtype=torch.uint8, generator=g)     # [E, N, K/2]
    scale = torch.randint(0, 253, (3, 5, 4), dtype=torch.uint8, generator=g)       # [E, N, K/32]
    try:
        w = dequant_mxfp4(packed, scale)
    except ValueError:                                    # a 252 scale x 6 can still fit; retry lower
        scale = scale.clamp(max=250)
        w = dequant_mxfp4(packed, scale)
    assert w.shape == (3, 5, 128)
    assert torch.equal(w.to(torch.bfloat16).to(torch.float32), w)
    for e in range(3):                                    # leading dims are independent 2-D problems
        assert torch.equal(w[e], dequant_mxfp4(packed[e], scale[e]))


def test_mxfp4_shape_mismatch_refused():
    with pytest.raises(ValueError, match="shapes"):
        dequant_mxfp4(torch.zeros(2, 32, dtype=torch.uint8), torch.zeros(2, 3, dtype=torch.uint8))


def test_mxfp4_matches_transformers_gpt_oss_decoder():
    """A third-party decoder of the same OCP layout (the one vLLM's Mxfp4MoEMethod consumes)."""
    mx = pytest.importorskip("transformers.integrations.mxfp4")
    g = torch.Generator().manual_seed(1)
    n, kh = 16, 64
    packed = torch.randint(0, 256, (n, kh), dtype=torch.uint8, generator=g)
    scale = torch.randint(100, 150, (n, kh // 16), dtype=torch.uint8, generator=g)
    ref = mx.convert_moe_packed_tensors(packed.view(1, n, kh // 16, 16), scale.view(1, n, kh // 16),
                                        dtype=torch.float32)[0].transpose(0, 1)
    assert torch.equal(dequant_mxfp4(packed, scale), ref)


# ── FP8 e4m3 block ───────────────────────────────────────────────────────────────────────────────

def _e4m3(vals):
    f = torch.tensor(vals, dtype=torch.float32)
    t = f.to(torch.float8_e4m3fn)
    assert torch.equal(t.to(torch.float32), f), "test values must be e4m3-exact"
    return t


def test_fp8_block_known_answer_plain_grid():
    """w[r, c] * s[r // 128, c // 128]; the scale is a multiplier despite the `_inv` name."""
    w = _e4m3([[1.0] * 256] * 130)                        # 130 rows -> 2 scale rows (ceil)
    s = torch.tensor([[2.0, 0.5], [3.0, 0.25]], dtype=torch.float32)
    d = dequant_fp8_block(w, s)
    assert d[0, 0] == 2.0 and d[0, 128] == 0.5 and d[127, 255] == 0.5
    assert d[128, 0] == 3.0 and d[129, 200] == 0.25
    with pytest.raises(ValueError, match="plain grid"):
        dequant_fp8_block(w, torch.ones(3, 2))            # 3 scale rows for 130 weight rows
    with pytest.raises(ValueError, match="tile K"):
        dequant_fp8_block(w, torch.ones(2, 3))


def test_fp8_block_refuses_nan_codes():
    w = torch.full((128, 128), 0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn)   # e4m3fn NaN
    with pytest.raises(ValueError, match="non-finite"):
        dequant_fp8_block(w, torch.ones(1, 1))


def test_bf16_rounding_is_round_to_nearest_even():
    """The canonical FP8 weight is rounded ONCE to bf16, RNE. Ties go to the even mantissa; a
    truncating (or round-half-up) implementation gets these wrong."""
    w = _e4m3([[1.0] * 128])
    tie_down = 1.0 + 2 ** -8           # halfway between 1.0 (even) and 1+2^-7 (odd) -> 1.0
    tie_up = 1.0 + 3 * 2 ** -8         # halfway between 1+2^-7 (odd) and 1+2^-6 (even) -> 1+2^-6
    above = 1.0 + 2 ** -8 + 2 ** -20   # just above a tie -> up
    for sv, want in ((tie_down, 1.0), (tie_up, 1.0 + 2 ** -6), (above, 1.0 + 2 ** -7)):
        got = to_bf16_values(dequant_fp8_block(w, torch.tensor([[sv]], dtype=torch.float32)))
        assert got[0, 0].item() == want, (sv, got[0, 0].item(), want)


def _rne(x: Fraction, mant_bits: int) -> Fraction:
    """Round a positive rational to `mant_bits` significand bits, ties to even (normal range)."""
    e = math.floor(math.log2(x))
    if Fraction(2) ** e > x:
        e -= 1
    if Fraction(2) ** (e + 1) <= x:
        e += 1
    q = x / Fraction(2) ** (e - mant_bits + 1)
    fl = math.floor(q)
    r = q - fl
    n = fl + (1 if r > Fraction(1, 2) or (r == Fraction(1, 2) and fl % 2) else 0)
    return n * Fraction(2) ** (e - mant_bits + 1)


def test_fp8_dequant_is_fp32_multiply_then_bf16_rne():
    """Pin the exact definition: IEEE fp32 multiply (RNE to 24 bits), then RNE to bf16 (8 bits).
    The product of a 4-bit e4m3 significand and a 24-bit fp32 scale is NOT always fp32-exact, so the
    'computed exactly in fp32' phrasing of the scratch oracle was wrong; this is the real contract."""
    g = torch.Generator().manual_seed(2)
    codes = torch.randint(0, 0x7E, (128, 128), dtype=torch.uint8, generator=g)     # positive, no NaN
    codes[codes == 0] = 0x38                                                        # avoid zeros (1.0)
    w = codes.view(torch.float8_e4m3fn)
    # a scale with a FULL 24-bit significand (torch.rand leaves low bits zero): 1.m, random m
    mant = torch.randint(1, 1 << 23, (1, 1), dtype=torch.int32, generator=g)
    s = (torch.full((1, 1), 0x3F800000, dtype=torch.int32) | mant | 1).view(torch.float32)
    got = to_bf16_values(dequant_fp8_block(w, s))
    sv = Fraction(s.item())
    inexact = 0
    for wv, gv in zip(w.to(torch.float32).reshape(-1)[:4096].tolist(), got.reshape(-1)[:4096].tolist()):
        exact = Fraction(wv) * sv
        p32 = _rne(exact, 24)
        inexact += p32 != exact
        assert Fraction(gv) == _rne(p32, 8), (wv, s.item())
    assert inexact > 0, "sample should include products an fp32 multiply has to round"


# ── MiMo-V2 fused qkv ────────────────────────────────────────────────────────────────────────────

def _fake_interleaved_qkv(n_q, n_kv, qk, v, tp, hidden, seed=0):
    """A checkpoint-style interleaved FP8 qkv + per-chunk scale grid, built from known Q/K/V."""
    g = torch.Generator().manual_seed(seed)
    rpc = (n_q * qk + n_kv * qk + n_kv * v) // tp
    cs = math.ceil(rpc / 128)
    w = (torch.randn(tp * rpc, hidden, generator=g) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)
    s = torch.rand(tp * cs, hidden // 128, generator=g) + 0.5
    deq = torch.empty(tp * rpc, hidden)
    for c in range(tp):                                   # chunk c, its b-th 128-row block
        for b in range(cs):
            lo, hi = c * rpc + b * 128, c * rpc + min((b + 1) * 128, rpc)
            deq[lo:hi] = w[lo:hi].float() * s[c * cs + b].repeat_interleave(128)
    q_pc, k_pc = n_q // tp * qk, n_kv // tp * qk
    Q = torch.cat([deq[c * rpc: c * rpc + q_pc] for c in range(tp)])
    K = torch.cat([deq[c * rpc + q_pc: c * rpc + q_pc + k_pc] for c in range(tp)])
    V = torch.cat([deq[c * rpc + q_pc + k_pc: (c + 1) * rpc] for c in range(tp)])
    return w, s, Q, K, V


def _vllm_qkv_gather(w_full, s_full, num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp, block=128):
    """vLLM v0.30.0 models/mimo_v2.py::_shard_fp8_qkv_proj, tp_rank=0, tp_size=1, returning the
    dequantised `grouped` rows instead of re-quantising them (ported verbatim via vendor_oracle)."""
    rows_per_chunk = w_full.shape[0] // ckpt_tp
    q_per_chunk = (num_heads // ckpt_tp) * head_dim
    k_per_chunk = (num_kv_heads // ckpt_tp) * head_dim
    chunk_scale_rows = -(-rows_per_chunk // block)
    rows = torch.arange(w_full.shape[0])
    if s_full.shape[0] == ckpt_tp * chunk_scale_rows:
        scale_index = (rows // rows_per_chunk) * chunk_scale_rows + (rows % rows_per_chunk) // block
    elif s_full.shape[0] == -(-w_full.shape[0] // block):
        scale_index = rows // block
    else:
        raise ValueError("bad qkv scale shape")
    qh, kvh = num_heads // ckpt_tp, num_kv_heads // ckpt_tp
    row_index = []
    for head in range(num_heads):
        row_index.append((head // qh) * rows_per_chunk + (head % qh) * head_dim + torch.arange(head_dim))
    for head in range(num_kv_heads):
        row_index.append((head // kvh) * rows_per_chunk + q_per_chunk + (head % kvh) * head_dim
                         + torch.arange(head_dim))
    for head in range(num_kv_heads):
        row_index.append((head // kvh) * rows_per_chunk + q_per_chunk + k_per_chunk
                         + (head % kvh) * v_head_dim + torch.arange(v_head_dim))
    index = torch.cat(row_index)
    return w_full[index].to(torch.float32) * s_full[scale_index[index]].repeat_interleave(block, dim=1)


def _llamacpp_qkv_dequant(weight, scale_inv, n_q, n_kv, hd, vhd, bs=128):
    """llama.cpp conversion/mimo.py::_tp_aware_qkv_dequant (ported verbatim; a third opinion)."""
    q_size, k_size, v_size = n_q * hd, n_kv * hd, n_kv * vhd
    total_rows = q_size + k_size + v_size
    tp = None
    for cand in (8, 4):
        if total_rows % cand != 0:
            continue
        rpr = total_rows // cand
        if scale_inv.shape[0] == cand * ((rpr + bs - 1) // bs):
            tp = cand
            break
    assert tp is not None
    q_per, k_per, v_per = q_size // tp, k_size // tp, v_size // tp
    rows_per_rank = q_per + k_per + v_per
    blocks_per_rank = (rows_per_rank + bs - 1) // bs
    row_idx = torch.arange(total_rows)
    scale_row_idx = (row_idx // rows_per_rank) * blocks_per_rank + ((row_idx % rows_per_rank) // bs)
    dequant = weight.float() * scale_inv.float()[scale_row_idx].repeat_interleave(bs, dim=1)[:, : weight.shape[1]]
    qs, ks, vs = [], [], []
    for r in range(tp):
        base = r * rows_per_rank
        qs.append(dequant[base: base + q_per])
        ks.append(dequant[base + q_per: base + q_per + k_per])
        vs.append(dequant[base + q_per + k_per: base + rows_per_rank])
    return torch.cat(qs + ks + vs, dim=0)


@pytest.mark.parametrize("n_q,n_kv,tp,scale_rows", [(64, 4, 4, 108), (64, 8, 4, 116), (128, 8, 8, 216)],
                         ids=["flash-global-108", "flash-swa-116", "pro-216"])
def test_qkv_rechunk_synthetic(n_q, n_kv, tp, scale_rows):
    """The released grids: Flash global (108 rows, not the plain 106), Flash SWA/MTP (116, contiguous
    by accident but still interleaved), Pro (216, not 212). Known answer + two independent ports."""
    w, s, Q, K, V = _fake_interleaved_qkv(n_q, n_kv, 192, 128, tp, hidden=256)
    assert s.shape[0] == scale_rows
    q, k, v = qkv_rechunk(w, s, n_q, n_kv, 192, 128, ckpt_tp=tp)
    assert torch.equal(q, Q) and torch.equal(k, K) and torch.equal(v, V)
    mine = torch.cat([q, k, v])
    assert torch.equal(mine, _vllm_qkv_gather(w, s, n_q, n_kv, 192, 128, ckpt_tp=tp))
    assert torch.equal(mine, _llamacpp_qkv_dequant(w, s, n_q, n_kv, 192, 128))
    # negative control: the vendor HF file's plain split of a correctly-scaled dequant is wrong
    rpc = w.shape[0] // tp
    cs = math.ceil(rpc / 128)
    r = torch.arange(w.shape[0])
    full = dequant_fp8_block(w, s, (r // rpc) * cs + (r % rpc) // 128)
    assert not torch.equal(full[: n_q * 192], Q)


def test_qkv_rechunk_refuses_a_grid_it_does_not_know():
    w, s, *_ = _fake_interleaved_qkv(64, 4, 192, 128, 4, hidden=128)
    with pytest.raises(ValueError, match="per-chunk grid"):
        qkv_rechunk(w, s[:106], 64, 4, 192, 128, ckpt_tp=4)      # the naive plain grid
    with pytest.raises(ValueError, match="chunk"):
        qkv_rechunk(w, s, 64, 4, 192, 128, ckpt_tp=8)            # wrong checkpoint TP


# ── keep-fp32 rule ───────────────────────────────────────────────────────────────────────────────

def test_router_bias_must_stay_fp32():
    for name in ("model.layers.1.mlp.gate.e_score_correction_bias",                  # MiMo-V2
                 "model.layers.3.block_sparse_moe.e_score_correction_bias"):         # MiniMax-M2.5
        assert keep_fp32(name)
        b = torch.tensor([1.7, 2.2], dtype=torch.float32)
        assert check_keep_fp32(name, b) is b
        with pytest.raises(ValueError, match="must stay float32"):
            check_keep_fp32(name, b.to(torch.bfloat16))
    assert not keep_fp32("model.layers.1.mlp.gate.weight")
    assert check_keep_fp32("model.norm.weight", torch.ones(2, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_fp4_table_matches_the_ocp_codes():
    assert FP4_E2M1.tolist() == LUT
