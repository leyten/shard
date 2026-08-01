"""A CPU `kernel` module for DeepSeek-V4-Flash — so a V4 layer range can be proven correct GPU-less.

DeepSeek's reference decoder (phase0/deepseek_v4_ref/inference/model.py) hard-imports its kernels at
module scope:

    from kernel import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn

and `rotate_activation` reaches for a seventh, `fast_hadamard_transform.hadamard_transform`, inside
the function body. Every one of those is GPU-only: kernel.py is tilelang (`@tilelang.jit`, CUDA
codegen at first call), and fast_hadamard_transform ships a CUDA extension with no torch fallback.
Importing the reference on a CPU box therefore fails outright -- and a stage whose parity can only
be checked on rented hardware is a stage whose parity is never checked. So this module supplies the
same seven names in plain torch and registers them under `kernel` / `fast_hadamard_transform` in
sys.modules BEFORE the reference is imported. The vendored files stay byte-identical (see
deepseek_v4_ref/PROVENANCE.md); they just get the API they ask for.

The math is not invented here. Each function below is a transcription of the tilelang kernel of the
same name in phase0/deepseek_v4_ref/inference/kernel.py, MIT (c) DeepSeek. Where a constant looks
arbitrary it is theirs: the 1e-4 amax floor, the 448/6.0 format maxima, the 6*2^-126 fp4 floor, the
per-128 activation blocks against per-32 fp4 weight blocks, the sinkhorn's first-row-normalize being
a true softmax while every later one divides by (sum+eps).

THIS IS THE CPU REFERENCE, NOT A BIT-EMULATOR OF THE GPU KERNELS. The quantizers are exact -- they
are elementwise, and both sides round the same way -- but the GEMMs are not: tilelang accumulates
blockwise (128-wide K tiles into a separate scale-corrected accumulator, fp8_gemm_kernel:248) and
`torch.matmul` reassociates however BLAS feels like. That is fine, because the bar these stand-ins
have to clear is stage-vs-oracle parity where BOTH sides run THESE functions. A GPU box re-proves
the stage against real tilelang later; that comparison is allclose by construction, and pretending
otherwise here would only hide which of the two reassociations moved.

Two rounding rules are load-bearing enough to spell out, because getting them subtly wrong produces
a model that runs and is quietly off by one ulp of scale on a fraction of blocks:

  fast_round_scale   scale = 2^ceil(log2(amax/max)). kernel.py does it with IEEE bit surgery
                     (fast_log2_ceil: exponent - 127 + (mantissa != 0)), which is EXACT at powers of
                     two. `log2().ceil()` in float is not -- log2(2^k) can come back a hair over k
                     and round up to 2^(k+1), doubling the scale for exactly the inputs (448*2^k)
                     that make the tightest quantization. torch.frexp is the exact equivalent:
                     x = m*2^e with m in [0.5,1), so ceil(log2 x) = e-1 when m == 0.5 else e.

  e2m1 ties-to-even  the fp4 grid is {0,.5,1,1.5,2,3,4,6} as codes 0..7, and a midpoint rounds to
                     the neighbour with the EVEN code, which is what the hardware cast does:
                     0.25->0, 0.75->1.0, 1.25->1.0, 1.75->2.0, 2.5->2.0, 3.5->4.0, 5.0->4.0. Note
                     this is not "round half up" and not "round half away from zero"; two of the
                     seven midpoints go down and would silently bias the Indexer's scores.
"""
import os
import sys
import types

import torch

# Which V4 kernels the reference gets. "tilelang" is the real kernel.py next to model.py (needs a
# CUDA device -- tilelang codegens at first call, so importability alone proves nothing), "cpu" is
# this module. "auto" resolves to cpu exactly when there is no GPU to run the real ones on.
V4_KERNELS = os.environ.get("V4_KERNELS", "auto")

FP8_MAX = 448.0
FP4_MAX = 6.0
# e2m1, in code order: convert.py's FP4_TABLE[:8]. The negatives are codes 8..15, i.e. code | 0x8.
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MID = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


# ── the quantizers ───────────────────────────────────────────────────────────────────────────────

def _round_scale(amax, max_inv):
    """kernel.py's fast_round_scale: 2^ceil(log2(amax*max_inv)), exact at powers of two.

    `max_inv` is passed (not divided by) to match the kernel, which folds 1/448 into an fp32
    constant -- amax*(1/448) and amax/448 differ in the last bit for some amax, and that bit decides
    whether ceil() lands on k or k+1."""
    x = amax * max_inv
    mant, exp = torch.frexp(x)
    return torch.ldexp(torch.ones_like(x), torch.where(mant == 0.5, exp - 1, exp))


def _blocks(x, block_size):
    """[..., N] -> a contiguous fp32 COPY shaped [..., N//bs, bs], one row per scale.

    The reference hands us NON-CONTIGUOUS views (`act_quant(kv[..., :-rd], ...)` is a slice off the
    head dim), which is why the kernel wrapper does `z = x.contiguous()` and writes back through
    `x.copy_(y)` rather than in place. Same contract here."""
    n = x.size(-1)
    assert n % block_size == 0, f"v4_kernels_cpu: last dim {n} not a multiple of {block_size}"
    return x.contiguous().float().unflatten(-1, (n // block_size, block_size))


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """Block-wise fp8 (e4m3) quantization, one scale per row per `block_size` along the last dim.

    scale_fmt not None selects the power-of-two (MXFP) scale, i.e. round_scale in the kernel; the
    reference sets it from `scale_dtype == "fp8"`, since an e8m0 scale can hold nothing else.
    inplace=True is the QAT simulation the runtime path actually uses -- quantize and dequantize
    straight back into x's dtype, no fp8 tensor ever materialised."""
    z = _blocks(x, block_size)
    amax = z.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = _round_scale(amax, 1.0 / FP8_MAX) if scale_fmt is not None else amax * (1.0 / FP8_MAX)
    q = (z / s).clamp(-FP8_MAX, FP8_MAX)
    if inplace:
        y = (q.to(torch.float8_e4m3fn).float() * s).flatten(-2).to(x.dtype)
        x.copy_(y)
        return x
    return q.to(torch.float8_e4m3fn).flatten(-2), s.squeeze(-1).to(scale_dtype)


def _e2m1(a):
    """|a| in [0, 6] -> (nearest e2m1 value, its 3-bit code), ties to the even code. See the module
    docstring: bucketize(right=False) picks the low neighbour on an exact midpoint and right=True
    picks the high one, so they disagree only AT a midpoint -- and there the even code is the high
    one exactly when the low one is odd."""
    mid = torch.tensor(E2M1_MID, dtype=torch.float32, device=a.device)
    lo = torch.bucketize(a, mid, right=False)
    hi = torch.bucketize(a, mid, right=True)
    code = lo + ((hi != lo) & (lo % 2 == 1)).to(lo.dtype)
    return torch.tensor(E2M1, dtype=torch.float32, device=a.device)[code], code


def fp4_act_quant(x, block_size=32, inplace=False):
    """Block-wise fp4 (e2m1) quantization. The scale is ALWAYS power-of-two rounded -- it is stored
    as e8m0, which has no mantissa to hold anything else.

    The runtime CPU path only ever calls this inplace (Indexer q, and the Indexer's rotated
    Compressor kv); the packed form exists so a converted fp4 expert can be fed to fp4_gemm."""
    z = _blocks(x, block_size)
    amax = z.abs().amax(-1, keepdim=True).clamp_min(FP4_MAX * 2.0 ** -126)
    s = _round_scale(amax, 1.0 / FP4_MAX)
    q = (z / s).clamp(-FP4_MAX, FP4_MAX)
    val, code = _e2m1(q.abs())
    if inplace:
        y = (torch.where(q < 0, -val, val) * s).flatten(-2).to(x.dtype)
        x.copy_(y)
        return x
    code = (code | ((q < 0).to(code.dtype) * 8)).flatten(-2).to(torch.uint8)
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()
    return packed.view(torch.float4_e2m1fn_x2), s.squeeze(-1).to(torch.float8_e8m0fnu)


# ── the GEMMs ────────────────────────────────────────────────────────────────────────────────────

def _dequant(v, s, block):
    """[..., K] values x [..., K//block] scales -> fp32. e8m0 scales come back through .float()."""
    return v.float() * s.float().repeat_interleave(block, -1)[..., :v.size(-1)]


def unpack_fp4(b):
    """float4_e2m1fn_x2 [N, K//2] -> fp32 [N, K], LOW NIBBLE FIRST.

    Nibble order is convert.py's, not a guess: cast_e2m1fn_to_e4m3fn does
    `stack([TABLE[x & 0xF], TABLE[x >> 4]], -1).flatten(2)`, so byte i holds logical K positions
    (2i, 2i+1) in (low, high) order. Get this backwards and every fp4 expert is a transposed-nibble
    mess that still runs."""
    u = b.view(torch.uint8)
    table = torch.tensor(E2M1 + tuple(-v for v in E2M1), dtype=torch.float32, device=b.device)
    return torch.stack([table[(u & 0x0F).long()], table[(u >> 4).long()]], dim=-1).flatten(-2)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
    """C[M,N] = dequant(A[M,K]) @ dequant(B[N,K])^T. A is scaled per 1x128 on K, B per 128x128."""
    k, n = a.size(-1), b.size(0)
    da = _dequant(a.reshape(-1, k), a_s.reshape(-1, a_s.size(-1)), 128)
    db = _dequant(b, b_s.repeat_interleave(128, 0)[:n], 128)
    c = da @ db.t()
    return c.view(*a.shape[:-1], n).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
    """C[M,N] = dequant(A_fp8[M,K]) @ dequant(B_fp4[N,K])^T. A per 1x128 on K, B per 1x32 on K."""
    k, n = a.size(-1), b.size(0)
    da = _dequant(a.reshape(-1, k), a_s.reshape(-1, a_s.size(-1)), 128)
    db = unpack_fp4(b).float() * b_s.float().repeat_interleave(32, -1)[:, :k]
    c = da @ db.t()
    return c.view(*a.shape[:-1], n).to(torch.get_default_dtype())


# ── attention, hyper-connections, rotation ───────────────────────────────────────────────────────

SPARSE_ATTN_BLOCK = 64          # kernel.py sparse_attn_kernel's `block` -- the reduction quantum


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Gather-then-softmax attention over topk_idxs, with a learned per-head sink in the denominator.

    q [b,s,h,d] bf16, kv [b,n,d] bf16, attn_sink [h] fp32, topk_idxs [b,s,topk] int32 where -1 means
    "no position" (the kernel zeroes that KV row and pushes the score to -inf, so it contributes to
    neither the numerator nor the denominator).

    BLOCKED IN 64s, LIKE THE KERNEL, AND THAT IS LOAD-BEARING. kernel.py walks topk in
    `ceildiv(topk, 64)` fixed 64-wide blocks with a running max/sum, so out-of-range lanes are
    ALWAYS present (`idxs[i] = -1` past `topk`) and an extra all-masked block rescales by
    exp(0) == 1 and adds exactly 0 -- padding a topk list with -1 is BITWISE free on the GPU. A
    single flat `p.sum(-1)` / `p @ gathered` over the true width is the same function in exact
    arithmetic but NOT the same rounding: widening the reduction regroups its pairwise tree, so a
    31-wide list and the same list padded to 48 disagreed in the last bits on ~1% of calls and
    occasionally crossed a bf16 boundary. Anything that feeds this a fixed-width padded index list
    (v4_whole_layer_graph's capture-safe decode) would then fail a parity test the GPU passes. The
    block loop makes the emulation padding-invariant for the same reason the kernel is.

    NOT emulated: the head padding to 16 in kernel.py's wrapper (a GPU tiling detail -- it slices
    the padding straight back off), the bf16 cast of P before the second gemm (`acc_s_cast`), and
    the all-masked row, which the kernel leaves as NaN (scores_max stays -inf, exp(sink - -inf) is
    NaN) and which is treated here as an all-zero output. The reference never produces such a row:
    every query keeps at least its own window position."""
    b, s, h, d = q.shape
    topk = topk_idxs.size(-1)
    block = SPARSE_ATTN_BLOCK
    ar_b = torch.arange(b, device=kv.device)[:, None, None]
    acc_o = torch.zeros(b, s, h, d, dtype=torch.float32, device=q.device)
    sum_exp = torch.zeros(b, s, h, dtype=torch.float32, device=q.device)
    m = torch.full((b, s, h), float("-inf"), dtype=torch.float32, device=q.device)
    for t in range(0, max(topk, 1), block):
        n = min(block, topk - t)
        idx = topk_idxs.new_full((b, s, block), -1)
        if n > 0:
            idx[..., :n] = topk_idxs[..., t:t + n]
        idx = idx.long()
        valid = idx >= 0
        gathered = kv[ar_b, idx.clamp_min(0)].float()
        gathered = torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))
        scores = torch.einsum("bshd,bstd->bsht", q.float(), gathered) * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
        m_prev = m
        m = torch.maximum(m, scores.amax(-1))
        rescale = torch.exp(m_prev - m)
        rescale = torch.where(torch.isnan(rescale), torch.ones_like(rescale), rescale)
        p = torch.exp(scores - m.unsqueeze(-1))
        sum_exp = sum_exp * rescale + p.sum(-1)
        acc_o = acc_o * rescale.unsqueeze(-1) + torch.einsum("bsht,bstd->bshd", p, gathered)
    sum_exp = sum_exp + torch.exp(attn_sink.float().view(1, 1, h) - m)
    return (acc_o / sum_exp.unsqueeze(-1)).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Splits the hyper-connection mixer into (pre, post, comb) with comb Sinkhorn-normalized.

    mixes [b,s,(2+hc)*hc] fp32 -> pre [b,s,hc], post [b,s,hc], comb [b,s,hc,hc]. Transcribed from
    hc_split_sinkhorn_kernel: the FIRST row-normalize is a true softmax (its denominator has NO eps,
    and +eps lands on the result), every later normalize divides by (sum + eps), and the column
    normalize runs once more than the row one -- iters-1 full row/col rounds follow the opening
    softmax+column pass. Off-by-one there is a different fixed point, not a rounding difference."""
    hc = hc_mult
    mixes = mixes.float()
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = mixes[..., 2 * hc:].unflatten(-1, (hc, hc)) * hc_scale[2] + hc_base[2 * hc:].view(hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def hadamard_transform(x, scale=1.0):
    """fast_hadamard_transform's entry point: scale * (H_d @ x) along the last dim, H unnormalized.

    Sylvester order (H_d = H_2 tensor-power), which is what the CUDA extension computes and what
    makes the transform its own inverse at scale = d^-0.5 -- the scale `rotate_activation` passes.
    Iterative butterflies in fp32; d must be a power of two, as it is in the extension."""
    d = x.size(-1)
    assert d & (d - 1) == 0, f"v4_kernels_cpu: hadamard needs a power-of-two last dim, got {d}"
    y = x.float()
    h = 1
    while h < d:
        y = y.unflatten(-1, (d // (2 * h), 2, h))
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack([a + b, a - b], dim=-2).flatten(-3)
        h *= 2
    return (y * scale).to(x.dtype)


# ── registration ─────────────────────────────────────────────────────────────────────────────────

_MODULES = {
    "kernel": {"act_quant": act_quant, "fp4_act_quant": fp4_act_quant, "fp8_gemm": fp8_gemm,
               "fp4_gemm": fp4_gemm, "sparse_attn": sparse_attn,
               "hc_split_sinkhorn": hc_split_sinkhorn},
    "fast_hadamard_transform": {"hadamard_transform": hadamard_transform},
}


def backend():
    """Which kernels this box resolves to under V4_KERNELS. See the constant's comment."""
    if V4_KERNELS in ("cpu", "tilelang"):
        return V4_KERNELS
    if V4_KERNELS != "auto":
        raise RuntimeError(f"V4_KERNELS={V4_KERNELS!r} — expected auto, cpu or tilelang")
    return "cpu" if not torch.cuda.is_available() else "tilelang"


def _foreign(name):
    """Is something OTHER than ours already occupying `name` in this process?

    Only sys.modules counts, not importability. `kernel` is a maximally generic module name, so a
    live import of one is worth refusing to clobber -- but a kernel.py merely sitting on sys.path is
    the vendored tilelang file itself, which is the exact thing we are standing in for, and
    v4_ref_cpu puts that directory on the path on purpose."""
    mod = sys.modules.get(name)
    return mod is not None and not getattr(mod, "_v4_cpu_backend", False)


def install():
    """Give the reference its kernels, and report which backend it got. Idempotent.

    Must run BEFORE model.py is imported: `from kernel import ...` sits at its module scope, so the
    binding is resolved once and can never be swapped afterwards.

    'tilelang' does nothing at all -- the real kernel.py is a plain module next to model.py, so it
    resolves off sys.path (v4_ref_cpu's job) with no shim in the way."""
    be = backend()
    if be == "tilelang":
        return be
    clash = [n for n in _MODULES if _foreign(n)]
    if clash and V4_KERNELS != "cpu":
        raise RuntimeError(f"v4_kernels_cpu: {clash} already imported by something else; "
                           f"set V4_KERNELS=cpu to override deliberately")
    for name, attrs in _MODULES.items():
        mod = sys.modules.get(name)
        if mod is None or not getattr(mod, "_v4_cpu_backend", False):
            mod = types.ModuleType(name)
            mod._v4_cpu_backend = True
            sys.modules[name] = mod
        for attr, value in attrs.items():
            setattr(mod, attr, value)
    return be


__all__ = ["backend", "install", "act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm", "unpack_fp4",
           "sparse_attn", "hc_split_sinkhorn", "hadamard_transform"]
