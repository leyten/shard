"""DeepSeek-V4-Flash's sparse attention, retiled to fit an sm_120 shared-memory budget.

Five of the six vendored tilelang kernels JIT and run on a 5090 (sm_120) unchanged. The sixth does
not, and it is the one every attention layer goes through:

    RuntimeError: Failed to set the allowed dynamic shared memory size to 141312

`sparse_attn_kernel` (deepseek_v4_ref/inference/kernel.py) stages the WHOLE head dimension in shared
memory -- `q_shared [h, d]`, `o_shared [h, d]`, `acc_s_cast [h, block]` -- so at V4's real attention
shape (h = n_local_heads = 64, d = head_dim = 512, block = 64) one thread block asks for 138 KiB.
Hopper and datacenter Blackwell have 228 KiB of opt-in shared memory per block and it fits. Consumer
Blackwell does not: `shared_memory_per_block_optin` is 101376 B (99 KiB) on a 5090 AND on a Pro
6000, and the launch is refused before a warp runs. It is a hardware limit on the SM's shared-memory
partition -- no environment variable, no compile flag, and not a tilelang bug.

WHY A RETILE IS LEGAL, AND WHY IT IS THE ONLY CHANGE
Every head's online softmax in that kernel is INDEPENDENT. `q_shared` rows never interact across h:
the two `T.gemm`s are row-parallel over the head axis (`GemmWarpPolicy.FullRow`, C[h, block] and
C[h, d] both indexed by the same i), the running `scores_max`/`sum_exp`/`scores_scale` are per-head
fragments of length h, and the sink term `exp(attn_sink[i] - scores_max[i])` reads its own head's
row. So h is a pure parallel axis that the vendored kernel happens to keep inside one thread block.
Move it OUT -- grid `(m, b, ceil(h/h_block))` instead of `(m, b)` -- and every shared buffer that
carries an h shrinks by h/h_block, with no cross-block communication and no change to the arithmetic
any single head performs.

THE DELTA, LINE FOR LINE (everything else is DeepSeek's, transcribed):
  1. grid            `T.Kernel(m, b, threads=...)` -> `T.Kernel(m, b, n_h_blocks, threads=...)`,
                     n_h_blocks = ceil(h / h_block), new block index `bz`.
  2. h -> h_block    in `q_shared`, `o_shared`, `acc_s_cast`, `acc_s`, `acc_o`, the five per-head
                     fragments, and every `T.Parallel(h, ...)` extent.
  3. head offset     `q[by, bx, :, :]` -> `q[by, bx, bz*h_block:(bz+1)*h_block, :]`, the same on the
                     `o` store, and `attn_sink[i]` -> `attn_sink[bz*h_block + i]`.
  4. (h_block, block, threads) are arguments instead of the literals 64/64/256.
The KV gather, the -1 masking (`idxs[i] == -1` zeroes the KV row AND pushes the score to -inf, so a
padded slot contributes to neither numerator nor denominator), the online-softmax rescale order, the
bf16 round-trip of the probabilities through `acc_s_cast` before the PV gemm, and the sink landing
in the denominator AFTER the loop are byte-faithful. At h_block == h, block == 64, threads == 256
this generates the vendored kernel with a degenerate third grid dimension, which is what makes the
cross-check meaningful: on a 5090, at every shape where the vendored kernel still fits 99 KiB, an
unchunked run of this kernel is BIT-IDENTICAL to it (0 mismatches in 1.5M elements) and a chunked
one differs in 5 elements per 1.7M by one bf16 ulp -- T.gemm's fp32 accumulation order under a
different warp tiling, not the head split, which is exact by construction.

WHAT THE TILING COSTS AND BUYS (measured, RTX 5090, tilelang 0.1.12 / torch 2.10.0+cu128)
Each head-chunk block re-gathers the same KV rows, so gather traffic multiplies by ceil(h/h_block),
4x at the shipped tiling. In exchange `acc_o [h_block, d]` fp32 drops from 128 KiB of registers per
block to 32 KiB, so blocks-per-SM goes up. At the real decode shape (b=1, s=1, h=64, d=512,
topk=640) it lands at 49.7 us/call; the alternatives are 54.1 us at threads=128 and 62.3 us at
block=32, and (32,64) does not fit at all. Prefill s=33 is 51.3 us, i.e. the kernel is launch- and
gather-bound at these sizes, not FLOP-bound. Full sweep:
docs/receipts/v4-sm120-sparse-attn-20260801.json.

TWO tilelang CONSTRAINTS A RING BRING-UP MUST KNOW (both are compile-time asserts, not smem):
  h_block % 16 == 0        `M must be divisible by 16` -- the m16n8k16 MMA atom. h_block=8 and
                           h_block=4 are NOT available, so the "make the tile as small as you like"
                           intuition is wrong; 16 is the floor and the head padding has to reach it.
  block % (threads/4) == 0 `warp_col_tiles must be divisible by 8, got N`. FullRow degenerates to
                           column-splitting once M is one MMA atom, so each of the threads/32 warps
                           takes block/(threads/32) columns and that has to be a multiple of 8.
                           block=32 therefore needs threads<=128, block=64 allows threads=256.

WHAT THIS MODULE IS NOT: a second implementation of V4 attention. `sparse_attn_eager` is not a new
transcription -- it delegates to `v4_kernels_cpu.sparse_attn`, the CPU backend's stand-in, on
whatever device the tensors already live on. One transcription used by both backends is the point:
an A/B against a SECOND hand-written reference measures the two transcriptions against each other,
not the kernel.

self-test (needs a CUDA device):  python3 phase0/v4_sparse_attn_sm120.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4_kernels_cpu

# (h_block, block, threads), best measured first -- ordered by decode throughput at V4's real shape
# on a 5090, NOT by shared-memory frugality, which would pick the small tile and lose 20%. Both
# entries satisfy the two tilelang constraints in the module docstring; a third that did not would
# fail at JIT, on the ring, at layer 0. See `choose_tile`: first entry that fits wins.
TILES = ((16, 64, 256), (16, 32, 128))
# Force a tiling for an experiment: V4_SM120_TILE=16,32,128 (h_block,block,threads). Unvalidated.
TILE_ENV = "V4_SM120_TILE"
# The vendored kernel's own tiling, which is what a device with room for it keeps running.
VENDORED_BLOCK, VENDORED_THREADS, NUM_STAGES = 64, 256, 2
# The MMA atom's M. h_block below this does not compile, and the wrapper pads heads up to it.
M_ATOM = 16


def kernel_smem(h_block, block, d, threads=VENDORED_THREADS):
    """Dynamic shared bytes one thread block of this tiling asks the driver for.

    Not an estimate -- this reproduces every launch-failure number tilelang has printed on the box:
    141312 for the vendored (64, 64, t256), 153600 for (16, 128, t256), 104448 for (32, 64, t256),
    103424 for (32, 64, t128). The layout is confirmed in the generated CUDA: `o_shared` is aliased
    ONTO `q_shared` at offset 0 (it is dead until the loop ends), `kv_shared` follows, then
    `acc_s_cast`; the 8 B/thread on top is tilelang's cross-warp reduction workspace."""
    return 2 * (h_block * d + block * d + h_block * block) + 8 * threads


def smem_limit(device=None):
    """This device's opt-in dynamic shared memory per block, i.e. the number the launch checks.

    `shared_memory_per_block` (48 KiB everywhere since Volta) is the DEFAULT, not the ceiling; a
    kernel that calls cudaFuncSetAttribute -- which every tilelang kernel does -- gets up to
    `shared_memory_per_block_optin`. That is 227 KiB on H100/B200 and 99 KiB on sm_120."""
    p = torch.cuda.get_device_properties(0 if device is None else device)
    return getattr(p, "shared_memory_per_block_optin", None) or p.shared_memory_per_block


def padded_heads(h, h_block=M_ATOM):
    """The head count the kernel is actually built for: h rounded up to a multiple of h_block.

    kernel.py's wrapper does this at 16 and calls it "pad heads to kernel efficiency"; it is not an
    efficiency, it is the MMA atom, and a partial chunk would read q/attn_sink past their end."""
    return -(-h // h_block) * h_block


def choose_tile(h, d, limit=None):
    """(h_block, block, threads) for this shape: the vendored tiling if it fits, else TILES.

    Returning the whole head dimension when it fits is deliberate -- on a big-shared-memory device
    this file then generates DeepSeek's own tiling and there is nothing to explain away. The
    vendored kernel is theirs and stays theirs; nothing here is an improvement on it."""
    if env := os.environ.get(TILE_ENV):
        return tuple(int(v) for v in env.split(","))
    limit = smem_limit() if limit is None else limit
    full = padded_heads(h)
    if kernel_smem(full, VENDORED_BLOCK, d, VENDORED_THREADS) <= limit:
        return full, VENDORED_BLOCK, VENDORED_THREADS
    for h_block, block, threads in TILES:
        if kernel_smem(h_block, block, d, threads) <= limit:
            return h_block, block, threads
    raise RuntimeError(
        f"v4 sm120: no tiling for h={h} d={d} fits {limit} B of shared memory (smallest candidate "
        f"{TILES[-1]} needs {kernel_smem(*TILES[-1][:2], d, TILES[-1][2])} B)")


# ── the kernel ───────────────────────────────────────────────────────────────────────────────────

_KERNELS = {}


def sparse_attn_kernel(h: int, d: int, scale=None, h_block: int = M_ATOM,
                       block: int = VENDORED_BLOCK, threads: int = VENDORED_THREADS,
                       num_stages: int = NUM_STAGES):
    """Sparse multi-head attention via index gathering + online softmax, one head CHUNK per block.

    DeepSeek's `sparse_attn_kernel` with the head dimension lifted into the grid; see the module
    docstring for the four-line delta. tilelang is imported HERE rather than at module scope: it
    drags in a CUDA toolchain, and `install_sm120`'s no-op path on a CPU box must neither pay for
    that nor fail on it. Memoized by argument tuple, the way `@tilelang.jit` memoizes the vendored
    kernel -- the decorator's own cache cannot help, since `_build` is a fresh closure each call."""
    key = (h, d, scale, h_block, block, threads, num_stages)
    if key in _KERNELS:
        return _KERNELS[key]
    assert h % h_block == 0, f"v4 sm120: h={h} not a multiple of h_block={h_block} (padded_heads)"
    assert h_block % M_ATOM == 0, f"v4 sm120: h_block={h_block} is not a multiple of {M_ATOM} (MMA)"
    import tilelang
    import tilelang.language as T
    tilelang.set_log_level("WARNING")
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
    BF16, FP32, INT32 = "bfloat16", "float32", "int32"

    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")
    topk = T.symbolic("topk")
    if scale is None:
        scale = (1.0 / d) ** 0.5

    num_blocks = tilelang.cdiv(topk, block)
    n_h_blocks = h // h_block

    @tilelang.jit(pass_configs=pass_configs)
    def _build():

        @T.prim_func
        def sparse_attn_sm120_kernel_(
            q: T.Tensor[(b, m, h, d), BF16],
            kv: T.Tensor[(b, n, d), BF16],
            o: T.Tensor[(b, m, h, d), BF16],
            attn_sink: T.Tensor[(h,), FP32],
            topk_idxs: T.Tensor[(b, m, topk), INT32],
        ):
            with T.Kernel(m, b, n_h_blocks, threads=threads) as (bx, by, bz):
                q_shared = T.alloc_shared((h_block, d), BF16)
                kv_shared = T.alloc_shared((block, d), BF16)
                o_shared = T.alloc_shared((h_block, d), BF16)
                acc_s_cast = T.alloc_shared((h_block, block), BF16)

                idxs = T.alloc_fragment(block, INT32)
                acc_s = T.alloc_fragment((h_block, block), FP32)
                acc_o = T.alloc_fragment((h_block, d), FP32)
                scores_max = T.alloc_fragment(h_block, FP32)
                scores_max_prev = T.alloc_fragment(h_block, FP32)
                scores_scale = T.alloc_fragment(h_block, FP32)
                scores_sum = T.alloc_fragment(h_block, FP32)
                sum_exp = T.alloc_fragment(h_block, FP32)

                T.clear(acc_o)
                T.clear(sum_exp)
                T.fill(scores_max, -T.infinity(FP32))
                T.copy(q[by, bx, bz * h_block:(bz + 1) * h_block, :], q_shared)

                for t in T.Pipelined(num_blocks, num_stages=num_stages):
                    for i in T.Parallel(block):
                        idxs[i] = T.if_then_else(t * block + i < topk, topk_idxs[by, bx, t * block + i], -1)
                    for i, j in T.Parallel(block, d):
                        kv_shared[i, j] = T.if_then_else(idxs[i] != -1, kv[by, idxs[i], j], 0)
                    for i, j in T.Parallel(h_block, block):
                        acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))
                    T.gemm(q_shared, kv_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(h_block, block):
                        acc_s[i, j] *= scale
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(h_block):
                        scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                    for i, j in T.Parallel(h_block, block):
                        acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(h_block):
                        sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(h_block, d):
                        acc_o[i, j] *= scores_scale[i]
                    T.gemm(acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i in T.Parallel(h_block):
                    sum_exp[i] += T.exp(attn_sink[bz * h_block + i] - scores_max[i])
                for i, j in T.Parallel(h_block, d):
                    acc_o[i, j] /= sum_exp[i]
                T.copy(acc_o, o_shared)
                T.copy(o_shared, o[by, bx, bz * h_block:(bz + 1) * h_block, :])

        return sparse_attn_sm120_kernel_

    _KERNELS[key] = _build()
    return _KERNELS[key]


def sparse_attn(
    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """kernel.py's `sparse_attn` wrapper, on the retiled kernel. Same signature, same contract.

    The vendored wrapper pads heads to 16 and strips the padding back off the output; this one pads
    to a multiple of `h_block`, which for h < 16 at any shipped tiling is exactly that same 16. The
    padding is ZEROS, not garbage, for the same reason it is there: a zero q row scores 0 against
    every gathered KV row and a zero sink keeps the denominator finite, so a padded head costs
    arithmetic and cannot produce a NaN that survives the strip."""
    b, s, h, d = q.size()
    h_block, block, threads = choose_tile(h, d)
    pad = padded_heads(h, h_block) - h
    if pad:
        q = torch.cat([q, q.new_zeros(b, s, pad, d)], dim=2)
        attn_sink = torch.cat([attn_sink, attn_sink.new_zeros(pad)])
    o = torch.empty_like(q)
    kernel = sparse_attn_kernel(q.size(2), d, softmax_scale, h_block, block, threads)
    kernel(q, kv, o, attn_sink, topk_idxs)
    if pad:
        o = o.narrow(2, 0, h).contiguous()
    return o


def sparse_attn_eager(
    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """The same math in plain torch, on whatever device the tensors are on -- the A/B's other side.

    Deliberately `v4_kernels_cpu.sparse_attn` itself rather than a fresh transcription: that
    function is already the audited stand-in every CPU parity test runs against, it takes its device
    from its inputs, and one transcription cannot disagree with itself. It differs from the kernel
    in two documented places -- it keeps the probabilities in fp32 where the kernel rounds them to
    bf16 for the PV gemm, and it returns zeros for an all-masked query row where the kernel returns
    NaN -- neither of which the reference's own index construction can reach (every query position
    keeps at least its own window slot)."""
    return v4_kernels_cpu.sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)


# ── the hook ─────────────────────────────────────────────────────────────────────────────────────

def install_sm120(h=64, d=512):
    """Rebind `kernel.sparse_attn` to the retiled one IF this device cannot run the vendored one.

    ORDER IS LOAD-BEARING, exactly as it is for `v4_kernels_cpu.install`: model.py resolves
    `from kernel import ... sparse_attn ...` at its module scope, so this has to run BEFORE model.py
    is executed or the reference keeps the binding it already made. `v4_ref_cpu.load_ref()` calls it
    in the one window where that is true -- after the inference dir is on sys.path, before the exec.

    Three ways it does nothing, all of them the right nothing:
      no CUDA          the CPU backend is in force and `kernel` is v4_kernels_cpu's shim anyway.
      device fits      H100/B200-class, 227 KiB opt-in. DeepSeek's kernel stays DeepSeek's.
      cpu shim loaded  V4_KERNELS=cpu on a GPU box; the shim's sparse_attn is the one under test.

    (h, d) default to V4's real attention shape, because that is what the decision is about. The toy
    configs the test suite builds are small enough to fit anywhere, so deciding on THEM would leave
    the 141312 B launch to fail later, on the ring, at layer 0 of a loaded 158 GiB model."""
    if not torch.cuda.is_available():
        return None
    limit = smem_limit()
    if kernel_smem(padded_heads(h), VENDORED_BLOCK, d, VENDORED_THREADS) <= limit:
        return None
    import kernel                                    # the real kernel.py, next to model.py
    if getattr(kernel, "_v4_cpu_backend", False):
        return None
    kernel.sparse_attn = sparse_attn
    return choose_tile(h, d, limit)


def _smoke():
    """JIT + run at V4's real attention shape, decode and prefill, against the eager reference."""
    assert torch.cuda.is_available(), "v4 sm120 smoke needs a CUDA device"
    torch.manual_seed(0)
    dev, h, d, n, topk = "cuda", 64, 512, 640, 640
    tile = choose_tile(h, d)
    print(f"smem limit {smem_limit()}  vendored {kernel_smem(h, VENDORED_BLOCK, d)}  "
          f"tile {tile} {kernel_smem(tile[0], tile[1], d, tile[2])}")
    for s in (1, 33):
        q = torch.randn(1, s, h, d, dtype=torch.bfloat16, device=dev)
        kv = torch.randn(1, n, d, dtype=torch.bfloat16, device=dev)
        sink = torch.randn(h, dtype=torch.float32, device=dev)
        idx = torch.randint(0, n, (1, s, topk), dtype=torch.int32, device=dev)
        idx[..., ::7] = -1
        o = sparse_attn(q, kv, sink, idx, d ** -0.5)
        ref = sparse_attn_eager(q, kv, sink, idx, d ** -0.5)
        assert torch.isfinite(o.float()).all(), f"non-finite output at s={s}"
        print(f"s={s:<3d} out {tuple(o.shape)}  max|kernel-eager| "
              f"{(o.float() - ref.float()).abs().max().item():.3e}")


if __name__ == "__main__":
    _smoke()
