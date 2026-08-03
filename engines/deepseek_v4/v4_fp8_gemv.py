"""The fp8 GEMV path at decode occupancy — the largest per-layer cost the MoE work left behind.

WHERE THE TIME IS (08-01 ring, 6x RTX 5090, lossless recipe, ~1.06 ms/layer):
    routed fp4 MoE     99.5 us  @ 969 GB/s (57% of peak)   -- grouped, tile-swept, DONE
    everything else   ~0.96 ms  -- and the bulk of it is fp8 GEMVs at single-GEMV occupancy
Per decode token every layer runs SEVEN single-row fp8 GEMMs through `model.linear` -> `fp8_gemm`:
attention's wq_a / wq_b / wkv / wo_b, the ratio-4 layers' indexer wq_b, and the shared expert's
w1 / w3 / w2 — ~107 MB of fp8 weights per ratio-4 layer, read at whatever bandwidth the vendored
kernel's grid can reach. That grid is `(ceildiv(N,128), ceildiv(M,32))` of 128 threads, so at M=1:

    wq_a  [1024,4096]    8 blocks        shared w1/w3 [2048,4096]   16 blocks each
    wkv   [ 512,4096]    4 blocks        shared w2    [4096,2048]   32 blocks
    wo_b  [4096,8192]   32 blocks        idx wq_b     [8192,1024]   64 blocks
    wq_b [32768,1024]  256 blocks  <- the one healthy shape

A 5090 has 170 SMs. Four to thirty-two blocks is a starved weight stream: the shared expert's three
launches measured ~70-160 GB/s (~4-9% of peak) on the 08-01 ring, ~0.18 ms/layer on their own. The
proof the gap closes is on the SAME card in the SAME file family: the grouped fp4 kernel is the
identical GEMV-shaped work at 192 blocks and runs at 969 GB/s. Occupancy is the whole disease.

TWO LEVERS, BOTH OPT-IN, BOTH DEFAULT OFF, BOTH SELF-GATING:

V4_FP8_GEMV — re-tile the vendored fp8 GEMM where it is GEMV-shaped. `install()` rebinds the
  reference's module-level `fp8_gemm` (the binding `model.linear` resolves at call time, eager and
  inside a whole-layer capture alike); the replacement hands every M > 32 call — prefill, big verify
  chunks — straight back to the vendored kernel and serves M <= 32 with `fp8_gemm_tiled_kernel`, a
  TRANSCRIPTION of `kernel.fp8_gemm_kernel` whose only free parameters are grid/pipeline shape:
  block_N, num_stages, threads. block_M=32 and block_K=128 are PINNED — block_K is the fp8 scale
  group, so the per-element fp32 accumulation chain (per-128-block tensor-core partial, times
  a-scale x b-scale, added in ascending K order, rounded once to bf16) is the vendored chain at any
  tile. `V4_FP8_GEMV=1` (or `auto`) picks the tile per shape — the narrowest block_N in
  {128,64,32,16} that reaches ~192 blocks, the block count the fp4 kernel measured 969 GB/s at; a
  shape already there (wq_b) is left on the vendored kernel untouched. `V4_FP8_GEMV=BN[,S[,T]]`
  forces one tile everywhere (the bench's sweep mode; research/v4_fp8_gemv_bench.py).

V4_FP8_SHARED — the shared expert's w1 and w3 as ONE launch. The shared expert has no routing: same
  three matrices, every token, every layer, and its w1/w3 read the SAME activation — the exact
  structure the routed path already fuses (`v4_moe_grouped._BANK_GROUPS`: "w1 and w3 share a GEMM").
  `shared_bank_layout()` re-lays them at Stage construction as one contiguous [2*inter, dim] fp8
  bank (+ one scale bank), REPOINTING each Linear's parameters at bank slices exactly as
  `_relayout_moe` does for the routed experts — zero net VRAM, and the reference path reads the
  same bytes through the views, so prefill / fallbacks are byte-identical by construction. The
  patched `Expert.forward` then claims a banked expert at M <= 32: ONE act_quant of x (the
  reference quantizes the same x once per matrix; act_quant is deterministic, so one call is the
  same bytes), ONE fp8 GEMM over the bank (through the V4_FP8_GEMV dispatch, so the fused launch is
  also occupancy-tiled when that lever is armed), a view split, and then the reference's OWN SwiGLU
  arithmetic and its OWN `self.w2(...)` call — w2 composes with whatever `fp8_gemm` is bound.
  THREE matrices in ONE launch is structurally impossible: w2 contracts the other way and consumes
  the SwiGLU of the first two. Two launches is the floor, the same floor the routed path ships.

ARGUED IS NOT PROVEN — THE GATE. Losslessness on this engine is absolute (a 2-ULP gather-order
change once cost 24% acceptance), so neither lever takes its own argument's word:
  * V4_FP8_GEMV: the first claimed GEMM at each (N, K, scale dtype) builds the tuned kernel and
    runs it against the CAPTURED vendored `fp8_gemm` on seeded full-range inputs (every non-NaN fp8
    byte pattern eligible, e8m0 scales drawn as exponents) at M = 1, 8 and 32, and serves the tile
    only if every output is `torch.equal`. A tilelang codegen that reassociates at some
    thread/tile combination — or fails to compile — DECLINES, loudly, and that shape serves the
    vendored kernel, byte-identical to the lever being off.
  * V4_FP8_SHARED: the first claimed forward probes THE SERVING COMPOSITION — the fused bank GEMM,
    through the live dispatch, tile and all — against two vendored per-matrix calls on the bank's
    own halves, `torch.equal` on both halves, else the expert serves the reference forward (which
    still rides tiled per-matrix GEMVs if V4_FP8_GEMV proved them). The concat argument being
    gated, not trusted: C[j] depends on A, W row j and W's row-j scale alone in this kernel, and
    both banks concatenate at a 128-row scale-group boundary (inter % 128 asserted at layout), so
    fusing changes only how many N-tiles the grid walks — the same per-element argument the routed
    w13 fusion shipped under, now with a runtime proof per box.
  Probes never run inside a CUDA graph capture (`is_current_stream_capturing` short-circuits to the
  vendored path); the whole-layer warmup runs every GEMM eagerly on a side stream first, so by
  capture time every verdict is in and the captured program is the proven one. Both claimed paths
  are capture-safe: static shapes, no host syncs, no allocation surprises beyond the vendored
  wrapper's own output alloc.

WHAT THIS DOES NOT CLAIM, so nobody over-reads it:
  * M > 32 — prefill and wide verify chunks stay on the vendored kernel (its grid is healthy there).
  * `wo_a` — the layer's largest single weight read (67 MB bf16) is a bare `torch.einsum` in the
    reference, not a `linear()`; batching it losslessly would mean matching cuBLAS's batched
    accumulation order bit for bit, which no transcription can promise. Measured, named, left.
  * The fp32 compressor GEMVs (`Compressor.wkv/wgate`, ~17-42 MB/layer) and the bf16
    `weights_proj` / gate — `F.linear` paths, not fp8_gemm's. Next in line, not this file.
  * act_quant's ~0.12 ms/layer of latency-bound launches — fusing it into the GEMM prologue is a
    numerics-bearing change and a separate, gated decision.
  * A drafter block's shared expert — RingDrafter builds its DSparkBlocks lazily, long after
    `Stage.__init__` ran the layout; those keep the reference path (their per-matrix GEMMs still
    ride V4_FP8_GEMV).

Registered in phase0/v4_levers.py (V4_FP8_GEMV / V4_FP8_SHARED), carried by v4_pipe.ENG_ENV.
Sweep + measured GB/s per (shape, tile): research/v4_fp8_gemv_bench.py.

self-test (needs a CUDA device):  python3 phase0/v4_fp8_gemv.py
"""
import os

import torch

# ── the two flags, parsed once, defensively ──────────────────────────────────────────────────────

_SHIPPED_TILE = (128, 4, 128)   # kernel.fp8_gemm_kernel's block_N / num_stages / threads
_BLOCK_M = 32                   # pinned: the A tile; also the widest M the tiled path claims
_BLOCK_K = 128                  # pinned: == the fp8 scale group; changing it would reassociate
_TARGET_BLOCKS = 192            # the grid the fp4 grouped kernel measured 969 GB/s at (07-31 bench)


def _parse_gemv(s):
    """V4_FP8_GEMV '' | '1'/'auto' | 'BN[,STAGES[,THREADS]]' -> (mode, why-invalid).

    mode is None (off), "auto" (per-shape tile), or a forced (bn, stages, threads). Bounds are the
    kernel's envelope: block_N must DIVIDE 128 — a wider or non-dividing tile would span two
    128-row weight-scale groups and the kernel's scalar Scale_B lookup would scale half the tile
    with the wrong group. A bad string is IGNORED loudly, never an exception: a knob must not be
    able to take a stage down."""
    if not s or s == "0":
        return None, None
    if s in ("1", "auto"):
        return "auto", None
    try:
        vals = [int(p) for p in s.split(",")]
    except ValueError:
        return None, f"not integers: {s!r}"
    if not 1 <= len(vals) <= 3:
        return None, f"want BN[,STAGES[,THREADS]], got {s!r}"
    bn = vals[0]
    stages = vals[1] if len(vals) > 1 else _SHIPPED_TILE[1]
    threads = vals[2] if len(vals) > 2 else _SHIPPED_TILE[2]
    if bn <= 0 or 128 % bn:
        return None, f"block_N {bn} must divide 128 (the fp8 weight-scale group)"
    if not 1 <= stages <= 6:
        return None, f"num_stages {stages} outside 1..6"
    if threads % 32 or not 32 <= threads <= 512:
        return None, f"threads {threads} must be a multiple of 32 in 32..512"
    return (bn, stages, threads), None


V4_FP8_GEMV = os.environ.get("V4_FP8_GEMV", "")
_GEMV_MODE, _GEMV_INVALID = _parse_gemv(V4_FP8_GEMV)
if _GEMV_INVALID:
    print(f"[v4] V4_FP8_GEMV={V4_FP8_GEMV!r} IGNORED — {_GEMV_INVALID}", flush=True)

V4_FP8_SHARED = os.environ.get("V4_FP8_SHARED", "0") not in ("", "0")

# Captured at install().
_REF_FP8_GEMM = None       # the vendored fp8_gemm as the process had it bound — fallback AND oracle
_REF_EXPERT_FORWARD = None  # the vendored Expert.forward — every unclaimed shape's path
_MOD = None                # the loaded dsv4_model module — scale_fmt / scale_dtype / act_quant

_KERNELS = {}              # (N, K, tl_dtype, tile) -> compiled kernel, memoized like the vendored
_GEMV_VERDICTS = {}        # (N, K, tl_dtype) -> tile tuple (serve) | False (declined) | "shipped"
_SHARED_VERDICTS = {}      # (2*inter, dim, tl_dtype) -> True (fuse) | False (declined)


def _tl_dtype(scale_dtype):
    return "float8_e8m0fnu" if scale_dtype == torch.float8_e8m0fnu else "float32"


# ── the kernel: kernel.fp8_gemm_kernel with the grid/pipeline shape as parameters ────────────────

def fp8_gemm_tiled_kernel(N, K, scale_dtype="float32", tile=None):
    """`kernel.fp8_gemm_kernel(N, K)` verbatim, with (block_N, num_stages, threads) free.

    Everything an output element computes is the vendored kernel's: the [block_M, 128] x
    [block_N, 128] tensor-core partial per K block, `scales_a[row, k] * scales_b[row_group, k]`
    applied in fp32, accumulated over K blocks IN ASCENDING ORDER into `C_local_accum`, one
    bf16 round at the store. block_N changes how many N-tiles the grid walks and nothing an
    element sees; num_stages changes how many K iterations the pipeline keeps in flight, not their
    order; threads changes the warp partition of the SAME per-element chains. That argument is
    enforced, not trusted — `_resolve_gemv` only hands a tile here after `_probe_gemv` measured it
    `torch.equal` against the vendored kernel at this exact (N, K).

    tilelang is imported HERE, not at module scope (it drags in a CUDA toolchain a CPU install must
    not pay for). Memoized by the full key, since the wrapper closure defeats @tilelang.jit's own
    cache."""
    key = (N, K, scale_dtype, tile)
    if key in _KERNELS:
        return _KERNELS[key]
    import tilelang
    import tilelang.language as T
    tilelang.set_log_level("WARNING")
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
    FP8, FP32, BF16 = "float8_e4m3", "float32", "bfloat16"
    out_dtype, accum_dtype = BF16, FP32
    group_size = 128
    block_M = _BLOCK_M
    block_N, num_stages, threads = tile if tile is not None else _SHIPPED_TILE
    block_K = _BLOCK_K
    # The kernel reads ONE weight scale per (128-row group, K block); a tile wider than a group
    # would scale its overhanging rows with the wrong group. The parser refuses such tiles; this
    # holds the line for any caller that reaches the builder directly.
    assert 128 % block_N == 0, f"block_N={block_N} must divide 128 (the fp8 weight-scale group)"

    M = T.symbolic("M")

    @tilelang.jit(pass_configs=pass_configs)
    def _build():

        @T.prim_func
        def fp8_gemm_tiled_kernel_(
            A: T.Tensor[(M, K), FP8],
            B: T.Tensor[(N, K), FP8],
            C: T.Tensor[(M, N), out_dtype],
            scales_a: T.Tensor[(M, T.ceildiv(K, group_size)), scale_dtype],
            scales_b: T.Tensor[(T.ceildiv(N, group_size), T.ceildiv(K, group_size)), scale_dtype],
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (
                bx,
                by,
            ):
                A_shared = T.alloc_shared((block_M, block_K), FP8)
                B_shared = T.alloc_shared((block_N, block_K), FP8)
                C_shared = T.alloc_shared((block_M, block_N), out_dtype)
                Scale_C_shared = T.alloc_shared((block_M), FP32)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

                T.use_swizzle(panel_size=10)
                T.clear(C_local)
                T.clear(C_local_accum)

                K_iters = T.ceildiv(K, block_K)
                for k in T.Pipelined(K_iters, num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    # One scale per (128-row group, K block): with block_N | 128 the whole tile sits
                    # inside one group, exactly as the vendored block_N=128 tile does.
                    Scale_B = T.Cast(FP32, scales_b[bx * block_N // group_size, k])
                    for i in T.Parallel(block_M):
                        Scale_C_shared[i] = T.Cast(FP32, scales_a[by * block_M + i, k]) * Scale_B

                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                    for i, j in T.Parallel(block_M, block_N):
                        C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]
                    T.clear(C_local)
                T.copy(C_local_accum, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return fp8_gemm_tiled_kernel_

    _KERNELS[key] = _build()
    return _KERNELS[key]


def _run_tiled(a, a_s, b, b_s, scale_dtype, tile):
    """`kernel.fp8_gemm`'s wrapper verbatim, dispatching to the tiled kernel."""
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), "Scaling factor tensors must be contiguous"
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    kernel = fp8_gemm_tiled_kernel(N, K, _tl_dtype(scale_dtype), tile)
    kernel(a.view(M, K), b, c.view(M, N), a_s.view(M, -1), b_s)
    return c


# ── tile choice and the losslessness gate ────────────────────────────────────────────────────────

def _auto_tile(N):
    """The narrowest re-tiling that reaches ~_TARGET_BLOCKS blocks, or None to keep the vendored.

    192 blocks is where the fp4 grouped kernel measured 969 GB/s on this card family; a shape whose
    shipped grid is already there (wq_b: 256 blocks) is left on the vendored kernel — the 07-31 tile
    sweep measured narrower tiles at ~1.0x THERE, so re-tiling a healthy shape buys nothing and
    costs a probe."""
    if N // 128 >= _TARGET_BLOCKS:
        return None
    for bn in (64, 32, 16):
        if N // bn >= _TARGET_BLOCKS:
            return (bn, _SHIPPED_TILE[1], _SHIPPED_TILE[2])
    return (16, _SHIPPED_TILE[1], _SHIPPED_TILE[2])


def _probe_seeded(N, K, tl_dtype, M, seed=0x84F8):
    """Seeded full-range inputs at (M, N, K): every non-NaN fp8 weight byte, e8m0 scales as
    exponents around 1.0, activations through a clamp so no NaN can fake a mismatch (NaN != NaN).

    Dtypes EXPLICIT everywhere (the serve path sets the process default to bf16, and a bare randn
    would hand the kernel bf16 scales where it declares fp32 — the probe would then decline every
    tile for a reason that has nothing to do with the tile)."""
    dev = "cuda"
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed + M)
    a = torch.randn(M, K, device=dev, dtype=torch.float32,
                    generator=gen).clamp_(-3, 3).to(torch.float8_e4m3fn)
    wb = torch.randint(0, 256, (N, K), device=dev, dtype=torch.uint8, generator=gen)
    wb[(wb & 0x7F) == 0x7F] = 0        # e4m3fn's NaN encodings (S.1111.111): full range MINUS NaN
    w = wb.view(torch.float8_e4m3fn)
    if tl_dtype == "float8_e8m0fnu":
        a_s = torch.randint(120, 135, (M, (K + 127) // 128), device=dev, dtype=torch.uint8,
                            generator=gen).view(torch.float8_e8m0fnu)
        w_s = torch.randint(120, 135, ((N + 127) // 128, (K + 127) // 128), device=dev,
                            dtype=torch.uint8, generator=gen).view(torch.float8_e8m0fnu)
    else:
        a_s = torch.rand(M, (K + 127) // 128, device=dev, dtype=torch.float32, generator=gen) + 0.5
        w_s = torch.rand((N + 127) // 128, (K + 127) // 128, device=dev, dtype=torch.float32,
                         generator=gen) + 0.5
    return a, a_s, w, w_s


def _probe_gemv(N, K, tl_dtype, tile):
    """Tuned kernel vs the captured vendored `fp8_gemm` at this exact (N, K): `torch.equal` at
    EVERY M the tiled path claims (1..32 — decode, the drafter's block, every verify-chunk width),
    or a decline with a reason. Exhaustive rather than sampled because it is nearly free — the JIT
    is per (N, K, tile) and 64 GEMV-sized calls are tens of microseconds — and it turns "argued
    M-invariant" into "measured at every claimable M on this box".

    Any failure — a miscompile, an OOM in the JIT, a genuine numeric divergence — is caught and
    returned as a decline; the caller then serves the vendored kernel, byte-identical to the lever
    being off."""
    sd = torch.float8_e8m0fnu if tl_dtype == "float8_e8m0fnu" else torch.float32
    try:
        for M in range(1, _BLOCK_M + 1):
            a, a_s, w, w_s = _probe_seeded(N, K, tl_dtype, M)
            c_ref = _REF_FP8_GEMM(a, a_s, w, w_s, sd)
            c_tuned = _run_tiled(a, a_s, w, w_s, sd, tile)
            torch.cuda.synchronize()
            if not torch.equal(c_ref, c_tuned):
                d = (c_ref.float() - c_tuned.float()).abs().max().item()
                return False, (f"tuned != vendored at M={M} (max|d| {d:.3e}) — the tile "
                               f"reassociates on this box")
        return True, None
    except Exception as e:                             # noqa: BLE001 — a knob must not kill a stage
        return False, f"{type(e).__name__}: {e}"


def _resolve_gemv(N, K, tl_dtype):
    """The tile this (N, K) may serve with: the tuned one iff its probe PASSED, else None.

    Judged once per (N, K, scale dtype) and memoized. Refuses to build or probe inside a CUDA graph
    capture — a JIT and a comparison do not belong in a captured program; the whole-layer warmup
    runs every GEMM eagerly first, so by capture time the verdict is already in."""
    if _GEMV_MODE is None:
        return None
    key = (N, K, tl_dtype)
    if key in _GEMV_VERDICTS:
        v = _GEMV_VERDICTS[key]
        return v if isinstance(v, tuple) else None
    # A capture can only be underway on an initialized context — and asking CUDA anything on a box
    # whose driver cannot answer must not take the dispatch down.
    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        return None                        # unjudged mid-capture: serve the vendored, decide later
    tile = _auto_tile(N) if _GEMV_MODE == "auto" else _GEMV_MODE
    if tile is None:
        _GEMV_VERDICTS[key] = "shipped"
        print(f"[v4] V4_FP8_GEMV at N={N} K={K}: shipped grid already >= {_TARGET_BLOCKS} blocks — "
              f"vendored kernel serves", flush=True)
        return None
    ok, why = _probe_gemv(N, K, tl_dtype, tile)
    _GEMV_VERDICTS[key] = tile if ok else False
    print(f"[v4] V4_FP8_GEMV={V4_FP8_GEMV} at N={N} K={K}: "
          + (f"PROVEN tile={tile} — torch.equal vs the vendored kernel at M=1/8/{_BLOCK_M}"
             if ok else f"DECLINED — {why}"), flush=True)
    return tile if ok else None


def _fp8_gemm_fast(a, a_s, b, b_s, scale_dtype=torch.float32):
    """The rebound `fp8_gemm`: decode-shaped calls on a proven tile, everything else vendored."""
    if a.is_cuda:
        K = a.size(-1)
        if a.numel() // K <= _BLOCK_M:
            tile = _resolve_gemv(b.size(0), K, _tl_dtype(scale_dtype))
            if tile is not None:
                return _run_tiled(a, a_s, b, b_s, scale_dtype, tile)
    return _REF_FP8_GEMM(a, a_s, b, b_s, scale_dtype)


_fp8_gemm_fast._v4_fp8_gemv = True


def gemv_status():
    """The lever audit's observation: off / invalid / armed / on/k-of-n / declined/n.

    'armed' is the honest pre-run state — the probe is lazy per (N, K), so before the first claimed
    GEMM nothing has been judged and nothing may read as OK (v4_levers' rule). A shape whose auto
    tile IS the shipped grid counts as on: leaving a healthy grid alone is the lever doing its job."""
    if not V4_FP8_GEMV or V4_FP8_GEMV == "0":
        return "off"
    if _GEMV_MODE is None:
        return f"invalid({_GEMV_INVALID})"
    if not _GEMV_VERDICTS:
        return "armed"
    ok = sum(1 for v in _GEMV_VERDICTS.values() if v is not False)
    n = len(_GEMV_VERDICTS)
    return f"on/{ok}-of-{n}" if ok else f"declined/{n}"


# ── the shared expert: w1 + w3 as one bank, one quant, one launch ────────────────────────────────

def _lay_shared(e):
    """Repoint one shared Expert's w1/w3 (weight + scale) at slices of one contiguous bank.

    `_relayout_moe`'s move at 1/96th the size: allocate [2*inter, dim] fp8 (+ the [2*inter/128,
    dim/128] scale bank), copy the current bytes in, and set `p.data = bank[rows]` — a row range of
    a contiguous 2-D tensor is itself contiguous, so both Linears keep exactly the tensor
    `linear()` reads and the reference path cannot tell. `Linear.__init__` aliases `weight.scale`
    to the SAME Parameter as `.scale`, so swapping `.scale.data` updates what `linear()` reads
    through `weight.scale` too. The transient is one bank (~17 MB at shipped dims) — nothing like
    the routed layout's 3.2 GiB problem, so no release-first choreography is needed.

    Copying rather than assuming garbage keeps this correct at BOTH call sites: `Stage.__init__`
    runs it pre-load (garbage in, checkpoint written through the views by `load_state_dict`), and a
    test running it post-fill keeps its weights.

    Declines, leaving the module untouched, unless the expert is exactly the shape the fused GEMM
    claims: fp8 weights, matching [inter, dim] pair, and inter % 128 == 0 — the fusion seam MUST
    land on a 128-row weight-scale-group boundary or w3's first rows would be scaled by w1's last
    scale group. dim % 128 for the scale bank's column count. Never lays twice."""
    if getattr(e, "_v4_w13", None) is not None:
        return False
    w1, w3 = getattr(e, "w1", None), getattr(e, "w3", None)
    if w1 is None or w3 is None or w1.weight.dtype != torch.float8_e4m3fn \
            or w3.weight.dtype != torch.float8_e4m3fn:
        return False
    inter, dim = w1.weight.shape
    if tuple(w3.weight.shape) != (inter, dim) or inter % 128 or dim % 128:
        return False
    dev = w1.weight.device
    with torch.no_grad():
        bank = torch.empty(2 * inter, dim, dtype=w1.weight.dtype, device=dev)
        sbank = torch.empty(2 * (inter // 128), dim // 128, dtype=w1.scale.dtype, device=dev)
        # byte copies, like _gather_fp: fp8/e8m0 are 1-byte dtypes with thin kernel coverage
        bank[:inter].view(torch.uint8).copy_(w1.weight.detach().view(torch.uint8))
        bank[inter:].view(torch.uint8).copy_(w3.weight.detach().view(torch.uint8))
        sbank[:inter // 128].view(torch.uint8).copy_(w1.scale.detach().view(torch.uint8))
        sbank[inter // 128:].view(torch.uint8).copy_(w3.scale.detach().view(torch.uint8))
        w1.weight.data = bank[:inter]
        w3.weight.data = bank[inter:]
        w1.scale.data = sbank[:inter // 128]
        w3.scale.data = sbank[inter // 128:]
    e._v4_w13 = (bank, sbank)
    return True


def shared_bank_layout(module):
    """Give every shared expert under `module` the w13 bank. Returns how many took it.

    The loader's entry point — `Stage.__init__` calls it right after the routed `bank_layout`,
    between construction and `load()`, so the checkpoint lands in the bank through the views. A
    no-op under V4_FP8_SHARED=0: nothing allocated, nothing repointed, the default path
    byte-identical. Duck-typed on `shared_experts` (the MoE attribute) so a test can pass a
    stand-in and a stage passes `stage.layers`."""
    if not V4_FP8_SHARED:
        return 0
    mods = module.modules() if hasattr(module, "modules") else [module]
    return sum(1 for m in mods
               if getattr(m, "shared_experts", None) is not None
               and _lay_shared(m.shared_experts))


def _probe_shared(N2, K, tl_dtype):
    """The SERVING composition — one GEMM over a [N2, K] bank, live dispatch, tile and all —
    against two vendored per-matrix calls on the bank's halves. `torch.equal` on both, or decline."""
    sd = torch.float8_e8m0fnu if tl_dtype == "float8_e8m0fnu" else torch.float32
    inter = N2 // 2
    try:
        for M in range(1, _BLOCK_M + 1):
            a, a_s, w13, s13 = _probe_seeded(N2, K, tl_dtype, M, seed=0x5A13)
            fused = _fp8_gemm_fast(a, a_s, w13, s13, sd)
            lo = _REF_FP8_GEMM(a, a_s, w13[:inter], s13[:inter // 128], sd)
            hi = _REF_FP8_GEMM(a, a_s, w13[inter:], s13[inter // 128:], sd)
            torch.cuda.synchronize()
            if not (torch.equal(fused[..., :inter], lo) and torch.equal(fused[..., inter:], hi)):
                d = (fused.float() - torch.cat([lo, hi], -1).float()).abs().max().item()
                return False, f"fused != per-matrix at M={M} (max|d| {d:.3e})"
        return True, None
    except Exception as e:                             # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _resolve_shared(N2, K, tl_dtype):
    """May a [N2, K] w13 bank serve as one launch? Judged once, memoized, never inside a capture."""
    key = (N2, K, tl_dtype)
    v = _SHARED_VERDICTS.get(key)
    if v is not None:
        return v
    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        return False                       # unjudged mid-capture: reference path, decide later
    ok, why = _probe_shared(N2, K, tl_dtype)
    _SHARED_VERDICTS[key] = ok
    print(f"[v4] V4_FP8_SHARED at [{N2},{K}]: "
          + ("PROVEN — one launch, torch.equal per half vs the vendored per-matrix calls"
             if ok else f"DECLINED — {why}"), flush=True)
    return ok


def _w13_gemm(x, w13, s13):
    """gate|up = x @ w13^T in one call: quantize x ONCE, one GEMM over the bank. None = declined.

    OFF CUDA this runs the two vendored per-matrix calls over the bank's halves and concatenates —
    not a re-derivation, the installed `fp8_gemm` itself, per matrix — which is exactly what the
    fused launch computes and is how a CPU box proves the DISPATCH (the rewiring, the bank views,
    the quant-once step) without a card and without trusting a CPU BLAS to be N-invariant. On CUDA
    the fused call serves only after `_resolve_shared`'s probe."""
    xq, xs = _MOD.act_quant(x, _MOD.block_size, _MOD.scale_fmt, _MOD.scale_dtype)
    sd = _MOD.scale_dtype
    inter = w13.size(0) // 2
    if not x.is_cuda:
        lo = _REF_FP8_GEMM(xq, xs, w13[:inter], s13[:inter // 128], sd)
        hi = _REF_FP8_GEMM(xq, xs, w13[inter:], s13[inter // 128:], sd)
        return torch.cat([lo, hi], dim=-1)
    if not _resolve_shared(w13.size(0), x.size(-1), _tl_dtype(sd)):
        return None
    return _fp8_gemm_fast(xq, xs, w13, s13, sd)


def _decline_shared(e, why, x, weights):
    """Hand this call to the reference forward and RECORD that we did (v4_moe_grouped._decline's
    shape: host-side counters only, read by tests and the audit, never a device op)."""
    tally = getattr(e, "_shared_declined", None)
    if tally is None:
        tally = e._shared_declined = {}
    tally[why] = tally.get(why, 0) + 1
    return _REF_EXPERT_FORWARD(e, x, weights)


def _shared_forward(self, x, weights=None):
    """Expert.forward with the banked w1/w3 as one launch; everything else is the reference.

    The claimed path is the reference's arithmetic on the same operands: one act_quant of x (the
    reference quantizes the SAME x once inside w1's linear() and once inside w3's — deterministic,
    so one call is the same bytes), the fused GEMM whose halves are gated `torch.equal` to the
    per-matrix calls, then the reference's own clamp/SwiGLU/weight lines verbatim, and the
    reference's own `self.w2(...)` — which rides whatever `fp8_gemm` is bound, so the two levers
    compose. Elementwise ops are batch- and slice-invariant; nothing here can move a bit that the
    gate did not already rule on."""
    bank = getattr(self, "_v4_w13", None)
    if bank is None:
        return _REF_EXPERT_FORWARD(self, x, weights)
    if x.numel() // x.size(-1) > _BLOCK_M:
        return _decline_shared(self, "m>32", x, weights)
    both = _w13_gemm(x, *bank)
    if both is None:
        return _decline_shared(self, "gate-declined", x, weights)
    dtype = x.dtype
    inter = bank[0].size(0) // 2
    gate = both[..., :inter].float()
    up = both[..., inter:].float()
    if self.swiglu_limit > 0:
        up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
        gate = torch.clamp(gate, max=self.swiglu_limit)
    h = torch.nn.functional.silu(gate) * up
    if weights is not None:
        h = weights * h
    self._shared_steps = getattr(self, "_shared_steps", 0) + 1
    return self.w2(h.to(dtype))


_shared_forward._v4_fp8_shared = True


def shared_status():
    """off / armed / on/k-of-n / declined/n, from the fused-launch verdicts. Bank coverage is the
    stage's (`Stage._shared_banked`); this is the kernel gate's half."""
    if not V4_FP8_SHARED:
        return "off"
    if not _SHARED_VERDICTS:
        return "armed"
    ok = sum(1 for v in _SHARED_VERDICTS.values() if v)
    n = len(_SHARED_VERDICTS)
    return f"on/{ok}-of-{n}" if ok else f"declined/{n}"


# ── install ──────────────────────────────────────────────────────────────────────────────────────

def install(mod):
    """Arm whichever of the two levers is requested. Returns (gemv_took, shared_took).

    Runs AFTER model.py is executed (v4_ref_cpu.load_ref's window), independent of the MoE.forward
    chain: this rebinds the module-level `fp8_gemm` (what `linear()` resolves at call time) and
    `Expert.forward` — never `MoE.forward`. Idempotent via the markers. The vendored `fp8_gemm` is
    captured BEFORE any rebind, because it is both the fallback and the probes' oracle.

    V4_FP8_GEMV is CUDA-gated like the grouped install (the kernel is tilelang; a CPU box would
    defer the JIT failure to a real layer). V4_FP8_SHARED installs on any device: its unclaimed and
    off-CUDA paths ARE the reference (per-matrix vendored calls), so a CPU parity box exercises the
    real bound path at zero numeric risk — which is what tests/test_v4_fp8_gemv.py leans on."""
    global _REF_FP8_GEMM, _REF_EXPERT_FORWARD, _MOD
    _MOD = mod
    if _REF_FP8_GEMM is None and not getattr(mod.fp8_gemm, "_v4_fp8_gemv", False):
        _REF_FP8_GEMM = mod.fp8_gemm
    took_gemv = False
    if _GEMV_MODE is not None and torch.cuda.is_available() \
            and not getattr(mod.fp8_gemm, "_v4_fp8_gemv", False):
        mod.fp8_gemm = _fp8_gemm_fast
        took_gemv = True
    took_shared = False
    if V4_FP8_SHARED and not getattr(mod.Expert.forward, "_v4_fp8_shared", False):
        _REF_EXPERT_FORWARD = mod.Expert.forward
        mod.Expert.forward = _shared_forward
        took_shared = True
    return took_gemv, took_shared


# ── smoke (CUDA): the gates prove themselves at the shipped shapes ───────────────────────────────

def _smoke():
    """JIT + probe every shipped fp8 GEMV shape, then the fused shared expert vs the reference."""
    assert torch.cuda.is_available(), "v4 fp8 gemv smoke needs a CUDA device"
    import v4_moe_grouped
    global _GEMV_MODE, V4_FP8_GEMV
    mod = v4_moe_grouped._load_model_module()
    if _GEMV_MODE is None:
        V4_FP8_GEMV, _GEMV_MODE = "auto", "auto"     # the smoke exists to exercise the lever
    globals()["V4_FP8_SHARED"] = True
    install(mod)
    shapes = (("attn.wq_a", 1024, 4096), ("attn.wkv", 512, 4096), ("attn.wq_b", 32768, 1024),
              ("attn.wo_b", 4096, 8192), ("idx.wq_b", 8192, 1024),
              ("shared.w13", 4096, 4096), ("shared.w2", 4096, 2048))
    for name, N, K in shapes:
        tile = _resolve_gemv(N, K, "float8_e8m0fnu")
        v = _GEMV_VERDICTS[(N, K, "float8_e8m0fnu")]
        print(f"{name:12s} N={N:6d} K={K:5d}  -> {'tile ' + str(tile) if tile else v}")
    print(f"gemv_status  {gemv_status()}")

    # the fused shared expert against the reference forward, real dims, both weight layouts
    args = v4_moe_grouped.real_dims_args(mod)
    torch.manual_seed(3)
    with mod.set_dtype(torch.bfloat16), torch.device("cuda"):
        ref_e = mod.Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)
        fus_e = mod.Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)
    with torch.no_grad():
        for lin, (out_f, in_f) in ((ref_e.w1, (args.moe_inter_dim, args.dim)),
                                   (ref_e.w3, (args.moe_inter_dim, args.dim)),
                                   (ref_e.w2, (args.dim, args.moe_inter_dim))):
            w, s = v4_moe_grouped._fp8_block_quant(
                torch.randn(out_f, in_f, dtype=torch.bfloat16, device="cuda") * 0.02)
            lin.weight.data.copy_(w)
            lin.scale.data.copy_(s)
        for k in ("w1", "w2", "w3"):
            getattr(fus_e, k).weight.data.copy_(getattr(ref_e, k).weight.detach())
            getattr(fus_e, k).scale.data.copy_(getattr(ref_e, k).scale.detach())
    assert _lay_shared(fus_e), "the w13 bank layout must take on a shipped-dims fp8 expert"
    for t in range(8):
        x = torch.randn(1, args.dim, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            ref = _REF_EXPERT_FORWARD(ref_e, x)
            got = _shared_forward(fus_e, x)
        eq = torch.equal(ref, got)
        print(f"shared draw {t}  equal={eq}  "
              f"max|d|={(ref.float() - got.float()).abs().max().item():.3e}")
        assert eq, "the fused shared expert is not bit-exact to the reference"
    print(f"shared_status  {shared_status()}   steps={fus_e._shared_steps}")
    print("OK")


if __name__ == "__main__":
    _smoke()
