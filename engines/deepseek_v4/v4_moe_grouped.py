"""One grouped fp4 launch for a decode step's routed experts — the MoE's kernel-launch tax, removed.

MEASURED (RTX 5090, real weights, layer 7 of DeepSeek-V4-Flash, one token, from the 7-agent perf
pass that motivated this file):
    whole layer   3.77 ms      of which  MoE 2.39 ms,  attention + hyper-connections 1.41 ms
The MoE's arithmetic is microseconds — six routed fp4 experts plus the shared one, each a handful of
tiny GEMVs on a single token. The 2.39 ms is almost entirely the CPU getting there: the reference
runs one Python loop iteration per active expert, and each expert is ~20 device launches
(act_quant + fp4_gemm for w1, w3, w2, plus the SwiGLU elementwise chain), so a layer spends ~120
launches to move ~140 µs of memory. `v4_moe_decode` already deleted the reference's per-expert host
syncs (`bincount().tolist()` + a `torch.where` per expert); what it did NOT delete is the launch
count — it still walks the six experts one at a time, each its own stack of kernels.

This deletes the launch count too, and the last host sync with it. At b*s == 1 on a score-routed
layer the six routed expert ids are a six-element device tensor, and every GEMV they name has the
SAME left operand (the one token). So:

    w13,w2 = gather(w*_bank, ids)                  # device gather of the six routed experts, no sync
    xq, xs = act_quant(x)                          # ONE quant of the token, shared by all experts
    both   = grouped_fp4_gemm(xq, w13)             # ONE launch: all six w1 AND all six w3
    gate6, up6 = both[:, :inter], both[:, inter:]  # a view split, not a copy
    h6     = weight * (silu(clamp(gate6)) * clamp(up6))   # batched SwiGLU, torch, all six at once
    hq, hs = act_quant(h6)                         # ONE quant of the six intermediates
    out6   = grouped_fp4_gemm(hq, w2)              # ONE launch: all six w2
    y      = sum(out6 in ascending-expert-id order) + shared_expert(x)

W1 AND W3 SHARE ONE LAUNCH because they are the same GEMM — both are dim -> inter_dim against the
SAME quantized token — so the bank stores them as one [E, 2*inter_dim, dim] block (w1's rows then
w3's) and one grouped call fills both halves. That is not a micro-optimisation on this card: the
grouped kernel's grid is `(ceildiv(N, 128), G)`, i.e. 16 x 6 = 96 blocks at N = inter_dim, against
an RTX 5090's 170 SMs — it does not fill the machine, so DOUBLING N is very nearly free while
halving the launches. The w2 group cannot join them: it contracts inter_dim -> dim (a different K)
and its input is the SwiGLU of the first group's output, so it is necessarily a second launch after
an activation. Two launches per layer is the floor this structure allows, and this hits it.

Two routed-GEMM launches instead of ~120, and — crucially — ZERO `.tolist()`. The slot->expert map
never touches the host: the six ids gather the routed experts out of the bank with a device index
(`_gather_fp`), the kernel is then a plain grid-indexed batched GEMM over that gathered [G, N, K]
bank, and the ascending-id accumulation order is a device `argsort` gather plus a fixed six-add loop,
not a host-side `sorted()`. That removes the data-dependent `.tolist()` that is the first of the
three things blocking a CUDA-graph capture of the decode layer.

(The tidier design — pass the whole 256-expert bank and dereference `W[eids[slot]]` INSIDE the kernel
— does not survive this sm_120 tilelang build: a per-block data-dependent leading index into a
packed-fp4 bank mis-addresses, uniform eids work but distinct eids collapse every slot onto one
weight. The torch gather is the working equivalent; it is a handful of extra device launches, still
no host sync, still CUDA-graph-capturable, and with w1/w3 fused it is 2 GEMMs + 4 gathers against
the reference's ~120.)

BIT-EXACT AT s == 1, BY CONSTRUCTION, NOT BY TOLERANCE — the bar is `torch.equal` against the
reference MoE, and this reaches it the same way `v4_sparse_attn_sm120` reaches it against its
vendored kernel: the arithmetic is transcribed, not re-derived.
  * The grouped GEMM is `kernel.fp4_gemm_kernel` with a batch (expert-slot) axis added to the grid,
    the gathered weight/scale indexed by grid position. block_M=32, block_N=128, block_K=32,
    threads=128, the FP4->FP8 cast, the per-32 weight scale x per-128 act scale, and the fp32
    accumulate-then-round-to-bf16 are all IDENTICAL to the vendored kernel. Each grid block computes
    one expert's output row exactly as the vendored kernel computes it for a single-token GEMM
    (the full row-0-aligned A tile, only row g stored) — a per-output-element dot product that
    depends on nothing but that block's A row and B tile, so the block_K=32 = fp4 weight-scale group
    alignment means no scale reassociation and the result is the same fp32 word.
  * FUSING w1 AND w3 INTO ONE LAUNCH CANNOT MOVE A BIT, for that same reason: `C[g, j]` depends on
    the A row and on W row j alone, never on N or on which N-tile j fell in. Concatenating the two
    weight blocks only changes how many N-tiles the grid walks, so `both[:, :inter]` is the fp32
    word `grouped_fp4_gemm(xq, w1)` would have produced and `both[:, inter:]` is w3's. (Nothing here
    needs inter_dim to be a multiple of block_N — a tile straddling the w1/w3 seam still computes
    each of its output elements from its own W row. The KERNEL does need the fused N to be a whole
    number of block_N tiles, because its store loop is unpredicated; the fusion RELAXES that from
    inter_dim % 128 to inter_dim % 64, and `grouped_fp4_gemm` asserts it either way.)
  * The SwiGLU, the routing-weight multiply, the `.to(dtype)` and the two `act_quant`s stay in the
    reference's own torch / tilelang code, run on the SAME operands, batched over the expert axis.
    Elementwise and per-row reductions are batch-invariant, so row g of the batch is bit-identical to
    the reference's single-expert call. (This is why the SwiGLU is NOT fused into the kernel: silu's
    transcendental would have to match torch's last ULP, and `torch.equal` does not forgive a ULP.
    Keeping it in torch makes the equality hold, not merely hold closely.)
  * The accumulation opens at fp32 zero and adds the six bf16 expert rows in ASCENDING expert id,
    then the shared expert — exactly `v4_moe_decode`'s order, which is exactly the reference's loop
    order. fp32 add is commutative but not associative, so the order is load-bearing; the device
    `argsort` reproduces it without a host sync.

FALLS BACK, UNTOUCHED, for everything it does not claim — the same envelope as `v4_moe_decode`, for
the same reasons:
  s > 1 (prefill / a speculation chunk)   grouped-and-padded MoE is NOT invariant to token count
                                          (~2e-3 at B>=4, the documented M2.5 grouped-MoE finding:
                                          padding a group reassociates its reduction). The decode
                                          fast path is the ONLY place the equality holds, so it is
                                          the only place this runs. An s>1 grouped path is a separate,
                                          quality-gated decision, not this file.
  world_size > 1                          the reference all-reduces the routed sum across ranks
                                          before the shared expert; skipping a rank's experts without
                                          that reduction silently drops them.

HASH-ROUTED LAYERS (layer_id < n_hash_layers) ARE CLAIMED TOO, which they were not before. They were
excluded because `tid2eid` can name the same expert twice and the reference's duplicate-index
`y[idx] +=` is not a repeated add — and on a stage that owns the head that exclusion is expensive:
layers 0, 1 and 2 of a 43-layer model are a HALF of a 6-layer head stage, and they were paying the
reference's ~120 launches while their score-routed neighbours paid 8.

What the reference actually does with a repeat is worth spelling out, because it is not addition.
`y[idx] += v` with `idx = [0, 0]` reads row 0 TWICE, adds the two expert outputs to those two copies,
and scatters both back to row 0 — where the LAST write wins. So a duplicated expert contributes
exactly ONE of its slots, the one with the largest slot index, and every earlier duplicate is
DISCARDED. (Verified against torch, not inferred: `y[[0,0]] += [[1],[10]]` from zero leaves 10.)

That is a pure function of the six ids, so it reproduces on device with no host sync:

    keep[k] = not any(ids[k'] == ids[k] for k' > k)      # a [G, G] compare, 36 elements
    out6    = where(keep[:, None], out6, 0)              # the discarded slots are NOT READ

and a discarded slot then adds exactly zero into the fp32 accumulator, which is exact. A SELECT and
not a multiply by the mask, because the reference never evaluates those slots at all: `inf * 0.0` is
NaN, and a routed output that overflowed bf16 would poison the token rather than be skipped. The surviving
slots carry distinct expert ids, so the ascending-id fold is unchanged and the order still matches
the reference's `for i in range(...)`. The one thing this leans on that a CPU emulator does not give
for free is that the vendored fp4 GEMM is ROW-INVARIANT — the reference runs the duplicated expert as
a 2-row GEMM and we run it as two 1-row grid blocks — which is true of the tilelang kernel (each
block computes each output element from its own A row) and NOT true of `torch.matmul`, whose BLAS
blocking depends on M. See
`tests/test_v4_moe_grouped.py::test_a_repeated_hash_expert_is_dropped_exactly_as_the_reference_drops_it`.

WEIGHT BANK — THE LOAD-TIME LAYOUT, WHICH IS WHAT LETS THIS RUN ON A FULL STAGE. The per-step gather
slices the routed experts out of ONE contiguous [n_experts, N, K] fp4 bank (+ its scale bank), not
the reference's per-expert `nn.Parameter`s (which cannot be gathered in a single op). Building that
bank by STACKING the per-expert tensors is a second copy of the layer's weights — ~3.2 GiB per layer
at the shipped dims beside the ~3.4 GiB already resident — and a stage holding seven or eight layers
on a 32 GiB card cannot afford it. That is why the lever measured 3.19x on a one-layer bench and then
never fired in production: `_expert_bank` would check free VRAM and decline.

`bank_layout()` removes the copy instead of budgeting for it. At LOAD time, before a single weight is
read off disk, it RELEASES the layer's per-expert tensors, hands the segments back, and allocates the
per-layer bank into the memory they just vacated, repointing each expert `Linear`'s parameter at its
slice (`p.data = bank[j]`, a zero-copy view — the slice of a contiguous [E, N, K] bank is itself
contiguous). `Stage.load`'s `load_state_dict` writes THROUGH those views into the bank, so the
checkpoint lands in the bank and nowhere else: ONE copy of the weights on the card, the same bytes
the non-grouped path would hold, and the bank costs nothing on top — not in `memory_allocated` and
not in `memory_reserved`, which is the harder half and the one the first version of this got wrong.
See `_relayout_moe` for why the release has to come FIRST.

Keeping the per-expert `Linear`s as views (rather than deleting them) is also what keeps the s > 1
fallback alive. The reference `MoE.forward` — which prefill, a verify chunk, a hash-routed layer and
any world_size > 1 rank all still take — reaches the weights through `self.experts[i].w1.weight`.
Those are the same objects, the same dtypes, the same contiguous bytes, addressing bank memory; the
reference path runs UNCHANGED and stays bit-exact by construction rather than by a re-derived s > 1
grouped kernel (which would not be bit-exact anyway: grouped-and-padded MoE is not token-count
invariant). One layout, both paths, one copy.

The lazy stack survives as the fallback for an MoE the loader never banked (the GPU parity harness's
standalone layer, a drafter block), and there it still measures free VRAM and DECLINES rather than
OOMing mid-first-token and taking the stage, and the ring, with it.

MEASURED ON A REAL STAGE (RTX 5090, 31.36 GiB usable, the converted 43-layer checkpoint):
    stage[40:43), real weights          allocated       on the card      layers with a bank
      V4_MOE_GROUPED=0                  10.168 GiB      10.793 GiB       0/3
      bank layout                       10.168 GiB      10.830 GiB       3/3
      stacking the bank instead         19.730 GiB      20.355 GiB       3/3
    stage[0:7), shipped dims -- what a 43-layer ring actually asks a 5090 to hold
      V4_MOE_GROUPED=0                  23.641 GiB      24.270 GiB       0/7
      bank layout                       23.641 GiB      24.381 GiB       7/7
      stacking the bank instead         26.829 GiB      27.457 GiB       1/7, SIX DECLINED
Those are END-STATE numbers on the PRE-FIX code, and end state was never the binding constraint.

ARITHMETIC, NOT MEASUREMENT -- everything from here to the speed table is computed from
config.json (dim 4096, moe_inter_dim 2048, 256 fp4 experts) by building the real Blocks on the meta
device, and NOTHING below has been run on a card yet. Per layer the routed experts are 3.1875 GiB in
1536 blocks, held as FOUR banks since w1 and w3 fused (w13 2048.00 MiB + its scale 128.00 MiB, w2
1024.00 MiB + its scale 64.00 MiB); the layers run
3.352-3.402 GiB; eight of them plus the head embedding is 27.98 GiB resident (26.998 + 0.986).
Building a layer's banks BEFORE releasing what they replace peaks at steady + 3200 MiB (the stranded
run at the last request, the worst of them); releasing first peaks at steady. Against a 30.76 GiB
budget (31.36 usable less the ~0.6 GiB CUDA context) that is the whole difference:
    head stage    steady     OLD peak      NEW peak
      [0:7)       24.62      27.74  fits    24.62     <- why seven layers measured clean
      [0:8)       27.98      31.11  OOM     27.98     <- and why eight did not
Eight is the ceiling for a head or middle stage: nine layers is 30.40 GiB of weights alone. It is NOT
the ceiling for a dspark tail -- `RingDrafter`'s three DSparkBlocks are three more 256-expert MoEs
(10.28 GiB), built lazily on the first dspark job, i.e. after load; that stage tops out around four
layers and is a separate problem this fix does not touch. Nor does 8 leave room to prefill at
V4_MAX_SEQ=8192 (a ratio-4 layer's Indexer scores are ~5.5 GiB at s=8192 against 2.78 GiB free).
`tests/test_v4_moe_grouped.py` gates the ordering at the real expert count with a storage-liveness
meter (`test_bank_layout_never_outruns_what_it_released`).
And the speed it was all for -- stage[40:43), real weights, median of 40 decode steps:
      eager        26.902 -> 18.599 ms/step   (8.967 -> 6.200 ms/layer, 1.45x)
      V4_CUDA_GRAPH=1  23.627 -> 15.387 ms/step   (7.876 -> 5.129 ms/layer, 1.54x)
      MoE.forward alone at the decode shape      4.75 -> 2.02 ms (2.35x)
      41 decoded steps, `torch.equal` against the same run with the lever off, both graph settings.
The 2.35x is the honest multi-layer number; the one-layer bench's 3.19x was measured with nothing
else competing for the card's launch queue.

Opt-in, default OFF: `V4_MOE_GROUPED=1` rebinds `MoE.forward` after `v4_moe_decode` has (it captures
whatever forward it finds and falls back to it, so the two compose); with the env unset this module
installs nothing and the decode path is byte-identical to today. tilelang is CUDA-only, so on a CPU
box `install()` is a no-op and the kernel is never JIT'd.

self-test (needs a CUDA device):  python3 phase0/v4_moe_grouped.py
"""
import os

import torch

V4_MOE_GROUPED = os.environ.get("V4_MOE_GROUPED", "0") not in ("", "0")

# Captured at install (see install()).
_REF_FORWARD = None    # the MoE.forward present when we installed — our fallback for unclaimed shapes
_MOD = None            # the loaded dsv4_model module — source of scale_fmt / scale_dtype / act_quant
_WORLD_SIZE = 1

_KERNELS = {}          # (N, K, tl_scale_dtype) -> compiled grouped kernel, memoized like the vendored

# The kernel's tile dims, hoisted because the WRAPPER has to enforce them. The vendored fp4 GEMM
# stores its tile with a bounds-checked `T.copy`; the grouped one stores row g with a raw
# `for j in T.Parallel(block_N)` loop, which is UNPREDICATED — so N must be a whole number of
# block_N tiles or the last block writes past the row. And the A tile is block_M rows, so a launch
# can carry at most that many expert slots. Both hold at the shipped dims and at every call this
# module makes; they are asserted rather than commented because the failure is a silent memory
# stomp, not an exception.
_BLOCK_M, _BLOCK_N = 32, 128


# ── the kernel ───────────────────────────────────────────────────────────────────────────────────

def grouped_fp4_gemm_kernel(N, K, scale_dtype="float32"):
    """`kernel.fp4_gemm_kernel(N, K)` with a batch (expert-slot) axis on the grid.

    C[g, :] = A_fp8[g, :] @ W_fp4[g, :, :]^T, for g in 0..G-1, one grid block per (N-tile, g). The
    routed experts are GATHERED into a contiguous [G, N, K] bank by the caller — the kernel indexes
    that bank by its GRID position g, the way the vendored kernel indexes A by its block index `by`.

    Why the gather rather than a device-side `W[eids[g]]` dereference (the tidier design this file
    first tried): on this sm_120 tilelang build a per-block DATA-DEPENDENT leading index into a
    packed-fp4 bank does not address correctly — a uniform eid across the grid works, but distinct
    eids collapse every slot onto one weight (verified: `[7,7]` bit-exact, `[7,3]` wrong). Moving the
    slot->expert map into a torch gather (`_gather_fp`, device-side, no host sync) sidesteps it and
    leaves the kernel a plain grid-indexed batched GEMM.

    Everything a single output element sees is the vendored kernel's: block_M=32, block_N=128,
    block_K=32 (= the fp4 weight-scale group, so no scale reassociation), threads=128, the FP4->FP8
    cast through fp32, the per-32 weight scale times per-128 act scale, and the fp32-accumulate /
    bf16-store. Each block copies the full row-0-aligned A tile (aligned like the vendored's
    `A[by*block_M]` at by=0 — an unaligned `A[g]` start mis-tiles), multiplies EVERY A row by this
    block's gathered weight W[g], and stores only row g, so C[g] = A[g] @ W[g] — bit-identical to the
    vendored single-token GEMM (proven at G=1). Requires G <= block_M (32); decode routes 6.

    tilelang is imported HERE, not at module scope: it drags in a CUDA toolchain that a CPU install
    must not pay for. Memoized by (N, K, scale_dtype) exactly as `kernel.fp4_gemm_kernel` is by its
    `@tilelang.jit`, since the wrapper closure defeats that decorator's own cache."""
    key = (N, K, scale_dtype)
    if key in _KERNELS:
        return _KERNELS[key]
    import tilelang
    import tilelang.language as T
    tilelang.set_log_level("WARNING")
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
    FP8, FP4, FP32, BF16 = "float8_e4m3", "float4_e2m1fn", "float32", "bfloat16"
    out_dtype, accum_dtype = BF16, FP32
    act_group_size = 128
    weight_group_size = 32
    block_M = _BLOCK_M
    block_N = _BLOCK_N
    block_K = 32                       # == weight_group_size: one scale per K-block, no reassociation
    n_sub = act_group_size // block_K  # 4 K-blocks per act-scale group

    G = T.symbolic("G")                # number of routed slots this launch fills (6 at decode)

    @tilelang.jit(pass_configs=pass_configs)
    def _build():

        @T.prim_func
        def grouped_fp4_gemm_kernel_(
            A: T.Tensor[(G, K), FP8],
            W: T.Tensor[(G, N, K), FP4],
            C: T.Tensor[(G, N), out_dtype],
            scales_a: T.Tensor[(G, T.ceildiv(K, act_group_size)), scale_dtype],
            scales_w: T.Tensor[(G, N, T.ceildiv(K, weight_group_size)), scale_dtype],
        ):
            with T.Kernel(T.ceildiv(N, block_N), G, threads=128) as (bx, g):
                A_shared = T.alloc_shared((block_M, block_K), FP8)
                B_fp4_shared = T.alloc_shared((block_N, block_K), FP4)
                B_shared = T.alloc_shared((block_N, block_K), FP8)
                C_shared = T.alloc_shared((block_M, block_N), out_dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)
                scale_a_frag = T.alloc_fragment((block_M,), FP32)
                scale_b_frag = T.alloc_fragment((block_N,), FP32)

                T.use_swizzle(panel_size=10)
                T.clear(C_local)
                T.clear(C_local_accum)

                K_iters = T.ceildiv(K, block_K)
                for k in T.Pipelined(K_iters, num_stages=2):
                    # Copy the FULL row-0-aligned A tile (the ≤block_M activation rows, bounds-padded)
                    # exactly like the vendored kernel's `A[by*block_M, ...]` at by=0 — NOT `A[g,...]`,
                    # whose unaligned row start T.copy mis-tiles. This block multiplies every A row by
                    # THIS block's gathered weight W[g]; only row g is stored, so the other rows are
                    # wasted work, but the access stays aligned and bit-exact to the vendored GEMM.
                    T.copy(A[0, k * block_K], A_shared)
                    T.copy(W[g, bx * block_N, k * block_K], B_fp4_shared)
                    # FP4->FP8 through fp32 to dodge the ambiguous C++ overload (as the vendored does).
                    for i, j in T.Parallel(block_N, block_K):
                        B_shared[i, j] = T.Cast(FP8, T.Cast(FP32, B_fp4_shared[i, j]))
                    for i in T.Parallel(block_N):
                        scale_b_frag[i] = T.Cast(FP32, scales_w[g, bx * block_N + i, k])
                    for i in T.Parallel(block_M):
                        scale_a_frag[i] = T.Cast(FP32, scales_a[i, k // n_sub])
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                    for i, j in T.Parallel(block_M, block_N):
                        C_local_accum[i, j] += C_local[i, j] * scale_a_frag[i] * scale_b_frag[j]
                    T.clear(C_local)

                T.copy(C_local_accum, C_shared)
                # Store row g — this expert's output for its own activation row — into C[g, this tile].
                for j in T.Parallel(block_N):
                    C[g, bx * block_N + j] = C_shared[g, j]

        return grouped_fp4_gemm_kernel_

    _KERNELS[key] = _build()
    return _KERNELS[key]


def grouped_fp4_gemm(a, a_s, w, w_s, scale_dtype=torch.float32):
    """C[g,:] = a_fp8[g,:] @ w_fp4[g,:,:]^T, per-128 act scale x per-32 e8m0 weight scale.

    a  [G, K] fp8, a_s [G, K//128];  w [G, N, K//2] float4_e2m1fn_x2 (logical [G, N, K]) ALREADY
    gathered to the routed experts, w_s [G, N, K//32] e8m0.  Mirrors `kernel.fp4_gemm`'s wrapper
    (contiguity asserts, fp32-vs-e8m0 scale dtype, output in the process default dtype).

    OFF CUDA it runs the same function as a per-slot loop over the installed `kernel.fp4_gemm` — one
    G-element `for`, not a re-derivation — which is exactly what the grid-indexed kernel computes and
    is how a CPU box proves the DISPATCH (routing, the ascending fold, the hash keep-mask, which bank
    rows reach which expert) without a card. It does NOT prove the tilelang kernel's numerics; those
    are argued in the module docstring and gated by the `@pytest.mark.hardware` parity tests. The
    import is local so the CPU stand-ins (`v4_kernels_cpu.install`) can be swapped in per test."""
    assert a.is_contiguous() and w.is_contiguous(), "grouped fp4: a and w must be contiguous"
    assert a_s.is_contiguous() and w_s.is_contiguous(), "grouped fp4: scales must be contiguous"
    assert w.size(1) % _BLOCK_N == 0, (
        f"grouped fp4: N={w.size(1)} is not a whole number of {_BLOCK_N}-wide tiles — the kernel's "
        f"store loop is unpredicated and the last block would write past the row")
    assert a.size(0) <= _BLOCK_M, (
        f"grouped fp4: {a.size(0)} expert slots exceeds the block_M={_BLOCK_M} A tile the kernel "
        f"copies; only rows below it can be stored")
    if not a.is_cuda:
        import kernel
        return torch.cat([kernel.fp4_gemm(a[g:g + 1], a_s[g:g + 1], w[g], w_s[g], scale_dtype)
                          for g in range(a.size(0))])
    tl_dtype = "float8_e8m0fnu" if scale_dtype == torch.float8_e8m0fnu else "float32"
    G, K = a.shape
    N = w.size(1)
    c = a.new_empty(G, N, dtype=torch.get_default_dtype())
    kernel = grouped_fp4_gemm_kernel(N, K, tl_dtype)
    kernel(a, w, c, a_s, w_s)
    return c


# ── the weight bank ──────────────────────────────────────────────────────────────────────────────

def _gather_fp(t, ids):
    """Advanced-index the routed experts out of a packed-fp4 / e8m0 bank along dim 0.

    CUDA torch does not implement `index_cuda` for float4_e2m1fn_x2 (or e8m0), so reinterpret to
    uint8 (both are 1-byte dtypes, so the view is free and shape-preserving), gather with the device
    id tensor (no host sync), and reinterpret back. `ids.long()` is the index dtype torch wants; the
    cast is a device op, still sync-free."""
    return t.view(torch.uint8)[ids.long()].view(t.dtype)


# ── the load-time bank layout: the experts ARE the bank ──────────────────────────────────────────

# (bank key, the expert Linears whose rows that bank concatenates, in bank row order). w1 and w3 share
# a bank because they share a GEMM: same K (dim), same left operand, so one [E, 2*inter, dim] block
# feeds one grouped launch and the halves split back out as views. w2 contracts the other way
# (inter -> dim) and consumes the SwiGLU of the first, so it is necessarily its own bank and its own
# launch. Each entry yields two banks, `key` for the weights and `key + "_s"` for the e8m0 scales.
_BANK_GROUPS = (("w13", ("w1", "w3")), ("w2", ("w2",)))
# Every routed-expert Linear, in the reference's own attribute names — the keys the lazy stack and
# the tests walk.
_EXPERT_KINDS = tuple(k for _key, kinds in _BANK_GROUPS for k in kinds)


def _relayout_moe(moe, preserve=True):
    """Make one MoE's routed experts VIEWS of a contiguous per-layer bank, in place. Returns True if
    it took.

    This is the load-time layout choice that makes the grouped kernel affordable: instead of stacking
    a SECOND copy of the experts on the first decode token (`_expert_bank`), the bank IS where the
    experts live. Per weight kind it allocates the [E, N, K] bank and repoints that expert's
    parameter at its slice:

        p.data = bank[j, r0:r1]  # zero-copy: a row range of slot j of a contiguous bank is contiguous

    which drops the last reference to the tensor the constructor allocated. Net VRAM is one copy of
    the weights — exactly what the non-grouped path holds — and the bank adds nothing.

    The row range is what lets w1 and w3 SHARE a bank (`_BANK_GROUPS`) without either of them losing
    the property the reference path depends on: `bank[j, 0:inter]` and `bank[j, inter:2*inter]` are
    each contiguous, each keep their own `.scale` attribute, and each are exactly the tensor
    `linear()` would have read. One bank, one grouped launch, two Linears that cannot tell.

    NEVER ASK THE DRIVER FOR A BYTE YOU HAVE NOT ALREADY HANDED BACK. That is the whole content of
    `preserve=False`, and it is what the first version of this function got wrong. Freeing the
    per-expert tensors AS the copy walks (bank allocated first, originals released one at a time
    behind it) keeps `memory_allocated` exactly flat — there really is only ever one copy — but it
    means every bank is a request for memory the layer is still holding. At the shipped dims that is
    +1024 MiB of live bytes per weight kind, and, worse, the released blocks are the WRONG SHAPE to
    satisfy it: 256 scattered 4 MiB per-expert blocks, sharing 20 MiB large-pool segments with the
    kinds not yet relaid, cannot be coalesced into the one 1 GiB request that replaces them, and
    no release — automatic or explicit — can hand back a segment that is not wholly free. So
    `memory_reserved` — what the driver and the next `cudaMalloc` actually see — climbs by the banks
    built so far while the bytes they replaced stay stranded, and only falls at the END of the layer,
    once the last weight kind frees the run. Peak reserved is therefore steady + 3200 MiB at the
    shipped dims (the stranded run at the w2 request, the worst of the six), not the +1.07 GiB the
    allocated high-water mark suggests, and an 8-layer stage dies at load on a 32 GiB card. The
    reported OOM reads back as that, to within a few MiB on each term: 27.97 GiB allocated (the
    arithmetic says 27.98 for an 8-layer head stage — and it is FLAT, because the release IS real),
    2.13 GiB reserved but unallocated (the freed originals of the four kinds already relaid, which by
    symmetry is the same 1024 + 64 + 1024 + 64 = 2176 MiB their banks cost), 669 MiB free, and a
    1024.00 MiB request — the w2 weight bank, the fifth of the six banks that layout used — unmeetable.
    (It is four banks now that w1 and w3 share one; the ordering argument is unchanged, and the
    numbers above are the six-bank run that produced the OOM.)

    `preserve=False` inverts the order and the excursion disappears: release EVERY routed-expert
    block of the layer first (rebind each parameter to a void tensor), `empty_cache`, and only THEN
    allocate the banks — into the 3.19 GiB the layer just vacated. Peak allocated and peak
    reserved both equal the steady state; nothing is ever resident twice, in the model or in the
    allocator.

    Two things that ordering does NOT rest on, stated so nobody re-derives them wrongly:
      * `empty_cache` is not what rescues the allocation. The caching allocator already runs
        `release_available_cached_blocks` and then `release_cached_blocks` and retries `cudaMalloc`
        before it reports OOM — which is why the old code died with 2.13 GiB still stranded AFTER
        that automatic release, and why the first version of this docstring calling the per-kind
        `empty_cache` "not optional" was wrong. The ORDERING is the fix. The call stays because it
        makes the return eager and deterministic, so `memory_reserved` reflects the model between
        layers rather than only when something is about to fail.
      * The pools do not net out exactly. A layer releases 3072 MiB of 4 MiB weight blocks into the
        LARGE pool and 192 MiB of 256 KiB scale blocks into the SMALL pool (the 1 MiB threshold), but
        every bank — including the 128 MiB and 64 MiB scale banks — is a large-pool request. So 192 MiB
        per layer has to make the round trip through the driver, which is exactly what the
        `empty_cache` above is for: the layer's scale blocks are freed together and are contiguous
        within the small pool (the weights went elsewhere), so their segments are wholly free and do
        return. If that ever stopped holding it would cost 192 MiB per layer, not a whole bank.

    What it costs is the tensors' CONTENTS, which is why it is not the default and why the caller
    has to ask for it. It is exactly right at `Stage.__init__`, the only place it is used: the Blocks
    were built two statements earlier out of `torch.empty`, so every routed-expert byte is
    uninitialised garbage, and `Stage.load` then `load_state_dict(strict=True)`s every one of them
    THROUGH the views into the bank. Nothing readable is discarded because nothing readable exists
    yet. (A stage that is constructed and never loaded serves garbage either way — the same garbage
    the constructor would have left.)

    `preserve=True` keeps the copy, for the caller that has already written the weights (the GPU
    parity harness's standalone layer, `build_real_dims_moe(bank=True)`). Its transient is one whole
    bank, which since the w1/w3 fusion is 32/51 of the layer rather than 16/51 — 2 GiB at the shipped
    dims. One layer on a card with room can afford that; a full stage is what `preserve=False` exists
    for. The copy goes through `.view(torch.uint8)` for the same reason `_gather_fp` does: fp4 and
    e8m0 are 1-byte dtypes with thin kernel coverage, and a byte copy is exactly as correct and
    always present.

    Declines, leaving the module untouched, when: the experts are not fp4 (the grouped kernel is
    fp4-only, and a bf16 CPU parity model must stay byte-identical), or something already put a bank
    on this module (never lay out twice, and never over the lazy stack). Both checks run before
    anything is released, so a decline is total. A `preserve=False` run that RAISES partway is not,
    though: the parameters it already voided stay voided and there is nothing to restore them from.
    That is a stage that must not serve, and does not — `Stage.__init__` propagates, the strict
    `load_state_dict` would reject the 0-element shapes anyway, and the only thing that can raise
    here is the OOM this ordering exists to prevent."""
    if getattr(moe, "_grouped_bank", None):
        return False
    lo, hi = moe.experts_start_idx, moe.experts_end_idx
    experts = [moe.experts[i] for i in range(lo, hi)]
    if not experts or experts[0].w1.weight.dtype != torch.float4_e2m1fn_x2:
        return False
    # One entry per bank the layer will own: the parameters that become its row ranges, where each
    # one starts, and the shape / dtype / device to allocate it with. A bank concatenates its group's
    # Linears along dim 0 per expert, so `rows` carries the split point (w13 = w1's rows then w3's).
    # Built up front so `preserve=False` can release every one of them before the first bank is asked.
    plan = []
    for key, kinds in _BANK_GROUPS:
        for attr, suffix in (("weight", ""), ("scale", "_s")):
            first = [getattr(getattr(experts[0], k), attr) for k in kinds]
            rows = [t.shape[0] for t in first]
            t0 = first[0]
            params = [[getattr(getattr(e, k), attr) for k in kinds] for e in experts]
            shape = (len(experts), sum(rows)) + tuple(t0.shape[1:])
            plan.append((key + suffix, params, rows, shape, t0.dtype, t0.device))
    cuda = experts[0].w1.weight.is_cuda
    if not preserve:
        # Hand the whole layer's per-expert run back FIRST. One void tensor per (dtype, device) is
        # shared by every parameter of that kind -- these are placeholders for the width of one
        # `empty_cache`, and each is replaced by its bank slice below.
        void = {}
        for _key, params, _rows, _shape, dtype, device in plan:
            v = void.setdefault((dtype, device), torch.empty(0, dtype=dtype, device=device))
            for group in params:
                for p in group:
                    p.data = v
        if cuda:
            torch.cuda.empty_cache()   # the run is wholly free now, so the segments actually return
    bank = {}
    with torch.no_grad():
        for key, params, rows, shape, dtype, device in plan:
            b = torch.empty(shape, dtype=dtype, device=device)
            bu = b.view(torch.uint8)
            for j, group in enumerate(params):
                off = 0
                for n, p in zip(rows, group):
                    if preserve:
                        bu[j, off:off + n].copy_(p.detach().view(torch.uint8))
                    p.data = b[j, off:off + n]   # the parameter now ADDRESSES the bank; its old
                    off += n                     # block frees
            bank[key] = b
            if preserve and cuda:      # hand the freed per-expert blocks back before the next bank
                torch.cuda.empty_cache()
    moe._grouped_bank = bank
    return True


def bank_layout(module, preserve=True):
    """Give every fp4 routed-expert MoE under `module` the bank layout. Returns how many took it.

    The loader's entry point — `v4_stage.Stage.__init__` calls it on the stage's Blocks right after
    they are constructed and before `Stage.load` fills them, with `preserve=False`, which is what
    keeps the layout's peak equal to its steady state on a full stage (see `_relayout_moe`). A no-op
    under `V4_MOE_GROUPED=0`, which is what keeps the default path byte-identical: nothing is
    allocated, nothing is repointed, and every expert keeps the tensor its constructor gave it.

    Every fp4 MoE gets it, including the hash-routed layers the grouped kernel will never claim. The
    layout is free (one copy either way), it is the same memory shape on every layer, and a stage that
    owns layers 0-7 owns three hash layers — sparing them would buy nothing and would make the stage's
    memory profile depend on which layers it happens to hold.

    Duck-typed on `experts` + `experts_start_idx` rather than an `isinstance(m, mod.MoE)` so this
    module does not have to have been `install()`ed (and so a test can pass a stand-in)."""
    if not V4_MOE_GROUPED:
        return 0
    return sum(1 for m in module.modules()
               if hasattr(m, "experts") and hasattr(m, "experts_start_idx")
               and _relayout_moe(m, preserve))


# VRAM the LAZY bank build must leave FREE on the card after it copies. That build (`_expert_bank`,
# the fallback for an MoE `bank_layout` never reached) is an exact DUPLICATE of the layer's
# routed-expert weights, and it happens on the first decode token, AFTER the graph pools are pinned
# and the KV cache is allocated — the worst possible moment to discover the card is full. 2 GiB covers
# the per-step gathers, activations, and the compressed-KV region still growing under a long job.
_BANK_HEADROOM_BYTES = 2 << 30


def _bank_fits(experts):
    """Would stacking these experts leave `_BANK_HEADROOM_BYTES` free? (True off-CUDA: no bound to check.)"""
    if not torch.cuda.is_available():
        return True
    need = sum(t.numel() * t.element_size()
               for e in experts for k in _EXPERT_KINDS
               for t in (getattr(e, k).weight, getattr(e, k).scale))
    free, _total = torch.cuda.mem_get_info()
    return free - need >= _BANK_HEADROOM_BYTES


def _expert_bank(moe):
    """This MoE's [E, N, K] fp4 banks (+ e8m0 scale banks), stacking them if nobody laid them out.

    The normal case on a stage is that `bank_layout` already ran at load and `_grouped_bank` is
    sitting on the module — the experts ARE the bank, there is nothing to build, and this returns on
    its first line. Every decode step goes through here, so that is the path that matters.

    THE STACK BELOW IS THE FALLBACK, for an MoE the loader never reached: the GPU parity harness's
    standalone layer, or a module built outside `Stage`. It COPIES, and on the real model that is a
    second copy of the layer's experts — at the shipped dims (dim 4096, moe_inter_dim 2048, 256 fp4
    routed experts) ~3.2 GiB per layer against ~3.4 GiB already resident. One layer on a 32 GiB card
    can afford that; a seven-layer stage cannot, which is exactly what `bank_layout` exists to fix.

    So when it cannot afford it, it DECLINES rather than OOMs: returning None sends the caller to the
    reference forward for good on this layer. That is the difference between a lever that quietly does
    not engage and a stage process that dies mid-first-token and cascades the whole ring — nothing
    upstream catches an OOM out of `ffn` (v4_stage's _BlockGraphs guards only the graph capture)."""
    bank = getattr(moe, "_grouped_bank", None)
    if bank is not None:
        return bank if bank is not False else None
    lo, hi = moe.experts_start_idx, moe.experts_end_idx
    experts = [moe.experts[i] for i in range(lo, hi)]
    if not _bank_fits(experts):
        moe._grouped_bank = False                          # decided ONCE: never retry, never thrash
        print(f"[v4] grouped MoE declined on layer {getattr(moe, 'layer_id', '?')} — stacking the "
              f"expert bank would not fit beside the per-expert weights; this layer stays on the "
              f"decode path. A stage gets the bank from bank_layout() at load and never reaches "
              f"this; an MoE built outside Stage does.", flush=True)
        return None
    # Same grouping the load-time layout builds, stacked instead of aliased: w1's rows then w3's, so
    # the fast path reads one bank per grouped launch either way. Through `.view(torch.uint8)` for the
    # same reason `_gather_fp` is: fp4 and e8m0 are 1-byte dtypes with thin cat/stack kernel coverage.
    bank = {}
    for key, kinds in _BANK_GROUPS:
        for attr, suffix in (("weight", ""), ("scale", "_s")):
            dtype = getattr(getattr(experts[0], kinds[0]), attr).dtype
            stacked = torch.stack([torch.cat([getattr(getattr(e, k), attr).view(torch.uint8)
                                              for k in kinds]) for e in experts])
            bank[key + suffix] = stacked.contiguous().view(dtype)
    moe._grouped_bank = bank
    return bank


# ── the forward ──────────────────────────────────────────────────────────────────────────────────

def _decline(moe, why, x, input_ids):
    """Hand this step to the captured forward and RECORD that we did, per layer, per reason.

    A lever that silently does nothing is the failure mode this engine keeps paying for: the profile
    that motivated the hash-layer work read `fp4_gemm_kernel 72 calls` and `grouped_fp4_gemm_kernel
    6 calls` on a 6-layer stage, which is FOUR layers that never grouped at all — invisible in the
    kernel table, and only recoverable by dividing by 18. `coverage()` turns that into a number the
    profiler and `tests/test_v4_moe_grouped.py::test_every_layer_of_a_stage_is_grouped` can read
    directly. Host-side counters only: no device work, nothing to sync, nothing in a graph."""
    tally = getattr(moe, "_grouped_declined", None)
    if tally is None:
        tally = moe._grouped_declined = {}
    tally[why] = tally.get(why, 0) + 1
    return _REF_FORWARD(moe, x, input_ids)


def _keep_last_of_each(ids):
    """Which routed slots survive the reference's duplicate-index `y[idx] +=`. See the module doc.

    `y[idx] += v` with a repeated index scatters every duplicate to the same row and the LAST write
    wins, so an expert named by several slots contributes only its highest slot and the rest are
    discarded outright. keep[k] is False exactly when some later slot names the same expert. Pure
    device arithmetic on a [G, G] compare — G is 6, so this is 36 elements and no host sync."""
    same = ids[:, None] == ids[None, :]
    return ~same.triu(1).any(dim=1)


def grouped_forward(self, x, input_ids):
    """MoE.forward for a single-token, single-rank step — two grouped launches, no host sync.
    Every other shape falls through to the captured reference forward. See module doc."""
    shape = x.size()
    xv = x.view(-1, self.dim)
    if xv.size(0) != 1:
        return _decline(self, "s>1", x, input_ids)
    if _WORLD_SIZE > 1:
        return _decline(self, "world_size>1", x, input_ids)

    weights, indices = self.gate(xv, input_ids.flatten())
    ids = indices[0].to(torch.int32)                       # [G] on device — no .tolist()
    bank = _expert_bank(self)
    if bank is None:                                       # the bank would not fit — decline, loudly
        return _decline(self, "bank-would-not-fit", x, input_ids)

    # Gather the six routed experts into contiguous [G, N, K] banks (device-side, no host sync — see
    # `_gather_fp` on why a torch gather rather than a device-side kernel index). One gather per
    # bank — w1 and w3 share theirs — then the kernel is a plain grid-indexed batched GEMM.
    w13, w13_s = _gather_fp(bank["w13"], ids), _gather_fp(bank["w13_s"], ids)
    w2, w2_s = _gather_fp(bank["w2"], ids), _gather_fp(bank["w2_s"], ids)

    scale_fmt, scale_dtype = _MOD.scale_fmt, _MOD.scale_dtype
    block = _MOD.block_size
    act_quant = _MOD.act_quant

    # w1 / w3: one act_quant of the token, broadcast to the G expert slots, ONE grouped GEMM over the
    # fused [G, 2*inter, dim] bank. The token is quantized ONCE and its G identical rows are what
    # per-expert `w1(x)` would each quantize, so row g stays bit-identical to the reference's
    # expert-g call; the N-split back into gate/up is a view, and cannot move a bit (module doc).
    G = ids.numel()
    xq1, xs1 = act_quant(xv, block, scale_fmt, scale_dtype)
    xq = xq1.expand(G, -1).contiguous()
    xs = xs1.expand(G, -1).contiguous()
    both = grouped_fp4_gemm(xq, xs, w13, w13_s, scale_dtype)
    inter = both.size(1) // 2
    gate6, up6 = both[:, :inter], both[:, inter:]

    # SwiGLU-with-clamp + routing weight + cast, batched over experts — the reference's own torch ops
    # (Expert.forward), one row per expert, so bit-identical per row.
    g = gate6.float()
    u = up6.float()
    if self.experts[self.experts_start_idx].swiglu_limit > 0:
        lim = self.experts[self.experts_start_idx].swiglu_limit
        u = torch.clamp(u, min=-lim, max=lim)
        g = torch.clamp(g, max=lim)
    h = torch.nn.functional.silu(g) * u
    h = weights[0, :, None] * h                            # weights[0, k] is expert-slot k's weight
    h = h.to(xv.dtype)

    # w2: one act_quant of the six intermediates, one grouped GEMM (gathered bank, as above).
    hq, hs = act_quant(h, block, scale_fmt, scale_dtype)
    out6 = grouped_fp4_gemm(hq, hs, w2, w2_s, scale_dtype)

    # A hash gate can name the same expert twice, and the reference DISCARDS all but that expert's
    # last slot (module doc). Zeroing the discarded rows reproduces it exactly: they then add +0.0
    # into the fp32 accumulator, which is not an approximation of skipping them, it IS skipping them.
    # `where` and NOT a multiply by the mask: the reference never evaluates the discarded slot at all,
    # so whatever it would have contained must not reach the sum -- and `inf * 0.0` is NaN, which
    # would poison the token instead of dropping the slot. A select cannot, whatever the row holds.
    # Top-k routing cannot repeat, so a score-routed layer skips the compare entirely.
    if self.gate.hash:
        keep = _keep_last_of_each(ids)[:, None]
        out6 = torch.where(keep, out6, torch.zeros_like(out6))

    # Accumulate in ascending expert id — the reference's loop order — via a device argsort gather and
    # a fixed six-add fold. fp32 add is not associative, so the order is what keeps this bit-exact.
    # `stable=True` so a repeated id (hash layers) has a defined slot order; the discarded slots add
    # zero either way, so stability is for determinism, not for the equality.
    out_sorted = out6[torch.argsort(ids, stable=True)]
    y = torch.zeros_like(xv, dtype=torch.float32)
    for slot in range(out_sorted.size(0)):
        y += out_sorted[slot:slot + 1]
    y += self.shared_experts(xv)
    self._grouped_steps = getattr(self, "_grouped_steps", 0) + 1
    return y.type_as(xv).view(shape)


def coverage(module):
    """Per-MoE grouping coverage under `module`: {layer_id: (grouped steps, {reason: declines})}.

    The answer to "did the lever actually fire, on every layer?" — the question the profile's kernel
    table could only answer by arithmetic, and the one three levers in a row have got wrong. A layer
    that never grouped shows 0 grouped steps and the reason it did not; a stage where every layer
    grouped shows every entry non-zero. Duck-typed like `bank_layout`, so a test can pass a bare MoE
    or a stand-in and a profiler can pass `stage.layers`.

    READ IT AS "DID THIS LAYER EVER GROUP", NOT AS A STEP COUNT, under CUDA graphs. The counters are
    host-side python, so a captured region runs them once at capture and never on replay — a graphed
    stage would report 1 step where it served hundreds. WHICH layers grouped stays exact, and that is
    the question this exists to answer; the per-step number is only honest eager. That is no longer
    hypothetical: `V4_MOE_IN_GRAPH=1` captures this forward inside the whole-layer graph
    (v4_whole_layer_graph), so on such a stage every grouped layer reports exactly the handful of
    steps its captures and its probe ran, and `v4_levers`' MoE-in-graph coverage is the live count."""
    out = {}
    for m in module.modules() if hasattr(module, "modules") else [module]:
        if hasattr(m, "experts") and hasattr(m, "experts_start_idx"):
            out[getattr(m, "layer_id", len(out))] = (getattr(m, "_grouped_steps", 0),
                                                     dict(getattr(m, "_grouped_declined", {})))
    return out


def install(mod):
    """Rebind `mod.MoE.forward` to the grouped decode path. Returns True if it took.

    Runs AFTER model.py is executed (like `v4_moe_decode.install`), and AFTER `v4_moe_decode` has
    rebound the forward: it captures whatever forward is currently there as its fallback, so the two
    compose — grouped handles the single-token score-routed step, and everything it declines lands on
    the decode path (which in turn declines to the true reference). Idempotent, and a no-op under
    `V4_MOE_GROUPED=0` or without CUDA (the kernel is CUDA-only; installing on a CPU box would defer
    the JIT failure to layer 0 of a real run)."""
    global _REF_FORWARD, _MOD, _WORLD_SIZE
    if not V4_MOE_GROUPED or getattr(mod.MoE.forward, "_v4_grouped", False):
        return False
    if not torch.cuda.is_available():
        return False
    _REF_FORWARD = mod.MoE.forward
    _MOD = mod
    _WORLD_SIZE = int(getattr(mod, "world_size", 1) or 1)
    grouped_forward._v4_grouped = True
    mod.MoE.forward = grouped_forward
    return True


def _fp8_block_quant(w, block=128):
    """Quantize a bf16 weight to (fp8, e8m0 scale) with the 128x128 block layout `fp8_gemm` reads.

    A weight `Linear` is per-block on BOTH dims — scale is [ceil(out/128), ceil(in/128)] e8m0, one
    power-of-2 per 128x128 tile — which is NOT what `act_quant` produces (that is the per-ROW 1x128
    activation layout, scale [out, in//128]). The shipped model's shared expert and gate are fp8, so
    the harness has to build valid fp8 weights the weight way, not the activation way. Values are
    synthetic, so this only has to be a self-consistent fp8/e8m0 pair the kernel can consume; the
    power-of-2 scale matches the ue8m0 the run uses."""
    out, inn = w.shape
    b = w.float().unflatten(0, (out // block, block)).unflatten(-1, (inn // block, block))
    amax = b.abs().amax(dim=(1, 3)).clamp_min(1e-4)              # [out//128, in//128]
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))  # power of 2, as e8m0 requires
    q = (b / scale[:, None, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.reshape(out, inn).contiguous(), scale.to(torch.float8_e8m0fnu).contiguous()


def build_real_dims_moe(mod, args, seed=0, layer_id=7, bank=False):
    """A single MoE at V4's shipped dims with random-but-self-consistent quantized weights.

    `bank=True` applies the shipped LOAD-TIME layout (`_relayout_moe`) after the weights are written,
    so the harness proves the kernel against the exact tensor layout a stage serves from rather than
    against the lazy stack. It is applied after, not before, only because this harness fills the
    weights by hand where a stage fills them from a checkpoint; the layout copies either way.

    No 158 GiB checkpoint: routed experts get valid fp4 (packed weight + e8m0 scale) by quantizing a
    random bf16 draw through the reference's own `fp4_act_quant`; the shared expert is fp8, so it gets
    a 128x128-block fp8 weight (`_fp8_block_quant`, the layout `fp8_gemm` reads — NOT `act_quant`'s
    per-row activation layout); the gate stays bf16 (F.linear). `layer_id >= n_hash_layers` picks
    score routing. The
    globals (`default_dtype`, `scale_fmt`, `scale_dtype`) are what `Transformer.__init__` sets for
    dtype=fp8/scale_dtype=fp8 — a stage building Blocks directly must set them itself (see
    v4_ref_cpu's docstring), so the harness does too."""
    from kernel import fp4_act_quant
    torch.manual_seed(seed)
    with mod.set_dtype(torch.bfloat16), torch.device("cuda"):
        moe = mod.MoE(layer_id, args)
    blk = mod.block_size

    def rand(*sh):
        return torch.randn(*sh, dtype=torch.bfloat16, device="cuda") * 0.02

    with torch.no_grad():
        for i in range(moe.experts_start_idx, moe.experts_end_idx):
            e = moe.experts[i]
            for lin, out_f, in_f in ((e.w1, args.moe_inter_dim, args.dim),
                                     (e.w3, args.moe_inter_dim, args.dim),
                                     (e.w2, args.dim, args.moe_inter_dim)):
                w, s = fp4_act_quant(rand(out_f, in_f), mod.fp4_block_size)
                lin.weight.data.copy_(w)
                lin.scale.data.copy_(s)
        for lin, out_f, in_f in ((moe.shared_experts.w1, args.moe_inter_dim, args.dim),
                                 (moe.shared_experts.w3, args.moe_inter_dim, args.dim),
                                 (moe.shared_experts.w2, args.dim, args.moe_inter_dim)):
            w, s = _fp8_block_quant(rand(out_f, in_f), blk)
            lin.weight.data.copy_(w)
            lin.scale.data.copy_(s)
        moe.gate.weight.data.copy_(rand(args.n_routed_experts, args.dim).to(moe.gate.weight.dtype))
        if moe.gate.hash:                        # layer_id < n_hash_layers: an expert-id table, and
            moe.gate.tid2eid.data.random_(0, args.n_routed_experts)   # its ids may REPEAT
        else:
            moe.gate.bias.data.normal_(0, 0.02)
    if bank:
        assert _relayout_moe(moe), "the bank layout must take on a shipped-dims fp4 MoE"
    return moe.eval()


def real_dims_args(mod):
    return mod.ModelArgs(dim=4096, moe_inter_dim=2048, n_routed_experts=256, n_activated_experts=6,
                         n_shared_experts=1, n_hash_layers=3, score_func="sqrtsoftplus",
                         route_scale=1.5, swiglu_limit=10.0, dtype="fp8", scale_dtype="fp8",
                         expert_dtype="fp4")


def _load_model_module():
    import importlib.util
    import sys
def _vendored(name):
    """Locate a vendored reference tree, in the repo AND on a deployed box.

    The repo keeps upstream trees under vendor/ so the engine directories hold only our code. A
    rented stage does NOT have that layout: deploy copies the modules flat into /root with the
    reference tree beside them, which is the whole reason the engines are flat modules. So try the
    sibling first (deployed), then the repo's vendor/. Resolving only one way silently breaks the
    other, and the box is the one that would not be caught by a test run."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, name),
                 os.path.join(here, os.pardir, os.pardir, "vendor", name)):
        if os.path.isdir(cand):
            return os.path.normpath(cand)
    return os.path.join(here, name)


    inf = os.path.join(_vendored("deepseek_v4_ref"), "inference")
    if inf not in sys.path:
        sys.path.insert(0, inf)
    spec = importlib.util.spec_from_file_location("dsv4_model", os.path.join(inf, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dsv4_model"] = mod
    spec.loader.exec_module(mod)
    torch.set_default_dtype(torch.bfloat16)
    mod.world_size, mod.rank = 1, 0
    mod.default_dtype = torch.float8_e4m3fn
    mod.scale_fmt, mod.scale_dtype = "ue8m0", torch.float8_e8m0fnu
    return mod


def _smoke():
    """JIT + run at V4's real MoE shape against the reference — `torch.equal`, both layouts.

    Two MoEs from the SAME seed: one left in the reference's per-expert layout, one given the shipped
    bank layout. The bank one is the oracle's mirror, so this pins three things at once — the grouped
    decode step is bit-exact to the reference (the original bar), the bank layout does not perturb it,
    and the s > 1 reference path reading the banked experts as views is bit-exact to the same path
    reading standalone tensors (the fallback prefill/verify-chunk still take)."""
    assert torch.cuda.is_available(), "v4 moe grouped smoke needs a CUDA device"
    mod = _load_model_module()
    args = real_dims_args(mod)
    ref_moe = build_real_dims_moe(mod, args)
    bank_moe = build_real_dims_moe(mod, args, bank=True)
    assert bank_moe._grouped_bank, "the bank layout did not take"
    ref_forward = mod.MoE.forward
    install(mod)
    hash_ref = build_real_dims_moe(mod, args, seed=1, layer_id=0)
    hash_bank = build_real_dims_moe(mod, args, seed=1, layer_id=0, bank=True)
    # PIN a duplicated routing rather than hoping a random `tid2eid` draws one: at 256 experts and
    # topk 6 a draw repeats only ~5.7% of the time, so eight random draws miss the duplicate branch
    # — the branch that needs the kernel to be row-invariant — about 62% of runs. Expert 3 named
    # three times and 17 twice is the reference discarding three of the six slots.
    pattern = torch.tensor([3, 3, 17, 3, 200, 17], dtype=torch.int32, device="cuda")
    with torch.no_grad():
        for m in (hash_ref, hash_bank):
            m.gate.tid2eid.data.copy_(pattern.expand_as(m.gate.tid2eid))
    for tag, (a, b) in (("decode", (ref_moe, bank_moe)), ("hash", (hash_ref, hash_bank))):
        for t in range(8):
            x = torch.randn(1, 1, args.dim, dtype=torch.bfloat16, device="cuda")
            ids = torch.randint(0, args.vocab_size, (1, 1), device="cuda")
            with torch.no_grad():
                ref = ref_forward(a, x, ids)
                got = grouped_forward(b, x, ids)
            eq = torch.equal(ref, got)
            d = (ref.float() - got.float()).abs().max().item()
            print(f"{tag} draw {t}  equal={eq}  max|d|={d:.3e}")
            assert eq, f"grouped MoE ({tag}) on the bank layout is not bit-exact to the reference"
    for t, s in enumerate((2, 5, 17)):
        x = torch.randn(1, s, args.dim, dtype=torch.bfloat16, device="cuda")
        ids = torch.randint(0, args.vocab_size, (1, s), device="cuda")
        with torch.no_grad():
            ref = ref_forward(ref_moe, x, ids)
            got = grouped_forward(bank_moe, x, ids)      # s > 1: falls through, reads the bank views
        eq = torch.equal(ref, got)
        print(f"prefill s={s}  equal={eq}  max|d|={(ref.float() - got.float()).abs().max().item():.3e}")
        assert eq, "the s > 1 fallback over banked experts is not bit-exact to the reference"
    print("bit-exact over 8 score-routed + 8 hash-routed decode draws + 3 prefill shapes, "
          "on the bank layout")


if __name__ == "__main__":
    _smoke()
