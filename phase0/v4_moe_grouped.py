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

    w1,w3,w2 = gather(w*_bank, ids)                # device gather of the six routed experts, no sync
    xq, xs = act_quant(x)                          # ONE quant of the token, shared by all experts
    gate6  = grouped_fp4_gemm(xq, w1)              # ONE launch: all six w1 as a grid-indexed batch
    up6    = grouped_fp4_gemm(xq, w3)              # ONE launch: all six w3
    h6     = weight * (silu(clamp(gate6)) * clamp(up6))   # batched SwiGLU, torch, all six at once
    hq, hs = act_quant(h6)                         # ONE quant of the six intermediates
    out6   = grouped_fp4_gemm(hq, w2)              # ONE launch: all six w2
    y      = sum(out6 in ascending-expert-id order) + shared_expert(x)

Three routed-GEMM launches instead of ~120, and — crucially — ZERO `.tolist()`. The slot->expert map
never touches the host: the six ids gather the routed experts out of the bank with a device index
(`_gather_fp`), the kernel is then a plain grid-indexed batched GEMM over that gathered [G, N, K]
bank, and the ascending-id accumulation order is a device `argsort` gather plus a fixed six-add loop,
not a host-side `sorted()`. That removes the data-dependent `.tolist()` that is the first of the
three things blocking a CUDA-graph capture of the decode layer.

(The tidier design — pass the whole 256-expert bank and dereference `W[eids[slot]]` INSIDE the kernel
— does not survive this sm_120 tilelang build: a per-block data-dependent leading index into a
packed-fp4 bank mis-addresses, uniform eids work but distinct eids collapse every slot onto one
weight. The torch gather is the working equivalent; it is a handful of extra device launches, still
no host sync, still CUDA-graph-capturable, still ~8 launches against the reference's ~120.)

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
  hash-routed layers (layer_id < 3)       `tid2eid` can name the same expert twice, and the
                                          duplicate-index `y[idx] +=` is the reference's semantics.
                                          Top-k routing cannot repeat, so on a score-routed decode
                                          step the six ids are distinct and no host-side dedup — and
                                          so no sync — is needed.

WEIGHT BANK — AND THE VRAM BOUND THAT KEEPS THIS OFF A FULL RING FOR NOW. The per-step gather slices
the routed experts out of ONE contiguous [n_experts, N, K] fp4 bank (+ its scale bank), not the
reference's per-expert `nn.Parameter`s (which cannot be gathered in a single op). This file stacks
them on first use and caches them on the module. THAT STACK IS A COPY: at the shipped dims it is
~3.2 GiB per layer beside ~3.7 GiB of routed weights that are still alive, so a stage holding more
than a layer or two cannot afford it. Making it affordable is a LOAD-TIME layout choice (store the
experts as a bank and drop the per-expert tensors) that belongs in the loader and is NOT implemented
here. Until it is, `_expert_bank` measures free VRAM and DECLINES — the layer falls back to the
decode path and says so — rather than OOMing mid-first-token and taking the stage, and the ring, with
it. For a single layer on a 32 GiB card, which is what the parity/bench harness builds, the cached
copy fits and the fast path runs.

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
    block_M = 32
    block_N = 128
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
    (contiguity asserts, fp32-vs-e8m0 scale dtype, output in the process default dtype)."""
    assert a.is_contiguous() and w.is_contiguous(), "grouped fp4: a and w must be contiguous"
    assert a_s.is_contiguous() and w_s.is_contiguous(), "grouped fp4: scales must be contiguous"
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


# VRAM the bank build must leave FREE on the card after it copies. The bank is an exact DUPLICATE of
# the layer's routed-expert weights (see _expert_bank), and it is built lazily, on the first decode
# token, AFTER the graph pools are pinned and the KV cache is allocated — the worst possible moment to
# discover the card is full. 2 GiB covers the per-step gathers, activations, and the compressed-KV
# region still growing under a long job.
_BANK_HEADROOM_BYTES = 2 << 30


def _bank_fits(experts):
    """Would stacking these experts leave `_BANK_HEADROOM_BYTES` free? (True off-CUDA: no bound to check.)"""
    if not torch.cuda.is_available():
        return True
    need = sum(t.numel() * t.element_size()
               for e in experts for t in (e.w1.weight, e.w1.scale, e.w3.weight, e.w3.scale,
                                          e.w2.weight, e.w2.scale))
    free, _total = torch.cuda.mem_get_info()
    return free - need >= _BANK_HEADROOM_BYTES


def _expert_bank(moe):
    """Stack this MoE's routed experts into contiguous [E, N, K] fp4 banks (+ e8m0 scale banks).

    Cached on the module the first time a decode step runs it. `_gather_fp` slices the six routed
    experts out of this bank per step; the reference's per-expert `nn.Parameter`s cannot be gathered
    in one op, a bank can.

    THE STACK COPIES, AND ON THE REAL MODEL THAT IS A SECOND COPY OF THE LAYER'S EXPERTS. At the
    shipped dims (dim 4096, moe_inter_dim 2048, 256 fp4 routed experts) that is ~3.2 GiB per layer
    against ~3.7 GiB of routed weights — so a stage holding more than a layer or two CANNOT afford it
    while the reference's per-expert `nn.Parameter`s are still alive. The fix is a LOAD-TIME layout
    choice (store the experts as a bank and drop the per-expert tensors) that lives in the loader, not
    here, and is NOT on this branch.

    Until it is, this DECLINES rather than OOMs: returning None sends the caller to the reference
    forward for good on this layer. That is the difference between a lever that quietly does not
    engage and a stage process that dies mid-first-token and cascades the whole ring — nothing
    upstream catches an OOM out of `ffn` (v4_stage's _BlockGraphs guards only the graph capture)."""
    bank = getattr(moe, "_grouped_bank", None)
    if bank is not None:
        return bank if bank is not False else None
    lo, hi = moe.experts_start_idx, moe.experts_end_idx
    experts = [moe.experts[i] for i in range(lo, hi)]
    if not _bank_fits(experts):
        moe._grouped_bank = False                          # decided ONCE: never retry, never thrash
        print(f"[v4] grouped MoE declined on layer {getattr(moe, 'layer_id', '?')} — the expert bank "
              f"does not fit alongside the per-expert weights; this layer stays on the decode path. "
              f"V4_MOE_GROUPED needs the load-time bank layout to run on a full stage.", flush=True)
        return None
    bank = {
        "w1": torch.stack([e.w1.weight for e in experts]).contiguous(),
        "w1_s": torch.stack([e.w1.scale for e in experts]).contiguous(),
        "w3": torch.stack([e.w3.weight for e in experts]).contiguous(),
        "w3_s": torch.stack([e.w3.scale for e in experts]).contiguous(),
        "w2": torch.stack([e.w2.weight for e in experts]).contiguous(),
        "w2_s": torch.stack([e.w2.scale for e in experts]).contiguous(),
    }
    moe._grouped_bank = bank
    return bank


# ── the forward ──────────────────────────────────────────────────────────────────────────────────

def grouped_forward(self, x, input_ids):
    """MoE.forward for a single-token, score-routed, single-rank step — three grouped launches, no
    host sync. Every other shape falls through to the captured reference forward. See module doc."""
    shape = x.size()
    xv = x.view(-1, self.dim)
    if xv.size(0) != 1 or _WORLD_SIZE > 1 or self.gate.hash:
        return _REF_FORWARD(self, x, input_ids)

    weights, indices = self.gate(xv, input_ids.flatten())
    ids = indices[0].to(torch.int32)                       # [G] on device — no .tolist()
    bank = _expert_bank(self)
    if bank is None:                                       # the bank would not fit — decline, loudly
        return _REF_FORWARD(self, x, input_ids)

    # Gather the six routed experts into contiguous [G, N, K] banks (device-side, no host sync — see
    # `_gather_fp` on why a torch gather rather than a device-side kernel index). One gather per
    # weight, then the kernel is a plain grid-indexed batched GEMM.
    w1, w1_s = _gather_fp(bank["w1"], ids), _gather_fp(bank["w1_s"], ids)
    w3, w3_s = _gather_fp(bank["w3"], ids), _gather_fp(bank["w3_s"], ids)
    w2, w2_s = _gather_fp(bank["w2"], ids), _gather_fp(bank["w2_s"], ids)

    scale_fmt, scale_dtype = _MOD.scale_fmt, _MOD.scale_dtype
    block = _MOD.block_size
    act_quant = _MOD.act_quant

    # w1 / w3: one act_quant of the token, broadcast to the G expert slots, two grouped GEMMs. The
    # token is quantized ONCE and its G identical rows are what per-expert `w1(x)` would each
    # quantize, so row g stays bit-identical to the reference's expert-g call.
    G = ids.numel()
    xq1, xs1 = act_quant(xv, block, scale_fmt, scale_dtype)
    xq = xq1.expand(G, -1).contiguous()
    xs = xs1.expand(G, -1).contiguous()
    gate6 = grouped_fp4_gemm(xq, xs, w1, w1_s, scale_dtype)
    up6 = grouped_fp4_gemm(xq, xs, w3, w3_s, scale_dtype)

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

    # Accumulate in ascending expert id — the reference's loop order — via a device argsort gather and
    # a fixed six-add fold. fp32 add is not associative, so the order is what keeps this bit-exact.
    out_sorted = out6[torch.argsort(ids)]
    y = torch.zeros_like(xv, dtype=torch.float32)
    for slot in range(out_sorted.size(0)):
        y += out_sorted[slot:slot + 1]
    y += self.shared_experts(xv)
    return y.type_as(xv).view(shape)


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


def build_real_dims_moe(mod, args, seed=0, layer_id=7):
    """A single MoE at V4's shipped dims with random-but-self-consistent quantized weights.

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
        moe.gate.bias.data.normal_(0, 0.02)
    return moe.eval()


def real_dims_args(mod):
    return mod.ModelArgs(dim=4096, moe_inter_dim=2048, n_routed_experts=256, n_activated_experts=6,
                         n_shared_experts=1, n_hash_layers=3, score_func="sqrtsoftplus",
                         route_scale=1.5, swiglu_limit=10.0, dtype="fp8", scale_dtype="fp8",
                         expert_dtype="fp4")


def _load_model_module():
    import importlib.util
    import sys
    inf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepseek_v4_ref", "inference")
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
    """JIT + run at V4's real MoE shape, decode, against the reference — checks `torch.equal`."""
    assert torch.cuda.is_available(), "v4 moe grouped smoke needs a CUDA device"
    mod = _load_model_module()
    args = real_dims_args(mod)
    moe = build_real_dims_moe(mod, args)
    ref_forward = mod.MoE.forward
    install(mod)
    for t in range(8):
        x = torch.randn(1, 1, args.dim, dtype=torch.bfloat16, device="cuda")
        ids = torch.randint(0, args.vocab_size, (1, 1), device="cuda")
        with torch.no_grad():
            ref = ref_forward(moe, x, ids)
            got = grouped_forward(moe, x, ids)
        eq = torch.equal(ref, got)
        print(f"draw {t}  equal={eq}  max|d|={(ref.float() - got.float()).abs().max().item():.3e}")
        assert eq, "grouped MoE is not bit-exact to the reference"
    print("bit-exact over 8 draws")


if __name__ == "__main__":
    _smoke()
