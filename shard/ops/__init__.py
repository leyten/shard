"""shard.ops: receipt-grade (batch-invariant) Triton kernels for the v2 stage runtime (BUILD-PLAN R4).

  invariant_moe   MXFP4 W4A16 grouped MoE GEMM, weights packed 4-bit, dequant in-kernel, SPLIT_K=1
  invariant_attn  fixed-split diff-kv attention (QK 192 / V 128), sinks, window 128 / global, paged KV
  hw2_invariant_cost  HW2 T2/T4 measurement: invariant vs fast cost + on-device byte-identity probe

Device families
  NVIDIA sm_80+ via Triton: Ampere 3090/A100 (sm_80/86: bf16 MMA, no FP8/FP4 needed -- the weights are
  decoded to bf16 in registers), Ada 4090 (sm_89), Blackwell 5090 / RTX 6000 Pro (sm_120), GB10 Spark
  (sm_121, ARM host). AMD ROCm via Triton's AMD backend: expected to compile, untested. Apple: NOT covered
  (Triton has no Metal backend); the MLX path needs the same contract as Metal kernels
  (mx.fast.metal_kernel): fixed threadgroup tiles, full-K sequential accumulators, split boundaries at
  absolute KV positions, fixed-order merge -- mx.quantized_matmul / gather_qmm / fast SDPA pick kernels by
  shape and are not invariant as-is. Each (backend, arch) is its own numerics_id class: bytes are promised
  equal only within a class (ARCHITECTURE-v2 §6); the pins are invariant_moe.NUMERICS_PIN and
  invariant_attn.NUMERICS_PIN.

Cost vs the fast path (ESTIMATE, unmeasured -- hw2_invariant_cost decides; the gate is <= 1.5x per layer at
decode/verify sizes, design-runtime §6 R1). MiMo-Flash layer, 5090-class card:
  * MoE decode (B=1..64): memory-bound on the packed expert bytes, which both paths read in full. With
    SPLIT_K=1 the grid is 8 experts x 64 N-tiles = 512 programs for gate/up at B=1 (3 waves on 170 SMs), so
    full-K per program still fills the machine; expected 1.0-1.3x Marlin. Risk: dequant ALU + M=16 MMA
    padding at B=1, Triton codegen quality vs hand Marlin.
  * Global attention decode at 32k: 128 fixed splits per (token, kv head) -> ample parallelism at B=1;
    partial write + merge adds ~5% traffic; BLOCK_KV=16 tiles are small. Expected 1.1-1.5x FlashInfer.
  * SWA attention (128 keys): launch-bound; two launches instead of one (CUDA graphs amortize).
  * Prefill: the per-token program re-reads K/V per query row -> many times slower than a Q-tiled kernel.
    Not in the decode gate but it is TTFT: the Q-tiled variant is TODO (invariant by the same masked-tile
    no-op argument).
"""


def numerics_parts(device=None) -> dict:
    """Everything that decides the receipt-grade bytes, for numerics_id: the kernel pins plus the
    toolchain (Triton/ptxas codegen decides FMA contraction, ex2.approx, division) and the EXACT compute
    capability (not an arch family). Same parts => same compiled kernels => same bytes (the HW2 probe
    checks the last step on each device)."""
    import torch
    import triton

    from . import invariant_attn, invariant_moe
    cuda = torch.cuda.is_available()
    return dict(moe=invariant_moe.NUMERICS_PIN, attn=invariant_attn.NUMERICS_PIN, triton=triton.__version__,
                torch=torch.__version__, cuda=torch.version.cuda,
                capability=list(torch.cuda.get_device_capability(device)) if cuda else None,
                backend="cuda" if cuda else ("interpreter" if invariant_moe.INTERPRET else "none"))
