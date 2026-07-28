"""FlashKDA on the GPU, behind the same `fla.ops.kda` names k3_kda_cpu registers.

k3_kda_cpu is the CPU reference the M1 parity suite is measured against; this is the kernel that
has to agree with it on real hardware. It is the third value of K3_KDA_BACKEND:

    cpu       plain torch, this module's reference (M1, CI, any box with no GPU)
    fla       the real flash-linear-attention package (Triton), if one is installed
    flashkda  FlashKDA's CUDA kernel for the recurrence + k3_kda_cpu's torch conv / gated norm

The split in that last line is deliberate. FlashKDA (MoonshotAI/FlashKDA @ d2ff19a) ships ONE thing
-- the delta recurrence, forward only -- and G0 (2026-07-28) proved that thing on sm_120: builds in
54 s with FLASH_KDA_CUDA_ARCHS=120a, one `fwd_launch.sm_120a.cubin` in the .so, decode 21.17 us at
K3's exact shape (H=96, D=128), 16.40 us graph-replayed, bit-exact against the vendor's own torch
reference at T=1 and 6.10e-5 on the chunked path. The short convolution and the gated RMSNorm around
it stay k3_kda_cpu's torch code, which runs on CUDA unchanged and is already pinned to fla's
published math -- renting a second package for two elementwise ops would buy nothing and cost a
parity story, and fla's Triton ShortConvolution cannot hold a fixed-address cache for graph capture.

WHAT THE KERNEL DOES ITSELF (read off csrc/, not guessed -- getting any of these wrong is silent)
  * L2-normalizes q and k, with the epsilon INSIDE the rsqrt (`rsqrtf(q_sq + 1e-6f)`,
    fwd_kernel1.cuh) -- the same formula as k3_kda_cpu._l2norm. Pass RAW q/k. Pre-normalizing is
    redundant work that also perturbs the bf16 the kernel is about to re-normalize.
  * Applies `scale` (bf16) internally. Pass raw q, not q*scale.
  * Applies sigmoid to `beta`. Pass the b_proj LOGITS. Passing a pre-sigmoided beta double-sigmoids.
  * Computes the whole gate from raw `g`: lower_bound * sigmoid(exp(A_log) * (g + dt_bias)), carried
    in log2 units. A_log/dt_bias/lower_bound are mandatory `fwd` arguments; there is no
    gate-already-applied mode. So this backend REFUSES a caller that asks for either fold to happen
    outside -- K3's decoder always asks for both in-kernel, and anything else is a shape of call we
    have not run.

THE ONE THAT WOULD HAVE BEEN SILENT: STATE LAYOUT
FlashKDA's initial_state/final_state are [B, H, V, K] -- V-major, i.e. exactly fla's
`transpose_state_layout=True` / `state_v_first`, which is what Moonshot's decoder always passes. So
the cached state goes STRAIGHT in with no transpose, and the transpose is needed only for a caller
that asked for the plain [B,H,K,V] layout. K and V are both head_dim for KDA, so every shape check
passes either way and a wrong choice here is invisible in the shapes and wrong in the numbers --
the exact trap k3_kda_cpu's docstring flags for M2. tests/test_k3_stage_m2.py's on-GPU parity check
runs TWO sequential decode steps for this reason: one step from a zero state cannot catch it.

STATIC STATE (K3_STATIC_STATE / K3_CUDA_GRAPH). With `STATIC_STATE` set the ops write the new state
back INTO the caller's buffer and return that same object, so `cache.recurrent_states[li] = ret` is
an address-preserving no-op and a captured graph keeps replaying against the live state. Off, they
allocate and return a fresh tensor -- M1's behaviour exactly. FlashKDA takes separate in/out state
pointers and one CTA owns one (sequence, head) slice read-then-write, so aliasing them WOULD be
sound, but the vendor never tests it; we pay one captured device-side copy (6.0 MiB/layer/request at
K3's shape) rather than rely on an unexercised aliasing property.

PRECISION NOTE: an fp32 state is storage only -- the kernel converts it down to bf16 for the running
accumulation (smem_cvt_fp32_to_bf16, fwd_kernel2.cuh) and back up on store. The CPU reference
accumulates in true fp32, so a long carried state is where the two diverge fastest.
"""
import os

import torch

import k3_kda_cpu

# Set by k3_stage when K3_STATIC_STATE (implied by K3_CUDA_GRAPH) is on. Module-level rather than a
# call argument because the call sites are inside Moonshot's vendored decoder, which we do not edit.
STATIC_STATE = False

# FlashKDA hard-checks `D == 128` in C++ (csrc/flash_kda.cpp) -- it is a head_dim-128 kernel, so the
# toy configs the CPU parity suite uses (head_dim 8) can never run through it. Named here so the
# error says that rather than surfacing as a bare TORCH_CHECK from inside the extension.
KERNEL_HEAD_DIM = 128

_MOD = None


def flash():
    """The flash_kda extension module, or a RuntimeError naming the build that produces it."""
    global _MOD
    if _MOD is None:
        try:
            import flash_kda
        except ImportError as e:
            raise RuntimeError(
                "K3_KDA_BACKEND=flashkda but `flash_kda` is not importable. Build it with "
                "FLASH_KDA_CUDA_ARCHS=120a MAX_JOBS=32 pip install --no-build-isolation -e . "
                "(G0: 54 s from a clean checkout on sm_120; needs CUDA >= 12.9, SM90+)") from e
        _MOD = flash_kda
    return _MOD


def available():
    """Is this backend usable on this box? Importability alone is not enough -- the kernel needs a
    device, and a CPU-only box that resolved `flashkda` would die at the first launch."""
    try:
        import flash_kda                                              # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def fused_recurrent_kda(q, k, v, g, beta, A_log=None, dt_bias=None, initial_state=None,
                        output_final_state=False, scale=None, use_qk_l2norm_in_kernel=False,
                        use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
                        allow_neg_eigval=False, lower_bound=None, safe_gate=None,
                        transpose_state_layout=False, state_v_first=False,
                        cu_seqlens=None, **kwargs):
    """KDA's delta recurrence on the GPU. Same signature as k3_kda_cpu's, same return contract.

    q,k,g: [B,T,H,K]  v: [B,T,H,V]  beta: [B,T,H]  ->  (o [B,T,H,V], final_state or None)."""
    transpose_state_layout = transpose_state_layout or state_v_first
    if cu_seqlens is not None:
        raise NotImplementedError("k3_kda_flash: varlen (cu_seqlens) is a batched-serving concern, "
                                  "deferred past M2 — pass one unpadded sequence per call")
    if not (use_qk_l2norm_in_kernel and use_gate_in_kernel and use_beta_sigmoid_in_kernel):
        raise NotImplementedError(
            "k3_kda_flash: FlashKDA folds the q/k l2-norm, the gate and the beta sigmoid in "
            "UNCONDITIONALLY, so it cannot serve a caller that pre-computed any of them "
            f"(l2norm={use_qk_l2norm_in_kernel} gate={use_gate_in_kernel} "
            f"beta_sigmoid={use_beta_sigmoid_in_kernel}). K3's decoder asks for all three.")
    if allow_neg_eigval:
        raise NotImplementedError("k3_kda_flash: allow_neg_eigval (beta*2) has no kernel equivalent")
    B, T, H, Kd = q.shape
    HV, Vd = v.shape[2], v.shape[-1]
    if (HV, Vd) != (H, Kd):
        raise NotImplementedError(
            f"k3_kda_flash: the kernel requires k/v/g to have q's exact shape — no GQA, V == K. "
            f"Got q [B,T,{H},{Kd}] v [B,T,{HV},{Vd}].")
    if Kd != KERNEL_HEAD_DIM:
        raise NotImplementedError(
            f"k3_kda_flash: FlashKDA is a head_dim-{KERNEL_HEAD_DIM} kernel (TORCH_CHECK in "
            f"csrc/flash_kda.cpp) and this config has head_dim {Kd}. Real K3 is 128; a toy parity "
            f"config has to use 128 too, or run K3_KDA_BACKEND=cpu.")
    if A_log.shape != (H,):
        raise RuntimeError(
            f"k3_kda_flash: A_log is {tuple(A_log.shape)}, the kernel indexes it by head and wants "
            f"({H},). K3 ships it padded to head_dim; k3_stage._fixup slices it at load — a stage "
            f"that skipped that slice would decay the wrong heads.")

    bf = torch.bfloat16
    q = q.to(bf).contiguous(); k = k.to(bf).contiguous(); v = v.to(bf).contiguous()
    g = g.to(bf).contiguous()
    beta = beta.to(bf).contiguous()                       # b_proj hands us fp32; the kernel wants bf16
    A = A_log.float().contiguous()
    dtb = dt_bias.float().reshape(H, Kd).contiguous()     # flat [H*K] is rejected; the kernel wants [H,K]
    if scale is None:
        scale = Kd ** -0.5

    # [B,H,V,K] is the kernel's layout AND the layout a transpose_state_layout=True caller already
    # holds, so the common path copies straight in. Only a plain-layout caller needs the transpose.
    s_in = torch.zeros(B, H, Vd, Kd, dtype=torch.float32, device=q.device)
    if initial_state is not None:
        init = initial_state.float()
        s_in.copy_(init if transpose_state_layout else init.transpose(-1, -2))
    s_out = torch.empty_like(s_in)
    out = torch.empty_like(v)
    flash().fwd(q, k, v, g, beta, float(scale), out, A, dtb, float(lower_bound),
                initial_state=s_in, final_state=s_out)

    o = out.to(v.dtype)
    if not output_final_state:
        return o, None
    final = s_out if transpose_state_layout else s_out.transpose(-1, -2).contiguous()
    if STATIC_STATE and initial_state is not None:
        # Address-preserving update: the caller's buffer IS the cache slot a captured graph replays
        # against, so hand back the same object rather than a fresh allocation.
        initial_state.copy_(final)
        return o, initial_state
    return o, final


def chunk_kda(*args, **kwargs):
    """Prefill path -- FlashKDA has ONE entry point; `fwd` chunks internally (CHUNK = 16 rows).

    NOT bit-identical to a token-at-a-time walk (G0: 6.10e-5 max abs at T=128 vs the vendor's own
    torch reference, against 0.0 at T=1) -- chunked matmul reassociation plus an fp16-accumulate
    inner GEMM, the same class as every other chunked linear-attention kernel. That is why the CPU
    backend deliberately makes both paths the same sequential recurrence: it is the reference, not
    an emulator of these numerics."""
    return fused_recurrent_kda(*args, **kwargs)


__all__ = ["KERNEL_HEAD_DIM", "available", "chunk_kda", "flash", "fused_recurrent_kda"]
