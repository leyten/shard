"""K3's Attention-Residual collapse, on fla-core's fused kernel instead of the reference's matmul.

AttnRes is the op that makes a K3 stage's payload (hidden, block_residual) rather than hidden: every
layer RMS-norms each of its block snapshots plus the running prefix sum, scores them against a
learned [1, hidden] projection, softmaxes over sources and takes the weighted sum
(kimi_k3_ref/modeling_kimi_linear.py:1075 `_apply_attn_res`). It runs TWICE per layer plus once at
the model output -- 187 times in a 93-layer forward -- so at decode it is pure launch overhead and
at prefill it is the difference between 1.2 ms and 27 ms.

    K3_ATTNRES_BACKEND = auto | fla | ref
      ref   Moonshot's own function, untouched. The M1 default and every CPU parity test.
      fla   `fla.ops.attnres.fused_attnres` (fla-core 0.5.2). G0 (2026-07-28, sm_120) measured it at
            NB=8 over hidden 7168: decode 118.6 us eager but 8.19 us CUDA-graph-replayed (a 14x
            collapse -- all three variants are launch-bound at T=1, so graphs are what make AttnRes
            free), prefill T=8192 1.239 ms against the reference's 27.49 ms (22x). Agreement with
            the reference was <= 1.6e-2 abs / 2.8e-3 rel at bf16, i.e. rounding.
      auto  fla when it is importable AND there is a CUDA device, else ref.

INSTALLATION IS A MODULE-LEVEL REBIND, NOT A WRAPPER. `_forward_attn_residual` resolves
`_apply_attn_res` out of its own module globals, and the vendored reference is kept byte-identical
to Moonshot's, so the swap has to happen where `_tf_compat` already absorbs version skew: one
auditable rebind on the reference module. k3_stage.logits() and KimiLinearModel.forward pick it up
for free, which is the point -- the tail's own norm/proj pair collapses the stack one last time and
a tail running a different kernel from its own layers would be a very quiet bug.

TWO THINGS THE KERNEL DOES NOT DO THE WAY YOU MIGHT ASSUME
  * `scale` defaults to 1.0, not D**-0.5. K3's reference does not scale its scores either
    (`scores = (k * score_weight).sum(-1)`), so the default is right here and would be wrong for
    almost any other caller of this op.
  * The gluon backend is OFF unless FLA_ATTNRES_GLUON is set, and the flag is read per call. This
    module sets it at import so `fla` means the backend G0 actually measured, not the default
    Triton path that happens to share the entry point.
"""
import os

import torch

K3_ATTNRES_BACKEND = os.environ.get("K3_ATTNRES_BACKEND", "auto")
# Read live by fla's dispatcher on every call, so setting it here (before the first call, not
# necessarily before `import fla`) is enough. Left alone if the operator already chose.
os.environ.setdefault("FLA_ATTNRES_GLUON", "1")

_FUSED = None


def _fused():
    """fla's fused_attnres, memoized, or None if the package is not importable."""
    global _FUSED
    if _FUSED is None:
        try:
            from fla.ops.attnres import fused_attnres
        except ImportError:
            _FUSED = False
        else:
            _FUSED = fused_attnres
    return _FUSED or None


def available():
    """fla-core's attnres kernels are Triton and raise on a CPU tensor, so a device is part of it."""
    return _fused() is not None and torch.cuda.is_available()


def backend():
    """Which AttnRes implementation this box resolves to under K3_ATTNRES_BACKEND."""
    if K3_ATTNRES_BACKEND in ("ref", "fla"):
        return K3_ATTNRES_BACKEND
    if K3_ATTNRES_BACKEND != "auto":
        raise RuntimeError(f"K3_ATTNRES_BACKEND={K3_ATTNRES_BACKEND!r} — expected auto, fla or ref")
    return "fla" if available() else "ref"


def fused_apply_attn_res(prefix_sum, block_residual, proj, norm):
    """`_apply_attn_res`'s contract, served by fla's kernel.

    prefix_sum [T, H]; block_residual [T, NB, H]; proj an nn.Linear with a [1, H] weight; norm a
    KimiRMSNorm. The kernel takes the sources as a SEQUENCE, one [T, H] tensor each, in the same
    order the reference concatenates them (blocks first, prefix sum last) -- handing it the stacked
    [T, NB, H] tensor instead would be read as T sources of shape [NB, H], which is not a shape
    error, just a different model."""
    fused = _fused()
    srcs = [block_residual[:, i] for i in range(block_residual.shape[1])] + [prefix_sum]
    return fused(proj.weight.squeeze(0), srcs, norm.weight, rms_eps=norm.variance_epsilon)


def install(M):
    """Point the reference module's `_apply_attn_res` at the resolved backend. Returns the backend.

    Idempotent: the original is stashed as `_apply_attn_res_ref` on first call and restored when the
    resolved backend is `ref`, so a process that flips K3_ATTNRES_BACKEND between Stages cannot end
    up with a half-swapped module."""
    if not hasattr(M, "_apply_attn_res_ref"):
        M._apply_attn_res_ref = M._apply_attn_res
    be = backend()
    if be == "fla" and _fused() is None:
        raise RuntimeError(
            "K3_ATTNRES_BACKEND=fla but `fla.ops.attnres` is not importable — pip install "
            "'fla-core==0.5.2' (the version G0 proved on sm_120)")
    M._apply_attn_res = fused_apply_attn_res if be == "fla" else M._apply_attn_res_ref
    return be


__all__ = ["available", "backend", "fused_apply_attn_res", "install"]
