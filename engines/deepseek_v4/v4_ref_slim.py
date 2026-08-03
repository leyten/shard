"""Reference-compute "slim" overrides for the DeepSeek-V4-Flash decode path — removable per-layer work.

A V4 decode layer is CPU-launch-bound, not FLOP-bound: on a 5090 with real weights one token through
one layer is 3.77 ms of ~240 kernel launches, and the arithmetic under them is microseconds. So the
way to make the ring faster is not a faster kernel, it is FEWER launches. This module removes two
blocks of per-layer work that a bf16-KV, short-to-medium-context deployment pays for nothing, each
behind its own default-OFF env flag, each an install-override that rebinds a method on the vendored
reference AFTER model.py is executed — exactly as v4_moe_decode.install() rebinds MoE.forward and
v4_sparse_attn_sm120.install_sm120() rebinds a kernel before it. With both flags off this file changes
nothing: the reference path is byte-identical.

────────────────────────────────────────────────────────────────────────────────────────────────────
ITEM 1 — SKIP THE INDEXER WHILE THE CONTEXT IS SHORT   (V4_REF_SLIM, default off)

Every ratio-4 attention layer (21 of the 43 in the shipped config) runs an Indexer to pick the top
`index_topk` compressed KV slots for sparse attention. That is a big fp8 GEMM (wq_b, q_lora_rank 1024
-> n_heads*head_dim 8192), a Hadamard rotate, an fp4 quant, its OWN Compressor (two fp32 GEMMs), a
score einsum, and a top-k — ~15-22 launches on the hot path of a layer that only has ~240 to begin
with. But the selection it computes is:

    topk_idxs = index_score.topk(min(index_topk, end_pos // ratio), dim=-1)[1]        model.py:433

and while `end_pos // ratio <= index_topk` that `min` is `end_pos // ratio` — the top-k selects EVERY
compressed slot. At the shipped index_topk=512, ratio=4 that holds for every position up to end_pos =
2051 (2051//4 = 512; 2052//4 = 513 is the first discriminating step). In that regime the Indexer's
attended SET is, element for element, the set the ratio-128 layers already build with no indexer at
all — `get_compress_topk_idxs` (model.py:275). So the whole scoring apparatus computes an answer that
is knowable a priori, and this override returns that fixed index instead of running it.

WHY THE SET IS PROVABLY IDENTICAL (not "usually", not "within a tolerance"):
  decode  (start_pos>0, seqlen==1):  the indexer top-k over `end_pos//ratio` slots, select-all, is a
          permutation of arange(end_pos//ratio), then `+ offset`. get_compress_topk_idxs is
          arange((start_pos+1)//ratio) + offset, and (start_pos+1)//ratio == end_pos//ratio. Same set.
  prefill (start_pos==0):  the indexer causally -inf-masks slots past each row's boundary, top-k
          selects all `seqlen//ratio` slots (select-all), then re-masks the over-boundary ones to -1
          with `topk_idxs >= arange(1,seqlen+1)//ratio` — the exact mask get_compress_topk_idxs
          applies. Per row the valid (non -1) set is {j+offset : j < (i+1)//ratio}, identical; the -1
          padding count matches; only the WITHIN-ROW ORDER differs.

WHAT THE APPROXIMATION IS — ONE bf16 ULP, FROM GATHER ORDER, AND NOTHING ELSE.  The two indices differ
only in order: the indexer returns them score-sorted, the fixed index returns them ascending. sparse_attn
gathers the SAME KV rows either way, but its online softmax reduces them in the given order, and fp32
addition is not associative, so the attention output can move by ~1 bf16 ULP. This is the SAME class of
difference the shipped sm_120 sparse-attn retile already accepts (v4_sparse_attn_sm120: "5 elements per
1.7M by one bf16 ulp"), so it changes nothing about the ring's numerical contract. The correctness of
WHICH keys are attended is exact; only the last-bit rounding of the average over them is order-sensitive.
(Corroborating, not load-bearing: vLLM's DeepSeek-V3.2 indexer path is reported to drop the indexer
Hadamard rotate with "no accuracy effect" — the indexer's exact numerics are known to be forgiving. Our
claim is stronger and does not rest on it: in the select-all regime the SELECTION is exact regardless of
any score numerics, because every slot is selected.)

THE CORRECTNESS GATE — THE COMPRESSOR MUST STAY ADVANCED FOR A JOB THAT CAN CROSS 2048.  The Indexer's
scoring is skippable, but its Compressor is STATE: `self.compressor(x, start_pos)` (model.py:423) folds
each token into the indexer's own compressed KV cache, which the indexer READS the moment it re-engages
past the crossover. Skip the compressor while short and the indexer, when it finally has to
discriminate, scores against a half-empty cache and picks the wrong keys — silently, in plausible
numbers. So the default is to keep running the (cheap, 2-GEMM) compressor and skip only the scoring;
the compressor is skipped too ONLY for a job whose maximum position is GUARANTEED to stay in the
select-all regime (`set_job_max_pos`, below). tests/test_v4_ref_slim.py pins both: the re-engage past
the crossover is bit-exact when the compressor was kept, and a run that under-declares its horizon
(compressor skipped, then crosses anyway) diverges.

────────────────────────────────────────────────────────────────────────────────────────────────────
ITEM 2 — SKIP THE inplace KV/Q QAT QUANT-SIMULATION   (V4_REF_SLIM_NOQAT, default off)

Every layer does `act_quant(kv[..., :-rd], 64, ..., inplace=True)` (model.py:512) and the indexer
layers add `fp4_act_quant(q, ..., inplace=True)` (:422); the Compressor (:376/:378) and DSparkAttention
(:761/:780) do the same. `inplace=True` is NOT a quantization that saves memory — it quantizes to
fp8/fp4 and DEQUANTIZES straight back to bf16, a pure precision REDUCTION whose only purpose is to make
a bf16 run match a deployment that stores an fp8/fp4 KV cache (quantization-aware training simulation).
It also forces a `.contiguous()` copy every call, because `kv[..., :-rd]` is a non-contiguous slice.

For a deployment that stores a bf16 KV cache there is nothing to simulate: leaving KV at full bf16 is
STRICTLY MORE precision than the reference, not less. This override no-ops the inplace path (returns the
tensor untouched) and delegates every non-inplace call — the real fp8/fp4 GEMM quantization in
Linear.forward (model.py:120/123) — to the reference unchanged. It is gated SEPARATELY from item 1 and
must stay OFF for a deployment that really does store an fp8 KV cache, where the QAT match is load-bearing.
APPROXIMATE (it removes a deliberate precision reduction); the decode logits move within a documented,
bounded amount that tests/test_v4_ref_slim.py measures.

────────────────────────────────────────────────────────────────────────────────────────────────────
HOW THE SERVE PATH DRIVES THIS

  install(mod)            wired into v4_ref_cpu.load_ref() after v4_moe_decode.install(mod); reads the
                          two env flags, so it is a no-op unless one is set. A stage inherits it.
  set_job_max_pos(n)      the coordinator calls this at job start with the job's GUARANTEED maximum
                          absolute position (prompt length + max_new_tokens). It is the ONLY input the
                          skip needs beyond start_pos, which the layer already has: with it the override
                          knows whether the indexer will ever re-engage and therefore whether the
                          indexer Compressor is skippable. Unknown (None, the default) is always safe —
                          the compressor is kept advanced, so a job of any length stays correct and only
                          the guaranteed-short optimisation is forgone. Call set_job_max_pos(None)
                          between jobs; the value is process-wide, matching model.py's other globals.
  set_active / set_active_noqat   runtime on/off for the two items once installed (tests A/B with them;
                          production leaves them on).
"""
import os

import torch

V4_REF_SLIM = os.environ.get("V4_REF_SLIM", "0") not in ("", "0")
V4_REF_SLIM_NOQAT = os.environ.get("V4_REF_SLIM_NOQAT", "0") not in ("", "0")

# Captured at install(), before anything is rebound — the reference the fast paths fall back to.
_REF_INDEXER_FORWARD = None                 # dsv4_model.Indexer.forward
_GET_COMPRESS_TOPK_IDXS = None              # dsv4_model.get_compress_topk_idxs, the fixed index
_REF_ACT_QUANT = None                       # dsv4_model.act_quant (the CPU stand-in or tilelang)
_REF_FP4_ACT_QUANT = None                   # dsv4_model.fp4_act_quant

# Runtime toggles. Default on: once install() rebinds, the item runs. Tests flip them to A/B in one
# process without rebuilding the class; production never touches them.
_ACTIVE = True                              # item 1
_ACTIVE_NOQAT = True                        # item 2

# The job's guaranteed maximum absolute position (prompt + max_new_tokens), or None for "unknown".
# Only decides whether the indexer's Compressor may be skipped; None keeps it advanced (always safe).
_JOB_MAX_POS = None


# ── item 1: skip the indexer scoring while every compressed slot is selected anyway ────────────────

def _keep_compressor(indexer, ratio):
    """Must the indexer's Compressor keep advancing while we skip its scoring?

    YES unless the job is guaranteed to stay in the select-all regime for its whole length — because
    a job that crosses `index_topk*ratio` re-engages the indexer, and the indexer reads a Compressor
    cache that was never filled if we skipped it. `_JOB_MAX_POS is None` means "horizon unknown", the
    safe answer to which is always keep it advanced."""
    if _JOB_MAX_POS is None:
        return True
    return (_JOB_MAX_POS // ratio) > indexer.index_topk


def slim_indexer_forward(self, x, qr, start_pos, offset):
    """Indexer.forward with the scoring removed while the top-k selects every compressed slot.

    Falls through to the reference for the discriminating regime (`end_pos//ratio > index_topk`) and
    whenever the item is toggled off. In the select-all regime it returns the fixed compressed index —
    the same SET the reference top-k would, differing only in order (see the module docstring) — and,
    unless the job is guaranteed short, keeps the Compressor advanced so a later re-engage is correct."""
    if not _ACTIVE:
        return _REF_INDEXER_FORWARD(self, x, qr, start_pos, offset)
    bsz, seqlen, _ = x.size()
    ratio = self.compress_ratio
    end_pos = start_pos + seqlen
    if end_pos // ratio > self.index_topk:              # the top-k really selects a subset now
        return _REF_INDEXER_FORWARD(self, x, qr, start_pos, offset)
    if _keep_compressor(self, ratio):
        # The one piece of the indexer that is STATE, not a query — advance it exactly as the
        # reference's first lines do (model.py:414-416, :423), so re-engagement is bit-exact.
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        self.compressor(x, start_pos)
    return _GET_COMPRESS_TOPK_IDXS(ratio, bsz, seqlen, start_pos, offset)


# ── item 2: don't simulate an fp8/fp4 KV cache a bf16 deployment doesn't have ───────────────────────

def slim_act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """act_quant, with the inplace QAT round-trip removed. Non-inplace (real fp8 GEMM quant) is the
    reference's, untouched — only the inplace `quantize-then-dequantize-into-x` becomes a no-op."""
    if _ACTIVE_NOQAT and inplace:
        return x
    return _REF_ACT_QUANT(x, block_size, scale_fmt, scale_dtype, inplace)


def slim_fp4_act_quant(x, block_size=32, inplace=False):
    """fp4_act_quant, inplace QAT round-trip removed; non-inplace (the packed fp4 form) untouched."""
    if _ACTIVE_NOQAT and inplace:
        return x
    return _REF_FP4_ACT_QUANT(x, block_size, inplace)


# ── the hooks ──────────────────────────────────────────────────────────────────────────────────────

def set_job_max_pos(n):
    """Declare the job's guaranteed maximum absolute position, or None to forget it. See the docstring.

    n = prompt_len + max_new_tokens. Only affects whether the indexer Compressor may be skipped; the
    per-step indexer skip itself is decided from start_pos and needs nothing from here."""
    global _JOB_MAX_POS
    _JOB_MAX_POS = None if n is None else int(n)


def set_active(on=True):
    """Runtime on/off for item 1 (the indexer skip) once installed."""
    global _ACTIVE
    _ACTIVE = bool(on)


def set_active_noqat(on=True):
    """Runtime on/off for item 2 (the QAT-sim skip) once installed."""
    global _ACTIVE_NOQAT
    _ACTIVE_NOQAT = bool(on)


def install(mod, item1=None, item2=None):
    """Rebind the reference's methods for whichever items are enabled. Returns {item: took?}. Idempotent.

    Runs AFTER model.py is executed (v4_moe_decode.install's window, not v4_kernels_cpu.install's):
    item 1 replaces a method on the Indexer class the exec created, item 2 rebinds the module-global
    `act_quant`/`fp4_act_quant` names that model.py's `from kernel import ...` bound and every call site
    reads. `item1`/`item2` default to the env flags; a test passes explicit bools. A no-op — leaving the
    reference byte-identical — when the flag is off or the item is already installed."""
    global _REF_INDEXER_FORWARD, _GET_COMPRESS_TOPK_IDXS, _REF_ACT_QUANT, _REF_FP4_ACT_QUANT
    i1 = V4_REF_SLIM if item1 is None else bool(item1)
    i2 = V4_REF_SLIM_NOQAT if item2 is None else bool(item2)
    took = {"indexer_skip": False, "noqat": False}
    if i1 and not getattr(mod.Indexer.forward, "_v4_ref_slim", False):
        _REF_INDEXER_FORWARD = mod.Indexer.forward
        _GET_COMPRESS_TOPK_IDXS = mod.get_compress_topk_idxs
        slim_indexer_forward._v4_ref_slim = True
        mod.Indexer.forward = slim_indexer_forward
        took["indexer_skip"] = True
    if i2 and not getattr(mod.act_quant, "_v4_ref_slim", False):
        _REF_ACT_QUANT = mod.act_quant
        _REF_FP4_ACT_QUANT = mod.fp4_act_quant
        slim_act_quant._v4_ref_slim = True
        slim_fp4_act_quant._v4_ref_slim = True
        mod.act_quant = slim_act_quant
        mod.fp4_act_quant = slim_fp4_act_quant
        took["noqat"] = True
    return took


def uninstall(mod):
    """Put the reference's own methods back. Returns {item: removed?}. Idempotent; safe if never installed.

    `install` rebinds names on a module object that is a PROCESS SINGLETON (v4_ref_cpu.load_ref caches
    it), so "OFF by default changes nothing" is only checkable if ON is also reversible — otherwise the
    first caller to arm the slim path arms it for everything that shares the interpreter afterwards.
    That is not hypothetical: tests/test_v4_ref_slim.py armed it module-scoped and left the Indexer
    rebound for the rest of the pytest process, which made tests/test_v4_stage.py's chunked-verify
    parity test compare a slim LOOP against a reference CHUNK and fail (the sets matched, the gather
    ORDER did not -- see _chunk_indexer, which reimplements the indexer and never sees this rebind).

    `set_active(False)` makes the installed override DELEGATE, which is enough for behaviour; this
    restores the reference's own function object, which is what an identity assertion can see."""
    global _REF_INDEXER_FORWARD, _GET_COMPRESS_TOPK_IDXS, _REF_ACT_QUANT, _REF_FP4_ACT_QUANT
    gone = {"indexer_skip": False, "noqat": False}
    if getattr(mod.Indexer.forward, "_v4_ref_slim", False) and _REF_INDEXER_FORWARD is not None:
        mod.Indexer.forward = _REF_INDEXER_FORWARD
        _REF_INDEXER_FORWARD = _GET_COMPRESS_TOPK_IDXS = None
        gone["indexer_skip"] = True
    if getattr(mod.act_quant, "_v4_ref_slim", False) and _REF_ACT_QUANT is not None:
        mod.act_quant = _REF_ACT_QUANT
        mod.fp4_act_quant = _REF_FP4_ACT_QUANT
        _REF_ACT_QUANT = _REF_FP4_ACT_QUANT = None
        gone["noqat"] = True
    return gone
