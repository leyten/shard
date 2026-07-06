"""hf_runtime — the generic HF/Transformers ModelRuntime (one engine, every model).

`GenericHFRuntime` implements `shard/node.py`'s `ModelRuntime` interface over the proven
`pipeline.load_stage` / `run_block` path (AutoModelForCausalLM with a device_map that puts
only this block's layers [lo:hi) on the GPU and everything else on "meta"). It is the
universal fallback the MODEL_RUNTIME.md decision calls for: any architecture transformers can
load — including the existing gpt-oss path — serves through this with NO new engine code.

This is NOT a new interface. It is one adapter behind the in-house ModelRuntime firewall
(docs/MODEL_RUNTIME.md): the orchestration above (coordinate_pipe) speaks only token-ids +
hidden-states + argmax over sockets and rides on this unchanged. The model-specific concerns
(rotary tables, attention impl, MoE routing, quant) stay hidden inside load_stage/run_block.

Relationship to the other runtimes:
  - M25Runtime    (phase0/m25_stage.py)  — the tuned hand-rolled betanet fast-path.
  - GenericHFRuntime (this file)         — bf16/native-precision universal fallback (HF).
  - VllmRuntime   (to build, MODEL_RUNTIME.md step 3) — vLLM loader for production
                                            NVFP4/MXFP4/FusedMoE kernels; the generalization
                                            of m25_stage from one model to all of them. This
                                            file is its torch-only sibling and the reference
                                            for the GPU spike (step 2).

Picked per model via the signed registry's `adapter` field (M1): a row with
adapter `generic-vllm` selects VllmRuntime; a `generic-hf`/`hf` row selects this. The
registry also supplies layer_count / tokenizerId so nothing here is hardcoded per arch.

Boundary law: pure engine. Knows weights/activations/kv; nothing about sockets, peers,
receipts, accounts, or $.
"""
from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from shard.node import LayerRange, ModelRuntime


class GenericHFRuntime(ModelRuntime):
    """ModelRuntime backed by pipeline.load_stage / run_block (transformers generic path).

    nstages/stage carry the ring position so the even-split fallback in load_stage matches the
    legacy launcher; lo/hi (from the scheduler plan) override it for the VRAM-aware uneven
    split. Either way load_stage maps everything outside [lo:hi) to meta, so a node holds only
    its block of a model far too big for its card.
    """

    #: registry adapter ids that select this runtime (generic-vllm's torch-only sibling).
    ADAPTERS = ("generic-hf", "hf")

    def __init__(self, model: str, layer_range: LayerRange,
                 is_head: bool = False, is_tail: bool = False, device: str = "cuda:0",
                 nstages: int = 1, stage: int = 0, dtype: str = "auto", attn: str = "eager"):
        super().__init__(model, layer_range, is_head=is_head, is_tail=is_tail, device=device)
        self.nstages = nstages
        self.stage = stage
        self.dtype = dtype
        self.attn = attn
        self._parts: dict | None = None     # the pipeline.load_stage payload (loaded weights)
        self._cache: Any = None             # this block's KV cache (DynamicCache)

    # ---- lifecycle ----
    def load_shard(self) -> None:
        """load only this block's layers (+ embed if head, + norm/lm_head if tail) to vram.

        Weight key names come from transformers' own per-arch module naming via
        AutoModelForCausalLM — derived from the checkpoint, never hardcoded to one
        architecture (satisfies the MODEL_RUNTIME.md "derive, not hardcode" seam for this
        backend; the manifest-fetch path enforces the same on the weight side)."""
        from pipeline import load_stage      # lazy: imports torch + transformers
        self._parts = load_stage(
            self.model, self.stage, self.nstages, device=self.device,
            dtype=self.dtype, attn=self.attn,
            lo=self.layer_range.start, hi=self.layer_range.end)
        # reconcile the role flags / range with what load_stage actually placed.
        self.layer_range = LayerRange(self._parts["lo"], self._parts["hi"])
        self.reset()

    def reset(self) -> None:
        """drop this block's KV cache. called on a new request or a rollback past the cache."""
        from transformers import DynamicCache  # lazy
        self._cache = DynamicCache()

    def heartbeat(self) -> dict:
        """liveness + vram, reported to the scheduler."""
        import torch  # lazy
        used = (torch.cuda.memory_allocated(self.device) / 1e9
                if str(self.device).startswith("cuda") and torch.cuda.is_available() else 0.0)
        return {"model": self.model, "lo": self.layer_range.start, "hi": self.layer_range.end,
                "is_head": self.is_head, "is_tail": self.is_tail, "vram_gb": round(used, 2)}

    # ---- forward (the hot path) ----
    def embed(self, token_ids):
        """head only: token-ids -> hidden states for layer 0."""
        if not self.is_head:
            raise NotImplementedError("embed is a head-only op")
        assert self._parts is not None, "load_shard() first"
        return self._parts["embed"](token_ids)

    def forward(self, hidden_states, start_pos: int):
        """run this block's layers over `hidden_states` at absolute position `start_pos`.

        KV is managed internally (DynamicCache); a re-prefill at an earlier start_pos
        overwrites stale speculative KV, so a spec-decode rollback needs no extra crop — the
        same convention the m25 serve loop and specpipe already rely on."""
        from pipeline import run_block       # lazy
        assert self._parts is not None, "load_shard() first"
        return run_block(hidden_states, self._parts, self._cache, start_pos)

    def logits(self, hidden_states):
        """tail only: final norm + lm_head -> logits [.., vocab]."""
        if not self.is_tail:
            raise NotImplementedError("logits is a tail-only op")
        assert self._parts is not None, "load_shard() first"
        h = self._parts["norm"](hidden_states)
        return self._parts["lm_head"](h)


def make_runtime(adapter: str, model: str, layer_range: LayerRange, **kw) -> ModelRuntime:
    """Select + instantiate a ModelRuntime for a registry `adapter` id.

    Keeps the registry->runtime mapping in ONE place. 'generic-vllm' routes to VllmRuntime
    once it lands (MODEL_RUNTIME.md step 3); until then 'generic-hf'/'hf' use the universal HF
    fallback here. M25Runtime stays selected by its own betanet path. Fail closed on unknown."""
    if adapter in GenericHFRuntime.ADAPTERS:
        return GenericHFRuntime(model, layer_range, **kw)
    if adapter == "generic-vllm":
        # The production quant/MoE kernels live behind VllmRuntime (to build). Until the GPU
        # spike (MODEL_RUNTIME.md step 2) gates it, fall back to the HF path so a generic-vllm
        # row is still servable in native precision rather than unservable.
        try:
            from vllm_runtime import VllmRuntime  # type: ignore  # noqa: F401
            return VllmRuntime(model, layer_range, **kw)
        except Exception:
            return GenericHFRuntime(model, layer_range, **kw)
    raise KeyError(f"no ModelRuntime for adapter {adapter!r} "
                   f"(have: generic-hf/hf, generic-vllm; m25 has its own path)")
