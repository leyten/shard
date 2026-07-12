"""MlxRuntime — Apple-silicon (MLX) backend behind the ModelRuntime firewall.

The third implementation of shard/node.py's `ModelRuntime` (next to M25Runtime's hand-rolled
CUDA path and the planned VllmRuntime): serve one contiguous block of a model's layers out of
a Mac's unified memory, speaking the SAME per-node contract the proven m25 serve loop runs on
(head embeds; forward(hidden, start_pos) with crop-to-start_pos KV rollback; tail does final
norm + lm_head; EAGLE aux hidden-states captured after specific absolute layer indices). The
model layer is rented from mlx-lm (mlx_lm.models.minimax + gather_qmm quant kernels for
MiniMax-M2.5); the contract is ours — see docs/MODEL_RUNTIME.md.

STATUS — untested-but-faithful: written WITHOUT Apple silicon on hand. The mlx-facing calls
mirror mlx-lm's own load_model/pipeline patterns (per-layer DecoderLayer(x, mask, cache)
calls, make-prompt-cache-style per-layer KVCache objects, None-padded unowned layer slots,
config["quantization"]-driven nn.quantize), but nothing here has touched a Metal device.
Everything that CAN run offline lives in pure functions below (weight-key selection, the
bf16 wire-bit bridging) plus shard/mlx_stub.py (the same class surface in numpy), and is
gated by tests/test_mlx_runtime.py. docs/MLX_RUNTIME.md lists the exact Mac-gated checks
that must pass before this goes near a ring.

Numerics + receipts (the honest part):
  * community MLX conversions (e.g. mlx-community/MiniMax-M2.5-4bit) are group-64 AFFINE
    4-bit over the experts AND the attention; our NVIDIA path is NVFP4 experts + bf16
    attention. Same accepted-kernel-numerics CLASS as the fp8 wire / graph-manual attention:
    high per-step greedy agreement, NO token-exact cross-backend parity, divergence grows
    with generation length. A Mac stage makes the served model a placement-defined
    mixed-quant composite — quote g PER BACKEND (never compare a Mac ring's g to a CUDA
    ring's), and receipts MUST pin (backend, quant scheme, checkpoint hash) per stage.
  * recommended before any real deployment: our OWN mlx_lm.convert with a keep-attention-
    bf16 predicate (~+4GB total on M2.5 — attn is ~44M params/layer vs ~3.6B expert
    params/layer) so the Mac's precision POLICY matches the NVIDIA path (NVFP4-class
    experts-only quant). That closes the policy gap, not bit-exactness.
  * wired-limit: MLX has no expert streaming — a stage's weights must be RESIDENT in
    unified memory. Exceeding macOS's iogpu.wired_limit_mb (default ~2/3–3/4 of RAM) is
    swap-death, not graceful degradation; size layer ranges to the RAISED limit table in
    docs/MLX_RUNTIME.md (~2.06 GB/layer at 4-bit gs64) and let the admission probe verify.
  * fp8 wire: M25_FP8_WIRE frames must be DEQUANTIZED TO BF16 at the Mac boundary for now —
    mx float8 support is unverified (memo flag). forward() accepts float32 or bf16-bit
    (uint16) numpy only; teaching it e4m3 bytes + scale is a TODO gated on that check.

Batching: B=1 only. forward_prefill_stream/forward_decode_batch stay NotImplementedError —
the de-lockstep per-stream loop can still schedule a Mac stage at B=1, and mlx-lm's
BatchKVCache is there when a batched Mac path earns its keep.
"""
import importlib
import json
import os

import numpy as np

from .node import LayerRange, ModelRuntime


# ── pure helpers (no mlx; offline-testable) ───────────────────────────────────

def select_weight_keys(weight_map: dict, lo: int, hi: int, *,
                       is_head: bool, is_tail: bool, tied: bool = False) -> list:
    """Exactly the tensor names a stage holding layers [lo:hi) must materialize, derived
    from the checkpoint's index `weight_map` — never hardcoded beyond the model family's
    documented "model.layers.{j}." / "model.embed_tokens" / "model.norm" / "lm_head" naming
    (the same prefixes shard/fetch.shards_for_block selects FILES by; this selects TENSORS,
    so a quantized checkpoint's .scales/.biases companions ride along automatically because
    they share the prefix).

    `tied` comes from the CONFIG (tie_word_embeddings), never inferred from the map: a tail
    whose index merely LACKS lm_head.* on an untied model must fail here, not fall back to
    projecting logits through a random-init substitute (the silent-corruption class this
    loader exists to refuse). Tied tails pull embed_tokens (logits via embed.as_linear).
    Pure function so the selection logic is testable without mlx or weights on disk."""
    prefixes = tuple(f"model.layers.{j}." for j in range(lo, hi))
    keys = {k for k in weight_map if k.startswith(prefixes)} if prefixes else set()
    if is_head:
        keys |= {k for k in weight_map if k.startswith("model.embed_tokens.")}
        if not any(k.startswith("model.embed_tokens.") for k in weight_map):
            raise ValueError("head stage but weight_map has no model.embed_tokens.* — "
                             "corrupt/partial index, refusing a random embedding")
    if is_tail:
        keys |= {k for k in weight_map if k.startswith("model.norm.")}
        if not any(k.startswith("model.norm.") for k in weight_map):
            raise ValueError("tail stage but weight_map has no model.norm.* — corrupt/partial index")
        if tied:
            if not any(k.startswith("model.embed_tokens.") for k in weight_map):
                raise ValueError("tied-embedding tail but weight_map has no model.embed_tokens.* — "
                                 "corrupt/partial index")
            keys |= {k for k in weight_map if k.startswith("model.embed_tokens.")}
        else:
            if not any(k.startswith("lm_head.") for k in weight_map):
                raise ValueError("untied model but weight_map has no lm_head.* — corrupt/partial "
                                 "index, refusing a random lm_head (config says tie_word_embeddings"
                                 "=false; a tied checkpoint would say so in its config)")
            keys |= {k for k in weight_map if k.startswith("lm_head.")}
    return sorted(keys)


def files_for_keys(weight_map: dict, keys) -> list:
    """The safetensors files those tensors live in (what the loader actually opens)."""
    return sorted({weight_map[k] for k in keys})


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bit-pattern (uint16), round-to-nearest-even. numpy has no native
    bfloat16 and we take no new dep (ml_dtypes), so bf16 wire frames are carried as their
    raw bits in uint16 arrays — the same 2 bytes/element the CUDA path ships. Assumes
    finite activations (NaN payloads may not round-trip bit-exactly — e.g. a signaling
    NaN can round to +Inf; forward() rejects non-finite frames at the boundary)."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((b + np.uint32(0x7FFF) + ((b >> np.uint32(16)) & np.uint32(1)))
            >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """bf16 bit-pattern (uint16) -> float32 (exact: bf16 is a truncated float32)."""
    return (np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32)
            << np.uint32(16)).view(np.float32)


def aux_layers_from_env() -> frozenset:
    """m25_stage's aux convention, verbatim: capture only when M25_EAGLE=1, layer ids from
    M25_EAGLE_AUX (default "1,30,58" — SpecForge RAW decoder-layer indices, the OUTPUT
    residual stream of those layers; see m25_stage.py's index-convention comment)."""
    if os.environ.get("M25_EAGLE", "0") == "0":
        return frozenset()
    return frozenset(int(x) for x in os.environ.get("M25_EAGLE_AUX", "1,30,58").split(","))


# ── the runtime ────────────────────────────────────────────────────────────────

def _mlx():
    """Import the mlx stack at CALL time, never at module import: this module must be
    importable on any box (CI, CUDA rings, the coordinator's scheduler) so the offline
    tests and placement code can see the class without Apple silicon."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.models import base as base_mod
        from mlx_lm.models import cache as cache_mod
    except ImportError as e:
        raise ImportError(
            "MlxRuntime requires mlx + mlx-lm (Apple-silicon backend): pip install mlx mlx-lm "
            f"— underlying: {e}") from e
    return mx, nn, base_mod, cache_mod


class MlxRuntime(ModelRuntime):
    """Serves layers [lo:hi) of an MLX-converted checkpoint on Apple silicon.

    `model` is the LOCAL DIRECTORY of an MLX conversion (config.json +
    model.safetensors.index.json + safetensors — e.g. a fetched
    mlx-community/MiniMax-M2.5-4bit); the verified-fetch layer (shard/fetch.py) delivers
    it, this class only loads. `aux_layers` = ABSOLUTE decoder-layer indices whose OUTPUT
    residual stream is snapshotted into `.aux` ({layer_idx: np.ndarray [S,H]}) after each
    forward — m25_stage's EAGLE aux contract; None derives it from M25_EAGLE/M25_EAGLE_AUX
    like the CUDA stage does. Wire dtypes: forward() takes numpy float32 or bf16-bits
    (uint16) [1,S,H] and returns the SAME representation; embed/logits return float32."""

    def __init__(self, model: str, layer_range: LayerRange,
                 is_head: bool = False, is_tail: bool = False, device: str = "metal",
                 aux_layers=None):
        super().__init__(model, layer_range, is_head, is_tail, device)
        self.aux_layers = aux_layers_from_env() if aux_layers is None \
            else frozenset(int(a) for a in aux_layers)
        self.aux = {}            # last forward's {layer_idx: np [S,H]} for in-range aux layers
        self._model = None       # mlx-lm Model, layers None-padded outside [lo:hi)
        self._inner = None       # the .model submodule (embed_tokens/layers/norm live here)
        self._layers = None      # absolute-index layer list (None outside the range)
        self._cache = None       # one mlx-lm KVCache per OWNED layer, parallel to range(lo,hi)
        self._quant = None       # checkpoint quantization dict (receipt pinning)
        self._tied = False

    # ---- lifecycle ----
    def load_shard(self) -> None:
        """Range loader: materialize ONLY layers [lo:hi) (+ embed if head, + final norm &
        lm_head if tail) from the MLX conversion. The pattern is mlx-lm's own lazy
        pipeline load: build the full Model skeleton (mlx array init is lazy — unevaluated
        random inits cost no memory), None-pad every unowned slot (PipelineMixin's trick,
        which also keeps ABSOLUTE layer indexing so start_pos/aux ids need no remapping),
        quantize per config["quantization"], load only the range's tensors with
        strict=False, then mx.eval exactly those arrays so nothing else ever materializes."""
        mx, nn, base_mod, cache_mod = _mlx()
        lo, hi = self.layer_range.start, self.layer_range.end
        with open(os.path.join(self.model, "config.json")) as f:
            cfg = json.load(f)
        with open(os.path.join(self.model, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]

        self._tied = bool(cfg.get("tie_word_embeddings", False))
        keys = select_weight_keys(weight_map, lo, hi, is_head=self.is_head,
                                  is_tail=self.is_tail, tied=self._tied)
        for li in range(lo, hi):
            if not any(k.startswith(f"model.layers.{li}.") for k in keys):
                raise RuntimeError(f"weight_map covers no tensors for layer {li} — "
                                   "index/config mismatch, refusing a silently-random layer")
        weights = {}
        for fn in files_for_keys(weight_map, keys):   # mx.load is lazy: reading the file maps
            data = mx.load(os.path.join(self.model, fn))     # it, bytes move only at mx.eval
            weights.update({k: data[k] for k in keys if weight_map[k] == fn and k in data})
        missing = [k for k in keys if k not in weights]
        if missing:
            raise RuntimeError(f"index lists {len(missing)} tensors absent from their files "
                               f"(first: {missing[0]}) — corrupt/partial checkpoint")

        # model class via mlx-lm's registry remap (M2.5's config says model_type
        # "minimax_m2"; mlx-lm serves it from models/minimax.py). Fall back to the
        # documented map if the constant moves — the memo verified the mapping, not its
        # import path.
        try:
            from mlx_lm.utils import MODEL_REMAPPING as remap
        except ImportError:
            remap = {"minimax_m2": "minimax"}
        mt = cfg["model_type"]
        arch = importlib.import_module(f"mlx_lm.models.{remap.get(mt, mt)}")
        args = arch.ModelArgs.from_dict(cfg)
        model = arch.Model(args)
        inner = getattr(model, "model", model)        # embed_tokens/layers/norm submodule
        layers = inner.layers
        n_layers = len(layers)
        if not (0 <= lo < hi <= n_layers):
            raise ValueError(f"layer range [{lo}:{hi}) outside model's {n_layers} layers")

        if hasattr(model, "sanitize"):                # no-op on pre-converted checkpoints;
            weights = model.sanitize(weights)         # restacks per-expert keys on raw ones
        for i in range(n_layers):                     # None-pad unowned slots (PipelineMixin)
            if not (lo <= i < hi):
                layers[i] = None
        if not (self.is_head or (self.is_tail and self._tied)):
            inner.embed_tokens = None                 # unowned edges: drop so a stray eval
        if not self.is_tail:                          # can never materialize them
            inner.norm = None
            if hasattr(model, "lm_head"):
                model.lm_head = None

        q = cfg.get("quantization")
        if q is not None:                             # mlx-lm load_model's quant recipe: per-
            def class_predicate(p, m):                # path overrides ride in the same dict
                if p in q:                            # (e.g. MoE gates at 8-bit, or False to
                    return q[p]                       # skip); a module is quantized only if
                if not hasattr(m, "to_quantized"):    # the checkpoint shipped its .scales —
                    return False                      # out-of-range modules have none in our
                return f"{p}.scales" in weights       # subset, and they are None-padded anyway
            kw = {"group_size": q["group_size"], "bits": q["bits"]}
            if "mode" in q:                           # mxfp4-mode conversions declare it
                kw["mode"] = q["mode"]
            nn.quantize(model, class_predicate=class_predicate, **kw)

        # THE completeness audit — the anti-silent-corruption gate. After pruning, the module
        # tree's parameters are EXACTLY this stage's owned set, so diff it BOTH ways against
        # the loaded keys: any parameter the checkpoint subset doesn't feed (a partial layer,
        # a missing quant .biases companion, an lm_head the index dropped) would stay at mlx's
        # lazy RANDOM init and materialize at first forward — wrong hidden-states behind valid
        # receipts, the exact enemy. Any key without a home is an index/model mismatch. Only
        # then load, strict (shape-checked) — non-strict has nothing left to forgive.
        from mlx.utils import tree_flatten  # noqa: PLC0415 — lazy like the rest of the mlx stack
        params = {k for k, _ in tree_flatten(model.parameters())}
        unfed = sorted(params - set(weights))
        if unfed:
            raise RuntimeError(f"{len(unfed)} module parameters not covered by the checkpoint "
                               f"subset (first: {unfed[0]}) — refusing a partially-random module")
        homeless = sorted(set(weights) - params)
        if homeless:
            raise RuntimeError(f"{len(homeless)} checkpoint keys have no home in the pruned "
                               f"module (first: {homeless[0]}) — index/model mismatch")
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(list(weights.values()))               # materialize EXACTLY the range's bytes
        self._model, self._inner, self._layers = model, inner, layers
        self._quant = q
        self._cache = [cache_mod.KVCache() for _ in range(lo, hi)]

    def reset(self) -> None:
        """Drop this block's KV (new request, or a rollback past the cached span). Fresh
        cache OBJECTS, not trims — mirrors m25 Layer.reset()'s clean-slate semantics."""
        self.aux = {}
        if self._model is None:
            self._cache = None
            return
        _, _, _, cache_mod = _mlx()
        self._cache = [cache_mod.KVCache()
                       for _ in range(self.layer_range.start, self.layer_range.end)]

    def heartbeat(self) -> dict:
        """Liveness + unified-memory telemetry. Never raises — a heartbeat that throws
        reads as a dead node. Includes the quant scheme so the scheduler/receipts can pin
        the stage's numerics class (backend, quant, checkpoint — see module docstring)."""
        hb = {"alive": True, "backend": "mlx", "device": self.device, "model": self.model,
              "layers": [self.layer_range.start, self.layer_range.end],
              "is_head": self.is_head, "is_tail": self.is_tail,
              "loaded": self._model is not None}
        try:
            import mlx.core as mx
        except ImportError as e:
            hb["alive"] = False
            hb["error"] = f"mlx unavailable: {e}"
            return hb
        if self._quant is not None:
            hb["quant"] = {k: self._quant[k] for k in ("bits", "group_size", "mode")
                           if k in self._quant}
            # digest of the FULL quant dict: a mixed-precision conversion (per-path
            # overrides, e.g. attention-bf16) must not heartbeat identically to a uniform
            # one — receipts pin the digest, not just the headline bits.
            import hashlib  # noqa: PLC0415
            hb["quant_sha"] = hashlib.sha256(
                json.dumps(self._quant, sort_keys=True).encode()).hexdigest()[:16]
        try:                                          # Metal-only introspection; guarded so a
            hb["device_info"] = dict(mx.metal.device_info())   # CPU-build mlx still heartbeats
        except Exception:
            pass
        try:
            get_mem = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
            hb["active_mem_bytes"] = int(get_mem())
        except Exception:
            pass
        return hb

    # ---- forward (the hot path) ----
    def _require_loaded(self):
        if self._model is None:
            raise RuntimeError("load_shard() first")

    def _to_np(self, a, bits: bool) -> np.ndarray:
        """mx array -> numpy in the caller's wire representation (bf16 bit-pattern uint16,
        or float32). bf16 leaves as raw bits — numpy can't hold bf16 natively."""
        mx = _mlx()[0]
        if bits:
            return np.array(mx.view(a.astype(mx.bfloat16), mx.uint16))
        return np.array(a.astype(mx.float32))

    def embed(self, token_ids) -> np.ndarray:
        """Head only: token-ids -> layer-0 hidden states, float32 numpy [1,S,H]."""
        self._require_loaded()
        if not self.is_head:
            raise RuntimeError("embed() is head-only")
        mx = _mlx()[0]
        ids = np.asarray(token_ids)
        if ids.ndim == 1:
            ids = ids[None]
        h = self._inner.embed_tokens(mx.array(ids.astype(np.int32)))
        return self._to_np(h, bits=False)

    def forward(self, hidden_states, start_pos: int) -> np.ndarray:
        """Run this block's layers over `hidden_states` ([1,S,H] numpy, float32 or bf16-bits
        uint16) starting at absolute position `start_pos`; returns the same shape in the
        same representation. Compute is mx bf16 (own conversion at the boundary — see the
        fp8-wire note in the module docstring).

        KV rollback mirrors m25 Layer.attn EXACTLY: each owned layer's cache is CROPPED
        back to start_pos before the layer runs (KVCache.trim moves the write offset; the
        next update_and_fetch overwrites the stale speculative slots), so a re-prefill at
        an earlier start needs no extra bookkeeping. A start_pos AHEAD of the cache would
        silently mis-position RoPE (mlx-lm layers read positions off cache.offset), so a
        gap is a hard error, never a garbage forward.

        EAGLE aux: after each layer whose ABSOLUTE index is in `aux_layers` ∩ [lo:hi), the
        output residual stream is snapshotted into self.aux[li] as [S,H] numpy bf16-bits
        (uint16) — ALWAYS bf16 regardless of the frame representation, m25's unconditional
        `_AUX ... .to(torch.bfloat16)` (the drafter-side contract is identical)."""
        self._require_loaded()
        mx, _, base_mod, _ = _mlx()
        lo, hi = self.layer_range.start, self.layer_range.end
        h_np = np.asarray(hidden_states)
        if h_np.dtype not in (np.float32, np.uint16):
            # NEVER coerce: an fp8-wire frame (uint8 e4m3 bytes) coerced to float32 becomes
            # the integers 0..255 as activations — a full garbage forward with healthy
            # transport. fp8 frames must be dequantized to bf16 at the Mac boundary.
            raise TypeError(f"forward() takes float32 or bf16-bits (uint16) frames, got "
                            f"{h_np.dtype} — dequantize fp8 wire frames at the boundary")
        bits = h_np.dtype == np.uint16
        x = bf16_bits_to_f32(h_np) if bits else h_np
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(f"expected [1,S,H] hidden states, got {h_np.shape}")
        if not np.isfinite(x).all():
            raise ValueError("non-finite hidden states at the stage boundary — upstream "
                             "corruption; refusing to propagate")
        h = mx.array(np.ascontiguousarray(x, dtype=np.float32)).astype(mx.bfloat16)

        for c in self._cache:                         # crop-to-start_pos = rollback semantics
            if c.offset > start_pos:
                c.trim(c.offset - start_pos)
            elif c.offset < start_pos:
                raise ValueError(f"KV gap: cache at {c.offset}, forward at start_pos "
                                 f"{start_pos} — protocol never skips positions")
        # bottom-right-aligned causal mask offset by the (now-cropped) cache span — the
        # exact call mlx-lm model forwards make; layers accept whatever it returns.
        mask = base_mod.create_attention_mask(h, self._cache)
        self.aux = {}
        for off, li in enumerate(range(lo, hi)):
            h = self._layers[li](h, mask, cache=self._cache[off])
            if li in self.aux_layers:
                # ALWAYS bf16 bits, regardless of the frame representation — m25 captures
                # aux as bf16 unconditionally and the wire's aux contract is bf16 (fp8-aux
                # packing happens wrapper-side). TODO(perf, Mac): collect handles and
                # mx.eval once after the loop instead of a Metal eval per aux layer.
                self.aux[li] = self._to_np(h[0], True)
        mx.eval(h)
        return self._to_np(h, bits)

    def logits(self, hidden_states) -> np.ndarray:
        """Tail only: final norm + lm_head (embed.as_linear when tied) -> float32 numpy
        [1,S,vocab]. RMSNorm precision is mlx's kernel (fp32 accumulation internally) vs
        m25_pipe._tail_logits' explicit fp32 — same accepted-kernel-numerics class."""
        self._require_loaded()
        if not self.is_tail:
            raise RuntimeError("logits() is tail-only")
        mx = _mlx()[0]
        h_np = np.asarray(hidden_states)
        if h_np.dtype not in (np.float32, np.uint16):
            raise TypeError(f"logits() takes float32 or bf16-bits (uint16), got {h_np.dtype} "
                            "— dequantize fp8 wire frames at the boundary")
        x = bf16_bits_to_f32(h_np) if h_np.dtype == np.uint16 else h_np
        h = mx.array(np.ascontiguousarray(x, dtype=np.float32)).astype(mx.bfloat16)
        h = self._inner.norm(h)
        out = self._inner.embed_tokens.as_linear(h) if self._tied else self._model.lm_head(h)
        return self._to_np(out, bits=False)
