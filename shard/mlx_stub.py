"""MlxRuntimeStub — MlxRuntime's contract double, pure numpy, no mlx, no weights.

Same class surface as shard/mlx_runtime.MlxRuntime (load_shard / reset / embed / forward /
logits / heartbeat / .aux) over a tiny deterministic fake model, so protocol-level code and
tests exercise the EXACT call/shape/dtype/rollback contract on any box. The point is the
contract, not the math: each fake layer has real per-position KV state (a cached value row
per absolute position, attended by a causal running mean), so a broken crop-to-start_pos
rollback CHANGES the output — the offline tests can prove the spec-decode rollback semantics
byte-for-byte instead of trivially passing on a stateless matmul.

Fake model: N_LAYERS=4, H=8, VOCAB=32. Layer li:  v = x@Wv;  cache[li] = crop(start)+v;
att[s] = mean(cache[:start+s+1]);  x = x + att@Wo + tanh(x@Wm).  All weights are fixed
functions of the layer index (seeded default_rng), so two stubs with the same range are
bit-identical — determinism is part of the contract (receipts).
"""
import numpy as np

from .mlx_runtime import aux_layers_from_env, bf16_bits_to_f32, f32_to_bf16_bits
from .node import LayerRange, ModelRuntime

N_LAYERS = 4
H = 8
VOCAB = 32


def _mat(seed: int, rows: int, cols: int, scale: float) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal((rows, cols)) * scale).astype(np.float32)


class MlxRuntimeStub(ModelRuntime):
    """MlxRuntime's surface in numpy. Wire contract mirrored exactly: forward takes
    [1,S,H] float32 or bf16-bits (uint16) and returns the SAME representation; embed and
    logits return float32; aux is {absolute_layer_idx: [S,H]} for aux_layers ∩ [lo:hi),
    refreshed per forward. B=1 only, like the real thing."""

    def __init__(self, model: str = "stub", layer_range: LayerRange = None,
                 is_head: bool = False, is_tail: bool = False, device: str = "cpu",
                 aux_layers=None):
        super().__init__(model, layer_range or LayerRange(0, N_LAYERS), is_head, is_tail, device)
        self.aux_layers = aux_layers_from_env() if aux_layers is None \
            else frozenset(int(a) for a in aux_layers)
        self.aux = {}
        self._loaded = False
        self._cache = None       # {absolute layer idx: [n_cached_positions, H] float32}

    # ---- lifecycle ----
    def load_shard(self) -> None:
        lo, hi = self.layer_range.start, self.layer_range.end
        if not (0 <= lo < hi <= N_LAYERS):
            raise ValueError(f"layer range [{lo}:{hi}) outside the stub's {N_LAYERS} layers")
        s = 0.5 / np.sqrt(H)
        self._Wv = {li: _mat(100 + li, H, H, s) for li in range(lo, hi)}
        self._Wo = {li: _mat(200 + li, H, H, s) for li in range(lo, hi)}
        self._Wm = {li: _mat(300 + li, H, H, s) for li in range(lo, hi)}
        if self.is_head:
            self._emb = _mat(7, VOCAB, H, 1.0)
        if self.is_tail:
            self._norm_w = np.ones(H, dtype=np.float32)
            self._lm = _mat(9, VOCAB, H, s)
        self._cache = {li: np.zeros((0, H), dtype=np.float32) for li in range(lo, hi)}
        self._loaded = True

    def reset(self) -> None:
        self.aux = {}
        if self._loaded:
            self._cache = {li: np.zeros((0, H), dtype=np.float32) for li in self._cache}

    def heartbeat(self) -> dict:
        return {"alive": True, "backend": "stub", "device": self.device, "model": self.model,
                "layers": [self.layer_range.start, self.layer_range.end],
                "is_head": self.is_head, "is_tail": self.is_tail, "loaded": self._loaded}

    # ---- forward ----
    def _require_loaded(self):
        if not self._loaded:
            raise RuntimeError("load_shard() first")

    def embed(self, token_ids) -> np.ndarray:
        self._require_loaded()
        if not self.is_head:
            raise RuntimeError("embed() is head-only")
        ids = np.asarray(token_ids)
        if ids.ndim == 1:
            ids = ids[None]
        return self._emb[ids].astype(np.float32)     # [1,S,H]

    def forward(self, hidden_states, start_pos: int) -> np.ndarray:
        self._require_loaded()
        h_np = np.asarray(hidden_states)
        if h_np.dtype not in (np.float32, np.uint16):   # NEVER coerce — the real class's
            raise TypeError(f"forward() takes float32 or bf16-bits (uint16) frames, got "
                            f"{h_np.dtype} — dequantize fp8 wire frames at the boundary")
        bits = h_np.dtype == np.uint16
        x3 = bf16_bits_to_f32(h_np) if bits else h_np.astype(np.float32)
        if x3.ndim != 3 or x3.shape[0] != 1 or x3.shape[2] != H:
            raise ValueError(f"expected [1,S,{H}] hidden states, got {h_np.shape}")
        S = x3.shape[1]
        x = x3[0]                                    # [S,H]
        self.aux = {}
        for li in range(self.layer_range.start, self.layer_range.end):
            c = self._cache[li]
            if c.shape[0] < start_pos:               # a gap would mis-position the fake KV,
                raise ValueError(f"KV gap: cache at {c.shape[0]}, forward at start_pos "
                                 f"{start_pos}")     # exactly like real RoPE would
            v = x @ self._Wv[li]
            c = np.concatenate([c[:start_pos], v], 0)   # CROP to start_pos, then append —
            self._cache[li] = c                          # rollback overwrites stale spec KV
            att = np.stack([c[:start_pos + s + 1].mean(0) for s in range(S)])
            x = x + att @ self._Wo[li] + np.tanh(x @ self._Wm[li])
            if li in self.aux_layers:
                self.aux[li] = f32_to_bf16_bits(x)   # ALWAYS bf16 bits (m25's unconditional
                                                     # bf16 aux; wire aux contract is bf16)
        out = x[None]
        return f32_to_bf16_bits(out) if bits else out.astype(np.float32)

    def logits(self, hidden_states) -> np.ndarray:
        self._require_loaded()
        if not self.is_tail:
            raise RuntimeError("logits() is tail-only")
        h_np = np.asarray(hidden_states)
        x = bf16_bits_to_f32(h_np) if h_np.dtype == np.uint16 else h_np.astype(np.float32)
        v = np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6)    # rms norm, m25 tail shape
        return ((x / v) * self._norm_w) @ self._lm.T            # [1,S,VOCAB] float32
