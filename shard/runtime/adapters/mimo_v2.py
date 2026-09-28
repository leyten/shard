"""MiMo-V2 (V2.6 Flash / Pro) adapter, the device-neutral half: config -> per-layer specs, and the
checkpoint's tensor plan (names, dtypes, shapes as released). Stdlib only, so every backend (CPU
reference, CUDA, MLX) sizes, fetches and validates a span from the same code; the canonical math
and its CPU implementation live in shard/runtime/cpu/mimo_v2.py.

Architecture facts (config.json + the vendor's modeling_mimo_v2.py @ 5711b268, verified on the
released Flash headers; see scratchpad/v2/design-runtime.md §1):
  * `hybrid_layer_pattern[i]`: 0 = global attention, 1 = sliding window (`sliding_window`; key j is
    visible to query i iff i - w < j <= i). Global and SWA layers have their own head counts, head
    dims (QK 192 / V 128 on Flash), RoPE theta (1e7 / 1e4) and KV heads (4 / 8 on Flash).
  * partial RoPE: rotate-half over the FIRST int(qk_dim * partial_rotary_factor) dims of each head.
  * `v *= attention_value_scale` BEFORE caching; learned per-head sink logits (a zero-value extra
    key) on the layers the config flags (SWA only on V2.6).
  * FFN: layer i dense iff moe_layer_freq[i] == 0; MoE = sigmoid router, fp32
    `e_score_correction_bias` added for SELECTION only, weights from the un-biased scores,
    renormalised, no shared experts.
  * storage: experts MXFP4 (U8 packed + U8 E8M0, per expert per projection); qkv and the dense MLP
    FP8-e4m3 with fp32 128x128 block scales, the fused qkv PRE-SHARDED for ckpt_tp =
    num_key_value_heads chunks with a per-chunk scale grid (shard/formats/mimo_v2.py); o_proj,
    norms, router weight and sinks BF16; the router bias F32.
Anything else (group-limited routing, shared experts, split q/k/v, tied embeddings, non-default
RoPE, another quantization) is refused, never guessed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

ARCH = "mimo_v2"
MX_BLOCK = 32                     # OCP MX block (== shard.formats.quant.MX_BLOCK, asserted in tests)


class UnsupportedConfig(ValueError):
    """The config names a MiMo variant this adapter does not implement."""


@dataclass(frozen=True)
class LayerSpec:
    idx: int
    kind: str                   # "global" | "swa"
    n_q: int
    n_kv: int
    qk_dim: int
    v_dim: int
    rope_dim: int
    rope_theta: float
    window: int | None          # SWA: keys (p - window, p]; global: None
    has_sink: bool
    is_moe: bool

    @property
    def kv_row_bytes(self) -> int:
        """bf16 K + V of one position of this layer (the canonical mirror row)."""
        return 2 * self.n_kv * (self.qk_dim + self.v_dim)


@dataclass(frozen=True)
class MimoConfig:
    hidden: int
    n_layers: int
    vocab: int
    eps: float
    v_scale: float | None
    ckpt_tp: int                # the fused qkv's checkpoint TP degree = global num_key_value_heads
    intermediate: int
    moe_intermediate: int
    n_experts: int
    top_k: int
    norm_topk: bool
    routed_scale: float
    window: int
    fp8_block: int
    layers: tuple[LayerSpec, ...]

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(s.kind for s in self.layers)

    @classmethod
    def from_hf(cls, c: dict) -> "MimoConfig":
        """Parse a MiMo-V2 `config.json`; refuse anything the canonical math does not implement."""
        def need(cond, what):
            if not cond:
                raise UnsupportedConfig(f"MiMo-V2 adapter: {what}")
        need(c.get("model_type") == ARCH, f"model_type {c.get('model_type')!r} is not {ARCH!r}")
        need(c.get("hidden_act") == "silu", f"hidden_act {c.get('hidden_act')!r}")
        need(c.get("attention_projection_layout") == "fused_qkv",
             f"attention_projection_layout {c.get('attention_projection_layout')!r} (fused_qkv only)")
        need(not c.get("attention_bias"), "attention_bias")
        need(not c.get("tie_word_embeddings"), "tied embeddings")
        need(c.get("scoring_func") == "sigmoid" and c.get("topk_method") == "noaux_tc",
             f"router {c.get('scoring_func')}/{c.get('topk_method')}")
        need(c.get("n_group") == 1 and c.get("topk_group") == 1, "group-limited routing")
        need(not c.get("n_shared_experts"), "shared experts")
        need(not c.get("rope_scaling"), "rope_scaling")
        rp = c.get("rope_parameters") or {}
        need(rp.get("rope_type", "default") == "default", f"rope_type {rp.get('rope_type')!r}")
        prf = float(c["partial_rotary_factor"])
        need(rp.get("partial_rotary_factor", prf) == prf, "rope_parameters.partial_rotary_factor != partial_rotary_factor")
        q = c.get("quantization_config") or {}
        need(q.get("quant_method") == "fp8" and q.get("fmt") == "e4m3" and q.get("store_dtype") == "mxfp4",
             f"quantization {q.get('quant_method')}/{q.get('fmt')}/{q.get('store_dtype')}")
        bs = q.get("weight_block_size")
        need(isinstance(bs, list) and len(bs) == 2 and bs[0] == bs[1], f"weight_block_size {bs}")
        need(q.get("mxfp4_block_size") == MX_BLOCK, f"mxfp4_block_size {q.get('mxfp4_block_size')}")
        L = int(c["num_hidden_layers"])
        pat, moe = list(c["hybrid_layer_pattern"]), list(c["moe_layer_freq"])
        need(len(pat) == L and len(moe) == L and set(pat) <= {0, 1} and set(moe) <= {0, 1},
             "hybrid_layer_pattern / moe_layer_freq must name every layer with 0/1")
        w = int(c["sliding_window"])
        need(w > 0 and c.get("sliding_window_size", w) == w, "sliding_window")
        H = int(c["hidden_size"])
        need(H % MX_BLOCK == 0 and int(c["moe_intermediate_size"]) % MX_BLOCK == 0, "MXFP4 needs K % 32 == 0")
        tp = int(c["num_key_value_heads"])
        specs = []
        for i in range(L):
            swa = pat[i] == 1
            pre = "swa_" if swa else ""
            n_q = int(c[f"{pre}num_attention_heads"])
            n_kv = int(c[f"{pre}num_key_value_heads"])
            qk = int(c[f"{pre}head_dim"])
            v = int(c[f"{pre}v_head_dim"])
            rd = int(qk * prf)
            need(rd % 2 == 0 and 0 < rd <= qk, f"layer {i}: rotary dim {rd} of {qk}")
            need(n_q % n_kv == 0 and n_q % tp == 0 and n_kv % tp == 0,
                 f"layer {i}: heads {n_q}q/{n_kv}kv vs checkpoint tp {tp}")
            sink = bool(c.get("add_swa_attention_sink_bias" if swa else "add_full_attention_sink_bias", False))
            specs.append(LayerSpec(i, "swa" if swa else "global", n_q, n_kv, qk, v, rd,
                                   float(c["swa_rope_theta" if swa else "rope_theta"]),
                                   w if swa else None, sink, bool(moe[i])))
        rs = c.get("routed_scaling_factor")
        return cls(hidden=H, n_layers=L, vocab=int(c["vocab_size"]), eps=float(c["layernorm_epsilon"]),
                   v_scale=c.get("attention_value_scale"), ckpt_tp=tp,
                   intermediate=int(c["intermediate_size"]), moe_intermediate=int(c["moe_intermediate_size"]),
                   n_experts=int(c["n_routed_experts"]), top_k=int(c["num_experts_per_tok"]),
                   norm_topk=bool(c["norm_topk_prob"]), routed_scale=1.0 if rs is None else float(rs),
                   window=w, fp8_block=int(bs[0]), layers=tuple(specs))


# ── the tensor plan (the checkpoint's own names; derived from the config, never guessed) ───────

def _fp8(name, rows, cols, b):
    return {name: ("F8_E4M3", (rows, cols)),
            name + "_scale_inv": ("F32", (math.ceil(rows / b), math.ceil(cols / b)))}


def layer_prefix(i: int) -> str:
    return f"model.layers.{i}."


def layer_plan(cfg: MimoConfig, i: int) -> dict:
    """{tensor name: (safetensors dtype, shape)} of decoder layer i, exactly as released."""
    s, H, b = cfg.layers[i], cfg.hidden, cfg.fp8_block
    p = layer_prefix(i)
    rows = s.n_q * s.qk_dim + s.n_kv * s.qk_dim + s.n_kv * s.v_dim
    rpc = rows // cfg.ckpt_tp
    out = {p + "input_layernorm.weight": ("BF16", (H,)),
           p + "post_attention_layernorm.weight": ("BF16", (H,)),
           p + "self_attn.qkv_proj.weight": ("F8_E4M3", (rows, H)),
           # per-CHUNK scale grid: ckpt_tp chunks of ceil(rpc / b) rows (shard.formats.mimo_v2)
           p + "self_attn.qkv_proj.weight_scale_inv": ("F32", (cfg.ckpt_tp * math.ceil(rpc / b), math.ceil(H / b))),
           p + "self_attn.o_proj.weight": ("BF16", (H, s.n_q * s.v_dim))}
    if s.has_sink:
        out[p + "self_attn.attention_sink_bias"] = ("BF16", (s.n_q,))
    if not s.is_moe:
        I = cfg.intermediate
        out.update(_fp8(p + "mlp.gate_proj.weight", I, H, b))
        out.update(_fp8(p + "mlp.up_proj.weight", I, H, b))
        out.update(_fp8(p + "mlp.down_proj.weight", H, I, b))
        return out
    E, I = cfg.n_experts, cfg.moe_intermediate
    out[p + "mlp.gate.weight"] = ("BF16", (E, H))
    out[p + "mlp.gate.e_score_correction_bias"] = ("F32", (E,))
    for e in range(E):
        for proj, (n, k) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            q = f"{p}mlp.experts.{e}.{proj}."
            out[q + "weight"] = ("U8", (n, k // 2))
            out[q + "weight_scale"] = ("U8", (n, k // MX_BLOCK))
    return out


def edge_plan(cfg: MimoConfig, *, head: bool, tail: bool, n_fc_taps: int = 0) -> dict:
    """Embedding (head), final norm + untied lm_head (tail), and the DFlash drafter's `fc.weight`
    [H, n_taps * H] (dflash/ checkpoint) when a stage computes fc_partial tap sums."""
    H, out = cfg.hidden, {}
    if head:
        out["model.embed_tokens.weight"] = ("BF16", (cfg.vocab, H))
    if tail:
        out["model.norm.weight"] = ("BF16", (H,))
        out["lm_head.weight"] = ("BF16", (cfg.vocab, H))
    if n_fc_taps:
        out["fc.weight"] = ("BF16", (H, n_fc_taps * H))
    return out
