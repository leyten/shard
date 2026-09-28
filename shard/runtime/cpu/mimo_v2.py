"""MiMo-V2 on the CPU reference backend: canonical weights from the checkpoint bytes, and the
canonical per-row layer math every MiMo backend is tested against (BUILD-PLAN R2).

Promoted from the verified scratch oracle (scratchpad/v2/mimo/mimo_ref.py, 2026-09-28). Measured
there on the REAL Flash layers 0 (global, dense) and 1 (SWA, MoE): fp32 parity vs the vendor's own
`MiMoV2DecoderLayer` max_abs 2.7e-7 / 3e-8, top-8 routing sets equal, bf16 within 0.5 bf16 ulp of
the vendor's bf16 path. The config and tensor plan come from the device-neutral adapter
(shard/runtime/adapters/mimo_v2.py); the checkpoint FORMAT decoders are not duplicated here (MXFP4,
FP8 block scales, the bf16 rounding and the keep-fp32 rule: shard/formats/quant.py; the pre-sharded
fused qkv: shard/formats/mimo_v2.py).

Canonical numerics ("canonical bf16", ARCHITECTURE-v2 §6), the definition:
  * weights: MXFP4 -> bf16 exact; FP8 x fp32 block scale = one fp32 multiply, then bf16 RNE.
  * activations: every op takes bf16 rows, computes in fp32 and rounds its output ONCE to bf16:
    rmsnorm, each linear, RoPE (fp32 cos/sin), the v pre-scale, attention (fp32 scores, softmax incl.
    the sink logit, fp32 PV), silu(g)*u, the MoE combine (fp32, ascending expert id), each residual
    add. The router is fp32 end to end (bf16 inputs, fp32 bias). RMSNorm always normalizes the
    bf16-rounded residual sum, so a stage boundary carrying bf16 rows is invisible.
  These fix WHAT is rounded WHERE, not the order of an fp32 sum; a backend with other kernels has
  its own numerics_id and matches this one to rounding tolerance, not bit for bit.

This implementation's own bytes: every op runs on ONE row ([1, D] tensors, a matrix-vector product
per row, attention per query over exactly its visible keys, copied to one canonical layout first), so
a row's bytes never depend on which rows share a call. That makes THIS backend batch-, chunk- and
partition-invariant (a takeover rebuild is bit-equal to the fault-free run). It is how the CPU gets
invariance (MKL's batched GEMM is not row-invariant: measured), not a requirement on GPU kernels,
which get batch/partition invariance from fixed tiles (R4) and do not promise chunk invariance.
Weights are held as fp32 containers of bf16 values (experts stay packed, see ExpertStore).
act=torch.float32 runs the same code without the bf16 roundings (the tight vendor parity check).
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from ...formats.mimo_v2 import qkv_rechunk
from ...formats.quant import (MX_BLOCK, check_keep_fp32, dequant_fp8_block, dequant_mxfp4,
                              keep_fp32, to_bf16_values)
from ..adapters.mimo_v2 import LayerSpec, MimoConfig, edge_plan, layer_plan, layer_prefix

# safetensors dtype -> the numpy storage a WeightSource hands over (numpy has no bf16 / fp8)
ST_NUMPY = {"BF16": np.uint16, "F32": np.float32, "F8_E4M3": np.uint8, "U8": np.uint8}
_TORCH_VIEW = {"BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn}


class WeightError(ValueError):
    """A checkpoint tensor is missing or has the wrong dtype/shape."""


def read_tensor(src, name: str, st_dtype: str, shape: tuple) -> torch.Tensor:
    """One checkpoint tensor from a WeightSource as a typed torch tensor (bf16 / fp8 / f32 / u8),
    refusing a missing tensor, a declared or stored dtype other than the plan's, a wrong shape and a
    down-cast router bias (quant.check_keep_fp32)."""
    dtype_of = getattr(src, "dtype_of", None)
    try:
        declared = dtype_of(name) if dtype_of is not None else st_dtype
        a = np.asarray(src.read(name))
    except KeyError:
        raise WeightError(f"weight source lacks {name}") from None
    if declared != st_dtype:
        if keep_fp32(name):
            raise WeightError(f"{name} must stay float32 (router selection bias), stored as {declared}")
        raise WeightError(f"{name}: stored as {declared}, the checkpoint format says {st_dtype}")
    if keep_fp32(name) and a.dtype != np.float32:
        raise WeightError(f"{name} must stay float32 (router selection bias), got numpy {a.dtype}")
    if a.dtype != ST_NUMPY[st_dtype] or tuple(a.shape) != tuple(shape):
        raise WeightError(f"{name}: got {a.dtype}{list(a.shape)}, want {st_dtype}{list(shape)}")
    t = torch.from_numpy(np.array(a, copy=True, order="C"))
    t = t.view(_TORCH_VIEW[st_dtype]) if st_dtype in _TORCH_VIEW else t
    try:
        return check_keep_fp32(name, t)
    except ValueError as e:
        raise WeightError(str(e)) from None


def _fp32(t: torch.Tensor) -> torch.Tensor:
    """A private, contiguous fp32 copy (aligned like every other weight: BLAS paths depend on it)."""
    return t.to(torch.float32).contiguous().clone()


# ── canonical ops (one row at a time; see the module docstring) ────────────────────────────────

def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [1, K] -> fp32 [1, N] = W @ x in fp32, one matrix-vector product on a fresh row."""
    return torch.mv(w, x.to(torch.float32).reshape(-1).clone()).unsqueeze(0)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float, act: torch.dtype) -> torch.Tensor:
    xf = x.to(torch.float32)
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w).to(act)


def rope_inv_freq(rope_dim: int, theta: float) -> torch.Tensor:
    """The vendor's inv_freq arithmetic (fp32)."""
    return 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.int64).to(torch.float32) / rope_dim))


def rope_cos_sin(p: int, inv_freq: torch.Tensor):
    """cos/sin [1, rope_dim] for absolute position p, rotate-half layout, kept in fp32."""
    ang = torch.tensor([p], dtype=torch.int64).to(torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat([ang, ang], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos, sin, rope_dim: int, act: torch.dtype) -> torch.Tensor:
    """x [1, heads, d]: rotate the FIRST rope_dim dims of every head, leave the rest untouched."""
    xr = x[..., :rope_dim].to(torch.float32)
    x1, x2 = xr[..., : rope_dim // 2], xr[..., rope_dim // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    out = xr * cos[:, None, :] + rot * sin[:, None, :]
    return torch.cat([out.to(act), x[..., rope_dim:].to(act)], dim=-1)


def attend(q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, sink: torch.Tensor | None,
           scaling: float, act: torch.dtype) -> torch.Tensor:
    """One query row over exactly its visible keys. q [1, n_q, qk]; K [n, n_kv, qk], V [n, n_kv, v]
    in key-position order (any source layout: they are copied to one canonical layout first, so
    every caller hits the same BLAS path); sink fp32 [n_q] = one extra logit per head whose value
    vector is zero. Returns [1, n_q * v] in act."""
    n, n_kv, qk = K.shape
    n_q = q.shape[1]
    grp = n_q // n_kv                                                   # q head h -> kv head h // grp
    kv = K.to(torch.float32).transpose(0, 1).contiguous()              # [n_kv, n, qk]
    vv = V.to(torch.float32).transpose(0, 1).contiguous()              # [n_kv, n, v]
    qt = q[0].to(torch.float32).reshape(n_kv, grp, qk).contiguous()
    sc = torch.bmm(qt, kv.transpose(1, 2)) * scaling                    # [n_kv, grp, n]
    if sink is not None:
        sc = torch.cat([sc, sink.view(n_kv, grp, 1)], dim=-1)
    pr = torch.softmax(sc, dim=-1)[..., :n]
    return torch.bmm(pr, vv).reshape(1, -1).to(act)


def mlp(x: torch.Tensor, wg, wu, wd, act: torch.dtype) -> torch.Tensor:
    g = linear(x, wg).to(act).to(torch.float32)
    u = linear(x, wu).to(act).to(torch.float32)
    return linear((F.silu(g) * u).to(act), wd).to(act)


# ── weights of one layer ──────────────────────────────────────────────────────────────────────

class ExpertStore:
    """The routed experts of one MoE layer, resident PACKED (MXFP4 bytes + E8M0 scales, what a GPU
    keeps too) and dequantized exactly on first use; a small LRU holds the bf16 matrices."""

    def __init__(self, packed: dict, max_cached: int = 64):
        self.packed = packed                  # e -> {proj: (packed u8, scale u8)}
        self.cache: OrderedDict[int, tuple] = OrderedDict()
        self.max_cached = max_cached

    def nbytes(self) -> int:
        return sum(p.numel() + s.numel() for d in self.packed.values() for p, s in d.values())

    def get(self, e: int):
        """fp32 (gate [I,H], up [I,H], down [H,I]), every value exactly bf16."""
        if e in self.cache:
            self.cache.move_to_end(e)
            return tuple(t.to(torch.float32) for t in self.cache[e])
        mats = []
        for proj in ("gate_proj", "up_proj", "down_proj"):
            w = dequant_mxfp4(*self.packed[e][proj])
            wb = w.to(torch.bfloat16)
            if not torch.equal(wb.to(torch.float32), w):         # quant.dequant_mxfp4's promise
                raise WeightError(f"expert {e} {proj}: an MXFP4 value is not bf16-exact")
            mats.append(wb)
        self.cache[e] = tuple(mats)
        while len(self.cache) > self.max_cached:
            self.cache.popitem(last=False)
        return tuple(t.to(torch.float32).contiguous() for t in mats)


class MimoLayer:
    """Canonical weights + math of decoder layer `spec.idx`."""

    def __init__(self, cfg: MimoConfig, spec: LayerSpec, w: dict, experts: ExpertStore | None = None):
        self.cfg, self.spec, self.w, self.experts = cfg, spec, w, experts
        self.scaling = spec.qk_dim ** -0.5
        self.inv_freq = rope_inv_freq(spec.rope_dim, spec.rope_theta)

    @classmethod
    def load(cls, cfg: MimoConfig, i: int, src, expert_cache: int = 64) -> "MimoLayer":
        s, plan = cfg.layers[i], layer_plan(cfg, i)
        t = {n: read_tensor(src, n, *ds) for n, ds in plan.items() if ".experts." not in n}
        p = layer_prefix(i)
        try:
            wq, wk, wv = qkv_rechunk(t[p + "self_attn.qkv_proj.weight"], t[p + "self_attn.qkv_proj.weight_scale_inv"],
                                     s.n_q, s.n_kv, s.qk_dim, s.v_dim, ckpt_tp=cfg.ckpt_tp, block=cfg.fp8_block)
            w = {"wq": _fp32(to_bf16_values(wq)), "wk": _fp32(to_bf16_values(wk)), "wv": _fp32(to_bf16_values(wv))}
            del wq, wk, wv
            if not s.is_moe:
                for n in ("gate_proj", "up_proj", "down_proj"):
                    w[n] = _fp32(to_bf16_values(dequant_fp8_block(
                        t[f"{p}mlp.{n}.weight"], t[f"{p}mlp.{n}.weight_scale_inv"], block=cfg.fp8_block)))
        except ValueError as e:
            raise WeightError(f"layer {i}: {e}") from None
        w["ln1"] = _fp32(t[p + "input_layernorm.weight"])
        w["ln2"] = _fp32(t[p + "post_attention_layernorm.weight"])
        w["wo"] = _fp32(t[p + "self_attn.o_proj.weight"])
        w["sink"] = _fp32(t[p + "self_attn.attention_sink_bias"]) if s.has_sink else None
        experts = None
        if s.is_moe:
            w["router"] = _fp32(t[p + "mlp.gate.weight"])
            w["router_bias"] = t[p + "mlp.gate.e_score_correction_bias"].contiguous().clone()   # fp32, never rounded
            packed = {}
            for e in range(cfg.n_experts):
                q = f"{p}mlp.experts.{e}."
                packed[e] = {proj: (read_tensor(src, q + proj + ".weight", *plan[q + proj + ".weight"]),
                                    read_tensor(src, q + proj + ".weight_scale", *plan[q + proj + ".weight_scale"]))
                             for proj in ("gate_proj", "up_proj", "down_proj")}
            experts = ExpertStore(packed, expert_cache)
        return cls(cfg, s, w, experts)

    @staticmethod
    def resident_bytes(cfg: MimoConfig, i: int) -> int:
        """Bytes this layer holds on the CPU backend: fp32 containers for the dense tensors, experts
        packed (the LRU of dequantized experts is workspace, reported via load_peak_extra)."""
        s, H = cfg.layers[i], cfg.hidden
        n = 2 * H + (s.n_q * s.qk_dim + s.n_kv * s.qk_dim + s.n_kv * s.v_dim) * H + H * s.n_q * s.v_dim
        n += s.n_q if s.has_sink else 0
        if not s.is_moe:
            return 4 * (n + 3 * cfg.intermediate * H)
        I, E = cfg.moe_intermediate, cfg.n_experts
        packed = E * (2 * I * (H // 2 + H // MX_BLOCK) + H * (I // 2 + I // MX_BLOCK))
        return 4 * (n + E * H + E) + packed

    # The layer, split where a backend's KV state sits between the halves. Each call takes a
    # segment's rows and runs the canonical ops one row at a time; grouping rows only lets an
    # expert be dequantized once per call (a row's MoE sum still runs in ascending expert id).
    def _rope(self, p: int):
        key = (self.spec.rope_dim, self.spec.rope_theta, p)
        cs = _ROPE_CACHE.get(key)
        if cs is None:
            if len(_ROPE_CACHE) >= _ROPE_CACHE_MAX:
                _ROPE_CACHE.clear()
            cs = _ROPE_CACHE[key] = rope_cos_sin(p, self.inv_freq)
        return cs

    def project_qkv(self, h: torch.Tensor, start: int, act: torch.dtype, trace: dict | None = None):
        """h [n, H] layer-input rows at positions start.. -> q [n, n_q, qk], k [n, n_kv, qk]
        (post-RoPE: what is cached), v [n, n_kv, v] (post v-scale: what is cached)."""
        s, w = self.spec, self.w
        xs, qs, ks, vs = [], [], [], []
        for r in range(h.shape[0]):
            x = rmsnorm(h[r:r + 1], w["ln1"], self.cfg.eps, act)
            q = linear(x, w["wq"]).to(act).view(1, s.n_q, s.qk_dim)
            k = linear(x, w["wk"]).to(act).view(1, s.n_kv, s.qk_dim)
            v = linear(x, w["wv"]).to(act).view(1, s.n_kv, s.v_dim)
            if self.cfg.v_scale is not None:
                v = (v.to(torch.float32) * self.cfg.v_scale).to(act)
            cos, sin = self._rope(start + r)
            xs.append(x)
            qs.append(apply_rope(q, cos, sin, s.rope_dim, act))
            ks.append(apply_rope(k, cos, sin, s.rope_dim, act))
            vs.append(v)
        q, k, v = torch.cat(qs), torch.cat(ks), torch.cat(vs)
        if trace is not None:
            trace.update(attn_in=torch.cat(xs), q=q, k=k, v=v)
        return q, k, v

    def attend(self, q, K, V, act):
        """One query row q [1, n_q, qk] over its visible keys K/V [n, n_kv, d] -> [1, n_q * v]."""
        return attend(q, K, V, self.w["sink"], self.scaling, act)

    def route(self, x: torch.Tensor):
        """fp32 router on one row: (logits [1, E], top-k ids [1, k] sorted by biased score,
        weights [1, k] from the un-biased scores)."""
        logits = linear(x, self.w["router"])
        scores = torch.sigmoid(logits)
        choice = scores + self.w["router_bias"]                         # the bias only picks experts
        idx = torch.topk(choice, self.cfg.top_k, dim=-1, sorted=True).indices
        wt = scores.gather(1, idx)
        if self.cfg.norm_topk:
            wt = wt / (wt.sum(-1, keepdim=True) + 1e-20)
        return logits, idx, wt * self.cfg.routed_scale

    def ffn(self, xs: list, act: torch.dtype, trace: dict | None = None) -> list:
        """Rows [1, H] -> FFN output rows [1, H]."""
        w = self.w
        if not self.spec.is_moe:
            return [mlp(x, w["gate_proj"], w["up_proj"], w["down_proj"], act) for x in xs]
        routes = [self.route(x) for x in xs]
        ids = [idx[0].tolist() for _, idx, _ in routes]
        acc = [torch.zeros(1, x.shape[1], dtype=torch.float32) for x in xs]
        for e in sorted(set().union(*ids)):                             # ascending expert id
            W = self.experts.get(e)
            for r, row_ids in enumerate(ids):
                if e in row_ids:
                    y = mlp(xs[r], *W, act)
                    acc[r] = acc[r] + y.to(torch.float32) * routes[r][2][0, row_ids.index(e)]
        if trace is not None:
            trace.update(router_logits=torch.cat([lg for lg, _, _ in routes]),
                         topk_idx=torch.cat([idx for _, idx, _ in routes]),
                         topk_w=torch.cat([wt for _, _, wt in routes]))
        return [a.to(act) for a in acc]

    def finish(self, h: torch.Tensor, attn: torch.Tensor, act: torch.dtype, trace: dict | None = None):
        """h [n, H] layer inputs, attn [n, n_q * v] -> the layer output rows [n, H]."""
        os_, h1s, x2s = [], [], []
        for r in range(h.shape[0]):
            o = linear(attn[r:r + 1], self.w["wo"]).to(act)
            h1 = (h[r:r + 1].to(torch.float32) + o.to(torch.float32)).to(act)
            os_.append(o)
            h1s.append(h1)
            x2s.append(rmsnorm(h1, self.w["ln2"], self.cfg.eps, act))
        ms = self.ffn(x2s, act, trace)
        out = torch.cat([(h1.to(torch.float32) + m.to(torch.float32)).to(act) for h1, m in zip(h1s, ms)])
        if trace is not None:
            trace.update(attn_core=attn, attn_out=torch.cat(os_), resid_mid=torch.cat(h1s),
                         ffn_in=torch.cat(x2s), ffn_out=torch.cat(ms), out=out)
        return out


_ROPE_CACHE: dict = {}          # (rope_dim, theta, position) -> (cos, sin); pure function of the key
_ROPE_CACHE_MAX = 1 << 17


class MimoEdges:
    """Head/tail tensors of a stage: embedding (kept bf16, rows gathered), final norm, lm_head,
    and the fc_partial column slices of the DFlash `fc.weight` for the taps the stage holds."""

    def __init__(self, cfg: MimoConfig, embed=None, norm=None, lm_head=None, fc: dict | None = None):
        self.cfg, self.embed_w, self.norm, self.lm_head, self.fc = cfg, embed, norm, lm_head, fc or {}

    @classmethod
    def load(cls, cfg: MimoConfig, src, *, head: bool, tail: bool, fc_taps: tuple = (),
             fc_order: tuple = ()) -> "MimoEdges":
        plan = edge_plan(cfg, head=head, tail=tail, n_fc_taps=len(fc_order) if fc_taps else 0)
        t = {n: read_tensor(src, n, *ds) for n, ds in plan.items()}
        H, fc = cfg.hidden, {}
        if fc_taps:
            W = t["fc.weight"].to(torch.float32)
            for tap in fc_taps:
                j = fc_order.index(tap)
                fc[tap] = W[:, j * H:(j + 1) * H].contiguous().clone()
        return cls(cfg, embed=t.get("model.embed_tokens.weight"),
                   norm=_fp32(t["model.norm.weight"]) if tail else None,
                   lm_head=_fp32(t["lm_head.weight"]) if tail else None, fc=fc)

    @staticmethod
    def resident_bytes(cfg: MimoConfig, *, head: bool, tail: bool, n_fc_held: int = 0) -> int:
        H, V = cfg.hidden, cfg.vocab
        return (2 * V * H if head else 0) + (4 * (H + V * H) if tail else 0) + 4 * n_fc_held * H * H

    def embed(self, ids: torch.Tensor, act: torch.dtype) -> torch.Tensor:
        return self.embed_w[ids.to(torch.int64)].to(act)

    def logits(self, h: torch.Tensor, act: torch.dtype) -> torch.Tensor:
        """h [1, H] pre-final-norm -> fp32 logits [1, V]."""
        return linear(rmsnorm(h, self.norm, self.cfg.eps, act), self.lm_head)

    def fc_add(self, acc: torch.Tensor, tap: int, row: torch.Tensor) -> torch.Tensor:
        """fc_partial: acc [1, H] fp32 + W_fc[:, slice(tap)] @ row (fp32)."""
        return acc + linear(row, self.fc[tap])


# ── the promoted oracle ───────────────────────────────────────────────────────────────────────

class KVCache:
    """K (post-RoPE) and V (post-v-scale) of ONE sequence, EVERY position kept (the window is
    applied when reading), so truncate() is exact. The oracle's storage, not a runtime's."""

    def __init__(self, specs: dict, dtype: torch.dtype):
        self.k = {i: torch.empty(0, s.n_kv, s.qk_dim, dtype=dtype) for i, s in specs.items()}
        self.v = {i: torch.empty(0, s.n_kv, s.v_dim, dtype=dtype) for i, s in specs.items()}

    def length(self, layer: int | None = None) -> int:
        if layer is not None:
            return self.k[layer].shape[0]
        lens = {k.shape[0] for k in self.k.values()}
        if len(lens) != 1:
            raise ValueError(f"layers disagree on length: {lens}")
        return lens.pop()

    def truncate(self, pos: int) -> None:
        for i in self.k:
            if not 0 <= pos <= self.k[i].shape[0]:
                raise ValueError(f"truncate to {pos} outside [0, {self.k[i].shape[0]}]")
            self.k[i], self.v[i] = self.k[i][:pos].clone(), self.v[i][:pos].clone()


class MimoOracle:
    """The CPU oracle (promoted scratch mimo_ref.MiMoRef): a set of decoder layers (+ embedding /
    final norm) of ONE sequence, dense KV, optional per-op traces. Golden vectors are defined by it."""

    def __init__(self, cfg: MimoConfig, layers: dict, dtype: torch.dtype = torch.bfloat16,
                 edges: MimoEdges | None = None):
        self.cfg, self.layers, self.dtype, self.edges = cfg, layers, dtype, edges
        self.specs = {i: cfg.layers[i] for i in layers}

    @classmethod
    def from_source(cls, config: dict, src, layers, dtype: torch.dtype = torch.bfloat16, *,
                    head: bool = False, tail: bool = False, expert_cache: int = 64) -> "MimoOracle":
        cfg = MimoConfig.from_hf(config)
        edges = MimoEdges.load(cfg, src, head=head, tail=tail) if head or tail else None
        return cls(cfg, {i: MimoLayer.load(cfg, i, src, expert_cache) for i in layers}, dtype, edges)

    def new_cache(self) -> KVCache:
        return KVCache(self.specs, self.dtype)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.edges.embed(ids, self.dtype)

    def final_norm(self, h: torch.Tensor) -> torch.Tensor:
        return torch.cat([rmsnorm(h[t:t + 1], self.edges.norm, self.cfg.eps, h.dtype) for t in range(h.shape[0])])

    def forward(self, h: torch.Tensor, start_pos: int, cache: KVCache, layers=None,
                traces: dict | None = None) -> torch.Tensor:
        """h [T, H] (dtype == self.dtype) at positions start_pos.. through `layers` in order. Every
        layer's cache length must equal start_pos (truncate first to roll back)."""
        if h.dtype != self.dtype or h.dim() != 2:
            raise ValueError(f"h must be [T, H] {self.dtype}")
        act = self.dtype
        for i in (sorted(self.layers) if layers is None else layers):
            L, s = self.layers[i], self.specs[i]
            if cache.length(i) != start_pos:
                raise ValueError(f"layer {i}: cache length {cache.length(i)} != start_pos {start_pos}")
            tr = {} if traces is not None else None
            q, k, v = L.project_qkv(h, start_pos, act, tr)
            cache.k[i], cache.v[i] = torch.cat([cache.k[i], k]), torch.cat([cache.v[i], v])
            attn = []
            for t in range(h.shape[0]):
                p = start_pos + t
                lo = 0 if s.window is None else max(0, p - s.window + 1)
                attn.append(L.attend(q[t:t + 1], cache.k[i][lo:p + 1], cache.v[i][lo:p + 1], act))
            h = L.finish(h, torch.cat(attn), act, tr)
            if traces is not None:
                traces[i] = tr
        return h
