"""A tiny MiMo-V2-shaped model for CI, plus the conformance reference of the CPU MiMo backend.

Library for tests (not collected). Three parts:

1. `tiny_config` / `TinyCheckpoint`: a random model with the REAL MiMo-V2 structure at tiny dims:
   global + SWA layers (layer 0 dense, the rest MoE, Flash's G SSSS G ... pattern), sinks on SWA
   only, partial RoPE (int(48 * 0.334) = 16 of 48 dims, i.e. Flash's 64/192), QK 48 / V 32
   (asymmetric like 192/128), different KV heads for global (2) and SWA (4) layers, the v pre-scale,
   a sigmoid MoE with an fp32 correction bias around 2.0 (where a bf16 ulp is 0.0156). Tensors are
   written in the RELEASED checkpoint format: FP8-e4m3 qkv pre-sharded for ckpt_tp =
   num_key_value_heads chunks and interleaved [Q_0|K_0|V_0|Q_1|...] with a PER-CHUNK scale grid
   (global chunks of 272 rows -> 6 scale rows, not the plain grid's 5: Flash's 108-vs-106 case),
   FP8 dense MLP, MXFP4 experts (packed nibbles + E8M0), BF16 o_proj / norms / router / sinks,
   F32 router bias. `logical` holds the same weights decoded INDEPENDENTLY of shard/formats (own
   e4m3 and e2m1 formulas from the chosen codes, own bf16 rounding, own de-interleave), in the
   natural layout the vendor module consumes.
2. `MimoReference`: an independent state model driven through the same calls as a StageRuntime
   (dict of K/V rows per layer and position, no ring, no reclamation, no validation; rows below a
   rebuild start are simply absent). It shares the adapter's canonical per-row math with CpuRuntime
   (that math is checked against the vendor module separately), so what it checks is the STATE:
   ring, rollback, frontier, mirroring, rebuild.
3. `MimoFactory`: the RuntimeFactory tests/runtime_conformance.py drives.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from shard.runtime.adapters.mimo_v2 import MimoConfig
from shard.runtime.api import KVBlob, Rows, SegOut, StageSpec, StepOut, kv_layout_id, rebuild_start
from shard.runtime.cpu.mimo_v2 import MimoEdges, MimoLayer
from shard.runtime.cpu.runtime import ACT, CpuRuntime, draw, from_rows, to_rows

FLASH_LIKE = "GSSSSGSS"


def tiny_config(*, window: int = 8, pattern: str = FLASH_LIKE, experts: int = 8, top_k: int = 2) -> dict:
    L = len(pattern)
    return {
        "architectures": ["MiMoV2ForCausalLM"], "model_type": "mimo_v2",
        "hidden_size": 256, "num_hidden_layers": L, "vocab_size": 96, "tie_word_embeddings": False,
        "hybrid_layer_pattern": [0 if c == "G" else 1 for c in pattern],
        "moe_layer_freq": [0] + [1] * (L - 1),
        "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 48, "v_head_dim": 32,
        "swa_num_attention_heads": 8, "swa_num_key_value_heads": 4, "swa_head_dim": 48, "swa_v_head_dim": 32,
        "partial_rotary_factor": 0.334, "rope_theta": 10000000.0, "swa_rope_theta": 10000.0,
        "rope_parameters": {"partial_rotary_factor": 0.334, "rope_theta": 10000000.0, "rope_type": "default"},
        "sliding_window": window, "sliding_window_size": window,
        "add_swa_attention_sink_bias": True, "add_full_attention_sink_bias": False,
        "attention_value_scale": 0.707, "layernorm_epsilon": 1e-6, "hidden_act": "silu",
        "attention_projection_layout": "fused_qkv", "attention_bias": False,
        "intermediate_size": 256, "moe_intermediate_size": 64, "n_routed_experts": experts,
        "num_experts_per_tok": top_k, "norm_topk_prob": True, "routed_scaling_factor": None,
        "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_group": 1, "topk_group": 1,
        "n_shared_experts": None, "max_position_embeddings": 4096,
        "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128],
                                "activation_scheme": "dynamic", "store_dtype": "mxfp4", "mxfp4_block_size": 32,
                                "ignored_layers": [f"model.layers.{i}.self_attn.o_proj" for i in range(L)]},
    }


# ── independent decoders (the checkpoint's formats, written from the spec, not from shard) ────

def _e4m3(code: int) -> float:
    s, e, m = code >> 7, (code >> 3) & 0xF, code & 7
    if e == 0xF and m == 7:
        return float("nan")
    mag = m / 8 * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
    return -mag if s else mag


def _e2m1(code: int) -> float:
    s, e, m = code >> 3, (code >> 1) & 3, code & 1
    mag = m * 0.5 if e == 0 else 2.0 ** (e - 1) * (1 + m / 2)
    return -mag if s else mag


E4M3 = np.array([_e4m3(c) for c in range(256)])
E2M1 = np.array([_e2m1(c) for c in range(16)])


def bf16_rne(x: np.ndarray) -> np.ndarray:
    """fp32 -> nearest-even bf16 value, as fp32 (finite inputs)."""
    b = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    b = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16
    return b.astype(np.uint32).view(np.float32)


def _bf16_bits(x: np.ndarray) -> np.ndarray:
    return (bf16_rne(x).view(np.uint32) >> 16).astype(np.uint16)


class TinyCheckpoint:
    """names/read/dtype_of over checkpoint-format tensors (+ `logical` decoded weights)."""

    def __init__(self, config: dict, seed: int = 0, fc_taps: tuple = ()):
        self.config = config
        self.tensors, self.dtypes, self.logical = {}, {}, {}
        rng = np.random.default_rng(seed)
        c, H = config, config["hidden_size"]
        tp = c["num_key_value_heads"]
        for i, kind in enumerate(c["hybrid_layer_pattern"]):
            p = f"model.layers.{i}."
            pre = "swa_" if kind == 1 else ""
            n_q, n_kv = c[f"{pre}num_attention_heads"], c[f"{pre}num_key_value_heads"]
            qk, v = c[f"{pre}head_dim"], c[f"{pre}v_head_dim"]
            lg = self.logical[i] = {}
            lg["ln1"] = self._bf16(p + "input_layernorm.weight", 1 + 0.1 * rng.standard_normal(H))
            lg["ln2"] = self._bf16(p + "post_attention_layernorm.weight", 1 + 0.1 * rng.standard_normal(H))
            lg["qkv"] = self._qkv(p + "self_attn.qkv_proj.weight", rng, n_q, n_kv, qk, v, tp, H)
            lg["o_proj"] = self._bf16(p + "self_attn.o_proj.weight", 0.07 * rng.standard_normal((H, n_q * v)))
            if kind == 1 and c["add_swa_attention_sink_bias"]:
                lg["sink"] = self._bf16(p + "self_attn.attention_sink_bias", 0.5 + rng.standard_normal(n_q))
            if not c["moe_layer_freq"][i]:
                I = c["intermediate_size"]
                for n, shape in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
                    lg[n] = self._fp8_plain(f"{p}mlp.{n}.weight", rng, shape)
                continue
            E, I = c["n_routed_experts"], c["moe_intermediate_size"]
            lg["router"] = self._bf16(p + "mlp.gate.weight", 0.05 * rng.standard_normal((E, H)))
            bias = (2.0 + 0.02 * rng.standard_normal(E)).astype(np.float32)
            self._put(p + "mlp.gate.e_score_correction_bias", bias, "F32")
            lg["router_bias"] = bias.copy()
            lg["experts"] = {}
            for e in range(E):
                lg["experts"][e] = tuple(self._mxfp4(f"{p}mlp.experts.{e}.{n}.weight", rng, shape)
                                         for n, shape in (("gate_proj", (I, H)), ("up_proj", (I, H)),
                                                          ("down_proj", (H, I))))
        self.logical["embed"] = self._bf16("model.embed_tokens.weight", rng.standard_normal((c["vocab_size"], H)))
        self.logical["norm"] = self._bf16("model.norm.weight", 1 + 0.1 * rng.standard_normal(H))
        self.logical["lm_head"] = self._bf16("lm_head.weight", 0.06 * rng.standard_normal((c["vocab_size"], H)))
        if fc_taps:
            self.logical["fc"] = self._bf16("fc.weight", 0.06 * rng.standard_normal((H, len(fc_taps) * H)))
        self.artifact_cid = f"tiny-mimo-{seed}-{len(c['hybrid_layer_pattern'])}-{c['sliding_window']}"

    # writers: store the checkpoint bytes, return the logical (decoded) fp32 values
    def _put(self, name, arr, dtype):
        self.tensors[name], self.dtypes[name] = np.ascontiguousarray(arr), dtype

    def _bf16(self, name, x):
        self._put(name, _bf16_bits(np.asarray(x, dtype=np.float32)), "BF16")
        return bf16_rne(np.asarray(x, dtype=np.float32))

    @staticmethod
    def _codes(rng, shape):
        x = torch.from_numpy(np.clip(100 * rng.standard_normal(shape), -448, 448).astype(np.float32))
        return x.to(torch.float8_e4m3fn).view(torch.uint8).numpy()

    def _fp8_plain(self, name, rng, shape):
        n, k = shape
        codes = self._codes(rng, shape)
        s = (rng.uniform(0.5, 1.5, (math.ceil(n / 128), math.ceil(k / 128))) * 2.0 ** -11).astype(np.float32)
        self._put(name, codes, "F8_E4M3")
        self._put(name + "_scale_inv", s, "F32")
        full = s.astype(np.float64)[np.arange(n)[:, None] // 128, np.arange(k)[None, :] // 128]
        return bf16_rne((E4M3[codes] * full).astype(np.float32))

    def _qkv(self, name, rng, n_q, n_kv, qk, v, tp, H):
        """Interleaved per-chunk FP8 qkv; returns the natural [Q; K; V] logical weight."""
        rpc = (n_q * qk + n_kv * qk + n_kv * v) // tp
        cs = math.ceil(rpc / 128)
        codes = self._codes(rng, (tp * rpc, H))
        s = (rng.uniform(0.5, 1.5, (tp * cs, math.ceil(H / 128))) * 2.0 ** -12).astype(np.float32)
        self._put(name, codes, "F8_E4M3")
        self._put(name + "_scale_inv", s, "F32")
        deq = np.empty((tp * rpc, H), dtype=np.float32)
        for ch in range(tp):
            for r in range(rpc):
                deq[ch * rpc + r] = (E4M3[codes[ch * rpc + r]]
                                     * np.repeat(s[ch * cs + r // 128].astype(np.float64), 128)[:H]).astype(np.float32)
        deq = bf16_rne(deq)
        q_pc, k_pc = n_q // tp * qk, n_kv // tp * qk
        parts = ([deq[ch * rpc: ch * rpc + q_pc] for ch in range(tp)]
                 + [deq[ch * rpc + q_pc: ch * rpc + q_pc + k_pc] for ch in range(tp)]
                 + [deq[ch * rpc + q_pc + k_pc: (ch + 1) * rpc] for ch in range(tp)])
        return np.concatenate(parts)

    def _mxfp4(self, name, rng, shape):
        n, k = shape
        codes = rng.integers(0, 16, shape, dtype=np.uint8)
        exps = rng.integers(121, 123, (n, k // 32), dtype=np.uint8)
        self._put(name, (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8), "U8")
        self._put(name[: -len("weight")] + "weight_scale", exps, "U8")
        return (E2M1[codes] * np.repeat(2.0 ** (exps.astype(np.float64) - 127), 32, axis=1)).astype(np.float32)

    # WeightSource
    def names(self):
        return sorted(self.tensors)

    def read(self, name):
        return self.tensors[name].copy()

    def dtype_of(self, name):
        return self.dtypes[name]

    def without(self, name) -> "TinyCheckpoint":
        cp = object.__new__(TinyCheckpoint)
        cp.__dict__.update(self.__dict__)
        cp.tensors = {k: v for k, v in self.tensors.items() if k != name}
        cp.dtypes = {k: v for k, v in self.dtypes.items() if k != name}
        return cp

    def replaced(self, name, arr, dtype) -> "TinyCheckpoint":
        cp = self.without(name)
        cp.tensors, cp.dtypes = dict(cp.tensors), dict(cp.dtypes)
        cp.tensors[name], cp.dtypes[name] = arr, dtype
        return cp


# ── the reference (independent state, shared canonical math) ───────────────────────────────────

class MimoReference:
    """Driven through open/close/length/commit/truncate/step/export_state/import_state/
    rebuild_from/embed/project like a StageRuntime. Trusts its caller: an oracle, not a runtime."""

    def __init__(self, config: dict, src, spec: StageSpec, layers: dict | None = None):
        self.spec = spec
        self.cfg = cfg = MimoConfig.from_hf(config)
        self.window = cfg.window
        span = range(spec.lo, spec.hi)
        self.kinds = cfg.kinds[spec.lo:spec.hi]
        self.layers = layers or {i: MimoLayer.load(cfg, i, src) for i in span}
        order = tuple(sorted(set(spec.taps)))
        self.taps = tuple(t for t in order if spec.lo <= t < spec.hi)
        fc = spec.tap_mode == "fc_partial" and bool(self.taps)
        self.edges = MimoEdges.load(cfg, src, head=spec.is_head, tail=spec.is_tail,
                                    fc_taps=self.taps if fc else (), fc_order=order)
        self.glob = tuple(i for i in span if cfg.layers[i].kind == "global")
        geo = {(cfg.layers[i].n_kv, cfg.layers[i].qk_dim, cfg.layers[i].v_dim) for i in self.glob}
        self.layout = kv_layout_id(f"{spec.model}#{cfg.fingerprint()}", self.glob, "bf16", geo.pop() if geo else (0, 0, 0))
        self.s = {}

    def open(self, sid, max_pos, spec_window):
        self.s[sid] = {"kv": {i: {} for i in range(self.spec.lo, self.spec.hi)},
                       "length": 0, "committed": 0, "upto": 0, "rb": None}

    def close(self, sid):
        self.s.pop(sid, None)

    def length(self, sid):
        return self.s[sid]["length"]

    def commit(self, sid, pos):
        self.s[sid]["committed"] = max(self.s[sid]["committed"], pos)

    def truncate(self, sid, pos):
        self._cut(self.s[sid], pos)

    @staticmethod
    def _cut(s, pos):
        for d in s["kv"].values():
            for q in [q for q in d if q >= pos]:
                del d[q]
        s["length"] = pos

    def export_state(self, sid, lo_pos, hi_pos, kinds=("global",)):
        kv, n = self.s[sid]["kv"], hi_pos - lo_pos
        rows = tuple(to_rows(torch.cat([torch.cat([kv[i][q][0].reshape(1, -1), kv[i][q][1].reshape(1, -1)], 1)
                                        for q in range(lo_pos, hi_pos)]) if n else
                             torch.zeros(0, self._kvw(i), dtype=ACT)) for i in self.glob)
        return KVBlob(lo_pos, hi_pos, self.glob, rows, self.layout, "mimo-reference")

    def _kvw(self, i):
        s = self.cfg.layers[i]
        return s.n_kv * (s.qk_dim + s.v_dim)

    def import_state(self, sid, blob, upto):
        s = self.s[sid]
        for i, r in zip(blob.layers, blob.rows):
            sp = self.cfg.layers[i]
            t = from_rows(r)
            for q in range(blob.lo_pos, upto):
                row = t[q - blob.lo_pos]
                s["kv"][i][q] = (row[: sp.n_kv * sp.qk_dim].reshape(1, sp.n_kv, sp.qk_dim),
                                 row[sp.n_kv * sp.qk_dim:].reshape(1, sp.n_kv, sp.v_dim))
        s["upto"] = upto

    def rebuild_from(self, sid, F, exact_from=None, floor=None):
        s = self.s[sid]
        floor = F if floor is None else floor
        a = rebuild_start(self.kinds, self.window, F, s["upto"], floor if exact_from is None else min(exact_from, floor))
        s["length"], s["committed"] = a, max(s["committed"], floor)
        s["rb"] = F if a < F else None
        return a

    def embed(self, ids):
        return to_rows(self.edges.embed(torch.from_numpy(ids.data.astype(np.int64)), ACT))

    def project(self, hidden, *, tokens=False, start_pos=0, sampling=None):
        x = from_rows(hidden)
        lg = [self.edges.logits(x[r:r + 1], ACT)[0].numpy() for r in range(x.shape[0])]
        if tokens:
            return Rows("i32", np.array([draw(r, sampling, start_pos + i) for i, r in enumerate(lg)], dtype=np.int32))
        return Rows("f32", np.stack(lg).astype(np.float32))

    def step(self, x):
        return StepOut(tuple(self._seg(seg) for seg in x.segs))

    def _seg(self, seg):
        s = self.s[seg.sid]
        start, n = seg.start_pos, seg.n
        if start < s["length"]:
            self._cut(s, start)
        rb = s["rb"]
        if self.spec.is_head:
            cur = self.edges.embed(torch.from_numpy(seg.x.data.astype(np.int64)), ACT)
        else:
            cur = from_rows(seg.x)
        fc = self.spec.tap_mode == "fc_partial" and seg.taps
        acc = None
        if fc:
            acc = from_rows(seg.tap_in).to(torch.float32) if seg.tap_in is not None \
                else torch.zeros(n, self.cfg.hidden, dtype=torch.float32)
        taps = {}
        for i in range(self.spec.lo, self.spec.hi):
            L, sp, kv = self.layers[i], self.cfg.layers[i], s["kv"][i]
            glob = sp.kind == "global"
            q, k, v = L.project_qkv(cur, start, ACT)
            attn = []
            for r in range(n):
                p = start + r
                if not (glob and rb is not None and p < s["upto"]):
                    kv[p] = (k[r:r + 1], v[r:r + 1])
                present = [j for j in range(0 if glob else max(0, p - self.window + 1), p + 1) if j in kv]
                K = torch.cat([kv[j][0] for j in present])
                V = torch.cat([kv[j][1] for j in present])
                attn.append(L.attend(q[r:r + 1], K, V, ACT))
            cur = L.finish(cur, torch.cat(attn), ACT)
            if seg.taps and i in self.taps:
                if fc:
                    acc = torch.cat([self.edges.fc_add(acc[r:r + 1], i, cur[r:r + 1]) for r in range(n)])
                else:
                    taps[i] = to_rows(cur)
        s["length"] = start + n
        if rb is not None and s["length"] >= rb:
            s["rb"] = None
        logits = tokens = None
        if seg.logits != "none" or seg.tokens != "none":
            rows = {"all": range(n), "last": range(n - 1, n)}
            if seg.logits != "none":
                logits = Rows("f32", np.stack([self.edges.logits(cur[r:r + 1], ACT)[0].numpy()
                                               for r in rows[seg.logits]]).astype(np.float32))
            if seg.tokens != "none":
                tokens = Rows("i32", np.array([draw(self.edges.logits(cur[r:r + 1], ACT)[0].numpy(), seg.sampling, start + r)
                                               for r in rows[seg.tokens]], dtype=np.int32))
        return SegOut(seg.sid, to_rows(cur), logits, taps, None if acc is None else to_rows(acc.to(ACT)), tokens)


# ── the conformance factory ────────────────────────────────────────────────────────────────────

def bf16_values(r: Rows) -> np.ndarray:
    return (r.data.astype(np.uint32) << np.uint32(16)).view(np.float32)


class MimoFactory:
    """RuntimeFactory (tests/runtime_conformance.py) for CpuRuntime on the tiny MiMo model."""
    numerics_mode = "invariant"
    chunk_invariant = True

    def __init__(self, window: int = 8, W: int = 4, pattern: str = FLASH_LIKE, taps: tuple = (0, 3, 7),
                 max_ctx: int = 512, seed: int = 0):
        self.config = tiny_config(window=window, pattern=pattern)
        self.window, self.W, self.max_ctx, self.taps = window, W, max_ctx, taps
        self.layer_kinds = MimoConfig.from_hf(self.config).kinds
        self.ckpt = TinyCheckpoint(self.config, seed=seed, fc_taps=taps)
        self.name = f"CpuRuntime[mimo-tiny,{pattern},w{window}W{W}]"
        self.H, self.V = self.config["hidden_size"], self.config["vocab_size"]

    def _spec(self, lo, hi, is_head, is_tail, tap_mode="raw", spec_window=None, max_sessions=8,
              max_ctx=None, kv_budget_bytes=0):
        return StageSpec("mimo-tiny", lo, hi, is_head=is_head, is_tail=is_tail, taps=self.taps,
                         tap_mode=tap_mode, max_sessions=max_sessions, max_ctx=max_ctx or self.max_ctx,
                         spec_window=max(spec_window or 64, self.W), kv_budget_bytes=kv_budget_bytes)

    def make_unloaded(self, lo, hi, *, is_head=False, is_tail=False, **kw):
        return CpuRuntime(self._spec(lo, hi, is_head, is_tail, **kw), self.config), self.ckpt

    def make(self, lo, hi, *, is_head=False, is_tail=False, **kw):
        rt, src = self.make_unloaded(lo, hi, is_head=is_head, is_tail=is_tail, **kw)
        rt.load(src)
        return rt

    def reference(self, lo, hi, *, is_head=False, is_tail=False, tap_mode="raw"):
        return MimoReference(self.config, self.ckpt, self._spec(lo, hi, is_head, is_tail, tap_mode=tap_mode))

    def kv_bytes_per_pos(self, lo, hi):
        g = s = 0
        for i in range(lo, hi):
            pre = "swa_" if self.layer_kinds[i] == "swa" else ""
            b = 2 * self.config[f"{pre}num_key_value_heads"] * (self.config[f"{pre}head_dim"]
                                                                 + self.config[f"{pre}v_head_dim"])
            g, s = (g + b, s) if self.layer_kinds[i] == "global" else (g, s + b)
        return g, s

    def hidden(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        return to_rows(torch.randn(n, self.H, generator=g).to(ACT))

    def ids(self, n, seed):
        return Rows("i32", np.random.default_rng(seed).integers(0, self.V, n).astype(np.int32))

    def perturb(self, rows, seed):
        rng = np.random.default_rng(seed)
        if rows.data.ndim == 1:
            d = rows.data.astype(np.int64)
            return Rows(rows.dtype, ((d + 1 + rng.integers(0, self.V - 1, d.shape)) % self.V).astype(rows.data.dtype))
        new = self.hidden(rows.n, 10_000 + seed)
        assert all(new.row_bytes(i) != rows.row_bytes(i) for i in range(rows.n))
        return new

    def fuse_taps(self, raw):
        W = torch.from_numpy(self.ckpt.logical["fc"])
        cat = torch.cat([torch.from_numpy(bf16_values(raw[t])).reshape(raw[t].n, -1) for t in sorted(self.taps)], 1)
        return to_rows((cat @ W.T).to(ACT))

    def close_enough(self, a, b, what):
        if what == "reference":
            return a == b
        x, y = bf16_values(a), bf16_values(b)
        scale = max(float(np.abs(y).max()), 1e-6)
        return x.shape == y.shape and float(np.abs(x - y).max()) <= 4 * 2.0 ** (math.floor(math.log2(scale)) - 7)

    def make_broken(self, lo, hi):
        rt = CpuRuntime(self._spec(lo, hi, lo == 0, hi == len(self.layer_kinds)), self.config)
        rt.load(self.ckpt.without(f"model.layers.{lo}.self_attn.o_proj.weight"))
