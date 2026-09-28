"""HW1 part B: MiMo-V2.6-Flash layer op microbench on real tensors (BUILD-PLAN §4.1 B1-B4, K5-K7).

Layers 0 (global, dense), 1 (SWA + MoE, the only kind with attention sinks: add_swa_attention_sink_bias)
and 5 (global + MoE) come from the tensor-granular ranged fetch (`python -m shard.fetch ranged`),
one `layer-000NN.safetensors` per layer. Formats go through shard.formats (qkv per-chunk re-chunk +
FP8-block dequant to bf16, MXFP4 exact).

Backends (the one that ACTUALLY ran is recorded per record; fallbacks are labelled, never silent):
  experts   marlin        engines/kimi_k3 MarlinRoutedExperts with the MiMo->K3 shims (k3_shims())
            torch_dequant bf16 dequantized experts, per-expert loop (reference / CPU / fallback)
  attention flashinfer_sink flashinfer JIT BatchAttentionWithAttentionSinkWrapper (paged, NHD)
            torch_sink    explicit sink-column softmax (reference / CPU / fallback)
Gates run BEFORE any timing and are written as checks:
  sink_effect    the attention backend's output WITH sinks differs from the same call with sinks=-inf
                 (flashinfer 0.6.12's plain `run(sinks=)` is silently ignored on sm_120) AND matches
                 the torch reference within tolerance. A backend failing it is not timed.
  marlin_vs_ref  Marlin expert output == dequant reference within tolerance (proves the MXFP4 ->
                 compressed-tensors re-layout; a wrong nibble order is fluent garbage, not a crash).

Modes: ops (B1-B4 timings + footprint + derived per-layer / replay rate), dump (fixed-input layer-
shaped op chain -> tensors + sha256 for research/hw1/xfamily.py).

`--stub` builds a synthetic MiMo-shaped layer at tiny dims (same tensor NAMES, dtypes and packing as
the checkpoint) and runs on CPU: the CI path.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
import hw1_common as C  # noqa: E402
from shard.formats.quant import dequant_mxfp4, to_bf16_values  # noqa: E402
from shard.formats.mimo_v2 import qkv_rechunk  # noqa: E402

# MiMo-V2.6-Flash config.json facts (see scratchpad/v2/mimo/vendor/config.json)
REAL = dict(hidden=4096, heads=64, qk=192, v=128, kv_full=4, kv_swa=8, experts=256, topk=8, inter=2048,
            window=128, ckpt_tp=4, eps=1e-6)
TINY = dict(hidden=128, heads=8, qk=24, v=16, kv_full=4, kv_swa=8, experts=8, topk=2, inter=64,
            window=8, ckpt_tp=4, eps=1e-6)
PATTERN = {0: "full", 1: "swa", 5: "full"}          # hybrid_layer_pattern: 0 = global, 1 = SWA
MOE = {0: False, 1: True, 5: True}                  # moe_layer_freq[0] == 0 (layer 0 dense)
T_GRID = (1, 2, 4, 8, 9, 16, 17, 32, 64, 128, 256, 900, 2048)
PREFILL_T = (900, 2048)
SPAN_LAYERS = 8
PREFILL_PREFIX = 8192
N_SWA, N_FULL = 39, 9       # hybrid_layer_pattern: 39 SWA, 9 global of 48       # the global-layer prefix for the "replay deep in a session" variant


# ── synthetic checkpoint (tiny, MiMo names/dtypes/packing) ──────────────────────────────────────
def _fp8_block(w, block=128):
    """fp32 [N,K] -> (fp8 [N,K], fp32 scale_inv [ceil(N/b), ceil(K/b)]) on the plain grid."""
    n, k = w.shape
    s = torch.zeros(math.ceil(n / block), math.ceil(k / block))
    q = torch.empty(n, k, dtype=torch.float8_e4m3fn)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            blk = w[i * block:(i + 1) * block, j * block:(j + 1) * block]
            sc = blk.abs().max().clamp(min=1e-8) / 448.0
            s[i, j] = sc
            q[i * block:(i + 1) * block, j * block:(j + 1) * block] = (blk / sc).to(torch.float8_e4m3fn)
    return q, s


def _mxfp4(n, k, g):
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=g)
    scale = torch.randint(118, 124, (n, k // 32), dtype=torch.uint8, generator=g)   # 2^-9..2^-4
    return packed, scale


def synth_layer(li, d=TINY, seed=0):
    """{tensor name: tensor} for decoder layer li, MiMo naming. The qkv is fp8 with ONE chunk per
    ckpt_tp (per-chunk scale grid), exactly the shape class qkv_rechunk expects."""
    g = torch.Generator().manual_seed(seed + li)
    kind = PATTERN.get(li, "swa")
    nkv = d["kv_swa"] if kind == "swa" else d["kv_full"]
    H, tp = d["hidden"], d["ckpt_tp"]
    p = f"model.layers.{li}."
    out = {}
    rows_pc = (d["heads"] * d["qk"] + nkv * d["qk"] + nkv * d["v"]) // tp
    chunks, scales = [], []
    for _ in range(tp):
        q, s = _fp8_block(torch.randn(rows_pc, H, generator=g) * 0.05)
        chunks.append(q)
        scales.append(s)
    out[p + "self_attn.qkv_proj.weight"] = torch.cat(chunks)
    out[p + "self_attn.qkv_proj.weight_scale_inv"] = torch.cat(scales)
    out[p + "self_attn.o_proj.weight"] = (torch.randn(H, d["heads"] * d["v"], generator=g) * 0.05).bfloat16()
    out[p + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    out[p + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    if kind == "swa":
        out[p + "self_attn.attention_sink_bias"] = (torch.randn(d["heads"], generator=g) * 2).bfloat16()
    if MOE.get(li, True):
        out[p + "mlp.gate.weight"] = (torch.randn(d["experts"], H, generator=g) * 0.05).bfloat16()
        out[p + "mlp.gate.e_score_correction_bias"] = torch.randn(d["experts"], generator=g) * 0.01
        for e in range(d["experts"]):
            for nm, (n, k) in (("gate_proj", (d["inter"], H)), ("up_proj", (d["inter"], H)),
                               ("down_proj", (H, d["inter"]))):
                w, s = _mxfp4(n, k, g)
                out[f"{p}mlp.experts.{e}.{nm}.weight"] = w
                out[f"{p}mlp.experts.{e}.{nm}.weight_scale"] = s
    else:
        I = d["inter"] * 2
        for nm, (n, k) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            q, s = _fp8_block(torch.randn(n, k, generator=g) * 0.05)
            out[f"{p}mlp.{nm}.weight"], out[f"{p}mlp.{nm}.weight_scale_inv"] = q, s
    return out


def write_synth(dirpath, layers=(0, 1, 5), d=TINY, seed=0):
    from safetensors.torch import save_file
    os.makedirs(dirpath, exist_ok=True)
    for li in layers:
        save_file(synth_layer(li, d, seed), os.path.join(dirpath, f"layer-{li:05d}.safetensors"))
    return dirpath


# ── loading (real or synthetic, same code) ──────────────────────────────────────────────────────
def load_layer(dirpath, li, d, device):
    from safetensors.torch import load_file
    t = load_file(os.path.join(dirpath, f"layer-{li:05d}.safetensors"))
    p = f"model.layers.{li}."
    kind = PATTERN.get(li, "swa")
    nkv = d["kv_swa"] if kind == "swa" else d["kv_full"]
    wq, wk, wv = qkv_rechunk(t[p + "self_attn.qkv_proj.weight"], t[p + "self_attn.qkv_proj.weight_scale_inv"],
                             d["heads"], nkv, d["qk"], d["v"], d["ckpt_tp"])
    L = types.SimpleNamespace(li=li, kind=kind, nkv=nkv, d=d, device=device, raw=t, prefix=p)
    L.wqkv = to_bf16_values(torch.cat([wq, wk, wv])).bfloat16().to(device)         # FP8 upcast: the 3090 path too
    L.wo = t[p + "self_attn.o_proj.weight"].to(device)
    L.ln1 = t[p + "input_layernorm.weight"].to(device)
    L.ln2 = t[p + "post_attention_layernorm.weight"].to(device)
    sb = t.get(p + "self_attn.attention_sink_bias")
    L.sinks = sb.float().to(device) if sb is not None else None
    L.moe = (p + "mlp.gate.weight") in t
    if L.moe:
        L.gate = t[p + "mlp.gate.weight"].float().to(device)                # router stays fp32
        L.bias = t[p + "mlp.gate.e_score_correction_bias"].float().to(device)
    return L


def weight_bytes(L):
    return sum(v.numel() * v.element_size() for v in L.raw.values())


# ── ops ─────────────────────────────────────────────────────────────────────────────────────────
def rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def router(L, x):
    """fp32 sigmoid router with the noaux_tc correction bias (selection only), top-k renormalized."""
    scores = torch.sigmoid(x.float() @ L.gate.t())
    ids = torch.topk(scores + L.bias, L.d["topk"], dim=-1).indices
    w = scores.gather(-1, ids)
    return ids, w / w.sum(-1, keepdim=True)


class TorchExperts:
    """Reference experts: exact MXFP4 dequant (on CPU, shard.formats) of only the experts a call routes
    to, cached on the device up to `budget_gb` (all 256 of a real layer would be 12.9 GB of bf16 and
    OOM a 24 GB card). A fallback, never calibration grade; used as the marlin_vs_ref reference."""
    backend = "torch_dequant"

    def __init__(self, L, budget_gb=4.0):
        self.L, self.p, self.cache, self.bytes, self.budget = L, L.prefix + "mlp.experts.", {}, 0, budget_gb * 2**30

    def _w(self, e):
        if e in self.cache:
            return self.cache[e]
        raw, dev = self.L.raw, self.L.device
        ws = tuple(dequant_mxfp4(raw[f"{self.p}{e}.{nm}.weight"], raw[f"{self.p}{e}.{nm}.weight_scale"])
                   .bfloat16().to(dev) for nm in ("gate_proj", "up_proj", "down_proj"))
        nb = sum(t.numel() * 2 for t in ws)
        if self.bytes + nb <= self.budget:
            self.cache[e] = ws
            self.bytes += nb
        return ws

    def __call__(self, x, ids, w):
        out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
        for e in torch.unique(ids).tolist():
            w1, w3, w2 = self._w(e)
            tok, slot = (ids == e).nonzero(as_tuple=True)
            h = x[tok]
            a = torch.nn.functional.silu((h @ w1.t()).float()) * (h @ w3.t()).float()
            out.index_add_(0, tok, (a.bfloat16() @ w2.t()).float() * w[tok, slot, None])
        return out.to(x.dtype)


def k3_shims(L, max_tokens=4096):
    """The shims the skeptic listed for reusing K3's MarlinRoutedExperts on MiMo: a cfg with K3's field
    names, a synthetic compressed-tensors `mxfp4-pack-quantized` block for spec(), K3_MOE_MAX_TOKENS >=
    the largest T (default 64), SiLU (hidden_act != 'situ' keeps K3's situ hook off), and a get_tensor
    that maps K3 names (w1/w3/w2 .weight_packed/.weight_scale) onto MiMo's. Pure python (CI-tested)."""
    d = L.d
    cfg = types.SimpleNamespace(num_experts=d["experts"], num_experts_per_token=d["topk"],
                                hidden_size=d["hidden"], routed_expert_hidden_size=None,
                                moe_intermediate_size=d["inter"], hidden_act="silu")
    qc = {"quant_method": "compressed-tensors", "format": "mxfp4-pack-quantized",
          "config_groups": {"group_0": {"targets": ["experts"], "weights": {
              "num_bits": 4, "group_size": 32, "symmetric": True, "strategy": "group", "type": "float"}}},
          "ignore": []}
    names = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
    kind = {"weight_packed": "weight", "weight_scale": "weight_scale"}

    def get_tensor(name):                     # "<prefix>.experts.<e>.w1.weight_packed" -> MiMo tensor
        head, e, w, k = name.rsplit(".", 3)
        return L.raw[f"{L.prefix}mlp.experts.{e}.{names[w]}.{kind[k]}"]

    env = {"K3_MOE_MAX_TOKENS": str(max(max_tokens, 2048))}
    return cfg, qc, get_tensor, env


class MarlinExperts:
    backend = "marlin"

    def __init__(self, L, max_tokens):
        cfg, qc, get_tensor, env = k3_shims(L, max_tokens)
        os.environ.update(env)
        for p in (os.path.join(REPO, "vendor"), os.path.join(REPO, "engines", "kimi_k3")):
            if p not in sys.path:
                sys.path.insert(0, p)             # k3_moe_mxfp4 hard-imports kimi_k3_ref (vendor/)
        import k3_moe_mxfp4 as K
        K.spec(qc)
        self.m = K.MarlinRoutedExperts(cfg, qc, device=L.device).load(get_tensor, L.prefix + "mlp")
        try:                                      # the kernel class vLLM really dispatched to, not a label
            self.kernel = type(K.fused_experts_of(self.m.quant_method)).__name__
        except Exception as e:  # noqa: BLE001
            self.kernel = f"unknown ({type(e).__name__})"
        self.param_bytes = sum(t.numel() * t.element_size() for t in self.m.experts.parameters())

    def __call__(self, x, ids, w):
        return self.m.moe_infer(x, ids, w)


def attn_ref(q, k, v, sinks, window, start):
    """Reference attention with sinks. q [T,Hq,Dq] at positions start..start+T-1, k/v [S,Hkv,*] over
    positions 0..S-1 (S = start+T). Causal; SWA keeps keys with pos > qpos - window; the sink is an
    extra softmax column per head that contributes no value."""
    T, Hq, Dq = q.shape
    S, Hkv, _ = k.shape
    rep = Hq // Hkv
    kk = k.repeat_interleave(rep, 1).float()
    vv = v.repeat_interleave(rep, 1).float()
    lg = torch.einsum("thd,shd->hts", q.float(), kk) * (Dq ** -0.5)
    qp = torch.arange(start, start + T, device=q.device)[:, None]
    kp = torch.arange(S, device=q.device)[None, :]
    bad = kp > qp
    if window:
        bad |= kp <= qp - window
    lg = lg.masked_fill(bad[None], float("-inf"))
    if sinks is not None:
        lg = torch.cat([lg, sinks.float()[:, None, None].expand(Hq, T, 1)], -1)
    p = torch.softmax(lg, -1)[..., :S]
    return torch.einsum("hts,shd->thd", p, vv).to(q.dtype)


class TorchAttn:
    backend = "torch_sink"

    def bind(self, q, k, v, sinks, window, start):
        return lambda: attn_ref(q, k, v, sinks, window, start)

    def __call__(self, *a):
        return self.bind(*a)()


class FlashInferSinkAttn:
    """flashinfer JIT BatchAttentionWithAttentionSinkWrapper, one sequence, page_size 16, NHD."""
    backend = "flashinfer_sink"

    def __init__(self, device, hq, hkv, dq, dv, window):
        import flashinfer
        self.fi, self.dev, self.hq, self.hkv, self.dq, self.dv = flashinfer, device, hq, hkv, dq, dv
        self.window = window - 1 if window else -1      # flashinfer window_left excludes the query itself
        ws = torch.empty(256 << 20, dtype=torch.uint8, device=device)
        self.w = flashinfer.BatchAttentionWithAttentionSinkWrapper(
            ws, kv_layout="NHD", backend="fa2", q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
            head_dim_qk=dq, head_dim_vo=dv, window_left=self.window)
        self.planned = None

    def bind(self, q, k, v, sinks, window, start):
        """Plan + build the paged cache ONCE, outside the timed call; the returned closure is only
        wrapper.run (so a timing never includes the KV re-layout copy or the planner)."""
        T, S, P = q.shape[0], k.shape[0], 16
        npg = -(-S // P)
        if self.planned != (T, S):
            i32 = dict(dtype=torch.int32, device=self.dev)
            self.w.plan(torch.tensor([0, T], **i32), torch.tensor([0, npg], **i32), torch.arange(npg, **i32),
                        torch.tensor([S - (npg - 1) * P], **i32), self.hq, self.hkv, self.dq, P,
                        head_dim_vo=self.dv, causal=True, window_left=self.window,
                        q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
            self.planned = (T, S)
        pad = npg * P - S
        kc = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad)).view(npg, P, self.hkv, self.dq)
        vc = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad)).view(npg, P, self.hkv, self.dv)
        snk = (sinks if sinks is not None else torch.full((self.hq,), -1e30, device=self.dev)).float().contiguous()
        return lambda: self.w.run(q, (kc, vc), snk, self.dq ** -0.5)

    def __call__(self, *a):
        return self.bind(*a)()


def sink_effect_check(res, attn, d, device, tol=2e-2, seed=0, S=4096):
    """Gate: sinks must change the output, and the backend must match the reference. S >> window, so a
    wrong window_left fails too. Returns bool."""
    g = torch.Generator().manual_seed(seed)
    T = 9
    hq, hkv = d["heads"], d["kv_swa"]
    q = torch.randn(T, hq, d["qk"], generator=g).bfloat16().to(device)
    k = torch.randn(S, hkv, d["qk"], generator=g).bfloat16().to(device)
    v = torch.randn(S, hkv, d["v"], generator=g).bfloat16().to(device)
    sinks = (torch.randn(hq, generator=g) * 2 + 1).float().to(device)
    start = S - T
    try:
        with_s = attn(q, k, v, sinks, d["window"], start).float()
        no_s = attn(q, k, v, torch.full_like(sinks, -1e30), d["window"], start).float()
    except Exception as e:  # noqa: BLE001 -- a backend that cannot run is a failed gate, recorded
        res.check("sink_effect", False, f"{attn.backend}: {type(e).__name__}: {e}", backend=attn.backend)
        return False
    ref = attn_ref(q, k, v, sinks, d["window"], start).float()
    effect = (with_s - no_s).abs().max().item()
    err = (with_s - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
    ok = effect > 1e-3 and err < tol
    res.check("sink_effect", ok, f"{attn.backend}: |with-without sinks| {effect:.3e}, rel err vs ref {err:.3e}",
              backend=attn.backend, effect=effect, rel_err=err)
    return ok


def full_attn_check(res, attn, d, device, tol=2e-2, seed=1, S=4096):
    """Gate for the global-attention wrapper (no sinks, no window) against the reference."""
    g = torch.Generator().manual_seed(seed)
    T, hq, hkv = 9, d["heads"], d["kv_full"]
    q = torch.randn(T, hq, d["qk"], generator=g).bfloat16().to(device)
    k = torch.randn(S, hkv, d["qk"], generator=g).bfloat16().to(device)
    v = torch.randn(S, hkv, d["v"], generator=g).bfloat16().to(device)
    try:
        out = attn(q, k, v, None, 0, S - T).float()
    except Exception as e:  # noqa: BLE001
        res.check("full_attn", False, f"{attn.backend}: {type(e).__name__}: {e}", backend=attn.backend)
        return False
    ref = attn_ref(q, k, v, None, 0, S - T).float()
    err = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
    res.check("full_attn", err < tol, f"{attn.backend}: rel err vs ref {err:.3e}", backend=attn.backend, rel_err=err)
    return err < tol


def marlin_check(res, L, fast, ref, tol=3e-2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(17, L.d["hidden"], generator=g) * 0.5).bfloat16().to(L.device)
    ids, w = router(L, x)
    a, b = fast(x, ids, w).float(), ref(x, ids, w).float()
    err = (a - b).abs().max().item() / (b.abs().max().item() + 1e-9)
    ok = err < tol
    res.check("marlin_vs_ref", ok, f"{fast.backend} vs {ref.backend}: rel err {err:.3e}", rel_err=err,
              kernel=getattr(fast, "kernel", None))
    return ok


def pick_experts(res, L, want, max_tokens):
    ref = TorchExperts(L)
    if want == "marlin":
        try:
            m = MarlinExperts(L, max_tokens)
            if marlin_check(res, L, m, ref):
                ref.cache.clear()                  # the reference's device cache is not part of the stage
                if L.device == "cuda":
                    torch.cuda.empty_cache()
                return m
        except Exception as e:  # noqa: BLE001 -- fall back, labelled
            res.check("marlin_vs_ref", False, f"marlin unavailable: {type(e).__name__}: {e}")
    return ref


def pick_attn(res, d, device, want):
    cands = []
    if want == "flashinfer":
        try:
            cands.append(FlashInferSinkAttn(device, d["heads"], d["kv_swa"], d["qk"], d["v"], d["window"]))
        except Exception as e:  # noqa: BLE001
            res.check("sink_effect", False, f"flashinfer_sink unavailable: {type(e).__name__}: {e}")
    cands.append(TorchAttn())
    for a in cands:
        if sink_effect_check(res, a, d, device):
            return a
    return None


def pick_full(res, swa, d, device):
    """The global-layer attention backend: flashinfer (window -1) if the SWA one is, gated; else torch."""
    if swa is not None and swa.backend == "flashinfer_sink":
        a = attn_full(swa, d, device)
        if a is not None and full_attn_check(res, a, d, device):
            return a
    t = TorchAttn()
    full_attn_check(res, t, d, device)
    return t


# ── the timed grid ──────────────────────────────────────────────────────────────────────────────
INPUT_KEYS = {"attn": "L1.attn_in", "ffn": "L1.ffn_in"}     # golden t32 (mimo_ref): normalised layer inputs


def _hid(T, d, device, inputs, role="ffn", seed=0):
    """T rows of layer input. Real golden rows first (never repeated: repeats would cap the distinct
    routing patterns); rows beyond the real count are fresh random rows at the real rows' RMS."""
    g = torch.Generator().manual_seed(seed)
    if inputs is None:
        return (torch.randn(T, d["hidden"], generator=g) * 0.5).bfloat16().to(device)
    x = inputs[INPUT_KEYS[role]].float().reshape(-1, d["hidden"])[:T]
    if x.shape[0] < T:
        rms_ = x.pow(2).mean().sqrt()
        x = torch.cat([x, torch.randn(T - x.shape[0], d["hidden"], generator=g) * rms_])
    return x.bfloat16().to(device)


FALLBACK_EXPERTS_MAX_T = 64


def run_ops(res, Ls, experts, attn, d, device, tgrid, bs, ctxs, ss, warmup, iters, inputs, full=None):
    for L in Ls:
        for T in tgrid:
            xa = _hid(T, d, device, inputs, "attn")
            x = _hid(T, d, device, inputs, "ffn")
            prm = {"layer": L.li, "T": T}
            st = C.time_fn(lambda: xa @ L.wqkv.t(), warmup=warmup, iters=iters, device=device)
            res.record("B3", "qkv", prm, st)
            a = torch.zeros(T, d["heads"] * d["v"], dtype=torch.bfloat16, device=device)
            st = C.time_fn(lambda: a @ L.wo.t(), warmup=warmup, iters=iters, device=device)
            res.record("B3", "o_proj", prm, st)
            if L.moe:
                st = C.time_fn(lambda: router(L, x), warmup=warmup, iters=iters, device=device)
                res.record("B1", "router", prm, st)
                ids, w = router(L, x)
                ex = experts[L.li]
                if ex.backend != "marlin" and device == "cuda" and T > FALLBACK_EXPERTS_MAX_T:
                    res.record("B1", "experts", prm, skipped=f"fallback {ex.backend}: not timed above "
                               f"T={FALLBACK_EXPERTS_MAX_T} (not calibration grade)")
                    continue
                st = C.time_fn(lambda: ex(x, ids, w), warmup=warmup, iters=iters, device=device)
                res.record("B1", "experts", prm, st, backend=ex.backend, kernel=getattr(ex, "kernel", None),
                           distinct_experts=int(torch.unique(ids).numel()))
    if attn is None:
        res.record("B2", "attn", {}, skipped="no attention backend passed the sink_effect gate")
        return
    for kind, hkv, win in (("swa", d["kv_swa"], d["window"]), ("full", d["kv_full"], 0)):
        use = attn if kind == "swa" else (full or TorchAttn())
        # prefill shapes (the replay-rate input). A replayed chunk of a long session attends a prefix: the
        # global layers are also timed at a PREFILL_PREFIX-token prefix (SWA cost is window-bounded).
        for T in [t for t in tgrid if t in PREFILL_T]:
            for start in ((0,) if kind == "swa" else (0, PREFILL_PREFIX)):
                q = torch.randn(T, d["heads"], d["qk"]).bfloat16().to(device)
                k = torch.randn(start + T, hkv, d["qk"]).bfloat16().to(device)
                v = torch.randn(start + T, hkv, d["v"]).bfloat16().to(device)
                sk = torch.zeros(d["heads"], device=device) if kind == "swa" else None
                st = C.time_fn(use.bind(q, k, v, sk, win, start), warmup=warmup, iters=iters, device=device)
                res.record("B2", "attn_prefill", {"kind": kind, "T": T, "start": start}, st, backend=use.backend)
        for ctx in ctxs:
            for B in bs:
                for s in ss:
                    prm = {"kind": kind, "B": B, "ctx": ctx, "s": s}
                    sk = torch.zeros(d["heads"], device=device) if kind == "swa" else None
                    # B sequences, each with its OWN q/K/V (one shared KV would sit in L2 and flatter B>1):
                    # B calls. A batched plan is R5's job; this is the per-sequence cost, labelled as such.
                    fns = [use.bind(torch.randn(s, d["heads"], d["qk"]).bfloat16().to(device),
                                    torch.randn(ctx + s, hkv, d["qk"]).bfloat16().to(device),
                                    torch.randn(ctx + s, hkv, d["v"]).bfloat16().to(device), sk, win, ctx)
                           for _ in range(B)]
                    st = C.time_fn(lambda: [f() for f in fns], warmup=warmup, iters=iters, device=device)
                    res.record("B2", "attn", prm, st, backend=use.backend, batched_kernel=False)


_FULL = {}


def attn_full(sw, d, device):
    if "a" not in _FULL:
        try:
            _FULL["a"] = FlashInferSinkAttn(device, d["heads"], d["kv_full"], d["qk"], d["v"], 0)
        except Exception:  # noqa: BLE001
            _FULL["a"] = None
    return _FULL["a"]


def invariance(res, L, ex, d, device, mixes=11, seed=0):
    """B4: the probe row's output bytes across 11 batch compositions. Expect drift with a non-invariant
    grouped GEMM (proves the byte test CAN fail); torch_dequant on CPU is typically invariant."""
    g = torch.Generator().manual_seed(seed)
    probe = (torch.randn(1, d["hidden"], generator=g) * 0.5).bfloat16().to(device)
    hs = set()
    for m in range(mixes):
        other = (torch.randn(1 + 3 * m, d["hidden"], generator=g) * 0.5).bfloat16().to(device)
        x = torch.cat([other[: m], probe, other[m:]])
        ids, w = router(L, x)
        y = ex(x, ids, w)[m]
        hs.add(hashlib.sha256(y.float().cpu().numpy().tobytes()).hexdigest())
    res.record("B4", "probe_bytes", {"layer": L.li, "mixes": mixes}, distinct_hashes=len(hs),
               backend=ex.backend)


def attn_ms(recs, T, ctx=8192, start=0):
    """attention per layer at shape T, weighted by the Flash pattern (39 SWA : 9 global). Prefill T ->
    attn_prefill; decode T (s) -> attn at B=1, the ctx closest to `ctx`."""
    def one(kind):
        if T in PREFILL_T:
            want = {"kind": kind, "T": T, "start": start if kind == "full" else 0}
            xs = [r["ms"] for r in recs if r["op"] == "attn_prefill" and r["params"] == want]
        else:
            xs = [r for r in recs if r["op"] == "attn" and r["params"]["kind"] == kind
                  and r["params"]["B"] == 1 and r["params"]["s"] == T]
            if xs:
                best = min(xs, key=lambda r: abs(r["params"]["ctx"] - ctx))["params"]["ctx"]
                xs = [r["ms"] for r in xs if r["params"]["ctx"] == best]
        return min(xs) if xs else None
    swa, full = one("swa"), one("full")
    if swa is None or full is None:
        return None
    return (N_SWA * swa + N_FULL * full) / (N_SWA + N_FULL)


def derive(res, d):
    """per-layer estimate (B1+B2+B3, +10% glue) and the replay rate for an 8-layer span."""
    recs = [r for r in res.doc["records"] if not r.get("skipped") and "ms" in r]

    def ms(op, **kw):
        xs = [r["ms"] for r in recs if r["op"] == op and all(r["params"].get(k) == v for k, v in kw.items())]
        return min(xs) if xs else None

    out = {}
    for T in PREFILL_T + (9,):
        per = []
        for li in (1, 5):
            parts = [ms("qkv", layer=li, T=T), ms("o_proj", layer=li, T=T), ms("router", layer=li, T=T),
                     ms("experts", layer=li, T=T)]
            if None in parts:
                continue
            per.append(sum(parts))
        if not per:
            continue
        a_ms = attn_ms(recs, T)
        if a_ms is None:
            continue
        layer_ms = (sum(per) / len(per) + a_ms) * 1.1
        out[f"layer_ms_T{T}"] = layer_ms
        if T in PREFILL_T:
            out[f"replay_tok_s_T{T}"] = T / (layer_ms * SPAN_LAYERS) * 1e3
            deep = attn_ms(recs, T, start=PREFILL_PREFIX)
            if deep is not None:                               # same span, global layers at an 8k prefix
                out[f"replay_tok_s_T{T}_prefix8k"] = T / ((sum(per) / len(per) + deep) * 1.1 * SPAN_LAYERS) * 1e3
    rates = [v for k, v in out.items() if k.startswith("replay_tok_s") and not k.endswith("prefix8k")]
    if rates:
        out["replay_tok_s_span"] = min(rates)
    res.doc["derived"].update(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ops", choices=("ops", "dump"))
    ap.add_argument("--weights", help="dir with layer-000NN.safetensors (python -m shard.fetch ranged)")
    ap.add_argument("--layers", default="0,1,5")
    ap.add_argument("--inputs", help="golden t32 safetensors (mimo_ref): uses L1.attn_in / L1.ffn_in")
    ap.add_argument("--experts", default="marlin", choices=("marlin", "torch"))
    ap.add_argument("--attn", default="flashinfer", choices=("flashinfer", "torch"))
    ap.add_argument("--tgrid", default=",".join(map(str, T_GRID)))
    ap.add_argument("--bs", default="1,2,4,8,16,32")
    ap.add_argument("--ctxs", default="1024,8192,32768")
    ap.add_argument("--ss", default="1,9,17")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--stub", action="store_true", help="synthetic tiny MiMo-shaped layers on CPU")
    ap.add_argument("--dump", help="dump mode: output .safetensors path")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ints = lambda s: [int(x) for x in s.split(",") if x]
    device = "cpu" if a.stub or not torch.cuda.is_available() else "cuda"
    d = TINY if a.stub else REAL
    res = C.Results("mimo_ops", a.mode, device=device, stub=a.stub, argv=argv or sys.argv[1:])
    wdir = a.weights
    if a.stub and not wdir:
        import tempfile
        wdir = write_synth(tempfile.mkdtemp(prefix="hw1_mimo_"), ints(a.layers), d)
    inputs = None
    if a.inputs:
        from safetensors.torch import load_file
        inputs = load_file(a.inputs)
        missing = [k for k in INPUT_KEYS.values() if k not in inputs]
        if missing:
            sys.exit(f"--inputs lacks {missing}")
    res.doc["derived"]["inputs"] = "real_rows_then_rms_matched_random" if inputs is not None else "random"
    Ls = []
    for li in ints(a.layers):
        if device == "cuda":
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
        L = load_layer(wdir, li, d, device)
        Ls.append(L)
        res.record("K7", "footprint", {"layer": li}, disk_bytes=weight_bytes(L),
                   resident_bytes=(torch.cuda.memory_allocated() - before) if device == "cuda" else None,
                   note="dense/attn only; experts counted in the experts record")
    maxT = max(ints(a.tgrid))
    experts = {}
    for L in Ls:
        if L.moe:
            before = torch.cuda.memory_allocated() if device == "cuda" else 0
            experts[L.li] = pick_experts(res, L, a.experts if device == "cuda" else "torch", maxT)
            ex_bytes = sum(v.numel() * v.element_size() for k, v in L.raw.items() if ".experts." in k)
            ex = experts[L.li]
            res.record("K7", "footprint_experts", {"layer": L.li}, backend=ex.backend, disk_bytes=ex_bytes,
                       resident_bytes=getattr(ex, "param_bytes", None),          # parameters, not workspace
                       allocated_delta=(torch.cuda.memory_allocated() - before) if device == "cuda" else None)
    attn = pick_attn(res, d, device, a.attn if device == "cuda" else "torch")
    full = pick_full(res, attn, d, device)
    res.doc["derived"]["calibration_grade"] = bool(
        inputs is not None and attn is not None and attn.backend == "flashinfer_sink"
        and full.backend == "flashinfer_sink" and all(e.backend == "marlin" for e in experts.values()))
    if a.mode == "dump":
        import xfamily
        xfamily.dump(res, Ls, experts, attn, d, device, inputs, a.dump or a.out + ".safetensors", router, rms)
        res.write(a.out)
        return 0
    run_ops(res, Ls, experts, attn, d, device, ints(a.tgrid), ints(a.bs), ints(a.ctxs), ints(a.ss),
            a.warmup, a.iters, inputs, full)
    for L in Ls:
        if L.moe:
            invariance(res, L, experts[L.li], d, device)
    derive(res, d)
    res.write(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
