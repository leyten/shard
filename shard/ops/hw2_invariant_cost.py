"""HW2 T2/T4 stub: what does receipt-grade cost, and is it invariant ON THE DEVICE?

The riskiest assumption of shard v2 (design-runtime §6 R1): a batch-invariant forward costs <= 1.5x the fast
path at decode/verify token counts and 32k context. This script measures, per MiMo-V2.6-Flash layer op:

  invariant   shard.ops.invariant_moe / invariant_attn (the receipt-grade pins)
  fast        batch-dependent variants of the same kernels (split-K / split size picked from the batch --
              an in-house proxy that is always available) and, when importable, the real fast stack
              (vLLM Marlin MXFP4 MoE, FlashInfer attention with sinks) -- wiring those is R5's op library,
              recorded as "unavailable" until then
and reports invariant_ms / best_fast_ms per op and per layer (MoE + attention), plus an on-device byte-identity
probe (probe rows alone vs inside random batches, and run-to-run) -- the thing the CPU interpreter cannot
show: bf16 MMA behaviour, ex2.approx, FTZ, compiler scheduling.

GPU:      python -m shard.ops.hw2_invariant_cost --out hw2_<arch>.json
GPU-less: TRITON_INTERPRET=1 python -m shard.ops.hw2_invariant_cost --smoke --out /tmp/hw2.json
          (tiny shapes; timings meaningless and flagged; exercises every code path + the JSON schema)
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

FLASH = dict(H=4096, I=2048, E=256, top_k=8, hq=64, hkv_global=4, hkv_swa=8, window=128,
             decode=[1, 4, 8, 16, 32, 64], verify=[(1, 8), (4, 8), (8, 8)], prefill=[512],
             ctx=[4096, 32768], reps=20, probe_moe=(2, 17, 64, 1), probe_attn=(3, 9, 40))
SMOKE = dict(H=256, I=128, E=16, top_k=4, hq=16, hkv_global=1, hkv_swa=2, window=128,
             decode=[1], verify=[(1, 2)], prefill=[], ctx=[80], reps=1, probe_moe=(3,), probe_attn=(3,))
RATIO_GATE = 1.5


def _timer(fn, reps: int) -> float:
    """Median ms. CUDA events on a GPU; perf_counter otherwise (smoke only)."""
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        out = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            out.append(a.elapsed_time(b))
        return statistics.median(out)
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1e3)
    return statistics.median(out)


def _experts(cfg, dev, g):
    E, H, I = cfg["E"], cfg["H"], cfg["I"]
    r = lambda *s: torch.randint(0, 256, s, dtype=torch.uint8, generator=g).to(dev)          # noqa: E731
    sc = lambda *s: torch.randint(118, 124, s, dtype=torch.uint8, generator=g).to(dev)       # noqa: E731
    return r(E, 2 * I, H // 2), sc(E, 2 * I, H // 32), r(E, H, I // 2), sc(E, H, I // 32)


def _route(T, cfg, dev, g):
    ids = torch.stack([torch.randperm(cfg["E"], generator=g)[:cfg["top_k"]] for _ in range(T)])
    w = torch.rand(T, cfg["top_k"], generator=g) + 0.05
    return ids.to(dev), (w / w.sum(-1, keepdim=True)).to(dev)


def _kv(n_seq, ctx, hkv, dev, g, page=16):
    npg = -(-ctx // page)
    kc = torch.randn(n_seq * npg, page, hkv, 192, generator=g).to(torch.bfloat16).to(dev)
    vc = torch.randn(n_seq * npg, page, hkv, 128, generator=g).to(torch.bfloat16).to(dev)
    bt = torch.arange(n_seq * npg, dtype=torch.int32).view(n_seq, npg).to(dev)
    return kc, vc, bt


def fast_backends():
    """Which real fast kernels are importable here (R5 wires them; until then: recorded, not timed)."""
    have = {}
    for name, mod in (("vllm_marlin_mxfp4", "vllm"), ("flashinfer_sinks", "flashinfer")):
        try:
            __import__(mod)
            have[name] = "importable (timing not wired: TODO R5)"
        except Exception:                           # noqa: BLE001 -- any import failure = unavailable
            have[name] = "unavailable"
    return have


def measure(cfg: dict, dev: str) -> dict:
    from . import invariant_attn as ia
    from . import invariant_moe as im
    g = torch.Generator().manual_seed(0)
    w13, s13, w2, s2 = _experts(cfg, dev, g)
    rows = []
    token_counts = [("decode", b) for b in cfg["decode"]] + [("verify", b * k) for b, k in cfg["verify"]] \
        + [("prefill", n) for n in cfg["prefill"]]
    for kind, T in token_counts:
        x = torch.randn(T, cfg["H"], generator=g).to(torch.bfloat16).to(dev)
        ids, w = _route(T, cfg, dev, g)
        inv = _timer(lambda: im.moe_experts_forward(x, ids, w, w13, s13, w2, s2), cfg["reps"])
        fast = _timer(lambda: im.moe_experts_forward(x, ids, w, w13, s13, w2, s2, naive=True), cfg["reps"])
        rows.append(dict(op="moe", kind=kind, tokens=T, invariant_ms=inv, fast_ms={"triton_splitk": fast}))
        for layer, hkv, window, sinks in (("global", cfg["hkv_global"], 0, False),
                                          ("swa", cfg["hkv_swa"], cfg["window"], True)):
            for ctx in cfg["ctx"]:
                n_seq = T if kind == "decode" else max(1, T // 8)
                kc, vc, bt = _kv(n_seq, ctx, hkv, dev, g)
                seq = torch.arange(T, dtype=torch.int32, device=dev) % n_seq
                pos = torch.full((T,), ctx - 1, dtype=torch.int32, device=dev) - (torch.arange(T, device=dev) // n_seq).int()
                q = torch.randn(T, cfg["hq"], 192, generator=g).to(torch.bfloat16).to(dev)
                sk = torch.randn(cfg["hq"], generator=g).to(dev) if sinks else None
                inv = _timer(lambda: ia.paged_diffkv_attention(q, kc, vc, bt, seq, pos, window, sk), cfg["reps"])
                fast = _timer(lambda: ia.paged_diffkv_attention(q, kc, vc, bt, seq, pos, window, sk,
                                                                 split=ia.naive_split(T)), cfg["reps"])
                rows.append(dict(op=f"attn_{layer}", kind=kind, tokens=T, ctx=ctx, invariant_ms=inv,
                                 fast_ms={"triton_batch_split": fast}))
                del kc, vc
    for r in rows:
        r["ratio"] = r["invariant_ms"] / min(r["fast_ms"].values())
    return dict(rows=rows, invariance=probe_invariance(cfg, dev, w13, s13, w2, s2, g))


def probe_invariance(cfg, dev, w13, s13, w2, s2, g) -> dict:
    """On-device byte identity: one probe token alone vs inside random batches, twice (run-to-run)."""
    from . import invariant_attn as ia
    from . import invariant_moe as im
    px = torch.randn(1, cfg["H"], generator=g).to(torch.bfloat16).to(dev)
    pid, pw = _route(1, cfg, dev, g)
    base = im.moe_experts_forward(px, pid, pw, w13, s13, w2, s2)[0]
    moe_ok = []
    for n in cfg["probe_moe"]:
        x = torch.randn(n, cfg["H"], generator=g).to(torch.bfloat16).to(dev)
        ids, w = _route(n, cfg, dev, g)
        x[n // 2], ids[n // 2], w[n // 2] = px[0], pid[0], pw[0]
        out = im.moe_experts_forward(x, ids, w, w13, s13, w2, s2)
        moe_ok.append(bool(torch.equal(out[n // 2].view(torch.int16), base.view(torch.int16))))
    ctx = cfg["ctx"][-1]
    hkv = cfg["hkv_swa"]
    kc, vc, bt = _kv(4, ctx, hkv, dev, g)
    pq = torch.randn(1, cfg["hq"], 192, generator=g).to(torch.bfloat16).to(dev)
    sk = torch.randn(cfg["hq"], generator=g).to(dev)
    one = lambda t: torch.tensor(t, dtype=torch.int32, device=dev)                            # noqa: E731
    attn_ok = []
    for window in (0, cfg["window"]):
        ref = ia.paged_diffkv_attention(pq, kc, vc, bt, one([0]), one([ctx - 1]), window, sk)[0]
        for n in cfg["probe_attn"]:
            q = torch.randn(n, cfg["hq"], 192, generator=g).to(torch.bfloat16).to(dev)
            seq = torch.randint(0, 4, (n,), generator=g).int().to(dev)
            pos = torch.randint(0, ctx, (n,), generator=g).int().to(dev)
            q[n - 1], seq[n - 1], pos[n - 1] = pq[0], 0, ctx - 1
            out = ia.paged_diffkv_attention(q, kc, vc, bt, seq, pos, window, sk)
            attn_ok.append(bool(torch.equal(out[n - 1].view(torch.int16), ref.view(torch.int16))))
    return dict(moe_probe_byte_identical=moe_ok, attn_probe_byte_identical=attn_ok,
                all_ok=all(moe_ok) and all(attn_ok))


def summarize(res: dict, meaningful: bool = True) -> dict:
    small = [r for r in res["rows"] if r["kind"] in ("decode", "verify")]
    worst = max(small, key=lambda r: r["ratio"]) if small else None
    by_tokens = {}
    for r in small:
        by_tokens.setdefault((r["kind"], r["tokens"]), []).append(r)
    layer = {}
    for (kind, T), rs in by_tokens.items():             # one MoE + one global + one SWA attention per (T, ctx)
        for ctx in sorted({r.get("ctx") for r in rs if r.get("ctx")}):
            parts = [r for r in rs if r["op"] == "moe" or r.get("ctx") == ctx]
            inv = sum(r["invariant_ms"] for r in parts)
            fast = sum(min(r["fast_ms"].values()) for r in parts)
            layer[f"{kind}:{T}:ctx{ctx}"] = inv / fast
    return dict(worst_op=worst and {k: worst[k] for k in ("op", "kind", "tokens", "ratio")},
                layer_ratio=layer, gate=RATIO_GATE,
                pass_ratio=(bool(layer) and max(layer.values()) <= RATIO_GATE) if meaningful else None,
                pass_invariance=res["invariance"]["all_ok"])


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true", help="tiny shapes (CPU interpreter OK)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = SMOKE if a.smoke else FLASH
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and not a.smoke:
        raise SystemExit("no GPU: run with --smoke (and TRITON_INTERPRET=1)")
    from . import invariant_attn as ia
    from . import invariant_moe as im
    res = measure(cfg, dev)
    res["meta"] = dict(device=torch.cuda.get_device_name() if dev == "cuda" else "cpu-interpreter",
                       capability=list(torch.cuda.get_device_capability()) if dev == "cuda" else None,
                       smoke=a.smoke, timings_meaningful=dev == "cuda" and not a.smoke,
                       pins=[im.NUMERICS_PIN, ia.NUMERICS_PIN], fast_backends=fast_backends(),
                       torch=torch.__version__)
    res["summary"] = summarize(res, res["meta"]["timings_meaningful"])
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    print(json.dumps(res["summary"], indent=1))
    return res


if __name__ == "__main__":
    main()
