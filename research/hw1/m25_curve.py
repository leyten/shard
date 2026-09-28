"""HW1 part A: the M2.5 stage on the existing engine (BUILD-PLAN §4.1 A0-A4).

Modes (one context per process -- the static KV buffer is sized at import from M25_KV_MAXLEN):
  anchor   A0: runs the UNMODIFIED research/m25_batched_moe_bench.py (the receipt's own code path)
           and parses its table; gate K1 = 42/26 ms @B4 and 71/41 ms @B8 at 16k, 12 layers, +-20%.
  curve    A1: stage service time vs batch: S.run_block_decode_b (eager) and BatchGraphRunner
           (graphed, where _manual_ok allows -- recorded per point), per-stream vs batched MoE.
  tau      A2: one stream, s in {1,2,4,9,17}: run_block_decode_row (eager) + RowGraphRunner
           (graphed); r_s = tau(s)/tau(1) within the same mode.
  prefill  A3: run_block_prefill_b chunk n x start offset -> tok/s per layer.
  kv       A4: index-copy KV positions [F-D, F) of every layer to pinned host and back.

  PYTHONPATH=engines/minimax_m25:engines/kimi_k3:phase0:shard:. M25_BATCH=32 M25_KV_MAXLEN=9000 \
    M25_DIR=/root/m25 python -u research/hw1/m25_curve.py --mode curve --ctxs 8192 --out results/a1_8k.json

`load_stage()` is the only import of m25_stage; the CPU tests swap in a tiny fake module.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hw1_common as C  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ANCHOR = {(16384, 4): (42.0, 26.0), (16384, 8): (71.0, 41.0)}     # (ctx,B) -> (per-stream, batched) ms
ANCHOR_TOL = 0.20
MIXED_MIN_MAXLEN = 16384                                           # mixed-context variant needs the tail room


def load_stage():
    import m25_stage as S
    return S


# ── A0: anchor ──────────────────────────────────────────────────────────────────────────────────
_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x")


def parse_anchor(text):
    """rows of the bench's throughput table -> {(ctx, B): (per_stream_ms, batched_ms)}."""
    out = {}
    for line in text.splitlines():
        m = _ROW.match(line)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = (float(m.group(3)), float(m.group(4)))
    return out


def anchor(res, bench_cmd=None, timeout=3600):
    cmd = bench_cmd or [sys.executable, "-u", os.path.join(REPO, "research", "m25_batched_moe_bench.py"),
                        "--layers", "12", "--lo", "24", "--ctxs", "16384", "--bs", "4,8"]
    env = dict(os.environ)
    env.setdefault("M25_BATCH", "8")
    env.setdefault("M25_KV_MAXLEN", "17000")
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    rows = parse_anchor(p.stdout)
    res.doc["derived"]["anchor_stdout_tail"] = p.stdout[-4000:]
    ok_all = p.returncode == 0 and bool(rows)
    for (ctx, B), (want_ps, want_bt) in ANCHOR.items():
        got = rows.get((ctx, B))
        if got is None:
            res.record("A0", "anchor", {"ctx": ctx, "B": B}, skipped="row missing from bench output")
            ok_all = False
            continue
        ps, bt = got
        res.record("A0", "anchor", {"ctx": ctx, "B": B, "layers": 12}, per_stream_ms=ps, batched_ms=bt,
                   want_per_stream_ms=want_ps, want_batched_ms=want_bt,
                   timing="bench's own time.time()+synchronize, warmup 3, iters 8")
        ok_all &= abs(ps / want_ps - 1) <= ANCHOR_TOL and abs(bt / want_bt - 1) <= ANCHOR_TOL
    res.check("K1", ok_all, f"anchor within {ANCHOR_TOL:.0%} (rc={p.returncode}); >20% off -> STOP")
    return ok_all


# ── shared stage fixtures ───────────────────────────────────────────────────────────────────────
def _torch():
    import torch
    return torch


def _load_layers(S, lo, n):
    torch = _torch()
    layers = [S.Layer(i) for i in range(lo, lo + n)]
    S.get_pe()
    gb = torch.cuda.memory_allocated() / 1e9 if S.dev != "cpu" and torch.cuda.is_available() else 0.0
    return layers, gb


def _x(S, B, s, seed=0):
    torch = _torch()
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(B, s, S.H, generator=g) * 0.1).to(torch.bfloat16).to(S.dev)


def kv_row_bytes(S, maxlen, n_layers):
    """bytes of ONE batch row's static K+V over the stage (what M25_BATCH multiplies)."""
    per = 1 if S.M25_KV_FP8 else 2
    return 2 * S.NKV * maxlen * S.HD * per * n_layers


def _oom_types():
    torch = _torch()
    return (torch.cuda.OutOfMemoryError,) if hasattr(torch.cuda, "OutOfMemoryError") else (RuntimeError,)


def _fits(S, B, ctx, s):
    if B > S.M25_BATCH:
        return f"B={B} > M25_BATCH={S.M25_BATCH} (KV rows are allocated at import)"
    if ctx + s + 1 > S.M25_KV_MAXLEN:
        return f"ctx {ctx}+{s} > M25_KV_MAXLEN {S.M25_KV_MAXLEN}"
    return None


# ── A1: service curve ───────────────────────────────────────────────────────────────────────────
def curve(res, S, layers, ctxs, bs, ss, moes, graphed_modes, warmup, iters, mixed=(8, 16), seed=0):
    torch = _torch()
    nL = len(layers)
    for ctx in ctxs:
        for B in bs:
            for s in ss:
                why = _fits(S, B, ctx, s)
                for moe in moes:
                    for graphed in graphed_modes:
                        prm = {"ctx": ctx, "B": B, "s": s, "moe": moe, "graphed_req": graphed, "layers": nL}
                        if why:
                            res.record("A1", "decode_b", prm, skipped=why)
                            continue
                        S.M25_BATCH_MOE = moe == "batched"
                        starts = torch.full((B,), ctx, dtype=torch.long, device=S.dev)
                        x = _x(S, B, s, seed)
                        ran_graphed = False
                        try:
                            if graphed:
                                gr = S.BatchGraphRunner(layers, S.vllm_ctx(), B, s)
                                alen = gr._bucket(ctx + s)
                                host = [ctx] * B                   # the engine passes a HOST list (no sync)
                                fn = lambda: gr.run(host, x)
                                st = C.time_fn(fn, warmup=warmup, iters=iters, device=S.dev)
                                ran_graphed = alen in gr.graphs     # _manual_ok / capture failure -> eager
                            else:
                                vcfg = S.vllm_ctx()
                                fn = lambda: S.run_block_decode_b(layers, starts, x, vcfg)
                                st = C.time_fn(fn, warmup=warmup, iters=iters, device=S.dev)
                        except _oom_types():
                            if S.dev != "cpu":
                                torch.cuda.empty_cache()
                            res.record("A1", "decode_b", prm, skipped="oom")
                            continue
                        res.record("A1", "decode_b", prm, st, graphed=ran_graphed,
                                   ms_per_layer=st["ms"] / nL, tok_per_ms=B * s / st["ms"])
    # mixed-context variant (lognormal starts, median 3k), eager, batched MoE, s=9
    import random
    rng = random.Random(seed)
    cap = S.M25_KV_MAXLEN - 10
    for B in mixed:
        prm = {"ctx": "lognormal_med3k", "B": B, "s": 9, "moe": "batched", "graphed_req": False, "layers": nL}
        why = _fits(S, B, 0, 9) or (None if cap >= MIXED_MIN_MAXLEN else
                                     f"M25_KV_MAXLEN {S.M25_KV_MAXLEN} < {MIXED_MIN_MAXLEN}: would clip the lognormal tail")
        if why:
            res.record("A1", "decode_b_mixed", prm, skipped=why)
            continue
        import math
        st_list = [int(min(cap, max(256, math.exp(math.log(3000) + rng.gauss(0, 1.0))))) for _ in range(B)]
        S.M25_BATCH_MOE = True
        starts = torch.tensor(st_list, dtype=torch.long, device=S.dev)
        x = _x(S, B, 9, seed)
        vcfg = S.vllm_ctx()
        st = C.time_fn(lambda: S.run_block_decode_b(layers, starts, x, vcfg), warmup=warmup, iters=iters,
                       device=S.dev)
        res.record("A1", "decode_b_mixed", dict(prm, starts=st_list), st, graphed=False,
                   ms_per_layer=st["ms"] / nL)


# ── A2: chunk vs one-token verify ───────────────────────────────────────────────────────────────
def tau(res, S, layers, ctxs, ss, warmup, iters):
    torch = _torch()
    nL = len(layers)
    for ctx in ctxs:
        base = {}
        for graphed in (False, True):
            for s in ss:
                prm = {"ctx": ctx, "s": s, "graphed_req": graphed, "layers": nL}
                why = _fits(S, 1, ctx, s)
                if why:
                    res.record("A2", "decode_row", prm, skipped=why)
                    continue
                x = _x(S, 1, s)
                vcfg = S.vllm_ctx()
                if graphed:
                    gr = S.RowGraphRunner(layers, vcfg, s)
                    before = getattr(S, "_GRAPH_COUNT", 0)
                    st = C.time_fn(lambda: gr.run(0, ctx, x), warmup=warmup, iters=iters, device=S.dev)
                    ran = getattr(S, "_GRAPH_COUNT", 0) > before or bool(getattr(gr, "graphs", {}))
                else:
                    st = C.time_fn(lambda: S.run_block_decode_row(layers, 0, ctx, x, vcfg),
                                   warmup=warmup, iters=iters, device=S.dev)
                    ran = False
                key = (graphed, ran)
                if s == ss[0]:
                    base[key] = st["ms"]
                r_s = st["ms"] / base[key] if key in base else None   # only within the SAME mode
                res.record("A2", "decode_row", prm, st, graphed=ran, r_s=r_s, ms_per_layer=st["ms"] / nL)


# ── A3: prefill ─────────────────────────────────────────────────────────────────────────────────
def prefill(res, S, layers, chunks, offsets, warmup, iters):
    nL = len(layers)
    for off in offsets:
        for n in chunks:
            prm = {"n": n, "start": off, "layers": nL}
            why = _fits(S, 1, off, n)
            if why:
                res.record("A3", "prefill_b", prm, skipped=why)
                continue
            x = _x(S, 1, n)
            vcfg = S.vllm_ctx()
            st = C.time_fn(lambda: S.run_block_prefill_b(layers, 0, off, x, vcfg), warmup=warmup,
                           iters=iters, device=S.dev)
            res.record("A3", "prefill_b", prm, st, tok_per_s=n / st["ms"] * 1e3,
                       layer_tok_per_s=n * nL / st["ms"] * 1e3)


# ── A4: KV mirror export/import ─────────────────────────────────────────────────────────────────
def _kv_bufs(L):
    if getattr(L, "bkc", None) is not None:
        return L.bkc[0:1], L.bvc[0:1]                 # row 0 of the batched static KV
    return L.kc, L.vc


def kv(res, S, layers, F, deltas, warmup, iters):
    torch = _torch()
    nL = len(layers)
    pin = S.dev != "cpu" and torch.cuda.is_available()
    for d in deltas:
        prm = {"F": F, "delta": d, "layers": nL}
        if d > F:
            res.record("A4", "kv_export", prm, skipped=f"delta {d} > F {F}")
            continue
        idx = torch.arange(F - d, F, device=S.dev)
        bufs = [_kv_bufs(L) for L in layers]
        if any(k is None or k.shape[2] < F for k, _ in bufs):
            res.record("A4", "kv_export", prm, skipped="KV buffer shorter than F (raise M25_KV_MAXLEN)")
            continue
        host = [[torch.empty(tuple(k.shape[:2]) + (d,) + tuple(k.shape[3:]), dtype=k.dtype,
                             pin_memory=pin) for k in kvp] for kvp in bufs]
        nbytes = sum(h.numel() * h.element_size() for hp in host for h in hp)

        def d2h():
            for (k, v), (hk, hv) in zip(bufs, host):
                hk.copy_(k.index_select(2, idx), non_blocking=pin)
                hv.copy_(v.index_select(2, idx), non_blocking=pin)

        def h2d():
            for (k, v), (hk, hv) in zip(bufs, host):
                k.index_copy_(2, idx, hk.to(S.dev, non_blocking=pin))
                v.index_copy_(2, idx, hv.to(S.dev, non_blocking=pin))

        for op, fn in (("kv_export", d2h), ("kv_import", h2d)):
            st = C.time_fn(fn, warmup=warmup, iters=iters, device=S.dev)
            res.record("A4", op, prm, st, bytes=nbytes, gb_per_s=nbytes / st["ms"] / 1e6,
                       bytes_per_token_layer=nbytes / d / nL, pinned=pin)


# ── main ────────────────────────────────────────────────────────────────────────────────────────
def ints(s):
    return [int(x) for x in str(s).split(",") if x]


def main(argv=None, stage=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("anchor", "curve", "tau", "prefill", "kv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--lo", type=int, default=24)
    ap.add_argument("--ctxs", default="8192")
    ap.add_argument("--bs", default="1,2,4,8,16,32")
    ap.add_argument("--ss", default=None, help="curve: 1,9; tau: 1,2,4,9,17")
    ap.add_argument("--moe", default="perstream,batched")
    ap.add_argument("--graphed", default="0,1")
    ap.add_argument("--chunks", default="128,512,900,2048")
    ap.add_argument("--offsets", default="0,8192,30720")
    ap.add_argument("--F", type=int, default=32768)
    ap.add_argument("--deltas", default="16,128,1024,32768")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--bench-cmd", default=None, help="anchor: override the bench command (tests)")
    ap.add_argument("--stub", action="store_true", help="mark results as stub (CPU tests)")
    a = ap.parse_args(argv)
    if a.mode == "anchor":
        res = C.Results("m25_curve", "anchor", stub=a.stub, argv=argv or sys.argv[1:])
        ok = anchor(res, a.bench_cmd.split() if a.bench_cmd else None)
        res.write(a.out)
        return 0 if ok else 2
    S = stage or load_stage()
    if S.M25_BATCH < 2:     # decode_row / prefill_b / the static KV all live in the [B,...] batched rows
        sys.exit("set M25_BATCH>=2 (and >= max --bs): the batched KV rows are allocated at import")
    res = C.Results("m25_curve", a.mode, device=S.dev, stub=a.stub, argv=argv or sys.argv[1:])
    layers, gb = _load_layers(S, a.lo, a.layers)
    res.doc["derived"].update(weights_gb=gb, layers=a.layers, lo=a.lo, M25_BATCH=S.M25_BATCH,
                              M25_KV_MAXLEN=S.M25_KV_MAXLEN, M25_KV_FP8=bool(S.M25_KV_FP8),
                              kv_row_gb=kv_row_bytes(S, S.M25_KV_MAXLEN, a.layers) / 1e9)
    if a.mode == "curve":
        curve(res, S, layers, ints(a.ctxs), ints(a.bs), ints(a.ss or "1,9"), a.moe.split(","),
              [bool(int(g)) for g in a.graphed.split(",")], a.warmup, a.iters)
    elif a.mode == "tau":
        tau(res, S, layers, ints(a.ctxs), ints(a.ss or "1,2,4,9,17"), a.warmup, a.iters)
    elif a.mode == "prefill":
        prefill(res, S, layers, ints(a.chunks), ints(a.offsets), a.warmup, a.iters)
    elif a.mode == "kv":
        kv(res, S, layers, a.F, ints(a.deltas), a.warmup, a.iters)
    res.write(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
