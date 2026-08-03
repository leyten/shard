#!/usr/bin/env python3
"""Sweep the fp8 GEMV tiles on the card: GB/s per (shape, tile), equality-gated, one recommendation.

WHY THIS EXISTS. At decode every V4 layer runs seven single-row fp8 GEMMs through `model.linear`,
and the vendored kernel's grid — `ceildiv(N,128)` blocks of 128 threads at M=1 — leaves most of
them at 4-32 blocks on a 170-SM card. The 08-01 ring measured the shared expert's three launches at
~70-160 GB/s against the 969 GB/s the SAME card reaches on the grouped fp4 kernel's 192-block grid.
phase0/v4_fp8_gemv re-tiles the same kernel (block_N, num_stages, threads — grid/pipeline shape
only) and gates every tile on `torch.equal`; this bench is where the tile CHOICE comes from.

WHAT IT MEASURES. Every shipped decode GEMV shape, at M=1, for each candidate tile:

    equal      torch.equal against the VENDORED kernel on seeded full-range inputs (every non-NaN
               fp8 weight byte, e8m0 scales), the same gate `_probe_gemv` enforces at serve time.
               A config that is not bit-exact is priced at n/a and can never be recommended.
    us/call    CUDA-event time over --reps calls, 20 discarded warmup (the tilelang JIT lands there).
    GB/s       bytes the call must move (weight + scales + activation + output) over that time.

The per-shape recommendation is the fastest EQUAL tile; `auto` in the summary is what
`V4_FP8_GEMV=1` would pick for that shape, so the gap between the two columns is what a per-shape
override table would still buy. The fused shared-expert launch is measured against its own
two-launch baseline (w1 + w3 separately at the same tile), pricing V4_FP8_SHARED's half.

CAVEATS, so the number is read honestly: this is the kernel ALONE on an idle card — no act_quant,
no graph replay, no neighbours competing for the memory system — so the predicted per-layer saving
is an upper bound. And it is one card, one clock state: run it on a box that will serve; serve-time
still re-proves every tile per (N, K) on that box, this bench choosing one is advice.

Needs a CUDA device. Usage:
    python3 research/v4_fp8_gemv_bench.py [--reps 200] [--json out.json]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase0"))

import v4_fp8_gemv as FG  # noqa: E402

# The decode step's per-layer fp8 GEMMs at the shipped dims (config.json: dim 4096, inter 2048,
# q_lora 1024, o_groups*o_lora 8192, 64x512 attention heads, 64x128 indexer heads).
SHAPES = (
    ("attn.wq_a", 1024, 4096),
    ("attn.wkv", 512, 4096),
    ("attn.wq_b", 32768, 1024),
    ("attn.wo_b", 4096, 8192),
    ("idx.wq_b", 8192, 1024),
    ("shared.w1", 2048, 4096),      # one of the two unfused shared launches V4_FP8_SHARED removes
    ("shared.w13", 4096, 4096),     # the fused replacement: w1|w3 as one bank
    ("shared.w2", 4096, 2048),
)
CANDIDATES = (None, "64,4", "64,2", "32,4", "32,2", "16,4", "16,2", "16,6",
              "16,4,256", "32,4,256", "16,4,64")
M = 1                               # the decode shape; the serve gate also probes 8 and 32
TL = "float8_e8m0fnu"               # the ring's scale dtype (config scale_fmt=ue8m0)


def call_bytes(N, K):
    """fp8 weight + e8m0 weight scales + fp8 activation + e8m0 act scale + bf16 output."""
    return (N * K + ((N + 127) // 128) * ((K + 127) // 128) + K + (K + 127) // 128 + N * 2)


def bench(fn, reps):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps * 1e3               # us/call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "this bench needs the CUDA device it is choosing tiles for"
    torch.set_default_dtype(torch.bfloat16)             # the serve process's output dtype
    dev = torch.cuda.get_device_name(0)
    print(f"device {dev}   reps {a.reps}   M={M}   scales {TL}\n")

    # the vendored kernel, exactly as linear() would call it
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "vendor", "deepseek_v4_ref", "inference"))
    import kernel as vk
    FG._REF_FP8_GEMM = FG._REF_FP8_GEMM or vk.fp8_gemm
    sd = torch.float8_e8m0fnu

    out = {"device": dev, "reps": a.reps, "shapes": {}}
    recos = {}
    for name, N, K in SHAPES:
        aq, a_s, w, w_s = FG._probe_seeded(N, K, TL, M)
        ref = vk.fp8_gemm(aq, a_s, w, w_s, sd)
        base_us = bench(lambda: vk.fp8_gemm(aq, a_s, w, w_s, sd), a.reps)
        gb = call_bytes(N, K)
        rows = [("vendored", True, base_us, gb / base_us / 1e3)]
        best = ("vendored", base_us)
        for cand in CANDIDATES:
            if cand is None:
                continue
            tile, _err = FG._parse_gemv(cand)
            if tile is None or tile == "auto":
                continue
            try:
                got = FG._run_tiled(aq, a_s, w, w_s, sd, tile)
                torch.cuda.synchronize()
                eq = torch.equal(ref, got)
                us = bench(lambda: FG._run_tiled(aq, a_s, w, w_s, sd, tile), a.reps) if eq else None
            except Exception:                            # noqa: BLE001 — a candidate may not compile
                eq, us = False, None
            rows.append((cand, eq, us, gb / us / 1e3 if us else None))
            if eq and us is not None and us < best[1]:
                best = (cand, us)
        auto = FG._auto_tile(N)
        auto_s = "vendored" if auto is None else f"{auto[0]},{auto[1]},{auto[2]}"
        blocks = (N + 127) // 128
        print(f"{name:12s} [{N:5d},{K:4d}]  {blocks:3d} shipped blocks   auto={auto_s}")
        for cand, eq, us, gbs in rows:
            mark = " <- best" if cand == best[0] else ""
            print(f"    {str(cand):10s} equal={str(eq):5s} "
                  f"{f'{us:7.1f} us  {gbs:6.0f} GB/s' if us else '     n/a'}{mark}")
        speed = base_us / best[1]
        print(f"    -> {best[0]} at {best[1]:.1f} us = {speed:.2f}x the vendored grid\n")
        recos[name] = {"best": best[0], "us": best[1], "vendored_us": base_us, "speedup": speed}
        out["shapes"][name] = {"N": N, "K": K, "auto": auto_s,
                               "rows": [{"tile": str(c), "equal": e, "us": u, "gbps": g}
                                        for c, e, u, g in rows], **recos[name]}

    # V4_FP8_SHARED's half: the fused w13 launch vs two separate launches at each side's best tile
    two = recos["shared.w1"]["us"] * 2
    one = recos["shared.w13"]["us"]
    print(f"shared expert w1+w3: two launches {two:.1f} us vs fused {one:.1f} us "
          f"= {two / one:.2f}x (plus one act_quant removed)")
    layer_now = sum(recos[n]["vendored_us"] for n, _, _ in SHAPES if n not in ("shared.w13",))
    layer_best = (sum(recos[n]["us"] for n, _, _ in SHAPES
                      if n not in ("shared.w1", "shared.w13")) + one)
    print(f"per-layer fp8-GEMV time (ratio-4 layer, shared.w1 twice for w1+w3):\n"
          f"    vendored grids  {layer_now + recos['shared.w1']['vendored_us']:7.1f} us\n"
          f"    tiled + fused   {layer_best:7.1f} us")
    out["shared_fused_speedup"] = two / one
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
