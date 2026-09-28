#!/usr/bin/env python3
"""Mesh analysis for design-mesh.md (workstream 3), promoted from scratchpad/v2 for HW1 (H0).
Pure stdlib, CPU only.

Every constant is tagged MEAS (with source) or EST. Run: python3 tools/mesh_calc.py [--calib hw1.json]
(--calib adds the HW1-measured replay rate as a MEAS row and the measured mirror import time)
Sections: footprint -> hop law -> replay stall (single, then MASS under load) -> exposure -> economics.
"""
import math
import random

# ---------------------------------------------------------------- model facts (HF config.json, fetched 2026-09-28)
MIMO_FLASH = dict(name="MiMo-V2.6-Flash", layers=48, hidden=4096, full_attn_layers=9,
                  kv_heads_full=4, k_dim=192, v_dim=128, total_gb=173.0)   # total EST (PROGRAM.md)
MIMO_PRO = dict(name="MiMo-V2.6-Pro", layers=70, hidden=6144, full_attn_layers=10,
                kv_heads_full=8, k_dim=192, v_dim=128, total_gb=566.0)     # total EST (PROGRAM.md)
M25 = dict(name="MiniMax-M2.5", layers=62, hidden=3072, gb_per_layer=2.33)  # MEAS ADMISSION_SPEC


def gb_per_layer(m):
    return m["gb_per_layer"] if "gb_per_layer" in m else m["total_gb"] / m["layers"]


def kv_bytes_per_token(m, layers=None):
    """bf16 KV of full-attention layers per token (SWA layers are O(window), constant)."""
    frac = 1.0 if layers is None else layers / m["layers"]
    return m["kv_heads_full"] * (m["k_dim"] + m["v_dim"]) * 2 * m["full_attn_layers"] * frac


def layers_held(vram_gb, m, reserve_gb=1.5, kv_gb=1.0):
    return min(m["layers"], max(0, int((vram_gb - reserve_gb - kv_gb) // gb_per_layer(m))))


# ---------------------------------------------------------------- hop law
# L = (hops+1) legs * h  +  layers * c_layer  +  tail_extra ;  tok/s = useful_inflight / L
USEFUL_INFLIGHT = 4.6   # MEAS docs/V4_FLASH_ENGINE.md (block 8 floor 1: 4.64)
C_LAYER_MS = 1.15       # MEAS-derived: V4 s0-s4 on-box 8.3-10.0 ms for 8 layers
TAIL_EXTRA_MS = 11.45   # MEAS V4 tail: draft 9.16 + head/logits ~2.3 (DFlash would be ~3 + 2.3, WS2 EST)
V4_L_MS, V4_LAYERS, V4_HOPS = 152.0, 43, 6
H_V4_FIT = (V4_L_MS - V4_LAYERS * C_LAYER_MS - TAIL_EXTRA_MS) / (V4_HOPS + 1)  # per-leg ms FITTED to V4


def session_L(n_hops, h_ms, layers, extra_leg_ms=0.0):
    return (n_hops + 1) * h_ms + extra_leg_ms + layers * C_LAYER_MS + TAIL_EXTRA_MS


# ---------------------------------------------------------------- replay (single session)
LOG_BYTES_PER_POS = lambda m: 2 * m["hidden"]   # fp8 hidden row + fp8 fused-tap row (WS2 §4.2)
FIXED_OVERHEAD_S = 1.0   # EST: rollback round + lease + path set-up
DETECT_HARD_S = 2.0      # EST: transport error + confirm
DETECT_MEAS_S = 4.3      # MEAS hot-standby-failover-20260623 (heartbeat harness)
# replay prefill rate for an 8-layer MiMo-Flash span on a 5090:
#   LB  1000 = M2.5 MEAS bound (30k in 24.0 s through 5 stages ~12.4 L each => >=1250 tok/s/stage
#              => >=15.5k layer-tok/s) scaled by ~1.9x MiMo active params per layer => ~8.2k/8 L
#   MID 4000 EST, HIGH 16000 EST (roofline-ish, 15B-active MoE)
RATES = {"LB(M2.5 proxy) 1000": 1000.0, "MID 4000 (EST)": 4000.0, "HIGH 16000 (EST)": 16000.0}


def replay_single_s(ctx, rate, uplink_mbps, m=MIMO_FLASH, detect=DETECT_HARD_S, chunk=2048):
    xfer = ctx * LOG_BYTES_PER_POS(m) * 8 / (uplink_mbps * 1e6)
    comp = ctx / rate
    n = max(1, math.ceil(ctx / chunk))
    return detect + FIXED_OVERHEAD_S + max(xfer, comp) + min(xfer, comp) / n


# ---------------------------------------------------------------- mass replay under load
def sample_ctx(rng):
    """EST context distribution: lognormal, median 3k, sigma 1.0 (p90 ~11k, p99 ~30k), capped 128k."""
    return int(min(131072, max(256, math.exp(math.log(3000) + rng.gauss(0, 1.0)))))


SWA_RF_POS = 900   # EST: positions an 8-layer Flash span must recompute to rebuild its SWA windows after a
                   # full-attention-KV-mirror takeover (128 x ~7 chained SWA layers), plus mirror lag below
MIRROR_LAG_POS = 16


def mass_replay(S, k, phi, rate, uplink_mbps, k_up=2, detect=DETECT_HARD_S, trials=400, seed=3, mirror=False):
    """A holder serving S sessions dies. Sessions spread evenly over (k-1) alternates; each alternate gives
    replay a GPU share phi and processes its replays shortest-first. Log transfers come from the upstream
    holders (k_up of them), each uplink shared by the transfers it serves. Returns stall p50/p99 over all
    displaced sessions."""
    rng = random.Random(seed)
    stalls = []
    alts = max(1, k - 1)
    for _ in range(trials):
        ctxs = [sample_ctx(rng) for _ in range(S)]
        if mirror:   # full-attention KV already on the backup; only SWA windows + lag are recomputed
            ctxs = [min(c, SWA_RF_POS) + MIRROR_LAG_POS for c in ctxs]
        per_alt = [[] for _ in range(alts)]
        for i, c in enumerate(ctxs):
            per_alt[i % alts].append(c)
        share = max(1, math.ceil(S / k_up))       # concurrent transfers per upstream uplink
        for q in per_alt:
            q.sort()
            cum = 0.0
            for c in q:
                cum += c / (rate * phi)
                xfer = c * LOG_BYTES_PER_POS(MIMO_FLASH) * 8 / (uplink_mbps * 1e6 / share)
                stalls.append(detect + FIXED_OVERHEAD_S + max(cum, xfer))
    stalls.sort()
    return stalls[len(stalls) // 2], stalls[int(0.99 * len(stalls))]


# ---------------------------------------------------------------- exposure
def p_session_hit(n_nodes, session_min, node_mtbf_h):
    return 1 - math.exp(-n_nodes / (node_mtbf_h * 60.0) * session_min)


# ---------------------------------------------------------------- economics
def node_usd_per_hr(agg_out, frac_layers, p_out, p_in_eff, in_out, share=0.70):
    return agg_out * 3600 * (p_out + in_out * p_in_eff) / 1e6 * frac_layers * share


def prefill_adjusted(agg_decode, uncached_in_per_out, span_prefill_rate):
    """Conservative: decode at agg_decode saturates the stage; prefill of the uncached input competes
    for the same stage time. A_eff = 1 / (1/A_dec + u/R)."""
    return 1.0 / (1.0 / agg_decode + uncached_in_per_out / span_prefill_rate)


MIRROR_GBPS = None      # --calib: measured KV import GB/s (HW1 A4)


def apply_calib(path):
    import json
    global MIRROR_GBPS
    cal = json.load(open(path))
    if cal.get("schema") != "shard-calib/1":
        raise SystemExit(f"{path}: not a shard-calib/1 file")
    m = cal["mesh"]
    if m.get("replay_rate_tok_s"):
        RATES["MEAS (HW1)"] = float(m["replay_rate_tok_s"])
    if m.get("mirror_gb_per_s"):
        MIRROR_GBPS = float(m["mirror_gb_per_s"])
    print(f"CALIBRATED from {path}{' (STUB DATA)' if cal.get('stub') else ''}: replay "
          f"{m.get('replay_rate_tok_s')} tok/s, mirror {MIRROR_GBPS} GB/s")
    return cal


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=None, help="shard-calib/1 JSON from research/hw1/calib_fit.py")
    a = ap.parse_args(argv)
    if a.calib:
        apply_calib(a.calib)
    bar = "=" * 100
    print(bar + "\n1. FOOTPRINT (EST gb/layer from totals; reserve 1.5 GB + 1 GB KV)")
    for m in (MIMO_FLASH, MIMO_PRO, M25):
        row = [f"{v}GB:{layers_held(v, m)}" for v in (8, 12, 16, 24, 32, 48, 96, 128, 192, 256, 512)]
        print(f"  {m['name']:<18} {gb_per_layer(m):5.2f} GB/L -> " + " ".join(row))
    for m in (MIMO_FLASH, MIMO_PRO):
        kv = kv_bytes_per_token(m)
        print(f"  {m['name']} full-attn KV bf16 {kv/1024:.1f} KB/token -> 32k = {kv*32768/2**20:.0f} MB/session; "
              f"log row (fp8 hidden + fp8 tap) {LOG_BYTES_PER_POS(m)/1024:.0f} KB/position")

    print(bar + f"\n2. HOP LAW  L = (hops+1)*h + layers*{C_LAYER_MS} + {TAIL_EXTRA_MS} ms; tok/s = 4.6/L")
    print(f"  h fitted to V4 (152 ms, 6 hops, 43 L): {H_V4_FIT:.1f} ms/leg.  Independent M2.5 MEAS per-hop "
          f"RTT_eff: 19 ms tight, 30 ms wide (ADMISSION_SPEC). 6 ms 'metro' has NO receipt.")
    print(f"  check: V4 -> {USEFUL_INFLIGHT/session_L(6, H_V4_FIT, 43)*1000:.1f} tok/s (by construction; MEAS 30.15)")
    rows = [("Flash 6x5090", 6, 48), ("Flash 16x16GB (3 L)", 16, 48), ("Flash 10x24GB (5 L)", 10, 48),
            ("Flash 48x8GB (1 L)", 48, 48), ("Flash 2x96GB", 2, 48), ("Pro 7x96GB", 7, 70),
            ("Pro 24x32GB", 24, 70)]
    print(f"  {'path':<22}" + "".join(f"{f'h={h}':>14}" for h in (6, round(H_V4_FIT, 1), 19, 30)))
    for label, n, layers in rows:
        cells = "".join(f"{USEFUL_INFLIGHT/session_L(n, h, layers)*1000:>9.1f} t/s " for h in (6, H_V4_FIT, 19, 30))
        print(f"  {label:<22}{cells}")
    cc = USEFUL_INFLIGHT / session_L(6, H_V4_FIT, 48, extra_leg_ms=(90 - 2 * H_V4_FIT) / 2) * 1000
    print(f"  Flash 6x5090 with one leg replaced by a 90 ms-RTT cross-cell leg (h=45): {cc:.1f} t/s")

    print(bar + "\n3a. SINGLE-SESSION REPLAY STALL, 8-layer Flash span (detect 2.0 s EST + 1.0 s fixed EST)")
    for rl, rate in RATES.items():
        for up in (30, 200, 1000):
            cells = "  ".join(f"{c//1000:>3}k:{replay_single_s(c, rate, up):6.1f}s" for c in (1000, 4000, 16000, 32000, 128000))
            print(f"   {rl:<20} uplink {up:>4} Mbps: {cells}")
    print(f"   per-1k-token span compute: " + ", ".join(f"{rl.split()[0]} {1000/r:.2f} s" for rl, r in RATES.items()))
    print("   MEAS reference: heal_hot whole-ring (gpt-oss, 4x4090) detect 4.3 s + re-prefill 8.9 s for ~447 tok"
          " -> today's code has multi-second fixed costs; the model's fixed term must be re-fit on M1")

    print(bar + "\n3b. MASS REPLAY: a holder with S sessions dies (ctx ~ lognormal median 3k), uplink 200 Mbps")
    print(f"   {'S':>3} {'k':>2} {'phi':>5} {'rate':>6}   stall p50 / p99 over displaced sessions")
    for S in (4, 16, 32):
        for k in (2, 3):
            for phi in (0.15, 0.5):
                for rate in sorted({1000.0, 4000.0} | {v for k_, v in RATES.items() if k_.startswith("MEAS")}):
                    p50, p99 = mass_replay(S, k, phi, rate, 200)
                    print(f"   {S:>3} {k:>2} {phi:>5} {int(rate):>6}   {p50:6.1f} s / {p99:6.1f} s")

    print("   --- with full-attention KV MIRRORED to the backup holder (takeover recomputes SWA windows only):")
    for S in (16, 32):
        for k in (2, 3):
            p50, p99 = mass_replay(S, k, 0.5, 4000.0, 200, mirror=True)
            q50, q99 = mass_replay(S, k, 0.5, 1000.0, 200, mirror=True)
            print(f"   {S:>3} {k:>2}   0.5  rate 4000: {p50:5.1f} s / {p99:5.1f} s    rate 1000: {q50:5.1f} s / {q99:5.1f} s")
    fa = 2560 * 1.5 / 1024; swa = 5120 * 6.5 / 1024
    print(f"   mirror bandwidth per session-hop at 30 tok/s: full-attn only {fa:.1f} KB/tok -> {fa*30*8/1024:.2f} Mbps;"
          f" + SWA rows {fa+swa:.1f} KB/tok -> {(fa+swa)*30*8/1024:.1f} Mbps (bf16, 8-layer span, EST 1.5 GA layers)")

    if MIRROR_GBPS:
        mb = 1.5 * kv_bytes_per_token(MIMO_FLASH) / MIMO_FLASH["full_attn_layers"] * 32768
        print(f"   MEAS mirror import (HW1 A4 {MIRROR_GBPS:.1f} GB/s): 32k session, 8-layer span "
              f"{mb/1e6:.0f} MB -> {mb/MIRROR_GBPS/1e9:.3f} s")
    print(bar + "\n4. EXPOSURE: P(session sees >=1 path-node failure), 6-node path")
    for mtbf in (4, 12, 48):
        print(f"   MTBF {mtbf:>2} h  " + "  ".join(f"{mins:>3}min:{100*p_session_hit(6, mins, mtbf):5.1f}%"
                                               for mins in (1, 5, 20, 60)))

    print(bar + "\n5. ECONOMICS per node $/h (share 70%; in:out 15:1, 60% of input billed as cache reads)")
    prices = {"OR MiMo-V2.6-Flash 0.14/0.28 c0.003": (0.28, 0.14, 0.003),
              "OR MiniMax-M2.5 0.27/1.08 c0.027": (1.08, 0.27, 0.027),
              "c0mpute premium EST 0.50/2.00 c0.05": (2.00, 0.50, 0.05)}
    unc = 15 * 0.4   # uncached input tokens per output token that must be prefilled
    for pl, (po, pi, pc) in prices.items():
        p_in_eff = 0.4 * pi + 0.6 * pc
        for al, a in (("111.5 (M2.5 proxy, MEAS B=8)", 111.5), ("180 (EST B=16)", 180.0)):
            for rl, rate in (("pf LB", 1000.0), ("pf MID", 4000.0), ("decode-only", None)):
                a_eff = a if rate is None else prefill_adjusted(a, unc, rate)
                print(f"   {pl:<37} {al:<30} {rl:<11} A_eff {a_eff:6.1f}  5090(8/48) "
                      f"${node_usd_per_hr(a_eff, 8/48, po, p_in_eff, 15):.3f}/h  96GB(25/48) "
                      f"${node_usd_per_hr(a_eff, 25/48, po, p_in_eff, 15):.3f}/h")
    for gpu, w in (("5090", (300, 450)), ("4060", (80, 115))):
        for kwh in (0.10, 0.30):
            print(f"   electricity {gpu} {w[0]}-{w[1]} W @ ${kwh:.2f}/kWh = ${w[0]*kwh/1000:.3f}-{w[1]*kwh/1000:.3f}/h")
    print("   break-even aggregate (decode-only) for a 5090 holding 8/48 L at 400 W, $0.30/kWh:")
    for pl, (po, pi, pc) in prices.items():
        per = node_usd_per_hr(1.0, 8 / 48, po, 0.4 * pi + 0.6 * pc, 15)
        print(f"     {pl:<37} {0.12/per:6.0f} out tok/s through its layers")
    for label, n, kw, a in (("6x5090 path", 6, 0.4, 111.5), ("2xPro6000 path", 2, 0.6, 160.6)):
        for kwh in (0.10, 0.30):
            print(f"   electricity floor {label} @ {a} agg, ${kwh}/kWh: ${n*kw*kwh/(a*3600/1e6):.2f} per M output tokens"
                  f" (decode-only; prefill adds more)")
    for po in (0.05, 0.10, 0.20):
        for a in (150, 400):
            print(f"   4060 whole-server 9B EST: {a} agg @ ${po}/M out -> "
                  f"${node_usd_per_hr(a, 1.0, po, po/3*0.4 + po/60*0.6, 15):.3f}/h")


if __name__ == "__main__":
    main()
