"""HW1 results -> the sims' --calib input, plus the pre-registered gates K1-K9 (BUILD-PLAN §4.2/§4.3).

  python research/hw1/calib_fit.py results/*.json --out calib/hw1.json
  python tools/pipespec_sim.py --calib calib/hw1.json --part mimo,multi
  python tools/mesh_calc.py --calib calib/hw1.json

Output schema `shard-calib/1`:
  pipespec.rows_table   {rows: cost(rows)/cost(1)} for ONE stage forward -- replaces Ring.mult():
                        rows <= 17 from A2 (one stream, s rows; graphed if it ran graphed), rows above
                        from A1 (B streams x s=9, batched MoE, eager) spliced on at the largest A2 s
  pipespec.r9, m25_ms_per_layer_s9 (A2, same mode as rows_table), kfac (MiMo op-sum / A1 eager B=1 s=9)
  mesh.replay_rate_tok_s (MiMo 8-layer span at prefill shapes), mirror_gb_per_s (A4 import),
       c_layer_ms (MiMo per-layer at s=9)
  gates.K1..K9          {value, threshold, pass|null (null = not measured)}
Every value records the file it came from; a missing input leaves the key null (the sim keeps its
EST constant) -- never a silent default. Stub inputs are refused unless --allow-stub.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hw1_common as C  # noqa: E402

CALIB_SCHEMA = "shard-calib/1"
MIMO_MIRROR_BYTES_32K = 1.5 * 4 * (192 + 128) * 2 * 32768     # 1.5 global layers/8-layer span, bf16 K+V


def _recs(docs, script=None, mode=None, op=None):
    for path, d in docs:
        if (script and d["script"] != script) or (mode and d["mode"] != mode):
            continue
        for r in d["records"]:
            if r.get("skipped") or (op and r["op"] != op):
                continue
            yield path, r


def _pick_ctx(ctxs, want=8192):
    return min(ctxs, key=lambda c: abs(c - want)) if ctxs else None


def _interp(table, x):
    ks = sorted(table)
    if x <= ks[0]:
        return table[ks[0]]
    for a, b in zip(ks, ks[1:]):
        if a <= x <= b:
            t = (math.log(x) - math.log(a)) / (math.log(b) - math.log(a))
            return table[a] + t * (table[b] - table[a])
    a, b = ks[-2], ks[-1]                                          # linear extrapolation
    return table[b] + (table[b] - table[a]) / (b - a) * (x - b)


def fit(docs):
    src, gates = {}, {}
    pip = {"rows_table": None, "r9": None, "m25_ms_per_layer_s9": None, "kfac": None}
    mesh = {"replay_rate_tok_s": None, "mirror_gb_per_s": None, "c_layer_ms": None}

    # ── A2: tau table (prefer the mode that actually ran graphed at every s) ──
    a2 = list(_recs(docs, "m25_curve", "tau", "decode_row"))
    ctx = _pick_ctx(sorted({r["params"]["ctx"] for _, r in a2}))
    tau_table, tau_mode = None, None
    for graphed in (True, False):
        pts = {r["params"]["s"]: (p, r) for p, r in a2 if r["params"]["ctx"] == ctx and r["graphed"] == graphed}
        if 1 in pts and len(pts) >= 2:
            base = pts[1][1]["ms"]
            tau_table = {s: r["ms"] / base for s, (p, r) in pts.items()}
            tau_mode = "graphed" if graphed else "eager"
            pip["m25_ms_per_layer_s9"] = pts[9][1]["ms_per_layer"] if 9 in pts else None
            src["tau"] = sorted({p for p, _ in pts.values()})
            break
    if tau_table:
        pip["r9"] = tau_table.get(9)
        src["tau_mode"] = tau_mode
        src["tau_ctx"] = ctx
        gates["K3"] = {"value": pip["r9"], "threshold": "<= 2.5 (>= 3 -> pertoken default)",
                       "pass": None if pip["r9"] is None else pip["r9"] <= 2.5}

    # ── A1: service curve (eager, batched MoE, like-for-like) ──
    a1 = [(p, r) for p, r in _recs(docs, "m25_curve", "curve", "decode_b") if not r["graphed"]]
    c1 = _pick_ctx(sorted({r["params"]["ctx"] for _, r in a1}))
    bat = {(r["params"]["B"], r["params"]["s"]): r for _, r in a1
           if r["params"]["ctx"] == c1 and r["params"]["moe"] == "batched"}
    if (1, 1) in bat:
        base = bat[(1, 1)]["ms"]
        batch_table = {B * s: r["ms"] / base for (B, s), r in bat.items()}
        src["curve"] = sorted({p for p, _ in a1})
        src["curve_ctx"] = c1
        if tau_table:
            top = max(tau_table)
            rows = dict(tau_table)
            for k, v in batch_table.items():
                if k > top:
                    rows[k] = tau_table[top] * v / _interp(batch_table, top)
        else:
            rows = batch_table
        pip["rows_table"] = {str(k): v for k, v in sorted(rows.items())}
        if (16, 9) in bat and (1, 9) in bat:
            gain = (16 * 9 / bat[(16, 9)]["ms"]) / (9 / bat[(1, 9)]["ms"])
            gates["K2"] = {"value": gain, "threshold": ">= 3.5x tokens/ms (2.5-3.5 cap S 16; < 2.5 fork)",
                           "pass": gain >= 3.5}
    k4 = {r["params"]["ctx"]: r["ms_per_layer"] for _, r in a1
          if r["params"]["B"] == 1 and r["params"]["s"] == 1 and r["params"]["moe"] == "perstream"}
    lo_c, hi_c = (min(k4), max(k4)) if k4 else (None, None)
    if k4 and hi_c >= 4 * lo_c:
        v = k4[hi_c] / k4[lo_c]
        gates["K4"] = {"value": v, "threshold": f"ctx {hi_c} <= 2x ctx {lo_c} per layer", "pass": v <= 2.0}

    # ── A0 ──
    for path, d in docs:
        if d["script"] == "m25_curve" and d["mode"] == "anchor":
            k1 = [c for c in d["checks"] if c["id"] == "K1"]
            if k1:
                gates["K1"] = {"value": k1[0]["detail"], "threshold": "within 20%", "pass": k1[0]["pass"]}

    # ── B: MiMo ──
    for path, d in docs:
        if d["script"] != "mimo_ops" or d["mode"] != "ops":
            continue
        dv = d["derived"]
        if dv.get("replay_tok_s_span"):
            mesh["replay_rate_tok_s"] = dv["replay_tok_s_span"]
            src["replay"] = path
            r = dv["replay_tok_s_span"]
            gates["K5"] = {"value": r, "threshold": ">= 4000 tok/s (1500-4000 k>=3,S<=16; <1000 drop claim)",
                           "pass": r >= 4000}
        if dv.get("layer_ms_T9"):
            mesh["c_layer_ms"] = dv["layer_ms_T9"]
            m25 = bat.get((1, 9)) if a1 else None
            if m25:
                pip["kfac"] = dv["layer_ms_T9"] / m25["ms_per_layer"]
                src["kfac"] = [path, "A1 eager B=1 s=9 ctx %s" % c1]
                gates["K6"] = {"value": pip["kfac"], "threshold": "<= 2.0 (> 3 re-plan)", "pass": pip["kfac"] <= 2.0}
        fp = [r for r in d["records"] if r["op"] == "footprint_experts" and r.get("resident_bytes")]
        if fp:
            worst = max(r["resident_bytes"] / r["disk_bytes"] for r in fp)
            gates.setdefault("K7", {"value": {}, "threshold": "<= 1.1x on-disk", "pass": True})
            fam = d["device"]["family"]
            gates["K7"]["value"][fam] = worst
            gates["K7"]["pass"] = gates["K7"]["pass"] and worst <= 1.1 and all(
                r.get("backend") == "marlin" for r in fp)

    # ── A4 / C: mirror, K8, K9 ──
    imp = [r for _, r in _recs(docs, "m25_curve", "kv", "kv_import")]
    if imp:
        big = max(imp, key=lambda r: r["params"]["delta"])
        mesh["mirror_gb_per_s"] = big["gb_per_s"]
        t = MIMO_MIRROR_BYTES_32K / (big["gb_per_s"] * 1e9)
        gates["K9"] = {"value": t, "threshold": "<= 0.5 s (32k session, 8-layer span import)", "pass": t <= 0.5}
    sha = {r["params"]["kib"]: r for _, r in _recs(docs, "sys_bench", None, "sha256")}
    frm = {r["params"]["n"]: r for _, r in _recs(docs, "sys_bench", None, "frame_rt")}
    if 8 in sha and 9 in frm and (16, 9) in bat:
        ovh_ms = (16 * 9 * sha[8]["us_per_row"] + 16 * frm[9]["us_per_msg"]) / 1e3
        frac = ovh_ms / bat[(16, 9)]["ms"]
        gates["K8"] = {"value": frac, "threshold": "<= 2% of A1 service time at B=16", "pass": frac <= 0.02}
    return {"schema": CALIB_SCHEMA, "pipespec": pip, "mesh": mesh, "gates": gates, "source": src}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-stub", action="store_true", help="accept stub results (tests only)")
    a = ap.parse_args(argv)
    docs = [(p, C.load(p)) for p in a.results]
    stubs = [p for p, d in docs if d["stub"]]
    if stubs and not a.allow_stub:
        sys.exit(f"refusing stub results (CPU-test numbers) as calibration: {stubs}")
    cal = fit(docs)
    cal["stub"] = bool(stubs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(cal, f, indent=1, sort_keys=True)
    for k, g in sorted(cal["gates"].items()):
        print(f"{k}: {'PASS' if g['pass'] else ('FAIL' if g['pass'] is False else 'n/a')}  "
              f"{g['value']}  [{g['threshold']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
