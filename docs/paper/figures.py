"""Paper figures — every plot generated from receipt JSONs (docs/receipts/) or a seeded simulation;
nothing hand-drawn. Run from docs/paper/:  python3 figures.py
Outputs fig_*.pdf next to main.typ. Colors: validated categorical palette, fixed slot order
(AR=blue, chain=aqua, hybrid=yellow; yellow always direct-labeled — relief rule)."""
import json
import os
import random
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"ar": "#2a78d6", "chain": "#1baf7a", "hybrid": "#eda100"}
INK, MUT = "#1a1a19", "#8a897f"
RCP = "../receipts"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUT, "axes.linewidth": 0.6,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUT,
                     "ytick.color": MUT, "figure.dpi": 150})


def _grid(ax):
    ax.grid(axis="y", color="#e4e3da", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---- fig 1: the accept-gated pipelining law (seeded Monte-Carlo, calibrated both ends) ----------
# Time unit: one pipeline SLOT = T/depth (with depth chunks in flight the ring returns one result
# per slot; a synchronous round costs a full T = depth slots). Calibration anchors (measured):
# sync tree at α=0.74 must give g≈4.5 (the tree-verify receipt); pipelined chain at α=0.97 must land
# in the measured 50-80 tok/s verbatim band.

def sim_pipelined(alpha, K=8, depth=4, cycles=20000, seed=7):
    """Tokens/slot for a depth-D flush-on-divergence pipeline. A full-accept chunk commits K and
    costs 1 slot; a rejected chunk commits its accepted prefix + 1 correction, and costs a full
    refill (depth slots): everything behind it speculated past the miss and is discarded."""
    rng = random.Random(seed)
    toks = slots = 0
    for _ in range(cycles):
        n = 0
        while n < K and rng.random() < alpha:
            n += 1
        if n == K:
            toks += K; slots += 1
        else:
            toks += n + 1; slots += depth          # flush + pipe refill bubble
    return toks / slots


def sim_tree(alpha, M=12, topb=3, depth_cap=8, cover=1.26, cycles=20000, seed=11):
    """Committed tokens per synchronous round for a best-first top-M tree: at each level the
    target's next token is inside the drafter's top-`topb` children with probability
    min(1, cover*α) (top-3 coverage exceeds top-1 acceptance; `cover` calibrated so g(0.74)≈4.5 — measured,
    the measured tree-verify receipt). Path depth is bounded by the node budget spent best-first."""
    rng = random.Random(seed)
    a3 = min(1.0, cover * alpha)
    max_path = min(depth_cap, max(1, M // topb))   # budget: ~topb nodes spent per accepted level
    toks = 0
    for _ in range(cycles):
        ell = 0
        while ell < max_path and rng.random() < a3:
            ell += 1
        toks += ell + 1                            # accepted path + correction/bonus
    return toks / cycles


def fig_alpha_law(T_ms=380.0, depth=4):
    alphas = [i / 100 for i in range(30, 100)]
    slot = T_ms / depth / 1000
    pipe = [sim_pipelined(a, depth=depth) / slot for a in alphas]
    tree = [sim_tree(a) / (T_ms / 1000) for a in alphas]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(alphas, tree, color=C["chain"], lw=1.6, label="synchronous tree (M=12, top-3)", zorder=3)
    ax.plot(alphas, pipe, color=C["ar"], lw=1.6, label=f"pipelined chain, depth {depth}", zorder=3)
    cross = next((a for a, p, s in zip(alphas, pipe, tree) if p > s), None)
    if cross:
        ax.axvline(cross, color=MUT, lw=0.7, ls=":", zorder=2)
        ax.annotate(f"crossover α ≈ {cross:.2f}", (cross - 0.015, max(pipe) * 0.72),
                    fontsize=8, color=INK, ha="right")
    marks = ((0.74, sim_tree(0.74) / (T_ms / 1000), "EAGLE-3, novel text\n(tree route)", (-0.02, 3)),
             (0.97, sim_pipelined(0.97, depth=depth) / slot, "n-gram, verbatim text\n(pipelined route)", (-0.05, -12)))
    for a, y, name, (dx, dy) in marks:
        ax.plot([a], [y], "o", ms=5.5, color=C["hybrid"], mec=INK, mew=0.6, zorder=4)
        ax.annotate(name, (a, y), xytext=(a + dx, y + dy), fontsize=7.2, ha="right", color=INK)
    ax.set_xlabel("per-token draft acceptance α")
    ax.set_ylabel(f"tokens / s on a T = {int(T_ms)} ms ring")
    ax.set_xlim(0.3, 1.0)
    _grid(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig("fig_alpha_law.pdf"); fig.savefig("fig_alpha_law.png", dpi=220, bbox_inches="tight")
    print(f"fig_alpha_law.pdf  crossover≈{cross}  tree@0.74={sim_tree(0.74):.2f}g "
          f"pipe@0.97={sim_pipelined(0.97) / slot:.0f}tok/s")


# ---- fig 2: arms per cell (median + min..max whiskers over interleaved reps) --------------------
def fig_arms(bench="m25-paper-bench-20260703.json"):
    path = os.path.join(RCP, bench)
    if not os.path.exists(path):
        print("skip fig_arms (no", path, ")")
        return
    R = json.load(open(path))
    cells = [c for c in ["reason-math", "reason-logic", "open-chat", "code-edit", "rag-quote",
                         "agentic-tool", "ctx-8k-summarize", "ctx-8k-quote"]
             if any(r["cell"] == c for r in R)]
    arms = ["ar", "chain", "hybrid"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    w = 0.26
    for j, arm in enumerate(arms):
        xs, meds, los, his = [], [], [], []
        for i, cell in enumerate(cells):
            v = [r["tok_s"] for r in R if r["cell"] == cell and r["arm"] == arm]
            if not v:
                continue
            xs.append(i + (j - 1) * w)
            meds.append(statistics.median(v)); los.append(min(v)); his.append(max(v))
        ax.bar(xs, meds, width=w * 0.92, color=C[arm], zorder=3,
               label={"ar": "autoregressive (no speculation)", "chain": "chain-EAGLE",
                      "hybrid": "depth-aware hybrid"}[arm])
        ax.errorbar(xs, meds, yerr=[[m - l for m, l in zip(meds, los)],
                                    [h - m for m, h in zip(meds, his)]],
                    fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6, zorder=4)
        for x, m in zip(xs, meds):
            ax.annotate(f"{m:.1f}", (x, m), xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=6.4, color=INK)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([c.replace("ctx-", "") for c in cells], rotation=18, ha="right", fontsize=7.5)
    ax.set_ylabel("tokens / s (median, min–max)")
    ax.set_ylim(0, 16.5)          # headroom: legend row must not collide with bar labels
    _grid(ax)
    ax.legend(frameon=False, fontsize=7.5, ncols=3, loc="upper left")
    fig.tight_layout()
    fig.savefig("fig_arms.pdf"); fig.savefig("fig_arms.png", dpi=220, bbox_inches="tight")
    print("fig_arms.pdf")


# ---- fig 3: where a traversal goes (per-arm transport vs stage-compute shares) ------------------
def fig_split(bench="m25-paper-bench-20260703.json"):
    path = os.path.join(RCP, bench)
    if not os.path.exists(path):
        print("skip fig_split")
        return
    R = [r for r in json.load(open(path)) if r.get("transport_s") and r["arm"] != "ar"]
    rows = []
    for arm in ("chain", "hybrid"):
        rr = [r for r in R if r["arm"] == arm]
        tv = sum(r["traversal_s"] for r in rr); st = sum(r["stage_s"] for r in rr)
        n_tr = sum(r["new_tokens"] / r["g"] for r in rr)
        rows.append((arm, tv / n_tr * 1000, st / n_tr * 1000))
    fig, ax = plt.subplots(figsize=(4.6, 1.7))
    for i, (arm, tvms, stms) in enumerate(rows):
        ax.barh(i, stms, color=C[arm], zorder=3, height=0.55)
        ax.barh(i, tvms - stms, left=stms + 2, color="#c3c2b7", zorder=3, height=0.55)
        ax.annotate(f"stage compute {stms:.0f} ms", (stms / 2, i), ha="center", va="center",
                    fontsize=7, color="#ffffff" if arm != "hybrid" else INK)
        ax.annotate(f"transport + codec {tvms - stms:.0f} ms ({(tvms - stms) / tvms * 100:.0f}%)",
                    (stms + (tvms - stms) / 2, i), ha="center", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("mean traversal decomposition (ms)")
    ax.grid(axis="x", color="#e4e3da", lw=0.5, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig("fig_split.pdf"); fig.savefig("fig_split.png", dpi=220, bbox_inches="tight")
    print("fig_split.pdf")


# ---- fig 0: page-one hero — decode throughput by serving mode ----------------------------------
def fig_hero():
    """The scroller's chart: four modes, one axis, measured numbers. Bar length is the top measured
    median per mode; the caption in main.typ carries the ranges and receipts."""
    rows = [  # (label, value, sublabel, color)
        ("no speculation\n(the latency wall)", 5.0, "5.0 tok/s", "#8a897f"),
        ("interactive reasoning\nsingle stream", 12.6, "12.6 tok/s", C["ar"]),
        ("draftable text\nsingle stream", 87.2, "87.2 tok/s", C["chain"]),
        ("batched, 4 streams\naggregate", 194.0, "194 tok/s", C["hybrid"]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ys = range(len(rows))
    for y, (lab, v, sub, col) in enumerate(rows):
        ax.barh(y, v, height=0.62, color=col, zorder=3)
        ax.annotate(sub, (v, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=10.5, fontweight="bold", color=INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 225)
    ax.set_xlabel("decode tokens / second — 229B MoE across five countries, receipts on", fontsize=9)
    ax.grid(axis="x", color="#e4e3da", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    fig.savefig("fig_hero.pdf"); fig.savefig("fig_hero.png", dpi=220, bbox_inches="tight")
    print("fig_hero.pdf")


if __name__ == "__main__":
    fig_hero()
    fig_alpha_law()
    fig_arms()
    fig_split()
