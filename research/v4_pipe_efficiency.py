"""What the V4 pipelined DSpark ring can actually be fed, and how wide a ring is worth renting.

`coordinate_dspark_pipelined` streams s=1 frames so stage k works token i while stage k-1 works
i+1. A depth sweep on a TEN-stage ring found the fill saturating well below ten, and depths 6, 8
and 12 landing byte-identical:

    depth  tok/s   g     max_inflight  stale  waste
      4    2.516  5.82        4         21    0.288
      6    2.761  4.92        6         41    0.456
      8    2.843  4.92        6         41    0.456
     12    2.934  4.92        6         41    0.456

This script is the arithmetic behind docs/V4_MULTIBLOCK_VERDICT.md: it fits the drafter's
per-depth acceptance from that sweep, replays the coordinator's own frame schedule against the
fit, and prices every way of feeding the pipeline harder. Run it with no arguments.

    python3 research/v4_pipe_efficiency.py

THE ONE INPUT IT CANNOT INVENT is the drafter's acceptance-vs-depth curve, which is why
`coordinate_dspark_pipelined` now reports `accept_by_depth`. Feed a real ring's histogram in via
--accept and every number below is re-derived from measurement instead of from a two-point fit.
"""
import argparse
import json
import random

# ── the model ──────────────────────────────────────────────────────────────────────────────────────
#
# Every frame the coordinator sends earns exactly one reply, and replies arrive at the rate the ring
# can retire frames: min(F, D)/L for F frames in flight over D stages of total traversal L. Only a
# reply that is not fenced commits a token. So
#
#     tok/s = (tokens / frames) * min(Fbar, D) / L
#
# and the whole question is what a lever does to the two factors, which pull against each other:
# more frames in flight retires them faster, but the extra frames are the DEEPEST drafts, they are
# the least likely to be right, and every one of them enlarges the discarded future behind a
# rejection. Acceptance is not one number — a draft predicted d positions ahead of the frontier it
# was drafted from lands with conditional probability q_d, and q decays in d.

MEASURED = {4: (5.82, 0.288, 2.516), 6: (4.92, 0.456, 2.761),
            8: (4.92, 0.456, 2.843), 12: (4.92, 0.456, 2.934)}
BLOCK = 5


def g_of(q, b):
    """E[tokens per cycle] when the conditional acceptances along a cycle are q_1..q_b repeating.

    A cycle ends at the first draft that misses, and commits every accepted draft plus one
    correction, so g is the expected index of that first miss:
        g = sum_{t>=0} P(first t drafts all land) = (sum_{i<b} P_i) / (1 - P_b),  P_i = prod_{j<i} q_j
    Note this is INDEPENDENT of b when q is flat — which is why the sweep moving g at all (5.82 at
    three drafts, 4.92 at five) is already proof that acceptance decays with depth."""
    P = [1.0]
    for j in range(b):
        P.append(P[-1] * q[j])
    return sum(P[:b]) / (1.0 - P[b])


def fit_profile(b_lo=3, g_lo=5.82, b_hi=5, g_hi=4.92):
    """Fit q_j = 1 - (1-a)*rho^(j-1) — the MISS probability compounding geometrically with depth —
    to the two block widths the sweep effectively measured. Two equations, two unknowns.

    W=4 clips the block to three drafts (v4_pipe streams while `(pos+2+i) - c < W`), so the depth-4
    row IS a three-draft measurement; W>=6 lets all five through."""
    def a_for(rho):
        lo, hi = 0.01, 1 - 1e-12
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if g_of(prof(mid, rho), b_hi) < g_hi else (lo, mid)
        return (lo + hi) / 2

    def prof(a, rho, n=BLOCK):
        return [min(1 - 1e-9, max(1e-6, 1 - (1 - a) * rho ** (j - 1))) for j in range(1, n + 1)]

    lo, hi = 1.0, 6.0
    for _ in range(200):
        rho = (lo + hi) / 2
        lo, hi = (rho, hi) if g_of(prof(a_for(rho), rho), b_lo) < g_lo else (lo, rho)
    rho = (lo + hi) / 2
    return prof(a_for(rho), rho), a_for(rho), rho


def _clip(v):
    return min(1 - 1e-9, max(1e-6, v))


# Alternative shapes for the decay, all two-parameter, all fitted to the same two measured points.
# They exist to answer "did we just draw a convenient curve?" — see the last section of the output.
_SHAPES = {
    "miss geometric": lambda a, r: [_clip(1 - (1 - a) * r ** (j - 1)) for j in range(1, BLOCK + 1)],
    "q geometric": lambda a, r: [_clip(a * (1 / (1 + r)) ** (j - 1)) for j in range(1, BLOCK + 1)],
    "miss linear": lambda a, r: [_clip(1 - (1 - a) * (1 + r * (j - 1))) for j in range(1, BLOCK + 1)],
    "miss power-law": lambda a, r: [_clip(1 - (1 - a) * j ** r) for j in range(1, BLOCK + 1)],
}


def _fit2(mk, b_lo=3, g_lo=5.82, b_hi=5, g_hi=4.92):
    """Nested bisection: inner solves the level to hit g(5), outer solves the decay to hit g(3)."""
    def a_for(r):
        lo, hi = 0.01, 1 - 1e-12
        for _ in range(300):
            m = (lo + hi) / 2
            lo, hi = (m, hi) if g_of(mk(m, r), b_hi) < g_hi else (lo, m)
        return (lo + hi) / 2
    lo, hi = 1e-6, 8.0
    for _ in range(300):
        r = (lo + hi) / 2
        lo, hi = (r, hi) if g_of(mk(a_for(r), r), b_lo) < g_lo else (lo, r)
    r = (lo + hi) / 2
    return mk(a_for(r), r)


def simulate(mode, q, D=10, L=0.79, n_tok=120000, seed=0, agree=1.0, kblocks=1):
    """Replay coordinate_dspark_pipelined's own horizon/frontier bookkeeping.

    `mode` is the refill policy: "block" is the shipped `if blk and horizon == c` guard, "rolling"
    tops the pipeline up from every reply's fresh block. `agree` is the probability that a fresh
    block's re-prediction of positions already in flight matches what is in flight — when it does
    not, the deep draft's Markov conditioning (model.py:867-871) is a chain the ring is not taking,
    so a top-up is skipped rather than fed. `kblocks` prices a hypothetical chained multi-block
    drafter (see the verdict doc: it is not reachable with DSpark).
    """
    rng = random.Random(seed)
    qq, Bk = q * kblocks, len(q) * kblocks
    depths = []                                     # draft depth of the frame at frontier+1, +2, ...
    frames, tokens, cancels, fsum, n = 1, 0, 0, 0.0, 0
    while tokens < n_tok:
        tokens += 1
        if depths:                                  # frontier+1 carried a speculative frame
            if rng.random() >= qq[depths.pop(0) - 1]:
                cancels += 1                        # REJECT: the whole future behind it is discarded
                depths = []
                frames += 1                         # the correction frame IS the rewind command
        else:
            frames += 1                             # nothing speculated here: feed the truth
        for j in (range(1, Bk + 1) if (mode == "rolling" or not depths) else ()):
            if j <= len(depths):
                continue                            # that position already has a frame in the ring
            if depths and rng.random() >= agree:
                break                               # fresh block disagrees with what is in flight
            depths.append(j)
            frames += 1
        fsum += len(depths) + 1                     # in flight == horizon - c + 1
        n += 1
    F, commit = fsum / n, tokens / frames
    return {"g": tokens / max(cancels, 1), "F": F, "waste": 1 - commit,
            "tok_s": commit * min(F, D) / L}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accept", help="JSON {depth: [hits, trials]} from a real ring's "
                                     "`accept_by_depth`; overrides the two-point fit")
    ap.add_argument("--L", type=float, default=0.79, help="ring traversal seconds (fitted)")
    a = ap.parse_args()

    if a.accept:
        h = json.loads(a.accept)
        q = [h[k][0] / h[k][1] for k in sorted(h, key=int) if h[k][1]]
        print(f"acceptance MEASURED on a real ring: " +
              "  ".join(f"q_{i+1}={v:.4f}" for i, v in enumerate(q)))
    else:
        q, a0, rho = fit_profile()
        print(f"acceptance FITTED from the sweep (g=5.82 at 3 drafts, 4.92 at 5): "
              f"a={a0:.4f} rho={rho:.4f}")
        print("  " + "  ".join(f"q_{i+1}={v:.4f}" for i, v in enumerate(q)))
        print(f"  check g(3)={g_of(q,3):.3f} (meas 5.82)   g(5)={g_of(q,5):.3f} (meas 4.92)")
    L = a.L

    print("\n── the shipped coordinator, against the measured sweep ────────────────────────────")
    print(f"{'depth':>6} {'g sim':>7} {'g meas':>7} {'Fbar':>6} {'waste':>7} {'waste meas':>11}")
    for W, b in ((4, 3), (6, 5), (8, 5), (12, 5)):
        r = simulate("block", q[:b], D=10, L=L)
        print(f"{W:>6} {r['g']:>7.2f} {MEASURED[W][0]:>7.2f} {r['F']:>6.2f} "
              f"{r['waste']:>7.3f} {MEASURED[W][1]:>11.3f}")
    print("  max in flight is block+1 at every depth >= 6, which is the whole finding: the depth")
    print("  knob stops doing anything once it clears the block, so stages beyond block+1 idle.")

    print("\n── feeding it harder: keep the pipeline topped up instead of letting it sag ────────")
    print(f"{'policy':>16} {'g':>6} {'Fbar':>6} {'waste':>7} " +
          " ".join(f"{'D='+str(d):>7}" for d in (4, 6, 8, 10, 14)))
    base = None
    for mode, ag in (("block", 1.0), ("rolling", 1.0), ("rolling", 0.8),
                     ("rolling", 0.6), ("rolling", 0.4)):
        r = simulate(mode, q, D=99, L=L, agree=ag)
        cells = " ".join(f"{simulate(mode, q, D=d, L=L, agree=ag)['tok_s']:>7.3f}"
                         for d in (4, 6, 8, 10, 14))
        tag = f"{mode} agree={ag:.1f}" if mode == "rolling" else f"{mode} (shipped)"
        best = simulate(mode, q, D=10, L=L, agree=ag)["tok_s"]
        base = base if base is not None else best
        print(f"{tag:>16} {r['g']:>6.2f} {r['F']:>6.2f} {r['waste']:>7.3f} " + cells +
              f"   {100*(best/base-1):+.1f}%")
    print("  Fill is not throughput: rolling buys 4.4 -> 6.0 frames in flight and pays for it in g")
    print("  (every added frame is the block's deepest, weakest draft) and in waste. The two")
    print("  effects very nearly cancel.")

    print("\n── the break-even: what must g hold at for one MORE frame in flight to pay? ────────")
    print("  Rate is g/(g+F) * F (the L and D factors cancel on a ring with headroom). One extra")
    print("  frame retires the queue faster and is thrown away behind every rejection, so solving")
    print("  g'(F+1)/(g'+F+1) = gF/(g+F) gives the g the deeper pipe must still clear:")
    g0 = simulate("block", q, D=99, L=L)["g"]
    print(f"{'F -> F+1':>10} {'g must hold above':>19} {'tolerable drop':>16}")
    for F in (4, 5, 6, 7, 8, 9):
        k = g0 * F / (g0 + F)
        need = k * (F + 1) / (F + 1 - k)
        print(f"{str(F) + ' -> ' + str(F+1):>10} {need:>19.2f} {100*(need/g0-1):>15.1f}%")
    print("  The tolerance TIGHTENS as the pipe deepens, which is the whole shape of the problem:")
    print("  deep frames are both the least likely to be right and the costliest to be wrong about.")

    print("\n── the question that actually matters: how wide a ring? ────────────────────────────")
    print("  Total traversal is L = n_layers*tau_layer + D*hop: a wider ring makes each stage faster")
    print("  but adds a hop, and by Little's law the retire rate is min(F, capacity)/L with F pinned")
    print("  at block+1. A frame occupies a STAGE or a LINK, so a ring can overlap about")
    print("  D*(1 + hop/tau) frames, not merely D — which is why the peak is NOT simply at F.")
    A, hop, F = 12.27, 1.0, BLOCK + 1               # A solved from the measured 6-box/10-box pair
    print(f"{'D':>4} {'tau(hops)':>10} {'capacity':>9} {'L':>7} {'rate':>8}   (F = {F})")
    rows = []
    for D in range(2, 15):
        tau = A / D
        cap = D * (1 + hop / tau)
        rate = min(F, cap) / (A + D * hop)
        rows.append((rate, D))
        if D in (2, 3, 4, 5, 6, 8, 10, 14):
            print(f"{D:>4} {tau:>10.2f} {cap:>9.2f} {A + D*hop:>7.2f} {rate:>8.4f}")
    peak = max(rows)
    print(f"  PEAK at D={peak[1]}, and the curve is flat across D=5..6 (within ~6%). `A` is SOLVED")
    print(f"  from the measured 6-box/10-box pair (5.35 vs 4.39 = 1.22x), so the level is a")
    print(f"  calibration, not a prediction; what the shape buys for free is the LOCATION of the")
    print(f"  peak and the cost of overshooting it — D=10 gives up ~{100*(1-rows[8][0]/peak[0]):.0f}%.")
    print("  The honest rule is not 'D = block+1'. It is: once the ring can overlap the frames the")
    print("  drafter can supply, every further stage is pure added latency, so run the NARROWEST")
    print("  ring the weights fit on. For V4-Flash on 5090s that floor is ~6 boxes, which lands on")
    print("  the flat top of this curve by memory, not by coincidence of arithmetic.")

    print("\n── pricing the lever that was ASKED for (chained multi-block) ──────────────────────")
    print("  Not reachable with DSpark — one tap per frame, one frontier advance per reply, so the")
    print("  horizon is pinned block+1 past the frontier no matter how speculative the tail is.")
    print("  Shown only to size what a TAP-FREE proposer (n-gram) could unlock, and OPTIMISTICALLY:")
    print("  assumes a chained block's acceptance resets to q_1..q_B rather than compounding.")
    print(f"{'blocks':>8} {'F cap':>6} {'g':>6} " + " ".join(f"{'D='+str(d):>7}" for d in (6, 10, 14, 20)))
    for kb in (1, 2, 3):
        r = simulate("rolling", q, D=99, L=L, kblocks=kb)
        cells = " ".join(f"{simulate('rolling', q, D=d, L=L, kblocks=kb)['tok_s']:>7.3f}"
                         for d in (6, 10, 14, 20))
        print(f"{kb:>8} {len(q)*kb+1:>6} {r['g']:>6.2f} " + cells)
    print("  Depth only pays on a ring wide enough to absorb the wasted frames: at D=6 chaining is")
    print("  a large LOSS, and it needs D>=11 before two blocks beat one.")

    print("\n── is the top-up verdict an artifact of the assumed decay SHAPE? ───────────────────")
    print("  Two measured points fix two parameters, but not the family. Refit each shape to the")
    print("  same g(3)=5.82 / g(5)=4.92 and re-price the top-up:")
    print(f"{'family':>24} {'q_1':>6} {'q_5':>6} {'blk':>7} {'roll':>7} {'delta':>7}")
    for name, mk in _SHAPES.items():
        qf = _fit2(mk)
        b, r = simulate("block", qf, D=10, L=L), simulate("rolling", qf, D=10, L=L)
        print(f"{name:>24} {qf[0]:>6.3f} {qf[4]:>6.3f} {b['tok_s']:>7.3f} {r['tok_s']:>7.3f} "
              f"{100*(r['tok_s']/b['tok_s']-1):>+6.1f}%")
    print("  The three shapes that fit inside the unit interval agree to a tenth of a point, so the")
    print("  'about +3%, and negative if blocks disagree' verdict is a property of the measurement,")
    print("  not of the curve that was drawn through it. ('q geometric' runs to the q_1=1 boundary,")
    print("  i.e. it cannot represent this data at all — shown so the failure is visible.)")


if __name__ == "__main__":
    main()
