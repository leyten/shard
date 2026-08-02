"""What a pipelined-fill lever is WORTH — refill floor first, tap-free proposer second — priced by a
frame-exact replay of `coordinate_dspark_pipelined`'s round logic plus Little's law, from measured
inputs.

Deliberately NOT phase0/v4_pipe_sim.py. That is a discrete-event simulator of the whole ring,
calibrated against a depth sweep on an earlier box set; this replays the shipped coordinator's round
statement for statement (the refill guard, the floor, the epoch fence, the in-flight cap) and prices
the result against one ring's measured numbers. Two models, disjoint machinery, same questions —
where they agree the answer is a property of the round structure rather than of either fit.

THE 07-31 RING (six distinct EU 5090s, 512-token novel prompt, best config, decode 17.64 tok/s):

    on_box per stage   16.71 / 14.15 / 12.56 / 16.70 / 13.71 / 18.00 ms   (sum 91.8)
    wire per circuit   110.7 ms         =>  L = 202.5 ms measured, not fitted
    g (tokens/cancel)  11.13            =>  46 cycles over 512 tokens, ~18 frames each

TWO CORRECTIONS THIS FILE CARRIES, both the same class of error:

  1. "slowest stage 18.0 ms => ceiling 55.6 tok/s; measured 17.64 => 32% full" divides a TOKEN rate
     by a FRAME ceiling. The ring retires ~28.5 frames/s against the 55.6/s cap — 51% — and at the
     measured 0.62 tokens/frame the honest token ceiling is ~34 tok/s (1.95x), reached near depth
     L/tau_max ~= 11, not 16. The depth-16 "49 tok/s" needs 79 frames/s through an 18 ms stage.
  2. docs/V4_MULTIBLOCK_VERDICT.md §4 priced the rolling top-up at +3% and declined to build it.
     That price was a property of the OLD drafter (q5=0.68, g=4.9); the lever's value is almost
     entirely q at the block's deepest index, and this ring's drafter (g=11.13) reprices it to ~+20%.
     The verdict's method stands; its number was calibration-bound. Hence V4_REFILL_FLOOR.

WHAT THE REPLAY DOES NOT MODEL, stated where the numbers are made: a topped-up frame's acceptance is
drawn from the same per-index curve as a drain-refill frame's, i.e. the mid-run block is assumed to
re-predict the in-flight positions it overlaps. The coordinator now measures that assumption directly
(`topup_accept_by_depth`, `topup_agree/topup_disagree`) — feed a live run's histogram back in with
--accept and the biggest free input here disappears.

    python3 phase0/v4_ngram_econ.py                    # 07-31 ring: floors, then the tap-free arms
    python3 phase0/v4_ngram_econ.py --ring old         # the previous ten-stage-era calibration
    python3 phase0/v4_ngram_econ.py --accept '{"1": [h, t], ...}'   # a live accept_by_depth
"""
import argparse
import json
import random
import statistics

RINGS = {
    # tau = per-stage on_box (ms); wire = measured circuit wire (ms); base = measured decode tok/s;
    # g/eta = measured tokens-per-cancel and tokens-per-frame the drafter fit must reproduce.
    "0731": dict(tau=[16.71, 14.15, 12.56, 16.70, 13.71, 18.00], wire=110.7, base=17.64,
                 g=11.13, eta=0.618),
    # The ring the first tap-free verdict was priced on. Its drafter fits are kept verbatim
    # (v4_pipe_sim's 5-point and the MULTIBLOCK 2-point); L was fitted, not measured.
    "old": dict(tau=[13.7, 11.3, 16.6, 14.6, 26.0, 37.4], wire=None, base=10.06,
                g=None, eta=None),
}
B = 5
# Marginal cost of one drafted block on the tail, ms: measured 20.3 on the previous ring, 30 as the
# pessimistic arm. A raised floor defeats lazy drafting (the hint licensing withholds the last
# floor+1 positions), so the tail pays for blocks the lazy baseline skipped — charged below as
# delta(blocks/frame) x this, onto the tail's stage time.
DRAFT_MS = (20.3, 30.0)


def q_curve(a1, rho, b=B):
    """Per-index conditional acceptance q_j = 1 - (1-a1)*rho^(j-1) — the MULTIBLOCK fit family."""
    return [max(0.0, min(1.0, 1.0 - (1.0 - a1) * rho ** j)) for j in range(b)]


def run(mode, W, aseq, q_ng=0.0, r_ng=1.0, floor=1, n=20000, seed=1):
    """Frame-exact replay of the shipped round logic. -> (eta, Fbar, g, drafts_per_frame).

    `mode`: dspark (block only), hybrid (block + tap-free top-up past it), ngram (no block).
    `floor`: V4_REFILL_FLOOR — consume a reply's block at or below this in-flight level, streaming
    only positions past the deepest frame in flight, exactly as coordinate_dspark_pipelined does."""
    rng = random.Random(seed)
    c = horizon = 0
    sent = {0: True}
    bad = None                       # shallowest position known to be a wrong proposal, if any
    frames, toks, cancels, drafts, depths = 1, 0, 0, 0, []
    while toks < n:
        pos = c
        good = sent.get(pos + 1)
        c, toks = pos + 1, toks + 1
        if good is None or not good:
            if good is not None:                       # REJECT: discard the future
                for p in [p for p in sent if p > pos]:
                    del sent[p]
                horizon, bad, cancels = pos, None, cancels + 1
            frames += 1
            sent[c] = True
            horizon = max(horizon, c)
        if bad is not None and bad <= c:
            bad = None
        if mode != "ngram" and horizon - c + 1 <= floor:
            drafts += 1
            base = horizon
            for i in range(B):
                p = pos + 2 + i
                if p <= base:                          # in flight: never re-streamed
                    continue
                if p - c >= W:
                    break
                ok = bad is None and rng.random() < aseq[i]
                if not ok and bad is None:
                    bad = p
                sent[p] = ok
                horizon, frames = max(horizon, p), frames + 1
        if mode != "dspark" and horizon - c + 1 < W and rng.random() < r_ng:
            base = horizon
            for i in range(W - (base - c + 1)):
                p = base + 1 + i
                if p - c >= W:
                    break
                ok = bad is None and rng.random() < q_ng
                if not ok and bad is None:
                    bad = p
                sent[p] = ok
                horizon, frames = max(horizon, p), frames + 1
        depths.append(horizon - c + 1)
        sent.pop(pos - 1, None)
    return toks / frames, statistics.fmean(depths), toks / max(cancels + 1, 1), drafts / frames


def fit_drafter(g_meas, eta_meas):
    """(a1, rho) reproducing the measured g at floor=1, breaking ties toward the measured eta.

    g is weighted 10x: it is the run's hard counter (512 tokens / 46 cycles), while eta was derived
    from the brief's '~18 frames per round', which the round structure itself bounds looser (a cancel
    cannot discard more frames than the pipe holds). The weakest input in this file either way; a
    live accept_by_depth replaces it wholesale."""
    def err(a1, rho, n):
        e, _, g, _ = run("dspark", 16, q_curve(a1, rho), n=n)
        return 10.0 * ((g - g_meas) / g_meas) ** 2 + ((e - eta_meas) / eta_meas) ** 2
    best = None
    for a1 in [x / 1000 for x in range(880, 986, 5)]:
        for rho in [x / 100 for x in range(80, 161, 4)]:
            v = err(a1, rho, 8000)
            if best is None or v < best[0]:
                best = (v, a1, rho)
    _, a1, rho = best
    for _ in range(2):                                 # refine
        for da in (-0.004, -0.002, 0, 0.002, 0.004):
            for dr in (-0.03, -0.015, 0, 0.015, 0.03):
                v = err(a1 + da, rho + dr, 20000)
                if v < best[0]:
                    best = (v, a1 + da, rho + dr)
        _, a1, rho = best
    return a1, rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", default="0731", choices=sorted(RINGS))
    ap.add_argument("--accept", default=None,
                    help="a live run's accept_by_depth as JSON {'1': [hits, trials], ...}; replaces "
                         "the fitted drafter curve with the measured one")
    a = ap.parse_args()
    ring = RINGS[a.ring]
    tau = list(ring["tau"])
    tau_max = max(tau)
    cap = 1000.0 / tau_max

    if a.accept:
        hb = {int(k): v for k, v in json.loads(a.accept).items()}
        aseq = [(hb[j][0] / hb[j][1]) if j in hb and hb[j][1] else 0.0 for j in range(1, B + 1)]
        curves = [("measured accept_by_depth", aseq)]
    elif a.ring == "old":
        curves = [("v4_pipe_sim 5-point fit", [0.796 * 0.945 ** i for i in range(B)]),
                  ("MULTIBLOCK 2-point fit", [0.859, 0.827, 0.787, 0.739, 0.680])]
    else:
        a1, rho = fit_drafter(ring["g"], ring["eta"])
        # Two curves on purpose, and the spread between them is the honest error bar. The replay
        # cannot land on (g=11.13, eta=0.618) at once — 18 frames per 11.13-token cycle implies
        # ~6.9 discarded frames per cancel, more than a 6-deep pipe can hold — so the brief's
        # "~18 frames/round" was soft and the fit favours the hard number, g. The second curve is
        # the ten-agent model's calibration for this ring (fitted to the run's frame counters and
        # validated on mean/max in-flight), whose decay shape the MULTIBLOCK verdict also found.
        curves = [(f"fit to g={ring['g']} (a1={a1:.4f} rho={rho:.3f})", q_curve(a1, rho)),
                  ("calibrated model a=0.9236 rho=1.2271", q_curve(0.9236, 1.2271))]

    for name, aseq in curves:
        eta, fbar, g, dpf = run("dspark", 16, aseq)
        base = ring["base"]
        # L: measured where the ring measured it (compute + wire circuit); the replay's Fbar is a
        # per-reply mean, so the model is closed by calibrating an effective L on the dspark arm and
        # cross-checking it against the measured circuit rather than silently replacing it.
        L_fit = eta * fbar / base * 1000.0
        L_meas = (sum(tau) + ring["wire"]) if ring["wire"] else None
        print(f"\n=== ring {a.ring}  drafter: {name}")
        print(f"    q = {[round(x, 3) for x in aseq]}")
        print(f"    baseline replay: eta {eta:.3f}  Fbar {fbar:.2f}  g {g:.2f}"
              + (f"  (measured g {ring['g']})" if ring["g"] else ""))
        print(f"    L effective {L_fit:.0f} ms"
              + (f"; measured circuit {L_meas:.0f} ms — the gap is burst overlap the "
                 f"per-reply mean cannot see" if L_meas else ""))
        print(f"    frame rate {base / eta:5.1f}/s vs bottleneck cap {cap:.1f}/s "
              f"= {(base / eta) / cap * 100:.0f}% busy;  frames/token {1 / eta:.2f}")

        def tps(e, f, tmax=tau_max):
            return e * min(f / L_fit * 1000.0, 1000.0 / tmax)

        print(f"    {'policy':<34}{'eta':>7}{'Fbar':>6}{'g':>7}{'blk/f':>7}{'tok/s':>8}{'vs base':>9}")

        def row(lbl, mode, W, floor=1, q=0.0, r=1.0, tmax=tau_max):
            e, f, gg, d = run(mode, W, aseq, q, r, floor=floor)
            t = tps(e, f, tmax)
            print(f"    {lbl:<34}{e:>7.3f}{f:>6.2f}{gg:>7.2f}{d:>7.2f}{t:>8.2f}"
                  f"{(t / base - 1) * 100:>8.1f}%")
            return t, e, d

        _, e1, d1 = row("floor=1 (shipped)", "dspark", 16, floor=1)
        for fl in (2, 3, 4, 5):
            row(f"floor={fl}", "dspark", 16, floor=fl)
        # THE LAZY-DEFEAT SENSITIVITY: at floor=5 every committed reply drafts, so the tail pays
        # delta(blocks/frame) x the marginal draft cost on top of its stage time. drafts/frame at
        # floor 1 is the lazy baseline's (only consumed blocks are drafted).
        e5, f5, _, d5 = run("dspark", 16, aseq, floor=5)
        for c_draft in DRAFT_MS:
            t_tail = tau[-1] + (d5 - d1) * c_draft
            tmax = max(tau_max, t_tail)
            t = tps(e5, f5, tmax)
            print(f"    {'floor=5, draft charged ' + f'{c_draft:.1f}ms':<34}{e5:>7.3f}{f5:>6.2f}"
                  f"{'':>7}{d5:>7.2f}{t:>8.2f}{(t / base - 1) * 100:>8.1f}%   "
                  f"(tail {tau[-1]:.1f} -> {t_tail:.1f} ms)")
        # The tap-free arms, second-order on purpose: measured n-gram acceptance is a property of
        # the PROMPT (0.96 edit / 0.016 novel, phase0/v4_ngram_accept.py) and the headline benchmark
        # is a novel prompt. Priced at the floor they would ride on.
        for W in (8, 12):
            row(f"floor=5 + ngram top-up W={W} q=.90", "hybrid", W, floor=5, q=0.90, r=1.0)
        row("floor=5 + ngram, novel GATED r=0", "hybrid", 8, floor=5, q=0.016, r=0.0)
        row("floor=5 + ngram, novel UNGATED", "hybrid", 8, floor=5, q=0.016, r=1.0)
        # Break-even: the flat acceptance a tap-free extension needs to beat the floored baseline.
        base5 = tps(e5, f5)
        lo, hi = 0.0, 1.0
        for _ in range(14):
            mid = (lo + hi) / 2
            e, f, _, _ = run("hybrid", 8, aseq, mid, 1.0, floor=5, n=6000)
            if tps(e, f) < base5:
                lo = mid
            else:
                hi = mid
        print(f"    BREAK-EVEN for a tap-free extension on top of floor=5, W=8: flat q >= {hi:.2f}"
              f"  (measured n-gram novel q = 0.016)")


if __name__ == "__main__":
    main()
