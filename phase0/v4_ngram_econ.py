"""What the tap-free proposer is WORTH — the break-even, per workload class, from measured inputs.

Deliberately NOT phase0/v4_pipe_sim.py. That is a discrete-event simulator of the whole ring and it
is calibrated against a depth sweep on the PREVIOUS (ten times slower) box set; this is a frame-exact
replay of `coordinate_dspark_pipelined`'s round logic plus Little's law, calibrated against ONE
number on the current ring. Two models, disjoint machinery, same question — where they agree the
answer is a property of the round structure rather than of either fit.

THE CORRECTION THIS FILE EXISTS TO MAKE. "Slowest stage 37.4 ms, so the ceiling is 26.8 tok/s; we
measure 10.06; therefore the pipe is 37% full" divides a TOKEN rate by a FRAME ceiling. A pipelined
speculative ring computes frames it then discards, so the two differ by exactly the acceptance. Replay
the shipped round logic against the measured DSpark acceptance and it commits 0.62 tokens per frame at
4.4 in flight: to emit 10.06 tok/s the ring must be retiring 16.3 frames/s, which is 61% of the
ceiling. The pipe is not 37% full and the missing capacity is mostly not idle -- it is already being
spent on speculation that gets thrown away. `frames / generated` off any real run settles it: near 1.0
is fill-bound, and this model says 1.62.

    tok/s = (tokens / frames) x min(Fbar / L, 1 / tau_max)

`tau_max` and the per-stage times are measured. `L` is the one fitted parameter, set so the `dspark`
arm reproduces the ring's measured 10.06 tok/s. The acceptance sequences and the n-gram propose-rates
are measured (phase0/v4_ngram_accept.py); nothing else is free.

    python3 phase0/v4_ngram_econ.py

Reports every policy under BOTH published DSpark acceptance fits, because the weakest input here is
that curve and the verdict should be seen to survive either one.
"""
import random, statistics

TAU = [13.7, 11.3, 16.6, 14.6, 26.0, 37.4]
TAU_MAX = max(TAU)
A_SIM = [0.796 * 0.945 ** i for i in range(5)]
A_2PT = [0.859, 0.827, 0.787, 0.739, 0.680]
B, BASE = 5, 10.06


def run(mode, W, aseq, q_ng, r_ng=1.0, n=6000, seed=1):
    rng = random.Random(seed)
    c = horizon = 0
    sent = {0: True}
    bad = None                       # shallowest position known to be a wrong proposal, if any
    frames, toks, depths = 1, 0, []
    while toks < n:
        pos = c
        good = sent.get(pos + 1)
        c, toks = pos + 1, toks + 1
        if good is None or not good:
            if good is not None:                       # REJECT: discard the future
                for p in [p for p in sent if p > pos]:
                    del sent[p]
                horizon, bad = pos, None
            frames += 1
            sent[c] = True
            horizon = max(horizon, c)
        if bad is not None and bad <= c:
            bad = None
        if mode != "ngram" and horizon == c:
            for i in range(B):
                p = pos + 2 + i
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
    return toks / frames, statistics.fmean(depths)


def tps(eta, fbar, L):
    return eta * min(fbar / L, 1.0 / TAU_MAX) * 1000.0


for name, aseq in (("v4_pipe_sim 5-point fit", A_SIM), ("MULTIBLOCK 2-point fit", A_2PT)):
    eta, fbar = run("dspark", 16, aseq, 0.0)
    L = eta * fbar / BASE * 1000.0
    cap = 1000.0 / TAU_MAX
    print(f"\n=== DSpark acceptance: {name}  {[round(x, 3) for x in aseq]} ===")
    print(f"  baseline eta {eta:.3f}  Fbar {fbar:.2f}  ->  L {L:.0f} ms")
    print(f"  frame rate {BASE/eta:5.1f}/s  vs bottleneck cap {cap:.1f}/s  "
          f"= {(BASE/eta)/cap*100:.0f}% busy;  frames/token {1/eta:.2f}")
    print(f"  {'policy':<28}{'W':>3}{'q_ng':>7}{'rate':>6}{'eta':>7}{'Fbar':>7}{'tok/s':>8}{'vs dspark':>11}")

    def row(lbl, mode, W, q, r=1.0):
        e, f = run(mode, W, aseq, q, r)
        t = tps(e, f, L)
        print(f"  {lbl:<28}{W:>3}{q:>7.3f}{r:>6.2f}{e:>7.3f}{f:>7.2f}{t:>8.2f}{(t/BASE-1)*100:>10.1f}%")
        return t

    row("dspark (baseline)", "dspark", 16, 0.0)
    for W in (7, 8, 10, 16):
        row("hybrid  edit", "hybrid", W, 0.96, 0.63)
        row("hybrid  edit_heavy", "hybrid", W, 0.91, 0.32)
        row("hybrid  novel, UNGATED", "hybrid", W, 0.016, 1.00)
        row("hybrid  novel, GATED", "hybrid", W, 0.016, 0.00)
        row("ngram   edit", "ngram", W, 0.96, 1.00)
        row("ngram   novel", "ngram", W, 0.016, 1.00)
    for mode in ("hybrid", "ngram"):
        best = None
        for W in range(6, 21):
            lo, hi = 0.0, 1.0
            for _ in range(14):
                mid = (lo + hi) / 2
                e, f = run(mode, W, aseq, mid, n=3000)
                if tps(e, f, L) < BASE:
                    lo = mid
                else:
                    hi = mid
            if hi < 0.995 and (best is None or hi < best[1]):
                best = (W, hi)
        print(f"  BREAK-EVEN {mode:<7}: " + (f"flat q >= {best[1]:.3f}, best depth W={best[0]}"
                                             if best else "UNREACHABLE — q=1.0 still loses at every W"))
