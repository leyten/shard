"""The SHIPPED refill floor + wide block (V4_DSPARK_BLOCK) on v4_pipe_sim's queueing chain — the
second machinery behind docs/V4_FILL_ECONOMICS.md. phase0/v4_ngram_econ.py prices the same arms
with a frame-exact replay that CLAMPS at the bottleneck rate; this one serves every frame through
v4_pipe_sim's per-stage FIFO chain, so it prices the queueing the clamp hides — the term that
decides the zero-benefit floor. v4_pipe_sim's own `rolling` arm is NOT the shipped lever (it
models chain-drafting off the horizon, values misaligned by construction); this drives the sim's
chain with coordinate_dspark_pipelined's actual round: the floor guard, topups past the deepest
in-flight frame only, cancels fencing the future.

Two topup-acceptance assumptions bracket the one number no model has:

  agree   a topped-up frame at slot j accepts at q_j — the fresh block re-predicts the in-flight
          prefix identically (the replay's assumption; topup_disagree ~ 0)
  reroll  it accepts at prod(q_1..q_j) — the fresh block re-rolls its whole prefix (maximal
          disagreement; docs/V4_MULTIBLOCK_VERDICT.md §4's hazard at full strength)

The truth is a per-run measurement (topup_agree/topup_disagree, topup_accept_by_depth); feed it
back by replacing the fitted curves below.

Run: python3 research/v4_fill_sim.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase0"))
import random
from collections import deque
from v4_pipe_sim import Ring, Stage, _Engine, _Chain, Frame

TAUS = [9.48, 8.24, 9.29, 8.67, 8.14, 6.00]
RING = Ring(stages=[Stage(p_ms=t, ckpt_ms=0.0, snd_ms=0.0, ovh_ms=0.0) for t in TAUS],
            hop_ms=110.7 / 7, draft_ms=0.0, logit_ms=0.0)


def sim_floor(ring, aseq, W=16, floor=1, mode="agree", n_tokens=4000, seed=7):
    """coordinate_dspark_pipelined's round with V4_REFILL_FLOOR, on the sim's stage chain."""
    block = len(aseq)
    rng = random.Random(seed)
    eng = _Engine()
    S = {"c": 0, "horizon": -1, "epoch": 0, "gen": 0, "cycles": 0, "t": 0.0,
         "frames": 0, "stale": 0}
    sent = {}
    outstanding = deque()
    depths = []
    truth = {}

    def T(p):
        return truth.setdefault(p, 1_000_000 + p)

    def feed(t, pos, tok):
        sent[pos] = tok
        S["horizon"] = max(S["horizon"], pos)
        S["frames"] += 1
        f = Frame(pos=pos, tok=tok, epoch=S["epoch"], s=1, committed=(tok == T(pos)))
        outstanding.append(f)
        chain.inject(t, f)

    def on_reply(t, f):
        S["t"] = t
        exp = outstanding.popleft()
        assert exp is f
        if f.epoch != S["epoch"]:
            S["stale"] += 1
            return
        pos = f.pos
        assert pos == S["c"], f"reply {pos} with the frontier at {S['c']}"
        m = T(pos + 1)
        fed = sent.get(pos + 1)
        S["c"] = pos + 1
        S["gen"] += 1
        if S["gen"] >= n_tokens:
            return
        if fed is None:
            feed(t, S["c"], m)
        elif fed != m:
            S["cycles"] += 1
            S["epoch"] += 1
            chain.cancel(S["epoch"])
            for p in [p for p in sent if p > pos]:
                del sent[p]
            S["horizon"] = pos
            feed(t, S["c"], m)
        # THE FLOOR, the shipped guard: consume this reply's block at or below the floor, streaming
        # only positions past the deepest frame in flight. Values drawn at stream time so the
        # conditional acceptance at judgment is the modelled q.
        if S["horizon"] - S["c"] + 1 <= floor:
            base = S["horizon"]
            topup = base > S["c"]
            for i in range(block):
                p = pos + 2 + i
                if p <= base:
                    continue                       # in flight: measured, never re-streamed
                if p - S["c"] >= W:
                    break
                j = p - S["c"]                     # slot index == draft depth
                if mode == "agree" or not topup:
                    ok = rng.random() < aseq[j - 1]
                else:                              # reroll: the whole re-predicted prefix must land
                    prod = 1.0
                    for x in aseq[:j]:
                        prod *= x
                    ok = rng.random() < prod
                feed(t, p, T(p) if ok else -p)
        depths.append(S["horizon"] - S["c"] + 1)
        sent.pop(pos - 1, None)

    chain = _Chain(eng, ring, on_reply)
    feed(0.0, 0, T(0))
    eng.run(until_fn=lambda: S["gen"] >= n_tokens)
    cyc = max(S["cycles"], 1)
    return dict(tok_s=1000.0 * S["gen"] / S["t"], g=S["gen"] / cyc,
                fpt=S["frames"] / S["gen"], stale=S["stale"],
                fbar=sum(depths) / len(depths))


FIT = [0.935, 0.925, 0.913, 0.899, 0.882]
CAL = [0.924, 0.906, 0.885, 0.859, 0.827]

for name, aseq in (("fit-to-g", FIT), ("calibrated", CAL)):
    q5 = aseq[-1]
    base = sim_floor(RING, aseq, floor=1)
    print(f"\n=== queueing sim, shipped semantics, drafter {name}  "
          f"(baseline {base['tok_s']:.2f} tok/s, g {base['g']:.2f}, f/t {base['fpt']:.2f})")
    print(f"  {'arm':<44}{'tok/s':>7}{'g':>7}{'f/t':>6}{'Fbar':>6}{'vs base':>9}")

    def row(lbl, r):
        print(f"  {lbl:<44}{r['tok_s']:>7.2f}{r['g']:>7.2f}{r['fpt']:>6.2f}{r['fbar']:>6.2f}"
              f"{(r['tok_s'] / base['tok_s'] - 1) * 100:>8.1f}%")

    for mode in ("agree", "reroll"):
        for fl in (3, 5):
            row(f"floor={fl} block=5 [{mode}]", sim_floor(RING, aseq, floor=fl, mode=mode))
        for k in (8, 10, 13):
            for tag, qd in (("q5", q5), (".50", 0.5), ("ZERO", 0.0)):
                wide = aseq + [qd] * (k - 5)
                row(f"wide k={k} floor={k} q_deep={tag} [{mode}]",
                    sim_floor(RING, wide, floor=k, mode=mode))
