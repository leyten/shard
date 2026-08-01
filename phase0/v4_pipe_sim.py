#!/usr/bin/env python3
"""Discrete-event simulator for the V4 DSpark ring — where the pipelined path loses its throughput.

WHY THIS EXISTS. `coordinate_dspark_pipelined` measured ~3.08 tok/s on the 6-box Nordic 5090 ring.
The first hypothesis was BUBBLES — that the pipeline sat empty. The coordinator's own counters killed
that: `max_inflight 7, mean_inflight 5.5, unsent_frames 0` on a SIX-stage ring. The pipe is full.
What it is full of is `stale_replies 51` against `accepted 33` — MISSPECULATION WASTE. A rejection
does not cost one frame, it costs every frame already in flight behind it, so waste scales with depth
times rejection rate, and depth is exactly what a deep pipeline needs to stay full. That tension is
the whole problem and it is what this simulator is for.

WHAT IT MODELS, and it is the real topology, not a cartoon:
  * a coordinator that injects step frames at the head and reads replies off the return leg;
  * D stages in a FIFO chain, each a SINGLE server with zero internal concurrency — that is exactly
    `v4_stage.Stage.forward` (validate -> `_seek` -> checkpoint -> compute -> return, one at a time,
    no prefetch, no double buffer), which is why a cancelled frame still costs the ring full service;
  * `coordinate_dspark_pipelined` reproduced statement for statement: the `sent` map, `horizon`, the
    epoch fence, the sender thread that drops queued frames of a dead epoch, and — the line that
    decides the shape of everything — `if blk and horizon == c:`, which streams a fresh block ONLY
    when the committed frontier has caught up with the deepest frame sent;
  * `coordinate_dspark`'s chunk round, including the `_seek`/`_replay` of the accepted prefix;
  * `coordinate`'s one-frame-per-token greedy loop.

VALIDATION. Two free parameters (per-position compute scale, per-leg latency) are fitted to TWO
throughputs. Four further observables are then PREDICTED, not fitted, and checked against the
coordinator's instrumented counters: `max_inflight`, `mean_inflight`, stale replies per cycle, and
cycles. Those come out of the coordinator's LOGIC and the acceptance rate, not out of the clock, so
matching them is a real test of whether the round structure in the sim is the round structure on the
ring. `--validate` prints all of it.

Pure stdlib. No torch, no GPU, no network.
"""

from __future__ import annotations

import argparse
import heapq
import random
import json
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ── THE MEASURED RING ──────────────────────────────────────────────────────────────────────────────
# Six distinct Nordic single-5090 boxes, 43 layers tiled 8/8/8/8/8/3, V4_TIMING, 2026-08-01.
# Source: phase0/v4_plan.py docstring (branch v4/speed-aware-tiling) and commit 4ae8efc.
# `fwd` and `on_box` are per-frame MEANS in ms (v4_pipe._StepTimer.report prints `acc/n`).

STAGES = [
    # id   layers  fwd_ms   on_box_ms  send_mbps
    ("s0",  8,     454.30,  460.30,    150.7),
    ("s1",  8,     357.61,  362.27,    177.6),
    ("s2",  8,     495.94,  503.39,     99.0),
    ("s3",  8,     489.61,  579.37,      5.6),   # 87.71 ms of its budget is send() — BROKEN UPLINK
    ("s4",  8,     200.39,  202.79,    285.4),
    ("s5",  3,      64.31,   97.68,   None),     # tail: +logits 12.35 +draft 20.32, no forward leg
]
ON_BOX = [s[3] for s in STAGES]                  # 2205.80 ms of real work per frame, over 6 stages
FWD = [s[2] for s in STAGES]
LAYERS = [s[1] for s in STAGES]
N_LAYERS = sum(LAYERS)                           # 43
TAIL_DRAFT_MS = 20.32                            # the tail's MTP advance + block, per drafted FRAME
TAIL_LOGIT_MS = 12.35                            # the tail's ParallelHead, per frame at s = 15
MEASURED_S = 15.0                                # mean positions per frame in the timed window
# s3 spent 87.71 ms in send() at 5.6 MB/s, so the frame payload is 87.71e-3 * 5.6 = 0.491 MB at s=15.
PAYLOAD_MB_AT_15 = 87.71 / 1000.0 * 5.6

# End-to-end throughputs, and the pipelined arm's INSTRUMENTED counters (48-token run, 2026-08-01).
MEASURED = {
    "greedy":        1.80,   # coordinate()                  — one s=1 frame per token, blocking
    "serial":        1.43,   # coordinate_dspark()           — one [cur]+drafts chunk per round
    "pipelined":     3.08,   # coordinate_dspark_pipelined() — the instrumented run (3.00-3.16)
    "pipelined_lo":  3.00,
    "pipelined_hi":  3.16,
    "g":             3.692,  # generated / cycles
    "block":         5,      # dspark_block_size
    # --- predicted, never fitted: these fall out of the coordinator's logic + the acceptance rate ---
    "generated":     48,
    "rounds":        13,     # cycles == cancels + 1
    "accepted":      33,     # draft tokens that matched
    "max_inflight":  7,
    "mean_inflight": 5.5,
    "stale_replies": 51,     # frames the ring computed and the fence then discarded
    "unsent_frames": 0,      # the sender never won the race to drop a queued frame: ALL of them ship
}

# THE DEPTH SWEEP — measured back-to-back on one warm 6-box ring, same prompt, 48 tokens, block 5.
# `same_output` was True at every depth: losslessness holds, depth changes what is SPECULATED, never
# what is committed. `unsent_frames` was 0 at every depth. This curve is the validation target.
#   depth = V4_SPEC_DEPTH (the in-flight cap the coordinator was launched with)
SWEEP = [
    # depth  tok/s   g     rounds stale  waste  max_inf  mean_inf  cancels
    (2,      2.487,  4.80,  10,    10,   0.217,   2,      2.00,     9),
    (3,      2.865,  4.36,  11,    15,   0.300,   3,      2.54,    10),
    (4,      3.257,  4.00,  12,    23,   0.404,   4,      3.13,    11),
    (6,      3.030,  3.69,  13,    42,   0.560,   6,      4.65,    12),
    (8,      2.597,  3.69,  13,    51,   0.607,   7,      5.50,    12),
    (16,     2.755,  3.69,  13,    51,   0.607,   7,      5.50,    12),
]
GEN = 48                                          # generated tokens in every sweep row
# `waste` = stale / (stale + accepted), so the accepted count is recoverable and is INDEPENDENT
# evidence about the drafter: accepted = stale*(1-w)/w  ->  36, 35, 34, 33, 33, 33.
# Per speculated position, acceptance = accepted / (accepted + cancels):
#   depth 2 (1 draft in flight)  0.800      depth 4 (3 drafts)  0.756
#   depth 3 (2 drafts)           0.778      depth 7 (6 drafts)  0.733
# Acceptance DECAYS with a draft's distance from the committed frontier at draft time — the MTP
# predicts the whole block from ONE hidden, so draft i conditions on i-1 unverified tokens. A model
# that treats acceptance as one number gets the optimal depth wrong.
# CAVEAT, recorded so nobody re-derives it in six weeks: the repo's own V4_PIPELINED_SPEC.md:36 reads
# "DSpark (1.34 effective tok/s) merely ties greedy (1.43)", i.e. it labels 1.43 as GREEDY on an
# EARLIER 7-country ring, while commits 89eaa3b/ba93551 label 1.43 as the SERIAL dspark baseline of
# this Nordic ring and give the pipelined result as 3.83 (and, 23 minutes later, 3.81). The greedy
# 1.80 comes from the ring session and is not in the repo. `--greedy` re-runs everything against a
# different greedy number; the sensitivity is reported in docs/V4_PIPELINE_EFFICIENCY.md.

# v4_stage.py:237-249, measured on the first live 7x5090 ring: what an s-position frame costs at one
# stage, in units of a single-token traversal. These are NOT free parameters.
CHUNK_MULT_AT_6 = 3.93          # `_run_chunk`, the fast verify path
LOOP_MULT_AT_6 = 5.88           # the per-position `_run` loop (fast verify off)


# ── ring parameters ────────────────────────────────────────────────────────────────────────────────

@dataclass
class Stage:
    """One box's cost model. Four charges, and each is a different function of the frame:

      p_ms    per POSITION of layer compute (this is the model, and it is what a wide ring divides)
      ckpt_ms per FRAME, and ONLY when speculation is armed — `Stage.forward` pushes a checkpoint
              whose `_snapshot()` clones every layer's window ring plus both accumulators of every
              compressor before it computes anything. It is inside the `fwd` phase, so it is inside
              the measured `on_box`, and it does NOT scale with s. Greedy never pays it (`coordinate`
              does not arm `spec`); a chunk round pays it once; the pipelined path pays it B+1 times
              per cycle. That asymmetry is exactly why the three measured throughputs cannot be
              reconciled without it.
      snd_ms  per POSITION of wire serialisation out of this box (payload bytes / measured uplink)
      ovh_ms  per FRAME of everything else the timer saw (pre/out)
    """
    p_ms: float
    ckpt_ms: float
    snd_ms: float
    ovh_ms: float
    layers: float = 0.0


@dataclass
class Ring:
    stages: List[Stage]
    hop_ms: float
    draft_ms: float = TAIL_DRAFT_MS         # tail: MTP advance + block, per drafted frame, per FRAME
    logit_ms: float = 0.0                   # tail: ParallelHead, per POSITION
    restore_ms: float = 0.0                 # `_seek` -> `_restore`: a device memcpy, no compute
    coord_ms: float = 0.0

    @property
    def D(self):
        return len(self.stages)

    @property
    def legs(self):
        """coord->head is one, D-1 between the stages, tail->coord is one."""
        return self.D + 1

    def tau(self, i, spec=True):
        """One s=1 frame at stage i: the pipeline's period is max over these."""
        s = self.stages[i]
        d = s.p_ms + s.snd_ms + s.ovh_ms + (s.ckpt_ms if spec else 0.0)
        if i == self.D - 1:
            d += self.logit_ms
        return d

    def taus(self, spec=True):
        return [self.tau(i, spec) for i in range(self.D)]

    @property
    def tau_max(self):
        return max(self.taus())

    @property
    def sum_tau(self):
        return sum(self.taus())

    @property
    def rtt_ms(self):
        """One s=1 frame all the way round with speculation OFF — exactly what greedy measures."""
        return sum(self.taus(spec=False)) + self.legs * self.hop_ms + self.coord_ms

    def frame_ms(self, i, s, spec=True, n_replay=0):
        st = self.stages[i]
        d = st.ovh_ms + s * st.snd_ms + chunk_mult(s) * st.p_ms
        if spec:
            d += st.ckpt_ms + n_replay * st.p_ms          # `_replay` re-drives, and pushes NO checkpoint
        if i == self.D - 1:
            d += s * self.logit_ms
        return d

    def balanced(self):
        m = statistics.fmean([s.p_ms for s in self.stages])
        mc = statistics.fmean([s.ckpt_ms for s in self.stages])
        return self._with(stages=[Stage(m, mc, s.snd_ms, s.ovh_ms, s.layers) for s in self.stages])

    def _with(self, **kw):
        d = dict(stages=[Stage(s.p_ms, s.ckpt_ms, s.snd_ms, s.ovh_ms, s.layers) for s in self.stages],
                 hop_ms=self.hop_ms, draft_ms=self.draft_ms, logit_ms=self.logit_ms,
                 restore_ms=self.restore_ms, coord_ms=self.coord_ms)
        d.update(kw)
        return Ring(**d)


# The timed window recorded s = 15.0 mean positions per frame. What an s-position frame costs at one
# stage, relative to an s=1 one, is v4_stage.py's own live-ring measurement extrapolated linearly:
# 1 at s=1, 3.93 at s=6 => slope 0.586/position => 9.20 at s=15. NOT a free parameter.
CHUNK_SLOPE = (CHUNK_MULT_AT_6 - 1.0) / 5.0
CHI15 = 1.0 + CHUNK_SLOPE * (MEASURED_S - 1.0)


def chunk_mult(s):
    return 1.0 if s <= 1 else 1.0 + CHUNK_SLOPE * (s - 1)


def measured_ring(kappa: float, phi: float, hop_ms: float, *, ckpt=True, **kw) -> Ring:
    """The measured ring under three free parameters.

        kappa  per-position layer compute, as a multiple of the box's measured `fwd`
        phi    per-FRAME speculative checkpoint, as a multiple of the same `fwd` (spec only)
        hop_ms one-way latency of one leg

    `fwd` is the SHAPE, not the scale: `kappa` and `phi` set the scale, because the timed frames were
    s = 15.0 mean and the chunk path's true multiplier at that width is not known (v4_stage.py measured
    it at s=6 only). The measured uplinks, the tail's logits and draft, and each box's frame overhead
    ARE used at their measured values. `ckpt=False` is the mutation: it deletes the checkpoint charge
    and is what makes the fit impossible, which is how we know the term is load-bearing."""
    sts = []
    for i, (_id, L, fwd, ob, mbps) in enumerate(STAGES):
        snd15 = 0.0 if mbps is None else PAYLOAD_MB_AT_15 / mbps * 1000.0
        ovh = ob - fwd - snd15
        if i == len(STAGES) - 1:
            ovh -= TAIL_LOGIT_MS + TAIL_DRAFT_MS
        sts.append(Stage(p_ms=kappa * fwd, ckpt_ms=(phi * fwd if ckpt else 0.0),
                         snd_ms=snd15 / MEASURED_S, ovh_ms=max(0.0, ovh), layers=L))
    return Ring(stages=sts, hop_ms=hop_ms, logit_ms=TAIL_LOGIT_MS / MEASURED_S, **kw)


def retile(base: Ring, n: int, *, speedup=1.0, hop_ms=None, quality=None) -> Ring:
    """The same 43 layers over `n` boxes. Layer work and checkpoint work are BOTH per-layer, so both
    divide; the per-frame overhead, the uplink cost and the wire legs do NOT — every extra box adds
    its own overhead and one more leg each way. `quality` resamples the measured 2.5x box-speed spread
    onto the new width, because a 10-box rental draws from the same pool and its slowest box prices
    every round."""
    per_layer_p = sum(s.p_ms for s in base.stages) / N_LAYERS
    per_layer_c = sum(s.ckpt_ms for s in base.stages) / N_LAYERS
    ovh = statistics.fmean([s.ovh_ms for s in base.stages])
    snd = statistics.median([s.snd_ms for s in base.stages])
    if quality is None:
        q0 = sorted((STAGES[i][2] / STAGES[i][1]) for i in range(len(STAGES)))
        mean_q = statistics.fmean(q0)
        q0 = [x / mean_q for x in q0]
        quality = [q0[round(i * (len(q0) - 1) / max(1, n - 1))] for i in range(n)]
    L = N_LAYERS / n
    sts = [Stage(p_ms=per_layer_p * L * quality[i] / speedup,
                 ckpt_ms=per_layer_c * L * quality[i] / speedup,
                 snd_ms=snd, ovh_ms=ovh, layers=L) for i in range(n)]
    return Ring(stages=sts, hop_ms=base.hop_ms if hop_ms is None else hop_ms,
                draft_ms=base.draft_ms, logit_ms=base.logit_ms, restore_ms=base.restore_ms,
                coord_ms=base.coord_ms)


# ── the engine ─────────────────────────────────────────────────────────────────────────────────────

class _Engine:
    __slots__ = ("t", "_q", "_n")

    def __init__(self):
        self.t = 0.0
        self._q: list = []
        self._n = 0

    def at(self, t, fn):
        self._n += 1
        heapq.heappush(self._q, (t, self._n, fn))

    def run(self, until_fn=None):
        while self._q:
            t, _, fn = heapq.heappop(self._q)
            self.t = t
            fn(t)
            if until_fn is not None and until_fn():
                return


@dataclass
class Frame:
    pos: int
    tok: int
    epoch: int
    s: int = 1
    committed: bool = False          # the tail's `acc`: this position is on the committed path
    n_replay: int = 0                # positions `_replay` re-drives on the rewind this frame causes
    reply_at: float = 0.0
    skipped: bool = False


class _Chain:
    """D single-server stages in a FIFO chain, plus a wire leg at each end."""

    def __init__(self, eng: _Engine, ring: Ring, on_reply, oob_cancel=False, spec=True):
        self.eng, self.ring, self.on_reply = eng, ring, on_reply
        self.oob_cancel = oob_cancel
        self.spec = spec
        self.D = ring.D
        self.busy = [False] * self.D
        self.q = [deque() for _ in range(self.D)]
        self.spos = [0] * self.D                       # each stage's `_pos`
        self.live_epoch = 0
        self.busy_ms = [0.0] * self.D
        self.wasted_ms = [0.0] * self.D
        self.served = [0] * self.D

    def inject(self, t, frame: Frame):
        self.eng.at(t + self.ring.hop_ms, lambda tt, f=frame: self._arrive(tt, 0, f))

    def _arrive(self, t, i, frame):
        self.q[i].append(frame)
        self._pump(t, i)

    def _pump(self, t, i):
        if self.busy[i] or not self.q[i]:
            return
        f = self.q[i].popleft()
        self.busy[i] = True
        d = self._service(i, f)
        self.busy_ms[i] += d
        self.served[i] += 1
        if f.epoch < self.live_epoch:
            self.wasted_ms[i] += d
        self.eng.at(t + d, lambda tt, ii=i, ff=f: self._done(tt, ii, ff))

    def _service(self, i, f: Frame) -> float:
        """What `Stage.forward` costs for this frame at this stage. `_seek` first (a restore, plus
        `_replay` of the accepted prefix inside the spent checkpoint — ZERO positions when the frames
        are s=1, which is precisely the pipelined path's structural advantage), then the checkpoint
        push, the frame itself, and — tail only, on a frame it judges committed — the MTP draft."""
        if self.oob_cancel and f.epoch < self.live_epoch:
            f.skipped = True
            return 0.0
        n_replay = f.n_replay if f.pos < self.spos[i] else 0
        d = self.ring.frame_ms(i, f.s, spec=self.spec, n_replay=n_replay)
        if n_replay or f.pos < self.spos[i]:
            d += self.ring.restore_ms
        if i == self.D - 1 and f.committed:
            d += self.ring.draft_ms
        self.spos[i] = f.pos + f.s
        return d

    def _done(self, t, i, f):
        self.busy[i] = False
        if i + 1 < self.D:
            self.eng.at(t + self.ring.hop_ms, lambda tt, ii=i + 1, ff=f: self._arrive(tt, ii, ff))
        else:
            f.reply_at = t + self.ring.hop_ms
            self.eng.at(f.reply_at, lambda tt, ff=f: self.on_reply(tt, ff))
        self._pump(t, i)

    def cancel(self, epoch):
        """What an out-of-band control path to every stage WOULD do. With `oob_cancel=False` this only
        records the epoch for the wasted-work accounting: the shipped in-band FIFO ring computes the
        stale frames anyway, because a cancel is injected at the head BEHIND them
        (V4_PIPELINED_SPEC.md 9.3)."""
        self.live_epoch = max(self.live_epoch, epoch)


# ── results ────────────────────────────────────────────────────────────────────────────────────────

@dataclass
class Result:
    mode: str
    tokens: int
    wall_ms: float
    toks_per_s: float
    cycles: int = 1
    g: float = 1.0
    frames: int = 0
    wasted_frames: int = 0
    util: List[float] = field(default_factory=list)
    mean_inflight: float = 0.0
    max_inflight: int = 0


# ── the three coordinators ─────────────────────────────────────────────────────────────────────────

def sim_greedy(ring: Ring, n_tokens=200) -> Result:
    """`coordinate`: one s=1 frame, block on the reply, repeat."""
    eng = _Engine()
    st = {"n": 0, "t": 0.0}

    def on_reply(t, f):
        st["n"] += 1
        st["t"] = t
        if st["n"] < n_tokens:
            chain.inject(t + ring.coord_ms, Frame(pos=f.pos + 1, tok=0, epoch=0, committed=False))

    chain = _Chain(eng, ring, on_reply, spec=False)
    chain.inject(0.0, Frame(pos=0, tok=0, epoch=0))
    eng.run()
    return Result(mode="greedy", tokens=n_tokens, wall_ms=st["t"],
                  toks_per_s=1000.0 * n_tokens / st["t"], cycles=n_tokens, g=1.0, frames=n_tokens,
                  util=[b / st["t"] for b in chain.busy_ms])


def accept_seq(a1: float, rho: float, n: int) -> List[float]:
    """Per-draft acceptance by POSITION IN THE BLOCK: a_i = a1 * rho^(i-1).

    Not a modelling flourish — it is what the ring measured. The MTP drafts the whole block from one
    committed hidden, so draft i conditions on i-1 tokens nobody has verified, and its hit rate falls
    off. `rho = 1` recovers the constant-acceptance model, which is the mutation that gets the optimal
    depth wrong."""
    return [a1 * rho ** i for i in range(n)]


def _run_len(a_seq: Sequence[float], block: int, rng) -> int:
    """Leading correct drafts in one block, drawn against the PER-INDEX acceptances.

    A seeded PRNG, not a low-discrepancy sequence: the quantile trick correlates the draws across
    indices and biases the run length short, which showed up immediately as a g the ring did not
    report. Seeded, so a sim result is reproducible and an intervention comparison is not reading
    sampling noise."""
    k = 0
    while k < block and rng.random() < (a_seq[k] if k < len(a_seq) else a_seq[-1]):
        k += 1
    return k


def accept_for_g(g_target: float, block: int) -> float:
    """Per-draft acceptance `a` with E[committed per cycle] = 1 + sum_{k=1..B} a^k = g. Monotone."""
    lo, hi = 0.0, 0.9999999
    for _ in range(200):
        a = (lo + hi) / 2
        if 1.0 + sum(a ** k for k in range(1, block + 1)) < g_target:
            lo = a
        else:
            hi = a
    return (lo + hi) / 2


def sim_serial(ring: Ring, accept, block=5, n_tokens=200, replay=True, seed=7) -> Result:
    """`coordinate_dspark`: `[cur] + drafts` as ONE frame of s = block+1, then block on the reply.

    The stage cost of that round is the design doc's "replay penalty" and it is TWO charges, both
    real: `_seek` rewinds onto the previous round's accepted prefix and `_replay` re-drives those
    positions ONE AT A TIME through every layer (eager, never graphed, never chunked), and only then
    does the new chunk itself cost `chunk_mult(s)`."""
    eng = _Engine()
    st = {"tok": 0, "t": 0.0, "cycles": 0, "frames": 0, "i": 0}
    pending: List[int] = []

    aseq = accept if isinstance(accept, (list, tuple)) else [accept] * (block + 2)
    rng = random.Random(seed)

    def send_round(t):
        k = _run_len(aseq, block, rng)
        st["i"] += 1
        committed = k + 1                                # accepted drafts + the correction/bonus
        pending.append(committed)
        st["frames"] += 1
        chain.inject(t, Frame(pos=0, tok=0, epoch=0, s=block + 1, committed=True,
                              n_replay=committed if replay else 0))

    def on_reply(t, f):
        st["t"] = t
        st["cycles"] += 1
        st["tok"] += pending.pop(0)
        if st["tok"] < n_tokens:
            send_round(t + ring.coord_ms)

    chain = _Chain(eng, ring, on_reply)
    chain.spos = [10 ** 9] * ring.D                      # every round after the first rewinds
    send_round(0.0)
    eng.run()
    return Result(mode="serial dspark", tokens=st["tok"], wall_ms=st["t"],
                  toks_per_s=1000.0 * st["tok"] / st["t"], cycles=st["cycles"],
                  g=st["tok"] / st["cycles"], frames=st["frames"],
                  util=[b / st["t"] for b in chain.busy_ms])


def sim_pipelined(ring: Ring, accept, block=5, W=16, n_tokens=200, *, oob_cancel=False,
                  rolling=False, label=None, seed=7) -> Result:
    """`coordinate_dspark_pipelined`, reproduced statement for statement.

    The line that decides the shape of everything is `if blk and horizon == c:` — a fresh block is
    streamed ONLY when the committed frontier has caught up with the deepest frame sent. Mid-run
    replies carry a block and it is deliberately dropped, so the in-flight depth decays B+1, B, ..., 1
    and only then refills. `rolling=True` deletes that gate (chain-drafting off the deepest in-flight
    position), which is the "continuous speculation instead of discrete rounds" intervention.

    `oob_cancel=True` is the out-of-band control path V4_PIPELINED_SPEC.md 9.3 says this topology does
    NOT have: stages skip frames of a dead epoch instead of computing them. It is an upper bound — the
    sim lets the cancel reach every stage instantly, which no real control path can."""
    aseq = accept if isinstance(accept, (list, tuple)) else [accept] * (block + 2)
    rng = random.Random(seed)
    eng = _Engine()
    truth: Dict[int, int] = {}

    def T(p):
        if p not in truth:
            truth[p] = 1_000_000 + p
        return truth[p]

    S = {"c": 0, "horizon": -1, "epoch": 0, "gen": 0, "cycles": 0, "t": 0.0,
         "frames": 0, "stale": 0, "blk": 0}
    sent: Dict[int, int] = {}
    outstanding: deque = deque()
    depths: List[int] = []

    def feed(t, pos, tok):
        sent[pos] = tok
        S["horizon"] = max(S["horizon"], pos)
        S["frames"] += 1
        f = Frame(pos=pos, tok=tok, epoch=S["epoch"], s=1, committed=(tok == T(pos)))
        outstanding.append(f)
        chain.inject(t, f)

    def draft_block(base):
        """The tail's block for frontier `base`: proposes base+1 .. base+block."""
        k = _run_len(aseq, block, rng)
        S["blk"] += 1
        return [T(base + 1 + i) if i < k else -(base + 1 + i) for i in range(block)]

    def on_reply(t, f):
        S["t"] = t
        exp = outstanding.popleft()
        assert exp is f
        if f.epoch != S["epoch"]:                        # THE FENCE
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
        if fed is None:                                  # nothing speculated here: feed the truth
            feed(t + ring.coord_ms, S["c"], m)
        elif fed != m:                                   # REJECT -> cancel, rewind, correct
            S["cycles"] += 1
            S["epoch"] += 1
            chain.cancel(S["epoch"])
            for p in [p for p in sent if p > pos]:
                del sent[p]
            S["horizon"] = pos
            feed(t + ring.coord_ms, S["c"], m)
        blk = draft_block(pos + 1) if f.committed else None
        gate = (S["horizon"] == S["c"]) if not rolling else (S["horizon"] - S["c"] + 1 < W)
        if blk and gate:
            base = S["horizon"] if rolling else pos + 1
            for i, d in enumerate(blk):
                p = base + 1 + i
                if p - S["c"] >= W:
                    break
                if p in sent:
                    continue
                feed(t + ring.coord_ms, p, d)
        depths.append(S["horizon"] - S["c"] + 1)
        sent.pop(pos - 1, None)

    chain = _Chain(eng, ring, on_reply, oob_cancel=oob_cancel)
    feed(0.0, 0, T(0))                                   # warm start at the frontier, as after prefill
    eng.run(until_fn=lambda: S["gen"] >= n_tokens)
    wall = S["t"]
    cycles = max(S["cycles"], 1)
    name = label or ("pipelined" + (" +oob" if oob_cancel else "") + (" +rolling" if rolling else ""))
    return Result(mode=name, tokens=S["gen"], wall_ms=wall, toks_per_s=1000.0 * S["gen"] / wall,
                  cycles=cycles, g=S["gen"] / cycles, frames=S["frames"],
                  wasted_frames=S["frames"] - S["gen"],
                  util=[b / wall for b in chain.busy_ms],
                  mean_inflight=round(statistics.fmean(depths), 2) if depths else 0.0,
                  max_inflight=max(depths) if depths else 0)


# ── calibration ────────────────────────────────────────────────────────────────────────────────────
#
# TWO STAGES, and they use DISJOINT evidence, which is what gives the result any force.
#
#   1. the DRAFTER (a1, rho) is fitted to the sweep's ACCEPTANCE column — accepted/(accepted+cancels)
#      at each depth. That column is a property of the model, not of the clock: no timing number is
#      used, and no timing number can be tuned to fix it.
#   2. the RING (kappa, hop) is fitted to the sweep's SIX tok/s points. Two free parameters, six
#      targets.
#
# Everything else is measured or structural: the per-stage fwd/on_box split, the five uplink rates,
# the payload size, the tail's logits and draft, the coordinator's round logic, and the block size.
# The remaining observables — max_inflight, mean_inflight, stale replies, cycles, and the greedy
# throughput — are then PREDICTED and checked. None of them is fitted to anything.

# The coordinator reported max_inflight 7 at W>=8 with dspark_block_size 5, and exactly W below that.
# The v4/full-stack source streams `[correction] + block` = B+1 = 6, so the shipped build streams one
# frame deeper. The sim uses the depth the ring actually showed.
BLOCK_EFF = 6

# No RTT was ever measured on this ring (V4_PIPELINED_SPEC.md 9.6 #1). A scattered Nordic ring cannot
# have legs below a few ms, so the fit is floored here rather than allowed to drive the wire to zero
# and blame everything on compute. Where it lands ON the floor, that is the finding: the wire is not
# what this ring is spending its time on.
HOP_FLOOR_MS = 4.0


def drafts_at(depth):
    """How many drafts the coordinator can have in flight at cap `depth`: `_feed` breaks at
    `p - c >= W`, and the correction frame at `c` occupies one slot."""
    return max(0, min(BLOCK_EFF, depth - 1))


def measured_acceptance():
    """accepted/(accepted+cancels) per sweep row, recovered from `waste` and `stale`."""
    out = []
    for depth, _t, _g, rounds, stale, waste, _mi, _mn, cancels in SWEEP:
        accepted = stale * (1.0 - waste) / waste
        out.append((drafts_at(depth), accepted / (accepted + cancels)))
    return out


def fit_drafter():
    """(a1, rho) from the acceptance column alone. No timing number is used."""
    obs = measured_acceptance()

    def model(a1, rho, k):
        a = accept_seq(a1, rho, max(1, k))
        w, num, den = 1.0, 0.0, 0.0
        for i in range(k):                       # weight index i by the chance of reaching it
            num += w * a[i]
            den += w
            w *= a[i]
        return num / den if den else a1

    best = (1e9, 0.8, 1.0)
    for ai in range(400):
        a1 = 0.60 + 0.001 * ai
        for ri in range(120):
            rho = 0.88 + 0.001 * ri
            e = max(abs(model(a1, rho, k) - v) for k, v in obs)
            if e < best[0]:
                best = (e, a1, rho)
    return {"err": best[0], "a1": best[1], "rho": best[2], "obs": obs,
            "fit": [(k, model(best[1], best[2], k)) for k, _ in obs]}


def _sweep_pred(kappa, hop, a1, rho, n_tokens):
    r = measured_ring(kappa, 0.0, hop)
    aseq = accept_seq(a1, rho, BLOCK_EFF + 1)
    return r, [sim_pipelined(r, aseq, BLOCK_EFF, W=d, n_tokens=n_tokens) for d, *_ in SWEEP]


def calibrate(*, n_tokens=120, draft=None):
    """(kappa, hop) against the six measured tok/s points of the depth sweep."""
    draft = draft or fit_drafter()
    a1, rho = draft["a1"], draft["rho"]
    tgt = [row[1] for row in SWEEP]
    cache = {}

    def cost(k, h):
        key = (round(k, 8), round(h, 5))
        if key in cache:
            return cache[key]
        if k <= 0 or h < HOP_FLOOR_MS:
            return 1e9, None, None
        r, res = _sweep_pred(k, h, a1, rho, n_tokens)
        errs = [((x.toks_per_s - t) / t) ** 2 for x, t in zip(res, tgt)]
        # greedy is the 7th target and the only DIRECT measurement of the round-trip: `coordinate`
        # sends one s=1 frame per token and blocks, so 1000/greedy IS the RTT. Without it the fit is
        # degenerate — the sweep alone pins RTT and tau_max but not how RTT splits between compute
        # and wire, and the optimiser drives the wire to zero.
        gp = sim_greedy(r, n_tokens).toks_per_s
        errs.append(((gp - MEASURED["greedy"]) / MEASURED["greedy"]) ** 2)
        cache[key] = ((sum(errs) / len(errs)) ** 0.5, r, res)
        return cache[key]

    best = (1e9, 0.05, 20.0, None, None)
    for ki in range(26):
        for hi in range(21):
            k, h = 0.01 + 0.012 * ki, HOP_FLOOR_MS + 3.0 * hi
            e, r, res = cost(k, h)
            if e < best[0]:
                best = (e, k, h, r, res)
    k, h = best[1], best[2]
    step = [0.006, 2.0]
    for _ in range(40):
        moved = False
        for j in (0, 1):
            for sgn in (1, -1):
                q = [k, h]
                q[j] += sgn * step[j]
                e, r, res = cost(*q)
                if e < best[0] - 1e-12:
                    best, k, h, moved = (e, q[0], q[1], r, res), q[0], q[1], True
        if not moved:
            step = [x / 2 for x in step]
            if step[0] < 1e-6:
                break
    return {"rms": best[0], "kappa": best[1], "hop_ms": best[2], "ring": best[3], "res": best[4],
            "a1": a1, "rho": rho, "draft": draft, "block": BLOCK_EFF,
            "aseq": accept_seq(a1, rho, BLOCK_EFF + 1)}


# ── reports ────────────────────────────────────────────────────────────────────────────────────────

def _hr(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


def _bdp(r):
    return r.rtt_ms / max(r.taus())


def validate(n_tokens=300):
    _hr("VALIDATION — 2 free ring parameters against 6 measured throughputs, then 4 predicted counters")
    d = fit_drafter()
    print(f"  STEP 1 — the drafter, from the ACCEPTANCE column only (no timing number is used):")
    print(f"    a_i = {d['a1']:.3f} x {d['rho']:.3f}^(i-1)   worst residual {d['err'] * 100:.2f}%")
    print(f"    {'drafts in flight':>18}{'measured a':>12}{'model':>9}")
    for (k, v), (_k, f) in zip(d["obs"], d["fit"]):
        print(f"    {k:>18}{v:>12.3f}{f:>9.3f}")
    print(f"    -> acceptance of the LAST draft in a full block is "
          f"{d['a1'] * d['rho'] ** (BLOCK_EFF - 1):.3f} vs {d['a1']:.3f} for the first.")

    cal = calibrate(n_tokens=n_tokens, draft=d)
    r = cal["ring"]
    tp = r.taus()
    print(f"\n  STEP 2 — the ring, from the six tok/s points: kappa={cal['kappa']:.4f}  "
          f"hop={cal['hop_ms']:.1f} ms   RMS {cal['rms'] * 100:.1f}%")
    print(f"    tau  [{', '.join(f'{x:.0f}' for x in tp)}] ms/frame   sum {sum(tp):.0f}, "
          f"max {max(tp):.0f} (stage s{tp.index(max(tp))})")
    print(f"    RTT  {r.rtt_ms:.0f} ms  ({r.legs * r.hop_ms:.0f} ms wire over {r.legs} legs, "
          f"{sum(tp):.0f} ms compute)")
    print(f"    per-layer {sum(st.p_ms for st in r.stages) / N_LAYERS:.2f} ms/layer/token")
    print(f"    BANDWIDTH-DELAY PRODUCT  RTT/tau_max = {_bdp(r):.1f} frames")

    print(f"\n  STEP 3 — everything else, PREDICTED:")
    hdr = (f"    {'depth':>6}|{'tok/s meas':>11}{'sim':>7}{'err':>7}|{'g meas':>8}{'sim':>6}"
           f"|{'stale meas':>11}{'sim':>6}|{'maxinf':>7}{'sim':>5}|{'meaninf':>8}{'sim':>6}")
    print(hdr)
    worst = {"g": 0.0, "stale": 0.0, "inf": 0.0, "tok": 0.0}
    for (depth, tok, g, rounds, stale, waste, mi, mn, canc), x in zip(SWEEP, cal["res"]):
        sc = GEN / x.tokens
        e_t = (x.toks_per_s - tok) / tok
        worst["tok"] = max(worst["tok"], abs(e_t))
        worst["g"] = max(worst["g"], abs(x.g - g) / g)
        worst["stale"] = max(worst["stale"], abs(x.wasted_frames * sc - stale) / stale)
        worst["inf"] = max(worst["inf"], abs(x.mean_inflight - mn) / mn,
                           abs(x.max_inflight - mi) / mi)
        print(f"    {depth:>6}|{tok:>11.3f}{x.toks_per_s:>7.2f}{100 * e_t:>6.0f}%"
              f"|{g:>8.2f}{x.g:>6.2f}|{stale:>11.0f}{x.wasted_frames * sc:>6.0f}"
              f"|{mi:>7}{x.max_inflight:>5}|{mn:>8.2f}{x.mean_inflight:>6.2f}")
    gre = sim_greedy(r, n_tokens).toks_per_s
    ser = sim_serial(r, cal["aseq"], MEASURED["block"], n_tokens).toks_per_s
    print(f"\n    fitted   tok/s across the sweep: worst {worst['tok'] * 100:.0f}%, "
          f"RMS {cal['rms'] * 100:.0f}%")
    for k, lbl in (("g", "g (tokens/cycle)"), ("stale", "stale replies"), ("inf", "in-flight depth")):
        print(f"    PREDICTED {lbl:<20} worst residual {worst[k] * 100:5.1f}%  "
              f"{'PASS' if worst[k] <= 0.20 else 'FAIL'}")
    print(f"    PREDICTED greedy tok/s         {gre:.2f} vs measured {MEASURED['greedy']:.2f} "
          f"({100 * (gre - MEASURED['greedy']) / MEASURED['greedy']:+.0f}%)")
    print(f"    PREDICTED serial tok/s         {ser:.2f} vs measured {MEASURED['serial']:.2f} "
          f"({100 * (ser - MEASURED['serial']) / MEASURED['serial']:+.0f}%)")
    w = max(worst["g"], worst["stale"], worst["inf"])
    print(f"\n  worst residual among the PREDICTED (unfitted) counters: {w * 100:.1f}%   "
          f"{'PASS (<= 20%)' if w <= 0.20 else 'FAIL'}")

    print(f"\n  MUTATION — the anti-vacuity check. Re-fit (kappa, hop) with rho pinned to 1.0, i.e.")
    print(f"  acceptance treated as one number instead of decaying with draft index:")
    flat = {"a1": 0.762, "rho": 1.0, "err": 0.0, "obs": d["obs"], "fit": d["fit"]}
    cflat = calibrate(n_tokens=n_tokens, draft=flat)
    best_flat = max(range(len(SWEEP)), key=lambda i: cflat["res"][i].toks_per_s)
    best_real = max(range(len(SWEEP)), key=lambda i: cal["res"][i].toks_per_s)
    print(f"    decaying acceptance: RMS {cal['rms'] * 100:5.1f}%, optimum at depth "
          f"{SWEEP[best_real][0]}   (measured optimum: depth 4)")
    print(f"    flat acceptance:     RMS {cflat['rms'] * 100:5.1f}%, optimum at depth "
          f"{SWEEP[best_flat][0]}")
    print(f"    g at depth 2 / depth 16: decaying {cal['res'][0].g:.2f}/{cal['res'][-1].g:.2f}, "
          f"flat {cflat['res'][0].g:.2f}/{cflat['res'][-1].g:.2f}, measured 4.80/3.69")
    return cal, w


def best_depth(ring, aseq, block=None, n_tokens=200, dmax=14, **kw):
    """Sweep the in-flight cap and return (depth*, result*, [(depth, tok/s)])."""
    block = BLOCK_EFF if block is None else block
    curve = [(d, sim_pipelined(ring, aseq, block, W=d, n_tokens=n_tokens, **kw))
             for d in range(2, dmax + 1)]
    star = max(curve, key=lambda x: x[1].toks_per_s)
    return star[0], star[1], [(d, x.toks_per_s) for d, x in curve]


def depth_rule(cal, n_tokens=200):
    """(a) The optimal depth as a rule, not a number: against stage count and against acceptance.

    The mechanism is a tug of war. Depth is what keeps the bottleneck fed, and the depth needed for
    that is the BANDWIDTH-DELAY PRODUCT, RTT / tau_max — below it the ring runs at depth/RTT instead
    of 1/tau_max. Depth is also what a rejection destroys, and it destroys everything in flight, so
    waste grows linearly in it. And — the part that is not classic BDP — depth LOWERS g, because the
    MTP drafts the whole block from one hidden and draft i conditions on i-1 unverified tokens."""
    r, aseq = cal["ring"], cal["aseq"]
    _hr("(a) THE OPTIMAL SPECULATION DEPTH — a rule, for tuning a ring you have not built yet")
    print("  vs STAGE COUNT (43 layers re-tiled; each box adds a leg of latency and its own overhead)")
    print(f"    {'D':>4}{'tau_max':>9}{'RTT':>7}{'BDP':>7}{'depth*':>8}{'tok/s*':>9}"
          f"{'g*':>6}{'waste*':>8}{'vs D=6*':>9}")
    ref = None
    for d in (3, 4, 5, 6, 8, 10, 12, 16):
        rr = retile(r, d)
        ds, x, _ = best_depth(rr, aseq, n_tokens=n_tokens)
        ref = ref if ref is not None else (x.toks_per_s if d == 6 else None)
        print(f"    {d:>4}{max(rr.taus()):>9.0f}{rr.rtt_ms:>7.0f}{_bdp(rr):>7.1f}{ds:>8}"
              f"{x.toks_per_s:>9.2f}{x.g:>6.2f}{100 * x.wasted_frames / x.frames:>7.0f}%"
              + (f"{'':>9}" if ref is None else
                 f"{100 * (x.toks_per_s / ref - 1):>8.0f}%"))
        if d == 6:
            ref = x.toks_per_s
    print("\n  vs ACCEPTANCE (a1 scaled; the same 0.945 per-index decay), on the measured D=6 ring")
    print(f"    {'a1':>6}{'g@d=7':>8}{'BDP':>7}{'depth*':>8}{'tok/s*':>9}")
    for a1 in (0.65, 0.70, 0.75, 0.796, 0.85, 0.90, 0.95):
        aq = accept_seq(a1, cal["rho"], BLOCK_EFF + 1)
        ds, x, _ = best_depth(r, aq, n_tokens=n_tokens)
        g7 = sim_pipelined(r, aq, BLOCK_EFF, W=16, n_tokens=n_tokens).g
        print(f"    {a1:>6.3f}{g7:>8.2f}{_bdp(r):>7.1f}{ds:>8}{x.toks_per_s:>9.2f}")
    print("\n  THE RULE:  depth* = clamp( round(RTT / tau_max) - 1 , 2 , B+2 )")
    print("  i.e. one below the bandwidth-delay product. It sits BELOW the BDP rather than at it")
    print("  because the last frame of depth buys the least fill (the bottleneck is already nearly")
    print("  saturated) and costs the most waste (it is the draft with the lowest acceptance).")
    print("  Acceptance barely moves it: a1 has to change a lot to shift depth* by one frame, so")
    print("  tune depth off the RING's RTT/tau_max, not off the drafter.")


def width(cal, n_tokens=200):
    """(b) Does width still pay once misspeculation AND the g(depth) coupling are modelled?"""
    r, aseq = cal["ring"], cal["aseq"]
    d6, x6, _ = best_depth(r, aseq, n_tokens=n_tokens)
    _hr("(b) THE 10-STAGE RING — width, priced against waste")
    for d in (6, 10):
        rr = retile(r, d)
        ds, x, curve = best_depth(rr, aseq, n_tokens=n_tokens)
        dflt = sim_pipelined(rr, aseq, BLOCK_EFF, W=16, n_tokens=n_tokens)
        print(f"  D={d:<3} tau_max {max(rr.taus()):.0f} ms  RTT {rr.rtt_ms:.0f} ms  "
              f"BDP {_bdp(rr):.1f}  ceiling 1/tau_max {1000 / max(rr.taus()):.2f} tok/s")
        print(f"        depth* {ds}  ->  {x.toks_per_s:.2f} tok/s  (g {x.g:.2f}, waste "
              f"{100 * x.wasted_frames / x.frames:.0f}%)")
        print(f"        at the SHIPPED default V4_SPEC_DEPTH=16 -> {dflt.toks_per_s:.2f} tok/s "
              f"({100 * (dflt.toks_per_s / x.toks_per_s - 1):+.0f}% vs tuned)")
        print(f"        curve: " + "  ".join(f"{dd}:{v:.2f}" for dd, v in curve[:10]))
    r10 = retile(r, 10)
    d10, x10, _ = best_depth(r10, aseq, n_tokens=n_tokens)
    print(f"\n  D=10 vs D=6, both at their own optimal depth: "
          f"{x10.toks_per_s:.2f} vs {x6.toks_per_s:.2f} = {100 * (x10.toks_per_s / x6.toks_per_s - 1):+.0f}%")
    print(f"  The naive expectation (width halves tau_max, so it doubles throughput) is "
          f"{1000 / max(r10.taus()) / (1000 / max(r.taus())):.2f}x on the CEILING and it does not")
    print(f"  survive: a wider ring has a higher BDP, so it needs MORE frames in flight, and every")
    print(f"  one of them dies with the next rejection.")
    print(f"\n  what g would width need to pay off like the naive model?")
    print(f"    {'a1':>6}{'g':>7}{'D=6':>8}{'D=10':>8}{'gain':>8}")
    for a1 in (0.70, 0.796, 0.85, 0.90, 0.95, 0.98):
        aq = accept_seq(a1, cal["rho"], BLOCK_EFF + 1)
        _, xa, _ = best_depth(r, aq, n_tokens=n_tokens)
        _, xb, _ = best_depth(r10, aq, n_tokens=n_tokens)
        g = sim_pipelined(r, aq, BLOCK_EFF, W=16, n_tokens=n_tokens).g
        print(f"    {a1:>6.2f}{g:>7.2f}{xa.toks_per_s:>8.2f}{xb.toks_per_s:>8.2f}"
              f"{100 * (xb.toks_per_s / xa.toks_per_s - 1):>7.0f}%")


def gradient(cal, n_tokens=200):
    """(c) d(tok/s)/dg at the optimum, so acceptance work can be priced against compute work."""
    r, rho = cal["ring"], cal["rho"]
    _hr("(c) THE GRADIENT — what an acceptance point is worth, at the optimal depth")
    print(f"    {'a1':>6}{'g':>7}{'depth*':>8}{'tok/s':>9}{'d(tok/s)/dg':>14}")
    prev, rows = None, []
    for a1 in (0.70, 0.75, 0.796, 0.85, 0.90):
        aq = accept_seq(a1, rho, BLOCK_EFF + 1)
        ds, x, _ = best_depth(r, aq, n_tokens=n_tokens)
        g = sim_pipelined(r, aq, BLOCK_EFF, W=16, n_tokens=n_tokens).g
        d = "" if prev is None else f"{(x.toks_per_s - prev[1]) / (g - prev[0]):>14.3f}"
        print(f"    {a1:>6.3f}{g:>7.2f}{ds:>8}{x.toks_per_s:>9.2f}{d}")
        rows.append((g, x.toks_per_s))
        prev = (g, x.toks_per_s)
    lo, hi = rows[1], rows[3]
    slope = (hi[1] - lo[1]) / (hi[0] - lo[0])
    base = [v for gg, v in rows if abs(gg - rows[2][0]) < 1e-9][0]
    print(f"\n    central difference around g={rows[2][0]:.2f}:  "
          f"d(tok/s)/dg = {slope:.3f} tok/s per unit g   ({100 * slope / base:.0f}% per unit g)")
    print(f"    compare COMPUTE: 2x on tau gives "
          f"{best_depth(retile(r, 6, speedup=2.0), cal['aseq'], n_tokens=n_tokens)[1].toks_per_s:.2f} "
          f"tok/s from {base:.2f} "
          f"({100 * (best_depth(retile(r, 6, speedup=2.0), cal['aseq'], n_tokens=n_tokens)[1].toks_per_s / base - 1):.0f}%)")
    print(f"    so +1.0 of g ~ {slope / base * 100:.0f}%, and g can only reach "
          f"{1 + 1 / (1 - 0.796):.1f} even at a=1 for the last draft; compute has no such cap.")


def interventions(cal, n_tokens=200):
    r, aseq = cal["ring"], cal["aseq"]
    d0, base, _ = best_depth(r, aseq, n_tokens=n_tokens)
    shipped = sim_pipelined(r, aseq, BLOCK_EFF, W=16, n_tokens=n_tokens)
    _hr("INTERVENTIONS — ranked, each measured at ITS OWN optimal depth")
    rows = [("0. shipped today (V4_SPEC_DEPTH=16)", shipped, 16),
            (f"1. tune depth to {d0} (free, today)", base, d0)]
    def add(name, rr, **kw):
        d, x, _ = best_depth(rr, aseq, n_tokens=n_tokens, **kw)
        rows.append((name, x, d))
    add("2. out-of-band cancel", r, oob_cancel=True)
    add("3. balanced stages", r.balanced())
    fx = statistics.fmean([st.ovh_ms for i, st in enumerate(r.stages) if i != 3])
    sd = statistics.median([st.snd_ms for i, st in enumerate(r.stages) if i != 3])
    r_fix = r._with(stages=[Stage(x.p_ms, x.ckpt_ms, sd if i == 3 else x.snd_ms,
                                  fx if i == 3 else x.ovh_ms, x.layers)
                            for i, x in enumerate(r.stages)])
    add("4. replace the 5.6 MB/s box", r_fix)
    add("5. WIDE: 10 stages", retile(r, 10))
    add("6. NARROW: 4 fat stages", retile(r, 4))
    add("7. 2x compute", retile(r, 6, speedup=2.0))
    add("8. 4x compute", retile(r, 6, speedup=4.0))
    aq9 = accept_seq(0.90, cal["rho"], BLOCK_EFF + 1)
    d9, x9, _ = best_depth(r, aq9, n_tokens=n_tokens)
    rows.append(("9. drafter to a1=0.90 (g 3.7->5.4)", x9, d9))
    for name, x, d in rows:
        print(f"  {name:<40}{x.toks_per_s:>7.2f} tok/s"
              f"{100 * (x.toks_per_s / shipped.toks_per_s - 1):>8.0f}%   depth {d:<3} "
              f"waste {100 * x.wasted_frames / x.frames:>3.0f}%  bottleneck {max(x.util) * 100:>5.1f}%")
    _hr("COMBINATIONS, each at its own optimal depth")
    combos = [
        ("tuned depth + balanced + fixed uplink", r_fix.balanced(), {}, aseq),
        ("... + out-of-band cancel", r_fix.balanced(), dict(oob_cancel=True), aseq),
        ("... + D=10", retile(r_fix.balanced(), 10), dict(oob_cancel=True), aseq),
        ("... + 2x compute", retile(r_fix.balanced(), 10, speedup=2.0), dict(oob_cancel=True), aseq),
        ("... + 4x compute", retile(r_fix.balanced(), 10, speedup=4.0), dict(oob_cancel=True), aseq),
        ("... + drafter a1=0.90", retile(r_fix.balanced(), 10, speedup=4.0), dict(oob_cancel=True),
         accept_seq(0.90, cal["rho"], BLOCK_EFF + 1)),
        ("... + 8x compute, tight 5ms hops",
         retile(r_fix.balanced(), 10, speedup=8.0, hop_ms=5.0), dict(oob_cancel=True),
         accept_seq(0.90, cal["rho"], BLOCK_EFF + 1)),
    ]
    for name, rr, kw, aq in combos:
        d, x, _ = best_depth(rr, aq, n_tokens=n_tokens, **kw)
        print(f"  {name:<44}{x.toks_per_s:>7.2f} tok/s   depth {d:<3} "
              f"(ceiling 1/tau_max {1000 / max(rr.taus()):.1f})")


def reach(cal, n_tokens=200, target=20.0):
    """What ring shape reaches the target — and what becomes the binding constraint on the way.

    Every ring-shape lever is bounded: width is +16%, balancing +24%, out-of-band cancel +19%, a
    better drafter +23%. They multiply to well under 2x. Only tau itself is unbounded, so the sweep
    that matters is over COMPUTE, and the thing to watch on the way is which constraint takes over."""
    r, aseq = cal["ring"], cal["aseq"]
    aq9 = accept_seq(0.90, cal["rho"], 24)
    _hr(f"WHAT REACHES {target:.0f} tok/s — sweeping compute, with every ring lever already applied")
    print( "  configuration: balanced, uplink fixed, out-of-band cancel, drafter a1=0.90, own depth*")
    print(f"    {'compute':>8}{'D':>4}{'B':>4}{'hop':>6}{'tau_max':>9}{'RTT':>7}{'BDP':>6}"
          f"{'depth*':>8}{'ceiling':>9}{'tok/s':>8}")
    fx = statistics.fmean([st.ovh_ms for i, st in enumerate(r.stages) if i != 3])
    sd = statistics.median([st.snd_ms for i, st in enumerate(r.stages) if i != 3])
    rf = r._with(stages=[Stage(x.p_ms, x.ckpt_ms, sd if i == 3 else x.snd_ms,
                               fx if i == 3 else x.ovh_ms, x.layers) for i, x in enumerate(r.stages)])
    for sp, D, B, hop in [(1, 10, 6, None), (2, 10, 6, None), (4, 10, 6, None), (8, 10, 6, None),
                          (8, 10, 12, None), (8, 10, 12, 5.0), (16, 10, 12, 5.0),
                          (8, 6, 12, 5.0), (8, 16, 12, 5.0), (16, 16, 16, 5.0)]:
        rr = retile(rf.balanced(), D, speedup=sp, hop_ms=hop).balanced()
        d, x, _ = best_depth(rr, aq9, block=B, n_tokens=n_tokens, dmax=B + 3, oob_cancel=True)
        print(f"    {sp:>7}x{D:>4}{B:>4}{rr.hop_ms:>6.0f}{max(rr.taus()):>9.1f}{rr.rtt_ms:>7.0f}"
              f"{_bdp(rr):>6.1f}{d:>8}{1000 / max(rr.taus()):>9.1f}{x.toks_per_s:>8.2f}")
    print("\n  The block size is what stops the wide/fast rings: the coordinator can never hold more")
    print("  than B+2 frames in flight, so once RTT/tau_max climbs past that the pipe cannot fill and")
    print("  throughput stops tracking 1/tau_max. B is not binding today (depth* 6 < 7); it becomes")
    print("  the binding constraint the moment compute improves.")


def anatomy(cal, n_tokens=200):
    """Where the wall clock goes at the SHIPPED configuration — fill loss against waste loss."""
    r, aseq = cal["ring"], cal["aseq"]
    x = sim_pipelined(r, aseq, BLOCK_EFF, W=16, n_tokens=n_tokens)
    tp = r.taus()
    _hr("WHERE THE THROUGHPUT GOES — it is NOT bubbles")
    print(f"  pipeline-rate ceiling  1/tau_max        {1000 / max(tp):6.2f} tok/s   "
          f"(tau_max {max(tp):.0f} ms, stage s{tp.index(max(tp))})")
    print(f"  achieved at V4_SPEC_DEPTH=16            {x.toks_per_s:6.2f} tok/s   "
          f"= {100 * x.toks_per_s * max(tp) / 1000:.0f}% of it")
    print(f"  bottleneck stage busy                   {max(x.util) * 100:5.1f}% of the wall clock")
    print(f"  frames computed per token committed     {x.frames / x.tokens:6.2f}")
    print(f"  of the frames computed, discarded       {100 * x.wasted_frames / x.frames:5.0f}%")
    print(f"\n  fill loss  (bottleneck idle)            {100 - max(x.util) * 100:5.1f}%")
    print(f"  waste loss (computed, then discarded)   {100 * x.wasted_frames / x.frames:5.1f}%")
    print(f"  RTT {r.rtt_ms:.0f} ms = {sum(tp):.0f} ms compute + {r.legs * r.hop_ms:.0f} ms wire "
          f"({100 * r.legs * r.hop_ms / r.rtt_ms:.0f}% wire)")
    return x


def main():
    ap = argparse.ArgumentParser(description="V4 pipelined-speculation ring simulator")
    ap.add_argument("--validate", action="store_true", help="calibration + validation only")
    ap.add_argument("--tokens", type=int, default=300,
                    help="tokens per simulated run. Below ~200 the stale-reply count is dominated "
                         "by sampling noise and the validation will report a false FAIL.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    cal, worst = validate(n_tokens=args.tokens)
    if args.validate:
        return 0 if worst <= 0.20 else 1
    anatomy(cal, args.tokens)
    depth_rule(cal, args.tokens)
    width(cal, args.tokens)
    gradient(cal, args.tokens)
    interventions(cal, args.tokens)
    reach(cal, args.tokens)
    if args.json:
        r = cal["ring"]
        print("\n" + json.dumps({"kappa": cal["kappa"], "hop_ms": cal["hop_ms"], "a1": cal["a1"],
                                 "rho": cal["rho"], "tau_ms": r.taus(), "rtt_ms": r.rtt_ms,
                                 "bdp": _bdp(r), "worst_predicted_residual": worst}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
