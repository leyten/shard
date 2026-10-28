#!/usr/bin/env python3
"""shard v2 / workstream 2 — generic pipelined-speculation simulator, calibrated to M2.5 and MiMo.

Promoted from scratchpad/v2 for HW1 (H0). Pure stdlib. Built on the round logic of
engines/deepseek_v4/v4_pipe_sim.py (loaded from this checkout, so it never drifts from the
canonical sim) and CROSS-CHECKED against it (`--selfcheck`): on a V4-shaped ring with a tail
drafter charged per committed frame, this file's per-token round must reproduce
v4_pipe_sim.sim_pipelined's tok/s, g and in-flight exactly (same RNG draws, same gate).

What it adds over v4_pipe_sim, and why each is needed for the M2.5/MiMo question:
  * per-model s-scaling of a stage (`Ring.cost`): V4's chunk path is ~linear in positions (3.93x at
    s=6) and replays; an MoE stage with graphed [1,K+1] rows is expert-DEDUP-bound (distinct experts
    touched by S rows = E*(1-(1-k/E)^S)) and never replays. This single curve decides whether
    streaming s=1 frames beats one s=K+1 chunk.
  * drafter LOCATION: 'head' (coordinator on the head box, a serial GPU server — M2.5's EAGLE) vs
    'tail' (drafts inside the tail's service — V4's DSpark, a tail-local DFlash), lazy or eager.
  * four coordinators on one engine: greedy, CHUNK depth-1 (M2.5's shipped EAGLE round),
    CHUNK depth-D stacked (the 07-02 panel's killed lever — n-gram-style stale chunks), and
    PER-TOKEN pipelined (V4's coordinate_dspark_pipelined round, F=1 drain gate, W cap).
  * MANY SESSIONS on one ring: stages optionally BATCH every queued frame into one forward
    (continuous batching) or serve them one by one (M2.5 de-lockstep solo frames).
"""
from __future__ import annotations

import argparse
import math
import importlib.util
import os
import random
import statistics
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_SIM_SRC = os.path.join(REPO, "engines", "deepseek_v4", "v4_pipe_sim.py")
ROWS_TABLE = None       # --calib: {rows: cost(rows)/cost(1)} measured on HW1 (A2 + A1), replaces the model


def load_v4sim():
    spec = importlib.util.spec_from_file_location("v4_pipe_sim", _SIM_SRC)
    m = importlib.util.module_from_spec(spec)
    sys.modules["v4_pipe_sim"] = m
    spec.loader.exec_module(m)
    return m


V4 = load_v4sim()
_run_len = V4._run_len
_Engine = V4._Engine


# ── acceptance ─────────────────────────────────────────────────────────────────────────────────────

def aseq(a1, rho, n):
    return [a1 * rho ** i for i in range(n)]


def g_of(a, K):
    """E[committed per round] for a K-draft block = 1 + sum_k prod_{i<=k} a_i  (tau in DFlash terms)."""
    g, p = 1.0, 1.0
    for i in range(K):
        p *= a[i] if i < len(a) else a[-1]
        g += p
    return g


def fit_a1(g_target, K, rho):
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if g_of(aseq(mid, rho, K), K) < g_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ── the ring ───────────────────────────────────────────────────────────────────────────────────────

def distinct(S, E=256, k=8):
    return E * (1.0 - (1.0 - k / E) ** S)


@dataclass
class Ring:
    """p1[i]: stage i compute for ONE position (ms). ovh[i]: per-frame fixed cost (ms).
    r9: cost(S=9)/cost(S=1) for one frame — the s-scaling of ONE row. `scal` picks the curve:
      'moe'    expert-dedup + small linear term, pinned so cost(9)/cost(1) == r9 (M2.5, MiMo)
      'linear' V4's measured chunk slope (0.586/position) — the pathological case
    legs: D+1 one-way latencies (coord->head, stage->stage..., tail->coord)."""
    p1: List[float]
    ovh: List[float]
    legs: List[float]
    r9: float = 2.0
    scal: str = "moe"
    E: int = 256
    k: int = 8
    draft_where: str = "head"       # 'head' | 'tail'
    t_block: float = 11.0           # ms to draft one block (EAGLE3 K=8 serial ~11 ms measured 07-18)
    draft_lazy: bool = True         # tail drafts only when the round will consume the block
    coord_ms: float = 0.3
    batch_rows: int = 1             # >1: a stage serves every queued frame (<= rows) as ONE forward
    # WAN links (legs 1..D) as SERIAL servers: per-message cost + bytes/bandwidth, then `legs` of pure
    # propagation. A stage batch leaves as ONE message. msg_ms=0 and bw=None -> pure-delay legs.
    msg_ms: float = 0.0
    kb_per_pos: float = 6.0         # fp8 hidden 3 KB + EAGLE aux share (L30/L58 bf16 on later legs)
    bw_MBps: Optional[float] = None

    @property
    def D(self):
        return len(self.p1)

    def mult(self, S):
        if S <= 1:
            return 1.0
        if ROWS_TABLE is not None and self.scal == "moe":
            return _interp(ROWS_TABLE, S)
        if self.scal == "linear":
            return 1.0 + V4.CHUNK_SLOPE * (S - 1)
        # cost(S) ∝ f + l*(S-1) + e*distinct(S)/k ; choose e share so that mult(9) == r9 with l = 0.02 f
        d9 = distinct(9, self.E, self.k) / self.k
        l = 0.02
        # (f + 8 l f + e d9) / (f + e) = r9  with f = 1 - e  -> solve for e
        # (1-e)(1+8l) + e d9 = r9 (1)  -> e (d9 - 1 - 8l) = r9 - 1 - 8l
        e = max(0.0, min(1.0, (self.r9 - 1 - 8 * l) / (d9 - 1 - 8 * l)))
        f = 1.0 - e
        return f + l * f * (S - 1) + e * distinct(S, self.E, self.k) / self.k

    def cost(self, i, S):
        return self.ovh[i] + self.p1[i] * self.mult(S)

    def lap(self, S=1):
        return sum(self.legs) + sum(self.cost(i, S) for i in range(self.D))


# ── the chain: D stages, per-leg latency, solo or batching servers ────────────────────────────────

@dataclass
class Frame:
    sess: int
    pos: int
    s: int
    epoch: int
    committed: bool = True
    need_block: bool = False        # tail-local drafter: this frame's reply must carry a block
    kind: str = "step"


class Chain:
    def __init__(self, eng, ring: Ring, on_reply):
        self.eng, self.r, self.on_reply = eng, ring, on_reply
        self.q = [deque() for _ in range(ring.D)]
        self.busy = [False] * ring.D
        self.busy_ms = [0.0] * ring.D
        self.frames = 0
        self.rows = 0
        self.link_free = [0.0] * (ring.D + 1)

    def inject(self, t, f):
        self.frames += 1
        self.eng.at(t + self.r.legs[0], lambda tt, ff=f: self._arrive(tt, 0, ff))

    def _arrive(self, t, i, f):
        self.q[i].append(f)
        self._pump(t, i)

    def _pump(self, t, i):
        if self.busy[i] or not self.q[i]:
            return
        take = [self.q[i].popleft()]
        while self.q[i] and len(take) < self.r.batch_rows:
            take.append(self.q[i].popleft())
        S = sum(f.s for f in take)
        d = self.r.cost(i, S) + (self.r.ovh[i] * 0.15 * (len(take) - 1))
        if i == self.r.D - 1 and self.r.draft_where == "tail":
            nd = sum(1 for f in take if (f.need_block or not self.r.draft_lazy) and f.committed)
            if nd:
                d += self.r.t_block * (1.0 + 0.25 * (nd - 1))   # drafts batch across sessions
        self.busy[i] = True
        self.busy_ms[i] += d
        if i == 0:
            self.rows += S
        self.eng.at(t + d, lambda tt, ii=i, fs=take: self._done(tt, ii, fs))

    def _done(self, t, i, fs):
        self.busy[i] = False
        L = i + 1
        S = sum(f.s for f in fs)
        start = max(t, self.link_free[L])
        dur = self.r.msg_ms + (S * self.r.kb_per_pos / 1000.0 / self.r.bw_MBps * 1000.0
                               if self.r.bw_MBps else 0.0)
        self.link_free[L] = start + dur
        at = start + dur + self.r.legs[L]
        for f in fs:
            if i + 1 < self.r.D:
                self.eng.at(at, lambda tt, ii=i + 1, ff=f: self._arrive(tt, ii, ff))
            else:
                self.eng.at(at, lambda tt, ff=f: self.on_reply(tt, ff))
        self._pump(t, i)


class HeadDrafter:
    """The coordinator's drafter on the head box: ONE serial server shared by every session."""

    def __init__(self, eng, t_block):
        self.eng, self.t_block = eng, t_block
        self.free_at = 0.0
        self.busy_ms = 0.0

    def draft(self, t, fn):
        start = max(t, self.free_at)
        self.free_at = start + self.t_block
        self.busy_ms += self.t_block
        self.eng.at(self.free_at, fn)


# ── sessions (coordinators) ────────────────────────────────────────────────────────────────────────

class Session:
    def __init__(self, sid, sim, a, K, n_tokens, seed):
        self.sid, self.sim, self.a, self.K, self.n = sid, sim, a, K, n_tokens
        self.rng = random.Random(seed)
        self.gen = 0
        self.t_done = None
        self.cycles = 0
        self.frames = 0
        self.stale = 0

    @property
    def done(self):
        return self.gen >= self.n


class Greedy(Session):
    def start(self, t):
        self._send(t, 0)

    def _send(self, t, pos):
        self.frames += 1
        self.sim.chain.inject(t, Frame(self.sid, pos, 1, 0))

    def reply(self, t, f):
        self.gen += 1
        self.cycles += 1
        if self.done:
            self.t_done = t
            return
        self._send(t + self.sim.ring.coord_ms, f.pos + 1)


class Chunk1(Session):
    """M2.5's shipped EAGLE round: draft K from the verified tap, send [cur]+K as ONE frame, wait."""

    def start(self, t):
        self._round(t, 0, fresh=True)

    def _round(self, t, pos, fresh):
        def go(tt):
            self.frames += 1
            self.sim.chain.inject(tt, Frame(self.sid, pos, self.K + 1, 0, need_block=True))
        if self.sim.ring.draft_where == "head":
            self.sim.head.draft(t, go)
        else:
            go(t)          # the tail drafted this block inside the previous frame's service

    def reply(self, t, f):
        k = _run_len(self.a, self.K, self.rng)
        self.gen += k + 1
        self.cycles += 1
        if self.done:
            self.t_done = t
            return
        self._round(t + self.sim.ring.coord_ms, f.pos + k + 1, fresh=True)


class ChunkD(Session):
    """The 07-02 panel's lever: up to `depth` K-chunks in flight, chunk j+1 drafted on top of chunk
    j's UNVERIFIED drafts (n-gram / stale EAGLE). Chunk j+1 is valid only if chunk j fully accepts;
    any partial accept flushes everything behind it."""

    def __init__(self, *a, depth=4, **kw):
        super().__init__(*a, **kw)
        self.depth = depth
        self.epoch = 0
        self.inflight = deque()
        self.nextpos = 0

    def start(self, t):
        self._fill(t)

    def _fill(self, t):
        while len(self.inflight) < self.depth:
            f = Frame(self.sid, self.nextpos, self.K + 1, self.epoch)
            self.inflight.append(f)
            self.nextpos += self.K            # the next chunk starts at this chunk's last draft
            self.frames += 1
            self.sim.chain.inject(t, f)

    def reply(self, t, f):
        if f.epoch != self.epoch:
            self.stale += 1
            return
        assert self.inflight.popleft() is f
        k = _run_len(self.a, self.K, self.rng)
        if k == self.K:
            self.gen += self.K                # the bonus is judged as the next chunk's first draft
            if not self.inflight:
                self.gen += 1
                self.nextpos = f.pos + self.K + 1
        else:
            self.gen += k + 1
            self.cycles += 1
            self.epoch += 1
            self.inflight.clear()
            self.nextpos = f.pos + k + 1
        if self.done:
            self.t_done = t
            return
        self._fill(t + self.sim.ring.coord_ms)


class PerToken(Session):
    """V4's coordinate_dspark_pipelined round, statement for statement (v4_pipe_sim.sim_pipelined),
    generalised over drafter location. Block = K drafts proposing pos+2..pos+K+1 off the tap of the
    committed frame at pos. F=1: a block is streamed only when the frontier reaches the horizon."""

    def __init__(self, *a, W=16, F=1, **kw):
        super().__init__(*a, **kw)
        self.W, self.F = W, F
        self.c = 0
        self.horizon = -1
        self.epoch = 0
        self.sent: Dict[int, int] = {}
        self.depths: List[int] = []

    @staticmethod
    def T(p):
        return 1_000_000 + p

    def start(self, t):
        self._feed(t, 0, self.T(0))

    def _feed(self, t, pos, tok):
        self.sent[pos] = tok
        self.horizon = max(self.horizon, pos)
        self.frames += 1
        f = Frame(self.sid, pos, 1, self.epoch, committed=(tok == self.T(pos)))
        nxt = self.sent.get(pos + 1)
        f.need_block = True  # recomputed at the tail below via the hint (dnxt) semantics
        f._sess = self
        self.sim.chain.inject(t, f)

    def need_block(self, f):
        """The lazy tail's licence (V4 dnxt hint): a block is needed iff this committed frame's reply
        will reopen speculation — nothing streamed at pos+1, or what was streamed there is wrong."""
        nxt = self.sent.get(f.pos + 1)
        return nxt is None or nxt != self.T(f.pos + 1)

    def _block(self, base):
        k = _run_len(self.a, self.K, self.rng)
        return [self.T(base + 1 + i) if i < k else -(base + 1 + i) for i in range(self.K)]

    def reply(self, t, f):
        if f.epoch != self.epoch:
            self.stale += 1
            return
        pos = f.pos
        assert pos == self.c, (pos, self.c)
        m = self.T(pos + 1)
        fed = self.sent.get(pos + 1)
        self.c = pos + 1
        self.gen += 1
        if self.done:
            self.t_done = t
            self.epoch += 1
            return
        tc = t + self.sim.ring.coord_ms
        if fed is None:
            self._feed(tc, self.c, m)
        elif fed != m:
            self.cycles += 1
            self.epoch += 1
            for p in [p for p in self.sent if p > pos]:
                del self.sent[p]
            self.horizon = pos
            self._feed(tc, self.c, m)
        blk = self._block(pos + 1) if f.committed else None
        if blk and self.horizon - self.c + 1 <= self.F:
            base = self.horizon
            ep = self.epoch

            def stream(tt, blk=blk, base=base, pos=pos, ep=ep):
                if ep != self.epoch or self.done:
                    return
                for i, d in enumerate(blk):
                    p = pos + 2 + i
                    if p - self.c >= self.W:
                        break
                    if p <= base or p in self.sent:
                        continue
                    self._feed(tt, p, d)
            if self.sim.ring.draft_where == "head":
                self.sim.head.draft(tc, stream)
            else:
                stream(tc)
        self.depths.append(self.horizon - self.c + 1)
        self.sent.pop(pos - 1, None)


# ── the simulation ─────────────────────────────────────────────────────────────────────────────────

class Sim:
    def __init__(self, ring: Ring):
        self.ring = ring
        self.eng = _Engine()
        self.chain = Chain(self.eng, ring, self._reply)
        self.head = HeadDrafter(self.eng, ring.t_block)
        self.sessions: List[Session] = []

    def _reply(self, t, f):
        self.sessions[f.sess].reply(t, f)

    def run(self, sessions, stagger_ms=0.0):
        self.sessions = sessions
        # the tail's lazy licence needs the session's view at service time
        orig = self.chain._pump

        def pump(t, i, _orig=orig):
            if i == self.ring.D - 1:
                for f in self.chain.q[i]:
                    s = self.sessions[f.sess]
                    if isinstance(s, PerToken):
                        f.need_block = f.committed and f.epoch == s.epoch and s.need_block(f)
            _orig(t, i)
        self.chain._pump = pump
        for j, s in enumerate(sessions):
            s.start(j * stagger_ms)
        self.eng.run(until_fn=lambda: all(s.done for s in sessions))
        wall = max(s.t_done or self.eng.t for s in sessions)
        per = [1000.0 * s.gen / (s.t_done or wall) for s in sessions]
        agg = 1000.0 * sum(s.gen for s in sessions) / wall
        gen = sum(s.gen for s in sessions)
        return {
            "per_stream": statistics.fmean(per), "agg": agg,
            "g": statistics.fmean([s.gen / max(1, s.cycles) for s in sessions]),
            "frames_per_tok": self.chain.frames / max(1, gen),
            "rows_per_tok": self.chain.rows / max(1, gen),
            "util_max": max(b / wall for b in self.chain.busy_ms),
            "head_draft_util": self.head.busy_ms / wall,
            "mean_inflight": (statistics.fmean([d for s in sessions if isinstance(s, PerToken)
                                                for d in s.depths]) if any(isinstance(s, PerToken)
                                                                           for s in sessions) else 0),
        }


def run(ring, mode, a, K, n=600, B=1, seed=7, **kw):
    sim = Sim(ring)
    cls = {"greedy": Greedy, "chunk1": Chunk1, "chunkD": ChunkD, "pertoken": PerToken}[mode]
    sess = [cls(j, sim, a, K, n, seed + 1000 * j, **kw) for j in range(B)]
    return sim.run(sess, stagger_ms=7.0)


def best_of(ring, mode, a, K, seeds=(7, 11, 13, 17, 19), **kw):
    """Mean over seeds (acceptance sampling noise), not max — this is a model, not a ring."""
    rs = [run(ring, mode, a, K, seed=s, **kw) for s in seeds]
    out = {k: statistics.fmean([r[k] for r in rs]) for k in rs[0]}
    return out


# ── calibrations ───────────────────────────────────────────────────────────────────────────────────
#
# M2.5, 6x5090 EU ring of perstream-trees-ab-20260712 (BG head [0:2], PL/SE/HU/RS/BG x 12 layers):
#   B=1 mix K=8 chains: 22.5 tok/s at g=3.8 committed/round  ->  round = 169 ms
#   decomposition (reasoning-B4 chain round, graph-replayed [1,9] rows): stage compute sum 43.1 ms,
#   span sum 44.6 ms, round 169 ms.  EAGLE3 serial draft 10.6-12.7 ms/round (warmring-20260718).
#   => wire+codec residual ~ 169 - 11 - 44.6 = 113 ms over 6 WAN legs (+ a local coord->head leg).
M25_LAYERS = [2, 12, 12, 12, 12, 12]
M25_MS_PER_LAYER_S9 = 43.1 / 62


LINK = {"msg_ms": 0.0, "bw_MBps": None, "kb_per_pos": 6.0}     # set by fit_links()


def m25_ring(r9=2.0, leg=None, draft_where="head", t_block=11.0, D_layers=M25_LAYERS, batch_rows=1,
             per_layer_s9=None, ovh_total=1.5, tail_extra=0.0, link=None):
    per_layer_s9 = M25_MS_PER_LAYER_S9 if per_layer_s9 is None else per_layer_s9
    probe = Ring(p1=[1.0], ovh=[0.0], legs=[0, 0], r9=r9)
    m9 = probe.mult(9)
    p1 = [L * per_layer_s9 / m9 for L in D_layers]
    p1[-1] += tail_extra
    D = len(D_layers)
    ovh = [ovh_total / D] * D
    if leg is None:
        leg = (169.0 - 11.0 - 0.3 - sum(ovh) - 43.1 - 0.5) / D     # fitted below, kept for reference
    legs = [0.5] + [leg] * (D - 1) + [leg]
    return Ring(p1=p1, ovh=ovh, legs=legs, r9=r9, draft_where=draft_where, t_block=t_block,
                batch_rows=batch_rows, **(link or LINK))


def fit_m25_leg(a, K=8, r9=2.0, target=22.5, link=None):
    lo, hi = 0.0, 60.0
    for _ in range(30):
        mid = (lo + hi) / 2
        x = best_of(m25_ring(r9=r9, leg=mid, link=link), "chunk1", a, K)["per_stream"]
        if x > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# The measured B-curve (tree off, K=8 chains, de-lockstep solo frames, head EAGLE), two receipts:
# perstream-delockstep-20260711 23.0/30.9/54.3/111.5 and perstream-trees-ab-20260712 22.5/43.5/56.8/112.4;
# per-B mix g (committed/round) from the 07-12 receipt: 3.8 / 3.66 / 3.13 / 3.52.
BCURVE = {1: 22.75, 2: 37.2, 4: 55.5, 8: 112.0}
BG = {1: 3.8, 2: 3.66, 4: 3.13, 8: 3.52}


def fit_links(r9=2.0, verbose=True):
    """(msg_ms, bw) of the WAN links against the B-curve; leg propagation re-fitted at B=1 each time."""
    best = None
    for msg in (0.25, 0.5, 1.0, 2.0, 3.0, 5.0):
        for bw in (4.0, 6.0, 8.0, 12.0, 20.0, None):
            link = {"msg_ms": msg, "bw_MBps": bw, "kb_per_pos": 6.0}
            a1 = aseq(fit_a1(BG[1], 8, 0.95), 0.95, 10)
            leg = fit_m25_leg(a1, 8, r9, link=link)
            err, pred = 0.0, {}
            for B in (2, 4, 8):
                aB = aseq(fit_a1(BG[B], 8, 0.95), 0.95, 10)
                x = best_of(m25_ring(r9=r9, leg=leg, link=link), "chunk1", aB, 8, B=B, n=200,
                            seeds=(7, 11))["agg"]
                pred[B] = x
                err += (math.log(x / BCURVE[B])) ** 2
            rms = (err / 3) ** 0.5
            if best is None or rms < best[0]:
                best = (rms, link, leg, pred)
    rms, link, leg, pred = best
    if verbose:
        print(f"  link fit: msg {link['msg_ms']} ms/msg, bw {link['bw_MBps']} MB/s, prop leg {leg:.1f} ms"
              f"  -> B=2/4/8 agg {pred[2]:.0f}/{pred[4]:.0f}/{pred[8]:.0f} vs measured "
              f"{BCURVE[2]:.0f}/{BCURVE[4]:.0f}/{BCURVE[8]:.0f}  (RMS log err {rms:.2f})")
    return link, leg, rms


# MiMo-V2.6-Flash: 48 layers, hidden 4096, 256 experts top-8, moe_intermediate 2048 (13.4 MB/expert
# at MXFP4 vs M2.5's ~8 MB NVFP4), 64 heads x 192 qk. ASSUMPTION: the workstream-1 MXFP4 runtime
# reaches M2.5's graphed efficiency per byte/FLOP => ~1.7x M2.5's per-layer time. Bracketed x0.7..x2.
MIMO_LAYER_FACTOR = 1.7


def mimo_ring(n_stages, r9=2.0, leg=18.7, draft_where="tail", t_block=2.5, kfac=None,
              batch_rows=1):
    kfac = MIMO_LAYER_FACTOR if kfac is None else kfac
    per = 48 / n_stages
    layers = [per] * n_stages
    return m25_ring(r9=r9, leg=leg, draft_where=draft_where, t_block=t_block, D_layers=layers,
                    per_layer_s9=M25_MS_PER_LAYER_S9 * kfac, batch_rows=batch_rows)


def _interp(table, x):
    """log-x linear interpolation over a {rows: mult} table; linear extrapolation past the ends."""
    ks = sorted(table)
    if x <= ks[0]:
        return table[ks[0]]
    for a, b in zip(ks, ks[1:]):
        if a <= x <= b:
            t = (math.log(x) - math.log(a)) / (math.log(b) - math.log(a))
            return table[a] + t * (table[b] - table[a])
    a, b = ks[-2], ks[-1]
    return table[b] + (table[b] - table[a]) / (b - a) * (x - b)


def apply_calib(path):
    """Load a shard-calib/1 file (research/hw1/calib_fit.py): measured stage s/B-scaling table, the
    M2.5 per-layer basis and kfac. Null keys keep the EST constants (printed, never silent)."""
    import json
    global ROWS_TABLE, M25_MS_PER_LAYER_S9, MIMO_LAYER_FACTOR
    cal = json.load(open(path))
    if cal.get("schema") != "shard-calib/1":
        raise SystemExit(f"{path}: not a shard-calib/1 file")
    p = cal["pipespec"]
    if p.get("rows_table"):
        ROWS_TABLE = {int(k): float(v) for k, v in p["rows_table"].items()}
    if p.get("m25_ms_per_layer_s9"):
        M25_MS_PER_LAYER_S9 = float(p["m25_ms_per_layer_s9"])
    if p.get("kfac"):
        MIMO_LAYER_FACTOR = float(p["kfac"])
    print(f"CALIBRATED from {path}{' (STUB DATA)' if cal.get('stub') else ''}: rows_table "
          f"{'measured' if ROWS_TABLE else 'EST model'}, per-layer s9 {M25_MS_PER_LAYER_S9:.3f} ms, "
          f"kfac {MIMO_LAYER_FACTOR:.2f}")
    return cal


# ── reports ────────────────────────────────────────────────────────────────────────────────────────

def selfcheck():
    """This file's PerToken round == v4_pipe_sim.sim_pipelined on a V4-shaped ring (non-lazy tail
    drafter, linear s-scaling, uniform legs). Same seed -> same block draws -> same numbers."""
    cal_ring = V4.measured_ring(0.2839, 0.0, 4.0)
    a = V4.accept_seq(0.796, 0.945, 7)
    ref = V4.sim_pipelined(cal_ring, a, 6, W=16, n_tokens=300, seed=7)
    ring = Ring(p1=[s.p_ms for s in cal_ring.stages],
                ovh=[s.ovh_ms + s.snd_ms for s in cal_ring.stages],
                legs=[4.0] * (cal_ring.D + 1), scal="linear", draft_where="tail",
                t_block=V4.TAIL_DRAFT_MS, draft_lazy=False, coord_ms=0.0)
    ring.ovh[-1] += cal_ring.logit_ms
    sim = Sim(ring)
    s = PerToken(0, sim, a, 6, 300, 7, W=16)
    sim.run([s])
    mine = 1000.0 * s.gen / s.t_done
    print(f"selfcheck vs v4_pipe_sim.sim_pipelined (V4 ring, W=16, block 6, 300 tok):")
    print(f"   v4_pipe_sim  tok/s {ref.toks_per_s:.3f}  g {ref.g:.3f}  frames {ref.frames}  "
          f"max_inflight {ref.max_inflight}")
    print(f"   this file    tok/s {mine:.3f}  g {s.gen / max(1, s.cycles):.3f}  frames {s.frames}  "
          f"max_inflight {max(s.depths)}")
    ok = abs(mine - ref.toks_per_s) / ref.toks_per_s < 0.02 and s.frames == ref.frames
    print(f"   {'MATCH' if ok else 'MISMATCH'}")
    return ok


def m25_report():
    print("\n" + "=" * 100)
    print("M2.5 on the 07-12 6x5090 EU ring (calibrated: chunk depth-1 EAGLE K=8, g=3.8 -> 22.5 tok/s)")
    print("=" * 100)
    K = 8
    rows = []
    for rho, lbl in ((1.0, "flat"), (0.95, "decay .95")):
        a1 = fit_a1(3.8, K, rho)
        a = aseq(a1, rho, K + 2)
        for r9 in (1.3, 2.0):
            leg = fit_m25_leg(a, K, r9)
            ring = m25_ring(r9=r9, leg=leg)
            base = best_of(ring, "chunk1", a, K)
            gr = best_of(ring, "greedy", a, K)
            pt = best_of(ring, "pertoken", a, K, W=K + 1)
            # tail-local EAGLE (aux already reaches the tail; draft inside the tail's service)
            rt = m25_ring(r9=r9, leg=leg, draft_where="tail", t_block=11.0)
            pt_tail = best_of(rt, "pertoken", a, K, W=K + 1)
            ch_tail = best_of(rt, "chunk1", a, K)
            rows.append((lbl, r9, a1, leg, gr, base, pt, ch_tail, pt_tail, ring))
    print(f"  {'accept':>10}{'r9':>5}{'a1':>7}{'leg ms':>8}|{'greedy':>8}{'chunk1':>8}"
          f"{'pertok':>8}{'Δ':>6}|{'chunk1 tailEAGLE':>17}{'pertok tailEAGLE':>17}{'Δ':>6}")
    for lbl, r9, a1, leg, gr, base, pt, cht, ptt, ring in rows:
        print(f"  {lbl:>10}{r9:>5.1f}{a1:>7.3f}{leg:>8.1f}|{gr['per_stream']:>8.1f}"
              f"{base['per_stream']:>8.1f}{pt['per_stream']:>8.1f}"
              f"{100 * (pt['per_stream'] / base['per_stream'] - 1):>5.0f}%|"
              f"{cht['per_stream']:>17.1f}{ptt['per_stream']:>17.1f}"
              f"{100 * (ptt['per_stream'] / base['per_stream'] - 1):>5.0f}%")
    print("\n  PER-MESSAGE LINK COST sensitivity (decay .95, r9=2.0; legs re-fitted so chunk1 stays 22.5):")
    a = aseq(fit_a1(3.8, K, 0.95), 0.95, K + 2)
    for msg in (0.5, 2.0, 4.0):
        link = {"msg_ms": msg, "bw_MBps": None, "kb_per_pos": 6.0}
        lg = fit_m25_leg(a, K, 2.0, link=link)
        pt = best_of(m25_ring(r9=2.0, leg=lg, link=link), "pertoken", a, K, W=K + 1)
        print(f"    {msg:>4.1f} ms/message/link: per-token {pt['per_stream']:5.1f} "
              f"({100 * (pt['per_stream'] / 22.5 - 1):+.0f}% vs chunk1)")
    lbl, r9, a1, leg, *_ , ring = rows[-1]
    print(f"\n  ring (decay .95, r9=2.0): legs {[round(x, 1) for x in ring.legs]}  "
          f"tau(s=1) {[round(ring.cost(i, 1), 1) for i in range(ring.D)]}  "
          f"tau(s=9) {[round(ring.cost(i, 9), 1) for i in range(ring.D)]}")
    print(f"  lap(s=1) {ring.lap(1):.0f} ms, lap(s=9) {ring.lap(9):.0f} ms, BDP lap/tau_max "
          f"{ring.lap(1) / max(ring.cost(i, 1) for i in range(ring.D)):.0f} frames")
    return rows


def m25_dflash_report(rows):
    print("\n  DRAFTER SWAP — z-lab/MiniMax-M2.5-DFlash (block 8 = 7 drafts, taps [1,13,24,36,47,59],")
    print("  card tau 3.9-4.5 greedy) vs EAGLE3; DFlash = one parallel forward, t_block ~3 ms.")
    print(f"  {'tau':>5}{'r9':>5}|{'chunk1 head':>12}{'chunk1 tail':>12}{'pertok tail':>12}"
          f"{'pertok head':>12}|  vs EAGLE chunk1 baseline")
    for tau in (3.8, 4.2, 4.5):
        for r9 in (1.3, 2.0):
            base_row = [r for r in rows if r[0] == "decay .95" and r[1] == r9][0]
            leg, base = base_row[3], base_row[5]["per_stream"]
            K = 7
            a1 = fit_a1(tau, K, 0.93)
            a = aseq(a1, 0.93, K + 2)
            ch = best_of(m25_ring(r9=r9, leg=leg, draft_where="head", t_block=3.0), "chunk1", a, K)
            ct = best_of(m25_ring(r9=r9, leg=leg, draft_where="tail", t_block=3.0), "chunk1", a, K)
            pt = best_of(m25_ring(r9=r9, leg=leg, draft_where="tail", t_block=3.0), "pertoken", a, K,
                         W=K + 1)
            ph = best_of(m25_ring(r9=r9, leg=leg, draft_where="head", t_block=3.0), "pertoken", a, K,
                         W=K + 1)
            print(f"  {tau:>5.1f}{r9:>5.1f}|{ch['per_stream']:>12.1f}{ct['per_stream']:>12.1f}"
                  f"{pt['per_stream']:>12.1f}{ph['per_stream']:>12.1f}|  "
                  f"{100 * (max(ch['per_stream'], ct['per_stream'], pt['per_stream']) / base - 1):+.0f}%"
                  f" best, pipelining alone {100 * (pt['per_stream'] / ct['per_stream'] - 1):+.0f}%")


def kill_0702_report():
    """Reproduce the 07-02 panel (chunk-depth pipelining is accept-gated) and set per-token beside it."""
    print("\n" + "=" * 100)
    print("THE 07-02 KILL, re-run: stacked K=8 chunks (depth 4) vs depth-1 chunk vs per-token, flat alpha")
    print("(its ring: T_traversal ~ 280-400 ms WAN-bound; here the 07-12 ring, r9=2.0, head drafter)")
    print("=" * 100)
    K = 8
    a_ref = aseq(fit_a1(3.8, K, 1.0), 1.0, K + 2)
    leg = fit_m25_leg(a_ref, K, 2.0)
    ring = m25_ring(r9=2.0, leg=leg)
    # n-gram drafting (chunkD) is target-free -> ~free draft; EAGLE rows keep 11 ms
    ring_ng = m25_ring(r9=2.0, leg=leg, t_block=0.3)
    print(f"  {'alpha':>6}{'g(K=8)':>8}|{'chunk1':>8}{'chunkD=4':>10}{'D4/D1':>7}|{'pertok':>8}{'pt/D1':>7}")
    for al in (0.60, 0.70, 0.74, 0.80, 0.85, 0.90, 0.95, 0.97):
        a = [al] * (K + 2)
        c1 = best_of(ring_ng, "chunk1", a, K)
        cd = best_of(ring_ng, "chunkD", a, K, depth=4)
        pt = best_of(ring_ng, "pertoken", a, K, W=K + 1)
        print(f"  {al:>6.2f}{g_of(a, K):>8.2f}|{c1['per_stream']:>8.1f}{cd['per_stream']:>10.1f}"
              f"{cd['per_stream'] / c1['per_stream']:>7.2f}|{pt['per_stream']:>8.1f}"
              f"{pt['per_stream'] / c1['per_stream']:>7.2f}")


def v4_contrast_report():
    """Why V4 won: its s-scaling is linear (+replay), so the same ring logic flips sign."""
    print("\n" + "=" * 100)
    print("CONTRAST — the same M2.5 ring geometry with V4's LINEAR s-scaling (3.93x at s=6)")
    print("=" * 100)
    K = 8
    a = aseq(fit_a1(3.8, K, 0.95), 0.95, K + 2)
    for scal, r9 in (("moe", 1.3), ("moe", 2.0), ("moe", 4.0), ("linear", None)):
        ring = m25_ring(r9=r9 or 2.0, leg=18.7, draft_where="tail", t_block=3.0)
        ring.scal = scal
        c1 = best_of(ring, "chunk1", a, K)
        pt = best_of(ring, "pertoken", a, K, W=K + 1)
        m9 = ring.mult(9)
        print(f"  scaling {scal:>6} cost(9)/cost(1)={m9:>5.2f}:  chunk1 {c1['per_stream']:>6.1f}  "
              f"pertoken {pt['per_stream']:>6.1f}  ({100 * (pt['per_stream'] / c1['per_stream'] - 1):+.0f}%)")
    # compute-bound: tau large vs wire (V4's eager 07-31 ring was 604 ms compute vs 28 ms wire)
    for leg in (4.0, 18.7):
        ring = m25_ring(r9=2.0, leg=leg, draft_where="tail", t_block=3.0,
                        per_layer_s9=M25_MS_PER_LAYER_S9 * 10)
        c1 = best_of(ring, "chunk1", a, K)
        pt = best_of(ring, "pertoken", a, K, W=K + 1)
        print(f"  compute-bound (10x per-layer), leg {leg:>4.1f} ms, MoE r9=2.0: chunk1 "
              f"{c1['per_stream']:>5.1f}  pertoken {pt['per_stream']:>5.1f}  "
              f"({100 * (pt['per_stream'] / c1['per_stream'] - 1):+.0f}%)")


def mimo_report():
    print("\n" + "=" * 100)
    print("MiMo-V2.6-Flash + shipped dflash/ (block 8 -> 7 drafts, taps [0,11,23,35,47]) — PREDICTION")
    print("=" * 100)
    print(f"  per-layer = {MIMO_LAYER_FACTOR:.2f}x M2.5 (bracket 0.7x..2x); draft 2.5 ms tail-local; leg 18.7 ms")
    print(f"  {'hops':>5}{'tau':>5}{'kfac':>6}{'leg':>6}|{'greedy':>7}{'chunk1':>8}{'pertok':>8}"
          f"{'Δ':>6}| lap(1) tau_max(1)")
    out = []
    for n in (6, 3):
        for tau in (3.2, 4.3, 5.5):
            for kfac, leg in ((MIMO_LAYER_FACTOR, 18.7), (MIMO_LAYER_FACTOR * 0.7, 12.0),
                              (MIMO_LAYER_FACTOR * 2.0, 25.0)):
                K = 7
                a = aseq(fit_a1(tau, K, 0.93), 0.93, K + 2)
                ring = mimo_ring(n, kfac=kfac, leg=leg)
                gr = best_of(ring, "greedy", a, K)
                c1 = best_of(ring, "chunk1", a, K)
                pt = best_of(ring, "pertoken", a, K, W=K + 1)
                out.append((n, tau, kfac, leg, c1, pt))
                print(f"  {n:>5}{tau:>5.1f}{kfac:>6.2f}{leg:>6.1f}|{gr['per_stream']:>7.1f}"
                      f"{c1['per_stream']:>8.1f}{pt['per_stream']:>8.1f}"
                      f"{100 * (pt['per_stream'] / c1['per_stream'] - 1):>5.0f}%|"
                      f" {ring.lap(1):>5.0f} {max(ring.cost(i, 1) for i in range(ring.D)):>6.1f}")
    return out


def multisession_report():
    """Q3. Head-side serial work per round is FITTED to the measured B-curve (it is the only term that
    bends it: stage util stays <0.4 at B=8 on the 07-12 decomposition). Then the drafter moves."""
    print("\n" + "=" * 100)
    print("MULTI-SESSION on the M2.5 ring (r9=2.0). Calibration: head serial work per EAGLE round fitted")
    print("to the measured B-curve 22.75/37/55.5/112 with per-B mix g 3.8/3.66/3.13/3.52.")
    print("=" * 100)
    HEAD_SERIAL = 26.0     # fitted: 11 ms measured draft + ~15 ms extend/coordinator per round
    a1 = aseq(fit_a1(BG[1], 8, 0.95), 0.95, 10)
    leg = fit_m25_leg(a1, 8, 2.0, link={"msg_ms": 0.0, "bw_MBps": None, "kb_per_pos": 6.0})
    nolink = {"msg_ms": 0.0, "bw_MBps": None, "kb_per_pos": 6.0}
    # re-fit the propagation with the fitted head serial work so B=1 stays at 22.5
    lo, hi = 0.0, 60.0
    for _ in range(25):
        mid = (lo + hi) / 2
        x = best_of(m25_ring(r9=2.0, leg=mid, t_block=HEAD_SERIAL, link=nolink), "chunk1", a1, 8)
        lo, hi = (mid, hi) if x["per_stream"] > 22.5 else (lo, mid)
    leg = (lo + hi) / 2
    print(f"  calibrated: head serial {HEAD_SERIAL} ms/round, prop leg {leg:.1f} ms")
    print("  check vs measured B-curve (today's config):", end="")
    for B in (2, 4, 8):
        aB = aseq(fit_a1(BG[B], 8, 0.95), 0.95, 10)
        x = best_of(m25_ring(r9=2.0, leg=leg, t_block=HEAD_SERIAL, link=nolink), "chunk1", aB, 8, B=B,
                    n=200, seeds=(7, 11))
        print(f"  B{B} {x['agg']:.0f} (meas {BCURVE[B]:.0f}, stage util {x['util_max']:.2f}, head "
              f"{x['head_draft_util']:.2f})", end="")
    print()
    tau_df = 4.3
    aD = aseq(fit_a1(tau_df, 7, 0.93), 0.93, 9)
    aE = aseq(fit_a1(3.8, 8, 0.95), 0.95, 10)
    print(f"\n  fixed content (EAGLE g 3.8 / DFlash tau {tau_df}), per-stream | aggregate tok/s:")
    arms = [
        ("chunk1", "head", 1, 8, aE, HEAD_SERIAL, "EAGLE head, chunk K8, solo frames (today)"),
        ("chunk1", "tail", 1, 7, aD, 4.0, "DFlash tail, chunk b8, solo frames"),
        ("chunk1", "tail", 64, 7, aD, 4.0, "DFlash tail, chunk b8, stage-BATCHED"),
        ("pertoken", "tail", 1, 7, aD, 4.0, "DFlash tail, per-token b8, solo frames"),
        ("pertoken", "tail", 64, 7, aD, 4.0, "DFlash tail, per-token b8, stage-BATCHED"),
    ]
    print(f"  {'arm':<44}" + "".join(f"{'B=' + str(B):>14}" for B in (1, 2, 4, 8, 16, 32)))
    for mode, where, rows, K, a, tb, lbl in arms:
        cells = []
        for B in (1, 2, 4, 8, 16, 32):
            ring = m25_ring(r9=2.0, leg=leg, draft_where=where, t_block=tb, batch_rows=rows, link=nolink)
            kw = {"W": K + 1} if mode == "pertoken" else {}
            x = best_of(ring, mode, a, K, B=B, n=300, seeds=(7, 11, 13), **kw)
            cells.append(f"{x['per_stream']:>5.1f}|{x['agg']:>5.0f}({x['util_max']:.1f})")
        print(f"  {lbl:<44}" + "".join(f"{c:>14}" for c in cells))
    print("  cell = per-stream|aggregate(max stage util)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--part", default="all")
    ap.add_argument("--calib", default=None, help="shard-calib/1 JSON from research/hw1/calib_fit.py")
    args = ap.parse_args()
    ok = selfcheck()
    if args.calib:
        apply_calib(args.calib)
    if args.selfcheck:
        return 0 if ok else 1
    parts = args.part.split(",")
    if "all" in parts or "m25" in parts:
        rows = m25_report()
        m25_dflash_report(rows)
    if "all" in parts or "kill" in parts:
        kill_0702_report()
    if "all" in parts or "v4" in parts:
        v4_contrast_report()
    if "all" in parts or "mimo" in parts:
        mimo_report()
    if "all" in parts or "multi" in parts:
        multisession_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
