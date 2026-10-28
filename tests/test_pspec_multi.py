"""S3 gate (partial): many pspec sessions through ONE entry loop (coord.run_many: one Link, one
FrameSender, replies demultiplexed on sid) over the real node server and F4's fake mesh, and the
node batcher's priority classes (BUILD-PLAN §1 S3).

  * 16 toy sessions (mixed policies, tail and entry drafters, EOS) run together equal their solo
    runs result for result (tokens, bursts, accept counts) and the direct greedy loop;
  * interleaving fuzz: random leg delays, random cancels mid-flight, EOS, mixed W/K: every
    uncancelled session is lossless, every cancelled one is a prefix of its stream;
  * fence isolation: one session's sep never fences, moves or corrupts another's;
  * priority classes: truth/spec frontier-first, then prefill, whatever the arrival order.
"""
import random

import numpy as np
import pytest

import fake_mesh as FM
from fake_mesh import FakeMesh, direct_greedy, hexsid, reply_tokens
from shard.pspec import coord
from test_pspec_node import MeshLink, _miss, _req, _settle, _state, _submit, _wrong, entry_drafter

BIG = [FM.SIM_DEVICES[k] for k in ("rtxpro6000-96g", "dgx-spark-128g", "mac-m3ultra-128g",
                                   "rtxpro6000-96g")]


def sid_drafter(box, K, miss):
    """Tail drafter plugin keyed by the lease's sid: box[sid_hex] is that session's stream."""
    def factory(lease):
        T = box[lease["sid"]]

        def blocks(q, _m):
            return [_wrong(T[p], d) if miss(p, d) else T[p]
                    for d, p in enumerate(range(q + 2, q + 2 + K), 1) if p < len(T)]
        return blocks
    return factory


@pytest.fixture
def mesh_factory():
    made = []

    def make(D, **kw):
        kw.setdefault("stuck_after", 30.0)
        m = FakeMesh(D, **kw)
        made.append(m)
        return m

    yield make
    for m in made:
        m.close()


def _jobs(m, box, n, rng, K):
    """n job specs: (sid, prompt, max_new, policy, tail?, W, eos, T)."""
    jobs = []
    for i in range(n):
        sid = 1000 + i
        prompt = [rng.randrange(FM.VOCAB) for _ in range(rng.randrange(2, 12))]
        max_new = rng.randrange(6, 26)
        T = direct_greedy(m.weights, prompt, max_new + K + 2)
        P = len(prompt)
        eos = (T[P + rng.randrange(max_new)],) if rng.random() < 0.25 else ()
        box[hexsid(sid)] = T
        jobs.append((sid, prompt, max_new, rng.choice(coord.POLICIES), rng.random() < 0.5,
                     rng.choice((4, 6, 8)), eos, T))
    return jobs


def _open(m, job, sid=None, K=4):
    s_id, prompt, max_new, policy, tail, W, eos, T = job
    sid = s_id if sid is None else sid
    m.open(sid, max_pos=len(prompt) + max_new - 1, W=W, drafter="toy" if tail else None)
    return coord.Session(prompt, max_new, sid=hexsid(sid), rte=1, W=W, policy=policy, eos_ids=eos,
                         drafter=None if tail else entry_drafter(T, K, _miss(0.6, s_id)),
                         seal=bytes)


def _want(job):
    _sid, prompt, max_new, _pol, _tail, _W, eos, T = job
    P = len(prompt)
    out = []
    for t in T[P:P + max_new]:
        out.append(t)
        if t in eos:
            break
    return out


def _mesh(mesh_factory, D, box, K, seed, legs_hi=3.0, **kw):
    rng = random.Random(seed)
    return mesh_factory(D, clock="real", cost_scale=0.0, devices=BIG[:D], s_max=16,
                        legs_ms=[rng.uniform(0.0, legs_hi) for _ in range(D + 1)],
                        drafters={"toy": sid_drafter(box, K, _miss(0.6, seed))}, **kw)


_VOLATILE = ("stale_replies", "unsent_frames", "max_inflight", "mean_inflight")


def test_sixteen_sessions_equal_their_solo_runs(mesh_factory):
    K, box = 4, {}
    m = _mesh(mesh_factory, 3, box, K, seed=7)
    rng = random.Random(7)
    jobs = _jobs(m, box, 16, rng, K)
    solo = {}
    for job in jobs:                                 # solo: the same session alone on the mesh
        box[hexsid(job[0] + 500)] = job[7]
        s = _open(m, job, sid=job[0] + 500, K=K)
        solo[job[0]] = (coord.run(MeshLink(m), s, timeout=20.0), s.trace)
        m.close_session(job[0] + 500)
    sessions = [_open(m, job, K=K) for job in jobs]
    link = MeshLink(m)
    res = coord.run_many(link, sessions, timeout=20.0)
    for job, s in zip(jobs, sessions):
        r, (r1, trace1) = res[hexsid(job[0])], solo[job[0]]
        assert r["tokens"] == _want(job), job[0]
        assert s.trace == trace1, job[0]
        assert {k: v for k, v in r.items() if k not in _VOLATILE} == \
            {k: v for k, v in r1.items() if k not in _VOLATILE}, job[0]
    assert sum(r["cancels"] for r in res.values()) > 0 and sum(r["accepted"] for r in res.values()) > 0
    assert max(n.snapshot().get("max_batch", 0) for n in m.nodes) > 1     # really batched together
    assert {r["sid"] for r in link.replies} == {hexsid(j[0]) for j in jobs}


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_interleaving_fuzz_cancels_eos_delays(mesh_factory, seed):
    K, box = 3, {}
    rng = random.Random(100 + seed)
    D = rng.choice((2, 3, 4))
    m = _mesh(mesh_factory, D, box, K, seed=seed, legs_hi=5.0)
    jobs = _jobs(m, box, 10, rng, K)
    sessions = [_open(m, job, K=K) for job in jobs]
    cancel_at = {hexsid(j[0]): rng.randrange(1, 8) for j in jobs if rng.random() < 0.3}
    seen = {}

    def on_reply(s, rep):
        seen[s.sid] = seen.get(s.sid, 0) + 1
        if cancel_at.get(s.sid) == seen[s.sid]:
            s.cancel()
    res = coord.run_many(MeshLink(m), sessions, timeout=20.0, on_reply=on_reply)
    cancelled = 0
    for job in jobs:
        r, want = res[hexsid(job[0])]["tokens"], _want(job)
        if hexsid(job[0]) in cancel_at and len(r) < len(want):
            cancelled += 1
            assert r == want[:len(r)], job[0]
        else:
            assert r == want, job[0]
    assert cancelled > 0, (cancelled, cancel_at)
    snap = [n.snapshot() for n in m.nodes]
    assert all(s.get("faults", 0) == 0 and not any(k.startswith("refused") for k in s) for s in snap)
    # the next job on the same mesh is unaffected by whatever the cancelled ones left queued
    job = _jobs(m, box, 1, random.Random(seed), K)[0]
    job = (job[0] + 900,) + job[1:]
    box[hexsid(job[0])] = job[7]
    assert coord.run(MeshLink(m), _open(m, job, K=K), timeout=20.0)["tokens"] == _want(job)


def test_fence_isolation_across_sessions(mesh_factory):
    """Session A's sep races ahead; B's frames at sep 0 still compute, and A's stale frame, queued
    at the same parked hop as B's, is fenced at dequeue without touching B."""
    m = mesh_factory(3, devices=BIG[:3])
    pa, pb = [3, 3, 7, 1], [9, 4, 4]
    Ta, Tb = direct_greedy(m.weights, pa, 10), direct_greedy(m.weights, pb, 10)
    A, B = 1, 2
    for sid, p in ((A, pa), (B, pb)):
        m.open(sid)
        _req(m, sid, kind="P", pos=0, ids=p, lg="last")
    Pa, Pb = len(pa), len(pb)
    assert _req(m, A, kind="T", pos=Pa, ids=[Ta[Pa]], sep=5)["acc"] is True
    _settle(m)
    b_before = _state(m, B)
    # A's stale frame (wrong id, rewinding) and B's sep-0 frame queue at the same parked head
    m.hang(0)
    sa = _submit(m, A, kind="S", pos=Pa, ids=[_wrong(Ta[Pa], 0)], sep=0)
    sb = _submit(m, B, kind="T", pos=Pb, ids=[Tb[Pb]], sep=0)
    m.clock.settle()
    m.resume(0)
    ra, rb = m.collect(A, sa), m.collect(B, sb)
    assert ra["fen"] and not rb["fen"]
    assert reply_tokens(rb) == [Tb[Pb + 1]] and rb["acc"] is True
    _settle(m)
    b_after = _state(m, B)
    assert [r[0] for r in b_after[0]] == [r[0] + 1 for r in b_before[0]]   # B moved by its frame only
    assert all(n.snapshot().get("fenced") == 1 for n in m.nodes)
    # both continue on their own streams
    assert reply_tokens(_req(m, A, kind="T", pos=Pa + 1, ids=[Ta[Pa + 1]], sep=5)) == [Ta[Pa + 2]]
    assert reply_tokens(_req(m, B, kind="T", pos=Pb + 1, ids=[Tb[Pb + 1]], sep=0)) == [Tb[Pb + 2]]


def test_priority_classes_frontier_first_then_prefill(mesh_factory):
    """The head's budget (6) cannot hold all three frames: the frontier frames run first, in
    frontier order, and the prefill that arrived FIRST waits a step (round-robin would run it
    first and alone)."""
    m = mesh_factory(2, devices=BIG[:2])
    m.nodes[0]._budget = 6
    steps = []
    rt = m.nodes[0].rt
    inner = rt.step

    def spy(x):
        steps.append([(s.sid, s.kind, s.start_pos) for s in x.segs])
        return inner(x)
    rt.step = spy
    p2, p3 = [5, 6, 7, 8], [1, 2, 3]
    T2, T3 = direct_greedy(m.weights, p2, 6), direct_greedy(m.weights, p3, 6)
    for sid, p in ((2, p2), (3, p3)):
        m.open(sid)
        _req(m, sid, kind="P", pos=0, ids=p, lg="last")
    m.open(1)
    steps.clear()
    m.hang(0)
    s1 = m.submit(1, kind="P", pos=0, ids=[4, 4, 4, 4, 4, 4])               # arrives first
    s2 = m.submit(2, kind="S", pos=len(p2), ids=[T2[4]], cpos=len(p2) - 2)  # pos - cpos = 2
    s3 = m.submit(3, kind="T", pos=len(p3), ids=[T3[3]], cpos=len(p3))      # pos - cpos = 0
    m.clock.settle()                                 # all three queued at the parked head
    m.resume(0)
    for sid, seq in ((1, s1), (2, s2), (3, s3)):
        m.collect(sid, seq)
    assert [(sid, k) for sid, k, _p in steps[0]] == [(3, "decode"), (2, "decode")]
    assert [(sid, k) for sid, k, _p in steps[1]] == [(1, "prefill")]


@pytest.mark.parametrize("n_dec,rows", [(4, 1), (6, 3)])
def test_a_prefill_is_not_starved_by_decode_load(mesh_factory, n_dec, rows):
    """Strict priority would park a prefill behind a steady decode load for as long as it lasts
    (48 steps here); the STARVE guard runs it within a few steps, even when more decodes than
    fit one step are starved alongside it (starved sessions go longest-waiting first)."""
    m = mesh_factory(1, devices=BIG[:1])
    node = m.nodes[0]
    node._budget = 4
    steps = []
    inner = node.rt.step

    def spy(x):
        steps.append([s.kind for s in x.segs])
        return inner(x)
    node.rt.step = spy
    sids = list(range(1, n_dec + 1))
    prompts = {sid: [sid, sid + 1, sid + 2] for sid in sids}
    T = {sid: direct_greedy(m.weights, p, 3 + rows * 12 + 2) for sid, p in prompts.items()}
    for sid, p in prompts.items():
        m.open(sid, W=8)
        _req(m, sid, kind="P", pos=0, ids=p, lg="last")
    m.open(99)
    m.hang(0)
    seqs = []
    for k in range(12 if rows == 1 else 8):          # a decode load that fills every step
        for sid in sids:
            pos = 3 + rows * k
            seqs.append((sid, m.submit(sid, kind="T" if rows == 1 else "S", pos=pos,
                                       ids=T[sid][pos:pos + rows], cpos=pos, lg="all")))
    seqs.append((99, m.submit(99, kind="P", pos=0, ids=[1, 1])))
    steps.clear()
    m.clock.settle()
    m.resume(0)
    for sid, seq in seqs:
        m.collect(sid, seq)
    at = next(i for i, st in enumerate(steps) if "prefill" in st)
    assert at <= 8, (at, len(steps))                 # STARVE = 4; strict priority: 12 / 48


class _LoggingLink(MeshLink):
    """MeshLink that records every frame it put on the wire, in order."""

    def __init__(self, mesh):
        super().__init__(mesh)
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)
        super().send(msg)


def _replay_session(m, frames, max_pos, W):
    """The recorded frames of ONE session replayed without the mesh: one bare runtime per hop
    span, the node's fence rule, each frame committed at its cpos and run in the hop's
    budget-sized steps (the node's `_run_big` split). -> {seq: tail tokens per the frame's lg}
    (None for a fenced frame). Batch invariance makes a per-session replay exact."""
    rts = []
    for i in range(m.D):
        rt = m.runtime(i)._rt
        rt.open(1, max_pos=max_pos, spec_window=W)
        rts.append(rt)
    sep_seen, out = 0, {}
    for f in frames:
        if f["fen"] or f["sep"] < sep_seen:
            out[f["seq"]] = None
            continue
        sep_seen = f["sep"]
        n, pos = f["n"], f["pos"]
        x = FM.Rows("i32", np.frombuffer(f["x"].data, dtype="<i4").astype(np.int32))
        kind = "prefill" if f["kind"] == "P" else ("decode" if n == 1 else "verify")
        for i, rt in enumerate(rts):
            if f["cpos"] >= 0:
                rt.commit(1, f["cpos"])
            b, outs, tail = m.nodes[i]._budget, [], i == m.D - 1
            for lo in range(0, n, b):
                hi = min(n, lo + b)
                tok = "none"
                if tail:
                    tok = "all" if f["kind"] != "P" else ("last" if hi == n else "none")
                outs.append(rt.step(FM.StepIn((FM.Segment(1, pos + lo, hi - lo, x.slice(lo, hi),
                                                         kind=kind, tokens=tok),))).outs[0])
            x = FM.Rows.concat([o.hidden for o in outs])
        toks = [int(t) for o in outs if o.tokens is not None for t in o.tokens.data]
        lg = f["lg"]
        out[f["seq"]] = toks if lg == "all" else toks[-1:] if lg == "last" else []
    return out


def test_adversarial_shape_mode_equals_the_replayed_schedule(mesh_factory):
    """Shape-dependent numerics (like float kernels): a verify chunk's bytes differ from the same
    rows decoded one by one, so the mesh is NOT the direct greedy loop. What must hold: every
    reply the entry got equals a replay of the recorded frame schedule, session by session, even
    with the sessions batched together on every hop."""
    K, box = 4, {}
    m = _mesh(mesh_factory, 3, box, K, seed=11, mode="adversarial")
    m.nodes[1]._budget = 3                  # the mid hop re-chunks every larger frame (_run_big)
    jobs = _jobs(m, box, 8, random.Random(11), K)
    sessions = [_open(m, job, K=K) for job in jobs]
    link = _LoggingLink(m)
    res = coord.run_many(link, sessions, timeout=20.0)
    got = {(r["sid"], r["seq"]): (None if r["fen"] else [int(t) for t in r["toks"].numpy()])
           for r in link.replies}
    differs = 0
    for job in jobs:
        h = hexsid(job[0])
        frames = [f for f in link.sent if f["sid"] == h]
        want = _replay_session(m, frames, len(job[1]) + job[2] - 1, job[5])
        assert len(want) == len(frames) and all((h, seq) in got for seq in want)
        for seq, toks in want.items():
            assert got[(h, seq)] == toks, (job[0], seq)
        differs += res[h]["tokens"] != _want(job)
    assert max(n.snapshot().get("max_batch", 0) for n in m.nodes) > 1
    assert m.nodes[1].snapshot()["steps"] > m.nodes[0].snapshot()["steps"]    # it did re-chunk
    assert differs > 0              # the mode is really shape-dependent (else this is vacuous)


def test_a_refused_session_fails_alone(mesh_factory):
    """One session leased a row short of what it uses: a hop refuses it mid-run (window). That
    session fails with the hop's reason; every other session on the same sender and the same
    hops finishes lossless."""
    K, box = 4, {}
    m = _mesh(mesh_factory, 3, box, K, seed=21)
    jobs = _jobs(m, box, 6, random.Random(21), K)
    jobs[2] = bad = jobs[2][:6] + ((),) + jobs[2][7:]           # no EOS: it must reach the end
    sessions = []
    for job in jobs:
        s = _open(m, job, K=K)
        if job is bad:                                # re-lease it two rows short on every hop
            m.close_session(job[0])
            _settle(m)
            m.open(job[0], max_pos=len(job[1]) + job[2] - 3, W=job[5],
                   drafter="toy" if job[4] else None)
        sessions.append(s)
    res = coord.run_many(MeshLink(m), sessions, timeout=20.0)
    rb = res[hexsid(bad[0])]
    assert rb["ok"] is False and "window" in rb["error"], rb
    assert rb["tokens"] == _want(bad)[:len(rb["tokens"])]
    for job in jobs:
        if job is not bad:
            r = res[hexsid(job[0])]
            assert r["ok"] and r["tokens"] == _want(job), job[0]


def test_one_sessions_bad_reply_fails_that_session_only(mesh_factory):
    """A reply the coordinator cannot accept (here: one session's `acc` flipped on the way) fails
    that session; the others on the same sender finish lossless."""
    K, box = 4, {}
    m = _mesh(mesh_factory, 3, box, K, seed=31)
    jobs = _jobs(m, box, 5, random.Random(31), K)
    victim, n = hexsid(jobs[1][0]), [0]

    class Tamper(MeshLink):
        def recv(self, timeout):
            rep = super().recv(timeout)
            if rep["sid"] == victim and rep["acc"] is not None and not rep["fen"]:
                n[0] += 1
                if n[0] == 3:
                    rep = dict(rep, acc=not rep["acc"])
            return rep
    sessions = [_open(m, job, K=K) for job in jobs]
    res = coord.run_many(Tamper(m), sessions, timeout=20.0)
    assert res[victim]["ok"] is False and "RuntimeError" in res[victim]["error"]
    for job in jobs:
        if hexsid(job[0]) != victim:
            assert res[hexsid(job[0])]["ok"] and res[hexsid(job[0])]["tokens"] == _want(job)
