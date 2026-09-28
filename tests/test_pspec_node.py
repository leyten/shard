"""S2 gate: node-side per-session fence/seek/commit, seq echo, predecessor binding, and the tail's
half of the accept rule, on the real node server (shard/mesh/node.py) over F4's fake mesh
(BUILD-PLAN §1 S2). tests/scripted_pipe.py `ScriptedPipeline._step`/`_reply` is the spec.

F4 (tests/test_fake_mesh.py) already pins gap / rollback / window refusals, the header-only fence,
the rte and predecessor drops and the malformed-stream reset. This file adds what S2 needs on top:
  * the fence changes NOTHING (runtime length/frontier, the tail's accept frontier), and with the
    fence stubbed the same schedule corrupts the session (anti-vacuity);
  * the tail judges `acc` with pspec.accept.Frontier off the eid, honours `lg`, drafts `blk`, and
    refuses a cpos past its own frontier before anything moves;
  * a stream reset (pred -> tail, tail -> entry) is not a death: the tail's session survives;
  * END TO END: S1's coordinator (`coord.run(link, coord.Session(...))`) over D = 1..7 mixed
    simulated devices on real loopback TCP, chunk and per-token policies, tail and entry drafters
    with partial acceptance: tokens == the direct greedy loop, every reply's echo checked by the
    coordinator. A reply whose echo is tampered with is detected.
"""
import random
import time

import numpy as np
import pytest

import fake_mesh as FM
from fake_mesh import ENTRY, FakeMesh, HopError, direct_greedy, hexsid, reply_tokens
from shard import frame
from shard.mesh import links as LK
from shard.mesh.clock import Timeout
from shard.mesh.node import NodeServer
from shard.pspec import coord


# ── the coordinator's seat on the mesh ────────────────────────────────────────────────────────
class HopRefusal(coord.SessionFailed):
    def __init__(self, sid, peer, code, seq):
        super().__init__(sid, f"{peer} refused frame seq {seq}: {code}")
        self.peer, self.code, self.seq = peer, code, seq


class MeshLink:
    """coord.Link over the mesh entry's TcpLinks (real loopback TCP). Frames go to the head.
    Replies come from the tail, refusals (`ack` with a seq) from ANY hop, each on its own one-way
    stream into the entry's inbox, which is why this is not coord.SocketLink (one return socket
    cannot carry a mid hop's refusal). A refusal or an undeliverable frame raises ConnectionError
    naming the hop (coord.SessionFailed: run_many fails that session only); lease acks (no seq)
    are skipped. `tamper(reply) -> reply` corrupts replies."""

    def __init__(self, mesh, tamper=None):
        self.m, self.tamper = mesh, tamper
        self.replies = []

    def send(self, msg):
        self.m.entry.links.send(self.m.peers[0], msg)

    def recv(self, timeout):
        end = time.monotonic() + timeout
        while True:
            left = end - time.monotonic()
            if left <= 0:
                raise TimeoutError("mesh: no reply")
            try:
                ev = self.m.entry.inbox.get(timeout=left)
            except Timeout:
                raise TimeoutError("mesh: no reply") from None
            if ev[0] == "msg":
                _, peer, msg, _t = ev
                if msg["op"] == "r":
                    self.replies.append(msg)
                    return self.tamper(msg) if self.tamper is not None else msg
                if msg["op"] == "c" and msg["cmd"] == "ack" and msg["seq"] is not None:
                    raise HopRefusal(msg["sid"], peer, msg["code"], msg["seq"])
                continue
            _, peer, meta, _t = ev
            if meta.get("op") == "f":
                raise coord.SessionFailed(meta["sid"], f"frame seq {meta.get('seq')} "
                                                       f"undeliverable to {peer}")


def _wrong(t, d):
    return (t + 1 + d) % FM.VOCAB             # a valid id that is never the truth


def _miss(alpha, seed):
    def miss(p, d):
        return random.Random(f"{seed}/{p}/{d}").random() >= alpha
    return miss


def tail_drafter(box, K, miss):
    """A toy tail drafter plugin: factory(lease) -> blocks(cfront, mfront), proposals for
    cfront+2..cfront+K+1 read off the greedy stream box["T"], wrong where miss(pos, depth)."""
    def factory(_lease):
        def blocks(q, _m):
            T = box["T"]
            return [_wrong(T[p], d) if miss(p, d) else T[p]
                    for d, p in enumerate(range(q + 2, q + 2 + K), 1) if p < len(T)]
        return blocks
    return factory


def entry_drafter(T, K, miss):
    def drafter(ids):
        s = len(ids)
        return [_wrong(T[p], d) if miss(p, d) else T[p]
                for d, p in enumerate(range(s, s + K), 1) if p < len(T)]
    return drafter


def run_mesh(m, sid, prompt, max_new, *, policy, W=8, drafter=None, tail=True, link=None,
             timeout=20.0):
    """One pspec job over the mesh: leases exactly the rows the job can use, then coord.run."""
    P = len(prompt)
    m.open(sid, max_pos=P + max_new - 1, W=W, drafter="toy" if tail else None)
    s = coord.Session(prompt, max_new, sid=hexsid(sid), rte=1, W=W, policy=policy,
                      drafter=None if tail else drafter, seal=bytes)
    link = link or MeshLink(m)
    r = coord.run(link, s, timeout=timeout)
    return r, s, link


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


def _eid(ids):
    return frame.eid_pack(ids)


def _submit(m, sid, *, kind, pos, ids, sep=0, cpos=None, lg="all", eid=True, rte=1):
    """FakeMesh.submit plus the eid (FakeMesh's submit carries none). eid: True = the frame's
    ids, a list = those ids instead, False = none."""
    seq = m._seq[sid]
    m._seq[sid] = seq + 1
    n = len(ids)
    if cpos is None:
        cpos = pos + n - 1 if kind == "P" else pos
    if eid is True:
        eid = ids
    body = frame.frame(sid=hexsid(sid), rte=rte, seq=seq, sep=sep, pos=pos, n=n, kind=kind, cpos=cpos,
                       lg=lg, cx="ids", x=frame.Tensor("i32", (n,), np.asarray(ids, dtype="<i4")),
                       eid=_eid(eid) if eid else None)
    m._sent_at[(hexsid(sid), seq)] = m.clock.now()
    m.entry.links.send(m.peers[0], body)
    return seq


def _req(m, sid, **kw):
    return m.collect(sid, _submit(m, sid, **kw))


def _state(m, sid):
    """Every hop's (length, committed) and the tail's accept frontier."""
    rows = tuple((n.rt.info(sid).length, n.rt.info(sid).committed) for n in m.nodes)
    fr = m.nodes[-1]._leases[sid].front
    return rows, (fr.cfront, fr.mfront)


def _settle(m):
    m.clock.settle()
    if not m.clock.virtual:
        time.sleep(0.2)


# ── S2: the fence ─────────────────────────────────────────────────────────────────────────────
def _fence_schedule(m, sid=1):
    """prefill; T at P (sep 1); a STALE sep-0 frame rewinding row P to a wrong token; then the
    on-path frame at P+1 (sep 1). -> (stale reply, state before/after the stale frame, last reply,
    truth)."""
    prompt = [5, 9, 2, 7]
    P = len(prompt)
    T = direct_greedy(m.weights, prompt, 6)
    m.open(sid, W=8)
    assert reply_tokens(_req(m, sid, kind="P", pos=0, ids=prompt, lg="last")) == [T[P]]
    r = _req(m, sid, kind="T", pos=P, ids=[T[P]], sep=1)
    assert reply_tokens(r) == [T[P + 1]] and r["acc"] is True
    _settle(m)
    before = _state(m, sid)
    stale = _req(m, sid, kind="S", pos=P, ids=[_wrong(T[P], 0)], sep=0, cpos=P)
    _settle(m)
    after = _state(m, sid)
    last = _req(m, sid, kind="T", pos=P + 1, ids=[T[P + 1]], sep=1, cpos=P + 1)
    return stale, before, after, last, T, P


@pytest.mark.parametrize("D", [1, 3])
def test_stale_sep_is_fenced_and_changes_nothing(mesh_factory, D):
    m = mesh_factory(D)
    stale, before, after, last, T, P = _fence_schedule(m)
    assert stale["fen"] and stale["toks"] is None and stale["acc"] is None
    assert (stale["seq"], stale["sep"], stale["pos"], stale["n"]) == (2, 0, P, 1)
    assert after == before                          # no KV row, no frontier, no accept frontier
    assert all(n.snapshot().get("fenced") == 1 for n in m.nodes)
    assert reply_tokens(last) == [T[P + 2]] and last["acc"] is True


def test_with_the_fence_stubbed_the_stale_frame_corrupts(mesh_factory, monkeypatch):
    """Anti-vacuity: the same schedule with the node's fence disabled lets the stale frame rewind
    row P to a wrong token, and the next on-path frame (still acc=True: its first row feeds the
    tail's frontier token) answers a token the greedy stream does not have."""
    monkeypatch.setattr(NodeServer, "_fenced", staticmethod(lambda L, f: False))
    m = mesh_factory(3)
    stale, before, after, last, T, P = _fence_schedule(m)
    assert not stale["fen"]                          # it was computed: row P now holds the wrong id
    assert last["acc"] is True and reply_tokens(last) != [T[P + 2]]


# ── S2: seek / window / the tail's cpos check ─────────────────────────────────────────────────
def test_refusals_move_nothing_and_the_tail_frontier_holds(mesh_factory):
    m = mesh_factory(3)
    prompt = [3, 1, 4, 1, 5]
    P = len(prompt)
    T = direct_greedy(m.weights, prompt, 8)
    m.open(1, W=4)
    assert _req(m, 1, kind="P", pos=0, ids=prompt, lg="last")["acc"] is True
    r = _req(m, 1, kind="S", pos=P, ids=T[P:P + 3], cpos=P)
    assert reply_tokens(r) == T[P + 1:P + 4] and r["acc"] is True
    _settle(m)
    before = _state(m, 1)
    assert before[1] == (P + 2, T[P + 3])
    cases = [
        (dict(kind="T", pos=P + 5, ids=[T[P + 5]], cpos=P + 2), "gap"),
        (dict(kind="T", pos=P + 1, ids=[T[P + 1]], cpos=P + 2), "rollback"),     # below its cpos
        (dict(kind="S", pos=P + 2, ids=T[P + 2:P + 7], cpos=P + 2), "window"),  # > cpos + W
    ]
    for kw, code in cases:
        with pytest.raises(HopError) as ei:
            _req(m, 1, **kw)
        assert (ei.value.code, ei.value.peer) == (code, "hop0"), kw
        _settle(m)
        assert _state(m, 1) == before, kw
    # the tail resumes at its frontier: a rewind to cf+1 re-feeds the committed token there
    r = _req(m, 1, kind="T", pos=P + 3, ids=[T[P + 3]], cpos=P + 3)
    assert reply_tokens(r) == [T[P + 4]] and r["acc"] is True
    # a cpos past the tail's own frontier + 1 is refused AT THE TAIL before anything moves there
    # (the upstream hops, which cannot judge, already committed and computed it: the session is
    # wedged until the entry fails over, which is what a loud refusal is for)
    _settle(m)
    before = _state(m, 1)
    with pytest.raises(HopError) as ei:
        _req(m, 1, kind="T", pos=P + 4, ids=[T[P + 4]], cpos=P + 5)
    assert (ei.value.code, ei.value.peer) == ("bad", "hop2")
    _settle(m)
    rows, fr = _state(m, 1)
    assert fr == before[1] and rows[-1] == before[0][-1]


def test_off_path_frame_is_judged_false_and_lg_is_honoured(mesh_factory):
    m = mesh_factory(2)
    prompt = [8, 8, 1]
    P = len(prompt)
    T = direct_greedy(m.weights, prompt, 8)
    m.open(1, W=8)
    r = _req(m, 1, kind="P", pos=0, ids=prompt, lg="last")
    assert r["acc"] is True and reply_tokens(r) == [T[P]]
    # a chunk that diverges at row 1: the tail's frontier stops at P+1
    r = _req(m, 1, kind="S", pos=P, ids=[T[P], T[P + 1], _wrong(T[P + 2], 0)], lg="last")
    assert r["acc"] is True and len(reply_tokens(r)) == 1
    fr = m.nodes[-1]._leases[1].front
    _settle(m)
    assert (fr.cfront, fr.mfront) == (P + 1, T[P + 2])   # row P+1's draft was wrong: T[P+2] next
    # a frame past the diverged row is off the committed path: acc False, nothing moves
    cf = (fr.cfront, fr.mfront)
    r = _req(m, 1, kind="S", pos=P + 3, ids=[T[P + 3]], cpos=P + 1, lg="none")
    assert r["acc"] is False and reply_tokens(r) == [] and (fr.cfront, fr.mfront) == cf
    # a frame without an eid is not judged (acc absent)
    r = _req(m, 1, kind="S", pos=cf[0] + 1, ids=[cf[1]], cpos=P + 1, eid=False)
    assert r["acc"] is None


# ── S2: predecessor binding and reconnect ─────────────────────────────────────────────────────
def test_a_frame_from_a_non_predecessor_is_dropped_at_the_tail(mesh_factory):
    m = mesh_factory(3)
    prompt = [2, 7, 1, 8]
    P = len(prompt)
    T = direct_greedy(m.weights, prompt, 4)
    m.open(1)
    _req(m, 1, kind="P", pos=0, ids=prompt, lg="last")
    _settle(m)
    before = _state(m, 1)
    H = FM.HIDDEN
    body = frame.frame(sid=hexsid(1), rte=1, seq=50, sep=0, pos=P, n=1, kind="T", cpos=P, lg="all",
                       cx="bf16", x=frame.Tensor("bf16", (1, 2 * H), np.zeros(2 * H, "<u2")),
                       eid=_eid([T[P]]))
    m.links_of("hop0").send("hop2", body)            # hop0 is not the tail's predecessor
    m.entry.links.send("hop2", body)                 # nor is the entry
    _settle(m)
    assert m.nodes[2].snapshot()["drop_not_pred"] == 2
    assert _state(m, 1) == before
    r = _req(m, 1, kind="T", pos=P, ids=[T[P]])
    assert reply_tokens(r) == [T[P + 1]] and r["acc"] is True


def _reset_streams(node, out_peers):
    """A stream reset, not a death: abort the node's inbound streams and its pooled outbound
    streams to `out_peers`. The node process and its sessions stay up."""
    with node.links._lock:
        inbound = list(node.links._inbound)
    for s in inbound:
        LK._abort(s)
    for p in out_peers:
        o = node.links._outs.get(p)
        if o is not None and o.sock is not None:
            LK._shut(o.sock)


@pytest.mark.parametrize("D", [2, 4])
def test_reconnect_keeps_the_tail_alive(mesh_factory, D):
    m = mesh_factory(D, clock="real", cost_scale=0.0)
    prompt = [6, 2, 6, 5]
    P = len(prompt)
    T = direct_greedy(m.weights, prompt, 10)
    m.open(1)
    _req(m, 1, kind="P", pos=0, ids=prompt, lg="last")
    r = _req(m, 1, kind="S", pos=P, ids=T[P:P + 3])
    assert r["acc"] is True and reply_tokens(r) == T[P + 1:P + 4]
    tail = m.nodes[-1]
    redials = tail.links.stats.get("redials", 0)
    _reset_streams(tail, [ENTRY])
    time.sleep(0.2)
    for k in range(3, 7):                            # every stream re-dialled, the session lives
        r = _req(m, 1, kind="T", pos=P + k, ids=[T[P + k]])
        assert r["acc"] is True and reply_tokens(r) == [T[P + k + 1]]
        if k == 4:
            _reset_streams(tail, [ENTRY])
            time.sleep(0.2)
    assert tail.alive and tail.links.stats["redials"] >= redials + 2
    assert tail.snapshot().get("faults", 0) == 0


# ── S2 end to end: S1's coordinator over the real node server ─────────────────────────────────
MIXES = [(1, "chunk", True), (2, "pertoken", True), (3, "chunk", False), (4, "pertoken", False),
         (5, "chunk", True), (6, "pertoken", True), (7, "chunk", False), (7, "pertoken", True)]


@pytest.mark.parametrize("D,policy,tail", MIXES)
def test_coordinator_over_the_mesh_is_lossless(mesh_factory, D, policy, tail):
    K, W, alpha = 4, 8, 0.6
    box = {}
    miss = _miss(alpha, seed=D)
    rng = random.Random(D)
    legs = [rng.uniform(0.0, 3.0) for _ in range(D + 1)]
    m = mesh_factory(D, clock="real", cost_scale=0.0, legs_ms=legs,
                     drafters={"toy": tail_drafter(box, K, miss)})
    out = {}
    for sid, (plen, max_new) in enumerate([(5, 24), (9, 17), (3, 30)], 1):
        prompt = [rng.randrange(FM.VOCAB) for _ in range(plen)]
        T = direct_greedy(m.weights, prompt, max_new + K + 2)
        box["T"] = T
        r, s, link = run_mesh(m, sid, prompt, max_new, policy=policy, W=W, tail=tail,
                              drafter=entry_drafter(T, K, miss))
        assert r["tokens"] == T[plen:plen + max_new], (sid, r)
        out[sid] = r
        m.close_session(sid)
    total = {k: sum(r[k] for r in out.values()) for k in ("accepted", "cancels", "generated", "frames")}
    assert total["accepted"] > 0 and total["cancels"] > 0, total     # partial acceptance, really
    assert total["frames"] < total["generated"] + len(out) * 2 or policy == "pertoken"
    snap = m.nodes[-1].snapshot()
    if tail:
        assert snap["judged_acc"] > 0
    assert snap.get("faults", 0) == 0 and not any(k.startswith("refused") for k in snap)


@pytest.mark.parametrize("D,floor,W", [(3, 3, 8), (5, 2, 6), (7, 4, 8)])
def test_deep_pipelines_with_topups_are_lossless(mesh_factory, D, floor, W):
    """Per-token frames with a refill floor > 1 keep up to W positions in flight, so every cancel
    leaves a tail of stale frames on the wire: the tail judges them off-path (acc False), the
    coordinator discards them by sep, and the committed stream is still the greedy one."""
    box, rng = {}, random.Random(D)
    m = mesh_factory(D, clock="real", cost_scale=0.0,
                     legs_ms=[rng.uniform(0.0, 3.0) for _ in range(D + 1)],
                     drafters={"toy": tail_drafter(box, 5, _miss(0.7, D))})
    tot = {"stale_replies": 0, "topups": 0}
    for sid in (1, 2, 3):
        prompt = [rng.randrange(FM.VOCAB) for _ in range(6)]
        T = direct_greedy(m.weights, prompt, 40)
        box["T"] = T
        m.open(sid, max_pos=6 + 30 - 1, W=W, drafter="toy")
        s = coord.Session(prompt, 30, sid=hexsid(sid), rte=1, W=W, policy="pertoken", floor=floor,
                          seal=bytes)
        r = coord.run(MeshLink(m), s, timeout=20.0)
        assert r["tokens"] == T[6:36], sid
        for k in tot:
            tot[k] += r[k]
    snap = m.nodes[-1].snapshot()
    assert tot["stale_replies"] > 0 and tot["topups"] > 0, tot
    assert snap["judged_off"] == tot["stale_replies"]       # each stale reply judged off-path


def test_coordinator_over_the_mesh_with_eos(mesh_factory):
    box = {}
    m = mesh_factory(3, clock="real", cost_scale=0.0, drafters={"toy": tail_drafter(box, 3, _miss(0.7, 1))})
    prompt = [11, 22, 33, 44]
    P = len(prompt)
    T = direct_greedy(m.weights, prompt, 40)
    box["T"] = T
    eos = T[P + 9]
    want = T[P:P + T[P:].index(eos) + 1]
    m.open(1, max_pos=P + 39, W=8, drafter="toy")
    s = coord.Session(prompt, 40, sid=hexsid(1), rte=1, W=8, policy="pertoken", eos_ids=(eos,),
                      seal=bytes)
    r = coord.run(MeshLink(m), s, timeout=20.0)
    assert r["tokens"] == want


@pytest.mark.parametrize("field", ["seq", "sep", "pos", "n", "sid"])
def test_a_tampered_echo_is_detected(mesh_factory, field):
    box = {}
    m = mesh_factory(2, clock="real", cost_scale=0.0, drafters={"toy": tail_drafter(box, 3, _miss(0.6, 2))})
    prompt = [1, 2, 3]
    box["T"] = direct_greedy(m.weights, prompt, 20)
    count = [0]

    def tamper(rep):
        count[0] += 1
        if count[0] != 4:
            return rep
        rep = dict(rep)
        rep[field] = "00000000000000ff" if field == "sid" else rep[field] + 1
        return rep
    with pytest.raises(RuntimeError, match=r"reply \(sid, rte, seq, sep, pos, n\)"):
        run_mesh(m, 1, prompt, 15, policy="pertoken", link=MeshLink(m, tamper=tamper))


def test_a_node_refusal_surfaces_at_the_coordinator(mesh_factory):
    """A lease one row short of what the job uses: the node refuses (window) and the coordinator
    fails loudly naming the hop, instead of hanging."""
    box = {}
    m = mesh_factory(3, clock="real", cost_scale=0.0, drafters={"toy": tail_drafter(box, 4, _miss(1.0, 0))})
    prompt = [4, 5, 6]
    P = len(prompt)
    box["T"] = direct_greedy(m.weights, prompt, 30)
    m.open(1, max_pos=P + 10 - 2, W=8, drafter="toy")
    s = coord.Session(prompt, 10, sid=hexsid(1), rte=1, W=8, policy="chunk", seal=bytes)
    with pytest.raises(HopRefusal) as ei:
        coord.run(MeshLink(m), s, timeout=10.0)
    assert ei.value.code == "window" and ei.value.peer == "hop0"


# ── skeptic-review regressions ────────────────────────────────────────────────────────────────
def test_a_drafter_fault_costs_the_block_not_the_node(mesh_factory):
    box, calls = {}, [0]
    good = tail_drafter(box, 3, _miss(0.6, 5))

    def flaky(lease):
        if lease["sid"] == hexsid(9):
            raise RuntimeError("plugin failed to load")
        inner = good(lease)

        def blocks(q, mf):
            calls[0] += 1
            if calls[0] % 3 == 0:
                raise RuntimeError("drafter fault")
            return inner(q, mf)
        return blocks
    m = mesh_factory(3, clock="real", cost_scale=0.0, drafters={"toy": flaky})
    assert m.open(9, drafter="toy", expect_ok=False)["hop2"] == "unsupported"
    prompt = [7, 3, 9, 1]
    T = direct_greedy(m.weights, prompt, 30)
    box["T"] = T
    r, _s, _l = run_mesh(m, 1, prompt, 20, policy="pertoken")
    assert r["tokens"] == T[4:24]
    assert m.nodes[2].alive and m.nodes[2].snapshot()["draft_errors"] > 0
    assert m.open(9, drafter=None, expect_ok=False)["hop2"] == "ok"   # the tail kept nothing


def test_reopen_restores_the_frontier_token_from_the_tails_record(mesh_factory):
    m = mesh_factory(3)
    prompt = [2, 9, 4]
    P = len(prompt)
    T = direct_greedy(m.weights, prompt, 10)
    m.open(1, W=8)
    _req(m, 1, kind="P", pos=0, ids=prompt, lg="last")
    assert _req(m, 1, kind="S", pos=P, ids=T[P:P + 4])["acc"] is True     # frontier at P+3
    F = P + 2
    for peer in m.peers:
        m.entry.links.send(peer, frame.control("lease_update", sid=hexsid(1), rte=2,
                                               truncate_to=F, sig=b"s"))
    m.entry.pump_until(lambda: all(m.entry.acks.get((hexsid(1), p), ("",))[1:2] == (2,)
                                   for p in m.peers), m.clock.now() + 1.0)
    fr = m.nodes[-1]._leases[1].front
    _settle(m)
    assert (fr.cfront, fr.mfront) == (F - 1, T[F])  # from the record, not from the entry
    r = _req(m, 1, kind="T", pos=F, ids=[_wrong(T[F], 0)], cpos=F, rte=2)  # a wrong re-feed
    assert r["acc"] is False
    r = _req(m, 1, kind="T", pos=F, ids=[T[F]], cpos=F, rte=2)
    assert r["acc"] is True and reply_tokens(r) == [T[F + 1]]


def test_solo_refuses_an_eid_that_is_not_the_ids_it_computes(mesh_factory):
    m = mesh_factory(1)
    prompt = [5, 5, 6]
    P = len(prompt)
    T = direct_greedy(m.weights, prompt, 5)
    m.open(1)
    _req(m, 1, kind="P", pos=0, ids=prompt, lg="last")
    with pytest.raises(HopError) as ei:
        _req(m, 1, kind="T", pos=P, ids=[_wrong(T[P], 0)], eid=[T[P]])
    assert ei.value.code == "bad"
    assert _req(m, 1, kind="T", pos=P, ids=[T[P]])["acc"] is True


def test_a_second_prefill_is_refused_at_the_tail(mesh_factory):
    m = mesh_factory(2)
    prompt = [1, 1, 2, 3]
    P = len(prompt)
    m.open(1)
    _req(m, 1, kind="P", pos=0, ids=prompt, lg="last")
    _settle(m)
    before = _state(m, 1)[1]
    with pytest.raises(HopError) as ei:
        _req(m, 1, kind="P", pos=P, ids=[40, 41, 42], lg="last")
    assert (ei.value.code, ei.value.peer) == ("bad", "hop1")
    _settle(m)
    assert _state(m, 1)[1] == before


def test_a_judged_frame_over_the_tail_budget(mesh_factory):
    """_run_big with judging: a 7-row spec frame through a tail whose budget is 3."""
    m = mesh_factory(2)
    m.nodes[-1]._budget = 3
    prompt = [3, 8, 3]
    P = len(prompt)
    T = direct_greedy(m.weights, prompt, 12)
    m.open(1, W=8)
    _req(m, 1, kind="P", pos=0, ids=prompt, lg="last")
    ids = T[P:P + 5] + [_wrong(T[P + 5], 0), T[P + 6]]
    r = _req(m, 1, kind="S", pos=P, ids=ids, lg="all")
    assert r["acc"] is True and reply_tokens(r)[:5] == T[P + 1:P + 6]
    fr = m.nodes[-1]._leases[1].front
    _settle(m)
    assert (fr.cfront, fr.mfront) == (P + 4, T[P + 5])
    assert m.nodes[-1].snapshot()["steps"] >= 1 + 3
