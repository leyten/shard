"""F4 gate: node server v0 + tests/fake_mesh.py + the injectable clock (BUILD-PLAN §1 F4).

  * toy greedy through the mesh == the direct toy loop (tests/oracle.py's position oracle) for
    D = 1..7 hops of MIXED simulated devices, several sessions at once, both clocks;
  * injected leg delays are honoured: exactly per leg in virtual time, within ±10% on the wall;
  * kill / hang / partition / malformed-frame each produce their expected, observable failure
    (never a harness hang), and the mesh recovers;
  * a virtual-time run of 10^3 faults finishes in < 5 min;
  * anti-vacuity: a hop with other weights, ignored delays and no-op injectors all FAIL the
    checks that would otherwise pass.

Run: python3 -m pytest tests/test_fake_mesh.py -q
"""
import random
import time

import numpy as np
import pytest

import fake_mesh as FM
from fake_mesh import (ENTRY, Deadline, FakeMesh, HopError, LinkDown, Refused, direct_greedy,
                       fault_storm, hexsid, reply_tokens)
from oracle import oracle_token
from shard import frame


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


def _prompts(k, seed):
    rng = random.Random(seed)
    return [[rng.randrange(FM.VOCAB) for _ in range(rng.randrange(4, 13))] for _ in range(k)]


def _check_equal_direct(m, prompts, n_new):
    got = m.generate_many({100 + i: p for i, p in enumerate(prompts)}, n_new)
    for i, p in enumerate(prompts):
        T = direct_greedy(m.weights, p, n_new, pattern=m.pattern, window=m.window, mode=m.mode)
        assert got[100 + i] == T[len(p):], f"session {i}: mesh {got[100 + i]} != direct {T[len(p):]}"
    # every reply on the wire obeyed the position oracle, and came from the tail
    Ts = {hexsid(100 + i): direct_greedy(m.weights, p, n_new, pattern=m.pattern, window=m.window,
                                         mode=m.mode) for i, p in enumerate(prompts)}
    for peer, r, _ in m.entry.received:
        assert peer == m.peers[-1]
        assert reply_tokens(r)[-1] == oracle_token(Ts[r["sid"]], r["pos"] + r["n"] - 1)
    assert len(m.entry.received) == len(prompts) * n_new
    return got


# ── the equality gate ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("D", range(1, 8))
def test_greedy_through_the_mesh_equals_the_direct_loop(mesh_factory, D):
    rng = random.Random(D)
    legs = [rng.uniform(0.5, 20.0) for _ in range(D + 1)]
    m = mesh_factory(D, legs_ms=legs)
    prompts = _prompts(3, D)
    for i in range(3):
        m.open(100 + i)
    _check_equal_direct(m, prompts, 16)
    # the hops did the work: every hop computed every row of every session
    rows = sum(len(p) + 15 for p in prompts)
    assert [n.snapshot()["rows"] for n in m.nodes] == [rows] * D
    # and the fleet is mixed, with spans proportional to memory
    if D >= 2:
        assert len({n.rt.describe().device.arch_class for n in m.nodes}) >= 2
    gbs = [d.gb for d in m.devices]
    sizes = [hi - lo for lo, hi in m.spans]
    assert sum(sizes) == len(m.pattern)
    for a in range(D):
        for b in range(D):
            if gbs[a] > gbs[b]:
                assert sizes[a] >= sizes[b], (gbs, sizes)


@pytest.mark.parametrize("D", [1, 4, 7])
def test_greedy_on_the_real_clock(mesh_factory, D):
    m = mesh_factory(D, clock="real", cost_scale=0.0)
    for i in range(2):
        m.open(100 + i)
    _check_equal_direct(m, _prompts(2, 50 + D), 10)


@pytest.mark.parametrize("D", [2, 5])
def test_adversarial_numerics_same_schedule_same_bytes(mesh_factory, D):
    """Bytes that depend on segment shapes (like float kernels) still match the direct loop:
    the node forwards frames unsplit, so every hop runs the direct loop's shapes."""
    m = mesh_factory(D, mode="adversarial")
    for i in range(2):
        m.open(100 + i)
    _check_equal_direct(m, _prompts(2, 70 + D), 12)


def test_a_frame_larger_than_a_hops_budget_runs_in_steps(mesh_factory):
    """A 24 GB hop's budget is 32 tokens: a 45-token prompt runs there as 2 steps, forwarded
    whole (chunk-invariant numerics make that byte-identical)."""
    m = mesh_factory(3)
    assert m.devices[0].budget < 45
    m.open(100, max_pos=128)
    prompt = _prompts(1, 3)[0] * 5
    prompt = (prompt + prompt)[:45]
    got = m.generate(100, prompt, 6)
    assert got == direct_greedy(m.weights, prompt, 6)[45:]
    assert m.nodes[0].snapshot()["steps"] == 2 + 5


def test_many_sessions_are_batched(mesh_factory):
    devs = [FM.SIM_DEVICES[k] for k in ("rtxpro6000-96g", "dgx-spark-128g", "mac-m3ultra-128g")]
    m = mesh_factory(3, devices=devs, legs_ms=[2, 2, 2, 2])     # big hops: S_max 16 each
    prompts = _prompts(8, 11)
    for i in range(8):
        m.open(100 + i)
    _check_equal_direct(m, prompts, 8)
    assert max(n.snapshot().get("max_batch", 0) for n in m.nodes) > 1


def test_the_fleet_is_heterogeneous_and_admission_follows_it(mesh_factory):
    m = mesh_factory(4)                           # 24 GB, 32 GB, 96 GB, 128 GB
    recs = [n.describe() for n in m.nodes]
    assert [r["device"]["total_mem_bytes"] >> 30 for r in recs] == [24, 32, 96, 128]
    assert len({r["device"]["device_name"] for r in recs}) == 4
    assert {r["device"]["arch_class"] for r in recs} == {"sm_86", "sm_120", "sm_121"}
    assert [r["device"]["unified_memory"] for r in recs] == [False, False, False, True]
    assert len({tuple(r["service_ms"][0]) for r in recs}) == 4       # different cost curves
    assert recs[0]["role"] == "head" and recs[-1]["role"] == "tail"
    assert [n.s_max for n in m.nodes] == [4, 6, 16, 16]              # min(16, table // 2)
    for i in range(4):
        m.open(100 + i)
    codes = m.open(104, expect_ok=False)                             # the 24 GB hop is full
    assert codes == {"hop0": "busy", "hop1": "ok", "hop2": "ok", "hop3": "ok"}
    mac = FakeMesh(2, devices=[FM.SIM_DEVICES["mac-m3ultra-128g"], FM.SIM_DEVICES["rtx3090-24g"]],
                   stuck_after=30.0)
    try:
        assert mac.spans == [(0, 20), (20, 24)]                      # the Mac carries more layers
        assert mac.nodes[0].describe()["device"]["backend"] == "mlx"
        mac.open(1)
        assert mac.generate(1, [1, 2, 3], 5) == direct_greedy(mac.weights, [1, 2, 3], 5)[3:]
    finally:
        mac.close()


# ── delays ────────────────────────────────────────────────────────────────────────────────────
def leg_timing_errors(m, sid, n_frames=3):
    """Run decode frames with tracing on; compare each hop's arrival time and the reply's
    arrival with the model: sent + legs so far + every earlier hop's modelled compute."""
    errs = []
    P = 5
    r = m.request(sid, kind="P", pos=0, ids=list(range(P)))
    tok = reply_tokens(r)[0]
    for k in range(n_frames):
        m.trace_log.clear()
        seq = m.submit(sid, kind="T", pos=P + k, ids=[tok])
        r = m.collect(sid, seq)
        tok = reply_tokens(r)[0]
        t0 = m.sent_at(sid, seq)
        arrive = {peer: t for ev, peer, s, q, _, _, t in m.trace_log if ev == "in" and q == seq}
        t = t0
        for i, node in enumerate(m.nodes):
            t += m.legs_ms[i] / 1000.0
            if abs(arrive[m.peers[i]] - t) > 1e-9:
                errs.append((k, m.peers[i], arrive[m.peers[i]] - t0, t - t0))
            t += node.rt.cost_s(1)
        t += m.legs_ms[-1] / 1000.0
        got = [e for e in m.entry.received if e[1]["sid"] == hexsid(sid) and e[1]["seq"] == seq][0][2]
        if abs(got - t) > 1e-9:
            errs.append((k, ENTRY, got - t0, t - t0))
    return errs


def test_leg_delays_are_exact_in_virtual_time(mesh_factory):
    legs = [3.0, 11.0, 0.5, 7.0, 19.0]
    m = mesh_factory(4, legs_ms=legs, trace=True)
    m.open(1)
    assert leg_timing_errors(m, 1) == []
    m.set_legs([40.0, 0.0, 25.0, 1.0, 2.0])       # changed on a live mesh
    m.open(2)
    assert leg_timing_errors(m, 2) == []


def test_ignored_delays_are_caught(mesh_factory, monkeypatch):
    """Anti-vacuity: links that silently drop the delay fail the timing check."""
    monkeypatch.setattr(FM.FaultyLinks, "_delay", lambda self, peer: 0.0)
    m = mesh_factory(3, legs_ms=[5.0, 5.0, 5.0, 5.0], trace=True)
    m.open(1)
    assert leg_timing_errors(m, 1, n_frames=1) != []


def test_leg_delays_are_honoured_on_the_wall_clock(mesh_factory):
    """Real time: the median round trip grows by the injected leg delays, within ±10%."""
    m = mesh_factory(3, clock="real", cost_scale=0.0)
    m.open(1)
    tok = reply_tokens(m.request(1, kind="P", pos=0, ids=[1, 2, 3]))[0]
    pos = 3

    def median_rtt(k=7):
        nonlocal tok, pos
        out = []
        for _ in range(k):
            t0 = time.monotonic()
            tok = reply_tokens(m.request(1, kind="T", pos=pos, ids=[tok]))[0]
            out.append(time.monotonic() - t0)
            pos += 1
        return sorted(out)[k // 2]

    base = median_rtt()
    legs = [20.0, 45.0, 30.0, 55.0]
    m.set_legs(legs)
    with_delay = median_rtt()
    added, want = with_delay - base, sum(legs) / 1000.0
    assert abs(added - want) <= 0.10 * want, f"added {added * 1000:.1f} ms for {want * 1000:.0f} ms of legs"


# ── fault injectors, one at a time ────────────────────────────────────────────────────────────
def _decode_ready(m, sid, prompt=(1, 2, 3, 4)):
    m.open(sid)
    tok = reply_tokens(m.request(sid, kind="P", pos=0, ids=list(prompt)))[0]
    return len(prompt), tok


CLOCKS = [pytest.param("virtual", id="virtual"), pytest.param("real", id="real")]


def _mk(mesh_factory, clock, D=4, **kw):
    if clock == "real":
        kw.setdefault("deadline", 0.4)
        kw.setdefault("cost_scale", 0.0)
    kw.setdefault("legs_ms", [2.0] * (D + 1))
    return mesh_factory(D, clock=clock, **kw)


@pytest.mark.parametrize("clock", CLOCKS)
@pytest.mark.parametrize("hop", [0, 2, 3])
def test_kill_is_a_fast_named_link_error(mesh_factory, clock, hop):
    m = _mk(mesh_factory, clock)
    pos, tok = _decode_ready(m, 1)
    m.kill(hop)
    t0 = m.clock.now()
    with pytest.raises((LinkDown, HopError)) as ei:
        m.request(1, kind="T", pos=pos, ids=[tok])
    e = ei.value
    if hop == 0:
        assert isinstance(e, LinkDown) and e.peer == "hop0"
    else:
        assert isinstance(e, HopError) and e.code == "next_down" and e.peer == f"hop{hop - 1}"
    assert e.t - t0 < m.deadline                  # a link error, not a timeout
    m.restart(hop)                                # recovery: a fresh session serves again
    m.open(2)
    assert m.generate(2, [5, 6, 7], 4) == direct_greedy(m.weights, [5, 6, 7], 4)[3:]


@pytest.mark.parametrize("clock", CLOCKS)
def test_hang_is_a_deadline_and_resume_serves_late(mesh_factory, clock):
    m = _mk(mesh_factory, clock)
    pos, tok = _decode_ready(m, 1)
    m.hang(2)
    seq = m.submit(1, kind="T", pos=pos, ids=[tok])
    with pytest.raises(Deadline) as ei:
        m.collect(1, seq)
    elapsed = ei.value.t - m.sent_at(1, seq)
    if clock == "virtual":
        assert elapsed == pytest.approx(m.deadline, abs=1e-9)
    else:
        assert m.deadline <= elapsed < m.deadline + 0.5
    m.resume(2)
    r = m.request(1, kind="T", pos=pos, ids=[tok])          # re-feed the same position
    T = direct_greedy(m.weights, [1, 2, 3, 4], 3)
    assert reply_tokens(r) == [T[pos + 1]]
    m.clock.settle()
    late = [x for _, x, _ in m.entry.received if x["seq"] == seq]
    if clock == "virtual":
        assert late and reply_tokens(late[0]) == [T[pos + 1]]           # the slow node answered


@pytest.mark.parametrize("clock", CLOCKS)
@pytest.mark.parametrize("leg", [0, 2, 4])
def test_partition_is_a_deadline_and_heal_recovers(mesh_factory, clock, leg):
    m = _mk(mesh_factory, clock)
    pos, tok = _decode_ready(m, 1)
    a, b = m.legs()[leg]
    m.partition(a, b)
    seq = m.submit(1, kind="T", pos=pos, ids=[tok])
    with pytest.raises(Deadline) as ei:
        m.collect(1, seq)
    if clock == "virtual":
        assert ei.value.t - m.sent_at(1, seq) == pytest.approx(m.deadline, abs=1e-9)
    assert m.links_of(a).stats["blocked"] >= 1
    m.heal()
    r = m.request(1, kind="T", pos=pos, ids=[tok])
    assert reply_tokens(r) == [direct_greedy(m.weights, [1, 2, 3, 4], 3)[pos + 1]]


@pytest.mark.parametrize("clock", CLOCKS)
@pytest.mark.parametrize("how", FM.MALFORMED)
@pytest.mark.parametrize("leg", [0, 2, 4])
def test_malformed_frame_resets_one_stream_only(mesh_factory, clock, how, leg):
    m = _mk(mesh_factory, clock)
    pos, tok = _decode_ready(m, 1)
    src, dst = m.legs()[leg]
    before = m.bad_frames(dst)
    m.malformed(src, dst, how)
    T = direct_greedy(m.weights, [1, 2, 3, 4], 4)
    try:                                          # the next message on that stream may be lost
        r = m.request(1, kind="T", pos=pos, ids=[tok])
        lost = False
    except Deadline:
        lost = True
    if lost or how == "truncated":
        r = m.request(1, kind="T", pos=pos, ids=[tok])      # re-feed: a fresh stream
    assert reply_tokens(r) == [T[pos + 1]]
    m.clock.settle()
    if clock == "real":
        for _ in range(50):
            if m.bad_frames(dst) > before:
                break
            time.sleep(0.02)
    assert m.bad_frames(dst) == before + 1
    assert all(n.alive for n in m.nodes)
    r = m.request(1, kind="T", pos=pos + 1, ids=[T[pos + 1]])
    assert reply_tokens(r) == [T[pos + 2]]


# ── node server v0 semantics ──────────────────────────────────────────────────────────────────
def test_gap_rollback_window_are_refused_loudly(mesh_factory):
    m = mesh_factory(2)
    pos, tok = _decode_ready(m, 1)
    with pytest.raises(HopError) as ei:
        m.request(1, kind="T", pos=pos + 3, ids=[tok])             # skips positions
    assert ei.value.code == "gap" and ei.value.peer == "hop0"
    with pytest.raises(HopError) as ei:
        m.request(1, kind="T", pos=1, ids=[tok])                  # below the committed frontier
    assert ei.value.code == "rollback"
    with pytest.raises(HopError) as ei:
        m.request(1, kind="S", pos=pos, ids=[tok] * 9, cpos=pos)  # W = 8 rows past the frontier
    assert ei.value.code == "window"
    r = m.request(1, kind="T", pos=pos, ids=[tok])                # the session is intact
    assert reply_tokens(r) == [direct_greedy(m.weights, [1, 2, 3, 4], 3)[pos + 1]]


def test_speculative_rewind_and_the_fence(mesh_factory):
    """A wrong draft is computed, then the true token re-fed at the same position (implicit
    truncate); a frame with an older sep is fenced (header-only through every hop)."""
    m = mesh_factory(3)
    prompt = [9, 8, 7]
    T = direct_greedy(m.weights, prompt, 6)
    m.open(1)
    P = len(prompt)
    assert reply_tokens(m.request(1, kind="P", pos=0, ids=prompt)) == [T[P]]
    wrong = (T[P] + 1) % FM.VOCAB
    r = m.request(1, kind="S", pos=P, ids=[T[P], wrong, wrong], lg="all", sep=0, cpos=P)
    toks = reply_tokens(r)
    assert len(toks) == 3 and toks[0] == T[P + 1]           # rows after the wrong draft: junk
    r = m.request(1, kind="T", pos=P + 1, ids=[T[P + 1]], sep=1)  # correction at the mismatch
    assert reply_tokens(r) == [T[P + 2]]
    seq = m.submit(1, kind="S", pos=P + 2, ids=[T[P + 2]], sep=0)   # stale sep: fenced
    r = m.collect(1, seq)
    assert r["fen"] and r["toks"] is None and r["sep"] == 0 and r["seq"] == seq
    assert all(n.snapshot().get("fenced") == 1 for n in m.nodes)
    r = m.request(1, kind="T", pos=P + 2, ids=[T[P + 2]], sep=1)  # nothing was computed
    assert reply_tokens(r) == [T[P + 3]]


def test_frames_need_the_lease_and_the_predecessor(mesh_factory):
    m = mesh_factory(3)
    pos, tok = _decode_ready(m, 1)
    # wrong rte: dropped at the head (no lease matches), so the entry sees only a deadline
    seq = m.submit(1, kind="T", pos=pos, ids=[tok], rte=7)
    with pytest.raises(Deadline):
        m.collect(1, seq)
    assert m.nodes[0].snapshot()["drop_no_lease"] == 1
    # the entry is not hop1's predecessor: a frame sent straight to hop1 is dropped
    body = frame.frame(sid=hexsid(1), rte=1, seq=99, sep=0, pos=pos, n=1, kind="T", cpos=pos,
                       lg="last", cx="bf16", x=frame.Tensor("bf16", (1, 2 * FM.HIDDEN),
                                                            np.zeros(2 * FM.HIDDEN, "<u2")))
    m.entry.links.send("hop1", body)
    m.clock.settle()
    if not m.clock.virtual:
        time.sleep(0.2)
    assert m.nodes[1].snapshot()["drop_not_pred"] == 1
    r = m.request(1, kind="T", pos=pos, ids=[tok])
    assert reply_tokens(r) == [direct_greedy(m.weights, [1, 2, 3, 4], 3)[pos + 1]]


def test_lease_admission_refusals(mesh_factory):
    m = mesh_factory(2)
    m.open(1)
    assert m.open(1, expect_ok=False) == {"hop0": "exists", "hop1": "exists"}
    assert m.open(2, W=65, expect_ok=False) == {"hop0": "w_too_deep", "hop1": "w_too_deep"}
    assert m.open(3, max_pos=10 ** 6, expect_ok=False) == {"hop0": "kv_full", "hop1": "kv_full"}
    # a lease for another span / role
    lease = frame.control("lease_open", sid=hexsid(4), rte=1, lo=0, hi=3, role="head", pred=ENTRY,
                          next="hop1", ret=ENTRY, max_pos=64, W=4, cx="ids", salt="5a" * 32,
                          nonce="00" * 8, lease=b"x")
    m.entry.links.send("hop0", lease)
    m.entry.pump_until(lambda: (hexsid(4), "hop0") in m.entry.acks, m.clock.now() + 1.0)
    assert m.entry.acks[(hexsid(4), "hop0")][0] == "span"
    with pytest.raises(Refused):
        m.open(1)


def test_lease_update_reroutes_with_a_new_rte(mesh_factory):
    """rte bump on every hop, truncate_to = F: frames of the old rte die, the new rte serves."""
    m = mesh_factory(3)
    prompt = [4, 4, 2]
    T = direct_greedy(m.weights, prompt, 5)
    pos, tok = _decode_ready(m, 1, prompt)
    assert tok == T[pos]
    reply_tokens(m.request(1, kind="T", pos=pos, ids=[tok]))
    F = pos                                       # re-feed the token at F on the new route
    for peer in m.peers:
        m.entry.links.send(peer, frame.control("lease_update", sid=hexsid(1), rte=2,
                                               truncate_to=F, sig=b"s"))
    m.entry.pump_until(lambda: all(m.entry.acks.get((hexsid(1), p), ("",))[0] == "ok"
                                   and m.entry.acks[(hexsid(1), p)][1] == 2 for p in m.peers),
                       m.clock.now() + 1.0)
    seq = m.submit(1, kind="T", pos=F, ids=[T[F]], rte=1)          # old route: dropped
    with pytest.raises(Deadline):
        m.collect(1, seq)
    r = m.request(1, kind="T", pos=F, ids=[T[F]], rte=2)
    assert reply_tokens(r) == [T[F + 1]] and r["rte"] == 2


# ── the storm ─────────────────────────────────────────────────────────────────────────────────
def test_thousand_faults_in_virtual_time(mesh_factory):
    """10^3 injected faults (kill / hang / partition / malformed, random hop or leg, before or
    mid-flight) on a 5-hop mixed fleet with leg delays: every one surfaces as its expected
    failure, every reply ever received matches the oracle, the mesh always recovers, < 5 min."""
    m = mesh_factory(5, legs_ms=[4.0, 9.0, 2.0, 15.0, 6.0, 3.0])
    t0 = time.monotonic()
    rep = fault_storm(m, 1000, seed=2026)
    wall = time.monotonic() - t0
    print(f"\n10^3 faults: {wall:.1f} s wall, {m.clock.now():.1f} s virtual, "
          f"{rep['replies_checked']} replies checked, outcomes {rep['outcomes']}")
    assert rep["violations"] == [], rep["violations"][:10]
    assert all(sum(v.values()) >= 200 for v in rep["outcomes"].values()), rep["outcomes"]
    assert rep["outcomes"]["hang"].get("Deadline", 0) == sum(rep["outcomes"]["hang"].values())
    assert rep["outcomes"]["kill"].get("LinkDown", 0) > 0
    assert rep["outcomes"]["kill"].get("HopError:next_down", 0) > 0
    assert m.clock.now() > 1000 * 0.5             # the deadlines really elapsed, virtually
    assert wall < 300


def test_fault_storm_on_the_real_clock(mesh_factory):
    m = mesh_factory(3, clock="real", deadline=0.3, cost_scale=0.0, legs_ms=[1.0] * 4)
    rep = fault_storm(m, 16, seed=7)
    assert rep["violations"] == [], rep["violations"]


@pytest.mark.parametrize("clock", CLOCKS)
def test_kills_and_restarts_leak_no_threads(clock):
    """Every crashed node's threads exit, so 10^3 kills cannot exhaust the process."""
    import threading
    base = threading.active_count()
    m = FakeMesh(4, clock=clock, legs_ms=[2.0] * 5, stuck_after=30.0,
                 **({"deadline": 0.3, "cost_scale": 0.0} if clock == "real" else {}))
    try:
        rep = fault_storm(m, 24, seed=5)
        assert rep["violations"] == []
        assert threading.active_count() < base + 8 * m.D + 8        # bounded while running
    finally:
        m.close()
    for _ in range(200):
        if threading.active_count() <= base:
            break
        time.sleep(0.05)
    assert threading.active_count() <= base, [t.name for t in threading.enumerate()]


def test_noop_injectors_are_caught(mesh_factory, monkeypatch):
    """Anti-vacuity: if kill, hang or partition did nothing, the storm must say so."""
    monkeypatch.setattr(FakeMesh, "kill", lambda self, i: None)
    monkeypatch.setattr(FakeMesh, "hang", lambda self, i: None)
    monkeypatch.setattr(FakeMesh, "partition", lambda self, a, b: None)
    m = mesh_factory(3, legs_ms=[2.0] * 4)
    rep = fault_storm(m, 12, seed=1)
    for kind in ("kill", "hang", "partition"):
        assert any(kind in v for v in rep["violations"]), (kind, rep["violations"])


def test_a_hop_with_other_weights_breaks_equality(mesh_factory):
    """Anti-vacuity: the mesh's tokens come from the hops' compute."""
    m = mesh_factory(3)
    m.restart(1, weights=FM.ToyWeights(m.pattern, hidden=FM.HIDDEN, vocab=FM.VOCAB, seed=99))
    m.open(1)
    prompt = [1, 2, 3, 4, 5]
    assert m.generate(1, prompt, 10) != direct_greedy(m.weights, prompt, 10)[5:]
