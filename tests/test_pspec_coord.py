"""shard/pspec/coord.py: the single-session pipelined-speculation coordinator (BUILD-PLAN S1 gate).

Driven over tests/scripted_pipe.py: D in-process stages with the oracle model, every message as
canonical shard-frame/2 bytes, stages that refuse gaps, rewinds below the committed frontier and
frames past the spec window, and a tail that judges `acc` with the shipped Frontier and refuses a
cpos past the frontier it has committed. Losslessness is `tokens == toy_greedy(...)` to the token;
an off-path row answers a GARBAGE token no greedy stream contains.

  1. the gate matrix: zero / full / partial / random acceptance x frame policy x drafter location
     x W x refill floor x pipeline depth, and with real per-leg latency
  2. EOS and max_new anywhere in a block, and a run that ends at the prefill
  3. W truncation and the refill floor F in {1, B}
  4. the per-token stream equals the chunk stream (tokens and every burst)
  5. cpos never exceeds c-1, and is tight
  6. the seam: echo checks, acc cross-check, SocketLink over real sockets, the sender-side fence
  7. V4's §7c scenarios (tests/test_v4_pipe.py:826-1023) re-expressed on v2
  8. differential: the real V4 coordinator and the real M2.5 coordinate_pipe, same oracle, same
     drafts, compared result for result and frame for frame
"""
import socket
import threading

import pytest

from scripted_pipe import (SID, ScriptedPipeline, alpha_miss, entry_drafter,
                           run_pspec, tail_blocks, toy_greedy, truth)

from shard import frame as F
from shard.pspec import coord

T = truth(600)
P = 7
K = 6                                             # the drafter's block: B

ACCEPT = {
    "full": None,
    "zero": alpha_miss(0.0),
    "partial": lambda p, d: d >= 3,               # right at depth 1-2, wrong from depth 3
    "random": alpha_miss(0.75, seed=11),
}


def _src(src, accept, k=K, t=T):
    miss = ACCEPT[accept] if isinstance(accept, str) else accept
    return {"blocks": tail_blocks(t, k, miss)} if src == "tail" else {"drafter": entry_drafter(t, k, miss)}


# ── 1. the gate matrix ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("depth", [1, 5])
@pytest.mark.parametrize("floor", [1, K])
@pytest.mark.parametrize("W", [1, 3, K + 1, 16])
@pytest.mark.parametrize("accept", sorted(ACCEPT))
@pytest.mark.parametrize("src", ["tail", "entry"])
@pytest.mark.parametrize("policy", coord.POLICIES)
def test_committed_tokens_equal_toy_greedy(policy, src, accept, W, floor, depth):
    r, pipe, _ = run_pspec(T, P, 45, depth=depth, W=W, floor=floor, policy=policy, **_src(src, accept))
    assert r["tokens"] == toy_greedy(T, P, 45), "the committed stream left the greedy stream"
    assert r["max_inflight"] <= W and all(f["n"] <= W for f in pipe.log[1:])
    if W == 1:
        assert r["max_inflight"] == 1, "W=1 is plain greedy: the frontier frame alone"
    if accept == "zero" or W == 1:
        assert r["accepted"] == 0
    can_miss = {"full": False, "partial": W - 1 >= 3, "zero": W > 1, "random": W > 1}[accept]
    assert (r["cancels"] > 0) == can_miss
    if accept == "full" and W > 1:
        assert r["accepted"] > 0


@pytest.mark.parametrize("policy", coord.POLICIES)
@pytest.mark.parametrize("src", ["tail", "entry"])
def test_lossless_with_real_leg_latency_and_a_deep_pipe(policy, src):
    """Real concurrency: 6 stages, 1 ms per leg, so frames overlap in the pipe and the sender races
    the receiver's cancels."""
    r, pipe, _ = run_pspec(T, P, 60, depth=6, leg_s=0.001, W=K + 1, policy=policy,
                           **_src(src, "random"))
    assert r["tokens"] == toy_greedy(T, P, 60)
    assert r["cancels"] > 0


# ── 2. EOS and max_new mid-block ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("policy", coord.POLICIES)
@pytest.mark.parametrize("src", ["tail", "entry"])
@pytest.mark.parametrize("at", [1, 2, 3, 5, 8, 13, 21])
def test_eos_anywhere_in_a_block_stops_on_it(policy, src, at):
    eos = (toy_greedy(T, P, at)[-1],)
    want = toy_greedy(T, P, 10 ** 6, eos=eos)
    r, _, _ = run_pspec(T, P, 200, policy=policy, eos=eos, **_src(src, "full"))
    assert r["tokens"] == want and r["tokens"][-1] == eos[0]
    assert eos[0] not in r["tokens"][:-1], "stopped at the first EOS, inclusive"


@pytest.mark.parametrize("policy", coord.POLICIES)
@pytest.mark.parametrize("src", ["tail", "entry"])
def test_max_new_lands_anywhere_in_a_block(policy, src):
    for max_new in range(1, 2 * K + 3):
        r, _, _ = run_pspec(T, P, max_new, policy=policy, **_src(src, "full"))
        assert r["tokens"] == toy_greedy(T, P, max_new), max_new


@pytest.mark.parametrize("policy", coord.POLICIES)
@pytest.mark.parametrize("max_new,eos", [(1, ()), (8, (T[P],))])
def test_a_run_that_ends_at_the_prefill_streams_nothing(policy, max_new, eos):
    r, pipe, _ = run_pspec(T, P, max_new, policy=policy, eos=eos, **_src("entry", "full"))
    assert r["tokens"] == [T[P]] and r["frames"] == 0
    assert [f["pos"] for f in pipe.log] == [0], "only the prefill reached the pipe"


# ── 3. W truncation and the refill floor ───────────────────────────────────────────────────────

@pytest.mark.parametrize("policy", coord.POLICIES)
@pytest.mark.parametrize("W", [2, 3, 5])
def test_W_truncates_a_wider_block(policy, W):
    """A block of 8 against a cap of W: at most W positions in flight, no frame over W rows, and the
    stage's spec window (W past its committed frontier) is never exceeded (it would refuse)."""
    r, pipe, _ = run_pspec(T, P, 40, W=W, floor=8, policy=policy, **_src("tail", "full", k=8))
    assert r["tokens"] == toy_greedy(T, P, 40)
    assert r["max_inflight"] == W
    assert max(f["n"] for f in pipe.log[1:]) == (W if policy == "chunk" else 1)


def test_floor_B_pins_the_per_token_pipe_at_B_plus_one():
    lo, _, _ = run_pspec(T, P, 40, W=16, floor=1, policy="pertoken", **_src("tail", "full"))
    hi, pipe, _ = run_pspec(T, P, 40, W=16, floor=K, policy="pertoken", **_src("tail", "full"))
    assert lo["tokens"] == hi["tokens"] == toy_greedy(T, P, 40)
    assert lo["topups"] == 0 and hi["topups"] > 0
    assert hi["max_inflight"] == K + 1 and hi["mean_inflight"] > lo["mean_inflight"]
    seen = {}
    for f in pipe.log[1:]:                                   # a frame on the wire is never re-streamed
        assert f["pos"] > seen.get(f["sep"], -1), f
        seen[f["sep"]] = f["pos"]


@pytest.mark.parametrize("src", ["tail", "entry"])
@pytest.mark.parametrize("accept", ["partial", "random"])
def test_floor_B_is_lossless_through_rejections(src, accept):
    r, _, _ = run_pspec(T, P, 50, W=16, floor=K, policy="pertoken", **_src(src, accept))
    assert r["tokens"] == toy_greedy(T, P, 50) and r["cancels"] > 0 and r["topups"] > 0


@pytest.mark.parametrize("src", ["tail", "entry"])
def test_the_floor_is_inert_under_chunk(src):
    """A chunk burst is judged whole by one reply, so one frame is in flight and there is nothing to
    top up: floor B streams exactly what floor 1 streams."""
    a, _, sa = run_pspec(T, P, 40, floor=1, policy="chunk", **_src(src, "random"))
    b, _, sb = run_pspec(T, P, 40, floor=K, policy="chunk", **_src(src, "random"))
    assert sa.trace == sb.trace and a["frames"] == b["frames"] and b["topups"] == 0


# ── 4. per-token stream == chunk stream ────────────────────────────────────────────────────────

@pytest.mark.parametrize("src", ["tail", "entry"])
@pytest.mark.parametrize("accept", sorted(ACCEPT))
def test_the_per_token_stream_equals_the_chunk_stream(src, accept):
    """Same drafter, floor 1: both policies commit the same (greedy) tokens; only the framing and the
    refill point differ. Per-token refills on the reply that leaves ONE frame in flight (V4's
    'drained to the frontier'), so after a fully accepted block it streams the next block one frame
    earlier than a chunk, which is judged whole and has nothing in flight after its reply."""
    pt, ppt, _ = run_pspec(T, P, 60, W=K + 1, policy="pertoken", **_src(src, accept))
    ch, pch, sch = run_pspec(T, P, 60, W=K + 1, policy="chunk", **_src(src, accept))
    assert pt["tokens"] == ch["tokens"] == toy_greedy(T, P, 60)
    assert all(f["n"] == 1 for f in ppt.log[1:]) and pt["frames"] == pt["rows"]
    assert ch["frames"] == len(sch.trace), "chunk: one frame per burst"
    assert [(f["pos"], f["toks"]) for f in pch.log[1:]] == sch.trace, "one frame in flight at a time"


# ── 5. cpos ────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("policy", coord.POLICIES)
@pytest.mark.parametrize("floor", [1, K])
def test_cpos_never_exceeds_c_minus_1_and_is_tight(policy, floor):
    """cpos is the final-row frontier F = c-1 (c = committed count). On the wire: it never passes the
    frame's own first row, never moves backwards, a frame AT cpos feeds the committed token there,
    and every speculation epoch opens with a frame at exactly cpos (the watermark does not lag).
    The tail independently refuses any cpos past the frontier it has committed, so every other test
    in this file checks the bound too."""
    r, pipe, _ = run_pspec(T, P, 60, W=K + 1, floor=floor, policy=policy, **_src("tail", "random"))
    stream = T[:P] + r["tokens"]
    assert pipe.log[0]["cpos"] == P - 1
    last, first_of_sep = -1, {}
    for f in pipe.log[1:]:
        assert f["cpos"] <= f["pos"], f
        assert f["cpos"] >= last, "cpos moved backwards"
        last = f["cpos"]
        if f["pos"] == f["cpos"]:
            assert f["toks"][0] == stream[f["pos"]], "a frame at cpos must feed the committed token"
        first_of_sep.setdefault(f["sep"], f)
    assert all(f["pos"] == f["cpos"] for f in first_of_sep.values())
    assert len(first_of_sep) == r["cancels"] + 1


# ── 6. the seam ────────────────────────────────────────────────────────────────────────────────

class _Corrupt:
    """A Link that rewrites one field of the n-th reply (a reordering/dropping return channel)."""

    def __init__(self, inner, nth, **field):
        self.inner, self.nth, self.field, self.k = inner, nth, field, 0

    def send(self, msg):
        self.inner.send(msg)

    def recv(self, timeout):
        rep = self.inner.recv(timeout)
        self.k += 1
        return dict(rep, **self.field) if self.k == self.nth else rep


@pytest.mark.parametrize("field", [{"seq": 999}, {"pos": 999}, {"sep": 7}, {"n": 3}, {"rte": 1},
                                   {"sid": "00000000000000ff"}])
def test_a_reply_that_does_not_echo_its_frame_fails_loudly(field):
    with ScriptedPipeline(T, blocks=tail_blocks(T, K)) as pipe:
        s = coord.Session(T[:P], 30, sid=SID, W=K + 1, policy="pertoken", seal=bytes)
        with pytest.raises(RuntimeError, match="reordered or dropped"):
            coord.run(_Corrupt(pipe, 3, **field), s, timeout=10)


def test_the_tail_disagreeing_about_the_accept_fails_loudly():
    with pytest.raises(RuntimeError, match="two ends of a lossless round have diverged"):
        run_pspec(T, P, 20, lie=False, **_src("tail", "full"))


@pytest.mark.parametrize("why", [{"judge": False}, {"seal": None}])
def test_a_tail_drafter_needs_a_tail_that_judges_acc(why):
    """Blocks from the tail need the tail's frontier: no acc (not judging, or no eid to judge with)
    fails the job at its first reply."""
    with pytest.raises(RuntimeError, match="carries no `acc`"):
        run_pspec(T, P, 20, **why, **_src("tail", "full"))


@pytest.mark.parametrize("why", [{"judge": False}, {"seal": None}])
def test_an_entry_drafter_runs_without_acc(why):
    r, _, _ = run_pspec(T, P, 30, **why, **_src("entry", "random"))
    assert r["tokens"] == toy_greedy(T, P, 30)


def test_over_real_sockets_via_socketlink():
    """The adapter the node server and the fake mesh use: frames and replies as bytes on sockets."""
    a_pipe, b_pipe = socket.socketpair()
    a_ret, b_ret = socket.socketpair()
    done = threading.Event()
    with ScriptedPipeline(T, depth=3, blocks=tail_blocks(T, K, ACCEPT["random"])) as pipe:
        def head():
            try:
                while True:
                    pipe.send(F.recv(b_pipe))
            except (ConnectionError, OSError):
                return

        def tail():
            while not done.is_set():
                try:
                    F.send(b_ret, pipe.recv(0.2))
                except TimeoutError:
                    continue
                except (ConnectionError, OSError):
                    return
        ts = [threading.Thread(target=head, daemon=True), threading.Thread(target=tail, daemon=True)]
        for t in ts:
            t.start()
        try:
            s = coord.Session(T[:P], 50, sid=SID, W=K + 1, policy="pertoken", seal=bytes)
            r = coord.run(coord.SocketLink(a_pipe, a_ret), s, timeout=10)
        finally:
            done.set()
            for x in (a_pipe, b_pipe, a_ret, b_ret):
                x.close()
            for t in ts:
                t.join(5)
    assert r["tokens"] == toy_greedy(T, P, 50) and r["cancels"] > 0


class _Rec:
    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)


def _frame(seq, sep, pos=9):
    return F.frame(sid=SID, rte=0, seq=seq, sep=sep, pos=pos, n=1, kind="S", cpos=pos - 1, lg="all",
                   cx="ids", x=F.Tensor("i32", (1,), (7).to_bytes(4, "little")))


def sender_fence_case(mod):
    """V4's sender-side fence test (test_v4_pipe.py:1501): frames queued for a now-dead sep never
    reach the wire, and dropping one leaves the accounting exact."""
    rec, st = _Rec(), mod.PipeSpecState()
    snd = mod.FrameSender(rec)
    st.sep = 1                                               # the cancel already happened
    snd.put(st, _frame(0, 0))                                # stale: must be dropped
    snd.put(st, _frame(1, 1))                                # live: must be sent
    snd.start()
    snd.stop()
    assert [m["seq"] for m in rec.sent] == [1], "a fenced frame reached the wire"
    with st.lock:
        assert st.pending == 1 and st.unsent == 1 and list(st.outstanding) == [(1, 9, 1, 1)]


def test_sender_side_fence_drops_queued_frames_without_stranding_the_receiver():
    sender_fence_case(coord)


def test_one_sender_serves_several_sessions():
    """Each queued item names its session's state, so a shared sender fences per session."""
    rec, a, b = _Rec(), coord.PipeSpecState(), coord.PipeSpecState()
    snd = coord.FrameSender(rec)
    a.sep = 1
    snd.put(a, _frame(0, 0))
    snd.put(b, _frame(0, 0))
    snd.start()
    snd.stop()
    assert len(rec.sent) == 1 and a.unsent == 1 and b.unsent == 0 and len(b.outstanding) == 1


@pytest.mark.parametrize("prompt,max_new,kw,match", [
    (T[:P], 5, {"W": 0}, "in-flight cap"), (T[:P], 5, {"floor": 0}, "refill floor"),
    (T[:P], 5, {"policy": "tree"}, "frame policy"), (T[:P], 0, {}, "max_new"),
    ([], 5, {}, "prefill frame"), ([1] * (F.MAX_ROWS + 1), 5, {}, "prefill frame")])
def test_session_refuses_bad_arguments(prompt, max_new, kw, match):
    with pytest.raises(ValueError, match=match):
        coord.Session(prompt, max_new, sid=SID, **kw)


# ── 7. V4 §7c, re-expressed on v2 (tests/test_v4_pipe.py:826-1023) ─────────────────────────────
PIPE_PROMPT = [1, 2, 3]
TRUTH = {p: 100 + p for p in range(3, 40)}
VT = PIPE_PROMPT + [TRUTH[p] for p in range(3, 40)]


def _perfect_block(q, n=3):
    return [TRUTH[q + 2 + i] for i in range(n)]


def _pipelined(blocks, max_new, eos=(), **kw):
    kw.setdefault("W", 16)
    return run_pspec(VT, 3, max_new, blocks=blocks, eos=eos, policy="pertoken", **kw)


def test_7c_streams_the_block_as_separate_s1_frames():
    r, pipe, _ = _pipelined(_perfect_block, 6)
    assert r["tokens"] == [TRUTH[p] for p in range(3, 9)]
    assert (pipe.log[0]["toks"], pipe.log[0]["pos"], pipe.log[0]["kind"]) == ([1, 2, 3], 0, "P")
    steps = pipe.log[1:]
    assert all(f["n"] == 1 for f in steps), "a per-token frame is n=1, always"
    assert [f["pos"] for f in steps[:4]] == [3, 4, 5, 6]
    assert [f["toks"][0] for f in steps[:4]] == [TRUTH[p] for p in range(3, 7)]


def test_7c_full_accept_never_cancels_and_fills_the_pipeline():
    r, _, _ = _pipelined(_perfect_block, 10)
    assert r["tokens"] == [TRUTH[p] for p in range(3, 13)]
    assert r["cancels"] == 0 and r["accept_hist"] == {r["accepted"]: 1}
    assert r["max_inflight"] == 4, "block of 3 + the frontier frame in flight"
    assert r["accepted"] >= 6 and r["g"] == r["generated"]
    assert r["stale_replies"] + r["unsent_frames"] > 0, "the post-max_new drain is fenced, not judged"


def test_7c_partial_accept_cancels_at_the_first_divergence():
    r, pipe, _ = _pipelined(lambda q: [_perfect_block(q)[0], 777, 778], 8)
    assert r["tokens"] == [TRUTH[p] for p in range(3, 11)]
    assert r["cancels"] >= 1 and r["stale_replies"] + r["unsent_frames"] >= 1
    assert r["accepted"] >= 1
    seps = sorted({f["sep"] for f in pipe.log})
    assert seps == list(range(r["cancels"] + 1)), "the sep is a dense counter, one bump per cancel"
    first, last = {}, {}
    for f in pipe.log[1:]:
        first.setdefault(f["sep"], (f["toks"][0], f["pos"]))
        last[f["sep"]] = f["pos"]
    for sep in sorted(first)[1:]:
        tok, pos = first[sep]
        assert pos <= last[sep - 1], "the correction opens behind the discarded future (the rewind)"
        assert tok == TRUTH[pos], "a correction frame carries the model's token, never the draft"


def test_7c_zero_accept_degrades_to_greedy_and_stays_lossless():
    r, _, _ = _pipelined(lambda q: [901, 902, 903], 6)
    assert r["tokens"] == [TRUTH[p] for p in range(3, 9)]
    assert r["accepted"] == 0 and r["cancels"] == 3 and r["accept_hist"] == {0: 4}


def test_7c_stops_at_eos_inside_a_block():
    r, _, _ = _pipelined(_perfect_block, 10, eos=(TRUTH[6],))
    assert r["tokens"] == [TRUTH[3], TRUTH[4], TRUTH[5], TRUTH[6]] and r["generated"] == 4


def test_7c_honors_max_new_mid_block():
    r, _, _ = _pipelined(_perfect_block, 3)
    assert r["tokens"] == [TRUTH[3], TRUTH[4], TRUTH[5]]


@pytest.mark.parametrize("max_new,eos", [(1, ()), (8, (TRUTH[3],))])
def test_7c_run_that_ends_at_the_prefill_streams_nothing(max_new, eos):
    r, pipe, _ = _pipelined(_perfect_block, max_new, eos=eos)
    assert r["tokens"] == [TRUTH[3]] and r["frames"] == 0
    assert [f["pos"] for f in pipe.log] == [0]


def test_7c_caps_the_frames_in_flight_at_the_stages_rollback_depth():
    r, _, _ = _pipelined(_perfect_block, 8, W=2)
    assert r["tokens"] == [TRUTH[p] for p in range(3, 11)] and r["max_inflight"] <= 2


def test_7c_fails_loudly_when_the_tail_disagrees_about_the_accept():
    with pytest.raises(RuntimeError, match="two ends of a lossless round have diverged"):
        _pipelined(_perfect_block, 6, lie=False)


def test_7c_needs_a_tail_that_is_actually_drafting():
    with pytest.raises(RuntimeError, match="carries no `acc`"):
        _pipelined(_perfect_block, 4, judge=False)


# ── 8. differential against the real V4 and M2.5 coordinators ──────────────────────────────────

def _v4_signature(r):
    """test_v4_pipe._accept_pattern: everything a pipelined round did that is not a thread race."""
    d = {k: r[k] for k in ("tokens", "frames", "drafted", "generated", "accepted", "cancels",
                           "cycles", "g", "accept_hist", "max_inflight", "mean_inflight")}
    d["fenced"] = r["stale_replies"] + r["unsent_frames"]
    return d


V4_CASES = {
    "perfect": (lambda q: _perfect_block(q), {"max_new": 10}),
    "partial": (lambda q: [_perfect_block(q)[0], 777, 778], {"max_new": 8}),
    "zero": (lambda q: [901, 902, 903], {"max_new": 6}),
    "eos": (lambda q: _perfect_block(q), {"max_new": 10, "eos_ids": (TRUTH[6],)}),
    "max_new": (lambda q: _perfect_block(q), {"max_new": 3}),
    "max_new=1": (lambda q: _perfect_block(q), {"max_new": 1}),
    "W=2": (lambda q: _perfect_block(q), {"max_new": 8, "depth": 2}),
    "W=3 miss": (lambda q: [_perfect_block(q)[0], 9, 9], {"max_new": 12, "depth": 3}),
    "floor=2": (lambda q: _perfect_block(q), {"max_new": 14, "floor": 2}),
    "floor=3": (lambda q: _perfect_block(q), {"max_new": 14, "floor": 3}),
    "floor=3 deep miss": (lambda q: _perfect_block(q)[:2] + [90777], {"max_new": 12, "floor": 3}),
    "floor=3 miss@8": (lambda q: [(424242 if q + 2 + i == 8 else TRUTH[q + 2 + i]) for i in range(3)],
                       {"max_new": 12, "floor": 3}),
    "wide W=6": (lambda q: _perfect_block(q, n=8), {"max_new": 12, "depth": 6}),
    "wide floor=8 miss": (lambda q: [(424242 if i + 1 >= 6 else TRUTH[q + 2 + i]) for i in range(8)],
                          {"max_new": 12, "floor": 8}),
    "shrinking block": (lambda q: _perfect_block(q, n=(1 if q >= 6 else 3)), {"max_new": 12}),
}


@pytest.mark.parametrize("case", sorted(V4_CASES))
def test_v4_differential_same_round_as_coordinate_dspark_pipelined(case):
    """The lifted loop and V4's coordinate_dspark_pipelined, over their own scripted rings with the
    same blocks: every deterministic observable of the round is identical (tokens, frames streamed,
    blocks consumed, accepts, cancels, the accept histogram, the fill, the fenced count)."""
    pytest.importorskip("torch")
    V4T = pytest.importorskip("test_v4_pipe")
    blocks, kw = V4_CASES[case]
    r4, _ = V4T._pipelined(blocks, **kw)
    r2, _, _ = run_pspec(VT, 3, kw["max_new"], W=kw.get("depth", 16), floor=kw.get("floor", 1),
                         eos=kw.get("eos_ids", ()), blocks=blocks, policy="pertoken")
    assert _v4_signature(r2) == _v4_signature(r4)


class _M25Drafter:
    """One proposal function for both loops. M2.5's request/fetch contract, but proposing from the
    latest COMMITTED stream (via on_commit), which is what its depth-1 drafter (EAGLE, drafting off
    its own extended context) does."""

    def __init__(self, prompt, propose):
        self.prompt, self.propose, self.out = list(prompt), propose, []

    def on_commit(self, out, _dt):
        self.out = list(out)

    def request(self, ids, k):
        self.k = k

    def fetch(self):
        return self.propose(self.prompt + self.out)[:self.k]

    def cancel(self):
        pass


@pytest.mark.parametrize("accept", sorted(ACCEPT))
def test_m25_differential_chunk_at_depth_1_is_coordinate_pipe_frame_for_frame(accept):
    """`chunk` + an entry drafter + floor 1 + W = K+1 IS today's M2.5 depth-1 round: the real
    m25_pipe.coordinate_pipe (n-gram path, depth=1) and this loop, over the same oracle with the
    same proposals, put the same frames on the wire ((start, token_ids) == (pos, x)), prefill
    included, and commit the same tokens (M2.5 does not clip the round that crosses max_new)."""
    pytest.importorskip("torch")
    FR = pytest.importorskip("fake_ring")
    if FR.S.M25_EAGLE or FR.S.M25_TREE:
        pytest.skip("M25_EAGLE/M25_TREE set in this environment: not the plain depth-1 loop")
    Tm = FR.novel_T(400, seed=5)
    Pm, Km = 12, 6
    miss = ACCEPT[accept]
    d = _M25Drafter(Tm[:Pm], entry_drafter(Tm, Km, miss))
    res, ring = FR.run_coordinator(Tm, Pm, d, K=Km, depth=1, max_new=60, on_commit=d.on_commit)
    r, pipe, _ = run_pspec(Tm, Pm, 60, W=Km + 1, policy="chunk", drafter=entry_drafter(Tm, Km, miss))
    m25 = [(e["start"], e["token_ids"]) for e in ring.log if e["op"] == "verify"]
    assert m25 == [(f["pos"], f["toks"]) for f in pipe.log]
    assert res["output_ids"][:60] == r["tokens"] == toy_greedy(Tm, Pm, 60)
