"""The single-session pipelined-speculation coordinator (the entry side), on shard-frame/2.

Lifted from V4's `coordinate_dspark_pipelined` (engines/deepseek_v4/v4_pipe.py:1761-2281): the
state shared with the sender (`PipeSpecState`, :1761-1777, as-is but for the `epoch` -> `sep`
rename of ARCHITECTURE-v2 §0.3), the sender-side fence (`FrameSender`, :1780-1832), the feed
(:1985-2028) and the receiver (:2060-2281). The round logic is V4's; what changed:

  * Frames are shard-frame/2 (shard/frame.py) instead of V4 step dicts: every frame carries
    (sid, rte, seq, sep, pos, n, kind, cpos) and every reply must echo (sid, rte, seq, sep, pos, n)
    exactly. V4 matched on (pos, epoch) with defaults (v4_pipe.py:2081).
  * A frame holds n >= 1 rows. `FramePolicy` decides how a burst (the tokens one reply streams) is
    framed: `pertoken` (n = 1 each, V4's shape) or `chunk` (one frame). One accept rule judges
    every n (pspec/accept.py).
  * Blocks come from the tail's drafter (reply `blk`, V4) or from an entry-side drafter (a
    callable over the committed ids: M2.5's n-gram/EAGLE shape, called only when a block is used).
  * The state machine (`Session`) does no I/O and owns no thread; `run` drives one session over a
    `Link`, and a multi-session loop (S3) can drive many sessions through one shared sender.
  * The sender stamps nothing: seq/sep/cpos are fixed when the frame is built and validated.
  * Not lifted (V4 measurement or V4-only levers): the lazy-draft hints, runner-up/rescue and
    per-depth counters, topup agreement counters, fill time-integration, lever audit.

FRONTIER AND cpos. `c` = committed token count (prompt included); f = c-1 is the position of the
newest committed token. Every frame is built at pos >= f and carries cpos = f, the final-row
frontier F of ARCHITECTURE C18: rows < F were computed on committed inputs, the row at F may
still hold a draft that the correction re-feeds. The deepest rewind this coordinator can ever ask
for is a correction at some f' > f, so declaring rows < f final is safe even for a stage that
applies cpos the moment a frame arrives (before its seek). V4 sent f-1 (v4_pipe.py:2022), one row
more conservative than its own stage needs.

cpos IS ALSO THE ROLLBACK FLOOR (runtime/api.py `commit`): an entry that may be re-adopted from the
gateway's last checkpoint (ARCHITECTURE §4.3, `truncate_to = len(ckpt_ids) - 1`) must not let
stages commit past it. `Session(acked=...)` / `ack(count)` cap cpos at acked-1 (count = committed
tokens the gateway holds, prompt included); None = no re-adoption to protect. A lagging cpos
shrinks the speculation: nothing is streamed past cpos + W - 1, and a lag that leaves no room for
the frontier frame itself (f > cpos + W - 1) raises rather than send a frame a stage must refuse.
Whether the lease W absorbs the checkpoint lag or cpos waits for a synchronous ack is M2's call.

IN-FLIGHT CAP W. A burst streams [m] + block[:W-1] from f, so positions in flight never exceed
f..f+W-1 and every frame's last row ends at or before cpos + W: the stage's spec window. Nothing is
streamed past the last row the job can use either (row c-1 + tokens still to generate - 1), so a
lease of max_pos = prompt + max_new - 1 rows always suffices.

REFILL FLOOR. A reply's block is consumed when fewer than `floor`+1 positions are in flight
(floor 1 = V4's shipped drain-only round). A chunk burst is judged whole by one reply, so under
`chunk` only one frame is ever in flight and the floor has nothing to top up.

M2.5 AT DEPTH 1 IS `chunk` + an entry drafter + floor 1 + W >= K+1: after the prefill reply the
drafter proposes K tokens for the committed ids and one frame [cur] + drafts goes out at cur's
position; its reply commits `plan_verify_round(drafts, toks)` (the full-accept bonus included);
the next frame is [last committed] + a fresh block at that token's position, which is also the
rewind. That is m25_pipe.coordinate_pipe's depth-1 loop (:937-1008) frame for frame, given a
drafter that proposes from the committed stream: tests/test_pspec_coord.py drives the real
coordinate_pipe and this loop over the same oracle and proposals and compares their wire frames. Two presentation differences are left to the M2.5 adapter (S4): M2.5
strips the EOS token from its output and does not clip a round that crosses max_new; this loop
emits through the first EOS and clips at max_new. One drafter-timing difference: M2.5 requests the
next block while the chunk is in flight (`d_request(dprefix + ds)`), before the full-accept bonus
exists, so a drafter that proposes for its request snapshot (NgramDrafter) drafts one position
behind after every full accept at depth 1 (pinned in the tests: a perfect proposer's frames
differ). EAGLE, M2.5's actual depth-1 drafter, drafts from the context it extends over committed
positions (phase0/eagle_draft.py:302), which is the contract this loop assumes: it asks the drafter
after the commit. EAGLE also needs the reply's aux rows, which shard-frame/2 replies do not carry
yet (design §3.3 `ctx`); that and the drafter's ctx input are S4/D1's, not this loop's.
"""
import collections
import queue
import threading
from array import array
from typing import Protocol

from .. import frame as F
from .accept import REJECT, RUN, accept

POLICIES = ("chunk", "pertoken")


class Link(Protocol):
    """The seam between the coordinator and whatever carries its frames.

    `send` is called from the sender thread only, `recv` from the receiver only. Messages are
    shard-frame/2 dicts: `send` gets a validated frame, `recv` returns a decoded REPLY (op "r"):
    control messages on the same stream (a hop's `ckpt`, C19) are the Link's to route elsewhere.
    A dead link raises ConnectionError (shard.frame.FrameError is one) or a timeout (OSError)."""

    def send(self, msg: dict) -> None: ...

    def recv(self, timeout: float) -> dict: ...


class SessionFailed(ConnectionError):
    """Raised by a Link's `recv` when the network refused or lost ONE session's frame (a hop's
    `ack{seq, code}`, an undeliverable frame): `run_many` fails that session and goes on with
    the others; `run` (one session) lets it propagate like any dead link."""

    def __init__(self, sid, why):
        super().__init__(f"pspec: session {sid}: {why}")
        self.sid, self.why = sid, why


class SocketLink:
    """Link over two sockets: frames to the head (`pipe`), replies from the tail (`ret`). This is
    the adapter for the node server and the fake mesh: both terminate the entry's streams on
    sockets and speak shard-frame/2. Control messages arriving on `ret` go to `on_control`; with
    none set, one is a protocol error (ConnectionError)."""

    def __init__(self, pipe, ret, on_control=None):
        self.pipe, self.ret, self.on_control = pipe, ret, on_control

    def send(self, msg):
        F.send(self.pipe, msg)

    def recv(self, timeout):
        self.ret.settimeout(timeout)
        while True:
            msg = F.recv(self.ret)
            if msg["op"] == "r":
                return msg
            if msg["op"] != "c" or self.on_control is None:
                raise ConnectionError(f"pspec: op {msg['op']!r} on the return stream with no "
                                      f"control handler")
            self.on_control(msg)


class PipeSpecState:
    """The only state the sender thread and the receiver share, and the lock over it.

    `sep` is written by the receiver (on a cancel, and on stop) and read by the sender (to drop
    frames of a cancelled speculation). `outstanding` is appended by the sender at the moment a
    frame really goes on the wire and popped by the receiver as replies arrive, so it is the exact
    FIFO of frames the ring owes an answer for. `pending` counts frames handed to the sender and not
    yet accounted for by EITHER path (replied, or dropped unsent). `unsent` counts the drops."""

    def __init__(self):
        self.lock = threading.Lock()
        self.sep = 0
        self.outstanding = collections.deque()               # (seq, pos, n, sep) per frame on the wire
        self.pending = 0
        self.unsent = 0


class FrameSender(threading.Thread):
    """Push frames at the head without ever blocking the receiver: the accept loop must never be
    parked in a socket write while a reply is waiting to be read.

    It is also the SENDER-SIDE half of the sep fence: a cancel fires while a whole block may still
    be queued here, and those frames must not reach the wire. The check and the `outstanding`
    append happen under the session's lock, so a frame is recorded as owed a reply exactly when it
    is about to be sent. Each queued item names its session's state, so one sender can serve every
    session on a link."""

    def __init__(self, link):
        super().__init__(daemon=True, name="pspec-sender")
        self.link = link
        self.q = queue.Queue()
        self.err = None
        self._stopped = False

    def put(self, st, msg):
        with st.lock:
            st.pending += 1
        self.q.put((st, msg))

    def stop(self):
        """Drain and join. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        self.q.put(None)
        self.join(timeout=10)

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            st, msg = item
            with st.lock:
                if msg["sep"] != st.sep:                     # fenced: a cancelled block never ships
                    st.pending -= 1
                    st.unsent += 1
                    continue
                st.outstanding.append((msg["seq"], msg["pos"], msg["n"], msg["sep"]))
            try:
                self.link.send(msg)
            except Exception as e:                           # noqa: BLE001 - the receiver reports it
                self.err = e
                return


def _ints(t):
    """A decoded i32 Tensor (or None) -> list of python ints."""
    if t is None:
        return []
    a = array("i")
    a.frombytes(t.data)
    return list(a)


def _i32(vals):
    return F.Tensor("i32", (len(vals),), array("i", [int(v) for v in vals]))


class Session:
    """One generation stream's speculation state machine. No I/O, no threads.

    `prefill()` returns the first frame; `on_reply(rep)` consumes one reply (in arrival order) and
    returns the frames to stream next, already stamped with their sep. The caller hands frames to
    a `FrameSender` built over the same `st` and must deliver replies in the order the link
    returns them. `stop` turns True once EOS or max_new is committed; then every later reply is
    drained, never judged, and the caller is done when `owed()` reaches 0.

    `drafter`: None = blocks come from the tail's drafter (the reply's `blk`, V4) and every reply
    must carry `acc`, because that drafter advances on the tail's own frontier and the two ends
    have to agree. Otherwise a callable `drafter(committed_ids) -> [tokens]` proposing the tokens
    for positions len(ids), len(ids)+1, ... (an entry-side drafter); `acc` is then optional and
    cross-checked when present. `seal(plaintext) -> bytes` seals the `eid` (the frame's ids, for
    the tail); None sends no eid, which leaves the tail unable to judge `acc`. `acked`: committed
    tokens (prompt included) the gateway holds, which caps cpos (see the module docstring); None
    when there is no re-adoption to protect. Advance it with `ack`."""

    def __init__(self, prompt_ids, max_new, *, sid, rte=0, W=16, floor=1, policy="chunk",
                 eos_ids=(), drafter=None, seal=None, on_token=None, acked=None):
        prompt = [int(t) for t in prompt_ids]
        if not 1 <= len(prompt) <= F.MAX_ROWS:
            raise ValueError(f"pspec: a prompt of {len(prompt)} ids does not fit one prefill frame "
                             f"(1..{F.MAX_ROWS})")
        if max_new < 1:
            raise ValueError("pspec: max_new must be >= 1 (the prefill always yields a token)")
        if W < 1:
            raise ValueError(f"pspec: in-flight cap W={W}; the frontier frame alone needs 1")
        if floor < 1:
            raise ValueError(f"pspec: refill floor {floor}; the pipe refills at or below the floor, "
                             f"so it must be at least 1 (1 = drain-only)")
        if policy not in POLICIES:
            raise ValueError(f"pspec: frame policy {policy!r} not in {POLICIES}")
        if acked is not None and acked < len(prompt):
            raise ValueError(f"pspec: acked={acked} below the prompt ({len(prompt)} ids): the "
                             f"gateway holds the prompt it sent")
        self.sid, self.rte, self.W, self.floor, self.policy = sid, int(rte), int(W), int(floor), policy
        self.max_new, self.eos = int(max_new), {int(t) for t in eos_ids}
        self.drafter, self.seal, self.on_token = drafter, seal, on_token
        self.require_acc = drafter is None
        self.acked = None if acked is None else int(acked)
        self.st = PipeSpecState()
        self.prompt = prompt
        self.ids = list(prompt)                              # committed stream, prompt included
        self.out = []                                        # generated tokens
        self.sent = {}                                       # pos -> token streamed there, this sep
        self.horizon = -1                                    # highest position streamed, this sep
        self.stop = False
        self._prefilled = False
        self._seq = 0
        self.frames = self.rows = self.drafted = self.issued = self.topups = 0
        self.accepted = self.cancels = self.stale = self.run = 0
        self.hist, self.depths = {}, []
        self.trace = []                                      # (pos, tokens) per burst, for tests

    # ── frames ───────────────────────────────────────────────────────────────────────────────────
    def ack(self, count):
        """The gateway now holds `count` committed tokens (prompt included). Monotone."""
        if self.acked is None:
            raise RuntimeError("pspec: ack() on a session built without acked=; its cpos was never "
                               "capped, so stages may already be committed past any checkpoint")
        self.acked = max(self.acked, int(count))

    def _cpos(self):
        f = len(self.ids) - 1
        return f if self.acked is None else min(f, self.acked - 1)

    def _room(self, f):
        """Draft positions a burst may stream past the frontier f: none past the last row the job
        can use, none past the stage's spec window (cpos + W - 1). Negative: not even the
        frontier frame fits (the gateway's ack lags by W or more)."""
        return min(self.max_new - len(self.out) - 1, self._cpos() + self.W - 1 - f)

    def _frontier(self, f):
        room = self._room(f)
        if room < 0:
            raise RuntimeError(
                f"pspec: the frontier frame at {f} does not fit the spec window W={self.W} past cpos "
                f"{self._cpos()}: the gateway's ack lags the committed stream by W or more")
        return room

    def _frame(self, pos, toks, kind, lg):
        """Build and validate one frame; the seq is spent even if the sender later drops it."""
        seq, self._seq = self._seq, self._seq + 1
        eid = None if self.seal is None else self.seal(F.eid_pack(toks))
        return F.frame(sid=self.sid, rte=self.rte, seq=seq, sep=self.st.sep, pos=pos, n=len(toks),
                       kind=kind, cpos=self._cpos(), lg=lg, cx="ids", x=_i32(toks), eid=eid)

    def prefill(self):
        """The prompt as one frame at position 0; its reply returns the model's token at P."""
        if self._seq:
            raise RuntimeError("pspec: prefill() twice")
        return self._frame(0, self.prompt, "P", "last")

    def _stream(self, run):
        """Frame a contiguous run of (pos, token) under the policy and record it as streamed."""
        if not run:
            return []
        self.trace.append((run[0][0], [t for _, t in run]))
        f = len(self.ids) - 1
        groups = [run] if self.policy == "chunk" else [[r] for r in run]
        out = []
        for g in groups:
            pos, toks = g[0][0], [t for _, t in g]
            kind = "T" if pos + len(toks) - 1 <= f else "S"          # T: every row is committed
            out.append(self._frame(pos, toks, kind, "all"))
            for p, t in g:
                self.sent[p] = t
            self.horizon = max(self.horizon, g[-1][0])
            self.frames += 1
            self.rows += len(toks)
        return out

    # ── the receiver ─────────────────────────────────────────────────────────────────────────────
    def owed(self):
        """Replies the caller must still wait for. While generating: every frame handed to the
        sender and not yet accounted for. After stop: only frames already on the wire, because the
        stop bumped the sep, so any frame the sender has not reached yet is dropped unsent."""
        with self.st.lock:
            return len(self.st.outstanding) if self.stop else self.st.pending

    def cancel(self):
        """Stop generating now (a client cancel): nothing more is streamed, every queued frame is
        dropped before the wire, and the replies still owed are drained, never judged. The tokens
        committed so far stay a prefix of the uncancelled stream. Idempotent."""
        self.stop = True
        self.fence()

    def fence(self):
        """Bump the sep: every queued frame of the current speculation is dropped before the wire."""
        with self.st.lock:
            self.st.sep += 1

    def _commit(self, toks):
        """Append committed tokens until EOS or max_new. -> how many were committed."""
        k = 0
        for t in toks:
            self.ids.append(t)
            self.out.append(t)
            k += 1
            if self.on_token is not None:
                self.on_token(t)
            if t in self.eos or len(self.out) >= self.max_new:
                self.stop = True
                break
        return k

    def _check_acc(self, rep, pos):
        acc = rep["acc"]
        if acc is None:
            if self.require_acc:
                raise RuntimeError(
                    f"pspec: the reply for the frame at {pos} carries no `acc`, but blocks come from "
                    f"the tail's drafter, which advances on the tail's own frontier. The tail is not "
                    f"judging it (not drafting, or no eid reached it)")
        elif not acc:
            raise RuntimeError(
                f"pspec: the tail judged the frame at {pos} speculative and this coordinator judged "
                f"it committed, off the same frame and the same replies. The two ends of a lossless "
                f"round have diverged, so the drafter is advancing over a history the ring is not "
                f"taking. One accept rule, and it is plan_verify_round.")

    def on_reply(self, rep):
        if rep.get("op") != "r":                             # before any state moves
            raise RuntimeError(f"pspec: a {rep.get('op')!r} message where a reply belongs: the Link "
                               f"must route control messages elsewhere")
        with self.st.lock:
            if not self.st.outstanding:
                raise RuntimeError("pspec: a reply arrived for no frame on the wire")
            seq, pos, n, sep = self.st.outstanding.popleft()
            self.st.pending -= 1
        got = (rep["sid"], rep["rte"], rep["seq"], rep["sep"], rep["pos"], rep["n"])
        if got != (self.sid, self.rte, seq, sep, pos, n):
            raise RuntimeError(
                f"pspec: reply (sid, rte, seq, sep, pos, n) = {got} for the frame "
                f"{(self.sid, self.rte, seq, sep, pos, n)}: the return channel reordered or dropped "
                f"a frame, and every accept after it would be attributed to the wrong position")
        if rep["fen"] or sep != self.st.sep:                 # THE FENCE: a cancelled future's reply
            self.stale += 1
            return []
        if self.stop:                                        # draining after EOS/max_new
            return []
        toks = _ints(rep["toks"])
        if not self._prefilled:
            return self._on_prefill(rep, toks)
        return self._on_decode(rep, pos, n, toks)

    def _on_prefill(self, rep, toks):
        self._check_acc(rep, 0)
        if len(toks) != 1:
            raise RuntimeError(f"pspec: a prefill reply carries {len(toks)} tokens, want 1 (lg=last)")
        self._prefilled = True
        self._commit(toks)
        if self.stop:
            self.fence()
            return []
        if self.drafter is None:
            self.issued += bool(rep["blk"] is not None)
        f = len(self.ids) - 1
        room = self._frontier(f)
        blk = self._block(rep) if room > 0 else []
        burst = [(f, self.ids[f])] + [(f + 1 + i, d) for i, d in enumerate(blk[:room])]
        if blk:
            self.drafted += 1
        return self._stream(burst)

    def _block(self, rep):
        """The block for positions f+1.. : the reply's (tail drafter) or the entry drafter's."""
        if self.drafter is None:
            return _ints(rep["blk"])
        self.issued += 1
        return [int(t) for t in self.drafter(list(self.ids))]

    def _on_decode(self, rep, pos, n, toks):
        f = len(self.ids) - 1
        # A reply is only BELIEVED when its frame was on the committed path: it starts at the
        # frontier and fed the committed token there. Two independent ways to know, and both must
        # say yes: this loop's record of what it streamed, and the tail's `acc`.
        if pos != f:
            raise RuntimeError(f"pspec: reply for position {pos} with the committed frontier at {f}: "
                               f"replies must land in committed order")
        if self.sent.get(pos) != self.ids[pos]:
            raise RuntimeError(
                f"pspec: about to judge the frame at {pos}, which fed {self.sent.get(pos)!r} while the "
                f"committed token there is {self.ids[pos]!r}: a fenced frame reached the accept path")
        self._check_acc(rep, pos)
        if len(toks) != n:
            raise RuntimeError(f"pspec: the reply for {n} rows carries {len(toks)} tokens (lg=all)")
        fed = [self.sent.get(pos + j + 1) for j in range(n)]
        committed, verdict = accept(toks, fed)
        k = self._commit(committed)
        hits = sum(1 for j in range(k) if fed[j] == committed[j])
        self.accepted += hits
        self.run += hits
        if self.stop:
            self.fence()                                     # nothing more reaches the ring
            return []
        if self.drafter is None:
            self.issued += bool(rep["blk"] is not None)
        f = len(self.ids) - 1
        m = self.ids[f]
        burst = []
        if verdict == REJECT:                                # cancel: fence, rewind, correct
            self.cancels += 1
            self.hist[self.run] = self.hist.get(self.run, 0) + 1
            self.run = 0
            self.fence()
            for p in [p for p in self.sent if p >= f]:       # the discarded future
                del self.sent[p]
            self.horizon = f - 1
        room = self._room(f)
        if verdict != RUN:                                   # the frontier frame: pos f IS the rewind
            room = self._frontier(f)
            burst.append((f, m))
        top = f if verdict != RUN else self.horizon          # deepest position in flight after it
        if top - f + 1 <= self.floor and room > 0:           # at or under the floor: consume a block
            blk = self._block(rep)
            if blk:
                self.drafted += 1
                self.topups += top > f
                for i, d in enumerate(blk[:room]):
                    p = f + 1 + i
                    if p > top:                              # a frame on the wire is never re-streamed
                        burst.append((p, d))
        frames = self._stream(burst)
        self.depths.append(self.horizon - f + 1)
        for p in [p for p in self.sent if p < f]:            # settled: never read again
            del self.sent[p]
        return frames

    def result(self):
        """V4's result keys. `g` is tokens per cycle (cycles = cancels + 1), as V4 counts it; under
        `chunk` every rejection is a cycle, so tokens per round there is generated / frames."""
        hist = dict(self.hist)                               # the run still open is a cycle too
        hist[self.run] = hist.get(self.run, 0) + 1
        gen = len(self.out)
        cycles = self.cancels + 1
        return {"ok": True, "tokens": list(self.out), "prompt_tokens": len(self.prompt),
                "generated": gen, "policy": self.policy, "W": self.W, "floor": self.floor,
                "frames": self.frames, "rows": self.rows, "drafted": self.drafted,
                "drafts_issued": self.issued, "topups": self.topups, "accepted": self.accepted,
                "cancels": self.cancels, "cycles": cycles, "g": gen / cycles,
                "accept_hist": hist,
                "max_inflight": max(self.depths) if self.depths else 0,
                "mean_inflight": round(sum(self.depths) / len(self.depths), 2) if self.depths else 0.0,
                "frames_per_token": round(self.frames / gen, 3) if gen else 0.0,
                "stale_replies": self.stale, "unsent_frames": self.st.unsent}


def run(link, session, timeout=600.0):
    """Drive one Session over a Link to completion. -> session.result().

    One sender thread per call. The receiver waits while `session.owed()` is non-zero: a live
    pipeline always owes a reply (a cancel streams its correction in the same breath), so owing
    nothing before the stop is a bug and raises instead of hanging."""
    sender = FrameSender(link)
    sender.start()
    try:
        sender.put(session.st, session.prefill())
        while True:
            if sender.err is not None:
                raise RuntimeError(f"pspec: the frame sender died: {type(sender.err).__name__}: "
                                   f"{sender.err}") from sender.err
            if session.owed() == 0:
                if session.stop:
                    break
                raise RuntimeError(
                    f"pspec: nothing in flight at position {len(session.ids) - 1} with "
                    f"{len(session.out)} of {session.max_new} tokens generated: the pipeline "
                    f"emptied without stopping")
            try:
                rep = link.recv(timeout)
            except Exception:
                if sender.err is not None:
                    raise RuntimeError(f"pspec: the frame sender died: {type(sender.err).__name__}: "
                                       f"{sender.err}") from sender.err
                raise
            for fr in session.on_reply(rep):
                sender.put(session.st, fr)
    finally:
        session.fence()                                      # drop anything still queued
        sender.stop()
    if sender.err is not None:
        raise RuntimeError(f"pspec: the frame sender died: {type(sender.err).__name__}: "
                           f"{sender.err}") from sender.err
    return session.result()


def run_many(link, sessions, timeout=600.0, on_reply=None):
    """Drive many Sessions over ONE Link and ONE FrameSender to completion (the S3 entry loop).
    -> {sid: session.result()}.

    Replies are demultiplexed on `sid`; each session sees its own replies in the order the link
    returns them, which is all `Session.on_reply` needs (the per-session FIFO holds end to end:
    one sender, and every hop keeps a FIFO per session). A reply for no session here, or for a
    session that owes nothing, is a protocol error. `on_reply(session, reply)` (optional) runs
    before each reply is consumed: the seam a caller uses to cancel a session (`cancel()`) or
    advance its ack. A session is done when it stopped and owes nothing; one that owes nothing
    before stopping is a bug and raises, as in `run`.

    ISOLATION. A `SessionFailed` from the link fails THAT session only: it is cancelled (its
    queued frames never ship), later replies for it are dropped, and its result carries
    ok=False and the reason; the caller closes its lease. Any other link error ends the loop."""
    by_sid = {}
    for s in sessions:
        if s.sid in by_sid:
            raise ValueError(f"pspec: two sessions share sid {s.sid}")
        by_sid[s.sid] = s
    sender = FrameSender(link)
    sender.start()

    def died():
        return RuntimeError(f"pspec: the frame sender died: {type(sender.err).__name__}: "
                            f"{sender.err}")

    try:
        for s in sessions:
            sender.put(s.st, s.prefill())
        live, failed = set(by_sid), {}
        while True:
            if sender.err is not None:
                raise died() from sender.err
            for sid in list(live):
                s = by_sid[sid]
                if s.owed() == 0:
                    if not s.stop:
                        raise RuntimeError(
                            f"pspec: session {sid}: nothing in flight at position {len(s.ids) - 1} "
                            f"with {len(s.out)} of {s.max_new} tokens generated")
                    live.discard(sid)
            if not live:
                break
            try:
                rep = link.recv(timeout)
            except SessionFailed as e:
                if e.sid in live:
                    failed[e.sid] = e.why
                    by_sid[e.sid].cancel()
                    live.discard(e.sid)
                elif e.sid not in by_sid:
                    raise
                continue
            except Exception:
                if sender.err is not None:
                    raise died() from sender.err
                raise
            if rep.get("sid") in failed:                     # a failed session's late reply
                continue
            s = by_sid.get(rep.get("sid"))
            if s is None or rep["sid"] not in live:
                raise RuntimeError(f"pspec: a reply for session {rep.get('sid')!r}, which has "
                                   f"nothing on the wire here")
            if on_reply is not None:
                on_reply(s, rep)
            for fr in s.on_reply(rep):
                sender.put(s.st, fr)
    finally:
        for s in sessions:
            s.fence()
        sender.stop()
    if sender.err is not None:
        raise died() from sender.err
    out = {sid: s.result() for sid, s in by_sid.items()}
    for sid, why in failed.items():
        out[sid].update(ok=False, error=why)
    return out
