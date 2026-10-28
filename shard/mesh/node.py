"""Node server v0 (ARCHITECTURE-v2 §4.2): one layer span, many sessions, shard-frame/2 over links.

A node server owns one StageRuntime (any device: the runtime contract is device-neutral) and
serves its layer span to every session whose lease names it. One thread, the node loop, owns
the runtime and all session state, so every rule below is sequential and there is no locking
of the data plane. Link threads only hand it messages through a clock Mailbox.

Per inbound frame f of session s (§4.2):
  1. lease check: (sid, rte) must match a live lease and f must arrive on a stream from the
     lease's predecessor PeerId. Otherwise it is dropped and counted (an abuse signal).
  2. queue: per-session FIFO. Everything below happens when the batcher reaches f, so it sees
     the state every earlier frame of s left behind.
  3. fence: f.fen or f.sep < s.sep_seen -> forwarded header-only (the tail replies `fen`),
     nothing is computed. Otherwise s.sep_seen = f.sep.
  4. seek, judged against the frontier f.cpos will set: pos > length is a gap, pos below the
     frontier a rollback, pos + n past frontier + W a window refusal. Head token ids are
     bounded by the vocabulary here too. A frame refused by these checks changes nothing (not
     even the frontier) and is reported loudly: `ack{seq, code}` to the lease's `ret`.
     Otherwise commit(cpos) (rows < cpos are final); pos < length is an implicit truncate.
     Content the runtime alone can judge (e.g. hidden values it rejects) is refused after
     the commit: cpos is the entry's statement about committed tokens, not about this
     frame's payload, so the frontier it set stands.
  5. batch: when the loop is free it runs ONE runtime step holding at most one segment per
     session under the runtime's token budget, in the §4.2 priority classes (rebuild, then
     truth/spec frontier-first, then prefill; round-robin ties, no starvation: `_pick`). A frame
     larger than the budget runs alone as consecutive steps and is forwarded whole.
  6. forward: non-tail hops send the output rows to the lease's successor with the same header
     and their load byte appended to `ld`. The tail replies to `ret` with the tokens `lg` asks
     for (none on a prefill sub-chunk that is not `last`).
If the successor is unreachable after a re-dial, the frame is lost and the node sends
`ack{seq, next_down}` to `ret`. A malformed inbound message resets that stream only.

Control (§5.3): lease_open (admission: S_max, load rho < 0.85, span and role, codecs, then the
runtime's own KV / W refusals; answered with `ack` to the sender), lease_update (rte bump,
re-pointing, truncate_to; acked, idempotent on a retransmission) and close, both accepted only
from the lease's `ret` (the entry) until M2 signs updates. Every other command is counted and
ignored (M1 fills `_CONTROL`); an `ack` is never answered, so two nodes cannot ack each other
forever.

Many sessions from day one. S_max (default 16) caps NEW leases and is itself capped at half the
runtime's session table, keeping the other half as takeover headroom (C21).

Not yet, each with a named seam:
  * lease tokens and lease_update signatures are accepted unverified (`_verify_lease`,
    `_verify_update`; M3 / M2);
  * the takeover share (phi >= 0.5 while a takeover is pending) waits for takeover (M1); the
    priority classes themselves are in `_pick`;
  * prefill sub-chunk pipelining across hops (a too-large frame is computed in steps and
    forwarded whole; `sub`/`last` from the entry pass through);
  * activation log, mirroring, takeover, log_fetch, drain, ckpt, stats (M1: `_CONTROL`);
  * the tap lane (a lease with ct=bf16 is refused as unsupported);
  * the eid AEAD under the lease's `ekey` (D1: `_open_eid` is the identity seam) and the lazy-draft
    licence (`hint` passes through; the tail drafts on every committed frame);
  * sampling parameters: lease_open carries only `seed`, so the tail draws greedily.

THE TAIL JUDGES (S2; tests/scripted_pipe.py `_reply` is the spec). A tail (or solo) lease keeps a
`pspec.accept.Frontier`. A frame that carries an `eid` has its ids read with `frame.eid_unpack`;
the tail computes every row's token, judges `acc` with the Frontier (the prefill sets it), and
answers only the rows `lg` asks for. A frame whose cpos claims more than the tail has committed
(cpos > cfront + 1) is refused before anything moves. With a drafter named in the lease (one the
node was built with: `drafters={name: factory(lease) -> blocks(cfront, mfront)}`), every accepted
frame's reply carries `blk`, the proposals for cfront+2.. (V4's convention); a drafter fault
costs that reply its block, never the node. A lease_update's truncate_to below the frontier
re-opens it at F with the token this tail computed there (`Lease.path`), so the entry's re-feed
is judged, not trusted. A solo hop also refuses an eid that is not the ids it computes, and a
second prefill is refused (it would reset the frontier unjudged).

Wire codecs. Hidden rows travel as the runtime's wire dtype: bf16 -> cx=bf16, fp8_e4m3 -> cx=fp8
with per-row `xs`. shard-frame/2 has no integer codec, so a runtime whose wire dtype is i32 (the
integer ToyRuntime) sends its rows bit-exact as bf16 PAIRS: cx=bf16, shape [n, 2H]. The bytes on
the wire (and so every §7.1 wire row) are the i32 rows' bytes.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .. import frame
from ..pspec.accept import Frontier
from ..runtime.api import (KVFull, PositionGap, RollbackPastCommit, Rows, SessionExists, Segment,
                           SpecWindowExceeded, StageRuntimeError, StepIn, WTooDeep)
from .clock import Closed, RealClock, Timeout
from .links import TcpLinks

_ROLE = {(True, False): "head", (False, False): "mid", (False, True): "tail", (True, True): "solo"}
RHO_ADMIT = 0.85          # §4.2: no new lease while the hop's load is at or above this


def _code(e: Exception) -> str:
    """ack code for a runtime refusal of one frame."""
    if isinstance(e, PositionGap):
        return "gap"
    if isinstance(e, RollbackPastCommit):
        return "rollback"
    if isinstance(e, SpecWindowExceeded):
        return "window"
    if isinstance(e, (KVFull, WTooDeep, SessionExists)):
        return "bad"
    if isinstance(e, StageRuntimeError) and isinstance(e, ValueError):     # BadRequest
        return "bad"
    return "fault"


class _Refuse(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass
class Lease:
    """One session's lease on this hop (§5.3 lease_open). M1 hangs the activation log, the
    mirror and the checkpoint state here."""
    sid: int
    hex: str
    rte: int
    role: str
    pred: str
    next: str | None
    ret: str
    backup: str | None
    max_pos: int
    W: int
    cx: str
    seed: int | None
    sep_seen: int = 0
    front: Frontier | None = None        # tail/solo: the pspec accept frontier (module docstring)
    draft: object = None                 # tail/solo: blocks(cfront, mfront) -> [ids], or None
    path: dict = field(default_factory=dict)   # tail/solo: pos -> the model's token there, on
                                               # the committed path (restores a re-opened mfront)


class NodeServer:
    """Serves `runtime`'s layer span to many sessions over `links_factory` links."""

    def __init__(self, runtime, peer_id: str, *, resolve, clock=None, s_max: int = 16,
                 links_factory=TcpLinks, host: str = "127.0.0.1", trace=None, drafters=None):
        self.rt = runtime
        self._drafters = dict(drafters or {})
        self.cap = runtime.describe()
        self.peer_id = peer_id
        self.clock = clock or RealClock()
        self.role = _ROLE[(self.cap.is_head, self.cap.is_tail)]
        self.s_max = max(1, min(s_max, self.cap.max_sessions // 2 or 1))
        self._budget = self.cap.max_tokens_per_step
        self._in_codecs = {"ids"} if self.cap.is_head else self._wire_codecs(self.cap.codecs_in)
        self._out_cx = next(iter(self._wire_codecs((self.cap.wire_dtype,))), "bf16")
        self._trace = trace
        self._inbox = self.clock.mailbox(f"{peer_id}.inbox")
        self.links = links_factory(peer_id, resolve, clock=self.clock, on_message=self._on_msg,
                                   on_send_error=self._on_send_error, host=host)
        self._leases: dict[int, Lease] = {}
        self._fifo: dict[int, deque] = {}
        self._rr: deque = deque()            # sessions with queued frames, round-robin order
        self._skips: dict[int, int] = {}     # consecutive steps a ready session was passed over
        self._queued_rows = 0
        self._dead = False
        self._slock = threading.Lock()
        self.stats: dict[str, int] = {}
        self.addr = None
        self._loop_thread = None

    # ── lifecycle ─────────────────────────────────────────────────────────────────────────
    def start(self) -> tuple[str, int]:
        self.addr = self.links.start()
        self._loop_thread = self.clock.spawn(f"{self.peer_id}.loop", self._loop)
        return self.addr

    def stop(self) -> None:
        """Graceful: finish the current step, drop queued work, close sessions and links."""
        self._inbox.put(("stop",))

    def kill(self) -> None:
        """Crash (the fault harness's kill -9): links reset at once, all state is lost."""
        self._dead = True
        self._inbox.close()
        self.links.close(abrupt=True)

    @property
    def alive(self) -> bool:
        return not self._dead

    def count(self, key: str, k: int = 1) -> None:
        with self._slock:
            self.stats[key] = self.stats.get(key, 0) + k

    def snapshot(self) -> dict:
        with self._slock:
            return dict(self.stats)

    def describe(self) -> dict:
        """This hop as a directory would see it (a subset of the §4.4 holder record)."""
        c = self.rt.describe()
        return {"peer": self.peer_id, "model": c.model, "span": [c.lo, c.hi], "role": self.role,
                "device": c.to_dict()["device"], "numerics_id": c.numerics_id,
                "service_ms": [list(e) for e in c.service_ms], "kv_free": c.kv_free_bytes,
                "sessions": len(self._leases), "s_max": self.s_max, "rho": self._rho(),
                "queue": self._queued_rows}

    # ── link callbacks (link threads) ─────────────────────────────────────────────────────
    def _on_msg(self, peer: str, msg: dict) -> None:
        self._inbox.put(("msg", peer, msg, self.clock.now()))

    def _on_send_error(self, peer: str, meta: dict, err) -> None:
        self._inbox.put(("send_failed", peer, meta))

    # ── the node loop ─────────────────────────────────────────────────────────────────────
    def _pause(self) -> None:
        """Harness seam (a hang injector parks the loop here). No-op in production."""

    def _loop(self) -> None:
        try:
            while not self._dead:
                try:
                    ev = self._inbox.get(timeout=0 if self._rr else None)
                except Timeout:
                    ev = None
                while ev is not None and not self._dead:
                    self._pause()
                    self._handle(ev)
                    try:
                        ev = self._inbox.get(timeout=0)
                    except Timeout:
                        ev = None
                if self._rr and not self._dead:
                    self._pause()
                    self._step_once()
        except Closed:
            pass
        finally:
            if not self._dead:              # graceful stop: release every session
                for sid in list(self._leases):
                    self._drop_session(sid)
                self._dead = True
                self.links.close()

    def _handle(self, ev) -> None:
        kind = ev[0]
        if kind == "msg":
            _, peer, m, t = ev
            if m["op"] == "f":
                self._on_frame(peer, m, t)
            elif m["op"] == "c":
                self._on_control(peer, m)
            else:
                self.count("unexpected")
        elif kind == "send_failed":
            self._on_send_failed(ev[1], ev[2])
        elif kind == "stop":
            raise Closed("stop")

    # ── control ───────────────────────────────────────────────────────────────────────────
    _CONTROL = {"lease_open": "_lease_open", "lease_update": "_lease_update", "close": "_close"}

    def _on_control(self, peer: str, m: dict) -> None:
        name = self._CONTROL.get(m["cmd"])
        if name is None:
            self.count(f"ctl_{m['cmd']}_ignored")
            # Only a lease_open-shaped request earns an answer; an `ack` (or anything a peer
            # sends as a report) is never answered, or two nodes could ack each other forever.
            if m["cmd"] == "takeover":
                self._ack(peer, m["sid"], m["rte"], "unsupported")
            return
        getattr(self, name)(peer, m)

    def _verify_lease(self, m: dict) -> bool:
        """M3 seam: check the orchestrator-signed lease token. v0 accepts any token."""
        return True

    def _verify_update(self, m: dict) -> bool:
        """M2 seam: check the entry's signature over frame.signing_bytes(m). v0 accepts."""
        return True

    def _admit(self, sid: int, m: dict) -> str:
        if sid in self._leases:
            return "exists"
        if (m["lo"], m["hi"]) != (self.cap.lo, self.cap.hi) or m["role"] != self.role:
            return "span"
        if m["ct"] != "none":
            return "unsupported"
        if m["drafter"] is not None and (m["drafter"] not in self._drafters
                                         or m["role"] not in ("tail", "solo")):
            return "unsupported"
        if m["cx"] not in self._in_codecs:
            return "unsupported"
        if len(self._leases) >= self.s_max or self._rho() >= RHO_ADMIT:
            return "busy"
        if not self._verify_lease(m):
            return "bad"
        return "ok"

    def _lease_open(self, peer: str, m: dict) -> None:
        sid = int(m["sid"], 16)
        code = self._admit(sid, m)
        if code == "ok":
            try:
                self.rt.open(sid, m["max_pos"], m["W"])
            except KVFull:
                code = "kv_full"
            except WTooDeep:
                code = "w_too_deep"
            except SessionExists:
                code = "exists"
            except StageRuntimeError:
                code = "bad"
        draft = None
        if code == "ok" and m["drafter"] is not None:
            try:                              # a plugin fault refuses this lease, nothing more
                draft = self._drafters[m["drafter"]](m)
            except Exception:                 # noqa: BLE001
                self.rt.close(sid)
                code = "unsupported"
        if code == "ok":
            tail = m["role"] in ("tail", "solo")
            self._leases[sid] = Lease(sid=sid, hex=m["sid"], rte=m["rte"], role=m["role"],
                                      pred=m["pred"], next=m["next"], ret=m["ret"],
                                      backup=m["backup"], max_pos=m["max_pos"], W=m["W"], cx=m["cx"],
                                      seed=None if m["seed"] is None else int(m["seed"], 16),
                                      front=Frontier() if tail else None, draft=draft)
        self.count(f"lease_{code}")
        self._ack(peer, m["sid"], m["rte"], code)

    def _lease_update(self, peer: str, m: dict) -> None:
        sid = int(m["sid"], 16)
        L = self._leases.get(sid)
        if L is None or peer != L.ret:        # v0 authority: the lease's entry (M2: signature)
            self.count("update_refused")      # answered, so a re-adopting entry learns fast
            self._ack(peer, m["sid"], m["rte"], "bad")
            return
        if m["rte"] == L.rte:                 # a retransmission: already applied
            self._ack(peer, m["sid"], m["rte"], "ok")
            return
        if m["rte"] < L.rte or not self._verify_update(m):
            self._ack(peer, m["sid"], m["rte"], "bad")
            return
        F = m["truncate_to"]
        if 0 <= F < self.rt.length(sid):
            try:
                self.rt.truncate(sid, F)
            except StageRuntimeError as e:
                self._ack(peer, m["sid"], m["rte"], _code(e))
                return
        fr = L.front
        if fr is not None and fr.cfront is not None and 0 <= F <= fr.cfront:
            # re-opened at F: the model's token there is the one this tail computed on the
            # committed path; only a tail without that record (none yet) takes the entry's word
            fr.cfront, fr.mfront = F - 1, L.path.get(F)
            for p in [p for p in L.path if p > F]:
                del L.path[p]
        L.rte = m["rte"]                      # queued frames of the old rte die at dequeue
        for k in ("pred", "next", "ret"):
            if m[k] is not None:
                setattr(L, k, m[k])
        if m["backup"] is not None:
            L.backup = m["backup"] or None
        self._ack(peer, m["sid"], m["rte"], "ok")

    def _close(self, peer: str, m: dict) -> None:
        sid = int(m["sid"], 16)
        L = self._leases.get(sid)
        if L is None or peer != L.ret:        # only the lease's entry may end it
            self.count("close_refused")
            if L is not None:
                self._ack(peer, m["sid"], L.rte, "bad")
            return
        self._drop_session(sid)

    def _drop_session(self, sid: int) -> None:
        self._leases.pop(sid, None)
        q = self._fifo.pop(sid, None)
        if q:
            self._queued_rows -= sum(f["n"] for f, _ in q)
        try:
            self._rr.remove(sid)
        except ValueError:
            pass
        self._skips.pop(sid, None)
        self.rt.close(sid)

    def _ack(self, peer: str, sid_hex: str, rte: int, code: str, seq: int | None = None) -> None:
        self._send(peer, frame.control("ack", sid=sid_hex, rte=rte, seq=seq, code=code))

    # ── frames ────────────────────────────────────────────────────────────────────────────
    def _on_frame(self, peer: str, f: dict, t: float) -> None:
        sid = int(f["sid"], 16)
        L = self._leases.get(sid)
        if L is None or f["rte"] != L.rte:
            self.count("drop_no_lease")
            return
        if peer != L.pred:
            self.count("drop_not_pred")
            return
        q = self._fifo.setdefault(sid, deque())
        if not q:
            self._rr.append(sid)
            self._skips.pop(sid, None)
        q.append((f, t))
        self._queued_rows += f["n"]
        self.count("frames_in")
        if self._trace is not None:
            self._trace("in", self.peer_id, f["sid"], f["seq"], f["pos"], f["n"], t)

    def _pop(self, sid: int):
        f, t = self._fifo[sid].popleft()
        self._queued_rows -= f["n"]
        return f, t

    STARVE = 4       # steps a ready session may be passed over before it goes first

    def _pick(self):
        """(sessions for one batched step, or a session whose frame exceeds the budget).
        Head-of-line housekeeping first: stale-rte frames die, fenced frames pass through.
        Then the §4.2 priority classes over each session's head frame: rebuild feeds, then
        TRUTH/SPEC frontier-first (smallest pos - cpos), then PREFILL; ties in round-robin
        order. A session passed over STARVE steps in a row goes first, longest-waiting first
        whatever its class (a prefill, or a frame over the budget, is never starved by a steady
        decode load, however many decodes are waiting too)."""
        ready = []
        for i, sid in enumerate(list(self._rr)):
            q = self._fifo.get(sid)
            L = self._leases.get(sid)
            while q:
                f, _ = q[0]
                if L is None or f["rte"] != L.rte:
                    self._pop(sid)
                    self.count("drop_stale_rte")
                elif self._fenced(L, f):
                    self._pop(sid)
                    self._forward_fence(L, f)
                else:
                    break
            if q:
                f = q[0][0]
                cls = 0 if f["kind"] == "R" else (2 if f["kind"] == "P" else 1)
                w = self._skips.get(sid, 0)
                age = -w if w >= self.STARVE else 0
                ready.append(((age == 0, age, cls, f["pos"] - f["cpos"], i), sid, f["n"]))
        ready.sort()
        chosen, total = [], 0
        for _key, sid, n in ready:
            if n > self._budget:
                if not chosen:
                    return [], sid
                continue
            if total + n <= self._budget:
                chosen.append(sid)
                total += n
        return chosen, None

    @staticmethod
    def _fenced(L: Lease, f: dict) -> bool:
        """THE FENCE (§4.2 step 2), judged at dequeue: a fenced frame or one of an older sep
        computes nothing, commits nothing and moves no frontier."""
        return f["fen"] or f["sep"] < L.sep_seen

    def _step_once(self) -> None:
        chosen, big = self._pick()
        served = chosen or ([big] if big is not None else [])
        if chosen:
            self._run(chosen)
        elif big is not None:
            self._run_big(big)
        rest = [s for s in self._rr if s not in served and self._fifo.get(s)]
        self._rr = deque(rest + [s for s in served if self._fifo.get(s)])
        for s in served:
            self._skips.pop(s, None)
        for s in rest:
            self._skips[s] = self._skips.get(s, 0) + 1

    def _segment(self, L: Lease, f: dict, x: Rows | None = None, lo: int = 0, hi: int | None = None):
        n = f["n"] if hi is None else hi - lo
        if f["kind"] == "R":
            raise _Refuse("unsupported")      # rebuild feeds come with takeover (M1)
        if f["ct"] != "none":
            raise _Refuse("unsupported")
        if x is None:
            x = self._rows_in(f)
        # the SEG kind is the FRAME's (a chunk of a verify frame is still a verify segment)
        kind = "prefill" if f["kind"] == "P" else ("decode" if f["n"] == 1 else "verify")
        tokens = "none"
        if L.role in ("tail", "solo") and (f["kind"] != "P" or f["last"]):
            tokens = f["lg"]
            if self._judged(L, f) and tokens != "all":    # the judge reads every row's token
                tokens = "last" if f["kind"] == "P" else "all"
        return Segment(L.sid, f["pos"] + lo, n, x, kind=kind, tokens=tokens)

    @staticmethod
    def _judged(L: Lease, f: dict) -> bool:
        """Does the tail judge `acc` on this frame? (a tail lease, an eid, a frame it answers)"""
        return L.front is not None and f["eid"] is not None and (f["kind"] != "P" or f["last"])

    def _open_eid(self, L: Lease, eid) -> bytes:
        """D1 seam: open the AEAD under the lease's ekey. v0: the eid rides as plaintext."""
        return bytes(eid)

    def _eid_ids(self, L: Lease, f: dict) -> list:
        try:
            ids, _dnxt = frame.eid_unpack(self._open_eid(L, f["eid"]), f["n"])
        except ValueError:
            raise _Refuse("bad") from None
        return ids

    def _check_judged(self, L: Lease, f: dict) -> None:
        """Before anything moves: the eid opens to n ids (a solo hop's are the ids it computes),
        a prefill is the lease's first, and a decode frame has a prefilled frontier to be judged
        against and a cpos claiming no more than the tail has committed."""
        if not self._judged(L, f):
            return
        ids = self._eid_ids(L, f)
        if L.role == "solo" and f["cx"] == "ids" and f["x"] is not None:
            if ids != [int(t) for t in np.frombuffer(f["x"].data, dtype="<i4")]:
                raise _Refuse("bad")          # the ids it judges must be the ids it computes
        cf = L.front.cfront
        if f["kind"] == "P":
            if cf is not None:                # one prefill per lease: a later P would reset
                raise _Refuse("bad")          # the frontier to whatever it fed, unjudged
        elif cf is None or f["cpos"] > cf + 1:
            raise _Refuse("bad")

    def _admit_frame(self, L: Lease, f: dict) -> None:
        """Seek checks for the whole frame against the frontier its cpos WILL set, then commit.
        A refused frame changes nothing: not even the committed frontier (a gap frame's cpos
        must not raise the rollback floor)."""
        self._check_judged(L, f)
        info = self.rt.info(L.sid)
        pos, n = f["pos"], f["n"]
        c = max(info.committed, f["cpos"])
        if pos > info.length:
            raise PositionGap(f"frame at {pos}, length {info.length}")
        if pos < info.length and pos < c:
            raise RollbackPastCommit(f"frame at {pos} below the frontier {c}")
        if pos + n > c + info.spec_window or pos + n > info.max_pos or f["cpos"] > info.max_pos:
            raise SpecWindowExceeded(f"frame ends at {pos + n}, frontier {c} + W {info.spec_window}")
        if f["cpos"] >= 0:
            self.rt.commit(L.sid, f["cpos"])

    def _run(self, sids: list) -> None:
        work = {}
        for sid in sids:
            f, _ = self._pop(sid)
            L = self._leases[sid]
            try:
                seg = self._segment(L, f)
                self._admit_frame(L, f)
            except _Refuse as r:
                self._refuse(L, f, r.code)
                continue
            except StageRuntimeError as e:
                self._refuse(L, f, _code(e))
                continue
            L.sep_seen = f["sep"]
            work[sid] = (L, f, seg)
        while work:
            try:
                out = self.rt.step(StepIn(tuple(seg for _, _, seg in work.values())))
            except StageRuntimeError as e:
                if e.sid in work:
                    L, f, _ = work.pop(e.sid)
                    self._refuse(L, f, _code(e))
                    continue
                self._fault(work)
                return
            except Exception:                 # backend fault: those sessions are undefined (§4.1)
                self._fault(work)
                return
            self.count("steps")
            self.count("rows", sum(seg.n for _, _, seg in work.values()))
            self._max("max_batch", len(work))
            for (L, f, _), o in zip(work.values(), out.outs):
                self._emit(L, f, o.hidden, o.tokens)
            return

    def _run_big(self, sid: int) -> None:
        """A frame larger than the token budget: validate the whole frame first, then run it
        as consecutive steps and forward it whole. The seek checks cannot fail mid-frame; a
        content refusal or backend error in a later chunk leaves the session half-advanced,
        so the session is closed and reported as a fault (§4.1: "undefined, fail over").
        Under shape-dependent numerics the bytes then depend on this hop's budget: they
        replay from the recorded schedule, they do not equal an unsplit run (§0.6)."""
        f, _ = self._pop(sid)
        L = self._leases[sid]
        n = f["n"]
        try:
            x = self._rows_in(f)
            self._segment(L, f, x.slice(0, 1), 0, 1)       # kind / ct checks
            self._admit_frame(L, f)
        except _Refuse as r:
            self._refuse(L, f, r.code)
            return
        except StageRuntimeError as e:
            self._refuse(L, f, _code(e))
            return
        L.sep_seen = f["sep"]
        hidden, toks = [], []
        try:
            for lo in range(0, n, self._budget):
                hi = min(n, lo + self._budget)
                seg = self._segment(L, f, x.slice(lo, hi), lo, hi)
                if seg.tokens == "last" and hi < n:
                    seg = Segment(seg.sid, seg.start_pos, seg.n, seg.x, kind=seg.kind)
                o = self.rt.step(StepIn((seg,))).outs[0]
                self.count("steps")
                self.count("rows", seg.n)
                hidden.append(o.hidden)
                if o.tokens is not None:
                    toks.append(o.tokens)
        except Exception:
            self._fault({sid: (L, f, None)})
            return
        self._max("max_batch", 1)
        self._emit(L, f, Rows.concat(hidden), Rows.concat(toks) if toks else None)

    def _refuse(self, L: Lease, f: dict, code: str) -> None:
        self.count(f"refused_{code}")
        self._ack(L.ret, f["sid"], L.rte, code, seq=f["seq"])

    def _fault(self, work: dict) -> None:
        for sid, (L, f, _) in list(work.items()):
            self.count("faults")
            self._drop_session(sid)
            self._ack(L.ret, f["sid"], L.rte, "fault", seq=f["seq"])

    def _max(self, key: str, v: int) -> None:
        with self._slock:
            if v > self.stats.get(key, 0):
                self.stats[key] = v

    # ── output ────────────────────────────────────────────────────────────────────────────
    def _rho(self) -> float:
        return min(1.0, self._queued_rows / max(1, self._budget))

    def _ld(self, f: dict) -> list:
        return (list(f["ld"]) + [int(round(255 * self._rho()))])[-frame.MAX_HOPS:]

    def _emit(self, L: Lease, f: dict, hidden: Rows, tokens: Rows | None) -> None:
        hdr = {k: f[k] for k in ("sid", "seq", "sep", "pos", "n")}
        if L.role in ("head", "mid"):
            msg = dict(hdr, rte=L.rte, kind=f["kind"], cpos=f["cpos"], sub=f["sub"],
                       last=f["last"], lg=f["lg"], hint=f["hint"], ld=self._ld(f), eid=f["eid"],
                       **self._payload_out(hidden))
            self._send_frame(L, frame.frame(**msg))
            return
        if f["kind"] == "P" and not f["last"]:
            return
        toks = np.zeros(0, dtype="<i4") if tokens is None else np.ascontiguousarray(tokens.data, dtype="<i4")
        acc = blk = None
        if self._judged(L, f):
            acc, blk = self._judge(L, f, [int(t) for t in toks])
        lg = f["lg"]
        toks = toks if lg == "all" else toks[-1:] if lg == "last" else toks[:0]
        r = frame.reply(**hdr, rte=L.rte, ld=self._ld(f), acc=acc, blk=blk,
                        toks=frame.Tensor("i32", (len(toks),), np.ascontiguousarray(toks)))
        self._send(L.ret, r)
        self.count("replies")
        if self._trace is not None:
            self._trace("reply", self.peer_id, f["sid"], f["seq"], f["pos"], f["n"], self.clock.now())

    def _judge(self, L: Lease, f: dict, toks: list):
        """The tail's half of the accept rule (pspec.accept.Frontier) -> (acc, blk tensor|None).
        Frames reach it in the session's FIFO order, and fenced or refused frames never do."""
        fr, ids, pos = L.front, self._eid_ids(L, f), f["pos"]
        if f["kind"] == "P":
            fr.prefill(pos + f["n"], toks[-1])
            L.path[pos + f["n"]] = toks[-1]
            acc = True
        else:
            if fr.mfront is None and pos == fr.cfront + 1:     # re-opened, no record: trusted
                fr.mfront = ids[0]
                L.path[pos] = ids[0]
            acc, k = fr.judge(pos, ids, toks)
            for j in range(k + 1):                             # k = -1 when off the path
                L.path[pos + j + 1] = toks[j]
        blk = None
        if acc and L.draft is not None:
            try:                              # a plugin fault costs this reply its block only
                b = [int(t) for t in L.draft(fr.cfront, fr.mfront)][:frame.MAX_BLK]
            except Exception:                 # noqa: BLE001
                b = []
                self.count("draft_errors")
            if b and min(b) >= 0 and max(b) < self.cap.vocab_size:
                blk = frame.Tensor("i32", (len(b),), np.asarray(b, dtype="<i4"))
        self.count("judged_acc" if acc else "judged_off")
        return acc, blk

    def _forward_fence(self, L: Lease, f: dict) -> None:
        self.count("fenced")
        hdr = {k: f[k] for k in ("sid", "seq", "sep", "pos", "n")}
        if L.role in ("head", "mid"):
            self._send_frame(L, frame.frame(**hdr, rte=L.rte, kind=f["kind"], cpos=f["cpos"],
                                            sub=f["sub"], last=f["last"], lg=f["lg"],
                                            cx=self._out_cx, fen=True, ld=self._ld(f)))
        elif f["kind"] != "P" or f["last"]:
            self._send(L.ret, frame.reply(**hdr, rte=L.rte, fen=True, ld=self._ld(f)))

    def _send_frame(self, L: Lease, msg: dict) -> None:
        self._send(L.next, msg)
        self.count("frames_out")
        if self._trace is not None:
            self._trace("out", self.peer_id, msg["sid"], msg["seq"], msg["pos"], msg["n"], self.clock.now())

    def _send(self, peer: str, msg: dict) -> None:
        if not self._dead:
            self.links.send(peer, msg)

    def _on_send_failed(self, peer: str, meta: dict) -> None:
        self.count("send_failed")
        if meta.get("op") != "f" or meta.get("sid") is None:
            return
        L = self._leases.get(int(meta["sid"], 16))
        if L is not None and L.next == peer:
            self.count("next_down")
            self._ack(L.ret, meta["sid"], L.rte, "next_down", seq=meta["seq"])

    # ── codecs ────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _wire_codecs(dtypes) -> set:
        out = set()
        for d in dtypes:
            if d == "bf16" or d == "i32":     # i32: bit-exact bf16 pairs (module docstring)
                out.add("bf16")
            elif d == "fp8_e4m3":
                out.add("fp8")
        return out

    def _rows_in(self, f: dict) -> Rows:
        x, cx, n, H = f["x"], f["cx"], f["n"], self.cap.hidden_size
        codecs = self.cap.codecs_in
        if cx == "ids":
            if not self.cap.is_head:
                raise _Refuse("bad")
            ids = np.frombuffer(x.data, dtype="<i4").astype(np.int32)
            if len(ids) and int(ids.max()) >= self.cap.vocab_size:
                raise _Refuse("bad")          # refused before cpos is committed
            return Rows("i32", ids)
        if self.cap.is_head:
            raise _Refuse("bad")
        if cx == "bf16" and "bf16" in codecs and x.shape == (n, H):
            return Rows("bf16", np.frombuffer(x.data, dtype="<u2").reshape(n, H).astype(np.uint16))
        if cx == "bf16" and "i32" in codecs and x.shape == (n, 2 * H):
            return Rows("i32", np.frombuffer(x.data, dtype="<i4").reshape(n, H).astype(np.int32))
        if cx == "fp8" and "fp8_e4m3" in codecs and x.shape == (n, H):
            return Rows("fp8_e4m3", np.frombuffer(x.data, dtype=np.uint8).reshape(n, H).copy(),
                        np.frombuffer(f["xs"].data, dtype="<f4").astype(np.float32))
        raise _Refuse("bad")

    @staticmethod
    def _payload_out(rows: Rows) -> dict:
        d = rows.data
        if rows.dtype == "bf16":
            return {"cx": "bf16", "x": frame.Tensor("bf16", d.shape, np.ascontiguousarray(d, dtype="<u2"))}
        if rows.dtype == "fp8_e4m3":
            return {"cx": "fp8", "x": frame.Tensor("fp8", d.shape, np.ascontiguousarray(d)),
                    "xs": frame.Tensor("f32", (rows.n,), np.ascontiguousarray(rows.scale, dtype="<f4"))}
        if rows.dtype == "i32":
            d = np.ascontiguousarray(d, dtype="<i4")
            return {"cx": "bf16", "x": frame.Tensor("bf16", (d.shape[0], 2 * d.shape[1]), d.view("<u2"))}
        raise ValueError(f"no shard-frame/2 codec for wire dtype {rows.dtype}")
