"""Injectable clocks for the mesh data plane (ARCHITECTURE-v2 G12).

Everything timed in the node server and its links (and, later, the entry's deadlines) goes
through a Clock: `now()`, `sleep()`, a `Mailbox` whose `get()` can time out, `spawn()` for
threads, `idle()` around a blocking socket read, and a `conn()` account per TCP stream.
`RealClock` maps these to time.monotonic, threading and no-ops, so production pays nothing.

`VirtualClock` exists so that 10^3 injected faults with 2 s deadlines run in seconds, without
making the harness lie about races. The rules:

* An ACTOR is a thread the clock knows (`spawn`, or `adopt` for the test's own thread). It is
  BUSY while it runs and IDLE only while it is blocked inside a clock primitive
  (`Mailbox.get`, `sleep`, `settle`) or inside `with clock.idle():` (a blocking socket read).
* A message on a TCP stream is IN FLIGHT from the sender's `acct.sent()` (called before the
  write) until the receiver's `acct.took()` (called once the whole message is read), or until
  the stream dies (`acct.dead()` releases everything sent and not taken).
* load = busy actors + in-flight messages. Virtual time moves ONLY when load is 0: `now` jumps
  to the earliest pending timer, and every timer due at that instant fires. A fired actor is
  marked busy before anyone can see load 0 again.
* Wake-ups hand over busy-ness atomically. `Mailbox.put` marks a waiting consumer busy before it
  notifies, so there is never an instant where work exists, its consumer is idle and load is 0.

So a deadline expires only when nothing left in the system can make progress without time
passing: a frame still on a socket, in a queue or being computed can never be timed out
"early", and a delay can never be skipped. Thread interleavings and socket I/O stay real; only
waiting is virtual. What virtual time does NOT model is real slowness: compute costs zero
virtual time unless a runtime sleeps its modelled cost (tests/fake_mesh.py SimRuntime does).

An accounting bug (an uncounted wake-up, a lost message that was never released) or a livelock
(two peers answering each other forever, so load never reaches 0) shows up as a frozen clock,
never as a silent hang: a wait of an actor adopted with `raise_on_stuck` raises `Stuck` with a
dump of busy actors, timers and live streams once virtual time has not moved for `stuck_after`
real seconds.

Where faults land. A timer fires only at quiescence, so a fault injected "at virtual time t"
lands when every frame is on a delayed leg, inside a node's modelled compute, or queued: a
protocol-level instant. It never lands inside a syscall (mid-write, mid-dial, between a read
and its hand-off). Those races need the real clock (tests/test_fake_mesh.py runs a real-clock
storm for them).
"""
from __future__ import annotations

import heapq
import itertools
import threading
import time
from collections import deque
from contextlib import contextmanager


class Timeout(Exception):
    """Mailbox.get timed out."""


class Closed(Exception):
    """The mailbox was closed."""


class Stuck(RuntimeError):
    """VirtualClock: no progress for `stuck_after` real seconds (an accounting bug or deadlock)."""


class _NullAcct:
    """RealClock's stream account: nothing to count."""
    id = 0

    def sent(self) -> None:
        pass

    def took(self) -> None:
        pass

    def dead(self) -> None:
        pass


_NULL_ACCT = _NullAcct()


class _RealMailbox:
    """A single-consumer queue whose get() can time out (real time)."""

    def __init__(self, name: str = "mailbox"):
        self.name = name
        self._q: deque = deque()
        self._cv = threading.Condition()
        self._closed = False

    def put(self, item) -> bool:
        with self._cv:
            if self._closed:
                return False
            self._q.append(item)
            self._cv.notify()
            return True

    def get(self, timeout: float | None = None):
        with self._cv:
            end = None if timeout is None else time.monotonic() + max(0.0, timeout)
            while not self._q:
                if self._closed:
                    raise Closed(self.name)
                left = None if end is None else end - time.monotonic()
                if left is not None and left <= 0:
                    raise Timeout(self.name)
                self._cv.wait(left)
            return self._q.popleft()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def __len__(self) -> int:
        with self._cv:
            return len(self._q)


class RealClock:
    """Wall-clock time. Every accounting hook is a no-op."""
    virtual = False

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, dt: float) -> None:
        if dt > 0:
            time.sleep(dt)

    def settle(self) -> None:
        """Give in-flight work a moment (real time cannot know when the system is idle)."""
        time.sleep(0.02)

    def mailbox(self, name: str = "mailbox") -> _RealMailbox:
        return _RealMailbox(name)

    def spawn(self, name: str, fn, *args) -> threading.Thread:
        t = threading.Thread(target=fn, args=args, name=name, daemon=True)
        t.start()
        return t

    def adopt(self, name: str = "main", raise_on_stuck: bool = True) -> None:
        pass

    def disown(self) -> None:
        pass

    @contextmanager
    def idle(self):
        yield

    def conn(self, label: str = "") -> _NullAcct:
        return _NULL_ACCT

    def conn_lookup(self, cid) -> _NullAcct:
        return _NULL_ACCT


# ── virtual time ──────────────────────────────────────────────────────────────────────────────

class _Actor:
    __slots__ = ("name", "busy", "timer", "fired", "raises", "cv", "where")

    def __init__(self, name: str, lock, raises: bool):
        self.name, self.busy, self.timer, self.fired = name, True, None, False
        self.raises, self.cv, self.where = raises, threading.Condition(lock), ""


class _Acct:
    """In-flight accounting of one TCP stream, shared by its two in-process ends."""
    __slots__ = ("clock", "id", "label", "n_sent", "n_took", "is_dead")

    def __init__(self, clock: "VirtualClock", cid: int, label: str):
        self.clock, self.id, self.label = clock, cid, label
        self.n_sent = self.n_took = 0
        self.is_dead = False

    def sent(self) -> None:
        """Before writing one message: it is in flight until taken or the stream dies."""
        c = self.clock
        with c._lock:
            if not self.is_dead:
                self.n_sent += 1
                c._load += 1

    def took(self) -> None:
        """The reader holds one whole message (the calling reader actor is busy)."""
        c = self.clock
        with c._lock:
            c._set(c._me(), True)
            if not self.is_dead and self.n_took < self.n_sent:
                self.n_took += 1
                c._load -= 1

    def dead(self) -> None:
        """The stream is gone: whatever was sent and not taken never arrives."""
        c = self.clock
        with c._lock:
            if self.is_dead:
                return
            self.is_dead = True
            c._load -= self.n_sent - self.n_took
            c._accts.pop(self.id, None)
            if c._load == 0:
                c._advance()


class _VMailbox:
    """Single-consumer mailbox in virtual time. put() marks a waiting consumer busy."""

    def __init__(self, clock: "VirtualClock", name: str):
        self._c, self.name = clock, name
        self._q: deque = deque()
        self._waiter: _Actor | None = None
        self._closed = False

    def put(self, item) -> bool:
        c = self._c
        with c._lock:
            if self._closed:
                return False
            self._q.append(item)
            w = self._waiter
            if w is not None:
                c._set(w, True)
                w.cv.notify()
            return True

    def get(self, timeout: float | None = None):
        c = self._c
        with c._lock:
            me = c._me()
            deadline = None if timeout is None else c._now + max(0.0, float(timeout))
            while True:
                if self._q:
                    return self._q.popleft()
                if self._closed:
                    raise Closed(self.name)
                if deadline is not None and c._now >= deadline:
                    raise Timeout(self.name)
                if self._waiter is not None and self._waiter is not me:
                    raise RuntimeError(f"mailbox {self.name!r} has one consumer")
                self._waiter = me
                me.where = f"get({self.name})"
                try:
                    c._block(me, deadline, lambda: bool(self._q) or self._closed)
                finally:
                    self._waiter = None
                    me.where = ""

    def close(self) -> None:
        c = self._c
        with c._lock:
            self._closed = True
            w = self._waiter
            if w is not None:
                c._set(w, True)
                w.cv.notify()

    def __len__(self) -> int:
        with self._c._lock:
            return len(self._q)


class VirtualClock:
    """Discrete virtual time over real threads and real sockets (see the module docstring)."""
    virtual = True

    def __init__(self, start: float = 0.0, stuck_after: float = 60.0):
        self._lock = threading.Lock()
        self._now = float(start)
        self._load = 0
        self._timers: list = []           # heap of (deadline, seq, actor)
        self._seq = itertools.count(1)
        self._ids = itertools.count(1)
        self._tls = threading.local()
        self._actors: dict[int, _Actor] = {}
        self._accts: dict[int, _Acct] = {}
        self.stuck_after = stuck_after
        self.advances = 0                 # how many times time jumped (stats)

    # ── time ──────────────────────────────────────────────────────────────────────────────
    def now(self) -> float:
        with self._lock:
            return self._now

    def sleep(self, dt: float) -> None:
        if dt <= 0:
            return
        with self._lock:
            me = self._me()
            me.where = f"sleep({dt:.6f})"
            self._block(me, self._now + dt, lambda: False)
            me.where = ""

    def settle(self) -> None:
        """Wait until everything else is idle at the CURRENT instant (time does not move)."""
        with self._lock:
            me = self._me()
            me.where = "settle"
            self._block(me, self._now, lambda: False)
            me.where = ""

    def mailbox(self, name: str = "mailbox") -> _VMailbox:
        return _VMailbox(self, name)

    # ── actors ────────────────────────────────────────────────────────────────────────────
    def _new_actor(self, name: str, raises: bool) -> _Actor:
        a = _Actor(name, self._lock, raises)
        self._actors[id(a)] = a
        self._load += 1
        return a

    def _close_actor(self, a: _Actor) -> None:
        a.timer = None
        self._actors.pop(id(a), None)
        if a.busy:
            a.busy = False
            self._load -= 1
            if self._load == 0:
                self._advance()

    def adopt(self, name: str = "main", raise_on_stuck: bool = True) -> None:
        """Register the calling thread as a busy actor (the test's own thread)."""
        with self._lock:
            if getattr(self._tls, "actor", None) is None:
                self._tls.actor = self._new_actor(name, raise_on_stuck)

    def disown(self) -> None:
        with self._lock:
            a = getattr(self._tls, "actor", None)
            if a is not None:
                self._tls.actor = None
                self._close_actor(a)

    def spawn(self, name: str, fn, *args) -> threading.Thread:
        """Start fn(*args) on a new thread that is a busy actor from before it starts."""
        with self._lock:
            a = self._new_actor(name, False)

        def run():
            self._tls.actor = a
            try:
                fn(*args)
            finally:
                with self._lock:
                    self._close_actor(a)

        t = threading.Thread(target=run, name=name, daemon=True)
        t.start()
        return t

    @contextmanager
    def idle(self):
        """Mark the calling actor idle around a blocking call the clock cannot see (a socket
        read whose wake-up is covered by the stream's in-flight count)."""
        with self._lock:
            me = self._me()
            me.where = "io"
            self._set(me, False)
        try:
            yield
        finally:
            with self._lock:
                me.where = ""
                self._set(me, True)

    def _me(self) -> _Actor:
        a = getattr(self._tls, "actor", None)
        if a is None:
            raise RuntimeError(f"thread {threading.current_thread().name!r} is not a VirtualClock "
                               "actor (start it with clock.spawn, or clock.adopt it)")
        return a

    # ── streams ───────────────────────────────────────────────────────────────────────────
    def conn(self, label: str = "") -> _Acct:
        with self._lock:
            a = _Acct(self, next(self._ids), label)
            self._accts[a.id] = a
            return a

    def conn_lookup(self, cid) -> _Acct:
        with self._lock:
            a = self._accts.get(cid) if isinstance(cid, int) else None
            if a is None:                     # unknown or already dead: counts nothing
                a = _Acct(self, 0, "dead")
                a.is_dead = True
            return a

    # ── core (all under self._lock) ───────────────────────────────────────────────────────
    def _set(self, a: _Actor, busy: bool) -> None:
        if a.busy == busy:
            return
        a.busy = busy
        self._load += 1 if busy else -1
        if self._load == 0:
            self._advance()

    def _advance(self) -> None:
        while self._load == 0 and self._timers:
            t = self._timers[0][0]
            fired = []
            while self._timers and self._timers[0][0] <= t:
                _, seq, a = heapq.heappop(self._timers)
                if a.timer == seq:            # else cancelled / superseded
                    a.timer, a.fired = None, True
                    if not a.busy:
                        a.busy = True
                        self._load += 1
                    fired.append(a)
            if fired:
                if t > self._now:
                    self._now = t
                    self.advances += 1
                for a in fired:
                    a.cv.notify()

    def _block(self, a: _Actor, deadline, done) -> None:
        """Park actor a (idle) until done() or its timer at `deadline` fires; return busy."""
        if deadline is not None:
            seq = next(self._seq)
            a.timer, a.fired = seq, False
            heapq.heappush(self._timers, (deadline, seq, a))
        self._set(a, False)
        seen, since = self._now, time.monotonic()
        while not (a.fired or done()):
            a.cv.wait(0.5)
            if self._now != seen:
                seen, since = self._now, time.monotonic()
            elif a.raises and time.monotonic() - since > self.stuck_after:
                a.timer, a.fired = None, False
                self._set(a, True)
                raise Stuck(self._dump(a))
        a.timer, a.fired = None, False
        self._set(a, True)

    def _dump(self, me: _Actor) -> str:
        busy = sorted(a.name for a in self._actors.values() if a.busy)
        waits = sorted(f"{a.name}:{a.where}" for a in self._actors.values() if not a.busy)
        timers = sorted((round(t, 6), a.name) for t, s, a in self._timers if a.timer == s)[:10]
        live = [(x.label, x.n_sent - x.n_took) for x in self._accts.values() if x.n_sent > x.n_took]
        return (f"virtual clock stuck at t={self._now:.6f} (waiting: {me.name}): load={self._load}, "
                f"busy={busy}, idle={waits[:40]}, timers={timers}, in-flight streams={live}")

    def dump(self) -> str:
        with self._lock:
            return self._dump(_Actor("(dump)", self._lock, False))
