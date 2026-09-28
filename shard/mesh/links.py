"""Links: the node server's view of the network, over plain TCP (ARCHITECTURE-v2 §4.6 stand-in).

Sidecar v2 will give a node `dial(PeerId) -> stream` from a pool of warm streams and tag every
inbound stream with the peer's verified PeerId. `TcpLinks` is that surface on loopback (and on
any network a test or a trusted LAN provides), so the node server is written against the final
shape and M4 swaps the transport, not the node:

* Streams are one-way. A dialer only writes, an acceptor only reads. Replies travel on a
  stream the replier dials. Each message is one shard-frame/2 body in the unchanged
  shard/transport framing (8-byte big-endian length, then the body), read with `frame.recv`.
* A dialer opens a stream with a hello (a transport-framed JSON header `{"hello": PeerId,
  "acct": id}`). Loopback has no crypto, so the PeerId is self-declared; the sidecar verifies
  it in production. `acct` pairs the two ends' in-flight accounting under a VirtualClock.
* `send(peer, msg)` validates and encodes at the call (a bad message raises ValueError at the
  caller), then queues the bytes on the peer's outbox. One outbox thread per peer writes in
  order. A dead pooled stream is detected before the write (the acceptor never writes, so a
  readable dialer socket means FIN/RST) and re-dialled; a write error re-dials once (§4.2: a
  reset is not a death). If the re-dial fails too, `on_send_error(peer, meta, err)` reports the
  message as undeliverable. That second failed dial is the "dial-back probe" of §8.
* A malformed inbound message resets THAT stream only (`stats["bad_frames"]`), never the node.

Fault seams for the harness (tests/fake_mesh.py) are two overridable methods: `_delay(peer)`
(seconds added to each message's delivery time; messages stay FIFO) and `_blocked(peer)`
(a partition: the message vanishes at delivery time).
"""
from __future__ import annotations

import json
import re
import select
import socket
import struct
import threading
from collections import deque

from .. import frame
from .clock import Closed, RealClock, Timeout

_PEER = re.compile(r"[A-Za-z0-9]{1,128}")
_HELLO_MAX = 1024


class LinkError(ConnectionError):
    """A peer cannot be reached (unknown address, refused dial, dead stream after a re-dial)."""


def _read_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        b = sock.recv(n - len(buf))
        if not b:
            raise ConnectionError("stream closed")
        buf += b
    return bytes(buf)


def _hello_bytes(me: str, acct: int) -> bytes:
    head = json.dumps({"hello": me, "acct": acct}).encode()
    body = struct.pack("!I", len(head)) + head
    return struct.pack("!Q", len(body)) + body


def _read_hello(sock: socket.socket) -> tuple[str, int]:
    (n,) = struct.unpack("!Q", _read_exact(sock, 8))
    if not 4 <= n <= _HELLO_MAX:
        raise ConnectionError(f"hello of {n} B")
    body = _read_exact(sock, n)
    (hl,) = struct.unpack_from("!I", body, 0)
    if 4 + hl != n:
        raise ConnectionError("hello carries blobs")
    try:
        obj = json.loads(body[4:])
    except ValueError as e:
        raise ConnectionError(f"hello is not JSON: {e}") from None
    peer, acct = (obj.get("hello"), obj.get("acct")) if isinstance(obj, dict) else (None, None)
    if not isinstance(peer, str) or not _PEER.fullmatch(peer) or type(acct) is not int:
        raise ConnectionError(f"bad hello {str(obj)[:80]}")
    return peer, acct


def _meta(msg: dict) -> dict:
    """What an outbox remembers of a message, for on_send_error."""
    return {k: msg.get(k) for k in ("op", "cmd", "sid", "rte", "seq", "pos", "n")}


def _abort(sock: socket.socket | None) -> None:
    """Close with RST (SO_LINGER 0) after waking any thread blocked on the socket."""
    if sock is None:
        return
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    except OSError:
        pass
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


class _Outbox:
    __slots__ = ("peer", "mb", "sock", "acct", "thread")

    def __init__(self, peer, mb):
        self.peer, self.mb, self.sock, self.acct, self.thread = peer, mb, None, None, None


class TcpLinks:
    """Dial-by-PeerId stream pool + peer-tagged inbound streams over TCP."""
    dial_timeout = 2.0          # real seconds; loopback refuses or accepts at once

    def __init__(self, me: str, resolve, *, on_message, on_send_error=None, on_stream_error=None,
                 clock=None, host: str = "127.0.0.1"):
        if not _PEER.fullmatch(me):
            raise ValueError(f"bad PeerId {me!r}")
        self.me, self._resolve, self.host = me, resolve, host
        self.clock = clock or RealClock()
        self._on_message, self._on_send_error = on_message, on_send_error
        self._on_stream_error = on_stream_error
        self._lock = threading.Lock()
        self._outs: dict[str, _Outbox] = {}
        self._inbound: set = set()
        self._srv: socket.socket | None = None
        self._closed = False
        self.addr: tuple[str, int] | None = None
        self.stats = {"sent": 0, "blocked": 0, "send_failed": 0, "redials": 0, "received": 0,
                      "bad_frames": 0, "streams_in": 0, "streams_reset": 0}

    # ── lifecycle ─────────────────────────────────────────────────────────────────────────
    def start(self) -> tuple[str, int]:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind((self.host, 0))
        srv.listen(64)
        self._srv, self.addr = srv, srv.getsockname()[:2]
        self.clock.spawn(f"{self.me}.accept", self._accept_loop, srv)
        return self.addr

    def close(self, abrupt: bool = False) -> None:
        """Stop everything. abrupt=True is a crash: streams are reset (RST), queued messages
        are dropped. abrupt=False drops queued messages too (v0 has no drain)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            outs, ins = list(self._outs.values()), list(self._inbound)
        if self._srv is not None:
            _abort(self._srv)
        for o in outs:
            o.mb.close()
        for s in ins:
            _abort(s) if abrupt else _shut(s)
        for o in outs:
            if o.thread is not None and o.thread is not threading.current_thread():
                o.thread.join(timeout=0 if self.clock.virtual else 2.0)
        for o in outs:                        # after the join: an outbox may still hold its stream
            self._drop_stream(o, abrupt)

    def _count(self, key: str, k: int = 1) -> None:
        with self._lock:
            self.stats[key] += k

    # ── outbound ──────────────────────────────────────────────────────────────────────────
    def send(self, peer: str, msg: dict) -> None:
        """Validate + encode now, deliver in order later. Undeliverable -> on_send_error."""
        body = b"".join(frame.encode(msg))
        self._enqueue(peer, struct.pack("!Q", len(body)) + body, _meta(msg))

    def _enqueue(self, peer: str, data: bytes, meta: dict) -> None:
        with self._lock:
            if self._closed:
                return
            o = self._outs.get(peer)
            if o is None:
                o = self._outs[peer] = _Outbox(peer, self.clock.mailbox(f"{self.me}->{peer}"))
                o.thread = self.clock.spawn(f"{self.me}.out.{peer}", self._outbox, o)
        o.mb.put((self.clock.now() + self._delay(peer), data, meta))

    def _outbox(self, o: _Outbox) -> None:
        pending: deque = deque()
        while True:
            try:
                if not pending:
                    pending.append(o.mb.get())
                due = pending[0][0]
                wait = due - self.clock.now()
                if wait > 0:
                    try:
                        pending.append(o.mb.get(timeout=wait))
                    except Timeout:
                        pass
                    continue
            except Closed:
                return
            _, data, meta = pending.popleft()
            self._deliver(o, data, meta)

    def _deliver(self, o: _Outbox, data: bytes, meta: dict) -> None:
        if self._closed:
            return
        if self._blocked(o.peer):
            self._count("blocked")
            return
        err = None
        for attempt in range(2):
            try:
                if o.sock is None or not _alive(o.sock):
                    self._drop_stream(o)
                    self._dial(o)
                s, acct = o.sock, o.acct         # close() may drop o's stream concurrently
                if s is None:
                    raise LinkError("links closed")
                if meta.get("counted", True):
                    acct.sent()
                s.sendall(data)
                self._count("sent")
                return
            except OSError as e:                # incl. ConnectionError / LinkError
                err = e
                self._drop_stream(o)
                if self._closed:
                    return
        self._count("send_failed")
        if self._on_send_error is not None:
            self._on_send_error(o.peer, meta, err)

    def _dial(self, o: _Outbox) -> None:
        try:
            addr = self._resolve(o.peer)
        except (KeyError, LookupError) as e:
            raise LinkError(f"no address for {o.peer}") from e
        if addr is None:
            raise LinkError(f"no address for {o.peer}")
        s = socket.create_connection(addr, timeout=self.dial_timeout)
        try:
            s.settimeout(None)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            acct = self.clock.conn(f"{self.me}->{o.peer}")
            s.sendall(_hello_bytes(self.me, acct.id))
        except OSError:
            s.close()
            raise
        o.sock, o.acct = s, acct
        self._count("redials")

    def _drop_stream(self, o: _Outbox, abrupt: bool = True) -> None:
        s, a = o.sock, o.acct
        o.sock = o.acct = None
        if a is not None:
            a.dead()
        if s is not None:
            _abort(s) if abrupt else _shut(s)

    # ── inbound ───────────────────────────────────────────────────────────────────────────
    def _accept_loop(self, srv: socket.socket) -> None:
        while True:
            try:
                with self.clock.idle():
                    s, _ = srv.accept()
            except OSError:
                return
            with self._lock:
                if self._closed:
                    _abort(s)
                    return
                self._inbound.add(s)
                self.stats["streams_in"] += 1
            self.clock.spawn(f"{self.me}.in", self._reader, s)

    def _reader(self, s: socket.socket) -> None:
        acct = peer = None
        try:
            with self.clock.idle():
                peer, cid = _read_hello(s)
            acct = self.clock.conn_lookup(cid)
            while True:
                with self.clock.idle():
                    msg = frame.recv(s)
                acct.took()
                if self._closed:
                    return
                self._count("received")
                self._on_message(peer, msg)
        except frame.FrameError as e:           # a bad message resets this stream, nothing else
            self._count("bad_frames")
            self._count("streams_reset")
            if self._on_stream_error is not None and not self._closed:
                self._on_stream_error(peer, e)
        except (ConnectionError, OSError, ValueError, struct.error):
            pass
        finally:
            if acct is not None:
                acct.dead()
            with self._lock:
                self._inbound.discard(s)
            _abort(s)

    # ── fault seams (tests/fake_mesh.py overrides these) ──────────────────────────────────
    def _delay(self, peer: str) -> float:
        return 0.0

    def _blocked(self, peer: str) -> bool:
        return False


def _alive(sock: socket.socket) -> bool:
    """A dialer never receives: a readable socket means the acceptor closed or reset it."""
    try:
        r, _, _ = select.select([sock], [], [], 0)
        if not r:
            return True
        return False
    except (OSError, ValueError):
        return False


def _shut(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass
