"""A mixed-fleet toy mesh over real loopback TCP, with fault injectors and virtual time (F4).

WHAT IT IS. D in 1..7 node servers (shard/mesh/node.py), each serving one span of an integer toy
model (shard/runtime/toy.py) through the device-neutral StageRuntime contract, wired hop to hop
over real 127.0.0.1 TCP with shard-frame/2. `FakeEntry` sits where the entry sits (§4.3): it opens
a lease on every hop, feeds token ids to the head and collects the tail's replies and the hops'
acks. Later PRs plug in here: S1 drives `submit`/`collect` from its pspec coordinator, S3 runs many
sessions through `generate_many`, M1/M2 add takeover and re-routing on the same nodes and
injectors (`fault_storm` is M2's 10^3-fault gate in miniature), P1 hashes the wire rows the nodes
emit.

HETEROGENEOUS BY CONSTRUCTION (the vision principle). Every node is a simulated device with a
device-neutral capability record (`SIM_DEVICES`: 24 GB sm_86 / sm_89 cards, a 32 GB sm_120 card,
a 96 GB sm_120 workstation card, a 128 GB GB10 Spark and a 128 GB Mac), a layer span proportional
to its memory (`assign_spans`: bigger devices carry more layers), its own token budget, session
table and per-layer cost model (`SimRuntime`). Fleets cycle through a mix; they are never uniform.

TRUTH. `direct_greedy` runs the same toy model as ONE whole-model runtime in a plain loop with the
mesh's segment schedule. Its sequence T is the position oracle of tests/oracle.py: a reply for a
frame whose last row is position p must carry oracle_token(T, p).

TIME. clock="virtual" (shard.mesh.clock.VirtualClock) or "real". In virtual time a 2 s deadline
costs nothing, yet it can only expire once nothing in the mesh can progress without time passing,
leg delays are exact, and compute takes exactly the modelled cost. The flip side: a fault
scheduled "mid-flight" in virtual time lands at a quiescent instant (a frame on a delayed leg,
inside a node's modelled compute, or queued), never inside a syscall. Real mode runs the same code
on the wall clock (delays via real waits, no accounting); the real-clock storm is what lands
faults mid-write, mid-dial and between a read and its hand-off.

SHAPE-DEPENDENT NUMERICS (mode="adversarial", like float kernels). A hop that re-chunks a frame
over its token budget changes the bytes, so the mesh then equals `replay_chain` (the same
schedule replayed hop by hop), not `direct_greedy`.

FAULTS. Each must surface as an observable failure at the entry, never as a harness hang:
  kill(i)          crash: streams reset, runtime state gone. Next frame: LinkDown (head) or
                   HopError(next_down) from the predecessor, fast. A frame inside the node when it
                   dies is lost: Deadline.
  hang(i)          the node loop parks, sockets stay open: Deadline at exactly the deadline. On
                   resume() it processes what piled up, late (a slow node).
  partition(a, b)  messages between two peers vanish at delivery time (both ways): Deadline.
  malformed(a, b)  bytes that are not a shard-frame/2 message on a's stream to b: b resets that
                   stream only (links bad_frames += 1) and stays up; at most the next message on
                   the stream is lost (Deadline), and a re-feed of the same position succeeds.
Recovery: restart(i), resume(i), heal().
"""
from __future__ import annotations

import dataclasses
import json
import os
import random
import struct
import sys
import threading
from dataclasses import dataclass
from functools import partial

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oracle import oracle_token                                                   # noqa: E402
from shard import frame                                                           # noqa: E402
from shard.mesh.clock import RealClock, Timeout, VirtualClock                     # noqa: E402
from shard.mesh.links import TcpLinks                                             # noqa: E402
from shard.mesh.node import NodeServer                                            # noqa: E402
from shard.runtime.api import DeviceInfo, Rows, Segment, StageSpec, StepIn        # noqa: E402
from shard.runtime.toy import ToyRuntime, ToyWeights                              # noqa: E402

PATTERN = "SSGSSGSGSSSGSSGSGSSSGSSG"      # 24 layers: SWA runs of 1-3 between global layers
WINDOW, HIDDEN, VOCAB = 4, 4, 64
ENTRY = "entry"


# ── simulated devices ─────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SimDevice:
    """A device as the mesh sees it: a DeviceInfo plus a cost model (ms per held layer per step,
    plus ms per token per held layer), a token budget and a session table."""
    key: str
    info: DeviceInfo
    layer_ms: float
    token_ms: float
    budget: int
    max_sessions: int

    @property
    def gb(self) -> int:
        return self.info.total_mem_bytes >> 30


def _dev(key, name, backend, arch, host, gb, unified, bw_tb_s, hw, emu, layer_ms, token_ms, budget,
         sessions):
    info = DeviceInfo(backend=backend, device_name=name, arch_class=arch, host_arch=host, n_devices=1,
                      total_mem_bytes=gb << 30, free_mem_bytes=(gb << 30) * 9 // 10,
                      unified_memory=unified, mem_bw_bytes_s=bw_tb_s * 1e12, mem_bw_source="spec",
                      formats_hw=hw, formats_emulated=emu)
    return SimDevice(key, info, layer_ms, token_ms, budget, sessions)


_BASE = ("fp32", "bf16", "fp16", "int8")
SIM_DEVICES = {d.key: d for d in (
    _dev("rtx3090-24g", "sim RTX 3090", "cuda", "sm_86", "x86_64", 24, False, 0.936, _BASE,
         ("fp8_e4m3", "mxfp4", "nvfp4", "int4"), 0.50, 0.020, 32, 8),
    _dev("rtx4090-24g", "sim RTX 4090", "cuda", "sm_89", "x86_64", 24, False, 1.008,
         _BASE + ("fp8_e4m3",), ("mxfp4", "nvfp4", "int4"), 0.45, 0.015, 32, 8),
    _dev("rtx5090-32g", "sim RTX 5090", "cuda", "sm_120", "x86_64", 32, False, 1.792,
         _BASE + ("fp8_e4m3", "mxfp4", "nvfp4", "int4"), (), 0.28, 0.008, 64, 12),
    _dev("rtxpro6000-96g", "sim RTX PRO 6000", "cuda", "sm_120", "x86_64", 96, False, 1.792,
         _BASE + ("fp8_e4m3", "mxfp4", "nvfp4", "int4"), (), 0.28, 0.008, 128, 32),
    _dev("dgx-spark-128g", "sim DGX Spark (GB10)", "cuda", "sm_121", "aarch64", 128, True, 0.273,
         _BASE + ("fp8_e4m3", "mxfp4", "nvfp4", "int4"), (), 1.80, 0.030, 128, 32),
    _dev("mac-m3ultra-128g", "sim Mac Studio M3 Ultra", "mlx", "apple_m3", "arm64", 128, True, 0.819,
         ("fp32", "bf16", "fp16"), ("fp8_e4m3", "mxfp4", "int8", "int4"), 0.62, 0.020, 64, 32),
)}
FLEET_MIX = ("rtx3090-24g", "rtx5090-32g", "rtxpro6000-96g", "dgx-spark-128g", "rtx4090-24g",
             "mac-m3ultra-128g", "rtx5090-32g")


def fleet(D: int, offset: int = 0) -> list[SimDevice]:
    """D devices cycling through the mixed fleet (never uniform for D >= 2)."""
    return [SIM_DEVICES[FLEET_MIX[(offset + i) % len(FLEET_MIX)]] for i in range(D)]


def assign_spans(n_layers: int, devices) -> list[tuple[int, int]]:
    """Contiguous layer spans proportional to device memory, at least one layer each (largest
    remainder). A stand-in for placement (M3) that already treats capacity as GB, not cards."""
    D = len(devices)
    if not 1 <= D <= n_layers:
        raise ValueError(f"{D} devices for {n_layers} layers")
    mem = [d.info.total_mem_bytes for d in devices]
    ideal = [n_layers * m / sum(mem) for m in mem]
    alloc = [max(1, int(x)) for x in ideal]
    while sum(alloc) > n_layers:
        i = max((i for i in range(D) if alloc[i] > 1), key=lambda i: alloc[i] - ideal[i])
        alloc[i] -= 1
    while sum(alloc) < n_layers:
        i = max(range(D), key=lambda i: ideal[i] - alloc[i])
        alloc[i] += 1
    spans, lo = [], 0
    for a in alloc:
        spans.append((lo, lo + a))
        lo += a
    return spans


class SimRuntime:
    """A ToyRuntime dressed as a simulated device. describe() reports the SimDevice's capability
    record and a service curve from its cost model; step() computes (real CPU), then waits the
    modelled compute time on the mesh clock. Everything else delegates to the toy."""

    def __init__(self, inner: ToyRuntime, dev: SimDevice, clock, cost_scale: float = 1.0):
        self._rt, self.dev, self.clock, self.cost_scale = inner, dev, clock, cost_scale
        self.spec = inner.spec

    def __getattr__(self, name):
        return getattr(self._rt, name)

    def cost_s(self, tokens: int) -> float:
        n_layers = self.spec.hi - self.spec.lo
        return self.cost_scale * n_layers * (self.dev.layer_ms + self.dev.token_ms * tokens) / 1000.0

    def describe(self):
        cap = self._rt.describe()
        svc = tuple((1, b, 0, 1000.0 * self.cost_s(b)) for b in (1, 8, 32) if b <= cap.max_tokens_per_step)
        return dataclasses.replace(cap, device=self.dev.info, service_ms=svc, self_probe_ok=True)

    def warmup(self, *a, **kw):
        return self.describe()

    def step(self, x):
        out = self._rt.step(x)
        self.clock.sleep(self.cost_s(sum(s.n for s in x.segs)))
        return out


# ── faults ────────────────────────────────────────────────────────────────────────────────────
class Faults:
    """Per-leg delays and partitions, shared by every FaultyLinks of one mesh."""

    def __init__(self):
        self._lock = threading.Lock()
        self._delay: dict = {}
        self._cut: set = set()

    def set_delay(self, src: str, dst: str, seconds: float) -> None:
        with self._lock:
            self._delay[(src, dst)] = float(seconds)

    def delay(self, src: str, dst: str) -> float:
        with self._lock:
            return self._delay.get((src, dst), 0.0)

    def partition(self, a: str, b: str) -> None:
        with self._lock:
            self._cut |= {(a, b), (b, a)}

    def heal(self) -> None:
        with self._lock:
            self._cut.clear()

    def blocked(self, src: str, dst: str) -> bool:
        with self._lock:
            return (src, dst) in self._cut


class FaultyLinks(TcpLinks):
    """TcpLinks whose delay / partition seams read a Faults table, plus raw-byte injection."""

    def __init__(self, me, resolve, *, faults: Faults, **kw):
        super().__init__(me, resolve, **kw)
        self.faults = faults

    def _delay(self, peer):
        return self.faults.delay(self.me, peer)

    def _blocked(self, peer):
        return self.faults.blocked(self.me, peer)

    def inject_raw(self, peer: str, data: bytes, counted: bool = True) -> None:
        """Write `data` verbatim on the pooled stream to `peer`, in order with real messages.
        counted=False for bytes the receiver cannot finish reading on their own (a truncated
        frame waits for the next message's bytes): they are not in flight on the clock."""
        self._enqueue(peer, data, {"op": "raw", "counted": counted})


def malformed_bytes(how: str, rng: random.Random) -> bytes:
    """Self-delimiting bad input (so the receiver always sees the end of it):
    garbage   valid 8-byte length, random body;
    schema    a valid transport body that is not shard-frame/2 (an unknown frame field);
    oversize  a length claim past MAX_FRAME (refused before any body byte is read);
    truncated claims 16 bytes, sends 5: the NEXT message's bytes complete it into garbage."""
    if how == "garbage":
        body = rng.randbytes(rng.randrange(1, 64))
        return struct.pack("!Q", len(body)) + body
    if how == "schema":
        head = json.dumps({"v": 2, "op": "f", "sid": "0" * 16, "junk": 1}).encode()
        body = struct.pack("!I", len(head)) + head
        return struct.pack("!Q", len(body)) + body
    if how == "oversize":
        return struct.pack("!Q", frame.MAX_FRAME + 1)
    if how == "truncated":
        return struct.pack("!Q", 16) + b"\xff" * 5
    raise ValueError(how)


MALFORMED = ("garbage", "schema", "oversize", "truncated")


class FakeNode(NodeServer):
    """NodeServer + a hang injector (parks the node loop at its pause seam)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._hung = False
        self._gate = self.clock.mailbox(f"{self.peer_id}.gate")

    def _pause(self):
        while self._hung and not self._dead:
            try:
                self._gate.get()
            except Exception:                 # Closed: the node was killed while hung
                return

    def hang(self):
        self._hung = True
        self._inbox.put(("poke",))

    def resume(self):
        self._hung = False
        self._gate.put(True)

    def kill(self):
        super().kill()
        self._gate.close()


# ── what the entry observes ───────────────────────────────────────────────────────────────────
class MeshError(Exception):
    def __init__(self, what: str, t: float, peer: str | None = None):
        super().__init__(f"{what} (peer={peer}, t={t:.6f})")
        self.t, self.peer = t, peer


class Deadline(MeshError):
    """No reply within the deadline."""


class LinkDown(MeshError):
    """The entry could not deliver to `peer` (refused dial after a re-dial)."""


class HopError(MeshError):
    """A hop answered ack{code} for this frame (`peer` = the hop that reported it)."""

    def __init__(self, code: str, t: float, peer: str):
        super().__init__(f"ack {code}", t, peer)
        self.code = code


class Refused(Exception):
    def __init__(self, codes: dict):
        super().__init__(f"lease_open refused: {codes}")
        self.codes = codes


class FakeEntry:
    """The entry's seat reduced to what F4 needs: its own links + a filing inbox."""

    def __init__(self, resolve, clock, faults: Faults):
        self.clock = clock
        self.inbox = clock.mailbox("entry.inbox")
        self.links = FaultyLinks(ENTRY, resolve, faults=faults, clock=clock,
                                 on_message=self._on_msg, on_send_error=self._on_err)
        self.replies: dict = {}          # (sid_hex, seq) -> (peer, reply, t)
        self.errors: dict = {}           # (sid_hex, seq) -> MeshError
        self.acks: dict = {}             # (sid_hex, peer) -> (code, rte, t)
        self.received: list = []         # every reply ever: (peer, reply, t)
        self.unexpected = 0

    def start(self):
        return self.links.start()

    def _on_msg(self, peer, m):
        self.inbox.put(("msg", peer, m, self.clock.now()))

    def _on_err(self, peer, meta, err):
        self.inbox.put(("link", peer, meta, self.clock.now()))

    def _file(self, ev) -> None:
        if ev[0] == "msg":
            _, peer, m, t = ev
            if m["op"] == "r":
                self.replies[(m["sid"], m["seq"])] = (peer, m, t)
                self.received.append((peer, m, t))
            elif m["op"] == "c" and m["cmd"] == "ack":
                if m["seq"] is None:
                    self.acks[(m["sid"], peer)] = (m["code"], m["rte"], t)
                else:
                    self.errors.setdefault((m["sid"], m["seq"]), HopError(m["code"], t, peer))
            else:
                self.unexpected += 1
        else:
            _, peer, meta, t = ev
            if meta.get("op") == "f":
                self.errors.setdefault((meta["sid"], meta["seq"]), LinkDown("undeliverable", t, peer))
            elif meta.get("op") == "c" and meta.get("sid") is not None:
                self.acks.setdefault((meta["sid"], peer), ("link", meta.get("rte"), t))

    def pump_until(self, done, deadline: float) -> bool:
        """File inbound events until done() or `deadline` (clock time)."""
        while not done():
            left = deadline - self.clock.now()
            if left <= 0:
                return False
            try:
                ev = self.inbox.get(timeout=left)
            except Timeout:
                return done()
            self._file(ev)
        return True


# ── the direct loop (the truth) ───────────────────────────────────────────────────────────────
def direct_greedy(weights: ToyWeights, prompt, n_new: int, *, pattern=PATTERN, window=WINDOW,
                  mode="chunk_invariant", max_ctx=512, W=8) -> list[int]:
    """ONE runtime holding every layer, fed the mesh's schedule (the prompt as one prefill
    segment, then one decode row per token), greedy. Returns T = prompt + continuation."""
    L, P = len(pattern), len(prompt)
    rt = ToyRuntime(StageSpec("toy", 0, L, is_head=True, is_tail=True, max_ctx=max_ctx, spec_window=64),
                    pattern=pattern, window=window, hidden=weights.hidden, vocab=weights.vocab,
                    mode=mode, max_tokens_per_step=max(64, P))
    rt.load(weights)
    rt.open(1, max_pos=P + n_new + 1, spec_window=W)
    rt.commit(1, P - 1)
    out = rt.step(StepIn((Segment(1, 0, P, Rows("i32", np.array(prompt, dtype=np.int32)),
                                  kind="prefill", tokens="last"),))).outs[0]
    toks = [int(out.tokens.data[-1])]
    for i in range(n_new - 1):
        pos = P + i
        rt.commit(1, pos)
        out = rt.step(StepIn((Segment(1, pos, 1, Rows("i32", np.array([toks[-1]], dtype=np.int32)),
                                      kind="decode", tokens="last"),))).outs[0]
        toks.append(int(out.tokens.data[0]))
    return list(prompt) + toks


def replay_chain(mesh: "FakeMesh", prompt, n_new: int) -> list[int]:
    """The mesh's schedule replayed WITHOUT the mesh: one runtime per hop span, each fed every
    frame the way the node server runs it (a frame over the hop's token budget in budget-sized
    steps). Under shape-dependent numerics a re-chunking hop changes the bytes, so this, not
    direct_greedy, is what the mesh must equal: "replay of the recorded schedule reproduces
    the bytes" (ARCHITECTURE-v2 §0.6). Returns the continuation."""
    P = len(prompt)
    rts = []
    for i in range(mesh.D):
        rt = mesh.runtime(i)._rt                          # the bare ToyRuntime, no modelled cost
        rt.open(1, max_pos=P + n_new + 1, spec_window=8)
        rts.append(rt)

    def run(pos, ids, cpos, kind):
        x, n = Rows("i32", np.array(ids, dtype=np.int32)), len(ids)
        for i, rt in enumerate(rts):
            rt.commit(1, cpos)
            b, outs = mesh.devices[i].budget, []
            for lo in range(0, n, b):
                hi = min(n, lo + b)
                last = i == mesh.D - 1 and hi == n
                seg = Segment(1, pos + lo, hi - lo, x.slice(lo, hi), tokens="last" if last else "none",
                              kind="prefill" if kind == "P" else ("decode" if n == 1 else "verify"))
                outs.append(rt.step(StepIn((seg,))).outs[0])
            x = Rows.concat([o.hidden for o in outs])
        return int(outs[-1].tokens.data[-1])

    toks = [run(0, prompt, P - 1, "P")]
    for k in range(n_new - 1):
        toks.append(run(P + k, [toks[-1]], P + k, "T"))
    return toks


def reply_tokens(r: dict) -> list[int]:
    return [int(v) for v in r["toks"].numpy()]


# ── the mesh ──────────────────────────────────────────────────────────────────────────────────
def hexsid(sid: int) -> str:
    return f"{sid:016x}"


class FakeMesh:
    """entry -> hop0 -> ... -> hop{D-1} -> entry over loopback TCP. See the module docstring."""

    def __init__(self, D: int = 3, *, clock: str = "virtual", devices=None, pattern: str = PATTERN,
                 window: int = WINDOW, hidden: int = HIDDEN, vocab: int = VOCAB,
                 mode: str = "chunk_invariant", legs_ms=None, deadline: float = 2.0,
                 cost_scale: float = 1.0, max_ctx: int = 512, s_max: int = 16, weights_seed: int = 0,
                 stuck_after: float = 60.0, trace: bool = False, drafters=None):
        self.clock = VirtualClock(stuck_after=stuck_after) if clock == "virtual" else RealClock()
        self.clock.adopt("test")
        self.D, self.pattern, self.window, self.mode = D, pattern, window, mode
        self.devices = list(devices) if devices is not None else fleet(D)
        if len(self.devices) != D:
            raise ValueError("one device per hop")
        self.weights = ToyWeights(pattern, hidden=hidden, vocab=vocab, seed=weights_seed)
        self.spans = assign_spans(len(pattern), self.devices)
        self.deadline, self.cost_scale, self.max_ctx, self.s_max = deadline, cost_scale, max_ctx, s_max
        self.drafters = dict(drafters or {})       # tail drafter plugins (node.py: `drafters`)
        self.faults = Faults()
        self.peers = [f"hop{i}" for i in range(D)]
        self._lock = threading.Lock()
        self._addr: dict = {}
        self.trace_log: list | None = [] if trace else None
        self._seq: dict = {}
        self._sent_at: dict = {}
        self.entry = FakeEntry(self._resolve, self.clock, self.faults)
        self._addr[ENTRY] = self.entry.start()
        self.nodes: list = [None] * D
        for i in range(D):
            self._start_node(i)
        self.set_legs(legs_ms if legs_ms is not None else [0.0] * (D + 1))

    # ── wiring ────────────────────────────────────────────────────────────────────────────
    def _resolve(self, peer):
        with self._lock:
            return self._addr[peer]

    def legs(self) -> list[tuple[str, str]]:
        path = [ENTRY] + self.peers + [ENTRY]
        return list(zip(path, path[1:]))

    def set_legs(self, legs_ms) -> None:
        if len(legs_ms) != self.D + 1:
            raise ValueError(f"{self.D + 1} legs")
        self.legs_ms = [float(x) for x in legs_ms]
        for (src, dst), ms in zip(self.legs(), self.legs_ms):
            self.faults.set_delay(src, dst, ms / 1000.0)

    def _on_trace(self, event, peer, sid, seq, pos, n, t):
        self.trace_log.append((event, peer, sid, seq, pos, n, t))

    def runtime(self, i: int, weights=None) -> SimRuntime:
        dev, (lo, hi) = self.devices[i], self.spans[i]
        spec = StageSpec("toy", lo, hi, is_head=i == 0, is_tail=i == self.D - 1,
                         max_sessions=dev.max_sessions, max_ctx=self.max_ctx, spec_window=64)
        rt = ToyRuntime(spec, pattern=self.pattern, window=self.window, hidden=self.weights.hidden,
                        vocab=self.weights.vocab, mode=self.mode, max_tokens_per_step=dev.budget)
        rt.load(weights or self.weights)
        return SimRuntime(rt, dev, self.clock, self.cost_scale)

    def _start_node(self, i: int, weights=None) -> None:
        node = FakeNode(self.runtime(i, weights), self.peers[i], resolve=self._resolve,
                        clock=self.clock, s_max=self.s_max,
                        links_factory=partial(FaultyLinks, faults=self.faults),
                        trace=self._on_trace if self.trace_log is not None else None,
                        drafters=self.drafters)
        addr = node.start()
        with self._lock:
            self._addr[self.peers[i]] = addr
        self.nodes[i] = node

    def links_of(self, peer: str) -> FaultyLinks:
        return self.entry.links if peer == ENTRY else self.nodes[self.peers.index(peer)].links

    def close(self) -> None:
        for n in self.nodes:
            if n is not None and n.alive:
                n.kill()
        self.entry.links.close(abrupt=True)
        self.entry.inbox.close()
        self.clock.disown()

    # ── sessions ──────────────────────────────────────────────────────────────────────────
    def open(self, sid: int, *, max_pos: int = 256, W: int = 8, expect_ok: bool = True,
             drafter: str | None = None) -> dict:
        """lease_open on every hop; wait for every ack (or the deadline)."""
        h = hexsid(sid)
        for k in [k for k in self.entry.acks if k[0] == h]:
            del self.entry.acks[k]
        for i, peer in enumerate(self.peers):
            head, tail = i == 0, i == self.D - 1
            role = "solo" if head and tail else "head" if head else "tail" if tail else "mid"
            kw = dict(sid=h, rte=1, lo=self.spans[i][0], hi=self.spans[i][1], role=role,
                      pred=ENTRY if head else self.peers[i - 1],
                      next=None if tail else self.peers[i + 1], ret=ENTRY, max_pos=max_pos, W=W,
                      cx="ids" if head else "bf16", nonce=f"{sid:016x}{i:016x}", lease=b"toy-lease")
            if role != "mid":
                kw["salt"] = "5a" * 32
            if tail:
                kw.update(seed=h, ekey="e1" * 32, drafter=drafter)
            self.entry.links.send(peer, frame.control("lease_open", **kw))
        end = self.clock.now() + self.deadline
        self.entry.pump_until(lambda: all((h, p) in self.entry.acks for p in self.peers), end)
        codes = {p: self.entry.acks.get((h, p), ("timeout",))[0] for p in self.peers}
        self._seq[sid] = 0
        if expect_ok and any(c != "ok" for c in codes.values()):
            raise Refused(codes)
        return codes

    def close_session(self, sid: int) -> None:
        for peer in self.peers:
            self.entry.links.send(peer, frame.control("close", sid=hexsid(sid)))

    def submit(self, sid: int, *, kind: str, pos: int, ids, lg: str = "last", cpos: int | None = None,
               sep: int = 0, fen: bool = False, rte: int = 1) -> int:
        seq = self._seq[sid]
        self._seq[sid] = seq + 1
        n = len(ids)
        if cpos is None:
            cpos = pos + n - 1 if kind == "P" else pos
        kw = dict(sid=hexsid(sid), rte=rte, seq=seq, sep=sep, pos=pos, n=n, kind=kind, cpos=cpos,
                  lg=lg, cx="ids")
        if fen:
            kw["fen"] = True
        else:
            kw["x"] = frame.Tensor("i32", (n,), np.asarray(ids, dtype="<i4"))
        self._sent_at[(hexsid(sid), seq)] = self.clock.now()
        self.entry.links.send(self.peers[0], frame.frame(**kw))
        return seq

    def sent_at(self, sid: int, seq: int) -> float:
        return self._sent_at[(hexsid(sid), seq)]

    def collect(self, sid: int, seq: int) -> dict:
        """The reply for (sid, seq), or the first error for it, or Deadline at sent + deadline."""
        key = (hexsid(sid), seq)
        e = self.entry
        e.pump_until(lambda: key in e.replies or key in e.errors, self._sent_at[key] + self.deadline)
        if key in e.replies:
            e.errors.pop(key, None)
            return e.replies.pop(key)[1]
        if key in e.errors:
            raise e.errors.pop(key)
        raise Deadline("no reply", self.clock.now())

    def request(self, sid: int, **kw) -> dict:
        return self.collect(sid, self.submit(sid, **kw))

    def generate(self, sid: int, prompt, n_new: int) -> list[int]:
        """Greedy through the mesh: one prefill frame, then one truth frame per token."""
        return self.generate_many({sid: prompt}, n_new)[sid]

    def generate_many(self, prompts: dict, n_new: int) -> dict:
        """Greedy for many sessions at once: each round submits one frame per session, then
        collects them all, so the hops see concurrent sessions and batch them."""
        out = {sid: [] for sid in prompts}
        for step in range(n_new):
            seqs = {}
            for sid, prompt in prompts.items():
                if step == 0:
                    seqs[sid] = self.submit(sid, kind="P", pos=0, ids=prompt)
                else:
                    seqs[sid] = self.submit(sid, kind="T", pos=len(prompt) + step - 1, ids=[out[sid][-1]])
            for sid, seq in seqs.items():
                r = self.collect(sid, seq)
                P = len(prompts[sid])
                want = (hexsid(sid), 1, seq, 0, 0 if step == 0 else P + step - 1, P if step == 0 else 1)
                got = (r["sid"], r["rte"], r["seq"], r["sep"], r["pos"], r["n"])
                assert got == want and not r["fen"], f"reply header {got} != {want}"
                toks = reply_tokens(r)
                assert len(toks) == 1, toks
                out[sid].append(toks[0])
        return out

    # ── injectors ─────────────────────────────────────────────────────────────────────────
    def kill(self, i: int) -> None:
        self.nodes[i].kill()

    def restart(self, i: int, weights=None) -> None:
        if self.nodes[i].alive:
            self.nodes[i].kill()
        self._start_node(i, weights)

    def hang(self, i: int) -> None:
        self.nodes[i].hang()

    def resume(self, i: int) -> None:
        self.nodes[i].resume()

    def partition(self, a: str, b: str) -> None:
        self.faults.partition(a, b)

    def heal(self) -> None:
        self.faults.heal()

    def malformed(self, src: str, dst: str, how: str = "garbage", rng: random.Random | None = None) -> None:
        self.links_of(dst)                    # dst must exist
        self.links_of(src).inject_raw(dst, malformed_bytes(how, rng or random.Random(0)),
                                      counted=how != "truncated")

    def at(self, dt: float, fn) -> threading.Thread:
        """Run fn() after dt on the mesh clock, from its own actor (a mid-flight injection)."""
        fired = threading.Event()

        def run():
            self.clock.sleep(dt)
            fn()
            fired.set()

        t = self.clock.spawn("injector", run)
        t.fired = fired
        return t

    def bad_frames(self, peer: str) -> int:
        return self.links_of(peer).stats["bad_frames"]

    def rtt(self, n: int = 1) -> float:
        """Modelled round trip of one frame of n rows: every leg + every hop's compute."""
        return sum(self.legs_ms) / 1000.0 + sum(node.rt.cost_s(n) for node in self.nodes)


# ── the fault storm (virtual time) ────────────────────────────────────────────────────────────
FAULT_KINDS = ("kill", "hang", "partition", "malformed")


def fault_storm(mesh: FakeMesh, n_faults: int, *, seed: int = 0, n_prompts: int = 6, n_new: int = 5):
    """n_faults trials. Each opens a fresh session, runs greedy requests, injects ONE fault
    (kind, target, request index, before or mid-flight at a random offset), and checks:
      * every reply ever received (late ones included) carries the oracle's token;
      * no failure is blamed on a fault that had not fired yet (a spurious loss is a violation);
      * the fault surfaces as the failure its kind must produce, at the right place and time:
        kill -> LinkDown(dead head) / HopError(next_down) from the dead hop's predecessor
        (before the request: always, and faster than the deadline), or Deadline if the frame
        died inside the node; hang / partition -> Deadline at exactly sent + deadline, with
        evidence (the partitioned leg blocked a message; the hung node computed at most the step
        it was in; and for every Deadline, the lost frame entered the faulted leg or node and
        never came out the far side);
        malformed -> the receiver's bad_frames += 1, the receiver stays up, at most one Deadline;
      * after recovery (restart / resume / heal) the mesh serves again: the lost position
        re-fed on the same session succeeds (hang, partition, malformed), and every next trial
        opens its lease on every hop (kill).
    Returns {"trials", "outcomes": {kind: {outcome: count}}, "violations": [...]}."""
    if mesh.trace_log is None:
        raise ValueError("fault_storm needs FakeMesh(trace=True): Deadlines are proven from traces")
    rng = random.Random(seed)
    prompts = [[rng.randrange(VOCAB) for _ in range(rng.randrange(3, 9))] for _ in range(n_prompts)]
    Ts = [direct_greedy(mesh.weights, p, n_new + 2, pattern=mesh.pattern, window=mesh.window,
                        mode=mesh.mode) for p in prompts]
    T_of: dict = {}
    outcomes = {k: {} for k in FAULT_KINDS}
    violations: list = []
    checked = 0

    def bad(trial, what):
        violations.append(f"trial {trial}: {what}")

    def check_received():
        nonlocal checked
        rec = mesh.entry.received
        while checked < len(rec):
            peer, r, _ = rec[checked]
            checked += 1
            T = T_of.get(r["sid"])
            toks = reply_tokens(r) if r["toks"] is not None else []
            want = None if T is None else oracle_token(T, r["pos"] + r["n"] - 1)
            if peer != mesh.peers[-1] or not toks or toks[-1] != want:
                bad("*", f"reply {r['sid']}/{r['seq']} pos {r['pos']} from {peer}: {toks} != {want}")

    for trial in range(n_faults):
        sid = 0x5000 + trial
        kind = FAULT_KINDS[trial % len(FAULT_KINDS)] if trial < len(FAULT_KINDS) else rng.choice(FAULT_KINDS)
        pi = rng.randrange(n_prompts)
        prompt, T = prompts[pi], Ts[pi]
        T_of[hexsid(sid)] = T
        legs = mesh.legs()
        hop = rng.randrange(mesh.D)
        leg = legs[rng.randrange(len(legs))]
        how = rng.choice(MALFORMED)
        at_step = rng.randrange(n_new)
        during = rng.random() < 0.5
        offset = rng.uniform(0.0, mesh.rtt(len(prompt) if at_step == 0 else 1))
        base_bad = mesh.bad_frames(leg[1]) if kind == "malformed" else 0
        mesh.trace_log.clear()
        blocked0 = sum(mesh.links_of(p).stats["blocked"] for p in leg)
        inj: dict = {}                        # when the fault really fired, and evidence

        def inject():
            inj["t"] = mesh.clock.now()
            if kind == "kill":
                mesh.kill(hop)
            elif kind == "hang":
                node = mesh.nodes[hop]
                inj["hang0"] = (None, node.snapshot().get("steps", 0))
                mesh.hang(hop)
            elif kind == "partition":
                mesh.partition(*leg)
            else:
                mesh.malformed(leg[0], leg[1], how, rng=random.Random(trial))

        codes = mesh.open(sid, expect_ok=False)
        if any(c != "ok" for c in codes.values()):
            bad(trial, f"mesh not recovered: lease_open {codes}")
            mesh.close_session(sid)
            continue

        def req(step):
            if step == 0:
                return mesh.submit(sid, kind="P", pos=0, ids=prompt)
            p = len(prompt) + step - 1
            return mesh.submit(sid, kind="T", pos=p, ids=[T[p]])

        injector, failure, fail_step, fail_seq = None, None, None, None
        step = 0
        while True:
            if step == at_step:
                if during:
                    injector = mesh.at(offset, inject)
                else:
                    inject()
            if step >= n_new:                 # past the plan: probe until the fault has shown
                if injector is not None and not injector.fired.is_set():
                    mesh.clock.sleep(offset)  # a late mid-flight fault lands first
                if step >= n_new + (1 if kind == "malformed" else 2):
                    break                     # malformed: one probe carries any truncated tail
            seq = req(step)
            try:
                mesh.collect(sid, seq)
            except (Deadline, LinkDown, HopError) as e:
                failure, fail_step, fail_seq = e, step, seq
                break
            step += 1
        if injector is not None and not injector.fired.is_set():
            mesh.clock.sleep(offset)
        if injector is not None:
            injector.join(5.0)
        check_received()

        name = "ok" if failure is None else type(failure).__name__ + (
            f":{failure.code}" if isinstance(failure, HopError) else "")
        outcomes[kind][name] = outcomes[kind].get(name, 0) + 1
        t_sent = mesh.sent_at(sid, fail_seq) if fail_seq is not None else None
        if failure is not None and ("t" not in inj or failure.t < inj["t"]):
            bad(trial, f"{name} at request {fail_step} BEFORE the {kind} fired "
                       f"(planned at request {at_step}): a failure the fault cannot explain")
        elif kind == "kill":
            dead = mesh.peers[hop]
            pred = ENTRY if hop == 0 else mesh.peers[hop - 1]
            if failure is None:
                bad(trial, f"kill of {dead} never surfaced")
            elif isinstance(failure, LinkDown):
                if hop != 0 or failure.peer != dead:
                    bad(trial, f"kill of {dead}: LinkDown names {failure.peer}")
            elif isinstance(failure, HopError):
                if failure.code != "next_down" or failure.peer != pred:
                    bad(trial, f"kill of {dead}: ack {failure.code} from {failure.peer}")
            elif not during:
                bad(trial, f"kill of {dead} before the request surfaced as {name}, not a fast link error")
            if isinstance(failure, (LinkDown, HopError)) and failure.t - t_sent >= mesh.deadline:
                bad(trial, f"kill of {dead}: link error only after {failure.t - t_sent:.3f}s")
        elif kind in ("hang", "partition"):
            if not isinstance(failure, Deadline):
                bad(trial, f"{kind} ({hop if kind == 'hang' else leg}) surfaced as {name}")
            elif abs(failure.t - (t_sent + mesh.deadline)) > 1e-9 and mesh.clock.virtual:
                bad(trial, f"{kind}: Deadline at {failure.t - t_sent:.9f}s, not {mesh.deadline}s")
            if kind == "partition" and sum(mesh.links_of(p).stats["blocked"] for p in leg) == blocked0:
                bad(trial, f"partition {leg}: no message was ever blocked on it")
            if kind == "hang" and "hang0" in inj:
                # hung: at most the step it was in finishes (the lost frame may have arrived
                # before the hang, so arrival is proven by custody below, not by a counter)
                ran = mesh.nodes[hop].snapshot().get("steps", 0) - inj["hang0"][1]
                if ran > 1:
                    bad(trial, f"hang of {mesh.peers[hop]}: computed {ran} steps while hung")
        else:
            if failure is not None and not isinstance(failure, Deadline):
                bad(trial, f"malformed {how} on {leg} surfaced as {name}")
        if isinstance(failure, Deadline) and "t" in inj:
            # Custody evidence: the lost frame (or its reply) entered the faulted leg or node
            # and never came out the far side. A loss anywhere else is not this fault's.
            if kind in ("kill", "hang"):
                src = ENTRY if hop == 0 else mesh.peers[hop - 1]
                dst = ENTRY if hop == mesh.D - 1 else mesh.peers[hop + 1]
            else:
                src, dst = leg
            h = hexsid(sid)

            def traced(ev, peer):
                return any(e[0] == ev and e[1] == peer and e[2] == h and e[3] == fail_seq
                           for e in list(mesh.trace_log))

            left = src == ENTRY or traced("out", src) or traced("reply", src)
            reached = (any(r["sid"] == h and r["seq"] == fail_seq for _, r, _ in mesh.entry.received)
                       if dst == ENTRY else traced("in", dst))
            if not left or reached:
                bad(trial, f"{kind}: request {fail_seq} was lost, but not between {src} and {dst} "
                           f"(left {src}: {left}, reached {dst}: {reached})")

        # recovery, then prove the mesh serves again
        if kind == "kill":
            mesh.restart(hop)
        elif kind == "hang":
            mesh.resume(hop)
        elif kind == "partition":
            mesh.heal()
        if kind != "kill" and failure is not None:
            # Re-feed the lost position on the same session: hops that already computed it
            # rewind (pos >= the committed frontier), the others continue. A lost PREFILL cannot
            # be re-fed (its cpos committed the prompt rows on the hops that ran it), so that
            # recovery is a fresh session, as tier 3 would do.
            rsid = sid
            if fail_step == 0:
                mesh.close_session(sid)
                rsid = sid + 0x100000
                T_of[hexsid(rsid)] = T
                mesh.open(rsid, expect_ok=False)
            try:
                if fail_step == 0:
                    r = mesh.collect(rsid, mesh.submit(rsid, kind="P", pos=0, ids=prompt))
                else:
                    r = mesh.collect(sid, req(fail_step))
                if reply_tokens(r)[-1] != oracle_token(T, r["pos"] + r["n"] - 1):
                    bad(trial, "re-feed after recovery: wrong token")
            except (Deadline, LinkDown, HopError) as e:
                bad(trial, f"re-feed after {kind} recovery failed: {e!r}")
            if rsid != sid:
                mesh.close_session(rsid)
        if kind == "hang" and isinstance(failure, Deadline) and fail_step > 0:
            # A hang delays, it never loses: the resumed node finishes the stuck request, and
            # its late reply lands before the re-feed's (same FIFO path). Else it was lost.
            # (A lost prefill is recovered on a fresh session after closing this one, so its
            # late frame meets closed leases downstream: no late reply is owed there.)
            h = hexsid(sid)
            mesh.clock.settle()
            if not any(r["sid"] == h and r["seq"] == fail_seq for _, r, _ in mesh.entry.received):
                bad(trial, f"hang of {mesh.peers[hop]}: request {fail_seq} never answered after resume")
        if kind == "malformed":
            mesh.clock.settle()
            got = mesh.bad_frames(leg[1]) - base_bad
            dst = leg[1]
            if got != 1:
                bad(trial, f"malformed {how} {leg}: receiver bad_frames += {got}")
            if dst != ENTRY and not mesh.nodes[mesh.peers.index(dst)].alive:
                bad(trial, f"malformed {how} killed {dst}")
        mesh.close_session(sid)
        check_received()
    mesh.clock.sleep(sum(mesh.legs_ms) / 1000.0 + 0.05)     # the last closes cross their legs
    mesh.clock.settle()
    check_received()
    for node in mesh.nodes:                   # every session was closed: nothing leaked
        if node.alive and node._leases:
            bad("end", f"{node.peer_id} still holds leases {sorted(node._leases)[:4]}")
    return {"trials": n_faults, "outcomes": outcomes, "violations": violations,
            "replies_checked": checked}
