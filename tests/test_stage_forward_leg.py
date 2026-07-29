"""The FORWARD leg of a multi-stage chain: real serve() stages wired head -> middle -> tail.

tests/fake_ring.py plays head+tail as ONE thread over a socketpair, so it drives the coordinator
against an oracle and never builds a forward leg at all. Nothing else in the suite runs a CHAIN of
real serve() stages, which is why the failure the 2026-07-29 capstone ring hit (receipt
`docs/receipts/capstone-ring-20260729.json`) had no offline home: a 6-stage ring formed, every stage
reported READY, every socket read healthy, and then EVERY job died on the coordinator's 90s stall
watchdog with zero tokens and 0% GPU, the head logging
`[s0] edge closed (BrokenPipeError); reset + drop forward link` at job start.

THE HOLE those tests could not see: a stage dials its successor at launch — BEFORE it binds its own
listener — and every stage loads weights concurrently, so the first stage to finish loading dials a
successor whose engine is not listening yet. The leg is a sidecar tunnel, which accepts the local
conn regardless, fails to reach the far engine, and closes it (sidecar/main.go runForward /
runInbound). The engine never READS that leg, so the socket stays "connected" through the rest of
the pull. The FIRST WRITE of the next job is what discovers it, that write is the RESET, and
dropping it there is unrecoverable: the coordinator reads the TAIL, so no error can reach it and the
job can only time out.

Run: python3 -m pytest tests/test_stage_forward_leg.py -q
"""
import os
import socket
import struct
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
fr = pytest.importorskip("fake_ring")               # bootstraps a fake M25_DIR + m25_pipe on CPU

MP = fr.MP
send_msg, recv_msg = fr.send_msg, fr.recv_msg

H = 8                                               # fake hidden size / vocab: nothing here computes
VOCAB = 32
TOK = "f" * 32                                      # every real ring runs C2 token auth — so does this


class _FakeLayer:
    def reset(self):
        pass


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _dial(port, timeout=5):
    s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    s.settimeout(timeout)
    return s


@pytest.fixture
def spawn(monkeypatch):
    """Start real serve() stages with the model ripped out — fake layers, identity block, zero
    logits. Everything these tests assert on is socket lifecycle, so no weights and no compute.

    Readiness comes from stubbing `_probe_listener`, which serve() calls on the line after
    `srv.bind()`/`listen()`. That is an exact, race-free signal: probing by BINDING the port would
    race the stage's own bind and kill it, and probing by CONNECTING would put a stray conn through
    the stage's greeting classification."""
    ready = set()

    def _probe_listener(probe_port):
        ready.add(probe_port - 3)                   # serve(): probe_port defaults to engine port + 3
        return None                                 # None = probing disabled, the fail-closed default

    def _load(stage, nstages, lo, hi):
        parts = {"layers": [_FakeLayer() for _ in range(lo, hi)],
                 "head": stage == 0, "tail": stage == nstages - 1}
        if parts["head"]:
            parts["embed_w"] = torch.zeros(VOCAB, H)
        return parts

    monkeypatch.delenv("SHARD_PROBE_PORT", raising=False)
    monkeypatch.setattr(MP, "_probe_listener", _probe_listener)
    monkeypatch.setattr(MP, "_load", _load)
    monkeypatch.setattr(MP, "_block", lambda grs, layers, start, x, vcfg: x)
    monkeypatch.setattr(MP, "_tail_logits", lambda h, parts: torch.zeros(h.shape[0], h.shape[1], VOCAB))
    monkeypatch.setattr(MP, "dev", "cpu")
    monkeypatch.setattr(MP, "RECEIPTS", False)
    monkeypatch.setattr(MP, "SWARM_TOKEN", TOK)
    monkeypatch.setattr(MP.S, "_CTX", (None, None), raising=False)
    monkeypatch.setattr(MP.S, "M25_EAGLE", False, raising=False)
    monkeypatch.setattr(MP.S, "M25_STAGE_TIMING", False, raising=False)

    def _spawn(stage, nstages, nxt, timeout=5):
        """One stage, listening. It dials `nxt` at launch and that dial is strict, so `nxt` must
        already be up — build a chain TAIL-FIRST, exactly like a real ring forms."""
        port = _free_port()
        threading.Thread(target=MP.serve, args=(stage, nstages, stage, stage + 1, port, nxt, timeout),
                         daemon=True).start()
        deadline = time.monotonic() + 20
        while port not in ready:
            assert time.monotonic() < deadline, f"stage {stage} never came up on :{port}"
            time.sleep(0.01)
        return port

    return _spawn


def _open_return(tail_port):
    """The coordinator's return leg. _tail_accept acks it the moment it classifies the greeting, so
    this needs no predecessor to exist yet."""
    ret = _dial(tail_port)
    send_msg(ret, {"op": "hello_return", "token": TOK})
    assert recv_msg(ret) == "ret_ok"
    return ret


def _open_pipe(head_port):
    """The coordinator's forward leg into the head (shard/coordinate.connect_ring)."""
    pipe = _dial(head_port)
    send_msg(pipe, {"op": "hello_pred", "token": TOK})
    return pipe


class _FlakyTunnel(threading.Thread):
    """The sidecar forward tunnel, in its one behavior that matters here: it accepts the engine's
    connection before it has a path to the successor, takes the engine's greeting, and then closes
    that connection when it gives up (RST, so the engine's next write fails on the spot rather than
    one frame later). Every conn after the first is tunnelled straight through — the successor came
    up meanwhile, which is the state the capstone ring was in when its jobs started dying."""

    def __init__(self, port, upstream_port):
        super().__init__(daemon=True)
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", port))
        self.srv.listen(8)
        self.port = port
        self.upstream = upstream_port
        self.dropped = threading.Event()
        self._socks = []

    def run(self):
        first = True
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            self._socks.append(c)
            if first:
                first = False
                threading.Thread(target=self._drop, args=(c,), daemon=True).start()
            else:
                threading.Thread(target=self._pipe, args=(c,), daemon=True).start()

    def _drop(self, c):
        """Take the engine's launch-time greeting first — its boot dial is strict, and killing the
        conn underneath that write would fail the STAGE instead of the job this test is about."""
        c.settimeout(10)
        try:
            c.recv(65536)
        except OSError:
            pass
        try:
            c.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            c.close()                                # RST: "no activation stream to the successor"
        except OSError:
            pass
        self.dropped.set()

    def _pipe(self, c):
        try:
            u = socket.create_connection(("127.0.0.1", self.upstream))
        except OSError:
            c.close()
            return
        self._socks.append(u)
        for a, b in ((c, u), (u, c)):
            threading.Thread(target=self._copy, args=(a, b), daemon=True).start()

    @staticmethod
    def _copy(a, b):
        try:
            while True:
                d = a.recv(65536)
                if not d:
                    break
                b.sendall(d)
        except OSError:
            pass
        for s in (a, b):
            try:
                s.close()
            except OSError:
                pass

    def close(self):
        for s in [self.srv] + self._socks:
            try:
                s.close()
            except OSError:
                pass


def test_three_stage_chain_carries_a_job(spawn):
    """Coverage first: a real head -> middle -> tail chain carries reset / verify / receipt and every
    reply comes back on the coordinator's return channel. Without this the suite has no forward leg."""
    p2 = spawn(2, 3, "127.0.0.1:1")                  # tail-first: each stage dials its successor at boot
    ret = _open_return(p2)
    p1 = spawn(1, 3, f"127.0.0.1:{p2}")
    p0 = spawn(0, 3, f"127.0.0.1:{p1}")
    pipe = _open_pipe(p0)

    send_msg(pipe, {"op": "reset", "swarm_id": "s", "job_id": "j"})
    assert recv_msg(ret) == "ok"
    send_msg(pipe, {"op": "verify", "token_ids": [1, 2, 3], "start": 0})
    assert recv_msg(ret) == [0, 0, 0]
    send_msg(pipe, {"op": "receipt"})
    assert recv_msg(ret) == []
    for s in (ret, pipe):
        s.close()


def test_reset_survives_a_forward_leg_that_died_while_idle(spawn):
    """THE CAPSTONE-RING FAILURE, offline. The head's forward leg is killed while the ring sits idle
    between launch and the first job. The head cannot learn that from a socket it never reads, so the
    job's reset is the write that discovers it — and the coordinator, which reads the TAIL, can only
    sit there until its stall watchdog fires. The reset must reach the tail anyway."""
    p2 = spawn(2, 3, "127.0.0.1:1")
    ret = _open_return(p2)
    p1 = spawn(1, 3, f"127.0.0.1:{p2}")
    tun = _FlakyTunnel(_free_port(), p1)
    tun.start()
    p0 = spawn(0, 3, f"127.0.0.1:{tun.port}")        # the head's leg is the tunnel, not stage 1
    assert tun.dropped.wait(15), "the tunnel never dropped the head's pre-connect"
    pipe = _open_pipe(p0)

    send_msg(pipe, {"op": "reset", "swarm_id": "s", "job_id": "j"})
    ret.settimeout(20)
    try:
        ack = recv_msg(ret)
    except (OSError, EOFError) as e:                 # what the ring did: nothing ever came back
        pytest.fail(f"the ring never acked the job's reset ({type(e).__name__}) — the head dropped "
                    f"the frame when its idle forward leg turned out to be dead")
    assert ack == "ok"
    send_msg(pipe, {"op": "verify", "token_ids": [1, 2, 3], "start": 0})
    assert recv_msg(ret) == [0, 0, 0]                # and the rebuilt leg carries the job, not just the ack
    for s in (ret, pipe):
        s.close()
    tun.close()
