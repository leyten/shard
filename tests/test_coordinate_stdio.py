"""The coordinator driven the way the c0mpute node daemon actually drives it (leg 8, node half).

test_shard_coordinate.py calls serve_jobs() in-process with a list iterator, so it can never see
anything the REAL stdin path does. The daemon's CoordinatorProcess spawns
`python -m shard.coordinate` with node's `stdio: ['pipe','pipe','pipe']` — which is socketpair(2)
on POSIX, not pipe(2) — writes one NDJSON job line per swarm:job, and waits on the stdout contract.
This drives exactly that shape (subprocess + socketpair stdio + the daemon's job payload) against
the CPU fake ring, and asserts a job submitted after SHARD_COORD_READY streams tokens and completes.

The 2026-07-28 stranger-serve ring (docs/receipts/stranger-serve-20260728.json, bug S1) was read as
a coordinator that "accepts jobs it never executes". This harness is the control that rules the
stdin path out: the same spawn shape, the same payload, over a socketpair, serves.

(The live head showed `wchan=do_poll` with flat CPU. That narrows the state to "blocked in a TIMED
socket recv" — CPython routes those through poll(2) — which is coordinate_pipe's reset ack or
connect_ring's ret_ok, but NOT stdin, which blocks in `unix_stream_data_wait`. It does not
distinguish the two ring waits from each other; `/proc/PID/syscall` would, since read(2)'s arg0 is
the fd.)
"""
import json
import os
import select
import socket
import subprocess
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fake_ring as FR                                    # noqa: E402  (bootstraps env + sys.path)
from fake_ring import FakeRing, repetitive_T             # noqa: E402

TESTS = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(TESTS, "coordinate_stdio_driver.py")


@pytest.fixture(autouse=True)
def _libp2p_codec(monkeypatch):
    """Pin BOTH ends of the ring sockets to the libp2p codec.

    node_kv binds send_msg/recv_msg at import time from SHARD_TRANSPORT. The driver subprocess is a
    fresh interpreter where fake_ring sets that env first, so it always speaks libp2p — but in THIS
    process node_kv may already have been imported under the PSK wire by an earlier test module, and
    then FakeRing would answer the child in a different codec on the same socket. Every other
    fake-ring test lives inside one process and cannot see this; only a cross-process one can."""
    from shard import transport as t                      # noqa: PLC0415 — the codec the child uses
    monkeypatch.setattr(FR, "send_msg", t.send_msg)
    monkeypatch.setattr(FR, "recv_msg", t.recv_msg)


class CoordProc:
    """`python <driver>` spawned with SOCKETPAIR stdio — byte-for-byte the daemon's spawn shape."""

    def __init__(self, head, tail, **kw):
        self.p_in, c_in = socket.socketpair()
        self.p_out, c_out = socket.socketpair()
        self.p_err, c_err = socket.socketpair()
        args = [sys.executable, DRIVER, "--head", head, "--tail", tail]
        for k, v in kw.items():
            args += ["--" + k.replace("_", "-"), str(v)]
        self.proc = subprocess.Popen(args, stdin=c_in.fileno(), stdout=c_out.fileno(),
                                     stderr=c_err.fileno(), close_fds=True, cwd=TESTS)
        for s in (c_in, c_out, c_err):
            s.close()
        self._buf = b""
        self.stderr = []
        threading.Thread(target=self._drain_err, daemon=True).start()

    def _drain_err(self):
        while True:
            try:
                b = self.p_err.recv(65536)
            except OSError:
                return
            if not b:
                return
            self.stderr.append(b.decode(errors="replace"))

    def submit(self, job):
        """CoordinatorProcess.submit: one JSON line written to the socketpair stdin."""
        self.p_in.sendall((json.dumps(job) + "\n").encode())

    def readline(self, deadline):
        """Next complete stdout line, or None once `deadline` passes (never blocks past it)."""
        while True:
            nl = self._buf.find(b"\n")
            if nl >= 0:
                line, self._buf = self._buf[:nl], self._buf[nl + 1:]
                return line.decode(errors="replace").strip()
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            if not select.select([self.p_out], [], [], min(left, 0.5))[0]:
                continue
            b = self.p_out.recv(65536)
            if not b:
                return None
            self._buf += b

    def collect(self, until_tag, budget):
        """Contract lines until `until_tag` (or the budget runs out): [(tag, fields), ...]."""
        deadline = time.monotonic() + budget
        out = []
        while True:
            line = self.readline(deadline)
            if line is None:
                return out
            if not line.startswith("SHARD_"):
                continue
            tag, _, rest = line.partition(" ")
            try:
                out.append((tag, json.loads(rest)))
            except ValueError:
                out.append((tag, {}))
            if tag == until_tag:
                return out

    def close(self):
        try:
            self.proc.kill()
            self.proc.wait(10)
        except Exception:                            # noqa: BLE001 — teardown is best-effort
            pass
        for s in (self.p_in, self.p_out, self.p_err):
            try: s.close()
            except OSError: pass


def _listener():
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    return srv, "127.0.0.1:%d" % srv.getsockname()[1]


def _serve_ring(head_srv, ret_srv, T, ready):
    """Play the ring the coordinator dials: the head engine's listener (it recvs job frames) and
    the head sidecar's return -forward to the tail (it sends replies + the ret_ok ack)."""
    head_srv.settimeout(30)
    ret_srv.settimeout(30)
    pipe, _ = head_srv.accept()
    ret, _ = ret_srv.accept()
    for s in (pipe, ret):
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    if os.environ.get("SHARD_SWARM_TOKEN"):
        FR.recv_msg(pipe)                            # hello_pred (C2 identity-bound greeting)
    FR.recv_msg(ret)                                 # hello_return
    FR.send_msg(ret, "ret_ok")                       # what m25_pipe._tail_accept acks with
    ring = FakeRing(pipe, ret, T)
    ring.tail_slack = 4
    ring.start()
    ready.append(ring)
    return ring


def _job(job_id="j-stdio-1", max_new=32):
    """The daemon's swarm:job -> CoordinatorProcess.submit payload (shard-worker.ts:484-509);
    `tools` is absent because JSON.stringify drops undefined. reasoning:False because the
    synthetic ring's stream carries no think block — a defaulted (reasoning:True) job would
    correctly withhold the whole stream as chain-of-thought (see the reasoning-split tests in
    test_shard_coordinate.py); the daemon sends the flag through since the think-split fix."""
    return {"jobId": job_id, "swarmId": "sw-stdio", "nonce": "ab" * 32,
            "messages": [{"role": "user", "content": "fake"}], "maxNew": max_new,
            "reasoning": False}


def test_job_submitted_on_socketpair_stdin_is_served():
    """RED for stranger-serve S1: a job written to the coordinator's stdin the way the daemon
    writes it must reach the ring and stream tokens — not sit in the process unread."""
    head_srv, head = _listener()
    ret_srv, ret = _listener()
    T = repetitive_T(600)
    ring_box = []
    th = threading.Thread(target=_serve_ring, args=(head_srv, ret_srv, T, ring_box), daemon=True)
    th.start()
    coord = CoordProc(head, ret, prompt_len=40, target_len=600, timeout=30, connect_retry=30)
    try:
        boot = coord.collect("SHARD_COORD_READY", 120)
        assert any(t == "SHARD_COORD_READY" for t, _ in boot), \
            f"coordinator never reported READY: {boot} / stderr={''.join(coord.stderr)[-2000:]}"
        coord.submit(_job(max_new=32))
        got = coord.collect("SHARD_JOB_DONE", 120)
        tags = [t for t, _ in got]
        assert "SHARD_JOB_DONE" in tags, (
            "job written to the coordinator's stdin never completed — the daemon's serve gap "
            f"(S1). lines={got} stderr={''.join(coord.stderr)[-3000:]}")
        done = [f for t, f in got if t == "SHARD_JOB_DONE"][0]
        assert done["ok"] and done["jobId"] == "j-stdio-1"
        assert done["tokensGenerated"] == 32
        deltas = [f["delta"] for t, f in got if t == "SHARD_JOB_TOKEN"]
        assert deltas and "".join(deltas) == done["response"]
        # the daemon-visible proof that the line left stdin, emitted BEFORE the ring is touched
        assert tags.index("SHARD_JOB_START") == 0 and got[0][1]["jobId"] == "j-stdio-1"
    finally:
        coord.close()
        for s in (head_srv, ret_srv):
            try: s.close()
            except OSError: pass


def test_two_jobs_in_sequence_on_socketpair_stdin():
    """The daemon keeps ONE long-lived coordinator per swarm: the second swarm:job of a session
    must serve too (the reader stays armed after a completed job)."""
    head_srv, head = _listener()
    ret_srv, ret = _listener()
    T = repetitive_T(600)
    ring_box = []
    th = threading.Thread(target=_serve_ring, args=(head_srv, ret_srv, T, ring_box), daemon=True)
    th.start()
    coord = CoordProc(head, ret, prompt_len=40, target_len=600, timeout=30, connect_retry=30)
    try:
        assert any(t == "SHARD_COORD_READY" for t, _ in coord.collect("SHARD_COORD_READY", 120))
        for n in (1, 2):
            coord.submit(_job(job_id=f"j-seq-{n}", max_new=16))
            got = coord.collect("SHARD_JOB_DONE", 120)
            done = [f for t, f in got if t == "SHARD_JOB_DONE"]
            assert done, (f"job j-seq-{n} never completed: {got} "
                          f"stderr={''.join(coord.stderr)[-3000:]}")
            assert done[0]["jobId"] == f"j-seq-{n}" and done[0]["tokensGenerated"] == 16
    finally:
        coord.close()
        for s in (head_srv, ret_srv):
            try: s.close()
            except OSError: pass
