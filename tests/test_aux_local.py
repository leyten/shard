"""CPU tests for the head-local aux lane primitives (_aux_local_handshake / _pull_aux_local) over
REAL sockets — the timeout/degrade/drain behaviors the fake-ring e2e can't exercise (its queue stub
has no socket deadlines). The contract: an old head that never acks must DEGRADE the job to the
ridden-ring path within wait_s (never stall); stale frames from a dead job must be drained; a
present-but-mispaired frame must abort LOUD.

Run: python3 -m pytest tests/test_aux_local.py -q
"""
import os
import socket
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
fr = pytest.importorskip("fake_ring")               # bootstraps env + imports m25_pipe on CPU

MP = fr.MP
from node_kv import send_msg, TransportError        # noqa: E402  (the real codec)


def pair():
    a, b = socket.socketpair()
    a.settimeout(5.0); b.settimeout(5.0)
    return a, b


def test_handshake_ok_and_stale_drain():
    coord, head = pair()
    send_msg(head, {"op": "aux_local", "job": "dead", "seq": 9, "aux": {}})   # dead job's leftovers
    send_msg(head, {"op": "aux_local_ok", "job": "dead"})
    send_msg(head, {"op": "aux_local_ok", "job": "j1"})
    assert MP._aux_local_handshake(coord, "j1", wait_s=2.0) is True
    assert coord.gettimeout() == 5.0                # socket deadline restored after the handshake


def test_handshake_old_head_degrades_within_deadline():
    import time
    coord, head = pair()                            # head never speaks (old build)
    t0 = time.monotonic()
    assert MP._aux_local_handshake(coord, "j1", wait_s=0.4) is False
    assert time.monotonic() - t0 < 2.0              # bounded degrade, never a stall
    assert coord.gettimeout() == 5.0


def test_handshake_dead_socket_degrades():
    coord, head = pair()
    head.close()
    assert MP._aux_local_handshake(coord, "j1", wait_s=0.4) is False


def test_pull_pairs_and_aborts_loud():
    coord, head = pair()
    a1 = (torch.randn(2, 3, 8)).to(torch.bfloat16)
    send_msg(head, {"op": "aux_local", "job": "j1", "seq": 0, "aux": {"1": a1}})
    aux = MP._pull_aux_local(coord, "j1", 0)
    assert torch.equal(aux["1"], a1)
    send_msg(head, {"op": "aux_local", "job": "j1", "seq": 2, "aux": {}})     # seq skew (1 expected)
    with pytest.raises(TransportError, match="aux_local pairing broken"):
        MP._pull_aux_local(coord, "j1", 1)
    send_msg(head, {"op": "aux_local", "job": "OTHER", "seq": 2, "aux": {}})  # wrong job
    with pytest.raises(TransportError, match="aux_local pairing broken"):
        MP._pull_aux_local(coord, "j1", 2)


def test_pull_skips_keepwarm_noops():
    coord, head = pair()                            # recv_data's noop-skip must hold on this lane too
    send_msg(head, {"op": "noop"})
    send_msg(head, {"op": "aux_local", "job": "j1", "seq": 0, "aux": {}})
    assert MP._pull_aux_local(coord, "j1", 0) == {}


def test_abort_drain_unblocks_the_lane():
    """The review's F1: an aborted armed job with unpulled frames must drain them so the head is
    never left blocked mid-send and the reused socket starts the next job clean."""
    coord, head = pair()
    for k in range(3):                              # 3 in-flight frames the aborted job never pulled
        send_msg(head, {"op": "aux_local", "job": "t0", "seq": k, "aux": {}})
    MP._drain_aux_local(coord, 3, wait_s=0.5)
    send_msg(head, {"op": "aux_local_ok", "job": "t1"})     # next job's handshake sees a CLEAN lane
    assert MP._aux_local_handshake(coord, "t1", wait_s=1.0) is True
    assert coord.gettimeout() == 5.0                # drain restored the socket deadline


def test_drain_tolerates_quiet_and_dead_lane():
    coord, head = pair()
    MP._drain_aux_local(coord, 2, wait_s=0.2)       # nothing buffered: stops early, no raise
    head.close()
    MP._drain_aux_local(coord, 2, wait_s=0.2)       # dead peer: absorbed, no raise
