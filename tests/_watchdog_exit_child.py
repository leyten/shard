"""Child process for test_l3_real_exit_subprocess (underscore prefix: not collected by pytest).

Drives the REAL shard.coordinate.serve_jobs against a fake ring whose tail mutes on the first
decode frame, with the reply heartbeat DISABLED (M25_REPLY_TIMEOUT=0 -> the decode recv rides the
full 60s job timeout) — so ONLY the stall watchdog (M25_JOB_STALL_S=1) can end this process. The
parent asserts: exit code 1, the stall-watchdog SHARD_JOB_FATAL on stdout, and a wall time far
under the 60s recv budget. os._exit is NOT stubbed here — this is the real kill path.
"""
import argparse
import json
import os
import socket
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))          # repo root: `import shard` (script mode only puts tests/ on the path)

import fake_ring as FR                              # noqa: E402  (bootstraps env + sys.path)
from fake_ring import FakeRing, FakeTok, novel_T    # noqa: E402

import shard.coordinate as C                        # noqa: E402

T = novel_T(400)
c_pipe, r_pipe = socket.socketpair()
c_ret, r_ret = socket.socketpair()
c_ret.settimeout(60)
ring = FakeRing(r_pipe, r_ret, T, mute_after_decode=1)
ring.tail_slack = 8
ring.start()

a = argparse.Namespace(K=8, depth=4, ngram_n=3, max_ctx=0, timeout=60, prefill_chunk=24)
job = json.dumps({"jobId": "j-child", "nonce": "aa" * 16, "maxNew": 32,
                  "messages": [{"role": "user", "content": "fake"}]})
rc = C.serve_jobs(FR.MP, FakeTok(T[:60]), c_pipe, c_ret, a, iter([job]))
# unreachable when the watchdog fires (os._exit): reaching here means it never tripped
print("CHILD_SURVIVED rc=%d" % rc, flush=True)
sys.exit(0)
