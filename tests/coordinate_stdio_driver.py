"""Subprocess driver that reproduces `shard.coordinate.main()`'s serving tail EXACTLY, minus the
GPU: real connect_ring over real TCP, real _emit to the real sys.stdout, real serve_jobs over the
real sys.stdin, real m25_pipe.coordinate_pipe — only the tokenizer (FakeTok) and the ring peers
(tests/fake_ring.FakeRing, spoken by the parent) are stand-ins.

Exists so tests can drive the coordinator the way the c0mpute daemon's CoordinatorProcess does:
spawned with socketpair stdio (node's `stdio: ['pipe','pipe','pipe']` is socketpair(2), not
pipe(2)) and fed NDJSON job lines on stdin. The in-process tests hand serve_jobs a list iterator,
which cannot see anything the real stdin path does.

Not a test itself; tests/test_coordinate_stdio.py spawns it.
"""
import argparse
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_TESTS)
for _p in (_REPO, _TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fake_ring as FR                                    # noqa: E402  (bootstraps env + sys.path)
from fake_ring import FakeTok, repetitive_T               # noqa: E402

import shard.coordinate as C                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--tail", required=True)
    ap.add_argument("--prompt-len", type=int, default=40)
    ap.add_argument("--target-len", type=int, default=600)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--connect-retry", type=int, default=30, dest="connect_retry")
    d = ap.parse_args()

    # main()'s argparse defaults verbatim (the daemon passes only --head/--tail/--dir/--receipts)
    a = argparse.Namespace(head=d.head, tail=d.tail, dir=None, K=8, depth=4, ngram_n=3,
                           max_ctx=131072, prefill_chunk=4096, timeout=d.timeout,
                           connect_retry=d.connect_retry, receipts=False, check=False)

    tok = FakeTok(repetitive_T(d.target_len)[:d.prompt_len])
    pipe, ret = C.connect_ring(FR.MP, a.head, a.tail, a.timeout, retry_s=a.connect_retry)
    C._emit("SHARD_COORD_READY", head=a.head, tail=a.tail, eagle=FR.MP.eagle_armed(), receipts=False)
    return C.serve_jobs(FR.MP, tok, pipe, ret, a, sys.stdin)


if __name__ == "__main__":
    sys.exit(main())
