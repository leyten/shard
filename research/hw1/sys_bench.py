"""HW1 part C: system costs on the box AND on the entry host (BUILD-PLAN §4.1 C, gate K8).

  sha256      rows/s at 8 / 12 / 30 KiB (receipt leaf hashing; MiMo hidden 4096 bf16 = 8 KiB/row)
  frame       shard-frame/2 encode + decode per message at n = 1 / 9 / 2048 (hidden 4096, bf16)
  copy        pinned D2H / H2D GB/s (CUDA only; recorded as skipped on CPU)

Runs fully on CPU. `python research/hw1/sys_bench.py --out results/c_<host>.json`
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import hw1_common as C  # noqa: E402


def sha_rows(res, sizes_kib, rows, warmup, iters):
    for kib in sizes_kib:
        blob = [random.Random(kib * 31 + i).randbytes(kib * 1024) for i in range(rows)]
        st = C.time_fn(lambda: [hashlib.sha256(b).digest() for b in blob], warmup=warmup, iters=iters,
                       device="cpu")
        res.record("C", "sha256", {"kib": kib, "rows": rows}, st, rows_per_s=rows / st["ms"] * 1e3,
                   us_per_row=st["ms"] * 1e3 / rows)


def frame_codec(res, ns, hidden, warmup, iters):
    from shard import frame as F
    for n in ns:
        x = F.Tensor("bf16", (n, hidden), random.Random(n).randbytes(n * hidden * 2))
        m = F.frame(sid="0123456789abcdef", rte=3, seq=17, sep=5, pos=100, lg="last", n=n, kind="T",
                    cpos=99, cx="bf16", x=x)

        def rt():
            body = b"".join(F.encode(m))
            F.decode(body)
            return body
        size = len(rt())
        st = C.time_fn(rt, warmup=warmup, iters=iters, device="cpu")
        res.record("C", "frame_rt", {"n": n, "hidden": hidden}, st, bytes=size,
                   us_per_msg=st["ms"] * 1e3, mb_per_s=size / st["ms"] / 1e3)


def pinned_copy(res, mbs, warmup, iters):
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        cuda = False
    for mb in mbs:
        prm = {"mb": mb}
        if not cuda:
            res.record("C", "d2h", prm, skipped="no cuda")
            res.record("C", "h2d", prm, skipped="no cuda")
            continue
        n = mb << 20
        dv = torch.empty(n, dtype=torch.uint8, device="cuda")
        hb = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        for op, fn in (("d2h", lambda: hb.copy_(dv, non_blocking=True)),
                       ("h2d", lambda: dv.copy_(hb, non_blocking=True))):
            st = C.time_fn(fn, warmup=warmup, iters=iters, device="cuda")
            res.record("C", op, prm, st, gb_per_s=n / st["ms"] / 1e6)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--kib", default="8,12,30")
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--ns", default="1,9,2048")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--mbs", default="16,128,512")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    ints = lambda s: [int(x) for x in s.split(",") if x]
    res = C.Results("sys_bench", "sys", device=a.device, stub=a.stub, argv=argv or sys.argv[1:])
    sha_rows(res, ints(a.kib), a.rows, a.warmup, a.iters)
    frame_codec(res, ints(a.ns), a.hidden, a.warmup, a.iters)
    pinned_copy(res, ints(a.mbs), a.warmup, a.iters)
    res.write(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
