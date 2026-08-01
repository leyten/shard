"""What one V4 ring hop actually costs, in bytes, bf16 wire against fp8 wire (V4_FP8_WIRE).

Run it to re-derive the arithmetic on any box — it builds REAL frames with the engine's own
_make_step_frame and measures them through the REAL transport codec rather than multiplying shapes
on paper, so the JSON header, the per-blob length prefixes and the frame length prefix are all
counted where they fall.

    python3 phase0/v4_wire_bench.py                     # the shipped config (dim 4096, hc_mult 4)
    python3 phase0/v4_wire_bench.py --dir /root/v4      # a checkpoint's own config.json

WHY THIS IS WORTH A TOOL. V4's inter-stage payload is `h [b, s, hc_mult, dim]` — the hyper-
connections keep FOUR residual streams per token all the way down the stack and only collapse at
hc_head — so a hop moves 4x what a normal pipeline-parallel model would, 32 KiB per token per hop at
dim 4096. On a scattered WAN ring that is the dominant cost, and consumer uplinks fall off a cliff
as frames get fat (measured 2026-07: 131 KB frames at 48.7 MB/s, 1160 KB frames at 0.2 MB/s), so
halving the bytes is worth more than the 2x suggests. The two numbers that matter are at opposite
ends of the size range and want reporting separately: the s=1 pipelined DECODE frame, which is the
steady-state path, and the PREFILL frame, which is one frame the size of the whole prompt.
"""
import argparse
import json
import os
import struct
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import torch  # noqa: E402

try:
    import transport as T
except ImportError:
    from shard import transport as T
import v4_pipe as VP  # noqa: E402


def _wire_len(frame):
    """Bytes this frame puts on the wire — the same arithmetic send_msg itself returns (an 8 B
    length prefix over the packed segments). No PSK seal: the V4 ring runs on shard/transport.py,
    where the libp2p sidecar provides encryption, NOT on phase0/wire.py's ChaCha20-Poly1305 path."""
    return 8 + sum(len(p) for p in T._pack_parts(frame))


def _header(frame):
    return 4 + struct.unpack_from("!I", T._pack_parts(frame)[0], 0)[0]


def _frames(s, dim, hc):
    """The bf16 and fp8 forms of ONE step frame carrying s positions, built by the engine itself."""
    h = torch.zeros(1, s, hc, dim, dtype=torch.bfloat16)
    ids = torch.zeros(1, s, dtype=torch.int64)
    was = VP.V4_FP8_WIRE
    try:
        VP.V4_FP8_WIRE = False
        bf16, _ = VP._make_step_frame(h, ids, 0, None)
        VP.V4_FP8_WIRE = True
        fp8, _ = VP._make_step_frame(h, ids, 0, None)
    finally:
        VP.V4_FP8_WIRE = was
    return bf16, fp8


def report(dim, hc, block, max_seq, out=print):
    out(f"V4 wire bytes per hop — dim {dim}, hc_mult {hc}, dspark_block_size {block}, "
        f"max_seq {max_seq}")
    out(f"payload/token: bf16 h {hc*dim*2:,} B | fp8 h {hc*dim:,} B | scales {hc*2:,} B "
        f"| ids {8} B")
    out("")
    out(f"{'frame':>30} {'s':>6} {'bf16 B':>14} {'fp8 B':>14} {'saved':>14} {'x':>6}")
    rows = [("prefill, whole prompt", max_seq), ("prefill, 2048-token prompt", 2048),
            (f"dspark verify chunk (k+1)", block + 1), ("decode, s=1 (pipelined)", 1)]
    for label, s in rows:
        if s <= 0:
            continue
        b, f = _frames(s, dim, hc)
        nb, nf = _wire_len(b), _wire_len(f)
        cap = "" if nb <= T.MAX_FRAME else "  <-- OVER MAX_FRAME on bf16"
        out(f"{label:>30} {s:>6} {nb:>14,} {nf:>14,} {nb-nf:>14,} {nb/nf:>6.3f}{cap}")
    b, _f = _frames(1, dim, hc)
    out("")
    out(f"framing overhead on the s=1 decode frame: header {_header(b)} B, frame total "
        f"{_wire_len(b):,} B, of which payload {hc*dim*2 + 8:,} B "
        f"-> overhead {_wire_len(b) - (hc*dim*2 + 8)} B "
        f"({(_wire_len(b) - (hc*dim*2+8)) / _wire_len(b) * 100:.2f}%)")
    out(f"MAX_FRAME is {T.MAX_FRAME:,} B; the largest prefill that fits in ONE frame is "
        f"{max_single_frame_prefill(dim, hc, False):,} positions on bf16, "
        f"{max_single_frame_prefill(dim, hc, True):,} on fp8")


def max_single_frame_prefill(dim, hc, fp8):
    """The longest prompt that still fits in ONE step frame. Frame length is affine in s, so it is
    read off two MEASURED points rather than by bisecting (which would try to allocate a 16 GiB
    probe tensor) or by multiplying shapes on paper (which is what loses the ids and the header —
    and losing them is exactly why the bf16 answer lands just UNDER 8192, not at it)."""
    i = 1 if fp8 else 0
    a, b = 1000, 2000                                    # both 4-digit, so the JSON header is fixed
    la = _wire_len(_frames(a, dim, hc)[i])
    lb = _wire_len(_frames(b, dim, hc)[i])
    per = (lb - la) // (b - a)
    return (T.MAX_FRAME - (la - a * per)) // per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="checkpoint dir whose config.json to size")
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--hc-mult", type=int, default=4)
    ap.add_argument("--block", type=int, default=5, help="dspark_block_size")
    ap.add_argument("--max-seq", type=int, default=VP.V4_MAX_SEQ)
    a = ap.parse_args()
    if a.dir:
        with open(os.path.join(a.dir, "config.json")) as f:
            c = json.load(f)
        a.dim, a.hc_mult = c.get("dim", a.dim), c.get("hc_mult", a.hc_mult)
        a.block = c.get("dspark_block_size", a.block)
    report(a.dim, a.hc_mult, a.block, a.max_seq)


if __name__ == "__main__":
    main()
