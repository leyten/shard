"""shard-frame/2: the v2 message schema (frame / reply / control) on top of shard/transport.

This module defines WHAT a v2 message says, not how bytes cross a link. The link is the unchanged
`shard/transport.py` framing: 8-byte big-endian body length, then a body of `!I` header length +
JSON header + (`!Q` length + raw blob)*, `MAX_FRAME` cap, a malformed frame is a ConnectionError.
A shard-frame/2 body IS a transport body. For the same message, `encode()` produces exactly the
bytes `transport._pack` produces (tests pin this), and `transport.recv_msg` can read one.
ARCHITECTURE-v2 §5 is the spec. The field tables live in `_FRAME`, `_REPLY` and `_CONTROL` below.

DEVICE-AGNOSTIC PAYLOADS. A tensor is `Tensor(dtype, shape, data)`: a dtype tag (bf16, f16, fp8,
f32, i32, u8), a shape and raw little-endian bytes. Encoding and decoding import neither torch
nor numpy, so a CPU, Apple MLX or AMD peer speaks the protocol with the stdlib alone.
`.numpy()` and `.torch()` are lazy conveniences. numpy has no bf16/fp8, so `.numpy()` returns
the raw storage (uint16 / uint8 bits). Only `send`/`recv` import `shard.transport` (and so
torch), to reuse its socket code. A torch-less peer reads the 8-byte length itself and calls
`decode()`.

CANONICAL BYTES. One message has exactly one encoding, because §7 receipts hash wire bytes
and the entry signs `lease_update`. The rules:
- JSON header: keys in schema order, `json.dumps` default separators (transport's), ASCII only;
- ints only, no floats;
- optional fields at their default are omitted;
- tensor blobs numbered in schema order, each referenced exactly once, none trailing.

`decode()` re-encodes the header it parsed and requires byte equality. Every accepted body is
therefore canonical: `b"".join(encode(decode(b))) == b`, and the fuzz tests assert exactly that.
Anything else raises `FrameError`, a ConnectionError. Edge supervision (`except EDGE_ERRORS`)
treats it as a dead edge, never a process death. `encode()` raises ValueError instead, so a
sender bug fails loudly at the sender and is not mistaken for a flaky link.

LAYOUT OF A DECODE FRAME (MiMo-Flash, H = 4096, bf16, n = 1): 8 (length) + 4 (header length) +
JSON header + 8 per blob + the 8192 B row. Overhead on top of the row:
- 243 B for a bare frame (223 B JSON);
- 354 B mid-path with 7 `ld` bytes and an `eid` (plus its ciphertext);
- 431 B with every number at its maximum width and a hint.

`tests/test_frame.py::test_byte_budget` pins these bounds for each message type.

DECISIONS WHERE §5 IS SILENT OR AMBIGUOUS (each consistent with the §7 receipts):
- Wire row (§7.1) = `x[i] ‖ xs[i] ‖ tap[i]`, the exact received/sent blob bytes (`wire_row`).
  `eid` and the header are excluded, so an OUT leaf computed from the sent message equals the IN
  leaf computed from the received one.
- Signed or opaque payloads (`lease` token, `ck` checkpoint, `kv` blob, `sig`, `stats` record,
  `eid`) ride as raw u8 blobs and are never re-serialised. Their bytes are what was signed. Their
  schemas belong to their owners (P1, M3, the runtime).
- Small fixed secrets and ids (`sid`, `seed` u64, `salt`, `ekey` 32 B, `nonce`) are lowercase hex
  strings. A u64 as a JSON number is unsafe past 2^53 in other languages. JSON numbers are ≤ u32.
- `pos + n ≤ 2^32 − 1`, so the tail's OUT leaf index p+1 (§7.1) fits u32. `-1 ≤ cpos ≤ pos + n`.
  `seq`, `sep` and `rte` never wrap: a session ends first.
- `hint = {dnxt?, lic?}`: `dnxt` = the token id already streamed after this frame (v1 `dnxt`),
  `lic` = v1 `dprev`, the licence to reuse the block returned one position back
  (v4_dspark_draft.wants_block).
- `sub`/`last` are prefill-only. Any other kind must leave them at 0/true (omitted).
- `fen` frames are header-only: no `x`/`xs`/`tap`/`eid`. Fenced replies keep sid/rte/seq/sep
  (§5.2) and carry no `toks`/`acc`/`blk`.
- `cx = ids` implies `ct = none`: the tap lane starts at the head (no layers precede it).
- `eid` ≤ 4·n + 64 bytes (ids + AEAD nonce/tag slack), so it cannot become a side channel.
- `toks` length ∈ {0, 1, n}, matching `lg` = none/last/all. Token ids are ≥ 0 (a negative id is a
  device-side assert on CUDA embedding). The runtime still bounds ids by the vocabulary.
- `acc` is tri-state: absent / true / false. `dt` rows are integer microseconds
  `[span_us, comp_us]`, not float ms, because floats have no canonical JSON form across languages.
- `lease_open.role` adds `solo` (head and tail in one hop): per the device-agnostic principle, a
  device holding the whole model is a 1-hop path. Secrets are role-gated at encode and decode:
  `salt` only for head/tail/solo, `ekey`/`seed` only for tail/solo, and never to a mid hop.
- Position ranges are named `pos_lo`/`pos_hi` everywhere, half-open. §5.3 `log_fetch` says
  `lo_pos/hi_pos`; §7.2 and `mirror` use `pos_lo`. Layer ranges `lo`/`hi` are half-open, like v1
  receipts.
- `mirror.ck` is optional: a checkpoint rides only at grid crossings. `mirror_close` carries
  `sid, primary, lo, hi` (identity only). `ckpt` carries `sid, ck`. `drain` carries nothing.
- `log_rows` holds one codec per message, with the same payload fields as a frame, so the
  rebuilding hop can check its IN leaves with `wire_row`. `seg` = [[start_pos, n, codec, kind]]
  (§7.1 SEG). A range that changes codec is served as several messages.
- `lease_update.sig` signs `signing_bytes(msg)`: the canonical body with `sig` left out.
"""
import json
import os
import re
import struct
import sys
from array import array

SCHEMA = 2
# Same knob and default as shard.transport.MAX_FRAME (tests pin the equality). It is duplicated,
# not imported, so encoding and decoding stay torch-free.
MAX_FRAME = int(os.environ.get("M25_MAX_FRAME") or 256 * 1024 * 1024)
MAX_HEADER = 1 << 20      # JSON header cap: bounds the parse before json.loads sees a byte
MAX_ROWS = 0xFFFF         # n is u16
MAX_HIDDEN = 0xFFFF
MAX_HOPS = 255            # ld / dt entries
MAX_BLK = 64              # tail drafter block length K
EID_SLACK = 64            # eid ≤ 4·n + this (AEAD nonce + tag)
U16, U32, I31 = 0xFFFF, 0xFFFFFFFF, 0x7FFFFFFF


class FrameError(ConnectionError):
    """A received body that is not a canonical, in-bounds shard-frame/2 message (a dead edge)."""


# ── device-agnostic tensors ─────────────────────────────────────────────────────────────────────
# tag -> (transport dtype string, itemsize, numpy storage dtype). The transport strings are just
# identifiers on the wire, kept so a v2 body stays a valid transport body.
_DT = {
    "bf16": ("torch.bfloat16", 2, "<u2"),
    "f16": ("torch.float16", 2, "<f2"),
    "f32": ("torch.float32", 4, "<f4"),
    "fp8": ("torch.float8_e4m3fn", 1, "u1"),
    "i32": ("torch.int32", 4, "<i4"),
    "u8": ("torch.uint8", 1, "u1"),
}
_BY_WIRE = {w: k for k, (w, _, _) in _DT.items()}


class Tensor:
    """dtype tag + shape + raw little-endian bytes. `data` is a flat memoryview (zero-copy over the
    received frame on decode). Every supported device family is little-endian."""
    __slots__ = ("dtype", "shape", "data")

    def __init__(self, dtype, shape, data):
        if dtype not in _DT:
            raise ValueError(f"unknown dtype {dtype!r}")
        shape = tuple(shape)
        numel = 1
        for d in shape:
            if type(d) is not int or d < 0:
                raise ValueError(f"bad tensor dim {d!r}")
            numel *= d
        mv = memoryview(data)
        if mv.ndim != 1 or mv.format != "B":
            mv = mv.cast("B")
        if numel * _DT[dtype][1] != len(mv):
            raise ValueError(f"{dtype}{list(shape)} needs {numel * _DT[dtype][1]} bytes, got {len(mv)}")
        self.dtype, self.shape, self.data = dtype, shape, mv

    def row(self, i) -> memoryview:
        """bytes of row i along dim 0 (a 1-D tensor's rows are its elements)."""
        if not 0 <= i < self.shape[0]:
            raise IndexError(f"row {i} out of range {self.shape[0]}")
        rb = len(self.data) // self.shape[0]
        return self.data[i * rb:(i + 1) * rb]

    def numpy(self):
        import numpy as np
        return np.frombuffer(self.data, dtype=np.dtype(_DT[self.dtype][2])).reshape(self.shape)

    def torch(self):
        import torch
        dt = getattr(torch, _DT[self.dtype][0].split(".", 1)[1])
        if not len(self.data):
            return torch.empty(self.shape, dtype=dt)
        return torch.frombuffer(bytearray(self.data), dtype=torch.uint8).view(dt).reshape(self.shape)

    @classmethod
    def from_torch(cls, t):
        import torch
        t = t.detach().cpu().contiguous()
        tag = _BY_WIRE.get(str(t.dtype))
        if tag is None:
            raise ValueError(f"no shard-frame/2 dtype for {t.dtype}")
        return cls(tag, tuple(t.shape), memoryview(t.reshape(-1).view(torch.uint8).numpy()))

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            return NotImplemented
        return self.dtype == other.dtype and self.shape == other.shape and self.data == other.data

    __hash__ = None

    def __repr__(self):
        return f"Tensor({self.dtype}, {list(self.shape)}, {len(self.data)} B)"


# ── field checks: each returns the normalised value or raises ValueError ────────────────────────
_REQ = object()           # "no default": the field is required


def _int(lo, hi):
    def chk(v):
        if type(v) is not int or not lo <= v <= hi:       # `type is`: JSON true must never pass as 1
            raise ValueError(f"want int in [{lo}, {hi}], got {v!r}")
        return v
    return chk


def _bool(v):
    if type(v) is not bool:
        raise ValueError(f"want bool, got {v!r}")
    return v


def _enum(*vals):
    def chk(v):
        if type(v) is not str or v not in vals:
            raise ValueError(f"want one of {vals}, got {v!r}")
        return v
    return chk


def _re(pattern, what):
    rx = re.compile(pattern)

    def chk(v):
        if type(v) is not str or not rx.fullmatch(v):
            raise ValueError(f"want {what}, got {v!r}"[:120])
        return v
    return chk


def _const(c):
    def chk(v):
        if type(v) is not type(c) or v != c:
            raise ValueError(f"want {c!r}, got {v!r}")
        return v
    return chk


_U8, U16_, U32_, _TOK = _int(0, 255), _int(0, U16), _int(0, U32), _int(0, I31)
_CPOS = _int(-1, U32)                                   # -1 = none


def _u8list(v):
    if type(v) not in (list, tuple) or len(v) > MAX_HOPS:
        raise ValueError(f"want ≤ {MAX_HOPS} u8, got {str(v)[:60]}")
    return [_U8(b) for b in v]


def _dt(v):
    if type(v) not in (list, tuple) or len(v) > MAX_HOPS:
        raise ValueError(f"want ≤ {MAX_HOPS} [span_us, comp_us] rows")
    out = []
    for r in v:
        if type(r) not in (list, tuple) or len(r) != 2:
            raise ValueError(f"dt row {str(r)[:40]} is not [span_us, comp_us]")
        out.append([U32_(r[0]), U32_(r[1])])
    return out


def _hint(v):
    if type(v) is not dict or not set(v) <= {"dnxt", "lic"}:
        raise ValueError(f"hint must be {{dnxt?, lic?}}, got {str(v)[:60]}")
    dnxt, lic = v.get("dnxt"), v.get("lic", False)
    if dnxt is not None:
        _TOK(dnxt)
    _bool(lic)
    return None if dnxt is None and not lic else {"dnxt": dnxt, "lic": lic}


_CODEC = _enum("ids", "bf16", "fp8")
_SEGKIND = _enum("decode", "verify", "prefill", "rebuild")


def _seg(v):
    if type(v) not in (list, tuple) or not 1 <= len(v) <= MAX_ROWS:
        raise ValueError("seg must hold 1..65535 [start_pos, n, codec, kind] records")
    out = []
    for r in v:
        if type(r) not in (list, tuple) or len(r) != 4:
            raise ValueError(f"seg record {str(r)[:60]} is not [start_pos, n, codec, kind]")
        s, n = U32_(r[0]), U32_(r[1])
        if n < 1 or s + n > U32:
            raise ValueError(f"seg record [{s}, {n}] runs past u32")
        out.append([s, n, _CODEC(r[2]), _SEGKIND(r[3])])
    return out


def _tensor(*dtypes):
    def chk(v):
        if not isinstance(v, Tensor) or v.dtype not in dtypes or not 1 <= len(v.shape) <= 2:
            raise ValueError(f"want a 1-2 D Tensor of {dtypes}, got {v!r}")
        return v
    return chk


def _bytes(lo, hi):
    def chk(v):
        if not isinstance(v, (bytes, bytearray, memoryview)):
            raise ValueError(f"want bytes, got {type(v).__name__}")
        mv = memoryview(v)
        if mv.ndim != 1 or mv.format != "B":
            mv = mv.cast("B")
        if not lo <= len(mv) <= hi:
            raise ValueError(f"want {lo}..{hi} bytes, got {len(mv)}")
        return mv
    return chk


_HEX64 = _re(r"[0-9a-f]{16}", "16 lowercase hex chars (u64)")
_HEX32B = _re(r"[0-9a-f]{64}", "64 lowercase hex chars (32 bytes)")
_NONCE = _re(r"(?:[0-9a-f]{2}){8,64}", "8..64 bytes as lowercase hex")
_PEER = _re(r"[A-Za-z0-9]{1,128}", "a libp2p PeerId (base58/base32, ≤ 128 chars)")
_NAME = _re(r"[a-z0-9][a-z0-9_.:-]{0,63}", "a drafter id [a-z0-9_.:-], ≤ 64 chars")


# ── schemas: (key, check, default | _REQ, blob kind: None | "t" tensor | "b" bytes) ─────────────
# Order is the canonical header order. Blobs come after scalars, and are numbered in this order.
def _head3(op):
    return [("v", _const(SCHEMA), _REQ, None), ("op", _const(op), _REQ, None)]


_FRAME = _head3("f") + [
    ("sid", _HEX64, _REQ, None),
    ("rte", U32_, _REQ, None),          # route epoch: keys the lease
    ("seq", U32_, _REQ, None),          # per-session injection seq, echoed by the reply
    ("sep", U32_, _REQ, None),          # speculation epoch: drives the fence
    ("pos", U32_, _REQ, None),          # absolute start position = the rewind command
    ("n", U16_, _REQ, None),
    ("kind", _enum("P", "T", "S", "R"), _REQ, None),
    ("cpos", _CPOS, _REQ, None),        # final-row frontier F = c-1; rows < F are final
    ("sub", U16_, 0, None),
    ("last", _bool, True, None),
    ("fen", _bool, False, None),
    ("lg", _enum("none", "last", "all"), _REQ, None),
    ("hint", _hint, None, None),
    ("cx", _CODEC, _REQ, None),
    ("ct", _enum("none", "bf16"), "none", None),
    ("ld", _u8list, (), None),
    ("x", _tensor("i32", "bf16", "fp8"), None, "t"),
    ("xs", _tensor("f32"), None, "t"),
    ("tap", _tensor("bf16"), None, "t"),
    ("eid", _bytes(1, MAX_FRAME), None, "b"),
]

_REPLY = _head3("r") + [
    ("sid", _HEX64, _REQ, None),
    ("rte", U32_, _REQ, None),
    ("seq", U32_, _REQ, None),
    ("sep", U32_, _REQ, None),
    ("pos", U32_, _REQ, None),
    ("n", U16_, _REQ, None),
    ("acc", _bool, None, None),
    ("fen", _bool, False, None),
    ("ld", _u8list, (), None),
    ("dt", _dt, (), None),
    ("toks", _tensor("i32"), None, "t"),
    ("blk", _tensor("i32"), None, "t"),
]

_SID = ("sid", _HEX64, _REQ, None)
_RTE = ("rte", U32_, _REQ, None)
_LO, _HI = ("lo", U16_, _REQ, None), ("hi", U16_, _REQ, None)
_PLO, _PHI = ("pos_lo", U32_, _REQ, None), ("pos_hi", U32_, _REQ, None)
_PAYLOAD = [("x", _tensor("i32", "bf16", "fp8"), None, "t"), ("xs", _tensor("f32"), None, "t"),
            ("tap", _tensor("bf16"), None, "t")]

_CONTROL = {
    "lease_open": [_SID, _RTE, _LO, _HI,
                   ("role", _enum("head", "mid", "tail", "solo"), _REQ, None),
                   ("pred", _PEER, _REQ, None), ("next", _PEER, None, None),
                   ("ret", _PEER, _REQ, None), ("backup", _PEER, None, None),
                   ("max_pos", U32_, _REQ, None), ("W", U16_, _REQ, None),
                   ("cx", _CODEC, _REQ, None), ("ct", _enum("none", "bf16"), "none", None),
                   ("drafter", _NAME, None, None), ("seed", _HEX64, None, None),
                   ("salt", _HEX32B, None, None), ("ekey", _HEX32B, None, None),
                   ("nonce", _NONCE, _REQ, None),
                   ("lease", _bytes(1, 4096), _REQ, "b")],
    "lease_update": [_SID, _RTE, ("pred", _PEER, None, None), ("next", _PEER, None, None),
                     ("ret", _PEER, None, None), ("backup", _PEER, None, None),
                     ("truncate_to", _CPOS, _REQ, None),
                     ("sig", _bytes(1, 512), _REQ, "b")],
    "ckpt": [_SID, ("ck", _bytes(1, 65536), _REQ, "b")],
    "close": [_SID],
    "drain": [],
    "mirror": [_SID, _RTE, _PLO, _PHI, ("kv", _bytes(0, MAX_FRAME), _REQ, "b"),
               ("ck", _bytes(1, 65536), None, "b")],
    "mirror_open": [_SID, ("primary", _PEER, _REQ, None), _LO, _HI, ("max_pos", U32_, _REQ, None)],
    "mirror_close": [_SID, ("primary", _PEER, _REQ, None), _LO, _HI],
    "takeover": [_SID, _RTE, ("F", _CPOS, _REQ, None), ("g_entry", U32_, _REQ, None),
                 ("pred", _PEER, _REQ, None), ("next", _PEER, None, None),
                 ("ret", _PEER, _REQ, None), ("lease", _bytes(1, 4096), _REQ, "b")],
    "log_fetch": [_SID, _PLO, _PHI],
    "log_rows": [_SID, _PLO, _PHI, ("cx", _CODEC, _REQ, None),
                 ("ct", _enum("none", "bf16"), "none", None), ("seg", _seg, _REQ, None)] + _PAYLOAD,
    "stats": [("rec", _bytes(1, 65536), _REQ, "b")],
}
_CONTROL = {cmd: _head3("c") + [("cmd", _const(cmd), _REQ, None)] + f for cmd, f in _CONTROL.items()}
MAX_BLOBS = max(sum(1 for f in s if f[3]) for s in [_FRAME, _REPLY, *_CONTROL.values()])


def _schema(msg):
    op = msg.get("op")
    if op == "f":
        return _FRAME
    if op == "r":
        return _REPLY
    if op == "c":
        s = _CONTROL.get(msg.get("cmd")) if type(msg.get("cmd")) is str else None
        if s is None:
            raise ValueError(f"unknown control cmd {str(msg.get('cmd'))[:40]!r}")
        return s
    raise ValueError(f"unknown op {str(op)[:40]!r}")


# ── cross-field rules ───────────────────────────────────────────────────────────────────────────
def _want(t, dtype, shape, what):
    if t is None or t.dtype != dtype or t.shape != shape:
        raise ValueError(f"{what}: want {dtype}{list(shape)}, got {t!r}")


def _ids_nonneg(t, what):
    a = array("i")                                        # 4-byte C int on every CPython target
    a.frombytes(t.data)
    if sys.byteorder != "little":                         # pragma: no cover (wire is little-endian)
        a.byteswap()
    if len(a) and min(a) < 0:
        raise ValueError(f"{what}: negative token id")


def _payload(m, n):
    """x / xs / tap for n rows under codecs cx / ct (frames and log_rows)."""
    x, cx, ct = m["x"], m["cx"], m["ct"]
    if x is None:
        raise ValueError("x is required (only a fenced frame is header-only)")
    if cx == "ids":
        _want(x, "i32", (n,), "x")
        _ids_nonneg(x, "x")
        H = None
    else:
        H = x.shape[1] if len(x.shape) == 2 else 0
        if not 1 <= H <= MAX_HIDDEN:
            raise ValueError(f"x: hidden size {H} outside 1..{MAX_HIDDEN}")
        _want(x, cx, (n, H), "x")
    if cx == "fp8":
        _want(m["xs"], "f32", (n,), "xs (per-row fp8 scales)")
    elif m["xs"] is not None:
        raise ValueError("xs is only legal with cx=fp8")
    if ct == "bf16":
        if H is None:
            raise ValueError("ct=bf16 needs a hidden-state body (cx != ids)")
        _want(m["tap"], "bf16", (n, H), "tap")
    elif m["tap"] is not None:
        raise ValueError("tap is only legal with ct=bf16")


def _span(pos, n):
    if n < 1:
        raise ValueError("n must be ≥ 1")
    if pos + n > U32:
        raise ValueError(f"pos+n = {pos + n} past u32 (the tail's OUT leaf index p+1 must fit)")


def _check_frame(m):
    n, pos = m["n"], m["pos"]
    _span(pos, n)
    if m["cpos"] > pos + n:
        raise ValueError(f"cpos {m['cpos']} beyond this frame's rows (pos+n = {pos + n})")
    if m["kind"] != "P" and (m["sub"] != 0 or not m["last"]):
        raise ValueError("sub/last are prefill-only")
    if m["cx"] == "ids" and m["ct"] != "none":
        raise ValueError("cx=ids implies ct=none: the tap lane starts at the head")
    if m["fen"]:
        if any(m[k] is not None for k in ("x", "xs", "tap", "eid")):
            raise ValueError("a fenced frame is header-only")
        return
    _payload(m, n)
    if m["eid"] is not None and len(m["eid"]) > 4 * n + EID_SLACK:
        raise ValueError(f"eid {len(m['eid'])} B > 4·n + {EID_SLACK}")


def _check_reply(m):
    n = m["n"]
    _span(m["pos"], n)
    if m["fen"]:
        if any(m[k] is not None for k in ("toks", "blk", "acc")):
            raise ValueError("a fenced reply carries no toks/blk/acc")
    else:
        t = m["toks"]
        if t is None or len(t.shape) != 1 or t.shape[0] not in (0, 1, n):
            raise ValueError(f"toks: want i32[0|1|n={n}], got {t!r}")
        _ids_nonneg(t, "toks")
    b = m["blk"]
    if b is not None:
        if len(b.shape) != 1 or not 1 <= b.shape[0] <= MAX_BLK:
            raise ValueError(f"blk: want i32[1..{MAX_BLK}], got {b!r}")
        _ids_nonneg(b, "blk")


def _check_control(m):
    cmd = m["cmd"]
    if "lo" in m and not m["lo"] < m["hi"]:
        raise ValueError(f"layer range [{m['lo']}, {m['hi']}) is empty")
    if cmd == "lease_open":
        role = m["role"]
        tailish = role in ("tail", "solo")
        for key, allowed in (("next", not tailish), ("seed", tailish), ("ekey", tailish),
                             ("salt", role != "mid")):
            if (m[key] is not None) != allowed:
                raise ValueError(f"lease_open role={role}: {key} must be "
                                 f"{'present' if allowed else 'absent'}")
        if m["drafter"] is not None and role == "mid":
            raise ValueError("a mid hop has no drafter")
    elif cmd == "mirror" and m["pos_lo"] > m["pos_hi"]:
        raise ValueError("mirror: pos_lo > pos_hi")
    elif cmd in ("log_fetch", "log_rows"):
        n = m["pos_hi"] - m["pos_lo"]
        if not 1 <= n <= MAX_ROWS:
            raise ValueError(f"{cmd}: range of {n} rows outside 1..{MAX_ROWS}")
        if cmd == "log_rows":
            _payload(m, n)
            if len(m["seg"]) > n:
                raise ValueError("log_rows: more seg records than rows")


def _normalize(msg, skip=()):
    """Validate a message dict (from a caller or a parsed header) and return a new dict with every
    schema field present (defaults filled). Unknown keys are refused, never dropped."""
    if type(msg) is not dict:
        raise ValueError(f"a message is a dict, got {type(msg).__name__}")
    schema = _schema(msg)
    unknown = set(msg) - {f[0] for f in schema}
    if unknown:
        raise ValueError(f"unknown field(s) {sorted(map(str, unknown))[:4]}")
    out = {}
    for key, chk, dflt, _ in schema:
        if key in skip:
            continue
        v = msg.get(key)
        if v is None:
            if dflt is _REQ:
                raise ValueError(f"missing required field {key!r}")
            out[key] = list(dflt) if type(dflt) is tuple else dflt
            continue
        try:
            out[key] = chk(v)
        except ValueError as e:
            raise ValueError(f"{key}: {e}") from None
    {"f": _check_frame, "r": _check_reply, "c": _check_control}[out["op"]](out)
    return out


def _head(m, schema):
    """canonical JSON header + blob list for a normalised message"""
    obj, blobs = {}, []
    for key, _, dflt, kind in schema:
        if key not in m:
            continue
        v = m[key]
        if dflt is not _REQ and v == (list(dflt) if type(dflt) is tuple else dflt):
            continue
        if kind:
            t = v if kind == "t" else Tensor("u8", (len(v),), v)
            obj[key] = {"__t__": len(blobs), "dtype": _DT[t.dtype][0], "shape": list(t.shape)}
            blobs.append(t.data)
        elif key == "hint":                               # dnxt may be token 0: test `is not None`
            obj[key] = {k: v[k] for k in ("dnxt", "lic") if v[k] is not None and v[k] is not False}
        else:
            obj[key] = v
    return json.dumps(obj).encode("ascii"), blobs


# ── public codec ────────────────────────────────────────────────────────────────────────────────
def frame(**kw) -> dict:
    """a validated frame dict (op f). Callers mutate a copy and re-validate via encode()."""
    return _normalize({"v": SCHEMA, "op": "f", **kw})


def reply(**kw) -> dict:
    return _normalize({"v": SCHEMA, "op": "r", **kw})


def control(cmd, **kw) -> dict:
    return _normalize({"v": SCHEMA, "op": "c", "cmd": cmd, **kw})


def encode(msg) -> list:
    """message -> body segments (bytes-likes; tensor blobs are zero-copy views). Their concatenation
    is the canonical transport body. Raises ValueError for an invalid message."""
    m = _normalize(msg)
    head, blobs = _head(m, _schema(m))
    if len(head) > MAX_HEADER:
        raise ValueError(f"header {len(head)} B > MAX_HEADER")
    parts = [struct.pack("!I", len(head)) + head]
    for b in blobs:
        parts += [struct.pack("!Q", len(b)), b]
    size = sum(len(p) for p in parts)
    if size > MAX_FRAME:
        raise ValueError(f"message body {size} B > MAX_FRAME ({MAX_FRAME})")
    return parts


def signing_bytes(msg) -> bytes:
    """What `lease_update.sig` signs: the canonical body of the message with `sig` left out."""
    m = _normalize(msg, skip=("sig",))
    head, blobs = _head(m, _schema(m))
    return b"".join([struct.pack("!I", len(head)), head] +
                    [p for b in blobs for p in (struct.pack("!Q", len(b)), b)])


def _decode(body):
    mv = memoryview(body)
    if mv.ndim != 1 or mv.format != "B":
        mv = mv.cast("B")
    if len(mv) > MAX_FRAME:
        raise FrameError(f"body {len(mv)} B > MAX_FRAME")
    if len(mv) < 4:
        raise FrameError("body shorter than its header length")
    (hlen,) = struct.unpack_from("!I", mv, 0)
    if hlen > MAX_HEADER or 4 + hlen > len(mv):
        raise FrameError(f"header length {hlen} out of bounds")
    hb = bytes(mv[4:4 + hlen])
    obj = json.loads(hb.decode("ascii"))       # floats/NaN parse, then fail the `type is int` checks
    if type(obj) is not dict:
        raise FrameError("header is not a JSON object")
    blobs, off = [], 4 + hlen
    while off < len(mv):
        if len(blobs) == MAX_BLOBS or off + 8 > len(mv):
            raise FrameError("trailing bytes / too many blobs")
        (blen,) = struct.unpack_from("!Q", mv, off)
        off += 8
        if blen > len(mv) - off:
            raise FrameError(f"blob length {blen} past the end of the body")
        blobs.append(mv[off:off + blen])
        off += blen
    schema = _schema(obj)
    nxt = 0
    for key, _, _, kind in schema:
        if not kind or obj.get(key) is None:
            continue
        d = obj[key]
        if (type(d) is not dict or d.keys() != {"__t__", "dtype", "shape"}
                or type(d["__t__"]) is not int or d["__t__"] != nxt or nxt >= len(blobs)
                or type(d["dtype"]) is not str or d["dtype"] not in _BY_WIRE
                or type(d["shape"]) is not list or not 1 <= len(d["shape"]) <= 2):
            raise FrameError(f"{key}: bad tensor descriptor {str(d)[:80]}")
        t = Tensor(_BY_WIRE[d["dtype"]], d["shape"], blobs[nxt])
        if kind == "b":
            if t.dtype != "u8" or len(t.shape) != 1:
                raise FrameError(f"{key}: bytes field must be u8[len]")
            t = t.data
        obj[key] = t
        nxt += 1
    if nxt != len(blobs):
        raise FrameError(f"{len(blobs) - nxt} unreferenced blob(s)")
    m = _normalize(obj)
    if _head(m, schema)[0] != hb:
        raise FrameError("non-canonical header encoding")
    return m


def decode(body) -> dict:
    """canonical transport body -> validated message dict (all schema fields present). Tensors are
    `Tensor`s viewing `body` (zero-copy); bytes fields are memoryviews. Anything that is not a
    canonical, in-bounds shard-frame/2 message raises FrameError, and nothing else escapes."""
    try:
        return _decode(body)
    except FrameError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError,
            struct.error) as e:
        raise FrameError(f"malformed shard-frame/2 ({type(e).__name__}: {str(e)[:120]})") from None


def wire_row(msg, i) -> bytes:
    """§7.1 wire row i (position pos+i, or pos_lo+i for log_rows): x[i] ‖ xs[i] ‖ tap[i], i.e. the
    exact blob bytes on the wire. `eid` and header fields are not part of it."""
    if msg.get("x") is None:
        raise ValueError("no payload rows (fenced or not a row-carrying message)")
    return b"".join(t.row(i) for t in (msg["x"], msg.get("xs"), msg.get("tap")) if t is not None)


# ── socket helpers: shard/transport's framing and I/O, unchanged ────────────────────────────────
def _transport():
    try:
        import transport                                  # flat layout (engine files side by side)
    except ImportError:
        from shard import transport
    return transport


def send(sock, msg) -> int:
    """encode + send one message; returns bytes on the wire (incl. the 8-byte length)."""
    parts = encode(msg)
    n = sum(len(p) for p in parts)
    _transport()._sendall_vectored(sock, [struct.pack("!Q", n)] + parts)
    return 8 + n


def recv(sock) -> dict:
    """receive + decode one message. A closed link or a bad message raises ConnectionError
    (FrameError for the latter). An oversized length is refused before the body is read."""
    read = _transport()._read_exact
    (n,) = struct.unpack("!Q", read(sock, 8))
    if n > MAX_FRAME:
        raise FrameError(f"frame length {n} exceeds MAX_FRAME ({MAX_FRAME})")
    return decode(read(sock, n))
