"""shard-frame/2 codec (shard/frame.py): round-trip, canonical bytes, bounds, fuzz, byte budget.

The wire is hostile input. Every receive path must end in a validated message or a FrameError (a
ConnectionError, so edge supervision treats it as a dead edge), never another exception, a hang
or an unbounded allocation. Receipts hash wire bytes, so the codec must also be canonical: every
accepted body is exactly the encoding of the message it decodes to.

Run: python3 -m pytest tests/test_frame.py -q
"""
import json
import os
import random
import socket
import struct
import subprocess
import sys
import threading

import pytest

from shard import frame as F

SID = "0123456789abcdef"
PEER = "12D3KooWPeerA"
U32 = F.U32
H = 64                     # small hidden size for the generic tests; the budget tests use 4096


def rb(k, seed=0):
    return random.Random(seed * 7919 + k).randbytes(k)


def hid(n, h=H, dtype="bf16", seed=1):
    return F.Tensor(dtype, (n, h), rb(n * h * {"bf16": 2, "fp8": 1}[dtype], seed))


def ids(vals):
    return F.Tensor("i32", (len(vals),), struct.pack(f"<{len(vals)}i", *vals))


def body(m):
    return b"".join(F.encode(m))


def split(b):
    (hlen,) = struct.unpack_from("!I", b, 0)
    return json.loads(b[4:4 + hlen]), b[4 + hlen:]


def raw(h):
    return struct.pack("!I", len(h)) + h


def join(obj, rest, dumps=json.dumps):
    h = dumps(obj).encode()
    return struct.pack("!I", len(h)) + h + rest


BASE = dict(sid=SID, rte=3, seq=17, sep=5, pos=100, lg="last")


# ── samples: every frame kind x codec, reply shapes, every control cmd ─────────────────────────
def frames():
    out = []
    for kind in "PTSR":
        n = 9
        out.append(F.frame(**BASE, n=n, kind=kind, cpos=99, cx="ids", x=ids(range(n)),
                           eid=rb(4 * n + 28)))
        out.append(F.frame(**BASE, n=n, kind=kind, cpos=99, cx="bf16", x=hid(n), ld=[1, 2]))
        out.append(F.frame(**BASE, n=n, kind=kind, cpos=-1, cx="fp8", x=hid(n, dtype="fp8"),
                           xs=F.Tensor("f32", (n,), rb(4 * n)), ct="bf16", tap=hid(n, seed=2),
                           hint={"lic": True}, eid=rb(40)))
        out.append(F.frame(**BASE, n=n, kind=kind, cpos=109, cx="bf16", ct="bf16", fen=True))
    out.append(F.frame(**BASE, n=4, kind="P", sub=3, last=False, cpos=-1, cx="bf16", x=hid(4)))
    out.append(F.frame(**BASE, n=1, kind="S", cpos=100, cx="bf16", x=hid(1), hint={"lic": True},
                       eid=rb(4 * 2 + F.EID_SLACK)))           # id + encrypted dnxt, max slack
    return out


def replies():
    return [
        F.reply(**{k: BASE[k] for k in ("sid", "rte", "seq", "sep", "pos")}, n=9,
                toks=ids(range(9)), acc=False, blk=ids([5] * 8), ld=[0, 255], dt=[[10, 3], [7, 7]]),
        F.reply(sid=SID, rte=0, seq=0, sep=0, pos=0, n=1, toks=ids([7])),
        F.reply(sid=SID, rte=1, seq=2, sep=3, pos=4, n=5, toks=ids([])),
        F.reply(sid=SID, rte=9, seq=8, sep=7, pos=6, n=5, fen=True, ld=[3]),
    ]


def controls():
    lease = rb(200)
    return [
        F.control("lease_open", sid=SID, rte=1, lo=0, hi=7, role="head", pred=PEER, next="Qm2",
                  ret=PEER, backup="Qm3", max_pos=32768, W=8, cx="bf16", salt="a" * 64,
                  nonce="0f" * 16, lease=lease),
        F.control("lease_open", sid=SID, rte=1, lo=7, hi=14, role="mid", pred=PEER, next="Qm2",
                  ret=PEER, max_pos=32768, W=8, cx="fp8", ct="bf16", nonce="0f" * 16, lease=lease),
        F.control("lease_open", sid=SID, rte=1, lo=42, hi=48, role="tail", pred=PEER, ret=PEER,
                  max_pos=32768, W=8, cx="bf16", drafter="mimo-mtp3", seed="00000000000000ff",
                  salt="a" * 64, ekey="b" * 64, nonce="0f" * 16, lease=lease),
        F.control("lease_open", sid=SID, rte=1, lo=0, hi=48, role="solo", pred=PEER, ret=PEER,
                  max_pos=32768, W=8, cx="bf16", seed="0" * 16, salt="a" * 64, ekey="b" * 64,
                  nonce="0f" * 8, lease=lease),
        F.control("lease_update", sid=SID, rte=2, next="QmNew", truncate_to=127, sig=rb(64)),
        F.control("ckpt", sid=SID, ck=rb(300)),
        F.control("close", sid=SID),
        F.control("drain"),
        F.control("mirror", sid=SID, rte=2, pos_lo=100, pos_hi=128, kv=rb(1000), ck=rb(300)),
        F.control("mirror", sid=SID, rte=2, pos_lo=128, pos_hi=128, kv=b""),
        F.control("mirror_open", sid=SID, primary=PEER, lo=0, hi=7, max_pos=32768),
        F.control("mirror_close", sid=SID, primary=PEER, lo=0, hi=7),
        F.control("takeover", sid=SID, rte=3, lo=42, hi=48, role="tail", pred=PEER, ret=PEER,
                  max_pos=32768, W=8, cx="bf16", drafter="mimo-mtp3", seed="0" * 16,
                  salt="a" * 64, ekey="b" * 64, nonce="0f" * 16, F=255, g_entry=128, lease=lease),
        F.control("log_fetch", sid=SID, pos_lo=0, pos_hi=900),
        F.control("log_rows", sid=SID, pos_lo=10, pos_hi=14, cx="fp8", ct="bf16",
                  seg=[[0, 12, "bf16", "prefill"], [12, 2, "bf16", "decode"]],
                  x=hid(4, dtype="fp8"), xs=F.Tensor("f32", (4,), rb(16)), tap=hid(4, seed=3)),
        F.control("stats", rec=b'{"cell":"eu-1"}'),
    ]


ALL = frames() + replies() + controls()


def _id(m):
    return f"{m['op']}-{m.get('cmd') or m.get('kind') or ('fen' if m.get('fen') else 'r')}"


# ── round trip ──────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("m", ALL, ids=[_id(m) for m in ALL])
def test_roundtrip_every_message(m):
    b = body(m)
    d = F.decode(b)
    assert d == m                                           # every field, tensors byte-for-byte
    assert body(d) == b                                     # and re-encodes to the same bytes


def test_socket_roundtrip_and_transport_reads_it():
    from shard import transport as T
    a, b = socket.socketpair()
    a.settimeout(5)
    b.settimeout(5)
    msgs = frames()[:6] + replies()[:2] + controls()[:3]
    t = threading.Thread(target=lambda: [F.send(a, m) for m in msgs] + [F.send(a, msgs[1])])
    t.start()
    try:
        for m in msgs:
            assert F.recv(b) == m
        obj = T.recv_msg(b)                                 # a v2 body IS a transport body
        assert obj["op"] == "f" and obj["x"].dtype.is_floating_point is True
    finally:
        t.join()
        a.close()
        b.close()


def test_malformed_frame_keeps_the_stream_aligned():
    """A malformed message is not a death signal (§8): recv consumed exactly its length, so the
    next well-formed message on the same stream still decodes."""
    a, b = socket.socketpair()
    b.settimeout(5)
    good = body(frames()[1])
    for junk in (b"\x00\x00\x00\x02{}", b"not a frame at all", good[:-1] + b"X" * 9):
        a.sendall(struct.pack("!Q", len(junk)) + junk + struct.pack("!Q", len(good)) + good)
        with pytest.raises(F.FrameError):
            F.recv(b)
        assert F.recv(b) == frames()[1]
    a.close()
    b.close()


# ── canonical bytes ─────────────────────────────────────────────────────────────────────────────
def test_encoding_is_deterministic_and_input_shape_independent():
    m = frames()[2]
    alt = dict(reversed(list(m.items())))                  # other key order
    alt["ld"] = tuple(m["ld"])
    alt["eid"] = bytearray(m["eid"])
    alt["hint"] = {"lic": True}
    assert body(m) == body(m) == body(alt)
    raw = {k: v for k, v in m.items() if k not in ("sub", "last", "fen", "ld")}   # defaults omitted
    assert body(raw) == body(m)


def test_canonical_equals_transport_pack():
    """The canonical encoding is literally shard/transport's framing of the same object."""
    torch = pytest.importorskip("torch")
    from shard import transport as T
    for m in frames() + replies() + controls():
        hdr, _ = split(body(m))
        obj = {}
        for k, v in hdr.items():
            if isinstance(v, dict) and "__t__" in v:
                val = m[k]
                if isinstance(val, F.Tensor):
                    obj[k] = val.torch()
                else:                                       # bytes field -> u8 tensor
                    obj[k] = F.Tensor("u8", (len(val),), val).torch()
            else:
                obj[k] = v
        assert T._pack(obj) == body(m), _id(m)


def _mutations(b):
    """header rewrites that keep the meaning but change the bytes: all must be refused."""
    hdr, rest = split(b)
    compact = lambda o: json.dumps(o, separators=(",", ":"))          # noqa: E731
    yield "compact separators", join(hdr, rest, compact)
    yield "sorted keys", join(hdr, rest, lambda o: json.dumps(o, sort_keys=True))
    yield "trailing space", join(hdr, rest, lambda o: json.dumps(o) + " ")
    yield "indent", join(hdr, rest, lambda o: json.dumps(o, indent=1))
    for k, v in (("fen", False), ("sub", 0), ("last", True), ("ld", []), ("ct", "none"),
                 ("hint", None), ("hint", {}), ("eid", None), ("tap", None)):
        yield f"explicit default {k}", join({**hdr, k: v}, rest)
    yield "escaped key", join(hdr, rest, lambda o: json.dumps(o).replace('"sid"', '"\\u0073id"'))
    yield "bool for int", join({**hdr, "rte": True}, rest)
    yield "float for int", join({**hdr, "rte": float(hdr["rte"])}, rest)
    yield "negative zero", join(hdr, rest, lambda o: json.dumps(o).replace('"seq": 17', '"seq": -0'))
    yield "exponent", join(hdr, rest, lambda o: json.dumps(o).replace('"seq": 17', '"seq": 1.7e1'))
    yield "duplicate key", join(hdr, rest, lambda o: json.dumps(o).replace('"v": 2', '"v": 2, "v": 2'))
    yield "epoch alias", join({**hdr, "epoch": 5}, rest)
    yield "ep alias", join({**hdr, "ep": 5}, rest)
    yield "uppercase sid", join({**hdr, "sid": SID.upper()}, rest)
    yield "utf-16 header", raw(json.dumps(hdr).encode("utf-16")) + rest
    yield "trailing blob", b + struct.pack("!Q", 0)
    yield "NaN", join(hdr, rest, lambda o: json.dumps(o).replace('"seq": 17', '"seq": NaN'))


def test_non_canonical_encodings_are_refused():
    m = frames()[2]                                         # fp8 + xs + tap + hint + eid
    b = body(m)
    assert join(*split(b)) == b                             # the helper itself is faithful
    for what, bad in _mutations(b):
        with pytest.raises(F.FrameError):
            F.decode(bad)
            pytest.fail(what)


def test_blob_order_is_canonical():
    """Swapping two blobs AND their descriptor indices yields a consistent but non-canonical body."""
    m = frames()[2]
    hdr, _ = split(body(m))
    blobs = [m["x"].data, m["xs"].data, m["tap"].data, m["eid"]]
    hdr["x"]["__t__"], hdr["xs"]["__t__"] = 1, 0
    order = [blobs[1], blobs[0], blobs[2], blobs[3]]
    rest = b"".join(struct.pack("!Q", len(p)) + bytes(p) for p in order)
    with pytest.raises(F.FrameError):
        F.decode(join(hdr, rest))


# ── field bounds (both the encode and the decode path) ──────────────────────────────────────────
def _bad_both(m, **change):
    """`change` must be refused by encode (ValueError) and, spliced into a canonical-looking header,
    by decode (FrameError)."""
    bad = {**m, **change}
    with pytest.raises(ValueError):
        F.encode(bad)
    hdr, rest = split(body(m))
    for k, v in change.items():
        if not isinstance(v, (F.Tensor, bytes, bytearray, memoryview)):
            hdr[k] = v
    with pytest.raises(F.FrameError):
        F.decode(join(hdr, rest))


FRAME_BAD = [
    ("rte", [-1, U32 + 1, True, "3", 3.0, None]),
    ("seq", [-1, U32 + 1, False]),
    ("sep", [-1, U32 + 1]),
    ("pos", [-1, U32 + 1, U32]),                             # pos+n must still fit u32
    ("n", [0, 65536]),
    ("kind", ["X", "p", 1]),
    ("cpos", [-2, 110, U32 + 1]),                            # > pos+n = 109
    ("lg", ["some", None]),
    ("cx", ["fp16", "ids"]),                                 # ids body mismatch
    ("ct", ["fp8"]),
    ("sid", ["0123", SID.upper(), "g" * 16, 12345, SID + "0"]),
    ("v", [1, 3, "2"]),
    ("ld", [[256], [-1], [0] * 256, "ab", [True]]),
    ("sub", [1, -1]),                                        # sub/last are prefill-only (kind T)
    ("last", [False, 0]),
    ("fen", [1]),
    ("hint", [{"dnxt": 5}, {"dnxt": 0, "lic": True}, {"lic": 1}, {"dprev": True}, [1]]),
]


@pytest.mark.parametrize("field,values", FRAME_BAD, ids=[f for f, _ in FRAME_BAD])
def test_frame_field_bounds(field, values):
    m = F.frame(**BASE, n=9, kind="T", cpos=99, cx="bf16", x=hid(9))
    for v in values:
        _bad_both(m, **{field: v})


def test_frame_bounds_accept_extremes():
    x = hid(1)
    F.decode(body(F.frame(sid="f" * 16, rte=U32, seq=U32, sep=U32, pos=U32 - 1, n=1, kind="S",
                          cpos=U32, lg="all", cx="bf16", x=x, ld=[255] * 255,
                          hint={"lic": True})))
    F.decode(body(F.frame(sid="0" * 16, rte=0, seq=0, sep=0, pos=0, n=1, kind="P", sub=65535,
                          last=False, cpos=-1, lg="none", cx="bf16", x=x)))


def test_payload_rules():
    n = 3
    base = dict(**BASE, n=n, kind="T", cpos=99)
    bad = [
        dict(cx="bf16"),                                                   # no x, not fenced
        dict(cx="bf16", x=hid(n + 1)),                                     # rows != n
        dict(cx="bf16", x=hid(n, dtype="fp8")),                            # dtype != cx
        dict(cx="bf16", x=F.Tensor("bf16", (n, 0), b"")),                  # H = 0
        dict(cx="bf16", x=F.Tensor("bf16", (n,), rb(2 * n))),              # 1-D hidden
        dict(cx="fp8", x=hid(n, dtype="fp8")),                             # fp8 without xs
        dict(cx="fp8", x=hid(n, dtype="fp8"), xs=F.Tensor("f32", (n + 1,), rb(4 * n + 4))),
        dict(cx="bf16", x=hid(n), xs=F.Tensor("f32", (n,), rb(4 * n))),    # xs without fp8
        dict(cx="bf16", x=hid(n), ct="bf16"),                              # ct without tap
        dict(cx="bf16", x=hid(n), tap=hid(n)),                             # tap without ct
        dict(cx="bf16", x=hid(n), ct="bf16", tap=hid(n, h=H * 2)),         # tap H != x H
        dict(cx="ids", x=ids([1, 2, 3]), ct="bf16", tap=hid(n)),           # tap at the entry
        dict(cx="ids", x=ids([1, -2, 3])),                                 # negative token id
        dict(cx="ids", x=hid(n)),
        dict(cx="bf16", x=hid(n), eid=rb(4 * (n + 1) + F.EID_SLACK + 1)),  # eid too long
        dict(cx="bf16", x=hid(n), eid=b""),
        dict(cx="bf16", x=hid(n), fen=True),                               # fenced with payload
        dict(cx="bf16", fen=True, eid=rb(8)),
        dict(cx="ids", ct="bf16", fen=True),                               # tap lane at the entry
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            F.encode({"v": 2, "op": "f", **base, **kw})
    F.encode({"v": 2, "op": "f", **base, "cx": "bf16", "x": hid(n),
              "eid": rb(4 * (n + 1) + F.EID_SLACK)})


def test_prefill_eid_rides_sub0_and_covers_the_whole_chunk():
    """the head splits a prefill chunk into sub-chunks but holds no ekey, so it cannot split eid:
    the whole chunk's eid rides sub=0, and later pieces carry none"""
    p = dict(**BASE, n=4, kind="P", cpos=-1, cx="bf16", x=hid(4))
    whole = rb(4 * (F.MAX_ROWS + 1) + F.EID_SLACK)
    F.decode(body(F.frame(**p, sub=0, last=False, eid=whole)))
    for kw in (dict(sub=1, eid=rb(16)), dict(sub=0, eid=whole + b"x")):
        with pytest.raises(ValueError):
            F.frame(**p, **kw)
    entry = dict(**BASE, n=4, kind="P", cpos=-1, cx="ids", x=ids([1, 2, 3, 4]))   # entry -> head
    F.frame(**entry, eid=rb(4 * 5 + F.EID_SLACK))       # its n IS the whole chunk: tight bound
    with pytest.raises(ValueError):
        F.frame(**entry, eid=rb(4 * 5 + F.EID_SLACK + 1))


def test_reply_bounds():
    r = dict(v=2, op="r", sid=SID, rte=1, seq=2, sep=3, pos=4, n=5)
    for kw in (dict(), dict(toks=ids([1, 2])), dict(toks=ids([1, 2, 3, 4, 5, 6])),
               dict(toks=ids([-1])), dict(toks=F.Tensor("i32", (1, 1), rb(4))),
               dict(toks=ids([1]), blk=ids([])), dict(toks=ids([1]), blk=ids([1] * 65)),
               dict(toks=ids([1]), blk=ids([-5])), dict(fen=True, toks=ids([1])),
               dict(fen=True, acc=True), dict(toks=ids([1]), acc=1),
               dict(toks=ids([1]), dt=[[1.5, 2]]), dict(toks=ids([1]), dt=[[1, 2, 3]]),
               dict(toks=ids([1]), dt=[[1, 2]] * 256), dict(toks=ids([1]), dt=[[-1, 0]])):
        with pytest.raises(ValueError):
            F.encode({**r, **kw})
    for k in ("sid", "rte", "seq", "sep"):                  # a fenced reply keeps its header (§5.2)
        with pytest.raises(ValueError):
            F.encode({k2: v for k2, v in {**r, "fen": True}.items() if k2 != k})


def test_control_bounds():
    lo = controls()[0]
    bad = [
        dict(role="mid"),                            # mid with salt
        dict(role="tail"),                           # tail with next, no seed/ekey
        dict(next=None),                             # head without next
        dict(ekey="b" * 64),                         # ekey to a head
        dict(seed="0" * 16),
        dict(lo=7, hi=7), dict(lo=8, hi=7), dict(hi=65536),
        dict(nonce="0f" * 7), dict(nonce="0f" * 65), dict(nonce="0f0"), dict(nonce="0F" * 16),
        dict(salt="a" * 63), dict(pred="bad peer"), dict(pred="x" * 129), dict(pred=""),
        dict(lease=b""), dict(lease=rb(4097)), dict(W=-1), dict(max_pos=U32 + 1),
        dict(cx="ids2"), dict(role="entry"), dict(drafter="Bad Name"),
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            F.encode({**lo, **kw})
    mid = controls()[1]
    for kw in (dict(salt="a" * 64), dict(ekey="b" * 64), dict(seed="0" * 16), dict(drafter="x")):
        with pytest.raises(ValueError):
            F.encode({**mid, **kw})                 # secrets never reach a mid hop
    tail = controls()[2]
    for kw in (dict(ekey=None), dict(seed=None), dict(salt=None), dict(next="Qm2")):
        with pytest.raises(ValueError):
            F.encode({**tail, **kw})
    take = controls()[12]                            # takeover leases like lease_open
    for kw in (dict(ekey=None), dict(nonce=None), dict(role="mid", next="Qm2"),
               dict(F=-2), dict(g_entry=-1), dict(lease=None), dict(ct="bf16", cx="ids")):
        with pytest.raises(ValueError):
            F.encode({**take, **kw})
    with pytest.raises(ValueError):
        F.encode({**lo, "cx": "ids", "ct": "bf16"})
    for m, kw in ((controls()[8], dict(pos_lo=129)), (controls()[13], dict(pos_hi=0)),
                  (controls()[13], dict(pos_hi=65536)), (controls()[14], dict(pos_hi=15)),
                  (controls()[14], dict(seg=[])), (controls()[14], dict(seg=[[0, 0, "bf16", "decode"]])),
                  (controls()[14], dict(seg=[[0, 1, "bf16", "draft"]])),
                  (controls()[14], dict(seg=[[U32, 1, "bf16", "decode"]])),
                  (controls()[14], dict(seg=[[0, 1, "bf16", "decode"]] * 5)),
                  (controls()[4], dict(sig=None)), (controls()[4], dict(truncate_to=-2)),
                  (controls()[7], dict(sid=SID)), (controls()[6], dict(cmd="closed")),
                  (controls()[5], dict(ck=rb(65537)))):
        with pytest.raises(ValueError):
            F.encode({**m, **kw})


def test_unknown_op_and_cmd():
    for obj in ({"v": 2, "op": "x"}, {"v": 2}, {"v": 2, "op": "c", "cmd": "nope"},
                {"v": 2, "op": "c", "cmd": ["drain"]}, {"v": 2, "op": "c"}):
        with pytest.raises(F.FrameError):
            F.decode(join(obj, b""))
    with pytest.raises(F.FrameError):
        F.decode(join([1, 2], b""))


# ── epochs ──────────────────────────────────────────────────────────────────────────────────────
def test_two_epochs_are_independent_fields():
    m = F.frame(**{**BASE, "rte": 11, "sep": 22}, n=1, kind="S", cpos=99, cx="bf16", x=hid(1))
    d = F.decode(body(m))
    assert (d["rte"], d["sep"]) == (11, 22)
    for swap in ({"rte": 22, "sep": 11}, {"sep": 22, "rte": 22}):
        assert F.decode(body({**m, **swap})) != d
    r = F.decode(body(F.reply(sid=SID, rte=11, seq=17, sep=22, pos=100, n=1, fen=True)))
    assert (r["sid"], r["rte"], r["seq"], r["sep"], r["fen"]) == (SID, 11, 17, 22, True)


# ── fuzz ────────────────────────────────────────────────────────────────────────────────────────
def _decode_or_frameerror(b):
    try:
        d = F.decode(b)
    except F.FrameError:
        return None
    assert body(d) == bytes(b), "accepted a non-canonical body"      # canonical invariant
    return d


def test_fuzz_mutations():
    rng = random.Random(1234)
    samples = [body(m) for m in frames()[:4] + replies()[:2] + controls()[2:3] + controls()[14:15]]
    accepted = 0
    for s in samples:
        for _ in range(1500):
            b = bytearray(s)
            op = rng.randrange(5)
            if op == 0:                                    # bit flip anywhere
                i = rng.randrange(len(b))
                b[i] ^= 1 << rng.randrange(8)
            elif op == 1:                                  # byte change inside the header region
                i = rng.randrange(min(len(b), 4 + struct.unpack_from("!I", s)[0]))
                b[i] = rng.randrange(256)
            elif op == 2:                                  # truncate
                del b[rng.randrange(len(b)):]
            elif op == 3:                                  # insert junk
                i = rng.randrange(len(b) + 1)
                b[i:i] = rng.randbytes(rng.randrange(1, 9))
            else:                                          # delete a slice
                i = rng.randrange(len(b))
                del b[i:i + rng.randrange(1, 9)]
            if _decode_or_frameerror(b) is not None:
                accepted += 1
    assert accepted > 0                                    # payload bit flips stay valid messages


def test_fuzz_every_truncation():
    for m in (frames()[2], replies()[0], controls()[0]):
        b = body(m)
        for k in range(len(b)):
            with pytest.raises(F.FrameError):
                F.decode(b[:k])


def test_fuzz_random_bodies_and_headers():
    rng = random.Random(99)
    pool = [0, 1, -1, 2, 65535, 65536, U32, U32 + 1, 2 ** 63, 2 ** 64, "", "f", "T", "bf16",
            "ids", SID, True, False, None, [], [1], {}, {"__t__": 0}, [[1, 2]], "x" * 300]
    keys = [k for k, *_ in F._FRAME] + ["cmd", "junk"]
    for _ in range(3000):
        if rng.random() < 0.3:
            b = rng.randbytes(rng.randrange(0, 64))
        else:
            obj = {"v": 2, "op": rng.choice("frc")}
            if obj["op"] == "c":
                obj["cmd"] = rng.choice(list(F._CONTROL))
            for k in rng.sample(keys, rng.randrange(len(keys))):
                obj[k] = rng.choice(pool)
            b = join(obj, rng.randbytes(rng.randrange(0, 16)))
        _decode_or_frameerror(b)


def test_hostile_sizes_fail_fast_without_allocating():
    blob8 = struct.pack("!Q", 0)
    cases = [
        struct.pack("!I", 0xFFFFFFFF) + b"{}",                            # header length lie
        struct.pack("!I", F.MAX_HEADER + 1) + b" " * (F.MAX_HEADER + 1),  # header over the cap
        join({"v": 2, "op": "f"}, blob8 * 100000),                        # 100k empty blobs
        join({"v": 2, "op": "f"}, struct.pack("!Q", 2 ** 62) + b"xx"),     # blob length lie
        raw(b"[" * 100000 + b"]" * 100000),                               # deep nesting
        raw(b'{"v": 2, "rte": ' + b"9" * 5000 + b"}"),                     # 5000-digit int
    ]
    m = frames()[1]
    hdr, rest = split(body(m))
    hdr["x"]["shape"] = [2 ** 40, 2 ** 40]                               # huge shape, small blob
    cases.append(join(hdr, rest))
    hdr["x"]["shape"] = [1] * 100000                                      # many dims
    cases.append(join(hdr, rest))
    for b in cases:
        with pytest.raises(F.FrameError):
            F.decode(b)


def test_header_cap_trips_before_parsing():
    """an 8 MB JSON list would parse into ~40 MB of Python objects; the cap refuses it unread"""
    import tracemalloc
    b = raw(b"[" + b"0," * 4_000_000 + b"0]")
    tracemalloc.start()
    try:
        with pytest.raises(F.FrameError):
            F.decode(b)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 1 << 20, f"decode peaked at {peak} B on an oversized header"


def test_int_literals_are_bounded_regardless_of_process_settings():
    """a library may lift int_max_str_digits process-wide. A ~1M-digit literal (it fits the header
    cap) then costs ~1 s to convert; the bounded parse must refuse it without converting."""
    import time
    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        b = raw(b'{"v": 2, "op": "f", "rte": ' + b"7" * (F.MAX_HEADER - 64) + b"}")
        t = time.perf_counter()
        with pytest.raises(F.FrameError):
            F.decode(b)
        assert time.perf_counter() - t < 0.2
    finally:
        sys.set_int_max_str_digits(old)


def test_unreferenced_trailing_blob_is_refused():
    """only the header is re-encoded for the canonical check, so the blob count needs its own rule"""
    for m in (frames()[1], controls()[6], replies()[1]):
        b = body(m)
        for extra in (struct.pack("!Q", 0), struct.pack("!Q", 3) + b"abc"):
            with pytest.raises(F.FrameError):
                F.decode(b + extra)


def test_blob_flood_is_capped_before_it_allocates():
    """a 256 MiB body can hold 32M empty blobs, and one memoryview each would be ~6 GB. The cap
    must trip before the list grows."""
    import tracemalloc
    b = join({"v": 2, "op": "c", "cmd": "drain"}, struct.pack("!Q", 0) * 1_000_000)
    tracemalloc.start()
    try:
        with pytest.raises(F.FrameError):
            F.decode(b)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 1 << 20, f"decode peaked at {peak} B on a blob flood"


def test_shape_dims_are_bounded_before_the_size_product():
    """40k dims of 2^62 fit a 1 MiB header, and their bignum product costs seconds of CPU per
    frame. The descriptor check must refuse them first."""
    import time
    hdr, rest = split(body(frames()[1]))
    hdr["x"]["shape"] = [2 ** 62] * 40000
    b = join(hdr, rest)
    t = time.perf_counter()
    with pytest.raises(F.FrameError):
        F.decode(b)
    assert time.perf_counter() - t < 0.5


# ── MAX_FRAME ───────────────────────────────────────────────────────────────────────────────────
def test_max_frame_matches_transport():
    from shard import transport as T
    assert F.MAX_FRAME == T.MAX_FRAME
    assert F.MAX_BLOBS == 4


def test_max_frame_enforced(monkeypatch):
    m = frames()[1]
    size = len(body(m))
    monkeypatch.setattr(F, "MAX_FRAME", size - 1)
    with pytest.raises(ValueError):
        F.encode(m)
    monkeypatch.setattr(F, "MAX_FRAME", size)
    F.encode(m)
    b = body(m)
    monkeypatch.setattr(F, "MAX_FRAME", size - 1)
    with pytest.raises(F.FrameError):
        F.decode(b)


def test_recv_refuses_oversize_length_before_reading_the_body(monkeypatch):
    monkeypatch.setattr(F, "MAX_FRAME", 1024)
    a, b = socket.socketpair()
    b.settimeout(2)                   # if recv tried to read the body it would time out, not raise
    a.sendall(struct.pack("!Q", 1025))
    with pytest.raises(F.FrameError):
        F.recv(b)
    a.close()
    b.close()


def test_recv_stalled_large_claim_pins_only_what_arrived():
    """a peer claiming a 200 MiB frame that then stalls must not make recv allocate 200 MiB"""
    import tracemalloc
    a, b = socket.socketpair()
    b.settimeout(0.5)
    a.sendall(struct.pack("!Q", 200 << 20) + b"\x00" * 1000)
    tracemalloc.start()
    try:
        with pytest.raises(OSError):                        # socket timeout: the peer stalled
            F.recv(b)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
        a.close()
        b.close()
    assert peak < 4 << 20, f"recv pinned {peak} B for a stalled 200 MiB claim"


def test_recv_large_frame_over_the_chunked_path():
    n, h = 2048, 4096                                          # 16 MiB bf16 prefill piece
    import tracemalloc
    m = F.frame(**BASE, n=n, kind="P", cpos=-1, cx="bf16", x=hid(n, h))
    size = len(body(m))
    a, b = socket.socketpair()
    b.settimeout(10)
    t = threading.Thread(target=F.send, args=(a, m))
    tracemalloc.start()
    t.start()
    try:
        got = F.recv(b)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
        t.join()
        a.close()
        b.close()
    assert got == m
    # one growing buffer plus a transient calloc'd pad (not resident); a join of chunks costs 2x
    assert peak < 1.6 * size, f"recv peaked at {peak / size:.2f}x the body"


def test_transport_resolves_to_the_package_module(monkeypatch):
    """a foreign top-level `transport` module must never be picked up by shard.frame"""
    from shard import transport as T
    monkeypatch.setitem(sys.modules, "transport", object())
    monkeypatch.setattr(F, "_T", None)
    assert F._transport() is T


def test_recv_truncated_then_closed_is_connection_error():
    a, b = socket.socketpair()
    b.settimeout(2)
    good = body(frames()[1])
    a.sendall(struct.pack("!Q", len(good)) + good[:50])
    a.close()
    with pytest.raises(ConnectionError):
        F.recv(b)
    b.close()


def test_header_cap_on_encode(monkeypatch):
    monkeypatch.setattr(F, "MAX_HEADER", 200)
    with pytest.raises(ValueError):
        F.encode(controls()[0])


def test_largest_realistic_prefill_fits():
    n, h = 2048, 4096                                      # MiMo-Flash prefill sub-chunk + DFlash tap
    m = F.frame(**BASE, n=n, kind="P", cpos=-1, cx="bf16", x=hid(n, h), ct="bf16", tap=hid(n, h, seed=2))
    b = body(m)
    assert len(b) < F.MAX_FRAME and F.decode(b)["tap"] == m["tap"]


# ── byte budget ─────────────────────────────────────────────────────────────────────────────────
def _overhead(m):
    """wire bytes (incl. the 8-byte length) minus row payload (x, xs, tap) and eid ciphertext"""
    payload = sum(len(m[k].data) for k in ("x", "xs", "tap") if m.get(k) is not None)
    payload += len(m["eid"]) if m.get("eid") is not None else 0
    return 8 + len(body(m)) - payload


def test_byte_budget():
    h, U = 4096, U32
    bf = lambda n: F.Tensor("bf16", (n, h), bytes(2 * n * h))             # noqa: E731
    bare = dict(sid=SID, rte=3, seq=1234, sep=7, pos=12345, n=1, kind="T", cpos=12344, lg="last")
    worst = dict(sid="f" * 16, rte=U, seq=U, sep=U, pos=U - 1, n=1, kind="S", cpos=U, lg="last",
                 hint={"lic": True}, ld=[255] * 7)
    fp8 = dict(cx="fp8", x=F.Tensor("fp8", (1, h), bytes(h)), xs=F.Tensor("f32", (1,), bytes(4)))
    cases = [
        # MiMo-Flash decode frame: 8 KiB bf16 row + a header of about 200 B (§5.1)
        ("decode bare", F.frame(**bare, cx="bf16", x=bf(1)), 256),
        ("decode 7-hop + eid", F.frame(**bare, cx="bf16", x=bf(1), ld=[200] * 7, eid=bytes(32)), 384),
        ("decode worst widths", F.frame(**worst, cx="bf16", x=bf(1), eid=bytes(68)), 448),
        ("decode fp8 worst", F.frame(**worst, **fp8, eid=bytes(68)), 512),
        ("decode + DFlash tap", F.frame(**worst, cx="bf16", x=bf(1), ct="bf16", tap=bf(1),
                                        eid=bytes(68)), 544),
        ("prefill 2048 + tap", F.frame(**{**worst, "pos": U - 2048, "n": 2048, "kind": "P",
                                          "sub": 65535, "last": False, "cpos": U - 1},
                                       cx="bf16", x=bf(2048), ct="bf16", tap=bf(2048)), 576),
        ("ids into head n=9", F.frame(**{**worst, "pos": U - 9, "n": 9}, cx="ids",
                                      x=F.Tensor("i32", (9,), bytes(36)), eid=bytes(100)), 512),
    ]
    for what, m, bound in cases:
        ov = _overhead(m)
        assert ov <= bound, f"{what}: {ov} B overhead > {bound}"
    assert _overhead(cases[0][1]) / (2 * h) < 0.035       # < 3.5% on the 8 KiB decode row
    rep = dict(sid="f" * 16, rte=U, seq=U, sep=U, pos=U - 9, n=9, ld=[255] * 7)
    totals = [
        ("reply decode", F.reply(**{**rep, "pos": U - 1, "n": 1}, toks=ids([2 ** 31 - 1])), 288),
        ("reply block+dt", F.reply(**rep, acc=True, toks=ids([1] * 9), blk=ids([1] * 8),
                                   dt=[[U, U]] * 7), 640),
        ("reply fenced", F.reply(**rep, fen=True), 224),
        ("ckpt (excl. ck)", F.control("ckpt", sid=SID, ck=bytes(300)), 160 + 300),
        ("lease_open solo worst (excl. lease)",
         F.control("lease_open", sid="f" * 16, rte=U, lo=0, hi=65535, role="solo", pred="a" * 128,
                   ret="b" * 128, backup="c" * 128, max_pos=U, W=65535, cx="fp8", ct="bf16",
                   drafter="d" * 64, seed="f" * 16, salt="a" * 64, ekey="b" * 64, nonce="c" * 128,
                   lease=bytes(256)), 1280 + 256),
    ]
    for what, m, bound in totals:
        tot = 8 + len(body(m))
        assert tot <= bound, f"{what}: {tot} B > {bound}"


# ── wire rows (what §7.1 receipts hash) ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("m", [f for f in frames() if not f["fen"]] + controls()[14:15],
                         ids=lambda m: f"{m.get('cmd', m.get('kind'))}-{m['cx']}-{m['ct']}")
def test_wire_rows_match_across_the_link(m):
    b = body(m)
    d = F.decode(b)
    n = m["n"] if m["op"] == "f" else m["pos_hi"] - m["pos_lo"]
    for i in range(n):
        sent, got = F.wire_row(m, i), F.wire_row(d, i)
        assert sent == got
        want = bytes(m["x"].row(i))
        want += bytes(m["xs"].row(i)) if m["xs"] is not None else b""
        want += bytes(m["tap"].row(i)) if m["tap"] is not None else b""
        assert sent == want                                  # x ‖ xs ‖ tap, nothing else (no eid)
        assert bytes(m["x"].row(i)) in b                     # the bytes hashed are the wire bytes
    with pytest.raises(IndexError):
        F.wire_row(m, n)


def test_wire_row_needs_a_payload():
    with pytest.raises(ValueError):
        F.wire_row(frames()[3], 0)                           # fenced


# ── device-agnostic tensors ─────────────────────────────────────────────────────────────────────
def test_decode_needs_neither_torch_nor_numpy():
    code = (
        "import sys; sys.modules['torch'] = None; sys.modules['numpy'] = None\n"
        "from shard import frame as F\n"
        "x = F.Tensor('bf16', (2, 8), bytes(range(32)))\n"
        "m = F.frame(sid='00000000000000aa', rte=1, seq=1, sep=0, pos=0, n=2, kind='P', cpos=-1,\n"
        "            lg='last', cx='bf16', x=x)\n"
        "b = b''.join(F.encode(m)); d = F.decode(b)\n"
        "assert d == m and F.wire_row(d, 1) == bytes(range(16, 32))\n"
        "assert 'torch' not in [k for k, v in sys.modules.items() if v is not None]\n"
        "print('ok')\n")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                         timeout=60)
    assert out.stdout.strip() == "ok", out.stderr


def test_tensor_views_match_torch_bits():
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    g = torch.Generator().manual_seed(0)
    src = torch.randn(3, 5, generator=g)
    for dt in (torch.bfloat16, torch.float16, torch.float32, torch.float8_e4m3fn, torch.int32,
               torch.uint8):
        t = src.to(dt) if dt not in (torch.int32, torch.uint8) else (src * 50).to(dt)
        w = F.Tensor.from_torch(t)
        assert torch.equal(w.torch().view(torch.uint8), t.contiguous().view(torch.uint8))
        assert w.numpy().tobytes() == t.contiguous().view(torch.uint8).numpy().tobytes()
    bf = F.Tensor.from_torch(src.to(torch.bfloat16)).numpy()
    assert bf.dtype == np.uint16 and bf.shape == (3, 5)      # raw bf16 bits without torch dtypes
    with pytest.raises(ValueError):
        F.Tensor.from_torch(torch.zeros(2, dtype=torch.int64))
    empty = F.Tensor.from_torch(torch.zeros(0, 4, dtype=torch.bfloat16))
    assert empty.torch().shape == (0, 4)


def test_decoded_payload_is_read_only():
    """receipts hash the received bytes; a consumer must not be able to edit them in place"""
    d = F.decode(bytearray(body(frames()[2])))
    assert d["x"].data.readonly and d["eid"].readonly
    np = pytest.importorskip("numpy")
    assert not d["x"].numpy().flags.writeable


def test_tensor_rejects_strided_and_big_endian_buffers():
    np = pytest.importorskip("numpy")
    for buf in (memoryview(bytes(12))[::2], np.zeros((4, 4), "<u2")[:, ::2],
                np.zeros(6, ">f4"), np.zeros(6, [("a", ">f4")])):
        with pytest.raises(ValueError):
            F.Tensor("u8" if isinstance(buf, memoryview) else "bf16" if buf.itemsize == 2 else "f32",
                     (6,) if isinstance(buf, memoryview) else buf.shape, buf)
    with pytest.raises(ValueError):
        F.frame(**BASE, n=1, kind="T", cpos=99, cx="bf16", x=hid(1), eid=memoryview(bytes(8))[::2])
    assert F.Tensor("f32", (2, 3), np.zeros((2, 3), "<f4")).data.nbytes == 24


def test_tensor_rejects_size_lies():
    for args in (("bf16", (2, 3), bytes(11)), ("bf16", (2, -3), b""), ("f64", (1,), bytes(8)),
                 ("i32", (True,), bytes(4))):
        with pytest.raises(ValueError):
            F.Tensor(*args)


# ── signed control ──────────────────────────────────────────────────────────────────────────────
def test_lease_update_can_clear_the_backup():
    m = controls()[4]
    cleared = {**m, "backup": ""}
    assert F.decode(body(cleared))["backup"] == ""
    assert F.decode(body(m))["backup"] is None                  # absent = unchanged
    assert F.signing_bytes(cleared) != F.signing_bytes(m)
    with pytest.raises(ValueError):
        F.encode({**controls()[0], "backup": ""})              # only lease_update may clear


def test_lease_update_signing_bytes():
    m = controls()[4]
    s = F.signing_bytes(m)
    assert s == F.signing_bytes({**m, "sig": rb(64, 9)})     # the signature is not signed
    assert s == F.signing_bytes({k: v for k, v in m.items() if k != "sig"})
    assert s != F.signing_bytes({**m, "truncate_to": 128})
    assert s != F.signing_bytes({**m, "rte": 3})
    assert b"sig" not in s and b"sig" in body(m)
