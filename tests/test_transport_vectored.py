"""Vectored transport codec (lean-codec): send_msg hands the frame to the kernel as
scatter/gather segments (sendmsg) instead of concatenating every blob into one buffer and
then concatenating THAT with the length prefix (~2x the message in transient memcpy per hop
per direction on the aux-heavy EAGLE path).

The on-wire byte format is load-bearing — other nodes and older code speak it — so the core
guard here is BYTE-IDENTITY: the segmented pack path must produce exactly the bytes the
historical _pack produced, for representative payloads.

Run: python3 -m pytest tests/test_transport_vectored.py -q
"""
import json
import os
import socket
import struct
import sys
import threading

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "shard"))

torch = pytest.importorskip("torch")
import transport       # noqa: E402  libp2p production codec (shard/transport.py)


def _old_pack(obj) -> bytes:
    """the historical shard/transport._pack, verbatim — the wire-format oracle."""
    blobs = []

    def encode(o):
        if torch.is_tensor(o):
            t = o.detach().cpu().contiguous()
            blobs.append(t.reshape(-1).view(torch.uint8).numpy().tobytes())
            return {"__t__": len(blobs) - 1, "dtype": str(t.dtype), "shape": list(t.shape)}
        if isinstance(o, dict):
            return {k: encode(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [encode(v) for v in o]
        if o is None or isinstance(o, (bool, int, float, str)):
            return o
        raise TypeError(f"transport cannot encode {type(o).__name__}")

    head = json.dumps(encode(obj)).encode()
    out = bytearray(struct.pack("!I", len(head)) + head)
    for b in blobs:
        out += struct.pack("!Q", len(b)) + b
    return bytes(out)


def _payloads():
    torch.manual_seed(0)
    yield "control", {"op": "reset", "job": "j1", "k": 8, "ok": True, "f": 1.5, "none": None}
    yield "bf16", {"h": torch.randn(3, 7, 129, dtype=torch.bfloat16)}
    yield "fp8", {"h": torch.randn(2, 64).to(torch.float8_e4m3fn)}
    yield "mixed", {"ids": torch.arange(17, dtype=torch.int64), "mask": torch.zeros(4, dtype=torch.bool),
                    "nested": [{"t": torch.randn(5, 5)}, "s", 3], "empty": torch.empty(0, 4)}
    yield "noncontig", {"h": torch.randn(8, 8)[:, ::2]}


@pytest.mark.parametrize("name,obj", list(_payloads()), ids=[n for n, _ in _payloads()])
def test_pack_byte_identical_to_old(name, obj):
    assert transport._pack(obj) == _old_pack(obj)
    assert b"".join(bytes(p) for p in transport._pack_parts(obj)) == _old_pack(obj)


def test_send_msg_is_vectored():
    """send_msg must scatter/gather (sendmsg with the prefix + header + blobs as separate
    segments), never concatenate — and the segment stream must equal the legacy wire bytes."""
    obj = {"h": torch.randn(4, 33, dtype=torch.bfloat16), "aux": torch.randn(2, 9)}

    class FakeSock:
        def __init__(self):
            self.calls, self.sent_all = [], []

        def sendmsg(self, bufs):
            bufs = list(bufs)
            self.calls.append(len(bufs))
            n = sum(len(b) for b in bufs)
            self.sent_all.append(b"".join(bytes(b) for b in bufs))
            return n

        def sendall(self, data):        # the old path — must NOT be used
            raise AssertionError("send_msg concatenated the frame (sendall) instead of sendmsg")

    fs = FakeSock()
    n = transport.send_msg(fs, obj)
    wire = b"".join(fs.sent_all)
    body = _old_pack(obj)
    assert wire == struct.pack("!Q", len(body)) + body
    assert n == 8 + len(body)
    assert fs.calls and fs.calls[0] >= 3    # prefix, header, blobs went down as separate segments


def _recv_thread(sock, out):
    try:
        out.append(transport.recv_msg(sock))
    except Exception as e:              # surfaced by the main thread's assert
        out.append(e)


def _roundtrip(obj, sndbuf=None):
    a, b = socket.socketpair()
    try:
        if sndbuf:
            a.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, sndbuf)
        a.settimeout(10)
        b.settimeout(10)
        out = []
        t = threading.Thread(target=_recv_thread, args=(b, out))
        t.start()
        transport.send_msg(a, obj)
        t.join(10)
        assert out and not isinstance(out[0], Exception), f"recv failed: {out}"
        return out[0]
    finally:
        a.close()
        b.close()


def _assert_equal(sent, got):
    if torch.is_tensor(sent):
        assert got.dtype == sent.dtype and got.shape == sent.shape
        assert got.view(torch.uint8).numpy().tobytes() == \
            sent.contiguous().view(torch.uint8).numpy().tobytes()
    elif isinstance(sent, dict):
        assert set(got) == set(sent)
        for k in sent:
            _assert_equal(sent[k], got[k])
    elif isinstance(sent, (list, tuple)):
        assert len(got) == len(sent)
        for s, g in zip(sent, got):
            _assert_equal(s, g)
    else:
        assert got == sent


@pytest.mark.parametrize("name,obj", list(_payloads()), ids=[n for n, _ in _payloads()])
def test_roundtrip_over_socket(name, obj):
    got = _roundtrip(obj)
    _assert_equal({k: (v.contiguous() if torch.is_tensor(v) else v) for k, v in obj.items()}
                  if isinstance(obj, dict) else obj, got)


def test_partial_send_large_frame():
    """a frame far bigger than the send buffer forces partial sendmsg returns — the vectored
    sendall loop must resume mid-segment and deliver byte-exact."""
    obj = {"h": torch.randn(1500, 1500), "tag": "big"}
    got = _roundtrip(obj, sndbuf=8192)
    _assert_equal(obj["h"], got["h"])
    assert got["tag"] == "big"


def test_many_segments_exceeding_iov_cap():
    """>_IOV_CAP wire segments in one message (many small tensors) must still send: the loop
    caps each sendmsg call under the kernel's UIO_MAXIOV."""
    n_t = transport._IOV_CAP  # segments = 1 prefix + 1 header + 2*n_t > _IOV_CAP
    obj = {f"t{i}": torch.full((3,), float(i)) for i in range(n_t)}
    got = _roundtrip(obj)
    assert len(got) == n_t
    _assert_equal(obj["t7"], got["t7"])
