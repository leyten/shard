"""Wire-codec hardening against a hostile/malformed peer: a bad frame must be a DEAD EDGE
(ConnectionError), never an unbounded allocation. Covers BOTH inter-stage codecs, which share the
JSON-header + raw-blob format:
  - phase0/wire.py       — the PSK raw-TCP wire (ChaCha seal)
  - shard/transport.py   — the libp2p PRODUCTION transport (link auth is the sidecar's; no seal)

Two allocation vectors are closed here:
  1. the 8-byte frame-length prefix is attacker-controlled and read/buffered BEFORE the frame is
     authenticated, so an oversized claim must be refused before the body is read -> MAX_FRAME;
  2. _unpack trusted a tensor's declared shape, so an EMPTY blob + a HUGE shape drove
     torch.empty(huge) -> the declared shape's numel * dtype-size must equal the blob length.

Run: python3 -m pytest tests/test_wire_hardening.py -q
"""
import json
import os
import socket
import struct
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "phase0"))
sys.path.insert(0, os.path.join(_REPO, "shard"))

torch = pytest.importorskip("torch")
import wire            # noqa: E402  PSK codec
import transport       # noqa: E402  libp2p production codec

wire.use_key("test-secret-key")                      # the PSK wire needs an AEAD key for seal/open

CODECS = [wire, transport]
IDS = ["wire", "transport"]


def _pair():
    a, b = socket.socketpair()
    a.settimeout(3)
    b.settimeout(3)
    return a, b


def _framed(codec, plaintext):
    """Wrap a plaintext codec buffer as it arrives on the wire: sealed for the PSK wire, raw for the
    libp2p transport."""
    body = codec._seal(plaintext) if codec is wire else plaintext
    return struct.pack("!Q", len(body)) + body


def _tensor_frame(shape, blob, dtype="torch.float32"):
    """A plaintext codec buffer holding a single tensor node (header + one blob)."""
    head = json.dumps({"__t__": 0, "dtype": dtype, "shape": shape}).encode()
    return struct.pack("!I", len(head)) + head + struct.pack("!Q", len(blob)) + blob


# ---- 1. legit traffic still round-trips (regression guard for the added validation) ----------------

@pytest.mark.parametrize("codec", CODECS, ids=IDS)
def test_legit_roundtrip(codec):
    a, b = _pair()
    msg = {"op": "verify", "h": torch.randn(1, 5, 8, dtype=torch.bfloat16), "start": 7,
           "ids": [1, 2, 3], "flag": True, "empty": torch.empty(0, 8), "scalar": torch.tensor(3)}
    codec.send_msg(a, msg)
    got = codec.recv_msg(b)
    assert torch.equal(got["h"], msg["h"])
    assert got["start"] == 7 and got["ids"] == [1, 2, 3] and got["flag"] is True
    assert got["empty"].shape == (0, 8)               # zero-numel tensor survives the size check
    assert int(got["scalar"]) == 3                    # 0-dim tensor survives
    a.close(); b.close()


# ---- 2. oversized frame length is refused BEFORE the body is read ----------------------------------

@pytest.mark.parametrize("codec", CODECS, ids=IDS)
def test_oversized_length_rejected(codec):
    a, b = _pair()
    a.sendall(struct.pack("!Q", codec.MAX_FRAME + 1))   # claim one byte past the cap; send NO body
    with pytest.raises(ConnectionError):                # capped pre-alloc -> dead edge, not a 256 MiB buffer
        codec.recv_msg(b)
    a.close(); b.close()


@pytest.mark.parametrize("codec", CODECS, ids=IDS)
def test_length_at_cap_is_not_pre_rejected(codec, monkeypatch):
    """A frame exactly at MAX_FRAME must NOT be rejected by the cap (the guard is `> MAX_FRAME`).
    Shrink the cap so we can prove the boundary with a tiny real frame instead of a 256 MiB one."""
    a, b = _pair()
    payload = _framed(codec, _tensor_frame([2], struct.pack("<2f", 1.5, 2.5)))
    body_len = len(payload) - 8                          # the declared length field's value
    monkeypatch.setattr(codec, "MAX_FRAME", body_len)   # cap == this frame's exact length
    a.sendall(payload)
    got = codec.recv_msg(b)                              # boundary length passes; > cap would raise
    assert torch.allclose(got, torch.tensor([1.5, 2.5]))
    a.close(); b.close()


# ---- 3. a lying tensor shape must never drive torch.empty ------------------------------------------

@pytest.mark.parametrize("codec", CODECS, ids=IDS)
def test_empty_blob_huge_shape_rejected(codec):
    """The headline DoS: an empty blob paired with a huge declared shape used to hit
    `torch.empty(shape)` and allocate gigabytes. Now the size check rejects it with zero allocation."""
    with pytest.raises(ValueError):
        codec._unpack(_tensor_frame([100_000_000], b""))


@pytest.mark.parametrize("codec", CODECS, ids=IDS)
@pytest.mark.parametrize("shape,blob", [
    ([100], b"\x00" * 8),        # shape needs 400 bytes, blob has 8   (under-provisioned)
    ([1], b"\x00" * 400),        # shape needs 4 bytes,   blob has 400 (over-provisioned)
    ([2, 3], b"\x00" * 8),       # 2*3*4=24 needed, blob has 8
])
def test_shape_blob_size_mismatch_rejected(codec, shape, blob):
    with pytest.raises(ValueError):
        codec._unpack(_tensor_frame(shape, blob))


@pytest.mark.parametrize("codec", CODECS, ids=IDS)
@pytest.mark.parametrize("shape", [[-1], [2, -3], [1.5], ["x"]])
def test_bad_dim_rejected(codec, shape):
    with pytest.raises(ValueError):
        codec._unpack(_tensor_frame(shape, b""))


# ---- 4. end to end: recv_msg turns a hostile frame into ConnectionError (a dead edge), not a crash -

@pytest.mark.parametrize("codec", CODECS, ids=IDS)
def test_recv_msg_wraps_hostile_frame_as_connectionerror(codec):
    """A well-formed, authenticated frame whose CONTENT is hostile (empty blob + huge shape) must be
    absorbed by recv_msg as a ConnectionError so the per-edge supervision resets the link — it must
    not leak a ValueError up the stack (which would crash the stage and its warm weights)."""
    a, b = _pair()
    a.sendall(_framed(codec, _tensor_frame([100_000_000], b"")))
    with pytest.raises(ConnectionError):
        codec.recv_msg(b)
    a.close(); b.close()


def test_max_frame_default_is_sane():
    assert wire.MAX_FRAME == transport.MAX_FRAME == 256 * 1024 * 1024   # codecs stay in lockstep
