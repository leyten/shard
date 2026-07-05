"""inter-stage transport — the part shard owns, the wedge.

The networking lives in the libp2p sidecar (a Go daemon — see ../sidecar): identity,
encryption, NAT traversal, and the peer connections are its job. The engine keeps its
plain socket code and talks to the sidecar over localhost; the sidecar carries each
connection to/from the right ring neighbour over libp2p. So this module is just the
*message* layer the engine speaks over that socket — `send_msg`/`recv_msg`, a drop-in
for phase0/wire.py minus the ChaCha seal (libp2p already encrypts + authenticates the
link). The engine swaps transport by one import: `import wire` → `import shard.transport
as wire`.

Per the boundary law (docs/INTEGRATION.md) this is pure engine plumbing: peers and
bytes, nothing about c0mpute.

frame wire: 8-byte big-endian length + JSON header + raw tensor blobs (matches the
phase0/wire.py framing so call sites are unchanged).
"""
import json
import os
import socket
import struct

import numpy as np
import torch

# Upper bound on one inbound frame. The sidecar's libp2p link authenticates the peer but
# authorizes nothing, so a ring neighbour (or anyone who dials the sidecar) can stream an
# arbitrary length prefix; without a cap, recv_msg buffers up to 2**64 bytes into RAM. A
# real activation/KV frame is far below this; raise SHARD_MAX_FRAME if a larger model needs it.
_MAX_FRAME = int(os.environ.get("SHARD_MAX_FRAME", str(2 << 30)))   # 2 GiB


def _read_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("sidecar closed the connection")
        buf += chunk
    return bytes(buf)


# ── message codec (engine protocol): JSON header + raw tensor blobs ──
# Ported from phase0/wire.py's _pack/_unpack — the *message* format the engine speaks
# (control fields + tensors). No ChaCha seal: encryption + identity are the sidecar's job.
_DTYPES = {str(d): d for d in (torch.float32, torch.float16, torch.bfloat16,
                               torch.float8_e4m3fn, torch.float8_e5m2,   # fp8 activation transport (M25_FP8_WIRE)
                               torch.int64, torch.int32, torch.uint8, torch.bool)}


def _pack(obj) -> bytes:
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


def _unpack(buf: bytes):
    (hlen,) = struct.unpack_from("!I", buf, 0)
    head = json.loads(buf[4:4 + hlen])
    blobs, off = [], 4 + hlen
    while off < len(buf):
        (blen,) = struct.unpack_from("!Q", buf, off); off += 8
        blobs.append(buf[off:off + blen]); off += blen

    def decode(node):
        if isinstance(node, dict):
            if "__t__" in node:
                dt = _DTYPES.get(node["dtype"])
                if dt is None:
                    raise ValueError(f"unknown tensor dtype {node['dtype']!r}")
                shape, blob = node["shape"], blobs[node["__t__"]]
                if not blob:                              # an empty blob is only legit for a 0-element tensor; a
                    numel = 1                             # non-empty shape with no bytes is a memory-amplification
                    for d in shape: numel *= d            # bomb -- torch.empty would allocate the full shape from a
                    if numel != 0:                        # few header bytes, so reject it as a malformed frame
                        raise ValueError(f"empty blob for non-empty tensor shape {shape}")
                    return torch.empty(shape, dtype=dt)
                raw = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8).copy())
                return raw.view(dt).reshape(shape)
            return {k: decode(v) for k, v in node.items()}
        if isinstance(node, list):
            return [decode(v) for v in node]
        return node

    return decode(head)


def send_msg(sock: socket.socket, obj) -> int:
    """pack + send one message over a (sidecar-tunneled) socket; returns bytes on the wire.
    drop-in for phase0/wire.send_msg minus the SHARD_PSK seal (libp2p encrypts the link)."""
    frame = _pack(obj)
    sock.sendall(struct.pack("!Q", len(frame)) + frame)
    return 8 + len(frame)


def recv_msg(sock: socket.socket):
    """receive + unpack one message. drop-in for phase0/wire.recv_msg.
    A frame we can't parse into a message is a DEAD EDGE, not a process death: re-raise as
    ConnectionError so `except EDGE_ERRORS` supervision resets the connection (wire.py has carried
    this guard since the pickle removal; this production path lost it — one malformed frame, e.g. a
    PSK-mode peer dialing a libp2p-mode ring, crashed the stage and its warm weights)."""
    (n,) = struct.unpack("!Q", _read_exact(sock, 8))
    if n > _MAX_FRAME:                                    # refuse before allocating: a bogus length is a dead
        raise ConnectionError(f"frame length {n} exceeds SHARD_MAX_FRAME ({_MAX_FRAME})")   # edge, not an OOM
    frame = _read_exact(sock, n)
    try:
        return _unpack(frame)
    except (ValueError, KeyError, IndexError, TypeError, RuntimeError, struct.error, json.JSONDecodeError) as e:
        raise ConnectionError(f"malformed frame ({type(e).__name__}: {str(e)[:80]})") from e


# No-ops so `import shard.transport as wire` is a perfect drop-in: the libp2p sidecar
# provides encryption + identity, so there is no pre-shared key to load.
def key_from_env(var: str = "SHARD_PSK") -> None:
    return None


def use_key(material) -> None:
    return None
