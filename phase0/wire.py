"""shard wire: authenticated, pickle-free framing for every inter-stage edge.

phase 0 sent each message as raw pickle over plaintext TCP. on a WAN swarm that is
two separate problems:

  1. `recv_msg` ran `pickle.loads` on bytes straight off an untrusted socket -- an
     unauthenticated peer (or anyone who can reach the port) gets remote code
     execution, not just a malformed message.
  2. the activation tensor crossed the public internet in cleartext, so a passive
     observer on the path saw the same hidden states a malicious node would.

this module fixes the *wire*, not the *endpoints*. a participating node still has to
decrypt to run its layer, so it necessarily sees the activations it processes -- that
trust is unavoidable and is documented as such (see README "Private"). what changes:

  - NO pickle. a message is a typed frame: a JSON header (control fields, tensor
    dtype/shape) + raw little-endian tensor bytes. the receive path never constructs
    an arbitrary object, so a hostile frame is at worst a parse error -- never code
    execution, from any sender.
  - authenticated encryption. each frame is sealed with ChaCha20-Poly1305 under a
    pre-shared swarm key (env `SHARD_PSK`), fresh 96-bit nonce per frame. a passive
    observer learns nothing; a tampered or forged frame fails the auth tag and is
    rejected as a dead edge (ConnectionError -> the existing per-edge supervision
    resets it). confidentiality + integrity + sender-holds-the-key, on the wire.

key distribution here is a pre-shared secret -- right for a trusted/known deployment.
a permissionless swarm needs per-node identities and a handshake (Noise / QUIC-TLS
keyed by `cwt_` worker tokens); that is the transport's phase-1 follow-up, not this.

drop-in: `send_msg`/`recv_msg` keep their phase-0 signatures, so call sites are
unchanged. each entry point calls `wire.key_from_env()` once at startup.
"""
import hashlib, json, os, socket, struct
from json import JSONDecodeError
import numpy as np
import torch
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.exceptions import InvalidTag

_KEY = None                                  # 32-byte AEAD key, set once per process
_NONCE = 12                                  # ChaCha20-Poly1305 nonce length
# torch dtype <-> stable JSON-clean name, so the header serializes without pickle
_DTYPES = {str(d): d for d in (torch.float32, torch.float16, torch.bfloat16,
                               torch.float8_e4m3fn, torch.float8_e5m2,   # fp8 activation transport (M25_FP8_WIRE) — keep in lockstep with shard/transport.py
                               torch.int64, torch.int32, torch.uint8, torch.bool)}
_DSIZE = {d: torch.empty(0, dtype=d).element_size() for d in set(_DTYPES.values())}  # bytes/element, to size-check a blob

# Hard cap on one inbound frame's declared length. n is read (and the body buffered) BEFORE the AEAD
# seal is verified, so an unauthenticated peer's oversized claim must be refused BEFORE allocating the
# body — else any peer that can reach the port OOMs the node. Env M25_MAX_FRAME; default 256 MiB.
MAX_FRAME = int(os.environ.get("M25_MAX_FRAME") or 256 * 1024 * 1024)


def use_key(material):
    """set the process AEAD key from a shared secret (passphrase or hex, any length)."""
    global _KEY
    _KEY = hashlib.sha256(material.encode() if isinstance(material, str) else material).digest()


def key_from_env(var="SHARD_PSK"):
    """load the shared swarm key from the environment. fail loudly if unset -- there
    is deliberately no plaintext fallback, which would defeat the point."""
    v = os.environ.get(var)
    if not v:
        raise SystemExit(f"[wire] {var} is not set. export the same shared secret on "
                         f"every node, e.g.  export {var}=$(openssl rand -hex 32)")
    use_key(v)


# ---- pickle-free codec: object <-> bytes (JSON skeleton + raw tensor blobs) ----
def _pack(obj):
    blobs = []

    def encode(o):
        if torch.is_tensor(o):
            t = o.detach().cpu().contiguous()
            blobs.append(t.reshape(-1).view(torch.uint8).numpy().tobytes())
            return {"__t__": len(blobs) - 1, "dtype": str(t.dtype), "shape": list(t.shape)}
        if isinstance(o, dict):
            return {k: encode(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [encode(v) for v in o]          # tuples arrive as lists (safe: callers unpack positionally)
        if o is None or isinstance(o, (bool, int, float, str)):
            return o                               # JSON scalar passes through verbatim
        raise TypeError(f"wire cannot encode {type(o).__name__}")

    head = json.dumps(encode(obj)).encode()
    out = bytearray(struct.pack("!I", len(head)) + head)
    for b in blobs:
        out += struct.pack("!Q", len(b)) + b
    return bytes(out)


def _unpack(buf):
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
                numel = 1
                for d in shape:
                    if not isinstance(d, int) or d < 0:
                        raise ValueError(f"bad tensor dim {d!r}")
                    numel *= d
                if numel * _DSIZE[dt] != len(blob):    # declared shape must match the blob EXACTLY — a lying
                    raise ValueError(                  # shape (esp. an empty blob + huge shape) must never
                        f"tensor {shape} needs {numel * _DSIZE[dt]} bytes, blob has {len(blob)}")  # drive torch.empty
                if numel == 0:
                    return torch.empty(shape, dtype=dt)
                raw = torch.from_numpy(np.frombuffer(blob, dtype=np.uint8).copy())
                return raw.view(dt).reshape(shape)
            return {k: decode(v) for k, v in node.items()}
        if isinstance(node, list):
            return [decode(v) for v in node]
        return node

    return decode(head)


# ---- authenticated encryption: seal/open one frame ----
def _require_key():
    if _KEY is None:
        raise RuntimeError("wire key not set; call wire.key_from_env() at startup")


def _seal(plaintext):
    _require_key()
    nonce = os.urandom(_NONCE)
    return nonce + ChaCha20Poly1305(_KEY).encrypt(nonce, plaintext, None)


def _open(frame):
    _require_key()
    return ChaCha20Poly1305(_KEY).decrypt(frame[:_NONCE], frame[_NONCE:], None)


# ---- length-prefixed, sealed message transport (drop-in for the phase-0 pickle pair) ----
def _recvall(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return buf


def send_msg(sock, obj):
    frame = _seal(_pack(obj))
    sock.sendall(struct.pack("!Q", len(frame)) + frame)
    return 8 + len(frame)                          # bytes on the wire, for edge instrumentation


def recv_msg(sock):
    (n,) = struct.unpack("!Q", _recvall(sock, 8))
    if n > MAX_FRAME:                                  # refuse an oversized length BEFORE buffering the body (pre-auth)
        raise ConnectionError(f"frame length {n} exceeds MAX_FRAME ({MAX_FRAME})")
    frame = _recvall(sock, n)
    # any frame we can't authenticate AND parse into a message is a dead edge: re-raise as
    # ConnectionError so the existing `except EDGE_ERRORS` supervision resets the connection
    # instead of trusting it or crashing the node. an unauthenticated sender never gets past
    # _open (the RCE that pickle.loads exposed is gone); a buggy authenticated peer can at
    # worst produce a parse error, never code execution.
    try:
        return _unpack(_open(frame))
    except InvalidTag:
        raise ConnectionError("frame failed authentication (tampered or wrong SHARD_PSK)")
    except (ValueError, KeyError, IndexError, struct.error, JSONDecodeError) as e:
        raise ConnectionError(f"malformed frame ({type(e).__name__}: {e})")


if __name__ == "__main__":                         # self-test: round-trip exact, tamper + wrong key rejected
    use_key("test-swarm-secret")
    msg = {"op": "verify", "h": torch.randn(1, 5, 8, dtype=torch.bfloat16),
           "start": 12, "crop": None, "token_ids": [1, 2, 3], "par": [-1, 0], "dep": [0, 1]}
    got = _unpack(_open(_seal(_pack(msg))))
    assert got["op"] == "verify" and got["start"] == 12 and got["crop"] is None
    assert got["token_ids"] == [1, 2, 3] and got["par"] == [-1, 0] and got["dep"] == [0, 1]
    assert got["h"].dtype == torch.bfloat16 and torch.equal(got["h"], msg["h"])
    # bare tensor (node.py) and a (tensor, int) tuple (node_kv) round-trip too
    assert torch.equal(_unpack(_open(_seal(_pack(msg["h"])))), msg["h"])
    h2, s2 = _unpack(_open(_seal(_pack((msg["h"], 7)))))      # tuple -> list, unpacks the same
    assert s2 == 7 and torch.equal(h2, msg["h"])
    # a flipped byte anywhere in the frame -> rejected
    frame = bytearray(_seal(_pack(msg))); frame[-1] ^= 1
    try:
        _open(bytes(frame)); raise SystemExit("FAIL: tamper not detected")
    except InvalidTag:
        pass
    # a frame sealed under a different key -> rejected
    good = _seal(_pack({"op": "reset"}))
    use_key("a-different-secret")
    try:
        _open(good); raise SystemExit("FAIL: wrong key accepted")
    except InvalidTag:
        pass
    print("wire.py self-test PASSED: tensor+control round-trip exact, tamper rejected, wrong key rejected")
