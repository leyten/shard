"""shard-lease/1: orchestrator-signed lease tokens and the node-side admission gate.

A lease token authorizes ONE stream edge of ONE session: `from` may dial `to` and have its frames
(or, for a backup, its KV mirror) accepted for session `sid` at route epoch `rte`, for layers
[lo, hi), with `kv` tokens of KV reserved, until `exp` (ARCHITECTURE-v2 §4.6: sidecar admission by
a lease token {job, sid, rte, from, to, exp} instead of a static -allow list; frame.py carries it
as the opaque `lease` blob of lease_open, <= 4096 bytes). The orchestrator is the only signer at
launch (the directory); node-signed leases come with the DHT directory (design-mesh M6).

WIRE FORM. `b64url(payload) "." b64url(sig)`, no padding, where payload is the canonical claims
JSON (sorted keys, no whitespace, ASCII only) and sig = Ed25519(orchestrator key,
DOMAIN || payload). The verifier checks the signature over the exact payload bytes before parsing
them, so no JSON canonicalization is needed on the verify side of either implementation;
DOMAIN separates lease signatures from anything else the same key signs.

WHAT VERIFY REJECTS (each with a stable error code, shared with the TypeScript implementation):
  format        not two canonical unpadded base64url parts, or over MAX_TOKEN bytes
  signature     not signed by the pinned orchestrator key over DOMAIN || payload
  schema        payload not a JSON object with exactly CLAIMS, each of its type and charset
                (ints only, no floats or bools; ids restricted to token-safe charsets)
  issuer        `iss` is not the pinned key (belt and braces: the signature already binds it)
  audience      `to` is not the verifying node
  sender        `from` is not the authenticated PeerId of the dialer (a stolen token is useless:
                the sidecar tags inbound streams with the verified remote PeerId)
  not_yet_valid now < iat - skew
  expired       now >= exp
  ttl           exp - iat > MAX_TTL_S or exp <= iat

REPLAY (LeaseGate, node side). Re-presenting a live token is legitimate: a pooled stream that
resets is re-dialled with the same token (ARCHITECTURE C20). What a token must never do is
resurrect a session the node has moved past, so the gate additionally rejects
  stale_rte     rte below the highest route epoch seen for the sid (a lease_update or a newer
                token superseded it)
  closed        the sid was closed on this node (until every token for it has expired)
"""
from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519

from ..manifest import pub_b64

TYP = "shard-lease/1"
DOMAIN = b"shard-lease/1\n"
MAX_TOKEN = 4096                  # frame.py lease blob bound
MAX_TTL_S = 24 * 3600
U32 = (1 << 32) - 1
ROLES = ("head", "mid", "tail", "solo", "backup")

_PEER = r"[A-Za-z0-9]{1,128}"
_CLAIMS = {
    "typ": re.compile(re.escape(TYP)),
    "lid": re.compile(r"[0-9a-f]{16}"),
    "job": re.compile(r"[A-Za-z0-9_.:-]{1,64}"),
    "sid": re.compile(r"[0-9a-f]{16}"),
    "model": re.compile(r"[A-Za-z0-9_.:/-]{1,128}"),
    "cell": re.compile(r"[a-z0-9][a-z0-9_.:-]{0,63}"),
    "role": re.compile("|".join(ROLES)),
    "from": re.compile(_PEER),
    "to": re.compile(_PEER),
    "iss": re.compile(r"[A-Za-z0-9+/]{43}="),
    "rte": U32, "lo": 0xFFFF, "hi": 0xFFFF, "kv": U32, "iat": U32, "exp": U32,
}
CLAIMS = tuple(sorted(_CLAIMS))
_B64URL = re.compile(r"[A-Za-z0-9_-]+")


class LeaseError(Exception):
    """A token or a presentation was refused. `code` is one of the module docstring's codes."""

    def __init__(self, code: str, msg: str = ""):
        super().__init__(f"{code}: {msg}" if msg else code)
        self.code = code


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _unb64u(s: str) -> bytes:
    if not _B64URL.fullmatch(s) or len(s) % 4 == 1:
        raise LeaseError("format", "not unpadded base64url")
    raw = base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
    if _b64u(raw) != s:                                  # non-canonical trailing bits
        raise LeaseError("format", "non-canonical base64url")
    return raw


def check_claims(c) -> dict:
    """Schema check shared by issue and verify: exactly CLAIMS, each of its type and charset."""
    if not isinstance(c, dict) or set(c) != set(_CLAIMS):
        missing = sorted(set(_CLAIMS) - set(c)) if isinstance(c, dict) else []
        extra = sorted(set(c) - set(_CLAIMS)) if isinstance(c, dict) else []
        raise LeaseError("schema", f"claims must be exactly {CLAIMS} (missing {missing}, extra {extra})")
    for k, rule in _CLAIMS.items():
        v = c[k]
        if isinstance(rule, int):
            if type(v) is not int or not 0 <= v <= rule:
                raise LeaseError("schema", f"{k} must be an int in [0, {rule}]")
        elif type(v) is not str or not rule.fullmatch(v):
            raise LeaseError("schema", f"{k} {v!r} is not a valid {k}")
    if not c["lo"] < c["hi"]:
        raise LeaseError("schema", "lo < hi")
    if c["from"] == c["to"]:
        raise LeaseError("schema", "from == to")
    return c


def canonical_payload(claims: dict) -> bytes:
    return json.dumps(claims, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def issue(priv: ed25519.Ed25519PrivateKey, claims: dict) -> str:
    """Sign claims (all of CLAIMS except `iss`, which is stamped here) into a token string."""
    c = dict(claims, iss=pub_b64(priv))
    check_claims(c)
    if not 0 < c["exp"] - c["iat"] <= MAX_TTL_S:
        raise LeaseError("ttl", f"exp - iat must be in (0, {MAX_TTL_S}]")
    payload = canonical_payload(c)
    return f"{_b64u(payload)}.{_b64u(priv.sign(DOMAIN + payload))}"


def _no_dupes(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise LeaseError("schema", f"duplicate key {k}")
        d[k] = v
    return d


def _no_const(s):
    raise LeaseError("schema", f"non-finite number {s}")


def verify(token, orch_pub: str, *, audience: str, sender: str, now: int, skew: int = 30) -> dict:
    """Fail closed: return the claims, or raise LeaseError with the first failing check's code."""
    if isinstance(token, (bytes, bytearray, memoryview)):
        try:
            token = bytes(token).decode("ascii")
        except UnicodeDecodeError:
            raise LeaseError("format", "not ASCII") from None
    if not isinstance(token, str) or len(token) > MAX_TOKEN or token.count(".") != 1:
        raise LeaseError("format", "expected <payload>.<sig>")
    p64, s64 = token.split(".")
    payload, sig = _unb64u(p64), _unb64u(s64)
    try:
        pub = ed25519.Ed25519PublicKey.from_public_bytes(base64.b64decode(orch_pub, validate=True))
        pub.verify(sig, DOMAIN + payload)
    except (InvalidSignature, ValueError, binascii.Error) as e:
        raise LeaseError("signature", type(e).__name__) from None

    try:
        claims = json.loads(payload.decode("ascii"), object_pairs_hook=_no_dupes, parse_constant=_no_const)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise LeaseError("schema", f"payload is not JSON: {type(e).__name__}") from None
    check_claims(claims)
    if claims["iss"] != orch_pub:
        raise LeaseError("issuer", "iss is not the pinned orchestrator key")
    if claims["to"] != audience:
        raise LeaseError("audience", f"token is for {claims['to']}, not {audience}")
    if claims["from"] != sender:
        raise LeaseError("sender", f"token admits {claims['from']}, not {sender}")
    if not 0 < claims["exp"] - claims["iat"] <= MAX_TTL_S:
        raise LeaseError("ttl", "lifetime out of bounds")
    if now < claims["iat"] - skew:
        raise LeaseError("not_yet_valid", f"iat {claims['iat']} > now {now} + {skew}")
    if now >= claims["exp"]:
        raise LeaseError("expired", f"exp {claims['exp']} <= now {now}")
    return claims


@dataclass
class _SidState:
    rte: int = 0
    horizon: int = 0          # no token this state guards against can be valid after this
    closed: bool = False


@dataclass
class LeaseGate:
    """A node's admission state (module docstring REPLAY). One per node identity."""
    orch_pub: str
    me: str
    skew: int = 30
    _sids: dict = field(default_factory=dict)

    def admit(self, token, *, sender: str, now: int) -> dict:
        c = verify(token, self.orch_pub, audience=self.me, sender=sender, now=now, skew=self.skew)
        st = self._sids.get(c["sid"])
        if st is not None:
            if st.closed:
                raise LeaseError("closed", f"session {c['sid']} was closed on this node")
            if c["rte"] < st.rte:
                raise LeaseError("stale_rte", f"rte {c['rte']} < {st.rte}")
        else:
            st = self._sids[c["sid"]] = _SidState()
        st.rte = max(st.rte, c["rte"])
        st.horizon = max(st.horizon, c["iat"] + MAX_TTL_S)   # older tokens were issued before iat
        return c

    def supersede(self, sid: str, rte: int, now: int) -> None:
        """An entry-signed lease_update moved the session to route epoch `rte`."""
        st = self._sids.setdefault(sid, _SidState())
        st.rte, st.horizon = max(st.rte, rte), max(st.horizon, now + MAX_TTL_S)

    def close(self, sid: str, now: int) -> None:
        st = self._sids.setdefault(sid, _SidState())
        st.closed, st.horizon = True, max(st.horizon, now + MAX_TTL_S)

    def prune(self, now: int) -> None:
        """Forget a session once every token issued before its last event has expired: a token
        lives at most MAX_TTL_S from its iat, so nothing it guarded against can verify any more."""
        for sid in [s for s, st in self._sids.items() if st.horizon + self.skew <= now]:
            del self._sids[sid]
