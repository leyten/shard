"""shard-lease/1: orchestrator-signed lease tokens and the node-side admission gate.

A lease token authorizes ONE stream edge of ONE session: node `to` serves session `sid` at route
epoch `rte` for layers [lo, hi) with `kv` tokens of KV reserved, forwards to `next` and replies to
`ret`, and accepts that session's frames (roles head/mid/tail/solo) or KV mirror (role backup) from
the PeerId `from` only, until `exp`
(ARCHITECTURE-v2 §4.6 lease-token admission instead of a static -allow list; C20 predecessor
binding). The orchestrator is the only signer at launch (the directory); node-signed leases come
with the DHT directory (design-mesh M6).

FLOW. The entry delivers each hop its token inside lease_open / takeover (frame.py's opaque
`lease` blob, <= 4096 bytes); a backup's token rides with its mirror_open. The node checks the
signature, that it is the audience, and that the message's own fields (sid, rte, lo, hi, role,
pred / primary, next, ret) equal the signed claims (`binding_of`), then admits `from` for that
session. It never matters who carried the token: everything the message could steer is bound, and
the sidecar tags every inbound stream with the verified remote PeerId, so frames from anyone but
`from` are dropped (`LeaseGate.accepts`). A path change bumps the route epoch: surviving hops
learn it from the entry-signed lease_update (C6, `LeaseGate.supersede`), replacement edges get new
orchestrator tokens (`Directory.reroute`). Backups get no lease_update (§5.3 sends it to hops), so
a backup follows the epoch its admitted primaries mirror at. An edge is honoured until its own
token's `exp`; a longer session renews (`Directory.reroute(..., renew=True)`).

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
  not_yet_valid now < iat - skew
  expired       now >= exp
  ttl           exp - iat > MAX_TTL_S or exp <= iat

REPLAY (LeaseGate, node side). Re-delivering a live token (a retried lease_open) is idempotent.
What a token must never do is resurrect a session the node has moved past or grant more than its
message claims, so admission additionally rejects
  binding       the carrying message's fields differ from the signed claims (a head token in a
                mid lease_open, a backup token as a primary, another span or route epoch)
  stale_rte     rte below the highest route epoch seen for the sid (a lease_update or a newer
                token superseded it)
  closed        the sid was closed on this node (sids are single-use: the directory never
                re-leases one within MAX_TTL_S)
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
    "next": re.compile(r"[A-Za-z0-9]{0,128}"),     # the hop's successor ("" at the tail / for a backup)
    "ret": re.compile(r"[A-Za-z0-9]{0,128}"),      # where the tail replies (the entry; "" for a backup)
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


def verify(token, orch_pub: str, *, audience: str, now: int, skew: int = 30) -> dict:
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
    if not 0 < claims["exp"] - claims["iat"] <= MAX_TTL_S:
        raise LeaseError("ttl", "lifetime out of bounds")
    if now < claims["iat"] - skew:
        raise LeaseError("not_yet_valid", f"iat {claims['iat']} > now {now} + {skew}")
    if now >= claims["exp"]:
        raise LeaseError("expired", f"exp {claims['exp']} <= now {now}")
    return claims


BIND_KEYS = ("sid", "lo", "hi", "role", "from", "next", "ret")


def binding_of(msg) -> dict:
    """The claims a decoded shard-frame/2 control message asserts about its token: lease_open and
    takeover bind sid, rte, span, role, pred, next and ret; mirror_open binds sid, span and its
    primary (the mirror source) under role backup, with no next or ret. admit() requires the
    token to carry exactly these."""
    cmd = msg["cmd"]
    if cmd in ("lease_open", "takeover"):
        return {"sid": msg["sid"], "rte": msg["rte"], "lo": msg["lo"], "hi": msg["hi"], "role": msg["role"],
                "from": msg["pred"], "next": msg.get("next") or "", "ret": msg["ret"]}
    if cmd == "mirror_open":
        return {"sid": msg["sid"], "lo": msg["lo"], "hi": msg["hi"], "role": "backup", "from": msg["primary"],
                "next": "", "ret": ""}
    raise ValueError(f"{cmd} carries no lease")


@dataclass
class _SidState:
    rte: int = 0
    frame: tuple | None = None                               # (pred, exp) of the frame edge
    mirrors: dict = field(default_factory=dict)              # (lo, hi) -> (primary, exp)
    horizon: int = 0          # no token this state guards against can be valid after this
    closed: bool = False


@dataclass
class LeaseGate:
    """A node's admission state (module docstring FLOW and REPLAY). One per node identity."""
    orch_pub: str
    me: str
    skew: int = 30
    _sids: dict = field(default_factory=dict)

    def admit(self, token, *, now: int, bind) -> dict:
        """Verify a delivered token and bind it to its carrying message (binding_of). Returns the
        claims; afterwards accepts(sid, rte, claims['from'], kind, now) holds until its exp. A
        newer route epoch keeps the session's other edges (a backup's other primaries, a hop's
        predecessor when only its backup token was reissued) and replaces this one."""
        missing = [k for k in BIND_KEYS if k not in bind]
        if missing:
            raise ValueError(f"bind must name {BIND_KEYS}, missing {missing}")
        c = verify(token, self.orch_pub, audience=self.me, now=now, skew=self.skew)
        for k, v in bind.items():
            if c.get(k) != v:
                raise LeaseError("binding", f"message {k}={v!r}, token {k}={c.get(k)!r}")
        st = self._sids.get(c["sid"])
        if st is not None:
            if st.closed:
                raise LeaseError("closed", f"session {c['sid']} was closed on this node")
            if c["rte"] < st.rte:
                raise LeaseError("stale_rte", f"rte {c['rte']} < {st.rte}")
        else:
            st = self._sids[c["sid"]] = _SidState()
        st.rte = max(st.rte, c["rte"])
        if c["role"] == "backup":
            st.mirrors[(c["lo"], c["hi"])] = (c["from"], c["exp"])
        else:
            st.frame = (c["from"], c["exp"])
        st.horizon = max(st.horizon, c["iat"] + MAX_TTL_S)   # older tokens were issued before iat
        return c

    def accepts(self, sid: str, rte: int, sender: str, kind: str, now: int) -> bool:
        """May `sender` (the verified PeerId of an inbound stream) deliver this session's frames
        (kind "frame", at exactly the current route epoch) or KV mirror (kind "mirror", at the
        current epoch or a newer one: backups learn epochs from their primaries) right now?"""
        st = self._sids.get(sid)
        if st is None or st.closed:
            return False
        if kind == "frame":
            return rte == st.rte and st.frame is not None and st.frame[0] == sender and now < st.frame[1]
        return rte >= st.rte and any(p == sender and now < e for p, e in st.mirrors.values())

    def supersede(self, sid: str, rte: int, now: int, pred: str | None = None) -> None:
        """An entry-signed lease_update moved the session to route epoch `rte` (C6): the edges
        carry over (with their tokens' expiry), the frame predecessor becomes `pred` when the update
        names one. Older route epochs are refused from now on."""
        st = self._sids.setdefault(sid, _SidState())
        if rte > st.rte:
            st.rte = rte
            if pred is not None and st.frame is not None:
                st.frame = (pred, st.frame[1])
        st.horizon = max(st.horizon, now + MAX_TTL_S)

    def close(self, sid: str, now: int) -> None:
        st = self._sids.setdefault(sid, _SidState())
        st.closed, st.horizon = True, max(st.horizon, now + MAX_TTL_S)

    def prune(self, now: int) -> None:
        """Forget a session once every token issued before its last event has expired: a token
        lives at most MAX_TTL_S from its iat, so nothing it guarded against can verify any more
        (and accepts() already refuses every edge past its exp)."""
        for sid in [s for s, st in self._sids.items() if st.horizon + self.skew <= now]:
            del self._sids[sid]
