"""shard-lease/1: issue/verify round trip, every tamper fails closed, the node-side replay gate."""
import base64
import json

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from shard import frame
from shard.manifest import pub_b64
from shard.mesh import lease as L

KEY = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
PUB = pub_b64(KEY)
OTHER = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(1, 33)))
CLAIMS = {"typ": L.TYP, "lid": "00112233445566ff", "job": "job-1", "sid": "0123456789abcdef", "rte": 3,
          "model": "XiaomiMiMo/MiMo-V2.6-Flash", "cell": "eu-1", "role": "mid", "from": "PeerA",
          "to": "PeerB", "lo": 8, "hi": 16, "kv": 4096, "iat": 1_000, "exp": 4_600}


def tok(**over):
    return L.issue(KEY, {**CLAIMS, **over})


def ok(token, **kw):
    args = {"audience": "PeerB", "sender": "PeerA", "now": 2_000, **kw}
    return L.verify(token, PUB, **args)


def code(token, **kw):
    with pytest.raises(L.LeaseError) as e:
        ok(token, **kw)
    return e.value.code


def resign(payload: bytes, key=KEY) -> str:
    return f"{L._b64u(payload)}.{L._b64u(key.sign(L.DOMAIN + payload))}"


def test_round_trip_and_bounds():
    t = tok()
    c = ok(t)
    assert c == {**CLAIMS, "iss": PUB}
    assert len(t) < L.MAX_TOKEN
    assert ok(t.encode()) == c                             # the frame carries it as bytes


def test_rides_lease_open_as_the_opaque_token_blob():
    t = tok(role="head").encode()
    msg = {"v": 2, "op": "c", "cmd": "lease_open", "sid": CLAIMS["sid"], "rte": 3, "lo": 8, "hi": 16,
           "role": "mid", "pred": "PeerA", "next": "PeerC", "ret": "Entry", "max_pos": 32768, "W": 8, "cx": "bf16",
           "nonce": "00" * 8, "lease": t}
    back = frame.decode(b"".join(bytes(p) for p in frame.encode(msg)))
    assert bytes(back["lease"]) == t and ok(bytes(back["lease"]))["role"] == "head"


@pytest.mark.parametrize("field,val", [("rte", 4), ("lo", 7), ("hi", 48), ("to", "PeerC"), ("from", "PeerX"),
                                       ("kv", 1 << 20), ("exp", 99_999), ("role", "tail"), ("sid", "f" * 16),
                                       ("job", "job-2"), ("model", "other/model"), ("cell", "us-1")])
def test_any_changed_claim_breaks_the_signature(field, val):
    p64, s64 = tok().split(".")
    claims = json.loads(L._unb64u(p64))
    claims[field] = val
    forged = f"{L._b64u(L.canonical_payload(claims))}.{s64}"
    assert code(forged) == "signature"


def test_bit_flips_anywhere_are_rejected():
    t = tok()
    for i in range(len(t)):
        if t[i] == ".":
            continue
        flipped = t[:i] + ("A" if t[i] != "A" else "B") + t[i + 1:]
        assert code(flipped) in ("signature", "format"), i


def test_other_keys_and_domains_are_rejected():
    assert code(L.issue(OTHER, CLAIMS)) == "signature"      # validly signed by someone else
    payload = L.canonical_payload({**CLAIMS, "iss": PUB})
    no_domain = f"{L._b64u(payload)}.{L._b64u(KEY.sign(payload))}"
    assert code(no_domain) == "signature"                   # the same key, without DOMAIN


def test_iss_must_be_the_pinned_key_even_when_signed_by_it():
    payload = L.canonical_payload({**CLAIMS, "iss": pub_b64(OTHER)})
    assert code(resign(payload)) == "issuer"


@pytest.mark.parametrize("mutate", [
    lambda c: c.pop("kv"),                                  # missing claim
    lambda c: c.update(extra=1),                            # unknown claim
    lambda c: c.update(rte=3.0),                            # float where an int belongs
    lambda c: c.update(rte=True),                           # bool is not an int
    lambda c: c.update(rte=-1),
    lambda c: c.update(rte=1 << 32),
    lambda c: c.update(to="Peer-B"),                        # outside the PeerId charset
    lambda c: c.update(sid="0123456789ABCDEF"),             # sids are lowercase hex
    lambda c: c.update(role="relay"),
    lambda c: c.update(lo=16),                              # empty layer range
    lambda c: c.update(**{"from": "PeerB"}),                # self-edge
])
def test_schema_is_strict_even_under_a_valid_signature(mutate):
    c = {**CLAIMS, "iss": PUB}
    mutate(c)
    payload = json.dumps(c, sort_keys=True, separators=(",", ":")).encode()
    assert code(resign(payload)) == "schema"


def test_payload_json_tricks_are_rejected():
    body = json.dumps({**CLAIMS, "iss": PUB}, sort_keys=True, separators=(",", ":"))
    dup = body[:-1] + ',"rte":9}'
    assert code(resign(dup.encode())) == "schema"
    assert code(resign(body.replace('"kv":4096', '"kv":NaN').encode())) == "schema"
    assert code(resign(b"[1,2]")) == "schema"
    assert code(resign(b"\xff\xfe")) == "schema"


@pytest.mark.parametrize("bad", ["", "abc", "a.b.c", "a b.c", "=abc.def", tok() + "=", tok().replace(".", "..", 1),
                                 "x" * (L.MAX_TOKEN + 1)])
def test_malformed_wire_forms(bad):
    assert code(bad) in ("format", "signature")


def test_noncanonical_base64_is_rejected():
    """A 64-byte signature is 86 base64url chars: the last one carries 4 unused bits. Setting one
    decodes to the same bytes, so a lenient decoder would accept a second spelling of the token."""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    p64, s64 = tok().split(".")
    alt = s64[:-1] + alphabet[alphabet.index(s64[-1]) | 1]
    assert base64.urlsafe_b64decode(alt + "==") == base64.urlsafe_b64decode(s64 + "==")
    assert code(f"{p64}.{alt}") == "format"


def test_audience_sender_and_time():
    t = tok()
    assert code(t, audience="PeerC") == "audience"
    assert code(t, sender="PeerC") == "sender"              # a stolen token is useless to the thief
    assert code(t, now=4_600) == "expired"
    assert ok(t, now=4_599)["rte"] == 3
    assert code(t, now=969) == "not_yet_valid"
    assert ok(t, now=970)                                   # 30 s skew
    with pytest.raises(L.LeaseError) as e:
        L.issue(KEY, {**CLAIMS, "exp": CLAIMS["iat"] + L.MAX_TTL_S + 1})
    assert e.value.code == "ttl"


def test_gate_accepts_redials_and_refuses_stale_or_closed_sessions():
    g = L.LeaseGate(PUB, "PeerB")
    t3, t4 = tok(), tok(rte=4, lid="00112233445566fe")
    assert g.admit(t3, sender="PeerA", now=2_000)["rte"] == 3
    assert g.admit(t3, sender="PeerA", now=2_001)          # a reset stream re-dials with its token
    assert g.admit(t4, sender="PeerA", now=2_002)["rte"] == 4
    with pytest.raises(L.LeaseError) as e:
        g.admit(t3, sender="PeerA", now=2_003)
    assert e.value.code == "stale_rte"
    g.supersede(CLAIMS["sid"], 5, now=2_004)
    with pytest.raises(L.LeaseError) as e:
        g.admit(t4, sender="PeerA", now=2_005)
    assert e.value.code == "stale_rte"
    t5 = tok(rte=5, lid="00112233445566fd")
    assert g.admit(t5, sender="PeerA", now=2_006)
    g.close(CLAIMS["sid"], now=2_007)
    with pytest.raises(L.LeaseError) as e:
        g.admit(t5, sender="PeerA", now=2_008)
    assert e.value.code == "closed"


def test_gate_prunes_only_when_nothing_it_guards_can_verify():
    g = L.LeaseGate(PUB, "PeerB")
    long_old = tok(rte=1, lid="00112233445566aa", exp=CLAIMS["iat"] + L.MAX_TTL_S)
    short_new = tok(rte=2, lid="00112233445566bb", iat=1_100, exp=1_200)
    g.admit(short_new, sender="PeerA", now=1_150)
    g.prune(now=1_300)                                      # short_new expired, long_old has not
    with pytest.raises(L.LeaseError) as e:
        g.admit(long_old, sender="PeerA", now=1_300)
    assert e.value.code == "stale_rte"
    g.prune(now=1_100 + L.MAX_TTL_S + 31)
    assert g._sids == {}
