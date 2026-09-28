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
    args = {"audience": "PeerB", "now": 2_000, **kw}
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


def lease_open(token, **over):
    msg = {"v": 2, "op": "c", "cmd": "lease_open", "sid": CLAIMS["sid"], "rte": 3, "lo": 8, "hi": 16,
           "role": "mid", "pred": "PeerA", "next": "PeerC", "ret": "Entry", "max_pos": 32768, "W": 8,
           "cx": "bf16", "nonce": "00" * 8, "lease": token.encode()}
    msg.update(over)
    return frame.decode(b"".join(bytes(p) for p in frame.encode(msg)))


def test_rides_lease_open_and_binds_to_its_fields():
    back = lease_open(tok())
    g = L.LeaseGate(PUB, "PeerB")
    c = g.admit(bytes(back["lease"]), now=2_000, bind=L.binding_of(back))
    assert c["role"] == "mid" and g.accepts(CLAIMS["sid"], 3, "PeerA")
    assert not g.accepts(CLAIMS["sid"], 3, "PeerX") and not g.accepts(CLAIMS["sid"], 2, "PeerA")
    assert not g.accepts(CLAIMS["sid"], 3, "PeerA", kind="mirror")


@pytest.mark.parametrize("over", [{"role": "head", "pred": "Entry", "salt": "00" * 32}, {"lo": 9}, {"hi": 15},
                                  {"rte": 4},
                                  {"pred": "PeerX"}, {"sid": "f" * 16}])
def test_a_token_is_refused_in_a_message_that_claims_something_else(over):
    back = lease_open(tok(), **over)
    with pytest.raises(L.LeaseError) as e:
        L.LeaseGate(PUB, "PeerB").admit(bytes(back["lease"]), now=2_000, bind=L.binding_of(back))
    assert e.value.code == "binding"


def test_a_backup_token_is_only_a_mirror_edge():
    t = tok(role="backup", lid="00112233445566fc")
    g = L.LeaseGate(PUB, "PeerB")
    with pytest.raises(L.LeaseError) as e:                  # not as a primary lease_open
        g.admit(t, now=2_000, bind=L.binding_of(lease_open(t)))
    assert e.value.code == "binding"
    mo = {"cmd": "mirror_open", "sid": CLAIMS["sid"], "primary": "PeerA", "lo": 8, "hi": 16, "max_pos": 32768}
    g.admit(t, now=2_000, bind=L.binding_of(mo))
    assert g.accepts(CLAIMS["sid"], 3, "PeerA", kind="mirror") and not g.accepts(CLAIMS["sid"], 3, "PeerA")


def test_admit_demands_a_binding():
    with pytest.raises(ValueError):
        L.LeaseGate(PUB, "PeerB").admit(tok(), now=2_000, bind={"sid": CLAIMS["sid"]})


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


def test_audience_and_time():
    t = tok()
    assert code(t, audience="PeerC") == "audience"
    assert code(t, now=4_600) == "expired"
    assert ok(t, now=4_599)["rte"] == 3
    assert code(t, now=969) == "not_yet_valid"
    assert ok(t, now=970)                                   # 30 s skew
    with pytest.raises(L.LeaseError) as e:
        L.issue(KEY, {**CLAIMS, "exp": CLAIMS["iat"] + L.MAX_TTL_S + 1})
    assert e.value.code == "ttl"


BIND = {k: CLAIMS[k] for k in L.BIND_KEYS}


def admit(g, token, now):
    return g.admit(token, now=now, bind=BIND)


def test_gate_redelivery_is_idempotent_stale_and_closed_sessions_are_refused():
    g = L.LeaseGate(PUB, "PeerB")
    t3, t4 = tok(), tok(rte=4, lid="00112233445566fe")
    assert admit(g, t3, 2_000)["rte"] == 3
    assert admit(g, t3, 2_001)                             # a retried lease_open
    assert admit(g, t4, 2_002)["rte"] == 4 and not g.accepts(CLAIMS["sid"], 3, "PeerA")
    with pytest.raises(L.LeaseError) as e:
        admit(g, t3, 2_003)
    assert e.value.code == "stale_rte"
    g.supersede(CLAIMS["sid"], 5, now=2_004, pred="PeerZ")  # entry-signed lease_update: new predecessor
    assert g.accepts(CLAIMS["sid"], 5, "PeerZ") and not g.accepts(CLAIMS["sid"], 5, "PeerA")
    with pytest.raises(L.LeaseError) as e:
        admit(g, t4, 2_005)
    assert e.value.code == "stale_rte"
    g.supersede(CLAIMS["sid"], 6, now=2_006)                # no pred named: the edge carries over
    assert g.accepts(CLAIMS["sid"], 6, "PeerZ")
    g.close(CLAIMS["sid"], now=2_007)
    assert not g.accepts(CLAIMS["sid"], 6, "PeerZ")
    with pytest.raises(L.LeaseError) as e:
        admit(g, tok(rte=7, lid="00112233445566fd"), 2_008)
    assert e.value.code == "closed"


def test_gate_prunes_only_when_nothing_it_guards_can_verify():
    g = L.LeaseGate(PUB, "PeerB")
    long_old = tok(rte=1, lid="00112233445566aa", exp=CLAIMS["iat"] + L.MAX_TTL_S)
    short_new = tok(rte=2, lid="00112233445566bb", iat=1_100, exp=1_200)
    admit(g, short_new, 1_150)
    g.prune(now=1_300)                                      # short_new expired, long_old has not
    with pytest.raises(L.LeaseError) as e:
        admit(g, long_old, 1_300)
    assert e.value.code == "stale_rte"
    g.prune(now=1_100 + L.MAX_TTL_S + 31)
    assert g._sids == {}
