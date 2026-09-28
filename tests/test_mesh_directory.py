"""shard.mesh.directory: registry, Parallax-style bump, atomic lease admission, signed tokens."""
import copy

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from mesh_fleet import node
from shard.mesh import lease as L
from shard.mesh.directory import Directory, LeaseRefused
from shard.mesh.placement import MIMO_V26_FLASH as FLASH

M = FLASH.model
KEY = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
SID = "0123456789abcdef"


def fleet(kinds, **kw):
    d = Directory(KEY, entry="Entry1", **kw)
    for nid, k in kinds:
        d.join(node(nid, k), M, 0)
    return d


def twelve():
    return fleet([(f"F{i}", "5090") for i in range(12)])      # every span held twice


def lease(d, p, **kw):
    args = dict(job="job-1", sid=SID, rte=0, cell="eu-1", model=M, hops=p["hops"], kv_tokens=4096,
                load=0.2, now=100)
    args.update(kw)
    return d.lease(**args)


def test_lease_bumps_every_hop_and_backup_and_the_next_query_sees_it():
    d = twelve()
    p = d.paths("eu-1", M)[0]
    before = {v["id"]: v for v in d.snapshot("eu-1", M)}
    lease(d, p)
    after = {v["id"]: v for v in d.snapshot("eu-1", M)}
    for h in p["hops"]:
        assert after[h["id"]]["sessions"] == 1 and after[h["id"]]["rho"] == pytest.approx(0.2)
        assert after[h["id"]]["kv_free"] == before[h["id"]]["kv_free"] - 4096
        b = after[h["backup"]]
        assert (b["sessions"], b["rho"], b["kv_free"]) == (0, 0.0, before[h["backup"]]["kv_free"] - 4096)
    q = d.paths("eu-1", M)[0]
    assert {h["id"] for h in q["hops"]}.isdisjoint({h["id"] for h in p["hops"]})   # the bump spread it


def test_backup_on_the_path_is_refused_and_nothing_changes():
    d = twelve()
    p = copy.deepcopy(d.paths("eu-1", M)[0])
    p["hops"][0]["backup"] = p["hops"][1]["id"]
    snap = d.snapshot("eu-1", M)
    with pytest.raises(LeaseRefused) as e:
        lease(d, p)
    assert e.value.code == "backup_on_path" and d.snapshot("eu-1", M) == snap


@pytest.mark.parametrize("breakit,want", [
    (lambda d, p: d.update({"id": p["hops"][2]["id"], "model": M, "ts": 50, "sessions": 0, "rho": 0.86,
                            "kv_free": 10 ** 6}), "overloaded"),
    (lambda d, p: d.update({"id": p["hops"][2]["id"], "model": M, "ts": 50, "sessions": 16, "rho": 0.1,
                            "kv_free": 10 ** 6}), "overloaded"),
    (lambda d, p: d.update({"id": p["hops"][1]["backup"], "model": M, "ts": 50, "sessions": 0, "rho": 0.0,
                            "kv_free": 100}), "kv_full"),
    (lambda d, p: p["hops"][1].update(hi=p["hops"][1]["hi"] + 1), "bad_path"),
    (lambda d, p: p["hops"].pop(), "bad_path"),
    (lambda d, p: p["hops"][1].update(lo=p["hops"][1]["lo"] - 1) or p["hops"][0].update(hi=p["hops"][0]["hi"] - 1),
     "span_mismatch"),
    (lambda d, p: p["hops"][2].update(id="Ghost"), "unknown_holder"),
    (lambda d, p: p["hops"][2].update(backup=p["hops"][0]["backup"]), "backup_span"),   # off-path, wrong span
    (lambda d, p: d.report_failure(p["hops"][3]["id"], M, 60), "suspect"),
    (lambda d, p: d.report_failure(p["hops"][3]["backup"], M, 60), "suspect"),
])
def test_every_refusal_is_atomic(breakit, want):
    d = twelve()
    p = copy.deepcopy(d.paths("eu-1", M)[0])
    breakit(d, p)
    snap = d.snapshot("eu-1", M)
    with pytest.raises(LeaseRefused) as e:
        lease(d, p)
    assert e.value.code == want and d.snapshot("eu-1", M) == snap


def test_subnet_collision_duplicate_hop_and_duplicate_lease():
    d = fleet([("A", "pro6000-96g"), ("B", "pro6000-96g")])
    p = d.paths("eu-1", M)[0]
    lease(d, p)
    with pytest.raises(LeaseRefused) as e:
        lease(d, p)
    assert e.value.code == "duplicate_lease"                 # a re-lease must bump rte
    assert lease(d, p, rte=1)["lease"]
    d2 = fleet([("A", "pro6000-96g", ), ("B", "pro6000-96g")])
    d2.cells[("eu-1", M)].holdings["B"].node = d2.cells[("eu-1", M)].holdings["A"].node.__class__(
        **{**d2.cells[("eu-1", M)].holdings["B"].node.__dict__, "subnet": "s-A"})
    with pytest.raises(LeaseRefused) as e:
        lease(d2, p)
    assert e.value.code == "subnet_collision"
    hops = [dict(p["hops"][0], hi=10), dict(p["hops"][0], lo=10, hi=20), dict(p["hops"][1], lo=20)]
    with pytest.raises(LeaseRefused) as e:
        lease(d, {"hops": hops}, rte=2)
    assert e.value.code == "duplicate_hop"


def test_every_token_admits_exactly_its_edge_at_its_node():
    d = twelve()
    p = d.paths("eu-1", M)[0]
    le = lease(d, p)
    hops, toks = p["hops"], le["tokens"]
    assert len(toks) == 2 * len(hops)                        # one per hop, one per backup
    prim = [t for t in toks if t["role"] != "backup"]
    assert [t["from"] for t in prim] == ["Entry1"] + [h["id"] for h in hops[:-1]]
    assert [t["role"] for t in prim] == ["head"] + ["mid"] * (len(hops) - 2) + ["tail"]
    for t in toks:
        gate = L.LeaseGate(d.pubkey, t["to"])
        c = gate.admit(t["token"], sender=t["from"], now=100)
        assert (c["sid"], c["rte"], c["kv"], c["exp"]) == (SID, 0, 4096, 100 + d.lease_ttl_s)
        with pytest.raises(L.LeaseError):
            L.LeaseGate(d.pubkey, t["to"]).admit(t["token"], sender="Mallory", now=100)
        with pytest.raises(L.LeaseError):
            L.LeaseGate(d.pubkey, "Mallory").admit(t["token"], sender=t["from"], now=100)
    for t in (x for x in toks if x["role"] == "backup"):
        hop = next(h for h in hops if h["backup"] == t["to"])
        assert t["from"] == hop["id"]                        # the mirror stream: primary -> backup


def test_record_pushes_absorb_bumps_by_timestamp():
    d = twelve()
    p = d.paths("eu-1", M)[0]
    lease(d, p, now=100)
    h = p["hops"][0]["id"]
    view = lambda: next(v for v in d.snapshot("eu-1", M) if v["id"] == h)   # noqa: E731
    d.update({"id": h, "model": M, "ts": 90, "sessions": 2, "rho": 0.1, "kv_free": 50_000})
    assert (view()["sessions"], view()["kv_free"]) == (3, 50_000 - 4096)     # the bump stays on top
    d.update({"id": h, "model": M, "ts": 100, "sessions": 3, "rho": 0.3, "kv_free": 45_904})
    assert (view()["sessions"], view()["rho"], view()["kv_free"]) == (3, 0.3, 45_904)   # absorbed
    d.update({"id": h, "model": M, "ts": 95, "sessions": 0, "rho": 0.0, "kv_free": 1})
    assert view()["sessions"] == 3                            # an out-of-order push is ignored


def test_release_and_expiry_undo_the_bump():
    d = twelve()
    p = d.paths("eu-1", M)[0]
    snap = d.snapshot("eu-1", M)
    le = lease(d, p)
    d.release(le["lease"])
    assert d.snapshot("eu-1", M) == snap
    lease(d, p, rte=1, now=100)
    q = d.paths("eu-1", M)[0]
    lease(d, q, sid="fedcba9876543210", now=100 + d.lease_ttl_s)   # the first lease has expired
    ids = {h["id"] for h in p["hops"]} - {h["id"] for h in q["hops"]}
    assert all(v["sessions"] == 0 for v in d.snapshot("eu-1", M) if v["id"] in ids)


def test_report_failure_only_steers_routing():
    d = twelve()
    p = d.paths("eu-1", M)[0]
    bad = p["hops"][2]["id"]
    span = d.cells[("eu-1", M)].holdings[bad].span
    d.report_failure(bad, M, 50)
    assert all(bad not in [h["id"] for h in q["hops"]] for q in d.paths("eu-1", M, k=10))
    assert d.cells[("eu-1", M)].holdings[bad].span == span    # placement untouched
    d.update({"id": bad, "model": M, "ts": 51, "sessions": 0, "rho": 0.0, "kv_free": 10 ** 6})
    assert any(bad in [h["id"] for h in q["hops"]] for q in d.paths("eu-1", M, k=10))


def test_sticky_rejoin_and_repair_through_the_directory():
    d = fleet([(f"F{i}", "5090") for i in range(6)])          # one of each span
    span = d.cells[("eu-1", M)].holdings["F3"].span
    d.leave("F3", M, 10)
    back = node("F3", "5090", disk={M: [list(span)]})
    r = d.join(back, M, 20)
    assert (r["reason"], tuple(r["span"]), r["download_layers"]) == ("sticky", span, 0)
    d.leave("F3", M, 30)
    first = d.join(node("F6", "5090"), M, 40)
    assert tuple(first["span"]) == span                       # an arrival closes the hole first
    for i in range(7, 12):
        d.join(node(f"F{i}", "5090"), M, 40)
    cell = d.cells[("eu-1", M)]
    dead = cell.holdings["F9"].span
    d.leave("F9", M, 50)                                      # one of the second copies dies
    under = [i for i, v in enumerate(cell.counts()) if v < 2]
    assert under and all(dead[0] <= i < dead[1] or i >= 41 for i in under)
    # nobody is held 3 times, so any move would push another span below k_min: no repair
    assert d.repair("eu-1", M, 50 + d.policy.kmin_repair_s) == []
    r = d.join(node("F12", "5090"), M, 60)                    # the next arrival raises k instead
    assert set(range(*r["span"])) & set(under)


def test_mixed_fleet_end_to_end():
    d = fleet([("Spark", "spark-128g"), ("Pro", "pro6000-96g"), ("G1", "3090"), ("G2", "4090"),
               ("S8", "4060-8g"), ("Mac", "mac-192g")])
    roles = {v["id"]: v["role"] for v in d.snapshot("eu-1", M)}
    assert roles == {"Spark": "stage", "Pro": "stage", "G1": "stage", "G2": "stage", "Mac": "stage"}
    assert d.cells[("eu-1", M)].others["S8"].role == "verifier"   # no catalog in the default policy
    p = d.paths("eu-1", M)[0]
    le = lease(d, p)
    for t in le["tokens"]:
        assert L.LeaseGate(d.pubkey, t["to"]).admit(t["token"], sender=t["from"], now=100)
