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
    (lambda d, p: d.update({"id": p["hops"][2]["id"], "model": M, "sessions": 0, "rho": 0.86,
                            "kv_free": 10 ** 6}, 50), "overloaded"),
    (lambda d, p: d.update({"id": p["hops"][2]["id"], "model": M, "sessions": 16, "rho": 0.1,
                            "kv_free": 10 ** 6}, 50), "overloaded"),
    (lambda d, p: d.update({"id": p["hops"][1]["backup"], "model": M, "sessions": 0, "rho": 0.0,
                            "kv_free": 100}, 50), "kv_full"),
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


def test_subnet_collision_duplicate_hop_and_single_use_sids():
    d = fleet([("A", "pro6000-96g"), ("B", "pro6000-96g")])
    p = d.paths("eu-1", M)[0]
    le = lease(d, p)
    for job in ("job-1", "job-2"):
        with pytest.raises(LeaseRefused) as e:
            lease(d, p, job=job, rte=1)
        assert e.value.code == "sid_in_use"                  # a live sid, in any job
    d.release(le["lease"], 200)
    with pytest.raises(LeaseRefused) as e:
        lease(d, p, now=300)
    assert e.value.code == "sid_in_use"                      # a released sid stays retired
    assert lease(d, p, now=200 + L.MAX_TTL_S)["lease"]      # ... until no gate can remember it
    d2 = fleet([("A", "pro6000-96g", ), ("B", "pro6000-96g")])
    h = d2.cells[("eu-1", M)].holdings
    h["B"].node = h["A"].node.__class__(**{**h["B"].node.__dict__, "subnet": "s-A"})
    with pytest.raises(LeaseRefused) as e:
        lease(d2, p)
    assert e.value.code == "subnet_collision"
    hops = [dict(p["hops"][0], hi=10), dict(p["hops"][0], lo=10, hi=20), dict(p["hops"][1], lo=20)]
    with pytest.raises(LeaseRefused) as e:
        lease(d, {"hops": hops}, sid="1111111111111111")
    assert e.value.code == "duplicate_hop"


def gate_admits(d, t, now=100):
    """What the hop does with a token its lease_open carried: verify, bind to the message, admit."""
    g = L.LeaseGate(d.pubkey, t["to"])
    c = L.verify(t["token"], d.pubkey, audience=t["to"], now=now)
    bind = {k: c[k] for k in L.BIND_KEYS}
    assert bind["from"] == t["from"] and bind["role"] == t["role"]
    g.admit(t["token"], now=now, bind=bind)
    return g, c


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
        g, c = gate_admits(d, t)
        assert (c["sid"], c["rte"], c["kv"], c["exp"]) == (SID, 0, 4096, 100 + d.lease_ttl_s)
        kind = "mirror" if t["role"] == "backup" else "frame"
        assert g.accepts(SID, 0, t["from"], kind, 100) and not g.accepts(SID, 0, "Mallory", kind, 100)
        assert not g.accepts(SID, 0, t["from"], kind, 100 + d.lease_ttl_s)
        with pytest.raises(L.LeaseError):
            L.LeaseGate(d.pubkey, "Mallory").admit(t["token"], now=100, bind={k: c[k] for k in L.BIND_KEYS})
    for t in (x for x in toks if x["role"] == "backup"):
        hop = next(h for h in hops if h["backup"] == t["to"])
        assert t["from"] == hop["id"]                        # the mirror stream: primary -> backup


def test_record_pushes_absorb_a_bump_once_the_node_lists_the_session():
    d = twelve()
    p = d.paths("eu-1", M)[0]
    lease(d, p, now=100)
    h = p["hops"][0]["id"]
    view = lambda: next(v for v in d.snapshot("eu-1", M) if v["id"] == h)   # noqa: E731
    d.update({"id": h, "model": M, "sessions": 2, "rho": 0.1, "kv_free": 50_000, "sids": []}, 90)
    assert (view()["sessions"], view()["kv_free"]) == (3, 50_000 - 4096)     # the bump stays on top
    d.update({"id": h, "model": M, "sessions": 3, "rho": 0.3, "kv_free": 45_904, "sids": [SID]}, 101)
    assert (view()["sessions"], view()["rho"], view()["kv_free"]) == (3, 0.3, 45_904)   # absorbed
    d.update({"id": h, "model": M, "sessions": 2, "rho": 0.2, "kv_free": 50_000, "sids": []}, 102)
    assert view()["sessions"] == 2                            # the session closed: nothing comes back


def test_node_clocks_do_not_matter():
    """Records carry no node timestamp: a node stamping ms (or 30 s slow) cannot freeze its
    record, keep itself suspect, or lose/duplicate bumps."""
    d = twelve()
    p = d.paths("eu-1", M)[0]
    h = p["hops"][0]["id"]
    d.report_failure(h, M, 50)
    d.update({"id": h, "model": M, "sessions": 0, "rho": 0.0, "kv_free": 10 ** 6, "ts": 1.7e12}, 51)
    d.update({"id": h, "model": M, "sessions": 1, "rho": 0.1, "kv_free": 10 ** 6, "ts": 0}, 52)
    v = next(v for v in d.snapshot("eu-1", M) if v["id"] == h)
    assert (v["sessions"], v["suspect"]) == (1, False)


def test_release_and_expiry_undo_the_bump():
    d = twelve()
    p = d.paths("eu-1", M)[0]
    snap = d.snapshot("eu-1", M)
    le = lease(d, p)
    d.release(le["lease"], 150)
    assert d.snapshot("eu-1", M) == snap
    lease(d, p, sid="2222222222222222", now=200)
    q = d.paths("eu-1", M)[0]
    lease(d, q, sid="fedcba9876543210", now=200 + d.lease_ttl_s)   # the previous lease has expired
    ids = {h["id"] for h in p["hops"]} - {h["id"] for h in q["hops"]}
    assert ids and all(v["sessions"] == 0 for v in d.snapshot("eu-1", M) if v["id"] in ids)


def test_reroute_replaces_a_dead_hop_without_double_counting_survivors():
    d = twelve()
    p = d.paths("eu-1", M)[0]
    le = lease(d, p)
    dead = p["hops"][2]
    d.report_failure(dead["id"], M, 150)
    hops = [dict(h) for h in p["hops"]]
    hops[2] = {"id": dead["backup"], "lo": dead["lo"], "hi": dead["hi"], "backup": None}   # tier-1 takeover
    r = d.reroute(le["lease"], rte=1, hops=hops, now=160)
    changed = {(t["role"], t["from"], t["to"]) for t in r["tokens"]}
    assert changed == {("mid", hops[0]["id"], hops[1]["id"]),     # its bound `next` changed
                       ("mid", hops[1]["id"], hops[2]["id"]), ("mid", hops[2]["id"], hops[3]["id"])}
    assert r["exp"] == le["exp"]                              # new edges end with the session
    view = {v["id"]: v for v in d.snapshot("eu-1", M)}
    assert view[dead["id"]]["sessions"] == 0 and view[hops[2]["id"]]["sessions"] == 1
    assert all(view[h["id"]]["sessions"] == 1 for h in hops)  # survivors still count the session once
    for t in r["tokens"]:
        g, c = gate_admits(d, t, now=160)
        assert c["rte"] == 1
    with pytest.raises(LeaseRefused) as e:
        d.reroute(le["lease"], rte=1, hops=hops, now=170)
    assert e.value.code == "stale_rte"


def test_reroute_refusals_change_nothing_and_renew_reissues_every_edge():
    d = twelve()
    p = d.paths("eu-1", M)[0]
    le = lease(d, p)
    snap = d.snapshot("eu-1", M)
    bad = [dict(h) for h in p["hops"]]
    bad[1]["lo"] = float(bad[1]["lo"])                        # 9.0 is not an int anywhere
    with pytest.raises(LeaseRefused) as e:
        d.reroute(le["lease"], rte=1, hops=bad, now=150)
    assert e.value.code == "bad_path" and d.snapshot("eu-1", M) == snap
    r = d.reroute(le["lease"], rte=1, hops=p["hops"], now=3000)
    assert r["tokens"] == [] and r["exp"] == le["exp"]        # same path: nothing to issue
    r = d.reroute(le["lease"], rte=2, hops=p["hops"], now=3500, renew=True)
    assert len(r["tokens"]) == len(le["tokens"]) and r["exp"] == 3500 + d.lease_ttl_s
    view = {v["id"]: v for v in d.snapshot("eu-1", M)}
    assert all(view[h["id"]]["sessions"] == 1 for h in p["hops"])   # renewal is not a new session
    lease(d, d.paths("eu-1", M)[0], sid="3333333333333333", now=3600 + 60)   # expiry sweep runs
    assert d._leases[le["lease"]]["exp"] == 3500 + d.lease_ttl_s           # the renewed lease lives


def test_a_backup_is_charged_once_per_session_not_once_per_hop():
    d = fleet([("Mac", "mac-192g")] + [(f"F{i}", "5090") for i in range(6)])
    d.update({"id": "Mac", "model": M, "sessions": 16, "rho": 0.2, "kv_free": 200_000, "sids": []}, 1)
    (p, *_) = d.paths("eu-1", M, kv_tokens=40_000)
    assert {h["backup"] for h in p["hops"]} == {"Mac"} and len(p["hops"]) == 6
    lease(d, p, kv_tokens=40_000)
    assert next(v for v in d.snapshot("eu-1", M) if v["id"] == "Mac")["kv_free"] == 160_000


def test_a_device_holding_a_span_serves_one_model():
    from shard.mesh.placement import MIMO_V26_PRO
    d = fleet([("A", "pro6000-96g")])
    with pytest.raises(ValueError):
        d.join(node("A", "pro6000-96g"), MIMO_V26_PRO.model, 1)


def test_report_failure_only_steers_routing():
    d = twelve()
    p = d.paths("eu-1", M)[0]
    bad = p["hops"][2]["id"]
    span = d.cells[("eu-1", M)].holdings[bad].span
    d.report_failure(bad, M, 50)
    assert all(bad not in [h["id"] for h in q["hops"]] for q in d.paths("eu-1", M, k=10))
    assert d.cells[("eu-1", M)].holdings[bad].span == span    # placement untouched
    d.update({"id": bad, "model": M, "sessions": 0, "rho": 0.0, "kv_free": 10 ** 6}, 51)
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
        gate_admits(d, t)
