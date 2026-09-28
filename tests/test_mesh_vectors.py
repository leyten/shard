"""Replay tests/vectors/mesh_*.json against the Python implementation, as the TypeScript side will.

Profiles come from the vector files (not the Python constants), every step runs on a fresh
instance, and outputs compare exactly except floats (1e-9 relative). A second test regenerates
the vectors and fails on drift: an intentional change must re-run tests/mesh_vectors_gen.py.
"""
import json
import math
import os

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

import mesh_vectors_gen as G
from shard.mesh import lease as L
from shard.mesh.placement import ModelProfile

FILES = ("mesh_placement.json", "mesh_paths.json", "mesh_lease.json", "mesh_directory.json")


def load(name):
    with open(os.path.join(G.VEC_DIR, name)) as f:
        return json.load(f)


def same(got, want, path="$"):
    if isinstance(want, float) or (isinstance(got, float) and not isinstance(want, bool)):
        assert isinstance(got, (int, float)) and not isinstance(got, bool), path
        assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12), f"{path}: {got} != {want}"
    elif isinstance(want, dict):
        assert isinstance(got, dict) and set(got) == set(want), f"{path}: keys {sorted(got)} != {sorted(want)}"
        for k in want:
            same(got[k], want[k], f"{path}.{k}")
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), f"{path}: {got} != {want}"
        for i, (g, w) in enumerate(zip(got, want)):
            same(g, w, f"{path}[{i}]")
    else:
        assert got == want and type(got) is type(want), f"{path}: {got!r} != {want!r}"


def profiles(doc):
    return {k: ModelProfile.from_dict(v) for k, v in doc["profiles"].items()}


def args_of(step):
    return {k: v for k, v in step.items() if k not in ("op", "expect")}


def jsonable(x):
    return json.loads(json.dumps(x))


def ids(name):
    return [c["name"] for c in load(name)["cases"]]


@pytest.mark.parametrize("name", ids("mesh_placement.json"))
def test_placement_vectors(name):
    doc = load("mesh_placement.json")
    case = next(c for c in doc["cases"] if c["name"] == name)
    state = G.new_placement(case, profiles(doc))
    for i, st in enumerate(case["steps"]):
        same(jsonable(G.exec_placement(state, st["op"], args_of(st))), st["expect"], f"{name}#{i}:{st['op']}")


@pytest.mark.parametrize("name", ids("mesh_paths.json"))
def test_paths_vectors(name):
    doc = load("mesh_paths.json")
    case = next(c for c in doc["cases"] if c["name"] == name)
    same(jsonable(G.exec_paths(case, profiles(doc))), case["expect"], name)


@pytest.mark.parametrize("name", ids("mesh_directory.json"))
def test_directory_vectors(name):
    doc = load("mesh_directory.json")
    case = next(c for c in doc["cases"] if c["name"] == name)
    d = G.new_directory(case, profiles(doc))
    for i, st in enumerate(case["steps"]):
        same(jsonable(G.exec_directory(d, st["op"], args_of(st))), st["expect"], f"{name}#{i}:{st['op']}")


def test_lease_vectors():
    doc = load("mesh_lease.json")
    key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(doc["key"]["seed_hex"]))
    assert G.pub_b64(key) == doc["key"]["pub_b64"]
    assert (doc["domain"], doc["max_token"], doc["max_ttl_s"], doc["claims"]) == (
        L.DOMAIN.decode(), L.MAX_TOKEN, L.MAX_TTL_S, list(L.CLAIMS))
    for v in doc["issue"]:
        assert L.issue(key, v["claims"]) == v["token"]          # Ed25519 is deterministic
    for v in doc["verify"]:
        same(jsonable(G.exec_verify(v)), v["expect"], v["name"])
    for g in doc["gate"]:
        state = {"gate": L.LeaseGate(g["pub"], g["me"], skew=g["skew"])}
        for i, st in enumerate(g["steps"]):
            same(jsonable(G.exec_gate(state, st["op"], args_of(st))), st["expect"], f"{g['name']}#{i}")


def test_vectors_cover_the_m3_gate():
    names = " | ".join(ids("mesh_placement.json") + ids("mesh_paths.json") + ids("mesh_directory.json")).lower()
    for topic in ("join priority: holes", "k < k_min before", "disk-warm", "sticky", "repair: hole",
                  "repair: k < k_min", "bump", "backup on the path", "spark + 5090 + 3090", "only 24 gb",
                  "every device class", "role table", "reroute", "charged kv once", "hands over"):
        assert topic in names, topic
    codes = {v["expect"].get("error") for v in load("mesh_lease.json")["verify"]}
    assert codes >= {None, "signature", "schema", "issuer", "audience", "expired", "not_yet_valid", "format", "ttl"}
    gate_codes = {st["expect"].get("error") for g in load("mesh_lease.json")["gate"] for st in g["steps"]
                  if isinstance(st["expect"], dict)}
    assert gate_codes >= {None, "binding", "stale_rte", "closed"}


def test_vectors_are_current():
    fresh = G.build()
    for name in FILES:
        with open(os.path.join(G.VEC_DIR, name)) as f:
            assert f.read() == G.dump(fresh[name]), f"{name} drifted: run python3 tests/mesh_vectors_gen.py"
