"""Shared mesh test vectors: tests/vectors/mesh_{placement,paths,lease,directory}.json.

The c0mpute orchestrator implements the directory in TypeScript; these files are the contract
both implementations must pass. Each case is a list of steps {"op", ...args, "expect"}. A runner
executes the op with the args against a fresh instance and compares with `expect`: exactly for
ints, strings, bools, null and structure; floats within 1e-9 relative (both sides use IEEE
doubles and the op order the Python docstrings pin, so in practice they are equal).

Regenerate after an intentional behaviour change (the test fails on drift):
    python3 tests/mesh_vectors_gen.py
The recorder below runs every op on the real implementation while the cases are built, so an
expectation is always what the code does; tests/test_mesh_*.py assert WHY it is right.
"""
import hashlib
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: E402

from mesh_churn import LAYER, SIM_PROFILE  # noqa: E402
from mesh_fleet import DEVICES, DISTILL_9B, node  # noqa: E402
from shard.manifest import pub_b64  # noqa: E402
from shard.mesh import lease as L  # noqa: E402
from shard.mesh.directory import Directory, LeaseRefused  # noqa: E402
from shard.mesh.placement import (MIMO_V26_FLASH, MIMO_V26_PRO, Cell, Fit, ModelProfile, Node, Policy,  # noqa: E402
                                  candidate_paths, classify)

SCHEMA = "shard-mesh-vectors/1"
VEC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors")
PROFILES = {p.model: p for p in (MIMO_V26_FLASH, MIMO_V26_PRO, SIM_PROFILE)}
FLASH, SIM = MIMO_V26_FLASH.model, SIM_PROFILE.model
# A PUBLIC test key (not a secret): the orchestrator key every lease vector is signed with.
KEY_SEED = hashlib.sha256(b"shard-mesh-vectors/1 public test orchestrator key").digest()
KEY = ed25519.Ed25519PrivateKey.from_private_bytes(KEY_SEED)
CATALOG = {"small_models": [DISTILL_9B]}
SIM_POLICY = {"k_min": 2, "leg_ms": 0.0, "plan_sessions": 0, "unified_host_bytes": 0, "hole_repair_s": 600.0,
              "kmin_repair_s": 1800.0, "move_interval_s": 600.0}


def pol_dict(**kw) -> dict:
    return Policy.from_dict(kw).to_dict()


def sim_node(nid: str, layers: int, **kw) -> dict:
    d = {"id": nid, "cell": "sim", "subnet": f"s-{nid}", "reliability": 1.0, "uplink_mbps": 200.0,
         "device": {"backend": "cuda", "device_name": "sim", "arch_class": "sim", "host_arch": "x86_64",
                    "n_devices": 1, "total_mem_bytes": layers * LAYER, "free_mem_bytes": layers * LAYER,
                    "unified_memory": False, "mem_bw_bytes_s": 1e12, "mem_bw_source": "spec",
                    "formats_hw": ["bf16"], "formats_emulated": []}}
    d.update(kw)
    return d


# ── executors (the runner semantics the TypeScript port reproduces) ────────────────────────────

def exec_placement(state: dict, op: str, a: dict):
    prof, pol = state["profile"], state["policy"]
    cell = state["cell"]
    if op == "classify":
        return classify(Node.from_dict(a["node"]), prof, pol).to_dict()
    if op == "spans":
        return [list(s) for s in Fit(Node.from_dict(a["node"]), prof, pol).spans(a.get("min_len", 1))]
    if op == "join":
        return cell.join(Node.from_dict(a["node"]), a["now"]).to_dict()
    if op == "leave":
        return cell.leave(a["id"], a["now"])
    if op == "repair":
        return [m.to_dict() for m in cell.repair(a["now"])]
    if op == "complete_move":
        return cell.complete_move(a["id"], a["now"])
    if op == "counts":
        return cell.counts()
    raise ValueError(op)


def new_placement(case: dict, profiles: dict) -> dict:
    prof, pol = profiles[case["profile"]], Policy.from_dict(case["policy"])
    return {"profile": prof, "policy": pol, "cell": Cell(prof, pol)}


def exec_paths(case: dict, profiles: dict):
    return candidate_paths(case["holders"], profiles[case["profile"]], Policy.from_dict(case["policy"]),
                           **case["args"])


def exec_directory(d: Directory, op: str, a: dict):
    if op == "join":
        return d.join(a["node"], a["model"], a["now"])
    if op == "leave":
        return d.leave(a["id"], a["model"], a["now"])
    if op == "repair":
        return d.repair(a["cell"], a["model"], a["now"])
    if op == "complete_move":
        return d.complete_move(a["id"], a["model"], a["now"])
    if op == "update":
        return d.update(a["record"])
    if op == "report_failure":
        return d.report_failure(a["id"], a["model"], a["now"])
    if op == "snapshot":
        return d.snapshot(a["cell"], a["model"])
    if op == "paths":
        return d.paths(a["cell"], a["model"], **a.get("args", {}))
    if op == "lease":
        try:
            return {"ok": d.lease(**a["args"])}
        except LeaseRefused as e:
            return {"error": e.code}
    if op == "release":
        return d.release(a["lease"])
    raise ValueError(op)


def new_directory(case: dict, profiles: dict) -> Directory:
    key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(case["key_seed_hex"]))
    return Directory(key, profiles=profiles, policy=Policy.from_dict(case["policy"]), entry=case["entry"],
                     lease_ttl_s=case["lease_ttl_s"])


def exec_gate(state: dict, op: str, a: dict):
    g = state["gate"]
    if op == "admit":
        try:
            return {"ok": g.admit(a["token"], sender=a["sender"], now=a["now"])}
        except L.LeaseError as e:
            return {"error": e.code}
    if op == "supersede":
        return g.supersede(a["sid"], a["rte"], a["now"])
    if op == "close":
        return g.close(a["sid"], a["now"])
    if op == "prune":
        return g.prune(a["now"])
    raise ValueError(op)


def exec_verify(v: dict):
    try:
        return {"ok": L.verify(v["token"], v["pub"], audience=v["audience"], sender=v["sender"], now=v["now"],
                               skew=v["skew"])}
    except L.LeaseError as e:
        return {"error": e.code}


class Rec:
    """Runs each op on the live instance and records {op, args..., expect}."""

    def __init__(self, fn, state):
        self.fn, self.state, self.steps = fn, state, []

    def __call__(self, op, **a):
        out = self.fn(self.state, op, a)
        self.steps.append({"op": op, **a, "expect": _jsonable(out)})
        return out


def _jsonable(x):
    return json.loads(json.dumps(x))


# ── placement cases ───────────────────────────────────────────────────────────────────────────

def _pcase(name, why, profile, policy, build):
    case = {"name": name, "why": why, "profile": profile, "policy": policy}
    r = Rec(exec_placement, new_placement(case, PROFILES))
    build(r)
    case["steps"] = r.steps
    return case


def placement_cases():
    cases = []

    def roles(r):
        for kind in sorted(DEVICES):
            r("classify", node=node(f"X{kind.replace('-', '')}", kind))

    for leg in (6.0, 13.0, 19.0):
        cases.append(_pcase(f"role table, Flash, leg {leg} ms", "every device family gets a role from its "
                            "numbers; the stage bar moves with the cell's leg latency",
                            FLASH, pol_dict(leg_ms=leg, **CATALOG), roles))
    cases.append(_pcase("role table, Pro, leg 13 ms", "same devices, another profile: fit comes from the "
                        "profile, not the card", MIMO_V26_PRO.model, pol_dict(**CATALOG), roles))
    cases.append(_pcase("role table, no small-model catalog", "without a catalog small cards verify",
                        FLASH, pol_dict(), roles))

    def spans(r):
        for kind in ("4060-8g", "3090", "5090", "pro6000-96g", "spark-128g", "mac-64g", "mac-192g"):
            r("spans", node=node("S", kind))
        r("spans", node=node("S", "5090"), min_len=8)

    cases.append(_pcase("maximal spans per device", "head keeps the light layer 0 + embed, the tail pays "
                        "lm_head + MTP + embed copy; whole-model devices have exactly [0, 48)",
                        FLASH, pol_dict(), spans))

    def holes_first(r):
        for i, (lo, hi) in enumerate([(0, 24), (0, 24), (24, 40)]):
            r("join", node=sim_node(f"n{i}", hi - lo, span=[lo, hi]), now=0)
        r("counts")
        r("join", node=sim_node("n9", 8), now=0)
        r("counts")

    def under_first(r):
        for i, (lo, hi) in enumerate([(0, 24), (0, 24), (24, 48)]):
            r("join", node=sim_node(f"n{i}", 24, span=[lo, hi]), now=0)
        r("join", node=sim_node("n9", 24), now=0)
        r("counts")

    def disk_first(r):
        for i, (lo, hi) in enumerate([(0, 24), (0, 24), (24, 48), (24, 48)]):
            r("join", node=sim_node(f"n{i}", 24, span=[lo, hi]), now=0)
        r("join", node=sim_node("n8", 8), now=0)
        r("join", node=sim_node("n9", 8, disk={SIM: [[16, 24]]}), now=0)

    def stack(r):
        for i in range(16):
            r("join", node=sim_node(f"n{i}", 6), now=0)
        r("counts")

    cases += [
        _pcase("join priority: holes first", "a hole beats raising a k=1 range", SIM, SIM_POLICY, holes_first),
        _pcase("join priority: k < k_min before capacity", "no hole: the single-holder half wins", SIM,
               SIM_POLICY, under_first),
        _pcase("join priority: disk-warm among equals", "equal windows: the one on disk, zero download", SIM,
               SIM_POLICY, disk_first),
        _pcase("join priority: replicas stack on boundaries", "valley fill: the second copy of every span "
               "lands on the first copy's boundaries (whole-span backups exist)", SIM, SIM_POLICY, stack),
    ]

    def sticky(r):
        r("join", node=sim_node("n1", 8, span=[0, 8]), now=0)
        r("join", node=sim_node("n2", 40, span=[8, 48]), now=0)
        r("join", node=sim_node("n4", 8, span=[0, 8], disk={SIM: [[0, 8]]}), now=10)
        r("leave", id="n4", now=20)
        r("join", node=sim_node("n4", 8, disk={SIM: [[0, 8]]}), now=30)

    def sticky_hole(r):
        for nid, lo, hi in (("n1", 0, 8), ("n2", 0, 8), ("n3", 8, 40)):
            r("join", node=sim_node(nid, hi - lo, span=[lo, hi]), now=0)
        r("join", node=sim_node("n4", 8, span=[0, 8]), now=0)

    cases += [
        _pcase("sticky rejoin, zero download", "a returning node goes home even when k=1 layers exist "
               "elsewhere", SIM, SIM_POLICY, sticky),
        _pcase("rejoin closes a hole when its old span is redundant", "holes beat stickiness only if the "
               "old span keeps k_min without the node", SIM, SIM_POLICY, sticky_hole),
    ]

    def hole_repair(r):
        for i, (lo, hi) in enumerate([(0, 24), (0, 24), (24, 40), (24, 40), (40, 48)]):
            r("join", node=sim_node(f"n{i}", hi - lo, span=[lo, hi]), now=0)
        r("leave", id="n4", now=100)
        r("repair", now=699)
        mv = r("repair", now=700)
        r("counts")
        r("repair", now=701)
        r("complete_move", id=mv[0]["id"], now=800)
        r("counts")
        other = "n3" if mv[0]["id"] == "n2" else "n2"
        r("leave", id=other, now=850)                 # a second hole: its timer runs to 1450
        r("repair", now=1449)
        mv2 = r("repair", now=1450)
        r("complete_move", id=mv2[0]["id"], now=1500)
        r("counts")

    def hole_repair_disk(r):
        for i, (lo, hi) in enumerate([(0, 24), (0, 24), (24, 40), (24, 40), (40, 48)]):
            disk = {SIM: [[24, 48]]} if i == 1 else None
            r("join", node=sim_node(f"n{i}", hi - lo, span=[lo, hi], **({"disk": disk} if disk else {})), now=0)
        r("leave", id="n4", now=100)
        r("repair", now=700)

    # joins alternate halves: a third copy of a k=2 span would (rightly) take a hole instead
    halves = [(0, 24), (24, 48), (0, 24), (24, 48), (0, 24)]

    def kmin_repair(r):
        for i, (lo, hi) in enumerate(halves):
            r("join", node=sim_node(f"n{i}", 24, span=[lo, hi]), now=0)
        r("leave", id="n3", now=0)
        r("repair", now=1799)
        mv = r("repair", now=1800)
        r("complete_move", id=mv[0]["id"], now=1900)
        r("counts")

    def kmin_rejoin(r):
        for i, (lo, hi) in enumerate(halves):
            r("join", node=sim_node(f"n{i}", 24, span=[lo, hi]), now=0)
        r("leave", id="n3", now=0)
        r("join", node=sim_node("n3", 24), now=900)
        r("repair", now=1800)

    def no_mover(r):
        r("join", node=sim_node("n1", 24, span=[0, 24]), now=0)
        r("join", node=sim_node("n2", 16, span=[24, 40]), now=0)
        r("repair", now=100000)

    cases += [
        _pcase("repair: hole timer, mover, make-before-break, damping", "a hole is repaired only after "
               "hole_repair_s by a node held k_min times; its old span counts until complete_move; one move "
               "in flight and one per move_interval_s", SIM, SIM_POLICY, hole_repair),
        _pcase("repair: disk-warm mover first", "the mover with the target on disk wins (zero download)", SIM,
               SIM_POLICY, hole_repair_disk),
        _pcase("repair: k < k_min after kmin_repair_s", "only a node held k_min+1 times moves; nobody drops "
               "below k_min", SIM, SIM_POLICY, kmin_repair),
        _pcase("repair: a disk-warm rejoin preempts k_min repair", "the dead span's owner returns before the "
               "timer", SIM, SIM_POLICY, kmin_rejoin),
        _pcase("repair: no legal mover, no move", "nobody may drop a layer it alone holds", SIM, SIM_POLICY,
               no_mover),
    ]

    def mixed_short(r):
        for nid, kind in (("Spark", "spark-128g"), ("R5090", "5090"), ("R3090", "3090")):
            r("join", node=node(nid, kind), now=0)
        r("counts")
        r("join", node=node("R3090b", "3090"), now=1)
        r("counts")

    def only24(r):
        for i in range(18):
            r("join", node=node(f"G{i}", "3090" if i % 2 else "4090"), now=i)
        r("counts")

    def all_classes(r):
        for i, kind in enumerate(sorted(DEVICES)):
            r("join", node=node(f"X{i}", kind), now=i)
        r("counts")

    def churn(r):
        rng = random.Random(11)
        kinds = sorted(DEVICES)
        live = []
        for step in range(60):
            if live and rng.random() < 0.35:
                nid = live.pop(rng.randrange(len(live)))
                r("leave", id=nid, now=step * 60)
            else:
                nid = f"N{step}"
                live.append(nid)
                r("join", node=node(nid, rng.choice(kinds)), now=step * 60)
            if step % 5 == 4:
                r("repair", now=step * 60)
                r("counts")

    cases += [
        _pcase("mixed: Spark + 5090 + 3090 is 3 layers short", "31 + 8 + 6 = 45 of 48 Flash layers; one more "
               "24 GB card completes a 4-hop path", FLASH, pol_dict(**CATALOG), mixed_short),
        _pcase("mixed: a fleet of only 24 GB cards", "two full copies, 9 hops (the tail card holds 5)", FLASH,
               pol_dict(**CATALOG), only24),
        _pcase("mixed: every device class in one cell", "stages, a backup holder, whole-servers and a seeder",
               FLASH, pol_dict(**CATALOG), all_classes),
        _pcase("mixed: deterministic churn", "joins, leaves and repairs of random device families; spans must "
               "match exactly", FLASH, pol_dict(**CATALOG), churn),
    ]
    return cases


# ── paths cases ───────────────────────────────────────────────────────────────────────────────

def _views(kinds, policy):
    c = Cell(MIMO_V26_FLASH, Policy.from_dict(policy))
    for nid, kind in kinds:
        c.join(Node.from_dict(node(nid, kind)), 0.0)
    return [{"id": h, "span": list(x.span), "role": x.verdict.role,
             "layer_ms": None if math.isinf(x.verdict.layer_ms) else x.verdict.layer_ms,
             "subnet": x.node.subnet, "reliability": x.node.reliability, "rho": 0.0, "sessions": 0,
             "kv_free": 262144, "uplink_mbps": x.node.uplink_mbps} for h, x in sorted(c.holdings.items())]


def _qcase(name, why, holders, args, policy=None):
    case = {"name": name, "why": why, "profile": FLASH, "policy": policy or pol_dict(), "holders": holders,
            "args": args}
    case["expect"] = _jsonable(exec_paths(case, PROFILES))
    return case


def paths_cases():
    p = pol_dict()
    f14 = _views([(f"F{i}", "5090") for i in range(14)], p)
    head = candidate_paths(f14, MIMO_V26_FLASH, Policy())[0]["hops"][0]["id"]
    loaded = [dict(v, rho=0.5) if v["id"] == head else v for v in f14]
    full = [dict(v, sessions=16) if v["id"] == head else v for v in f14]
    fat = _views([("A", "pro6000-96g"), ("B", "pro6000-96g"), ("C", "pro6000-96g"), ("D", "pro6000-96g")], p)
    same = [dict(v, subnet="same") if v["id"] in ("A", "B") else v for v in fat]
    slow_up = [dict(v, uplink_mbps=5.0) if v["id"] == "A" else v for v in fat]
    mix = _views([("P1", "pro6000-96g"), ("P2", "pro6000-96g")] + [(f"G{i}", "3090") for i in range(9)], p)
    few = _views([("P1", "pro6000-96g"), ("P2", "pro6000-96g")] + [(f"G{i}", "3090") for i in range(4)], p)
    cpu = {"id": "C", "span": [0, 48], "role": "backup", "layer_ms": 4.0, "subnet": "C", "reliability": 0.9,
           "rho": 0.0, "sessions": 0, "kv_free": 262144, "uplink_mbps": 100.0}
    return [
        _qcase("14 x 5090: 6-hop mirror-backed paths", "29.2 tok/s at h = 13 (design-mesh §1)", f14, {"k": 3}),
        _qcase("load prices the queue", "rho 0.5 on the best head: q(rho) moves the path", loaded, {"k": 3}),
        _qcase("S_max filters a primary", "sessions >= s_max: not a primary (still a backup)", full,
               {"k": 3, "kv_tokens": 4096}),
        _qcase("kv filter", "kv_free below kv_tokens: neither primary nor backup", f14, {"k": 2, "kv_tokens": 300000}),
        _qcase("distinct subnets", "A and B share a subnet: never on one path", same, {"k": 5}),
        _qcase("measured rtt and the entry legs", "a 90 ms leg re-routes", fat,
               {"k": 3, "rtt": {"Entry1": {"A": 2.0}, "A": {"B": 90.0}}, "entry": "Entry1"}),
        _qcase("prefill ranks a slow uplink down", "16k prompt: A's 5 Mbps uplink dominates TTFT", slow_up,
               {"k": 3, "prompt_tokens": 16384}),
        _qcase("fat nodes carry sessions: replay-backed 2 hops beat a mirror-backed 9-hop chain", "no single "
               "holder mirrors a 27-layer hop, the union of 24 GB cards covers it; the mirror-backed 24 GB chain "
               "costs > backed_slack x the fat path", mix, {"k": 4}),
        _qcase("too few small cards: no failover for the fat hops", "4 x 24 GB cannot cover a 27-layer hop",
               few, {"k": 3}),
        _qcase("degraded: only a backup-role holder", "a big-but-slow holder carries the path when nothing "
               "else can", [cpu], {"k": 3}),
        _qcase("a solo holder backed by a backup-role holder", "the backup is never on its own path",
               [cpu, dict(cpu, id="M", role="stage", layer_ms=1.15, subnet="M", reliability=1.0)], {"k": 3}),
    ]


# ── lease cases ───────────────────────────────────────────────────────────────────────────────

BASE_CLAIMS = {"typ": L.TYP, "lid": "00112233445566ff", "job": "job-1", "sid": "0123456789abcdef", "rte": 3,
               "model": FLASH, "cell": "eu-1", "role": "mid", "from": "PeerA", "to": "PeerB", "lo": 8,
               "hi": 16, "kv": 4096, "iat": 1000, "exp": 4600}


def lease_doc():
    pub = pub_b64(KEY)
    other = ed25519.Ed25519PrivateKey.from_private_bytes(hashlib.sha256(b"not the orchestrator").digest())
    good = L.issue(KEY, BASE_CLAIMS)
    p64, s64 = good.split(".")

    def resign(payload: bytes, key=KEY, domain=L.DOMAIN):
        return f"{L._b64u(payload)}.{L._b64u(key.sign(domain + payload))}"

    def swap(field, val):
        c = json.loads(L._unb64u(p64))
        c[field] = val
        return f"{L._b64u(L.canonical_payload(c))}.{s64}"

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    body = json.dumps({**BASE_CLAIMS, "iss": pub}, sort_keys=True, separators=(",", ":"))
    tokens = [
        ("valid", good, {}),
        ("valid as bytes-equivalent string at the last valid second", good, {"now": 4599}),
        ("skew: 30 s before iat", good, {"now": 970}),
        ("rte changed, signature kept", swap("rte", 4), {}),
        ("to changed, signature kept", swap("to", "PeerC"), {"audience": "PeerC"}),
        ("kv changed, signature kept", swap("kv", 1 << 20), {}),
        ("exp extended, signature kept", swap("exp", 99999), {}),
        ("role changed, signature kept", swap("role", "tail"), {}),
        ("signed by another key", L.issue(other, BASE_CLAIMS), {}),
        ("signed without the domain tag", resign(L.canonical_payload({**BASE_CLAIMS, "iss": pub}), domain=b""), {}),
        ("iss is another key, signed by the pinned key", resign(L.canonical_payload({**BASE_CLAIMS,
                                                                                     "iss": pub_b64(other)})), {}),
        ("unknown claim", resign(body[:-1].encode() + b',"zz":1}'), {}),
        ("missing claim", resign(json.dumps({k: v for k, v in {**BASE_CLAIMS, "iss": pub}.items() if k != "kv"},
                                            sort_keys=True, separators=(",", ":")).encode()), {}),
        ("float rte", resign(body.replace('"rte":3', '"rte":3.0').encode()), {}),
        ("bool rte", resign(body.replace('"rte":3', '"rte":true').encode()), {}),
        ("duplicate key", resign(body[:-1].encode() + b',"rte":9}'), {}),
        ("NaN", resign(body.replace('"kv":4096', '"kv":NaN').encode()), {}),
        ("uppercase sid", resign(body.replace(BASE_CLAIMS["sid"], BASE_CLAIMS["sid"].upper()).encode()), {}),
        ("self edge", resign(body.replace('"to":"PeerB"', '"to":"PeerA"').encode()), {"audience": "PeerA"}),
        ("empty layer range", resign(body.replace('"lo":8', '"lo":16').encode()), {}),
        ("ttl over 24 h", resign(body.replace('"exp":4600', f'"exp":{1000 + L.MAX_TTL_S + 1}').encode()), {}),
        ("non-canonical base64 (unused bits set)", f"{p64}.{s64[:-1]}{alphabet[alphabet.index(s64[-1]) | 1]}", {}),
        ("padding", good + "=", {}),
        ("three parts", good + ".x", {}),
        ("not base64url", good.replace("-", "+").replace("_", "/") if ("-" in good or "_" in good) else
         "+" + good, {}),
        ("empty", "", {}),
        ("oversize", "A" * (L.MAX_TOKEN + 1), {}),
        ("wrong audience", good, {"audience": "PeerC"}),
        ("wrong sender (a stolen token)", good, {"sender": "Mallory"}),
        ("expired", good, {"now": 4600}),
        ("not yet valid", good, {"now": 969}),
    ]
    verify = []
    for name, tok, over in tokens:
        v = {"name": name, "token": tok, "pub": pub, "audience": "PeerB", "sender": "PeerA", "now": 2000,
             "skew": 30, **over}
        v["expect"] = _jsonable(exec_verify(v))
        verify.append(v)
    issue = [{"claims": {k: v for k, v in BASE_CLAIMS.items()}, "token": good}]
    for over in ({"role": "head", "from": "Entry1", "lo": 0}, {"role": "backup", "to": "PeerZ"},
                 {"role": "solo", "lo": 0, "hi": 48, "kv": 0}):
        c = {**BASE_CLAIMS, **over}
        issue.append({"claims": c, "token": L.issue(KEY, c)})

    def t(**over):
        return L.issue(KEY, {**BASE_CLAIMS, **over})

    gates = []
    for name, build in (
        ("redial ok, stale rte refused, supersede, close", lambda r: [
            r("admit", token=t(), sender="PeerA", now=2000),
            r("admit", token=t(), sender="PeerA", now=2001),
            r("admit", token=t(rte=4, lid="00112233445566fe"), sender="PeerA", now=2002),
            r("admit", token=t(), sender="PeerA", now=2003),
            r("supersede", sid=BASE_CLAIMS["sid"], rte=5, now=2004),
            r("admit", token=t(rte=4, lid="00112233445566fe"), sender="PeerA", now=2005),
            r("admit", token=t(rte=5, lid="00112233445566fd"), sender="PeerA", now=2006),
            r("close", sid=BASE_CLAIMS["sid"], now=2007),
            r("admit", token=t(rte=5, lid="00112233445566fd"), sender="PeerA", now=2008),
            r("admit", token=t(sid="fedcba9876543210"), sender="PeerA", now=2009)]),
        ("prune waits for every older token", lambda r: [
            r("admit", token=t(rte=2, lid="00112233445566bb", iat=1100, exp=1200), sender="PeerA", now=1150),
            r("prune", now=1300),
            r("admit", token=t(rte=1, lid="00112233445566aa", exp=1000 + L.MAX_TTL_S), sender="PeerA", now=1300),
            r("prune", now=1100 + L.MAX_TTL_S + 31),
            r("admit", token=t(rte=1, lid="00112233445566aa", iat=90000, exp=90100), sender="PeerA", now=90050)]),
    ):
        rec = Rec(exec_gate, {"gate": L.LeaseGate(pub, "PeerB")})
        build(rec)
        gates.append({"name": name, "me": "PeerB", "pub": pub, "skew": 30, "steps": rec.steps})
    return {"schema": SCHEMA, "kind": "lease", "about": __doc__.split("\n\n")[0],
            "wire": "b64url(canonical claims JSON) '.' b64url(Ed25519(key, DOMAIN || payload)), no padding",
            "domain": L.DOMAIN.decode(), "max_token": L.MAX_TOKEN, "max_ttl_s": L.MAX_TTL_S,
            "claims": list(L.CLAIMS), "key": {"seed_hex": KEY_SEED.hex(), "pub_b64": pub,
                                              "note": "PUBLIC test key; never a production key"},
            "issue": issue, "verify": verify, "gate": gates}


# ── directory cases ───────────────────────────────────────────────────────────────────────────

def _dcase(name, why, build, policy=None):
    case = {"name": name, "why": why, "key_seed_hex": KEY_SEED.hex(), "entry": "Entry1", "lease_ttl_s": 3600,
            "policy": policy or pol_dict()}
    r = Rec(exec_directory, new_directory(case, PROFILES))
    build(r)
    case["steps"] = r.steps
    return case


def _lease_args(p, **kw):
    a = {"job": "job-1", "sid": "0123456789abcdef", "rte": 0, "cell": "eu-1", "model": FLASH,
         "hops": p["hops"], "kv_tokens": 4096, "load": 0.2, "now": 100}
    a.update(kw)
    return a


def directory_cases():
    def twelve(r):
        for i in range(12):
            r("join", node=node(f"F{i}", "5090"), model=FLASH, now=0)

    def bump(r):
        twelve(r)
        (p, *_) = r("paths", cell="eu-1", model=FLASH, args={"k": 3})
        r("lease", args=_lease_args(p))
        r("snapshot", cell="eu-1", model=FLASH)
        r("paths", cell="eu-1", model=FLASH, args={"k": 3})

    def backup_on_path(r):
        twelve(r)
        (p, *_) = r("paths", cell="eu-1", model=FLASH, args={"k": 1})
        hops = [dict(h) for h in p["hops"]]
        hops[0]["backup"] = hops[1]["id"]
        r("lease", args=_lease_args({"hops": hops}))
        r("snapshot", cell="eu-1", model=FLASH)

    def refusals(r):
        twelve(r)
        (p, *_) = r("paths", cell="eu-1", model=FLASH, args={"k": 1})
        hops = p["hops"]
        bad = lambda i, **kw: [dict(h, **kw) if j == i else dict(h) for j, h in enumerate(hops)]  # noqa: E731
        r("lease", args=_lease_args({"hops": bad(1, hi=hops[1]["hi"] + 1)}))                     # bad_path
        r("lease", args=_lease_args({"hops": hops[:-1]}))                                       # bad_path
        r("lease", args=_lease_args({"hops": bad(2, id="Ghost")}))                              # unknown_holder
        r("lease", args=_lease_args({"hops": bad(2, backup=hops[0]["backup"])}))                # backup_span
        r("lease", args=_lease_args(p, sid="NOT-HEX"))                                          # bad_request
        r("update", record={"id": hops[2]["id"], "model": FLASH, "ts": 50, "sessions": 0, "rho": 0.9,
                            "kv_free": 262144})
        r("lease", args=_lease_args(p))                                                         # overloaded
        r("update", record={"id": hops[2]["id"], "model": FLASH, "ts": 51, "sessions": 0, "rho": 0.0,
                            "kv_free": 262144})
        r("update", record={"id": hops[1]["backup"], "model": FLASH, "ts": 51, "sessions": 0, "rho": 0.0,
                            "kv_free": 100})
        r("lease", args=_lease_args(p))                                                         # kv_full
        r("update", record={"id": hops[1]["backup"], "model": FLASH, "ts": 52, "sessions": 0, "rho": 0.0,
                            "kv_free": 262144})
        r("report_failure", id=hops[3]["id"], model=FLASH, now=60)
        r("lease", args=_lease_args(p))                                                         # suspect
        r("update", record={"id": hops[3]["id"], "model": FLASH, "ts": 61, "sessions": 0, "rho": 0.0,
                            "kv_free": 262144})
        r("lease", args=_lease_args(p))                                                         # ok
        r("lease", args=_lease_args(p))                                                         # duplicate_lease
        r("snapshot", cell="eu-1", model=FLASH)

    def absorb(r):
        twelve(r)
        (p, *_) = r("paths", cell="eu-1", model=FLASH, args={"k": 1})
        le = r("lease", args=_lease_args(p, now=100))
        h = p["hops"][0]["id"]
        r("update", record={"id": h, "model": FLASH, "ts": 90, "sessions": 2, "rho": 0.1, "kv_free": 50000})
        r("snapshot", cell="eu-1", model=FLASH)
        r("update", record={"id": h, "model": FLASH, "ts": 100, "sessions": 3, "rho": 0.3, "kv_free": 45904})
        r("update", record={"id": h, "model": FLASH, "ts": 95, "sessions": 0, "rho": 0.0, "kv_free": 1})
        r("snapshot", cell="eu-1", model=FLASH)
        r("release", lease=le["ok"]["lease"])
        r("snapshot", cell="eu-1", model=FLASH)
        r("lease", args=_lease_args(p, rte=1, now=200))
        r("lease", args=_lease_args(p, sid="fedcba9876543210", now=200 + 3600))   # the first one expired
        r("snapshot", cell="eu-1", model=FLASH)

    def mixed(r):
        for nid, kind in (("Spark", "spark-128g"), ("Pro", "pro6000-96g"), ("G1", "3090"), ("G2", "4090"),
                          ("S8", "4060-8g"), ("Mac", "mac-192g"), ("Cpu", "cpu-256g")):
            r("join", node=node(nid, kind), model=FLASH, now=0)
        r("snapshot", cell="eu-1", model=FLASH)
        (p, *_) = r("paths", cell="eu-1", model=FLASH, args={"k": 3, "kv_tokens": 4096})
        r("lease", args=_lease_args(p))
        r("leave", id="Pro", model=FLASH, now=200)
        r("paths", cell="eu-1", model=FLASH, args={"k": 3})
        r("join", node=node("Pro", "pro6000-96g", disk={FLASH: [[22, 48]]}), model=FLASH, now=300)

    def repair(r):
        for i in range(6):
            r("join", node=node(f"F{i}", "5090"), model=FLASH, now=0)
        r("leave", id="F3", model=FLASH, now=10)
        r("repair", cell="eu-1", model=FLASH, now=700)
        for i in range(6, 12):
            r("join", node=node(f"F{i}", "5090"), model=FLASH, now=800)
        r("leave", id="F9", model=FLASH, now=900)
        r("repair", cell="eu-1", model=FLASH, now=900 + 1800)
        r("snapshot", cell="eu-1", model=FLASH)

    return [
        _dcase("bump: the next query sees the lease", "every hop +1 session and +load, every backup reserves "
               "KV, the next best path avoids the leased hops", bump),
        _dcase("backup on the path is refused, nothing changes", "ARCHITECTURE §4.4 hard filter", backup_on_path),
        _dcase("every refusal code, each atomic", "bad_path, unknown_holder, backup_span, bad_request, "
               "overloaded, kv_full, suspect, then ok and duplicate_lease", refusals),
        _dcase("record pushes absorb bumps by timestamp; release and expiry undo them", "Parallax-style bump "
               "with 5 s pushes", absorb),
        _dcase("mixed fleet: roles, paths, tokens, a death and a sticky return", "Spark, 96 GB, 24 GB, 8 GB, "
               "a 192 GB Mac and a CPU box in one cell", mixed),
        _dcase("repair through the directory", "a hole is filled by the next arrival; no legal k_min mover",
               repair),
    ]


def build() -> dict:
    head = {"schema": SCHEMA, "about": __doc__.split("\n\n")[0],
            "profiles": {k: p.to_dict() for k, p in sorted(PROFILES.items())}}
    return {
        "mesh_placement.json": {**head, "kind": "placement", "cases": placement_cases()},
        "mesh_paths.json": {**head, "kind": "paths", "cases": paths_cases()},
        "mesh_lease.json": lease_doc(),
        "mesh_directory.json": {**head, "kind": "directory", "cases": directory_cases()},
    }


def dump(doc) -> str:
    """Sorted keys; objects expanded to depth 2 and lists of objects to depth 3, so each case
    field, step, vector or holder sits on its own line: readable diffs, far smaller than indent=1."""
    def enc(x, depth):
        expand = (isinstance(x, dict) and depth <= 2) or (
            isinstance(x, list) and depth <= 3 and all(isinstance(v, dict) for v in x))
        if not expand or not x:
            return json.dumps(x, sort_keys=True, separators=(",", ":"))
        pad = " " * (depth + 1)
        if isinstance(x, dict):
            items = [f"{pad}{json.dumps(k)}: {enc(x[k], depth + 1)}" for k in sorted(x)]
            return "{\n" + ",\n".join(items) + "\n" + " " * depth + "}"
        return "[\n" + ",\n".join(pad + enc(v, depth + 1) for v in x) + "\n" + " " * depth + "]"
    return enc(doc, 0) + "\n"


def main() -> None:
    os.makedirs(VEC_DIR, exist_ok=True)
    for name, doc in build().items():
        with open(os.path.join(VEC_DIR, name), "w") as f:
            f.write(dump(doc))
        print(name, len(dump(doc)) // 1024, "KiB")


if __name__ == "__main__":
    main()
