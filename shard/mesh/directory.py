"""Reference in-process directory: holder registry, load bumps, lease issuance, path query.

ARCHITECTURE-v2 §4.4. At launch the c0mpute orchestrator IS the directory; it implements this API
in TypeScript and both implementations pass tests/vectors/mesh_*.json. Later the same records move
into the sidecar's Kad-DHT and leases become node-signed (design-mesh M6).

  join(node, model, now)        place a node (shard.mesh.placement.Cell.join): role + span. A
                                device holding a span serves one model (design-mesh §3.3: the
                                deficit rule picks which model a joining node serves)
  leave(id, model, now)         offline; its span stays persisted, so a return is disk-warm
  repair(cell, model, now)      at most one make-before-break move; complete_move() applies it
  update(record, now)           the node's ~5 s load push {id, model, sessions, rho, kv_free,
                                sids}, stamped by the directory's clock on receipt
  snapshot(cell, model)         holder views, bumps included (what the router consumes)
  paths(cell, model, ...)       best-k chains with backups (placement.candidate_paths)
  lease(...)                    atomic admission + bump at every hop and backup; returns one
                                orchestrator-signed shard-lease/1 token per stream edge
  reroute(lease, rte, hops, t)  the same session on a changed path at a higher route epoch:
                                tokens only for edges that changed, bumps only for new holders
  release(lease, now)           undo the bumps of a lease that no longer serves
  report_failure(id, model, t)  routing signal only: the holder leaves new paths until it pushes
                                a record after t. Never a reputation hit (that stays
                                progress-gated, design-mesh §6.1)

BUMP (Parallax-style). A lease adds the session's load to every hop (sessions + 1, rho + load,
kv_free - kv_tokens) and reserves kv_tokens once at every backup, immediately, so the next path
query already sees it. The bump stays on top of the node's pushed numbers until the node's record
lists the session's sid (then its own numbers include it), or the lease is released or expires.
Node clocks are never trusted: records are stamped on receipt.

SIDS are single-use: a sid is refused while it is leased and for MAX_TTL_S after its release, so
no node's gate (which refuses closed and superseded sids) ever sees two sessions under one sid.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping

from cryptography.hazmat.primitives.asymmetric import ed25519

from ..manifest import pub_b64
from . import lease as lease_mod
from .placement import (HOLDING_ROLES, PEER_RE, PROFILES, Cell, ModelProfile, Node, Policy,
                        candidate_paths)


class LeaseRefused(Exception):
    """lease()/reroute() refused before changing any lease state. `code` is stable across
    implementations. (Expired leases are released first: that is time, not the request.)"""

    def __init__(self, code: str, msg: str = ""):
        super().__init__(f"{code}: {msg}" if msg else code)
        self.code = code


@dataclass
class _Load:
    sessions: int
    rho: float
    kv_free: int
    pending: dict = field(default_factory=dict)       # sid -> (d_sessions, d_rho, d_kv)
    suspect_since: float | None = None

    def eff(self) -> tuple[int, float, int]:
        s, r, kv = self.sessions, self.rho, self.kv_free
        for sid in sorted(self.pending):
            ds, dr, dk = self.pending[sid]
            s, r, kv = s + ds, r + dr, kv - dk
        return s, r, kv


def _hex16(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _roles(hops) -> list[str]:
    n = len(hops)
    return ["solo" if n == 1 else "head" if j == 0 else "tail" if j == n - 1 else "mid" for j in range(n)]


class Directory:
    def __init__(self, key: ed25519.Ed25519PrivateKey, *, profiles: Mapping[str, ModelProfile] = PROFILES,
                 policy: Policy = Policy(), entry: str = "entry", lease_ttl_s: int = 3600):
        if not PEER_RE.fullmatch(entry):
            raise ValueError("entry must be a PeerId")
        if not 0 < lease_ttl_s <= lease_mod.MAX_TTL_S:
            raise ValueError(f"lease_ttl_s must be in (0, {lease_mod.MAX_TTL_S}]")
        self._key, self.pubkey = key, pub_b64(key)
        self.profiles, self.policy, self.entry, self.lease_ttl_s = dict(profiles), policy, entry, lease_ttl_s
        self.cells: dict[tuple[str, str], Cell] = {}
        self._where: dict[tuple[str, str], str] = {}           # (model, id) -> cell
        self._load: dict[tuple[str, str], _Load] = {}          # holders only
        self._leases: dict[str, dict] = {}
        self._sids: dict[str, str] = {}                        # live sid -> lease id
        self._retired: dict[str, int] = {}                     # released sid -> refused until

    # ── placement ─────────────────────────────────────────────────────────────────────────────
    def _cell(self, cell: str, model: str) -> Cell:
        if model not in self.profiles:
            raise KeyError(f"no profile for model {model!r}")
        c = self.cells.get((cell, model))
        if c is None:
            c = self.cells[(cell, model)] = Cell(self.profiles[model], self.policy)
        return c

    def join(self, node, model: str, now: float) -> dict:
        n = node if isinstance(node, Node) else Node.from_dict(node)
        if (model, n.id) in self._where:
            raise ValueError(f"{n.id} already joined {model}")
        held = sorted(m for m, i in self._load if i == n.id)
        if held:
            raise ValueError(f"{n.id} holds a span of {held[0]}: a device holding a span serves one model")
        d = self._cell(n.cell, model).join(n, now)
        self._where[(model, n.id)] = n.cell
        if d.span is not None:
            self._load[(model, n.id)] = _Load(sessions=0, rho=0.0,
                                              kv_free=self.policy.plan_sessions * self.policy.plan_ctx)
        return d.to_dict()

    def leave(self, node_id: str, model: str, now: float) -> None:
        cell = self._where.pop((model, node_id), None)
        if cell is not None:
            self.cells[(cell, model)].leave(node_id, now)
            self._load.pop((model, node_id), None)

    def repair(self, cell: str, model: str, now: float) -> list[dict]:
        return [m.to_dict() for m in self._cell(cell, model).repair(now)]

    def complete_move(self, node_id: str, model: str, now: float) -> bool:
        return self.cells[(self._where[(model, node_id)], model)].complete_move(node_id, now)

    # ── registry ──────────────────────────────────────────────────────────────────────────────
    def update(self, rec: Mapping, now: float) -> None:
        """A holder's load push, stamped `now` by the directory. `sids` lists the sessions the node
        serves (as hop or backup): their bumps are inside its numbers from now on."""
        ld = self._load.get((rec["model"], rec["id"]))
        if ld is None:
            raise KeyError(f"{rec['id']} holds nothing for {rec['model']}")
        ld.sessions, ld.rho, ld.kv_free = int(rec["sessions"]), float(rec["rho"]), int(rec["kv_free"])
        acked = set(rec.get("sids", ()))
        ld.pending = {sid: b for sid, b in ld.pending.items() if sid not in acked}
        if ld.suspect_since is not None and now > ld.suspect_since:
            ld.suspect_since = None

    def report_failure(self, node_id: str, model: str, now: float) -> None:
        ld = self._load.get((model, node_id))
        if ld is not None:
            ld.suspect_since = now if ld.suspect_since is None else ld.suspect_since

    def snapshot(self, cell: str, model: str) -> list[dict]:
        c = self.cells.get((cell, model))
        if c is None:
            return []
        out = []
        for hid in sorted(c.holdings):
            h, ld = c.holdings[hid], self._load[(model, hid)]
            s, r, kv = ld.eff()
            t = h.verdict.layer_ms
            out.append({"id": hid, "span": list(h.span), "role": h.verdict.role,
                        "layer_ms": None if t == float("inf") else t, "subnet": h.node.subnet,
                        "reliability": h.node.reliability, "uplink_mbps": h.node.uplink_mbps,
                        "rho": r, "sessions": s, "kv_free": kv, "suspect": ld.suspect_since is not None,
                        "moving_to": list(h.target) if h.target else None})
        return out

    # ── routing ───────────────────────────────────────────────────────────────────────────────
    def paths(self, cell: str, model: str, *, k: int = 3, kv_tokens: int = 0, rtt: Mapping | None = None,
              entry: str | None = None, prompt_tokens: int = 0, max_new: int = 256) -> list[dict]:
        views = [v for v in self.snapshot(cell, model) if not v["suspect"]]
        return candidate_paths(views, self.profiles[model], self.policy, k=k, kv_tokens=kv_tokens, rtt=rtt,
                               entry=entry or self.entry, prompt_tokens=prompt_tokens, max_new=max_new)

    def _bad_request(self, **claims) -> None:
        probe = {"typ": lease_mod.TYP, "lid": "0" * 16, "role": "head", "to": "x", "lo": 0, "hi": 1,
                 "iss": self.pubkey, **claims}
        try:
            lease_mod.check_claims(probe)
        except lease_mod.LeaseError as e:
            raise LeaseRefused("bad_request", str(e)) from None

    def _check(self, c: Cell, model: str, hops: list, kv_tokens: int, entry: str, prior: Mapping) -> dict:
        """Validate a path; capacity only for what `prior` (id -> "hop" | "backup", this session's
        holders before a reroute) does not already hold. Returns the new bumps {id: (ds, dr, dk)}
        without the load (the caller fills it)."""
        pol, n = self.policy, self.profiles[model].n_layers
        if not hops or hops[0]["lo"] != 0 or hops[-1]["hi"] != n or any(
                a["hi"] != b["lo"] for a, b in zip(hops, hops[1:])) or any(h["lo"] >= h["hi"] for h in hops):
            raise LeaseRefused("bad_path", f"hops must tile [0, {n}) contiguously")
        ids = [h["id"] for h in hops]
        if len(set(ids)) != len(ids):
            raise LeaseRefused("duplicate_hop", "a holder appears twice on the path")
        if entry in ids:
            raise LeaseRefused("bad_path", "the entry cannot be a hop")
        subnets, roles = set(), {}
        for h in hops:
            hd = c.holdings.get(h["id"])
            if hd is None:
                raise LeaseRefused("unknown_holder", h["id"])
            if not (hd.span[0] <= h["lo"] and h["hi"] <= hd.span[1]):
                raise LeaseRefused("span_mismatch", f"{h['id']} holds {hd.span}, not [{h['lo']}, {h['hi']})")
            if hd.node.subnet in subnets:
                raise LeaseRefused("subnet_collision", h["id"])
            subnets.add(hd.node.subnet)
            ld = self._load[(model, h["id"])]
            if ld.suspect_since is not None:
                raise LeaseRefused("suspect", h["id"])
            s, r, _ = ld.eff()
            if prior.get(h["id"]) != "hop" and (s >= pol.s_max or r >= pol.rho_max):
                raise LeaseRefused("overloaded", h["id"])
            roles[h["id"]] = "hop"
        for h in hops:
            b = h.get("backup")
            if b is None:
                continue
            if b in ids:
                raise LeaseRefused("backup_on_path", f"backup {b} is a hop of this path")
            bd = c.holdings.get(b)
            if bd is None:
                raise LeaseRefused("unknown_holder", b)
            if bd.verdict.role not in HOLDING_ROLES or not (bd.span[0] <= h["lo"] and h["hi"] <= bd.span[1]):
                raise LeaseRefused("backup_span", f"backup {b} holds {bd.span}, not [{h['lo']}, {h['hi']})")
            if self._load[(model, b)].suspect_since is not None:
                raise LeaseRefused("suspect", b)
            roles[b] = "backup"
        for hid in sorted(roles):                     # KV once per holder per session
            if hid not in prior and self._load[(model, hid)].eff()[2] < kv_tokens:
                raise LeaseRefused("kv_full", hid)
        return roles

    def _edges(self, hops: list, entry: str) -> list[tuple]:
        out = []
        for j, (h, role) in enumerate(zip(hops, _roles(hops))):
            out.append((role, entry if j == 0 else hops[j - 1]["id"], h["id"], h["lo"], h["hi"]))
            if h.get("backup") is not None:
                out.append(("backup", h["id"], h["backup"], h["lo"], h["hi"]))
        return out

    def _tokens(self, le: dict, edges: list, now: int) -> list[dict]:
        out = []
        for role, frm, to, lo, hi in edges:
            lid = _hex16(le["id"], le["rte"], role, frm, to, lo, hi)
            claims = {"typ": lease_mod.TYP, "lid": lid, "job": le["job"], "sid": le["sid"], "rte": le["rte"],
                      "model": le["model"], "cell": le["cell"], "role": role, "from": frm, "to": to, "lo": lo,
                      "hi": hi, "kv": le["kv"], "iat": now, "exp": le["exp"]}
            out.append({"role": role, "from": frm, "to": to, "lid": lid, "token": lease_mod.issue(self._key, claims)})
        return out

    def _apply_bumps(self, le: dict, roles: dict) -> None:
        for hid, role in roles.items():
            bump = (1, le["load"], le["kv"]) if role == "hop" else (0, 0.0, le["kv"])
            if le["holders"].get(hid) != role:
                self._load[(le["model"], hid)].pending[le["sid"]] = bump
        for hid in set(le["holders"]) - set(roles):
            ld = self._load.get((le["model"], hid))
            if ld is not None:
                ld.pending.pop(le["sid"], None)
        le["holders"] = dict(roles)

    def lease(self, *, job: str, sid: str, rte: int, cell: str, model: str, hops: list, kv_tokens: int,
              load: float, now: int, entry: str | None = None) -> dict:
        """Admit a session on `hops` ([{"id", "lo", "hi", "backup": id | None}], e.g. a paths()
        entry) atomically: every check runs before any bump. Returns {"lease", "rte", "exp",
        "tokens": [{"role", "from", "to", "lid", "token"}]}: one token per hop (pred -> hop; the
        entry is the head's pred) and one per backup (primary -> backup, the mirror stream)."""
        entry = entry or self.entry
        if not isinstance(now, int) or not isinstance(kv_tokens, int) or kv_tokens < 0 or not 0 <= load < 1:
            raise LeaseRefused("bad_request", "now and kv_tokens are ints, 0 <= load < 1")
        self._bad_request(job=job, sid=sid, rte=rte, model=model, cell=cell, kv=kv_tokens, iat=now,
                          exp=now + self.lease_ttl_s, **{"from": entry})
        self._expire(now)
        c = self.cells.get((cell, model))
        if c is None:
            raise LeaseRefused("unknown_cell", f"{cell}/{model}")
        if sid in self._sids or self._retired.get(sid, -1) > now:
            raise LeaseRefused("sid_in_use", "sids are single-use: a new session needs a new sid")
        roles = self._check(c, model, hops, kv_tokens, entry, prior={})
        le = {"id": _hex16(job, sid), "job": job, "sid": sid, "rte": rte, "model": model, "cell": cell,
              "kv": kv_tokens, "load": float(load), "exp": now + self.lease_ttl_s, "hops": [dict(h) for h in hops],
              "entry": entry, "holders": {}}
        tokens = self._tokens(le, self._edges(hops, entry), now)
        self._apply_bumps(le, roles)
        self._leases[le["id"]], self._sids[sid] = le, le["id"]
        return {"lease": le["id"], "rte": rte, "exp": le["exp"], "tokens": tokens}

    def reroute(self, lease_id: str, *, rte: int, hops: list, now: int) -> dict:
        """Move a live session to `hops` at route epoch `rte` (> its current one): a failover
        replacement, a drain, a re-route at a turn boundary. Surviving edges keep their tokens (the
        entry-signed lease_update carries rte to them, C6); every changed edge gets a new token.
        Only holders new to the session are capacity-checked and bumped; dropped ones are
        released. Same return shape as lease(), tokens for the changed edges only."""
        self._expire(now)
        le = self._leases.get(lease_id)
        if le is None:
            raise LeaseRefused("unknown_lease", lease_id)
        if not isinstance(rte, int) or not le["rte"] < rte <= lease_mod.U32 or not isinstance(now, int):
            raise LeaseRefused("stale_rte", f"rte must exceed {le['rte']}")
        c = self.cells[(le["cell"], le["model"])]
        roles = self._check(c, le["model"], hops, le["kv"], le["entry"], prior=le["holders"])
        old = set(self._edges(le["hops"], le["entry"]))
        new = [e for e in self._edges(hops, le["entry"]) if e not in old]
        le.update(rte=rte, exp=now + self.lease_ttl_s, hops=[dict(h) for h in hops])
        tokens = self._tokens(le, new, now)
        self._apply_bumps(le, roles)
        return {"lease": le["id"], "rte": rte, "exp": le["exp"], "tokens": tokens}

    def _expire(self, now: int) -> None:
        for lid in [g for g, le in self._leases.items() if le["exp"] <= now]:
            self.release(lid, now)
        for sid in [s for s, t in self._retired.items() if t <= now]:
            del self._retired[sid]

    def release(self, lease_id: str, now: int) -> None:
        le = self._leases.pop(lease_id, None)
        if le is None:
            return
        self._apply_bumps(le, {})
        self._sids.pop(le["sid"], None)
        self._retired[le["sid"]] = now + lease_mod.MAX_TTL_S
