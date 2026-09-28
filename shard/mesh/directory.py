"""Reference in-process directory: holder registry, load bumps, lease issuance, path query.

ARCHITECTURE-v2 §4.4. At launch the c0mpute orchestrator IS the directory; it implements this API
in TypeScript and both implementations pass tests/vectors/mesh_*.json. Later the same records move
into the sidecar's Kad-DHT and leases become node-signed (design-mesh M6).

  join(node, model, now)       place a node (shard.mesh.placement.Cell.join): role + span
  leave(id, model, now)        offline; its span stays persisted, so a return is disk-warm
  repair(cell, model, now)     at most one make-before-break move; complete_move() applies it
  update(record)               the node's ~5 s load push {id, model, ts, sessions, rho, kv_free}
  snapshot(cell, model)        holder views, bumps included (what the router consumes)
  paths(cell, model, ...)      best-k chains with backups (placement.candidate_paths)
  lease(...)                   atomic admission + bump at every hop and backup; returns one
                               orchestrator-signed shard-lease/1 token per stream edge
  release(lease)               undo the bumps of a lease that is not (or no longer) serving
  report_failure(id, model, t) routing signal only: the holder leaves new paths until it pushes
                               a record newer than t. Never a reputation hit (that stays
                               progress-gated, design-mesh §6.1)

BUMP (Parallax-style). A lease adds the session's load to every hop (sessions + 1, rho + load,
kv_free - kv_tokens) and reserves kv_tokens at every backup, immediately, so the next path query
already sees it. A node's record carries its own timestamp `ts`: bumps issued at or before `ts`
are assumed to be inside the node's reported numbers and are dropped; later ones stay on top.
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
    """lease() refused before changing any state. `code` is stable across implementations."""

    def __init__(self, code: str, msg: str = ""):
        super().__init__(f"{code}: {msg}" if msg else code)
        self.code = code


@dataclass
class _Load:
    ts: float
    sessions: int
    rho: float
    kv_free: int
    pending: dict = field(default_factory=dict)       # lid -> (t, d_sessions, d_rho, d_kv)
    suspect_since: float | None = None

    def eff(self) -> tuple[int, float, int]:
        s, r, kv = self.sessions, self.rho, self.kv_free
        for lid in sorted(self.pending):
            _, ds, dr, dk = self.pending[lid]
            s, r, kv = s + ds, r + dr, kv - dk
        return s, r, kv


def _hex16(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


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
        self._load: dict[tuple[str, str], _Load] = {}
        self._leases: dict[str, dict] = {}

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
        d = self._cell(n.cell, model).join(n, now)
        self._where[(model, n.id)] = n.cell
        if d.span is not None:
            self._load[(model, n.id)] = _Load(ts=now, sessions=0, rho=0.0,
                                              kv_free=self.policy.plan_sessions * self.policy.plan_ctx)
        return d.to_dict()

    def leave(self, node_id: str, model: str, now: float) -> None:
        cell = self._where.pop((model, node_id), None)
        if cell is not None:
            self.cells[(cell, model)].leave(node_id, now)
            self._load.pop((model, node_id), None)

    def repair(self, cell: str, model: str, now: float) -> list[dict]:
        return [m.to_dict() for m in self._cell(cell, model).repair(now)]

    def complete_move(self, node_id: str, model: str, now: float) -> None:
        self.cells[(self._where[(model, node_id)], model)].complete_move(node_id, now)

    # ── registry ──────────────────────────────────────────────────────────────────────────────
    def update(self, rec: Mapping) -> None:
        """A holder's load push. Stale (older than the last record) pushes are ignored."""
        ld = self._load.get((rec["model"], rec["id"]))
        if ld is None:
            raise KeyError(f"{rec['id']} holds nothing for {rec['model']}")
        ts = float(rec["ts"])
        if ts < ld.ts:
            return
        ld.ts, ld.sessions, ld.rho, ld.kv_free = ts, int(rec["sessions"]), float(rec["rho"]), int(rec["kv_free"])
        ld.pending = {lid: p for lid, p in ld.pending.items() if p[0] > ts}
        if ld.suspect_since is not None and ts > ld.suspect_since:
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

    def lease(self, *, job: str, sid: str, rte: int, cell: str, model: str, hops: list, kv_tokens: int,
              load: float, now: int, entry: str | None = None) -> dict:
        """Admit a session on `hops` ([{"id", "lo", "hi", "backup": id | None}], e.g. a paths()
        entry) atomically: every check runs before any bump. Returns {"lease", "exp", "tokens":
        [{"role", "from", "to", "lid", "token"}]}: one token per hop (pred -> hop; the entry is the
        head's pred) and one per backup (primary -> backup, the mirror stream)."""
        entry = entry or self.entry
        c = self.cells.get((cell, model))
        if c is None:
            raise LeaseRefused("unknown_cell", f"{cell}/{model}")
        pol, n = self.policy, self.profiles[model].n_layers
        if not isinstance(now, int) or not isinstance(kv_tokens, int) or kv_tokens < 0 or not 0 <= load < 1:
            raise LeaseRefused("bad_request", "now and kv_tokens are ints, 0 <= load < 1")
        probe = {"typ": lease_mod.TYP, "lid": "0" * 16, "job": job, "sid": sid, "rte": rte, "model": model,
                 "cell": cell, "role": "head", "from": entry, "to": "x", "lo": 0, "hi": 1, "kv": kv_tokens,
                 "iat": now, "exp": now + self.lease_ttl_s, "iss": self.pubkey}
        try:
            lease_mod.check_claims(probe)
        except lease_mod.LeaseError as e:
            raise LeaseRefused("bad_request", str(e)) from None
        self._expire(now)
        group = _hex16(job, sid, rte)
        if group in self._leases:
            raise LeaseRefused("duplicate_lease", "a re-lease of a session must bump rte")
        if not hops or hops[0]["lo"] != 0 or hops[-1]["hi"] != n or any(
                a["hi"] != b["lo"] for a, b in zip(hops, hops[1:])) or any(h["lo"] >= h["hi"] for h in hops):
            raise LeaseRefused("bad_path", f"hops must tile [0, {n}) contiguously")
        ids = [h["id"] for h in hops]
        if len(set(ids)) != len(ids):
            raise LeaseRefused("duplicate_hop", "a holder appears twice on the path")
        if entry in ids:
            raise LeaseRefused("bad_path", "the entry cannot be a hop")
        need_kv: dict[str, int] = {}
        subnets = set()
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
            if s >= pol.s_max or r >= pol.rho_max:
                raise LeaseRefused("overloaded", h["id"])
            need_kv[h["id"]] = need_kv.get(h["id"], 0) + kv_tokens
            b = h.get("backup")
            if b is not None:
                if b in ids:
                    raise LeaseRefused("backup_on_path", f"backup {b} is a hop of this path")
                bd = c.holdings.get(b)
                if bd is None:
                    raise LeaseRefused("unknown_holder", b)
                if bd.verdict.role not in HOLDING_ROLES or not (bd.span[0] <= h["lo"] and h["hi"] <= bd.span[1]):
                    raise LeaseRefused("backup_span", f"backup {b} holds {bd.span}, not [{h['lo']}, {h['hi']})")
                if self._load[(model, b)].suspect_since is not None:
                    raise LeaseRefused("suspect", b)
                need_kv[b] = need_kv.get(b, 0) + kv_tokens
        for hid, kv in need_kv.items():
            if self._load[(model, hid)].eff()[2] < kv:
                raise LeaseRefused("kv_full", hid)
        # admitted: bump, then sign
        exp = now + self.lease_ttl_s
        tokens, bumps = [], []
        for j, h in enumerate(hops):
            role = "solo" if len(hops) == 1 else "head" if j == 0 else "tail" if j == len(hops) - 1 else "mid"
            edges = [(role, entry if j == 0 else hops[j - 1]["id"], h["id"], 1, float(load))]
            if h.get("backup") is not None:
                edges.append(("backup", h["id"], h["backup"], 0, 0.0))
            for erole, frm, to, ds, dr in edges:
                lid = _hex16(group, j, erole, to)
                claims = {"typ": lease_mod.TYP, "lid": lid, "job": job, "sid": sid, "rte": rte, "model": model,
                          "cell": cell, "role": erole, "from": frm, "to": to, "lo": h["lo"], "hi": h["hi"],
                          "kv": kv_tokens, "iat": now, "exp": exp}
                tok = lease_mod.issue(self._key, claims)   # raises LeaseError on a bad id or field
                tokens.append({"role": erole, "from": frm, "to": to, "lid": lid, "token": tok})
                bumps.append((to, lid, (now, ds, dr, kv_tokens)))
        for to, lid, b in bumps:
            self._load[(model, to)].pending[lid] = b
        self._leases[group] = {"model": model, "bumps": [(to, lid) for to, lid, _ in bumps], "exp": exp}
        return {"lease": group, "exp": exp, "tokens": tokens}

    def _expire(self, now: int) -> None:
        for lid in [g for g, le in self._leases.items() if le["exp"] <= now]:
            self.release(lid)

    def release(self, lease_id: str) -> None:
        le = self._leases.pop(lease_id, None)
        if le is None:
            return
        for to, lid in le["bumps"]:
            ld = self._load.get((le["model"], to))
            if ld is not None:
                ld.pending.pop(lid, None)
