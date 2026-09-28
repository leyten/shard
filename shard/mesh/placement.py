"""Mesh placement: what a MIXED fleet holds, what role each device plays, and the router's paths.

design-mesh.md §3-§4 and §6.7, ARCHITECTURE-v2 §4.4 (the M3 PR of BUILD-PLAN §1). Pure and
deterministic: plain data in, plain data out, no clock (callers pass `now`), no IO, no randomness.
The c0mpute orchestrator implements the same rules in TypeScript and both implementations must pass
tests/vectors/mesh_*.json, so every tie-break below is part of the spec.

DEVICE-AGNOSTIC BY CONSTRUCTION. Nothing here knows a card name. A node is its capability record:
`DeviceInfo` from `StageRuntime.describe()` (memory, bandwidth, unified-memory flag, formats), an
optional probe `Capability` for this model (measured footprint and service_ms), plus cell, subnet,
reliability, uplink and disk holdings. Every decision is arithmetic over those numbers and the
MODEL PROFILE (per-layer bytes, KV bytes, edge tensors), never over a card type:

  fit    Contiguous layers [lo, hi) fit iff
             weights(lo, hi) + KV headroom + edges + reserve (+ host state if unified) <= free memory
         weights = the profile's per-layer bytes, scaled by the node's own measured footprint when
         its probe reports one (an Ampere card that cannot keep MXFP4 packed shows up here, not as
         a special case); KV headroom = plan_sessions sessions at plan_ctx (global layers) and
         (window - 1 + spec_window) rows (SWA layers); edges = embed at the head, lm_head + drafter
         (+ its embed copy) at the tail; on a unified-memory device (Mac, DGX Spark) the node's
         host-side state (activation log, peers' KV mirrors) shares the same pool.
  cost   t = measured ms/layer (probe service_ms, decode shape), else the bandwidth bound
         active_bytes_per_layer / mem_bw, floored at the profile's launch-bound layer_ms. Unknown
         bandwidth and no measurement -> t = inf (never a stage until measured).
  stage  The hop law of design-mesh §1, L = (hops+1)*leg + sum(s_i * t_i) + tail, generalised
         from probe.derive_role's homogeneous-ring hop budget to a mixed path. A path meets the
         cell's single-session bar (tok/s = g / L) iff every hop carries its layers within the
         per-layer budget:
             budget = (g / bar * 1000 - leg_ms - tail_ms) / n_layers                  [ms/layer]
             hop cost = leg_ms + s * t <= s * budget   <=>   s >= leg_ms / (budget - t)
         so a device is a STAGE iff it holds s_min = ceil(leg_ms / (budget - t)) layers. Big
         devices clear it and shorten chains; a small or slow one would lengthen every path.

ROLES (classify(); first match wins, each a function of the numbers above):
  stage         fit >= s_min(t). Holds a span, serves paths, backs up its peers (k >= 2 means
                every holder is someone's backup). A device that fits the whole model is a 1-hop
                ("solo") path.
  backup        fit >= s_min at the PROFILE's per-layer cost but not at its own: big enough to
                hold a stage-sized span, too slow to shorten a chain. Holds a span, counts toward
                per-layer coverage k, takes mirrors and takeovers, never a primary hop unless no
                stage path exists (then the path is marked degraded).
  whole-server  can serve a small model of the policy's catalog whole (the c0mpute single-GPU
                path): the small card's best revenue is outside the mesh (design-mesh §8).
  verifier      can compute >= 1 layer of this model: a latency-tolerant audit replica.
  seeder        everything else (CPU boxes, unsupported formats): serves verified weight ranges.
Secondary roles (`also`) list the other non-stage roles a node qualifies for.

JOIN PRIORITY (design-mesh §3.3, lexicographic over the node's maximal candidate spans):
  1 close holes (layers with 0 holders)   2 raise layers below k_min
  3 raise the cell's minimum capacity (Petals; tie: the emptiest window)
  4 layers already on disk
  5 fill a coverage valley: start where coverage steps down, end where it steps up (so replicas
    stack on the same boundaries, hops find whole-span backups, and no 1-layer gap is stranded)
  6 lane-aligned start   7 lowest start
A returning node is re-seated on its persisted span with zero download (sticky), unless it can
close a hole and every layer of its old span is still held k_min times without it.

REPAIR (design-mesh §6.7, ARCHITECTURE C22): joins and leaves never move a healthy node. Only
repair() moves, and only (a) a node whose every layer is held >= k_min times, to close a hole that
persisted hole_repair_s, or (b) a node whose every layer is held >= k_min + 1 times, to raise a
layer below k_min that persisted kmin_repair_s (time for a disk-warm rejoin of the dead span
first). Disk-warm movers first, then the smallest span. Damped: one move in flight per cell, at
most one per move_interval_s, make-before-break (the old span keeps counting until the node
reports the new one warm). No throughput-rebalancing moves at all: damped balancing bought no
coverage over repair-only in placement_sim and costs multi-GB downloads.
"""
from __future__ import annotations

import math
import re
from bisect import bisect_left, insort
from dataclasses import dataclass
from operator import itemgetter
from typing import Iterable, Mapping

from ..probe import SPEC_V0
from ..runtime.api import FORMATS, LAYER_KINDS, Capability, DeviceInfo
from ..topology import predict_prefill_ms

GiB = 1 << 30
ROLES = ("stage", "backup", "whole-server", "verifier", "seeder")
HOLDING_ROLES = ("stage", "backup")

# Ids travel in lease tokens and frames: PeerId charset (shard/frame.py _PEER), cells short slugs.
PEER_RE = re.compile(r"[A-Za-z0-9]{1,128}")
CELL_RE = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,63}")


def _q9(x: float) -> int:
    """Float -> integer key at 1e-9 resolution, so near-equal sums tie identically in Python and
    TypeScript (floor(x * 1e9 + 0.5) is the same IEEE computation in both)."""
    return math.floor(x * 1e9 + 0.5)


# ── model profile ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelProfile:
    """What placement needs to know about one model. Per-layer bytes are resident weights in the
    reference format (MXFP4 experts stay packed); a node's measured footprint scales them.
    kv_bytes_per_pos = (global, swa) bytes per layer per position. g = useful tokens per round
    trip (drafter-dependent); layer_ms / tail_ms = the hop-law compute terms (design-mesh §1)."""
    model: str
    n_layers: int
    layer_bytes: tuple[int, ...]
    layer_kinds: tuple[str, ...]
    swa_window: int
    kv_bytes_per_pos: tuple[int, int]
    head_bytes: int
    tail_bytes: int
    shared_edge_bytes: int
    reserve_bytes: int
    weight_formats: tuple[str, ...]
    layer_ms: float
    active_bytes_per_layer: int
    tail_ms: float
    logits_ms: float
    g: float
    lane: int
    wire_bytes_per_pos: int

    def __post_init__(self):
        n = self.n_layers
        if n < 1 or len(self.layer_bytes) != n or len(self.layer_kinds) != n:
            raise ValueError("layer_bytes and layer_kinds need one entry per layer")
        if any(k not in LAYER_KINDS for k in self.layer_kinds):
            raise ValueError(f"layer kinds must be in {LAYER_KINDS}")
        if ("swa" in self.layer_kinds) != (self.swa_window > 0):
            raise ValueError("swa_window > 0 iff the model has SWA layers")
        ints = (*self.layer_bytes, *self.kv_bytes_per_pos, self.head_bytes, self.tail_bytes,
                self.shared_edge_bytes, self.reserve_bytes, self.active_bytes_per_layer,
                self.wire_bytes_per_pos)
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in ints):
            raise ValueError("byte counts are non-negative ints")
        if self.shared_edge_bytes > min(self.head_bytes, self.tail_bytes):
            raise ValueError("shared_edge_bytes is counted in both head_bytes and tail_bytes")
        if any(f not in FORMATS for f in self.weight_formats):
            raise ValueError(f"weight formats must be in {FORMATS}")
        if not (self.layer_ms > 0 and 0 <= self.logits_ms <= self.tail_ms and self.g > 0 and self.lane >= 1):
            raise ValueError("layer_ms > 0, 0 <= logits_ms <= tail_ms, g > 0, lane >= 1")

    def to_dict(self) -> dict:
        return {"model": self.model, "n_layers": self.n_layers, "layer_bytes": list(self.layer_bytes),
                "layer_kinds": list(self.layer_kinds), "swa_window": self.swa_window,
                "kv_bytes_per_pos": list(self.kv_bytes_per_pos), "head_bytes": self.head_bytes,
                "tail_bytes": self.tail_bytes, "shared_edge_bytes": self.shared_edge_bytes,
                "reserve_bytes": self.reserve_bytes, "weight_formats": list(self.weight_formats),
                "layer_ms": self.layer_ms, "active_bytes_per_layer": self.active_bytes_per_layer,
                "tail_ms": self.tail_ms, "logits_ms": self.logits_ms, "g": self.g, "lane": self.lane,
                "wire_bytes_per_pos": self.wire_bytes_per_pos}

    @classmethod
    def from_dict(cls, d: Mapping) -> "ModelProfile":
        d = dict(d)
        for k in ("layer_bytes", "layer_kinds", "kv_bytes_per_pos", "weight_formats"):
            d[k] = tuple(d[k])
        return cls(**d)


def _mimo(model, n, globals_, layer0, moe_global, moe_swa, kv_global, embed, tail_extra, active,
          lane, wire):
    kinds = tuple("global" if i in globals_ else "swa" for i in range(n))
    lb = tuple(layer0 if i == 0 else (moe_global if kinds[i] == "global" else moe_swa) for i in range(n))
    return ModelProfile(
        model=model, n_layers=n, layer_bytes=lb, layer_kinds=kinds, swa_window=128,
        kv_bytes_per_pos=(kv_global, 5120), head_bytes=embed, tail_bytes=embed + tail_extra + embed,
        shared_edge_bytes=embed, reserve_bytes=1_879_048_192, weight_formats=("mxfp4", "fp8_e4m3", "bf16"),
        layer_ms=1.15, active_bytes_per_layer=active, tail_ms=11.45, logits_ms=2.3, g=4.6, lane=lane,
        wire_bytes_per_pos=wire)


# MiMo-V2.6-Flash. Bytes from the safetensors headers (design-runtime §1, [V]): layer 0 dense
# 0.324 GB; MoE layers 3.553 GB (SWA attention) / 3.548 GB (global); embed = lm_head =
# 152576 x 4096 bf16. The tail holds lm_head + final norm + the MTP-3 drafter (1.19 GB) + an embed
# copy (MTP consumes emb(x_{p+1}); design-runtime §5 extras, ARCHITECTURE C11). KV per layer per
# position bf16: global 4 KV heads x (192+128) x 2 = 2560 B, SWA 8 heads = 5120 B. reserve 1.75 GiB
# (design-runtime §5 fit method). layer_ms 1.15 / tail_ms 11.45 / g 4.6 are the MEAS V4 hop-law
# terms (design-mesh §1; tail_ms = draft 9.16 + head/logits ~2.3 = logits_ms); with MTP-3 in chunk
# mode g is lower (ARCHITECTURE C23) - re-derive g after HW1. active_bytes_per_layer (EST): 8 of
# 256 MXFP4 experts (3 x 2048 x 4096 x 4.25 bit) + attention 0.128 GB + router = ~0.237 GB read
# per decode step.
MIMO_V26_FLASH = _mimo("XiaomiMiMo/MiMo-V2.6-Flash", 48, {0, 5, 11, 17, 23, 29, 35, 41, 47},
                       324_000_000, 3_548_000_000, 3_553_000_000, 2560, 1_249_902_592,
                       8_192 + 1_190_000_000, 237_000_000, lane=4, wire=8192)

# MiMo-V2.6-Pro (EST beyond the header bytes): 8.074 GB per MoE layer, 0.670 GB layer 0, embed
# 152576 x 6144 bf16, MTP 2.46 GB, 8 KV heads on global layers. layer_ms / tail_ms / g are carried
# from Flash UNMEASURED; active bytes ~0.53 GB (8 of 384 experts at H=6144 + 0.368 GB attention).
MIMO_V26_PRO = _mimo("XiaomiMiMo/MiMo-V2.6-Pro", 70, {0, 7, 15, 23, 31, 39, 47, 55, 62, 69},
                     670_000_000, 8_074_000_000, 8_074_000_000, 5120, 1_874_853_888,
                     12_288 + 2_460_000_000, 530_000_000, lane=5, wire=12288)

PROFILES = {p.model: p for p in (MIMO_V26_FLASH, MIMO_V26_PRO)}


# ── cell policy ───────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SmallModel:
    """A model small enough to serve whole on one device (c0mpute's single-GPU path).
    need_bytes = weights + KV + runtime reserve, as the catalog measured it."""
    name: str
    need_bytes: int
    formats: tuple[str, ...]


@dataclass(frozen=True)
class Policy:
    """Per-cell placement and routing knobs. leg_ms is the cell's measured one-way leg latency
    (default: the V4-fitted 13 ms, design-mesh §1); bar_tok_s the single-session bar (probe
    SPEC_V0). Times are seconds, the caller's clock."""
    k_min: int = 2
    leg_ms: float = 13.0
    bar_tok_s: float = SPEC_V0["bar_interactive_tok_s"]
    plan_sessions: int = 8
    plan_ctx: int = 32768
    spec_window: int = 64
    unified_host_bytes: int = 4 * GiB
    hole_repair_s: float = 600.0
    kmin_repair_s: float = 1800.0
    move_interval_s: float = 600.0
    s_max: int = 16
    rho_max: float = 0.85
    unreliability_ms: float = 50.0
    backed_slack: float = 1.3        # design-mesh §4.1: a path > 1.3x the best alternative is degraded
    small_models: tuple[SmallModel, ...] = ()

    def __post_init__(self):
        if self.k_min < 1 or self.plan_sessions < 0 or self.plan_ctx < 1 or self.spec_window < 0:
            raise ValueError("k_min >= 1, plan_sessions >= 0, plan_ctx >= 1, spec_window >= 0")
        if not (self.leg_ms >= 0 and self.bar_tok_s > 0 and 0 < self.rho_max < 1 and self.s_max >= 1
                and self.backed_slack >= 1):
            raise ValueError("leg_ms >= 0, bar_tok_s > 0, 0 < rho_max < 1, s_max >= 1, backed_slack >= 1")

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "small_models"}
        d["small_models"] = [{"name": m.name, "need_bytes": m.need_bytes, "formats": list(m.formats)}
                             for m in self.small_models]
        return d

    @classmethod
    def from_dict(cls, d: Mapping) -> "Policy":
        d = dict(d)
        d["small_models"] = tuple(SmallModel(m["name"], int(m["need_bytes"]), tuple(m["formats"]))
                                  for m in d.get("small_models", ()))
        return cls(**d)


# ── nodes ─────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Probe:
    """What placement keeps of a probe describe() for one model: the measured resident bytes of
    the probe span and the measured wall-clock ms of one decode step over the whole span (None if
    warmup did not run). A tail probe's step includes the logits (warmup asks for them)."""
    model: str
    lo: int
    hi: int
    layer_weight_bytes: tuple[int, ...]
    step_ms: float | None
    is_tail: bool = False

    @classmethod
    def from_capability(cls, cap: Capability) -> "Probe":
        shapes = cap.service_ms
        decode = sorted((c, ms) for s, b, c, ms in shapes if s == 1 and b == 1)
        ms = decode[0][1] if decode else (min(shapes, key=lambda e: (e[0] * e[1], e[2]))[3] if shapes else None)
        return cls(cap.model, cap.lo, cap.hi, tuple(cap.layer_weight_bytes), ms, cap.is_tail)


@dataclass(frozen=True)
class Node:
    id: str
    cell: str
    subnet: str
    device: DeviceInfo
    probe: Probe | None = None
    reliability: float = 1.0          # uptime EWMA in (0, 1]
    uplink_mbps: float = 0.0
    disk: tuple[tuple[str, int, int], ...] = ()   # (model, lo, hi) ranges verified on disk
    span: tuple[int, int] | None = None           # node-local persisted span (sticky)

    def __post_init__(self):
        if not PEER_RE.fullmatch(self.id) or not CELL_RE.fullmatch(self.cell):
            raise ValueError(f"node id must be a PeerId [A-Za-z0-9]{{1,128}} and cell a slug: "
                             f"{self.id!r}, {self.cell!r}")
        if not 0 < self.reliability <= 1:
            raise ValueError("reliability is an uptime fraction in (0, 1]")

    @classmethod
    def from_dict(cls, d: Mapping) -> "Node":
        """{"id", "cell", "subnet", "device": DeviceInfo fields, "capability": describe() dict | null,
        "reliability", "uplink_mbps", "disk": {model: [[lo, hi], ...]}, "span": [lo, hi] | null}"""
        dev = dict(d["device"])
        for k in ("formats_hw", "formats_emulated"):
            dev[k] = tuple(dev[k])
        cap = d.get("capability")
        disk = tuple((m, int(lo), int(hi)) for m, rs in sorted((d.get("disk") or {}).items()) for lo, hi in rs)
        span = d.get("span")
        return cls(id=d["id"], cell=d["cell"], subnet=d.get("subnet", d["id"]), device=DeviceInfo(**dev),
                   probe=Probe.from_capability(Capability.from_dict(cap)) if cap else None,
                   reliability=float(d.get("reliability", 1.0)), uplink_mbps=float(d.get("uplink_mbps", 0.0)),
                   disk=disk, span=(int(span[0]), int(span[1])) if span else None)

    def on_disk(self, model: str) -> set[int]:
        return {i for m, lo, hi in self.disk if m == model for i in range(lo, hi)}


def _probe_for(node: Node, profile: ModelProfile) -> Probe | None:
    p = node.probe
    return p if p is not None and p.model == profile.model and 0 <= p.lo < p.hi <= profile.n_layers else None


def footprint_scale(node: Node, profile: ModelProfile) -> float:
    """How much heavier this backend keeps a layer than the profile: the largest measured /
    profile ratio over the probe's representative layers (at least half the profile's heaviest
    layer, so a dense layer 0 cannot hide an MoE packing penalty). 1.0 without such a probe."""
    p = _probe_for(node, profile)
    if p is None or len(p.layer_weight_bytes) != p.hi - p.lo:
        return 1.0
    big = max(profile.layer_bytes) / 2
    ratios = [m / r for m, r in zip(p.layer_weight_bytes, profile.layer_bytes[p.lo:p.hi]) if r >= big]
    return max(ratios, default=1.0)


def layer_cost_ms(node: Node, profile: ModelProfile) -> float:
    """Per-layer decode ms: measured, else the bandwidth bound floored at profile.layer_ms.
    The bandwidth bound only bounds accelerators: a CPU verifying a block is compute-bound, so
    an unmeasured CPU backend (like unknown bandwidth) costs inf until its probe measures it."""
    p = _probe_for(node, profile)
    if p is not None and p.step_ms is not None:
        per = (p.step_ms - (profile.logits_ms if p.is_tail else 0.0)) / (p.hi - p.lo)
        if per > 0:                                  # a tail probe's logits are tail_ms's, not the layers'
            return float(per)
    bw = node.device.mem_bw_bytes_s
    if not bw > 0 or node.device.backend == "cpu":
        return math.inf
    return max(profile.layer_ms, profile.active_bytes_per_layer / bw * 1000.0)


class Fit:
    """need(lo, hi) in O(1) for one node against one profile."""

    def __init__(self, node: Node, profile: ModelProfile, policy: Policy):
        self.n, self.profile, self.policy = profile.n_layers, profile, policy
        s = footprint_scale(node, profile)
        wb = [b if s == 1.0 else math.ceil(b * s) for b in profile.layer_bytes]
        self._w, self._g = [0], [0]
        for i in range(self.n):
            self._w.append(self._w[-1] + wb[i])
            self._g.append(self._g[-1] + (profile.layer_kinds[i] == "global"))
        self.budget = node.device.free_mem_bytes - profile.reserve_bytes - (
            policy.unified_host_bytes if node.device.unified_memory else 0)
        kg, ks = profile.kv_bytes_per_pos
        self._kv_g = policy.plan_sessions * kg * policy.plan_ctx
        swa_rows = profile.swa_window - 1 + policy.spec_window if profile.swa_window else 0
        self._kv_s = policy.plan_sessions * ks * swa_rows

    def need(self, lo: int, hi: int) -> int:
        """Bytes beyond the reserve that span [lo, hi) needs."""
        p, n = self.profile, self.n
        n_g = self._g[hi] - self._g[lo]
        edges = (p.head_bytes if lo == 0 else 0) + (p.tail_bytes if hi == n else 0) - (
            p.shared_edge_bytes if lo == 0 and hi == n else 0)
        return self._w[hi] - self._w[lo] + n_g * self._kv_g + (hi - lo - n_g) * self._kv_s + edges

    def fits(self, lo: int, hi: int) -> bool:
        return 0 <= lo < hi <= self.n and self.need(lo, hi) <= self.budget

    def spans(self, min_len: int = 1) -> list[tuple[int, int]]:
        """Maximal feasible spans (not contained in another feasible span), ascending lo. For each
        lo the longest span that fits: need() only grows with hi below n (reaching n adds the tail
        tensors), so scan up to the first miss, then try [lo, n) on its own."""
        out, reach, n = [], 0, self.n
        for lo in range(n):
            hi = lo
            while hi + 1 < n and self.fits(lo, hi + 1):
                hi += 1
            if self.fits(lo, n):
                hi = n
            if hi > reach:
                reach = hi
                if hi - lo >= min_len:
                    out.append((lo, hi))
        return out


def stage_min_layers(t_ms: float, profile: ModelProfile, policy: Policy) -> int | None:
    """Fewest layers a hop with per-layer cost t must carry to keep a path at the bar (None: no
    span length does). Evaluated exactly in this order by every implementation."""
    budget = (profile.g / policy.bar_tok_s * 1000.0 - policy.leg_ms - profile.tail_ms) / profile.n_layers
    if not t_ms < budget:
        return None
    return max(1, math.ceil(policy.leg_ms / (budget - t_ms)))


@dataclass(frozen=True)
class Verdict:
    role: str
    also: tuple[str, ...]
    fit_layers: int
    s_min: int | None
    layer_ms: float
    small_models: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"role": self.role, "also": list(self.also), "fit_layers": self.fit_layers,
                "s_min": self.s_min, "layer_ms": None if math.isinf(self.layer_ms) else self.layer_ms,
                "small_models": list(self.small_models)}


def classify(node: Node, profile: ModelProfile, policy: Policy) -> Verdict:
    """The role this node plays for `profile` in a cell with `policy` (module docstring)."""
    t = layer_cost_ms(node, profile)
    formats_ok = all(node.device.format_support(f) for f in profile.weight_formats)
    fit = max((hi - lo for lo, hi in Fit(node, profile, policy).spans()), default=0) if formats_ok else 0
    s_min = stage_min_layers(t, profile, policy)
    s_base = stage_min_layers(profile.layer_ms, profile, policy)
    host = policy.unified_host_bytes if node.device.unified_memory else 0
    small = tuple(m.name for m in policy.small_models
                  if m.need_bytes + host <= node.device.free_mem_bytes
                  and all(node.device.format_support(f) for f in m.formats))
    quals = {"whole-server": bool(small), "verifier": fit >= 1, "seeder": True}
    if s_min is not None and fit >= s_min:
        role = "stage"
    elif s_base is not None and fit >= s_base and math.isfinite(t):
        role = "backup"
    else:
        role = next(r for r in ("whole-server", "verifier", "seeder") if quals[r])
    also = () if role == "stage" else tuple(r for r in ("whole-server", "verifier", "seeder")
                                            if r != role and quals[r])
    return Verdict(role, also, fit, s_min, t, small)


# ── cell placement state ──────────────────────────────────────────────────────────────────────

@dataclass
class Holding:
    node: Node
    verdict: Verdict
    span: tuple[int, int]
    target: tuple[int, int] | None = None      # repair move in flight (make-before-break)


@dataclass(frozen=True)
class Decision:
    id: str
    role: str
    span: tuple[int, int] | None
    download_layers: int
    reason: str
    verdict: Verdict

    def to_dict(self) -> dict:
        return {"id": self.id, "role": self.role, "span": list(self.span) if self.span else None,
                "download_layers": self.download_layers, "reason": self.reason, "verdict": self.verdict.to_dict()}


@dataclass(frozen=True)
class Move:
    id: str
    frm: tuple[int, int]
    to: tuple[int, int]
    download_layers: int
    reason: str            # "hole" | "k_min"

    def to_dict(self) -> dict:
        return {"id": self.id, "from": list(self.frm), "to": list(self.to),
                "download_layers": self.download_layers, "reason": self.reason}


class Cell:
    """Placement state of one (cell, model): holdings, sticky spans, deficit timers."""

    def __init__(self, profile: ModelProfile, policy: Policy):
        self.profile, self.policy = profile, policy
        self.holdings: dict[str, Holding] = {}
        self.others: dict[str, Verdict] = {}                  # non-holding roles
        self.persisted: dict[str, tuple[int, int]] = {}       # sticky spans by node id
        n = profile.n_layers
        self._hole_since: list[float | None] = [None] * n
        self._under_since: list[float | None] = [None] * n
        self._last_move: float | None = None

    # -- coverage ------------------------------------------------------------------------------
    def counts(self, exclude: str | None = None) -> list[int]:
        c = [0] * self.profile.n_layers
        for hid, h in self.holdings.items():
            if hid != exclude:
                for i in range(*h.span):
                    c[i] += 1
        return c

    def capacity(self, exclude: str | None = None) -> list[float]:
        c = [0.0] * self.profile.n_layers
        for hid in sorted(self.holdings):
            if hid != exclude:
                h = self.holdings[hid]
                v = self._cap(h.verdict.layer_ms, h.span[1] - h.span[0])
                for i in range(*h.span):
                    c[i] += v
        return c

    def _cap(self, t: float, span_len: int) -> float:
        """Frame-service capacity a holding adds to each of its layers, in units of the profile's
        per-layer cost: layer_ms / (span_len * t). A fat node spends longer on each frame."""
        return 0.0 if math.isinf(t) else self.profile.layer_ms / (span_len * t)

    def _refresh(self, now: float) -> None:
        c, k = self.counts(), self.policy.k_min
        keep = lambda since: now if since is None else since          # noqa: E731
        for i, v in enumerate(c):
            self._hole_since[i] = keep(self._hole_since[i]) if v == 0 else None
            self._under_since[i] = keep(self._under_since[i]) if v < k else None

    # -- span choice ---------------------------------------------------------------------------
    def span_key(self, span, t: float, cnt: list[int], cap: list[float], disk: set[int]) -> tuple:
        lo, hi = span
        n, k = self.profile.n_layers, self.policy.k_min
        c_new = self._cap(t, hi - lo)
        holes = sum(1 for i in range(lo, hi) if cnt[i] == 0)
        under = sum(1 for i in range(lo, hi) if 0 < cnt[i] < k)
        min_after = min(cap[i] + c_new if lo <= i < hi else cap[i] for i in range(n))
        window = 0.0
        for i in range(lo, hi):
            window += cap[i]
        valley = (lo == 0 or cnt[lo - 1] > cnt[lo]) + (hi == n or cnt[hi] > cnt[hi - 1])
        return (holes, under, _q9(min_after), -_q9(window), len(disk.intersection(range(lo, hi))),
                valley, lo % self.profile.lane == 0, -lo)

    def _choose(self, spans, t, cnt, cap, disk):
        return max(spans, key=lambda s: self.span_key(s, t, cnt, cap, disk))

    def _spans_for(self, node: Node, v: Verdict, cnt: list[int]) -> list[tuple[int, int]]:
        """A stage takes spans of at least s_min layers, except that a shorter maximal span may
        close a hole (a 24 GB card whose tail span is one layer under s_min still holds the tail
        rather than leave it a permanent hole; the path pays for it in rank_ms)."""
        fit = Fit(node, self.profile, self.policy)
        if v.role != "stage":
            return fit.spans()
        long = fit.spans(v.s_min)
        return long + [s for s in fit.spans() if s not in long and any(cnt[i] == 0 for i in range(*s))]

    # -- events --------------------------------------------------------------------------------
    def join(self, node: Node, now: float) -> Decision:
        if node.id in self.holdings or node.id in self.others:
            raise ValueError(f"node {node.id} already joined this cell")
        v = classify(node, self.profile, self.policy)
        if v.role not in HOLDING_ROLES:
            self.others[node.id] = v
            return Decision(node.id, v.role, None, 0, "role", v)
        fit = Fit(node, self.profile, self.policy)
        min_len = v.s_min if v.role == "stage" else 1
        cnt, cap = self.counts(), self.capacity()
        spans = self._spans_for(node, v, cnt)
        disk = node.on_disk(self.profile.model)
        best = self._choose(spans, v.layer_ms, cnt, cap, disk)
        span, reason = best, "join"
        old = self.persisted.get(node.id) or node.span
        if old is not None and fit.fits(*old) and (old[1] - old[0] >= min_len or tuple(old) in spans):
            hole_move = (self.span_key(best, v.layer_ms, cnt, cap, disk)[0] > 0
                         and all(cnt[i] >= self.policy.k_min for i in range(*old)))
            span, reason = (best, "hole") if hole_move else (tuple(old), "sticky")
        self.holdings[node.id] = Holding(node, v, span)
        self.persisted[node.id] = span
        self._refresh(now)
        return Decision(node.id, v.role, span, sum(1 for i in range(*span) if i not in disk), reason, v)

    def leave(self, node_id: str, now: float) -> None:
        """Offline (death or drain done). The persisted span stays: a return is disk-warm."""
        self.holdings.pop(node_id, None)
        self.others.pop(node_id, None)
        self._refresh(now)

    def repair(self, now: float) -> list[Move]:
        """At most one make-before-break move (module docstring). Returns [] or [Move]."""
        self._refresh(now)
        pol, n = self.policy, self.profile.n_layers
        if any(h.target for h in self.holdings.values()):
            return []
        if self._last_move is not None and now - self._last_move < pol.move_interval_s:
            return []
        cnt = self.counts()
        targets = {i for i in range(n) if cnt[i] == 0 and now - self._hole_since[i] >= pol.hole_repair_s}
        reason, floor = "hole", max(pol.k_min, 2)       # the mover leaves >= 1 holder behind
        if not targets:
            targets = {i for i in range(n)
                       if 0 < cnt[i] < pol.k_min and now - self._under_since[i] >= pol.kmin_repair_s}
            reason, floor = "k_min", pol.k_min + 1
        if not targets:
            return []
        under_now = sum(1 for v in cnt if v < pol.k_min)
        options = []
        for hid in sorted(self.holdings):
            h = self.holdings[hid]
            if any(cnt[i] < floor for i in range(*h.span)):
                continue
            cnt_wo, cap_wo = self.counts(exclude=hid), self.capacity(exclude=hid)
            spans = [s for s in self._spans_for(h.node, h.verdict, cnt_wo)
                     if s != h.span and any(s[0] <= i < s[1] for i in targets)]
            if not spans:
                continue
            disk = h.node.on_disk(self.profile.model)
            best = self._choose(spans, h.verdict.layer_ms, cnt_wo, cap_wo, disk)
            if reason == "k_min":
                after = [c + (best[0] <= i < best[1]) for i, c in enumerate(cnt_wo)]
                if sum(1 for v in after if v < pol.k_min) >= under_now:
                    continue
            dl = sum(1 for i in range(*best) if i not in disk)
            options.append((dl, h.span[1] - h.span[0], -h.node.reliability, hid, best))
        if not options:
            return []
        dl, _, _, hid, best = min(options)
        h = self.holdings[hid]
        h.target = best
        self._last_move = now
        return [Move(hid, h.span, best, dl, reason)]

    def complete_move(self, node_id: str, now: float) -> bool:
        """The mover reports its new span warm: switch (its old sessions drained first). If a
        death during the download left the mover the only holder of a layer it would drop, the
        move is cancelled instead (False): the node keeps serving, the new span stays on its disk."""
        h = self.holdings[node_id]
        if h.target is None:
            raise ValueError(f"{node_id} has no move in flight")
        others = self.counts(exclude=node_id)
        if any(others[i] == 0 for i in range(*h.span) if not h.target[0] <= i < h.target[1]):
            h.target = None
            self._refresh(now)
            return False
        h.span, h.target = h.target, None
        self.persisted[node_id] = h.span
        self._refresh(now)
        return True

    def cancel_move(self, node_id: str) -> None:
        h = self.holdings.get(node_id)
        if h is not None:
            h.target = None

    def holes(self) -> list[int]:
        return [i for i, v in enumerate(self.counts()) if v == 0]


# ── paths (the router's input) ────────────────────────────────────────────────────────────────

def _qf(rho: float) -> float:
    """M/D/1 wait factor on a node's frame-service utilization (design-mesh §4.3)."""
    return 1.0 + rho / (2.0 * (1.0 - rho))


def candidate_paths(holders: Iterable[Mapping], profile: ModelProfile, policy: Policy, *, k: int = 3,
                    kv_tokens: int = 0, rtt: Mapping | None = None, entry: str = "entry",
                    prompt_tokens: int = 0, max_new: int = 256) -> list[dict]:
    """Best-k chains through a cell's holders, each hop with a backup where one exists.

    holders: [{"id", "span": [lo, hi], "role": "stage"|"backup", "layer_ms", "subnet", "reliability",
               "rho", "sessions", "kv_free", "uplink_mbps"}]  (live, bumps included)
    rtt: {a: {b: one-way ms}} with the entry named `entry`; a missing pair costs policy.leg_ms.

    A hop starts where the previous one ended (anywhere inside its holder's span) and ends at the
    holder's span end, or where the span of a holder with a lower loaded per-layer cost
    (layer_ms * qf(rho)) starts, so a slow or busy holder hands over to a faster or idler one at
    the first place it could (Petals cut only at span ends; with equal costs nothing changes and
    the state space stays one state per holder). The search is
    a DAG over (holder, cut) states ordered by position, keeping per state the cheapest labels that
    are not dominated k times (prune: co-located holders never starve the distinct-subnet
    filter). Hop cost =
    leg + (q - p) * t * qf(rho) + unreliability_ms * (1 - reliability); step_ms adds the return
    leg and profile.tail_ms; a non-finite or negative rtt is an unreachable pair. Hard filters:
    primaries need rho < rho_max, sessions < s_max and kv_free >= kv_tokens; subnets on a path are
    distinct; a backup is never on its own path and must hold the whole hop with
    kv_free >= kv_tokens. Stage holders only; if they form no complete path, backup-role holders
    may carry hops and the path is `degraded`. Ranking by
    rank_ms = prefill_ms + (max_new / g) * step_ms (prefill_ms from topology.predict_prefill_ms,
    None and 0 when prompt_tokens == 0): paths within policy.backed_slack x the best rank_ms come
    first, ordered by failover class (_FAILOVER: mirror, replay, none), then rank_ms, then fewer
    hops, then ids; slower paths follow in the same order. The slack keeps a fat 2-hop path whose
    hops nobody mirrors ahead of a 9-hop mirror-backed chain of small cards (which would otherwise
    turn the fat devices into idle backups), while equal-cost paths still prefer mirrors. Among
    equal-cost cut variants of one holder sequence the later cuts win."""
    n, rtt = profile.n_layers, rtt or {}
    hs = sorted((dict(h) for h in holders), key=lambda h: h["id"])

    def leg(a, b):
        v = float(rtt.get(a, {}).get(b, policy.leg_ms))
        return v if 0 <= v < math.inf else math.inf

    def ok_primary(h):
        return (h["rho"] < policy.rho_max and h["sessions"] < policy.s_max and h["kv_free"] >= kv_tokens
                and h["layer_ms"] is not None)

    def full_key(c):                  # c = (cost, hops, parent label | None, holder, p, q)
        par = c[2]
        return ((par[2] if par else ()) + (c[3]["id"],), (par[3] if par else ()) + (-c[4],))

    def prune(cands, shared):
        """Scan candidates in (cost, hops, ids, later cuts first) order; keep a label unless its
        holder sequence is already kept or k kept labels dominate it: no costlier, with a conflict
        signature (its subnets that another primary shares) contained in its own. Then any suffix
        that avoids B's shared subnets avoids each dominator's too, so co-located holders can
        never starve the distinct-subnet filter; with unique subnets a state keeps about k labels.
        The one approximation: a dominator may already hold a holder that B's best suffix re-enters
        after an early cut (a holder never serves twice on a path), so that suffix may continue
        from a different label. Labels are materialised only when scanned:
        (cost, hops, ids, -starts, path, subnets, sig)."""
        cands.sort(key=itemgetter(0, 1))
        kept, seqs, i = [], set(), 0
        while i < len(cands):
            j = i + 1
            while j < len(cands) and cands[j][0] == cands[i][0] and cands[j][1] == cands[i][1]:
                j += 1
            run = sorted(cands[i:j], key=full_key) if j - i > 1 else cands[i:j]
            for c in run:
                seq, neg = full_key(c)
                subs = (c[2][5] if c[2] else frozenset()) | {c[3]["subnet"]}
                path = (c[2][4] if c[2] else ()) + ((c[3], c[4], c[5]),)
                sig = subs & shared
                if seq in seqs or sum(1 for a in kept if a[6] <= sig) >= k:
                    continue
                kept.append((c[0], c[1], seq, neg, path, subs, sig))
                seqs.add(seq)
            i = j
        return kept

    def search(roles):
        prim = {h["id"]: h for h in hs if h["role"] in roles and ok_primary(h)}
        per = {hid: h["layer_ms"] * _qf(h["rho"]) for hid, h in prim.items()}     # loaded ms per layer
        seen, shared = set(), set()
        for h in prim.values():
            (shared if h["subnet"] in seen else seen).add(h["subnet"])
        shared = frozenset(shared)
        states = sorted((q, hid) for hid, h in prim.items()
                        for q in {h["span"][1]} | {g["span"][0] for gid, g in prim.items()
                                                   if h["span"][0] < g["span"][0] < h["span"][1]
                                                   and per[gid] < per[hid]})
        by_end, ends, out = {}, [], []
        for q, hid in states:
            h = prim[hid]
            lo = h["span"][0]
            unrel = policy.unreliability_ms * (1.0 - h["reliability"])
            qf = _qf(h["rho"])
            cands = []
            if lo == 0 and leg(entry, hid) < math.inf:
                cands.append((leg(entry, hid) + ((q - 0) * h["layer_ms"] * qf + unrel), 1, None, h, 0, q))
            for p in ends[bisect_left(ends, lo):bisect_left(ends, q)]:
                for gid, glabels in by_end[p]:
                    lg = leg(gid, hid)
                    if gid == hid or lg == math.inf:
                        continue
                    hop = (q - p) * h["layer_ms"] * qf + unrel
                    for lab in glabels:
                        if h["subnet"] not in lab[5]:
                            cands.append((lab[0] + lg + hop, lab[1] + 1, lab, h, p, q))
            kept = prune(cands, shared)
            if q not in by_end:
                by_end[q] = []
                insort(ends, q)
            by_end[q].append((hid, kept))
            if q == n and leg(hid, entry) < math.inf:
                out += [(lab[0] + leg(hid, entry) + profile.tail_ms, lab[4]) for lab in kept]
        return out

    found, degraded = search(("stage",)), False
    if not found:
        found, degraded = search(("stage", "backup")), True
    paths = []
    for step, path in found:
        on_path = {x["id"] for x, _, _ in path}
        hops = []
        for j, (h, p, q) in enumerate(path):
            bk = sorted((b for b in hs if b["id"] not in on_path and b["role"] in HOLDING_ROLES
                         and b["span"][0] <= p and b["span"][1] >= q and b["kv_free"] >= kv_tokens),
                        key=lambda b: (b["role"] != "stage", b["rho"], -b["reliability"], b["id"]))
            role = ("solo" if len(path) == 1 else "head" if j == 0 else
                    "tail" if j == len(path) - 1 else "mid")
            hops.append({"id": h["id"], "lo": p, "hi": q, "role": role, "backup": bk[0]["id"] if bk else None})
        prefill = None
        if prompt_tokens > 0:
            ids = [h["id"] for h in hops]
            L = {a: {b: leg(a, b) for b in ids} for a in ids}
            prefill = predict_prefill_ms(ids, None, L, {i: leg(entry, i) for i in ids},
                                         {i: leg(i, entry) for i in ids},
                                         {h["id"]: float(h.get("uplink_mbps", 0.0)) for h in hs if h["id"] in on_path},
                                         prompt_tokens * profile.wire_bytes_per_pos, -(-prompt_tokens // 2048))
        rank = (prefill or 0.0) + max_new / profile.g * step
        others = [b["span"] for b in hs if b["id"] not in on_path and b["role"] in HOLDING_ROLES]
        failover = ("mirror" if all(h["backup"] for h in hops) else
                    "replay" if all(_covered(h["lo"], h["hi"], others) for h in hops) else "none")
        paths.append({"hops": hops, "step_ms": step, "prefill_ms": prefill, "rank_ms": rank,
                      "failover": failover, "degraded": degraded})
    best = min((p["rank_ms"] for p in paths), default=0.0)
    paths.sort(key=lambda p: (p["rank_ms"] > policy.backed_slack * best, _FAILOVER.index(p["failover"]),
                              _q9(p["rank_ms"]), len(p["hops"]), [h["id"] for h in p["hops"]]))
    return paths[:k]


# Failover class of a path, best first: every hop has a whole-span backup (tier 1 mirror
# takeover); every hop's layers are held by the union of off-path holders (tier 2 multi-holder
# replay, design-mesh §6.2); neither (a hop's death falls back to tier 3 or a coverage hole).
_FAILOVER = ("mirror", "replay", "none")


def _covered(p: int, q: int, spans) -> bool:
    pos = p
    for lo, hi in sorted(spans):
        if lo > pos:
            break
        pos = max(pos, hi)
        if pos >= q:
            return True
    return pos >= q
