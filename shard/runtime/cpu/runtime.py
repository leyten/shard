"""CpuRuntime: the CPU reference StageRuntime for MiMo-V2 (shard/runtime/api.py, BUILD-PLAN R2).

The math is the canonical per-row definition in shard/runtime/cpu/mimo_v2.py, the config and tensor
plan the device-neutral adapter's (shard/runtime/adapters/mimo_v2.py); this module owns the per-session STATE with the memory bounds a production backend has, so the CPU
reference validates the bookkeeping every GPU/MLX backend must copy, not a simplification of it:

  * global layers: K/V rows for every position [0, max_pos), bf16, canonical layout;
  * SWA layers: a ring of exactly (w - 1 + W) slots per layer and session (ARCHITECTURE-v2 §4.2:
    a ring of w slots is wrong under speculation). Every slot carries the position it holds and
    every read asserts it, so a lost row is an error, never a silently wrong window;
  * a segment's K/V are computed for all its rows first, each query row then gathers its visible
    keys from the cache (positions < start_pos) and from the segment's fresh rows, and the fresh
    rows are written afterwards. That is the shape of a GPU prefill (a transient chunk buffer), so
    a chunk longer than the ring never overwrites a row a later query of the same chunk reads;
  * takeover: import_state installs mirrored global rows [0, upto); rebuild segments READ them for
    every p < upto and never overwrite them; SWA windows start at the rebuild start a (§4.1).

Numerics: canonical bf16, one row at a time, so batch-, chunk- and partition-invariant; describe()
says chunk_invariant=True and a takeover rebuild is bit-equal to the fault-free state. The
numerics_id pins what the bytes depend on here: torch version, BLAS, the CPU ISA torch dispatches
to (MKL's summation order differs between AVX2 and AVX-512), the weights' artifact id, and the
sampler. Tokens are drawn on the host: greedy = the lowest id among exactly-maximal fp32 logits;
sampled = a float64 softmax over the top_k / top_p cut of logits / temperature, inverted with a
splitmix64 counter keyed by (seed, absolute position) only.
"""
from __future__ import annotations

import dataclasses
import platform
import re
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..adapters.mimo_v2 import MimoConfig, UnsupportedConfig, edge_plan, layer_plan, layer_prefix
from ..api import (ID_DTYPES, LOGITS_MODES, SEG_KINDS, BadRequest, Capability, KVBlob, KVFull,
                  LoadRefused, LoadReport, PositionGap, RollbackPastCommit, Rows, Sampling, SegOut,
                  Segment, SessionExists, SessionInfo, SpecWindowExceeded, StageSpec, StepIn,
                  StepOut, StepTooLarge, UnknownSession, WTooDeep, kv_layout_id, make_numerics_id,
                  rebuild_start)
from ..toy import host_device
from .mimo_v2 import MimoEdges, MimoLayer, WeightError

ACT = torch.bfloat16
KERNEL_PIN = "mimo_v2-cpu-ref/1"
SAMPLER_PIN = "draw-splitmix64-f64/1"
_M64 = (1 << 64) - 1


# ── bf16 rows <-> the Rows boundary ────────────────────────────────────────────────────────────

def to_rows(t: torch.Tensor) -> Rows:
    """bf16 torch [n, W] -> Rows("bf16") (uint16 bit patterns)."""
    return Rows("bf16", t.contiguous().view(torch.int16).numpy().view(np.uint16).copy())


def from_rows(r: Rows) -> torch.Tensor:
    return torch.from_numpy(r.data.view(np.int16).copy()).view(torch.bfloat16)


def _bf16_finite(a: np.ndarray) -> bool:
    return not ((a & 0x7F80) == 0x7F80).any()


# ── the tail's token draw (part of this backend's numerics) ────────────────────────────────────

def _mix64(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & _M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


def uniform01(seed: int, p: int) -> float:
    """A counter RNG: uniform in [0, 1) from (seed, absolute position) only."""
    return (_mix64(_mix64(seed & _M64) ^ (p & _M64)) >> 11) * 2.0 ** -53


def draw(logits: np.ndarray, sampling: Sampling | None, p: int) -> int:
    """Token at absolute position p from one fp32 logits row (see the module docstring)."""
    if sampling is None or sampling.temperature == 0 or sampling.top_k == 1:
        return int(np.flatnonzero(logits == logits.max())[0])
    z = logits.astype(np.float64) / sampling.temperature
    order = np.lexsort((np.arange(z.size), -z))                   # by logit desc, then id asc
    k = min(sampling.top_k or z.size, z.size)
    zk = z[order[:k]]
    pr = np.exp(zk - zk[0])
    pr /= pr.sum()
    if sampling.top_p < 1.0:
        k = int(np.searchsorted(np.cumsum(pr), sampling.top_p)) + 1
        pr = pr[:min(k, pr.size)]
    c = np.cumsum(pr)
    i = int(np.searchsorted(c, uniform01(sampling.seed, p) * c[-1], side="right"))
    return int(order[min(i, c.size - 1)])


def cpu_device():
    """DeviceInfo of this host for the CPU reference: fp32 in hardware, bf16 / FP8 / MXFP4 stored
    and dequantized (emulated)."""
    return dataclasses.replace(host_device(), formats_hw=("fp32",),
                               formats_emulated=("bf16", "fp8_e4m3", "mxfp4"))


def _blas() -> str:
    m = re.search(r"BLAS_INFO=(\w+)", torch.__config__.show())
    return m.group(1) if m else "unknown"


# ── sessions ──────────────────────────────────────────────────────────────────────────────────

@dataclass
class _Sess:
    max_pos: int
    W: int
    C: int                                    # SWA ring slots per layer = w - 1 + W
    reserved: int
    k: dict = field(default_factory=dict)     # layer -> bf16 [max_pos | C, n_kv, qk]
    v: dict = field(default_factory=dict)     # layer -> bf16 [max_pos | C, n_kv, v]
    tag: dict = field(default_factory=dict)   # SWA layer -> int64 [C], position in each slot (-1 empty)
    length: int = 0
    committed: int = 0
    upto: int = 0                             # mirrored global rows [0, upto) imported
    imported: bool = False                    # imported, rebuild_from not called yet
    rebuild_hi: int | None = None             # F while a takeover rebuild runs
    swa_lo: int = 0                           # lowest SWA row that exists (a after a rebuild)


class CpuRuntime:
    """StageRuntime for layers [spec.lo, spec.hi) of a MiMo-V2 model described by `config` (the
    checkpoint's config.json as a dict)."""
    _strict = True

    def __init__(self, spec: StageSpec, config: dict, *, max_tokens_per_step: int = 256,
                 expert_cache: int = 64):
        try:
            self.cfg = cfg = MimoConfig.from_hf(config)
        except (UnsupportedConfig, KeyError, TypeError, ValueError) as e:
            raise BadRequest(f"config: {e}") from None
        if spec.hi > cfg.n_layers:
            raise BadRequest(f"span [{spec.lo}, {spec.hi}) outside the {cfg.n_layers}-layer model")
        if spec.numerics != "invariant":
            raise BadRequest("the CPU reference computes canonical numerics only (numerics='invariant')")
        if spec.kv_dtype not in ("", "bf16"):
            raise BadRequest(f"the CPU reference keeps KV in bf16, not {spec.kv_dtype!r}")
        if max_tokens_per_step < 1:
            raise BadRequest("max_tokens_per_step >= 1")
        self.spec, self.window = spec, cfg.window
        self.max_tokens_per_step, self.expert_cache = max_tokens_per_step, expert_cache
        self.H, self.V = cfg.hidden, cfg.vocab
        self.kinds = cfg.kinds[spec.lo:spec.hi]
        span = range(spec.lo, spec.hi)
        self._glob = tuple(i for i in span if cfg.layers[i].kind == "global")
        self._swa = tuple(i for i in span if cfg.layers[i].kind == "swa")
        self._tap_order = tuple(sorted(set(spec.taps)))
        self._taps = tuple(t for t in self._tap_order if spec.lo <= t < spec.hi)
        self._fc = spec.tap_mode == "fc_partial" and bool(self._taps)
        geo = {(cfg.layers[i].n_kv, cfg.layers[i].qk_dim, cfg.layers[i].v_dim) for i in self._glob}
        if len(geo) > 1:
            raise BadRequest(f"global layers of one span must share a KV geometry, got {geo}")
        self._geo = geo.pop() if geo else (0, 0, 0)
        self._layout = kv_layout_id(spec.model, self._glob, "bf16", self._geo)
        self._budget = spec.kv_budget_bytes or max(1, cpu_device().free_mem_bytes // 4)
        self._used = 0
        self._sess: dict[int, _Sess] = {}
        self._layers: dict[int, MimoLayer] | None = None
        self._edges: MimoEdges | None = None
        self._service: tuple = ()
        self._probe = None
        self._nid = self._numerics_id("")

    def _numerics_id(self, cid: str) -> str:
        return make_numerics_id(
            backend="cpu", arch_class=f"cpu-{platform.machine() or 'unknown'}", mode="invariant",
            kernel_pins=(KERNEL_PIN, f"torch-{torch.__version__}", f"blas-{_blas()}",
                         f"isa-{torch.backends.cpu.get_cpu_capability()}", SAMPLER_PIN),
            quant="mimo_v2:mxfp4-exact+fp8b-bf16rne", artifact_cid=cid)

    # ── weights ─────────────────────────────────────────────────────────────────────────────
    def _plan(self) -> dict:
        sp, cfg = self.spec, self.cfg
        plan = {}
        for i in range(sp.lo, sp.hi):
            plan.update(layer_plan(cfg, i))
        plan.update(edge_plan(cfg, head=sp.is_head, tail=sp.is_tail,
                              n_fc_taps=len(self._tap_order) if self._fc else 0))
        return plan

    def load(self, src) -> LoadReport:
        """Materialize exactly the span's tensors. Refused (LoadRefused, nothing half-loaded): a
        missing tensor, a wrong dtype/shape, a down-cast router bias, and any tensor under the
        span's layer prefixes the plan does not know (a checkpoint variant this adapter would
        silently misread)."""
        sp, plan = self.spec, self._plan()
        have = set(src.names())
        missing = sorted(set(plan) - have)
        if missing:
            raise LoadRefused(f"weight source lacks {missing[0]!r} ({len(missing)} missing): "
                              "refusing a partially random span")
        pre = tuple(layer_prefix(i) for i in range(sp.lo, sp.hi))
        unknown = sorted(n for n in have if n.startswith(pre) and n not in plan)
        if unknown:
            raise LoadRefused(f"{unknown[0]!r} ({len(unknown)} unknown) is not part of the MiMo-V2 "
                              "layer format this adapter implements")
        try:
            layers = {i: MimoLayer.load(self.cfg, i, src, self.expert_cache) for i in range(sp.lo, sp.hi)}
            edges = MimoEdges.load(self.cfg, src, head=sp.is_head, tail=sp.is_tail,
                                   fc_taps=self._taps if self._fc else (), fc_order=self._tap_order)
        except WeightError as e:
            raise LoadRefused(str(e)) from None
        self._layers, self._edges = layers, edges
        self._nid = self._numerics_id(src.artifact_cid)
        nbytes = sum(int(np.prod(shape)) * (4 if dt in ("F32",) else 2 if dt == "BF16" else 1)
                     for dt, shape in plan.values())
        return LoadReport(n_tensors=len(plan), n_bytes=nbytes, artifact_cid=src.artifact_cid)

    def _loaded(self):
        if self._layers is None:
            raise BadRequest("load() first")

    # ── describe / warmup ───────────────────────────────────────────────────────────────────
    def _kv_bytes(self, layers) -> int:
        return sum(2 * self.cfg.layers[i].n_kv * (self.cfg.layers[i].qk_dim + self.cfg.layers[i].v_dim)
                   for i in layers)

    def describe(self) -> Capability:
        sp, cfg = self.spec, self.cfg
        lw = tuple(MimoLayer.resident_bytes(cfg, i) for i in range(sp.lo, sp.hi))
        big = max(max(s.n_q * s.qk_dim + s.n_kv * (s.qk_dim + s.v_dim), cfg.intermediate) * cfg.hidden
                  for s in cfg.layers[sp.lo:sp.hi])
        return Capability(
            device=cpu_device(), model=sp.model, lo=sp.lo, hi=sp.hi, is_head=sp.is_head,
            is_tail=sp.is_tail, layer_kinds=self.kinds,
            swa_window=self.window if self._swa else 0, hidden_size=self.H, vocab_size=self.V,
            wire_dtype="bf16", codecs_in=ID_DTYPES if sp.is_head else ("bf16",), logits_dtype="f32",
            kv_dtype="bf16", kv_layout_id=self._layout,
            global_kv_bytes_per_pos=self._kv_bytes(self._glob),
            swa_kv_bytes_per_pos=self._kv_bytes(self._swa), kv_page_positions=1,
            kv_free_bytes=self._budget - self._used, layer_weight_bytes=lw,
            edge_weight_bytes=MimoEdges.resident_bytes(cfg, head=sp.is_head, tail=sp.is_tail,
                                                       n_fc_held=len(self._taps) if self._fc else 0),
            load_peak_extra_bytes=8 * big, max_sessions=sp.max_sessions, max_ctx=sp.max_ctx,
            spec_window_max=sp.spec_window, max_tokens_per_step=self.max_tokens_per_step,
            tap_layers_held=self._taps, tap_mode=sp.tap_mode, numerics_id=self._nid,
            numerics_mode="invariant", chunk_invariant=True, service_ms=self._service,
            self_probe_ok=self._probe, open_sessions=len(self._sess))

    def warmup(self, buckets=(1, 8, 32), contexts=(0,), sessions=(1,)) -> Capability:
        """Time step() for every shape (s sessions x b tokens at context c) on private scratch
        sessions (negative sids, never charged to the KV pool, removed afterwards) after an
        untimed prefix of c rows each; self-probe: a repeated run must be byte-equal."""
        self._loaded()
        service, probe = [], True
        for n_s in sessions:
            for ctx in contexts:
                for b in buckets:
                    b, n_s, ctx = int(b), int(n_s), int(ctx)
                    if b < 1 or n_s * b > self.max_tokens_per_step or ctx + b > self.spec.max_ctx:
                        continue
                    runs, best = [], float("inf")
                    for _ in range(2):
                        sids = [-1 - i for i in range(n_s)]
                        for sid in sids:
                            s = self._new_sess(ctx + b, 1, 0)
                            s.committed = ctx + b
                            self._sess[sid] = s
                        try:
                            for sid in sids:
                                for p0 in range(0, ctx, self.max_tokens_per_step):
                                    n0 = min(self.max_tokens_per_step, ctx - p0)
                                    self.step(StepIn((self._probe_seg(sid, p0, n0),)))
                            t0 = time.perf_counter()
                            outs = self.step(StepIn(tuple(self._probe_seg(sid, ctx, b) for sid in sids))).outs
                            best = min(best, time.perf_counter() - t0)
                            runs.append(outs[0])
                        finally:
                            for sid in sids:
                                self._sess.pop(sid, None)
                    probe &= runs[0].hidden == runs[1].hidden and runs[0].logits == runs[1].logits
                    service.append((n_s, b, ctx, best * 1000.0))
        self._service, self._probe = tuple(service), bool(probe)
        return self.describe()

    def _probe_seg(self, sid: int, start: int, n: int) -> Segment:
        if self.spec.is_head:
            x = Rows("i32", (np.arange(start, start + n) % self.V).astype(np.int32))
        else:
            g = torch.Generator().manual_seed(start)
            x = to_rows(torch.randn(n, self.H, generator=g).to(ACT))
        return Segment(sid, start, n, x, kind="prefill", logits="last" if self.spec.is_tail else "none")

    # ── sessions ────────────────────────────────────────────────────────────────────────────
    def _new_sess(self, max_pos: int, W: int, reserved: int) -> _Sess:
        C = self.window - 1 + W
        s = _Sess(max_pos=max_pos, W=W, C=C, reserved=reserved)
        for i in self._glob + self._swa:
            L = self.cfg.layers[i]
            rows = max_pos if L.kind == "global" else C
            s.k[i] = torch.zeros(rows, L.n_kv, L.qk_dim, dtype=ACT)
            s.v[i] = torch.zeros(rows, L.n_kv, L.v_dim, dtype=ACT)
            if L.kind == "swa":
                s.tag[i] = np.full(C, -1, dtype=np.int64)
        return s

    def open(self, sid: int, max_pos: int, spec_window: int) -> None:
        self._loaded()
        if sid in self._sess:
            raise SessionExists(f"session {sid} is already open", sid=sid)
        if max_pos < 1 or spec_window < 1:
            raise BadRequest("max_pos and spec_window must be >= 1", sid=sid)
        if spec_window > self.spec.spec_window:
            raise WTooDeep(f"spec_window {spec_window} > supported {self.spec.spec_window}", sid=sid)
        if max_pos > self.spec.max_ctx:
            raise KVFull(f"max_pos {max_pos} > max_ctx {self.spec.max_ctx}", sid=sid)
        if len(self._sess) >= self.spec.max_sessions:
            raise KVFull(f"session table full ({self.spec.max_sessions})", sid=sid)
        need = self.describe().session_kv_bytes(max_pos, spec_window)
        if need > self._budget - self._used:
            raise KVFull(f"session needs {need} B of KV, {self._budget - self._used} B free", sid=sid)
        self._sess[sid] = self._new_sess(max_pos, spec_window, need)
        self._used += need

    def close(self, sid: int) -> None:
        s = self._sess.pop(sid, None)
        if s is not None:
            self._used -= s.reserved

    def _get(self, sid: int) -> _Sess:
        try:
            return self._sess[sid]
        except KeyError:
            raise UnknownSession(f"session {sid} is not open", sid=sid) from None

    def length(self, sid: int) -> int:
        return self._get(sid).length

    def info(self, sid: int) -> SessionInfo:
        s = self._get(sid)
        return SessionInfo(sid=sid, length=s.length, committed=s.committed, max_pos=s.max_pos,
                           spec_window=s.W, kv_bytes=s.reserved, imported_upto=s.upto,
                           rebuild_hi=s.rebuild_hi)

    def truncate(self, sid: int, pos: int) -> None:
        s = self._get(sid)
        if pos > s.length:
            raise PositionGap(f"truncate to {pos} past length {s.length}", sid=sid)
        if pos < s.committed:
            raise RollbackPastCommit(f"truncate to {pos} below the committed frontier {s.committed}", sid=sid)
        s.length = pos

    def commit(self, sid: int, pos: int) -> None:
        s = self._get(sid)
        if pos > s.max_pos:
            raise BadRequest(f"commit {pos} past max_pos {s.max_pos}", sid=sid)
        s.committed = max(s.committed, pos)

    # ── mirroring / takeover ────────────────────────────────────────────────────────────────
    def _kv_row_width(self) -> int:
        n_kv, qk, v = self._geo
        return n_kv * (qk + v)

    def export_state(self, sid: int, lo_pos: int, hi_pos: int, kinds=("global",)) -> KVBlob:
        s = self._get(sid)
        if tuple(kinds) != ("global",):
            raise BadRequest(f"only global-attention KV is mirrored, not {kinds}", sid=sid)
        if s.imported or s.rebuild_hi is not None:
            raise BadRequest("session is mid-takeover: nothing final to export yet", sid=sid)
        if not 0 <= lo_pos <= hi_pos:
            raise BadRequest(f"[{lo_pos}, {hi_pos}) is not a range", sid=sid)
        if hi_pos > min(s.committed, s.length):
            raise BadRequest(f"export to {hi_pos} passes the frontier min(committed {s.committed}, "
                             f"length {s.length}): speculative rows are never mirrored", sid=sid)
        n = hi_pos - lo_pos
        rows = tuple(to_rows(torch.cat([s.k[i][lo_pos:hi_pos].reshape(n, -1),
                                        s.v[i][lo_pos:hi_pos].reshape(n, -1)], dim=1))
                     for i in self._glob)
        return KVBlob(lo_pos, hi_pos, self._glob, rows, self._layout, self._nid)

    def import_state(self, sid: int, blob: KVBlob, upto: int) -> None:
        s = self._get(sid)
        if s.length != 0 or s.rebuild_hi is not None:
            raise BadRequest("import_state needs a fresh session (length 0, no rebuild running)", sid=sid)
        if blob.layout_id != self._layout or blob.layers != self._glob:
            raise BadRequest("KV layout mismatch: the blob is from another span/model/dtype", sid=sid)
        if blob.lo_pos != s.upto:
            raise BadRequest(f"imports must be contiguous: have [0, {s.upto}), blob starts at {blob.lo_pos}", sid=sid)
        if not blob.lo_pos <= upto <= blob.hi_pos or upto > s.max_pos:
            raise BadRequest(f"upto {upto} outside the blob [{blob.lo_pos}, {blob.hi_pos}) / max_pos", sid=sid)
        n_kv, qk, vd = self._geo
        for r in blob.rows:
            if r.dtype != "bf16" or r.data.ndim != 2 or r.data.shape[1] != self._kv_row_width() \
                    or not _bf16_finite(r.data):
                raise BadRequest(f"blob rows malformed for this layout: {r!r}", sid=sid)
        k = upto - blob.lo_pos
        for i, r in zip(blob.layers, blob.rows):
            t = from_rows(r)[:k]
            s.k[i][blob.lo_pos:upto] = t[:, :n_kv * qk].reshape(k, n_kv, qk)
            s.v[i][blob.lo_pos:upto] = t[:, n_kv * qk:].reshape(k, n_kv, vd)
        s.upto = upto
        s.imported = True

    def rebuild_from(self, sid: int, F: int, exact_from: int | None = None,
                     floor: int | None = None) -> int:
        s = self._get(sid)
        if s.length != 0 or s.rebuild_hi is not None:
            raise BadRequest("rebuild_from needs a fresh or freshly imported session", sid=sid)
        if not s.upto <= F <= s.max_pos:
            raise BadRequest(f"F={F} must satisfy imported upto {s.upto} <= F <= max_pos {s.max_pos}", sid=sid)
        floor = F if floor is None else floor
        if not 0 <= floor <= F or F - floor > s.W:
            raise BadRequest(f"floor {floor} must satisfy F - spec_window <= floor <= F={F}", sid=sid)
        if exact_from is not None and not 0 <= exact_from <= F:
            raise BadRequest(f"exact_from {exact_from} must lie in [0, F={F}]", sid=sid)
        try:
            a = rebuild_start(self.kinds, self.window, F, s.upto,
                              floor if exact_from is None else min(exact_from, floor))
        except BadRequest as e:
            raise BadRequest(str(e), sid=sid) from None
        s.length, s.swa_lo, s.imported = a, a, False
        s.committed = max(s.committed, floor)
        s.rebuild_hi = F if a < F else None
        return a

    # ── head / tail ─────────────────────────────────────────────────────────────────────────
    def embed(self, ids: Rows) -> Rows:
        self._loaded()
        if not self.spec.is_head:
            raise BadRequest("embed() is head-only")
        self._check_ids(ids, ids.n if isinstance(ids, Rows) else -1, None)
        return to_rows(self._edges.embed(torch.from_numpy(ids.data.astype(np.int64)), ACT))

    def project(self, hidden: Rows, *, tokens: bool = False, start_pos: int = 0,
                sampling: Sampling | None = None) -> Rows:
        self._loaded()
        if not self.spec.is_tail:
            raise BadRequest("project() is tail-only")
        self._check_hidden(hidden, hidden.n if isinstance(hidden, Rows) else -1, None, "hidden")
        x = from_rows(hidden)
        lg = torch.cat([self._edges.logits(x[r:r + 1], ACT) for r in range(x.shape[0])]).numpy()
        if not tokens:
            return Rows("f32", lg.astype(np.float32))
        return Rows("i32", np.array([draw(lg[i], sampling, start_pos + i) for i in range(lg.shape[0])],
                                    dtype=np.int32))

    # ── compute ─────────────────────────────────────────────────────────────────────────────
    def step(self, x: StepIn) -> StepOut:
        self._loaded()
        segs = tuple(x.segs)
        seen = set()
        for seg in segs:
            if seg.sid in seen:
                raise BadRequest(f"two segments of session {seg.sid} in one step", sid=seg.sid)
            seen.add(seg.sid)
            self._validate(seg)
        total = sum(seg.n for seg in segs)
        if total > self.max_tokens_per_step:
            raise StepTooLarge(f"{total} tokens > max_tokens_per_step {self.max_tokens_per_step}")
        return StepOut(tuple(self._run(self._sess[seg.sid], seg) for seg in segs))

    def _validate(self, seg: Segment) -> None:
        sid = seg.sid
        s = self._get(sid)
        bad = lambda msg: BadRequest(msg, sid=sid)   # noqa: E731
        if seg.kind not in SEG_KINDS:
            raise bad(f"segment kind {seg.kind!r} not in {SEG_KINDS}")
        if not isinstance(seg.n, int) or seg.n < 1:
            raise bad(f"segment n must be >= 1, got {seg.n!r}")
        if seg.kind == "decode" and seg.n != 1:
            raise bad("a decode segment is exactly one row")
        if seg.logits not in LOGITS_MODES or seg.tokens not in LOGITS_MODES:
            raise bad(f"logits/tokens modes must be in {LOGITS_MODES}")
        if (seg.logits != "none" or seg.tokens != "none") and not self.spec.is_tail:
            raise bad("logits/tokens requested from a non-tail stage")
        if seg.kind == "rebuild" and (seg.logits != "none" or seg.tokens != "none"):
            raise bad("rebuild segments return no logits or tokens")
        if seg.sampling is not None and not isinstance(seg.sampling, Sampling):
            raise bad("sampling must be a Sampling")
        start, end = seg.start_pos, seg.start_pos + seg.n
        if s.imported:
            raise bad("imported session: call rebuild_from before stepping")
        if s.rebuild_hi is not None:
            if seg.kind != "rebuild":
                raise bad(f"session is rebuilding to {s.rebuild_hi}: only rebuild segments")
            if start != s.length:
                raise bad(f"rebuild segments continue at length {s.length}, not {start}")
            if end > s.rebuild_hi:
                raise bad(f"rebuild segment ends at {end}, past F={s.rebuild_hi}")
        else:
            if seg.kind == "rebuild":
                raise bad("rebuild segment without rebuild_from")
            if start > s.length:
                raise PositionGap(f"segment at {start} but session {sid} has length {s.length}", sid=sid)
            if start < s.length and start < s.committed:
                raise RollbackPastCommit(f"segment at {start} rewinds below committed {s.committed}", sid=sid)
        if end > s.max_pos:
            raise bad(f"segment ends at {end}, past max_pos {s.max_pos}")
        if end > s.committed + s.W:
            raise SpecWindowExceeded(f"segment ends at {end} > committed {s.committed} + W {s.W}", sid=sid)
        if self.spec.is_head:
            self._check_ids(seg.x, seg.n, sid)
        else:
            self._check_hidden(seg.x, seg.n, sid, "hidden input")
        if seg.tap_in is not None:
            if self.spec.tap_mode != "fc_partial" or not seg.taps:
                raise bad("tap_in is only accepted with taps=True in fc_partial mode")
            self._check_hidden(seg.tap_in, seg.n, sid, "tap_in")

    def _check_ids(self, x, n: int, sid) -> None:
        if not isinstance(x, Rows) or x.dtype not in ID_DTYPES or x.data.ndim != 1 or x.n != n:
            raise BadRequest(f"token ids must be Rows i32/i64 [{n}], got {x!r}", sid=sid)
        if (x.data < 0).any() or (x.data >= self.V).any():
            raise BadRequest("token id outside the vocabulary", sid=sid)

    def _check_hidden(self, x, n: int, sid, what: str) -> None:
        if not isinstance(x, Rows) or x.dtype != "bf16" or x.data.shape != (n, self.H):
            raise BadRequest(f"{what} must be Rows bf16 [{n}, {self.H}], got {x!r}", sid=sid)
        if not _bf16_finite(x.data):
            raise BadRequest(f"{what} holds a non-finite value", sid=sid)

    def _run(self, s: _Sess, seg: Segment) -> SegOut:
        start, n = seg.start_pos, seg.n
        if start < s.length:
            s.length = start                                  # implicit truncate (validated)
        rebuilding = s.rebuild_hi is not None
        if self.spec.is_head:
            x = self._edges.embed(torch.from_numpy(seg.x.data.astype(np.int64)), ACT)
        else:
            x = from_rows(seg.x)
        fc = self.spec.tap_mode == "fc_partial" and seg.taps
        acc = None
        if fc:
            acc = from_rows(seg.tap_in).to(torch.float32) if seg.tap_in is not None \
                else torch.zeros(n, self.H, dtype=torch.float32)
        taps = {}
        for li in range(self.spec.lo, self.spec.hi):
            L = self._layers[li]
            q, kf, vf = L.project_qkv(x, start, ACT)
            attn = torch.cat([L.attend(q[r:r + 1], *self._visible(s, li, start + r, start, kf, vf, rebuilding), ACT)
                              for r in range(n)])
            out = L.finish(x, attn, ACT)
            self._write_kv(s, li, start, kf, vf, rebuilding)
            if seg.taps and li in self._taps:
                if fc:
                    acc = torch.cat([self._edges.fc_add(acc[r:r + 1], li, out[r:r + 1]) for r in range(n)])
                else:
                    taps[li] = to_rows(out)
            x = out
        s.length = start + n
        if rebuilding and s.length >= s.rebuild_hi:
            s.rebuild_hi = None
        logits = tokens = None
        if seg.logits != "none" or seg.tokens != "none":
            rows = {"all": range(n), "last": range(n - 1, n), "none": range(0)}
            want = sorted(set(rows[seg.logits]) | set(rows[seg.tokens]))
            lg = {r: self._edges.logits(x[r:r + 1], ACT)[0].numpy() for r in want}
            if seg.logits != "none":
                logits = Rows("f32", np.stack([lg[r] for r in rows[seg.logits]]).astype(np.float32))
            if seg.tokens != "none":
                tokens = Rows("i32", np.array([draw(lg[r], seg.sampling, start + r) for r in rows[seg.tokens]],
                                              dtype=np.int32))
        tap_acc = None
        if fc:
            tap_acc = to_rows(acc.to(ACT))
        return SegOut(seg.sid, to_rows(x), logits, taps, tap_acc, tokens)

    def _visible(self, s: _Sess, li: int, p: int, start: int, kf, vf, rebuilding: bool):
        """K, V rows [n, n_kv, d] of the keys query p sees, in position order: cached rows below
        the segment (below `upto` too while rebuilding a global layer: mirrored rows are read,
        never recomputed), the segment's fresh rows above."""
        L = self.cfg.layers[li]
        if L.kind == "global":
            cut = min(p + 1, max(start, s.upto) if rebuilding else start)
            return (torch.cat([s.k[li][:cut], kf[cut - start:p - start + 1]]),
                    torch.cat([s.v[li][:cut], vf[cut - start:p - start + 1]]))
        lo = max(s.swa_lo, p - self.window + 1, 0)
        qs = np.arange(lo, min(start, p + 1))
        slots = torch.from_numpy(qs % s.C)
        if self._strict and not (s.tag[li][qs % s.C] == qs).all():
            raise AssertionError(f"SWA ring lost a row it needs (layer {li}, position {p})")
        f0 = max(lo, start) - start
        return (torch.cat([s.k[li][slots], kf[f0:p - start + 1]]),
                torch.cat([s.v[li][slots], vf[f0:p - start + 1]]))

    def _write_kv(self, s: _Sess, li: int, start: int, kf, vf, rebuilding: bool) -> None:
        end = start + kf.shape[0]
        if self.cfg.layers[li].kind == "global":
            w0 = min(end, max(start, s.upto)) if rebuilding else start
            s.k[li][w0:end] = kf[w0 - start:]
            s.v[li][w0:end] = vf[w0 - start:]
            return
        for p in range(max(start, end - s.C), end):
            c = p % s.C
            s.k[li][c], s.v[li][c], s.tag[li][c] = kf[p - start], vf[p - start], p
