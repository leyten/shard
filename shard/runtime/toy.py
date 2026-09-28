"""ToyRuntime: the CPU reference StageRuntime over a tiny deterministic integer model.

It implements the whole protocol of shard/runtime/api.py (mixed segment batches, implicit and
explicit rollback, the commit frontier, SWA with window-1+W slack, global-KV export/import,
the takeover rebuild, raw and fc_partial taps, on-device token draws, embed/project,
describe/warmup) so the conformance suite (tests/runtime_conformance.py), the fake mesh and
the node server can be built and attacked before any GPU exists.

The model. Layer kinds come from a pattern string ("S" = sliding window w, "G" = global).
Every value is an integer mod P = 1_000_003 carried as int32 rows of width H. At layer l and
position p, the KV row is the layer's input x_p, and

    att_p[h] = sum_{q in window(p)} kv_q[h] * (q % 7 + 1) * 31^(p-q)      (mod P)
    out_p[h] = a_l[h]*x_p[h] + b_l[h]*x_p[h+1] + att_p[h+l] + c_l[h] + term  (mod P, h mod H)

window(p) = [0, p] for G, [p-w+1, p] for S. Integer arithmetic makes the bytes exact, and a
wrong row anywhere in a window changes the result (the 31^(p-q) weights see order and
distance). Two modes:
  * "chunk_invariant": term = 0. Bytes depend only on the inputs, so any schedule of the same
    rows gives the same bytes and a takeover rebuild is bit-equal to the fault-free state.
  * "adversarial": term = f(start_pos, n, row, layer) of the row's segment. Bytes still
    depend on nothing but the input, the causal prefix and the segment SHAPE (batch
    invariant, row independent, replayable) but chunking changes them, like real float
    kernels. What holds here holds without chunk invariance.
The tail's logits are lm_head . (out * norm) mod P; tokens are drawn by `toy_draw` (greedy =
first maximum; sampled = a top-k pick by a counter hash of (seed, position): integer logits
carry no probabilities, so temperature > 0 only switches sampling on and top_p is unused).

Storage mirrors a real runtime on purpose: global rows in a per-session [max_pos, H] array,
SWA rows in a ring of exactly (w - 1 + spec_window) slots per layer, so the classic "ring of w
slots" and off-by-one bugs are representable (tests/test_runtime_conformance.py's mutants).
`_strict` makes the correct toy assert its ring invariant; mutants modelling silent bugs turn
it off so the suite must catch them by their bytes.

`ToyReference` is an independent pure-Python definition of the same model driven through the
same calls (dicts per layer, no ring, no reclamation, no validation): the oracle the suite
compares every output and every exported KV byte against, including across takeovers.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass, field

import numpy as np

from .api import (ID_DTYPES, LOGITS_MODES, SEG_KINDS, BadRequest, Capability, DeviceInfo,
                  KVBlob, KVFull, LoadRefused, LoadReport, PositionGap, RollbackPastCommit, Rows,
                  Sampling, SegOut, Segment, SessionExists, SessionInfo, SpecWindowExceeded,
                  StageSpec, StepIn, StepOut, StepTooLarge, UnknownSession, WTooDeep,
                  kv_layout_id, make_numerics_id, rebuild_start)

P = 1_000_003          # the toy's field; every hidden/KV/logit value lies in [0, P)
_BASE = 31             # relative-position base of the attention weights
MODES = ("chunk_invariant", "adversarial")


def _shape_term(start: int, n: int, r: int, layer: int) -> int:
    """Adversarial mode: a per-row term that depends on the segment shape only."""
    return 1 + (start * 1_000_033 + n * 7_919 + r * 104_729 + layer * 31) % 997


def toy_draw(logits_row, sampling: Sampling | None, p: int) -> int:
    """The toy's token draw at absolute position p (part of the model's definition)."""
    row = [int(v) for v in logits_row]
    if sampling is None or sampling.temperature == 0:
        return max(range(len(row)), key=lambda i: (row[i], -i))
    order = sorted(range(len(row)), key=lambda i: (-row[i], i))
    k = min(sampling.top_k or len(row), len(row))
    r = int.from_bytes(hashlib.sha256(f"{sampling.seed}/{p}".encode()).digest()[:8], "little")
    return order[r % k]


def _geometry(hidden: int) -> tuple[int, int, int]:
    return (1, hidden, 0)          # one "KV head" whose K row is the layer input; no separate V


class ToyWeights:
    """A deterministic WeightSource for the toy model (seeded integers in [0, P))."""

    def __init__(self, pattern: str, hidden: int = 4, vocab: int = 16, taps=(), seed: int = 0,
                 _drop=()):
        self.pattern, self.hidden, self.vocab = pattern, hidden, vocab
        self.taps, self.seed = tuple(sorted(set(taps))), seed
        rng = np.random.default_rng(seed)
        t = {}
        for li in range(len(pattern)):
            for k in "abc":
                t[f"layers.{li}.{k}"] = rng.integers(1, P, size=hidden, dtype=np.int64)
        t["embed"] = rng.integers(0, P, size=(vocab, hidden), dtype=np.int64)
        t["norm"] = rng.integers(1, P, size=hidden, dtype=np.int64)
        t["lm_head"] = rng.integers(0, P, size=(vocab, hidden), dtype=np.int64)
        for tl in self.taps:
            t[f"tap_fc.{tl}"] = rng.integers(0, P, size=(hidden, hidden), dtype=np.int64)
        self._t = {k: v for k, v in t.items() if k not in set(_drop)}
        self._drop = tuple(_drop)
        ident = json.dumps([pattern, hidden, vocab, list(self.taps), seed])
        self.artifact_cid = "toy-" + hashlib.sha256(ident.encode()).hexdigest()[:16]

    def names(self):
        return sorted(self._t)

    def read(self, name: str) -> np.ndarray:
        return self._t[name].copy()

    def without(self, name: str) -> "ToyWeights":
        """The same source with one tensor missing (a corrupt/partial fetch)."""
        return ToyWeights(self.pattern, self.hidden, self.vocab, self.taps, self.seed,
                          _drop=self._drop + (name,))


def host_device() -> DeviceInfo:
    """DeviceInfo of the CPU this process runs on (the toy's device)."""
    m = platform.machine() or "unknown"
    total = free = 0
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = page * os.sysconf("SC_PHYS_PAGES")
        free = page * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        free = total
    return DeviceInfo(backend="cpu", device_name=platform.processor() or m, arch_class=f"cpu-{m}",
                      host_arch=m, n_devices=1, total_mem_bytes=int(total), free_mem_bytes=int(free),
                      unified_memory=True, mem_bw_bytes_s=0.0, mem_bw_source="unknown",
                      formats_hw=("fp32", "int32"), formats_emulated=("bf16", "fp16"))


@dataclass
class _Sess:
    max_pos: int
    W: int
    C: int                                   # SWA ring slots per layer
    reserved: int
    glob: dict = field(default_factory=dict)  # layer -> [max_pos, H] int64
    swa: dict = field(default_factory=dict)   # layer -> [C, H] int64 ring
    tag: dict = field(default_factory=dict)   # layer -> [C] position held by each slot (-1 = empty)
    length: int = 0
    committed: int = 0
    upto: int = 0                            # mirrored global rows [0, upto) imported
    imported: bool = False                   # imported, rebuild_from not yet called
    rebuild_hi: int | None = None            # F while a takeover rebuild is in progress
    swa_lo: int = 0                          # lowest materialized SWA row (a after a rebuild)


@dataclass(frozen=True)
class _StepCtx:
    total_tokens: int
    n_segs: int


class ToyRuntime:
    """CPU reference StageRuntime. `pattern` is the WHOLE model's layer kinds (like a config);
    the runtime serves spec.lo..spec.hi of it."""
    _strict = True

    def __init__(self, spec: StageSpec, *, pattern: str, window: int, hidden: int = 4,
                 vocab: int = 16, mode: str = "chunk_invariant", max_tokens_per_step: int = 64):
        if not pattern or set(pattern) - set("SG"):
            raise BadRequest(f"pattern must be a non-empty string over 'S'/'G', got {pattern!r}")
        if spec.hi > len(pattern):
            raise BadRequest(f"span [{spec.lo}, {spec.hi}) outside the {len(pattern)}-layer model")
        if window < 1 or hidden < 2 or vocab < 2 or max_tokens_per_step < 1:
            raise BadRequest("window >= 1, hidden >= 2, vocab >= 2, max_tokens_per_step >= 1")
        if mode not in MODES:
            raise BadRequest(f"mode {mode!r} not in {MODES}")
        if spec.numerics != "invariant":
            raise BadRequest("the toy's integer numerics are invariant; numerics='fast' is not offered")
        if spec.kv_dtype not in ("", "i32"):
            raise BadRequest(f"the toy keeps KV as i32, not {spec.kv_dtype!r}")
        self.spec, self.pattern, self.window = spec, pattern, window
        self.H, self.V, self.mode = hidden, vocab, mode
        self.max_tokens_per_step = max_tokens_per_step
        self.kinds_all = tuple("swa" if c == "S" else "global" for c in pattern)
        self.kinds = self.kinds_all[spec.lo:spec.hi]
        rng = range(spec.lo, spec.hi)
        self._glob_layers = tuple(li for li in rng if self.kinds_all[li] == "global")
        self._swa_layers = tuple(li for li in rng if self.kinds_all[li] == "swa")
        self._tap_layers = tuple(sorted(t for t in set(spec.taps) if spec.lo <= t < spec.hi))
        self._budget = spec.kv_budget_bytes or (1 << 40)
        self._used = 0
        self._sess: dict[int, _Sess] = {}
        self._w = None
        self._lw = {}
        self._service: tuple = ()
        self._probe = None
        self._layout = kv_layout_id(spec.model, self._glob_layers, "i32", _geometry(hidden))
        self._nid = self._numerics_id("")
        n = spec.max_ctx + 1
        self._pow = np.ones(n, dtype=np.int64)
        for k in range(1, n):
            self._pow[k] = self._pow[k - 1] * _BASE % P
        self._qw = np.arange(n, dtype=np.int64) % 7 + 1

    def _numerics_id(self, cid: str) -> str:
        return make_numerics_id(backend="cpu", arch_class="toy-int", mode=f"invariant/{self.mode}",
                                kernel_pins=("toy-v1",), quant="int-mod-p", artifact_cid=cid)

    def _needed(self) -> list[str]:
        lo, hi = self.spec.lo, self.spec.hi
        need = [f"layers.{li}.{k}" for li in range(lo, hi) for k in "abc"]
        if self.spec.is_head:
            need.append("embed")
        if self.spec.is_tail:
            need += ["norm", "lm_head"]
        if self.spec.tap_mode == "fc_partial":
            need += [f"tap_fc.{t}" for t in self._tap_layers]
        return need

    def _shape_of(self, name: str) -> tuple:
        H, V = self.H, self.V
        return {"embed": (V, H), "lm_head": (V, H), "norm": (H,)}.get(
            name, (H, H) if name.startswith("tap_fc.") else (H,))

    # ── lifecycle ──────────────────────────────────────────────────────────────────────────
    def load(self, src) -> LoadReport:
        need = self._needed()
        have = set(src.names())
        missing = [nm for nm in need if nm not in have]
        if missing:
            raise LoadRefused(f"weight source lacks {missing[0]!r} ({len(missing)} missing): "
                              "refusing a partially random span")
        w, nbytes = {}, 0
        for nm in need:
            a = np.asarray(src.read(nm))
            shape = self._shape_of(nm)
            if a.shape != shape or a.dtype.kind not in "iu" or (a < 0).any() or (a >= P).any():
                raise LoadRefused(f"tensor {nm!r} malformed: shape {a.shape} dtype {a.dtype}, want {shape}")
            w[nm] = a.astype(np.int64)
            nbytes += a.size * 4
        self._w = w
        self._lw = {li: (w[f"layers.{li}.a"], w[f"layers.{li}.b"], w[f"layers.{li}.c"])
                    for li in range(self.spec.lo, self.spec.hi)}
        self._nid = self._numerics_id(src.artifact_cid)
        return LoadReport(n_tensors=len(need), n_bytes=nbytes, artifact_cid=src.artifact_cid)

    def _loaded(self):
        if self._w is None:
            raise BadRequest("load() first")

    def describe(self) -> Capability:
        sp = self.spec
        return Capability(
            device=host_device(), model=sp.model, lo=sp.lo, hi=sp.hi, is_head=sp.is_head,
            is_tail=sp.is_tail, layer_kinds=self.kinds,
            swa_window=self.window if self._swa_layers else 0, hidden_size=self.H, vocab_size=self.V,
            wire_dtype="i32", codecs_in=ID_DTYPES if sp.is_head else ("i32",), logits_dtype="i32",
            kv_dtype="i32", kv_layout_id=self._layout,
            global_kv_bytes_per_pos=len(self._glob_layers) * self.H * 4,
            swa_kv_bytes_per_pos=len(self._swa_layers) * self.H * 4, kv_page_positions=1,
            kv_free_bytes=self._budget - self._used,
            weight_bytes=sum(4 * int(np.prod(self._shape_of(nm))) for nm in self._needed()),
            load_peak_extra_bytes=0, max_sessions=sp.max_sessions,
            max_ctx=sp.max_ctx, spec_window_max=sp.spec_window,
            max_tokens_per_step=self.max_tokens_per_step, tap_layers_held=self._tap_layers,
            tap_mode=sp.tap_mode, numerics_id=self._nid, numerics_mode="invariant",
            chunk_invariant=self.mode == "chunk_invariant", service_ms=self._service,
            self_probe_ok=self._probe, open_sessions=len(self._sess))

    def warmup(self, buckets=(1, 8, 32), contexts=(0,)) -> Capability:
        """Time one prefill step per (bucket, context) on scratch sessions (never registered,
        never charged to the KV pool) after an untimed prefix of `context` rows, and
        self-probe: the repeat run must be byte-equal."""
        self._loaded()
        service, probe = [], True
        for ctx in contexts:
            for b in buckets:
                b = min(int(b), self.max_tokens_per_step)
                if b < 1 or ctx + b > self.spec.max_ctx:
                    continue
                runs, best = [], float("inf")
                for _ in range(2):
                    s = self._new_sess(ctx + b, 1, 0)
                    s.committed = ctx + b
                    for p0 in range(0, ctx, self.max_tokens_per_step):
                        n0 = min(self.max_tokens_per_step, ctx - p0)
                        self._run(s, self._probe_seg(p0, n0), _StepCtx(n0, 1))
                    seg = self._probe_seg(ctx, b)
                    t0 = time.perf_counter()
                    runs.append(self._run(s, seg, _StepCtx(b, 1)))
                    best = min(best, time.perf_counter() - t0)
                probe &= runs[0].hidden == runs[1].hidden and runs[0].logits == runs[1].logits
                service.append((b, int(ctx), best * 1000.0))
        self._service, self._probe = tuple(service), bool(probe)
        return self.describe()

    def _probe_seg(self, start: int, n: int) -> Segment:
        if self.spec.is_head:
            x = Rows("i32", (np.arange(start, start + n) % self.V).astype(np.int32))
        else:
            x = Rows("i32", (np.arange(start * self.H, (start + n) * self.H) % P).astype(np.int32).reshape(n, self.H))
        return Segment(-1, start, n, x, kind="prefill", logits="last" if self.spec.is_tail else "none")

    # ── sessions ───────────────────────────────────────────────────────────────────────────
    def _swa_capacity(self, W: int) -> int:
        return self.window - 1 + W

    def _new_sess(self, max_pos: int, W: int, reserved: int) -> _Sess:
        C = self._swa_capacity(W)
        s = _Sess(max_pos=max_pos, W=W, C=C, reserved=reserved)
        for li in self._glob_layers:
            s.glob[li] = np.zeros((max_pos, self.H), dtype=np.int64)
        for li in self._swa_layers:
            s.swa[li] = np.zeros((C, self.H), dtype=np.int64)
            s.tag[li] = np.full(C, -1, dtype=np.int64)
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

    # ── mirroring / takeover ───────────────────────────────────────────────────────────────
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
        rows = tuple(Rows("i32", s.glob[li][lo_pos:hi_pos].astype(np.int32)) for li in self._glob_layers)
        return KVBlob(lo_pos, hi_pos, self._glob_layers, rows, self._layout, self._nid)

    def import_state(self, sid: int, blob: KVBlob, upto: int) -> None:
        s = self._get(sid)
        if s.length != 0 or s.rebuild_hi is not None:
            raise BadRequest("import_state needs a fresh session (length 0, no rebuild running)", sid=sid)
        if blob.layout_id != self._layout or blob.layers != self._glob_layers:
            raise BadRequest("KV layout mismatch: the blob is from another span/model/dtype", sid=sid)
        if blob.lo_pos != s.upto:
            raise BadRequest(f"imports must be contiguous: have [0, {s.upto}), blob starts at {blob.lo_pos}", sid=sid)
        if not blob.lo_pos <= upto <= blob.hi_pos or upto > s.max_pos:
            raise BadRequest(f"upto {upto} outside the blob [{blob.lo_pos}, {blob.hi_pos}) / max_pos", sid=sid)
        k = upto - blob.lo_pos
        for r in blob.rows:
            d = r.data
            if r.dtype != "i32" or d.ndim != 2 or d.shape[1] != self.H or (d < 0).any() or (d >= P).any():
                raise BadRequest(f"blob rows malformed for this layout: {r!r}", sid=sid)
        for li, r in zip(blob.layers, blob.rows):
            s.glob[li][blob.lo_pos:upto] = r.data[:k]
        s.upto = upto
        s.imported = True

    def rebuild_from(self, sid: int, F: int, exact_from: int | None = None) -> int:
        s = self._get(sid)
        if s.length != 0 or s.rebuild_hi is not None:
            raise BadRequest("rebuild_from needs a fresh or freshly imported session", sid=sid)
        if not s.upto <= F <= s.max_pos:
            raise BadRequest(f"F={F} must satisfy imported upto {s.upto} <= F <= max_pos {s.max_pos}", sid=sid)
        try:
            a = rebuild_start(self.kinds, self.window, F, s.upto, exact_from)
        except BadRequest as e:
            raise BadRequest(str(e), sid=sid) from None
        s.length, s.swa_lo, s.imported = a, a, False
        s.committed = max(s.committed, F)
        s.rebuild_hi = F if a < F else None
        return a

    # ── head / tail helpers ────────────────────────────────────────────────────────────────
    def embed(self, ids: Rows) -> Rows:
        self._loaded()
        if not self.spec.is_head:
            raise BadRequest("embed() is head-only")
        self._check_ids(ids, ids.n if isinstance(ids, Rows) else -1, None)
        return Rows("i32", self._w["embed"][ids.data.astype(np.int64)].astype(np.int32))

    def project(self, hidden: Rows) -> Rows:
        self._loaded()
        if not self.spec.is_tail:
            raise BadRequest("project() is tail-only")
        self._check_hidden(hidden, hidden.n if isinstance(hidden, Rows) else -1, None, "hidden")
        return Rows("i32", self._logits(hidden.data.astype(np.int64)).astype(np.int32))

    def _logits(self, x: np.ndarray) -> np.ndarray:
        y = x * self._w["norm"] % P
        return y @ self._w["lm_head"].T % P

    # ── compute ────────────────────────────────────────────────────────────────────────────
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
        ctx = _StepCtx(total, len(segs))
        return StepOut(tuple(self._run(self._sess[seg.sid], seg, ctx) for seg in segs))

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
        if not isinstance(x, Rows) or x.dtype != "i32" or x.data.shape != (n, self.H):
            raise BadRequest(f"{what} must be Rows i32 [{n}, {self.H}], got {x!r}", sid=sid)
        if (x.data < 0).any() or (x.data >= P).any():
            raise BadRequest(f"{what} value outside the toy field", sid=sid)

    def _run(self, s: _Sess, seg: Segment, ctx: _StepCtx) -> SegOut:
        start, n, H = seg.start_pos, seg.n, self.H
        if start < s.length:
            s.length = start                                  # implicit truncate (validated)
        rebuilding = s.rebuild_hi is not None
        if self.spec.is_head:
            x = self._w["embed"][seg.x.data.astype(np.int64)]
        else:
            x = seg.x.data.astype(np.int64)
        fc = self.spec.tap_mode == "fc_partial" and seg.taps
        acc = None
        if fc:
            acc = seg.tap_in.data.astype(np.int64) if seg.tap_in is not None else np.zeros((n, H), np.int64)
        taps = {}
        for li in range(self.spec.lo, self.spec.hi):
            out = np.empty((n, H), dtype=np.int64)
            for r in range(n):
                p = start + r
                self._write_kv(s, li, p, x[r], rebuilding)
                att = self._attend(s, li, p)
                out[r] = self._layer_row(li, x[r], att, self._row_term(ctx, seg, li, r, x))
            if seg.taps and li in self._tap_layers:
                t = self._tap_value(li, x, out)
                if fc:
                    acc = (acc + t @ self._w[f"tap_fc.{li}"].T) % P
                else:
                    taps[li] = Rows("i32", t.astype(np.int32))
            x = out
        s.length = start + n
        if rebuilding and s.length >= s.rebuild_hi:
            s.rebuild_hi = None
        logits = tokens = None
        if seg.logits != "none":
            logits = Rows("i32", self._logits(self._last_rows(x, seg.logits)).astype(np.int32))
        if seg.tokens != "none":
            rows = self._last_rows(x, seg.tokens)
            first = start + n - len(rows)
            lg = self._logits(rows)
            tokens = Rows("i32", np.array([self._draw(lg[i], seg.sampling, first + i, first + i - start)
                                           for i in range(len(rows))], dtype=np.int32))
        return SegOut(seg.sid, Rows("i32", x.astype(np.int32)), logits, taps,
                      None if acc is None else Rows("i32", acc.astype(np.int32)), tokens)

    # ── the model (hooks the mutants override) ─────────────────────────────────────────────
    def _last_rows(self, x: np.ndarray, mode: str) -> np.ndarray:
        return x[-1:] if mode == "last" else x

    def _draw(self, logits_row, sampling, p: int, r: int) -> int:
        return toy_draw(logits_row, sampling, p)

    def _row_term(self, ctx: _StepCtx, seg: Segment, li: int, r: int, x: np.ndarray) -> int:
        return 0 if self.mode == "chunk_invariant" else _shape_term(seg.start_pos, seg.n, r, li)

    def _layer_row(self, li: int, x: np.ndarray, att: np.ndarray, term: int) -> np.ndarray:
        a, b, c = self._lw[li]
        return (a * x + b * np.roll(x, -1) + np.roll(att, -(li % self.H)) + c + term) % P

    def _write_kv(self, s: _Sess, li: int, p: int, row: np.ndarray, rebuilding: bool) -> None:
        if self.kinds_all[li] == "global":
            if rebuilding and p < s.upto:
                return                        # mirrored rows are read, never overwritten (§4.1)
            s.glob[li][p] = row
        else:
            c = p % s.C
            s.swa[li][c] = row
            s.tag[li][c] = p

    def _window_lo(self, s: _Sess, p: int) -> int:
        return max(s.swa_lo, p - self.window + 1, 0)

    def _attend(self, s: _Sess, li: int, p: int) -> np.ndarray:
        if self.kinds_all[li] == "global":
            return self._attend_global(s, li, p)
        q0 = self._window_lo(s, p)
        qs = np.arange(q0, p + 1)
        slots = qs % s.C
        if self._strict and not (s.tag[li][slots] == qs).all():
            raise AssertionError(f"toy SWA ring lost a row it needs (layer {li}, position {p})")
        coef = self._qw[q0:p + 1] * self._pow[p - q0::-1] % P
        return (s.swa[li][slots] * coef[:, None]).sum(0) % P

    def _attend_global(self, s: _Sess, li: int, p: int) -> np.ndarray:
        coef = self._qw[:p + 1] * self._pow[p::-1] % P
        return (s.glob[li][:p + 1] * coef[:, None]).sum(0) % P

    def _tap_value(self, li: int, x_in: np.ndarray, out: np.ndarray) -> np.ndarray:
        return out                            # tap L = OUTPUT of layer L (vLLM aux convention)


class ToyReference:
    """The toy model's definition, driven through the calls the conformance suite makes (open,
    close, length, commit, truncate, step, export_state, import_state, rebuild_from, embed,
    project), written independently of ToyRuntime: plain Python ints, one dict of KV rows per
    layer and session, no ring, no reclamation, no validation. Rows below a rebuild start are
    simply absent, so windows read what is present (the partial-window semantics of §4.1).
    It trusts its caller: it is an oracle, not a runtime."""

    def __init__(self, weights: ToyWeights, spec: StageSpec, *, pattern: str, window: int,
                 mode: str = "chunk_invariant"):
        self.spec, self.window, self.mode = spec, window, mode
        self.H, self.V = weights.hidden, weights.vocab
        self.pattern = pattern
        self.kinds = tuple("swa" if c == "S" else "global" for c in pattern[spec.lo:spec.hi])
        self.glob = tuple(li for li in range(spec.lo, spec.hi) if pattern[li] == "G")
        self.tapset = tuple(sorted(t for t in set(spec.taps) if spec.lo <= t < spec.hi))
        self.W = {k: weights.read(k).tolist() for k in weights.names()}
        self.layout = kv_layout_id(spec.model, self.glob, "i32", _geometry(self.H))
        self.s = {}
        self._pow = [1]

    def _pw(self, k: int) -> int:
        while len(self._pow) <= k:
            self._pow.append(self._pow[-1] * _BASE % P)
        return self._pow[k]

    def open(self, sid, max_pos, spec_window):
        self.s[sid] = {"kv": {li: {} for li in range(self.spec.lo, self.spec.hi)},
                       "length": 0, "committed": 0, "upto": 0, "rb": None}

    def close(self, sid):
        self.s.pop(sid, None)

    def length(self, sid):
        return self.s[sid]["length"]

    def commit(self, sid, pos):
        self.s[sid]["committed"] = max(self.s[sid]["committed"], pos)

    def truncate(self, sid, pos):
        self._cut(self.s[sid], pos)

    def _cut(self, s, pos):
        for d in s["kv"].values():
            for q in [q for q in d if q >= pos]:
                del d[q]
        s["length"] = pos

    def export_state(self, sid, lo_pos, hi_pos, kinds=("global",)):
        kv = self.s[sid]["kv"]
        rows = tuple(Rows("i32", np.array([kv[li][q] for q in range(lo_pos, hi_pos)], dtype=np.int64)
                          .reshape(hi_pos - lo_pos, self.H).astype(np.int32)) for li in self.glob)
        return KVBlob(lo_pos, hi_pos, self.glob, rows, self.layout, "toy-reference")

    def import_state(self, sid, blob, upto):
        s = self.s[sid]
        for li, r in zip(blob.layers, blob.rows):
            for q in range(blob.lo_pos, upto):
                s["kv"][li][q] = [int(v) for v in r.data[q - blob.lo_pos]]
        s["upto"] = upto

    def rebuild_from(self, sid, F, exact_from=None):
        s = self.s[sid]
        a = rebuild_start(self.kinds, self.window, F, s["upto"], exact_from)
        s["length"], s["committed"] = a, max(s["committed"], F)
        s["rb"] = F if a < F else None
        return a

    def embed(self, ids):
        return Rows("i32", np.array([self.W["embed"][int(i)] for i in ids.data], dtype=np.int64)
                    .reshape(ids.n, self.H).astype(np.int32))

    def _lg(self, row):
        y = [row[h] * self.W["norm"][h] % P for h in range(self.H)]
        return [sum(y[h] * self.W["lm_head"][v][h] for h in range(self.H)) % P for v in range(self.V)]

    def project(self, hidden):
        return Rows("i32", np.array([self._lg([int(v) for v in r]) for r in hidden.data],
                                    dtype=np.int64).reshape(hidden.n, self.V).astype(np.int32))

    def step(self, x):
        return StepOut(tuple(self._seg(seg) for seg in x.segs))

    def _seg(self, seg):
        H = self.H
        s = self.s[seg.sid]
        start, n = seg.start_pos, seg.n
        if start < s["length"]:
            self._cut(s, start)
        rb = s["rb"]
        if self.spec.is_head:
            cur = [list(self.W["embed"][int(i)]) for i in seg.x.data]
        else:
            cur = [[int(v) for v in r] for r in seg.x.data]
        fc = self.spec.tap_mode == "fc_partial" and seg.taps
        acc = None
        if fc:
            acc = ([[int(v) for v in r] for r in seg.tap_in.data] if seg.tap_in is not None
                   else [[0] * H for _ in range(n)])
        taps = {}
        for li in range(self.spec.lo, self.spec.hi):
            kv = s["kv"][li]
            glob = self.pattern[li] == "G"
            a, b, c = self.W[f"layers.{li}.a"], self.W[f"layers.{li}.b"], self.W[f"layers.{li}.c"]
            nxt = []
            for r in range(n):
                p = start + r
                if not (glob and rb is not None and p < s["upto"]):
                    kv[p] = cur[r]
                att = [0] * H
                for q in range(0 if glob else max(0, p - self.window + 1), p + 1):
                    row = kv.get(q)
                    if row is None:
                        continue
                    cf = (q % 7 + 1) * self._pw(p - q) % P
                    for h in range(H):
                        att[h] = (att[h] + row[h] * cf) % P
                term = 0 if self.mode == "chunk_invariant" else _shape_term(start, n, r, li)
                xr = cur[r]
                nxt.append([(a[h] * xr[h] + b[h] * xr[(h + 1) % H] + att[(h + li) % H] + c[h] + term) % P
                            for h in range(H)])
            if seg.taps and li in self.tapset:
                if fc:
                    Wf = self.W[f"tap_fc.{li}"]
                    acc = [[(acc[r][j] + sum(Wf[j][h] * nxt[r][h] for h in range(H))) % P for j in range(H)]
                           for r in range(n)]
                else:
                    taps[li] = Rows("i32", np.array(nxt, dtype=np.int64).astype(np.int32))
            cur = nxt
        s["length"] = start + n
        if rb is not None and s["length"] >= rb:
            s["rb"] = None
        as_rows = lambda m, w: Rows("i32", np.array(m, dtype=np.int64).reshape(len(m), w).astype(np.int32))  # noqa: E731
        logits = tokens = None
        if seg.logits != "none":
            logits = as_rows([self._lg(r) for r in (cur[-1:] if seg.logits == "last" else cur)], self.V)
        if seg.tokens != "none":
            rows = cur[-1:] if seg.tokens == "last" else cur
            first = start + n - len(rows)
            tokens = Rows("i32", np.array([toy_draw(self._lg(r), seg.sampling, first + i)
                                           for i, r in enumerate(rows)], dtype=np.int32))
        return SegOut(seg.sid, as_rows(cur, H), logits, taps, None if acc is None else as_rows(acc, H), tokens)
