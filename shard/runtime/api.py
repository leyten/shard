"""StageRuntime: the device-agnostic contract for one layer span serving many sessions.

ARCHITECTURE-v2 §4.1 (scratchpad design; BUILD-PLAN F3). The node server (sessions, leases,
fence, batching, receipts, mirroring) and the pipelined-spec coordinator talk to a stage ONLY
through this protocol, so a CUDA card, an Apple-silicon Mac, a DGX Spark or a CPU box all plug
into the same data plane. `shard.node.ModelRuntime` (the v1 B=1 contract that MlxRuntime
implements) is untouched; a facade over this protocol is a later PR (R3).

Device neutrality is a hard rule, not a style choice:
  * This module imports only the stdlib and numpy. No torch, no torch.cuda, no mlx. A test
    imports it with torch blocked (tests/test_runtime_api.py).
  * Tensors cross the boundary as `Rows`: a C-contiguous HOST numpy array plus a logical
    dtype tag. bf16 travels as its uint16 bit pattern (numpy has no bf16; the same convention
    as shard/mlx_runtime.py), fp8 e4m3 as uint8 bytes plus per-row fp32 scales. A CUDA backend
    copies to host at the boundary; those bytes go on the wire (and into the receipt hash)
    anyway. torch/mlx tensors never appear in a signature.
  * `describe()` returns `Capability`, a plain-data record (JSON-ready) whose `DeviceInfo`
    names the backend family, the device, memory, bandwidth, the compute class and which
    numeric formats run in hardware vs emulated. The router and placement code reason about
    a 3090 (no fp8/fp4 hardware), a Spark (ARM, unified memory) and a Mac (MLX, unified)
    with the same functions (`sessions_at`, `format_support`).

Contract summary (details on each method of `StageRuntime`):
  * Positions are absolute. `length(sid)` rows are materialized. `committed` is the final-row
    frontier F: rows < F are never rolled back. A segment starting before `length` is an
    implicit truncate (legal iff start >= committed). A segment may end at most `spec_window`
    rows past the frontier; commit may run AHEAD of length (prompt rows are final before
    they are computed), which is how long prefill chunks and rebuilds stay legal.
  * SWA layers keep (window - 1 + spec_window) rows, so a rollback to the frontier after a
    full speculative window still finds every row it reads.
  * Mirroring exports global-attention KV rows [lo, hi) below the frontier in one canonical,
    backend-independent row layout (so a 5090 can take over from a Mac); a takeover imports
    [0, upto), calls `rebuild_from(F)` and feeds `rebuild` segments over [a, F). During a
    rebuild, global layers READ mirrored rows below `upto` and never overwrite them.
  * Determinism (numerics_mode == "invariant"): batch invariance, row independence (a row
    depends on its input, its causal prefix and its segment's shape, never on later rows or
    other segments), replayability of the recorded schedule, partition invariance for the
    canonical wire. `chunk_invariant` additionally makes a rebuild bit-equal to the
    fault-free state.
  * The tail returns token ids drawn on the device (greedy, or a counter RNG keyed by
    (seed, position)), so a verify step never has to ship [rows x vocab] logits to the host.

Threading and failure: one caller (the node's batcher) drives a runtime; calls are not
thread-safe. Every refusal is raised BEFORE any state changes. A backend fault after
validation (device error, OOM) leaves the sessions of that step undefined: the caller must
close them and fail over (§8).

Deliberately NOT in this contract yet (each is an additive change when its PR needs it):
tree verify (Segment positions / tree_parent), token ids at non-head stages (V4-style hash
routing; MiMo needs ids at the head only, and C12 keeps them off middle hops), a "recurrent"
layer kind with checkpointed rollback (K3/V4/linear attention), a per-segment output codec
(the node quantizes fp8 wire rows today), and an on-device drafter hook for the tail (D3; the
tail exposes hidden rows, embed() and project() meanwhile).
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field, fields
from typing import Iterable, Mapping, Protocol, runtime_checkable

import numpy as np

# ── vocabulary ────────────────────────────────────────────────────────────────────────────────

BACKENDS = ("cuda", "mlx", "rocm", "cpu")
# Numeric formats a device can compute in. "hw" = native matmul/tensor-core support;
# "emulated" = loadable but dequantized or emulated (e.g. MXFP4 weights on sm_86 via W4A16).
FORMATS = ("fp32", "bf16", "fp16", "fp8_e4m3", "fp8_e4m3fnuz", "fp8_e5m2", "mxfp4", "nvfp4",
           "int8", "int4", "int32")
LAYER_KINDS = ("global", "swa")
SEG_KINDS = ("prefill", "verify", "decode", "rebuild")
LOGITS_MODES = ("none", "last", "all")          # also the modes of Segment.tokens
TAP_MODES = ("raw", "fc_partial")
NUMERICS_MODES = ("invariant", "fast")
# Logical dtype -> numpy storage dtype of a Rows array. "fp8_e4m3" is OCP E4M3FN (NVIDIA,
# Apple, most CPUs); "fp8_e4m3fnuz" is the variant AMD MI300 computes in. Both carry per-row
# scales. "f16" is IEEE half (e.g. MLX fp16 KV).
STORAGE = {"bf16": np.uint16, "f16": np.float16, "f32": np.float32, "fp8_e4m3": np.uint8,
           "fp8_e4m3fnuz": np.uint8, "i32": np.int32, "i64": np.int64}
FP8_DTYPES = ("fp8_e4m3", "fp8_e4m3fnuz")
ID_DTYPES = ("i32", "i64")
# Canonical mirror row layout, versioned into every kv_layout_id: for one global layer at one
# position, K of every KV head (head-major, post-RoPE) then V of every KV head (after any value
# scale the kernel folds into the cache), in the stage's kv_dtype. A backend whose cache is
# paged, blocked or transposed converts at export/import; it never ships its internal layout.
KV_LAYOUT_VERSION = "kvl-canon-1"


# ── errors ────────────────────────────────────────────────────────────────────────────────────

class StageRuntimeError(Exception):
    """Base of every contract refusal. A refused call leaves every session unchanged.
    `sid` names the offending session when one segment of a batched step caused the refusal,
    so the node can drop that frame and resubmit the rest."""

    def __init__(self, msg: str = "", *, sid: int | None = None):
        super().__init__(msg)
        self.sid = sid


class KVFull(StageRuntimeError):
    """open() refused: the KV pool (or the session table) cannot hold max_pos + SWA slack."""


class WTooDeep(StageRuntimeError):
    """open() refused: spec_window deeper than this stage supports."""


class SessionExists(StageRuntimeError):
    pass


class UnknownSession(StageRuntimeError):
    pass


class RollbackPastCommit(StageRuntimeError):
    """truncate (explicit, or implicit via start_pos) below the committed frontier."""


class PositionGap(StageRuntimeError):
    """A segment or truncate points past length(sid): the protocol never skips positions."""


class SpecWindowExceeded(StageRuntimeError):
    """A segment would end more than spec_window rows past the committed frontier."""


class StepTooLarge(StageRuntimeError):
    """sum(n) over a step's segments exceeds max_tokens_per_step."""


class BadRequest(StageRuntimeError, ValueError):
    """A malformed segment, input, export or import (wrong dtype/shape/kind/order)."""


class LoadRefused(StageRuntimeError):
    """load() refused: a tensor the span needs is missing or malformed (never random-init)."""


# ── Rows: the backend-neutral array type ─────────────────────────────────────────────────────

@dataclass(frozen=True, eq=False)
class Rows:
    """Host rows crossing the StageRuntime boundary.

    `data` is [n] (token ids) or [n, width], C-contiguous, stored as STORAGE[dtype]. For
    "fp8_e4m3", `scale` holds one fp32 scale PER ROW: a per-tensor scale would couple rows and
    break row independence (ARCHITECTURE-v2 §4.1). Rows does not copy: a producer must not
    mutate an array after handing it over. Equality is byte equality."""
    dtype: str
    data: np.ndarray
    scale: np.ndarray | None = None

    def __post_init__(self):
        if self.dtype not in STORAGE:
            raise BadRequest(f"unknown Rows dtype {self.dtype!r} (known: {sorted(STORAGE)})")
        d = self.data
        if not isinstance(d, np.ndarray) or d.dtype != STORAGE[self.dtype]:
            got = getattr(d, "dtype", type(d).__name__)
            raise BadRequest(f"Rows {self.dtype} must be stored as {np.dtype(STORAGE[self.dtype])}, got {got}")
        if d.ndim not in (1, 2):
            raise BadRequest(f"Rows must be [n] or [n, width], got shape {d.shape}")
        if not d.flags.c_contiguous:
            object.__setattr__(self, "data", np.ascontiguousarray(d))
        if self.dtype in FP8_DTYPES:
            s = self.scale
            if not isinstance(s, np.ndarray) or s.dtype != np.float32 or s.shape != (d.shape[0],):
                raise BadRequest(f"{self.dtype} Rows need per-row fp32 scales of shape [n]")
        elif self.scale is not None:
            raise BadRequest(f"only fp8 Rows carry scales, not {self.dtype}")

    @property
    def n(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return 1 if self.data.ndim == 1 else int(self.data.shape[1])

    def row_bytes(self, i: int) -> bytes:
        """Wire bytes of row i: the row, then its fp8 scale (ARCHITECTURE-v2 §7.1 wire row)."""
        b = self.data[i].tobytes()
        return b + self.scale[i].tobytes() if self.scale is not None else b

    def slice(self, lo: int, hi: int) -> "Rows":
        return Rows(self.dtype, self.data[lo:hi], None if self.scale is None else self.scale[lo:hi])

    @staticmethod
    def concat(parts: Iterable["Rows"]) -> "Rows":
        parts = list(parts)
        if not parts:
            raise BadRequest("concat of no Rows")
        dt = {p.dtype for p in parts}
        if len(dt) != 1:
            raise BadRequest(f"concat of mixed dtypes {sorted(dt)}")
        sc = None if parts[0].scale is None else np.concatenate([p.scale for p in parts])
        return Rows(parts[0].dtype, np.concatenate([p.data for p in parts]), sc)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Rows):
            return NotImplemented
        if self.dtype != other.dtype or self.data.shape != other.data.shape:
            return False
        if self.data.tobytes() != other.data.tobytes():
            return False
        return (self.scale is None) == (other.scale is None) and (
            self.scale is None or self.scale.tobytes() == other.scale.tobytes())

    __hash__ = None

    def __repr__(self) -> str:
        h = hashlib.sha256(self.data.tobytes()).hexdigest()[:10]
        return f"Rows({self.dtype}, {tuple(self.data.shape)}, sha={h})"


# ── KV mirroring blob ─────────────────────────────────────────────────────────────────────────

_BLOB_MAGIC = b"SKV1"


@dataclass(frozen=True, eq=False)
class KVBlob:
    """Global-attention KV rows [lo_pos, hi_pos) of a span: what a primary mirrors to its backup.

    `layers` are the absolute ids of the span's global-attention layers (ascending), `rows[i]`
    is layer i's [hi_pos - lo_pos, kv_width] block in the runtime's KV storage dtype.
    `layout_id` is kv_layout_id(...): the canonical row layout version, model, layers, KV dtype
    and head geometry. An import refuses any other id.
    `numerics_id` is provenance only (a cross-class takeover is legal, §6)."""
    lo_pos: int
    hi_pos: int
    layers: tuple[int, ...]
    rows: tuple[Rows, ...]
    layout_id: str
    numerics_id: str

    def __post_init__(self):
        if not 0 <= self.lo_pos <= self.hi_pos:
            raise BadRequest(f"KVBlob range [{self.lo_pos}, {self.hi_pos}) is not a range")
        if len(self.layers) != len(self.rows):
            raise BadRequest("KVBlob needs one Rows block per layer")
        if list(self.layers) != sorted(set(self.layers)):
            raise BadRequest("KVBlob layers must be ascending and unique")
        for r in self.rows:
            if r.n != self.hi_pos - self.lo_pos:
                raise BadRequest(f"KVBlob block holds {r.n} rows for range [{self.lo_pos}, {self.hi_pos})")

    def pos_bytes(self, p: int) -> bytes:
        """KV bytes of the span at position p, layers ascending: the §7.1 KV-leaf payload."""
        if not self.lo_pos <= p < self.hi_pos:
            raise BadRequest(f"position {p} outside [{self.lo_pos}, {self.hi_pos})")
        return b"".join(r.row_bytes(p - self.lo_pos) for r in self.rows)

    def to_bytes(self) -> bytes:
        head = json.dumps({
            "lo": self.lo_pos, "hi": self.hi_pos, "layers": list(self.layers),
            "layout_id": self.layout_id, "numerics_id": self.numerics_id,
            "rows": [{"dtype": r.dtype, "shape": list(r.data.shape)} for r in self.rows],
        }, sort_keys=True, separators=(",", ":")).encode()
        body = b"".join(r.data.tobytes() + (b"" if r.scale is None else r.scale.tobytes())
                        for r in self.rows)
        return _BLOB_MAGIC + struct.pack("<I", len(head)) + head + body

    @classmethod
    def from_bytes(cls, buf: bytes) -> "KVBlob":
        """Parse to_bytes() output. Any malformed or trailing byte is a BadRequest."""
        if len(buf) < 8 or buf[:4] != _BLOB_MAGIC:
            raise BadRequest("not a KVBlob")
        (hl,) = struct.unpack("<I", buf[4:8])
        try:
            head = json.loads(buf[8:8 + hl])
            specs = head["rows"]
            off = 8 + hl
            rows = []
            for sp in specs:
                dt, shape = sp["dtype"], tuple(int(x) for x in sp["shape"])
                if dt not in STORAGE or len(shape) not in (1, 2):
                    raise BadRequest(f"bad row spec {sp}")
                item = np.dtype(STORAGE[dt]).itemsize
                nb = item * int(np.prod(shape))
                if off + nb > len(buf):
                    raise BadRequest("truncated KVBlob")
                data = np.frombuffer(buf, dtype=STORAGE[dt], count=int(np.prod(shape)), offset=off).reshape(shape).copy()
                off += nb
                scale = None
                if dt in FP8_DTYPES:
                    if off + 4 * shape[0] > len(buf):
                        raise BadRequest("truncated KVBlob scales")
                    scale = np.frombuffer(buf, dtype=np.float32, count=shape[0], offset=off).copy()
                    off += 4 * shape[0]
                rows.append(Rows(dt, data, scale))
            if off != len(buf):
                raise BadRequest(f"{len(buf) - off} trailing bytes after KVBlob")
            return cls(int(head["lo"]), int(head["hi"]), tuple(int(x) for x in head["layers"]),
                       tuple(rows), str(head["layout_id"]), str(head["numerics_id"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
            if isinstance(e, BadRequest):
                raise
            raise BadRequest(f"malformed KVBlob: {e}") from e

    def __eq__(self, other) -> bool:
        if not isinstance(other, KVBlob):
            return NotImplemented
        return self.to_bytes() == other.to_bytes()

    __hash__ = None


def kv_layout_id(model: str, layers: Iterable[int], kv_dtype: str,
                 geometry: tuple[int, int, int]) -> str:
    """Identity of a span's mirrored global-KV rows: the canonical layout version
    (KV_LAYOUT_VERSION), the model, the global layers, the KV dtype and the per-layer geometry
    (n_kv_heads, k_head_dim, v_head_dim). Equal ids promise equal meaning of every byte, so
    placement can pick a compatible backup from describe() alone; import refuses any other id."""
    n_kv, kd, vd = (int(g) for g in geometry)
    blob = json.dumps([KV_LAYOUT_VERSION, model, list(layers), kv_dtype, n_kv, kd, vd],
                      separators=(",", ":"))
    return "kvl1-" + hashlib.sha256(blob.encode()).hexdigest()[:24]


# ── capability record (describe) ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DeviceInfo:
    """What the hardware is, in terms every backend can fill and every router can compare.

    arch_class: the compute capability or its equivalent, e.g. "sm_86" (3090), "sm_89" (4090),
    "sm_120" (5090 / RTX 6000 Pro), "sm_121" (GB10 Spark), "apple_m2", "gfx1100", "cpu-x86_64".
    host_arch: the host CPU ISA ("x86_64", "aarch64" on a Spark, "arm64" on a Mac).
    free_mem_bytes: what THIS backend can still allocate, not the raw free counter: under the
    wired limit on a Mac (iogpu.wired_limit_mb), minus the host reserve on unified memory.
    unified_memory: device and host share one pool (Apple silicon, GB10, CPU), so host copies
    at the Rows boundary are cheap. mem_bw_bytes_s is an estimate; mem_bw_source says whether
    it was measured, taken from a spec sheet, or is unknown (0.0)."""
    backend: str
    device_name: str
    arch_class: str
    host_arch: str
    n_devices: int
    total_mem_bytes: int
    free_mem_bytes: int
    unified_memory: bool
    mem_bw_bytes_s: float
    mem_bw_source: str
    formats_hw: tuple[str, ...]
    formats_emulated: tuple[str, ...]

    def __post_init__(self):
        if self.backend not in BACKENDS:
            raise BadRequest(f"backend {self.backend!r} not in {BACKENDS}")
        if self.n_devices < 1 or self.total_mem_bytes < 0 or self.free_mem_bytes < 0:
            raise BadRequest("device counts and memory must be non-negative (n_devices >= 1)")
        if self.mem_bw_bytes_s < 0 or self.mem_bw_source not in ("measured", "spec", "unknown"):
            raise BadRequest("mem_bw_bytes_s >= 0 and mem_bw_source in measured/spec/unknown")
        bad = [f for f in self.formats_hw + self.formats_emulated if f not in FORMATS]
        if bad:
            raise BadRequest(f"unknown numeric formats {bad} (known: {FORMATS})")
        if set(self.formats_hw) & set(self.formats_emulated):
            raise BadRequest("a format is either hardware or emulated, not both")

    def format_support(self, fmt: str) -> str | None:
        """'hw', 'emulated', or None (cannot run it)."""
        if fmt in self.formats_hw:
            return "hw"
        if fmt in self.formats_emulated:
            return "emulated"
        return None


@dataclass(frozen=True)
class Capability:
    """describe(): the stage as the directory, router, admission and batcher see it.

    Static facts (span, layer kinds, KV geometry, weight footprint, limits) are valid right
    after construction, before load(), so placement can size a span on a candidate device
    without loading it. `numerics_id` is final after load() (it names the artifact CID);
    `service_ms` / `self_probe_ok` are filled by warmup(). `kv_free_bytes` and `open_sessions`
    are live. Every field is plain data (JSON via to_dict/from_dict)."""
    device: DeviceInfo
    model: str
    lo: int
    hi: int
    is_head: bool
    is_tail: bool
    layer_kinds: tuple[str, ...]          # per held layer, "global" | "swa"
    swa_window: int                       # w of the SWA layers (0 if the span holds none)
    hidden_size: int
    vocab_size: int
    wire_dtype: str                       # dtype of hidden rows and tap rows this stage emits
    codecs_in: tuple[str, ...]            # hidden dtypes it accepts (head: token-id dtypes)
    logits_dtype: str                     # dtype of logits rows (tail)
    kv_dtype: str                         # dtype of mirrored KV rows
    kv_layout_id: str                     # a backup must report the same id (kv_layout_id())
    global_kv_bytes_per_pos: int          # all global layers of the span, one position
    swa_kv_bytes_per_pos: int             # all SWA layers of the span, one position
    kv_page_positions: int                # KV allocation granule in positions (1 = exact)
    kv_free_bytes: int
    layer_weight_bytes: tuple[int, ...]   # resident weights per held layer, this backend's format
    edge_weight_bytes: int                # embed / final norm / lm_head / tap slices held
    load_peak_extra_bytes: int            # transient on top of weight_bytes while loading
    max_sessions: int
    max_ctx: int
    spec_window_max: int
    max_tokens_per_step: int
    tap_layers_held: tuple[int, ...]
    tap_mode: str
    numerics_id: str
    numerics_mode: str
    chunk_invariant: bool                 # bytes independent of segment shapes (rebuild exact)
    # (sessions in the step, tokens per session, context position, ms per step), from warmup()
    service_ms: tuple[tuple[int, int, int, float], ...] = ()
    self_probe_ok: bool | None = None     # warmup self-probe (repeat == first run, graph == eager)
    open_sessions: int = 0

    def __post_init__(self):
        if not 0 <= self.lo < self.hi:
            raise BadRequest(f"layer range [{self.lo}, {self.hi}) is empty")
        if len(self.layer_kinds) != self.hi - self.lo or any(k not in LAYER_KINDS for k in self.layer_kinds):
            raise BadRequest("layer_kinds must name each held layer 'global' or 'swa'")
        if ("swa" in self.layer_kinds) != (self.swa_window > 0):
            raise BadRequest("swa_window > 0 iff the span holds SWA layers")
        if self.tap_mode not in TAP_MODES or self.numerics_mode not in NUMERICS_MODES:
            raise BadRequest("bad tap_mode or numerics_mode")
        dts = (self.wire_dtype, self.logits_dtype, self.kv_dtype, *self.codecs_in)
        if any(d not in STORAGE for d in dts):
            raise BadRequest("wire/logits/kv dtypes and codecs_in must be Rows dtypes")
        if self.kv_page_positions < 1 or not self.numerics_id or not self.kv_layout_id:
            raise BadRequest("kv_page_positions >= 1, a numerics_id and a kv_layout_id are required")
        if len(self.layer_weight_bytes) != self.hi - self.lo:
            raise BadRequest("layer_weight_bytes needs one entry per held layer")
        if min(*self.layer_weight_bytes, self.edge_weight_bytes, self.load_peak_extra_bytes,
               self.vocab_size, self.hidden_size) < 0:
            raise BadRequest("sizes must be non-negative")

    @property
    def n_swa(self) -> int:
        return sum(k == "swa" for k in self.layer_kinds)

    @property
    def rebuild_positions(self) -> int:
        """Context-independent part of a takeover rebuild: (w-1) * n_swa positions (plus F - g)."""
        return max(self.swa_window - 1, 0) * self.n_swa

    @property
    def layer_ms(self) -> tuple[tuple[int, int, int, float], ...]:
        """Measured per-layer cost: service_ms spread over the held layers, per step shape."""
        n = self.hi - self.lo
        return tuple((s, b, c, ms / n) for s, b, c, ms in self.service_ms)

    @property
    def weight_bytes(self) -> int:
        return sum(self.layer_weight_bytes) + self.edge_weight_bytes

    def session_kv_bytes(self, ctx: int, spec_window: int) -> int:
        """KV one session reserves at open(max_pos=ctx, spec_window): global rows for every
        position, SWA rows for (w - 1 + spec_window) positions, both rounded up to pages."""
        pg = self.kv_page_positions
        up = lambda x: -(-x // pg) * pg                                   # noqa: E731
        swa_rows = self.swa_window - 1 + spec_window if self.swa_window else 0
        return self.global_kv_bytes_per_pos * up(ctx) + self.swa_kv_bytes_per_pos * up(swa_rows)

    def sessions_at(self, ctx: int, spec_window: int) -> int:
        """How many MORE sessions of this context open() admits right now."""
        per = self.session_kv_bytes(ctx, spec_window)
        by_kv = self.kv_free_bytes // per if per else self.max_sessions
        return max(0, min(self.max_sessions - self.open_sessions, by_kv))

    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, DeviceInfo):
                v = {g.name: (list(getattr(v, g.name)) if isinstance(getattr(v, g.name), tuple)
                              else getattr(v, g.name)) for g in fields(v)}
            elif f.name == "service_ms":
                v = [list(e) for e in v]
            elif isinstance(v, tuple):
                v = list(v)
            out[f.name] = v
        return out

    @classmethod
    def from_dict(cls, d: Mapping) -> "Capability":
        d = dict(d)
        dev = dict(d.pop("device"))
        for k in ("formats_hw", "formats_emulated"):
            dev[k] = tuple(dev[k])
        d["device"] = DeviceInfo(**dev)
        for k in ("layer_kinds", "codecs_in", "tap_layers_held", "layer_weight_bytes"):
            d[k] = tuple(d[k])
        d["service_ms"] = tuple((int(s), int(b), int(c), float(ms)) for s, b, c, ms in d["service_ms"])
        return cls(**d)


def make_numerics_id(*, backend: str, arch_class: str, mode: str, kernel_pins: Iterable[str] = (),
                     quant: str = "", artifact_cid: str = "") -> str:
    """numerics_id = hash(backend, kernel pins, arch class, quant scheme, artifact CID, mode)
    (ARCHITECTURE-v2 §6). Two stages are bit-comparable iff their numerics_ids match."""
    blob = json.dumps({"backend": backend, "arch_class": arch_class, "mode": mode,
                       "kernel_pins": list(kernel_pins), "quant": quant, "artifact_cid": artifact_cid},
                      sort_keys=True, separators=(",", ":"))
    return "nid1-" + hashlib.sha256(blob.encode()).hexdigest()[:32]


# ── construction, weights, sessions ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StageSpec:
    """What a runtime is asked to serve. `device` is an opaque backend selector
    ("cuda:0", "cuda:0,1" for a multi-GPU stage, "metal", "" = backend default).
    `max_sessions` is the hard session-table size: the node server caps NEW leases at S_max and
    uses the rest as takeover headroom (C21), so size it >= 2 * S_max where the backend can (a
    max_sessions=1 backend, MLX today, serves as a primary only). `kv_budget_bytes` caps the KV
    pool (0 = backend decides from free device memory); per-step transient workspace (e.g. a
    prefill chunk's n + w - 1 SWA rows) is the backend's own, reserved at warmup, never per
    session. `kv_dtype` "" = backend default; describe() reports what it chose."""
    model: str
    lo: int
    hi: int
    is_head: bool = False
    is_tail: bool = False
    taps: tuple[int, ...] = ()            # drafter tap layers (absolute); tap L = output of layer L
    tap_mode: str = "raw"
    max_sessions: int = 32
    max_ctx: int = 32768
    spec_window: int = 64                 # deepest W open() accepts
    numerics: str = "invariant"
    device: str = ""
    kv_budget_bytes: int = 0
    kv_dtype: str = ""

    def __post_init__(self):
        if not 0 <= self.lo < self.hi:
            raise BadRequest(f"layer range [{self.lo}, {self.hi}) is empty")
        if self.tap_mode not in TAP_MODES or self.numerics not in NUMERICS_MODES:
            raise BadRequest("bad tap_mode or numerics")
        if self.max_sessions < 1 or self.max_ctx < 1 or self.spec_window < 1 or self.kv_budget_bytes < 0:
            raise BadRequest("max_sessions, max_ctx, spec_window >= 1; kv_budget_bytes >= 0")


@runtime_checkable
class WeightSource(Protocol):
    """Yields this stage's tensors (range-fetched and hash-verified by shard/fetch). The
    runtime reads only the names its span needs and refuses to substitute a random init."""
    artifact_cid: str

    def names(self) -> Iterable[str]: ...

    def read(self, name: str) -> np.ndarray: ...


@dataclass(frozen=True)
class LoadReport:
    n_tensors: int
    n_bytes: int
    artifact_cid: str


@dataclass(frozen=True)
class SessionInfo:
    sid: int
    length: int                  # rows [0, length) are materialized
    committed: int               # final-row frontier F (may run ahead of length)
    max_pos: int
    spec_window: int
    kv_bytes: int                # KV this session reserves
    imported_upto: int           # mirrored global rows [0, imported_upto) installed by import_state
    rebuild_hi: int | None       # while a takeover rebuild is in progress: F


# ── compute ───────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Sampling:
    """How the tail draws truth tokens (ARCHITECTURE-v2 G1). temperature == 0, or top_k == 1, is
    greedy: a token whose logit is maximal (the lowest id among exact ties in the backend's own
    compute precision; the returned logits may be rounded coarser, so a caller may only rely on
    the chosen token's returned logit being a maximum). Otherwise the draw at absolute position
    p is a function of (logits row p, these parameters, seed, p) ONLY: a counter RNG keyed by
    (seed, p), never by the row's place in its segment or step, so draws are replayable and
    independent of batching and chunking. top_k 0 = no cut; top_p 1.0 = no cut. Like every
    output, draws replay within one numerics_id; a takeover by another numerics class continues
    validly but may draw differently."""
    seed: int = 0
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0

    def __post_init__(self):
        if self.temperature < 0 or self.top_k < 0 or not 0 < self.top_p <= 1:
            raise BadRequest("temperature >= 0, top_k >= 0, 0 < top_p <= 1")


@dataclass(frozen=True)
class Segment:
    """One session's rows in a step.

    x: head stage -> token ids ("i32"/"i64", [n]); any other stage -> hidden rows [n, H] in one
    of describe().codecs_in. Inputs are per segment so one step can mix codecs (an fp8 prefill
    sub-chunk next to a bf16 decode row).
    tap_in: the fc_partial accumulator from upstream ([n, H], wire dtype); None means zeros.
    Only with taps=True in fc_partial mode.
    kind: decode (n == 1) | verify | prefill | rebuild (only after rebuild_from; no logits or
    tokens; taps as requested).
    logits / tokens: tail only, "none" | "last" | "all". tokens are drawn on the device with
    `sampling` (None = greedy), so a verify step returns n int32s instead of n x vocab floats."""
    sid: int
    start_pos: int
    n: int
    x: Rows
    kind: str = "decode"
    tap_in: Rows | None = None
    logits: str = "none"
    taps: bool = True
    tokens: str = "none"
    sampling: Sampling | None = None

    @property
    def codec_in(self) -> str:
        return self.x.dtype

    @property
    def shape(self) -> tuple[int, int, str, str]:
        """The SEG record (start_pos, n, codec, kind) receipts log (§7.1)."""
        return (self.start_pos, self.n, self.codec_in, self.kind)


@dataclass(frozen=True)
class StepIn:
    segs: tuple[Segment, ...]     # at most one segment per session


@dataclass(frozen=True)
class SegOut:
    sid: int
    hidden: Rows                  # [n, H] residual stream after the span's last layer (pre-final-norm)
    logits: Rows | None = None    # tail: [1, V] ("last") or [n, V] ("all"), logits_dtype
    taps: Mapping[int, Rows] = field(default_factory=dict)   # raw mode: layer -> [n, H]
    tap_acc: Rows | None = None   # fc_partial mode: tap_in + sum over held taps of W_fc,l . tap_l
    tokens: Rows | None = None    # tail: i32 [1] ("last") or [n] ("all")


@dataclass(frozen=True)
class StepOut:
    outs: tuple[SegOut, ...]      # same order as StepIn.segs


def rebuild_start(layer_kinds: Iterable[str], window: int, F: int, upto: int,
                  exact_from: int | None = None) -> int:
    """First position a takeover must recompute (ARCHITECTURE-v2 §4.1 / C13, corrected rule).

        a = min(F - (w-1)*n_swa,  min over global layers l of  upto - (w-1)*i_l),  clamped at 0

    i_l = SWA layers below l inside the span. Receptive-field argument: rows fed at p >= a are
    exact inputs; each SWA layer's output is exact only (w-1) rows later, a global layer adds
    no shift because it reads the exact mirrored rows below `upto`. So the top SWA layer's rows
    [F-w+1, F) come out exact, and every global row written (p >= upto) has an exact input.
    Requires global layers to READ mirrored rows for p < upto and never overwrite them.

    exact_from: also make the rebuild's OUTPUT rows exact from this position on (a tail whose
    drafter must be rebuilt needs ~3*127 exact last-hidden rows for MTP-3, 1024 for DFlash):
    a <= exact_from - (w-1)*n_swa. See rebuild_exact_from."""
    kinds = list(layer_kinds)
    if not 0 <= upto <= F:
        raise BadRequest(f"need 0 <= upto ({upto}) <= F ({F})")
    if exact_from is not None and not 0 <= exact_from <= F:
        raise BadRequest(f"need 0 <= exact_from ({exact_from}) <= F ({F})")
    w1 = max(window - 1, 0)
    n_swa = sum(k == "swa" for k in kinds)
    a = F - w1 * n_swa
    below = 0
    for k in kinds:
        if k == "swa":
            below += 1
        else:
            a = min(a, upto - w1 * below)
    if exact_from is not None:
        a = min(a, exact_from - w1 * n_swa)
    return max(0, a)


def rebuild_exact_from(layer_kinds: Iterable[str], window: int, a: int) -> int:
    """First position whose rebuild OUTPUT rows (hidden, and taps) equal the fault-free ones:
    every SWA layer adds (w-1) rows of partial window above a. 0 if the rebuild starts at 0."""
    return 0 if a == 0 else a + max(window - 1, 0) * sum(k == "swa" for k in layer_kinds)


@runtime_checkable
class StageRuntime(Protocol):
    """State and math for layers [spec.lo, spec.hi) of one model, for many sessions."""
    spec: StageSpec

    def load(self, src: WeightSource) -> LoadReport:
        """Materialize exactly the span's tensors; LoadRefused on any missing/malformed one."""

    def warmup(self, buckets: tuple[int, ...] = (1, 8, 32), contexts: tuple[int, ...] = (0,),
               sessions: tuple[int, ...] = (1,)) -> Capability:
        """Compile/capture what the backend needs, run the self-probe (a repeat step is byte-
        equal to the first; on CUDA also graph == eager), and measure service_ms for every step
        shape (s sessions x b tokens each, every session at context c) that fits the token
        budget: a 16-session decode step at 32k context reads 16 contexts of KV, which a
        single-session curve would under-cost. Returns describe(); leaves no session behind; may
        reserve per-step workspace, which lowers kv_free_bytes."""

    def describe(self) -> Capability: ...

    def open(self, sid: int, max_pos: int, spec_window: int) -> None:
        """Reserve KV for max_pos global rows + (w-1+spec_window) SWA rows (what
        describe().session_kv_bytes says). KVFull if the pool or the session table cannot hold
        it (or max_pos > max_ctx); WTooDeep if spec_window > spec_window_max; SessionExists if
        sid is open. sids are arbitrary u64 values."""

    def close(self, sid: int) -> None:
        """Drop the session (also mid-rebuild) and release its KV. Unknown sid: no-op."""

    def length(self, sid: int) -> int: ...

    def info(self, sid: int) -> SessionInfo: ...

    def truncate(self, sid: int, pos: int) -> None:
        """length := pos. RollbackPastCommit if pos < committed, PositionGap if pos > length.
        During a rebuild every truncate is refused (committed == F > length)."""

    def commit(self, sid: int, pos: int) -> None:
        """Rows < pos are final: they are never rolled back and SWA history below
        pos - (w-1) may be reclaimed. Monotone (a lower pos is a no-op); may run ahead of
        length (prompt rows are final before they are computed); BadRequest past max_pos.
        commit is the ROLLBACK FLOOR: a caller that may still need to rewind (entry
        re-adoption from the gateway's last checkpoint, §4.3) must not commit past that point."""

    def export_state(self, sid: int, lo_pos: int, hi_pos: int,
                     kinds: tuple[str, ...] = ("global",)) -> KVBlob:
        """Global-attention KV rows [lo_pos, hi_pos) in the canonical layout (KV_LAYOUT_VERSION).
        hi_pos <= min(committed, length): rows at or past the frontier are speculative and never
        mirrored. Only kinds=("global",). Refused mid-takeover (imported or rebuilding)."""

    def import_state(self, sid: int, blob: KVBlob, upto: int) -> None:
        """Install mirrored global rows [blob.lo_pos, upto) into a FRESH session (length 0).
        blob.layout_id must equal describe().kv_layout_id (another numerics class is fine).
        Chunks must be contiguous from 0; blob.lo_pos <= upto <= blob.hi_pos (upto = g_eff may
        cut a blob short). SWA rows stay unmaterialized; the next call must be rebuild_from."""

    def rebuild_from(self, sid: int, F: int, exact_from: int | None = None,
                     floor: int | None = None) -> int:
        """Start a takeover rebuild of a fresh or imported session up to row F. `floor` (default
        F) is the replaced hop's rollback floor, i.e. its committed frontier (the entry's cpos,
        capped at what the gateway acknowledged, §4.3): committed := floor, and a later
        truncate(floor) must re-feed exactly, so F - floor <= spec_window. Returns exactly
        a = rebuild_start(layer_kinds, w, F, imported_upto, min(exact_from, floor)) (auditors
        recompute it); afterwards length == a, and only `rebuild` segments continuing at
        length, up to F, are accepted until length == F. Their output rows at
        p >= rebuild_exact_from(...) equal the fault-free ones (bit-equal under chunk
        invariance); earlier rows are defined but meaningless."""

    def embed(self, ids: Rows) -> Rows:
        """Head only: token ids -> layer-lo input rows [n, H] in wire_dtype (drafters and the
        v1 ModelRuntime facade use it; step() embeds on its own)."""

    def project(self, hidden: Rows, *, tokens: bool = False, start_pos: int = 0,
                sampling: Sampling | None = None) -> Rows:
        """Tail only: final norm + lm_head over given pre-final-norm rows -> logits [n, V] in
        logits_dtype, or with tokens=True the drawn ids i32 [n] (row i drawn at position
        start_pos + i), so a drafter reusing the target's head never ships vocab-wide rows."""

    def step(self, x: StepIn) -> StepOut:
        """Run every segment. The whole step is validated before any state changes: a refused
        step leaves every session exactly as it was, and the error's .sid names the segment."""
