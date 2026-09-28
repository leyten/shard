"""shard/runtime/api.py: device neutrality, the wire types, the capability record, the
rebuild rule, and the untouched v1 ModelRuntime contract (BUILD-PLAN F3)."""
import ast
import inspect
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from shard.runtime.api import (BadRequest, Capability, DeviceInfo, KVBlob, Rows, Sampling,
                               StageRuntime, StageSpec, WeightSource, kv_layout_id,
                               make_numerics_id, rebuild_exact_from, rebuild_start)
from shard.runtime.toy import ToyRuntime, ToyWeights

ROOT = pathlib.Path(__file__).resolve().parent.parent
PORTABLE = [*sorted((ROOT / "shard" / "runtime").glob("*.py")),
            *sorted((ROOT / "shard" / "runtime" / "adapters").glob("*.py")), ROOT / "tests" / "runtime_conformance.py"]


# ── device neutrality ─────────────────────────────────────────────────────────────────────────

def _imported_roots(path):
    roots = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", PORTABLE, ids=lambda p: p.name)
def test_runtime_contract_imports_only_stdlib_and_numpy(path):
    """No torch / torch.cuda / mlx / vendor stack in the contract, the reference or the suite:
    a Mac, a Spark or a CPU box imports them as-is."""
    allowed = set(sys.stdlib_module_names) | {"numpy", "shard", "__future__"}
    assert _imported_roots(path) <= allowed, _imported_roots(path) - allowed


def test_contract_runs_with_accelerator_stacks_blocked():
    """In a fresh interpreter where torch and mlx cannot be imported, the API, the toy and the
    conformance module load and a toy step runs."""
    code = (
        "import sys; sys.modules['torch'] = None; sys.modules['mlx'] = None\n"
        f"sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'tests')!r}]\n"
        "import numpy as np, runtime_conformance\n"
        "from shard.runtime.api import StageSpec, StepIn, Segment, Rows\n"
        "from shard.runtime.toy import ToyRuntime, ToyWeights\n"
        "rt = ToyRuntime(StageSpec('toy', 0, 3, is_head=True, is_tail=True), pattern='SGS', window=2)\n"
        "rt.load(ToyWeights('SGS')); rt.open(1, 16, 2); rt.commit(1, 4)\n"
        "o = rt.step(StepIn((Segment(1, 0, 4, Rows('i32', np.arange(4, dtype=np.int32)), 'prefill', logits='last'),)))\n"
        "assert o.outs[0].logits.n == 1 and 'torch' not in [m for m in sys.modules if sys.modules[m]]\n"
        "print('ok')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-2000:]


# One router/placement code path over a deliberately mixed fleet (numbers illustrative, from
# spec sheets / docs/HETERO_DEVICES.md; only the SHAPE of the record matters here).
GiB = 1 << 30
HW_BLACKWELL = ("fp32", "bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "mxfp4", "nvfp4", "int8")
FLEET = {
    "rtx3090": DeviceInfo("cuda", "NVIDIA GeForce RTX 3090", "sm_86", "x86_64", 1, 24 * GiB, 23 * GiB,
                          False, 936e9, "spec", ("fp32", "bf16", "fp16", "int8"), ("fp8_e4m3", "mxfp4", "nvfp4")),
    "rtx4090": DeviceInfo("cuda", "NVIDIA GeForce RTX 4090", "sm_89", "x86_64", 1, 24 * GiB, 23 * GiB,
                          False, 1008e9, "spec", ("fp32", "bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "int8"),
                          ("mxfp4", "nvfp4")),
    "rtx5090": DeviceInfo("cuda", "NVIDIA GeForce RTX 5090", "sm_120", "x86_64", 1, 32 * GiB, 31 * GiB,
                          False, 1792e9, "measured", HW_BLACKWELL, ()),
    "pro6000": DeviceInfo("cuda", "NVIDIA RTX PRO 6000", "sm_120", "x86_64", 1, 96 * GiB, 94 * GiB,
                          False, 1792e9, "spec", HW_BLACKWELL, ()),
    "spark": DeviceInfo("cuda", "NVIDIA GB10", "sm_121", "aarch64", 1, 128 * GiB, 110 * GiB,
                        True, 273e9, "spec", HW_BLACKWELL, ()),
    "m2pro": DeviceInfo("mlx", "Apple M2 Pro", "apple_m2", "arm64", 1, 16 * GiB, 11 * GiB,
                        True, 200e9, "spec", ("fp32", "bf16", "fp16"), ("int4", "int8", "mxfp4")),
    "7900xtx": DeviceInfo("rocm", "AMD Radeon RX 7900 XTX", "gfx1100", "x86_64", 1, 24 * GiB, 23 * GiB,
                          False, 960e9, "spec", ("fp32", "bf16", "fp16", "int8"), ("fp8_e4m3",)),
}


def _cap(dev, n_layers, weight_bytes_per_layer=3 * GiB):
    kinds = tuple("global" if i % 4 == 3 else "swa" for i in range(n_layers))
    n_g = kinds.count("global")
    wb = n_layers * weight_bytes_per_layer
    return Capability(
        device=dev, model="mimo-v2.6-flash", lo=10, hi=10 + n_layers, is_head=False, is_tail=False,
        layer_kinds=kinds, swa_window=128, hidden_size=4096, vocab_size=152064, wire_dtype="bf16",
        codecs_in=("bf16", "fp8_e4m3"), logits_dtype="f32", kv_dtype="bf16",
        kv_layout_id=kv_layout_id("mimo-v2.6-flash", [10 + i for i, k in enumerate(kinds) if k == "global"],
                                  "bf16", (4, 192, 128)),
        global_kv_bytes_per_pos=n_g * 4 * (192 + 128) * 2, swa_kv_bytes_per_pos=(n_layers - n_g) * 8 * (192 + 128) * 2,
        kv_page_positions=64, kv_free_bytes=max(0, dev.free_mem_bytes - wb - 2 * GiB),
        layer_weight_bytes=tuple(weight_bytes_per_layer for _ in kinds), edge_weight_bytes=0,
        load_peak_extra_bytes=GiB // 2,
        max_sessions=256, max_ctx=65536, spec_window_max=64, max_tokens_per_step=2048,
        tap_layers_held=(), tap_mode="fc_partial",
        numerics_id=make_numerics_id(backend=dev.backend, arch_class=dev.arch_class, mode="invariant"),
        numerics_mode="invariant", chunk_invariant=False,
        service_ms=((1, 1, 0, 4.0), (1, 16, 0, 6.5), (16, 1, 32768, 9.0)))


def _layers_that_fit(dev, cap, ctx, sessions):
    """What placement computes before assigning a span: every input comes from the records."""
    per_layer = max(cap.layer_weight_bytes) + sessions * cap.session_kv_bytes(ctx, 64) / (cap.hi - cap.lo)
    return int((dev.free_mem_bytes - cap.load_peak_extra_bytes) // per_layer)


def test_one_code_path_reasons_about_a_mixed_fleet():
    spans = {"rtx3090": 6, "rtx4090": 6, "rtx5090": 8, "pro6000": 26, "spark": 30, "m2pro": 2, "7900xtx": 6}
    caps = {k: _cap(FLEET[k], n) for k, n in spans.items()}
    keys = None
    for name, c in caps.items():
        d = json.loads(json.dumps(c.to_dict()))            # what the directory receives
        assert Capability.from_dict(d) == c
        keys = keys or set(d) | set(d["device"])
        assert set(d) | set(d["device"]) == keys, f"{name}: the record's shape depends on the backend"
        assert c.sessions_at(32768, 64) >= 1, name
        assert c.layer_ms == tuple((n, b, x, ms / (c.hi - c.lo)) for n, b, x, ms in c.service_ms)
    fmt = {k: FLEET[k].format_support("mxfp4") for k in FLEET}
    assert fmt == {"rtx3090": "emulated", "rtx4090": "emulated", "rtx5090": "hw", "pro6000": "hw",
                   "spark": "hw", "m2pro": "emulated", "7900xtx": None}
    assert FLEET["rtx3090"].format_support("fp8_e4m3") == "emulated"           # Ampere: no FP8 hw
    assert [k for k, d in FLEET.items() if d.unified_memory] == ["spark", "m2pro"]
    same_span = {k: _cap(FLEET[k], 6).sessions_at(32768, 64) for k in ("rtx3090", "rtx5090", "pro6000", "spark")}
    assert same_span["rtx3090"] < same_span["rtx5090"] <= same_span["pro6000"] <= same_span["spark"]
    fit = {k: _layers_that_fit(FLEET[k], caps[k], 32768, 8) for k in FLEET}
    assert fit["m2pro"] < fit["rtx3090"] <= fit["rtx5090"] < fit["pro6000"] < fit["spark"], fit
    assert all(v >= 1 for v in fit.values()), fit                  # every device class gets a role
    assert len({c.kv_layout_id for c in caps.values() if c.layer_kinds == caps["rtx3090"].layer_kinds}) == 1
    assert caps["rtx3090"].rebuild_positions == 127 * caps["rtx3090"].n_swa


def test_capability_validation():
    good = _cap(FLEET["rtx5090"], 4)
    with pytest.raises(BadRequest):
        DeviceInfo("tpu", "x", "v5", "x86_64", 1, 0, 0, False, 0.0, "unknown", (), ())
    with pytest.raises(BadRequest):
        DeviceInfo("cuda", "x", "sm_86", "x86_64", 1, 0, 0, False, 0.0, "unknown", ("fp8_e4m3",), ("fp8_e4m3",))
    with pytest.raises(BadRequest):
        Capability(**{**good.__dict__, "layer_kinds": ("swa",)})
    with pytest.raises(BadRequest):
        Capability(**{**good.__dict__, "swa_window": 0})
    per = good.session_kv_bytes(100, 3)
    assert per == good.global_kv_bytes_per_pos * 128 + good.swa_kv_bytes_per_pos * 192   # 64-position pages
    assert Capability(**{**good.__dict__, "open_sessions": 256}).sessions_at(16, 1) == 0


# ── wire types ────────────────────────────────────────────────────────────────────────────────

def test_rows_types_and_byte_equality():
    h = Rows("bf16", np.arange(6, dtype=np.uint16).reshape(3, 2))
    assert h.n == 3 and h.width == 2 and h.row_bytes(1) == np.array([2, 3], np.uint16).tobytes()
    assert h == Rows("bf16", np.arange(6, dtype=np.uint16).reshape(3, 2))
    assert h != Rows("bf16", np.arange(6, dtype=np.uint16).reshape(2, 3))
    assert h != Rows("i32", np.arange(6, dtype=np.int32).reshape(3, 2))
    assert Rows.concat([h.slice(0, 1), h.slice(1, 3)]) == h
    with pytest.raises(BadRequest):
        Rows("bf16", np.zeros((2, 2), np.float32))                  # never coerce a foreign dtype
    with pytest.raises(BadRequest):
        Rows("fp8_e4m3", np.zeros((2, 2), np.uint8))                # fp8 needs per-row scales
    with pytest.raises(BadRequest):
        Rows("fp8_e4m3", np.zeros((2, 2), np.uint8), np.ones(1, np.float32))   # not per-tensor
    q = Rows("fp8_e4m3", np.ones((2, 2), np.uint8), np.array([0.5, 2.0], np.float32))
    assert q.row_bytes(1) == bytes([1, 1]) + np.float32(2.0).tobytes()          # §7.1 wire row
    with pytest.raises(BadRequest):
        Rows("i32", np.zeros((2, 2, 2), np.int32))
    Rows("f16", np.zeros((2, 2), np.float16))                          # MLX fp16 KV
    with pytest.raises(BadRequest):
        Rows("fp8_e4m3fnuz", np.zeros((2, 2), np.uint8))            # AMD fp8 also per-row scaled


def test_sampling_and_error_attribution():
    with pytest.raises(BadRequest):
        Sampling(temperature=-1.0)
    with pytest.raises(BadRequest):
        Sampling(top_p=0.0)
    e = BadRequest("x", sid=7)
    assert e.sid == 7 and isinstance(e, ValueError) and BadRequest("y").sid is None


def test_kvblob_wire_roundtrip_and_refusals():
    rows = (Rows("bf16", np.arange(8, dtype=np.uint16).reshape(4, 2)),
            Rows("bf16", np.arange(8, 16, dtype=np.uint16).reshape(4, 2)))
    b = KVBlob(3, 7, (5, 9), rows, kv_layout_id("m", (5, 9), "bf16", (1, 1, 1)), "nid")
    assert KVBlob.from_bytes(b.to_bytes()) == b
    assert b.pos_bytes(4) == rows[0].row_bytes(1) + rows[1].row_bytes(1)
    raw = b.to_bytes()
    for bad in (raw[:-1], raw + b"\0", b"XXXX" + raw[4:], raw[:6]):
        with pytest.raises(BadRequest):
            KVBlob.from_bytes(bad)
    with pytest.raises(BadRequest):
        KVBlob(3, 7, (9, 5), rows, "l", "n")                        # layers must ascend
    with pytest.raises(BadRequest):
        KVBlob(3, 8, (5, 9), rows, "l", "n")                        # row count != range
    base = kv_layout_id("m", (5, 9), "bf16", (4, 192, 128))
    assert base != kv_layout_id("m", (5, 9), "bf16", (8, 96, 128))     # same width, other heads
    assert base != kv_layout_id("m", (5, 9), "fp8_e4m3", (4, 192, 128))
    assert base != kv_layout_id("m", (5, 10), "bf16", (4, 192, 128))


def test_numerics_id_names_every_input():
    base = dict(backend="cuda", arch_class="sm_120", mode="invariant", kernel_pins=("k1",),
                quant="mxfp4", artifact_cid="cid")
    nid = make_numerics_id(**base)
    assert nid == make_numerics_id(**base) and nid.startswith("nid1-")
    for k, v in (("backend", "mlx"), ("arch_class", "sm_89"), ("mode", "fast"),
                 ("kernel_pins", ("k2",)), ("quant", "nvfp4"), ("artifact_cid", "cid2")):
        assert make_numerics_id(**{**base, k: v}) != nid, k


# ── the rebuild rule, checked on rf_check's independent toy (ARCHITECTURE-v2 Appendix B1) ────

_P = 1_000_003


def _rf_layer(kind, w, inp, kv, p):
    kv[p] = inp
    lo = 0 if kind == "G" else max(0, p - w + 1)
    acc = 0
    for q in range(lo, p + 1):
        if q not in kv:
            return None
        acc = (acc * 31 + kv[q] * (q % 7 + 1)) % _P
    return (inp * 17 + acc) % _P


def _rf_run(span, w, x, N, state=None, start=0, keep_below=0, partial=False):
    kvs = state if state is not None else [dict() for _ in span]
    outs = {}
    for p in range(start, N):
        h = x[p]
        for li, kind in enumerate(span):
            saved = kvs[li].get(p) if kind == "G" and p < keep_below else None
            r = _rf_layer(kind, w, h, kvs[li], p)
            if saved is not None:
                kvs[li][p] = saved
            if r is None:
                assert partial
                lo = 0 if kind == "G" else max(0, p - w + 1)
                acc = 0
                for q in range(lo, p + 1):
                    if q in kvs[li]:
                        acc = (acc * 31 + kvs[li][q] * (q % 7 + 1)) % _P
                r = (h * 17 + acc) % _P
            h = r
        outs[p] = h
    return outs, kvs


def _rf_exact(span, w, F, g, a, keep_mirror):
    N = F + 12
    x = [(i * 7919 + 13) % _P for i in range(N)]
    ref, ref_kv = _rf_run(span, w, x, N)
    st = [dict() for _ in span]
    for li, k in enumerate(span):
        if k == "G":
            st[li].update({q: ref_kv[li][q] for q in range(g)})
    _rf_run(span, w, x, F, state=st, start=a, keep_below=g if keep_mirror else a, partial=True)
    outs, _ = _rf_run(span, w, x, N, state=st, start=F)
    return all(outs[p] == ref[p] for p in range(F, N))


def test_rebuild_start_is_exact_on_the_rf_check_toy_and_the_first_draft_is_not():
    w = 4
    first_draft_wrong = 0
    for span in ["SSG", "SGS", "GSS", "SSSG", "SSGSS", "SSSSSG", "SGSGS"]:
        kinds = ["swa" if c == "S" else "global" for c in span]
        for F, g in [(60, 60), (60, 50), (60, 40)]:
            a = rebuild_start(kinds, w, F, g)
            assert _rf_exact(span, w, F, g, a, keep_mirror=True), (span, F, g, a)
            first = min(g, F - (w - 1) * span.count("S"))           # C13 first draft, overwriting
            first_draft_wrong += not _rf_exact(span, w, F, g, first, keep_mirror=False)
    assert first_draft_wrong == 18                                  # Appendix B1: 18/21 corrupt


def test_rebuild_start_edges():
    S, G = "swa", "global"
    assert rebuild_start([S, S, S], 128, 1000, 0) == 1000 - 3 * 127   # no global layer: no mirror needed
    assert rebuild_start([G, G], 128, 1000, 1000) == 1000             # all global, mirror current
    assert rebuild_start([G, G], 128, 1000, 900) == 900
    assert rebuild_start([S, G], 128, 100, 100) == 0                  # clamped
    with pytest.raises(BadRequest):
        rebuild_start([S, G], 128, 100, 101)                          # upto past F
    with pytest.raises(BadRequest):
        rebuild_start([S, G], 128, 100, 50, exact_from=101)
    a = rebuild_start([S, S, S], 128, 2000, 0, exact_from=2000 - 381)  # MTP-3: ~381 exact last-hidden rows
    assert a == 2000 - 381 - 3 * 127 and rebuild_exact_from([S, S, S], 128, a) == 2000 - 381
    assert rebuild_exact_from([S, G], 128, 0) == 0


# ── compatibility ─────────────────────────────────────────────────────────────────────────────

MODEL_RUNTIME_V1 = {
    "__init__": "(self, model: str, layer_range: shard.node.LayerRange, is_head: bool = False, "
                "is_tail: bool = False, device: str = 'cuda:0')",
    "load_shard": "(self) -> None",
    "reset": "(self) -> None",
    "heartbeat": "(self) -> dict",
    "embed": "(self, token_ids)",
    "forward": "(self, hidden_states, start_pos: int)",
    "logits": "(self, hidden_states)",
    "forward_prefill_stream": "(self, hidden_states, stream: int, start_pos: int)",
    "forward_decode_batch": "(self, hidden_states, starts)",
}


def test_model_runtime_v1_contract_is_untouched():
    """StageRuntime is a new contract next to ModelRuntime, not a change to it: MlxRuntime and
    its stub keep the exact v1 surface (tests/test_mlx_runtime.py runs unchanged)."""
    from shard.mlx_runtime import MlxRuntime
    from shard.mlx_stub import MlxRuntimeStub
    from shard.node import ModelRuntime
    got = {n: str(inspect.signature(getattr(ModelRuntime, n))) for n in MODEL_RUNTIME_V1}
    assert got == MODEL_RUNTIME_V1
    assert issubclass(MlxRuntime, ModelRuntime) and issubclass(MlxRuntimeStub, ModelRuntime)


def test_toy_satisfies_the_protocols():
    rt = ToyRuntime(StageSpec("toy", 0, 2), pattern="SG", window=2)
    assert isinstance(rt, StageRuntime)
    assert isinstance(ToyWeights("SG"), WeightSource)


def test_describe_before_load_sizes_the_span():
    """Placement sizes a span on a candidate device from describe() BEFORE loading it."""
    spec = StageSpec("toy", 1, 4, is_tail=True, taps=(2,), tap_mode="fc_partial")
    rt = ToyRuntime(spec, pattern="SGSG", window=2)
    before = rt.describe()
    rep = rt.load(ToyWeights("SGSG", taps=(2,)))
    assert before.weight_bytes == rep.n_bytes and before.kv_layout_id == rt.describe().kv_layout_id
