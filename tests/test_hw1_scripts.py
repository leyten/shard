"""H0: the HW1 measurement scripts run on CPU against stubbed ops at tiny dims and emit schema-valid
JSON; calib_fit round-trips a results set into both sims' --calib inputs and changes their output.

The model compute is stubbed the test_tail_churn way (a fake m25_stage module handed to the real
script code); everything the scripts decide -- grid, skip rules, graphed-flag recording, warm-up
discard, gates, the calib fit -- is the real code under test.

Run: OMP_NUM_THREADS=2 python3 -m pytest tests/test_hw1_scripts.py -q
"""
import json
import os
import shutil
import subprocess
import sys
import time
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HW1 = os.path.join(REPO, "research", "hw1")
sys.path.insert(0, HW1)
sys.path.insert(0, REPO)

import hw1_common as C  # noqa: E402
import m25_curve  # noqa: E402
import mimo_ops  # noqa: E402
import xfamily  # noqa: E402
import calib_fit  # noqa: E402
import collect  # noqa: E402
import sys_bench  # noqa: E402

FAST = ["--warmup", "2", "--iters", "3"]


# ── a tiny fake m25_stage (the only thing stubbed) ───────────────────────────────────────────────
def fake_stage(batch=4, maxlen=160, graph_alen_max=64):
    S = types.ModuleType("m25_stage")
    S.dev, S.H, S.NKV, S.HD = "cpu", 16, 2, 8
    S.M25_BATCH, S.M25_KV_MAXLEN, S.M25_KV_FP8, S.M25_BATCH_MOE = batch, maxlen, False, False
    S._GRAPH_COUNT = 0
    W = torch.randn(16, 16, generator=torch.Generator().manual_seed(0)) * 0.1
    f = lambda h: (h.float() @ W).to(h.dtype)                             # noqa: E731

    class Layer:
        def __init__(self, li):
            self.li = li
            self.bkc = torch.randn(batch, S.NKV, maxlen, S.HD).bfloat16()
            self.bvc = torch.randn(batch, S.NKV, maxlen, S.HD).bfloat16()

    class _Runner:
        def __init__(self, *a):
            self.graphs = {}

        def _bucket(self, total):
            return 1 << (int(total) - 1).bit_length()

        def _manual_ok(self, alen):
            return alen <= graph_alen_max

        def _run(self, total, x):
            alen = self._bucket(total)
            if self._manual_ok(alen) and alen not in self.graphs:
                self.graphs[alen] = True
                S._GRAPH_COUNT += 1
            return f(x)

    class BatchGraphRunner(_Runner):
        def run(self, starts, x):
            return self._run(int(starts.max()) + x.shape[1], x)

    class RowGraphRunner(_Runner):
        def run(self, row, start, x):
            return self._run(start + x.shape[1], x)

    S.Layer, S.BatchGraphRunner, S.RowGraphRunner = Layer, BatchGraphRunner, RowGraphRunner
    S.get_pe = lambda: None
    S.vllm_ctx = lambda: None
    S.run_block_decode_b = lambda layers, starts, h, vcfg: f(h)
    S.run_block_decode_row = lambda layers, row, start, h, vcfg: f(h)
    S.run_block_prefill_b = lambda layers, b, start, h, vcfg: f(h)
    return S


def load(p):
    d = json.load(open(p))
    assert C.validate(d) == [], C.validate(d)
    return d


# ── hw1_common ──────────────────────────────────────────────────────────────────────────────────
def test_time_fn_discards_warmup_and_validates():
    calls = []

    def op():
        calls.append(1)
        if len(calls) == 1:
            time.sleep(0.2)                    # a cold first call (JIT / autotune / alloc)

    st = C.time_fn(op, warmup=2, iters=3, device="cpu")
    assert len(calls) == 5 and st["warmup"] == 2 and st["n"] == 3
    assert st["ms"] < 50 and st["method"] == "perf_counter"
    with pytest.raises(ValueError):
        C.time_fn(op, warmup=0, iters=3, device="cpu")


def test_validate_rejects_meaningless_timings():
    r = C.Results("x", "y", device="cpu", stub=True)
    r.record("A", "op", {}, {"ms": 1.0, "method": "perf_counter", "warmup": 0, "n": 5})
    assert any("warm-up" in e for e in C.validate(r.doc))
    r.doc["records"] = [{"id": "A", "op": "op", "params": {}, "skipped": None, "ms": float("nan"),
                         "method": "perf_counter", "warmup": 3, "n": 5}]
    assert any("finite" in e for e in C.validate(r.doc))
    r.doc["records"] = [{"id": "A", "op": "op", "params": {}, "skipped": None, "ms": 1.0,
                         "method": "perf_counter", "warmup": 3, "n": 5}]
    r.doc["device"]["kind"] = "cuda"                       # a GPU number timed without cuda events
    assert any("cuda_event" in e for e in C.validate(r.doc))
    with pytest.raises(ValueError):
        r.write("/tmp/never.json")


# ── A0 anchor wrapper ───────────────────────────────────────────────────────────────────────────
TABLE = """=== per-STAGE (12L) decode ms/traversal ===
    ctx  B  per-stream ms  batched ms  speedup WAN-bound?
  16384  4 {a:>14.2f} {b:>11.2f}    1.60x        yes
  16384  8 {c:>14.2f} {d:>11.2f}    1.70x        yes
"""


def _fake_bench(tmp_path, a, b, c, d):
    p = tmp_path / "bench.py"
    p.write_text(f"print({TABLE.format(a=a, b=b, c=c, d=d)!r})\n")
    return f"{sys.executable} {p}"


def test_anchor_pass_and_stop(tmp_path):
    out = tmp_path / "a0.json"
    assert m25_curve.main(["--mode", "anchor", "--out", str(out), "--stub",
                           "--bench-cmd", _fake_bench(tmp_path, 43.0, 25.0, 70.0, 42.0)]) == 0
    d = load(out)
    assert [c["pass"] for c in d["checks"] if c["id"] == "K1"] == [True]
    assert m25_curve.main(["--mode", "anchor", "--out", str(out), "--stub",       # 30% slow -> stop
                           "--bench-cmd", _fake_bench(tmp_path, 60.0, 26.0, 71.0, 41.0)]) == 2
    assert [c["pass"] for c in load(out)["checks"] if c["id"] == "K1"] == [False]


# ── A1-A4 over the fake stage ───────────────────────────────────────────────────────────────────
def test_curve_grid_skips_and_graphed_flag(tmp_path):
    out = tmp_path / "a1.json"
    S = fake_stage(batch=4, maxlen=160, graph_alen_max=64)
    m25_curve.main(["--mode", "curve", "--ctxs", "32,120", "--bs", "1,2,8", "--layers", "2",
                    "--out", str(out), "--stub"] + FAST, stage=S)
    d = load(out)
    recs = [r for r in d["records"] if r["op"] == "decode_b"]
    skip = [r for r in recs if r["skipped"]]
    assert skip and all("M25_BATCH" in r["skipped"] for r in skip if r["params"]["B"] == 8)
    timed = [r for r in recs if not r["skipped"]]
    g_small = [r for r in timed if r["params"]["graphed_req"] and r["params"]["ctx"] == 32]
    g_big = [r for r in timed if r["params"]["graphed_req"] and r["params"]["ctx"] == 120]
    assert g_small and all(r["graphed"] for r in g_small)          # captured
    assert g_big and not any(r["graphed"] for r in g_big)          # over _manual_ok -> ran eager, SAYS so
    assert {r["params"]["moe"] for r in timed} == {"perstream", "batched"}
    mixed = [r for r in d["records"] if r["op"] == "decode_b_mixed"]
    assert mixed and all(r["skipped"] for r in mixed)
    m25_curve.main(["--mode", "curve", "--ctxs", "32", "--bs", "1", "--ss", "1", "--moe", "batched", "--graphed", "0",
                    "--layers", "1", "--out", str(out), "--stub"] + FAST, stage=fake_stage(batch=16))
    mixed = [r for r in load(out)["records"] if r["op"] == "decode_b_mixed"]
    assert mixed and all("clip" in r["skipped"] for r in mixed)      # maxlen 160: never a clipped 'mixed' number


def test_tau_prefill_kv(tmp_path):
    S = fake_stage(batch=2, maxlen=160)
    for mode, extra in (("tau", ["--ctxs", "64", "--ss", "1,2,9,17"]),
                        ("prefill", ["--chunks", "16,32", "--offsets", "0,64"]),
                        ("kv", ["--F", "128", "--deltas", "16,64,256"])):
        out = tmp_path / f"{mode}.json"
        assert m25_curve.main(["--mode", mode, "--layers", "2", "--out", str(out), "--stub"] + FAST + extra,
                              stage=S) == 0
        d = load(out)
        assert d["records"], mode
    tau = load(tmp_path / "tau.json")["records"]
    eager = {r["params"]["s"]: r for r in tau if not r["params"]["graphed_req"]}
    assert eager[1]["r_s"] == pytest.approx(1.0)
    kv = load(tmp_path / "kv.json")["records"]
    assert any(r["skipped"] for r in kv if r["params"]["delta"] == 256)          # delta > F
    exp = [r for r in kv if r["op"] == "kv_export" and not r["skipped"]]
    assert exp and all(r["bytes_per_token_layer"] == 2 * 2 * 8 * 2 for r in exp)  # K+V, NKV 2, HD 8, bf16


def test_m25_refuses_unbatched_stage(tmp_path):
    with pytest.raises(SystemExit):
        m25_curve.main(["--mode", "tau", "--out", str(tmp_path / "x.json")], stage=fake_stage(batch=1))


# ── B: MiMo ops ─────────────────────────────────────────────────────────────────────────────────
MIMO_FAST = ["--stub", "--tgrid", "1,9,900", "--bs", "1,2", "--ctxs", "48", "--ss", "1,9"] + FAST


def test_mimo_ops_stub(tmp_path):
    out = tmp_path / "b.json"
    assert mimo_ops.main(MIMO_FAST + ["--out", str(out)]) == 0
    d = load(out)
    assert [c["pass"] for c in d["checks"] if c["id"] == "sink_effect"] == [True]
    ops = {r["op"] for r in d["records"] if not r["skipped"]}
    assert {"qkv", "o_proj", "router", "experts", "attn", "attn_prefill", "probe_bytes",
            "footprint", "footprint_experts"} <= ops
    assert {r["params"]["layer"] for r in d["records"] if r["op"] == "experts"} == {1, 5}   # layer 0 dense
    assert d["derived"]["replay_tok_s_span"] > 0 and d["derived"]["layer_ms_T9"] > 0
    assert 0 < d["derived"]["replay_tok_s_T900_prefix8k"]                 # global layers at an 8k prefix
    assert d["derived"]["inputs"] == "random"


def test_sink_gate_catches_a_backend_that_ignores_sinks():
    class IgnoresSinks:                                    # flashinfer 0.6.12 plain run(sinks=) on sm_120
        backend = "mutant"

        def __call__(self, q, k, v, sinks, window, start):
            return mimo_ops.attn_ref(q, k, v, None, window, start)

    r = C.Results("t", "t", device="cpu", stub=True)
    assert mimo_ops.sink_effect_check(r, IgnoresSinks(), mimo_ops.TINY, "cpu") is False
    assert mimo_ops.sink_effect_check(r, mimo_ops.TorchAttn(), mimo_ops.TINY, "cpu") is True

    class WrongWindow:                                     # honours sinks but not the SWA window
        backend = "mutant2"

        def __call__(self, q, k, v, sinks, window, start):
            return mimo_ops.attn_ref(q, k, v, sinks, 0, start)

    assert mimo_ops.sink_effect_check(r, WrongWindow(), mimo_ops.TINY, "cpu") is False


def test_attn_ref_sink_equals_lse_postscale():
    g = torch.Generator().manual_seed(3)
    q, k, v = torch.randn(3, 4, 8, generator=g), torch.randn(10, 2, 8, generator=g), torch.randn(10, 2, 8, generator=g)
    s = torch.randn(4, generator=g)
    a = mimo_ops.attn_ref(q, k, v, s, 0, 7)
    b = mimo_ops.attn_ref(q, k, v, None, 0, 7)
    kk = k.repeat_interleave(2, 1)
    lg = torch.einsum("thd,shd->hts", q, kk) * 8 ** -0.5
    lg = lg.masked_fill(torch.arange(10)[None, None] > torch.arange(7, 10)[None, :, None], float("-inf"))
    lse = torch.logsumexp(lg, -1)                           # [h, t]
    scale = torch.sigmoid(lse - s[:, None]).t()[..., None]  # out * e^lse / (e^lse + e^sink)
    assert torch.allclose(a, b * scale, atol=1e-5)


def test_k3_shims_map_names_and_pass_spec(tmp_path):
    d = mimo_ops.TINY
    L = mimo_ops.load_layer(mimo_ops.write_synth(str(tmp_path / "w"), (1,), d), 1, d, "cpu")
    cfg, qc, get_tensor, env = mimo_ops.k3_shims(L, max_tokens=900)
    assert int(env["K3_MOE_MAX_TOKENS"]) >= 2048 and cfg.hidden_act == "silu"
    w = get_tensor("model.layers.1.mlp.experts.3.w2.weight_packed")
    assert w is L.raw["model.layers.1.mlp.experts.3.down_proj.weight"]
    assert get_tensor("x.experts.0.w3.weight_scale") is L.raw["model.layers.1.mlp.experts.0.up_proj.weight_scale"]
    sys.path.insert(0, os.path.join(REPO, "engines", "kimi_k3"))
    K = pytest.importorskip("k3_moe_mxfp4")
    assert K.spec(qc)["group_size"] == 32


def test_qkv_load_matches_rechunk(tmp_path):
    d = mimo_ops.TINY
    L = mimo_ops.load_layer(mimo_ops.write_synth(str(tmp_path / "w"), (0, 1), d), 1, d, "cpu")
    assert L.kind == "swa" and L.nkv == d["kv_swa"] and L.sinks is not None
    assert L.wqkv.shape == (d["heads"] * d["qk"] + L.nkv * (d["qk"] + d["v"]), d["hidden"])
    L0 = mimo_ops.load_layer(str(tmp_path / "w"), 0, d, "cpu")
    assert L0.kind == "full" and L0.sinks is None and not L0.moe


# ── D: cross-family diff ────────────────────────────────────────────────────────────────────────
def test_xfamily_dump_and_compare(tmp_path):
    a_json, b_json = tmp_path / "a.json", tmp_path / "b.json"
    for p in (a_json, b_json):
        assert mimo_ops.main(["--stub", "--mode", "dump", "--dump", str(p) + ".st", "--out", str(p)]) == 0
    same = xfamily.compare(load(a_json), load(b_json))
    assert all(r["bit_equal"] for r in same["records"] if r["op"] == "xfamily_diff")
    from safetensors.torch import load_file, save_file   # perturb B: a different family's rounding
    t = load_file(str(b_json) + ".st")
    t["L1.out"] = (t["L1.out"].float() * 1.01).bfloat16()
    save_file(t, str(b_json) + ".st")
    ops = tmp_path / "ops.json"
    mimo_ops.main(MIMO_FAST + ["--out", str(ops)])
    out = tmp_path / "x.json"
    xfamily.main(["--a", str(a_json), "--b", str(b_json), "--a-ops", str(ops), "--b-ops", str(ops),
                  "--out", str(out)])
    d = load(out)
    diff = {r["params"]["tensor"]: r for r in d["records"] if r["op"] == "xfamily_diff"}
    assert not diff["L1.out"]["bit_equal"] and diff["L1.out"]["rel"] > 0 and diff["L0.out"]["bit_equal"]
    assert any(r["op"] == "xfamily_router" and r["topk_set_agreement"] == 1.0 for r in d["records"])
    sp = [r for r in d["records"] if r["op"] == "xfamily_speed"]
    assert sp and all(r["ratio_b_over_a"] > 0 for r in sp)
    chk = {c["id"]: c for c in d["checks"]}
    assert chk["K10"]["pass"] and chk["K10"]["rel"] > 0 and chk["K11"]["ratio"] == pytest.approx(1.0)
    t["L1.out"] = (t["L1.out"].float() * 1.2).bfloat16()               # a real cross-family break
    save_file(t, str(b_json) + ".st")
    assert xfamily.compare(load(a_json), load(b_json))["checks"][-1]["pass"] is False


# ── C: system ───────────────────────────────────────────────────────────────────────────────────
def test_sys_bench_cpu(tmp_path):
    out = tmp_path / "c.json"
    sys_bench.main(["--device", "cpu", "--stub", "--rows", "32", "--ns", "1,9", "--hidden", "256",
                    "--out", str(out)] + FAST)
    d = load(out)
    assert {r["op"] for r in d["records"] if not r["skipped"]} == {"sha256", "frame_rt"}
    assert all(r["skipped"] == "no cuda" for r in d["records"] if r["op"] in ("d2h", "h2d"))


# ── calib round trip into both sims ─────────────────────────────────────────────────────────────
def _synthetic_results(tmp_path):
    """A plausible measured set (numbers are synthetic, shapes are the real schema)."""
    paths = []

    def rec(script, mode, fill):
        r = C.Results(script, mode, device="cpu", stub=True)
        fill(r)
        p = str(tmp_path / f"{script}_{mode}.json")
        r.write(p)
        paths.append(p)

    T = lambda ms: {"ms": ms, "ms_p10": ms, "ms_p90": ms, "ms_mean": ms, "n": 5, "warmup": 3,  # noqa: E731
                    "method": "perf_counter", "wall_ms_per_iter": ms}

    def tau(r):
        for s, ms in ((1, 4.0), (2, 4.4), (4, 5.0), (9, 6.4), (17, 9.0)):
            r.record("A2", "decode_row", {"ctx": 8192, "s": s, "graphed_req": True, "layers": 8}, T(ms),
                     graphed=True, r_s=ms / 4.0, ms_per_layer=ms / 8)

    def curve(r):
        for ctx in (1024, 32768):
            for B in (1,):
                ms = 3.0 if ctx == 1024 else 4.5
                r.record("A1", "decode_b", {"ctx": ctx, "B": 1, "s": 1, "moe": "perstream", "graphed_req": False,
                                            "layers": 8}, T(ms), graphed=False, ms_per_layer=ms / 8)
        for B, ms1, ms9 in ((1, 5.0, 8.0), (4, 9.0, 14.0), (16, 20.0, 32.0), (32, 34.0, 60.0)):
            for s, ms in ((1, ms1), (9, ms9)):
                r.record("A1", "decode_b", {"ctx": 8192, "B": B, "s": s, "moe": "batched", "graphed_req": False,
                                            "layers": 8}, T(ms), graphed=False, ms_per_layer=ms / 8)

    def kv(r):
        r.record("A4", "kv_import", {"F": 32768, "delta": 32768, "layers": 8}, T(40.0), bytes=1, gb_per_s=12.0)

    def mimo(r):
        r.doc["derived"].update(replay_tok_s_span=2500.0, layer_ms_T9=1.6)

    def sysb(r):
        r.record("C", "sha256", {"kib": 8, "rows": 2048}, T(1.0), us_per_row=3.0)
        r.record("C", "frame_rt", {"n": 9, "hidden": 4096}, T(0.05), us_per_msg=50.0)

    def anchor(r):
        r.check("K1", True, "synthetic")

    rec("m25_curve", "tau", tau)
    rec("m25_curve", "curve", curve)
    rec("m25_curve", "kv", kv)
    rec("m25_curve", "anchor", anchor)
    rec("mimo_ops", "ops", mimo)
    rec("sys_bench", "sys", sysb)
    return paths


def test_calib_fit_gates_and_refuses_stub(tmp_path):
    paths = _synthetic_results(tmp_path)
    out = tmp_path / "calib.json"
    with pytest.raises(SystemExit):
        calib_fit.main(paths + ["--out", str(out)])            # stub numbers are never a calibration
    calib_fit.main(paths + ["--out", str(out), "--allow-stub"])
    cal = json.load(open(out))
    p, g = cal["pipespec"], cal["gates"]
    assert p["r9"] == pytest.approx(1.6) and p["m25_ms_per_layer_s9"] == pytest.approx(0.8)
    rows = {int(k): v for k, v in p["rows_table"].items()}
    assert rows[1] == 1.0 and rows[9] == pytest.approx(1.6) and max(rows) == 32 * 9   # A1 spliced on
    assert p["kfac"] == pytest.approx(1.6 / 1.0)                                       # 8 ms / 8 L
    assert g["K1"]["pass"] and g["K3"]["pass"] and g["K6"]["pass"]
    assert g["K2"]["value"] == pytest.approx((144 / 32) / (9 / 8))                     # 4.0x
    assert g["K4"]["value"] == pytest.approx(1.5) and g["K5"]["pass"] is False
    assert g["K9"]["pass"] and g["K8"]["value"] > 0
    assert cal["mesh"]["replay_rate_tok_s"] == 2500.0 and cal["stub"] is True
    assert g["K12"]["pass"] is False and g["K12"]["value"] == {"cpu": False}   # torch fallback != fast path


def test_calib_changes_both_sims(tmp_path):
    paths = _synthetic_results(tmp_path)
    cal = tmp_path / "calib.json"
    calib_fit.main(paths + ["--out", str(cal), "--allow-stub"])
    tools = os.path.join(REPO, "tools")
    env = dict(os.environ, OMP_NUM_THREADS="2")
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=tools, env=env, capture_output=True,  # noqa: E731
                                    text=True, timeout=300)
    m0, m1 = run("mesh_calc.py"), run("mesh_calc.py", "--calib", str(cal))
    assert m0.returncode == 0 and m1.returncode == 0, m1.stderr
    assert "MEAS mirror import" in m1.stdout and " 2500 " in m1.stdout and m0.stdout != m1.stdout
    p0, p1 = run("pipespec_sim.py", "--part", "mimo"), run("pipespec_sim.py", "--part", "mimo", "--calib", str(cal))
    assert p0.returncode == 0 and p1.returncode == 0, p1.stderr
    assert "CALIBRATED" in p1.stdout and "(STUB DATA)" in p1.stdout
    tbl = lambda s: [ln for ln in s.splitlines() if ln.strip()[:1].isdigit()]  # noqa: E731
    assert tbl(p0.stdout) and tbl(p0.stdout) != tbl(p1.stdout)


# ── collect + bootstrap ─────────────────────────────────────────────────────────────────────────
def test_collect_bundle(tmp_path):
    paths = _synthetic_results(tmp_path)
    out = tmp_path / "bundle.json"
    assert collect.main([str(tmp_path), "--out", str(out)]) == 0
    b = json.load(open(out))
    assert b["ok"] and b["stub"] and len(b["files"]) == len(paths)
    (tmp_path / "bad.json").write_text(json.dumps({"schema": "nope"}))
    assert collect.main([str(tmp_path), "--out", str(out)]) == 1


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_bootstrap_shellcheck():
    p = subprocess.run(["shellcheck", os.path.join(HW1, "bootstrap.sh")], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout
