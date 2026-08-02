"""V4_FP8_GEMV / V4_FP8_SHARED can only serve what they have proven, or exactly the reference.

phase0/v4_fp8_gemv re-tiles the vendored fp8 GEMM where it is GEMV-shaped and fuses the shared
expert's w1/w3 into one banked launch — and refuses to take its own word for either: every tile and
every fused shape is gated at run time on `torch.equal` against the vendored path, or declines to
it. The numeric half of that bar runs a tilelang kernel and therefore lives in the
`@pytest.mark.hardware` tests at the bottom. What a CPU box CAN pin, and what this file pins:

  * the flag parses defensively — a bad string is IGNORED loudly, never an exception;
  * default OFF is really off: no mode, statuses read "off", nothing rebound, zero verdicts;
  * the auto tile always divides 128 (the weight-scale-group constraint) and leaves an
    already-healthy grid (>= 192 blocks) on the vendored kernel;
  * the CPU dispatch path returns before tile resolution is even consulted;
  * the shared bank layout: takes on an fp8 expert, declines bf16 / mis-shaped / twice-laid,
    aliases the SAME bytes (the reference path over the views is byte-identical), and the FUSED
    forward through the real bound `Expert.forward` is bit-exact to the reference on the CPU
    kernels — rewiring, bank views, quant-once and the SwiGLU transcription all under `torch.equal`;
  * unclaimed shapes (no bank, M > 32) reach the reference forward untouched;
  * both levers are registered: v4_levers knows the envs, the source scrape finds them, ENG_ENV
    carries them (test_v4_levers holds those globally; the checks' verdict shapes are pinned here).

Run: python3 -m pytest tests/test_v4_fp8_gemv.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase0"))

torch = pytest.importorskip("torch")
FG = pytest.importorskip("v4_fp8_gemv")
LEVERS = pytest.importorskip("v4_levers")


# ── parsing ──────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("s, want", [
    ("1", "auto"),
    ("auto", "auto"),
    ("64", (64, 4, 128)),                  # BN alone: stages/threads fill from the shipped tile
    ("32,2", (32, 2, 128)),
    ("16,4,256", (16, 4, 256)),
    ("128,6,512", (128, 6, 512)),
])
def test_a_good_flag_parses(s, want):
    mode, err = FG._parse_gemv(s)
    assert mode == want and err is None


@pytest.mark.parametrize("s", [
    "banana", "64,x", "48", "0x10", "-16", "16,0", "16,7", "16,2,31", "16,2,1024", "16,2,128,9",
    "256",                                  # does not divide 128: would span two scale groups
])
def test_a_bad_flag_is_refused_with_a_reason_never_an_exception(s):
    mode, err = FG._parse_gemv(s)
    assert mode is None and err, f"{s!r} must be refused with a reason"


def test_the_empty_default_is_off_not_invalid():
    assert FG._parse_gemv("") == (None, None)
    assert FG._parse_gemv("0") == (None, None)


# ── default off ──────────────────────────────────────────────────────────────────────────────────

def test_default_off_holds_nothing_and_reports_off():
    # The suite runs without the envs; a leaked env here would poison every numeric test.
    assert os.environ.get("V4_FP8_GEMV", "") in ("", "0")
    assert os.environ.get("V4_FP8_SHARED", "") in ("", "0")
    assert FG._GEMV_MODE is None and not FG.V4_FP8_SHARED
    assert FG.gemv_status() == "off" and FG.shared_status() == "off"
    assert FG._resolve_gemv(4096, 4096, "float32") is None
    assert FG._GEMV_VERDICTS == {}, "resolving with the lever off must not record a verdict"


def test_default_off_install_rebinds_nothing():
    import types

    class _Expert:
        def forward(self, x, weights=None):
            return x
    ref_fwd = _Expert.forward

    def gemm(a, a_s, b, b_s, sd=torch.float32):
        return None
    mod = types.SimpleNamespace(fp8_gemm=gemm, Expert=_Expert)
    took = FG.install(mod)
    assert took == (False, False)
    assert mod.fp8_gemm is gemm and mod.Expert.forward is ref_fwd


def test_shared_bank_layout_is_a_noop_under_default_off():
    class _M:
        shared_experts = object()          # would blow up if the layout ever touched it

        def modules(self):
            return [self]
    assert FG.shared_bank_layout(_M()) == 0


# ── the auto tile ────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("N, want_bn", [
    (32768, None),      # 256 blocks at the shipped tile: healthy, leave the vendored kernel alone
    (24576, None),      # exactly 192
    (12288, 64),        # 96 shipped -> 192 at bn=64
    (8192, 32),
    (4096, 16),
    (2048, 16),         # 128 blocks is the best 16 can do; still the narrowest legal tile
    (512, 16),
])
def test_the_auto_tile_targets_the_measured_block_plateau(N, want_bn):
    tile = FG._auto_tile(N)
    if want_bn is None:
        assert tile is None
    else:
        assert tile == (want_bn, 4, 128)
        assert 128 % tile[0] == 0, "a tile must not span two 128-row weight-scale groups"


# ── dispatch: CPU never resolves, verdicts are memoized, capture refuses ─────────────────────────

def test_the_cpu_dispatch_returns_before_tile_resolution(monkeypatch):
    def boom(*a):
        raise AssertionError("_resolve_gemv consulted on the CPU path")
    monkeypatch.setattr(FG, "_resolve_gemv", boom)
    calls = []

    def ref(a, a_s, b, b_s, sd=torch.float32):
        calls.append(a.shape)
        return torch.zeros(a.size(0), b.size(0))
    monkeypatch.setattr(FG, "_REF_FP8_GEMM", ref)
    a = torch.zeros(1, 256, dtype=torch.uint8).view(torch.float8_e4m3fn)
    out = FG._fp8_gemm_fast(a, torch.ones(1, 2), torch.zeros(128, 256, dtype=torch.uint8).view(
        torch.float8_e4m3fn), torch.ones(1, 2))
    assert calls == [(1, 256)] and out.shape == (1, 128)


def test_a_recorded_verdict_is_never_reprobed(monkeypatch):
    monkeypatch.setattr(FG, "_GEMV_MODE", "auto")
    monkeypatch.setattr(FG, "_GEMV_VERDICTS",
                        {(4096, 4096, "float32"): (16, 4, 128), (2048, 4096, "float32"): False})

    def boom(*a):
        raise AssertionError("probed a shape whose verdict is already in")
    monkeypatch.setattr(FG, "_probe_gemv", boom)
    assert FG._resolve_gemv(4096, 4096, "float32") == (16, 4, 128)
    assert FG._resolve_gemv(2048, 4096, "float32") is None


def test_a_healthy_shape_is_left_on_the_vendored_kernel(monkeypatch):
    monkeypatch.setattr(FG, "V4_FP8_GEMV", "auto")
    monkeypatch.setattr(FG, "_GEMV_MODE", "auto")
    monkeypatch.setattr(FG, "_GEMV_VERDICTS", {})

    def boom(*a):
        raise AssertionError("probed a shape whose shipped grid is already at the plateau")
    monkeypatch.setattr(FG, "_probe_gemv", boom)
    assert FG._resolve_gemv(32768, 1024, "float32") is None
    assert FG._GEMV_VERDICTS == {(32768, 1024, "float32"): "shipped"}
    assert FG.gemv_status() == "on/1-of-1", "leaving a healthy grid alone is the lever working"


# ── the shared expert on the CPU kernels: layout, views, and the fused forward ───────────────────

def _fp8_expert(mod, dim=256, inter=128, seed=0, swiglu_limit=10.0):
    """A shared-expert-shaped fp8 Expert at CPU-affordable dims, weights self-consistent."""
    import v4_moe_grouped
    torch.manual_seed(seed)
    with mod.set_dtype(torch.bfloat16):
        prev = mod.default_dtype
        mod.default_dtype = torch.float8_e4m3fn
        try:
            e = mod.Expert(dim, inter, swiglu_limit=swiglu_limit)
        finally:
            mod.default_dtype = prev
    with torch.no_grad():
        for lin, (o, i) in ((e.w1, (inter, dim)), (e.w3, (inter, dim)), (e.w2, (dim, inter))):
            w, s = v4_moe_grouped._fp8_block_quant(torch.randn(o, i, dtype=torch.bfloat16) * 0.05)
            lin.weight.data.copy_(w)
            lin.scale.data.copy_(s)
    return e.eval()


@pytest.fixture()
def ref_mod():
    import v4_ref_cpu
    return v4_ref_cpu.load_ref()


@pytest.fixture()
def shared_on(ref_mod, monkeypatch):
    """The lever ARMED on the CPU box: flag forced in the module, forward bound, module wired."""
    monkeypatch.setattr(FG, "V4_FP8_SHARED", True)
    ref_forward = FG._REF_EXPERT_FORWARD or ref_mod.Expert.forward
    if not getattr(ref_mod.Expert.forward, "_v4_fp8_shared", False):
        monkeypatch.setattr(FG, "_REF_EXPERT_FORWARD", ref_forward)
        monkeypatch.setattr(FG, "_MOD", ref_mod)
        monkeypatch.setattr(FG, "_REF_FP8_GEMM", ref_mod.fp8_gemm)
        monkeypatch.setattr(ref_mod.Expert, "forward", FG._shared_forward)
    return ref_mod


def test_the_layout_takes_on_fp8_and_declines_everything_else(ref_mod, monkeypatch):
    monkeypatch.setattr(FG, "V4_FP8_SHARED", True)
    e = _fp8_expert(ref_mod)
    assert FG._lay_shared(e)
    assert not FG._lay_shared(e), "never lay twice"
    with ref_mod.set_dtype(torch.bfloat16):
        bf = ref_mod.Expert(256, 128)      # bf16 weights: the CPU parity model's shape
    assert not FG._lay_shared(bf)
    odd = _fp8_expert(ref_mod, dim=256, inter=128)
    odd.w3.weight.data = torch.zeros(64, 256, dtype=torch.float8_e4m3fn)   # mismatched pair
    assert not FG._lay_shared(odd)


def test_the_views_alias_the_bank_and_keep_the_bytes(ref_mod, monkeypatch):
    monkeypatch.setattr(FG, "V4_FP8_SHARED", True)
    e = _fp8_expert(ref_mod, seed=1)
    w1_before = e.w1.weight.detach().view(torch.uint8).clone()
    w3_before = e.w3.weight.detach().view(torch.uint8).clone()
    s1_before = e.w1.scale.detach().view(torch.uint8).clone()
    assert FG._lay_shared(e)
    bank, sbank = e._v4_w13
    assert e.w1.weight.data_ptr() == bank.data_ptr()
    assert e.w3.weight.data_ptr() == bank[bank.size(0) // 2:].data_ptr()
    assert e.w1.weight.is_contiguous() and e.w3.weight.is_contiguous()
    assert torch.equal(e.w1.weight.view(torch.uint8), w1_before)
    assert torch.equal(e.w3.weight.view(torch.uint8), w3_before)
    assert torch.equal(e.w1.scale.view(torch.uint8), s1_before)
    # `linear()` reads the scale through `weight.scale`; the alias must survive the repoint
    assert e.w1.weight.scale is e.w1.scale
    assert e.w1.scale.data_ptr() == sbank.data_ptr()


def test_the_fused_forward_is_bit_exact_through_the_bound_class(shared_on):
    """The whole claim on the CPU kernels: same Expert weights, reference vs banked-and-fused,
    through the real `Expert.forward` binding, `torch.equal` — with and without routing weights,
    at M = 1 and the drafter's M = 5."""
    mod = shared_on
    ref_e = _fp8_expert(mod, seed=2)
    fus_e = _fp8_expert(mod, seed=2)
    assert FG._lay_shared(fus_e)
    torch.manual_seed(7)
    for m in (1, 5):
        for weights in (None, torch.rand(m, 1, dtype=torch.float32)):
            x = torch.randn(m, 256, dtype=torch.bfloat16) * 0.5
            with torch.no_grad():
                want = FG._REF_EXPERT_FORWARD(ref_e, x, weights)
                got = fus_e(x, weights)
            assert torch.equal(want, got), (m, weights is not None)
    assert fus_e._shared_steps == 4


def test_the_swiglu_clamp_branch_is_transcribed_too(shared_on):
    mod = shared_on
    ref_e = _fp8_expert(mod, seed=3, swiglu_limit=0.02)     # low limit: the clamps actually bite
    fus_e = _fp8_expert(mod, seed=3, swiglu_limit=0.02)
    assert FG._lay_shared(fus_e)
    x = torch.randn(1, 256, dtype=torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(FG._REF_EXPERT_FORWARD(ref_e, x), fus_e(x))


def test_unclaimed_shapes_reach_the_reference_untouched(shared_on, monkeypatch):
    mod = shared_on
    e = _fp8_expert(mod, seed=4)
    x33 = torch.randn(33, 256, dtype=torch.bfloat16)        # M > 32: prefill-shaped
    with torch.no_grad():
        want_nobank = FG._REF_EXPERT_FORWARD(e, x33.clone())
        got_nobank = e(x33.clone())                          # no bank yet: reference path
    assert torch.equal(want_nobank, got_nobank)
    assert FG._lay_shared(e)

    def boom(*a):
        raise AssertionError("the fused GEMM ran on an M>32 call")
    monkeypatch.setattr(FG, "_w13_gemm", boom)
    with torch.no_grad():
        got = e(x33)
    assert torch.equal(want_nobank, got)
    assert e._shared_declined == {"m>32": 1}, "a decline must be recorded, not silent"


def test_the_reference_path_over_banked_views_is_byte_identical(shared_on):
    """Prefill / fallbacks read w1/w3 through the bank views; same bytes in, same bytes out."""
    mod = shared_on
    ref_e = _fp8_expert(mod, seed=5)
    bank_e = _fp8_expert(mod, seed=5)
    assert FG._lay_shared(bank_e)
    x = torch.randn(40, 256, dtype=torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(FG._REF_EXPERT_FORWARD(ref_e, x), FG._REF_EXPERT_FORWARD(bank_e, x))


def test_stage_layout_walks_moes_by_duck_type(shared_on, monkeypatch):
    monkeypatch.setattr(FG, "V4_FP8_SHARED", True)
    holder = torch.nn.Module()
    holder.se = _fp8_expert(shared_on, seed=6)

    class _MoEish(torch.nn.Module):
        def __init__(self, se):
            super().__init__()
            self.shared_experts = se
    m = torch.nn.ModuleList([_MoEish(holder.se)])
    assert FG.shared_bank_layout(m) == 1
    assert FG.shared_bank_layout(m) == 0, "already banked: nothing to do"


# ── install / registration ───────────────────────────────────────────────────────────────────────

def test_install_is_idempotent_and_captures_the_reference_once(monkeypatch):
    import types

    class _Expert:
        def forward(self, x, weights=None):
            return x
    ref_fwd = _Expert.forward

    def gemm(a, a_s, b, b_s, sd=torch.float32):
        return None
    mod = types.SimpleNamespace(fp8_gemm=gemm, Expert=_Expert)
    monkeypatch.setattr(FG, "V4_FP8_SHARED", True)
    monkeypatch.setattr(FG, "_REF_FP8_GEMM", None)
    monkeypatch.setattr(FG, "_REF_EXPERT_FORWARD", None)
    took = FG.install(mod)
    assert took[1] is True
    assert FG._REF_FP8_GEMM is gemm and FG._REF_EXPERT_FORWARD is ref_fwd
    assert getattr(mod.Expert.forward, "_v4_fp8_shared", False)
    took2 = FG.install(mod)
    assert took2 == (False, False), "a second install must find the markers and do nothing"
    assert FG._REF_EXPERT_FORWARD is ref_fwd, "the captured reference must survive a re-install"


def test_both_levers_are_registered_with_the_right_owner():
    for env in ("V4_FP8_GEMV", "V4_FP8_SHARED"):
        lv = LEVERS.LEVERS_BY_ENV[env]
        assert lv.owner == "v4_fp8_gemv" and lv.side == LEVERS.STAGE


def test_the_gemv_check_tells_the_states_apart(monkeypatch):
    check = LEVERS.LEVERS_BY_ENV["V4_FP8_GEMV"].check
    ctx = LEVERS.Ctx(LEVERS.STAGE)
    ctx.mod = None
    monkeypatch.setattr(FG, "V4_FP8_GEMV", "")
    assert check(ctx)[:2] == ("off", "off")
    monkeypatch.setattr(FG, "V4_FP8_GEMV", "banana")
    monkeypatch.setattr(FG, "_GEMV_MODE", None)
    monkeypatch.setattr(FG, "_GEMV_INVALID", "not integers")
    req, obs, ok = check(ctx)
    assert ok is False, "set-but-unparsed is a dead knob and must be a finding"
    monkeypatch.setattr(FG, "V4_FP8_GEMV", "auto")
    monkeypatch.setattr(FG, "_GEMV_MODE", "auto")
    monkeypatch.setattr(FG, "_GEMV_VERDICTS", {})
    assert check(ctx) == ("on", "armed", None), "armed is unjudged, never OK"
    monkeypatch.setattr(FG, "_GEMV_VERDICTS", {(4096, 4096, "float32"): (16, 4, 128)})
    assert check(ctx) == ("on", "on/1-of-1", True)
    monkeypatch.setattr(FG, "_GEMV_VERDICTS", {(4096, 4096, "float32"): False})
    req, obs, ok = check(ctx)
    assert (obs, ok) == ("declined/1", False), "a decline must be visible on the audit line"


def test_the_shared_check_requires_a_nonempty_bank(monkeypatch):
    check = LEVERS.LEVERS_BY_ENV["V4_FP8_SHARED"].check

    class _Expert:
        forward = FG._shared_forward

    class _Mod:
        Expert = _Expert
    ctx = LEVERS.Ctx(LEVERS.STAGE)
    ctx.mod = _Mod

    class _Stage:
        _shared_banked = 0
    ctx.stage = _Stage()
    monkeypatch.setattr(FG, "V4_FP8_SHARED", True)
    monkeypatch.setattr(FG, "_SHARED_VERDICTS", {(256, 4096, "float32"): True})
    req, obs, ok = check(ctx)
    assert ok is False, "installed with an empty bank is not engaged (the grouped/0 bug)"
    ctx.stage._shared_banked = 8
    req, obs, ok = check(ctx)
    assert ok is True and obs == "on/1-of-1/banked-8"


# ── hardware: the numeric gates themselves (the ring runs these) ─────────────────────────────────

@pytest.mark.hardware
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_hw_every_shipped_shape_probes_equal_or_declines():
    import v4_moe_grouped
    mod = v4_moe_grouped._load_model_module()
    FG.install(mod)
    if FG._REF_FP8_GEMM is None:
        FG._REF_FP8_GEMM = mod.fp8_gemm
    for N, K in ((1024, 4096), (512, 4096), (4096, 8192), (8192, 1024),
                 (2048, 4096), (4096, 2048), (4096, 4096)):
        tile = FG._auto_tile(N)
        assert tile is not None
        ok, why = FG._probe_gemv(N, K, "float8_e8m0fnu", tile)
        assert ok, f"N={N} K={K} tile={tile}: {why}"


@pytest.mark.hardware
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_hw_the_fused_shared_launch_is_bit_exact_at_real_dims():
    import v4_moe_grouped
    mod = v4_moe_grouped._load_model_module()
    FG.install(mod)
    if FG._REF_FP8_GEMM is None:
        FG._REF_FP8_GEMM = mod.fp8_gemm
    ok, why = FG._probe_shared(4096, 4096, "float8_e8m0fnu")
    assert ok, why
