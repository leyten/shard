"""The grouped MoE decode path is opt-in and defers cleanly, or it does not ship.

phase0/v4_moe_grouped replaces the routed-expert GEMMs of a decode step with ONE grouped fp4 launch.
Its numeric bar is `torch.equal` against the reference MoE — but that runs a tilelang kernel, which
is CUDA-only, so the parity proof lives on the GPU (`phase0/v4_moe_grouped.py::_smoke` and the bench
harness on the ring's tail box). What CI on a CPU box CAN pin, and what this file pins, is the
envelope around that kernel:

  * the module imports without dragging in a CUDA toolchain (tilelang is deferred into the builder),
  * it installs NOTHING unless `V4_MOE_GROUPED=1` AND a CUDA device is present, and
  * every shape it does not claim falls through to the forward it captured, untouched.

The numeric parity test is GPU-gated and skips here. Run: python3 -m pytest tests/test_v4_moe_grouped.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
REFCPU = pytest.importorskip("v4_ref_cpu")
GROUPED = pytest.importorskip("v4_moe_grouped")

SEED = 5


@pytest.fixture(scope="module")
def args():
    return REFCPU.cpu_args()


@pytest.fixture(scope="module")
def oracle(args):
    return REFCPU.build_oracle(args, SEED)


def _x(args, s, g, oracle):
    dt = oracle.layers[0].ffn.gate.weight.dtype
    return torch.randn(1, s, args.dim, generator=g).to(dt)


def _ids(args, s, g):
    return torch.randint(0, args.vocab_size, (1, s), generator=g)


def _spy(monkeypatch, world_size=1):
    """Point the module's fallback at a spy and pin world_size; return the call log."""
    calls = []
    monkeypatch.setattr(GROUPED, "_REF_FORWARD",
                        lambda s, x, i: (calls.append(1), torch.zeros_like(x))[1])
    monkeypatch.setattr(GROUPED, "_MOD", sys.modules["dsv4_model"])
    monkeypatch.setattr(GROUPED, "_WORLD_SIZE", world_size)
    return calls


def test_imports_without_cuda_toolchain():
    """Import must not pull in tilelang — the CUDA codegen is deferred into the kernel builder, so a
    CPU box (and `pytest --collect-only`) can import and reason about the module for free."""
    assert "tilelang" not in sys.modules
    assert callable(GROUPED.grouped_fp4_gemm_kernel)
    assert callable(GROUPED.grouped_forward)


def test_default_off_installs_nothing(oracle):
    """Env unset -> the module is inert: install() no-ops and leaves the forward the default stack
    already has (v4_moe_decode's), so serving is byte-identical to today. Takes `oracle` so the
    model module is loaded (it registers `dsv4_model`) before we reach for it."""
    assert GROUPED.V4_MOE_GROUPED is False
    mod = sys.modules["dsv4_model"]
    before = mod.MoE.forward
    assert GROUPED.install(mod) is False
    assert mod.MoE.forward is before


def test_install_refuses_without_cuda(oracle, monkeypatch):
    """Flag on but no CUDA device -> still a no-op. The kernel is CUDA-only; installing on a CPU box
    would only defer the JIT failure to layer 0 of a real run. Takes `oracle` so `dsv4_model` is
    loaded before we reach for it."""
    if torch.cuda.is_available():
        pytest.skip("CUDA present — the no-CUDA refusal cannot be exercised on this box")
    mod = sys.modules["dsv4_model"]
    monkeypatch.setattr(GROUPED, "V4_MOE_GROUPED", True)
    before = mod.MoE.forward
    assert GROUPED.install(mod) is False
    assert mod.MoE.forward is before


def test_multi_token_falls_back(args, oracle, monkeypatch):
    """Prefill and a speculation chunk are s > 1 — grouped-and-padded MoE is not token-count
    invariant, so the fast path must not claim them."""
    ffn = oracle.layers[-1].ffn                       # score-routed (layer >= n_hash_layers)
    assert not ffn.gate.hash
    calls = _spy(monkeypatch)
    g = torch.Generator().manual_seed(SEED)
    x, ids = _x(args, 5, g, oracle), _ids(args, 5, g)
    GROUPED.grouped_forward(ffn, x, ids)
    assert calls, "s > 1 must take the reference path"


def test_world_size_falls_back(args, oracle, monkeypatch):
    """world_size > 1 -> the reference all-reduces the routed sum across ranks before the shared
    expert; the grouped path has no all_reduce, so it must defer rather than drop a rank's experts."""
    ffn = oracle.layers[-1].ffn
    assert not ffn.gate.hash
    calls = _spy(monkeypatch, world_size=2)
    g = torch.Generator().manual_seed(SEED + 1)
    x, ids = _x(args, 1, g, oracle), _ids(args, 1, g)
    GROUPED.grouped_forward(ffn, x, ids)
    assert calls, "world_size > 1 must take the reference path"


def test_hash_layer_falls_back(args, oracle, monkeypatch):
    """Hash-routed layers (layer_id < n_hash_layers) can name the same expert twice, and the
    duplicate-index `y[idx] +=` is the reference's semantics — so the whole hash layer defers. This
    is also the ONLY place a repeat can occur (top-k routing cannot), so it subsumes the
    repeated-expert fallback without a per-step host-side dedup."""
    ffn = oracle.layers[0].ffn                        # hash-routed (layer 0 < n_hash_layers)
    assert ffn.gate.hash
    calls = _spy(monkeypatch)
    g = torch.Generator().manual_seed(SEED + 2)
    x, ids = _x(args, 1, g, oracle), _ids(args, 1, g)
    GROUPED.grouped_forward(ffn, x, ids)
    assert calls, "a hash-routed layer must take the reference path"


def _stub_mod():
    """The two things install() touches: a class with a `forward`, and `world_size`. Hermetic on
    purpose — rebinding the real `dsv4_model.MoE.forward` would follow every later test in the run."""
    ref = lambda self, x, ids: "reference"                 # noqa: E731 — identity is what we compare
    return type("stub", (), {"MoE": type("MoE", (), {"forward": ref}), "world_size": 1})


def test_install_order_is_the_precedence_grouped_over_decode(monkeypatch):
    """The wiring `v4_ref_cpu.load_ref()` implements, pinned as a contract.

    Each install captures whatever `MoE.forward` is bound AT THAT MOMENT as its own fallback, so the
    order of the two calls IS the precedence. Grouped second => grouped -> decode -> reference: the
    grouped kernel claims the single-token score-routed decode step and hands back everything it
    declines. Grouped FIRST would bury it under the decode path, which claims that same step and would
    therefore never fall through — the lever would be installed and dead."""
    DECODE = pytest.importorskip("v4_moe_decode")
    monkeypatch.setattr(DECODE, "V4_MOE_DECODE", True)
    monkeypatch.setattr(GROUPED, "V4_MOE_GROUPED", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)   # never used: nothing is CALLED here
    for m, names in ((DECODE, ("_REF_FORWARD", "_WORLD_SIZE")),
                     (GROUPED, ("_REF_FORWARD", "_MOD", "_WORLD_SIZE"))):
        for n in names:                                    # monkeypatch restores the module globals
            monkeypatch.setattr(m, n, getattr(m, n))

    mod = _stub_mod()                                      # shipped order: decode, then grouped
    reference = mod.MoE.forward
    assert DECODE.install(mod) is True
    assert mod.MoE.forward is DECODE.decode_forward
    assert GROUPED.install(mod) is True
    assert mod.MoE.forward is GROUPED.grouped_forward, "grouped must own the decode step"
    assert GROUPED._REF_FORWARD is DECODE.decode_forward, "grouped's fallback is the decode path"
    assert DECODE._REF_FORWARD is reference, "decode's fallback is still the untouched reference"

    rev = _stub_mod()                                      # the order that would silently kill it
    assert GROUPED.install(rev) is True
    assert DECODE.install(rev) is False, "decode must refuse to install OVER grouped (cycle guard)"
    assert rev.MoE.forward is GROUPED.grouped_forward


def test_decode_refuses_to_install_over_grouped_and_make_a_cycle(monkeypatch):
    """The trap the shipped order creates. Grouped captures `decode_forward` as its fallback; if
    decode then re-installs it would capture `grouped_forward`, and the two would call each other
    until the stack blows on the first prefill. load_ref runs the pair once so nothing reaches it
    today — the guard is for the second install path that does not exist yet."""
    DECODE = pytest.importorskip("v4_moe_decode")
    monkeypatch.setattr(DECODE, "V4_MOE_DECODE", True)
    monkeypatch.setattr(GROUPED, "V4_MOE_GROUPED", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for m, names in ((DECODE, ("_REF_FORWARD", "_WORLD_SIZE")),
                     (GROUPED, ("_REF_FORWARD", "_MOD", "_WORLD_SIZE"))):
        for n in names:
            monkeypatch.setattr(m, n, getattr(m, n))

    mod = _stub_mod()
    DECODE.install(mod)
    GROUPED.install(mod)
    assert DECODE.install(mod) is False, "a second decode install would close the loop"
    assert GROUPED._REF_FORWARD is DECODE.decode_forward
    assert DECODE._REF_FORWARD is not GROUPED.grouped_forward, "the cycle must not be armed"


def test_bank_declines_rather_than_ooms_when_it_would_not_fit(monkeypatch):
    """The bank is an exact DUPLICATE of a layer's routed-expert weights (~3.2 GiB at shipped dims,
    beside ~3.7 GiB still live), built lazily on the first decode token — after the graph pools are
    pinned. Nothing upstream catches an OOM out of `ffn`, so a stage that tried and failed would DIE
    and cascade the ring. It must decline instead: once, loudly, and permanently for that layer."""
    calls = []

    class _W:
        def __init__(self):
            self.weight = torch.zeros(4, 4, dtype=torch.uint8)
            self.scale = torch.zeros(1, 1)

    class _E:
        def __init__(self):
            self.w1, self.w2, self.w3 = _W(), _W(), _W()

    moe = type("MoE", (), {})()
    moe.experts_start_idx, moe.experts_end_idx, moe.layer_id = 0, 2, 7
    moe.experts = [_E(), _E()]
    monkeypatch.setattr(GROUPED, "_bank_fits", lambda experts: (calls.append(1), False)[1])
    assert GROUPED._expert_bank(moe) is None, "a bank that does not fit must not be built"
    assert moe._grouped_bank is False, "the refusal is cached, not re-decided every token"
    assert GROUPED._expert_bank(moe) is None
    assert len(calls) == 1, "the fit check must run ONCE per layer, not per decode step"

    monkeypatch.setattr(GROUPED, "_bank_fits", lambda experts: True)
    fresh = type("MoE", (), {})()
    fresh.experts_start_idx, fresh.experts_end_idx, fresh.layer_id = 0, 2, 7
    fresh.experts = [_E(), _E()]
    assert GROUPED._expert_bank(fresh) is not None, "a bank that fits is still built"


def test_bank_fit_check_leaves_headroom():
    """The rule itself: it is not "does it fit", it is "does it fit and leave room to keep serving" —
    the KV cache is still growing and the per-step gathers still need somewhere to land."""
    assert GROUPED._BANK_HEADROOM_BYTES >= (1 << 30)
    if not torch.cuda.is_available():
        assert GROUPED._bank_fits([]) is True, "off-CUDA there is no bound to check"


def test_both_flags_off_leaves_the_reference_forward(monkeypatch):
    """The default composition: neither lever installs, so `load_ref()` hands back a byte-identical
    reference no matter how many install() calls are wired into it."""
    DECODE = pytest.importorskip("v4_moe_decode")
    monkeypatch.setattr(DECODE, "V4_MOE_DECODE", False)
    monkeypatch.setattr(GROUPED, "V4_MOE_GROUPED", False)
    mod = _stub_mod()
    reference = mod.MoE.forward
    assert DECODE.install(mod) is False and GROUPED.install(mod) is False
    assert mod.MoE.forward is reference


def test_load_ref_installs_grouped_after_decode():
    """The seam itself: `v4_ref_cpu.load_ref()` must actually CALL grouped's install, in that order.
    A lever nothing reaches is not a lever, and the CPU box cannot prove this by running the kernel."""
    import inspect
    src = inspect.getsource(REFCPU.load_ref)
    assert "v4_moe_grouped.install(mod)" in src, "load_ref never installs the grouped kernel"
    assert src.index("v4_moe_decode.install(mod)") < src.index("v4_moe_grouped.install(mod)"), \
        "grouped must install AFTER decode or it is buried under it (see the precedence test)"


@pytest.mark.hardware
def test_bit_exact_on_gpu(monkeypatch):
    """The real bar. GPU-only: builds V4's shipped-dims MoE with random fp4 weights and checks
    `torch.equal` between the grouped path and the reference over several routings."""
    if not torch.cuda.is_available():
        pytest.skip("grouped MoE parity needs a CUDA device (tilelang)")
    mod = GROUPED._load_model_module()
    args = GROUPED.real_dims_args(mod)
    moe = GROUPED.build_real_dims_moe(mod, args)
    ref_forward = mod.MoE.forward
    monkeypatch.setattr(GROUPED, "_MOD", mod)
    monkeypatch.setattr(GROUPED, "_WORLD_SIZE", 1)
    for t in range(6):
        x = torch.randn(1, 1, args.dim, dtype=torch.bfloat16, device="cuda")
        ids = torch.randint(0, args.vocab_size, (1, 1), device="cuda")
        with torch.no_grad():
            ref = ref_forward(moe, x, ids)
            got = GROUPED.grouped_forward(moe, x, ids)
        assert torch.equal(ref, got), f"draw {t}: max|d| = {(ref.float() - got.float()).abs().max()}"
