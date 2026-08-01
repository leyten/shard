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
