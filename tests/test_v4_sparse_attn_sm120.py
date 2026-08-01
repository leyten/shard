"""The sm_120 sparse-attention retile, as far as a CPU box can grade it.

phase0/v4_sparse_attn_sm120.py exists because V4's real attention shape (h=64, d=512) asks the
vendored tilelang kernel for 141312 B of dynamic shared memory and consumer Blackwell caps at
101376. The kernel itself can only be graded on a GPU -- that receipt is measured on a 5090 and
banked in docs/receipts/v4-sm120-sparse-attn-20260801.json. What a CPU box CAN pin, and what
therefore lives here, is everything AROUND the kernel:

  the shared-memory model    it reproduces every launch-failure number tilelang printed on the box,
                             including the vendored 141312. If this arithmetic drifts, `choose_tile`
                             silently picks a tiling that cannot launch.
  the tile choice            a fitting device must be handed the VENDORED tiling, not ours.
  install_sm120's no-ops     no CUDA => nothing rebinds, nothing imports tilelang, nothing raises.
  the eager reference        it has to BE v4_kernels_cpu.sparse_attn, not a copy of it, or the A/B
                             on the box measures two transcriptions against each other.

Run: python3 -m pytest tests/test_v4_sparse_attn_sm120.py -q   (CPU-only, no GPU needed)
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase0"))
import v4_kernels_cpu as K                                                            # noqa: E402
import v4_sparse_attn_sm120 as S                                                      # noqa: E402

SM120_OPTIN = 101376        # shared_memory_per_block_optin on a 5090 and on a Pro 6000
HOPPER_OPTIN = 232448       # ... and on an H100/B200, where the vendored kernel fits


def test_module_imports_without_tilelang():
    """Importing this module must not drag in tilelang -- a CPU box has no CUDA toolchain.

    tilelang is imported inside `sparse_attn_kernel`, which is the only function that needs it.
    A module-scope import would make every CPU parity test depend on a GPU compiler."""
    assert "tilelang" not in sys.modules, "v4_sparse_attn_sm120 imported tilelang at module scope"
    assert callable(S.sparse_attn) and callable(S.sparse_attn_eager) and callable(S.install_sm120)


def test_smem_model_reproduces_the_observed_launch_failures():
    """The four numbers tilelang printed on the 5090, from `kernel_smem` alone.

    141312 is the failure this whole module exists for (the vendored 64x512 tiling at block=64,
    threads=256). The other three are configs swept on the box; they pin the two terms the model
    could plausibly get wrong -- that o_shared ALIASES q_shared rather than adding to it, and that
    the cross-warp reduction workspace is 8 B/thread."""
    assert S.kernel_smem(64, 64, 512, 256) == 141312, "vendored tiling at V4's real shape"
    assert S.kernel_smem(16, 128, 512, 256) == 153600
    assert S.kernel_smem(32, 64, 512, 256) == 104448
    assert S.kernel_smem(32, 64, 512, 128) == 103424


def test_shipped_tiles_fit_sm120_and_are_legal_for_tilelang():
    """Every TILES entry must fit 99 KiB AND satisfy tilelang's two compile-time asserts.

    h_block below the m16n8k16 atom is `M must be divisible by 16`, and a block/warp column tile
    that is not a multiple of 8 is `warp_col_tiles must be divisible by 8` -- both are JIT errors,
    i.e. they would surface on the ring at layer 0 rather than here."""
    assert S.TILES, "no tilings shipped"
    for h_block, block, threads in S.TILES:
        need = S.kernel_smem(h_block, block, 512, threads)
        assert need <= SM120_OPTIN, f"tile ({h_block},{block},{threads}) needs {need} B > {SM120_OPTIN}"
        assert h_block % S.M_ATOM == 0, f"h_block {h_block} is not a multiple of the MMA atom"
        warps = threads // 32
        assert block % warps == 0 and (block // warps) % 8 == 0, \
            f"tile ({h_block},{block},{threads}) gives warp_col_tiles {block / warps}"


def test_choose_tile_keeps_the_vendored_tiling_where_it_fits():
    """A big-shared-memory device gets DeepSeek's own tiling back, not ours."""
    assert S.choose_tile(64, 512, HOPPER_OPTIN) == (64, S.VENDORED_BLOCK, S.VENDORED_THREADS)
    assert S.choose_tile(64, 512, SM120_OPTIN) == S.TILES[0]
    # a toy shape fits anywhere, and h < 16 is rounded up to the MMA atom exactly as kernel.py's
    # own wrapper pads it -- a 4-head kernel does not compile.
    assert S.choose_tile(4, 128, SM120_OPTIN) == (16, S.VENDORED_BLOCK, S.VENDORED_THREADS)
    with pytest.raises(RuntimeError, match="no tiling"):
        S.choose_tile(64, 512, 32768)


def test_choose_tile_env_override():
    """V4_SM120_TILE forces a tiling for a sweep, over both the fit test and the candidate list."""
    os.environ[S.TILE_ENV] = "16,32,128"
    try:
        assert S.choose_tile(64, 512, HOPPER_OPTIN) == (16, 32, 128)
    finally:
        del os.environ[S.TILE_ENV]


def test_padded_heads_matches_the_vendored_wrapper():
    """kernel.py pads h to 16 and strips the output back; ours pads to a multiple of h_block."""
    assert [S.padded_heads(h) for h in (1, 4, 15, 16, 17, 64)] == [16, 16, 16, 16, 32, 64]
    assert S.padded_heads(64, 16) == 64


def test_install_sm120_noops_without_cuda():
    """No CUDA => no rebind, no tilelang import, no raise. This is the CPU suite's own process."""
    if torch.cuda.is_available():
        pytest.skip("CUDA present; this asserts the CPU box's path")
    K.install()                                   # puts the cpu shim under `kernel`
    before = sys.modules["kernel"].sparse_attn
    assert S.install_sm120() is None
    assert sys.modules["kernel"].sparse_attn is before, "install_sm120 rebound the CPU shim"
    assert "tilelang" not in sys.modules


def test_load_ref_hook_leaves_the_cpu_backend_alone():
    """v4_ref_cpu.load_ref()'s hook must be invisible on the CPU backend.

    The reference resolves `from kernel import ... sparse_attn ...` at module scope, so the hook has
    to run BEFORE the exec -- which means an over-eager hook would swap the CPU stand-in out from
    under every parity test in this repo. It runs only when backend() == 'tilelang'."""
    import v4_ref_cpu as R
    ref = R.load_ref()
    assert ref.sparse_attn is K.sparse_attn, "the CPU oracle is not running the CPU sparse_attn"


def test_eager_reference_is_the_cpu_stand_in_itself():
    """`sparse_attn_eager` must not be a second transcription -- bitwise, on real tensors.

    The GPU A/B compares the kernel against this function. If it were an independent rewrite, a
    disagreement would be ambiguous: kernel wrong, or reference wrong? Delegating removes the
    question, and this test keeps a future 'small cleanup' from forking it."""
    torch.manual_seed(0)
    b, s, h, d, n, topk = 2, 3, 4, 64, 40, 12
    q = torch.randn(b, s, h, d, dtype=torch.bfloat16)
    kv = torch.randn(b, n, d, dtype=torch.bfloat16)
    sink = torch.randn(h, dtype=torch.float32)
    idx = torch.randint(0, n, (b, s, topk), dtype=torch.int32)
    idx[..., ::5] = -1                            # the -1 padding both sides have to mask identically
    got = S.sparse_attn_eager(q, kv, sink, idx, d ** -0.5)
    want = K.sparse_attn(q, kv, sink, idx, d ** -0.5)
    assert torch.equal(got, want), "eager reference has drifted from v4_kernels_cpu.sparse_attn"
