"""The grouped MoE decode path is opt-in and defers cleanly, or it does not ship.

phase0/v4_moe_grouped replaces the routed-expert GEMMs of a decode step with ONE grouped fp4 launch.
Its numeric bar is `torch.equal` against the reference MoE — but that runs a tilelang kernel, which
is CUDA-only, so the parity proof lives on the GPU (`phase0/v4_moe_grouped.py::_smoke` and the bench
harness on the ring's tail box). What CI on a CPU box CAN pin, and what this file pins, is the
envelope around that kernel:

  * the module imports without dragging in a CUDA toolchain (tilelang is deferred into the builder),
  * it installs NOTHING unless `V4_MOE_GROUPED=1` AND a CUDA device is present, and
  * every shape it does not claim falls through to the forward it captured, untouched.

It also pins the LOAD-TIME BANK LAYOUT, which is not a numeric claim at all and so belongs here in
full: the grouped kernel needs a contiguous expert bank, and the layout's whole job is to make that
bank the ONLY copy of the weights rather than a duplicate the card cannot afford. That is a statement
about storage identity, about the bytes surviving the relay, and about `load_state_dict` writing
through the views into the bank — all of which a CPU box can check exactly, at tiny dims, with the
reference's own `MoE`.

The numeric parity test is GPU-gated and skips here. Run: python3 -m pytest tests/test_v4_moe_grouped.py -q
"""
import os
import sys
import weakref

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


# ── the load-time bank layout ────────────────────────────────────────────────────────────────────

_KINDS = ("w1", "w2", "w3")


def _fp4_moe(oracle, n_experts=4, expert_dtype="fp4", dim=64, inter=32):
    """A tiny MoE built from the REFERENCE's own classes with fp4 routed experts.

    Not a stub: the layout repoints real `nn.Parameter`s and the loader test drives a real
    `load_state_dict`, so anything less than the reference's `MoE`/`Expert`/`Linear` would prove
    nothing about the stage. fp4 needs `dim` a multiple of `fp4_block_size` (32); everything else is
    as small as the constructor's asserts allow. Takes `oracle` only to force `dsv4_model` loaded.
    `dim`/`inter` are widened to 128 by the one test that RUNS the thing — `v4_kernels_cpu.act_quant`
    needs the activation's last dim to be a multiple of `block_size` (128)."""
    mod = sys.modules["dsv4_model"]
    a = mod.ModelArgs(dim=dim, moe_inter_dim=inter, n_routed_experts=n_experts, n_activated_experts=2,
                      n_shared_experts=1, n_hash_layers=1, expert_dtype=expert_dtype,
                      dtype="bf16", scale_fmt=None, scale_dtype="fp32", vocab_size=32)
    with mod.set_dtype(torch.bfloat16):
        return mod.MoE(7, a)


def _paint(moe):
    """Give every routed expert tensor a distinct byte pattern, so a relay that scrambles or aliases
    the wrong slot cannot pass by luck."""
    with torch.no_grad():
        for i, e in enumerate(moe.experts):
            for j, k in enumerate(_KINDS):
                getattr(e, k).weight.data.view(torch.uint8).fill_(1 + i * 8 + j)
                getattr(e, k).scale.data.view(torch.uint8).fill_(129 + i * 8 + j)


def _painted(moe):
    return all(bool((getattr(e, k).weight.view(torch.uint8) == 1 + i * 8 + j).all())
               and bool((getattr(e, k).scale.view(torch.uint8) == 129 + i * 8 + j).all())
               for i, e in enumerate(moe.experts) for j, k in enumerate(_KINDS))


def test_bank_layout_is_a_noop_with_the_flag_off(oracle, monkeypatch):
    """Default OFF has to mean the loader allocates nothing and moves nothing — every expert keeps
    the exact tensor its constructor gave it, so a stage on the shipped default is byte-identical."""
    monkeypatch.setattr(GROUPED, "V4_MOE_GROUPED", False)
    moe = _fp4_moe(oracle)
    ptrs = [getattr(e, k).weight.data_ptr() for e in moe.experts for k in _KINDS]
    assert GROUPED.bank_layout(moe) == 0
    assert [getattr(e, k).weight.data_ptr() for e in moe.experts for k in _KINDS] == ptrs
    assert not hasattr(moe, "_grouped_bank"), "the flag-off path must not even mark the module"


def test_bank_layout_leaves_one_copy_of_the_weights_not_two(oracle, monkeypatch):
    """THE POINT OF THE WHOLE CHANGE, as a storage-identity claim.

    After the layout there is exactly ONE allocation per weight kind — the bank — and every routed
    expert's parameter is a VIEW into it, at its own offset, in expert order. Nothing was duplicated,
    so the bank the grouped kernel gathers from costs zero extra bytes and a seven-layer stage on a
    32 GiB card can hold it. The old lazy stack would have doubled these tensors instead."""
    monkeypatch.setattr(GROUPED, "V4_MOE_GROUPED", True)
    moe = _fp4_moe(oracle)
    assert GROUPED.bank_layout(moe) == 1
    bank = moe._grouped_bank
    for k, skey in (("w1", "w1_s"), ("w3", "w3_s"), ("w2", "w2_s")):
        for attr, key in (("weight", k), ("scale", skey)):
            b = bank[key]
            assert b.shape[0] == len(moe.experts) and b.is_contiguous()
            store = b.untyped_storage()
            for i, e in enumerate(moe.experts):
                p = getattr(getattr(e, k), attr)
                assert p.data_ptr() == b[i].data_ptr(), f"{k}.{attr}[{i}] is not the bank's slot {i}"
                assert p.untyped_storage().data_ptr() == store.data_ptr(), "a second allocation"
                assert p.is_contiguous(), "the s > 1 reference path needs contiguous expert weights"
            assert store.nbytes() == sum(getattr(getattr(e, k), attr).numel()
                                         for e in moe.experts), "the bank is not exactly one copy"


def test_bank_layout_carries_the_bytes_across(oracle, monkeypatch):
    """The relay copies, it does not reinterpret: every expert reads back exactly what it held, at
    its own slot. (Pre-load the tensors are uninitialised and this is vacuous on a stage — it matters
    because the same function is called by the GPU harness AFTER the weights are written.)"""
    monkeypatch.setattr(GROUPED, "V4_MOE_GROUPED", True)
    moe = _fp4_moe(oracle)
    _paint(moe)
    assert GROUPED.bank_layout(moe) == 1
    assert _painted(moe), "the layout moved an expert's bytes to the wrong slot"
    bank = moe._grouped_bank
    for i, e in enumerate(moe.experts):
        assert bool((bank["w1"][i].view(torch.uint8) == e.w1.weight.view(torch.uint8)).all())


def test_the_loader_writes_through_the_views_into_the_bank(oracle, monkeypatch):
    """The `Stage.load` seam, and with it the s > 1 fallback.

    The layout runs BEFORE the checkpoint is read, so `load_state_dict` must land in the bank rather
    than replacing the view with a fresh tensor — otherwise the stage would serve from per-expert
    copies while the grouped kernel gathered stale bank memory. It copies into the parameter, so it
    writes through; this pins that, and pins that only the loaded expert's slot moves."""
    monkeypatch.setattr(GROUPED, "V4_MOE_GROUPED", True)
    moe = _fp4_moe(oracle)
    _paint(moe)
    GROUPED.bank_layout(moe)
    bank = moe._grouped_bank
    tgt = 2
    sd = {k: torch.zeros_like(v) for k, v in moe.experts[tgt].state_dict().items()}
    sd["w1.weight"] = torch.full(tuple(moe.experts[tgt].w1.weight.shape), 9,
                                 dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    moe.experts[tgt].load_state_dict(sd, strict=True)
    assert bool((bank["w1"][tgt].view(torch.uint8) == 9).all()), "the load did not reach the bank"
    assert moe.experts[tgt].w1.weight.data_ptr() == bank["w1"][tgt].data_ptr(), "the view was replaced"
    for i in range(len(moe.experts)):
        if i != tgt:
            assert bool((bank["w1"][i].view(torch.uint8) == 1 + i * 8).all()), "a neighbour moved"


def test_banked_experts_keep_the_scale_the_reference_path_reads(oracle, monkeypatch):
    """`linear()` reaches an fp4 expert's scale as `weight.scale`, an attribute the reference hangs on
    the weight Parameter itself. The layout repoints `.data` rather than rebinding the Parameter
    precisely so that attribute survives — rebind it and prefill dies on a missing `.scale`."""
    monkeypatch.setattr(GROUPED, "V4_MOE_GROUPED", True)
    moe = _fp4_moe(oracle)
    GROUPED.bank_layout(moe)
    for i, e in enumerate(moe.experts):
        for k in _KINDS:
            lin = getattr(e, k)
            assert lin.weight.scale is lin.scale, "weight.scale must still BE the scale parameter"
            assert lin.scale.data_ptr() == moe._grouped_bank[k + "_s"][i].data_ptr()


def test_bank_layout_leaves_non_fp4_experts_alone(oracle, monkeypatch):
    """The grouped kernel is fp4-only, so a bf16 MoE (the CPU parity model, an unquantized config) is
    not relaid — no bank, no repointing, byte-identical."""
    monkeypatch.setattr(GROUPED, "V4_MOE_GROUPED", True)
    moe = _fp4_moe(oracle, expert_dtype=None)
    ptrs = [e.w1.weight.data_ptr() for e in moe.experts]
    assert GROUPED.bank_layout(moe) == 0
    assert [e.w1.weight.data_ptr() for e in moe.experts] == ptrs
    assert not hasattr(moe, "_grouped_bank")


def test_bank_layout_is_idempotent(oracle, monkeypatch):
    """A second call must not build a second bank on top of the first — that would be the duplicate
    this whole change exists to delete, allocated by the fix itself."""
    monkeypatch.setattr(GROUPED, "V4_MOE_GROUPED", True)
    moe = _fp4_moe(oracle)
    assert GROUPED.bank_layout(moe) == 1
    first = moe._grouped_bank
    assert GROUPED.bank_layout(moe) == 0
    assert moe._grouped_bank is first


def test_expert_bank_uses_the_load_time_bank_and_never_stacks(oracle, monkeypatch):
    """A banked layer must reach the fast path without going anywhere near the lazy stack: no fit
    check, no copy, no decline. The decline exists for an MoE the loader never reached; a stage's
    layers are all reached, which is why the lever now fires instead of declining."""
    monkeypatch.setattr(GROUPED, "V4_MOE_GROUPED", True)
    moe = _fp4_moe(oracle)
    GROUPED.bank_layout(moe)
    calls = []
    monkeypatch.setattr(GROUPED, "_bank_fits", lambda experts: (calls.append(1), True)[1])
    assert GROUPED._expert_bank(moe) is moe._grouped_bank
    assert calls == [], "a banked layer must not re-decide whether a stack would fit"


def test_stage_lays_out_the_bank_between_construction_and_load():
    """The seam, pinned as source: the layout has to run AFTER the Blocks exist (there is nothing to
    repoint before) and BEFORE `load()` (so the checkpoint writes through the views into the bank).
    Relayout after the load would mean copying — the duplicate this change removes. And it has to ask
    for `preserve=False`, which is the whole of the memory fix: the Blocks are two statements old, so
    the routed-expert bytes are still the constructor's uninitialised `torch.empty` and the layout is
    free to RELEASE each layer's per-expert run before it allocates that layer's banks. v4_stage is
    GPU-shaped and cannot be instantiated at V4's dims here, so the ordering is checked in the text."""
    import inspect
    call = "v4_moe_grouped.bank_layout(self.layers, preserve=False)"
    src = inspect.getsource(pytest.importorskip("v4_stage").Stage)
    assert call in src, "Stage must lay out the bank, and must lay it out release-first"
    init = src.index("def __init__")
    assert init < src.index(call) < src.index("def load("), \
        "the bank layout must sit in __init__, i.e. after construction and before load()"


def test_a_real_stage_lays_its_layers_out_release_first(monkeypatch):
    """The same seam, RUN rather than read — a real `Stage`, built with fp4 routed experts.

    The source check above cannot survive a refactor to positional arguments, and the CPU oracle's
    experts are bf16, so until this test the shipped `preserve=False` path was never once executed
    against a `Stage`. `cpu_args(expert_dtype="fp4")` gives one that the layout does claim: every
    layer gets a bank, every expert parameter is a view into it, and `_relayout_moe` is asked for the
    release-first ordering on every one of them."""
    VS = pytest.importorskip("v4_stage")
    monkeypatch.setattr(GROUPED, "V4_MOE_GROUPED", True)
    asked = []
    real = GROUPED._relayout_moe
    monkeypatch.setattr(GROUPED, "_relayout_moe",
                        lambda moe, preserve=True: (asked.append(preserve), real(moe, preserve))[1])
    args = REFCPU.cpu_args(expert_dtype="fp4")
    st = VS.Stage(0, 2, args, device="cpu")
    assert asked and all(p is False for p in asked), \
        f"Stage must relayout release-first; it asked for preserve={asked}"
    for L in st.layers:
        bank = getattr(L.ffn, "_grouped_bank", None)
        assert bank, "a stage's fp4 layer must come out of __init__ banked"
        for i, e in enumerate(L.ffn.experts):
            for k in _KINDS:
                lin = getattr(e, k)
                assert lin.weight.data_ptr() == bank[k][i].data_ptr()
                assert lin.scale.data_ptr() == bank[k + "_s"][i].data_ptr()
                assert lin.weight.scale is lin.scale, "the reference reads the scale off the weight"


# ── the layout's memory timeline: what it may hold, and when ─────────────────────────────────────
#
# The bank layout's ONLY justification is that it costs nothing on the card, and the first version of
# it was measured the way that claim invites: `torch.cuda.memory_allocated` before and after, on a
# 3-layer stage, where it is flat to the byte. It is flat because the release is real — every
# per-expert storage genuinely goes away. What is NOT flat is the moment in between. Allocating a
# [256, N, K] bank and only then walking the experts that it replaces means asking the driver for
# 1024 MiB the layer is still holding, and the blocks that come back are 256 scattered 4 MiB
# per-expert allocations sharing large-pool segments with the kinds not yet relaid — the wrong shape
# to satisfy the request that freed them, and not wholly-free segments, so `empty_cache` cannot
# return them either. `memory_reserved` therefore climbs by every bank built so far and only falls at
# the END of the layer. An 8-layer stage on a 32 GiB card dies there.
#
# These tests put a meter on that interval. `_Meter` tracks LIVE STORAGE BYTES exactly — a weakref
# finalizer on each `UntypedStorage`'s PyObject, which torch keeps pinned to the c10::StorageImpl, so
# a storage is counted out at the instant its last reference dies and not one statement later — and
# runs the ledger the failure was really about: bytes REQUESTED from the allocator against bytes
# RELEASED to it. `torch.cuda` is not available here and the numbers are not the point; the ORDERING
# is, and it is dimension-independent.


class _Meter:
    """Live storage bytes + the request-vs-release ledger, over a region of code.

    `orig` blocks are the constructor's per-expert tensors, `bank` blocks are what the layout asks
    for. `debt` is how far the layout has outrun itself: bytes it has taken from the allocator that
    the bytes it replaces had not yet given back. A debt of D means the process must be able to hold
    D bytes MORE than the model does, at that instant, in memory the allocator cannot recycle."""

    def __init__(self):
        self.live = self.peak = self.released = self.requested = self.worst_debt = 0
        self.banks = []

    def watch(self, t, kind):
        st = t.untyped_storage()
        n = st.nbytes()
        if n == 0:                      # the release-first path's void placeholders — nothing to meter
            return t
        self.live += n
        self.peak = max(self.peak, self.live)
        if kind == "bank":
            self.requested += n
            self.banks.append(n)
            self.worst_debt = max(self.worst_debt, self.requested - self.released)
        m = self

        def gone():
            m.live -= n
            if kind == "orig":
                m.released += n
        weakref.finalize(st, gone)
        return t

    def watch_experts(self, moe):
        for e in moe.experts:
            for k in _KINDS:
                self.watch(getattr(e, k).weight, "orig")
                self.watch(getattr(e, k).scale, "orig")
        return self.live


def _layout_metered(moe, preserve, monkeypatch):
    """Run the layout with every routed-expert storage and every bank allocation on a meter.

    `_relayout_moe` is the only python-level `torch.empty` caller inside the layout, so patching it
    for the width of the call catches the banks and nothing else."""
    m = _Meter()
    baseline = m.watch_experts(moe)
    m.released = m.requested = 0
    real_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **k: m.watch(real_empty(*a, **k), "bank"))
    n = GROUPED.bank_layout(moe, preserve=preserve)
    monkeypatch.undo()
    return m, baseline, n


def test_the_toy_dims_are_proportional_to_the_shipped_layer(oracle):
    """These tests run at 256 experts and toy widths; this pins that the SHAPE of the memory problem
    is the shipped one, so the ratios the next two tests assert are the ratios on the card.

    A routed expert is w1/w3 [inter, dim] + w2 [dim, inter] in packed fp4 (2 values per byte) plus an
    e8m0 scale per 32 fp4 elements along K, so per kind scale:weight is exactly 1:16 and the layer's
    six banks are 3 x 16/51 + 3 x 1/51 of the routed total whatever dim and inter are. At the shipped
    dim=4096 / moe_inter_dim=2048 / 256 experts that reads 3 x 1024.00 MiB + 3 x 64.00 MiB = 3.1875
    GiB per layer; here it is the same fractions of 816.00 KiB. Only the scale factor differs."""
    moe = _fp4_moe(oracle, n_experts=256)
    per_kind = {}
    for k in _KINDS:
        lin = getattr(moe.experts[0], k)
        per_kind[k] = lin.weight.untyped_storage().nbytes() * 256
        per_kind[k + "_s"] = lin.scale.untyped_storage().nbytes() * 256
    routed = sum(per_kind.values())
    assert len(moe.experts) == 256, "the shipped n_routed_experts, not a toy count"
    for k in _KINDS:
        assert per_kind[k] == 16 * per_kind[k + "_s"], "fp4 packs 2/byte, e8m0 is 1 per 32 — 1:16"
        assert per_kind[k] * 51 == routed * 16, "each weight bank is 16/51 of the layer's experts"
    # The shipped layer, from deepseek_v4_ref/inference/config.json: moe_inter_dim=2048, dim=4096,
    # n_routed_experts=256, expert_dtype=fp4.
    MiB = 1 << 20
    shipped_weight_bank = 2048 * (4096 // 2) * 256
    shipped_routed = 3 * shipped_weight_bank + 3 * (shipped_weight_bank // 16)
    assert shipped_weight_bank == 1024 * MiB, "the shipped weight bank is 1024.00 MiB"
    assert shipped_routed == 3264 * MiB, "the shipped layer's routed experts are 3.1875 GiB"
    assert routed * (shipped_weight_bank // per_kind["w1"]) == shipped_routed, \
        "the toy layer is the shipped 3.1875 GiB layer scaled, not a differently shaped one"


def test_bank_layout_never_outruns_what_it_released(oracle, monkeypatch):
    """THE REGRESSION GATE. On the stage's own path the layout must never hold a byte twice.

    Three claims, all exact, none of them a tolerance:
      * it ends holding exactly what it started with — the banks ARE the experts, nothing leaked,
      * its PEAK equals that too: at no instant is the layer resident twice, not even one kind of it,
      * and the ledger never goes into debt: every bank is allocated out of memory the layout has
        ALREADY handed back, so the allocator can satisfy it by recycling rather than by growing.
    The third is the one the card enforces and the one the first implementation failed. A debt of D
    is D bytes of `memory_reserved` above the model's own footprint, in blocks of the wrong shape to
    be reused; at the shipped dims the old ordering ran a debt of 1024 MiB per weight kind and left
    the layer's released run stranded behind it, which is 3.19 GiB of reserved-but-unusable per layer
    and the 8-layer OOM."""
    monkeypatch.setattr(GROUPED, "V4_MOE_GROUPED", True)
    moe = _fp4_moe(oracle, n_experts=256)
    m, baseline, n = _layout_metered(moe, False, monkeypatch)
    assert n == 1 and len(m.banks) == 6, "six banks: three weights, three scales"
    assert m.live == baseline, f"steady state moved by {m.live - baseline} B — the layout leaked"
    assert m.peak == baseline, (
        f"peak {m.peak} B against a steady {baseline} B: the layout held "
        f"{m.peak - baseline} B twice, i.e. {(m.peak - baseline) / max(m.banks):.2f} of a whole bank")
    assert m.worst_debt == 0, (
        f"the layout asked the allocator for {m.worst_debt} B it had not released — "
        f"{(m.worst_debt / max(m.banks)):.2f} of a bank, {m.worst_debt / baseline:.1%} of the layer")
    # and the banks really are the experts, at the right slot, after a release-first layout
    for i, e in enumerate(moe.experts):
        for k in _KINDS:
            assert getattr(e, k).weight.data_ptr() == moe._grouped_bank[k][i].data_ptr()
            assert getattr(e, k).scale.data_ptr() == moe._grouped_bank[k + "_s"][i].data_ptr()


def test_preserving_the_bytes_costs_a_whole_kind_which_is_why_the_stage_does_not(oracle, monkeypatch):
    """The other half of the gate: the copy-preserving order is measured, not assumed to be cheap.

    `preserve=True` cannot release before it allocates — it has to read the tensors it is replacing —
    so its peak is the layer plus one whole bank and its debt is one whole bank, every kind. That is
    the correct trade for the GPU parity harness (one layer, a card with room) and the wrong one for
    an 8-layer stage, and pinning the number here is what stops the two being confused again."""
    monkeypatch.setattr(GROUPED, "V4_MOE_GROUPED", True)
    moe = _fp4_moe(oracle, n_experts=256)
    m, baseline, n = _layout_metered(moe, True, monkeypatch)
    biggest = max(m.banks)
    assert n == 1
    assert m.live == baseline, "preserve=True must also end with exactly one copy"
    assert m.peak == baseline + biggest, (
        "preserve=True is expected to hold one whole bank on top of the layer — if that changed, "
        "the stage's preserve=False path may no longer be the thing that saves it")
    assert m.worst_debt == biggest
    assert biggest * 51 == baseline * 16, "the overshoot is a full weight kind, 16/51 of the layer"


def test_the_loader_still_writes_through_a_release_first_layout(oracle, monkeypatch):
    """The shipped path end to end: release-first layout, then `Stage.load`'s `load_state_dict`.

    Releasing the constructor's tensors is only safe because nothing readable is in them yet and the
    loader is strict — so the thing that has to hold is that the checkpoint still lands IN the bank,
    through the view, at the loaded expert's own slot and nowhere else."""
    monkeypatch.setattr(GROUPED, "V4_MOE_GROUPED", True)
    moe = _fp4_moe(oracle, n_experts=8)
    assert GROUPED.bank_layout(moe, preserve=False) == 1
    bank = moe._grouped_bank
    tgt = 5
    sd = {k: torch.zeros_like(v) for k, v in moe.experts[tgt].state_dict().items()}
    sd["w2.weight"] = torch.full(tuple(moe.experts[tgt].w2.weight.shape), 7,
                                 dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    moe.experts[tgt].load_state_dict(sd, strict=True)
    assert bool((bank["w2"][tgt].view(torch.uint8) == 7).all()), "the load did not reach the bank"
    assert moe.experts[tgt].w2.weight.data_ptr() == bank["w2"][tgt].data_ptr(), "the view was replaced"
    assert moe.experts[tgt].w2.weight.scale is moe.experts[tgt].w2.scale, "weight.scale did not survive"
    for i in range(len(moe.experts)):
        if i != tgt:
            assert not bool((bank["w2"][i].view(torch.uint8) == 7).all()), "a neighbour was written"


def test_a_release_first_layout_serves_bit_identically_to_no_layout(oracle, monkeypatch):
    """The correctness bar for throwing the constructor's tensors away, RUN rather than asserted.

    Two MoEs from one state dict — one plain, one given the release-first layout before it was
    loaded — driven through the reference's own `MoE.forward` on real fp4 weights. `torch.equal`, not
    a tolerance, at a decode shape and two chunk shapes, which is every shape a stage takes: the
    grouped kernel claims s == 1 only, and everything else reads these same parameters as the
    reference always did. If releasing the uninitialised tensors could ever cost a byte, this is
    where it would show. (The grouped KERNEL's own parity is CUDA-only and lives in the two
    `@pytest.mark.hardware` tests below; this is the layout's.)"""
    monkeypatch.setattr(GROUPED, "V4_MOE_GROUPED", True)
    from kernel import fp4_act_quant
    mod = sys.modules["dsv4_model"]
    dim = inter = 128                       # act_quant's block_size, the CPU kernels' one constraint
    g = torch.Generator().manual_seed(SEED)
    plain = _fp4_moe(oracle, n_experts=8, dim=dim, inter=inter)
    sd = {}
    for i in range(8):
        for k, out_f, in_f in (("w1", inter, dim), ("w2", dim, inter), ("w3", inter, dim)):
            w, s = fp4_act_quant(torch.randn(out_f, in_f, generator=g, dtype=torch.bfloat16),
                                 mod.fp4_block_size)
            sd[f"experts.{i}.{k}.weight"], sd[f"experts.{i}.{k}.scale"] = w.clone(), s.clone()
    for k, out_f, in_f in (("w1", inter, dim), ("w2", dim, inter), ("w3", inter, dim)):
        sd[f"shared_experts.{k}.weight"] = torch.randn(out_f, in_f, generator=g, dtype=torch.bfloat16)
    sd["gate.weight"] = torch.randn(*plain.gate.weight.shape, generator=g, dtype=torch.bfloat16)
    if plain.gate.bias is not None:
        sd["gate.bias"] = torch.randn(*plain.gate.bias.shape, generator=g, dtype=torch.float32)

    banked = _fp4_moe(oracle, n_experts=8, dim=dim, inter=inter)
    assert GROUPED.bank_layout(banked, preserve=False) == 1, "the layout must take before the load"
    for m in (plain, banked):
        m.load_state_dict(sd, strict=True)
        m.eval()
    for s in (1, 3, 9):
        x = torch.randn(1, s, dim, generator=g, dtype=torch.bfloat16)
        ids = torch.randint(0, 32, (1, s), generator=g)
        with torch.no_grad():
            want, got = plain(x, ids), banked(x, ids)
        assert torch.equal(want, got), f"the banked layout is not bit-exact at s={s}"


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


@pytest.mark.hardware
def test_bit_exact_on_gpu_over_the_bank_layout(monkeypatch):
    """The bar again, but against the layout a STAGE actually serves from.

    Two shipped-dims MoEs from the same seed: the oracle in the reference's per-expert layout, the
    subject relaid so its experts are views of one bank. Both halves of the envelope are checked on
    the subject — the grouped decode step (which reads the bank) and an s > 1 prefill (which falls
    through to the reference forward and reads the same bytes as per-expert views). If the layout
    perturbed either, this is where it shows."""
    if not torch.cuda.is_available():
        pytest.skip("grouped MoE parity needs a CUDA device (tilelang)")
    mod = GROUPED._load_model_module()
    args = GROUPED.real_dims_args(mod)
    ref_moe = GROUPED.build_real_dims_moe(mod, args)
    bank_moe = GROUPED.build_real_dims_moe(mod, args, bank=True)
    assert bank_moe._grouped_bank, "the bank layout did not take on a shipped-dims fp4 MoE"
    ref_forward = mod.MoE.forward
    monkeypatch.setattr(GROUPED, "_REF_FORWARD", ref_forward)
    monkeypatch.setattr(GROUPED, "_MOD", mod)
    monkeypatch.setattr(GROUPED, "_WORLD_SIZE", 1)
    for t in range(4):
        x = torch.randn(1, 1, args.dim, dtype=torch.bfloat16, device="cuda")
        ids = torch.randint(0, args.vocab_size, (1, 1), device="cuda")
        with torch.no_grad():
            ref = ref_forward(ref_moe, x, ids)
            got = GROUPED.grouped_forward(bank_moe, x, ids)
        assert torch.equal(ref, got), f"decode {t}: max|d| = {(ref.float() - got.float()).abs().max()}"
    for s in (2, 5):
        x = torch.randn(1, s, args.dim, dtype=torch.bfloat16, device="cuda")
        ids = torch.randint(0, args.vocab_size, (1, s), device="cuda")
        with torch.no_grad():
            ref = ref_forward(ref_moe, x, ids)
            got = GROUPED.grouped_forward(bank_moe, x, ids)
        assert torch.equal(ref, got), f"prefill s={s}: max|d| = {(ref.float() - got.float()).abs().max()}"
