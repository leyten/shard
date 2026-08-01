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
KERNELS = pytest.importorskip("v4_kernels_cpu")

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


def test_hash_layer_is_claimed_not_deferred(args, oracle, monkeypatch):
    """Hash-routed layers used to defer wholesale, and on a head stage that was half the layers.

    They defer no longer: `_keep_last_of_each` reproduces the duplicate-index `y[idx] +=` on device,
    so the only thing left in the way was the guard itself. This pins that the guard is GONE — the
    step must not reach the fallback. (What it computes is `test_hash_layer_is_bit_exact_*` below;
    this one is about which path it takes, and it uses a bf16 oracle layer that never gets that far,
    so the fallback spy would fire on the guard alone.)"""
    ffn = oracle.layers[0].ffn                        # hash-routed (layer 0 < n_hash_layers)
    assert ffn.gate.hash
    calls = _spy(monkeypatch)
    g = torch.Generator().manual_seed(SEED + 2)
    x, ids = _x(args, 1, g, oracle), _ids(args, 1, g)
    monkeypatch.setattr(GROUPED, "_expert_bank", lambda moe: None)   # stop before the fp4-only kernel
    GROUPED.grouped_forward(ffn, x, ids)
    assert [w for w in ffn._grouped_declined] == ["bank-would-not-fit"], \
        f"a hash layer must reach the bank, not be turned away first: {ffn._grouped_declined}"


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


def _slot(moe, i, kind, attr="weight"):
    """The bank rows expert `i`'s `kind` Linear is supposed to BE, as a tensor.

    w1 and w3 share the `w13` bank — w1's rows then w3's — because they share a grouped launch, so
    "which slot" is now a (bank, row range) rather than a bank index, and the tests have to ask the
    layout rather than assume `bank[kind][i]`."""
    bank = moe._grouped_bank
    for key, kinds in GROUPED._BANK_GROUPS:
        if kind not in kinds:
            continue
        b = bank[key + ("" if attr == "weight" else "_s")]
        off = sum(getattr(getattr(moe.experts[i], k), attr).shape[0] for k in kinds[:kinds.index(kind)])
        return b[i, off:off + getattr(getattr(moe.experts[i], kind), attr).shape[0]]
    raise KeyError(kind)


def _fp4_moe(oracle, n_experts=4, expert_dtype="fp4", dim=64, inter=32, layer_id=7, topk=2):
    """A tiny MoE built from the REFERENCE's own classes with fp4 routed experts.

    Not a stub: the layout repoints real `nn.Parameter`s and the loader test drives a real
    `load_state_dict`, so anything less than the reference's `MoE`/`Expert`/`Linear` would prove
    nothing about the stage. fp4 needs `dim` a multiple of `fp4_block_size` (32); everything else is
    as small as the constructor's asserts allow. Takes `oracle` only to force `dsv4_model` loaded.
    `dim`/`inter` are widened to 128 by the tests that RUN the thing — `v4_kernels_cpu.act_quant`
    needs the activation's last dim to be a multiple of `block_size` (128). `layer_id=0` is under
    `n_hash_layers`, i.e. the hash-routed gate whose ids can repeat."""
    mod = sys.modules["dsv4_model"]
    a = mod.ModelArgs(dim=dim, moe_inter_dim=inter, n_routed_experts=n_experts,
                      n_activated_experts=topk, swiglu_limit=10.0, route_scale=1.5,
                      n_shared_experts=1, n_hash_layers=1, expert_dtype=expert_dtype,
                      dtype="bf16", scale_fmt=None, scale_dtype="fp32", vocab_size=32)
    with mod.set_dtype(torch.bfloat16):
        return mod.MoE(layer_id, a)


def _loaded_fp4_moe(oracle, layer_id, n_experts=8, topk=4, dim=128, inter=128, seed=SEED, bank=True):
    """A tiny fp4 MoE with REAL quantized weights, ready to run through the CPU kernels.

    Weights go through the reference's own `fp4_act_quant`, so the routed experts are valid packed
    fp4 + e8m0 rather than reinterpreted noise (which would dequantize to inf and make every
    comparison vacuously equal). `bank=True` applies the shipped release-first layout; `bank=False`
    leaves the experts standalone so the fast path has to build the LAZY stack instead — the two are
    different code and both have to gather the same rows."""
    from kernel import fp4_act_quant
    mod = sys.modules["dsv4_model"]
    moe = _fp4_moe(oracle, n_experts=n_experts, dim=dim, inter=inter, layer_id=layer_id, topk=topk)
    g = torch.Generator().manual_seed(seed)
    if bank:
        # `_relayout_moe` rather than `bank_layout` so the helper does not depend on the env flag —
        # what the flag gates is covered by test_bank_layout_is_a_noop_with_the_flag_off.
        assert GROUPED._relayout_moe(moe, preserve=False), "the layout must take before the load"
    sd = {}
    for i in range(n_experts):
        for k, out_f, in_f in (("w1", inter, dim), ("w2", dim, inter), ("w3", inter, dim)):
            w, s = fp4_act_quant(torch.randn(out_f, in_f, generator=g, dtype=torch.bfloat16),
                                 mod.fp4_block_size)
            sd[f"experts.{i}.{k}.weight"], sd[f"experts.{i}.{k}.scale"] = w.clone(), s.clone()
    for k, out_f, in_f in (("w1", inter, dim), ("w2", dim, inter), ("w3", inter, dim)):
        sd[f"shared_experts.{k}.weight"] = torch.randn(out_f, in_f, generator=g,
                                                       dtype=torch.bfloat16) * 0.02
    sd["gate.weight"] = torch.randn(*moe.gate.weight.shape, generator=g, dtype=torch.bfloat16) * 0.02
    if moe.gate.bias is not None:
        sd["gate.bias"] = torch.randn(*moe.gate.bias.shape, generator=g, dtype=torch.float32) * 0.02
    if moe.gate.hash:
        sd["gate.tid2eid"] = torch.randint(0, n_experts, tuple(moe.gate.tid2eid.shape),
                                           generator=g, dtype=torch.int32)
    moe.load_state_dict(sd, strict=True)
    return moe.eval()


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
    the exact tensor its constructor gave it, so a stage on the shipped default is byte-identical.

    `preserve=False` is checked here too, and it is the whole reason the rest of the v4 suite does
    not have to re-prove itself against this change: the flag is read on `bank_layout`'s FIRST line,
    before `preserve` is looked at, so with the env unset — which is every test that is not this file
    — `Stage.__init__`'s new argument cannot reach a single tensor. Releasing before allocating is
    unobservable on the default path, not merely harmless on it."""
    monkeypatch.setattr(GROUPED, "V4_MOE_GROUPED", False)
    for preserve in (True, False):
        moe = _fp4_moe(oracle)
        ptrs = [getattr(e, k).weight.data_ptr() for e in moe.experts for k in _KINDS]
        assert GROUPED.bank_layout(moe, preserve=preserve) == 0
        assert [getattr(e, k).weight.data_ptr() for e in moe.experts for k in _KINDS] == ptrs
        assert all(getattr(e, k).weight.numel() for e in moe.experts for k in _KINDS), \
            "the flag-off path must not void a parameter even under preserve=False"
        assert not hasattr(moe, "_grouped_bank"), "the flag-off path must not even mark the module"


def test_bank_layout_leaves_one_copy_of_the_weights_not_two(oracle, monkeypatch):
    """THE POINT OF THE WHOLE CHANGE, as a storage-identity claim.

    After the layout there is exactly ONE allocation per BANK — and w1 and w3 now share one, since
    they share a grouped launch — and every routed expert's parameter is a VIEW into it, at its own
    row range, in expert order. Nothing was duplicated, so the bank the grouped kernel gathers from
    costs zero extra bytes and a seven-layer stage on a 32 GiB card can hold it. The old lazy stack
    would have doubled these tensors instead."""
    monkeypatch.setattr(GROUPED, "V4_MOE_GROUPED", True)
    moe = _fp4_moe(oracle)
    assert GROUPED.bank_layout(moe) == 1
    bank = moe._grouped_bank
    assert set(bank) == {"w13", "w13_s", "w2", "w2_s"}, \
        f"w1 and w3 must share a bank so they can share a launch; got {sorted(bank)}"
    for key, kinds in GROUPED._BANK_GROUPS:
        for attr, suffix in (("weight", ""), ("scale", "_s")):
            b = bank[key + suffix]
            assert b.shape[0] == len(moe.experts) and b.is_contiguous()
            store = b.untyped_storage()
            for i, e in enumerate(moe.experts):
                for k in kinds:
                    p = getattr(getattr(e, k), attr)
                    assert p.data_ptr() == _slot(moe, i, k, attr).data_ptr(), \
                        f"{k}.{attr}[{i}] is not its row range of the {key} bank"
                    assert p.untyped_storage().data_ptr() == store.data_ptr(), "a second allocation"
                    assert p.is_contiguous(), "the s > 1 reference path needs contiguous expert weights"
            assert store.nbytes() == sum(getattr(getattr(e, k), attr).numel()
                                         for e in moe.experts
                                         for k in kinds), "the bank is not exactly one copy"


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
        for k in _KINDS:
            assert bool((_slot(moe, i, k).view(torch.uint8)
                         == getattr(e, k).weight.view(torch.uint8)).all())


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
    assert bool((_slot(moe, tgt, "w1").view(torch.uint8) == 9).all()), "the load did not reach the bank"
    assert moe.experts[tgt].w1.weight.data_ptr() == _slot(moe, tgt, "w1").data_ptr(), \
        "the view was replaced"
    # w3 shares w1's bank, one row range along. The state dict zeroes it, and it was painted 19, so
    # this pins that the SECOND half of the shared bank is addressed too — a wrong offset would leave
    # the paint standing or would have let w1's 9s spill into it.
    assert bool((_slot(moe, tgt, "w3").view(torch.uint8) == 0).all()), \
        "w3's rows of the shared bank did not take the load"
    for i in range(len(moe.experts)):
        if i != tgt:
            assert bool((_slot(moe, i, "w1").view(torch.uint8) == 1 + i * 8).all()), "a neighbour moved"
            assert bool((_slot(moe, i, "w3").view(torch.uint8) == 1 + i * 8 + 2).all()), \
                "a neighbour's w3 rows moved"


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
            assert lin.scale.data_ptr() == _slot(moe, i, k, "scale").data_ptr()


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
                assert lin.weight.data_ptr() == _slot(L.ffn, i, k).data_ptr()
                assert lin.scale.data_ptr() == _slot(L.ffn, i, k, "scale").data_ptr()
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
    assert n == 1 and len(m.banks) == 4, "four banks: w13 + w2, each with its scale"
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
            assert getattr(e, k).weight.data_ptr() == _slot(moe, i, k).data_ptr()
            assert getattr(e, k).scale.data_ptr() == _slot(moe, i, k, "scale").data_ptr()


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
    assert biggest * 51 == baseline * 32, ("the overshoot is the fused w13 weight bank, 32/51 of the "
                                           "layer — it was 16/51 while w1 and w3 banked separately")


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
    assert bool((_slot(moe, tgt, "w2").view(torch.uint8) == 7).all()), "the load did not reach the bank"
    assert moe.experts[tgt].w2.weight.data_ptr() == _slot(moe, tgt, "w2").data_ptr(), \
        "the view was replaced"
    assert moe.experts[tgt].w2.weight.scale is moe.experts[tgt].w2.scale, "weight.scale did not survive"
    for i in range(len(moe.experts)):
        if i != tgt:
            assert not bool((_slot(moe, i, "w2").view(torch.uint8) == 7).all()), "a neighbour was written"


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


# ── what the grouped path actually claims, RUN on CPU ────────────────────────────────────────────
#
# The tilelang kernel is CUDA-only, so the numeric bar against real fp4 GEMMs lives in the two
# `@pytest.mark.hardware` tests below. Everything ABOVE the kernel, though — which expert lands in
# which bank row, that w1 and w3 come back out of one fused launch in the right halves, that a hash
# gate's repeated ids drop the way the reference drops them, the ascending-id fold, and how many
# launches the step actually issues — is dispatch, and a CPU box can run all of it exactly.
#
# It does so because `grouped_fp4_gemm` off CUDA is a per-slot loop over the installed
# `kernel.fp4_gemm` (see its docstring), which is what the grid-indexed kernel computes. That makes
# these `torch.equal` and not `allclose`: both sides run the SAME GEMM on the same operands, so any
# difference is the dispatch, which is the thing under test.


def _runnable(monkeypatch):
    """Wire the module globals `install()` would have set, without installing. Returns the model.

    Skips off the CPU backend: these tests RUN the MoE, and on a box where `kernel` resolved to the
    real tilelang file its GEMMs cannot take a CPU tensor. The GPU box proves the same claims through
    the `@pytest.mark.hardware` tests instead."""
    if KERNELS.backend() != "cpu":
        pytest.skip("runs the MoE through the CPU stand-in kernels")
    mod = sys.modules["dsv4_model"]
    monkeypatch.setattr(GROUPED, "_MOD", mod)
    monkeypatch.setattr(GROUPED, "_WORLD_SIZE", 1)
    monkeypatch.setattr(GROUPED, "_REF_FORWARD", mod.MoE.forward)
    return mod


@pytest.mark.parametrize("sel,want", [([3, 1, 3, 0], [False, True, True, True]),
                                      ([2, 2, 2, 2], [False, False, False, True]),
                                      ([0, 1, 2, 3], [True, True, True, True]),
                                      ([5, 4, 4, 5], [False, False, True, True])])
def test_keep_mask_reproduces_the_reference_duplicate_drop(sel, want):
    """The hash-layer claim, isolated from every GEMM.

    `y[idx] += v` with a repeated index is NOT a repeated add: it reads the row once per duplicate,
    adds, and scatters them all back, so the last write wins and every earlier duplicate is thrown
    away. `_keep_last_of_each` has to name exactly the survivors. This drives the reference's own
    statement — `torch.where(indices == i)` then `y[idx] += ...`, ascending i — against the mask, at
    values chosen so an accidental sum could not coincide with the drop."""
    ids = torch.tensor(sel, dtype=torch.int32)
    keep = GROUPED._keep_last_of_each(ids)
    assert keep.tolist() == want, "the survivor is the LAST slot naming each expert"
    v = (torch.arange(1.0, len(sel) + 1.0) ** 3)[:, None] * torch.ones(1, 3)
    indices = ids.view(1, -1)
    y = torch.zeros(1, 3)
    for i in sorted(set(sel)):                       # the reference's loop, ascending expert id
        idx, top = torch.where(indices == i)
        y[idx] += v[top]
    got = torch.zeros(1, 3)                          # what the grouped fold does with the mask
    for k in torch.argsort(ids, stable=True).tolist():
        got += v[k:k + 1] * keep[k]
    assert torch.equal(y, got), f"mask {keep.tolist()} does not reproduce the reference's drop"


@pytest.mark.parametrize("banked", [True, False])
@pytest.mark.parametrize("layer_id", [7, 0])
def test_the_grouped_step_is_bit_exact_on_cpu(oracle, monkeypatch, layer_id, banked):
    """`torch.equal` against the reference MoE, on real fp4 weights, for BOTH routings.

    layer_id 7 is score-routed (what the path already claimed) and layer_id 0 is hash-routed (what
    it now claims). `banked` runs it over the shipped load-time layout and over the lazy stack, which
    build the fused w13 bank by different code — aliasing vs `cat`+`stack` — and must agree."""
    mod = _runnable(monkeypatch)
    moe = _loaded_fp4_moe(oracle, layer_id, bank=banked)
    assert moe.gate.hash is (layer_id == 0)
    g = torch.Generator().manual_seed(SEED + layer_id)
    for t in range(6):
        x = torch.randn(1, 1, 128, generator=g, dtype=torch.bfloat16)
        ids = torch.randint(0, 32, (1, 1), generator=g)
        with torch.no_grad():
            want = mod.MoE.forward(moe, x, ids)
            got = GROUPED.grouped_forward(moe, x, ids)
        assert not getattr(moe, "_grouped_declined", None), \
            f"the step declined instead of grouping: {moe._grouped_declined}"
        assert torch.equal(want, got), \
            f"layer {layer_id} draw {t}: max|d| = {(want.float() - got.float()).abs().max()}"


def _rowwise_fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
    """`fp4_gemm` made ROW-INVARIANT — the property the tilelang kernel has and BLAS does not.

    The vendored kernel computes every output element from its own A row and B tile, so an M-row GEMM
    and M one-row GEMMs agree bit for bit. `torch.matmul` picks its blocking from M, so they do not:
    on this box `a[0:1] @ b.t()` and `(a[0:2] @ b.t())[0]` already differ in the last bits. That
    matters in exactly one place — the reference runs a DUPLICATED hash expert as a 2-row GEMM while
    the grouped path runs it as two 1-row grid blocks — so the duplicate test installs this to hold
    the emulator to the kernel's property instead of to OpenBLAS's."""
    rows = a.reshape(-1, a.size(-1))
    scales = a_s.reshape(-1, a_s.size(-1))
    out = torch.cat([KERNELS.fp4_gemm(rows[i:i + 1], scales[i:i + 1], b, b_s, scale_dtype)
                     for i in range(rows.size(0))])
    return out.view(*a.shape[:-1], b.size(0))


def test_a_repeated_hash_expert_is_dropped_exactly_as_the_reference_drops_it(oracle, monkeypatch):
    """The duplicate case itself, forced rather than waited for.

    A random `tid2eid` repeats often but not always, and "often" is not a test. This pins every
    token's routing to `[2, 2, 5, 2]` — expert 2 named three times — so the reference discards slots
    0 and 1 outright and keeps only slot 3, and the grouped path has to discard the same two. A path
    that summed the duplicates instead would be wrong by two whole experts and still look plausible."""
    mod = _runnable(monkeypatch)
    monkeypatch.setattr(sys.modules["kernel"], "fp4_gemm", _rowwise_fp4_gemm)
    monkeypatch.setattr(mod, "fp4_gemm", _rowwise_fp4_gemm)
    moe = _loaded_fp4_moe(oracle, layer_id=0)
    with torch.no_grad():
        moe.gate.tid2eid.copy_(torch.tensor([2, 2, 5, 2], dtype=torch.int32).expand_as(moe.gate.tid2eid))
    g = torch.Generator().manual_seed(SEED + 9)
    for t in range(4):
        x = torch.randn(1, 1, 128, generator=g, dtype=torch.bfloat16)
        ids = torch.randint(0, 32, (1, 1), generator=g)
        with torch.no_grad():
            want = mod.MoE.forward(moe, x, ids)
            got = GROUPED.grouped_forward(moe, x, ids)
        assert torch.equal(want, got), \
            f"draw {t}: max|d| = {(want.float() - got.float()).abs().max()}"


def test_a_discarded_hash_slot_cannot_poison_the_token_whatever_it_holds(oracle, monkeypatch):
    """The reference never EVALUATES a discarded slot, so its contents must not reach the sum.

    That is stronger than "it contributes zero", and the difference is not academic: masking by
    multiplying makes `inf * 0.0` a NaN, which would poison the whole token instead of dropping the
    slot. A select cannot. This spikes the discarded rows of the w2 launch with inf and NaN and
    demands the SAME token back, bit for bit, as the unspiked run — the discarded slots are not
    merely weighted out, they are not read.

    inf is reachable on a real card: the routed output is bf16, and an e8m0 weight scale times an
    fp4 magnitude can overflow it on a checkpoint we do not control."""
    mod = _runnable(monkeypatch)
    moe = _loaded_fp4_moe(oracle, layer_id=0)
    with torch.no_grad():                            # expert 2 named three times: slots 0 and 1 die
        moe.gate.tid2eid.copy_(torch.tensor([2, 2, 5, 2], dtype=torch.int32).expand_as(moe.gate.tid2eid))
    g = torch.Generator().manual_seed(SEED + 11)
    x = torch.randn(1, 1, 128, generator=g, dtype=torch.bfloat16)
    ids = torch.randint(0, 32, (1, 1), generator=g)
    assert GROUPED._keep_last_of_each(torch.tensor([2, 2, 5, 2])).tolist() == [False, False, True, True]

    with torch.no_grad():
        want = GROUPED.grouped_forward(moe, x, ids)

    real, calls = GROUPED.grouped_fp4_gemm, []

    def spiked(*a, **k):
        out = real(*a, **k)
        calls.append(1)
        if len(calls) == 2:                          # the w2 launch — the rows that get discarded
            out = out.clone()
            out[0], out[1] = float("inf"), float("nan")
        return out

    monkeypatch.setattr(GROUPED, "grouped_fp4_gemm", spiked)
    with torch.no_grad():
        got = GROUPED.grouped_forward(moe, x, ids)
    assert torch.isfinite(got.float()).all(), "a discarded slot's inf/NaN reached the token"
    assert torch.equal(want, got), "the discarded slots were read, not dropped"

def test_the_kernel_wrapper_refuses_shapes_its_store_loop_cannot_hold(oracle, monkeypatch):
    """The grouped kernel's store is UNPREDICATED — a precondition, not a preference.

    The vendored fp4 GEMM writes its tile with a bounds-checked `T.copy`; the grouped one writes row
    g with a raw `for j in T.Parallel(block_N)` loop, so an N that is not a whole number of 128-wide
    tiles makes the last block write past the row, and more expert slots than the block_M=32 A tile
    holds cannot be stored at all. Neither raises on a card: both are silent memory stomps. The
    wrapper asserts instead, and this pins that it does — the docstring said `G <= block_M` for
    months while nothing checked it."""
    def fp4(*shape):                       # torch.zeros has no fp4 CPU kernel; a uint8 view does
        return torch.zeros(*shape, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)

    act = torch.zeros(4, 128, dtype=torch.uint8).view(torch.float8_e4m3fn)
    with pytest.raises(AssertionError, match="whole number of 128-wide tiles"):
        GROUPED.grouped_fp4_gemm(act, torch.zeros(4, 1), fp4(4, 192, 64), torch.zeros(4, 192, 4))
    big = torch.zeros(33, 128, dtype=torch.uint8).view(torch.float8_e4m3fn)
    with pytest.raises(AssertionError, match="exceeds the block_M"):
        GROUPED.grouped_fp4_gemm(big, torch.zeros(33, 1), fp4(33, 128, 64), torch.zeros(33, 128, 4))


# ── the coverage gate: a matrix kind that silently un-groups must FAIL, not just get slower ──────


def _count_launches(monkeypatch, mod):
    """Count the launches a step issues, split by which path issued them.

    `dsv4_model.act_quant` / `dsv4_model.fp4_gemm` are the names `linear()` resolves, so they count
    the REFERENCE's per-expert work — one act_quant + one fp4_gemm per expert per matrix. The grouped
    path reaches its GEMM through `v4_moe_grouped.grouped_fp4_gemm` (and its CPU emulator imports
    `kernel.fp4_gemm` directly, which is deliberately NOT one of these names), so the two are
    distinguishable and 'the lever silently stopped covering w2' shows up as per-expert calls
    reappearing rather than as a number nobody reads."""
    log = {"act_quant": 0, "per_expert_fp4": 0, "grouped": []}
    real_q, real_g, real_grouped = mod.act_quant, mod.fp4_gemm, GROUPED.grouped_fp4_gemm

    def q(*a, **k):
        log["act_quant"] += 1
        return real_q(*a, **k)

    def gemm(*a, **k):
        log["per_expert_fp4"] += 1
        return real_g(*a, **k)

    def grouped(a, a_s, w, w_s, *rest, **k):
        log["grouped"].append((a.size(0), w.size(1)))
        return real_grouped(a, a_s, w, w_s, *rest, **k)

    monkeypatch.setattr(mod, "act_quant", q)
    monkeypatch.setattr(mod, "fp4_gemm", gemm)
    monkeypatch.setattr(GROUPED, "grouped_fp4_gemm", grouped)
    return log


@pytest.mark.parametrize("layer_id", [7, 0])
def test_every_routed_expert_matrix_is_grouped_and_none_is_left_per_expert(oracle, monkeypatch, layer_id):
    """THE REGRESSION GATE, and the one this whole change came out of.

    A real-GPU profile of a 6-layer stage read `fp4_gemm_kernel 72 calls` beside
    `grouped_fp4_gemm_kernel 6 calls`, which looks like partial coverage and is not: 72 is 4 x 18,
    i.e. FOUR whole layers that never grouped at all, and 18 is 6 experts x 3 matrices. Nothing in
    the kernel table says so. This makes the same statement testable at the layer:

      * a grouped decode step issues ZERO per-expert fp4 GEMMs — not one for w2, not one for anything;
      * it issues exactly TWO grouped launches, the first [G, 2*inter] (w1 and w3 fused) and the
        second [G, dim] (w2), so a kind that quietly fell back off the fused bank changes the shape;
      * and it quantizes exactly TWICE, against the reference's 3 per expert, which is the
        act_quant column of that same profile.
    The reference numbers are measured in the same run rather than hardcoded, so the ratio survives
    a change of expert count."""
    mod = _runnable(monkeypatch)
    moe = _loaded_fp4_moe(oracle, layer_id)
    G, inter, dim = moe.n_activated_experts, 128, 128
    g = torch.Generator().manual_seed(SEED + 4)
    x = torch.randn(1, 1, dim, generator=g, dtype=torch.bfloat16)
    ids = torch.randint(0, 32, (1, 1), generator=g)
    with torch.no_grad():                              # the reference's loop runs once per DISTINCT
        routed = moe.gate(x.view(-1, dim), ids.flatten())[1][0].tolist()   # expert, so a hash layer's
    distinct = len(set(routed))                        # repeats collapse iterations on that side only

    log = _count_launches(monkeypatch, mod)
    with torch.no_grad():
        mod.MoE.forward(moe, x, ids)
    ref = dict(log, grouped=list(log["grouped"]))
    assert ref["per_expert_fp4"] == 3 * distinct, "the reference runs w1, w3 and w2 once per expert"
    assert ref["act_quant"] == 3 * distinct, "and quantizes its input once per those"
    assert ref["grouped"] == []

    log.update(act_quant=0, per_expert_fp4=0)
    log["grouped"].clear()
    with torch.no_grad():
        GROUPED.grouped_forward(moe, x, ids)
    assert log["per_expert_fp4"] == 0, (
        f"{log['per_expert_fp4']} routed-expert GEMMs still ran one expert at a time — a matrix kind "
        f"has silently un-grouped (the reference issues {ref['per_expert_fp4']})")
    assert log["grouped"] == [(G, 2 * inter), (G, dim)], (
        f"expected one fused w1+w3 launch then one w2 launch over {G} experts, got {log['grouped']}")
    assert log["act_quant"] == 2, (
        f"the grouped step must quantize the token once and the intermediates once, not "
        f"{log['act_quant']} times (the reference: {ref['act_quant']})")


def test_coverage_names_the_layers_that_never_grouped(oracle, monkeypatch):
    """`coverage()` is the profiler-facing half of the same gate.

    Kernel counts can only say how many layers grouped by division; this says WHICH, and why the
    others did not. A stage whose every layer grouped reads all-nonzero, and one layer left on the
    reference path is a named entry with its reason, not a silently smaller speedup."""
    mod = _runnable(monkeypatch)
    fast = _loaded_fp4_moe(oracle, layer_id=7)
    fast.layer_id = 7
    slow = _loaded_fp4_moe(oracle, layer_id=0, seed=SEED + 1)
    slow.layer_id = 0
    holder = torch.nn.ModuleList([fast, slow])
    assert GROUPED.coverage(holder) == {7: (0, {}), 0: (0, {})}, "nothing has run yet"

    g = torch.Generator().manual_seed(SEED + 5)
    x1 = torch.randn(1, 1, 128, generator=g, dtype=torch.bfloat16)
    x3 = torch.randn(1, 3, 128, generator=g, dtype=torch.bfloat16)
    with torch.no_grad():
        GROUPED.grouped_forward(fast, x1, torch.randint(0, 32, (1, 1), generator=g))
        GROUPED.grouped_forward(fast, x1, torch.randint(0, 32, (1, 1), generator=g))
        GROUPED.grouped_forward(slow, x3, torch.randint(0, 32, (1, 3), generator=g))
    cov = GROUPED.coverage(holder)
    assert cov[7] == (2, {}), f"the score layer grouped both steps: {cov[7]}"
    assert cov[0] == (0, {"s>1": 1}), f"the s > 1 step must be named and counted: {cov[0]}"



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


@pytest.mark.hardware
def test_bit_exact_on_gpu_with_a_duplicated_hash_expert(monkeypatch):
    """The one claim that CANNOT be checked off a card, forced instead of hoped for.

    The hash path's correctness rests on the tilelang fp4 GEMM being ROW-INVARIANT: the reference
    runs a duplicated expert as an n-row GEMM and the grouped path runs n one-row grid blocks. That
    is a property of the kernel, so only a GPU can measure it — and until this test the GPU never
    saw a hash layer at all (both parity tests build `layer_id=7`, and `_smoke`'s random `tid2eid`
    draws contain no duplicate at all about 62% of the time at 256 experts and topk 6).

    So the routing is PINNED, not drawn: expert 3 named three times and 17 twice, which is the
    reference discarding three of six slots. If `torch.matmul`-style M-dependent blocking were ever
    true of the tilelang kernel, this is where it would show, and nothing else would."""
    if not torch.cuda.is_available():
        pytest.skip("grouped MoE parity needs a CUDA device (tilelang)")
    mod = GROUPED._load_model_module()
    args = GROUPED.real_dims_args(mod)
    ref_moe = GROUPED.build_real_dims_moe(mod, args, seed=3, layer_id=0)
    bank_moe = GROUPED.build_real_dims_moe(mod, args, seed=3, layer_id=0, bank=True)
    assert ref_moe.gate.hash and bank_moe._grouped_bank
    pattern = torch.tensor([3, 3, 17, 3, 200, 17], dtype=torch.int32, device="cuda")
    assert pattern.numel() == args.n_activated_experts
    with torch.no_grad():
        for m in (ref_moe, bank_moe):
            m.gate.tid2eid.data.copy_(pattern.expand_as(m.gate.tid2eid))
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
        assert torch.equal(ref, got), f"dup {t}: max|d| = {(ref.float() - got.float()).abs().max()}"
