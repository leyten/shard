"""v4_dspark_moe: the drafter's pair-grouped MoE, graded byte-for-byte against the reference.

The lever's claim splits in two, and this file pins the half a CPU box can pin DETERMINISTICALLY:
everything above the GEMM kernel. Off CUDA the pair path computes its GEMMs at the REFERENCE'S OWN
M-grouping (one fp4_gemm per distinct expert per matrix kind — v4_dspark_moe._pair_gemms_cpu), so
the routing schedule, the quantize-once row gather, the per-row ascending fold and the shared-expert
tail are all provable `torch.equal` here, end to end through real drafted rounds. The CUDA branch's
extra step — every (row, expert) pair as an independent slot of one grouped launch — rests on the
tilelang kernel's row-invariance, the property v4_moe_grouped's hash-duplicate path already stakes
on hardware; its parity test at the DRAFTER's shape is the `@pytest.mark.hardware` test at the
bottom, run on the ring's tail box, not here.

The bar is torch.equal, not allclose, for the reason the module docstring gives: a moved proposal
changes acceptance, and therefore changes the measurement it was supposed to speed up.

Also pinned: the envelope. Default OFF binds nothing and banks nothing; the bind is per INSTANCE
and the class chain is untouched (the verifier's losslessness is BY CONSTRUCTION, not by argument);
every declined shape lands on the class chain bit-identically; a bankless drafter declines rather
than lazy-stacking 3.2 GiB on the tail.

Run: python3 -m pytest tests/test_v4_dspark_moe.py -q
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
REFCPU = pytest.importorskip("v4_ref_cpu")
V4 = pytest.importorskip("v4_stage")
D = pytest.importorskip("v4_dspark_draft")
DM = pytest.importorskip("v4_dspark_moe")
GROUPED = pytest.importorskip("v4_moe_grouped")

SEED = 7
PROMPT = 13
RUNS = (1, 3, 2, 1, 1)          # pipelined n=1 rounds AND the serial multi-position advance


@pytest.fixture(autouse=True)
def _clean_module():
    """Every test starts and ends with the module's globals as imported. The bind itself is per
    instance on per-test objects, so there is no class method to restore — only the flag and the
    reference-module capture."""
    flag, mod, ws = DM.V4_DSPARK_MOE, DM._MOD, DM._WORLD_SIZE
    yield
    DM.V4_DSPARK_MOE, DM._MOD, DM._WORLD_SIZE = flag, mod, ws


# ── the end-to-end A/B: real drafted rounds, fp4 drafter MoEs, both paths ─────────────────────────

def _args():
    return REFCPU.cpu_args(n_routed_experts=64, n_activated_experts=6, dspark_block_size=5,
                           n_mtp_layers=3, compress_ratios=(0, 0, 4, 8, 4, 8, 4, 0, 0, 0, 0))


def _build(oracle, args):
    st = V4.Stage(0, args.n_layers, args, head=True, tail=True, dspark=True, device="cpu")
    for li in range(args.n_layers):
        st.layers[li].load_state_dict(oracle.layers[li].state_dict(), strict=True)
    st.embed_tokens.load_state_dict(oracle.embed.state_dict(), strict=True)
    st.norm.load_state_dict(oracle.norm.state_dict(), strict=True)
    st.lm_head.load_state_dict(oracle.head.state_dict(), strict=True)
    with torch.no_grad():
        for n in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            getattr(st, n).data.copy_(getattr(oracle, n).data)
    st._dspark = True
    dr = D.DSparkTail(st)
    for k, blk in enumerate(dr.mtp):
        sd = {n: v for n, v in oracle.mtp[k].state_dict().items() if n not in D.ALIAS_KEYS}
        blk.load_state_dict(sd, strict=False)
    DM.swap_in_fp4_moes(dr, moe_inter_dim=128, seed=SEED)
    return st, dr


def _record(oracle, args):
    """Prefill + rounds off ONE stage, so both drafters replay identical inputs."""
    st, _ = _build(oracle, args)
    ids = torch.randint(0, args.vocab_size, (1, PROMPT),
                        generator=torch.Generator().manual_seed(SEED))
    tok = st.logits_all(st.forward(st.embed(ids), ids, 0), full_logits=False).argmax(-1)
    prefill_main = st.tail_main_hidden().clone()
    g = torch.Generator().manual_seed(SEED + 100)
    seq = [int(tok)] + torch.randint(0, args.vocab_size, (sum(RUNS),), generator=g).tolist()
    rounds, pos, base = [], PROMPT, 0
    for rl in RUNS:
        chunk = torch.tensor([seq[base:base + rl]], dtype=torch.long)
        st.forward(st.embed(chunk), chunk, pos)
        committed = torch.tensor([seq[base + 1:base + rl + 1]], dtype=torch.long)
        rounds.append((committed, st.tail_main_hidden().clone(), pos))
        pos += rl
        base += rl
    return tok, prefill_main, rounds


@pytest.fixture(scope="module")
def world():
    """The shared harness: one oracle, one recorded set of rounds, and a replay function — so
    every leg in this file replays IDENTICAL inputs through its own drafter."""
    args = _args()
    oracle = REFCPU.build_oracle(args, SEED)
    tok, prefill_main, rounds = _record(oracle, args)

    def replay(dr):
        dr.prefill(tok, prefill_main)
        out = []
        for committed, main, pos in rounds:
            dr.advance_and_draft(committed, main, start_pos=pos)
            out.append(tuple(t.clone() for t in dr.last_spec))
        return out, [b.attn.kv_cache.clone() for b in dr.mtp]
    return args, oracle, replay


@pytest.fixture(scope="module")
def ab(world):
    """Both legs, computed once: (reference leg, lever leg, the lever leg's drafter)."""
    args, oracle, replay = world
    DM.V4_DSPARK_MOE = False
    _, dr_ref = _build(oracle, args)
    assert DM.install_drafter(dr_ref) == 0, "default OFF must bind nothing"
    ref = replay(dr_ref)

    DM.V4_DSPARK_MOE = True
    _, dr_fast = _build(oracle, args)
    assert DM.install_drafter(dr_fast) == len(dr_fast.mtp), "install must claim every drafter MoE"
    fast = replay(dr_fast)
    DM.V4_DSPARK_MOE = False
    return ref, fast, dr_fast


def test_proposals_bit_identical(ab):
    """THE claim: output_ids, logits and confidence — the whole triple a reply is built from —
    torch.equal across every round, including the multi-position advances."""
    (ref_out, _), (fast_out, _), _ = ab
    for i, (a, b) in enumerate(zip(ref_out, fast_out)):
        for x, y, what in zip(a, b, ("output_ids", "logits", "confidence")):
            assert torch.equal(x, y), f"drafter {what} diverged at round {i}"


def test_mtp_caches_bit_identical(ab):
    """The drafter's only persistent state walks the same bytes down both paths."""
    (_, ref_kv), (_, fast_kv), _ = ab
    for i, (rk, fk) in enumerate(zip(ref_kv, fast_kv)):
        assert torch.equal(rk, fk), f"mtp {i} kv_cache diverged"


def test_the_pair_path_actually_fired_every_round(ab):
    """Bug 4's test: bit-equality with zero coverage would only prove the fallback. Every block of
    every forward_spec must have been served by the pair path, with no declines."""
    _, _, dr_fast = ab
    calls = sum(RUNS)                     # one MoE step per committed position per block
    cov = DM.coverage(dr_fast)
    assert all(steps == calls and not declined for steps, declined in cov.values()), cov


def test_install_is_per_instance_and_the_class_chain_is_untouched(ab):
    """The verifier's losslessness argument: the CLASS forward the main model runs through is the
    exact object it was before install, and only the drafter's three instances carry a binding."""
    _, _, dr_fast = ab
    mod = V4.ref()
    assert mod.MoE.forward is not DM.draft_forward, "the CLASS must never carry the pair path"
    for blk in dr_fast.mtp:
        bound = blk.ffn.__dict__.get("forward")
        assert bound is not None and bound.__func__ is DM.draft_forward, \
            "each drafter MoE instance must carry its own pair-path binding"


def test_composes_with_v4_dspark_fast(ab, world):
    """The shipped tail runs BOTH drafter levers: V4_DSPARK_FAST collapses the intermediate
    advances into cache-only writes, and this lever pair-groups the kept forward's MoEs. The
    composition surface is the kept `forward_spec`'s block loop reaching the instance-bound
    forwards — proven here by replaying the same rounds down the combined path against the plain
    reference loop's triples and caches."""
    FAST = pytest.importorskip("v4_dspark_fast")
    args, oracle, replay = world
    (ref_out, ref_kv), _, _ = ab
    ref_method = D.DSparkTail.advance_and_draft
    saved = (FAST.V4_DSPARK_FAST, FAST._REF_ADVANCE)
    try:
        FAST.V4_DSPARK_FAST = True
        assert FAST.install(D), "the fast advance must install for the combined leg"
        DM.V4_DSPARK_MOE = True
        _, dr = _build(oracle, args)
        assert DM.install_drafter(dr) == len(dr.mtp)
        got_out, got_kv = replay(dr)
    finally:
        D.DSparkTail.advance_and_draft = ref_method
        FAST.V4_DSPARK_FAST, FAST._REF_ADVANCE = saved
    for i, (a, b) in enumerate(zip(ref_out, got_out)):
        for x, y, what in zip(a, b, ("output_ids", "logits", "confidence")):
            assert torch.equal(x, y), f"FAST+MOE {what} diverged at round {i}"
    for i, (rk, gk) in enumerate(zip(ref_kv, got_kv)):
        assert torch.equal(rk, gk), f"FAST+MOE mtp {i} kv_cache diverged"
    cov = DM.coverage(dr)
    # FAST runs ONE kept forward per round however long the committed run — len(RUNS) MoE steps.
    assert all(steps == len(RUNS) and not dec for steps, dec in cov.values()), cov


# ── the envelope, on a stub drafter (cheap: no stage, no oracle) ──────────────────────────────────

def _stub_tail(moe):
    return types.SimpleNamespace(mtp=[types.SimpleNamespace(ffn=moe, layer_id=moe.layer_id)])


def _tiny_fp4_moe(layer_id=7, n_experts=8, topk=4, dim=128, inter=128, bank=True, seed=SEED):
    """A loaded fp4 MoE at act_quant-compatible dims, banked preserve=True (the weights are real)."""
    from kernel import fp4_act_quant
    mod = V4.ref()
    a = mod.ModelArgs(dim=dim, moe_inter_dim=inter, n_routed_experts=n_experts,
                      n_activated_experts=topk, n_shared_experts=1, n_hash_layers=1,
                      score_func="sqrtsoftplus", route_scale=1.5, swiglu_limit=10.0,
                      expert_dtype="fp4", dtype="bf16", scale_fmt=None, scale_dtype="fp32",
                      vocab_size=64)
    with mod.set_dtype(torch.bfloat16):
        moe = mod.MoE(layer_id, a)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for i in range(n_experts):
            e = moe.experts[i]
            for lin, out_f, in_f in ((e.w1, inter, dim), (e.w3, inter, dim), (e.w2, dim, inter)):
                w, s = fp4_act_quant(torch.randn(out_f, in_f, generator=g, dtype=torch.bfloat16),
                                     mod.fp4_block_size)
                lin.weight.data.copy_(w)
                lin.scale.data.copy_(s)
        for lin in (moe.shared_experts.w1, moe.shared_experts.w2, moe.shared_experts.w3):
            lin.weight.data.normal_(0, 0.02, generator=g)
        moe.gate.weight.data.normal_(0, 0.02, generator=g)
        if moe.gate.hash:
            moe.gate.tid2eid.data.random_(0, n_experts, generator=g)
        else:
            moe.gate.bias.data.normal_(0, 0.02, generator=g)
    if bank:
        assert GROUPED._relayout_moe(moe, preserve=True)
    return moe.eval()


def _x(moe, s, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, s, moe.dim, generator=g).to(torch.bfloat16)
    ids = torch.randint(0, 64, (1, s), generator=g)
    return x, ids


def _arm(moe):
    DM.V4_DSPARK_MOE = True
    DM._MOD = V4.ref()
    DM._WORLD_SIZE = 1
    took = DM.install_drafter(_stub_tail(moe))
    return took


def test_default_off_binds_and_banks_nothing():
    DM.V4_DSPARK_MOE = False
    moe = _tiny_fp4_moe(bank=False)
    assert DM.install_drafter(_stub_tail(moe)) == 0
    assert "forward" not in moe.__dict__
    assert not hasattr(moe, "_grouped_bank"), "OFF must not even lay the bank"


def test_moe_level_bit_exact_over_many_draws():
    """The pair path against the untouched reference at the drafter's shape, 16 fresh routings —
    the MoE-level version of the end-to-end proof, cheap enough to sweep."""
    moe = _tiny_fp4_moe()
    assert _arm(moe) == 1
    ref_fwd = type(moe).forward
    for t in range(16):
        x, ids = _x(moe, 5, seed=t)
        with torch.no_grad():
            want = ref_fwd(moe, x, ids)
            got = moe(x, ids)
        assert torch.equal(want, got), f"pair path diverged from the reference on draw {t}"
    assert moe._draft_steps == 16 and not getattr(moe, "_draft_declined", {})


def test_declined_shapes_fall_through_bit_identically():
    """T == 1 and pairs > block_M land on the class chain with the reference's exact answer, and
    the decline is RECORDED — a lever that silently does nothing is the bug class this engine keeps
    paying for."""
    moe = _tiny_fp4_moe()
    assert _arm(moe) == 1
    ref_fwd = type(moe).forward
    for s, why in ((1, "s<=1"), (9, "pairs>block_M")):      # 9 * 4 = 36 > 32
        x, ids = _x(moe, s, seed=s)
        with torch.no_grad():
            want = ref_fwd(moe, x, ids)
            got = moe(x, ids)
        assert torch.equal(want, got)
        assert moe._draft_declined.get(why) == 1, (why, moe._draft_declined)


def test_hash_gate_is_never_claimed():
    """The drafter's blocks are score-routed at every real config (layer_id >= n_hash_layers), but
    the claim is checked, not assumed: a hash gate declines at install AND at forward."""
    moe = _tiny_fp4_moe(layer_id=0)                          # < n_hash_layers=1: hash-routed
    assert _arm(moe) == 0, "install must not claim a hash-routed MoE"
    assert "forward" not in moe.__dict__
    # and a binding that somehow lands on one anyway declines at forward, bit-identically
    # (the fixture already banked it, so the bank is NOT what saves this — the gate check is)
    moe.forward = types.MethodType(DM.draft_forward, moe)
    x, ids = _x(moe, 5)
    with torch.no_grad():
        want = type(moe).forward(moe, x, ids)
        got = moe(x, ids)
    assert torch.equal(want, got)
    assert moe._draft_declined.get("hash-routed") == 1


def test_no_bank_declines_and_never_lazy_stacks():
    """A drafter MoE the layout never reached must DECLINE — the lazy `_expert_bank` stack would be
    a 3.2 GiB duplicate on the most VRAM-pinned box in the ring, so the pair path must not reach
    it. Bind by hand (install would bank), call, and require: reference answer, a recorded decline,
    and NO bank appearing as a side effect."""
    moe = _tiny_fp4_moe(bank=False)
    DM.V4_DSPARK_MOE = True
    DM._MOD = V4.ref()
    DM._WORLD_SIZE = 1
    moe.forward = types.MethodType(DM.draft_forward, moe)
    x, ids = _x(moe, 5)
    with torch.no_grad():
        want = type(moe).forward(moe, x, ids)
        got = moe(x, ids)
    assert torch.equal(want, got)
    assert moe._draft_declined.get("no-bank") == 1
    assert not getattr(moe, "_grouped_bank", None), "declining must not build a bank"


def test_world_size_gt_1_declines():
    moe = _tiny_fp4_moe()
    assert _arm(moe) == 1
    DM._WORLD_SIZE = 2
    x, ids = _x(moe, 5)
    with torch.no_grad():
        want = type(moe).forward(moe, x, ids)
        got = moe(x, ids)
    assert torch.equal(want, got)
    assert moe._draft_declined.get("world_size>1") == 1


def test_non_fp4_experts_are_left_alone():
    """The bf16 CPU harness's drafter must stay byte-identical: install refuses it entirely."""
    mod = V4.ref()
    a = mod.ModelArgs(dim=128, moe_inter_dim=128, n_routed_experts=8, n_activated_experts=4,
                      n_shared_experts=1, n_hash_layers=1, expert_dtype=None, dtype="bf16",
                      scale_fmt=None, scale_dtype="fp32", vocab_size=64)
    with mod.set_dtype(torch.bfloat16):
        moe = mod.MoE(7, a)
    DM.V4_DSPARK_MOE = True
    assert DM.install_drafter(_stub_tail(moe)) == 0
    assert "forward" not in moe.__dict__ and not hasattr(moe, "_grouped_bank")


# ── the levers registry wiring ────────────────────────────────────────────────────────────────────

def test_lever_is_registered_and_judged_from_the_note():
    import v4_levers as VL
    lv = VL.LEVERS_BY_ENV.get("V4_DSPARK_MOE")
    assert lv is not None and lv.side == VL.STAGE and lv.owner == "v4_dspark_moe"
    saved = dict(VL._NOTES)
    try:
        VL._NOTES.pop("V4_DSPARK_MOE", None)
        DM.V4_DSPARK_MOE = True
        req, obs, ok = lv.check(VL.Ctx(VL.STAGE))
        assert (req, obs, ok) == ("on", "no-drafter-yet", None), (req, obs, ok)
        VL.note("V4_DSPARK_MOE", True)
        req, obs, ok = lv.check(VL.Ctx(VL.STAGE))
        assert (req, obs, ok) == ("on", "on", True), (req, obs, ok)
        VL.note("V4_DSPARK_MOE", False)                     # armed but the install did not take
        req, obs, ok = lv.check(VL.Ctx(VL.STAGE))
        assert (req, obs, ok) == ("on", "off", False), (req, obs, ok)
    finally:
        VL._NOTES.clear()
        VL._NOTES.update(saved)


# ── the GPU half of the claim, run on the ring's tail box ─────────────────────────────────────────

@pytest.mark.hardware
def test_bit_exact_on_gpu_at_the_drafter_shape():
    """The CUDA branch — every (row, expert) pair one slot of one grouped launch — against the
    untouched reference at V4's REAL dims and the REAL drafter shape (5 rows x 6 experts = 30
    pairs). This is the row-invariance claim on silicon; everything above the kernel was proven on
    CPU. Mirrors v4_moe_grouped's hardware tests, including the bank layout."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    mod = GROUPED._load_model_module()
    args = GROUPED.real_dims_args(mod)
    moe = GROUPED.build_real_dims_moe(mod, args, layer_id=43, bank=True)
    ref_fwd = mod.MoE.forward
    DM.V4_DSPARK_MOE = True
    DM._MOD = mod
    DM._WORLD_SIZE = 1
    moe.forward = types.MethodType(DM.draft_forward, moe)
    for t in range(8):
        x = torch.randn(1, 5, args.dim, dtype=torch.bfloat16, device="cuda")
        ids = torch.randint(0, args.vocab_size, (1, 5), device="cuda")
        with torch.no_grad():
            want = ref_fwd(moe, x, ids)
            got = moe(x, ids)
        assert torch.equal(want, got), f"pair path diverged on GPU draw {t}"
    assert moe._draft_steps == 8 and not getattr(moe, "_draft_declined", {})
