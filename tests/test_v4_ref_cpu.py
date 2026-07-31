"""DeepSeek-V4-Flash on CPU: the kernel stand-ins are right, and the oracle they carry runs.

Two halves. The first pins phase0/v4_kernels_cpu.py against the tilelang kernels it transcribes --
the quantizers elementwise and exactly (they ARE exact; only the GEMM accumulation order is allowed
to drift, and both sides of every parity test run these same functions anyway). The second builds
the whole vendored Transformer at toy dims and proves it runs, is deterministic from a seed, and
keeps its caches consistent between the prefill and decode branches.

The finding worth carrying into step 2 is in test_compressor_prefill_matches_decode /
test_prefill_vs_reference_decode_consistency: the Compressor's two branches agree BITWISE when fed
the same input, but a whole-model prefill-vs-decode comparison does NOT, because the reference's own
Linear layers reassociate with M (seqlen*batch) and one fp32 ulp there is amplified by eight layers
of random hyper-connections into ~10% of a logit. So the sharp assertion lives at the branch, and
the end-to-end one is a bounded smoke check. See each test's docstring for the measured numbers.

That matters beyond this file: a v4_stage that splits the layer range across boxes CANNOT be graded
bit-for-bit against a whole-model oracle unless it feeds each stage the same tensors at the same
shapes. Parity has to be measured stage-in/stage-out, not end-to-end on logits.

Run: python3 -m pytest tests/test_v4_ref_cpu.py -q   (CPU-only, no GPU / no checkpoint needed)
"""
import importlib.util
import math
import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase0"))
import v4_kernels_cpu as K                                                            # noqa: E402
import v4_ref_cpu as R                                                                # noqa: E402

BF16_ULP = 2.0 ** -8       # bf16 keeps 8 mantissa bits, so one ulp is 2^-8 of the local magnitude
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# ── the kernel stand-ins ─────────────────────────────────────────────────────────────────────────

def test_act_quant_roundscale_powers_of_two():
    """scale_fmt set => every scale is 2^ceil(log2(amax/448)), and EXACT at the powers of two.

    The second half is the whole reason _round_scale goes through frexp: an input whose amax is
    exactly 448*2^k must produce 2^k. A float log2().ceil() can return k+epsilon there and round up
    to 2^(k+1), halving the effective precision of the one block that needed it most."""
    torch.manual_seed(0)
    x = (torch.randn(9, 256) * 7).bfloat16()
    _, s = K.act_quant(x, 128, scale_fmt="ue8m0")
    mant, _ = torch.frexp(s)
    assert torch.equal(mant, torch.full_like(mant, 0.5)), "scales must be exact powers of two"

    ks = list(range(-6, 7))
    exact = torch.zeros(len(ks), 128, dtype=torch.bfloat16)
    for i, k in enumerate(ks):
        exact[i, 0] = 448.0 * 2.0 ** k          # the rest stays smaller, so this is the block amax
    _, s = K.act_quant(exact, 128, scale_fmt="ue8m0")
    want = torch.tensor([[2.0 ** k] for k in ks])
    assert torch.equal(s, want), f"off-by-one at a power of two: {s.flatten()} != {want.flatten()}"


def test_act_quant_inplace_matches_manual():
    """The fused quant-dequant is exactly the elementwise formula, per row per 128-block."""
    torch.manual_seed(1)
    x = (torch.randn(4, 384) * 3).bfloat16()
    got = K.act_quant(x.clone(), 128, None, torch.float32, True)

    # every step stays in fp32 tensors, as the kernel's registers do -- folding the amax through a
    # Python float would compute the scale in double and land a bit away for some blocks
    want = torch.empty_like(x)
    for r in range(x.size(0)):
        for b in range(x.size(1) // 128):
            blk = x[r, b * 128:(b + 1) * 128].float()
            s = blk.abs().max().clamp_min(1e-4) * (1.0 / 448.0)
            q = (blk / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
            want[r, b * 128:(b + 1) * 128] = q.bfloat16()
    assert torch.equal(got, want)

    # the amax floor is not decoration: an all-zero block must not divide by zero
    zeros = torch.zeros(1, 128, dtype=torch.bfloat16)
    assert torch.equal(K.act_quant(zeros.clone(), 128, None, torch.float32, True), zeros)


def test_fp4_quant_grid():
    """Every fp4 output lands on the e2m1 grid times its power-of-two scale, ties to the EVEN code."""
    torch.manual_seed(2)
    x = (torch.randn(6, 64) * 4).bfloat16()
    y = K.fp4_act_quant(x.clone(), 32, True)
    _, s = K.fp4_act_quant(x.clone(), 32)
    q = (y.float() / s.float().repeat_interleave(32, -1)).abs()
    grid = torch.tensor(E2M1)
    assert torch.isclose(q.unsqueeze(-1), grid).any(-1).all(), "an output left the e2m1 grid"

    # amax 6.0 forces scale 1, so these ARE the midpoints of the grid and each must go to the even
    # code: 0.25->0, 0.75->1, 1.25->1, 1.75->2, 2.5->2, 3.5->4, 5.0->4. Two of the seven round DOWN.
    probe = torch.zeros(1, 32, dtype=torch.bfloat16)
    mids = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]
    probe[0, :len(mids)] = torch.tensor(mids, dtype=torch.bfloat16)
    probe[0, len(mids):2 * len(mids)] = -torch.tensor(mids, dtype=torch.bfloat16)
    out = K.fp4_act_quant(probe.clone(), 32, True)
    want = [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0]
    assert out[0, :len(mids)].float().tolist() == want
    assert out[0, len(mids):2 * len(mids)].float().tolist() == [-v for v in want]


def test_fp8_gemm_matches_dense():
    """fp8_gemm IS dequant-then-matmul; the dense spelling must reproduce it bit for bit."""
    torch.manual_seed(3)
    a, a_s = K.act_quant((torch.randn(5, 256) * 2).bfloat16(), 128)
    b = (torch.randn(384, 256) * 2).bfloat16().to(torch.float8_e4m3fn)
    b_s = torch.rand(3, 2) + 0.5                             # one scale per 128x128 block of B

    da = a.float() * a_s.repeat_interleave(128, -1)
    db = b.float() * b_s.repeat_interleave(128, 0).repeat_interleave(128, 1)
    got = K.fp8_gemm(a, a_s, b, b_s)
    assert torch.equal(got, (da @ db.t()).to(torch.get_default_dtype()))
    assert got.shape == (5, 384)


def test_fp4_gemm_nibble_order():
    """B's fp4 nibbles unpack LOW FIRST, cross-checked against the vendored converter itself.

    convert.py's cast_e2m1fn_to_e4m3fn is the authority on the packing DeepSeek's checkpoints use
    (`stack([TABLE[x & 0xF], TABLE[x >> 4]], -1)`), so it -- not our reading of it -- is what
    unpack_fp4 is graded against. Swapping the nibbles produces a model that runs and is garbage.
    Its e8m0 block scales are exercised at 1.0, where its internal 2^6 headroom offset makes the
    recast exactly 64x the fp4 value."""
    pytest.importorskip("safetensors")               # convert.py imports it at module scope
    pytest.importorskip("tqdm")
    spec = importlib.util.spec_from_file_location(
        "dsv4_convert", os.path.join(R.INFERENCE_DIR, "convert.py"))
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    torch.manual_seed(4)
    n, k = 128, 128
    packed_u8 = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    packed = packed_u8.view(torch.float4_e2m1fn_x2)

    ref, ref_s = conv.cast_e2m1fn_to_e4m3fn(packed_u8.view(torch.int8), torch.ones(n, k // 32))
    assert torch.equal(K.unpack_fp4(packed), ref.float() * ref_s.float()[:, :1])

    # ...and through fp4_gemm: a one-hot A picks out column j of B, i.e. exactly one nibble.
    onehot = torch.zeros(2, k)
    onehot[0, 0], onehot[1, 1] = 1.0, 1.0            # K positions 0 and 1 = byte 0 low, byte 0 high
    a = onehot.to(torch.float8_e4m3fn)
    a_s = torch.ones(2, k // 128)
    b_s = torch.ones(n, k // 32).to(torch.float8_e8m0fnu)
    got = K.fp4_gemm(a, a_s, packed, b_s).float()
    table = torch.tensor(E2M1 + tuple(-v for v in E2M1))
    assert torch.equal(got[0], table[(packed_u8[:, 0] & 0x0F).long()])
    assert torch.equal(got[1], table[(packed_u8[:, 0] >> 4).long()])


def test_sparse_attn_matches_dense():
    """Full-coverage topk == dense softmax attention with the sink in the denominator, and -1 masks
    contribute exactly nothing."""
    torch.manual_seed(5)
    b, s, h, d, n = 2, 3, 4, 32, 12
    q = (torch.randn(b, s, h, d) * 0.5).bfloat16()
    kv = (torch.randn(b, n, d) * 0.5).bfloat16()
    sink = torch.randn(h)
    scale = d ** -0.5
    full = torch.arange(n, dtype=torch.int32).view(1, 1, n).expand(b, s, n).contiguous()

    scores = torch.einsum("bshd,bnd->bshn", q.float(), kv.float()) * scale
    m = scores.amax(-1, keepdim=True)
    p = (scores - m).exp()
    denom = p.sum(-1, keepdim=True) + (sink.view(1, 1, h, 1) - m).exp()
    want = (torch.einsum("bshn,bnd->bshd", p, kv.float()) / denom).bfloat16()

    got = K.sparse_attn(q, kv, sink, full, scale)
    assert torch.allclose(got.float(), want.float(), rtol=1e-3, atol=1e-3)
    assert torch.equal(got, K.sparse_attn(q, kv, sink, full, scale)), "must be deterministic"

    # drop the second half of the positions: the answer is attention over the first half alone
    masked = full.clone()
    masked[:, :, n // 2:] = -1
    half = torch.arange(n // 2, dtype=torch.int32).view(1, 1, -1).expand(b, s, -1).contiguous()
    assert torch.allclose(K.sparse_attn(q, kv, sink, masked, scale).float(),
                          K.sparse_attn(q, kv[:, :n // 2], sink, half, scale).float(),
                          rtol=1e-3, atol=1e-3)


def test_hc_sinkhorn_properties():
    """pre/post ranges, a doubly-stochastic comb, and the opening softmax+column pass verbatim.

    The transcription check runs at sinkhorn_iters=1, where the function IS its opening pass: a true
    softmax (no eps in the denominator, +eps on the result) followed by one column normalize by
    (sum + eps). Getting that first step wrong -- softmaxing with the eps inside, or starting from a
    row normalize -- lands on a different fixed point that still looks doubly stochastic."""
    torch.manual_seed(6)
    b, s, hc, eps = 2, 5, 4, 1e-6
    mixes = torch.randn(b, s, (2 + hc) * hc)
    hc_scale, hc_base = torch.rand(3) + 0.5, torch.randn((2 + hc) * hc)

    pre, post, comb = K.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, 20, eps)
    assert pre.shape == (b, s, hc) and post.shape == (b, s, hc) and comb.shape == (b, s, hc, hc)
    assert (pre > eps).all() and (pre < 1 + eps).all()
    assert (post > 0).all() and (post < 2).all()
    # the LAST operation is a column normalize, so columns sum to 1 to within the eps and rows only
    # to wherever 20 Sinkhorn rounds got -- asserting both at 1e-5 would be asserting convergence
    assert torch.allclose(comb.sum(-2), torch.ones(b, s, hc), atol=1e-5)
    assert torch.allclose(comb.sum(-1), torch.ones(b, s, hc), atol=1e-2)

    one = K.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, 1, eps)[2]
    want = (mixes[..., 2 * hc:].unflatten(-1, (hc, hc)) * hc_scale[2]
            + hc_base[2 * hc:].view(hc, hc)).softmax(-1) + eps
    want = want / (want.sum(-2, keepdim=True) + eps)
    assert torch.equal(one, want)
    assert torch.allclose(pre, torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps)
    assert torch.allclose(post, 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc]))


def test_hadamard():
    """Sylvester H, unnormalized, so scale = d^-0.5 makes the transform its own inverse."""
    torch.manual_seed(7)
    h2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    h4 = torch.kron(h2, h2)
    x4 = torch.randn(3, 4)
    # allclose, not equal: the butterflies sum the same four terms in a different order than a
    # matmul does. What is being pinned is the ROW ORDER of H -- a bit-reversed Hadamard is still
    # orthogonal, still self-inverse, and still the wrong rotation.
    assert torch.allclose(K.hadamard_transform(x4), x4 @ h4.t(), rtol=0, atol=1e-6)

    d = 128
    x = torch.randn(2, 5, d)
    s = d ** -0.5
    back = K.hadamard_transform(K.hadamard_transform(x, s), s)
    assert torch.allclose(back, x, rtol=0, atol=1e-5)
    with pytest.raises(AssertionError):
        K.hadamard_transform(torch.randn(3, 6))


# ── the oracle ───────────────────────────────────────────────────────────────────────────────────

def _tokens(bsz, n, seed=0):
    torch.manual_seed(seed)
    return torch.randint(0, R.cpu_args().vocab_size, (bsz, n))


def test_oracle_smoke():
    """A whole random V4 prefills, decodes, and drafts, with every shape the contract promises."""
    args = R.cpu_args()
    model = R.build_oracle(args, seed=0)
    b, prompt = 2, 17
    x = _tokens(b, prompt + 6, seed=1)

    output_ids, logits, main_hidden = model(x[:, :prompt])
    assert logits.shape == (b, args.vocab_size)
    assert output_ids.shape == (b,)
    assert main_hidden.shape == (b, prompt, len(args.dspark_target_layer_ids) * args.dim)
    assert model.forward_spec(output_ids, main_hidden) is None

    for i in range(prompt, prompt + 6):
        output_ids, logits, main_hidden = model(x[:, i:i + 1], i)
        assert logits.shape == (b, args.vocab_size)
        assert main_hidden.shape == (b, 1, len(args.dspark_target_layer_ids) * args.dim)
        draft_ids, draft_logits, confidence = model.forward_spec(output_ids, main_hidden, i)
        assert draft_ids.shape == (b, args.dspark_block_size + 1)
        assert draft_logits.shape == (b, args.dspark_block_size, args.vocab_size)
        assert confidence.shape == (b, args.dspark_block_size)
        assert torch.equal(draft_ids[:, 0], output_ids), "the draft block starts at the sampled token"
        for t in (logits, main_hidden, draft_logits, confidence):
            assert torch.isfinite(t).all()


def test_oracle_deterministic():
    """Same seed, same weights, same numbers -- the premise every parity test rests on."""
    x = _tokens(2, 8, seed=2)
    outs = []
    for _ in range(2):
        model = R.build_oracle(seed=5)
        _, logits, _ = model(x[:, :6])
        acc = [logits]
        for i in range(6, 8):
            _, logits, _ = model(x[:, i:i + 1], i)
            acc.append(logits)
        outs.append(acc)
    for a, b in zip(*outs):
        assert torch.equal(a, b)


def _wire_compressors(attn):
    """What Attention.forward/Indexer.forward do lazily on their first call (model.py:496, :414).

    Driving a Compressor on its own means doing that hand-off by hand -- it is the only reference
    internal this file reaches into, and it is cache plumbing, not math."""
    attn.compressor.kv_cache = attn.kv_cache[:, attn.window_size:]
    attn.compressor.freqs_cis = attn.freqs_cis
    if attn.indexer is not None:
        attn.indexer.freqs_cis = attn.freqs_cis
        attn.indexer.compressor.kv_cache = attn.indexer.kv_cache
        attn.indexer.compressor.freqs_cis = attn.freqs_cis


def test_compressor_prefill_matches_decode():
    """THE branch test: compressing a window in one prefill == compressing it token by token.

    Compressor.forward is two different programs (model.py:331 vs :349) -- a batched
    unflatten/overlap_transform/softmax on one side, a rolling kv_state/score_state ring on the
    other -- and only their agreement makes a decode continue a prefill. Fed the SAME input, they
    agree BITWISE on all 8 compressors of this config; on a different random input one of the 8 came
    back a single bf16 ulp out (the residue is the fp32 wkv/wgate reassociating at M=1 vs M=16, not
    the branch), so the bar is 2 ulp rather than exact. A transcription slip in either branch is two
    orders of magnitude past that.

    Covers all three shapes: ratio 4 overlapping, ratio 8 plain, and the Indexer's rotated one,
    whose tail is a Hadamard + fp4 quant-dequant rather than an fp8 one."""
    args = R.cpu_args()
    pre_model, dec_model = R.build_oracle(args, seed=8), R.build_oracle(args, seed=8)
    torch.manual_seed(9)
    n = 16
    x = (torch.randn(2, n, args.dim) * 0.5).bfloat16()

    checked = 0
    for la, lb in zip(pre_model.layers, dec_model.layers):
        if not la.attn.compress_ratio:
            continue
        _wire_compressors(la.attn)
        _wire_compressors(lb.attn)
        pairs = [(la.attn.compressor, lb.attn.compressor)]
        if la.attn.indexer is not None:
            pairs.append((la.attn.indexer.compressor, lb.attn.indexer.compressor))
        for ca, cb in pairs:
            ca(x, 0)
            for i in range(n):
                cb(x[:, i:i + 1], i)
            filled = n // ca.compress_ratio
            got, want = cb.kv_cache[:, :filled].float(), ca.kv_cache[:, :filled].float()
            tol = 2 * BF16_ULP * want.abs().max().item()
            assert (got - want).abs().max().item() <= tol, (
                f"ratio={ca.compress_ratio} rotate={ca.rotate}: decode-compressed KV differs from "
                f"prefill-compressed by {(got - want).abs().max().item():.3e} > {tol:.3e}")
            checked += 1
    assert checked == 8, f"expected 8 compressors in this config, drove {checked}"


def test_prefill_vs_reference_decode_consistency():
    """Prefill n, or prefill n-1 and decode one: the caches every layer keeps must agree.

    This is the cross-branch check at whole-model scale -- Attention's window ring, the Compressor's
    state ring, the Indexer's compressed cache and the offsets that address them all, at once. It is
    a BOUNDED check, not a bitwise one, and the reason is worth writing down: the two paths run the
    reference's own Linear layers at different M (1 vs seqlen), torch reassociates accordingly, and
    one fp32 ulp there is amplified by eight layers of random hyper-connections. Measured on these
    two configs, the honest disagreement is 0 and 16 bf16 ulp of the cache's own scale; zeroing a
    Compressor's state before the decode step lands at 163-229 and zeroing a window cache at 256.
    Hence the bar of 96 -- 6x over the noise, 1.7x under the smallest break. The SHARP pin on those
    branches is test_compressor_prefill_matches_decode, which has no amplification to fight.

    Two honest limits. The drift grows with context (~66 ulp by a 47-token split), so this is a
    check on short sequences, not a bound that holds at length. And it does not exercise the
    Indexer's SELECTION: below index_topk * compress_ratio tokens every compressed position is
    picked anyway, exactly as in the shipped config, so corrupting the Indexer's own cache here only
    permutes the gather order.

    Note the prefill logits are the LAST position only (ParallelHead full_logits=False), so the two
    paths are directly comparable at position n-1."""
    args = R.cpu_args()
    for bsz, split in ((1, 12), (2, 15)):          # 15 lands on both a ratio-4 and a ratio-8 boundary
        x = _tokens(bsz, split + 1, seed=10)
        a = R.build_oracle(args, seed=11)
        a(x[:, :split], 0)
        _, logits_a, _ = a(x[:, split:split + 1], split)
        b = R.build_oracle(args, seed=11)
        _, logits_b, _ = b(x[:, :split + 1], 0)
        assert logits_a.shape == logits_b.shape == (bsz, args.vocab_size)

        for i, (la, lb) in enumerate(zip(a.layers, b.layers)):
            ratio = la.attn.compress_ratio
            filled = la.attn.window_size + ((split + 1) // ratio if ratio else 0)
            ka = la.attn.kv_cache[:bsz, :filled].float()
            kb = lb.attn.kv_cache[:bsz, :filled].float()
            tol = 96 * BF16_ULP * kb.abs().max().item()
            assert (ka - kb).abs().max().item() <= tol, (
                f"b={bsz} split={split} layer {i} (ratio {ratio}): decode-path KV cache diverged by "
                f"{(ka - kb).abs().max().item():.3e} > {tol:.3e}")
        d = (logits_a.float() - logits_b.float()).abs().max().item()
        assert d <= 0.25 * logits_b.float().abs().max().item(), f"logits diverged by {d:.3e}"


def test_forward_spec_prefill_then_decode():
    """forward_spec is a KV-cache warm-up at start_pos 0 and a drafter after it.

    DSparkBlock.forward short-circuits to attn-only while start_pos == 0 (model.py:846), so the
    prefill call returns None and there is nothing to unpack -- a caller that assumes a triple
    breaks on the first token of every request."""
    args = R.cpu_args()
    model = R.build_oracle(args, seed=12)
    x = _tokens(1, 10, seed=13)
    output_ids, _, main_hidden = model(x[:, :9])
    assert model.forward_spec(output_ids, main_hidden, 0) is None

    output_ids, _, main_hidden = model(x[:, 9:10], 9)
    got = model.forward_spec(output_ids, main_hidden, 9)
    assert got is not None and len(got) == 3
    draft_ids, draft_logits, confidence = got
    assert draft_ids.dtype == x.dtype and draft_ids.shape == (1, args.dspark_block_size + 1)
    assert draft_logits.shape == (1, args.dspark_block_size, args.vocab_size)
    assert confidence.shape == (1, args.dspark_block_size)


def test_hash_gate_used():
    """The first n_hash_layers route by token id, not by score -- the gate is a lookup table there.

    Cheap and direct: hook the Gate and compare the indices it returned against tid2eid[input_ids].
    A hash layer that quietly fell through to the topk branch would still route two experts per
    token and still produce logits."""
    args = R.cpu_args()
    assert args.n_hash_layers >= 2
    model = R.build_oracle(args, seed=14)
    seen = {}
    for i in (0, args.n_hash_layers):
        gate = model.layers[i].ffn.gate
        gate.register_forward_hook(lambda m, inp, out, k=i: seen.__setitem__(k, out[1]))
    x = _tokens(2, 7, seed=15)
    model(x)

    hashed = model.layers[0].ffn.gate
    assert hashed.hash and not model.layers[args.n_hash_layers].ffn.gate.hash
    assert torch.equal(seen[0].long(), hashed.tid2eid[x.flatten()].long())
    assert seen[args.n_hash_layers].shape == (x.numel(), args.n_activated_experts)


def test_cpu_args_constraints():
    """The config is not free-form: cpu_args refuses the shapes the reference would fail on deep
    inside Attention. Kept honest here so a future tweak trips the assert, not a stack trace."""
    args = R.cpu_args()
    assert (args.head_dim - args.rope_head_dim) % 64 == 0
    assert args.index_head_dim % 32 == 0 and math.log2(args.index_head_dim).is_integer()
    assert len(args.compress_ratios) == args.n_layers + args.n_mtp_layers
    assert 4 in args.compress_ratios and any(r > 4 for r in args.compress_ratios)
    for bad in (dict(head_dim=96), dict(index_head_dim=96), dict(n_heads=5),
                dict(compress_ratios=(0, 0, 4, 8, 4, 8, 4, 0))):
        with pytest.raises(AssertionError):
            R.cpu_args(**bad)
