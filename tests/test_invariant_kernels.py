"""R4 gate: receipt-grade (batch-invariant) Triton kernels, run on CPU under TRITON_INTERPRET=1.

  * oracle: both kernels match a line-for-line port of the scratchpad MiMo CPU oracle's math
    (mimo_ref.Layer.ffn/_mlp and Layer.attention) within the cross-kernel tolerance class;
  * invariance: probe rows are BYTE-IDENTICAL across >= 11 batch mixes per kernel (batch sizes, neighbours,
    expert groupings, segment kinds, page layouts, split-count padding);
  * row independence: finite perturbation of every other row (and of the probe's own masked KV) leaves the
    probe's bytes unchanged;
  * anti-vacuity: the same kernels with a batch-DEPENDENT split (the fast-kernel heuristic) FAIL the same
    byte-identity check.

CI runs a fast subset; SHARD_R4_FULL=1 runs every mix (a few minutes in the interpreter).
"""
from __future__ import annotations

import os
import zlib

import pytest
import torch

if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"            # must precede the first triton.jit decoration
triton = pytest.importorskip("triton")

from shard.formats.quant import dequant_mxfp4  # noqa: E402
from shard.ops import invariant_attn as ia  # noqa: E402
from shard.ops import invariant_moe as im  # noqa: E402

import triton.language as tl  # noqa: E402

FULL = os.environ.get("SHARD_R4_FULL") == "1"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BF = torch.bfloat16

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")   # masked NaN lanes in the interpreter


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _pick(mixes, fast):
    return mixes if FULL else [m for m in mixes if m[0] in fast]


def _close(out, ref, what):
    """Cross-kernel tolerance class (ARCHITECTURE-v2 §6: different kernels agree to tolerance, never
    bytes): every element within 2 bf16 ulps of the oracle AT THE ROW'S SCALE (ulp of the row's max
    |value|; attention rounds P to bf16, an error that scales with the row, not the element), and
    cosine >= 0.9999. Returns the bit-equal fraction."""
    o, r = out.float().flatten(1), ref.float().flatten(1)
    ulp = lambda v: 2.0 ** (torch.floor(torch.log2(v.abs().clamp(min=2.0 ** -120))) - 7)   # noqa: E731
    tol = 2 * ulp(r.abs().amax(dim=1, keepdim=True))
    worst = ((o - r).abs() / tol).max().item()
    cos = torch.nn.functional.cosine_similarity(o.flatten(), r.flatten(), dim=0).item()
    assert worst <= 1.0 and cos >= 0.9999, f"{what}: worst {worst:.2f} x tol, cos {cos:.7f}"
    return (out == ref).float().mean().item()


# ------------------------------------------------------------------------------------------------
# MoE: MXFP4 W4A16 grouped GEMM
# ------------------------------------------------------------------------------------------------

E, TOP_K, H, INTER = 16, 4, 256, 128


def _experts(seed=0):
    g = _gen(seed)
    w13 = torch.randint(0, 256, (E, 2 * INTER, H // 2), dtype=torch.uint8, generator=g)
    s13 = torch.randint(119, 125, (E, 2 * INTER, H // 32), dtype=torch.uint8, generator=g)
    w2 = torch.randint(0, 256, (E, H, INTER // 2), dtype=torch.uint8, generator=g)
    s2 = torch.randint(119, 125, (E, H, INTER // 32), dtype=torch.uint8, generator=g)
    return w13, s13, w2, s2


EXPERTS = _experts()


def _route(n, g, only=None):
    pool = torch.tensor(only) if only is not None else None
    ids = []
    for _ in range(n):
        if pool is None:
            ids.append(torch.randperm(E, generator=g)[:TOP_K])
        else:
            ids.append(pool[torch.randperm(len(pool), generator=g)[:TOP_K]])
    ids = torch.stack(ids)
    w = torch.rand(n, TOP_K, generator=g) + 0.05
    return ids, w / w.sum(-1, keepdim=True)


def moe_oracle(x, ids, w):
    """Port of mimo_ref.Layer.ffn/_mlp (row-wise fp32 matvec, bf16 at each op output, fp32 combine in
    ascending expert id)."""
    w13, s13, w2, s2 = EXPERTS
    acc = torch.zeros(x.shape[0], H, dtype=torch.float32)
    for e in torch.unique(ids).tolist():
        rows, slot = torch.nonzero(ids == e, as_tuple=True)
        W13 = dequant_mxfp4(w13[e], s13[e])
        Wd = dequant_mxfp4(w2[e], s2[e])
        xr = x[rows].float()
        g = torch.stack([torch.mv(W13[:INTER], r) for r in xr]).to(BF).float()
        u = torch.stack([torch.mv(W13[INTER:], r) for r in xr]).to(BF).float()
        h = (torch.nn.functional.silu(g) * u).to(BF).float()
        y = torch.stack([torch.mv(Wd, r) for r in h]).to(BF)
        acc[rows] += y.float() * w[rows, slot][:, None]
    return acc.to(BF)


def run_moe(x, ids, w, naive=False):
    w13, s13, w2, s2 = (t.to(DEV) for t in EXPERTS)
    return im.moe_experts_forward(x.to(DEV), ids.to(DEV), w.to(DEV), w13, s13, w2, s2, naive=naive).cpu()


N_PROBE = 3
_gp = _gen(1)
PROBE_X = torch.randn(N_PROBE, H, generator=_gp).to(BF)
PROBE_IDS, PROBE_W = _route(N_PROBE, _gp)


def _moe_batch(mix):
    """-> (x, ids, w, rows_of_probe[i]) for a named batch mix. Probes keep their x, ids and weights."""
    name, n, where = mix
    g = _gen(zlib.crc32(name.encode()))
    only = None
    if name == "same_experts":
        only = PROBE_IDS[0].tolist()
    elif name == "disjoint_experts":
        only = [e for e in range(E) if e not in PROBE_IDS[0].tolist()]
    elif name == "one_hot_expert":
        only = [int(PROBE_IDS[0, 0])] + [e for e in range(E) if e not in PROBE_IDS[0].tolist()][:TOP_K - 1]
    x = torch.randn(n, H, generator=g)
    if name == "extreme_neighbours":
        x = x * torch.where(torch.rand(n, 1, generator=g) < 0.5, 1e4, 1e-4)
    x = x.to(BF)
    ids, w = _route(n, g, only)
    rows = []
    for i in range(N_PROBE if where != "dup" else 1):
        r = {"first": i, "last": n - 1 - i, "mid": n // 2 + i}.get(where, None)
        if where == "dup":
            r = [1, n // 2, n - 1]
        rows.append(r)
    if where == "dup":
        for r in rows[0]:
            x[r], ids[r], w[r] = PROBE_X[0], PROBE_IDS[0], PROBE_W[0]
        return x, ids, w, [(0, r) for r in rows[0]]
    for i, r in enumerate(rows):
        x[r], ids[r], w[r] = PROBE_X[i], PROBE_IDS[i], PROBE_W[i]
    return x, ids, w, list(enumerate(rows))


MOE_MIXES = [   # (name, batch size, probe placement) -- 13 mixes
    ("probes_only", 3, "first"),
    ("pair", 4, "last"),
    ("small", 7, "mid"),
    ("cross_tile", 17, "last"),
    ("same_experts", 40, "mid"),
    ("disjoint_experts", 40, "first"),
    ("one_hot_expert", 33, "mid"),
    ("dup_probe", 20, "dup"),
    ("extreme_neighbours", 9, "mid"),
    ("medium", 48, "last"),
    ("large", 100, "mid"),
    ("large_first", 64, "first"),
    ("odd", 23, "last"),
]


@pytest.fixture(scope="module")
def moe_baseline():
    """Each probe alone, B=1: the replay a challenger would run."""
    return [run_moe(PROBE_X[i:i + 1], PROBE_IDS[i:i + 1], PROBE_W[i:i + 1])[0] for i in range(N_PROBE)]


def test_moe_grouped_gemm_matches_oracle():
    g = _gen(7)
    x = torch.randn(20, H, generator=g).to(BF)
    ids, w = _route(20, g)
    frac = _close(run_moe(x, ids, w), moe_oracle(x, ids, w), "moe vs oracle")
    assert frac > 0.9, f"bit-equal fraction vs oracle only {frac:.3f}"


def test_moe_weights_stay_packed_and_dequant_exact():
    """In-kernel decode == shard.formats.quant.dequant_mxfp4 bit for bit (identity activations pull the
    dequantized weight columns out through the GEMM)."""
    w13, s13 = EXPERTS[0][:2], EXPERTS[1][:2]
    eye = torch.eye(H).to(BF)
    ids = torch.ones(H, 1, dtype=torch.long)
    out = im.mxfp4_grouped_gemm(eye.to(DEV), w13.to(DEV), s13.to(DEV), ids.to(DEV), a_div=1).cpu()
    assert torch.equal(out.float(), dequant_mxfp4(w13[1], s13[1]).T)


@pytest.mark.parametrize("mix", _pick(MOE_MIXES, {"probes_only", "cross_tile", "one_hot_expert", "dup_probe"}),
                         ids=lambda m: m[0])
def test_moe_probe_rows_byte_identical_across_mixes(mix, moe_baseline):
    x, ids, w, rows = _moe_batch(mix)
    out = run_moe(x, ids, w)
    for i, r in rows:
        assert torch.equal(out[r].view(torch.int16), moe_baseline[i].view(torch.int16)), \
            f"mix {mix[0]}: probe {i} at row {r} changed bytes"


def test_moe_probe_every_m_tile_slot(moe_baseline):
    """The probe lands in every row slot of its expert's BLOCK_M tile (neighbours routed to the same
    experts, probe moved from slot 0 to BLOCK_M-1): bytes never move."""
    g = _gen(31)
    for j in (range(im.BLOCK_M) if FULL else (0, 7, im.BLOCK_M - 1)):
        x = torch.randn(im.BLOCK_M, H, generator=g).to(BF)
        ids = PROBE_IDS[0].repeat(im.BLOCK_M, 1)
        w = torch.rand(im.BLOCK_M, TOP_K, generator=g) + 0.1
        x[j], w[j] = PROBE_X[0], PROBE_W[0]
        out = run_moe(x, ids, w)
        assert torch.equal(out[j].view(torch.int16), moe_baseline[0].view(torch.int16)), f"slot {j}"


def test_moe_guards():
    """Out-of-range expert ids are refused (their rows would never be written); T=0 is empty; scale codes
    that decode to overflow or subnormal weights are refused at load."""
    w13, s13, w2, s2 = EXPERTS
    x = torch.randn(2, H).to(BF)
    with pytest.raises(ValueError):
        im.moe_experts_forward(x, torch.tensor([[0, 1, 2, E]]).repeat(2, 1), torch.ones(2, TOP_K),
                               w13, s13, w2, s2)
    assert im.moe_experts_forward(x[:0], torch.zeros(0, TOP_K, dtype=torch.long), torch.zeros(0, TOP_K),
                                  w13, s13, w2, s2).shape == (0, H)
    im.validate_expert_scales(s13)
    for bad in (0, 2, 253, 255):
        with pytest.raises(ValueError):
            im.validate_expert_scales(torch.full((2, 2), bad, dtype=torch.uint8))


@triton.jit
def _rne_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, im.f32_to_bf16_rne(tl.load(x_ptr + offs, mask=m)), mask=m)


def test_rne_helper_matches_torch_and_keeps_nan():
    """f32_to_bf16_rne == torch's RNE cast for finite values/inf (incl. ties, negatives, subnormals,
    overflow), and every NaN pattern -- including NVIDIA's canonical 0x7FFFFFFF, which a bare carry would
    turn into -0.0 -- stays NaN."""
    g = _gen(3)
    bits = torch.cat([torch.randint(-2**31, 2**31 - 1, (4000,), generator=g, dtype=torch.int64),
                      torch.tensor([0x7FFFFFFF, 0xFFFFFFFF - 2**32, 0x7FC00000, 0xFFC00000 - 2**32, 0x7F800001,
                                    0x7F800000, 0xFF800000 - 2**32, 0x7F7FFFFF, 0x00008000, 0x00018000,
                                    0x3F808000, 0x3F818000, 0x00000001, 0x80000000 - 2**32])])
    x = bits.to(torch.int32).view(torch.float32)
    out = torch.empty(x.numel(), dtype=BF)
    _rne_kernel[(1,)](x.to(DEV), out_dev := out.to(DEV), x.numel(), BLOCK=triton.next_power_of_2(x.numel()))
    out = out_dev.cpu()
    nan = torch.isnan(x)
    assert torch.isnan(out[nan]).all()
    assert torch.equal(out[~nan].view(torch.int16), x[~nan].to(BF).view(torch.int16))


def test_moe_row_independence_under_finite_perturbation(moe_baseline):
    x, ids, w, rows = _moe_batch(("medium", 24, "mid"))
    probe_rows = {r for _, r in rows}
    g = _gen(99)
    for scale in (1e3, -7.0, 1e-3):
        xp = x.clone()
        for r in range(x.shape[0]):
            if r not in probe_rows:
                xp[r] = (x[r].float() * scale + torch.randn(H, generator=g)).to(BF)
        idp, wp = ids.clone(), w.clone()
        nids, nw = _route(x.shape[0], g)
        for r in range(x.shape[0]):
            if r not in probe_rows:
                idp[r], wp[r] = nids[r], nw[r]
        out = run_moe(xp, idp, wp)
        for i, r in rows:
            assert torch.equal(out[r].view(torch.int16), moe_baseline[i].view(torch.int16))


def test_moe_naive_split_k_control_fails_invariance():
    """Anti-vacuity: the same 8 tokens run as one batch vs as 2 batches of 4. The pinned kernel gives
    identical bytes; a split-K chosen from the token count (the fast-kernel heuristic: 4-way split-K for
    <= 4 tokens, partials in the output dtype as in Triton's split-K atomic_add recipe, pairwise-tree
    reduction) does not."""
    g = _gen(1234)
    x = torch.randn(8, H, generator=g).to(BF)
    ids, w = _route(8, g)
    whole = run_moe(x, ids, w)
    chunked = torch.cat([run_moe(x[i:i + 4], ids[i:i + 4], w[i:i + 4]) for i in range(0, 8, 4)])
    assert torch.equal(whole.view(torch.int16), chunked.view(torch.int16))
    n_whole = run_moe(x, ids, w, naive=True)
    n_chunk = torch.cat([run_moe(x[i:i + 4], ids[i:i + 4], w[i:i + 4], naive=True) for i in range(0, 8, 4)])
    assert torch.equal(n_whole.view(torch.int16), whole.view(torch.int16))       # split_k=1 at B=64
    rows_changed = (n_chunk.view(torch.int16) != n_whole.view(torch.int16)).any(dim=1).sum().item()
    assert rows_changed > 0, "naive batch-dependent split-K did not change any row: the test is vacuous"


# ------------------------------------------------------------------------------------------------
# attention: fixed-split diff-kv (192/128), sinks, window 128 / global, paged KV
# ------------------------------------------------------------------------------------------------

SPLIT = 64       # small pinned split so interpreter-sized contexts cross several splits
SEQ_LENS = [40, 170, 300, 90]           # KV already written for these many positions per session
PROBE_TOKENS = [(1, 149), (2, 299), (2, 130)]   # (session, position); session 1 has keys beyond 149
CONFIGS = {  # Flash group ratios: SWA 64/8 = 8 q heads per kv head (sinks), global 64/4 = 16 (no sinks)
    "swa": dict(hq=16, hkv=2, window=128, sinks=True),
    "global": dict(hq=16, hkv=1, window=0, sinks=False),
}


def _kv(cfg, seed=3):
    g = _gen(seed)
    ks = [torch.randn(n, cfg["hkv"], 192, generator=g).to(BF) for n in SEQ_LENS]
    vs = [torch.randn(n, cfg["hkv"], 128, generator=g).to(BF) for n in SEQ_LENS]
    return ks, vs


def _paged(ks, vs, page, seed):
    """Lay the sessions' KV into a shared page pool in a shuffled order (plus spare garbage pages)."""
    g = _gen(seed)
    hkv = ks[0].shape[1]
    npg = [-(-k.shape[0] // page) for k in ks]
    total = sum(npg) + 3
    perm = torch.randperm(total, generator=g)
    kc = torch.randn(total, page, hkv, 192, generator=g).to(BF)       # garbage in unused slots
    vc = torch.randn(total, page, hkv, 128, generator=g).to(BF)
    bt = torch.zeros(len(ks), max(npg), dtype=torch.int32)
    c = 0
    for s, (k, v) in enumerate(zip(ks, vs)):
        for j in range(npg[s]):
            pg = int(perm[c])
            c += 1
            bt[s, j] = pg
            n = min(page, k.shape[0] - j * page)
            kc[pg, :n], vc[pg, :n] = k[j * page:j * page + n], v[j * page:j * page + n]
    return kc, vc, bt


def _sinks(cfg):
    return torch.randn(cfg["hq"], generator=_gen(5)) * 2 if cfg["sinks"] else None


def attn_oracle(q, toks, ks, vs, cfg, sinks):
    """Port of mimo_ref.Layer.attention's per-query loop (fp32 scores, softmax incl. the sink logit,
    fp32 PV, one bf16 rounding)."""
    hq, hkv = cfg["hq"], cfg["hkv"]
    grp = hq // hkv
    out = torch.empty(len(toks), hq, 128, dtype=BF)
    for t, (s, p) in enumerate(toks):
        lo = max(0, p - cfg["window"] + 1) if cfg["window"] else 0
        kv = ks[s][lo:p + 1].transpose(0, 1).float()
        vv = vs[s][lo:p + 1].transpose(0, 1).float()
        qt = q[t].float().view(hkv, grp, 192)
        sc = torch.bmm(qt, kv.transpose(1, 2)) * 192 ** -0.5
        if sinks is not None:
            sc = torch.cat([sc, sinks.view(hkv, grp, 1)], dim=-1)
        pr = torch.softmax(sc, dim=-1)[..., :kv.shape[1]]
        out[t] = torch.bmm(pr, vv).reshape(hq, 128).to(BF)
    return out


def run_attn(q, toks, kc, vc, bt, cfg, sinks, split=SPLIT):
    seq = torch.tensor([s for s, _ in toks], dtype=torch.int32)
    pos = torch.tensor([p for _, p in toks], dtype=torch.int32)
    return ia.paged_diffkv_attention(q.to(DEV), kc.to(DEV), vc.to(DEV), bt.to(DEV), seq.to(DEV), pos.to(DEV),
                                     window=cfg["window"], sinks=None if sinks is None else sinks.to(DEV),
                                     split=split).cpu()


def _probe_q(cfg):
    return torch.randn(len(PROBE_TOKENS), cfg["hq"], 192, generator=_gen(11)).to(BF)


def _attn_mix(name, cfg, g):
    """-> list of (session, pos) tokens with the probes inserted, and where each probe sits.
    Segments: prefill = a run of consecutive positions, verify = k+1 rows whose later keys exist,
    decode = one row at the session's last written position."""
    dec = lambda s: [(s, SEQ_LENS[s] - 1)]                                   # noqa: E731
    run = lambda s, a, b: [(s, p) for p in range(a, b)]                      # noqa: E731
    probes = list(PROBE_TOKENS)
    body = {
        "probes_only": [],
        "one_decode": dec(0),
        "all_decodes": dec(0) + dec(1) + dec(2) + dec(3),
        "prefill_own_session": run(1, 100, 149) + run(1, 150, 160),          # probe inside a prefill chunk
        "verify_segment": run(1, 145, 149) + run(1, 150, 154) + run(2, 295, 299),
        "mixed_segments": run(3, 0, 32) + dec(0) + run(1, 160, 168) + dec(2),
        "big_batch": [(int(torch.randint(0, 4, (1,), generator=g)), 0) for _ in range(8)]
        + run(3, 30, 70) + run(0, 0, 20),
        "pos_zero_neighbours": [(s, 0) for s in range(4)] * 3,
        "long_neighbour_only_global": dec(2) + dec(2),                        # max split count padding
        "reversed": list(reversed(run(2, 200, 230))) + dec(3),
        "dup_probe": list(PROBE_TOKENS) + dec(1),
        "perturb_small": dec(0) + dec(3) + run(1, 150, 154) + run(0, 30, 34),
    }[name]
    toks = list(body)
    at = []
    for i, pt in enumerate(probes):
        idx = int(torch.randint(0, len(toks) + 1, (1,), generator=g))
        toks.insert(idx, pt)
        at = [(j, a + (1 if a >= idx else 0)) for j, a in at] + [(i, idx)]
    return toks, at


ATTN_MIXES = ["probes_only", "one_decode", "all_decodes", "prefill_own_session", "verify_segment",
              "mixed_segments", "big_batch", "pos_zero_neighbours", "long_neighbour_only_global",
              "reversed", "dup_probe"]
LAYOUTS = [("page16", 16, 21), ("page32_reshuffled", 32, 22), ("page8", 8, 23)]   # "how the KV was chunked"


@pytest.fixture(scope="module", params=list(CONFIGS) if FULL else ["swa", "global"])
def attn_env(request):
    cfg = CONFIGS[request.param]
    ks, vs = _kv(cfg)
    sinks = _sinks(cfg)
    kc, vc, bt = _paged(ks, vs, 16, 21)
    pq = _probe_q(cfg)
    base = [run_attn(pq[i:i + 1], [PROBE_TOKENS[i]], kc, vc, bt, cfg, sinks)[0] for i in range(len(PROBE_TOKENS))]
    return dict(cfg=cfg, ks=ks, vs=vs, sinks=sinks, kc=kc, vc=vc, bt=bt, pq=pq, base=base, name=request.param)


def _neighbour_q(n, cfg, g):
    return torch.randn(n, cfg["hq"], 192, generator=g).to(BF)


def _run_mix(env, name, layout=LAYOUTS[0], qscale=1.0, split=SPLIT):
    cfg = env["cfg"]
    g = _gen(zlib.crc32(name.encode()))
    toks, at = _attn_mix(name, cfg, g)
    q = (_neighbour_q(len(toks), cfg, g).float() * qscale).to(BF)
    for i, idx in at:
        q[idx] = env["pq"][i]
    if layout[0] == "page16":
        kc, vc, bt = env["kc"], env["vc"], env["bt"]
    else:
        kc, vc, bt = _paged(env["ks"], env["vs"], layout[1], layout[2])
    return run_attn(q, toks, kc, vc, bt, cfg, env["sinks"], split=split), at, q, toks


def test_attn_matches_oracle(attn_env):
    env = attn_env
    out, _, q, toks = _run_mix(env, "mixed_segments")
    ref = attn_oracle(q, toks, env["ks"], env["vs"], env["cfg"], env["sinks"])
    frac = _close(out, ref, f"attn {env['name']} vs oracle")
    assert frac > 0.5, f"bit-equal fraction vs oracle only {frac:.3f}"


@pytest.mark.parametrize("mix", ATTN_MIXES if FULL else ["probes_only", "verify_segment", "long_neighbour_only_global"])
def test_attn_probe_rows_byte_identical_across_mixes(attn_env, mix):
    out, at, _, _ = _run_mix(attn_env, mix)
    for i, idx in at:
        assert torch.equal(out[idx].view(torch.int16), attn_env["base"][i].view(torch.int16)), \
            f"{attn_env['name']} mix {mix}: probe {PROBE_TOKENS[i]} at row {idx} changed bytes"


@pytest.mark.parametrize("layout", LAYOUTS[1:] if FULL else LAYOUTS[1:2], ids=lambda l: l[0])
def test_attn_invariant_to_kv_page_layout(attn_env, layout):
    out, at, _, _ = _run_mix(attn_env, "mixed_segments" if FULL else "verify_segment", layout=layout)
    for i, idx in at:
        assert torch.equal(out[idx].view(torch.int16), attn_env["base"][i].view(torch.int16))


def test_attn_row_independence_under_finite_perturbation(attn_env):
    """Other tokens' q (huge/tiny), other sessions' KV, AND the probe session's keys the probe must not see
    (future/draft positions and, for SWA, positions before the window) are all perturbed with finite
    values: the probe bytes do not move."""
    env = attn_env
    cfg = env["cfg"]
    g = _gen(77)
    ks = [k.clone() for k in env["ks"]]
    vs = [v.clone() for v in env["vs"]]
    for s in (0, 3):
        ks[s] = (torch.randn(ks[s].shape, generator=g) * 50).to(BF)
        vs[s] = (torch.randn(vs[s].shape, generator=g) * 50).to(BF)
    ks[1][150:] = (torch.randn(ks[1][150:].shape, generator=g) * 1e3).to(BF)       # after probe (1,149)
    vs[1][150:] = (torch.randn(vs[1][150:].shape, generator=g) * 1e3).to(BF)
    if cfg["window"]:
        ks[1][:149 - 128 + 1] = (torch.randn(ks[1][:22].shape, generator=g) * 1e3).to(BF)
        vs[1][:22] = (torch.randn(vs[1][:22].shape, generator=g) * 1e3).to(BF)
    kc, vc, bt = _paged(ks, vs, 16, 21)
    for qscale in (1e3, 1e-3):
        toks, at = _attn_mix("perturb_small", cfg, _gen(5))
        q = (_neighbour_q(len(toks), cfg, g).float() * qscale).to(BF)
        for i, idx in at:
            q[idx] = env["pq"][i]
        out = run_attn(q, toks, kc, vc, bt, cfg, env["sinks"])
        i, idx = at[0]                                                      # probe (1, 149)
        assert PROBE_TOKENS[i] == (1, 149)
        assert torch.equal(out[idx].view(torch.int16), env["base"][i].view(torch.int16))


def test_attn_naive_batch_dependent_split_fails_invariance(attn_env):
    """Anti-vacuity: split size chosen from the token count (FlashDecoding-style) changes probe bytes."""
    env = attn_env
    alone = [run_attn(env["pq"][i:i + 1], [PROBE_TOKENS[i]], env["kc"], env["vc"], env["bt"], env["cfg"],
                      env["sinks"], split=ia.naive_split(1))[0] for i in range(len(PROBE_TOKENS))]
    g = _gen(zlib.crc32(b"all_decodes"))
    toks, _ = _attn_mix("all_decodes", env["cfg"], g)
    out, at, _, _ = _run_mix(env, "all_decodes", split=ia.naive_split(len(toks)))
    changed = [not torch.equal(out[idx].view(torch.int16), alone[i].view(torch.int16)) for i, idx in at]
    assert any(changed), "naive batch-dependent split did not change any probe: the test is vacuous"


# ------------------------------------------------------------------------------------------------
# HW2 measurement stub: GPU-less end to end
# ------------------------------------------------------------------------------------------------

def test_hw2_cost_stub_smoke(tmp_path):
    """The HW2 T2/T4 script runs on CPU (tiny shapes), emits its schema, never claims a speed verdict from
    interpreter timings, and its on-device invariance probe passes here."""
    import json

    from shard.ops import hw2_invariant_cost as hw2
    out = tmp_path / "hw2.json"
    res = hw2.main(["--smoke", "--out", str(out)])
    assert json.loads(out.read_text())["summary"] == res["summary"]
    assert {r["op"] for r in res["rows"]} == {"moe", "attn_global", "attn_swa"}
    assert all(r["ratio"] > 0 for r in res["rows"])
    assert res["summary"]["pass_ratio"] is None and res["meta"]["timings_meaningful"] is False
    assert res["summary"]["pass_invariance"] is True
