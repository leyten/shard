"""CPU tests for the TREE-frame CUDA-graph lever (_TGraphState / TreeRowGraphRunner / _block_tree_row).

Per-stream trees raised g +15-70% but LOST live because tree frames ran eager stage-side (154ms vs
45ms graphed chains — receipt perstream-trees-ab-20260712). The unlock pads each tree frame to the
fixed M25_TREE_PAD template and captures one graph per context bucket; these tests certify the
static-buffer math the captured kernels consume — GATE 0 of the correctness hierarchy, and the one
that catches the silent-wrong-commit class: a template-mapping off-by-one is mask DATA, so the
on-box graph-vs-eager-padded bit-equality gate would pass perfectly while committing WRONG tokens
with valid receipts. On CPU we pin:

  * _TGraphState.set == the eager tree math on the REAL region (mask VERBATIM from build_tree_mask,
    write columns, per-node RoPE rows), across randomized topologies and refreshes;
  * the dummy-node rules: KV writes land CONTIGUOUSLY at [start+n, start+npad) (the speculative-
    junk-past-the-frontier class the dirty-frontier contract already handles — never another row,
    never a committed slot below start), dummy rows clone node 0's mask row (bounded values, no
    all--inf NaN row);
  * padded attention == unpadded attention on the real rows (padding is attention-inert), and junk
    in the masked pad columns provably cannot influence real rows;
  * _block_tree_row routing: template gate, oversize-N/hatch/solo/off -> eager, the tree graphs'
    OWN budget (M25_TREE_GRAPH_MAX) with once-per-shape cap logging.

GPU-real behaviour (capture/replay bit-equality vs run_eager_ref, MoE token-count numerics, argmax
agreement, timing) is validated on-box by research/tree_graph_check.py (gates 1-2) and the ring A/B
(gate 3).

Run: python3 -m pytest tests/test_tree_graph.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
fr = pytest.importorskip("fake_ring")               # bootstraps env (fake M25_DIR) + imports m25_pipe on CPU

MP = fr.MP
S = fr.S

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase0"))
from tree_spec import build_tree_mask, _gqa_masked_attend            # noqa: E402

NKV, HD, GRP = 4, 16, 2
NH = NKV * GRP
NEG = float("-inf")


def _tree(n, seed=0):
    """A random valid tree: parents before children (node i's parent < i or -1), depths derived."""
    g = torch.Generator().manual_seed(seed)
    parents = [-1]
    for i in range(1, n):
        parents.append(int(torch.randint(-1, i, (1,), generator=g)))
    depths = []
    for p in parents:
        depths.append(1 if p < 0 else depths[p] + 1)
    return parents, depths


def _set(st, row, start, parents, depths, full_cos, full_sin):
    pos_ids = [(start - 1) + d for d in depths]
    st.set(row, start, len(parents), parents, pos_ids, full_cos, full_sin)
    return pos_ids


def _pe(rd=8, seed=0, maxpos=256):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(maxpos, rd, generator=g).to(torch.bfloat16),
            torch.randn(maxpos, rd, generator=g).to(torch.bfloat16))


# ---- 1. _TGraphState.set == the eager tree math on the REAL region --------------------------------

@pytest.mark.parametrize("n,start,seed", [(5, 9, 1), (8, 1, 2), (3, 40, 3), (8, 24, 4)])
def test_tgraphstate_real_region_matches_eager(n, start, seed, monkeypatch):
    monkeypatch.setattr(S, "NKV", NKV)
    npad, alen, rd, row = 8, 64, 8, 2
    cos, sin = _pe(rd, seed)
    parents, depths = _tree(n, seed)
    st = S._TGraphState(npad, alen, rd, "cpu")
    pos_ids = _set(st, row, start, parents, depths, cos, sin)
    # write columns: the full contiguous pad span (real at their exact eager slots, dummies after)
    assert st.wcp.tolist() == list(range(start, start + npad))
    # row index
    assert st.rows.tolist() == [row * NKV + i for i in range(NKV)]
    # per-node RoPE rows == the eager gather on the real region; dummies ride node 0's position
    assert torch.equal(st.cos[0, 0, :n], cos[torch.as_tensor(pos_ids)])
    assert torch.equal(st.sin[0, 0, :n], sin[torch.as_tensor(pos_ids)])
    assert torch.equal(st.cos[0, 0, n:], cos[pos_ids[0]].expand(npad - n, rd))
    # mask real region VERBATIM == build_tree_mask's full [n, start+n] bias
    m, _ = build_tree_mask(parents, depths, start, n)
    assert torch.equal(st.mask[0, 0, :n, :start + n], m[0, 0].to(torch.bfloat16))
    # real rows: everything past the written span is blocked (pad columns + bucket tail)
    assert (st.mask[0, 0, :n, start + n:] == NEG).all()
    # dummy rows clone node 0's mask row (bounded real-token computation, never an all--inf row)
    for j in range(n, npad):
        assert torch.equal(st.mask[0, 0, j], st.mask[0, 0, 0])


def test_tgraphstate_set_fully_overwrites(monkeypatch):
    monkeypatch.setattr(S, "NKV", NKV)
    npad, alen, rd = 8, 64, 8
    cos, sin = _pe(rd, 5)
    st = S._TGraphState(npad, alen, rd, "cpu")
    pa, da = _tree(8, 6)
    _set(st, 0, 20, pa, da, cos, sin)                # full frame first (no dummies)
    pb, db = _tree(4, 7)
    _set(st, 3, 7, pb, db, cos, sin)                 # then a smaller one — every buffer must refresh
    ref = S._TGraphState(npad, alen, rd, "cpu")
    _set(ref, 3, 7, pb, db, cos, sin)
    for a, b in [(st.wcp, ref.wcp), (st.rows, ref.rows), (st.cos, ref.cos),
                 (st.sin, ref.sin), (st.mask, ref.mask)]:
        assert torch.equal(a, b), "stale first-set() values survived the refresh"


def test_rgraphstate_has_no_wcp():
    """attn_tree_row's graph branch keys on hasattr(gr, 'wcp') — the chain-row state must never
    satisfy it (a chain capture routed down the tree branch would write through tree indices)."""
    assert not hasattr(S._RGraphState(4, 32, 8, "cpu"), "wcp")


# ---- 2. the padded WRITE: real slots exact, dummies contiguous past the read window ----------------

def test_padded_write_is_contiguous_and_row_local():
    B, npad, n, start, row, maxlen = 3, 8, 5, 9, 1, 64
    g = torch.Generator().manual_seed(8)
    bkc = torch.randn(B, NKV, maxlen, HD, generator=g).to(torch.bfloat16)   # pre-existing live KV
    before = bkc.clone()
    k = torch.randn(1, NKV, npad, HD, generator=g).to(torch.bfloat16)       # padded frame's K
    wcp = start + torch.arange(npad)
    rows_i = row * NKV + torch.arange(NKV)
    kf = bkc.view(-1, maxlen, HD)
    kf[rows_i[:, None], wcp[None, :]] = k[0]                                # the captured write, on CPU
    assert torch.equal(bkc[row, :, start:start + npad], k[0]), "pad-span write not exact"
    # ONLY [start, start+npad) of row `row` changed: the committed prefix below start untouched,
    # slots past the pad span untouched, other streams' rows untouched
    assert torch.equal(bkc[row, :, :start], before[row, :, :start]), "write reached below start"
    assert torch.equal(bkc[row, :, start + npad:], before[row, :, start + npad:]), "write past the pad span"
    others = [r for r in range(B) if r != row]
    assert torch.equal(bkc[others], before[others]), "padded write leaked into another stream's row"


def test_dummy_junk_cannot_influence_masked_attention():
    """The junk-slot class proof: perturbing the dummy columns [start+n, start+npad) (and the whole
    unwritten bucket tail) must not change any real row's output — they are -inf-masked, and masked
    FINITE values contribute exactly 0."""
    n, npad, start, alen = 5, 8, 9, 32
    g = torch.Generator().manual_seed(9)
    parents, depths = _tree(n, 9)
    q = torch.randn(1, NH, npad, HD, generator=g).to(torch.bfloat16)
    kc = torch.randn(1, NKV, alen, HD, generator=g).to(torch.bfloat16)
    vc = torch.randn(1, NKV, alen, HD, generator=g).to(torch.bfloat16)
    m, _ = build_tree_mask(parents, depths, start, n)
    mp = torch.full((1, 1, npad, alen), NEG)
    mp[0, 0, :n, :start + n] = m[0, 0]
    mp[0, 0, n:] = mp[0, 0, 0].clone()
    a = _gqa_masked_attend(q, kc, vc, mp.to(torch.bfloat16), GRP)
    kc2, vc2 = kc.clone(), vc.clone()
    kc2[:, :, start + n:] = 123.0; vc2[:, :, start + n:] = -77.0            # junk the masked span
    b = _gqa_masked_attend(q, kc2, vc2, mp.to(torch.bfloat16), GRP)
    assert torch.equal(a[:, :, :n], b[:, :, :n]), "masked junk influenced a real row"


# ---- 3. padded attention is attention-INERT on the real rows --------------------------------------

@pytest.mark.parametrize("n,npad", [(5, 8), (8, 8), (3, 16)])
def test_padded_attend_matches_unpadded_real_rows(n, npad):
    """The eager oracle at the kernel level: _gqa_masked_attend over the padded (npad rows, alen
    cols) shape must equal the unpadded (n rows, total cols) computation on the real rows — pad rows
    are independent (per-row softmax) and pad columns are -inf (exactly-0 weight on FINITE K/V)."""
    start, alen = 9, 32
    total = start + n
    g = torch.Generator().manual_seed(10)
    parents, depths = _tree(n, 10)
    q = torch.randn(1, NH, n, HD, generator=g).to(torch.bfloat16)
    kc = torch.randn(1, NKV, alen, HD, generator=g).to(torch.bfloat16)    # [total:] = finite junk
    vc = torch.randn(1, NKV, alen, HD, generator=g).to(torch.bfloat16)
    m, _ = build_tree_mask(parents, depths, start, n)
    want = _gqa_masked_attend(q, kc[:, :, :total], vc[:, :, :total], m.to(torch.bfloat16), GRP)
    qp = torch.empty(1, NH, npad, HD).to(torch.bfloat16)
    qp[:, :, :n] = q
    qp[:, :, n:] = q[:, :, :1]                                            # dummies clone node 0
    mp = torch.full((1, 1, npad, alen), NEG)
    mp[0, 0, :n, :total] = m[0, 0]
    mp[0, 0, n:] = mp[0, 0, 0].clone()
    got = _gqa_masked_attend(qp, kc, vc, mp.to(torch.bfloat16), GRP)
    assert torch.equal(got[:, :, :n], want), \
        "padding changed a real row's attention output (mask/shape leak — not attention-inert)"
    assert torch.isfinite(got).all(), "a pad row produced non-finite output"


# ---- 4. _block_tree_row routing --------------------------------------------------------------------

class FakeTGR:
    made = []

    def __init__(self, layers, vcfg, npad, dv="cpu"):
        self.npad = npad
        self.graphs = {}
        self.runs = []
        FakeTGR.made.append(self)

    def _bucket(self, total):
        for b in S.DECODE_BUCKETS:
            if b >= total:
                return min(b, S.M25_KV_MAXLEN)
        return S.M25_KV_MAXLEN

    def run(self, row, start, x, parents, pos_ids):
        alen = self._bucket(start + x.shape[1])
        if alen not in self.graphs:
            self.graphs[alen] = "g"
            S._TREE_GRAPH_COUNT += 1
        self.runs.append((row, start, x.shape[1], alen))
        return ("tgraph", self.npad, x.shape[1])


@pytest.fixture
def tgraph_env(monkeypatch):
    eager_calls = []

    def fake_eager(layers, row, start, x, vcfg, parents, pos_ids):
        eager_calls.append((row, start, x.shape[1]))
        return ("eager", x.shape[1])

    monkeypatch.setattr(S, "M25_CUDA_GRAPH_ACTIVE", True)
    monkeypatch.setattr(S, "M25_BATCH", 4)
    monkeypatch.setattr(S, "M25_TREE_GRAPH", True)
    monkeypatch.setattr(S, "M25_TREE_PAD", 24)
    monkeypatch.setattr(S, "M25_TREE_GRAPH_MAX", 2)
    monkeypatch.setattr(S, "_TREE_GRAPH_COUNT", 0)
    monkeypatch.setattr(S, "_GRAPH_SKIPPED", 0)
    monkeypatch.setattr(S, "TreeRowGraphRunner", FakeTGR)
    monkeypatch.setattr(S, "run_block_tree_row", fake_eager)
    monkeypatch.setattr(MP, "M25_BATCH_GRAPH", True)
    monkeypatch.setattr(MP, "_GRAPH_CAP_LOGGED", set())
    FakeTGR.made = []
    return eager_calls


def _tx(n):
    return torch.zeros(1, n, 4)


def test_tree_route_uses_the_template_and_caches(tgraph_env):
    grs = {}
    assert MP._block_tree_row(grs, [], 0, 100, _tx(13), None, [-1], [1]) == ("tgraph", 24, 13)
    assert MP._block_tree_row(grs, [], 1, 200, _tx(21), None, [-1], [1]) == ("tgraph", 24, 21)
    assert MP._block_tree_row(grs, [], 2, 300, _tx(24), None, [-1], [1]) == ("tgraph", 24, 24)
    assert list(grs) == [24] and len(FakeTGR.made) == 1, "one runner serves every frame size"
    assert S._TREE_GRAPH_COUNT == 1 and tgraph_env == []       # one bucket -> one graph


def test_tree_route_oversize_frame_is_eager(tgraph_env):
    assert MP._block_tree_row({}, [], 0, 100, _tx(25), None, [-1], [1]) == ("eager", 25)
    assert FakeTGR.made == [] and tgraph_env == [(0, 100, 25)]


def test_tree_route_hatch_solo_and_inactive_are_eager(tgraph_env, monkeypatch):
    monkeypatch.setattr(S, "M25_TREE_GRAPH", False)             # the escape hatch
    assert MP._block_tree_row({}, [], 0, 100, _tx(13), None, [-1], [1]) == ("eager", 13)
    monkeypatch.setattr(S, "M25_TREE_GRAPH", True)
    monkeypatch.setattr(S, "M25_BATCH", 1)                      # solo ring: no [B,...] rows
    assert MP._block_tree_row({}, [], 0, 100, _tx(13), None, [-1], [1]) == ("eager", 13)
    monkeypatch.setattr(S, "M25_BATCH", 4)
    monkeypatch.setattr(S, "M25_CUDA_GRAPH_ACTIVE", False)      # graphs off for the job
    assert MP._block_tree_row({}, [], 0, 100, _tx(13), None, [-1], [1]) == ("eager", 13)
    assert FakeTGR.made == [] and S._TREE_GRAPH_COUNT == 0


def test_tree_route_respects_own_budget(tgraph_env, capsys):
    grs = {}
    MP._block_tree_row(grs, [], 0, 100, _tx(13), None, [-1], [1])        # bucket 2048: capture #1
    MP._block_tree_row(grs, [], 0, 3000, _tx(13), None, [-1], [1])       # bucket 4096: capture #2 — budget spent
    assert S._TREE_GRAPH_COUNT == 2
    out = MP._block_tree_row(grs, [], 0, 5000, _tx(13), None, [-1], [1])  # NEW bucket 8192 -> eager, logged once
    assert out == ("eager", 13) and S._GRAPH_SKIPPED == 1
    assert capsys.readouterr().out.count("[graph] cap:") == 1
    MP._block_tree_row(grs, [], 0, 5100, _tx(13), None, [-1], [1])       # repeat skip: counted, NOT re-logged
    assert S._GRAPH_SKIPPED == 2 and "[graph] cap:" not in capsys.readouterr().out
    out = MP._block_tree_row(grs, [], 0, 150, _tx(13), None, [-1], [1])  # captured buckets keep replaying
    assert out == ("tgraph", 24, 13) and S._TREE_GRAPH_COUNT == 2


def test_tree_pad_default_covers_the_drafting_config():
    """The derived template must hold the biggest legal frame: trunk (tree_depth+1) + tree_m nodes.
    At the M=12/depth=8 defaults that is 21 -> npad 24."""
    assert S._TREE_NPAD_DEFAULT >= 12 + 8 + 1
    assert S._TREE_NPAD_DEFAULT % 8 == 0
