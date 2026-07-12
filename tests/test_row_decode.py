"""CPU tests for the de-lockstep row-decode KV math (adversarial-review adoption, 2026-07-11):
the flattened-view advanced-index WRITE must equal attn_decode_b's scatter_ for the same row —
the review's MAJOR-2 was a transposed RHS that crashed at s != NKV and silently wrote
head/position-TRANSPOSED KV at the s == NKV coincidence (verify-path corruption, valid receipts).
No Layer construction needed: the invariant is pure tensor math.

Run: python3 -m pytest tests/test_row_decode.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")

B, NKV, MAXLEN, HD = 4, 8, 256, 16


def scatter_reference(bkc, b, start, k):
    """attn_decode_b's write for one row: scatter_ along dim 2 with per-position indices."""
    s = k.shape[2]
    cp = (start + torch.arange(s)).view(1, 1, s, 1).expand(1, NKV, s, HD)
    bkc[b:b + 1].scatter_(2, cp, k)


def row_write(bkc, row, start, k):
    """attn_decode_row's write: flattened-view advanced indexing (k is [1,NKV,s,HD])."""
    s = k.shape[2]
    kf = bkc.view(-1, MAXLEN, HD)
    cp = start + torch.arange(s)
    rows_i = row * NKV + torch.arange(NKV)
    kf[rows_i[:, None], cp[None, :]] = k[0]


@pytest.mark.parametrize("s", [9, 8, 7, 5])          # incl. s == NKV (the silent-transpose coincidence)
@pytest.mark.parametrize("row", [0, 1, 3])
def test_row_write_equals_scatter(s, row):
    g = torch.Generator().manual_seed(s * 10 + row)
    k = (torch.randn(1, NKV, s, HD, generator=g)).to(torch.bfloat16)
    start = 37
    a = torch.zeros(B, NKV, MAXLEN, HD, dtype=torch.bfloat16)
    b_ = torch.zeros(B, NKV, MAXLEN, HD, dtype=torch.bfloat16)
    scatter_reference(a, row, start, k)
    row_write(b_, row, start, k)
    assert torch.equal(a, b_), f"row write != scatter reference at s={s} row={row}"
    others = [r for r in range(B) if r != row]
    assert torch.equal(b_[others], torch.zeros_like(b_[others])), "row write leaked into other rows"


def test_row_gather_reads_back_the_row():
    g = torch.Generator().manual_seed(0)
    bkc = (torch.randn(B, NKV, MAXLEN, HD, generator=g)).to(torch.bfloat16)
    kf = bkc.view(-1, MAXLEN, HD)
    for row in range(B):
        rows_i = row * NKV + torch.arange(NKV)
        alen = 64
        got = kf[rows_i[:, None], torch.arange(alen)[None, :]]
        assert torch.equal(got, bkc[row, :, :alen]), f"gather != row slice at row {row}"


def test_fp8_uint8_view_roundtrip():
    """The fp8-KV storage path: put through the uint8 flat view, read back via the fp8 view."""
    g = torch.Generator().manual_seed(1)
    k = (torch.randn(1, NKV, 9, HD, generator=g)).to(torch.bfloat16)
    enc = k.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    bkc = torch.zeros(B, NKV, MAXLEN, HD, dtype=torch.uint8)
    row_write(bkc, 2, 10, enc)
    kf = bkc.view(-1, MAXLEN, HD)
    rows_i = 2 * NKV + torch.arange(NKV)
    got = kf[rows_i[:, None], (10 + torch.arange(9))[None, :]].view(torch.float8_e4m3fn).to(torch.bfloat16)
    want = enc[0].view(torch.float8_e4m3fn).to(torch.bfloat16)
    assert torch.equal(got, want)


# ---- tree slots (attn_tree_row's KV path): write at [start,start+N) in row b, read [0,start+N) ------
# through the flat view, attend under the tree mask == a plain per-row cache doing attn_tree's
# crop-to-start write + read. Pure tensor math, both storage dtypes — the row-addressing HALF of the
# tree kernel; the frame/protocol half is pinned by the rows fake-ring equivalence gate.

def _tree_mask_and_attend(q, kcur, vcur, parents, start, N):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase0"))
    from tree_spec import build_tree_mask, _gqa_masked_attend
    depths = []
    for i, p in enumerate(parents):
        depths.append(1 if p < 0 else depths[p] + 1)
    mask, _ = build_tree_mask(parents, depths, start, N)
    return _gqa_masked_attend(q, kcur, vcur, mask.to(torch.bfloat16), 2)


@pytest.mark.parametrize("fp8", [False, True])
def test_tree_row_write_read_attend_equals_solo_cache(fp8):
    """Row-addressed tree KV == a dedicated solo cache given identical inputs: same write slots, same
    gather-read, same masked attend; other rows untouched. Overwrite semantics included: a prior
    speculative tail at [start,...) is replaced by the tree nodes exactly as attn_tree crops."""
    g = torch.Generator().manual_seed(7)
    start, N, row, GRP = 23, 7, 1, 2
    NH = NKV * GRP
    parents = [-1, 0, 0, 1, 1, 2, -1]                     # a small 2-root-ish tree (second -1 = anchor child)
    def enc(t):
        return t.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8) if fp8 else t
    def viewf(buf):
        return buf.view(torch.float8_e4m3fn) if fp8 else buf
    dt = torch.uint8 if fp8 else torch.bfloat16
    # committed prefix KV [0,start) already in row `row` of the batched cache AND the solo cache
    pref_k = (torch.randn(NKV, start, HD, generator=g)).to(torch.bfloat16)
    pref_v = (torch.randn(NKV, start, HD, generator=g)).to(torch.bfloat16)
    bkc = torch.zeros(B, NKV, MAXLEN, HD, dtype=dt); bvc = torch.zeros(B, NKV, MAXLEN, HD, dtype=dt)
    bkc[row, :, :start] = enc(pref_k); bvc[row, :, :start] = enc(pref_v)
    # stale speculative tail from a prior chain frame — the tree write must overwrite it
    stale = (torch.randn(NKV, 3, HD, generator=g)).to(torch.bfloat16)
    bkc[row, :, start:start + 3] = enc(stale)
    other = bkc[[r for r in range(B) if r != row]].clone()
    # tree-node K/V + queries
    k = (torch.randn(1, NKV, N, HD, generator=g)).to(torch.bfloat16)
    v = (torch.randn(1, NKV, N, HD, generator=g)).to(torch.bfloat16)
    q = (torch.randn(1, NH, N, HD, generator=g)).to(torch.bfloat16)
    # ROW path: flat-view write + gather read (attn_tree_row's ops, verbatim)
    kf = bkc.view(-1, MAXLEN, HD); vf = bvc.view(-1, MAXLEN, HD)
    rows_i = row * NKV + torch.arange(NKV)
    cp = start + torch.arange(N)
    kf[rows_i[:, None], cp[None, :]] = enc(k[0]); vf[rows_i[:, None], cp[None, :]] = enc(v[0])
    cols = torch.arange(start + N)
    kcur = viewf(kf)[rows_i[:, None], cols[None, :]].to(torch.bfloat16).unsqueeze(0)
    vcur = viewf(vf)[rows_i[:, None], cols[None, :]].to(torch.bfloat16).unsqueeze(0)
    got = _tree_mask_and_attend(q, kcur, vcur, parents, start, N)
    # SOLO reference: a dedicated [1,NKV,total,HD] cache, attn_tree's crop-to-start write
    sk = torch.cat([enc(pref_k).unsqueeze(0), enc(k[0]).unsqueeze(0)], 2)
    sv = torch.cat([enc(pref_v).unsqueeze(0), enc(v[0]).unsqueeze(0)], 2)
    want = _tree_mask_and_attend(q, viewf(sk).to(torch.bfloat16), viewf(sv).to(torch.bfloat16),
                                 parents, start, N)
    assert torch.equal(got, want), f"tree row attend != solo-cache attend (fp8={fp8})"
    assert torch.equal(bkc[[r for r in range(B) if r != row]], other), "tree write leaked into other rows"
    # the stale speculative tail is gone: slots [start,start+3) now hold the tree nodes
    got_slots = viewf(kf)[rows_i[:, None], cp[None, :3]].to(torch.bfloat16)
    assert torch.equal(got_slots, viewf(enc(k[0][:, :3])).to(torch.bfloat16)), "stale tail not overwritten"
