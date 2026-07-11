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
