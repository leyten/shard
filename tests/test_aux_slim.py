"""CPU tests for accepted-prefix aux slimming (the transport-bound era's return-leg cut):
_aux_keep_lens must equal the coordinator's accept rule EXACTLY (an undercount starves extend() of a
committed row -> silently degraded g, the worst bug class), _slim_aux_b/_unpack_b must round-trip
bit-exactly vs the unslimmed dequant on every kept row (bf16 + fp8 formats, through the REAL wire
codec), unknown formats must pass through full (fail-open), and old/new build mixes must degrade to
the full payload, never break.

Run: python3 -m pytest tests/test_aux_slim.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
fr = pytest.importorskip("fake_ring")               # bootstraps env (fake M25_DIR) + imports m25_pipe on CPU

MP = fr.MP
from transport import _pack, _unpack                # noqa: E402  (the real wire codec)

H = 64
K = 8


def coordinator_commit_len(ds, r):
    """The batch loop's accept rule, verbatim (m25_pipe coordinate_pipe_batch): committed length."""
    n = 0
    for j in range(K):
        if ds[j] == r[j]:
            n += 1
        else:
            break
    if n == K:
        return K + 1                                # full accept + bonus (depth-1 EAGLE regime)
    return n + 1                                    # accepted prefix + correction


# ---- 1. _aux_keep_lens == the coordinator's accept rule --------------------------------------------

def test_keep_lens_matches_commit_rule():
    cases = []
    anchor = 7
    ds_full = list(range(100, 100 + K))
    cases.append((ds_full, ds_full + [55]))                       # full accept -> K+1
    cases.append((ds_full, [999] + ds_full[1:] + [55]))           # diverge at 0 -> 1
    cases.append((ds_full, ds_full[:3] + [999] * (K - 2)))        # diverge at 3 -> 4
    cases.append(([1] * K, [1] * (K - 1) + [2, 3]))               # diverge at K-1 -> K
    tids = [[anchor] + ds for ds, _ in cases]
    rows = [r for _, r in cases]
    lens = MP._aux_keep_lens(tids, rows)
    want = [coordinator_commit_len(ds, r) for ds, r in cases]
    assert lens == want, f"{lens} != {want}"
    assert all(1 <= l <= K + 1 for l in lens)


def test_keep_lens_covers_done_pad_rows():
    # done streams send [cur]*(K+1); the coordinator ignores their result, any len >= 1 is fine
    tids = [[5] * (K + 1)]
    rows = [[5] * (K + 1)]                                        # pad row "fully accepts" itself
    assert MP._aux_keep_lens(tids, rows) == [K + 1]


# ---- 2. slim -> wire -> unpack == unslimmed dequant, bit-exact on kept rows ------------------------

def roundtrip(aux):
    """Through the REAL codec (string tag + tensors + nested lists must all survive the wire)."""
    return _unpack(_pack({"toks": [[0]], "aux": aux}))


def test_bf16_roundtrip_bitexact():
    g = torch.Generator().manual_seed(0)
    B, s = 4, K + 1
    full = (torch.randn(B, s, H, generator=g) * 0.4).to(torch.bfloat16)
    lens = [3, K + 1, 1, 5]
    slim = MP._slim_aux_b({"30": full}, lens)
    assert slim["30"][0] == "slim" and slim["30"][3] is None
    _, aux = MP._unpack_b(roundtrip(slim))
    o = aux["30"]
    assert o.shape == (B, max(lens), H) and o.dtype == torch.bfloat16
    for b, l in enumerate(lens):
        assert torch.equal(o[b, :l], full[b, :l]), f"stream {b}: kept rows not bit-exact"
        assert torch.equal(o[b, l:], torch.zeros_like(o[b, l:])), "padding must be zeros (never read)"


def test_fp8_roundtrip_matches_unslimmed_dequant():
    g = torch.Generator().manual_seed(1)
    B, s = 3, K + 1
    hcpu = (torch.randn(B, s, H, generator=g) * 0.7).to(torch.bfloat16)
    sc = (hcpu.abs().amax(dim=(1, 2)) / 448.0).clamp(min=1e-8)    # _merge_aux's per-stream fp8 pack
    q = (hcpu / sc.view(-1, 1, 1)).to(torch.float8_e4m3fn)
    entry = [q, [float(x) for x in sc]]
    # unslimmed reference dequant (the existing _unpack_b branch)
    _, ref_aux = MP._unpack_b({"toks": [[0]], "aux": {"58": [q, [float(x) for x in sc]]}})
    ref = ref_aux["58"]
    lens = [2, K + 1, 4]
    slim = MP._slim_aux_b({"58": entry}, lens)
    assert slim["58"][0] == "slim" and slim["58"][3] == [float(x) for x in sc]
    _, aux = MP._unpack_b(roundtrip(slim))
    o = aux["58"]
    for b, l in enumerate(lens):
        assert torch.equal(o[b, :l], ref[b, :l]), f"stream {b}: slim dequant != unslimmed dequant"


def test_unknown_format_passes_through_full():
    solo_shaped = [torch.zeros(3, H, dtype=torch.float8_e4m3fn), 0.5]   # [q, float] solo pair
    weird = "not-a-tensor"
    out = MP._slim_aux_b({"1": solo_shaped, "x": weird}, [1])
    assert out["1"] is solo_shaped and out["x"] is weird


# ---- 3. compat: no tids on the frame -> tail sends FULL aux (old-head mix) -------------------------

def test_unpack_b_still_handles_full_formats():
    g = torch.Generator().manual_seed(2)
    full = (torch.randn(2, K + 1, H, generator=g)).to(torch.bfloat16)
    _, aux = MP._unpack_b({"toks": [[0], [0]], "aux": {"30": full}})
    assert torch.equal(aux["30"], full)                            # bare tensor untouched
