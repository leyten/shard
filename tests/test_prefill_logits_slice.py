"""P1 — the tail projected [1, s, vocab] logits over EVERY prefill chunk position (a ~1.6GB
transient at s=4096) though the coordinator consumes only the final position's argmax. The tail now
slices to the last position on prefill frames, and the coordinator derives the chunk length from
the AUX shape (never len(toks)) so the EAGLE drafter's prefill context is NOT truncated — the
caveat that made a naive slice a silent drafter regression.

Token-exactness is pinned three ways: the slice itself (position-wise math), the EAGLE
extend-pairing invariant over a slicing ring, and end-to-end losslessness on both ring flavors.

Run: python3 -m pytest tests/test_prefill_logits_slice.py -q
"""
import os
import socket
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
fr = pytest.importorskip("fake_ring")               # bootstraps env + m25_pipe on CPU

from ngram_draft import NgramDrafter                # noqa: E402
from eagle_draft import EagleDrafter, HybridDrafter  # noqa: E402
from test_eagle_draft import _make_head             # noqa: E402  (synthetic EAGLE-3 head, H=32/vocab=100)

MP = fr.MP
S = fr.S
send_msg, recv_msg = fr.send_msg, fr.recv_msg

P = 60


def _hybrid_recorded(seed=0):
    d, embed = _make_head(seed)
    eg = fr.RecordingDrafter(EagleDrafter(d, embed, device="cpu", next_hidden="prenorm"))
    return HybridDrafter(NgramDrafter(ng=3, min_match=1, margin=64), eg), eg


# ---- the slice is token-exact ------------------------------------------------------------------------

def test_tail_logits_slice_token_exact():
    """_tail_logits is position-wise (rmsnorm + matmul per position): the final-position slice must
    yield the same argmax token as the full projection's last position."""
    torch.manual_seed(0)
    H, V, s = 32, 100, 24
    parts = {"norm_w": torch.randn(H).float(),
             "lm_head_w": torch.randn(V, H).to(torch.bfloat16)}
    h = torch.randn(1, s, H, dtype=torch.bfloat16)
    full = MP._tail_logits(h, parts)
    sliced = MP._tail_logits(h[:, -1:], parts)
    assert torch.allclose(full[:, -1:].float(), sliced.float(), rtol=1e-2, atol=1e-2)
    assert full[:, -1].argmax(-1).item() == sliced[:, -1].argmax(-1).item()


# ---- coordinator: chunk length from the aux shape, drafter context NOT truncated ---------------------

@pytest.mark.parametrize("flavor", ["novel", "repetitive"])
def test_extend_pairing_survives_sliced_prefill(flavor, monkeypatch):
    """Against a P1 (slicing) tail, every prefill chunk must still extend the EAGLE context by the
    FULL chunk (tokens[i] = T[base+i+1], contiguous tiling from 0). Before the fix len(toks)=1
    drove the extend -> the drafter saw 1 token per chunk (silently crippled accept)."""
    monkeypatch.setattr(S, "M25_EAGLE", True)
    monkeypatch.setattr(S, "M25_TREE", False, raising=False)
    T = fr.novel_T(420) if flavor == "novel" else fr.repetitive_T(420)
    hyb, rec = _hybrid_recorded()
    res, ring = fr.run_coordinator(T, P, hyb, K=8, depth=4, max_new=100,
                                   prefill_chunk=24, eagle_ring=True, slice_prefill=True)
    assert res["ok"], res
    out = res["output_ids"]
    assert out == T[P:P + len(out)], "LOSSLESSNESS BROKEN against a slicing tail"
    assert rec.extends, "no extend() calls recorded"
    # prefill extends must tile the whole prompt: first at 0, contiguous, full chunk lengths
    assert rec.extends[0][2] == 0
    cur, all_toks = 0, []
    for toks, aux, base in rec.extends:
        assert aux.shape[0] == len(toks)
        assert base == cur, f"extend base_pos {base} != {cur} — the prefill context was truncated"
        for i, t in enumerate(toks):
            assert bool(torch.all(aux[i].flatten() == float(base + i)))
            assert t == T[base + i + 1]
        cur += len(toks)
        all_toks += toks
    assert cur >= P, f"extends cover only {cur} positions — the prompt is {P} long (truncated context)"
    assert all_toks == T[1:1 + len(all_toks)]


def test_lossless_against_old_full_toks_tail(monkeypatch):
    """Version-mix guard: the aux-shape length derivation must be byte-identical against an OLD tail
    that still returns full prefill toks (the default FakeRing)."""
    monkeypatch.setattr(S, "M25_EAGLE", True)
    monkeypatch.setattr(S, "M25_TREE", False, raising=False)
    T = fr.novel_T(420)
    hyb_new, rec_new = _hybrid_recorded()
    res_new, _ = fr.run_coordinator(T, P, hyb_new, K=8, depth=4, max_new=100,
                                    prefill_chunk=24, eagle_ring=True, slice_prefill=True)
    hyb_old, rec_old = _hybrid_recorded()
    res_old, _ = fr.run_coordinator(T, P, hyb_old, K=8, depth=4, max_new=100,
                                    prefill_chunk=24, eagle_ring=True, slice_prefill=False)
    assert res_new["output_ids"] == res_old["output_ids"]   # token-exact across tail builds
    assert [(t, b) for t, _, b in rec_new.extends] == [(t, b) for t, _, b in rec_old.extends]


# ---- real serve() tail: prefill frames reply final-position-only -------------------------------------

class _FakeLayer:
    def reset(self):
        pass


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _dial(port, timeout=3):
    deadline = time.monotonic() + timeout
    while True:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
            s.settimeout(timeout)
            return s
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.02)


def test_real_tail_slices_prefill_only(monkeypatch):
    """The real serve() tail: a prefill frame replies with ONE token — the final position's argmax
    of the full computation — while a decode verify frame keeps every position."""
    monkeypatch.setattr(MP, "dev", "cpu")
    monkeypatch.setattr(MP, "RECEIPTS", False)
    monkeypatch.setattr(MP, "_load",
                        lambda stage, nstages, lo, hi: {"layers": [_FakeLayer()], "head": False, "tail": True})
    monkeypatch.setattr(MP, "_block", lambda grs, layers, start, x, vcfg: x)
    monkeypatch.setattr(MP, "_tail_logits", lambda h, parts: h)     # identity: argmax over the last dim
    monkeypatch.setattr(MP.S, "_CTX", (None, None), raising=False)
    monkeypatch.setattr(MP.S, "M25_EAGLE", False, raising=False)
    monkeypatch.setattr(MP.S, "M25_STAGE_TIMING", False, raising=False)
    port = _free_port()
    threading.Thread(target=MP.serve, args=(1, 2, 0, 1, port, "127.0.0.1:1", 5), daemon=True).start()
    ret = _dial(port)
    send_msg(ret, {"op": "hello_return"})
    assert recv_msg(ret) == "ret_ok"
    pred = _dial(port)
    send_msg(pred, {"op": "reset"})
    assert recv_msg(ret) == "ok"

    h = torch.arange(5 * 4, dtype=torch.bfloat16).reshape(1, 5, 4)
    # prefill frame -> ONE token, the FINAL position's argmax
    send_msg(pred, {"op": "verify", "h": h.clone(), "start": 0, "prefill": True})
    r = recv_msg(ret)
    assert isinstance(r, list) and len(r) == 1
    assert r[0] == h[0, -1].argmax().item()
    # decode frame -> every position (the accept walk consumes them all)
    send_msg(pred, {"op": "verify", "h": h.clone(), "start": 5})
    r = recv_msg(ret)
    assert isinstance(r, list) and len(r) == 5
    ret.close(); pred.close()
