"""Batched-decode KV bound guard (m25_stage._decode_kv_check).

The batched decode writes each stream's K/V at its absolute position via scatter_ along the static
KV buffer's MAXLEN axis. A stream whose position runs past M25_KV_MAXLEN scatters out of bounds — a
device-side CUDA assert that kills the whole stage (and its warm weights). The guard converts that
into a clean, recoverable RuntimeError (mirrors the batched-prefill guard). This unit-tests the
boundary arithmetic on CPU; the live OOB->clean-error behaviour is warm-validated on the ring.

Run: python3 -m pytest tests/test_kv_bound.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytest.importorskip("torch")
fr = pytest.importorskip("fake_ring")                # bootstraps env + imports m25_stage on CPU
S = fr.S


def test_decode_kv_check_boundary(monkeypatch):
    monkeypatch.setattr(S, "M25_KV_MAXLEN", 100)
    # writes indices [starts_max, starts_max+s-1]; in-bounds iff starts_max+s <= MAXLEN
    assert S._decode_kv_check(92, 8) == 100          # max index 99 == MAXLEN-1 -> exactly fits
    assert S._decode_kv_check(0, 100) == 100         # a single stream filling the whole buffer
    with pytest.raises(RuntimeError):
        S._decode_kv_check(93, 8)                     # total 101 -> would write index 100 (OOB)
    with pytest.raises(RuntimeError):
        S._decode_kv_check(100, 1)                    # one past the end


def test_decode_kv_check_message_is_actionable(monkeypatch):
    monkeypatch.setattr(S, "M25_KV_MAXLEN", 40960)
    with pytest.raises(RuntimeError, match="exceeds M25_KV_MAXLEN"):
        S._decode_kv_check(40960, 4)


def test_decode_kv_check_returns_total(monkeypatch):
    monkeypatch.setattr(S, "M25_KV_MAXLEN", 40960)
    assert S._decode_kv_check(1000, 9) == 1009        # the reused context length (feeds _bucket)


def _bare_graph_runner(s):
    """GraphRunner without __init__ (which needs CUDA + static-KV layers): only the attributes
    run()'s pre-capture path touches, so the host-side bound is unit-testable on CPU."""
    gr = object.__new__(S.GraphRunner)
    gr.s = s
    gr.graphs = {}
    gr.eager = set()
    return gr


def test_solo_graph_run_bound(monkeypatch):
    # The SOLO graph path: run() computes alen via _bucket (which CLAMPS to MAXLEN) and then set()
    # fills the static cp buffer with positions >= MAXLEN — the replayed index_copy_ scatters out
    # of bounds (device assert, dead stage). Row/BatchGraphRunner bound this host-side; solo must too.
    monkeypatch.setattr(S, "M25_KV_MAXLEN", 100)
    gr = _bare_graph_runner(8)
    with pytest.raises(RuntimeError, match="exceeds M25_KV_MAXLEN"):
        gr.run(93, None)                              # total 101 -> would write index 100 (OOB)
    with pytest.raises(RuntimeError, match="exceeds M25_KV_MAXLEN"):
        gr.run(100, None)                             # one past the end


def test_solo_graph_run_bound_exact_fit(monkeypatch):
    # total == MAXLEN writes max index MAXLEN-1 -> in bounds; must fall through past the bound
    # (eager-marked bucket routes to run_block, stubbed — no CUDA in the test env).
    monkeypatch.setattr(S, "M25_KV_MAXLEN", 100)
    monkeypatch.setattr(S, "run_block", lambda layers, sp, x, vcfg: "eager-ran")
    gr = _bare_graph_runner(8)
    gr.layers, gr.vcfg = [], None
    gr.eager = {100}                                  # permanently-eager bucket -> run_block fallback
    assert gr.run(92, "x") == "eager-ran"             # total 100 == MAXLEN -> exactly fits, no raise
