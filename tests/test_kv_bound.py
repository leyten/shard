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
