"""receipt-hash-reorder — signer.observe(_act_digest(x), _act_digest(h)) runs a .cpu() sync + hash;
on the solo verify hot path it ran BEFORE the frame send, serializing the digest into the round.
Moved AFTER the send so the hash overlaps the WAN hop (frame order is preserved — the stage loops
are single-threaded). Only active with SHARD_RECEIPTS=1.

Two pins: (1) the ORDER — send precedes observe on both the tail's and head/middle's plain verify
branches (fails before the reorder); (2) EQUIVALENCE — the real tail still observes exactly one
(in, out) digest pair per verify frame, with the same pre-wire bf16 digests, so receipts are
byte-identical to the pre-reorder chain.

Run: python3 -m pytest tests/test_receipt_observe_order.py -q
"""
import inspect
import os
import socket
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch = pytest.importorskip("torch")
fr = pytest.importorskip("fake_ring")               # bootstraps env + m25_pipe on CPU

MP = fr.MP
send_msg, recv_msg = fr.send_msg, fr.recv_msg


def test_observe_follows_send_on_hot_paths():
    """The digest+hash must sit AFTER the send on the solo verify branches, where it overlaps the
    WAN hop instead of stretching the round."""
    src = inspect.getsource(MP.serve)
    # tail plain-verify: the LAST bare _ret_send(o) is the plain branch's reply
    after_tail_send = src[src.rindex("_ret_send(o)"):]
    assert "signer.observe" in after_tail_send[:600], (
        "tail: signer.observe does not follow the reply send — the digest sync is back on the "
        "serial path")
    # head/middle plain-verify: the LAST nxt_kw.send(fwd) is the plain branch's forward
    after_fwd_send = src[src.rindex("nxt_kw.send(fwd)"):]
    assert "signer.observe" in after_fwd_send[:600], (
        "head/middle: signer.observe does not follow the ring forward")


class _FakeLayer:
    def reset(self):
        pass


class _RecordingSigner:
    calls = []

    def __init__(self, key, swarm_id, job_id, lo, hi, nonce=None):
        pass

    def observe(self, in_digest, out_digest):
        _RecordingSigner.calls.append((in_digest, out_digest))

    def finalize(self):
        return {}


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


def test_tail_receipts_unchanged_by_reorder(monkeypatch):
    """One observe per verify frame, digests of the pre-wire tensors — the hash-chain input is
    byte-identical to the pre-reorder order."""
    _RecordingSigner.calls = []
    monkeypatch.setattr(MP, "dev", "cpu")
    monkeypatch.setattr(MP, "RECEIPTS", True)
    monkeypatch.setattr(MP, "ReceiptSigner", _RecordingSigner)
    monkeypatch.setattr(MP, "load_or_make_node_key", lambda p: b"k")
    monkeypatch.setattr(MP, "_load",
                        lambda stage, nstages, lo, hi: {"layers": [_FakeLayer()], "head": False, "tail": True})
    monkeypatch.setattr(MP, "_block", lambda grs, layers, start, x, vcfg: x + 1)   # in != out
    monkeypatch.setattr(MP, "_tail_logits", lambda h, parts: h)
    monkeypatch.setattr(MP.S, "_CTX", (None, None), raising=False)
    monkeypatch.setattr(MP.S, "M25_EAGLE", False, raising=False)
    monkeypatch.setattr(MP.S, "M25_STAGE_TIMING", False, raising=False)
    port = _free_port()
    threading.Thread(target=MP.serve, args=(1, 2, 0, 1, port, "127.0.0.1:1", 5), daemon=True).start()
    ret = _dial(port)
    send_msg(ret, {"op": "hello_return"})
    assert recv_msg(ret) == "ret_ok"
    pred = _dial(port)
    send_msg(pred, {"op": "reset", "swarm_id": "s", "job_id": "j"})
    assert recv_msg(ret) == "ok"

    h1 = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    h2 = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    for h in (h1, h2):
        send_msg(pred, {"op": "verify", "h": h.clone(), "start": 0})
        assert isinstance(recv_msg(ret), list)
    send_msg(pred, {"op": "receipt", "receipts": []})
    recv_msg(ret)                                    # receipt sweep flushes the loop past the observes

    assert len(_RecordingSigner.calls) == 2, _RecordingSigner.calls
    for h, (di, do) in zip((h1, h2), _RecordingSigner.calls):
        assert di == MP._act_digest(h)               # input digest: the received activation
        assert do == MP._act_digest(h + 1)           # output digest: the block output, pre-wire
    ret.close(); pred.close()
