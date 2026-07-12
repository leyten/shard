"""Gateway batched dispatcher (phase0/m25_gateway.py) — concurrent requests ride ONE ring job.

CPU-only: the ring is faked (generate / coordinate_pipe_batch monkeypatched); under test is the
dispatch logic itself:
  * a burst inside the window becomes ONE coordinate_pipe_batch call with per-stream args,
  * a lone request takes the solo path (exact old behavior),
  * a client that dies mid-stream is marked dead and NEVER aborts its batch-mates,
  * a ring error reaches every waiting caller.

Run: python3 -m pytest tests/test_gateway_batch.py -q
"""
import os
import sys
import threading
import types

import pytest

os.environ["M25_GATEWAY_MOCK"] = "1"                    # keep import side-effect-free (no engine init)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase0"))

import m25_gateway as gw                                # noqa: E402


@pytest.fixture(autouse=True)
def _dispatcher(monkeypatch):
    """Real dispatcher thread over a clean queue, MOCK off so run_request enqueues."""
    monkeypatch.setattr(gw, "MOCK", False)
    monkeypatch.setattr(gw, "GW_BATCH", 4)
    monkeypatch.setattr(gw, "GW_WINDOW_MS", 60.0)
    monkeypatch.setattr(gw, "A", types.SimpleNamespace(K=8, depth=4, ngram_n=3, max_ctx=8192,
                                                       head="x:1", tail="x:2"))
    gw._QUEUE.clear()
    t = threading.Thread(target=gw._dispatcher, daemon=True)
    t.start()
    yield


def _submit_many(n, results, **kw):
    def one(i):
        try:
            results[i] = gw.run_request([{"role": "user", "content": f"p{i}"}], None, 32 + i,
                                        True, timeout=10, **kw)
        except Exception as e:  # noqa: BLE001
            results[i] = e
    ths = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in ths: t.start()
    for t in ths: t.join(20)


def test_burst_rides_one_batched_job(monkeypatch):
    calls = []

    def fake_batch(pipe, tok, messages_list, K, max_new, timeout, ret, drafters, **kw):
        calls.append({"B": len(messages_list), "max_new": max_new, "tools_b": kw.get("tools_b"),
                      "reasoning": kw.get("reasoning")})
        return {"streams": [{"text": f"s{b}", "n_tokens": 5, "prompt_tokens": 3, "g": 2.0,
                             "output_ids": [1, 2, 3, 4, 5]} for b in range(len(messages_list))],
                "B": len(messages_list), "dt": 0.5, "receipts": [], "rounds": 3}

    monkeypatch.setattr(gw, "coordinate_pipe_batch", fake_batch)
    monkeypatch.setattr(gw, "make_drafters_b", lambda B, n=3: [object()] * B)
    monkeypatch.setattr(gw, "_connect", lambda t: gw.SOCKS.update(
        pipe=types.SimpleNamespace(close=lambda: None), ret=types.SimpleNamespace(close=lambda: None)))
    results = {}
    _submit_many(3, results)
    assert len(calls) == 1 and calls[0]["B"] == 3       # ONE ring job for the whole burst
    assert sorted(calls[0]["max_new"]) == [32, 33, 34]  # per-stream args made it through
    assert all(results[i]["ok"] and results[i]["batched_B"] == 3 for i in range(3))
    assert {results[i]["text"] for i in range(3)} == {"s0", "s1", "s2"}   # right stream to right caller


def test_lone_request_takes_the_solo_path(monkeypatch):
    solo = []
    monkeypatch.setattr(gw, "generate", lambda m, t, mx, oc, reasoning=True: solo.append(1) or
                        {"ok": True, "text": "solo", "n_tokens": 2, "prompt_tokens": 1})
    monkeypatch.setattr(gw, "coordinate_pipe_batch",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("batched path used for B=1")))
    results = {}
    _submit_many(1, results)
    assert solo == [1] and results[0]["text"] == "solo"


def test_dead_client_never_aborts_batch_mates(monkeypatch):
    def fake_batch(pipe, tok, messages_list, K, max_new, timeout, ret, drafters, **kw):
        for cb in kw["on_commits"]:                     # simulate streaming commits to every client
            if cb: cb([1, 2], 0.1)
        return {"streams": [{"text": "t", "n_tokens": 2, "prompt_tokens": 1, "g": 1.0,
                             "output_ids": [1, 2]} for _ in messages_list],
                "B": len(messages_list), "dt": 0.2, "receipts": [], "rounds": 1}

    monkeypatch.setattr(gw, "coordinate_pipe_batch", fake_batch)
    monkeypatch.setattr(gw, "make_drafters_b", lambda B, n=3: [object()] * B)
    monkeypatch.setattr(gw, "_connect", lambda t: gw.SOCKS.update(
        pipe=types.SimpleNamespace(close=lambda: None), ret=types.SimpleNamespace(close=lambda: None)))

    def dying_cb(out, dt):
        raise gw.ClientGone("client left")
    results = {}
    seen = []

    def one(i):
        try:
            oc = dying_cb if i == 0 else (lambda out, dt: seen.append(i))
            results[i] = gw.run_request([{"role": "user", "content": f"p{i}"}], None, 32, True,
                                        on_commit=oc, timeout=10)
        except Exception as e:  # noqa: BLE001
            results[i] = e
    ths = [threading.Thread(target=one, args=(i,)) for i in range(2)]
    for t in ths: t.start()
    for t in ths: t.join(20)
    assert isinstance(results[0], gw.ClientGone)        # the dead client's caller sees ClientGone
    assert results[1]["ok"] and 1 in seen               # its batch-mate finished and streamed


def test_ring_error_reaches_every_caller(monkeypatch):
    monkeypatch.setattr(gw, "coordinate_pipe_batch",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ring down")))
    monkeypatch.setattr(gw, "make_drafters_b", lambda B, n=3: [object()] * B)
    monkeypatch.setattr(gw, "_connect", lambda t: gw.SOCKS.update(
        pipe=types.SimpleNamespace(close=lambda: None), ret=types.SimpleNamespace(close=lambda: None)))
    results = {}
    _submit_many(2, results)
    assert all(isinstance(results[i], RuntimeError) for i in range(2))


# ---- content routing: like-with-like batching + per-content K -------------------------------------

TOOL = [{"type": "function", "function": {"name": "f", "parameters": {}}}]


def _fake_batch_recorder(calls):
    def fake_batch(pipe, tok, messages_list, K, max_new, timeout, ret, drafters, **kw):
        calls.append({"B": len(messages_list), "K": K, "tools_b": kw.get("tools_b")})
        return {"streams": [{"text": f"s{b}", "n_tokens": 5, "prompt_tokens": 3, "g": 2.0,
                             "output_ids": [1, 2, 3, 4, 5]} for b in range(len(messages_list))],
                "B": len(messages_list), "dt": 0.5, "receipts": [], "rounds": 3}
    return fake_batch


def test_content_class_and_k_policy(monkeypatch):
    monkeypatch.setattr(gw, "CONTENT_K", True)
    assert gw._content_class([{"content": "hi"}], TOOL, False) == "draftable"
    assert gw._content_class([{"content": "hi"}], None, True) == "draftable"
    assert gw._content_class([{"content": "x" * 7000}], None, False) == "context"
    assert gw._content_class([{"content": "hi"}], None, False) == "novel"
    assert gw._class_k("draftable") == 8                # args.K (fixture)
    assert gw._class_k("context") == gw.K_CTX
    assert gw._class_k("novel") == gw.K_NOVEL
    monkeypatch.setattr(gw, "CONTENT_K", False)         # kill switch: every class rides args.K
    assert gw._class_k("novel") == 8


def test_dispatcher_batches_like_with_like(monkeypatch):
    """A mixed burst (2 tool-calling + 2 reasoning-off novel) must split into TWO single-class jobs
    with the class's K — never one mixed job — and every caller still gets its own stream."""
    calls = []
    monkeypatch.setattr(gw, "coordinate_pipe_batch", _fake_batch_recorder(calls))
    monkeypatch.setattr(gw, "make_drafters_b", lambda B, n=3: [object()] * B)
    monkeypatch.setattr(gw, "_connect", lambda t: gw.SOCKS.update(
        pipe=types.SimpleNamespace(close=lambda: None), ret=types.SimpleNamespace(close=lambda: None)))
    results = {}

    def one(i):
        tools = TOOL if i < 2 else None
        reasoning = i < 2
        try:
            results[i] = gw.run_request([{"role": "user", "content": f"p{i}"}], tools, 32,
                                        reasoning, timeout=10)
        except Exception as e:  # noqa: BLE001
            results[i] = e
    ths = [threading.Thread(target=one, args=(i,)) for i in range(4)]
    for t in ths: t.start()
    for t in ths: t.join(20)
    assert all(results[i]["ok"] for i in range(4)), results
    assert len(calls) == 2, f"mixed burst must split into two single-class jobs: {calls}"
    by_k = {c["K"]: c for c in calls}
    assert 8 in by_k and gw.K_NOVEL in by_k, f"per-class K missing: {calls}"
    assert all(t is not None for t in by_k[8]["tools_b"]), "draftable job must carry the tool streams"
    assert all(t is None for t in by_k[gw.K_NOVEL]["tools_b"]), "novel job must carry the bare streams"


def test_rows_result_shape_consumed(monkeypatch):
    """The de-lockstep rows result (extra delockstep/tree/agg keys, per-stream g) must assemble into
    per-caller results exactly like the lockstep shape — the gateway path for M25_DELOCKSTEP rings."""
    def fake_rows(pipe, tok, messages_list, K, max_new, timeout, ret, drafters, **kw):
        B = len(messages_list)
        return {"streams": [{"ok": True, "text": f"r{b}", "n_tokens": 4, "prompt_tokens": 2,
                             "g": 3.5, "output_ids": [9, 9, 9, 9]} for b in range(B)],
                "B": B, "rounds": 7, "depth": 1, "wasted": 0, "dt": 0.4, "prefill_s": 0.1,
                "receipts": [], "receipts_ok": None, "eagle": True, "delockstep": True,
                "tree": True, "graph_arm": None, "aux_local": True, "per_stage": {},
                "agg_tok_s": 20.0}
    monkeypatch.setattr(gw, "coordinate_pipe_batch", fake_rows)
    monkeypatch.setattr(gw, "make_drafters_b", lambda B, n=3: [object()] * B)
    monkeypatch.setattr(gw, "_connect", lambda t: gw.SOCKS.update(
        pipe=types.SimpleNamespace(close=lambda: None), ret=types.SimpleNamespace(close=lambda: None)))
    results = {}
    _submit_many(3, results)
    for i in range(3):
        assert results[i]["ok"] and results[i]["batched_B"] == 3
        assert results[i]["mean_accept"] == 3.5 and results[i]["output_ids"] == [9, 9, 9, 9]
    assert {results[i]["text"] for i in range(3)} == {"r0", "r1", "r2"}
