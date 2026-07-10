"""End-to-end offline gate (adversarial-review adoption, 2026-07-10): coordinate_pipe_batch's NEW eagle_on fill loop (fetch_b) vs the OLD
serial per-stream fetch() loop, on an identical deterministic fake ring with REAL Hybrid/EAGLE drafters.
Covers: n-gram hit stream, EAGLE-miss streams, divergence commits, full-accept + bonus, mid-batch EOS,
done-stream exclusion from act, multi-round extend growth. Committed outputs must be EXACTLY equal.

  python research/m25_batch_eagle_test.py
"""
import json, os, sys, tempfile, threading, queue, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase0"))

import torch
from safetensors.torch import save_file

H, NH, NKV, HD, I2, DVOCAB, VOCAB = 64, 4, 2, 16, 128, 96, 200
EOS = 199
K = 8
PAT = [3, 7, 11, 13, 17, 19]
PROMPTS = {0: list(range(2, 14)), 1: list(range(50, 59)), 2: PAT * 4}
EOS_STREAM, EOS_AT = 1, len(PROMPTS[1]) + 14


def truth(b, p):
    if b == 2:
        return PAT[p % len(PAT)]                      # periodic -> n-gram locks on (full accepts + bonus)
    if b == EOS_STREAM and p == EOS_AT:
        return EOS
    return ((b * 7919 + p) * 2654435761) % 150        # novel -> n-gram miss -> EAGLE (mostly diverges)


def aux_at(b, p, li):
    g = torch.Generator().manual_seed(b * 1000003 + p * 97 + li)
    return (torch.randn(H, generator=g) * 0.3).to(torch.bfloat16)


def _stub(name, **a):
    m = types.ModuleType(name)
    for k, v in a.items():
        setattr(m, k, v)
    sys.modules[name] = m


class _Chan:
    def __init__(self): self.q = queue.Queue()
    def settimeout(self, t): pass


AUX_IDS = [1, 30, 58]
_stub("m25_stage", H=3072, DIR="/tmp/none", EPS=1e-6, raw=lambda *a, **k: None,
      M25_EAGLE=True, M25_TREE=False, EAGLE_AUX_LAYER_IDS=AUX_IDS, _AUX={},
      cfg=types.SimpleNamespace(num_hidden_layers=62),
      vllm_ctx=lambda *a, **k: None, Layer=object, run_block=lambda *a, **k: None, _CTX=(None, None))
_stub("m25_tools", render_ids=lambda tok, messages, tools=None, reasoning=True: list(PROMPTS[int(messages[0]["content"])]),
      parse_completion=lambda t: {"content": t, "reasoning_content": "", "tool_calls": []})
_stub("node_kv", send_msg=lambda s, o: s.q.put(o), recv_msg=lambda s: s.q.get(timeout=15),
      EDGE_ERRORS=(Exception,), TransportError=RuntimeError)
_stub("receipt", ReceiptSigner=None, load_or_make_node_key=lambda *a, **k: None,
      verify_receipt=lambda *a, **k: None, verify_coverage=lambda *a, **k: None)

import m25_pipe                                        # noqa: E402
import eagle_draft                                     # noqa: E402
from eagle_draft import EagleDrafter, HybridDrafter    # noqa: E402
from ngram_draft import NgramDrafter                   # noqa: E402


def make_head(tmp, seed=0):
    json.dump({"hidden_size": H, "num_attention_heads": NH, "num_key_value_heads": NKV,
               "head_dim": HD, "rms_norm_eps": 1e-6, "rope_theta": 5e6,
               "draft_vocab_size": DVOCAB}, open(f"{tmp}/config.json", "w"))
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: (torch.randn(*s, generator=g) * 0.2).to(torch.bfloat16)
    w = {"fc.weight": r(H, 3 * H),
         "midlayer.input_layernorm.weight": (torch.ones(H) + torch.randn(H, generator=g) * 0.05).to(torch.bfloat16),
         "midlayer.hidden_norm.weight": (torch.ones(H) + torch.randn(H, generator=g) * 0.05).to(torch.bfloat16),
         "midlayer.self_attn.q_proj.weight": r(NH * HD, 2 * H),
         "midlayer.self_attn.k_proj.weight": r(NKV * HD, 2 * H),
         "midlayer.self_attn.v_proj.weight": r(NKV * HD, 2 * H),
         "midlayer.self_attn.o_proj.weight": r(H, NH * HD),
         "midlayer.post_attention_layernorm.weight": (torch.ones(H) + torch.randn(H, generator=g) * 0.05).to(torch.bfloat16),
         "midlayer.mlp.gate_proj.weight": r(I2, H),
         "midlayer.mlp.up_proj.weight": r(I2, H),
         "midlayer.mlp.down_proj.weight": r(H, I2),
         "norm.weight": (torch.ones(H) + torch.randn(H, generator=g) * 0.05).to(torch.bfloat16),
         "lm_head.weight": r(DVOCAB, H),
         "d2t": torch.randint(0, VOCAB - DVOCAB, (DVOCAB,), generator=g)}
    save_file(w, f"{tmp}/model.safetensors")


class _FakeTok:
    eos_token_id = EOS
    def decode(self, ids, skip_special_tokens=True): return ",".join(map(str, ids))


def _aux_block(b, start, s):
    """aux dict for a chunk of s positions at absolute start (position-keyed, run-invariant)."""
    return {str(li): torch.stack([aux_at(b, start + j, li) for j in range(s)], 0) for li in AUX_IDS}


def _batch_ring(pipe_in, ret_out, stop, slim=False):
    while not stop.is_set():
        try:
            m = pipe_in.q.get(timeout=0.25)
        except queue.Empty:
            continue
        op = m.get("op")
        if op == "reset_batch":
            ret_out.q.put("ok")
        elif op == "verify":                                  # per-stream prefill: solo-shaped {toks, aux}
            b = m["stream"]; st = m["start"]; s = len(m["token_ids"])
            ret_out.q.put({"toks": [truth(b, st + j + 1) for j in range(s)],
                           "aux": _aux_block(b, st, s)})
        elif op == "verify_batch":
            sb = m["start_b"]; tb = m["token_ids_b"]; B = len(tb)
            toks = [[truth(b, sb[b] + j + 1) for j in range(len(tb[b]))] for b in range(B)]
            aux = {str(li): torch.stack([torch.stack([aux_at(b, sb[b] + j, li)
                                                      for j in range(len(tb[b]))], 0)
                                         for b in range(B)], 0) for li in AUX_IDS}
            if slim:                                           # the tail's accepted-prefix slicing, via the
                lens = m25_pipe._aux_keep_lens(tb, toks)       # REAL helpers (round-trips _unpack_b's padded
                aux = m25_pipe._slim_aux_b(aux, lens)          # reconstruction end-to-end)
            ret_out.q.put({"toks": toks, "aux": aux})
        elif op == "receipt":
            ret_out.q.put([])


def run_batch(head_dir, tag, slim=False):
    gen = torch.Generator().manual_seed(5)
    embed = (torch.randn(VOCAB, H, generator=gen) * 0.3).to(torch.bfloat16)
    base = EagleDrafter(head_dir, embed, device="cpu", max_pos=2048, next_hidden="prenorm")
    drafters = [HybridDrafter(NgramDrafter(ng=3, margin=8), base.fork()) for _ in range(3)]
    pipe = _Chan(); ret = _Chan(); stop = threading.Event()
    t = threading.Thread(target=_batch_ring, args=(pipe, ret, stop, slim), daemon=True); t.start()
    try:
        msgs = [[{"role": "user", "content": str(b)}] for b in range(3)]
        r = m25_pipe.coordinate_pipe_batch(pipe, _FakeTok(), msgs, K, 40, 15, ret, drafters,
                                           prefill_chunk=0, max_ctx=0)
    finally:
        stop.set(); t.join(timeout=2)
    assert r["eagle"], "eagle_on must be True in this gate"
    outs = [s["output_ids"] for s in r["streams"]]
    gs = [s["g"] for s in r["streams"]]
    print(f"  [{tag}] rounds={r['rounds']} lens={[len(o) for o in outs]} g={gs}")
    return outs, gs, r["rounds"]


tmp = tempfile.mkdtemp()
make_head(tmp)

o_new, g_new, r_new = run_batch(tmp, "NEW fetch_b wiring")

_orig_fetch_b = eagle_draft.fetch_b
eagle_draft.fetch_b = lambda ds: [d.fetch() for d in ds]      # the OLD serial per-stream loop, verbatim
o_old, g_old, r_old = run_batch(tmp, "OLD serial-fetch wiring")
eagle_draft.fetch_b = _orig_fetch_b

assert o_new == o_old, f"COMMITTED OUTPUT DIVERGED\n new={o_new}\n old={o_old}"
assert g_new == g_old and r_new == r_old, f"telemetry diverged: g {g_new} vs {g_old}, rounds {r_new} vs {r_old}"
assert len(o_new[EOS_STREAM]) < len(o_new[0]), "EOS stream must finish mid-batch (act-shrink path exercised)"
assert any(g > 3 for g in g_new), "need a high-g (n-gram lock / full-accept+bonus) stream in the mix"
assert any(g < 2 for g in g_new), "need a divergence-heavy stream in the mix"
print("[adv-e2e] PASS — new fill-loop wiring == old serial wiring: committed outputs, g, rounds all equal")

# ---- accepted-prefix aux slimming: a SLIM-serving ring must be a pure payload change ----------------
# The ring slices aux to each stream's committed prefix via the REAL tail helpers (_aux_keep_lens +
# _slim_aux_b) and the coordinator reconstructs via _unpack_b — committed outputs, g and rounds must
# be EXACTLY the full-aux run's (the drafter consumed identical aux rows).
o_slim, g_slim, r_slim = run_batch(tmp, "SLIM-aux ring", slim=True)
assert o_slim == o_new, f"SLIM AUX CHANGED COMMITTED OUTPUT\n slim={o_slim}\n full={o_new}"
assert g_slim == g_new and r_slim == r_new, f"slim telemetry diverged: g {g_slim} vs {g_new}"
print("[adv-e2e] PASS — slim-aux ring == full-aux ring: outputs, g, rounds identical (payload-only change)")


# ---- graph-arm plumbing on reset_batch (the poisoned-arm guard, adversarial-review MAJOR) ----------
# A graph-stamped batched job must (a) CARRY the arm on its reset_batch, (b) fail LOUD when a stage
# refuses/ignores it (old-stage bare "ok"), (c) surface the tail's applied route + counters as
# graph_arm. An unstamped job must keep the bare-ok compat path (graph_arm None).

def _graph_ring(pipe_in, ret_out, stop, ack_graph):
    seen = {}
    while not stop.is_set():
        try:
            m = pipe_in.q.get(timeout=0.25)
        except queue.Empty:
            continue
        op = m.get("op")
        if op == "reset_batch":
            seen["graph_field"] = m.get("graph", "ABSENT")
            ret_out.q.put({"ok": 1, "graph": m["graph"], "graph_captured": 3, "graph_skipped": 1}
                          if ack_graph and "graph" in m else "ok")
        elif op == "verify":
            b = m["stream"]; st = m["start"]; s = len(m["token_ids"])
            ret_out.q.put({"toks": [truth(b, st + j + 1) for j in range(s)], "aux": _aux_block(b, st, s)})
        elif op == "verify_batch":
            sb = m["start_b"]; tb = m["token_ids_b"]
            ret_out.q.put({"toks": [[truth(b, sb[b] + j + 1) for j in range(len(tb[b]))] for b in range(len(tb))],
                           "aux": {str(li): torch.stack([torch.stack([aux_at(b, sb[b] + j, li)
                                                                      for j in range(len(tb[b]))], 0)
                                                         for b in range(len(tb))], 0) for li in AUX_IDS}})
        elif op == "receipt":
            ret_out.q.put({"receipts": [], "graph": True, "graph_captured": 3, "graph_skipped": 1})
    return seen


def _graph_job(graph_job, ack_graph, receipts=False):
    gen = torch.Generator().manual_seed(5)
    embed = (torch.randn(VOCAB, H, generator=gen) * 0.3).to(torch.bfloat16)
    base = EagleDrafter(tmp, embed, device="cpu", max_pos=2048, next_hidden="prenorm")
    drafters = [HybridDrafter(NgramDrafter(ng=3, margin=8), base.fork()) for _ in range(2)]
    pipe = _Chan(); ret = _Chan(); stop = threading.Event()
    t = threading.Thread(target=_graph_ring, args=(pipe, ret, stop, ack_graph), daemon=True); t.start()
    old_gj, old_rc = m25_pipe.M25_GRAPH_JOB, m25_pipe.RECEIPTS
    m25_pipe.M25_GRAPH_JOB = graph_job; m25_pipe.RECEIPTS = receipts
    try:
        msgs = [[{"role": "user", "content": str(b)}] for b in range(2)]
        return m25_pipe.coordinate_pipe_batch(pipe, _FakeTok(), msgs, K, 16, 15, ret, drafters,
                                              prefill_chunk=0, max_ctx=0)
    finally:
        m25_pipe.M25_GRAPH_JOB = old_gj; m25_pipe.RECEIPTS = old_rc
        stop.set(); t.join(timeout=2)


r = _graph_job(graph_job=True, ack_graph=True, receipts=True)          # stamped + honored + counters back
assert r["graph_arm"] == {"graph": True, "graph_captured": 3, "graph_skipped": 1}, r["graph_arm"]
assert r["receipts_ok"] is False                                       # fake ring signs nothing; fails closed
try:
    _graph_job(graph_job=True, ack_graph=False)                        # old/refusing stage acks bare "ok"
    raise AssertionError("graph-stamped reset_batch accepted a bare-ok ack — poisoned-arm guard MISSING")
except RuntimeError as e:
    assert "GRAPH REFUSED" in str(e) or "poisoned" in str(e), e
r = _graph_job(graph_job=None, ack_graph=True)                         # unstamped: bare-ok compat, no arm
assert r["graph_arm"] is None
print("[adv-e2e] PASS — reset_batch graph-arm: stamped+ack-checked+counters; refused toggle raises; compat intact")
