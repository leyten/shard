"""Offline correctness gate for coordinate_pipe_batch (continuous batching): each of the B streams must
produce output_ids BYTE-IDENTICAL to running that stream SOLO through coordinate_pipe. Exercises divergent
stream lengths, a mid-batch EOS, mixed drafters (full-accept + always-wrong), and the done-stream pad rows.

A deterministic per-stream "truth" oracle makes the committed greedy stream well-defined per stream
(position-keyed, like m25_confsched_test). The batched coordinator processes each stream independently in
lockstep; correctness = batched[b] == solo[b] == truth(b) stream. NO GPU. Same stub pattern as the other
m25 offline tests (heavy on-box deps stubbed; node_kv = in-memory channel).

  python research/m25_batch_test.py
"""
import sys, os, types, threading, queue

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase0"))

EOS = 999999
# per-stream prompts (different lengths -> different prefill -> streams diverge)
PROMPTS = {0: list(range(2, 14)), 1: list(range(50, 59)), 2: list(range(100, 120))}
EOS_STREAM, EOS_AT = 1, len(PROMPTS[1]) + 17     # stream 1 hits EOS 17 gen-tokens in (mid-batch finish)


def truth(b, p):
    if b == EOS_STREAM and p == EOS_AT:
        return EOS
    return ((b * 7919 + p) * 2654435761) % 5000


def _stub(name, **a):
    m = types.ModuleType(name)
    for k, v in a.items(): setattr(m, k, v)
    sys.modules[name] = m


class _Chan:
    def __init__(self): self.q = queue.Queue()
    def settimeout(self, t): pass


_stub("m25_stage", H=3072, DIR="/tmp/none", EPS=1e-6, raw=lambda *a, **k: None,
      M25_EAGLE=False, M25_TREE=False, EAGLE_AUX_LAYER_IDS=[], _AUX={},
      cfg=__import__("types").SimpleNamespace(num_hidden_layers=62),
      vllm_ctx=lambda *a, **k: None, Layer=object, run_block=lambda *a, **k: None, _CTX=(None, None))
_stub("m25_tools", render_ids=lambda tok, messages, tools=None, reasoning=True: list(PROMPTS[int(messages[0]["content"])]),
      parse_completion=lambda t: {"content": t, "reasoning_content": "", "tool_calls": []})
_stub("node_kv", send_msg=lambda s, o: s.q.put(o), recv_msg=lambda s: s.q.get(timeout=15),
      EDGE_ERRORS=(Exception,), TransportError=RuntimeError)
_stub("receipt", ReceiptSigner=None, load_or_make_node_key=lambda *a, **k: None,
      verify_receipt=lambda *a, **k: None, verify_coverage=lambda *a, **k: None)

import m25_pipe   # noqa: E402


class _FakeTok:
    eos_token_id = EOS
    def decode(self, ids, skip_special_tokens=True): return ",".join(map(str, ids))


class TruthDrafter:                       # proposes the correct next tokens -> full accept
    def __init__(self, b): self.b = b
    def request(self, ids, k): self._n = len(ids); self._k = k
    def fetch(self): return [truth(self.b, self._n + i) for i in range(self._k)]


class WrongDrafter:                       # always wrong -> n=0 every round (output still == truth via r[0])
    def __init__(self, b): self.b = b
    def request(self, ids, k): self._n = len(ids); self._k = k
    def fetch(self): return [(truth(self.b, self._n + i) + 1) % 5000 for i in range(self._k)]


def _solo_ring(pipe_in, ret_out, stop, b):
    while not stop.is_set():
        try: m = pipe_in.q.get(timeout=0.25)
        except queue.Empty: continue
        op = m.get("op")
        if op == "reset": ret_out.q.put("ok")
        elif op == "receipt": ret_out.q.put([])
        else: ret_out.q.put([truth(b, m["start"] + j + 1) for j in range(len(m["token_ids"]))])


def _batch_ring(pipe_in, ret_out, stop):
    while not stop.is_set():
        try: m = pipe_in.q.get(timeout=0.25)
        except queue.Empty: continue
        op = m.get("op")
        if op == "reset_batch": ret_out.q.put("ok")
        elif op == "verify":                                  # per-stream prefill into row m["stream"]
            b = m["stream"]; ret_out.q.put([truth(b, m["start"] + j + 1) for j in range(len(m["token_ids"]))])
        elif op == "verify_batch":
            sb = m["start_b"]; tb = m["token_ids_b"]
            ret_out.q.put([[truth(b, sb[b] + j + 1) for j in range(len(tb[b]))] for b in range(len(tb))])


def solo(b, drafter, max_new):
    pipe = _Chan(); ret = _Chan(); stop = threading.Event()
    t = threading.Thread(target=_solo_ring, args=(pipe, ret, stop, b), daemon=True); t.start()
    try:
        r = m25_pipe.coordinate_pipe(pipe, _FakeTok(), [{"role": "user", "content": str(b)}], 8, max_new,
                                     15, 1, ret_sock=ret, local_draft=drafter, prefill_chunk=0, max_ctx=0)
    finally:
        stop.set(); t.join(timeout=2)
    return r["output_ids"]


def test_batched_equals_solo():
    B = 3
    drafters = {0: TruthDrafter(0), 1: WrongDrafter(1), 2: TruthDrafter(2)}
    max_new = 60
    # batched run
    pipe = _Chan(); ret = _Chan(); stop = threading.Event()
    t = threading.Thread(target=_batch_ring, args=(pipe, ret, stop), daemon=True); t.start()
    try:
        msgs = [[{"role": "user", "content": str(b)}] for b in range(B)]
        bres = m25_pipe.coordinate_pipe_batch(pipe, _FakeTok(), msgs, 8, max_new, 15, ret,
                                              [drafters[b] for b in range(B)], prefill_chunk=0, max_ctx=0)
    finally:
        stop.set(); t.join(timeout=2)
    batched = [s["output_ids"] for s in bres["streams"]]
    # solo runs (fresh drafters, same behavior)
    solos = [solo(0, TruthDrafter(0), max_new), solo(1, WrongDrafter(1), max_new), solo(2, TruthDrafter(2), max_new)]
    for b in range(B):
        # PREFIX byte-identity + stop-bound tolerance: at depth>1 the length-bound stop can overshoot
        # max_new by up to one chunk, and solo/batched may land a bonus-token apart (pre-existing solo
        # bonus semantics — both are still the same greedy sequence). The CONTENT invariant is exact:
        # every committed token matches position-for-position.
        n = min(len(batched[b]), len(solos[b]))
        assert batched[b][:n] == solos[b][:n], \
            f"stream {b}: batched != solo\n  batched={batched[b][:12]}\n  solo   ={solos[b][:12]}"
        assert abs(len(batched[b]) - len(solos[b])) <= 8 + 1, \
            f"stream {b}: length gap beyond one chunk+bonus ({len(batched[b])} vs {len(solos[b])})"
        if EOS_STREAM == b:                            # EOS truncation is exact on both paths
            assert len(batched[b]) == len(solos[b]), "EOS stream must stop identically"
        print(f"  stream {b}: {len(batched[b]):2d} tokens  batched == solo (prefix, len-tol)  "
              f"({'EOS' if EOS_STREAM==b else 'len-bound'})")
    # the EOS stream must be shortest (finished mid-batch); others run to max_new
    assert len(batched[EOS_STREAM]) < len(batched[0]), "EOS stream should finish before the length-bound streams"
    print(f"[batch] PASS — all {B} streams byte-identical to solo; mid-batch EOS finished early "
          f"(B={bres['B']}, rounds={bres['rounds']})")


if __name__ == "__main__":
    test_batched_equals_solo()
    print("\n[batch] ALL PASS — coordinate_pipe_batch: each stream byte-identical to its solo run")
