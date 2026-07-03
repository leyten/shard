"""THE usability report: absolute tok/s for NORMAL usage on one warm ring, with full transparency —
exact prompts, context axis (0/8k/30k real-document tokens), and a realistic 8-turn conversation with
history carried. Reasoning ON everywhere, greedy, receipts as launched. One engine config per invocation
(env decides chain vs tree-hybrid: M25_EAGLE=1 [+ M25_TREE=1 M25_TREE_M=12 M25_TREE_TOPB=3]); the report
runner invokes it once per arm and the two JSONs land side by side in docs/receipts/.

Run on the head box as the SOLE coordinator:
  SHARD_TRANSPORT=libp2p HEAD_PORT=29610 TAIL_PORT=29612 CUDA_VISIBLE_DEVICES=0 M25_DIR=/root/m25 \
  M25_EAGLE=1 M25_EAGLE_DIR=/root/m25-eagle /root/venv/bin/python -u m25_usability_report.py
Writes /root/usability_report.json + prints a markdown report to stdout (tee it).
"""
import json, os, socket, time
import m25_stage as S
import m25_pipe as P
from m25_tools import parse_completion, THINK_END
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(S.DIR, trust_remote_code=True)
HEAD = ("127.0.0.1", int(os.environ.get("HEAD_PORT", "29610")))
TAIL = ("127.0.0.1", int(os.environ.get("TAIL_PORT", "29612")))
pipe = socket.create_connection(HEAD, timeout=1800); pipe.setsockopt(*P.NODELAY)
ret = socket.create_connection(TAIL, timeout=1800); ret.setsockopt(*P.NODELAY); ret.settimeout(1800)
P.send_msg(ret, {"op": "hello_return"}); P.recv_msg(ret)

K = 8; DEPTH = 4
ARM = ("tree-hybrid" if os.environ.get("M25_TREE") == "1" else
       "chain-EAGLE" if S.M25_EAGLE else "n-gram")


def _doc_tokens(path, n):
    """First ~n tokens of a real document (genuine varied text, not repetition)."""
    ids = tok(open(path).read(), add_special_tokens=False)["input_ids"][:n]
    return tok.decode(ids)


def run(messages, max_new=256):
    """One job on the warm ring; returns (result, ttft, visible)."""
    st = {"ttft": None, "vis": None}; t0 = time.time()
    def on_commit(out, dt):
        if st["ttft"] is None: st["ttft"] = time.time() - t0
        if st["vis"] is None and THINK_END in tok.decode(out, skip_special_tokens=True):
            st["vis"] = time.time() - t0
    r = P.coordinate_pipe(pipe, tok, messages, K, max_new, 1800, DEPTH, ret_sock=ret,
                          local_draft=P.make_drafter(3), prefill_chunk=2048, max_ctx=131072,
                          reasoning=True, on_commit=on_commit)
    return r, st


def row(name, messages, max_new=256):
    try:
        r, st = run(messages, max_new)
        p = parse_completion(r["text"])
        rtok = len(tok(p["reasoning_content"], add_special_tokens=False)["input_ids"]) if p["reasoning_content"] else 0
        rec = {"cell": name, "prompt_tokens": r["prompt_tokens"], "new_tokens": r["n_tokens"],
               "tok_s": round(r["tok_s"], 2), "g": round(r["toks_per_traversal"], 2),
               "accept": round(r["mean_accept"] / K, 3), "prefill_s": round(r["prefill_s"], 2),
               "ttft_s": round(st["ttft"] or 0, 2), "visible_s": round(st["vis"], 2) if st["vis"] else None,
               "think_tokens": rtok, "answer_tokens": max(0, r["n_tokens"] - rtok),
               "draft_s": r.get("draft_s"), "ring_wait_s": r.get("ring_wait_s"),
               # traversal/transport split (needs stages launched with M25_STAGE_TIMING=1, else None)
               "traversal_s": r.get("traversal_s"), "transport_s": r.get("transport_s"),
               "stage_s": r.get("stage_s"), "per_stage_ms": r.get("per_stage_ms"),
               "answer": (p["content"] or "")[:400]}
        print(f"| {name:<22} | {rec['prompt_tokens']:>6} | {rec['tok_s']:>5.1f} | {rec['g']:>4.1f} | "
              f"{rec['prefill_s']:>6.1f}s | {rec['ttft_s']:>5.1f}s | {rec['new_tokens']:>4} |", flush=True)
        return rec, p
    except Exception as e:
        print(f"| {name:<22} | FAILED: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return {"cell": name, "error": f"{type(e).__name__}: {str(e)[:200]}"}, None


report = {"arm": ARM, "K": K, "depth": DEPTH, "date": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
          "env": {k: v for k, v in os.environ.items() if k.startswith("M25_")},
          "workloads": [], "context_axis": [], "conversation": []}
hdr = "| cell                   | p_tok  | tok/s | g    | prefill | ttft  | ntok |"
print(f"\n=== USABILITY REPORT — arm: {ARM}, K={K} depth={DEPTH}, reasoning ON ===")

# ---- 1. workload cells (prompts verbatim in the JSON) --------------------------------------
DOC = _doc_tokens("/root/m25_pipe.py", 3500) if os.path.exists("/root/m25_pipe.py") else ""
WEATHER = [{"type": "function", "function": {"name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
CELLS = [
    ("reason-math", "A farmer has 17 sheep. All but 9 run away. How many sheep are left? Think it through, then give the number.", None),
    ("reason-logic", "Three light switches outside a windowless room each control one of three bulbs inside. You may flip switches as much as you like, but you can enter the room only once. How do you determine which switch controls which bulb? Reason step by step.", None),
    ("open-chat", "Explain the main tradeoffs between mixture-of-experts and dense transformer models, in a few sentences.", None),
    ("code-edit", "Here is some code:\n\n" + DOC + "\n\nAdd a concise docstring to the coordinate_pipe function describing its key arguments and what it returns.", None),
    ("rag-quote", "Here is some code:\n\n" + DOC + "\n\nWhat dictionary does coordinate_pipe return on success? Quote the exact return statement from the code.", None),
    ("agentic-tool", "What's the current weather in Tokyo? Use the get_weather tool.", WEATHER),
]
print("\n#### Workloads\n" + hdr)
for name, prompt, tools in CELLS:
    msgs = [{"role": "user", "content": prompt}]
    rec, _ = row(name, msgs)
    rec["prompt"] = prompt if len(prompt) < 500 else prompt[:200] + f" ...[{len(prompt)} chars of real code context]"
    if tools: rec["tools"] = "get_weather"
    report["workloads"].append(rec)

# ---- 2. context axis: real-document QA at ~8k and ~30k tokens ------------------------------
print("\n#### Context axis (real documents)\n" + hdr)
for nctx, path in [(8000, "/root/prompt_long.txt"), (30000, "/root/prompt_30k.txt")]:
    if not os.path.exists(path):
        print(f"| ctx-{nctx//1000}k | SKIPPED (no {path})"); continue
    doc = _doc_tokens(path, nctx)
    for kind, q, mn in [("summarize", "Summarize the key points of the document above in 5 concise bullets.", 256),
                        ("quote", "Repeat the final paragraph of the document above exactly, word for word.", 192)]:
        rec, _ = row(f"ctx-{nctx // 1000}k-{kind}", [{"role": "user", "content": doc + "\n\n" + q}], mn)
        rec["prompt"] = f"[~{nctx} tokens of {os.path.basename(path)}] + {q!r}"
        report["context_axis"].append(rec)

# ---- 3. multi-turn conversation (history carried; the 'does chat feel usable' number) ------
TURNS = [
    "What is a mixture-of-experts model, in simple terms?",
    "How does that compare to a dense model in memory and compute cost at inference time?",
    "Write a Python function that estimates tokens/sec for a pipelined ring of N stages, given per-stage latency in ms and a speculative accept length g.",
    "Modify it so g can vary per call via a parameter default; keep the docstring intact.",
    "Quote back only the docstring of that function, exactly as you wrote it.",
    "If a ring has 6 stages at 35 ms each and g is 4.5, what tokens/sec does your formula give? Work it out.",
    "Now assume a better drafter pushes g to 6. New number? Just the arithmetic.",
    "Summarize this whole conversation in 3 bullets.",
]
print("\n#### Multi-turn conversation (history carried)\n" + hdr)
history = []
for i, turn in enumerate(TURNS, 1):
    history.append({"role": "user", "content": turn})
    rec, parsed = row(f"turn-{i}", list(history), max_new=384)
    rec["prompt"] = turn
    report["conversation"].append(rec)
    if parsed is None:                                    # a failed turn breaks history realism -> stop the chat
        break
    history.append({"role": "assistant", "content": parsed["content"] or ""})

# ---- aggregate + dump -----------------------------------------------------------------------
ok = [c for sec in ("workloads", "context_axis", "conversation") for c in report[sec] if "tok_s" in c]
agg_tok = sum(c["new_tokens"] for c in ok); agg_t = sum(c["new_tokens"] / max(c["tok_s"], 1e-9) for c in ok)
report["decode_weighted_tok_s"] = round(agg_tok / max(agg_t, 1e-9), 2)
conv = [c for c in report["conversation"] if "tok_s" in c]
if conv:
    report["conversation_mean_tok_s"] = round(sum(c["new_tokens"] for c in conv) /
                                              max(sum(c["new_tokens"] / max(c["tok_s"], 1e-9) for c in conv), 1e-9), 2)
json.dump(report, open("/root/usability_report.json", "w"), indent=1)
print(f"\n[report] decode-weighted tok/s over ALL cells = {report['decode_weighted_tok_s']}"
      + (f"; conversation mean = {report['conversation_mean_tok_s']}" if conv else ""), flush=True)
print(f"[report] arm={ARM} -> /root/usability_report.json", flush=True)
