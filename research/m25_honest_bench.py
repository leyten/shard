"""HONEST normal-usage benchmark: single-stream, REASONING ON, diverse real prompts. NEVER copy-repetition,
NEVER think-skip. This is the permanent, truthful measure of usability (the copy-retrieval / think-skip numbers
were misleading). Reports, per prompt category: decode tok/s, n-gram accept%, g (committed tokens/traversal),
prefill_s, TTFT (time to first token), first-VISIBLE-content latency (time until past </think>), and the
reasoning-vs-content token split — so the cost of the <think> block is explicit.

Categories span the real workload mix: novel reasoning/math (n-gram-dead), open chat, draftable code-edit &
RAG-quote (n-gram-friendly), and an agentic tool-call. Run on the head box as the SOLE coordinator:
  SHARD_TRANSPORT=libp2p HEAD_PORT=29610 TAIL_PORT=29612 M25_DIR=/root/m25 python -u m25_honest_bench.py
Add M25_EAGLE=1 M25_EAGLE_DIR=/root/m25-eagle to swap the n-gram drafter for the n-gram+EAGLE-3 hybrid
(needs the coordinator process to see a GPU — CUDA_VISIBLE_DEVICES — for the head + M2.5 embed).
"""
import socket, os, time
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

# real code context for the draftable categories (genuine, varied — not repetition)
DOC = ""
for f in ["/root/m25_pipe.py"]:
    try:
        DOC = tok.decode(tok(open(f).read(), add_special_tokens=False)["input_ids"][:3500])
    except OSError:
        pass

WEATHER = [{"type": "function", "function": {"name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

TASKS = [
    ("reason-math",   [{"role": "user", "content": "A farmer has 17 sheep. All but 9 run away. How many sheep are left? Think it through, then give the number."}], None),
    ("reason-logic",  [{"role": "user", "content": "Three light switches outside a windowless room each control one of three bulbs inside. You may flip switches as much as you like, but you can enter the room only once. How do you determine which switch controls which bulb? Reason step by step."}], None),
    ("open-chat",     [{"role": "user", "content": "Explain the main tradeoffs between mixture-of-experts and dense transformer models, in a few sentences."}], None),
    ("code-edit",     [{"role": "user", "content": "Here is some code:\n\n" + DOC + "\n\nAdd a concise docstring to the coordinate_pipe function describing its key arguments and what it returns."}], None),
    ("rag-quote",     [{"role": "user", "content": "Here is some code:\n\n" + DOC + "\n\nWhat dictionary does coordinate_pipe return on success? Quote the exact return statement from the code."}], None),
    ("agentic-tool",  [{"role": "user", "content": "What's the current weather in Tokyo? Use the get_weather tool."}], WEATHER),
]
K = 8; MAXNEW = 256


def run(messages, tools):
    dr = P.make_drafter(3)                                      # n-gram, or HybridDrafter when M25_EAGLE=1
    st = {"ttft": None, "visible": None}
    t0 = time.time()
    def on_commit(out, dt):
        if st["ttft"] is None:
            st["ttft"] = time.time() - t0                          # reset+prefill+first token
        if st["visible"] is None and THINK_END in tok.decode(out, skip_special_tokens=True):
            st["visible"] = time.time() - t0                       # time until first content past </think>
    r = P.coordinate_pipe(pipe, tok, messages, K, MAXNEW, 1800, 4, ret_sock=ret, local_draft=dr,
                          tools=tools, prefill_chunk=2048, max_ctx=131072, reasoning=True, on_commit=on_commit)
    return r, st


DRAFTER = "n-gram+EAGLE hybrid" if S.M25_EAGLE else "n-gram baseline"
print(f"=== HONEST single-stream, REASONING ON, K={K} depth=4 ({DRAFTER}) ===", flush=True)
print(f"{'category':>13} {'p_tok':>6} {'tok/s':>6} {'accept':>7} {'g':>5} {'prefill':>8} {'ttft':>6} {'visible':>8} {'think':>6} {'answer':>7}", flush=True)
agg_tok = 0; agg_t = 0.0
for name, m, tools in TASKS:
    r, st = run(m, tools)
    p = parse_completion(r["text"])
    rtok = len(tok(p["reasoning_content"], add_special_tokens=False)["input_ids"]) if p["reasoning_content"] else 0
    atok = max(0, r["n_tokens"] - rtok)
    vis = f"{st['visible']:.1f}s" if st["visible"] is not None else ">max"
    agg_tok += r["n_tokens"]; agg_t += r["n_tokens"] / max(r["tok_s"], 1e-9)
    print(f"{name:>13} {r['prompt_tokens']:>6} {r['tok_s']:>6.1f} {r['mean_accept']/K*100:>6.0f}% {r['toks_per_traversal']:>5.1f} "
          f"{r['prefill_s']:>7.1f}s {st['ttft'] or 0:>5.1f}s {vis:>8} {rtok:>6} {atok:>7}", flush=True)
print(f"\n[honest] decode-weighted mean tok/s = {agg_tok/max(agg_t,1e-9):.1f}  (reasoning-ON, {DRAFTER})", flush=True)
print("[honest] done", flush=True)
