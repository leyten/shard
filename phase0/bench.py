"""shard phase 0+: reliability harness for the kv-cache 2-node pipeline.

loads the head once, then runs N generations against an already-running tail.
each generation opens a fresh connection (=> fresh caches on both nodes), so
this exercises the exact production path in node_kv.generate_one. asserts every
output is non-empty and reports decode tok/s + latency percentiles.

start the tail first (on the peer box), e.g.:
  python node_kv.py --role tail --split 24 --port 29501 --model Qwen/Qwen2.5-14B-Instruct
then run the bench (on the head box):
  python bench.py --split 24 --peer 172.17.0.3 --port 29501 --runs 20 \
      --model Qwen/Qwen2.5-14B-Instruct
"""

import argparse, statistics
import torch
from transformers import AutoTokenizer
from node_kv import load_parts, generate_one

PROMPTS = [
    "Explain decentralized computing in two sentences.",
    "Write a haiku about the ocean.",
    "List three benefits of distributed systems.",
    "What is 17 times 23? Answer with just the number.",
    "Name three programming languages and one strength of each.",
    "Summarize what a transformer neural network does in one sentence.",
    "Give two short tips for writing clear code.",
    "What causes the seasons on Earth?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--peer", default="172.17.0.3")
    ap.add_argument("--port", type=int, default=29501)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()
    dev = "cuda"
    parts = load_parts(args.model, args.split, "head", device=dev)
    tok = AutoTokenizer.from_pretrained(args.model)

    tok_s, total_s, failures = [], [], 0
    print(f"[bench] {args.runs} runs | model={args.model} | split={args.split} | "
          f"head holds {args.split}/{parts['n_layers']} layers", flush=True)
    for i in range(args.runs):
        prompt = PROMPTS[i % len(PROMPTS)]
        try:
            r = generate_one(parts, tok, args.peer, args.port, prompt, args.max_new, dev)
        except Exception as e:
            failures += 1
            print(f"[bench] run {i+1:2d}/{args.runs} FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        ok = r["n_tokens"] > 0 and r["text"].strip() != ""
        if not ok:
            failures += 1
        tok_s.append(r["tok_s"]); total_s.append(r["total_s"])
        snippet = " ".join(r["text"].split())[:60]
        print(f"[bench] run {i+1:2d}/{args.runs} {'OK   ' if ok else 'EMPTY'} "
              f"{r['n_tokens']:3d} tok  {r['tok_s']:6.2f} tok/s  {r['total_s']:4.1f}s | {snippet}", flush=True)

    def pctl(xs, p):
        if not xs:
            return 0.0
        if len(xs) == 1:
            return xs[0]
        return statistics.quantiles(xs, n=100)[p - 1]

    n_ok = args.runs - failures
    print("\n[bench] === SUMMARY ===", flush=True)
    print(f"[bench] {n_ok}/{args.runs} clean completions", flush=True)
    if tok_s:
        print(f"[bench] decode tok/s: median {statistics.median(tok_s):.2f}  "
              f"min {min(tok_s):.2f}  max {max(tok_s):.2f}", flush=True)
        print(f"[bench] total latency: p50 {statistics.median(total_s):.2f}s  "
              f"p95 {pctl(total_s, 95):.2f}s", flush=True)
    print(f"[bench] RESULT: {'PASS' if failures == 0 else 'FAIL'}", flush=True)
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
