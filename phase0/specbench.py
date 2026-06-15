"""shard phase 2: speculative-decoding sweep harness.

loads the target head once, then sweeps draft x K x workload against a running
specdec tail and prints an acceptance / tokens-per-traversal table. for each
draft it also runs adaptive-K, which should land near the per-workload optimum
without a fixed K. one connection, one generation per (draft, K, workload).

start the specdec tail first (on the peer box), then:
  python specbench.py --split 24 --peer 172.17.0.3 --port 29501 \
      --model Qwen/Qwen2.5-14B-Instruct \
      --drafts Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct --Ks 2,4,6,8
"""

import argparse, socket
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from node_kv import load_parts
from specdec import generate

WORKLOADS = {
    "prose": "Explain decentralized computing in two sentences.",
    "code":  "Write a Python function that returns the nth Fibonacci number.",
    "qa":    "What causes the seasons on Earth?",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--drafts", default="Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--peer", default="172.17.0.3")
    ap.add_argument("--port", type=int, default=29501)
    ap.add_argument("--Ks", default="2,4,6,8")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()
    dev = "cuda"

    thead = load_parts(args.model, args.split, "head", device=dev)
    tok = AutoTokenizer.from_pretrained(args.model)
    Ks = [int(x) for x in args.Ks.split(",")]
    drafts = [d for d in args.drafts.split(",") if d]
    sock = socket.socket(); sock.connect((args.peer, args.port))
    print(f"[specbench] target={args.model.split('/')[-1]} split={args.split} | "
          f"drafts={[d.split('/')[-1] for d in drafts]} | Ks={Ks}", flush=True)
    print(f"\n{'draft':<8} {'K':>4} {'workload':<7} {'acc/rnd':>8} {'tok/trav':>9} {'tok/s':>7}", flush=True)
    print("-" * 50, flush=True)

    rows = []
    for d in drafts:
        dm = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16).to(dev).eval()
        dn = d.split("/")[-1].replace("Qwen2.5-", "").replace("-Instruct", "")
        for K in Ks:
            for wl, prompt in WORKLOADS.items():
                r = generate(dm, thead, tok, sock, prompt, K, args.max_new, dev, args.timeout)
                rows.append((dn, K, wl, r["toks_per_traversal"]))
                print(f"{dn:<8} {K:>4} {wl:<7} {r['mean_accept']:>8.2f} {r['toks_per_traversal']:>9.2f} "
                      f"{r['tok_s']:>7.1f}", flush=True)
        for wl, prompt in WORKLOADS.items():           # adaptive-K, same draft
            r = generate(dm, thead, tok, sock, prompt, 4, args.max_new, dev, args.timeout, adaptive=True)
            print(f"{dn:<8} {'adp':>4} {wl:<7} {r['mean_accept']:>8.2f} {r['toks_per_traversal']:>9.2f} "
                  f"{r['tok_s']:>7.1f}  (mean K {r['mean_K']:.1f})", flush=True)
        del dm; torch.cuda.empty_cache()

    sock.close()
    print("-" * 50, flush=True)
    for wl in WORKLOADS:
        best = max((r for r in rows if r[2] == wl), key=lambda x: x[3])
        print(f"[specbench] best {wl:<5}: {best[0]} K={best[1]} -> {best[3]:.2f} tokens/traversal", flush=True)


if __name__ == "__main__":
    main()
