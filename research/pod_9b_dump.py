"""pod_9b_dump.py — POD-NATIVE GLM-4-9B per-position argmax (apples-to-apples with MTP, same hardware/precision).

Joe's call: don't lean on the beast 9B dump — run the 9B on the SAME pod as the GLM-5.2 target + MTP, so the
head-to-head is unimpeachable for peer review. Same metric as MTP: per-position greedy argmax. The 9B is a
standalone model (its argmax doesn't depend on the target), so this runs ring-DOWN on the free GPUs:
  teacher-forced forward over each corpus seq -> logits.argmax(-1) per position -> dump.
Then offline: h1_bench compare(this dump, the POD target dump) -> 9B acceptance on the same box as MTP.

  /root/vmoe/bin/python pod_9b_dump.py --model /root/glm4_9b --corpora /root/h1/corpora/code_{1024,8192,32768}.jsonl \
      --out /root/dump_9b_pod.jsonl
GLM-4-9B is 32k-capped (max_position_embeddings) -> no 100k. argmax[:-1] convention matches ring target dumps.
"""
import os, sys, json, time, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draft_accept_bench import draft_argmax_per_position, load_prompt_id_seqs
from transformers import AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/glm4_9b")
    ap.add_argument("--corpora", nargs="*", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--vocab", type=int, default=151552)
    ap.add_argument("--out", default="/root/dump_9b_pod.jsonl")
    args = ap.parse_args()
    print(f"[9b] loading {args.model} (bf16, device_map=auto, sdpa) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa").eval()
    dev = next(model.parameters()).device
    print(f"[9b] loaded; input device {dev}", flush=True)
    label_of = lambda p: os.path.splitext(os.path.basename(p))[0]
    with open(os.path.expanduser(args.out), "w") as out:
        for path in args.corpora:
            label = label_of(path)
            seqs = load_prompt_id_seqs(path, args.n or 10**9, None)
            print(f"[9b] {label}: {len(seqs)} seqs", flush=True)
            for idx, ids in enumerate(seqs):
                t0 = time.time()
                with torch.no_grad():
                    am = draft_argmax_per_position(model, ids, dev, args.vocab)
                out.write(json.dumps({"label": label, "idx": idx, "n": len(am) - 1, "argmax": am[:-1]}) + "\n"); out.flush()
                print(f"  [{label}] {idx+1}/{len(seqs)} ({time.time()-t0:.1f}s, len={len(ids)})", flush=True)
    print(f"[9b] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
