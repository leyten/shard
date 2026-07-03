"""Paper bench: INTERLEAVED multi-arm measurement on one warm ring, with repetitions — the
publication-honest version of m25_usability_report.py. Arms run cell-by-cell interleaved with the
arm order rotated per repetition, so time-of-day WAN/co-tenant drift (we measured 1.32x across two
sequential report runs) hits every arm equally instead of poisoning the comparison.

Arms (all coordinator-side flips on the same warm ring; stages stay untouched):
  ar     — NO speculative decoding: a null drafter proposes one never-matching token, so every
           traversal commits exactly the target's next token (true autoregressive over the ring).
           Runs a reduced cell set at reduced max_new (it is slow — that is the point).
  chain  — chain-EAGLE (M25_TREE=0): hybrid n-gram/EAGLE-3 linear drafts, depth-1.
  hybrid — depth-aware hybrid (M25_TREE=1): pipelined n-gram bursts + sync EAGLE tree rounds.

Run on the head box as the SOLE coordinator (same env as the usability report, M25_TREE unset —
the script flips it per job):
  SHARD_TRANSPORT=libp2p HEAD_PORT=29610 TAIL_PORT=29612 CUDA_VISIBLE_DEVICES=0 M25_DIR=/root/m25 \
  M25_EAGLE=1 M25_EAGLE_DIR=/root/m25-eagle M25_FP8_WIRE=1 M25_STAGE_TIMING=1 SHARD_RECEIPTS=1 \
  /root/venv/bin/python -u m25_paper_bench.py --reps 3
Writes /root/paper_bench.json (one record per job: arm, rep, cell, every coordinate_pipe metric
incl. the per-stage transport split and receipts_ok) + prints per-arm/cell medians.
"""
import argparse
import json
import os
import socket
import statistics
import time

import m25_stage as S
import m25_pipe as P
from m25_tools import parse_completion
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(S.DIR, trust_remote_code=True)
HEAD = ("127.0.0.1", int(os.environ.get("HEAD_PORT", "29610")))
TAIL = ("127.0.0.1", int(os.environ.get("TAIL_PORT", "29612")))
pipe = socket.create_connection(HEAD, timeout=1800); pipe.setsockopt(*P.NODELAY)
ret = socket.create_connection(TAIL, timeout=1800); ret.setsockopt(*P.NODELAY); ret.settimeout(1800)
P.send_msg(ret, {"op": "hello_return"}); P.recv_msg(ret)

K, DEPTH = 8, 4


class NullDrafter:
    """True-AR baseline: proposes a single token id that (statistically) never matches the target's
    argmax, so accept is 0 and every traversal commits exactly one corrected token. K=1 keeps the
    wasted wire minimal. Lossless like every arm (the ring's argmax is what gets committed)."""

    def request(self, ids, k):
        pass

    def fetch(self):
        return [199999]

    def cancel(self):
        pass


def _doc_tokens(path, n):
    ids = tok(open(path).read(), add_special_tokens=False)["input_ids"][:n]
    return tok.decode(ids)


DOC = _doc_tokens("/root/m25_pipe.py", 3500) if os.path.exists("/root/m25_pipe.py") else ""
WEATHER = [{"type": "function", "function": {"name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
CELLS = [  # (cell, prompt, tools, max_new, in_ar_arm)
    ("reason-math", "A farmer has 17 sheep. All but 9 run away. How many sheep are left? Think it through, then give the number.", None, 256, True),
    ("reason-logic", "Three light switches outside a windowless room each control one of three bulbs inside. You may flip switches as much as you like, but you can enter the room only once. How do you determine which switch controls which bulb? Reason step by step.", None, 256, False),
    ("open-chat", "Explain the main tradeoffs between mixture-of-experts and dense transformer models, in a few sentences.", None, 256, True),
    ("code-edit", "Here is some code:\n\n" + DOC + "\n\nAdd a concise docstring to the coordinate_pipe function describing its key arguments and what it returns.", None, 256, False),
    ("rag-quote", "Here is some code:\n\n" + DOC + "\n\nWhat dictionary does coordinate_pipe return on success? Quote the exact return statement from the code.", None, 256, True),
    ("agentic-tool", "What's the current weather in Tokyo? Use the get_weather tool.", WEATHER, 192, False),
    # pure-verbatim output: the pipelined n-gram regime's honest headline (accept ~0.97, depth pays)
    ("copy-verbatim", "Repeat the following text exactly, word for word, with no commentary:\n\n" + DOC[:4000], None, 320, True),
]
CTX = []  # (cell, prompt, max_new, reps_cap, in_ar_arm)
for nctx, path in [(8000, "/root/prompt_long.txt"), (30000, "/root/prompt_30k.txt")]:
    if not os.path.exists(path):
        continue
    doc = _doc_tokens(path, nctx)
    cap = None if nctx <= 8000 else 1                      # 30k pairs once per arm (prefill-dominated)
    CTX.append((f"ctx-{nctx // 1000}k-summarize", doc + "\n\nSummarize the key points of the document above in 5 concise bullets.", 256, cap, False))
    CTX.append((f"ctx-{nctx // 1000}k-quote", doc + "\n\nRepeat the final paragraph of the document above exactly, word for word.", 192, cap, nctx <= 8000))

AR_MAXNEW = 96                                             # AR is ~0.4s/token — cap its cells


def run_job(arm, cell, prompt, tools, max_new):
    S.M25_TREE = (arm == "hybrid")                         # coordinators read this dynamically per job
    drafter = NullDrafter() if arm == "ar" else P.make_drafter(3)
    k = 1 if arm == "ar" else K
    msgs = [{"role": "user", "content": prompt}]
    t0 = time.time()
    r = P.coordinate_pipe(pipe, tok, msgs, k, max_new, 1800, DEPTH, ret_sock=ret,
                          local_draft=drafter, prefill_chunk=2048, max_ctx=131072, reasoning=True)
    rec = {"arm": arm, "cell": cell, "wall_s": round(time.time() - t0, 2),
           "prompt_tokens": r["prompt_tokens"], "new_tokens": r["n_tokens"],
           "tok_s": round(r["tok_s"], 3), "g": round(r["toks_per_traversal"], 3),
           "accept": round(r["mean_accept"], 3), "rounds": r["rounds"], "wasted": r["wasted"],
           "prefill_s": round(r["prefill_s"], 2), "decode_s": r["decode_s"],
           "draft_s": r["draft_s"], "ring_wait_s": r["ring_wait_s"],
           "traversal_s": r.get("traversal_s"), "transport_s": r.get("transport_s"),
           "stage_s": r.get("stage_s"), "per_stage_ms": r.get("per_stage_ms"),
           "receipts_ok": r.get("receipts_ok"),
           "answer_head": (parse_completion(r["text"])["content"] or "")[:160]}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--arms", default="ar,chain,hybrid")
    ap.add_argument("--out", default="/root/paper_bench.json")
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    records, t00 = [], time.time()
    for rep in range(1, a.reps + 1):
        order = arms[(rep - 1) % len(arms):] + arms[:(rep - 1) % len(arms)]   # rotate: drift hits all arms
        for arm in order:
            for cell, prompt, tools, mn, in_ar in CELLS:
                if arm == "ar" and not in_ar:
                    continue
                rec = run_job(arm, cell, prompt, tools, AR_MAXNEW if arm == "ar" else mn)
                rec["rep"] = rep; records.append(rec)
                print(f"[{int(time.time() - t00):>5}s] rep{rep} {arm:<6} {cell:<18} "
                      f"{rec['tok_s']:>6.2f} tok/s  g={rec['g']:.2f}  receipts={rec['receipts_ok']}", flush=True)
                json.dump(records, open(a.out, "w"))
            for cell, prompt, mn, cap, in_ar in CTX:
                if (arm == "ar" and not in_ar) or (cap is not None and rep > cap):
                    continue
                rec = run_job(arm, cell, prompt, None, AR_MAXNEW if arm == "ar" else mn)
                rec["rep"] = rep; records.append(rec)
                print(f"[{int(time.time() - t00):>5}s] rep{rep} {arm:<6} {cell:<18} "
                      f"{rec['tok_s']:>6.2f} tok/s  g={rec['g']:.2f}  receipts={rec['receipts_ok']}", flush=True)
                json.dump(records, open(a.out, "w"))

    print("\n=== per-arm/cell median [min..max] tok/s over reps ===", flush=True)
    cells_seen = sorted({r["cell"] for r in records})
    hdr = "cell".ljust(20) + "".join(f"{arm:>22}" for arm in arms)
    print(hdr, flush=True)
    for cell in cells_seen:
        row = cell.ljust(20)
        for arm in arms:
            v = [r["tok_s"] for r in records if r["cell"] == cell and r["arm"] == arm]
            row += f"{statistics.median(v):>8.2f} [{min(v):>5.2f}..{max(v):>5.2f}]" if v else " " * 22
        print(row, flush=True)
    print(f"[paper-bench] {len(records)} jobs -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
