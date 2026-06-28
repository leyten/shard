"""Validate fp8 KV (M25_KV_FP8) end-to-end: BATCHED retrieval of a needle at ~20k context — a context that
does NOT fit at B=4 in bf16 (16384 buffer) but DOES in fp8 (32768 buffer at the same memory). If each batched
stream retrieves the needle and stays coherent, fp8 KV is lossless-enough for the engine's retrieval workload.
  SHARD_TRANSPORT=libp2p HEAD_PORT=29610 TAIL_PORT=29612 M25_DIR=/root/m25 python -u m25_fp8_needle.py
"""
import socket, os
import m25_stage as S
import m25_pipe as P
from transformers import AutoTokenizer
from ngram_draft import NgramDrafter

tok = AutoTokenizer.from_pretrained(S.DIR, trust_remote_code=True)
HEAD = ("127.0.0.1", int(os.environ.get("HEAD_PORT", "29610")))
TAIL = ("127.0.0.1", int(os.environ.get("TAIL_PORT", "29612")))
pipe = socket.create_connection(HEAD, timeout=900); pipe.setsockopt(*P.NODELAY)
ret = socket.create_connection(TAIL, timeout=900); ret.setsockopt(*P.NODELAY); ret.settimeout(900)
P.send_msg(ret, {"op": "hello_return"}); P.recv_msg(ret)

NEEDLE = "ZX-PAYLOAD-7731"
FILLER = "The decentralized inference swarm distributes transformer layers across scattered consumer GPUs over a wide-area network, amortizing the round-trip across many requests. "
half = FILLER * 400                                  # ~13k tokens each side -> ~26k total (beyond bf16@16k, within fp8@32k)
PROMPT = (half + f"\n\nIMPORTANT FACT: the secret payload code is {NEEDLE}. Remember it exactly.\n\n" + half +
          "\n\nQuestion: what is the secret payload code mentioned above? Answer with just the code.")
m = [{"role": "user", "content": PROMPT}]

B = 2
drs = [NgramDrafter(ng=3) for _ in range(B)]
print("running B=2 batched needle @ ~20k context (fp8 KV)...", flush=True)
r = P.coordinate_pipe_batch(pipe, tok, [m] * B, 8, 96, 900, ret, drs, depth=4, prefill_chunk=2048, max_ctx=131072)
print(f"prompt_tokens={r['streams'][0]['prompt_tokens']}  agg_tok_s={r['agg_tok_s']:.2f}  rounds={r['rounds']}", flush=True)
allhit = True
for b, s in enumerate(r["streams"]):
    hit = NEEDLE in s["text"]
    allhit &= hit
    print(f"  stream {b}: needle {'FOUND' if hit else 'MISSING'}  | {s['text'][:120]!r}", flush=True)
print(f"[fp8-needle] {'PASS — fp8 KV retrieves the needle at 2x context, batched' if allhit else 'FAIL — needle lost'}", flush=True)
