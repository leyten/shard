"""Decode tok/s as a function of CONTEXT, for pipeline (single-stream depth=4) vs batched (B=4 depth=4).
Draftable continuation task + think-skip so accept is high (meaningful tok/s, not the g=1 WAN floor).
One persistent connection. tok_s is DECODE rate (excludes prefill).
  SHARD_TRANSPORT=libp2p HEAD_PORT=29610 TAIL_PORT=29612 M25_DIR=/root/m25 python -u m25_ctx_table.py
"""
import socket, os
import m25_stage as S
import m25_pipe as P
from transformers import AutoTokenizer
from ngram_draft import NgramDrafter

tok = AutoTokenizer.from_pretrained(S.DIR, trust_remote_code=True)
# think-skip: close the forced <think> so the model emits draftable content from token 1
_TS = tok("</think>\n\n", add_special_tokens=False)["input_ids"]
_orig = P.render_ids
P.render_ids = lambda t, m, tools=None, add_generation_prompt=True, **kw: _orig(t, m, tools=tools, add_generation_prompt=add_generation_prompt) + _TS   # **kw: absorb reasoning= (think-skip forces the closed-think path regardless)

HEAD = ("127.0.0.1", int(os.environ.get("HEAD_PORT", "29610")))
TAIL = ("127.0.0.1", int(os.environ.get("TAIL_PORT", "29612")))
pipe = socket.create_connection(HEAD, timeout=1200); pipe.setsockopt(*P.NODELAY)
ret = socket.create_connection(TAIL, timeout=1200); ret.setsockopt(*P.NODELAY); ret.settimeout(1200)
P.send_msg(ret, {"op": "hello_return"}); P.recv_msg(ret)

# ~N tokens of a draftable repeated phrase -> the model continues it -> high n-gram accept at any context length
WORD = "swarm "
def prompt_ctx(n):
    ids = tok(WORD * (n + 8), add_special_tokens=False)["input_ids"][:n]
    return [{"role": "user", "content": "Continue this sequence exactly, same word repeated:\n" + tok.decode(ids)}]

CTXS = [512, 2048, 8192, 12000]   # 5-stage ring: B=4 KV + lm_head + prefill-logits transients on the 13-layer
# tail cap maxlen ~12k and prefill_chunk 1024 (6-stage rings, with 3-5GB more tail headroom, reached 16k @2048)
K = 8; MAXNEW = 128
# both rates are DECODE-only (prefill excluded) — coordinate_pipe and coordinate_pipe_batch now start the
# tok/s timer AFTER prefill, so single-stream vs batched-B=4 is apples-to-apples (the old agg_tok_s counted
# B x serial prefill in the denominator -> the bogus "collapse" at long ctx).
print(f"{'ctx_tok':>8} {'pipe tok/s':>10} {'B=4 agg':>8} {'per-strm':>9} {'B/pipe':>7} {'accept':>7} {'b_prefill':>10} {'coherent':>9}", flush=True)
for n in CTXS:
    m = prompt_ctx(n)
    dr = NgramDrafter(ng=3)
    rs = P.coordinate_pipe(pipe, tok, m, K, MAXNEW, 1800, 4, ret_sock=ret, local_draft=dr, prefill_chunk=1024, max_ctx=131072)
    drs = [NgramDrafter(ng=3) for _ in range(4)]
    rb = P.coordinate_pipe_batch(pipe, tok, [m] * 4, K, MAXNEW, 1800, ret, drs, depth=4, prefill_chunk=1024, max_ctx=131072)
    sp = rs["tok_s"]; ag = rb["agg_tok_s"]
    coh = "swarm" in (rb["streams"][0]["text"].lower())                    # batched output sane (not garbage)?
    print(f"{rs['prompt_tokens']:>8} {sp:>10.2f} {ag:>8.2f} {ag/4:>9.2f} {ag/max(sp,1e-9):>6.2f}x "
          f"{rs['mean_accept']/K*100:>6.0f}% {rb.get('prefill_s',0):>9.2f}s {('yes' if coh else 'NO'):>9}", flush=True)
print("[ctx-table] done", flush=True)
