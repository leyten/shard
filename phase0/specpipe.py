"""shard: speculative decoding over the N-node pipeline (specdec.py x pipeline.py).

specdec.py proved draft-and-verify over a 2-node split. pipeline.py proved an
N-stage split big enough to hold a 120B target (gpt-oss: partial per-node loading,
sliding-window masks, MXFP4). this puts them together: a small draft proposes K
tokens locally on the head, and the *distributed* target verifies all K in ONE
traversal of the whole chain -- the same single WAN round-trip that plain decode
spends on a single token now commits several. greedy acceptance, so the output is
token-for-token identical to plain pipeline decode; the target is never made whole
on any node.

the draft runs on its own GPU on the head (a 120B stage already fills a 24GB card,
so the draft can't share it); every other node is unchanged from pipeline.py and
holds only its block of the target. the verify op carries a lazy `crop`: the prior
round's rejected tokens are rolled back from every node's cache on the next verify,
piggybacked, so a round costs exactly one round-trip end to end.

  # tail (stage N-1)
  CUDA_VISIBLE_DEVICES=1 python specpipe.py --stage 3 --nstages 4 --model M --listen-port 29503
  # middle (stage i)
  CUDA_VISIBLE_DEVICES=0 python specpipe.py --stage 2 --nstages 4 --model M --listen-port 29502 --next H:29503
  # head (stage 0): stage block on one GPU, draft on another, drives generation
  CUDA_VISIBLE_DEVICES=0,2 python specpipe.py --stage 0 --nstages 4 --model M \
      --next 127.0.0.1:29501 --draft DRAFT --device cuda:0 --draft-device cuda:1 --adaptive
"""

import argparse, socket, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from pipeline import load_stage, run_block
from node_kv import send_msg, recv_msg, EDGE_ERRORS, TransportError


def serve_spec(parts, stage, nstages, listen_port, nxt, timeout, dev):
    """a non-head stage under speculative decoding. identical plumbing to
    pipeline.serve, but the op is `verify`: run this block on the K+1 proposed
    tokens, relay forward, and (at the tail) return the argmax for *every*
    position so the head can find the longest accepted prefix. `crop` rolls this
    node's cache back to the last committed length before running."""
    is_tail = stage == nstages - 1
    nxt_sock = None
    if not is_tail:
        host, port = nxt.split(":")
        nxt_sock = socket.socket(); nxt_sock.settimeout(timeout); nxt_sock.connect((host, int(port)))
        print(f"[s{stage}] connected forward to stage {stage+1} at {nxt}", flush=True)
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", listen_port)); srv.listen(1)
    print(f"[s{stage}] listening on :{listen_port} (edge timeout {timeout:.0f}s)", flush=True)
    while True:
        conn, addr = srv.accept(); conn.settimeout(timeout)
        print(f"[s{stage}] stage {stage-1} connected from {addr}", flush=True)
        cache = DynamicCache(); verifies = 0
        with torch.no_grad():
            while True:
                try:
                    msg = recv_msg(conn)
                    if msg["op"] == "reset":
                        cache = DynamicCache()
                        if nxt_sock: send_msg(nxt_sock, msg); recv_msg(nxt_sock)
                        send_msg(conn, "ok"); continue
                    if msg.get("crop") is not None:        # roll back the prior round's rejects
                        cache.crop(msg["crop"])
                    h = run_block(msg["h"].to(dev), parts, cache, msg["start"])
                    if is_tail:
                        h = parts["norm"](h)
                        toks = parts["lm_head"](h).argmax(-1)[0].tolist()
                    else:
                        send_msg(nxt_sock, {"op": "verify", "h": h.cpu(),
                                            "start": msg["start"], "crop": msg.get("crop")})
                        toks = recv_msg(nxt_sock)
                    send_msg(conn, toks); verifies += 1
                except EDGE_ERRORS as e:
                    why = "stalled" if isinstance(e, socket.timeout) else "closed"
                    print(f"[s{stage}] edge {why} after {verifies} verifies ({type(e).__name__}); resetting", flush=True)
                    try: conn.close()
                    except OSError: pass
                    break


def generate_spec(draft, parts, tok, sock, prompt, K, max_new, dev, draft_dev, timeout,
                  adaptive=False, k_min=1, k_max=12, draft_sock=None):
    """stage 0: draft proposes K tokens on its own GPU; the distributed target
    verifies [cur, d_1..d_K] in one chain traversal; greedy-accept the longest
    matching prefix. caches (draft + this node's block) crop locally; downstream
    nodes crop lazily on the next verify."""
    sock.settimeout(timeout)
    eos = tok.eos_token_id
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to(dev)
    prompt_ids = enc["input_ids"][0].tolist()   # for the in-house draft service (full-prefix queries)
    head_cache, draft_cache = DynamicCache(), DynamicCache()
    pos = 0
    out = []                                    # defined before any edge can fail (prefill incl.)

    def embed(tokens):                          # python list -> this node's block output
        x = torch.tensor([tokens], device=dev)
        return run_block(parts["embed"](x), parts, head_cache, pos)

    try:
        send_msg(sock, {"op": "reset"}); recv_msg(sock)
        # ---- prefill: target processes the whole prompt; draft fills its cache ----
        h = run_block(parts["embed"](ids), parts, head_cache, 0)
        preds = (send_msg(sock, {"op": "verify", "h": h.cpu(), "start": 0}), recv_msg(sock))[1]
        cur = preds[-1]                                     # target's token for position L
        pos = ids.shape[1]
        if draft_sock is None:                              # local transformers draft fills its cache
            with torch.no_grad():
                draft(input_ids=ids.to(draft_dev), past_key_values=draft_cache, use_cache=True)

        out = [cur]
        rounds, accepted_total = 0, 0
        kc, ema_n, k_hist = K, float(K), []
        tail_crop = None                                   # lazy downstream rollback, piggybacked
        t_draft = t_verify = 0.0                            # round-budget instrumentation
        t0 = time.time()
        with torch.no_grad():
            while len(out) < max_new and cur != eos:
                # 1. draft proposes kc tokens
                td = time.time()
                if draft_sock is not None:                  # in-house vLLM draft service (full prefix; prefix-cached)
                    send_msg(draft_sock, {"ids": prompt_ids + out, "k": kc})
                    drafts = recv_msg(draft_sock)
                else:                                       # local transformers draft (incremental cache)
                    drafts, dtok = [], cur
                    for i in range(kc + 1):
                        dl = draft(input_ids=torch.tensor([[dtok]], device=draft_dev),
                                   past_key_values=draft_cache, use_cache=True).logits
                        dtok = int(dl[0, -1].argmax())
                        if i < kc:
                            drafts.append(dtok)
                t_draft += time.time() - td
                # 2. verify [cur, d_1..d_kc] in one traversal (carry the prior round's rollback)
                tv = time.time()
                h = embed([cur] + drafts)
                send_msg(sock, {"op": "verify", "h": h.cpu(), "start": pos, "crop": tail_crop})
                r = recv_msg(sock)
                t_verify += time.time() - tv
                # 3. greedy acceptance: longest prefix with d_j == r_j
                n = 0
                for j in range(kc):
                    if drafts[j] == r[j]:
                        n += 1
                    else:
                        break
                committed = drafts[:n] + [r[n]]            # n accepted + 1 correction
                out.extend(committed)
                cur = r[n]
                pos += n + 1
                rounds += 1; accepted_total += n; k_hist.append(kc)
                # 4. roll caches back to the accepted length: this node's + draft's now,
                #    downstream nodes lazily on the next verify (no extra round-trip)
                head_cache.crop(pos)
                if draft_sock is None:
                    draft_cache.crop(pos)                   # vLLM service manages its own cache
                tail_crop = pos
                # 5. adaptive K: aim a couple beyond the running acceptance (EMA of n)
                ema_n = 0.7 * ema_n + 0.3 * n
                if adaptive:
                    kc = max(k_min, min(k_max, round(ema_n) + 2))
                if eos in committed:
                    break
    except EDGE_ERRORS as e:
        raise TransportError(f"pipeline edge failed at token {len(out)} ({type(e).__name__}: {e})") from e

    dt = time.time() - t0
    if eos in out:
        out = out[:out.index(eos)]
    return {
        "text": tok.decode(out, skip_special_tokens=True),
        "n_tokens": len(out), "rounds": rounds,
        "mean_accept": accepted_total / max(rounds, 1),
        "toks_per_traversal": (accepted_total + rounds) / max(rounds, 1),
        "tok_s": len(out) / max(dt, 1e-9),
        "mean_K": (sum(k_hist) / len(k_hist)) if k_hist else K,
        "k_lo": min(k_hist) if k_hist else K, "k_hi": max(k_hist) if k_hist else K,
        "draft_ms": t_draft / max(rounds, 1) * 1000, "verify_ms": t_verify / max(rounds, 1) * 1000,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--nstages", type=int, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")        # target
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--listen-port", type=int, default=29501)
    ap.add_argument("--next", default="")
    ap.add_argument("--device", default="cuda:0")           # this stage's block
    ap.add_argument("--draft-device", default="cuda:1")     # draft (head only; its own GPU)
    ap.add_argument("--draft-server", default="", help="host:port of the in-house vLLM draft service (else local draft)")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--adaptive", action="store_true", help="tune K live from the running acceptance rate")
    ap.add_argument("--sweep", default="", help="comma K list to measure on one load, 0=adaptive (e.g. 2,3,4,0)")
    ap.add_argument("--prompt", default="Explain decentralized computing in two sentences.")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()
    parts = load_stage(args.model, args.stage, args.nstages, device=args.device)

    if args.stage != 0:
        serve_spec(parts, args.stage, args.nstages, args.listen_port, args.next, args.timeout, args.device)
        return

    draft, draft_sock = None, None
    if args.draft_server:                                   # in-house vLLM draft service
        dh, dp = args.draft_server.split(":")
        draft_sock = socket.socket(); draft_sock.connect((dh, int(dp)))
        print(f"[s0] using in-house draft service at {args.draft_server}", flush=True)
    else:
        print(f"[s0] loading draft {args.draft} on {args.draft_device} ...", flush=True)
        draft = AutoModelForCausalLM.from_pretrained(args.draft, dtype="auto",
                                                     device_map={"": args.draft_device},
                                                     attn_implementation="eager").eval()
        print(f"[s0] draft loaded, draft_mem={torch.cuda.memory_allocated(args.draft_device)/1e9:.1f}GB", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    host, port = args.next.split(":")
    sock = socket.socket(); sock.settimeout(args.timeout); sock.connect((host, int(port)))
    print(f"[s0] connected forward to stage 1 at {args.next}; K={args.K}; generating ...", flush=True)
    if args.sweep:                          # load once, measure several K on one connection
        try:
            for kv in [int(x) for x in args.sweep.split(",")]:
                adaptive = (kv == 0)
                rr = generate_spec(draft, parts, tok, sock, args.prompt, (6 if adaptive else kv),
                                   args.max_new, args.device, args.draft_device, args.timeout,
                                   adaptive=adaptive, draft_sock=draft_sock)
                tag = f"adaptive(mean {rr['mean_K']:.1f})" if adaptive else f"K={kv}"
                print(f"[SWEEP {tag}] {rr['tok_s']:.2f} tok/s | {rr['toks_per_traversal']:.2f} tok/traversal | "
                      f"accept {rr['mean_accept']:.2f} | draft {rr['draft_ms']:.0f}ms + verify {rr['verify_ms']:.0f}ms/round "
                      f"-> async ceiling {rr['toks_per_traversal']/(max(rr['draft_ms'],rr['verify_ms'])/1000):.1f} tok/s", flush=True)
        finally:
            sock.close()
        return
    try:
        r = generate_spec(draft, parts, tok, sock, args.prompt, args.K, args.max_new,
                          args.device, args.draft_device, args.timeout, adaptive=args.adaptive,
                          draft_sock=draft_sock)
    except TransportError as e:
        print(f"\n[s0] TRANSPORT FAILURE: {e}", flush=True); raise SystemExit(2)
    finally:
        sock.close()
    kdesc = f"adaptive (mean {r['mean_K']:.1f}, {r['k_lo']}-{r['k_hi']})" if args.adaptive else f"{args.K}"
    print(f"\n[s0] === OUTPUT ===\n{r['text']}\n", flush=True)
    print(f"[s0] {r['n_tokens']} tokens in {r['rounds']} verify traversals | "
          f"mean accepted/round {r['mean_accept']:.2f} | {r['toks_per_traversal']:.2f} tokens/traversal "
          f"(vs 1.0 plain) | {r['tok_s']:.2f} tok/s | draft={args.draft.split('/')[-1]} K={kdesc} "
          f"({args.nstages}-stage pipeline)", flush=True)


if __name__ == "__main__":
    main()
