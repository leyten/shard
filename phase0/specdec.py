"""shard phase 2 (prototype): speculative decoding over the 2-node split.

a small draft model runs locally on the head and proposes K tokens; the big
split target verifies all K in ONE pipeline traversal. greedy acceptance, so the
output is token-for-token identical to plain greedy decode (node_kv.py) -- that
identity is the correctness oracle. each round yields (accepted + 1) tokens per
traversal instead of 1, which is what makes wan latency survivable.

per round (cache invariant: target+draft caches hold [0..pos-1], `cur` is the
committed token at position pos, not yet in cache):
  1. draft feeds [cur, d_1..d_K], keeps d_1..d_K  (proposals for pos+1..pos+K)
  2. verify feeds [cur, d_1..d_K] through the target -> argmax r_1..r_{K+1}
  3. accept longest prefix with d_j == r_j; correction = r_{n+1}
  4. crop draft + target-head + target-tail caches to the accepted length

tail: python specdec.py --role tail --split 24 --port 29501 --model Qwen/Qwen2.5-14B-Instruct
head: python specdec.py --role head --split 24 --peer 172.17.0.3 --port 29501 \
        --model Qwen/Qwen2.5-14B-Instruct --draft Qwen/Qwen2.5-0.5B-Instruct --K 4
"""

import argparse, socket, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from node_kv import load_parts, run_layers, send_msg, recv_msg, EDGE_ERRORS, TransportError


def serve_tail(parts, port, timeout, dev):
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port)); srv.listen(1)
    print(f"[tail] listening on :{port} (edge timeout {timeout:.0f}s)", flush=True)
    while True:
        conn, addr = srv.accept(); conn.settimeout(timeout)
        print(f"[tail] head connected from {addr}", flush=True)
        cache = DynamicCache(); verifies = 0
        with torch.no_grad():
            while True:
                try:
                    msg = recv_msg(conn)
                    op = msg["op"]
                    if op == "reset":
                        cache = DynamicCache(); send_msg(conn, "ok")
                    elif op == "crop":
                        cache.crop(msg["len"]); send_msg(conn, "ok")
                    elif op == "verify":
                        h = run_layers(msg["h"].to(dev), parts, cache, msg["start"])
                        h = parts["norm"](h)
                        toks = parts["lm_head"](h).argmax(-1)[0].tolist()
                        send_msg(conn, toks); verifies += 1
                    else:
                        send_msg(conn, "err")
                except EDGE_ERRORS as e:
                    why = "stalled" if isinstance(e, socket.timeout) else "closed"
                    print(f"[tail] edge {why} after {verifies} verifies ({type(e).__name__}); "
                          f"resetting", flush=True)
                    try: conn.close()
                    except OSError: pass
                    break


def _rpc(sock, msg):
    send_msg(sock, msg)
    return recv_msg(sock)


def generate(draft, thead, tok, sock, prompt, K, max_new, dev, timeout,
             adaptive=False, k_min=1, k_max=12):
    sock.settimeout(timeout)
    eos = tok.eos_token_id
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to(dev)
    draft_cache, thead_cache = DynamicCache(), DynamicCache()

    def embed(tokens):                      # tokens: python list -> head activations
        x = torch.tensor([tokens], device=dev)
        return run_layers(thead["embed"](x), thead, thead_cache, pos)

    try:
        _rpc(sock, {"op": "reset"})
        # ---- prefill: target processes the whole prompt, draft fills its cache ----
        pos = 0
        h = run_layers(thead["embed"](ids), thead, thead_cache, 0)
        preds = _rpc(sock, {"op": "verify", "h": h.cpu(), "start": 0})
        cur = preds[-1]                                     # target's token for position L
        pos = ids.shape[1]
        with torch.no_grad():
            draft(input_ids=ids, past_key_values=draft_cache, use_cache=True)

        out = [cur]
        rounds, accepted_total = 0, 0
        kc, ema_n, k_hist = K, float(K), []        # kc = current K (adapts if enabled)
        t0 = time.time()
        with torch.no_grad():
            while len(out) < max_new and cur != eos:
                # 1. draft proposes kc tokens (feed cur, d_1..d_kc -> keep d_1..d_kc)
                drafts, dtok = [], cur
                for i in range(kc + 1):
                    dl = draft(input_ids=torch.tensor([[dtok]], device=dev),
                               past_key_values=draft_cache, use_cache=True).logits
                    dtok = int(dl[0, -1].argmax())
                    if i < kc:
                        drafts.append(dtok)
                # 2. verify [cur, d_1..d_kc] in one traversal
                h = embed([cur] + drafts)
                r = _rpc(sock, {"op": "verify", "h": h.cpu(), "start": pos})   # [r_1..r_{kc+1}]
                # 3. greedy acceptance: longest prefix with d_j == r_j
                n = 0
                for j in range(kc):
                    if drafts[j] == r[j]:
                        n += 1
                    else:
                        break
                committed = drafts[:n] + [r[n]]             # n accepted + 1 correction
                out.extend(committed)
                cur = r[n]
                pos += n + 1
                rounds += 1; accepted_total += n; k_hist.append(kc)
                # 4. roll every cache back to the accepted length
                thead_cache.crop(pos); draft_cache.crop(pos)
                _rpc(sock, {"op": "crop", "len": pos})
                # 5. adaptive K: aim a couple beyond the running acceptance (EMA of n)
                ema_n = 0.7 * ema_n + 0.3 * n
                if adaptive:
                    kc = max(k_min, min(k_max, round(ema_n) + 2))
                if eos in committed:
                    break
    except EDGE_ERRORS as e:
        raise TransportError(f"tail unresponsive ({type(e).__name__}: {e})") from e

    dt = time.time() - t0
    if eos in out:
        out = out[:out.index(eos)]
    return {
        "text": tok.decode(out, skip_special_tokens=True),
        "n_tokens": len(out),
        "rounds": rounds,
        "mean_accept": accepted_total / max(rounds, 1),
        "toks_per_traversal": (accepted_total + rounds) / max(rounds, 1),
        "tok_s": len(out) / max(dt, 1e-9),
        "mean_K": (sum(k_hist) / len(k_hist)) if k_hist else K,
        "k_lo": min(k_hist) if k_hist else K,
        "k_hi": max(k_hist) if k_hist else K,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["head", "tail"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")     # target
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--peer", default="172.17.0.3")
    ap.add_argument("--port", type=int, default=29501)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--adaptive", action="store_true", help="tune K live from the running acceptance rate")
    ap.add_argument("--prompt", default="Explain decentralized computing in two sentences.")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()
    dev = "cuda"

    parts = load_parts(args.model, args.split, args.role)
    if args.role == "tail":
        serve_tail(parts, args.port, args.timeout, dev)
        return

    print(f"[head] loading draft {args.draft} ...", flush=True)
    draft = AutoModelForCausalLM.from_pretrained(args.draft, dtype=torch.bfloat16).to(dev).eval()
    print(f"[head] draft loaded, gpu_mem={torch.cuda.memory_allocated(dev)/1e9:.1f}GB", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    sock = socket.socket(); sock.connect((args.peer, args.port))
    print(f"[head] connected to tail {args.peer}:{args.port}; K={args.K}; generating ...", flush=True)
    try:
        r = generate(draft, parts, tok, sock, args.prompt, args.K, args.max_new, dev, args.timeout,
                     adaptive=args.adaptive)
    except TransportError as e:
        print(f"\n[head] TRANSPORT FAILURE: {e}", flush=True); raise SystemExit(2)
    finally:
        sock.close()
    kdesc = f"adaptive (mean {r['mean_K']:.1f}, {r['k_lo']}-{r['k_hi']})" if args.adaptive else f"{args.K}"
    print(f"\n[head] === OUTPUT ===\n{r['text']}\n", flush=True)
    print(f"[head] {r['n_tokens']} tokens in {r['rounds']} verify traversals | "
          f"mean accepted/round {r['mean_accept']:.2f} | {r['toks_per_traversal']:.2f} tokens/traversal "
          f"(vs 1.0 plain) | {r['tok_s']:.2f} tok/s | draft={args.draft.split('/')[-1]} K={kdesc}", flush=True)


if __name__ == "__main__":
    main()
