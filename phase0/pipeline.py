"""shard: N-node pipeline-parallel inference (generalizes node_kv.py past 2 stages).

a model's layers are split into N contiguous blocks, one per node. activations
flow forward stage 0 -> 1 -> ... -> N-1; the last stage samples and the token
bubbles back along the same chain to stage 0, which drives generation. each node
keeps a per-node KV-cache (layer_idx reindexed 0-based within its block).

stage i (0 = head/driver, N-1 = tail) holds layers [i*L/N : (i+1)*L/N]; the head
also holds the embedding, the tail the final norm + lm_head. every node holds the
rotary module (needed to build position embeddings for its block).

launch tail-first so each node connects to an already-listening successor:
  # tail (stage N-1): listens for stage N-2
  python pipeline.py --stage 3 --nstages 4 --model M --listen-port 29501
  # middle (stage i): connects forward to stage i+1, listens for stage i-1
  python pipeline.py --stage 1 --nstages 4 --model M --listen-port 29501 --next HOST2:29501
  # head (stage 0): connects forward to stage 1, drives generation
  python pipeline.py --stage 0 --nstages 4 --model M --next HOST1:29501 --prompt "..."
"""

import argparse, socket, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import wire
from node_kv import send_msg, recv_msg, EDGE_ERRORS, TransportError


def _causal_mask(q_len, kv_len, start, window, dtype, device):
    """additive mask: query i (abs pos start+i) attends key j (abs pos j) iff
    j <= start+i, and -- when window>0 -- also (start+i - j) < window (sliding)."""
    rows = torch.arange(q_len, device=device) + start
    cols = torch.arange(kv_len, device=device)
    allow = cols[None, :] <= rows[:, None]
    if window:
        allow = allow & ((rows[:, None] - cols[None, :]) < window)
    minv = torch.finfo(dtype).min
    return torch.where(allow, torch.zeros((), dtype=dtype, device=device),
                       torch.full((), minv, dtype=dtype, device=device))[None, None]


def run_block(h, parts, cache, start, par=None, dep=None):
    """run this node's layer block. models with mixed attention (e.g. gpt-oss:
    alternating full / sliding-128 layers) get the right mask per layer; dense
    models with no sliding window fall back to plain causal everywhere. when par/dep
    are given (tree spec) the q tokens form a tree: per-node positions (start+depth)
    and an ancestor mask instead of a plain causal one."""
    q_len = h.shape[1]
    sliding = parts.get("sliding")
    win_sz = parts.get("window", 0)
    if par is not None:                          # tree verify
        from tree import tree_mask
        pos = torch.tensor([[start + dep[i] for i in range(q_len)]], device=h.device)
        full = tree_mask(par, dep, start, 0, h.dtype, h.device)
        win = tree_mask(par, dep, start, win_sz, h.dtype, h.device) if (sliding and win_sz) else full
    else:
        kv_len = start + q_len
        pos = torch.arange(start, kv_len, device=h.device).unsqueeze(0)
        full = _causal_mask(q_len, kv_len, start, 0, h.dtype, h.device)
        win = _causal_mask(q_len, kv_len, start, win_sz, h.dtype, h.device) \
            if (sliding and win_sz) else full
    pe = parts["rotary"](h, pos)
    for i, layer in enumerate(parts["layers"]):
        mask = win if (sliding and sliding[i]) else full
        out = layer(h, attention_mask=mask, position_ids=pos,
                    past_key_values=cache, use_cache=True, position_embeddings=pe)
        h = out[0] if isinstance(out, tuple) else out
    return h


def load_stage(model_id, stage, nstages, device="cuda", dtype="auto"):
    """load ONLY this stage's contiguous block of layers onto the GPU (+ embed on
    the head, norm/lm_head on the tail). every other component is mapped to "meta"
    so it is never loaded -- this is what lets a node hold a slice of a model far
    too big for its card (a 57GB 120B over 4 nodes = ~14GB each). layer_idx is
    reindexed 0-based for the per-node cache. dtype="auto" preserves a checkpoint's
    own precision (e.g. gpt-oss mxfp4) instead of upcasting it."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    n_layers = cfg.num_hidden_layers
    lo = stage * n_layers // nstages
    hi = (stage + 1) * n_layers // nstages
    is_head, is_tail = stage == 0, stage == nstages - 1
    tied = bool(getattr(cfg, "tie_word_embeddings", False))   # then lm_head shares embed's weight
    dmap = {"model.embed_tokens": device if (is_head or (is_tail and tied)) else "meta",
            "model.rotary_emb": device,
            "model.norm": device if is_tail else "meta",
            "lm_head": device if is_tail else "meta"}
    for j in range(n_layers):
        dmap[f"model.layers.{j}"] = device if lo <= j < hi else "meta"
    print(f"[s{stage}] loading layers [{lo}:{hi}] of {model_id} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=dmap,
                                                 attn_implementation="eager")
    m = model.model
    parts = {"rotary": m.rotary_emb, "n_layers": n_layers, "lo": lo, "hi": hi, "_model": model}
    if is_head:
        parts["embed"] = m.embed_tokens
    if is_tail:
        parts["norm"] = m.norm
        parts["lm_head"] = model.lm_head
    kept = [m.layers[i] for i in range(lo, hi)]
    for i, layer in enumerate(kept):
        layer.self_attn.layer_idx = i
    parts["layers"] = torch.nn.ModuleList(kept)
    layer_types = getattr(cfg, "layer_types", None)
    parts["window"] = getattr(cfg, "sliding_window", 0) or 0
    parts["sliding"] = ([layer_types[j] == "sliding_attention" for j in range(lo, hi)]
                        if (layer_types and parts["window"]) else None)
    print(f"[s{stage}] loaded layers [{lo}:{hi}] ({hi-lo}/{n_layers}), "
          f"gpu_mem={torch.cuda.memory_allocated(device)/1e9:.1f}GB", flush=True)
    return parts


def serve(parts, stage, nstages, listen_port, nxt, timeout, dev):
    """a non-head stage: recv activations from the predecessor, run this block,
    forward to the successor (or sample if tail), then bubble the token back."""
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
        cache = DynamicCache(); steps = 0
        with torch.no_grad():
            while True:
                try:
                    msg = recv_msg(conn)
                    if msg["op"] == "reset":
                        cache = DynamicCache()
                        if nxt_sock: send_msg(nxt_sock, msg); recv_msg(nxt_sock)
                        send_msg(conn, "ok"); continue
                    h = run_block(msg["h"].to(dev), parts, cache, msg["start"])
                    if is_tail:
                        h = parts["norm"](h)
                        tok = int(parts["lm_head"](h[:, -1, :]).argmax(-1).item())
                    else:                                  # forward, get the token back
                        send_msg(nxt_sock, {"op": "fwd", "h": h.cpu(), "start": msg["start"]})
                        tok = recv_msg(nxt_sock)
                    send_msg(conn, tok); steps += 1
                except EDGE_ERRORS as e:
                    why = "stalled" if isinstance(e, socket.timeout) else "closed"
                    print(f"[s{stage}] edge {why} after {steps} steps ({type(e).__name__}); resetting", flush=True)
                    try: conn.close()
                    except OSError: pass
                    break


def drive(parts, tok, nxt, prompt, max_new, dev, timeout):
    """stage 0: embed, run the first block, push activations down the pipe, read
    the sampled token back, repeat. one generation per connection."""
    host, port = nxt.split(":")
    sock = socket.socket(); sock.settimeout(timeout); sock.connect((host, int(port)))
    print(f"[s0] connected forward to stage 1 at {nxt}; generating ...", flush=True)
    eos = tok.eos_token_id
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to(dev)
    cache = DynamicCache()
    out, start, t0, t_prefill = [], 0, time.time(), None

    def step(token_ids, start):
        h = run_block(parts["embed"](token_ids), parts, cache, start)
        send_msg(sock, {"op": "fwd", "h": h.cpu(), "start": start})
        return recv_msg(sock)

    try:
        send_msg(sock, {"op": "reset"}); recv_msg(sock)
        with torch.no_grad():
            nxt_tok = step(ids, start)
            t_prefill = time.time()
            start += ids.shape[1]; out.append(nxt_tok)
            for _ in range(max_new - 1):
                if nxt_tok == eos:
                    out.pop(); break
                nxt_tok = step(torch.tensor([[nxt_tok]], device=dev), start)
                start += 1; out.append(nxt_tok)
    except EDGE_ERRORS as e:
        raise TransportError(f"pipeline edge failed at token {len(out)} ({type(e).__name__}: {e})") from e
    finally:
        sock.close()
    if out and out[-1] == eos:
        out.pop()
    dec = time.time() - t_prefill if t_prefill else 0.0
    return {"text": tok.decode(out, skip_special_tokens=True), "n_tokens": len(out),
            "prefill_s": (t_prefill - t0) if t_prefill else 0.0,
            "tok_s": len(out) / max(dec, 1e-9), "total_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--nstages", type=int, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--listen-port", type=int, default=29501)
    ap.add_argument("--next", default="")              # host:port of stage+1 (non-tail nodes)
    ap.add_argument("--prompt", default="Explain decentralized computing in two sentences.")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--runs", type=int, default=1, help="generations on one load; run>0 are warm")
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()
    wire.key_from_env()                 # shared swarm key (SHARD_PSK); fail fast before the model load
    dev = "cuda"
    parts = load_stage(args.model, args.stage, args.nstages, device=dev)

    if args.stage == 0:
        tok = AutoTokenizer.from_pretrained(args.model)
        for run in range(args.runs):                 # run>0 are warm (JIT kernels already compiled)
            try:
                r = drive(parts, tok, args.next, args.prompt, args.max_new, dev, args.timeout)
            except TransportError as e:
                print(f"\n[s0] TRANSPORT FAILURE: {e}", flush=True); raise SystemExit(2)
            print(f"[run {run}{' warm' if run else ' cold'}] {r['n_tokens']} tokens | prefill {r['prefill_s']:.2f}s | "
                  f"decode {r['tok_s']:.2f} tok/s", flush=True)
        print(f"\n[s0] === OUTPUT ===\n{r['text']}\n", flush=True)
    else:
        serve(parts, args.stage, args.nstages, args.listen_port, args.next, args.timeout, dev)


if __name__ == "__main__":
    main()
