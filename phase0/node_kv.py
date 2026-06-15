"""shard phase 0+: 2-node split inference WITH kv-cache over our own tcp transport.

prefill processes the whole prompt once (fills each node's cache), then decode
sends only the single new token's activations each step. each node keeps a
DynamicCache for its own layers (layer_idx reindexed 0-based per node). one
generation per connection: a fresh head connection => fresh caches on both nodes.

tail: python node_kv.py --role tail --split 24 --port 29501
head: python node_kv.py --role head --split 24 --peer 172.17.0.3 --port 29501 --prompt "..."
"""

import argparse, socket, struct, pickle, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def _recvall(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return buf

def send_msg(sock, obj):
    data = pickle.dumps(obj)
    sock.sendall(struct.pack("!Q", len(data)) + data)

def recv_msg(sock):
    (n,) = struct.unpack("!Q", _recvall(sock, 8))
    return pickle.loads(_recvall(sock, n))


def load_parts(model_id, split, role, device="cuda", dtype=torch.bfloat16):
    print(f"[{role}] loading {model_id} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="eager")
    m = model.model
    n_layers = len(m.layers)
    parts = {"rotary": m.rotary_emb, "n_layers": n_layers}
    if role == "head":
        parts["embed"] = m.embed_tokens
        kept = [m.layers[i] for i in range(0, split)]
    else:
        kept = [m.layers[i] for i in range(split, n_layers)]
        parts["norm"] = m.norm
        parts["lm_head"] = model.lm_head
    for i, layer in enumerate(kept):          # reindex cache slots 0-based per node
        layer.self_attn.layer_idx = i
    parts["layers"] = torch.nn.ModuleList(kept)
    for key in ("embed", "layers", "norm", "lm_head", "rotary"):
        if isinstance(parts.get(key), torch.nn.Module):
            parts[key] = parts[key].to(device)
    del model, m
    import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"[{role}] loaded: {len(parts['layers'])}/{n_layers} layers, "
          f"gpu_mem={torch.cuda.memory_allocated(device)/1e9:.1f}GB", flush=True)
    return parts


def _mask(q_len, kv_len, start, dtype, device):
    # query i (abs pos start+i) attends key j (abs pos j) iff j <= start+i
    if q_len == 1:
        return None  # single new token attends to all cached keys
    rows = torch.arange(q_len, device=device) + start
    cols = torch.arange(kv_len, device=device)
    allow = cols[None, :] <= rows[:, None]
    minv = torch.finfo(dtype).min
    return torch.where(allow, torch.zeros((), dtype=dtype, device=device),
                       torch.full((), minv, dtype=dtype, device=device))[None, None]

def run_layers(h, parts, cache, start):
    q_len = h.shape[1]
    pos = torch.arange(start, start + q_len, device=h.device).unsqueeze(0)
    pe = parts["rotary"](h, pos)
    mask = _mask(q_len, start + q_len, start, h.dtype, h.device)
    for layer in parts["layers"]:
        h = layer(h, attention_mask=mask, position_ids=pos,
                  past_key_values=cache, use_cache=True, position_embeddings=pe)
    return h


def generate_one(parts, tok, peer, port, prompt, max_new, dev):
    """one full generation over a fresh connection (=> fresh caches on both
    nodes). returns the decoded text plus timing. opens and closes its own
    socket so it can be called in a loop against a persistent tail."""
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to(dev)
    cache = DynamicCache()
    sock = socket.socket(); sock.connect((peer, port))
    out, start, t0, t_prefill = [], 0, time.time(), None
    try:
        with torch.no_grad():
            # prefill the whole prompt
            h = run_layers(parts["embed"](ids), parts, cache, start)
            send_msg(sock, (h.cpu(), start))
            nxt = recv_msg(sock)
            t_prefill = time.time()
            start += ids.shape[1]
            out.append(nxt)
            # decode one token at a time
            for _ in range(max_new - 1):
                if nxt == tok.eos_token_id:
                    out.pop(); break
                h = run_layers(parts["embed"](torch.tensor([[nxt]], device=dev)), parts, cache, start)
                send_msg(sock, (h.cpu(), start))
                nxt = recv_msg(sock)
                start += 1
                out.append(nxt)
    finally:
        sock.close()
    if out and out[-1] == tok.eos_token_id:
        out.pop()
    decode_s = (time.time() - t_prefill) if t_prefill else 0.0
    return {
        "text": tok.decode(out, skip_special_tokens=True),
        "n_tokens": len(out),
        "prefill_s": (t_prefill - t0) if t_prefill else 0.0,
        "decode_s": decode_s,
        "total_s": time.time() - t0,
        "tok_s": len(out) / decode_s if decode_s > 0 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["head", "tail"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--peer", default="172.17.0.3")
    ap.add_argument("--port", type=int, default=29501)
    ap.add_argument("--prompt", default="Explain decentralized computing in two sentences.")
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()
    dev = "cuda"
    parts = load_parts(args.model, args.split, args.role)

    if args.role == "tail":
        srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", args.port)); srv.listen(1)
        print(f"[tail] listening on :{args.port}", flush=True)
        while True:
            conn, addr = srv.accept(); print(f"[tail] head connected from {addr}", flush=True)
            cache = DynamicCache()
            with torch.no_grad():
                while True:
                    try:
                        h, start = recv_msg(conn)
                    except (ConnectionError, EOFError):
                        print("[tail] head gone, waiting for next", flush=True); break
                    h = run_layers(h.to(dev), parts, cache, start)
                    h = parts["norm"](h)
                    nxt = int(parts["lm_head"](h[:, -1, :]).argmax(-1).item())
                    send_msg(conn, nxt)
        return

    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"[head] generating against tail {args.peer}:{args.port} ...", flush=True)
    r = generate_one(parts, tok, args.peer, args.port, args.prompt, args.max_new, dev)
    print(f"\n[head] === OUTPUT ===\n{r['text']}\n", flush=True)
    print(f"[head] {r['n_tokens']} tokens, total {r['total_s']:.1f}s | prefill {r['prefill_s']:.2f}s | "
          f"decode {r['tok_s']:.2f} tok/s (kv-cache, head holds {args.split}/{parts['n_layers']} layers)",
          flush=True)


if __name__ == "__main__":
    main()
