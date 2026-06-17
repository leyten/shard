"""shard phase 0: 2-node pipeline-parallel inference over our own tcp transport.

head node: tokenizer + embedding + layers[0:split], runs the generation loop, samples.
tail node: layers[split:] + final norm + lm_head, serves activations -> next token.

no kv-cache yet (recompute the full sequence each step). the point of phase 0 is to
prove a split model serves coherent tokens reliably through a transport we own —
the exact thing nothing off-the-shelf could do. kv-cache + sglang + spec-decode come next.

run tail first:
  python node.py --role tail  --split 18 --port 29501
then head:
  python node.py --role head  --split 18 --peer 172.17.0.3 --port 29501 --prompt "..."
"""

import argparse, socket, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wire
from wire import send_msg, recv_msg   # authenticated + encrypted + pickle-free wire (was raw pickle here)


# ---- load only this node's shard of the model ----
def load_parts(model_id, split, role, device="cuda", dtype=torch.bfloat16):
    print(f"[{role}] loading {model_id} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="eager")
    m = model.model
    n_layers = len(m.layers)
    parts = {"rotary": m.rotary_emb, "n_layers": n_layers}
    if role == "head":
        parts["embed"] = m.embed_tokens
        parts["layers"] = torch.nn.ModuleList([m.layers[i] for i in range(0, split)])
    else:
        parts["layers"] = torch.nn.ModuleList([m.layers[i] for i in range(split, n_layers)])
        parts["norm"] = m.norm
        parts["lm_head"] = model.lm_head
    for key in ("embed", "layers", "norm", "lm_head", "rotary"):
        if isinstance(parts.get(key), torch.nn.Module):
            parts[key] = parts[key].to(device)
    del model, m
    import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"[{role}] loaded: {len(parts['layers'])}/{n_layers} layers, "
          f"gpu_mem={torch.cuda.memory_allocated(device)/1e9:.1f}GB", flush=True)
    return parts


def _causal_bits(h, parts):
    T = h.shape[1]
    pos = torch.arange(T, device=h.device).unsqueeze(0)
    cos, sin = parts["rotary"](h, pos)
    minv = torch.finfo(h.dtype).min
    mask = torch.triu(torch.full((T, T), minv, device=h.device, dtype=h.dtype), diagonal=1)[None, None]
    return pos, (cos, sin), mask

def run_layers(h, parts):
    pos, pe, mask = _causal_bits(h, parts)
    for layer in parts["layers"]:
        h = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=pe, use_cache=False)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["head", "tail"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--peer", default="172.17.0.3")
    ap.add_argument("--port", type=int, default=29501)
    ap.add_argument("--prompt", default="Explain decentralized computing in two sentences.")
    ap.add_argument("--max-new", type=int, default=60)
    args = ap.parse_args()
    wire.key_from_env()                 # shared swarm key (SHARD_PSK); fail fast before the model load
    dev = "cuda"
    parts = load_parts(args.model, args.split, args.role)

    if args.role == "tail":
        srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", args.port)); srv.listen(1)
        print(f"[tail] listening on :{args.port}", flush=True)
        while True:
            conn, addr = srv.accept(); print(f"[tail] head connected from {addr}", flush=True)
            with torch.no_grad():
                while True:
                    try:
                        h = recv_msg(conn).to(dev)
                    except (ConnectionError, EOFError):
                        print("[tail] head gone, waiting for next", flush=True); break
                    h = run_layers(h, parts)
                    h = parts["norm"](h)
                    nxt = int(parts["lm_head"](h[:, -1, :]).argmax(-1).item())
                    send_msg(conn, nxt)

    tok = AutoTokenizer.from_pretrained(args.model)
    enc = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to(dev)
    sock = socket.socket(); sock.connect((args.peer, args.port))
    print(f"[head] connected to tail {args.peer}:{args.port}; generating ...", flush=True)
    out, t0 = [], time.time()
    with torch.no_grad():
        for _ in range(args.max_new):
            h = run_layers(parts["embed"](ids), parts)
            send_msg(sock, h.cpu())
            nxt = recv_msg(sock)
            if nxt == tok.eos_token_id:
                break
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], dim=1)
    dt = time.time() - t0
    print(f"\n[head] === OUTPUT ===\n{tok.decode(out, skip_special_tokens=True)}\n", flush=True)
    print(f"[head] {len(out)} tokens in {dt:.1f}s = {len(out)/dt:.2f} tok/s "
          f"(no-cache, head holds {args.split}/{parts['n_layers']} layers)", flush=True)


if __name__ == "__main__":
    main()
