"""MiniMax-M2.5 pipeline-parallel ring driver — coordinator + stage, on top of m25_stage.

Stage: holds a contiguous layer block [lo,hi), serves a relay-back ring over the shard/transport
codec. SAME code runs locally (coherence) and over the libp2p sidecar (scattered WAN). Message =
(start_pos, hidden, n_ret): the tail returns only the last n_ret hidden columns (n_ret=1 for a
plain decode / a long prefill -> avoids shipping the whole prompt's hiddens back; n_ret=K+1 for a
spec-verify chunk). Each Layer crops its KV to start_pos, so rejected speculative tokens roll back.

Coordinator: M2.5 chat-template -> embed -> ring -> norm -> lm_head. Two modes:
  --k 0 : plain greedy decode (1 token / ring traversal).
  --k N : LOSSLESS n-gram spec-decode — propose N drafts, verify all N+1 in one traversal, commit
          the accepted prefix + 1 correction. Output is byte-identical to greedy; N just moves speed
          (g tokens/traversal) on self-similar (copy/retrieval/code) workloads.

  stage:  M25_DIR=/root/m25 python m25_ring.py stage --lo 0 --hi 13 --port 29610 [--next H:P]
  coord:  M25_DIR=/root/m25 python m25_ring.py coord --stage H:P --max-new 64 --k 4 --prompt-file p
"""
import os, sys, time, socket, argparse, hashlib, torch
dev = "cuda"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transport as wire


def stage(lo, hi, port, nxt):
    os.environ.setdefault("M25_DIR", "/root/m25")
    import m25_stage as S
    vcfg = S.vllm_ctx()
    layers = [S.Layer(i) for i in range(lo, hi)]
    gb = torch.cuda.memory_allocated() / 1e9
    print(f"[stage {lo}:{hi}] loaded {hi-lo} layers ({gb:.1f} GB) — warming", flush=True)
    with torch.no_grad():
        _ = S.run_block(layers, 0, torch.randn(1, 4, S.H, dtype=torch.bfloat16, device=dev) * 0.1, vcfg)
        for L in layers:
            L.reset()
    torch.cuda.synchronize()
    print(f"[stage {lo}:{hi}] WARM, listening :{port}" + (f" -> {nxt}" if nxt else " (tail)"), flush=True)
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port)); srv.listen(4)
    fwd = None
    while True:
        conn, _ = srv.accept(); conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[stage {lo}:{hi}] connection", flush=True)
        try:
            while True:
                sp, h, n_ret = wire.recv_msg(conn)
                if sp == 0:
                    for L in layers:
                        L.reset()
                h = S.run_block(layers, sp, h.to(dev), vcfg)
                if nxt:
                    if fwd is None:
                        host, p = nxt.rsplit(":", 1)
                        fwd = socket.create_connection((host, int(p))); fwd.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    wire.send_msg(fwd, (sp, h, n_ret))
                    _, back, _ = wire.recv_msg(fwd)         # tail's [1,n_ret,H] relayed up
                    wire.send_msg(conn, (sp, back, n_ret))
                else:
                    wire.send_msg(conn, (sp, h[:, -n_ret:].contiguous(), n_ret))   # tail trims
        except (OSError, EOFError, ConnectionError) as e:
            print(f"[stage {lo}:{hi}] conn closed ({type(e).__name__}); reset", flush=True)
            fwd = None
            for L in layers:
                L.reset()


def coord(stage_ep, prompt, max_new, temp, seed, k):
    os.environ.setdefault("M25_DIR", "/root/m25")
    DIR = os.environ["M25_DIR"]
    from transformers import AutoTokenizer
    import m25_stage as S
    from ngram_draft import NgramDrafter
    embed_w = S.raw("model.embed_tokens.weight").to(torch.bfloat16).to(dev)
    lm_head_w = S.raw("lm_head.weight").to(torch.bfloat16).to(dev)
    norm_w = S.raw("model.norm.weight").float().to(dev)
    eps = S.EPS
    tok = AutoTokenizer.from_pretrained(DIR, trust_remote_code=True)
    host, p = stage_ep.rsplit(":", 1)
    s = socket.create_connection((host, int(p)), timeout=900); s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); s.settimeout(900)
    print(f"[coord] -> ring head @ {stage_ep}  (mode={'spec k='+str(k) if k else 'plain'})", flush=True)

    eos = tok.eos_token_id if isinstance(tok.eos_token_id, list) else [tok.eos_token_id]
    g = torch.Generator(device=dev); g.manual_seed(seed)

    def ring(token_ids, start_pos, n_ret):
        h = torch.nn.functional.embedding(token_ids, embed_w)   # [1, s, H]
        wire.send_msg(s, (start_pos, h, n_ret))
        _, hb, _ = wire.recv_msg(s)
        return hb.to(dev)                                       # [1, n_ret, H]

    def tok_of(hcol):                                           # hcol: [H]
        x = hcol.float(); x = x * torch.rsqrt(x.pow(2).mean() + eps) * norm_w
        lg = (x.to(torch.bfloat16) @ lm_head_w.t()).float()
        if temp and temp > 0:
            return int(torch.multinomial(torch.softmax(lg / temp, -1), 1, generator=g).item())
        return int(lg.argmax().item())

    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)["input_ids"].to(dev)
    n_prompt = ids.shape[1]
    committed = ids[0].tolist()
    t0 = time.time()
    hb = ring(ids, 0, 1)                                        # prefill: only the last hidden back
    t_prefill = time.time()
    first = tok_of(hb[0, -1])
    committed.append(first); outs = [first]
    traversals = 1

    if k <= 0:                                                  # plain greedy
        for i in range(max_new - 1):
            if outs[-1] in eos:
                outs.pop(); break
            hb = ring(torch.tensor([[committed[-1]]], device=dev), len(committed) - 1, 1)
            nxt = tok_of(hb[0, -1]); committed.append(nxt); outs.append(nxt); traversals += 1
    else:                                                       # lossless n-gram spec-decode
        d = NgramDrafter(ng=3)
        while len(outs) < max_new:
            if outs[-1] in eos:
                outs.pop(); break
            drafts = d.propose(committed, k)
            feed = [committed[-1]] + drafts                     # K+1 tokens
            hb = ring(torch.tensor([feed], device=dev), len(committed) - 1, k + 1)
            preds = [tok_of(hb[0, j]) for j in range(k + 1)]
            a = 0
            for j in range(k):
                if drafts[j] == preds[j]:
                    a += 1
                else:
                    break
            newtoks = drafts[:a] + [preds[a]]                   # accepted + 1 correction
            committed += newtoks; outs += newtoks; traversals += 1
            if any(t in eos for t in newtoks):
                while outs and outs[-1] in eos:
                    outs.pop()
                break

    outs = outs[:max_new]
    dt_dec = time.time() - t_prefill
    gtrav = len(outs) / max(traversals, 1)
    text = tok.decode(outs, skip_special_tokens=True)
    print(f"\n[coord] ctx={n_prompt}tok  prefill={t_prefill-t0:.2f}s  decode={len(outs)}tok in {dt_dec:.1f}s "
          f"= {len(outs)/dt_dec:.2f} tok/s  ({traversals} traversals, g={gtrav:.2f} tok/traversal)", flush=True)
    print(f"[coord] OUTPUT:\n{text[:1400]}", flush=True)
    print("SHA:", hashlib.sha256(text.encode()).hexdigest()[:12], flush=True)
    return {"ctx": n_prompt, "tok_s": len(outs) / dt_dec, "g": gtrav, "text": text}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="role", required=True)
    ps = sub.add_parser("stage"); ps.add_argument("--lo", type=int, required=True); ps.add_argument("--hi", type=int, required=True)
    ps.add_argument("--port", type=int, default=29610); ps.add_argument("--next", default=None)
    pc = sub.add_parser("coord"); pc.add_argument("--stage", required=True)
    pc.add_argument("--prompt", default="Explain what a decentralized GPU inference network is, in 3 sentences.")
    pc.add_argument("--prompt-file", default=None); pc.add_argument("--max-new", type=int, default=64)
    pc.add_argument("--k", type=int, default=0); pc.add_argument("--temp", type=float, default=0.0); pc.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.role == "stage":
        stage(a.lo, a.hi, a.port, a.next)
    else:
        prompt = open(a.prompt_file).read() if a.prompt_file else a.prompt
        coord(a.stage, prompt, a.max_new, a.temp, a.seed, a.k)
