"""the payoff: capture the static-cache stage forward as a CUDA graph and replay it in
the incremental setting. all per-round inputs live in STATIC buffers (hidden states,
position ids, cache write positions, masks); the cache writes via a static index buffer
so replay lands KV at the round's position; rotary + the 9 layers + MoE are all inside
the graph. verify the replay output matches the eager static forward, then time it.
"""
import sys, time, torch
sys.path.insert(0, "/root")
from pipeline import load_stage, _causal_mask
from transformers import DynamicCache

MODEL = "/root/models/gpt-oss-120b"; dev = "cuda"; MAXLEN = 128
parts = load_stage(MODEL, 1, 4, device=dev)
cfg = parts["_model"].config; hidden = cfg.hidden_size
layers = parts["layers"]; n_layers = len(layers)
sliding = parts.get("sliding"); win_sz = parts.get("window", 0); rotary = parts["rotary"]

torch.manual_seed(0); P = 12; Kp1 = 5
prefill = torch.randn(1, P, hidden, dtype=torch.bfloat16, device=dev) * 0.1
toks = torch.randn(1, Kp1, hidden, dtype=torch.bfloat16, device=dev) * 0.1

# static input buffers (filled per round, then replay)
h_buf = torch.zeros(1, Kp1, hidden, dtype=torch.bfloat16, device=dev)
pos_buf = torch.zeros(1, Kp1, dtype=torch.long, device=dev)
cp_buf = torch.zeros(Kp1, dtype=torch.long, device=dev)
mf_buf = torch.zeros(1, 1, Kp1, MAXLEN, dtype=torch.bfloat16, device=dev)
mw_buf = torch.zeros(1, 1, Kp1, MAXLEN, dtype=torch.bfloat16, device=dev)


class StaticKV:
    def __init__(self):
        z = lambda: [torch.zeros(1, 8, MAXLEN, 64, dtype=torch.bfloat16, device=dev) for _ in range(n_layers)]
        self.k, self.v = z(), z(); self.cp = None
    def update(self, key, value, layer_idx, *a, **kw):
        self.k[layer_idx].index_copy_(2, self.cp, key); self.v[layer_idx].index_copy_(2, self.cp, value)
        return self.k[layer_idx], self.v[layer_idx]


def _run(layer, x, m, pos, cache, pe):                    # the layer returns a BARE tensor
    o = layer(x, attention_mask=m, position_ids=pos, past_key_values=cache, use_cache=True, position_embeddings=pe)
    return o[0] if isinstance(o, tuple) else o


def prefill_cache(cache, h, start):                       # eager, one-time
    cache.cp = torch.arange(start, start + h.shape[1], device=dev)
    cp = cache.cp; pos = cp.unsqueeze(0)
    mf = _causal_mask(h.shape[1], MAXLEN, start, 0, torch.bfloat16, dev)
    mw = _causal_mask(h.shape[1], MAXLEN, start, win_sz, torch.bfloat16, dev) if win_sz else mf
    pe = rotary(h, pos); x = h
    for i, layer in enumerate(layers):
        m = mw if (sliding and sliding[i]) else mf
        x = _run(layer, x, m, pos, cache, pe)
    return x

def graph_fwd(cache):                                     # reads ONLY static buffers
    pe = rotary(h_buf, pos_buf); x = h_buf
    for i, layer in enumerate(layers):
        m = mw_buf if (sliding and sliding[i]) else mf_buf
        x = _run(layer, x, m, pos_buf, cache, pe)
    return x

def set_round(cache, h, wp):
    h_buf.copy_(h); pos_buf.copy_(torch.arange(wp, wp + Kp1, device=dev))
    cp_buf.copy_(torch.arange(wp, wp + Kp1, device=dev)); cache.cp = cp_buf
    mf_buf.copy_(_causal_mask(Kp1, MAXLEN, wp, 0, torch.bfloat16, dev))
    mw_buf.copy_(_causal_mask(Kp1, MAXLEN, wp, win_sz, torch.bfloat16, dev) if win_sz else mf_buf)

with torch.no_grad():
    # eager reference (fresh static cache)
    rc = StaticKV(); prefill_cache(rc, prefill, 0); rc.cp = torch.arange(P, P + Kp1, device=dev)
    cpr = rc.cp; posr = cpr.unsqueeze(0)
    mfr = _causal_mask(Kp1, MAXLEN, P, 0, torch.bfloat16, dev)
    mwr = _causal_mask(Kp1, MAXLEN, P, win_sz, torch.bfloat16, dev) if win_sz else mfr
    per = rotary(toks, posr); xr = toks
    for i, layer in enumerate(layers):
        m = mwr if (sliding and sliding[i]) else mfr
        xr = _run(layer, xr, m, posr, rc, per)
    o_ref = xr.float().clone()

    # graph cache: prefill then capture the round forward
    gc = StaticKV(); prefill_cache(gc, prefill, 0)
    set_round(gc, toks, P)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): graph_fwd(gc)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o_graph = graph_fwd(gc)
    set_round(gc, toks, P); g.replay(); torch.cuda.synchronize()
    diff = (o_graph.float() - o_ref).abs().max().item()
    print(f"GRAPH replay vs eager static forward: max-diff = {diff:.5f} -> "
          f"{'MATCH' if diff < 0.02 else 'MISMATCH'}", flush=True)

    # timing: eager static forward vs graph replay
    def eager_round():
        c = StaticKV(); prefill_cache(c, prefill, 0); c.cp = torch.arange(P, P + Kp1, device=dev)
        cp = c.cp; pos = cp.unsqueeze(0); pe = rotary(toks, pos); x = toks
        for i, layer in enumerate(layers):
            m = (mwr if (sliding and sliding[i]) else mfr)
            x = _run(layer, x, m, pos, c, pe)
        return x
    # time just the round forward (re-prefill excluded): reuse gc, vary nothing
    for _ in range(5): graph_fwd(gc)
    torch.cuda.synchronize(); t0 = time.time(); N = 50
    for _ in range(N): graph_fwd(gc)
    torch.cuda.synchronize(); eager_ms = (time.time() - t0) / N * 1000
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(N): g.replay()
    torch.cuda.synchronize(); graph_ms = (time.time() - t0) / N * 1000
    print(f"eager round fwd {eager_ms:.2f} ms | graph replay {graph_ms:.2f} ms | "
          f"SPEEDUP {eager_ms/graph_ms:.1f}x", flush=True)
