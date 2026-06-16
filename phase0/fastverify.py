"""fast verify: a static-cache + CUDA-graph stage forward for spec-decode, ~5x cheaper
than the eager DynamicCache path (the per-round Python/kernel-launch overhead is removed,
not the math). proven bit-exact vs the eager static forward in research/fastverify_graph.py.

usage in a serve node, per generation:
    fv.reset()                      # zero the static cache
    h = fv.prefill(h, 0)            # eager, variable-length prompt
    ... per decode round ...
    h = fv.decode(h, start)         # fixed q_len = K+1, replays ONE captured graph

the cache owns its write position (gpt-oss calls update(k,v,idx) with no cache_position),
so rollback on rejection is free: a round just writes at `start` (= the committed length),
overwriting the previous round's rejected KV. StaticKV holds the FULL MAXLEN cache for every
layer and the sliding window is applied purely by the mask (exactly like the eager path's
single DynamicCache), so MAXLEN just has to cover prompt+gen -- no rolling buffer needed.
"""
import torch
from pipeline import _causal_mask


class StaticKV:
    """fixed [1, kv_heads, MAXLEN, head_dim] K/V per layer; update() writes at the cache's
    own index buffer (self.cp) and returns the full buffer for masked attention."""
    def __init__(self, n_layers, kv_heads, head_dim, maxlen, dev):
        z = lambda: [torch.zeros(1, kv_heads, maxlen, head_dim, dtype=torch.bfloat16, device=dev)
                     for _ in range(n_layers)]
        self.k, self.v = z(), z()
        self.cp = None
    def update(self, key, value, layer_idx, *a, **kw):
        self.k[layer_idx].index_copy_(2, self.cp, key)
        self.v[layer_idx].index_copy_(2, self.cp, value)
        return self.k[layer_idx], self.v[layer_idx]


class FastVerify:
    def __init__(self, parts, maxlen=2048, dev="cuda"):
        self.parts = parts; self.maxlen = maxlen; self.dev = dev
        self.layers = parts["layers"]; self.n_layers = len(self.layers)
        self.sliding = parts.get("sliding"); self.win = parts.get("window", 0)
        self.rotary = parts["rotary"]
        cfg = parts["_model"].config
        self.hidden = cfg.hidden_size
        kvh = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
        hd = getattr(cfg, "head_dim", None) or (self.hidden // cfg.num_attention_heads)
        self.cache = StaticKV(self.n_layers, kvh, hd, maxlen, dev)
        self.kp1 = None; self.graph = None; self.out = None     # decode buffers built lazily

    def reset(self):
        for t in self.cache.k: t.zero_()
        for t in self.cache.v: t.zero_()

    def _layers(self, x, pos, pe, mf, mw):
        for i, layer in enumerate(self.layers):
            m = mw if (self.sliding and self.sliding[i]) else mf
            o = layer(x, attention_mask=m, position_ids=pos, past_key_values=self.cache,
                      use_cache=True, position_embeddings=pe)
            x = o[0] if isinstance(o, tuple) else o
        return x

    def prefill(self, h, start):                                # eager, any length
        n = h.shape[1]
        self.cache.cp = torch.arange(start, start + n, device=self.dev)
        pos = self.cache.cp.unsqueeze(0)
        mf = _causal_mask(n, self.maxlen, start, 0, torch.bfloat16, self.dev)
        mw = _causal_mask(n, self.maxlen, start, self.win, torch.bfloat16, self.dev) if self.win else mf
        return self._layers(h, pos, self.rotary(h, pos), mf, mw)

    def _build(self, kp1):
        self.kp1 = kp1
        self.h_buf = torch.zeros(1, kp1, self.hidden, dtype=torch.bfloat16, device=self.dev)
        self.pos_buf = torch.zeros(1, kp1, dtype=torch.long, device=self.dev)
        self.cp_buf = torch.zeros(kp1, dtype=torch.long, device=self.dev)
        self.mf_buf = torch.zeros(1, 1, kp1, self.maxlen, dtype=torch.bfloat16, device=self.dev)
        self.mw_buf = torch.zeros(1, 1, kp1, self.maxlen, dtype=torch.bfloat16, device=self.dev)

    def _set(self, h, start):
        self.h_buf.copy_(h)
        ar = torch.arange(start, start + self.kp1, device=self.dev)
        self.pos_buf.copy_(ar.unsqueeze(0)); self.cp_buf.copy_(ar)
        self.mf_buf.copy_(_causal_mask(self.kp1, self.maxlen, start, 0, torch.bfloat16, self.dev))
        self.mw_buf.copy_(_causal_mask(self.kp1, self.maxlen, start, self.win, torch.bfloat16, self.dev)
                          if self.win else self.mf_buf)

    def _graph_body(self):
        self.cache.cp = self.cp_buf
        return self._layers(self.h_buf, self.pos_buf, self.rotary(self.h_buf, self.pos_buf),
                            self.mf_buf, self.mw_buf)

    def decode(self, h, start):                                 # fixed q_len, graphed
        if self.kp1 != h.shape[1]:                              # (re)build for this K+1, capture once
            self._build(h.shape[1]); self.graph = None
        self._set(h, start)
        if self.graph is None:
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): self._graph_body()
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = self._graph_body()
            self._set(h, start)                                 # warmup wrote the cache; restore round inputs
        self.graph.replay()
        return self.out
