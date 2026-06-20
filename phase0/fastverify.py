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
from tree import tree_mask


class ContextOverflow(Exception):
    """A request needs more positions than the static cache (max_ctx) holds. Raised
    BEFORE any out-of-bounds write, so the cache is never corrupted — the serve loop
    catches it, resets the edge, and the node stays up (vs a silent CUDA OOB)."""


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
        self.cfg = cfg                                          # flip _attn_implementation per phase
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

    def _flex_masks(self, q, start):
        """flex BlockMasks over [q, MAXLEN]. causal already ignores the unwritten cache tail
        (keys > start+qi are masked), so attending over the full StaticKV buffer is correct.
        KV_LEN is fixed at maxlen -> flex compiles per q only, not per context length."""
        from torch.nn.attention.flex_attention import create_block_mask
        win = self.win
        def causal(b, h, qi, ki): return (start + qi) >= ki
        def swin(b, h, qi, ki): return ((start + qi) >= ki) & ((start + qi) - ki < win)
        # _compile=True builds the block mask without materialising the dense [q, maxlen] mask
        # (which is ~3GB at maxlen=100k and OOMs) -- this is the whole point of flex.
        mf = create_block_mask(causal, 1, None, q, self.maxlen, device=self.dev, _compile=True)
        mw = create_block_mask(swin, 1, None, q, self.maxlen, device=self.dev, _compile=True) if win else mf
        return mf, mw

    def prefill(self, h, start):                                # FLEX, O(n) on Ada; chunk-friendly (q = a chunk)
        n = h.shape[1]
        if start + n > self.maxlen:                             # fail clean, never corrupt the cache
            raise ContextOverflow(f"prefill needs {start + n} positions > max_ctx {self.maxlen} "
                                  f"(raise --max-ctx)")
        self.cfg._attn_implementation = "flex_attention"        # memory-efficient for the big-q prefill
        self.cache.cp = torch.arange(start, start + n, device=self.dev)
        pos = self.cache.cp.unsqueeze(0)
        mf, mw = self._flex_masks(n, start)
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

    def decode(self, h, start):                                 # fixed q_len, graphed EAGER
        if start + h.shape[1] > self.maxlen:                    # fail clean before any OOB write
            raise ContextOverflow(f"decode needs {start + h.shape[1]} positions > max_ctx "
                                  f"{self.maxlen} (raise --max-ctx)")
        # eager attention with q=K+1 is cheap even at 100k kv, and (unlike flex) the additive
        # mask + CUDA graph give a fixed-shape replay -> no per-step recompile. flip back to eager
        # after a flex prefill (rebuild the graph the first time this phase runs).
        if self.cfg._attn_implementation != "eager":
            self.cfg._attn_implementation = "eager"; self.graph = None
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

    # ---- TREE verify: a fixed-topology tree (M nodes) graphed like the linear path ----
    # build_tree(w,d) gives a FIXED structure (par/dep constant; only token values change),
    # so the M-node forward is graphable. tree KV is stored at contiguous scratch slots
    # [start, start+M); the tree mask routes each node to its ancestors' slots + the prefix.
    def _tbuild(self, M, par, dep):
        self.tM = M; self.tpar = list(par)
        self.th = torch.zeros(1, M, self.hidden, dtype=torch.bfloat16, device=self.dev)
        self.tpos = torch.zeros(1, M, dtype=torch.long, device=self.dev)
        self.tcp = torch.zeros(M, dtype=torch.long, device=self.dev)
        self.tmf = torch.zeros(1, 1, M, self.maxlen, dtype=torch.bfloat16, device=self.dev)
        self.tmw = torch.zeros(1, 1, M, self.maxlen, dtype=torch.bfloat16, device=self.dev)
        anc = torch.zeros(M, M, dtype=torch.bool, device=self.dev)           # anc[i,tj] = tj ancestor of i (incl self)
        for i in range(M):
            j = i
            while j != -1:
                anc[i, j] = True; j = par[j]
        self.tanc = anc
        self.tdepv = torch.tensor(dep, device=self.dev)

    def _tset(self, h, start, par, dep):
        """build the round's tree mask VECTORIZED (the topology is fixed; only `start`
        shifts the prefix boundary) -- no per-element GPU writes."""
        M = self.tM
        z = torch.zeros((), dtype=torch.bfloat16, device=self.dev)
        mnb = torch.full((), torch.finfo(torch.bfloat16).min, dtype=torch.bfloat16, device=self.dev)
        self.th.copy_(h)
        posM = start + self.tdepv                                            # [M] query abs positions
        self.tpos.copy_(posM.unsqueeze(0))
        self.tcp.copy_(torch.arange(start, start + M, device=self.dev))      # contiguous scratch slots
        cols = torch.arange(self.maxlen, device=self.dev)
        allow_f = torch.zeros(M, self.maxlen, dtype=torch.bool, device=self.dev)
        allow_f[:, :start] = True                                            # full: attend all committed prefix
        allow_f[:, start:start + M] = self.tanc                              # tree: attend ancestors (scratch slots)
        self.tmf[0, 0] = torch.where(allow_f, z, mnb)
        if self.win:
            allow_w = torch.zeros(M, self.maxlen, dtype=torch.bool, device=self.dev)
            allow_w[:, :start] = (posM[:, None] - cols[None, :start]) < self.win
            allow_w[:, start:start + M] = self.tanc & ((posM[:, None] - posM[None, :]) < self.win)
            self.tmw[0, 0] = torch.where(allow_w, z, mnb)
        else:
            self.tmw.copy_(self.tmf)

    def _tbody(self):
        self.cache.cp = self.tcp
        return self._layers(self.th, self.tpos, self.rotary(self.th, self.tpos), self.tmf, self.tmw)

    def tree_decode(self, h, start, par, dep):
        if getattr(self, "tM", None) != h.shape[1] or getattr(self, "tpar", None) != list(par):
            self._tbuild(h.shape[1], par, dep); self.tgraph = None
        self._tset(h, start, par, dep)
        if getattr(self, "tgraph", None) is None:
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): self._tbody()
            torch.cuda.current_stream().wait_stream(s)
            self.tgraph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.tgraph):
                self.tout = self._tbody()
            self._tset(h, start, par, dep)                                   # warmup dirtied the cache; restore inputs
        self.tgraph.replay()
        return self.tout

    def tree_gather(self, start, keep):
        """compact the accepted path's KV (scratch slots start+keep[i]) to contiguous
        committed positions start+i, every layer. keep[0]=0 (root). next round's prefix
        becomes [0, start+len(keep))."""
        src = torch.tensor([start + k for k in keep], device=self.dev)
        dst = torch.arange(start, start + len(keep), device=self.dev)
        for lk, lv in zip(self.cache.k, self.cache.v):
            lk.index_copy_(2, dst, lk.index_select(2, src))
            lv.index_copy_(2, dst, lv.index_select(2, src))
