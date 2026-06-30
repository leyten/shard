"""EAGLE-3 draft head for MiniMax-M2.5, run COORDINATOR-SIDE in the shard spec-decode loop.

The head (thoughtworks/MiniMax-M2.5-Eagle3 = a LlamaForCausalLMEagle3: 1 Llama layer + an fc that fuses
3 target aux hidden states + a 32k draft-vocab lm_head + a d2t draft->target id map) PREDICTS the next K
tokens from M2.5's auxiliary hidden states (layers 1/30/58 of the verified positions) + M2.5's own token
embeddings. It works on NOVEL/reasoning text, where the n-gram prompt-lookup drafter (verbatim-reuse only)
is blind.

LOSSLESS by construction: this only PROPOSES. The ring greedy-verifies every token and commits the accepted
prefix + one correction, so a bad draft is just rejected — drafter quality moves SPEED (g), never output.
=> the port does NOT need bit-exact parity with vLLM; it needs to PREDICT M2.5 well enough to raise accept,
which we measure + tune empirically on the real engine.

The real EAGLE-3 draft is a transformer that attends CAUSALLY over the ENTIRE committed sequence — each
prior position carries the target's fused aux feature. The earlier (broken) port started from an EMPTY KV
cache every propose() and so attended only to the <=K draft tokens, never the committed history; it ignored
the aux and degenerated to repetition. This version keeps a PERSISTENT context KV cache:

    reset()                      -> clear the context (new generation)
    extend(tokens, auxes, base)  -> append committed positions to the context cache (k/v only)
    propose(k)                   -> draft K tokens, first query over the FULL committed context, then
                                    autoregress K-1 more (temporary chain k/v, discarded after the call)

PAIRING / RoPE (derived from the verified reference, NOT guessed):
  * vLLM eagle proposer set_inputs_first_pass (llm_base_proposer.py L792-805):
        input_ids[:n-1] = target_token_ids[1:]          # shift the tokens LEFT by one
        input_ids[last] = next_token_ids                # last slot = the just-committed (bonus) token
        positions      = target_positions               # positions UNCHANGED (= the hidden's position)
        hidden_states  = target_hidden_states            # aux UNCHANGED
    => draft slot i holds (embed(t_{i+1}), a_i) at RoPE position i, where a_i is the target hidden whose
       argmax PREDICTED t_{i+1}. Equivalently, committed token t_j (j>=1) pairs with a_{j-1} at RoPE j-1:
       each token's embedding is paired with the hidden ONE POSITION EARLIER (the one that predicted it).
  * The draft layer math (residual=fc_out; hidden_norm; input_layernorm; cat[embed,hidden]; attn;
    post_attention_layernorm; mlp) mirrors vLLM llama_eagle3.py LlamaDecoderLayer.forward (L102-122) +
    _norm_after_residual (L88-93).
  * The autoregressive chain carries the PRENORM residual (vLLM llama_eagle3.py LlamaModel.forward L249-254:
    aux_output = hidden_prenorm; and proposer loop L641-706: hidden_states[next] = prev prenorm, positions
    += 1). next_hidden="prenorm" reproduces this; "final" is a tunable A/B.

extend() contract: tokens[i] is the committed token PREDICTED BY auxes[i] (the caller applies the EAGLE
left-shift: auxes are the target hidden at absolute positions base..base+n-1, tokens are the committed
token ids at absolute positions base+1..base+n). base_pos = absolute position of auxes[0] (= RoPE of slot 0).
To draft after the last committed token t_m: extend ...,[t_m],[a_{m-1}] (RoPE m-1); propose(k)'s first query
is that last slot, attending over the whole cache, predicting t_{m+1}; the chain continues from there.

Compat shims (deprecated single-aux path): set_hidden(aux) -> request(ids,k) -> fetch() == k target ids;
propose(ids, k) still works (uses the persistent context if present, else seeds a 1-slot context).
"""
import json
import torch
from safetensors.torch import load_file


def _rms(x, w, eps):
    v = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(v + eps)).to(x.dtype) * w


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)


class EagleDrafter:
    """Autoregressive K-token EAGLE-3 draft chain over a PERSISTENT committed-context KV cache. extend()
    grows the committed cache (the real EAGLE-3 attends causally over all of it); propose() drafts K tokens
    off that context, autoregressing with a TEMPORARY chain cache that is discarded after the call (only
    extend() ever mutates the committed cache)."""

    def __init__(self, eagle_dir, embed_tokens, device="cuda", max_pos=131072, next_hidden="final"):
        cfg = json.load(open(f"{eagle_dir}/config.json"))
        self.H = cfg["hidden_size"]; self.NH = cfg["num_attention_heads"]; self.NKV = cfg["num_key_value_heads"]
        self.HD = cfg["head_dim"]; self.eps = cfg["rms_norm_eps"]; self.theta = float(cfg["rope_theta"])
        self.dvocab = cfg["draft_vocab_size"]; self.GRP = self.NH // self.NKV; self.dev = device
        self.next_hidden = next_hidden                  # "prenorm" (res, = vLLM reference carry) | "final" (tunable)
        w = load_file(f"{eagle_dir}/model.safetensors")
        g = lambda n: w[n].to(torch.bfloat16).to(device)
        self.fc = g("fc.weight")
        self.in_ln = g("midlayer.input_layernorm.weight"); self.h_ln = g("midlayer.hidden_norm.weight")
        self.qp = g("midlayer.self_attn.q_proj.weight"); self.kp = g("midlayer.self_attn.k_proj.weight")
        self.vp = g("midlayer.self_attn.v_proj.weight"); self.op = g("midlayer.self_attn.o_proj.weight")
        self.post_ln = g("midlayer.post_attention_layernorm.weight")
        self.gp = g("midlayer.mlp.gate_proj.weight"); self.upp = g("midlayer.mlp.up_proj.weight"); self.dp = g("midlayer.mlp.down_proj.weight")
        self.norm = g("norm.weight"); self.lm = g("lm_head.weight")
        self.d2t = w["d2t"].to(device).long()
        self.embed = embed_tokens                       # M2.5 [vocab, H] bf16 on device
        inv = 1.0 / (self.theta ** (torch.arange(0, self.HD, 2, device=device).float() / self.HD))
        fr = torch.outer(torch.arange(max_pos, device=device).float(), inv)
        e = torch.cat([fr, fr], -1)
        self.cos = e.cos().to(torch.bfloat16); self.sin = e.sin().to(torch.bfloat16)   # [max_pos, HD]
        self.reset()
        self._aux = None; self._pending = None
        self.matched = True                             # EAGLE always "produces" a draft (for HybridDrafter symmetry)

    # ---- persistent committed-context cache -------------------------------------
    def reset(self):
        """Clear the committed context (start of a new generation)."""
        self.kc = []                                    # per committed slot: RoPE'd key [1, NKV, HD]
        self.vc = []                                    # per committed slot: value    [1, NKV, HD]
        self.ctx_len = 0
        self._last_h = None                             # fc_out [1,H] of the last committed slot (propose's step-0 residual)
        self._last_tok = None                           # token id of the last committed slot
        self._last_pos = -1                             # RoPE position of the last committed slot

    def _aux_to_mat(self, auxes, n):
        """Normalize auxes (tensor [n,3,H] / [n,3H], or a list of per-position [3,H]/[3H]) -> [n, 3H] bf16."""
        if not torch.is_tensor(auxes):
            auxes = torch.stack([torch.as_tensor(a) for a in auxes], 0)
        A = auxes.to(torch.bfloat16).to(self.dev).reshape(n, -1)
        assert A.shape[1] == self.fc.shape[1], f"aux feature {A.shape[1]} != fc in-dim {self.fc.shape[1]}"
        return A

    @torch.no_grad()
    def extend(self, tokens, auxes, base_pos):
        """Append committed positions to the context cache. tokens[i] pairs with auxes[i] at RoPE base_pos+i
        (the caller supplies the EAGLE shift: auxes[i] = the target hidden that predicted tokens[i]). Only the
        k/v are stored; the slot's query is re-formed in propose() (the last slot) / the chain (drafted slots)."""
        if tokens is None:
            return
        tokens = tokens.tolist() if torch.is_tensor(tokens) else list(tokens)
        n = len(tokens)
        if n == 0:
            return
        lin = torch.nn.functional.linear
        fc_out = lin(self._aux_to_mat(auxes, n), self.fc)      # [n,H] fused target feature per position
        for i in range(n):
            tok = int(tokens[i])
            h = fc_out[i:i + 1]                                 # [1,H]
            en = _rms(self.embed[tok].unsqueeze(0), self.in_ln, self.eps)
            hn = _rms(h, self.h_ln, self.eps)
            x = torch.cat([en, hn], -1)                         # [1,2H] (layer_idx 0: embed ⊕ hidden)
            kk = lin(x, self.kp).view(1, self.NKV, self.HD)
            vv = lin(x, self.vp).view(1, self.NKV, self.HD)
            p = min(base_pos + i, self.cos.shape[0] - 1)
            cos = self.cos[p].view(1, 1, self.HD); sin = self.sin[p].view(1, 1, self.HD)
            kk = kk * cos + _rotate_half(kk) * sin
            self.kc.append(kk); self.vc.append(vv)
            self._last_h = h; self._last_tok = tok; self._last_pos = p
        self.ctx_len += n

    # ---- the drafter ------------------------------------------------------------
    @torch.no_grad()
    def _draft(self, k):
        """Draft k tokens over the persistent context. Step 0's query is the LAST committed slot (whose k/v
        already sit in the cache); steps 1..k-1 autoregress, appending each drafted slot's k/v to a TEMPORARY
        chain (never the committed cache). Returns k target-space token ids."""
        if self.ctx_len == 0 or self._last_h is None:
            return [int(self._last_tok) if self._last_tok is not None else 0] * k
        lin = torch.nn.functional.linear
        Kp = torch.cat(self.kc, 0).repeat_interleave(self.GRP, 1)   # [L,NH,HD] committed keys (GQA-expanded)
        Vp = torch.cat(self.vc, 0).repeat_interleave(self.GRP, 1)
        ck = []; cv = []                                            # temporary draft-chain k/v (discarded after the call)
        out = []
        h = self._last_h; tok = self._last_tok; base = self._last_pos
        for i in range(k):
            en = _rms(self.embed[tok].unsqueeze(0), self.in_ln, self.eps)
            hn = _rms(h, self.h_ln, self.eps)
            x = torch.cat([en, hn], -1)                            # [1,2H]
            res = h                                                # residual = fc_out (step 0) / prev hidden (chain)
            q = lin(x, self.qp).view(1, self.NH, self.HD)
            p = min(base + i, self.cos.shape[0] - 1)
            cos = self.cos[p].view(1, 1, self.HD); sin = self.sin[p].view(1, 1, self.HD)
            q = q * cos + _rotate_half(q) * sin
            if i == 0:                                             # query = last committed slot (already cached)
                K, V = Kp, Vp
            else:                                                  # new chain slot: cache its k/v, then attend
                kk = lin(x, self.kp).view(1, self.NKV, self.HD)
                vv = lin(x, self.vp).view(1, self.NKV, self.HD)
                kk = kk * cos + _rotate_half(kk) * sin
                ck.append(kk.repeat_interleave(self.GRP, 1)); cv.append(vv.repeat_interleave(self.GRP, 1))
                K = torch.cat([Kp] + ck, 0); V = torch.cat([Vp] + cv, 0)
            qh = q.transpose(0, 1)                                 # [NH,1,HD]
            att = torch.softmax((qh @ K.permute(1, 0, 2).transpose(-1, -2)).float() / (self.HD ** 0.5), -1).to(qh.dtype)
            o = (att @ V.permute(1, 0, 2)).transpose(0, 1).reshape(1, self.NH * self.HD)
            res = lin(o, self.op) + res
            hn2 = _rms(res, self.post_ln, self.eps)
            res = lin(torch.nn.functional.silu(lin(hn2, self.gp)) * lin(hn2, self.upp), self.dp) + res
            hf = _rms(res, self.norm, self.eps)
            did = int(lin(hf, self.lm).argmax(-1))
            tok = did + int(self.d2t[did])
            out.append(tok)
            h = hf if self.next_hidden == "final" else res        # carry: prenorm (res) = vLLM reference
        return out

    @torch.no_grad()
    def propose(self, a, b=None):
        """Primary: propose(k) -> draft k tokens over the persistent context (built via extend()).
        Compat: propose(ids, k) -> if a context exists use it (ids ignored); else seed a 1-slot context from
        the last set_hidden() aux (deprecated single-position path), or degrade to repeat if no aux."""
        if b is None:
            return self._draft(a)
        ids, k = a, b
        if self.ctx_len > 0:
            return self._draft(k)
        if self._aux is None or not ids:                          # no context, no seed -> degrade to plain decode
            return [int(ids[-1]) if ids else 0] * k
        self.reset()                                              # legacy single-aux seed: pair ids[-1] with that aux
        self.extend([ids[-1]], self._aux, base_pos=max(len(ids) - 1, 0))
        return self._draft(k)

    # ---- compat shims (deprecated; prefer reset/extend/propose) ------------------
    def set_hidden(self, aux):
        """aux: the 3 target aux hidden states for the LAST verified position, shape [3,H] or [3H]
        (order = eagle_aux_hidden_state_layer_ids = layers [1,30,58]). Stashed for the legacy propose(ids,k)."""
        self._aux = None if aux is None else aux.reshape(-1).to(torch.bfloat16).to(self.dev)

    def request(self, ids, k):
        self._pending = (list(ids), k)

    def fetch(self):
        ids, k = self._pending
        return self.propose(ids, k)


class HybridDrafter:
    """n-gram FIRST (free, depth-pipelinable, nails verbatim-reuse) -> EAGLE on n-gram MISS (novel/reasoning
    text). The split that makes one engine fast across both regimes. Lossless. The coordinator keeps the
    EAGLE committed context current via reset()/extend() (the ring returns the target aux on the verify-return
    channel); matched tells it whether the last draft was the depth-pipelinable n-gram path or the serial
    EAGLE path."""

    def __init__(self, ngram, eagle):
        self.ngram = ngram; self.eagle = eagle; self._pending = None; self.matched = True

    def reset(self):
        self.eagle.reset()

    def extend(self, tokens, auxes, base_pos):
        self.eagle.extend(tokens, auxes, base_pos)

    def set_hidden(self, aux):
        self.eagle.set_hidden(aux)

    def request(self, ids, k):
        self.ngram.request(ids, k); self._pending = (list(ids), k)

    def fetch(self):
        ids, k = self._pending
        ng = self.ngram.fetch()
        if getattr(self.ngram, "matched", False):       # n-gram found a real longest-match -> draftable, use it (depth-pipeline)
            self.matched = True
            return ng
        self.matched = False                            # novel -> EAGLE (depth~1) over its persistent context
        return self.eagle.propose(ids, k)
