"""EAGLE-3 draft head for MiniMax-M2.5, run COORDINATOR-SIDE in the shard spec-decode loop.

The head (thoughtworks/MiniMax-M2.5-Eagle3 = a LlamaForCausalLMEagle3: 1 Llama layer + an fc that fuses
3 target aux hidden states + a 32k draft-vocab lm_head + a d2t draft->target id map) PREDICTS the next K
tokens from M2.5's auxiliary hidden states (layers 1/30/58 of the last verified token) + M2.5's own token
embeddings. It works on NOVEL/reasoning text, where the n-gram prompt-lookup drafter (verbatim-reuse only)
is blind.

LOSSLESS by construction: this only PROPOSES. The ring greedy-verifies every token and commits the accepted
prefix + one correction, so a bad draft is just rejected — drafter quality moves SPEED (g), never output.
=> the port does NOT need bit-exact parity with vLLM; it needs to PREDICT M2.5 well enough to raise accept,
which we measure + tune empirically on the real engine.

Architecture-on-a-ring (why it's coordinator-side, no extra round-trip): EAGLE needs the target's aux hidden
states for the LAST accepted token; those RIDE BACK on the verify-return channel the coordinator already
reads. set_hidden(aux) is called by the coordinator after each verify; the draft then runs locally (~0.4ms,
0.2B) like the n-gram drafter. (It can't depth-pipeline — it needs the verified hidden — so it runs depth~1;
the n-gram path keeps depth-pipelining. The HybridDrafter routes between them.)

Contract (mirrors NgramDrafter): set_hidden(aux) -> request(ids,k) -> fetch() == k proposed target ids.
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
    """Autoregressive K-token EAGLE-3 draft chain. Fresh KV per round (the fused aux hidden re-seeds the
    chain from the verified state each round — correct EAGLE-3 operation; no cross-round draft KV needed)."""

    def __init__(self, eagle_dir, embed_tokens, device="cuda", max_pos=131072, next_hidden="final"):
        cfg = json.load(open(f"{eagle_dir}/config.json"))
        self.H = cfg["hidden_size"]; self.NH = cfg["num_attention_heads"]; self.NKV = cfg["num_key_value_heads"]
        self.HD = cfg["head_dim"]; self.eps = cfg["rms_norm_eps"]; self.theta = float(cfg["rope_theta"])
        self.dvocab = cfg["draft_vocab_size"]; self.GRP = self.NH // self.NKV; self.dev = device
        self.next_hidden = next_hidden                  # "final" (model output, = vLLM proposer) | "prenorm" (tunable)
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
        self._aux = None; self._pending = None
        self.matched = True                             # EAGLE always "produces" a draft (for HybridDrafter symmetry)

    def set_hidden(self, aux):
        """aux: tensor of the 3 target aux hidden states for the LAST verified position, shape [3,H] or [3H]
        (order = eagle_aux_hidden_state_layer_ids = layers [1,30,58])."""
        self._aux = None if aux is None else aux.reshape(-1).to(torch.bfloat16).to(self.dev)

    def request(self, ids, k):
        self._pending = (list(ids), k)

    def fetch(self):
        ids, k = self._pending
        return self.propose(ids, k)

    @torch.no_grad()
    def propose(self, ids, k):
        if self._aux is None or not ids:               # no hidden yet (e.g. right after prefill) -> degrade to plain
            return [ids[-1] if ids else 0] * k
        lin = torch.nn.functional.linear
        h = lin(self._aux.unsqueeze(0), self.fc)        # [1,H] fused target feature (fc: 3H -> H)
        tok = ids[-1]; pos0 = len(ids) - 1
        kc = []; vc = []; out = []
        for i in range(k):
            emb = self.embed[tok].unsqueeze(0)          # [1,H]
            en = _rms(emb, self.in_ln, self.eps)
            res = h
            hn = _rms(h, self.h_ln, self.eps)
            x = torch.cat([en, hn], -1)                 # [1,2H]  (layer_idx 0: embed ⊕ hidden)
            q = lin(x, self.qp).view(1, self.NH, self.HD)
            kk = lin(x, self.kp).view(1, self.NKV, self.HD)
            vv = lin(x, self.vp).view(1, self.NKV, self.HD)
            p = min(pos0 + i, self.cos.shape[0] - 1)
            cos = self.cos[p].view(1, 1, self.HD); sin = self.sin[p].view(1, 1, self.HD)
            q = q * cos + _rotate_half(q) * sin
            kk = kk * cos + _rotate_half(kk) * sin
            kc.append(kk); vc.append(vv)
            K = torch.cat(kc, 0).repeat_interleave(self.GRP, 1)   # [T,NH,HD]
            V = torch.cat(vc, 0).repeat_interleave(self.GRP, 1)
            qh = q.transpose(0, 1)                       # [NH,1,HD]
            att = torch.softmax((qh @ K.permute(1, 0, 2).transpose(-1, -2)).float() / (self.HD ** 0.5), -1).to(qh.dtype)
            o = (att @ V.permute(1, 0, 2)).transpose(0, 1).reshape(1, self.NH * self.HD)
            res = lin(o, self.op) + res
            hn2 = _rms(res, self.post_ln, self.eps)
            res = lin(torch.nn.functional.silu(lin(hn2, self.gp)) * lin(hn2, self.upp), self.dp) + res
            hf = _rms(res, self.norm, self.eps)
            did = int(lin(hf, self.lm).argmax(-1))
            tok = did + int(self.d2t[did])
            out.append(tok)
            h = hf if self.next_hidden == "final" else res
        return out


class HybridDrafter:
    """n-gram FIRST (free, depth-pipelinable, nails verbatim-reuse) -> EAGLE on n-gram MISS (novel/reasoning
    text). The split that makes one engine fast across both regimes. Lossless. The coordinator calls
    set_hidden() with the ring-returned aux hidden states each verify; matched tells it whether the last
    draft was the depth-pipelinable n-gram path (so it can keep depth up) or the serial EAGLE path."""

    def __init__(self, ngram, eagle):
        self.ngram = ngram; self.eagle = eagle; self._pending = None; self.matched = True

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
        self.matched = False                            # novel -> EAGLE (depth~1)
        return self.eagle.propose(ids, k)
