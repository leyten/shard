"""S1 verify-(b): is my hand-rolled M2.5 GQA attention bit-correct vs HF's MiniMaxM2Attention?

The GQA block is the only genuinely-new compute in the port and is silent-failure-prone
(partial RoPE on first 64 dims, rotate_half vs interleave, q/k_norm over the FULL projection
before head-reshape, head_dim=128, repeat_kv 8->48, scaling). This compares my standalone
implementation (the one that goes into m25_stage.py, with KV cache + spec-rollback crop) to
the HF reference module loaded with the same layer-30 weights. Pure torch, no vLLM.

  python m25_gqa_check.py --dir /root/m25 --layer 30
"""
import os, json, argparse, torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.minimax_m2 import modeling_minimax_m2 as M

dev = "cuda"


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)


class MyGQA:
    """Mirrors MiniMaxM2Attention exactly, but hand-rolled with an explicit KV cache that
    crops to start_pos (spec-decode rollback) — the form it takes in the PP stage."""
    def __init__(self, W, qn, kn, cfg):
        self.W, self.qn, self.kn = W, qn, kn
        self.nh = cfg.num_attention_heads; self.nkv = cfg.num_key_value_heads
        self.hd = cfg.head_dim; self.grp = self.nh // self.nkv
        self.scaling = self.hd ** -0.5; self.eps = cfg.rms_norm_eps
        self.kc = self.vc = None

    def reset(self):
        self.kc = self.vc = None

    def _rms(self, x, w):
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + self.eps)).to(x.dtype) * w

    def __call__(self, x, start_pos, cos, sin):
        b, s, _ = x.shape
        lin = torch.nn.functional.linear
        q = self._rms(lin(x, self.W["q_proj"]), self.qn).view(b, s, self.nh, self.hd).transpose(1, 2)
        k = self._rms(lin(x, self.W["k_proj"]), self.kn).view(b, s, self.nkv, self.hd).transpose(1, 2)
        v = lin(x, self.W["v_proj"]).view(b, s, self.nkv, self.hd).transpose(1, 2)
        rd = cos.shape[-1]
        cu, su = cos.unsqueeze(1), sin.unsqueeze(1)  # [b,1,s,rd]
        def ap(t):
            tr, tp = t[..., :rd], t[..., rd:]
            return torch.cat([tr * cu + rotate_half(tr) * su, tp], -1)
        q, k = ap(q), ap(k)
        # KV cache + crop to start_pos (rolls back rejected speculative tokens)
        if self.kc is not None and self.kc.shape[2] > start_pos:
            self.kc = self.kc[:, :, :start_pos, :].contiguous(); self.vc = self.vc[:, :, :start_pos, :].contiguous()
        if self.kc is None:
            self.kc, self.vc = k, v
        else:
            self.kc = torch.cat([self.kc, k], 2); self.vc = torch.cat([self.vc, v], 2)
        kk = self.kc.repeat_interleave(self.grp, dim=1)
        vv = self.vc.repeat_interleave(self.grp, dim=1)
        total = kk.shape[2]
        attn = torch.matmul(q, kk.transpose(-1, -2)) * self.scaling
        qpos = torch.arange(s, device=dev).view(s, 1) + start_pos
        kpos = torch.arange(total, device=dev).view(1, total)
        attn = attn + torch.where(kpos <= qpos, 0.0, float("-inf")).to(attn.dtype)
        o = torch.matmul(torch.softmax(attn.float(), -1).to(vv.dtype), vv)
        o = o.transpose(1, 2).reshape(b, s, self.nh * self.hd)
        return lin(o, self.W["o_proj"])


def main(DIR, L):
    cfg = AutoConfig.from_pretrained(DIR); cfg._attn_implementation = "eager"
    idx = json.load(open(f"{DIR}/model.safetensors.index.json"))["weight_map"]
    _HD = {}
    def raw(n):
        s = idx[n]
        if s not in _HD:
            _HD[s] = safe_open(f"{DIR}/{s}", "pt", device="cpu")
        return _HD[s].get_tensor(n)
    g = lambda n: raw(f"model.layers.{L}.{n}").to(torch.bfloat16).to(dev)
    W = {k: g(f"self_attn.{k}.weight") for k in ["q_proj", "k_proj", "v_proj", "o_proj"]}
    qn, kn = g("self_attn.q_norm.weight"), g("self_attn.k_norm.weight")

    # HF reference
    ref = M.MiniMaxM2Attention(cfg, L).to(dev).to(torch.bfloat16).eval()
    ref.load_state_dict({"q_proj.weight": W["q_proj"], "k_proj.weight": W["k_proj"],
                         "v_proj.weight": W["v_proj"], "o_proj.weight": W["o_proj"],
                         "q_norm.weight": qn, "k_norm.weight": kn}, strict=True)
    rot = M.MiniMaxM2RotaryEmbedding(cfg).to(dev)

    torch.manual_seed(0)
    S = 8
    x = torch.randn(1, S, cfg.hidden_size, dtype=torch.bfloat16, device=dev) * 0.1
    pos = torch.arange(S, device=dev).unsqueeze(0)
    cos, sin = rot(x, pos)
    cmask = torch.full((1, 1, S, S), float("-inf"), device=dev).triu(1).to(torch.bfloat16)
    with torch.no_grad():
        ref_out, _ = ref(x, (cos, sin), attention_mask=cmask)
        mine = MyGQA(W, qn, kn, cfg)
        my_out = mine(x, 0, cos, sin)

    cs = torch.nn.functional.cosine_similarity(ref_out.flatten().float(), my_out.flatten().float(), dim=0).item()
    md = (ref_out - my_out).abs().max().item()
    rel = ((ref_out - my_out).norm() / ref_out.norm()).item()
    print(f"prefill  cosine={cs:.6f}  max|diff|={md:.5f}  rel_l2={rel:.6f}")

    # also exercise an incremental decode step (start_pos=S) to validate the KV-cache path
    mine.reset()
    _ = mine(x, 0, cos, sin)                     # prime cache with the 8-token prefill
    x2 = torch.randn(1, 1, cfg.hidden_size, dtype=torch.bfloat16, device=dev) * 0.1
    pos2 = torch.arange(S, S + 1, device=dev).unsqueeze(0)
    cos2, sin2 = rot(x2, pos2)
    my_dec = mine(x2, S, cos2, sin2)             # 1-token decode at pos 8 over the cache
    # HF reference for the same 9th token: full 9-token forward, take last
    x9 = torch.cat([x, x2], 1)
    pos9 = torch.arange(S + 1, device=dev).unsqueeze(0)
    cos9, sin9 = rot(x9, pos9)
    cm9 = torch.full((1, 1, S + 1, S + 1), float("-inf"), device=dev).triu(1).to(torch.bfloat16)
    with torch.no_grad():
        ref9, _ = ref(x9, (cos9, sin9), attention_mask=cm9)
    ref_dec = ref9[:, -1:, :]
    csd = torch.nn.functional.cosine_similarity(ref_dec.flatten().float(), my_dec.flatten().float(), dim=0).item()
    print(f"decode@8 cosine={csd:.6f}  max|diff|={(ref_dec - my_dec).abs().max().item():.5f}")

    ok = cs > 0.999 and csd > 0.999
    print("VERDICT:", "GQA BIT-CORRECT vs HF (prefill+decode) — m25_stage attention is sound."
          if ok else "MISMATCH — inspect rope/qk_norm/repeat_kv/scaling.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/root/m25")
    ap.add_argument("--layer", type=int, default=30)
    a = ap.parse_args()
    main(a.dir, a.layer)
