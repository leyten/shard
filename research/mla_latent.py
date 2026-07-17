"""mla_latent.py — MLA "absorb" attention: cache the compressed kv-latent (~kv_lora dims) + the shared rope
key instead of the decompressed full-head K/V, so attention runs in latent space and full-head K/V is NEVER
materialized. This is the fix for the 2026-06-26 KV cache wall (the GLM-5.2 dense ring OOMs ~16-24k because
leyten's attn caches decompressed full-head K/V). ~70x smaller cache -> 100k context + concurrency headroom.

The math (per head h), with c' = rms(kv_a_layernorm, latent), W_kn[h] the nope slice of kv_b, W_vb[h] the
value slice:
  naive score_nope[h] = q_pass[h] · k_nope[h]^T   where k_nope[h] = c' · W_kn[h]^T
                      = q_pass[h] · W_kn[h] · c'^T = (q_pass[h] · W_kn[h]) · c'^T
  => absorb W_kn into the query: q_abs[h] = q_pass[h] · W_kn[h]  (now in kv_lora space); cache c' (shared).
  naive out[h] = softmax[h] · value[h]   where value[h] = c' · W_vb[h]^T
              = (softmax[h] · c') · W_vb[h]^T
  => o_lat[h] = softmax[h] · c' (in kv_lora space), then expand once: out[h] = o_lat[h] · W_vb[h]^T.
The rope part (q_rot · k_rot^T) is unchanged — k_rot is shared across heads and cached directly (small).

This module proves the identity offline ($0, weight-agnostic) BEFORE any box integration. Run:
  python3 mla_latent.py        # asserts absorbed == naive on random tensors
"""
import torch


def naive_attn(x, W, dims, mask):
    """Leyten-style decompressed-cache attention (reference). Returns [b,s,H]."""
    b, s, H = x.shape
    nh, qkn, qkr, vh, kvl = dims["nh"], dims["qkn"], dims["qkr"], dims["vh"], dims["kvl"]
    qkh = qkn + qkr
    scale = qkh ** -0.5
    q = torch.nn.functional.linear(_rms(torch.nn.functional.linear(x, W["q_a"]), W["q_a_ln"]), W["q_b"])
    q = q.view(b, s, nh, qkh).transpose(1, 2)                          # [b,nh,s,qkh]
    q_pass, q_rot = torch.split(q, [qkn, qkr], -1)
    ckv = torch.nn.functional.linear(x, W["kv_a"])                     # [b,s,kvl+qkr]
    k_pass_c, k_rot = torch.split(ckv, [kvl, qkr], -1)
    cprime = _rms(k_pass_c, W["kv_a_ln"])                              # [b,s,kvl]
    k_pass = torch.nn.functional.linear(cprime, W["kv_b"]).view(b, s, nh, qkn + vh).transpose(1, 2)
    k_nope, value = torch.split(k_pass, [qkn, vh], -1)                 # [b,nh,s,qkn],[b,nh,s,vh]
    k_rot_e = k_rot.view(b, 1, s, qkr).expand(b, nh, s, qkr)
    Q = torch.cat([q_pass, q_rot], -1)                                # [b,nh,s,qkh]
    K = torch.cat([k_nope, k_rot_e], -1)
    score = (Q @ K.transpose(-1, -2)) * scale + mask
    out = torch.softmax(score.float(), -1).to(value.dtype) @ value    # [b,nh,s,vh]
    out = out.transpose(1, 2).reshape(b, s, nh * vh)
    return torch.nn.functional.linear(out, W["o"])


def absorbed_attn(x, W, dims, mask):
    """Latent-cache MLA: kv_b absorbed into q (scores) and o (values); full-head K/V never built."""
    b, s, H = x.shape
    nh, qkn, qkr, vh, kvl = dims["nh"], dims["qkn"], dims["qkr"], dims["vh"], dims["kvl"]
    qkh = qkn + qkr
    scale = qkh ** -0.5
    q = torch.nn.functional.linear(_rms(torch.nn.functional.linear(x, W["q_a"]), W["q_a_ln"]), W["q_b"])
    q = q.view(b, s, nh, qkh).transpose(1, 2)
    q_pass, q_rot = torch.split(q, [qkn, qkr], -1)                    # [b,nh,s,qkn],[b,nh,s,qkr]
    ckv = torch.nn.functional.linear(x, W["kv_a"])
    k_pass_c, k_rot = torch.split(ckv, [kvl, qkr], -1)
    cprime = _rms(k_pass_c, W["kv_a_ln"])                             # [b,s,kvl]  <-- THE CACHE (shared across heads)
    # split kv_b [nh*(qkn+vh), kvl] -> per-head W_kn [nh,qkn,kvl], W_vb [nh,vh,kvl]
    kv_b = W["kv_b"].view(nh, qkn + vh, kvl)
    W_kn, W_vb = kv_b[:, :qkn, :], kv_b[:, qkn:, :]
    # absorb W_kn into q_pass -> q in latent space [b,nh,s,kvl]
    q_abs = torch.einsum("bhsn,hnl->bhsl", q_pass, W_kn)
    score_nope = torch.einsum("bhsl,bml->bhsm", q_abs, cprime)        # [b,nh,s,s]  (cprime shared over heads)
    score_rope = torch.einsum("bhsr,bmr->bhsm", q_rot, k_rot)         # k_rot shared [b,s,qkr]
    score = (score_nope + score_rope) * scale + mask
    p = torch.softmax(score.float(), -1).to(cprime.dtype)
    o_lat = torch.einsum("bhsm,bml->bhsl", p, cprime)                 # [b,nh,s,kvl]  attention in latent space
    out = torch.einsum("bhsl,hvl->bhsv", o_lat, W_vb)                 # expand ONCE -> [b,nh,s,vh]
    out = out.transpose(1, 2).reshape(b, s, nh * vh)
    return torch.nn.functional.linear(out, W["o"])


def _rms(x, w, eps=1e-6):
    return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def _equiv_test():
    torch.manual_seed(0)
    b, s = 1, 24
    dims = dict(H=64, nh=8, qkn=16, qkr=8, vh=16, kvl=32)
    H, nh, qkn, qkr, vh, kvl = dims["H"], dims["nh"], dims["qkn"], dims["qkr"], dims["vh"], dims["kvl"]
    qkh = qkn + qkr
    dt = torch.float64                                               # high precision -> tight equivalence bound
    W = dict(
        q_a=torch.randn(48, H, dtype=dt), q_a_ln=torch.randn(48, dtype=dt),
        q_b=torch.randn(nh * qkh, 48, dtype=dt),
        kv_a=torch.randn(kvl + qkr, H, dtype=dt), kv_a_ln=torch.randn(kvl, dtype=dt),
        kv_b=torch.randn(nh * (qkn + vh), kvl, dtype=dt),
        o=torch.randn(H, nh * vh, dtype=dt),
    )
    x = torch.randn(b, s, H, dtype=dt)
    qpos = torch.arange(s).view(s, 1); kpos = torch.arange(s).view(1, s)
    mask = torch.where(kpos <= qpos, 0.0, float("-inf")).to(dt)
    a = naive_attn(x, W, dims, mask)
    c = absorbed_attn(x, W, dims, mask)
    err = (a - c).abs().max().item()
    print(f"[mla_latent] naive vs absorbed: max abs err = {err:.2e}  (shape {tuple(a.shape)})")
    assert err < 1e-9, f"ABSORB MISMATCH ({err:.2e}) — math is wrong, do NOT integrate"
    # cache-size proof
    full = nh * (qkh + vh); lat = kvl + qkr
    print(f"[mla_latent] per-position cache floats: decompressed full-head={full}  latent={lat}  -> {full/lat:.1f}x smaller")
    print("[mla_latent] EQUIVALENCE PASS — absorb identity holds; safe to integrate into Layer.attn")


def _chunked_cache_test():
    """De-risk the new cache PLUMBING: a sequence fed in chunks (cache append/crop + RoPE-across-chunks)
    must equal the same sequence all-at-once. This is the part the single-forward absorb test doesn't cover
    and the part that bit us on the box, so prove it offline too."""
    torch.manual_seed(1)
    b, S = 1, 24
    dims = dict(H=64, nh=8, qkn=16, qkr=8, vh=16, kvl=32)
    H, nh, qkn, qkr, vh, kvl = (dims[k] for k in ("H", "nh", "qkn", "qkr", "vh", "kvl"))
    qkh = qkn + qkr; scale = qkh ** -0.5
    dt = torch.float64
    W = dict(q_a=torch.randn(48, H, dtype=dt), q_a_ln=torch.randn(48, dtype=dt), q_b=torch.randn(nh*qkh, 48, dtype=dt),
             kv_a=torch.randn(kvl+qkr, H, dtype=dt), kv_a_ln=torch.randn(kvl, dtype=dt),
             kv_b=torch.randn(nh*(qkn+vh), kvl, dtype=dt), o=torch.randn(H, nh*vh, dtype=dt))
    kvb = W["kv_b"].view(nh, qkn+vh, kvl); W_kn, W_vb = kvb[:, :qkn, :].contiguous(), kvb[:, qkn:, :].contiguous()
    L = lambda x, Wt: torch.nn.functional.linear(x, Wt)

    def rope(t, pos):
        half = qkr // 2; ang = pos[:, None] / (10000 ** (torch.arange(half, dtype=dt) / half))
        cos = torch.cat([ang.cos(), ang.cos()], -1); sin = torch.cat([ang.sin(), ang.sin()], -1)
        t1, t2 = t[..., :half], t[..., half:]
        return t * cos + torch.cat([-t2, t1], -1) * sin

    class Lat:
        def __init__(self): self.cc = self.rc = None
        def attn(self, x, start_pos):
            s = x.shape[1]
            q = L(_rms(L(x, W["q_a"]), W["q_a_ln"]), W["q_b"]).view(b, s, nh, qkh).transpose(1, 2)
            q_pass, q_rot = torch.split(q, [qkn, qkr], -1)
            ckv = L(x, W["kv_a"]); k_pass_c, k_rot = torch.split(ckv, [kvl, qkr], -1)
            cprime = _rms(k_pass_c, W["kv_a_ln"])
            pos = torch.arange(s, dtype=dt) + start_pos
            q_rot, k_rot = rope(q_rot, pos), rope(k_rot, pos)
            if self.cc is not None and self.cc.shape[1] > start_pos:
                self.cc, self.rc = self.cc[:, :start_pos], self.rc[:, :start_pos]
            self.cc, self.rc = (cprime, k_rot) if self.cc is None else (torch.cat([self.cc, cprime], 1), torch.cat([self.rc, k_rot], 1))
            total = self.cc.shape[1]
            q_abs = torch.einsum("bhsn,hnl->bhsl", q_pass, W_kn)
            attn = (torch.einsum("bhsl,bml->bhsm", q_abs, self.cc) + torch.einsum("bhsr,bmr->bhsm", q_rot, self.rc)) * scale
            qpos = torch.arange(s).view(s, 1) + start_pos; kpos = torch.arange(total).view(1, total)
            attn = attn + torch.where(kpos <= qpos, 0.0, -1e30).to(dt)
            p = torch.softmax(attn, -1)
            o = torch.einsum("bhsl,hvl->bhsv", torch.einsum("bhsm,bml->bhsl", p, self.cc), W_vb)
            return L(o.transpose(1, 2).reshape(b, s, nh * vh), W["o"])

    x = torch.randn(b, S, H, dtype=dt)
    full = Lat().attn(x, 0)
    B = Lat(); outs = []; pos = 0
    while pos < S:
        e = min(pos + 7, S); outs.append(B.attn(x[:, pos:e], pos)); pos = e
    err = (full - torch.cat(outs, 1)).abs().max().item()
    print(f"[mla_latent] latent chunked vs all-at-once: max abs err = {err:.2e}")
    assert err < 1e-9, f"CACHE PLUMBING MISMATCH ({err:.2e})"
    print("[mla_latent] CHUNKED-CACHE PASS — append/crop/RoPE-across-chunks correct")


if __name__ == "__main__":
    _equiv_test()
    _chunked_cache_test()
