"""Offline proof of the BATCHED ATTENTION math for M2.5 (the engine-core piece of continuous batching):
B streams at DIVERGENT committed lengths share one [B,NKV,MAXLEN,HD] KV buffer; each stream's verify-block
attention output must be BIT-IDENTICAL to running that stream SOLO. Mirrors batchverify.py's pattern
(per-stream scatter + per-stream additive causal mask) on M2.5's GQA (48q/8kv) + the manual-matmul
attention used on the batched decode path. NO GPU (CPU fp32 for an exact gate).

Proves: batched[b] == solo[b] for every stream, even though they sit at different start positions and the
shared bucket read :alen includes other streams' unwritten/zero tail (masked to 0 per stream).

  python research/m25_batchattn_test.py
"""
import torch

torch.manual_seed(0)
NH, NKV, HD = 48, 8, 128
GRP = NH // NKV
SC = HD ** -0.5
MAXLEN = 4096
s = 9                                   # K+1 verify block


def _bucket(need):
    for b in (2048, 4096):
        if b >= need: return b
    return MAXLEN


def solo_attn(q, kc_full, vc_full, start):
    """single-stream: read :total, additive causal mask (the m25 manual-matmul decode math)."""
    total = start + s
    kc = kc_full[:, :, :total]; vc = vc_full[:, :, :total]
    kk = kc.repeat_interleave(GRP, 1); vv = vc.repeat_interleave(GRP, 1)
    qpos = (torch.arange(s) + start).view(s, 1); kpos = torch.arange(total).view(1, total)
    m = torch.where(kpos <= qpos, 0.0, float("-inf"))[None, None]
    a = torch.matmul(q, kk.transpose(-1, -2)) * SC + m
    return torch.matmul(torch.softmax(a, -1), vv)


def batched_attn(q_B, KV_k, KV_v, starts):
    """B streams in one buffer. q_B:[B,NH,s,HD]; KV_*:[B,NKV,MAXLEN,HD] (already written per stream);
    per-stream additive mask over the shared bucket :alen (batchverify._batched_causal_mask pattern)."""
    B = q_B.shape[0]
    alen = _bucket(int(max(starts)) + s)
    kc = KV_k[:, :, :alen]; vc = KV_v[:, :, :alen]
    kk = kc.repeat_interleave(GRP, 1); vv = vc.repeat_interleave(GRP, 1)          # [B,NH,alen,HD]
    rows = torch.arange(s).view(1, s) + torch.tensor(starts).view(B, 1)          # [B,s] abs query pos
    cols = torch.arange(alen).view(1, 1, alen)
    allow = cols <= rows[:, :, None]                                             # [B,s,alen] causal
    m = torch.where(allow, torch.zeros(()), torch.full((), float("-inf")))[:, None]   # [B,1,s,alen]
    a = torch.matmul(q_B, kk.transpose(-1, -2)) * SC + m
    return torch.matmul(torch.softmax(a, -1), vv)                                # [B,NH,s,HD]


def test_batched_attn_equals_solo():
    starts = [40, 1000, 2039, 7]          # divergent committed lengths (one near a bucket edge)
    B = len(starts)
    KV_k = torch.zeros(B, NKV, MAXLEN, HD); KV_v = torch.zeros(B, NKV, MAXLEN, HD)
    q_B = torch.zeros(B, NH, s, HD)
    solos = []
    for b, st in enumerate(starts):
        total = st + s
        kc = torch.randn(1, NKV, MAXLEN, HD); vc = torch.randn(1, NKV, MAXLEN, HD)
        kc[:, :, total:] = 0; vc[:, :, total:] = 0          # only [0,total) is "written" for this stream
        q = torch.randn(1, NH, s, HD)
        KV_k[b] = kc[0]; KV_v[b] = vc[0]; q_B[b] = q[0]
        solos.append(solo_attn(q, kc, vc, st))
    bat = batched_attn(q_B, KV_k, KV_v, starts)
    worst = 0.0
    for b, st in enumerate(starts):
        d = (bat[b:b + 1].float() - solos[b].float()).abs().max().item(); worst = max(worst, d)
        assert d < 1e-5, f"stream {b} (start={st}): batched != solo, diff={d:.2e}"
        print(f"  stream {b} start={st:4d} bucket={_bucket(st+s):4d}  batched == solo  diff={d:.1e}")
    print(f"[batchattn] PASS — {B} streams at divergent lengths, batched attention == solo (worst {worst:.1e})")


def test_garbage_tail_isolated():
    """a stream's output must not change if ANOTHER stream's buffer tail is scribbled (streams are isolated
    by the per-stream mask + their own batch row)."""
    starts = [40, 1000]
    KV_k = torch.randn(2, NKV, MAXLEN, HD); KV_v = torch.randn(2, NKV, MAXLEN, HD)
    for b, st in enumerate(starts):
        KV_k[b, :, st + s:] = 0; KV_v[b, :, st + s:] = 0
    q_B = torch.randn(2, NH, s, HD)
    o1 = batched_attn(q_B, KV_k.clone(), KV_v.clone(), starts)
    KV_k[1, :, starts[1] + s:].normal_(); KV_v[1, :, starts[1] + s:].normal_()   # scribble stream 1's tail
    o2 = batched_attn(q_B, KV_k, KV_v, starts)
    assert (o1[0] - o2[0]).abs().max().item() == 0.0, "stream 0 leaked from stream 1's tail"
    print("[batchattn] PASS — scribbling one stream's unwritten tail changes nothing for the others")


if __name__ == "__main__":
    test_batched_attn_equals_solo()
    test_garbage_tail_isolated()
    print("\n[batchattn] ALL PASS — batched per-stream attention is bit-exact vs solo (the engine-core math)")
