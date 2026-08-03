"""S2 risk check: is M2.5's NVFP4 FusedMoE TOKEN-COUNT-INVARIANT?

gpt-oss's mxfp4 MoE was deterministically non-invariant (a token's MLP output changed with
the batch's token count), which broke lossless batched/spec verify. This checks whether M2.5's
NVFP4 cutlass MoE has the same flaw: feed the SAME first token in batches of T=1,2,4,8 and
compare its MoE output row. Bit-exact across T => lossless K+1 verify is free.

  python m25_lossless_check.py --dir /root/m25 --layer 30
"""
import os, argparse, torch
os.environ.setdefault("M25_DIR", "/root/m25")
dev = "cuda"


def main(DIR, L):
    os.environ["M25_DIR"] = DIR
    import m25_stage as S
    vcfg = S.vllm_ctx()
    moe, gate = S._build_moe(L)
    from vllm.forward_context import set_forward_context
    torch.manual_seed(0)
    xfull = torch.randn(8, S.H, dtype=torch.bfloat16, device=dev) * 0.1
    outs = {}
    with torch.no_grad(), set_forward_context(None, vcfg):
        for T in [1, 2, 4, 8]:
            xt = xfull[:T].clone()
            rl = torch.nn.functional.linear(xt, gate)
            outs[T] = moe(xt, rl).clone()
    base = outs[1][0]
    print(f"row-0 MoE output invariance vs batch size (token L={L}):")
    all_exact = True
    for T in [2, 4, 8]:
        row = outs[T][0]
        exact = torch.equal(row, base)
        d = (row - base).abs().max().item()
        rel = ((row - base).norm() / base.norm().clamp_min(1e-9)).item()
        cs = torch.nn.functional.cosine_similarity(row.float(), base.float(), dim=0).item()
        print(f"  T={T}: exact={exact}  max|diff|={d:.6g}  rel_l2={rel:.3g}  cosine={cs:.7f}")
        all_exact = all_exact and exact
    if all_exact:
        print("VERDICT: NVFP4 MoE is TOKEN-COUNT-INVARIANT — lossless K+1 batched/spec verify is FREE on M2.5 (better than gpt-oss mxfp4).")
    else:
        # quantify: is it tiny (greedy still safe) or large (like mxfp4)?
        worst = max(((outs[T][0] - base).abs().max().item()) for T in [2, 4, 8])
        print(f"VERDICT: NOT bit-invariant (worst max|diff|={worst:.3g}). If tiny, greedy n-gram accept stays token-identical at the argmax; "
              "lossless-temperature needs a per-stream-MoE bypass (re-prove like GLM). If large, same class as the mxfp4 blocker.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/root/m25")
    ap.add_argument("--layer", type=int, default=30)
    a = ap.parse_args()
    main(a.dir, a.layer)
