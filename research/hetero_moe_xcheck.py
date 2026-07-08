"""Cross-kernel numeric compatibility: run ONE layer's NVFP4 MoE on a DETERMINISTIC
CPU-seeded input and dump the output — run on two boxes (5090 cutlass vs 4090 marlin),
then compare cosine offline. Answers whether a mixed-arch ring's stages drift enough to
false-flag the challenge spot-check (cos >= 0.99, shard/challenge.py).

  box A (5090):  python hetero_moe_xcheck.py --dir /root/m25 --layer 30 --backend cutlass --out /tmp/xcheck_cutlass.pt
  box B (4090):  python hetero_moe_xcheck.py --dir /root/m25 --layer 30 --backend marlin  --out /tmp/xcheck_marlin.pt
  anywhere:      python hetero_moe_xcheck.py --compare /tmp/xcheck_cutlass.pt /tmp/xcheck_marlin.pt
"""
import argparse
import sys

import torch


def compare(a_path, b_path):
    a = torch.load(a_path, map_location="cpu", weights_only=True).float()
    b = torch.load(b_path, map_location="cpu", weights_only=True).float()
    assert a.shape == b.shape, (a.shape, b.shape)
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    rel = ((a - b).norm() / (a.norm() + 1e-12)).item()
    per_tok = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    print(f"cosine(flat) = {cos:.6f}   rel_norm = {rel:.6f}")
    print(f"per-token cos: min {per_tok.min().item():.6f}  mean {per_tok.mean().item():.6f}")
    print("challenge verdict (cos>=0.99, rel<0.05):", "PASS" if cos >= 0.99 and rel < 0.05 else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2, default=None)
    ap.add_argument("--dir", default="/root/m25")
    ap.add_argument("--layer", type=int, default=30)
    ap.add_argument("--backend", default="cutlass")
    ap.add_argument("--out", required=False)
    ap.add_argument("--tokens", type=int, default=64)
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)

    sys.path.insert(0, "/root")
    from hetero_moe_probe import vllm_ctx, build_quant_config  # the probe's proven load path
    import json
    from safetensors import safe_open
    vcfg = vllm_ctx(a.backend)
    cfgj = json.load(open(f"{a.dir}/config.json"))
    H = cfgj["hidden_size"]; E = cfgj.get("num_local_experts", cfgj.get("num_experts"))
    K = cfgj["num_experts_per_tok"]
    I = cfgj.get("moe_intermediate_size") or cfgj.get("intermediate_size")
    idx = json.load(open(f"{a.dir}/model.safetensors.index.json"))["weight_map"]
    _HD = {}
    def raw(n):
        s = idx[n]
        if s not in _HD:
            _HD[s] = safe_open(f"{a.dir}/{s}", "pt", device="cpu")
        return _HD[s].get_tensor(n)

    Pmoe = f"model.layers.{a.layer}.block_sparse_moe."; Pexp = Pmoe + "experts."
    suffixes = sorted({k.split(f"{Pexp}0.w1.")[1] for k in idx if k.startswith(f"{Pexp}0.w1.")})
    eb = raw(Pmoe + "e_score_correction_bias").float().cuda() if Pmoe + "e_score_correction_bias" in idx else None
    qcfg = build_quant_config(a.dir)
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    kw = dict(num_experts=E, top_k=K, hidden_size=H, intermediate_size=I,
              params_dtype=torch.bfloat16, renormalize=cfgj.get("norm_topk_prob", True),
              use_grouped_topk=False, scoring_func="sigmoid",
              routed_scaling_factor=cfgj.get("routed_scaling_factor", 1.0),
              quant_config=qcfg, prefix=Pexp[:-1])
    if eb is not None:
        kw["e_score_correction_bias"] = eb
    moe = FusedMoE(**kw).cuda()
    params = dict(moe.named_parameters())
    for e in range(E):
        for proj, shard in [("w1", "w1"), ("w3", "w3"), ("w2", "w2")]:
            grp = "w2" if shard == "w2" else "w13"
            for suf in suffixes:
                name = f"{Pexp}{e}.{proj}.{suf}"; pname = f"{grp}_{suf}"
                if name in idx and pname in params:
                    moe.weight_loader(params[pname], raw(name).cuda(), name, shard, e)
    moe.quant_method.process_weights_after_loading(moe)

    # DETERMINISTIC input: CPU generator, fixed seed — identical bytes on every box/arch.
    g = torch.Generator(device="cpu").manual_seed(1234)
    x = (torch.randn(a.tokens, H, generator=g, dtype=torch.float32) * 0.1).to(torch.bfloat16).cuda()
    gate_w = raw(Pmoe + "gate.weight").to(torch.bfloat16).cuda()
    rl = torch.nn.functional.linear(x, gate_w)
    from vllm.forward_context import set_forward_context
    with torch.no_grad(), set_forward_context(None, vcfg):
        out = moe(x, rl)
    torch.save(out.float().cpu(), a.out)
    print(f"dumped {tuple(out.shape)} backend={a.backend} -> {a.out}")


if __name__ == "__main__":
    main()
