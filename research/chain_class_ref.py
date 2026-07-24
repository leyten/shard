"""Precedent-class reference for the gate-2 triage: the SHIPPED solo chain-graph lever changed
verify numerics too (manual bucketed-read attention vs eager SDPA-flash, m25_stage.py:143-162) and
was accepted via ring accept/g. Measure ITS 2-layer compounding on the same box/layers/input scale
as tree_pad_isolate — if the magnitude matches the padded-tree divergence, padding sits in the
already-live accepted numerics class and gate 3 (ring accept/g) is the correct judge, same as then.

  M25_DIR=/root/m25 python research/chain_class_ref.py --layers 29 31
"""
import argparse
import os
import sys

os.environ["M25_EAGLE"] = "1"
os.environ["M25_CUDA_GRAPH"] = "1"                  # forces M25_STATIC_KV (solo kc buffers)
os.environ.setdefault("M25_DIR", "/root/m25")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase0"))

import torch                                        # noqa: E402
import m25_stage as S                               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs=2, default=[29, 31])
    ap.add_argument("--s", type=int, default=7)
    ap.add_argument("--start", type=int, default=300)
    a = ap.parse_args()
    vcfg = S.vllm_ctx()
    layers = [S.Layer(i) for i in range(a.layers[0], a.layers[1])]
    g = torch.Generator(device="cpu").manual_seed(3)
    x = (torch.randn(1, a.s, S.H, generator=g) * 0.1).to(torch.bfloat16).cuda()

    def seed_solo(seed):
        gg = torch.Generator(device="cpu").manual_seed(seed)
        for L in layers:
            L.kc[0, :, :a.start] = (torch.randn(S.NKV, a.start, S.HD, generator=gg) * 0.1).to(torch.bfloat16).cuda()
            L.vc[0, :, :a.start] = (torch.randn(S.NKV, a.start, S.HD, generator=gg) * 0.1).to(torch.bfloat16).cuda()

    seed_solo(2000)
    ref = S.run_block(layers, a.start, x, vcfg).clone()          # eager SDPA-flash chain (master's route)
    seed_solo(2000)
    gr = S.GraphRunner(layers, vcfg, a.s)
    got = gr.run(a.start, x)                                     # graphed manual bucketed chain (shipped lever)
    d = (got.float() - ref.float()).abs()
    c = torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0).item()
    print(f"SHIPPED chain-graph class, {len(layers)} layers, s={a.s} start={a.start}: "
          f"cos={c:.6f} maxabs={d.max().item():.3e} "
          f"per-tok=[{', '.join(f'{v:.2e}' for v in d.amax(dim=-1)[0].tolist())}]", flush=True)


if __name__ == "__main__":
    main()
