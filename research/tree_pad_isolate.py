"""Isolate WHERE padded-vs-unpadded tree divergence enters (gate-2 FAIL triage): for the failing
case shape, compare eager-unpadded vs eager-padded PER LAYER and PER OP (attention output vs
post-MoE output), and PER NODE — is it one node class (leaves? dummy-adjacent?), one op (attn vs
MoE), compounding across layers, or input-magnitude-dependent?

  M25_DIR=/root/m25 python research/tree_pad_isolate.py --layers 29 31
"""
import argparse
import os
import random
import sys

os.environ["M25_TREE"] = "1"
os.environ["M25_CUDA_GRAPH"] = "1"
os.environ.setdefault("M25_BATCH", "4")
os.environ.setdefault("M25_DIR", "/root/m25")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase0"))

import torch                                        # noqa: E402
import m25_stage as S                               # noqa: E402


def rand_tree(n, seed):
    rng = random.Random(seed)
    parents = [-1] + [rng.randint(-1, i - 1) for i in range(1, n)]
    depths = []
    for p in parents:
        depths.append(1 if p < 0 else depths[p] + 1)
    return parents, depths


def seed_row(layers, row, upto, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    for L in layers:
        L.bkc[row, :, :upto] = S._kv_enc((torch.randn(S.NKV, upto, S.HD, generator=g) * 0.1).to(torch.bfloat16).cuda())
        L.bvc[row, :, :upto] = S._kv_enc((torch.randn(S.NKV, upto, S.HD, generator=g) * 0.1).to(torch.bfloat16).cuda())


def stats(a, b, tag):
    d = (a.float() - b.float()).abs()
    c = torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()
    pn = d.amax(dim=-1)[0]                                            # per-node maxabs [n]
    worst = int(pn.argmax().item())
    print(f"  {tag}: cos={c:.6f} maxabs={d.max().item():.3e} worst-node={worst} "
          f"per-node=[{', '.join(f'{v:.2e}' for v in pn.tolist())}]", flush=True)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs=2, default=[29, 31])
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--start", type=int, default=300)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--seed", type=int, default=200)
    a = ap.parse_args()
    vcfg = S.vllm_ctx()
    from vllm.forward_context import set_forward_context
    from tree_spec import build_tree_mask
    layers = [S.Layer(i) for i in range(a.layers[0], a.layers[1])]
    npad = S.M25_TREE_PAD
    n, start, row = a.n, a.start, a.row
    parents, depths = rand_tree(n, a.seed)
    pos_ids = [(start - 1) + d for d in depths]
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    x = (torch.randn(1, n, S.H, generator=g) * 0.1).to(torch.bfloat16).cuda()
    print(f"case n={n} npad={npad} start={start} row={row} parents={parents}", flush=True)

    # ---- arm A: eager UNPADDED, layer by layer, capturing attn-out and mlp-out per layer --------
    seed_row(layers, row, start, seed=1000)
    pos_t = torch.as_tensor(pos_ids, dtype=torch.long, device="cuda")
    m, _ = build_tree_mask(parents, depths, start, n)
    m = m.to(torch.bfloat16).cuda()
    A_attn, A_out = [], []
    h = x
    with torch.no_grad(), set_forward_context(None, vcfg):
        for L in layers:
            at = L.attn_tree_row(L._rms(h, L.in_ln), row, start, pos_t, m)
            h = h + at
            mo = L.mlp(L._rms(h, L.post_ln))
            h = h + mo
            A_attn.append(at.clone()); A_out.append(h.clone())

    # ---- arm B: eager PADDED through _TGraphState statics, same per-layer capture ---------------
    seed_row(layers, row, start, seed=1000)
    gr = S.TreeRowGraphRunner(layers, vcfg, npad)
    alen = gr._bucket(start + n)
    st = S._TGraphState(npad, alen, gr.rd, gr.dv, gr.aux_ids)
    st.set(row, start, n, parents, pos_ids, gr.cos, gr.sin)
    hp = torch.empty(1, npad, S.H, dtype=torch.bfloat16, device="cuda")
    hp[:, :n].copy_(x); hp[:, n:] = hp[:, :1]
    B_attn, B_out = [], []
    S._GR = st
    try:
        with torch.no_grad(), set_forward_context(None, vcfg):
            h = hp
            for L in layers:
                at = L.attn_tree_row(L._rms(h, L.in_ln), 0, 0, None, None)
                h = h + at
                mo = L.mlp(L._rms(h, L.post_ln))
                h = h + mo
                B_attn.append(at.clone()); B_out.append(h.clone())
    finally:
        S._GR = None

    # ---- compare per layer / per op on the REAL nodes -------------------------------------------
    for i, L in enumerate(layers):
        print(f"layer {L.li}:", flush=True)
        stats(A_attn[i], B_attn[i][:, :n], "attn-out")
        stats(A_out[i], B_out[i][:, :n], "block-out")

    # ---- MoE-only isolation: same post-attn hidden through mlp at n vs npad tokens --------------
    hbase = A_out[0].clone()                                          # a realistic hidden, [1,n,H]
    hpad = torch.cat([hbase, hbase[:, :1].expand(1, npad - n, S.H)], 1)
    L = layers[-1]
    with torch.no_grad(), set_forward_context(None, vcfg):
        mo_n = L.mlp(L._rms(hbase, L.post_ln))
        mo_p = L.mlp(L._rms(hpad, L.post_ln))
    stats(mo_n, mo_p[:, :n], "MoE-only n-vs-npad (same input)")

    # ---- attention-only isolation at n vs npad (fresh KV both times) ----------------------------
    seed_row(layers, row, start, seed=1000)
    with torch.no_grad(), set_forward_context(None, vcfg):
        at_n = layers[0].attn_tree_row(layers[0]._rms(x, layers[0].in_ln), row, start, pos_t, m)
    seed_row(layers, row, start, seed=1000)
    st2 = S._TGraphState(npad, alen, gr.rd, gr.dv, gr.aux_ids)
    st2.set(row, start, n, parents, pos_ids, gr.cos, gr.sin)
    S._GR = st2
    try:
        with torch.no_grad(), set_forward_context(None, vcfg):
            at_p = layers[0].attn_tree_row(layers[0]._rms(hp, layers[0].in_ln), 0, 0, None, None)
    finally:
        S._GR = None
    stats(at_n, at_p[:, :n], "attn-only layer0 n-vs-npad (same input)")


if __name__ == "__main__":
    main()
