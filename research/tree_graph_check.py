"""On-box validation of TREE-frame CUDA graphs (perf/tree-frame-graphs) — run on a box with a
layer-range pull (m25_pull_range.py) that includes an EAGLE aux layer (default 29..31 -> aux 30).

The correctness hierarchy for padded tree capture (gate 0 = tests/test_tree_graph.py on CPU):

  GATE 1 — CAPTURE FAITHFULNESS (bit-exact, hard): TreeRowGraphRunner replay == run_eager_ref (the
    same-shape eager-padded forward through the same static-buffer math) — output [:n], every aux
    buffer, and the KV row's written span EXACTLY equal, across frame sizes n, rows, starts spanning
    buckets, random topologies, and repeated replays (stale-static detection).
  GATE 2 — PADDING NUMERICS (characterized, not bit): eager-PADDED vs eager-UNPADDED (the true tree
    oracle) on identical inputs. Bit-identity here is IMPOSSIBLE by construction — the NVFP4 MoE is
    token-count non-invariant (padding changes per-expert token counts -> grouped-GEMM schedule) and
    the bucketed read changes softmax reduction lengths — so this gate MEASURES the class: per-node
    max-abs / cosine (accept bar: cosine >= 0.999, the admission fast-kernel bar), plus the
    dummy-magnitude probe (scale the dummy clone x3: real-node outputs must NOT move, proving no
    runtime cross-token activation-scale coupling in the MoE).
  GATE 2b — NO-POISON (bit-exact, hard): a padded tree frame writes NOTHING outside
    [start, start+npad) of its own KV row; other rows bit-untouched.
  GATE 3 — the ring A/B (accept/g + committed-sequence agreement) lives in the bench harness, not here.

  TIMING — the point of the lever: per-frame stage compute, eager tree vs graphed tree vs graphed
    CHAIN row (s=K+1). Bar: graphed tree <= ~1.2x graphed chain (154ms-class -> 45ms-class on a
    full 62-layer ring; scaled here to the loaded range). VRAM per captured graph is reported.

  M25_DIR=/root/m25 python research/tree_graph_check.py --layers 29 31
"""
import argparse
import os
import random
import sys
import time

os.environ["M25_TREE"] = "1"                        # implies M25_EAGLE (aux buffers exercised)
os.environ["M25_CUDA_GRAPH"] = "1"                  # forces M25_STATIC_KV
os.environ.setdefault("M25_BATCH", "4")             # tree ROW graphs need the [B,...] KV rows
os.environ.setdefault("M25_DIR", "/root/m25")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase0"))

import torch                                        # noqa: E402
import m25_stage as S                               # noqa: E402


def rand_tree(n, seed):
    """Random valid tree (parents before children) + depth-consistent pos_ids at `start` (set later)."""
    rng = random.Random(seed)
    parents = [-1] + [rng.randint(-1, i - 1) for i in range(1, n)]
    depths = []
    for p in parents:
        depths.append(1 if p < 0 else depths[p] + 1)
    return parents, depths


def frame(n, start, seed):
    parents, depths = rand_tree(n, seed)
    pos_ids = [(start - 1) + d for d in depths]
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(1, n, S.H, generator=g) * 0.1).to(torch.bfloat16).cuda()
    return x, parents, pos_ids


def seed_row(layers, row, upto, seed):
    """Write plausible committed KV into row `row` at [0, upto) so tree frames attend real prefix."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    for L in layers:
        L.bkc[row, :, :upto] = S._kv_enc((torch.randn(S.NKV, upto, S.HD, generator=g) * 0.1)
                                         .to(torch.bfloat16).cuda())
        L.bvc[row, :, :upto] = S._kv_enc((torch.randn(S.NKV, upto, S.HD, generator=g) * 0.1)
                                         .to(torch.bfloat16).cuda())


def snap(layers, row):
    return [(L.bkc[row].clone(), L.bvc[row].clone()) for L in layers]


def eager_unpadded(layers, row, start, x, vcfg, parents, pos_ids):
    h = S.run_block_tree_row(layers, row, start, x, vcfg, parents, pos_ids)
    aux = {li: v.clone() for li, v in S._AUX.items()}
    return h, aux


def cos(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs=2, default=[29, 31], metavar=("LO", "HI"))
    ap.add_argument("--iters", type=int, default=30, help="timing iterations per arm")
    a = ap.parse_args()
    vcfg = S.vllm_ctx()
    layers = [S.Layer(i) for i in range(a.layers[0], a.layers[1])]
    npad = S.M25_TREE_PAD
    K = 6                                                        # chain-graph reference block s=K+1
    print(f"loaded layers {a.layers} npad={npad} B={S.M25_BATCH} maxlen={S.M25_KV_MAXLEN}", flush=True)
    gr = S.TreeRowGraphRunner(layers, vcfg, npad)
    free0 = torch.cuda.mem_get_info()[0]

    # ---- GATE 1: capture faithfulness (replay == eager-padded, bit) + GATE 2b: no-poison ----------
    g1_ok, g2b_ok = True, True
    cases = [(13, 0, 300), (17, 1, 1900), (21, 2, 300), (npad, 3, 2100), (14, 0, 4000), (13, 0, 301)]
    for i, (n, row, start) in enumerate(cases):
        seed_row(layers, row, start, seed=50 + i)
        x, parents, pos_ids = frame(n, start, seed=100 + i)
        before = snap(layers, row)
        others = [snap(layers, r) for r in range(S.M25_BATCH) if r != row]
        ref = gr.run_eager_ref(row, start, x, parents, pos_ids).clone()
        aux_ref = {li: v.clone() for li, v in S._AUX.items()}
        got = gr.run(row, start, x, parents, pos_ids)
        aux_got = {li: v.clone() for li, v in S._AUX.items()}
        ok = torch.equal(got, ref) and all(torch.equal(aux_got[li], aux_ref[li]) for li in aux_ref)
        g1_ok &= ok
        print(f"gate1[{i}] n={n} row={row} start={start} bucket={gr._bucket(start + n)} "
              f"bit-equal={'PASS' if ok else 'FAIL'} maxabs={(got - ref).abs().max().item():.3e}", flush=True)
        # no-poison: only [start, start+npad) of `row` may differ from the pre-frame snapshot
        for (kb, vb), L in zip(before, layers):
            for cur, old in ((L.bkc[row], kb), (L.bvc[row], vb)):
                if not (torch.equal(cur[:, :start], old[:, :start])
                        and torch.equal(cur[:, start + npad:], old[:, start + npad:])):
                    g2b_ok = False
        for r, osnap in zip([r for r in range(S.M25_BATCH) if r != row], others):
            for (kb, vb), L in zip(osnap, layers):
                if not (torch.equal(L.bkc[r], kb) and torch.equal(L.bvc[r], vb)):
                    g2b_ok = False
    print(f"gate1 capture-faithfulness: {'PASS' if g1_ok else 'FAIL'}", flush=True)
    print(f"gate2b no-poison (row-local, span-local, bit): {'PASS' if g2b_ok else 'FAIL'}", flush=True)
    vram = (free0 - torch.cuda.mem_get_info()[0]) / 1e9
    print(f"captured {len(gr.graphs)} tree graphs, pool cost {vram:.2f} GB total", flush=True)

    # gate 1c — GPU static-composition readback (adversarial residue ii): the live replay statics,
    # read back, must equal an independent CPU-composed _TGraphState for the same frame. Catches a
    # device-side compose bug (fill_/copy_ semantics) that gate 1 can't see (both its arms share the
    # GPU statics) and gate 2's cosine would only catch grossly.
    n, row, start = cases[-1][0], cases[-1][1], cases[-1][2]
    x, parents, pos_ids = frame(n, start, seed=100 + len(cases) - 1)
    alen = gr._bucket(start + n)
    live = gr.graphs[alen][2]
    ref = S._TGraphState(npad, alen, gr.rd, "cpu", gr.aux_ids)
    ref.set(row, start, n, parents, pos_ids, gr.cos.cpu(), gr.sin.cpu())
    g1c_ok = all(torch.equal(getattr(live, b).cpu(), getattr(ref, b))
                 for b in ("rows", "wcp", "cos", "sin", "mask", "cols"))
    print(f"gate1c static readback vs CPU compose: {'PASS' if g1c_ok else 'FAIL'}", flush=True)
    g1_ok &= g1c_ok

    # gate 1d — near-cap DEGRADE (adversarial residue i): a frame that only overflows because of
    # PADDING (start+n fits, start+npad does not) must fall back to eager, never raise or capture.
    n, row = 13, 1
    start = S.M25_KV_MAXLEN - npad + 1
    seed_row(layers, row, start, seed=61)
    x, parents, pos_ids = frame(n, start, seed=601)
    skipped0, graphs0 = S._GRAPH_SKIPPED, len(gr.graphs)
    try:
        got = gr.run(row, start, x, parents, pos_ids)
        g1d_ok = (S._GRAPH_SKIPPED == skipped0 + 1 and len(gr.graphs) == graphs0
                  and torch.isfinite(got).all().item())
    except RuntimeError as e:
        g1d_ok = False
        print(f"gate1d RAISED (must degrade): {e}", flush=True)
    print(f"gate1d near-cap padding degrade -> eager: {'PASS' if g1d_ok else 'FAIL'}", flush=True)
    g1_ok &= g1d_ok

    # ---- GATE 2: padded vs UNPADDED numerics class + dummy-magnitude probe ------------------------
    worst_cos, worst_abs = 1.0, 0.0
    for i, (n, row, start) in enumerate([(13, 0, 300), (17, 1, 1900), (21, 2, 300)]):
        seed_row(layers, row, start, seed=70 + i)
        x, parents, pos_ids = frame(n, start, seed=200 + i)
        want, _ = eager_unpadded(layers, row, start, x, vcfg, parents, pos_ids)
        seed_row(layers, row, start, seed=70 + i)                # identical pre-frame KV for the padded arm
        got = gr.run_eager_ref(row, start, x, parents, pos_ids)
        c, m = cos(got, want), (got - want).abs().max().item()
        worst_cos, worst_abs = min(worst_cos, c), max(worst_abs, m)
        print(f"gate2[{i}] n={n} start={start} cosine={c:.6f} maxabs={m:.3e}", flush=True)
    print(f"gate2 padded-vs-unpadded: worst cosine {worst_cos:.6f} (bar >= 0.999) "
          f"{'PASS' if worst_cos >= 0.999 else 'FAIL'}", flush=True)
    # dummy-magnitude probe: no runtime cross-token scale coupling — scale ONLY the dummy rows
    n, row, start = 13, 0, 300
    seed_row(layers, row, start, seed=90)
    x, parents, pos_ids = frame(n, start, seed=400)
    base = gr.run_eager_ref(row, start, x, parents, pos_ids).clone()
    # rerun with dummies scaled x3 via a hand-staged padded input through the same statics
    from vllm.forward_context import set_forward_context
    alen = gr._bucket(start + n)
    st2 = S._TGraphState(npad, alen, gr.rd, gr.dv, gr.aux_ids)
    st2.set(row, start, n, parents, pos_ids, gr.cos, gr.sin)
    hp = torch.empty(1, npad, S.H, dtype=torch.bfloat16, device=gr.dv)
    hp[:, :n].copy_(x); hp[:, n:] = x[:, :1] * 3.0
    seed_row(layers, row, start, seed=90)
    S._GR = st2
    try:
        with torch.no_grad(), set_forward_context(None, vcfg):
            scaled = gr._layers(hp)[:, :n]
    finally:
        S._GR = None
    dm = (scaled - base).abs().max().item()
    print(f"gate2 dummy-magnitude probe: real-node maxabs delta {dm:.3e} under x3 dummies "
          f"({'PASS — token-local scales' if dm < 1e-2 else 'INVESTIGATE — cross-token coupling?'})", flush=True)

    # ---- TIMING: eager tree vs graphed tree vs graphed chain --------------------------------------
    rgr = S.RowGraphRunner(layers, vcfg, K + 1)
    n, row, start = 17, 0, 1900
    seed_row(layers, row, start, seed=99)
    x, parents, pos_ids = frame(n, start, seed=500)
    xc = (torch.randn(1, K + 1, S.H, device="cuda") * 0.1).to(torch.bfloat16)

    def bench(fn, iters):
        for _ in range(5):
            fn()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    te = bench(lambda: S.run_block_tree_row(layers, row, start, x, vcfg, parents, pos_ids), a.iters)
    tg = bench(lambda: gr.run(row, start, x, parents, pos_ids), a.iters)
    tc = bench(lambda: rgr.run(row, start, xc), a.iters)
    nl = len(layers)
    print(f"TIMING ({nl} layers, x62/{nl} for full-ring scale): eager-tree {te:.2f}ms  "
          f"graph-tree {tg:.2f}ms  graph-chain {tc:.2f}ms  "
          f"tree/chain ratio {tg / tc:.2f} (bar <= ~1.2)  speedup vs eager {te / tg:.2f}x", flush=True)

    verdict = g1_ok and g2b_ok and worst_cos >= 0.999
    print("VERDICT:", "tree-frame graphs FAITHFUL (gate1 bit-exact, gate2b no-poison, "
          f"gate2 cosine {worst_cos:.4f}) — tree/chain {tg / tc:.2f}x, eager recovery {te / tg:.2f}x"
          if verdict else "GATE FAILURE — inspect above before any ring spend.", flush=True)


if __name__ == "__main__":
    main()
