"""On-box validation of CUDA-graph EAGLE-aux compatibility (perf/graph-aux) — run on a box with a
layer-range pull (m25_pull_range.py) that INCLUDES an EAGLE aux layer (default 29..31 -> aux 30).

Proves, on N real M2.5 layers:
  1. BIT-EQUALITY: a GraphRunner replay == the eager-MANUAL block (same _GraphState buffers, aux
     captured run_block's way) — h AND every aux buffer EXACTLY equal, at 3 start_pos across 2 buckets.
     (Eager-manual<->graph-manual is the bit-identity claim; vs eager SDPA-flash the graphed decode is
     the same accepted-kernel-numerics class as fp8 wire — the ring A/B judges accept/g.)
  2. FRESHNESS — the stale-aux regression this lever exists to kill: a second replay at a DIFFERENT
     start_pos with the SAME input must publish DIFFERENT aux (RoPE/context move it) that again equals
     eager exactly. A stale buffer would return the previous values verbatim.
  3. OOM-SAFETY: a capture failure marks the bucket permanently eager and falls back to run_block
     (output still correct) — the stage never dies from graph capture.
  4. TIMING: eager run_block vs graph replay per block (synced per call, like the serve loop) — the
     launch-overhead recovery on THIS box's CPU.

  M25_DIR=/root/m25 python research/graph_aux_check.py --layers 29 31
"""
import argparse
import os
import sys
import time

os.environ["M25_EAGLE"] = "1"
os.environ["M25_CUDA_GRAPH"] = "1"                  # forces M25_STATIC_KV; also proves the old SystemExit guard is gone
os.environ.setdefault("M25_DIR", "/root/m25")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase0"))

import torch                                        # noqa: E402
import m25_stage as S                               # noqa: E402


def eager_manual(layers, st, x, vcfg):
    """The reference block: identical math to what the graph captured — manual-matmul attention via a
    _GraphState (S._GR set), aux captured per run_block's contract (aux layer OUTPUT residual, bf16)."""
    from vllm.forward_context import set_forward_context
    pe = S.get_pe()
    aux = {}
    S._GR = st
    try:
        h = x
        with torch.no_grad(), set_forward_context(None, vcfg):
            for L in layers:
                h = L.forward(h, 0, pe)
                if L.li in st.aux:
                    aux[L.li] = h[0].detach().to(torch.bfloat16).clone()
    finally:
        S._GR = None
    return h, aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs=2, default=[29, 31], metavar=("LO", "HI"),
                    help="layer range [LO,HI) — must contain an EAGLE aux layer (1/30/58)")
    ap.add_argument("--s", type=int, default=9, help="verify-block size (K+1)")
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    vcfg = S.vllm_ctx()
    layers = [S.Layer(i) for i in range(a.layers[0], a.layers[1])]
    aux_ids = [L.li for L in layers if L.li in S.EAGLE_AUX_LAYER_IDS]
    assert aux_ids, f"range {a.layers} has no EAGLE aux layer {S.EAGLE_AUX_LAYER_IDS} — pull one that does"
    print(f"loaded layers [{a.layers[0]}:{a.layers[1]}] ({torch.cuda.memory_allocated()/1e9:.2f} GB) "
          f"aux={aux_ids} s={a.s}", flush=True)
    cos, sin = S.get_pe()
    gr = S.GraphRunner(layers, vcfg, a.s)

    # eager-prefill a committed context [0, P) into the static KV so the compared blocks attend real
    # (position-dependent) state, like a serve-loop decode block would
    P = 1400
    torch.manual_seed(0)
    with torch.no_grad():
        S.run_block(layers, 0, (torch.randn(1, P, S.H, device="cuda") * 0.1).to(torch.bfloat16), vcfg)

    # 1+2. bit-equality + freshness. ONE input x reused across start_pos: aux freshness must come from
    # POSITION (RoPE + visible context), never from changing inputs — a stale static buffer would
    # reproduce the previous replay's aux verbatim.
    x = (torch.randn(1, a.s, S.H, device="cuda") * 0.1).to(torch.bfloat16)
    ok = True
    prev_aux = None
    for sp in (P, P + 400, 3000):                   # 1400/1800 -> bucket 2048 (shared graph), 3000 -> 4096
        alen = gr._bucket(sp + a.s)
        st_ref = S._GraphState(a.s, alen, gr.rd, "cuda", aux_ids)
        st_ref.set(sp, cos, sin)
        with torch.no_grad():
            h_e, aux_e = eager_manual(layers, st_ref, x, vcfg)
            h_g = gr.run(sp, x)
        same_h = torch.equal(h_g, h_e)
        for li in aux_ids:                          # published _AUX must ALIAS the graph's static buffer
            assert S._AUX[li] is gr.graphs[alen][2].aux[li], "run() must publish the static aux buffer into _AUX"
        aux_g = {li: S._AUX[li].clone() for li in aux_ids}
        same_aux = all(torch.equal(aux_g[li], aux_e[li]) for li in aux_ids)
        fresh = prev_aux is None or any(not torch.equal(aux_g[li], prev_aux[li]) for li in aux_ids)
        ok &= same_h and same_aux and fresh
        print(f"  start={sp:5} bucket={alen:5}  h graph==eager-manual: {same_h}  "
              f"aux graph==eager-manual: {same_aux}  aux fresh vs prev start: {fresh}", flush=True)
        prev_aux = aux_g
    print(f"captured graphs={S._GRAPH_COUNT} (expect 2: buckets 2048+4096)  skipped={S._GRAPH_SKIPPED}", flush=True)

    # 3. OOM-safety: a failing capture must mark the bucket eager and fall back to run_block
    gr2 = S.GraphRunner(layers, vcfg, a.s + 1)      # fresh runner (different s), nothing captured yet

    def boom(alen):
        raise RuntimeError("synthetic capture failure (OOM-safety probe)")
    gr2._capture = boom
    x2 = (torch.randn(1, a.s + 1, S.H, device="cuda") * 0.1).to(torch.bfloat16)
    with torch.no_grad():
        out = gr2.run(P, x2).clone()
        ref = S.run_block(layers, P, x2, vcfg)
    b2 = gr2._bucket(P + a.s + 1)
    oom_ok = torch.equal(out, ref) and b2 in gr2.eager and not gr2.graphs
    ok &= oom_ok
    print(f"  OOM-safety: fallback == run_block: {torch.equal(out, ref)}  bucket {b2} marked eager: "
          f"{b2 in gr2.eager}  nothing captured: {not gr2.graphs}", flush=True)

    # 4. timing: production-eager (run_block, SDPA path + python aux capture) vs graph replay, synced
    # per block like the serve loop (the reply is consumed/.cpu()'d every block)
    sp = P
    with torch.no_grad():
        for _ in range(5):
            S.run_block(layers, sp, x, vcfg); torch.cuda.synchronize()
            gr.run(sp, x)
        t0 = time.perf_counter()
        for _ in range(a.iters):
            S.run_block(layers, sp, x, vcfg); torch.cuda.synchronize()
        eager_ms = (time.perf_counter() - t0) / a.iters * 1e3
        t0 = time.perf_counter()
        for _ in range(a.iters):
            gr.run(sp, x)                           # syncs internally
        graph_ms = (time.perf_counter() - t0) / a.iters * 1e3
    print(f"timing ({len(layers)} layers, s={a.s}): eager {eager_ms:.2f} ms/block  graph {graph_ms:.2f} "
          f"ms/block  -> {eager_ms / max(graph_ms, 1e-9):.2f}x (stage ~13 layers scales ~linearly)", flush=True)
    print("VERDICT:", "PASS — graph replay bit-identical to eager-manual, aux fresh + position-correct "
          "per replay, OOM fallback safe." if ok else "FAIL — inspect the lines above.", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
