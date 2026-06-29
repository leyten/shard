"""SINGLE-BOX gate for the batched-serving fix: does batched MoE (ONE grouped GEMM over all B*(K+1)
tokens) cut a STAGE's per-traversal decode GPU time ~B x vs the per-stream loop, across context?

The batched-serving cliff is GPU-bound: per-stream MoE re-reads the 256 NVFP4 expert weights B times
per layer (decode is weight-read-bound). If batched MoE brings per-stage per-traversal GPU well below
the ~75ms WAN hop window at B=4 @16k, the depth-pipelined ring stays WAN-bound -> batched holds. This
measures exactly that on ONE box BEFORE any WAN-ring relaunch (the documented method scar).

Also gates CORRECTNESS: batched vs per-stream output drift per token (expect ~1e-3 schedule artifact,
NOT a bug) + isolation (B identical streams -> identical output rows).

  M25_BATCH=4 M25_KV_MAXLEN=33000 M25_DIR=/root/m25 python -u m25_batched_moe_bench.py --layers 12
  # add M25_KV_FP8=1 to halve the KV buffer (needed for ctx>16k at B>=4 on a 32GB card)
"""
import os, sys, time, argparse, torch
os.environ.setdefault("M25_DIR", "/root/m25")
import m25_stage as S

dev = "cuda"
K = 8; SDEC = K + 1                                  # verify block = K+1 tokens


def _time(fn, iters=8, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0       # ms/call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12, help="real layers to load (a full stage is ~12)")
    ap.add_argument("--lo", type=int, default=24, help="first layer index (mid-model is representative)")
    ap.add_argument("--ctxs", default="512,2048,8192,16384,32000")
    ap.add_argument("--bs", default="1,2,4")
    a = ap.parse_args()
    if S.M25_BATCH < 2:
        sys.exit("set M25_BATCH>=max(bs) in the env (allocates the [B,NKV,MAXLEN,HD] KV buffers)")
    ctxs = [int(x) for x in a.ctxs.split(",")]
    bs = [int(x) for x in a.bs.split(",")]
    Bmax = max(bs)

    vcfg = S.vllm_ctx()
    layers = [S.Layer(i) for i in range(a.lo, a.lo + a.layers)]
    nL = len(layers)
    gb = torch.cuda.memory_allocated() / 1e9
    print(f"loaded {nL} layers [{a.lo}:{a.lo+nL}] ({gb:.1f}GB, {gb/nL:.2f}/layer) "
          f"M25_BATCH={S.M25_BATCH} KV_MAXLEN={S.M25_KV_MAXLEN} KV_FP8={S.M25_KV_FP8}", flush=True)
    torch.manual_seed(0)
    S.get_pe()                                       # warm the rotary table once

    def run(starts, x):                              # one stage decode traversal (nL layers), batched (engine path)
        return S.run_block_decode_b(layers, starts, x, vcfg)

    # ---- 1) CORRECTNESS: batched vs per-stream MoE drift + isolation (B=4, mid context) ----
    print("\n=== correctness (B=4 @ ctx=8192): batched MoE vs per-stream MoE ===", flush=True)
    B = min(4, Bmax); ctx = min(8192, S.M25_KV_MAXLEN - SDEC - 1)
    starts = torch.full((B,), ctx, device=dev, dtype=torch.long)
    x = (torch.randn(B, SDEC, S.H, device=dev) * 0.1).to(torch.bfloat16)
    S.M25_BATCH_MOE = False; out_ps = run(starts.clone(), x.clone())
    S.M25_BATCH_MOE = True;  out_bt = run(starts.clone(), x.clone())
    d = (out_bt.float() - out_ps.float()).abs().max().item()
    rel = d / (out_ps.float().abs().max().item() + 1e-9)
    print(f"  batched-vs-per-stream max|diff| = {d:.3e}  rel = {rel:.2e}  "
          f"({'OK: schedule artifact (<1e-2 rel), per-token MoE correct' if rel < 1e-2 else 'LARGE — inspect'})", flush=True)
    # isolation: 4 identical input rows -> identical output rows under batched MoE
    xi = x[:1].repeat(B, 1, 1)
    oi = run(starts.clone(), xi)
    iso = (oi - oi[:1]).abs().max().item()
    print(f"  isolation (identical rows -> identical out): max|row-row0| = {iso:.3e}  "
          f"({'PASS' if iso < 1e-4 else 'FAIL — cross-stream contamination'})", flush=True)

    # ---- 2) THROUGHPUT: per-stage per-traversal decode ms, per-stream vs batched, across (B,ctx) ----
    print(f"\n=== per-STAGE ({nL}L) decode ms/traversal — per-stream MoE vs batched MoE ===", flush=True)
    print(f"{'ctx':>7} {'B':>2} {'per-stream ms':>14} {'batched ms':>11} {'speedup':>8} {'WAN-bound?':>10}", flush=True)
    WAN_HOP_MS = 75.0                                 # ~one WAN hop; per-stage GPU must stay well under this
    for ctx in ctxs:
        if ctx + SDEC + 1 > S.M25_KV_MAXLEN:
            print(f"{ctx:>7}  -- skip (exceeds KV_MAXLEN {S.M25_KV_MAXLEN}; raise --kv-maxlen or M25_KV_FP8=1)", flush=True)
            continue
        for B in bs:
            starts = torch.full((B,), ctx, device=dev, dtype=torch.long)
            x = (torch.randn(B, SDEC, S.H, device=dev) * 0.1).to(torch.bfloat16)
            try:
                S.M25_BATCH_MOE = False; t_ps = _time(lambda: run(starts.clone(), x.clone()))
                S.M25_BATCH_MOE = True;  t_bt = _time(lambda: run(starts.clone(), x.clone()))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{ctx:>7} {B:>2}  -- OOM", flush=True); continue
            sp = t_ps / max(t_bt, 1e-9)
            wb = "yes" if t_bt < WAN_HOP_MS else "GPU-BOUND"
            print(f"{ctx:>7} {B:>2} {t_ps:>14.2f} {t_bt:>11.2f} {sp:>7.2f}x {wb:>10}", flush=True)
    print("\n[batched-moe-bench] done — gate: batched ms < ~75 (WAN-bound) at B>=4 @16k, speedup ~B", flush=True)


if __name__ == "__main__":
    main()
