"""ring_target_dump.py — target (GLM-5.2) per-position argmax via LEYTEN'S DENSE RING on sm120.

The sm120-native path: leyten's stages (glm_swarm_nvfp4_kv) run GLM-5.2 with DENSE MLA (pure-torch
matmul attention, no DSA-sparse kernel) + NVFP4 MoE — which is exactly how the target is served in production
on the RTX PRO 6000. Stock vLLM can't (its sparse-MLA backend is sm100-only). This reuses the
offline-validated target_argmax_per_position + the proven ring bring-up from draft_accept_bench.run_onbox,
and just DUMPS the dense target argmax (compare offline to the already-dumped 9B via h1_bench compare).

  --smoke           : bring up the ring + ONE short prompt, print argmax, exit (de-risk before the sweep)
  --corpora f...    : dump target argmax per position -> --out jsonl ({label,idx,n,argmax})

Run on the 8xGPU box (cap4 env):  GLM_DIR=/root/glm52nvfp4 /root/vmoe/bin/python ring_target_dump.py ...
"""
import os, sys, json, time, argparse, socket
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draft_accept_bench import target_argmax_per_position, load_prompt_id_seqs


def bring_up(nstages):
    import torch
    import glm_capture_1node as L1
    import glm_swarm_nvfp4_kv as KV
    from glm_swarm_nvfp4_kv import send_msg, recv_msg, dev, eps
    bl = L1.blocks(nstages)
    print(f"[ring] launching {nstages}-stage loopback ring (target {L1.GLM_DIR})", flush=True)
    for k in range(nstages - 1, -1, -1):                 # tail-first
        L1.launch_stage(k, bl[k], nstages)
    for k in range(nstages - 1, -1, -1):
        if not L1.warm(k):
            print(f"[abort] stage{k} failed to warm", flush=True); sys.exit(1)
        print(f"  stage{k} OK layers {bl[k][0]}-{bl[k][-1]} (gpu{k})", flush=True)
    cdev = f"cuda:{nstages - 1}" if torch.cuda.device_count() > nstages - 1 else dev
    embed_w = KV.raw("model.embed_tokens.weight").to(torch.bfloat16).to(cdev)
    norm_w = KV.raw("model.norm.weight").float().to(cdev)
    lm_head_w = KV.raw("lm_head.weight").to(torch.bfloat16).to(cdev)
    ring = socket.create_connection(("127.0.0.1", L1.BASE), timeout=300)
    ring.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); ring.settimeout(600)
    print(f"[ring] warm; coord connected -> 127.0.0.1:{L1.BASE}", flush=True)
    return ring, send_msg, recv_msg, embed_w, norm_w, lm_head_w, eps, cdev, int(KV.cfg.vocab_size), KV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--corpora", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=0)
    ap.add_argument("--out", default="/root/h1/dump_target_ring.jsonl")
    args = ap.parse_args()
    ring, send_msg, recv_msg, embed_w, norm_w, lm_head_w, eps, cdev, tvocab, KV = bring_up(args.stages)

    if args.smoke:
        ids = KV.tok(  # tokenize a short code prompt with the target tokenizer
            "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n",
            add_special_tokens=False)["input_ids"] if hasattr(KV, "tok") else [755, 911, 264, 293, 982, 503, 257, 470, 264, 488, 1782, 220, 16]
        t0 = time.time()
        am = target_argmax_per_position(ring, ids, send_msg, recv_msg, embed_w, norm_w, lm_head_w, eps, cdev)
        dt = time.time() - t0
        print(f"[smoke] {len(ids)} ids -> {len(am)} argmax in {dt:.1f}s. first 12: {am[:12]}", flush=True)
        print(f"[smoke] vocab={tvocab}; argmax in-range: {all(0 <= a < tvocab for a in am)}", flush=True)
        print("[smoke] RING WORKS (dense target argmax produced)" if am else "[smoke] EMPTY — ring broken", flush=True)
        ring.close(); return

    label_of = lambda p: os.path.splitext(os.path.basename(p))[0]
    with open(os.path.expanduser(args.out), "w") as out:
        for path in args.corpora:
            label = label_of(path)
            seqs = load_prompt_id_seqs(path, args.n or 10**9, args.max_len or None)
            print(f"[ring] {label}: {len(seqs)} seqs", flush=True)
            for idx, ids in enumerate(seqs):
                t0 = time.time()
                tgt = target_argmax_per_position(ring, ids, send_msg, recv_msg, embed_w, norm_w, lm_head_w, eps, cdev)
                am = tgt[:-1]  # drop final open position (compare convention)
                out.write(json.dumps({"label": label, "idx": idx, "n": len(am), "argmax": am}) + "\n"); out.flush()
                if (idx + 1) % 4 == 0 or idx == len(seqs) - 1:
                    print(f"  [{label}] {idx+1}/{len(seqs)} (last {time.time()-t0:.1f}s, len={len(ids)})", flush=True)
    print(f"[ring] wrote {args.out}", flush=True)
    ring.close()


if __name__ == "__main__":
    main()
