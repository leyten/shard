"""ring_long.py — chunked-prefill long-context sweep on leyten's dense sm120 ring.

Why chunked: the one-shot dense prefill builds the full S×S attention score matrix → CUBLAS OOM past ~1k.
The stages ALREADY support incremental prefill via their per-connection MLA KV cache (start_pos>0 appends);
it's how leyten's coord decodes. So we feed each sequence in fixed chunks at increasing start_pos: chunk j
attends [chunk_j, all-cached-prior] (score matrix chunk×total, not S×S). No stage change — coord-side only.

Per seq it produces, in one ring pass:
  - target argmax per position  (-> offline 9B compare via h1_bench)  -> --target-out jsonl
  - MTP acceptance (post-model.norm convention, settled by ring_mtp --diag)  -> --mtp-out json
The leading corpus (1k) re-confirms chunking reproduces the known 0.857 before trusting 8k/32k.

RoPE: $GLM_MAXPOS (set below, pre-import) sizes the rotary table for 100k+ so positions don't index OOB.

Run (box, /root):  SHARD_WIRE= GLM_DIR=/root/glm52nvfp4 /root/vmoe/bin/python ring_long.py \
   --corpora /root/h1/corpora/code_{1024,8192,32768}.jsonl --n 8 --chunk 512
"""
import os, sys, json, time, argparse
os.environ.setdefault("GLM_MAXPOS", "131072")          # MUST precede stage launch (inherited) + coord _get_pe
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ring_mtp import bring_up, rms
from draft_accept_bench import load_prompt_id_seqs


def argmax_logits(x_bf16, lm_head_w, block=4096):
    """Blocked lm_head argmax over positions — avoids materializing the full [S, vocab] logits matrix
    (~20GB at S=32k, vocab~155k) in one allocation. Bounds it to [block, vocab] (~2.5GB)."""
    import torch
    out = []
    for i in range(0, x_bf16.shape[0], block):
        logits = x_bf16[i:i + block] @ lm_head_w.t()                  # [<=block, vocab]
        out.extend(logits.float().argmax(-1).tolist()); del logits
    return out


def forward_chunked(C, ids, chunk):
    """chunked ring forward -> (target_argmax [S], mtp_pred [S-1]). MTP uses the post-final-norm hidden."""
    torch = C["torch"]; dev = C["dev"]; eps = C["eps"]
    S = len(ids)
    h_all = torch.nn.functional.embedding(torch.tensor([ids], device=dev), C["embed"])   # [1,S,H]
    # --- main ring, chunked: chunk j at start_pos -> stage caches grow; collect per-chunk tail hidden ---
    outs = []; pos = 0
    while pos < S:
        end = min(pos + chunk, S)
        C["send"](C["ring"], pos, h_all[:, pos:end])     # start_pos=pos; pos==0 resets the stage caches (new seq)
        _, hb_c, _ = C["recv"](C["ring"])                # [1, end-pos, H]
        outs.append(hb_c[0].to(dev)); pos = end
    hbx = torch.cat(outs, 0)                             # [S,H] tail hidden (pre-final-norm)
    xn = hbx.float() * torch.rsqrt(hbx.float().pow(2).mean(-1, keepdim=True) + eps) * C["norm"]
    target_argmax = argmax_logits(xn.to(torch.bfloat16), C["lm_head"])
    # --- MTP head (post-model.norm hidden, concat [emb;hidden]); the NEXTN block is also run chunked ---
    next_emb = torch.nn.functional.embedding(torch.tensor([ids[1:]], device=dev), C["embed"])[0]
    emb_n = rms(torch, next_emb, C["enorm"], eps)
    h_slice = hbx[:S - 1]
    h_post = (h_slice.float() * torch.rsqrt(h_slice.float().pow(2).mean(-1, keepdim=True) + eps) * C["norm"]).to(torch.bfloat16)
    h_n = rms(torch, h_post, C["hnorm"], eps)
    mtp_in = torch.nn.functional.linear(torch.cat([emb_n, h_n], -1), C["ehp"])           # [S-1,H]
    C["mtp"].reset()
    mh_outs = []; pos = 0; M = S - 1
    while pos < M:
        end = min(pos + chunk, M)
        mh_c = C["KV"].run_block([C["mtp"]], pos, mtp_in[pos:end].unsqueeze(0), C["vcfg"])  # [1, end-pos, H]
        mh_outs.append(mh_c[0]); pos = end
    mh = torch.cat(mh_outs, 0)
    mh = (mh.float() * torch.rsqrt(mh.float().pow(2).mean(-1, keepdim=True) + eps) * C["shn"]).to(torch.bfloat16)
    mtp_pred = argmax_logits(mh, C["lm_head"])                                            # [S-1] predicts t[i+2]
    return target_argmax, mtp_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--corpora", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--target-out", default="/root/dump_target_long.jsonl")
    ap.add_argument("--mtp-out", default="/root/mtp_accept_long.json")
    args = ap.parse_args()
    C = bring_up(args.stages)
    label_of = lambda p: os.path.splitext(os.path.basename(p))[0]
    results = {}
    tout = open(os.path.expanduser(args.target_out), "w")
    for path in args.corpora:
        label = label_of(path)
        seqs = load_prompt_id_seqs(path, args.n or 10**9, None)
        matches = n = 0
        print(f"[long] {label}: {len(seqs)} seqs, chunk={args.chunk}", flush=True)
        for idx, ids in enumerate(seqs):
            t0 = time.time()
            ta, mp = forward_chunked(C, ids, args.chunk)
            tout.write(json.dumps({"label": label, "idx": idx, "n": len(ta) - 1, "argmax": ta[:-1]}) + "\n"); tout.flush()
            matches += sum(1 for i in range(len(mp)) if mp[i] == ta[i + 1]); n += len(mp)
            mem = C["torch"].cuda.max_memory_allocated() / 1e9
            print(f"  [{label}] {idx+1}/{len(seqs)} MTP_p={matches/max(n,1):.4f} ({time.time()-t0:.1f}s, len={len(ids)}, peakGPU0={mem:.1f}GB)", flush=True)
        p = matches / max(n, 1)
        results[label] = {"p_accept": round(p, 5), "accept_len": round(1.0 / (1 - p) if p < 1 else 999, 4),
                          "matches": matches, "n": n, "n_seqs": len(seqs)}
        json.dump(results, open(os.path.expanduser(args.mtp_out), "w"), indent=2)   # write after EACH corpus (survives a kill)
        print(f"[long] {label}: MTP p={p:.4f} accept_len~{results[label]['accept_len']:.3f}", flush=True)
    tout.close()
    print(f"[long] DONE: {results}", flush=True)
    C["ring"].close()


if __name__ == "__main__":
    main()
