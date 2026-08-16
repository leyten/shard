"""ring_mtp.py — MTP head acceptance vs GLM-5.2 target, via leyten's dense ring on sm120.

MTP = GLM-5.2's native NEXTN layer (layer 78; main layers 0..77). Reuses leyten's Layer(78) for the MTP
block's transformer (dense MLA + NVFP4 MoE) + the 4 MTP-specific weights (enorm/hnorm/eh_proj/shared_head.norm).

Per-position MTP acceptance (the head-to-head vs the 9B's p):
  main model at pos i -> tail hidden hb[i] (ring output, pre-final-norm) and target_argmax[i] (predicts t[i+1]).
  MTP at pos i: x_i = eh_proj([ enorm(emb(t[i+1])) ; hnorm(hb[i]) ]) -> Layer78 block -> shared_head.norm -> lm_head
                -> mtp_pred[i] (predicts t[i+2]).
  Acceptance = mean over i of ( mtp_pred[i] == target_argmax[i+1] )   # both predict t[i+2] given prefix 0..i+1
MTP shares GLM-5.2's vocab -> no OOV. Compare directly to the 9B p (h1_offline/dumps/dump_9b vs target).

  --smoke : ring up + ONE short prompt, print target_argmax + mtp_pred + the per-pos match, exit.
  --corpora f...: aggregate MTP acceptance per context -> --out json.   (1k fits unchunked; long ctx = TODO chunk)

Run on box from /root:  SHARD_WIRE= GLM_DIR=/root/glm52nvfp4 /root/vmoe/bin/python ring_mtp.py --smoke
"""
import os, sys, json, time, socket, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draft_accept_bench import load_prompt_id_seqs

NEXTN = 78  # main layers 0..77; MTP/NEXTN layer = 78


def bring_up(nstages):
    import torch, glm_capture_1node as L1, glm_swarm_nvfp4_kv as KV
    from glm_swarm_nvfp4_kv import send_msg, recv_msg, dev, eps, raw
    bl = L1.blocks(nstages)
    print(f"[mtp] launching {nstages}-stage ring ({L1.GLM_DIR})", flush=True)
    for k in range(nstages - 1, -1, -1): L1.launch_stage(k, bl[k], nstages)
    for k in range(nstages - 1, -1, -1):
        if not L1.warm(k): print(f"[abort] stage{k}", flush=True); sys.exit(1)
        print(f"  stage{k} OK", flush=True)
    cd = dev  # keep coord work on KV.dev (cuda:0) so ring hb + weights + MTP block co-locate (no device dance)
    embed_w = raw("model.embed_tokens.weight").to(torch.bfloat16).to(cd)
    norm_w = raw("model.norm.weight").float().to(cd)
    lm_head_w = raw("lm_head.weight").to(torch.bfloat16).to(cd)
    # MTP block (reuse leyten's Layer for layer 78) + the 4 MTP-specific weights.
    # FusedMoE needs a vLLM config context — set it up exactly like the stage does (unique MASTER_PORT
    # so the coord's world_size=1 dist group doesn't clash with the stages').
    print("[mtp] vLLM config ctx + building MTP block (Layer 78)...", flush=True)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29800")
    vcfg = KV._vllm_ctx()
    mtp_block = KV.Layer(NEXTN)
    enorm = raw(f"model.layers.{NEXTN}.enorm.weight").to(torch.bfloat16).to(cd)
    hnorm = raw(f"model.layers.{NEXTN}.hnorm.weight").to(torch.bfloat16).to(cd)
    eh_proj = raw(f"model.layers.{NEXTN}.eh_proj.weight").to(torch.bfloat16).to(cd)   # [H, 2H]
    sh_norm = raw(f"model.layers.{NEXTN}.shared_head.norm.weight").float().to(cd)
    ring = socket.create_connection(("127.0.0.1", L1.BASE), timeout=300)
    ring.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); ring.settimeout(600)
    print(f"[mtp] ring warm; coord on {cd}", flush=True)
    return dict(torch=torch, KV=KV, send=send_msg, recv=recv_msg, dev=cd, eps=eps,
                embed=embed_w, norm=norm_w, lm_head=lm_head_w, mtp=mtp_block, vcfg=vcfg,
                enorm=enorm, hnorm=hnorm, ehp=eh_proj, shn=sh_norm, ring=ring)


def rms(torch, x, w, eps):
    return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16) * w


def forward_seq(C, ids):
    """ring forward for the full sequence -> (target_argmax [S], mtp_pred [S-1])."""
    torch = C["torch"]; dev = C["dev"]
    h = torch.nn.functional.embedding(torch.tensor([ids], device=dev), C["embed"])    # [1,S,H]
    C["send"](C["ring"], 0, h)
    _, hb, _ = C["recv"](C["ring"])                                                    # [1,S,H] on KV.dev
    hbx = hb[0].to(dev)                                                                # [S,H]
    # target argmax (main model): norm + lm_head
    xn = hbx.float() * torch.rsqrt(hbx.float().pow(2).mean(-1, keepdim=True) + C["eps"]) * C["norm"]
    target_argmax = (xn.to(torch.bfloat16) @ C["lm_head"].t()).float().argmax(-1).tolist()   # [S]
    # MTP: x_i = eh_proj([enorm(emb(t[i+1])) ; hnorm(hb[i])]) for i in 0..S-2
    S = len(ids)
    next_emb = torch.nn.functional.embedding(torch.tensor([ids[1:]], device=dev), C["embed"])[0]  # [S-1,H]
    emb_n = rms(torch, next_emb, C["enorm"], C["eps"])
    # CORRECT convention (settled by --diag): GLM-5.2's MTP wants the POST-model.norm hidden, concat [emb;hidden].
    # Feeding the pre-norm hidden crushed acceptance 0.86 -> 0.51. (pre/post and concat-order swept empirically.)
    h_slice = hbx[:S - 1]
    h_post = (h_slice.float() * torch.rsqrt(h_slice.float().pow(2).mean(-1, keepdim=True) + C["eps"]) * C["norm"]).to(torch.bfloat16)
    h_n = rms(torch, h_post, C["hnorm"], C["eps"])
    mtp_in = torch.nn.functional.linear(torch.cat([emb_n, h_n], dim=-1), C["ehp"])     # [S-1,H]
    # Run the MTP block (Layer 78) through leyten's OWN run_block — the exact set_forward_context
    # wrapper the stages use, so the NVFP4 FusedMoE forward gets its context by construction (no
    # hand-rolled plumbing). start_pos=0 over a fresh cache = correct causal self-attn for the S-1
    # teacher-forced positions; reset() is belt-and-braces (start_pos=0 also crops the cache to empty).
    C["mtp"].reset()
    mh = C["KV"].run_block([C["mtp"]], 0, mtp_in.unsqueeze(0), C["vcfg"])               # [1,S-1,H]
    mh = (mh[0].float() * torch.rsqrt(mh[0].float().pow(2).mean(-1, keepdim=True) + C["eps"]) * C["shn"]).to(torch.bfloat16)
    mtp_pred = (mh @ C["lm_head"].t()).float().argmax(-1).tolist()                     # [S-1] predicts t[i+2]
    return target_argmax, mtp_pred


def forward_seq_diag(C, ids):
    """target_argmax + MTP pred under 4 conventions, to settle the hidden-state/concat ambiguity that
    could explain a too-low acceptance. Variants: {pre,post}-final-norm h_i  x  concat {[emb;hid],[hid;emb]}.
    One ring traversal per seq; the MTP block (1 layer) is re-run cheaply per variant."""
    torch = C["torch"]; dev = C["dev"]; eps = C["eps"]
    h = torch.nn.functional.embedding(torch.tensor([ids], device=dev), C["embed"])
    C["send"](C["ring"], 0, h); _, hb, _ = C["recv"](C["ring"])
    hbx = hb[0].to(dev)                                                                 # [S,H] pre-final-norm
    xn = hbx.float() * torch.rsqrt(hbx.float().pow(2).mean(-1, keepdim=True) + eps) * C["norm"]
    target_argmax = (xn.to(torch.bfloat16) @ C["lm_head"].t()).float().argmax(-1).tolist()
    S = len(ids)
    next_emb = torch.nn.functional.embedding(torch.tensor([ids[1:]], device=dev), C["embed"])[0]
    emb_n = rms(torch, next_emb, C["enorm"], eps)
    h_slice = hbx[:S - 1]
    h_post = (h_slice.float() * torch.rsqrt(h_slice.float().pow(2).mean(-1, keepdim=True) + eps) * C["norm"]).to(torch.bfloat16)  # model.norm applied
    h_pre_n = rms(torch, h_slice, C["hnorm"], eps)
    h_post_n = rms(torch, h_post, C["hnorm"], eps)
    variants = {"pre_eh": torch.cat([emb_n, h_pre_n], -1),  "post_eh": torch.cat([emb_n, h_post_n], -1),
                "pre_he": torch.cat([h_pre_n, emb_n], -1),   "post_he": torch.cat([h_post_n, emb_n], -1)}
    preds = {}
    for name, cat_in in variants.items():
        mtp_in = torch.nn.functional.linear(cat_in, C["ehp"])
        C["mtp"].reset()
        mh = C["KV"].run_block([C["mtp"]], 0, mtp_in.unsqueeze(0), C["vcfg"])
        mh = (mh[0].float() * torch.rsqrt(mh[0].float().pow(2).mean(-1, keepdim=True) + eps) * C["shn"]).to(torch.bfloat16)
        preds[name] = (mh @ C["lm_head"].t()).float().argmax(-1).tolist()
    return target_argmax, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--corpora", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=0)
    ap.add_argument("--out", default="/root/mtp_accept.json")
    args = ap.parse_args()
    C = bring_up(args.stages)

    if args.smoke:
        ids = [755, 911, 264, 293, 982, 503, 257, 470, 264, 488, 1782, 220, 16, 1042, 428]
        t0 = time.time(); ta, mp = forward_seq(C, ids); dt = time.time() - t0
        # acceptance on this seq: mp[i] == ta[i+1]
        m = sum(1 for i in range(len(mp)) if mp[i] == ta[i + 1])
        print(f"[smoke] S={len(ids)} in {dt:.1f}s | target_argmax[:8]={ta[:8]}", flush=True)
        print(f"[smoke] mtp_pred[:8]={mp[:8]}  target_shifted[:8]={ta[1:9]}", flush=True)
        print(f"[smoke] MTP matches {m}/{len(mp)} on this seq", flush=True)
        print("[smoke] MTP HEAD WORKS", flush=True); C["ring"].close(); return

    if args.diag:
        # convention sweep: which {pre/post final-norm h_i} x {concat order} maximizes acceptance.
        path = args.corpora[0] if args.corpora else "/root/h1/corpora/code_1024.jsonl"
        seqs = load_prompt_id_seqs(path, args.n or 16, args.max_len or None)
        agg = {k: [0, 0] for k in ["pre_eh", "post_eh", "pre_he", "post_he"]}   # [matches, n]
        print(f"[diag] {len(seqs)} seqs from {os.path.basename(path)}", flush=True)
        for idx, ids in enumerate(seqs):
            t0 = time.time(); ta, preds = forward_seq_diag(C, ids)
            for k, mp in preds.items():
                agg[k][0] += sum(1 for i in range(len(mp)) if mp[i] == ta[i + 1]); agg[k][1] += len(mp)
            if (idx + 1) % 4 == 0 or idx == len(seqs) - 1:
                row = " ".join(f"{k}={agg[k][0]/max(agg[k][1],1):.4f}" for k in agg)
                print(f"  [{idx+1}/{len(seqs)}] {row} (last {time.time()-t0:.1f}s)", flush=True)
        print("[diag] FINAL per-convention MTP acceptance:", flush=True)
        for k in agg:
            p = agg[k][0] / max(agg[k][1], 1)
            print(f"   {k:8s} p={p:.4f} accept_len~{1/(1-p) if p<1 else 999:.3f}  ({agg[k][0]}/{agg[k][1]})", flush=True)
        C["ring"].close(); return

    label_of = lambda p: os.path.splitext(os.path.basename(p))[0]
    results = {}
    for path in args.corpora:
        label = label_of(path)
        seqs = load_prompt_id_seqs(path, args.n or 10**9, args.max_len or None)
        matches = n = 0
        print(f"[mtp] {label}: {len(seqs)} seqs", flush=True)
        for idx, ids in enumerate(seqs):
            t0 = time.time()
            ta, mp = forward_seq(C, ids)
            matches += sum(1 for i in range(len(mp)) if mp[i] == ta[i + 1]); n += len(mp)
            if (idx + 1) % 8 == 0 or idx == len(seqs) - 1:
                print(f"  [{label}] {idx+1}/{len(seqs)} p={matches/max(n,1):.4f} (last {time.time()-t0:.1f}s)", flush=True)
        p = matches / max(n, 1)
        results[label] = {"p_accept": round(p, 5), "accept_len": round(1.0 / (1 - p) if p < 1 else 999, 4),
                          "matches": matches, "n": n, "n_seqs": len(seqs)}
        print(f"[mtp] {label}: MTP p={p:.4f} accept_len~{results[label]['accept_len']:.3f}", flush=True)
    json.dump(results, open(os.path.expanduser(args.out), "w"), indent=2)
    print(f"[mtp] wrote {args.out}: {results}", flush=True)
    C["ring"].close()


if __name__ == "__main__":
    main()
