"""draft_accept_bench.py — A/B speculative-draft acceptance: GLM-4.7-Flash vs GLM-4-9B for GLM-5.2.

ONE question, minimal moving parts: which draft gives the higher PER-POSITION GREEDY ACCEPTANCE
against the GLM-5.2 target? Acceptance p = mean over positions of (draft.argmax(pos) == target.argmax(pos))
on the SAME teacher-forced context; expected accept-length under the standard spec-decode geometric
model is ~ 1/(1-p). This DELIBERATELY avoids the spec-decode machinery (no CoordRing/stream, no rollback,
no cudagraph, no MTP, no tree, no autoregressive loop) — it is just forward passes + an argmax compare.

How it works (teacher-forced, ONE forward per sequence per model):
  TARGET (GLM-5.2, NVFP4, 8 GPUs): feed the WHOLE sequence through the loopback relay-back ring as a
    single prefill (start_pos=0). The ring returns the target hidden state for every position; we apply
    GLM-5.2's final norm + lm_head + argmax (the exact math reused from CoordRing.recv_logits) to get the
    target next-token argmax at each position. No generation loop — one traversal yields all positions.
  DRAFTS (GLM-4-9B, GLM-4.7-Flash; each fits on a single GPU): load with AutoModelForCausalLM, run ONE
    teacher-forced forward over the same ids, take logits.argmax(-1) per position.

Acceptance at position i compares the prediction of token (i+1): draft.argmax[i] vs target.argmax[i].
Positions are aligned 1:1 because both see the identical prefix ids[:i+1]. The last position is dropped
(no ground-truth next position to be a "draft" for in the spec sense — it's the open continuation).

Vocab mismatch handling (this is itself part of why matched-vocab 4.7-Flash should win):
  GLM-4.7-Flash shares GLM-5.2's tokenizer (vocab 154880) -> ids compare directly.
  GLM-4-9B is vocab 151552 -> a 9B argmax id can never be >= 151552, and a TARGET argmax id >= 151552
  is a token the 9B literally cannot predict -> counted as a non-match (never a coincidental hit). We
  also report how many target positions are 9B-OOV so the penalty is visible, not hidden.

Ring bring-up reuses glm_capture_1node.launch_stage (target = GLM_DIR NVFP4 across 8 GPUs), exactly the
proven plain-forward path. The acceptance number is obtainable on the FIRST run (no separate gather/
assemble step — we read logits live off the head socket).

Run ON the 8xRTX-6000 box (env /root/vmoe):
  /root/vmoe/bin/python draft_accept_bench.py \
      --short research/eagle3/code_corpus.jsonl \
      --long ~/longctx_eval/code_8192.jsonl ~/longctx_eval/code_32768.jsonl ~/longctx_eval/code_102400.jsonl \
      --drafts 9B=/root/glm4_9b_draft 47F=/root/glm47_flash \
      --n-short 64 --max-short 1024 --out /root/draft_accept.json

Offline (no GPU): python3 test_draft_accept_bench.py   validates the acceptance math + argmax compare +
vocab-mismatch handling on mock logits/ids.
"""
import os, sys, json, time, argparse

# ---------------------------------------------------------------------------------------------------
# PURE acceptance math + argmax-compare + vocab handling. NO torch, NO GPU — importable + unit-tested
# offline. This is the load-bearing logic; everything below it is just I/O + the proven forward drive.
# ---------------------------------------------------------------------------------------------------
def expected_accept_length(p):
    """Standard geometric spec-decode model: E[accepted run before first reject] = 1/(1-p).
    p is the per-position match probability. p==1 -> inf (clamped to a large finite for display)."""
    if p >= 1.0:
        return float("inf")
    if p <= 0.0:
        return 1.0
    return 1.0 / (1.0 - p)


def compare_argmax(draft_argmax, target_argmax, draft_vocab, target_vocab):
    """Per-position greedy acceptance for ONE sequence.

    draft_argmax, target_argmax : equal-length lists of predicted next-token ids (position-aligned;
        caller has already dropped the final open position and truncated to the common length).
    draft_vocab, target_vocab   : vocab sizes, used ONLY to charge vocab-mismatch non-matches.

    Returns dict: matches, n, oov_target (target id the draft's vocab can't represent),
    oob_draft (draft id outside draft vocab — should be 0 for a well-formed argmax; defensive).

    A position is a MATCH iff draft_argmax[i] == target_argmax[i] AND both ids are in-vocab for their
    respective models. A target id >= draft_vocab is a token the draft can never emit -> forced non-match
    (counted in oov_target). This is the honest penalty for a mismatched-vocab draft (GLM-4-9B); a
    matched-vocab draft (GLM-4.7-Flash) incurs zero such forced misses.
    """
    n = min(len(draft_argmax), len(target_argmax))
    matches = 0
    oov_target = 0
    oob_draft = 0
    for i in range(n):
        d = draft_argmax[i]
        t = target_argmax[i]
        if t >= draft_vocab:           # target predicted a token the draft's vocab lacks: can't match
            oov_target += 1
            continue
        if d >= draft_vocab:           # defensive: a draft argmax outside its own vocab is impossible
            oob_draft += 1
            continue
        if d == t:
            matches += 1
    return {"matches": matches, "n": n, "oov_target": oov_target, "oob_draft": oob_draft}


def acceptance_from_counts(matches, n):
    """p = matches / n over all compared positions (across all sequences at a context length)."""
    return (matches / n) if n else 0.0


# ---------------------------------------------------------------------------------------------------
# Corpus loading: short = {conversations|messages} (tokenize w/ target tokenizer) ; long = {prompt_ids}.
# ---------------------------------------------------------------------------------------------------
def load_prompt_id_seqs(path, n, max_len=None):
    """Long-context corpora: each line has pre-tokenized `prompt_ids` (target tokenizer). Returns list[list[int]]."""
    seqs = []
    with open(os.path.expanduser(path)) as f:
        for line in f:
            if len(seqs) >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            ids = rec.get("prompt_ids")
            if not ids or not isinstance(ids, list):
                continue
            ids = list(ids)
            if max_len:
                ids = ids[:max_len]
            if len(ids) >= 4:
                seqs.append(ids)
    return seqs


def load_conversation_seqs(tok, path, n, max_len):
    """Short corpus: {conversations|messages} -> tokenize with the TARGET tokenizer (so target argmax ids
    and the prompt ids are in the target's vocab space). Mirrors glm_capture._tokenize_corpus."""
    seqs = []
    with open(os.path.expanduser(path)) as f:
        for line in f:
            if len(seqs) >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            convs = rec.get("conversations") or rec.get("messages") or []
            msgs = [{"role": c["role"], "content": c["content"]} for c in convs if c.get("content")]
            if len(msgs) < 2:
                continue
            try:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                ids = tok(text)["input_ids"]
                if ids and isinstance(ids[0], (list, tuple)):
                    ids = ids[0]
                ids = list(ids)[:max_len]
            except Exception:
                continue
            if len(ids) >= 4:
                seqs.append(ids)
    return seqs


# ---------------------------------------------------------------------------------------------------
# ON-BOX target/draft forward drivers. Heavy imports (torch/transformers/KV) are deferred inside so the
# module + its pure math import cleanly with NO GPU / NO model present (the offline test relies on this).
# ---------------------------------------------------------------------------------------------------
def target_argmax_per_position(ring_sock, ids, send_msg, recv_msg, embed_w, norm_w, lm_head_w, eps, dev):
    """One prefill traversal of the GLM-5.2 ring -> target next-token argmax for EACH position.
    Reuses CoordRing.recv_logits math exactly: hidden -> RMSNorm(norm_w) -> lm_head -> argmax(-1).
    `ids` is a list[int]; returns list[int] of length len(ids) (argmax at every position)."""
    import torch
    h = torch.nn.functional.embedding(torch.tensor([ids], device=dev), embed_w)   # [1, S, H]
    send_msg(ring_sock, 0, h)                                                      # start_pos=0 resets stage KV
    _, hb, _ = recv_msg(ring_sock)                                                 # relay-back: reply on head sock
    x = hb[0].float()                                                              # [S, H]
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * norm_w
    return (xn.to(torch.bfloat16) @ lm_head_w.t()).float().argmax(-1).tolist()     # [S]


def draft_argmax_per_position(model, ids, dev, draft_vocab):
    """One teacher-forced forward of a single-GPU HF draft -> next-token argmax for EACH position.
    Ids fed to the draft are clamped into the draft's vocab (an id >= draft_vocab is a target-only token
    the draft can't embed; clamp to 0 so the forward runs — the OOV penalty is charged in compare_argmax
    against the TARGET argmax, not here). Returns list[int] of length len(ids)."""
    import torch
    safe = [t if t < draft_vocab else 0 for t in ids]
    with torch.no_grad():
        out = model(input_ids=torch.tensor([safe], device=dev), use_cache=False)
    return out.logits[0].float().argmax(-1).tolist()


def run_onbox(args):
    """The on-box bench. Brings up the ring, loads each draft on its own GPU, and for every sequence at
    every context length computes target argmax (ring) + each draft argmax (HF), then aggregates."""
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import glm_capture_1node as L1
    import glm_swarm_nvfp4_kv as KV
    from glm_swarm_nvfp4_kv import send_msg, recv_msg, dev, eps
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import socket

    # ---- 1) bring up the 8-GPU loopback ring (target GLM-5.2 NVFP4) via the PROVEN launcher ----
    nstages = args.stages
    bl = L1.blocks(nstages)
    print(f"[bench] launching {nstages}-stage loopback ring (target {L1.GLM_DIR})", flush=True)
    for k in range(nstages - 1, -1, -1):                  # tail-first (successors listen before heads dial)
        L1.launch_stage(k, bl[k], nstages)
    for k in range(nstages - 1, -1, -1):
        if not L1.warm(k):
            print(f"[abort] stage{k} failed to warm", flush=True)
            sys.exit(1)
        print(f"  stage{k} OK layers {bl[k][0]}-{bl[k][-1]} (gpu{k})", flush=True)
    print(f"[bench] ring warm; connecting coord -> 127.0.0.1:{L1.BASE}", flush=True)

    # ---- target side: tokenizer + embed/norm/lm_head on the COORD gpu, head socket ----
    tok = AutoTokenizer.from_pretrained(KV.DIR, trust_remote_code=True)
    target_vocab = int(KV.cfg.vocab_size)
    # coord-side weights live on the LAST gpu so they don't fight a draft for stage0's gpu memory
    cdev = f"cuda:{nstages - 1}" if torch.cuda.device_count() > nstages - 1 else dev
    embed_w = KV.raw("model.embed_tokens.weight").to(torch.bfloat16).to(cdev)
    norm_w = KV.raw("model.norm.weight").float().to(cdev)
    lm_head_w = KV.raw("lm_head.weight").to(torch.bfloat16).to(cdev)
    host = "127.0.0.1"
    ring = socket.create_connection((host, L1.BASE), timeout=300)
    ring.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    ring.settimeout(600)

    # ---- 2) build the work list: (length-label, [seqs]) ----
    work = []
    if args.short:
        work.append(("short", load_conversation_seqs(tok, args.short, args.n_short, args.max_short)))
    for lp in args.long:
        label = os.path.splitext(os.path.basename(os.path.expanduser(lp)))[0]   # e.g. code_8192
        work.append((label, load_prompt_id_seqs(lp, args.n_long, args.max_long or None)))
    for label, seqs in work:
        print(f"[bench] corpus {label}: {len(seqs)} seqs", flush=True)

    # ---- 3) load drafts (each on its OWN gpu so they coexist with the ring) ----
    drafts = {}
    # put drafts on the highest gpu indices, away from stage0 (gpu0); they're single-GPU and fit.
    dgpu = max(0, torch.cuda.device_count() - 1)
    for spec in args.drafts:
        name, path = spec.split("=", 1)
        ddev = f"cuda:{dgpu}"
        print(f"[bench] loading draft {name} <- {path} on {ddev}", flush=True)
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, trust_remote_code=True).to(ddev).eval()
        drafts[name] = {"model": m, "dev": ddev, "vocab": int(m.config.vocab_size)}
        dgpu = max(0, dgpu - 1)

    # ---- 4) drive: target argmax (ring) once per seq, then each draft argmax; aggregate counts ----
    # acc[name][label] = {matches, n, oov_target, oob_draft, seqs}
    acc = {name: {} for name in drafts}
    records = []
    for label, seqs in work:
        for name in drafts:
            acc[name].setdefault(label, {"matches": 0, "n": 0, "oov_target": 0, "oob_draft": 0, "seqs": 0})
        for si, ids in enumerate(seqs):
            t0 = time.time()
            tgt = target_argmax_per_position(ring, ids, send_msg, recv_msg, embed_w, norm_w, lm_head_w, eps, cdev)
            # acceptance compares prediction of position i (draft[i] vs target[i]); drop final open position
            tgt_cmp = tgt[:-1]
            for name, dd in drafts.items():
                drf = draft_argmax_per_position(dd["model"], ids, dd["dev"], dd["vocab"])
                drf_cmp = drf[:-1]
                c = compare_argmax(drf_cmp, tgt_cmp, dd["vocab"], target_vocab)
                a = acc[name][label]
                a["matches"] += c["matches"]; a["n"] += c["n"]
                a["oov_target"] += c["oov_target"]; a["oob_draft"] += c["oob_draft"]; a["seqs"] += 1
            if (si + 1) % 5 == 0 or si == len(seqs) - 1:
                print(f"  [{label}] {si + 1}/{len(seqs)} seqs (last {time.time()-t0:.1f}s, len={len(ids)})", flush=True)

    # ---- 5) finalize records + summary ----
    for name in drafts:
        for label, a in acc[name].items():
            p = acceptance_from_counts(a["matches"], a["n"])
            rec = {
                "draft": name, "context": label, "p_accept": round(p, 5),
                "expected_accept_len": round(expected_accept_length(p), 4),
                "n_positions": a["n"], "matches": a["matches"], "n_seqs": a["seqs"],
                "oov_target": a["oov_target"], "oob_draft": a["oob_draft"],
                "draft_vocab": drafts[name]["vocab"], "target_vocab": target_vocab,
            }
            records.append(rec)

    with open(os.path.expanduser(args.out), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\n[bench] wrote {len(records)} records -> {args.out}", flush=True)
    print_summary(records, list(drafts))
    ring.close()


def print_summary(records, draft_names):
    """Plain-text A/B table: GLM-4-9B vs GLM-4.7-Flash at each context length (no box-drawing)."""
    by = {}
    for r in records:
        by.setdefault(r["context"], {})[r["draft"]] = r
    print("\n==================== DRAFT ACCEPTANCE: per-position greedy match vs GLM-5.2 ====================")
    print("metric = fraction of positions where draft.argmax == target.argmax; len ~ 1/(1-p)\n")
    for ctx in by:
        print(f"-- context: {ctx} --")
        for name in draft_names:
            r = by[ctx].get(name)
            if not r:
                continue
            note = ""
            if r["oov_target"]:
                pct = 100.0 * r["oov_target"] / r["n_positions"] if r["n_positions"] else 0.0
                note = f"  ({r['oov_target']} target positions OOV for this draft's vocab = {pct:.2f}%)"
            print(f"   {name:<6} p={r['p_accept']:.4f}  accept_len~{r['expected_accept_len']:.3f}  "
                  f"n={r['n_positions']} positions ({r['n_seqs']} seqs){note}")
        # head-to-head verdict at this length
        present = [name for name in draft_names if by[ctx].get(name)]
        if len(present) == 2:
            a, b = present
            ra, rb = by[ctx][a], by[ctx][b]
            win = a if ra["p_accept"] > rb["p_accept"] else b
            d = abs(ra["p_accept"] - rb["p_accept"])
            print(f"   -> {win} wins at {ctx} by {d:.4f} acceptance\n")
        else:
            print()


def main():
    ap = argparse.ArgumentParser(description="GLM-4.7-Flash vs GLM-4-9B speculative-draft acceptance for GLM-5.2")
    ap.add_argument("--short", default="", help="short corpus jsonl ({conversations}); tokenized w/ target tok")
    ap.add_argument("--long", nargs="*", default=[], help="long-ctx jsonl(s) with prompt_ids (code_8192/32768/102400)")
    ap.add_argument("--drafts", nargs="+", required=True, help="name=path pairs, e.g. 9B=/root/glm4_9b_draft 47F=/root/glm47_flash")
    ap.add_argument("--n-short", type=int, default=64)
    ap.add_argument("--n-long", type=int, default=16)
    ap.add_argument("--max-short", type=int, default=1024)
    ap.add_argument("--max-long", type=int, default=0, help="0 = use full prompt_ids length")
    ap.add_argument("--stages", type=int, default=8)
    ap.add_argument("--out", default="/root/draft_accept.json")
    args = ap.parse_args()
    if not args.short and not args.long:
        ap.error("provide at least one of --short / --long")
    run_onbox(args)


if __name__ == "__main__":
    main()
