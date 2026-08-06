"""h1_bench.py — the H1 draft test orchestrator: MTP vs GLM-4-9B vs GLM-4.7-Flash for GLM-5.2,
swept across context length, structured so the RENTED RIG only ever runs the target.

The split (the whole point — keep metered compute minimal):
  draft argmax[i] and target argmax[i] are INDEPENDENT functions of the same fixed corpus ids, so:
    * `dump`    — serve ONE model (a draft on the beast, or the target on the rig) and write its
                  per-position next-token argmax for every corpus sequence to a portable jsonl.
    * `compare` — OFFLINE: load a target dump + each draft dump, run the validated compare_argmax,
                  aggregate per context length → the AR-draft acceptance table. No GPU, no network.
    * `mtp`     — RIG-only: drive generation over the corpora with vLLM MTP spec-decode enabled and
                  read accepted/draft tokens straight from vLLM's /metrics (the native accept-len).

So: drafts are dumped on the beast ($0); the rig dumps only the target + runs the mtp sweep; the
three-way, context-swept verdict is assembled offline by `compare` + the mtp numbers.

Dump format (one json per line): {"label": "code_8192", "idx": 3, "n": 8191, "argmax": [int,...]}.
`argmax` is extract_next_argmax(prompt_logprobs) — next-token greedy argmax per position, len S-1.

CLI:
  dump    --base URL --model NAME --vocab V --corpora f1.jsonl f2.jsonl ... --out dump.jsonl
  compare --target tgt.jsonl --target-vocab V --drafts 9B=d9.jsonl:151329 47F=d47.jsonl:154820 --out report.json
  mtp     --base URL --model NAME --corpora ... --max-tokens 256 --out mtp.json

Offline: python3 test_h1_bench.py  validates the compare/aggregation logic on mock dumps (no network).
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draft_accept_bench import compare_argmax, acceptance_from_counts, expected_accept_length, load_prompt_id_seqs
from vllm_accept import server_next_argmax

def label_of(path):
    return os.path.splitext(os.path.basename(os.path.expanduser(path)))[0]


# ---------------------------------------------------------------------------------------------------
# dump: one model's per-position argmax over the corpora -> portable jsonl (beast for drafts, rig for target)
# ---------------------------------------------------------------------------------------------------
def cmd_dump(args):
    with open(os.path.expanduser(args.out), "w") as out:
        for path in args.corpora:
            label = label_of(path)
            seqs = load_prompt_id_seqs(path, args.n or 10**9, args.max_len or None)
            print(f"[dump] {label}: {len(seqs)} seqs", flush=True)
            for idx, ids in enumerate(seqs):
                t0 = time.time()
                am = server_next_argmax(args.base, args.model, ids, args.vocab)
                out.write(json.dumps({"label": label, "idx": idx, "n": len(am), "argmax": am}) + "\n")
                out.flush()
                if (idx + 1) % 5 == 0 or idx == len(seqs) - 1:
                    print(f"  [{label}] {idx+1}/{len(seqs)} (last {time.time()-t0:.1f}s, len={len(ids)})", flush=True)
    print(f"[dump] wrote {args.out}", flush=True)


# ---------------------------------------------------------------------------------------------------
# compare (OFFLINE): target dump vs each draft dump -> per-context acceptance. PURE — unit-tested.
# ---------------------------------------------------------------------------------------------------
def load_dump(path):
    """-> {(label, idx): argmax_list}"""
    d = {}
    with open(os.path.expanduser(path)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            d[(r["label"], r["idx"])] = r["argmax"]
    return d


def compare_dumps(target_dump, draft_dumps, target_vocab):
    """target_dump: {(label,idx):argmax}. draft_dumps: {name: (dump, vocab)}. -> records list."""
    # aggregate per (name, label)
    acc = {}
    for name, (ddump, dvocab) in draft_dumps.items():
        for (label, idx), darg in ddump.items():
            targ = target_dump.get((label, idx))
            if targ is None:
                continue
            c = compare_argmax(darg, targ, dvocab, target_vocab)
            a = acc.setdefault((name, label), {"matches": 0, "n": 0, "oov_target": 0, "oob_draft": 0, "seqs": 0, "vocab": dvocab})
            a["matches"] += c["matches"]; a["n"] += c["n"]
            a["oov_target"] += c["oov_target"]; a["oob_draft"] += c["oob_draft"]; a["seqs"] += 1
    records = []
    for (name, label), a in acc.items():
        p = acceptance_from_counts(a["matches"], a["n"])
        records.append({
            "draft": name, "context": label, "p_accept": round(p, 5),
            "expected_accept_len": round(expected_accept_length(p), 4),
            "n_positions": a["n"], "matches": a["matches"], "n_seqs": a["seqs"],
            "oov_target": a["oov_target"], "oob_draft": a["oob_draft"],
            "draft_vocab": a["vocab"], "target_vocab": target_vocab,
        })
    return records


def cmd_compare(args):
    target = load_dump(args.target)
    drafts = {}
    for spec in args.drafts:
        name, rest = spec.split("=", 1)
        path, vocab = rest.split(":")
        drafts[name] = (load_dump(path), int(vocab))
    records = compare_dumps(target, drafts, args.target_vocab)
    with open(os.path.expanduser(args.out), "w") as f:
        json.dump(records, f, indent=2)
    print_table(records)
    print(f"\n[compare] wrote {args.out}")


def print_table(records):
    by = {}
    for r in records:
        by.setdefault(r["context"], {})[r["draft"]] = r
    order = sorted(by, key=lambda c: int("".join(ch for ch in c if ch.isdigit()) or 0))
    print("\n===== DRAFT ACCEPTANCE vs GLM-5.2 — per-position greedy match, swept by context =====")
    for ctx in order:
        print(f"-- {ctx} --")
        for name, r in sorted(by[ctx].items()):
            note = ""
            if r["oov_target"]:
                pct = 100.0 * r["oov_target"] / r["n_positions"] if r["n_positions"] else 0.0
                note = f"  ({r['oov_target']} OOV = {pct:.2f}%)"
            print(f"   {name:<6} p={r['p_accept']:.4f}  accept_len~{r['expected_accept_len']:.3f}  "
                  f"n={r['n_positions']} ({r['n_seqs']} seqs){note}")


# ---------------------------------------------------------------------------------------------------
# mtp (RIG): drive generation w/ MTP spec-decode on, read accept-len from vLLM /metrics per context.
# ---------------------------------------------------------------------------------------------------
def _spec_metrics(base):
    import urllib.request
    with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=30) as r:
        body = r.read().decode()
    out = {}
    for line in body.splitlines():
        if line.startswith("vllm:spec_decode_num_drafts_total") or \
           line.startswith("vllm:spec_decode_num_draft_tokens_total") or \
           line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            key = line.split("{")[0].split(":")[-1]
            out[key] = float(line.rsplit(" ", 1)[-1])
    return out


def _generate(base, model, ids, max_tokens):
    import urllib.request
    body = json.dumps({"model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        json.loads(r.read())


def cmd_mtp(args):
    results = {}
    for path in args.corpora:
        label = label_of(path)
        seqs = load_prompt_id_seqs(path, args.n or 10**9, args.max_len or None)
        before = _spec_metrics(args.base)
        for ids in seqs:
            _generate(args.base, args.model, ids, args.max_tokens)
        after = _spec_metrics(args.base)
        drafts = after.get("spec_decode_num_drafts_total", 0) - before.get("spec_decode_num_drafts_total", 0)
        dtoks = after.get("spec_decode_num_draft_tokens_total", 0) - before.get("spec_decode_num_draft_tokens_total", 0)
        acc = after.get("spec_decode_num_accepted_tokens_total", 0) - before.get("spec_decode_num_accepted_tokens_total", 0)
        accept_rate = (acc / dtoks) if dtoks else 0.0
        accept_len = 1.0 + (acc / drafts) if drafts else 1.0   # committed tokens per verify step
        results[label] = {"n_seqs": len(seqs), "drafts": drafts, "draft_tokens": dtoks, "accepted": acc,
                          "accept_rate": round(accept_rate, 4), "accept_len": round(accept_len, 4)}
        print(f"[mtp] {label}: accept_rate={accept_rate:.4f} accept_len~{accept_len:.3f} "
              f"(drafts={int(drafts)} accepted={int(acc)})", flush=True)
    with open(os.path.expanduser(args.out), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[mtp] wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description="H1 draft test: MTP vs 9B vs 4.7-Flash for GLM-5.2, context-swept")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump"); d.add_argument("--base", required=True); d.add_argument("--model", required=True)
    d.add_argument("--vocab", type=int, required=True); d.add_argument("--corpora", nargs="+", required=True)
    d.add_argument("--n", type=int, default=0); d.add_argument("--max-len", type=int, default=0); d.add_argument("--out", required=True)
    d.set_defaults(fn=cmd_dump)
    c = sub.add_parser("compare"); c.add_argument("--target", required=True); c.add_argument("--target-vocab", type=int, required=True)
    c.add_argument("--drafts", nargs="+", required=True, help="name=dump.jsonl:VOCAB"); c.add_argument("--out", default="/tmp/h1_report.json")
    c.set_defaults(fn=cmd_compare)
    m = sub.add_parser("mtp"); m.add_argument("--base", required=True); m.add_argument("--model", required=True)
    m.add_argument("--corpora", nargs="+", required=True); m.add_argument("--max-tokens", type=int, default=256)
    m.add_argument("--n", type=int, default=0); m.add_argument("--max-len", type=int, default=0); m.add_argument("--out", default="/tmp/h1_mtp.json")
    m.set_defaults(fn=cmd_mtp)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
