"""How often an n-gram proposal is RIGHT — measured per workload class, on real V4 tokens.

WHY THIS FILE EXISTS AND WHAT IT REFUSES TO DO. The n-gram (prompt-lookup) proposer is the only
tap-free way to speculate past the DSpark block (docs/V4_MULTIBLOCK_VERDICT.md §6.3), so it is the
only lever that can lift the pipelined ring's in-flight cap. But its acceptance is not a property of
the model, it is a property of the PROMPT: it proposes the continuation that followed the last
earlier occurrence of the current suffix, so it wins exactly as much as the output repeats the
context. A benchmark that asks for a corrected version of a function shown in the prompt will
therefore quote a tok/s that does NOT survive contact with "explain this" or "write me a scheduler".

So this measures acceptance on THREE workload classes and never averages them:

    edit      output substantially re-emits context — the agentic-coding edit loop
    novel_code  output is code the prompt never showed (generic boilerplate is all there is to copy)
    novel_prose output is prose reasoning over a code context

and reports each on its own line. The asymmetry between them IS the finding; a blended number
would hide it.

WHAT IS MEASURED, AND THE ONE ASSUMPTION. Speculative decoding accepts a draft when it equals the
TARGET MODEL's greedy token. We have no V4 weights here, so the stand-in for the model's greedy
stream is the real continuation text, tokenized with V4's own tokenizer
(phase0/deepseek_v4_ref/tokenizer.json, 129280 vocab). That is:

  * FAIR for `edit`, where the model's job IS to re-emit the shown text with a change, so its greedy
    stream and the reference text agree almost everywhere;
  * OPTIMISTIC for `novel_*`, because a target model's own greedy output is more self-repetitive
    than human-written reference text, but also PESSIMISTIC in the other direction, because the
    model's continuation is not the one the corpus happens to contain. The two do not cancel and we
    do not claim they do — `novel_*` numbers here are an ESTIMATE, `edit` is close to a measurement.

The quantity reported is the CONDITIONAL per-depth acceptance `q_j` — the probability that the j-th
proposed token is right GIVEN that 1..j-1 were — because that is the only shape the pipeline model
can use (docs/V4_PIPELINE_EFFICIENCY.md §3): the marginal frame of depth is the deepest draft, and
what it costs is set by q at ITS depth, not by an average.

    python3 phase0/v4_ngram_accept.py            # the three classes, per-depth q, at k=12
    python3 phase0/v4_ngram_accept.py --offset 5 # acceptance of a HYBRID's extension frames, which
                                                 # start 5 positions past the frontier (past the
                                                 # DSpark block) rather than at it
"""

import argparse
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ngram_draft import NgramDrafter                                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _vendored(name):
    """Locate a vendored reference tree, in the repo AND on a deployed box.

    The repo keeps upstream trees under vendor/ so the engine directories hold only our code. A
    rented stage does NOT have that layout: deploy copies the modules flat into /root with the
    reference tree beside them, which is the whole reason the engines are flat modules. So try the
    sibling first (deployed), then the repo's vendor/. Resolving only one way silently breaks the
    other, and the box is the one that would not be caught by a test run."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, name),
                 os.path.join(here, os.pardir, os.pardir, "vendor", name)):
        if os.path.isdir(cand):
            return os.path.normpath(cand)
    return os.path.join(here, name)


TOKENIZER = os.path.join(_vendored("deepseek_v4_ref"), "tokenizer.json")


def load_tokenizer(path=None):
    """V4's own tokenizer, vendored in-tree — no network, no HF cache, no download. Falling back to
    a whitespace split would change every number here (n-gram acceptance is a property of the token
    boundaries), so a missing tokenizer is a hard failure rather than a silent proxy."""
    path = path or TOKENIZER
    try:
        from tokenizers import Tokenizer
    except ImportError as e:                                           # pragma: no cover
        raise RuntimeError("v4 ngram-accept needs `tokenizers` to tokenize on V4's own vocabulary; "
                           "a whitespace proxy would not measure the same quantity") from e
    if not os.path.exists(path):                                       # pragma: no cover
        raise RuntimeError(f"no tokenizer at {path!r} — the vendored reference carries it")
    return Tokenizer.from_file(path)


# ── the workloads ──────────────────────────────────────────────────────────────────────────────────
# Real text, on disk, in this repo: a proposer that only ever sees synthetic corpora is measured on
# its own reflection. `edit` mutates a real source file the way a coding agent does (rename a symbol,
# change a constant, drop a branch) and asks for the whole file back — the case n-gram is built for.

def _read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def _edit_text(src, seed=0):
    """A realistic small edit: rename one identifier throughout, change two numeric literals, and
    delete one line. ~3-5% of tokens move; the other 95% is exactly the copy an n-gram lives on."""
    rng = random.Random(seed)
    out = src.replace("max_ext", "max_extend").replace("best_len", "best_run")
    out = out.replace("max_cand=48", "max_cand=64").replace("ng=3", "ng=4")
    lines = out.split("\n")
    drop = rng.randrange(len(lines) // 3, 2 * len(lines) // 3)
    del lines[drop]
    return "\n".join(lines)


def _edit_heavy(src, seed=1):
    """The same task done the way a model actually does it: the rename, plus a NEW guard clause every
    ~18 lines, a reflowed comment, and one block re-indented. ~20% of tokens move and — the part that
    matters — the copy is broken into short runs instead of one long one, which is where a
    prompt-lookup drafter loses its anchor. `edit` and `edit_heavy` bracket the realistic range; a
    verdict that only survives at `edit` is a verdict about verbatim copying."""
    rng = random.Random(seed)
    lines = _edit_text(src, seed).split("\n")
    out = []
    for i, ln in enumerate(lines):
        if i and i % 18 == 0:
            pad = " " * (len(ln) - len(ln.lstrip()))
            out.append(f"{pad}if not {rng.choice(('seq', 'tbl', 'cands', 'k'))}:  # added guard")
            out.append(f"{pad}    return []")
        out.append(ln.replace("# ", "#  ") if ln.lstrip().startswith("# ") else ln)
    return "\n".join(out)


def workloads():
    """(name, prompt_text, output_text) — the output is what a target model would be generating."""
    draft_src = _read("phase0/ngram_draft.py")
    moe_src = _read("engines/deepseek_v4/v4_moe_decode.py")
    verdict = _read("docs/V4_MULTIBLOCK_VERDICT.md")
    prose = "\n".join(l for l in verdict.split("\n") if not l.startswith(("|", "```", "    ")))
    return [
        # (a) EDIT-LIKE: the file is IN the prompt and the answer re-emits it with a small change.
        ("edit",
         "Here is phase0/ngram_draft.py:\n\n" + draft_src +
         "\n\nRename `max_ext` to `max_extend` and `best_len` to `best_run`, widen max_cand to 64, "
         "default ng to 4, and drop the stray line. Reply with the whole corrected file.\n\n",
         _edit_text(draft_src)),
        # (a') THE SAME EDIT, DONE HEAVILY. Same prompt, an answer that inserts as well as renames.
        ("edit_heavy",
         "Here is phase0/ngram_draft.py:\n\n" + draft_src +
         "\n\nRename `max_ext` to `max_extend` and `best_len` to `best_run`, add the missing guard "
         "clauses, and reflow the comments. Reply with the whole corrected file.\n\n",
         _edit_heavy(draft_src)),
        # (b) NOVEL CODE: a different module the prompt never showed. Only language boilerplate
        # (`def `, `self.`, `torch.`, indentation) is available to copy.
        ("novel_code",
         "Here is phase0/ngram_draft.py for context:\n\n" + draft_src +
         "\n\nNow write the MoE decode fast path for the V4 stage, from scratch.\n\n",
         moe_src),
        # (c) NOVEL PROSE: reasoning text over a code context — the "explain / plan / justify" half
        # of an agentic session, where there is nothing to copy at all.
        ("novel_prose",
         "Here is phase0/ngram_draft.py for context:\n\n" + draft_src +
         "\n\nExplain why chained multi-block drafting cannot lift the pipeline's in-flight cap.\n\n",
         prose),
    ]


# ── the measurement ────────────────────────────────────────────────────────────────────────────────

def per_depth_acceptance(prompt_ids, out_ids, k=12, ng=3, offset=0, **dkw):
    """Walk `out_ids` as a real chain-speculating coordinator would and count, at every draft depth,
    how often the proposal was right GIVEN the shallower ones were.

    `offset` models the HYBRID: the n-gram proposer is not extending the committed frontier, it is
    extending past `offset` positions that a trained drafter already proposed and that were (by the
    time an n-gram frame can be judged) accepted. So it conditions on `offset` extra TRUE tokens and
    is scored against the tokens after them. offset=0 is the pure-n-gram case.

    Advance rule is spec-decode's own: commit the accepted prefix plus the one free correction, so
    the number of TRIALS at each depth is what the real loop would produce, not a uniform sweep."""
    d = NgramDrafter(ng=ng, **dkw)
    seq = list(prompt_ids)
    hits = [0] * k
    trials = [0] * k
    runs = []
    i = 0
    n = len(out_ids)
    while i < n - offset - 1:
        seq_now = seq + out_ids[i:i + offset]                          # the drafter block, assumed right
        ds = d.propose(seq_now, k)
        base = i + offset
        acc = 0
        for j in range(k):
            if base + j >= n:
                break
            trials[j] += 1
            if ds[j] == out_ids[base + j]:
                hits[j] += 1
                acc += 1
            else:
                break
        runs.append(acc)
        step = offset + acc + 1                                        # accepted drafts + the free token
        seq.extend(out_ids[i:i + step])
        i += step
    return hits, trials, runs


def q_of(hits, trials):
    return [(h / t if t else float("nan")) for h, t in zip(hits, trials)]


def trace_workloads(path, min_gen=20):
    """Grouped real greedy traces: `{arm, prompt_ids, gen_ids}` per line, as dumped from a live ring.

    Strictly better evidence than the text workloads above where it exists, because `gen_ids` IS a
    target model's own greedy stream — the exact thing a draft is accepted against — rather than a
    corpus that stands in for one. What it cannot cover is the EDIT case: these prompts are short
    instructions (median ~71 tokens), so there is no pasted file to copy and the arms labelled `code`
    are novel generation. The two sources are complementary and are reported apart."""
    groups = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if len(r.get("gen_ids") or []) >= min_gen:
                groups.setdefault(r["arm"], []).append((r["prompt_ids"], r["gen_ids"]))
    return groups


def _accumulate(pairs, k, ng, offset, limit, dkw):
    hits, trials, runs, ptok = [0] * k, [0] * k, [], 0
    for p_ids, o_ids in pairs:
        h, t, ru = per_depth_acceptance(p_ids, o_ids[:limit], k=k, ng=ng, offset=offset, **dkw)
        for j in range(k):
            hits[j] += h[j]
            trials[j] += t[j]
        runs.extend(ru)
        ptok += len(p_ids)
    return hits, trials, runs, ptok


def report(k=12, ng=3, offset=0, limit=6000, tok_path=None, as_json=False, traces=None, margin=None):
    dkw = {} if margin is None else {"margin": margin}
    if traces:
        groups = trace_workloads(traces)
        src = [(arm, pairs) for arm, pairs in sorted(groups.items())]
    else:
        tk = load_tokenizer(tok_path)
        src = [(name, [(tk.encode(p).ids, tk.encode(o).ids)]) for name, p, o in workloads()]
    rows = []
    for name, pairs in src:
        hits, trials, runs, ptok = _accumulate(pairs, k, ng, offset, limit, dkw)
        q = q_of(hits, trials)
        rows.append({"workload": name, "prompt_tokens": ptok, "output_tokens": trials[0],
                     "q": [round(x, 4) for x in q], "hits": hits, "trials": trials,
                     "mean_run": round(statistics.fmean(runs), 3) if runs else 0.0,
                     "g": round(statistics.fmean(runs) + 1, 3) if runs else 1.0})
    if as_json:
        print(json.dumps({"k": k, "ng": ng, "offset": offset, "rows": rows}, indent=2))
        return rows
    print(f"\n=== n-gram acceptance on V4 tokens (ng={ng}, k={k}, offset={offset}) ===")
    print("  offset>0 scores the frames a HYBRID adds PAST a trained drafter's block\n")
    head = "  {:<12}{:>7}{:>7}  ".format("workload", "ptoks", "otoks")
    print(head + "".join(f"{'q' + str(j + 1):>7}" for j in range(min(k, 8))) + f"{'g':>8}")
    for r in rows:
        line = "  {:<12}{:>7}{:>7}  ".format(r["workload"], r["prompt_tokens"], r["output_tokens"])
        line += "".join(f"{r['q'][j]:>7.3f}" for j in range(min(k, 8)))
        print(line + f"{r['g']:>8.2f}")
    print("\n  q_j = P(draft j is right | drafts 1..j-1 were).  g = mean committed tokens per round.")
    print("  `edit` is close to a measurement; `novel_*` are estimates — see the module docstring.")
    return rows


def main():
    ap = argparse.ArgumentParser(description="per-depth n-gram acceptance, by workload class")
    ap.add_argument("--k", type=int, default=12, help="proposal length (max depth scored)")
    ap.add_argument("--ng", type=int, default=3, help="n-gram anchor length")
    ap.add_argument("--offset", type=int, default=0,
                    help="score frames this far past the frontier (the hybrid's extension)")
    ap.add_argument("--limit", type=int, default=6000, help="cap output tokens per workload")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--margin", type=int, default=None,
                    help="NgramDrafter.margin; 0 indexes the speculative tail too (offline best case)")
    ap.add_argument("--traces", default=None,
                    help="JSONL of real greedy traces {arm, prompt_ids, gen_ids} to score instead")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    report(k=a.k, ng=a.ng, offset=a.offset, limit=a.limit, tok_path=a.tokenizer, as_json=a.json,
           traces=a.traces, margin=a.margin)


if __name__ == "__main__":
    main()
