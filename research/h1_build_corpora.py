"""h1_build_corpora.py — build the context-swept eval corpora for the H1 draft test, OFFLINE.

Packs code chunks from research/eagle3/code_corpus.jsonl into fixed-length token sequences using the
GLM-5.2 (target) tokenizer, at 1k / 8k / 32k / 100k. Output: prompt_ids jsonl per length (the format
research/vllm_accept.py:load_prompt_id_seqs reads). No GPU, no model — tokenizer only.

  python h1_build_corpora.py
"""
import json, os
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TOKDIR = os.path.join(HERE, "h1_offline/tok/glm52")          # GLM-5.2 (target) tokenizer, downloaded offline
CORPUS = os.path.join(HERE, "eagle3/code_corpus.jsonl")
OUTDIR = os.path.join(HERE, "h1_offline/corpora")
os.makedirs(OUTDIR, exist_ok=True)

CODE_SRC = ("magicoder", "codefeedback", "codealpaca", "oss", "evol", "code", "feedback")
EXCLUDE = ("ultrachat", "general")
# (length_tokens, n_sequences) — 1k short + the three long-context slices
PLAN = [(1024, 64), (8192, 24), (32768, 16), (102400, 8)]

tok = AutoTokenizer.from_pretrained(TOKDIR, trust_remote_code=True)
print(f"[corpora] GLM-5.2 tokenizer vocab={tok.vocab_size}", flush=True)

def text(d):
    return "\n".join((c.get("content") or "") for c in (d.get("conversations") or []))

chunks = []
for line in open(CORPUS):
    try:
        d = json.loads(line)
    except Exception:
        continue
    src = (d.get("source") or "").lower()
    if any(x in src for x in EXCLUDE):
        continue
    if not any(x in src for x in CODE_SRC):
        continue
    t = text(d).strip()
    if t:
        chunks.append(t)
print(f"[corpora] {len(chunks)} code-source chunks", flush=True)

sep = tok("\n\n# ----\n\n", add_special_tokens=False)["input_ids"]
cids = [tok(c, add_special_tokens=False)["input_ids"] for c in chunks]
total_tok = sum(len(c) for c in cids)
print(f"[corpora] {total_tok} total tokens available", flush=True)

def build(L, n, out):
    seqs, i = [], 0
    while len(seqs) < n and i < len(cids):
        buf = []
        while len(buf) < L and i < len(cids):
            buf += (sep if buf else []) + cids[i]
            i += 1
        if len(buf) >= int(L * 0.8):
            seqs.append({"prompt_ids": buf[:L], "n_tok": min(len(buf), L), "source": "code-packed"})
    with open(out, "w") as f:
        f.write("\n".join(json.dumps(s) for s in seqs) + "\n")
    got = [s["n_tok"] for s in seqs]
    print(f"  {os.path.basename(out)}: {len(seqs)} seqs (target {L}, actual {min(got) if got else 0}-{max(got) if got else 0} tok)")
    return len(seqs)

print("[corpora] building:")
made = {}
for L, n in PLAN:
    made[L] = build(L, n, os.path.join(OUTDIR, f"code_{L}.jsonl"))
short = made.get(1024, 0)
ok = all(made[L] >= 1 for L, _ in PLAN)
print(f"\n[corpora] {'OK' if ok else 'INCOMPLETE'} -> {OUTDIR}  (1k={made[1024]} 8k={made[8192]} 32k={made[32768]} 100k={made[102400]})")
