# shardv2 — MLA-latent cache, MTP drafting, and a long-context draft study

This is a fork of [leyten/shard](https://github.com/leyten/shard) (Apache-2.0; NOTICE preserved) with three
contributions aimed at the WAN-sharded GLM-5.2 serving stack. Everything here builds on shard's dense
sm120 ring — it does not replace it.

## What's new

### 1. MLA-latent KV cache (the serving unlock) — `research/glm_swarm_nvfp4_kv.py` (`MLA_LATENT=1`)
The dense ring as written caches **decompressed full-head K/V**, so per-stage KV grows ~linearly with
context and tops out around **16–24k context on an 8×96 GB pod** (OOM at 32k regardless of chunk size). This
adds the standard DeepSeek **MLA "absorb"**: cache the ~`kv_lora`-dim latent + the shared rope key, and
absorb `kv_b` into the query (scores) and output projection (values), so full-head K/V is **never
materialized**. **~70× smaller KV cache** on GLM-5.2's real dims (32k: ~88 GB → ~1.3 GB/stage; 100k → ~4 GB).
- Gated by `MLA_LATENT=1` (env). Unset = the original full-head path, untouched — a clean A/B.
- Proven **mathematically equivalent** to the full-head path offline (`research/mla_latent.py`, errors
  ~1e-13) *before* any GPU, then verified on-box: `MLA_LATENT=1` reproduces the full-head 1k/8k numbers and
  clears 32k at ~38 GB on the coordinator (vs 94 GB OOM).
- Also adds `GLM_MAXPOS` so the RoPE table is sized for long context (the table was capped at 4096).
- This is the lever for **both** long context **and** concurrency (the freed KV is the batch budget).

### 2. MTP as the speculative draft (beats a standalone 9B) — `research/ring_mtp.py`
Uses GLM-5.2's native NEXTN (MTP) head as the coordinator's draft and measures per-position greedy
acceptance vs a standalone GLM-4-9B AR draft. Result: **MTP wins at every context and the margin widens** —
0.85/0.88/0.88/0.88 (1k/8k/32k/100k) vs the 9B's 0.79/0.82/0.75 (it decays near its 32k training edge and
can't reach 100k at all). MTP is also cheaper (no separate model; shares embed/lm_head) and co-located.
- `--diag` runs a 4-way convention sweep — GLM-5.2's MTP wants the **post-`model.norm`** hidden, concat
  `[enorm(emb) ; hnorm(post_norm_h)]`. Feeding the pre-norm hidden crushes acceptance 0.86 → 0.51 (a
  plausible-but-wrong number; see the findings report for the methodology warning).

### 3. Chunked teacher-forced prefill — `research/ring_long.py`
Feeds long sequences through the stages' existing KV cache in chunks (each chunk attends `[chunk, cached]`),
so acceptance can be measured at 8k/32k/100k without OOMing the full S×S score matrix. Proven equivalent to
all-at-once offline. Includes blocked lm_head argmax (avoids a `[S, vocab]` ~20 GB logits matrix at long S).

## The findings report
`docs/research/GLM52-draft-and-MLA-findings-for-peers.md` — the full MTP-vs-9B head-to-head across
1k/8k/32k/100k, the MLA mechanism, the convention-bug methodology note, and why the 9B is architecturally
absent past 32k (positional, not memory — and it can't use MLA, having no latent KV). Process detail in
`H1-process-and-results.md`; the test methodology in `H1-test-plan-v2.md`; a roadmap for turning these into
throughput/concurrency in `serving-optimization-plan.md`.

## Reproduce
```
# offline ($0): prove the MLA absorb + chunked-cache equivalence
python research/mla_latent.py
# on an 8×RTX-PRO-6000 (sm120) box with GLM-5.2-NVFP4 + shard's stage ring:
MLA_LATENT=1 GLM_DIR=<model> python research/ring_long.py --corpora <code_*.jsonl> --chunk 256
MLA_LATENT=1 GLM_DIR=<model> python research/ring_mtp.py --diag   # convention sweep
```
`research/h1_build_corpora.py` builds the context-swept corpora; `research/h1_bench.py compare` does the
offline acceptance compare; `research/pod_9b_dump.py` produces the 9B side.

## Relationship to upstream
shardv2 is additive: the MLA path is opt-in (`MLA_LATENT=1`), and shard's original ring/spec/tree code is
unchanged when it's off. Intended to be readable as a diff against leyten/shard and easy to pull from.
