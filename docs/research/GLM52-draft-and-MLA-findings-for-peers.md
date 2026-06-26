# GLM-5.2 speculative drafting + MLA serving — findings report

_Author: Henry (for Joe). Date: 2026-06-26. Audience: leyten & collaborators on the WAN-sharded GLM-5.2
serving stack. Status: data through 32k complete; 100k + 9B@32k in progress (tables marked). This document
explains the **why**, not just the numbers._

## 0. TL;DR

1. **Draft head-to-head (MTP vs GLM-4-9B), per-position greedy acceptance, on the dense sm120 ring:**
   GLM-5.2's native **MTP head beats** a standalone GLM-4-9B autoregressive draft at every measurable
   context, and the margin **widens** with context. MTP is also cheaper (no extra model on the coordinator;
   shares the target's embed/lm_head).
2. **The comparison is MLA-invariant.** MLA-latent caching is a *target-side* optimization; it does not
   change any draft's acceptance (proven by parity). So the head-to-head is genuinely MTP-vs-9B, with MLA
   held equal.
3. **MLA-latent caching unblocks long context + concurrency.** The dense ring as written caches decompressed
   full-head K/V and tops out ~16–24k context on an 8×96 GB pod. The standard MLA "absorb" (cache the
   ~kv_lora latent, run attention in latent space) cuts KV ~70× and lets 32k/100k fit at a few GB/stage.
4. **At long context the 9B can't compete *architecturally*** — not on acceptance, but because it cannot be
   there at all (see §5). MTP inherits the target's long-context training and MLA structure for free.

## 1. The metric and why it's topology-invariant

Per-position greedy acceptance: a draft token at position *i* is accepted iff
`draft.argmax(i) == target.argmax(i)` on the same teacher-forced prefix. This depends only on the two models
and the input — not on where anything runs (colocated TP=8 vs WAN pipeline; draft on tail vs coordinator).
So it is measurable on one box and transfers to the WAN ring, and "accepted run length" ≈ `1/(1−p)`.

## 2. Two ORTHOGONAL things — keep them separate

A recurring framing error is to bundle "MTP" and "MLA" together. They are independent:

- **Draft** (MTP head vs GLM-4-9B): *what proposes the speculative tokens.* This is the thing under test.
- **MLA-latent cache**: *how the **target's** KV is stored during its forward.* A serving optimization that
  applies to the target **regardless of which draft you use.**

Because MLA is target-side and mathematically equivalent to the full-head path, **it does not move any
draft's acceptance** — we verified this (parity, §3). Consequence for the report: the head-to-head is
**MTP-vs-9B with MLA held equal**, NOT "MTP+MLA vs 9B." If one objected "why not give the 9B's target MLA
too?" — you can: it changes nothing, because MLA is orthogonal to the draft. The *recommended stack* is
"MTP draft + MLA-cached target ring"; the *comparison* isolates the draft.

## 3. Results — MTP vs 9B acceptance, swept by context

All MTP numbers on the MLA-latent ring (chunked prefill); 9B is a standalone teacher-forced forward on the
same pod, compared to the **same** target argmaxes. Greedy per-position. (8 seqs each unless noted.)

| context | MTP accept p | MTP accept_len | 9B accept p | 9B accept_len | margin |
|---|---|---|---|---|---|
| 1k   | 0.849 | 6.6 | 0.793 | 4.8 | MTP +0.056 |
| 8k   | 0.879 | 8.3 | 0.820 | 5.6 | MTP +0.059 |
| 32k  | **0.883** | 8.5 | **0.745** | **3.9** | **MTP +0.138** |
| 100k | **0.881** | 8.4 | n/a — 9B 32k-capped (§5) | | MTP only |

_32k MTP = 0.8828 / accept_len 8.53 over 262,136 positions (8 seqs), peak coordinator GPU 38 GB on the
MLA-latent ring (vs 94 GB OOM with the full-head cache — the serving unlock made this datapoint exist).
100k MTP = 0.8812 over 102,400 positions (1 seq — an MTP-only existence proof; the 9B cannot reach 100k,
§5)._

**The two drafts diverge with context — MTP holds, the 9B decays.** MTP acceptance is essentially flat from
8k onward (0.849 → 0.879 → 0.883 → 0.881 across 1k/8k/32k/100k, accept-length ~8+). The 9B rises to 0.820 at
8k but then **degrades to 0.745 at 32k** — it starts breaking down *before* its hard wall, as RoPE positions
stretch toward the edge of its 32k training. So the margin **triples** from ~+0.056 (1k/8k) to **+0.138 at
32k**, and accept-length diverges to **8.5 vs 3.9** (MTP yields >2× the accepted tokens per ring traversal at
32k). The mechanism: a draft that is *part of the target* tracks the target's own representations at every
length and cannot drift from it; a separate draft both drifts AND degrades as it nears its own context
ceiling — then disappears entirely past 32k (§5). This is the clearest single result in the study: **the
draft advantage of MTP grows precisely in the long-context regime that matters for the WAN-sharded use
case.**

(1k also measured at 64 seqs: MTP 0.857 / 9B 0.812 — consistent.) MTP not only wins but its lead grows with
context, and its accept-length advantage is larger than the raw-p gap.

### 3a. Parity check — MLA does not change the numbers

Same corpus, full-head cache vs MLA-latent cache: 1k 0.845→0.849, 8k 0.876→0.879. Identical within bf16
nondeterminism. This is the empirical backing for "the comparison is MLA-invariant."

## 4. The convention bug that nearly inverted the result (methodology warning)

The first MTP pass reported **0.53** and "9B wins decisively." That was wrong: we fed the MTP head the
**pre-final-norm** hidden state. GLM-5.2's MTP wants the **post-`model.norm`** hidden (the same tensor that
feeds the main `lm_head`), concat order `[enorm(emb) ; hnorm(post_norm_h)]`. With the right convention,
0.53 → 0.86 — in DeepSeek-V3's reported MTP band. **Lesson for anyone reproducing this:** if a draft/MTP
acceptance lands *below the published literature band*, suspect a hidden-state/concat convention bug and
sweep the conventions before reporting — a degraded (not randomized) head yields a plausible-but-wrong
number. (HF `transformers` does not implement GLM-5.2's NEXTN module, so the convention can't be read off it;
we settled it by a 4-way empirical sweep.)

## 5. Why the 9B cannot be the long-context draft — and MLA does NOT fix that

Two independent reasons, both architectural:

1. **Positional, not memory.** GLM-4-9B was trained with `max_position_embeddings = 32768`. Past 32k its
   RoPE positions are out of distribution → garbage, *with any amount of VRAM*. MLA addresses memory; it does
   nothing for a positional-training ceiling. Extending it (YaRN/NTK + fine-tune) yields a *different* model
   with degraded quality, not "9B with MLA."
2. **The 9B can't use MLA at all.** MLA is not a bolt-on cache trick — it requires the model to be
   *architected and trained* with the low-rank latent KV projection (`kv_a`/`kv_b`). GLM-4-9B is a
   standard-attention model; there is no latent to compress (and its KV is small anyway — 9B is tiny —
   so memory was never its constraint).

**Therefore the long-context regime isn't a close call: the 9B is absent.** The only draft that scales with
the target is the one *built into* it. MTP inherits GLM-5.2's native long-context training and its MLA
structure for free. This is a stronger argument than any acceptance delta.

## 6. The MLA-latent cache rewrite (the serving unlock)

**Problem.** The dense ring (leyten's `Layer.attn`) caches **decompressed full-head** K/V. Per-stage cache
grows ~linearly with context (8k ≈ 22 GB, 32k ≈ 88 GB/stage); at 32k it leaves <8 GB on an 8×96 GB pod — no
room for the coordinator's overhead → OOM at any chunk size. **The dense ring tops out ~16–24k context.**

**Fix (standard DeepSeek MLA "absorb").** Cache the kv-latent `cprime` (~`kv_lora` dims) + the shared rope
key, and absorb `kv_b` into the query (for scores) and into the output projection (for values), so attention
runs in latent space and full-head K/V is **never materialized**:
`q_pass·k_nope^T = (q_pass·W_kn)·cprime^T`, and `softmax·value = (softmax·cprime)·W_vb^T`. The rope term is
unchanged (shared across heads, cached directly). **~70× smaller cache on GLM-5.2's real dims** (32k:
88 GB → ~1.3 GB/stage; 100k → ~4 GB).

**Validation discipline (offline-first, $0).** The absorb identity and the chunked cache plumbing
(append/crop + RoPE-across-chunks) were proven byte-equivalent to the full-head path on random tensors
*before any GPU* (`mla_latent.py`, errors 1e-13). On the box, `MLA_LATENT=1` reproduced the known 1k/8k
numbers (parity, §3a), then **cleared 32k at 38 GB on the coordinator** (vs the 94 GB OOM before). It also
unblocks **concurrency**: the freed KV memory is exactly what lets multiple streams share the ring.

**Residual note.** Once the cache stopped being the bottleneck, the next limit was the lm_head projecting a
full long sequence to a `[S, vocab]` logits matrix (~20 GB at 32k) — fixed by blocking the argmax over
positions. Mentioned because it's a generic long-context gotcha, not specific to this stack.

## 7. Recommendation

For the WAN-sharded GLM-5.2 serving stack: **MTP head as the draft + MLA-latent target cache.** MTP is the
stronger and cheaper draft at all contexts and the *only* one that reaches 100k; MLA-latent caching is what
makes the target serveable past ~24k and is the lever for concurrency. The two are independent wins that
compose.

## 8. Reproducibility

Code: `shard/research/` — `ring_long.py` (chunked driver, blocked argmax), `glm_swarm_nvfp4_kv.py`
(`MLA_LATENT=1` latent-cache attn), `mla_latent.py` (offline equivalence proofs), `ring_mtp.py` (MTP head +
`--diag` convention sweep), `pod_9b_dump.py` (pod-native 9B), `h1_bench.py compare` (offline acceptance).
Turnkey run: `h1env/h1_session.sh`. Full process log: `H1-process-and-results.md`, `H1-test-plan-v2.md`.
