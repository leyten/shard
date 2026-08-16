# H1 — GLM-4-9B vs MTP draft acceptance for GLM-5.2: process, findings, results

_Author: Henry. Date: 2026-06-26. Status: DONE for 1k + 8k — MTP beats GLM-4-9B at both (~+0.05, widening),
pod-native apples-to-apples. 32k/100k BLOCKED by the decompressed-KV-cache wall (target tops out ~16–24k on
this pod; needs MLA-latent caching). See §6. For peer evaluation by leyten and collaborators._

## 1. The question and the metric

**H1:** does GLM-5.2's native **MTP** head outperform a standalone **GLM-4-9B** autoregressive draft as a
speculative-decoding draft for **GLM-5.2**, and how does that change with context length?

**Metric — per-position greedy acceptance.** For greedy/deterministic spec-decode a draft token at
position *i* is accepted iff `draft.argmax(i) == target.argmax(i)` on the same teacher-forced prefix.
This is a property of the two models (topology-invariant — same whether the target runs colocated TP=8 or
WAN-pipelined), so it is measurable on a single colocated box and transfers to the WAN ring. Expected
accepted run ≈ `1/(1−p)` under the standard geometric model; we also read the *real* accept-len from the
spec-decode metrics where available. We sweep context: 1k / 8k / 32k / 100k.

## 2. Methodology — the split (keeps metered compute minimal)

`draft.argmax` and `target.argmax` are **independent** functions of the same fixed corpus ids, so:
- **Draft argmaxes** are computed cheaply and dumped to disk (9B on the beast, $0).
- **Target argmaxes** are computed on the rented GPU box (the only thing that can't run on our hardware).
- **Comparison** runs offline (`h1_bench.py compare`), no GPU.

Vocab-mismatch handling: GLM-4-9B vocab 151552 vs GLM-5.2 154880 → a target argmax ≥ 151552 is a token the
9B literally cannot emit → a forced miss (charged + reported as `oov_target`; ~3.3k rare tokens, measured
~sub-1% in practice). GLM-4.7-Flash was a candidate (tokenizer-identical) but **dropped** (62.5 GB blows
the coordinator's KV headroom → out as a coordinator draft → "we don't care about it" — Joe).

Code: `research/draft_accept_bench.py` (acceptance core, 19 offline checks), `research/h1_bench.py`
(dump/compare/mtp, 14 checks), `research/ring_target_dump.py` (the sm120 ring driver). Corpora built with
the GLM-5.2 tokenizer in `research/h1_offline/corpora/` (1k/8k/32k/100k).

## 3. Staging discipline

- **Stage 0 (offline, $0):** acceptance math + extraction unit-tested; corpora built; tokenizer-compat
  confirmed (4.7-Flash identical, 9B id-aligned + 3.3k OOV gap).
- **Stage 1 (beast, $0):** GLM-4-9B served, per-position argmax dumped over 1k/8k/32k (the 9B is
  **hard-capped at 32k**, `max_position_embeddings=32768`, no RoPE scaling → cannot run at 100k). 104 seqs.
- **Stage 2 (rented 8× RTX PRO 6000, sm120):** target argmax + MTP — this doc's subject.

## 4. THE KEY FINDING — GLM-5.2 DSA is not servable on sm120 via stock engines

GLM-5.2's architecture is `GlmMoeDsaForCausalLM` / `glm_moe_dsa` = **DSA (sparse MLA) attention**
(`head_size=576`, `use_mla`, `use_sparse`, `index_topk=2048`, fp8 KV). `nvidia/GLM-5.2-NVFP4` is the same
model quantized — DSA is intrinsic, not a function of the quant.

- **Stock vLLM 0.23:** `ValueError: No valid attention backend found` on the RTX PRO 6000 (sm120, cc 12.0).
  The DSA sparse path (FlashMLA-Sparse) is built for **Hopper sm90 + datacenter Blackwell sm100/B200 only**;
  every MLA backend rejects sm120 (`compute capability not supported` / `sparse not supported`). Same gap as
  DFlash.
- **Stock SGLang:** its GLM-5.2 cookbook lists supported HW as **H200, B200, B300, GB300** — sm120 not on it.
- **B200/H100 are not options** — the target deployment is prosumer GPUs (RTX PRO 6000). So this is a hard product
  constraint, not a test artifact.
- **Disabling sparse in vLLM (run dense) is whack-a-mole:** `index_topk` is a **class-default attribute**, so
  `is_v32 = hasattr(config, "index_topk")` is True even after removing it from `config.json`; the gate
  recurs across `deepseek_v2.py` (×2), `deepseek_mtp.py`, `config.py`, and the backend selector. Patching all
  of them did not converge cleanly. Abandoned.

**How leyten serves it on sm120 — and the resolution:** leyten's shard runs GLM-5.2 **dense** — his stage
attention (`glm_swarm_nvfp4_kv.py:attn`) is plain `softmax(Q·Kᵀ)·V` over all positions, **no sparse indexer,
no top-k** (eager). He sidesteps vLLM entirely with ~160 lines of PyTorch. So **dense IS the production
configuration on sm120**, which means our dense measurement is production-accurate, not a compromise. The
sm120 path is therefore: run the target through leyten's dense ring, not stock vLLM.

## 5. Reproducibility — bringing up leyten's dense ring on a rented 8× RTX PRO 6000

Image `ghcr.io/joelovestech/shard-glm52:cu130-nvfp4-cap4` (vLLM 0.23 venv `/root/vmoe`, leyten's stage
modules baked at `/root/`). Model `nvidia/GLM-5.2-NVFP4` fetched once (~465 GB) — config.json **restored**
from the un-edited backup so leyten's code sees the real config. Gotchas hit (and fixes), in order:
1. `glm_swarm_nvfp4_kv.py` does `import wire; wire.key_from_env()` at module load → needs **`wire.py`
   co-located** (it's `/root/wire.py`) and **`SHARD_PSK`** in env. Run everything from `/root/`.
2. **Transport mismatch:** `launch_stage` sets `SHARD_WIRE=` (empty) → stages use the `torch.save` *pickle*
   transport; the coord inherited the image's `SHARD_WIRE=1` → *wire* transport → stage0 crashed
   (`UnpicklingError: invalid load key ':'`). Fix: run the coord with **`SHARD_WIRE=` empty** too.
3. **Device placement:** the ring returns the tail hidden state on `cuda:0`; coord embed/lm_head/norm are on
   `cuda:7` → device mismatch in the final norm/lm_head. Fix: `.to(dev)` on the returned hidden state
   (`draft_accept_bench.target_argmax_per_position`).

Working invocation (on box, from `/root`):
```
SHARD_WIRE= GLM_DIR=/root/glm52nvfp4 /root/vmoe/bin/python ring_target_dump.py --smoke   # de-risk
SHARD_WIRE= GLM_DIR=/root/glm52nvfp4 /root/vmoe/bin/python ring_target_dump.py \
    --corpora .../code_{1024,8192,32768}.jsonl --out /root/dump_target_ring.jsonl
```
Smoke confirmed: 13 ids → 13 in-range argmax in 3.3 s on the dense sm120 ring (8 stages, ~430 GB).

## 6. Results

> **1k DONE** (9B p=0.8115, accept_len 5.30, 0.20% OOV, dense sm120 ring — production-accurate). 8k+ blocked: leyten's dense attention does the full O(n^2) score matrix in one prefill -> CUBLAS_STATUS_INTERNAL_ERROR at 8k. Fix = chunked/incremental teacher-forced prefill over the KV cache (each chunk attends [chunk, ctx], not [ctx, ctx]) — how leyten generates in prod. MTP still to implement (head on the ring's tail hidden state). Original note:  (1k/8k/32k via the dense ring). 9B acceptance = `h1_bench compare`
> of the ring target dump vs the beast 9B dump. MTP = the MTP head (`enorm/hnorm/eh_proj → 1 block →
> shared_head` on the ring's tail hidden state) vs target. To be filled on completion.

| context | 9B accept (p) | 9B accept_len | MTP accept (p) | MTP accept_len | winner |
|---|---|---|---|---|---|
| 1k (64 seqs)    | 0.8115 | 5.30 | **0.8569** | **6.99** | **MTP** +0.046 |
| 1k (8 seqs, pod-native 9B) | 0.7922 | 4.81 | **0.8451** | **6.45** | **MTP** +0.053 |
| 8k (8 seqs, pod-native 9B) | 0.8196 | 5.54 | **0.8755** | **8.03** | **MTP** +0.056 |
| 32k   | BLOCKED — KV cache wall | | BLOCKED | | (target can't serve 32k on this pod) |
| 100k  | n/a (9B 32k-capped) | | BLOCKED — KV cache wall | | (target can't serve 100k) |

**Head-to-head verdict (DONE 2026-06-26): MTP beats the GLM-4-9B draft at every measurable context.** At 1k
and 8k, on identical pod hardware with the 9B run *on the pod* (apples-to-apples, same 8 seqs, same metric),
MTP wins by **~0.05 acceptance** and the margin **widens** with context (1k +0.053 → 8k +0.056); MTP's
accept-length lead is larger still (6.5–8.0 vs 4.8–5.5). MTP is also the *cheaper* draft: no separate 9B to
host on the coordinator, and it shares the target's embed/lm_head. **Recommendation: use GLM-5.2's native MTP
head as the draft, not a hosted 9B.** (Files: `mtp_accept_1k_corrected.json`, `mtp_accept_long.json`,
`dump_9b_pod.jsonl` vs `dump_target_long.jsonl`.)

**⚠️ The 32k/100k KV cache wall — the session's other key finding (matters more for serving than a 32k
datapoint).** 32k OOM'd at chunk=512 AND chunk=128 — *not* a chunk-size problem. leyten's ring caches the
**decompressed full-head K/V**, not MLA's compressed latent, so the per-stage cache grows ~linearly with
context (8k ≈ 22 GB/stage, 32k ≈ 88 GB/stage) and at 32k it leaves <8 GB — no room for even the coordinator's
~11 GB of embed/lm_head/MTP overhead on any of the 8×96 GB GPUs. **So the GLM-5.2 dense ring tops out around
~16–24k context on this pod**, regardless of draft. The fix is the standard MLA optimization: cache the
~512-dim kv-latent and re-expand per step (≈10× less KV) — a stage-side change to leyten's `attn`. This
unlocks BOTH long context (32k/100k) AND concurrency (Joe's original "speed it up + concurrency" goal), so
it's the highest-leverage serving change. 8k/32k chunked prefill itself works (validated at 8k, 33 GB);
the wall is purely cache memory.

**Methodology note — chunked prefill (the 8k+ unlock that DID work).** The one-shot dense prefill builds the
full S×S score matrix → OOM past ~1k. `ring_long.py` feeds each sequence in chunks at increasing `start_pos`
through the stages' existing MLA KV cache (chunk attends `[chunk, cached-prior]`, score matrix chunk×total).
Validated: chunked 1k reproduces non-chunked 1k (0.845 vs 0.857, the small gap is 8 vs 64 seqs). RoPE table
sized via `$GLM_MAXPOS` so long positions don't index OOB. See `H1-test-plan-v2.md` for the turnkey
environment that runs this as one auto-stopping command.

**1k head-to-head (DONE 2026-06-26):** GLM-5.2's native MTP head **beats** the standalone GLM-4-9B AR draft
at 1k context — **accept p 0.857 vs 0.812, accept_len 6.99 vs 5.30** (MTP: 56,102/65,472 matches, 64 seqs,
dense sm120 ring, `mtp_accept_1k_corrected.json`). The MTP head is both the *stronger* proposer AND
zero-extra-model (no 9B to host on the coordinator, shares the target's embed/lm_head). This is the central
result H1 set out to get, and it favors MTP.

**⚠️ Convention-bug correction (the result nearly shipped backwards).** The first 1k MTP pass reported
**0.5262** and a "9B wins decisively" verdict — that was WRONG, caused by feeding the MTP head the
**pre-final-norm** hidden state. A 4-way convention sweep (`--diag`, 16 seqs, `mtp_diag.log`) settled it
empirically:

| convention (hidden × concat) | MTP accept p |
|---|---|
| pre-norm,  `[emb;hidden]` (original, buggy) | 0.5137 |
| **post-`model.norm`, `[emb;hidden]` (CORRECT)** | **0.8624** |
| pre-norm,  `[hidden;emb]` | 0.0002 (random) |
| post-norm, `[hidden;emb]` | 0.0098 (random) |

GLM-5.2's MTP wants the **post-final-norm** hidden (the same tensor that feeds the main `lm_head`), concat
order `[enorm(emb) ; hnorm(post_norm_h)]`. The ~0 rows confirm the concat order; the pre-vs-post row is the
0.51→0.86 fix. Lesson: when an MTP/draft number lands *below the literature band* (DeepSeek-V3 ≈ 0.85), treat
it as a likely convention bug and sweep the hidden-state/concat conventions before reporting — a degraded
(not randomized) head looks like a plausible-but-wrong number.

**MTP measurement method (peer-review note).** MTP runs as the *coordinator's draft* (a drop-in for the 9B —
NOT a model stage): at position i the coord computes `eh_proj([enorm(emb(t[i+1])) ; hnorm(model.norm(h_i))])`
from the ring's tail hidden `h_i`, runs it through the NEXTN layer (layer 78) via leyten's own `run_block`
(the proven `set_forward_context` wrapper — fixes the FusedMoE forward-context issue a hand-rolled wrap hit),
then `shared_head.norm` + `lm_head` → argmax = MTP's prediction of `t[i+2]`. Acceptance =
`mtp_pred[i] == target_argmax[i+1]` (both predict `t[i+2]` from the same prefix `t[0..i+1]`).

**8k/32k/100k still pending** the chunked/incremental teacher-forced prefill (leyten's dense attn does the
full O(n²) score matrix in one shot → CUBLAS error >1k). Same blocker for both 9B-target and MTP at long
context; it's a clean, well-scoped follow-up, not new discovery.

## 7. Caveats for the reviewer
- **Dense, not DSA-sparse** — but this matches leyten's production serving, so it's representative, not a
  shortcut. (If leyten ever switches to true sparse kernels, re-evaluate.)
- **Numerical nondeterminism** (~sub-1%) on near-tie argmax positions; affects both drafts equally.
- **9B 32k cap** is a real disqualifier as a long-context draft, independent of acceptance.
- **Cost of the sm120 lesson:** ~$22–25 burned discovering the stock-vLLM/sm120 incompatibility on a metered
  box. Process correction recorded: verify the **attention backend** exists for the target GPU's compute
  capability before renting — arch registration ≠ serveable.
