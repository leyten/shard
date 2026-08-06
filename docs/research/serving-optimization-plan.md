# GLM-5.2 WAN-sharded serving optimization — plan

_Author: Henry. Date: 2026-06-26. Goal: turn the H1 findings (MTP draft + MLA cache) into actual serving
throughput and concurrency. North star: beat ~30 tok/s single-stream on GLM-5.2 AND serve concurrent users.
Same discipline as the MLA work: prove offline ($0) before any metered run._

## 0. The metric (don't optimize the wrong number)

**NET tok/s = accept_len / (draft_time + verify_time)** per stream, and **aggregate tok/s across concurrent
streams.** H1 measured the *numerator* input (acceptance/accept_len); this phase builds the loop that turns
it into wall-clock tok/s, and the batching that turns one stream into many. Acceptance is already settled —
do NOT re-measure it; measure *time* on the real box — never quote remembered throughput.

## 1. What we already have (inputs, validated)

- **MTP draft** — acceptance 0.85–0.88 flat across 1k–100k, accept_len ~8 (H1). Co-located (1 NEXTN layer,
  cheap), shares embed/lm_head; draft_time is small and NOT a separate WAN hop (unlike the 9B).
- **MLA-latent cache** — ~70× less KV, proven equivalent + working on-box. Frees the memory that concurrency
  needs and removes the long-context wall.
- **leyten's ring** — KV-cached stages, relay-back, and a `GraphVerify` (CUDA-graph of the fixed K+1 verify
  shape) already in `glm_swarm_nvfp4_kv.py` (currently opt-in/untuned).

## 2. Levers, ordered by leverage

1. **MTP spec-decode DECODE LOOP (the core win).** Build the real loop: at each step the MTP head proposes
   the next token(s) from the target's tail hidden, the ring verifies K+1 in one traversal, accept the
   longest matching prefix, roll the KV back on reject (the stages already support `start_pos<len` rollback).
   This converts accept_len ~8 into ~Nx fewer ring traversals per token vs plain AR decode. **Biggest single
   throughput lever** because the ring traversal is the expensive unit.
2. **MLA-latent cache in the serving path.** Deploy `MLA_LATENT=1` (built). Cuts KV ~70× → the freed VRAM is
   the budget for concurrency (lever 4) and long context. Already validated; this is integration, not R&D.
3. **CUDA-graph the verify.** Turn on + tune leyten's `GraphVerify` so the K+1 verify replays from a captured
   graph (no per-call kernel-launch/Python overhead). Shrinks verify_time — the denominator. Self-disables on
   capture failure, so low-risk.
4. **Concurrency / continuous batching (the throughput multiplier).** Batch draft+verify across streams; with
   MLA-freed KV, many streams' caches fit. Per-traversal cost amortizes across the batch → aggregate tok/s
   scales. This is where "offer concurrency" is delivered. Needs a scheduler (admit/evict, per-stream KV) on
   the coordinator.
5. **Tree / multi-candidate MTP drafting.** MTP proposes a small tree (several continuations) verified in one
   traversal → higher accepted tokens per (expensive) traversal. leyten's tree path exists; pair it with the
   MTP head. Diminishing returns after lever 1, but real on the WAN ring where traversals dominate.
6. **Prefill / TTFT optimization (the honest edge).** Big-context prefill over the ring is traversal-heavy.
   With MLA memory, use large prefill chunks; move the coordinator/MTP off the hot GPU0 (today's chunk=128
   slowness was exactly this co-location squeeze). Improves TTFT for long coding contexts.

## 3. Offline-first staging (prove $0 before metered)

- **Loop correctness offline:** the spec-decode accept/rollback logic is testable against a reference greedy
  decode on small inputs (like `mla_latent.py` did for attention) — assert the spec loop produces
  *token-identical* output to plain AR decode (correctness is non-negotiable; speed is the only thing that
  should change). No GPU needed for the control-flow proof.
- **Scheduler logic offline:** the batching/admission/eviction policy is plain control flow — unit-test it
  with mock streams before it touches the ring.
- **Graph-verify shape audit:** confirm the fixed verify shapes offline from the model config.
- Only after green: one metered box session measuring *time* (NET tok/s single-stream, then aggregate across
  2/4/8 concurrent streams), via the turnkey `h1_session.sh` pattern (auto-stop, stop-not-destroy).

## 4. Suggested sequence (each a small, gated step — stay in the loop)

1. MTP decode loop + offline correctness proof (token-identical to AR). → 1 metered run: single-stream NET
   tok/s vs the ~30 baseline.
2. Turn on MLA + GraphVerify in that loop. → same run measures their time deltas.
3. Continuous batching + scheduler (offline-tested). → metered run: aggregate tok/s at 2/4/8 streams.
4. (Optional) tree drafting + prefill/TTFT tuning, if the numbers justify.

## 5. Cost & risk

- Most of the build is offline ($0): the loop, the scheduler, the correctness proofs. Metered runs are short
  and measure *time*, not search for results.
- Risk is correctness (a spec loop that diverges from AR output is a silent quality bug) — mitigated by the
  token-identical offline gate. Concurrency adds scheduler complexity — mitigated by offline unit tests.
- Open question to decide first: single-box (8 colocated GPUs, what we have) vs true WAN multi-box for the
  concurrency measurement. Single-box first (cheaper, isolates the loop+batching); WAN second (adds the real
  inter-stage latency the ring will see in production).

## 6. First decision for Joe

Start at **lever 1 (the MTP decode loop)** — it's the biggest throughput win, it's the thing H1's acceptance
numbers were *for*, and its correctness is fully offline-provable. I'd build the loop + the token-identical
offline proof first, then a single short metered run for the NET tok/s number vs the 30 baseline — before
committing to the concurrency/scheduler build. Confirm and I'll start the offline loop.
