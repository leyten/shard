# M2.5 PoC lever stack — adversarially verified projection (2026-07-16)

> Derived from the Inkling lever hunt (`inkling-lever-hunt-20260716.md`): the subset applicable to the
> RESIDENT M2.5 ring (WAN-bound: 45 ms summed stage compute / 169 ms round / 73% wire+idle, receipts
> 2026-07-11/12). Draft composition was attacked by a 3-lens skeptic panel (physics/double-count,
> engine-reality-vs-receipts, paper-evidence transfer); numbers below are the post-correction consensus.
> Baselines = the 07-12 K-tuned scorecard, best-K per (arm, B): tools 29.7 (K8) / reasoning 29.0 /
> code 23.2 (K8) / qa 21.1 / prose 18.5 / mix 17.9 / B1 22.5.

## The verified levers (exact tier — bit-exact receipts preserved)

| Lever | Corrected multiplier | Evidence status | First action (cheap gate) |
|---|---|---|---|
| **C. REAP expert pruning → fewer hops** | **×1.14 banked** (REAP12 → ~13 L/card → 5 hops); ×~1.3 stretch (REAP25 → 16 L → 4 hops, zero headroom) | STRONGEST of the set — method-general, published ≤1.5-pt agentic deltas at 25% on M2-family; hop→wire physics receipt-validated on OUR ring (4-hop measured ×1.33 vs ×1.32 predicted; 2-hop reasoning 51/stream) | Quality-gate on τ²-bench/BFCL-style agentic evals (NOT ppl) + NVFP4 prune-path validation |
| **A. Suffix-tree drafter (hybrid w/ EAGLE)** | ×1.1-1.4 code/tools B4 (upper half ONLY on genuinely repetitive multi-turn traffic); ×1.0-1.15 reasoning/qa; ×1.0 prose; ×1.2-1.6 at B1 (deep frames near-free there — K8-wins-at-B≤2 precedent) | SuffixDecoding (arXiv 2411.04975): AgenticSQL 6.3 vs EAGLE-3 3.6 accepted/step; hybrid ≥ EAGLE everywhere; but our tools chain g is ALREADY 4.5-5.0 — headroom is real, smaller than the paper ratio | **Half-day offline replay of our existing receipt traces through a suffix tree** — decides ×1.2 vs ×1.4 before any engine work |
| **B. Async post-verify drafting** | ×1.05-1.15, B1 only (batched drafter graphs already killed the draft tax; serial step ≈ 8-16 ms of a ~170 ms round; de-lockstep already fills the gap at B>1) | PEARL's ×1.29-1.55 measured where draft≈target time — regime does NOT transfer | One warm run reading `draft_s` from coordinate_pipe: build only if ≥~25 ms/round |
| **D. Speculative cascades (top-k accept)** | extra ×1.15-1.35, OPT-IN LABELED TIER only (breaks bit-exact receipts as default); fullest value on prose where A does nothing | sound; overlaps A's acceptance headroom on agentic arms | env-gated accept mode; k=1 must reproduce greedy bit-exact |
| ~~Trellis 2.5-3 bpw~~ | DEAD for resident M2.5: W×0.6 but C×~1.8 → ×1.05-1.1, not worth kernel-weeks (wire is RTT-floor-dominated, ÷1.8 bytes buys even less) | all three lenses concur | — |
| ~~DFlash M2.5 drafter~~ | τ 4.6-5.0 ≈ EAGLE parity — substitute, not additive | — | half-day A/B at most |

## Composite projection (exact tier)

| Arm (baseline) | Banked (A central + C 5-hop) | Stretch (REAP25 4-hop + repetitive traffic) |
|---|---|---|
| tools 29.7 | **34-42** | ~45 |
| code 23.2 | **29-36** | ~40 |
| reasoning 29.0 | **32-38** | ~43 |
| qa 21.1 | **24-28** | ~30 |
| prose 18.5 | **20-22** ← REAP alone closes the last 20-bar gap | ~24 |
| mix 17.9 | **19.5-21.5** | ~24 |
| B1 (22.5) | **31-40** (mid ~35-38) | ~45 |

**Headline: exact tier ×1.25-1.55 (central ~1.4) on the agentic arms, ×1.10-1.20 prose/mix,
B1 ×1.4-1.8. Opt-in cascade tier: ~×1.6-2.0 total (stretch ~2.2-2.4 on code/tools).**
Zero training anywhere; A+B+D are days-scale, C is weeks-scale (the quality gate is the work).

## Sequencing (bank-first)
1. REAP12 quality gate → 5-hop ring (×1.14, highest confidence, also a cost lever).
2. Suffix-tree OFFLINE trace replay (half-day) → then build behind the HybridDrafter seam if ≥×1.2.
3. Read `draft_s` from a warm run → build async post-verify only if the term is ≥25 ms.
4. Cascades as the labeled fast tier, marketed where A didn't move (prose).
Full skeptic verdicts: scratchpad m25_verdicts.json (session); lever provenance: inkling-lever-hunt-20260716.md.
