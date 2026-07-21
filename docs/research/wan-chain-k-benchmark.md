# The K lever: round-length sweeps on a scattered-WAN M2.5 ring, a zero-parameter round law, and a benchmarking protocol

Contributed by Mosaic Intelligence (2026-07-07). All numbers below were measured on our own
6× RTX 5090 scattered-EU ring running this repo's engine at commit `182e93b` (pristine,
config-only unless a flag is named), with signed receipts and raw per-run JSONs retained on
our side; receipt names are cited inline and can be furnished on request. This document
contributes (1) a finding about the chain round-length lever K that the paper's law section
currently overlooks, (2) a zero-parameter round law R(K) verified through K=128, and (3) the
pre-registration benchmarking protocol we used, offered as a discipline for future receipts.

---

## 1. Motivation: the paper's α/T dichotomy misses the K lever

The technical report ("Sharded Inference of a 229B-Parameter MoE over the Public Internet
at Interactive Speed", Zenodo DOI [10.5281/zenodo.21178430](https://doi.org/10.5281/zenodo.21178430)
— the concept DOI; the current version 1.2 resolves at 10.5281/zenodo.21180635) derives the
pipelining law (§4.2) and concludes that on draftable spans
"only raising α (drafter training) or cutting T (transport) moves that number." Our sweeps
show a third lever moves it a lot: **the round length K**, at fixed α and fixed T.
`research/m25_ctx_table.py` (the harness behind the promoted 70.7–87.2 tok/s draftable
number) pins K=8, depth 4. On the verbatim/n-gram cell, K=8 leaves most of the achievable
throughput on the table, because on high-α spans the binding constraint is rounds-to-finish,
not per-round acceptance risk.

The K lever is lossless by construction: K changes what is *drafted* per round, never what is
committed — the ring greedy-verifies every proposal, and every run below is byte-identical
(sha-verified) to plain greedy at its cell.

## 2. Setup (regime the verdicts bind to)

- MiniMax-M2.5 NVFP4 (229B-total / 10B-active), 5-stage ring + coordinator, 6× RTX 5090,
  scattered-EU WAN, **loop RTT 274.8 ms** (vs the paper's ~105 ms tight-EU goodring).
- Engine: this repo at `182e93b`, bf16 wire, keep-warm OFF, greedy, single-stream B=1,
  chain path (`coordinate_pipe`), hybrid n-gram+EAGLE drafter.
- Two cells: **420-tok verbatim copy** (420-token prose passage copied verbatim, max-new 420)
  and **long verbatim continuation** (~1k-token prompt, max-new 1600). Decode-only rate,
  prefill excluded — same convention as `m25_ctx_table.py`.
- Every run's output sha is checked against the cell's plain-greedy sha; a run that diverges
  is excluded from the lossless class by rule (none did in the rows below).

## 3. The K ladder (420-tok verbatim cell, depth 8)

| K   | tok/s (95% t-CI)  | n | rounds R | g exact  |
|-----|-------------------|---|----------|----------|
| 8   | 181.5 ± 3.4       | — | 54       | 7.59     |
| 16  | 239.5 ± 10.9      | 10 (pooled) | 29 | 14.5 |
| 24  | 273.8 ± 14.1      | 8 (pooled)  | 21 | 20.0 |
| 48  | **294.9 ± 25.4**  | 7 (pooled)  | 12 | 34.17 |
| 64  | 319.6 ± 10.8 (fresh session; 308.8 ± 13.2 at n=8 cross-session) | 4/8 | 10 | 41.0 |
| 96  | 275.7 ± 53.2      | 4 | —        | —        |
| 128 | 302.3 (exploratory) | 2 | 7      | 58.57    |

tok/s peaks at K=64 on this cell (64 > 80 > 72 > 96 on the finer grid); acceptance shows no
cliff (per-round accept fraction 0.95→0.63 across the ladder, still 0.46 at K=128) — the knee
is round-time, not acceptance collapse. Receipts: `mosaic-m25wan-ring2-record-20260706`,
`…-record-ext-20260706`, `…-record-k48-20260707`, `…-knee-20260707`.

## 4. Long-cell ladder and the zero-parameter round law

Long verbatim continuation (max-new 1600), depth 8:

| K   | tok/s (95% t-CI) | n | rounds R (every run) | g exact |
|-----|------------------|---|----------------------|---------|
| 64  | 647.7 ± 38.8     | 4 | 19                   | 53.05   |
| 96  | 647.4 ± 44.8     | 4 | 14                   | 72.0    |
| 128 | **710.2 ± 20.1** | 8 | 11                   | 91.64   |
| 192 | 606.6 ± 102.1    | 4 | 9                    | 112.0   |
| 256 | 624.6 ± 78.5     | 4 | 7                    | 144.0   |

**The knee is located at K=128, bracketed from both sides:** in the K>128 session both arms read
below their interleaved K=128 control (668.8 mean, n=2) — same-session ordering
128 > 256 > 192. Mechanism is round-time, not acceptance: byte identity held 10/10 at
K∈{128,192,256} (no proposal-quality envelope effect even at K=256 against the drafter's
margin=256), while per-round wall time grew ~137 → ~186 → ~232 ms — the +36%/+69% round-time
growth outpaces the 11 → 9 → 7 round savings. g keeps climbing past the knee (K=256 commits a
mean 144 tokens per traversal); above the knee raw g is no longer the binding constraint,
per-round wall time is. Receipt: `mosaic-m25wan-ring2-lk2-20260707`.

**Round law.** On a verbatim-saturated cell the number of verify rounds is deterministic and
zero-parameter: with C tokens to commit past the warmup rounds,

- 420-tok cell: `R(K) = 3 + ceil(407/K)` — reproduces all 8 ladder rungs exactly;
- long cell:  `R(K) = 4 + floor(1004/K)` — four pre-registered predictions hit exactly:
  R=14 at K=96, R=11 at K=128, R=9 at K=192 (4/4 runs), R=7 at K=256 (4/4 runs), with
  g = 1008/R exact to the digit each time (72.0 / 91.64 / 112.0 / 144.0). The law is verified
  at every tested K from 8 through 256 — it survives past the throughput knee, because it
  prices rounds, not speed.

Rounds are the whole story: tok/s = (tokens committed) / (R × round time). An affine
round-time model over-predicts high-K round time by +22–65% at the optimum (sub-affine
confirmed at K=128); a **commit-attribution** model (round time follows *committed* tokens;
draft-size slope ≈ 0) won two independent pre-registered model brackets on this ring
(boundary-K A/B ratio 1.038 vs a draft-attribution prediction of ~1.32; a K=59-dominance
prediction from the competing model was refuted, measured T64/T59 ≈ 0.93 vs predicted
[1.088, 1.090]). Receipts: `mosaic-m25wan-ring2-lk-20260707`, `…-boundaryk-20260707`.

The payload/knee heuristic `K* ≈ sqrt(S/(B·c))` (S = span length) explains why the long
cell's knee sits at K=128 while the 420-tok cell peaks at K=64. Content-robustness:
a fresh 818-word passage reproduced the sustained row within the pre-registered band (ratio
0.856, band [0.85, 1.15] — near the low edge, stated as such;
`mosaic-m25wan-ring2-classrobust-20260707`).

## 5. Fair comparator sentences (both configs, both RTTs, one sentence)

We compare against this repo's published numbers *class-fair, best-vs-best*, always naming
both configs and both network classes in the same sentence. The two sentences we use — noting
your promoted draftable ceiling is itself the verbatim/α≈0.97 class (n-gram drafter, measured
per-round accept 89%, K=8 depth-4, ~105 ms loop):

> Cell-B verbatim 420-token copy, single-stream B=1, chain K=48 depth-8, byte-identical to
> plain greedy: **294.9 ± 25.4 tok/s through a 274.8 ms scattered-EU WAN loop** vs the
> paper's promoted single-stream ceiling of **87 tok/s on draftable text (α ≈ 0.97, n-gram,
> K=8 depth-4, ~105 ms tight-EU loop)** — same workload class, ~3.4× the promoted ceiling on
> a ~2.6× worse network class, with the larger K/depth config stated.

> On interactive novel-text reasoning the same ring measures **10.87 ± 0.79 tok/s** vs the
> paper's promoted 10.7–12.6 median on a ~2.6× faster loop — the network-class advantage is
> upstream's, honestly stated. The verbatim record never travels without this anchor.

The sustained-class row (710.2 ± 20.1 at K=128, max-new 1600) is a different length class
and is never cross-quoted against the 420-tok headline.

## 6. The benchmarking protocol we ask future receipts to adopt

Everything above was produced under a discipline we found cheap and unreasonably effective;
we offer it here as a proposed convention for `docs/receipts/`:

1. **Pre-register before data.** Every measurement block has a registration document on disk
   *before* first ring contact: prediction bands, kill criteria, n per arm, drop rules, and the
   exact schedule. Negative results are banked under the same receipt naming as wins (our
   receipt set includes five pre-registered negatives, e.g. fp8-wire 0.96× kill-fired,
   keep-warm 0.89× on this ring class).
2. **Byte-identity as a class boundary.** A "lossless" claim requires output sha equality
   with pristine-engine greedy on every run of the row; runs that diverge (e.g. greedy
   tie-flips under trajectory-mixing policies) are excluded from the lossless class and
   banked separately with the divergence labeled.
3. **Regime binding.** Every verdict names its regime (model/quant, B, ring class, loop RTT,
   engine sha, flags, drafter, cell) and binds to it — no cross-ring or cross-class quoting.
4. **Config-stated comparisons.** A comparator sentence names both sides' K/depth/α/loop-RTT.
   Ranges promoted from crashed or n<3 sweeps are quoted as published but not leaned on for
   precision (we quote "87", not "87.2", for this reason).
5. **Interleaved A/B for engine deltas.** ON/OFF arms interleave on the same warm ring
   (the `research/m25_lever_bench.py` pattern generalizes this); paired ratios with t-CIs;
   WAN drift is real (we measured ±7% same-config cross-block drift at n=4 granularity) and
   only interleaving cancels it.

## 7. What we did NOT measure

- No batched (B>1) rows anywhere above; the B=1 idle-window structure is load-bearing for
  some of our engine deltas.
- No cross-model claim: all rows are M2.5-NVFP4 on this ring class.
- Both knees are bracketed from both sides within their sweeps (420-tok cell: 64 > 80 > 72 > 96;
  long cell: 128 > 256 > 192), but the grids are coarse — the true optima may sit between
  tested K values. No K was tested above 256.
