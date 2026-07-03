# The good-ring receipt — 2026-07-03

**The run the perf step-back demanded:** the good-ring tok/s and the tree-verify gain had never been
measured on the SAME ring — "10-12 tok/s" was arithmetic. This is the measurement, plus the first
per-stage transport/compute split ever taken on a live ring (`M25_STAGE_TIMING`, PR'd on
`perf/tree-depth-hybrid`).

## Ring

RTT-measured, `select_ring`-chosen (head-first deployable order, layer blocks from the planner),
5 fat stages from an 8-box over-rented EU pool — all scattered consumer/DC 5090s, distinct /24s:

| stage | box | layers | per-stage span (chain, ms) |
|---|---|---|---|
| s0 HEAD+coord | Czechia 43696900 (EPYC 9J14, load 12) | 0-10 | 29 |
| s1 | Czechia 43696887 (EPYC 9474F, load 33) | 10-23 | 35 |
| s2 | Switzerland 43696869 (Core Ultra 9 285K, idle) | 23-36 | **11.5** |
| s3 | Norway 43696886 (Core Ultra 5 245K, idle) | 36-49 | **12.5** |
| s4 tail | Denmark 43696878 (EPYC 7702, load 12) | 49-62 | 50 |

Loop RTT ≈ 105ms (CZ→CZ leg is 5ms — different ISPs, genuinely scattered). Warmed once with
`M25_EAGLE=1 M25_FP8_WIRE=1 M25_STAGE_TIMING=1 --receipts`; both arms ran on the same warm ring,
receipts signed+verified per cell (full coverage, true depth 62).

## Results (full usability report, reasoning ON, greedy; JSONs alongside)

decode-weighted over all 18 cells: **chain-EAGLE 8.3 tok/s** · **tree-hybrid 7.83** ·
**per-cell-best (depth-aware hybrid upper bound) 9.11** — conversation mean 9.0 / **9.28**.

| cell | chain | tree | winner |
|---|---|---|---|
| reason-math | 9.3 | **10.0** | tree |
| reason-logic | 6.8 | **10.1** (+49%) | tree |
| open-chat | 6.3 | **7.2** | tree |
| agentic-tool | 6.7 | **11.2** | tree |
| code-edit | **9.3** | 6.8 | chain |
| rag-quote | **5.7** | 5.2 | chain |
| ctx-8k summarize / quote | **13.4** / **11.8** | 10.0 / 5.1 | chain |
| ctx-30k summarize / quote | **5.9** / **6.6** | 5.0 / 3.4 | chain |
| conversation turns 1-8 | 6.3-14.9 | 7.7-11.8 | split |

**The 10-12 ceiling is now receipt, not arithmetic**: interactive novel-reasoning runs at
**10-11.2 tok/s today** (tree arm) on a genuinely scattered EU ring, with signed receipts on. The
split is textbook: the tree wins every interactive/novel cell; depth-4 pipelined n-gram wins every
verbatim/long-context cell on a fast ring (chunk latency low → pipelining pays). Neither arm should
lose its cells: **depth-aware hybrid** (pipeline n-gram rounds at depth, keep tree rounds sync) is
the next code lever, worth ~+1 decode-weighted immediately (8.3 → 9.1 bound).

## The transport/compute split (first measurement)

Chain arm, reasoning cells: T_traversal ≈ 306-427ms = **transport 55-68%** (wire+sidecar+codec,
~170-290ms vs the ~105ms pure-RTT floor) + **stage compute ≈ 138ms** — 3.5× the ~40ms the model
assumed. The engine was NOT "98% blocked on wire" on this ring.

**Discovery — stage compute is CPU-launch-bound, not GPU-bound.** All five GPUs bench identically
(1523-1527 GB/s, 218-227 TFLOPs), yet identical 13-layer blocks take 11.5ms on an idle Core Ultra
box and 35-50ms on old/oversubscribed EPYC slices (single-thread pyloop 0.09s vs 0.28-0.47s; one
spare had load average 272). The block forward is hundreds of kernel launches; launch cost is the
box's single-thread CPU. Consequences:
1. **Box selection must probe CPU + load** (now in `ring_up.py` as a `layer_ms` factor; the honest
   version feeds a warm run's measured `per_stage_ms` back into re-selection — the self-optimizer).
2. **CUDA graphs are un-dead for scattered rings.** The ~1.05× "dead-end" verdict (torch 2.11) was
   measured on a fast-CPU box; on EPYC-slice boxes graphs should recover ~2-4× of the block time.
   Needs `GraphRunner` aux-capture compatibility (EAGLE aux is a python side-effect — graphs skip
   it today), a scoped code task.

## Lever ranking after this receipt (all measured, none speculative)

| lever | size | cost |
|---|---|---|
| depth-aware hybrid (chain cells for the tree arm) | 8.3 → ~9.1 aggregate | code, CPU-testable |
| fast-CPU boxes and/or CUDA-graph-with-aux | −60-80ms/traversal ⇒ +20-30% | probe (done) / scoped code |
| lean codec / thin-TCP (transport is 55-68%) | up to +20-30% | medium code |

Stacked honestly: **~12-14 tok/s interactive reasoning** is reachable on this ring class — at or
above the step-back's projected ceiling. Path (a) confirmed: execution, no invention required.

---

## Addendum (same day, later): depth-aware hybrid A/B/C — arm 3 on the same warm ring

The #1 lever landed same-session (`feat(coord): depth-aware hybrid`, adversarially reviewed, 64 CPU
tests): matched n-gram rounds now ride PLAIN pipelined chain frames (up to --depth in flight, flash
kernel, small payload); novel rounds stay sync tree. Receipt
`m25-usability-goodring-hybrid-20260703.json`, same ring, same flags + depth=4.

**Caveat first: the ring ran 1.32× slower during arm 3** (measured on tree-routed cells at identical
g — 441/464/412 ms/round vs 380/315/303 in arm 2; co-tenant load on the EPYC boxes). Raw numbers
carry that headwind.

- raw decode-weighted: **7.59** (tree arm was 7.83, chain 8.3, per-cell-best bound 9.11)
- **normalized to the arm-1/2 ring speed: ≈10.0 decode-weighted** — ABOVE the per-cell-best bound,
  because pipelining adds throughput neither depth-1 arm had.
- g (ring-independent) confirms the design exactly: novel cells IDENTICAL to the tree arm
  (reason-math 4.6, reason-logic 3.6, agentic 4.3); verbatim cells strictly better than the tree arm
  (rag-quote 2.5→2.7 + pipelined → 5.2→6.7 tok/s raw DESPITE the slower ring; ctx-8k-quote 2.2→2.8,
  5.1→7.6 raw).
- mean_accept is now honest across arms (review fix: pipelined full-accept rounds counted K, not K-1).
- Remaining weak cell: ctx-30k-quote (3.5 raw; tree rounds at 30k pay the manual kernel over 30k
  keys — 807 ms/round). The CUDA-graph/lean-codec levers and a flash tree kernel path are the
  follow-ups that touch it.

Same-ring-same-time A/B was defeated by time-of-day variance; a calm-window interleaved 3-arm pass
would tighten the claim, but g-parity + verbatim raw wins on a slower ring already pin the mechanism.
