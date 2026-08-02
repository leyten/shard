# V4 pipelined speculation — where the throughput goes

**The premise this investigation started from was wrong, and the ring proved it wrong.** The brief was
"the pipeline is 85–89% bubble, find the bubbles." The coordinator's own instrumentation says the
opposite: on a six-stage ring it reported `max_inflight 7, mean_inflight 5.5, unsent_frames 0`. The
pipe is full. The bottleneck stage is busy **91%** of the wall clock. There is no fill problem.

What the pipe is full of is speculation the ring computes and then throws away — `stale_replies 51`
against `accepted 33`, i.e. **46–54% of every frame the ring computes is discarded**. And the fix is
not simply "speculate less", because depth is *simultaneously* what keeps the bottleneck fed and what
a rejection destroys. That tension is the whole problem, it has an interior optimum, and finding it
is what `phase0/v4_pipe_sim.py` is for.

Everything below is produced by that simulator. Run it: `python3 phase0/v4_pipe_sim.py --validate`.

---

## 1. The simulator, and why you should believe it

Pure-stdlib discrete-event sim, CPU only, no torch. It models the real topology rather than a
cartoon of it:

- **D single-server FIFO stages.** `v4_stage.Stage.forward` has zero internal concurrency — validate,
  `_seek`, checkpoint, compute, return, one frame at a time, no prefetch, no double buffer. That is
  why a cancelled frame still costs the ring its full service time, and it is the mechanism behind
  the whole waste story.
- **`coordinate_dspark_pipelined` reproduced statement for statement** — the `sent` map, `horizon`,
  the epoch fence, the sender thread that drops queued frames of a dead epoch, the `W` cap in `_feed`,
  and the line that decides the shape of everything, `if blk and horizon == c:`.
- `coordinate_dspark`'s chunk round including the `_seek`/`_replay` of the accepted prefix, and
  `coordinate`'s one-frame-per-token greedy loop, on the same stage model.

### 1.1 Calibration uses two disjoint bodies of evidence

| stage | free parameters | fitted against |
|---|---|---|
| drafter | `a1`, `rho` — per-index acceptance `a_i = a1 · rho^(i-1)` | the depth sweep's **acceptance column only**. No timing number is used, and no timing number can be tuned to fix it. |
| ring | `kappa` (per-position compute, ×each box's measured `fwd`), `hop_ms` | **seven** throughputs: the six-point depth sweep plus greedy |

Not free, all measured: the per-stage `fwd`/`on_box` split from the live `V4_TIMING` table
(460.30 / 362.27 / 503.39 / 579.37 / 202.79 / 97.68 ms per frame over the 8/8/8/8/8/3 tiling), the
five uplink rates (150.7 / 177.6 / 99.0 / **5.6** / 285.4 MB/s), the frame payload, the tail's
`logits 12.35` and `draft 20.32`, the chunk and replay multipliers from `v4_stage.py:237-249`, and
the block size.

**Result of step 1** — the drafter, from acceptance alone:

```
a_i = 0.796 x 0.945^(i-1)          worst residual 0.50%
  drafts in flight   measured a   model
                1        0.800    0.796
                2        0.778    0.777
                3        0.755    0.760
                5        0.733    0.737
                6        0.733    0.729
```

Acceptance of the **last** draft in a full block is **0.600** against **0.796** for the first. The MTP
predicts the whole block from one committed hidden, so draft *i* conditions on *i−1* tokens nobody has
verified yet. This is the coupling the coordinator flagged and it is real, small, and load-bearing.

**Result of step 2** — the ring:

```
kappa = 0.2839   hop = 4.0 ms       RMS 7.1% over seven throughputs
tau   = [132, 104, 144, 147, 58, 20] ms/frame     sum 604, max 147 (stage s3)
RTT   = 632 ms   =  604 ms compute + 28 ms wire over 7 legs
per-layer 13.62 ms/layer/token
bandwidth-delay product RTT / tau_max = 4.3 frames
```

### 1.2 What the sim then PREDICTS, having been fitted to none of it

```
 depth| tok/s meas    sim    err|  g meas   sim| stale meas   sim| maxinf  sim| meaninf   sim
     2|      2.487   2.60     5%|    4.80  4.55|         10    11|      2    2|    2.00  2.00
     3|      2.865   2.72    -5%|    4.36  4.55|         15    17|      3    3|    2.54  2.57
     4|      3.257   3.03    -7%|    4.00  4.23|         23    23|      4    4|    3.13  3.16
     6|      3.030   3.11     3%|    3.69  4.17|         42    41|      6    6|    4.65  4.52
     8|      2.597   2.85    10%|    3.69  3.70|         51    57|      7    7|    5.50  5.40
    16|      2.755   2.85     3%|    3.69  3.70|         51    57|      7    7|    5.50  5.40

  fitted    tok/s across the sweep      worst 10%, RMS 7%
  PREDICTED g (tokens/cycle)            worst residual 12.9%   PASS
  PREDICTED stale replies               worst residual 15.2%   PASS
  PREDICTED in-flight depth             worst residual  2.8%   PASS
  PREDICTED greedy tok/s                1.58 vs 1.80 measured  (-12%)
```

`g`, the stale-reply count and the in-flight depth are **structural** — they fall out of the
coordinator's round logic and the acceptance rate, not out of the clock. Reproducing them across the
whole depth sweep is a test of whether the round structure in the sim is the round structure on the
ring, and it passes at ≤15.2%.

### 1.3 Mutation check — the green is not vacuous

Re-fit the ring with `rho` pinned to 1.0, i.e. acceptance treated as one number instead of decaying
with draft index:

| model | RMS | g at depth 2 / depth 16 |
|---|---|---|
| decaying acceptance | **7.1%** | 4.55 / 3.70 |
| flat acceptance | 10.6% | 4.23 / 4.29 |
| **measured** | — | **4.80 / 3.69** |

The flat model cannot produce the measured *fall* in `g` with depth at all — it predicts `g` slightly
*rising*. The decay term is doing work, not decorating.

### 1.4 What the simulator gets wrong, stated plainly

- **Serial dspark: predicted 0.75 vs measured 1.43 tok/s (−47%).** The serial path is the one place
  the model leans on multipliers measured on a *different* ring (`v4_stage.py`'s "3.93× for s=6" on
  the 7×5090 ring) and on a replay depth inferred from `_seek`. It is out of the validated regime.
  **Do not use this simulator to reason about the serial chunk path.**
- **`hop_ms` lands on its 4 ms floor.** The fit wants even less. No RTT was ever measured on this ring
  (`V4_PIPELINED_SPEC.md` §9.6 #1 still lists it as unmeasured), so the floor is an assumption, not a
  measurement. What the fit is telling us is robust even so: **the wire is ~4% of the round trip.**
  632 ms per token, ~604 ms of it on-box. This ring is not transport-bound in any sense.
- **The measured optimum is depth 4, the sim's is depth 6.** Both curves are a plateau (sim: 4→3.02,
  5→3.07, 6→3.11; ring: 4→3.257, 6→3.030), and the ring's own repeatability is ~6% — depth 8 and
  depth 16 are the *identical* configuration and measured 2.597 vs 2.755. Treat the answer as **"the
  plateau is depth 4–6"**, not as a single integer. Sim and ring agree on the thing that matters:
  the shipped default is off the plateau.

---

## 2. Where the throughput actually goes

At the shipped `V4_SPEC_DEPTH=16`, on the calibrated ring:

```
pipeline-rate ceiling   1 / tau_max                7.32 tok/s   (tau_max 147 ms, stage s3)
achieved                                           2.85 tok/s   = 39% of it
bottleneck stage busy                              91.1% of the wall clock
frames computed per token committed                2.06
of the frames computed, discarded                  54%
```

So the loss decomposes as **9% fill, 54% waste**. Not 85% bubble. The earlier "26 tok/s ceiling /
11–15% efficiency" framing came from dividing the 579 ms per-frame `on_box` by the timed window's
`s = 15.0`; the ring's own depth sweep says the true per-frame bottleneck for an `s=1` decode frame
is **147 ms**, so the real ceiling of this ring is **7.3 tok/s**, and we are at 39–42% of it.

**The two ceilings, and which binds.** Throughput is
`(tokens / frames) × min(1/tau_max, depth/RTT)`. The first factor is acceptance and waste; the second
is the pipeline. At depth 16 the ring is already at the `1/tau_max` end, so **only `tau_max` and the
waste fraction can move it.**

---

## 3. (a) Optimal speculation depth — the rule

Depth is a tug of war between three effects:

1. **Fill.** Below the bandwidth-delay product `RTT / tau_max`, the ring runs at `depth/RTT` instead
   of `1/tau_max`. More depth is strictly good here.
2. **Waste.** A rejection kills *everything* in flight behind it, so discarded frames per cycle grow
   linearly in depth.
3. **Acceptance decay.** The marginal frame of depth is the *deepest* draft, and it has the *lowest*
   acceptance (0.600 vs 0.796). So the last frame of depth buys the least fill and costs the most
   waste — which is why the optimum sits *below* the BDP, not at it.

```
   D  tau_max    RTT    BDP  depth*   tok/s*    g*  waste*  vs D=6
   3      270    576    2.1       2     2.74  4.55     18%
   4      204    628    3.1       4     2.65  4.23     33%
   5      163    590    3.6       4     2.99  4.23     33%
   6      137    626    4.6       6     3.11  4.17     46%      0%
   8      103    631    6.1       6     3.52  4.17     46%    +13%
  10       83    659    7.9       6     3.60  4.17     46%    +16%
  12       70    662    9.5       6     3.72  4.17     46%    +20%
  16       53    691   13.0       6     3.77  4.17     46%    +21%
```

> **THE RULE: `depth* = clamp( round(RTT / tau_max) − 1 , 2 , B+2 )`** — one below the bandwidth-delay
> product, clamped by the hard in-flight cap the block size imposes.

Acceptance barely moves it (sweeping `a1` from 0.65 to 0.95 moves `depth*` between 4 and 6). **Tune
depth off the ring's `RTT/tau_max`, not off the drafter.** Both are printed by `V4_TIMING` plus one
greedy run: `tau_max` is the slowest stage's `on_box` for `s=1` frames, `RTT` is `1000/greedy tok/s`.

**Are we over-speculating today? Yes, and it is a free win.** `V4_SPEC_DEPTH=16` measured 2.755 and
depth 4 measured 3.257 — **+18% on the ring, for an environment variable.** The sim puts the same
change at +9% (it favours 6 over 4 inside the plateau). Either way the shipped default is off the
plateau and costs 8–18%. **Set `V4_SPEC_DEPTH=4` on the 6-box ring today.**

**Why depth saturates at 7.** The coordinator streams `[correction] + block`, and `_feed` breaks at
`p - c >= W`, so in-flight is capped at `min(W, B+2) = 7` for `dspark_block_size = 5`. That is why
depth 8 and depth 16 are the *same configuration* and measured identically. It also means **`W`
(=`V4_SPEC_DEPTH`) is not the knob above 7 — the block size is.**

---

## 4. (b) The 10-stage ring — does width still pay?

```
D=6    tau_max 137 ms   RTT 626 ms   BDP 4.6   ceiling 1/tau_max  7.32 tok/s
       depth* 6  ->  3.11 tok/s   (g 4.17, waste 46%)
       at V4_SPEC_DEPTH=16 -> 2.87 tok/s
D=10   tau_max  83 ms   RTT 659 ms   BDP 7.9   ceiling 1/tau_max 12.03 tok/s
       depth* 6  ->  3.60 tok/s   (g 4.17, waste 46%)
       at V4_SPEC_DEPTH=16 -> 3.54 tok/s
```

**D=10 buys +16% over D=6, both tuned. Not +77%, which is what the ceiling ratio suggests.**

The assumption that width roughly doubles throughput does **not** survive once misspeculation is
modelled, and the reason is precise: a wider ring has a **higher** BDP (7.9 vs 4.6) because `RTT`
barely falls — total layer work is conserved and each box adds a leg — while `tau_max` does. It
therefore *needs* more frames in flight to stay full, and every one of them dies with the next
rejection. The `B+2 = 7` cap then stops it filling to its BDP of 7.9 at all, so D=10 runs
**fill-limited**: its bottleneck is only 62% busy against 84% at D=6.

Does a better drafter rescue it? Barely:

```
    a1      g     D=6    D=10    gain
  0.70   3.30    2.96    3.39     +14%
  0.80   3.70    3.11    3.60     +16%
  0.85   4.62    3.45    3.94     +14%
  0.90   5.26    3.51    4.11     +17%
  0.95   6.00    3.92    4.36     +11%
  0.98   8.57    4.17    4.68     +12%
```

**Width is worth +11–17% across the whole plausible acceptance range.** There is no `g` at which
width starts paying like the naive model, because the block-size cap binds first.

**Plain answer for the provisioning decision:** the 10-box ring is worth roughly **+16%** (3.11 →
3.60 tok/s) and it is the *third*-best ring-shape lever, behind balancing (+24%) and comparable to
out-of-band cancel (+19%). If the 10 boxes are also better balanced than the current 6 — the measured
ring has a 2.5× per-layer spread and one box on a 5.6 MB/s uplink — the combination is worth more
than either alone. It does not change the order of magnitude. **Do not cancel it, but do not expect
it to be the step to 10 tok/s.** And raise `dspark_block_size` with it, or D=10 will run fill-limited
against the `B+2` cap.

Going the other way is worse: **D=4 fat boxes is −7%.**

---

## 5. (c) The gradient in `g`

```
    a1      g  depth*   tok/s   d(tok/s)/dg
 0.700   3.30       5    2.96
 0.750   3.45       5    3.13         1.106
 0.796   3.70       6    3.11        -0.100
 0.850   4.62       6    3.45         0.379
 0.900   5.26       5    3.51         0.090

central difference around g=3.70:  d(tok/s)/dg = 0.274 tok/s per unit g   (~9% per unit g)
```

> **+1.0 of `g` is worth about +9%. 2× on compute is worth +83%.**

And `g` is capped: with the measured 0.945 per-index decay, even a perfect first draft cannot push `g`
past ~5.9 at `B=5`. Compute has no such cap. **Price acceptance work at roughly one ninth of the same
effort spent on `tau`.** The curve is also non-monotone in the small (the −0.100 entry) because
`depth*` jumps by one between rows — another reason to read this as "acceptance is a weak lever", not
as a precise derivative.

---

## 6. Ranked interventions

All against the shipped configuration (2.85 tok/s in sim, 2.755–3.08 measured), each measured at its
own optimal depth.

| # | intervention | tok/s | Δ | note |
|---|---|---:|---:|---|
| 1 | **4× compute** (grouped fp4 MoE + whole-layer graphs) | 9.66 | **+239%** | the only unbounded lever |
| 2 | **2× compute** | 5.68 | **+99%** | |
| 3 | 10 stages, tuned depth | 3.60 | +26% | being provisioned |
| 4 | balanced stages | 3.54 | +24% | 2.5× per-layer spread today |
| 5 | drafter to `a1 = 0.90` (`g` 3.7→5.4) | 3.51 | +23% | hard; and capped |
| 6 | out-of-band cancel | 3.40 | +19% | needs a control socket per stage |
| 7 | replace the 5.6 MB/s box | 3.16 | +11% | one rental decision |
| 8 | **tune `V4_SPEC_DEPTH` 16 → 4–6** | 3.11 | **+9%** (+18% measured) | **free, today, one env var** |
| 9 | 4 fat stages | 2.65 | −7% | do not |

**Combinations** (each at its own optimal depth):

```
tuned depth + balanced + fixed uplink                3.57
  + out-of-band cancel                               3.66
  + D=10                                             3.69
  + 2x compute                                       6.44
  + 4x compute                                      10.24
  + drafter a1=0.90                                 11.34
  + 8x compute, tight 5 ms hops                     15.30
```

**Every ring-shape lever combined is worth about +30%** (2.85 → 3.69). They do not compose the way a
list of percentages suggests, because they all push against the same `B+2` in-flight cap and the same
46% waste floor.

---

## 7. Is 20 tok/s reachable on 10 boxes? No.

With **every** lever applied — balanced, uplink fixed, out-of-band cancel, drafter at `a1 = 0.90`,
depth tuned per configuration:

```
 compute   D   B   hop  tau_max    RTT   BDP  depth*  ceiling   tok/s
      1x  10   6     4     62.2    658  10.6       7     16.1    4.40
      2x  10   6     4     32.4    361  11.1       7     30.8    7.57
      4x  10   6     4     17.6    212  12.1       7     56.9    11.82
      8x  10   6     4     10.1    138  13.6       7     98.6    16.45
      8x  10  12     4     10.1    138  13.6      13     98.6    17.85
     16x  10  12     5      6.4    112  17.4      13    155.7    20.91
      8x   6  12     5     14.9    120   8.1      13     67.1    17.71
      8x  16  12     5      7.3    190  25.9      13    136.5    15.59
```

**Stated plainly: 20 tok/s is not reachable on 10 boxes at current per-token compute, with every
bubble and every wasted frame removed.** The ring-shape ceiling with all levers is ~4.4 tok/s. Compute
is 96% of the round trip; nothing else is in a position to matter.

**The ring shape that WOULD reach it**, from the sweep:

- **~16× on `tau`** at D=10 with `dspark_block_size ≥ 12` and ~5 ms hops → **20.9 tok/s**. That is the
  full compute stack (grouped fp4 MoE + whole-layer CUDA graphs + ref-slim + fast verify) landing at
  the low end of what the roofline allows: 13.62 ms/layer/token today against a ~0.1 ms fp4 roofline,
  i.e. we are ~136× off and 16× is not the outrageous part of this plan.
- **~8× on `tau` plus `B ≥ 12`** gets **17.9 tok/s** — within touching distance, and the cheapest
  credible route.
- Going wider than 10 is counter-productive: **D=16 at 8× is 15.6 tok/s, worse than D=10's 17.9**,
  because RTT grows with legs while `tau_max` shrinks, the BDP hits 25.9, and the block cap leaves the
  ring fill-limited.

**The order to do things in:** raise `dspark_block_size` (it is free and it is the cap that stops
every other lever), tune `V4_SPEC_DEPTH` off `RTT/tau_max`, balance the tiling and drop the 5.6 MB/s
box, take the 10-box ring for its +16% — and then spend everything else on `tau`.

---

## 8. What is extrapolation

Held honestly separate from what the ring measured:

| claim | status |
|---|---|
| depth 4–6 beats depth 16 on the 6-box ring | **measured** (+18% on the ring) |
| `a_i` decays ~0.945 per draft index | **measured** (acceptance column, 4 depths) |
| the pipe is full and the loss is waste, not bubbles | **measured** (`mean_inflight` 5.5/6, `unsent` 0, bottleneck 91%) |
| `tau_max ≈ 147 ms`, `RTT ≈ 632 ms`, wire ≈ 4% | **fitted** to 7 measured throughputs, RMS 7.1% |
| `depth* = round(RTT/tau_max) − 1` | **interpolation** — validated at D=6 only; the D-sweep is extrapolation |
| D=10 → +16% | **EXTRAPOLATION.** Assumes the 43 layers re-tile evenly, that a 10-box rental draws the same 2.5× quality spread, that per-box frame overhead is the current mean, and that hop latency is unchanged per leg. None of that is measured. The *sign* is robust (width helps, sublinearly); the magnitude is not. |
| `d(tok/s)/dg ≈ 0.27` at g=3.7 | **interpolation** within the measured acceptance range; the `a1 ≥ 0.9` rows are extrapolation |
| out-of-band cancel → +19% | **EXTRAPOLATION and an upper bound.** The sim lets the cancel reach every stage instantly. A real control path cannot, and `V4_PIPELINED_SPEC.md` §9.3 explains why the current one-forward-pipe topology has no such path at all. |
| everything at ≥2× compute | **EXTRAPOLATION.** Compute speedup is applied as a scalar on per-layer time. It does not model which kernels actually get faster, and the round-2 lever doc already warns the one-layer 3.19× MoE bench will not fully materialise on a ring. |
| serial dspark predictions | **DO NOT USE.** −47% against measurement; out of the validated regime. |

Two loose ends worth closing on the next ring run, both cheap:

1. **Measure a hop RTT.** It is still unmeasured, the fit pushes it to the floor, and it is the one
   parameter separating "604 ms of compute" from "some of that is wire I mis-attributed".
2. **Reconcile `max_inflight = 7` with the source.** `coordinate_dspark_pipelined` on `v4/full-stack`
   streams `[correction] + block` = 6 frames at `dspark_block_size = 5`. The ring reported 7. The sim
   uses the depth the ring showed, but the extra frame is unexplained by the code as written, and the
   `B+2` cap is now the binding constraint on every fast or wide ring — so it is worth knowing exactly
   what sets it.
