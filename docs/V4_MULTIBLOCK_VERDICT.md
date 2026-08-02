# V4 multi-block drafting — NO, and the ring is too wide

**Verdict: do not build chained multi-block drafting. It is not unprofitable, it is unreachable** —
the DSpark drafter cannot produce a draft for a position the ring has not already computed, so no
amount of speculating-on-speculation moves the pipeline's fill. The measured symptom is real and the
diagnosis behind it was not. **The action this releases is to shrink the ring from ten stages to six**,
worth ~1.22x on the same hardware, against the ~1.03x ceiling of the lever that was proposed.

Arithmetic: `research/v4_pipe_efficiency.py` (run it, no arguments). Written 2026-08-01.

---

## 1. The measured cap

`coordinate_dspark_pipelined` streams s=1 frames so stage k works token i while stage k-1 works i+1.
A depth sweep on a **ten-stage** ring:

| depth | tok/s | g | max_inflight | stale | waste |
|---:|---:|---:|---:|---:|---:|
| 4 | 2.516 | 5.82 | 4 | 21 | 0.288 |
| 6 | 2.761 | 4.92 | 6 | 41 | 0.456 |
| 8 | 2.843 | 4.92 | **6** | 41 | 0.456 |
| 12 | 2.934 | 4.92 | **6** | 41 | 0.456 |

Depths 6, 8 and 12 are byte-identical: **four of the ten stages are permanently idle.** And the
ten-box ring measures *slower* than a six-box one (4.39 vs 5.35 tok/s pipelined; greedy 1.64 vs 2.28,
which is the extra hops charging pure latency).

The `max_inflight` column is exactly reproduced by the coordinator's own bookkeeping. Drafts stream
while `(pos + 2 + i) - c < W` with `c = pos + 1`, so at most `W-1` of them go out and in-flight is
`min(W, B+1)`: 4 at W=4, and 6 at every W past the block. That much of the original diagnosis is
right, and `tests/test_v4_pipe.py::test_pipelined_ring_is_lossless_at_every_depth_and_caps_at_block_plus_one`
now pins it so a future change that genuinely lifts it will announce itself.

## 2. The cause is not what it looked like

The proposed reading was "the coordinator can only stream frames it has drafts for, and DSpark drafts
a fixed block of 5" — implying the fix is to ask for more drafts. **The tail is already drafting a
fresh block on every single frame of an accepted run, and the coordinator throws almost all of them
away** (`v4_pipe.py`, the `if blk and horizon == c` refill guard). Drafts are not the scarce thing.

The real ceiling is a lockstep between two facts:

1. **A draft needs a tap.** `DSparkBlock.forward_embed` (`deepseek_v4_ref/inference/model.py:851-858`)
   builds its block from **one** `main_hidden` — the main model's tap over `dspark_target_layer_ids` —
   plus `block_size-1` noise slots. `DSparkAttention` writes its cache from `main_x` alone
   (`model.py:783`). There is no path that re-enters the MTP trunk on its own output: the only
   coupling between drafts is the rank-256 Markov bias at `model.py:867-871`, applied to an
   already-computed logit matrix. So the drafter cannot advance one position without a tap, and only
   the ring produces taps.
2. **The tap and the commit ride the same reply.** The tail forwards frame `P`, taps it, and drafts
   `P+2 .. P+B+1`. That reply reaches the coordinator, which commits `P+1` off it (`c = pos + 1`).
   Horizon becomes `c+B`, frontier becomes `c`. **In flight = B+1. Every time, forever.**

So a "second block conditioned on the first block's drafts" is not a thing the coordinator is
declining to ask for. Conditioning it requires `main_hidden` at a block-A position, which requires the
ring to have forwarded that position — which is the very traversal the speculation is trying to run
ahead of. Making the tail advance *speculatively* over unverified positions changes nothing, because
the reply that would carry the deeper block is the same reply that advances the frontier past it.

Two independent reviews reached this cap; one of them attacked the argument above and broke the
*reasoning* while confirming the *conclusion* — the taps for speculative positions do exist and are
computed (`v4_dspark_draft.py:591`, before the accept test at `:615`), they are simply always one
frame behind the frontier. Corrected, the argument is lockstep, not availability.

Raising `dspark_block_size` is the one thing that *would* move the cap. Shapes permit it — no weight
tensor is dimensioned by it, and the kernel is symbolic in the block dimension. The weights do not:
they are trained at 5, and positions 5..9 would be noise slots conditioned only through the rank-256
Markov bias. That is a measurement no CPU test can make and a quality risk on the one thing
speculation cannot trade away.

> **ADDENDUM, 2026-08-02 — the lockstep above was re-checked after drafting became ~free
> (V4_DSPARK_MOE) and it HOLDS: cost was never its term, so chained multi-block stays dead. The
> width paragraph above is now a LEVER.** With the tail at 6.00 ms (3.5 ms under the binding head)
> and the ring fill-bound at 29% of its frame ceiling, the inference-time width is priced at
> +36..+61% for k=10 *if* the deep slots accept near the trained q5 — which remains exactly the
> measurement this section said no CPU test can make, so it ships as opt-in `V4_DSPARK_BLOCK`
> (default OFF) with `accept_by_depth` as its report card, bit-exactness proven at width on real
> socket rings. Economics, falsifiers and the zero-benefit floor: docs/V4_FILL_ECONOMICS.md.

## 3. The acceptance-decay arithmetic, and the break-even

Acceptance is not one number, and this is the load-bearing observation. Let `q_j` be the conditional
probability that the block's *j*-th draft lands, given the drafts before it did. Tokens per cycle is
then the expected index of the first miss:

```
g(b) = (sum_{i<b} P_i) / (1 - P_b),      P_i = prod_{j<=i} q_j
```

**If `q` were flat this is `1/(1-q)` regardless of `b`.** The sweep moves g from 5.82 at three drafts
to 4.92 at five, so acceptance provably decays with depth. Two equations, two unknowns, fitted with
`q_j = 1 - (1-a)·rho^(j-1)`:

| depth j | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| **q_j** | 0.859 | 0.827 | 0.787 | 0.739 | **0.680** |

(a = 0.8587, rho = 1.2271; reproduces g(3) = 5.820 and g(5) = 4.920 against measured 5.82 / 4.92.)

Now the throughput model. Every frame earns exactly one reply, replies retire at `min(F, D)/L`, and
only an unfenced reply commits a token:

```
tok/s = (tokens / frames) · min(Fbar, D) / L
```

Fitted against the sweep this gives L ~ 0.79 s and reproduces the measured waste column (0.383
modelled vs 0.456 measured at depth >= 6; 0.254 vs 0.288 at depth 4).

**A free falsification, from data the sweep already has.** The model says the fill *means* ~4.4 while
peaking at 6, because the shipped refill guard lets it sag 6,5,4,3,2 before topping back up. That
number was already recorded: `mean_inflight` is in the returned dict of every run in the sweep. If it
reads ~4.4, everything below stands. If it reads ~6.0, the sag is not happening, §4's lever is already
in effect, and the arithmetic here needs redoing before it is trusted.

**The break-even.** Adding one frame to a pipeline that already holds `F` costs one more frame thrown
away behind every rejection, so it only pays if the g it buys clears:

| F → F+1 | g must hold above | i.e. tolerable drop |
|---|---:|---:|
| 4 → 5 | 3.94 | −19.7% |
| 5 → 6 | 4.22 | −14.1% |
| 6 → 7 | 4.39 | −10.5% |
| 8 → 9 | 4.60 | −6.4% |

The tolerance narrows as the pipe deepens, which is the whole shape of the problem: **deep frames are
both the least likely to be right and the most expensive to be wrong about.**

## 4. What the nearest buildable lever is worth: +3%, and negative if you are unlucky

The one thing the coordinator *could* do without a new drafter is stop the sawtooth. In-flight peaks
at 6 but **means 4.0** — it sags 6,5,4,3,2 and refills only when the block is fully consumed. Taking
each reply's fresh block instead pins it at 6.0, a 1.5x gain in *fill*.

Fill is not throughput. The one position each mid-run block adds is always its **deepest** draft, at
`q_5 = 0.68`, the worst in the profile:

| policy | g | Fbar | waste | tok/s (D>=6) | vs shipped |
|---|---:|---:|---:|---:|---:|
| block (shipped) | 4.91 | 4.42 | 0.383 | 3.451 | — |
| rolling top-up | 4.41 | 6.00 | 0.531 | 3.560 | **+3.1%** |
| rolling, 80% agreement | 4.70 | 5.24 | 0.482 | 3.436 | −0.4% |
| rolling, 60% agreement | 5.24 | 4.17 | 0.390 | 3.222 | −6.6% |

The +3.1% is the *best* case, assuming a fresh block always re-predicts the in-flight positions
identically. It will not: the deep draft's Markov conditioning is its own block's prefix, not what is
actually in the ring, and once those disagree more than ~15% of the time the lever is **negative**.
Trading a lossless coordinator's risk for +3% is not a trade worth making, so this is not built —
only the comment at the refill guard is corrected, so the next reader does not re-derive it.

**Not an artifact of the fitted curve.** Two points fix two parameters but not the family, so the
script refits three other shapes (miss-linear, miss-power-law, q-geometric) to the same two measured
points. Every shape that can represent the data at all lands the top-up at **+3.0% to +3.2%** and puts
`q_5` at 0.68-0.69. The verdict is a property of the measurement, not of the curve drawn through it.

> **CORRECTION, 2026-08-02 — the +3% was a property of the OLD drafter, and the lever is now
> built.** Everything above prices the top-up against `q_5 = 0.68` / `g = 4.92`; the 07-31 six-stage
> ring measures `g = 11.13`, and the lever's value is almost entirely q at the block's deepest
> index. Re-priced on that ring the same rolling refill models **+11% to +45%** (the spread hangs on
> the per-depth decay shape — `phase0/v4_ngram_econ.py`, and a ten-agent discrete-event calibration
> lands at +21% inside it). It ships as **`V4_REFILL_FLOOR`** (default 1 = the drain-only round this
> section describes, frame for frame; `floor=B` pins in-flight at `block+1`), with the two risks this
> section named turned into per-run measurements instead of assumptions: the re-prediction
> disagreement is counted (`topup_agree`/`topup_disagree`) and topped-up frames are scored apart
> from drain-refill frames by depth (`topup_accept_by_depth` vs `accept_by_depth`). The non-monotone
> low end survives in both models — floor=2 prices between −2.5% and break-even — so the operating
> point is a measurement, not a knob to max out. §2's cap argument is untouched: the floor fills the
> pipe *up to* `block+1`; nothing here lifts it past that.

## 5. Usable ring width, as a function of block count

This is the number to size against. Total traversal is `L = n_layers·tau_layer + D·hop`: a wider ring
makes each stage faster but adds a hop, while by Little's law the retire rate is `min(F, capacity)/L`
with **F pinned at block+1**.

| blocks chained | in-flight cap F | usable ring width | reachable? |
|---:|---:|---:|---|
| 1 | 6 | **~5-6** | **yes — this is today** |
| 2 | 11 | ~10-11 | no (§2) |
| 3 | 16 | ~15-16 | no (§2) |

**Rent six stages, not ten.** One honesty note on the width column, because the obvious answer is
slightly wrong: it is tempting to say the peak sits exactly at `D = F = block+1`, but that assumes a
ring saturates at `D` frames. It does not — a frame occupies a *stage* or a *link*, so a ring overlaps
about `D·(1 + hop/tau)` frames. Solving the model's one free parameter against the measured
6-box/10-box pair (5.35 vs 4.39, a 1.22x calibration rather than a prediction) puts the peak at **D=5**,
with D=6 within ~6% of it and D=10 giving up ~22%.

So the rule is not "D = block+1". It is: **once the ring can overlap the frames the drafter is able to
supply, every further stage is pure added latency — so run the narrowest ring the weights fit on.**
For V4-Flash on 5090s the memory floor is ~6 boxes, which lands on the flat top of that curve for a
harder reason than the arithmetic. Six is the answer either way; ten buys four idle stages and four
extra hops.

If block count could be lifted, depth would only pay on a ring wide enough to absorb the wasted
frames — two blocks *lose* badly at D=6 (2.46 vs 3.56) and need D>=11 before they beat one. Depth and
width have to be bought together or not at all.

## 6. What to do instead

1. **Shrink the ring to six.** Free, ~1.22x, and it is the direct consequence of the cap.
2. **Measure `q_j` on the real ring before spending anything else.** `coordinate_dspark_pipelined` now
   reports `accept_by_depth` — hits/trials keyed by draft depth. Feed it back in with
   `research/v4_pipe_efficiency.py --accept '{"1": [h,t], ...}'` and every number above is re-derived
   from measurement instead of a two-point fit. The fitted curve is the weakest input to this verdict
   and it is now cheap to falsify.
3. **The only real cap-lifter is a tap-free proposer.** `phase0/ngram_draft.py` (`NgramDrafter`,
   prompt-lookup, zero model and zero KV) needs no `main_hidden`, so it can propose arbitrarily far
   ahead — and `_feed` is already proposal-agnostic, as is the tail's accept rule, which asks only
   whether a frame sits at `_cfront+1` carrying `_mfront`. It is wired to `coordinate_spec`, not to
   the pipelined path. That is the lever with headroom (cap becomes `V4_SPEC_DEPTH`, default 16), and
   it is worth building **only on a ring wide enough to pay for it and only after (2)** — n-gram
   acceptance on code is far below MTP's, and §3's break-even table is unforgiving about deep frames.
4. **Do not raise `dspark_block_size`.** Shapes allow it, the trained weights do not.

## 7. Correction to `V4_PIPELINED_SPEC.md`

§6's caveat reads "chain-draft the next block off the current block's **predicted hidden**". The MTP
stack has no predicted-hidden output that can stand in for `main_hidden` — `main_proj` takes
`dim · len(target_layer_ids)` = 12288 while the trunk's own `h` is `hc_mult · dim` = 16384, and the
shapes not fitting is the least of it. That sentence is what made multi-block look like an available
upside lever; it has been corrected in place to point here.
