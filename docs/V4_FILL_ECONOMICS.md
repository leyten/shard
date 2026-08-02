# V4 fill economics after the cheap drafter — floor first, width second

**The cost structure that made every fill lever lose is gone, and the family reprices from
"+3%, don't build" to the largest single opportunity in the engine.** `V4_DSPARK_MOE` took the
tail from 14.64 to 6.00 ms/frame with drafting ~0, the ring came out flat
(9.48 / 8.24 / 9.29 / 8.67 / 8.14 / 6.00 ms — the HEAD binds), and the measured 24.75 tok/s is
**29% of the 85 tok/s frame ceiling**. This doc re-derives the fill family on that ring, names what
was built, and gives the numbers that falsify each claim.

Models: `phase0/v4_ngram_econ.py` (frame-exact replay, `--ring 0801`) and `research/v4_fill_sim.py`
(the same round on `v4_pipe_sim`'s per-stage FIFO chain — it queues where the replay clamps). Two
drafter calibrations bracket every number. Written 2026-08-02.

---

## 1. Why the old verdicts inverted — two structural facts, both new

1. **Drafting is off the bill.** The old tail spent 9.16 of its 14.64 ms drafting, and it was the
   binding stage — so pinning the refill floor made the tail draft 2.5x more often and *halved* the
   frame ceiling (floor=3 measured −40%, floor=5 −32%, against models predicting +11..+45%). Today
   the tail is 6.00 ms with draft ~0 and sits **3.5 ms under the binding head**: even drafting on
   every frame at width 13 does not re-bind it. The lazy-defeat charge prices to zero
   (`v4_ngram_econ.py`'s draft-charged rows: +0.0 at 0.4 ms and at 2.0 ms).
2. **The cap now binds, twice over.** Compute fell ~2x but the circuit barely moved (wire is
   unchanged by compute levers): L ≈ 129–161 ms against tau_max 9.48 ms puts the bandwidth-delay
   product at **~14–17 frames**. The structural in-flight cap is still `block+1 = 6`, and the shipped
   drain-only round *means* ~4.2 of even that. The ring is fill-bound with its stages ~70% idle —
   which is precisely what makes speculative frames nearly free: waste lands on capacity nobody was
   using.

The falsifier for the whole premise: `inflight_time_avg` (now in every pipelined run's dict) over
the judged-reply rate gives L by Little's law. **If the next run's L comes out ≈ 60 ms (BDP ≈ 6),
the fill headroom does not exist and everything below reprices to ~nothing.** The carried
110.7 ms wire is the one number here that is not this config's own measurement.

## 2. The refill floor, re-priced (`V4_REFILL_FLOOR`, already built)

Priced across both machineries and both calibrations, vs the shipped floor=1 at 24.75 tok/s:

| arm | replay (fit / calibrated) | queueing sim (fit / calibrated) |
|---|---:|---:|
| floor=3 | +9.7% / +5.1% | +8.7% / +4.8% |
| **floor=5** | **+18.2% / +10.7%** | **+15.2% / +9.0%** |
| floor=5, topups fully re-rolled | — | +0.2% / −1.5% |

**Expect floor=5 at +9..+18% (28–29 tok/s), degrading to ~flat — not negative — in the worst
disagreement case.** The one assumption the spread hangs on is topup agreement: a mid-run block's
deep drafts condition on its own re-predicted prefix, and the models bracket that between "always
agrees" and "always re-rolls". The run measures it directly (`topup_agree`/`topup_disagree`,
`topup_accept_by_depth`). **Falsifier: floor=5 loses if the topup books show deep-slot conditional
acceptance collapsing toward `prod(q_1..q_j)` — i.e. `topup_accept_by_depth[5]` at ~0.55 rather
than `accept_by_depth[5]`'s ~0.88.** The 07-31 re-test (13.3/13.3/14.7 under a co-tenant at load
11.39) was contention noise, not a measurement; it decided nothing.

## 3. The cap-lifter: a WIDER block from the same tap (`V4_DSPARK_BLOCK`, built here)

The MULTIBLOCK lockstep proof was re-checked against the code and **still holds — cost was never
its term** (§5). What it never bounded is the number of proposals from ONE tap:
`dspark_block_size` dimensions no weight tensor, every vendored shape is symbolic in it, and the
Markov chain in `forward_head` conditions each slot on the whole sampled prefix. `V4_DSPARK_BLOCK`
(tail-only, opt-in, default OFF, 1..32) builds the drafter at an inference-time width; the
coordinator already adapts to whatever length the reply carries, so in-flight fills to
`min(W, width+1)`.

Priced with the trained curve extended by deep slots at `q_deep`, pinned by `floor=width`:

| arm | replay (fit / cal) | sim agree (fit / cal) | sim reroll (fit / cal) |
|---|---:|---:|---:|
| k=8, q_deep=q5 | +47% / +33% | +43% / +29% | +23% / +20% |
| **k=10, q_deep=q5** | **+61% / +44%** | **+51% / +36%** | +34% / +24% |
| k=13, q_deep=q5 | +78% / +51% | +62% / +43% | +47% / +36% |
| k=10, q_deep=0.5 | +18% / +18% | +14% / +12% | +14% / +12% |
| k=10, q_deep=0 (zero benefit) | +9% / +10% | +4% / +4% | +4% / +4% |

**Expect k=10 at +36..+61% over today (33–40 tok/s) if the deep slots hold near the trained q5,
and ~floor=5-equivalent at q_deep ≈ 0.5.** k=13 saturates the BDP; go there only after k=8/10
measure well.

**The honest price, and why this is a measurement rather than a win:** `get_dspark_topk_idxs`
gives every block query the same index set — the block attends to itself **bidirectionally** — so
extra noise slots perturb the trained slots' logits too (proven on CPU:
`tests/test_v4_dspark.py::test_widening_perturbs_the_trained_slots...`). A widened run's
`accept_by_depth[1..5]` is therefore a new curve, not the old one plus extras.

**Falsifiers for k=10, computed by `v4_ngram_econ.py`:** the lever loses to plain floor=5 if the
measured deep-slot acceptance lands **below ~0.50 flat** (fit curve; 0.14 on the calibrated one) at
intact shallow slots, **or** if widening degrades the trained slots by **more than ~9%**
(shallow multiplier < 0.91) even with perfect deep slots. Both are one read of `accept_by_depth` on
the first `V4_DSPARK_BLOCK` ring run against the trained-width baseline, same prompts.

## 4. The zero-benefit floor, quantified — it is NOT zero

If the extra speculation is never accepted, throughput does **not** equal today's:

* vs the **floor=5** operating point it rides on: **−1% to −9%** (junk frames add stale service on
  a bottleneck that is no longer fully idle, and every cancel's correction traverses behind them);
* vs the shipped **floor=1** round: still +3..+10%, because the width carries the floor's fill for
  its trained slots either way.

So the failure mode of the wide bet is "gave back the floor's margin", not "cratered the ring" —
bounded, and visible in one run as `frames_per_token` exploding while `accept_by_depth[6..]` reads
zeros.

## 5. Chained multi-block: the lockstep argument re-checked, and it HOLDS. Stop.

Re-derived against the code with drafting at ~0: `DSparkAttention.forward` writes its cache from
`main_x` alone (model.py:783), `forward_embed` consumes exactly one `main_hidden`, and
`advance_and_draft` requires one tap per advanced position — there is still no path that re-enters
the MTP trunk on its own output, and `main_proj`'s input (12288) cannot take the trunk's own h
(16384). A second drafter call needs a tap one position deeper; only the ring's traversal makes
taps; and the reply that carries the deeper tap's block is the same reply that advances the
frontier past it. **In-flight stays ≤ block+1 no matter how many times the drafter runs or how
cheap running it is. Cost was never the binding term, so cheap drafting changes nothing. Dead;
not pursued.** (Faking the missing tap — re-calling with a stale one — is reachable mechanically
but is strictly the wide block with extra steps and a poisoned cache slot; the wide block is the
disciplined form of the same bet.) n-gram stays dead on merit: measured acceptance 0.012 on novel
code against a break-even of ~0.62–0.75.

## 6. What the next ring run reads (instrumentation shipped with this branch)

* `inflight_time_avg` + `decode_wall_s` — the TIME-weighted fill (the event-weighted
  `mean_inflight` overstates by ~6.5%); with the judged-reply rate this measures L and settles the
  BDP premise of §1.
* `frames_per_token`, `block_len` — the waste multiple and the width actually served.
* `accept_by_depth` / `topup_accept_by_depth` / `topup_agree` / `topup_disagree` — the per-depth
  curves that replace every fitted number above; depths > 5 are the wide block's report card.

The sweep that decides everything, one warm ring, six runs:
`{floor 1, floor 5} x {width off, 8, 10}`, `V4_SPEC_DEPTH=16`, same novel prompts, co-tenancy
checked before trusting any number (the 07-31 floor re-test died to a loaded head).

## 7. Correctness status

Losslessness is absolute and proven on real localhost socket rings against the vendored
reference's own greedy decode: the full selftest matrix (greedy / spec / serial dspark / pipelined
/ lazy at depths 2, 3, 16 / floors 2, 3, 5) is bit-identical under `V4_DSPARK_BLOCK=7`
(`test_the_wide_drafter_serves_exactly_what_the_default_serves`), the widened ring holds width+1
frames in flight through rewinds on every cycle with receipts settling
(`test_the_wide_drafter_lifts_the_pipelined_cap_on_a_real_ring`), and the widened drafter is
bit-equal to a reference widened the same way
(`test_width_override_drafts_wider_bit_equal_to_a_widened_reference`). The scripted-coordinator
suite covers wide blocks through rejections at depths 6–7, floors 1 and width, and the W-truncation
guard. Width moves only what is speculated; a draft is committed only when the ring's own reply
equals it.
