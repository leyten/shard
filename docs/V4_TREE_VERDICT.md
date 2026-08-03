# V4 tree speculation — reachable, but it does not buy what it was re-opened for

**Verdict: do not build the tree yet. Branching IS reachable from the vendored head — unlike
multi-block it needs no retraining — but a tree cannot lift the in-flight cap it was re-opened to
lift.** It adds *candidates at the same six positions*, not deeper positions, so its entire value is
rescue-at-rejection: **+18.3% × β** for the best buildable shape, where β — how often the drafter's
runner-up is the token the model actually committed at the missed slot — **is unmeasured**, against
~600–1000 lines of per-lane state forking on the hot serve path. The action this releases is the
gate: `V4_DRAFT_TOP2` ships the runner-up column on the existing linear round and both coordinators
count `rescue_by_depth`, so one warm-ring run measures β and settles the build. Written 2026-08-02,
against the 07-31 six-stage EU ring's numbers (L ≈ 156 ms, τ_max 9.5 ms, block 5, g 11.13,
per-depth acceptance {1: .93, 2: .85, 3: .83, 4: .95, 5: .84}).

---

## 1. Reachability: the head DOES expose branches (this is not multi-block)

`DSparkBlock.forward_head` (model.py:860-874) computes **full per-slot logits** —
`self.head(self.norm(x), full_logits=True)` is `[b, block_size, vocab]` — so top-k alternatives
exist at every slot. And the block's hidden states cannot depend on which tokens are drafted:
`forward_embed` builds the queries from **noise-token embeddings** (model.py:854; only slot 0 is the
anchor). The *only* inter-slot token conditioning is the rank-256 Markov bigram head, added to
`logits[:, i]` in place before slot i+1 samples (model.py:866-871). So a branch continuation is
computable exactly as the trained model defines it — one extra `markov_head` call per node, a
rank-256 embed + [vocab, 256] GEMV — with the trained block attention **untouched**: same width 5,
same noise queries, same `get_dspark_topk_idxs` geometry. The reason `V4_DSPARK_BLOCK` width lost
(off-distribution bidirectional block attention perturbing even shallow slots) structurally cannot
recur here; a tree's candidates are alternative samples from the *same* trained per-slot
distributions.

## 2. What a tree cannot do: the cap is positions, not frames

The pipelined in-flight cap is `block + 1 = 6` because no draft exists for a position more than
block+1 ahead of the newest judged reply. A tree proposes alternatives **at those same positions**
— the horizon does not move. The BDP deficit (6 in flight against 14–17 the wire would carry) is
untouched; only a *deeper trained drafter* moves it, and §2 of `docs/V4_MULTIBLOCK_VERDICT.md`
proves chaining cannot fake one.

What a tree does buy is the cancel. The cycle model, calibrated on tonight's ring: bursts of 6
chain with period L (refill rides the drain reply; 6 × 9.5 ms = 57 ms of occupancy < 156 ms of
latency), and each cycle commits 1 + the accepted prefix. With the measured per-depth acceptance:

    E[commits/cycle] = 1 + (.93 + .7905 + .6561 + .6233 + .5236) = 4.52
    T = 4.52 / 0.156 = 29.0 tok/s        (measured tonight: 29.2 — the model is calibrated)
    hard ceiling  = 6 / 0.156 = 38.5 tok/s at 100% coverage — no tree exceeds it, ever

A branch in flight converts "cancel = commits stop at the miss" into "commits continue on the
surviving branch," worth at most 38.5/29.0 = +33% at β = 1 with unlimited frames.

## 3. The comb, priced

Best shape: main chain + at slot j an alternative token with its own Markov-conditioned linear
continuation to depth 5 (frames: 6−j extra). Per-slot value coefficient R_j — the β-multiplier of
E[commits] — from the measured acceptance profile (rescued continuations assumed to accept at the
same per-depth rates, which §1 makes plausible and the gate's `rescue_by_depth` can check):

| branch at slot j | R_j    | extra frames | R_j per frame |
|---:|---:|---:|---:|
| 1 | 0.265 | 5 | 0.053 |
| 2 | **0.458** | 4 | 0.114 |
| 3 | **0.369** | 3 | 0.123 |
| 4 | 0.060 | 2 | 0.030 |
| 5 | 0.100 | 1 | 0.100 |

Frame budget: burst occupancy must stay under L, i.e. ≤ 16 frames at τ_max 9.5 ms.

* **Full comb** (all slots): 21 frames = 200 ms > L — the ring turns compute-bound and the period
  stretches; break-even needs **β > 1.0**. Impossible. (This is the old M2.5 "trees: +15-70% g but
  3.4× compute = net loss" verdict re-deriving itself in the new regime.)
* **S = {2, 3}** (the weak slots): 13 frames = 123.5 ms < 156 ms — fits, barely. Gain =
  (0.458 + 0.369)·β / 4.52 = **+18.3% × β**.
* **S = {2}**: 10 frames = 95 ms — safe. Gain = **+10.1% × β**.

The zero-benefit floor is not zero: per-lane snapshot swaps on every stage, the tail computing
branch candidates, fatter fenced futures at cancels, and the occupancy margin shrinking 57 → 123 ms
(a τ regression to ~12 ms flips the ring compute-bound) price at **−2 to −4%**. Break-even:
**β\* ≈ 0.17–0.2**. At the 0.25–0.45 a head with 88–93% top-1 typically rescues, the tree nets
**+4 to +8%** — on the far side of real machinery:

## 4. The verifier machinery is real, and today's cannot express it

The tail's incremental frontier rule (`q == cfront+1 ∧ token == mfront`) already performs branch
*selection* untouched — a wrong-branch frame simply fails the frontier test. Everything else
cannot: the coordinator keys `sent`/`ddepth` by position (one frame per position), requires replies
in committed order, and fences by epoch alone (no cancel-by-subtree). The stages are the killer: a
branch frame at `p < _pos` triggers `Stage._seek`, which **spends** checkpoints and **clobbers the
main lane's window ring and compressor accumulators** — the next main-lane frame then either raises
or silently attends to the branch's KV. Concurrent futures need per-lane forking on every stage
(lane-tagged frames, per-lane window/accumulator/ckpt-ring state, swap on lane switch, join on
verdict) plus a coordinator restructure. The primitives exist (`_snapshot`/`_restore` already clone
per frame), but honestly: **~600–1000 lines across v4_stage/v4_pipe, on the hot path, plus a
losslessness proof over fork/join/cancel interleavings.**

## 5. The gate, shipped instead

`V4_DRAFT_TOP2` (opt-in, default OFF, tail-only, registered): the tail attaches `d2` — the head's
runner-up token per slot of the block it just drafted, read off `last_spec`'s logits, whose top-1
is provably the draft itself (`tests/test_v4_dspark.py::test_second_choices_is_the_runner_up_at_
the_drafts_own_slot` pins the bias-then-sample order this rests on). At a mismatch the missed
slot's predecessors are all committed, so its biased runner-up IS the candidate a comb would have
had in flight. Both coordinators return `rescue_by_depth: {depth: (hits, trials)}`; β̂_j =
hits/trials at slot j. Reply metadata only — no frame, block, accept rule or committed token
changes — proven bit-identical to greedy on a real socket ring, lever on and off
(`tests/test_v4_pipe.py::test_draft_top2_on_a_real_ring_is_lossless_and_measures_the_gate`).

**The decision rule, in one line:** build the tree iff

    0.458·β̂₂ + 0.369·β̂₃  ≥  4.52 × (floor cost ≈ 0.03)  ≈ 0.14      (β̂ ≥ ~0.17 uniform)

**The falsifier:** one pipelined run on the warm ring with `V4_DRAFT_TOP2=1`. If
`rescue_by_depth` at slots 2–3 reads **β̂ < 0.17, the tree is dead on this ring** — and with it the
fill family is exhausted: the remaining gap to 38.5 tok/s *is* the cap of 6, and only a deeper
trained drafter moves that.
