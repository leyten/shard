# V4 pipelined speculation — the async `coordinate_dspark`

**The highest-ceiling structural lever for the DeepSeek-V4-Flash ring.** Turn the serial, one-chunk
DSpark round into a *streamed* one: send the drafted block as `B+1` separate `s=1` frames back-to-back
so the `D=6` pipeline stages fill instead of `5/6` sitting idle, and the per-token replay penalty
disappears. This is the path from the current ~1.3 tok/s (DSpark ties greedy) to a projected **10–20
tok/s single-stream**. Designed 2026-08-01. Gated on — and unblocked by — the W-deep speculative
rollback proven in `phase0/v4_stage.py` (`_spec_ckpts`/`_seek`) and
`tests/test_v4_stage.py::test_multi_deep_rollback_across_boundaries`.

This document is the design + throughput projection + effort estimate. The correctness proof it depends
on is already landed and green; see "The gate, and why it is green" below.

---

## 1. The ceiling we are hitting

`coordinate_dspark` today (`phase0/v4_pipe.py:1208`) is a synchronous propose→verify→commit loop:

```
reset(spec+dspark) → prefill → { send [cur]+drafts as ONE chunk ; WAIT for the reply ;
                                 accept the matching prefix + 1 correction ; re-open at committed } ...
```

Two structural costs, both measured by the 7-agent research round, and both a property of *how the
block moves*, not of the model:

1. **Exactly one speculative chunk is ever in flight, so `5/6` stages are idle.** The round is a single
   request/reply against the whole `D=6`-stage ring. While stage 3 computes the chunk, stages 1,2 and
   4,5,6 have nothing to do — the ring's aggregate compute is `~1/D` utilized. The WAN round-trip is
   serial on top of that.

2. **The block travels as ONE frame and each stage replays it token-by-token** (`v4_stage.py` `forward`,
   the `start_pos>0 and s>1` branch loops per position — `phase0/v4_stage.py:538-542` region). A
   `K`-token block costs `K` sequential per-token stage computes *at every stage*. This is the
   "replay penalty," and it is **why DSpark (1.34 effective tok/s) merely ties greedy (1.43)**: the
   drafter's `g` is spent paying for the serial replay, so speculation buys nothing.

The lever removes both at once.

## 2. The lever — stream `B+1` `s=1` frames (PipeInfer / FlowSpec)

Instead of one `[cur]+drafts` chunk, stream the block as **separate `s=1` frames, back-to-back,
without waiting for each reply**:

```
frame(cur , pos=c)                 → tail computes main_hidden(c), drafts d1..dB locally (MTP), replies {m_c, [d1..dB]}
frame(d1  , pos=c+1)  ┐
frame(d2  , pos=c+2)  │  streamed the instant the draft block arrives; ≤W in flight at once
 ...                  │
frame(dB  , pos=c+B)  ┘            → tail replies {m_{c+i}} per frame as each clears the ring
```

Now stage `k` processes token `i` while stage `k-1` processes token `i+1`: the `D` stages fill, and
every frame is `s=1`, so the per-stage replay loop never runs. This is exactly PipeInfer
(SC'24, arXiv:2407.11798 — built for high-latency heterogeneous clusters, 2.15× resilient to low
acceptance) and FlowSpec (2507.02620, distributed) applied to our ring.

**The drafter is unchanged and already speculation-safe.** DeepSeek's MTP (`dspark_block_size=5`,
`n_mtp_layers=3`, target layers `[40,41,42]`, all tail-owned) predicts a **whole block of 5 tokens from
one committed `main_hidden`** — locally on the tail, no ring traversal per draft. Crucially,
`v4_dspark_draft.py` (its `test_cache_never_speculative`) proves the drafter **advances its MTP cache
only over committed positions and never un-advances it** — "a rejected draft leaves the drafter's state
untouched by construction." So the async path drafts a block off the committed frontier exactly as the
sync path does, streams it for verification, and commits the accepted prefix. **No drafter rollback is
ever needed.** The only thing that had to become multi-deep is the *layer* stage's rollback — because
now `B+1` frames are in flight when a rejection is detected, not one chunk.

## 3. The gate, and why it is green

The whole lever was blocked on one correctness question, because the speculative rollback rests on an
invariant with **zero margin** (`phase0/v4_stage.py` `_snapshot` docstring): the compressed KV regions
(`kv_cache[:, win:]` and `Indexer.kv_cache`) are deliberately **not** snapshotted, on the argument that
a slot a rejected frame poisoned is *always rewritten before its first read*. Bounded to one chunk
today, a W-deep rollback crossing a `ratio`-block **compression boundary** could have violated it and
been silently wrong in plausible numbers.

It does not. The argument is **depth-invariant**, and this is the load-bearing result:

> The read set at position `P` is `[0, (P+1)//ratio)` — a function of `P` alone. Compressed slot `j`
> enters it **exactly** at `P = q = (j+1)*ratio - 1`, the same position that writes it, for every ratio
> and at no earlier `P`. A rewind to `r = (last committed + 1)` re-processes every position `≥ r` in
> order, so a slot poisoned by a rejected frame (necessarily at some `q ≥ r`, since positions `< r`
> committed) is rewritten at its own `q` **before** that `q` reads it — however many boundaries the `W`
> rejected frames crossed. The argument is **ratio-agnostic**: proving it at ratio 4 (overlap) and
> ratio 8 (plain) on the CPU oracle covers the shipped model's ratio-4 / ratio-128 mix.

What the snapshot must carry **fully** for this to hold: the whole window ring **plus both accumulators
of every compressor, including the Indexer's own** (`kv_state`/`score_state`, all rows). The overlap
compressor (ratio 4) mutates them at each boundary via the `kv_state[:ratio] = kv_state[ratio:]` shift
(`model.py:359`); a partial clone would restore a half-shifted ring.

Proven on the CPU oracle (ground truth = sequential decode), all green:

- `test_multi_deep_rollback_across_boundaries` — streams `W` `s=1` frames, rewinds up to `W` deep across
  several ratio-4 (overlap) and ratio-8 (plain) boundaries, **NaN-poisons** the entire stale compressed region, and
  matches sequential decode **bit-for-bit** (payload *and* logits at every subsequent step) — including
  a long-prompt case where the Indexer discriminates (past `index_topk*ratio`) and a rewind **deeper
  than the window ring**.
- `test_multi_deep_rollback_mutation_check` — the anti-vacuity proof: dropping **any one** snapshotted
  region (window / `kv_state` / `score_state` / the Indexer's accumulators) makes the rollback diverge.
  So the green above is load-bearing, not an accident of harmless numbers.
- `test_rewind_deeper_than_W_refuses`, `test_commit_drops_settled_checkpoints`, and the split-chain
  case pin the bounded-`W` refusal, the commit-drop, and per-stage independence.
- `test_multi_deep_rollback_at_a_larger_ratio` (ratio 16) and `test_multi_deep_rollback_batched` (b=2)
  close the two coverage gaps that would otherwise weaken the toy-oracle→shipped-model inference:
  ratio-agnosticism beyond 4/8, and a snapshot that restores *all* batch rows.

**Independently attacked.** An adversarial verifier tried to break the claim with its own harness
(scratch, not committed): 36 cases across interleaved push/spend/re-push rollbacks, boundary-exact
targets (on vs one-past, both ratios, incl. the double boundary), deep multi-boundary rewinds with the
overlap shift inside the rejected tail, correction tokens differing from the drafts, `commit()`
off-by-one and maxlen eviction, long-context (Indexer discriminating), coarse-checkpoint replay, and 6
randomized fuzzer seeds — every one NaN-poisoned and graded against a fresh sequential oracle on both
`h` and logits. **No case diverged.** Its own harness was mutation-audited (8/8) so the green is not
vacuous. The single red it reported was a bug in its parametrization — it asked to rewind to an
*evicted* frame, and `_seek` refused correctly.

**Verdict on the gate: GO.** Multi-deep rollback is bit-exact, at any depth `≤ W`, across compression
boundaries. The lever is safe to build.

## 4. The coordinator state model (async `coordinate_dspark`)

The stage side is done: `forward` already calls `_seek(start_pos)` on entry, and `_seek` is now
`W`-deep. **A rollback needs no new stage op** — the next frame's `start_pos` *is* the rewind command,
and the `W`-deep ring lets it reach back across the in-flight frames. What changes is the coordinator:
it becomes **two cooperating threads over shared, lock-guarded state**, plus an epoch fence.

### State (guarded by one lock)

| field | meaning |
|---|---|
| `committed[]`, `c` | committed token ids; `c = len(committed)-1` = committed frontier (absolute pos) |
| `block[]`, `block_base` | the current draft block from the tail, and the position its first draft sits at |
| `sent_frontier` | absolute pos of the last frame injected at the head |
| `inflight` | `sent_frontier - c` — frames sent but not yet judged; **kept `≤ W`** |
| `epoch` | bumped on every cancel; every frame and reply carries it; stale-epoch replies are dropped |

### Sender thread

```
after prefill (commits first token, primes MTP window):
loop:
    wait until a draft block for the committed frontier is available (from the tail's cur-frame reply)
    for d,i in block:                      # stream, do not block
        wait until inflight < W
        send {op:step, ids:[[d]], start_pos: c+1+i, epoch}
        sent_frontier = c+1+i ; inflight += 1
    # block exhausted: idle until the receiver commits/cancels and posts the next block
```

### Receiver thread — **early inference cancellation**

Replies arrive in position order. For the reply of the frame at position `p` (model greedy token `m_p`,
the token that should sit at `p+1`):

```
drop if reply.epoch != epoch                      # fenced: belongs to a cancelled block
if m_p == token_we_sent_for(p+1):                 # draft accepted
    commit m_p at p+1 ; c += 1 ; inflight -= 1
    emit m_p (on_token)
else:                                             # FIRST mismatch — the truth diverged here
    commit m_p at p+1 as the correction ; c += 1  # m_p is the model's real token at p+1
    epoch += 1                                     # fence every still-in-flight frame > p+1
    inflight = 0
    post: sender must (a) send the correction frame {ids:[[m_p]], start_pos: c} → each stage
          _seek(c)s back across the discarded frames automatically, then processes it, and
          the tail drafts a FRESH block off main_hidden(c); (b) stream that block.
```

Three things make this correct and bounded:

- **The rollback is implicit and `W`-deep.** When the correction frame arrives at `start_pos = c`, each
  stage is at `_pos = sent_frontier+1` and `_seek(c)` restores the snapshot taken before position `c`
  and spends the discarded future. Because every speculative frame pushed a snapshot, and `W ≥` the max
  in-flight depth (`B+1 ≈ D ≈ 6`; default `V4_SPEC_DEPTH=16`), the snapshot at `c` still exists.
- **The epoch fence** stops a straggler reply for a discarded frame `> p+1` from being mistaken for an
  accept. Frames already in the ring when the cancel fires are still *processed* by the stages (their
  `_seek` undoes them) — that wasted work is the pipeline bubble. The **optimized** variant sends a
  one-byte `cancel(epoch)` control frame the stages honor by *skipping* queued frames of the fenced
  epoch before computing them, shrinking the bubble below `D·τ` (this is PipeInfer's "early inference
  cancellation" proper; the base design is correct without it).
- **`commit(c)`** is called on the stages as the frontier advances, dropping settled checkpoints so the
  `W`-ring's clone memory (`≈ (B+1)` × window-ring + both accumulators/compressor) stays bounded.

### Accept rule stays single-sourced

`v4_dspark_draft.plan_verify_round` is still the one accept rule. In the async loop it is applied
*incrementally* (per streamed reply) rather than per block, but it is the identical comparison —
"longest draft prefix that matches the model's greedy replies, plus one correction" — so the committed
stream is byte-identical to greedy, exactly as the sync path is. The tail advances its drafter over the
committed prefix (`n+1`) the same way; nothing about the drafter's contract changes.

## 5. Throughput projection

Model (PipeInfer's, our variables): committed tokens per rollback cycle `= R+1` where `R = a/(1-a)` is
the expected accepted run at acceptance rate `a`; cycle time `= (R + D)·τ` — `R+1` frames stream at the
pipeline's steady `τ`, plus a `~D·τ` bubble to drain the mispredicted tail and refill the pipeline
(with the fresh block's cur frame). `D = 6`, `τ` = per-token per-stage step time.

```
tok/s = (R + 1) / ((R + D) · τ)
```

| `a` | `R` | eager `τ=60ms` | graphed `τ=40ms` | grouped-MoE+graph `τ=26ms` |
|----:|----:|---------------:|-----------------:|---------------------------:|
| 0.65 | 1.86 | 6.1 | 9.1  | 14.0 |
| 0.70 | 2.33 | 6.7 | 10.0 | 15.4 |
| 0.75 | 3.00 | 7.4 | 11.1 | 17.1 |
| 0.80 | 4.00 | 8.3 | 12.5 | 19.2 |

For contrast, the **serial** path (one chunk, per-token replay, `K=4`, no stage overlap): at `a=0.7`,
`~1.9 / 2.8 / 4.3 tok/s` at the same three `τ` — and that *excludes* the WAN round-trip the sync loop
also pays per round, which is why the measured DSpark sits near 1.3. **The lever is a 3–5× structural
step even before the WAN round-trip it also removes from the critical path.**

**Landing zone: 10–12.5 tok/s at graphed `τ`, 15–19 tok/s at grouped-MoE+graph `τ`, for `a=0.7–0.8`,
`D=6` — hitting the 10–20 tok/s single-stream target.**

## 6. Honest caveats (what these numbers assume)

- **Compute must dominate the per-hop wire time.** The model is `1/τ`-bound only if a frame clears one
  stage (`τ`) faster than it crosses one WAN hop (`L`). On a scattered EU single-5090 ring `L` can rival
  `τ`; then steady throughput is `1/max(τ, L)` and the fill/drain bubble is `D·(τ+L)`, not `D·τ`.
  Pipelining *hides* latency for in-flight frames (its whole point, and PipeInfer's resilience result),
  but the table is a **compute-bound ceiling** and must be confirmed against a real ring's `L`.
- **`a = 0.65–0.8` is assumed, not measured on our path.** The serial DSpark tying greedy is consistent
  with the *replay penalty* eating `g`, not necessarily with low `a`; the pipelined path removes that
  confound, so `a` should reflect the true MTP acceptance — which DeepSeek reports as high but on their
  own infra. First real-ring measurement of `a` is the single biggest input to these numbers.
- **`dspark_block_size = 5` caps the per-block run, and that cap is HARD — see
  [`V4_MULTIBLOCK_VERDICT.md`](V4_MULTIBLOCK_VERDICT.md).** This bullet used to call multi-block
  speculation ("chain-draft the next block off the current block's predicted hidden") an upside lever
  beyond the base projection. It is not a lever; it is unreachable. There is no predicted hidden that
  can stand in for `main_hidden`, a draft therefore needs a tap only the ring can produce, and the
  reply carrying a frame's block is the same reply that advances the frontier past it — so in-flight
  is pinned at `block+1 = 6` however deep the knob is set. Measured: depths 6, 8 and 12 are
  byte-identical at `max_inflight = 6`.
  **The consequence for §5 is the ring width.** `D` is not free above 6: stages past `block+1` idle
  while still charging a hop, which is why the measured ten-box ring is *slower* than the six-box one
  (4.39 vs 5.35 tok/s). Read the `D = 6` in this section's table as the ring width to rent, not a
  starting point to scale up from.
- **`τ = 26ms` is itself a target**, contingent on the grouped-MoE kernel + CUDA graphs landing on V4;
  `40ms` graphed is the nearer-term figure and still yields 10–12.5 tok/s.

## 7. Effort estimate

| piece | status / effort |
|---|---|
| `W`-deep stage rollback (`_spec_ckpts`, `_seek`, `commit`) | **DONE, proven bit-exact** (this branch) |
| Drafter rollback | **Not needed** — drafter never persists speculative state (`test_cache_never_speculative`) |
| Async coordinator (sender + receiver threads, shared state, epoch fence, early-cancel) | **Moderate** — a new `coordinate_dspark_async` beside the sync one; ~a few hundred lines. The hard part (correctness of rollback) is retired; this is transport plumbing + the accept loop moved per-reply. |
| Stage serve loop | **Minimal** — already processes frames in order and `_seek`s via `forward`; an s=1 frame is its common case. The optional `cancel(epoch)` skip is the only new op, and only for the bubble-shrinking optimization. |
| Wire protocol | epoch on every frame/reply; the tail's cur-frame reply already carries `draft` (reused). |
| Real-ring measurement | required before trusting §5 — `a`, `L` vs `τ`, and the achieved `τ`. |

## 8. GO / NO-GO

**GO to build.** The correctness gate — the only thing that could have made this unsafe — is proven
green: multi-deep rollback is bit-exact across compression boundaries, and the mutation-check shows the
proof is not vacuous. The drafter needs no rollback. The coordinator rewrite is moderate, well-scoped
transport work with the risk retired. The projection lands in the 10–20 tok/s target on a compute-bound
ring.

**The one thing to measure first, on a real warm ring:** whether `τ` dominates the per-hop WAN latency
`L`. If it does, §5 holds and this is the single highest-ceiling lever on the V4 ring. If `L ≳ τ`,
throughput is `1/max(τ,L)` and the win is smaller but still removes the replay penalty and the serial
`5/6`-idle waste — a strict improvement over the current path in every regime.

---

## 9. WHAT WAS BUILT (2026-08-01) — `coordinate_dspark_pipelined`

Landed on `v4/pipelined-coord`, CPU-proven, opt-in. The serial `coordinate_dspark` is untouched and
still the default; `V4_PIPELINED_SPEC=1` (env) or a job's `"pipelined": true` selects the new path.

### 9.1 The wire, in full

Three fields are added to the step frame and they ride every hop (`v4_pipe._PASSTHRU`):

| field | on | meaning |
|---|---|---|
| `epoch` | every step frame of a pipelined job | the speculation generation; bumped on every cancel |
| `cpos` | every streamed frame | the settled watermark: each stage calls `Stage.commit(cpos)` |
| `fenced` | a frame a stage refused to compute | propagates, so every stage makes the same decision |

The tail's reply echoes `epoch` and `pos` (which frame it answers) and carries `acc` — the tail's own
judgment that this frame's position is committed — where the chunked reply carries `n`. A fenced frame
is answered `{"fenced": true, "epoch", "pos"}` with nothing computed. Nothing else changed: the reset
gains a `pipelined` flag beside `spec`/`dspark`, and **the serial path is byte-identical with the flag
off** (no `epoch` ⇒ no fence, no `cpos` ⇒ no commit, no extra reply keys).

`cpos = c - 1`, never `c`. The deepest rewind the coordinator can still ask for is `c` itself (a cancel
commits at some `p+1 > c` and re-opens *there*), so a checkpoint ending at or before `c-1` is provably
unreachable and everything that could cover `c` survives. This is what bounds the `W`-ring's clone
memory without a separate control frame — one integer on a frame that was going down the ring anyway.

### 9.2 The round, as implemented (and one correction to §4)

The tail advances its drafter **one position per COMMITTED frame**, using that frame's own tap and its
own greedy reply, and the block it returns proposes `q+2 .. q+B+1`. That is exactly the serial round's
`[cur] + drafts` decomposed: the coordinator commits `m_q` at `q+1` off the same reply and streams
`[m_q] + block` from `q+1`. Advancing one position at a time is bit-identical to the serial `n+1`-
position call (`advance_and_draft` loops per position internally) — pinned by
`test_pipelined_stepwise_advance_equals_one_multiposition_advance`, because if the two cadences drifted,
an acceptance rate measured on one path would say nothing about the other.

**The tail runs the accept rule too, and it needs exactly two scalars to do it.** Frames arrive in the
order they were injected, so the frame at `q` is on the committed path iff `q == cfront + 1` and its
token equals `mfront` (the greedy the tail produced at `cfront`). A rejected frame does not move the
frontier, so every frame behind it fails the same test — the poisoning of a speculative tail falls out
of the rule rather than needing a flag, including the sharp case where a later draft happens to equal
the greedy token the *rejected* frame produced.

**§4 said the cancel costs a `D·τ` drain. It does not have to.** The reply that reveals a rejection is
the reply of a *committed* frame, so the tail has already advanced its drafter over the correction token
and drafted a block off the correct history. The coordinator streams `[correction] + block` immediately:
the pipeline refills in the same breath it drains, and the only cost of a cancel is the wasted work of
the frames already in the ring. §5's `(R + D)·τ` is therefore conservative on the fill side.

### 9.3 The epoch fence — what it is, and what it honestly is not

The fence is **load-bearing in the coordinator's receiver** and nowhere else. Replies for discarded
frames are still on their way back when the correction is streamed; without the epoch they would be
read as judgments of the *new* frames at the same positions and commit the wrong tokens. Three places
act on it:

- **receiver** — pops the FIFO of frames actually sent, asserts the reply names that `(pos, epoch)`,
  and drops anything whose epoch is not current. This is the one that prevents corruption.
- **sender** — drops queued frames of a dead epoch before they reach the wire (a cancel can fire with a
  whole block still queued). Correctness never rests on winning that race: a frame that slips out is
  fenced by the receiver anyway. The check and the "owed a reply" bookkeeping happen under one lock, so
  a dropped frame can never leave the receiver waiting for an answer that will not come.
- **stages** — pass a fenced/stale frame on *without computing it*: no forward, no checkpoint, no state
  touched. **This is a guard, not PipeInfer's early-cancellation optimisation.** On a single in-band
  FIFO ring a cancel cannot overtake the frames it cancels — it is injected at the head *behind* them —
  so in normal operation no stage ever sees a stale frame, and shrinking the bubble below `D·τ` would
  need an out-of-band control path to every stage that this topology (one forward pipe, one return)
  does not have. What the guard buys is that a replaying relay, a re-dialled leg or a coordinator bug
  cannot `_seek` a stage backwards onto a discarded future; it drops, loudly and observably, instead.

`test_stale_epoch_frame_is_dropped_by_the_ring_and_changes_nothing` proves both halves on real stages:
the injected frame is answered `fenced` and the stream is unchanged, **and** with `_fenced` stubbed out
the same frame corrupts the stream — so the green is the fence working, not the frame being harmless.

### 9.4 Receipts: fenced frames are excluded from the chain, on every stage

A fenced frame does no work, so no stage calls `signer.observe` on it. That is safe *because the
decision propagates*: the first stage to fence a frame marks it, and every stage downstream honours the
marker, so all stages observe the same sequence of frames and `out_root == in_root` still holds hop for
hop. (On a FIFO ring the epochs alone would already agree — a stage that has seen epoch `e+1` got it
from its predecessor — but the marker makes it true by construction rather than by argument.) In a
normal pipelined run the ring never sees a fenced frame at all, because the sender drops them first, so
the receipt chain covers exactly the frames the serial path would have produced. `receipts_ok is True`
with `check_chain=True` is asserted on every real-ring pipelined test.

### 9.5 What was proven, on CPU

`tests/test_v4_pipe.py` §7c (protocol, no model) and §10 (real stages over the tiny checkpoint), plus
`tests/test_v4_dspark.py` §7b (the tail's frontier rule). All green, `OMP_NUM_THREADS=1`.

- **Lossless, bit-identical to the reference Transformer's own greedy stream**, in every regime:
  zero-accept (the real MTP drafter at random weights — every block rejected, so every reply cancels
  and rewinds `W` deep), full-accept (the block substituted with the reference's continuation — no
  cancel ever fires), partial accept, EOS landing mid-block with frames still in the ring, `max_new`
  landing mid-block, and **a cancel that lands exactly on a compression boundary** (positions 7 and 11
  at cpu_args' ratio-4/ratio-8 mix — the zero-margin case the whole rollback rests on).
- **Identical to the serial `coordinate_dspark` path** token for token, on the same warm ring, and the
  two interleave on one ring without either leaking state into the other.
- **Receipts settle** (`verify_coverage`, `check_chain=True`, coverage tiling all layers) on every one.
- **The pipeline fills**: `max_inflight == dspark_block_size + 1` on the full-accept path (4 at the toy
  block size of 3), i.e. the block plus the frame at the frontier are all in the ring at once, which is
  the structural claim. The in-process ring is GIL-serialised, so the wall clock there is meaningless —
  the depth is the observable, and it is reported per job (`max_inflight`, `mean_inflight`).

### 9.5b THE REFILL FLOOR (added 2026-08-02) — `V4_REFILL_FLOOR`

The shipped round above refills only when the pipe drains to the frontier, so in-flight saws
`B+1 .. 2` while the tail drafts a block on **every** committed frame and the coordinator discards
every mid-run one. `V4_REFILL_FLOOR` (coordinator lever, default 1 = the drain-only round frame for
frame) consumes a reply's block at or below the named in-flight level, streaming only positions past
the deepest frame in flight; `floor = B` pins in-flight at `block+1`. Priced at **+11..+45%** on the
07-31 ring, the spread hanging on the per-depth acceptance decay — see
`docs/V4_MULTIBLOCK_VERDICT.md` §4's correction and `phase0/v4_ngram_econ.py`. Three prices travel
with it, all instrumented rather than assumed: a mid-run block's deep drafts condition on its own
prefix, not the in-flight frames (`topup_agree/topup_disagree`, and topped-up frames scored apart in
`topup_accept_by_depth`); the family is non-monotone at the low end (floor=2 can price below
floor=1 — the marginal frames are the block's deepest); and a raised floor defeats lazy drafting by
degrees (the `_hints` licensing withholds the last `floor+1` positions, so the skip stays a provable
fact and the "tail skipped a block the round needed" raise stays sound at every floor — fully
defeated at `floor >= B`, with the bill visible as `drafts_issued` against `frames`).

### 9.6 Remaining risk for a real-ring run

1. **`a` and `L` are still unmeasured** (§6, unchanged). This build removes the confound; it does not
   answer the question.
2. **Backpressure is untested at WAN scale.** The coordinator's frames are token ids (tens of bytes) so
   its sends cannot realistically block, but the *ring* now carries `B+1` frames of `4 × dim` activation
   where it carried one chunk. Bytes/token are unchanged; bytes in flight are not.
3. **`W` vs the block.** In-flight depth is capped at `V4_SPEC_DEPTH` (16) and the stage's rollback ring
   uses the same number, so a rewind can always reach. A future continuous/multi-block speculation that
   streams deeper must raise both or it will hit `_seek`'s loud refusal.
4. **The bubble is now wasted ring work, not idle time.** After a cancel the stale frames still traverse
   and compute at every stage before the correction reaches them. That is the `D·τ` cost, and on this
   topology it can only be removed with an out-of-band cancel path (§9.3).
5. **Single sequence.** The tail's reply protocol is row 0's, as in the serial path; batching still needs
   the ragged accept path first.
