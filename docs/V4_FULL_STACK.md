# V4 full stack — the composed ring launch

Every DeepSeek-V4-Flash perf lever built so far, on one branch, over the one that actually moved a
live ring: **pipelined speculation**. This is the branch the headline 6-box measurement runs from.

**The base is measured, not projected.** On a live 6-box ring, identical code, same prompt:

| arm | tok/s |
|---|---|
| greedy | 1.80 |
| DSpark, serial | 1.43 |
| **DSpark, PIPELINED** | **3.81 / 3.83** |

2.67× over serial, output bit-identical (`pipelined_equals_greedy: true`), 6/6 receipts verified
out-of-process, reproducible to 0.4%. The accept histogram shows rounds committing 7, 8 and 10 tokens
— impossible serially, where a round is capped at `block_size + 1 = 6` — so the pipeline genuinely
fills. **That build carried no compute levers at all.** This branch adds them.

Every lever is opt-in and default OFF, and two separate things follow from that, verified separately:

- **The recipe serves what the default serves** — asserted every run, in
  `tests/test_v4_full_stack.py::test_the_ring_recipe_serves_exactly_what_the_default_serves`.
- **The composed default serves what the pre-merge base served** — a one-time check against
  `v4/pipelined-coord`'s own captured selftest output, taken before the first merge and re-compared
  after the last. 0 mismatches on all five paths, both selftests. See the test matrix at the bottom.

---

## THE ENV RECIPE

### Stage-side, per box

Through the launcher (preferred — the mode is an argument, so it cannot be lost in a string):

```python
stage_launch_cmd(..., cuda_graph="whole",
                 extra_env="V4_MOE_GROUPED=1 V4_MOE_DECODE=1 V4_DSPARK_FAST=1 V4_REF_SLIM=1 ")
```

which produces, per stage:

```
V4_CUDA_GRAPH=whole V4_MOE_GROUPED=1 V4_MOE_DECODE=1 V4_DSPARK_FAST=1 V4_REF_SLIM=1
```

`stage_launch_cmd` **raises** on a `cuda_graph` value the stage would resolve to `off`, so a typo
fails at launch instead of producing a ring that runs eager while the launch line says `whole`.

### Coordinator-side

```
V4_PIPELINED_SPEC=1
```

`V4_PIPELINED_SPEC` is read by the **coordinator only** — it chooses which coordinate function drives
the job. A stage never reads it: the stage's snapshot/rollback is armed by the `spec` flag on the
**reset frame** the coordinator sends. Setting it on the stages is harmless and does nothing.

**`V4_SPEC_DEPTH` is read by BOTH sides and they must agree.** It is the coordinator's in-flight
window `W` (`coordinate_dspark_pipelined`) *and* the stage's rollback checkpoint ring depth
(`Stage._spec_ckpts.maxlen`). Both default to 16, so the safe move is to **leave it unset everywhere**.
Raise it on the coordinator alone and a rejection W frames downstream tries to rewind past the oldest
checkpoint a stage still holds — which refuses loudly rather than serving stale state, but it kills
the job. If you tune it, tune it on every box in the ring.

### MUST STAY OFF, and why

| flag | why it stays off |
|---|---|
| `V4_FAST_VERIFY` | **Structurally dead under pipelining, not merely unhelpful.** `Stage.forward` tests `start_pos == 0 or s == 1` BEFORE `_chunk_ok(s)`, and every pipelined frame is `s == 1`, so the chunk path is unreachable. Worse than a no-op: the flag also swaps in `_ChunkBlock` and reserves `_chunk_cap` extra KV rows per attention on every stage, so it costs VRAM to buy nothing. |
| `V4_REF_SLIM_NOQAT` | **The one lever in the stack that is not lossless.** It removes the reference's deliberate fp8/fp4 QAT *simulation*, so the run is strictly MORE precise than the reference and therefore a different answer — the toy config's tokens diverge from step 3. It is invisible to the selftest (ring and reference move together) and disqualifying for an arm whose claim is "bit-identical to greedy". Correct to enable only for a deployment that really does store a bf16 KV cache and has re-derived the bar. |
| `V4_DSPARK_CONF_GATE` | Two reasons. (1) `conf` is a raw logit, not a probability — the threshold has to be calibrated against the real model first (`conf_probe`). (2) It gates the tail's OFFERED BLOCK LENGTH, and the pipelined coordinator never sends a block. Setting both now **raises** rather than being ignored. |
| `V4_MOE_DECODE=0` | Never set this. It is the fallback the grouped kernel hands every shape it declines (s>1, world_size>1, hash-routed layers). |

### The A/B to run

Three arms, one variable each, all on the same warm ring:

```
1. serial baseline    (nothing set)                    -> expect ~1.43
2. pipelining only    V4_PIPELINED_SPEC=1              -> expect ~3.81   (the known-good anchor)
3. full stack         the recipe above                 -> the headline
```

Arm 2 is not optional. It is the only thing that separates "the compute levers multiplied" from "the
ring was faster today", and it is the arm whose number we already know.

### What to check in the stage log before trusting any number

Each stage prints its `__repr__` — this reports **observed** state, not the env it was handed:

```
<V4Stage [0:7) IWCIWCI head=True tail=False torch.bfloat16 on cuda:0 pos=0 kernels=tilelang
 dspark=off taps=[] spec=off/16 graph=whole fast_verify=off moe=multi>grouped>decode/7
 ref_slim=indexer levers=ok>
```

- `graph=whole` — armed. `graph=off` with a `GRAPH REFUSED ...` line above it means it declined and
  said why. `graph=off` with **no** line means the flag never arrived.
- `moe=multi>grouped>decode/7` — the WHOLE live `MoE.forward` chain, top first, plus the count of
  layers that took the bank. Every lever that rebound the forward appears. **A single name where you
  set two flags is the sixth instance of this bug**: before the chain was reported, `V4_MOE_MULTI=1`
  put `multi` on top and the status printed `moe=ref` while grouped was installed and serving.
  **`moe=...grouped/0` is the ring failure that made this branch necessary**: the flag read as on for
  every job while the bank declined on every layer.
- `ref_slim=indexer` — item 1 only. `indexer+noqat` means someone set NOQAT; stop the run.
- `levers=ok` — every requested lever was verified against LIVE state. Anything else is a finding:
  `!V4_MOE_GROUPED:mismatch` (asked for, not engaged), `!V4_LAZY_DRAFT:wrong` (a coordinator lever
  set on a stage), `!V4_TOPK_STABLE:unknown` (a `V4_*` var nothing reads). The full table is printed
  under the repr by `v4_levers.report()`; `V4_LEVERS_STRICT=1` makes any finding a refusal to serve,
  which is what a run that is about to produce a NUMBER should set.

### Which process must carry which var

`v4_levers.LEVERS` is the source of truth and `tests/test_v4_levers.py` keeps it total against the
source. Setting a coordinator lever on the stages does nothing at all — that cost a night.

| var | side | read by |
|---|---|---|
| `V4_MOE_GROUPED` / `V4_MOE_DECODE` / `V4_MOE_MULTI` (+`_MAX`) | **stage** | `v4_moe_*` at import, installed by `v4_ref_cpu.load_ref()` |
| `V4_CUDA_GRAPH` / `V4_GRAPH_MAX` | **stage** | `v4_stage` at import (graph mode is emitted by `stage_launch_cmd`, never via ENG_ENV) |
| `V4_FAST_VERIFY` (+`_MAX`) | **stage** | `v4_stage` at import |
| `V4_REF_SLIM` / `V4_REF_SLIM_NOQAT` | **stage** | `v4_ref_slim` at import |
| `V4_DSPARK_FAST` (+`V4_DSPARK_GRAPH`) | **stage (tail only)** | `v4_dspark_fast`, installed onto `DSparkTail` |
| `V4_FP8_WIRE` | **stage** | `v4_pipe._make_step_frame`, i.e. every non-tail stage packs its own output |
| `V4_KERNELS` | **stage** | `v4_kernels_cpu` at import |
| `V4_SPEC_DEPTH` | **both** | the coordinator's in-flight window AND the stage's rollback ring |
| `V4_PIPELINED_SPEC` | **coordinator** | `_coord_cli` picks the coordinator loop |
| `V4_LAZY_DRAFT` | **coordinator** | `coordinate_dspark_pipelined`; the tail reacts to frame hints, not to its own env |
| `V4_DSPARK_CONF_GATE` (+`_MIN`/`_THRESH`) | **coordinator** | `coordinate_dspark` (serial path only) |
| `V4_TOPK_STABLE` | **NOT IMPLEMENTED** | nothing. Written up as an acceptance fix; no phase0 module reads it. Setting it is reported UNKNOWN |

---

## Levers: what composes, what is exclusive

| lever | env | composes with pipelining? | measured alone | why |
|---|---|---|---|---|
| pipelined speculation | `V4_PIPELINED_SPEC=1` | **the base** | **2.67× on a live 6-box ring** | streams `s=1` frames instead of one `[cur]+drafts` block per round, so the D stages fill |
| whole-layer CUDA graphs | `V4_CUDA_GRAPH=whole` | **YES** | **2.15× on the layer** (attn core graphed, real routed MoE eager) | `_run`'s graph path needs `start_pos > 0`, `s == 1`, not replaying — exactly what a pipelined frame is |
| island CUDA graphs | `V4_CUDA_GRAPH=1` | **YES** (weaker) | 1.22× on the layer | same gate; graphs only the hc_pre/hc_post/norm islands |
| grouped fp4 MoE | `V4_MOE_GROUPED=1` | **YES** | MoE.forward 2.35×; whole stage 1.45× eager / 1.54× graphed | claims the single-token score-routed step; declines s>1 to `v4_moe_decode` |
| MoE decode fast path | `V4_MOE_DECODE=1` | **YES** (default) | — | the fallback under grouped |
| DSpark draft collapse | `V4_DSPARK_FAST=1` | **YES** | 4.51× at n=6, tail-local | rebinds `DSparkTail.advance_and_draft`, orthogonal to the wire shape |
| ref-slim, item 1 | `V4_REF_SLIM=1` | **PARTLY — read the note** | ~15-22 of ~240 launches/layer on 21 of 43 layers | rebinds `Indexer.forward`. Under `V4_CUDA_GRAPH=whole` it reaches **prefill only**, not the graphed decode steps |
| W-deep rollback ring | `V4_SPEC_DEPTH=N` | **YES** (required) | — | pipelining streams W frames before the first is judged; a rewind past the oldest checkpoint refuses loudly. Default 16 |
| chunked verify | `V4_FAST_VERIFY=1` | **NO — mutually exclusive** | — | see below |
| ref-slim, item 2 | `V4_REF_SLIM_NOQAT=1` | composes, but **not lossless** | — | removes a precision reduction; changes the stream |
| confidence gate | `V4_DSPARK_CONF_GATE=1` | **NO — refused** | uncalibrated | gates a block length the pipelined path never sends |

**Do not multiply these together and predict a number.** They are measured on different things — a
ring, a layer, a stage, a drafter call — and three of them (graphs, grouped MoE, ref-slim) all attack
the same bottleneck, CPU launch count, so they overlap rather than stack. Two numbers in particular
are routinely over-read:

- **`whole` is 2.15×, not 7.31×.** The receipt's 7.31× wall / 12.01× cpu is the ceiling measured with
  a STUB MoE; the deployable mode leaves the real routed MoE eager between two graphs.
- **grouped MoE is 2.35× on `MoE.forward`, not on the stage.** On a real multi-layer stage that lands
  as 1.45×/1.54×. The older 3.19× came from a one-layer bench with nothing else competing for the
  launch queue.

The composed number is what the ring measures. That is the point of running arm 2 as an anchor.

### The exclusivity, exactly

`Stage.forward` dispatches in this order:

```python
if start_pos == 0 or s == 1:      # prefill, or ANY single-position frame
    out = self._run(...)          # <- graphs, grouped MoE, ref-slim item 1 all live HERE
elif self._chunk_ok(s):
    out = self._run_chunk(...)    # <- the fast-verify chunk
else:
    out = cat([self._run(one position at a time) ...])   # s=1 each, so the levers apply again
```

`s == 1` is tested first, so a pipelined frame can never reach the chunk path. Asserted on the AST of
the dispatch itself, not on prose: `test_fast_verify_is_bypassed_by_every_s_equals_1_frame`.

And the chunk path, when the *serial* coordinator does reach it, bypasses three things:

- **CUDA graphs** — the graph gate requires `h.shape[1] == 1`.
- **ref-slim item 1** — `_chunk_attention` calls `_chunk_indexer(...)`, its own s-position
  reimplementation, never `self.indexer.forward(...)`. So rebinding the class method cannot reach it.
  Pinned by `test_indexer_skip_does_not_reach_the_chunked_verify_path`.
- **grouped MoE** — `grouped_forward` declines `s > 1` to the decode path.

**ref-slim item 2 is the exception that proves the rule**: it rebinds the module-level `act_quant` /
`fp4_act_quant` names, and `_chunk_attention` reads `M.act_quant(...)`, so that one *does* compose
with the chunk path. Class-method overrides are bypassed; module-global overrides are not.

This bypass is not theoretical — it surfaced as an 18-test cross-file failure. See below.

### The same bypass hits `V4_REF_SLIM` under `whole` graphs — read this before reading the A/B

`v4_whole_layer_graph` never calls `Indexer.forward` either. It passes the Indexer **module** as data
to `_indexer_decode_cs`, its own capture-safe reimplementation, because the reference's version bakes
a growing read width no graph can hold. **So under `V4_CUDA_GRAPH=whole`, rebinding `Indexer.forward`
cannot reach a graphed decode step.**

`V4_REF_SLIM=1` is still in the recipe, but for a different reason than the flag advertises:

- **Prefill takes it.** Prefill is `start_pos == 0` and the graph gate needs `start_pos > 0`, so
  prefill runs the reference `Attention.forward` → `self.indexer(...)` → the slim path. On a ring
  whose first-token latency is a real cost, that is worth having.
- **Any un-graphed layer takes it** (past `V4_GRAPH_MAX`, or a capture that failed).
- **What it does NOT buy under `whole` is the decode-step launch saving** — the graph already
  collapses those launches.

So do not read a flat ref-slim A/B under `whole` as "ref-slim did nothing". Measure it under
`V4_CUDA_GRAPH=1` (island) or off, where the indexer runs eager, if you want its decode number.
Pinned by `test_indexer_skip_does_not_reach_a_WHOLE_LAYER_GRAPHED_decode_step`.

Item 2 (noqat) is different again: `_Ref` snapshots the module-level `act_quant`/`fp4_act_quant` at
`WholeBlockGraphs` construction — after `load_ref()` installed them — so that one *does* reach a
graphed step. (Corollary worth knowing: install/uninstall of ref-slim **after** a Stage is built would
leave the graph on the old binding while eager took the new one. `load_ref()` fixes the order, so
nothing on the serve path can hit it.)

### Where pipelining and the graphs actually touch — read this before trusting a graphed ring

A pipelined cancel calls `Stage._replay` to rebuild the window ring and both compressor accumulators
over the accepted prefix. `_replay` sets `_replaying = True`, and `_run`'s graph gate excludes it. So
**a rollback rebuilds state through the EAGER path, while the frames that originally wrote that state
went through the GRAPH.**

If graphed and eager differ by one bit, every cancel leaves the stage holding state the un-cancelled
run would not have had — silently, behind valid receipts. And pipelining cancels *often*: the CPU
selftest shows 3 cancels in 4 cycles.

That makes the whole-layer graph's **Tier-1 bar (`graphed == its eager twin`, `torch.equal`, including
across bucket crossings) the correctness precondition** for running `V4_CUDA_GRAPH` together with
`V4_PIPELINED_SPEC` — not a quality nicety. Tier 2 (vs the vendored reference, tie-bounded) is *not*
sufficient on its own: an approximate-but-defensible graph would still poison every rollback. Pinned
by `test_a_pipelined_rollback_rebuilds_state_eager_under_graphs`.

**Follow-up, deliberately not taken here:** `_replay` is `s=1` per position and discards its outputs,
so it could use the graph — faster, and it would dissolve this dependency outright. That is a
behaviour change to a separately verified module and belongs in its own change, measured on its own.

---

## Merge decisions

Four merges onto `v4/pipelined-coord`. What was kept, and why:

**`Stage.__init__`** — pipelining added `spec_depth`, round 2 added `fast_verify`. Both kept; they are
independent.

**The coordinator choice** — pipelining added "which coordinator", round 2 added the `conf_*` knobs to
`coordinate_dspark`. Composed explicitly: the knobs go to the serial call, and asking for both
**raises**. A gate that silently did nothing would be measured on the ring as "confidence gating did
not help", which is the same class of lie as a graph that never captured.

**`V4_CUDA_GRAPH`** — the islands branch and the whole-layer branch each added the flag block, git
kept both copies, and the bool version sat above the mode version as dead code that read as live.
Removed. This is precisely the artifact the reachability suite exists to catch, and the merge produced
one on the first try.

**`Stage.__repr__`** — every status field, reporting **observed** state rather than declared. It is
the line an operator reads on a live ring to answer "did the lever fire?", so it reads the function
actually bound to the reference class and the count of layers that actually took the bank.

**The tie-break, reconciled from two independent triages.** `v4/whole-layer-graph-fix` and the
whole-layer branch found the same bug separately: `torch.topk`'s tie order is not width-invariant, and
`index_score` is bf16 behind a `relu_()` that floors negatives to a hard 0.0, so ties at the k-th rank
are routine (23 of 120 decode steps). A bucketed or fixed-width read could therefore select a
*different compressed slot* than the reference's narrow one.

- **Block-faithful `sparse_attn`** — taken from `e1db515`. The CPU oracle reduced in one flat pass
  over the true topk width, so padding with `-1` regrouped its pairwise tree and moved last bits on
  ~1% of calls; the real kernel walks fixed 64-wide blocks where an all-masked block rescales by
  `exp(0) == 1` and adds 0. An oracle not blocked the same way fails a parity the GPU passes. 0
  mismatches over 200 trials × 3 pad widths.
- **Width-invariant selection** — taken from the whole-layer branch, and this was a real choice. Both
  implementations select the same SET. `e1db515`'s (k-th value + cumsum admission) emits it in
  ascending INDEX order; the whole-layer branch's (stable descending sort) emits it in descending
  SCORE order — the lane order the reference's own `topk` hands `sparse_attn`. Since `sparse_attn`'s
  per-block reduction is order-sensitive, the lanes matter: at V4's real dims over 40 decode steps,
  ascending-index gave **2/40** bit-exact vs the reference, descending-score gave **40/40**.
- **Bucketing** — kept. It is the cost lever, and it is legal *only* because the selection is
  width-invariant: with `torch.topk` the answer would depend on which bucket a position landed in,
  i.e. on capture history.

---

## What the composition found

Six things, none of which any lever's own worktree could have seen, because each is a property of two
levers meeting. Two were bugs and are fixed; four are properties and are now pinned by tests.

| # | finding | kind | where it lives now |
|---|---|---|---|
| 1 | `v4_ref_slim` leaked its overrides across the whole test process | **bug, fixed** | `v4_ref_slim.uninstall()` + fixture teardown |
| 2 | `V4_CUDA_GRAPH` was defined twice after the merge; the dead bool copy read as live | **bug, fixed** | removed in `v4_stage` |
| 3 | `stage_launch_cmd` could only ever emit island mode | **gap, fixed** | `cuda_graph=` takes the mode and refuses a bad one |
| 4 | `V4_REF_SLIM_NOQAT` changes the served tokens | property | asserted divergent, excluded from the recipe |
| 5 | `V4_REF_SLIM` is bypassed by `whole` graphs (prefill only) | property | pinned; recipe note above |
| 6 | a pipelined rollback replays EAGER, so graphs need Tier-1 | property | pinned; section above |

**1. `v4_ref_slim` leaked across the whole test process.** `tests/test_v4_ref_slim.py` armed the slim
overrides in a module-scoped fixture and never removed them, and `v4_ref_cpu.load_ref()` caches the
reference module process-wide. So every later test file inherited a rebound `Indexer.forward`. That is
what produced the 18 `test_fast_verify_attends_exactly_what_the_loop_attends[2-*]` failures that
appeared **only** in a full-suite run: the LOOP took the slim fixed index while the CHUNK took the
real indexer — same key set, different gather order.

It was previously logged as an unreproducible flake needing "cumulative process state". It is not; it
reproduces deterministically from two files, `tests/test_v4_ref_slim.py` + `tests/test_v4_stage.py`.
It is **not** the topk tie-break. Fixed by giving `v4_ref_slim` an `uninstall()` and using it in the
fixture. The bypass it exposed is now pinned as a permanent test, because it is a real property of the
composition.

**2. `V4_REF_SLIM_NOQAT` changes the served tokens**, found by comparing streams *across* configs.
"ALL PASS" cannot catch this: the selftest compares each config's ring against *that config's*
reference, so a lever that moves both together passes in-config while changing what the ring serves.

---

## Known caveat, carried in but NOT introduced by this branch

`docs/receipts/v4-whole-layer-graph-20260801.json` records it as `REFERENCE_TOPK_IS_NOT_DETERMINISTIC`:
the **vendored** Indexer resolves exact score ties by an artifact of `torch.topk`'s selection
algorithm — undefined order, backend-specific, not length-invariant. So the same reference on the same
input can pick different compressed slots at different array widths, CPU thread counts, or devices.

**Scope it correctly, in both directions.**

*It does not affect this measurement.* In a pipeline-parallel ring each layer lives on exactly one
box; every position is computed at its own natural width `(p+1)//ratio`, so a replay of position `p`
uses the same width `p` used the first time; and one process means one thread count. Deterministic
within a run — which is why every arm here is reproducible and bit-identical.

*It does affect two things worth naming before anyone claims them.* (a) **Cross-box reproducibility**:
two boxes asked to recompute the same layer can legitimately disagree on a tie, so a receipt must not
be read as "anyone re-running this gets these bytes". (b) **A spot-check auditor** re-running a layer
elsewhere could reject an honest stage for no modelling reason.

`v4_whole_layer_graph._select_topk_width_invariant` pins the tie order for the **capture-safe** path.
The **eager** reference path is not pinned, and should be, before the network depends on cross-box
determinism. That is a separate change and is not on this branch.

---

## Test matrix

CPU-only, `OMP_NUM_THREADS=1`.

| suite | result |
|---|---|
| full v4 suite (`pytest tests/ -k v4`) | **392 passed, 5 skipped** (5 = CUDA capture / tilelang parity) |
| lever reachability + losslessness (`tests/test_v4_full_stack.py`) | **23 passed** |
| whole-layer Tier-1 / Tier-2 / freshness | passing (CUDA arms skip on this box) |

Both pipe selftests, every config, ALL PASS (14/14 assertions each):

| config | env | tokens |
|---|---|---|
| default | — | identical to base branch |
| pipelined | `V4_PIPELINED_SPEC=1` | identical |
| pipe + island graphs | `+ V4_CUDA_GRAPH=1` | identical |
| pipe + whole graphs | `+ V4_CUDA_GRAPH=whole` | identical |
| pipe + ref-slim | `+ V4_REF_SLIM=1` | identical |
| pipe + dspark-fast | `+ V4_DSPARK_FAST=1` | identical |
| pipe + grouped MoE | `+ V4_MOE_GROUPED=1` | identical |
| pipe + fast-verify | `+ V4_FAST_VERIFY=1` | identical (and unreachable) |
| pipe + spec-depth 4 | `+ V4_SPEC_DEPTH=4` | identical |
| serial + fast-verify | `V4_FAST_VERIFY=1` | identical |
| **the ring recipe** | all six | **identical** |
| pipe + ref-slim + NOQAT | `+ V4_REF_SLIM_NOQAT=1` | **DIVERGES from step 3 — excluded** |

24/24 runs `ALL PASS`; 22/24 also token-identical to the pre-merge base on all five paths (`ring`,
`ref`, `spec`, `dspark`, `pipe`). The 2 exceptions are the NOQAT config, by design and asserted.

## Reproduce

```
OMP_NUM_THREADS=1 python3 -m pytest tests/ -q -k v4
OMP_NUM_THREADS=1 python3 -m pytest tests/test_v4_full_stack.py -q
OMP_NUM_THREADS=1 python3 engines/deepseek_v4/v4_pipe.py selftest
OMP_NUM_THREADS=1 python3 engines/deepseek_v4/v4_pipe.py selftest-relay
```

`OMP_NUM_THREADS=1` is not decoration: the toy-config tests run ~22× faster with it, and `torch.topk`'s
tie order varies with thread count, which is one of the things the width-invariant selection removes.
