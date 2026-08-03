# V4 perf round 2 — the composed launch recipe

> **SUPERSEDED for the ring launch by [`V4_FULL_STACK.md`](V4_FULL_STACK.md).** That branch composes
> these levers onto PIPELINED speculation, adds whole-layer CUDA graphs, and fixes the grouped-MoE
> bank layout — which changes two conclusions below (marked CORRECTED). This file is kept because the
> per-lever measurements and the mutual-exclusion argument in it are still the source of record.

Four independently-built, independently-measured levers for the DeepSeek-V4-Flash engine, integrated
onto one branch and proven together on CPU. Each was verified ALONE in its own worktree; this file is
what the composition actually costs and how to drive it on the ring.

Every lever is opt-in and default OFF. With no `V4_*` set, this branch is byte-identical to master.

---

## The levers

| env | what | where it runs | measured alone | numerics |
|---|---|---|---|---|
| `V4_MOE_GROUPED=1` | routed-expert GEMMs of a decode step in ONE grouped fp4 launch | every stage, CUDA only, **s == 1 only** | MoE.forward 4.15 ms → 1.30 ms (3.19×) on a ONE-LAYER bench — on a REAL multi-layer stage 4.75 → 2.02 ms (2.35×), see the correction below | bit-exact (`torch.equal`, 3-way vs reference and vs `v4_moe_decode`) |
| `V4_DSPARK_FAST=1` | collapse the drafter's wasted intermediate forwards to a cache-advance | TAIL only (the drafter lives there) | 4.51× at n=6 (123.7 → 27.4 ms) | bit-exact, CPU + GPU |
| `V4_FAST_VERIFY=1` | chunked verify — one pass per layer instead of a per-token loop | every stage, **s > 1 only** | — (GPU round measures it) | not bit-identical to the loop (batched-GEMM reassociation) |
| `V4_CUDA_GRAPH=1` | partial CUDA graphs over the position/data-independent decode islands | every stage, CUDA only, **s == 1 only** | ~68 of ~240 launches/layer, ~+12% steady-state single-stream — and on `V4_FULL_STACK` the flag takes a MODE, with `whole` graphing the attention core too | bit-exact (replays the reference's own kernels) |
| `V4_REF_SLIM=1` | skip the Indexer's SCORING while the top-k selects every compressed slot | every stage, **via `Indexer.forward` only** | ~15-22 of ~240 launches/layer on 21 of 43 layers | selection exact; ≤1-2 bf16 ULP from gather ORDER |

`V4_MOE_DECODE` is already ON by default on master and stays on — it is the fallback the grouped
kernel hands its declined shapes to.

## READ THIS BEFORE SETTING THE ENV — two things the composition found

**1. `V4_FAST_VERIFY` and the other three levers are MUTUALLY EXCLUSIVE per position.** This is the
finding that most changes how the ring should be measured.

`Stage.forward` dispatches a chunk one of two ways. With fast-verify OFF it runs the per-token loop,
so every position is `s == 1` and graphs, grouped MoE and the slim indexer all fire. With
fast-verify ON and a chunk it accepts, it runs `_run_chunk` at `s > 1`, where:

* CUDA graphs refuse (`h.shape[1] == 1` gate),
* grouped MoE **and** the `v4_moe_decode` fast path both fall through to the true reference
  `MoE.forward`, with its host syncs (`xv.size(0) != 1` gate),
* the slim indexer is never consulted — the chunked path has its own `_chunk_indexer` that does not
  route through `Indexer.forward` at all.

On a DRAFTED ring this is not an edge case: `dspark_block_size = 5` means every drafted round sends
`s = 6`, and `_chunk_ok(6)` is true, so **nearly all decode work would take the unlevered path.**
Stacking `V4_FAST_VERIFY` on top of the others does not add its win to theirs — it replaces them.

So `V4_FAST_VERIFY` is a **competing** strategy, not an additive lever, and the ring has to pick one:
the levered per-token loop, or the unlevered chunk. Measure them head to head. The branch's own
numbers make the loop plausibly the winner at `s = 6` (the chunk pass is ~4× a single-token traversal
at *reference* cost, the loop is ~5.8× but each position gets MoE 4.15 → 1.30 ms plus ~12% from
graphs plus the indexer skip), but that is arithmetic, not a measurement — settle it on the ring.

**2. `V4_MOE_GROUPED` will DECLINE on a full stage, by design, until the loader changes.**
**~~CORRECTED — the loader DID change.~~** What follows was true of this branch and is no longer true
of `V4_FULL_STACK`; it is left in place because it is why the lever read as ON and did nothing on the
first ring, which is the failure worth remembering.

The kernel gathers experts out of a contiguous bank that is an exact DUPLICATE of the layer's
routed-expert weights — ~3.2 GiB per layer at shipped dims, beside the ~3.7 GiB already resident. A
7-layer stage on a 32 GiB card cannot hold both. The bank build therefore checks free VRAM and
declines — once per layer, with a printed line — falling back to the decode path, rather than OOMing
on the first decode token and killing the stage. On the live ring it declined on EVERY layer of every
stage, so the lever never fired at all.

The fix (`v4_moe_grouped.bank_layout`, called from `Stage.__init__` between constructing the Blocks
and `load()`) makes the routed experts BE the bank: each expert parameter is repointed at a slice of a
contiguous per-layer bank and `load_state_dict` writes through the view, so the checkpoint lands in
the bank and nowhere else. VRAM is byte-identical to the non-grouped path (10.168 GiB on a 3-layer
stage either way) and the kernel fires 7/7 and 8/8 instead of 1/7.

**The honest multi-layer numbers, which are lower than the one-layer bench:** MoE.forward 4.75 → 2.02
ms (2.35×), whole stage eager 26.9 → 18.6 ms/step (1.45×), with CUDA graphs 23.6 → 15.4 (1.54×). The
**~~3.19× eager ceiling~~** above came from a ONE-LAYER bench with nothing else competing for the
launch queue; 2.35× is what a real stage sees. Bit-exact either way (41 decode steps `torch.equal`
with the lever off, both graph settings; the s>1 paths `torch.equal` too).

Still true: the grouped kernel's ~11× / 0.377 ms number was measured with the whole MoE forward inside
a CUDA graph, and the serve path does not deliver that — `_BlockGraphs` leaves `ffn` eager BY
CONSTRUCTION, and whole-layer mode leaves the *real routed* MoE eager between two graphs for the same
reason (the reference's expert dispatch drains the device and branches in Python).

## Recipe — what to launch the GPU ring with

Stage-side (per box). `stage_launch_cmd`'s `extra_env` is placed AFTER its own `V4_CUDA_GRAPH=1`
default, so anything set here wins:

```
extra_env="V4_MOE_GROUPED=1 V4_REF_SLIM=1 V4_DSPARK_FAST=1 "
```

`V4_CUDA_GRAPH=1` needs no mention — `stage_launch_cmd(cuda_graph=True)` is the default and the ring
launch already sets it. Pass `cuda_graph=False` only for a cold-first-token A/B.

`V4_FAST_VERIFY` is deliberately NOT in that set — see (1) above. Measure it as an alternative.

Coordinator-side: nothing. The `V4_DSPARK_CONF_*` family is the only coordinator-side lever and it
stays off (below).

### The A/B ladder, so a regression is attributable

The ring is expensive to re-form, so run these as successive jobs on ONE warm ring. Steps 1-4 are
additive (all live on the `s == 1` path); step 5 is the competing arm, run LAST and ALONE.

1. baseline: graphs only (what `stage_launch_cmd` already gives you)
2. `+ V4_MOE_GROUPED=1` — check the stage log for the decline line before reading the number
3. `+ V4_REF_SLIM=1`
4. `+ V4_DSPARK_FAST=1` (drafted jobs only; it does nothing on a greedy ring)
5. **separately**: step 1 `+ V4_FAST_VERIFY=1`, with the levers of 2-4 OFF, compared against step 4

Steps 1-4 must stay token-identical to step 1 on the same prompt, with the one caveat in "numerics"
below. Step 5 is the one where a token may legitimately move.

### Precedence, and why the order is load-bearing

`v4_ref_cpu.load_ref()` installs the MoE overrides in this order, and the order IS the precedence:

```
v4_moe_decode.install(mod)      # captures the reference forward
v4_moe_grouped.install(mod)     # captures the DECODE forward
```

Each install captures whatever `MoE.forward` is bound at that moment as its own fallback, so the
chain is **grouped → decode → reference**: grouped claims the single-token score-routed decode step
and hands back what it declines (s>1, world_size>1, hash-routed layers); decode takes those; the
reference gets what neither claims. Reversed, decode would sit on top, claim that same decode step,
and the grouped kernel would be installed and unreachable. `tests/test_v4_moe_grouped.py` pins it.

So `V4_MOE_GROUPED=1` beats `V4_MOE_DECODE=1` for the decode path by construction — you do not turn
`V4_MOE_DECODE` off to use it, and you should not.

## What must stay OFF, and why

**`V4_DSPARK_CONF_GATE` (and `V4_DSPARK_CONF_THRESH` / `V4_DSPARK_CONF_MIN`) — uncalibrated.**
Confidence-gated adaptive send-length is built and tested, but the agent that built it measured that
the `conf` the tail returns is a RAW LOGIT near 0 that goes negative — it is not a probability, so
there is no threshold in [0,1] that means anything. Turning it on before a live calibration run
trims draft blocks on a number nobody has fit, which costs acceptance for no reason. It is lossless
either way (the accept rule is unchanged), so this is a throughput risk, not a correctness one.
Calibrate on a ring first: log `conf` against actual survival, then pick the floor.

**`V4_REF_SLIM_NOQAT` — APPROXIMATE, and it is not what this round is measuring.**
It drops the inplace act_quant/fp4_act_quant QAT SIMULATION — a deliberate precision REDUCTION whose
only job is to make a bf16 run match a deployment that stores an fp8/fp4 KV cache. Dropping it leaves
KV at full bf16, which is strictly MORE precision than the reference, but it is still a numerics
change: the decode logits move by a documented bounded amount and the run is no longer bit-exact
against the reference. `V4_REF_SLIM` (item 1) is gated separately and is the lossless half. Keep NOQAT
off for the lossless measurement; it is a separate, later, quality-gated experiment, and it must stay
off permanently for any deployment that really does store an fp8 KV cache.

**`V4_MOE_DECODE=0`** — never. It is the grouped kernel's fallback for every shape grouped declines.

## Numerics — what "lossless" does and does not cover

Bit-exact against the vendored reference: `V4_CUDA_GRAPH`, `V4_MOE_GROUPED`, `V4_DSPARK_FAST`,
`V4_MOE_DECODE`.

**Not** bit-exact, and both are documented as such by the branches that built them:

* `V4_REF_SLIM` — the attended SET is exact (in the select-all regime every compressed slot is
  selected, so the selection is knowable a priori and provably identical). What moves is ORDER: the
  reference's indexer returns slots score-sorted, the fixed index returns them ascending, and
  `sparse_attn`'s online softmax reduces fp32 in the given order. ≤1-2 bf16 ULP on the attention
  output. Same class as the sm_120 sparse-attn retile the ring already accepts.
* `V4_FAST_VERIFY` — batched-GEMM reassociation vs the per-token loop.

Consequence for the gate: if the ring measurement's pass condition is "tokens identical to the
reference stream", `V4_REF_SLIM` can in principle flip a token at an argmax near-tie. It did not at
CPU toy scale (below), but at 43 real layers the ULP has more chances. Grade it as "token stream
matches the graphs-only baseline", and if it diverges, diff the logits before blaming the lever.

One further interaction worth knowing if steps 3 and 5 are ever combined: `slim_indexer_forward`
returns slots ASCENDING while the chunked path's `_chunk_indexer` returns them SCORE-SORTED, so the
same absolute position attended inside a chunk and re-run eagerly (a rollback replay, or the next
`s == 1` round) reduces in different orders. That is a ULP source that exists with neither flag
alone. Another reason step 5 is run with the others off.

## The job horizon (the seam the ring has to carry)

`V4_REF_SLIM` skips the Indexer's scoring, but the Indexer's **Compressor is STATE**, not a query. It
may be skipped only for a job that provably never leaves the select-all regime — otherwise the
indexer re-engages past `index_topk * ratio` (position 2052 at the shipped config) against a
half-filled cache and picks the wrong keys, silently, in plausible-looking numbers.

So the job's horizon rides the **reset frame** that already opens a job, propagated down the ring
unchanged, and every stage calls `v4_ref_slim.set_job_max_pos` before the first step lands. The
coordinators declare an UPPER BOUND, never an estimate:

| path | declared `max_pos` |
|---|---|
| `coordinate` | `len(prompt) + max_new` (exact — greedy has no overshoot) |
| `coordinate_spec` | `+ K + 1` (the draft chunk a round puts on the wire above the committed length) |
| `coordinate_dspark` | `+ _SPEC_POS_MARGIN` (the tail's MTP block size is not knowable at reset) |

The safety direction is asymmetric and everything leans the safe way: **over-declare** → the
compressor keeps advancing → correct at any length, costs two cheap GEMMs. **Under-declare** → wrong.
**Absent** (an older coordinator's frame) → `None` → which `v4_ref_slim` already reads as "unknown",
whose safe answer is keep-advanced. A frame without the key therefore degrades to
correct-but-unoptimised, never to incorrect.

## The sm_120 quirk the grouped kernel is built around — do not "clean it up"

On this tilelang build, a per-block DATA-DEPENDENT leading index into a packed-fp4 bank does not
address correctly: uniform expert ids work, distinct ids silently collapse every slot onto one
weight (verified `[7,7]` bit-exact, `[7,3]` wrong). The kernel gathers the routed experts into a
contiguous `[G, N, K]` bank with a device-side uint8-reinterpret gather and indexes it by GRID
position — no host sync. The tidier `W[eids[g]]` dereference is the version that is wrong. Same for
`v4_dspark_fast`: its intermediate KV advance is deliberately PER-POSITION because the batched
version was bit-exact on CPU and diverged on GPU (an M=k matmul reassociates differently from k
separate M=1 ones, and the confidence head moved at k≥2).

## First-token tax

Graphs on a genuinely fresh box cost a one-time capture cascade — measured ~533 s on the first live
ring. That is NOT the graph capture (microseconds); it is tilelang JIT-autotuning and compiling the
sparse-attn / fp8 / fp4 kernels at V4's real shapes the first time each fires inside a capture
warm-up. tilelang memoises to its on-disk cache, so a re-warm on the same still-rented box reuses
them. Adding `V4_MOE_GROUPED=1` adds one more kernel to that first-token compile, at whatever
`(N, K, scale_dtype)` the shipped dims give — budget for it once per box, not once per job, and do
not read the first token's latency as the ring's latency.
