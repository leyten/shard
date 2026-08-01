# DeepSeek-V4-Flash engine — living state

The V4-Flash engine is a **separate engine** from M2.5. All of its code lives in `phase0/v4_*.py`.
The vendored reference under `phase0/deepseek_v4_ref/` is kept **byte-identical** and is driven,
never reimplemented. This file is the cross-session anchor: read it first, update it last.

## RESUME HERE (2026-08-01)

Best measured single-stream number on a real scattered ring: **3.68 tok/s**, pipelined
speculative decode, output **bit-identical to greedy**, receipts verified out-of-process.
Target is 20 tok/s. The open work is listed under "What is actually blocking 20" below.

## The model

| | |
|---|---|
| Parameters | 284B total, 13B active |
| Layers | 43 |
| Hidden | 4096 |
| Experts | 256 routed (6 active) + 1 shared |
| Precision | FP4 experts, FP8 rest |
| Licence | MIT |

Two architectural facts drive every engineering decision:

**Hyper-connections (mHC).** The inter-stage payload is `h [b, s, hc_mult=4, dim]` — four residual
streams that persist across every layer and collapse only at `hc_head`. V4 therefore sends roughly
**4x the wire bytes** of a normal pipeline-parallel model, which makes transport a first-class cost.

**DSpark.** A semi-autoregressive drafter of 3 MTP `DSparkBlock`s tapping layers 40-42, with
`dspark_block_size=5`. Because it taps the last three layers it is **tail-local by construction**:
the whole drafter must sit on the stage that owns layers 40, 41 and 42. That constraint is load
bearing — a re-tile that splits 40-42 across boxes silently breaks drafting.

## Measured on hardware (6-box Nordic RTX 5090 ring, distinct machines)

| arm | tok/s | notes |
|---|---|---|
| greedy | 1.84 - 1.90 | first token 5.6 - 7.3 s warm |
| DSpark serial | 1.05 | slower than greedy: one fat block frame per round |
| **DSpark pipelined** | **3.47 - 3.68** | **bit-identical to greedy**, 6/6 receipts valid |

Pipelined speculation is **1.94x greedy** and **3.5x serial**. Losslessness is not assumed: the
bench asserts `pipelined_equals_greedy` and it holds.

Receipts were verified **out-of-process**: 6 distinct signers, full 0-43 layer coverage, every
signature valid, and a deliberately tampered receipt is correctly rejected. Note that a receipt
must be passed through `wire_receipt()` before verification — the `stage` field is a debug label
applied *after* signing, and leaving it in puts it in the preimage and fails a valid receipt.

## What is actually blocking 20 tok/s

**0. First, two ceilings that were wrong — do not repeat them.**
A "26 tok/s ceiling / 11-15% efficiency" figure circulated all day. It was an **artefact**: it
divided a per-frame `on_box` of 579 ms by a timed window of `s = 15`. The true `s=1` bottleneck is
**147 ms**, so this ring's ceiling is **7.3 tok/s and we measure 39% of it**. Separately, any
"ceiling = 1/max-stage-time" number silently assumes a full pipe; see §3. Quote
`min(1/max_stage, (block+1)/round_latency)` or the number is fiction.

**1. Compute, not bubbles and not transport.** A validated discrete-event model (calibrated on
disjoint evidence, RMS 7.1%, predicting g / stale replies / in-flight depth across a whole depth
sweep, worst residual 15.2%) decomposes the loss as **9% fill, 54% misspeculation waste**. The
bottleneck stage is **91% busy** and `unsent_frames` is 0 — the pipe is full of *discarded* work,
not empty. RTT is 632 ms of which only **~28 ms is wire**, so the ring is **96% compute-bound** and
fp8 wire buys almost nothing on decode (it is a prefill/TTFT lever).

**Lever prices, measured, not assumed:** `d(tok/s)/dg ~ 0.27` at g=3.7 (about **9% per unit of g**,
and g caps near 5.9 with the measured decay) against **83% for a 2x cut in per-stage tau**, which
does not cap. **Compute is worth about nine times acceptance.** Reaching 20 tok/s needs draft block
>= 12 *plus* roughly **8-16x on tau**; every ring-shape lever combined ceilings out near 4.4 tok/s.

For scale: an 8-layer stage reads ~1.2 GB of fp4 weights per token, ~0.7 ms at the card's
bandwidth, against 147 ms measured — **~200x off roofline**. At batch 1 / s=1 that is small-kernel
launch latency, not bandwidth, which is exactly the regime CUDA graphs address.

**2. The ring is VRAM-bound (6 boxes).** Six 32 GB cards hold ~45 layers against the model's 43.
An exact max-stage-minimising planner (verified against brute force on 4000 pools, zero
mismatches) prices the best possible re-tiling at only **1.151x**; with caps lifted the same boxes
reach 1.61x better, a gap no tiling can close. Straggler **ejection is infeasible**, not merely
unhelpful: every 5-box subset holds 37 layers < 43. The fix is a **wider ring**, which lowers
max-stage-time and adds VRAM headroom in one move.

**3. Ring WIDTH is capped by the draft block — width and block size are ONE knob.** In-flight
frames saturate at **block_size + 1 = 6**, proven by depth 6 / 8 / 12 returning byte-identical
results on a 10-stage ring (`max_inflight` 6, `stale_replies` 41, g 4.92 every time). A 10-box ring
therefore leaves **four stages permanently idle** and measured *slower* than 6 boxes (pipelined
decode 4.39 vs 5.35; greedy 1.64 vs 2.28 — pure added hop latency). 6 stages is simultaneously the
fill cap and the VRAM floor, which is why 8/8/8/8/8/3 is the right topology. To use more width you
must raise the draft block first.

**4. Acceptance: root-caused, and it is a BUG.** g fell **4.0 -> 3.05** serial (3.37 pipelined).
Neither named lever was at fault — both are bit-exact in isolation. The mechanism is **gather
order**: `slim_indexer_forward` returns compressed slots *ascending* while the chunked verify path's
`_chunk_indexer` returns them *score-descending*. Same position, same selected SET, different ORDER
depending on whether it arrived in a verify chunk or an s=1 frame — which changes **30-35% of
attention output elements** by ~2 bf16 ULP at the shipped shape, so the verifier rejects its own
drafter. This predicts the serial-vs-pipelined split exactly. Fix: `V4_TOPK_STABLE` canonicalises
the index (a pure permutation); 32/32 positions compare equal where 0/32 did before.

Two traps in that investigation worth keeping: the effect is **0.0% at topk 24/128/192 and only
appears at 256+** (shipped is 640), so a tolerance test at toy width is structurally blind; and the
tie source is **not** `relu_()` (3 exact zeros in 719,400 candidates) but **bf16 pigeonhole** —
16.8% of candidates collide. Also note ring `accept_hist` shows keys a block-5 rule cannot produce,
so ring-g chains rounds and is not the same quantity as harness-g; compare g only within one
accounting scheme, depth, and generation length.

**5. Measure at realistic length.** At 48-64 tokens, prefill is 32-48% of wall clock, so end-to-end
tok/s understates the generation rate by ~1.7x — report `decode_tok_s` too. Acceptance is also far
better on long runs (g 10.3 at 320 tokens vs 4.0 at 48), because rounds chain across blocks.

## Levers (all opt-in, default OFF)

| lever | env | status |
|---|---|---|
| Pipelined speculation | `V4_PIPELINED_SPEC=1` | **shipped, 1.94x, lossless** |
| Grouped fp4 MoE | `V4_MOE_GROUPED=1` | 3.19x eager / 11x graphed **on CPU**; on hardware see below |
| MoE bank layout | (with grouped) | **FIXED** — was an allocation-ORDER bug, peak 31.11 -> 27.98 GiB |
| Whole-layer CUDA graphs | `V4_CUDA_GRAPH=1` | **~1.0x ALONE** — needs grouped MoE to be capturable |
| DSpark fast draft | `V4_DSPARK_FAST=1` | 4.51x draft, bit-exact; **exonerated** on the g regression |
| Ref-slim decode | `V4_REF_SLIM=1` | bit-exact in isolation; **exonerated** (see gather order) |
| Stable top-k | `V4_TOPK_STABLE=1` | canonicalises index order — the real acceptance fix |
| fp8 wire | `V4_FP8_WIRE=1` | 1.977x on the s=1 frame, but the ring is 96% compute-bound |

**The bank layout was an ORDER bug, not a leak.** Each bank was allocated *before* the experts it
replaces were released, so peak held both; the freed blocks are 256 scattered 4 MiB allocations,
the wrong shape to satisfy a 1024 MiB request. Steady-state cost of the bank is **+0 bytes** —
the bank *is* the experts. Fix is release-first (`preserve=False`). Proven with storage finalizers,
not inferred.

**Two silent holes found the same way, both worth re-checking after any refactor:** `V4_FP8_WIRE`
never reached a stage process at all (`stage_launch_cmd` hand-built the env and omitted it — now an
`ENG_ENV` allowlist, as M2.5 already had), and the grouped kernel **never fires on the drafter**,
because only `Stage.layers` gets the bank layout while `RingDrafter`'s three DSparkBlocks (10.28 GiB,
allocated lazily on the first dspark job) fall back. A lever that silently does nothing looks
exactly like a lever that does not work.

## Ring ops that cost us real time

- **Screen boxes on CPU load, and screen them twice.** This workload is CPU-launch-bound, so a
  contended host loses ~30x throughput with the GPU sitting at 0%. Read the load average a few
  minutes apart: the first read includes the box's own provisioning burst, and the 5/15-minute
  figures separate a transient from a co-tenant.
- **Place by role, not by index.** The tail runs the drafter, `lm_head` and sampling every round
  and carries the most VRAM; the head gates every round. Put the two idlest boxes there.
- **Dedup harder than `machine_id`.** Two offers with different `machine_id`, different `host_id`
  and different geolocation labels reported *identical* load triples — the same physical host
  wearing two badges. Same-IP boxes also cannot dial each other (hairpin) and will break a ring.
- **Never combine `pkill` and a launch in one ssh command** — the launch text in the same cmdline
  matches the pattern and kills what it just started. Use separate sessions.
- **The tail needs `V4_CUDA_GRAPH=0`.** Graph pools cost ~18 GB on the box that also holds the MTP
  blocks, the fp32 head and the embedding.
- **The cold-start JIT cascade is SERIALIZED across stages.** A stage compiles its tilelang kernels
  when the first frame reaches it, so on a cold ring the compiles happen one after another down the
  chain — measured on a 10-box ring at roughly 1-2 min per stage, ~12-15 min to first token, once.
  It looks exactly like a hung ring; check the stage logs for advancing per-stage timestamps before
  concluding anything is stuck. Worth pre-warming each stage with a dummy forward IN PARALLEL at
  boot, which would turn a serial 15 min into a parallel 2 min and get a wide ring measuring sooner.

## Layout

| file | role |
|---|---|
| `phase0/v4_kernels_cpu.py` | CPU stand-ins for the fp4/fp8/sparse-attn kernels, so parity runs GPU-less |
| `phase0/v4_ref_cpu.py` | loads the vendored reference and builds the golden oracle |
| `phase0/v4_stage.py` | one contiguous layer range; snapshot/restore/replay for speculative rollback |
| `phase0/v4_pipe.py` | ring coordinator: greedy, serial spec, pipelined spec, receipts |
| `phase0/v4_dspark_draft.py` | the tail-local DSpark drafter and verify planning |
| `phase0/v4_plan.py` | exact max-stage tiling planner + uplink health |
