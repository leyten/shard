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

**1. Pipeline efficiency, not compute.** Summed per-frame on-box work is 2205 ms across 6 stages
while stages report idle `recv` of 1794-2258 ms. The slowest stage is 579 ms, so the pipelined
ceiling is ~26 tok/s and we measure 3.68 — roughly **11-15% efficiency**. Around 85% of the
available throughput is lost to bubbles, which is a bigger prize than any single kernel lever.

**2. The ring is VRAM-bound (6 boxes).** Six 32 GB cards hold ~45 layers against the model's 43.
An exact max-stage-minimising planner (verified against brute force on 4000 pools, zero
mismatches) prices the best possible re-tiling at only **1.151x**; with caps lifted the same boxes
reach 1.61x better, a gap no tiling can close. Straggler **ejection is infeasible**, not merely
unhelpful: every 5-box subset holds 37 layers < 43. The fix is a **wider ring**, which lowers
max-stage-time and adds VRAM headroom in one move.

**3. Acceptance regressed.** With `V4_DSPARK_FAST=1 V4_REF_SLIM=1`, g fell **4.0 -> 3.05** serial
(3.37 pipelined), with 4 of 21 rounds accepting zero tokens. It is fully deterministic and
reproducible. Since g multiplies throughput, this cancelled most of the compute levers' gain —
the composed stack measured roughly net-neutral. Suspect: an accuracy shortcut applied
inconsistently between the draft and verify paths, so the verifier rejects its own drafter.

**4. Known correctness risk — `topk` non-determinism.** The vendored reference's `torch.topk` tie
order is undefined *and width-dependent*, and `relu_()` floors negatives to hard 0.0, manufacturing
ties. Two boxes can therefore select different compressed KV slots for the same input. A
width-invariant tie-break (value DESC, index ASC) fixed this elsewhere; the eager path still needs
auditing.

## Levers (all opt-in, default OFF)

| lever | env | status |
|---|---|---|
| Pipelined speculation | `V4_PIPELINED_SPEC=1` | **shipped, 1.94x, lossless** |
| Grouped fp4 MoE | `V4_MOE_GROUPED=1` | 3.19x eager / 11x graphed, bit-exact |
| MoE bank layout | (with grouped) | **OOMs an 8-layer stage at load — under repair** |
| Whole-layer CUDA graphs | `V4_CUDA_GRAPH=1` | needs VRAM headroom; keep OFF on the tail |
| DSpark fast draft | `V4_DSPARK_FAST=1` | 4.51x draft; implicated in the g regression |
| Ref-slim decode | `V4_REF_SLIM=1` | indexer skip; prime suspect for the g regression |

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

## Layout

| file | role |
|---|---|
| `phase0/v4_kernels_cpu.py` | CPU stand-ins for the fp4/fp8/sparse-attn kernels, so parity runs GPU-less |
| `phase0/v4_ref_cpu.py` | loads the vendored reference and builds the golden oracle |
| `phase0/v4_stage.py` | one contiguous layer range; snapshot/restore/replay for speculative rollback |
| `phase0/v4_pipe.py` | ring coordinator: greedy, serial spec, pipelined spec, receipts |
| `phase0/v4_dspark_draft.py` | the tail-local DSpark drafter and verify planning |
| `phase0/v4_plan.py` | exact max-stage tiling planner + uplink health |
