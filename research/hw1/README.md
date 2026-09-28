# HW1: the first hardware measurement session (revised for device diversity)

Supersedes the box line of BUILD-PLAN §4 ("1 × 5090, optional 4090 leg"). The vision principle is that
any device with VRAM is part of one machine, so the first session proves the engine's ops on **more than
one device family**, not just measures a 5090. Nothing here is rented until leyten says go.

## Boxes (vast, 1 GPU each, `cuda_max_good >= 13.0`, verified, direct SSH, per-session labels)

| box | family | why it is in HW1 | runs |
|---|---|---|---|
| RTX 5090 32 GB | Blackwell sm_120 | the calibration box: the M2.5 anchor and every §4.1 measurement | A0-A4, B1-B4, C, D dump |
| RTX 3090 24 GB | Ampere sm_86: **no FP8 / FP4 hardware** | MXFP4 must be dequantized in-kernel (Marlin), the FP8-block qkv is upcast to bf16 at load; the cheapest card class we want paid | C, B1-B4, D dump |
| RTX 4090 24 GB (optional) | Ada sm_89 | retires W1 R3 (does Ada keep MXFP4 packed?) and gives a third point on the speed curve | B1-B4, D dump |

The same entry host (this VPS) also runs C once, for the entry-side hashing/codec cost.

The M2.5 part stays on the 5090 only. M2.5 is the calibration vehicle and its receipts are all
sm_120; a 12-layer stage (28 GB of weights) does not fit a 24 GB card anyway. The per-family
question is about the MiMo engine, and B/D answer it directly.

Env on every box: the fleet pin `phase0/requirements_vmoe.txt` (torch 2.11.0, vLLM 0.23.0,
flashinfer 0.6.12). `bootstrap.sh setup` refuses to continue on a pin mismatch.

## Measurements (pre-registered; ids as in BUILD-PLAN §4.1)

| id | script | what |
|---|---|---|
| A0 | `m25_curve.py --mode anchor` | Re-runs the **unmodified** `research/m25_batched_moe_bench.py` (12 layers, lo 24, 16k, B 4/8) and parses its table |
| A1 | `--mode curve` | `run_block_decode_b` eager and `BatchGraphRunner` graphed. B 1..32 × ctx 1k/8k/32k (one process per ctx) × s 1/9 × MoE per-stream/batched. Records per point whether it really ran graphed (`_manual_ok`, graph cap). Plus a lognormal mixed-context variant |
| A2 | `--mode tau` | One stream, s 1/2/4/9/17, eager `run_block_decode_row` and graphed `RowGraphRunner`. r_s = τ(s)/τ(1) only within one mode |
| A3 | `--mode prefill` | `run_block_prefill_b`, n 128/512/900/2048 × start 0/8k/30k |
| A4 | `--mode kv` | index-copy of KV positions [F−Δ, F) to pinned host and back, Δ 16..32k |
| B1 | `mimo_ops.py` | fp32 router + Marlin MXFP4 experts (K3 `MarlinRoutedExperts` + shims) on layers 1 and 5, T 1..2048 |
| B2 | ″ | flashinfer JIT sink wrapper, SWA (window 128, sinks) and global. Decode B × ctx × s, plus prefill T 900/2048 |
| B3 | ″ | qkv (FP8-block dequantized to bf16 at load, per-chunk re-chunked) and o_proj GEMMs |
| B4 | ″ | the probe row's output hash over 11 batch mixes (drift expected with Marlin) |
| C | `sys_bench.py` | sha256 rows/s at 8/12/30 KiB, `shard-frame/2` round-trip at n 1/9/2048, pinned D2H/H2D |
| D | `mimo_ops.py --mode dump` + `xfamily.py` | the same fixed-input op chain on every family: per-tensor diff, router top-k agreement, speed ratio at matched ops |

Two gates run before any MiMo timing, per box:
- `sink_effect`: the attention backend's output with sinks must differ from sinks = −inf and match the
  torch reference. It catches flashinfer 0.6.12 silently ignoring `run(sinks=)` on sm_120.
- `marlin_vs_ref`: Marlin expert output must equal the dequantized reference. It catches a wrong
  MXFP4 → compressed-tensors packing, which produces fluent garbage rather than a crash.

A backend that fails its gate is replaced by the torch reference. The record carries the label.

## Pass / kill

K1-K9 are unchanged from BUILD-PLAN §4.2. `calib_fit.py` computes all of them from the results. K7 is
now evaluated per family. K10-K12 are new, for the extra device families:

| # | metric | pass | if not |
|---|---|---|---|
| K1 | A0 vs 42/26 ms @B4, 71/41 ms @B8 | within 20% | **stop**, the env is wrong |
| K2 | A1 batched gain, B=16 vs B=1, s=9, 8k, eager vs eager | ≥ 3.5× tok/ms | 2.5-3.5: S cap 16. < 2.5: fork to leyten |
| K3 | r9 (A2, 8k) | ≤ 2.5 | ≥ 3: `pertoken` becomes the low-load default |
| K4 | A1 B=1 s=1, 32k vs 1k per layer | ≤ 2× | cap S by context bucket |
| K5 | MiMo replay rate per 8-layer span | ≥ 4000 tok/s | 1500-4000: k ≥ 3, S ≤ 16. < 1000: drop the seconds-level failover claim |
| K6 | kfac | ≤ 2.0 | > 3: re-plan toward fat nodes before HW3 |
| K7 | MXFP4 resident / on-disk, **per family**, Marlin backend | ≤ 1.1× | that family's 24 GB cards leave the stage role (placement_sim Ada-penalty case) |
| K8 | hashing + codec vs A1 at B=16 | ≤ 2% | move hashing off the hot thread before M1 |
| K9 | 32k mirror import, 8-layer span | ≤ 0.5 s | bigger tier-1 fixed term in mesh_calc |
| **K10** | D: 3090 (and 4090) vs 5090, per-layer output rel diff / router top-k set agreement | ≤ 2e-2 / ≥ 0.97 | cross-class backups (ARCH §6) are restricted to the same numerics class until R4 lands |
| **K11** | D: MiMo per-layer time at T=9, 3090 / 5090 (bandwidth ratio predicts ≈ 1.9×) | ≤ 2.5× | > 3×: the 3090 serves as drafter, replica or prefill, not as a hot-path stage. It still gets a paying role |
| **K12** | every family passes `sink_effect` and `marlin_vs_ref` with the fast backend | pass | that family serves on the labelled fallback. The kernel fix becomes an R5 item before HW2 |

## Run order and box time

1. Workspace: rent with the label `hw1-<family>-<date>`, pin the iids for the dead-man switch, verify
   the instance, then scp the repo.
2. Box: `bootstrap.sh setup` and `bootstrap.sh fetch`. M2.5 [24, 36) and MiMo layers 0/1/5 download in
   parallel, ≈ 45 min on the 5090. The 3090/4090 set `HW1_FAMILY_ONLY=1` and fetch only MiMo (≈ 8.7 GB).
3. `HW1_BOX_LABEL=... bootstrap.sh run`. Order: A0 (gate K1, the script exits 2 on failure) → C →
   B + D dump → A2 → A1 (3 processes) → A3 + A4. The 5090 needs ≈ 7-9 h; a family box ≈ 2-2.5 h.
4. `bootstrap.sh collect`, then scp the tarball back.
5. Workspace: run `xfamily.py` (K10/K11) and
   `calib_fit.py results/*.json --out scratchpad/v2/calib/hw1.json`, then
   `tools/pipespec_sim.py --calib ...` and `tools/mesh_calc.py --calib ...`. Update ARCH §11
   (old → new).

Copy `scratchpad/v2/mimo/golden/t32_bf16.safetensors` to `research/hw1/golden_t32_bf16.safetensors` on
the box so B and D run on **real layer-0 outputs**. The file is untracked. Without it the results say
`inputs: random`.

## Cost (live vast quotes, 2026-09-28 19:35 UTC, read-only `vastai search offers`)

Filter: 1 GPU, verified, rentable, `cuda_max_good >= 13.0`. Prices are $/h (dph_total):

| GPU | offers | min | p25 | median |
|---|---|---|---|---|
| RTX 5090 | 49 | 0.446 | 0.538 | 0.658 |
| RTX 3090 | 28 | 0.126 | 0.188 | 0.243 |
| RTX 4090 | 52 | 0.388 | 0.471 | 0.569 |

| box | hours incl. provisioning | at p25 | at median |
|---|---|---|---|
| 5090 | ≈ 9.5 | $5.1 | $6.3 |
| 3090 | ≈ 3 | $0.6 | $0.7 |
| 4090 (optional) | ≈ 3 | $1.4 | $1.7 |
| +30% for dud boxes and re-rents | | $2.1 | $2.6 |
| **total** | | **≈ $9** | **≈ $11** |

That is about 2× the old single-box $4-8, and still well inside the balance.

## Known limits (from the self-review; the numbers carry labels, they are not hidden)

- B2 at B > 1 loops per sequence (`batched_kernel: false`). The batched attention kernel is R5's job.
  B2 at B > 1 is an upper bound.
- The D op chain has no RoPE and is not the vendor layer. It isolates kernel and format numerics per
  family. The full-layer cross-family check waits for R2 (mimo_ref promoted).
- kfac compares a MiMo eager op-sum (+10% glue) with an M2.5 eager stage per layer at B=1, s=9. Both
  sides are eager. Graph gains on MiMo are unknown until HW2.
