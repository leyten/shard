# Leg 7 / 8 build plan — synthesized from the 2026-07-14 design panel

Three design passes (Leg 7 node artifact, Leg 8 control↔serving, ModelRuntime/consolidation)
were run against the post-audit codebase and converge on a single critical-path spine. This is the
sequenced plan; the per-angle detail lives in the session transcripts.

## The one thing that matters: the spine

All three legs land on the same seam — **`python -m shard.stage` backed by a real `ModelRuntime`
boundary.** Build this once and it unlocks everything else:

- **Leg 7** needs `python -m shard.stage` as the local join entrypoint (today the equivalent is an
  operator ssh harness: `m25_scatter_pipe.launch_stage`/`stage_cmd` builds a bash string and runs it
  over root ssh; `push_code` scp's 11 loose files). Promote it to a supervised CLI that takes an
  **assignment JSON** (`{lo,hi,is_tail,batch,kv_maxlen,graph_off,next,swarm_token,allow[],eng_env}`)
  on stdin and emits `READY`/health on stdout — no root-shell interpolation.
- **Mixed Torch/MLX rings** need `shard.stage` to dispatch to `M25Runtime` (cuda) vs `MlxRuntime`
  (Mac) instead of `m25_pipe` calling `m25_stage` globals directly. This is `shard/node.py`'s
  aspirational `ModelRuntime` finally becoming the live boundary (audit §4).
- **The consolidation** is the same carve: extract the runtime out of the 2000-line `m25_pipe.py`,
  behind a golden A/B receipt, ending at `StageServer(ModelRuntime)`.

The blocker that makes a mixed ring impossible today — MLX returns numpy `uint16` bf16-bits the
Torch-only codec can't encode — is closed by a **backend-neutral `WireTensor`** at the
`shard/transport.py` seam: header carries the *logical* dtype (`"bfloat16"`, not a framework type),
each side's adapter materializes natively. Byte-identical for Torch↔Torch; interoperable Torch↔MLX.

## Sequenced phases

Each phase is independently shippable and de-risks the next. The audit's C1/C2/H-tier trust-boundary
fixes are already merged (`e8b3869`), so this is the packaging + wiring on top.

### Phase A — the spine (critical path)
1. `shard/stage.py` → `python -m shard.stage`: promote `stage_cmd`/`launch_stage`; assignment-JSON in,
   `READY`/health out; ModelRuntime-by-arch dispatch; inherits loopback bind + `negotiated_max_ctx`.
   CPU-testable against a fake ring. *This is the one genuinely-new engine deliverable.*
2. `WireTensor` in `shard/transport.py` (torch path byte-identical) + torch↔numpy↔mlx round-trip test.
3. Close the pyproject gap: package `phase0` (+ normalize its flat imports) and ship the sidecar
   binary in the runtime artifact; declare real runtime deps; add the clean-env import test and the
   **required** Go-sidecar-build CI job (build failure fails CI).

### Phase B — Leg 8, make earnings real (money stays 100% on c0mpute)
The engine **already computes the settlement verdict every job** (`receipts_ok`) — it just discards
the evidence. So Leg 8 is persistence + delivery, not crypto:
1. **Job identity** — thread `swarm_id` (launcher env) + `job_id` (reuse the gateway's `chatcmpl-N`)
   into `coordinate_pipe(...)` so receipts stop stamping the `"swarm"/"job"` defaults.
2. **Durable receipt outbox** (`phase0/settle_outbox.py`) — append-only, fsync'd, written in the
   dispatcher *before* the client is told "200 OK". Record = `{swarm_id,job_id,epoch,nonce,receipts[
   signed bodies],receipts_ok,tokens{aggregate,per_stream}}` — no `$`, no accounts.
3. **Epoch-aware assignments** (correctness bomb — do not skip): `_load_assignments` caches the file
   once for process lifetime and `heal.py` splices a node with a *new* key → a **healthy healed job
   settles as fraud, nobody paid.** Give the assignment map an `epoch`, have `heal.py`/the launcher
   rewrite it, bust the cache on a new-epoch reset; a ring whose epoch ≠ file epoch fails closed.
4. **Delivery** (`phase0/settle_deliver.py`) — POST undelivered records to `SHARD_SETTLE_URL`, mark
   delivered on 2xx, retry (at-least-once; c0mpute idempotent on `(swarm_id,job_id)`).
5. **c0mpute binding** (separate repo): `job:complete` → `python -m shard.verify` (existing pinned
   CLI) → for each `stage.pubkey`, map `pubkey→account`, `recordEarning(getWorkerRevenueShare×rate×
   tokens)`. Replaces `pay=console.log`. The only step that touches money, all of it c0mpute-side.

Seams: `SHARD_ASSIGNMENTS` file (epoch-keyed) = assignment; `settle_outbox` record = receipts+tokens;
`python -m shard.verify` = settlement.

### Phase C — residential / home nodes (gated on the Ghent reachability test, task #6)
The Leg-7 panel is explicit: the residential path is a **prerequisite** for "a home machine serves
interactively," and the physics routes rather than rejects:
- **Reachability**: sidecar AutoNAT + DCUtR hole-punching + circuit-relay-v2 (audit flags relay as
  latent — reservations not renewed, default limits unfit for activation streams). QUIC/UDP makes
  hole-punching succeed through far more router types. **Decide the depth from the real Ghent-4090
  data (task #6) — do not build blind.**
- **Uplink dictates role, not admission**: the `residential-upload-ab` receipt is stark (~20 Mbps →
  a single 16k-prefill hop ≈ 2400 s). A home GPU is admitted as **seeder / verifier / batched-filler
  / tail-edge** (the probe already emits this), *not* as a prefill-forwarding interactive middle.
  Honest v0 acceptance: a home machine enrolls, is measured, seeds, verifies, fills batched capacity;
  datacenter-class boxes still anchor interactive rings.

### Phase D — consolidation (ongoing, low-risk, parallelizable)
Golden-A/B-gated carve of `m25_pipe` (Steps A→D → `StageServer(ModelRuntime)`); cut the one live
legacy tie (`shard/challenge.py` → `pipeline.run_block`, reparent onto `ModelRuntime.forward_block`)
then **quarantine** the gpt-oss stack into `phase0/legacy/` (still importable — the audit protects
"older experiments"); designate `shard/transport.py` the one codec; unify the four coordinators
(`chain/_tree/_batch/_rows`) behind one mode-dispatching coordinator and **version the protocol**
(stamp a version on the reset frame).

## Decisions flagged for leyten (genuine forks, not engine calls)
- **Packaging**: content-addressed **torrent-artifact over the existing block-exchange** (recommended)
  vs container vs pip. The wrapper is `@c0mpute/worker --shard`, not a second installer.
- Auto-update: release-key custody + canary %.
- Standby-seeding default-on + bandwidth cap + consent UX.
- Windows WSL2-only at v0.
- Worker-UI tier naming / earnings display.

## Leg-7 acceptance (from the spec)
A stranger's machine, one command, no operator ssh — enrolls, is measured, is placed by the loop,
serves with valid receipts, re-joins warm in ≤3 min. Phase A+B deliver that for a datacenter-class
box; Phase C extends it to the home/NAT tier at the roles the physics allows.
