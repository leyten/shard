# Throughput-Aware Ring Selection — Session Worklog

**Branches:** `shard/prod-improvements` · `c0mpute/receipt-verification`
**Commits:** `shard 3344c91` · `c0mpute 8d7d3c2`
**Status:** code-complete, 138 tests green ($0 offline), **not yet run on GPUs**

---

## TL;DR

The ring scheduler used to order GPUs by **minimum WAN latency**. That's the wrong
objective: it can't see GPU compute, so it would hand a slow-but-fat-VRAM card the biggest
layer block and tank throughput. This session replaced that with **throughput-aware
selection** — pick the subset + order of GPUs that maximizes *predicted tok/s* (WAN +
compute + speculative accept-rate) — and made it **learn the real fleet from completed
runs** (EWMA of per-GPU ms/layer, edge RTT, per-model accept-rate), with GPU-class priors
when cold.

Headline proof (offline, deterministic test):

```
cold  ring=[FAT]      ~77.5 tok/s   (no data → one fat card, fewest hops)
  → learns FAT is slow, the lean cards are fast →
warm  ring=[F1,F2]   ~100.0 tok/s   (switched to the fast pair)
```

---

## Why the old planner was wrong

A speculative round (coordinator drafts K tokens, the ring verifies all K in one forward
traversal, the tail returns the accept count) costs:

```
round_ms = c_out[head] + Σ L[i→i+1] + c_in[tail]      # WAN loop (the old planner did THIS)
         + Σ_stages( layers_i × ms_per_layer_i )       # COMPUTE (planner was blind to it)
         + draft_ms                                     # coordinator's local K-draft

tok/s ≈ 1000 × tokens_per_round / round_ms,   tokens_per_round ≈ accept_rate × K  (≤ K+1)
```

`topology.py` minimized only the WAN loop. Two failure modes:
1. **Selection blindness** — when the pool is oversubscribed, VRAM-greedy `allocate()` gives
   the fattest card the most layers even if it's compute-slow.
2. **No throughput number** — the orchestrator couldn't rank candidate rings or log a
   prediction.

**Key insight that kept it cheap:** for a *fixed* set of nodes the compute sum is
order-independent (total layers are fixed; each node's share × its ms/layer doesn't depend
on ring position). So ordering a fixed selection is *still* the min-latency loop — we reuse
`optimal_loop` unchanged. Compute only changes **which nodes get selected** and **the tok/s
estimate**.

---

## What changed, file by file

### shard (the engine / control-plane side)

**`shard/throughput.py`** *(new)* — the throughput model.
- `est_tok_s(round_ms, accept_rate, K)` — tokens/sec with the accept-cap (≤ K+1) and zero-guards.
- `round_ms(...)` — composes WAN loop + total compute + draft into one round's wall time.
- `best_ring(...)` — searches feasible node **subsets**, fits each (`allocate_fn`, wraps
  `scheduler.allocate`), orders it (`optimal_loop` — reused), scores predicted tok/s, returns
  the winner `{ring_order, layers, tok_s, round_ms, coordinator}`. Pool sizes are small (a
  ring is a handful of GPUs), so the subset search is exact and cheap; infeasible
  (insufficient-VRAM) subsets are pruned immediately.

**`shard/perf_store.py`** *(new)* — the "learns from real runs" half.
- EWMA (`alpha=0.3`, ~2-sample half-life) store of:
  - `ms_per_layer[node]` — per-GPU speed
  - `rtt[a→b]` — directional edge latency
  - `accept[model]` — speculative accept fraction
- **GPU-class priors** (`classify_gpu` → H100 0.18, A100 0.30, 4090 0.42, 3090 0.70, … ms/layer)
  so the *first* ring is a decent guess and every one after is tuned.
- First real sample replaces the weak class prior; subsequent samples EWMA-blend (damped,
  tracks drift like thermal throttling without unbounded history).
- Atomic JSON persistence (`os.replace`) — survives orchestrator restarts, never a
  half-written store.

**`phase0/scheduler_svc.py`** *(modified)* — the HTTP bridge.
- `POST /plan {... "objective":"tok_s"}` → `plan_tok_s()`: throughput-aware selection, returns
  the plan **plus `est_tok_s` / `est_round_ms`**.
- `POST /telemetry` → `PERF.observe_run(record)`: folds a completed run into the store.
- `--perf-store <path>` (or `SHARD_PERF_STORE`) persists learned numbers across restarts.
- **Default `/plan` is byte-identical to before** (latency-only `plan()`), and `tok_s` falls
  back to `plan()` on a 1-node pool — fully backward-compatible.

**`phase0/specpipe.py`** *(modified)* — added `rounds` to the `--dump` JSON
(`mean_accept` + `K` were already there), so `accept_rate` is fully derivable downstream.
Nothing synthesized — all genuinely measured by the coordinator.

**`phase0/throughput_test.py`** *(new)* — 29 offline tests: the est_tok_s math, the headline
"slow fat GPU loses to fast lean pair" decision, round_ms composition, PerfStore EWMA +
priors + persistence, the end-to-end `/plan` + `/telemetry` learn-then-replan loop, and
backward-compat.

### c0mpute (the network / payment side)

**`lib/orchestrator/ringScheduler.ts`** *(modified)*
- `PlanInput` gains `objective: 'tok_s' | 'latency'`, `K`, `maxStages` (forwarded to the
  service); `PlanResult` gains `estTokS` / `estRoundMs` (mapped back).
- **`reportRingTelemetry(...)`** *(new)* — fire-and-forget POST to `/telemetry`. Swallows
  errors so learning **never blocks job completion**.

**`lib/orchestrator/orchestrator.ts`** *(modified)*
- `processShardQueue` requests `objective: 'tok_s', K: 4` and logs the predicted tok/s.
- New `ringStages` map records each stage's `node_id` + `n_layers` at dispatch.
- `handleJobComplete` receives the worker's `perf` payload and posts measured telemetry
  (tokens / rounds / mean_accept / K) to `/telemetry`, then cleans up. `teardownRing` clears
  `ringStages` too.

**`c0mpute-worker/src/shard-worker.ts`** *(modified)* — `readResult` also extracts `perf`
(`n_tokens` / `rounds` / `mean_accept` / `K` / `tok_s`) from the coordinator's `--dump` file
and forwards it on `job:complete`.

**`lib/orchestrator/ringScheduler.test.ts`** *(modified)* — +10 tests (objective/K/max_stages
forwarded, est mapping, backward-compat omission, telemetry endpoint/shape/error-swallow).

---

## The closed learning loop

```
user submits shard-glm-5.2
   │
   ▼
processShardQueue → planRing(objective:tok_s) ──POST /plan──▶ scheduler_svc.plan_tok_s
   │                                                              │ uses PERF (ms/layer,
   │   ◀──────────── ring + est_tok_s ────────────────────────────  rtt, accept) + priors
   ▼
buildRingAssignments → dispatch → workers spawn sidecar + specpipe → ring serves
   │
   ▼
coordinator --dump {n_tokens, rounds, mean_accept, K, tok_s}
   │
   ▼
shard-worker readResult → job:complete {perf}
   │
   ▼
handleJobComplete → splitRingPayout (pay all stages) 
   │              └─ reportRingTelemetry ──POST /telemetry──▶ PERF.observe_run (EWMA update)
   ▼                                                              │
next ring for this model plans against the LEARNED fleet ◀────────┘
```

---

## Test status

| Suite | Count | Result |
|-------|-------|--------|
| Python: scheduler_svc | 6 | ✅ |
| Python: plan_ring | 6 | ✅ |
| Python: throughput (new) | 29 | ✅ |
| TS: shardPayout | 15 | ✅ |
| TS: ringScheduler (+10) | 27 | ✅ |
| TS: ringAssembly | 18 | ✅ |
| TS: ringIntegration | 11 | ✅ |
| TS: shard-mode | 26 | ✅ |
| **Total** | **138** | **✅ 0 tsc errors both packages** |

---

## Honest limits

- **accept_rate is learned** (specpipe measures it). **Per-GPU ms/layer uses class priors**
  until each stage timestamps its own compute — a small instrumentation add that needs the
  fleet smoke to wire and validate (can't be faked offline). The code posts only what's
  really measured and leaves ms/layer to priors until then.
- **RTT mesh is still caller-supplied / flat-seeded.** VRAM fit is correct; topology isn't
  latency-optimal until a real worker↔worker probe is wired. The throughput model consumes a
  real mesh the moment one exists — no further changes needed there.
- **Nothing has run on real GPUs.** All 138 tests are offline. The find→assemble→serve chain
  is code-complete and proven in logic, not on metal.

---

## Distance to the dream: "user picks model → c0mpute finds nearest GPU combo → just works"

Decomposed into four verbs:

| Verb | State | Gap |
|------|-------|-----|
| **select** model | ✅ works | catalog hand-curated (`SHARD_MODELS`). Dream = paste any HF repo, auto-derive layers/quant/gb-per-layer. |
| **find** the combo | ✅ **this session** | fits heterogeneous VRAM, picks throughput-best subset+order, learns the fleet. But "nearest" = best of who's *already idle*, not summon-on-demand. |
| **assemble** the ring | ✅ code-complete | proven offline (138 tests); never run on GPUs. |
| **serve** tokens | ⚠️ unproven | closed loop exists in code; no token watched end-to-end yet. |

**Three real gaps remain — none are "rethink the design":**

1. **Hardware proof (the fleet smoke).** The one thing that can't be faked offline. Small
   effort, only metered cost (~$2/hr × N). Collapses the find→assemble→serve validation in
   one run and would seed the perf store with real ms/layer instead of priors.
2. **Elastic supply.** "Finds the combo" assumes the GPUs exist in the pool. Today an
   un-fillable job just queues until enough volunteers join. Dream needs a deep pool or
   auto-rented fill-in capacity — a supply/economics problem, not a scheduling one.
3. **Fault tolerance mid-request.** A volunteer drops mid-token; today's floor is "heal by
   rebuild + re-prefill." Seamless KV migration is the genuine research item — the difference
   between "demo" and "I'd trust it."

**Score:** decision-making ~90% there. Gap to the dream is *prove it* (one hardware run),
*feed it* (supply), *harden it* (fault tolerance).

---

*Generated as a session worklog. Source of truth is the commits above.*
