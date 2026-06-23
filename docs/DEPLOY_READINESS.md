# Deploy-readiness — what's left to make the engine usable & deployable

Status after 2026-06-23: the concrete bar is met (28.2 tok/s decode at >100k on a copy/retrieval
task, 3-node WA/WA/TX WAN swarm, greedy-exact, signed per-stage receipts live — see STATE.md). This
is the honest gap list between "works in a niche" and "deployable general engine."

Tags: 🔴 blocking for any honest deploy · 🟡 needed to escape the niche · 🟢 scale-hardening.
**E**ngineering (shard repo) · **R**esearch · **I**ntegration (c0mpute repo / not this engine).

## 1. Latency (what users feel)
- ◑ **E — Time-to-first-token at long ctx. PARTIAL (2026-06-23).** Pipelined prefill (`prefill_depth` chunks
  in flight, stages overlap) landed: **30k TTFT 105.6→55.0s (1.9×)**, **110k 226.8→193.3s (1.17×)**; even-split
  N=4 + pipelining take 110k from ~556s (old 18/9/9) to 193s. **The <60s bar is NOT met on 4×4090** — the 100k
  case is handoff-bound (the 24MB/chunk inter-stage activation send is *synchronous*, stalling overlap at long
  ctx). Next: **async inter-stage send** + more stages (the speedup scales with stage count). fp8/int8 KV is a
  decode/memory win, not a prefill-compute one. [receipt](receipts/prefill-ttft-20260623.json).
- 🟡 **E — Decode on NOVEL generation. NOT REACHABLE on this WAN topology (researched 2026-06-23).** 28 tok/s is
  copy/retrieval; novel prose at 100k is ~2–4 tok/s and **≥20 tok/s on novel gen at 100k is not achievable with
  any drop-in draft over this ring** — the wall is g×RTT and novel text caps g low; EAGLE/EAGLE-3/Medusa/MTP are
  structurally defeated (they consume the target's hidden state, which is born on the tail node a full WAN
  round-trip from the head drafter). Best *lossless* lever (modest, single-digit tok/s): a windowed/fp8-KV small
  draft (Qwen 0.5–1.5B, 50–120 MB windowed) + n-gram hybrid. Real upside only via PPSD-style early-exit
  self-speculation (one-time adapter train; LAN-proven only). **Pitch novel-long-ctx as batch/latency-tolerant,
  never interactive.** This reframes the old "biggest unlock" line: it's a topology wall, not an engine TODO.
- 🟢 **E — fp8/int8 KV + weights** to cut the per-stage 100k-attention bottleneck (the fat node is the floor).

## 2. Generality (does what people pay for)
- ✅ **E — Sampling, not just greedy. DONE (2026-06-23).** Lossless speculative *sampling* (temperature/top-p/top-k)
  at parity: deterministic-drafter rejection sampling at the tail (`shard/specsample.py`), output distribution ==
  the target's temp/top-p distribution (math TV 0.0053; on-swarm TV(spec,plain)==noise floor; 3 coherent sampled
  generations). temp≤0 stays bit-identical to greedy. [receipt](receipts/sampling-lossless-20260623.json).
- 🔴 **E — Concurrent request batching.** One stream at a time today; throughput economics need continuous batching.
- 🟡 **I — >1 model in the catalog** (incl. an uncensored one — the actual differentiator). Manifest/fetch supports it.

## 3. Reliability (survives a real consumer-GPU swarm)
- ◑ **R — Mid-request fault tolerance. DEMONSTRATED (2026-06-23).** Killed a middle node mid-generation under
  load → detected in ~4s, committed 189 tokens preserved, pre-warmed spare spliced in (only spare + the victim's
  predecessor relaunch; other survivors auto-re-handshake), re-prefilled prompt+committed, continued to 256
  tokens — same request, continuation byte-preserved. Engine: `coordinate_pipe(resume_ids, resumable)`; healer:
  `phase0/heal.py`. Failover ~131s (cold-spare reload dominated). [receipt](receipts/fault-tolerance-20260623.json).
  *Remaining for "fast":* a HOT pre-warmed standby (removes the reload → failover = just the re-prefill) and
  "re-prefill of JUST the dropped block" via upstream activation checkpointing (vs the full prompt+committed).
- 🔴 **E/I — Live heal + hot spares** (pre-warmed) so a drop is a <few-sec blip, not a cold relaunch.
- 🟡 **E — SLA behavior.** Graceful degradation + health so the orchestrator routes around flaky nodes.

## 4. It's a live permissionless network, not a hand-deployed engine
- 🔴 **I — One-command join** (deps + driver check + pull only the assigned block + register + serve), home-NAT.
- 🔴 **I — Live scheduler/control-plane** — wire `shard/scheduler.py` into the c0mpute orchestrator; swarms form
  from the live pool automatically.
- 🔴 **I — PAY** — per-node USDC on `worker_earnings`, keyed on verified receipts. The line between engine and network.

## 5. Trust enforcement (the moat bites)
- 🔴 **I — Enforce the layer-block challenge live** (random redundant recompute on a trusted node → strike).
  Primitive built (`shard/challenge.py`); policy loop is c0mpute-side.
- 🔴 **I — Stake + slash + graded reputation** the scheduler consumes (c0mpute rep is binary today).
- 🟢 **R — Crypto proof-of-compute** (ZK/commitments) to replace recompute-and-compare. Long horizon; the
  receipt's in/out-root slot is the drop-in point. Economic enforcement covers launch.

## 6. Privacy (currently an unaddressed leak)
- 🔴 **E/I — Boundary-layer pinning** — keep leaky embed/final blocks on staked/trusted nodes; untrusted
  volunteers hold only deep middle blocks.
- 🟡 **I — Per-request trusted-only routing** for sensitive jobs. Don't sell for sensitive use until done.

## 7. Economics & ops
- 🔴 **I — Metered pricing lane** (flat per-tier mis-prices a slow frontier swarm).
- 🟡 **E — Supply** (enough idle/volunteer GPUs that a swarm forms without renting).
- 🟢 **E — Production ops:** harden the supervisor, monitoring, security pass on transport + rendezvous.

---

## Minimum for a first real (niche) deploy
§3 fault tolerance · §4 join + scheduler + PAY · §5 challenge enforcement + slashing · §2 sampling · §7 pricing
→ uncensored/private long-context retrieval on volunteer GPUs, provably honest.

## To be a general alternative to centralized AI
Add §1 (TTFT + real draft for novel gen) · §2 batching · §6 privacy. Even then: competes on **access,
idle-compute cost, and trustless verification — never raw speed** (WAN latency floor). Never message "faster
than OpenAI"; the truthful pitch is "frontier models on terms they won't give you, provably honest."

## Highest-leverage ENGINE-side (shard repo) next steps
1. ✅ **Lossless speculative sampling** → real workloads, not just greedy. **DONE 2026-06-23.**
2. ◑ **Mid-request fault tolerance** → survives a real swarm. **DEMONSTRATED 2026-06-23** (cold spare; next:
   hot standby + block-only re-prefill to make the failover a <few-sec blip).
3. ◑ **Faster prefill / TTFT** → **PARTIAL 2026-06-23** (pipelined, ~2× at 30k, handoff-bound at 100k); next:
   async inter-stage send + more stages for <60s/100k.
4. ✗ **Real long-context draft for novel gen** → researched as **NOT reachable** on this WAN ring (g×RTT wall);
   reframe as batch/latency-tolerant. Modest lossless lift via windowed small-draft + n-gram hybrid; real upside
   only via PPSD early-exit self-spec (research bet).
(PAY / live scheduler / challenge-enforcement / pricing are c0mpute-repo integration, a separate effort.)
