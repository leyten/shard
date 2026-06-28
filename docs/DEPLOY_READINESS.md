# Deploy-readiness — what's left to make the engine usable & deployable

> ## ⓘ CURRENT MODEL = MiniMax-M2.5 (the body below is gpt-oss-120B-era history, kept for reference)
> Served model pivoted to **MiniMax-M2.5-NVFP4** (229B-A10B, 62L GQA) on 2026-06-25. M2.5 deploy status:
>
> | Capability | Status (warm, 5×5090 EU libp2p ring) |
> |---|---|
> | Tool calling | ✅ PASS |
> | Multi-turn (9-turn recall) | ✅ PASS |
> | **Long-context (the niche product)** | ✅ **OOM FIXED 2026-06-28** — 28.7k-token prefill no-OOM (SDPA, flash+cudnn on sm_120), needle retrieved at ~14k depth |
> | Signed per-stage receipts | ✅ PASS (coverage [0:62], all sigs valid) |
> | Lossless spec sampling | ✅ (from gpt-oss work, transport-agnostic) |
> | Warm throughput | 20.63 tok/s short-ctx (session 6); 6.36 tok/s long-ctx decode |
>
> **Remaining M2.5 engine work (approved plan):** Phase 3 throughput (StaticKV + CUDA-graph — long-ctx
> decode is WAN-bound at 6.36 tok/s), Phase 4 fault-tolerance port to m25_pipe, Phase 5 multi-request
> concurrency. Network/PAY/challenge = c0mpute-side. Receipts: `docs/receipts/m25-usability-20260628.json`.
> Runbook: `phase0/DEPLOY_M25.md`.

---

Status after 2026-06-23: the concrete bar is met (28.2 tok/s decode at >100k on a copy/retrieval
task, 3-node WA/WA/TX WAN swarm, greedy-exact, signed per-stage receipts live — see STATE.md). This
is the honest gap list between "works in a niche" and "deployable general engine."

Tags: 🔴 blocking for any honest deploy · 🟡 needed to escape the niche · 🟢 scale-hardening.
**E**ngineering (shard repo) · **R**esearch · **I**ntegration (c0mpute repo / not this engine).

## 1. Latency (what users feel)
- ✅ **E — WARM libp2p speed measured = PARITY with raw-TCP (2026-06-24).** On a copy/retrieval task over an
  N=4 scattered US ring, warm libp2p decode ran **18.4–36.0 tok/s** (WAN-RTT dependent, peak 36 @ recv 171ms);
  at MATCHED WAN round-trip latency **libp2p 25.28 ≈ raw-TCP 25.55 (~1%)**, output **BIT-IDENTICAL**. The
  libp2p sidecar (per-node keys, Noise, no PSK) adds negligible per-round tax; the tok/s spread is WAN jitter
  on a genuinely cross-country ring, NOT the transport. Closes the "2.86 cold floor was meaningless" thread.
  [receipt](receipts/libp2p-warm-ab-20260624.json).
- ◑ **E — Time-to-first-token at long ctx. ASYNC SEND DONE (2026-06-23); 110k is now compute-bound.** Pipelined
  prefill landed last session (`prefill_depth` chunks in flight) but was handoff-bound at long ctx (synchronous
  24MB/chunk send). **Async inter-stage send now fixes the handoff** (per-stage `_AsyncSender` thread + 32MB socket
  buffers): same-ring A/B → **30k 153.3→60.8s (2.52×)**, **110k 245.9→210.0s (1.17×)**. At 30k the prefill is
  handoff-bound so async restores the pipeline overlap (2.5×); at 110k it's **compute-bound** (each chunk attends
  to ~110k of accumulated context, dwarfing the handoff) so async helps modestly. **`<60s@110k` is therefore a
  COMPUTE wall on 4×4090, not handoff** — more stages lowers per-stage compute (a real but bounded lever, untested
  this session). The async win would be LARGER on thin consumer uplinks (handoff-dominated). fp8/int8 KV is a
  decode/memory win, not a prefill-compute one. [receipt](receipts/async-send-ttft-20260623.json).
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
- ◑ **E — Concurrent request batching. PRIMITIVE BUILT + crux isolated (2026-06-23).** The batch=1 fixed-shape
  CUDA-graph verify was lifted to B streams (`phase0/batchverify.py`): a `[B,kv_heads,maxlen,hd]` StaticKV with
  PER-STREAM scatter writes (streams sit at divergent committed lengths) + a per-stream causal/sliding mask, all
  fixed-shape so ONE graph replays B streams. **The graph primitive is correct** (B=1 bit-exact vs FastVerify,
  intra-batch deterministic, eager==graph) and gives **real aggregate throughput on a real gpt-oss-120B block:
  1.60×@B4, 2.10×@B8** (fast/batched-MoE) or **1.24×@B8** (lossless/per-stream-MoE). A remaining gap to full
  lossless batching: the mxfp4 MoE kernel (`matmul_ogs`) is *deterministically* token-count-non-invariant
  (B=1-vs-B=2 MLP outputs diverge reproducibly) — the SAME root cause as the documented cross-K FP non-determinism.
  So the crux the goal flagged is now precisely characterized: it's the **MoE kernel, not the CUDA graph**. (The
  practical magnitude of that drift on real activations is unresolved — the single-box test uses random block
  inputs, which are pathological; a ring test with real activations is the clean measurement.) Lossless paths:
  per-stream MoE (1.24×, attention stays batched) or a token-count-invariant MoE kernel.
  [receipt](receipts/batched-verify-20260623.json).
- 🟡 **I — >1 model in the catalog** (incl. an uncensored one — the actual differentiator). Manifest/fetch supports it.

## 3. Reliability (survives a real consumer-GPU swarm)
- ◑ **R — Mid-request fault tolerance. DEMONSTRATED (2026-06-23).** Killed a middle node mid-generation under
  load → detected in ~4s, committed 189 tokens preserved, pre-warmed spare spliced in (only spare + the victim's
  predecessor relaunch; other survivors auto-re-handshake), re-prefilled prompt+committed, continued to 256
  tokens — same request, continuation byte-preserved. Engine: `coordinate_pipe(resume_ids, resumable)`; healer:
  `phase0/heal.py`. Failover ~131s (cold-spare reload dominated). [receipt](receipts/fault-tolerance-20260623.json).
- ✅ **E — HOT standby failover. DONE (2026-06-23).** The cold ~131s is now ~33s: the spare is pre-launched WARM
  (weights in VRAM, flex disk-cached → NO reload, the thing that dominated the cold path) and the victim's
  PREDECESSOR is REWIRED to it WITHOUT relaunching (healer writes `/root/.shard_next_<pred>`; `serve_spec_fast`
  re-reads it on relink). Killed a middle node mid-gen: **423 committed tokens preserved, re-prefill 8.9s, failover
  blip ~32.6s** (vs 131s cold), request completed, continuation byte-preserved. `phase0/heal_hot.py`. The residual
  ~20s of the 32.6 is the demo re-launching the coordinator (a harness artifact, not a real failover cost — a
  warm coordinator drops it to ~detect+re-prefill ≈ 13s). [receipt](receipts/hot-standby-failover-20260623.json).
  *Remaining:* "re-prefill of JUST the dropped block" via upstream activation checkpointing (matters at long ctx;
  for a short prompt the full re-prefill is already 8.9s).
- ◑ **E — HOT-standby failover OVER LIBP2P. MECHANISM DONE, end-to-end resume WIP (2026-06-24).** The hot-heal
  ported to the real permissionless transport (`phase0/heal_hot_libp2p.py`). libp2p needs a different heal than
  raw-TCP: the pred engine only ever dials its LOCAL sidecar, which survives the victim's death, so raw-TCP's
  `.shard_next` ip:port rewire never fires — the fix RELAUNCHES the predecessor's sidecar with its ring-forward
  repointed to the warm spare. PROVEN: pred relinks + a DIRECT libp2p connection to the spare + the spare
  survives the churn (after a real `specpipe.py` crash-bug fix: a bad-message handler hit `msg` before binding).
  REMAINING: the multi-hop re-handshake (head→spare→stage2→tail) doesn't deliver end-to-end over libp2p yet —
  the libp2p control-plane re-wiring after a live node substitution is the work left. The engine resume
  primitive itself is transport-agnostic (proven on raw-TCP above). [receipt](receipts/libp2p-fullstack-20260624.json).
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
