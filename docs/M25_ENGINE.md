# M2.5 Engine — LIVING STATE  ⟵ READ THIS FIRST every session, UPDATE IT LAST

> **The single source of truth for the sharded MiniMax-M2.5 inference engine.** Kept CURRENT (overwritten,
> not appended). History → `STATE.md`; measurements → `docs/receipts/`; per-task plans → `.claude/plans/`.
>
> **DISCIPLINE (the cross-session system):**
> 1. **Session START:** read THIS file (and the one linked plan) before touching code. Do NOT re-derive state
>    from code/research — if you feel the urge to, this doc failed; fix it instead.
> 2. **Session END:** update `RESUME HERE` + `PROVEN` + `ROADMAP` + any new `DECISION`/`OPS` lesson. Commit it.
> 3. A pointer to this file lives in auto-memory (`m25-engine-living-state`) so even a cold session finds it.

---

## 🚀 WHAT'S LEFT BEFORE LAUNCH → **`docs/LAUNCH.md`** (THE single, severity-ordered list)
> Legs 1–6 are ✅ BANKED (incl. residential home-GPU serving, 2026-07-14). The remaining launch blockers
> live in ONE place now — `docs/LAUNCH.md`. The rule: **all P0 + P1 checked → good to launch.** Stop
> tracking scattered "NEXT" lists in your head; if it's not in LAUNCH.md, it's not a launch blocker.

---

## RESUME HERE  (the one next action)

### ⇒ 2026-07-20 (LATEST-4) — P0-#5 RING-VALIDATED + KNOBS FORWARDED (PR #122); WEDGE FINDING BANKED
**Controlled-ring validation done** (3×5090 Poland/Germany/Poland, ~$3.5, receipt
`docs/receipts/eagle-watchdog-ring-20260720.json`; ring torn down, 0 boxes).
- **Integration PASS:** the shipped #120 build + forwarded knobs serve M2.5 END-TO-END over a real
  3-region WAN ring, EAGLE on, **degraded=false** (L1/L3 don't false-trip a healthy job), 3 signed
  stage receipts chained under ONE settlement nonce (hash chain intact; receiptsOk=false is CORRECT
  fail-closed on a 30/62-layer partial ring). The Leg-8 settlement path + the watchdog code coexist
  live.
- **PR #122 MERGED (knob forwarding):** M25_RET_STALL_S / M25_DRAFT_BUDGET_S / M25_JOB_STALL_S /
  M25_REPLY_TIMEOUT were NOT in the launcher's ENG_ENV → **dead in any real deployment** (a flag a
  stage never receives does nothing). Added; verified M25_RET_STALL_S=20 reaches the live tail env.
- **M1 wedge NOT reproducible on a datacenter ring — that IS the finding.** EAGLE runs depth-1 (no
  return backlog), loopback buffers autotune to 4-6MB (one partial-model aux ~166KB never fills
  them), and the vast container lacks NET_ADMIN (no tc/iptables to throttle the tail uplink). This
  is exactly WHY 07-14 only hit the residential home box: the wedge is SUSTAINED backpressure that
  needs a slow uplink. M1's socket-level bound is already proven offline (test_ret_stallguard.py:
  real serve() tail, 32KB rcvbuf + 2MB aux → a genuine sendall wedge → recovers). No further ring
  spend warranted; a true residential-drain repro needs NET_ADMIN (tc netem) or the full 62L model.

**⇒ THE ONE NEXT ACTION: back to the no-spend launch queue** — P0-#1 Leg-7 residue (manifest
resolution, `sidecar -seed`, node-side challenge sketch, warm re-join receipt) → P0-#3 relay
automation → P0-#6 churn-survival PROOF (kill a stage mid-serve in the sim → re-form → next request
serves) → P1-#4 hardening → P1-#3 WSL2 → rehearsal. Small c0mpute follow-up queued: the daemon
restarts a stall-killed coordinator with `M25_EAGLE=0` (P11 restart-degraded path). Balance ~$26.

### ⇒ 2026-07-20 (LATEST-3) — P0-#5 EAGLE WATCHDOG SHIPPED (PR #120); PAY-MODEL MERGED (c0mpute #41)
**The mitigation ladder is on master — "worst case slower, or a clean fast fail; never a silent
hang" on the deployment path.** Root cause of the 07-14 hang stays open by design (mitigation-first);
the code audit crowned a prime suspect: **the tail's UNTIMED return socket** — one send wedged
against a dead return path (relay/conntrack) parked the tail inside sendall forever, unable to
select/re-accept/reset: the whole warm ring hung behind it (and loopback's infinite bandwidth is
exactly why the CPU fake ring never reproduced it).
- **L1 (coord):** `M25_DRAFT_BUDGET_S` (1.0s default, 0 off) — an eagle-routed draft step over
  budget 2× consecutively (first exempt) or drafting dominating the wall flips the JOB to
  n-gram-only in place (HybridDrafter.disable_eagle flag-latch; depth pipelining restored; result
  carries `eagle_degraded`). The serial draft chain was the one round leg NO socket timeout sees.
- **M1 (tail):** ret gets `M25_RET_STALL_S` (180s default, 0 = old untimed) — per-PROGRESS bound
  (libp2p transport sends per-sendmsg call; slow-but-draining never trips; `sendall`'s
  total-deadline semantics verified empirically and rejected). Trip → existing _ret_send EDGE
  absorb: drop ret, keep pred+KV, re-adopt next hello_return.
- **eagle:0 reset flag:** every stage silences aux for a degraded session — the degraded arm ==
  the proven plain ring ON THE WIRE (aux ≈166KB/hop decode, ~75MB/chunk prefill = the payload
  suspect). Absent field = old behavior; all builds interop.
- **L2 (shard.coordinate):** EDGE fault while EAGLE armed → ONE degraded retry on a MANDATORY
  fresh re-dial (the tail only goes stale on a fresh hello_return — a plain reset on old sockets
  eats a late reply as its ack), resume_ids under the SAME settlement nonce, same delta state
  (no dup/gap; receipts sweep once, on the surviving attempt). EAGLE/TREE sticky-off for the
  process after (daemon restart re-arms from env); `SHARD_JOB_DONE.degraded`; `SHARD_JOB_RETRY`
  emit (deployed daemons ignore unknown tags — verified). JobRejected never retries.
- **L3 (backstop):** `M25_JOB_STALL_S` (auto = job timeout+60s) no-progress watchdog — prefill
  replies COUNT as progress (thin-uplink chunks are legally slow) — best-effort FATAL +
  UNCONDITIONAL os._exit → daemon restarts, fail-closed. Covers wedged-in-torch drafters and
  stuck sends. (Adversarial verify found+fixed D1: the emit could block its own kill.)
Verification: explore map → design panel (2) → build → independent adversarial pass. 22 new tests
(fake-ring wedge knobs, real-os._exit subprocess, real serve() tail survives a wedged ret,
losslessness across the retry); **suite 584 passed / 1 skipped**. Plan + controlled-ring runbook:
`.claude/plans/eagle-watchdog-mitigation.md`.
**Also this session: c0mpute PR #41 (pay-model) MERGED** (verified 11/11 at head first; inert
behind `SWARM_PAYOUT_ENABLED` until launch; prod untouched). Balance $29.47.

**⇒ THE ONE NEXT ACTION: the controlled-ring proof (P0-#5 residue).** Standard 4-5×5090 EU ring
(supply healthy ~$0.44-0.53/box/hr, 16 unique-IP offers), three arms per the runbook: ① master
build + mid-decode iptables DROP on the ret leg → EXPECT the 07-14 wedge (tail stuck, ring dead);
② #120 build + same fault → tail trips the stall bound, ring survives, next job serves; ③ #120 +
tc-throttled 2-5Mbps tail uplink + EAGLE → slow but never dead. Then pin the root cause on
whichever arm reproduces. Dead-man switch standing (pin iids → heartbeat → empty pin). After:
Leg-7 residue → relay automation (P0-#3) → churn proof (P0-#6) → hardening → WSL2 → rehearsal.
Small c0mpute follow-up queued: daemon restarts a stall-killed coordinator with `M25_EAGLE=0`.

### ⇒ 2026-07-20 (LATEST-2) — PAY-MODEL BUILT + DECIDED; DEV-SAFETY LESSON; NEXT = EAGLE WATCHDOG
**Pay-model (P1-#2 residue) DONE as code — c0mpute PR #41, OPEN, merges/deploys AT LAUNCH (not before).**
leyten's economics applied: USDC via the EXISTING credits/revenue-share economy (NO points ledger, NO new
token mechanics; farming unprofitable BY CONSTRUCTION via the 30%/20%-by-staking platform cut). A settled
job's COLLECTED revenue splits **flat by layers**, then **each stage keeps its OWN `getWorkerRevenueShare`**
(per-worker cut applied AFTER the split — staked 80% / unstaked 70%, never blended: a swarm is N independent
operators). One `recordEarning` per stage on the existing payout rails; swarm errors now refund. GATED behind
`SWARM_PAYOUT_ENABLED` (off) → inert until launch. `swarm-payout-test.ts` 11/11 (mixed ring pays $0.765 where
a blended cut pays $0.750 — proves per-worker). Price `pricePerMTokensUsd=$0.50/M` staged on the profile
(Kloot's call). Phase-2 per-token BILLING (make $/M drive the charge) = at/post-launch, its own deploy.
Memory: [[c0mpute-economics-applied-to-shard]].

**🛑 OPERATING LESSON (critical, leyten caught it): PROD RUNS `tsx` FROM THE WORKING DIR.** The live
services (`c0mpute-orchestrator.service`, `c0mpute-web.service`) exec straight out of
`/root/.openclaw/workspace/c0mpute` with `Restart=on-failure`. So ANY edit / branch checkout there is one
crash away from being live. **NEVER develop in the prod c0mpute tree.** All c0mpute code work now happens in
the clone **`/root/.openclaw/workspace/c0mpute-dev`** (own GitHub remote, shared `node_modules`, its OWN
empty `data/` so no dev test touches the live SQLite) → PR → and prod is only ever touched (git pull +
deliberate `systemctl restart`) in a window leyten okayed. The prod tree stays clean-on-master, always.
**Nothing deploys before the PoC is finished — the pay-model + map-live-latch land at go-live, together.**

### ⇒ 2026-07-20 (LATEST) — EPOCH BOMB DEFUSED + THE MAP IS WIRED LIVE (P1-#1 ✅, P1-#2 code-complete)
- **Assignment-EPOCH fix (c0mpute #37):** settlement now runs against a per-job `JobSettleSnapshot`
  frozen at dispatch — mid-job churn (even the daemon's own socket-recycle) can neither strand honest
  work unpaid nor hand the coordinator a fraud mark; verify-fail = a TRUE fraud signal now. Proven by
  a dedicated churn test (tail yanked mid-job, 9/9). P1-#2's residue = the pay-model $ fork ONLY.
- **Map → live state (c0mpute #38 + #39, DEPLOYED):** loopback-gated `/api/network` (public shape
  identity-free: truncated PeerIds, NO pubkeys/accounts/IPs — test-enforced, network-feed-test 19/19)
  → `scripts/network-map.ts` generator (5-min systemd timer `c0mpute-networkmap`, server-side geo via
  cached /24 lookups, jittered, IP-stripped) → `network.json` next to the page → `network.html`
  HOT-SWAPS the sim with any fresh non-empty feed (sim = the pre-launch fallback; never an empty
  globe). Live-verified on shard.c0mpute.ai; prod orchestrator restarted with #37/#38. The globe
  flips to real state automatically (≤6 min) once real daemons announce to the prod orchestrator.
- Settlement counters (per-node tokens/receipts, tokens-today, throughput window) now accumulate in
  `recordSwarmStageEarning` — the pay-model $ mapping itself remains leyten's untouched fork.

**⇒ THE ONE NEXT ACTION (per the CTO plan): EAGLE P0-#5, MITIGATION-FIRST** — build the coordinator
watchdog + degrade-to-plain-decode offline (turns "silent hang on residential tails" into "worst case
slower, never dead"), then validate on a cheap 2-3 box relay ring and only then chase root cause.
After that, known-shape work in order: Leg-7 residue (manifest resolution, `sidecar -seed`, challenge
sketch, warm re-join receipt) → relay automation (P0-#3) → churn-survival proof (P0-#6) → hardening
(P1-#4) → WSL2 (P1-#3) → rehearsal day. Balance ~$29.6; the dead-man switch is standing (pin iids at
rent, heartbeat in monitors, empty pin at teardown).

### ⇒ 2026-07-18 (LATEST) — WARM-RING DAY: suffix gate = NO-BUILD, draft_s CLOSED, LEG 8 LIVE-VALIDATED
One 6×5090 EU ring (~$10; + ~$60 idle-overnight incident → dead-man switch installed, memory
[[vast-deadman-switch]]). Receipts: **suffix-replay-verdict-20260718.json + warmring-20260718.json**.
- **Lever A (suffix drafter): DEAD at deployment shape.** 110 real on-engine traces (bit-exact ids via
  SWEEP_TRACE_DUMP): unique+structural traffic ×1.00-1.08 pess on EVERY arm (conv-8turn 1.03/1.08,
  code-w/-variants 1.01/1.03); the only wins are verbatim doc-quoting (shipped n-gram's turf) and
  byte-identical repeats (a gateway response-cache feature, ×1.3-2.9 ceiling). Bench-identical B4
  streams are bit-identical (md5) — cross-full numbers are memorization, labeled as such.
- **Lever B (async post-verify): NO-BUILD confirmed live** — draft_s 10.6-12.7 ms/round (< half the bar).
- **LEG 8 ENGINE HALF LIVE-VALIDATED:** `python -m shard.coordinate` served a real 97-token job on the
  real ring — 31 streamed deltas, 6 receipts tiling [0:62), receipts_ok under PINNED assignments, and
  EVERY stage signed the INJECTED settlement nonce (the #114 threading, proven on hardware). P1-#2
  residue = control-plane only (assignment-EPOCH + pay-model $, both leyten's).
- **The stack now rides on REAP (C) ×1.14 + opt-in cascades (D).** Fresh ring rows in the warmring
  receipt (max_new=384, not comparable to the 07-12 scorecard); ngram-vs-hybrid A/B reconfirmed.
- **NEW OPS RULES:** torch-CUDA preflight EVERY box before launch (2/8 rented duds passed nvidia-smi
  but failed torch.cuda.init); the vast dead-man switch is standing (pin iids at rent, heartbeat in
  monitors, empty pin at teardown).

**⇒ THE ONE NEXT ACTION: wire the live map (shard.c0mpute.ai) to real orchestrator state** (unchanged
from 07-17 — nothing ahead of it left in the lever queue; REAP is the next SPEND item when leyten
greenlights the eval-gated prune run, balance ~$29.6).

### ⇒ 2026-07-17 EOD (LATEST) — LEVER GATES RUN TO THE NO-SPEND LIMIT; LEG 8 NODE HALF **DONE**
**The lever stack was worked first (leyten's order) and every no-spend gate is now closed or blocked:**
- **Gate 1 (suffix drafter): tooling BANKED, verdict CORPUS-BLOCKED.** The "replay existing receipt
  traces" premise was FALSE — no receipt/log on disk carries full M2.5 generations (all head-truncated
  100-400 chars, token ids never written; Explore-verified). Shipped **#113**: `research/suffix_replay.py`
  (SuffixDecoding-faithful offline replay: local anchor-matcher + cross-request trie, honest opt/pess
  multiplier band, acceptance-depth histogram → K graph buckets) + `SWEEP_TRACE_DUMP` in both bench
  harnesses so ANY future warm-ring sweep banks a real corpus as a side effect. An API-proxy corpus path
  (exact bench prompts + agentic tool-loop episodes, HF router) is built + pipeline-proven but stalled at
  11/60 requests — **HF inference credits exhausted (402)**. Partial signal: reasoning/prose ≈×1.00-1.03
  (as predicted); the θ-routing tradeoff is sharp (θ=4 routes 12-24% at acc 2-4 = LOSES to EAGLE; θ=12
  accurate but ~2% of rounds). tools/code/agentic = the deciding arms, still unmeasured. **Unblock =
  ~$2-5 HF credit top-up (~30 min to verdict) OR the next warm ring** (runbook:
  `.claude/plans/suffix-replay-gate.md`).
- **Gate 2 (draft_s): resolved NO-BUILD-NOW** — the 07-10 receipt already bounds drafting under the
  25 ms bar (B1 round 212 ms INCLUDING drafting; B4 <150 ms / 4 streams). The confirming `draft_s`
  timer read piggybacks on the next warm ring. PEARL-style async post-verify stays unbuilt.
- **Gate 3 (REAP): NVFP4 prune path structurally VALIDATED, rest is spend-gated** — 62L × 256 experts,
  top-8; every expert = a self-contained tensor group (w1/w2/w3 + weight_scale/_2 + input_scale), gates
  unquantized bf16 → prune = drop expert groups + shrink gate rows + rewrite config. Saliency calibration
  + τ²/BFCL-style agentic eval gate need a GPU (vast, $0 today).

**LEG 8 NODE/ENGINE HALF — DONE + PROVEN (shard #114 + c0mpute #36):** request → served → settled works
END-TO-END with real daemons, GPU-less. `python -m shard.coordinate` (#114) = the gateway's
coordinate_pipe driving as a stdin/stdout CLI (SHARD_COORD_READY / SHARD_JOB_TOKEN / SHARD_JOB_DONE /
SHARD_JOB_FATAL; ring-dial retries at boot). **Settlement-nonce threading was the correctness catch:**
c0mpute's settleJob verifies receipts with `expected_nonce` = the swarm:job nonce, but coordinate_pipe
self-minted — all four coordinators gained `job_nonce=None` so stages sign exactly what settlement
checks (default self-mint unchanged; fake-ring-proven: nonce lands in the reset op verbatim, joined
deltas == final response, job-fault isolation; 563 tests green). c0mpute #36: `CoordinatorProcess` seam
(supervised, NDJSON stdin, complete-lines-only parsing), the RETURN TUNNEL closed exactly like
m25_scatter_pipe's proven wiring (head sidecar -forwards RETURN_PORT=base+12 to the tail's sidecar; tail
allows the head PeerId; ZERO tail-engine changes — hello_return classification already existed),
`swarm:job` handler (fail-closed completes: no server error event exists), shim coordinator that PROBES
the return tunnel with a framed roundtrip before READY, and `shard-daemon-sim --serve
[--accept-receipts]` (+ `SERVE=1 npm run try-shard`). **2-daemon proof: auto-form → dispatch →
return-tunnel roundtrip → streamed deltas (stream==response) → complete → settlement credits BOTH stages
by layers.** leg8-serve-test still 10/10.

**⇒ THE ONE NEXT ACTION: wire the live map (shard.c0mpute.ai) to real orchestrator state** — both
blockers (auto-form #34, serving #35+#36) are gone; the map is still a simulation. Then: remaining
daemon edges (challenge sketch, warm re-join receipt, `sidecar -seed`, manifest resolution, relay
auto-discovery P0-#3) + P1-#4 hardening. **NEXT WARM RING checklist (whenever vast credit returns):**
launch the sweep with `SWEEP_TRACE_DUMP` (suffix corpus) + read `draft_s` (gate 2) + validate leg-8
serving on a real ring + EAGLE hang P0-#5 (leyten deferred). **leyten forks flagged:** ① ~$2-5 HF credit
= the suffix verdict this week; ② vast top-up = REAP (the strongest lever) + live validation; ③
assignment-EPOCH settlement fix still open (P1-#2 correctness bomb); ④ pay-model $ mapping (stub logs).

### ⇒ 2026-07-17 (leyten's call: perf levers BEFORE every other leg) — THE VERIFIED LEVER STACK
Source: the Inkling-spike 50-agent lever hunt, M2.5 subset corrected by a 3-lens adversarial panel →
**`docs/research/m25-lever-stack-verified-20260716.md`** (in-repo; hunt provenance lives on branch
`spike/inkling-5090`, clone `shard-inkling`, `docs/research/inkling-lever-hunt-20260716.md`). Memory:
[[m25-lever-stack-verified]]. Projection (exact tier, vs the 07-12 K-tuned scorecard): **tools 29.7→34-42,
code 23.2→29-36, reasoning 29→32-38, qa 21.1→24-28, prose 18.5→20-22 (bar ✓ from REAP alone),
mix 17.9→19.5-21.5, B1 22.5→31-40; opt-in labeled cascade tier ~×1.6-2.0 total.** Zero training anywhere.
Order (cheapest gate first; 1 is no-spend):
1. **Suffix-tree drafter — OFFLINE trace replay first** (half-day, no engine change, no spend): replay
   existing receipt traces through a suffix tree (SuffixDecoding, arXiv 2411.04975) → measures real match
   lengths on OUR traffic and decides ×1.2 vs ×1.4 on tools/code. Build behind the HybridDrafter seam only
   if ≥×1.2. It competes with EAGLE (chain g already 4.5-5.0) — NOT with n-gram (mix-B4-ngram g=0.01).
   Depth must be ADAPTIVE (deep only on confident matches — the K8-dead-slot receipt is the warning).
2. **`draft_s` check** (one warm-ring run, reads the existing coordinate_pipe timer): build async
   post-verify drafting ONLY if the serial draft step ≥~25 ms/round (expected 8-16 ms → likely skip;
   PEARL's ×1.3-1.5 does NOT transfer, our drafting tax is already dead).
3. **REAP12 expert prune → 5-hop rings (×1.14 banked — the STRONGEST lever)**: method-general, published
   ≤1.5-pt agentic deltas @25% on M2-family; hop→wire physics receipt-validated on our own ring (4-hop
   measured ×1.33 vs ×1.32 predicted). Gate on τ²/BFCL-style AGENTIC evals (not ppl) + validate the NVFP4
   prune path. **Closes the prose 20-bar by itself.** Stretch: REAP25 → 4 hops (~×1.3, zero-headroom fit).
   Also a ÷1.2-1.5 fleet-cost lever.
4. **Speculative cascades (top-k accept) = opt-in LABELED tier** (+×1.15-1.35, most value on prose):
   env-gated accept mode; k=1 must reproduce greedy bit-exact; receipts carry the acceptance rule.
DEAD — do not build (panel-confirmed): trellis sub-4-bit for resident M2.5 (wire is RTT-floor-dominated,
compute penalty eats the win), trees (still, until tree-frame CUDA graphs), DFlash (≈EAGLE parity;
half-day A/B at most). _(Leg 8 node/engine half = immediately after this stack; EAGLE hang P0-#5 still
deferred by leyten — real vast spend. NOTE: vast credit is $0 — gate 1 needs none.)_

### ⇒ 2026-07-15/16 — LEG 7 DONE (self-serve join), MAP LIVE, LEG 8 SERVER-HALF SHIPPED
Big session. Three fronts advanced; launch list re-synced (`docs/LAUNCH.md`, 2026-07-16).

**LEG 7 (node daemon) — effectively complete.** `npm run try-shard` (a stranger's box) self-provisions
engine+venv+sidecar+weights with ZERO env vars → enroll → announce → assign → verified pull → READY → serving;
multi-stage rings FORM (2-node libp2p ring proven). Shipped: shard **#103/#104/#106/#108**, c0mpute
**#27/#28/#29/#30/#31/#32**. Verified peers-first fetch = `python -m shard.fetch` (#108) wired into the daemon
(#32). Auto-update REMOVED (leyten, #27). **P0-#4 (OpenAI-API correctness) was ALREADY DONE** (audit #96;
verified + ticked, shard #110). Remaining daemon edges: node-side challenge sketch, warm re-join receipt,
standby `sidecar -seed`, network manifest resolution, relay auto-discovery (P0-#3).

**THE MAP (P1-#1) — DONE + DEPLOYED: https://shard.c0mpute.ai** (c0mpute #33 = `data-site/network.html`,
served via nginx on the kloot box `/var/www/shard.c0mpute.ai/`). A DoubleZero-style spinnable 3D dotted globe
(pure canvas, no libs) that's a NETWORK EXPLORER — click a node → role/layers/up-down/RTT/receipts panel; locks
onto the visitor's continent on load; c0mpute design system (real argent via the same Typekit kit). STILL A
SIMULATION → wiring to live orchestrator state is the follow-on (now unblocked by auto-form). Rebuild:
scratchpad/build_globe.py → cp to /var/www. Ops in memory [[shard-demo-deployment]].

**LEG 8 (P1-#2 "make it serve") — SERVER HALF SHIPPED (leyten's pick), NODE/ENGINE half remains.**
- c0mpute **#34**: the live server AUTO-FORMS rings from real announces (`attachSwarmLoop` gained `resolveModel`
  + a debounced form-from-free-candidates loop; new `lib/orchestrator/model-profiles.ts` = M25 profile). This was
  the headline gap — `formSwarm` was NEVER called on the running server (demo-only). Handle now captured (was
  discarded). RTT = labelled uniform placeholder (measured N×N round = refinement).
- c0mpute **#35**: `attachSwarmLoop.serveRequest(model,messages,params,{onToken,onDone,onError})` finds a ready
  swarm, emits `swarm:job {swarmId,jobId,messages,nonce,maxNew,reasoning,tools}` to the coordinator, relays
  `swarm:job_token` deltas + `swarm:job_complete` back (one complete event finishes the client stream AND
  settles). Orchestrator `tryDispatchSwarm` routes sharded-model requests here (ollama/image untouched). PROVEN
  no-GPU end-to-end: `scripts/leg8-serve-test.ts` 10/10 — auto-form→dispatch→stream→complete→settle credits both stages.

**⇒ THEN (second in line, after the 07-17 lever stack above) — finish Leg 8 = the NODE/ENGINE serving half:**
1. **DAEMON coordinator handler** (`c0mpute-worker/src/shard-worker.ts`): on `swarm:job` (only if `current`
   assignment `isHead` + matching swarmId) drive generation → emit `swarm:job_token {jobId,delta}` per commit →
   `swarm:job_complete {swarmId,jobId,nonce,tokensGenerated,response,receipts}`. Shim-fakeable (extend
   `scripts/shard-python-shim.py` + `shard-runner.ts` `runCoordinator`).
2. **`python -m shard.coordinate`** (shard): thin entrypoint over `phase0/m25_pipe.py coordinate_pipe` — job on
   stdin (messages+params+swarm_id/job_id/nonce), stdout contract `SHARD_JOB_TOKEN/DONE/FATAL`, threads the
   settlement nonce, sweeps receipts. (m25_gateway.py already does this over HTTP; make it socket-drivable.)
3. **Return tunnel** tail→coordinator on the head sidecar (real-topology, like the forward-leg) — for the real
   ring; the mock doesn't need it.
Then extend the mock harness (sim `serveRequest` + shim coordinator) to prove request→served→settled with real
daemons. THEN: wire the live map to real orchestrator state. Pay-model $ mapping = leyten's fork (stub).
_(Deferred by leyten: EAGLE hang P0-#5 = real vast spend, do AFTER the no-spend code blockers.)_

### ⇒ 2026-07-15 (LATER) — LEG 7 SELF-SERVE JOIN WORKS END-TO-END LOCALLY: one command, zero env vars
**The user's test is now `clone → one command`.** Shipped this block: c0mpute **#29 #30 #31**.
- **Mock orchestrator harness (c0mpute #29, `scripts/shard-daemon-sim.ts`):** the missing test tool — real
  `verifyBindingProof` + real `decideRole`/`shard.probe` + the real `attachSwarmLoop` control plane; only
  PLACEMENT stubbed (`SimSeam`), `swarm-loop` gained an injectable `seam`. Lets the full daemon lifecycle run
  with zero cloud (`--once` exits 0 on serving = CI-able).
- **Self-provision + forward-leg addressing (c0mpute #30):** `shard-setup.ts` = enroll step 0 installs the
  engine checkout + a pinned python venv + the sidecar (sha256-pinned release download, go-build fallback) +
  probe slice, ZERO env vars, idempotent. Forward-leg: `swarm:assign` now carries each peer's dialable sidecar
  multiaddrs (`NodeCapabilities.addrs`, captured from the sidecar ADDR lines at enroll); a non-tail stage dials
  its successor (`-forward`) + pins inbound to its predecessor PeerId (`-allow`). Sidecar is now a daemon-scoped
  fixture (standby→ring-legs→restored, generation-guarded). **2-node libp2p ring PROVEN** — probe crossed both
  sidecars over a DIRECT hole-punched conn (`SHIM_FORWARD_ROUNDTRIP` + `SHIM_INBOUND`, both stages READY).
  **This lifted the tail-only restriction — multi-stage rings form.**
- **One-command test (c0mpute #31): `npm run try-shard`** boots the sim + a daemon and streams enroll →
  announce → assign → pull → READY → serving. GPU box = real self-provisioned stage; GPU-less = shim. Doc:
  `c0mpute-worker/SHARD_QUICKSTART.md`. Sidecar release CI: shard **#106** (`sidecar-release.yml`, operator-
  triggered; publish v0.1.0 via workflow_dispatch, then bump the sha pin in `shard-setup.ts`).
- **Live bug found + fixed:** the sidecar's framed tunnel uses an **8-byte** BE length prefix
  (`shard/transport.py` `send_msg`), not 4 — a 4-byte guess is read as a garbage length and dies at the 60s
  frame deadline. (Harness shim only; the real engine already speaks 8-byte.)
- **Decentralization stance (leyten asked):** PoC launches **M1 verifiable-centralized** (one orchestrator,
  but every decision is a replayable artifact — admission=`shard.probe`, placement=`shard.plan`, settlement=
  signed receipts; data plane already P2P). Decentralized orchestration = post-launch staged migration
  (`c0mpute/PLACEMENT_AS_PROTOCOL.md` M0→M4, "swarm forms with orchestrator DEAD" = the milestone). The trap to
  avoid is UNVERIFIABLE centralization, not centralization. Gateway/demand layer stays centralized+replaceable.

**⇒ NEXT — finish Leg 7 (ranked; forward-leg + self-provision DONE):**
1. **Peers-first verified fetch CLI** — promote `shard.fetch.fetch_block_range` + ChainProvider to a CLI the
   daemon's `pullRange` calls (torrent path, live-proven in-engine); today it's the HF mirror.
2. **Runtime artifact (§8-3)** — signed content-addressed sm120 bundle over block-exchange (kills the pip term).
3. **Node-side `swarm:challenge` sketch** (until then NEVER drive startSpotCheck on shard swarms — silence =
   spot_check_fail = reputation death spiral) + **RTT-mesh round** (probe --net-only vs assigned peers).
4. **Warm re-join ≤3min acceptance receipt** + the Ink/blessed-contrib **map UI** (P1-#1, the launch face).
_(Parallel: EAGLE offline fix P0-#5. Perf PRs #100/#101 + tree-graph = P2.)_

### ⇒ 2026-07-15 — LEG 7 STARTED: shard.stage CLI + P0-#2 landmines DEAD + the daemon skeleton MERGED
**Shipped (all merged same-session):** shard **#103** (yesterday's stranded local commits — `--external-tail`
+ LAUNCH.md were never pushed into #102; caught + landed), shard **#104**, c0mpute **#27 #28**.
- **`python -m shard.stage` (shard #104)** — the operator SSH launch string promoted into the CLI the daemon
  execs (NODE_DAEMON §4/§8-2): assignment as flags, engine env derived in-process, machine-readable stdout
  contract **SHARD_STAGE_OK / SHARD_STAGE_READY (emitted by serve()) / SHARD_STAGE_FATAL + nonzero exit**,
  `--check` preflight. Secrets env-only. **P0-#2's two named landmines are DEAD** — `/root/.hf_token` → env →
  `~/.hf_token` → hf login chain; `node_kv`'s flat `import transport` → `shard.transport` fallback (no
  hand-set PYTHONPATH); receipt-key default `~`-relative. Pinned by a clean-env no-PYTHONPATH subprocess gate
  (tests/test_shard_stage.py); suite 554 passed.
- **AUTO-UPDATE REMOVED from `@c0mpute/worker` (c0mpute #27, leyten call 2026-07-15):** the worker never
  self-updates — §7 fork #1 RESOLVED as "no auto-update"; upgrades = explicit npm install.
- **THE DAEMON SKELETON (c0mpute #28): `--mode shard` is REAL** — `shard-worker.ts` (enroll: sidecar-proved
  node-bind + probe-measured cap → announce → standby; on `swarm:assign`: pull range → sidecar → supervised
  `shard.stage` → `swarm:ready`; self-heal restart budget; teardown recycles the socket so `onNodeGone` frees
  the lease) + `shard-runner.ts` (subprocess seams). 3-file recon (worker plumbing + the REAL server protocol)
  + adversarial review pre-ship (6 MAJORs found+fixed — the worst: node-bind's `model` field must be a profile
  DICT or the role verdict silently never lands; non-tail assignments are a guaranteed boot-crash loop until
  peer addressing exists → refused loudly).
- **Protocol truth (recon, vs the spec's guesses):** events in code TODAY = `node:announce` / `swarm:assign` /
  `swarm:ready {swarmId}` / `swarm:challenge(_result)` / `swarm:job_complete`; **no heartbeat, no swarmToken,
  no peer multiaddrs in assign** (sidecar out-of-band), announce-TTL + RTT-mesh collection + auto-form trigger
  = DOC-ONLY (formSwarm is never called by the running server; demo-driven only).

**⇒ NEXT — finish Leg 7 (DONE this session: forward-leg addressing, self-provision, one-command test,
verified peers-first fetch. Remaining ranked gaps, TODO'd in code):**
1. **Weight SEEDING at standby** — the daemon holds verified ranges but doesn't `sidecar -seed` them, so
   peers-first fetch (wired, shard #108 + c0mpute #32) finds 0 providers and always falls to the mirror.
   Wire standby seeding → the torrent path goes live (bootstrap addrs already flow in the assign payload).
2. **Network signed-manifest resolution** — today the daemon LOCALLY publishes a manifest (integrity only,
   no publisher trust). Resolve the network's signed manifest + pinned publisher pubkey from `manifestRef`
   (the catalog seam) so `--pubkey` pins a real publisher.
3. **Runtime artifact (§8-3)** — signed content-addressed sm120 bundle over block-exchange (kills the pip term).
4. **Node-side `swarm:challenge` sketch** (until then: do NOT drive startSpotCheck on shard swarms — silence
   = spot_check_fail = reputation death spiral) + **RTT-mesh round** (probe --net-only vs assigned peers).
5. **Warm re-join ≤3 min acceptance receipt** + **the live map UI (P1-#1, the launch face — leyten's fork:
   framing/aesthetics worth his input before building).**
_(Parallel track unchanged: the EAGLE offline fix (P0-#5). Perf PRs #100/#101 + tree-graph = P2.)_

### ⇒ 2026-07-15 (LATEST) — VERIFIED PEERS-FIRST FETCH wired end to end
Shipped shard **#108** (`python -m shard.fetch` CLI — signed manifest + [lo:hi] → verified selective pull,
LocalDir>libp2p>mirror chain, SHARD_FETCH_DONE/FATAL contract; 4 tests) + c0mpute **#32** (daemon pullRange
calls it; manifest built from HF metadata at enroll — LFS oids ARE sha256, no weight DL; bootstrap = ringmate
sidecar addrs from the assign payload; raw-snapshot fallback). Closes the daemon↔"every byte CID-verified"
gap. Verified on the one-command demo. NEXT-1 (seeding) makes the peer sourcing actually fire.

### ⇒ 2026-07-14/15 — RESIDENTIAL RING SERVED + LAUNCH LIST + PACKAGE DECISION → tomorrow = START LEG 7
**Tonight's outcomes:**
- **Residential home GPU SERVED M2.5 (leg 6 residential half, BANKED).** leyten's Ghent 4090 (consumer, WSL2,
  double-NAT, mid-game) joined a 5×EU-5090 vast ring as the tail via a relay hole-punch, torrented its 14 GB
  tail layers from a peer seeder, served coherent output. ~2 tok/s g=1 (no drafter). Built
  `--external-tail MADDR:LO:HI` + `--swarm-token` in `m25_scatter_pipe.py` (skip-SSH external tail wired via
  its relay circuit maddr; on branch `feat/sidecar-nat-flags`). Memory: [[residential-reachability-proven-via-relay]].
- **THE LAUNCH LIST → `docs/LAUNCH.md`** — ONE severity-ordered "what's left before launch" (6 P0 + 4 P1).
  Stop tracking scattered NEXT lists. Rule: **all P0+P1 checked → launch.** Launch = "no wizard" (a stranger
  runs one command, joins on its own) + a live map.
- **PACKAGE DECISION (leyten, tonight):** the node daemon (Leg 7 / P0-#1) = **`--mode shard` inside the
  EXISTING `@c0mpute/worker`**, NOT a new package. The worker (~1850 LOC TS, at `workspace/c0mpute/c0mpute-worker`)
  already has a mode system (`max`/`image`) + real plumbing — KEEP it: the CLI shell, the socket.io
  control-plane + auth + infinite-reconnect (`worker.ts`), `setup.ts` (394 LOC), benchmark→register,
  `update.ts` auto-update, `config.ts`. WRITE FRESH: `shard-worker.ts` (enroll → layer-range role → torrent
  weights → drive `python -m shard.stage` + sidecar subprocess → serve → self-heal) + the Ink/blessed-contrib
  terminal map UI (all greenfield) + a shard-flavored probe (VRAM/uplink/RTT). The ollama/image workload code
  is NOT reusable (wrong shape) — you're adding a mode, not reusing that.
- **P0 BUGS found live (now in LAUNCH.md):** EAGLE silently hangs the coordinator on the residential-tail ring
  (P0-#5; fix OFFLINE — reproduce on a controlled 2-box datacenter ring, it does NOT repro on the CPU fake-ring
  where EAGLE works); portability landmines — `m25_pull_range.py` hardcodes `/root/.hf_token`, `node_kv`'s flat
  `import transport` needs `PYTHONPATH=phase0:shard` off the flat vast layout (P0-#2).

**⇒ TOMORROW — THE ONE NEXT ACTION: start the PoC finish = BUILD LEG 7 (the self-serve shard-mode daemon).**
1. Pull up `NODE_DAEMON.md` (c0mpute branch `docs/node-daemon-spec`, unmerged) and reconcile it against the
   real worker code above.
2. Scaffold `--mode shard` in `@c0mpute/worker` per the keep/write-fresh split.
3. Kill the P0-#2 portability bugs in shard (the `/root/.hf_token` hardcode + the `import transport` PYTHONPATH)
   — they block any stranger's box.
_(Parallel track: the EAGLE offline fix. Perf-lever PRs #100/#101 + tree-graph #4 = P2, post-launch.)_

### ⇒ PoC DEFINITION-OF-DONE (agreed with leyten 2026-07-11) — the legs, status + owner
The betanet PoC = a permissionless network where any GPU joins, gets measured, gets a role, serves
real inference verifiably, and gets paid. Leg status:
1. **Interactive speed** — ✅ BANKED (20-30 tok/s solo receipts, graph-aux).
2. **Permissionless join + measured admission** — ✅ BANKED (probe live-proven, role-at-node-bind).
3. **Torrent weight propagation** — ✅ BANKED (ringmate pulls, DHT, same-peer resume).
4. **Verifiable serving** — ✅ BANKED (per-stage receipts everywhere, batched attested, fail-closed).
5. **Batched viability** — RE-SCOPED by leyten 2026-07-11 to **20-30 tok/s PER STREAM at B≥2**;
   LARGELY MET 2026-07-11 (receipt perstream-delockstep-20260711: DE-LOCKSTEP #84 live, B=4
   per-stream medians reasoning 26.9 / tools 26.8 / code 23.9 / qa 19.4 ✓, prose/sum/mix 13.6-17.2
   g-bound; agg B-curve → 23.0/30.9/54.3/111.5). **2026-07-12 (receipt perstream-trees-ab-20260712,
   3 reps + K-reference, all receipts valid):** the two follow-up levers MERGED and MEASURED —
   * **Per-stream trees (#86, M25_TREE + rows)**: the g lever WORKS (+15-70% committed/round,
     reasoning 2.31→3.92) but **HIT THE KILL CRITERION** live: −29..−63%/stream vs K-tuned chains
     on every arm. Decomposition (M25_STAGE_TIMING): tree frames run EAGER stage-side — 154ms
     summed stage compute vs 45ms graph-replayed chains (3.4×), rounds 328 vs 169ms. **What binds
     = missing CUDA graphs on the tree kernel**, not g/wire. Trees stay env-armed only; the unlock
     is designed in `.claude/plans/tree-graph-capture.md` (padded-N capture; mind the dummy-KV
     trash-slot hazard documented there). NOTE #86 changed rows g to COMMITTED/round (uniform
     chain+tree) — pre-#86 receipts quote accept-only g, ~1 lower on divergence-heavy content.
   * **Content routing + per-(class,B) K (#87 + #88)**: THE surprise win — **K=6 chains lift every
     g-bound B4 arm 20-100%**: reasoning 14→**29.0**, mix 14.2→**17.9 (≥16 bar MET)**, prose →18.5,
     summarize →**18.2 (07-11 fp8-collapse did NOT reproduce — content-shift noise, closed)**,
     qa →21.1, code →19.8; tools keeps K=8 (g~5, 29.7). K=8 still wins at B≤2 (22.5/21.7/stream).
     Gateway now routes K by (content class, batch width); M25_DELOCKSTEP default ON.
   * **FULL-bar scorecard at B=4 (K-tuned chains)**: reasoning 29.0 ✓ tools 29.7 ✓ qa 21.1 ✓
     mix 17.9 ✓(≥16) code 19.8 ≈ summarize 18.2, prose 18.5 (target 20 — 1.5 short; remaining
     levers: tree-frame graphs + K=5 novel tier). B=2 tier: 21.7/stream ✓.
6. **Any-device proof** — ✅ THE FAT-CARD HALF BANKED 2026-07-12 (receipt
   hetero-fatcard-join-20260712): a 96 GB RTX PRO 6000 joined THROUGH THE NETWORK (announce →
   probe → role → shard.plan w/ per-node measured caps, #90 + c0mpute #20 — ring_up never ran),
   anchored a 4-hop ring (31 L + coordinator) with three 5090s, and **the FULL per-stream bar is
   MET on every content class incl. prose** (K-tuned medians of 3, B=4/stream: reasoning 37.2 /
   tools 34.4 / code 29.3 / summarize 26.2 / qa 24.9 / prose 23.4 / mix 23.8; solo 30.1; agg
   B-curve 30/50/96/136; 72/72 receipts valid). The prose 18.5-vs-20 gap CLOSED by admission, not
   an engine lever — fat cards remove hops from everyone's ring. Second Pro 6000 measured graph
   replay corrupt (cosine 0.0) → relegated off-ring by its own verdict (role verifier, binding
   fast_kernel) — the silent-corruption class caught at admission. Spec: 96 GB tier = MEASURED
   (2329.5 MB/L, 35-layer cap, transient 72 MB).
   **MlxRuntime (the non-NVIDIA half): MAC GATE GREEN 2026-07-12 (receipt
   mlx-mac-gate-20260712).** Built + merged #91 (3-skeptic review fixed 3 loader MAJORs), then
   leyten provided a Scaleway key and a rented M2 Pro (16 GB, €0.21/h) ran **11/11
   real-silicon checks** against the real mlx-community/MiniMax-M2.5-4bit slice [29:32):
   range load 1.9 GiB/layer; forward byte-deterministic in both wire representations; **KV
   rollback byte-equal on Metal** (gap refused); aux contract; corrupt-index refusal fires on
   the real index; real tail logits (200k vocab) + head embed; **wrapper byte-identical to a
   hand-driven mlx-lm layer loop**; first measured Apple row **0.94 ms/layer decode (M2 Pro,
   B=1, mlx 0.32.0)**. pytest green on BOTH platforms (Mac 24/2skip incl. the real-mlx smoke).
   All 13 MAC-VALIDATE assumptions held as written. Ops: Scaleway creds
   ~/.config/scw/shard_creds.env (0600, NEVER tracked); M4 quotas=0 until leyten's card
   verifies (M4-XL 64 GB = the fat demo-stage box; M4-SP 16 GB high-stock); the Mac stays warm
   through its 24 h lease. **REMAINING for leg 6: fake-ring stage swap → ONE live mixed ring
   (Mac + 5090s + fat card, all network-joined) = the flagship demo receipt.** Full-model
   reference greedy-agreement + the wired-limit table need a ≥96 GB Mac (post-quota).
   **Placement-as-protocol DESIGNED (c0mpute PR #21, merged):** 3-angle panel + adversarial
   verify (8 MAJORs caught: vapor slashing, nonexistent planInputsHash cited as shipped, wrong
   determinism CI, forgeable gap-evidence). c0mpute/PLACEMENT_AS_PROTOCOL.md: records-as-hints +
   re-measure-at-formation + member verify-and-sign + demand-side anchor; M0→M4 migration (M1
   verifiable-centralized shippable now); §10 = leyten forks (global-truth vs demand-artifact,
   purchasable placement, staked sets, emissions wash-trading gate).
7. **The join gateway (node daemon)** — ❌ SPEC'D 2026-07-12 (c0mpute NODE_DAEMON.md, #22; the
   leg the DoD missed — leyten: "what is the gateway between the network and a user machine?").
   Decision: ships INSIDE `@c0mpute/worker` (one product, one install; shard mode = a long-lived
   daemon: enroll → standby → serve). The 45-min virgin-box join decomposes into bytes-moving,
   not deciding (placement = seconds, measured) — so the fixes are packaging: the RUNTIME as a
   signed content-addressed artifact over the weights block-exchange (kills the pip term, pins
   the numerics env receipts quote), ranges retained on disk (re-join ≤3 min), probe overlap.
   One new engine seam: `python -m shard.stage` (promote scatter_pipe's launch_stage). Acceptance:
   a stranger's machine, ONE command, no operator ssh — enrolled, measured, placed by the loop,
   serving with valid receipts, warm re-join. leyten forks flagged in the spec §7 (auto-update
   key/staging, seeding-default consent, Windows tier).
8. **(gated) Market leg** — c0mpute #16 open since 07-08; everything it needs exists (node_role
   verdicts, pay-by-layers, settle seam) but it is a PRODUCT-DIRECTION call: **leyten's fork, not
   engine work**. Paper publish also on leyten (refresh the headline with the new B-curve +
   per-stream scorecard first).
Sequence (updated 2026-07-12 evening): **Mac session next** (Scaleway M4-XL or leyten's Mac —
MlxRuntime Mac gate → live Mac stage → the mixed-spectrum demo receipt) → **leg-7 node daemon**
(NODE_DAEMON.md build list: worker shard mode + `shard.stage` entrypoint + the sm120 runtime
artifact — the "one command, no operator ssh" acceptance) → tree-frame graphs (NOTE: the prose
bar is now MET on the hetero ring class; the tree plan stays parked as a 5090-ring-class lever,
`.claude/plans/tree-graph-capture.md`) → probe fidelity leftovers → placement-as-protocol M1
(verifiable-centralized — shippable, designed) → market iff #16.

### ⇒ 2026-07-12 (latest) — THE MOAT, END-TO-END: the network partitioned 9 strangers into 2 optimal rings + served both in parallel (receipt fleet-multiswarm-20260712)
leyten's ask: rent a heterogeneous EU fleet, let the network build MULTIPLE optimal swarms, sweep
them. Delivered. 9-node pool (3× RTX PRO 6000 WS, 3× 5090, 1× H100 NVL, 1× 4090 — EU, RTT
8.5-76ms), each self-measured via the probe; the loop's own `formSwarm` (greedy fewest-hops-first,
one-slot leases) carved **TWO complete rings** and served MiniMax-M2.5 on **both in parallel**,
**144/144 receipts valid**. Merged c0mpute **#26** (fleet driver), shard receipt banked. Fleet
torn down (instances-v1==0), ~$18 of the ~$86 balance.
- **SWARM 1 (2-hop fat, 2× Pro 6000: CZ[0:29]→LT[29:62])** — the FASTEST scattered ring measured
  to date: reasoning **51.1**/stream, tools 45.9, summarize 40.7, code 40.3, qa 35.4, mix 30.1,
  **prose 29.5**; solo 42.1; mix-B8 agg 160.6. Every class crushes the 20 bar.
- **SWARM 2 (3-hop mixed: DK 5090 coord[0:9] → NO Pro EAGER[9:42] → BG H100 tail[42:62])** —
  reasoning 34.7, mix 25.2, prose 22.6, qa 24.8; solo 37.9. Still fully servable — and its 33L
  middle was a **graph-CORRUPT Pro 6000** (cos 0.0) EAGER-ADMITTED by the refined logic (relegate
  a graph-fail card only if its eager layer_ms ALSO drags; a 96GB card at 0.29ms eager beats a
  graph 5090) and its tail the **H100 on marlin** — a card an allowlist discards + a card on its
  non-native path, both serving. Deploy launches such stages graph-OFF (scatter_pipe `:eager` order tag).
- **The hop-count gradient is MONOTONIC**: prose 2-hop 29.5 > 4-hop hetero 23.4 > 6×5090 18.5.
  Fat cards SUBTRACT HOPS — the admission thesis's sharpest proof. Pool ceiling correctly found:
  3 leftover consumer cards (28L) can't hold a 3rd 62L copy → residual (routed, not wasted).
- **MEASURED spec rows (were extrapolated): H100 NVL** marlin 4057 MB/L, transient +4824,
  layer_ms 0.184 graph, 20L/93GB; **4090** marlin 0.263 graph. HETERO_DEVICES updated.
- Ops: connect-during-tunnel-settle wedged sw2's first coordinator job (banked 3-5min hazard) →
  clean pid-kill + coordinator restart against the settled ring fixed it (no re-warm needed); 6/12
  boxes were ssh-key duds, `vastai attach ssh` rescued 4; run-once rent script still spawned 1
  stray (killed). Driver bugs fixed at source: keyless-node RTT-matrix realignment; eager-admit.

### ⇒ 2026-07-12 (later) — LEG 6 SESSION: fat-card hetero join BANKED + MlxRuntime merged (Mac-gated) + placement-as-protocol designed
Merged: shard **#90 #91** + c0mpute **#20 #21**. Ring up+down same session (~$10 of the ~$96
balance; instances-v1==0 verified). Full story in leg 6 above; the operational trail:
- **The join flow that worked** (scratchpad/runbook_hetero_20260712.md): rent_hetero → hetero_boot
  (probe --measure per box) → hetero_join (node keys + serial net probes + announce set) →
  scripts/hetero-join-live.ts (REAL SwarmManager: announce→admit→relegate→plan) → hetero_deploy
  (pulls per the plan) → scatter_pipe warm-only → 6-pass sweep (3 reps × K∈{8,6}) → hetero_bank.
- **Loop fixes that WERE the job:** plan_ring per-node measured caps (layer_vram_mb/cap_layers/
  total_vram_mb density/load_peak_extra_mb/layer_ms — select_ring already took dicts, #90);
  c0mpute NodeCapabilities + formSwarm thread the measured vector (#20); the loop e2e demo had
  been INFEASIBLE on master since the 07-09 profile revision — fixed + now demos the hetero join.
- **OPS (new dings, same lessons):** `pkill -f` self-match struck AGAIN (use safe_kill/pid-kill,
  period); re-running a create-side rent script rents DUPLICATES (5 extras, caught+destroyed in
  ~2 min — rent scripts must be run-once); `nohup CMD &` over vast ssh HANGS the client even with
  full redirects — `setsid ... </dev/null &` + timeout-tolerant wrapper (the remote usually
  succeeded); vast `-p 29600:29600` maps to a RANDOM host port — dial-back must advertise the
  mapped port; scatter_pipe assumes /tmp/sidecar already pushed (fresh bootstrap paths must scp it);
  2/8 boxes were ssh-key-propagation duds (destroy ~15 min, race-the-replacement).

### ⇒ 2026-07-10 (later) — THE DRAFTING TAX IS DEAD (PR #74): batched drafter forward + batched-stage CUDA graphs, live-proven
The two ranked levers landed, adversarially reviewed pre-ring, and re-swept apples-to-apples
(receipt **batched-levers-sweep-20260710**). Merged **#74 #75 #76**. Ring up+down same session
(6×5090 EU, one head replaced mid-session); leyten topped the vast balance +$100.

**1. BATCHED DRAFTER FORWARD (the big lever, coordinator-side).** `eagle_draft.draft_batch` runs the
B EAGLE fork chains as ONE [B,...] forward per chain step (linears/norms/argmax batch rowwise;
attention stays per-fork over its own ragged context; 2 host syncs × K × B collapse to one
`.tolist()`); `fetch_b` = the batched HybridDrafter fetch (n-gram per stream, misses in one chain),
driven from `coordinate_pipe_batch`'s fill loop. **Byte-identical to serial per row** — CPU gate
(research/m25_draft_batch_test.py), e2e fake-ring gate (research/m25_batch_eagle_test.py), and
LIVE-validated: at the old numerics env the re-sweep reproduced the prior receipt's content-g
token-exact (summarize 5.80==5.80, tools 3.73==3.73...). Solo path untouched.

**2. BATCHED-STAGE CUDA GRAPHS.** `BatchGraphRunner` (m25_stage.py) captures `run_block_decode_b` at
the job-fixed [B,K+1] shape, one graph per context bucket — solo's _GraphState design batched (static
cp/RoPE/per-stream-mask buffers refreshed per replay; aux copy captured in-graph, [B,s,H] statics).
Routed via `_block_b`; `M25_BATCH_GRAPH=0` = per-lever hatch (in scatter_pipe ENG_ENV). The
adversarial review (3 skeptics; no wrong-output path found) flipped two MAJORs pre-ring: (a) batched
jobs silently inherited the last SOLO job's runtime graph arm → `reset_batch` now applies
`_reset_flags`, tail acks the APPLIED route+counters when stamped (M25_GRAPH_JOB), coordinator raises
on refusal + surfaces `graph_arm` per job; (b) capture-at-zero-headroom could pin a pool that OOMs
the next eager prefill (dead stage) → free-VRAM pre-check, short → LOUD permanent-eager. Live: the
guard refused B=8 capture on the kv-8192 brim tail exactly as designed; at kv 4096 all captures clean.

**3. THE RE-SWEEP (3 passes, same 12 arms/prompts/K/max_new, all receipts valid).** B-curve
2.96/3.65/7.70/9.72 → **7.50/5.67/7.81/11.90** at the old env (bf16 wire) → **18.11/9.64/14.54/15.48**
on the fp8-wire build. B=1 rounds 1158→458ms at equal env (the tax gone); v1 off→on isolates graphs
(B4 1398→787ms rounds). **THE BAR (mix-B4 ≥25 / B8 ≥50) NOT met — the round is now TRANSPORT-bound:**
EAGLE aux (3 × [B,K+1,H]) ∝ B dominates; at bf16 wire B4 rounds stay ~1600ms (payload), fp8 halves it.
**fp8 wire = 2× lever AND a numerics env** — it shifts greedy content: g_mix 3.55 bf16 vs 2.48 fp8
(quote g per wire mode). qa-B4 v2 arm = WAN-stall outlier (footnoted, receipts valid).

**4. Spec:** ADMISSION_SPEC g_batched operative stays **2.5** but re-derived (= the direct fp8-wire
measurement, no drafter-tax discount). Ops lessons: HF xet pulls stall (kill+resume kick-cycle);
`pkill -f` self-match struck AGAIN via a heredoc body carrying the pattern (split kill/launch into
separate ssh calls); a vast host with wedged nvidia-uvm survives reboot+stop/start → destroy+replace.

**NEXT SESSION — MlxRuntime (leg 6), leyten's call 2026-07-12:**
1. **MlxRuntime** — the any-device proof, THE active build (see leg 6 above for the build shape:
   ModelRuntime seam → M2.5 layer-range on Apple silicon, MoE dequant = the risk item → same
   attest/receipt contract → offline parity gates vs torch → ONE mixed ring w/ a Mac stage for the
   demo receipt). CPU/local-first; no ring spend until the demo. Needs a target Mac to serve from —
   confirm which machine with leyten at session start if none is already reachable.
2. PARKED-DESIGNED: **tree-frame CUDA graphs** (the prose-bar lever; tree stage compute 154→~50-70ms
   candidate → the g lift nets +20-40%/stream instead of −13%). Design + the dummy-KV trash-slot
   hazard: `.claude/plans/tree-graph-capture.md`. Offline gates + adversarial review before any
   ring, same kill criterion. Pick up after leg 6 (or interleave if MLX blocks on hardware).
3. Then: prose's last 1.5 tok/s (tree graphs + K=5 novel tier, both live-measurable), probe
   fidelity leftovers, market iff c0mpute #16 (leyten's fork). Paper headline refresh (B-curve +
   per-stream scorecard) awaits leyten's publish call.

**RING: none live** (instances-v1==0 verified 2026-07-12 EVENING post-hetero-teardown; balance
~$86 — the leg-6 session used ~$10). c0mpute WIP untouched (worktree used for the loop changes;
NETWORK_ARCHITECTURE.md gained §11 pointing at PLACEMENT_AS_PROTOCOL.md).

### ⇒ 2026-07-10 — FULL DRAFTER IN THE BATCHED PATH (PR #72) + batching = the STANDARD path + the 12-arm use-case sweep
leyten's call executed end-to-end: wire the full drafting stack into the batcher, make batching the
engine standard, sweep B × real AI use cases. Merged **#71 #72**. Ring up+down same session
(6×5090, measured-footprint 12L plan, ~$4.4); **balance ~$16.9**.

**1. EAGLE-IN-BATCH BUILT (PR #72, adversarially reviewed pre-ring).** `coordinate_pipe_batch` now runs
the solo stack PER STREAM: hybrid n-gram→EAGLE via `EagleDrafter.fork()` (shared read-only head, own
context), per-stream aux `[B,s,H]` w/ per-STREAM fp8 scales, solo's depth rule + bonus-commit,
per-stream g/streaming/tools/reasoning/max_new, receipt sweep + verify (fail-closed). Stages attest
batched rounds (they did NOT before — and `reset_batch` now makes a FRESH nonce'd signer; the review
caught a stale SOLO signer bleeding valid-looking receipts into batched jobs). Also from the review:
batched coordinator no longer inherits solo's 20s rx deadline on a reused socket; EAGLE-with-no-aux
fails LOUD (was: silent worse-than-ngram paid measurement); B > ring M25_BATCH nacks cleanly (was:
stage-process death). Offline batched==solo byte-identity gate REVIVED (stubs dead since da9f11d) and
green. Gateway: batching is the STANDARD concurrency path — dispatcher collects a burst
(M25_GW_WINDOW_MS, cap M25_GW_BATCH ≤ ring M25_BATCH) into ONE ring job; lone request = unchanged solo
path; dead client never aborts batch-mates. `--serve` hands the ring's batch width to the gateway.

**2. THE SWEEP (12 arms live, receipt batched-sweep-eagle-20260710): drafting quality TRANSFERS;
throughput is now DRAFTER-bound.** Content-mix **g = 3.6 at B=4** (solo band!), per use case:
summarize/verbatim **5.8** > tools-JSON 3.7 ≈ reasoning 3.7 > code 2.9 > qa 2.4 > prose 2.2. The
equal-transport A/B: n-gram g=0.22 @220ms rounds vs hybrid g=3.62 @1.7s rounds — a **16× drafting
lift** but only 1.6× aggregate (4.62→7.41), because the coordinator drafts B EAGLE chains SERIALLY
(~0.25s/stream/round) and EAGLE pins depth to 1. B-curve: 2.96/3.65/7.70/**9.72** (B=1/2/4/8; 3.3× at
B=8). Best arm: summarize-B4 **16.07 agg**. 72/72 receipt sigs valid across all batched jobs; coherent
output everywhere. **The 20-agg bar on this build is DRAFTING-bound, not WAN-bound.**

**3. Spec revised from measurement:** `g_batched` 1.5 → **2.5 operative** (measured content 3.6
discounted for the drafter tax; ADMISSION_SPEC.md updated with the full content band + the engine-not-
policy framing).

**NEXT SESSION (the quantified throughput levers, in order):**
1. **Batch the drafter forward**: run the B EAGLE chains as ONE [B,...] forward (same weights,
   per-stream KV rows — embarrassingly batchable). Kills the ~0.25s×B serial term; projected mixed
   B=8 agg ~3× today's. The single biggest lever, coordinator-side only.
2. **CUDA-graph the drafter chain** (8 launch-bound micro-steps/stream today) and/or
   **graph-capture `run_block_decode_b`** (batched stages run eager; ring+eager floor measured 220ms).
3. **Depth>1 with stale-context EAGLE** A/B (pipelining vs draft-quality tradeoff).
4. Then the deferred: MlxRuntime → probe fidelity leftovers → market iff c0mpute #16.

**RING: none live** (verified 0). c0mpute WIP untouched.

### ⇒ 2026-07-09 (later) — ADMISSION MECHANISM BUILT+LIVE-PROVEN, on-ring TORRENT LOOP CLOSED, numbers v0→MEASURED
All three live milestones landed in one session, and the measurements REVISED the spec (the LIVING-doc
loop working as designed). Merged: shard **#65 #66 #67 #68 #69** + c0mpute **#19**. Ring torn down
(instances-v1==0), **balance ~$21.3** (session ~$10.6).

**1. CAPABILITY PROBE — built, adversarially reviewed, LIVE-VALIDATED (`shard/probe.py`, #65-67).**
The full ADMISSION_SPEC mechanism: pure role function (`python3 -m shard.probe`, stdio like shard.plan;
binding[] names the denying gate = the revise signal), `--measure` (one REAL layer: footprint,
arch-dependent load transient, graph-replayed layer_ms, binary fast-kernel = native kernel + CUDA-graph
capture+replay), `--serve/--net-only` (receiver-timed uplink, nonce dial-back w/ NAT advertise port,
connect-time RTT in topology units). The 22-test suite pins the spec table AT the receipts' numbers.
The ADVERSARIAL REVIEW flipped a marquee verdict pre-ship (cap 13 admitted a 32GB card at 30ms scatter
that the 13-15 receipt denies) — model of the discipline. **LIVE: a rented 4090 measured
{footprint 4057, marlin repack transient 4824(!), graph BIT-exact, uplink 210 receiver-timed vs 1316
listed, real pool-RTT 72ms vs 'NL' geo} → role VERIFIER (hops_vs_rtt binds) — correct physics.**
c0mpute #19: node-bind accepts a measured cap, server drives shard.probe (role NEVER self-reported),
verdict+cap persisted in `node_role` (placement/pricing/telemetry). Trust framing honest:
trust-then-punish v0; probe peers must be control-plane-ASSIGNED; signed transcripts = later hardening.

**2. NUMBERS v0 → MEASURED (docs/ADMISSION_SPEC.md revised + `M25_PROFILE` corrected, receipts in
docs/receipts/):** the real cutlass FULL-layer footprint is **2330 MB** (old 1700 was experts-only,
~35% light) — the "swizzle peak" MECHANISM story was wrong (transient is arch-dependent: cutlass 72MB,
marlin 4.8GB repack); the 15-layer OOM was footprint arithmetic. Direct warm reads: 13-layer middle =
**31.5/32.6 GiB (brim)**; 12-layer tail = 30.4. **12 = the 32GB plan target — 12-vs-13 RESOLVED.**
A 13-layer TAIL OOM'd loading lm_head (1.15GB) live → `plan.py` now models `tail_reserve_mb=1400`
(#69) + `layer_vram_mb=2330`, `cap_layers=12`. 24GB marlin holds **4 layers not 5** (N=16). Uplink
thresholds = receiver-timed single-stream (listings are ~6× optimistic). `g_batched` 4.0→**1.5**
(content-mix; see #3). Consequence to own: **5-box 32GB rings are brim-riding — 62 layers wants 6.**

**3. ON-RING TORRENT LOOP CLOSED (#68 + receipt onring-seed-shards-20260709).** The 5×5090 EU ring
launched `--seed-shards --receipts --batch 4` (every stage's sidecar seeds its verified range, DHT via
predecessors) and a JOINER (the 4090) pulled its full 28.6GB stage range **6/7 weight shards from
RINGMATES** (NO alone SERVED 3), every byte re-hashed vs the signed manifest, WHILE the ring served.
Attempt #1 exposed a real hole — one 60s WAN stall at 4.27/5GB discarded the partial → mirror — fixed
as **same-peer offset-resume + progress-gated retries (#68)**; rerun sourced peers-first. Known gap:
the head's initial DHT provide fires pre-mesh (direct-dial covers it; periodic re-provide = follow-up).

**4. BATCHED B=4 LIVE (receipt batched-b4-live-20260709):** coherence PASS, **DATA-ISOLATION PASS**
(stream outputs independent of batch-mates = batched stays verifiable), B=1/2/4 = 5.19/8.63/**11.95
agg** (clean 1.66×/2.30× WAN amortization) — BUT on n-gram-undraftable content (g≈1 floor). The 155-agg
receipt is the draftable-content (g≈4) regime. Batched viability is g-dependent + pool-relative; at
g=1 clearing 20 agg needs ~B=8 or a tight pool. Spec + SPEC_V0 revised accordingly.

**NEXT SESSION (in order — reprioritized 2026-07-10, leyten's call: batched becomes the engine
standard once the drafter is in):**
1. **EAGLE-in-batch, then batched-as-standard.** Wire the FULL drafting stack into
   `coordinate_pipe_batch`: per-stream aux buffers ([B,s,H] in the graph capture, mirroring the solo
   aux-in-graph design) + B drafter states at the coordinator (hybrid n-gram→EAGLE per stream; tree
   second). The batched path drafting n-gram-only is why B=4 measured 11.95 at the g≈1 floor while
   single-stream does 20-30 with EAGLE — same physics, missing drafter. THEN re-measure the batched
   bar honestly: REALISTIC prompt mix + B=4/8 arms (the counting prompt was n-gram-ADVERSARIAL, below
   any real floor — don't set g_batched from it; set it from the mix measurement). Direction once it
   holds: the batched CODE PATH becomes the default for every ring — B=1 reduces to solo (batchverify
   proven bit-exact at B=1) so one code path serves all roles and every benchmark runs B-parameterized
   on one warm ring. B stays a per-role SERVING-POLICY knob, not a constant: interactive anchors keep
   single-stream priority (per-stream latency at B>1 is strictly worse; batched fills idle capacity),
   and batched KV VRAM (B×MAXLEN per layer) keeps feeding plan/admission as modeled. The isolation
   gate (PASS, receipt) is what makes the unification safe: per-stream receipts/verification unchanged.
2. **MlxRuntime** — the any-device flagship (Mac/MLX; model file + 4-bit artifact + per-layer
   callability exist). CPU/local-first; the real "any device joins" proof.
3. **Probe fidelity follow-ups:** align `--measure`'s per-layer tensor set with what m25_stage actually
   keeps resident (probe loads the whole layer prefix — measured 2330 vs stage-implied ~2426 all-in at
   B=4, close but unaudited); periodic DHT re-provide in the sidecar; probe the >48GB tier when a box
   is cheap.
4. **Market migration stage 1** IF leyten greenlights c0mpute #16 — the role verdict in `node_role` is
   exactly the "asks in announce" input the cheapest-adequate formation needs.
5. Wire warm per-stage `per_stage_ms` feedback into re-selection (ring_up's honest gap; the probe's
   layer_ms is the admission-time half of it).

**RING: none live** (verified 0). c0mpute WIP (`onchain-staking.ts`) untouched.

### ⇒ 2026-07-09 — HETEROGENEOUS swarm PROVEN live + CAPABILITY-ADMISSION spec DERIVED (the strategic capstone)
Live-tested a mixed 5090+4090 ring end-to-end, then turned leyten's "capability function, not allowlist"
admission decision into a derived, adversarially-verified minimum-spec. This is the on-thesis output; the
ring re-measurement that preceded it was over-spend (see BALANCE).

**SHIPPED — shard PR #62 MERGED (`eagle/hetero-swarm`):**
- **Per-node VRAM footprint** in `select_ring` (`layer_vram_mb` accepts a dict): a marlin 4090
  (4.25 GB/layer, ~2.3× the 5090's 1.85) is sized to ~5 layers automatically, no OOM; scalar path
  byte-identical (goldens green). `ring_up` detects arch → footprint; head pinned to a 5090.
- **Marlin is CUDA-graph-safe** (proven live: 4090 stage 32.65 ms → **8.02 ms**, receipts valid) → graph-aux
  now default-ON for EVERY arch (`M25_EAGER_NONBLACKWELL=1` escape hatch). A non-Blackwell card is a
  FULL-speed ring stage, not an eager drag.
- **Live proof:** 6-box mixed ring (5×5090 cutlass + 1×4090 marlin) served coherently, all 6 per-stage
  receipts VALID across reasoning/verbatim/copy. Mixed-arch numeric compat confirmed live.
- **Docs:** stale "~10-12 single-stream" corrected to the graph-aux truth (see PROVEN); admission reframed
  as a capability function.

**DERIVED + VERIFIED — the capability-admission MINIMUM SPEC (`docs/ADMISSION_SPEC.md`, PR #63 OPEN for
leyten):** admission is a GPU-model-independent function, but NOT VRAM-only. `tok/s = g/T`,
`T = N·RTT + C`, `N_hops = ceil(62 / layers-the-weakest-node-holds)` — layers set hops, hops set speed.
**THREE co-binding constraints:** (1) VRAM→layers→hop-floor (gate on load PEAK not resident — 32 GB 5090
OOMs at 15 layers, cap 12; marlin's 4.25 GB/layer is fatal for 140 GB); (2) RTT (30→19 ms doubled tok/s
13-15→32; a 48 GB node at 80 ms tanks a ring); (3) uplink (decode trivial, but 16k prefill 50 MB/hop → 15
Mbps residential = 160 s TTFT). Compute is a BINARY graph-safe-fast-kernel check.
**HONEST ANCHOR VERDICT:** a 24 GB non-Blackwell card (4090/3090) can **NEVER** anchor fast single-stream
M2.5 (5 layers → 13 hops → needs g≥5.9, real g 3.3-4.5); a 32 GB 5090 is **marginal — clears 20 only on a
tight ≤24 ms REGIONAL ring** (the reason-math-32 receipt is tight-ring-only; a 5-stage ring needs 12.4
layers but 32 GB caps at 12 → a naive admitter OOMs); the comfortable fast-M2.5 anchor is a **48 GB
fast-kernel card (N≤4)**. M2.5's 140 GB makes interactive M2.5 Blackwell/pro-anchored — physics, not policy.
The long tail is **ROUTED not rejected**: batched-fill (24 GB marlin = ~36 tok/s agg at B=4), verify, seed,
and SMALLER MODELS (self-organizing multi-model = the real heterogeneity play). **The 20-bar is a ROLE TAG,
not a binary gate.** Min-spec table + the trustless-probe design in the doc.
**LIVING SPEC (leyten): the FRAMEWORK is settled, the NUMBERS are v0** — every threshold (uplink ≥200, the
12-layer cap, RTT ≤25, layer minimums, g) is a derived estimate to VALIDATE + REVISE by testing; when live
data diverges, CHANGE the number. The probe (#1 below) + batched test (#3) exist to correct it. Not a constant.

**NEXT SESSION — build the admission MECHANISM the spec calls for + continue the torrent path (in order):**
1. **The trustless capability PROBE** (the admission spec's core ask, CPU/one-box testable): a 1-block load
   probe measuring **peak-VRAM-at-load** (not free — else admit-then-OOM), `layer_ms`, the binary
   fast-kernel check, **uplink**, and **RTT-to-assigned-neighbors + NAT-dialability**; output a ROLE
   (interactive-anchor / batched-filler / verifier / seeder / reject) via the `ADMISSION_SPEC.md` function.
   shard owns probe+physics, c0mpute owns the role decision — wire it into c0mpute admission (before
   placement) and the market (price per role).
2. **On-ring `--seed-shards` LIVE** — the torrent half owed: push `/tmp/sidecar_new` (the DHT-capable binary)
   into the ring bootstrap so a joiner pulls its verified range from a RINGMATE, mirror only as fallback.
   (WAN peer-fetch already proven: 4090 seeded → box pulled 5 GB @95 MB/s.)
3. **Batched-aggregate LIVE confirmation** — the single measurement that validates the inclusion half of the
   admission spec (a 24 GB card → ring-worthy in batched). Only genuine live spend left worth doing.
4. **MlxRuntime** — the any-device flagship (Mac/MLX; model + 4-bit artifact + per-layer callability exist).
5. **Market migration stage 1** IF leyten greenlights c0mpute PR #16 (asks in announce + cheapest-adequate).

**BALANCE: ~$32** (this hetero session burned ~$90 re-measuring 13-15 tok/s — the physics already had the
answer; next session is SPEND-CONSCIOUS: CPU/one-box first, ring only for #2/#3 where live is the point).
**RING: none live** (instances-v1==0 verified). c0mpute WIP (`onchain-staking.ts`) untouched.

---

### ⇒ 2026-07-08 (later) — TORRENT-FIRST session: P2P propagation BUILT+WAN-proven, heterogeneous 4090 PROVEN, market design panelled
Acted on the three torrent tasks. All three moved; the loop still serves (regression pass ran live).

**1. HETEROGENEOUS DEVICES — non-Blackwell PROVEN, tier table shipped (shard PR #61).** A live-rented 4090
(sm_89, Ada) RUNS the M2.5 NVFP4 MoE via **marlin** (`M25_MOE_BACKEND`): cutlass REFUSES pre-sm_120
("kernel does not support current device"), emulation is sm_120-only Triton (illegal-memory on Ada), marlin
loads the SAME signed NVFP4 checkpoint via dequant-in-kernel at **4.08 GB/layer (~2.4× the cutlass 1.7 GB
footprint) and 0.35 ms/tok** decode MoE. So a 4090 is ring-worthy with FEWER layers — exactly what the
VRAM-sized planner already does. Shipped `M25_MOE_BACKEND=auto` (cutlass on sm_120+, marlin below — a
stage's arch is a node fact, byte-identical to old default on 5090s). `docs/HETERO_DEVICES.md` = the device
tier table vs the 20 tok/s bar (NVIDIA line + Apple Silicon MLX [a big Mac out-decodes a mid NVIDIA card;
M3 Ultra 512GB holds the WHOLE model = the natural full-replica auditor] + AMD [7900 XTX ring-worthy behind
a llama.cpp backend] + CPU [seeder / torch-free challenge judge]) + the any-device build list (ranked:
compute probe → per-node backend+layer_vram → MlxRuntime → per-format manifest v2 → format-matched
spot-check). `research/hetero_moe_probe.py` + `hetero_moe_xcheck.py` (deterministic cross-kernel cosine dump;
marlin dump banked, cutlass reference is a one-box follow-up). 3090 (Ampere) probe was pip-slow, non-blocking
— predicted marlin-same. leyten's "not just NVIDIA — any device (MacBook/Mac/AMD)" is answered: the ENGINE
seam (ModelRuntime + per-arch backend + per-format manifest + cosine spot-check) is heterogeneity-first; the
one genuinely new build is per-format artifacts + format-matched auditing.

**2. TORRENT WEIGHT-FETCH — BUILT, WAN-proven live, adversarially hardened (shard PR #61, the big one).**
The Go sidecar SOURCE was in the repo after all (`sidecar/main.go`, the M25_ENGINE "binary only" gotcha was
STALE). Extended it (`sidecar/blockx.go`): kad-DHT (`/shard` prefix, isolated from public IPFS), `-seed
manifest=modelDir` (PROVIDE every held shard CID + serve `/shard/blockx/1.0.0`, offset-resumable), `-fetch-cid`
(find-providers → block-exchange, direct-dials bootstrap peers when the DHT is dry). Python: `Libp2pProvider`
(real, was the stub) + `ChainProvider` (peers first, mirror/HF origin last, verified PER SOURCE so a hostile
seeder is dropped and the pull continues; trust unchanged — every byte re-hashed vs the signed manifest).
**PROVEN**: 6 local 2-peer tests (`tests/test_blockx.py`: peer fetch w/ empty mirror, A→B→C propagation with A
dead, hostile-seeder fallback, hostile+honest race → honest bytes win, dead-peer mirror fallback, fail-closed)
AND **over the real internet** — a rented FR 4090 seeded, this box pulled a 5 GB shard at **95 MB/s, sha-matched**
(+ config shard 0.2s). An **adversarial review** found the trust root INTACT (no path to VRAM poison — every
attack ends in the re-hash deleting the file + mirror recovery) but 4 real open-net holes, ALL FIXED +
regression-tested: uncapped control-frame `make()` → OOM (capped 1 MiB), no stream deadlines → slowloris
(idle deadlines both sides), one shared `.p2p.part` let a hostile seeder poison honest transfers → forced
permanent mirror fallback (per-peer pid-scoped partials, no cross-peer resume), partials never cleaned
(removed on exit). Seeding lifecycle wired into both launchers (`--seed-shards`: a stage seeds its verified
range from the SAME tunnel daemon; DHT setup backgrounded so it never delays 'tunnel up'). NOTE: live
on-ring `--seed-shards` NOT validated this pass — the ring boxes hold the OLD `/tmp/sidecar` (no `-seed`
flag); the WAN transfer proof (4090→box 5GB) stands independently. Next ring must push `/tmp/sidecar_new`.

**3. MARKET DECENTRALIZATION — design-panelled, c0mpute PR #16 OPEN for leyten (NOT merged — product direction).**
3 design agents (market-mechanism / distributed-systems / migration-pragmatics) + synthesis →
`c0mpute/MARKET_DECENTRALIZATION.md`. The shape for §10.1-B: **ask = µUSD per layer-token** (composes into
`splitTokens`, no settlement change); the **adequacy floor**, not the price unit, prices the slow-node
externality (pay-per-token already couples node revenue to ring speed); **deterministic auditable ring
formation** around the UNCHANGED `shard.plan` + signed RingCharters (members verify-and-sign, no auctioneer,
no consensus); **chunked client-ack settlement** (`paid = min(claim, client ack)`) closes the token-count gap
AND deletes the central settler; DHT/gossip discovery on the sidecar (same DHT the propagation ships). 6
migration stages, each locally testable + visibly decentralizing (flagship: a swarm forms with the
orchestrator DEAD). Honest central-residue table + hard problems named (emissions wash-trading is #1 — design
before any emission schedule). Left OPEN — it's a product-direction decision.

**REGRESSION RING PASS — the loop STILL SERVES (control-plane changes did NOT touch the decode hot path).**
Warm 5×5090 EU ring (BG→UK→NO→DE→BG, 0/10/13/13/13/13 split), verified pull landed clean on all 5 (signed
manifest, session publisher key). Novel-reasoning job: 200 coherent tokens, **2.90 tok/s** (g=2.24, n-gram
mean_accept 1.24/8, transport 79% — a high-RTT ring, s4 hop ~51ms), **5 signed per-stage receipts ALL VALID +
full [0:62) coverage + chain intact**. That novel number matches prior real-ring novel prompts (3.84 on
2026-07-07, perf-not-the-point). Copy/draftable job: 220 tokens, **5.58 tok/s** (g=4.29, mean_accept 3.29/8, transport 78%). ⚠️ **This run UNDER-MEASURED — graph-aux was OFF** (launched `M25_EAGLE=1` but NOT `M25_CUDA_GRAPH`/`M25_STATIC_KV`), so it ran the slow eager path (~157ms/traversal) on a high-RTT ring. With graph-aux this ring class does **~24 decode-weighted / 30-32 reasoning-heavy** (PR #25, proven). EVERY perf ring MUST launch `M25_CUDA_GRAPH=1 M25_STATIC_KV=1 --kv-maxlen 16384`. Receipts ALL VALID regardless. The regression QUESTION — did the
session's control-plane/docs changes break serving? — is answered NO (it serves + verifies; the SPEED was mis-measured by omitting graph-aux, my error).

**RING: TORN DOWN (instances-v1==0 verified) — all 5 ring + 4090 + 3090 probe boxes.** Vast credit ~$127 start (this session used ~$5.13). Live iids tracked in
/tmp/live_iids.txt. The 4090 seeder (iid 44216090) + 3090 (44216970) are separate probe boxes.

**NEXT (torrent critical path continues):** (a) push `/tmp/sidecar_new` into the ring bootstrap (ring_up +
scatter_pipe) so `--seed-shards` is live-validated on a ring (peer pulls its range from a ringmate, mirror
only as fallback); (b) `MlxRuntime` — the flagship any-device build (MLX has the model, the 4-bit artifact,
per-layer callability + native bf16; days not weeks) + per-format manifest v2; (c) market migration stage 1
(asks in announce + cheapest-adequate formation) IF leyten greenlights PR #16; (d) the cross-kernel cutlass
xcheck one-box run (confirm marlin-vs-cutlass cosine ≥ 0.99 so the spot-check won't false-flag a 4090).


### ⇒ 2026-07-08 (STRATEGIC SHIFT — leyten) [EXECUTED this session — see the block above; kept for the task detail]: make the PoC TORRENT-LIKE.
leyten's direction: the recent sessions drifted into hardening + privacy, which is NOT the key path. The
north star ([[north-star-torrent-for-compute]]) is a permissionless, torrent-like compute fabric. Privacy is
DEFERRED (PoC runs fully open); trust/hardening is DONE ENOUGH. **The next session works these three, in
this order of leyten's emphasis — all on-thesis "make it more torrent":**

**1. HETEROGENEOUS GPUs — let cards OTHER than the 5090 join swarms (which GPUs keep usable speed?).**
   - The planner ALREADY handles the easy half: `assign_layers`/`select_ring` (shard/topology.py) size each
     node's block by VRAM + measured per-layer time, so a smaller/slower card just gets FEWER layers. What's
     missing is (a) making the ENGINE run a shard on non-Blackwell arch, and (b) admission tiers + a compute probe.
   - **THE gating question = NVFP4 kernel portability.** The checkpoint is `nvidia/MiniMax-M2.5-NVFP4` (4-bit
     experts, Blackwell-native sm_120). Can a 4090 (Ada sm_89) / 3090 (Ampere sm_86) run the NVFP4 MoE at all,
     or only Blackwell (5090/5080/5070)? **Lead found this session:** the MoE backend is already selectable —
     `M25_MOE_BACKEND` = `cutlass | emulation | marlin` (phase0/m25_stage.py ~L213). cutlass is the sm_120 fast
     path; `emulation`/`marlin` are the likely non-Blackwell fallbacks (dequant NVFP4→fp8/bf16). VERIFY on a
     single rented 4090/3090 (~$0.30, one-box probe): does the shard load + run a block, at what VRAM cost
     (4-bit→8/16-bit inflates 2-4×, so a 4090 holds fewer layers) and what layer_ms.
   - **Usable-speed frame:** tok/s = g / T_traversal; a slow card raises its stage compute. Decode is
     memory-bandwidth + CPU-launch bound (5090≈1792, 4090≈1008, 3090≈936, 4070Ti≈672, 3060≈360 GB/s). Produce
     an ALLOWED-GPU tier table: which cards are ring-worthy (fine with fewer layers), which only fit off-ring
     roles (seeder/verifier), which are too slow. Build list: per-node compute PROBE feeding layer_ms, the
     per-arch kernel selection in ModelRuntime (M25_MOE_BACKEND per node), mixed-precision-stage numeric
     compatibility on the wire/receipts, VRAM+compute admission floor per GPU class.
   - **DECIDED (leyten): the minimum usable-speed bar = 20 tok/s+.** The allow-list is "cards that keep a ring
     at ≥20 tok/s." This is REACHABLE single-stream: graph-aux (PR #25, `M25_CUDA_GRAPH`+`M25_STATIC_KV`) proved
     **~24 decode-weighted / 30-32 reasoning-heavy** on a good scattered EU ring (reason-math 32, agentic 31) —
     20+ sits BELOW that, comfortably. (The old "~10-12 ceiling" is PRE-graph-aux and STALE — do not cite it;
     see line ~825 PROVEN + [[graph-aux-raised-single-stream-ceiling]].) Draftable-verbatim (50-80) and
     batched-aggregate (155) go higher. Heterogeneity must not be what drops a ring below the bar, and EVERY
     perf ring launches graph-aux ON. Do NOT co-locate to manufacture a number
     ([[never-colocate-usable-speed-on-scattered]]); the levers are scattered-native (graph-aux, draftable, g,
     RTT-ordered topology, batched).

**2. NODE PROPAGATION for shards — BUILD + TEST the torrent weight-fetch (the "torrent" half).**
   - Today a joining node pulls its verified layer range from HF (`MirrorProvider`). The torrent path — pull
     from PEERS holding the range — is the seam `Libp2pProvider` (shard/fetch.py:157), wired end-to-end but
     STUBBED (raises `ProviderUnavailable` → falls back to mirror). Verification is unchanged (re-hash vs the
     signed manifest), so an untrusted peer source is SAFE — swapping the source changes nothing about trust.
   - UNBUILT: the actual transfer — **provide/seed** (announce to the DHT this node holds CID X for layers
     [lo,hi)), **find-providers** (discover peers with a CID), **block-exchange** (fetch bytes peer→peer,
     bitswap-style), + the **fallback chain** (peer → HF origin) and the seeding lifecycle. CID contract is
     live (`sidecar -fetch-cid`, CIDv1 raw sha2-256). **GOTCHA:** the Go sidecar SOURCE isn't in the repo
     (binary only) — step 1 is sorting the sidecar (get/rebuild its source, or build the transfer in Python
     libp2p). Much of this is CPU-testable with **2 local peers, no GPU** (seed a manifest's shards from peer
     A, fetch+verify on peer B, kill A → fall back to HF). Build it, test it locally, THEN on a ring.

**3. NETWORK STRUCTURE — decentralize the orchestrator (leyten wants NO central orchestrator).**
   - Today c0mpute has a CENTRAL control plane (admission, placement via `shard.plan`, per-swarm coordinator,
     settlement). NETWORK_ARCHITECTURE.md §5/§10.1 frames the fork: (A) central scheduler first vs (B)
     market-as-optimizer; the control plane holds NO weights/keys BY DESIGN so it can decentralize.
     **DECIDED (leyten): go for the MARKET (option B)** — nodes PRICE their compute, requests route to
     cheapest-adequate, supply/demand balances the network with NO central planner; the market IS the
     self-optimizer (most decentralized end-state, dovetails with economics §6). Central-first (A) is NOT the
     path. Map where centralization lives (discovery, admission, placement, coordination, settlement) and
     design the market migration: **DHT discovery** (libp2p is already the transport — announce/find over the
     DHT, no central registry), **self-forming swarms** (nodes gossip capability + PRICE and locally form a
     coverable low-RTT ring vs. a central planner call), **coordinator** (per-swarm head or elected, not a
     central driver), **settlement** (peer-attested receipts / on-chain vs a central settler), and the
     **pricing/bidding** clearing that balances supply↔demand. Decide what's cheap for the PoC vs. genuinely
     hard; stage it. This is the deepest build — design-panel it, surface the shape to leyten before committing.

**OWED — a REGRESSION "does it still work" ring pass** (leyten flagged, rightly, that it's been days + many
changes with no live validation). NOTE: this session's changes were placement/control-plane + docs, NOT the
decode hot path (m25_pipe / m25_scatter_pipe untouched), so serving speed should be unmoved — but we still
owe ONE warm ring pass confirming the loop serves at the expected **~24-32 tok/s single-stream WITH graph-aux**
(`M25_CUDA_GRAPH=1 M25_STATIC_KV=1`, PR #25 — NOT the ~10-12 that predates it) or the higher batched/draftable
numbers before trusting the stack. Do it early, and NEVER omit graph-aux on a perf ring.

**RINGS — TEST ON FULL RINGS FREELY (leyten): the vast balance (~$130) is there to USE.** Don't be timid
about spinning real rings to validate #1 (heterogeneous cards live) and #2 (peer propagation live) — the
one-box + 2-local-peer probes are the CHEAP first step, but graduate to a full ring without asking. Use the
DETACHED watcher (scratchpad/ring_watcher.py + health_probe.py: CUDA-803 + no-boot recoveries baked in),
keep rings WARM through an investigation, kill zombie/dud extras, track live iids. Tear down when the work
is banked.

**Shipped-and-parked (do NOT keep polishing):** the safety rails — boundary pinning (opt-in private tier),
graded reputation, layer-block spot-check — are MERGED (shard #57, c0mpute #15) and the PoC runs fully open.
Cheat-detection (receipts + auditor spot-check + reputation) stays on; privacy is a documented deferred gap.
See the block below for detail; it is DONE, not the focus.

### ⇒ 2026-07-08 — PoC RUNS FULLY OPEN (privacy deferred); cheat-detection rails BUILT + PROVEN
**leyten's call (2026-07-08):** for the PoC, **any machine joins any swarm and holds any slice.** Prompt
privacy is a **known, accepted limitation** — mandatory boundary pinning would need trusted nodes in ~40%
of every ring and re-introduce the supply bottleneck open admission exists to avoid. What runs on the open
network is **CHEAT detection** (needs no trusted stage in a ring): **receipts** (skip/fabricate/replay →
pay nobody), a **spot-check** verified by a we-run **auditor** (a few boxes we operate off-ring — the
sharded canary analogue, zero supply tax), and **graded reputation** (kicks repeat cheaters at admission).
Cheat-detection catches a node doing the work WRONG; it can't catch one doing it right while copying the
prompt (snooping is passive) — that residual IS the deferred privacy gap. `DEFAULT_SWARM_CONFIG.privacy =
null` (open). The boundary-pinning rails below stay BUILT + PROVEN as the **opt-in private tier** for later.

All three rails are built, CPU-tested, adversarially fuzzed, and proven end-to-end against the REAL shard
seams (the pinning path via `rails-demo.ts`; the open path via `swarm-loop-demo.ts`).

**SHIPPED — shard branch `net/boundary-pinning` (5 commits; suite 233 green):**
- **Boundary-layer pinning** (`select_ring(trusted={...}, boundary_in, boundary_out)` + `shard.plan`
  `privacy=` seam). The head/tail roles and every stage holding a `[0,b_in)` or `[62-b_out,62)` layer
  must be trusted; strangers hold only deep-middle. Grounded in the inversion literature (2602.16760,
  2507.16372): naive prompt-token recovery ~59%→35% by 8 layers, output side leaks worse (logit lens)
  → default 8/8, `b_out ≥ b_in`, floor 4/4, regulated tier 12/12. Trust is a CONSTRAINT not a score;
  `trusted=None` is byte-identical legacy (goldens unchanged). Order search is ends-constrained +
  boundary-spill-aware (`_pin_floors`): a trusted contiguous prefix/suffix covers the window even when
  a single end node is too small. FAILS CLOSED (no trusted node / untrusted require → None).
- **Torch-free challenge seam** (`python3 -m shard.challenge`, `compare_sketches`): the control plane
  judges spot-check sketches with no CUDA stack (the GPU nodes produce them). Fail-closed on malformed.
- **Adversarial verification (self-run + a deep subagent, ~8500 machine-checked specs):** the privacy
  guarantee is **SOUND** — 0 untrusted nodes on a boundary across every returned spec. The review found
  **5 non-leak bugs, ALL FIXED + regression-locked**: (1) a false-infeasible when the window spills a
  small end node (self-found via a brute-force oracle); (2) a **FAIL-OPEN seam** — plan.py read the
  `trusted` flag by truthiness, so a string `"false"` would admit strangers → now strict bool
  (fail-closed); (3) an overlap false-infeasible (b_in+b_out ≥ n_layers double-counted floors, refused
  even all-trusted rings); (4) oversize windows didn't clamp; (5) duplicate node ids collided. After the
  fixes select_ring matches the brute-force feasibility oracle **exactly** on 5000 cases that include the
  overlap/oversize regime (0 mismatch, 0 false-infeasible, 0 leaks).

**SHIPPED — c0mpute branch `net/safety-rails` (2 commits; full tsc clean):**
- **GradedReputation** (`lib/orchestrator/reputation.ts`) — per-node score gating `boundary`
  (STAKE-gated: score alone never earns it) / `middle` (open-admission default) / `relegated`
  (off-stage) / `rejected` (refused at announce). Recent-behaviour scoring like the canary ban; 2
  consecutive spot-check fails reject. Replaces the binary ban for shard nodes. snapshot/restore.
- **SwarmManager wiring** — `SwarmConfig.privacy` (default 8/8) feeds `shard.plan` a per-node `trusted`
  flag ASSIGNED from stake+reputation (never self-reported); fails CLOSED without a trust oracle; a
  plan that put a stranger on a boundary stage is rejected before any assign is emitted. `startSpotCheck`
  → `shard.challenge` judges a stranger's redundant recompute, verdict feeds reputation, silent suspect
  fails on timeout, a failed check degrades the swarm. Settlement/churn also feed reputation.
- **Proven** headless (`scripts/rails-test.ts`, 18 assertions) AND end-to-end vs the REAL shard.plan +
  shard.challenge (`scripts/rails-demo.ts`): 3 staked + 4 stranger nodes, the real planner keeps every
  stranger off the boundary layers, the real spot-check catches a faked block (cosine ≈ 0) → struck → relegated.

**RING WATCHER (task #2) — the two live-pass fault-recoveries, CPU-tested** (`scratchpad/health_probe.py`
self-tests pass; `scratchpad/ring_watcher.py`): (a) CUDA-803 dud now caught by a real `torch.cuda.init()`
+ alloc probe (the old `nvidia-smi`+VRAM gate PASSED it → crashed at launch); (b) NO-BOOT box (transient
scp/ssh drop → no `boot.log`) distinguished from PENDING so the watcher re-bootstraps/swaps instead of
polling the full 35-min window. Wraps the proven rent_pool→ring_up flow; live-validated only on a ring.

**⇒ PRIVACY FORK — DECIDED (leyten 2026-07-08): run the PoC fully OPEN, defer prompt privacy.** Mandatory
pinning was rejected (taxes open supply). The pinning rails stay built as the opt-in private tier; the one
economics decision that remains leyten's — WHO counts as a staked/trusted node — only matters when that
private tier is turned on, NOT on the PoC critical path (`GradedReputation.isStaked` seam →
`lib/onchain-staking.ts`).

**⇒ MERGED to master (2026-07-08): shard PR #57 + c0mpute PR #15** (both squash-merged, branches deleted).
The rails are live in the tree, PoC runs fully open. Next, NOT rails: RTT probe + auto-form trigger, pay
wiring onto `recordEarning`, token-attested pay, P2P shard propagation (torrent half), and standing up the
**spot-check auditor node(s)** (the we-run recompute box the open-PoC spot-check verifies against). A LIVE
fully-open ring end-to-end (announce→place→pull→serve→settle with receipts + a live spot-check) is the
next real-hardware milestone. **RING: none live.** Vast credit ~$130 (this session: $0).

### ⇒ 2026-07-07 (night) — REAL-RING PASS PASSED: the permissionless loop closed end-to-end on LIVE GPUs
On a real scattered 5×5090 EU ring (NO→NO→LV→DE→DK, distinct subnets, no co-location), the whole loop ran:
**place (`select_ring`) → VERIFIED PULL (signed manifest, #48 fix) → auto-form → serve → SETTLE (`shard.verify`)
→ pay per shard.** Receipt `docs/receipts/m25-realring-loop-20260707.md` + the signed set
`m25-realring-loop-receipts-20260707.json`:
- **Verified pull proven at scale:** each stage pulled ONLY its layer range from the signed
  `nvidia/MiniMax-M2.5-NVFP4` manifest (62L, 29 weight shards, 139.9GB), re-hashed every byte; all 5 landed
  clean (24–33GB/box), the #48 resume fix held on every one.
- **Served** 160 tokens over the ring (3.84 tok/s — novel prompt, n-gram g=1.0, high-RTT hop; perf not the point).
- **Settled on REAL receipts:** 5 signed per-stage receipts, every sig VALID, activation chain intact
  (`out_root[i]==in_root[i+1]` across all 5), full [0:62) coverage, per-job nonce; `shard.verify` ok=True; the
  per-shard-per-token split fired 26+34+34+33+33 = 160. **PR #53** added `SHARD_RECEIPT_DUMP` (the coordinator
  exports the receipt set for the settle seam) — merged, proven live.
- **OPS lessons for the ring watcher (task #2, still owed):** (a) a box that fails to bootstrap (transient
  scp/ssh drop → no `boot.log`) is polled forever — detect + re-bootstrap/swap; (b) a GPU-driver dud (`CUDA
  803`) passed the VRAM health check but failed `torch.cuda.init()` — health checks must probe CUDA init. Both
  hit this pass and were hand-fixed. **RING TORN DOWN** (instances-v1==0 verified); ~$? of vast credit used.
- **Publisher key was ephemeral/test** — the durable manifest-signing identity is still a c0mpute-catalog call (leyten's).

**DECIDED + LOCKED (leyten):** admission = **OPEN** (proven VRAM floor; placement decides role — open supply is
the endgame, avoids a curated bottleneck), pay = **by layers** (ungameable; boundary-role premium likely v2),
coordinator = central (PoC). `DEFAULT_SWARM_CONFIG = {open, layers}`; `c0mpute/PERMISSIONLESS_LOOP.md` = the loop spec.

**⇒ NEXT SESSION (in order) — turn the proven mechanism into a safe, self-running OPEN service. The loop MECHANISM
works (sim + live); what remains is the trust rails + automation, NOT more mechanism:**
1. **SAFETY RAILS = the OPEN-launch blocker (do FIRST).** Open ADMISSION ≠ open TRAFFIC — do not serve untrusted
   jobs until: (a) **boundary-layer pinning** — placement keeps the leaky embedding/final layers (35–59% of a
   prompt is reconstructable from their activations) on staked/trusted nodes; strangers hold only deep-middle. A
   `select_ring` trust-pin input it doesn't take yet (a `require`-like constraint + a per-node trust flag). (b)
   **graded reputation** — a per-node score gating roles, replacing c0mpute's binary ban ([[c0mpute-reputation-needs-upgrade]]).
   (c) **layer-block spot-check** — seeded redundant recompute of a random block on a trusted node
   (`shard/challenge.py` primitive + 13 tests EXIST; wire into c0mpute placement/verification). Until live, run
   rings only over trusted/own boxes even though admission is open.
2. **Ring watcher (task #2)** — detached, self-healing provisioning with the TWO fault-recoveries the live pass
   exposed: (a) a box that never bootstraps (no `boot.log` → currently polled forever) → detect + re-bootstrap/swap;
   (b) a CUDA-driver DUD (`Error 803`) that passes the VRAM health check but fails `torch.cuda.init()` → health
   checks must probe CUDA init, not just free VRAM. Provisioning is SLOW (~10min) + flaky (2 of 5 boxes needed
   swapping this pass). Prior art: detached `swarm_master.sh` style + this session's `scratchpad/ring_pass_*.sh`.
3. **Socket node-agent** — a node ANNOUNCES to the orchestrator over the wire + runs `fetch_block_range` (verified
   pull) + `m25_scatter_pipe` on `swarm:assign` (this pass drove assign→pull via SSH push; the c0mpute-worker is
   whole-model today). Turns the loop from driver-script-orchestrated into truly node-driven.
4. **RTT probe + auto-form trigger** (`formSwarm` needs a measured RTT matrix over the candidate pool + a trigger:
   pool reaches a coverable set / demand) · **pay wiring** (map the verified per-shard split onto `recordEarning()`;
   split is correct, only the $ mapping waits) · **token-attested pay** (receipts attest WHICH layers ran, not the
   token COUNT — coordinator's number trusted up to a cap; bind it to the job before real open payout).
5. **P2P shard propagation (task #4)** — swap `MirrorProvider`→`Libp2pProvider` so a joiner pulls its verified
   range from PEERS not HF (the "torrent" half). Python seam READY; UNBUILT = the Go sidecar DHT transfer
   (provide/find-providers/block-exchange) + peer→HF fallback. Sidecar source not in the repo (binary only) — may need it.

**RING: none live** (vast instances-v1==0 verified). Vast credit ~$130 (real-ring pass used ~$5–8). Publisher key is
ephemeral/test — the durable manifest-signing identity is a c0mpute-catalog call (leyten's).

### ⇒ 2026-07-07 (later) — [SUPERSEDED by the current-state block above — kept for the PR/build trail] loop first demonstrated in sim; #46 fixed
Acted on the PIVOT below. The loop now runs end-to-end — **announce → admit → PLACE → assign → (pull/form/serve
sim) → SETTLE → pay per shard** — against the REAL shard decision code, and dishonest settlements pay NOBODY.

**SHIPPED (merged to shard master):**
- **#48 — the #46 over-sized-download bug FIXED.** Root cause pinned exactly: the overshoot was `have_partial(19MiB)
  + full_total` — a resume the CDN answered with the WHOLE body tagged 206 (Content-Range from 0, redirect drops
  the Range), and `_download` trusted the bare 206 and APPENDED. Fix: place the body by its **Content-Range**, cap
  writes at the manifest size, drop-and-restart on an unplaceable range. Adversarially verified SOUND (overshoot
  class eliminated by construction; residual soft cases caught by the sha256 backstop). The verified pull is safe now.
- **#49 `shard.plan` (PLACE seam)** + **#50 `shard.verify` (SETTLE seam)** — the graduation bricks. `plan_ring`
  lifts ring_up's calibration (VRAM reserves, launch-bound layer_ms, head placement) into a tracked, tested fn;
  `python3 -m shard.plan` / `shard.verify` are JSON-in/out CLIs so a TS orchestrator drives the same proven
  `select_ring` + `receipt.verify_coverage` over stdio (deps still one way). The three shard pieces — plan, fetch
  (verified), verify — are contract-compatible and now callable from the network layer.

**c0mpute PR #14 (`net/permissionless-loop`) — since MERGED (settlement hardened; open+layers locked):**
The graduation into the orchestrator. `lib/orchestrator/swarm.ts` (`SwarmManager`: admit → candidate pool →
`formSwarm` calls the plan seam + emits `swarm:assign` → `markReady` → `settleJob` calls the verify seam + splits
pay per shard) + `swarm-seam.ts` (spawns the shard modules) + `swarm-loop.ts` wired into `orchestrator.ts` in ONE
additive constructor call (+20 lines; the whole-model worker path is untouched). `scripts/swarm-loop-demo.ts`
proves it with NO GPU against the real seams: 6 nodes announce, a 4GB node refused, `shard.plan` forms a 5-stage
ring (coordinator = most-central node, slow low-uplink 4090 relegated to verifier/standby), a 480-tok job splits
per shard (77+101+101+101+100=480), and replayed-nonce + coverage-gap settlements pay nobody. Full project tsc clean.
Built in an isolated worktree; leyten's uncommitted c0mpute work (onchain-staking.ts) was left untouched.

**⇒ 2 FORKS — since DECIDED + LOCKED (open admission + pay by layers; details in the current-state block above):**
- **A. Admission — curated vs open** (§10.3): curated allowlist (betanet-first, the default) vs open proven-VRAM
  floor (permissionless). Mechanism is identical; only *who may announce* differs. Recommend build-both (done),
  run first live rings curated, flip to open on his word.
- **B. Pay split across stages** (§6): `layers` (proportional to work, default) vs `equal`, with room for a
  boundary-role premium. Recommend `layers` for the PoC, revisit with the privacy stance.
- (Decided for the PoC, not a fork: coordinator runs the planner centrally — §10.1, the likely A→B path.)

**NEXT (this list is DONE/superseded — the real-ring pass RAN and forks are LOCKED; the current NEXT is in the block at the top):**
1. **Real-ring pass** = the ultimate proof. Needs two things still owed: (a) the **watcher-based ring
   orchestration** (task below / the PIVOT's ask — still NOT built; needed before spinning a ring); (b) a
   **sharded node-agent** that listens for `swarm:assign` and runs `fetch_block_range` (verified pull) +
   `m25_scatter_pipe` (the c0mpute-worker is whole-model today). Then: announce real vast boxes → orchestrator
   places → they pull verified → ring serves → receipts settle. Costs vast $ — gate on leyten.
2. **RTT probe + auto-form trigger** — `formSwarm` needs a measured RTT matrix over the candidate pool (a short
   node-probe round) + a trigger (pool reaches a coverable set / demand). Manager exposes `formSwarm(...)` for it.
3. **Pay wiring** — map the verified per-shard split onto `recordEarning()` once fork B is decided (the split is
   already correct; only the $ mapping waits). **RING: none live** (vast instances-v1 == 0 verified this session).
4. **P2P layer-shard propagation (leyten-flagged 2026-07-07) — the "torrent" half; removes the centralized HF
   download so a joining node pulls its verified range from PEERS.** Python side is READY: `shard/fetch.py` has the
   `Libp2pProvider` seam (stubbed) + `fetch_block_range(provider=...)` is pluggable; verification (hash vs signed
   manifest) is unchanged, so an untrusted peer source is safe — swapping `MirrorProvider`→`Libp2pProvider` is
   additive and changes nothing about trust. `weight-seeder` is already a `select_ring` relegation role (a
   low-uplink node relegated to seed shards). UNBUILT = the **Go sidecar transfer** (roadmap step 8): PROVIDE/seed
   (announce to the DHT this node holds CID X), FIND-PROVIDERS (discover peers holding a CID), block exchange
   (fetch bytes peer→peer, bitswap-style), + the fallback chain (peer → HF origin when no peer has it) and the
   seeding lifecycle. NOTE: the sidecar source isn't in the shard repo (binary at `/tmp/sidecar`) — may need the
   sidecar codebase. Slots after/parallel to the real-ring pass. See task #4.

### ⇒ 2026-07-07 (late) — [DONE this session — see the current-state block at top] PIVOT: build the permissionless loop, use REAL orchestration for rings
The engine-hardening + trust-primitive work is DONE and over-invested (a review found the day drifted into
"safe CPU-testable" engine internals while the actual PoC gap — the permissionless network driving a sharded
swarm — sat at zero). **NEXT SESSION'S JOB = the permissionless loop, NOT more engine hardening:** a minimal
end-to-end path where a node ANNOUNCES capability → c0mpute ADMITS + PLACES a layer range → it PULLS that range
verified (#45) → the ring AUTO-FORMS + serves → per-shard-per-token METERING fires. `select_ring` (built in
shard/topology.py) must GRADUATE into the c0mpute orchestrator (`c0mpute/lib/orchestrator/orchestrator.ts` today
drives whole-model single-GPU workers, NOT sharded swarms). Surface the design/economics forks (curated-vs-open,
pay model, coordinator trust, shard↔c0mpute boundary) to leyten — don't guess them.

**RING PROVISIONING — STOP hand-driving `rent_pool`/`ring_up` + manual polling.** It cost this session two
rabbit holes (a truncated-download bug, then a size mismatch) babysat by hand. Use proper WATCHER-based
orchestration (automated fault detection + recovery), like the detached `swarm_master.sh`-style flow, not inline
poll-and-debug. Sort the robust provisioning approach BEFORE the next ring session.

**Verified weight-fetch (#45/#46) status:** the deploy path (`fetch_block_range` + `phase0/m25_pull_verified.py`)
+ manifest generate/verify are shipped and CPU-tested; a real 5GB M2.5 shard pulled + sha-matched the signed
manifest launcher-side. But the full-ring validation is INCOMPLETE and **#46 (the truncated-download resume fix)
has a bug: it produced OVER-sized downloads on the live ring (5,018,451,080 vs the manifest+HF x-linked-size
4,998,528,136).** Investigate before trusting the verified pull (likely the Range resume double-appends, or HF
returns a full 200 body on a 206 request). The real M2.5 manifest generates in seconds via
`publish_manifest.py --hf nvidia/MiniMax-M2.5-NVFP4`. Ring TORN DOWN (instances-v1==0 verified).

**GIT stays clean (memory git-voice-solo-shard):** every commit/PR/tracked doc reads as leyten's OWN solo work —
no autonomous/AI/loop framing.

### ⇒ 2026-07-07: "SAFE TO BE PERMISSIONLESS" sweep — 7 hardening PRs + warm-ring validation
Cleared the safe, CPU-testable hardening backlog. **Merged #34-#40, all CPU-tested + adversarially verified:**
**#34** churn (F6 per-reply decode heartbeat → blip failover in seconds not up-to-1800s; F8 real-`serve()`-tail
churn test, teeth-checked vs the pre-#26 bug). **#35** wire DoS (`MAX_FRAME` cap + tensor shape/blob validation →
closed empty-blob→`torch.empty(huge)` OOM, both codecs; pre-fix allocated a 1M-elem tensor from 0 bytes). **#36**
the MOAT (TIER 2.2): per-job nonce (anti-replay) + `out_root==in_root` chain binding, coordinator-trusted-challenge,
gated `not M25_FP8_WIRE`. **#37** gateway (client-disconnect no longer re-runs the whole gen via `ClientGone`;
stream write timeout bounds a stalled client; `reasoning=False` stream-dup fixed). **#38** batched-decode KV bound
guard (no OOB scatter crash). **#40** adversarial tests for the verified weight-fetch trust root
(`shard/fetch.py`/`manifest.py`, 14 tests: tamper+delete / path-traversal / bad-sig / wrong-pin / cache re-hash —
the primitive had ZERO coverage). `select_ring` false-infeasible was already fixed (stale roadmap). Also fixed the
stale tok/s number (→ ~24/~30). Suite **176 green**.

**WARM-RING VALIDATION (live 5-stage EU ring, ~$3-4, receipt `docs/receipts/m25-warmring-validation-20260707.md`):**
provisioned via rent_pool→ring_up on the current master code, and PRODUCTION-VALIDATED the two never-before-live-
tested pieces. **① Receipt moat (#36) LIVE:** the 5 signed per-stage receipts chain EXACTLY (`out_root[i]==in_root[i+1]`
across all 5 scattered stages) — PROVE verdict ALL valid + full coverage + nonce + chain (lossless wire). **② Churn
(#34/#26) LIVE:** killed the coordinator MID-DECODE, a NEW coordinator completed a job on the same ring with NO
re-warm. graph-aux rep2 skipped (mechanism-verified; this ring slow-CPU/lossless, won't re-pin 24 without a
STATIC_KV re-warm). **OPS LESSON banked:** FWD_RET/FWD_RING tunnels take ~3-5min to establish after warm (initial
dial refused → CONN DIRECT later) — the first post-warm coord WEDGES until they're up; give it a long timeout /
background it, don't kill early. Ring TORN DOWN (instances-v1==0 verified).

**NEXT — the safe CPU-testable hardening backlog is DRAINED; what remains is a bigger tier (in progress):**
- **Endpoint receipt bindings — DECIDED: DEFER, not building (2-perspective trust-model review).** The token
  binding is security theater under coordinator-trusted-challenge: `tok_out_root` (tail) is a TAUTOLOGY — the
  coordinator observing/using the reply tokens IS the binding, there's no independent correct-answer oracle;
  `tok_in_root` (head) only proves a node SAW the tokens (handed to it for free), and the coordinator can ALREADY
  bit-exactly recompute the head's `in_root` from `embedding(token_ids)` (embedding is a pure gather → hardware-
  independent) if it wants — strictly stronger than `tok_in_root`, no new field. Neither proves COMPUTE; a node
  hashes the correct endpoint values it already holds while skipping the matmuls. Compute-honesty (the real gap)
  is `shard/challenge.py`'s job — seeded redundant-recompute + cosine spot-check — **now covered by 13 adversarial
  tests** (`tests/test_challenge.py`: honest recompute passes, lazy/constant/wrong block fails, ULP-drift tolerant,
  rel-norm guard). Also boundary-law: token/I/O semantics belong in the c0mpute economics layer, not the engine's
  activation receipt. REVISIT only if coordinators become untrusted for output attribution → then build a
  client-facing receipt-of-service at the c0mpute layer over (prompt, answer), never a `tok_out_root` in the chain.
- **TIER 2.4 weight-fetch deploy-wiring** — the trust root is now fully tested BOTH sides: verify (`fetch_block`,
  #40) AND generate (`publish_manifest`, #44 — build_from_dir round-trips through fetch_block). And generating the
  real M2.5 manifest is a SECONDS-long `publish_manifest.py --hf <repo>` command (HF LFS oid == sha256 → no 115GB
  download). REMAINING is not a build blocker but two decisions/steps: (a) PUBLISHER IDENTITY — who signs the M2.5
  manifest + where the key/catalog pin lives (a c0mpute-catalog / trust-root call, leyten's); (b) swap `ring_up`'s
  `snapshot_download` for `fetch_block(manifest)` and validate the verified pull on a ring. Needs (a) + a ring.
- **FWD_RET robustness** — the return path could dial the tail directly instead of via the head sidecar (the
  ~3-5min tunnel-establish flakiness above is the motivation). libp2p/sidecar infra change — can't CPU-validate.
- **keep-warm jitter A/B** — needs a jittery/residential (DoubleZero) bare-metal path.
- **The real remaining project = Bucket B: the c0mpute permissionless loop** (join→admit→place→run→pay driving the
  engine). `select_ring` is built in shard; it needs to graduate into c0mpute. This is the north-star gap.

### ⇒ 2026-07-05/06: FOUR things SHIPPED to master (graph-aux + churn fix + safe_kill + keep-warm); ring DESTROYED
Perf-lever evening → became a perf + robustness sweep. All merged to master via clean PRs (no Claude trailer):
- **PR #25 graph-aux** — CUDA-graph EAGLE-aux compatibility. THE win: on slow-CPU boxes stage compute drops
  **157→40ms/traversal (~4×, drift-proof per-stage timing)**; ring decode-weighted **chain 13.6→+graph 23.7
  = +74%** (clean 4-rep rotated; the earlier raw +167% was WAN-drift-inflated baseline). Reason-math 18→32,
  agentic 14→31. Runtime per-job toggle via reset op; bounded capture set (M25_GRAPH_MAX, default 16);
  OOM-safe. Requires M25_STATIC_KV; use `--kv-maxlen 16384` on graph rings (graph pools pressure the fat tail).
  GPU-validated (research/graph_aux_check.py: graph≡eager-manual for h+aux, aux-freshness proven).
- **PR #26 return-channel churn fix** — PoC-CRITICAL. The tail closed a LIVE coordinator's return channel on
  ANY internal-ring blip (kept ret only when already stale) → forced full reconnect → raced the return-tunnel
  recovery → WEDGE. Fatal for permissionless (internal-leg blips are the steady state; reproduced repeatedly
  live). Fix: keep ret across a predecessor blip, hold session `stale` until the next reset re-arms
  (stale=carried on re-accept). Adversarially reviewed correctness-safe; validated live (a mid-decode blip
  healed via coordinator retry instead of wedging). **DEBT (review F6/F8):** a short per-reply recv heartbeat
  so blip failover is seconds not up-to-timeout; an in-process serve()-tail churn test (fake_ring mocks the tail).
- **PR #27 safe_kill** — permanent fix for the self-killing `pkill -f` footgun (kills its own launcher shell
  whose cmdline contains the pattern → silent launch-wipe; bit us ~5× this session). `phase0/safe_kill.sh`
  excludes self+ancestors, deployed to every box via push_code. Memory rule [[never-raw-pkill-f-use-safe-kill]].
- **PR #28 keep-warm** — cwnd keep-warm noops on idle legs (TCP slow-start-after-idle collapses cwnd between
  tokens → 2-4× slower frames; measured on the leg probe). A CONSISTENCY/tail-latency lever for jittery
  public-internet paths, NOT throughput. **Default-ON for --serve (interactive gateway)**; off for measurement
  paths. Neutral on calm rings (4-rep A/B ratio 1.01-1.04 — an earlier 'breaks pipelining' read was drift).
  **DEBT: ring-level benefit proven only at TCP layer; owe a keep-warm ON/OFF A/B on a jittery/residential path
  (ties to the DoubleZero pilot) before defaulting on beyond interactive.**

**DoubleZero assessment banked** (Austin Federa contact, memory [[doublezero-pilot-assessment]]): thesis-
compatible underlay, ZERO engine changes, but median tok/s gain is under the noise floor; the real prize is
tail-latency/jitter elimination (DZ p99≈median) + flagship-AI-tenant partnership. NOT feasible on vast
(GRE/no-NAT); needs bare-metal (HOSTKEY/vshosting). It's the natural home for the keep-warm jitter A/B.

**RING DESTROYED** end of session (all 7 vast boxes, verified `instances-v1`==0; results banked). Next ring
via the proven rent_pool→ring_up 2-step (scratchpad). ~$? of the vast balance used this evening.

**NEXT (pick up here):** (1) Perf queue: graph-aux is THE lever landed (+74% mech-verified). Churn
follow-ups **F6 heartbeat** + **F8 serve()-tail churn test** are DONE (PR fix/churn-heartbeat-tail-test,
110 tests green): F6 = a per-reply DECODE deadline (`M25_REPLY_TIMEOUT`, default 20s) so a mid-decode
internal-leg blip fails over in seconds, not up-to-timeout (prefill + batched keep the full budget); F8 =
CPU coverage driving the REAL serve() tail through a pred blip (ret survives + stale gate) and a
mid-session hello_return (new ret adopted, pred+KV survive), adversarially verified to fail on the pre-#26
close-ret bug. Remaining perf/robustness: the **keep-warm jitter validation** (ON/OFF A/B on a
jittery/residential path — the DoubleZero pilot is the natural home). (2) FWD_RET return-tunnel setup
flakiness bit hard this session (slow/variable to establish
after warm; wedged several bench relaunches) — worth a robustness look (it's a single fragile libp2p tunnel;
the return path could dial the tail directly instead of routing through the head sidecar). (3) Rep2 of the
full 6-arm interleaved lever bench never completed cleanly (churn-wedge + tunnel flakiness) — graph-aux is
mechanism-verified so the verdict stands; a clean full rep2 is optional polish. (4) Bench tool committed:
`research/m25_lever_bench.py`. Receipt data lived in scratchpad (rep1_complete.json 36 jobs + confirm.json
48-job keep-warm×graph A/B) — bank to docs/receipts/ if a permanent record is wanted.

*(prior entry, superseded ops-wise; paper is PUBLISHED now per leyten:)*
### ⇒ 2026-07-03 (late): PAPER v1 DONE + the paper test evening banked; ring DESTROYED (results banked)
leyten green-lit the c0mpute technical report (author: leyten — c0mpute; inspired-by/positioned-against the
Dolphin AI 2-GPU LAN study). **`docs/paper/main.typ` → main.pdf (8pp, typst) is a complete v1** with
receipt-generated figures (figures.py: α-law calibrated MC, 3-arm bars, transport split). The test evening
(4 phases, ~2.5h, all banked in docs/receipts/m25-paper-*):
- **Interleaved 3-arm bench** (AR-null-drafter / chain / hybrid, arm order rotated per rep, one warm ring,
  calm window): AR = 4.8-5.0 tok/s FLAT g=1.00 (the latency wall, measured); interactive novel cells
  10.7-12.6 median (reason-math chain 12.6 [12.3..13.1]); 64/64 jobs receipts-verified.
- **Pure-verbatim pipelined regime: 70.7-87.2 tok/s single-stream** (ctx_table think-skip); **B=4 batched
  150-194 tok/s aggregate** @0.5-2k on this ring (session-9 receipt covers 155@16.4k on 6 stages). NEW
  systems constraint: batch KV vs stage fatness — the 13-layer tail (weights+KV+lm_head+prefill-logits
  transients) caps B=4 ctx ~12k where 6-stage rings hit 16k.
- **Verification is FREE: +0.05ms on an 11.7ms idle-box stage span (~0.4%)**; end-to-end on/off deltas are
  pure WAN drift (span data bounds the true cost 2 orders below).
- **FT timeline receipt**: kill -9 coordinator mid-decode t=14s → NEW coordinator completes a full job by
  t=48s, zero re-warm; receipts PROVE at t=60s.
- **NEW routing insight**: tree-vs-chain preference is RING-SPEED-DEPENDENT (tree's fixed surcharge loses
  on fast windows T≈250ms, wins on common T≈400ms) → T-aware router = cheap refinement, queued.
- Ops: ring + spares DESTROYED (verified 0 live); ~$45 of the $100 mandate used total. Next ring gets
  provisioned via ring_up's CPU probe (scratchpad/rent_pool.py + ring_up.py = the proven 2-step flow).
**NEXT:** (1) leyten reviews/publishes the PDF (repo docs/paper/main.pdf; site/X his call). (2) Perf lever
queue unchanged: CUDA-graph aux compat → lean codec → T-aware router + calm-window re-pin. (3) TIER-2
trust (freshness/binding — the paper's own Limitations names it).

*(prior same-day state, superseded ops-wise but numbers stand:)*
### 2026-07-03: THE GOOD-RING RECEIPT IS BANKED — 10-11 tok/s interactive reasoning MEASURED; fork resolved: path (a) EXECUTE
leyten picked **(a)**: execute to the ceiling, declare the honest number, re-point the perf narrative at
batched 155 agg + draftable/agentic. The receipt run happened same evening (receipt
`docs/receipts/m25-goodring-receipt-20260703.md` + the two arm JSONs; ~$6 spent, ~$100-mandate has ~$30 used):
- **RTT-measured, select_ring-planned 5-stage EU ring** (scratchpad/ring_up.py: pool→mesh-RTT→select_ring→
  ranged pull; head-first orientation FIXED in shard/topology.py — the old order was undeployable).
  Loop RTT ≈105ms. Both report arms on ONE warm ring, receipts verified per cell.
- **Numbers: chain 8.3 decode-weighted / tree 7.83 / per-cell-best 9.11; interactive novel cells (tree):
  reason-math 10.0, reason-logic 10.1, agentic 11.2, conversation 9.28.** The "10-12" projection is now
  MEASURED. Tree wins every interactive cell; depth-4 pipelined n-gram wins every verbatim/ctx cell (fast
  ring ⇒ pipelining pays) — chain took the aggregate, so **depth-aware hybrid is the #1 code lever** (~+1 aggregate).
- **First transport/compute split (M25_STAGE_TIMING, landed on the branch): transport 55-68% of T_traversal
  (~170-290ms vs ~105ms RTT floor); stage compute ≈138ms NOT ~40ms.** And the compute is
  **CPU-KERNEL-LAUNCH-BOUND, not GPU-bound**: identical 5090s (1525GB/s, ~220TF all five), but idle
  Core-Ultra boxes run 13 layers in 11.5ms while loaded/old EPYC slices take 35-50ms (pyloop 0.09s vs
  0.28-0.47s, one spare at load-average 272). Consequences: (1) box selection must probe single-thread
  CPU+load (ring_up now does, crude factor); (2) **CUDA-graphs are UN-DEAD for scattered rings** — the
  ~1.05× dead-end verdict came from a fast-CPU box; on EPYC slices graphs recover ~2-4× of block time, but
  GraphRunner must learn to emit EAGLE aux (python side-effect — graphs skip it today). Scoped code task.
- **DEPTH-AWARE HYBRID: DONE + MEASURED (same day, arm 3 on the same ring).** Matched n-gram rounds now
  ride plain PIPELINED chain frames (up to --depth in flight; the tree's 1-wide-tree framing paid the
  manual off-flash kernel + trunk re-feed for zero accept gain); novel rounds stay sync tree. Landed as
  `feat(coord): depth-aware hybrid` + adversarial-review fixes (honest mean_accept on pipelined rounds;
  fake-ring KV content model + through-divergence pairing tests; 64 CPU tests). **Measured: raw 7.59
  decode-weighted on a 1.32×-slower ring window (co-tenant jitter, measured at identical g) ≈ 10.0
  normalized — above the 9.11 per-cell-best bound; g novel == tree exactly, g verbatim strictly up
  (rag 2.5→2.7, 8k-quote 2.2→2.8; raw tok/s beat the tree arm on those cells DESPITE the slower ring).**
  Receipt addendum in `docs/receipts/m25-goodring-receipt-20260703.md`.
- **NEXT SESSION, in order:** (1) **CUDA-graph aux compatibility** (+20-30% on slow-CPU boxes, makes the
  ring CPU-agnostic; NOTE from review: hybrid refeed frames have variable size L+K ≤ TREE_DEPTH+1+K → up
  to ~TREE_DEPTH extra graph captures — bound the capture set or bucket sizes). (2) **lean codec /
  thin-TCP** (transport 55-68% measured — up to +20-30%). (3) a CALM-WINDOW interleaved 3-arm pass to
  re-pin the hybrid number without time-of-day jitter (cells alternate arms, one warm ring). Stacked
  honest ceiling on this ring class: **~12-14 interactive**. (4) then TIER-2 trust (receipt
  freshness/binding) + TIER-3 gateway/wire hardening = the betanet-integration path. Review-logged debt:
  tree-path resume_ids handling diverges from chain (out excludes resume; pre-existing), 30k tree rounds
  pay the manual kernel over full ctx (807ms/round — flash tree kernel is the fix).
- **RING IS WARM (7 boxes, ~$3.6/hr): 5-stage ring CZ900(head)/CZ887/CH/NO/DK + spares IT/HU** (GB spare
  destroyed, load-272 dud). iids 43696900,43696887,43696869,43696886,43696878 + 43696880,43696881. Head ssh
  ssh3.vast.ai:16900; drive reports as SOLE coordinator on the head (report_chain/report_tree.log there).
  Same-ring re-measure after the hybrid lands = clean A/B/C.
- Branch `perf/tree-depth-hybrid` (9 commits, pushed): fake-ring harness + bonus/honest-g + usability
  harness + panel docs + **M25_STAGE_TIMING** + **topology head-first fix**. PR NOT yet open (no gh token
  on this box): leyten one-clicks github.com/leyten/shard/compare/master...perf/tree-depth-hybrid.

*(2026-07-02 night panel verdict — the physics this receipt confirms, kept for the record:)*
### PRIOR FOCUS: BREAK THE ~5 tok/s REASONING FLOOR — step back, find the lever or the breakthrough
leyten is not satisfied with ~5 tok/s single-stream reasoning and wants this to be THE focus: take a step
back, think from first principles, decide whether we're missing something buildable or need a genuine
breakthrough. **Do NOT resume by grinding TIER-1 cleanup — resume HERE.**

**The physics (verified against the record, not a guess):** tok/s = g / T_traversal.
- **g ≈ 3.6** committed tokens per ring traversal on novel reasoning, and it is **STABLE across every ring
  ever run** (11.8 tok/s ring @g3.7, serial-path ring @g3.6, tonight @g3.6). g is the drafter's accept —
  ring-independent. Tree-verify lifts it to ~4.5 (measured), a marginal move.
- **T_traversal ≈ 900ms** on tonight's ring = coordinator draft + 6× serial stage MoE-compute + ~12 WAN
  legs + return. tok/s = 3.6/0.9 ≈ 4. On a good ring T_traversal drops and the SAME g gives ~12 (06-30).
- **THE STRUCTURAL WALL:** the DRAFTABLE path (n-gram) PIPELINES depth-4 → 50-80 tok/s; the REASONING path
  (EAGLE) is FORCED SERIAL depth-1 (`m25_pipe.py` `cur_depth = 1 if S.M25_EAGLE`) because EAGLE needs the
  ring's verified hidden from traversal N to draft N+1. So reasoning pays the full 900ms serially every
  ~3.6 tokens. **The floor is SERIAL LATENCY on the reasoning path, not accept.** This is the thing to break.
- **Two levers only:** (1) raise g (tree/better drafter — marginal, we've mostly done it); (2) cut or HIDE
  T_traversal. The big one is #2 — specifically, can the reasoning path PIPELINE like n-gram does? Prime
  candidate: a **standalone small draft model** on the coordinator (drafts autoregressively WITHOUT the
  ring's hidden → pipelinable depth-4; lower accept than EAGLE but pipelined-low may beat serial-high at
  T=900ms). Plus: fewest-fattest stages (6→4/5 cuts hops), is stage compute launch- or compute-bound,
  RTT-ordered ring, staleness-tolerant EAGLE. **Measure-first:** the engine already returns
  decode_s/draft_s/ring_wait_s — a single warm run prints where the 900ms actually goes (transport vs
  stage-compute vs draft), which decides which lever matters. Don't build before that breakdown is read.

**PANEL VERDICT (3 agents: latency-hiding / traversal-time / ceiling-skeptic, 2026-07-02 night) — NO
breakthrough exists for novel-reasoning single-stream; it's accept-gated PHYSICS, proven with a calibrated
sim. The honest ceiling is ~10-12 tok/s on a good tight EU ring, and the path there is EXECUTION, not invention.**

- **The pipelining lever is DEAD (proven, not assumed).** The latency-hiding agent built a Monte-Carlo of
  `coordinate_pipe`'s real flush-on-divergence pipeline, calibrated at BOTH ends (α=0.74 per-token accept →
  g3.6/4.0 tok/s = measured reason-math EXACTLY; α=0.97 → 50-75 = the n-gram ceiling). Result: **pipelining
  depth-D is accept-gated** — a depth-D chunk is only valid if the prior chunk FULLY accepts (α^K); at
  α=0.74, K=8 that's ~8%, so ~92% of traversals flush the pipe and depth buys ~nothing. **Pipelining only
  pays above α≈0.80 — the verbatim regime n-gram already exploits.** No novel-reasoning drafter reaches 0.80
  (EAGLE-3, which PEEKS at the target hidden, tops at 0.74; anything blind is worse). So: standalone draft
  model (α0.5-0.7 → 2.2-3.7 tok/s, LOSES to serial-tree 4.6), staleness-tolerant EAGLE (depth-2 → 4.07,
  loses to tree, and staleness drops accept further), Medusa/MTP (=tree, already built), block-parallel/
  Jacobi (α≈0 on novel text) — ALL dead or dominated by the tree-verify we already run. **Tree-verify
  (raise g at depth-1, zero flush penalty) beats every pipelining scheme at every T_ring for α=0.74.**
- **T_traversal is 98% "blocked on the ring"** (the coordinator draft+commit is 2% — every serial-path
  micro-opt is spent), and ~80-90% of that is TRANSPORT on a good ring. It's **7 legs, not 12** (the tail
  return is already one direct leg via `serve_tail_direct`). Tonight's 900ms was a slow draw; the good-ring
  floor is ~400-500ms (06-30 hit it). The scattered-WAN T floor is ~300-340ms (5 legs × ~28ms RTT + ~40ms
  compute + ~90ms overhead) — uncrossable without co-location (banned).
- **THE EXECUTION PATH to ~10-12 (all depth-1, all scatter-native, most already built/scoped):**
  1. **Topology / RTT-ordered ring** (biggest lever — it's the 2.4× between tonight's 900ms and a good
     ring's ~400ms). `plan_ring`→`select_ring`→`--order`; the false-infeasible fix is already MERGED (PR #13).
     Value is variance-reduction: it STOPS you paying 900ms. Needs the measure-before-pull launch flow.
  2. **Fewest-fattest stages: 6→5** (−1 leg −1 sidecar hop ≈ −55ms, +14%, ~free — just the layer split;
     M2.5's 115GB/32GB fits in 5). 4 is VRAM-infeasible with the EAGLE head on the coord stage.
  3. **Lean codec / thin-TCP** — kill the ~180ms non-RTT overhead (pickle serialize + libp2p sidecar
     loopback+Noise) with a fixed binary frame for [ids,h_fp8,aux_fp8]. +18% (~450→380ms). Medium effort.
  4. **Tree-verify (DONE, g3.7→4.5)** rides on top. Stacked: T~325-400ms × g4.5 → **~11-14 tok/s**.
  5. **Tail-side drafter** (run the EAGLE head on the tail, draft the instant the hidden exists, inject
     tail→head) — minor ~5-15% T shave on rings where the coord is remote; bundle with topology.
- **The ONE receipt that ends the argument (do FIRST next session):** the good-ring tok/s (11.8) and the
  tree +18% have NEVER been measured on the SAME ring — "10-12" is arithmetic, not evidence. ONE
  RTT-ordered, 5-stage, good-EU-ring tree-verify run converts it to a real ~11-14 or a real disappointment.
  Cheap ($5, one warm run). Bundle the T_traversal per-stage-timestamp breakdown (~20 lines) to split
  transport-vs-compute finally.
- **The only ceiling-RAISER left is g** (α toward 0.80): train a better EAGLE head on M2.5 reasoning traces
  (SpecForge, ~$400-2000). Raises g directly AND is the only thing that would ever unlock pipelining as a
  bonus (α≥0.80 flips the whole table). But EAGLE-3's authors already sit at ~0.74 on hard reasoning — +0.06
  absolute is a RESEARCH bet, not an engineering certainty. Queue it, don't bank on it.
- **⇒ STRATEGIC FORK for leyten (genuine, surface it):** single-stream novel-reasoning is PHYSICS-capped at
  ~10-12 on scatter — that's the honest tolerable-demo number, at/ahead of the field (Petals ~1 tok/s true-
  global @176B; nobody does usable single-stream 100B+ over WAN). The engine's actual WINS are elsewhere:
  **batched 155 tok/s aggregate** ($/token, latency-tolerant — under-marketed) and the **draftable/agentic
  path** (50-80 tok/s). Also note: the 5-hop serial chain is FORCED by M2.5's 115GB not fitting fewer cards
  — a RIGHT-SIZED reasoning model (~30-70B, 1-2 hops) would be genuinely fast single-stream on the same
  fabric (the north-star "many models" angle). OPTIONS: (a) execute to ~11-12, declare it the tolerable
  demo number, re-point the perf narrative at batched+agentic; (b) also serve a smaller model for the
  interactive/reasoning tier; (c) spend on the EAGLE-head research bet. Full panel outputs archived in
  `.claude/plans/` if needed.
_(NOTE: co-location/datacenter/NVLink is BANNED as a "solution" — [[never-colocate-usable-speed-on-scattered]].
The lever must be scattered-native. Reframe option on the table per the skeptic: is single-stream reasoning
even the right target vs batched 155 tok/s aggregate — but leyten's call is that usable single-stream matters.)_

---
*(prior TIER-1 session, banked — still valid, just no longer the focus:)*
**LATEST (2026-07-02 evening) — TIER-1 session: wedge fix + CRITICAL trust fix + tree-verify v2, ALL
warm-validated on a fresh 6×5090 EU ring (HU/HU/DK/CZ/BG/CZ). Two branches ready to land, in order:**
1. **`fix/ring-wedge-receipt-truth` (4 commits)** — (a) RING WEDGE FIXED: specpipe's churn recovery ported
   into m25_pipe (forward-link rebuild, independent tail pred/ret lifecycles, stale-job drop) + hardened
   after 2 adversarial review passes (speaking-pred adoption, TCP keepalive 60/20/3, guarded ret_ok,
   fresh-ret keep on the gateway-retry race, transport.py malformed-frame guard). WARM-PROVEN: coord
   kill -9 mid-decode (depth=4 in flight) → NEW coordinator on the same ring, NO re-warm (receipt
   `m25-ring-wedge-smoke-20260702.json`). The re-warm-per-coordinator tax is gone. (b) CRITICAL receipt
   fix: coverage verified against the model's TRUE depth (62), fail-closed on empty receipts — the
   skip-layers-and-still-get-paid hole is shut (`tests/test_receipt_coverage.py`; specpipe: `--n-layers`).
2. **`eagle/tree-verify-v2` (4 commits, stacked on 1)** — tree-verify REBUILT on the merged base with the
   fleet's payload fixes: top-M best-first tree (M25_TREE_M=12/TOPB=3/DEPTH=8 — kills the 62-node 2^d
   shape), fp8 tree traffic (_hsend + fp8 aux), manual broadcast-GQA tree kernel (dense mask is off-flash
   on sm_120), receipts attested through the tree path, hybrid n-gram routing kept (matched rounds verify
   as a 1-wide tree + bank the bonus token). **WARM A/B (receipt `m25-tree-verify-v2-ab-20260702.json`):
   tree WINS the tight EU ring +18% decode-weighted (3.9→4.6); reason-math 4.8→6.0, reason-logic 3.0→4.7,
   code-edit 4.5→6.1 — v1's tok/s loss WAS wire payload, as the fleet concluded.** Losslessness gate: 76
   identical tokens then one near-tie kernel flip (manual-vs-SDPA numerics; documented class, same as fp8
   wire). Known gap: rag-quote 5.6→4.2 (depth-4 pipelined n-gram beats the sync depth-1 tree round on
   verbatim) → depth-aware hybrid = the measured next lever. 33 CPU tests (8 new tree tests incl.
   `propose_tree(topb=1) == propose()` exactly).
- **NEXT:** (1) leyten merges the two PRs in order (`docs/roadmap-fleet-findings` is superseded — its
  commit is cherry-picked here; delete that branch). (2) **Depth-aware hybrid** (pipeline n-gram rounds at
  depth, keep tree rounds sync — recovers rag-quote 5.6→4.2, the one cell tree loses; code change, CPU-
  testable). (3) **Topology-ordered launch** (wire `plan_ring` sidecar-RTT into provisioning so ring order
  stops being a lottery draw; fix the select_ring false-infeasible first, TIER 3) → then ONE over-rented
  RTT-ordered warm run = the ABSOLUTE 10–12 check (tonight's ring was ~2.4× slower per traversal than the
  2026-06-30 good ring at identical g — the relative +18% is banked, the absolute target needs a good
  ring); min-match within-run A/B rides along. (4) TIER-2 receipt freshness/binding, TIER-3 gateway/wire.
  Backlog: `.claude/plans/fleet-findings-20260702.md`. NOTE: the high-RTT global-scatter cell is DROPPED
  (decision below) — do not resurrect it.

*(2026-07-02 morning — serial-path A/B, MERGED as PR #10:)* master 4.3 → branch 5.7 = **+33%
decode-weighted** (jitter-robust, both orderings) + rag-quote accept **13→44%** (whole-prompt drafter
context). Receipt `docs/receipts/m25-eagle-serial-path-ab-20260702.json`. min-match still unproven.

*(pre-A/B, kept for context — the branch build:)*
**Branch `perf/eagle-serial-path` (worktree `/root/.openclaw/workspace/shard-perf`), tested (18 CPU tests pass).**
- What the branch fixes (all found by reading + a 12-reviewer adversarial fleet; leyten directed: ENGINE PERF focus):
  (1) `EagleDrafter` was O(ctx) per draft round (list-KV re-cat + GQA repeat_interleave every propose; ~8 tiny
  kernels/token in extend) → preallocated in-place KV + batched extend + broadcast-GQA; CPU bench 156×
  prefill-extend / 3.8× decode round; proposals regression-locked to the old impl (`tests/test_eagle_draft.py`).
  (2) EAGLE aux payload (3×[K+1,H] bf16 ≈ 166KB/hop ≈ 3× the h payload) now fp8-packs (`M25_FP8_AUX`, defaults to
  M25_FP8_WIRE; drafter-only → losslessness untouched). (3) The drafter saw only the LAST prefill chunk (512-token
  context window!) → every chunk now extends the EAGLE context as it arrives (accept ↑ on long prompts, unmeasured).
  (4) Divergences no longer compute-then-discard a full stale draft (`cancel()`). (5) n-gram `matched` needed zero
  context agreement → coincidence anchors starved EAGLE on novel text; now `best_len>=1` routes (M25_NGRAM_MINMATCH).
  (6) K=8 defaults landed (coord+gateway were still 6). (7) fp8 dtypes added to `wire.py` (raw-TCP path rejected
  every M25_FP8_WIRE frame — codec drift vs transport.py). (8) M25_CUDA_GRAPH+M25_EAGLE now fails loud (stale-aux
  poison). (9) `coordinate_pipe` returns `decode_s/draft_s/ring_wait_s` — the warm run finally attributes the
  ~180ms/traversal that isn't RTT.
- **Review fleet (12 reviewers + adversarial verifiers, run wf_6818d2f6-5cf) — verification still completing;**
  headline verified-or-strong findings BEYOND this branch, ranked for perf: (a) tree-verify's measured tok/s loss is
  largely SELF-INFLICTED (~6-7× wire bytes/traversal: trunk re-feed + un-fp8'd aux + dense-mask-off-flash attn +
  worst-case 2^d fan-out shape) → fix payload+shape+mask-split BEFORE the high-RTT measure, it may flip the tight-ring
  verdict too; (b) ring-wedge root cause CONFIRMED in code (stages dial `nxt_sock` once, tail closes pred on coord
  death → cascade, nobody re-dials) — the re-warm tax is a fixable bug; (c) batched-decode KV write has NO MAXLEN
  guard (OOB scatter CUDA-assert kills the stage); (d) receipt coverage check is self-referential (layer_count from
  the receipts themselves — pass n_layers explicitly), receipts have no freshness/chain-link binding, and
  `transport.py` (production path!) lost wire.py's malformed-frame hardening (one bad frame kills a stage — betanet
  blocker, not perf); (e) `m25_scatter_pipe` forwards M25_* env to stages but NOT coord/gateway (measurement-poison
  trap); (f) STATE.md/FLEET_STATE.md/RESUME_B.md are dead-stale (history agent) — cull or supersede.
- Next actions (ranked): (1) land `perf/eagle-serial-path`; (2) warm EAGLE run: read the breakdown, A/B branch vs
  master, A/B M25_FP8_AUX + MINMATCH (accept must not regress); (3) tree-verify payload/shape/mask fixes on a rebased
  branch, THEN the high-RTT measure; (4) wedge fix (nxt_sock re-dial + tail keeps pred on ret death); (5) batched
  MAXLEN guard + scatter-launcher env forwarding; (6) the (d) soundness cluster when back on trust work.

*(previous session, kept for context:)*
**2026-07-01 — all on `master`; `select_ring` is now UPLOAD-AWARE. NEXT = the selection-driven warm run.**
Tonight landed on master: handshake fix + `select_ring` + EAGLE-chain (PRs #7/#8/#9) + **fp8 wire** (cherry-pick
c4588bf) + **upload-bandwidth-aware `select_ring` + role relegation** (this session). Branches deleted; only
`eagle/tree-verify` remains unmerged. PoC = **the BETANET** (M2.5 engine integrated INTO c0mpute, permissionless)
— NOT a standalone fast ring (don't relabel it as just "usable speed").

- **`select_ring` UPLOAD-AWARE (this session, on master).** The #1 residential lever landed. Objective is now
  TOTAL REQUEST TIME `T = prefill_ms + D*decode_step_ms` with per-node UPLOAD a first-class cost (sender-uplink
  bound; the residential bind). Prefill's [S,H] activation (~100MB/hop @16k) is the wall; the selector tails the
  lowest-upload node (the tail forwards nothing), drops nodes whose uplink would dominate prefill, and RELEGATES
  them to off-critical roles (weight-seeder / aggregator-relay / hot-standby / decode-only-replica / spot-check-
  verifier) instead of discarding capacity. Prefill transport modeled as the engine's chunked+pipelined makespan
  `(sum_fwd(u)+(C-1)*max_fwd(u))/C` (C=1 SUM ↔ C large MAX). PURE, and BYTE-IDENTICAL to the old decode-only path
  when `up_mbps` is omitted (golden-snapshot regression-tested). VALIDATED offline (`scratchpad/sim_network.py`,
  volunteer/residential pool): aware/oracle ~0.98 across ctx while blind/oracle collapses 0.98→0.80 as ctx grows;
  request-time speedup 1.01×@2k → 1.09×@16k → 1.32×@64k; **TTFT (first-token) speedup 2.5–5× (p95 up to 19×)**;
  the rental/fat-uplink pool shows a smaller gap (sanity). Adversarial review (2 attackers found nothing; 1 found
  + I reproduced/fixed a pre-existing funnel false-infeasible: subnet-blind `must`-set). Tests: `tests/
  test_topology.py` (10, all pass). Commit c2e226e. c0mpute self-optimizer feeds it measured up_mbps; it stays pure.
- **WARM A/B (2026-07-01): attempted on 8 real scattered EU boxes; premise CONFIRMED, full automation infra-blocked.**
  Rented 8 subnet-distinct EU boxes (CZ/HR/PL/NO×2/BG/CZ/HU, echo-only, no model — the [S,H] TRANSPORT is the term
  under test). MEASURED real bandwidth heterogeneity across ring hops from one box: **8, 16, 39, 40, 50, 61, 127 Mbps**
  — i.e. real scattered rings DO have residential-tier slow hops (8–16 Mbps) that wall prefill (a 100MB @16k activation
  over an 8 Mbps hop ≈ 100s vs ~6s over 127). That confirms the premise. BUT the fully-automated per-node-UPLOAD
  aware-vs-blind A/B did not complete, blocked by vast-container infra: (1) **no NET_ADMIN** → `tc` egress-shaping
  unavailable (switched to app-layer send-pacing); (2) **NAT hairpin** (a box can't reach its own public IP → self must
  be excluded from probes); (3) an 8s socket timeout killed >8s uploads (fixed → settimeout 300); (4) detached echo
  servers didn't persist + (5) **vast ssh-proxy RATE-LIMITED** my repeated debug runs → all probes failed. Tore down
  cleanly (0 live, ~$4). PATH TO A CLEAN NUMBER (cheap, no throttle needed — natural EU uplinks are already 8–127 Mbps):
  ONE GENTLE run — sequential per-box, verified servers, spaced SSH, no retries-in-a-burst — after the proxy cools;
  tools staged in `scratchpad/measure_uplinks.py`. The engine change itself is offline-validated + reviewed + landed.

- **Handshake deadlock FIXED** (`_tail_accept`): acks the coord-return the instant it's identified instead of
  waiting for the lazily-connecting predecessor. Validated on a real decoded row. Covers coord + gateway.
- **The "junk ring 2.6 tok/s" was NO node selection** (rental-lottery boot order: Spain/Norway + a 400W box).
  Drafter reproduced exactly (reason-math 34%/g3.7) → engine fine. We're AHEAD of Petals (≈5-6 tok/s @70B; us
  ~12 @230B on a good ring; their geo-distributed ~2× WAN penalty matches ours).
- **`shard/topology.select_ring`** = the self-optimizer's pure core (subset+order+layer-split minimizing predicted
  decode step-time; drops weak/co-located; fewest-fattest; `require` pins the coord/head). Reviewed (2 critical
  false-infeasible bugs fixed), regression-tested, calibrated. `scratchpad/plan_ring.py` = vast glue (measure→
  select→--order); `scratchpad/sim_network.py` = offline simulator ($0 dev loop, reproduces tonight's rings).
- **fp8 activations on the wire (`M25_FP8_WIRE`)** — halves bytes/hop. MEASURED A/B (5-EU ring): bf16 4.87 → fp8
  5.30 = **+9% on vast** (high-bw → per-hop is RTT-windowing-bound, not bytes; fp8's ~2× is the RESIDENTIAL/
  bytes-bound regime). QUALITY: fp8 keeps M2.5 correct+coherent (same primes, sound reasoning) but NOT bit-exact
  (flips a token → greedy diverges). So fp8 = usable-M2.5 quality, NOT lossless. Per-channel scale = the
  tightening lever if a precision-sweep shows loss.

- **⚠ RESIDENTIAL BOTTLENECK (3-agent research) — the bind is the SENDER's UPLOAD.** Asymmetric residential (fast
  down, slow up) strands the downlink; the ring runs at its slowest uplink. DECODE survives (~3-5 tok/s @20Mbps,
  →8-12 w/ fp8+fiber); **long-context PREFILL is the wall** (100MB+/hop → ~3-6min TTFT @16k, ~20min @100k on
  20Mbps cable). NOT monolithic: FIBER (sym 100M-1G, ~40% US homes) → bottleneck VANISHES; the killer is the slow
  CABLE/DSL UPSTREAM specifically. You CANNOT conjure upload on a too-small pipe (QoS/FEC/transport-multipath all
  spend upload or need a 2nd physical link — can't beat line rate). The torrent move that WORKS = use the DOWNLOAD
  direction: fan-in (split the activation across W senders, receiver aggregates W uplinks → ~W× eff up) + a
  relay/supernode tier for heavy prefill.
  RANKED LEVERS: (1) **upload as a first-class (prefill-DOMINANT) cost in `select_ring`** + relegate low-uplink
  nodes to off-critical roles (spot-check verifier / hot-standby / weight-seeder / decode-only replica) — biggest,
  free, scatter-pure; (2) fewer/fatter hops (−40-60% prefill upload); (3) fp8 done → int4+compression next (drafts
  free under lossless verify, prefill measured-lossy, codec-in-manifest for receipts); (4) BBR + persistent
  connections (CUBIC collapses ~70% @1% loss; BBR shrugs it); (5) chunked-prefill overlap + route long-ctx to the
  fiber subset; (6) relay/supernode tier = the ONE THESIS-RISK lever (curated-transport crutch unless
  permissionless+staked).
- **ADMISSION vs PLACEMENT (decided framing):** do NOT gate joins with a single hard threshold — it discards nodes
  useful in off-critical roles and shrinks the permissionless pool. **Admission** = a coarse PROVEN floor (real
  GPU, reachable, can carry *some* role) in c0mpute; **Placement** = capability-matched roles in the self-optimizer
  (the "threshold" is PER-ROLE inside `select_ring`, not a velvet rope at the door). Both on MEASURED/VERIFIED
  capability, never self-reported (lying-uplink attack → caught by probing + the receipt hash-chain).

- **NEXT ACTIONS (ranked):** (a) ~~upload-aware `select_ring` + relegation~~ **DONE** (this session; offline-validated,
  tested, on master). (b) **selection-driven warm run** (over-rent ~8, `plan_ring` measures→selects→`--order`, warm,
  benchmark predicted-vs-actual request_ms; also wire per-node upload into `plan_ring` — it currently measures RTT/
  VRAM/power but NOT uplink, so add an upload probe before this run); (c) residential-bw A/B (tc-throttle a ring to
  20Mbps, measure decode+prefill bf16-vs-fp8 — boxes torn down, re-provision); (d) self-optimizer graduates to
  c0mpute (shard=engine, c0mpute→shard only; roles become placement hints the network layer acts on). Roadmap:
  Vivaldi coords = O(N) all-pairs latency at scale; tree-verify (`eagle/tree-verify`) = engine lever for high-RTT.

---
*(historical — the EAGLE hybrid work that reached ~12 tok/s on a good ring:)*
**Goal:** make M2.5 usable on NORMAL reasoning-ON usage (currently ~3 tok/s single-stream — see PROVEN).
**Approach (approved plan `.claude/plans/graceful-greeting-seahorse.md`):** a HybridDrafter = n-gram for
draftable output ⊕ **EAGLE-3** for novel reasoning, run coordinator-side (aux hidden states ride the verify
return — no extra round-trip). Lossless (ring greedy-verifies).

**GO signal is already IN (no vLLM re-measure needed):** thoughtworks published EAGLE-3-on-M2.5 = 2.11×
HumanEval / 1.78× MT-bench (≈ ~2.5 reasoning accept) — the head's own authors confirmed it works. So **GO** on
building the integration; the *real* accept number now comes from OUR engine.

**RESULT (2026-06-30): EAGLE-3 WORKS — reasoning lifted off the ~1% floor.** The real bug (a 4-agent panel
found it; the off-by-one layer hypothesis earlier this session was a red herring): the EAGLE-3 draft head is a
TRANSFORMER that attends causally over the WHOLE committed sequence (each position carries the target aux
feature), but our port ran `propose()` from an EMPTY KV cache every call → no context → it ignored the aux and
degenerated to token-repetition (~1% accept). **FIXED:** `EagleDrafter` keeps a persistent committed-context KV
cache (`reset`/`extend`/`propose`); `coordinate_pipe` feeds per-position committed aux via `extend()` each
commit (the ring already returned aux for every chunk position — we were keeping only the last). Validated on a
5-EU scattered ring (branch `eagle/chain-diagnostics`, commits 0dc939a + 76ab7e2):
reason-math **8.0 tok/s / 30% / g3.4**, reason-logic 6.4/14%, open-chat 5.9/11%, code-edit 6.9/11%,
rag-quote 7.6/15%, agentic-tool **15.2/50%/g5.0**; **decode-weighted mean 7.0 tok/s** (was 0.9 broken / ~3
n-gram baseline). The panel: reference-diff caught the missing context attention; SpecForge killed the
"standardize aux" idea + confirmed raw-aux→fc and layers {1,30,58}; code-audit forced the decisive
`fc(aux)`-varies test; out-of-box mapped the space. Receipt `docs/receipts/m25-eagle-onengine-20260629.md`.

**vLLM PIN:** newer vLLM (0.24.0) broke the NVFP4 MoE load (`quant_method`→`_quant_method`, then
`w13_weight_scale_2`). `swarm_up` bootstrap now pins `vllm==0.23.0` (m25_stage also getattr-shims the rename).

**NEXT ACTION = chase the remaining accept upside (the ring is WARM — KEEP it, see memory keep-rings-warm):**
1. **Layer A/B: DONE** — {1,30,58} (SpecForge) beats {0,29,57} (reason-math 34% vs 30%); reverted to capture
   `L.li` so the default `M25_EAGLE_AUX=1,30,58` maps to those layers (commit 1289088).
2. **Full-accept bonus token (minor):** `coordinate_pipe` n==K branch drops the verified `r[K]` — committing it
   is a free token (the EAGLE pairing is now correct via `extend()`, so this is efficiency, not correctness).
   Small on reasoning (few full-accept rounds); more on agentic. Low priority.
3. **Tree-verify (roadmap #2 — the BIG lever):** GPU idle during the WAN round-trip → verify a TREE of
   candidates per traversal → ~2× accept (2.5→4–5). Needs a tree-attention mask threaded through every stage +
   coordinator best-path selection. The natural next build now that single-chain EAGLE works.
Then land the branch (PR → squash-merge), update PROVEN. ⚠️ Before any warm: verify every box's `/tmp/sidecar`
size == local ref (a truncated one crashed the launcher once).

**MEASURE on a scattered ring, DEBUG on a single box (don't conflate):** EAGLE's payoff is that its draft
COMPUTE is FREE — hidden by the WAN round-trip idle (KEY DECISIONS). A colocated box has no WAN idle, so EAGLE
adds SERIAL per-token latency → tok/s reads flat/worse even at good accept = the WRONG regime to *measure* the
product (also the datacenter pattern the north star rejects, `c0mpute-scattered-not-colocated`). BUT accept
LENGTH and any integration bug are network-independent, so DEBUGGING is correctly + cheaply done on one box.

**DEAD END found (don't repeat): vLLM M2.5 under TP requires GPU P2P** — `MiniMaxText01RMSNormTP` uses a
Lamport/IPC all-reduce → `cudaErrorPeerAccessUnsupported (217)` on consumer-5090 hosts w/o NVLink + ACS-blocked
PCIe (most vast boxes). `NCCL_P2P_DISABLE`/`VLLM_DISABLE_CUSTOM_ALL_REDUCE` DON'T fix it (separate path). So
can't GO/NO-GO via vLLM TP on typical vast hosts. Our PIPELINE engine avoids it (point-to-point sockets). If
vLLM-on-M2.5 is ever needed, the host must support P2P (NVLink box, or ACS-disabled — unverifiable pre-rent).

**OPS this session (EAGLE on-engine run):** (1) **`swarm_up` had no continent filter** — only excluded Asia +
deduped region → grabbed 2 cheap Canada boxes into a 4-EU ring (transatlantic, ~80-100ms hops). FIXED: added a
`EUROPE` allowlist (`scratchpad/swarm_up.py`); for the live ring, `scratchpad/swarm_add.py` surgically swapped
the 2 NA boxes for EU (rent+verify replacements BEFORE destroying). Always verify `instances-v1` count after.
(2) **Zombie box:** `swarm_add`'s `create()` returned None on a transient timeout but vast HAD made the box →
untracked, billing. Caught by the post-swap instance-count check. Always count instances after any rent.
(3) **Truncated sidecar:** one box's `/tmp/sidecar` was 7.8MB not 29MB (bootstrap scp left a wrong/partial
binary) → `peerid()` got no PEERID, launcher crashed. Verify `stat -c%s /tmp/sidecar` == local ref on all boxes
before warm. (4) **Ring wedges after each coordinator** → re-warm before every new coordinator process.

---

## North star → current goal
- **North star:** torrent-for-compute — permissionless scattered GPUs serving big models, trustless. M2.5 = PoC.
- **Current goal:** a sharded M2.5 engine that is *usable + viable*. NOT one metric — the whole product.
- **tok/s MEASURED (normal reasoning-ON, single-stream, scattered ring):** post-graph-aux (PR #25) **~24
  decode-weighted on a good EU ring; reasoning-heavy cells ~30–32** (reason-math 32, agentic 31; +74% over the
  13.6 no-graph chain, one clean 4-rep rotated rep — a full rep2 is outstanding polish). Graph-aux cut stage
  compute **157→40 ms/traversal** on slow-CPU boxes, which lowered T_traversal and RAISED the old WAN-bound
  ~12–20 estimate (that cap assumed ~138 ms stage compute; the lever helps MOST on loaded/old boxes = the
  permissionless steady state). **Pre-graph-aux** this projected to ~10–12 on a good tight EU ring / ~5–6 on
  high-RTT global scatter (DROPPED as a target). NOTE: most of the 79 fleet findings are NOT tok/s
  (trust/gateway/wire); on the perf path graph-aux is the landed lever and tree-verify/topology remain. This
  number is ON the scattered ring — NOT via co-location ([[never-colocate-usable-speed-on-scattered]]).
- **TWO-TIER framing (decided):** **scattered ring = cheap/permissionless/THROUGHPUT** (latency-tolerant); a
  **co-located/regional node or mini-cluster = fast/INTERACTIVE** (M2.5-NVFP4 ~115 GB fits on 1× H200 / 2× H200 /
  4× RTX6000-Blackwell → no WAN → 30–50 tok/s, physics-guaranteed). WAN-sharded single-stream is the *hardest*
  way to serve M2.5; use the right tier per workload. The engine serves the whole spectrum.

## PROVEN  (numbers + receipts — measured, honest)
| capability | status / number | source |
|---|---|---|
| **CUDA-graph EAGLE-aux (slow-CPU rings)** | **stage compute 157→40ms/traversal (~4×, drift-proof); decode-weighted chain 13.6→23.7 = +74%** (4-rep rotated EU ring); reason-math 18→32, agentic 14→31. Kernel-launch overhead removed on slow-CPU boxes | **master** (PR #25), receipt scratchpad rep1/confirm json, GPU-check research/graph_aux_check.py |
| Batched throughput | **155 tok/s agg @16k (2.60× single), coherent** (B=4, batched-MoE, fp8 KV) | commit f3894d6, m25-batched-serving-fixed |
| Single-stream DRAFTABLE (copy/RAG/verbatim) | 50–81 tok/s (n-gram, accept high) | m25_ctx_table |
| **Single-stream NORMAL reasoning-ON (EAGLE hybrid)** | **~5.7 tok/s decode-wtd on a jittery lottery ring / ~7 on a good tight EU ring** (2026-07-02 warm A/B, merged serial-path; was ~3 n-gram-only, ~1.8 raw) | receipt m25-eagle-serial-path-ab-20260702 |
| **TREE-verify v2 (hybrid, tight EU ring)** | **+18% decode-wtd over chain on the SAME warm ring (3.9→4.6); reason-math 4.8→6.0, reason-logic 3.0→4.7, code-edit 4.5→6.1; g novel 3.7→4.5 at M=12** — flips v1's 'tree loses tok/s on tight rings' (payload, not physics). rag-quote gap = sync tree vs pipelined n-gram | receipt m25-tree-verify-v2-ab-20260702, branch eagle/tree-verify-v2 |
| **Ring churn survival (wedge fix + heartbeat)** | coord kill -9 mid-decode → new coordinator, same ring, NO re-warm — **WARM-VALIDATED LIVE 2026-07-07** on a 5-stage EU ring (coord B 66 tok completed after A killed mid-decode). + F6 per-reply decode heartbeat (blip failover in seconds) | receipts m25-ring-wedge-smoke-20260702 + **m25-warmring-validation-20260707**, PRs #26/#34 |
| Tools / multi-turn / long-ctx(≥30k needle) | PASS | _validate pass, prior receipts |
| Trustless verification (moat) | signed per-stage receipts, lossless, coverage vs TRUE depth + fail-closed + **per-job nonce (anti-replay) + `out_root==in_root` chain binding (#36)** — **WARM-VALIDATED LIVE 2026-07-07**: 5-stage chain held exactly across scattered EU stages, PROVE ALL valid. TIER 2.2 CLOSED (endpoint bindings = follow-up) | shard/receipt.py, tests/test_receipt_binding.py, receipt m25-warmring-validation-20260707 |
| Reasoning control (no-think fast mode) | wired (`reasoning` flag, render_ids closes `<think>`) | commit da9f11d |
| **EAGLE hybrid drafter (reasoning)** | **WORKS: reason-math 34%/g3.7/11.8tok/s, open-chat 13%, agentic 50%/g5.0; ~7 tok/s decode-weighted** (was 0.9 broken). Bug was missing context attention (persistent context KV); aux layers {1,30,58} | **merged to master** (PR #7) |
| **Self-optimizer core (`select_ring`)** | UPLOAD-AWARE: minimizes total request time (prefill+D·decode) with per-node uplink first-class; tails/drops slow-upload nodes + relegates them to off-critical roles; picks subset+order+layer-split; adversarially reviewed (3 false-infeasible bugs fixed total), 10 regression tests, byte-identical legacy path | **master** (`shard/topology.py`, `tests/test_topology.py`) |
| **Upload-aware selection (offline validation)** | aware/oracle ~0.98 vs blind 0.98→0.80 as ctx grows; **TTFT speedup 2.5–5× (p95 19×)** on the residential pool; request 1.0→1.32× (2k→64k); rental gap smaller (sanity) | `scratchpad/sim_network.py`, this doc RESUME HERE |
| **fp8 activations on the wire** | **+9% on high-bw vast** (bf16 4.87→fp8 5.30; ~2× is the residential bytes-bound regime); quality preserved (correct+coherent) but NOT bit-exact | **master** (`M25_FP8_WIRE`, commit c4588bf) |
| **Residential bottleneck (3-agent research)** | bind = sender UPLOAD; decode survives, long-ctx PREFILL is the wall on cable/DSL (fine on fiber); fix = upload-aware selection + use download direction, NOT QoS | RESUME HERE, this doc |

**Root cause of slow reasoning (structural, not a bug):** tok/s = g(committed/traversal) × traversal_rate(≈1/round-trip).
n-gram gives g≈9 on verbatim-reuse but **g≈1 on novel reasoning** (nothing to copy) → bare WAN floor. Fix = a
learned drafter (EAGLE) that predicts novel text. Physics cap: even perfect drafter ~12–20 tok/s on a tight
ring, ~3 on global scatter (NO project — Petals/Parallax/etc — does usable single-stream on 100B+ over global WAN).

## IN-FLIGHT
- **EAGLE hybrid drafter** (`phase0/eagle_draft.py`): `EagleDrafter` (ports thoughtworks/MiniMax-M2.5-Eagle3,
  a LlamaForCausalLMEagle3: fc fuses aux layers [1,30,58] → 1 Llama layer → 32k draft-vocab → d2t→target).
  `HybridDrafter` = n-gram-first → EAGLE-on-miss. CHAIN version built + CPU-smoke-validated + committed (11dc4ee).
  Ring plumbing wired (opt-in `M25_EAGLE`): aux capture in `m25_stage.run_block`, threaded forward + returned by
  the tail (`_merge_aux`), coordinator seeds via `_eagle_seed` + runs depth=1. Coordinator construction wired via
  `make_drafter()` (one source for coord/gateway/bench). Ran on a real all-EU ring 2026-06-29 → accept ~0–3%;
  **root cause found OFFLINE = aux LAYER off-by-one** (the head's `[1,30,58]` are vLLM aux-list indices, embed=0,
  so = post-layer-{0,29,57}; we captured by raw layer index = post-layer-{1,30,58}). **FIXED** in `m25_stage`
  (capture keyed `L.li+1`); codec/wire/structure/fc-norm ruled out offline. **✓ CONFIRMED + MERGED to master** —
  the real fix was context attention (persistent KV), reason-math 34%/g3.7/11.8 tok/s. No longer in-flight.

## ROADMAP (findings-backed, 2026-07-02) — do in tier order

> Grounded in the **2026-07-02 review fleet** (12 subsystem reviewers + adversarial verify → **79 CONFIRMED /
> 5 refuted**; full per-finding detail incl. evidence + fix in `.claude/plans/fleet-findings-20260702.md`
> [+ `-full.json`]). The merged serial-path PR (#10) already closed ~10 of them (the EAGLE/aux/fp8/env/K8/
> cuda-graph cluster). What remains is tiered below. KEY SIGNAL: after the merge, **only 1 of the remaining
> HIGHs is perf** — the high-severity risk has moved OFF the single-stream perf hot path and onto TRUST (the
> moat) and GATEWAY/WIRE robustness. We are near the single-stream perf ceiling (tree-verify aside).

**TIER 0 — DONE (PR #10, merged):** serial-path recovery — drafter O(ctx)→O(1), aux fp8+whole-prompt context,
cancel(), n-gram min-match routing, K=8 defaults, wire fp8 dtype, CUDA_GRAPH+EAGLE guard, launcher env/scp/REPO.
Warm-validated **+33% decode-weighted + rag-quote accept 13→44%** (receipt m25-eagle-serial-path-ab-20260702).

**TIER 1 — PERF / tok/s (the only remaining speed levers; everything else is correctness):**
1. **✅ DONE (branch fix/ring-wedge-receipt-truth, warm-proven 2026-07-02) — Ring-wedge fix** (`pipe`/`launcher`/`critpath`, 3 reviewers). `nxt_sock` dialed once, never re-dialed; tail
   closes `pred` on coord death → cascade → the re-warm tax. Fix = re-dial `nxt_sock` on send fail + tail keeps
   draining `pred` when only `ret` dies. NOT tok/s but the iteration-velocity multiplier + churn-survival. Own
   branch, own warm smoke-test. **Do first** (makes every later measure cheaper).
2. **✅ BUILT + EU-MEASURED (branch eagle/tree-verify-v2, +18% decode-wtd 2026-07-02; high-RTT cell still open) — EAGLE TREE-verify** — the accept lever. Rebase `eagle/tree-verify`
   (worktree `shard-treemeasure`) on merged master (inherits fp8-aux + O(1) drafter → shrinks its payload wall),
   then the 3 fleet fixes: **fp8 the tree aux; split prefix-attn from the N×N tree block to stay on flash (not the
   dense-mask fallback); right-size the fan-out vs the fixed 2^d**. Fleet verdict: the measured tok/s LOSS was
   ~6–7× SELF-INFLICTED wire payload, NOT physics, and the tree math is correct. Measure tight-EU (does payload
   fix flip it?) then high-RTT scatter (its natural regime). See [[m25-tree-verify-measured-state]].
3. **Topology-optimized ring order** — `plan_ring` sidecar-RTT measure → `select_ring` → `--order` BEFORE the pull
   (order is baked at pull time). This session's rental-lottery order split the 2 co-located CZ boxes with GB
   between them; a measured order recovers that. (⚠ fix the `select_ring` false-infeasible bug first, TIER 3.)
4. **Cheap:** min-match within-run A/B (still unproven, one jittery pass); full-accept bonus token (coordinate_pipe
   drops verified r[K] on n==K — free token, small on reasoning); stream `<think>` live (UX, not tok/s).

**TIER 2 — TRUST / the moat (correctness debt; 1 CRITICAL + 4 high, flagged by 3 reviewers):**
1. **✅ DONE (branch fix/ring-wedge-receipt-truth) — CRITICAL — receipt coverage is self-referential** (`receipt.verify_coverage`, `pipe._verify_receipts`).
   `layer_count` is derived FROM the receipts being checked, so a ring that OMITS layers still "tiles fully" and
   passes → a node can skip its block and still be paid. ~10-line fix (pass the model's true `n_layers`
   explicitly). **Do alongside the wedge branch — a skip-compute-and-get-paid hole shouldn't sit open even in a
   perf sprint.**
2. **✅ DONE (branch fix/receipt-freshness-binding, 13 tests) — freshness + chain binding.** Coordinator issues a
   per-JOB random nonce on the reset frame; every stage signs it into its receipt; `verify_coverage(expected_nonce=)`
   rejects a set whose nonce isn't this job's → a replayed old receipt (stale nonce) fails closed. Plus CHAIN binding:
   `verify_coverage(check_chain=)` asserts each block's `out_root == next block's in_root` (an attested output must be
   what the next node attests it received) — catches fabricated/spliced roots, holds by construction on the lossless
   wire (gated `not M25_FP8_WIRE`, since fp8 transport is intentionally lossy). Coordinator-trusted-challenge threat
   model (leyten's call). SCOPE: chain binds interior edges; the head's input (↔ prompt embedding) and the tail's
   final output (↔ coordinator's observed reply tokens) are endpoint bindings, noted as follow-ups. The deeper
   activation proof-of-compute stays the crypto-later seam.
3. **✅ DONE (earlier) — tree-verify path emits receipts** — the `M25_TREE` blocks now call `signer.observe` on both
   the tail and head/middle stages (verification no longer silently off under trees).
4. Verified-fetch trust root (`shard/fetch.py`) — the verification primitive is now HARDENED + TESTED (14
   adversarial tests, `tests/test_fetch.py`: tampered/size/CID reject+delete, path-traversal refused,
   bad-sig/wrong-pin/unsigned manifest rejected, cache re-hash). REMAINING: the real M2.5 deploy still
   bypasses it (`ring_up` uses raw `snapshot_download`) — route the betanet weight pull through `fetch_block`,
   which needs a signed M2.5 manifest (an offline shard-hashing job) + swapping the deploy pull. Bigger build.

**TIER 3 — ROBUSTNESS (gateway + wire + contained bugs; a batch-into-one-session hardening pass):**
- **✅ Gateway (DONE, PR #37, 11 tests):** client-disconnect no longer re-runs the whole generation (client write
  failures raise `ClientGone`, a non-OSError coordinate_pipe lets propagate → abort, never retry); a stalled client
  is bounded by a stream write timeout (`M25_STREAM_WRITE_TIMEOUT`, default 30s) instead of pinning the ring ~30min;
  `reasoning=False` no longer duplicates the answer (`_split_stream` is reasoning-aware); `_drop_socks` closes before
  clearing (churn-safe reconnect, no fd leak). The reconnect-wedge itself is the tail side, already fixed in PR #26.
- **✅ Wire/transport (DONE, PR fix/wire-alloc-dos, 25 hostile-frame tests):** the 64-bit length prefix is now
  capped pre-alloc (`MAX_FRAME`, env `M25_MAX_FRAME`, default 256 MiB) in BOTH codecs, and `_unpack` validates a
  tensor's declared shape against its blob length — closing an EMPTY-blob + huge-shape frame that drove
  `torch.empty(attacker_shape)` (a third alloc vector beyond the finding). The libp2p transport's malformed-frame
  guard was already restored earlier. Adversarially verified (pre-fix allocated a 1M-elem tensor from a 0-byte blob).
- **✅ Contained bugs (DONE):** batched-decode KV write now bound-checked (`_decode_kv_check`, mirrors the prefill
  guard → clean RuntimeError instead of an OOB scatter CUDA-assert that killed the stage; CPU-tested boundary, live
  OOB→clean-error warm-validate pending). `select_ring` false-infeasible was ALREADY fixed (subnet-blind co-location
  cover + require-compatible cover, `tests/test_topology.py::test_no_false_infeasible_rtt_trim` — roadmap line stale).

**TIER 4 — cleanup (38 medium + 19 low):** batched-path perf (per-layer host syncs, redundant full-cache copy,
synchronous batched prefill), test-gaps on load-bearing logic (EAGLE bookkeeping, tree primitives, fetch trust
root — several now covered by `tests/test_eagle_draft.py`), dead code (`shard/specdec.py` stub, scheduler), and
doc/state staleness (STATE.md/FLEET_STATE.md/RESUME_B.md dead — cull or supersede). Detail in the findings file.

**LATER (unchanged, not fleet items):** two-tier co-located fast interactive deploy; depth-aware hybrid (n-gram
depth=4 / EAGLE depth=1); batch-invariant emulation MoE (verifiable batched, OOMs vLLM 0.23); train-our-own
EAGLE-3 only if the stock head underperforms (~$400–2000, SpecForge).

## KEY DECISIONS (don't relitigate)
- **REGIONAL-FIRST; the high-RTT global cell is DROPPED (2026-07-02, leyten).** Steady-state, rings are
  REGIONAL by construction — `select_ring`'s whole job is picking close subsets; a global ring is a
  placement failure, not a target regime. The global measure was only ever go/no-go for tree-verify when
  tree LOST on tight rings; v2 WINS on the tight EU ring (+18%), so no decision hangs on a global number
  (directionally free: more WAN idle → tree wins by more; betanet thin-supply cross-region rings are a
  transient we tolerate, not optimize). Design + marketing numbers are regional numbers.
- **Drafter = EAGLE-3, NOT MTP/DeepSeek.** Vocab-lock: a drafter must emit M2.5's 200064 vocab → DeepSeek heads
  don't transfer; M2.5 MTP weights were never released. EAGLE-3 > MTP in accept anyway. (DeepSeek-q answered.)
- **Tree is the target; chain-validate first** — don't build intricate tree-verify on an unvalidated EAGLE base.
- **Lossless ⇒ the drafter port needs NO bit-exact vLLM parity** — only to predict well; tune accept empirically.
- **On a WAN ring the drafter's COMPUTE is free** (hidden by the round-trip) — only accept-LENGTH matters, not
  draft speed. So "faster drafter" (MTP parallel heads) doesn't help; "more accurate / wider tree" does.
- **Benchmark honesty:** reasoning ON, diverse real prompts, never copy-repetition + think-skip (those inflated
  every past number). `research/m25_honest_bench.py` is the permanent measure.
- **Engine-genericity:** own the moat (ring/transport/spec-decode/verification/economics), RENT model execution +
  the drafter MODEL (EAGLE head) behind the `local_draft` seam.

## OPS PLAYBOOK (vast — STOP re-learning this)
- **Provision:** `vastai search offers 'gpu_name=RTX_5090 num_gpus>=N cuda_max_good>=13.2 rentable=true ...'`.
  Image `vastai/base-image:cuda-13.2.1-auto`. SSH key `/root/.ssh/vast_c0mpute` (account key, auto-attached).
- **~40% of boxes are duds** (this session: broken DNS, hf_transfer stall, sshd-won't-load-key). So:
  - **VERIFY UPFRONT before any 115 GB pull:** (a) SSH works (retry ~2 min for key propagation; if still denied,
    destroy — don't pay for unreachable), (b) raw HF speed (`curl -r 0-524288000` a shard) > ~100 MB/s.
  - **DNS fix:** many boxes have a dead local resolver → `echo nameserver 8.8.8.8 > /etc/resolv.conf` first.
  - **hf_transfer stalls** (freezes mid-download): fall back `HF_HUB_ENABLE_HF_TRANSFER=0`.
  - Prefer non-Asia for low-latency rings; use `inet_down` filter but it's often wrong — verify.
- **Ring launch:** `phase0/m25_scatter_pipe.py --order REGION:iid:lo:hi ... --K 8 --depth 4 [--batch B]
  [--warm-only]`. `--warm-only` warms stages+sidecars then STOPS so a measurement tool runs as the SOLE first
  coordinator (the ring's nxt_sock breaks if a gateway connects first → ALWAYS re-warm before a new coordinator
  process). M2.5 needs ≥5 stages on 5090s (115 GB / 32 GB). fp8 KV (`M25_KV_FP8=1`) for B≥4 at ≥16k.
- **Teardown:** `echo y | vastai destroy instance <iid>` (prompts y/N; piping is required), then verify
  `vastai show instances-v1 --raw` == 0. Always tear down idle boxes (cost).
- **Provision/bootstrap tools:** `scratchpad/swarm_up.py` (rent+bootstrap N), `scratchpad/swarm_boot.py`
  (bootstrap pre-curated iids). They push code + `/tmp/sidecar` + `.hf_token` + pull layer ranges.

## KEY FILES + FLAGS
- `phase0/m25_stage.py` — the M2.5 PP stage. Flags: `M25_BATCH`(=B), `M25_BATCH_MOE`(batched grouped-GEMM),
  `M25_MOE_BACKEND`(cutlass|emulation|marlin), `M25_KV_FP8`, `M25_KV_MAXLEN`, `M25_SDPA`, `M25_EAGLE`(aux capture),
  `M25_EAGLE_AUX`(=1,30,58). `_AUX` holds captured aux hidden states.
- `phase0/m25_pipe.py` — `coordinate_pipe`(single, +`reasoning`, +`_unpack`/`_eagle_seed` EAGLE seeding),
  `coordinate_pipe_batch`(batched, decode-rate timer fix), `serve`(+`_merge_aux` aux threading),
  `make_drafter`(THE drafter factory: n-gram, or n-gram+EAGLE hybrid when `M25_EAGLE=1`; `M25_EAGLE_DIR`=head).
- `phase0/eagle_draft.py` — `EagleDrafter` + `HybridDrafter` (the split). `phase0/ngram_draft.py` — `+matched` flag.
- `phase0/m25_tools.py` — `render_ids(reasoning=)`. `phase0/m25_gateway.py` — OpenAI /v1, `reasoning`/`reasoning_effort`.
- Benchmarks: `research/m25_honest_bench.py` (THE honest measure), `m25_eagle_gonogo.py` (vLLM accept),
  `m25_ctx_table.py` (ctx sweep), `m25_batched_moe_bench.py` (per-stage decode ms).
- Receipts: `docs/receipts/m25-honest-reasoning-baseline-20260629.md`, `m25-batched-serving-fixed`(memory).
