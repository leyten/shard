# 🚀 ROAD TO LAUNCH — the single source of truth

> **The rule:** when every **P0** and **P1** below is checked, it's good to launch. Everything else is post-launch.
> **Launch = "no wizard":** a stranger runs *one command*, their GPU joins the network on its own, their dot
> lights up on a live map, and it works — with nobody (no operator, no SSH, no hand-holding) in the loop.
>
> This file is THE list. If it's not here, it's not a launch blocker — stop carrying it in your head.
> _Last synced: 2026-07-20._
>
> **REMAINING BLOCKERS AT A GLANCE (2026-07-20):** P0-#1 residue (challenge sketch, warm re-join
> receipt, `sidecar -seed`, manifest resolution) · P0-#3 relay automation · **P0-#5 EAGLE watchdog
> (mitigation SHIPPED, PR #120 — residue = controlled-ring validation + root cause)** · P0-#6
> churn-survival PROOF · P1-#3 WSL2 turnkey · P1-#4 adversary hardening → then the rehearsal day.
> DONE: P0-#2, P0-#4, P1-#1 (map live), P1-#2 (Leg 8 + epoch + pay-model built+merged).

---

## ✅ DONE — the mountain you already climbed (do NOT re-litigate these)
The **physics is proven, with receipts, repeatedly.** These are settled:
- **Interactive speed** — 20–30 tok/s solo (graph-aux).
- **Permissionless join + measured admission** — a box probes itself → the network assigns its role.
- **Torrent weight propagation** — nodes pull their model shards from *peers*, every byte CID-verified.
- **Verifiable serving** — per-stage signed receipts, fail-closed (a cheating node is catchable).
- **Batched per-stream viability** — 20–30 tok/s/stream on most content classes (fat/hetero rings).
- **Any-device** — 96 GB fat card joined + served; Apple/MLX gate green; **a residential home 4090
  (double-NAT, WSL2, mid-game) joined via relay hole-punch, torrented its weights from a peer, and
  served M2.5 — 2026-07-14.**

You have a working decentralized inference network. What's left is turning "I can drive it" into
"anyone can use it and it won't embarrass me."

---

## 🔴 P0 — LAUNCH BLOCKERS (without these, a stranger can't join or it breaks in public)

1. **The node daemon — self-serve join (Leg 7).** *THE gate.* Today every join was hand-wired by an
   operator over SSH. Target: a stranger installs one thing → it enrolls, gets measured, gets a role,
   torrents its weight range, and serves — **zero operator in the loop.** (Spec: c0mpute `NODE_DAEMON.md`;
   ships inside `@c0mpute/worker`; needs the `python -m shard.stage` entrypoint + the runtime shipped as a
   signed content-addressed artifact so there's no fragile `pip` step.)
   _Progress 2026-07-15/16 — LARGELY DONE: `python -m shard.stage` (shard #104) + the `--mode shard` daemon
   (c0mpute #28 skeleton, #29 mock-orchestrator harness, #30 self-provision + forward-leg addressing, #31
   one-command test, #32 verified peers-first fetch via `python -m shard.fetch` shard #108). **A stranger's
   box: `npm run try-shard` → self-provisions (engine+venv+sidecar+weights, ZERO env vars) → enroll → announce
   → assign → pull (verified) → stage READY → serving.** Multi-stage rings FORM (forward-leg peer multiaddrs in
   swarm:assign; 2-node libp2p ring proven, bytes across both sidecars). Auto-update REMOVED (c0mpute #27).
   Sidecar release CI = shard #106. **Remaining for the checkmark:** node-side challenge sketch (P0-#1/verifiable
   edge; never spot-check shard swarms until it exists), warm re-join ≤3min acceptance receipt, standby seeding
   (`sidecar -seed`) to light up the wired torrent path, network signed-manifest resolution from `manifestRef`._

2. **Kill the portability landmines** — every "works only on a vast /root box" assumption. Found live
   today: ~~`m25_pull_range.py` hardcodes `/root/.hf_token`~~ ✅, ~~`node_kv`'s flat `import transport` needs
   `PYTHONPATH` off the flat layout~~ ✅ (both dead in shard #104, pinned by a clean-env no-PYTHONPATH
   subprocess gate). Plus the whole SSH + `/root`-flat-layout premise (that half = the daemon, P0-#1).
   A stranger's box (home dir, non-root, weird paths, WSL2) must Just Work with none of today's hand-patches.

3. **Relay / NAT infrastructure, automated.** NAT'd home nodes reach the network via a public relay +
   DCUtR hole-punch — *proven today* — but someone must RUN public relays and the daemon must auto-discover
   + reserve on them. Today I ran the relay by hand on one box.

4. ~~**OpenAI-API correctness** (audit M2)~~ ✅ **DONE** (verified 2026-07-15; was remediated in the
   audit fix, PR #96 — LAUNCH.md was stale). `phase0/m25_gateway.py`: strict `max_tokens`/`max_completion_tokens`
   cap separated from context headroom + truncated via `_cap_output`; earliest-EOS enforced on BOTH the
   streaming (`_cap_output` before `detok.feed`) and final paths; `tool_choice` none/named/required validated
   AND enforced (errors if a required tool call is missing); non-greedy `temperature`/`top_p`/`top_k` rejected
   400 (decoding is greedy). 61 gateway contract tests green.

5. **The g-lever (EAGLE) reliable on home-node topologies.** EAGLE speculative decode is what lifts a
   scattered ring past its ~2 tok/s transport floor. It works on datacenter rings but **silently hung the
   coordinator on the residential-tail path — 2026-07-14.** Must be fixed + robust, or home nodes serve at
   an embarrassing 2 tok/s. _(Being fixed offline — reproduce on a controlled ring, not on rentals.)_
   _Progress 2026-07-20 — **MITIGATION SHIPPED (PR #120), mitigation-first per plan:** every
   EAGLE-implicated stall class now ends "worst case slower, or a clean fast fail — never a silent
   hang." Four layers, each covering a class the others can't: **L1** coordinator draft-budget
   watchdog (a slow/wedged drafter — the one leg no socket timeout sees — degrades the job to
   n-gram IN PLACE, pipelining restored); **M1** the tail's untimed return socket got a
   per-PROGRESS stall bound (one wedged send used to park the tail inside sendall forever and hang
   the whole warm ring — the strongest root-cause candidate, and exactly why the CPU fake ring
   never reproduced it); **eagle:0 on the reset wire** (a degraded session silences aux on every
   stage — the degraded arm equals the proven plain ring ON THE WIRE); **L2** `shard.coordinate`
   runs ONE degraded retry on a mandatory fresh re-dial, resuming committed tokens under the same
   settlement nonce (receipts sweep once, deltas no dup/gap); **L3** a per-job stall backstop
   (prefill replies count as progress) with an unconditional os._exit → daemon restart →
   fail-closed complete. Design panel + independent adversarial verify (1 defect found+fixed);
   22 new tests incl. a real-os._exit subprocess proof; suite 584 green. **Remaining for the
   checkmark:** the controlled-ring proof (master arm reproduces the wedge, #120 arm survives it +
   a tc-throttled thin-uplink arm — runbook in `.claude/plans/eagle-watchdog-mitigation.md`), root
   cause pinned, and the small daemon follow-up (restart coordinator with M25_EAGLE=0 after a
   stall-kill FATAL)._

6. **Self-healing node lifecycle** (no operator babysitting). Nodes churn constantly in the wild — join,
   leave, die mid-serve. Today the launcher needed 3 relaunches + a reboot + a manual box-swap. The network
   must survive dud/dropped nodes on its own. (Largely the daemon's job — pairs with P0-1.)
   _Progress 2026-07-20 — the MECHANISMS all landed: daemon self-heal restart budgets (c0mpute #28),
   lease-freeing on `onNodeGone` + auto-re-form from free candidates (#34), fail-closed jobs (#36), and
   the assignment-EPOCH settlement fix so a healed/re-placed job pays correctly (#37). **Remaining = the
   PROOF:** kill a stage mid-serve in the sim → swarm re-forms → the next request serves; then once on
   the rehearsal ring. (Mid-job resume exists engine-side; that depth is post-launch polish.)_

---

## 🟠 P1 — NEEDED FOR A CREDIBLE LAUNCH (technically works, but weak/risky without these)

1. ~~**The live map — the visible layer.**~~ ✅ **DONE + DEPLOYED (2026-07-15/16): https://shard.c0mpute.ai.**
   A DoubleZero-style spinnable 3D globe (pure-canvas dotted sphere, no libs) is the whole page: real Natural
   Earth land + country borders projected on the sphere, glowing green serving nodes, raised great-circle arcs
   with token pulses. It's a NETWORK EXPLORER — click a node → detail panel (role, layer range held, up/down,
   RTT, uptime, receipts). Locks onto the visitor's continent on load (timezone-based), then drag-only. Stats
   flank it (gpus online, countries, rings, throughput, tokens served). Built in c0mpute's own design system
   (data.c0mpute.ai: pure black / white-graded / argent-pixel numerals / green live dot; links the SAME Typekit
   kit so the font is real). Source: c0mpute repo `data-site/network.html` (c0mpute #33), deployed via nginx on
   the kloot box (`/var/www/shard.c0mpute.ai/`; see memory `shard-demo-deployment`). **WIRED TO LIVE STATE 2026-07-20**
   (c0mpute #38/#39): loopback-gated `/api/network` (public shape identity-free — truncated PeerIds, no
   IPs/pubkeys/accounts, test-enforced) → `network.json` generator (5-min systemd timer, server-side geo
   via cached /24 lookups, jittered coords) → the page HOT-SWAPS the sim with any fresh non-empty feed
   (sim stays the fallback so the globe is never empty pre-launch). Deployed + verified live; flips to
   real data automatically when real nodes announce. The
   "chat window watching one prompt stream" is a SEPARATE surface (inference stays private per leyten; the map
   is network-view only).

2. **Control plane + settlement (Leg 8 wiring).** Requests in → streams out → who-served-what → payment.
   Includes the **assignment-EPOCH fix** (a healed/re-placed job must not settle as fraud — a correctness bomb).
   _Progress 2026-07-15/16 — SERVER HALF DONE: (c0mpute #34) the live server AUTO-FORMS rings from real
   announces (`attachSwarmLoop` `resolveModel` + debounced form-from-free-candidates; was never called on the
   running server — demo-only); (c0mpute #35) `serveRequest` DISPATCHES a request to a ready swarm's coordinator
   (`swarm:job` + nonce), relays `swarm:job_token`/`swarm:job_complete` back to the client, and settles — the
   orchestrator routes sharded-model requests to it. Proven no-GPU end-to-end (`scripts/leg8-serve-test.ts`,
   10/10): auto-form → dispatch → stream → complete → settlement credits both stages. **NODE/ENGINE half DONE
   2026-07-17 (shard #114 + c0mpute #36):** `python -m shard.coordinate` (stdin jobs, SHARD_JOB_TOKEN/DONE/FATAL,
   settlement-nonce threaded into every stage's receipt — the correctness catch: settleJob verifies against the
   swarm:job nonce), the daemon `CoordinatorProcess` + `swarm:job` handler (fail-closed completes), and the
   tail→coordinator RETURN TUNNEL closed with m25_scatter_pipe's proven sidecar wiring (zero tail-engine
   changes). Proven GPU-less with 2 real daemons: request → served → settled, both stages credited by layers
   (`SERVE=1 npm run try-shard`). **Live-ring validation DONE 2026-07-18** (warmring-20260718 receipt:
   shard.coordinate served a real job on a real 6-box ring; 6 receipts tiling [0:62) verified under pinned
   assignments, every stage signing the injected settlement nonce). **Assignment-EPOCH fix DONE 2026-07-20** (c0mpute #37:
   per-job settlement snapshot frozen at dispatch — churn can neither strand honest work unpaid nor frame
   the coordinator as fraud; epoch-settle-test 9/9). **PAY-MODEL BUILT 2026-07-20 (c0mpute PR #41, OPEN —
   merges/deploys at launch):** leyten's decision applied — USDC via the EXISTING revenue-share economy;
   a settled job's collected revenue splits FLAT BY LAYERS, then each stage keeps ITS OWN
   `getWorkerRevenueShare` (per-worker cut AFTER the split — staked 80% / unstaked 70%, never blended);
   one `recordEarning` per stage on the existing payout rails; swarm errors now refund. GATED behind
   `SWARM_PAYOUT_ENABLED` (off) so it's inert until launch; swarm-payout-test 11/11. Price
   `pricePerMTokensUsd = $0.50/M` staged on the model profile. **P1-#2 is CODE-COMPLETE.** Residue is
   launch-time only: (a) flip the flag + deploy at go-live; (b) Phase-2 per-token BILLING (make $/M drive
   the actual charge — reshapes the live submit path, its own reviewed deploy, POST/at-launch)._

3. **Windows / WSL2 turnkey.** Most home users. Proven workable today (WSL2 *mirrored* networking + CUDA),
   but the setup must be one step, not the manual dance we did.

4. **Adversary-safety hardening** (audit). Path-traversal/symlink escape in seeding + fetch, key file
   permissions, per-peer/global fetch budgets + deadlines. Today it's **operator-safe, not adversary-safe** —
   which is fine while you run every node, and not fine the moment strangers do.

---

## 🟡 P2 — POST-LAUNCH POLISH (real, not blocking — ship without them)
- Perf levers: batched-prefill pipelining (PR #100), cross-request prefix-KV cache (PR #101),
  tree-frame CUDA graphs (the last prose-bar + g lever, plan `tree-graph-capture.md`).
- Prose per-stream bar on 5090-*only* rings (already met on fat/hetero rings).
- Full-model MLX reference on a ≥96 GB Mac.
- High-availability / production ops (the audit's HA gap).

---

## ⚪ NOT ENGINEERING — your call, decouple from the tech launch
- **Market / economics (Leg 8 the-money-part).** ✅ **DECIDED + BUILT 2026-07-20** (leyten): USDC payouts
  ride the EXISTING credits/revenue-share economy — no new token mechanics, no points ledger; farming is
  unprofitable BY CONSTRUCTION because the platform keeps its 30%/20%-by-staking cut. Per-worker cut after
  a flat-by-layers split; `$0.50/M` price (Kloot's call). Code = c0mpute PR #41 (open, gated off, deploys
  at launch). Remaining money-side item = Phase-2 per-token billing (at/post-launch). See memory
  [[c0mpute-economics-applied-to-shard]].
- **Paper publish / announcement timing.** Yours.

---

**Definition of launch, one line:** *all P0 ✅ + all P1 ✅ → flip the switch.* A stranger joins in one
command, their dot appears on the map, tokens stream, and none of it needs you. That's the finish line.
