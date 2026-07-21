# 🚀 ROAD TO LAUNCH — the single source of truth

> **The rule:** when every **P0** and **P1** below is checked, it's good to launch. Everything else is post-launch.
> **Launch = "no wizard":** a stranger runs *one command*, their GPU joins the network on its own, their dot
> lights up on a live map, and it works — with nobody (no operator, no SSH, no hand-holding) in the loop.
>
> This file is THE list. If it's not here, it's not a launch blocker — stop carrying it in your head.
> _Last synced: 2026-07-20._
>
> **REMAINING BLOCKERS AT A GLANCE (2026-07-21):** the buildable launch list is DONE — what's left
> is **one real-hardware ring session** (validates the whole new stack end-to-end: verified pull →
> seeding → churn → warm re-join ≤3min receipt) plus **leyten-gated publish actions** (npm publish
> `@c0mpute/worker` ≥2.8.3; the one-time offline-key manifest publish; the Ghent WSL smoke) → then
> the rehearsal day.
>
> **DONE 2026-07-21 (one session, 10 PRs, no spend):** sidecar `v0.1.0` PUBLISHED + pinned (was a
> hidden blocker for every Go-less join) · **P1-#4 adversary hardening COMPLETE** — reputation gate
> + spot-check scheduler + auditors wired (c0mpute #47), ring-wide C2 `SHARD_SWARM_TOKEN` closes the
> head allow-all hole (#46), metered OpenAI API routes the swarm model so it's sellable (#46) · **P11
> restart-degraded** coordinator EAGLE-off after a stall-kill (#46) · **P1-#3 WSL2 turnkey** daemon
> side — `wsl-setup.sh` + `WINDOWS.md` (#48) · **P0-#6 churn self-heal PROVEN in sim** + the real bug
> it caught (#45) · **launch-day failure playbook** (#46). DONE 07-20: P0-#1 residue 3/4 (manifest
> resolution, seeding, challenge probe), P0-#3 relays. Earlier: P0-#2/#4/#5, P1-#1/#2.

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
   Sidecar release CI = shard #106.
   _Progress 2026-07-20 EOD — residue 3/4 CLOSED: **network manifest resolution** (shard #125:
   `mf1:<name>@<cid>` refs, `--manifest-cid` bytes-pin + `--expect-*` cross-checks + signed monotonic
   version, all fail-closed engine-side; c0mpute #42: daemon resolves the network doc, pins the baked
   publisher key, self-publish + raw serving pull DELETED — launch runbook residue in c0mpute
   LAUNCH_READINESS item 5); **standby seeding** (c0mpute #42: `-seed` at every sidecar boot,
   `swarm:assign.seeders` = free candidates + `SWARM_SEED_ADDRS`, GPU-less sim proof end-to-end);
   **node-side challenge probe** (shard #126 sketch device-RNG fix + #127 loopback probe door — busy
   refuses mid-job, every termination re-opens, fail-closed sans token; c0mpute #43: daemon answers
   `swarm:challenge` via the door, crypto-random seeds + commit-first projSeed + busy=flake-not-fail;
   spot-checking shard swarms is now MECHANICALLY possible — bank tooling + shadow-mode threshold
   validation before enforcement). **Remaining for the checkmark:** warm re-join ≤3min acceptance
   receipt — REAL-box measurement (weights-load dominated), folded into the P0-#6 churn-proof ring
   session (one ring, two receipts)._

2. **Kill the portability landmines** — every "works only on a vast /root box" assumption. Found live
   today: ~~`m25_pull_range.py` hardcodes `/root/.hf_token`~~ ✅, ~~`node_kv`'s flat `import transport` needs
   `PYTHONPATH` off the flat layout~~ ✅ (both dead in shard #104, pinned by a clean-env no-PYTHONPATH
   subprocess gate). Plus the whole SSH + `/root`-flat-layout premise (that half = the daemon, P0-#1).
   A stranger's box (home dir, non-root, weird paths, WSL2) must Just Work with none of today's hand-patches.

3. ~~**Relay / NAT infrastructure, automated.**~~ ✅ **DONE 2026-07-20 (c0mpute #44 + live infra).**
   NAT'd home nodes reach the network via a public relay + DCUtR hole-punch (proven 07-14; relay =
   rendezvous only, the data path upgrades to direct). Now automated on both halves: **operator** —
   `shard-relay.service` (sidecar `-relay -quic`, persistent key ⇒ stable PeerId, Restart=always)
   runs on two existing paid public boxes, reservations verified end-to-end from a test sidecar;
   **network** — the daemon resolves `/relays.json` off the orchestrator origin at enroll (cached,
   offline-tolerant), VALIDATES every entry (a malformed `-relays` entry is sidecar-fatal — one bad
   list push must never kill the fleet; operator env outranks), and arms `-relays` on every sidecar
   boot so a NAT'd node announces its circuit addrs from first boot. The repo ships `relays.json`
   EMPTY (no public IPs in git); the launch deploy fills it (addrs in the ops notes). _Deferred,
   not launch-blocking: AutoNAT-gated reservation (today every node with a list reserves — harmless
   at launch scale, circuit addrs rank below direct in dial order)._

4. ~~**OpenAI-API correctness** (audit M2)~~ ✅ **DONE** (verified 2026-07-15; was remediated in the
   audit fix, PR #96 — LAUNCH.md was stale). `phase0/m25_gateway.py`: strict `max_tokens`/`max_completion_tokens`
   cap separated from context headroom + truncated via `_cap_output`; earliest-EOS enforced on BOTH the
   streaming (`_cap_output` before `detok.feed`) and final paths; `tool_choice` none/named/required validated
   AND enforced (errors if a required tool call is missing); non-greedy `temperature`/`top_p`/`top_k` rejected
   400 (decoding is greedy). 61 gateway contract tests green.

5. ~~**The g-lever (EAGLE) reliable on home-node topologies.**~~ ✅ **DONE 2026-07-20 (mitigation-first).**
   EAGLE speculative decode is what lifts a scattered ring past its ~2 tok/s transport floor. It worked on
   datacenter rings but **silently hung the coordinator on the residential-tail path — 2026-07-14.** The
   four-layer mitigation makes every EAGLE-implicated stall class end "worst case slower, or a clean fast
   fail — never a silent hang" (PR #120), the knobs are forwardable in production (PR #122), and it's
   ring-validated: **integration PASS** (shipped build serves M2.5 over a real 3-region WAN ring,
   degraded=false, receipts chained) with the **wedge finding banked** (the sustained-backpressure M1
   wedge is not reproducible on a fast datacenter ring by construction — depth-1 EAGLE + multi-MB buffers
   + no NET_ADMIN — which is exactly why 07-14 only hit the home box; M1's socket bound is proven offline
   in test_ret_stallguard.py). Receipt `docs/receipts/eagle-watchdog-ring-20260720.json`. _(Follow-up, not
   a blocker: the daemon restarts a stall-killed coordinator with M25_EAGLE=0 — the P11 restart-degraded
   path.)_
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
   the assignment-EPOCH settlement fix so a healed/re-placed job pays correctly (#37)._
   _Progress 2026-07-20 night — **the SIM PROOF is DONE and it caught a real launch bug (c0mpute #45).**
   `churn-proof.sh`: three REAL daemons, 2-stage ring + one free spare, the TAIL daemon SIGKILLed
   mid-serve. RED first: `onNodeGone` freed the ring and then NOTHING — auto-form's only trigger was an
   ANNOUNCE, so a churned network stayed down until fresh supply happened by (120s receipt in the run
   log). Fix: node death schedules a re-form for its swarm's model. GREEN: form → serve+settle #1 →
   crash-kill → DEGRADED/slots freed → auto re-form from survivor+spare within seconds → serve+settle
   #2 → CHURN_PROOF COMPLETE, exit-code-gated. **Remaining for the checkmark:** the same kill once on
   the rehearsal ring, plus the warm re-join ≤3min receipt (same ring session). (Mid-job resume exists
   engine-side; that depth is post-launch polish.)_

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
   the coordinator as fraud; epoch-settle-test 9/9). **PAY-MODEL BUILT + MERGED 2026-07-20 (c0mpute PR #41,
   on master, GATED OFF — the flag flips at launch):** leyten's decision applied — USDC via the EXISTING revenue-share economy;
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
