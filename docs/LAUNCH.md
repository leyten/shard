# 🚀 ROAD TO LAUNCH — the single source of truth

> **The rule:** when every **P0** and **P1** below is checked, it's good to launch. Everything else is post-launch.
> **Launch = "no wizard":** a stranger runs *one command*, their GPU joins the network on its own, their dot
> lights up on a live map, and it works — with nobody (no operator, no SSH, no hand-holding) in the loop.
>
> This file is THE list. If it's not here, it's not a launch blocker — stop carrying it in your head.
> _Last synced: 2026-07-16._

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

6. **Self-healing node lifecycle** (no operator babysitting). Nodes churn constantly in the wild — join,
   leave, die mid-serve. Today the launcher needed 3 relaunches + a reboot + a manual box-swap. The network
   must survive dud/dropped nodes on its own. (Largely the daemon's job — pairs with P0-1.)

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
   the kloot box (`/var/www/shard.c0mpute.ai/`; see memory `shard-demo-deployment`). **STILL A SIMULATION** —
   wiring it to the orchestrator's live node+swarm feed is the follow-on (now unblocked: rings auto-form). The
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
   (`SERVE=1 npm run try-shard`). **Remaining for the checkmark:** live-ring validation (vast), the
   assignment-EPOCH fix (still the correctness bomb), and the pay-model $ credit mapping (leyten's fork —
   `recordSwarmStageEarning` stub)._

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
- **Market / economics (Leg 8 the-money-part).** Everything it needs *exists* (role verdicts,
  pay-by-layers, settle seam). It's a product-direction decision (global-truth vs demand-artifact, staking,
  emissions) — not a code blocker. Don't let it gate the tech launch.
- **Paper publish / announcement timing.** Yours.

---

**Definition of launch, one line:** *all P0 ✅ + all P1 ✅ → flip the switch.* A stranger joins in one
command, their dot appears on the map, tokens stream, and none of it needs you. That's the finish line.
