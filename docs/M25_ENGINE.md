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

## RESUME HERE  (the one next action)

### ⇒ PRIMARY FOCUS (leyten, 2026-07-02 night): BREAK THE ~5 tok/s REASONING FLOOR — step back, find the lever or the breakthrough
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
- **tok/s TARGET (normal reasoning-ON, single-stream, scattered ring — the honest projection):** today **~5.7–7**
  (merged serial-path). After TIER-1 perf (tree-verify fixed + topology order + small levers): **~10–12 on a good
  tight EU ring**, **~5–6 on high-RTT global scatter** — approaching the ~12–20 physics cap (a perfect drafter can't
  accept every novel reasoning token; the reason-math cell is the hard floor, ~7–9 even post-tree). NOTE: most of
  the 79 fleet findings are NOT tok/s (trust/gateway/wire); tree-verify + topology are the only real speed levers
  left on this path. This number is ON the scattered ring — NOT via co-location ([[never-colocate-usable-speed-on-scattered]]).
- **TWO-TIER framing (decided):** **scattered ring = cheap/permissionless/THROUGHPUT** (latency-tolerant); a
  **co-located/regional node or mini-cluster = fast/INTERACTIVE** (M2.5-NVFP4 ~115 GB fits on 1× H200 / 2× H200 /
  4× RTX6000-Blackwell → no WAN → 30–50 tok/s, physics-guaranteed). WAN-sharded single-stream is the *hardest*
  way to serve M2.5; use the right tier per workload. The engine serves the whole spectrum.

## PROVEN  (numbers + receipts — measured, honest)
| capability | status / number | source |
|---|---|---|
| Batched throughput | **155 tok/s agg @16k (2.60× single), coherent** (B=4, batched-MoE, fp8 KV) | commit f3894d6, m25-batched-serving-fixed |
| Single-stream DRAFTABLE (copy/RAG/verbatim) | 50–81 tok/s (n-gram, accept high) | m25_ctx_table |
| **Single-stream NORMAL reasoning-ON (EAGLE hybrid)** | **~5.7 tok/s decode-wtd on a jittery lottery ring / ~7 on a good tight EU ring** (2026-07-02 warm A/B, merged serial-path; was ~3 n-gram-only, ~1.8 raw) | receipt m25-eagle-serial-path-ab-20260702 |
| **TREE-verify v2 (hybrid, tight EU ring)** | **+18% decode-wtd over chain on the SAME warm ring (3.9→4.6); reason-math 4.8→6.0, reason-logic 3.0→4.7, code-edit 4.5→6.1; g novel 3.7→4.5 at M=12** — flips v1's 'tree loses tok/s on tight rings' (payload, not physics). rag-quote gap = sync tree vs pipelined n-gram | receipt m25-tree-verify-v2-ab-20260702, branch eagle/tree-verify-v2 |
| **Ring churn survival (wedge fix)** | coord kill -9 mid-decode → new coordinator, same ring, NO re-warm (6.7→6.6 tok/s); forward links rebuild, tail keeps warm KV | receipt m25-ring-wedge-smoke-20260702, branch fix/ring-wedge-receipt-truth |
| Tools / multi-turn / long-ctx(≥30k needle) | PASS | _validate pass, prior receipts |
| Trustless verification | signed per-stage receipts, lossless — coverage now vs TRUE model depth, fail-closed on empty (fix on branch fix/ring-wedge-receipt-truth; freshness/replay binding still open, TIER 2.2) | shard/receipt.py, tests/test_receipt_coverage.py |
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
2. Receipts have no freshness/content binding (old receipts replay); bind to the job's actual tokens/activations.
3. Tree-verify path emits NO receipts (verification silently off under M25_TREE) — wire the hash-chain through it.
4. Verified-fetch trust root (`shard/fetch.py`) is bypassed by the real M2.5 deploy path (HF pull, unverified) —
   route the betanet weight pull through the content-addressed manifest check.

**TIER 3 — ROBUSTNESS (gateway + wire + contained bugs; a batch-into-one-session hardening pass):**
- **Gateway** (`m25_gateway`, 4 high): reconnect wedges the warm ring; client-disconnect mid-stream silently
  re-runs the ENTIRE generation; `reasoning=False` streaming duplicates the whole answer as reasoning+content; a
  slow/stalled streaming client blocks the single-stream ring up to 30 min (add a write timeout / decouple).
- **Wire/transport** (2 high, security): unauthenticated 64-bit length prefix → any peer forces unbounded alloc
  (cap it pre-alloc); the libp2p PRODUCTION transport LOST wire.py's malformed-frame hardening (one bad frame
  kills a stage — port the try/except).
- **Contained bugs:** batched-decode KV write has no `M25_KV_MAXLEN` guard (OOB scatter kills the stage — copy
  the prefill guard); `select_ring` false-infeasible (require-blind `>_TRIM` funnel returns None when a feasible
  ring exists — TIER 1.3 depends on this).

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
