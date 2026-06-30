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

**LATEST (2026-06-30 late) — NEXT ACTION = validate the self-optimizer on a real ring.**
- **Handshake bring-up deadlock FIXED** (branch `ops/tail-handshake`): the tail required BOTH the coord-return
  AND the (lazily-connecting) predecessor before acking `ret_ok` → circular deadlock. `_tail_accept` now acks
  the return channel the instant it's identified. Validated on a real decoded row. Covers coord + gateway.
- **Usability table ran but on a JUNK ring (2.6 tok/s).** Root cause: the M2.5 launcher NEVER selected nodes —
  it took the rental lottery's boot order (spread boxes incl. Spain/Norway + a 400W-capped GPU). The drafter
  reproduced EXACTLY (reason-math 34%/g3.7) → the engine is fine; the gap was 100% bad hardware + no selection.
  **Reframe (real comparables, don't despair):** Petals ≈ 5-6 tok/s for a 70B model; we do ~12 for 230B on a
  GOOD ring → we're AHEAD. The ~2× WAN penalty we measured matches Petals' own geo-distributed number.
- **FIX BUILT: `shard/topology.select_ring`** (branch `feat/topology-select-ring`) — the self-optimizer's pure
  core: from a measured pool pick the subset+order+layer-split minimizing predicted decode step-time (WAN +
  summed compute = physical, no hand-weights); drop throttled/far/co-located nodes; fewest-fattest stages; pin
  the coord via `require`. Adversarially reviewed (2 critical false-"infeasible" bugs fixed), regression-tested,
  calibrated (predicts the measured tok/s). `scratchpad/plan_ring.py` = vast glue (measure → select → --order).
- **NEXT:** over-rent ~8 (swarm_up now has a free-VRAM gate + /24-subnet dedup, NOT geolocation), `plan_ring`
  selects the ring, warm it, benchmark → compare predicted vs actual tok/s (validates the whole loop with a
  number). THEN the self-optimizer graduates to c0mpute (shard stays the engine; c0mpute→shard dep only).
- **PRs to land:** `ops/tail-handshake`, `feat/topology-select-ring`, + `eagle/chain-diagnostics`,
  `eagle/tree-verify`. **Roadmap (prior-art):** Vivaldi network-coordinates = O(N) all-pairs latency (no N²
  pings) is the node-selection scaling unlock; tree-verify (built, `eagle/tree-verify`) is the ENGINE lever for
  high-RTT global scatter. The `select_ring` test surfaced COORD PLACEMENT (separate-coord case) as a lever too.

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
- **TWO-TIER framing (decided):** **scattered ring = cheap/permissionless/THROUGHPUT** (latency-tolerant); a
  **co-located/regional node or mini-cluster = fast/INTERACTIVE** (M2.5-NVFP4 ~115 GB fits on 1× H200 / 2× H200 /
  4× RTX6000-Blackwell → no WAN → 30–50 tok/s, physics-guaranteed). WAN-sharded single-stream is the *hardest*
  way to serve M2.5; use the right tier per workload. The engine serves the whole spectrum.

## PROVEN  (numbers + receipts — measured, honest)
| capability | status / number | source |
|---|---|---|
| Batched throughput | **155 tok/s agg @16k (2.60× single), coherent** (B=4, batched-MoE, fp8 KV) | commit f3894d6, m25-batched-serving-fixed |
| Single-stream DRAFTABLE (copy/RAG/verbatim) | 50–81 tok/s (n-gram, accept high) | m25_ctx_table |
| **Single-stream NORMAL reasoning-ON** | **~3 tok/s (HONEST baseline; reason-math=1.8, 68s to first visible answer; n-gram only drafts verbatim-reuse)** | receipt m25-honest-reasoning-baseline-20260629, commit da9f11d |
| Tools / multi-turn / long-ctx(≥30k needle) | PASS | _validate pass, prior receipts |
| Trustless verification | signed per-stage receipts, lossless, coverage-checked | shard/receipt.py, PROOF.md |
| Reasoning control (no-think fast mode) | wired (`reasoning` flag, render_ids closes `<think>`) | commit da9f11d |
| **EAGLE hybrid drafter (reasoning)** | **WORKS: reason-math 34%/g3.7/11.8tok/s, open-chat 13%, agentic 50%/g5.0; ~7 tok/s decode-weighted** (was 0.9 broken). Bug was missing context attention (persistent context KV); aux layers {1,30,58} A/B-confirmed > {0,29,57} | branch eagle/chain-diagnostics |

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
  (capture keyed `L.li+1`); codec/wire/structure/fc-norm ruled out offline. **Re-confirm accept on a ring next.**

## ROADMAP / ranked levers (do in this order)
1. **vLLM tree GO/NO-GO** (NEXT) — measure EAGLE-3 reasoning accept on M2.5 (tree number). Justifies #2.
2. **EAGLE TREE-verify in the ring** — the GPU is IDLE during the WAN round-trip, so verifying a candidate
   TREE per traversal is ~free → ~2× accept (2.5→4–5). Needs a tree-attention mask threaded through every stage
   + coordinator best-path selection. (Tree "regresses" only in the batched compute-bound regime; single-stream
   idle-GPU inverts that — it's the natural fit. **The queued big lever — keep in mind.**)
3. **Two-tier deploy** — stand up a co-located fast tier (M2.5 on 1–2 H200 / 4× RTX6000-Blackwell, no WAN) for
   interactive; keep the scattered ring for cheap/throughput. Physics-guaranteed fast single-stream.
4. **Depth-aware hybrid** — n-gram path keeps depth=4 (pipelined), EAGLE path depth=1 (v1 forces depth=1 globally).
5. **Stream the `<think>` live** (UX) — turns the 68 s reason-math wait into R1/o1-style visible thinking; free.
6. Later: batch-invariant emulation MoE (verifiable batched, OOMs in vLLM 0.23 today); train-our-own EAGLE-3 on
   our reasoning/agentic distribution ONLY if the stock head underperforms (~$400–2000, SpecForge).

## KEY DECISIONS (don't relitigate)
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
