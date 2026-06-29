# M2.5 EAGLE-3 hybrid drafter — FIRST on-engine measurement (2026-06-29)

**Verdict: GATE FAILS as-is.** The EAGLE-3 hybrid lifts reasoning accept to only **~0–3%**, not the
~2.5 / ~30% the head's authors reported (thoughtworks 1.78× MT-bench). The n-gram path is healthy, so
the ring is fine — the fault is isolated to the EAGLE drafter. Root cause not yet pinned (see below).

## Setup
- **Ring:** 6× RTX_5090, all-EU scattered over libp2p (coherent low-RTT scatter, NOT colocated):
  `Sweden(head,0:10) → Spain(10:20) → Norway(20:30) → France(30:40) → Czechia(40:51) → Greece(tail,51:62)`.
- Warmed with `M25_EAGLE=1` (stages capture aux hidden at layers [1,30,58], `_merge_aux` threads them
  to the tail, coordinator seeds `EagleDrafter` via `_eagle_seed`, runs depth=1).
- Coordinator = `research/m25_honest_bench.py` on the head box, sole coordinator, `M25_EAGLE=1
  M25_EAGLE_DIR=/root/m25-eagle` (thoughtworks/MiniMax-M2.5-Eagle3), sharing stage-0's GPU.
- EAGLE head loaded cleanly on-engine (no OOM / shape error) — first real load.

## Result (single-stream, reasoning ON, K=8, depth forced to 1 on the EAGLE path)
| category | tok/s | accept | g | path |
|---|---|---|---|---|
| reason-math | 0.7 | **1%** | 1.1 | novel → EAGLE |
| reason-logic | 0.8 | **3%** | 1.2 | novel → EAGLE |
| open-chat | 0.7 | **0%** | 1.0 | novel → EAGLE |
| code-edit | 1.1 | 8% | 1.7 | mixed (docstring = mostly novel) |
| rag-quote | 1.6 | 22% | 2.8 | n-gram (verbatim copy) |
| agentic-tool | 0.7 | 1% | 1.1 | novel → EAGLE |

Decode-weighted **0.9 tok/s** — worse than the n-gram baseline because the EAGLE path forces depth=1
(no pipelining) while accept stays ~0. n-gram path confirmed working (rag-quote 22% / g 2.8).

## Diagnosis so far
- **RULED OUT — wire codec.** `shard/transport.py` `_pack`/`_unpack` recurse through nested dicts and
  tensors at any depth, so the aux dict `{"1":…, "30":…, "58":…}` transmits fine.
- **Reviewed, correct on paper** — capture (`m25_stage.run_block`: `_AUX[L.li] = layer-output[0]` for
  `L.li in [1,30,58]`, matches the head config's `eagle_aux_hidden_state_layer_ids`), threading
  (`_merge_aux` accumulates head→tail), seed order (`_eagle_seed` stacks in `[1,30,58]` order → fc 3H→H).
- **INCONCLUSIVE — live aux-value probe.** `scratchpad/diag_eagle.py` (checks aux arrival + EAGLE-vs-greedy
  accept proxy) hung in import/load on the head box (`wchan=pipe_read`, never opened ring sockets — a
  tooling issue, not the ring). The ring also **wedges after each coordinator disconnects** (tail's
  `serve_tail_direct` re-`accept()`s 2 fresh conns but the predecessor stays connected) → every new
  coordinator needs a re-warm. Two warm cycles spent; did not get the aux values. Torn down to stop billing.

## ROOT CAUSE FOUND OFFLINE (post-teardown, no GPU) — aux LAYER off-by-one
Diffed our port against the vLLM eagle3 reference + the head's actual tensors, and verified the plumbing:
- **RULED OUT (offline):** wire codec; aux-survives-wire (`scratchpad/aux_plumbing_test.py` — bit-identical
  round-trip + correct [3,H] seed); `propose` STRUCTURE (midlayer concat order, residual handling, next-step
  hidden, d2t all match vLLM `llama_eagle3.py`); fc-norm (the head ships NO `fc_norm`/`input_norm`/
  `norm_before_fc` weights — just fc/midlayer-norms/norm/lm_head/d2t/t2d — so raw-aux→fc is correct).
- **THE BUG:** the head config's `eagle_aux_hidden_state_layer_ids = [1,30,58]` are vLLM **aux-LIST indices**,
  where index 0 = the EMBEDDING output and index K+1 = the OUTPUT of decoder layer K (vLLM's target forward:
  `_maybe_add_hidden_state([],0,…)` then `idx+1` after each layer — `vllm/model_executor/models/llama.py`).
  So `[1,30,58]` = post-layer-**{0,29,57}**. Our `m25_stage.run_block` captured by RAW layer index → stored
  post-layer-**{1,30,58}**. We fed the trained `fc` features shifted by one layer → degraded predictions
  (~0 accept). **Fixed:** capture layer L keyed by its vLLM index `L.li+1` (commit this session); env-tunable
  (`M25_EAGLE_AUX=2,31,59` reproduces the old/wrong capture for an A/B).

## Next step — CONFIRM the fix on a scattered ring (one cheap run)
Re-provision EU ring (now `swarm_up` EU-filtered), warm with `M25_EAGLE=1`, run `m25_honest_bench.py`. GATE:
reason-math/-logic accept should jump from ~1–3% → meaningfully higher. A/B the default `[1,30,58]` (fixed)
vs `M25_EAGLE_AUX=2,31,59` (old) to prove the off-by-one was it. If accept is better-but-not-~2.5, then chase
the secondary knobs (seed position `h_n`, `next_hidden=final|prenorm`). If it's still ~0, the aux VALUES need
a single-box vLLM-eagle3 reference comparison (needs an H200 for single-GPU M2.5, no TP-P2P).

## Next step (offline / single-box — NOT a scattered ring)
A correctness bug is **network-independent**, so debug on ONE box (no ring-wedge, no WAN, cheap): load M2.5
+ the eagle3 head, capture aux the way a reference (vLLM eagle3 / SpecForge) does, compare to our
`run_block` capture, and validate `EagleDrafter.propose` against the reference's draft. This does NOT
contradict "measure the product on a scattered ring" — that principle is about the *throughput regime*;
finding a *code/representation bug* is regime-independent and the single box is the right tool. Single-GPU
(no TP) also sidesteps the vLLM-M2.5-TP-needs-P2P dead end.

Cost: ~6–8 USD of vast churn (duds + topology swap + 2 warm cycles). Result: a clean, honest gate answer.
