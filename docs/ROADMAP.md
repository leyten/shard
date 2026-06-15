# Shard roadmap

Phased so the riskiest thing is proven first and cheaply. Each phase has one goal and a hard pass/fail.

## Phase 0 — prove the transport (target: 1-2 days)

The single hardest thing: reliably serve tokens through a multi-stage split. Do it on easy mode first.

- 2 nodes, **same LAN** (or two boxes in the same datacenter), low latency.
- Split one model that genuinely needs 2 GPUs across them with our own QUIC transport moving activations.
- No speculative decoding yet. Plain token-by-token is fine, it'll be slow, that's expected.
- **Pass = a coherent completion comes out, reliably, 20 times in a row, with a measured tok/s.**

**✓ PASSED (2026-06-15).** 2-node split with per-node KV-cache. Proven on Qwen2.5-14B (29.5GB bf16, OOMs one 24GB card) split 24/24 across two 4090s: 20/20 clean completions, ~17 tok/s decode, via our own TCP transport. Code in `phase0/` (`node_kv.py` = the split node, `bench.py` = the 20x reliability harness). Caveat: nodes are on a low-latency same-host link, not WAN yet.

**Hardened (2026-06-15):** the transport is now fail-fast and instrumented — the direct answer to the opaque-transport failure mode, a black-box edge that just returns "broken pipe". Per-edge socket timeouts (a dead OR frozen node raises a `TransportError` with context — which peer, which decode step, dropped vs timed-out — never an unbounded `recv`); `os._exit` on unrecoverable failure so the process dies in ms, not after a ~10s torch/cuda teardown; per-edge health logging (round-trip latency + uplink MB). Verified by killing (SIGKILL) and freezing (SIGSTOP) the tail mid-generation — head fails cleanly both ways, no hang. This pre-validates part of Phase 1's edge-supervision goal on the easy link.

This was the thing that blocked us all day. It's the momentum milestone.

## Phase 1 — make the transport survive WAN (target: 3-5 days)

- Same 2 nodes, now on **different networks behind NAT**.
- Add hole-punching via a rendezvous (c0mpute orchestrator), relay fallback for symmetric NAT.
- Add fp8/int8 activation quantization to cut uplink bandwidth.
- Add edge supervision: kill a node mid-stream, confirm the pipeline detects it and the request fails cleanly (not a hang).
- **Pass = reliable WAN serving + an honest WAN tok/s number** (will be latency-bound and slow, that's the point, it sets the baseline spec-decode has to beat).

**◑ PARTIAL (2026-06-15).** Reliable WAN serving demonstrated over a real transatlantic link: Norway head → North Carolina tail (~115ms RTT), 2-node Qwen2.5-3B split moving activations over our own TCP transport via a direct open port. Honest WAN tok/s: plain decode **6.0 tok/s** (latency-bound, ~133ms/token round-trip) → speculative decoding **20.5 tok/s on code (3.4×)**, 9.7 on prose (1.6×) — the multiplier cashing out over real internet. Edge supervision (timeouts/fail-fast) carried from Phase 0 hardening. Remaining: NAT hole-punching + relay fallback (a direct open port stands in for now), and fp8/int8 activation quantization.

## Phase 2 — speculative decoding (target: ~1 week)

The payoff. Add the draft-verify loop.

- Small draft model on the entry node, propose K, verify K across the swarm in one traversal.
- Adaptive K based on measured latency + live acceptance rate.
- **Pass = land in the paper's regime: meaningfully more tok/s than Phase 1 on the same links, in the ~8-9 tok/s ballpark at 80ms for a small target.** This is the proof the whole approach is real.

## Phase 3 — permissionless swarm + c0mpute (target: 1-2 weeks)

- One-line installer, auto-installs deps, plug and play.
- `cwt_` worker auth, per-token USDC payout for a node's contribution.
- Dynamic layer allocation across **4+ heterogeneous** consumer GPUs for a real **100B+** target.
- Scheduler handles joins/leaves and rebuilds the pipeline live.
- **Pass = a stranger runs one command, their GPU joins, takes a layer block, and earns for tokens it helped produce.**

**◑ PARTIAL (2026-06-15).** The "100B+ target across 4+ GPUs" piece is proven: gpt-oss-120b (120B params, MXFP4 ~57GB) split **9 layers/node across 4× RTX 4090** (~16GB each, ~64GB total — no single 24GB card holds it) via `phase0/pipeline.py`, the N-node pipeline. Each node loads ONLY its block (`device_map` → `meta` for the rest); coherent output at ~6.3 tok/s. Remaining Phase 3: permissionless one-command join, `cwt_` auth + per-token payouts, dynamic layer allocation, and live pipeline rebuild on join/leave.

## Phase 4 — privacy + hardening (ongoing)

- Boundary-layer pinning (keep the leaky embedding + final layers on trusted nodes).
- Per-request "trusted nodes only" option.
- Fault tolerance: node drops mid-generation are recovered, not just failed.
- Security pass on the rendezvous + transport.
- The privacy claim earns its word here, phase by phase, never overclaimed earlier.

## The honest risk register

- **WAN transport across arbitrary NAT is genuinely hard.** A funded team didn't nail it. We de-risk by owning + instrumenting the layer and proving it on LAN first.
- **Spec-decode acceptance rate over real links sets the real tok/s.** If the draft is weak or the domain is hard, fewer tokens accept and the number drops. Mitigate: good draft model, adaptive K, measure honestly.
- **Privacy vs the pillar.** The leak is real. Boundary pinning helps, full privacy is research. Do not sell what isn't true yet.
- **Scope.** This is multi-week serious engineering, not a weekend. The phasing means we learn whether it works in days (Phase 0), not at the end.

## What we need

- 2 GPU boxes for Phase 0-1 (we have Vast boxes up now, keep 2, drop the rest to stop the spend).
- A draft model: small, uncensored, target-compatible tokenizer.
- The c0mpute orchestrator as rendezvous + scheduler host (already running).
- A model to start with that needs exactly 2 GPUs for Phase 0 (pick during Phase 0 setup).
