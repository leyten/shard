# shard roadmap

phased so the riskiest thing is proven first and cheaply. each phase has one goal and a hard pass/fail.

## phase 0 — prove the transport (target: 1-2 days)

the single hardest thing: reliably serve tokens through a multi-stage split. do it on easy mode first.

- 2 nodes, **same lan** (or two boxes in the same datacenter), low latency.
- split one model that genuinely needs 2 gpus across them with our own quic transport moving activations.
- no speculative decoding yet. plain token-by-token is fine, it'll be slow, that's expected.
- **pass = a coherent completion comes out, reliably, 20 times in a row, with a measured tok/s.**

**✓ PASSED (2026-06-15).** 2-node split with per-node kv-cache. proven on Qwen2.5-14B (29.5GB bf16, OOMs one 24GB card) split 24/24 across two 4090s: 20/20 clean completions, ~17 tok/s decode, via our own tcp transport. code in `phase0/` (`node_kv.py` = the split node, `bench.py` = the 20x reliability harness). caveat: nodes are on a low-latency same-host link, not wan yet.

this was the thing that blocked us all day. it's the momentum milestone.

## phase 1 — make the transport survive wan (target: 3-5 days)

- same 2 nodes, now on **different networks behind nat**.
- add hole-punching via a rendezvous (c0mpute orchestrator), relay fallback for symmetric nat.
- add fp8/int8 activation quantization to cut uplink bandwidth.
- add edge supervision: kill a node mid-stream, confirm the pipeline detects it and the request fails cleanly (not a hang).
- **pass = reliable wan serving + an honest wan tok/s number** (will be latency-bound and slow, that's the point, it sets the baseline spec-decode has to beat).

## phase 2 — speculative decoding (target: ~1 week)

the payoff. add the draft-verify loop.

- small draft model on the entry node, propose K, verify K across the swarm in one traversal.
- adaptive K based on measured latency + live acceptance rate.
- **pass = land in the paper's regime: meaningfully more tok/s than phase 1 on the same links, in the ~8-9 tok/s ballpark at 80ms for a small target.** this is the proof the whole approach is real.

## phase 3 — permissionless swarm + c0mpute (target: 1-2 weeks)

- one-line installer, auto-installs deps, plug and play.
- `cwt_` worker auth, per-token usdc payout for a node's contribution.
- dynamic layer allocation across **4+ heterogeneous** consumer gpus for a real **100B+** target.
- scheduler handles joins/leaves and rebuilds the pipeline live.
- **pass = a stranger runs one command, their gpu joins, takes a layer block, and earns for tokens it helped produce.**

## phase 4 — privacy + hardening (ongoing)

- boundary-layer pinning (keep the leaky embedding + final layers on trusted nodes).
- per-request "trusted nodes only" option.
- fault tolerance: node drops mid-generation are recovered, not just failed.
- security pass on the rendezvous + transport.
- the privacy claim earns its word here, phase by phase, never overclaimed earlier.

## the honest risk register

- **wan transport across arbitrary nat is genuinely hard.** a funded team didn't nail it. we de-risk by owning + instrumenting the layer and proving it on lan first.
- **spec-decode acceptance rate over real links sets the real tok/s.** if the draft is weak or the domain is hard, fewer tokens accept and the number drops. mitigate: good draft model, adaptive K, measure honestly.
- **privacy vs the pillar.** the leak is real. boundary pinning helps, full privacy is research. do not sell what isn't true yet.
- **scope.** this is multi-week serious engineering, not a weekend. the phasing means we learn whether it works in days (phase 0), not at the end.

## what we need

- 2 gpu boxes for phase 0-1 (we have vast boxes up now, keep 2, drop the rest to stop the spend).
- a draft model: small, uncensored, target-compatible tokenizer.
- the c0mpute orchestrator as rendezvous + scheduler host (already running).
- a model to start with that needs exactly 2 gpus for phase 0 (pick during phase 0 setup).
