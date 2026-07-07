# M2.5 warm-ring validation — 2026-07-07

Production validation of this session's hardening work (PRs #34/#36/#38) on a **live 5-stage scattered
EU ring** over libp2p. Provisioned autonomously via `rent_pool → ring_up`; all commits' changes ran on
the boxes (launcher `m25_scatter_pipe` pushes the current master engine code to every stage).

**Ring:** HU(head/coord)→HU→DK→NO→UK(tail), 62 layers split 10/13/13/13/13.
IIDs 44121542/44121561/44121556/44121563/44121544 (+ Bulgaria 44121548 spare). Loop RTT tight-EU
(the two HU boxes 4ms apart, NO↔DK 17ms). Head + tail are slow-CPU boxes (cpu_factor 3.2 / 2.9).
**Wire: LOSSLESS (M25_FP8_WIRE off) so the receipt chain-check is ON.**

---

## ✅ VALIDATION 1 — receipt freshness + chain binding (PR #36), LIVE

A reasoning job (EAGLE, `SHARD_RECEIPTS=1`, 96 tokens) on the head as sole coordinator. The five signed
per-stage receipts chain **exactly** — each block's `out_root` equals the next block's `in_root`, on 5
real scattered stages:

```
stage 0   layers[0:10]   in 4c8d2cdf96b9  ->  out 19e6fd449e56   pub VGzskepI7LAV  sig VALID
stage 1   layers[10:23]  in 19e6fd449e56  ->  out d92ceb5d4ea0   pub cZBkW53kjPiZ  sig VALID   (in == s0.out)
stage 2   layers[23:36]  in d92ceb5d4ea0  ->  out a1f276ece315   pub opoVPKYMHdGv  sig VALID   (in == s1.out)
stage 3   layers[36:49]  in a1f276ece315  ->  out 712903de8ace   pub hQrBnv/chAWg  sig VALID   (in == s2.out)
stage tail layers[49:62] in 712903de8ace  ->  out a58972551569   pub tmrxb638Mjcw  sig VALID   (in == s3.out)
```

**`PROVE verdict: ALL receipts valid + full layer coverage`** — signatures verify, blocks tile [0:62]
gaplessly, the per-job nonce matches on every receipt, and the chain holds (lossless wire → `check_chain`
on). This is the moat working end-to-end in production, not just CPU unit tests: a node cannot be paid
without its own signed receipt, cannot fabricate a neighbour's, cannot replay an old job's receipt (stale
nonce), and cannot splice fabricated roots (the chain would break).

**Perf (this slow ring, chain mode, lossless):** 96 tok @ **5.25 tok/s**, g=3.65, mean_accept 2.65/8,
prefill 5.55s. traversal = transport 77% + stage-compute 23%; per-stage span ms s0=26 s1=37 s2=11 s3=10
s4(tail)=74 — the two slow-CPU HU/UK boxes + lossless (2×) payload dominate, which is exactly the regime
CUDA-graph-aux targets. Coherent reasoning output.

## OPS LESSON — FWD_RET/FWD_RING tunnels are slow to establish after warm
The head sidecar's libp2p forwards to s1 and the tail **refused the initial dial** and only reached
`CONN DIRECT` ~3-5 min later (s1 at t+3m, tail at t+5m). The FIRST coordinator launched right after warm
WEDGES until they connect. **Fix in practice:** give the first post-warm coord a long timeout / run it in
the background; don't kill it early. (This is the documented FWD_RET flakiness; a real fix — dial the tail
directly instead of via the head sidecar — remains queued.)

## ✅ VALIDATION 3 — ring churn survival (PR #34/#26), LIVE

Started coordinator A (EAGLE reasoning, max-new 240), let it prefill + decode, then `kill -9` it
**mid-decode** (in-flight frames on the ring). With NO re-warm, started a NEW coordinator B on the same
ring:

```
=== KILLED coord A mid-decode; NO re-warm; starting NEW coord B on the same ring ===
[coord] pipelined (K=8 depth=4 ngram=3) -> head 127.0.0.1:29610, ret 127.0.0.1:29612
[coord] 66tok  8.47 tok/s  g=3.61  mean_accept=2.61/8  prefill=1.74s  depth=4
```

**Coord B completed a full job.** The stages survived the coordinator's death, the tail kept its warm KV +
return channel, and a fresh coordinator re-armed and drove the ring to completion — no relaunch, no re-warm.
This is the wedge/churn fix (return-channel-across-churn #26 + heartbeat/wedge #34) working in production:
the "re-warm per coordinator" tax is gone on a real scattered ring. (B's prefill 1.74s / 8.47 tok/s vs
val1's 5.55s / 5.25 — B ran on already-established tunnels, so faster.)

## VALIDATION 2 — graph-aux (~24 tok/s rep2): NOT re-run this session
graph-aux (PR #25) is already **mechanism-verified** (`research/graph_aux_check.py`: graph ≡ eager-manual;
GPU-proven), and this particular ring is slow-CPU + lossless (chain 5.25), so a graph A/B here would show
the +74% delta but not cleanly re-pin the ~24 headline (that needs a faster window + fp8). Re-running it
requires a full re-warm with `M25_STATIC_KV=1` (graph's prereq) — deferred as low-ROI vs the two
never-before-live-tested validations above. The ~24 figure stands as mechanism-verified + rep1-measured.

---

## Verdict
This session's two biggest new pieces — the **receipt freshness+chain moat (#36)** and **ring churn
survival (#34/#26)** — are now **production-validated on a live 5-stage scattered EU ring**, not just CPU
unit tests. Wire (#35) + gateway (#37) + KV guard (#38) rode along in the same warm run (lossless wire
carried every frame; the coordinate_pipe path with all merged hardening drove both jobs cleanly). Ring
torn down after the battery; spend ~$3-4 of the vast balance.

