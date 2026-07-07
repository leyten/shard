# Real-ring pass — the permissionless loop end-to-end on live GPUs (2026-07-07)

The loop ran on a real scattered ring: **place → verified pull → auto-form → serve → settle → pay per shard**,
with the placement and settlement decisions driven by shard's own seams and the receipts verified on real
hardware. No co-location — five EU consumer 5090s on distinct subnets.

## Ring
`select_ring` measured the pool and placed a head-first ring on the most-central node:

| stage | node | region | layers | role |
|---|---|---|---|---|
| 0 | 44153731 | Norway | [0:10] | coordinator / head |
| 1 | 44153714 | Norway | [10:23] | stage |
| 2 | 44153719 | Latvia | [23:36] | stage |
| 3 | 44153724 | Germany | [36:49] | stage |
| 4 | 44157831 | Denmark | [49:62] | tail |

## Verified pull (the #46 fix, at full-ring scale)
Every stage pulled ONLY its layer range from the **signed content-addressed manifest**
(`nvidia/MiniMax-M2.5-NVFP4`, 62 layers, 29 weight shards, 139.9 GB total), re-hashing every byte via
`fetch_block_range`. All pulls landed clean (24–33 GB per box) with no overshoot — the resume path fixed in #48
held on every one, including two boxes that had to be re-bootstrapped/replaced mid-pass.

## Serve
All five stages warmed and formed the ring (`s0..s4 WARM`, sidecars up, coordinator-return connected). One job
served over the ring — **160 tokens, 3.84 tok/s** (novel-reasoning prompt, n-gram-only drafter g=1.0; transport
53% / stage-span 47% of a 165 s traversal; the ring carried a high-RTT hop). Perf is not the point of this pass;
the point is the loop closing.

## Settle + pay (the loop's own seam, on real receipts)
The coordinator swept the ring for **5 signed per-stage receipts**. Exported and run through
`python3 -m shard.verify`:

- every signature **VALID**;
- the activation chain **holds exactly** across all five scattered stages — `out_root[i] == in_root[i+1]`
  (`d5690c4d… → f9d1300f… → 27064d44… → 9b7c42aa… → 27ea75cb… → ab47fef0…`) on the lossless wire;
- coverage tiles `[0:62)` with no gap or overlap; per-job nonce present (anti-replay);
- **settle ok=True**, and the per-shard-per-token split fired:

```
160 tokens across 5 shards (by layers):
  [ 0:10] 10L  qz5kFV3IzMpB..  -> 26
  [10:23] 13L  kvNeNNBG+dSA..  -> 34
  [23:36] 13L  +FjeVMgv1Ldk..  -> 34
  [36:49] 13L  VqjnX34aVW/3..  -> 33
  [49:62] 13L  ZJDww1hZ92zS..  -> 33
  Σ = 160  (== job's tokens)
```

Signed receipt set: `m25-realring-loop-receipts-20260707.json`.

## Ops notes (feed the ring watcher)
- One box (Hungary) **never bootstrapped** — a transient scp/ssh drop at the bootstrap step left it alive-but-idle
  while the puller polled a `boot.log` that never appeared. Re-bootstrapped by hand. A watcher must detect
  "no boot.log after N s" and re-bootstrap/swap.
- One box (Hungary tail) was a **GPU-driver dud** (`CUDA Error 803`) that passed the VRAM health check but failed
  `torch.cuda.init()`. Health checks must probe CUDA init, not just free VRAM. Replaced with a fresh Denmark box.
- The publisher key here is **ephemeral/test** — the durable manifest-signing identity is a c0mpute-catalog call.
- Ring torn down (instances-v1 == 0 verified); results banked.
