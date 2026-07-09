# M2.5 Admission Spec — capability function, not an allowlist

The decision (leyten, 2026-07-08): a node is admitted to a swarm by a **GPU-model-independent capability
function**, never a vendor/model allowlist. This doc turns that choice into a *derived, implementable*
spec: the physics that sets the minimum, the numbers per role, and the honest verdict on what can
actually anchor a fast M2.5 ring. Numbers are derived from live 5090+4090 EU rings and
adversarially cross-checked against both the scatter receipt (13-15 tok/s) and the tight receipt (32).

## Why capability > allowlist (and what it costs)

An allowlist needs a maintainer deciding "5090 in, 4090 out" — a central gatekeeper, the exact thing a
permissionless network removes. A capability function is self-executing (the node proves what it can do,
the function decides), future-proof (admits hardware that doesn't exist yet), and **model-parameterized**
(the same function, fed a model's layer size + count, works for M2.5 today and any model tomorrow).

The cost the choice creates: you now need a **trustless capability probe** — a measurement a node can't
spoof (self-reported specs are worthless; a liar just makes a slow ring). That probe is the real
engineering the design choice buys decentralization with. It is not free, but it is right.

## The physics that sets the minimum

Single decode token: `tok/s = g / T`, `T = N·RTT_eff + C`.
- `g` = tokens accepted per ring traversal (EAGLE+n-gram spec-decode; measured 3.3-4.5).
- `N` = number of ring stages = **`ceil(62 / layers_per_node)`** — the token visits all 62 layers once,
  serially, no matter how the ring is cut, so **the weakest node's layer count sets the hop count**.
- `RTT_eff` = per-hop transport. Measured: **~30 ms wide EU scatter, ~19 ms tight regional**.
- `C` = total compute over 62 layers, partition-independent: ~46 ms all-5090, ~56 ms mixed.

The model reproduces both live receipts: 6-stage scatter @ g≈4 → predicted miss of 20 → measured 13-15 ✓;
5-stage tight @ g≈4.5 → predicted clear with margin → measured 32 ✓.

**Three co-binding constraints — VRAM alone is the wrong function:**
1. **VRAM → layers → hop FLOOR.** `layers = (VRAM − peak_overhead) / footprint`. Blackwell cutlass
   1.85 GB/layer (1.7 weight + 0.15 KV); marlin (Ada/Ampere) **4.25 GB/layer — a 2.3× penalty** that is
   fatal for a 140 GB model. Gate on the **load-time PEAK, not resident**: a 32 GB 5090 OOM'd at 15
   layers (22 GB resident) on the NVFP4 weight-swizzle peak (~+4.3 GB); 12 layers is the safe cap.
2. **RTT → actual speed.** Same cards, RTT 30→19 ms doubled throughput (13-15 → 32) with zero VRAM
   change. A 48 GB node at 80 ms RTT tanks a ring that a 32 GB node at 20 ms carries. RTT-to-assigned-
   neighbors + NAT-dialability are part of the capability vector, not an afterthought.
3. **Uplink → prefill TTFT.** Decode activation is trivial (~3 KB/token = 0.5 Mbps). But prefill at 16k
   is **50 MB/hop** (S×H×1B, fp8): 400 Mbps → 6 s TTFT over 6 hops; **15 Mbps residential → 160 s**.
   Uplink is a genuine second gate that VRAM-only admission misses; chunked prefill barely helps
   (compute is ~1 ms/layer, it can't hide a 50 MB transfer).

Compute is NOT a number — it's a **binary "has a graph-safe fast kernel."** No modern GPU is
compute-bound at its VRAM-limited layer count (5090 0.75 ms/layer, 4090 marlin+graph 1.6 ms/layer). The
gate only bites the fallback paths: no CUDA-graph (3-10× launch overhead), no native NVFP4/marlin path
(fp16 dequant 10-50 ms/layer), or CPU (seconds/layer).

## Minimum-spec table (derived)

| Role | min layers | min VRAM (by arch) | kernel | uplink | RTT |
|---|---|---|---|---|---|
| **Interactive anchor** (20-32 single-stream) | 13-16 scatter / 7-12 tight | **48 GB Blackwell** (~21 L); 32 GB 5090 (12 L) **tight-ring-only, marginal**; **24 GB marlin = NO** | native + CUDA-graph | ≥200 Mbps (16k TTFT) | ≤25 ms to neighbors |
| **Batched filler** (aggregate 20+, per-stream ~8) | 3-5 | ~16 GB Blackwell / **24 GB marlin (5 L) ✓** | native + graph | ≥100 Mbps | relaxed (long N amortizes) |
| **Verifier** (spot-check 1 block) | 1+ | 2-8 GB any GPU; CPU ok (slow) | any | modest | n/a |
| **Seeder** (weight propagation) | 0 | 0 — CPU / phone + disk | none | upload BW only | n/a |

Layers a card actually holds (peak-gated): 16 GB → 6 Blackwell / 3 marlin · 24 GB → 10 / **5** · 32 GB →
**12** / 7 · 48 GB → ~21 / ~10 · 80 GB → ~40 / ~17. The `12` on a 32 GB 5090 is the load-tested cap.

## The honest anchor verdict

- **A consumer 24 GB non-Blackwell card (4090/3090) can NEVER anchor fast single-stream M2.5.** 5 layers
  → 13 stages → needs g≥5.9 even at tight RTT; real g is 3.3-4.5. The 4.25 GB/layer marlin footprint is
  the killer. This is physics, not policy.
- **Even a 32 GB 5090 is a MARGINAL anchor** — 12 layers, N=6; it clears 20 only on a **tight ≤24 ms
  regional ring at g≥4** (regional low-RTT peering ≠ banned co-location). On true wide scatter it lands
  13-16. Caveat on the reason-math-32 receipt: a 5-stage 62-layer ring needs 12.4 layers/node, but 32 GB
  caps at 12 — that receipt was either 6-stage, or offloaded the embed/final layers off a stage; a naive
  "12 layers/5090 → N=5" admitter comes **2 layers short and OOMs**. So the reproducible single-stream
  20-32 claim is **tight-regional-ring-only**.
- **The comfortable fast-M2.5 anchor is a 48 GB fast-kernel card (N≤4)**, or a tight regional ring of
  32 GB 5090s. M2.5's 140 GB size makes interactive M2.5 **Blackwell/pro-anchored** — full stop.
- **This does NOT reject the long tail — it ROUTES it.** Weak/consumer hardware earns its keep exactly as
  the torrent thesis predicts: **batched-fill** (a 24 GB marlin card gives ~36 tok/s aggregate at B=4),
  **verification**, **seeding**, and **anchoring SMALLER models** (a 30-40 B NVFP4 fits meaningfully in 5
  layers). The capability function, parameterized per model, sends every card to where it adds value.
  That is the self-organizing multi-model network — the real heterogeneity play, not "make every card
  serve M2.5."

## The bar is a ROLE TAG, not a binary gate

The founder's "20 tok/s" (and "~30 across usage cases") is the **interactive-anchor** tag, not a
network-wide admit/reject. Batched aggregate is the network's economic product and admits ~4-10× more
hardware. A single-stream gate would reject the 24 GB marlin card that delivers 36 tok/s aggregate — the
opposite of the torrent thesis. So admission emits a **role**, and the market prices each role.

## The admission function (implementable)

On join, the probe MEASURES (never self-report):
```
cap = {
  peak_vram_mb,        # a 1-block load probe: measures the swizzle/context PEAK, gives arch + footprint
  has_fast_kernel,     # did the block run native NVFP4/marlin under a CUDA graph? (binary)
  layer_ms,            # timed decode forward on the probe block
  uplink_mbps,         # measured upload
  rtt_to_pool,         # RTT to the candidate ring's members (not the node in isolation)
  nat_dialable,        # AutoNAT / hole-punch reachability
}
layers = (peak_vram_mb - ctx_overhead) / footprint(arch)     # peak-gated, not resident
N_single  = ceil(model_layers / layers)
N_batched = ceil(model_layers / layers)   # same coverage; the BAR is what changes with B
role =
  RING_INTERACTIVE if has_fast_kernel and N_single ≤ N_max(rtt_to_pool, g) and uplink ≥ 200 and nat_dialable
  else RING_BATCHED if has_fast_kernel and N_batched ≤ N_max_batched(rtt, g, B) and uplink ≥ 100 and nat_dialable
  else VERIFIER     if can_recompute_one_block
  else SEEDER       if uplink ≥ min and disk ≥ range
  else REJECT       # genuinely offline / no bandwidth
```
`N_max(rtt, g) = (g/bar − C) / rtt` — the hop budget, evaluated against the node's *actual* RTT
neighborhood, so a fast card in a bad region is correctly relegated, not admitted to stall a ring.

## Build implications (what the probe must do that VRAM-only misses)

1. **Peak-VRAM load probe**, not a free-VRAM read — else admit-then-OOM at the swizzle peak.
2. **RTT-to-assigned-neighbors + NAT-dialability** in the vector — a 48 GB node at 80 ms or behind CGNAT
   passes VRAM and tanks/breaks the ring.
3. **Uplink probe** — the prefill/long-context gate; residential-cable nodes pass VRAM and fail TTFT.
4. **Binary fast-kernel check** — native NVFP4/marlin under a graph; the fp16-dequant/CPU fallbacks are
   too slow to anchor.
5. **Emit a ROLE**, feeding placement (`select_ring`, which already sizes by measured VRAM+layer_ms) and
   the market (each role priced). Per the boundary law: shard owns the probe + physics; c0mpute owns the
   role decision. See `HETERO_DEVICES.md` (physics/tier table) and MARKET_DECENTRALIZATION.md (pricing).
