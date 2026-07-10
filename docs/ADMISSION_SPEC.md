# M2.5 Admission Spec — capability function, not an allowlist

> **LIVING SPEC — v0 (2026-07-09). The FRAMEWORK is settled; the NUMBERS are not.** What's decided is the
> *shape*: admission is a GPU-model-independent capability function over `{peak-VRAM, fast-kernel, layer_ms,
> RTT-to-neighbors, uplink, dialable} → role`, and the physics `tok/s = g/(N·RTT + C)` with
> `N = 62/layers-per-node`. Every THRESHOLD below (uplink ≥ 200 Mbps, the 12-layer/32 GB cap, RTT ≤ 25 ms,
> the layer-count minimums, the g assumptions) is a **derived estimate to be VALIDATED and REVISED by
> measurement** — the probe + the batched/live tests exist precisely to correct them. If testing shows
> uplink must be stricter, or VRAM can be looser, or the g/RTT assumptions were off, **change the numbers.**
> Treat this doc as a hypothesis to falsify, not a fixed law. Update it when a run says otherwise.

The decision (leyten, 2026-07-08): a node is admitted to a swarm by a **GPU-model-independent capability
function**, never a vendor/model allowlist. This doc turns that choice into a *derived, implementable*
spec: the physics that sets the minimum, the numbers per role, and the honest verdict on what can
actually anchor a fast M2.5 ring. Numbers are derived from live 5090+4090 EU rings and
adversarially cross-checked against both the scatter receipt (13-15 tok/s) and the tight receipt (32) —
they REPRODUCE both, which is why the framework is trusted; the exact thresholds still need live validation.

## Why capability > allowlist (and what it costs)

An allowlist needs a maintainer deciding "5090 in, 4090 out" — a central gatekeeper, the exact thing a
permissionless network removes. A capability function is self-executing (the node proves what it can do,
the function decides), future-proof (admits hardware that doesn't exist yet), and **model-parameterized**
(the same function, fed a model's layer size + count, works for M2.5 today and any model tomorrow).

The cost the choice creates: you now need a **trustless capability probe** — a measurement a node can't
spoof (self-reported specs are worthless; a liar just makes a slow ring). That probe is the real
engineering the design choice buys decentralization with. It is not free, but it is right.

**IMPLEMENTED: `shard/probe.py`** — the pure role function (`python3 -m shard.probe`, `{cap, model?,
spec?}` JSON in → verdict out, the seam c0mpute drives), the `--measure` GPU one-block probe, the
`--serve`/`--net-only` network probe (receiver-timed uplink, nonce dial-back, connect-time RTT in
`topology`'s units). Honest v0 trust semantics: **trust-then-punish, not can't-lie** — the network half
is measured by the other end (probe peers must be ASSIGNED by the control plane, never candidate-chosen:
one colluding fast receiver would inflate max-over-peers uplink); the GPU half runs the real block but a
lie is caught at placement/receipts/challenge, i.e. the ring eats one bad formation before reputation
ejects. Signed probe transcripts + pool-run GPU spot-probes are the hardening that closes that window.

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
1. **VRAM → layers → hop FLOOR.** `layers = (VRAM − reserve − load_transient) / (footprint + kv)`.
   **MEASURED 2026-07-09 (capability probe, one real layer + warm-stage reads):** Blackwell cutlass
   **2.33 GB/layer full-layer resident** (NVFP4 experts + bf16 attn + norms — the earlier 1.7 GB was
   experts-only and under-modeled by ~35%); marlin (Ada/Ampere) **4.06 GB/layer — still a ~1.75×
   penalty** that is fatal for a 140 GB model. The load TRANSIENT is arch-dependent and the probe
   measures it: **cutlass ~72 MB (tiny), marlin ~4.8 GB (the dequant/repack buffer — bigger than the
   old "swizzle peak" guess).** The original "+4.3 GB swizzle" mechanism story was WRONG: the 15-layer
   OOM was plain footprint arithmetic (15 × 2.33 ≈ 35 GB > 32). Direct warm-stage evidence: a
   13-layer stage serves at **31.5/32.6 GiB — one allocation from OOM**; a 12-layer tail sits at
   30.4 GiB. **12 layers is the 32 GB plan target** ((32768−1500)/2480 = 12.6), by measurement.
2. **RTT → actual speed.** Same cards, RTT 30→19 ms doubled throughput (13-15 → 32) with zero VRAM
   change. A 48 GB node at 80 ms RTT tanks a ring that a 32 GB node at 20 ms carries. RTT-to-assigned-
   neighbors + NAT-dialability are part of the capability vector, not an afterthought.
3. **Uplink → prefill TTFT.** Decode activation is trivial (~3 KB/token = 0.5 Mbps). But prefill at 16k
   is **50 MB/hop** (S×H×1B, fp8): 400 Mbps → 6 s TTFT over 6 hops; **15 Mbps residential → 160 s**.
   Uplink is a genuine second gate that VRAM-only admission misses; chunked prefill barely helps
   (compute is ~1 ms/layer, it can't hide a 50 MB transfer). Thresholds are RECEIVER-TIMED
   single-TCP-stream numbers — measured 2026-07-09: a vast box LISTED at 1316 Mbps up delivered
   **210 Mbps** receiver-timed cross-WAN. Listings and speed-test numbers do not qualify a node.

Compute is NOT a number — it's a **binary "has a graph-safe fast kernel."** No modern GPU is
compute-bound at its VRAM-limited layer count (5090 0.75 ms/layer, 4090 marlin+graph 1.6 ms/layer). The
gate only bites the fallback paths: no CUDA-graph (3-10× launch overhead), no native NVFP4/marlin path
(fp16 dequant 10-50 ms/layer), or CPU (seconds/layer).

## Minimum-spec table (derived)

| Role | min layers | min VRAM (by arch) | kernel | uplink | RTT |
|---|---|---|---|---|---|
| **Interactive anchor** (20-32 single-stream) | 13-16 scatter / 7-12 tight | **48 GB Blackwell** (~21 L); 32 GB 5090 (12 L) **tight-ring-only, marginal**; **24 GB marlin = NO** | native + CUDA-graph | ≥200 Mbps (16k TTFT) | ≤25 ms to neighbors |
| **Batched filler** (aggregate 20+, per-stream ~8) | 3-5 | ~16 GB Blackwell / **24 GB marlin (4 L, measured)** | native + graph | ≥100 Mbps | relaxed but POOL-relative (measured: 72 ms kills even B=4) |
| **Verifier** (spot-check 1 block) | 1+ | 2-8 GB any GPU; CPU ok (slow) | any | modest | n/a |
| **Seeder** (weight propagation) | 0 | 0 — CPU / phone + disk | none | upload BW only | n/a |

Layers a card actually holds (measured footprint 2.33/4.06 GB + kv 0.15, reserve 1.5, arch transient):
16 GB → **5** Blackwell / 2 marlin · 24 GB → **9** / **4** · 32 GB → **12** / 6 · 48 GB → ~19 / ~10 ·
80 GB → ~32 / ~17 (the ≥48 GB column is density-extrapolated, unproven at size — probe on join decides).

**12-vs-13 — RESOLVED BY MEASUREMENT (2026-07-09).** The v0 tension (plan profile 13 vs spec 12) is
settled: the full-layer footprint is 2.33 GB, so 12 is arithmetic, and the live reads prove it — a
13-layer middle warmed but served at **31.5/32.6 GiB (brim)**, and a 13-layer TAIL **OOM'd loading the
1.15 GiB lm_head** (middles ≠ tail: `plan.py` now models `tail_reserve_mb=1400`). `plan.py` and
`probe.ADMISSION_MODEL_V0` both carry the measured numbers (layer_vram 2330, cap 12). The cap is
proven on 32 GB cards only, so the probe scales the proven DENSITY to card size (48 GB → 18 → N=4)
instead of flat-clamping — a flat cap made 48 GB indistinguishable from 32 GB.

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
  the torrent thesis predicts: **batched-fill**, **verification**, **seeding**, and **anchoring SMALLER
  models** (a 30-40 B NVFP4 fits meaningfully in 4 layers). The capability function, parameterized per
  model, sends every card to where it adds value. That is the self-organizing multi-model network — the
  real heterogeneity play, not "make every card serve M2.5."
- **Batched viability is g-DEPENDENT and pool-relative (measured 2026-07-09, REVISED 2026-07-10 with
  the full drafting stack in the batched path).** The 12-arm live sweep (6×5090 ring, hybrid
  n-gram→EAGLE per stream, receipt batched-sweep-eagle-20260710): **content-mix g = 3.6** (band: prose
  2.2, qa 2.4, code 2.9, reasoning 3.7, tools-JSON 3.7, summarize/verbatim 5.8) — the 16× lift over the
  n-gram-only floor (g 0.22 at equal transport) proves drafting quality transfers to batch. The
  DATA-ISOLATION gate passes and every batched round is receipt-attested (72/72 sigs valid).
  **Operative g_batched = 2.5**: the measured 3.6 discounted for the CURRENT engine's
  drafter-serialization tax (B EAGLE chains run serially coordinator-side, ~0.25 s/stream/round, and
  EAGLE pins the in-flight window to 1 — B-scaling measured 3.3× at B=8, not 8×). On this engine build
  the 20-agg bar is DRAFTING-bound, not WAN-bound: the unlock is engine work (batch the drafter
  forward, graph the drafter chain, graph the batched stage path), not admission policy. Revise
  g_batched up as those land. Pool-relativity stands: the live probe verdict for a 24 GB card 72 ms
  from its pool was VERIFIER, correctly.

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
6. **Feed measured values back to REVISE this spec (the numbers are v0).** The probe produces real
   per-node `layer_ms`/uplink/RTT and every served ring produces real tok/s vs the predicted `g/T`. When
   live data diverges from a threshold here — a role that clears the bar the spec said it wouldn't, or an
   admitted node that stalls a ring — that is a signal to CHANGE the number, not to trust the doc. The
   admission spec is a hypothesis the network's own telemetry falsifies and tunes; it is the self-optimizer's
   input, not a constant. Keep it current the same way as the living-state doc.
