# Heterogeneous devices — which hardware can join a swarm

The north star is torrent-like: **any device joins a swarm**, not an NVIDIA allow-list. This doc
records what it actually takes for a non-5090 card — and, further out, an Apple Silicon Mac or an
AMD GPU — to serve a layer shard in the scattered pipeline ring, plus a device tier table against
the usable-speed bar. The planner already does the easy half (`assign_layers`/`select_ring` size
each node's block by VRAM + measured per-layer time, so a slower card just holds fewer layers); the
work is (a) running the shard on a non-Blackwell backend and (b) admission tiers + a compute probe.

## The gating question, answered: NVFP4 portability off Blackwell

The checkpoint is `nvidia/MiniMax-M2.5-NVFP4` (4-bit experts, Blackwell-native sm_120). The MoE
backend is selectable — `M25_MOE_BACKEND = cutlass | marlin | emulation` (`phase0/m25_stage.py`),
now defaulting to **`auto`** (cutlass on sm_120+, marlin below; a stage's arch is a node fact, not a
ring-wide env). Probed on a live rented card:

| GPU | arch | cutlass | marlin | emulation | MoE VRAM/layer | decode MoE latency |
|---|---|---|---|---|---|---|
| RTX 5090 | sm_120 | ✅ native FP4 (fast path) | ✅ | ✅ (Triton, batch-invariant) | ~1.7 GB (4-bit) | ~0.65 ms (graph) |
| **RTX PRO 6000 WS (96 GB)** | **sm_120** | ✅ native (probe 2026-07-12: full-layer footprint **2329.5 MB**, transient 72 MB, graph **bit-exact**, layer_ms 0.24 graph / 0.75 eager — Max-Q edition) | ✅ | ✅ | 2.33 GB full-layer (== 5090; density-scaled cap **35 layers**) | 0.24 ms (graph) |
| **RTX 4090** | **sm_89 (Ada)** | ❌ *"kernel does not support current device"* | ✅ **RUNS** | ❌ *illegal memory access* | **4.08 GB** (dequant, ~2.4×) | **0.35 ms/tok** (T=1), 0.48 (T=8) |
| RTX 3090 | sm_86 (Ampere) | ❌ (expected) | ✅ (probe pending) | ❌ (expected) | ~4 GB (est) | est ~0.5-0.9 ms |

**Same-model-name ≠ same verdict (2026-07-12, the reason admission MEASURES):** two rented RTX PRO
6000 WS boxes probed the same footprint but opposite fast-kernel verdicts — one graph-replayed
bit-exact (cosine 1.0), the other's replay produced garbage (cosine 0.0, twice, deterministic; runs
eager at 0.96 ms/layer). The second box was relegated off-ring by its own measured verdict — a
graph-armed stage on it would corrupt silently behind valid-looking transport receipts. A per-model
allowlist admits both; the capability function admits one and routes the other.

**Verdict: a 4090 serves the M2.5 NVFP4 MoE today via marlin** — same signed checkpoint, no
re-quant, dequant-in-kernel to fp8/bf16. cutlass is Blackwell-only (refuses pre-sm_120); emulation
is a sm_120 Triton path (illegal-memory on Ada). The cost is VRAM: 4-bit → ~fp8 inflates the expert
footprint **~2.4×** (4.08 GB/layer measured vs 1.7 on cutlass), so a 24 GB 4090 holds fewer layers
than a 32 GB 5090 — exactly what the VRAM-sized planner already accounts for.

**Numeric compatibility:** a marlin stage's output differs from a cutlass stage's at the kernel-drift
level (two dequant kernels of the same 4-bit weights), which is the ULP class the spot-check was
built for (`shard/challenge.py`, cos ≥ 0.99, rel < 0.05). `research/hetero_moe_xcheck.py` dumps a
deterministic CPU-seeded MoE forward per backend for an offline cosine check; the marlin dump is
banked, the cutlass reference is a one-box run. (Cross-*quant* — a Mac's MLX-4bit vs NVFP4 — is a
bigger drift and needs the format-matched auditing below; cross-*kernel* on the same NVFP4 weights
is not.)

## Device tier table (against the ≥20 tok/s bar)

> This table is **illustrative output of the capability admission function** (below) evaluated on
> common cards today — NOT an allowlist. Admission is by measured VRAM/compute/bandwidth, not model name.

Usable-speed frame: `tok/s = g / T_traversal`; a stage holding K layers adds `K × layer_ms` to the
ring, and the ring is only as fast as its slowest stage + WAN hop (~15-40 ms). Single-stream WITH
graph-aux (`M25_CUDA_GRAPH` + `M25_STATIC_KV`, PR #25) does **~24 decode-weighted / 30-32 reasoning-heavy**
on a good EU ring; draftable-verbatim and batched-aggregate go higher still. So the 20 tok/s bar is
comfortably above what a heterogeneous ring must protect — heterogeneity must not be what drops a
ring below it, and every perf ring MUST launch with graph-aux on (a no-graph run under-measures ~2×). Layers-held ≈ (VRAM − reserve) / (per-layer weights + KV). NEVER
co-locate to manufacture the number — every verdict below is for scattered WAN placement.

| Device | BW GB/s | Mem GB | ~layers @ arch footprint | est layer_ms | Verdict |
|---|---|---|---|---|---|
| **RTX PRO 6000 (WS/S)** | 1792 | 96 | **35** (2.33 GB, density-scaled cap; MEASURED 2026-07-12) | **0.24 (measured, graph)** | **ring ANCHOR — proven live**: held 31 L + coordinator in a 4-hop hetero ring the c0mpute loop placed |
| RTX 5090 | 1792 | 32 | ~13 (1.7 GB) | 0.65-1.5 | **ring — proven** |
| RTX 4090 | 1008 | 24 | ~4-5 (4.08 GB marlin) | 1.2-2.5 | **ring** — fewer layers, marlin |
| RTX 3090 | 936 | 24 | ~4-5 | 1.3-2.9 | **ring** (probe pending) |
| 4070 Ti (S) | 504-672 | 12-16 | ~2-3 | 2-4 | ring-marginal (small block) |
| RTX 3060 | 360 | 12 | ~2 | 3-7 | edge — batched regime only / prefer off-ring |
| M3 Ultra | 819 | 96-512 | up to **62** (MLX 4-bit ~2.1 GB) | **0.30 (measured)** | **ring + the full-replica auditor node** |
| M2 Ultra | 800 | 64-192 | 25-62 | ~0.3 | **ring** (MLX build needed) |
| M4 Max | 546 | 36-128 | ~14-40 | 0.45-0.8 | **ring** (watch uplink/prefill) |
| M4 Pro | 273 | 24-64 | ~7-20 | 0.9-1.9 | ring-marginal — residential uplink is the gate |
| base M-series | 100-150 | 16-32 | ~4-9 | 2-4.5 | off-ring: seeder / light verifier |
| AMD 7900 XTX | 960 | 24 | ~9 | 0.45-0.65 (Vulkan) | **ring-worthy silicon** — gated on a llama.cpp backend |
| CPU box (DDR5) | ~90 | 64-192 | many (host RAM) | ~2-6 (unproven) | off-ring: **seeder** (`-seed`), torch-free sketch judge, coordinator/gateway |

Key inversion: a big Mac (0.30 ms/layer) is a *better* ring stage than a mid NVIDIA card, and the
M3 Ultra 512 GB is the only consumer device that holds the whole 140 GB model — the natural
format-matched spot-check auditor and instant-heal standby. A CPU box can't decode fast but is a
first-class **seeder** and a torch-free challenge judge (`shard/challenge.py` already runs on CPU).

## Build list (ranked by effort)

1. **[S] Per-node compute probe at admission** — measure `layer_ms` + `up_mbps` + free VRAM
   empirically (a few forward passes) and feed `select_ring` (it already consumes them). Needed for
   NVIDIA heterogeneity too; the driver-API CUDA probe (`scratchpad/health_probe.py`) is the
   liveness half.
2. **[S] ✅ SHIPPED — per-node backend + per-node `layer_vram_mb`** — `M25_MOE_BACKEND=auto` picks
   cutlass/marlin per arch; `select_ring`'s `layer_vram_mb` now accepts a per-node dict (scalar path
   byte-identical, goldens green), and `ring_up` detects each node's GPU name → per-arch footprint
   (5090 cutlass 1.7 GB, 4090/3090 marlin ~4.1 GB) + a marlin compute penalty, so a marlin card is
   placed as a thin stage automatically.
3. **[S] CPU-side fp8 wire unpack fallback** — for Mac/AMD safety on the fp8 codec path.
4. **[M] `MlxRuntime`** — a `ModelRuntime` (`shard/node.py`) over `mlx_lm.models.minimax`: load only
   layers [lo,hi) from the MLX-4bit artifact, drive `layers[i](h, mask, cache)`, crop KV to
   `start_pos`, bf16 at the boundary. The model file, the conversion (`mlx-community/MiniMax-M2.5-4bit`),
   and native bf16 all already exist — days, not weeks.
5. **[M] Manifest v2: per-format shard sets** under one model_id (nvfp4 / mlx4 / gguf-Q4_K), each
   file content-addressed exactly as now; the publisher converts + signs, anyone re-derives the
   hashes. Stage receipts name the artifact CID they computed with.
6. **[M] Format-matched spot-check** — the auditor recomputes a challenged block with the *same
   format* the node attested to (CPU dequant for GGUF/MLX), so cross-quant drift falls back to the
   kernel class and the 0.99 threshold holds. One cheap pre-experiment: measure real cross-quant
   cosine at 13-layer granularity before building.
7. **[L] llama.cpp-embedded runtime** — unlocks AMD + CPU + pre-Ampere, but llama.cpp has no
   hidden-states-in→layers→hidden-states-out API; a custom ggml graph over the loaded GGUF layers.
   After the Mac path proves mixed-quant rings live.

## Admission is a CAPABILITY FUNCTION, not an allowlist (leyten, 2026-07-08)

> Full derived spec (min numbers per role, the physics, the honest anchor verdict, the probe
> the function needs): **`docs/ADMISSION_SPEC.md`**. Summary below.

**The table above is illustrative OUTPUT, not the mechanism.** Admission is NOT "5090 yes, 3060 no" —
it is a GPU-model-independent function of *measured* capability, evaluated when a node joins. That is
the most decentralized form of joining: anyone brings any hardware, and the function judges it. A
card is never on or off an allowlist by name; it is admitted or rejected by what it can measurably do.

**Collected on join** (measured, never self-reported — a liar just makes a slow ring and gets
relegated by the receipts): free VRAM, compute speed (`layer_ms` from a probe forward — captures the
CPU kernel-launch factor too), uplink Mbps, region/RTT to existing pools.

**The function** (`allow-ring` / `relegate-off-ring` / `reject`):
1. **VRAM feasibility** — can it hold ≥1 layer at its per-arch footprint? (else seeder/verifier only.)
2. **Usable-speed + hop viability** — would including this node in a *coverable* swarm keep that swarm
   ≥ the bar (20 tok/s), or does accommodating its minimum contribution force too many hops? A node
   that can hold only ~2 layers forces a 62-layer model into ~31 stages ≈ 31 WAN hops — far over the
   traversal budget — so it is rejected *for the ring* even if its per-layer compute is fine. The
   criterion is a property of the **swarm shape the node forces (coverage-per-hop)**, not the node's
   speed alone. This is why a "bad" card is excluded: not because of its name, but because the minimal
   swarm that could use it isn't viable.
3. **Output**: ring-worthy / off-ring (seeder / verifier / coordinator / standby — a wage via the
   market, see MARKET_DECENTRALIZATION.md) / reject.

Boundary: shard owns the **probe + physics** (`select_ring` already sizes each node by measured
VRAM + `layer_ms`; the per-node VRAM footprint shipped this session is the placement half). c0mpute
owns the **admission decision** (the same measured inputs run through allow/relegate/reject *before*
placement). The compute/uplink probe (build item 1) is the shared front door; see
`admission-is-capability-function-not-allowlist` in memory.

## Prior art (why nobody has shipped a *verified* heterogeneous ring)

exo (MLX + tinygrad) hit un-root-caused mixed-engine drift and **retreated to homogeneous MLX** in
1.0 — because it had no per-stage verification to catch a misbehaving backend. Petals shipped an
open swarm with int8 activations and *zero* output verification (hash-committed I/O was future
work). llama.cpp RPC mixes Metal + CUDA hosts with "no quality degradation" but is explicit LAN PoC
code ("never on an open network"). The pieces they all lacked — signed activation receipts, a cosine
spot-check, content-addressed per-format manifests — are the ones this repo already has, which is
what makes a heterogeneous ring trustable rather than just possible.
