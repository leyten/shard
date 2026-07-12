# MlxRuntime — the Apple-silicon backend (MAC GATE GREEN 2026-07-12)

`shard/mlx_runtime.py` implements `ModelRuntime` (shard/node.py — the firewall of
docs/MODEL_RUNTIME.md) on MLX: one Mac serves a contiguous layer range of an MLX-converted
checkpoint out of unified memory, speaking the exact per-node contract the proven CUDA stage
(phase0/m25_stage.py) runs on.

**GATE STATUS: 11/11 real-silicon checks PASSED on a rented Scaleway M2 Pro (16 GB, macOS
Tahoe, mlx 0.32.0 / mlx-lm 0.31.3, python 3.13) against the REAL
`mlx-community/MiniMax-M2.5-4bit` checkpoint, layers [29:32)** — receipt
`docs/receipts/mlx-mac-gate-20260712.json`. Highlights: range load 5.77 GiB active for 3
layers (≈1.9 GiB/layer, matching the memo's math); forward finite + run-to-run byte-equal in
BOTH wire representations; **KV rollback byte-equal to a fresh cache on-device** (the m25 crop
contract holds on Metal; gap hard-errors); aux capture per contract; the corrupt-index refusal
fires on the real index; real tail logits (200 064 vocab) + head embed; **our wrapper is
byte-identical to a hand-driven mlx-lm layer loop** (mask/positions/trim bookkeeping adds
zero drift); **measured decode 0.94 ms/layer (M2 Pro, B=1)** — the first real Apple-silicon
row for the admission spec. All 13 MAC-VALIDATE API assumptions held as written.

## Build shape

- **Range loader** (`load_shard`): read `config.json` + `model.safetensors.index.json`,
  select ONLY the tensors for layers `[lo:hi)` (+ embed if head, + final norm & lm_head if
  tail) via the pure `select_weight_keys` (derived from the index `weight_map`, never
  hardcoded key lists); lazy `mx.load` per file; build the full mlx-lm model skeleton
  (`mlx_lm.models.minimax` for M2.5 via the `minimax_m2 → minimax` remap), **None-pad every
  unowned layer slot** (mlx-lm's PipelineMixin pattern — keeps absolute layer indexing),
  quantize per `config["quantization"]` (per-path overrides honored, e.g. 8-bit MoE gates),
  `load_weights(strict=False)` with the subset, then `mx.eval` exactly the range's arrays.
- **KV + rollback**: one mlx-lm `KVCache` per owned layer. `forward(hidden, start_pos)`
  **crops each cache back to `start_pos`** (`.trim`) before running, mirroring m25's
  spec-decode rollback semantics exactly: a re-prefill at an earlier start overwrites stale
  speculative KV; a start_pos AHEAD of the cache is a hard error (RoPE positions come off
  `cache.offset` in mlx-lm layers). `reset()` drops the cache objects.
- **EAGLE aux**: `aux_layers` (absolute indices; env parity with `M25_EAGLE`/`M25_EAGLE_AUX`,
  default `1,30,58`) — the output residual stream after each in-range aux layer lands in
  `.aux[{layer_idx}]` as `[S,H]` numpy, m25's `_AUX` shape and moment.
- **Wire dtypes**: `forward` takes numpy `[1,S,H]` float32 **or bf16 bit-patterns as uint16**
  (numpy has no bf16; `f32_to_bf16_bits`/`bf16_bits_to_f32` are the pure bridge) and returns
  the same representation; compute is mx bf16. `embed`/`logits` return float32.
  **fp8 wire frames must be dequantized to bf16 before a Mac boundary for now** — mx float8
  support is unverified (TODO gated on checking it on-device).
- **Contract double**: `shard/mlx_stub.py` (`MlxRuntimeStub`) — same surface in pure numpy
  over a 4-layer stateful fake, so protocol tests exercise call/shape/dtype/rollback offline.

## Validated OFFLINE (tests/test_mlx_runtime.py, green with no mlx installed)

- module import + honest ImportError at call time without mlx; heartbeat never raises;
- weight-key selection against a synthetic minimax-shaped index (experts/gate/norms/quant
  companions, head/tail edges, tied-embedding fallback, `layers.1.` vs `layers.10.` prefix
  collision, file resolution);
- bf16 bit-bridge round-trip; stub pipeline shapes/dtypes/determinism; byte-exact KV
  rollback (rolled-back runtime == fresh runtime), reset, gap refusal; aux capture in-range
  only, per-forward refresh, aux-under-rollback equality.

## Needs a Mac (the gate — in order) — STATUS after 2026-07-12 (M2 Pro 16 GB run)

Done: items 1 (range load), 3 (all API assumptions), 4 (on-device rollback), 6's decode half
(0.94 ms/layer measured; prefill still unmeasured), plus wrapper-vs-mlx-lm byte parity.
Remaining, in order: **7 (fake-ring stage swap → live mixed ring — the flagship)**, 2
(full-model reference greedy agreement — needs a ≥96 GB Mac; the 16 GB box can't hold the
model), 5 (wired-limit sizing table at larger RAM), 6's prefill measurement.

1. **Real range load** of `mlx-community/MiniMax-M2.5-4bit`: `load_shard` on a mid-range
   `[20:30)`, memory ≈ 10 × ~2.06 GB (not the full 129 GB) — proves lazy load + None-pad.
2. **Layer-range forward finite + faithful**: stage-range forward output matches the same
   slice of a full-model mlx-lm reference run (same checkpoint) to weight-quant tolerance;
   greedy next-token agreement over a few hundred steps vs `mlx_lm.generate`.
3. **API assumptions** (the MAC-VALIDATE list in the PR/report): `MODEL_REMAPPING` import
   path, `DecoderLayer(x, mask, cache)` signature, `create_attention_mask(h, cache)` return
   type, `KVCache.trim/offset` semantics, `nn.quantize(..., mode=...)` kwarg, None-assignment
   to module attributes, `mx.view(a, mx.uint16)` for the bf16 bridge, `Embedding.as_linear`,
   **`model.sanitize(weights)` a true no-op on a pre-converted checkpoint's range SUBSET**
   (a sanitize that reshapes subsets unexpectedly is silent-corruption-shaped), and
   `mlx.utils.tree_flatten(model.parameters())` post-pruning = exactly the owned set (the
   completeness audit + strict load depend on it).
4. **Rollback on-device**: the stub's rollback test rerun against the real runtime
   (fresh-vs-rolled-back equality at bf16 tolerance).
5. **Wired-limit sizing**: raise `iogpu.wired_limit_mb`, load N layers, confirm no swap;
   fill the table below with measured values.
6. **decode ms/layer** for the admission probe (derived, unmeasured: ~0.45 ms M4 Max, ~0.9
   M4 Pro) + prefill throughput (Apple silicon's weak point — MEASURE before giving a Mac a
   prefill-heavy role).
7. **Fake-ring stage swap**: replace one stage of tests/fake_ring.py's pipeline with the Mac
   runtime over the wire; then a live scattered ring A/B for the backend's own g.

## Numerics / receipt policy

Community MLX 4-bit = group-64 **affine** quant of experts **and attention**; the NVIDIA
path = NVFP4 experts + bf16 attention. Same accepted-kernel-numerics class as fp8 wire:
high per-step greedy agreement, **no token-exact cross-backend parity**, drift grows with
length. Therefore: (1) quote g per backend, never across; (2) receipts pin
**(backend, quant scheme, checkpoint hash)** per stage — a Mac stage makes the served model
a placement-defined mixed-quant composite and the receipt must say so; (3) before real
deployment, run our **own conversion keeping attention bf16** (`mlx_lm.convert` mixed-quant
predicate, ~+4 GB total) so the Mac matches the NVIDIA precision policy.

## Target demo box

Scaleway Apple-silicon M4-XL (64 GB): usable ~44 GB at the default wired limit, ~56 GB
raised → **~20–26 layers** of M2.5-4bit (≈ 2× a 5090 stage). A 62-layer ring closes with
two such Macs + one 48 GB CUDA anchor, or one Mac replacing two 5090 stages on an existing
ring — the heterogeneity proof the engine-genericity decision calls for.
