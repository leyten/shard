# Vendored reference: MiMo-V2 decoder layer (math only)

**Source:** `modeling_mimo_v2.py` of https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL at revision
`5711b268169967567844e1e560e8a3966da959b1` (sha256 of the file:
`a8c3cb3aae473bcc15f023010547c919f15eba6546e6ed7efb61a8937b12f3ad`).
**License:** Apache-2.0 (the file's own header, Copyright 2026 Xiaomi Corporation and The HuggingFace
Inc. team; the model repo itself is MIT). The header is preserved in `modeling_mimo_v2_layer.py`.

## What is here and why

The CPU MiMo backend (`shard/runtime/adapters/mimo_v2.py` + `shard/runtime/cpu.py`) is tested in CI
against the vendor's OWN decoder-layer math on a tiny random MiMo-shaped model
(`tests/test_mimo_cpu.py`). The full vendor file imports ~15 transformers internals (Cache, masking,
attention registry, rope registry, hub kernels, generation) whose churn has already broken it once
(the file targets transformers 5.3; 5.12 renamed the mask builder's kwargs). So only the decoder-layer
subset is vendored, as a file that needs nothing but torch:

`rotate_half`, `apply_rotary_pos_emb`, `repeat_kv`, `eager_attention_forward`, `MiMoV2RMSNorm`,
`MiMoV2MLP`, `MiMoV2MoEGate`, `MiMoV2MoE`, `MiMoV2Attention`, `MiMoV2DecoderLayer`,
`MiMoV2RotaryEmbedding`.

## Edits (every math line is verbatim; the non-trivial edits are marked `# shard:` in the file)

1. Imports: the transformers imports are gone; `ACT2FN` is a local `{"silu": nn.SiLU()}`.
2. `@use_kernel_forward_from_hub("RMSNorm")` dropped (a hub-kernel swap, not math).
3. Attention dispatch: the vendor looks up `ALL_ATTENTION_FUNCTIONS[config._attn_implementation]` and
   falls back to eager whenever sinks exist; here eager is the only path (the paged/SDPA branches and
   their kwargs are removed).
4. `MiMoV2RotaryEmbedding`: only the `default` rope type (MiMo-V2.6's); `ROPE_INIT_FUNCTIONS` and the
   `@dynamic_rope_update` decorator (a no-op for `default`) are gone, and
   `PretrainedConfig.standardize_rope_params()` is replaced by filling `rope_theta` /
   `partial_rotary_factor` into `rope_parameters` when absent.
5. Type annotations naming transformers types (`Cache`, `TransformersKwargs`) are loosened.
6. Added (shard's, not vendor code): `mimo_mask(T, window, dtype)`, the additive eager mask.

## Evidence the subset is the vendor module

On 2026-09-28 the subset was run against the FULL unmodified file (transformers 5.12.1, masks from
transformers' `create_causal_mask` / `create_sliding_window_causal_mask`) with identical weights and
inputs: **bit-identical** outputs in fp32 and bf16 for a global dense, an SWA MoE and a global MoE
layer of the tiny test model. The full file itself was run on the real Flash layers 0, 1 and 5 in the
scratch oracle work (`scratchpad/v2/mimo/`, parity numbers in the R2 receipt).

## Known vendor-file defect this does NOT inherit silently

The vendor's fused-qkv path does `qkv.split([q, k, v])`, which is wrong on the released FP8 checkpoint
(the rows are pre-sharded per checkpoint-TP chunk, `shard/formats/mimo_v2.py`). Tests therefore hand
this module a de-interleaved `[Q; K; V]` weight; the adapter's own loader de-interleaves with
`shard.formats.mimo_v2.qkv_rechunk`.
