# Runbook: add a model

Turn "add a model" from a per-architecture engine integration (weeks) into a config row +
a quantize + a smoke test (an afternoon). This is the whole point of M1–M4.

For any architecture the runtime already supports (`generic-vllm` / `generic-hf`), there is
**no new attention code, no new MoE code, no new engine path, no new launcher**. You add a
signed registry row, quantize + host the weights, and run one smoke. The only irreducible
per-model cost is the quantization.

## TL;DR

```bash
# 1. derive + sign a registry row from the model config (no weights downloaded)
python phase0/add_model.py add --hf <org/Model> --id shard-<name> \
    --quant <nvfp4|mxfp4|fp8> --adapter <generic-vllm|glm-nvfp4> \
    --tokenizer <org/Model> \
    --src registry/models.src.json --key ~/.shard/publisher.key --out registry/models.json

# 2. quantize the weights to the served format + host the shards (the one real per-model cost)
#    then publish the signed weight manifest and paste its CID into the row, re-sign:
python phase0/publish_manifest.py --hf <org/Model> --key ~/.shard/publisher.key \
    --out registry/manifests/shard-<name>.json
#    -> edit registry/models.src.json: set weightManifestCid, re-run step 1's `add` (idempotent)

# 3. tokenizer + chat-template round-trip on the coordinator (HARD BLOCKER if it fails)
python phase0/tokenizer_roundtrip.py --model shard-<name>

# 4. plan + a single 3-stage smoke gen + receipt verify (metered fleet)
python phase0/plan_ring.py --ids <a,b,c> --model shard-<name>
python phase0/launch.py --model shard-<name> --stages <...> --layers <...> --receipts

# 5. open a PR with the new registry row (registry/models.json + models.src.json)
```

## Why this works (M1–M4)

- **M1 — one signed registry.** A model is defined exactly once in `registry/models.json`
  (`shard-models/1`), signed with the publisher ed25519 key. Both repos read it: shard
  (`shard/registry.py`) and c0mpute (`lib/orchestrator/modelRegistry.ts`). The layer count /
  bytes-per-layer / quant / engine path no longer live in three places that drift (the
  gpt-oss 120-vs-36 bug). Editing a value in one file changes both repos; a tampered field
  fails the signature.
- **Runtime seam — `ModelRuntime` (upstream `shard/node.py`).** The in-house firewall between
  the moat (ring/transport/spec-decode/receipts, all model-agnostic) and the commodity model
  layer. `coordinate_pipe` speaks only token-ids + hidden-states + argmax, so it rides on any
  `ModelRuntime` impl unchanged. Adapters: `M25Runtime` (tuned betanet path), `GenericHFRuntime`
  (`phase0/hf_runtime.py`, the universal HF fallback — any arch transformers can load, incl.
  gpt-oss, with no new engine code), and `VllmRuntime` (to build — vLLM NVFP4/MXFP4/FusedMoE
  kernels, MODEL_RUNTIME.md step 3). `make_runtime(adapter, ...)` maps the registry `adapter`
  field to the impl.
- **Adapter selection.** The registry `adapter` field picks the runtime: `generic-vllm`
  (the open-model zoo, quant/tokenizers/chat templates for free), `generic-hf`/`hf` (the
  torch-only fallback), or `glm-nvfp4` (the specialized GLM path kept where it earns its keep).
- **M4 — `add_model.py`.** Wraps the checklist: reads the config, computes `gbPerLayer` +
  `kvGbPerLayer` for the scheduler fit, emits + signs the registry row.

## The quantize step (irreducible — a human picks the format and runs it)

Quantization cannot be abstracted away. Make it mechanical, not research, with a documented
recipe per format. Pick the format that matches the `adapter`/runtime you target:

| quant  | typical use                         | notes |
|--------|-------------------------------------|-------|
| nvfp4  | Blackwell (sm_120), MoE             | 4-bit + fp8 block scales; the GLM-NVFP4 specialized path |
| mxfp4  | gpt-oss and other MXFP4 checkpoints | 4-bit + e8m0 block scales; often ships pre-quantized on HF |
| fp8    | broad GPU support, 2x of nvfp4 VRAM | 8-bit + scales |

If the model already ships pre-quantized on HF in your target format, skip quantization and
point `enginePath` at the downloaded checkpoint — `add_model` accepts pre-quantized shards.

## The fit math (so a new model schedules on day one)

`plan_ring` / `scheduler_svc` size rings from `gbPerLayer` + `kvGbPerLayer`. `add_model`
computes both from the config:

- `gbPerLayer` = (attention + MLP params for one layer) × bytes/param-at-quant. MoE layers
  count `num_experts × expert_mlp` (+ shared expert + router). Deliberately conservative so
  the VRAM fit never over-commits a card.
- `kvGbPerLayer` = `2 × n_kv_heads × head_dim × ctx × 2B` (bf16 KV). Overestimates for MLA
  models (the latent KV is far smaller) — conservative is correct for a fit bound. Override
  per-model in the registry row once measured.

Both are overridable on `plan_ring` (`--gb-per-layer`, `--kv-gb-per-layer`) and in the row.

## The tokenizer round-trip (silent-failure guard)

A wrong chat template degrades quality with **no error**. Treat a failing round-trip as a
hard blocker in onboarding: the coordinator must format a prompt with the registry's
`tokenizerId` + `chatTemplate` and detokenize back to the same text. `add_model` does NOT
sign off on this — run step 3 explicitly before the smoke.

## Acceptance (what "done" means)

- A model in a runtime-supported arch goes from "nothing" to a signed, verifiable registry
  row with **zero engine code** — proven offline by `phase0/add_model_test.py`.
- The cross-repo test (`tests/registry_cross_repo_test.py`) shows both repos resolve the new
  id to the same spec (the gpt-oss-120-vs-36 guard).
- The 3-stage smoke produces verifiable receipts that tile `[0:layerCount]` (metered fleet).

## Verify (all $0)

```bash
python3 shard/registry_test.py              # registry sign/verify/tamper
python3 phase0/hf_runtime_test.py           # generic ModelRuntime adapter + boundary law
python3 phase0/add_model_test.py            # onboarding core + M4 acceptance (offline half)
python3 tests/registry_cross_repo_test.py   # both repos agree (needs ../c0mpute + npx tsx)
```
