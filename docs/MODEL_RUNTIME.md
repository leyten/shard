# Model runtime — the engine-genericity decision

**One engine, every model.** The decision that makes shard a *sharded-inference engine* instead of an
*M2.5 server*. Decided 2026-06-28.

## The decision

**Own the moat in-house; rent model execution behind a firewall interface.**

- **In-house (the moat):** the WAN ring, transport, spec-decode coordinator, scheduler/topology, signed
  receipts, verification numerics + determinism, economics. This is what `shard/` and the `coordinate_pipe`
  orchestration already are — and they are already model-agnostic (`m25_pipe.py` header: *coordinate_pipe
  only orchestrates token-ids + argmax over sockets; reused UNCHANGED, we only provide M2.5-native stage
  serve loops*).
- **Inherited (commodity):** the per-architecture forward pass + quant kernels, pulled from an existing model
  registry (vLLM preferred for the production quant/MoE/attention kernels; HF Transformers as the universal
  fallback). Accessed **only** through the `ModelRuntime` interface (`shard/node.py`) so a given backend is
  **swappable per model** and the blast radius of churn / an exotic arch / a determinism gap is one adapter.

This is **not** "all in-house" (re-deriving every architecture by hand is a treadmill that keeps us behind
the frontier and burns the focus the moat needs) and **not** "depend on vLLM as the engine" (its
datacenter/NCCL/trusted-owner assumptions actively fight the hostile-WAN, untrusted, per-stage-verified
regime). It is: borrow the model zoo, own the network.

## Why

Model coverage is **table stakes, not a moat** — demand leaves the instant we don't run the model they want,
but nobody pays a premium because we *do*. The durable moats are **supply liquidity**, **trustless
verification**, and **incentive/economic design** — none of which is the model layer. So the model layer must
be broad + instant (inherited), and the team's effort must compound on the moat (in-house).

The one real pro-"build it all" argument — *verification needs deterministic execution we control* — narrows,
when followed through, to **owning the verification/commitment layer** (already in-house) plus **pinning
determinism on inherited kernels** (a config + validation cost; cf. batch-invariant MoE,
[[m25-batch-invariant-moe]]). The ZK path proves the *math* (sumcheck over the matmuls), not a specific
kernel, so it does not require owning the forward pass. Determinism does not justify hand-building the zoo.

## The seam

| Concern | Where | Notes |
|---|---|---|
| Transport / NAT / encrypted wire | **in-house** | `shard/transport.py`, `phase0/wire.py` |
| Spec-decode coordinator + drafter | **in-house** | `coordinate_pipe` — already model-agnostic |
| Scheduler / topology / heal | **in-house** | `shard/scheduler.py`, `topology.py` |
| Receipts / challenge / verification | **in-house** | `shard/receipt.py`, `challenge.py` — the moat |
| Determinism on the verify path | **in-house** | pin inherited kernels (batch-invariant config) |
| Per-architecture forward pass | **inherited** | vLLM/Transformers model class, behind `ModelRuntime` |
| Quant / MoE / attention kernels | **inherited** | vLLM (NVFP4 FusedMoE on sm_120 already proven) |
| Weight key names / layer slicing | **derive** | from the manifest `weight_map` + config, not hardcoded |
| Tokenizer / chat template / tool parse | **inherited** | the model's own renderer — a `chat_template.jinja`, or a Python one where the model ships no template at all (K3); plugin per family, `phase0/model_tools.py` |

## The interface

`shard/node.py` defines `ModelRuntime` — the firewall. It captures the real per-node serve contract that
`m25_stage.py` + `m25_pipe.py` already run on (`reset()` / `forward(hidden, start_pos)` / `run_block`, head
embeds, tail does norm + lm_head → logits, KV lives per-layer and crops to `start_pos` for spec-decode
rollback). Two implementations satisfy it:

1. **`M25Runtime`** (exists, hand-rolled `phase0/m25_stage.py`) — the tuned betanet fast-path. Keep it.
2. **`VllmRuntime`** (to build) — generic: registry → model class → slice layer list to `[lo:hi]` → drive the
   block forward. This is the generalization of `m25_stage.py` from one model to all of them.

Note: this **supersedes** the `docs/ARCHITECTURE.md` "per-node runtime wraps SGLang" line. The refinement:
wrap a model-execution *library* as a swappable backend behind `ModelRuntime` — never as the engine; the
orchestration stays ours.

## Plan

1. Promote `shard/node.py` from stub → `ModelRuntime` interface (the firewall). **← done with this decision.**
2. De-risk the load-bearing assumption with a GPU spike: instantiate a model from the registry, slice its
   layers to a block, run that block's forward in isolation, confirm finite + matches a full-model reference.
   **Needs a GPU box — gated on ops go-ahead (don't improvise vast launches).**
3. Build `VllmRuntime` behind the interface once the spike holds.
4. Fix the two model-agnostic leaks. **Both done.** Weight keys now derive from the manifest `weight_map`
   (PR #139/#140) instead of hardcoded `model.layers.{j}` / `model.embed_tokens` / `model.norm` / `lm_head`.
   Chat rendering + tool parsing now dispatch per model family through `phase0/model_tools.py` — `m25_tools`
   (MiniMax XML) and `k3_tools` (Kimi-K3 XTML) — selected at the call site from the served model id, with
   `m25` as the fallback so an existing ring binds exactly the objects it always did.
5. **Prove genericity by onboarding model #2** — a small dense model (7-8B Qwen/Llama) end-to-end over the
   existing ring. The executable proof that this is an engine, not an M2.5 server.
6. Retire the hand-rolled M2.5 path once `VllmRuntime` matches it bit-for-bit on M2.5.

See [[north-star-torrent-for-compute]] — M2.5 is the betanet/PoC; this decision is what lets the catalog widen
to "many ever-bigger models." Training stays a separate engine (the rails carry over, the execution core
does not).
