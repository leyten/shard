# Proving a real decentralized swarm run

"You served a 120B model across consumer GPUs on different networks" is an extraordinary
claim, so it should be checkable by a skeptic — not taken on trust. This doc defines what a
*verifiable* Shard run looks like and how anyone can confirm one independently. Every run can
emit a **run receipt** (`phase0/proof_receipt.py`); receipts live in `docs/receipts/`.

## What would a fake look like?

The cheap fakes we're ruling out:
1. **One box pretending to be many** — running the whole model on a single machine and
   claiming it was distributed.
2. **Localhost, not WAN** — "distributed" processes all on one host/LAN, no real internet hop.
3. **Cherry-picked / fabricated tok/s** — a number with nothing reproducible behind it.
4. **Wrong output** — a fast pipeline that doesn't actually compute the model correctly.

The receipt is designed so each of these fails an independent check.

## The four proofs

**1. The nodes are genuinely distinct, distributed machines.**
The receipt records, per node: public **IP**, **geolocation** (city/region, from the host
provider), **GPU UUID**, and GPU model. Distinct IPs across different ASNs/regions and
distinct GPU UUIDs can't come from one box. *Verify:* the IPs resolve to different
networks/cities; the GPU UUIDs are all different physical GPUs.

**2. The links are real WAN, not localhost.**
The receipt records the **measured RTT of every pipeline edge** (`phase0/mesh.py`, app-level
round-trip over the live transport). Real inter-city internet is tens-to-hundreds of ms;
localhost is <1 ms. *Verify:* the edge RTTs are WAN-scale and match the geographic distances.

**3. The output is correct — and reproducible bit-for-bit.**
Shard uses **greedy decoding**, so the swarm's output is **token-identical** to a single-node
reference run of the same model + prompt. The receipt includes the prompt, the generated
token ids, and their hash. *Verify:* run the same prompt through the reference (or any
standard inference of the same model) and confirm the tokens match the hash. A pipeline that
faked the compute would not reproduce.

**4. Anyone can re-run the whole thing.**
The engine is open source (Apache-2.0). The receipt embeds the exact commit, model, layer→node
assignment, and launch commands. *Verify:* stand up your own nodes and reproduce — same code,
same result.

## Receipt schema (`docs/receipts/<run_id>.json`)

```json
{
  "run_id": "...", "utc": "...", "shard_commit": "<git sha>",
  "engine_file": "research/...", "engine_sha256": "<hash of the engine source that ran>",
  "model": "gpt-oss-120b", "quant": "mxfp4",
  "prompt": "...", "output_text": "...",
  "output_token_ids": [ ... ], "output_sha256": "<hash of token ids>",
  "tok_s_warm": 24.8, "decode": "greedy (exact)",
  "nodes": [
    {"role": "coordinator|stage|tail", "layer_range": [a, b],
     "public_ip": "x.x.x.x", "geo": "Kansas, US",
     "gpu_uuid": "GPU-...", "gpu_name": "RTX 4090"}
  ],
  "edges": [ {"from": "stage0", "to": "stage1", "rtt_ms": 41.2} ],
  "reference": {"source": "single-node decode", "tokens_match": true, "token_ids": [ ... ]}
}
```

`engine_sha256` is the hash of the exact engine source that produced the run — the precise,
commit-independent reproducibility anchor (a commit can't embed its own hash, so this is what a
skeptic checks the engine file against). `reference.token_ids` lets anyone re-run the verifier with
`--ref-tokens` and confirm the lossless check directly, rather than trusting the `tokens_match` flag.

## How to verify a receipt (skeptic's checklist)

1. **Distinct machines:** all `public_ip` differ and resolve to different networks/regions; all
   `gpu_uuid` differ.
2. **Real WAN:** every `edges[].rtt_ms` is WAN-scale (≫ 1 ms) and consistent with the geos.
3. **Correct output:** re-run the same `model` + `prompt` with greedy decoding anywhere; confirm
   the token ids hash to `output_sha256`.
4. **Reproduce:** check out `shard_commit`, bring up nodes, run the embedded commands.

## Receipts on file

- **GLM-5.2 744B NVFP4 at ~30 tok/s over WAN** — 7 GPUs in 6 US states, pipelined spec-decode
  + CUDA-graphed draft: [`receipts/glm52-nvfp4-wan-20260618.json`](receipts/glm52-nvfp4-wan-20260618.json).
- **gpt-oss-120B at ~40 tok/s over WAN** — 3 stages (12 layers each) + an in-region coordinator
  across 4 US states, pipelined spec-decode; the permissionless build target:
  [`receipts/gpt-oss-120b-wan-20260619.json`](receipts/gpt-oss-120b-wan-20260619.json).

## Scope / honesty

- Decoding is **greedy and deterministic**: same prompt → same tokens, every run. The receipt's
  `tokens_match` is the **lossless-optimization check** — the CUDA-graphed speculative path is
  byte-identical to the plain eager path of the *same engine* (computed from two real run dumps),
  so the speedup changes nothing about the output.
- For a **quantized** model, bit-exact reproduction across *different* engines/backends is not
  achievable in general (and not unique to Shard): batched vs single-token kernels round
  floating-point differently, so at a genuine near-tie two correct greedy decoders can pick
  different — both valid — continuations. So the proof is *within-engine* reproducibility +
  coherent correct output, not "matches your laptop's HF decode token-for-token."
- A receipt proves *a specific run* was real, distributed, and correct. It is not a claim of
  uptime, throughput SLAs, or that every run hits the same number — tok/s is prompt- and
  topology-dependent and reported as a range.
