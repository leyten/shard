# Shard

Pipeline-parallel LLM inference across GPUs on separate machines. A model too
large for any single card is split into contiguous blocks of layers — one shard
per GPU — and a request is served by streaming activations through the shards in
order. No datacenter, no single host, and no node ever holds the whole model.

Shard is the inference engine for [c0mpute](https://c0mpute.ai).

## GLM-5.2 (744B) across seven scattered prosumer GPUs, over the open internet

**A 744-billion-parameter frontier model, served at ~30 tok/s across seven
prosumer Blackwell GPUs in six US states — over WAN, greedy, deterministic.**
GLM-5.2 (NVFP4, 78 layers) is split **13 layers per node** across 6× RTX PRO 6000;
no single card holds it, six do. Each node loads **only its own block**. A coordinator
holds no model layers — just the token embedding/head and a small CUDA-graphed
GLM-4-9B draft that proposes tokens, which the distributed 744B verifies.

| Setup | tok/s (warm) | Output |
|-------|--------------|--------|
| GLM-5.2 744B NVFP4, 6× RTX PRO 6000 across 6 US states (NV · TX · MN · MO · UT + WA coord), WAN, pipelined spec-decode + CUDA-graphed draft | **~30** | greedy, deterministic |

Every run emits a **verifiable receipt** — distinct GPU UUIDs / public IPs / regions,
measured WAN edge RTTs (22–75 ms), the output token hash, and a lossless-optimization
check. This run's receipt: [`docs/receipts/glm52-nvfp4-wan-20260618.json`](docs/receipts/glm52-nvfp4-wan-20260618.json)
(see [docs/PROOF.md](docs/PROOF.md) for how a skeptic checks it).

That is the whole thesis in one line: a frontier-size model, far too big for any
single card, served across machines on different networks — activations crossing
the country on every traversal — at a speed that is actually usable.

## How it got there

Plain pipeline decode over WAN is latency-bound: one round-trip per token, ~1–2
tok/s, unusable. The path to 30 was a sequence of measured steps, each committed:

| Step | tok/s | What changed |
|------|-------|--------------|
| plain KV decode | 1.87 | latency-bound baseline (one token per round-trip) |
| + deep-draft spec-decode (GLM-4-9B), relay-back | 1.99 | one traversal commits several tokens |
| + **ring direct-return** | 2.94 | tail returns to the coordinator in one hop — 7 ring hops, not a 12-hop relay-back |
| + **async pipelining** | 16.6 | overlap many verify traversals in flight → throughput-bound, not latency-bound; the WAN drops to ~5% of the loop |
| + **CUDA-graphed draft** | **~30** | with the WAN hidden, the draft was 94% of the loop; CUDA-graphing it (3.8×) lifts the whole pipeline |

**The key insight: over WAN the round-trip is the scarce resource, not compute** —
so speculative decoding, marginal in a datacenter, becomes the whole game. A small
draft proposes K tokens; the distributed 744B verifies them in a single pipeline
traversal; greedy acceptance commits the verified prefix. Then two compounding wins:

- **Async pipelining over the ring.** Because the ring is direct-return, multiple
  verify chunks can be in flight at once. The coordinator drafts a continuous stream
  and pumps overlapping chunks into the pipeline without waiting — so the loop runs at
  the pipeline's *throughput*, not its *latency*. The WAN, which dominated every prior
  attempt, drops to ~5% of the loop.

- **CUDA-graphed draft.** Once the WAN is hidden, the GLM-4-9B draft (single-token
  decode, launch-overhead-bound) becomes 94% of the loop. Capturing it as a CUDA graph
  cuts it 3.8× (49.7→13.1 ms/tok). The hard part was making the static KV cache honor
  speculative rollback under graph capture — solved by driving the write slot through a
  static-address position tensor; the result is **byte-identical to the eager path**, so
  the optimization is provably lossless. (`research/glm_swarm_nvfp4_cg.py`,
  `research/glm_swarm_nvfp4_cg_diff.py`.)

## How it works

A transformer is a stack of layers. Shard splits the stack into contiguous blocks,
one block per GPU. A token is produced by passing activations through the blocks in
order; each node keeps a KV-cache for its own layers.

    coordinator (WA) ── GLM-4-9B draft (CUDA-graphed) + embed / lm_head
         │
         ├─► stage0 ─► stage1 ─► stage2 ─► stage3 ─► stage4 ─► stage5 ─┐  (verify chunks, pipelined)
         │   NV         TX         (·)        MN         MO        UT    │
         │   0–12       13–25      26–38      39–51      52–64     65–77 │
         └──────────────── direct return (tail → coordinator, 1 hop) ────┘

The coordinator (entry node) holds **no** 744B layers — only the draft and a thin
driver. Each round: the draft proposes K tokens; the coordinator ships `[cur, d₁..dₖ]`
into stage 0, which embeds them; the chain verifies all K+1 in one forward traversal;
the tail returns the argmaxes straight to the coordinator (one hop, not relayed back);
the coordinator greedy-accepts the longest matching prefix. Many such chunks are in
flight at once (the pipeline), and the draft replays a captured CUDA graph against a
static KV cache.

## Why this is hard

Splitting a model across co-located GPUs is well understood. Doing it across machines
on the open internet, fast enough to be usable, is not — and that is the part Shard
owns.

- **Latency.** Every token traverses the whole pipeline. Speculative decoding amortizes
  one round-trip over many committed tokens; pipelining overlaps the traversals so the
  WAN stops being the floor; the CUDA-graphed draft keeps what's left cheap.
- **Transport.** The activation tensor crosses the public internet on every step. Shard
  owns this layer — supervised edges that fail fast and reconnect, per-edge health
  logging, no opaque "broken pipe." The wire is authenticated and encrypted with
  pickle-free framing (`phase0/wire.py`; ChaCha20-Poly1305 under a shared `SHARD_PSK`),
  so a passive observer learns nothing and a forged frame is a parse error, not code
  execution. (NAT hole-punching + relay fallback for home routers is the remaining
  Phase 1 work; a direct open port stands in today.)

## Design principles

Shard is c0mpute infrastructure, held to its three guarantees:

- **Uncensored.** The engine runs models as-is. No content filter in the inference path.
- **Decentralized.** Anyone can join a GPU with one command and be assigned a block of
  layers. No central inference server.
- **Private.** No node holds the whole model — a real start, not the whole story. The
  wire is sealed (authenticated encryption, pickle-free), so the leak is not on the
  path; but a *participating* node must decrypt to run its layer, so it sees the
  activations it processes. Intermediate activations can still leak a fraction of a
  user's tokens to a malicious node. The plan — pin leaky boundary layers to trusted
  nodes, per-request trusted routing, never overclaim — is in
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). It is the number-one open problem and is
  treated as one.

## gpt-oss-120B at ~40 tok/s over WAN — the permissionless build target

120B (MXFP4, 36 layers) across **3 scattered RTX 4090s in different US states** + a
coordinator, **~40 tok/s (peak ~42), greedy, exact**. This is the rig the permissionless
work (Phase 3+) is built on — plain 24GB consumer cards, the hardware a real volunteer runs.
This run's verifiable receipt (distinct GPU UUIDs / IPs / states, WAN edge RTTs, output
hash, sync-vs-pipelined token match): [`docs/receipts/gpt-oss-120b-wan-20260619.json`](docs/receipts/gpt-oss-120b-wan-20260619.json).

The climb from a latency-bound ~18 tok/s, each step measured:

| Step | tok/s | What changed |
|------|-------|--------------|
| pipelined spec-decode (4-stage) | 25.8 | async-draft overlap + many verify chunks in flight + RTT-optimal ring order |
| + **3-stage (12-layer) ring** | 28.8 | fatter stages → 4 WAN hops instead of 5 (12 layers fits a 24GB card) |
| + **coordinator placed in-region** | **~40 (peak ~42)** | the coordinator holds no model layers, so it can live anywhere; moving it off the cross-country leg cut the ring 174→102 ms |

The last step is the one nobody looks for: the cheapest node in the system — the
layer-less coordinator — was sitting a continent away from the swarm, paying two long
round-trips on every token. Putting it next to the stages, on the same scattered nodes,
was a ~40% latency cut for free. Full record:
[docs/research/wan-speculative-decoding.md](docs/research/wan-speculative-decoding.md).

GLM-5.2 (above) remains the **frontier-size** flagship — 6× the parameters at 744B;
gpt-oss-120B is the faster, consumer-card build target the network is bootstrapped on.

## Repository layout

    phase0/   transport + deploy: wire.py (sealed framing), mesh.py (edge RTTs),
              proof_receipt.py (run-receipt build/verify), launch + bench tooling
    research/ the swarm drivers — glm_swarm_nvfp4_kv.py (NVFP4 KV-cached stages),
              glm_swarm_nvfp4_pipe.py (pipelined spec-decode), glm_swarm_nvfp4_cg.py
              (CUDA-graphed draft), *_cg_diff.py / *_fwdcmp.py (correctness diagnostics)
    docs/     ARCHITECTURE, ROADMAP, PROOF.md, receipts/, and the research records
    shard/    engine module scaffolding (node, transport, specdec, topology)

## Roadmap

- **Phase 0 — Transport, proven.** Reliable serving through a multi-stage split.
- **Phase 1 — WAN.** Different networks behind NAT: hole-punching, relay fallback,
  activation quantization, edge supervision.
- **Phase 2 — Speculative decoding.** Draft-and-verify over the swarm — **done at
  GLM-5.2 744B scale, ~30 tok/s greedy over WAN** (and gpt-oss-120B at ~18–25, above).
- **Phase 3 — Permissionless swarm.** One-command join, dynamic layer allocation
  across heterogeneous GPUs, per-token payouts, fault tolerance.

Full detail, pass/fail criteria, and risks: [docs/ROADMAP.md](docs/ROADMAP.md).

## License

[Apache License 2.0](LICENSE) © 2026 leyten
