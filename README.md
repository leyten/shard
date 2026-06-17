# Shard

Pipeline-parallel LLM inference across GPUs on separate machines. A model too
large for any single card is split into contiguous blocks of layers — one shard
per GPU — and a request is served by streaming activations through the shards in
order. No datacenter, no single host, and no node ever holds the whole model.

Shard is the inference engine for [c0mpute](https://c0mpute.ai).

## gpt-oss-120B across four consumer GPUs, over the open internet

**120 billion parameters, served at ~18–25 tok/s across four RTX 4090s in
different US states — over WAN, output exact.** The model is split 9 layers per
node across 4× RTX 4090 (~16 GB each); no single 24 GB card holds it, four do.
Each node loads **only its own block** — the rest of the model is never
materialized on it. A small in-house draft proposes tokens and the distributed
120B verifies them in one traversal of the chain.

| Setup | tok/s (warm) | Output |
|-------|--------------|--------|
| 120B, 4× RTX 4090 across 3 US sites (Kansas ×2 · Illinois · N. Carolina), WAN, spec-decode + fast verify | **18.5 – 24.8** (prompt-dependent) | exact greedy decode |

That is the whole thesis in one line: a frontier-size model, far too big for any
consumer GPU, served across machines on different networks — activations crossing
the country on every traversal — at a speed that is actually usable.

## How it got there

Plain pipeline decode over WAN is latency-bound: one round-trip per token, ~1–6
tok/s, unusable. The path to 25 was a sequence of measured steps, each recorded in
the [research log](docs/research/wan-speculative-decoding.md):

| Step | tok/s (120B, WAN) | What changed |
|------|-------------------|--------------|
| plain decode | ~3–5 | latency-bound baseline |
| + speculative decoding (in-house 20B draft) | 13.3 | one traversal commits several tokens |
| + coordinator + clustered US + direct return | 7.83\* | real 4-scattered-GPU topology, tail returns to the entry node in one hop |
| **+ fast verify (static-cache CUDA graph)** | **18.6 → 24.8** | the verify compute, not the WAN, was the wall |

\*the 7.83 is the harder 4-separate-box topology that the fast verify then lifts to ~19–25.

**The key insight: over WAN the round-trip is the scarce resource, not compute** —
so speculative decoding, marginal in a datacenter, becomes the whole game. A small
draft proposes K tokens; the distributed 120B verifies all K in a single pipeline
traversal (the same round-trip plain decode spends on one token); greedy acceptance
keeps the output **token-for-token identical to plain decode**.

Two non-obvious results made it fast:

- **In-house draft (no training).** gpt-oss-20B under vLLM on one 4090 drafts at
  ~5 ms/tok vs 62 ms in plain transformers — a 12× cheaper draft from optimized
  MXFP4 kernels + CUDA graphs, no training. It runs on the entry node; it holds no
  authority (the target verifies every token), so centralizing it is safe.

- **Fast verify (static-cache CUDA graph).** The eager verify was ~75% removable
  Python / kernel-launch overhead. Capturing a 120B stage's forward as a CUDA
  graph against a pre-allocated static KV cache makes it **bit-exact and ~5×
  faster** — cutting the clustered verify from 372 ms to ~135 ms (warm) and taking
  the 4-separate-GPU topology from 7.83 to **24.8 tok/s**, past this track's 20
  tok/s success criterion. (`phase0/fastverify.py`, `research/fastverify_graph.py`.)

## How it works

A transformer is a stack of layers. Shard splits the stack into contiguous blocks,
one block per GPU. A token is produced by passing activations through the blocks in
order; each node keeps a KV-cache for its own layers.

    coordinator ──► draft (20B, proposes K)
         │
         └─► stage 0 ──► stage 1 ──► stage 2 ──► stage 3 ──┐  (verify K+1 in one traversal)
             KS           KS          IL          NC        │
             ▲────────────── direct return ─────────────────┘

The coordinator (entry node) holds **no** 120B layers — only the draft and a thin
driver. Each round: the draft proposes K tokens; the coordinator sends `[cur, d₁..dₖ]`
into stage 0, which embeds them; the chain verifies all K+1 in one forward traversal;
the tail returns the argmaxes straight to the coordinator (one hop, not relayed back);
the coordinator greedy-accepts the longest matching prefix. The verify forward on each
stage replays a captured CUDA graph against a static KV cache — the fast verify.

## Why this is hard

Splitting a model across co-located GPUs is well understood. Doing it across machines
on the open internet, fast enough to be usable, is not — and that is the part Shard
owns.

- **Latency.** Every token traverses the whole pipeline. Speculative decoding amortizes
  one round-trip over many committed tokens; the fast verify keeps the traversal cheap
  enough that the WAN, not the compute, sets the floor.
- **Transport.** The activation tensor crosses the public internet on every step. Shard
  owns this layer — supervised edges that fail fast and reconnect, per-edge health
  logging, no opaque "broken pipe." The wire is authenticated and encrypted with
  pickle-free framing (`phase0/wire.py`; ChaCha20-Poly1305 under a shared `SHARD_PSK`),
  so a passive observer learns nothing and a forged frame is a parse error, not code
  execution. (NAT hole-punching + relay fallback for home routers is the remaining
  Phase 1 work, where per-node identities + a keyed handshake replace the shared key;
  a direct open port stands in today.)

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

## Repository layout

    phase0/   the working engine — node_kv.py (2-node split), pipeline.py (N-stage),
              specpipe.py (speculative decoding over the split), fastverify.py
              (static-cache CUDA-graph verify), tree.py (tree speculation), bench.py
    research/ the WAN spec-decode experiments — draft_server.py (in-house vLLM draft),
              fastverify_*.py (the de-risking probes), launch scripts
    docs/     ARCHITECTURE, ROADMAP, and research/wan-speculative-decoding.md (the
              full design record + measured milestone log)
    shard/    engine module scaffolding (node, transport, specdec, scheduler)

## Roadmap

- **Phase 0 — Transport, proven.** Reliable serving through a multi-stage split.
- **Phase 1 — WAN.** Different networks behind NAT: hole-punching, relay fallback,
  activation quantization, edge supervision.
- **Phase 2 — Speculative decoding.** Draft-and-verify over the swarm — **done at
  120B scale, ~18–25 tok/s exact over WAN** (see above).
- **Phase 3 — Permissionless swarm.** One-command join, dynamic layer allocation
  across heterogeneous GPUs, per-token payouts, fault tolerance.

Full detail, pass/fail criteria, and risks: [docs/ROADMAP.md](docs/ROADMAP.md).
The WAN speculative-decoding design record: [docs/research/wan-speculative-decoding.md](docs/research/wan-speculative-decoding.md).

## License

[Apache License 2.0](LICENSE) © 2026 leyten
