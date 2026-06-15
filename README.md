# Shard

Pipeline-parallel LLM inference across GPUs on separate machines. A model too
large for any single card is split into contiguous blocks of layers — one shard
per GPU — and a request is served by streaming activations through the shards in
order. No datacenter, no single host, and no node ever holds the whole model.

Shard is the inference engine for [c0mpute](https://c0mpute.ai).

## 100B+ across the swarm

**gpt-oss-120b — 120 billion parameters — served across 4 nodes.** Split 9 layers
per node across 4× RTX 4090 (~16 GB each, ~64 GB total); no single 24 GB card can
hold it, four can. Each node loads **only its own block** — the rest of the model
is never materialized on that node. Coherent output at 6.3 tok/s (4-stage
pipeline, co-located).

A model far too big for any consumer GPU, running across several — the whole
point, proven. Combined with the WAN transport below (proven Norway → North
Carolina at ~115 ms), this is the path to an internet-scale swarm.

## Status

**Phase 0 complete** — a two-node split serving tokens with a per-node KV-cache,
over a transport we control.

The figures below are an early milestone. They show that the split-and-stream
mechanism is correct and reliable on co-located hardware; they are **not** a
final performance spec. Throughput and latency will change as WAN transport
(Phase 1) and speculative decoding (Phase 2) land.

**Phase 0 — 2× RTX 4090, co-located, low-latency link:**

| Metric      | Result                                                            |
|-------------|-------------------------------------------------------------------|
| Model       | Qwen2.5-14B-Instruct — bf16, 29.5 GB (exceeds a single 24 GB 4090) |
| Split       | 24 / 24 layers across two GPUs (~14.8 GB each)                    |
| Reliability | 20 / 20 clean completions                                         |
| Throughput  | ~16 tok/s decode (median 16.2)                                    |
| Transport   | Custom TCP — per-edge timeouts, fault detection, instrumented     |

The 14B model does not fit on a single 24 GB card in bf16; split across two, it
serves reliably. That is the point of the milestone — a model too big for one
GPU, running across several.

Not done yet, stated plainly: the Phase 0 figures above are co-located; the WAN
numbers below run over a direct open port between hosts. NAT hole-punching and a
relay fallback (for nodes behind home routers) are the remaining Phase 1 work.
See [docs/ROADMAP.md](docs/ROADMAP.md).

## Speculative decoding (Phase 2, prototype)

A small draft model runs locally on the entry node and proposes K tokens; the
split target verifies all K in a single pipeline traversal. Greedy acceptance,
so the output is exact — token-for-token identical to plain decode.

Measured on the 2-node 14B split (co-located, K=6):

| Draft | Workload | Accepted / round | Tokens per traversal |
|-------|----------|------------------|----------------------|
| 1.5B  | prose    | 2.3 / 6          | 3.3×                 |
| 0.5B  | code     | 5.1 / 6          | 6.0×                 |

Acceptance is independent of the link, so the traversal count carries straight to
WAN. Over a real transatlantic link (Norway head → North Carolina tail, ~115 ms
RTT, Qwen2.5-3B split 18/18, 0.5B draft, adaptive K):

| Workload | Plain decode | Spec-decode | Speedup |
|----------|--------------|-------------|---------|
| Code     | 6.0 tok/s    | 20.5 tok/s  | 3.4×    |
| Prose    | 5.9 tok/s    | 9.7 tok/s   | 1.6×    |

Plain decode is latency-bound — one round-trip per token (~133 ms/step measured).
A spec-decode round costs one round-trip too but commits several tokens, turning a
latency-bound 6 tok/s into 20 on code: the difference between unusable and usable
over the open internet.

## How it works

A transformer is a stack of layers. Shard splits the stack into contiguous
blocks and places one block on each GPU. A token is produced by passing
activations through the blocks in order:

    Prompt ─► embed ─► Node 0             Node 1 ─► sample ─► token
                       layers 0–23  ────►  layers 24–47
                       RTX 4090            RTX 4090

Each node holds only its block in VRAM, runs that block's forward pass, and ships
the resulting hidden-state tensor to the next node. Each node keeps a KV-cache
for its own layers, so the prompt is processed once and decoding sends only the
new token's activations downstream.

## Why this is hard

Splitting a model across co-located GPUs is well understood. Doing it across
machines on the open internet, fast enough to be usable, is not — and that is the
part Shard is built to own.

**Latency.** Decoding is one token at a time, and every token traverses the whole
pipeline. With nodes on home connections at 50–80 ms per hop, a multi-stage
pipeline costs hundreds of milliseconds per token — 1–2 tok/s, unusable. Shard's
answer is speculative decoding over the swarm: a small draft model on the entry
node proposes several tokens, and the large split model verifies them in a single
pass through the pipeline, amortizing one round trip over many tokens. (Phase 2.)

**Transport.** The activation tensor crosses the public internet between machines
behind NAT on every step. It has to be fast, reliable, and able to reconnect when
a home connection drops. Shard owns this layer instead of treating it as a black
box: direct QUIC between adjacent stages with hole-punching, a relay fallback for
symmetric NAT, quantized activations to cut bandwidth, and supervised edges that
fail fast and reconnect. Every edge logs its own health — there is no opaque
"broken pipe." (Phase 1.)

## Design principles

Shard is c0mpute infrastructure, and is held to its three guarantees:

- **Uncensored.** The engine runs models as-is. No content filter in the
  inference path.
- **Decentralized.** Anyone can join a GPU with one command and be assigned a
  block of layers. The scheduler is light and replaceable; there is no central
  inference server.
- **Private.** No node holds the whole model — a real start, but not the whole
  story. Intermediate activations can still leak a meaningful fraction of a
  user's tokens to a malicious node in the pipeline. Shard's plan — pin the leaky
  boundary layers to trusted nodes, offer per-request trusted routing, and never
  overclaim — is detailed in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). This
  is the number-one open problem, and it is treated as one.

## Running Phase 0

Two machines, each with a CUDA GPU, plus Python, `torch`, and `transformers`.
Weights download from Hugging Face on first run.

On the tail node (holds the second half of the layers):

    python phase0/node_kv.py --role tail --split 24 --port 29501 \
        --model Qwen/Qwen2.5-14B-Instruct

On the head node (holds the embedding and first half, and drives generation):

    python phase0/node_kv.py --role head --split 24 --port 29501 \
        --peer <TAIL_IP> --model Qwen/Qwen2.5-14B-Instruct \
        --prompt "Explain pipeline parallelism in two sentences."

To reproduce the reliability numbers above:

    python phase0/bench.py --split 24 --port 29501 --peer <TAIL_IP> \
        --runs 20 --model Qwen/Qwen2.5-14B-Instruct

For a quick smoke test on a smaller model, drop `--model` (defaults to
Qwen2.5-3B-Instruct) and use `--split 18`.

## Roadmap

- **Phase 0 — Transport, proven (done).** Reliable serving through a two-node
  split on a low-latency link.
- **Phase 1 — WAN.** The same split across machines on different networks behind
  NAT: hole-punching, relay fallback, activation quantization, edge supervision.
- **Phase 2 — Speculative decoding.** Draft-and-verify over the swarm, to make
  WAN latency survivable.
- **Phase 3 — Permissionless swarm.** One-command join, dynamic layer allocation
  across heterogeneous GPUs, per-token payouts.

Full detail, pass/fail criteria, and risks: [docs/ROADMAP.md](docs/ROADMAP.md).

## Repository layout

    phase0/   Working two-node split inference — node.py, node_kv.py, bench.py
    shard/    Engine modules — node, transport, specdec, scheduler (scaffolding)
    docs/     ARCHITECTURE, ROADMAP

## License

[Apache License 2.0](LICENSE) © 2026 leyten
