# shard

one 100B+ model, running on gpus nobody owns together.

shard is c0mpute's decentralized inference engine. it splits a model that is far too big for any single consumer gpu into shards, one block of layers per gpu, scattered across the internet, and serves real tokens out of the reassembled whole. no datacenter, no single owner, no central inference server.

the name: the model is broken into shards, one per gpu. the swarm reassembles them into a single working model for the length of a request, then they're just shards again. and it carries the privacy story for free: no single node ever holds the whole model, only its shard.

## why this exists

a 100B model needs ~200GB of memory. a consumer gpu has 24. so one card can't hold it. but four can, if you split the model into blocks of layers and pipe a token through them in order. that part is not new and it mostly works.

the hard part is doing it over the open internet, between machines behind home routers, fast enough to be usable. that is where the existing tools fall down, and that is the part shard is built to own.

## the two bets

1. **speculative decoding over the swarm.** the wall over wan is latency: every generated token has to traverse every node, so at 50-80ms per hop you get 1-2 tok/s and it's useless. the fix (from the paper this is based on) is to run a small draft model locally, have it guess several tokens, and let the big split model verify all of them in a single trip through the swarm. that turns one round-trip per token into one round-trip per several tokens, and makes wan latency survivable.

2. **a transport we actually own.** moving the activations between gpu stages, reliably, across messy nat, is the thing that breaks in practice. shard owns that layer instead of treating it as a black box: direct quic between stages with hole-punching, a relay fallback, quantized activations to cut bandwidth, and real backpressure and reconnection.

## the three pillars

shard is c0mpute infra, so it is vetted against all three:

- **uncensored** — the engine runs models as-is. no content filter in the inference path.
- **decentralized** — anyone can join a gpu with one command and get assigned a shard of layers. the scheduler is light and replaceable, there is no single inference server.
- **private** — no node holds the whole model, which is a real start. but intermediate activations can still leak a meaningful fraction of the user's tokens to a malicious node in the pipeline. see `docs/ARCHITECTURE.md#privacy` for the options. it is the number one thing to get right.

## status

**phase 0 done.** a model split across two nodes serves coherent tokens over our own transport, with a per-node kv-cache. proven on the case that actually needs two gpus: Qwen2.5-14B is 29.5GB in bf16 and OOMs a single 24GB 4090, but split 24/24 layers across two cards (~14.8GB each) it runs reliably — 20/20 clean completions, ~17 tok/s decode. this is the exact thing nothing off-the-shelf could do.

honest caveats: the two nodes are still on a low-latency link (same-host bridge), not real wan; no speculative decoding yet. those are next — wan/nat transport (phase 1), then spec-decode (phase 2). see `docs/ROADMAP.md`.

## layout

- `docs/ARCHITECTURE.md` — the full technical design
- `docs/ROADMAP.md` — phased build plan, milestones, risks
- `shard/` — the engine (stubs for now, one module per layer of the design)
