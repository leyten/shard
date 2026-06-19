# The c0mpute network: swarm inference on scattered consumer GPUs

*How the c0mpute network — the permissionless GPU network behind [c0mpute.ai](https://c0mpute.ai) —
is being reshaped so its nodes can **swarm**: pool into clusters that serve models far too big for
any single card. The engine that makes a swarm fast over the open internet is **Shard**. c0mpute is
the network; Shard is the engine.*

---

## The idea

Frontier AI runs in datacenters because frontier models don't fit on one GPU. c0mpute is building
the other path: a permissionless network where anyone plugs in a GPU, and those GPUs **swarm** —
pool together to serve a model far too big for any single card. No datacenter. No gatekeeper.
Uncensored, private, and the people who supply the compute get paid.

c0mpute already runs single-GPU inference across volunteer GPUs. This is the next layer — **swarm
inference**: many scattered consumer GPUs collectively serving a 120B or 744B model over the open
internet, fast enough to actually use.

And it's real, not a whitepaper. gpt-oss-120B at **~40 tok/s**, GLM-5.2 744B at **~30 tok/s**, on
consumer RTX 4090s and Blackwell cards scattered across US states — every run emits a verifiable
receipt (distinct GPU IDs, real WAN latencies, output hash). Here's how the network works.

## A swarm

A model is a stack of layers. Split the stack into contiguous blocks, put one block on each GPU,
and stream activations through them in order. No node ever holds the whole model. A swarm is:

- **Stage nodes** — each holds a contiguous block of layers (its *shard*).
- **A coordinator** — holds *no* model layers; it runs a small draft model and drives generation.

Activations flow `coordinator → stage 0 → stage 1 → … → tail → back to coordinator` — one loop per
step.

## The one law: latency dominates

Over the open internet the bottleneck isn't compute, it's the round-trip — every token traverses the
whole ring. So the network's #1 job isn't "find GPUs that fit," it's **assemble low-latency local
clusters**. We learned this the expensive way: moving the (layer-less) coordinator from across the
country *into* the swarm's region cut the ring **174 ms → 102 ms** and lifted throughput **~50%** —
for free. The entire topology is built around this single fact.

## Heterogeneous GPUs, interchangeable shards

Nodes are all different — 16 GB, 24 GB, 80 GB. So blocks are **sized to each node's VRAM**, not
uniform. And a consequence people miss: **prefer fat nodes.** One 80 GB GPU replaces three 24 GB
GPUs *and* removes two WAN hops — fewer hops = faster. Big GPUs don't just add capacity, they shorten
the ring.

A shard is just weights + kernels, so **the same block runs on any capable GPU** — which gives
redundancy, failover, and reassignment. Two honest caveats: the KV-cache for an in-flight request is
node-local (failing a node over mid-request means recompute, not seamless migration), and across GPU
architectures quantized outputs can differ at genuine near-ties (both valid — expected, not a bug).

## How a swarm self-assembles

The network keeps a **sparse, decaying latency graph** of who's close to whom (a full N² matrix
doesn't scale to thousands of nodes, and jitter makes stale data lie). To serve a model, the
scheduler:

1. Picks a **low-latency cluster** with enough total VRAM for the model + a coordinator.
2. Assigns contiguous blocks fit to each node's VRAM (fat nodes first → fewer hops).
3. Orders the ring to minimize the loop, and **places the coordinator in-region**.
4. Designates a coordinator/draft node (the draft must share the target's tokenizer family).

New nodes do **not** reshuffle running swarms — that's pure churn (reloading models, dropping
in-flight requests). They land in a **pool**, and are used to form new swarms or to **heal** one
(replace a dropped node). Global re-optimization runs on a slow cadence or when something breaks —
never on every join.

## How models reach the nodes — trustlessly

A node only needs *its* block, so each node holds and serves a fraction of the model. The
decentralized way to distribute weights is node-to-node propagation (BitTorrent-style) — but the
load-bearing piece isn't the transport, it's **content-addressing**. Every model ships a signed
**manifest**: the hash of every shard. A node fetches chunks from *any* peer and verifies each
against the manifest. In a permissionless network that's everything — **a malicious peer physically
cannot feed you corrupted weights.** Once chunks are hashed, peer-to-peer propagation is almost free
and gets faster as the network grows. We start hybrid (a seed mirror + manifest verification) and let
peer-propagation take over.

## How it stays fast

Splitting a model across the internet is easy; making it *usable* is the part Shard owns. The trick
is **speculative decoding**: a small draft proposes several tokens, the distributed model verifies
them all in one ring traversal, and greedy acceptance commits the verified prefix — so one
round-trip commits many tokens. Layer on **pipelining** (many traversals in flight at once) and the
WAN stops being the floor. That's how 36 layers across 4 states reach ~40 tok/s, greedy and
deterministic. ([receipts](receipts/) · [how a skeptic checks one](PROOF.md))

## Trust, pay, privacy — and what's still hard

c0mpute already has the economy: contribute compute, get paid per token. Swarms plug into the same
rails — each node earns for the tokens its block helped produce. The honest frontier:

- **Trustless verification.** How do you know a node actually ran its layer instead of forwarding
  plausible garbage? Redundant spot-checks + staking with slashing. This is the real research; we
  won't pretend it's solved.
- **Privacy.** A node decrypts to run its layer, so it sees the activations it processes. Mitigation:
  pin the leaky boundary layers (embedding, final layers) to staked/trusted nodes; route sensitive
  requests to trusted-only. "Private" earns its word phase by phase, never on day one.
- **Fault tolerance.** Recovering a request when a node vanishes mid-token (KV migration) is harder
  than failover-for-new-requests.

## The path

Centralized control plane first — the c0mpute orchestrator forms the clusters, allocates the blocks,
routes the requests. It holds no weights and no user data, so decentralizing it later (gossip,
elected schedulers) is a clean swap, not a rewrite. Ship the network that works today, remove the
trust assumptions one at a time, in the open, measured, never overclaimed.

**The foundation in one line:** a latency-aware clustering engine that assembles fat-node, in-region
rings from content-verified shards, and serves frontier models across GPUs nobody owns together.
Permissionless inference — anyone joins, anyone earns.
