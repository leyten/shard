# Shard × c0mpute — the done-right architecture

*The build spec for taking a hand-deployed swarm to a permissionless one, with no rip-out
debt. This is the internal engineering contract; [NETWORK.md](NETWORK.md) is the public
narrative. (Supersedes the transport/scheduler/integration sections of the older
[ARCHITECTURE.md](ARCHITECTURE.md), which predates the proven stack — SGLang→transformers,
aioquic→libp2p, shared PSK→per-node keys.) Build target: **gpt-oss-120B on 4× RTX 4090**,
the most practical swarm a stranger can join.*

---

## 0. The law

> **Dependencies point one way: `c0mpute → shard`, never the reverse.**

Shard is a pure engine: given a model and a set of peers, it swarms them to serve tokens
fast, with content-verified shards and per-node verifiable receipts. Shard knows **nothing**
about `$ZERO`, `privy_id`, USDC, payments, reputation, or the orchestrator. If shard ever
imports a c0mpute concept, the boundary leaked — and that's the rip-out debt we refuse.

This isn't tidiness. It's the proof of permissionlessness: an engine that doesn't depend on
the network's permission system can run under *any* network. **Shard is BitTorrent the
protocol; c0mpute is the swarm that speaks it.**

Every "which repo?" is answered by this rule. Runs on a contributor's GPU → shard. Network
brain (identity, money, reputation, tracker, catalog) → c0mpute. The wire between them → a
protocol contract, defined here, named on both sides.

---

## 1. The two halves

**shard (the engine — what a contributor installs):**
- The inference engine (pipeline split + speculative decoding + pipelining — proven: ~40
  tok/s gpt-oss-120B, ~30 tok/s GLM-5.2 744B over WAN).
- The **node-agent**: loads its assigned layer block, joins the ring, runs the block,
  forwards activations, emits signed per-batch receipts.
- The **transport sidecar** (libp2p) and the **layer-block challenge** primitive.

**c0mpute (the network brain):**
- The **tracker** (orchestrator): node pool, registration, job routing.
- The **scheduler**: forms swarms from the pool (the assignment protocol).
- The **latency graph**, the **swarm registry**, the **reputation** system, the **economy**
  (per-node payment), the **MODEL_CATALOG**.

**The contract:** the protocol messages in §7–§8. Implemented on both sides; owned by neither.

---

## 2. Identity (kills the shared PSK)

Today every node shares one `SHARD_PSK`. That's replaced by **per-node keys**:

- Each node generates a **libp2p keypair → PeerId** (stable network identity). This lives in
  shard — it's the engine's identity, nothing c0mpute-specific.
- **Binding to a c0mpute account:** at registration the node signs a challenge proving
  control of *both* its PeerId and its `cwt_`/`privy_id`. c0mpute records the binding. The
  node-agent only ever exposes "here is my PeerId + a signature"; c0mpute does the binding.
  Law preserved.

Result: authentication is per-node and cryptographic; `SHARD_PSK` retires.

---

## 3. Transport — libp2p via sidecar

A mature **Go/Rust libp2p daemon runs as a sidecar** on each node; the Python engine talks
to it over a local Unix socket / gRPC (local hop ≈ 0.1 ms, nothing against a WAN ring).

What we use libp2p for:
- **Noise/TLS encryption + QUIC** on every link.
- **NAT traversal** — AutoNAT detects reachability, **DCUtR hole-punching**, **circuit-relay-v2**
  fallback for hard NATs. *This is the whole point — home GPUs behind NAT are the thesis.*
- **Kademlia DHT** for peer discovery + content routing.
- **Direct streams** for the activation hot-path between adjacent ring stages. Gossipsub is
  for control/discovery only — **never on the hot path.**

The sidecar replaces `phase0/wire.py`'s role (PSK-authenticated TCP). The sidecar owns
connection + NAT; the engine streams activation bytes through it.

---

## 4. Model propagation — content-addressed

A node only needs *its* block, so it only fetches a fraction of the model.

- **Manifest** (signed JSON): `{model_id, layer_count, tokenizer, arch, shards:[{shard_id,
  sha256, size}], publisher_pubkey, signature}`.
- A node fetches its shards from **any provider** via libp2p content routing and **verifies
  each chunk against the manifest hash** on arrival. A malicious peer physically cannot feed
  you corrupted weights.
- **Catalog:** c0mpute's `MODEL_CATALOG` holds a pointer to the manifest (CID/URL +
  publisher pubkey). Adding a model = publish a manifest + add a catalog entry.

**Propagation seam:** the source is pluggable. Now → a seed **mirror is just the first
provider**. Later → peers announce the shards they hold and **P2P takes over** — additive,
zero rework, because the fetch was content-verified from day one.

---

## 5. The swarm engine (shard — proven)

A model is a stack of layers split into contiguous blocks, one block per GPU; activations
stream through in a ring. A **coordinator** holds no layers, runs a small draft, and drives
**speculative decoding** (draft proposes K tokens → distributed model verifies all in one
ring traversal → greedy commit), with **pipelining** (many traversals in flight). The
node-agent's job per swarm: load its block, link its ring neighbors via the sidecar, run the
block, forward activations, **sign a receipt per batch**.

---

## 6. Receipts + verification (the canary, upgraded for swarms)

c0mpute's current anti-cheat is a whole-model canary: send a math+nonce prompt, check the
answer. **It cannot probe a stage-node** — a node holding layers 12–23 never sees a prompt,
it transforms an activation tensor. Swarms need two new things, and they split cleanly across
the law:

**(a) Signed receipts (shard emits, c0mpute consumes).** Each stage node signs
`{swarm_id, job_id, batch_id, layer_range, in_hash, out_hash}` with its node key. The
coordinator collects receipts and submits them on `job:complete`. Payment integrity falls
out for free: c0mpute pays each node per its signed receipts; the coordinator **can't
fabricate** a node's receipt (needs its key), and a node **can't be paid** without producing
one. This kills coordinator-takes-all and stops the coordinator stealing pay.

**(b) Layer-block challenge (shard provides the primitive, c0mpute owns the policy).** A
verifier feeds a stage-node a **known activation** whose correct `out_hash` was computed by
re-running that block on a trusted/redundant node. Mismatch → strike. Shard exposes only
"run this block on this input → output hash"; *when* to probe, *how* to score, *when* to
eject is c0mpute policy. This is the stage-node analogue of the existing canary.

**Reputation upgrade (c0mpute).** Today reputation is binary (`banned` / not). Swarms need a
**graded score** the scheduler consumes — to prefer reliable nodes, to **pin leaky boundary
layers (embedding/final) to the most-trusted nodes** (privacy), and to gate who may
coordinate. The existing recent-window ban logic stays; we add the gradient on top.

**Verification seam:** economic-now (strike → reputation hit → eject + withhold pay) →
crypto-later. The receipt's `in_hash/out_hash` slot is exactly where a cheap proof drops in.

---

## 7. The scheduler / tracker (c0mpute — the centralized seam)

Holds the node pool, a **sparse, decaying latency graph** (full N² doesn't scale; stale
jitter lies), the swarm registry, reputation, and the catalog. To serve a model:

1. Pick a **low-latency cluster** with enough VRAM for the model + a coordinator.
2. Fit **contiguous blocks to each node's VRAM, fat nodes first** (fewer hops).
3. Order the ring to minimize the loop; **place the coordinator in-region** (this lever was
   worth ~50% — 174→102 ms).
4. Assign.

**`swarm:assign`** → each chosen node:
```
{ swarm_id, manifest_ref, layer_start, layer_end, role: "stage"|"coordinator",
  peers: [{ peer_id, addr, layer_range }], coordinator_peer }
```
Node fetches its shards (if uncached), loads, links neighbors, signals ready. All ready →
swarm live.

**Heal, don't reshuffle.** A dropped node → pull a replacement from the pool, reload its
block, re-link the ring. New joins land in the **pool**, never reshuffle a running swarm
(pure churn). Global re-optimization runs on a slow cadence or on break, never per-join.

**Control-plane seam:** centralized first. It holds **no weights and no user data**, so
decentralizing it later (gossip, elected schedulers) is a clean swap, not a rewrite.

---

## 8. Job flow + payment (per-node)

```
client → c0mpute tracker → job:new → coordinator
coordinator runs coordinate_pipe over the ring
  → job:token (stream) → client
  → job:complete { response, tokensGenerated, receipts[] } → tracker
tracker verifies receipts → attributes tokens per node → pays via worker_earnings
```

Each node earns for the tokens its block helped produce (coordinator earns for draft +
drive). Same c0mpute rails, no coordinator-takes-all.

---

## 9. The seams, stated honestly

| Seam | Now (correct, not debt) | Later (clean swap) |
|---|---|---|
| Control plane | centralized scheduler (holds no weights/data) | gossip / elected |
| Verification | economic: strike + eject + withhold pay | cryptographic proof in the hash slot |
| Propagation | seed mirror = first provider | P2P peers take over (additive) |

None of these is an "in-between" that gets ripped out — each is the right interface with a
capability switched on later.

---

## 10. Build sequence — done right, exercised early

No throwaway code, but a real swarm runs *soon* so the design gets pressure-tested.

1. **Identity + handshake** — per-node keys ⟷ `privy_id` binding; retire `SHARD_PSK`.
2. **libp2p sidecar + activation data-plane** — replace `wire.py`; 2-node swarm over libp2p,
   LAN → NAT. *First NAT-traversed link.*
3. **Manifest + content-addressed fetch** — mirror as first provider; node pulls + verifies
   its block.
4. **Scheduler (central) + assignment** — auto bring-up a swarm from a pool. *First fully
   automatic swarm.*
5. **Job routing + signed receipts + per-node payment** — **live & earning on c0mpute.**
6. **Reputation upgrade + layer-block spot-check** — trust hardening.
7. **Heal + mid-request fault tolerance.**
8. **P2P propagation** takes over from the mirror (additive).

Each step leaves a running, correct system.

---

## 11. What's genuine research (not pretending otherwise)

- **Layer-block spot-check at scale**, and eventually a cheap cryptographic proof to replace
  re-compute-and-compare.
- **Mid-request KV fault tolerance** — a node vanishing mid-token means recompute, not
  seamless migration.
- **Decentralized scheduling** — the control-plane seam, when we take it.
- **Privacy** — boundary-layer pinning + trusted-only routing; earns its word phase by phase.

Everything before those is engineering we've already proven variants of.
