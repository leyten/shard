# Shard — build status

One glance, full picture. The whole network is **5 verbs**. SERVE is done; we're building the other four.

## The map
1. **JOIN**  — a stranger's GPU gets in *(identity, NAT transport, pull its slice of weights)*
2. **FORM**  — the network picks nearby nodes and wires them into a swarm *(scheduler, assignment, heal)*
3. **SERVE** — the swarm answers the request, fast — ✅ **DONE** (~40 tok/s gpt-oss-120B, ~30 GLM-5.2 744B over WAN)
4. **PROVE** — each node proves it actually ran its layer *(signed receipts, layer-block spot-check)*
5. **PAY**   — each node gets paid for its bit *(per-node, c0mpute rails)*

Every line in [docs/INTEGRATION.md](docs/INTEGRATION.md) is just one of these five, done right.

## Build steps  (→ verb · status)
| # | Step | Verb | Status |
|---|------|------|--------|
| 0 | Engine (pipeline + spec-decode + pipelining) | SERVE | ✅ done |
| 1 | libp2p sidecar + per-node identity + data-plane (retire `SHARD_PSK`) | JOIN | ✅ **done** |
| 2 | NAT traversal + bind identity ↔ c0mpute account | JOIN | ◀ next |
| 3 | Manifest + content-addressed weight fetch | JOIN | todo |
| 4 | Scheduler + assignment protocol | FORM | todo |
| 5 | Job routing + signed receipts + per-node pay | PROVE/PAY | todo |
| 6 | Reputation upgrade + layer-block spot-check | PROVE | todo |
| 7 | Heal + mid-request fault tolerance | FORM | todo *(research)* |
| 8 | P2P propagation takes over from mirror | JOIN | todo *(additive)* |

## Now
**Step 1 (JOIN transport) DONE.** ✅ The real gpt-oss-120B, split across 4 scattered boxes (UT·CA·NV·WA) over **libp2p with per-node keys and no `SHARD_PSK`**, produced **bit-identical** greedy tokens to the committed `wire.py` receipt (sha `f646e0db…3f70`, 87 tokens). Proven incrementally: 1.1 key-auth round-trip → 1.2 engine↔sidecar tensors → 1.3a transparent TCP-over-libp2p tunnel → 1.3b PSK-free message codec → 1.3d-i cross-box libp2p over real WAN → 1.3d-ii the full 120B ring. Sidecar = `sidecar/main.go`; engine wire = `shard/transport.py`; the engine ran unmodified except `import wire → import shard.transport as wire`.

**Loose ends before committing step 1:**
- Prune the superseded 1.2 unix-socket bridge (sidecar `-engine` mode + `Edge`/`ActivationCodec`) — the tunnel replaced it.
- Re-enable the **perf path** over the sidecar (direct-return + pipelining → ~40 tok/s; the relay-back sync proof ran at 22.9 tok/s). The 2-connection tail is the only fiddly bit. Correctness-independent.
- Land a libp2p receipt (mirrors the wire.py one) as the banked proof.

**Next (step 2):** NAT traversal (home GPUs behind NAT — DCUtR + relay; today used Vast's mapped public ports) + bind each node's libp2p key ↔ its c0mpute `cwt_` account.

## Decisions locked
- **Boundary law:** dependencies point one way — `c0mpute → shard`, never reverse. Shard is a pure engine.
- **Transport:** libp2p via a **Go** (`go-libp2p`) sidecar; Python engine talks to it over a local Unix socket.
- **Identity folds into the libp2p step** (libp2p gives keypair identity for free — a separate identity layer would be throwaway).
- **Verification:** graded reputation + a layer-block challenge (canary-style); economic-now (eject + withhold pay) → crypto-later.
- **`$ZERO` staking** = yield only, no slashing — orthogonal to verification, left out of it.
