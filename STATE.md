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
| 2 | NAT traversal + bind identity ↔ c0mpute account | JOIN | ◀ **building** |
| 3 | Manifest + content-addressed weight fetch | JOIN | todo |
| 4 | Scheduler + assignment protocol | FORM | todo |
| 5 | Job routing + signed receipts + per-node pay | PROVE/PAY | todo |
| 6 | Reputation upgrade + layer-block spot-check | PROVE | todo |
| 7 | Heal + mid-request fault tolerance | FORM | todo *(research)* |
| 8 | P2P propagation takes over from mirror | JOIN | todo *(additive)* |

## Now
**Step 1 (JOIN transport) DONE.** ✅ The real gpt-oss-120B, split across 4 scattered boxes (UT·CA·NV·WA) over **libp2p with per-node keys and no `SHARD_PSK`**, produced **bit-identical** greedy tokens to the committed `wire.py` receipt (sha `f646e0db…3f70`, 87 tokens). Proven incrementally: 1.1 key-auth round-trip → 1.2 engine↔sidecar tensors → 1.3a transparent TCP-over-libp2p tunnel → 1.3b PSK-free message codec → 1.3d-i cross-box libp2p over real WAN → 1.3d-ii the full 120B ring. Sidecar = `sidecar/main.go`; engine wire = `shard/transport.py`; the engine ran unmodified except `import wire → import shard.transport as wire`.

**Perf path re-enabled (direct-return + pipelining over libp2p):** **44.79 tok/s warm @ depth 2**, bit-identical (sha `f646e0db…3f70`, `tokens_match_sync=True`) — i.e. **parity-or-better vs the trusted-wire 39.8** (this window's return leg was 45 ms). Sweep: PIPE d2 warm 44.8 / d4 warm 39.3 / SYNC warm 33.5. The fix was a latent race in `serve_tail_fast` — it now identifies the return channel by content (`hello_return`), not arrival order. So libp2p adds no real tax; QUIC stays a step-2 lever, not needed for parity.

Done & committed: prune of the dead 1.2 bridge, the libp2p receipt, the tail fix.

**Step 2 (building) — NAT + identity binding.**
- ✅ **2.1** sidecar NAT stack: QUIC + DCUtR + circuit-relay-v2 (service + client) + AutoNAT + `-announce` + explicit `client.Reserve` + conn monitor (RELAY/DIRECT). On **go-libp2p v0.48** / Go 1.25.11. `sidecar/main.go`.
- ✅ **2.2** relay join AND direct hole-punch both PROVEN. Relay: a genuinely NAT-blocked node reserves a relay slot + data crosses both ways (real boxes + lab). Direct line: built a controlled two-NAT lab with Linux netns (`/tmp/netlab.sh` + `/tmp/holepunch.py`) — **two nodes each behind their own NAT formed a DIRECT QUIC line via DCUtR and moved 100 KB byte-identical** (relay caps ~2 KB, so 100 KB proves it went direct). Required: go-libp2p v0.48, full-cone (UPnP-style) NAT, and a *routable* IP range — TEST-NET (203.0.113.x) is silently `blocked observed address` by libp2p; use real public ranges (11.0.0.x). The earlier "datacenter Docker NAT un-punchable" finding stands (that box is harsher than a home router) — but a full-cone home router punches through; restricted/symmetric fall back to the (proven) relay.
- ☐ **2.3** identity ↔ `cwt_` binding: node signs a challenge proving control of (PeerId, cwt_); c0mpute records it (c0mpute-repo change — shard signs, c0mpute records). Unblocked.

## Decisions locked
- **Boundary law:** dependencies point one way — `c0mpute → shard`, never reverse. Shard is a pure engine.
- **Transport:** libp2p via a **Go** (`go-libp2p`) sidecar; Python engine talks to it over a local Unix socket.
- **Identity folds into the libp2p step** (libp2p gives keypair identity for free — a separate identity layer would be throwaway).
- **Verification:** graded reputation + a layer-block challenge (canary-style); economic-now (eject + withhold pay) → crypto-later.
- **`$ZERO` staking** = yield only, no slashing — orthogonal to verification, left out of it.
