# Shard — build status

One glance, full picture. The whole network is **5 verbs**. Engine side: JOIN, FORM, SERVE, and
the PROVE primitives are now in; only PAY (c0mpute rails) and the live integration remain.

## The map
1. **JOIN**  — a stranger's GPU gets in *(identity, NAT transport, pull its slice of weights)* — ✅ **DONE** (steps 1–3): libp2p identity, NAT, and content-addressed verified weight fetch.
2. **FORM**  — the network picks nearby nodes and wires them into a swarm *(scheduler, assignment, heal)* — ✅ **engine done**: `shard/scheduler.py` auto-fits the model to heterogeneous VRAM (fat node first) + RTT-orders the ring; heal-by-rebuild. Live control-plane integration is next.
3. **SERVE** — the swarm answers the request, fast — ✅ **DONE incl. long context**: ~40 tok/s short-ctx, and **28.2 tok/s decode at >100k context** via n-gram spec-decode (2026-06-23, receipt above).
4. **PROVE** — each node proves it actually ran its layer — ✅ **primitives done + receipts demonstrated live**: signed per-stage receipts (`shard/receipt.py`, wired into the serve loop, in/out roots chain across the ring) + a tolerance-based layer-block challenge (`shard/challenge.py`). Reputation/policy is c0mpute-side.
5. **PAY**   — each node gets paid for its bit *(per-node, c0mpute rails)* — the remaining piece, c0mpute-side (per-node `worker_earnings` keyed on verified receipts).

Every line in [docs/INTEGRATION.md](docs/INTEGRATION.md) is just one of these five, done right.

## ≥20 tok/s at >100k context — DONE (2026-06-23): 28.2 tok/s, n-gram spec-decode
The "fast generation at long context is the open piece" is now closed. On a **3-node WAN swarm
(WA·WA·TX, all distinct hosts/GPUs)** a ~107k-token prompt decoded at **28.18 tok/s** — past the
20 tok/s target — greedy-exact, correct output ([receipt](docs/receipts/gpt-oss-120b-100k-ngram-20260623.json)).
The levers, all on the `--fast` path:
- **Model-free n-gram (prompt-lookup) spec-decode** (`phase0/ngram_draft.py`, wired into
  `coordinate_pipe` as `local_draft`). The 20B draft OOMs at 100k; n-gram needs NO model and NO KV.
  The win is **longest-suffix matching**: a generic 2-gram has many homes across a 107k context, so
  pick the earlier occurrence whose *preceding* context matches longest — that uniquely locates the
  region being copied. Indexing only the *committed* prefix (not the speculative tail) keeps it
  **O(1)/round** after a one-time scan (was 130ms/round with a naive rebuild). g≈7.6 tok/traversal.
- **Window-KV read** (`FV_WINDOW=1`): gpt-oss's sliding-attention layers read only their 128-key
  window at decode, O(window) not O(ctx) — ~halves the per-round compute. Numerically-equivalent
  (ULP), opt-in.
- **VRAM-aware uneven split** (`load_stage --lo/--hi`): the 48GB box holds 18 layers, the two 24GB
  boxes 9 each — fits 100k KV on heterogeneous cards (the `shard/scheduler.py` allocator auto-derives this).
- **Tail prefill-logit fix**: the tail ran `lm_head` over the whole 4096-token prefill chunk (a ~1.5GB
  logit tensor → 24GB OOM); now only the last token's logit (the only one consumed). Faster + no OOM.
- **`reasoning=low`** + a copy/retrieval-style task (the latency-tolerant long-context demand: code
  retrieval, extraction, RAG-with-citation): the output reuses the context, which is where n-gram wins.
  On novel-prose generation n-gram falls back toward the plain-decode floor (~1.8 tok/s) — stated honestly.
Config: K=16, depth=2, ngram-n=3, max-ctx 131072. Bring-up: `phase0/launch_ngram.py`.

## 100k context (2026-06-20) — PROVEN for prompts; fast generation is the open piece
The "long context is a dealbreaker" worry, resolved on the real rig
([receipt](docs/receipts/gpt-oss-120b-95k-context-20260620.json), [feasibility](docs/receipts/gpt-oss-120b-100k-feasibility-20260620.json)):
- **A distributed gpt-oss-120B swarm (N=4 scattered 4090s) prefilled a 95,690-token prompt and
  correctly comprehended it** — no OOM, prefill **207 tok/s** (~8 min). The memory wall is gone.
- **How:** the gpt-oss attention sink blocks sdpa and the flash-sink kernel is Hopper-only, so eager
  was the only option and it OOMs ~8k. **flex_attention** is Ada-native, handles sinks+sliding, and
  is O(n) — validated cosine 0.9996 vs eager. Plus **chunked prefill** (bounded per-chunk activations,
  KV accumulates) and an N=4 split so 95k KV fits a 24GB card. Wired in `pipeline.run_block` +
  `specpipe` (`--attn flex_attention`, `--prompt-file`, `--prefill-chunk`).
- **Decode at long ctx — usable.** Naive decode was 0.51 tok/s @95k (flex recompiled every step as KV
  grew). Fix: **flex for prefill, eager for decode.** Decode flips the layers to eager and uses the
  existing CUDA-graphed path (q=K+1 eager attention is cheap even over 100k keys; fixed-shape replay, no
  recompile); flex only does the big-query prefill. End to end on the N=4 swarm: **92k prompt, prefill
  157 tok/s, decode 3.5 tok/s** (7× the naive path), correct output
  ([receipt](docs/receipts/gpt-oss-120b-100k-decode-20260620.json)). Usable for long-context (non-interactive).
- **More decode speed (next):** sliding-window ring-buffer KV (half the layers only need a 128-key read →
  ~2×), spec-decode (needs a long-context-tuned draft; the vanilla 20B draft OOMs at 100k), fp8 KV, fewer stages.

## Serving hardening (2026-06-20) — long-context + losslessness, real 4×4090 runs
A reported "output breaks past ~20k context / spec-decode degrades quality" sent us back to the rig
(4 scattered RTX 4090s: CA·NC·WA stages + Utah coord/draft). Findings, all from live runs
([receipt](docs/receipts/gpt-oss-120b-context-20260620.json)):
- **"Breaks past 2048" was TWO independent hard walls, both now fixed.** (1) The `--fast` path's static
  KV cache was `maxlen=2048` with **no bounds check** → silent CUDA corruption past it. Fix: **`--max-ctx`**
  sizes the cache + a **`ContextOverflow`** guard fails clean (proven: 2480-tok prompt at max_ctx=2048 →
  named error, stage stays up). (2) The vLLM draft ran `max_model_len=2048` → generation past 2048 hung the
  draft. Fix: launchers pass **`--max-len`**. **Proof:** with both lifted, the swarm sustained greedy decode to
  **2633 context (585 past the old wall) at 23 tok/s** (sha `8a70cdee…`).
- **Two remaining ceilings (quantified, not fixed):** eager **prefill** of a >~2k-token *prompt* OOMs on a
  24GB/12-layer card (~3.2 GB transient, caught cleanly); and the per-step mask is **O(max_ctx)** so decode
  slows with context (36.7 tok/s @2k → 22 @10k). KV cache ≈72 KB/token; safe `max_ctx`≈10k. **20k needs**
  flash/chunked prefill + incremental mask + fp8 KV, or more stages (fewer layers/box).
- **Losslessness, adjudicated:** **fixed-K is deterministic** (K4 vs K4 bit-identical; matches the trusted-wire
  receipt) — but **cross-K diverges at a floating-point near-tie** (K4 vs K8 first-diff @ token 92), FP
  non-associativity in the batched CUDA-graph verify, **not** quality degradation. Each output is a valid greedy
  decode. So the critic's claim is half-right and reframed: spec-decode ≠ lossy; it's cross-K FP non-determinism.
- **Also fixed:** the swarm bring-up's zombie-draft teardown (a `pkill -f draft_server` was self-matching the
  kill shell → boxes never freed; now kills by port+GPU pid). Noted gap: ring **forward links don't auto-recover**
  on coordinator churn (fault-tolerance, step 7).

## Build steps  (→ verb · status)
| # | Step | Verb | Status |
|---|------|------|--------|
| 0 | Engine (pipeline + spec-decode + pipelining) | SERVE | ✅ done |
| 1 | libp2p sidecar + per-node identity + data-plane (retire `SHARD_PSK`) | JOIN | ✅ **done** |
| 2 | NAT traversal + bind identity ↔ c0mpute account | JOIN | ✅ **done** |
| 3 | Manifest + content-addressed weight fetch | JOIN | ✅ **done** (`shard/manifest.py`, `shard/fetch.py`, validated, path-traversal-hardened) |
| 4 | Scheduler + assignment protocol | FORM | ✅ **engine done** (`shard/scheduler.py` VRAM-fit + RTT order; live bring-up integration next) |
| 5 | Job routing + signed receipts + per-node pay | PROVE/PAY | ◑ **receipts done + live** (`shard/receipt.py`); per-node pay = c0mpute rails |
| 6 | Reputation upgrade + layer-block spot-check | PROVE | ◑ **challenge primitive done** (`shard/challenge.py`); reputation = c0mpute |
| 7 | Heal + mid-request fault tolerance | FORM | todo *(research)* |
| 8 | P2P propagation takes over from mirror | JOIN | todo *(additive)* |

## Now
**Step 1 (JOIN transport) DONE.** ✅ The real gpt-oss-120B, split across 4 scattered boxes (UT·CA·NV·WA) over **libp2p with per-node keys and no `SHARD_PSK`**, produced **bit-identical** greedy tokens to the committed `wire.py` receipt (sha `f646e0db…3f70`, 87 tokens). Proven incrementally: 1.1 key-auth round-trip → 1.2 engine↔sidecar tensors → 1.3a transparent TCP-over-libp2p tunnel → 1.3b PSK-free message codec → 1.3d-i cross-box libp2p over real WAN → 1.3d-ii the full 120B ring. Sidecar = `sidecar/main.go`; engine wire = `shard/transport.py`; the engine ran unmodified except `import wire → import shard.transport as wire`.

**Perf path re-enabled (direct-return + pipelining over libp2p):** **44.79 tok/s warm @ depth 2**, bit-identical (sha `f646e0db…3f70`, `tokens_match_sync=True`) — i.e. **parity-or-better vs the trusted-wire 39.8** (this window's return leg was 45 ms). Sweep: PIPE d2 warm 44.8 / d4 warm 39.3 / SYNC warm 33.5. The fix was a latent race in `serve_tail_fast` — it now identifies the return channel by content (`hello_return`), not arrival order. So libp2p adds no real tax; QUIC stays a step-2 lever, not needed for parity.

Done & committed: prune of the dead 1.2 bridge, the libp2p receipt, the tail fix.

**Step 2 (JOIN — NAT + identity) DONE.**
- ✅ **2.1** sidecar NAT stack: QUIC + DCUtR + circuit-relay-v2 (service + client) + AutoNAT + `-announce` + explicit `client.Reserve` + conn monitor (RELAY/DIRECT). On **go-libp2p v0.48** / Go 1.25.11. `sidecar/main.go`.
- ✅ **2.2** relay join AND direct hole-punch both PROVEN. Relay: a genuinely NAT-blocked node reserves a relay slot + data crosses both ways (real boxes + lab). Direct line: built a controlled two-NAT lab with Linux netns (`/tmp/netlab.sh` + `/tmp/holepunch.py`) — **two nodes each behind their own NAT formed a DIRECT QUIC line via DCUtR and moved 100 KB byte-identical** (relay caps ~2 KB, so 100 KB proves it went direct). Required: go-libp2p v0.48, full-cone (UPnP-style) NAT, and a *routable* IP range — TEST-NET (203.0.113.x) is silently `blocked observed address` by libp2p; use real public ranges (11.0.0.x). The earlier "datacenter Docker NAT un-punchable" finding stands (that box is harsher than a home router) — but a full-cone home router punches through; restricted/symmetric fall back to the (proven) relay.
- ✅ **2.3** identity ↔ `cwt_` binding — proven end-to-end, cross-language. **shard signs:** `sidecar -prove <nonce>` → {PeerId, sig} with the node key (`-verify` is the reference check). **c0mpute verifies + records:** `c0mpute/lib/identity.ts verifyBindingProof` (Node-native ed25519 + inline base58/PeerId decode, zero new deps), `lib/db.ts` `worker_identity` table + `bindPeerId`/`getPeerIdOwner`, `app/api/node-bind/route.ts` (cwt_-auth + HMAC challenge nonce). Tested: Go signs → TS verifies (correct=true, tampered/impersonated=false), bind→lookup round-trips. c0mpute-side committed in the c0mpute repo (not shard).

**Step 2 done.** JOIN's hard parts are in: engine on the wire @ ~45 tok/s, NAT-traversable, per-node paid identity. Remaining JOIN piece: **step 3** (content-addressed weight fetch — how a node pulls its layer block, trustlessly). Then **FORM** (step 4 scheduler/assignment).

## Decisions locked
- **Boundary law:** dependencies point one way — `c0mpute → shard`, never reverse. Shard is a pure engine.
- **Transport:** libp2p via a **Go** (`go-libp2p`) sidecar; Python engine talks to it over a local Unix socket.
- **Identity folds into the libp2p step** (libp2p gives keypair identity for free — a separate identity layer would be throwaway).
- **Verification:** graded reputation + a layer-block challenge (canary-style); economic-now (eject + withhold pay) → crypto-later.
- **`$ZERO` staking** = yield only, no slashing — orthogonal to verification, left out of it.
