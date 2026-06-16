# WAN-optimal speculative decoding

*Research track. Status: active (2026-06-16). This is a design record — the
evidence that motivated it and the plan it justifies, in the spirit of an RFC.*

## Why this track exists

The phased [roadmap](../ROADMAP.md) was a hypothesis about where the hard parts
were. Phase 2 (speculative decoding) and the 100B+-over-WAN demo are done — and
in building them we *measured* something the roadmap didn't anticipate. This doc
records that finding and the research it justifies. Changing course on measured
evidence is the roadmap working as designed, not straying from it.

## The measured problem

Spec-decode on gpt-oss-120b, split 4 stages across two machines over a real
transatlantic link, warm steady-state, fixed K=4. Per spec round:

```
draft  309 ms   (5 forwards of the gpt-oss-20b draft, ~62 ms/tok)
verify 224 ms   (one pipeline traversal: WAN round-trip + node compute)
-----  533 ms   -> 3.49 tokens committed -> 6.6 tok/s
```

Two facts fall out of this:

1. **The WAN is not the wall.** The verify traversal is only 224 ms. If the draft
   were free, that alone gives `3.49 / 0.224 = 15.6 tok/s` on the *current* setup;
   with more tokens per traversal (below), the verify-bound ceiling is ~25–30.
2. **The draft is the wall.** 309 of the 533 ms is the draft, and it cannot be
   made cheaper *in this stack on consumer GPUs*: gpt-oss is MXFP4-MoE, and the
   measured options are all closed on Ada (RTX 4090) —
   - `sdpa`: not implemented for gpt-oss in transformers 5.6.
   - `torch.compile`: Inductor can't trace the MXFP4 triton kernels (`InductorError`).
   - flash: the attention-sink ("S aux") kernel is **Hopper-only**; fa4 not packaged.

   The draft's actual arithmetic is ~2 ms; the other ~60 ms is kernel-launch /
   Python / MXFP4-dequant / MoE-routing overhead the stack won't remove.

## The insight

**Over WAN the round-trip is the scarce resource.** This inverts the optimization
versus datacenter inference: techniques that are marginal in a datacenter become
the whole game. Two levers, in forced order:

1. **Make the draft cheap.** It's the wall, and *nothing else helps until it falls*
   (see below — tree speculation actively *hurts* while the draft is expensive).
2. **Then maximize tokens per traversal** — tree speculation. The verify's WAN cost
   is fixed per traversal regardless of how many candidate tokens it carries, so
   once drafting is cheap you stuff the traversal with a branching tree and accept
   the best path. This is where the WAN regime makes spec-decode shine.

Why the order is forced: we are draft-bound (309 > 224). A tree means more draft
forwards, so while the draft is expensive a tree raises cost faster than tokens —
a measured regression. Tree spec is a multiplier on a *cheap* draft, not a fix for
an expensive one.

## Paths to a cheap draft (increasing effort)

- **P1 — optimized kernels for the 20b draft (no training, keeps exactness).**
  vLLM/SGLang ship Ada-compatible gpt-oss kernels + CUDA graphs; gpt-oss-20b runs
  fast on 4090s under vLLM in the wild. Target ~12–15 ms/tok (≈4× cheaper). The
  integration wrinkle is cache rollback on rejection (the draft must propose K then
  rewind on a reject), which vLLM's internal cache isn't built to expose — so this
  is "use their kernels in our decode loop," not "call their server." **De-risk
  first:** benchmark gpt-oss-20b decode under vLLM on a single 4090.

- **P2 — EAGLE-style draft head (training; the strongest endpoint).** Train a
  ~1-layer head on the *120b's own hidden states*; it predicts the next feature,
  the frozen target LM head turns it into a token. Both cheap (~5 ms/tok) and
  higher-acceptance than an off-the-shelf 20b. It fits the pipeline beautifully:
  the head node needs the target's last hidden state, which the tail already
  computes — so the feature piggybacks on the token the tail returns each round.
  Cost is data collection (running the 120b to dump features), amortizable because
  every served request generates training features for free.

- **P3 — distilled small dense draft.** A tiny model with the gpt-oss tokenizer,
  distilled from the 120b. Training project; loses the MoE/MXFP4 overhead entirely.

## Tree speculation (phase 2 of this track)

Draft a *tree* of continuations (branch at uncertain positions), flatten it, send
it through the pipeline in one traversal with a tree-structured attention mask
(each node attends only to its ancestors), and accept the longest valid root-to-leaf
path. The verify returns logits for every tree node; the head walks the tree. The
cache keeps the accepted path and discards the rest. Greedy acceptance keeps output
exact regardless of tree shape.

## Success criteria

**20 tok/s, exact output, on 4 non-co-located consumer GPUs (4090 each) + a draft,
serving gpt-oss-120b over real WAN.** Honest intermediate target: ~12–15 tok/s from
a cheap draft alone (P1) before tree spec.

## The arithmetic that says it's reachable

With a cheap draft (`~12 ms/tok`) and tree speculation (`~6 tokens/traversal`):
`6 / (verify 224 ms) ≈ 27 tok/s` on the current 2-machine setup. For the harder
4-non-co-located case the verify grows (3 forward hops vs 1 edge), which is exactly
why tree spec is mandatory there — it keeps tokens-per-traversal high enough to beat
the deeper pipeline. The physics permits 20; the draft cost is the only thing in
the way, and it has known fixes.

## Status / log

- **2026-06-16** — Measured the round breakdown (above); identified the draft-cost
  wall and confirmed off-the-shelf kernel paths are closed on Ada. Decision: pursue
  a cheap draft (P1 first, de-risk vLLM-on-Ada), then tree speculation. Instrumented
  `phase0/specpipe.py` with per-round `draft_ms`/`verify_ms` + an async-ceiling
  readout.
- **2026-06-16 — P1 de-risked, and it lands.** gpt-oss-20b under vLLM 0.23.0 on a
  single RTX 4090 (MXFP4, CUDA graphs): **5.1 ms/tok vs transformers' 61.7 ms/tok —
  a 12× cheaper draft, no training.** The ~60 ms overhead was a transformers limit,
  not a hardware one. Recomputing the round with a 5.1 ms draft (K=4, draft ≈ 26 ms
  + verify 224 ms): **~14 tok/s, up from 6.5** — and the round is now *verify-bound*,
  which is exactly where tree speculation and faster node kernels pay off. Decision:
  **P1 is the draft path (no EAGLE training needed yet).** Next: integrate a
  vLLM-served draft into the spec loop (rollback handled by vLLM prefix-caching, so
  each round re-proposes from the committed prefix cheaply), then (a) tree spec and
  (b) run the *target* stages under vLLM kernels too, to cut the verify's compute
  half. Projected with both: 20+ tok/s on consumer GPUs over WAN, output still exact.
- **2026-06-16 — P1 integrated and measured. It holds.** The in-house draft service
  (`research/draft_server.py`, vLLM gpt-oss-20b behind a socket; the entry node's
  managed draft, chosen over per-worker drafting since the draft holds no authority
  and quality sets UX speed) is wired into `phase0/specpipe.py` via `--draft-server`.
  Each round queries it for K tokens from the committed prefix; vLLM prefix-caching
  is the rollback, for free. Warm over the same Sweden↔NC WAN, K=4:
  **6.5 → 13.3 tok/s (2.04×)**, draft 309 ms → **30 ms/round**, acceptance held
  (2.34 vs 2.49), output still token-for-token exact. The round is now verify-bound
  (224 ms), so the linear-spec ceiling here is ~15 tok/s — the remaining gap to 20+
  is **tree speculation** (more accepted tokens per fixed-cost traversal) and, for
  more headroom, running the target stages under vLLM kernels too. Next: tree spec.
- **2026-06-16 — the real c0mpute topology, end to end.** Built the **coordinator
  architecture**: an in-house entry node holds *only* the draft + a thin driver
  (no 120B layers; `specpipe.py --coordinator`), and the full gpt-oss-120b lives on
  4 separate consumer GPUs scattered across the world — stage 0 (`--served-head`)
  embeds the token ids the coordinator sends. Ran it for real: in-house draft (US)
  + 120B across **Washington → France → UK → Singapore** (US→EU→Asia). It works,
  output exact. Warm K=4: **2.82 tok/s**, and the breakdown is the whole story —
  **draft 28 ms** (the in-house vLLM draft is cheap and local, as designed) but
  **verify 1082 ms**: the activation literally circles the globe (4 sequential
  inter-continental hops forward + relayed back) per traversal. So the architecture
  and the draft are *solved*; what's left is the WAN cost, and the levers are now
  concrete and ordered: (1) the scheduler must **cluster nodes by latency** — this
  was the maximally-scattered worst case; a regional swarm is multiples faster;
  (2) **direct tail→head return** (skip the relayed return, ~halve it); (3) **tree
  speculation** to amortize the expensive traversal over more tokens. Also observed
  live: one swarm node (Canada) **dropped offline mid-setup** and had to be replaced
  — real consumer nodes vanish, which is exactly the Phase 4 fault-tolerance case
  (re-route around a dead node, don't fail the request).
- **2026-06-16 — clustered re-run: geography was most of the penalty.** Same
  coordinator architecture, same exact 120B, but the 4 swarm nodes moved from 4
  continents to 4 US boxes (~cross-country hops). Warm K=4: **6.24 tok/s, verify
  474 ms** — vs the global-scatter **2.82 tok/s, verify 1082 ms**. So clustering
  alone **2.2×'d** it and halved the verify; the draft stayed cheap (28 ms) the
  whole time. The full measured spectrum now reads:

  | topology | WAN edges | verify | tok/s |
  |---|---|---|---|
  | 2 machines, transatlantic | 1 | 224 ms | 13.3 |
  | 4 nodes, clustered US | 4 | 474 ms | 6.24 |
  | 4 nodes, global scatter | 4 | 1082 ms | 2.82 |

  Two levers fall out, cleanly: **hop latency** (cluster nodes — worth 2.2×, the
  scheduler's job) and **edge count** (4 separate nodes pay 4 WAN edges vs the
  2-machine's 1; even clustered, more hops cost). The remaining gap to ~13 is
  addressable without touching the draft: **direct tail→head return** (the result
  currently relays back through every node — cut that and ~halve the WAN → ~10
  tok/s), partial co-location, and tree speculation. The in-house draft is solved
  across every topology; what's left is purely WAN structure, and it's well understood.
- **2026-06-16 — direct return, built and measured (+25%).** Implemented
  `--direct-return`: the tail sends each verify result straight to the coordinator
  (1 hop) instead of relaying it back up the chain (4 hops). The coordinator (entry
  node) has no open inbound port, so it connects *out* to the tail and the tail
  replies on that channel (a `hello_return` handshake, `select`-distinguished from
  the predecessor's activation connection); the intermediate stages become
  forward-only. Clustered US, warm K=4: **6.24 → 7.83 tok/s, verify 474 → 372 ms**,
  output still exact (3.10 tok/traversal, unchanged). The save is real but modest
  *here* because one return hop (WA2→Washington) was same-host — on a swarm of
  fully-distinct nodes it saves more. Spectrum: global-scatter 2.82 → clustered-relay
  6.24 → clustered-direct 7.83 → 2-machine (1 edge) 13.3. The gap to ~13 is the 4
  *forward* hops (inherent to 4 separate nodes), closed by tree speculation +
  partial co-location — neither touches the draft. Next: tree speculation.
- **2026-06-16 — tree speculation, built and measured. Mechanically correct; the
  finding is the bottleneck moved.** Full tree spec end to end: a branching draft
  (`draft_server.build_tree`), a tree attention mask + per-node positions
  (`run_block`, `tree.py`), accept-best-path, and a lazy distributed cache **gather**
  (keep only the accepted path's KV — the hard part, `tree.gather_cache`). On the
  clustered 120B swarm, w=3/d=6 (19-node tree): **4.81 tokens/traversal vs linear's
  3.10**, accept 3.76/round vs 2.10 — the tree does exactly what it should, more
  tokens committed per WAN round-trip. **But wall-clock fell: 5.48 vs 7.83 tok/s**,
  because the verify is **compute-bound, not latency-bound**: the eager-MXFP4 target
  runs ~34 ms/token, so a 19-node tree costs ~4× the verify compute and that
  outweighs the tokens-per-traversal gain. Tree spec is a *latency*-amortization
  technique; it pays off only when the round-trip dominates and extra tree nodes are
  nearly free. The bottleneck has moved cleanly: from the draft (solved with vLLM
  kernels) to the **target's eager-MXFP4 verify compute** — the same wall, the same
  fix (run the target stages under optimized/CUDA-graphed kernels). With a
  latency-bound verify, the tree's 4.81 tokens/traversal converts directly to
  throughput. Tree code is built, correct, and waiting on fast target kernels —
  the last integration toward the 20 tok/s thesis.

- **2026-06-16 — fast-verify de-risked: the gpt-oss layer block CUDA-graphs (4.2x,
  exact).** Probe (`research/probe_cudagraph.py`): a 9-layer 120B stage forward,
  eager **22.9 ms** vs CUDA-graph replay **5.4 ms = 4.2x**, output bit-identical
  (max-diff 0.0000). gpt-oss's MoE routing captures (batched all-experts kernel, not
  dynamic gather) -- the ~34 ms/token eager verify is ~75% removable launch/Python
  overhead, same as the draft. Projected: cutting the verify compute 4.2x drops the
  clustered linear-direct verify 372 -> ~190 ms -> round ~215 ms -> ~14 tok/s, and
  makes the verify latency-bound so tree spec finally pays off too. Build: a static
  pre-allocated KV cache + captured stage-forward graph (the realization of the
  fast verify). The path to 10-15 tok/s on 4 separate consumer GPUs is real.

- **2026-06-16 — fast verify REALIZED: static-cache CUDA graph, bit-exact, 5.3x.**
  `research/fastverify_graph.py`: the stage forward now runs in the real incremental
  decode against a pre-allocated static KV cache (StaticKV, MAXLEN=128 = the sliding
  window so every layer's cache is one fixed width), captured as ONE CUDA graph and
  replayed each round -- KV writes land at the round's position via a static index
  buffer, and rotary + 9 layers + MoE are all inside the graph. **Replay is bit-exact
  vs the eager static forward (max-diff 0.00000) and 5.3x faster (23.6 -> 4.4 ms).**
  Two findings that cost the most to pin down:
  (1) gpt-oss's attention calls `cache.update(k,v,layer_idx)` with NO cache_position,
  so the cache must own the write position -- StaticKV takes an explicit write_pos,
  which also makes spec-decode rollback trivial (set write_pos to the committed length;
  rejected KV is overwritten next round).
  (2) the static-128 forward differs from a DynamicCache forward by ~250 on hidden
  states of magnitude ~1000 -- but this is NOT a bug: eager DynamicCache itself differs
  batch-vs-sequential by ~250 (q_len-dependent MoE/bf16 rounding). spec-decode is exact
  at the ARGMAX level, which tolerates it (the eager spec-decode proved this). the right
  exactness test is end-to-end tokens, not stage hidden states.
  Projected: verify compute /5.3 -> clustered linear-direct verify 372 -> ~180 ms ->
  ~14 tok/s on 4 separate consumer GPUs, output still exact. Next: wire StaticKV+graph
  into specpipe serve (prefill eager, decode rounds graphed, rollback = write_pos), then
  deploy + measure over WAN.
