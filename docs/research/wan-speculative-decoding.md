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
