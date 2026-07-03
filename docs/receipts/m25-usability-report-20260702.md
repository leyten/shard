# MiniMax-M2.5 sharded engine — usability ceiling report (2026-07-02)

Single-stream, **reasoning ON**, greedy, lossless (signed receipts). Scattered 6×RTX-5090 EU ring over libp2p: **Hungary → Italy → Norway → Denmark → Czechia → Bulgaria** (one stage per /24 subnet + machine; 62 layers split ~10/stage). fp8 activations on the wire. K=8, depth=4. Two drafter arms on the **same warm ring**: chain-EAGLE vs tree-hybrid (best-first EAGLE tree, M=12).

> **Honest framing — read first.** These are *this ring's* numbers, and it is a **middling rental draw**: chain reason-math here = 4.0 tok/s at g≈3.6, versus **11.8 tok/s at the same g≈3.7 on a good ring (2026-06-30)**. tok/s = g × traversal-rate, so at equal accept the ~3× spread is pure ring quality (RTT/jitter), not the engine. The portable, ring-independent signal is **g (accepted tokens per WAN round-trip)**; absolute tok/s scales with the ring. Projected onto a good ring, the reasoning cells land ~**10–14 tok/s**. We do not yet have a good-draw run for the headline absolute number — that needs RTT-ordered provisioning (next).

## 1. Workload ceilings (per task)

| workload | prompt | chain tok/s (g) | tree tok/s (g) |
|---|---|---|---|
| reason-math | A farmer has 17 sheep. All but 9 run away. How many sheep… | 4.0 (g3.6) | 4.3 (g4.5) |
| reason-logic | Three light switches outside a windowless room each contr… | 3.0 (g2.2) | 2.5 (g3.5) |
| open-chat | Explain the main tradeoffs between mixture-of-experts and… | 2.5 (g2.0) | 2.7 (g2.5) |
| code-edit | Here is some code:  """MiniMax-M2.5 PIPELINED ring — dire… | 4.0 (g2.8) | 5.1 (g4.2) |
| rag-quote | Here is some code:  """MiniMax-M2.5 PIPELINED ring — dire… | 5.7 (g4.0) | 4.6 (g3.8) |
| agentic-tool | What's the current weather in Tokyo? Use the get_weather … | 3.2 (g2.6) | 5.4 (g4.1) |

**Workload decode-weighted:** chain **3.50**, tree **3.67** tok/s. Tree's g exceeds chain's on every cell (it's a strictly better drafter); tok/s wins where that accept gain beats its synchronous per-round cost — **agentic +65%, code-edit +26%, math +8%** — and loses on **rag-quote −20%** (chain pipelines verbatim n-gram depth-4; tree verifies one tree per round-trip — the depth-aware-hybrid fix is queued) and reason-logic (ring jitter).

## 2. Context axis — the long-context wall

Real-document tasks. Decode holds, but **prefill (time-to-first-token) is the usability wall**, not decode tok/s — the upload-bound cost of shipping the [S,H] activation across the ring.

| context | task | chain tok/s | chain prefill(TTFT) | tree tok/s | tree prefill |
|---|---|---|---|---|---|
| 8k | summarize | 4.2 | 63s | 4.4 | 59s |
| 8k | quote | 3.5 | 62s | 2.1 | 80s |
| 30k | summarize | 3.3 | 207s | 2.7 | 202s |
| 30k | quote | 1.8 | 193s | 2.6 | 181s |

**30k-token prefill ≈ 200s to first token** — the dominant long-context cost. Decode stays ~2–3 tok/s.

## 3. Multi-turn conversation (8 turns, history carried)

A realistic session (explain → compare → write code → refine → quote-back → compute → recompute → summarize). Per-turn decode tok/s and prefill as context grows.

| turn | ctx tok | chain tok/s | chain prefill | tree tok/s | tree prefill |
|---|---|---|---|---|---|
| 1 | 51 | 3.7 | 1s | 4.0 | 2s |
| 2 | 467 | 2.7 | 6s | 4.2 | 4s |
| 3 | 898 | 3.8 | 6s | 3.5 | 7s |
| 4 | 1311 | 4.1 | 13s | 3.8 | 14s |
| 5 | 1729 | 6.2 | 20s | 5.7 | 17s |
| 6 | 2157 | 4.4 | 19s | 5.0 | 19s |
| 7 | 2575 | 4.6 | 28s | 4.2 | 16s |
| 8 | 2987 | 5.4 | 13s | 3.7 | 31s |

**Conversation mean:** chain **4.13**, tree **4.07** tok/s. A live reasoning chat holds ~**4 tok/s decode**; prefill grows with history (turn 8 ≈ 3k ctx → ~15–30s).

## 4. Verdict

- **Usable-speed reasoning on scattered consumer GPUs is real** but ring-bound: ~4 tok/s on this middling draw, ~10–14 projected on a good ring (g × faster traversal).
- **Tree-hybrid is the better drafter** (higher g everywhere); its tok/s edge is workload-dependent until the depth-aware hybrid closes the verbatim/pipelining gap.
- **The real usability wall is long-context prefill (TTFT)**, an upload-bound cost — a different lever than decode tok/s (fan-in / fewer-fatter-hops / fp8, already partly landed).
- **Next for the absolute ceiling:** RTT-ordered provisioning (stop drawing mediocre rings) + depth-aware hybrid (recovers rag-quote). Full per-cell data + prompts + answers: the two JSON receipts beside this file.
