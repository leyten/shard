# V4 pipelined speculation — the tap-free proposer, and what it is actually worth

> **STATUS, 2026-08-02: measured, priced, and left UNMERGED.** The implementation this document
> describes (`V4_PIPE_PROPOSER`, the match gate, the hybrid) lives on branch
> `v4/ngram-pipe-proposer` and was deliberately not ported onto `v4/truth`: the verdict below —
> ~+1% on real coding output, +69% only on copying — is why. The primary fill lever became the
> refill floor instead (`V4_REFILL_FLOOR`, see `docs/V4_MULTIBLOCK_VERDICT.md` §4's correction),
> which needs no new proposer and no workload luck. On the 07-31 ring the bar got harder, not
> easier: on top of `floor=5` a tap-free extension breaks even only at flat q ≥ 0.62–0.89
> (`phase0/v4_ngram_econ.py`), and measured n-gram acceptance on novel text is 0.016 — a 40× miss.
> The acceptance harness (`phase0/v4_ngram_accept.py`) and the per-depth instrumentation are
> merged; the proposer machinery stays on its branch until someone shows a tap-free proposer with
> real acceptance on NOVEL text, which n-gram is not.

## The verdict, before anything else

**The lever is safe, it is built, and on real agentic-coding output it is worth about +1%.** It is
worth +69% on exactly one thing: reproducing text that is already in the prompt.

Measured against **real greedy output from a real ring** (40.9k tokens of M2.5 traces), with the
match gate at its shipped setting — `rate` is the fraction of rounds the proposer speaks on at all,
`q` its acceptance when it does, and the last column is the priced throughput against DSpark-only:

| real workload | prompt | rate | q | vs `dspark` |
|---|---:|---:|---:|---:|
| `reasoning` | 89 | 0.000 | — | **0%** |
| `code-edit` — a real edit task, 3.5k of pasted context | 3563 | 0.004 | 1.00 | **+0.7%** |
| `code` | 71 | 0.011 | 0.75 | **+0.9%** |
| `mix` (agentic traffic) | 99 | 0.012 | 0.82 | **+1.2%** |
| `summarize` | 244 | 0.042 | 0.92 | +4.0% |
| `ctx-8k-quote` — "quote this document back" | 8051 | 0.316 | 1.00 | **+69%** |

**So the honest answer to "does this move an agentic-coding benchmark": no.** It fires on 0.4% of
rounds on the one real code-edit trace available, because a model asked to fix a function answers
with a patch and an explanation, not with the file typed out again.

(The `vs dspark` column is net of the model's ~1% Monte-Carlo noise floor, which is what the
`reasoning` row — propose-rate exactly 0.000, and therefore exactly `dspark` by construction — is
there to calibrate. Raw figures are +1.0/+1.7/+1.9/+2.2/+5.0/+70.0.)

**And the flattering number, named as such.** A synthetic workload that asks for the *whole file
back* with a rename applied — the most copy-heavy form an edit task can take — measures q₁ 0.836 and
prices at **+63%**. That is the number to distrust. It assumes the model re-emits its context
verbatim, and the real edit trace says it does not. Quote it only alongside the 0.004 above, and only
if your agent harness genuinely re-emits whole files.

| proxy workload (reference TEXT, not model output) | n-gram q₁ | q at depth ≥2 | g | vs `dspark` |
|---|---:|---:|---:|---:|
| `edit` — whole file re-emitted, one rename + two constants | 0.836 | ~0.99 | 7.50 | +63% |
| `edit_heavy` — same, ~20% of tokens moved | 0.613 | ~0.97 | 5.53 | +26% |
| `novel_code` — code the prompt never showed | 0.010 | — | 1.02 | 0% |
| `novel_prose` — reasoning over a code context | 0.006 | — | 1.01 | 0% |

Two orders of magnitude between them. **Every tok/s this lever adds is a repetition number.** DSpark-
only and DSpark+n-gram are reported separately throughout; this document never blends them.

**Recommendation.** Ship it — it is opt-in, default OFF, provably free when it cannot help (§4), and
it is the only thing that lifts the `block+1` in-flight cap at all. Do not budget any throughput for
it on coding work. Then run one ring job on the ACTUAL bench prompt and read
`ngram_silent / (ngram_rounds + ngram_silent)` off the return dict: that single ratio decides, in one
run, whether the workload is a copy workload. Everything else here is arithmetic on top of it.

Reproduce: `python3 phase0/v4_ngram_accept.py --k 8`, `--traces <jsonl> --margin 0 --ng 2`,
`python3 phase0/v4_ngram_econ.py`.

---

## 1. What was built

`coordinate_dspark_pipelined` gained `V4_PIPE_PROPOSER`, default `dspark` and byte-identical to what
shipped:

- **`dspark`** — the trained MTP block, and nothing else. In-flight pinned at `block+1`.
- **`hybrid`** — the MTP block, plus n-gram frames topping the chain up to `V4_SPEC_DEPTH`.
- **`ngram`** — no block at all; the tail is told to stop drafting (the reset's `draft` flag), which
  removes the MTP advance from the ring's usually-slowest stage while keeping its `acc`.

Plus the **match gate** (`V4_PIPE_NGRAM_MINMATCH`, default 8): the proposer only speaks when its
anchor recurrence carried at least that many tokens of agreeing preceding context. §4 is why that one
knob is the whole difference between a lever and a liability.

## 2. Why the cap existed, and why a tap-free proposer lifts it

`docs/V4_MULTIBLOCK_VERDICT.md` established it: a DSpark draft needs `main_hidden`, a tap only the
ring produces, and the reply carrying frame `P`'s block is the same reply that commits `P+1`. So the
horizon lands at `c+B` and the frontier at `c`, forever, and `V4_SPEC_DEPTH` above `B+1` changes
nothing. An n-gram proposal needs no tap and no forward — it is the continuation that followed the
last earlier occurrence of the current suffix — so it can name a position the ring has never
computed. The cap becomes the depth knob.

Measured on the localhost ring (`tests/test_v4_pipe.py::test_a_tap_free_proposer_lifts_the_block_plus_one_cap`):
in-flight **4 → 12** at the same block size, with the emitted stream still the reference's.

## 3. The proposal-agnostic claim, verified rather than inherited

The claim was that the tail's accept rule and the coordinator's `_feed` do not care where a token
came from. Both halves check out, and the second one is the load-bearing one:

`_feed(pos, tok)` builds `{"op": "step", "ids": [[tok]], "start_pos": pos, ...}`. Nothing else. There
is no drafter identity on the wire and no field a proposer could poison.

The tail's decision, in full (`v4_dspark_draft.RingDrafter._on_chunk_pipelined`):

```python
if start_pos != self._cfront + 1 or int(ids[0, 0]) != self._mfront:
    return {"acc": False}                          # speculative, and wrong: judge nothing
```

`_cfront` is the last position this tail judged committed and `_mfront` is the greedy token it
produced for the next one — both computed from its OWN replies. The decision is a function of
`(frame position, frame token)` against two scalars the tail owns. It never reads its block, and
there is no branch anywhere in the accept path that depends on the proposer.

**Not taken on inspection alone.** `test_a_tap_free_proposer_lifts_the_block_plus_one_cap` drives the
ring with a proposer the tail has never heard of and asserts `ngram_accepted > 0` with
`ngram_accept_by_depth` populated past the block: the tail accepted, at depth, tokens it did not
propose, and the stream stayed exactly greedy's.

## 4. The floor, which is the only reason `hybrid` is safe

The obvious framing — "keep DSpark's block at the near horizon, let n-gram extend past it into stages
that would otherwise idle, worst case we lose nothing but idle time" — **is wrong, and the ring shows
why.** Two facts collide:

1. The MTP block is streamed only when `horizon == c`. Once the top-up has filled the chain to `W`,
   that is never true again until a cancel.
2. The MTP-eligible window slides forward one position per commit, and the top-up has already taken
   those positions. A fresh block cannot reclaim a position that is on the wire.

So an ungated hybrid streams the trained block **once per cycle** and runs on n-gram quality after
that. On a workload where n-gram is wrong, it does not merely waste the frames it adds — it
**truncates DSpark's runs**, cutting every run at the first n-gram frame instead of letting the block
refill and continue. Ungated, `hybrid` is *worse* than `dspark` on novel text, not equal to it.

**The match gate fixes it at the source: a proposer with no evidence says nothing at all.** Measured
propose-rate at `min_match=8`, over the same workloads:

| workload | propose-rate | acceptance when it does propose |
|---|---:|---:|
| `edit` | 0.625 | 0.995 |
| `edit_heavy` | 0.316 | 0.969 |
| `novel_code` | **0.000** (≈1400 rounds) | — |
| `novel_prose` | **0.000** (≈2900 rounds) | — |

At propose-rate 0.000 no frame is fed, no epoch is bumped, no stage rewinds, and the hybrid is
`dspark` frame for frame. That is asserted on real sockets, not argued:
`test_the_match_gate_makes_the_hybrid_cost_nothing_when_there_is_nothing_to_copy` runs both proposers
on one warm ring and requires identical `frames`, `cancels`, `stale_replies`, `mean_inflight`,
`max_inflight` and `tokens`.

**So the honest price of the lever is: zero on workloads it cannot help, provided the gate is on.**
Turn `V4_PIPE_NGRAM_MINMATCH` down to 0 and that guarantee is gone.

The second safety property is structural rather than measured: every n-gram frame is fed past the
deepest frame already in flight, so it sits behind every MTP frame in the chain. A rejection kills
only what is behind it, so a wrong n-gram proposal can discard other n-gram frames **and nothing
else** — the block's frames, and the tokens they commit, are untouched
(`test_ngram_frames_are_always_deeper_than_the_block_they_extend`).

## 5. Losslessness

The proposer only proposes. The tail's greedy token at the frame's position is what commits, and a
rejected frame is fenced and rewound exactly as an MTP one is — so the emitted stream is the
reference Transformer's greedy stream bit for bit, whichever proposer filled the pipe and however
deep it filled it. Proven, not asserted, on real `shard.transport` sockets over a real multi-stage
ring:

- `phase0/v4_pipe.py selftest` — `ngram` and `hybrid` at W = 2, 6, 16, plus a perfect proposer and
  the gated hybrid: **all bit-identical to the vendored reference's greedy decode, receipts settling.**
- `tests/test_v4_pipe.py::test_a_tap_free_proposer_is_lossless_at_every_depth` — the same grid as a
  parametrised test.

The grid is driven by a deliberately bad proposer, so it is the **deep-rejection** path: at W=16 one
reply cancels fifteen in-flight frames and every stage rewinds across almost the whole of its
checkpoint ring (`_spec_ckpts`, `maxlen = V4_SPEC_DEPTH`, the same number on both sides). That is the
case that could have gone wrong silently. The opposite extreme — every proposal accepted, W frames
genuinely in flight, nothing rewinding — is covered by the perfect-proposer arms.

## 6. What it is worth — and the correction to "the pipe is 37% full"

**First, the premise this work was commissioned on does not survive its own inputs.** The brief was:
slowest stage 37.4 ms ⇒ ceiling 26.8 tok/s, measured 10.06, therefore **37% fill**, therefore ~16 idle
stage-slots to give away. That divides a TOKEN rate by a FRAME ceiling. They are not the same number,
because a pipelined speculative ring computes frames it then discards.

Run the shipped coordinator's own round logic against the DSpark acceptance the same brief supplies
(q₁..q₅ = 0.86/0.83/0.79/0.74/0.68) and it commits **0.62 tokens per frame** with **4.4 frames in
flight**. To emit 10.06 tok/s it must therefore be retiring **16.3 frames/s** — which is **61% of the
26.8 frames/s ceiling**, not 37%. The missing 39% is not idle time waiting to be filled; roughly
two-thirds of it is frames the ring already computes and the fence already throws away.

So the headroom from filling alone is at most **1.64×**, not 2.7×, and every frame added to reach it
is a frame that a rejection also destroys. This is the same class of error
`docs/V4_PIPELINE_EFFICIENCY.md` §2 caught on the previous ring, and it is settled for free by one
counter the coordinator already returns: `frames / generated`. Near 1.0 ⇒ fill-bound. This model says
**1.62**.

**The arithmetic.** Frame-exact replay of the coordinator's round logic, then Little's law with the
measured stage times: `tok/s = (tokens/frames) × min(F̄/L, 1/τ_max) × 1000`, with `τ_max = 37.4 ms`
and `L` calibrated so `dspark` reproduces the measured 10.06 tok/s (giving `L = 272 ms`). n-gram
acceptance and propose-rate are the measured ones from §0 and §4; nothing else is free.

| proposer | workload | W | tok/s | vs `dspark` |
|---|---|---:|---:|---:|
| `dspark` | any | 16 (shipped) | **10.06** | — |
| `hybrid` | `edit` (q 0.96, rate 0.63) | 8 | **16.38** | **+63%** |
| `hybrid` | `edit_heavy` (q 0.91, rate 0.32) | 8 | **12.68** | **+26%** |
| `hybrid` | `novel_*`, gate ON (rate 0.000) | 8 | **10.16** | **+1%** ≡ identical |
| `hybrid` | `novel_*`, gate OFF | 8 | 9.50 | −6% |
| `ngram` | `edit`, ungated | 8 | 20.94 | +108% |
| `ngram` | `novel_*` | 8 | 3.40 | **−66%** |

Read it in this order:

1. **`dspark` is unaffected by the depth knob** — its in-flight is pinned at `block+1` whatever `W`
   says, so `V4_SPEC_DEPTH` is not a DSpark tuning knob at all. It becomes one the moment a tap-free
   proposer is on.
2. **W=16, the shipped default, is the wrong depth for every tap-free arm.** The optimum is 7–8; at
   16 the `edit` hybrid gives back two thirds of its win (+17% instead of +63%) because the marginal
   frame is both the least likely to be right and the most expensive to be wrong about. **If this
   lever is turned on, `V4_SPEC_DEPTH` must come down with it.**
3. **The break-even, as a flat n-gram acceptance:** `hybrid` needs **q ≥ 0.13**, pure `ngram` needs
   **q ≥ 0.74**. Measured n-gram q is 0.96/0.91 on the edit cases and **0.016** on novel. So the
   hybrid clears its bar by 7× where n-gram works and misses it by 8× where it does not — which is
   the entire argument for the gate, restated as a number.
4. **Pure `ngram` is not deployable and is not recommended**, despite the biggest number in the
   table. It has no fallback: gate it and it degenerates to greedy, ungate it and a novel workload
   costs two thirds of the throughput. `hybrid` keeps the trained drafter underneath, which is what
   makes its downside bounded.

**Robustness.** Every ordering above holds under the alternative DSpark acceptance fit
(`v4_pipe_sim`'s 5-point `a_i = 0.796·0.945^(i-1)`, which puts the baseline at 71% bottleneck-busy
and 1.89 frames/token). Under it the hybrid's break-even rises to q ≥ 0.47 — a 3.6× stiffer bar,
and the one number that genuinely moves between the two fits — while the `edit` win at W=7 is +51%
against +51%, i.e. unchanged. The orderings and the recommendation are the same either way; the two
fits bracket the honest range.

**What falsifies the win, and did.** The `edit` row rests on q ≈ 0.96 at a propose-rate of 0.63,
measured against reference TEXT standing in for a model's greedy stream. The real-trace arms
(`--traces`) replace that stand-in with an actual ring's output and the propose-rate collapses to
0.004 on a genuine 3.5k-context code-edit — so the +63% is an artefact of assuming the model re-emits
its context verbatim. **The proxy was optimistic by roughly two orders of magnitude in RATE**, though
not in acceptance: when the gated proposer does speak on the real traces it is right 75-100% of the
time. It simply almost never speaks.

The residual uncertainty runs the other way and is worth one measurement: those traces are M2.5, from
a usability suite whose outputs are ~250 tokens. A coding agent whose tool calls write whole files
back would sit closer to the proxy. `ngram_silent / (ngram_rounds + ngram_silent)` on the real bench
prompt is the one number that settles which regime a given harness is in.

## 7. What the next ring run must measure

The coordinator now reports, per proposer and never blended:

```
accept_by_depth        {depth: (hits, trials)}   the MTP block's
ngram_accept_by_depth  {depth: (hits, trials)}   the tap-free frames', at their depth past the frontier
ngram_frames           frames the proposer put in flight
ngram_accepted         how many of them the ring agreed with
ngram_rounds/ngram_silent   how often the gate let it speak, and how often it did not
```

`ngram_silent / (ngram_rounds + ngram_silent)` is the propose-rate on the REAL workload, which is the
number that decides whether this lever is on or off for a given deployment. `ngram_accept_by_depth`
is the per-depth acceptance the throughput model needs instead of an assumption.

One more counter settles a live disagreement about this ring and costs nothing to read:
`stale_replies` against `generated`. If frames-per-token is near 1 the pipe is fill-bound and depth
is free; if it is near 2 the ring is already spending half its compute on discarded speculation and
depth is expensive. `docs/V4_PIPELINE_EFFICIENCY.md` measured the latter on the previous ring and
found the "N% full / idle stages" framing there to be an artefact of dividing a per-frame `on_box` by
the timed window's mean `s`. Do not size this lever off a fill estimate that has not been checked
against that counter.
