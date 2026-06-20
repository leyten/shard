# Mid-generation self-healing

*Design record. Status: increment 1 (the decision logic) landed; engine + sidecar
wiring are the follow-ups. In the spirit of the other docs here — what it does, what
it deliberately leaves out, and how a skeptic checks it.*

## The problem

The swarm runs on consumer GPUs on home connections. They drop. Today, if any one
stage node dies in the middle of a request, the whole request dies with it: the
coordinator hits a transport error and gives up (`phase0/specpipe.py` raises
`TransportError`; `phase0/node_kv.py` calls `os._exit`). Fast *detection* is already
done — per-edge timeouts, fail-fast, no silent hang. What is missing is *recovery*.

A pipeline of N home GPUs where any single hiccup kills every in-flight request is
not usable. So one node vanishing mid-request has to become a recoverable event, not
a fatal one. SERVE is fast; this is what makes that speed actually deliverable.

## The shape of recovery: stop, swap, replay, continue

When a stage dies, the coordinator already holds everything it needs to rebuild: the
prompt and the tokens committed so far. Recovery is four steps:

1. **Stop.** Halt sending, drop the in-flight verify chunks (the coordinator already
   discards stale chunks on a divergence — same machinery).
2. **Swap.** Find the dead stage, bring in a warm spare that already holds the same
   block of layers, and re-wire the ring around it.
3. **Replay.** Reset every live stage's KV cache and re-feed the committed prefix, so
   the rebuilt ring is back in the exact state it was in before the death.
4. **Continue.** Resume decoding from the committed position.

Because decoding is greedy and we resume from the exact committed tokens, the output
is token-identical to a run that never failed. That is the whole correctness claim,
and it is checked directly (see "Why the output stays identical").

## What this PR contains

Only the decision logic, as a pure module: `shard/heal.py`. No torch, no sockets, no
model — so it runs and is unit-tested anywhere (`tests/test_heal.py`, stdlib only).
The engine and the libp2p sidecar call into it; they are the next PRs (they touch the
GPU serve path and need a real multi-node rig to validate end to end).

Three pieces:

- `locate_failure(reports)` — the coordinator only sits on the stage-0 and tail edges,
  so a middle-stage death shows up as a *result that never returns*, not a broken
  coordinator socket. Each stage reports over a control channel: is it reachable, the
  highest verify-chunk it received from upstream, the highest it forwarded downstream.
  The failed node is the first unreachable stage, or the first *gap* — a stage whose
  predecessor forwarded a chunk it never received. This also catches a frozen (not
  dead) node, which a plain liveness ping would miss.
- `plan_replay(prompt_ids, committed_out)` — builds the prefix to replay. The safe
  prefix is the prompt plus all committed output **except the current driver token**,
  because re-feeding it must greedily *reproduce* that driver token.
- `run_recovery(ops, prompt_ids, committed_out)` — the ordered state machine
  (localize → check spare → activate → re-wire → reset → replay → verify → resume)
  over an injected `SwarmOps` interface, so the engine implements the real node
  operations later and this stays testable with a fake.

## Why the output stays identical

`plan_replay` deliberately excludes the last committed token from the replay prefix.
After re-feeding `prompt + committed[:-1]`, the model's next-token argmax must equal
`committed[-1]` — because that token was itself produced as the greedy argmax of that
exact prefix. `run_recovery` checks this (`replay` result vs `expected_next`) and
**aborts the recovery if it does not match**, rather than silently producing a
different continuation. So recovery is either token-identical or it fails loudly; it
never quietly diverges.

## Scope — what increment 1 does and does not do

Handles: a single fail-stop or frozen stage during one request, when a warm spare for
that block exists. Middle-stage death is the hard case and is covered first; stage-0
and tail are the same primitive with a small edge case (the coordinator re-wires).

Explicitly deferred (named, not hidden):

- **Spare coverage.** Increment 1 assumes one warm spare for the protected block. A
  spare per block (so *any* single death is survivable), or cold-loading from a pool,
  is scheduler work — it comes with automatic swarm formation.
- **Long context.** Replaying a very long committed prefix can exceed the request
  budget; long-context recovery needs incremental/chunked replay.
- **Multiple simultaneous deaths, coordinator death, and Byzantine nodes** (a node
  returning plausible-but-wrong activations) — all out of scope here.

## Follow-up PRs (need a GPU rig)

1. **Engine wiring** (`phase0/specpipe.py`): a per-stage control channel beside the
   data socket; the missing primitive of re-dialing the downstream socket after
   startup (stages today only ever re-accept upstream); and turning the coordinator's
   "timeout → fail" into "timeout → `run_recovery`".
2. **Sidecar admin** (`sidecar/main.go`): the forward target is fixed at launch; add a
   small local admin API so a forward route can be swapped to the spare at runtime
   (no weight reload, no sidecar restart).

## Acceptance (for the engine PR)

Run a fixed prompt at fixed `K` and `depth`, record `output_ids`. Run again, kill the
protected middle stage after at least one committed chunk, recover, and assert the
final `output_ids` match the first run **exactly** (compare token ids, not decoded
text — a quantized model's text can match while ids differ at a near-tie). Also assert
no weight reload happened during recovery and the no-failure tok/s is unchanged.

The hardest part is rebuilding the exact KV state after pipelined stale chunks; that
is why the test compares ids and why `run_recovery` verifies the replay before
resuming.
