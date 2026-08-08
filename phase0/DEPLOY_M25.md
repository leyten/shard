# MiniMax-M2.5 — single disciplined GPU validation pass

The engine is **code-complete and locally proven (no-GPU)**: tool calling, multi-turn context,
signed per-stage receipts, and the OpenAI `/v1` gateway all pass local tests
(`research/m25_{tools,gateway,receipt}_test.py`, 47 assertions). The ONLY thing that needs GPUs is
the warm-libp2p validation + the CUDA-graph perf lever. This runbook makes that pass mechanical, so
it does NOT repeat the morning-killers (stuck downloads, `pkill` self-match, blind 30-min waits).

## Hard ops rules (every step)
> Before ANY vast action, re-read how it was done last time + the logged mistakes — memories
> `follow-existing-runbook-not-improvise`, `minimax-m25-base-model-decision` (the gotchas block),
> `shard-m25-deploy-ready-session5`, `vast-expose-29600-port`. Never improvise around it.
- **Provision ONLY boxes with `cuda_max_good>=13.0`.** THE killer that aborted the 2026-06-26 pass:
  `pip install vllm` pulls vLLM 0.23 + a `vllm._C` built against `libcudart.so.13`, which a CUDA-12.8
  driver (R570) cannot load (no consumer forward-compat) — the stage dies at `import vllm._C`. There is
  no cu128 vllm-0.23 wheel. ~half of vast's 5090s are still on 12.8; filter them out at provision time
  (`scratchpad/provision.py` does this). Also require **distinct `public_ipaddr`** per box — co-located
  instances share an IP and aren't a scattered ring.
- **No blind waits.** Every download/launch runs under a hard per-phase deadline. Stuck > deadline →
  kill + replace the box, never sit on the shell.
- **Over-provision 6-for-5** (7-8 if SSH-key propagation is flaky). Rent extra for a 5-stage ring; drop
  the slowest/flakiest. SSH must be **serial with gaps + fire-and-forget** (`nohup`/`setsid`) — the vast
  SSH proxy rate-limits under burst (parallel `ThreadPoolExecutor` SSH tripped it, all boxes denied
  `publickey` mid-run). Never drive stages from a bash `while read` loop (ssh eats the piped stdin →
  only stage 0 launches); use `ssh -n`/`</dev/null` or python `subprocess`.
- **`setsid … </dev/null &` + `fuser -k <port>/tcp`. NEVER `pkill -f`** (it self-matches the launch
  string and kills the launcher — the documented footgun).
- **Robust precheck before bootstrapping a box:** SSH-retry + `urllib` HF-reachability (NOT `curl`,
  not preinstalled) + GPU-count. Some vast hosts DNS-hijack huggingface.co — apt/pip work, HF doesn't.
- **Verify every file push** (grep a known line) before relying on it. scp inside `( … ) &` can
  silently not land.

## Topology
5 scattered US 5090s, even ~12-13 layers/stage over 62L, direct-return pipeline (head fire-forwards,
tail returns to coord). Sidecar binary at `/tmp/sidecar` (prebuilt June-19; can't rebuild on go1.22).
Always create boxes with `--env '-p 29600:29600'` (inter-stage transport unreachable otherwise).

## Sequence (driven by `m25_scatter_pipe.py`)
1. **Precheck** each candidate box: `ssh` reachable, `urllib` GET on the HF tokenizer_config 200, GPU
   count == expected. Drop failures, pull from the over-provision pool.
2. **Bootstrap** (per box, deadline ~12 min): venv + `pip install vllm` (→ vLLM 0.23 + torch/cu13 +
   flashinfer, just works on sm_120) + push code. Push set now includes **`m25_tools.py` +
   `model_tools.py`** (hard deps of `m25_pipe`; `model_tools` is the per-model tools seam and picks
   `m25_tools` for any M2.5 id) and **`receipt.py` + `manifest.py`** (so `SHARD_RECEIPTS=1` loads).
3. **Pull layer-range shards** (deadline ~15 min, hf_transfer; fallback `HF_HUB_ENABLE_HF_TRANSFER=0`
   if it STALLS): `m25_pull_range.py --lo L --hi H` per stage; `--head` adds embed+tokenizer, `--tail`
   adds norm+lm_head. **Verify** each box reports the expected shard count before launching.
4. **Sidecars** then **stages** (tail-first), each launched `setsid`, health-grepped (`tunnel up|
   listening` for sidecar, `WARM` for stage), retried, never `pkill`ed.
5. **Coordinator / gateway** on the head box.

## Validation (what the pass must prove, WARM over libp2p)
- **tok/s — SWEEP K, don't guess.** Per-stage GPU compute is FLAT in token count (launch-overhead-bound),
  so a bigger draft block is ~free on GPU — the real ceiling is the inter-stage payload (`h/trav` grows
  with K), not compute. We've only ever run K=6; find the actual peak:
  `m25_pipe.py coord --head … --tail … --sweep 4,6,8,12,16 --sweep-depth 2,4,8 --prompt-file copy.txt`
  prints one tok/s + g + accept% + h/trav table and the winning (K,depth). Baseline 15.79 @ K6/d4. Keep
  K≤16 (n-gram drafter's `margin=256` covers depth≤8,K≤16; bigger K needs a wider margin). The sweep
  driver (`_sweep_summary`) is unit-proven off-box: `research/m25_sweep_test.py` (8/8).
- **Confidence-scheduled depth — A/B it (opt-in `M25_CONF_SCHED=1` on the COORD, default OFF).** Adapts the
  in-flight verify depth from the running acceptance EMA: high accept → full `--depth` (throughput), a bad-draft
  streak → throttle toward 1 (fewer stale WAN chunks discarded). K stays fixed so it's CUDA-graph-safe and
  byte-lossless — proven so off-box (`research/m25_confsched_test.py`: output identical ON vs OFF == greedy
  truth, high + zero accept). On a high-accept copy/retrieval task it's inert (sits at full depth) = no
  regression to the warm baseline; the win shows on variable-acceptance (novel/chat) gen. Run the sweep once
  with the flag and once without on a chat-style prompt to measure. `confidence.py` is in the push set.
- **Long context (≥30k)**: set `M25_MAX_POS` ≥ the prompt+gen length on every stage (default 131072).
  The rotary table is now sized from it — a table shorter than the context silently returns garbage RoPE
  (the old hard-coded 8192 cap broke any >8k run, incl. this very test). Use this for the pipelined-prefill
  number too.
- **Tool calling**: serve the gateway, POST `/v1/chat/completions` with `tools=[…]`, assert
  `finish_reason=="tool_calls"` and a structured `tool_calls[0].function`. (Parser already proven
  locally against the real tokenizer; this confirms the model emits the format end-to-end.)
- **Multi-turn context**: 2-3 turn conversation incl. a tool result; long-context prefill (≥30k) for
  the pipelined-prefill number.
- **Receipts**: `SHARD_RECEIPTS=1` on every stage + coord. Coord prints N signed receipts, all sigs
  VALID, coverage `[0:62]` no gap/overlap. (`x_shard.receipts_ok` in the gateway response.)

## Deploy — serve the OpenAI /v1 gateway over the ring
Once the ring is warm (stages WARM, sidecars up), serve it as an OpenAI-compatible endpoint in one command:

    python engines/minimax_m25/m25_scatter_pipe.py --order <region:iid:lo:hi ...> --K 8 --depth 4 --serve [--receipts]

`--serve` brings up the ring then starts `m25_gateway.py` on the head (127.0.0.1:18000, persistent, via
setsid/nohup) instead of a one-shot coord job, and prints the tunnel command. Reach it:

    ssh -i ~/.ssh/vast_c0mpute -p <head_ssh_port> -L 8000:127.0.0.1:18000 root@<head_host>
    curl http://localhost:8000/v1/chat/completions -H 'content-type: application/json' \
      -d '{"model":"minimax-m2.5","messages":[{"role":"user","content":"hi"}],"stream":true}'

Endpoints: `/v1/chat/completions` (messages + tools + tool_choice + streaming) and `/v1/models`. Responses
carry `x_shard` telemetry (tok_s, mean_accept, receipts_ok). This gateway is the **c0mpute integration seam**
— c0mpute calls this `/v1` endpoint (tunnel or expose :18000). HTTP layer proven in MOCK
(`M25_GATEWAY_MOCK=1`); the engine path is the same `coordinate_pipe` the `--validate` pass exercises warm.

**Beta limits (state them honestly):** single-stream (one ring; concurrent callers queue on `RING_LOCK`);
GREEDY decode (`temperature`/`top_p`/`top_k` accepted but NOT applied — the tail argmaxes; lossless sampling
is a separate engine lever); on a node death the gateway retry RESTARTS the request (the `resume_ids`/
`resumable` primitive exists in `coordinate_pipe` but the gateway doesn't drive heal+resume yet). Adequate
for a niche/beta deploy — see `docs/DEPLOY_READINESS.md` for the full gap list.

## CUDA-graph lever (#6 — develop ON the box, it's empirical; NOT a free win)
The per-traversal ~95ms GPU is launch-overhead-bound (19.7ms/stage, FLAT in token count) → a CUDA graph
that cuts kernel-launch count is THE tok/s lever. But this is a real on-box engineering task with a
lossless-correctness risk, not a quick edit. The shape of the work:
- **Graph the BLOCK shape (s=K+1), NOT s=1.** Under spec-decode the hot path is the verify of a K+1-token
  draft block (`m25_pipe.py` sends `[dprefix[-1]]+ds`, ~line 104) — single-token decode never runs. So
  capture `run_block` at a **fixed K_max+1** shape and mask unused positions; because compute is flat in
  token count, graphing at K_max costs ~the same as s=1, so one graph serves any K≤K_max. (The earlier
  "graph the s=1 shape" note was wrong — that's the plain-decode path we don't run.)
- **KV is the hard part: full-context static preallocation DOESN'T FIT.** A graph needs all KV at fixed
  addresses, but `[1,NKV,131072,HD]×2 ≈ 537 MB/layer` → ~6 GB/stage just for KV — won't fit beside the
  weights on a 5090. Two real options: **(a) paged KV** (vLLM-style fixed-address pages; lossless — the
  correct path) or **(b) a sliding KV window** (cheap + bounded, but NON-LOSSLESS — changes numerics, so
  only acceptable in the latency-tolerant long-ctx copy regime where window-KV was already used, never as
  the default). The grow-by-`cat` in `Layer.attn` (~L144-149) is fine for **prefill** (a few big eager
  passes) — leave it; only the decode path needs the static/paged buffer.
- **Varying write offset isn't graph-capturable as a Python-int slice.** `k_buf[:,:,start_pos:…]` bakes
  the offset at capture. Use `index_copy_` with a **tensor** position + a **tensor** causal/validity mask,
  both read from small static input buffers updated before each replay (the runbook's "pass start_pos/
  cur_len via static buffers"). Prefill stays eager.
- **Opt-in `M25_CUDA_GRAPH=1`, default OFF — the eager path stays byte-identical and is what a normal
  swarm pass runs.** Confirm the NVFP4 cutlass FusedMoE is graph-safe (vLLM graphs it internally —
  expected OK, but UNVERIFIED off-box; confirm on the box before trusting it).
- **Bit-equivalence gate is mandatory, not optional.** The graphed stage is on the VERIFY path, so a
  capture/replay bug corrupts the committed output (spec-decode losslessness assumes the verify stage is
  exact) — not just a slow number, a WRONG one. Gate: greedy output ids identical to the eager run on the
  same prompt, every time, before reporting any tok/s.

## Privacy posture (already true, state it; don't over-claim)
- libp2p transport is Noise-encrypted node-to-node by default — no PSK, per-node keys.
- **Intermediate stages only ever see hidden-state tensors, never tokens/text.** Only the head sees
  input token ids; only the tail produces output tokens. So no single middle node can reconstruct the
  prompt or the answer. Stronger guarantees (coordinator-blind prompts, activation obfuscation) are
  research-grade, out of scope for the beta.
