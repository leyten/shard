# H1 Test Plan v2 — the "no-landmines" environment + run procedure

_Author: Henry. Date: 2026-06-26. Purpose: complete the draft-acceptance sweep (and follow-on draft
tests) efficiently, with the failure modes from the 2026-06-26 session engineered out rather than
hand-avoided._

## 1. What today's session cost us (the landmines)

| # | Landmine | Root cause | Structural fix |
|---|---|---|---|
| 1 | sm120/DSA unservable, found live (~$16) | rented before checking the attention backend exists for cc12.0 | **Pre-flight gate** asserts servability offline before any rent |
| 2 | stale code ran (scp race) | launched before the file landed on the box | **md5-gated deploy** — refuse to launch on hash mismatch |
| 3 | `UnpicklingError` | coord `SHARD_WIRE=1` vs stage pickle | **single sourced env** (`h1_env.sh`) — one transport everywhere |
| 4 | `ModuleNotFoundError: wire` | module import at load, wrong cwd | env file pins `cwd=/root` + `PYTHONPATH` |
| 5 | cuda:0 vs cuda:7 device mismatch | coord weights off-device from ring output | co-locate coord work on one device (codified in driver) |
| 6 | `Forward context is not set` (FusedMoE) | hand-rolled `set_forward_context` | always run blocks through leyten's `run_block` |
| 7 | MTP 0.53 (nearly shipped backwards) | fed pre-norm hidden, no convention check | **convention sweep is a mandatory step** before reporting any draft number |
| 8 | CUBLAS OOM at 8k | dense O(n²) full-prefill | **chunked/incremental prefill** (the one real build) |
| 9 | ~5 box restarts, ~6 min re-warm each | one process per test, re-warms the ring every time | **warm once, run the whole matrix** in a single driver |

## 2. What's DONE (don't re-run)

- sm120 path proven: GLM-5.2 served **dense** via leyten's ring (stock vLLM/SGLang can't — cc12.0 gap).
- Convention settled: MTP wants **post-`model.norm` hidden, concat `[emb;hidden]`** (`ring_mtp.py --diag`).
- **1k head-to-head: MTP 0.857 / accept_len 6.99 BEATS 9B 0.812 / 5.30.** Banked.
- 9B per-position argmax dumped on beast for 1k/8k/32k ($0, `dump_9b.jsonl`) — the 9B side of 8k/32k is
  already in hand; the box only needs the **target** + **MTP** argmaxes at those lengths.

## 3. What REMAINS (the actual test backlog)

The long-context curve — the interesting part, because the 9B is hard-capped at 32k while MTP rides to 100k:

| context | target argmax | MTP accept | 9B accept | needs |
|---|---|---|---|---|
| 1k   | ✅ | ✅ 0.857 | ✅ 0.812 | done |
| 8k   | ⬜ | ⬜ | (9B dump ready) | chunked prefill |
| 32k  | ⬜ | ⬜ | (9B dump ready) | chunked prefill |
| 100k | ⬜ | ⬜ | n/a (9B capped) | chunked prefill + 100k corpus (have) |

So the box backlog = **{target dump, MTP sweep} × {8k, 32k, 100k}** = 6 measurements, all gated on one
build (chunked prefill). The 9B acceptance at each length is then computed **offline** from the dumps.

## 4. The "no-landmines" environment (build once, reuse)

Four small artifacts, all in `shard/research/`, all validated offline before any rent:

1. **`h1_preflight.py` (offline gate, $0).** Asserts every precondition and prints GREEN/RED:
   model servable on target cc (dense ring), all harness modules present, corpora + tokenizer-compat,
   convention locked, 9B dumps present for the ctx range, chunked-prefill self-test passes. **No rent
   until GREEN.** (Kills #1.)
2. **`h1_env.sh` (sourced by every launch).** Exports `SHARD_WIRE=` (pickle, consistent), `SHARD_PSK`,
   `GLM_DIR`, `PYTHONPATH=/root`, and `cd /root`. One source of truth for the box env. (Kills #3, #4.)
3. **`h1_deploy.sh` (md5-gated).** scp's the harness set, re-hashes each file on the box, **aborts the
   launch on any mismatch.** No more stale-code runs. (Kills #2.)
4. **`h1_run_all.py` (one driver, warm once).** Brings the 8-stage ring + coord weights up **a single
   time**, then walks the whole test matrix (target dump + MTP, every ctx length) in one process,
   co-locating all coord work on one device, running every block through `run_block`. Writes a status
   line per sequence (→ bridge outbox → Joe) and **auto-stops the vast instance on completion** so the
   box never idles metered. (Kills #5, #6, #9 + cost-idle + cadence.)

The one real engineering piece: **chunked/incremental teacher-forced prefill** inside the ring forward —
feed each sequence in fixed chunks through the KV cache (chunk attends `[chunk, cached_context]`,
O(n·chunk) not O(n²)), exactly how leyten generates in prod. Built and **self-tested offline against the
1k path** (must reproduce the known 1k numbers before it's trusted at 8k+). (Kills #8.)

## 5. The run procedure (one clean box session)

```
# OFFLINE ($0)
python h1_preflight.py            # must print ALL GREEN
# (chunked-prefill self-test runs inside preflight; reproduces 1k target+MTP)

# RENT
vastai start instance <id>        # stop-not-destroy preserved the 435GB model + harness
bash h1_deploy.sh                 # md5-gated; aborts on mismatch
ssh ... 'source h1_env.sh && python h1_preflight.py --on-box'   # box-side green check

# ONE WARM, FULL MATRIX
ssh ... 'source h1_env.sh && nohup python h1_run_all.py \
   --ctx 8192 32768 102400 --modes target mtp --auto-stop &'
# driver warms ONCE, runs all 6 measurements, pulls results, then vastai-stops itself

# OFFLINE ($0)
python h1_bench.py compare ...    # 9B acceptance vs the new target dumps
# update H1-process-and-results.md with the full curve
```

## 6. Cost & cadence

- **One warm-up** instead of ~5 (saves ~$5 + ~25 min wall-clock vs today's thrash).
- **Auto-stop** guarantees no idle metered time.
- Estimated full long-context session: ~1–1.5 hr (~$15–25) for all 6 measurements, 100k being the heavy
  one. Compared against today's ~$45 of restart/discovery thrash for a single context point.
- Status: one line per sequence to the bridge outbox; no silent gaps.

## 7. Scope

Confirmed by Joe 2026-06-26: **just the MTP-vs-9B curve** (no EAGLE-3 / n-gram / concurrency for now).

## 8. IMPLEMENTED (2026-06-26) — the artifacts exist and are validated

All under `shard/research/`. The blocking landmines (#1, #2, #3, #4, #8, #9 + RoPE cap) are now removed in
code, not by hand-care:

| Artifact | Kills | Status |
|---|---|---|
| `ring_long.py` | #8 (O(n²) OOM), #9 (re-warm tax) | **DONE** — chunked prefill; warms once, sweeps all ctx; writes target dump + MTP acceptance; result file written after each corpus (survives a kill) |
| `glm_swarm_nvfp4_kv.py` `_get_pe` patch | RoPE cap at 4096 | **DONE** — honors `$GLM_MAXPOS` (131072), so 8k/32k/100k positions don't index OOB |
| `h1env/h1_env.sh` | #3 (transport mismatch), #4 (wire/cwd) | **DONE** — one sourced env: `SHARD_WIRE=` empty, `GLM_MAXPOS`, `GLM_DIR`, `PYTHONPATH`, `cd /root` |
| `h1env/h1_deploy.sh` | #2 (scp race) | **DONE** — md5-gates every file; refuses to leave a mismatched file; exit 1 blocks launch |
| `h1env/h1_preflight.py` | #1 (rent-before-check) | **DONE** — offline (corpora/9B-dumps/convention/fixes) + `--on-box` (model/venv/8-GPU-free/env); offline run = ALL GREEN |
| `h1env/h1_session.sh` | idle-meter, re-download | **DONE** — turnkey: offline-preflight → start → resolve endpoint → md5 deploy → on-box preflight → ONE warm → chunked sweep → pull → **auto-stop**; uses `vastai stop` (model preserved, zero re-download) |

**Validated live (2026-06-26):** chunked prefill reproduces the known 1k number (chunked 1k MTP = 0.845 vs
non-chunked 0.857) and **8k runs clean** — no OOM (peak 33 GB / 96), no RoPE crash, ~14 s/seq, 8k MTP ≈ 0.87.
So the proper 1k/8k/32k/100k sweep is now a **single command**: `./h1env/h1_session.sh` (auto-stops itself).

**One open empirical risk for 100k only:** the dense MLA KV cache is stored decompressed (full-head), so it
grows ~linearly with context — 8k≈33 GB, 32k projected ~50–60 GB, **100k may approach the 96 GB ceiling per
stage.** If 100k OOMs, the fix is a stage-side change (cache the compressed kv-latent and re-expand per step,
leyten-style) — flagged, not yet needed. 8k/32k are comfortably within budget.

## 9. The one remaining manual step (offline, $0)

After a session, `dump_target_long.jsonl` (target argmax) is compared to the beast `dump_9b.jsonl` via
`h1_bench.py compare` to get the **9B** acceptance at 8k/32k (the 9B side was dumped free on beast; the box
only produces the target + MTP). Then update `H1-process-and-results.md` §6 with the full curve.

## 10. MLA-latent cache rewrite (unblocks 32k/100k + concurrency) — Phase 1+2 DONE offline

The 32k/100k wall is the decompressed full-head KV cache. Fix = DeepSeek MLA "absorb": cache the kv-latent
(`cprime`, ~kv_lora dims) + the shared rope key, and absorb `kv_b` into the query (scores) and output proj
(values) so full-head K/V is never materialized. **~70× smaller cache on GLM-5.2's real dims** (32k:
~88 GB → ~1.3 GB/stage; 100k → ~4 GB/stage).

- **`mla_latent.py` (offline, $0): PROVEN.** Two tests, both machine-precision equivalent: (1) absorbed vs
  naive single-forward (err 1e-13); (2) latent chunked-cache (append/crop + RoPE-across-chunks) vs
  all-at-once (err 5e-13). The risky algebra + the new plumbing are validated *before* any GPU.
- **`glm_swarm_nvfp4_kv.py`: integrated, gated by `MLA_LATENT=1`.** `Layer.__init__` precomputes the
  absorbed `W_kn`/`W_vb`; `_attn_latent` runs attention in latent space with `cc`/`rc` caches; `reset` and
  `gather_kv` handle the latent caches. Default (flag unset) = leyten's original full-head path, untouched —
  so it's a clean A/B. Syntax-checked.
- **Phase 3 (box, GATED on Joe's go — metered):** (a) `MLA_LATENT=1` at 1k must reproduce the known
  MTP/target numbers (vs `MLA_LATENT=0`) — the only thing not provable offline (real nvfp4 weights + cache
  plumbing end-to-end); (b) then 32k and 100k, which should sit at ~1–4 GB/stage. The turnkey
  `h1_session.sh` runs it (add `MLA_LATENT=1` to `h1_env.sh` for the latent run).
