# Shard fleet — 2026-06-23

## Session 3 (batched verify + async inter-stage send) — LIVE
Goal: #1 concurrent/continuous request batching (batched fast-verify CUDA graph) + #2 async
inter-stage send to cash in pipelined-prefill TTFT at 100k. Rented 6 distinct-host scattered US
4090s (cuda-13.2.1-auto, ALL with `--env '-p 29600:29600'`), genuinely scattered states:
| role | id | geo | ip | ssh | :29600 |
|------|-----|-----|-----|-----|--------|
| il (head) | 42248154 | Illinois | 104.12.231.85 | 40206 | 40242 |
| nv | 42248163 | Nevada | 173.239.95.142 | 41359 | 41370 |
| mi | 42248167 | Michigan | 216.234.102.170 | 14391 | 14482 |
| nj | 42248170 | New Jersey | 71.104.167.38 | 50596 | 50662 |
| ca | 42248173 | California | 192.234.50.153 | 3306 | 3323 |
| wa | 42248182 | Washington | 50.175.95.210 | 50232 | 50125 |
~$2.5/hr total. 6 distinct machine_ids (IL/NV/MI/NJ/CA/WA). Bootstrap: fleet.py push
setup_box.sh,get_model.py,stage_bootstrap.sh → fire stage_bootstrap detached → poll stage_ready.txt.
N=4 ring for the #2 baseline (vs old 193s@110k); N=6 to chase <60s. One box doubles as the #1
single-box batched-verify dev box. TEARDOWN: `vastai destroy instance <id>` per box when done.

RESULTS (committed): #1 batched verify proven on CA box (B=1 bit-exact, 1.6×@B4/2.1×@B8 throughput;
MoE token-count non-invariance isolated). #2 async-send A/B on the N=4 ring (IL·NV·CA·NJ):
30k 153.3→60.8s (2.52×), 110k 245.9→210.0s (1.17×, compute-bound). N=6 (MI+WA) and #3 hot-standby
NOT run (budget; documented as next levers). Fleet TORN DOWN after the commit (no idle spend).

## Session 3b (re-spin: N=6/5 TTFT + hot-standby + libp2p re-validation) — LIVE
User asked to re-spin for the 3 deferred levers. Rented 6 scattered US 4090s; NJ (42259379) had a
persistent SSH publickey failure (known bad-host pattern) → destroyed. **5 working boxes:**
| role | id | geo | ip | ssh | :29600 |
|------|-----|-----|-----|-----|--------|
| ca (head) | 42259365 | California | 192.234.50.153 | 5720 | 5744 |
| nv | 42259369 | Nevada | 173.239.95.142 | 41677 | 41684 |
| fl | 42259376 | Florida | 47.202.225.24 | 16537 | 16646 |
| il | 42259382 | Illinois | 104.12.231.85 | 40363 | 40369 |
| wa | 42259384 | Washington | 166.113.48.8 | 16826 | 16432 |
Pushed: full engine + `/tmp/sidecar` (libp2p, pre-built June-19, works) + `transport.py`. New tooling:
`heal_hot.py` (#3 hot standby: pre-warmed spare + predecessor rewire via `/root/.shard_next_<pred>`,
no reload), `launch_libp2p.py` (ring over the sidecar transport, SHARD_TRANSPORT=libp2p, no PSK).
Plan: N=5 TTFT (more-stages lever) → #3 hot-standby demo (4-ring + 1 spare) → libp2p re-validation.

RESULTS (committed): N=5 TTFT — 110k 201.7s (~4% better than N=4's 210, compute-bound), 30k 69.3s (WORSE
than 60.8 — more stages doesn't beat the compute wall). #3 hot-standby — kill NV mid-gen: 423 tok preserved,
re-prefill 8.9s, failover blip ~32.6s (vs 131s cold; ~90s reload eliminated), request completed. libp2p —
transport + engine-swap PROVEN (2MB round-trip 192ms; engines warm over libp2p), fresh full-ring gen blocked
by vast SSH/daemon-launch flakiness (fleet-ops, not engine). One box (FL 42259376) leaked 19GB VRAM (unclean
CUDA kill) — needs reset; destroyed with the rest. FLEET TORN DOWN.

## Session 3c (libp2p re-validation, 3rd fleet) — TORN DOWN
After 3b's libp2p gen was blocked, rented 4 fresh high-rel US boxes (CA·NV·IL·UT) to nail it. ROOT CAUSE
was a self-inflicted `launch_libp2p.py` bug, NOT libp2p: `launch_sidecar` ran `pkill -f /tmp/sidecar` but the
launch command string contains "/tmp/sidecar" → pkill self-matched + killed its own shell before the daemon
started (the documented specpipe self-match footgun); + the health-check grepped "listening" when tunnel-mode
sidecars log "tunnel up". Fixed both (fuser -k port instead of pkill; grep 'tunnel up|listening'; ssh-retry
hardening; --model flag). RESULT: a fresh 3-stage 120B ring (IL·UT·CA) ran a full distributed gen over the
libp2p sidecar (no PSK, per-node keys, async-send engine) — prefill 87.9s, 48 tok, coherent output, sha
d74c32c4. (NV was the slow-download laggard, never joined; 3-stage sufficed.) HF anon-pull throttling made
downloads crawl — wire HF_TOKEN into the bootstrap next time (token in session history, repo is public so
keep it in a gitignored secret). All 4 destroyed. FLEET TORN DOWN (no idle spend).

## Session 2 (deploy-readiness: sampling / TTFT / fault tolerance) — TORN DOWN
Rented 5 distinct-host scattered US 4090s (cuda-13.2.1-auto, `-p 29600:29600`): WA·MN·NC·NJ ring (even
N=4, 9 layers/box) + OH hot-spare. Used for: lossless speculative sampling (DONE), pipelined-prefill TTFT
A/B (partial), mid-request fault tolerance (demonstrated). All 5 destroyed when done — see git commits
`f3fba2d` (engine) + `39fdd5b` (receipts/docs). New tooling this session:
- `specpipe --temp/--top-p/--top-k/--seed` → lossless sampling; `--sample-test N` → on-swarm losslessness proof.
- `specpipe --prefill-depth D` → pipelined prefill (overlap chunks across stages).
- `phase0/heal.py --ring … --spare … --kill-stage k` → mid-request fault-tolerance demo (kill→heal→resume).
- `research/specsample_proof.py` → local (no-GPU) losslessness proof of the acceptance math.
Bring-up was `launch_ngram.py` with even split (omit `--layers`); coordinator on the head box.
Even N=4 (9/9/9/9 on four 24GB 4090s) fits ~110k KV with room — faster than the old heterogeneous 18/9/9
(110k prefill 226.8s vs ~556s). NOTE: still create boxes WITH `-p 29600:29600` (vast-expose-29600 memory).

## Session 1 (long-context perf) — TORN DOWN
Goal: ≥20 tok/s on >100k context, swarm on ≤4 neighbouring western states, trustworthy output.

## Instances (vast, account=leyten, key ~/.ssh/vast_c0mpute, image cuda-13.2.1-auto)
All DISTINCT host_id (no co-location). Created WITH `--env '-p 29600:29600'` (inter-stage
transport port — REQUIRED, see vast-expose-29600-port memory). N=3: the cuda-13.2 western pool
had only 3 distinct usable hosts (both CA hosts broke — 224600 ssh-key, 392559 docker-pull).
VRAM-aware uneven split: 48GB box holds 18 layers, two 24GB boxes 9 each (load_stage --lo/--hi).
NO separate coord box: model-free n-gram coordinator is CPU-only, runs ON the head box.
| id | label | role | host_id | VRAM | layers | $/hr |
|----|-------|------|---------|------|--------|------|
| 42195546 | shard-stage-wa2 | stage0 (head) + coordinator | 96690 | 48GB | [0:18] | 1.120 |
| 42195544 | shard-stage-wa1 | stage1 | 22965 | 24GB | [18:27] | 0.362 |
| 42195547 | shard-stage-tx | stage2 (tail) | 558496 | 24GB | [27:36] | 0.336 |

~$1.82/hr total. Ring: wa2(head,18L) -> wa1(9L) -> tx(tail,9L) -> return to wa2. 2 states (WA,TX).
STAGES=42195546,42195544,42195547  layers=18,9,9  head=42195546
Launch (brings up ring + runs ngram long-ctx coordinator):
  cd phase0 && SHARD_PSK=$(cat ~/.shard_psk) python3 launch_ngram.py --stages 42195546,42195544,42195547 --layers 18,9,9 --max-ctx 131072 --prompt-file /root/prompt_long.txt --prefill-chunk 4096 --depth 4 --K 4 --ngram-n 3 --max-new 256
Re-run coordinator on warm ring: add --no-launch. Window-KV: relaunch stages with FV_WINDOW=1.
VALIDATED 2026-06-23: short prompt -> coherent gpt-oss-120B output across WA->WA->TX.
(destroyed earlier: the no-29600 set, flaky CA hosts 224600 + 392559)

## Ops
- Control: `cd phase0 && python3 fleet.py ls|eps|wait|exec|push|warm`
- Bootstrap: setup_box.sh (deps) + get_model.py (120b stages, +20b coord)
- Launch: `python3 launch_oss.py --stages tx1,tx2,ca1,wa1 --coord ca2 --max-ctx 98304 ...`
- Teardown when done: `vastai destroy instance <id>` (per box)
- PSK: ~/.shard_psk (gitignored)

## Progress log
- 2026-06-23: fleet rented. Provisioning + 120b download starting.
- 2026-06-23: RESULT — **28.18 tok/s decode at >100k context** (past the 20 target), greedy-exact,
  receipt `docs/receipts/gpt-oss-120b-100k-ngram-20260623.json`. Signed per-stage receipts
  demonstrated live (PROVE). All work committed + pushed (origin/master, leyten/anon).
- 2026-06-23: **fleet torn down** (all instances destroyed — no idle spend). Re-spin via the Launch
  command above (re-rent 3 distinct western cuda-13.2 hosts WITH `-p 29600:29600`; ~25min model dl).
