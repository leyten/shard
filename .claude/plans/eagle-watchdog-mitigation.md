# P0-#5 EAGLE watchdog — mitigation-first (2026-07-20)

## The field failure
2026-07-14, residential-tail ring (Ghent 4090 behind relay): plain ring served ~2 tok/s (g=1, no
drafter, M25_EAGLE off everywhere). Turning EAGLE on silently hung the coordinator. Does NOT repro
on the CPU fake ring. Root cause UNKNOWN → mitigation must be generic: "worst case slower, never dead."

## What the code audit established (explore agent, verified by hand)
- NO infinite recv exists on the coordinator path. Decode recv = `_reply_timeout` 20s (F6,
  `m25_pipe.py:797/:813`). Reset-ack + prefill recvs = FULL job timeout (600s coordinate CLI /
  1800s gateway) — multi-minute wedge windows, not infinite.
- The serial EAGLE draft chain (`d_fetch()` at `m25_pipe.py:808` → `eagle_draft._draft`) is
  UNBUDGETED compute — a slow/stuck drafter today = silent crawl, no timeout fires. Likely 07-14 shape.
- A decode stall today = socket.timeout → EDGE_ERRORS → TransportError → `shard/coordinate.py`
  serve_jobs returns 1 → process exit → daemon restart. ABORT, not degrade (job dies; c0mpute
  fail-closed completes it — an error, not a hang, but also not "slower").
- EAGLE pins `cur_depth=1` (`:806`, re-read each iteration) — no pipelining; plain path runs depth=4.
- Aux (the EAGLE payload, 3 layers × [K+1,H] bf16 ≈ 166KB/hop solo) is attached per-stage in serve()
  (`fwd["aux"] = _merge_aux(...)` sites) gated ONLY by each stage's own M25_EAGLE env. The
  coordinator cannot silence it today → if the root cause is aux wedging the residential return
  leg, a coordinator-side-only degrade retries into the SAME hang.

## The design (3 layers)
- **L1 in-job degrade (m25_pipe.coordinate_pipe):** budget the draft step (healthy 10-13ms;
  `M25_DRAFT_BUDGET_S`). On breach → local `eagle_active=False`: drafter → ngram-only
  (HybridDrafter.disable_eagle(); bare Eagle → null), extend() stops, `:806` returns to pipelined
  `depth`. No wire change. Result dict records the degrade.
- **L2 job-level degraded retry (shard/coordinate.run_job):** resumable=True; on EAGLE-implicated
  EDGE abort → ONE retry: S.M25_EAGLE off process-locally, fresh ngram drafter,
  resume_ids=committed partial, SAME job_nonce, same on_commit state dict (text-cumulative deltas
  dedup automatically). Reset op gains `eagle:0` + serve() session honor so the degraded session
  carries NO aux on any leg (degraded arm == the proven-good 07-14 plain arm).
- **L3 stall backstop (shard/coordinate):** watchdog thread; no progress (commit OR prefill reply
  — coordinate_pipe gains an optional on_progress tick) for `M25_JOB_STALL_S` → SHARD_JOB_FATAL +
  hard exit → daemon restarts, fail-closed complete. Catches in-torch deadlock, sendall wedge,
  everything. Never a silent freeze.

## Status
- [x] Explore map (blocking points, seams)
- [x] Design panel synthesis — key corrections: flag-based disable only (empty proposals crash the
      fixed-K accept walk); L2 retry MUST re-dial (plain reset on old sockets eats a stale reply as
      its ack); L3 progress must tick on prefill replies (75MB aux chunk @0.5Mbps ≈ 20min legit);
      M1 = tail wedged in sendall on the UNTIMED ret = strongest root-cause candidate; sendall
      timeout = TOTAL deadline (verified empirically) → per-sendmsg-call stall bound instead.
- [x] Implemented on `eagle/watchdog` (4 atomic commits): L1 b6496a3, M1 5151b64, eagle:0 41978f0,
      L2+L3 3df198e. Full suite 584 passed, 1 skipped.
- [x] Stalled-tail proofs: test_coordinate_watchdog.py (10, incl. real-os._exit subprocess + lossless
      resume across the retry), test_ret_stallguard.py (4, real serve() tail survives a wedged ret),
      test_eagle_degrade.py (4), test_reset_eagle_flag.py (3).
- [x] Adversarial verify pass on the branch (1 defect D1 found+fixed) → PR #120 SQUASH-MERGED (4d5df7f).
- [x] Launcher knob forwarding (PR #122, merged): M25_RET_STALL_S / M25_DRAFT_BUDGET_S / M25_JOB_STALL_S /
      M25_REPLY_TIMEOUT added to ENG_ENV — they were DEAD in a real deployment (a flag a stage never
      receives does nothing). Found while setting up the ring; needed for a tunable stall bound.
- [x] Controlled-ring run (2026-07-20, ~$3.5, 3x5090 Poland/Germany/Poland, receipt
      `docs/receipts/eagle-watchdog-ring-20260720.json`). **Integration = PASS**: the shipped build
      + forwarded knobs serve M2.5 end-to-end over a real 3-region WAN ring, EAGLE on,
      degraded=false (no false trip), 3 signed receipts chained under one settlement nonce.
      **M1 wedge NOT reproducible in vivo — and that's the finding**: EAGLE is depth-1 (no return
      backlog), loopback buffers autotune to 4-6MB (one partial-model aux never fills them), and the
      container lacks NET_ADMIN (can't tc-throttle the tail uplink). That is precisely WHY 07-14
      only hit the residential box — the wedge is a sustained-backpressure phenomenon needing a slow
      uplink. M1's socket-level bound is already proven offline in test_ret_stallguard.py (real
      serve() tail, 32KB rcvbuf + 2MB aux = a genuine sendall wedge → recovers). Ring torn down (0 boxes).
- [ ] Daemon-side escalation (c0mpute follow-up, P11): on a stall-kill FATAL ("stall-watchdog" in
      error), restart the coordinator with M25_EAGLE=0 — the restart-degraded path for a wedged GPU.
- [ ] (optional, future) A residential-drain repro would need a box WITH NET_ADMIN (tc netem rate-limit
      on the tail uplink) or the full 62-layer model (~75MB prefill aux > buffer). Not blocking — the
      Standard 4-5 box EU M2.5 ring (brim config, torch-CUDA preflight every box). THREE arms:
      1. **Wedge arm (the 07-14 shape), master build:** EAGLE on; mid-decode kill the tail's return
         path WITHOUT closing TCP (iptables DROP on the ret leg — models the relay/conntrack path
         death). Expect: coordinator aborts on the 20s heartbeat but the TAIL stays wedged in the
         untimed ret sendall — no re-adopt, ring dead until relaunch. That reproduces "silent hang."
      2. **Wedge arm, branch build:** same fault → tail trips M25_RET_STALL_S → drops ret, stays
         alive → re-adopt + next job serves; coordinator side fails fast / retries degraded.
      3. **Thin-uplink arm, branch build:** tc egress ~2-5Mbps on the tail; EAGLE on. Expect slow
         but never dead: prefill progresses (L3 ticks), L1 may trip if drafting dominates, degrade
         completes the job (eagle:0 shrinks the wire after any retry).
      Dead-man switch: pin iids, heartbeat in every monitor loop, empty pin at teardown; keep the
      ring warm through the investigation.

KNOWN-ACCEPTED (M2): attempt-1 EAGLE prefill on a thin-uplink tail is legally slow (aux ~75MB/chunk
pre-degrade); the fix-at-source options (fp8 prefill aux, admission-gating EAGLE on tail uplink) are
post-mitigation levers. Documented, not built.

Baseline (branch point): tests/test_reply_heartbeat.py + test_fake_ring.py + test_shard_coordinate.py = 43 passed.
