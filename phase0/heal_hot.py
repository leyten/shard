"""HOT-standby mid-request fault tolerance: kill a node mid-generation, failover in SECONDS — not the
~131s cold reload of heal.py. Two things make the difference vs heal.py:

  * the SPARE is pre-launched WARM (weights already in VRAM) and its flex/inductor cache is pre-compiled,
    so it does NOT reload on failover (the reload was what dominated the 131s cold path); and
  * the victim's PREDECESSOR is REWIRED to the spare WITHOUT relaunching — the healer writes the spare's
    endpoint to /root/.shard_next_<pred> and the predecessor's existing relink loop reconnects there
    (engine: serve_spec_fast mk_fwd reads that override). No stage is relaunched, no weights reloaded.

So a node drop costs ~detect + reconnect + ONE re-prefill, not a model load. Flow:
  1. launch the N-stage ring + a hot spare (holding the victim's block, --next=successor); pre-warm flex.
  2. run a resumable generation; after --kill-after s, kill the victim's GPU process (drop under load).
  3. coordinator detects the dead ring, dumps the committed tokens, exits 3 (no tokens lost).
  4. HOT HEAL: write the spare endpoint to the predecessor's override -> it relinks to the warm spare.
  5. RESUME: re-invoke the coordinator with --resume-file -> re-prefill prompt+committed -> continue.

  python heal_hot.py --ring A,B,C,D --spare E --kill-stage 1 --kill-after 14 \
      --prompt-file /root/ft_prompt.txt --max-new 256 --max-ctx 16384

Teardown is manual (vastai destroy)."""
import argparse, time, json

from launch_oss import ep, fire, instances, rssh, warm_stage, M120, PORT, PSK
from launch_ngram import launch_stage_uneven
from heal import even_ranges, coord_cmd, gpu_kill, wait_ft


def prewarm_flex(inst, stage, nstages, lo, hi, max_ctx, label):
    """compile+cache the flex_attention kernel on this box's disk so the spare's FIRST prefill (during
    failover) is a cache hit, not a fresh ~30-60s compile. Runs a throwaway load+prefill, then exits
    (freeing VRAM) so the real serve process can load cleanly afterwards."""
    lohi = f"lo={lo},hi={hi}" if lo >= 0 else "lo=None,hi=None"
    py = ("import torch; from pipeline import load_stage; from fastverify import FastVerify; "
          f"p=load_stage('{M120}',{stage},{nstages},{lohi}); fv=FastVerify(p,maxlen={max_ctx}); fv.reset(); "
          "h=torch.zeros(1,128,p['_model'].config.hidden_size,dtype=torch.bfloat16,device='cuda'); "
          "fv.prefill(h,0); print('PREWARM_OK')")
    r = rssh(inst, f"cd /root && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SHARD_PSK={PSK} "
                   f"python3 -c \"{py}\" 2>&1 | tail -3", 900)
    ok = "PREWARM_OK" in (r.stdout + r.stderr)
    print(f"  prewarm {label}: {'OK' if ok else 'FAIL '+r.stdout[-200:]}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", required=True, help="live ring ids, head first")
    ap.add_argument("--spare", type=int, required=True, help="spare id (will hold the victim's block, pre-warmed)")
    ap.add_argument("--kill-stage", type=int, default=1, help="ring index to kill (a MIDDLE stage)")
    ap.add_argument("--kill-after", type=int, default=14)
    ap.add_argument("--prompt-file", default="/root/ft_prompt.txt")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-ctx", type=int, default=16384)
    ap.add_argument("--timeout", type=int, default=30, help="coordinator edge timeout (how fast a dead node surfaces)")
    ap.add_argument("--receipt", default="/root/heal_hot_receipt.json")
    ap.add_argument("--no-launch", action="store_true", help="ring + spare already warm; just run the kill/heal demo")
    a = ap.parse_args()
    ring_ids = [int(x) for x in a.ring.split(",")]
    insts = instances()
    ring = [insts[i] for i in ring_ids]; spare = insts[a.spare]
    head, tail = ring[0], ring[-1]; nstages = len(ring)
    assert a.kill_stage not in (0, nstages - 1), "kill a MIDDLE stage (not head/coordinator or tail)"
    victim = ring[a.kill_stage]; pred_idx = a.kill_stage - 1
    ranges = even_ranges(nstages); lo, hi = ranges[a.kill_stage]
    eps = [ep(s) for s in ring]
    succ_ep = f"{eps[a.kill_stage + 1][0]}:{eps[a.kill_stage + 1][1]}"
    spare_ep = f"{ep(spare)[0]}:{ep(spare)[1]}"
    tail_ep = f"{eps[-1][0]}:{eps[-1][1]}"
    print(f"[hot] ring {[r['id'] for r in ring]} victim=stage{a.kill_stage} {victim['id']} "
          f"spare={spare['id']} (block [{lo}:{hi}], --next {succ_ep})", flush=True)

    if not a.no_launch:
        # 1. ring tail-first (even split), then the HOT spare (pre-warm its flex first, then launch warm)
        print("[hot] launching ring tail-first ...", flush=True)
        for k in range(nstages - 1, -1, -1):
            nxt = f"{eps[k + 1][0]}:{eps[k + 1][1]}" if k < nstages - 1 else None
            klo, khi = ranges[k]
            launch_stage_uneven(ring[k], k, nstages, nxt, served_head=(k == 0), lo=klo, hi=khi,
                                max_ctx=a.max_ctx, timeout=a.timeout + 1770)
            _, ok = warm_stage(ring[k], f"stage{k} {ring[k]['id']}")
            print(f"  {'OK' if ok else 'FAIL'} stage{k}", flush=True)
            if not ok: print("[abort] ring stage failed to warm"); return
        print("[hot] pre-warming spare flex (disk cache) ...", flush=True)
        prewarm_flex(spare, a.kill_stage, nstages, lo, hi, a.max_ctx, f"spare {spare['id']}")
        print("[hot] launching HOT spare (warm standby, holds the victim's block) ...", flush=True)
        launch_stage_uneven(spare, a.kill_stage, nstages, succ_ep, served_head=False, lo=lo, hi=hi,
                            max_ctx=a.max_ctx, timeout=a.timeout + 1770)
        _, ok = warm_stage(spare, f"spare {spare['id']}")
        print(f"  spare {'OK' if ok else 'FAIL'}", flush=True)
        if not ok: print("[abort] spare failed to warm"); return

    # 2. start the resumable generation on the head (cold prefill warms the ring's flex), then kill mid-gen
    fire(head, "rm -f /root/ft.json /root/coord.log; " +
         coord_cmd(nstages, tail_ep, a.prompt_file, a.max_new, a.max_ctx, "/root/ft.json", a.timeout))
    print(f"[hot] generation started; killing stage {a.kill_stage} in {a.kill_after}s ...", flush=True)
    time.sleep(a.kill_after)
    t_kill = time.time()
    gpu_kill(victim)
    print(f"[hot] KILLED victim {victim['id']} at t={a.kill_after}s", flush=True)

    # 3. wait for the coordinator to surface the failure + the committed tokens
    d1 = wait_ft(head, "/root/ft.json", budget=60)
    if d1 is None:
        print("[hot] coordinator did not report; aborting"); return
    committed = d1.get("output_ids", []); detect_s = time.time() - t_kill
    print(f"[hot] death surfaced in ~{detect_s:.0f}s; committed {len(committed)} tok (ok={d1.get('ok')})", flush=True)
    if d1.get("ok"):
        print("[hot] request finished before the kill landed — raise --max-new / lower --kill-after"); return

    # 4. HOT HEAL: rewire the predecessor to the warm spare (NO relaunch, NO reload)
    t_heal = time.time()
    rssh(ring[pred_idx], f"echo '{spare_ep}' > /root/.shard_next_{pred_idx}", 30)
    print(f"[hot] rewired stage{pred_idx} ({ring[pred_idx]['id']}) -> spare {spare['id']} (no reload)", flush=True)

    # 5. RESUME: re-prefill prompt+committed on the healed ring (predecessor->spare->successor->...), continue
    healed_tail_ep = tail_ep                                    # tail unchanged (we kill a middle stage)
    # pass the FULL checkpoint (versioned envelope) through: the resume coordinator validates its
    # job/prompt/model/settings binding + token digest and refuses a stale/foreign checkpoint
    rssh(head, "cat > /root/ft2_in.json <<'EOF'\n" + json.dumps(d1) + "\nEOF", 30)
    resume_max = len(committed) + 24                            # short continuation: isolate the failover blip, not a long decode
    fire(head, "rm -f /root/ft.json /root/coord.log; " +
         coord_cmd(nstages, healed_tail_ep, a.prompt_file, resume_max, a.max_ctx, "/root/ft.json",
                   a.timeout + 1770, resume_file="/root/ft2_in.json"))
    print("[hot] resuming on the healed ring ...", flush=True)
    d2 = wait_ft(head, "/root/ft.json", budget=600)
    failover_s = time.time() - t_heal
    if d2 is None or not d2.get("ok"):
        print(f"[hot] resume did not complete: {d2}"); return
    total = d2.get("output_ids", [])
    reprefill_s = d2.get("prefill_s", 0.0)                      # the re-prefill of prompt+committed (the disruption)
    blip_s = detect_s + (failover_s - (len(total) - len(committed)) / max(d2.get("tok_s", 1e9), 1e-9))
    print(f"\n[hot] === REQUEST COMPLETED despite mid-request node death (HOT failover) ===", flush=True)
    print(f"  committed before drop : {len(committed)} tok", flush=True)
    print(f"  total after heal      : {len(total)} tok", flush=True)
    print(f"  continuation preserved: {total[:len(committed)] == committed}", flush=True)
    print(f"  detect (kill->surfaced)        : ~{detect_s:.0f}s", flush=True)
    print(f"  re-prefill (prompt+committed)  : {reprefill_s:.1f}s   <- the failover disruption; NO model reload", flush=True)
    print(f"  FAILOVER BLIP (detect+heal+reprefill, output interrupted) : ~{blip_s:.0f}s", flush=True)
    print(f"     vs COLD heal.py ~131s (dominated by the ~90s spare weight reload we now skip)", flush=True)
    print(f"  (total wall incl. the full {len(total)-len(committed)}-tok continuation decode: ~{failover_s:.0f}s)", flush=True)
    receipt = {"test": "hot-standby-fault-tolerance", "model": "gpt-oss-120b", "nstages": nstages,
               "victim_stage": a.kill_stage, "victim_id": victim["id"], "spare_id": spare["id"],
               "kill_after_s": a.kill_after, "detect_s": round(detect_s, 1),
               "committed_before_drop": len(committed), "total_after_resume": len(total),
               "continuation_preserved": (total[:len(committed)] == committed),
               "reprefill_s": round(reprefill_s, 1), "failover_blip_s": round(blip_s, 1),
               "total_wall_incl_continuation_s": round(failover_s, 1), "cold_baseline_s": 131,
               "mechanism": "hot spare (weights in VRAM, flex pre-cached) + predecessor rewire via /root/.shard_next_<pred> (no relaunch, no reload)",
               "output_text": d2.get("text", "")[:1200], "output_ids": total}
    rssh(head, "cat > %s <<'EOF'\n%s\nEOF" % (a.receipt, json.dumps(receipt)), 30)
    print(f"[hot] receipt -> {a.receipt} on {head['id']}", flush=True)
    print(f"\n  OUTPUT:\n{d2.get('text','')[:600]}", flush=True)


if __name__ == "__main__":
    main()
