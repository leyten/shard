"""Mid-request fault tolerance demo + receipt: kill a node mid-generation, the SAME request still
completes (not a restart). The control-plane healer (what c0mpute's orchestrator would do) drives
the engine's resume primitive (specpipe `coordinate_pipe(resume_ids=..., resumable=True)`):

  1. start a generation on the warm N-node ring (coordinator on the head, --ft-dump => resumable;
     on a dead edge it dumps the committed tokens and exits 3 instead of failing the request).
  2. after --kill-after seconds, kill the GPU process on the victim stage (a real node drop UNDER
     LOAD — mid-decode).
  3. the coordinator detects the broken ring (edge timeout), writes {ok:false, output_ids:<committed>},
     exits. No tokens are lost.
  4. HEAL: re-form the ring with the pre-warmed SPARE in the victim's slot (it already holds the same
     layer block) and relaunch the stages tail-first (survivors re-handshake fresh; their KV is rebuilt
     by the re-prefill).
  5. RESUME: re-invoke the coordinator with --resume-file <committed>; it re-prefills prompt+committed
     on the healed ring and continues decoding to completion. The user sees one pause (the failover
     re-prefill), then continuous output — the request COMPLETES.

  python heal.py --ring 42234412,42234402,42234398,42234405 --spare 42234410 \
      --kill-stage 1 --kill-after 12 --prompt-file /root/ft_prompt.txt --max-new 256

--ring is the live ring (head first, tail last); --kill-stage is an index into it (kill a MIDDLE
stage, not the head/coordinator). Even layer split across the healed ring. Writes a receipt JSON.
Teardown is manual (vastai destroy)."""
import argparse, time, json

from launch_oss import ep, fire, instances, rssh, warm_stage, M120, PORT, PSK
from launch_ngram import launch_stage_uneven


def even_ranges(nstages, total=36):
    base, rem = total // nstages, total % nstages
    bounds, cur = [0], 0
    for s in range(nstages):
        cur += base + (1 if s < rem else 0); bounds.append(cur)
    return [(bounds[k], bounds[k + 1]) for k in range(nstages)]


def coord_cmd(nstages, tail_ep, prompt_file, max_new, max_ctx, ft_dump, timeout, resume_file=""):
    rf = f" --resume-file {resume_file}" if resume_file else ""
    return (f"cd /root && SHARD_PSK={PSK} setsid bash -c 'python3 specpipe.py --coordinator --nstages {nstages} "
            f"--model {M120} --ngram-draft --ngram-n 3 --pipe --depth 4 --K 4 --next 127.0.0.1:{PORT} "
            f"--direct-return --tail {tail_ep} --prompt-file {prompt_file} --prefill-chunk 4096 "
            f"--max-ctx {max_ctx} --max-new {max_new} --reasoning low --timeout {timeout} "
            f"--ft-dump {ft_dump}{rf} > /root/coord.log 2>&1' </dev/null >/dev/null 2>&1 &")


def gpu_kill(inst):
    """drop the node UNDER LOAD: kill its GPU compute process (the running stage). targeted by GPU
    pid -- never a pkill -f that could self-match. this severs the ring at this stage."""
    rssh(inst, "nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; "
               "fuser -k %d/tcp 2>/dev/null; echo KILLED" % PORT, 30)


def wait_ft(head, ft_dump, budget):
    """poll the head for the coordinator's ft-dump result (ok true=completed / false=node died)."""
    for _ in range(budget):
        r = rssh(head, f"cat {ft_dump} 2>/dev/null", 20)
        try:
            d = json.loads(r.stdout)
            return d
        except Exception:
            time.sleep(2)
    return None


def heal_minimal(healed, victim_idx, max_ctx, timeout):
    """MINIMAL heal: only the SPARE (new, at victim_idx) and the victim's PREDECESSOR (its --next now
    points at the spare) need (re)launch. Every other survivor already dropped its broken links on the
    failure and re-handshakes them on re-accept, so it keeps its loaded weights — no reload. This is
    what makes a node drop a localized repair (2 stage loads + one re-prefill) rather than a full ring
    cold-start. healed = ring with the spare substituted at victim_idx; head first."""
    nstages = len(healed); eps = [ep(s) for s in healed]; ranges = even_ranges(nstages)
    # spare first (so the predecessor has something to connect forward to)
    sp_next = f"{eps[victim_idx + 1][0]}:{eps[victim_idx + 1][1]}" if victim_idx < nstages - 1 else None
    lo, hi = ranges[victim_idx]
    launch_stage_uneven(healed[victim_idx], victim_idx, nstages, sp_next,
                        served_head=(victim_idx == 0), lo=lo, hi=hi, max_ctx=max_ctx, timeout=timeout)
    _, ok = warm_stage(healed[victim_idx], f"spare stage{victim_idx} {healed[victim_idx]['id']}")
    print(f"  heal spare {'OK' if ok else 'FAIL'} {healed[victim_idx]['id']}", flush=True)
    if not ok:
        return False
    pred = victim_idx - 1                            # predecessor's --next must now point at the spare
    lo, hi = ranges[pred]
    launch_stage_uneven(healed[pred], pred, nstages, f"{eps[victim_idx][0]}:{eps[victim_idx][1]}",
                        served_head=(pred == 0), lo=lo, hi=hi, max_ctx=max_ctx, timeout=timeout)
    _, ok = warm_stage(healed[pred], f"pred stage{pred} {healed[pred]['id']}")
    print(f"  heal pred  {'OK' if ok else 'FAIL'} {healed[pred]['id']}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", required=True, help="live ring ids, head first (e.g. WA,MN,NC,NJ)")
    ap.add_argument("--spare", type=int, required=True, help="pre-warmed spare id (holds the victim's block)")
    ap.add_argument("--kill-stage", type=int, default=1, help="ring index to kill (a MIDDLE stage, not head)")
    ap.add_argument("--kill-after", type=int, default=12, help="seconds into generation to kill the victim")
    ap.add_argument("--prompt-file", default="/root/ft_prompt.txt")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-ctx", type=int, default=131072)
    ap.add_argument("--timeout", type=int, default=30, help="coordinator edge timeout: how fast a dead node surfaces")
    ap.add_argument("--receipt", default="/root/heal_receipt.json")
    a = ap.parse_args()
    ring_ids = [int(x) for x in a.ring.split(",")]
    insts = instances()
    ring = [insts[i] for i in ring_ids]; spare = insts[a.spare]
    head, tail = ring[0], ring[-1]
    nstages = len(ring)
    tail_ep = f"{ep(tail)[0]}:{ep(tail)[1]}"
    victim = ring[a.kill_stage]
    assert a.kill_stage != 0, "kill a middle/tail stage, not the head (it runs the coordinator)"
    print(f"[ft] ring {[r['id'] for r in ring]} head={head['id']} tail={tail['id']} "
          f"victim=stage{a.kill_stage} {victim['id']} ({victim.get('geolocation')}) spare={spare['id']}", flush=True)

    # 1. start the generation (resumable) on the head, in the background
    fire(head, "rm -f /root/ft.json /root/coord.log; " +
         coord_cmd(nstages, tail_ep, a.prompt_file, a.max_new, a.max_ctx, "/root/ft.json", a.timeout))
    print(f"[ft] generation started; killing stage {a.kill_stage} ({victim['id']}) in {a.kill_after}s ...", flush=True)
    time.sleep(a.kill_after)

    # 2. kill the victim mid-generation (node drop under load)
    t_kill = time.time()
    gpu_kill(victim)
    print(f"[ft] KILLED victim {victim['id']} at t={a.kill_after}s into the request", flush=True)

    # 3. wait for the coordinator to surface the failure + hand back committed tokens
    d1 = wait_ft(head, "/root/ft.json", budget=60)
    if d1 is None:
        print("[ft] coordinator did not report; aborting", flush=True); return
    committed = d1.get("output_ids", [])
    detect_s = time.time() - t_kill
    print(f"[ft] node death surfaced in ~{detect_s:.0f}s; committed {len(committed)} tokens before the drop "
          f"(ok={d1.get('ok')}, {d1.get('error')})", flush=True)
    if d1.get("ok"):
        print("[ft] request finished BEFORE the kill landed — re-run with a larger --max-new / earlier --kill-after", flush=True)
        return

    # 4. HEAL: spare replaces the victim; (re)launch only the spare + the victim's predecessor
    healed = ring[:a.kill_stage] + [spare] + ring[a.kill_stage + 1:]
    print(f"[ft] healing: {[h['id'] for h in healed]} (spare {spare['id']} takes stage {a.kill_stage}); "
          f"relaunching spare + predecessor only ...", flush=True)
    t_heal = time.time()
    if not heal_minimal(healed, a.kill_stage, a.max_ctx, a.timeout + 1770):  # generous edge timeout for resume prefill
        print("[ft] heal failed to warm; aborting", flush=True); return
    healed_tail_ep = f"{ep(healed[-1])[0]}:{ep(healed[-1])[1]}"

    # 5. RESUME: re-prefill prompt+committed on the healed ring, continue to completion
    # stage the FULL checkpoint (versioned envelope, not bare output_ids) where the resume
    # coordinator reads it, THEN launch it — the coordinator validates the envelope's job/prompt/
    # model/settings binding + token digest, so a stale or foreign checkpoint is refused
    rssh(healed[0], "cat > /root/ft2_in.json <<'EOF'\n" + json.dumps(d1) + "\nEOF", 30)
    fire(healed[0], "rm -f /root/ft.json /root/coord.log; " +
         coord_cmd(nstages, healed_tail_ep, a.prompt_file, a.max_new, a.max_ctx, "/root/ft.json",
                   a.timeout + 1770, resume_file="/root/ft2_in.json"))
    print("[ft] resuming on the healed ring (re-prefill prompt+committed, continue) ...", flush=True)
    d2 = wait_ft(healed[0], "/root/ft.json", budget=600)
    heal_resume_s = time.time() - t_heal
    if d2 is None or not d2.get("ok"):
        print(f"[ft] resume did not complete: {d2}", flush=True); return
    total = d2.get("output_ids", [])
    print(f"\n[ft] === REQUEST COMPLETED despite mid-request node death ===", flush=True)
    print(f"  committed before drop : {len(committed)} tokens", flush=True)
    print(f"  total after heal+resume: {len(total)} tokens", flush=True)
    print(f"  failover (heal + re-prefill + resume): ~{heal_resume_s:.0f}s", flush=True)
    print(f"  continuation preserved : {total[:len(committed)] == committed}", flush=True)
    print(f"\n  OUTPUT:\n{d2.get('text','')[:800]}", flush=True)
    receipt = {"test": "mid-request-fault-tolerance", "model": "gpt-oss-120b", "nstages": nstages,
               "victim_stage": a.kill_stage, "victim_id": victim["id"], "spare_id": spare["id"],
               "kill_after_s": a.kill_after, "committed_before_drop": len(committed),
               "total_after_resume": len(total), "continuation_preserved": (total[:len(committed)] == committed),
               "failover_s": round(heal_resume_s, 1), "output_text": d2.get("text", ""),
               "committed_ids": committed, "output_ids": total}
    rssh(healed[0], "cat > %s <<'EOF'\n%s\nEOF" % (a.receipt, json.dumps(receipt)), 30)
    print(f"\n[ft] receipt -> {a.receipt} on {healed[0]['id']}", flush=True)


if __name__ == "__main__":
    main()
