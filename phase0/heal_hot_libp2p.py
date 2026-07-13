"""HOT-standby mid-request failover OVER THE LIBP2P TRANSPORT (per-node keys, NO PSK) — the libp2p
analogue of heal_hot.py. Kill a node mid-generation; the SAME request finishes in seconds, not a
restart, with the activations crossing the real permissionless transport.

What's different vs the raw-TCP heal_hot.py: the inter-stage links go through the libp2p sidecars,
so the predecessor engine only ever dials its LOCAL sidecar (:29611), which carries the bytes over
libp2p to the next node. That local socket stays ALIVE when the remote victim dies — so unlike
raw-TCP (a direct socket to the victim that breaks on death), the dead peer never breaks the pred's
socket and the engine never relinks. The fix: at heal time we RELAUNCH the predecessor's sidecar with
its ring-forward :29611 repointed to the warm SPARE. That breaks the pred engine's local socket -> it
relinks -> reconnects to :29611 (now tunneled to the spare) -> the ring is whole. No engine relaunch,
no weight reload: a node drop costs detect + sidecar-repoint + ONE re-prefill.

Flow:
  1. ring sidecars + a HOT spare (its own sidecar: inbound->engine, forward 29611->victim's successor;
     engine holds the victim's block, pre-warmed). Then PRE-WARM the ring (throwaway gen) so the real
     gen's decode graphs are already captured and a kill lands mid-decode (not in the cold first round).
  2. resumable libp2p gen on the head; after --kill-after s, kill the victim's GPU process.
  3. coordinator detects the dead ring, dumps committed tokens, exits 3 (no tokens lost).
  4. HOT HEAL: relaunch the predecessor's sidecar with -forward 29611=<spare_maddr> -> pred engine
     relinks to the warm spare (still dialing only localhost; the sidecar does the network).
  5. RESUME (retry): re-invoke the coordinator with --resume-file -> re-prefill prompt+committed ->
     continue (the first attempt triggers the pred's relink; the next rides the healed path to done).

  SHARD_PSK=$(cat ~/.shard_psk) python3 heal_hot_libp2p.py --ring A,B,C,D --spare E \
      --kill-stage 1 --kill-after 24 --prompt-file /root/prompt_30k.txt --max-ctx 40960

Teardown is manual (vastai destroy)."""
import argparse, time, json

from launch_oss import ep, instances, rssh, fire, warm_stage, M120
from launch_libp2p import (peerid, maddr, launch_sidecar, launch_engine,
                           LIBP2P, ENG_IN, FWD_RING, FWD_RET)
from heal import even_ranges, gpu_kill, wait_ft


def libp2p_coord_cmd(nstages, prompt_file, max_new, max_ctx, ft_dump, timeout,
                     depth, K, ngram_n, prefill_chunk, resume_file=""):
    """resumable n-gram coordinator over libp2p (SHARD_TRANSPORT=libp2p, no PSK): --ft-dump makes the
    request resumable (on a dead edge it writes {ok:false, output_ids:<committed>} and exits 3)."""
    rf = f" --resume-file {resume_file}" if resume_file else ""
    return (f"cd /root && SHARD_TRANSPORT=libp2p setsid bash -c 'python3 specpipe.py --coordinator "
            f"--nstages {nstages} --model {M120} --ngram-draft --ngram-n {ngram_n} --pipe --depth {depth} "
            f"--K {K} --next 127.0.0.1:{ENG_IN} --direct-return --tail 127.0.0.1:{FWD_RET} "
            f"--prompt-file {prompt_file} --prefill-chunk {prefill_chunk} --max-ctx {max_ctx} "
            f"--max-new {max_new} --reasoning low --timeout {timeout} --ft-dump {ft_dump}{rf} "
            f"> /root/coord.log 2>&1' </dev/null >/dev/null 2>&1 &")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", required=True, help="live ring ids, head first")
    ap.add_argument("--spare", type=int, required=True, help="spare id (holds the victim's block, pre-warmed)")
    ap.add_argument("--kill-stage", type=int, default=1, help="ring index to kill (a MIDDLE stage)")
    ap.add_argument("--kill-after", type=int, default=24)
    ap.add_argument("--prompt-file", default="/root/prompt_30k.txt")
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--max-ctx", type=int, default=40960)
    ap.add_argument("--timeout", type=int, default=30, help="coordinator edge timeout (how fast a dead node surfaces)")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--ngram-n", type=int, default=3)
    ap.add_argument("--prefill-chunk", type=int, default=4096)
    ap.add_argument("--receipt", default="/root/heal_hot_libp2p_receipt.json")
    ap.add_argument("--no-launch", action="store_true", help="ring + spare already warm; just run the kill/heal demo")
    a = ap.parse_args()
    ring_ids = [int(x) for x in a.ring.split(",")]
    insts = instances()
    ring = [insts[i] for i in ring_ids]; spare = insts[a.spare]
    head, tail = ring[0], ring[-1]; nstages = len(ring)
    assert a.kill_stage not in (0, nstages - 1), "kill a MIDDLE stage (not head/coordinator or tail)"
    victim = ring[a.kill_stage]; pred_idx = a.kill_stage - 1
    ranges = even_ranges(nstages); lo, hi = ranges[a.kill_stage]
    print(f"[hot-libp2p] ring {[r['id'] for r in ring]} victim=stage{a.kill_stage} {victim['id']} "
          f"pred=stage{pred_idx} {ring[pred_idx]['id']} spare={spare['id']} (block [{lo}:{hi}])", flush=True)

    # PeerIds/multiaddrs for ring + spare — needed BOTH to launch AND to repoint the predecessor at heal time
    print("[hot-libp2p] collecting PeerIds (ring + spare) ...", flush=True)
    pids = [peerid(s) for s in ring]; spid = peerid(spare)
    maddrs = [maddr(ring[k], pids[k]) for k in range(nstages)]
    spare_maddr = maddr(spare, spid)
    succ_maddr = maddrs[a.kill_stage + 1]
    pred_ann = maddr(ring[pred_idx], pids[pred_idx]).rsplit("/p2p/", 1)[0]
    pred_inbound = f"127.0.0.1:{ENG_IN}" if pred_idx > 0 else ""
    print(f"  spare {spare['id']} ({spare.get('geolocation')}) {spare_maddr}", flush=True)

    if not a.no_launch:
        # ring sidecars (like launch_libp2p): inbound (k>0) + ring-forward + the head's coordinator-return tunnel
        print("[hot-libp2p] launching ring sidecars ...", flush=True)
        for k, s in enumerate(ring):
            ann = maddr(s, pids[k]).rsplit("/p2p/", 1)[0]
            forwards, inbound = [], (f"127.0.0.1:{ENG_IN}" if k > 0 else "")
            if k < nstages - 1:
                forwards.append(f"127.0.0.1:{FWD_RING}={maddrs[k + 1]}")
            if k == 0:
                forwards.append(f"127.0.0.1:{FWD_RET}={maddrs[-1]}")
            launch_sidecar(s, ann, inbound, forwards)
        # spare sidecar: inbound -> spare engine; forward 29611 -> the victim's successor (so the spare slots in)
        launch_sidecar(spare, spare_maddr.rsplit("/p2p/", 1)[0], f"127.0.0.1:{ENG_IN}",
                       [f"127.0.0.1:{FWD_RING}={succ_maddr}"])
        time.sleep(4)

        # ring engines tail-first (even split), then the HOT spare engine (holds the victim's block)
        print("[hot-libp2p] launching ring engines tail-first ...", flush=True)
        for k in range(nstages - 1, -1, -1):
            launch_engine(ring[k], k, nstages, served_head=(k == 0), max_ctx=a.max_ctx,
                          timeout=a.timeout + 1770, sync_send=False, model=M120,
                          lo=ranges[k][0], hi=ranges[k][1])
            _, ok = warm_stage(ring[k], f"stage{k} {ring[k]['id']}")
            print(f"  {'OK' if ok else 'FAIL'} stage{k}", flush=True)
            if not ok:
                print(rssh(ring[k], "tail -5 /root/sidecar.log; echo ---; tail -8 /root/stage.log", 30).stdout, flush=True)
                print("[abort] ring stage failed to warm"); return
        print("[hot-libp2p] launching HOT spare engine (warm standby, holds the victim's block) ...", flush=True)
        launch_engine(spare, a.kill_stage, nstages, served_head=False, max_ctx=a.max_ctx,
                      timeout=a.timeout + 1770, sync_send=False, model=M120, lo=lo, hi=hi)
        _, ok = warm_stage(spare, f"spare {spare['id']}")
        print(f"  spare {'OK' if ok else 'FAIL'}", flush=True)
        if not ok:
            print(rssh(spare, "tail -5 /root/sidecar.log; echo ---; tail -8 /root/stage.log", 30).stdout, flush=True)
            print("[abort] spare failed to warm"); return

    # 1b. PRE-WARM the ring with a throwaway short gen so the CUDA graphs are captured ACROSS the ring —
    # otherwise the REAL gen's first decode rounds (cold graph capture over 4 WAN stages) take 45-90s and a
    # kill at ~20s lands before any token commits (committed=0). After this, the real gen decodes immediately.
    print("[hot-libp2p] pre-warming the ring (throwaway gen, captures the decode graphs) ...", flush=True)
    fire(head, "rm -f /root/ftwarm.json /root/coord.log; " +
         libp2p_coord_cmd(nstages, a.prompt_file, 12, a.max_ctx, "/root/ftwarm.json", a.timeout + 600,
                          a.depth, a.K, a.ngram_n, a.prefill_chunk))
    dw = wait_ft(head, "/root/ftwarm.json", budget=240)
    print(f"  pre-warm {'done' if dw else 'TIMED OUT (continuing)'}", flush=True)

    # 2. resumable gen on the WARM ring, then kill mid-decode (decode now starts right after the short prefill)
    fire(head, "rm -f /root/ft.json /root/coord.log /root/.shard_next_*; " +
         libp2p_coord_cmd(nstages, a.prompt_file, a.max_new, a.max_ctx, "/root/ft.json", a.timeout,
                          a.depth, a.K, a.ngram_n, a.prefill_chunk))
    print(f"[hot-libp2p] generation started; killing stage {a.kill_stage} in {a.kill_after}s ...", flush=True)
    time.sleep(a.kill_after)
    t_kill = time.time()
    gpu_kill(victim)
    print(f"[hot-libp2p] KILLED victim {victim['id']} at t={a.kill_after}s", flush=True)

    # 3. wait for the coordinator to surface the failure + the committed tokens
    d1 = wait_ft(head, "/root/ft.json", budget=90)
    if d1 is None:
        print("[hot-libp2p] coordinator did not report; aborting"); return
    committed = d1.get("output_ids", []); detect_s = time.time() - t_kill
    print(f"[hot-libp2p] death surfaced in ~{detect_s:.0f}s; committed {len(committed)} tok (ok={d1.get('ok')})", flush=True)
    if d1.get("ok"):
        print("[hot-libp2p] request finished before the kill landed — raise --max-new / lower --kill-after"); return

    # 4. HOT HEAL: relaunch the PREDECESSOR's SIDECAR with its ring-forward (29611) repointed to the SPARE.
    # libp2p needs this (vs raw-TCP's .shard_next ip:port rewire): the pred engine only ever dials its LOCAL
    # sidecar :29611, which stays alive when the remote victim dies — so the dead peer never breaks the pred's
    # socket and the engine never relinks. Restarting the pred sidecar DOES break that local socket: the pred
    # engine relinks, reconnects to :29611 (now tunneled to the warm spare), and the ring is whole again. No
    # engine relaunch, no weight reload — the spare was warm the whole time.
    t_heal = time.time()
    pred_forwards = [f"127.0.0.1:{FWD_RING}={spare_maddr}"]
    if pred_idx == 0:
        pred_forwards.append(f"127.0.0.1:{FWD_RET}={maddrs[-1]}")    # keep the head's coordinator-return tunnel
    launch_sidecar(ring[pred_idx], pred_ann, pred_inbound, pred_forwards)
    print(f"[hot-libp2p] repointed stage{pred_idx} ({ring[pred_idx]['id']}) sidecar :{FWD_RING} -> spare "
          f"{spare['id']} (no engine relaunch, no reload)", flush=True)

    # 5. RESUME (retry): the FIRST resume's reset is what makes the pred engine notice its now-broken forward
    # socket (the old sidecar died) and relink to the repointed :29611 -> spare; that attempt drops fast, the
    # next one rides the healed path to completion. Re-prefills prompt+committed, continues to completion.
    # pass the FULL checkpoint (versioned envelope) through: the resume coordinator validates its
    # job/prompt/model/settings binding + token digest and refuses a stale/foreign checkpoint
    rssh(head, "cat > /root/ft2_in.json <<'EOF'\n" + json.dumps(d1) + "\nEOF", 30)
    resume_max = len(committed) + 24                            # short continuation: isolate the failover blip
    d2 = None
    for attempt in range(4):
        fire(head, "rm -f /root/ft.json /root/coord.log; " +
             libp2p_coord_cmd(nstages, a.prompt_file, resume_max, a.max_ctx, "/root/ft.json", a.timeout + 1770,
                              a.depth, a.K, a.ngram_n, a.prefill_chunk, resume_file="/root/ft2_in.json"))
        print(f"[hot-libp2p] resume attempt {attempt + 1} on the healed ring (over libp2p) ...", flush=True)
        d2 = wait_ft(head, "/root/ft.json", budget=240)
        if d2 is not None and d2.get("ok"):
            break
        print(f"  attempt {attempt + 1} didn't complete (pred still relinking to the spare); retrying ...", flush=True)
        time.sleep(3)
    failover_s = time.time() - t_heal
    if d2 is None or not d2.get("ok"):
        print(f"[hot-libp2p] resume did not complete after retries: {d2}"); return
    total = d2.get("output_ids", [])
    reprefill_s = d2.get("prefill_s", 0.0)
    blip_s = detect_s + (failover_s - (len(total) - len(committed)) / max(d2.get("tok_s", 1e9), 1e-9))
    preserved = total[:len(committed)] == committed
    print(f"\n[hot-libp2p] === REQUEST COMPLETED despite mid-request node death (HOT failover over LIBP2P) ===", flush=True)
    print(f"  committed before drop : {len(committed)} tok", flush=True)
    print(f"  total after heal      : {len(total)} tok", flush=True)
    print(f"  continuation preserved: {preserved}", flush=True)
    print(f"  detect (kill->surfaced)        : ~{detect_s:.0f}s", flush=True)
    print(f"  re-prefill (prompt+committed)  : {reprefill_s:.1f}s   <- the failover disruption; NO model reload", flush=True)
    print(f"  FAILOVER BLIP (detect+heal+reprefill) : ~{blip_s:.0f}s", flush=True)
    receipt = {"test": "hot-standby-fault-tolerance-LIBP2P", "model": "gpt-oss-120b", "transport": "libp2p (per-node keys, no PSK)",
               "nstages": nstages, "victim_stage": a.kill_stage, "victim_id": victim["id"], "spare_id": spare["id"],
               "kill_after_s": a.kill_after, "detect_s": round(detect_s, 1),
               "committed_before_drop": len(committed), "total_after_resume": len(total),
               "continuation_preserved": preserved, "reprefill_s": round(reprefill_s, 1),
               "failover_blip_s": round(blip_s, 1), "total_wall_incl_continuation_s": round(failover_s, 1),
               "mechanism": "pre-wired spare libp2p tunnel (extra -forward on pred sidecar) + predecessor engine "
                            "rewire via /root/.shard_next_<pred> (no sidecar relaunch, no weight reload)",
               "output_text": d2.get("text", "")[:1200], "output_ids": total}
    rssh(head, "cat > %s <<'EOF'\n%s\nEOF" % (a.receipt, json.dumps(receipt)), 30)
    print(f"[hot-libp2p] receipt -> {a.receipt} on {head['id']}", flush=True)
    print(f"\n  OUTPUT:\n{d2.get('text','')[:600]}", flush=True)


if __name__ == "__main__":
    main()
