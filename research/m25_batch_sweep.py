"""Batched-serving SWEEP over a live warm ring: B x AI-use-case x drafting arm.

The measurement the admission spec's batched numbers come from — replaces the single
adversarial counting prompt (n-gram g~1 floor, below any real workload) with a REALISTIC
use-case mix, and A/Bs the full drafting stack (hybrid n-gram->EAGLE per stream, the
solo stack now wired into coordinate_pipe_batch) against n-gram-only.

Arms:
  1. mixed-content B sweep (B = 1/2/4/8, hybrid drafting) — the headline batched curve
  2. per-use-case pure batches at B=4 (hybrid)             — which content earns what g
  3. drafting A/B at B=4 mixed (hybrid vs ngram-only)      — EAGLE's isolated lift

Run ON the head box of a --warm-only ring launched with M25_EAGLE=1 M25_BATCH=8:
  SHARD_TRANSPORT=libp2p HEAD_PORT=29610 TAIL_PORT=29612 M25_DIR=/root/m25 \
  M25_EAGLE=1 M25_EAGLE_DIR=/root/m25-eagle ... python -u m25_batch_sweep.py
Emits one RESULT json line per arm (the receipt's raw rows).
"""
import itertools
import json
import os
import socket
import time

import m25_stage as S
import m25_pipe as P
if os.environ.get("SHARD_TRANSPORT") != "libp2p":   # raw-wire mode: load SHARD_PSK (libp2p sidecar self-seals)
    import wire; wire.key_from_env()
from transformers import AutoTokenizer
from ngram_draft import NgramDrafter

HEAD = ("localhost", int(os.environ.get("HEAD_PORT", "29610")))
TAIL = ("localhost", int(os.environ.get("TAIL_PORT", "29612")))
EAGLE_DIR = os.environ.get("M25_EAGLE_DIR", "/root/m25-eagle")
MAX_NEW = int(os.environ.get("SWEEP_MAX_NEW", "96"))
K = int(os.environ.get("SWEEP_K", "8"))

# ---- the use-case suite (the "different AI use cases") --------------------------------
PASSAGE = (
    "The lighthouse at Cape Arran had stood for one hundred and forty years, its lamp turned "
    "by clockwork that the keepers wound every four hours. When the automation board voted to "
    "electrify the light in 1963, the last keeper, Ewan Morrison, refused to leave the island. "
    "He argued that the fog bell still required a human hand in winter, when ice made the "
    "striker stick, and that no relay could smell a storm coming the way a keeper could. The "
    "board relented for seven years, until a cable was laid from the mainland and the light "
    "became a number on a control panel ninety miles away. Morrison stayed on as caretaker "
    "without pay, tending the garden and the brass, and when he died the island passed to a "
    "seabird trust that keeps his logbooks in a glass case. Visitors still wind the clockwork "
    "once a year on the anniversary of the switch-on, a ceremony the trust calls the Turning."
)
CASES = {
    "code":      "Write a Python function that parses a CSV file and returns per-column statistics "
                 "(mean, min, max) as a dict. Include type hints and a docstring.",
    "prose":     "Write the opening three paragraphs of a short story about a lighthouse keeper who "
                 "finds a message in a bottle.",
    "reasoning": "A train leaves station A at 9:00 at 80 km/h. Another leaves station B, 240 km away, "
                 "at 9:30 at 100 km/h toward A. At what time do they meet? Think step by step.",
    "summarize": "Summarize the following passage in two sentences, then quote verbatim the single "
                 "sentence you consider most important.\n\n" + PASSAGE,
    "tools":     "Return a JSON object describing three European cities with fields: name, country, "
                 "population_estimate, landmark. JSON only, no prose.",
    "qa":        "Explain the difference between TCP and UDP in five short bullet points.",
}
MIX = list(CASES.values())                             # round-robin mixed-content pool

tok = AutoTokenizer.from_pretrained(S.DIR, trust_remote_code=True)
pipe = socket.create_connection(HEAD, timeout=600); pipe.setsockopt(*P.NODELAY)
ret = socket.create_connection(TAIL, timeout=600); ret.setsockopt(*P.NODELAY); ret.settimeout(600)
P.send_msg(ret, {"op": "hello_return"}); P.recv_msg(ret)

def make_drafters(B, kind):
    """Per-stream drafters: 'ngram' (the old batched floor) or 'hybrid' (the full solo stack via the
    ONE engine factory — forked EAGLE singleton + fresh n-gram per stream, same as the gateway)."""
    if kind == "ngram":
        return [NgramDrafter(ng=3) for _ in range(B)]
    return P.make_drafters_b(B)                        # honors M25_EAGLE (must be 1 for hybrid arms)


def warmup():
    """Untimed job per B shape BEFORE the timed arms: the first verify_batch of each (B, s) lazily
    CAPTURES the batched CUDA graphs on every stage (seconds, serialized along the ring) — inside a
    timed arm that pollutes the B-curve, worst at B=1. Runners persist across jobs on the warm
    stages, so one throwaway job per B pays the whole capture bill off the clock. SWEEP_WARMUP=0
    skips (e.g. a deliberately-eager M25_BATCH_GRAPH=0 pass has nothing to capture)."""
    for B in (1, 2, 4, 8):
        prompts = list(itertools.islice(itertools.cycle(MIX), B))
        msgs = [[{"role": "user", "content": p}] for p in prompts]
        r = P.coordinate_pipe_batch(pipe, tok, msgs, K, 8, 600, ret, make_drafters(B, "hybrid"),
                                    prefill_chunk=512, max_ctx=8192)
        print(f"[warmup] B={B} rounds={r['rounds']} graph_arm={r.get('graph_arm')}", flush=True)


def run_arm(name, prompts, kind):
    drafters = make_drafters(len(prompts), kind)
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    r = P.coordinate_pipe_batch(pipe, tok, msgs, K, MAX_NEW, 600, ret, drafters,
                                prefill_chunk=512, max_ctx=8192)
    row = {
        "arm": name, "B": len(prompts), "drafting": kind, "eagle": r["eagle"],
        "agg_tok_s": round(r["agg_tok_s"], 2), "rounds": r["rounds"], "depth": r["depth"],
        "wasted": r["wasted"], "dt": round(r["dt"], 2), "prefill_s": round(r["prefill_s"], 2),
        "receipts": len(r.get("receipts") or []), "receipts_ok": r.get("receipts_ok"),
        "graph_arm": r.get("graph_arm"),                # the tail's APPLIED route + counters (M25_GRAPH_JOB
                                                        # stamped jobs) — an arm must never lie about its route
        "aux_local": r.get("aux_local"),                # armed head-local lane vs ridden-ring aux (never conflate)
        "per_stream": [{"tok_s": round(s["n_tokens"] / max(r["dt"], 1e-9), 2), "g": s["g"],
                        "n": s["n_tokens"]} for s in r["streams"]],
        "g_mean": round(sum(s["g"] for s in r["streams"]) / len(r["streams"]), 3),
    }
    print("RESULT " + json.dumps(row), flush=True)
    for s in r["streams"][:2]:                          # eyeball coherence on the first two streams
        print(f"    [{name}] {s['text'][:100]!r}", flush=True)
    return row


def run_all():
    """The 12 arms, names/prompts/K/max_new UNCHANGED (apples-to-apples vs batched-sweep-eagle-20260710)."""
    rows = []
    # 1. mixed-content B sweep, full drafting
    for B in (1, 2, 4, 8):
        prompts = list(itertools.islice(itertools.cycle(MIX), B))
        rows.append(run_arm(f"mix-B{B}", prompts, "hybrid"))
    # 2. per-use-case pure batches at B=4, full drafting
    for case, prompt in CASES.items():
        rows.append(run_arm(f"{case}-B4", [prompt] * 4, "hybrid"))
    # 3. drafting A/B at B=4 mixed: EAGLE's isolated lift over the old n-gram-only floor
    rows.append(run_arm("mix-B4-ngram", MIX[:4], "ngram"))
    rows.append(run_arm("mix-B4-hybrid", MIX[:4], "hybrid"))
    return rows


print(f"=== SWEEP: K={K} max_new={MAX_NEW} eagle_dir={EAGLE_DIR} M25_EAGLE={S.M25_EAGLE} ===", flush=True)
results = []
# SWEEP_GRAPH_ARMS="off,on": run the WHOLE arm set once per batched-graph arm on ONE warm ring —
# per-job stamped via M25_GRAPH_JOB -> reset_batch (ack-verified on every job, so a silently-eager
# "on" pass is impossible). off = the old build's decode path (lever-1 isolation + ring-parity anchor
# vs the old receipt's 220ms ngram floor); on = the full new build. Unset = plain single pass
# (whatever the launch env routes).
GRAPH_PASSES = [p.strip() for p in os.environ.get("SWEEP_GRAPH_ARMS", "").split(",") if p.strip()]
# SWEEP_AUX_ARMS="off,on": the head-local-aux A/B on ONE warm ring — both passes run the full
# graphed build (M25_GRAPH_JOB stamped + ack'd); only the coordinator-side M25_AUX_LOCAL flips per
# pass (the head arms purely off the reset_batch field, so no stage relaunch). off = ridden-ring
# aux (with #78 slimming, the build default); on = + the head-local lane (#79). Rows tagged
# aux_pass; the per-job result also carries r["aux_local"] (the ARMED truth, never assumed).
AUX_PASSES = [p.strip() for p in os.environ.get("SWEEP_AUX_ARMS", "").split(",") if p.strip()]
# SWEEP_DELOCK_ARMS="off,on": the de-lockstep A/B on ONE warm ring — graph-stamped, aux_local per
# build default; only the coordinator-side dispatch flips per pass (rows vs lockstep frames).
DELOCK_PASSES = [p.strip() for p in os.environ.get("SWEEP_DELOCK_ARMS", "").split(",") if p.strip()]
if DELOCK_PASSES:
    P.M25_GRAPH_JOB = True
    if os.environ.get("SWEEP_WARMUP", "1") != "0":
        P.M25_DELOCKSTEP = True                        # warm BOTH graph shapes (row + batched)
        warmup()
        P.M25_DELOCKSTEP = False
        warmup()
    for p in DELOCK_PASSES:
        P.M25_DELOCKSTEP = (p == "on")
        print(f"=== PASS delockstep={p} ===", flush=True)
        results += [{**row, "delock_pass": p} for row in run_all()]
elif AUX_PASSES:
    P.M25_GRAPH_JOB = True
    P.M25_AUX_LOCAL = False
    if os.environ.get("SWEEP_WARMUP", "1") != "0":
        warmup()                                       # graph captures off the clock, once
    for p in AUX_PASSES:
        P.M25_AUX_LOCAL = (p == "on")
        print(f"=== PASS aux_local={p} ===", flush=True)
        results += [{**row, "aux_pass": p} for row in run_all()]
elif not GRAPH_PASSES:
    if os.environ.get("SWEEP_WARMUP", "1") != "0":
        warmup()
    results = run_all()
else:
    for p in GRAPH_PASSES:
        P.M25_GRAPH_JOB = (p == "on")
        print(f"=== PASS graph={p} ===", flush=True)
        if p == "on" and os.environ.get("SWEEP_WARMUP", "1") != "0":
            warmup()                                   # captures happen off the clock, once per (B,s)
        results += [{**row, "graph_pass": p} for row in run_all()]

print("=== SUMMARY ===", flush=True)
for row in results:
    print(f"  {row.get('graph_pass', '-'):>4} {row['arm']:>16} B={row['B']} {row['drafting']:>6}: "
          f"agg={row['agg_tok_s']:7.2f} g={row['g_mean']:.2f} "
          f"per-stream~{row['per_stream'][0]['tok_s']:.1f} rounds={row['rounds']} "
          f"receipts={row['receipts']} ok={row.get('receipts_ok')}", flush=True)
print("[sweep] done", flush=True)
