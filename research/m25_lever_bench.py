"""Lever bench: INTERLEAVED A/B of the two ring-detached perf levers on ONE warm ring —
CUDA-graph aux (M25_GRAPH_JOB) and cwnd keep-warm (M25_KEEPWARM_JOB) — over both drafter regimes
(chain / depth-aware hybrid). The m25_paper_bench methodology (cells interleaved, arm order rotated
per rep, receipts per job) applied to lever attribution: every arm is a per-job runtime flip on the
same warm ring, so stage relaunch drift never touches the comparison.

Arms (all coordinator-side; stages launched with M25_STATIC_KV=1 M25_EAGLE=1 and NEITHER lever
forced, so the reset op arms/disarms them per job):
  chain        tree=0 graph=0 warm=0     — the merged-master baseline regime
  chain+G      tree=0 graph=1 warm=0     — graph verifies every s=K+1 frame (the slow-CPU lever)
  chain+W      tree=0 graph=0 warm=150   — keep every leg's cwnd warm across the serial idle
  chain+GW     tree=0 graph=1 warm=150
  hybrid       tree=1 graph=0 warm=0     — the shipped-best regime baseline
  hybrid+GW    tree=1 graph=1 warm=150   — the shipped stack (tree rounds stay eager by design)

Per (tree,graph) combo one UNRECORDED warmup job pays the graph captures (review F8) before any
recorded rep. The bench ends with an explicit graph=0/warm=0 job so the ring never leaks an
experiment arm to later users (review F5). If the coordinator's graph request is refused by the
ring (F2 ack assert), the bench ABORTS loudly rather than banking a mislabeled arm.

Run on the head box as the SOLE coordinator on a --warm-only ring:
  SHARD_TRANSPORT=libp2p HEAD_PORT=29610 TAIL_PORT=29612 CUDA_VISIBLE_DEVICES=0 M25_DIR=/root/m25 \
  M25_EAGLE=1 M25_EAGLE_DIR=/root/m25-eagle M25_STATIC_KV=1 M25_FP8_WIRE=1 M25_STAGE_TIMING=1 \
  SHARD_RECEIPTS=1 /root/venv/bin/python -u m25_lever_bench.py --reps 3
Writes /root/lever_bench.json + prints the per-cell and decode-weighted improvement table.
"""
import argparse
import json
import os
import socket
import statistics
import time

import m25_stage as S
import m25_pipe as P
from m25_tools import parse_completion
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(S.DIR, trust_remote_code=True)
HEAD = ("127.0.0.1", int(os.environ.get("HEAD_PORT", "29610")))
TAIL = ("127.0.0.1", int(os.environ.get("TAIL_PORT", "29612")))


def _connect():
    """(Re)establish the coordinator's two sockets onto the (self-healed) warm ring: pipe->head,
    ret<-tail + the hello_return handshake. A 60-min scattered-WAN bench WILL hit transient sidecar
    blips; the RING self-heals its internal links (churn recovery), but the coordinator's own sockets
    don't — so run_job reconnects here and retries the failed job (the engine's mid-session
    hello_return re-adoption is exactly this path). On a fresh drop the ring takes seconds-to-tens-of-
    seconds to rebuild forward links + re-accept a mid-session hello_return, so this RETRIES with growing
    backoff (bounded hello ack wait) — an 8s one-shot raced the heal and crashed the run."""
    last = None
    for i in range(8):
        p = r = None
        try:
            p = socket.create_connection(HEAD, timeout=1800); p.setsockopt(*P.NODELAY)
            r = socket.create_connection(TAIL, timeout=30); r.setsockopt(*P.NODELAY)
            P.send_msg(r, {"op": "hello_return"}); P.recv_msg(r)   # 30s ack wait: a not-yet-ready tail fails fast -> retry
            r.settimeout(1800)
            return p, r
        except Exception as e:                                    # tail/head not healed yet: close partial, back off, retry
            last = e
            for s in (p, r):
                try:
                    if s is not None: s.close()
                except OSError: pass
            time.sleep(5 + i * 5)                                 # 5,10,...,40s -> ~180s total for the ring to heal
    raise RuntimeError(f"reconnect failed after 8 tries: {type(last).__name__}: {last}")


pipe, ret = _connect()

K, DEPTH = 8, 4
WARM_MS = int(os.environ.get("LEVER_WARM_MS", "150"))     # keep-warm interval for the W arms

ARMS = {  # name -> (tree, graph, warm_ms)
    "chain":     (False, False, 0),
    "chain+G":   (False, True, 0),
    "chain+W":   (False, False, WARM_MS),
    "chain+GW":  (False, True, WARM_MS),
    "hybrid":    (True, False, 0),
    "hybrid+GW": (True, True, WARM_MS),
}
BASE_OF = {"chain+G": "chain", "chain+W": "chain", "chain+GW": "chain", "hybrid+GW": "hybrid"}


def _doc_tokens(path, n):
    ids = tok(open(path).read(), add_special_tokens=False)["input_ids"][:n]
    return tok.decode(ids)


DOC = _doc_tokens("/root/m25_pipe.py", 3500) if os.path.exists("/root/m25_pipe.py") else ""
WEATHER = [{"type": "function", "function": {"name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
CELLS = [  # (cell, prompt, tools, max_new) — one cell per traversal regime the levers touch
    ("reason-math", "A farmer has 17 sheep. All but 9 run away. How many sheep are left? Think it through, then give the number.", None, 256),
    ("reason-logic", "Three light switches outside a windowless room each control one of three bulbs inside. You may flip switches as much as you like, but you can enter the room only once. How do you determine which switch controls which bulb? Reason step by step.", None, 256),
    ("agentic-tool", "What's the current weather in Tokyo? Use the get_weather tool.", WEATHER, 192),
    ("rag-quote", "Here is some code:\n\n" + DOC + "\n\nWhat dictionary does coordinate_pipe return on success? Quote the exact return statement from the code.", None, 256),
    ("copy-verbatim", "Repeat the following text exactly, word for word, with no commentary:\n\n" + DOC[:4000], None, 320),
]
if os.path.exists("/root/prompt_long.txt"):
    _doc8k = _doc_tokens("/root/prompt_long.txt", 8000)
    CELLS.append(("ctx-8k-quote", _doc8k + "\n\nRepeat the final paragraph of the document above exactly, word for word.", None, 192))
if os.environ.get("LEVER_CELLS"):                          # focused subset (e.g. a keepwarm x graph confirmation)
    _want = [c.strip() for c in os.environ["LEVER_CELLS"].split(",")]
    CELLS = [c for c in CELLS if c[0] in _want]


def _set_arm(tree, graph, warm_ms):
    """Arm the levers for the NEXT job. Regime + graph ride MODULE globals the coordinator reads
    (P.M25_GRAPH_JOB is captured at import, so os.environ won't reach _reset_op — set it directly,
    as bool: True/False both emit an explicit reset 'graph' field so the ack-assert fires and the
    OFF arm actively disables graph on the stages). keepwarm's kw_job is read at CALL time inside
    coordinate_pipe, so os.environ is correct there."""
    S.M25_TREE = tree
    P.M25_GRAPH_JOB = graph
    os.environ["M25_KEEPWARM_JOB"] = str(warm_ms)


def run_job(arm, cell, prompt, tools, max_new, record=True, retries=3):
    global pipe, ret
    tree, graph, warm_ms = ARMS[arm] if arm in ARMS else arm
    _set_arm(tree, graph, warm_ms)
    msgs = [{"role": "user", "content": prompt}]
    t0 = time.time()
    r = None
    for attempt in range(retries):
        _set_arm(tree, graph, warm_ms)                     # re-arm each attempt (module globals survive, but be explicit)
        drafter = P.make_drafter(3)                        # fresh drafter per attempt (clean n-gram/EAGLE state)
        try:
            t0 = time.time()
            r = P.coordinate_pipe(pipe, tok, msgs, K, max_new, 120, DEPTH, ret_sock=ret,
                                  local_draft=drafter, prefill_chunk=1024, max_ctx=131072, reasoning=True)
            break
        except Exception as e:                             # edge died mid-job: the ring self-heals its links; reconnect the coordinator + retry
            if attempt == retries - 1:
                print(f"  [skip] {arm} {cell} failed {retries}x ({type(e).__name__}: {str(e)[:80]})", flush=True)
                return None
            print(f"  [retry {attempt+1}] {arm} {cell} edge failed ({type(e).__name__}); healing + reconnecting", flush=True)
            for s in (pipe, ret):
                try: s.close()
                except OSError: pass
            time.sleep(8)                                  # let the ring rebuild its internal forward links before we re-hello
            try:
                pipe, ret = _connect()                     # _connect retries with backoff internally
            except Exception as ce:                        # ring still not healed after ~180s — leave attempt to the loop
                print(f"  reconnect not ready ({type(ce).__name__}); continuing retry loop", flush=True)
    if not record or r is None:
        return None
    name = arm if isinstance(arm, str) else "?"
    return {"arm": name, "cell": cell, "tree": tree, "graph": graph, "warm_ms": warm_ms,
            "wall_s": round(time.time() - t0, 2),
            "prompt_tokens": r["prompt_tokens"], "new_tokens": r["n_tokens"],
            "tok_s": round(r["tok_s"], 3), "g": round(r["toks_per_traversal"], 3),
            "accept": round(r["mean_accept"], 3), "rounds": r["rounds"], "wasted": r["wasted"],
            "prefill_s": round(r["prefill_s"], 2), "decode_s": r["decode_s"],
            "draft_s": r["draft_s"], "ring_wait_s": r["ring_wait_s"],
            "traversal_s": r.get("traversal_s"), "transport_s": r.get("transport_s"),
            "stage_s": r.get("stage_s"), "stage_compute_s": r.get("stage_compute_s"),
            "per_stage_ms": r.get("per_stage_ms"), "receipts_ok": r.get("receipts_ok"),
            "graph_arm": r.get("graph_arm"),
            "answer_head": (parse_completion(r["text"])["content"] or "")[:160]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default="/root/lever_bench.json")
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    # RESUME: load any records already banked (a prior run that died mid-way) and skip the (rep,arm,cell)
    # triples already done — a scattered-ring bench can lose a run to one blip; don't redo good jobs.
    records = json.load(open(a.out)) if os.path.exists(a.out) else []
    done = {(r["rep"], r["arm"], r["cell"]) for r in records}
    if records:
        print(f"[resume] loaded {len(records)} banked jobs; skipping those (rep,arm,cell)", flush=True)
    t00 = time.time()

    warm_combos = sorted({(ARMS[m][0], ARMS[m][1]) for m in arms})
    for tree, graph in warm_combos:                        # pay graph captures OFF the record (F8)
        print(f"[warmup] tree={tree} graph={graph}", flush=True)
        run_job((tree, graph, 0), "warmup", CELLS[0][1], None, 64, record=False)

    for rep in range(1, a.reps + 1):
        order = arms[(rep - 1) % len(arms):] + arms[:(rep - 1) % len(arms)]   # rotate: drift hits all arms
        for arm in order:
            for cell, prompt, tools, mn in CELLS:
                if (rep, arm, cell) in done:               # already banked by a prior run
                    continue
                rec = run_job(arm, cell, prompt, tools, mn)
                if rec is None:                            # skipped after retries — record nothing, move on
                    continue
                rec["rep"] = rep; records.append(rec)
                print(f"[{int(time.time() - t00):>5}s] rep{rep} {arm:<10} {cell:<14} "
                      f"{rec['tok_s']:>6.2f} tok/s  g={rec['g']:.2f}  receipts={rec['receipts_ok']}", flush=True)
                json.dump(records, open(a.out, "w"))

    _set_arm(False, False, 0)                              # never leak an experiment arm (F5)
    run_job((False, False, 0), "park", "ok", None, 8, record=False)

    print("\n=== per-cell median tok/s [min..max] ===", flush=True)
    cells_seen = [c[0] for c in CELLS]
    print("cell".ljust(16) + "".join(f"{m:>22}" for m in arms), flush=True)
    for cell in cells_seen:
        row = cell.ljust(16)
        for m in arms:
            v = [r["tok_s"] for r in records if r["cell"] == cell and r["arm"] == m]
            row += f"{statistics.median(v):>8.2f} [{min(v):>5.2f}..{max(v):>5.2f}]" if v else " " * 22
        print(row, flush=True)

    print("\n=== decode-weighted tok/s per arm (Σtokens/Σdecode_s) + delta vs base ===", flush=True)
    dw = {}
    for m in arms:
        rs = [r for r in records if r["arm"] == m]
        dw[m] = sum(r["new_tokens"] for r in rs) / max(sum(r["decode_s"] for r in rs), 1e-9)
    for m in arms:
        base = BASE_OF.get(m)
        delta = f"  ({100 * (dw[m] / dw[base] - 1):+.1f}% vs {base})" if base and dw.get(base) else ""
        print(f"  {m:<10} {dw[m]:>6.2f}{delta}", flush=True)
    print(f"[lever-bench] {len(records)} jobs -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
