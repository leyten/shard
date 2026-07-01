"""Honest offline simulator for the self-optimizer. Physics from sub-agent A (4-bucket traversal,
calibrated to reproduce tonight's REAL rings: junk 950ms/2.6tok-s and good 310ms/12tok-s). Pool
distributions from sub-agent B (cited: King/RIPE latency, FCC/Ookla bimodal upload, libp2p DCUtR
~70% direct-connect, Weibull churn). Lets us develop/stress-test select_ring against thousands of
realistic scattered pools for $0 instead of renting GPUs.

  python scratchpad/sim_network.py --pools 3000 --pool rental
  python scratchpad/sim_network.py --pools 3000 --pool volunteer

The crux A resolved: tonight's slowness is TRANSPORT the 227ms ping under-measured (libp2p relay/
framing), NOT compute or power caps. So the dominant lever is real-libp2p-latency + relay-avoidance,
which select_ring captures IF it's fed the effective per-hop cost (latency + libp2p overhead + xfer),
not raw ping. This sim quantifies how much that matters vs naive/latency-only selection.
"""
import sys, argparse, random, statistics, math
sys.path.insert(0, "/root/.openclaw/workspace/shard")
from itertools import combinations
from shard.topology import select_ring, optimal_loop, assign_layers, node_capacity

# ---- model constants (sub-agent A; finalize O_* with a $0.30 on-box libp2p echo) ----
N_LAYERS = 62
LAYER_MB, KV_MB = 1700.0, 150.0
LDECODE_MS = 2.5                       # per-layer decode ms @ stock power (62*2.5 = 155ms ring compute)
H, DTYPE = 3072, 2                     # MiniMax-M2.5 hidden size; bf16 wire (fp8 -> DTYPE=1, halves bytes)
DRAFT = 10                             # draft tokens/traversal (decode activation = DRAFT*H*dtype)
PREFILL_CHUNK = 4096                   # engine's chunked+pipelined prefill (coordinate_pipe prefill_chunk)
DECODE_TOKENS = 256                    # representative generation length D (request = prefill + D*decode_step)
# MEASURED 2026-07-01 (experiment_transport.py): the per-hop killer is MOVING THE ACTIVATION, NOT relay, and on
# an asymmetric home link the bind is the SENDER'S UPLOAD (fast down, slow up). A warm libp2p tunnel did a 120KB
# round-trip in 128ms @ 58ms RTT (2.6x faster than cold direct TCP's 336ms -- persistent connection, no per-hop
# slow-start). So decode hop = K_RTT*RTT (windowing) + bytes/UP[sender], NO relay penalty. K_RTT~2.8 reproduces
# both rings. DECODE activation is tiny (~60KB) and survives; long-context PREFILL ([S,H] ~100MB @16k) is the WALL.
K_RTT = 2.8                            # warm-tunnel windowing: a small activation costs ~2.8 round-trips
DECODE_BYTES = DRAFT * H * DTYPE       # decode activation/hop ~60KB (fp8 halves); tiny -> survives residential
G = 3.7                                # accepted tokens/traversal (held constant for ranking)

def prefill_bytes(s_tokens):           # the [S,H] activation forwarded per hop during prefill (~100MB @16k)
    return s_tokens * H * DTYPE

def num_chunks(s_tokens):              # pipeline chunk count -> sets the SUM<->MAX prefill regime
    return max(1, math.ceil(s_tokens / PREFILL_CHUNK))

def derate(power):                     # memory-bound decode barely cares about a power cap (A: ~1.1x @400W)
    return 1.0 + 0.30 * max(0.0, (575.0 - power) / 575.0)

def xfer_ms(nbytes, up_mbps):          # ms to push nbytes over the SENDER's uplink (bits / bandwidth)
    return nbytes * 8.0 / (max(up_mbps, 0.5) * 1000.0)

# ---- pool distributions (sub-agent B) ----
REGIONS = ["EU-c", "EU-n", "EU-s", "EU-w", "US-e", "US-w", "Asia"]
BASE = {"EU-c": 12, "EU-n": 18, "EU-s": 20, "EU-w": 15, "US-e": 14, "US-w": 16, "Asia": 25}
INTER = {("EU-c", "EU-n"): 25, ("EU-c", "EU-s"): 28, ("EU-c", "EU-w"): 18, ("EU-c", "US-e"): 95,
         ("EU-n", "EU-s"): 40, ("EU-w", "US-e"): 80, ("US-e", "US-w"): 65, ("EU-c", "Asia"): 180,
         ("US-w", "Asia"): 110, ("US-e", "Asia"): 150, ("EU-c", "EU-w"): 18}
def inter(ra, rb):
    return INTER.get((ra, rb)) or INTER.get((rb, ra)) or 130

POOLS = {  # vram pmf, power-cap frac, bw mixture (frac_low, med_low, med_hi), hard-NAT frac, region spread
    "rental":    dict(vram=[(32000, .45), (24000, .15), (48000, .2), (80000, .2)],
                      cap=0.5, bw=(0.25, 80, 600), nat=0.10, regions=["EU-c", "EU-n", "EU-s", "EU-w", "US-e"]),
    "volunteer": dict(vram=[(8000, .25), (12000, .15), (16000, .25), (24000, .2), (48000, .08), (80000, .07)],
                      cap=0.55, bw=(0.6, 20, 300), nat=0.15, regions=REGIONS),
}

def _pick(pmf, rng):
    r = rng.random(); a = 0.0
    for v, p in pmf:
        a += p
        if r <= a:
            return v
    return pmf[-1][0]

def gen_pool(n, cfg, rng):
    nodes = []
    for i in range(n):
        reg = rng.choice(cfg["regions"])
        flo, mlo, mhi = cfg["bw"]
        up = rng.lognormvariate(math.log(mlo if rng.random() < flo else mhi), 0.5)
        nodes.append(dict(i=i, region=reg, subnet=f"net{i}",
                          vram=_pick(cfg["vram"], rng),
                          power=rng.choice([400, 450, 500]) if rng.random() < cfg["cap"] else 575,
                          up_mbps=max(3.0, up),
                          hard_nat=rng.random() < cfg["nat"],
                          churn=_pick([("drive", .5), ("session", .35), ("stable", .15)], rng)))
    L = [[0.0] * n for _ in range(n)]
    relay = [[False] * n for _ in range(n)]
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            ra, rb = nodes[a]["region"], nodes[b]["region"]
            base = BASE[ra] if ra == rb else inter(ra, rb)
            L[a][b] = round(base * rng.uniform(0.9, 1.25) + rng.uniform(0, 6), 1)   # multi-modal-ish jitter
            # libp2p DCUtR: a pair relays if either is hard-NAT, else ~30% residual hole-punch failure
            relay[a][b] = nodes[a]["hard_nat"] or nodes[b]["hard_nat"] or (rng.random() < 0.30)
    return nodes, L, relay

# ---- fidelity: TRUE per-traversal ms (cycle model, coord on head) ----
# The a->b flow is bottlenecked by the SENDER a's UPLOAD (asymmetric residential: the receiver's DOWNLOAD is
# fast, so it never binds; using min(up[a],up[b]) would wrongly penalize a slow-up node as a RECEIVER). If you
# ever model throttled downlinks (satellite), add a `down_mbps` field and use min(up[a], down[b]) -- never up[b].
def hop_cost(a, b, L, nodes):
    return K_RTT * L[a][b] + xfer_ms(DECODE_BYTES, nodes[a]["up_mbps"])   # decode: windowing + tiny upload (sender)

def true_step_ms(order, layers, L, nodes):                      # per-token DECODE step (serial ring, SUM over hops)
    edges = list(zip(order, order[1:])) + [(order[-1], order[0])]   # head->...->tail->head
    transport = sum(hop_cost(a, b, L, nodes) for a, b in edges)
    compute = sum(layers[n] * LDECODE_MS * derate(nodes[n]["power"]) for n in order)
    return transport + compute

def true_prefill_ms(order, s_tokens, L, nodes):
    """TTFT ground truth -- the residential WALL. Each FORWARDING (non-tail) stage uploads the [S,H] activation;
    the engine pipelines it as C chunks, so the makespan interpolates SUM (C=1) and MAX (C large):
       transport = (sum_fwd(u) + (C-1)*max_fwd(u)) / C ,  u_s = xfer_ms(prefill_bytes(S), up[s]).
    The TAIL forwards nothing (returns only first-token logits) -> exempt. Latency = one forward traversal (plain
    RTT, not K_RTT: windowing is negligible against a ~100MB flow). Prefill COMPUTE omitted -- at cable uplinks
    it's transport-dominated by ~50-100x; it is the SAME split for aware/blind so it can't change their ranking."""
    fwd = order[:-1]                                            # non-tail forwarders
    lat = sum(L[a][b] for a, b in zip(order, order[1:])) + L[order[-1]][order[0]]   # forward + tiny logits return
    if not fwd:
        return lat
    us = [xfer_ms(prefill_bytes(s_tokens), nodes[n]["up_mbps"]) for n in fwd]
    C = num_chunks(s_tokens)
    return lat + (sum(us) + (C - 1) * max(us)) / C

def true_request_ms(order, layers, s_tokens, L, nodes, d=DECODE_TOKENS):   # the objective select_ring minimizes
    return true_prefill_ms(order, s_tokens, L, nodes) + d * true_step_ms(order, layers, L, nodes)

def true_tok_s(order, layers, L, nodes):                        # decode throughput (secondary metric)
    return 1000.0 * G / true_step_ms(order, layers, L, nodes)

def survival(order, nodes):                                      # P(no node drops mid-request), churn proxy
    p = {"drive": 0.90, "session": 0.985, "stable": 0.999}       # per-node survive-a-request prob
    s = 1.0
    for n in order:
        s *= p[nodes[n]["churn"]]
    return s

# ---- what the selector chooses: UPLOAD-AWARE vs UPLOAD-BLIND (today's shipping behaviour) ----
def _head(nodes, L, cap_ok):
    return min(cap_ok, key=lambda i: sum(L[i][j] for j in range(len(nodes)) if j != i))

def plan(nodes, L, s_tokens, aware):
    """aware=True feeds select_ring per-node UPLOAD + the prefill/decode byte sizes so it minimizes total
    request time; aware=False is the legacy decode-step-only selection (upload-blind, what ships today).
    Both are fed the SAME raw RTT mesh and pinned to the SAME head, so the difference is purely upload-awareness."""
    n = len(nodes)
    free = {i: nodes[i]["vram"] for i in range(n)}
    lms = {i: LDECODE_MS * derate(nodes[i]["power"]) for i in range(n)}
    subnet = {i: nodes[i]["subnet"] for i in range(n)}
    cap_ok = [i for i in range(n) if free[i] >= LAYER_MB + KV_MB]
    if not cap_ok:
        return None
    head = _head(nodes, L, cap_ok)
    lms = dict(lms); lms[head] *= 1.3
    c_out = [L[head][i] if i != head else 1.0 for i in range(n)]
    c_in = [L[i][head] if i != head else 1.0 for i in range(n)]
    up = {i: nodes[i]["up_mbps"] for i in range(n)}
    kw = dict(free_vram_mb=free, layer_ms=lms, subnet=subnet, n_layers=N_LAYERS, layer_vram_mb=LAYER_MB,
              kv_mb_per_layer=KV_MB, slack=4, require=head)
    if aware:
        kw.update(up_mbps=up, prefill_bytes=prefill_bytes(s_tokens), decode_bytes=DECODE_BYTES,
                  decode_steps=DECODE_TOKENS, prefill_chunks=num_chunks(s_tokens))
    return select_ring(range(n), L, c_out, c_in, **kw)

def oracle(nodes, L, s_tokens):
    """Best achievable request_ms given the deployment constraint (coord pinned on the same central head).
    Searches all subnet-distinct subsets (k=2..8), orders each by the upload-aware fold, scores TRUE
    request_ms -- the aspirational bound the selector is measured against."""
    n = len(nodes)
    caps = {i: node_capacity(nodes[i]["vram"], LAYER_MB, KV_MB) for i in range(n)}
    usable = [i for i in range(n) if caps[i] > 0]
    if not usable:
        return None
    head = _head(nodes, L, usable)
    up = {i: nodes[i]["up_mbps"] for i in range(n)}
    D = DECODE_TOKENS
    # upload-aware effective ordering matrices (same fold select_ring uses), over the usable set
    EL = {a: {b: (1 + D) * L[a][b] + xfer_ms(prefill_bytes(s_tokens), up[a]) + D * xfer_ms(DECODE_BYTES, up[a])
              for b in usable if b != a} for a in usable}
    Eout = {a: (1 + D) * (L[head][a] if a != head else 1.0) for a in usable}
    Ein = {a: (1 + D) * (L[a][head] if a != head else 1.0) + D * xfer_ms(DECODE_BYTES, up[a]) for a in usable}
    lms = {i: LDECODE_MS * derate(nodes[i]["power"]) for i in range(n)}
    best = None
    for k in range(2, min(len(usable), 8) + 1):
        for sub in combinations(usable, k):
            if head not in sub or len(set(nodes[i]["subnet"] for i in sub)) < k:
                continue
            order, _ = optimal_loop(sub, EL, Eout, Ein)
            alloc = assign_layers(order, N_LAYERS, {i: caps[i] for i in sub}, lms)
            if alloc is None:
                continue
            t = true_request_ms(order, alloc, s_tokens, L, nodes)
            if best is None or t < best[0]:
                best = (t, order, alloc)
    return best


def calibrate():
    """sanity: DECODE model still reconstructs tonight's two rings (~950 / ~310 ms) + a PREFILL sanity."""
    nodes = [dict(power=575 if i else 400, up_mbps=300) for i in range(5)]
    L = [[0 if i == j else 48 for j in range(5)] for i in range(5)]
    order = list(range(5)); layers = {0: 14, 1: 12, 2: 12, 3: 12, 4: 12}
    junk = true_step_ms(order, layers, L, nodes)
    nodes2 = [dict(power=575, up_mbps=600) for _ in range(5)]
    L2 = [[0 if i == j else 14 for j in range(5)] for i in range(5)]
    good = true_step_ms(order, layers, L2, nodes2)
    print(f"[calibrate] decode junk ring -> {junk:.0f}ms ({1000*G/junk:.1f} tok/s)   target ~950 / ~3.9")
    print(f"[calibrate] decode good ring -> {good:.0f}ms ({1000*G/good:.1f} tok/s)   target ~310 / ~12")
    # prefill sanity: 5-stage ring, 16k prompt, 20 Mbps residential cable uplinks -> minutes of TTFT (the WALL)
    res = [dict(power=575, up_mbps=20) for _ in range(5)]
    pf = true_prefill_ms(order, 16384, L, res)
    fib = true_prefill_ms(order, 16384, L2, [dict(power=575, up_mbps=600) for _ in range(5)])
    print(f"[calibrate] prefill 16k @20Mbps cable -> {pf/1000:.0f}s TTFT (the residential WALL); "
          f"@600Mbps fiber -> {fib/1000:.1f}s (vanishes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", type=int, default=800)
    ap.add_argument("--pool", choices=list(POOLS), default="volunteer")
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--ctx", type=int, nargs="+", default=[2048, 16384, 65536],
                    help="prompt lengths S to sweep (decode-dominated -> prefill-dominated)")
    a = ap.parse_args()
    calibrate()
    cfg = POOLS[a.pool]
    p = lambda xs, q: sorted(xs)[max(0, min(len(xs) - 1, int(len(xs) * q)))]
    print(f"\n=== {a.pool} pool, {a.pools} sims/ctx, size {a.size} — UPLOAD-AWARE vs UPLOAD-BLIND on TRUE request time ===")
    print(f"    ctx(S) | aware/oracle | blind/oracle | request_speedup (mean, p95) | TTFT_speedup (mean, p95) | k_aw")
    for S in a.ctx:
        rng = random.Random(7)                                  # same pools across ctx for a clean comparison
        aware_gap, blind_gap, speedup, ttft_su, k_aw, infeasible = [], [], [], [], [], 0
        for _ in range(a.pools):
            nodes, L, _ = gen_pool(a.size, cfg, rng)
            r_aw = plan(nodes, L, S, aware=True)
            r_bl = plan(nodes, L, S, aware=False)
            orc = oracle(nodes, L, S)
            if r_aw is None or r_bl is None or orc is None:
                infeasible += 1
                continue
            t_aw = true_request_ms(r_aw["order"], r_aw["layers"], S, L, nodes)
            t_bl = true_request_ms(r_bl["order"], r_bl["layers"], S, L, nodes)
            pf_aw = true_prefill_ms(r_aw["order"], S, L, nodes)
            pf_bl = true_prefill_ms(r_bl["order"], S, L, nodes)
            aware_gap.append(orc[0] / t_aw)                     # <=1; 1.0 = optimal (lower request_ms is better)
            blind_gap.append(orc[0] / t_bl)
            speedup.append(t_bl / max(t_aw, 1e-9))              # >1 => aware serves the whole request faster
            ttft_su.append(pf_bl / max(pf_aw, 1e-9))            # >1 => aware reaches first token faster (the WALL)
            k_aw.append(r_aw["k"])
        if not speedup:
            print(f"  {S:6d} |  (all infeasible: {infeasible}/{a.pools})"); continue
        print(f"  {S:6d} |    {statistics.mean(aware_gap):.3f}     |    {statistics.mean(blind_gap):.3f}     "
              f"|  {statistics.mean(speedup):.2f}x (p95 {p(speedup,0.95):.2f}x) "
              f"|  {statistics.mean(ttft_su):.2f}x (p95 {p(ttft_su,0.95):.2f}x) "
              f"|  {statistics.mean(k_aw):.1f}  ({infeasible} infeas)")
    # role relegation coverage: every dropped node gets an off-critical-path role (no capacity discarded)
    rng = random.Random(11); role_hist = {}; covered = dropped_tot = 0
    for _ in range(300):
        nodes, L, _ = gen_pool(a.size, cfg, rng)
        r = plan(nodes, L, 16384, aware=True)
        if r is None:
            continue
        roles = r.get("roles", {})
        for d in r["dropped"]:
            dropped_tot += 1
            if d in roles:
                covered += 1; role_hist[roles[d]] = role_hist.get(roles[d], 0) + 1
    if dropped_tot:
        print(f"\n  role relegation: {covered}/{dropped_tot} dropped nodes assigned a role "
              f"({100*covered/dropped_tot:.0f}% coverage)  {role_hist}")


if __name__ == "__main__":
    main()
