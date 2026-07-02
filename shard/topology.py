"""latency-optimal pipeline ordering — the heart of serving a *scattered* swarm.

c0mpute nodes are random consumer GPUs on home links, never co-located. every token
traverses coordinator -> head -> ... -> tail -> (direct return) -> coordinator, so the
per-token WAN cost is:

    entry hop  +  sum of forward hops  +  return hop
    c_out[h]   +  sum L[node_i, node_i+1]  +  c_in[t]

the visit order is free (any node can hold any contiguous block), so the cheapest
pipeline is the minimum-latency Hamiltonian loop with the coordinator as the depot.
internet RTT is asymmetric and doesn't track geography (peering), so we optimize on the
*measured* mesh, not distance.

  - optimal_loop : exact (Held-Karp) for n<=16, nearest-neighbor + 2-opt above.
  - select_and_order : pick the best k of n online nodes AND order them (exact for n<=16).

both take L (L[i][j] = ms from node i to node j, asymmetric ok), c_out (coordinator->i),
c_in (i->coordinator). pure python, no deps; run `python -m shard.topology` for a demo.
"""
from itertools import combinations

INF = float("inf")
_TRIM = 12          # max candidates fed to exhaustive Held-Karp; the network layer funnels bigger pools first


def _up(up_mbps, n):
    """Upload Mbps for node n, floored to avoid div-by-zero. An unmeasured/zero node reads as very
    slow (0.5 Mbps) -> costed OFF the critical path, then relegated. Unmeasured == assume bad uplink
    is the safe default (a lying/absent uplink can't sneak onto a load-bearing hop)."""
    v = up_mbps.get(n, 0.0) if hasattr(up_mbps, "get") else up_mbps[n]
    return max(float(v), 0.5)


def _xfer_ms(nbytes, up_mbps, n):
    """ms to upload `nbytes` from node n over its uplink (bits / bandwidth). Residential is asymmetric
    (fast down, slow up), so a hop a->b is bound by the SENDER a's UPLOAD, never the receiver's."""
    return nbytes * 8.0 / (_up(up_mbps, n) * 1000.0)


def loop_cost(order, L, c_out, c_in):
    """total per-traversal latency for a given node ordering (entry + hops + return)."""
    if not order:
        return 0.0
    cost = c_out[order[0]] + c_in[order[-1]]
    for a, b in zip(order, order[1:]):
        cost += L[a][b]
    return cost


def _held_karp(nodes, L, c_out, c_in):
    """exact min-latency loop over exactly `nodes` (indices). O(k^2 2^k), k<=~16."""
    idx = list(nodes)
    k = len(idx)
    if k == 1:
        return idx, c_out[idx[0]] + c_in[idx[0]]
    pos = {n: i for i, n in enumerate(idx)}                 # node -> bit position
    dp = [[INF] * k for _ in range(1 << k)]                 # dp[mask][j] = min cost ... ending at j
    par = [[-1] * k for _ in range(1 << k)]
    for j in range(k):
        dp[1 << j][j] = c_out[idx[j]]
    for mask in range(1 << k):
        for j in range(k):
            if dp[mask][j] == INF or not (mask >> j) & 1:
                continue
            base = dp[mask][j]
            for m in range(k):
                if (mask >> m) & 1:
                    continue
                nmask = mask | (1 << m)
                cand = base + L[idx[j]][idx[m]]
                if cand < dp[nmask][m]:
                    dp[nmask][m] = cand
                    par[nmask][m] = j
    full = (1 << k) - 1
    best, bj = INF, -1
    for j in range(k):
        c = dp[full][j] + c_in[idx[j]]
        if c < best:
            best, bj = c, j
    order, mask, j = [], full, bj                           # reconstruct
    while j != -1:
        order.append(idx[j])
        pj = par[mask][j]
        mask ^= (1 << j)
        j = pj
    order.reverse()
    return order, best


def _nn_2opt(nodes, L, c_out, c_in, rounds=4):
    """heuristic for large k: nearest-neighbor seed, then 2-opt segment reversals."""
    idx = list(nodes)
    # nearest-neighbor from the cheapest entry hop
    start = min(idx, key=lambda n: c_out[n])
    tour, rest = [start], set(idx) - {start}
    while rest:
        last = tour[-1]
        nxt = min(rest, key=lambda n: L[last][n])
        tour.append(nxt); rest.discard(nxt)
    best = loop_cost(tour, L, c_out, c_in)
    improved = True
    while improved:
        improved = False
        for i in range(len(tour) - 1):
            for j in range(i + 1, len(tour)):
                cand = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                c = loop_cost(cand, L, c_out, c_in)
                if c + 1e-9 < best:
                    tour, best, improved = cand, c, True
    return tour, best


def optimal_loop(nodes, L, c_out, c_in):
    """min-latency pipeline order over all `nodes`. exact <=16, heuristic above."""
    nodes = list(nodes)
    if len(nodes) <= 16:
        return _held_karp(nodes, L, c_out, c_in)
    return _nn_2opt(nodes, L, c_out, c_in)


def select_and_order(nodes, L, c_out, c_in, k):
    """pick the best k of n online nodes AND order them into the cheapest loop.

    exact for n<=16 (Held-Karp answer ranges over size-k subsets); above that, solve the
    full order then greedily drop the node whose removal helps most until k remain.
    """
    nodes = list(nodes)
    if k >= len(nodes):
        return optimal_loop(nodes, L, c_out, c_in)
    if len(nodes) <= 16:
        best_order, best_cost = None, INF
        for subset in combinations(nodes, k):
            order, cost = _held_karp(subset, L, c_out, c_in)
            if cost < best_cost:
                best_order, best_cost = order, cost
        return best_order, best_cost
    order, _ = _nn_2opt(nodes, L, c_out, c_in)              # greedy-drop from a good full tour
    while len(order) > k:
        drop = min(range(len(order)),
                   key=lambda i: loop_cost(order[:i] + order[i + 1:], L, c_out, c_in))
        order = order[:drop] + order[drop + 1:]
    return _nn_2opt(order, L, c_out, c_in)


# ---- health + capability-aware selection (the self-optimizer's pure core) ----
# select_and_order picks the lowest-LATENCY ring. select_ring picks the lowest predicted
# STEP-TIME ring (WAN round-trip + per-stage compute), drops unhealthy/co-located nodes, and
# sizes each block to the node's speed. tok/s = accept_gain / step_ms and accept_gain is ~constant
# across rings, so minimizing predicted step_ms maximizes usable tok/s. The objective is physical
# (milliseconds), never hand-tuned weights: a power-capped or far node just shows up as more ms.
# PURE: measured stats in, a RingSpec out — no probing/IO here (that's the network layer's job).


def predict_step_ms(order, layers, L, c_out, c_in, layer_ms, up_mbps=None, decode_bytes=0.0):
    """Predicted per-traversal time of one decode step (ms): WAN round-trip (depends on ORDER) +
    sum of per-stage compute (depends on the layer ASSIGNMENT) + the per-step activation UPLOAD when
    `up_mbps` is given. `layers[n]` = #layers node n holds; `layer_ms[n]` = MEASURED ms to run one
    layer for a decode step on n (a throttled GPU measures higher — don't infer it from watts).
    Decode's activation is small (a few draft tokens' hidden state) but NOT free below ~fiber: at
    20 Mbps a ~50KB bundle costs ~20ms/hop, so on residential links decode transport is real and
    each stage uploads its per-step output once around the loop. On fiber `decode_bytes/up -> ~0` and
    this reduces to the pure round-trip+compute model (transport was rightly omitted there)."""
    ms = loop_cost(order, L, c_out, c_in) + sum(layers[n] * layer_ms[n] for n in order)
    if up_mbps is not None and decode_bytes:
        ms += sum(_xfer_ms(decode_bytes, up_mbps, n) for n in order)   # every stage uploads once/step
    return ms


def predict_prefill_ms(order, layers, L, c_out, c_in, up_mbps, prefill_bytes, prefill_chunks=1,
                       prefill_layer_ms=None):
    """Predicted TTFT (ms) — the RESIDENTIAL WALL. One forward traversal of the prompt + the [S,H]
    activation UPLOAD (upload-bound, dominant) + optional prefill compute. The prompt's [S,H]
    activation (`prefill_bytes` per hop, e.g. 16k*3072*2 ~= 100MB) is pipelined across the ring as
    `prefill_chunks` (C) chunks, so each FORWARDING (non-tail) stage uploads its whole [S,H] split
    into C pieces. The pipeline makespan interpolates the two physical regimes:
        transport = ( sum_fwd(u) + (C-1)*max_fwd(u) ) / C ,   u_s = _xfer_ms(prefill_bytes, up, s)
    C=1 (a single blob per hop) -> SUM (serial: each stage waits for the whole activation);
    C large (fine chunking) -> MAX (steady-state pipeline, bounded by the slowest uplink). The engine
    runs chunked+pipelined prefill (prefill_chunk, prefill_depth), so C = ceil(S/prefill_chunk).
    The TAIL forwards nothing onward (it returns only the first token's logits, tiny) -> EXEMPT, so a
    low-upload node belongs at the TAIL. Compute optional: residential prefill is transport-dominated
    (~100MB@20Mbps = ~40s/hop vs seconds of compute); pass `prefill_layer_ms` for fiber-accurate TTFT."""
    lat = loop_cost(order, L, c_out, c_in)                       # one traversal; return = first-token logits (small)
    fwd = order[:-1]                                             # non-tail stages upload [S,H] onward
    C = max(1, int(prefill_chunks))
    if fwd:
        us = [_xfer_ms(prefill_bytes, up_mbps, n) for n in fwd]
        transport = (sum(us) + (C - 1) * max(us)) / C
    else:
        transport = 0.0
    compute = sum(layers[n] * prefill_layer_ms[n] for n in order) if prefill_layer_ms else 0.0
    return lat + transport + compute


def _relegate(order, dropped, caps, subnet, up_mbps, layer_ms):
    """Advisory off-critical-path role for every DROPPED node, derived from WHY the objective dropped
    it — NOT a fresh absolute threshold. This is the PLACEMENT half of the decided admission/placement
    framing: the "threshold" is per-role capability against the CHOSEN ring, never a velvet rope at the
    door. c0mpute makes the final placement; these are hints. Coverage is TOTAL (every dropped node
    gets a role). The only capacity split is physical: cap==0 (can't be a stage) vs cap>=1.
      weight-seeder      : cap==0 — serves weight shards from disk (the torrent fetch path); no VRAM/
                           compute/latency needs. The universal floor.
      aggregator/relay   : upload >= the ring's BEST uplink AND subnet-distinct — a fiber-class node
                           wasted as a mere stage; spend its scarce UPLOAD as a prefill fan-in / relay
                           supernode (the research's top off-ring lever). Mechanism lives in c0mpute.
      hot-standby        : subnet-twin of a chosen stage — warm PASSIVE failover for that block (co-
                           location is fine for a spare; a twin is latency-close, so failover keeps the
                           ring's step_ms — we never route a high-latency node here).
      decode-only-replica: compute ring-competitive but dropped for its slow UPLOAD — decode's tiny
                           activation survives its uplink; candidate member of a decode-only ring (ring
                           formation, which needs >=k subnet-distinct peers, lives in c0mpute).
      spot-check-verifier: any other block-capable node — samples & recomputes a stage to catch
                           cheaters (latency/upload tolerant, async, bounded demand)."""
    if not order:
        return {}
    ring_subnets = {subnet[n] for n in order}
    ring_best_up = max(_up(up_mbps, n) for n in order)          # the ring's fastest uplink
    ring_worst_compute = max(layer_ms[n] for n in order)        # slowest per-layer compute the ring admitted
    roles = {}
    for n in dropped:
        if caps.get(n, 0) == 0:
            roles[n] = "weight-seeder"                          # can't hold a stage -> seed weights
        elif _up(up_mbps, n) >= ring_best_up and subnet[n] not in ring_subnets:
            roles[n] = "aggregator"                             # better-connected than the whole ring
        elif subnet[n] in ring_subnets:
            roles[n] = "hot-standby"                            # subnet-twin of a stage -> warm failover
        elif layer_ms[n] <= ring_worst_compute:
            roles[n] = "decode-only-replica"                    # compute-fine, dropped for upload -> decode is ok
        else:
            roles[n] = "spot-check-verifier"                    # slow compute/high latency -> sampled recompute
    return roles


def node_capacity(free_vram_mb, layer_vram_mb, kv_mb_per_layer=0):
    """max contiguous layers a node can hold (weights + KV cache), >= 0."""
    per = layer_vram_mb + kv_mb_per_layer
    return int(free_vram_mb // per) if per > 0 else 0


def assign_layers(order, n_layers, caps, layer_ms):
    """Size each node's contiguous block to MINIMIZE total decode-step compute — the SUM of per-stage
    times, which is exactly what predict_step_ms scores and the right model for single-traversal
    autoregressive decode (token t+1 can't enter the ring until t exits, so per-step latency is the
    sum, not a pipeline makespan). Every stage must hold >=1 layer (no empty hops), so: floor 1 per
    node, then pile the remaining layers onto the lowest-layer_ms nodes up to their VRAM `caps`.
    Returns {node: cnt} (sums to n_layers, every value >=1) or None if the subset can't give each
    stage a layer and still hold the model. (A PIPELINED throughput regime minimizes the max stage
    instead; predict_step_ms would then switch to max — the two must stay in lockstep.)"""
    k = len(order)
    if n_layers <= 0 or n_layers < k:                           # need >=1 layer per stage
        return None
    if sum(caps[n] for n in order) < n_layers:
        return None
    alloc = {n: 1 for n in order}                               # every stage holds >=1 layer (no empty hops)
    rem = n_layers - k
    for n in sorted(order, key=lambda n: layer_ms[n]):          # remaining layers -> cheapest-per-layer first (min sum)
        take = min(caps[n] - alloc[n], rem)
        if take > 0:
            alloc[n] += take; rem -= take
        if rem <= 0:
            break
    return alloc if rem == 0 else None


def select_ring(nodes, L, c_out, c_in, *, free_vram_mb, layer_ms, subnet,
                n_layers, layer_vram_mb, kv_mb_per_layer=0, slack=2, exclude=None, require=None,
                up_mbps=None, prefill_bytes=0.0, decode_bytes=0.0, decode_steps=1,
                prefill_chunks=1, prefill_layer_ms=None, relegate=True):
    """The self-optimizer's pure core. From a candidate POOL, choose the subset + ring order +
    per-node layer split that MINIMIZES predicted request time, subject to:
      * VRAM feasibility — the chosen nodes must hold the whole model (+ KV),
      * NEVER co-locate — no two stages share a `subnet` key (datacenter/network),
      * health — a power-capped/slow node has a high `layer_ms`, so it's dropped or given fewer
        layers automatically; no hand-tuned weights, just physical milliseconds.
    Prefers the FEWEST nodes that fit (each extra node is another full WAN round-trip — fewer,
    fatter stages win over scatter), trying sizes k_min..k_min+slack so a faster larger set can
    still win.

    UPLOAD-AWARE (opt-in via `up_mbps={node: Mbps}`): the objective becomes TOTAL REQUEST TIME
    T = prefill_ms + decode_steps * decode_step_ms, with per-node UPLOAD a first-class cost. This is
    the residential lever: a home link is asymmetric (fast down, SLOW up) and the per-hop bottleneck
    is MOVING THE ACTIVATION on the sender's uplink. Decode's activation is tiny (survives), but
    long-context PREFILL ([S,H] ~= 100MB/hop @16k) is the WALL — minutes of TTFT on a 20 Mbps cable
    uplink. So the selector (a) tails the lowest-upload node (the tail forwards nothing — see
    predict_prefill_ms), (b) drops nodes whose upload would dominate prefill, and (c) RELEGATES those
    dropped nodes to off-critical-path roles (see _relegate) instead of discarding useful capacity.
    Bytes are PRE-MULTIPLIED by the caller (workload- and dtype-agnostic core): `prefill_bytes`=S*H*
    dtype, `decode_bytes`=draft_tokens*H*dtype (fp8 wire => halve them), `prefill_chunks`=ceil(S/
    prefill_chunk) sets the SUM<->MAX pipeline regime, `prefill_layer_ms` (optional) adds prefill
    compute for fiber-accurate TTFT. Missing/absent `up_mbps` == today's pure decode-step objective
    (BYTE-IDENTICAL legacy path); when set, the spec also carries prefill_ms/request_ms/roles.

    Returns a RingSpec dict, or None if the pool can't hold the model:
      {order, blocks:{n:(lo,hi)}, layers:{n:cnt}, step_ms, tok_s_per_g, dropped, k}
      (+ prefill_ms, request_ms, roles:{dropped_node: role}  when up_mbps is given).
    `require` pins a node that MUST be in the ring (our coordinator runs on the head box, so for
    that deployment pass the head node and set c_out/c_in relative to it -> the loop becomes the
    correct head->...->tail->head cycle). `exclude` drops nodes outright. NOTE: `slack` is the pool
    headroom you rented (N+slack) — selection can only drop bad nodes when the pool exceeds what the
    model strictly needs. Assumes the pool is already pre-filtered to a tractable candidate set (the
    network layer funnels thousands -> ~16 via latency coordinates before calling this); if larger,
    it pre-trims to the 14 lowest-RTT usable nodes (always keeping `require`)."""
    if require is not None and exclude and require in set(exclude):
        raise ValueError("`require` and `exclude` name the same node")
    nodes = [n for n in nodes if not exclude or n not in exclude]
    caps = {n: node_capacity(free_vram_mb[n], layer_vram_mb, kv_mb_per_layer) for n in nodes}
    usable = [n for n in nodes if caps[n] > 0]
    if require is not None and require not in usable:
        return None                                              # the pinned coord/head can't hold a block

    def feasible_cap(pool):                                      # max layers coverable using DISTINCT subnets
        best = {}
        for n in pool:
            best[subnet[n]] = max(best.get(subnet[n], 0), caps[n])
        return sum(best.values())
    if feasible_cap(usable) < n_layers:                          # feasibility on the FULL set, honoring co-location
        return None                                              # genuinely can't serve the model (no false negative)

    by_cap = sorted(usable, key=lambda n: caps[n], reverse=True)
    acc, k_min, used_sub = 0, 0, set()                           # k_min = fewest DISTINCT-subnet nodes that fit
    for n in by_cap:
        if subnet[n] in used_sub:
            continue
        used_sub.add(subnet[n]); acc += caps[n]; k_min += 1
        if acc >= n_layers:
            break

    if len(usable) > _TRIM:                                      # latency funnel, but never trim out nodes feasibility
        keep = sorted(usable, key=lambda n: c_out[n] + c_in[n])[:_TRIM]   # or `require` need
        must, seen = set(), set()                                # a DISTINCT-subnet cover (+slack) must survive: a
        for m in by_cap:                                         # subnet-BLIND top-cap `must` starves the pool of
            if subnet[m] in seen:                                # feasible cover when the fattest cards are co-located
                continue                                         # (that was a false-"infeasible" bug) -> pick fattest
            seen.add(subnet[m]); must.add(m)                     # node per NEW subnet, mirroring the k_min walk
            if len(must) >= k_min + slack:
                break
        if require is not None:                                  # ...and a REQUIRE-compatible cover: `require`
            must.add(require)                                    # sits in every ring, so its subnet's slot is
            seen, acc = {subnet[require]}, caps[require]         # spent on IT (a same-subnet fat card can never
            for m in by_cap:                                     # join it) -> keep OTHER-subnet fat nodes until
                if subnet[m] in seen:                            # they cover the model, else a require-blind
                    continue                                     # `must` starves the pool the same way (that
                seen.add(subnet[m]); must.add(m); acc += caps[m] # was false-"infeasible" bug #3)
                if acc >= n_layers:
                    break
        usable = keep + [n for n in must if n not in keep]

    aware = up_mbps is not None
    D = max(0, int(decode_steps))
    if aware:
        # Effective ORDERING matrices: fold the frequency-weighted latency (each traversal happens
        # once for prefill + D times for decode) with a per-edge upload cost, so the SAME optimal_loop
        # returns a request-time-optimal order. The big [S,H] prefill term rides ONLY forward edges
        # (sender a uploads to its successor); the return hop (Ein) carries just the first token's
        # logits, so the tail's expensive forward upload vanishes -> optimal_loop naturally TAILS the
        # lowest-upload node. Compute is order-independent (summed per node), so it's added at scoring,
        # not here; hence minimizing this fold == minimizing request_ms over orders (exact for C=1,
        # a tight tail-exempting heuristic for C>1 where prefill is a MAX the loop-sum can't express).
        EL = {a: {} for a in usable}
        Eout, Ein = {}, {}
        for a in usable:
            pf_a = _xfer_ms(prefill_bytes, up_mbps, a)
            dc_a = _xfer_ms(decode_bytes, up_mbps, a)
            Eout[a] = (1 + D) * c_out[a]                         # entry: coord uploads tiny token-ids -> latency only
            Ein[a] = (1 + D) * c_in[a] + D * dc_a               # return: tail uploads decode output D times, no [S,H]
            for b in usable:
                if a != b:
                    EL[a][b] = (1 + D) * L[a][b] + pf_a + D * dc_a

    def _score(order, alloc):
        step = predict_step_ms(order, alloc, L, c_out, c_in, layer_ms,
                               up_mbps if aware else None, decode_bytes)
        if not aware:
            return step, step, 0.0                              # legacy: rank == decode step; no prefill
        pf = predict_prefill_ms(order, alloc, L, c_out, c_in, up_mbps, prefill_bytes,
                                prefill_chunks, prefill_layer_ms)
        return pf + D * step, step, pf                          # rank by total request time

    def _search(k_lo, k_hi):
        found = None
        for k in range(k_lo, min(k_hi, len(usable)) + 1):
            for subset in combinations(usable, k):
                if require is not None and require not in subset:        # coord/head must be in the ring
                    continue
                if len(set(subnet[n] for n in subset)) < k:              # never co-locate (all distinct subnets)
                    continue
                order, _ = optimal_loop(subset, EL, Eout, Ein) if aware else optimal_loop(subset, L, c_out, c_in)
                alloc = assign_layers(order, n_layers, {n: caps[n] for n in subset}, layer_ms)
                if alloc is None:
                    continue
                rank, step, pf = _score(order, alloc)
                if found is None or rank < found[0]:                     # ties: smaller k wins (k ascending, strict <)
                    found = (rank, order, alloc, k, step, pf)
        return found

    kmax = k_min + slack
    best = _search(k_min, kmax)
    if best is None:                                             # co-location can push the true minimum k above k_min+slack
        best = _search(kmax + 1, len(usable))                    # -> widen rather than falsely report infeasible
    if best is None:
        return None
    rank, order, alloc, k, step, pf = best
    blocks, lo = {}, 0
    for n in order:
        blocks[n] = (lo, lo + alloc[n]); lo += alloc[n]
    dropped = [n for n in nodes if n not in order]
    spec = {"order": order, "blocks": blocks, "layers": alloc, "step_ms": round(step, 1),
            "tok_s_per_g": round(1000.0 / step, 2) if step > 0 else INF, "dropped": dropped, "k": k}
    if aware:
        spec["prefill_ms"] = round(pf, 1)
        spec["request_ms"] = round(rank, 1)
        if relegate:
            spec["roles"] = _relegate(order, dropped, caps, subnet, up_mbps, layer_ms)
    return spec


# ---- demo: a scattered-US mesh, optimal loop vs naive ordering ----
if __name__ == "__main__":
    # ~ms one-way, lat/long-ish placement; internet RTT, not distance: a couple of
    # asymmetric peering quirks baked in so geography is NOT the answer.
    cities = ["WA", "OR", "CA", "TX", "KS", "IL", "GA", "NC", "VA", "NY"]
    xy = {"WA": (0, 9), "OR": (0, 7), "CA": (1, 3), "TX": (5, 1), "KS": (6, 5),
          "IL": (8, 6), "GA": (9, 2), "NC": (11, 3), "VA": (11, 5), "NY": (12, 8)}
    def base(a, b):
        (x1, y1), (x2, y2) = xy[a], xy[b]
        return 4.0 + 2.3 * ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5      # ms
    n = len(cities)
    L = [[0.0] * n for _ in range(n)]
    for i, a in enumerate(cities):
        for j, b in enumerate(cities):
            if i != j:
                L[i][j] = base(a, b) + (3.0 if (i + 2 * j) % 5 == 0 else 0.0)  # asym noise
    L[2][8] = L[8][2] = 9.0          # CA<->VA: a fat peering pipe, far but fast
    c_out = [base("WA", c) for c in cities]   # coordinator near WA (entry hop)
    c_in = [base("WA", c) * 0.9 for c in cities]  # direct return, slightly cheaper path

    nodes = list(range(n))
    geo = loop_cost(nodes, L, c_out, c_in)            # input order = a clean geographic guess
    join = [4, 9, 2, 7, 0, 5, 8, 3, 6, 1]            # arbitrary join order (the real case)
    join_cost = loop_cost(join, L, c_out, c_in)
    order, cost = optimal_loop(nodes, L, c_out, c_in)
    name = lambda o: " -> ".join(cities[i] for i in o)
    print(f"arbitrary join order {join_cost:6.1f} ms   {name(join)}")
    print(f"geographic guess     {geo:6.1f} ms   {name(nodes)}")
    print(f"OPTIMAL loop         {cost:6.1f} ms   {name(order)}")
    print(f"  -> {join_cost / cost:.2f}x vs how nodes actually join, {geo / cost:.2f}x vs a hand geo-guess")
    sub_order, sub_cost = select_and_order(nodes, L, c_out, c_in, k=6)
    print(f"best 6 of {n}          {sub_cost:6.1f} ms   {name(sub_order)}")
