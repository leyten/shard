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


def predict_step_ms(order, layers, L, c_out, c_in, layer_ms):
    """Predicted per-traversal time of one decode step (ms): WAN round-trip (depends on ORDER) +
    sum of per-stage compute (depends on the layer ASSIGNMENT). The activation-transfer term is
    ~0 for single-token decode (one hidden state per hop) so it's omitted — it dominates PREFILL/
    TTFT, not tok/s. `layers[n]` = #layers node n holds; `layer_ms[n]` = MEASURED ms to run one
    layer for a decode step on n (a throttled GPU measures higher — don't infer it from watts)."""
    return loop_cost(order, L, c_out, c_in) + sum(layers[n] * layer_ms[n] for n in order)


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
                n_layers, layer_vram_mb, kv_mb_per_layer=0, slack=2, exclude=None, require=None):
    """The self-optimizer's pure core. From a candidate POOL, choose the subset + ring order +
    per-node layer split that MINIMIZES predicted decode step-time (=> maximizes tok/s), subject to:
      * VRAM feasibility — the chosen nodes must hold the whole model (+ KV),
      * NEVER co-locate — no two stages share a `subnet` key (datacenter/network),
      * health — a power-capped/slow node has a high `layer_ms`, so it's dropped or given fewer
        layers automatically; no hand-tuned weights, just physical milliseconds.
    Prefers the FEWEST nodes that fit (each extra node is another full WAN round-trip — fewer,
    fatter stages win over scatter), trying sizes k_min..k_min+slack so a faster larger set can
    still win. Returns a RingSpec dict, or None if the pool can't hold the model:
      {order, blocks:{n:(lo,hi)}, layers:{n:cnt}, step_ms, tok_s_per_g, dropped, k}.
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
        must = set(by_cap[:k_min + slack]) | ({require} if require is not None else set())
        usable = keep + [n for n in must if n not in keep]

    def _search(k_lo, k_hi):
        found = None
        for k in range(k_lo, min(k_hi, len(usable)) + 1):
            for subset in combinations(usable, k):
                if require is not None and require not in subset:        # coord/head must be in the ring
                    continue
                if len(set(subnet[n] for n in subset)) < k:              # never co-locate (all distinct subnets)
                    continue
                order, _ = optimal_loop(subset, L, c_out, c_in)
                alloc = assign_layers(order, n_layers, {n: caps[n] for n in subset}, layer_ms)
                if alloc is None:
                    continue
                step = predict_step_ms(order, alloc, L, c_out, c_in, layer_ms)
                if found is None or step < found[0]:                     # ties: smaller k wins (k ascending, strict <)
                    found = (step, order, alloc, k)
        return found

    kmax = k_min + slack
    best = _search(k_min, kmax)
    if best is None:                                             # co-location can push the true minimum k above k_min+slack
        best = _search(kmax + 1, len(usable))                    # -> widen rather than falsely report infeasible
    if best is None:
        return None
    step, order, alloc, k = best
    blocks, lo = {}, 0
    for n in order:
        blocks[n] = (lo, lo + alloc[n]); lo += alloc[n]
    return {"order": order, "blocks": blocks, "layers": alloc, "step_ms": round(step, 1),
            "tok_s_per_g": round(1000.0 / step, 2), "dropped": [n for n in nodes if n not in order], "k": k}


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
