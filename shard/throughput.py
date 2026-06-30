"""throughput-aware ring selection — pick the ring that maximizes predicted tok/s.

topology.py answers "given THESE nodes, what order is cheapest?" — pure WAN latency. But
serving a swarm is not a latency problem, it's a throughput problem. One speculative round
(coordinator drafts K tokens, the ring verifies all K in a single forward traversal, the
tail returns the accept count) costs:

    round_ms = c_out[head] + sum L[i->i+1] + c_in[tail]      # WAN loop (topology.py)
             + sum_over_stages( layers_i * ms_per_layer_i )  # COMPUTE (topology can't see)
             + draft_ms                                       # coordinator's local K-draft

    tok/s = 1000 * tokens_per_round / round_ms,   tokens_per_round ~= accept_rate * K (<= K+1)

The crucial fact that keeps this cheap: for a FIXED set of nodes the compute sum is
order-independent (total layers are fixed; each node's share * its ms/layer doesn't depend
on where it sits in the loop). So ordering a fixed selection is STILL the min-latency loop
— we reuse topology.optimal_loop unchanged. Compute only changes:

  (1) SELECTION when the pool is oversubscribed: a fat-VRAM card that is compute-slow should
      lose to a leaner-but-faster one. allocate() alone (VRAM-greedy) is blind to this.
  (2) the tok/s estimate itself, so the orchestrator can rank rings and log a real number.

So best_ring searches feasible node SUBSETS, fits each (scheduler.allocate), orders it
(optimal_loop), scores predicted tok/s, and returns the winner. Inputs ms_per_layer (per
node) and the rtt mesh are LEARNED from completed runs (see perf_store.py); cold nodes fall
back to GPU-class priors. Pure python, no deps.
"""
from itertools import combinations

from .topology import optimal_loop, loop_cost


def round_ms(order, L, c_out, c_in, layers_by_node, ms_per_layer, node_ids,
             draft_ms=0.0):
    """per-round wall time: WAN loop + total compute + coordinator draft.

    order      : indices into node_ids giving the stage order (head-first).
    L,c_out,c_in : latency mesh (ms), same convention as topology.py.
    layers_by_node : node_id -> n_layers it holds this fit.
    ms_per_layer   : node_id -> ms to push one round's tokens through one layer.
    draft_ms   : coordinator's local draft cost for the round (K * per-draft-step).
    """
    wan = loop_cost(order, L, c_out, c_in)
    compute = sum(layers_by_node.get(node_ids[i], 0) * ms_per_layer.get(node_ids[i], 0.0)
                  for i in order)
    return wan + compute + draft_ms


def est_tok_s(round_time_ms, accept_rate, K):
    """tokens/sec from a round time and the speculative accept rate.

    one round emits ~accept_rate*K accepted tokens, capped at K+1 (greedy spec-decode emits
    at most K verified + 1 bonus). accept_rate in [0,1]; K>=1. Guards a zero round_time.
    """
    if round_time_ms <= 0:
        return 0.0
    toks = min(max(accept_rate, 0.0) * K, K + 1)
    if toks <= 0:
        toks = 1.0                       # even all-reject still commits the 1 bonus token
    return 1000.0 * toks / round_time_ms


def _mesh_for(subset_ids, all_ids, L, c_out, c_in):
    """slice the global mesh down to subset_ids, returning (idx, L', c_out', c_in')
    indexed 0..len(subset)-1 so topology's solvers operate on just the subset."""
    pos = {nid: i for i, nid in enumerate(all_ids)}
    sub = [pos[n] for n in subset_ids]
    Lp = [[0.0 if a == b else L[sub[a]][sub[b]] for b in range(len(sub))]
          for a in range(len(sub))]
    co = [c_out[s] for s in sub]
    ci = [c_in[s] for s in sub]
    return list(range(len(sub))), Lp, co, ci


def best_ring(node_ids, vram_gb, L, c_out, c_in, *, allocate_fn, ms_per_layer,
              draft_ms_by_node, accept_rate, K, total_layers, max_stages=None,
              min_stages=1):
    """pick the SUBSET + ORDER of nodes that maximizes predicted tok/s.

    node_ids       : all candidate node_ids (parallel to vram_gb, L, c_out, c_in indices).
    vram_gb        : node_id -> usable vram (passed to allocate_fn for the fit).
    allocate_fn    : (subset_ids) -> {node_id: n_layers} or None if the subset can't hold
                     the model. Wraps scheduler.allocate so this file stays fit-agnostic.
    ms_per_layer   : node_id -> learned ms/layer (GPU-class prior when cold).
    draft_ms_by_node : node_id -> coordinator draft cost if THAT node is head.
    accept_rate, K : speculative model params (learned per model; sane default otherwise).
    total_layers   : sanity — a fit must cover exactly this many layers.
    max_stages     : cap ring size (more stages = more WAN hops; rarely worth >6-8).
    returns: dict(ring_order=[node_id...], layers={node_id:n}, tok_s=float,
                  round_ms=float, coordinator=node_id) or None if nothing fits.
    """
    n = len(node_ids)
    if n == 0:
        return None
    hi = min(max_stages or n, n)
    best = None
    # try every feasible subset size, smallest first (fewer hops is usually faster); within a
    # size, every subset. Pool sizes here are small (a ring is a handful of GPUs), so this is
    # cheap and exact. allocate_fn prunes infeasible (insufficient-VRAM) subsets immediately.
    for ksz in range(max(min_stages, 1), hi + 1):
        for subset in combinations(range(n), ksz):
            subset_ids = [node_ids[i] for i in subset]
            layers = allocate_fn(subset_ids)
            if not layers:
                continue                         # subset can't hold the model
            if sum(layers.values()) != total_layers:
                continue                         # defensive: fit must be exact
            idx, Lp, co, ci = _mesh_for(subset_ids, node_ids, L, c_out, c_in)
            order, _ = optimal_loop(idx, Lp, co, ci)         # cheapest loop for THIS subset
            ordered_ids = [subset_ids[i] for i in order]
            head = ordered_ids[0]
            rt = round_ms(order, Lp, co, ci, layers, ms_per_layer, subset_ids,
                          draft_ms=draft_ms_by_node.get(head, 0.0))
            ts = est_tok_s(rt, accept_rate, K)
            if best is None or ts > best["tok_s"] + 1e-9:
                best = {"ring_order": ordered_ids, "layers": layers, "tok_s": ts,
                        "round_ms": rt, "coordinator": head}
    return best
