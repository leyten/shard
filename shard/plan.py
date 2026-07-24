"""Deployable-ring planner — the seam c0mpute's control plane calls to place a sharded swarm.

`select_ring` (topology.py) is the pure decision core: clean numbers in, a ring out. But turning what
a node ANNOUNCES (GPU, free VRAM, a CPU probe, its subnet) plus a measured RTT mesh into those clean
numbers takes engine calibration — VRAM reserves, the launch-bound per-layer time, and head placement.
That calibration lived inline in `scratchpad/ring_up.py` (throwaway SSH glue). `plan_ring` lifts it into
a tracked, tested function so the network layer can PLACE a swarm with one call instead of re-deriving
the engine's memory/compute model.

Boundary (docs/INTEGRATION.md): c0mpute owns MEASUREMENT (collecting `nodes` + `rtt`); shard owns the
ENGINE model (reserves, per-layer ms, the select_ring decision). Deps point one way: c0mpute -> shard.
`python3 -m shard.plan` reads `{nodes, rtt, model}` JSON on stdin and prints the plan on stdout, so a
TypeScript orchestrator drives the same proven planner as `ring_up` without porting its subtleties.
"""
import json
import sys

from .topology import select_ring

# M2.5-on-5090 anchors (docs/M25_ENGINE.md, mirrored from ring_up.py) — the DEFAULT model profile.
# The caller (c0mpute's model catalog) SHOULD pass an explicit `model` profile; these keep the seam
# runnable for M2.5 without one.
M25_PROFILE = {
    "n_layers": 62,
    "layer_vram_mb": 2330.0,     # NVFP4 experts + bf16 attn + norms, per decoder layer — MEASURED
                                 # 2026-07-09 (capability probe, one real layer resident: 2329.5 MB;
                                 # a warm 13-layer stage read 31.5/32.6 GiB — the old 1700 estimate
                                 # under-modeled by ~35% and packed stages one allocation from OOM)
    "kv_mb_per_layer": 150.0,    # at the 40960 KV cap (B=1; batched callers scale by B*maxlen/40960)
    "layer_ms_base": 0.65,       # per-layer decode compute on an idle fast-CPU 5090 box
    "reserve_mb": 1500.0,        # CUDA context + allocator slack per box
    "head_reserve_mb": 4096.0,   # coordinator process on the head: embed + EAGLE head + its context
    "tail_reserve_mb": 1400.0,   # tail stage: final norm + lm_head (measured 1.15 GiB bf16 + slack —
                                 # a 13-layer tail OOM'd on it live while 13-layer middles warmed fine)
    "cap_layers": 12,            # 32 GB ceiling by MEASURED footprint ((32768-1500)/2480 = 12.6);
                                 # 13 ran warm but at 31.5/32.6 GiB — brim-riding, not a plan target
    "head_layer_ms_mult": 1.3,   # the head box also runs the coordinator
    # per-hop activation payloads for upload-aware placement (bf16 wire, hidden 3072 — see
    # phase0/m25_stage.py). Without these the residential objective priced every byte at ZERO and
    # upload-aware placement decayed to pure latency. Nominal request unit: a one-chunk (4096-tok)
    # prefill + 256 decode traversals of a (K+1)-token draft bundle (K=8, the measured sweet spot);
    # callers with a real workload override (fp8 wire => halve bytes, long prompts => S*H*2 and
    # prefill_chunks = ceil(S/4096)).
    "prefill_bytes": 4096 * 3072 * 2.0,   # chunk*H*2 — one [chunk,H] prefill hop (~25 MB)
    "decode_bytes": 9 * 3072 * 2.0,       # (K+1)*H*2 — one draft round's hidden bundle (~54 KB)
    "decode_steps": 256,                  # nominal decode traversals per request (~600 tok @ g~2.5)
    "prefill_chunks": 1,
}

# ── Kimi-K3 ───────────────────────────────────────────────────────────────────
# Every number below is MEASURED on real K3 weights (the G0 kernel probe, 2026-07-28, sm_120 —
# scratchpad/k3-research-20260727/g0-results.md) or read off the checkpoint's own config/shard
# headers. K3 is 93 decoder layers of a 2.78T-param MoE shipped natively in MXFP4; a layer is an
# ATOMIC 15.8 GiB stage unit, which is what makes its placement a different problem from M2.5's.
_K3_N_LAYERS = 93
_K3_HIDDEN = 7168                # text_config.hidden_size
_K3_MLA_LAYERS = 24              # gated-MLA layers (0-indexed 3,7,11,…,91,92 — a 3:1 interleave)
_K3_KDA_LAYERS = 69              # Kimi delta-attention layers: FIXED recurrent state, no paged KV
_K3_MLA_KV_BYTES_PER_TOKEN = 576 * 2.0    # kv_lora_rank 512 + qk_rope 64 elements, bf16, per layer
_K3_KDA_STATE_MB = 6.0                    # MEASURED recurrent state per KDA layer per REQUEST (fp32)
# AttnRes: a K3 layer boundary carries `prefix_sum` + a `block_residual` stack of `num_blocks`
# snapshots, i.e. (1+num_blocks) x hidden x 2B per token, not one hidden state. Blocks are appended
# at layers {0,12,24,36,48,60,72,84}, so the payload steps 2 units (28 KiB/token after L11) up to
# 9 units (126 KiB/token at L92) against a plain transformer's 1 unit (14 KiB). Summed over the 92
# layer boundaries that is 492 units => a MEAN of 492/92 = 5.348 units = 74.87 KiB/token.
_K3_BOUNDARY_UNITS_MEAN = 492 / 92
_K3_BOUNDARY_BYTES_PER_TOKEN = _K3_HIDDEN * 2.0 * _K3_BOUNDARY_UNITS_MEAN     # ~76.7 kB/token/hop
_K3_PREFILL_CHUNK = 1024         # tokens per pipelined prefill chunk — see the prefill_bytes note


def k3_kv_mb_per_layer(batch: int = 8, maxlen: int = 32768) -> float:
    """K3's per-request state, folded into the planner's ONE per-layer KV number.

    The planner (and probe.derive_layers) models a block as `layers * (layer_vram_mb +
    kv_mb_per_layer)`: one scalar, charged to every layer alike. K3's state is not one thing —
    24 MLA layers hold a paged KV cache that grows with CONTEXT (576 elements/token/layer bf16 =
    27.6 kB/token across the 24), and 69 KDA layers hold a FIXED recurrent state that does not
    grow with context at all (6.00 MiB/layer/request measured, 0.404 GiB/request across the 69).

    The honest fold is the per-layer AVERAGE at a stated (batch, maxlen), which is what this
    returns — not a per-layer vector, because there is nowhere to put one. The approximation is
    good because the MLA layers are evenly interleaved 3:1, so any contiguous block of >=4 layers
    carries close to its share: the worst realistic case (a 5-layer stage that lands 2 MLA layers
    instead of 1.3) is under-modeled by ~170 MB, against the ~5 GB of slack a 5-layer stage has
    left on a 96 GB card. State that does NOT scale this way (a 1M-token context is 27 GiB of MLA
    KV per request on its own) must be planned by calling this with the real numbers.
    """
    mla_mb = _K3_MLA_LAYERS * _K3_MLA_KV_BYTES_PER_TOKEN * maxlen * batch / (1024 * 1024)
    kda_mb = _K3_KDA_LAYERS * _K3_KDA_STATE_MB * batch          # context-independent, per request
    return (mla_mb + kda_mb) / _K3_N_LAYERS


K3_PROFILE = {
    "n_layers": _K3_N_LAYERS,
    "layer_vram_mb": 16203.0,    # 15.823 GiB — MEASURED 2026-07-28: one layer resident on a 5090
                                 # (15.823) and a 5-layer pack on a 96 GB Pro 6000 (79.113/5 =
                                 # 15.8226), agreeing to 0.4 MB. Not an independent sample: the pack
                                 # reused layer 46's weights 5x, so it is the same tensors on a
                                 # second card. Includes the Marlin-repacked
                                 # MXFP4 experts (14.643 GiB, NOT dequantized) + bf16 attention,
                                 # shared experts, latent projections and router.
                                 #
                                 # UNIFORM, and deliberately the MAXIMUM. K3's layers differ (68
                                 # KDA-MoE at 15.823 GiB, 24 MLA-MoE at 15.429, layer 0 dense at
                                 # 2.180), but nothing downstream can express that: plan_ring and
                                 # select_ring both size a block as layers*(layer_vram+kv) from one
                                 # scalar. Given one number, it must be the heaviest layer —
                                 # under-modeling VRAM is the failure mode that OOMs a stage at load
                                 # (M2.5's 1700-vs-2330 estimate packed stages one allocation from
                                 # OOM). The over-model is small and paid in the safe direction:
                                 # 93*15.823 = 1471.5 GiB planned against 1448.4 GiB real (+1.6%),
                                 # concentrated in the one stage that holds layer 0 (13.6 GiB heavy
                                 # — it may plan 5 layers where 6 would physically fit).
    "kv_mb_per_layer": k3_kv_mb_per_layer(batch=8, maxlen=32768),   # 109.9 MB — see the function.
                                 # B=8 x 32k is the phase-1 target; callers with a real workload
                                 # recompute (the MLA term scales with B*maxlen, the KDA term with
                                 # B alone — they do NOT share a scale factor, so no single
                                 # multiplier corrects this number).
    "layer_ms_base": 1.15,       # MEASURED B=1 with a whole-layer CUDA graph: 1.1186 ms on a 5090,
                                 # 1.1738 on a Pro 6000. EAGER is 1.20-1.51 — a node whose graph
                                 # capture failed must announce layer_ms=1.4, not be planned at
                                 # this. (93 layers => C = 107 ms graphed, 130 eager.)
                                 # CAVEAT, and G0 calls it the single biggest source of error in C:
                                 # the layer measured is a KDA layer, and 24 of the 93 are gated-MLA
                                 # layers that read a paged KV cache and will cost MORE at long
                                 # context. Treating all 93 as KDA-like is an approximation.
    "reserve_mb": 1500.0,        # CUDA context + allocator slack per box, as on M2.5, and it has to
                                 # cover a MEASURED gap: at the 5-layer pack the driver reported
                                 # 89.661 GiB in use against an allocator peak of 88.332 — 1.329 GiB
                                 # of context and fragmentation the allocator's own numbers never
                                 # show (the context alone is 0.546 GiB at zero allocation). The K3
                                 # load transient is NOT folded in here — see load_peak_extra_mb.
    "load_peak_extra_mb": 9440.0,  # 9.218 GiB: loading an MXFP4 layer peaks at resident + the
                                 # largest single tensor being Marlin-repacked (w13, 9.19 GiB).
                                 # MEASURED 25.041 GiB peak for a 15.823 GiB layer, IDENTICAL on
                                 # both cards — a property of the model, not of the box, which is
                                 # why the profile declares it. A node that never ran a K3 probe is
                                 # then still peak-gated, and a node that DID measure it overrides
                                 # rather than adds (plan_ring/probe.derive_layers take the node's
                                 # value when present). Docking both would subtract ~19 GB and cost
                                 # a 32 GB card its only layer — the all-5090 ring would read as
                                 # infeasible, not merely conservative.
                                 #
                                 # This is what produces the G0-proven caps by arithmetic rather
                                 # than by a hardcoded ceiling: (97249-10940)/16313 = 5.29 layers on
                                 # a 96 GB card, (32607-10940)/16313 = 1.33 on a 32 GB 5090.
    "head_reserve_mb": 4096.0,   # coordinator on the head: embed_tokens 2.188 GiB (2241 MB, untied)
                                 # + the coordinator process, tokenizer and logit buffers. Phase 1
                                 # has no drafter to hold (num_nextn_predict_layers = 0, no MTP or
                                 # EAGLE head in the checkpoint). FUTURE: the DSpark drafter is
                                 # 21.4 GB (~20.4 GiB) — adopting it raises this reserve by more
                                 # than a whole layer slot, so a phase-2 head holds ~4 layers where
                                 # it holds 5 today. On a 32 GB card this reserve is what makes the
                                 # head the tightest seat in an all-5090 ring: it needs
                                 # per_layer + reserve + load_peak + this = 31349 MB free, so a 5090
                                 # with more than ~1.2 GB already in use cannot head one.
    "tail_reserve_mb": 2500.0,   # final norm + lm_head 2.188 GiB (2241 MB, untied) + the model-level
                                 # output AttnRes proj/norm (KB) + slack. Same failure this guards on
                                 # M2.5: a tail packed to its brim OOMs loading the output head.
    "cap_layers": 5,             # G0-PROVEN on a 94.97 GiB Pro 6000: 5 layers resident 79.113 GiB /
                                 # 88.332 GiB peak, and the SIXTH layer OOMs allocating 588 MiB. A
                                 # 32 GB 5090 holds exactly ONE (25.04 GiB peak, 6.3 GiB headroom).
                                 # Note both of those already fall out of the reserve arithmetic
                                 # above: this stays a non-binding SANITY ceiling, which is the only
                                 # thing it can honestly be here. density_cap_layers scales a proven
                                 # cap LINEARLY with card size, and K3's true ceiling is not linear
                                 # in it — the repack transient is a fixed additive term, so the
                                 # real rule is floor((vram - 9.2 GiB) / 15.8 GiB). The linear rule
                                 # reads 14-15 layers on a 96 GB card, well above the budget's 5,
                                 # so it never binds and never has to be right.
    "head_layer_ms_mult": 1.3,   # carried from M2.5 (the coordinator penalty is engine-side, not
                                 # model-side) — NOT re-measured on K3.
    # per-hop payloads. K3's boundary is not a hidden state: see _K3_BOUNDARY_BYTES_PER_TOKEN.
    # predict_step_ms charges ONE decode_bytes to every stage alike, so the cost model cannot
    # express a payload that grows with depth; the mean over the 92 boundaries is the faithful
    # single number, and a ring's stage boundaries are near-evenly spaced so their mean lands on
    # it. The error is per-hop (a head stage is over-charged ~2.7x, the tail under-charged ~1.7x),
    # not in aggregate.
    "prefill_bytes": 4096 * _K3_BOUNDARY_BYTES_PER_TOKEN,   # S*(1+nb)*H*2 for a 4096-token prompt
                                 # = 299 MiB/hop, against M2.5's 25 MB. Pipelined as prefill_chunks,
                                 # and the chunk is what has to FIT: transport.MAX_FRAME is 256 MiB,
                                 # and at the deepest boundary (9 units, 126 KiB/token) a 4096-token
                                 # chunk is 504 MiB — unsendable. 2048 lands at 252 MiB, inside the
                                 # frame but with no room for framing overhead. 1024 (126 MiB
                                 # worst-case) is the largest chunk that is comfortably deployable,
                                 # so the nominal 4096-token prefill is FOUR chunks, not one.
                                 # THE ENGINE DOES NOT DEFAULT TO THIS: coordinate.py ships
                                 # --prefill-chunk 4096, which a K3 ring must override or its first
                                 # deep hop dies on "frame length ... exceeds MAX_FRAME".
    "decode_bytes": 1 * _K3_BOUNDARY_BYTES_PER_TOKEN,       # ~74.9 KiB — one token's boundary
                                 # payload. Phase 1 runs g=1: the checkpoint carries no MTP head and
                                 # no drafter is trained, so a traversal moves ONE token, not M2.5's
                                 # (K+1)-token draft bundle.
    "decode_steps": 600,         # nominal decode traversals per request. Same nominal ~600-token
                                 # response M25_PROFILE models, but at g=1 that is 600 traversals
                                 # rather than 256 — and K3 reasons on every request by default.
    "prefill_chunks": 4096 // _K3_PREFILL_CHUNK,
}

# The engine profile per catalog model_id — the seam c0mpute's control plane resolves before it
# calls plan_ring (its catalog holds a model_id and a manifest ref; the calibration is ours). Keys
# are the manifest model_id, which NAMES THE QUANT: two quantizations of one model are two entries,
# never one, because layer_vram_mb (and the weight_map behind it) differ.
PROFILES = {
    "nvidia/MiniMax-M2.5-NVFP4": M25_PROFILE,
    "moonshotai/Kimi-K3-MXFP4": K3_PROFILE,
}


def profile_for(model_id: str) -> dict:
    """The engine profile for a catalog `model_id`, as a copy the caller may override.

    Unknown ids raise rather than falling back to M2.5's calibration: planning an unknown model at
    another model's per-layer footprint is exactly the admit-then-OOM failure the measured numbers
    exist to prevent. ValueError, not KeyError — `_main` reserves KeyError for a request that is
    missing a field, and an unserviceable model_id is a bad VALUE, not an absent one."""
    try:
        return dict(PROFILES[model_id])
    except KeyError:
        raise ValueError(f"no engine profile for model_id {model_id!r} "
                         f"(known: {', '.join(sorted(PROFILES))})") from None

_SLACK = 3                       # default pool headroom for the exact subset search: k_min..k_min+3.
                                 # slack=len(nodes) made select_ring's exact search range over EVERY
                                 # k up to the pool size (combinatorial in a wide pool), defeating
                                 # the trim funnel; select_ring still widens past this on its own
                                 # when co-location forces a bigger ring, so feasibility is intact.

_UNREACHABLE = 9000.0            # RTT sentinel: treat >= this as "no usable path" when ranking centrality

_PROVEN_CAP_VRAM_MB = 32768.0    # the card size cap_layers was proven on; bigger cards scale by density


def density_cap_layers(cap_layers, total_vram_mb):
    """The proven layer DENSITY scaled to the card size — a flat cap collapsed a 96 GB card
    to the 32 GB verdict (the spec's core distinction). ONE rule, shared with probe.derive_layers.

    ROUNDED, not truncated. `_PROVEN_CAP_VRAM_MB` is the card's NOMINAL size, and no real card
    reports it: the 5090s this cap was proven on report 32103-32117 MB once ECC/driver overhead is
    taken out. Truncating then docked every one of them a full layer for a <2% shortfall against a
    marketing number, which is how a 7-node pool that physically holds 62 layers was rejected as
    "need more/fatter nodes". Rounding to nearest restores the intended ceiling on the proven card
    without the ceil() behaviour of granting a 13th layer to a card one MB over the anchor (that
    lands at ~98% VRAM, the configuration that OOM'd live).

    This is only ever a SANITY CEILING; the binding rule is the footprint arithmetic in plan_ring
    (measured layer_vram_mb + kv + reserves), which stays strictly conservative.
    """
    return max(0, int(round(int(cap_layers) * float(total_vram_mb) / _PROVEN_CAP_VRAM_MB)))


def plan_ring(nodes, rtt, model=None, *, slack=None, privacy=None):
    """Place a deployable sharded ring from announced capabilities + a measured RTT mesh.

    nodes: [{"id": <hashable>, "free_vram_mb": float, "subnet": str,
             "cpu_factor": float=1.0,          # >=1; pyloop/0.10 + load — a slow/loaded box drafts slower
             "up_mbps": float|None,            # optional; present on ALL nodes -> upload-aware placement
             "trusted": bool=False,            # ASSIGNED by the control plane (stake/reputation), never
                                               # self-reported by the node
             # per-node MEASURED capability (the probe's cap vector; every field optional — absent
             # fields fall back to the model profile, so a homogeneous pool plans byte-identically):
             "layer_vram_mb": float|None,      # this node's per-layer footprint (arch/backend-specific:
                                               # cutlass ~2330, marlin ~4060 — a hetero ring needs both)
             "cap_layers": int|None,           # probe-verdict layer ceiling for THIS card (wins outright)
             "total_vram_mb": float|None,      # else: density-scale the proven cap to the card size
             "load_peak_extra_mb": float|None, # measured load/run transient above resident (peak
                                               # gate). ABSENT -> the model profile's own, when it
                                               # declares one: a transient that is a property of the
                                               # MODEL (K3's MXFP4 repack peaks the same 9.2 GiB on
                                               # every card) must gate a node that never probed it.
                                               # The node's own measurement WINS rather than adding,
                                               # or an honest probe would dock the same peak twice
             "layer_ms": float|None}]          # measured decode ms/layer (overrides base*cpu_factor)
    rtt:   NxN one-way ms matrix, row/col order aligned to `nodes` (rtt[i][i] ignored).
    model: profile dict (see M25_PROFILE), or a catalog model_id resolved through `profile_for`
           (see PROFILES); defaults to M2.5.
    slack: select_ring pool headroom; defaults to min(len(nodes), 3) — enough to drop weak/co-located
           boxes without letting the exact subset search range over every k up to the pool size.
    privacy: {"boundary_in": int, "boundary_out": int} — turn on BOUNDARY-LAYER PINNING: the ring's
             head/tail (they handle raw prompt / output tokens) and every stage holding a boundary
             layer must be `trusted` nodes; strangers hold only deep-middle layers. The head is the
             most central TRUSTED capable node under pinning (it runs the coordinator, which sees
             the raw prompt). None (default) = placement exactly as before.

    Returns a plan dict, or None if the pool genuinely can't hold the model (with pinning: can't
    hold it SAFELY — e.g. no trusted node for an end):
      {"order":  [node_id, ...],                       # head-first, deployable
       "head":   node_id,
       "stages": [{"id", "index", "lo", "hi", "head", "tail", "layers",
                   "boundary"}...],                    # "boundary" only when privacy pinning is on
       "dropped":[node_id, ...],
       "roles":  {node_id: role},                      # only when every node carries up_mbps
       "step_ms", "tok_s_per_g", "k",
       "request_ms", "prefill_ms",                     # only when upload-aware
       "privacy": {"boundary_in", "boundary_out", "boundary_stages"}}   # only when pinning is on
    """
    if isinstance(model, str):
        model = profile_for(model)
    m = {**M25_PROFILE, **(model or {})}
    n = len(nodes)
    if n == 0:
        return None
    ids = [nd["id"] for nd in nodes]
    if len(set(ids)) != len(ids):                            # duplicate ids collide in the output maps
        raise ValueError("duplicate node id in `nodes`")     # (order/roles/boundary_stages) -> mis-deploy
    layer_vram, kv = float(m["layer_vram_mb"]), float(m["kv_mb_per_layer"])
    cap_layers = int(m["cap_layers"])

    # 1) calibrate free VRAM per node: strip the per-box reserve AND the node's measured load-peak
    #    transient (the admit-then-OOM gate), then cap at the proven layer ceiling — density-scaled
    #    to the card size when the node announces one (a flat cap collapsed a 96 GB card to the
    #    32 GB verdict). The footprint is per-NODE too: a marlin card (~4.1 GB/layer) and a cutlass
    #    card (~2.3 GB) hold very different blocks, and select_ring already takes the per-node dict.
    lv = {i: float(nodes[i].get("layer_vram_mb") or layer_vram) for i in range(n)}
    per_layer = {i: lv[i] + kv for i in range(n)}

    def _node_cap(i):
        if nodes[i].get("cap_layers") is not None:
            return int(nodes[i]["cap_layers"])                    # probe-verdict ceiling wins outright
        total = float(nodes[i].get("total_vram_mb") or 0.0)
        return density_cap_layers(cap_layers, total) if total > 0 else cap_layers

    free = {i: min(max(nodes[i]["free_vram_mb"] - float(m["reserve_mb"])
                       - float(nodes[i].get("load_peak_extra_mb")
                               or m.get("load_peak_extra_mb") or 0.0), 0.0),
                   _node_cap(i) * per_layer[i])
            for i in range(n)}
    cap_ok = [i for i in range(n) if free[i] >= per_layer[i]]
    if not cap_ok:
        return None                                          # no node can hold even one layer

    # 2) head = most central capable node (lowest total RTT to the rest); it runs the coordinator.
    #    Under privacy pinning the coordinator sees the raw prompt, so the head must be TRUSTED —
    #    rank centrality over trusted capable nodes only.
    pin = privacy is not None
    # STRICT bool — trust is the security boundary, so read it fail-CLOSED: only a genuine `True`
    # (JSON `true`) marks a node trusted. A truthy string like "false"/"0" or an int must NOT sneak a
    # node into the trust set (a control plane that serialized the flag as a string would otherwise
    # fail OPEN — the one way a stranger could reach a boundary while the plan claims to be pinned).
    trusted = {i for i in range(n) if nodes[i].get("trusted") is True} if pin else None
    head_pool = [i for i in cap_ok if i in trusted] if pin else cap_ok
    if not head_pool:
        return None                                          # pinning on, but no trusted node can hold a block

    def centrality(i):
        # clamp each edge at the sentinel instead of OMITTING unreachable ones — omission summed a
        # fully-disconnected node to 0, which won min() and made it the mandatory head (undeployable)
        return sum(min(float(rtt[i][j]), _UNREACHABLE) for j in range(n) if j != i)

    def _connected_cap(i):
        # layers reachable from i: its own budget + every capable peer with a finite path BOTH ways
        return int(free[i] // per_layer[i]) + sum(
            int(free[j] // per_layer[j]) for j in cap_ok
            if j != i and rtt[i][j] < _UNREACHABLE and rtt[j][i] < _UNREACHABLE)
    head_pool = [i for i in head_pool if _connected_cap(i) >= int(m["n_layers"])]
    if not head_pool:
        return None                              # no candidate head can REACH enough capacity to serve
    head = min(head_pool, key=centrality)
    free[head] = max(free[head] - float(m["head_reserve_mb"]), 0.0)

    # 3) launch-bound per-layer time: base * the node's cpu_factor; the head pays a coordinator
    #    penalty. A node announcing a MEASURED layer_ms (the probe's graph-replayed decode number)
    #    is placed at that, not the modeled base — a box whose graph capture failed runs eager at
    #    ~4x and must be planned as what it measured, not what its GPU label suggests.
    layer_ms = {i: (float(nodes[i]["layer_ms"]) if nodes[i].get("layer_ms") is not None
                    else float(m["layer_ms_base"]) * float(nodes[i].get("cpu_factor", 1.0)))
                for i in range(n)}
    layer_ms[head] *= float(m["head_layer_ms_mult"])

    # 4) coordinator entry/return hops are measured relative to the chosen head.
    c_out = [rtt[head][i] if i != head else 1.0 for i in range(n)]
    c_in = [rtt[i][head] if i != head else 1.0 for i in range(n)]
    subnet = {i: nodes[i]["subnet"] for i in range(n)}

    # 5) upload-aware placement iff EVERY node announced an uplink (residential lever); else decode-only.
    ups = [nodes[i].get("up_mbps") for i in range(n)]
    aware = all(u is not None for u in ups)
    extra = {}
    if aware:
        extra = {"up_mbps": {i: float(ups[i]) for i in range(n)},
                 "prefill_bytes": float(m.get("prefill_bytes", 0.0)),
                 "decode_bytes": float(m.get("decode_bytes", 0.0)),
                 "decode_steps": int(m.get("decode_steps", 1)),
                 "prefill_chunks": int(m.get("prefill_chunks", 1))}

    if pin:
        extra["trusted"] = trusted
        extra["boundary_in"] = int(privacy.get("boundary_in", 0))
        extra["boundary_out"] = int(privacy.get("boundary_out", 0))

    # 6) the TAIL stage also holds the final norm + lm_head (measured 1.15 GiB bf16 on
    #    M2.5 — a 13-layer tail OOM'd loading it on a 32 GB 5090, live 2026-07-09, while
    #    the same 13 layers warmed fine as a middle). The reserve applies to WHICHEVER
    #    node lands the tail, which select_ring decides — so plan, check the landed
    #    tail's block against the reserve, and if it doesn't fit, bake the reserve into
    #    that node's budget EXACTLY ONCE and re-plan (the tail may move). Convergence is
    #    checked against the ORIGINAL budget: the old loop compared against the already-
    #    docked value — re-demanding the reserve on top of itself — so a node that
    #    reappeared as tail was docked again each round (a feasible pool read as
    #    infeasible), and after 4 blind rounds the LAST spec was returned even when its
    #    tail never fit at all. The docked set is finite and only grows, so this
    #    converges in <= n rounds or honestly reports None.
    tail_reserve = float(m.get("tail_reserve_mb", 0.0))
    base_free = dict(free)                       # budgets to validate against (head reserve included)
    docked = set()                               # nodes whose budget already models the tail reserve
    spec = None
    for _ in range(n + 1):
        spec = select_ring(range(n), rtt, c_out, c_in, free_vram_mb=free, layer_ms=layer_ms,
                           subnet=subnet, n_layers=int(m["n_layers"]), layer_vram_mb=lv,
                           kv_mb_per_layer=kv, slack=min(n, _SLACK) if slack is None else int(slack),
                           require=head, **extra)
        if spec is None:
            return None
        tail_i = spec["order"][-1]
        lo, hi = spec["blocks"][tail_i]
        if tail_reserve == 0.0 or base_free[tail_i] >= (hi - lo) * per_layer[tail_i] + tail_reserve:
            break                                # the landed tail fits block + reserve in its budget
        if tail_i in docked:
            return None                          # reserve already modeled and it STILL can't fit
        docked.add(tail_i)
        free[tail_i] = max(base_free[tail_i] - tail_reserve, 0.0)
    else:
        return None                              # no tail placement converged: the reserve fits nowhere
    assert spec["order"][0] == head, "select_ring must return a head-first (deployable) order"
    # a deployable ring never traverses an unreachable (sentinel) edge: the forward hops and the
    # tail -> head coordinator return must all be measured, finite paths — if feasibility forced
    # one in, there IS no usable ring, so say so instead of shipping a dead hop
    _o = spec["order"]
    if (any(rtt[a][b] >= _UNREACHABLE for a, b in zip(_o, _o[1:]))
            or (_o[-1] != head and rtt[_o[-1]][head] >= _UNREACHABLE)):
        return None
    # belt-and-braces: every stage's block must fit the node's ORIGINAL budget (the tail
    # including its reserve) — a violation here is a planner bug, never a deployable answer
    for i in spec["order"]:
        lo, hi = spec["blocks"][i]
        need = (hi - lo) * per_layer[i] + (tail_reserve if i == spec["order"][-1] else 0.0)
        if need > base_free[i] + 1e-6:
            raise RuntimeError(f"planned block [{lo}:{hi}) needs {need:.0f} MB on node {ids[i]!r} "
                               f"whose budget is {base_free[i]:.0f} MB")

    boundary = set(spec.get("boundary", []))
    order = [ids[i] for i in spec["order"]]
    last = len(spec["order"]) - 1
    stages = []
    for k, i in enumerate(spec["order"]):
        lo, hi = spec["blocks"][i]
        st = {"id": ids[i], "index": k, "lo": lo, "hi": hi,
              "head": k == 0, "tail": k == last, "layers": hi - lo}
        if pin:
            st["boundary"] = i in boundary
        stages.append(st)
    out = {
        "order": order,
        "head": ids[head],
        "stages": stages,
        "dropped": [ids[i] for i in spec["dropped"]],
        "step_ms": spec["step_ms"],
        "tok_s_per_g": spec["tok_s_per_g"],
        "k": spec["k"],
    }
    if aware:
        out["request_ms"] = spec.get("request_ms")
        out["prefill_ms"] = spec.get("prefill_ms")
        out["roles"] = {ids[int(i)]: r for i, r in spec.get("roles", {}).items()}
    if pin:
        out["privacy"] = {"boundary_in": extra["boundary_in"], "boundary_out": extra["boundary_out"],
                          "boundary_stages": [ids[i] for i in spec["order"] if i in boundary]}
    return out


def _main() -> int:
    """`python3 -m shard.plan` — JSON in ({nodes, rtt, model?, slack?}), JSON out (the plan, or null).
    `model` is a profile dict or a catalog model_id string (PROFILES)."""
    try:
        req = json.load(sys.stdin)
    except Exception as e:  # noqa: BLE001 — a malformed request is a caller error, report it as JSON
        json.dump({"error": f"bad request json: {e}"}, sys.stdout)
        return 2
    try:
        plan = plan_ring(req["nodes"], req["rtt"], req.get("model"), slack=req.get("slack"),
                         privacy=req.get("privacy"))
    except KeyError as e:
        json.dump({"error": f"missing field: {e}"}, sys.stdout)
        return 2
    except Exception as e:  # noqa: BLE001
        json.dump({"error": f"plan failed: {e}"}, sys.stdout)
        return 1
    json.dump(plan, sys.stdout)          # `null` when the pool can't hold the model — a valid answer
    return 0


if __name__ == "__main__":
    sys.exit(_main())
