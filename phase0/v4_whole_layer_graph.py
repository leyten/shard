"""Whole-layer CUDA graphs for a V4 decode step: the attention core made capture-safe.

`v4_stage`'s island graphs (V4_CUDA_GRAPH=1) capture only the position- and data-INDEPENDENT Block
methods (hc_pre/hc_post/the two RMSNorms) and leave `attn` and `ffn` EAGER between them, because three
things in the attention core bake position or data into a graph and replay wrong:

  #1 the ROTATING window write  `kv_cache[:, start_pos % win] = kv`  -- a slot that moves every step;
  #2 the GROWING compressed read  the Indexer scores over `kv_cache[:, :end_pos // ratio]` (a slice
     whose width grows with position) and picks `topk(min(index_topk, end_pos // ratio))` (a k that
     grows), and the window/compress index lists are rebuilt per position;
  #3 the COMPRESS-STEP branch  the Compressor's `should_compress = (start_pos+1) % ratio == 0` is a
     per-position Python branch that either writes a compressed slot or does nothing.

This module removes all three so the WHOLE decode layer (attn + islands, and with a graph-safe MoE the
FFN too) captures as one graph -- the standard paged-decode-graph technique M2.5 got from vLLM's
static-KV SDPA and K3 punted on for MLA. It does NOT edit the vendored reference (that stays
byte-identical, deepseek_v4_ref/PROVENANCE.md); it is a second transcription of the reference's decode
branch that calls the reference's OWN kernels and parameters, so a correct capture replays the same
kernels on the same bytes -- bit-exact, not reassociation-drifted. The three fixes, each proven against
`Attention.forward` on CPU before a GPU ever runs it (tests/test_v4_whole_layer.py):

  #1 DEVICE-SIDE POSITION.  `start_pos` enters as a length-1 device buffer, not a Python int. Every
     slot that was `p % win` / `p % ratio` / `p // ratio` / `(p+1)//ratio` is that arithmetic on the
     buffer, and every `kv_cache[:, slot] = v` becomes `index_copy_(1, slot, v)` -- the same store to
     the same byte, chosen at REPLAY time from the copied-in position instead of frozen at capture.

  #2 BUCKETED WIDTH + MASK + A STABLE TIE-BREAK.  The Indexer's einsum reads a BUCKET of `kv_cache`
     (m25_stage.py's DECODE_BUCKETS discipline: round the read up to a rung, one graph per rung,
     capture a new one when a rung is crossed) and masks scores at positions >= end_pos//ratio to
     -inf; its selection takes a FIXED k = index_topk and maps any pick that landed on a masked
     (future) slot to -1. sparse_attn already treats a -1 index as "no position"
     (v4_sparse_attn_sm120.py: it zeroes that KV row and pushes the score to -inf, contributing to
     neither numerator nor denominator), so the -1 padding is exactly a no-op -- the extra -inf blocks
     of the online softmax rescale by exp(0)=1 and add 0 -- and the SCORES over the valid columns come
     out bit-identical to the reference's narrow einsum (measured: max|narrow - wide| = 0.0).
     THE SELECTION IS A STABLE SORT, NOT A TOPK, AND THAT IS LOAD-BEARING. v4_stage's `_chunk_indexer`
     already records why for the chunked path: "Masking a common-width score to fake the shorter reads
     would change which slots a topk near-tie picks". `relu_()` floors every negative score to a hard
     0.0, so ties are everywhere, and torch.topk breaks them by an artifact of its partition that
     varies with array WIDTH and with CPU THREAD COUNT -- the reference is not even reproducible
     against itself. Selecting by (score DESC, index ASC) is width-independent, so a bucket crossing is
     invisible and the answer does not depend on capture history. It agrees with the reference on
     every tie-free selection and may pick a different column of IDENTICAL score otherwise: a NAMED,
     bounded Tier-2 divergence (see the grading note below and the receipt's `ties` risk).
     The window/compress index LISTS
     (get_window_topk_idxs / get_compress_topk_idxs) are the reference's own, built eagerly per step
     into a static buffer and copied in before replay: they are a handful of tiny launches, not the
     GEMM-and-attention bulk the graph is for, and building them in-graph would mean transcribing
     their per-position branch structure for no dispatch saving.

  #3 TWO GRAPHS, PICKED BY POSITION.  `should_compress` is known on the HOST from `start_pos`, so the
     block is captured TWICE -- a no-compress variant (the Compressor only advances kv_state/
     score_state) and a compress variant (it also pools, writes the compressed slot, and shifts the
     overlap state) -- and the replay picks between them by `(start_pos+1) % ratio == 0`. This keeps
     ZERO data-dependent control flow inside a graph, which is what makes the compress path provably
     bit-exact; the alternative (one graph, the compress writes masked to no-ops) would have to mask
     the overlap state's running shift too, a strictly larger surface to get wrong for a step that
     happens once every `ratio` positions. A pure sliding-window layer (compress_ratio 0) has no
     Compressor and captures ONE graph.

THE MoE, AND WHICH PATH IT IS ON -- the difference between `moe_mode="eager"` and `"graph"`.
The VENDORED MoE routes per token through `bincount(...).tolist()` (a host sync) and a data-dependent
expert loop, and `v4_moe_decode`'s s==1 fast path still opens with `indices[0].tolist()`; NEITHER can
go inside a graph, and a capture that took one anyway would bake the CAPTURE token's expert list into
the replayed program and serve every later token through the wrong experts, silently, forever. That
is why `moe_mode="eager"` (graph the attention core, run the real routed MoE eager between two
graphs) was the honest intermediate, and `moe_mode="stub"` (a FIXED expert set, the real decode
activation count) the timing probe.

`v4_moe_grouped.grouped_forward` removed the sync. Under `V4_MOE_GROUPED=1` on a BANKED fp4 layer a
decode step's whole routing is device-side -- a topk (or a `tid2eid` gather), a `_gather_fp` of the
routed experts out of the layer's contiguous bank, two grouped GEMMs, a `[G, G]` keep-mask for hash
duplicates and an `argsort` fold -- with no `.tolist()`, no `nonzero`, no `bincount` and no shape that
depends on which experts were picked. So `moe_mode="graph"` captures the WHOLE layer, MoE included, as
ONE graph. Audited rather than assumed, on CPU, in tests/test_v4_moe_in_graph.py: the block's aten
program (op sequence, every shape, every non-tensor arg) is recorded at six DIFFERENT routings --
including a hash layer whose ids repeat -- and required to be IDENTICAL, with a `TorchFunctionMode`
watching for any device drain. Identical program + every routing input arriving through a static
buffer is exactly the CUDA-graph replay contract.

REFUSED, PER LAYER, RATHER THAN ASSUMED (`WholeBlockGraphs._moe_refusal`). The mode is a REQUEST; a
layer whose MoE is not provably sync-free drops back to `moe_mode="eager"`, loudly and counted. Four
checks, each closing a way the capture could replay stale routing:
  * the live `MoE.forward` chain must reach `grouped_forward` at s == 1 (peeling `multi`, which hands
    a single token straight down). A chain topped by `decode_forward` or the raw reference is the
    failure above, and it is the DEFAULT chain -- `V4_MOE_GROUPED` is off unless asked for.
  * the layer must already hold a real `_grouped_bank` from `bank_layout()` at load. Without one,
    `_expert_bank` would STACK the bank on first call -- inside the capture, out of the graph's
    PRIVATE memory pool -- and a later capture sharing that pool would hand those bytes to something
    else while the module still pointed at them.
  * `world_size == 1`, the same envelope the kernel itself declines outside.
  * and then it is OBSERVED, not inferred: one eager probe step must increment `_grouped_steps` and
    add no `_grouped_declined` entry. A lever that reads as on and is not on is this engine's most
    expensive recurring bug (phase0/v4_levers.py); a capture is the worst place to start trusting one.
`ids` becomes LOAD-BEARING in this mode -- the first `n_hash_layers` route on `tid2eid[input_ids]` --
so `run()` refuses a `None` ids rather than replaying the capture step's token.

WHAT IT COSTS: the per-step `_gather_fp` of the routed experts is an allocation INSIDE the capture, so
it is pinned in the graph's private pool for the life of the graph -- at the shipped dims ~77 MiB per
layer (6 slots x 8 MiB w13 + 4 MiB w2, plus scales), shared across that layer's bucket/compress
variants because they share one pool. ~0.5 GiB on a 7-layer stage, against the ~6 GiB of headroom the
bank layout left. It is not free and it is not the reason to refuse.

Opt-in, default OFF: `V4_MOE_IN_GRAPH=1` (registered in phase0/v4_levers.py). With the env unset
`v4_stage` builds `moe_mode="eager"` and this file behaves exactly as it did.

HOW THIS IS GRADED (m25's two-tier bar, research/graph_aux_check.py -- the bar its CUDA graphs, the
single biggest M2.5 win at +74%, actually shipped under):

  TIER 1, HARD, torch.equal:   graphed == the EAGER TWIN (`WholeBlockGraphs._eager`) -- same math, same
      bucket decisions, no capture. This is the real capture-correctness proof: a graph replays the
      same kernels on the same bytes, so it has no licence to differ by one ulp, and every failure
      mode this module can have (a position baked at capture, a stale static buffer, a state write
      double-applied) shows up here.
  TIER 2, NAMED + BOUNDED:     vs the VENDORED reference -- bit-exact wherever the Indexer's selection
      is unambiguous, and otherwise divergent only by which member of an EXACT score tie is attended
      (never a lower-scoring column; proven on the selection function). Token-identity against the
      reference is not a bar the reference can meet: its topk tie-break is not reproducible against
      itself across array widths or OMP_NUM_THREADS.
  FRESHNESS GATE:              the same input at a DIFFERENT position must MOVE the output. Cheapest
      insurance against the worst silent failure here -- position baked in at capture -- which m25 was
      actually bitten by (stale EAGLE aux, a Python side-effect inside the captured region that
      replay() skipped, so the drafter fed on prefill aux forever behind valid receipts).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The Indexer's read length `end_pos // ratio` GROWS every `ratio` positions, so a graph cannot bake
# it and reading the whole `max_seq_len // ratio` at every position pays full-context cost from token
# one. m25_stage.py's answer (DECODE_BUCKETS, :208) is the one used here: round the read up to a rung,
# capture a graph per rung, and cross a rung by capturing a NEW graph -- so the count is a priori
# bounded by the ladder, not by the sequence length. x4 rungs, and the small ones are real: a decode
# that has only compressed 20 slots should read 64, not 16384.
INDEXER_BUCKETS = (16, 64, 256, 1024, 4096, 16384, 65536)
# Every captured graph pins its own workspace; cap the set process-wide exactly like v4_stage's island
# graphs and m25's M25_GRAPH_MAX. Past the cap a layer stays EAGER -- counted and logged, never a crash.
V4_GRAPH_MAX = int(os.environ.get("V4_GRAPH_MAX", "192"))
_GRAPH_COUNT = 0        # whole-layer graphs captured so far, across every Stage in this process
_GRAPH_SKIPPED = 0      # captures a layer skipped because the cap was hit or a capture failed

# Pull the routed MoE INSIDE the whole-layer graph (see the module docstring). Opt-in, default OFF:
# it is only correct on a layer whose MoE reaches `v4_moe_grouped.grouped_forward`, which is itself
# opt-in and CUDA-only, and a layer that cannot prove it drops back to `moe_mode="eager"`.
V4_MOE_IN_GRAPH = os.environ.get("V4_MOE_IN_GRAPH", "0") not in ("", "0")
_MOE_GRAPHED = 0        # layers whose MoE really was captured inside their whole-layer graph
_MOE_REFUSED = 0        # layers that asked for it and were refused (each one printed why)


def bucket_width(need, maxw, floor=0):
    """Smallest INDEXER_BUCKETS rung >= `need`, floored at `floor` and clamped to `maxw`.

    m25_stage._bucket, per-cache, with one extra constraint this cache has and m25's did not: the read
    must be wide enough to SELECT from. `_select_topk_width_invariant` takes a fixed k = index_topk, so
    a bucket narrower than index_topk could not hold the picks -- pass floor=index_topk. (When
    index_topk >= maxw the ladder degenerates to the full width, which is correct, just not a saving:
    at V4's shipped shape index_topk is 512 against max_seq_len//4 = 16384, so the rungs are real.)"""
    need = max(need, floor)
    for b in INDEXER_BUCKETS:
        if b >= need:
            return min(b, maxw)
    return maxw


# ── the reference's kernels and globals, resolved once ───────────────────────────────────────────

class _Ref:
    """The handful of names the capture-safe transcription borrows from the loaded reference module.

    Grabbed off `dsv4_model` (the exec'd model.py) rather than re-imported, so `act_quant`, `sparse_attn`
    and friends are the SAME objects the reference calls -- on a GPU box that means the tilelang / sm120
    kernels, and the module globals scale_fmt/scale_dtype/fp4_block_size are whatever _set_globals put
    there. A second import could bind a different backend and the capture would no longer be the same
    kernels as eager."""

    def __init__(self, M):
        self.apply_rotary_emb = M.apply_rotary_emb
        self.rotate_activation = M.rotate_activation
        self.act_quant = M.act_quant
        self.fp4_act_quant = M.fp4_act_quant
        self.sparse_attn = M.sparse_attn
        self.M = M

    @property
    def scale_fmt(self):
        return self.M.scale_fmt

    @property
    def scale_dtype(self):
        return self.M.scale_dtype

    @property
    def fp4_block_size(self):
        return self.M.fp4_block_size


# ── the capture-safe decode transcription (bit-exact vs Attention.forward, start_pos>0, seqlen==1) ──

def _compressor_decode_cs(R, C, x, pos, compress):
    """Compressor.forward's decode branch, device-indexed. `compress` is should_compress (host-known).

    Line for line against model.py Compressor.forward (start_pos>0): the only changes are the four
    position-baked stores (`kv_state[:, ratio+p%ratio]`, `score_state[:, ...]`, the overlap shift, and
    `kv_cache[:, p//ratio]`) done through index_copy_/copy_ on device-derived slots, and the ape /
    freqs_cis rows fetched by index_select instead of a Python-int index. The rope row is the
    compressor's OWN `freqs_cis[start_pos + 1 - ratio]` (model.py:372), not attention's current-position
    row -- the Indexer's compressor is bound to the same freqs table and lags attention by ratio-1.
    Returns nothing (the state IS the output); on a no-compress step it advances kv_state/score_state
    and stops, exactly as the reference returns None before touching kv_cache."""
    ratio, overlap, d, rd = C.compress_ratio, C.overlap, C.head_dim, C.rope_head_dim
    slot_r = pos % ratio
    comp_slot = pos // ratio
    dtype = x.dtype
    x = x.float()
    kv = C.wkv(x)
    score = C.wgate(x)
    # score += self.ape[start_pos % ratio]  -> ape row picked on device
    score = score + C.ape.index_select(0, slot_r)
    kv1 = kv.squeeze(1)
    score1 = score.squeeze(1)
    ks = C.kv_state.narrow(0, 0, 1)
    ss = C.score_state.narrow(0, 0, 1)
    if overlap:
        ks.index_copy_(1, ratio + slot_r, kv1.unsqueeze(1))
        ss.index_copy_(1, ratio + slot_r, score1.unsqueeze(1))
        if compress:
            kv_state = torch.cat([ks[:, :ratio, :d], ks[:, ratio:, d:]], dim=1)
            score_state = torch.cat([ss[:, :ratio, :d], ss[:, ratio:, d:]], dim=1)
            kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
            ks[:, :ratio].copy_(ks[:, ratio:])
            ss[:, :ratio].copy_(ss[:, ratio:])
    else:
        ks.index_copy_(1, slot_r, kv1.unsqueeze(1))
        ss.index_copy_(1, slot_r, score1.unsqueeze(1))
        if compress:
            kv = (ks * ss.softmax(dim=1)).sum(dim=1, keepdim=True)
    if not compress:
        return
    kv = C.norm(kv.to(dtype))
    freqs_row = C.freqs_cis.index_select(0, pos + 1 - ratio)
    R.apply_rotary_emb(kv[..., -rd:], freqs_row)
    if C.rotate:
        kv = R.rotate_activation(kv)
        R.fp4_act_quant(kv, R.fp4_block_size, True)
    else:
        R.act_quant(kv[..., :-rd], 64, R.scale_fmt, R.scale_dtype, True)
    C.kv_cache.narrow(0, 0, 1).index_copy_(1, comp_slot, kv)


def _indexer_decode_cs(R, I, x, qr, pos, end_ratio, freqs_row, offset, arange_w, compress, read_w):
    """Indexer.forward's decode branch with the GROWING read/topk made fixed-width + masked.

    Reference (start_pos>0): scores q against `kv_cache[:, :end_pos//ratio]`, `topk(min(index_topk,
    end_pos//ratio))`, `+= offset`. Here: score against the whole `kv_cache` (fixed MAXW), mask
    columns >= end_ratio to -inf, take a FIXED topk of index_topk, and send any selection at a masked
    (future) column to -1. The valid selections match the reference in value AND order (a -inf column
    never outranks a finite one, so the top min(index_topk, end_ratio) finite columns are identical),
    and the -1 padding is a sparse_attn no-op -- so the concatenated topk feeds byte-identical
    attention. `offset` (= window_size on decode) shifts a compressed slot index into the attn
    kv_cache's compressed region, exactly as the reference's `topk_idxs += offset`."""
    n_local_heads, head_dim, rd = I.n_local_heads, I.head_dim, I.rope_head_dim
    index_topk = I.index_topk
    q = I.wq_b(qr)
    q = q.unflatten(-1, (n_local_heads, head_dim))
    R.apply_rotary_emb(q[..., -rd:], freqs_row)
    q = R.rotate_activation(q)
    R.fp4_act_quant(q, R.fp4_block_size, True)
    _compressor_decode_cs(R, I.compressor, x, pos, compress)
    weights = I.weights_proj(x) * (I.softmax_scale * I.n_heads ** -0.5)
    # BUCKETED read, not the whole cache: `read_w` is a host-side rung >= this position's end_ratio,
    # fixed for the graph that captured it (see INDEXER_BUCKETS). Columns in [end_ratio, read_w) hold
    # slots this position may not see yet and are masked out below.
    index_score = torch.einsum("bshd,btd->bsht", q, I.kv_cache.narrow(0, 0, 1).narrow(1, 0, read_w))
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    return select_compress_topk(index_score, end_ratio, index_topk, offset, arange_w)


def _select_topk_width_invariant(score, valid, k, arange_w):
    """Top-k of `score` by (value DESC, index ASC) -- the same set and order at ANY padded width.

    THE ONE THING A FIXED-WIDTH TOPK CANNOT BORROW FROM torch. `Tensor.topk` breaks ties by whatever
    its selection algorithm happens to do (a non-stable partial_sort on CPU, a radix select on CUDA);
    that order is not part of the contract and it is NOT invariant to the length of the array. So
    `score[:end].topk(k)` and `score.masked_fill(idx>=end, -inf).topk(k)` -- mathematically the same
    query -- can return DIFFERENT SETS whenever the k-th and (k+1)-th values are equal. They tie
    constantly here: `index_score` is bf16 and passes through `relu_()`, which floors every negative
    column to a hard 0.0, so whole blocks of candidates share a value exactly. That is what broke the
    first cut of this file -- a fixed-width read picked a different compressed KV slot than the
    reference at decode pos 43 and the hidden state diverged by ~2e-3 (tests/test_v4_whole_layer.py).

    (value, -index) is a TOTAL order on the valid columns, so the top-k under it is unique and cannot
    depend on how many -inf columns are padded on the end. Concretely: take the k-th largest VALUE
    (a multiset property, already width-invariant), admit every column strictly above it, then admit
    the tied columns in ascending index until k are held. All fixed-shape, all device-side -- no host
    sync, nothing baked at capture -- so this is what makes a fixed-width OR bucketed read legitimate
    rather than merely usually-right.

    It does NOT reproduce torch's tie order, and cannot: see the module docstring's TIER 2 note.

    IMPLEMENTED AS A STABLE DESCENDING SORT, and the lane ORDER that falls out is worth as much as the
    set. `sort(descending=True, stable=True)` IS the (value DESC, index ASC) order: equal values keep
    ascending index, and the -inf padding sorts to the tail whatever the width, so the first k are the
    same picks IN THE SAME LANES at any padded width.

    The alternative -- take the k-th value, admit `> kth`, then admit ties by cumsum -- selects the
    same SET more cheaply, but emits it in ascending INDEX order. That is a second, independent source
    of last-bit divergence from the reference: `topk` hands sparse_attn its picks in DESCENDING SCORE
    order, and sparse_attn's per-block reduction is order-sensitive, so re-laning the same support set
    regroups the sum. Measured on this branch at V4's real dims, moe_eager vs the vendored reference
    over 40 decode steps: ascending-index lanes gave 2/40 steps bit-exact (max 9.4e-2), descending-score
    lanes gave 40/40 bit-exact (0.0). The sort costs a little more than the cumsum trick and is worth
    it -- it removes a whole class of Tier-2 noise, leaving genuine ties as the only difference."""
    s = score.masked_fill(~valid, float("-inf"))
    pick = s.sort(dim=-1, descending=True, stable=True).indices.narrow(-1, 0, k)
    return pick, valid.gather(-1, pick)


def select_compress_topk(index_score, end_ratio, index_topk, offset, arange_w):
    """Bucket-width masked selection of the compressed slots to attend: the #2 rewrite, in one place.

    Reference (Indexer.forward, decode): `index_score[..., :end_pos//ratio].topk(min(index_topk,
    end_pos//ratio))[1] + offset` -- a GROWING read width and a GROWING k, neither capturable. Here the
    scores arrive at a BUCKET width >= end_ratio, columns past `end_ratio` are masked to -inf, k is the
    fixed `index_topk`, and any pick that lands on a masked (future) column becomes -1, which
    sparse_attn treats as "no position".

    THE TIE-BREAK IS WHAT MAKES BUCKETING LEGAL, and it is the whole reason `topk` is not used here.
    v4_stage's own `_chunk_indexer` says it for the chunked path: "Masking a common-width score to fake
    the shorter reads would change which slots a topk near-tie picks". `Tensor.topk` does not define
    its tie order -- a non-stable partial_sort on CPU, a radix select on CUDA -- and that order is NOT
    invariant to array length, while `index_score` is bf16 behind a `relu_()` that floors every
    negative column to a hard 0.0, so ties at the k-th rank are routine (measured: 23 of 120 decode
    steps). With `topk` the answer would depend on WHICH BUCKET a position happened to land in, i.e. on
    capture history. `_select_topk_width_invariant` imposes a total order instead, so a position gets
    the same slots at bucket 64 as at 16384 and a rung crossing is invisible."""
    arange = arange_w.view(1, 1, -1)
    valid = arange < end_ratio
    pick, kept = _select_topk_width_invariant(index_score, valid, index_topk, arange)
    return torch.where(kept, pick + offset, pick.new_full((), -1)).int()


def attn_decode_cs(R, A, x, pos, win_topk, comp_topk, arange_w, compress, read_w):
    """Attention.forward's decode branch (start_pos>0, seqlen==1), capture-safe. Bit-exact vs eager.

    `x` is the attn_norm output (the reference passes exactly this to wq_a/wkv/compressor/indexer).
    `win_topk` is get_window_topk_idxs' result for this step (fed via a static buffer); `comp_topk` is
    get_compress_topk_idxs' padded result for a non-indexer compress layer, or None. The order of state
    writes is the reference's -- indexer (which advances its own compressor + kv_cache), THEN the window
    store, THEN the attn compressor, THEN sparse_attn -- because sparse_attn's topk can reference the
    compressed slot the compressor just wrote (the margin is exactly zero; see v4_stage._snapshot)."""
    win, ratio, rd = A.window_size, A.compress_ratio, A.rope_head_dim
    freqs_row = A.freqs_cis.index_select(0, pos)
    # q
    qr = q = A.q_norm(A.wq_a(x))
    q = A.wq_b(q).unflatten(-1, (A.n_local_heads, A.head_dim))
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + A.eps)
    R.apply_rotary_emb(q[..., -rd:], freqs_row)
    # window kv + topk
    kv = A.wkv(x)
    kv = A.kv_norm(kv)
    R.apply_rotary_emb(kv[..., -rd:], freqs_row)
    R.act_quant(kv[..., :-rd], 64, R.scale_fmt, R.scale_dtype, True)
    topk_idxs = win_topk
    if ratio:
        end_ratio = (pos + 1) // ratio
        if A.indexer is not None:
            compress_topk_idxs = _indexer_decode_cs(R, A.indexer, x, qr, pos, end_ratio, freqs_row,
                                                    win, arange_w, compress, read_w)
        else:
            compress_topk_idxs = comp_topk
        topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
    # decode store + compress, then attend
    A.kv_cache.narrow(0, 0, 1).index_copy_(1, pos % win, kv)
    if ratio:
        _compressor_decode_cs(R, A.compressor, x, pos, compress)
    o = R.sparse_attn(q, A.kv_cache.narrow(0, 0, 1), A.attn_sink, topk_idxs, A.softmax_scale)
    R.apply_rotary_emb(o[..., -rd:], freqs_row, True)
    # o
    o = o.view(1, 1, A.n_local_groups, -1)
    wo_a = A.wo_a.weight.view(A.n_local_groups, A.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    return A.wo_b(o.flatten(2))


# ── a graph-safe MoE stub (fixed expert set), so the whole layer can be captured end to end ──────────

def moe_stub(L, x, ids=None):
    """A FIXED-expert-set stand-in for MoE.forward -- graph-safe, and the real decode work profile.

    The reference MoE picks experts per token through `bincount(...).tolist()` (a host sync) and a
    data-dependent `for i in range(...): where(indices==i)` loop, neither of which a graph can hold.
    This runs the shared expert plus a CONSTANT routed pair (n_activated_experts of them, weight 1),
    which is the same number of SwiGLU FFNs a decode token really fires -- so it is the right launch/
    FLOP shape for a whole-layer timing, while being a fixed program. It is NOT the routed model's
    output; it exists to prove the whole-layer graph composes and to measure it. The sibling
    grouped-fp4 MoE (phase0/v4_moe_grouped.py) is the real graph-safe replacement."""
    ffn = L.ffn
    shape = x.size()
    xf = x.view(-1, ffn.dim)
    y = ffn.shared_experts(xf)
    k = ffn.n_activated_experts
    picked = 0
    for i in range(ffn.experts_start_idx, ffn.experts_end_idx):
        if ffn.experts[i] is None or picked >= k:
            continue
        y = y + ffn.experts[i](xf)
        picked += 1
    return y.type_as(x).view(shape)


def real_moe(L, x, ids):
    """The routed MoE as the process has it bound. The `moe_eager` intermediate, the parity oracle,
    and -- when `_moe_refusal` clears the layer -- the thing captured INSIDE the whole-layer graph."""
    return L.ffn(x, ids)


# ── may this layer's routed MoE go inside a graph? ───────────────────────────────────────────────

def grouped_at_one_token(mod):
    """Does a SINGLE-TOKEN step on `mod` reach `v4_moe_grouped.grouped_forward`?

    `MoE.forward` is a CHAIN of rebinds (v4_levers.moe_chain): multi -> grouped -> decode -> ref, each
    link capturing the one below as its fallback. `multi` declines `T == 1` on its first line and hands
    the step straight down, so it is transparent here; every other link that sits ABOVE grouped -- i.e.
    grouped absent from the chain, or `decode`/`ref` reached first -- means a decode token is dispatched
    by a `.tolist()` and a data-dependent expert loop, which is precisely what must not be captured.

    Reads the LIVE bound functions, never an env flag: `v4_moe_grouped.install()` declines silently off
    CUDA and under `V4_MOE_GROUPED=0`, so the flag proves nothing about what this process will run."""
    import v4_levers
    chain = v4_levers.moe_chain(mod)
    while chain and chain[0] == "multi":            # transparent at one token, by construction
        chain = chain[1:]
    return bool(chain) and chain[0] == "grouped"


def moe_probe(L, dev, dt, ids):
    """Run this layer's MoE ONCE, eagerly, and report whether the grouped path took the step.

    OBSERVED, NOT INFERRED. Everything above is structural -- the right function is bound, the bank
    exists -- and this engine's recurring bug is exactly a structure that looked right and did not
    fire (v4_levers' six instances). `grouped_forward` counts its own steps and records every decline
    with a reason, so one probe answers the question directly. The MoE is stateless, so the probe
    perturbs nothing; the counters it moves are put back, because they are what `coverage()` reports
    and a probe must not appear in that number.

    Zeros rather than a random draw on purpose: every reason grouped can decline (token count, world
    size, a missing bank) is a property of the SHAPE and the config, never of the activation, so the
    cheapest deterministic input is a faithful observation. Returns None if it grouped, else why."""
    moe = L.ffn
    steps0 = getattr(moe, "_grouped_steps", 0)
    declined0 = dict(getattr(moe, "_grouped_declined", {}) or {})
    x = torch.zeros(1, 1, moe.dim, dtype=dt, device=dev)
    with torch.no_grad():
        moe(x, ids)
    steps1 = getattr(moe, "_grouped_steps", 0)
    declined1 = dict(getattr(moe, "_grouped_declined", {}) or {})
    moe._grouped_steps = steps0
    if declined0:
        moe._grouped_declined = declined0
    elif getattr(moe, "_grouped_declined", None) is not None:
        moe._grouped_declined.clear()
    new = [k for k in declined1 if declined1[k] != declined0.get(k, 0)]
    if new:
        return f"the probe step DECLINED grouping: {new}"
    if steps1 != steps0 + 1:
        return (f"the probe step did not reach the grouped kernel "
                f"(_grouped_steps {steps0} -> {steps1}, expected +1)")
    return None


def _moe_refusal(L, dev, dt, ids, mod):
    """Why this layer's routed MoE must NOT be captured, or None. See the module docstring.

    Every branch here is a way a graph could replay STALE ROUTING or read freed memory, and each one
    is cheaper to refuse than to detect afterwards -- a wrong-expert token is a plausible token."""
    if not grouped_at_one_token(mod):
        import v4_levers
        return (f"a single-token step does not reach the grouped MoE (live chain: "
                f"{'>'.join(v4_levers.moe_chain(mod))}) — capturing a `.tolist()` dispatch would "
                f"freeze the capture step's expert set into every replay")
    bank = getattr(L.ffn, "_grouped_bank", None)
    if not bank:
        return ("this layer has no load-time expert bank (bank_layout declined or never ran), so the "
                "first grouped step would STACK one — an allocation out of the graph's private pool "
                "that a later capture in that pool may hand to something else")
    import v4_moe_grouped
    if int(getattr(v4_moe_grouped, "_WORLD_SIZE", 1) or 1) > 1:
        return "world_size > 1: the reference all-reduces the routed sum, and the grouped path cannot"
    if ids is None:
        return "no token ids: the first n_hash_layers route on tid2eid[input_ids]"
    return moe_probe(L, dev, dt, ids)


def moe_graph_coverage(stage):
    """(captured, refused, undecided) MoE-in-graph layers on `stage`. The lever audit's observation.

    Undecided is the honest third state and not a rounding error: capture is LAZY, so before the first
    decode token every layer has ASKED and none has been judged. A stage that reports `0 captured` with
    everything undecided has not run yet; one that reports it with everything refused is the finding.

    Counts only the layers that ASKED (`moe_requested`), which survives the demotion a refusal makes."""
    bgs = [bg for bg in (getattr(stage, "_block_graphs", None) or [])
           if getattr(bg, "moe_requested", False)]
    got = [bg.moe_in_graph for bg in bgs]
    return sum(g is True for g in got), sum(g is False for g in got), sum(g is None for g in got)


# ── a whole decode block, capture-safe, from static buffers ─────────────────────────────────────────

def block_pre_cs(R, L, h, pos, win_topk, comp_topk, arange_w, compress, read_w):
    """Block.forward up to the MoE input: hc_pre/attn_norm, attention, hc_post, hc_pre/ffn_norm.

    Returns (ffn_in, residual, post, comb) -- everything the MoE and the trailing hc_post consume. The
    split is where a graph must break for `moe_mode="eager"`: the routed MoE runs between here and
    block_post_cs, and this half plus that half is the whole attention-core dispatch, graphed."""
    residual = h
    x, post, comb = L.hc_pre(h, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base)
    x = L.attn_norm(x)
    x = attn_decode_cs(R, L.attn, x, pos, win_topk, comp_topk, arange_w, compress, read_w)
    x = L.hc_post(x, residual, post, comb)
    residual = x
    x, post, comb = L.hc_pre(x, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base)
    x = L.ffn_norm(x)
    return x, residual, post, comb


def block_post_cs(L, ffn_out, residual, post, comb):
    """Block.forward's trailing ffn hc_post: expand the MoE output back to hc_mult streams."""
    return L.hc_post(ffn_out, residual, post, comb)


def block_decode_cs(R, L, h, ids, pos, win_topk, comp_topk, arange_w, compress, moe, read_w):
    """One Block's decode step, capture-safe -- Block.forward with attn_decode_cs and a chosen MoE.

    Same structure as model.py Block.forward: hc_pre/attn_norm, attention, hc_post, hc_pre/ffn_norm,
    ffn, hc_post. `moe(L, x, ids)` is real_moe for the parity oracle (bit-exact vs the reference block)
    or moe_stub for a graph-safe whole-layer capture. Everything is a pure function of (h, pos, the two
    topk buffers) plus the layer's constant parameters and its per-stage KV/compressor state."""
    ffn_in, residual, post, comb = block_pre_cs(R, L, h, pos, win_topk, comp_topk, arange_w, compress,
                                                read_w)
    ffn_out = moe(L, ffn_in, ids)
    return block_post_cs(L, ffn_out, residual, post, comb)


# ── the reference's own per-step index lists, built eagerly into a static buffer ─────────────────────

def build_win_topk(M, win, start_pos):
    """get_window_topk_idxs for one decode step -> [1,1,win] int32, on the default (serve) device."""
    return M.get_window_topk_idxs(win, 1, 1, start_pos)


def build_comp_topk(M, ratio, start_pos, offset, maxw):
    """get_compress_topk_idxs for a non-indexer compress layer, padded to a fixed width with -1.

    The reference's compress index list for decode is arange(0, (start_pos+1)//ratio) + offset -- a
    width that GROWS with position. Padded to `maxw = max_seq_len // ratio` with -1 (a sparse_attn
    no-op) it is a fixed-shape static-buffer feed whose valid prefix is byte-identical to the
    reference's."""
    idx = M.get_compress_topk_idxs(ratio, 1, 1, start_pos, offset)   # [1,1,K], K grows
    k = idx.size(-1)
    out = idx.new_full((1, 1, maxw), -1)
    if k:
        out[..., :k] = idx
    return out.int()


# ── whole-layer CUDA graph: capture a decode block, replay it per step ───────────────────────────────

def _layer_state(L):
    """The per-layer buffers a decode step mutates -- what warmup dirties and a capture must not keep.

    Same set _snapshot/reset reason about: the attn kv_cache (window ring + compressed region), the
    Indexer's kv_cache, and every Compressor's kv_state/-inf score_state. Cloned for a snapshot,
    written back for a restore -- so a warm-up run that JITs the tilelang kernels leaves no trace on
    the real decode state the first replay starts from."""
    bufs = [L.attn.kv_cache]
    A = L.attn
    if A.compress_ratio:
        bufs += [A.compressor.kv_state, A.compressor.score_state]
        if A.indexer is not None:
            bufs += [A.indexer.kv_cache, A.indexer.compressor.kv_state, A.indexer.compressor.score_state]
    return bufs


class WholeBlockGraphs:
    """One decode Block captured as CUDA graph(s): the whole layer (attn + islands + MoE) per replay.

    `moe_mode`:
      "stub"   the entire block is ONE graph -- attn core, islands, and moe_stub (a fixed expert set).
               Proves the whole-layer capture mechanism and its ceiling end to end; not the routed
               model's output, so it composes with a graph-safe grouped MoE, not with routing.
      "eager"  attn + islands are graphed and the real routed MoE runs EAGER between two graphs
               (`g_pre` up to the ffn input, `g_post` the ffn hc_post). Real-serving-safe TODAY: the
               output is the reference's, bit-exact, with the attention core's dispatch folded away.
      "graph"  the entire block is ONE graph WITH the real routed MoE inside it (V4_MOE_IN_GRAPH=1).
               A REQUEST, not a promise: `_moe_refusal` judges the layer on the first decode step and
               demotes it to "eager" if its MoE is not provably sync-free. See the module docstring.

    A compress-ratio layer captures a compress and a no-compress variant of every graphed region that
    contains the Compressor (g_pre / the whole block); the replay picks by (start_pos+1) % ratio. A
    pure sliding-window layer captures one. Capture is lazy on the first decode step and never raises:
    a failed/over-budget capture drops the layer to a capture-safe eager block."""

    def __init__(self, L, stage, moe_mode="eager"):
        self.L = L
        self.st = stage
        self.moe_mode = moe_mode
        self.a = stage.args
        self.dev = stage.device
        self.dt = stage.dtype
        self.R = _Ref(stage._M)
        self.win = L.attn.window_size
        self.ratio = L.attn.compress_ratio
        self.has_indexer = bool(self.ratio) and L.attn.indexer is not None
        self.moe = moe_stub if moe_mode == "stub" else real_moe
        # `moe_requested` survives a demotion (which rewrites moe_mode), so the audit can still tell
        # "asked and was refused" from "never asked". `moe_in_graph` stays None until the first decode
        # step judges it — capture is lazy, so "has not run yet" must not read as "declined".
        self.moe_requested = moe_mode == "graph"
        self.moe_in_graph = None
        self.eager = False
        # The compressed cache this layer reads, and therefore what a bucket is clamped to.
        if self.has_indexer:
            self.maxw = L.attn.indexer.kv_cache.size(1)
        elif self.ratio:
            self.maxw = self.a.max_seq_len // self.ratio
        else:
            self.maxw = 0
        # static input buffers, shape-independent of the bucket (fixed addresses, copied into per replay)
        self.h_buf = torch.zeros(1, 1, self.a.hc_mult, self.a.dim, dtype=self.dt, device=self.dev)
        self.pos_buf = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.ids_buf = torch.zeros(1, 1, dtype=torch.long, device=self.dev)
        self.win_topk_buf = torch.zeros(1, 1, self.win, dtype=torch.int32, device=self.dev)
        self._bufs = {}       # bucket -> (comp_topk_buf | None, arange | None), allocated per rung
        self._graphs = {}     # (bucket, compress) -> graph + static io
        self._pool = None     # one shared graph memory pool across every variant
        self.ho = self.ffn_out_buf = self.g_post = None

    # -- per-step shape decisions, all HOST-side before a replay --

    def _plan(self, start_pos):
        """(bucket, compress) for this position -- the graph key. Both are host-known from start_pos."""
        if not self.ratio:
            return 0, False
        end_ratio = (start_pos + 1) // self.ratio
        floor = self.L.attn.indexer.index_topk if self.has_indexer else 0
        return bucket_width(end_ratio, self.maxw, floor), (start_pos + 1) % self.ratio == 0

    def _bufs_for(self, bucket):
        """The bucket-shaped static buffers: the compress index list and the mask's arange."""
        if bucket not in self._bufs:
            comp = (torch.full((1, 1, bucket), -1, dtype=torch.int32, device=self.dev)
                    if self.ratio and not self.has_indexer else None)
            ar = torch.arange(bucket, device=self.dev) if self.has_indexer else None
            self._bufs[bucket] = (comp, ar)
        return self._bufs[bucket]

    # -- the captured function(s) --

    def _block(self, compress, bucket):
        """block_decode_cs on THIS layer's static buffers, at this bucket. The capture target."""
        comp, ar = self._bufs_for(bucket)
        return block_decode_cs(self.R, self.L, self.h_buf, self.ids_buf, self.pos_buf,
                               self.win_topk_buf, comp, ar, compress, self.moe, bucket)

    def _pre(self, compress, bucket):
        comp, ar = self._bufs_for(bucket)
        return block_pre_cs(self.R, self.L, self.h_buf, self.pos_buf, self.win_topk_buf,
                            comp, ar, compress, bucket)

    def _capture_pos(self, compress, bucket):
        """A representative decode position to capture this (bucket, compress) variant at.

        The graph is position-generic (pos enters through pos_buf and is read at replay), so capture
        only needs a position that is IN BOUNDS for this variant:
          * the compress branch reads `freqs_cis[pos+1-ratio]`, so pos+1 must be divisible by ratio and
            pos >= ratio-1 -- exactly the positions it is ever replayed at;
          * end_ratio must fit the bucket, since the read is narrowed to it.
        Picks the largest position this bucket owns, so the capture exercises the widest mask it will
        ever see rather than an all-masked degenerate one."""
        if not self.ratio:
            return max(1, self.win)
        top = min(bucket, self.maxw)                       # the largest end_ratio this bucket serves
        p = top * self.ratio - 1                           # (p+1)//ratio == top, and (p+1) % ratio == 0
        if not compress:
            p = max(p - 1, 1)                              # one step earlier is a non-compress step
            if (p + 1) % self.ratio == 0:
                p = max(p - 1, 1)
        return p

    def _feed_capture(self, compress, bucket, ids=None):
        """Point the static buffers at a valid representative step for capturing this variant.

        `ids` seeds the token-id buffer with the step that triggered the capture. It changes nothing
        about the graph (routing is recomputed from `ids_buf` at every replay), but it means the
        warm-up runs on a real token rather than on the zero the constructor left, which is what a
        hash-routed layer's `tid2eid[0]` would otherwise be."""
        p = self._capture_pos(compress, bucket)
        self.pos_buf.fill_(p)
        if ids is not None:
            self.ids_buf.copy_(ids.view(1, 1))
        self.win_topk_buf.copy_(build_win_topk(self.R.M, self.win, p))
        comp, _ = self._bufs_for(bucket)
        if comp is not None:
            comp.copy_(build_comp_topk(self.R.M, self.ratio, p, self.win, bucket))

    def _warm_and_capture(self, fn, restore=True):
        """Warm up on a side stream (tilelang JITs and SYNCS on first call), then capture.

        The warm-up EXECUTES, so it advances the layer's KV/compressor state; it is bracketed by a
        snapshot/restore because real decode must start from the prefill state, never from a throwaway
        capture-position step. A capture itself runs nothing."""
        global _GRAPH_COUNT
        snap = [b.clone() for b in _layer_state(self.L)] if restore else None
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        if restore:
            for b, s in zip(_layer_state(self.L), snap):
                b.copy_(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._pool), torch.no_grad():
            out = fn()
        if self._pool is None:
            self._pool = g.pool()
        if restore:
            for b, s in zip(_layer_state(self.L), snap):
                b.copy_(s)
        _GRAPH_COUNT += 1
        return g, out

    def _resolve_moe_mode(self, ids):
        """Judge ONCE whether this layer's routed MoE may be captured, and demote it if not.

        Runs on the first decode step, before the first capture, because `_moe_refusal`'s last check
        is a live probe and there is nothing to probe before the stage is loaded. A refusal leaves the
        layer on the proven `moe_mode="eager"` path -- graphed attention core, eager routed MoE -- and
        SAYS SO: this lever's whole risk is being believed while it is not on."""
        global _MOE_GRAPHED, _MOE_REFUSED
        why = _moe_refusal(self.L, self.dev, self.dt, ids, self.R.M)
        self.moe_in_graph = why is None
        if why is None:
            _MOE_GRAPHED += 1
            return
        _MOE_REFUSED += 1
        self.moe_mode = "eager"
        print(f"[v4] V4_MOE_IN_GRAPH: layer {self.L.layer_id} keeps its routed MoE EAGER — {why}",
              flush=True)

    def _drop_moe_from_graph(self):
        """A layer that fell back to no graph at all has no MoE in a graph either — say so.

        The audit counts LIVE state, and `moe_in_graph` left True on a layer that never captured (an
        exhausted budget, a failed capture) is the same lie the whole lever registry exists to stop."""
        global _MOE_GRAPHED, _MOE_REFUSED
        if self.moe_in_graph:
            self.moe_in_graph, _MOE_GRAPHED, _MOE_REFUSED = False, _MOE_GRAPHED - 1, _MOE_REFUSED + 1
        self.moe_mode = "eager" if self.moe_mode == "graph" else self.moe_mode

    def _capture(self, key, ids=None):
        """Capture the graph(s) for one (bucket, compress) key. moe_eager captures g_pre (+ g_post once)."""
        bucket, compress = key
        # ALWAYS point the static buffers at a valid position for this variant FIRST: the warm-up runs
        # for real, and a compress variant captured at the zero-filled pos_buf would read
        # `freqs_cis[1 - ratio]`, i.e. a negative row.
        self._feed_capture(compress, bucket, ids)
        if self.moe_mode != "eager":
            g, out = self._warm_and_capture(lambda: self._block(compress, bucket))
            return {"graph": g, "out": out}
        outer = [b.clone() for b in _layer_state(self.L)]
        if self.ho is None:                                 # shape probe (EXECUTES; restored below)
            with torch.no_grad():
                ex = self._pre(False, bucket)
            self.ho = [torch.zeros_like(t) for t in ex]     # ffn_in, residual, post, comb
            self.ffn_out_buf = torch.zeros_like(ex[0])

        def pre_fn():
            for buf, t in zip(self.ho, self._pre(compress, bucket)):
                buf.copy_(t)

        g, _ = self._warm_and_capture(pre_fn)
        entry = {"graph": g}
        if self.g_post is None:
            # g_post is bucket- AND compress-independent (it only reads the hand-off buffers), so it is
            # captured once. Prime those buffers with real data first so it captures over non-garbage.
            with torch.no_grad():
                self._feed_capture(compress, bucket)
                g.replay()
                self.ffn_out_buf.copy_(real_moe(self.L, self.ho[0], self.ids_buf))
            gp, out = self._warm_and_capture(
                lambda: block_post_cs(self.L, self.ffn_out_buf, self.ho[1], self.ho[2], self.ho[3]),
                restore=False)
            self.g_post = {"graph": gp, "out": out}
        for b, s in zip(_layer_state(self.L), outer):
            b.copy_(s)
        return entry

    # -- the serve-path entry --

    def run(self, h, ids, start_pos):
        """One graphed decode step for this Block, or a capture-safe eager fallback. Never raises.

        A position whose bucket has no graph yet captures one HERE, lazily -- so a long decode crosses
        a rung at most `len(INDEXER_BUCKETS)` times and every other step is a pure replay. Past
        V4_GRAPH_MAX the layer goes permanently eager, counted and logged (m25's budget discipline).

        `ids` IS LOAD-BEARING under moe_mode="graph" and only there: the first `n_hash_layers` route
        their MoE on `tid2eid[input_ids]`, so a graph that never had the step's token copied into
        `ids_buf` would replay the CAPTURE step's experts on every token after it -- plausible output,
        wrong model. The other modes run the MoE outside the graph and pass `ids` to it directly, so
        None there is merely the score-routed case and stays legal."""
        global _GRAPH_SKIPPED
        if self.eager:
            return self._eager(h, ids, start_pos)
        if self.moe_requested and self.moe_in_graph is None:
            self._resolve_moe_mode(ids)
        if self.moe_mode == "graph" and ids is None:
            raise RuntimeError(
                f"v4 whole-layer graph: layer {self.L.layer_id} captured its routed MoE "
                f"(V4_MOE_IN_GRAPH=1) and was handed no token ids — a hash-routed layer would replay "
                f"the capture step's expert set. Carry the ids with the payload.")
        key = self._plan(start_pos)
        entry = self._graphs.get(key)
        if entry is None:
            need = 2 if (self.moe_mode == "eager" and self.g_post is None) else 1
            if _GRAPH_COUNT + need > V4_GRAPH_MAX:
                self.eager, _GRAPH_SKIPPED = True, _GRAPH_SKIPPED + need
                self._drop_moe_from_graph()
                print(f"[v4] graph budget V4_GRAPH_MAX={V4_GRAPH_MAX} spent — layer "
                      f"{self.L.layer_id} stays eager", flush=True)
                return self._eager(h, ids, start_pos)
            try:
                entry = self._graphs[key] = self._capture(key, ids)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                torch.cuda.synchronize()
                self.eager, self._graphs, _GRAPH_SKIPPED = True, {}, _GRAPH_SKIPPED + need
                self._drop_moe_from_graph()
                print(f"[v4] whole-layer capture failed for layer {self.L.layer_id} at bucket "
                      f"{key[0]}: {type(e).__name__}: {e} — layer stays eager", flush=True)
                return self._eager(h, ids, start_pos)
        self._feed(h, ids, start_pos, key[0])
        entry["graph"].replay()
        if self.moe_mode != "eager":
            return entry["out"].clone()
        self.ffn_out_buf.copy_(real_moe(self.L, self.ho[0], ids))    # real routed MoE, eager
        self.g_post["graph"].replay()
        return self.g_post["out"].clone()

    def _feed(self, h, ids, start_pos, bucket):
        """Copy this step's inputs into the static buffers the graph replays over."""
        self.h_buf.copy_(h)
        self.pos_buf.fill_(start_pos)
        if ids is not None:
            self.ids_buf.copy_(ids.view(1, 1))
        self.win_topk_buf.copy_(build_win_topk(self.R.M, self.win, start_pos))
        comp, _ = self._bufs_for(bucket)
        if comp is not None:
            comp.copy_(build_comp_topk(self.R.M, self.ratio, start_pos, self.win, bucket))

    def _eager(self, h, ids, start_pos):
        """The capture-safe block run WITHOUT a graph -- the fallback, and the TIER-1 grading twin.

        Same math, same bucket decisions, no capture: `graphed == this` is the hard bit-exactness bar
        (research/graph_aux_check.py's `eager_manual`, m25's precedent), because a graph that replays
        the same kernels on the same bytes has no licence to differ from it by even one ulp."""
        bucket, compress = self._plan(start_pos)
        pos = torch.tensor([start_pos], dtype=torch.long, device=self.dev)
        win_topk = build_win_topk(self.R.M, self.win, start_pos)
        comp_topk = (build_comp_topk(self.R.M, self.ratio, start_pos, self.win, bucket)
                     if self.ratio and not self.has_indexer else None)
        ar = torch.arange(bucket, device=self.dev) if self.has_indexer else None
        return block_decode_cs(self.R, self.L, h, ids, pos, win_topk, comp_topk, ar,
                               compress, self.moe, bucket)
