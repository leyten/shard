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

WHAT STAYS EAGER, AND WHY THAT IS HONEST.  The MoE routes per token through `bincount(...).tolist()`
(a host sync) and a data-dependent expert loop, so it is un-graphable as written -- the sibling
grouped-fp4-MoE kernel (phase0/v4_moe_grouped.py) is what makes it capture-safe. Until that composes,
`moe_mode="stub"` runs a FIXED expert set (shared + a constant routed pair, the real decode
activation count) so the WHOLE-LAYER graph mechanism and its timing are provable end to end, and
`moe_mode="eager"` graphs attn+islands and leaves the real routed MoE eager between two graphs -- the
honest intermediate. Neither stub number is passed off as the routed model's.

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


def bucket_width(need, maxw):
    """Smallest INDEXER_BUCKETS rung >= `need`, clamped to `maxw`. m25_stage._bucket, per-cache."""
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


def select_compress_topk(index_score, end_ratio, index_topk, offset, arange_w):
    """Bucket-width masked selection of the compressed slots to attend: the #2 rewrite, in one place.

    Reference (Indexer.forward, decode): `index_score[..., :end_pos//ratio].topk(min(index_topk,
    end_pos//ratio))[1] + offset` -- a GROWING read width and a GROWING k, neither capturable. Here the
    scores arrive at a BUCKET width >= end_ratio, columns past `end_ratio` are masked to -inf, k is the
    fixed `index_topk`, and any pick that lands on a masked (future) column becomes -1, which
    sparse_attn treats as "no position".

    THE STABLE SORT IS WHAT MAKES BUCKETING LEGAL, and that is the whole reason it is here rather than
    a `topk`. v4_stage's own `_chunk_indexer` says it for the chunked path: "Masking a common-width
    score to fake the shorter reads would change which slots a topk near-tie picks" -- torch.topk's
    tie-break is an artifact of its partition, so with topk the answer would depend on WHICH BUCKET a
    position happened to land in, i.e. on capture history. Selecting by (score DESC, index ASC) is
    width-independent by construction, so a position gets the same slots at bucket 64 as at 16384 and
    a rung crossing is invisible. Proven in tests/test_v4_whole_layer.py."""
    valid = (arange_w < end_ratio).view(1, 1, -1)
    index_score = index_score.masked_fill(~valid, float("-inf"))
    # DETERMINISTIC selection: top-k by (score DESC, column index ASC), via a STABLE sort.
    #
    # `topk` cannot be used here and this is the one place the fixed-width read is NOT a free
    # rewrite of the reference. index_score is riddled with EXACT ties -- `relu_()` floors every
    # negative score to a hard 0.0, and bf16 collides -- and torch.topk's tie-break is an artifact of
    # its partition, so it depends on the array WIDTH (the reference's end_pos//ratio vs this fixed
    # max width) and on the CPU thread count. Two runs of the SAME reference at different
    # OMP_NUM_THREADS can pick different members of a tie, and a fixed-width read picks a different
    # one again. A stable descending sort pins it: among equal scores the lowest column index wins,
    # on every width, thread count, and device.
    #
    # So this is deterministic where the reference is not, and it agrees with the reference on every
    # tie-FREE selection (a -inf column never outranks a finite one, so the top-k finite columns are
    # the same set). Where the reference's own tie-break is ambiguous it may select a different
    # column of IDENTICAL score -- see tests/test_v4_whole_layer.py, which proves the selected score
    # multiset always matches, and docs/receipts/v4-whole-layer-graph-20260801.json's `ties` risk.
    order = index_score.sort(dim=-1, descending=True, stable=True)[1]
    topk_idxs = order[..., :index_topk]
    topk_idxs = torch.where(topk_idxs < end_ratio, topk_idxs + offset, topk_idxs.new_full((), -1))
    return topk_idxs.int()


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
    """The reference routed MoE (eager). Used as the `moe_eager` intermediate and the parity oracle."""
    return L.ffn(x, ids)


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
        return bucket_width(end_ratio, self.maxw), (start_pos + 1) % self.ratio == 0

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

    def _feed_capture(self, compress, bucket):
        """Point the static buffers at a valid representative step for capturing this variant."""
        p = self._capture_pos(compress, bucket)
        self.pos_buf.fill_(p)
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

    def _capture(self, key):
        """Capture the graph(s) for one (bucket, compress) key. moe_eager captures g_pre (+ g_post once)."""
        bucket, compress = key
        # ALWAYS point the static buffers at a valid position for this variant FIRST: the warm-up runs
        # for real, and a compress variant captured at the zero-filled pos_buf would read
        # `freqs_cis[1 - ratio]`, i.e. a negative row.
        self._feed_capture(compress, bucket)
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
        V4_GRAPH_MAX the layer goes permanently eager, counted and logged (m25's budget discipline)."""
        global _GRAPH_SKIPPED
        if self.eager:
            return self._eager(h, ids, start_pos)
        key = self._plan(start_pos)
        entry = self._graphs.get(key)
        if entry is None:
            need = 2 if (self.moe_mode == "eager" and self.g_post is None) else 1
            if _GRAPH_COUNT + need > V4_GRAPH_MAX:
                self.eager, _GRAPH_SKIPPED = True, _GRAPH_SKIPPED + need
                print(f"[v4] graph budget V4_GRAPH_MAX={V4_GRAPH_MAX} spent — layer "
                      f"{self.L.layer_id} stays eager", flush=True)
                return self._eager(h, ids, start_pos)
            try:
                entry = self._graphs[key] = self._capture(key)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                torch.cuda.synchronize()
                self.eager, self._graphs, _GRAPH_SKIPPED = True, {}, _GRAPH_SKIPPED + need
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
