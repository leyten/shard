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

  #2 FIXED MAX-WIDTH + MASK + A WIDTH-INVARIANT TIE-BREAK.  The Indexer's einsum reads the WHOLE
     `kv_cache` (a fixed [., MAXW, .]) and masks scores at positions >= end_pos//ratio to -inf; the
     selection takes a FIXED k = index_topk and pads unfilled slots with -1. sparse_attn treats a -1
     index as "no position" (it zeroes that KV row and pushes the score to -inf), and because the
     kernel walks topk in FIXED 64-wide blocks, an all-masked block rescales by exp(0)=1 and adds 0 --
     so the -1 padding is bitwise free. THE MASK IS NOT ENOUGH ON ITS OWN, THOUGH: `Tensor.topk`'s
     tie order is an unspecified artifact of its selection algorithm and is NOT invariant to array
     length, and `index_score` is bf16 behind a `relu_()` that floors negatives to a hard 0.0, so
     exact ties at the k-th rank are common and a masked wide topk can select a DIFFERENT SET than
     the reference's narrow one. The selection therefore uses an explicit (value DESC, index ASC)
     total order (_select_topk_width_invariant) whose result is identical at ANY read width >=
     end_pos//ratio -- which is also what makes BUCKETING the read a free cost lever rather than a
     correctness change. The window/compress index LISTS (get_window_topk_idxs /
     get_compress_topk_idxs) are the reference's own, built eagerly per step into a static buffer and
     copied in before replay: they are a handful of tiny launches, not the GEMM-and-attention bulk
     the graph is for, and building them in-graph would mean transcribing their per-position branch
     structure for no dispatch saving.

     TIER 1 vs TIER 2 -- WHAT "BIT-EXACT" MEANS HERE, EXACTLY.  Tier 1 (hard, torch.equal): this
     path is a deterministic function of (h, pos, state) that does not depend on the padded read
     width, and a graph of it replays byte-identically to running it eager. Tier 2 (named, bounded):
     against the VENDORED reference it is bit-identical at every position where the reference's own
     top-k is well defined, and may pick a different compressed slot only where the scores TIE at the
     k-th rank -- a case in which the reference's answer is torch's arbitrary order, not the model's
     intent, and already differs between the CPU and CUDA topk backends. That is a real, if small,
     accuracy caveat and not a rounding difference; tests/test_v4_whole_layer.py counts the ties and
     reports the first step one bites.

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
honest intermediate. Neither stub number is passed off as the routed model's; the bit-exact bar is the
attention core (proven vs the reference) and the graph composition (proven vs the same eager stub).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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

    It does NOT reproduce torch's tie order, and cannot: see the module docstring's TIER 2 note."""
    s = score.masked_fill(~valid, float("-inf"))
    kth = s.topk(k, dim=-1).values.narrow(-1, k - 1, 1)          # k-th largest value, width-invariant
    gt = s > kth                                                 # strictly inside the cut
    eq = (s == kth) & valid                                      # tied exactly at the cut
    need = k - gt.sum(-1, keepdim=True)                          # how many tied columns to admit
    take = gt | (eq & (eq.cumsum(-1) <= need))                   # cumsum = rank among ties, index ASC
    w = arange_w.size(-1)
    key = torch.where(take, w - arange_w, arange_w.new_full((), -1))   # distinct, DESC in index
    pick = key.topk(k, dim=-1).indices                           # -> ascending index, no ties left
    return pick, take.gather(-1, pick)


def _indexer_decode_cs(R, I, x, qr, pos, end_ratio, freqs_row, offset, arange_maxw, compress):
    """Indexer.forward's decode branch with the GROWING read/topk made fixed-width + masked.

    Reference (start_pos>0): scores q against `kv_cache[:, :end_pos//ratio]`, `topk(min(index_topk,
    end_pos//ratio))`, `+= offset`. Here: score against the whole `kv_cache` (fixed MAXW), mask
    columns >= end_ratio to -inf, and select a FIXED index_topk of them with a tie-break that does
    not depend on the read width (_select_topk_width_invariant), padding the unfilled slots with -1.
    A -1 is a sparse_attn no-op -- the kernel walks topk in fixed 64-wide blocks, so an all-masked
    block rescales by exp(0)==1 and adds 0 (v4_kernels_cpu.sparse_attn emulates that blocking for
    the same reason) -- so the padded list feeds byte-identical attention.

    `offset` (= window_size on decode) shifts a compressed slot index into the attn kv_cache's
    compressed region, exactly as the reference's `topk_idxs += offset`."""
    n_local_heads, head_dim, rd = I.n_local_heads, I.head_dim, I.rope_head_dim
    index_topk = I.index_topk
    q = I.wq_b(qr)
    q = q.unflatten(-1, (n_local_heads, head_dim))
    R.apply_rotary_emb(q[..., -rd:], freqs_row)
    q = R.rotate_activation(q)
    R.fp4_act_quant(q, R.fp4_block_size, True)
    _compressor_decode_cs(R, I.compressor, x, pos, compress)
    weights = I.weights_proj(x) * (I.softmax_scale * I.n_heads ** -0.5)
    # `arange_maxw`'s length IS the read width: pass a full-width arange for the shipped fixed-max
    # read, or a bucket-sized one to read only as far as the position needs. The selection is
    # width-invariant, so that choice is pure cost -- but the bucket must hold index_topk slots.
    read_w = arange_maxw.size(-1)
    index_score = torch.einsum("bshd,btd->bsht", q, I.kv_cache.narrow(0, 0, 1).narrow(1, 0, read_w))
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    arange = arange_maxw.view(1, 1, -1)
    valid = arange < end_ratio
    pick, kept = _select_topk_width_invariant(index_score, valid, index_topk, arange)
    topk_idxs = torch.where(kept, pick + offset, pick.new_full((), -1))
    return topk_idxs.int()


def attn_decode_cs(R, A, x, pos, win_topk, comp_topk, arange_maxw, compress):
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
                                                    win, arange_maxw, compress)
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

def block_pre_cs(R, L, h, pos, win_topk, comp_topk, arange_maxw, compress):
    """Block.forward up to the MoE input: hc_pre/attn_norm, attention, hc_post, hc_pre/ffn_norm.

    Returns (ffn_in, residual, post, comb) -- everything the MoE and the trailing hc_post consume. The
    split is where a graph must break for `moe_mode="eager"`: the routed MoE runs between here and
    block_post_cs, and this half plus that half is the whole attention-core dispatch, graphed."""
    residual = h
    x, post, comb = L.hc_pre(h, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base)
    x = L.attn_norm(x)
    x = attn_decode_cs(R, L.attn, x, pos, win_topk, comp_topk, arange_maxw, compress)
    x = L.hc_post(x, residual, post, comb)
    residual = x
    x, post, comb = L.hc_pre(x, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base)
    x = L.ffn_norm(x)
    return x, residual, post, comb


def block_post_cs(L, ffn_out, residual, post, comb):
    """Block.forward's trailing ffn hc_post: expand the MoE output back to hc_mult streams."""
    return L.hc_post(ffn_out, residual, post, comb)


def block_decode_cs(R, L, h, ids, pos, win_topk, comp_topk, arange_maxw, compress, moe):
    """One Block's decode step, capture-safe -- Block.forward with attn_decode_cs and a chosen MoE.

    Same structure as model.py Block.forward: hc_pre/attn_norm, attention, hc_post, hc_pre/ffn_norm,
    ffn, hc_post. `moe(L, x, ids)` is real_moe for the parity oracle (bit-exact vs the reference block)
    or moe_stub for a graph-safe whole-layer capture. Everything is a pure function of (h, pos, the two
    topk buffers) plus the layer's constant parameters and its per-stage KV/compressor state."""
    ffn_in, residual, post, comb = block_pre_cs(R, L, h, pos, win_topk, comp_topk, arange_maxw, compress)
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
        self.built = False
        # static input buffers (fixed addresses; copied into before every replay)
        self.h_buf = torch.zeros(1, 1, self.a.hc_mult, self.a.dim, dtype=self.dt, device=self.dev)
        self.pos_buf = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.ids_buf = torch.zeros(1, 1, dtype=torch.long, device=self.dev)
        self.win_topk_buf = torch.zeros(1, 1, self.win, dtype=torch.int32, device=self.dev)
        if self.ratio and not self.has_indexer:
            self.maxw_c = self.a.max_seq_len // self.ratio
            self.comp_topk_buf = torch.full((1, 1, self.maxw_c), -1, dtype=torch.int32, device=self.dev)
        else:
            self.maxw_c = 0
            self.comp_topk_buf = None
        self.arange = (torch.arange(L.attn.indexer.kv_cache.size(1), device=self.dev)
                       if self.has_indexer else None)
        self._graphs = {}     # compress(bool) -> dict of graph+static-io for this variant
        self._pool = None     # one shared graph memory pool across variants

    # -- the captured function(s) --

    def _block(self, compress):
        """block_decode_cs on THIS layer's static buffers, for a compress flag. The capture target."""
        return block_decode_cs(self.R, self.L, self.h_buf, self.ids_buf, self.pos_buf,
                               self.win_topk_buf, self.comp_topk_buf, self.arange, compress, self.moe)

    def _capture_pos(self, compress):
        """A valid representative decode position to capture this variant at.

        The graph is position-generic (pos enters through pos_buf, read at replay), so capture only
        needs an in-bounds position -- but the two variants have different in-bounds sets: the compress
        branch reads `freqs_cis[pos+1-ratio]`, so its position must be a real compress step
        (pos+1 divisible by ratio) AND >= ratio-1, which is exactly the set of positions it is ever
        replayed at. The no-compress branch never touches that row, so any decode pos works; win keeps
        it in the window's steady state and away from a compress boundary."""
        if compress:
            return 2 * self.ratio - 1                      # (2r) % r == 0 and 2r-1 >= r-1
        p = max(1, self.win)
        if self.ratio and (p + 1) % self.ratio == 0:
            p += 1
        return p

    def _feed_capture(self, compress):
        """Set the static buffers to a valid representative step for capturing `compress`'s variant."""
        p = self._capture_pos(compress)
        self.pos_buf.fill_(p)
        self.win_topk_buf.copy_(build_win_topk(self.R.M, self.win, p))
        if self.comp_topk_buf is not None:
            self.comp_topk_buf.copy_(build_comp_topk(self.R.M, self.ratio, p, self.win, self.maxw_c))

    def _capture_one(self, compress):
        """Warm up (JIT the tilelang kernels), snapshot state, capture, restore. Never executes state."""
        self._feed_capture(compress)
        snap = [b.clone() for b in _layer_state(self.L)]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                self._block(compress)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        for b, s in zip(_layer_state(self.L), snap):
            b.copy_(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._pool), torch.no_grad():
            out = self._block(compress)
        if self._pool is None:
            self._pool = g.pool()
        for b, s in zip(_layer_state(self.L), snap):
            b.copy_(s)
        return {"graph": g, "out": out}

    def _build(self):
        if self.moe_mode == "eager":
            self._build_moe_eager()
        else:
            variants = [False, True] if self.ratio else [False]
            for c in variants:
                self._graphs[c] = self._capture_one(c)
        self.built = True

    # -- moe_eager: two graphs (attn+islands) around the real routed MoE --

    def _pre(self, compress):
        return block_pre_cs(self.R, self.L, self.h_buf, self.pos_buf, self.win_topk_buf,
                            self.comp_topk_buf, self.arange, compress)

    def _build_moe_eager(self):
        """Capture g_pre (compress variants) and g_post, with the real MoE eager between at replay.

        Both g_pre variants write their four outputs (ffn_in, residual, post, comb) into ONE set of
        hand-off buffers, so the single g_post -- which is compress-independent -- reads a fixed
        address whichever variant ran. g_post's other input, the eager MoE's output, arrives through
        ffn_out_buf."""
        # Everything here that EXECUTES (the shape probe and the priming replay, unlike a capture)
        # advances the layer's KV/compressor state, so the whole build is bracketed by snapshot/restore:
        # real decode must start from the prefill state, not from a throwaway capture-position step.
        outer_snap = [b.clone() for b in _layer_state(self.L)]
        self._feed_capture(False)
        with torch.no_grad():
            ex = self._pre(False)                                        # shape probe (mutates state)
        self.ho = [torch.zeros_like(t) for t in ex]                      # ffn_in, residual, post, comb
        self.ffn_out_buf = torch.zeros_like(ex[0])
        variants = [False, True] if self.ratio else [False]
        for c in variants:
            self._graphs[c] = self._capture_pre_one(c)
        # Prime the hand-off + ffn_out buffers with valid data so g_post captures over non-garbage.
        with torch.no_grad():
            self._feed_capture(False)
            self._graphs[False]["graph"].replay()
            self.ffn_out_buf.copy_(real_moe(self.L, self.ho[0], self.ids_buf))
        self.g_post = self._capture_post()
        for b, s in zip(_layer_state(self.L), outer_snap):
            b.copy_(s)

    def _capture_pre_one(self, compress):
        """Capture g_pre for one compress variant: run block_pre_cs, store its outputs in `self.ho`."""
        self._feed_capture(compress)
        snap = [b.clone() for b in _layer_state(self.L)]

        def pre_fn():
            outs = self._pre(compress)
            for buf, t in zip(self.ho, outs):
                buf.copy_(t)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                pre_fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        for b, s in zip(_layer_state(self.L), snap):
            b.copy_(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._pool), torch.no_grad():
            pre_fn()
        if self._pool is None:
            self._pool = g.pool()
        for b, s in zip(_layer_state(self.L), snap):
            b.copy_(s)
        return {"graph": g}

    def _capture_post(self):
        """Capture g_post: block_post_cs(ffn_out_buf, *hand-off). Reads static buffers, mutates nothing."""
        def post_fn():
            return block_post_cs(self.L, self.ffn_out_buf, self.ho[1], self.ho[2], self.ho[3])

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                post_fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._pool), torch.no_grad():
            out = post_fn()
        return {"graph": g, "out": out}

    def _run_moe_eager(self, h, ids, start_pos):
        compress = bool(self.ratio) and (start_pos + 1) % self.ratio == 0
        self._feed(h, ids, start_pos)
        self._graphs[compress]["graph"].replay()        # writes ffn_in + residual/post/comb hand-offs
        self.ffn_out_buf.copy_(real_moe(self.L, self.ho[0], ids))   # real routed MoE, eager
        self.g_post["graph"].replay()
        return self.g_post["out"].clone()

    # -- the serve-path entry --

    def run(self, h, ids, start_pos):
        """One graphed decode step for this Block, or a capture-safe eager fallback. Never raises."""
        if self.eager:
            return self._eager(h, ids, start_pos)
        if not self.built:
            try:
                self._build()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                torch.cuda.synchronize()
                self.eager, self._graphs = True, {}
                print(f"[v4] whole-layer capture failed for layer {self.L.layer_id}: "
                      f"{type(e).__name__}: {e} — layer stays eager", flush=True)
                return self._eager(h, ids, start_pos)
        if self.moe_mode == "eager":
            return self._run_moe_eager(h, ids, start_pos)
        compress = bool(self.ratio) and (start_pos + 1) % self.ratio == 0
        self._feed(h, ids, start_pos)
        v = self._graphs[compress]
        v["graph"].replay()
        return v["out"].clone()

    def _feed(self, h, ids, start_pos):
        """Copy this step's inputs into the static buffers the graph replays over."""
        self.h_buf.copy_(h)
        self.pos_buf.fill_(start_pos)
        if ids is not None:
            self.ids_buf.copy_(ids.view(1, 1))
        self.win_topk_buf.copy_(build_win_topk(self.R.M, self.win, start_pos))
        if self.comp_topk_buf is not None:
            self.comp_topk_buf.copy_(build_comp_topk(self.R.M, self.ratio, start_pos, self.win, self.maxw_c))

    def _eager(self, h, ids, start_pos):
        """The capture-safe block run WITHOUT a graph -- the fallback and the parity oracle's twin."""
        compress = bool(self.ratio) and (start_pos + 1) % self.ratio == 0
        pos = torch.tensor([start_pos], dtype=torch.long, device=self.dev)
        win_topk = build_win_topk(self.R.M, self.win, start_pos)
        comp_topk = (build_comp_topk(self.R.M, self.ratio, start_pos, self.win, self.maxw_c)
                     if self.comp_topk_buf is not None else None)
        return block_decode_cs(self.R, self.L, h, ids, pos, win_topk, comp_topk, self.arange,
                               compress, self.moe)
