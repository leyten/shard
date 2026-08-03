"""DeepSeek-V4-Flash pipeline stage: one contiguous layer block, driven through shard's contract.

The V4 analogue of k3_stage.py / m25_stage.py. Step 1 (v4_ref_cpu) built the oracle -- the whole
reference Transformer in one process, on CPU, at toy dimensions. This is the thing that oracle
grades: a layer range [lo, hi) instantiated from DeepSeek's OWN `Block` and driven so that N stages
chained over a wire produce, token for token, what the single-process model produces.

WHAT MAKES V4 DIFFERENT FROM EVERY STAGE WE HAVE SHIPPED

1. THE INTER-STAGE PAYLOAD IS FOUR HIDDEN STATES, NOT ONE.
   Hyper-Connections. `Transformer.forward` expands the embedding to `h.unsqueeze(2).repeat(1,1,4,1)`
   (model.py:916) and every Block takes AND returns `[b, s, hc_mult=4, dim]`; the four streams are
   mixed per layer by a Sinkhorn-normalized combination matrix (`hc_pre`/`hc_post`) and collapse to
   one only at `hc_head`, after the last layer. So a boundary carries 4x dim, not dim -- 32 KiB per
   token at V4's real shape (dim 4096, bf16) against the 8 KiB a plain transformer would move. That
   is a property of the architecture, not of our split, and there is no legal place to collapse
   early: drop three streams at a hop and the stage still runs and is silently wrong.

2. TOKEN IDS TRAVEL WITH THE PAYLOAD.
   `Block.forward(x, start_pos, input_ids)` hands `input_ids` to the MoE Gate, and the first
   `n_hash_layers` (3 of 43 in the shipped config) route by `tid2eid[input_ids]` -- a hash table, not
   a score. A stage that forwards only `h` would route those layers off whatever ids happened to be
   in the Gate last, so `forward(h, ids, start_pos)` takes both. The ids are 8 bytes/token against
   the payload's 32 KiB; there is no reason to be clever about it, and
   tests/test_v4_stage.py::test_ids_reach_the_hash_gate is the red test that keeps them threaded.

3. THE REFERENCE'S DECODE BRANCH IS SINGLE-TOKEN-ONLY, AND SPECULATION IS MULTI-TOKEN.
   At `start_pos > 0` Attention writes `self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)`
   (model.py:535) and the Compressor's decode path indexes `kv_state[:bsz, start_pos % ratio]` the
   same way -- both are hard `seqlen == 1` assumptions, because DeepSeek's own loop only ever decodes
   one token. Step 4/5's verify pass sends a whole speculated chunk at `start_pos > 0`. So this
   stage LOOPS internally: prefill (`start_pos == 0`) goes through the reference's own multi-token
   branch untouched, and a chunk at `start_pos > 0` is replayed one position at a time and
   re-concatenated. That is exact, not an approximation -- the HC mixing is per-position and the only
   cross-position flow is through the KV/compressor state, which the loop advances in exactly the
   order sequential decode would. tests/test_v4_stage.py::test_chunk_loop_equals_stepwise is the
   proof, and it is the test that has to stay green for the verify path to mean anything.

   That loop is the REFERENCE PATH and stays the default. `V4_FAST_VERIFY=1` opts a stage into the
   CHUNKED path instead -- one pass per layer over all s positions -- which is the whole content of
   the "── the chunked verify path ──" section below, and the reason a g≈4 speculation can be worth
   more than 1.0x. See that section's header for what it does and what it costs.

THE MATH IS RENTED, NOT REWRITTEN (docs/MODEL_RUNTIME.md). This file instantiates DeepSeek's
`Block`, `RMSNorm`, `ParallelEmbedding` and `ParallelHead` and calls them; it reimplements no
attention, no compressor, no hyper-connection, and not one line of the `hc_head` collapse (which is
a METHOD ON Block taking the Transformer-level parameters as arguments -- so the tail borrows its
own last block to run it, exactly as `Transformer.forward:922` does). phase0/deepseek_v4_ref/ is the
vendored reference, byte-identical with its provenance; phase0/v4_kernels_cpu.py is the CPU stand-in
for its tilelang kernels, so all of this is provable without renting a GPU. Version skew and the
things a stage must do that a Transformer does for itself are absorbed HERE, in one auditable place.

THE ONE THING A STAGE MUST DO THAT `Transformer.__init__` USED TO DO FOR IT
model.py keeps `world_size, rank, default_dtype, scale_fmt, scale_dtype` as MODULE globals and
assigns them from ModelArgs inside `Transformer.__init__` (model.py:881). Every `Linear` reads
`default_dtype` at CONSTRUCTION and every `act_quant` call site reads `scale_fmt`/`scale_dtype` at
CALL time. A stage never builds a Transformer, so `_set_globals` does it -- mirroring that
constructor line for line -- before the first Block exists. Get it wrong and the stage quietly
builds bf16 weights against an fp8 checkpoint, or rounds activation scales one way while the oracle
rounds them the other. Because they are process-wide, two Stages with different ModelArgs in one
process is refused rather than silently letting the last constructor win.

PER-STAGE STATE, NEVER ON THE WIRE
All of it lives in the reference's own non-persistent buffers: `Attention.kv_cache` (a `window_size`
ring buffer followed by the compressed region), `Compressor.kv_state`/`score_state` (the fp32 decode
accumulators), and the Indexer's `kv_cache` plus its own Compressor's pair. Two of those are reached
through aliases bound LAZILY on first forward (`compressor.kv_cache = self.kv_cache[:, win:]`,
model.py:497), which is why `reset()` zeroes the underlying buffers IN PLACE rather than rebuilding
modules: the aliases are views, so they follow, and the weights survive.

WHAT IS A SEAM HERE AND NOT YET A FEATURE
  _spec       arm it and every forward at `start_pos > 0` checkpoints the rollback-able state before
              it is touched, into a position-keyed ring of the last W (`_spec_ckpts`); `_seek` then
              rewinds up to W deep (restore the covering checkpoint + replay the accepted prefix),
              which is what makes a rejected speculation safe to commit. Pipelined speculation streams
              s=1 frames back-to-back so 5-of-6 idle stages fill, and a rejection W frames downstream
              has to rewind across every boundary those frames crossed — see `_seek`/`_snapshot` for
              why crossing a `ratio`-block COMPRESSION boundary stays bit-exact at any depth ≤ W.
              `commit` drops checkpoints the ring has settled past.
  _dspark     arm it and the stage records `h.mean(dim=2)` after each owned `dspark_target_layer_id`,
              which is the drafter's input (`Transformer.forward:921`). Inert otherwise: the greedy
              path clones nothing.
  fp8 wire, batching, CUDA graphs      not in this step. m25's mechanisms port, but a 4-stream
              payload changes what is worth packing, and that is a measurement, not a guess.

  self-test:  python3 phase0/v4_stage.py --layers 0 4
"""
import argparse, glob, json, os, sys, torch
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors import safe_open

# The lever registry. Imported at module scope ON PURPOSE, and deliberately not guarded: a stage that
# cannot audit its own levers is a stage whose numbers mean nothing, so a deploy that forgets this
# file must fail at import -- loudly, in the launch log, before the ring forms -- rather than serve
# unaudited. v4_levers imports nothing from here at module scope, so this cannot cycle.
import v4_levers

# Nothing here reads the checkpoint at import time (k3_stage's rule): resolving lazily costs one
# memoized call and lets `import v4_stage` work on a box with no model on disk -- which is every box
# that runs the parity tests.
V4_DIR = os.environ.get("V4_DIR", "/root/v4")
dev = os.environ.get("V4_DEV", "cuda")
# The dtype every un-annotated parameter and buffer is CONSTRUCTED at -- generate.py's
# `torch.set_default_dtype(torch.bfloat16)`, which is what ParallelEmbedding's bare `torch.empty` and
# the kv_cache `torch.zeros` pick up. It is NOT the weight format: that comes from `args.dtype`
# (fp8/bf16) and `args.expert_dtype` (fp4) through the module globals below.
V4_DTYPE = os.environ.get("V4_DTYPE", "bfloat16")
# The chunked verify path (below). OFF by default: the per-token loop is the reference path and the
# thing everything else is graded against, so the fast one is opted into per stage, never inherited.
V4_FAST_VERIFY = bool(int(os.environ.get("V4_FAST_VERIFY", "0") or 0))
# How many chunk positions a fast stage reserves room for. It is the width of the scratch region
# appended to every `Attention.kv_cache` (see `_chunk_attention`), so it is paid in VRAM once per
# layer -- 16 rows against a 128-row window is noise, and a chunk wider than this simply falls back
# to the loop rather than reallocating mid-job.
V4_FAST_VERIFY_MAX = int(os.environ.get("V4_FAST_VERIFY_MAX", "16") or 16)

# CUDA graphs over a decode step, opt-in, default OFF (the default path stays byte-identical and the
# CPU parity suite never touches this). UNLIKE K3, a V4 layer CANNOT be captured whole: three of its
# pieces bake position or data into a graph and are wrong on replay --
#   * the attention core writes `kv_cache[:, start_pos % win]` (a ROTATING ring slot), reads a
#     COMPRESSED region whose valid width GROWS with position, and the Indexer's score einsum runs
#     over `kv_cache[:, :end_pos // ratio]` (a growing slice); a graph freezes all three at capture.
#   * the Compressor's `should_compress = (start_pos+1) % ratio == 0` is a per-position PYTHON branch,
#     and its compressed write slot `kv_cache[:, start_pos // ratio]` grows.
#   * the MoE picks its experts per token (`indices[0].tolist()` — a host sync even on the decode fast
#     path), so a captured graph runs ONE token's expert set.
# So the capture is PARTIAL: it graphs the POSITION- and DATA-INDEPENDENT islands the reference
# exposes as pure Block methods -- the two `hc_pre` (mix + Sinkhorn), the two `hc_post`, and the two
# attn/ffn RMSNorms -- and leaves `attn` and `ffn` eager between them (measured: ~68 of ~240
# launches/layer, the MoE half and the position-dependent attention core stay eager BY CONSTRUCTION).
# Bit-exact, not approximately: each island graph replays the reference's OWN kernels on operands fed
# through static buffers, so a correct capture is the same math on the same bytes, no reassociation.
#
# ISLAND MODE IS NOT THE ONLY MODE ANY MORE. v4_whole_layer_graph.py makes the attention core itself
# capture-safe (a fixed ring slot chosen at replay from a copied-in position, a bucketed indexer read,
# a width-invariant selection), so "cannot be captured whole" above describes the REFERENCE's decode
# branch as written, not the ceiling. The flag therefore takes a MODE, not a bool:
# "0"/off (default), "1"/"island" (hc_pre/hc_post/norm islands only, attn+MoE eager), "whole"/"eager"
# (the WHOLE decode layer -- the capture-safe attention core folded in too, real routed MoE eager
# between two graphs, bit-exact to the reference; v4_whole_layer_graph.py). See _graph_mode.
# AND THE MoE ITSELF IS CAPTURABLE NOW, on a layer that reaches the grouped fp4 kernel: `.tolist()`
# above describes the reference's dispatch and v4_moe_decode's, not v4_moe_grouped's, whose routing is
# device-side end to end. `V4_MOE_IN_GRAPH=1` (default OFF, on top of whole mode) folds it in, per
# layer, only where that is provable -- v4_whole_layer_graph's docstring has the audit and the refusal.
V4_CUDA_GRAPH = os.environ.get("V4_CUDA_GRAPH", "0")


def _graph_mode(v=None):
    """Resolve V4_CUDA_GRAPH (env string or a monkeypatched bool) to off/island/whole."""
    v = V4_CUDA_GRAPH if v is None else v
    if v in (True, "1", "island", "on"):
        return "island"
    if v in ("whole", "2", "eager"):
        return "whole"
    return "off"


# Every captured graph pins its own workspace pool; cap the set process-wide (3 graphs per layer).
# Past the cap a layer stays EAGER (counted, never a crash), exactly like K3's K3_GRAPH_MAX.
V4_GRAPH_MAX = int(os.environ.get("V4_GRAPH_MAX", "192"))
_GRAPH_COUNT = 0        # island graphs captured so far, across every Stage in this process
_GRAPH_SKIPPED = 0      # island graphs a layer skipped because the cap was hit or a capture failed

_REF = None
_ARGS = {}
_WM = {}
_HD = {}
_GLOBALS = None


# ── resolving the reference + the checkpoint ─────────────────────────────────────────────────────

def ref():
    """DeepSeek's reference model.py, with usable kernels installed around it (memoized).

    Delegates to v4_ref_cpu.load_ref() rather than repeating the import machinery: the ORDER there
    is load-bearing (v4_kernels_cpu.install() must run before `from kernel import ...` is resolved at
    model.py's module scope, and the inference dir has to be on sys.path for the tilelang backend
    where install() deliberately does nothing), and two copies of that would drift."""
    global _REF
    if _REF is None:
        import v4_ref_cpu
        _REF = v4_ref_cpu.load_ref()
    return _REF


def config(d=None):
    """The ModelArgs for a checkpoint dir, straight off its config.json (memoized per dir).

    generate.py:81 does exactly this -- the shipped config.json's keys ARE the dataclass's field
    names, deliberately. `max_batch_size`/`max_seq_len` are the two the serving path overrides."""
    d = d or V4_DIR
    if d not in _ARGS:
        with open(f"{d}/config.json") as f:
            _ARGS[d] = ref().ModelArgs(**json.load(f))
    return _ARGS[d]


def weight_map(d=None):
    """The converted checkpoint's tensor -> file map (memoized per dir).

    convert.py writes `model{rank}-mp{mp}.safetensors` and NO index json -- unlike an HF release
    there is nothing to read the map out of, so it is built by walking each shard's key list once.
    Cheap: safetensors headers are read without touching tensor data."""
    d = d or V4_DIR
    if d not in _WM:
        files = sorted(glob.glob(os.path.join(d, "model*-mp*.safetensors")))
        if not files:
            raise RuntimeError(
                f"v4: no model*-mp*.safetensors in {d!r} — this loader reads convert.py's OUTPUT "
                f"format, not an HF release. Run deepseek_v4_ref/inference/convert.py first.")
        wm = {}
        for f in files:
            with safe_open(f, "pt", device="cpu") as h:
                for n in h.keys():
                    wm[n] = os.path.basename(f)
        _WM[d] = wm
    return _WM[d]


def raw(n, d=None):
    """One tensor by name, off a cached safetensors handle. k3_stage.raw, per-dir."""
    d = d or V4_DIR
    s = weight_map(d)[n]
    key = (d, s)
    if key not in _HD:
        _HD[key] = safe_open(os.path.join(d, s), "pt", device="cpu")
    return _HD[key].get_tensor(n)


def _set_globals(M, args):
    """Assign model.py's module globals, mirroring `Transformer.__init__` (model.py:881-886).

    A Stage builds Blocks directly, so nothing else ever runs those five lines -- see this module's
    docstring for what silently breaks when they are left at their import-time defaults.

    They are PROCESS-WIDE, which is the trap: a second Stage built from different ModelArgs would
    rebind them under the first Stage's already-constructed weights. Refused, loudly. (Two stages
    from the SAME args in one process is the normal single-box multi-stage case and is fine.)"""
    global _GLOBALS
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if world_size != 1:
        raise RuntimeError(
            f"v4: world_size={world_size}. A Stage is PIPELINE parallelism — the reference's "
            f"tensor-parallel path all_reduces inside RowParallelLinear/MoE/Indexer against a "
            f"process group a stage does not own. Run one rank per stage.")
    new = (world_size, rank,
           torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16,
           "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt,
           torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32)
    if _GLOBALS is not None and _GLOBALS != new:
        raise RuntimeError(
            f"v4: model.py's globals are already {_GLOBALS} and this Stage's args want {new}. They "
            f"are MODULE globals set once by Transformer.__init__, and every Linear read the old "
            f"ones at construction — rebinding them now would leave the existing stage's weights "
            f"in a format nothing agrees on. One ModelArgs per process.")
    M.world_size, M.rank, M.default_dtype, M.scale_fmt, M.scale_dtype = new
    _GLOBALS = new
    return new


# ── the chunked verify path ──────────────────────────────────────────────────────────────────────
#
# WHAT IT BUYS. A verify chunk of s tokens costs, on the per-token loop, s TIMES a single-token
# traversal -- measured at 3.93x for s=6 on the first live 7x5090 ring, which is why g≈4 speculation
# netted ~1.0x end to end. Nothing about that is arithmetic: at s=1 every layer is dispatch- and
# weight-streaming-bound, so the SECOND token through a layer is nearly free if it rides the same
# pass.
#
# WHAT IT MEASURED, on one real V4 layer on a 5090 at ctx 1024, s = 6 (ms per call, chunk vs the
# loop vs a single token):
#
#   layer 41 (ratio 128), attention + HC only      loop 5.73x   chunk 1.38x    <- the mechanism
#   layer 40 (ratio 4 + indexer), attention + HC   loop 5.81x   chunk 1.92x
#   layer 41, whole layer                          loop 5.88x   chunk 4.08x
#   layer 40, whole layer                          loop 5.84x   chunk 3.94x    <- 1.48x end to end
#
# So the chunk mechanics deliver what they promised -- six positions of attention and
# hyper-connections for 1.4 single-token traversals -- and the MoE eats most of it back. That is not
# a defect in this path: a 256-expert MoE routing 6 tokens to 6 experts each touches up to 36 experts
# instead of 6, and the reference's `MoE.forward` walks them in a PYTHON LOOP, one fp4 GEMM triple
# per expert. It is the next lever (a grouped expert GEMM), it is bigger than this one, and it is
# orthogonal: nothing here has to change for it to land.
#
# WHY IT CANNOT BE RENTED. `Attention.forward`'s decode branch is a hard `seqlen == 1`
# (`kv_cache[:bsz, start_pos % win] = kv.squeeze(1)`, model.py:535) and so is `Compressor.forward`'s
# (`kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)`, :353/:362). DeepSeek's own loop never decodes
# more than one token, so there is no multi-token decode branch to call -- not a version skew, an
# absent feature. This is therefore a DELIBERATE, documented exception to rent-don't-rewrite
# (docs/MODEL_RUNTIME.md): the chunk MECHANICS are written here, driving the reference's own weights
# and calling its own `Block.forward`, `hc_pre`/`hc_post`, `MoE`, norms and `Compressor.forward`.
# The exception is kept honest by the parity bar, and the surface is deliberately one function:
# everything except `Attention.forward` is multi-token-correct in the reference already.
#
# WHAT ONE CHUNK PASS DOES, AND WHY EACH PIECE IS SHAPED THE WAY IT IS
#
#   window attention   position p attends the 128 ring slots holding p-win+1..p. The ring cannot
#       simply be written for all s positions first: chunk position p+i lands on slot (p+i) % win,
#       which is exactly the slot holding the OLDEST entry of p's own window, so a pre-written ring
#       answers p's oldest window slots with tokens from p's FUTURE. So the chunk's kv goes to a
#       SCRATCH region appended to kv_cache, the ring keeps its pre-chunk contents for the whole
#       attention, and the per-position index rows point at scratch for the chunk's own positions and
#       at the ring for everything earlier (`_ChunkPlan.window_rows`). The ring is committed AFTER
#       the read, which reproduces the reference's write-before-read within a position exactly (p
#       reads its own kv -- out of scratch, same bits) while keeping the earlier positions' view
#       intact. sparse_attn is already per-position (`T.Kernel(m, b)`, kernel.py:301 -- one thread
#       block per position), so s positions in one call is the SAME arithmetic per position.
#
#   compressor state   `kv_state`/`score_state` are fp32 accumulators advanced one position at a
#       time with an emit at (p+1) % ratio == 0, and the overlap variant shifts its own window on
#       every emit. That is genuinely sequential, so it stays a python loop over s calls to the
#       REFERENCE's own `Compressor.forward` -- at [b, 1, dim] each, the same shapes the per-token
#       path uses, hence bit-identical by construction (and measured so, on the GPU too: kv_state
#       and score_state come back torch.equal there while everything around them has drifted).
#
#   indexer            `index_score`'s kv read is `kv_cache[:bsz, :end_pos // ratio]`, and end_pos is
#       PER POSITION -- a chunk that crosses a compression boundary has positions with different read
#       lengths AND different topk widths (`min(index_topk, end_pos // ratio)`). Batching those with
#       one masked topk would silently change which slots win. So the chunk is split into the
#       CONTIGUOUS RUNS that share a read length (at most ceil(s/ratio)+1 of them; one for a chunk
#       that crosses nothing) and each run runs the reference's own einsum+topk at exactly the width
#       that run's positions would have had alone. Rows are then right-padded to a common width with
#       -1, which sparse_attn treats as "no position" -- on the GPU kernel a padded 64-block is
#       provably a no-op (scores_max unchanged, scores_scale = exp(0) = 1, gemm adds 0).
#
#   MoE / hyper-connections / norms   per-position by construction and already multi-token: rented
#       verbatim through `Block.forward`. This is where most of the s-fold saving lands.
#
# THE ONE THING THIS PATH IS NOT: BIT-IDENTICAL TO THE LOOP. Every mechanism above is exact -- same
# operands, same indices, same state, same order -- but torch picks its kernels by tensor SIZE, and a
# batched pass hands it bigger ones. MKL uses a different sgemm for M = 1 than for M = 6 (the
# reassociation `_tail_logit_rows`, v4_pipe.py:698, already keeps away from the logits, measured
# there at ~4e-7); cuBLAS and tilelang do the same; and it is not only the GEMMs -- bf16 `rsqrt`
# rounds one ulp differently once a tensor is big enough to take its vectorized path. No
# implementation of a batched pass avoids that, so "bit-exact against the loop" is not on the menu
# and pretending otherwise would be the lie. What IS provable is that nothing else differs, and that
# is what tests/test_v4_stage.py's section 9 proves, exactly:
#
#   every chunk position attends the same PLACES as the loop, in the same order, and those places
#   hold the same BYTES (the gathered kv are compared, which is what catches a ring written before
#   it is read -- the meanings alone would not), and after the chunk the window ring, the compressed
#   region and both fp32 accumulators are torch.equal to the loop's, per layer, on an identical input
#
# with what is left measured rather than waved at: <= 24 bf16 ulps of payload drift at s <= 6 and an
# identical greedy token stream over 90/90 swept chunks (9/18 at s = 8, where this toy's tensors
# start crossing torch's kernel thresholds). Default OFF is what makes that an honest trade rather
# than a silent one.
#
# ON THE GPU IT IS BETTER THAN THAT, AND THE SPLIT IS INSTRUCTIVE. Run on a 5090 against the real
# fp8/fp4 checkpoint, with the layer input held identical, the whole chunked ATTENTION comes back
# torch.equal to the loop -- output, window ring, compressed region, indexer cache, both fp32
# accumulators, on a ratio-4 layer and a ratio-128 one. tilelang's GEMMs tile K identically whatever
# M is and sparse_attn is one thread block per position, so the part written here is exactly
# invariant where it actually runs. What still drifts there is the RENTED half: cuBLAS picks a gemv
# for M = 1 and a gemm for M = 6, so hc_pre and the MoE reassociate (measured directly: fp32
# K=16384 linear differs at 6e-5, bf16 4096x4096 at 5e-1). Batching costs ulps in the parts nobody
# wrote and nothing in the parts we did.


class _ChunkPlan:
    """The index geometry of one s-token chunk at `start_pos` -- what the reference builds one
    position at a time, built once for the whole chunk and shared by every layer in the stage.

    Everything here is integer bookkeeping over `window_size`, `compress_ratio` and the positions
    themselves; nothing depends on the weights, so one plan serves all layers (the per-layer part is
    the scratch base, which is why `window_rows` is keyed by it -- a ratio-0 layer's kv_cache is
    shorter than a ratio-4 layer's, so its scratch starts somewhere else).

    Built on the host and moved once: these are a few hundred int32 per chunk, and building them
    with device ops would cost more launches than the pass is trying to save."""

    def __init__(self, args, start_pos, s, bsz, device, cap):
        self.start_pos, self.s, self.bsz, self.device, self.cap = start_pos, s, bsz, device, cap
        self.win = args.window_size
        self._win_rows, self._comp_rows, self._slots = {}, {}, None

    def _to(self, rows):
        """[s][k] python ints -> the reference's own topk_idxs shape/dtype: [bsz, s, k] int32."""
        t = torch.tensor(rows, dtype=torch.int32).view(self.s, -1)
        return t.unsqueeze(0).expand(self.bsz, -1, -1).contiguous().to(self.device)

    def ring_slots(self):
        """The window-ring slot each chunk position owns: [s] long, for the post-attention commit."""
        if self._slots is None:
            self._slots = torch.tensor([(self.start_pos + i) % self.win for i in range(self.s)],
                                       dtype=torch.long, device=self.device)
        return self._slots

    def window_rows(self, base):
        """`get_window_topk_idxs` (model.py:261) for every chunk position, retargeted at `base`.

        The reference's decode row is the ring in TEMPORAL order -- for p >= win-1,
        `[sp+1..win-1, 0..sp]` with sp = p % win, whose k-th entry holds absolute position
        p - win + 1 + k; for 0 < p < win-1 it is `[0..p]` right-padded with -1, whose k-th entry
        holds position k. Both are reproduced exactly, and then every entry whose position belongs
        to THIS chunk (>= start_pos) is pointed at the scratch row that holds it instead of at the
        ring slot it will eventually occupy. That single substitution is what makes one pass legal:
        the ring still holds the pre-chunk tokens for the whole attention, so a chunk position's
        oldest window entries are the tokens that were really there, not the chunk's own tail."""
        if base not in self._win_rows:
            win, sp0, rows = self.win, self.start_pos, []
            for j in range(self.s):
                p = sp0 + j
                if p >= win - 1:
                    sp = p % win
                    slots = list(range(sp + 1, win)) + list(range(sp + 1))
                    pos = [p - win + 1 + k for k in range(win)]
                else:
                    slots = list(range(p + 1)) + [-1] * (win - p - 1)
                    pos = list(slots)
                rows.append([base + (q - sp0) if q >= sp0 else v for v, q in zip(slots, pos)])
            self._win_rows[base] = self._to(rows)
        return self._win_rows[base]

    def lengths(self, ratio):
        """How many compressed slots each chunk position may read: `(p + 1) // ratio`, per position.

        This is the reference's `end_pos // ratio` with end_pos = p + 1 (model.py:413,426,433) -- the
        prefix of the compressed region that exists AT p, whose last slot is the one p itself just
        wrote. Non-decreasing across the chunk, which is what makes `groups` contiguous."""
        return [(self.start_pos + j + 1) // ratio for j in range(self.s)]

    def groups(self, ratio):
        """The chunk split into contiguous runs sharing a compressed read length. -> [(j0, j1, n)].

        One run for a chunk that crosses no compression boundary, one more per boundary crossed."""
        n, out, j0 = self.lengths(ratio), [], 0
        for j in range(1, self.s + 1):
            if j == self.s or n[j] != n[j0]:
                out.append((j0, j, n[j0]))
                j0 = j
        return out

    def compress_rows(self, ratio, offset):
        """`get_compress_topk_idxs` (model.py:275) for every chunk position, padded to one width.

        The reference's decode row is the whole prefix `arange((p+1) // ratio) + offset`, so the rows
        of a chunk that crosses a boundary differ in LENGTH. They are right-padded with -1 (sparse
        attention's "no position") to the widest, which is the last position's -- the pad is a
        masked-out column, never a slot."""
        if (ratio, offset) not in self._comp_rows:
            n = self.lengths(ratio)
            wide = n[-1]
            rows = [[offset + t for t in range(k)] + [-1] * (wide - k) for k in n]
            self._comp_rows[(ratio, offset)] = (
                self._to(rows) if wide else
                torch.zeros(self.bsz, self.s, 0, dtype=torch.int32, device=self.device))
        return self._comp_rows[(ratio, offset)]


def _chunk_compressor(c, x, start_pos):
    """Advance a Compressor over a chunk: the REFERENCE's own decode branch, one position at a time.

    Deliberately not batched. `kv_state`/`score_state` are running fp32 accumulators with a boundary
    emit at `(p + 1) % ratio == 0`, and the ratio-4 overlap variant copies its second half over its
    first ON that emit (model.py:359-360) -- a genuinely sequential recurrence over at most s steps.
    Batching it would mean reimplementing the emit schedule; looping it costs s pairs of small fp32
    GEMMs and is bit-identical to the per-token path by construction, because the shapes it hands the
    reference ([b, 1, dim]) are the ones the per-token path hands it.

    `.contiguous()` because a [b, s, dim] slice is not the freshly-allocated [b, 1, dim] the loop
    would have passed, and F.linear is entitled to notice."""
    for j in range(x.size(1)):
        c(x[:, j:j + 1].contiguous(), start_pos + j)


def _chunk_indexer(self, x, qr, start_pos, offset, plan):
    """`Indexer.forward` (model.py:408) for a whole chunk. -> topk_idxs [b, s, k] (right-padded -1).

    q, the weights projection and the compressor are position-independent or sequential, so they run
    once for the chunk. The SCORING is the part a chunk cannot do in one shot: `end_pos // ratio` is
    the read length AND (through `min(index_topk, ...)`) the topk width, and both change mid-chunk
    at every compression boundary. Masking a common-width score to fake the shorter reads would
    change which slots a topk near-tie picks, so instead the chunk is split into runs that share a
    length (`_ChunkPlan.groups`) and each run runs the reference's own einsum + topk at exactly its
    own width -- identical arithmetic to what those positions would have got alone, one call per
    boundary crossed instead of one per position."""
    M = ref()
    bsz, seqlen, _ = x.size()
    freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
    ratio, rd = self.compress_ratio, self.rope_head_dim
    if self.compressor.kv_cache is None:
        self.compressor.kv_cache = self.kv_cache
        self.compressor.freqs_cis = self.freqs_cis
    q = self.wq_b(qr)
    q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
    M.apply_rotary_emb(q[..., -rd:], freqs_cis)
    q = M.rotate_activation(q)
    M.fp4_act_quant(q, M.fp4_block_size, True)
    _chunk_compressor(self.compressor, x, start_pos)
    weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
    rows, wide = [], 0
    for j0, j1, n in plan.groups(ratio):
        k = min(self.index_topk, n)
        if k == 0:                                          # nothing compressed yet: no rows to pick
            rows.append(torch.zeros(bsz, j1 - j0, 0, dtype=torch.long, device=x.device))
            continue
        score = torch.einsum("bshd,btd->bsht", q[:, j0:j1].contiguous(), self.kv_cache[:bsz, :n])
        score = (score.relu_() * weights[:, j0:j1].unsqueeze(-1)).sum(dim=2)
        rows.append(score.topk(k, dim=-1)[1] + offset)
        wide = max(wide, k)
    if len(rows) == 1:
        return rows[0]
    return torch.cat([r if r.size(-1) == wide else
                      torch.cat([r, r.new_full((bsz, r.size(1), wide - r.size(-1)), -1)], dim=-1)
                      for r in rows], dim=1)


def _chunk_attention(self, x, start_pos, plan):
    """`Attention.forward`'s decode branch (model.py:490-548) for a WHOLE s-token chunk at once.

    Line for line the reference's, with three substitutions and nothing else:

      the window write   the chunk's kv goes to the SCRATCH rows at the end of kv_cache BEFORE the
                         attention and to the ring AFTER it, instead of to the ring before it. The
                         index rows (`_ChunkPlan.window_rows`) send each position at its own kv in
                         scratch, so write-before-read still holds per position -- while the ring
                         still answers earlier positions with the tokens that were really there.
                         Both writes carry the same bits; the ring commit is what the NEXT chunk and
                         the rollback snapshot read.

      the compressor     s sequential calls into the reference's own decode branch
                         (`_chunk_compressor`), all of them before the attention, because a position
                         reads the compressed slot it just wrote. A later chunk position can only
                         write slots at or beyond an earlier one's read bound `(p+1) // ratio`, so
                         doing all of them first cannot contaminate an earlier position -- the same
                         zero-margin argument `Stage._snapshot` makes for not snapshotting the
                         compressed region, one chunk further along.

      the index rows     built for s positions instead of one, per-position exact (see _ChunkPlan).

    Everything else -- q/kv projections, rope, the fp8 QAT quantizers, sparse_attn, the grouped o
    projection -- is the reference's own code on a [b, s, ...] tensor, which is the shape its prefill
    branch already feeds it."""
    M = ref()
    bsz, seqlen, _ = x.size()
    freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
    win, ratio, rd = self.window_size, self.compress_ratio, self.rope_head_dim
    if self.compress_ratio and self.compressor.kv_cache is None:
        self.compressor.kv_cache = self.kv_cache[:, win:]
        self.compressor.freqs_cis = self.freqs_cis
        if self.indexer is not None:
            self.indexer.freqs_cis = self.freqs_cis
    # q
    qr = q = self.q_norm(self.wq_a(x))
    q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
    q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
    M.apply_rotary_emb(q[..., -rd:], freqs_cis)

    # win kv & topk_idxs
    kv = self.wkv(x)
    kv = self.kv_norm(kv)
    M.apply_rotary_emb(kv[..., -rd:], freqs_cis)
    M.act_quant(kv[..., :-rd], 64, M.scale_fmt, M.scale_dtype, True)
    base = self.kv_cache.size(1) - plan.cap                 # the scratch region, past every real slot
    topk_idxs = plan.window_rows(base)
    if self.compress_ratio:
        offset = win
        if self.indexer is not None:
            compress_topk_idxs = _chunk_indexer(self.indexer, x, qr, start_pos, offset, plan).int()
        else:
            compress_topk_idxs = plan.compress_rows(ratio, offset)
        topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)

    # compress kv & attn
    self.kv_cache[:bsz, base:base + seqlen] = kv            # every position's own kv, readable now
    if self.compress_ratio:
        _chunk_compressor(self.compressor, x, start_pos)
    o = M.sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
    self.kv_cache[:bsz, plan.ring_slots()] = kv            # commit the ring AFTER every read of it
    M.apply_rotary_emb(o[..., -rd:], freqs_cis, True)

    # o
    o = o.view(bsz, seqlen, self.n_local_groups, -1)
    wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    return self.wo_b(o.flatten(2))


_CHUNK_BLOCK = None


def chunk_block_cls(M):
    """`Block` with a chunk-capable attention (memoized per process).

    A SUBCLASS, not a monkeypatch, and chosen at construction: a stage built without the fast path
    holds the reference's own `Block` and cannot take a branch that does not exist in it, which is
    what makes "OFF by default changes nothing" checkable rather than promised. The submodule tree,
    the parameter names and therefore every state_dict are identical, so `load()` and the tests'
    in-process weight transfer neither know nor care.

    `_chunk` is armed by `Stage._run_chunk` for the duration of one pass and is None everywhere else
    -- including inside `_replay`, which is why a rollback replays through the reference path."""
    global _CHUNK_BLOCK
    if _CHUNK_BLOCK is None or _CHUNK_BLOCK.__mro__[1] is not M.Block:
        class _ChunkAttention(M.Attention):
            _chunk = None

            def forward(self, x, start_pos):
                if self._chunk is None:
                    return super().forward(x, start_pos)
                return _chunk_attention(self, x, start_pos, self._chunk)

        class _ChunkBlock(M.Block):
            attention_cls = _ChunkAttention

        _CHUNK_BLOCK = _ChunkBlock
    return _CHUNK_BLOCK


# ── the stage ────────────────────────────────────────────────────────────────────────────────────

class Stage:
    """One V4 layer block [lo:hi), optionally carrying the head's embedding and/or the tail's head.

    Contract, consumed the way m25_pipe consumes m25_stage:
        embed(token_ids)            -> h [b, s, hc_mult, dim]      head only
        forward(h, ids, start_pos)  -> h [b, s, hc_mult, dim]      every stage
        logits_all(h)               -> [b, s, vocab] fp32          tail only
        tail_main_hidden()          -> [b, s, len(targets)*dim]    tail only, _dspark armed
        reset()                     -> drop all per-stage state
    `h` and `ids` are plain tensors, so shard/transport.py encodes them as-is.

    The reference's `embed`/`head` modules are exposed as `embed_tokens`/`lm_head`, k3_stage's and
    m25_stage's names -- `embed` is the contract's method and `head`/`tail` are the role flags, so
    the reference's own attribute names are already taken. Step 4's DSpark drafter wants both module
    objects (`DSparkBlock.embed`/`.head`, model.py:903), which is what `dspark=True` on a tail stage
    is for: it loads the embedding on a stage that would otherwise have no use for it."""

    def __init__(self, lo, hi, args=None, *, head=False, tail=False, dspark=False,
                 device=None, dtype=None, spec_depth=None, fast_verify=None):
        self.lo, self.hi = lo, hi
        self.args = args if args is not None else config()
        self.device = device or dev
        self.dtype = dtype or getattr(torch, V4_DTYPE)
        self.head, self.tail, self._dspark = head, tail, dspark
        self._fast = V4_FAST_VERIFY if fast_verify is None else bool(fast_verify)
        self._chunk_cap = V4_FAST_VERIFY_MAX if self._fast else 0
        M = ref()
        self._M = M                       # the exec'd reference module (v4_whole_layer_graph borrows its kernels)
        self.globals = _set_globals(M, self.args)
        a = self.args
        if not 0 <= lo < hi <= a.n_layers:
            raise RuntimeError(f"v4 stage[{lo}:{hi}) is not a range inside 0..{a.n_layers}")
        block_cls = chunk_block_cls(M) if self._fast else M.Block
        # `with torch.device(...)` + the reference's own set_dtype contextmanager is generate.py's
        # construction environment (generate.py:77,87) reproduced exactly. Both matter: the dtype
        # decides what the bare `torch.empty`/`torch.zeros` parameters and kv buffers come out as,
        # the device keeps a 158 GiB model from being built on the host and then copied.
        with torch.device(self.device), M.set_dtype(self.dtype):
            self.layers = torch.nn.ModuleList([block_cls(li, a) for li in range(lo, hi)])
            self.embed_tokens = M.ParallelEmbedding(a.vocab_size, a.dim) if (head or dspark) else None
            if tail:
                self.norm = M.RMSNorm(a.dim, a.norm_eps)
                self.lm_head = M.ParallelHead(a.vocab_size, a.dim, a.norm_eps, a.hc_eps)
                # The hc_head collapse is a Block METHOD driven by Transformer-LEVEL parameters
                # (model.py:908-910 + :922). The tail owns the parameters and borrows its own last
                # block to run them, which is literally what `layer.hc_head(...)` does after the
                # reference's loop falls out with `layer` still bound to the last one.
                with M.set_dtype(torch.float32):
                    self.hc_head_fn = torch.nn.Parameter(torch.empty(a.hc_mult, a.hc_mult * a.dim))
                    self.hc_head_base = torch.nn.Parameter(torch.empty(a.hc_mult))
                    self.hc_head_scale = torch.nn.Parameter(torch.empty(1))
        # Re-lay the routed experts as ONE contiguous bank per layer, so the grouped fp4 MoE kernel
        # can gather them without holding a second copy of the weights. HERE, between construction and
        # `load()`, is the only place it is free: it repoints each expert Linear's parameter at a
        # slice of the bank, and `load()`'s `load_state_dict` then writes THROUGH that view, so the
        # checkpoint lands in the bank and nowhere else -- one copy on the card, the same bytes the
        # non-grouped path holds. Doing it after load would mean stacking a duplicate, which at the
        # shipped dims is ~3.2 GiB per layer and is exactly why the lever declined on a full stage.
        # preserve=False because the Blocks are two statements old and every routed-expert byte is
        # still the constructor's uninitialised `torch.empty`: the layout may RELEASE each layer's
        # per-expert run before it allocates that layer's banks, instead of asking the driver for a
        # bank while the memory it replaces is still resident. That ordering is the difference
        # between a peak of 27.98 GiB and 31.17 GiB on an 8-layer stage -- see v4_moe_grouped.
        # No-op under V4_MOE_GROUPED=0 (the default): nothing allocated, nothing repointed.
        import v4_moe_grouped
        self._moe_banked = banked = v4_moe_grouped.bank_layout(self.layers, preserve=False)
        if banked:
            print(f"[v4] stage[{lo}:{hi}): grouped-MoE bank layout on {banked} layer(s) — the routed "
                  f"experts ARE the bank, no duplicate", flush=True)
        # Same window, same mechanism, 1/96th the size: V4_FP8_SHARED re-lays each layer's SHARED
        # expert w1/w3 as one contiguous [2*inter, dim] bank (weight + scale), repointing the
        # Linears at slices so `load()` writes the checkpoint straight through the views — one copy
        # on the card, the reference path reads the same bytes, and the fused single-launch path
        # (v4_fp8_gemv._shared_forward) has its bank at serve time. No-op under V4_FP8_SHARED=0.
        import v4_fp8_gemv
        self._shared_banked = shared_banked = v4_fp8_gemv.shared_bank_layout(self.layers)
        if shared_banked:
            print(f"[v4] stage[{lo}:{hi}): shared-expert w13 bank on {shared_banked} layer(s) — "
                  f"w1+w3 serve as ONE fp8 launch", flush=True)
        if self._fast:
            self._reserve_chunk_scratch()
        for m in self._owned_modules():
            m.eval()
        # Ascending layer order, NOT the tuple's -- the reference appends a tap inside its own
        # `for i, layer in enumerate(self.layers)` loop (model.py:920), so the concatenation order is
        # the layers' order however dspark_target_layer_ids happens to be written.
        self._tap_ids = tuple(sorted(li for li in a.dspark_target_layer_ids if lo <= li < hi))
        self._spec = False
        # W-deep rollback ring: pipelined speculation streams W s=1 frames before the first reply
        # comes back, so a rejection may have to rewind across all of them. maxlen caps how far —
        # a rewind past the oldest live checkpoint refuses loudly rather than serving stale state.
        self._spec_depth = spec_depth if spec_depth is not None else int(os.environ.get("V4_SPEC_DEPTH", 16))
        self._spec_ckpts = deque(maxlen=self._spec_depth)
        self._last_tap = {}
        self._pos = 0
        self._replaying = False
        self.reset()
        # One graph object per layer, capturing lazily on the first decode step (see V4_CUDA_GRAPH):
        # island mode graphs only the hc_pre/hc_post/norm islands, whole mode graphs the WHOLE decode
        # layer (attention core included, real routed MoE eager). A stage that cannot graph stays fully
        # eager and says why -- never a silent half-capture.
        self._block_graphs = None
        self._graph_mode = _graph_mode()
        if self._graph_mode != "off":
            why = self._graph_refusal()
            if why:
                print(f"[v4] GRAPH REFUSED for stage[{lo}:{hi}): {why} — staying eager", flush=True)
            elif self._graph_mode == "whole":
                import v4_whole_layer_graph as _wl
                # V4_MOE_IN_GRAPH pulls the routed MoE INSIDE the capture, which is only correct on a
                # layer whose MoE reaches the grouped fp4 kernel. It is a REQUEST: each layer judges
                # itself on its first decode step and drops back to the eager-MoE split if it cannot
                # prove it (v4_whole_layer_graph._moe_refusal). Default OFF.
                mm = "graph" if _wl.V4_MOE_IN_GRAPH else "eager"
                self._block_graphs = [_wl.WholeBlockGraphs(L, self, moe_mode=mm) for L in self.layers]
            else:
                self._block_graphs = [_BlockGraphs(L, self) for L in self.layers]

    def _graph_refusal(self):
        """Why this stage cannot graph its decode step, or None. Loud and specific, never silent.

        Serves BOTH modes, and neither has anything device-independent to refuse over. In island mode
        the captured regions are the position- and data-INDEPENDENT Block methods (hc_pre/hc_post/the
        norms), so unlike K3 there is no growing KV or per-position branch INSIDE a graph. In whole
        mode the attention core is captured too, but v4_whole_layer_graph is what makes it
        position-independent (the ring slot is chosen at replay from a copied-in position, the indexer
        read is bucketed, the selection is width-invariant) -- so again nothing here to refuse.
        The only hard requirement either way is a CUDA device to capture on.

        A refusal PRINTS. `graph=off` in the repr with a refusal line above it means the lever
        declined and said why; `graph=off` with no line means the flag never arrived."""
        if not str(self.device).startswith("cuda"):
            return f"device is {self.device} (CUDA graphs are a GPU-only capture)"
        return None

    @property
    def _spec_ckpt(self):
        """The newest live checkpoint, or None. The one-chunk seam generalized to a ring of W: the
        full-accept no-op and the guard tests read the most recent as 'the' checkpoint."""
        return self._spec_ckpts[-1] if self._spec_ckpts else None

    def _owned_modules(self):
        yield self.layers
        for n in ("embed_tokens", "norm", "lm_head"):
            if getattr(self, n, None) is not None:
                yield getattr(self, n)

    def _reserve_chunk_scratch(self):
        """Append `_chunk_cap` rows to every `Attention.kv_cache` for the chunked path to write into.

        The chunk's own kv has to live in the SAME tensor sparse_attn is reading (its indices are
        offsets into one buffer), and the alternative -- concatenating the chunk onto the cache per
        layer per chunk -- copies the entire compressed region, which at V4's shape is the largest
        buffer in the stage. Widening the buffer once costs `cap` rows per layer and copies nothing.

        Safe against the reference path by construction: the added rows sit PAST every index the
        reference can produce (window indices are < win, compressed indices are < win + max_seq_len /
        ratio), so a stage that never takes the chunk branch cannot read or write them. The lazily
        bound aliases follow, because they are taken from `self.kv_cache` on the first forward and
        this runs in the constructor: `compressor.kv_cache = kv_cache[:, win:]` simply ends with
        `cap` unused slots on the end, past the `max_seq_len // ratio` it can address.

        MUST run before the first forward, for the same reason: an alias bound to the OLD buffer
        would leave the compressor writing into a tensor nothing else reads."""
        with torch.no_grad():
            for L in self.layers:
                a = L.attn
                b, n, d = a.kv_cache.shape
                a.kv_cache = a.kv_cache.new_zeros(b, n + self._chunk_cap, d)

    # ---- state ----

    def _compressors(self):
        """Every Compressor this stage owns, and whether it hangs off an Indexer.

        Attention only grows `.compressor`/`.indexer` when `compress_ratios[layer_id]` is non-zero,
        and only ratio 4 builds an Indexer (model.py:472-477) -- so a pure sliding-window layer has
        neither and a ratio-128 layer has a compressor but no indexer."""
        for L in self.layers:
            attn = L.attn
            if not attn.compress_ratio:
                continue
            yield attn.compressor, False
            if attn.indexer is not None:
                yield attn.indexer.compressor, True

    def reset(self):
        """Drop every per-stage tensor a job accumulated, IN PLACE. Weights survive, addresses hold.

        Four buffers, and every one of them has to be restored to the value its constructor wrote,
        not merely to zero: `Attention.kv_cache` zeros (window ring + compressed region),
        `Indexer.kv_cache` zeros, `Compressor.kv_state` zeros, and `Compressor.score_state` -inf.
        That last one is the one that bites -- it is a softmax logit accumulator, so a zero there is
        not "empty", it is "this slot has weight 1", and a prefill only rewrites the `remainder` rows
        past the last full compression block (model.py:339-341). Leave the previous job's tail in
        there and the next prefill mixes it into the first compressed KV.

        Rebuilding the modules would clear it too and would throw the weights away with it. Zeroing
        in place also keeps the lazily-bound aliases valid: `compressor.kv_cache` is a VIEW of
        `attn.kv_cache[:, win:]` (model.py:497) and `indexer.compressor.kv_cache` IS
        `indexer.kv_cache`, so both follow the buffer they were bound to."""
        with torch.no_grad():
            for L in self.layers:
                L.attn.kv_cache.zero_()
                if L.attn.compress_ratio and L.attn.indexer is not None:
                    L.attn.indexer.kv_cache.zero_()
            for c, _ in self._compressors():
                c.kv_state.zero_()
                c.score_state.fill_(float("-inf"))
        self._pos = 0
        self._last_tap = {}
        self._spec_ckpts.clear()

    def _snapshot(self):
        """Clone exactly the state a rejected speculation can poison. `_seek` restores it.

        WHAT IS IN: the window ring `kv_cache[:, :win]`, because a rejected token's kv sits at
        `start_pos % win` and would be READ by the next accepted token at the same absolute position
        only if it were not overwritten first -- it is, but the window is also read by every query in
        the next `win` positions, so a wrong slot is live immediately. And both Compressor
        accumulators (the layer's and the Indexer's), because they are running softmax state that a
        rejected token folds into and no later write undoes.

        WHAT IS DELIBERATELY OUT: the compressed regions -- `kv_cache[:, win:]` and
        `Indexer.kv_cache`. A stale slot written by rejected speculation is ALWAYS REWRITTEN BEFORE
        ITS FIRST READ, which is the correctness argument for not snapshotting the largest buffer in
        the stage (at V4's shape, `max_seq_len // 4` compressed slots against a 128-slot window). The
        margin, though, is EXACTLY ZERO, and that is the part worth stating precisely: slot `j` is
        produced at position `q = (j + 1) * ratio - 1`, and the read set at position P is
        `[0, (P + 1) // ratio)`, whose last element is the slot P itself just wrote. So the first
        read of a slot is AT `q`, never after it -- inside the same position, and safe only because
        the reference WRITES BEFORE IT READS within that position:
        `self.compressor(x, start_pos)` (model.py:537) precedes
        `sparse_attn(q, self.kv_cache[:bsz], ...)` (:538), and the Indexer's own
        `self.compressor(x, start_pos)` (:423) precedes its
        `einsum(..., self.kv_cache[:bsz, :end_pos // ratio])` (:426).

        Swap either pair -- a re-vendored model.py, or a fused attention kernel that reads the
        compressed region from a value captured at `Attention.forward` entry -- and a rejected
        chunk's slot IS read stale, silently, in plausible-looking numbers.
        tests/test_v4_stage.py::test_rollback_survives_a_poisoned_compressed_region is the red test
        that pins it: it fills every slot this argument calls safe-to-be-stale with NaN, after a
        rewind that lands ON a compression boundary, and the stream must still come out bit-exact.

        THE ARGUMENT IS DEPTH-INVARIANT, which is what lets `_seek` rewind W frames and not one. The
        read set `[0, (P+1)//ratio)` is a function of the POSITION P alone -- slot j enters it exactly
        at P = q = (j+1)*ratio-1, the same position that writes it, at no earlier P and for no compress
        ratio. A rewind to r = (last committed + 1) re-processes every position >= r in order, so a
        slot poisoned by a rejected frame (necessarily at some q >= r, since positions < r committed)
        is rewritten at its own q before that q reads it -- however many boundaries, ratio-4 overlap
        or ratio-8 plain, the W rejected frames crossed. What the snapshot must carry FULLY for this to
        hold is the whole window ring plus BOTH accumulators of every compressor incl. the Indexer's
        (kv_state/score_state, all rows) -- the overlap compressor mutates them at each boundary (the
        `kv_state[:ratio] = kv_state[ratio:]` shift, model.py:359) and a partial clone would restore a
        half-shifted ring. tests/test_v4_stage.py::test_multi_deep_rollback_across_boundaries streams
        W s=1 frames, rewinds the whole way across several boundaries, NaN-poisons, and its
        mutation-check proves a snapshot that drops any of those rows is caught.

        The Indexer's own kv_cache needs no window snapshot at all: unlike Attention's it is
        entirely compressed slots, with no ring prefix (model.py:405)."""
        win = self.args.window_size
        snap = []
        with torch.no_grad():
            for L in self.layers:
                snap.append({"win": L.attn.kv_cache[:, :win].clone()})
            for c, _ in self._compressors():
                snap.append({"kv_state": c.kv_state.clone(), "score_state": c.score_state.clone()})
        return snap

    def _restore(self, snap):
        """Write a `_snapshot()` back in place. `_seek`'s rollback; unused on the greedy path."""
        win = self.args.window_size
        with torch.no_grad():
            n = len(self.layers)
            for L, e in zip(self.layers, snap[:n]):
                L.attn.kv_cache[:, :win].copy_(e["win"])
            for (c, _), e in zip(self._compressors(), snap[n:]):
                c.kv_state.copy_(e["kv_state"])
                c.score_state.copy_(e["score_state"])

    def _replay(self, h, ids, start_pos):
        """Re-feed an accepted prefix through the layers to rebuild what a restore rolled back.

        STATE ONLY: it advances the window ring and both compressor accumulators over
        [start_pos, start_pos + s) exactly as sequential decode would, and throws the outputs away.
        The per-token loop is `forward`'s, for `forward`'s reason -- the reference's decode branch
        writes a squeezed `seqlen == 1` into `kv_cache[:, start_pos % win]`.

        Three deliberate differences from `forward`, each of which is a bug if it is dropped:
          * NO new checkpoint. The one being spent is the only one that covers this interval, and
            overwriting it mid-rollback would leave the stage unable to rewind again.
          * NO taps. `_last_tap` still describes the VERIFY chunk, whose taps the drafter consumed
            before the rejection was known; the drafter is advanced over committed positions only and
            is never re-driven from a replay, so re-recording here would hand it its own history back.
            (The taps are recomputed into a throwaway dict and dropped -- a mean over 4 streams.)
          * NO prefill branch. A replay is by construction at start_pos > 0."""
        self._replaying = True
        try:
            with torch.no_grad():
                for i in range(h.shape[1]):
                    self._run(h[:, i:i + 1], ids[:, i:i + 1], start_pos + i, {})
        finally:
            self._replaying = False
        self._pos = start_pos + h.shape[1]

    def _seek(self, start_pos):
        """Move the stage to `start_pos`, or refuse. k3_stage._seek's shape, V4's reasons.

        A rewind IS the speculative rollback: a rejection re-feeds the ring from the last COMMITTED
        position, so every stage has to put its per-token state back to what that position left
        behind. V4's is simpler than K3's -- no recurrent state to unwind and no KV to crop, because
        the reference's buffers are POSITION-INDEXED (`kv_cache[:, p % win]`, `kv_state[:, p % ratio]`).
        Restore the snapshot taken before `start_pos`, re-feed any accepted prefix, and every slot the
        rejected tail touched has been rewritten by the token that really belongs there; the
        compressed regions are argued away — at any depth ≤ W — in `_snapshot`'s docstring.

        UP TO W CHUNKS DEEP. Pipelined speculation streams s=1 frames back-to-back without waiting for
        replies, so the rejection that arrives while W frames are in flight has to rewind across all W.
        `_spec_ckpts` is the ring of their pre-frame snapshots; this picks the NEWEST checkpoint whose
        [start_pos, start_pos+s] still contains the target (an exact per-position snapshot needs no
        replay; a coarser one replays its accepted prefix) and then SPENDS it and every checkpoint
        after it — the discarded speculative future. Checkpoints BEFORE the target survive, so a
        deeper rewind in a later round is still possible; `commit` is what finally drops them. A
        target the whole ring cannot cover is a coordinator asking for a position no snapshot
        describes, and serving it off the current state would be silently wrong instead of loudly
        broken — so it refuses, naming the interval the ring does cover.

        THE FULL-ACCEPT PATH NEVER REACHES ANY OF THIS. A round that accepts all g = s-1 drafts
        commits g+1 tokens, so the next frame opens exactly at `_pos` — the no-op return above.
        Rollback costs exactly nothing on the rounds speculation is winning."""
        if start_pos == self._pos:
            return
        if start_pos > self._pos:
            raise RuntimeError(
                f"v4 stage[{self.lo}:{self.hi}]: start_pos {start_pos} is ahead of the {self._pos} "
                f"tokens this stage has seen — a gap means the skipped tokens were never fed "
                f"through this block's layers (reset() first, or replay from {self._pos})")
        ck = next((c for c in reversed(self._spec_ckpts)
                   if c["start_pos"] <= start_pos <= c["start_pos"] + c["s"]), None)
        if ck is None:
            covered = ("none" if not self._spec_ckpts else
                       f"[{self._spec_ckpts[0]['start_pos']}, "
                       f"{self._spec_ckpts[-1]['start_pos'] + self._spec_ckpts[-1]['s']}]")
            raise RuntimeError(
                f"v4 stage[{self.lo}:{self.hi}]: cannot rewind {self._pos} -> {start_pos}; the spec "
                f"checkpoint covers {covered} (the last W speculative frames' ring). A rollback only "
                f"rewinds inside that ring — arm _spec (the reset's `spec` flag does it) and rewind "
                f"before `commit` or the maxlen cap drops the checkpoint. reset() is the only other "
                f"way back.")
        self._restore(ck["state"])
        self._pos = ck["start_pos"]
        n = start_pos - ck["start_pos"]                     # the accepted prefix of the spent frame
        if n:
            self._replay(ck["h"][:, :n], ck["ids"][:, :n], ck["start_pos"])
        self._pos = start_pos
        while self._spec_ckpts and self._spec_ckpts[-1]["start_pos"] >= ck["start_pos"]:
            self._spec_ckpts.pop()                          # spent: the frame + every one it un-did

    def commit(self, pos):
        """Drop every checkpoint the ring has settled irrevocably past. The coordinator's ack.

        A checkpoint covering [p, p+s] is only ever rewound INTO to reach a position in that span; once
        `pos` tokens are committed the ring will never ask to go below `pos`, so a checkpoint that ends
        at or before it is dead weight. Dropping frees its clones (a window ring + both accumulators
        per compressor, per frame) — the memory the W-deep ring costs — without disturbing anything
        still in flight. Idempotent and cheap; the maxlen cap is the backstop when commits lag."""
        keep = deque((c for c in self._spec_ckpts if c["start_pos"] + c["s"] > pos),
                     maxlen=self._spec_ckpts.maxlen)
        self._spec_ckpts = keep

    # ---- the serve contract ----

    def embed(self, token_ids):
        """Head only: token ids -> h [b, s, hc_mult, dim]. `Transformer.forward:914-916`, verbatim.

        The `repeat` is not a broadcast: hyper-connections need four INDEPENDENT streams and the
        first Block's hc_pre mixes them by position, so a view would alias four copies of one."""
        if self.embed_tokens is None:
            raise RuntimeError(f"v4 stage[{self.lo}:{self.hi}]: no embedding — head=False "
                               f"(pass dspark=True on a tail that needs one for the drafter)")
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        with torch.no_grad():
            h = self.embed_tokens(ids)
            return h.unsqueeze(2).repeat(1, 1, self.args.hc_mult, 1)

    def _run(self, h, ids, start_pos, taps):
        """One pass of this stage's layers over `h` at `start_pos`, collecting any owned taps.

        A single-token DECODE step (start_pos > 0, seqlen == 1) routes each layer through its captured
        graph when the stage is armed -- island mode replays the hc_pre/hc_post/norm islands (attn/ffn
        eager between), whole mode replays the whole layer incl. the capture-safe attention core (real
        routed MoE eager between two graphs, or INSIDE the one graph under V4_MOE_IN_GRAPH on a layer
        that proved it can be). Prefill, a multi-token chunk, and a rollback replay
        (`_replaying`) all stay on the eager per-layer call -- the graph is a fixed b=1,s=1 shape and
        the reference's own decode branch is the thing it is proven against."""
        bg = self._block_graphs
        graphed = bg is not None and not self._replaying and start_pos > 0 and h.shape[1] == 1
        for i, (li, L) in enumerate(zip(range(self.lo, self.hi), self.layers)):
            h = bg[i].run(h, ids, start_pos) if graphed else L(h, start_pos, ids)
            if self._dspark and li in self._tap_ids:
                # THE TAP MUST STAY OUT HERE, on the Python side of the replay. It is safe today
                # because a graph spans at most ONE layer, so this runs per step on that layer's fresh
                # output. Move it inside a captured region -- or let a graph span more than one layer,
                # so this line falls between two captured layers -- and it becomes m25's stale-EAGLE-aux
                # bug verbatim (commit e8d2c82): a Python-level side effect inside a graph is recorded
                # ONCE at capture and skipped by every replay, so the drafter would feed on the
                # capture step's aux forever, behind valid receipts and entirely plausible numbers.
                # tests/test_v4_whole_layer.py::test_graph_output_is_fresh_across_positions is the gate.
                taps.setdefault(li, []).append(h.mean(dim=2).detach().clone())
        # A graphed layer's output ALIASES its hc_post graph's static buffer; the last layer's escapes
        # the stage (onto the wire, or into logits) and must not be overwritten by the next step's
        # replay. One clone of [1,1,4,dim] per token, only on the graphed path.
        return h.clone() if graphed else h

    def _chunk_ok(self, s):
        """Can this chunk go through the fast path? Anything else falls back to the per-token loop.

        Two bounds, both real: the scratch region is `_chunk_cap` rows wide, and a chunk wider than
        the window would wrap the ring onto itself (two chunk positions on one slot), which the
        single index_copy_ commit does not define. Falling back rather than refusing keeps the fast
        flag a PERFORMANCE switch -- a stage that meets an oversized chunk answers it correctly and
        slowly, which is the right failure for a knob a coordinator sets."""
        return self._fast and 1 < s <= min(self._chunk_cap, self.args.window_size)

    def _run_chunk(self, h, ids, start_pos, taps):
        """The whole chunk through every layer in ONE pass each. See "the chunked verify path".

        The plan is per CHUNK, not per layer: window geometry and compression lengths depend only on
        positions, so all layers share it (`window_rows` is keyed by the layer's scratch base, the
        one part that differs). Arming is a plain attribute on each attention and is dropped in a
        `finally` -- a fault mid-pass must not leave a stale plan that the next single-token forward
        would then read as "this is a chunk"."""
        plan = _ChunkPlan(self.args, start_pos, h.shape[1], h.shape[0], self.device, self._chunk_cap)
        for L in self.layers:
            L.attn._chunk = plan
        try:
            return self._run(h, ids, start_pos, taps)
        finally:
            for L in self.layers:
                L.attn._chunk = None

    def forward(self, h, ids, start_pos):
        """Run this stage's layers. BOTH `h` and `ids` are inputs -- see the module docstring.

        Prefill goes through the reference's own multi-token branch. A chunk at `start_pos > 0` is
        replayed one position at a time, because the reference's decode branch writes
        `kv_cache[:, start_pos % win]` and `kv_state[:, start_pos % ratio]` from a squeezed
        `seqlen == 1` (model.py:353,362,535) and would silently write only the chunk's last token
        otherwise. Exact, not approximate: HC mixing is per-position, and the loop advances the KV
        and compressor state in exactly the order sequential decode would.

        A stage built with `fast_verify` (V4_FAST_VERIFY=1) runs that chunk in ONE pass per layer
        instead -- same state, same order, ~s times fewer dispatches. It is opt-in and it is not the
        reference: see "the chunked verify path" for the mechanism and for the one thing it gives up.

        VALIDATE FIRST, SEEK SECOND. `_seek` MUTATES -- a rewind restores, replays and spends the
        checkpoint -- so a frame that is going to be rejected must be rejected before it can do that.
        Otherwise a malformed chunk leaves the stage rewound with its checkpoint gone, and the
        retry of that same round can no longer roll back any deeper than it already did."""
        if ids is None:
            raise RuntimeError(
                f"v4 stage[{self.lo}:{self.hi}]: forward() needs the token ids, not just the hidden "
                f"state — the first {self.args.n_hash_layers} layers route their MoE by "
                f"tid2eid[input_ids] (see this module's docstring). Carry them with the payload.")
        h = h.to(device=self.device, dtype=self.dtype)
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        s = h.shape[1]
        if ids.shape[:2] != h.shape[:2]:
            raise RuntimeError(f"v4 stage[{self.lo}:{self.hi}]: ids {tuple(ids.shape)} do not match "
                               f"the payload's [b, s] = {tuple(h.shape[:2])}")
        self._seek(start_pos)
        if self._spec and start_pos > 0:
            # Taken BEFORE anything is touched: the whole point is to be able to put the stage back
            # the way an unaccepted frame found it. `h`/`ids` ride along so a rewind can re-drive the
            # accepted prefix without asking the previous stage to re-send it. Pushed onto the W-deep
            # ring — pipelined speculation streams the next frame before this one is judged, so the
            # ring may hold W un-judged snapshots at once (maxlen evicts the oldest, `commit` the
            # settled). A re-armed forward at the same start_pos (a rewind then a fresh frame) simply
            # pushes a newer checkpoint; `_seek` reads the newest that covers its target.
            self._spec_ckpts.append({"start_pos": start_pos, "s": s, "state": self._snapshot(),
                                     "h": h.clone(), "ids": ids.clone()})
        taps = {}
        with torch.no_grad():
            if start_pos == 0 or s == 1:
                out = self._run(h, ids, start_pos, taps)
            elif self._chunk_ok(s):
                out = self._run_chunk(h, ids, start_pos, taps)
            else:
                out = torch.cat([self._run(h[:, i:i + 1], ids[:, i:i + 1], start_pos + i, taps)
                                 for i in range(s)], dim=1)
        self._last_tap = {li: torch.cat(v, dim=1) for li, v in taps.items()}
        self._pos = start_pos + s
        return out

    def logits_all(self, h, full_logits=True):
        """Tail only: collapse the four HC streams, final norm, output head. -> [b, s, vocab] fp32.

        `Transformer.forward:922-923`. ParallelHead keeps its weight in fp32 and computes in fp32, so
        the logits come back fp32 whatever the payload was.

        `full_logits=False` is the reference's own flag (model.py:733): it slices to the last
        position BEFORE the vocab projection. Decode wants exactly that, and at V4's shape the
        difference is not cosmetic -- a 4096-token prefill's full logits are 2 GiB of fp32 at vocab
        129280. Default True because the verify path needs one row per speculated token."""
        if not self.tail:
            raise RuntimeError(f"v4 stage[{self.lo}:{self.hi}]: logits_all() on a non-tail stage")
        h = h.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            x = self.layers[-1].hc_head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
            return self.lm_head(self.norm(x), full_logits=full_logits)

    def tail_main_hidden(self):
        """The DSpark drafter's input for the LAST forward: [b, s, len(targets) * dim].

        `Transformer.forward:921,925` -- the mean over the four HC streams after each target layer,
        concatenated in layer order. Armed by `_dspark`; the greedy path records nothing and this
        raises rather than handing back a stale tensor from whenever it was last armed."""
        want = tuple(sorted(self.args.dspark_target_layer_ids))
        missing = [li for li in want if not self.lo <= li < self.hi]
        if missing:
            raise RuntimeError(
                f"v4 stage[{self.lo}:{self.hi}]: dspark target layers {missing} are not in this "
                f"stage's range. The drafter consumes all of {want} concatenated, so they must land "
                f"on ONE stage — at V4's shape that means the tail owns at least "
                f"{max(want) - min(want) + 1} layers.")
        if not self._dspark:
            raise RuntimeError(f"v4 stage[{self.lo}:{self.hi}]: tail_main_hidden() with _dspark off "
                               f"— arm the stage before the forward whose taps you want")
        if not self._last_tap:
            raise RuntimeError(f"v4 stage[{self.lo}:{self.hi}]: no taps recorded — forward() first")
        return torch.cat([self._last_tap[li] for li in want], dim=-1)

    # ---- weights ----

    def load(self, d=None):
        """Load this stage's layer range (and its boundary tensors) out of a CONVERTED checkpoint.

        The format is convert.py's output, which is what `generate.py:91` feeds straight into
        `load_model(Transformer, ...)` -- so the file's names ARE `Transformer.state_dict()`'s names
        and a layer range is a prefix scan, no per-model key table. Coverage is enforced by
        load_state_dict(strict=True): a range that matched nothing, or a layer missing a tensor,
        raises rather than serving random-init weights behind a valid receipt.

        Keys outside this stage's ranges are simply not looked at -- `mtp.*` in particular, which is
        step 4's drafter, not the tail's.

        The dtypes line up by construction: the Blocks were built from the SAME ModelArgs that
        convert.py's `--expert-dtype` was chosen for, so a GPU box constructs with the real config
        (dtype fp8, expert_dtype fp4) and loads the converted tensors as-is. A CPU parity box builds
        args.dtype='bf16' and needs a bf16 checkpoint to match; that is what the tests write."""
        d = d or V4_DIR
        wm = weight_map(d)
        for li in range(self.lo, self.hi):
            prefix = f"layers.{li}."
            names = [n for n in wm if n.startswith(prefix)]
            if not names:
                raise RuntimeError(f"v4 stage[{self.lo}:{self.hi}]: no tensor under {prefix!r} in "
                                   f"{d!r} — wrong checkpoint, or convert.py was never run")
            self.layers[li - self.lo].load_state_dict(
                {n[len(prefix):]: raw(n, d) for n in names}, strict=True)
        if self.embed_tokens is not None:
            self.embed_tokens.load_state_dict({"weight": raw("embed.weight", d)})
        if self.tail:
            self.norm.load_state_dict({"weight": raw("norm.weight", d)})
            self.lm_head.load_state_dict({"weight": raw("head.weight", d)})
            # Bare Transformer-level parameters, not a submodule -- there is no state_dict to be
            # strict with, so the shape check is by hand.
            for n in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                t, p = raw(n, d), getattr(self, n)
                if tuple(t.shape) != tuple(p.shape):
                    raise RuntimeError(f"v4 stage[{self.lo}:{self.hi}]: {n} is {tuple(t.shape)} in "
                                       f"the checkpoint, this config declares {tuple(p.shape)}")
                with torch.no_grad():
                    p.data.copy_(t)
        return self

    def __repr__(self):
        import v4_kernels_cpu
        kinds = "".join("W" if not L.attn.compress_ratio else
                        ("I" if L.attn.indexer is not None else "C") for L in self.layers)
        return (f"<V4Stage [{self.lo}:{self.hi}) {kinds} head={self.head} tail={self.tail} "
                f"{self.dtype} on {self.device} pos={self._pos} "
                f"kernels={v4_kernels_cpu.backend()} "
                f"dspark={'on' if self._dspark else 'off'} taps={list(self._tap_ids)} "
                f"spec={'on' if self._spec else 'off'}/{self._spec_depth} "
                f"graph={self._graph_mode if self._block_graphs is not None else 'off'} "
                f"fast_verify={f'<={self._chunk_cap}' if self._fast else 'off'} "
                f"moe={self._moe_status()} "
                f"ref_slim={self._ref_slim_status()} "
                f"levers={v4_levers.summary(self)}>")

    def _moe_status(self):
        """The WHOLE live MoE.forward chain, top first, and whether the bank layout took.

        OBSERVED, not declared: it reads the functions bound on the reference class, not the env
        flags, because the whole point of this line is to answer "did the lever fire?" on a live
        ring. The grouped install declines silently off-CUDA and the bank layout declines per layer,
        so `V4_MOE_GROUPED=1` alone proves nothing.

        THE CHAIN, NOT THE TOP OF IT. This used to classify the single bound function against the two
        markers it knew and fall through to the string "ref" for anything else -- so a ring running
        `V4_MOE_MULTI=1 V4_MOE_GROUPED=1`, where the live chain is multi -> grouped -> decode -> ref,
        printed `moe=ref` on all six stages while the grouped kernel was installed and serving every
        decode step. An operator chased that for a night. A partial observation is worse than none:
        it reads as evidence. `v4_levers.moe_chain` walks every link, so a lever added later appears
        instead of erasing the ones below it.

        AND `grouped/8` STILL DOES NOT PROVE THE KERNEL FIRED. The N counts layers that were BANKED
        at load, and a banked layer can still decline every decode step -- which is exactly what a
        6-layer profile caught: `grouped/6` in the repr, and four of the six on the reference path
        all night. Banked is a load-time fact; whether a layer grouped is a run-time one, and the
        run-time answer is `v4_moe_grouped.coverage(self.layers)`."""
        chain = v4_levers.moe_chain(ref())
        s = ">".join(chain) or "?"
        return f"{s}/{self._moe_banked}" if "grouped" in chain else s

    def _ref_slim_status(self):
        """Which v4_ref_slim overrides are live on the reference — same observed-not-declared rule."""
        M = ref()
        on = [n for n, f in (("indexer", M.Indexer.forward), ("noqat", M.act_quant))
              if getattr(f, "_v4_ref_slim", False)]
        return "+".join(on) if on else "off"


# ── partial CUDA graphs over a decode step ─────────────────────────────────────────────────────────

class _Graphlet:
    """Capture + replay ONE pure, fixed-shape function of tensors -- a hyper-connection or a norm.

    `fn` must be side-effect free and position/data-independent in STRUCTURE: it may read the layer's
    (constant) parameters by closure and it consumes only the tensors passed to `run`, which arrive
    through static input buffers, so the same captured kernels produce the right answer for any step's
    data. The captured outputs are allocated INSIDE the graph and alias its pool -- consume or clone
    them before the next replay of this same graphlet (the caller does, layer to layer).

    Warm-up runs `fn` three times on a side stream before the capture: a tilelang kernel (the fused
    hc_split_sinkhorn) autotunes and SYNCS on its first call, and a sync inside a capture is fatal."""

    def __init__(self, fn, examples):
        global _GRAPH_COUNT
        self.fn = fn
        self.ins = tuple(e.clone() for e in examples)          # static input buffers, fixed addresses
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                self.fn(*self.ins)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            outs = self.fn(*self.ins)
        self.outs = outs if isinstance(outs, tuple) else (outs,)
        torch.cuda.synchronize()
        _GRAPH_COUNT += 1

    def run(self, *args):
        for buf, a in zip(self.ins, args):
            buf.copy_(a)
        self.graph.replay()
        return self.outs if len(self.outs) > 1 else self.outs[0]


class _BlockGraphs:
    """The three decode-step island graphs for one Block, with `attn` and `ffn` eager between them.

    A Block's decode forward is, in order:
        residual = h
        x, post, comb = hc_pre(h, attn_params);  x = attn_norm(x)      << GRAPH g1
        x = attn(x, start_pos)                                          << EAGER (rotating KV, sparse)
        x = hc_post(x, residual, post, comb)
        residual = x
        x, post, comb = hc_pre(x, ffn_params);   x = ffn_norm(x)       << GRAPH g2 (with the hc_post)
        x = ffn(x, ids)                                                 << EAGER (data-routed MoE)
        x = hc_post(x, residual, post, comb)                           << GRAPH g3
    g1 ends at the attention input; g2 spans the attn hc_post through the ffn input; g3 is the ffn
    hc_post. Everything a graph captures is a pure function of its inputs (no start_pos, no KV, no
    routing) so the shape is a fixed b=1,s=1 and one capture serves every decode position.

    Capture happens ALL AT ONCE on the first decode step, from zero-filled example tensors of the
    known shapes, BEFORE the real `attn`/`ffn` run -- so a capture that fails leaves the per-stage KV
    state untouched and the step falls back to a whole-Block eager call with nothing double-advanced."""

    def __init__(self, L, stage):
        self.L = L
        self.st = stage
        self.g = None
        self.eager = False

    def _g1_fn(self, h):
        y, post, comb = self.L.hc_pre(h, self.L.hc_attn_fn, self.L.hc_attn_scale, self.L.hc_attn_base)
        return self.L.attn_norm(y), post, comb

    def _g2_fn(self, attn_out, residual, post_a, comb_a):
        x2 = self.L.hc_post(attn_out, residual, post_a, comb_a)
        y, post_f, comb_f = self.L.hc_pre(x2, self.L.hc_ffn_fn, self.L.hc_ffn_scale, self.L.hc_ffn_base)
        return self.L.ffn_norm(y), x2, post_f, comb_f

    def _g3_fn(self, ffn_out, residual, post_f, comb_f):
        return self.L.hc_post(ffn_out, residual, post_f, comb_f)

    def _build(self):
        """Derive every graph's example inputs from a zero hidden state (the island fns are stateless,
        so running them for shapes touches nothing), then capture all three."""
        a = self.st.args
        dt, dev = self.st.dtype, self.st.device
        h = torch.zeros(1, 1, a.hc_mult, a.dim, dtype=dt, device=dev)
        with torch.no_grad():
            attn_in, post_a, comb_a = self._g1_fn(h)
            attn_out = torch.zeros_like(attn_in)               # attn returns the attn-input's shape
            ffn_in, x2, post_f, comb_f = self._g2_fn(attn_out, h, post_a, comb_a)
            ffn_out = torch.zeros_like(ffn_in)                 # ffn returns the ffn-input's shape
        self.g = (_Graphlet(self._g1_fn, (h,)),
                  _Graphlet(self._g2_fn, (attn_out, h, post_a, comb_a)),
                  _Graphlet(self._g3_fn, (ffn_out, x2, post_f, comb_f)))

    def run(self, h, ids, start_pos):
        """One graphed decode step for this Block, or a whole-Block eager fallback. Never raises."""
        global _GRAPH_SKIPPED
        if self.eager:
            return self.L(h, start_pos, ids)
        if self.g is None:
            if _GRAPH_COUNT + 3 > V4_GRAPH_MAX:
                self.eager, _GRAPH_SKIPPED = True, _GRAPH_SKIPPED + 3
                print(f"[v4] graph budget V4_GRAPH_MAX={V4_GRAPH_MAX} spent — layer "
                      f"{self.L.layer_id} stays eager", flush=True)
                return self.L(h, start_pos, ids)
            try:
                self._build()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                torch.cuda.synchronize()
                self.eager, self.g, _GRAPH_SKIPPED = True, None, _GRAPH_SKIPPED + 3
                print(f"[v4] graph capture failed for layer {self.L.layer_id}: "
                      f"{type(e).__name__}: {e} — layer stays eager", flush=True)
                return self.L(h, start_pos, ids)
        g1, g2, g3 = self.g
        attn_in, post_a, comb_a = g1.run(h)
        attn_out = self.L.attn(attn_in, start_pos)
        ffn_in, x2, post_f, comb_f = g2.run(attn_out, h, post_a, comb_a)
        ffn_out = self.L.ffn(ffn_in, ids)
        return g3.run(ffn_out, x2, post_f, comb_f)


def _selftest(lo, hi, d):
    """Build a stage, run a prefill and a decode step through it, print what crossed the boundary.

    Falls back to v4_ref_cpu's toy ModelArgs when there is no checkpoint at `d`, so the mechanics are
    exercisable on any box; with a real checkpoint it loads the range for real."""
    have = os.path.exists(f"{d}/config.json") and glob.glob(os.path.join(d, "model*-mp*.safetensors"))
    if have:
        args = config(d)
    else:
        import v4_ref_cpu
        args = v4_ref_cpu.cpu_args()
        print(f"[v4] no converted checkpoint at {d!r} — running at v4_ref_cpu.cpu_args() scale "
              f"(n_layers={args.n_layers} dim={args.dim}) with random weights", flush=True)
        hi = min(hi, args.n_layers)
    st = Stage(lo, hi, args, head=(lo == 0), tail=(hi == args.n_layers), device="cpu")
    if have:
        st.load(d)
    else:
        # A Stage is not an nn.Module (k3_stage's shape -- it owns modules, it is not one), so
        # init_random gets a throwaway container holding exactly what this stage owns.
        holder = torch.nn.Module()
        holder.layers = st.layers
        if st.embed_tokens is not None:
            holder.embed_tokens = st.embed_tokens
        if st.tail:
            holder.norm, holder.lm_head = st.norm, st.lm_head
            for n in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                setattr(holder, n, getattr(st, n))
        v4_ref_cpu.init_random(holder, 0)
    print(st, flush=True)
    ids = torch.randint(0, args.vocab_size, (1, 9))
    h = st.embed(ids) if st.head else torch.randn(1, 9, args.hc_mult, args.dim, dtype=st.dtype)
    h = st.forward(h, ids, 0)
    per_tok = h.shape[2] * h.shape[3] * h.element_size()
    print(f"[v4] prefill h {tuple(h.shape)} — {per_tok / 1024:.1f} KiB/token on the wire "
          f"({h.shape[2]}x a plain transformer's hidden state)", flush=True)
    nxt = torch.randint(0, args.vocab_size, (1, 1))
    h = st.forward(h[:, -1:], nxt, 9)
    if st.tail:
        print(f"[v4] decode logits {tuple(st.logits_all(h).shape)}", flush=True)
    print(f"[v4] {st}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=V4_DIR)
    ap.add_argument("--layers", type=int, nargs=2, default=[0, 4], metavar=("LO", "HI"))
    a = ap.parse_args()
    _selftest(a.layers[0], a.layers[1], a.dir)
