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
                 device=None, dtype=None, spec_depth=None):
        self.lo, self.hi = lo, hi
        self.args = args if args is not None else config()
        self.device = device or dev
        self.dtype = dtype or getattr(torch, V4_DTYPE)
        self.head, self.tail, self._dspark = head, tail, dspark
        M = ref()
        self.globals = _set_globals(M, self.args)
        a = self.args
        if not 0 <= lo < hi <= a.n_layers:
            raise RuntimeError(f"v4 stage[{lo}:{hi}) is not a range inside 0..{a.n_layers}")
        # `with torch.device(...)` + the reference's own set_dtype contextmanager is generate.py's
        # construction environment (generate.py:77,87) reproduced exactly. Both matter: the dtype
        # decides what the bare `torch.empty`/`torch.zeros` parameters and kv buffers come out as,
        # the device keeps a 158 GiB model from being built on the host and then copied.
        with torch.device(self.device), M.set_dtype(self.dtype):
            self.layers = torch.nn.ModuleList([M.Block(li, a) for li in range(lo, hi)])
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
        self.reset()

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
        with torch.no_grad():
            for i in range(h.shape[1]):
                self._run(h[:, i:i + 1], ids[:, i:i + 1], start_pos + i, {})
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
        """One pass of this stage's layers over `h` at `start_pos`, collecting any owned taps."""
        for li, L in zip(range(self.lo, self.hi), self.layers):
            h = L(h, start_pos, ids)
            if self._dspark and li in self._tap_ids:
                taps.setdefault(li, []).append(h.mean(dim=2).detach().clone())
        return h

    def forward(self, h, ids, start_pos):
        """Run this stage's layers. BOTH `h` and `ids` are inputs -- see the module docstring.

        Prefill goes through the reference's own multi-token branch. A chunk at `start_pos > 0` is
        replayed one position at a time, because the reference's decode branch writes
        `kv_cache[:, start_pos % win]` and `kv_state[:, start_pos % ratio]` from a squeezed
        `seqlen == 1` (model.py:353,362,535) and would silently write only the chunk's last token
        otherwise. Exact, not approximate: HC mixing is per-position, and the loop advances the KV
        and compressor state in exactly the order sequential decode would.

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
                f"spec={'on' if self._spec else 'off'}>")


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
