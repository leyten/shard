"""DeepSeek-V4-Flash's OWN DSpark speculator, driven from the tail. Step 4 of the V4 engine.

The K3 engine had to build its drafter from scratch: that model shipped weights and a paper and no
drafter code. V4 ships all three -- the MTP stages ARE `DSparkBlock` in the vendored model.py, their
weights ARE in the release (`mtp.0/1/2.*`), and `Transformer.forward_spec` IS the call sequence. So
this file RENTS the drafter the way v4_stage rents the layers -- it builds the `n_mtp_layers`
DSparkBlocks exactly as `Transformer.__init__` does (model.py:898-904), aliases their `embed`/`head`
to the tail's own modules exactly as the reference does, loads `mtp.*` out of the converted
checkpoint, and mirrors `forward_spec` line for line. It reimplements no attention, no Markov head,
no confidence head, and no sampling. What it adds is the one thing the reference does not have: a
POSITION-DISCIPLINED accept/advance protocol, because DeepSeek's own loop only ever decodes one
token and a ring commits a variable-length prefix per round.

WHY THE DRAFTER LIVES ON THE TAIL AND NOT ON THE COORDINATOR
`forward_spec` consumes `main_hidden` -- the mean over the four HC streams after each
`dspark_target_layer_ids` layer, concatenated (`Transformer.forward:920-925`, and
`Stage.tail_main_hidden()`). At V4's shape that is 3 x 4096 x 2 B = 24 KiB per token, and only the
tail holds it. Streaming it to a coordinator to draft there would cost more than the draft saves, so
the tail drafts LOCALLY and returns the block with the token. That is the whole reason v4_pipe has a
`TAIL_DRAFTER` seam instead of a coordinator-side drafter.

── THE DECODE PROTOCOL ───────────────────────────────────────────────────────────────────────────
Positions are the thing to get right; every bug in a speculator is an off-by-one in this paragraph.

`Transformer.forward(x_i, i)` returns (token predicted for position i+1, logits, main_hidden AT
position i). `forward_spec(input_ids, main_hidden, i)` then does two things at once:

  ADVANCE  the mtp KV cache by exactly ONE position: `DSparkAttention` writes `main_kv` (projected
           from main_hidden at position i) into slot `i % window_size` (model.py:783). The decode
           branch is hard `seqlen == 1` -- a multi-position main_hidden lands only its last row.
  DRAFT    a block of `dspark_block_size` tokens whose queries sit at positions i+1 .. i+block_size
           (model.py:772). The returned `output_ids` is [b, block_size+1]: column 0 is the token you
           passed in (which lives at position i+1) and columns 1.. are the block_size NEW drafts,
           for positions i+2 .. i+block_size+1.

So `input_ids` is the token at position i+1 and `main_hidden` is the tap at position i. One call =
one committed position + one fresh block. At `start_pos == 0` the same function is a PREFILL
instead: it builds the window from the whole prompt's main_hidden and returns None.

THE FIRST ROUND. After the ring prefills the prompt (positions 0..P-1) the tail holds main_hidden
for all of them, and `prefill()` mirrors model.py's `__main__` exactly: one `forward_spec(first
predicted token, main_hidden, 0)`, which covers 0..P-1, so `_pos = P-1`. There is DELIBERATELY no
draft yet -- the first verify chunk is `[cur]` ALONE (one token, no drafts, which
`plan_verify_round` handles as the degenerate `drafts == []` case), its tap gives main_hidden at
position P, and the `advance_and_draft` that follows produces the first real block.

That costs exactly one g=1 round per generation, and it is not because the alternative corrupts
anything: calling the decode branch at `start_pos = P-1` with the prompt's LAST tap re-writes slot
`(P-1) % win` with a bit-identical value (measured against the reference: the cache does not move)
and drafts precisely the block the first chunk wants. It is skipped because it is a call the
reference's own loop never makes, and because its idempotence rests on an invariant nothing here
enforces -- that the single tap row handed to it is the same one the prefill already folded in. If
the first round ever shows up in a measurement, that is the thing to revisit deliberately, with the
oracle grading it, rather than a shortcut to take on the way past.

THE STEADY ROUND. The ring verifies chunk `[cur] + drafts` at positions p..p+g. The tail forwards it
and answers with its own greedy token at every chunk position; `plan_verify_round` turns that into
(n accepted, committed). Positions p..p+n are now COMMITTED, so the mtp cache advances by n+1
single-position calls -- call j feeds main_hidden at position p+j with the token at position p+j+1
(an accepted draft for j<n, the correction `cur` for j=n). Only the LAST call's block is kept; the
intermediate blocks are discarded, which costs `n` tiny mtp forwards and buys call-for-call fidelity
with the reference. `_pos` is asserted to move by exactly one per call: a gap or an overlap is an
upstream protocol bug and raises rather than drafting off a cache that means something else.

── WHY THE MTP CACHE NEVER NEEDS A ROLLBACK ──────────────────────────────────────────────────────
`DSparkAttention.forward` writes exactly one slot, `kv_cache[:bsz, start_pos % win] = main_kv`, and
main_kv comes from main_hidden -- a COMMITTED position. The block's own K/V is concatenated onto the
cache for the attention and thrown away (model.py:784); nothing speculative is ever stored. So a
rejected draft leaves the drafter's state untouched by construction, and `reset()` exists for a new
sequence, not for a rollback. (tests/test_v4_dspark.py::test_cache_never_speculative is the proof:
the slots the draft block's positions WOULD occupy stay zero.) That is the opposite of the main
stage, whose window ring and compressor accumulators DO need `Stage._snapshot()/_restore()`, which
`Stage._seek` spends on the round after a partial accept. The asymmetry is the reason `RingDrafter`
advances the drafter over the committed prefix and never un-advances it: on the stage side a
rejected chunk has to be rolled back, on this side it was never recorded.

── THE NUMERICS RULE STEP 5 MUST READ ────────────────────────────────────────────────────────────
The tail's VERIFY logits must be computed PER POSITION, with the same GEMM shape greedy decode uses
(`logits_all(h[:, j:j+1], full_logits=False)`, M = b rows through `ParallelHead`), NOT as one
[b, s, vocab] batch. Step 2 measured it: fp32 GEMM reassociation at a different M shifts logits by
~4e-7, which is invisible until an argmax near-tie flips and the speculated stream silently stops
being the greedy stream. Speculation is only lossless if the verify path is bit-identical to the
path it claims to replace. The drafter's own head is already per-position in the way that matters --
`forward_head` samples `logits[:, i]` one position at a time (model.py:871), and its GEMM is over a
CONSTANT [b, block_size, dim] every round -- so the drafter is round-to-round deterministic, but ONLY
because its temperature is pinned to greedy in the constructor. Read that comment; the shipped
config would otherwise have it drawing from a Gumbel-max. The rule itself is about the MAIN model's
logits, which this file does not compute and v4_pipe does.

── TWO THINGS THE CPU SUITE STRUCTURALLY CANNOT CATCH ────────────────────────────────────────────
1. THE SHIPPED CONFIG HAS NO `max_seq_len`. config.json declares neither it nor `max_batch_size`,
   so `v4_stage.config()` takes ModelArgs' defaults (4096 / 4), though generate.py:85 sets 64k for
   its own interactive path. The `freqs_cis` table, every kv_cache and this drafter's end-of-context
   guard are all sized off that number, so a long-context ring would stop drafting at ~4090. Step 5
   fills the hole where a ring can see it -- `v4_pipe`'s V4_MAX_SEQ / V4_MAX_BATCH (8192 / 1) supply
   both fields when the checkpoint's own config is silent -- and `RingDrafter` degrades to greedy at
   whatever limit ends up set, rather than raising inside the tail's serve loop.
2. `get_dspark_topk_idxs` (model.py:744) builds its index with a bare `torch.arange` on the AMBIENT
   default device. Construction here happens inside `with torch.device(...)`, but that call is at
   FORWARD time, and generate.py gets away with it only because it does a global
   `torch.set_default_device("cuda")` (generate.py:78). The same pattern is already in the main
   attention (model.py:513), so this is inherited, not introduced -- but a GPU bring-up that skips
   the global default device will hand a CPU tensor to a CUDA kernel, and a CPU-only test suite can
   never see it.

── WHAT IS NOT HERE ──────────────────────────────────────────────────────────────────────────────
The confidence head's USE (it is returned raw; gating draft length on it is a measurement, not a
guess), a ragged batch (one accept length per round, see `advance_and_draft` and `RingDrafter`), and
CUDA graphs. Also note the cost this adds to the tail: `n_mtp_layers` DSparkBlocks are full Blocks
with their own MoE -- 3 of them at V4's shipped config, i.e. the tail carries ~3 layers of extra
weights that the placement planner has to budget for, on top of the dspark target layers all having
to land on it.

self-test:  python3 phase0/v4_dspark_draft.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _v4():
    """phase0/v4_stage, imported on FIRST USE, not at module scope (v4_pipe's LAZY MODEL IMPORT rule).

    `plan_verify_round` is pure python and both ends of the ring need it -- the tail to know how far
    to advance the drafter, v4_pipe's coordinator to know what to emit -- and there must be exactly
    ONE implementation of an accept rule, or the two would drift and desynchronise silently. So the
    coordinator imports THIS module, and this module must not drag the checkpoint loader (and its
    safetensors dependency) into a process that only speaks protocol."""
    import v4_stage
    return v4_stage


# THE INFERENCE-TIME BLOCK WIDTH (V4_DSPARK_BLOCK, 0 = the trained dspark_block_size). One draft
# call proposes `block_size` tokens from ONE tap, and the pipelined in-flight cap is block+1 — the
# lockstep docs/V4_MULTIBLOCK_VERDICT.md §2 proves cannot be chained past. Width is the one axis
# that cap leaves open: no weight tensor is dimensioned by `block_size` and every vendored shape is
# symbolic in it (forward_embed's noise ids, the block rows DSparkAttention concatenates, the
# freqs_cis slice, forward_head's Markov loop), so a wider block is pure configuration on the same
# math. THE HONEST PRICE, named where the knob is read: `get_dspark_topk_idxs` gives every block
# query the SAME index set — the block attends to itself BIDIRECTIONALLY — so extra noise slots
# perturb the trained slots' logits too. Widening is therefore not "the old block plus free extras":
# acceptance may move at EVERY depth, which only a real ring can measure (`accept_by_depth` under
# the widened width against the trained width, same prompts). Opt-in, default OFF, tail-only; the
# coordinator adapts to whatever length the reply carries and needs no flag.
V4_DSPARK_BLOCK = int(os.environ.get("V4_DSPARK_BLOCK", "0") or 0)

# THE TREE GATE (V4_DRAFT_TOP2). Branching speculation — several candidate continuations from one
# tap — is priced in docs/V4_TREE_VERDICT.md at +18.3% x beta for the best buildable shape, where
# beta is the RESCUE RATE: how often, at a cancel, the model's committed token is the drafter's own
# RUNNER-UP at exactly the slot that missed. beta is the whole go/no-go and nothing measures it
# today, so this lever ships the runner-up column on the existing linear round: `d2[i]` is the
# second-highest-logit token at the same head slot that produced `draft[i]` (`forward_head` returns
# the full per-slot logits, model.py:863, with the main path's Markov bias already applied — and at
# a mismatch, the missed slot's predecessors are all committed, so its biased runner-up IS the
# candidate a comb tree would have had in flight). The coordinators count, at each cancel, whether
# `d2` named the committed token (`rescue_by_depth`). Reply metadata only — no frame, block, accept
# rule or committed token changes — so losslessness is untouched by construction, and the tests pin
# it on the real socket ring anyway. Opt-in, default OFF, tail-only, greedy drafter only (at
# temperature > 0 the sampled draft need not be the top-1, and "runner-up" stops meaning anything).
V4_DRAFT_TOP2 = os.environ.get("V4_DRAFT_TOP2", "0") not in ("", "0")

# The two keys a DSparkBlock's state_dict has but a converted checkpoint deliberately does not.
# `mtp[k].embed`/`mtp[k].head` are ASSIGNED the main model's modules (model.py:903-904), so they are
# registered submodules pointing at storage that is already in the file under `embed.weight` /
# `head.weight`; safetensors refuses to write one tensor twice, and convert.py:91 skips them.
ALIAS_KEYS = ("embed.weight", "head.weight")


def second_choices(tail):
    """The drafter's runner-up token per slot of the block it JUST produced. -> [g] ints, or None.

    `last_spec[1]` is `forward_head`'s logits, [b, block_size, vocab]: slot i's row produced
    `draft[i]` (= output_ids[:, i+1]) by argmax AFTER the Markov bias landed in-place
    (model.py:869-871), so its top-1 is the draft itself and its top-2 is the head's own second
    choice for the same position under the same conditioning — the branch candidate the tree gate
    counts. Both the reference and the V4_DSPARK_FAST path leave `last_spec` holding the kept
    block's real triple, so this reads either one.

    None when there is nothing sound to read: no block yet, or a sampling drafter (top-1 is then not
    the draft, so "runner-up" would be a lie with plausible-looking values)."""
    if tail.last_spec is None or tail.temperature != 0.0:
        return None
    return tail.last_spec[1][0].topk(2, dim=-1).indices[:, 1].tolist()


def plan_verify_round(drafts, replies):
    """The accept rule, as one pure function both the pipe and the tests use. -> (n, committed).

    `drafts` are the g tokens the drafter proposed for chunk positions p+1..p+g. `replies` are the
    model's own greedy tokens for those same positions, one per chunk position -- so `replies[j]` is
    what the model says belongs at p+j+1, and there are g+1 of them because the chunk is
    `[cur] + drafts` and the tail answers at every position including the last.

    Longest matching prefix, then the correction: the first j where the draft and the model disagree
    ends the accept, and `replies[n]` is committed in its place. A FULL accept (n == g) commits
    `replies[g]` too -- the free bonus token, which the model produced from the last accepted draft
    and which cost nothing extra to compute. `len(committed) == n + 1` either way, and every token in
    it came out of the MAIN model, which is what makes the round lossless.

    The degenerate `drafts == []` (round 0, and any round the drafter is skipped) falls out of the
    same rule: n = 0, committed = [replies[0]]. That is plain greedy decode.

    BOTH ENDS RUN THIS, which is why it is a function and not a loop inside the coordinator. The
    coordinator runs it to decide what to emit. The TAIL has to run it too, on the same round, to
    know how far to advance the drafter -- it holds the chunk it just forwarded (so it has the
    drafts) and its own per-position replies (so it has the answers), and it must advance before it
    can return the next block in the same reply. Two implementations of "longest matching prefix"
    that ever disagreed would desynchronise the mtp cache from the committed stream silently: the
    drafts would keep looking plausible while conditioning on a history the ring never took."""
    # int(), not list(): the drafts arrive as a row of `advance_and_draft`'s tensor and the replies
    # as python ints off the tail's argmax, and a list of 0-dim tensors mixed with ints compares and
    # serialises in ways nobody wants on a wire.
    drafts, replies = [int(t) for t in drafts], [int(t) for t in replies]
    if len(replies) != len(drafts) + 1:
        raise RuntimeError(
            f"v4 dspark: {len(drafts)} drafts need {len(drafts) + 1} replies (one per chunk "
            f"position, the chunk being [cur] + drafts), got {len(replies)}")
    n = 0
    for d, r in zip(drafts, replies):
        if d != r:
            break
        n += 1
    return n, drafts[:n] + [replies[n]]


class DSparkTail:
    """The `n_mtp_layers` DSpark stages, hanging off a dspark-armed tail Stage.

    Contract, consumed by v4_pipe's TAIL_DRAFTER seam in step 5:
        prefill(pred_ids, main_hidden)              -> _pos          once, after the ring's prefill
        advance_and_draft(ids_seq, main_hidden_seq) -> (drafts, confidence)   once per verify round
        reset()                                     -> new sequence
        load(dir)                                   -> mtp.* off a converted checkpoint
    `last_spec` holds the reference's raw `(output_ids, logits, confidence)` triple from the last
    draft, for a caller that wants the drafter's own logits or its anchor column.

    It takes the STAGE, not a ModelArgs, for two reasons: the tail owns the `embed`/`head` modules
    the DSparkBlocks alias (which is why the tail is built `dspark=True` -- a tail otherwise has no
    use for an embedding), and `Stage.__init__` has already assigned model.py's five module globals
    from these exact args. Constructing DSparkBlocks against un-set globals would build bf16 weights
    for an fp8 checkpoint; see v4_stage's docstring for the whole trap.

    `temperature` is the DRAFTER's, and it defaults to greedy on purpose -- see the constructor."""

    def __init__(self, stage, temperature=0.0):
        M = _v4().ref()
        a = stage.args
        if not stage.tail:
            raise RuntimeError("v4 dspark: the drafter runs on the TAIL — main_hidden is the tap "
                               "over the last layers and no other stage holds it")
        if not a.dspark_block_size:
            raise RuntimeError("v4 dspark: this config has dspark_block_size=0, i.e. no MTP stages. "
                               "Serve greedily instead of building a drafter that cannot draft.")
        if stage.embed_tokens is None or getattr(stage, "lm_head", None) is None:
            raise RuntimeError("v4 dspark: the tail has no embedding — DSparkBlock.embed/.head ALIAS "
                               "the main model's modules (model.py:903), so build the tail Stage "
                               "with dspark=True")
        self.stage = stage
        self.args = a
        self.device, self.dtype = stage.device, stage.dtype
        self.block_size = a.dspark_block_size
        self.hidden_dim = len(a.dspark_target_layer_ids) * a.dim
        # `torch.device` + the reference's own set_dtype is generate.py's construction environment,
        # the same one Stage builds its Blocks in: the dtype decides what the bare torch.empty
        # parameters and the kv_cache zeros come out as, the device keeps them off the host.
        with torch.device(self.device), M.set_dtype(self.dtype):
            self.mtp = torch.nn.ModuleList(
                [M.DSparkBlock(a.n_layers + k, a) for k in range(a.n_mtp_layers)])
        for blk in self.mtp:
            blk.embed = stage.embed_tokens        # registered submodules, not plain attributes --
            blk.head = stage.lm_head              # hence ALIAS_KEYS in every state_dict below
            # THE DRAFTER SAMPLES UNLESS YOU STOP IT, AND THE SHIPPED CONFIG DOES NOT.
            # `DSparkBlock.__init__` copies `args.temperature` (model.py:828) and `forward_head`
            # draws with it (model.py:871). V4's config.json declares NO temperature key, so
            # `v4_stage.config()` gets the dataclass default of 1.0 and every draft block would come
            # out of a Gumbel-max off the process-global RNG -- nondeterministic, and drawn against
            # a verifier that is greedy by default (v4_pipe.sample_token). Acceptance would fall to
            # roughly chance and the drafter would buy nothing. generate.py never hits this because
            # it assigns args.temperature from its own CLI flag before building the model
            # (generate.py:82); a Stage built straight from config.json does not.
            # So it is pinned HERE, explicitly, and defaults to greedy: match the verifier. A caller
            # running a sampled verifier can raise it, but lossless speculation at temperature > 0
            # needs the rejection-sampling accept rule, not this one.
            blk.temperature = float(temperature)
        self.temperature = float(temperature)
        # THE WIDTH OVERRIDE, applied where every consumer reads it: `self.block_size` plans the
        # advance/cliff guards, each vendored block's own `block_size` shapes forward_embed and the
        # forward_head loop, and the fast/MoE installers capture whatever is live here. Set BEFORE
        # any weight loads or rebinds so no path ever sees two widths. The bound is well inside
        # v4_pipe's `_SPEC_POS_MARGIN` (64), the overshoot a serial dspark reset declares for the
        # chunk `[cur] + block` — a block wider than that could rope a verify chunk past the
        # horizon the job promised its stages.
        if V4_DSPARK_BLOCK:
            if not 1 <= V4_DSPARK_BLOCK <= 32:
                raise ValueError(
                    f"v4 dspark: V4_DSPARK_BLOCK={V4_DSPARK_BLOCK} — the inference-time block "
                    f"width must be 1..32 (0/unset = the trained width, "
                    f"{a.dspark_block_size} in this checkpoint)")
            self.block_size = int(V4_DSPARK_BLOCK)
            for blk in self.mtp:
                blk.block_size = self.block_size
        self.mtp.eval()
        self.alias_missing = []
        self._pos = None
        self.last_spec = None
        self.reset()

    # ---- state ----

    def reset(self):
        """Drop the sequence, IN PLACE. One buffer, unlike the main stage's four.

        `compress_ratios` is asserted 0 at the DSpark layer ids (`DSparkAttention.forward`'s first
        line), so these blocks have no Compressor and no Indexer -- no kv_state, no score_state, no
        -inf accumulator to restore. `Attention.__init__` sizes their kv_cache to `window_size`
        alone and zeroes it, so zeroing is the constructor state exactly."""
        with torch.no_grad():
            for blk in self.mtp:
                blk.attn.kv_cache.zero_()
        self._pos = None
        self.last_spec = None

    @property
    def pos(self):
        """The last main-model position the mtp cache has seen, or None before a prefill.

        The next `advance_and_draft` must start at `pos + 1`; step 5 asserts against it."""
        return self._pos

    # ---- the reference's own call sequence ----

    def _forward_spec(self, input_ids, main_hidden, start_pos):
        """`Transformer.forward_spec` (model.py:929-936) without a Transformer.

        Six lines, mirrored rather than called, because instantiating a Transformer to reach them
        would allocate the whole main stack -- 158 GiB at V4's real shape, on a box that already
        holds the tail's layers. The ModuleList is ours; every line inside it is DeepSeek's."""
        h, main_x = self.mtp[0].forward_embed(main_hidden, input_ids)
        for layer in self.mtp:
            h = layer(h, start_pos, input_ids, main_x)
        if start_pos == 0:
            return None
        return self.mtp[-1].forward_head(h, input_ids)

    # ---- input normalisation ----

    def _hidden(self, x, want_s=None):
        """main_hidden -> [b, s, len(targets) * dim] on this drafter's device/dtype."""
        t = torch.as_tensor(x).to(device=self.device, dtype=self.dtype)
        if t.dim() != 3 or t.shape[-1] != self.hidden_dim:
            raise RuntimeError(
                f"v4 dspark: main_hidden is {tuple(t.shape)}, expected [b, s, {self.hidden_dim}] "
                f"— that is Stage.tail_main_hidden(), the {len(self.args.dspark_target_layer_ids)} "
                f"target-layer taps concatenated")
        if t.shape[1] == 0:
            raise RuntimeError("v4 dspark: main_hidden has no positions — a prefill needs the whole "
                               "prompt's taps and an advance needs one per committed position")
        if want_s is not None and t.shape[1] != want_s:
            raise RuntimeError(f"v4 dspark: main_hidden has {t.shape[1]} positions, the token ids "
                               f"have {want_s} — one tap per committed position")
        if t.shape[0] > self.args.max_batch_size:
            raise RuntimeError(f"v4 dspark: batch {t.shape[0]} exceeds max_batch_size "
                               f"{self.args.max_batch_size} — the mtp kv_cache is sized for it")
        return t

    def _flat_ids(self, x):
        """A single token per batch row -> [b] long, which is what `forward_embed` indexes with."""
        t = torch.as_tensor(x, dtype=torch.long, device=self.device)
        if t.dim() == 0:
            t = t.view(1)
        elif t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if t.dim() != 1:
            raise RuntimeError(f"v4 dspark: token ids {tuple(t.shape)}, expected [b] or [b, 1]")
        return t

    def _seq_ids(self, x):
        """A run of committed tokens -> [b, n] long. A 1-D input is one batch row, not one column."""
        t = torch.as_tensor(x, dtype=torch.long, device=self.device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if t.dim() != 2 or t.shape[1] == 0:
            raise RuntimeError(f"v4 dspark: committed ids {tuple(t.shape)}, expected [b, n>0] "
                               f"(a 1-D input is read as one batch row's run)")
        return t

    # ---- the protocol ----

    def prefill(self, pred_ids, main_hidden):
        """Build the mtp window from the whole prompt. -> the position the cache now stands at.

        `pred_ids` is the token the main model predicted from the LAST prompt position (i.e. the
        token at position P, which the ring has not fed anywhere yet) and `main_hidden` is the tap
        for positions 0..P-1. This is model.py's `__main__` line for line: at `start_pos == 0` the
        DSpark blocks run only their attention, which writes the window and returns `x` untouched,
        and `forward_spec` returns None -- there is no draft here (see the module docstring's FIRST
        ROUND paragraph for why not, and why that is deliberate)."""
        if self._pos is not None:
            raise RuntimeError(f"v4 dspark: already prefilled to position {self._pos} — reset() "
                               f"before starting another sequence")
        mh = self._hidden(main_hidden)
        ids = self._flat_ids(pred_ids)
        if ids.shape[0] != mh.shape[0]:
            raise RuntimeError(f"v4 dspark: {ids.shape[0]} token ids against a batch of "
                               f"{mh.shape[0]} in main_hidden")
        with torch.no_grad():
            out = self._forward_spec(ids, mh, 0)
        if out is not None:
            raise RuntimeError("v4 dspark: forward_spec at start_pos=0 must be a prefill and return "
                               "None — the vendored reference changed under us")
        self._pos = mh.shape[1] - 1
        return self._pos

    def advance_and_draft(self, input_ids_seq, main_hidden_seq, start_pos):
        """Commit a run of positions and draft the next block. -> (drafts [b, g], confidence [b, g]).

        `main_hidden_seq[:, j]` is the tap at position `_pos + 1 + j` and `input_ids_seq[:, j]` is
        the committed token at position `_pos + 2 + j` -- i.e. the pairing `forward_spec` wants,
        shifted by one, because a position's tap predicts the NEXT token. `start_pos` is the
        absolute position of column 0 and is REQUIRED, not inferred: it is checked against the
        drafter's own cursor, and a caller that cannot say which position it is advancing over is a
        caller that does not know. A gap means positions were never fed and an overlap means they
        were fed twice, and either one makes every subsequent draft attend to the wrong history.

        THE RUN IS THE COMMITTED PREFIX, NOT THE CHUNK. Feeding the whole verify chunk (`taps`
        instead of `taps[:, :n+1]`) passes every check this function can make -- a full accept
        commits g+1 positions, so length alone cannot tell the two apart -- and drafts off a history
        the ring rejected. The NEXT round's cursor check catches it, one round late. Slice to the
        accept; `plan_verify_round` returns the n that says where.

        One reference call per position, in order, keeping only the last block. The intermediate
        blocks are thrown away on purpose: they are the drafts for histories the ring already moved
        past, and re-deriving them is a few small GEMMs against the alternative of a bespoke
        multi-position advance that would have to be proved equal to the reference's own.

        `confidence` is column-aligned with the drafts -- `confidence[:, i]` is the head's fp32 score
        for `drafts[:, i]`, because `forward_head` builds it from the hidden state at draft query i
        and the Markov embedding of the token that query was conditioned on (model.py:866-873). It is
        returned raw and unused: gating draft length on it is a measurement step 5 can make, not a
        threshold to invent here.

        AT b > 1 THE BATCH ADVANCES IN LOCKSTEP. The mtp cache is indexed by POSITION, not per row,
        so one call moves every row by the same n -- which is only correct if every row accepted the
        same number of drafts. `plan_verify_round` is per sequence; run it once per row and, until
        step 5 has a ragged path, require the answers to agree. Nothing here can detect the
        violation, because the tensors are the right shape either way."""
        if self._pos is None:
            raise RuntimeError("v4 dspark: advance before prefill — the mtp window is empty, so the "
                               "first draft would attend to zeros. Call prefill() after the ring's.")
        ids = self._seq_ids(input_ids_seq)
        n = ids.shape[1]
        mh = self._hidden(main_hidden_seq, want_s=n)
        if ids.shape[0] != mh.shape[0]:
            raise RuntimeError(f"v4 dspark: ids batch {ids.shape[0]} against main_hidden batch "
                               f"{mh.shape[0]}")
        if n > self.block_size + 1:
            raise RuntimeError(
                f"v4 dspark: an advance over {n} positions, but one round can commit at most "
                f"{self.block_size + 1} (g={self.block_size} accepted drafts plus the bonus). This "
                f"is the committed PREFIX of one verify round, not a whole chunk or several rounds.")
        if start_pos != self._pos + 1:
            raise RuntimeError(
                f"v4 dspark: advance at {start_pos} but the mtp cache stands at {self._pos}, so the "
                f"next position is {self._pos + 1}. A "
                f"{'gap' if start_pos > self._pos + 1 else 'overlap'} is an upstream protocol bug: "
                f"the drafter must be advanced over exactly the COMMITTED positions of every round, "
                f"no more and no less.")
        end = self._pos + n + self.block_size + 1
        if end > self.args.max_seq_len:
            raise RuntimeError(f"v4 dspark: a block drafted here would rope out to position {end}, "
                               f"past max_seq_len {self.args.max_seq_len} — stop drafting before "
                               f"the context limit, not inside the reference's freqs_cis slice")
        out = None
        with torch.no_grad():
            for j in range(n):
                pos = self._pos + 1
                out = self._forward_spec(ids[:, j], mh[:, j:j + 1], pos)
                self._pos = pos
        self.last_spec = out
        output_ids, _, confidence = out
        return output_ids[:, 1:], confidence

    # ---- weights ----

    def load(self, d=None):
        """Load `mtp.*` out of a CONVERTED checkpoint. Everything strict except the two aliases.

        `Stage.load` deliberately ignores `mtp.*`; this is the other half. The names in the file ARE
        `Transformer.state_dict()`'s names (convert.py writes what `load_model` reads), so stage k is
        the `mtp.k.` prefix scan and nothing needs a per-model key table.

        The one concession is ALIAS_KEYS. `embed`/`head` are the TAIL's modules, already loaded by
        `Stage.load`, registered here a second time under the DSparkBlock; convert.py skips them
        because safetensors will not write shared storage twice. So this loads with strict=False and
        then checks the report by hand: anything missing beyond those two aliases is a corrupt or
        wrong checkpoint and raises, rather than serving a random-init Markov head behind a valid
        receipt."""
        V4 = _v4()
        d = d or V4.V4_DIR
        wm = V4.weight_map(d)
        self.alias_missing = []
        for k, blk in enumerate(self.mtp):
            prefix = f"mtp.{k}."
            names = [n for n in wm if n.startswith(prefix)]
            if not names:
                raise RuntimeError(
                    f"v4 dspark: no tensor under {prefix!r} in {d!r} — this checkpoint carries no "
                    f"MTP stage {k}. The drafter's weights ship with the model (mtp.0/1/2.*); a "
                    f"checkpoint without them can only be served greedily.")
            missing, unexpected = blk.load_state_dict(
                {n[len(prefix):]: V4.raw(n, d) for n in names}, strict=False)
            extra = sorted(set(missing) - set(ALIAS_KEYS))
            if extra or unexpected:
                raise RuntimeError(
                    f"v4 dspark: mtp stage {k} in {d!r} is not the checkpoint this config declares — "
                    f"missing {extra}, unexpected {sorted(unexpected)}. (Only {list(ALIAS_KEYS)} may "
                    f"be missing: they alias the tail's embed/head and convert.py skips them.)")
            self.alias_missing.append(tuple(sorted(missing)))
        return self

    def __repr__(self):
        return (f"<V4DSparkTail x{len(self.mtp)} block={self.block_size} "
                f"targets={tuple(self.args.dspark_target_layer_ids)} {self.dtype} on {self.device} "
                f"pos={self._pos}>")


class RingDrafter:
    """The tail-side half of a drafted round: v4_pipe's `TAIL_DRAFTER` seam over a `DSparkTail`.

    v4_pipe calls `on_chunk(msg, st, out)` on every step frame of a dspark-armed job — the frame as
    received, the tail Stage that has just forwarded it, and the reply already holding `token` (the
    sampled next token) and `tokens` (the model's greedy token at every chunk position). What comes
    back is merged into that reply, so ONE ring traversal carries both the verified tokens and the
    next block of drafts.

    WHAT CROSSES THE WIRE. `draft` (g ints), `n` (the accept length) and `conf` (g floats, diagnostic
    and unused). `main_hidden` -- 24 KiB per position at V4's shape -- never leaves the box; that is
    the entire reason the drafter runs here rather than on the coordinator, and it is why a drafted
    round costs the ring a few dozen bytes more than a greedy one.

    WHY THE TAIL RUNS THE ACCEPT RULE TOO. It must advance the mtp cache over exactly the COMMITTED
    positions before it can draft the next block, and "committed" is `plan_verify_round`'s answer for
    the round it just forwarded -- which it holds in full: the chunk's drafts are `ids[:, 1:]` and the
    replies are its own per-position argmax. The coordinator runs the SAME function over the SAME two
    inputs and asserts the two answers agree (v4_pipe.coordinate_dspark), so a divergence is a loud
    protocol failure rather than a drafter quietly conditioning on a history the ring never took.

    SINGLE SEQUENCE. The reply protocol is row 0's alone (`_serve_tail` samples `logits[0]`), so a
    b>1 chunk cannot even express the other rows' accept lengths; it is refused rather than silently
    drafting for row 0 and advancing the whole batch's lockstep cache. Batching needs the ragged
    accept path first.

    PIPELINED MODE (`self.pipelined`, armed per job by the reset's `pipelined` flag) serves
    v4_pipe.coordinate_dspark_pipelined, which streams the block as separate `s=1` frames instead of
    one chunk. The accept rule is the same rule applied one position at a time and the reply carries
    `acc` (this frame's position is committed) where the chunked reply carries `n`; see
    `_on_chunk_pipelined`.

    LAZY DRAFTING, and the measurement that forced it. A pipelined round streams ONE block per cycle
    and this class drafted one on EVERY committed frame, so the tail paid for `g` of them and the
    coordinator threw `g - 1` away. On the 6-stage EU ring that discarded work is the tail's whole
    margin: 7.33 ms of its 37.37 ms on-box is its own three layers, ~2.1 ms is lm_head + sampling
    (measured against a greedy job on the same ring), and the remaining ~28 ms is this drafter — on
    the stage that binds the entire ring, against a next-slowest of 26.0 ms. So the coordinator now
    hints each frame (`dnxt`/`dprev`, see `wants_block`) and a hinted frame produces only its STATE,
    via v4_dspark_fast's cache-advance-only write. Nothing about the round changes — same frames,
    same blocks consumed, same cancels, same tokens — only what the tail computes for the blocks
    nobody was ever going to read."""

    def __init__(self, tail):
        self.tail = tail                                   # a DSparkTail (its 4-method contract)
        self._done = False
        # PIPELINED mode (v4_pipe.coordinate_dspark_pipelined), armed per JOB by the reset's
        # `pipelined` flag. `_cfront` is the position of the last frame this tail judged COMMITTED and
        # `_mfront` is the model's greedy token that frame produced -- i.e. the token that belongs at
        # `_cfront + 1`. Those two scalars ARE the incremental accept rule; see _on_chunk_pipelined.
        self.pipelined = False
        self._cfront = None
        self._mfront = None
        # LAZY DRAFTING: (position, block) of the last block this drafter actually produced, which is
        # what a `dprev` hint dereferences. Only ever written where a block is returned, so a fenced
        # or skipped frame cannot age it into looking like the previous position's.
        self._last = None

    def on_chunk(self, msg, st, out):
        ids = torch.as_tensor(msg["ids"], dtype=torch.long)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.dim() != 2 or ids.shape[0] != 1:
            raise RuntimeError(f"v4 dspark: a drafted round is single-sequence today, got ids "
                               f"{tuple(ids.shape)} — see RingDrafter's docstring")
        if self.pipelined:
            return self._on_chunk_pipelined(ids, msg, st, out)
        start_pos = int(msg["start_pos"])
        main = st.tail_main_hidden()
        if start_pos == 0:
            # THE RING'S PREFILL. The tail holds the tap for every prompt position and `out["token"]`
            # is the token the model predicted for position P, so this is model.py's __main__ line
            # exactly: one forward_spec over 0..P-1, which drafts NOTHING (the module docstring's
            # FIRST ROUND paragraph says why the free-looking first block is deliberately skipped).
            # Round 1 is therefore the bare `[cur]` chunk, and `plan_verify_round` takes it as the
            # degenerate drafts == [] case.
            self.tail.reset()
            self._done = False
            self.tail.prefill([int(out["token"])], main)
            return {}
        replies = out.get("tokens")
        if replies is None:
            raise RuntimeError(
                "v4 dspark: a dspark job's reset must arm `spec` as well — the accept rule needs the "
                "model's greedy token at EVERY chunk position, and the tail only computes those when "
                "_spec is armed (v4_pipe.coordinate_dspark sends both flags)")
        n, committed = plan_verify_round(ids[0, 1:].tolist(), replies)
        # THE CONTEXT CLIFF. `advance_and_draft` refuses to draft a block that would rope past
        # max_seq_len, which happens within block_size+1 positions of the limit. That refusal is
        # right for the drafter and fatal for the ring — an exception here kills the tail's serve
        # loop and with it every later job — so the last rounds of a max-length generation degrade to
        # greedy instead: stop drafting, stop advancing, keep answering. V4_MAX_SEQ sets where.
        end = start_pos + n + self.tail.block_size + 1     # where the next block would rope to
        self._done = self._done or end > self.tail.args.max_seq_len
        if self._done:
            return {"draft": [], "n": n}
        # ADVANCE OVER THE COMMITTED PREFIX, NOT THE CHUNK: positions start_pos..start_pos+n are the
        # ones the ring keeps, `committed` holds the tokens that live at start_pos+1..start_pos+n+1,
        # and `main[:, :n+1]` are those positions' taps. Feeding the whole chunk instead passes every
        # check the drafter can make and drafts off a history the ring rejected.
        blk, conf = self.tail.advance_and_draft([committed], main[:, :n + 1], start_pos=start_pos)
        r = {"draft": blk[0].tolist(), "n": n,
             "conf": [round(c, 4) for c in conf[0].float().tolist()]}
        if V4_DRAFT_TOP2:
            d2 = second_choices(self.tail)                 # None on a sampling drafter: send nothing
            if d2 is not None:
                r["d2"] = d2
        return r

    def _on_chunk_pipelined(self, ids, msg, st, out):
        """One STREAMED `s=1` frame of a pipelined round. -> {acc, draft, conf}, merged into the reply.

        Pipelined speculation does not send a chunk: it streams `B+1` separate one-token frames
        back-to-back without waiting, so the drafter can no longer be advanced "over the round's
        committed prefix" -- there is no round, and when a frame is forwarded nobody yet knows whether
        its token will survive. `plan_verify_round` still decides every acceptance; it is just applied
        ONE POSITION AT A TIME, and the tail applies it to itself.

        THE INCREMENTAL ACCEPT RULE, and why two scalars are the whole of it. Frames reach the tail in
        the order the coordinator injected them (the ring is FIFO per leg), so the tail sees the frame
        at `q` after the frame at `q-1`. It computed that frame's greedy token `_mfront`, which IS the
        model's token at `q` whenever position `q-1` is itself committed. So the frame at `q` is on the
        committed path exactly when `q == _cfront + 1` and the token it carries equals `_mfront` --
        which is `plan_verify_round`'s "does the draft match the model's reply at this position",
        position by position. A rejected frame does not move `_cfront`, so every frame behind it fails
        the `q == _cfront + 1` test as well: the poisoning of a speculative tail falls out of the rule
        rather than needing a flag. The correction frame the coordinator sends after a cancel lands at
        `_cfront + 1` carrying `_mfront`, and re-opens the frontier.

        THE DRAFTER ADVANCES ONLY ON A COMMITTED FRAME, ONE POSITION, and drafts off the tap of THAT
        frame -- `advance_and_draft(ids=[m], main=tap(q), start_pos=q)` leaves `_pos = q`, so the block
        it returns proposes positions `q+2 .. q+B+1`. The coordinator commits `m` at `q+1` off this
        same reply and streams `[m] + block` from `q+1`, which is exactly the serial path's
        `[cur] + drafts` chunk decomposed into `s=1` frames. Advancing one position per call is
        bit-identical to the serial path's `n+1`-position call: `advance_and_draft` loops per position
        internally and keeps the last block, so the mtp cache walks the same committed positions in the
        same order and the block for a given history is the same block.

        NOTHING HERE ROLLS BACK, because nothing here is ever speculative -- the mtp cache only ever
        records committed positions (the module docstring's WHY THE MTP CACHE NEVER NEEDS A ROLLBACK,
        and test_cache_never_speculative). That is what makes the drafter side of pipelined speculation
        free: only the main stage's window ring had to learn to rewind W frames deep."""
        start_pos = int(msg["start_pos"])
        main = st.tail_main_hidden()
        if start_pos == 0:
            # THE RING'S PREFILL, identical to the serial path: one forward_spec over 0..P-1, which
            # drafts nothing. It also seeds the frontier -- the last prompt position is committed by
            # construction and `out["token"]` is the model's token at the one after it, so the first
            # streamed frame (the coordinator feeding that token at position P) accepts.
            self.tail.reset()
            self._done = False
            self._last = None                              # a new job never dereferences the old one's
            self.tail.prefill([int(out["token"])], main)
            self._cfront, self._mfront = main.shape[1] - 1, int(out["token"])
            return {"acc": True}
        if self._cfront is None:
            raise RuntimeError("v4 dspark: a pipelined frame before the prefill — the mtp window is "
                               "empty and there is no committed frontier to judge this frame against")
        if ids.shape[1] != 1:
            raise RuntimeError(f"v4 dspark: a pipelined round streams s=1 frames, got s={ids.shape[1]} "
                               f"— the whole point is that no stage replays a block position by "
                               f"position, so a multi-token frame is a coordinator bug")
        replies = out.get("tokens")
        if replies is None:
            raise RuntimeError(
                "v4 dspark: a pipelined job's reset must arm `spec` as well — the accept rule needs "
                "the model's greedy token at this frame's position, and the tail only computes those "
                "when _spec is armed")
        if start_pos != self._cfront + 1 or int(ids[0, 0]) != self._mfront:
            return {"acc": False}                          # speculative, and wrong: judge nothing
        m = int(replies[0])
        self._cfront, self._mfront = start_pos, m
        # THE CONTEXT CLIFF, as in the serial path: stop drafting (and stop advancing) before
        # `advance_and_draft` would rope a block past max_seq_len, rather than raising inside the
        # tail's serve loop and killing every later job. The frontier keeps moving, so the round
        # degrades to a one-frame-in-flight pipeline, which is plain greedy decode.
        self._done = self._done or (start_pos + self.tail.block_size + 1) > self.tail.args.max_seq_len
        if self._done:
            return {"acc": True, "draft": []}
        if not wants_block(msg, m, self._last):
            # LAZY DRAFTING: this frame's block would be discarded, so only its STATE is produced.
            # `advance_and_draft` fuses the two — one `forward_spec` per committed position both
            # writes the mtp KV slot and derives the block — but the WRITE is all a committed position
            # leaves behind (`DSparkAttention.forward` writes `kv_cache[:, pos % win]` and nothing
            # else), and v4_dspark_fast._advance_cache_only reproduces exactly those bytes. It is the
            # same primitive the fast path already uses for the intermediate positions of a serial
            # round, held to the reference's M=1 shapes and pinned bit-exact against the reference
            # loop in tests/test_v4_dspark_fast.py — here it is simply applied one position at a time.
            #
            # THE CURSOR MOVES ANYWAY, and that is the whole trap: `_advance_cache_only` advances no
            # cursor of its own, so skipping the block must still leave `_pos` on the position the
            # ring committed. A drafter left behind the committed stream raises inside the next
            # `advance_and_draft`, and there is no try/except around the tail's step handler — it
            # would kill the serve loop and every job after it. The guards below are the reference's
            # own, run BEFORE the write, so a hint that arrived on the wrong frame fails the job here
            # rather than corrupting the window silently.
            t = self.tail
            if t.pos is None or start_pos != t.pos + 1:
                raise RuntimeError(
                    f"v4 dspark: a lazy advance at {start_pos} but the mtp cache stands at {t.pos} — "
                    f"the drafter's cursor must walk exactly the committed positions whether or not "
                    f"it drafts on them")
            with torch.no_grad():
                _fast()._advance_cache_only(t, t._hidden(main, want_s=1), start_pos)
            t._pos = start_pos
            return {"acc": True}
        blk, conf = self.tail.advance_and_draft([[m]], main, start_pos=start_pos)
        draft = blk[0].tolist()
        self._last = (start_pos, draft)
        r = {"acc": True, "draft": draft,
             "conf": [round(c, 4) for c in conf[0].float().tolist()]}
        if V4_DRAFT_TOP2:
            d2 = second_choices(self.tail)                 # None on a sampling drafter: send nothing
            if d2 is not None:
                r["d2"] = d2
        return r


def _fast():
    """v4_dspark_fast, memoised. Imported lazily like `_v4()` — that module rebinds THIS one's
    `advance_and_draft` in `install()`, so a module-level import would be a cycle — and cached
    because the lazy path dereferences it once per skipped frame."""
    global _FAST
    if _FAST is None:
        import v4_dspark_fast
        _FAST = v4_dspark_fast
    return _FAST


_FAST = None


def wants_block(msg, m, last=None):
    """Will the coordinator consume a block off this frame's reply? -> draft it, or don't.

    UNHINTED MEANS DRAFT. `dnxt` is set only by a lazy-armed coordinator and only on frames it has
    PROVEN cannot drain the pipeline (v4_pipe's `_hints`), so its absence — an eager coordinator,
    an older one, or a frame the proof did not cover — falls through to the eager path. The lever
    can cost throughput by hinting too little; it cannot cost the round a block it needed.

    WHAT THE HINT IS. `dnxt` is the token the coordinator has already streamed at `start_pos + 1`.
    The round ends on this reply exactly when that token disagrees with `m`, the model's own token
    at that position — which is `plan_verify_round`'s accept test at one position, the same test
    this class already applies to itself. Agreement means the frame at `start_pos + 1` is
    committed too and the speculation runs on, so nothing will ask this reply for a block.

    THE REJECTION STILL DRAFTS EAGERLY, and it has to: the reply that reveals a rejection is the
    reply the coordinator refills from (coordinate_dspark_pipelined's FRESH BLOCK COMES WITH THE
    REJECTION). Skipping there would drain the pipeline and idle a full ring traversal waiting for
    a block — far more than the drafting it saves. So a mismatch drafts, and only agreement
    skips.

    `dprev` IS THE SAME HINT, DEREFERENCED HERE. The last frame of a burst has a successor the
    coordinator had not decided when it sent the frame — it is the head of the block the drain reply
    is about to return — so the coordinator names the source instead of the value, and `last` is what
    this drafter returned on the frame one position back: `(position, block)`. Every condition that
    makes the dereference sound is re-checked here rather than assumed, because the coordinator's
    half of it (that the frame at `H - 1` is the drain) cannot be seen from this side: the block must
    be ours, from EXACTLY the previous position, and non-empty. Any of those failing falls back to
    drafting, which is always safe."""
    nxt = msg.get("dnxt")
    if nxt is None and msg.get("dprev") and last is not None:
        at, blk = last
        if blk and at == int(msg["start_pos"]) - 1:
            nxt = blk[0]
    return nxt is None or int(nxt) != m


def ring_drafter(stage, ckpt_dir=None, temperature=0.0):
    """Build the tail's drafter: a `DSparkTail` over `stage`, its `mtp.*` loaded from `ckpt_dir`.

    `ckpt_dir=None` skips the load, which is the in-process path where the weights were transferred
    by hand; a serving tail always passes the dir it loaded its own layers from, and a checkpoint
    without `mtp.*` raises there rather than drafting out of uninitialised memory.

    Under `V4_DSPARK_FAST=1` this installs v4_dspark_fast, which rebinds `advance_and_draft` to the
    cache-advance-only fast path (the intermediate committed positions cost one KV write each instead
    of a full MoE forward). Default OFF and a no-op otherwise — the reference loop is bit-exact and
    the baseline every A/B measures against."""
    import v4_dspark_fast
    v4_dspark_fast.install(sys.modules[__name__])
    # THE TAIL IS BUILT AT THE FIRST DSPARK RESET, long after the stage's startup lever audit ran, so
    # that audit can only ever report this lever as unloaded. Record the live rebind here, where it is
    # finally knowable, and say it once in the tail's log — otherwise `V4_DSPARK_FAST=1` is a flag no
    # process on the ring ever confirms, which is the whole class this engine keeps paying for.
    import v4_levers
    live = getattr(DSparkTail.advance_and_draft, "_v4_dspark_fast", False)
    v4_levers.note("V4_DSPARK_FAST", live)
    print(f"[dspark] V4_DSPARK_FAST requested={v4_dspark_fast.V4_DSPARK_FAST} "
          f"observed={'on' if live else 'off'}", file=sys.stderr, flush=True)
    tail = DSparkTail(stage, temperature=temperature)
    # THE DRAFTER'S MoE LEVER (V4_DSPARK_MOE), banked + bound between construction and load — the
    # only window where the bank's `preserve=False` release-first layout is free (every routed-
    # expert byte is still torch.empty) and `load()` then writes the checkpoint THROUGH the bank
    # views. Per-instance: it touches these three MoEs and nothing else in the process. Recorded
    # here for the same reason V4_DSPARK_FAST is — the tail is built long after the startup lever
    # audit, so this is the first moment the rebind is knowable.
    import v4_dspark_moe
    took = v4_dspark_moe.install_drafter(tail)
    v4_levers.note("V4_DSPARK_MOE", took == len(tail.mtp) and took > 0)
    print(f"[dspark] V4_DSPARK_MOE requested={v4_dspark_moe.V4_DSPARK_MOE} "
          f"observed={took}/{len(tail.mtp)} drafter MoEs on the pair path",
          file=sys.stderr, flush=True)
    # THE WIDTH, recorded off the LIVE tail for the same reason the two levers above are: the
    # drafter is built at the first dspark reset, long after the startup audit, so this is the
    # first moment the width a run will actually draft at is knowable.
    v4_levers.note("V4_DSPARK_BLOCK", str(tail.block_size) if V4_DSPARK_BLOCK else "off")
    if V4_DSPARK_BLOCK:
        print(f"[dspark] V4_DSPARK_BLOCK requested={V4_DSPARK_BLOCK} observed={tail.block_size} "
              f"(trained width {tail.args.dspark_block_size})", file=sys.stderr, flush=True)
    # THE TREE GATE, recorded off the LIVE tail like the levers above: armed but useless (a sampling
    # drafter has no runner-up) is reported as off, so the audit shows what replies will carry.
    v4_levers.note("V4_DRAFT_TOP2", V4_DRAFT_TOP2 and tail.temperature == 0.0)
    if V4_DRAFT_TOP2:
        print(f"[dspark] V4_DRAFT_TOP2 requested=on observed="
              f"{'on' if tail.temperature == 0.0 else 'off (sampling drafter)'}",
              file=sys.stderr, flush=True)
    if ckpt_dir is not None:
        tail.load(ckpt_dir)
    return RingDrafter(tail)


def _selftest():
    """Prefill + three drafted rounds against the reference's own forward_spec, on CPU.

    The parity the tests prove, in one command: same weights on both sides, the oracle drives
    `Transformer.forward`/`forward_spec` and this drives Stage + DSparkTail, and every draft block,
    logit and confidence has to come back bit-identical."""
    import v4_ref_cpu

    args = v4_ref_cpu.cpu_args()
    prompt, rounds = 13, 3
    oracle = v4_ref_cpu.build_oracle(args, 0)
    st = _v4().Stage(0, args.n_layers, args, head=True, tail=True, dspark=True, device="cpu")
    for li in range(args.n_layers):
        st.layers[li].load_state_dict(oracle.layers[li].state_dict(), strict=True)
    st.embed_tokens.load_state_dict(oracle.embed.state_dict(), strict=True)
    st.norm.load_state_dict(oracle.norm.state_dict(), strict=True)
    st.lm_head.load_state_dict(oracle.head.state_dict(), strict=True)
    with torch.no_grad():
        for n in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            getattr(st, n).data.copy_(getattr(oracle, n).data)
    st._dspark = True
    dr = DSparkTail(st)
    for k, blk in enumerate(dr.mtp):
        sd = {n: v for n, v in oracle.mtp[k].state_dict().items() if n not in ALIAS_KEYS}
        assert set(blk.load_state_dict(sd, strict=False).missing_keys) == set(ALIAS_KEYS)
    print(st, flush=True)
    print(dr, flush=True)

    torch.manual_seed(0)
    ids = torch.randint(0, args.vocab_size, (1, prompt))
    o_tok, _, o_main = oracle(ids)
    assert oracle.forward_spec(o_tok, o_main) is None
    tok = st.logits_all(st.forward(st.embed(ids), ids, 0), full_logits=False).argmax(-1)
    assert torch.equal(tok, o_tok), "prefill token diverged"
    assert torch.equal(st.tail_main_hidden(), o_main), "prefill tap diverged"
    dr.prefill(tok, st.tail_main_hidden())
    for k, blk in enumerate(dr.mtp):
        assert torch.equal(blk.attn.kv_cache, oracle.mtp[k].attn.kv_cache), f"mtp {k} cache diverged"
    print(f"[v4] prefill  pos={dr.pos}  mtp window bit-identical to the reference", flush=True)

    for i in range(prompt, prompt + rounds):
        o_tok, _, o_main = oracle(tok.unsqueeze(1), i)
        o_spec = oracle.forward_spec(o_tok, o_main, i)
        h = st.forward(st.embed(tok.unsqueeze(1)), tok.unsqueeze(1), i)
        tok = st.logits_all(h, full_logits=False).argmax(-1)
        drafts, conf = dr.advance_and_draft(tok.unsqueeze(1), st.tail_main_hidden(), start_pos=i)
        assert torch.equal(tok, o_tok), f"token diverged at {i}"
        for got, want, what in zip(dr.last_spec, o_spec, ("output_ids", "logits", "confidence")):
            assert torch.equal(got, want), f"{what} diverged at {i}"
        print(f"[v4] round {i}  anchor {dr.last_spec[0][0, 0].item()} -> drafts "
              f"{drafts[0].tolist()}  confidence "
              f"{[round(c, 3) for c in conf[0].float().tolist()]}", flush=True)
    print(f"[v4] {rounds} drafted rounds bit-identical to Transformer.forward_spec", flush=True)


if __name__ == "__main__":
    _selftest()
