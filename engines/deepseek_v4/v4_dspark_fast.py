"""Collapse the DSpark drafter's per-round cost: the intermediate forwards are wasted, so skip them.

MEASURED (RTX 5090, real DeepSeek-V4-Flash weights, the tail's own drafter, one drafted round):
    advance_and_draft   224.55 ms      GPU idle for nearly all of it -- CPU-dispatch-bound.

WHERE IT GOES, AND WHY MOST OF IT IS THROWN AWAY.
`DSparkTail.advance_and_draft` commits a run of `n` positions per round (n = accepted + 1, up to
`block_size + 1` = 6) by calling the reference's `forward_spec` ONCE PER POSITION, in order, and
KEEPING ONLY THE LAST BLOCK (v4_dspark_draft.advance_and_draft's loop). Every one of those calls runs
the full `n_mtp_layers` DSparkBlocks -- MoE and all -- plus the head and the block_size-long Markov
sampling loop, and the intermediate n-1 of them exist ONLY for the single persistent write they leave
behind: `DSparkAttention.forward` writes exactly `kv_cache[:, start_pos % win] = main_kv` (model.py:783)
and nothing else. `main_kv` derives from `main_hidden` alone (`kv_norm(wkv(main_norm(main_proj(mh))))`,
model.py:759 <- forward_embed:853), the draft block's own K/V is concatenated for the attention and
dropped (model.py:784), and no other buffer in a DSparkBlock is persistent
(tests/test_v4_dspark.py::test_cache_never_speculative pins it). So ~5 of every 6 draft forwards
compute a full MoE stack, a vocab-wide head GEMM and a Markov loop, and discard all of it to advance
one KV slot.

TWO LEVERS, both opt-in, both default OFF (this is the spec-decode VERIFY path -- a wrong draft that
gets ACCEPTED corrupts committed output, so bit-exactness against the reference is the correctness
gate, not a nicety):

  1. CACHE-ADVANCE-ONLY (V4_DSPARK_FAST=1).  For the n-1 intermediate committed positions run ONLY
     the state-advance -- `main_kv = kv_norm(wkv(main_norm(main_proj(mh))))`, rope, act_quant, slot
     write -- and the FULL `forward_spec` once, for the last (kept) position. Collapses n full drafter
     forwards into 1 full forward plus a handful of tiny per-position GEMMs. BIT-EXACT: the only state
     a `DSparkAttention` intermediate call persists is that one slot, and this writes the identical
     bytes to it (verified torch.equal vs the reference loop in tests/test_v4_dspark_fast.py and on
     the GPU). The tempting batched GEMM (all n-1 positions through one main_proj/wkv) is bit-exact on
     CPU but NOT on a GPU -- an M=k matmul reassociates its reduction differently from k separate M=1
     ones and the confidence head diverged at k>=2 -- so the advance holds the reference's exact M=1
     shapes. 224 ms -> ~40 ms.

  2. HEAD GRAPH (V4_DSPARK_GRAPH=1, CUDA only, requires lever 1).  CUDA-graph the KEPT forward's
     `forward_head` -- the vocab-wide head GEMM, the block_size Markov GEMMs, the confidence head. It
     is the one piece of `forward_spec` that is FIXED-SHAPE and POSITION-INDEPENDENT (no start_pos, no
     rope, no KV -- model.py:860) and has NO host sync (greedy `sample` is an argmax), so it captures
     cleanly with a single static input buffer and no `_GraphState`. The MoE-bearing block bodies do
     NOT graph: the reference's expert dispatch drains the device (`indices[0].tolist()`,
     v4_moe_decode) and branches on the result in Python, which a graph cannot capture -- so they stay
     eager and lever 2's ceiling is the head's launch overhead, not the whole forward. Graph replay is
     bit-identical to eager (same ops, same operands; see m25_stage's GraphRunner for the pattern).

Nothing here edits the vendored reference or `v4_dspark_draft`: `install()` rebinds
`DSparkTail.advance_and_draft` at runtime, exactly as `v4_moe_decode.install()` rebinds `MoE.forward`.
`v4_dspark_draft.ring_drafter` calls it, so a serving --dspark tail picks the fast path up when the
env asks and every other caller (the CPU suite) stays on the reference path untouched.

self-test:  V4_DSPARK_FAST=1 python3 phase0/v4_dspark_fast.py
"""
import os

import torch

# Read at import like v4_moe_decode's V4_MOE_DECODE. A test flips the module attribute (not the env)
# to A/B in one process; install() reads the attribute, so the flip takes on the next install().
V4_DSPARK_FAST = os.environ.get("V4_DSPARK_FAST", "0") not in ("", "0")
V4_DSPARK_GRAPH = os.environ.get("V4_DSPARK_GRAPH", "0") not in ("", "0")

_REF_ADVANCE = None                     # DSparkTail.advance_and_draft, kept for uninstall + fallback


def _model():
    """The vendored reference module — act_quant / apply_rotary_emb / the scale_fmt globals live on
    it, set by v4_stage._set_globals when the Stage was built."""
    import v4_stage
    return v4_stage.ref()


# ── lever 1: cache-advance-only for the intermediate committed positions ──────────────────────────

def _advance_cache_only(self, main_hidden_int, base_pos):
    """Write the mtp KV slots for `k` intermediate committed positions WITHOUT a full forward.

    `main_hidden_int` is [b, k, hidden] -- the taps for positions base_pos .. base_pos+k-1, which the
    ring has already committed and whose draft blocks are discarded. The only thing a full
    `forward_spec` at each of these positions would leave behind is `DSparkAttention`'s single slot
    write (model.py:783), and that depends on `main_hidden` alone -- the committed token id only ever
    reaches the draft block `x`, which these positions throw away -- so this reproduces JUST that
    write, per layer, and nothing else.

    PER POSITION, NOT BATCHED, and that is a measured decision, not an oversight. Batching the k
    positions through main_proj/wkv into one GEMM is mathematically row-independent and IS bit-exact
    on CPU, but on a GPU an M=k matmul reassociates its K-reduction differently from k separate M=1
    matmuls: the confidence head (raw fp32) diverged at k>=2 in the GPU bench, with the drafts still
    matching by argmax luck. This path is the spec-decode VERIFY path, so it holds the reference's
    EXACT shapes -- main_proj/wkv at M=1, the same [b,1,*] the reference's own per-position
    forward_spec runs -- and stays torch.equal to it. main_x is still computed once per position and
    shared across layers, exactly as forward_embed does (model.py:853). Advances no cursor."""
    M = _model()
    k = main_hidden_int.shape[1]
    bsz = main_hidden_int.shape[0]
    for j in range(k):
        pos = base_pos + j
        # main_x: main_norm(main_proj(tap)) at M=1, the reference's forward_embed shape (model.py:853),
        # computed once and shared across the mtp layers.
        main_x = self.mtp[0].main_norm(self.mtp[0].main_proj(main_hidden_int[:, j:j + 1]))
        for blk in self.mtp:
            attn = blk.attn
            rd = attn.rope_head_dim
            win = attn.window_size
            main_kv = attn.kv_norm(attn.wkv(main_x))                             # [b, 1, head_dim]
            M.apply_rotary_emb(main_kv[..., -rd:], attn.freqs_cis[pos:pos + 1])
            M.act_quant(main_kv[..., :-rd], 64, M.scale_fmt, M.scale_dtype, True)  # in-place QAT
            attn.kv_cache[:bsz, pos % win] = main_kv.squeeze(1)


# ── lever 2: CUDA-graph the kept forward's position-independent head ──────────────────────────────

class _HeadGraph:
    """Capture + replay `mtp[-1].forward_head` at a fixed [b, block_size, hc_mult, dim] input shape.

    forward_head is the ONLY part of forward_spec that carries no start_pos: it collapses the four HC
    streams (hc_head), projects the vocab-wide logits, runs the block_size Markov-bias sampling loop
    and the confidence head (model.py:860-873). No rope, no KV, no host sync on the greedy path -- so
    one static input buffer and no position statics capture it. Replay is bit-identical to eager: the
    same ops on the same operands, and the loop's data-dependent embedding lookups re-read their
    device index tensors on every replay (the argmax'd token from the previous step)."""

    def __init__(self, head_block, b, block_size, hc_mult, dim, dtype, device):
        self.blk = head_block
        self.h = torch.zeros(b, block_size, hc_mult, dim, dtype=dtype, device=device)
        self.ids = torch.zeros(b, dtype=torch.long, device=device)
        self.graph = None
        self.out = None
        self.failed = False

    def _capture(self):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):                                  # warm up allocator + autotune
                self.blk.forward_head(self.h, self.ids)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            out = self.blk.forward_head(self.h, self.ids)
        self.graph, self.out = g, out

    def run(self, h, input_ids):
        """One graphed forward_head. Returns CLONED (output_ids, logits, confidence) -- the caller
        keeps them (last_spec, the wire) past the next replay, which overwrites the static outputs."""
        if self.graph is None:
            try:
                self._capture()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                torch.cuda.synchronize()                        # drain any in-flight side-stream work
                self.failed = True
                print(f"[v4 dspark] head-graph capture failed: {type(e).__name__}: {e} -> eager",
                      flush=True)
                return None
        self.h.copy_(h)
        self.ids.copy_(input_ids)
        self.graph.replay()
        return tuple(t.clone() for t in self.out)


def _kept_forward_spec(self, input_ids, main_hidden, start_pos):
    """The one forward_spec whose block IS kept -- eager blocks, optionally graphed head.

    Mirrors `DSparkTail._forward_spec` (start_pos > 0 always here: a kept position is committed and
    >= 1). The block loop stays eager because its MoE cannot be captured; only forward_head, which is
    pure and fixed-shape, is routed to `_HeadGraph` when V4_DSPARK_GRAPH and the shape/temperature/
    device allow it."""
    h, main_x = self.mtp[0].forward_embed(main_hidden, input_ids)
    for blk in self.mtp:
        h = blk(h, start_pos, input_ids, main_x)
    if _graph_eligible(self, h):
        hg = self._fast_head_graph
        if hg is None:
            b, s, hc, dim = h.shape
            hg = _HeadGraph(self.mtp[-1], b, s, hc, dim, h.dtype, h.device)
            self._fast_head_graph = hg
        if not hg.failed:
            out = hg.run(h, input_ids)
            if out is not None:
                return out                                      # graph capture failure -> eager below
    return self.mtp[-1].forward_head(h, input_ids)


def _graph_eligible(self, h):
    """The head graph runs only where it is both correct and fixed-shape.

    Greedy only (temperature > 0 draws from an RNG, which a plain capture would freeze), CUDA only
    (there is no graph off-device), b == 1 (the ring's shape; the captured buffer is one batch), and
    forward_head's own input width -- `block_size` HC-expanded rows (model.py:856, the draft block
    before the +1 anchor column), constant every round, so a shorter chunk falls back rather than
    capturing a zoo of shapes."""
    return (V4_DSPARK_GRAPH and h.is_cuda and self.temperature == 0.0
            and h.shape[0] == 1 and h.shape[1] == self.block_size)


# ── the fast advance_and_draft (rebinds the reference's) ──────────────────────────────────────────

def fast_advance_and_draft(self, input_ids_seq, main_hidden_seq, start_pos):
    """`DSparkTail.advance_and_draft` with the intermediate forwards collapsed. Same signature, same
    return, same guards -- a drop-in the ring cannot tell from the reference except by the clock.

    The guards are the reference's verbatim (they are cheap and each catches a protocol bug that
    would otherwise draft off the wrong history); only the advance body changes: cache-advance-only
    over the n-1 intermediate positions, then one full forward for the kept block."""
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
    with torch.no_grad():
        if n > 1:
            # positions start_pos .. start_pos+n-2 are intermediate: only their KV slot survives.
            _advance_cache_only(self, mh[:, :n - 1], start_pos)
        pos = start_pos + n - 1                                 # the kept (last committed) position
        out = _kept_forward_spec(self, ids[:, n - 1], mh[:, n - 1:n], pos)
        self._pos = pos
    self.last_spec = out
    output_ids, _, confidence = out
    return output_ids[:, 1:], confidence


# ── install / uninstall (the v4_moe_decode pattern) ───────────────────────────────────────────────

def install(module=None):
    """Rebind `DSparkTail.advance_and_draft` to the fast path. Returns True if it took.

    No-op unless V4_DSPARK_FAST (default off) -- the CPU suite and any A/B baseline run the reference
    path untouched. Idempotent. `module` is v4_dspark_draft; defaults to importing it, so
    `ring_drafter` can call `install()` with no args and a test can pass the module it already holds."""
    global _REF_ADVANCE
    if not V4_DSPARK_FAST:
        return False
    if module is None:
        import v4_dspark_draft as module
    tail = module.DSparkTail
    if getattr(tail.advance_and_draft, "_v4_dspark_fast", False):
        return False
    _REF_ADVANCE = tail.advance_and_draft
    fast_advance_and_draft._v4_dspark_fast = True
    tail.advance_and_draft = fast_advance_and_draft
    # A per-instance graph handle, so reset()/new sequences reuse the captured (weight-only) graph.
    if not hasattr(tail, "_fast_head_graph"):
        tail._fast_head_graph = None
    return True


def uninstall(module=None):
    """Restore the reference `advance_and_draft`. For an in-process A/B; not on the serving path."""
    global _REF_ADVANCE
    if _REF_ADVANCE is None:
        return False
    if module is None:
        import v4_dspark_draft as module
    module.DSparkTail.advance_and_draft = _REF_ADVANCE
    _REF_ADVANCE = None
    return True


def _selftest():
    """Bit-exact parity + a timing A/B against the reference advance loop, on whatever device is here.

    Reuses v4_dspark_draft's own toy-scale harness: a Stage + DSparkTail holding one oracle's weights,
    prefilled, then the same drafted rounds down both the reference and the fast path. Every draft
    block, logit, confidence and mtp KV buffer has to come back torch.equal. On CUDA it also flips
    V4_DSPARK_GRAPH and re-checks; on CPU the graph is inert and only lever 1 is exercised."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import v4_ref_cpu
    import v4_stage
    import v4_dspark_draft as D

    global V4_DSPARK_FAST
    args = v4_ref_cpu.cpu_args()
    oracle = v4_ref_cpu.build_oracle(args, 0)

    def build():
        st = v4_stage.Stage(0, args.n_layers, args, head=True, tail=True, dspark=True, device="cpu")
        for li in range(args.n_layers):
            st.layers[li].load_state_dict(oracle.layers[li].state_dict(), strict=True)
        st.embed_tokens.load_state_dict(oracle.embed.state_dict(), strict=True)
        st.norm.load_state_dict(oracle.norm.state_dict(), strict=True)
        st.lm_head.load_state_dict(oracle.head.state_dict(), strict=True)
        with torch.no_grad():
            for n in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                getattr(st, n).data.copy_(getattr(oracle, n).data)
        st._dspark = True
        dr = D.DSparkTail(st)
        for k, blk in enumerate(dr.mtp):
            sd = {n: v for n, v in oracle.mtp[k].state_dict().items() if n not in D.ALIAS_KEYS}
            blk.load_state_dict(sd, strict=False)
        return st, dr

    def record():
        """Prefill + a few FULL-accept rounds of varying run length, driven monotonically so the
        stage never rewinds. Returns (prefill_tok, prefill_main, [(committed, main, start_pos)])."""
        st, _ = build()
        p = 13
        ids = torch.randint(0, args.vocab_size, (1, p), generator=torch.Generator().manual_seed(3))
        tok = st.logits_all(st.forward(st.embed(ids), ids, 0), full_logits=False).argmax(-1)
        prefill_main = st.tail_main_hidden()
        seq = [int(tok)] + [7, 151, 293, 41, 88, 19, 200]           # forced monotonic committed run
        rounds, pos, base = [], p, 0
        for run_len in (1, 3, 2):                                    # n = 1, 3, 2 -> intermediate advances
            chunk = torch.tensor([seq[base:base + run_len]], dtype=torch.long)
            st.forward(st.embed(chunk), chunk, pos)
            committed = torch.tensor([seq[base + 1:base + run_len + 1]], dtype=torch.long)
            rounds.append((committed, st.tail_main_hidden().clone(), pos))
            pos += run_len
            base += run_len
        return tok, prefill_main, rounds

    tok, prefill_main, rounds = record()

    def replay(dr):
        dr.prefill(tok, prefill_main)
        blocks = []
        for committed, main, start_pos in rounds:
            blk, conf = dr.advance_and_draft(committed, main, start_pos=start_pos)
            blocks.append((blk.clone(), conf.clone(), dr.last_spec[1].clone()))
        return blocks, [b.attn.kv_cache.clone() for b in dr.mtp]

    V4_DSPARK_FAST = False
    _, dr = build()
    ref_blocks, ref_caches = replay(dr)

    V4_DSPARK_FAST = True
    assert install(D), "install must take when V4_DSPARK_FAST is on"
    _, dr2 = build()
    fast_blocks, fast_caches = replay(dr2)
    for i, ((rb, rc, rl), (fb, fc, fl)) in enumerate(zip(ref_blocks, fast_blocks)):
        assert torch.equal(rb, fb), f"draft ids diverged round {i}"
        assert torch.equal(rc, fc), f"confidence diverged round {i}"
        assert torch.equal(rl, fl), f"draft logits diverged round {i}"
    for i, (rk, fk) in enumerate(zip(ref_caches, fast_caches)):
        assert torch.equal(rk, fk), f"mtp {i} kv_cache diverged"
    print("[v4 dspark] cache-advance-only: bit-exact vs reference advance_and_draft", flush=True)
    uninstall(D)
    print("[v4 dspark] self-test OK", flush=True)


if __name__ == "__main__":
    _selftest()
