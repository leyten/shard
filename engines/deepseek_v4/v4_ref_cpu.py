"""DeepSeek-V4-Flash's reference Transformer, running on a CPU box — the oracle v4_stage is graded on.

Step 1 of the V4 engine is not a stage; it is the thing a stage can be WRONG against. `v4_stage`
(step 2) will instantiate the vendored `Block`s for a contiguous layer range and thread `h [b,s,4,d]`
between boxes, and the only way to know it threads them correctly is to run the whole model in one
process and compare. This module builds that whole model, with random-but-deterministic weights, at
toy dimensions, with no GPU and no 158 GiB of checkpoint.

RENT, DO NOT REWRITE. Nothing here reimplements V4's math -- it drives
phase0/deepseek_v4_ref/inference/model.py verbatim (see deepseek_v4_ref/PROVENANCE.md). What it
supplies is the two things that file needs and a CPU box does not have:

  1. kernels      `from kernel import act_quant, ...` at model.py's module scope is tilelang, i.e.
                  CUDA codegen at first call. v4_kernels_cpu.install() puts plain-torch stand-ins
                  under that name FIRST -- module-scope imports bind once, so after model.py is
                  loaded the choice is frozen for the life of the process.
  2. weights      every parameter in the reference is `torch.empty` (it expects load_model to
                  overwrite them from safetensors), so an unloaded Transformer emits NaN or worse
                  from uninitialised memory. init_random fills all of them from one seed.

Three facts about the reference that shape everything downstream:

  GLOBALS ARE SET IN Transformer.__init__.  model.py keeps world_size/rank/default_dtype/scale_fmt/
  scale_dtype as MODULE globals and assigns them from ModelArgs inside `Transformer.__init__`
  (model.py:881). Every `Linear` reads default_dtype at construction and every `act_quant` call site
  reads scale_fmt/scale_dtype at call time. v4_stage never builds a Transformer -- it builds Blocks
  directly -- so IT WILL HAVE TO SET THOSE GLOBALS ITSELF, or it silently gets bf16 weights and
  unrounded scales while the oracle got fp8 and ue8m0. That is the first thing step 2 has to do.

  compress_ratios IS INDEXED BY DSPARK LAYERS TOO.  DSparkBlock passes `args.n_layers + stage_id` to
  Attention.__init__, which reads `args.compress_ratios[layer_id]` and asserts it is 0. So the tuple
  must be n_layers + n_mtp_layers long, not n_layers -- the shipped config.json is 46 entries for 43
  layers for exactly this reason.

  THE COMPRESSOR IS STATEFUL ACROSS SEQUENCES.  kv_state/score_state are non-persistent buffers that
  a prefill only partially rewrites (just the `remainder` rows past the last full compression
  block). Re-prefilling a used model therefore starts from the previous sequence's tail, not from
  the -inf the constructor wrote. Every oracle here is built fresh per sequence; a serving path will
  need an explicit reset.

Self-test:  python3 phase0/v4_ref_cpu.py
"""
import importlib.util
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4_kernels_cpu

def _vendored(name):
    """Locate a vendored reference tree, in the repo AND on a deployed box.

    The repo keeps upstream trees under vendor/ so the engine directories hold only our code. A
    rented stage does NOT have that layout: deploy copies the modules flat into /root with the
    reference tree beside them, which is the whole reason the engines are flat modules. So try the
    sibling first (deployed), then the repo's vendor/. Resolving only one way silently breaks the
    other, and the box is the one that would not be caught by a test run."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, name),
                 os.path.join(here, os.pardir, os.pardir, "vendor", name)):
        if os.path.isdir(cand):
            return os.path.normpath(cand)
    return os.path.join(here, name)


REF_DIR = _vendored("deepseek_v4_ref")
INFERENCE_DIR = os.path.join(REF_DIR, "inference")
# encoding_dsv4.encode_messages -- the chat template the real prompts go through. Nothing here uses
# it yet; step 3 (a real prompt on a real ring) does, and it lives next to the model, not on pip.
ENCODING_DIR = os.path.join(REF_DIR, "encoding")

_REF = None


def load_ref():
    """DeepSeek's reference model.py, with usable kernels installed around it (memoized).

    ORDER IS LOAD-BEARING: install() before the file is read, because `from kernel import ...` is
    resolved at model.py's module scope and never again. INFERENCE_DIR goes on sys.path for the
    tilelang backend, where install() deliberately does nothing and that flat import has to find the
    real kernel.py sitting next to model.py. install_sm120() is in the same window and for the same
    reason: it swaps ONE of those kernels (sparse_attn, whose vendored tiling asks for 138 KiB of
    shared memory and cannot launch on sm_120) and a swap made after the exec would bind nothing.
    It no-ops on the CPU backend and on any device with room for the vendored kernel.

    Loaded under the name `dsv4_model` rather than `model` -- `model` is far too generic a top-level
    name to claim in a process that also imports shard's own modules."""
    global _REF
    if _REF is None:
        v4_kernels_cpu.install()
        if INFERENCE_DIR not in sys.path:
            sys.path.insert(0, INFERENCE_DIR)
        if v4_kernels_cpu.backend() == "tilelang":
            import v4_sparse_attn_sm120
            v4_sparse_attn_sm120.install_sm120()
        spec = importlib.util.spec_from_file_location("dsv4_model", os.path.join(INFERENCE_DIR, "model.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["dsv4_model"] = mod
        spec.loader.exec_module(mod)
        # AFTER the exec, unlike the two above: this one replaces a method on a class the exec
        # created. It strips the 1+k device drains the reference's expert-dispatch loop costs per
        # layer per decoded token, bit-exactly, and falls back to the reference for every other
        # shape (V4_MOE_DECODE=0 to A/B it).
        import v4_moe_decode
        v4_moe_decode.install(mod)
        # THEN the grouped fp4 kernel, and the ORDER OF THESE TWO IS THE PRECEDENCE. Each install
        # captures whatever `MoE.forward` is bound at that moment as its own fallback, so installing
        # grouped SECOND makes the chain grouped -> decode -> reference: the grouped kernel claims the
        # single-token score-routed decode step (V4_MOE_GROUPED=1, CUDA only), hands everything it
        # declines (s>1, world_size>1, hash-routed layers) to the decode fast path, and that hands
        # what IT declines to the untouched reference. Both off => reference, byte-identical.
        # Installing them the other way round would bury the grouped kernel under the decode path and
        # it would never run — this line's position is the whole wiring.
        import v4_moe_grouped
        v4_moe_grouped.install(mod)
        # LAST of the three, and again the position IS the precedence. Both levers above are gated on
        # ONE token, so the DSpark drafter — whose MoEs run at `dspark_block_size` rows, never 1 —
        # falls through both and lands on the vendored dispatch loop, on the tail, every drafted
        # round. This claims that small-block shape and hands every single-token step DOWN to the
        # chain above, so the main decode path is untouched. V4_MOE_MULTI=1 to arm it; default OFF.
        import v4_moe_multi
        v4_moe_multi.install(mod)
        # Same window, same reason: reference-compute "slim" overrides that remove removable per-layer
        # work (the indexer while context is short; the inplace KV/Q QAT sim). Both behind default-OFF
        # env flags (V4_REF_SLIM / V4_REF_SLIM_NOQAT), so this is a no-op — reference byte-identical —
        # unless one is set. See phase0/v4_ref_slim.py.
        import v4_ref_slim
        v4_ref_slim.install(mod)
        # Independent of the MoE chain above: this rebinds the module-level `fp8_gemm` (what
        # `linear()` resolves at call time) and `Expert.forward`, never `MoE.forward`. Both levers
        # default OFF — with the envs unset this is a no-op and the reference is byte-identical.
        # The occupancy-tiled kernel is CUDA-only and every tile it serves is probed torch.equal
        # against the vendored kernel first; see phase0/v4_fp8_gemv.py.
        import v4_fp8_gemv
        v4_fp8_gemv.install(mod)
        _REF = mod
    return _REF


# A V4 small enough for a laptop and structurally identical to the 158 GiB one. Every value that
# LOOKS arbitrary is pinned by a constraint the reference asserts or a branch it would otherwise
# never take; cpu_args() checks them all rather than letting a bad config surface as a shape error
# ten frames deep inside Attention.
_CPU_ARGS = dict(
    # deterministic + unquantized: temperature 0 makes sample() an argmax, and dtype/scale_fmt/
    # expert_dtype keep every Linear in bf16 so act_quant's inplace QAT path is exercised while
    # fp8_gemm/fp4_gemm stay out of the forward (they are unit-tested directly instead).
    dtype="bf16", scale_fmt=None, scale_dtype="fp32", expert_dtype=None, temperature=0.0,
    max_batch_size=2, max_seq_len=256, vocab_size=512,
    dim=256, n_layers=8, n_heads=4, o_groups=2, q_lora_rank=64, o_lora_rank=32,
    head_dim=128, rope_head_dim=64, window_size=16,
    # index_topk 8 over ratio-4 compression means the Indexer starts DISCRIMINATING past 32 tokens
    # and picks everything below that -- the same regime the shipped config is in (512 over ratio 4
    # = no selection under 2048 tokens), which is why short-prompt tests cannot exercise it.
    index_head_dim=128, index_n_heads=4, index_topk=8,
    # 4 -> overlapping Compressor + Indexer; 8 -> plain Compressor. 8 rather than the shipped 128 so
    # a 33-token prompt still crosses a compression boundary and the branch is actually taken.
    # Length is n_layers + n_mtp_layers, tail zeros for the DSpark stages (see the module docstring).
    compress_ratios=(0, 0, 4, 8, 4, 8, 4, 0, 0, 0),
    original_seq_len=64, rope_factor=4, compress_rope_theta=160000,
    n_routed_experts=8, n_activated_experts=2, n_shared_experts=1, moe_inter_dim=64,
    n_hash_layers=2, score_func="sqrtsoftplus", route_scale=1.5, swiglu_limit=10.0,
    hc_mult=4, hc_sinkhorn_iters=20,
    n_mtp_layers=2, dspark_block_size=3, dspark_noise_token_id=511,
    dspark_target_layer_ids=(5, 6, 7), dspark_markov_rank=16,
)


def cpu_args(**overrides):
    """A small ModelArgs that still takes every branch of the real one.

    The constraints, and where each comes from:
      head_dim - rope_head_dim divisible by 64   Attention/DSparkAttention quantize the non-rope
                                                 half of kv with `act_quant(kv[..., :-rd], 64, ...)`,
                                                 which asserts the last dim is a multiple of 64.
      index_head_dim a power of two, % 32 == 0   the Indexer hadamard-rotates q (power of two) and
                                                 then fp4-quantizes it in 32-wide blocks.
      index_head_dim > rope_head_dim             q[..., -rope_head_dim:] has to be a strict slice.
      n_heads % o_groups == 0                    wo_a is viewed as [groups, o_lora_rank, -1].
      compress_ratios: len n_layers+n_mtp_layers, values in {0, 4, R>4}, 0 at the DSpark ids
                                                 ratio 4 is the only one that builds an Indexer
                                                 (Attention.__init__:474) and the only one with
                                                 overlapping compression; DSparkAttention asserts 0.
      dspark_target_layer_ids within n_layers    they index the main stack's outputs.
      dspark_noise_token_id < vocab_size         it is embedded.
    """
    ref = load_ref()
    args = ref.ModelArgs(**{**_CPU_ARGS, **overrides})
    nope = args.head_dim - args.rope_head_dim
    assert nope > 0 and nope % 64 == 0, f"head_dim-rope_head_dim={nope} must be a positive multiple of 64"
    idx_d = args.index_head_dim
    assert idx_d & (idx_d - 1) == 0 and idx_d % 32 == 0, f"index_head_dim={idx_d} must be a power of two and a multiple of 32"
    assert idx_d > args.rope_head_dim, f"index_head_dim={idx_d} must exceed rope_head_dim={args.rope_head_dim}"
    assert args.n_heads % args.o_groups == 0, "n_heads must be divisible by o_groups"
    n_mtp = args.n_mtp_layers if args.dspark_block_size else 0
    assert len(args.compress_ratios) == args.n_layers + n_mtp, \
        f"compress_ratios must be {args.n_layers + n_mtp} long (n_layers + n_mtp_layers)"
    assert all(r == 0 for r in args.compress_ratios[args.n_layers:]), "DSpark layers must have compress_ratio 0"
    assert all(r in (0, 4) or r > 4 for r in args.compress_ratios), "compress ratios must be 0, 4, or > 4"
    assert all(i < args.n_layers for i in args.dspark_target_layer_ids), "dspark target layers must exist"
    assert args.dspark_noise_token_id < args.vocab_size, "noise token must be in vocab"
    return args


def init_random(model, seed=0):
    """Fill EVERY parameter deterministically. The reference allocates them with torch.empty.

    Three kinds, because three kinds behave differently under a normal:
      norm weights   1 + N(0, 0.02). RMSNorm's constructor writes ones; centring these on zero would
                     scale every hidden state to noise and nothing downstream would be measurable.
      integer        Gate.tid2eid on a hash layer is an int32 expert-id table, not a weight -- it is
                     indexed, so it must be a valid expert id.
      everything else N(0, 0.02).
    Buffers are deliberately untouched: kv_cache/kv_state/score_state/freqs_cis are already
    correctly initialised by their modules (zeros, -inf, and the YaRN table), and a random freqs_cis
    is not a rotation."""
    ref = load_ref()
    norms = {f"{n}.weight" for n, m in model.named_modules() if isinstance(m, ref.RMSNorm)}
    ints = {f"{n}.tid2eid": m.weight.size(0) for n, m in model.named_modules()
            if isinstance(m, ref.Gate) and m.hash}
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in ints:
                p.random_(0, ints[name])
            elif name in norms:
                p.normal_(1.0, 0.02)
            else:
                p.normal_(0.0, 0.02)
    return model


def build_oracle(args=None, seed=0):
    """A whole random V4 on CPU, in eval mode. This is the golden side of every parity test.

    Built under default dtype bf16, which is what generate.py sets and what the reference assumes:
    Linear/ParallelEmbedding/kv_cache all take their dtype from it at construction. Only
    construction -- the forward reads torch.get_default_dtype() nowhere the bf16 config touches --
    so callers do not need the context, and the tests deliberately run outside it."""
    ref = load_ref()
    args = args or cpu_args()
    with ref.set_dtype(torch.bfloat16):
        model = ref.Transformer(args)
    return init_random(model, seed).eval()


def _smoke():
    """model.py's own __main__, on CPU and at cpu_args() scale."""
    args = cpu_args()
    torch.manual_seed(0)
    prompt, steps = 33, 40
    x = torch.randint(0, args.vocab_size, (1, prompt + steps))
    model = build_oracle(args)

    output_ids, logits, main_hidden = model(x[:, :prompt])
    assert model.forward_spec(output_ids, main_hidden) is None, "prefill forward_spec must return None"
    print(f"prefill  logits {tuple(logits.shape)}  main_hidden {tuple(main_hidden.shape)}  "
          f"next {output_ids.tolist()}")
    checksum = 0.0
    for i in range(prompt, prompt + steps):
        output_ids, logits, main_hidden = model(x[:, i:i + 1], i)
        output_ids, spec_logits, confidence = model.forward_spec(output_ids, main_hidden, i)
        for t in (logits, main_hidden, spec_logits, confidence):
            assert torch.isfinite(t).all(), f"non-finite tensor at step {i}"
        checksum += logits.float().sum().item() + spec_logits.float().sum().item()
    print(f"decode   x{steps}  spec_logits {tuple(spec_logits.shape)}  "
          f"output_ids {tuple(output_ids.shape)}  confidence {tuple(confidence.shape)}")
    print(f"backend  {v4_kernels_cpu.backend()}   checksum {checksum:.4f}")


if __name__ == "__main__":
    _smoke()
