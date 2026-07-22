"""MiniMax-M2.5 PP stage core — the ONE new compute file for the port.

A decoder Layer = hand-rolled standard GQA attention (bit-checked vs HF MiniMaxM2Attention,
m25_gqa_check.py) + vLLM NVFP4 FusedMoE experts (executes on sm_120, m25_moe_probe.py). Each
Layer carries its own KV cache and crops to start_pos for spec-decode rollback. Written to the
phase0 serve-loop contract (reset() / forward(x, start_pos, pe) / run_block) so specpipe's
coordinate_pipe + libp2p transport + receipts + heal ride on top unchanged.

M2.5 specifics (verified from the real nvidia/MiniMax-M2.5-NVFP4 config):
  62 layers, hidden 3072, GQA 48q/8kv head_dim 128, full-width q/k_norm before reshape,
  partial RoPE (first 64 dims, rotate_half), rope_theta 5e6, 256 experts / top-8, sigmoid
  router + per-layer e_score_correction_bias, NO shared expert, NO dense layers.

  self-test:  python m25_stage.py --dir /root/m25 --layers 29 30
"""
import os, sys, json, argparse, torch
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29577")
os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("LOCAL_RANK", "0")
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.minimax_m2 import modeling_minimax_m2 as M
from torch.nn.attention import sdpa_kernel, SDPBackend                 # SDPA prefill attn (long-ctx OOM fix)
from torch.nn.attention.bias import causal_lower_right                 # bottom-right causal (NOT is_causal)

dev = "cuda"
_CTX = None


def _cli_dir(argv):
    """Pre-parse --dir from a script invocation. Module init below consumes M25_DIR at import time
    (AutoConfig + the safetensors index), so the self-test's --dir must win BEFORE that runs — the
    old flow parsed it in __main__ AFTER init and silently ignored it."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--dir")
    return p.parse_known_args(argv)[0].dir


if __name__ == "__main__":
    _d = _cli_dir(sys.argv[1:])
    if _d:
        os.environ["M25_DIR"] = _d
DIR = os.environ.get("M25_DIR", "/root/m25")
cfg = AutoConfig.from_pretrained(DIR)
H, NH, NKV, HD = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
E = getattr(cfg, "num_local_experts", getattr(cfg, "num_experts", None))
K = cfg.num_experts_per_tok
I = getattr(cfg, "moe_intermediate_size", None) or cfg.intermediate_size
EPS = cfg.rms_norm_eps
GRP = NH // NKV
SCALING = HD ** -0.5
# Memory-efficient attention: never materialize the [1,NH,s,total] score matrix (the prefill OOM root —
# at 10k ctx the naive matmul+fp32-softmax was ~6.5GB/stage). SDPA's flash/efficient/cudnn backends do
# online softmax, so prefill attn is O(s) not O(s*total). Default ON; M25_SDPA=0 keeps the naive path for A/B.
M25_SDPA = os.environ.get("M25_SDPA", "1") != "0"
_SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION,
                  SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]   # fused first; MATH = never-OOM safety net
# Static-buffer KV (opt-in): preallocate [1,NKV,MAXLEN,HD] per layer + index_copy_ writes, instead of
# grow-by-cat. Gives FIXED addresses (the prerequisite for CUDA-graph capture + batched concurrency) and
# avoids cat fragmentation at long ctx. Default OFF (cat path stays the proven default). MAXLEN is bounded:
# a full 131072 buffer is ~537MB/layer*2 ≈ 7GB/stage and won't fit beside ~27GB weights on a 32GB 5090, so
# the cap defaults to 40960 (≈2.2GB/13-layer stage, covers the ≥30k deploy target). Reads stay :total exact
# with causal_lower_right, so the static path is BIT-IDENTICAL to cat (proven: research/m25_statickv_test.py).
M25_STATIC_KV = os.environ.get("M25_STATIC_KV", "0") != "0"
M25_KV_MAXLEN = int(os.environ.get("M25_KV_MAXLEN", "40960"))
# Continuous batching (opt-in M25_BATCH=B): run B concurrent requests through one ring traversal so the WAN
# round-trip amortizes across all B (aggregate-throughput lever). Each Layer gets a [B,NKV,MAXLEN,HD] KV;
# the fixed-shape DECODE block batches (per-stream scatter + per-stream additive causal mask — batchverify
# pattern, proven bit-exact), the MoE runs PER STREAM (NVFP4 MoE is token-count non-invariant), prefill
# writes one row. Each stream's output is byte-identical to solo. Default 1 (single-stream path untouched).
M25_BATCH = int(os.environ.get("M25_BATCH", "1"))
# Batched MoE (opt-in M25_BATCH_MOE=1, default OFF): in batched decode run ONE grouped-GEMM over ALL
# B*(K+1) tokens instead of the per-stream loop (B separate calls over K+1 tokens each). Decode is
# weight-read-bound, so the per-stream loop re-reads the 256 NVFP4 expert weights B times per layer AND
# runs the heavy grouped-GEMM machinery on a wasteful ~9-token batch B*nlayers times per traversal — the
# GPU-bound cliff that collapses batched serving at context (per-stream MoE GPU grows ~linearly with B).
# Batched: the expert weights are read ONCE and reused across all B streams' tokens (~B x fewer reads),
# and one well-utilized grouped GEMM replaces B tiny ones. The MoE is mathematically per-token (no
# cross-token reduction) so every token's OUTPUT is a valid MoE output either way; the cost is that the
# NVFP4 cutlass kernel SCHEDULE (tile/Split-K, picked by per-expert token count) varies with batch
# composition -> a stream's bytes are no longer reproducible-in-isolation (challenger needs the batch),
# so this trades the per-stream bit-exact RECEIPT for throughput. Restoring both = a batch-INVARIANT
# grouped GEMM (moe_align_block_size + SPLIT_K=1), the follow-up. Default OFF keeps the verifiable path.
M25_BATCH_MOE = os.environ.get("M25_BATCH_MOE", "0") != "0"
# EAGLE hybrid drafter (opt-in M25_EAGLE=1): capture the target's auxiliary hidden states (what the
# EAGLE-3 draft head consumes) during each ring traversal so the coordinator's EagleDrafter can predict
# the next tokens on NOVEL/reasoning text (where the n-gram drafter is blind). The captured states ride
# the existing forward->return path to the coordinator (NO extra round-trip). A stage records ONLY the
# aux layers in its own [lo,hi) range; serve() threads them forward; the tail returns them with the
# verify result. Default OFF (the n-gram-only path is untouched).
#
# INDEX CONVENTION: `eagle_aux_hidden_state_layer_ids` = [1,30,58] are RAW decoder-layer indices — SpecForge
# (the head's training framework) hooks `layers[idx]` for idx in [1,30,58] and stores `output[0]`, i.e. the
# OUTPUT residual stream of decoder layers 1/30/58. So capture layer L's output when L.li is in the list.
# (A brief 2026-06-29 "fix" mis-read vLLM's internal idx+1 and captured {0,29,57}; the 2026-06-30 on-engine
# A/B confirmed {1,30,58} wins: reason-math 34% vs 30%, open-chat 13% vs 11% — reverted to this.)
M25_EAGLE = os.environ.get("M25_EAGLE", "0") != "0"
# EAGLE tree-verify (opt-in M25_TREE=1, implies M25_EAGLE): instead of verifying ONE linear draft per ring
# traversal, verify a whole draft TREE (N nodes, each with its own parent + RoPE position) in a SINGLE forward
# under an ancestor-only attention mask, then commit the longest accepted root->leaf path (tree_spec). A tree
# packs more candidate continuations per (expensive WAN) round-trip than a chain -> more tokens/traversal at the
# same depth. Only the attention mask + per-node RoPE differ from a normal decode (the MoE/MLP stay per-token).
# The coordinator sets BOTH flags; default OFF (the linear-EAGLE / n-gram path is untouched).
M25_TREE = os.environ.get("M25_TREE", "0") != "0"
if M25_TREE:
    M25_EAGLE = True                                 # tree-verify consumes the EAGLE aux + drafter; M25_TREE implies M25_EAGLE
# Per-stage timing stamps (opt-in M25_STAGE_TIMING=1): each stage appends [stage, span_ms, compute_ms] to the
# frame's stage_dt list (forward-accumulated like the EAGLE aux, returned by the tail), so the coordinator can
# split a traversal into stage-local work vs transport+overhead. Deltas are local monotonic — clock-skew-free.
M25_STAGE_TIMING = os.environ.get("M25_STAGE_TIMING", "0") != "0"
EAGLE_AUX_LAYER_IDS = [int(x) for x in os.environ.get("M25_EAGLE_AUX", "1,30,58").split(",")]   # decoder-layer indices (SpecForge convention)
_AUX = {}                                            # layer-id -> last run_block's hidden for that layer, [s, H] (this stage's aux layers only)
# Opt-in fp8 KV (M25_KV_FP8=1): store the batched KV cache as float8_e4m3 (HALF the bf16 footprint -> 2x the
# context/streams that fit) and dequant to bf16 just before SDPA/matmul (no fp8-attention kernel needed — we
# own the read). fp8 is float (relative precision ~6%), and K/V are post-RMSNorm O(1) so no scale is needed;
# the HD=128 dot-product averages the per-element error down ~/sqrt(128). Validate the needle before trusting.
M25_KV_FP8 = os.environ.get("M25_KV_FP8", "0") != "0"
# torch has no index_copy_/scatter_ kernel for Float8_e4m3fn on CUDA, so STORE the fp8 cache as raw uint8 bytes
# (which DO support the scatter/index ops) and bit-reinterpret to fp8 only for the dequant-on-read.
_KVDT = torch.uint8 if M25_KV_FP8 else torch.bfloat16
_F8MAX = 448.0                                       # float8_e4m3fn max finite; clamp to avoid NaN (V is unnormed)
def _kv_enc(t):   # bf16 activation -> storage dtype (clamped fp8 bytes as uint8, or bf16 passthrough)
    return t.clamp(-_F8MAX, _F8MAX).to(torch.float8_e4m3fn).view(torch.uint8) if M25_KV_FP8 else t
def _kv_view(buf):   # reinterpret the FULL contiguous buffer as fp8 for read-slicing (slice.view() on a
    return buf.view(torch.float8_e4m3fn) if M25_KV_FP8 else buf   # non-contiguous slice gives bad strides -> CUDA assert


def _bucket(need):                                  # smallest decode bucket >= need, clamped to MAXLEN
    for b in (2048, 4096, 8192, 16384, 32768, 65536, 131072):
        if b >= need:
            return min(b, M25_KV_MAXLEN)
    return M25_KV_MAXLEN


def _decode_kv_check(starts_max, s):
    """Bound the batched-decode KV write to the static buffer. A stream whose absolute position runs
    past M25_KV_MAXLEN would scatter OUT OF BOUNDS along the MAXLEN axis — a device-side CUDA assert
    that kills the stage (and its warm weights). Raise a clean, recoverable RuntimeError instead
    (mirrors the batched-prefill guard). Returns the total context length (max written index + 1)."""
    total = starts_max + s
    if total > M25_KV_MAXLEN:
        raise RuntimeError(f"batched decode context {total} exceeds M25_KV_MAXLEN {M25_KV_MAXLEN} (raise --kv-maxlen or shorten the prompt)")
    return total
# CUDA-graph decode (opt-in M25_CUDA_GRAPH): capture run_block at a FIXED (s=K+1, bucket) shape so a verify
# block replays as ONE graph — removes per-kernel launch overhead. Needs M25_STATIC_KV + M25_SDPA. Varying
# start_pos is carried into the graph by _GR's STATIC buffers: RoPE slice (cos/sin), index_copy_ positions
# (cp), and a bucketed additive causal mask. Prefill stays eager; default OFF.
#
# WHEN IT PAYS (updated 2026-07-05): the lever recovers kernel-LAUNCH overhead, so its value tracks the
# HOST CPU, not the GPU. On an idle-CPU box with torch 2.11 the eager block is already ~3ms (graph nets
# ~1.05x — the 2026-06-28 "not worth it" verdict); but slow/contended-CPU stages measure 35-50ms/block
# vs 11.5ms (2026-07-03 receipt: compute is CPU-kernel-launch-bound) where a replay recovers ~2-4x of
# block time. Numerics + compatibility:
#   * Capture/replay is CORRECT and FAITHFUL — graph vs eager-manual diff = 0.0. The masked read uses a
#     MANUAL matmul + static additive mask (a microbench showed SDPA+dense-mask falls off flash 8-14x on
#     sm_120; manual is the fastest GRAPHABLE bucketed variant, and attention is a tiny slice of the
#     block). Manual is bit-identical eager-manual<->graph-manual but NOT bit-identical to the eager
#     SDPA-flash decode — the same accepted-kernel-numerics class as fp8 wire; the ring A/B judges
#     accept/g regression. receipts: m25-cudagraph-production / m25-attn-microbench-20260628.
#   * M25_EAGLE is COMPATIBLE: aux capture is baked INTO the graph (a static [s,H] buffer per aux layer
#     in _GraphState, refreshed by a CAPTURED device-side copy on every replay — see GraphRunner). The
#     old hard SystemExit guard (stale-aux poisoning) is gone; research/graph_aux_check.py is the
#     on-box proof (bit-equality + aux freshness across start_pos).
M25_CUDA_GRAPH = os.environ.get("M25_CUDA_GRAPH", "0") != "0"
if M25_CUDA_GRAPH:
    M25_STATIC_KV = True
# BOUND the capture set: each (block size s, context bucket) pair is ONE captured graph, and hybrid
# refeed frames make s VARIABLE (K+1 up to ~K+TREE_DEPTH+2), so unbounded lazy capture could accumulate
# graphs (each pins its own workspace memory) until the stage OOMs. M25_GRAPH_MAX caps TOTAL captured
# graphs process-wide; past it a NEW (s,bucket) runs EAGER (silent fallback, counted in _GRAPH_SKIPPED,
# never a crash) while already-captured shapes keep replaying. See m25_pipe._block for the routing.
M25_GRAPH_MAX = int(os.environ.get("M25_GRAPH_MAX", "16"))
_GRAPH_COUNT = 0        # graphs captured so far, across ALL GraphRunners in this process
_GRAPH_SKIPPED = 0      # blocks run eager because the cap was hit or the bucket's capture failed
# Tree-frame CUDA graphs (M25_TREE_GRAPH=0 = escape hatch): the per-stream tree frames (M25_TREE at
# B>1) ran EAGER stage-side — 154ms summed stage compute vs 45ms for graph-replayed chains (3.4x,
# receipt perstream-trees-ab-20260712, the measured kill reason for trees at B) — because a tree's
# node count N (= re-fed trunk + M drafted nodes) varies per round and defeats fixed-shape capture.
# The unlock: PAD each tree frame to ONE fixed template Npad (M25_TREE_PAD, default derived from the
# drafting config: trunk_max + M rounded up to 8 — 24 at the M=12/depth=8 defaults), capture one
# graph per context bucket in a TreeRowGraphRunner, and carry the per-round tree (row, start,
# parents, pos_ids, n) through _TGraphState's static buffers exactly like _RGraphState carries
# (row, start). THE DUMMY-NODE RULES (the two silent-corruption traps, designed out):
#   * KV: dummy nodes write their k/v CONTIGUOUSLY at [start+n, start+npad) — beyond the frame's
#     read window and exactly the speculative-junk-past-the-frontier class every chain verify frame
#     already leaves at its rejected slots; the dirty-frontier contract overwrites junk before any
#     later frame can read it, so padding adds NO new corruption class. (A sacrificial-column scheme
#     was rejected: it collides with legal live KV at the context cap and needs a cross-cutting
#     bound change.)
#   * VALUES: dummy rows CLONE node 0 (input row, mask row, RoPE position), so their 62-layer hidden
#     trajectory is a real token's — bounded by construction. Zero/garbage dummies are off-manifold:
#     an eventual bf16 overflow -> inf -> _rms -> NaN k/v in the junk slots, and a masked read of NaN
#     poisons every real row (0 * NaN = NaN in probs@V) with valid receipts.
# ONE template (not a ladder): every tree frame verifies in the same kernel/numerics context and the
# graph count stays a priori bounded — tree graphs get their OWN budget (M25_TREE_GRAPH_MAX, default
# 4 = the bucket count at the 16k deploy cap) so lazily-captured chain shapes (variable refeed s)
# can't starve the shapes that pay the 154ms tax. N > Npad falls back eager (counted, never fatal).
M25_TREE_GRAPH = os.environ.get("M25_TREE_GRAPH", "1") != "0"
_TREE_NPAD_DEFAULT = -(-(int(os.environ.get("M25_TREE_M", "12")) + int(os.environ.get("M25_TREE_DEPTH", "8")) + 1) // 8) * 8
M25_TREE_PAD = int(os.environ.get("M25_TREE_PAD", str(_TREE_NPAD_DEFAULT)))
M25_TREE_GRAPH_MAX = int(os.environ.get("M25_TREE_GRAPH_MAX", "4"))
_TREE_GRAPH_COUNT = 0   # tree graphs captured so far (own budget, separate from _GRAPH_COUNT's chain shapes)
# RUNTIME toggle (the per-job A/B lever): the hot path (m25_pipe._block) consults M25_CUDA_GRAPH_ACTIVE,
# not the env constant, so ONE warm ring can interleave graph-on/graph-off arms per job — a stage
# relaunch per arm would reintroduce the time-of-day drift the interleaved methodology exists to kill.
# Initialized from M25_CUDA_GRAPH; flipped by set_graph() off the coordinator's reset op
# ({"graph": true/false}; field absent = keep the current setting, so old coordinators change nothing).
M25_CUDA_GRAPH_ACTIVE = M25_CUDA_GRAPH
DECODE_BUCKETS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
_GR = None        # active _GraphState during capture (None = eager); attn reads its static buffers


def set_graph(on):
    """Set the runtime graph route (reset-op plumbing). Enabling REQUIRES M25_STATIC_KV + M25_SDPA from
    process start — the static KV buffers a graph replays into are allocated at Layer construction — so
    an A/B ring launches with M25_STATIC_KV=1 M25_EAGLE=1 (M25_CUDA_GRAPH unset) and flips graphs per
    job; eager-with-static-KV is bit-identical to the cat path (research/m25_statickv_test.py), keeping
    the graph-off arm representative of master. Without the prereqs a graph=true request is REFUSED
    loudly and ignored — never crash, never silently claim graphs. Returns the resulting setting (the
    tail acks it back to the coordinator, which raises on a mismatch — see m25_pipe._check_reset_ack;
    the refusal string below is GREP-STABLE ("GRAPH REFUSED") because head/middle refusals never reach
    that ack: the bench runbook greps every stage log for it before trusting an arm)."""
    global M25_CUDA_GRAPH_ACTIVE
    on = bool(on)
    if on and not (M25_STATIC_KV and M25_SDPA):
        print("[graph] GRAPH REFUSED: reset asked graph=true but M25_STATIC_KV/M25_SDPA are off "
              "(static buffers are allocated at Layer construction) — staying eager", flush=True)
        return M25_CUDA_GRAPH_ACTIVE
    if M25_CUDA_GRAPH_ACTIVE != on:
        print(f"[graph] decode route -> {'GRAPH' if on else 'EAGER'} (per-job toggle)", flush=True)
    M25_CUDA_GRAPH_ACTIVE = on
    return M25_CUDA_GRAPH_ACTIVE


NORM_TOPK = getattr(cfg, "norm_topk_prob", True)
ROUTED_SCALE = getattr(cfg, "routed_scaling_factor", 1.0)

_idx = json.load(open(f"{DIR}/model.safetensors.index.json"))["weight_map"]
_HD = {}
def raw(n):
    s = _idx[n]
    if s not in _HD:
        _HD[s] = safe_open(f"{DIR}/{s}", "pt", device="cpu")
    return _HD[s].get_tensor(n)


def vllm_ctx():
    global _CTX
    if _CTX is not None:
        return _CTX[1]
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.config import VllmConfig, set_current_vllm_config, get_current_vllm_config
    from vllm.v1.worker.workspace import init_workspace_manager
    torch.cuda.set_device(0)
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://", backend="nccl")
    vcfg = VllmConfig()
    try:                                              # cutlass (native FP4, fast, NON-invariant) | emulation (Triton in-kernel dequant: amortizing + batch-INVARIANT under VLLM_BATCH_INVARIANT=1, sm_120, NVFP4 footprint) | marlin
        # "auto" picks per-arch so a heterogeneous ring needs no per-node env: cutlass is the
        # sm_120 native-FP4 fast path and REFUSES older cards; marlin runs the same NVFP4
        # checkpoint on Ada/Ampere (4090-probed: dequant-in-kernel, ~2x weight VRAM, emulation
        # is sm_120-only Triton). A stage's arch is a node fact, not a ring-wide setting.
        want = os.environ.get("M25_MOE_BACKEND", "auto")
        if want == "auto":
            want = "cutlass" if torch.cuda.get_device_capability(0) >= (12, 0) else "marlin"
            print(f"[stage] moe_backend auto -> {want} (sm_{''.join(map(str, torch.cuda.get_device_capability(0)))})", flush=True)
        vcfg.kernel_config.moe_backend = want
    except Exception as e:
        print("warn moe_backend:", e, flush=True)
    ctx = set_current_vllm_config(vcfg); ctx.__enter__()
    initialize_model_parallel(1); init_workspace_manager(torch.device("cuda"))
    _CTX = (ctx, vcfg)
    return vcfg


_QCFG = None
def quant_config():
    global _QCFG
    if _QCFG is not None:
        return _QCFG
    cfgj = json.load(open(f"{DIR}/config.json"))
    qc = cfgj.get("quantization_config")
    hfq = json.load(open(f"{DIR}/hf_quant_config.json")) if os.path.exists(f"{DIR}/hf_quant_config.json") else None
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
    src = hfq["quantization"] if (hfq and "quantization" in hfq) else qc
    _QCFG = ModelOptNvFp4Config.from_config(src)
    return _QCFG


def _build_moe(li):
    """vLLM NVFP4 FusedMoE for layer li's 256 experts (the m25_moe_probe-proven path)."""
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    Pmoe = f"model.layers.{li}.block_sparse_moe."
    Pexp = Pmoe + "experts."
    suffixes = sorted({k.split(f"{Pexp}0.w1.")[1] for k in _idx if k.startswith(f"{Pexp}0.w1.")})
    eb = raw(Pmoe + "e_score_correction_bias").float().to(dev)
    moe = FusedMoE(num_experts=E, top_k=K, hidden_size=H, intermediate_size=I, params_dtype=torch.bfloat16,
                   renormalize=NORM_TOPK, use_grouped_topk=False, scoring_func="sigmoid",
                   routed_scaling_factor=ROUTED_SCALE, e_score_correction_bias=eb,
                   quant_config=quant_config(), prefix=Pexp[:-1]).to(dev)
    params = dict(moe.named_parameters())
    for e in range(E):
        for proj, shard in [("w1", "w1"), ("w3", "w3"), ("w2", "w2")]:
            grp = "w2" if shard == "w2" else "w13"
            for suf in suffixes:
                name = f"{Pexp}{e}.{proj}.{suf}"
                pname = f"{grp}_{suf}"
                if name in _idx and pname in params:
                    moe.weight_loader(params[pname], raw(name).to(dev), name, shard, e)
    qm = getattr(moe, "quant_method", None) or getattr(moe, "_quant_method", None)   # vLLM renamed quant_method -> _quant_method in newer builds
    qm.process_weights_after_loading(moe)
    gate = raw(Pmoe + "gate.weight").to(torch.bfloat16).to(dev)
    return moe, gate


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)


class Layer:
    """One M2.5 decoder layer: bf16 GQA + NVFP4 MoE, own KV cache (crops to start_pos)."""
    def __init__(self, li):
        self.li = li
        P = f"model.layers.{li}."
        g = lambda n: raw(P + n).to(torch.bfloat16).to(dev)
        self.in_ln = g("input_layernorm.weight")
        self.post_ln = g("post_attention_layernorm.weight")
        self.q_proj = g("self_attn.q_proj.weight"); self.k_proj = g("self_attn.k_proj.weight")
        self.v_proj = g("self_attn.v_proj.weight"); self.o_proj = g("self_attn.o_proj.weight")
        self.q_norm = g("self_attn.q_norm.weight"); self.k_norm = g("self_attn.k_norm.weight")
        self.moe, self.gate = _build_moe(li)
        self.kc = self.vc = None
        if M25_STATIC_KV:                                          # fixed-address buffers (graph/concurrency prereq)
            self.kc = torch.zeros(1, NKV, M25_KV_MAXLEN, HD, dtype=torch.bfloat16, device=dev)
            self.vc = torch.zeros(1, NKV, M25_KV_MAXLEN, HD, dtype=torch.bfloat16, device=dev)
        self.bkc = self.bvc = None
        if M25_BATCH > 1:                                          # [B,NKV,MAXLEN,HD] per-stream KV for continuous batching (fp8 if M25_KV_FP8)
            self.bkc = torch.zeros(M25_BATCH, NKV, M25_KV_MAXLEN, HD, dtype=_KVDT, device=dev)
            self.bvc = torch.zeros(M25_BATCH, NKV, M25_KV_MAXLEN, HD, dtype=_KVDT, device=dev)

    def reset(self):
        if M25_STATIC_KV:
            return                                                # logical reset: writes overwrite at start_pos, reads are :total-bounded (no zeroing needed)
        self.kc = self.vc = None

    def _rms(self, x, w):
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + EPS)).to(x.dtype) * w

    def attn(self, x, start_pos, cos, sin):
        b, s, _ = x.shape
        lin = torch.nn.functional.linear
        q = self._rms(lin(x, self.q_proj), self.q_norm).view(b, s, NH, HD).transpose(1, 2)
        k = self._rms(lin(x, self.k_proj), self.k_norm).view(b, s, NKV, HD).transpose(1, 2)
        v = lin(x, self.v_proj).view(b, s, NKV, HD).transpose(1, 2)
        rd = cos.shape[-1]
        gr = _GR                                                   # CUDA-graph state during capture (None = eager)
        if gr is not None:                                         # graph: RoPE slice comes from a static buffer (start_pos varies, can't bake a Python slice)
            cu = gr.cos.unsqueeze(0).unsqueeze(0); su = gr.sin.unsqueeze(0).unsqueeze(0)
        else:
            cu = cos[start_pos:start_pos + s].unsqueeze(0).unsqueeze(0)   # [1,1,s,rd]
            su = sin[start_pos:start_pos + s].unsqueeze(0).unsqueeze(0)
        def ap(t):
            tr, tp = t[..., :rd], t[..., rd:]
            return torch.cat([tr * cu + _rotate_half(tr) * su, tp], -1)
        q, k = ap(q), ap(k)
        total = start_pos + s
        # amask: the bottom-right causal mask. Eager uses causal_lower_right (a CausalBias flag the kernel
        # reads with no dense tensor — O(s) memory; is_causal is top-left and WRONG). The graphed path can't
        # use it (the bucketed read kc[:,:,:alen] has an unwritten tail at [total:alen] that causal_lower_right
        # mis-aligns to alen-1) — so it uses a static ADDITIVE mask (small for s=K+1, computed before replay).
        if gr is not None:                                         # graphed verify block: static cp write, bucketed read, static additive mask
            self.kc.index_copy_(2, gr.cp, k); self.vc.index_copy_(2, gr.cp, v)
            kcur, vcur, amask = self.kc[:, :, :gr.alen, :], self.vc[:, :, :gr.alen, :], gr.mask
        elif M25_STATIC_KV:                                        # fixed-address write at start_pos; rollback = overwrite + read :total
            if total > M25_KV_MAXLEN:
                raise RuntimeError(f"context {total} exceeds M25_KV_MAXLEN {M25_KV_MAXLEN} (raise it or unset M25_STATIC_KV)")
            cp = torch.arange(start_pos, total, device=dev)
            self.kc.index_copy_(2, cp, k); self.vc.index_copy_(2, cp, v)
            kcur, vcur, amask = self.kc[:, :, :total, :], self.vc[:, :, :total, :], causal_lower_right(s, total)
        else:
            if self.kc is not None and self.kc.shape[2] > start_pos:
                self.kc = self.kc[:, :, :start_pos, :].contiguous(); self.vc = self.vc[:, :, :start_pos, :].contiguous()
            if self.kc is None:
                self.kc, self.vc = k, v
            else:
                self.kc = torch.cat([self.kc, k], 2); self.vc = torch.cat([self.vc, v], 2)
            total = self.kc.shape[2]
            kcur, vcur, amask = self.kc, self.vc, causal_lower_right(s, total)
        if gr is not None:
            # GRAPHED decode: MANUAL matmul + the static additive mask (amask=gr.mask). Microbench (sm_120):
            # SDPA-with-dense-mask falls off flash (8-14x slower); manual is the fastest GRAPHABLE bucketed
            # variant (~2.4x flash) — and since attention is a tiny slice of the block (MoE/projections
            # dominate ~1.9ms/layer), the block graph still nets the launch-overhead win. Manual is also
            # bit-identical eager↔graph (same op) so toggling the graph is safe.
            kk = kcur.repeat_interleave(GRP, dim=1); vv = vcur.repeat_interleave(GRP, dim=1)
            a = torch.matmul(q, kk.transpose(-1, -2)) * SCALING + amask
            o = torch.matmul(torch.softmax(a.float(), -1).to(vv.dtype), vv)
        elif M25_SDPA:
            with sdpa_kernel(_SDPA_BACKENDS):
                o = torch.nn.functional.scaled_dot_product_attention(
                    q, kcur, vcur, attn_mask=amask, scale=SCALING, enable_gqa=True)
        else:                                                          # naive reference path (M25_SDPA=0, A/B; never graphed)
            kk = kcur.repeat_interleave(GRP, dim=1); vv = vcur.repeat_interleave(GRP, dim=1)
            attn = torch.matmul(q, kk.transpose(-1, -2)) * SCALING
            qpos = torch.arange(s, device=dev).view(s, 1) + start_pos
            kpos = torch.arange(total, device=dev).view(1, total)
            attn = attn + torch.where(kpos <= qpos, 0.0, float("-inf")).to(attn.dtype)
            o = torch.matmul(torch.softmax(attn.float(), -1).to(vv.dtype), vv)
        o = o.transpose(1, 2).reshape(b, s, NH * HD)
        return lin(o, self.o_proj)

    def mlp(self, x):
        shp = x.shape
        h = x.reshape(-1, H)
        rl = torch.nn.functional.linear(h, self.gate)   # [T, E]
        return self.moe(h, rl).view(shp)

    def forward(self, x, start_pos, pe):
        cos, sin = pe
        x = x + self.attn(self._rms(x, self.in_ln), start_pos, cos, sin)
        x = x + self.mlp(self._rms(x, self.post_ln))
        return x

    # ---- EAGLE tree-verify (M25_TREE): one forward over a draft tree, ancestor-only attention mask ----
    def attn_tree(self, x, start, pos_ids, mask):
        """TREE-VERIFY attention (single-stream EAGLE). x=[1,N,H] the N drafted tree nodes; pos_ids=[N] long =
        each node's ABSOLUTE RoPE position (siblings share one, so positions are NOT contiguous); mask=
        [1,1,N,start+N] additive bias from tree_spec.build_tree_mask (0 = attend, -inf = block), already on
        device in x's dtype (run_block_tree builds it ONCE and every layer reuses it). Per-node partial RoPE
        via gather (like attn_decode_b), q_norm/k_norm, KV written CROPPED-to-start at slots [start,start+N)
        (a tree node's k/v at start+i overwrites any prior speculative slot there — same crop-to-start as
        attn()), read over [0,start+N). The mask makes every node attend the WHOLE committed prefix [0:start]
        plus its root->node ancestors inside the tree block, never its siblings. Attention is ALWAYS the
        manual broadcast-GQA kernel (_gqa_masked_attend): a dense float mask knocks SDPA off flash on sm_120
        and N is tiny, so manual is both the fast and the bit-reproducible choice (no SDPA backend variance).
        MoE/MLP stay per-token (run_block_tree)."""
        from tree_spec import _gqa_masked_attend, _rope_gather       # tree-path-only dep (pure torch; pushed flat next to this file)
        _, N, _ = x.shape
        lin = torch.nn.functional.linear
        cos, sin = get_pe(); rd = cos.shape[-1]                       # same rotary table attn()/run_block use
        q = self._rms(lin(x, self.q_proj), self.q_norm).view(1, N, NH, HD).transpose(1, 2)
        k = self._rms(lin(x, self.k_proj), self.k_norm).view(1, N, NKV, HD).transpose(1, 2)
        v = lin(x, self.v_proj).view(1, N, NKV, HD).transpose(1, 2)
        q = _rope_gather(q, cos, sin, pos_ids, rd); k = _rope_gather(k, cos, sin, pos_ids, rd)
        total = start + N
        if M25_STATIC_KV:                                             # fixed-address write at [start,total); read :total
            if total > M25_KV_MAXLEN:
                raise RuntimeError(f"tree context {total} exceeds M25_KV_MAXLEN {M25_KV_MAXLEN} (raise it or unset M25_STATIC_KV)")
            cp = torch.arange(start, total, device=dev)
            self.kc.index_copy_(2, cp, k); self.vc.index_copy_(2, cp, v)
            kcur, vcur = self.kc[:, :, :total, :], self.vc[:, :, :total, :]
        else:                                                        # cat path: crop any prior speculative tail to start, then append the N nodes
            if self.kc is not None and self.kc.shape[2] > start:
                self.kc = self.kc[:, :, :start, :].contiguous(); self.vc = self.vc[:, :, :start, :].contiguous()
            if self.kc is None:
                self.kc, self.vc = k, v
            else:
                self.kc = torch.cat([self.kc, k], 2); self.vc = torch.cat([self.vc, v], 2)
            kcur, vcur = self.kc, self.vc
        o = _gqa_masked_attend(q, kcur, vcur, mask, GRP)
        o = o.transpose(1, 2).reshape(1, N, NH * HD)
        return lin(o, self.o_proj)

    # ---- continuous batching (M25_BATCH>1): prefill writes one row; decode batches all rows ----
    def attn_prefill_b(self, x, b, start, cos, sin):
        """PER-STREAM prefill into batch-row b (x: [1, L, H]); SDPA-flash over :total (same as single-stream)."""
        _, s, _ = x.shape
        lin = torch.nn.functional.linear
        q = self._rms(lin(x, self.q_proj), self.q_norm).view(1, s, NH, HD).transpose(1, 2)
        k = self._rms(lin(x, self.k_proj), self.k_norm).view(1, s, NKV, HD).transpose(1, 2)
        v = lin(x, self.v_proj).view(1, s, NKV, HD).transpose(1, 2)
        rd = cos.shape[-1]
        cu = cos[start:start + s].unsqueeze(0).unsqueeze(0); su = sin[start:start + s].unsqueeze(0).unsqueeze(0)
        def ap(t):
            tr, tp = t[..., :rd], t[..., rd:]
            return torch.cat([tr * cu + _rotate_half(tr) * su, tp], -1)
        q, k = ap(q), ap(k)
        total = start + s
        if total > M25_KV_MAXLEN:                      # clean error, not an out-of-bounds CUDA assert that kills the stage
            raise RuntimeError(f"batched prefill context {total} exceeds M25_KV_MAXLEN {M25_KV_MAXLEN} (raise --kv-maxlen or shorten the prompt)")
        cp = torch.arange(start, total, device=dev)
        self.bkc[b:b + 1].index_copy_(2, cp, _kv_enc(k)); self.bvc[b:b + 1].index_copy_(2, cp, _kv_enc(v))   # b:b+1 view → row b (fp8 bytes if M25_KV_FP8)
        with sdpa_kernel(_SDPA_BACKENDS):
            o = torch.nn.functional.scaled_dot_product_attention(
                q, _kv_view(self.bkc)[b:b + 1, :, :total].to(torch.bfloat16), _kv_view(self.bvc)[b:b + 1, :, :total].to(torch.bfloat16),
                attn_mask=causal_lower_right(s, total), scale=SCALING, enable_gqa=True)
        return lin(o.transpose(1, 2).reshape(1, s, NH * HD), self.o_proj)

    def attn_decode_b(self, x, starts, cos, sin):
        """BATCHED decode (x: [B, s, H], starts: [B] long). Per-stream RoPE/scatter/mask (batchverify
        pattern, proven bit-exact vs solo). Manual matmul over a shared bucket; per-stream mask isolates
        each stream + zeros its unwritten tail. Under a BatchGraphRunner capture (_GR set, batched
        state) the varying starts come in through STATIC buffers (cp/cos/sin/mask, refreshed by set()
        before every replay) — the same design as attn()'s solo graph branch."""
        B, s, _ = x.shape
        lin = torch.nn.functional.linear
        q = self._rms(lin(x, self.q_proj), self.q_norm).view(B, s, NH, HD).transpose(1, 2)
        k = self._rms(lin(x, self.k_proj), self.k_norm).view(B, s, NKV, HD).transpose(1, 2)
        v = lin(x, self.v_proj).view(B, s, NKV, HD).transpose(1, 2)
        rd = cos.shape[-1]
        gr = _GR                                     # batched-graph capture state (None = eager)
        if gr is not None:                           # graph: statics carry the varying starts (attn() pattern)
            cp = gr.cp                               # [B,s] abs positions
            cu, su = gr.cos, gr.sin                  # [B,1,s,rd] per-stream RoPE, pre-gathered by set()
        else:
            cp = starts.view(B, 1) + torch.arange(s, device=dev).view(1, s)           # [B,s] abs positions
            cu = cos[cp].unsqueeze(1); su = sin[cp].unsqueeze(1)                       # [B,1,s,rd] per-stream RoPE
        def ap(t):
            tr, tp = t[..., :rd], t[..., rd:]
            return torch.cat([tr * cu + _rotate_half(tr) * su, tp], -1)
        q, k = ap(q), ap(k)
        idx = cp.view(B, 1, s, 1).expand(B, NKV, s, HD)
        if gr is None:
            mx = _decode_kv_check(int(starts.max().item()), s)   # clean error, not an OOB scatter CUDA-assert that kills the stage
            alen = _bucket(mx)
        else:
            alen = gr.alen                           # bounds-checked by the runner BEFORE replay (a host
                                                     # sync is illegal inside capture)
        self.bkc[:B].scatter_(2, idx, _kv_enc(k)); self.bvc[:B].scatter_(2, idx, _kv_enc(v))   # write all rows (fp8 bytes if M25_KV_FP8)
        # HYBRID: small/medium context -> ONE batched manual matmul (fast, ~3x batched throughput); big context ->
        # per-stream flash SDPA (the GQA-repeat [B,48,alen,HD] would OOM at big alen, and a batched dense mask
        # forces SDPA->MATH which also OOMs). Threshold on the GQA-repeat size (kk+vv bf16 bytes).
        if gr is not None:                           # graphed: manual matmul + STATIC additive mask — the
            kk = _kv_view(self.bkc)[:B, :, :alen].to(torch.bfloat16).repeat_interleave(GRP, 1)   # same kernel
            vv = _kv_view(self.bvc)[:B, :, :alen].to(torch.bfloat16).repeat_interleave(GRP, 1)   # class as the
            a = torch.matmul(q, kk.transpose(-1, -2)) * SCALING + gr.mask                        # eager manual
            o = torch.matmul(torch.softmax(a.float(), -1).to(vv.dtype), vv)                      # path below
        elif B * NH * alen * HD * 4 <= 1_400_000_000:                              # batched manual matmul path
            kk = _kv_view(self.bkc)[:B, :, :alen].to(torch.bfloat16).repeat_interleave(GRP, 1)
            vv = _kv_view(self.bvc)[:B, :, :alen].to(torch.bfloat16).repeat_interleave(GRP, 1)
            cols = torch.arange(alen, device=dev).view(1, 1, alen)
            amask = torch.where(cols <= cp[:, :, None], 0.0, float("-inf")).to(torch.bfloat16)[:, None]
            a = torch.matmul(q, kk.transpose(-1, -2)) * SCALING + amask
            o = torch.matmul(torch.softmax(a.float(), -1).to(vv.dtype), vv)
        else:                                                                      # big-ctx: per-stream flash SDPA (memory-safe)
            outs = []
            for b in range(B):
                total = int(starts[b].item()) + s
                with sdpa_kernel(_SDPA_BACKENDS):
                    outs.append(torch.nn.functional.scaled_dot_product_attention(
                        q[b:b + 1], _kv_view(self.bkc)[b:b + 1, :, :total].to(torch.bfloat16),
                        _kv_view(self.bvc)[b:b + 1, :, :total].to(torch.bfloat16),
                        attn_mask=causal_lower_right(s, total), scale=SCALING, enable_gqa=True))
            o = torch.cat(outs, 0)
        return lin(o.transpose(1, 2).reshape(B, s, NH * HD), self.o_proj)

    def mlp_b(self, x):                                                            # [B,s,H] -> [B,s,H]
        if M25_BATCH_MOE:                                                          # ONE grouped GEMM over all B*s tokens (amortizes expert-weight reads ~B x; trades per-stream bit-exactness)
            B, s, _ = x.shape
            h = x.reshape(B * s, H)
            rl = torch.nn.functional.linear(h, self.gate)
            return self.moe(h, rl).view(B, s, H)
        return torch.cat([self.mlp(x[b:b + 1]) for b in range(x.shape[0])], 0)     # per-stream MoE (token-count invariant -> verifiable, but B x the weight reads)

    def forward_prefill_b(self, x, b, start, pe):
        cos, sin = pe
        x = x + self.attn_prefill_b(self._rms(x, self.in_ln), b, start, cos, sin)
        x = x + self.mlp(self._rms(x, self.post_ln))                              # 1 stream, L tokens == solo
        return x

    def forward_decode_b(self, x, starts, pe):
        cos, sin = pe
        x = x + self.attn_decode_b(self._rms(x, self.in_ln), starts, cos, sin)
        x = x + self.mlp_b(self._rms(x, self.post_ln))                            # per-stream MoE
        return x

    # ---- de-lockstep (M25_DELOCKSTEP): ONE stream's solo-style decode frame against KV ROW b ----
    def attn_decode_row(self, x, row, start, cos, sin):
        """DE-LOCKSTEP decode: x [1,s,H] for ONE stream against batched-KV ROW `row`. Math mirrors
        attn_decode_b's manual-matmul branch at B=1 (bit-equal target), but the row is addressed
        DYNAMICALLY through a flattened [B*NKV, MAXLEN, HD] view with index tensors — eager builds
        them from ints; a graph capture (_GR = _RGraphState) reads STATIC buffers refreshed per
        replay, so ONE captured graph serves EVERY row (a per-row graph set would pin B x the pool
        memory). Writes land only in row b (advanced-index put); reads gather [NKV, alen, HD] via
        static row+col indices."""
        _, s, _ = x.shape
        lin = torch.nn.functional.linear
        q = self._rms(lin(x, self.q_proj), self.q_norm).view(1, s, NH, HD).transpose(1, 2)
        k = self._rms(lin(x, self.k_proj), self.k_norm).view(1, s, NKV, HD).transpose(1, 2)
        v = lin(x, self.v_proj).view(1, s, NKV, HD).transpose(1, 2)
        rd = cos.shape[-1]
        gr = _GR
        if gr is not None and hasattr(gr, "rows"):     # row-graph capture/replay: statics carry row+start
            rows_i, cp, cu, su, alen = gr.rows, gr.cp, gr.cos, gr.sin, gr.alen
        else:
            gr = None                                  # a foreign capture state never routes this path
            cp = start + torch.arange(s, device=dev)                       # [s] abs positions
            rows_i = row * NKV + torch.arange(NKV, device=dev)             # [NKV] flat kv-head rows
            cu = cos[cp].unsqueeze(0).unsqueeze(0); su = sin[cp].unsqueeze(0).unsqueeze(0)   # [1,1,s,rd]
            total = start + s
            if total > M25_KV_MAXLEN:                  # clean error, never an OOB put that kills the stage
                raise RuntimeError(f"row decode context {total} exceeds M25_KV_MAXLEN {M25_KV_MAXLEN}")
            alen = _bucket(total)
        def ap(t):
            tr, tp = t[..., :rd], t[..., rd:]
            return torch.cat([tr * cu + _rotate_half(tr) * su, tp], -1)
        q, k = ap(q), ap(k)
        kf = self.bkc.view(-1, M25_KV_MAXLEN, HD)      # [B*NKV, MAXLEN, HD] flat views (dtype-preserving)
        vf = self.bvc.view(-1, M25_KV_MAXLEN, HD)
        kf[rows_i[:, None], cp[None, :]] = _kv_enc(k[0])   # k[0] is [NKV,s,HD] already (review MAJOR-2:
        vf[rows_i[:, None], cp[None, :]] = _kv_enc(v[0])   # a transpose here crashes at s!=NKV and writes
                                                           # TRANSPOSED KV at the s==NKV coincidence)
        if gr is not None:                             # graphed: static cols gather + static additive mask
            kk = _kv_view(kf)[rows_i[:, None], gr.cols[None, :]].to(torch.bfloat16)   # [NKV, alen, HD]
            vv = _kv_view(vf)[rows_i[:, None], gr.cols[None, :]].to(torch.bfloat16)
            amask = gr.mask                            # [1,1,s,alen] additive
        else:
            cols = torch.arange(alen, device=dev)
            kk = _kv_view(kf)[rows_i[:, None], cols[None, :]].to(torch.bfloat16)
            vv = _kv_view(vf)[rows_i[:, None], cols[None, :]].to(torch.bfloat16)
            amask = torch.where(cols.view(1, alen) <= cp.view(s, 1), 0.0,
                                float("-inf")).to(torch.bfloat16)[None, None]
        kk = kk.repeat_interleave(GRP, 0); vv = vv.repeat_interleave(GRP, 0)          # [NH, alen, HD]
        a = torch.matmul(q[0], kk.transpose(-1, -2)) * SCALING + amask[0]             # [NH, s, alen]
        o = torch.matmul(torch.softmax(a.float(), -1).to(vv.dtype), vv)               # [NH, s, HD]
        return lin(o.transpose(0, 1).reshape(1, s, NH * HD), self.o_proj)

    def forward_decode_row(self, x, row, start, pe):
        cos, sin = pe
        x = x + self.attn_decode_row(self._rms(x, self.in_ln), row, start, cos, sin)
        x = x + self.mlp(self._rms(x, self.post_ln))   # ONE stream, s tokens — the solo MoE path
        return x

    def attn_tree_row(self, x, row, start, pos_ids, mask):
        """TREE-VERIFY attention for ONE de-lockstep stream against batched-KV ROW `row`: attn_tree's
        math (per-node RoPE gather — siblings share a position, KV written at [start,start+N) and read
        over [0,start+N) under the ancestor-only additive mask, manual broadcast-GQA) with
        attn_decode_row's flat-view row addressing on the shared [B,NKV,MAXLEN,HD] buffers (fp8 bytes
        under M25_KV_FP8, like every batched-KV op). A tree node's k/v at start+i overwrites any prior
        speculative slot there — the same crop-to-start semantics as attn_tree — and only row `row` is
        touched. Eager builds cp/RoPE/mask per round; under a TreeRowGraphRunner capture (_GR is a
        _TGraphState — the ONLY state class with `wcp`) the varying (row, start, tree shape) come in
        through STATIC buffers refreshed by set() before every replay: wcp writes the frame's Npad
        rows CONTIGUOUSLY at [start, start+npad) (real nodes at their exact eager slots, dummies as
        speculative junk past the read window — see the M25_TREE_GRAPH module comment), the padded
        mask blocks the pad/bucket-tail columns for every real row, and the read spans the fixed
        bucket alen (masked past total) instead of the exact total."""
        from tree_spec import _gqa_masked_attend, _rope_gather
        _, N, _ = x.shape
        lin = torch.nn.functional.linear
        q = self._rms(lin(x, self.q_proj), self.q_norm).view(1, N, NH, HD).transpose(1, 2)
        k = self._rms(lin(x, self.k_proj), self.k_norm).view(1, N, NKV, HD).transpose(1, 2)
        v = lin(x, self.v_proj).view(1, N, NKV, HD).transpose(1, 2)
        gr = _GR
        if gr is not None and hasattr(gr, "wcp"):      # tree-graph capture/replay: statics carry row/start/tree
            rows_i, wcp, cols, mask = gr.rows, gr.wcp, gr.cols, gr.mask
            cu, su = gr.cos, gr.sin                    # [1,1,Npad,rd] pre-gathered per-node RoPE rows
            rd = cu.shape[-1]
            tr, tp = q[..., :rd], q[..., rd:]          # _rope_gather's exact partial-RoPE math on the
            q = torch.cat([tr * cu + _rotate_half(tr) * su, tp], -1)   # static rows (bit-equal ops)
            tr, tp = k[..., :rd], k[..., rd:]
            k = torch.cat([tr * cu + _rotate_half(tr) * su, tp], -1)
        else:
            gr = None                                  # a foreign capture state never routes this path
            cos, sin = get_pe(); rd = cos.shape[-1]
            q = _rope_gather(q, cos, sin, pos_ids, rd); k = _rope_gather(k, cos, sin, pos_ids, rd)
            total = start + N
            if total > M25_KV_MAXLEN:                  # clean error, never an OOB put that kills the stage
                raise RuntimeError(f"tree row context {total} exceeds M25_KV_MAXLEN {M25_KV_MAXLEN}")
            wcp = torch.arange(start, total, device=dev)
            rows_i = row * NKV + torch.arange(NKV, device=dev)
            cols = torch.arange(total, device=dev)
        kf = self.bkc.view(-1, M25_KV_MAXLEN, HD)      # [B*NKV, MAXLEN, HD] flat views (dtype-preserving)
        vf = self.bvc.view(-1, M25_KV_MAXLEN, HD)
        kf[rows_i[:, None], wcp[None, :]] = _kv_enc(k[0])   # k[0] is [NKV,N,HD] already (attn_decode_row's
        vf[rows_i[:, None], wcp[None, :]] = _kv_enc(v[0])   # MAJOR-2-proofed write, same op)
        kcur = _kv_view(kf)[rows_i[:, None], cols[None, :]].to(torch.bfloat16).unsqueeze(0)   # [1,NKV,total|alen,HD]
        vcur = _kv_view(vf)[rows_i[:, None], cols[None, :]].to(torch.bfloat16).unsqueeze(0)
        o = _gqa_masked_attend(q, kcur, vcur, mask, GRP)
        o = o.transpose(1, 2).reshape(1, N, NH * HD)
        return lin(o, self.o_proj)


_PE = None
# Rotary table length. MUST cover the full context: attn() indexes cos[start_pos:start_pos+s],
# so a table shorter than the prompt+gen length silently returns a short/empty slice (garbage RoPE)
# the moment a position exceeds it. The old hard-coded 8192 broke any >8k context (incl. the runbook's
# >=30k long-ctx validation). Default 131072 matches the coordinator's max_ctx; bump via M25_MAX_POS.
_MAXPOS = int(os.environ.get("M25_MAX_POS", "131072"))
def get_pe(maxpos=None):
    global _PE
    if _PE is None:
        mp = maxpos or _MAXPOS
        rot = M.MiniMaxM2RotaryEmbedding(cfg).to(dev)
        dummy = torch.zeros(1, 1, H, dtype=torch.bfloat16, device=dev)
        pos = torch.arange(mp, device=dev).unsqueeze(0)
        cos, sin = rot(dummy, pos)
        _PE = (cos[0], sin[0])                                       # [mp, 64]
    return _PE


def run_block(layers, start_pos, h, vcfg):
    from vllm.forward_context import set_forward_context
    pe = get_pe()
    with torch.no_grad(), set_forward_context(None, vcfg):
        for L in layers:
            h = L.forward(h, start_pos, pe)
            if M25_EAGLE and L.li in EAGLE_AUX_LAYER_IDS:   # snapshot the OUTPUT residual stream of decoder layers [1,30,58] for the EAGLE head
                _AUX[L.li] = h[0].detach().to(torch.bfloat16)
    return h


def run_block_tree(layers, start, h, vcfg, parents, pos_ids):   # EAGLE tree-verify: one forward over the draft tree
    """Run the N drafted tree nodes (h: [1,N,H]) through this stage's layers in ONE forward under an
    ancestor-only attention mask, so each node attends the committed prefix [0:start] + its root->node chain
    (never its siblings). `parents` [N] (-1 = the committed anchor) and `pos_ids` [N] (per-node ABSOLUTE RoPE
    position, siblings shared) describe the tree. The additive mask is built ONCE per stage-call, on device in
    h's dtype, and reused across every layer (the old branch built fp32-on-CPU and re-cast per layer);
    attn_tree does attention, the MoE/MLP stay per-token (the tree only changes attention). Same residual
    structure as Layer.forward; captures EAGLE aux exactly like run_block."""
    from vllm.forward_context import set_forward_context
    from tree_spec import build_tree_mask
    N = h.shape[1]
    pos_ids = torch.as_tensor(pos_ids, dtype=torch.long, device=dev)
    depths = (pos_ids - (start - 1)).tolist()                  # build_tree_mask wants depths; pos_ids == (start-1)+depth (we keep pos_ids for RoPE)
    mask, _ = build_tree_mask(parents, depths, start, N)       # [1,1,N,start+N] additive bias; returned positions ignored (pos_ids drive RoPE)
    mask = mask.to(h.dtype).to(dev)
    with torch.no_grad(), set_forward_context(None, vcfg):
        for L in layers:
            h = h + L.attn_tree(L._rms(h, L.in_ln), start, pos_ids, mask)
            h = h + L.mlp(L._rms(h, L.post_ln))
            if M25_EAGLE and L.li in EAGLE_AUX_LAYER_IDS:       # snapshot the OUTPUT residual stream for the EAGLE head (same as run_block)
                _AUX[L.li] = h[0].detach().to(torch.bfloat16)
    return h


def run_block_prefill_b(layers, b, start, h, vcfg):     # continuous batching: prefill stream into row b
    from vllm.forward_context import set_forward_context
    pe = get_pe()
    with torch.no_grad(), set_forward_context(None, vcfg):
        for L in layers:
            h = L.forward_prefill_b(h, b, start, pe)
            if M25_EAGLE and L.li in EAGLE_AUX_LAYER_IDS:   # per-stream prefill aux ([s,H], row b) — same
                _AUX[L.li] = h[0].detach().to(torch.bfloat16)   # contract as run_block (h is [1,s,H])
    return h


def run_block_decode_b(layers, starts, h, vcfg):        # continuous batching: batched decode, all rows
    from vllm.forward_context import set_forward_context
    pe = get_pe()
    with torch.no_grad(), set_forward_context(None, vcfg):
        for L in layers:
            h = L.forward_decode_b(h, starts, pe)
            if M25_EAGLE and L.li in EAGLE_AUX_LAYER_IDS:   # batched aux: keep EVERY row — [B,s,H] (the
                _AUX[L.li] = h.detach().to(torch.bfloat16)  # coordinator slices per stream for its drafter)
    return h


class _GraphState:
    """Static per-block buffers that carry the varying start_pos INTO a captured graph: the RoPE slice
    (cos/sin), the index_copy_ write positions (cp), and the bucketed additive causal mask. set() updates
    them IN PLACE (the same addresses the graph captured), so a replay attends the correct span at the new
    start_pos. mask is [1,1,s,alen] additive bf16 — tiny for s=K+1, so materializing it is free.
    `aux_ids` (M25_EAGLE) adds one static [s,H] bf16 OUTPUT buffer per EAGLE aux layer in the runner's
    range: GraphRunner._layers copy_()s each aux layer's output residual into it during capture, so the
    copy is PART of the graph and every replay refreshes the buffer — fresh position-correct aux, never
    the stale prefill aux that used to make M25_CUDA_GRAPH+M25_EAGLE a hard SystemExit."""
    def __init__(self, s, alen, rd, dv, aux_ids=()):
        self.s, self.alen = s, alen
        self.cos = torch.zeros(s, rd, dtype=torch.bfloat16, device=dv)
        self.sin = torch.zeros(s, rd, dtype=torch.bfloat16, device=dv)
        self.cp = torch.zeros(s, dtype=torch.long, device=dv)
        self.mask = torch.zeros(1, 1, s, alen, dtype=torch.bfloat16, device=dv)
        self.aux = {li: torch.zeros(s, H, dtype=torch.bfloat16, device=dv) for li in aux_ids}
        self._kpos = torch.arange(alen, device=dv).view(1, alen)
        self._ar = torch.arange(s, device=dv)

    def set(self, start_pos, full_cos, full_sin):
        self.cos.copy_(full_cos[start_pos:start_pos + self.s])
        self.sin.copy_(full_sin[start_pos:start_pos + self.s])
        self.cp.copy_(self._ar + start_pos)
        qpos = (self._ar + start_pos).view(self.s, 1)                  # abs query positions
        self.mask.copy_(torch.where(self._kpos <= qpos, 0.0, float("-inf")).to(torch.bfloat16)[None, None])


class GraphRunner:
    """Capture + replay a CUDA graph of a stage's run_block at a FIXED verify-block shape (s=K+1), one
    graph per context bucket. Opt-in M25_CUDA_GRAPH; the serve loop routes fixed-shape verify blocks here
    and leaves prefill eager. BIT-EQUIVALENCE to eager is a HARD correctness gate — the graphed stage is on
    the spec-decode VERIFY path, so a capture bug corrupts committed output, not just a slow number."""
    def __init__(self, layers, vcfg, s, dv=dev):
        assert M25_STATIC_KV and M25_SDPA, "M25_CUDA_GRAPH requires M25_STATIC_KV + M25_SDPA"
        self.layers, self.vcfg, self.s, self.dv = layers, vcfg, s, dv
        self.cos, self.sin = get_pe(); self.rd = self.cos.shape[-1]
        self.graphs = {}                                              # bucket alen -> (graph, h_static, state, out_static)
        self.eager = set()                                            # buckets whose capture FAILED -> permanently eager
        self.aux_ids = [L.li for L in layers if M25_EAGLE and L.li in EAGLE_AUX_LAYER_IDS]   # aux layers this stage publishes

    def _bucket(self, total):
        for b in DECODE_BUCKETS:
            if b >= total:
                return min(b, M25_KV_MAXLEN)
        return M25_KV_MAXLEN

    def _layers(self, h):
        st = _GR                                                     # the _GraphState being captured (set by _capture)
        for L in self.layers:
            h = L.forward(h, 0, (self.cos, self.sin))                # start_pos unused in graph mode (attn reads _GR)
            if L.li in st.aux:                                       # EAGLE aux: a device-side copy into the static
                st.aux[L.li].copy_(h[0])                             # buffer, CAPTURED -> every replay refreshes it
        return h

    def _capture(self, alen):
        from vllm.forward_context import set_forward_context
        global _GR, _GRAPH_COUNT
        h = (torch.randn(1, self.s, H, device=self.dv) * 0.1).to(torch.bfloat16)   # static input buffer
        st = _GraphState(self.s, alen, self.rd, self.dv, self.aux_ids)
        st.set(alen - self.s, self.cos, self.sin)                    # capture-time start_pos (total == alen)
        _GR = st
        try:
            side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.no_grad(), set_forward_context(None, self.vcfg):
                for _ in range(3):
                    self._layers(h)                                  # warm-up before capture
            torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.no_grad(), set_forward_context(None, self.vcfg):
                out = self._layers(h)
        finally:
            _GR = None                                               # capture done; attn back to eager for prefill
        self.graphs[alen] = (g, h, st, out)
        _GRAPH_COUNT += 1                                            # counts against the process-wide M25_GRAPH_MAX

    def run(self, start_pos, x):
        """Run one verify/decode block at start_pos through the graph. Returns the STATIC output buffer —
        the caller must consume/copy it before the next run (the serve loop sends .cpu()). Bit-identical
        to eager-MANUAL attention (see the module comment for the SDPA-flash numerics class). OOM-SAFE:
        a capture failure mid-serve marks the bucket permanently eager and falls back to run_block — a
        stage must never die from graph capture."""
        global _GRAPH_SKIPPED
        total = start_pos + self.s
        if total > M25_KV_MAXLEN:                    # host-side bound (Row/BatchGraphRunner have it): a replay's
            # captured index_copy_ reads st.cp, which set() would fill with positions >= MAXLEN — an OOB
            # device assert that kills the stage. Raise the same clean, recoverable error as the eager path.
            raise RuntimeError(f"context {total} exceeds M25_KV_MAXLEN {M25_KV_MAXLEN} (raise --kv-maxlen or shorten the prompt)")
        alen = self._bucket(total)
        if alen not in self.graphs:
            if alen in self.eager:
                _GRAPH_SKIPPED += 1
                return run_block(self.layers, start_pos, x, self.vcfg)
            try:
                self._capture(alen)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                # A warmup/capture failure can leave SIDE-STREAM work in flight (the exception
                # propagates before current_stream().wait_stream(side)): warmup's garbage
                # kc.index_copy_ at [alen-s, alen) racing the eager fallback's real KV writes on the
                # default stream would corrupt committed tokens WITH valid receipts. Drain the device
                # BEFORE any eager work. (The `alen in self.eager` path above needs no sync: it runs
                # no side-stream work, and the first failure already drained here.)
                torch.cuda.synchronize()
                self.eager.add(alen); _GRAPH_SKIPPED += 1
                print(f"[graph] capture failed (s={self.s}, alen={alen}): {type(e).__name__}: {e} "
                      f"-> bucket marked permanently eager", flush=True)
                return run_block(self.layers, start_pos, x, self.vcfg)
        g, h, st, out = self.graphs[alen]
        st.set(start_pos, self.cos, self.sin)                        # update varying-start_pos buffers IN PLACE
        h.copy_(x)
        g.replay(); torch.cuda.synchronize()
        # Publish this replay's aux (M25_EAGLE), mirroring run_block's _AUX contract. _AUX[li] ALIASES
        # the static buffer (no clone): the serve loop consumes _AUX synchronously via _merge_aux
        # (immediate .cpu()/fp8-pack) before the next replay can overwrite it.
        for li in self.aux_ids:
            _AUX[li] = st.aux[li]
        return out


def run_block_decode_row(layers, row, start, h, vcfg):    # de-lockstep: ONE stream's [1,s,H] decode frame
    from vllm.forward_context import set_forward_context
    pe = get_pe()
    with torch.no_grad(), set_forward_context(None, vcfg):
        for L in layers:
            h = L.forward_decode_row(h, row, start, pe)
            if M25_EAGLE and L.li in EAGLE_AUX_LAYER_IDS:   # SOLO-shaped aux ([s,H]) — the coordinator's
                _AUX[L.li] = h[0].detach().to(torch.bfloat16)   # per-stream drafter consumes it like solo
    return h


def run_block_tree_row(layers, row, start, h, vcfg, parents, pos_ids):
    """De-lockstep tree-verify: ONE stream's drafted tree (h: [1,N,H]) in one forward against KV ROW
    `row` — run_block_tree's ancestor-only mask + per-node RoPE positions, attn_tree_row's row
    addressing. MoE/MLP stay per-token on the solo path (the tree only changes attention); aux
    snapshots SOLO-shaped [N,H], like run_block_decode_row — the coordinator's per-stream drafter
    gathers by node index (_eagle_aux_nodes)."""
    from vllm.forward_context import set_forward_context
    from tree_spec import build_tree_mask
    N = h.shape[1]
    pos_ids = torch.as_tensor(pos_ids, dtype=torch.long, device=dev)
    depths = (pos_ids - (start - 1)).tolist()                  # pos_ids == (start-1)+depth (run_block_tree's rule)
    mask, _ = build_tree_mask(parents, depths, start, N)       # [1,1,N,start+N] additive bias
    mask = mask.to(h.dtype).to(dev)
    with torch.no_grad(), set_forward_context(None, vcfg):
        for L in layers:
            h = h + L.attn_tree_row(L._rms(h, L.in_ln), row, start, pos_ids, mask)
            h = h + L.mlp(L._rms(h, L.post_ln))
            if M25_EAGLE and L.li in EAGLE_AUX_LAYER_IDS:
                _AUX[L.li] = h[0].detach().to(torch.bfloat16)
    return h


class _RGraphState:
    """Static buffers for the de-lockstep ROW graph: the row index (rows = row*NKV + arange(NKV)),
    abs positions cp [s], RoPE rows cos/sin [1,1,s,rd], additive mask [1,1,s,alen], and the fixed
    column index cols [alen] the KV gather reads through. set(row, start) refreshes row+start IN
    PLACE — ONE captured graph serves every KV row and every start within the bucket."""
    def __init__(self, s, alen, rd, dv, aux_ids=()):
        self.s, self.alen = s, alen
        self.rows = torch.zeros(NKV, dtype=torch.long, device=dv)
        self.cp = torch.zeros(s, dtype=torch.long, device=dv)
        self.cos = torch.zeros(1, 1, s, rd, dtype=torch.bfloat16, device=dv)
        self.sin = torch.zeros(1, 1, s, rd, dtype=torch.bfloat16, device=dv)
        self.mask = torch.zeros(1, 1, s, alen, dtype=torch.bfloat16, device=dv)
        self.cols = torch.arange(alen, device=dv)
        self.aux = {li: torch.zeros(s, H, dtype=torch.bfloat16, device=dv) for li in aux_ids}
        self._ar = torch.arange(s, device=dv)
        self._nkv = torch.arange(NKV, device=dv)

    def set(self, row, start, full_cos, full_sin):
        self.rows.copy_(row * NKV + self._nkv)
        cp = start + self._ar
        self.cp.copy_(cp)
        self.cos.copy_(full_cos[cp][None, None]); self.sin.copy_(full_sin[cp][None, None])
        self.mask.copy_(torch.where(self.cols.view(1, self.alen) <= cp.view(self.s, 1), 0.0,
                                    float("-inf")).to(torch.bfloat16)[None, None])


class RowGraphRunner:
    """Capture + replay the de-lockstep row-decode block at fixed [1, s=K+1], one graph per context
    bucket, serving EVERY KV row through _RGraphState's static row/start buffers. Same safety
    contract as Batch/GraphRunner: host-side bounds check, free-VRAM pre-check, capture failure ->
    LOUD permanent-eager, counts against M25_GRAPH_MAX."""
    def __init__(self, layers, vcfg, s, dv=dev):
        assert M25_BATCH > 1, "row graphs need the launch-time [B,...] KV rows"
        self.layers, self.vcfg, self.s, self.dv = layers, vcfg, s, dv
        self.cos, self.sin = get_pe(); self.rd = self.cos.shape[-1]
        self.graphs = {}
        self.eager = set()
        self.aux_ids = [L.li for L in layers if M25_EAGLE and L.li in EAGLE_AUX_LAYER_IDS]

    def _bucket(self, total):
        for b in DECODE_BUCKETS:
            if b >= total:
                return min(b, M25_KV_MAXLEN)
        return M25_KV_MAXLEN

    def _layers(self, h):
        st = _GR
        for L in self.layers:
            h = L.forward_decode_row(h, 0, 0, (self.cos, self.sin))   # row/start unused: attn reads _GR
            if L.li in st.aux:
                st.aux[L.li].copy_(h[0])
        return h

    def _capture(self, alen):
        from vllm.forward_context import set_forward_context
        global _GR, _GRAPH_COUNT
        need = 2 * NH * alen * HD * 2 + 2 * NH * self.s * alen * 4 + (1 << 30)   # kk/vv repeat + scores + margin
        free = torch.cuda.mem_get_info()[0]
        if free < need:
            raise RuntimeError(f"free VRAM {free / 1e9:.1f}GB < row-capture estimate {need / 1e9:.1f}GB")
        h = (torch.randn(1, self.s, H, device=self.dv) * 0.1).to(torch.bfloat16)
        st = _RGraphState(self.s, alen, self.rd, self.dv, self.aux_ids)
        st.set(0, alen - self.s, self.cos, self.sin)
        # Capture writes garbage k/v into LIVE row 0 at [alen-s, alen) — and unlike Batch/GraphRunner,
        # the trigger is ONE stream's frontier, so row 0's committed KV may sit at/above those slots
        # (review MAJOR-3: a short stream's capture destroyed a long stream's prompt KV — corrupted
        # output, valid receipts). Save row 0's slots per layer and restore AFTER the device is
        # drained (a restore before the sync could be overwritten by in-flight side-stream garbage).
        lo = alen - self.s
        saved = [(L, L.bkc[0, :, lo:alen].clone(), L.bvc[0, :, lo:alen].clone()) for L in self.layers]
        _GR = st
        try:
            side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.no_grad(), set_forward_context(None, self.vcfg):
                for _ in range(3):
                    self._layers(h)
            torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.no_grad(), set_forward_context(None, self.vcfg):
                out = self._layers(h)
        finally:
            _GR = None
            torch.cuda.synchronize()                 # drain BEFORE restoring live KV (both paths)
            for L, kb, vb in saved:
                L.bkc[0, :, lo:alen] = kb; L.bvc[0, :, lo:alen] = vb
        self.graphs[alen] = (g, h, st, out)
        _GRAPH_COUNT += 1

    def run(self, row, start, x):
        global _GRAPH_SKIPPED
        total = start + self.s
        if total > M25_KV_MAXLEN:
            raise RuntimeError(f"row decode context {total} exceeds M25_KV_MAXLEN {M25_KV_MAXLEN}")
        alen = self._bucket(total)
        if alen in self.eager or NH * alen * HD * 4 > 1_400_000_000:
            _GRAPH_SKIPPED += 1
            return run_block_decode_row(self.layers, row, start, x, self.vcfg)
        if alen not in self.graphs:
            try:
                self._capture(alen)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                torch.cuda.synchronize()
                self.eager.add(alen); _GRAPH_SKIPPED += 1
                print(f"[graph] row capture failed (s={self.s}, alen={alen}): {type(e).__name__}: {e} "
                      f"-> bucket marked permanently eager", flush=True)
                return run_block_decode_row(self.layers, row, start, x, self.vcfg)
        g, h, st, out = self.graphs[alen]
        st.set(row, start, self.cos, self.sin)
        h.copy_(x)
        g.replay(); torch.cuda.synchronize()
        for li in self.aux_ids:                        # solo-shaped [s,H] aux, consumed synchronously
            _AUX[li] = st.aux[li]
        return out


class _TGraphState:
    """Static buffers for the de-lockstep TREE graph (_RGraphState's tree analog): row index `rows`,
    KV write columns `wcp` [Npad] = start + arange(npad) — real node i at its exact eager slot
    start+i, dummy pad nodes CONTIGUOUSLY after at [start+n, start+npad) (speculative junk past the
    frame's read window, the class the dirty-frontier contract already overwrites-before-read) —
    pre-gathered per-node RoPE rows cos/sin [1,1,Npad,rd] (siblings share a position), the fixed
    read-column index cols [alen], and the PADDED additive mask [1,1,Npad,alen]: rows/cols [:n]
    carry build_tree_mask's values VERBATIM (composed ON DEVICE: prefix zeros + the H2D'd [n,n] tree
    block — never a CPU build of the full [Npad,alen] mask, which costs ms on exactly the contended-
    CPU stages graphs exist for), every real row blocks the pad/unwritten-bucket columns (-inf), and
    each DUMMY row clones row 0's mask row — dummies also clone row 0's input and RoPE position
    (TreeRowGraphRunner.run), so their computation IS a real token's: bounded through all layers, no
    all--inf softmax row (NaN), no off-manifold overflow. set(row, start, n, parents, pos_ids)
    refreshes everything IN PLACE — one captured graph serves every KV row, start, and tree TOPOLOGY
    within a bucket; only the node COUNT is padded, the structure lives in mask/RoPE values."""
    def __init__(self, npad, alen, rd, dv, aux_ids=()):
        self.npad, self.alen = npad, alen
        self.rows = torch.zeros(NKV, dtype=torch.long, device=dv)
        self.wcp = torch.zeros(npad, dtype=torch.long, device=dv)
        self.cos = torch.zeros(1, 1, npad, rd, dtype=torch.bfloat16, device=dv)
        self.sin = torch.zeros(1, 1, npad, rd, dtype=torch.bfloat16, device=dv)
        self.mask = torch.zeros(1, 1, npad, alen, dtype=torch.bfloat16, device=dv)
        self.cols = torch.arange(alen, device=dv)
        self.aux = {li: torch.zeros(npad, H, dtype=torch.bfloat16, device=dv) for li in aux_ids}
        self._nkv = torch.arange(NKV, device=dv)
        self._ar = torch.arange(npad, device=dv)

    def set(self, row, start, n, parents, pos_ids, full_cos, full_sin):
        """row/start/n are host ints, parents/pos_ids HOST lists off the wire — no device sync
        anywhere in the refresh. n = the frame's REAL node count (<= npad); [n:npad) are dummies."""
        from tree_spec import build_tree_mask            # tree-path-only dep, like attn_tree
        dv = self.rows.device
        self.rows.copy_(row * NKV + self._nkv)
        self.wcp.copy_(start + self._ar)                 # real at [start,start+n), dummies contiguous after
        pid = torch.full((self.npad,), int(pos_ids[0]), dtype=torch.long)   # dummies ride node 0's position
        pid[:len(pos_ids)] = torch.as_tensor(pos_ids, dtype=torch.long)
        pid = pid.to(dv, non_blocking=False)
        self.cos.copy_(full_cos[pid][None, None]); self.sin.copy_(full_sin[pid][None, None])
        depths = [int(p) - (start - 1) for p in pos_ids]              # pos_ids == (start-1)+depth (run_block_tree's rule)
        block, _ = build_tree_mask(parents, depths, 0, n)             # start=0 -> JUST the [1,1,n,n] tree block
        self.mask.fill_(float("-inf"))                                # pad rows/cols + bucket tail: blocked
        self.mask[0, 0, :n, :start] = 0.0                             # every real node attends the whole prefix
        self.mask[0, 0, :n, start:start + n].copy_(block[0, 0].to(torch.bfloat16))   # the only H2D: [n,n]
        self.mask[0, 0, n:] = self.mask[0, 0, 0].clone()              # dummy rows = node 0's mask row (clone:
                                                                      # the source aliases the assign target)


class TreeRowGraphRunner:
    """Capture + replay the de-lockstep TREE-verify block at a fixed PADDED node count [1, Npad], one
    graph per context bucket — the eager-tax unlock for per-stream trees (tree frames measured 154ms
    summed stage compute vs 45ms graphed chains; the g lever worked (+15-70% committed/round), the
    eager tax killed it). A frame's N real nodes are padded to Npad in the static input buffer (pad
    rows clone row 0); _TGraphState carries (row, start, tree, n) through per-replay set(). Output
    and EAGLE aux are sliced [:n] on return, so the wire/coordinator contract is shaped exactly like
    eager. Same safety contract as RowGraphRunner: host-side bounds check, free-VRAM pre-check,
    live-row-0 KV save/restore around capture (MAJOR-3), side-stream drain before any eager
    fallback, capture failure -> LOUD permanent-eager. Budget: M25_TREE_GRAPH_MAX (own counter —
    see the M25_TREE_GRAPH module comment).

    NUMERICS CLASS: replay is bit-identical to the same-shape eager-padded forward (run_eager_ref —
    the capture-faithfulness gate), but padded-vs-UNPADDED differs in low bits: the NVFP4 MoE is
    token-count NON-invariant (padding changes per-expert token counts -> grouped-GEMM schedule) and
    the bucketed read changes the softmax reduction length. Same accepted-kernel-numerics class as
    chain graphs vs eager SDPA-flash / fp8 wire; the per-frame gate is ARGMAX agreement (all the
    coordinator consumes — tree_greedy_walk commits identically if argmax agrees), the ring A/B
    judges accept/g. See research/tree_graph_check.py for the full gate hierarchy."""
    def __init__(self, layers, vcfg, npad, dv=dev):
        assert M25_BATCH > 1, "tree row graphs need the launch-time [B,...] KV rows"
        self.layers, self.vcfg, self.npad, self.dv = layers, vcfg, npad, dv
        self.cos, self.sin = get_pe(); self.rd = self.cos.shape[-1]
        self.graphs = {}                                 # bucket alen -> (graph, h_static, state, out_static)
        self.eager = set()                               # buckets whose capture FAILED -> permanently eager
        self.aux_ids = [L.li for L in layers if M25_EAGLE and L.li in EAGLE_AUX_LAYER_IDS]

    def _bucket(self, total):
        for b in DECODE_BUCKETS:
            if b >= total:
                return min(b, M25_KV_MAXLEN)
        return M25_KV_MAXLEN

    def _manual_ok(self, alen):                          # kk/vv gathers + broadcast-GQA fp32 scores must fit
        return 2 * NKV * alen * HD * 2 + NKV * GRP * self.npad * alen * 4 <= 1_400_000_000

    def _layers(self, h):
        st = _GR                                         # the _TGraphState being captured
        for L in self.layers:                            # run_block_tree_row's loop; attn reads _GR statics
            h = h + L.attn_tree_row(L._rms(h, L.in_ln), 0, 0, None, None)   # row/start/pos/mask unused in graph mode
            h = h + L.mlp(L._rms(h, L.post_ln))
            if L.li in st.aux:                           # EAGLE aux: device-side copy into the static
                st.aux[L.li].copy_(h[0])                 # buffer, CAPTURED -> every replay refreshes it
        return h

    def _capture(self, alen):
        from vllm.forward_context import set_forward_context
        global _GR, _TREE_GRAPH_COUNT
        # Free-VRAM pre-check (RowGraphRunner's discipline, tree-shaped): the broadcast-GQA kernel has
        # NO NH-wide repeat_interleave, so the estimate is NKV-based — kk/vv gathers [NKV,alen,HD] bf16
        # + score/softmax transients [NKV,GRP,npad,alen] (bf16 + fp32 + fp32 + bf16 ~ 2.5x fp32) + margin.
        need = 2 * NKV * alen * HD * 2 + 10 * NKV * GRP * self.npad * alen + (1 << 30)
        free = torch.cuda.mem_get_info()[0]
        if free < need:
            raise RuntimeError(f"free VRAM {free / 1e9:.1f}GB < tree-capture estimate {need / 1e9:.1f}GB")
        h = (torch.randn(1, self.npad, H, device=self.dv) * 0.1).to(torch.bfloat16)
        st = _TGraphState(self.npad, alen, self.rd, self.dv, self.aux_ids)
        # Capture-time tree: a full-Npad CHAIN (parents [-1,0,1,..], depths 1..npad) at start=alen-npad
        # so total==alen. Mask/wcp/RoPE are VALUES in static buffers — captured kernels depend on
        # shapes, not values, so any valid topology captures the general case.
        st.set(0, alen - self.npad, self.npad, list(range(-1, self.npad - 1)),
               [alen - self.npad + i for i in range(self.npad)], self.cos, self.sin)
        # Capture writes garbage k/v into LIVE row 0 at [alen-npad, alen) — RowGraphRunner's MAJOR-3:
        # a short stream's capture must not destroy a long stream's committed KV. Save row 0's slots
        # per layer, restore AFTER the device drain (both success and failure paths).
        lo = alen - self.npad
        saved = [(L, L.bkc[0, :, lo:alen].clone(), L.bvc[0, :, lo:alen].clone()) for L in self.layers]
        _GR = st
        try:
            side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.no_grad(), set_forward_context(None, self.vcfg):
                for _ in range(3):
                    self._layers(h)                      # warm-up before capture
            torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.no_grad(), set_forward_context(None, self.vcfg):
                out = self._layers(h)
        finally:
            _GR = None
            torch.cuda.synchronize()                     # drain BEFORE restoring live KV (both paths)
            for L, kb, vb in saved:
                L.bkc[0, :, lo:alen] = kb; L.bvc[0, :, lo:alen] = vb
        self.graphs[alen] = (g, h, st, out)
        _TREE_GRAPH_COUNT += 1                           # tree graphs' own budget (M25_TREE_GRAPH_MAX)

    def run(self, row, start, x, parents, pos_ids):
        """One padded tree-verify block: x [1,N,H] off the wire (N <= npad, enforced by the router),
        parents/pos_ids host lists. Returns the [:N] SLICE of the static output buffer — consumed
        (sent/digested) before the next run, like every runner. The host-side bound covers the FULL
        pad span [start, start+npad) — a replay's captured index_put_ reads st.wcp, and dummy
        columns past MAXLEN would be an OOB device assert that kills the stage — but a frame that
        only overflows because of PADDING (start+n still fits) degrades to eager, never errors: the
        graph must not shrink the servable context."""
        global _GRAPH_SKIPPED
        n = x.shape[1]
        alen = self._bucket(start + n)
        if start + self.npad > M25_KV_MAXLEN or alen in self.eager or not self._manual_ok(alen):
            _GRAPH_SKIPPED += 1
            return run_block_tree_row(self.layers, row, start, x, self.vcfg, parents, pos_ids)
        if alen not in self.graphs:
            try:
                self._capture(alen)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                torch.cuda.synchronize()                 # drain in-flight side-stream garbage FIRST
                self.eager.add(alen); _GRAPH_SKIPPED += 1
                print(f"[graph] tree capture failed (npad={self.npad}, alen={alen}): {type(e).__name__}: {e} "
                      f"-> bucket marked permanently eager", flush=True)
                return run_block_tree_row(self.layers, row, start, x, self.vcfg, parents, pos_ids)
        g, h, st, out = self.graphs[alen]
        st.set(row, start, n, parents, pos_ids, self.cos, self.sin)
        h[:, :n].copy_(x)
        if n < self.npad:                                # dummy rows clone node 0 (bounded real-token values;
            h[:, n:] = h[:, :1]                          # fully overwritten every replay — no stale leakage)
        g.replay(); torch.cuda.synchronize()
        for li in self.aux_ids:                          # [:n] slice of the static [npad,H] aux — solo-shaped,
            _AUX[li] = st.aux[li][:n]                    # consumed synchronously by _merge_aux like every runner
        return out[:, :n]

    def run_eager_ref(self, row, start, x, parents, pos_ids):
        """Eager-execute the PADDED forward through a fresh _TGraphState, no graph — the bit-identity
        oracle for the capture-faithfulness gate (research/tree_graph_check.py): replay output must
        equal this EXACTLY (same kernels, same shapes, same staging). Not on the serve path."""
        from vllm.forward_context import set_forward_context
        global _GR
        n = x.shape[1]
        alen = self._bucket(start + n)
        st = _TGraphState(self.npad, alen, self.rd, self.dv, self.aux_ids)
        st.set(row, start, n, parents, pos_ids, self.cos, self.sin)
        h = torch.empty(1, self.npad, H, dtype=torch.bfloat16, device=self.dv)
        h[:, :n].copy_(x)
        if n < self.npad:
            h[:, n:] = h[:, :1]
        _GR = st
        try:
            with torch.no_grad(), set_forward_context(None, self.vcfg):
                out = self._layers(h)
        finally:
            _GR = None
        for li in self.aux_ids:
            _AUX[li] = st.aux[li][:n]
        return out[:, :n]


class _BGraphState:
    """_GraphState's batched-decode analog: static per-block buffers that carry the varying PER-STREAM
    starts INTO a captured run_block_decode_b graph — abs positions cp [B,s] (drives the RoPE gather,
    the KV scatter index and the per-stream mask), pre-gathered RoPE rows cos/sin [B,1,s,rd], and the
    per-stream additive causal mask [B,1,s,alen] (isolates each stream + zeros its unwritten bucket
    tail, exactly the eager math). set() refreshes them IN PLACE before every replay. `aux_ids`
    (M25_EAGLE) adds one static [B,s,H] OUTPUT buffer per aux layer, copy_()d inside the capture so
    every replay publishes fresh per-stream aux — the solo aux-in-graph design, batched."""
    def __init__(self, B, s, alen, rd, dv, aux_ids=()):
        self.B, self.s, self.alen = B, s, alen
        self.cp = torch.zeros(B, s, dtype=torch.long, device=dv)
        self.cos = torch.zeros(B, 1, s, rd, dtype=torch.bfloat16, device=dv)
        self.sin = torch.zeros(B, 1, s, rd, dtype=torch.bfloat16, device=dv)
        self.mask = torch.zeros(B, 1, s, alen, dtype=torch.bfloat16, device=dv)
        self.aux = {li: torch.zeros(B, s, H, dtype=torch.bfloat16, device=dv) for li in aux_ids}
        self._ar = torch.arange(s, device=dv)
        self._cols = torch.arange(alen, device=dv).view(1, 1, alen)

    def set(self, starts, full_cos, full_sin):
        """starts: per-stream absolute start positions (host list/sequence — never a device sync)."""
        cp = torch.as_tensor(starts, dtype=torch.long, device=self.cp.device).view(self.B, 1) \
            + self._ar.view(1, self.s)
        self.cp.copy_(cp)
        self.cos.copy_(full_cos[cp].unsqueeze(1)); self.sin.copy_(full_sin[cp].unsqueeze(1))
        self.mask.copy_(torch.where(self._cols <= cp[:, :, None], 0.0,
                                    float("-inf")).to(torch.bfloat16)[:, None])


class BatchGraphRunner:
    """Capture + replay a CUDA graph of run_block_decode_b at a FIXED [B, s=K+1] batched-decode shape,
    one graph per context bucket — GraphRunner's continuous-batching analog (batched stages ran EAGER;
    the measured B=4 ring+eager round floor was ~220ms of mostly launch overhead). B is the JOB's row
    count (<= launch M25_BATCH), fixed for the job, so the shape is graph-friendly. The graphed
    attention is the batched manual matmul + static additive mask (the eager small-ctx path's own
    kernels — same class); buckets past the manual-path threshold run eager (per-stream flash SDPA is
    not graphable and the GQA-repeat would OOM). Capture failure -> permanent-eager bucket, never a
    dead stage (solo's OOM-safety + side-stream drain). Counts against the process-wide M25_GRAPH_MAX
    budget like every other graph."""
    def __init__(self, layers, vcfg, B, s, dv=dev):
        assert M25_BATCH > 1 and B <= M25_BATCH, "batched graph needs the launch-time [B,...] KV rows"
        self.layers, self.vcfg, self.B, self.s, self.dv = layers, vcfg, B, s, dv
        self.cos, self.sin = get_pe(); self.rd = self.cos.shape[-1]
        self.graphs = {}                                              # bucket alen -> (graph, h_static, state, out_static)
        self.eager = set()                                            # buckets whose capture FAILED -> permanently eager
        self.aux_ids = [L.li for L in layers if M25_EAGLE and L.li in EAGLE_AUX_LAYER_IDS]

    def _bucket(self, total):
        for b in DECODE_BUCKETS:
            if b >= total:
                return min(b, M25_KV_MAXLEN)
        return M25_KV_MAXLEN

    def _manual_ok(self, alen):                                       # the eager hybrid's threshold: only the
        return self.B * NH * alen * HD * 4 <= 1_400_000_000           # manual-matmul path is graphable

    def _layers(self, h):
        st = _GR                                                      # the _BGraphState being captured
        starts = st.cp[:, 0]                                          # shape-correct; the graphed attn reads _GR, not this
        for L in self.layers:
            h = L.forward_decode_b(h, starts, (self.cos, self.sin))
            if L.li in st.aux:                                        # EAGLE aux: device-side copy into the static
                st.aux[L.li].copy_(h)                                 # buffer, CAPTURED -> every replay refreshes it
        return h

    def _capture(self, alen):
        from vllm.forward_context import set_forward_context
        global _GR, _GRAPH_COUNT
        # Free-VRAM pre-check: a capture that SUCCEEDS with ~zero headroom is WORSE than no capture —
        # the pool pins multi-GB-scale transients forever and the NEXT eager prefill's transients OOM
        # uncaught (dead stage, wedged warm ring). Demand the pool estimate (kk/vv GQA-repeat dominates)
        # + a prefill-sized margin up front; short -> RuntimeError -> run()'s handler marks the bucket
        # permanently eager, LOUDLY. Batched pools are 5-30x solo-sized; brim stages must degrade to
        # eager predictably, never die.
        need = 2 * self.B * NH * alen * HD * 2 + 2 * self.B * NH * self.s * alen * 4 + (1 << 30)
        free = torch.cuda.mem_get_info()[0]
        if free < need:
            raise RuntimeError(f"free VRAM {free / 1e9:.1f}GB < capture estimate {need / 1e9:.1f}GB "
                               f"(pool + prefill margin)")
        h = (torch.randn(self.B, self.s, H, device=self.dv) * 0.1).to(torch.bfloat16)   # static input buffer
        st = _BGraphState(self.B, self.s, alen, self.rd, self.dv, self.aux_ids)
        st.set([alen - self.s] * self.B, self.cos, self.sin)          # capture-time starts (total == alen)
        _GR = st
        try:
            side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.no_grad(), set_forward_context(None, self.vcfg):
                for _ in range(3):
                    self._layers(h)                                   # warm-up before capture
            torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.no_grad(), set_forward_context(None, self.vcfg):
                out = self._layers(h)
        finally:
            _GR = None                                                # capture done; back to eager for prefill etc.
        self.graphs[alen] = (g, h, st, out)
        _GRAPH_COUNT += 1                                             # counts against the process-wide M25_GRAPH_MAX

    def run(self, starts, x):
        """One batched verify block. `starts` is the HOST list off the wire (the bounds check must not
        sync, and set() re-uploads it into the static buffers). Returns the STATIC output buffer — the
        caller consumes it before the next run, like GraphRunner. OOM-SAFE: capture failure drains the
        side stream then marks the bucket permanently eager (see GraphRunner.run for why the drain
        must precede any eager KV write)."""
        global _GRAPH_SKIPPED
        mx = _decode_kv_check(max(starts), self.s)                    # same clean bound as the eager path
        alen = self._bucket(mx)
        if alen in self.eager or not self._manual_ok(alen):
            _GRAPH_SKIPPED += 1
            return run_block_decode_b(self.layers,
                                      torch.as_tensor(starts, dtype=torch.long, device=self.dv), x, self.vcfg)
        if alen not in self.graphs:
            try:
                self._capture(alen)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                torch.cuda.synchronize()                              # drain in-flight side-stream garbage FIRST
                self.eager.add(alen); _GRAPH_SKIPPED += 1
                print(f"[graph] batched capture failed (B={self.B}, s={self.s}, alen={alen}): "
                      f"{type(e).__name__}: {e} -> bucket marked permanently eager", flush=True)
                return run_block_decode_b(self.layers,
                                          torch.as_tensor(starts, dtype=torch.long, device=self.dv), x, self.vcfg)
        g, h, st, out = self.graphs[alen]
        st.set(starts, self.cos, self.sin)                            # varying per-stream starts, IN PLACE
        h.copy_(x)
        g.replay(); torch.cuda.synchronize()
        for li in self.aux_ids:                                       # publish this replay's [B,s,H] aux — ALIASES the
            _AUX[li] = st.aux[li]                                     # static buffer; _merge_aux consumes it synchronously
        return out


def _selftest(layer_ids):
    vcfg = vllm_ctx()
    layers = [Layer(i) for i in layer_ids]
    gb = torch.cuda.memory_allocated() / 1e9
    print(f"loaded layers {layer_ids} ({gb:.2f} GB, {gb/len(layer_ids):.2f} GB/layer)", flush=True)
    torch.manual_seed(0)
    x = torch.randn(1, 8, H, dtype=torch.bfloat16, device=dev) * 0.1
    h = run_block(layers, 0, x, vcfg)
    print(f"prefill(8): out {tuple(h.shape)} finite={torch.isfinite(h).all().item()} mean|h|={h.abs().mean():.4f}", flush=True)
    x2 = torch.randn(1, 1, H, dtype=torch.bfloat16, device=dev) * 0.1
    h2 = run_block(layers, 8, x2, vcfg)
    print(f"decode(@8): out {tuple(h2.shape)} finite={torch.isfinite(h2).all().item()} mean|h|={h2.abs().mean():.4f}", flush=True)
    for L in layers:
        L.reset()
    ok = torch.isfinite(h).all().item() and torch.isfinite(h2).all().item()
    print("VERDICT:", f"m25_stage Layer chain ({len(layer_ids)} real layers) runs GQA+NVFP4-MoE, finite — assembled stage is sound."
          if ok else "NON-FINITE — inspect.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/root/m25")
    ap.add_argument("--layers", type=int, nargs="+", default=[29, 30])
    a = ap.parse_args()
    _selftest(a.layers)
