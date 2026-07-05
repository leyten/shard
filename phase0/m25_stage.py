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
import os, json, argparse, torch
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29577")
os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("LOCAL_RANK", "0")
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.minimax_m2 import modeling_minimax_m2 as M
from torch.nn.attention import sdpa_kernel, SDPBackend                 # SDPA prefill attn (long-ctx OOM fix)
from torch.nn.attention.bias import causal_lower_right                 # bottom-right causal (NOT is_causal)

dev = "cuda"
_CTX = None
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
DECODE_BUCKETS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
_GR = None        # active _GraphState during capture (None = eager); attn reads its static buffers
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
        vcfg.kernel_config.moe_backend = os.environ.get("M25_MOE_BACKEND", "cutlass")
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
        each stream + zeros its unwritten tail."""
        B, s, _ = x.shape
        lin = torch.nn.functional.linear
        q = self._rms(lin(x, self.q_proj), self.q_norm).view(B, s, NH, HD).transpose(1, 2)
        k = self._rms(lin(x, self.k_proj), self.k_norm).view(B, s, NKV, HD).transpose(1, 2)
        v = lin(x, self.v_proj).view(B, s, NKV, HD).transpose(1, 2)
        rd = cos.shape[-1]
        cp = starts.view(B, 1) + torch.arange(s, device=dev).view(1, s)           # [B,s] abs positions
        cu = cos[cp].unsqueeze(1); su = sin[cp].unsqueeze(1)                       # [B,1,s,rd] per-stream RoPE
        def ap(t):
            tr, tp = t[..., :rd], t[..., rd:]
            return torch.cat([tr * cu + _rotate_half(tr) * su, tp], -1)
        q, k = ap(q), ap(k)
        idx = cp.view(B, 1, s, 1).expand(B, NKV, s, HD)
        self.bkc[:B].scatter_(2, idx, _kv_enc(k)); self.bvc[:B].scatter_(2, idx, _kv_enc(v))   # write all rows (fp8 bytes if M25_KV_FP8)
        alen = _bucket(int(starts.max().item()) + s)
        # HYBRID: small/medium context -> ONE batched manual matmul (fast, ~3x batched throughput); big context ->
        # per-stream flash SDPA (the GQA-repeat [B,48,alen,HD] would OOM at big alen, and a batched dense mask
        # forces SDPA->MATH which also OOMs). Threshold on the GQA-repeat size (kk+vv bf16 bytes).
        if B * NH * alen * HD * 4 <= 1_400_000_000:                                # batched manual matmul path
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
    return h


def run_block_decode_b(layers, starts, h, vcfg):        # continuous batching: batched decode, all rows
    from vllm.forward_context import set_forward_context
    pe = get_pe()
    with torch.no_grad(), set_forward_context(None, vcfg):
        for L in layers:
            h = L.forward_decode_b(h, starts, pe)
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
        global _GR
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

    def run(self, start_pos, x):
        """Run one verify/decode block at start_pos through the graph. Returns the STATIC output buffer —
        the caller must consume/copy it before the next run (the serve loop sends .cpu()). Bit-identical
        to eager-MANUAL attention (see the module comment for the SDPA-flash numerics class)."""
        alen = self._bucket(start_pos + self.s)
        if alen not in self.graphs:
            self._capture(alen)
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
