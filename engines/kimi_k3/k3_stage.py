"""Kimi-K3 pipeline stage: one contiguous layer block, driven through shard's stage contract.

The K3 analogue of m25_stage.py. M1 built a CPU-correct, parity-proven layer range for both
attention types with the AttnRes boundary threaded through the contract; M2 puts the GPU kernels
G0 proved on sm_120 behind it, each under its own knob, with the CPU path still the default on a
box with no CUDA (see "the M2 knobs" below).

WHAT MAKES K3 DIFFERENT FROM EVERY STAGE WE HAVE SHIPPED
A plain transformer stage is a function of one tensor: h_out = block(h_in). K3's decoder is not.
Every layer takes AND returns two tensors -- a running `prefix_sum` and a `block_residual` stack of
(num_tokens, num_blocks, hidden) -- and softmaxes over all preceding block snapshots to decide what
its input actually is (kimi_k3_ref/modeling_kimi_linear.py:973 `_forward_attn_residual`, :1075
`_apply_attn_res`). A new snapshot is appended wherever `layer_idx % attn_res_block_size == 0`, i.e.
at layers {0,12,24,36,48,60,72,84} of the 93 -- 8 blocks by the end. So:

    THE INTER-STAGE PAYLOAD IS (hidden, block_residual), NOT hidden.

Drop `block_residual` at a hop and the stage still runs, still produces plausible logits, and is
silently wrong -- which is exactly the red test in tests/test_k3_stage.py. It also costs real wire
bytes: 14 KiB/token becomes 28 KiB after L11 and 126 KiB by L92 (bf16), ~75 KiB/token averaged over
the 92 boundaries. That is a property of the architecture, not of our split.

The math is RENTED, not rewritten (docs/MODEL_RUNTIME.md): this file instantiates Moonshot's own
`KimiDecoderLayer` for a layer range and drives it. It never reimplements KDA, MLA, LatentMoE, the
`situ` activation or AttnRes. phase0/kimi_k3_ref/ is that reference, vendored verbatim with its
provenance; phase0/k3_kda_cpu.py is the CPU stand-in for the Triton-only `fla` package it imports,
so all of this is provable without renting a GPU.

STATE, AND THE ONE THING M1 REFUSES TO DO
Per-stage and never on the wire: the MLA layers' KV, and the KDA layers' recurrent + conv state
(both live in the reference's `KimiDynamicCache`, indexed by ABSOLUTE layer index, so a stage's
range slots in unchanged). `reset()` drops all of it. `start_pos` rewinds MLA by cropping the KV --
but a KDA recurrent state is a fixed-size summary of every token it has seen and CANNOT be cropped,
so a stage holding KDA layers REFUSES a rewind instead of silently answering from a poisoned state.
M1 is therefore greedy sequential decode only. Speculative rollback needs SpecLA-style compact-factor
bookkeeping or a state checkpoint/restore, and is M3 work.

THE M2 KNOBS (all default to M1's behaviour on a box with no CUDA)
  K3_KDA_BACKEND      auto|cpu|fla|flashkda   the delta recurrence         (k3_kda_cpu/k3_kda_flash)
  K3_ATTNRES_BACKEND  auto|ref|fla            the AttnRes collapse         (k3_attnres)
  K3_MOE_BACKEND      auto|none|marlin        the MXFP4 routed experts     (k3_moe_mxfp4)
  K3_STATIC_STATE     0|1                     fixed-address KDA state (implied by the next one)
  K3_CUDA_GRAPH       0|1                     capture the whole layer block, one graph per
                                              (tokens, num_blocks); K3_GRAPH_MAX caps the set
  K3_DIR, K3_DEV, K3_DTYPE                    where/what/which precision, as in M1

WHAT M2 STILL LEAVES OPEN (the M25_* mechanisms deliberately NOT ported)
  M25_BATCH / M25_BATCH_MOE       continuous batching; the KDA ops here reject cu_seqlens outright
  M25_FP8_WIRE                    fp8 activation transport -- and `block_residual` is the bigger
                                  half of a K3 payload, so it is the thing worth packing
  M25_EAGLE / M25_TREE            drafting; K3 ships no MTP head (num_nextn_predict_layers 0), the
                                  candidate is the external DSpark drafter
  graphs over an MLA layer        an MLA stage's KV grows by cat, so it has no fixed addresses to
                                  capture; K3_CUDA_GRAPH refuses such a stage rather than
                                  half-capturing it (69 of 93 layers are KDA and DO graph)

  self-test:  python3 phase0/k3_stage.py --dir /root/k3 --layers 0 4
"""
import argparse, contextlib, json, os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors import safe_open
# node_kv's convention, verbatim: the flat import serves the single-dir box layout (launchers push
# shard/*.py into /root/ next to phase0's files); off it the module lives at shard.weightkeys.
try:
    import weightkeys
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from shard import weightkeys

# Unlike m25_stage, NOTHING here reads the checkpoint at import time. m25's import-time AutoConfig is
# what forced its `_cli_dir` pre-parse (the self-test's --dir arrived too late to matter); resolving
# lazily costs one memoized call and lets `import k3_stage` work on a box with no model on disk.
K3_DIR = os.environ.get("K3_DIR", "/root/k3")
dev = os.environ.get("K3_DEV", "cuda")
# "" -> bf16 on a GPU (what the checkpoint's non-expert tensors already are), fp32 on CPU, where
# bf16 is both slow and too coarse to tell a parity break from rounding.
K3_DTYPE = os.environ.get("K3_DTYPE", "")
# Fixed-address KDA state: the recurrent summary and the three conv windows are preallocated once
# and updated IN PLACE, so `cache_params.recurrent_states[li] = ret` stops being a reallocation and
# a captured graph keeps replaying against the live state. m25's M25_STATIC_KV, minus the length
# cap -- a KDA state is a fixed-size summary of the whole history, so there is no MAXLEN to bound.
K3_STATIC_STATE = os.environ.get("K3_STATIC_STATE", "0") != "0"
# Whole-layer-block CUDA graphs. G0 (sm_120, real layer-46 weights) measured the capture of an
# entire K3 layer at 1.1186 ms against 1.5141 ms eager on a 5090, and the pieces individually at
# 21.2 -> 16.4 us (FlashKDA decode) and 118.6 -> 8.19 us (AttnRes, a 14x collapse -- at T=1 every
# AttnRes variant is launch-bound, so graphs are the whole lever there).
#
# UNLIKE m25, THE KEY IS NOT A CONTEXT BUCKET. A KDA-only stage's forward reads no position at all
# (no RoPE, no mask, and the recurrent state is position-free), so the only things that vary are the
# token count and the DEPTH OF THE ATTNRES STACK -- which grows as blocks are appended at layers
# {0,12,24,...}. One graph per (tokens, num_blocks); for a fixed layer range num_blocks is in fact
# constant, so a decode stage converges on exactly one graph.
K3_CUDA_GRAPH = os.environ.get("K3_CUDA_GRAPH", "0") != "0"
if K3_CUDA_GRAPH:
    K3_STATIC_STATE = True
# Every captured graph pins its own workspace pool, so cap the set process-wide. Past the cap a new
# shape runs EAGER (counted, never a crash), exactly like m25's M25_GRAPH_MAX.
K3_GRAPH_MAX = int(os.environ.get("K3_GRAPH_MAX", "8"))
_GRAPH_COUNT = 0        # graphs captured so far, across every Stage in this process
_GRAPH_SKIPPED = 0      # blocks that ran eager because the cap was hit or a capture failed

_REF = None
_CFG = {}
_WM = {}
_HD = {}


# ── resolving the reference + the checkpoint ─────────────────────────────────────────────────────

# Symbols the reference imports from a transformers module they have since moved out of, as
# (importing module, name, where it lives now). The vendored reference is kept byte-identical to
# Moonshot's, so version skew is absorbed HERE, in one auditable place, rather than by editing code
# we do not own. Written for transformers 4.56 (its own assert), still current at 5.12.
_TF_MOVED = (("transformers.utils.generic", "OutputRecorder", "transformers.utils.output_capturing"),)


def _tf_compat():
    """Back-fill the moved symbols. Loud on a name we cannot find anywhere: a silent miss would
    surface as an ImportError from inside the vendored file, pointing at the wrong culprit."""
    import importlib
    for where, name, moved_to in _TF_MOVED:
        mod = importlib.import_module(where)
        if hasattr(mod, name):
            continue
        setattr(mod, name, getattr(importlib.import_module(moved_to), name))


def ref():
    """Moonshot's reference decoder module, with usable kernels installed around it (memoized).

    ORDER IS LOAD-BEARING, in both directions. `modeling_kimi_linear` resolves `from fla...` at
    module scope, so the KDA backend has to be in place BEFORE the first import and can never be
    swapped after. The AttnRes backend is the opposite: it rebinds a name ON the imported module, so
    it can only be installed after."""
    global _REF
    if _REF is None:
        import k3_attnres, k3_kda_cpu
        k3_kda_cpu.set_static_state(K3_STATIC_STATE)
        k3_kda_cpu.install()
        _tf_compat()
        from kimi_k3_ref import modeling_kimi_linear as M
        k3_attnres.install(M)
        _REF = M
    return _REF


def config(d=None):
    """The TEXT config for a checkpoint dir, as a KimiLinearConfig (memoized per dir).

    K3 ships a multimodal wrapper whose decoder config is nested under `text_config`; a bare
    KimiLinear checkpoint is not nested. Attention is pinned to eager because the reference's own
    KimiLinearModel forces flash_attention_2 in __init__ -- correct for an 8xB300 node, fatal
    everywhere we want to prove correctness."""
    d = d or K3_DIR
    if d not in _CFG:
        ref()
        from kimi_k3_ref.configuration_kimi_k3 import KimiLinearConfig
        cfgj = json.load(open(f"{d}/config.json"))
        cfg = KimiLinearConfig(**cfgj.get("text_config", cfgj))
        cfg._attn_implementation = "eager"
        _CFG[d] = cfg
    return _CFG[d]


def weight_map(d=None):
    """The checkpoint's tensor -> file index (memoized per dir)."""
    d = d or K3_DIR
    if d not in _WM:
        _WM[d] = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
    return _WM[d]


def raw(n, d=None):
    """One tensor by name, off a cached safetensors handle. m25_stage.raw, per-dir."""
    d = d or K3_DIR
    s = weight_map(d)[n]
    key = (d, s)
    if key not in _HD:
        _HD[key] = safe_open(f"{d}/{s}", "pt", device="cpu")
    return _HD[key].get_tensor(n)


def quant_config(d=None):
    """K3's routed-expert quantization block as the checkpoint writes it, or None.

    Stays a pure reader -- validating here would make the seam's shape depend on how much of the
    format this repo happens to support today. `k3_moe_mxfp4.spec()` is where the block is checked
    (format, group_size, num_bits, what is in the ignore list) and `k3_moe_mxfp4.backend()` is where
    it turns into a decision; both are pure dict work, so the format contract is testable with no
    GPU and no vLLM.

    K3's answer: `mxfp4-pack-quantized`, group_size 32, num_bits 4, E8M0 scales, routed experts only
    -- and the checkpoint IS the quantized release, there is no bf16 upstream to fall back to. With
    the backend off, `Stage.load` still fails loudly on a packed expert (M1's rule), because a stage
    that silently skipped them would serve random-init experts behind a valid receipt."""
    d = d or K3_DIR
    cfgj = json.load(open(f"{d}/config.json"))
    return cfgj.get("quantization_config") or cfgj.get("text_config", {}).get("quantization_config")


# ── the stage ────────────────────────────────────────────────────────────────────────────────────

def _causal_mask(q_len, kv_len, start, dtype, device):
    """additive mask: query i (absolute position start+i) attends key j iff j <= start+i.

    Bottom-right aligned, like m25_stage's causal_lower_right -- a decode step's single query sits at
    the END of the kv, not the top-left. finfo.min rather than -inf, pipeline._causal_mask's
    convention, so a fully-masked row cannot turn into nan."""
    rows = torch.arange(q_len, device=device) + start
    cols = torch.arange(kv_len, device=device)
    minv = torch.finfo(dtype).min
    return torch.where(cols[None, :] <= rows[:, None],
                       torch.zeros((), dtype=dtype, device=device),
                       torch.full((), minv, dtype=dtype, device=device))[None, None]


def seed_block_residual(h):
    """The empty AttnRes stack a forward pass starts from: (num_tokens, 0, hidden).

    KimiLinearModel.forward:1188 -- rebuilt per PASS, not carried across decode steps, which is why
    it belongs in the payload rather than in per-stage state."""
    return h.new_zeros(h.shape[0] * h.shape[1], 0, h.shape[2])


class Stage:
    """One K3 layer block [lo:hi), optionally carrying the head's embedding and/or the tail's head.

    Contract, consumed the way m25_pipe consumes m25_stage:
        embed(token_ids)              -> (h, block_residual)          head only
        forward(h, block_residual, start_pos) -> (h, block_residual)  every stage
        logits(h, block_residual)     -> [B, S, vocab]                tail only
        reset()                       -> drop all per-stage state
    Both members of the pair are plain tensors, so shard/transport.py encodes them as-is."""

    def __init__(self, lo, hi, cfg=None, *, head=False, tail=False, device=None, dtype=None):
        self.lo, self.hi = lo, hi
        self.cfg = cfg if cfg is not None else config()
        self.device = device or dev
        self.dtype = dtype or (torch.bfloat16 if str(self.device).startswith("cuda")
                               else torch.float32)
        if K3_DTYPE:
            self.dtype = getattr(torch, K3_DTYPE)
        M = ref()
        # One config object is shared by every module built from it, and KimiLinearModel.__init__
        # OVERWRITES this field with flash_attention_2 (right on an 8xB300, fatal anywhere we can
        # actually check correctness). A Stage constructed after one of those would inherit it, so
        # pin it here rather than trusting whoever handed us the config.
        self.cfg._attn_implementation = "eager"
        import k3_moe_mxfp4
        # The routed experts are decided BEFORE the layers are built, not after: at K3's real shape
        # letting the reference materialize its 896 bf16 experts costs 58.6 GiB per layer, which is
        # an OOM on any card and on most hosts. Under the marlin backend they are constructed on the
        # meta device (free) and dropped immediately -- the MXFP4 tensors are what actually load.
        self.quant = getattr(self.cfg, "quantization_config", None)
        self.moe_backend = k3_moe_mxfp4.backend(self.quant)
        build = (k3_moe_mxfp4.hollow_experts(M) if self.moe_backend == "marlin"
                 else contextlib.nullcontext())
        with build:
            self.layers = torch.nn.ModuleList([M.KimiDecoderLayer(self.cfg, li)
                                               for li in range(lo, hi)])
        self.moe = {}
        if self.moe_backend == "marlin":
            for L in self.layers:
                msb = getattr(L, "block_sparse_moe", None)
                if msb is None:                       # a dense layer (first_k_dense_replace) has .mlp
                    continue
                msb.experts = torch.nn.ModuleList()   # drop the meta placeholders before any .to()
                ex = k3_moe_mxfp4.MarlinRoutedExperts(self.cfg, self.quant, device=self.device)
                msb.moe_infer = ex.moe_infer          # instance attribute shadows the reference's
                self.moe[L.layer_idx] = ex
        self.has_kda = any(L.is_linear_attn for L in self.layers)
        self.has_mla = any(not L.is_linear_attn for L in self.layers)
        self.head, self.tail = head, tail
        H, eps = self.cfg.hidden_size, self.cfg.rms_norm_eps
        if head:
            self.embed_tokens = torch.nn.Embedding(self.cfg.vocab_size, H, self.cfg.pad_token_id)
        if tail:
            self.norm = M.KimiRMSNorm(H, eps=eps)
            self.output_attn_res_norm = M.KimiRMSNorm(H, eps=eps)
            self.output_attn_res_proj = torch.nn.Linear(H, 1, bias=False)
            self.lm_head = torch.nn.Linear(H, self.cfg.vocab_size, bias=False)
        for m in self._owned_modules():
            m.to(device=self.device, dtype=self.dtype).eval()
        self._keep_gate_params_fp32()
        self.cache = None
        self._pos = 0
        self.reset()
        self.graph = None
        if K3_CUDA_GRAPH:
            why = self._graph_refusal()
            if why:
                print(f"[k3] GRAPH REFUSED for stage[{lo}:{hi}): {why} — staying eager", flush=True)
            else:
                self.graph = _StageGraph(self)

    def _owned_modules(self):
        yield self.layers
        for n in ("embed_tokens", "norm", "output_attn_res_norm", "output_attn_res_proj", "lm_head"):
            if hasattr(self, n):
                yield getattr(self, n)

    def _keep_gate_params_fp32(self):
        """Undo the blanket dtype cast for the two KDA parameters the reference declares fp32.

        `A_log` and `dt_bias` are not weights in a GEMM -- they parameterize the decay itself, as
        exp(A_log) * (g + dt_bias) inside a sigmoid, and every backend consumes them in fp32 (the CPU
        one calls .float() on the way in; FlashKDA TORCH_CHECKs kFloat32 and rejects anything else).
        A stage-wide .to(bfloat16) would round them to ~3 decimal digits first and then widen the
        rounded value back, which is a quiet accuracy loss for zero saving: both are tiny (96 and
        96x128 per layer). No-op on the CPU path, where the stage dtype is already fp32."""
        with torch.no_grad():
            for L in self.layers:
                if not L.is_linear_attn:
                    continue
                a = L.self_attn
                a.A_log.data = a.A_log.data.float()
                a.dt_bias.data = a.dt_bias.data.float()

    def _graph_refusal(self):
        """Why this stage cannot be graph-captured, or None. Loud and specific, never silent."""
        if not str(self.device).startswith("cuda"):
            return f"device is {self.device}"
        if self.has_mla:
            mla = [L.layer_idx for L in self.layers if not L.is_linear_attn]
            return (f"layers {mla} are MLA and their KV cache grows by torch.cat, so there are no "
                    f"fixed addresses to capture (static KV for MLA is not in M2 scope)")
        for L in self.layers:
            msb = getattr(L, "block_sparse_moe", None)
            if msb is not None and L.layer_idx not in self.moe:
                return (f"layer {L.layer_idx} runs the reference's own moe_infer, whose "
                        f"tokens_per_expert.cpu() is a host sync that a capture cannot contain — "
                        f"set K3_MOE_BACKEND=marlin")
        if not K3_STATIC_STATE:
            return "K3_STATIC_STATE is off, so the KDA state reallocates on every step"
        return None

    # ---- state ----

    def reset(self):
        """Drop every per-stage tensor a job accumulated: MLA KV, KDA recurrent + conv state, pos.

        Under K3_STATIC_STATE this becomes a LOGICAL reset (m25_stage.reset's shape): the KDA
        buffers are zeroed in place and keep their addresses, because a captured graph replays
        against those exact pointers and a fresh cache between jobs would leave it reading a freed
        allocation. Without it, a real free -- M1's behaviour."""
        if K3_STATIC_STATE and self.cache is not None:
            with torch.no_grad():
                for li in range(self.lo, self.hi):
                    r = self.cache.recurrent_states[li]
                    if r is not None:
                        r.zero_()
                    for c in (self.cache.conv_states[li] or ()):
                        c.zero_()
                    self.cache.key_cache[li] = self.cache.value_cache[li] = None
            self._pos = 0
            return
        self.cache = ref().KimiDynamicCache(config=self.cfg)
        self._pos = 0
        if K3_STATIC_STATE:
            self._alloc_state()

    def _alloc_state(self):
        """Preallocate this stage's KDA state so every step writes the SAME addresses.

        Both halves have to be fixed, not one: a graph that captured a static recurrent state and a
        reallocating conv cache replays half against live data and half against a dead pointer. The
        conv window carries activations so it takes the stage dtype; the recurrent state is fp32,
        which is what both backends produce (and what FlashKDA stores, though it accumulates in
        bf16 internally). Sizes at K3's shape: 6.00 MiB + 0.56 MiB per layer, per request."""
        with torch.no_grad():
            for L in self.layers:
                if not L.is_linear_attn:
                    continue
                a, li = L.self_attn, L.layer_idx
                z = lambda *s, dt: torch.zeros(*s, dtype=dt, device=self.device)   # noqa: E731
                self.cache.recurrent_states[li] = z(1, a.num_heads, a.head_dim, a.head_dim,
                                                    dt=torch.float32)
                qk = a.head_k_dim * a.num_k_heads
                self.cache.conv_states[li] = tuple(
                    z(1, width, a.conv_size, dt=self.dtype)
                    for width in (qk, qk, a.head_dim * a.num_heads))

    def _seek(self, start_pos):
        """Move the stage to `start_pos`, or refuse. See this module's docstring on KDA rollback."""
        if start_pos == self._pos:
            return
        if start_pos > self._pos:
            raise RuntimeError(
                f"k3 stage[{self.lo}:{self.hi}]: start_pos {start_pos} is ahead of the {self._pos} "
                f"tokens this stage has seen — a gap means the skipped tokens were never fed "
                f"through this block's layers (reset() first, or replay from {self._pos})")
        if self.has_kda:
            raise RuntimeError(
                f"k3 stage[{self.lo}:{self.hi}]: cannot rewind {self._pos} -> {start_pos}. A KDA "
                f"layer's recurrent state is a fixed-size summary of every token it consumed and "
                f"has no per-token history to crop, so a rewind would answer from a poisoned state. "
                f"M1 serves greedy sequential decode only; rollback is M3 (state checkpoint/restore "
                f"or SpecLA compact factors). reset() and replay to rewind today.")
        for li in range(self.lo, self.hi):
            if self.cache.key_cache[li] is not None:
                self.cache.key_cache[li] = self.cache.key_cache[li][:, :, :start_pos].contiguous()
                self.cache.value_cache[li] = self.cache.value_cache[li][:, :, :start_pos].contiguous()
        self._pos = start_pos

    # ---- the serve contract ----

    def embed(self, token_ids):
        """Head only: token ids -> (h, empty block_residual). Seeds the AttnRes stack for the pass."""
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        h = self.embed_tokens(ids)
        return h, seed_block_residual(h)

    def forward(self, h, block_residual, start_pos):
        """Run this stage's layers. BOTH tensors are inputs and BOTH are outputs -- see the module
        docstring: `block_residual` is the payload half a plain-transformer stage does not have."""
        if block_residual is None:
            raise RuntimeError(
                f"k3 stage[{self.lo}:{self.hi}]: forward() needs the AttnRes block_residual, not "
                f"just the hidden state — a K3 layer takes and returns BOTH (see this module's "
                f"docstring). Seed it with seed_block_residual(h) at the head, then carry whatever "
                f"the previous stage returned.")
        self._seek(start_pos)
        h = h.to(device=self.device, dtype=self.dtype)
        br = block_residual.to(device=self.device, dtype=self.dtype)
        s = h.shape[1]
        if self.graph is not None:
            out = self.graph.run(h, br)
            if out is not None:                  # None = this shape is not graphed (see _StageGraph)
                self._pos = start_pos + s
                return out
        mask = (_causal_mask(s, start_pos + s, start_pos, self.dtype, self.device)
                if self.has_mla else None)
        with torch.no_grad():
            for L in self.layers:
                h, br = L(h, attention_mask=(None if L.is_linear_attn else mask),
                          past_key_values=self.cache, block_residual=br)
        self._pos = start_pos + s
        return h, br

    def logits(self, h, block_residual):
        """Tail only: collapse the AttnRes stack one last time, final norm, output head.

        KimiLinearModel.forward:1215 -- the output projection is its OWN norm/proj pair, not the
        last layer's, so a tail that forgot `block_residual` cannot even produce a shape error."""
        M = ref()
        H = self.cfg.hidden_size
        h = h.to(device=self.device, dtype=self.dtype)
        br = block_residual.to(device=self.device, dtype=self.dtype)
        shape = h.shape
        with torch.no_grad():
            h = M._apply_attn_res(h.reshape(-1, H), br, self.output_attn_res_proj,
                                  self.output_attn_res_norm).view(shape)
            return self.lm_head(self.norm(h))

    # ---- weights ----

    def load(self, d=None):
        """Load this stage's layer range (and its boundary tensors) out of a checkpoint dir.

        Namespace-tolerant by construction: the layer number comes from the tensor's own name and
        everything around it is the checkpoint's business (shard/weightkeys.py), so K3's
        `language_model.model.layers.N.` loads with no per-model key table. Coverage is enforced by
        load_state_dict(strict=True) -- a range that matched nothing raises rather than serving a
        random-init block behind a valid receipt."""
        d = d or K3_DIR
        wm = weight_map(d)
        ns = weightkeys.namespace(wm)
        for li in range(self.lo, self.hi):
            prefix = f"{ns['layers']}.{li}."
            names = [n for n in wm if weightkeys.layer_of(n) == li and n.startswith(prefix)]
            if not names:
                raise RuntimeError(f"k3 stage[{self.lo}:{self.hi}]: no tensor under {prefix!r} — "
                                   f"the checkpoint's decoder resolved to {ns['layers']!r}")
            ex = self.moe.get(li)
            if ex is None:
                sd = {n[len(prefix):]: raw(n, d) for n in names}
                self.layers[li - self.lo].load_state_dict(self._fixup(sd, prefix), strict=True)
                continue
            # The 4-bit experts do not go through load_state_dict at all: they are streamed into
            # vLLM's packed buffers one expert at a time (14.6 GiB/layer would otherwise sit in host
            # RAM as a dict) and repacked at the end of THIS layer. Everything else -- the router,
            # the latent projections, the norms, the shared experts -- is bf16 and loads normally.
            epfx = "block_sparse_moe.experts."
            sd = {k: raw(n, d) for n, k in ((n, n[len(prefix):]) for n in names)
                  if not k.startswith(epfx)}
            missing, unexpected = self.layers[li - self.lo].load_state_dict(
                self._fixup(sd, prefix), strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"k3 stage[{self.lo}:{self.hi}]: layer {li} bf16 load is not exact — "
                    f"missing {sorted(missing)[:4]} unexpected {sorted(unexpected)[:4]}. Only the "
                    f"routed experts may be absent from the state dict under the marlin backend.")
            ex.load(lambda n: raw(n, d), f"{prefix}block_sparse_moe").arm()
        if self.head:
            self.embed_tokens.load_state_dict({"weight": raw(ns["embed"] + ".weight", d)})
        if self.tail:
            self.norm.load_state_dict({"weight": raw(f"{ns['norm']}.weight", d)})
            self.lm_head.load_state_dict({"weight": raw(ns["lm_head"] + ".weight", d)})
            # The output AttnRes pair hangs off the inner model next to `norm`, and carries no layer
            # number, so weightkeys classifies it at no boundary -- name it explicitly.
            inner = ns["inner"] + "." if ns["inner"] else ""
            self.output_attn_res_norm.load_state_dict(
                {"weight": raw(f"{inner}output_attn_res_norm.weight", d)})
            self.output_attn_res_proj.load_state_dict(
                {"weight": raw(f"{inner}output_attn_res_proj.weight", d)})
        return self

    def _fixup(self, sd, prefix):
        """Reconcile the checkpoint's tensors with what the reference module declares.

        One real discrepancy, and it is silent: K3 ships `A_log` PADDED -- 128 entries for a
        96-head KDA layer -- while the reference declares the parameter at num_heads and the
        kernels only ever index the first num_heads of it. Loading it whole is a shape error;
        loading it padded-but-reshaped would decay the wrong heads. Slice, and only for A_log:
        llama.cpp validated exactly this slice token-for-token against the reference (commit
        506b7e90). Any OTHER shape mismatch is a bug and load_state_dict(strict=True) will say so."""
        A = "self_attn.A_log"
        packed = sorted(k for k in sd if k.rsplit(".", 1)[-1] in
                        ("weight_packed", "weight_scale", "weight_shape", "weight_global_scale"))
        if packed:
            raise NotImplementedError(
                f"k3 stage[{self.lo}:{self.hi}]: {prefix}{packed[0]} is a packed MXFP4 tensor and "
                f"this path materializes bf16 only — set K3_MOE_BACKEND=marlin to run the 4-bit "
                f"routed experts (k3_moe_mxfp4), which needs vLLM and a CUDA device")
        if A in sd:
            want = self.cfg.linear_attn_config["num_heads"]
            if sd[A].shape[0] > want:
                sd[A] = sd[A][:want].contiguous()
        return sd

    def __repr__(self):
        import k3_attnres, k3_kda_cpu
        kinds = "".join("K" if L.is_linear_attn else "M" for L in self.layers)
        be = f"kda={k3_kda_cpu.backend()} attnres={k3_attnres.backend()} moe={self.moe_backend}"
        return (f"<K3Stage [{self.lo}:{self.hi}) {kinds} head={self.head} tail={self.tail} "
                f"{self.dtype} on {self.device} pos={self._pos} {be} "
                f"graph={'on' if self.graph is not None else 'off'}>")


# ── whole-layer-block CUDA graphs ────────────────────────────────────────────────────────────────

class _StageGraph:
    """Capture + replay this stage's entire layer block, one graph per (tokens, num_blocks).

    G0 proved the capture end to end on real layer-46 weights (FlashKDA + fla AttnRes + Marlin
    MXFP4 experts in one graph): 1.1186 ms replayed against 1.5141 ms eager on a 5090, and capture
    succeeded at every batch size on both cards it was run on.

    WHY THE KEY IS (s, nb) AND NOT A CONTEXT BUCKET. m25 keys its graphs on a KV-length bucket
    because attention reads a span that grows with position. A KDA-only stage reads no position at
    all: no RoPE, no mask, and the recurrent state is a fixed-size summary rather than a window. The
    only things that change shape are the token count and the depth of the AttnRes stack, and for a
    fixed layer range the latter is constant -- so a decode stage settles on exactly ONE graph, and
    the dict is there to keep a prefill or a differently-placed stage from silently sharing it.

    THE CAPTURE MUTATES LIVE STATE. Warm-up runs the real layers against the real recurrent buffers
    and advances them, and unlike an MLA KV crop there is no way to undo a recurrence. The whole
    KDA state is cloned before warm-up and written back after the device is drained -- both on the
    success path and on the failure path, since a half-warmed state is a corrupted job, not a slow
    one. (m25's RowGraphRunner MAJOR-3, in the shape a summary state forces.)

    The returned pair ALIASES the graph's static output buffers: consume or copy it before the next
    forward, exactly like m25's GraphRunner.run."""

    def __init__(self, stage):
        self.st = stage
        self.graphs = {}                     # (s, nb) -> (graph, h_static, br_static, out_static)
        self.eager = set()                   # shapes whose capture failed -> permanently eager

    def _state(self):
        """Every KDA buffer a capture can disturb: the recurrent summary and the three conv windows."""
        c, out = self.st.cache, []
        for li in range(self.st.lo, self.st.hi):
            if c.recurrent_states[li] is not None:
                out.append(c.recurrent_states[li])
            out.extend(c.conv_states[li] or ())
        return out

    def _layers(self, h, br):
        for L in self.st.layers:
            h, br = L(h, attention_mask=None, past_key_values=self.st.cache, block_residual=br)
        return h, br

    def _capture(self, s, nb):
        global _GRAPH_COUNT
        st = self.st
        H = st.cfg.hidden_size
        h = torch.zeros(1, s, H, dtype=st.dtype, device=st.device)
        br = torch.zeros(s, nb, H, dtype=st.dtype, device=st.device)
        saved = [(t, t.clone()) for t in self._state()]
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.no_grad():
                for _ in range(3):           # also warms fla's attnres autotuner, which SYNCS —
                    self._layers(h, br)      # doing that inside a capture is fatal
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.no_grad():
                out = self._layers(h, br)
        finally:
            torch.cuda.synchronize()         # drain before touching live buffers (both paths)
            with torch.no_grad():
                for t, backup in saved:
                    t.copy_(backup)
        self.graphs[(s, nb)] = (g, h, br, out)
        _GRAPH_COUNT += 1

    def plan(self, h, br):
        """Route a (h, block_residual) pair: ("replay"|"capture"|"budget"|"eager", key).

        Pure -- no device work, no allocation, no capture -- so the routing rules are provable
        without a GPU (tests/test_k3_stage_m2.py), which is the half of graph plumbing that is
        cheapest to get subtly wrong and most expensive to debug on rented hardware. `key` is None
        only for a shape that is out of scope entirely, where there is nothing to remember."""
        if h.shape[0] != 1 or br.shape[0] != h.shape[1]:
            return "eager", None             # batched, or a payload that is not this stage's tokens
        key = (h.shape[1], br.shape[1])
        if key in self.graphs:
            return "replay", key
        if key in self.eager:
            return "eager", key
        if _GRAPH_COUNT >= K3_GRAPH_MAX:
            return "budget", key
        return "capture", key

    def run(self, h, br):
        """Replay this (s, nb), or return None to say "run it eager" -- never raise, never crash.

        A stage must not die because a graph would not capture: a fallback costs milliseconds, a
        dead stage costs the warm weights behind it."""
        global _GRAPH_SKIPPED
        what, key = self.plan(h, br)
        if what == "eager":
            _GRAPH_SKIPPED += key is not None
            return None
        if what == "budget":
            self.eager.add(key)
            _GRAPH_SKIPPED += 1
            print(f"[k3] graph budget K3_GRAPH_MAX={K3_GRAPH_MAX} spent — shape {key} stays eager",
                  flush=True)
            return None
        if what == "capture":
            try:
                self._capture(*key)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                torch.cuda.synchronize()
                self.eager.add(key)
                _GRAPH_SKIPPED += 1
                print(f"[k3] graph capture failed for (tokens, blocks)={key}: "
                      f"{type(e).__name__}: {e} — shape marked permanently eager", flush=True)
                return None
        g, hs, brs, out = self.graphs[key]
        hs.copy_(h)
        brs.copy_(br)
        g.replay()
        torch.cuda.synchronize()
        return out


def _selftest(lo, hi, d):
    cfg = config(d)
    st = Stage(lo, hi, cfg, head=(lo == 0), tail=(hi == cfg.num_hidden_layers), device="cpu").load(d)
    print(st, flush=True)
    if st.head:
        h, br = st.embed([[1, 2, 3]])
    else:                                                     # a middle stage is fed by its peer
        h = torch.randn(1, 3, cfg.hidden_size, dtype=st.dtype, device=st.device)
        br = seed_block_residual(h)
    h, br = st.forward(h, br, 0)
    print(f"[k3] h {tuple(h.shape)} block_residual {tuple(br.shape)} "
          f"(payload {1 + br.shape[1]}x hidden per token)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=K3_DIR)
    ap.add_argument("--layers", type=int, nargs=2, default=[0, 4], metavar=("LO", "HI"))
    a = ap.parse_args()
    _selftest(a.layers[0], a.layers[1], a.dir)
