"""Kimi-K3 pipeline stage: one contiguous layer block, driven through shard's stage contract.

The K3 analogue of m25_stage.py, at MILESTONE 1 scope -- a CPU-correct, parity-proven layer range
for both attention types, with the AttnRes boundary threaded through the contract. No GPU kernels
here; M2 swaps them in (see "what M1 leaves open" below).

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

WHAT M1 LEAVES OPEN (the M25_* mechanisms deliberately NOT ported yet)
  M25_CUDA_GRAPH / M25_STATIC_KV  graph capture + fixed-address KV; needs the GPU kernels first
  M25_BATCH / M25_BATCH_MOE       continuous batching; the KDA ops here reject cu_seqlens outright
  M25_FP8_WIRE                    fp8 activation transport -- and `block_residual` is the bigger
                                  half of a K3 payload, so it is the thing worth packing
  M25_EAGLE / M25_TREE            drafting; K3 ships no MTP head (num_nextn_predict_layers 0), the
                                  candidate is the external DSpark drafter
Knobs this file does read: K3_DIR, K3_DEV, K3_DTYPE, plus K3_KDA_BACKEND (in k3_kda_cpu).

  self-test:  python3 phase0/k3_stage.py --dir /root/k3 --layers 0 4
"""
import argparse, json, os, sys, torch

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


def _tf_compat_post(M):
    """`create_causal_mask` renamed input_embeds -> inputs_embeds and dropped cache_position after
    the reference was written. Only KimiLinearModel.forward calls it -- a Stage builds its own mask
    -- so this exists for the whole-model reference the parity test measures against.

    Rebound on the REFERENCE MODULE's own binding, never on transformers.masking_utils: the vendored
    file did `from ... import create_causal_mask`, so shadowing its name reaches exactly it, while
    patching the transformers global would change masking for every other model in the process."""
    import inspect
    from transformers.masking_utils import create_causal_mask
    params = inspect.signature(create_causal_mask).parameters
    if "input_embeds" in params:
        return
    def _shim(**kw):
        kw["inputs_embeds"] = kw.pop("input_embeds", None)
        return create_causal_mask(**{k: v for k, v in kw.items() if k in params})
    M.create_causal_mask = _shim


def ref():
    """Moonshot's reference decoder module, with a usable KDA backend installed first (memoized).

    The import order matters: `modeling_kimi_linear` resolves `from fla...` at module scope, so the
    backend has to be on sys.modules BEFORE the first import and can never be swapped after."""
    global _REF
    if _REF is None:
        import k3_kda_cpu
        k3_kda_cpu.install()
        _tf_compat()
        from kimi_k3_ref import modeling_kimi_linear as M
        _tf_compat_post(M)
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
    """K3's routed-expert quantization, or None -- the seam a 4-bit MoE backend resolves through.

    M1 materializes bf16 tensors only and `Stage.load` fails loudly on a packed expert, because a
    stage that silently skipped them would serve random-init experts behind a valid receipt.

    TODO(M2): K3's format is `mxfp4-pack-quantized`, group_size 32, num_bits 4 -- the checkpoint IS
    the quantized release, there is no bf16 upstream. G0 (2026-07-28, sm_120) measured the path:
    vLLM 0.26 resolves this to CompressedTensorsW4A4Mxfp4MoEMethod, the two SM100+ backends fail
    is_supported_config on sm_120 and it falls through to MarlinExperts -- a repack, NOT a bf16
    dequant, so the VRAM saving survives and only FP4 tensor-core throughput is lost. Wire that in
    here the way m25_stage.quant_config feeds _build_moe, and keep EMULATION out of the priority
    list (it materializes bf16 every forward)."""
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
        self.layers = torch.nn.ModuleList([M.KimiDecoderLayer(self.cfg, li)
                                           for li in range(lo, hi)])
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
        self.cache = None
        self._pos = 0
        self.reset()

    def _owned_modules(self):
        yield self.layers
        for n in ("embed_tokens", "norm", "output_attn_res_norm", "output_attn_res_proj", "lm_head"):
            if hasattr(self, n):
                yield getattr(self, n)

    # ---- state ----

    def reset(self):
        """Drop every per-stage tensor a job accumulated: MLA KV, KDA recurrent + conv state, pos.

        Unlike m25_stage.reset this is a real free, not a logical one -- there are no fixed-address
        buffers to overwrite until M2 brings static KV."""
        self.cache = ref().KimiDynamicCache(config=self.cfg)
        self._pos = 0

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
            sd = {n[len(prefix):]: raw(n, d) for n in wm
                  if weightkeys.layer_of(n) == li and n.startswith(prefix)}
            if not sd:
                raise RuntimeError(f"k3 stage[{self.lo}:{self.hi}]: no tensor under {prefix!r} — "
                                   f"the checkpoint's decoder resolved to {ns['layers']!r}")
            self.layers[li - self.lo].load_state_dict(self._fixup(sd, prefix), strict=True)
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
                f"k3 stage[{self.lo}:{self.hi}]: {prefix}{packed[0]} is a packed MXFP4 tensor and M1 "
                f"materializes bf16 only — the 4-bit MoE backend is M2 (see quant_config()'s TODO)")
        if A in sd:
            want = self.cfg.linear_attn_config["num_heads"]
            if sd[A].shape[0] > want:
                sd[A] = sd[A][:want].contiguous()
        return sd

    def __repr__(self):
        kinds = "".join("K" if L.is_linear_attn else "M" for L in self.layers)
        return (f"<K3Stage [{self.lo}:{self.hi}) {kinds} head={self.head} tail={self.tail} "
                f"{self.dtype} on {self.device} pos={self._pos}>")


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
