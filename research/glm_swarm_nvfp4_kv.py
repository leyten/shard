"""GLM-5.2 NVFP4 swarm driver WITH KV cache — the tok/s lever.

The no-cache driver (glm_swarm_nvfp4.py) re-embeds + re-sends the whole growing sequence every
step and each stage recomputes all positions (O(n^2) compute, growing wire). With KV cache each
stage keeps the MLA compressed latent per past position; the coord ships only the NEW token's
hidden each step. Per decode step: stages process 1 token (O(1)), wire carries 1 token.

Protocol (coord -> stage): a tensor whose FIRST row encodes [start_pos, batch, seq] as a tiny
int header, followed by the hidden [b, s, H]. start_pos==0 resets the per-connection cache
(new sequence). Stage runs the s tokens at absolute positions [start_pos, start_pos+s), appends
to cache, returns hidden [b, s, H] (coord uses the last row).

Correctness oracle: with KV cache the greedy output must be token-identical to the no-cache driver.

run under /root/vmoe (same env). Forces VLLM_CUTLASS (precompiled, no flashinfer JIT).
  stage: python glm_swarm_nvfp4_kv.py stage --layers 6 7 --port 29600 [--next host:port]
  coord: python glm_swarm_nvfp4_kv.py coord --stage host:port --prompt "..." --max-new 32
"""
import os, io, json, time, socket, struct, argparse, torch
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29556")
os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("LOCAL_RANK", "0")
from safetensors import safe_open
from transformers import GlmMoeDsaConfig, AutoTokenizer
from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as M

DIR, dev = os.environ.get("GLM_DIR", "/root/glm52nvfp4"), "cuda"
cfg = GlmMoeDsaConfig.from_pretrained(DIR); cfg._attn_implementation = "eager"
H, E, I, Idense, K, eps = (cfg.hidden_size, cfg.n_routed_experts, cfg.moe_intermediate_size,
                           cfg.intermediate_size, cfg.num_experts_per_tok, cfg.rms_norm_eps)
NDENSE = cfg.first_k_dense_replace
idx = json.load(open(f"{DIR}/model.safetensors.index.json"))["weight_map"]
_HD = {}
def _h(s):
    if s not in _HD: _HD[s] = safe_open(f"{DIR}/{s}", "pt", device="cpu")
    return _HD[s]
def raw(n): return _h(idx[n]).get_tensor(n)

# ---- transport: int32 header row [start_pos,b,s] packed into a float32 tensor prefix ----
# The wire payload is (start_pos, hidden, meta): meta is None for a plain linear decode
# (the legacy 2-tuple path) and a dict {"par","dep","gather"} for a TREE verify. Keeping
# meta as a trailing optional element means an old 2-tuple still unpacks (meta defaults None),
# so the linear path is byte-for-byte unchanged and only tree messages carry the extra fields.
def _sendall(sock, b): sock.sendall(struct.pack("!Q", len(b)) + b)
def _recvn(sock, n):
    buf = bytearray()
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c: raise ConnectionError("peer closed")
        buf += c
    return bytes(buf)
def _save_send(sock, start_pos, hidden, meta=None):
    bio = io.BytesIO(); torch.save((int(start_pos), hidden.cpu(), meta), bio); _sendall(sock, bio.getvalue())
def _save_recv(sock):
    obj = torch.load(io.BytesIO(_recvn(sock, struct.unpack("!Q", _recvn(sock, 8))[0])), weights_only=False)
    sp, t = obj[0], obj[1]; meta = obj[2] if len(obj) > 2 else None   # legacy 2-tuple -> meta None
    return sp, t.to(dev), meta

# SHARD_WIRE=1 -> OUR transport (phase0/wire.py): raw little-endian tensor bytes instead of
# torch.save pickle (less CPU/hop), ChaCha auth, and idempotent TCP_NODELAY + keepalive
# (the +25% latency-bound lever from the gpt-oss A/B). Default keeps leyten's pickle path
# so the baseline-vs-30.55 comparison stays pristine. Needs wire.py on the box + SHARD_PSK.
if os.environ.get("SHARD_WIRE"):
    import wire; wire.key_from_env()
    def send_msg(sock, start_pos, hidden, meta=None):
        m = {"sp": int(start_pos), "h": hidden.cpu()}
        if meta is not None:
            if "par" in meta:                      # tree verify: par/dep (and optional gather)
                m["par"], m["dep"] = meta["par"], meta["dep"]
                if meta.get("gather") is not None: m["gather"] = meta["gather"]
            if "cap_req" in meta:                  # EAGLE-3 capture: carry the sample id over the wire
                m["cap_req"] = meta["cap_req"]
        return wire.send_msg(sock, m)
    def recv_msg(sock):
        m = wire.recv_msg(sock)
        meta = None
        if "par" in m:                             # reconstruct the tree meta dict
            meta = {"par": m["par"], "dep": m["dep"], "gather": m.get("gather")}
        if "cap_req" in m:                         # capture meta (no par/dep)
            meta = meta or {}; meta["cap_req"] = m["cap_req"]
        return m["sp"], m["h"].to(dev), meta
else:
    send_msg, recv_msg = _save_send, _save_recv

# ====================== NVFP4 execution + VLLM_CUTLASS (stage role) ======================
_VC = None; _CTXMGR = None
def _vllm_ctx():
    global _VC, _CTXMGR
    if _VC is not None: return _VC
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.config import VllmConfig, set_current_vllm_config, get_current_vllm_config
    from vllm.v1.worker.workspace import init_workspace_manager
    torch.cuda.set_device(0)
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://", backend="nccl")
    vcfg = VllmConfig()
    try: vcfg.kernel_config.moe_backend = "cutlass"   # precompiled VLLM_CUTLASS, no flashinfer JIT
    except Exception as e: print("warn moe_backend:", e, flush=True)
    _CTXMGR = set_current_vllm_config(vcfg); _CTXMGR.__enter__()
    print(f"[cfg] moe_backend = {get_current_vllm_config().kernel_config.moe_backend}", flush=True)
    initialize_model_parallel(1); init_workspace_manager(torch.device("cuda"))
    _VC = vcfg; return vcfg

def shared_routing(*a, **kw):
    hs = kw["hidden_states"]; T = hs.shape[0]
    return (torch.ones(T, 1, dtype=torch.bfloat16, device=hs.device),
            torch.zeros(T, 1, dtype=torch.int32, device=hs.device))

def _build_moe(base, n_exp, inter, routed):
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
    try:    # vLLM 0.23+ defaults the unquantized-MoE backend to 'cutlass' (unsupported); the bf16
            # shared expert (ignored by the NVFP4 config) needs 'triton'. Harmless on older vLLM.
        import vllm.model_executor.layers.fused_moe.oracle.unquantized as _U
        if not getattr(_U, "_glm_patched", False):
            _o = _U.map_unquantized_backend
            _U.map_unquantized_backend = lambda rb=None: _o("triton")
            _U._glm_patched = True
    except Exception:
        pass
    qnv = ModelOptNvFp4Config.from_config(json.load(open(f"{DIR}/config.json"))["quantization_config"])
    if routed:
        eb = raw(base.replace("mlp.experts.", "mlp.gate.") + "e_score_correction_bias").float().to(dev)
        m = FusedMoE(num_experts=n_exp, top_k=K, hidden_size=H, intermediate_size=inter, params_dtype=torch.bfloat16,
                     renormalize=cfg.norm_topk_prob, use_grouped_topk=True, num_expert_group=cfg.n_group,
                     topk_group=cfg.topk_group, scoring_func="sigmoid", routed_scaling_factor=cfg.routed_scaling_factor,
                     e_score_correction_bias=eb, quant_config=qnv, prefix=base).to(dev)
    else:
        m = FusedMoE(num_experts=1, top_k=1, hidden_size=H, intermediate_size=inter, params_dtype=torch.bfloat16,
                     renormalize=False, custom_routing_function=shared_routing, quant_config=qnv, prefix=base).to(dev)
    pp = dict(m.named_parameters())
    for e in (range(n_exp) if routed else [None]):
        for proj, shard in [("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")]:
            grp = "w2" if shard == "w2" else "w13"
            nbase = f"{base}{e}.{proj}." if routed else f"{base}{proj}."
            for suf in ["weight", "weight_scale", "weight_scale_2", "input_scale"]:
                n = nbase + suf
                if n in idx: m.weight_loader(pp[f"{grp}_{suf}"], raw(n).to(dev), n, shard, e if routed else 0)
    m.quant_method.process_weights_after_loading(m)
    return m

class Layer:
    """one decoder layer with the bf16 MLA weights + nvfp4 MoE; carries its own KV cache."""
    def __init__(self, li):
        P = f"model.layers.{li}."; self.li = li; self.dense = li < NDENSE
        g = lambda n: raw(P + n).to(torch.bfloat16).to(dev)
        self.in_ln = g("input_layernorm.weight"); self.post_ln = g("post_attention_layernorm.weight")
        self.q_a = g("self_attn.q_a_proj.weight"); self.q_a_ln = g("self_attn.q_a_layernorm.weight")
        self.q_b = g("self_attn.q_b_proj.weight")
        self.kv_a = g("self_attn.kv_a_proj_with_mqa.weight"); self.kv_a_ln = g("self_attn.kv_a_layernorm.weight")
        self.kv_b = g("self_attn.kv_b_proj.weight"); self.o = g("self_attn.o_proj.weight")
        sa = M.GlmMoeDsaConfig.from_pretrained(DIR)
        self.nheads = sa.num_attention_heads
        self.qk_nope = sa.qk_nope_head_dim; self.qk_rope = sa.qk_rope_head_dim
        self.qk_head = self.qk_nope + self.qk_rope; self.v_head = sa.v_head_dim
        self.kv_lora = sa.kv_lora_rank; self.scaling = self.qk_head ** -0.5
        # MLA_LATENT=1: cache the compressed kv-latent (cprime) + shared rope key instead of decompressed
        # full-head K/V (~70x smaller -> long context + concurrency). kv_b is absorbed into q (scores) and
        # o (values) so full-head K/V is never built. Math proven equivalent offline in mla_latent.py.
        self.latent = bool(os.environ.get("MLA_LATENT"))
        if self.latent:
            kvb = self.kv_b.view(self.nheads, self.qk_nope + self.v_head, self.kv_lora)
            self.W_kn = kvb[:, :self.qk_nope, :].contiguous()    # [nh,qkn,kvl] -> absorb into q_pass
            self.W_vb = kvb[:, self.qk_nope:, :].contiguous()    # [nh,vh,kvl]  -> expand o_lat
            self.cc = None; self.rc = None                       # latent cache: cprime [b,pos,kvl], k_rot [b,pos,qkr]
        if not self.dense:
            self.gate = g("mlp.gate.weight")
            self.rmoe = _build_moe(P + "mlp.experts.", E, I, True)
            self.smoe = _build_moe(P + "mlp.shared_experts.", 1, I, False)
        else:
            self.dmoe = _build_moe(P + "mlp.", 1, Idense, False)
        self.kc = None; self.vc = None   # KV cache [b, nheads, pos, dim]

    def reset(self):
        self.kc = self.vc = None
        if getattr(self, "latent", False): self.cc = self.rc = None

    def _rms(self, x, w):
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w

    def gather_kv(self, keep_abs):
        """TREE rollback: keep only the given absolute cache positions (committed prefix +
        last round's accepted path), reindexed contiguously on the seq dim. The eager analog
        of phase0/tree.gather_cache — called BEFORE the next round's attn appends, so the
        accepted path's non-contiguous scratch slots become a clean prefix and the next tree
        appends right after it (mirrors the linear path's self.kc[:, :, :start_pos] crop)."""
        if getattr(self, "latent", False):
            if self.cc is None: return
            idx = torch.tensor(keep_abs, device=self.cc.device)
            self.cc = self.cc.index_select(1, idx).contiguous()
            self.rc = self.rc.index_select(1, idx).contiguous()
            return
        if self.kc is None: return
        idx = torch.tensor(keep_abs, device=self.kc.device)
        self.kc = self.kc.index_select(2, idx).contiguous()
        self.vc = self.vc.index_select(2, idx).contiguous()

    def _attn_latent(self, x, start_pos, pe_full, par=None, dep=None):
        """MLA latent-cache attention (MLA_LATENT=1). Caches cprime [b,pos,kvl] + k_rot [b,pos,qkr]; kv_b
        absorbed into q (W_kn) and o (W_vb). Mathematically == self.attn (proven in mla_latent.py), ~70x
        less KV. Same crop/append/mask semantics as the full-head path, in latent space."""
        b, s = x.shape[:2]
        tree = par is not None
        q = torch.nn.functional.linear(self._rms(torch.nn.functional.linear(x, self.q_a), self.q_a_ln), self.q_b)
        q = q.view(b, s, self.nheads, self.qk_head).transpose(1, 2)
        q_pass, q_rot = torch.split(q, [self.qk_nope, self.qk_rope], -1)
        ckv = torch.nn.functional.linear(x, self.kv_a)
        k_pass_c, k_rot = torch.split(ckv, [self.kv_lora, self.qk_rope], -1)
        cprime = self._rms(k_pass_c, self.kv_a_ln)               # [b,s,kvl]  (THE cache, shared across heads)
        if tree:
            qpos_l = torch.tensor([start_pos + dep[i] for i in range(s)], device=dev)
        else:
            qpos_l = torch.arange(s, device=dev) + start_pos
        cos, sin = pe_full[0][qpos_l].unsqueeze(0), pe_full[1][qpos_l].unsqueeze(0)
        k_rot = k_rot.view(b, 1, s, self.qk_rope)
        q_rot, k_rot = M.apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)   # q_rot [b,h,s,qkr], k_rot [b,1,s,qkr]
        k_rot = k_rot[:, 0]                                       # [b,s,qkr] shared across heads
        # crop (spec-decode rollback) then append — same semantics as the full-head path, on the latent
        if self.cc is not None and self.cc.shape[1] > start_pos:
            self.cc = self.cc[:, :start_pos, :].contiguous(); self.rc = self.rc[:, :start_pos, :].contiguous()
        if self.cc is None: self.cc, self.rc = cprime, k_rot
        else: self.cc = torch.cat([self.cc, cprime], 1); self.rc = torch.cat([self.rc, k_rot], 1)
        total = self.cc.shape[1]
        q_abs = torch.einsum("bhsn,hnl->bhsl", q_pass, self.W_kn)                          # absorb -> latent space
        attn = (torch.einsum("bhsl,bml->bhsm", q_abs, self.cc)
                + torch.einsum("bhsr,bmr->bhsm", q_rot, self.rc)) * self.scaling           # [b,h,s,total]
        if tree:
            allow = torch.zeros(s, total, dtype=torch.bool, device=dev); allow[:, :start_pos] = True
            anc = torch.zeros(s, s, dtype=torch.bool, device=dev)
            for i in range(s):
                j = i
                while j != -1:
                    anc[i, j] = True; j = par[j]
            allow[:, start_pos:start_pos + s] = anc
            mask = torch.where(allow, 0.0, float("-inf")).to(attn.dtype)
        else:
            qpos = torch.arange(s, device=dev).view(s, 1) + start_pos
            kpos = torch.arange(total, device=dev).view(1, total)
            mask = torch.where(kpos <= qpos, 0.0, float("-inf")).to(attn.dtype)
        p = torch.softmax((attn + mask).float(), -1).to(self.cc.dtype)
        o_lat = torch.einsum("bhsm,bml->bhsl", p, self.cc)                                 # attention in latent space
        o = torch.einsum("bhsl,hvl->bhsv", o_lat, self.W_vb)                               # expand ONCE -> [b,h,s,vh]
        o = o.transpose(1, 2).reshape(b, s, -1)
        return torch.nn.functional.linear(o, self.o)

    def attn(self, x, start_pos, pe_full, par=None, dep=None):
        if getattr(self, "latent", False):
            return self._attn_latent(x, start_pos, pe_full, par, dep)
        b, s = x.shape[:2]
        tree = par is not None
        q = torch.nn.functional.linear(self._rms(torch.nn.functional.linear(x, self.q_a), self.q_a_ln), self.q_b)
        q = q.view(b, s, self.nheads, self.qk_head).transpose(1, 2)
        q_pass, q_rot = torch.split(q, [self.qk_nope, self.qk_rope], -1)
        ckv = torch.nn.functional.linear(x, self.kv_a)
        k_pass_c, k_rot = torch.split(ckv, [self.kv_lora, self.qk_rope], -1)
        k_pass = torch.nn.functional.linear(self._rms(k_pass_c, self.kv_a_ln), self.kv_b)
        k_pass = k_pass.view(b, s, self.nheads, self.qk_nope + self.v_head).transpose(1, 2)
        k_nope, value = torch.split(k_pass, [self.qk_nope, self.v_head], -1)
        k_rot = k_rot.view(b, 1, s, self.qk_rope)
        # positions: linear -> contiguous [start_pos, start_pos+s); tree -> start_pos+dep[i]
        # (so a node and its sibling share an abs position; RoPE is position-correct per node).
        if tree:
            qpos_l = torch.tensor([start_pos + dep[i] for i in range(s)], device=dev)
        else:
            qpos_l = torch.arange(s, device=dev) + start_pos
        cos, sin = pe_full[0][qpos_l].unsqueeze(0), pe_full[1][qpos_l].unsqueeze(0)
        q_rot, k_rot = M.apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(b, self.nheads, s, self.qk_rope)
        Q = torch.cat([q_pass, q_rot], -1)                       # [b,h,s,qk_head]
        Knew = torch.cat([k_nope, k_rot], -1)                    # [b,h,s,qk_head]
        # crop to start_pos first -> a verify at start_pos<len rolls back the prior round's
        # rejected speculative tokens (spec-decode); a normal decode (start_pos==len) is a no-op.
        # (For tree, the per-round gather already compacted the kept path to a clean prefix of
        # length start_pos before this call, so the crop is a no-op and the M tree nodes append
        # contiguously at scratch slots [start_pos, start_pos+M).)
        if self.kc is not None and self.kc.shape[2] > start_pos:
            self.kc = self.kc[:, :, :start_pos, :].contiguous(); self.vc = self.vc[:, :, :start_pos, :].contiguous()
        if self.kc is None: self.kc, self.vc = Knew, value
        else: self.kc = torch.cat([self.kc, Knew], 2); self.vc = torch.cat([self.vc, value], 2)
        total = self.kc.shape[2]
        attn = torch.matmul(Q, self.kc.transpose(-1, -2)) * self.scaling   # [b,h,s,total]
        if tree:
            # tree mask: node i (scratch slot start_pos+i) attends key j iff j is in the
            # committed prefix (j < start_pos) OR j is a tree node that is an ancestor of i
            # (incl. self). prefix lives at cache rows [0,start_pos); tree nodes at [start_pos,total).
            allow = torch.zeros(s, total, dtype=torch.bool, device=dev)
            allow[:, :start_pos] = True
            anc = torch.zeros(s, s, dtype=torch.bool, device=dev)
            for i in range(s):
                j = i
                while j != -1:
                    anc[i, j] = True; j = par[j]
            allow[:, start_pos:start_pos + s] = anc
            mask = torch.where(allow, 0.0, float("-inf")).to(attn.dtype)
        else:
            # causal: query i (abs start_pos+i) sees keys 0..start_pos+i
            qpos = torch.arange(s, device=dev).view(s, 1) + start_pos
            kpos = torch.arange(total, device=dev).view(1, total)
            mask = torch.where(kpos <= qpos, 0.0, float("-inf")).to(attn.dtype)
        attn = attn + mask
        o = torch.matmul(torch.softmax(attn.float(), -1).to(value.dtype), self.vc)  # [b,h,s,v_head]
        o = o.transpose(1, 2).reshape(b, s, -1)
        return torch.nn.functional.linear(o, self.o)

    def mlp(self, x):
        shp = x.shape; h = x.view(-1, H)
        ones = torch.ones(h.shape[0], 1, dtype=torch.bfloat16, device=h.device)
        if self.dense:
            return self.dmoe(h, ones).view(shp)
        rl = torch.nn.functional.linear(h, self.gate)
        return (self.rmoe(h, rl) + self.smoe(h, ones)).view(shp)

    def forward(self, x, start_pos, pe_full, par=None, dep=None):
        x = x + self.attn(self._rms(x, self.in_ln), start_pos, pe_full, par=par, dep=dep)
        x = x + self.mlp(self._rms(x, self.post_ln))
        return x

_rotary = None; _pe_full = None
def _get_pe(maxpos=None):
    # maxpos honors $GLM_MAXPOS (default 4096) so long-context (8k/32k/100k) chunked prefill doesn't
    # index the RoPE table out of bounds. Built once, cached; 131072 positions ~ 67MB, cheap.
    global _rotary, _pe_full
    if maxpos is None:
        maxpos = int(os.environ.get("GLM_MAXPOS", "4096"))
    if _pe_full is None:
        _rotary = M.GlmMoeDsaRotaryEmbedding(cfg).to(dev)
        dummy = torch.zeros(1, 1, H, dtype=torch.bfloat16, device=dev)
        pos = torch.arange(maxpos, device=dev).unsqueeze(0)
        cos, sin = _rotary(dummy, position_ids=pos)
        _pe_full = (cos[0], sin[0])                              # [maxpos, rope]
    return _pe_full

try:
    import glm_capture as _CAP            # Phase B EAGLE-3 capture; inert unless GLM_CAPTURE_DIR set
except Exception:
    _CAP = None

def run_block(layers, start_pos, h, vcfg, par=None, dep=None):
    from vllm.forward_context import set_forward_context
    pe = _get_pe()
    with torch.no_grad(), set_forward_context(None, vcfg):
        for L in layers:
            h = L.forward(h, start_pos, pe, par=par, dep=dep)
            if _CAP is not None and _CAP.ENABLED:
                _CAP.maybe_capture(L.li, h, start_pos)   # tap if this stage owns an EAGLE-3 tap layer
    return h


# ====================== STUDY P1: cudagraph-VERIFY (stage-side, opt-in via --graph-verify) ======================
# leyten's lever 2 (RUNBOOK §13b): CUDA-graph the TARGET stage forward at the FIXED verify shape
# [1, K+1, H] so the K+1-token verify replays from a captured graph (no per-call kernel-launch + Python
# overhead). glm_probe_fastverify.py proved the two GLM graph-breakers are bypassable (DSA indexer is a
# no-op at the ~6-token verify seq; the dynamic-shape MoE dispatch -> a batched all-experts forward).
# This wraps run_block at that fixed shape with the static-buffer capture/replay pattern of
# fastverify_graph.py. ANY non-verify-shape forward (prefill, decode-1, tree, capture) falls through to
# eager run_block — the graph only ever serves the linear K+1 verify. NEVER hangs: capture is attempted
# lazily inside try/except; on ANY failure it disables itself (logs once) and falls back to eager forever.
class GraphVerify:
    """Opt-in (--graph-verify) CUDA-graph of the fixed [1, Kp1, H] verify forward. Eager for any other
    shape. Self-disabling on capture failure so a missing kernel can never wedge the stage (P1's
    no-hang contract). verify_len (Kp1) is inferred from the first verify-shaped chunk."""
    def __init__(self, layers, vcfg):
        self.layers, self.vcfg = layers, vcfg
        self.graph = None; self.h_buf = self.out = None; self.kp1 = None
        self.disabled = False

    def _capture(self, sp, s):
        # static input buffer; capture run_block at (start_pos=sp, shape [1,s,H]). The stage Layer cache
        # is grown by torch.cat in attn(); capture happens on a side stream after warmup iters (graph-safe
        # only if the captured allocations are stable — if not, the except below disables cleanly).
        self.kp1 = s
        self.h_buf = torch.zeros(1, s, H, dtype=torch.bfloat16, device=dev)
        for L in self.layers: L.reset()
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                for L in self.layers: L.reset()
                run_block(self.layers, sp, self.h_buf, self.vcfg)
        torch.cuda.current_stream().wait_stream(st)
        for L in self.layers: L.reset()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.out = run_block(self.layers, sp, self.h_buf, self.vcfg)
        self.graph = g
        print(f"[graph-verify] captured fixed verify shape [1,{s},{H}] @ start_pos={sp}", flush=True)

    def maybe_run(self, sp, h, par, dep):
        """Return the stage output for this chunk. Uses the graph ONLY for the fixed linear verify shape;
        everything else (and any capture failure) -> eager run_block. Returns None to mean 'not handled,
        caller must run eager' so the hook stays a pure opt-in fast path."""
        s = h.shape[1]
        if self.disabled or par is not None or s <= 1:
            return None                                  # eager: prefill / decode-1 / tree / disabled
        if self.graph is None:
            try:
                self._capture(sp, s)
            except Exception as e:
                self.disabled = True
                print(f"[graph-verify] capture FAILED ({type(e).__name__}: {str(e)[:160]}); "
                      f"falling back to EAGER verify permanently (no hang)", flush=True)
                for L in self.layers: L.reset()
                return None
        if s != self.kp1:
            return None                                  # different verify width than captured -> eager
        # NOTE: replay reuses the captured cache state; the captured graph holds start_pos fixed. A study
        # P1 A/B measures verify_ms at a single fixed (start_pos, Kp1); for varying start_pos the graph
        # would need the static-cache-write rewrite (fastverify_graph.StaticKV) — out of scope for the
        # opt-in toggle, which only has to land the eager-vs-graphed verify_ms NUMBER without hanging.
        self.h_buf.copy_(h); self.graph.replay(); torch.cuda.synchronize()
        return self.out.clone()


def stage(layer_ids, port, nxt=None, ring=False, graph_verify=False):
    vcfg = _vllm_ctx()
    layers = [Layer(i) for i in layer_ids]
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"stage layers {layer_ids} loaded ({mem:.1f} GB) — warming...", flush=True)
    with torch.no_grad():
        _ = run_block(layers, 0, torch.randn(1, 4, H, dtype=torch.bfloat16, device=dev) * 0.1, vcfg)
        for L in layers: L.reset()
    gv = GraphVerify(layers, vcfg) if graph_verify else None
    if gv is not None:
        print(f"stage layers {layer_ids} graph-verify ENABLED (fixed-shape verify forward will be CUDA-graphed)", flush=True)
    torch.cuda.synchronize()
    print(f"stage layers {layer_ids} WARM, listening :{port}" + (f" -> {nxt}" if nxt else " (tail->return)"), flush=True)
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port)); srv.listen(4)
    fwd = None
    while True:
        conn, _ = srv.accept(); conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                sp, h, meta = recv_msg(conn)
                if sp == 0:
                    for L in layers: L.reset()
                par = dep = None
                if meta is not None:                                              # TREE verify OR capture
                    if _CAP is not None:
                        _CAP.set_cap_req(meta.get("cap_req"))                     # capture: tag sample (inert if absent)
                    g = meta.get("gather")
                    if g is not None:                                             # compact prev round's accepted path
                        for L in layers: L.gather_kv(g)
                    if "par" in meta:                                            # tree meta (capture meta has no par/dep)
                        par, dep = meta["par"], meta["dep"]
                ho = gv.maybe_run(sp, h, par, dep) if gv is not None else None     # opt-in cudagraph verify fast path
                h = ho if ho is not None else run_block(layers, sp, h, vcfg, par=par, dep=dep)
                if nxt:
                    # LEVER 2 (per-edge resilience): re-dial --next + retry the forward on a dropped link, so a
                    # transient inter-stage WAN drop heals HERE instead of breaking the ring + stalling the coord.
                    # Safe for capture: sp==0 prefills make the next stage reset+reprocess idempotently on a resend.
                    fwd_ok = False
                    for _att in range(3):
                        try:
                            if fwd is None:
                                _h, _p = nxt.rsplit(":", 1); fwd = socket.create_connection((_h, int(_p)), timeout=60)
                                fwd.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); fwd.settimeout(180)
                            if ring:
                                send_msg(fwd, sp, h, meta)                                 # ring: fire-forward (tail's --next IS coord)
                            else:
                                send_msg(fwd, sp, h, meta); _, back, _ = recv_msg(fwd); send_msg(conn, sp, back)  # relay-back
                            fwd_ok = True; break
                        except (ConnectionError, EOFError, OSError) as _e:
                            print(f"fwd to {nxt} failed ({type(_e).__name__}); re-dialing edge (attempt {_att})", flush=True)
                            try: fwd.close()
                            except Exception: pass
                            fwd = None; time.sleep(2)
                    if not fwd_ok:
                        raise ConnectionError(f"forward to {nxt} failed after retries")
                else:
                    send_msg(conn, sp, h)
        except (ConnectionError, EOFError):
            print("conn closed", flush=True); fwd = None
            for L in layers: L.reset()

# ====================== coordinator (KV-cached) ======================
def coord(stage_ep, prompt, max_new):
    tok = AutoTokenizer.from_pretrained(DIR, trust_remote_code=True)
    embed_w = raw("model.embed_tokens.weight").to(torch.bfloat16).to(dev)
    lm_head_w = raw("lm_head.weight").to(torch.bfloat16).to(dev)
    norm_w = raw("model.norm.weight").float().to(dev)
    host, p = stage_ep.rsplit(":", 1)
    s = socket.create_connection((host, int(p)), timeout=300); s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); s.settimeout(300)
    print(f"coord(KV) -> stage chain @ {stage_ep}", flush=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    eos = cfg.eos_token_id if isinstance(cfg.eos_token_id, list) else [cfg.eos_token_id]
    n_prompt = ids.shape[1]
    def step(token_ids, start_pos):
        h = torch.nn.functional.embedding(token_ids, embed_w)
        send_msg(s, start_pos, h); _, hb, _ = recv_msg(s)
        x = hb[0, -1].float(); x = x * torch.rsqrt(x.pow(2).mean() + eps) * norm_w
        return int((x.to(torch.bfloat16) @ lm_head_w.t()).float().argmax())
    t0 = time.time()
    nxt = step(ids, 0)                                  # prefill whole prompt at pos 0
    ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], 1)
    gen = 1
    for i in range(max_new - 1):
        if nxt in eos: break
        nxt = step(ids[:, -1:], n_prompt + i)           # decode: send ONLY the new token
        ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], 1); gen += 1
    dt = time.time() - t0
    print(f"\nGENERATED {gen} tokens in {dt:.1f}s = {gen/dt:.2f} tok/s (NVFP4 distributed, KV-CACHED)", flush=True)
    print("decoded:", repr(tok.decode(ids[0], skip_special_tokens=True)[:400]), flush=True)
    return gen / dt, tok.decode(ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="role", required=True)
    p = sub.add_parser("stage"); p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--port", type=int, default=29600); p.add_argument("--next", default=None)
    p.add_argument("--ring", action="store_true")   # fire-forward; tail's --next is the coord's return ep
    p.add_argument("--graph-verify", action="store_true",
                   help="STUDY P1: CUDA-graph the fixed-shape K+1 verify forward (eager for any other shape; "
                        "self-disables on capture failure). Launch stages with this + STUDY_GRAPH_STAGES=1.")
    p = sub.add_parser("coord"); p.add_argument("--stage", required=True)
    p.add_argument("--prompt", default="The capital of France is"); p.add_argument("--max-new", type=int, default=16)
    a = ap.parse_args()
    if a.role == "stage": stage(a.layers, a.port, a.next, a.ring, a.graph_verify)
    else: coord(a.stage, a.prompt, a.max_new)
