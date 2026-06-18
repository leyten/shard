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
def _sendall(sock, b): sock.sendall(struct.pack("!Q", len(b)) + b)
def _recvn(sock, n):
    buf = bytearray()
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c: raise ConnectionError("peer closed")
        buf += c
    return bytes(buf)
def send_msg(sock, start_pos, hidden):
    bio = io.BytesIO(); torch.save((int(start_pos), hidden.cpu()), bio); _sendall(sock, bio.getvalue())
def recv_msg(sock):
    sp, t = torch.load(io.BytesIO(_recvn(sock, struct.unpack("!Q", _recvn(sock, 8))[0])), weights_only=False)
    return sp, t.to(dev)

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
        if not self.dense:
            self.gate = g("mlp.gate.weight")
            self.rmoe = _build_moe(P + "mlp.experts.", E, I, True)
            self.smoe = _build_moe(P + "mlp.shared_experts.", 1, I, False)
        else:
            self.dmoe = _build_moe(P + "mlp.", 1, Idense, False)
        self.kc = None; self.vc = None   # KV cache [b, nheads, pos, dim]

    def reset(self): self.kc = self.vc = None

    def _rms(self, x, w):
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w

    def attn(self, x, start_pos, pe_full):
        b, s = x.shape[:2]
        q = torch.nn.functional.linear(self._rms(torch.nn.functional.linear(x, self.q_a), self.q_a_ln), self.q_b)
        q = q.view(b, s, self.nheads, self.qk_head).transpose(1, 2)
        q_pass, q_rot = torch.split(q, [self.qk_nope, self.qk_rope], -1)
        ckv = torch.nn.functional.linear(x, self.kv_a)
        k_pass_c, k_rot = torch.split(ckv, [self.kv_lora, self.qk_rope], -1)
        k_pass = torch.nn.functional.linear(self._rms(k_pass_c, self.kv_a_ln), self.kv_b)
        k_pass = k_pass.view(b, s, self.nheads, self.qk_nope + self.v_head).transpose(1, 2)
        k_nope, value = torch.split(k_pass, [self.qk_nope, self.v_head], -1)
        k_rot = k_rot.view(b, 1, s, self.qk_rope)
        cos, sin = pe_full[0][start_pos:start_pos+s].unsqueeze(0), pe_full[1][start_pos:start_pos+s].unsqueeze(0)
        q_rot, k_rot = M.apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(b, self.nheads, s, self.qk_rope)
        Q = torch.cat([q_pass, q_rot], -1)                       # [b,h,s,qk_head]
        Knew = torch.cat([k_nope, k_rot], -1)                    # [b,h,s,qk_head]
        if self.kc is None: self.kc, self.vc = Knew, value
        else: self.kc = torch.cat([self.kc, Knew], 2); self.vc = torch.cat([self.vc, value], 2)
        total = self.kc.shape[2]
        attn = torch.matmul(Q, self.kc.transpose(-1, -2)) * self.scaling   # [b,h,s,total]
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

    def forward(self, x, start_pos, pe_full):
        x = x + self.attn(self._rms(x, self.in_ln), start_pos, pe_full)
        x = x + self.mlp(self._rms(x, self.post_ln))
        return x

_rotary = None; _pe_full = None
def _get_pe(maxpos=4096):
    global _rotary, _pe_full
    if _pe_full is None:
        _rotary = M.GlmMoeDsaRotaryEmbedding(cfg).to(dev)
        dummy = torch.zeros(1, 1, H, dtype=torch.bfloat16, device=dev)
        pos = torch.arange(maxpos, device=dev).unsqueeze(0)
        cos, sin = _rotary(dummy, position_ids=pos)
        _pe_full = (cos[0], sin[0])                              # [maxpos, rope]
    return _pe_full

def run_block(layers, start_pos, h, vcfg):
    from vllm.forward_context import set_forward_context
    pe = _get_pe()
    with torch.no_grad(), set_forward_context(None, vcfg):
        for L in layers:
            h = L.forward(h, start_pos, pe)
    return h

def stage(layer_ids, port, nxt=None):
    vcfg = _vllm_ctx()
    layers = [Layer(i) for i in layer_ids]
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"stage layers {layer_ids} loaded ({mem:.1f} GB) — warming...", flush=True)
    with torch.no_grad():
        _ = run_block(layers, 0, torch.randn(1, 4, H, dtype=torch.bfloat16, device=dev) * 0.1, vcfg)
        for L in layers: L.reset()
    torch.cuda.synchronize()
    print(f"stage layers {layer_ids} WARM, listening :{port}" + (f" -> {nxt}" if nxt else " (tail->return)"), flush=True)
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port)); srv.listen(4)
    fwd = None
    while True:
        conn, _ = srv.accept(); conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                sp, h = recv_msg(conn)
                if sp == 0:
                    for L in layers: L.reset()
                h = run_block(layers, sp, h, vcfg)
                if nxt:
                    if fwd is None:
                        host, p = nxt.rsplit(":", 1); fwd = socket.create_connection((host, int(p)))
                        fwd.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    send_msg(fwd, sp, h); _, back = recv_msg(fwd); send_msg(conn, sp, back)
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
        send_msg(s, start_pos, h); _, hb = recv_msg(s)
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
    p = sub.add_parser("coord"); p.add_argument("--stage", required=True)
    p.add_argument("--prompt", default="The capital of France is"); p.add_argument("--max-new", type=int, default=16)
    a = ap.parse_args()
    if a.role == "stage": stage(a.layers, a.port, a.next)
    else: coord(a.stage, a.prompt, a.max_new)
