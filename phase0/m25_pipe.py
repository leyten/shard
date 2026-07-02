"""MiniMax-M2.5 PIPELINED ring — direct-return stages + the PROVEN coordinate_pipe coordinator.

The lean m25_ring driver was synchronous (one ring traversal at a time), so every token paid the
full ~280ms ring latency. specpipe.coordinate_pipe keeps `depth` verify chunks IN FLIGHT (the GLM
2.9->16.6 lever) and is model-agnostic — it only orchestrates token-ids + argmax over sockets, with
a pluggable n-gram drafter. So we reuse it UNCHANGED and only provide M2.5-native stage serve loops
that speak its wire protocol: reset / verify(token_ids|h, start). The KV is purely start-based — a
fresh chunk at an earlier `start` overwrites stale speculative KV — which is EXACTLY m25_stage's
crop-to-start behaviour, so rollback needs no extra bookkeeping. Direct-return: middle stages
fire-forward, the tail returns straight to the coordinator (serve_tail_direct's 2-connection model).

  stage:  SHARD_TRANSPORT=libp2p M25_DIR=/root/m25 python m25_pipe.py stage --stage 0 --nstages 5 \
              --lo 0 --hi 10 --port 29610 --next 127.0.0.1:29611
  coord:  SHARD_TRANSPORT=libp2p M25_DIR=/root/m25 python m25_pipe.py coord --head 127.0.0.1:29610 \
              --tail 127.0.0.1:29612 --K 6 --depth 4 --max-new 256 --prompt-file p.txt
"""
import os, sys, socket, select, time, argparse, hashlib, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("M25_DIR", "/root/m25")
import json
import m25_stage as S
from m25_tools import render_ids, parse_completion          # tool-calling: chat-template render + output parse
from node_kv import send_msg, recv_msg, EDGE_ERRORS, TransportError   # libp2p codec (SHARD_TRANSPORT=libp2p)
try:                                                    # opt-in confidence-scheduled depth (M25_CONF_SCHED=1)
    from confidence import ConfidenceScheduler
except Exception:
    ConfidenceScheduler = None

dev = "cuda"
NODELAY = (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def _keepalive(s):
    """Bound silent-peer death (no FIN: box power-loss, NAT idle expiry) to ~2min instead of NEVER: the
    churn-resilient tail waits in select(), which cannot see a half-open predecessor, and the old 600s
    recv-timeout teardown is gone (that timeout WAS the idle-wedge). Keepalive makes the kernel probe the
    peer and error the socket, which wakes select and routes through the normal death paths. Matters on
    direct-TCP rings; libp2p conns are loopback-to-sidecar (never half-open)."""
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 20)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
    except (OSError, AttributeError):                    # non-Linux: at least the plain keepalive attempt
        pass

try:                                                    # PROVE: opt-in signed per-stage receipts (trustless verify)
    from receipt import ReceiptSigner, load_or_make_node_key, verify_receipt, verify_coverage
except Exception:
    ReceiptSigner = None
RECEIPTS = bool(os.environ.get("SHARD_RECEIPTS")) and ReceiptSigner is not None
NODE_KEY_PATH = os.environ.get("SHARD_NODE_KEY", "/root/.shard_node_key")

# opt-in fp8 activations on the wire: the MEASURED per-hop bottleneck is moving the bf16 activation, so
# transporting it as fp8 (e4m3) halves bytes/hop (~2x tok/s) at a small, MEASURED precision cost. Lossy but
# still deterministic+verifiable (receipts hash the fp8-engine's activations). bf16 path is unchanged when off.
M25_FP8_WIRE = bool(int(os.environ.get("M25_FP8_WIRE", "0")))
# fp8 EAGLE-aux on the wire: with M25_EAGLE the aux hidden states (3 layers x [K+1,H] bf16 ~ 166KB/hop)
# are ~3x the activation payload itself and ride EVERY hop + the return. They only feed the DRAFTER
# (losslessness untouched — the ring still greedy-verifies), so fp8 is safe by construction; only accept
# could move, and per-tensor-scale e4m3 on O(1) hidden states is well inside the head's tolerance.
# Defaults to the M25_FP8_WIRE setting (one "fp8 on the wire" concept); override with M25_FP8_AUX=0/1.
M25_FP8_AUX = bool(int(os.environ.get("M25_FP8_AUX", "1" if M25_FP8_WIRE else "0")))

def _pack_h(h):
    """fp8 (e4m3) per-tensor quantize a hidden-state activation for transport. A per-tensor scale keeps the
    residual-stream OUTLIER channels inside e4m3's ±448 range. Returns (fp8_cpu_tensor, float_scale)."""
    scale = (h.detach().abs().amax() / 448.0).clamp(min=1e-8)
    return (h / scale).to(torch.float8_e4m3fn).cpu(), float(scale)

def _unpack_h(q, scale):
    return q.to(torch.bfloat16) * scale                 # cpu bf16; caller moves .to(dev)

def _hsend(msg):
    """Outgoing stage message: fp8-pack the activation 'h' when M25_FP8_WIRE (half the bytes/hop), else just
    move it to cpu as before. Only touches 'h'/'h8' — aux/other fields are untouched (aux stays bf16)."""
    h = msg.get("h")
    if torch.is_tensor(h):
        if M25_FP8_WIRE and h.dtype != torch.float8_e4m3fn:
            msg["h"], msg["h8"] = _pack_h(h)
        else:
            msg["h"] = h.cpu()
    return msg

def _hrecv(msg):
    """Dequantize an fp8-packed activation back to bf16 in place (no-op on a bf16 message)."""
    if isinstance(msg, dict) and "h8" in msg:
        msg["h"] = _unpack_h(msg["h"], msg.pop("h8"))
    return msg


def _act_digest(t):
    """Deterministic byte digest of an activation tensor for the receipt hash-chain (fp16 bytes)."""
    return t.detach().to(torch.float16).contiguous().cpu().numpy().tobytes()


def _verify_receipts(receipts, layer_count):
    """Coordinator-side PROVE: every per-stage receipt's signature must verify AND the blocks must
    tile [0:layer_count] with no gap/overlap — so no node is paid without proving its own block and
    the coordinator cannot fabricate one. layer_count is the model's TRUE depth (config), never
    derived from the receipts under test: a ring that omits layers must FAIL coverage, not shrink
    the target to whatever it did attest. Returns True/False (fails closed). Prints a per-stage line."""
    bodies = [{k: v for k, v in rr.items() if k != "stage"} for rr in receipts]
    ok = True
    for rr, body in zip(receipts, bodies):
        try:
            verify_receipt(body)
            print(f"  stage {rr.get('stage')}: layers[{body['layer_start']}:{body['layer_end']}] "
                  f"n={body['n_chunks']} in_root {body['in_root'][:12]} out_root {body['out_root'][:12]} "
                  f"pub {body['pubkey'][:12]} — sig VALID", flush=True)
        except Exception as e:
            ok = False; print(f"  stage {rr.get('stage')}: sig FAILED ({e})", flush=True)
    try:
        verify_coverage(bodies, layer_count)
    except Exception as e:
        ok = False; print(f"  coverage FAILED: {e}", flush=True)
    return ok


def _unpack(resp):
    """EAGLE verify return is {"toks":[ids], "aux":{li:[s,H]}}; the plain path returns just [ids].
    fp8-packed aux entries ([fp8_tensor, scale] pairs, M25_FP8_AUX) are dequantized here once, so every
    downstream consumer (_eagle_seed/_eagle_aux_range/debug) sees plain bf16 [s,H] tensors."""
    if isinstance(resp, dict):
        aux = resp.get("aux")
        if aux:
            for k, v in aux.items():
                if isinstance(v, list):                 # [fp8_cpu_tensor, float_scale] from _merge_aux
                    aux[k] = _unpack_h(v[0], v[1])
        return resp.get("toks"), aux
    return resp, None


def _eagle_seed(aux, pos):
    """Stack the 3 aux hidden states at chunk position `pos` -> [3,H] for EagleDrafter.set_hidden()."""
    import torch as _t
    return _t.stack([aux[str(li)][pos] for li in S.EAGLE_AUX_LAYER_IDS], 0)


def _eagle_aux_range(aux, lo, hi):
    """Stack the 3 aux hidden states for chunk positions [lo,hi) -> [hi-lo,3,H] for EagleDrafter.extend()
    (one slice+stack per layer — not a per-position Python loop, which cost seconds on long prefills)."""
    import torch as _t
    return _t.stack([aux[str(li)][lo:hi] for li in S.EAGLE_AUX_LAYER_IDS], 1)


def _eagle_aux_nodes(aux, node_indices):
    """Gather the 3 aux hidden states at arbitrary FLAT verify-node indices -> [len,3,H]. The tree walk
    indexes the verify's nodes (anchor + accepted path), not contiguous chunk positions, so EagleDrafter.extend()
    needs the predicting-aux gathered by node index, not by range."""
    import torch as _t
    return _t.stack([_eagle_seed(aux, i) for i in node_indices], 0)


def _build_tree_msg(trunk, tree, vbase):
    """Tree-verify wire payload: a causal TRUNK (the last committed path, re-fed at absolute positions vbase+i)
    followed by the M draft-tree nodes off its last slot (the anchor). Returns (token_ids, parents, pos_ids) for
    {op:verify,tree:True,start:vbase}: the ring writes KV cropped-to-start=vbase, and build_tree_mask makes every
    node attend the committed prefix [0:vbase] + the trunk + its root->node ancestors (siblings never see each
    other). parents index the flat node set (-1 = attend committed prefix only); siblings share a depth-RoPE pos."""
    L = len(trunk); M = len(tree["tokens"]); token_ids = list(trunk) + list(tree["tokens"])
    parents = [i - 1 for i in range(L)]; pos_ids = [vbase + i for i in range(L)]      # trunk = causal chain
    for j in range(M):
        pj = tree["parents"][j]
        parents.append(L - 1 if pj == -1 else L + pj)                                 # anchor(-1)->trunk's last node
        pos_ids.append(vbase + (L - 1) + tree["depths"][j])                           # depth-based RoPE pos
    return token_ids, parents, pos_ids


_EAGLE = None


def _eagle_singleton():
    """Build the EagleDrafter ONCE (load the EAGLE-3 head + M2.5 embed_tokens onto the coordinator GPU)
    and reuse it across jobs — its only per-job state (_aux/_pending) is overwritten by set_hidden()/
    request() each prefill, so back-to-back jobs stay clean. The embed (200064x3072 bf16 ~1.2GB) + the
    0.2B head fit alongside whatever else shares the coordinator GPU. M25_EAGLE_DIR = the head checkpoint."""
    global _EAGLE
    if _EAGLE is None:
        from eagle_draft import EagleDrafter
        eagle_dir = os.environ.get("M25_EAGLE_DIR", "/root/m25-eagle")
        embed = S.raw("model.embed_tokens.weight").to(torch.bfloat16).to(dev)
        # next_hidden = which hidden the autoregressive draft chain carries forward. "prenorm" = the residual
        # stream (correct: the final norm is a readout-only transform for the lm_head); "final" = the
        # post-final-norm vector (collapses the chain to token-repetition — observed on-engine). Tunable A/B.
        nh = os.environ.get("M25_EAGLE_NEXT_HIDDEN", "prenorm")
        _EAGLE = EagleDrafter(eagle_dir, embed, device=dev, next_hidden=nh)
        print(f"[eagle] head loaded from {eagle_dir} + M2.5 embed on {dev}", flush=True)
    return _EAGLE


def make_drafter(ngram_n=3):
    """Coordinator-side drafter factory — ONE place so coord/_validate/sweep, the gateway, and the honest
    benchmark all build the same thing from the same env. Plain NgramDrafter by default; when M25_EAGLE=1,
    a HybridDrafter (n-gram-first on draftable text -> EAGLE-3 on novel/reasoning misses). The EagleDrafter
    is the shared singleton; the n-gram half is fresh per job (clean index)."""
    from ngram_draft import NgramDrafter
    ng = NgramDrafter(ng=ngram_n, min_match=int(os.environ.get("M25_NGRAM_MINMATCH", "1")))
    if not S.M25_EAGLE:
        return ng
    from eagle_draft import HybridDrafter
    return HybridDrafter(ng, _eagle_singleton())


def coordinate_pipe(pipe_sock, tok, messages, K, max_new, timeout, depth, ret_sock, local_draft,
                    tools=None, prefill_chunk=4096, max_ctx=0, prefill_depth=8, on_commit=None,
                    swarm_id="swarm", job_id="job", resume_ids=None, resumable=False, reasoning=True):
    """PIPELINED coordinator copied verbatim from specpipe.coordinate_pipe (n-gram local_draft path,
    greedy, direct-return) — keep `depth` verify chunks in flight so throughput approaches the ring's
    per-chunk THROUGHPUT, not its full latency (the GLM 2.9->16.6 lever). Self-contained: only sockets
    + the drafter + tokenizer. eos handled as int-or-list for M2.5."""
    if S.M25_TREE:                                          # EAGLE tree-verify (M25_TREE): one draft TREE per traversal
        return coordinate_pipe_tree(pipe_sock, tok, messages, K, max_new, timeout, depth, ret_sock, local_draft,
                                    tools=tools, prefill_chunk=prefill_chunk, max_ctx=max_ctx, prefill_depth=prefill_depth,
                                    on_commit=on_commit, swarm_id=swarm_id, job_id=job_id, resume_ids=resume_ids,
                                    resumable=resumable, reasoning=reasoning)
    pipe_sock.settimeout(timeout)
    rx = ret_sock if ret_sock is not None else pipe_sock
    def d_request(ids, k): local_draft.request(ids, k)
    def d_fetch(): return local_draft.fetch()
    # discard a stale pending request WITHOUT computing it — fetch() runs the whole proposal (on the
    # EAGLE path a K-step serial chain) just to throw it away on every divergence + at drain.
    d_cancel = getattr(local_draft, "cancel", d_fetch)
    _eos = tok.eos_token_id
    eos_set = set(_eos) if isinstance(_eos, (list, tuple)) else {_eos}
    prompt_ids = render_ids(tok, messages, tools=tools, reasoning=reasoning)   # chat-template + tools; reasoning=False closes <think> -> direct answer (fast)
    resume_ids = list(resume_ids or [])                     # FT resume: re-prefill prompt+committed onto a healed ring, continue (not restart)
    gen_ids = list(prompt_ids) + resume_ids
    if max_ctx:
        max_new = max(len(resume_ids) + 16, min(max_new, max_ctx - len(gen_ids) - 16))
    out = []; t_draft = t_recv = 0.0; prefill_s = 0.0; receipts = []
    try:
        send_msg(pipe_sock, {"op": "reset", "temp": 0.0, "top_p": 1.0, "top_k": 0, "seed": 0,
                             "swarm_id": swarm_id, "job_id": job_id}); recv_msg(rx)
        t_pf = time.time()
        eagle_on = S.M25_EAGLE and hasattr(local_draft, "extend")
        if eagle_on:
            from eagle_draft import prefill_pair_tokens
            local_draft.reset()                                       # fresh EAGLE context per job (drafter is a shared singleton)
        def _pf_extend(start_i, toks_i, aux_i):
            """Feed ONE prefill chunk's aux into the EAGLE context as it arrives, so the drafter sees the
            WHOLE prompt (the old code kept only the LAST chunk -> the drafter attended ~prefill_chunk
            tokens of context on long prompts) and the extend compute hides in the next chunk's WAN wait."""
            if eagle_on and aux_i is not None:
                local_draft.extend(prefill_pair_tokens(gen_ids, start_i, toks_i),
                                   _eagle_aux_range(aux_i, 0, len(toks_i)), base_pos=start_i)
        if prefill_chunk and len(gen_ids) > prefill_chunk:
            starts = list(range(0, len(gen_ids), prefill_chunk))
            def _send_pf(i): send_msg(pipe_sock, {"op": "verify", "token_ids": gen_ids[i:i + prefill_chunk], "start": i, "prefill": True})
            d = min(max(prefill_depth, 1), len(starts)); sent = 0; toks = None
            while sent < d: _send_pf(starts[sent]); sent += 1
            for j in range(len(starts)):
                toks, aux = _unpack(recv_msg(rx))
                if sent < len(starts): _send_pf(starts[sent]); sent += 1
                _pf_extend(starts[j], toks, aux)              # after the refill send: extend overlaps the ring
            cur = toks[-1]
        else:
            send_msg(pipe_sock, {"op": "verify", "token_ids": gen_ids, "start": 0}); toks, aux = _unpack(recv_msg(rx)); cur = toks[-1]
            _pf_extend(0, toks, aux)
        prefill_s = time.time() - t_pf
        pos = len(gen_ids); out = resume_ids + [cur]        # preserve recovered tokens; cur = next after them
        if on_commit: on_commit(out, 0.0)               # stream: first token from prefill
        inflight = []; discard = 0; send_pos = pos; dprefix = gen_ids + [cur]
        valid = accepted = wasted = 0; t0 = time.time(); done = False
        conf = (ConfidenceScheduler(1, depth, lo=0.3, hi=0.7)               # opt-in DSpark depth throttle (M25_CONF_SCHED)
                if (ConfidenceScheduler and os.environ.get("M25_CONF_SCHED")) else None)  # K fixed (graph-safe); only in-flight depth adapts
        d_request(dprefix, K)
        while not done:
            cur_depth = 1 if S.M25_EAGLE else (conf.value() if conf else depth)   # EAGLE needs the verified hidden -> can't pipeline (v1: depth 1); else full depth
            while len(inflight) < cur_depth and not done:
                td = time.time(); ds = d_fetch(); t_draft += time.time() - td
                send_msg(pipe_sock, {"op": "verify", "token_ids": [dprefix[-1]] + ds, "start": send_pos})
                inflight.append((send_pos, ds)); dprefix = dprefix + ds; send_pos += K
                d_request(dprefix, K)
            tr = time.time(); resp = recv_msg(rx); t_recv += time.time() - tr
            r, aux = _unpack(resp)
            sp, ds = inflight.pop(0)
            if discard > 0: discard -= 1; wasted += 1; continue
            n = 0
            for j in range(K):
                if ds[j] == r[j]: n += 1
                else: break
            valid += 1; accepted += n
            if os.environ.get("M25_EAGLE_DEBUG") and S.M25_EAGLE and valid <= 3:   # diagnostic: is aux arriving + is EAGLE drafting?
                _mt = getattr(local_draft, "matched", None)
                if aux is None:
                    print(f"[eagle-dbg] r{valid}: aux=None (ring returned plain toks -> EAGLE degrades to repeat) matched={_mt} ds={ds[:5]} r={r[:5]} acc={n}", flush=True)
                else:
                    _ok = all(str(li) in aux for li in S.EAGLE_AUX_LAYER_IDS)
                    _sd = _eagle_seed(aux, n) if _ok else None
                    _nm = float(_sd.float().norm()) if _ok else -1.0
                    _eg = getattr(local_draft, "eagle", local_draft)            # decisive probe: does fc(aux) VARY across prompts?
                    _fcs = "n/a"
                    if _ok and hasattr(_eg, "fc"):
                        _fc = torch.nn.functional.linear(_sd.reshape(-1).to(_eg.fc.dtype).to(_eg.fc.device).unsqueeze(0), _eg.fc)
                        _fcs = f"norm={float(_fc.norm()):.2f} v3={[round(x,3) for x in _fc[0,:3].float().tolist()]}"
                    print(f"[eagle-dbg] r{valid}: seednorm={_nm:.1f} fc(aux):{_fcs} matched={_mt} ds={ds[:5]} r={r[:5]} acc={n}", flush=True)
            if conf: conf.observe(n, K)                                     # acceptance EMA (free, from the verify result)
            if n == K:
                out.extend(ds); pos += K; cur = ds[-1]; committed = ds
                # FULL-ACCEPT BONUS: r[K] is the target's greedy token at the position just past the last
                # accepted draft (lossless — the whole draft matched, so the prefix IS the greedy sequence).
                # When nothing is in flight (depth-1: EAGLE/tree) that position is NOT covered by a queued
                # chunk, so committing it now advances the frontier K+1 not K -> ~1/K fewer WAN round-trips on
                # draftable text, and makes toks_per_traversal honest. Under pipelining (depth>1) the next
                # in-flight chunk already re-derives it at no extra RTT, so we leave it (committing would
                # collide with that chunk's start).
                if not inflight and len(r) > K:
                    out.append(r[K]); committed = ds + [r[K]]; cur = r[K]
                    pos += 1; send_pos += 1; dprefix = dprefix + [r[K]]
            else:
                committed = ds[:n] + [r[n]]; out.extend(committed); cur = r[n]; pos += n + 1
                discard = len(inflight); d_cancel(); dprefix = prompt_ids + out; send_pos = pos; d_request(dprefix, K)
            if eagle_on and aux is not None:                 # grow the EAGLE context with the newly committed positions
                local_draft.extend(committed, _eagle_aux_range(aux, 0, len(committed)), base_pos=sp)   # committed[i] predicted by aux[i] (target hidden one pos earlier)
            if on_commit: on_commit(out, time.time() - t0)   # stream: this commit's running output
            if len(out) >= max_new or (cur in eos_set) or (eos_set & set(committed)): done = True
        d_cancel()
        while inflight: recv_msg(rx); inflight.pop(0)
        if RECEIPTS:                                        # PROVE: sweep the ring once for signed per-stage receipts
            send_msg(pipe_sock, {"op": "receipt", "receipts": []}); receipts = recv_msg(rx)
    except EDGE_ERRORS as e:
        if resumable:                                       # a node died: hand committed tokens back so the control plane heals + resumes (not restart)
            committed = out if out else list(resume_ids)
            return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}", "resumable": True,
                    "output_ids": committed, "n_tokens": len(committed),
                    "text": tok.decode(committed, skip_special_tokens=True)}
        raise TransportError(f"pipeline edge failed at token {len(out)} ({type(e).__name__}: {e})") from e
    dt = time.time() - t0
    for ee in eos_set:
        if ee in out: out = out[:out.index(ee)]; break
    # True depth from the model config — never from the receipts themselves (self-referential coverage
    # let a layer-omitting ring pass). Fail CLOSED when receipts were requested but none came back.
    receipts_ok = (_verify_receipts(receipts, S.cfg.num_hidden_layers) if receipts
                   else (False if RECEIPTS else None))
    return {"ok": True, "text": tok.decode(out, skip_special_tokens=True), "n_tokens": len(out), "rounds": valid,
            # HONEST g: committed tokens per verify round = frontier advance (pos) / rounds. NOT the old
            # (accepted+valid)/valid, which counted a bonus token on EVERY full-accept round even when the
            # pipelined path dropped it -> up to +1 inflation, not comparable to the tree arm's exact g.
            "mean_accept": accepted / max(valid, 1), "toks_per_traversal": (pos - len(gen_ids)) / max(valid, 1),
            "tok_s": len(out) / max(dt, 1e-9), "wasted": wasted, "prefill_s": prefill_s, "output_ids": out,
            "prompt_tokens": len(prompt_ids), "resume_tokens": len(resume_ids),
            "receipts": receipts, "receipts_ok": receipts_ok,
            # decode-loop breakdown: draft_s = serial drafter compute, ring_wait_s = blocked on the ring's
            # return channel, decode_s = the whole decode wall. What's NOT wait or draft is coordinator-side
            # commit/extend/serialize overhead — the profile that ranks the next serial-path fix.
            "decode_s": round(dt, 3), "draft_s": round(t_draft, 3), "ring_wait_s": round(t_recv, 3),
            "final_confidence": conf.confidence() if conf else None}


def coordinate_pipe_tree(pipe_sock, tok, messages, K, max_new, timeout, depth, ret_sock, local_draft,
                         tools=None, prefill_chunk=4096, max_ctx=0, prefill_depth=8, on_commit=None,
                         swarm_id="swarm", job_id="job", resume_ids=None, resumable=False, reasoning=True):
    """EAGLE TREE-VERIFY coordinator (M25_TREE=1): one draft TREE per ring traversal instead of one linear
    chain. Each round re-feeds the last committed path as a causal trunk + grows a top-M EAGLE tree off its
    last node (the anchor); the ring verifies the whole thing in ONE forward under an ancestor-only mask
    (run_block_tree) and tree_greedy_walk commits the longest accepted path + 1 correction/bonus. depth=1
    structurally — one tree in flight per traversal — so K/depth are unused; the speculation budget is
    M25_TREE_M nodes (best-first by cumulative draft log-prob, M25_TREE_TOPB children per expansion, depth
    cap M25_TREE_DEPTH — the fleet-confirmed fix for the fixed-2^d waste, 62 nodes for +0.7 g).

    GREEDY / LOSSLESS by construction (the ring greedy-verifies every node; the tree only changes WHICH
    tokens are proposed). CORRECTNESS GATE: with M25_TREE_TOPB=1 the tree degenerates to a single chain and
    the committed output must byte-match chain-EAGLE greedy (same-kernel caveat: attn_tree is the manual
    reference kernel, so gate against M25_SDPA=0 stages or accept rare near-tie argmax flips). Prefill, the
    receipts sweep and the return-dict shape are identical to coordinate_pipe (honest_bench parses it the
    same; receipts stay ON so the tree A/B carries the same attestation cost as the chain)."""
    from tree_spec import tree_greedy_walk
    pipe_sock.settimeout(timeout)
    rx = ret_sock if ret_sock is not None else pipe_sock
    _eos = tok.eos_token_id
    eos_set = set(_eos) if isinstance(_eos, (list, tuple)) else {_eos}
    prompt_ids = render_ids(tok, messages, tools=tools, reasoning=reasoning)
    resume_ids = list(resume_ids or [])
    gen_ids = list(prompt_ids) + resume_ids
    if max_ctx:
        max_new = max(len(resume_ids) + 16, min(max_new, max_ctx - len(gen_ids) - 16))
    out = []; prefill_s = 0.0; t_draft = t_recv = 0.0; receipts = []
    tree_m = int(os.environ.get("M25_TREE_M", "12"))
    tree_topb = int(os.environ.get("M25_TREE_TOPB", "3"))
    tree_depth = int(os.environ.get("M25_TREE_DEPTH", "8"))
    eg = getattr(local_draft, "eagle", local_draft)         # the EagleDrafter (HybridDrafter.eagle, or itself)
    if not hasattr(eg, "propose_tree"):
        raise RuntimeError("M25_TREE=1 needs the EAGLE drafter (set M25_EAGLE=1 + M25_EAGLE_DIR on the coordinator)")
    try:
        send_msg(pipe_sock, {"op": "reset", "temp": 0.0, "top_p": 1.0, "top_k": 0, "seed": 0,
                             "swarm_id": swarm_id, "job_id": job_id}); recv_msg(rx)
        t_pf = time.time()                                  # ---- prefill: IDENTICAL to coordinate_pipe ----
        eg.reset()                                          # fresh EAGLE context per job (drafter is a shared singleton)
        from eagle_draft import prefill_pair_tokens
        def _pf_extend(start_i, toks_i, aux_i):
            """Feed ONE prefill chunk's aux into the EAGLE context as it arrives (whole-prompt drafter
            context — the accept lever the serial-path A/B proved on rag-quote, 13->44%)."""
            if aux_i is not None:
                eg.extend(prefill_pair_tokens(gen_ids, start_i, toks_i),
                          _eagle_aux_range(aux_i, 0, len(toks_i)), base_pos=start_i)
        if prefill_chunk and len(gen_ids) > prefill_chunk:
            starts = list(range(0, len(gen_ids), prefill_chunk))
            def _send_pf(i): send_msg(pipe_sock, {"op": "verify", "token_ids": gen_ids[i:i + prefill_chunk], "start": i, "prefill": True})
            d = min(max(prefill_depth, 1), len(starts)); sent = 0; toks = None
            while sent < d: _send_pf(starts[sent]); sent += 1
            for j in range(len(starts)):
                toks, aux = _unpack(recv_msg(rx))
                if sent < len(starts): _send_pf(starts[sent]); sent += 1
                _pf_extend(starts[j], toks, aux)            # after the refill send: extend overlaps the ring
            cur = toks[-1]
        else:
            send_msg(pipe_sock, {"op": "verify", "token_ids": gen_ids, "start": 0}); toks, aux = _unpack(recv_msg(rx)); cur = toks[-1]
            _pf_extend(0, toks, aux)
        if aux is None:                                     # fail loud, not a mid-job TypeError: stages must run M25_EAGLE
            raise TransportError("tree-verify got no aux from the ring — launch stages with M25_TREE=1/M25_EAGLE=1")
        prefill_s = time.time() - t_pf
        out = [cur]; pending_path = [cur]; vbase = len(gen_ids)      # cur = first gen token at abs pos vbase
        if on_commit: on_commit(out, 0.0)                            # stream: first token from prefill
        rounds = 0; total_committed = 0; t0 = time.time(); done = False
        ng = getattr(local_draft, "ngram", None)            # HybridDrafter's n-gram half (None on a bare EagleDrafter)
        while not done:
            L = len(pending_path)
            td = time.time()
            tree = None
            if ng is not None:                              # HYBRID routing, same split as the chain path: n-gram
                ng.request(prompt_ids + out, K)             # FIRST, K draft tokens (verbatim runs are long — do
                ds = ng.fetch()                             # NOT cap at tree_depth; the fake-ring harness showed
                if ds and getattr(ng, "matched", False):    # tree_depth<K silently halves the n-gram g). Miss -> EAGLE tree.
                    tree = {"tokens": list(ds), "parents": [-1] + list(range(len(ds) - 1)),
                            "depths": list(range(1, len(ds) + 1))}   # a matched n-gram chain IS a 1-wide tree
            if tree is None:
                tree = eg.propose_tree(tree_m, topb=tree_topb, max_depth=tree_depth)
            t_draft += time.time() - td
            token_ids, parents, pos_ids = _build_tree_msg(pending_path, tree, vbase)
            send_msg(pipe_sock, {"op": "verify", "tree": True, "token_ids": token_ids,
                                 "parents": parents, "pos_ids": pos_ids, "start": vbase})
            tr = time.time(); r, aux = _unpack(recv_msg(rx)); t_recv += time.time() - tr
            path_idx, committed = tree_greedy_walk(tree["tokens"], tree["parents"], r[L:], r[L - 1])
            out.extend(committed); vbase += L; pending_path = committed; cur = committed[-1]
            # EAGLE extend: committed[0] predicted by the anchor (flat node L-1); committed[k>0] by the (k-1)-th
            # accepted path node (flat node L+path_idx[k-1]). Slice to len(committed) (== 1+len(path_idx)).
            pred_idx = ([L - 1] + [L + pi for pi in path_idx])[:len(committed)]
            eg.extend(committed, _eagle_aux_nodes(aux, pred_idx), base_pos=vbase - 1)   # base_pos = anchor's abs pos (predicting hidden), per the extend contract & the chain path
            rounds += 1; total_committed += len(committed)
            if on_commit: on_commit(out, time.time() - t0)           # stream: this round's running output
            if len(out) >= max_new or (cur in eos_set) or (eos_set & set(committed)): done = True
        if RECEIPTS:                                        # PROVE: sweep the ring once for signed per-stage receipts
            send_msg(pipe_sock, {"op": "receipt", "receipts": []}); receipts = recv_msg(rx)
    except EDGE_ERRORS as e:
        if resumable:                                       # a node died: hand committed tokens back so the control plane heals + resumes
            committed = out if out else list(resume_ids)
            return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}", "resumable": True,
                    "output_ids": committed, "n_tokens": len(committed),
                    "text": tok.decode(committed, skip_special_tokens=True)}
        raise TransportError(f"tree pipeline edge failed at token {len(out)} ({type(e).__name__}: {e})") from e
    dt = time.time() - t0
    for ee in eos_set:
        if ee in out: out = out[:out.index(ee)]; break
    receipts_ok = (_verify_receipts(receipts, S.cfg.num_hidden_layers) if receipts
                   else (False if RECEIPTS else None))
    accepted = total_committed - rounds                     # accept per round = len(committed)-1 (the +1 is the correction/bonus)
    return {"ok": True, "text": tok.decode(out, skip_special_tokens=True), "n_tokens": len(out), "rounds": rounds,
            "mean_accept": accepted / max(rounds, 1), "toks_per_traversal": total_committed / max(rounds, 1),
            "tok_s": len(out) / max(dt, 1e-9), "wasted": 0, "prefill_s": prefill_s, "output_ids": out,
            "prompt_tokens": len(prompt_ids), "resume_tokens": len(resume_ids),
            "receipts": receipts, "receipts_ok": receipts_ok,
            "decode_s": round(dt, 3), "draft_s": round(t_draft, 3), "ring_wait_s": round(t_recv, 3),
            "final_confidence": None}


def _sdpa_backend_probe(stage):
    """Fail loud at warm-up (not mid-prefill OOM) if no FUSED SDPA backend serves the prefill shape on this
    GPU. A fused backend (flash/cudnn/efficient) does online softmax = O(s) memory; the MATH fallback
    materializes the [1,NH,s,total] score matrix = the very OOM the SDPA fix removes. Reports which engage."""
    avail = []
    qd = torch.randn(1, S.NH, 64, S.HD, dtype=torch.bfloat16, device=dev)
    kd = torch.randn(1, S.NKV, 256, S.HD, dtype=torch.bfloat16, device=dev)
    mask = S.causal_lower_right(64, 256)
    for name, be in [("flash", S.SDPBackend.FLASH_ATTENTION), ("cudnn", S.SDPBackend.CUDNN_ATTENTION),
                     ("efficient", S.SDPBackend.EFFICIENT_ATTENTION)]:
        try:
            with S.sdpa_kernel([be]):
                torch.nn.functional.scaled_dot_product_attention(qd, kd, kd, attn_mask=mask,
                                                                 scale=S.SCALING, enable_gqa=True)
            avail.append(name)
        except Exception:
            pass
    if avail:
        print(f"[s{stage}] SDPA fused backends available on sm_120: {avail}", flush=True)
    else:
        print(f"[s{stage}] WARN SDPA: NO fused backend serves the prefill shape — falls back to MATH "
              f"(materializes scores; long-ctx will OOM). Lower prefill_chunk or set M25_SDPA=0.", flush=True)


def coordinate_pipe_batch(pipe_sock, tok, messages_list, K, max_new, timeout, ret_sock, drafters,
                          depth=4, tools=None, prefill_chunk=4096, max_ctx=0, reasoning=True):
    """CONTINUOUS-BATCHING coordinator: B independent spec-decode streams share ONE ring traversal per
    round, so the WAN round-trip is amortized across all B (aggregate-throughput lever). SYNCHRONOUS
    (one batched verify per round — no per-stream depth pipelining; the batching itself provides the
    win). Each stream's output is byte-identical to a solo coordinate_pipe run (per-stream KV row +
    per-stream causal mask + per-stream MoE on the stage side guarantee it). Prefill is PER-STREAM
    (variable length) into batch-row b; only the fixed-shape K+1 decode is batched. Greedy.

    Protocol: reset_batch -> prefill each stream (op=verify, stream=b) -> per round, op=verify_batch
    with token_ids_b/start_b for the ACTIVE streams; the ring returns B argmax rows."""
    B = len(messages_list)
    rx = ret_sock if ret_sock is not None else pipe_sock
    pipe_sock.settimeout(timeout)
    _eos = tok.eos_token_id
    eos_set = set(_eos) if isinstance(_eos, (list, tuple)) else {_eos}
    prompts = [render_ids(tok, m, tools=tools, reasoning=reasoning) for m in messages_list]
    mx = [max(16, min(max_new, max_ctx - len(p) - 16)) if max_ctx else max_new for p in prompts]
    out = [[] for _ in range(B)]; pos = [0] * B; cur = [0] * B; done = [False] * B
    t_recv = 0.0; t_pf = time.time()
    send_msg(pipe_sock, {"op": "reset_batch", "B": B}); recv_msg(rx)
    for b in range(B):                                   # PER-STREAM prefill into row b (variable length)
        gen = prompts[b]
        if prefill_chunk and len(gen) > prefill_chunk:
            rr = None
            for i in range(0, len(gen), prefill_chunk):
                send_msg(pipe_sock, {"op": "verify", "stream": b, "token_ids": gen[i:i + prefill_chunk], "start": i, "prefill": True})
                rr = recv_msg(rx)
            cur[b] = rr[-1]
        else:
            send_msg(pipe_sock, {"op": "verify", "stream": b, "token_ids": gen, "start": 0, "prefill": True}); cur[b] = recv_msg(rx)[-1]
        pos[b] = len(gen); out[b] = [cur[b]]
        if cur[b] in eos_set or len(out[b]) >= mx[b]: done[b] = True
        drafters[b].request(prompts[b] + [cur[b]], K)
    prefill_s = time.time() - t_pf; t0 = time.time()        # start the DECODE-rate timer after prefill (matches coordinate_pipe; agg_tok_s is steady-state decode, not TTFT-polluted)
    # PIPELINED: keep `depth` batched verify-rounds in flight so the WAN is HIDDEN (the synchronous depth=1 path
    # paid full ring latency L every round -> B/L; this restores depth-pipelining -> aggregate ~ B x single-stream).
    # Each round speculatively advances ALL B streams; on a stream's divergence we drop that stream's stale
    # in-flight chunks (per-row discard) and re-draft. Each stream stays data-isolated (output depends on B, not
    # on batch-mates) and byte-faithful to solo up to the batched-matmul tiling. Mirrors coordinate_pipe per row.
    rounds = 0; wasted = 0
    dprefix = [prompts[b] + [cur[b]] for b in range(B)]     # speculative continuation per stream (prefill already requested)
    spos = list(pos)                                        # send position per stream (advances K per drafted chunk)
    discard = [0] * B                                       # stale-chunk skip counter per stream after a divergence
    inflight = []                                           # FIFO of rounds; each = [(spos_b, ds_b) | None] over b
    while not all(done) or inflight:
        while len(inflight) < depth and not all(done):      # fill the in-flight window (speculative per stream)
            tids = []; row = []; sb = []
            for b in range(B):
                if done[b]:
                    tids.append([cur[b]] * (K + 1)); row.append(None); sb.append(pos[b]); continue
                ds = drafters[b].fetch()
                tids.append([dprefix[b][-1]] + ds); row.append((spos[b], ds)); sb.append(spos[b])
                dprefix[b] = dprefix[b] + ds; spos[b] += K; drafters[b].request(dprefix[b], K)
            send_msg(pipe_sock, {"op": "verify_batch", "token_ids_b": tids, "start_b": sb})
            inflight.append(row)
        if not inflight:
            break
        tr = time.time(); rb = recv_msg(rx); t_recv += time.time() - tr   # rb: [B][K+1] per-stream argmax
        row = inflight.pop(0); rounds += 1
        for b in range(B):
            if row[b] is None or done[b]:
                continue
            if discard[b] > 0:                              # stale chunk from before this stream's last divergence
                discard[b] -= 1; wasted += 1; continue
            _, ds = row[b]; r = rb[b]; n = 0
            for j in range(K):
                if ds[j] == r[j]: n += 1
                else: break
            if n == K:
                out[b].extend(ds); pos[b] += K; cur[b] = ds[-1]; committed = ds
            else:                                           # divergence: commit prefix, drop this stream's stale in-flight, re-draft
                committed = ds[:n] + [r[n]]; out[b].extend(committed); cur[b] = r[n]; pos[b] += n + 1
                discard[b] = sum(1 for rr in inflight if rr[b] is not None)
                drafters[b].fetch(); dprefix[b] = prompts[b] + out[b]; spos[b] = pos[b]; drafters[b].request(dprefix[b], K)
            if len(out[b]) >= mx[b] or (cur[b] in eos_set) or (eos_set & set(committed)):
                done[b] = True
    dt = time.time() - t0
    res = []
    for b in range(B):                                  # trim at first eos, per stream
        o = out[b]
        for ee in eos_set:
            if ee in o: o = o[:o.index(ee)]; break
        res.append({"ok": True, "output_ids": o, "n_tokens": len(o), "prompt_tokens": len(prompts[b]),
                    "text": tok.decode(o, skip_special_tokens=True)})
    return {"streams": res, "B": B, "rounds": rounds, "depth": depth, "wasted": wasted, "dt": dt,
            "prefill_s": prefill_s, "agg_tok_s": sum(len(r["output_ids"]) for r in res) / max(dt, 1e-9)}


def _load(stage, nstages, lo, hi):
    S.vllm_ctx()
    layers = [S.Layer(i) for i in range(lo, hi)]
    parts = {"layers": layers, "head": stage == 0, "tail": stage == nstages - 1}
    if parts["head"]:
        parts["embed_w"] = S.raw("model.embed_tokens.weight").to(torch.bfloat16).to(dev)
    if parts["tail"]:
        parts["norm_w"] = S.raw("model.norm.weight").float().to(dev)
        parts["lm_head_w"] = S.raw("lm_head.weight").to(torch.bfloat16).to(dev)
    print(f"[s{stage}] loaded layers [{lo}:{hi}] ({torch.cuda.memory_allocated()/1e9:.1f}GB) — warming", flush=True)
    with torch.no_grad():
        S.run_block(layers, 0, torch.randn(1, 4, S.H, dtype=torch.bfloat16, device=dev) * 0.1, S._CTX[1])
        for L in layers:
            L.reset()
    torch.cuda.synchronize()
    if S.M25_SDPA:
        _sdpa_backend_probe(stage)
    return parts


def _tail_logits(h, parts):
    x = h.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + S.EPS) * parts["norm_w"]
    return (x.to(torch.bfloat16) @ parts["lm_head_w"].t())   # [1, s, vocab]


def _block(grs, layers, start, x, vcfg):
    """Run one block. Route fixed-shape verify/decode blocks (small s = K+1) through a lazily-captured
    CUDA graph when M25_CUDA_GRAPH (the proven 3.4x lever); prefill (large s) stays eager. grs caches one
    GraphRunner per block size. The graphed path is bit-equivalent to run_block (proven), so receipts +
    spec-decode losslessness are preserved."""
    if S.M25_CUDA_GRAPH and x.shape[1] <= 64:
        s = x.shape[1]
        gr = grs.get(s)
        if gr is None:
            grs[s] = gr = S.GraphRunner(layers, vcfg, s)
        return gr.run(start, x)
    return S.run_block(layers, start, x, vcfg)


def _merge_aux(upstream):
    """EAGLE: accumulate this stage's captured aux hidden states (S._AUX, only the aux layers in [lo,hi))
    onto whatever upstream stages already collected, so the tail returns all of [1,30,58] to the coordinator.
    Keys are str(layer_id); values are [s,H] bf16 cpu tensors — or [fp8_tensor, scale] pairs under
    M25_FP8_AUX (halves the dominant EAGLE decode payload; upstream entries are already packed and pass
    through untouched; the coordinator dequantizes once in _unpack). No-op unless M25_EAGLE."""
    acc = dict(upstream or {})
    if S.M25_EAGLE:
        for li, h in S._AUX.items():
            if M25_FP8_AUX:
                q, sc = _pack_h(h)
                acc[str(li)] = [q, sc]
            else:
                acc[str(li)] = h.cpu()
    return acc


def _tail_accept(srv, pending=None, ret=None, timeout=None):
    """Tail bring-up handshake. TWO connections land on the tail: the coordinator-RETURN channel (greets
    with {op:'hello_return'} the instant it connects, because the coordinator sends data immediately) and
    the PREDECESSOR ring stream (silent until the first job byte flows). With the libp2p sidecars the
    predecessor's downstream connect to our ENG_IN is established LAZILY — only when the upstream stage
    first forwards data — and that only happens after the coordinator gets `ret_ok`. So requiring BOTH
    connections before acking the return channel (the old `c1=accept(); c2=accept()`) is a circular
    deadlock: no `ret_ok` until the predecessor connects, no predecessor data until `ret_ok`. The tail
    then wedges on the 2nd accept and the coordinator hangs forever on recv(ret_ok) with EMPTY output.

    Fix: accept connections one at a time and ack the return channel the INSTANT we identify it — do not
    wait for the predecessor. Whichever connection greets with hello_return is the return channel; a
    connection that speaks a JOB frame first is a (re-dialing direct-TCP) predecessor and its frame is
    handed back as `first_msg` (specpipe fill()'s semantic — closing it kill-looped stage replacement);
    a SILENT connection is adopted as the predecessor once the return channel exists (the libp2p
    predecessor connects lazily and never speaks first). Returns (ret, pred, first_msg); first_msg is
    None on the silent-predecessor path. Blocks until both channels exist.

    `pending` seeds the accepted-but-unidentified pool with leftovers from a torn-down session (a new
    coordinator's hello_return may already have been accepted when the old predecessor died) — closing
    them instead would EOF the reconnecting peer and re-wedge. `ret` seeds an already-live return
    channel (kept across a pred-death that raced a fresh coordinator) — then only the predecessor is
    awaited. `timeout` bounds the greeting read so a half-sent frame can't hang bring-up."""
    pred, first, pending = None, None, list(pending or [])
    while ret is None or pred is None:
        if ret is not None and pred is None and pending:   # silent conn + live ret -> it's the predecessor
            pred = pending.pop()
            for extra in pending:                          # 2-conn ring never leaves extras, but stay clean
                try: extra.close()
                except OSError: pass
            pending = []
            break
        ready, _, _ = select.select([srv] + pending, [], [])   # wake on a new conn OR a pending conn speaking
        for s in ready:
            if s is srv:
                c, _ = srv.accept(); c.setsockopt(*NODELAY); _keepalive(c); pending.append(c); continue
            pending.remove(s)                              # it spoke -> classify by content
            if timeout:
                s.settimeout(timeout)
            try:
                hello = recv_msg(s)
            except EDGE_ERRORS:
                try: s.close()
                except OSError: pass
                continue
            if isinstance(hello, dict) and hello.get("op") == "hello_return":
                if ret is not None:                        # newer coordinator wins (the old one is dead/stale)
                    try: ret.close()
                    except OSError: pass
                ret = s
                try:
                    send_msg(ret, "ret_ok")                # ACK NOW so the coordinator proceeds — pred can connect after
                except EDGE_ERRORS:                        # greeter died between hello and ack: not a session event
                    try: ret.close()
                    except OSError: pass
                    ret = None
            elif isinstance(hello, dict) and "op" in hello:
                if pred is not None:                       # a speaking predecessor replaces a stale one
                    try: pred.close()
                    except OSError: pass
                pred = s; first = hello                    # its first frame is real job data — hand it back
            else:
                try: s.close()                             # unexpected greeter (junk/probe) -> drop, keep waiting
                except OSError: pass
    return ret, pred, first


def serve(stage, nstages, lo, hi, port, nxt, timeout):
    parts = _load(stage, nstages, lo, hi)
    layers = parts["layers"]
    vcfg = S._CTX[1]
    graph_runners = {}                                # opt-in CUDA-graph cache (M25_CUDA_GRAPH); persists across jobs
    def _dial_fwd():
        host, p = nxt.rsplit(":", 1)
        s = socket.socket(); s.settimeout(timeout); s.connect((host, int(p))); s.setsockopt(*NODELAY)
        _keepalive(s)
        return s
    nxt_sock = None
    if not parts["tail"]:
        nxt_sock = _dial_fwd()                        # launch-time dial stays strict: a dead --next at boot is a launcher bug
        print(f"[s{stage}] forward connected -> {nxt}", flush=True)
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port)); srv.listen(2)
    print(f"[s{stage}] WARM, listening :{port}", flush=True)

    if parts["tail"]:
        node_key = load_or_make_node_key(NODE_KEY_PATH) if RECEIPTS else None
        # serve_tail_direct: ack the coordinator-return as soon as we identify it, THEN take the (lazily
        # connecting) predecessor — see _tail_accept for why requiring both up front deadlocks bring-up.
        # CHURN-RESILIENT (specpipe.serve_tail_fast's model): pred and ret have INDEPENDENT lifecycles.
        # ret dies alone (coordinator exit/crash, gateway restart) -> keep the predecessor + warm KV,
        # accept the next hello_return MID-SESSION, and drop the dead job's in-flight replies until the
        # next reset (a stale reply poisons the new coordinator's handshake). Only a pred death tears the
        # session down — and takes ret with it, so a reset-ok can never go to a dead coordinator's channel.
        # This tail was the wedge's first domino: it closed pred whenever ret died, which EOF'd every
        # upstream stage in turn and left the whole warm ring needing a relaunch per coordinator.
        ret = pred = None; pending = []; stale = False

        def _ret_send(o):
            # Deliver a reply to the coordinator-return. On failure the RETURN channel is dead, not the
            # session: drop only ret (the next coordinator brings a fresh one) and mark the job stale.
            nonlocal ret, stale
            if ret is None:
                return
            try:
                send_msg(ret, o)
            except EDGE_ERRORS as e:
                print(f"[tail] return edge died on send ({type(e).__name__}); keeping predecessor+KV", flush=True)
                try: ret.close()
                except OSError: pass
                ret = None; stale = True

        while True:
            if pred is None:
                # re-accept the predecessor (and the return channel unless a live one was carried over)
                ret, pred, queued = _tail_accept(srv, pending, ret=ret, timeout=timeout)
                pending = []                     # consumed (became ret/pred or were closed) — don't double-select
                pred.settimeout(timeout)         # bounds a mid-frame stall; idle waiting happens in select below
                stale = False                    # any stale in-flight died with the old predecessor
                print("[tail] predecessor + coord-return connected", flush=True)
            signer = None
            with torch.no_grad():
                try:
                    while True:
                        # Multiplex: pred carries job data; srv carries a reconnecting coordinator's
                        # hello_return, which MUST be accepted mid-session or coordinator churn wedges the
                        # warm ring; pending holds accepted-but-silent conns (never block-recv a silent
                        # conn — the _tail_accept bring-up deadlock, same reasoning). `queued` is a job
                        # frame already read off a newly-adopted predecessor — process it before selecting.
                        if queued is not None:
                            msg, queued = _hrecv(queued), None
                        else:
                            ready, _, _ = select.select([srv, pred] + pending, [], [])
                            if srv in ready:
                                c, _ = srv.accept(); c.setsockopt(*NODELAY); _keepalive(c); pending.append(c)
                                if len(pending) > 8:       # reap silent junk before it grows the select set
                                    old = pending.pop(0)
                                    try: old.close()
                                    except OSError: pass
                                continue
                            spoke = next((s for s in pending if s in ready), None)
                            if spoke is not None:
                                pending.remove(spoke)
                                spoke.settimeout(timeout)  # a half-sent greeting must not hang the live session
                                try:
                                    hello = recv_msg(spoke)
                                except EDGE_ERRORS:
                                    try: spoke.close()
                                    except OSError: pass
                                    continue
                                if isinstance(hello, dict) and hello.get("op") == "hello_return":
                                    if ret is not None:    # coordinator churn: the old return channel is dead
                                        try: ret.close()   # even if this write-only socket never told us
                                        except OSError: pass
                                        stale = True       # in-flight traffic belongs to the dead job
                                    ret = spoke
                                    try:
                                        send_msg(ret, "ret_ok"); ret.settimeout(None)   # ret is untimed, like bring-up
                                        print("[tail] coord-return (re)connected mid-session", flush=True)
                                    except EDGE_ERRORS:    # reconnector died between hello and ack: not a pred event
                                        try: ret.close()
                                        except OSError: pass
                                        ret = None
                                elif isinstance(hello, dict) and "op" in hello:
                                    # a NEW predecessor speaking its first job frame (direct-TCP stage
                                    # replacement after a silent pred death) — adopt it, keep the frame
                                    try: pred.close()
                                    except OSError: pass
                                    pred = spoke; queued = hello; stale = False
                                    print("[tail] predecessor REPLACED mid-session", flush=True)
                                else:
                                    try: spoke.close()     # unexpected greeter (junk/probe) -> drop
                                    except OSError: pass
                                continue
                            if pred not in ready:
                                continue
                            msg = _hrecv(recv_msg(pred))
                        if stale or ret is None:
                            # These messages belong to a job whose coordinator died: don't compute them,
                            # never answer them. The next job boundary (reset) re-arms the session — but
                            # only once a live return channel exists to ack it (a reset with ret=None is
                            # the dead coordinator's own; its successor always hellos before sending).
                            if ret is None or msg.get("op") not in ("reset", "reset_batch"):
                                continue
                            stale = False
                        if msg["op"] == "reset":
                            for L in layers:
                                L.reset()
                            if RECEIPTS:                    # start this job's per-stage activation hash-chain
                                signer = ReceiptSigner(node_key, msg.get("swarm_id", "swarm"),
                                                       msg.get("job_id", "job"), lo, hi)
                            _ret_send("ok"); continue
                        if msg["op"] == "receipt":          # job done: sign + return the full ring's receipts
                            if RECEIPTS and signer is not None:
                                msg.setdefault("receipts", []).append({"stage": "tail", **signer.finalize()})
                            _ret_send(msg.get("receipts", [])); continue
                        if msg["op"] == "reset_batch":      # continuous batching: logical reset of all rows
                            for L in layers: L.reset()
                            _ret_send("ok"); continue
                        if msg["op"] == "verify_batch":     # batched decode: [B,K+1,H] -> per-stream argmax [B][K+1]
                            h = S.run_block_decode_b(layers, torch.tensor(msg["start_b"], device=dev), msg["h"].to(dev), vcfg)
                            _ret_send(_tail_logits(h, parts).argmax(-1).tolist()); continue
                        if msg.get("prefill") and "stream" in msg:  # BATCHED prefill into row b (single-stream prefill has no 'stream' -> falls through to the normal path)
                            h = S.run_block_prefill_b(layers, msg["stream"], msg["start"], msg["h"].to(dev), vcfg)
                            _ret_send(_tail_logits(h, parts).argmax(-1)[0].tolist()); continue
                        if msg.get("tree"):                 # EAGLE tree-verify: per-node argmax over the tree-masked block
                            x = msg["h"].to(dev)
                            h = S.run_block_tree(layers, msg["start"], x, vcfg, msg["parents"], msg["pos_ids"])
                            if RECEIPTS and signer is not None:   # attest tree blocks too — verification must not silently turn off under M25_TREE
                                signer.observe(_act_digest(x), _act_digest(h))
                            toks = _tail_logits(h, parts).argmax(-1)[0].tolist()
                            _ret_send({"toks": toks, "aux": _merge_aux(msg.get("aux"))} if S.M25_EAGLE else toks); continue
                        x = msg["h"].to(dev)
                        h = _block(graph_runners, layers, msg["start"], x, vcfg)
                        if RECEIPTS and signer is not None:   # attest this block's input->output transform
                            signer.observe(_act_digest(x), _act_digest(h))
                        toks = _tail_logits(h, parts).argmax(-1)[0].tolist()
                        _ret_send({"toks": toks, "aux": _merge_aux(msg.get("aux"))} if S.M25_EAGLE else toks)
                except EDGE_ERRORS as e:
                    # PREDECESSOR death (ret failures are absorbed in _ret_send and never land here):
                    # tear the session down and re-accept. The ret goes too — a reset-ok must never reach
                    # a dead coordinator's channel — UNLESS it was JUST replaced by a reconnecting
                    # coordinator (stale set, no reset consumed yet): then this EOF is the OLD session's
                    # cascade arriving late, and killing the fresh ret would fail the very retry that
                    # churn recovery exists for.
                    keep = ret if (stale and ret is not None) else None
                    print(f"[tail] predecessor edge closed ({type(e).__name__}); re-accepting "
                          f"{'predecessor (fresh coord-return kept)' if keep else 'both channels'}", flush=True)
                    for L in layers:
                        L.reset()
                    for s in (pred, None if keep else ret):
                        if s is not None:
                            try: s.close()
                            except OSError: pass
                    pred, ret = None, keep
        return

    # head / middle: single predecessor connection, FIRE-FORWARD (direct mode, no relay-back)
    node_key = load_or_make_node_key(NODE_KEY_PATH) if RECEIPTS else None
    while True:
        tries = 0
        while nxt_sock is None:                       # forward link dropped (churn cascade): rebuild it BEFORE
            try:                                      # accepting a new predecessor, so the ring re-handshakes
                nxt_sock = _dial_fwd()                # front-to-back onto WARM stages — no relaunch, no reload
                print(f"[s{stage}] forward link rebuilt -> {nxt}", flush=True)
            except OSError:
                tries += 1                            # dial forever: a stage holding warm weights is worth more
                if tries % 60 == 0:                   # waiting than dead (downstream may be mid-restart)
                    print(f"[s{stage}] forward re-dial {nxt} still failing ({tries} tries)", flush=True)
                time.sleep(0.5)
        conn, _ = srv.accept(); conn.setsockopt(*NODELAY); _keepalive(conn)
        print(f"[s{stage}] predecessor connected", flush=True)
        signer = None
        with torch.no_grad():
            try:
                while True:
                    msg = _hrecv(recv_msg(conn))
                    if msg["op"] == "reset":
                        for L in layers:
                            L.reset()
                        if RECEIPTS:                            # start this job's per-stage activation hash-chain
                            signer = ReceiptSigner(node_key, msg.get("swarm_id", "swarm"),
                                                   msg.get("job_id", "job"), lo, hi)
                        send_msg(nxt_sock, msg); continue       # propagate reset down the chain
                    if msg["op"] == "receipt":                  # job done: sign + accumulate forward to the tail
                        if RECEIPTS and signer is not None:
                            msg.setdefault("receipts", []).append({"stage": stage, **signer.finalize()})
                        send_msg(nxt_sock, msg); continue
                    if msg["op"] == "reset_batch":              # continuous batching: propagate logical reset
                        for L in layers: L.reset()
                        send_msg(nxt_sock, msg); continue
                    if msg["op"] == "verify_batch":             # batched decode: head embeds [B,K+1], else fwd [B,K+1,H]
                        if parts["head"]:
                            h = torch.nn.functional.embedding(torch.tensor(msg["token_ids_b"], device=dev), parts["embed_w"])
                        else:
                            h = msg["h"].to(dev)
                        h = S.run_block_decode_b(layers, torch.tensor(msg["start_b"], device=dev), h, vcfg)
                        send_msg(nxt_sock, _hsend({"op": "verify_batch", "h": h, "start_b": msg["start_b"]})); continue
                    if msg.get("prefill") and "stream" in msg:  # BATCHED prefill into row b (single-stream prefill has no 'stream' -> normal path)
                        if parts["head"]:
                            h = torch.nn.functional.embedding(torch.tensor([msg["token_ids"]], device=dev), parts["embed_w"])
                        else:
                            h = msg["h"].to(dev)
                        h = S.run_block_prefill_b(layers, msg["stream"], msg["start"], h, vcfg)
                        send_msg(nxt_sock, _hsend({"op": "verify", "stream": msg["stream"], "h": h, "start": msg["start"], "prefill": True})); continue
                    if msg.get("tree"):                         # EAGLE tree-verify: tree-masked block, thread the tree forward
                        if parts["head"]:
                            h = torch.nn.functional.embedding(torch.tensor([msg["token_ids"]], device=dev), parts["embed_w"])
                        else:
                            h = msg["h"].to(dev)
                        x = h
                        h = S.run_block_tree(layers, msg["start"], h, vcfg, msg["parents"], msg["pos_ids"])
                        if RECEIPTS and signer is not None:     # attest tree blocks too — verification must not silently turn off under M25_TREE
                            signer.observe(_act_digest(x), _act_digest(h))
                        fwd = {"op": "verify", "tree": True, "h": h, "start": msg["start"],
                               "parents": msg["parents"], "pos_ids": msg["pos_ids"]}
                        if S.M25_EAGLE:
                            fwd["aux"] = _merge_aux(msg.get("aux"))
                        send_msg(nxt_sock, _hsend(fwd)); continue
                    if "token_ids" in msg:                      # head: embed the coordinator's token ids
                        h = torch.nn.functional.embedding(torch.tensor([msg["token_ids"]], device=dev), parts["embed_w"])
                    else:
                        h = msg["h"].to(dev)
                    x = h
                    h = _block(graph_runners, layers, msg["start"], h, vcfg)
                    if RECEIPTS and signer is not None:         # attest this block's input->output transform
                        signer.observe(_act_digest(x), _act_digest(h))
                    fwd = {"op": "verify", "h": h, "start": msg["start"]}
                    if S.M25_EAGLE:                              # carry aux hidden states forward to the tail (EAGLE)
                        fwd["aux"] = _merge_aux(msg.get("aux"))
                    send_msg(nxt_sock, _hsend(fwd))
            except EDGE_ERRORS as e:
                print(f"[s{stage}] edge closed ({type(e).__name__}); reset + drop forward link", flush=True)
                for L in layers:
                    L.reset()
                for s in (conn, nxt_sock):            # deliberately drop the forward link too: the next stage
                    if s is not None:                 # sees EOF and cascades, so the WHOLE ring re-handshakes
                        try: s.close()                # fresh (warm weights intact) and a new coordinator can
                        except OSError: pass          # drive it — specpipe's proven recovery choreography
                nxt_sock = None


def _sweep_summary(rows):
    """Pure: format a K/depth sweep into an aligned table + the best-throughput row. No torch/model
    deps so it unit-tests standalone (research/m25_sweep_test.py). `h_kb` is the per-traversal
    inter-stage hidden-state payload (K+1)*H*fp16 — the bandwidth term that caps how far K pays off
    once GPU compute is flat in token count, so it's printed next to tok/s to read the sweep."""
    hdr = f"{'K':>3} {'depth':>5} {'tok/s':>7} {'g':>6} {'accept':>7} {'prefill':>8} {'ntok':>5} {'h/trav':>8}"
    lines = ["", "=== M2.5 swarm sweep (decode tok/s, warm over libp2p) ===", hdr, "-" * len(hdr)]
    for r in rows:
        flag = "" if r.get("ok") else "  <-- FAIL"
        lines.append(f"{r['K']:>3} {r['depth']:>5} {r['tok_s']:>7.2f} {r['g']:>6.2f} "
                     f"{r['accept'] * 100:>6.0f}% {r['prefill_s']:>7.2f}s {r['ntok']:>5} {r['h_kb']:>6.1f}K{flag}")
    ok = [r for r in rows if r.get("ok") and r["tok_s"] > 0]
    best = max(ok, key=lambda r: r["tok_s"]) if ok else None
    if best:
        lines.append("-" * len(hdr))
        lines.append(f"BEST: K={best['K']} depth={best['depth']} -> {best['tok_s']:.2f} tok/s "
                     f"(g={best['g']:.2f}, accept={best['accept'] * 100:.0f}%)")
    return "\n".join(lines), best


def _run_job(pipe, ret, tok, messages, k, max_new, timeout, d, ngram_n, prefill_chunk, tools=None):
    """One coordinate_pipe job with a FRESH drafter (clean n-gram state per config). Sockets are
    reused across jobs — coordinate_pipe drains in-flight + opens each job with `reset`, which clears
    every stage's KV, so back-to-back jobs on the same ring are clean. make_drafter adds the EAGLE
    hybrid when M25_EAGLE=1."""
    drafter = make_drafter(ngram_n)
    return coordinate_pipe(pipe, tok, messages, k, max_new, timeout, d, ret_sock=ret,
                           local_draft=drafter, tools=tools, prefill_chunk=prefill_chunk, max_ctx=131072)


def _validate(pipe, ret, tok, K, depth, ngram_n, prefill_chunk, timeout, longctx_path):
    """FULL usability pass on ONE warm ring (jobs reuse the socket like the sweep). Exercises every
    deploy-ready capability end-to-end over libp2p and prints a PASS/FAIL per capability. Receipts are
    proven on every job when the ring was launched with SHARD_RECEIPTS=1."""
    WEATHER = [{"type": "function", "function": {"name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "city name"}},
                               "required": ["city"]}}}]

    print("\n[validate] === FULL USABILITY PASS (warm, libp2p) ===", flush=True)

    # 1) TOOL CALLING — model must emit a structured get_weather call
    m = [{"role": "user", "content": "Use the get_weather tool to check the weather in Paris."}]
    r = _run_job(pipe, ret, tok, m, K, 256, timeout, depth, ngram_n, prefill_chunk, tools=WEATHER)
    p = parse_completion(r["text"]); tc = p["tool_calls"]
    print(f"[validate] 1.TOOLS      {'PASS' if tc else 'FAIL'}  tool_calls={json.dumps(tc, ensure_ascii=False)[:220]}  "
          f"receipts_ok={r.get('receipts_ok')}  {r['tok_s']:.1f}tok/s", flush=True)

    # 2) EXTENDED CONVO — a real ~9-turn back-and-forth; final turn must RECALL a fact stated in turn 1
    #    (tests render_ids threading a long history + cross-turn recall, the "long convo" usability dimension)
    m = [{"role": "user", "content": "Hey, I'm setting up a decentralized inference swarm. My node ID is SWARM-NODE-4417 and I'm running 5 RTX 5090s scattered across Europe."},
         {"role": "assistant", "content": "Nice — 5x5090 scattered across Europe is a solid ring. What model are you serving on it?"},
         {"role": "user", "content": "MiniMax-M2.5, sharded across the nodes over libp2p. I'm getting about 20 tokens per second warm."},
         {"role": "assistant", "content": "That's a healthy warm number for a 5-stage pipeline over WAN. Are you using speculative decoding to hide the per-hop latency?"},
         {"role": "user", "content": "Yeah, n-gram drafting — works great on copy and retrieval tasks. I also need tool calling and long context to work."},
         {"role": "assistant", "content": "Both are supported: the coordinator threads tools through the chat template, and chunked prefill handles long context without OOM."},
         {"role": "user", "content": "Good. I'm also worried about trusting the nodes I don't control."},
         {"role": "assistant", "content": "Each node signs a per-stage receipt with its own key and the coordinator verifies full layer coverage, so no node is paid without proving its block."},
         {"role": "user", "content": "Perfect. Now remind me — what was the node ID I gave you at the very start, and how many GPUs did I say I'm running?"}]
    r = _run_job(pipe, ret, tok, m, K, 96, timeout, depth, ngram_n, prefill_chunk, tools=None)
    p = parse_completion(r["text"]); ans = (p["content"] or "").strip()
    recall = ("SWARM-NODE-4417" in ans) and ("5" in ans)
    print(f"[validate] 2.CONVO(9-turn) {'PASS (recalled turn-1 facts)' if recall else 'PARTIAL/FAIL'}  "
          f"answer={ans[:200]!r}  receipts_ok={r.get('receipts_ok')}", flush=True)

    # 3) LONG CONTEXT — needle retrieval far past the old 8192 RoPE cap (proves the rope fix + chunked prefill)
    lc = open(longctx_path).read()
    m = [{"role": "user", "content": lc}]
    r = _run_job(pipe, ret, tok, m, K, 96, timeout, depth, ngram_n, prefill_chunk, tools=None)  # M2.5 reasons before answering; 24 only covered the restate (false FAIL on 2026-06-28)
    p = parse_completion(r["text"]); ans = (p["content"] or r["text"]).strip()
    hit = "ZX-PAYLOAD-7731" in r["text"]   # needle anywhere in the output (model surfaces it via reasoning, then answers)
    print(f"[validate] 3.LONG-CTX   {'PASS (needle found)' if hit else 'FAIL'}  prompt_tokens={r['prompt_tokens']}  "
          f"prefill={r['prefill_s']:.1f}s  answer={ans[:80]!r}  receipts_ok={r.get('receipts_ok')}", flush=True)

    print("[validate] === END ===", flush=True)


def coord(head_ep, tail_ep, prompt, K, max_new, depth, ngram_n, timeout, sweep=None, sweep_depth=None, prefill_chunk=512, validate=False):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S.DIR, trust_remote_code=True)
    hh, hp = head_ep.rsplit(":", 1); th, tp = tail_ep.rsplit(":", 1)
    pipe = socket.create_connection((hh, int(hp)), timeout=timeout); pipe.setsockopt(*NODELAY)
    ret = socket.create_connection((th, int(tp)), timeout=timeout); ret.setsockopt(*NODELAY); ret.settimeout(timeout)
    send_msg(ret, {"op": "hello_return"})                       # identify the return channel to the tail
    recv_msg(ret)                                               # wait ret_ok: tail confirmed ret before any reset flows
    messages = [{"role": "user", "content": prompt}]

    if validate:                                               # full usability pass (tools+multi-turn+long-ctx+receipts)
        _validate(pipe, ret, tok, K, depth, ngram_n, prefill_chunk, timeout, "/root/longctx_prompt.txt")
        return

    if sweep or sweep_depth:                                    # K/depth throughput sweep -> tok/s table
        Ks = sweep or [K]; Ds = sweep_depth or [depth]
        print(f"[coord] SWEEP K={Ks} depth={Ds} ngram={ngram_n} -> head {head_ep}, ret {tail_ep}", flush=True)
        rows = []
        for d in Ds:
            for k in Ks:
                row = {"K": k, "depth": d, "tok_s": 0.0, "g": 0.0, "accept": 0.0,
                       "prefill_s": 0.0, "ntok": 0, "h_kb": (k + 1) * S.H * 2 / 1024, "ok": False, "text": ""}
                try:
                    r = _run_job(pipe, ret, tok, messages, k, max_new, timeout, d, ngram_n, prefill_chunk)
                    row.update(tok_s=r["tok_s"], g=r["toks_per_traversal"], accept=r["mean_accept"] / max(k, 1),
                               prefill_s=r["prefill_s"], ntok=r["n_tokens"], ok=r.get("ok", False), text=r.get("text", ""))
                except Exception as e:
                    row["err"] = f"{type(e).__name__}: {e}"
                rows.append(row)
                print(f"[sweep] K={k:>2} depth={d}: {row['tok_s']:>6.2f} tok/s  g={row['g']:.2f}  "
                      f"accept={row['accept'] * 100:.0f}%  ({'ok' if row['ok'] else row.get('err', 'FAIL')})", flush=True)
        table, best = _sweep_summary(rows)
        print(table, flush=True)
        if best:
            print("\n[sweep] best output:\n" + (parse_completion(best["text"])["content"] or best["text"])[:800], flush=True)
        return

    print(f"[coord] pipelined (K={K} depth={depth} ngram={ngram_n}) -> head {head_ep}, ret {tail_ep}", flush=True)
    r = _run_job(pipe, ret, tok, messages, K, max_new, timeout, depth, ngram_n, prefill_chunk)
    if r.get("ok"):
        parsed = parse_completion(r["text"])
        print(f"\n[coord] {r['n_tokens']}tok  {r['tok_s']:.2f} tok/s  g={r['toks_per_traversal']:.2f}  "
              f"mean_accept={r['mean_accept']:.2f}/{K}  prefill={r['prefill_s']:.2f}s  depth={depth}", flush=True)
        if r.get("decode_s"):                                # where the decode wall went (serial-path profile)
            other = r["decode_s"] - r.get("draft_s", 0) - r.get("ring_wait_s", 0)
            print(f"[coord] decode {r['decode_s']:.1f}s = ring-wait {r.get('ring_wait_s', 0):.1f}s "
                  f"+ draft {r.get('draft_s', 0):.1f}s + coord-other {other:.1f}s", flush=True)
        if parsed["reasoning_content"]:
            print("[coord] THINK:\n" + parsed["reasoning_content"][:600], flush=True)
        print("[coord] OUTPUT:\n" + (parsed["content"] or "")[:1200], flush=True)
        if parsed["tool_calls"]:
            print("[coord] TOOL_CALLS: " + json.dumps(parsed["tool_calls"], ensure_ascii=False)[:800], flush=True)
        if r.get("receipts"):
            print(f"[coord] === PROVE: {len(r['receipts'])} signed per-stage receipts ===", flush=True)
            print(f"[coord] PROVE verdict: {'ALL receipts valid + full layer coverage' if r.get('receipts_ok') else 'FAILED'}", flush=True)
        print("SHA:", hashlib.sha256(r["text"].encode()).hexdigest()[:12], flush=True)
    else:
        print("[coord] FAILED:", r, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="role", required=True)
    ps = sub.add_parser("stage")
    ps.add_argument("--stage", type=int, required=True); ps.add_argument("--nstages", type=int, required=True)
    ps.add_argument("--lo", type=int, required=True); ps.add_argument("--hi", type=int, required=True)
    ps.add_argument("--port", type=int, default=29610); ps.add_argument("--next", default=None)
    ps.add_argument("--timeout", type=int, default=600)
    pc = sub.add_parser("coord")
    pc.add_argument("--head", required=True); pc.add_argument("--tail", required=True)
    pc.add_argument("--prompt", default="Explain a decentralized inference swarm in 3 sentences.")
    pc.add_argument("--prompt-file", default=None); pc.add_argument("--K", type=int, default=8)   # K=8 = the measured sweet spot (2026-06-27 sweep; K=6 left ~2x on the table)
    pc.add_argument("--depth", type=int, default=4); pc.add_argument("--max-new", type=int, default=256)
    pc.add_argument("--ngram-n", type=int, default=3); pc.add_argument("--timeout", type=int, default=600)
    pc.add_argument("--sweep", default=None, help="comma K list, e.g. 4,6,8,12,16 (drafter margin is safe to K<=16)")
    pc.add_argument("--sweep-depth", default=None, help="comma depth list, e.g. 2,4,8 (default: --depth)")
    pc.add_argument("--prefill-chunk", type=int, default=512, help="prefill tokens per ring traversal; under M25_SDPA (default) attn is O(chunk) not O(chunk*ctx), so this is now a TTFT/bandwidth knob, not the OOM guard")
    pc.add_argument("--validate", action="store_true", help="full usability pass: tools + multi-turn + long-ctx (needle) + receipts, one warm ring")
    a = ap.parse_args()

    def _ilist(s): return [int(x) for x in s.split(",") if x.strip()] if s else None

    if os.environ.get("SHARD_TRANSPORT") != "libp2p":   # raw-wire mode: load the PSK before any send/recv (libp2p sidecar self-seals)
        import wire; wire.key_from_env()

    if a.role == "stage":
        serve(a.stage, a.nstages, a.lo, a.hi, a.port, a.next, a.timeout)
    else:
        prompt = open(a.prompt_file).read() if a.prompt_file else a.prompt
        coord(a.head, a.tail, prompt, a.K, a.max_new, a.depth, a.ngram_n, a.timeout,
              sweep=_ilist(a.sweep), sweep_depth=_ilist(a.sweep_depth), prefill_chunk=a.prefill_chunk, validate=a.validate)
