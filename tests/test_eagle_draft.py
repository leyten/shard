"""EagleDrafter regression net — the buffered/batched drafter must PREDICT identically to the original
list-based implementation (kept verbatim below as _RefEagle). The drafter is lossless-by-construction
(it only proposes; the ring greedy-verifies), so the bar is equal PROPOSALS, not bit-equal floats — the
tests run in fp32 so batched-vs-looped linear algebra can't flip an argmax and equality is exact in
practice. Also covers: chain-scratch isolation (propose() never mutates the committed cache), the
prefill chunk-pairing invariant, and fp8 tensors round-tripping the raw-TCP wire codec.

Run: pytest tests/test_eagle_draft.py -q   (CPU-only, no GPU / no model dir needed)
"""
import json
import os
import sys
import tempfile

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase0"))
from eagle_draft import EagleDrafter, prefill_pair_tokens, _rms, _rotate_half  # noqa: E402

H, NH, NKV, HD, I = 32, 4, 2, 8, 64
DV, TV = 50, 100                      # draft vocab, target vocab
K = 8


def _make_head(seed=0):
    """Tiny synthetic EAGLE-3 head dir + M2.5-style embed (mirrors the real head's STRUCTURE)."""
    d = tempfile.mkdtemp()
    json.dump({"hidden_size": H, "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD,
               "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "draft_vocab_size": DV},
              open(f"{d}/config.json", "w"))
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: (torch.randn(*s, generator=g) * 0.02).to(torch.bfloat16)
    W = {
        "fc.weight": rnd(H, 3 * H),
        "midlayer.input_layernorm.weight": torch.ones(H, dtype=torch.bfloat16),
        "midlayer.hidden_norm.weight": torch.ones(H, dtype=torch.bfloat16),
        "midlayer.self_attn.q_proj.weight": rnd(NH * HD, 2 * H),
        "midlayer.self_attn.k_proj.weight": rnd(NKV * HD, 2 * H),
        "midlayer.self_attn.v_proj.weight": rnd(NKV * HD, 2 * H),
        "midlayer.self_attn.o_proj.weight": rnd(H, NH * HD),
        "midlayer.post_attention_layernorm.weight": torch.ones(H, dtype=torch.bfloat16),
        "midlayer.mlp.gate_proj.weight": rnd(I, H),
        "midlayer.mlp.up_proj.weight": rnd(I, H),
        "midlayer.mlp.down_proj.weight": rnd(H, I),
        "norm.weight": torch.ones(H, dtype=torch.bfloat16),
        "lm_head.weight": rnd(DV, H),
        "d2t": torch.randint(0, TV - DV, (DV,), generator=g, dtype=torch.int64),
    }
    save_file(W, f"{d}/model.safetensors")
    embed = (torch.randn(TV, H, generator=g) * 0.02).to(torch.bfloat16)
    return d, embed


class _RefEagle(EagleDrafter):
    """The ORIGINAL (pre-buffer) implementation, verbatim from master — per-slot k/v lists, torch.cat +
    repeat_interleave per propose. The optimized drafter must match its proposals."""

    def reset(self):
        self.kc = []
        self.vc = []
        self.ctx_len = 0
        self._last_h = None
        self._last_tok = None
        self._last_pos = -1

    @torch.no_grad()
    def extend(self, tokens, auxes, base_pos):
        if tokens is None:
            return
        tokens = tokens.tolist() if torch.is_tensor(tokens) else list(tokens)
        n = len(tokens)
        if n == 0:
            return
        lin = torch.nn.functional.linear
        fc_out = lin(self._aux_to_mat(auxes, n), self.fc)
        for i in range(n):
            tok = int(tokens[i])
            h = fc_out[i:i + 1]
            en = _rms(self.embed[tok].unsqueeze(0), self.in_ln, self.eps)
            hn = _rms(h, self.h_ln, self.eps)
            x = torch.cat([en, hn], -1)
            kk = lin(x, self.kp).view(1, self.NKV, self.HD)
            vv = lin(x, self.vp).view(1, self.NKV, self.HD)
            p = min(base_pos + i, self.cos.shape[0] - 1)
            cos = self.cos[p].view(1, 1, self.HD); sin = self.sin[p].view(1, 1, self.HD)
            kk = kk * cos + _rotate_half(kk) * sin
            self.kc.append(kk); self.vc.append(vv)
            self._last_h = h; self._last_tok = tok; self._last_pos = p
        self.ctx_len += n

    @torch.no_grad()
    def _draft(self, k):
        if self.ctx_len == 0 or self._last_h is None:
            return [int(self._last_tok) if self._last_tok is not None else 0] * k
        lin = torch.nn.functional.linear
        Kp = torch.cat(self.kc, 0).repeat_interleave(self.GRP, 1)
        Vp = torch.cat(self.vc, 0).repeat_interleave(self.GRP, 1)
        ck = []; cv = []
        out = []
        h = self._last_h; tok = self._last_tok; base = self._last_pos
        for i in range(k):
            en = _rms(self.embed[tok].unsqueeze(0), self.in_ln, self.eps)
            hn = _rms(h, self.h_ln, self.eps)
            x = torch.cat([en, hn], -1)
            res = h
            q = lin(x, self.qp).view(1, self.NH, self.HD)
            p = min(base + i, self.cos.shape[0] - 1)
            cos = self.cos[p].view(1, 1, self.HD); sin = self.sin[p].view(1, 1, self.HD)
            q = q * cos + _rotate_half(q) * sin
            if i == 0:
                Kt, Vt = Kp, Vp
            else:
                kk = lin(x, self.kp).view(1, self.NKV, self.HD)
                vv = lin(x, self.vp).view(1, self.NKV, self.HD)
                kk = kk * cos + _rotate_half(kk) * sin
                ck.append(kk.repeat_interleave(self.GRP, 1)); cv.append(vv.repeat_interleave(self.GRP, 1))
                Kt = torch.cat([Kp] + ck, 0); Vt = torch.cat([Vp] + cv, 0)
            qh = q.transpose(0, 1)
            att = torch.softmax((qh @ Kt.permute(1, 0, 2).transpose(-1, -2)).float() / (self.HD ** 0.5), -1).to(qh.dtype)
            o = (att @ Vt.permute(1, 0, 2)).transpose(0, 1).reshape(1, self.NH * self.HD)
            res = lin(o, self.op) + res
            hn2 = _rms(res, self.post_ln, self.eps)
            res = lin(torch.nn.functional.silu(lin(hn2, self.gp)) * lin(hn2, self.upp), self.dp) + res
            hf = _rms(res, self.norm, self.eps)
            did = int(lin(hf, self.lm).argmax(-1))
            tok = did + int(self.d2t[did])
            out.append(tok)
            h = hf if self.next_hidden == "final" else res
        return out


_WEIGHT_ATTRS = ("fc", "in_ln", "h_ln", "qp", "kp", "vp", "op", "post_ln",
                 "gp", "upp", "dp", "norm", "lm", "embed", "cos", "sin")


def _upcast_fp32(drafter):
    """fp32 weights so batched-vs-looped linear can't flip an argmax (drafter needs no bit-exactness in
    prod; the TEST wants deterministic equality)."""
    for a in _WEIGHT_ATTRS:
        setattr(drafter, a, getattr(drafter, a).float())
    drafter.reset()                       # rebuild caches at the new dtype
    return drafter


def _pair(seed=0, next_hidden="prenorm"):
    d, embed = _make_head(seed)
    new = _upcast_fp32(EagleDrafter(d, embed, device="cpu", next_hidden=next_hidden))
    ref = _upcast_fp32(_RefEagle(d, embed, device="cpu", next_hidden=next_hidden))
    return new, ref


def _aux(g, n):
    return torch.randn(n, 3, H, generator=g)


def test_propose_matches_reference_across_rounds():
    """Interleaved extend/propose over several rounds — proposals must match the original exactly."""
    for seed in range(3):
        for nh in ("prenorm", "final"):
            new, ref = _pair(seed, nh)
            g = torch.Generator().manual_seed(100 + seed)
            gt = torch.Generator().manual_seed(200 + seed)
            base = 0
            for rnd in range(4):
                n = [37, 1, 9, 5][rnd]                       # long prefill chunk, single commits, K+1 chunks
                toks = torch.randint(0, TV, (n,), generator=gt).tolist()
                aux = _aux(g, n)
                new.extend(toks, aux, base_pos=base)
                ref.extend(toks, aux.clone(), base_pos=base)
                base += n
                got, want = new.propose(K), ref.propose(K)
                assert got == want, f"seed={seed} nh={nh} round={rnd}: {got} != {want}"


def test_chain_scratch_never_mutates_committed_cache():
    """propose() writes chain k/v past ctx_len (scratch): repeated propose() is idempotent, and an
    extend() after propose() yields the same state as never having proposed."""
    new, ref = _pair(7)
    g = torch.Generator().manual_seed(7)
    gt = torch.Generator().manual_seed(8)
    toks = torch.randint(0, TV, (20,), generator=gt).tolist()
    aux = _aux(g, 20)
    new.extend(toks, aux, base_pos=0)
    ref.extend(toks, aux.clone(), base_pos=0)
    p1 = new.propose(K)
    assert new.propose(K) == p1, "second propose() differs — chain scratch leaked into the cache"
    toks2 = torch.randint(0, TV, (3,), generator=gt).tolist()
    aux2 = _aux(g, 3)
    new.extend(toks2, aux2, base_pos=20)                 # overwrites the scratch tail
    ref.extend(toks2, aux2.clone(), base_pos=20)         # ref never proposed since its last extend? it did not
    assert new.propose(K) == ref.propose(K), "extend after propose diverged from a propose-free reference"


def test_legacy_single_aux_path_matches_reference():
    """set_hidden -> request -> fetch (the deprecated single-aux seed) still matches."""
    new, ref = _pair(11)
    g = torch.Generator().manual_seed(11)
    seed_aux = torch.randn(3, H, generator=g)
    ids = [5, 6, 7, 8]
    for d in (new, ref):
        d.set_hidden(seed_aux.clone())
        d.request(ids, K)
    assert new.fetch() == ref.fetch()


def test_buffer_growth_across_doubling_boundary():
    """extend past the initial capacity — grown buffers must preserve the committed prefix."""
    new, ref = _pair(13)
    g = torch.Generator().manual_seed(13)
    gt = torch.Generator().manual_seed(14)
    base = 0
    for n in (900, 300, 400):                            # crosses the 1024 initial capacity
        toks = torch.randint(0, TV, (n,), generator=gt).tolist()
        aux = _aux(g, n)
        new.extend(toks, aux, base_pos=base)
        ref.extend(toks, aux.clone(), base_pos=base)
        base += n
    assert new.propose(K) == ref.propose(K)


def test_prefill_pair_tokens_invariant():
    """Concatenating the pairing over all chunks must equal gen_ids[1:] + [first_gen_token], for any
    chunking — the whole-prompt EAGLE context is exactly the old single-chunk contract, extended."""
    gen_ids = list(range(1000, 1097))                    # 97 tokens (not a multiple of any chunk size)
    first_gen = 7
    for chunk in (8, 16, 97, 200):
        starts = list(range(0, len(gen_ids), chunk))
        paired = []
        for s in starts:
            toks = [0] * min(chunk, len(gen_ids) - s)    # tail argmax stand-in; only the LAST chunk's last value is read
            if s == starts[-1]:
                toks[-1] = first_gen
            paired += prefill_pair_tokens(gen_ids, s, toks)
        assert paired == gen_ids[1:] + [first_gen], f"chunk={chunk}"


def test_wire_codec_roundtrips_fp8():
    """M25_FP8_WIRE tensors must survive the raw-TCP codec (they already survive shard/transport.py —
    the two tables must stay in lockstep)."""
    wire = pytest.importorskip("wire")
    t = (torch.randn(3, 5) * 4).to(torch.float8_e4m3fn)
    got = wire._unpack(wire._pack({"h": t, "start": 3}))
    assert got["start"] == 3 and got["h"].dtype == torch.float8_e4m3fn
    assert torch.equal(got["h"].view(torch.uint8), t.view(torch.uint8))
