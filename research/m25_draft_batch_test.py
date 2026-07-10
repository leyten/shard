"""Offline byte-identity gate for the BATCHED drafter forward (eagle_draft.draft_batch / fetch_b).

The batched coordinator runs B EAGLE chains; serial per-stream _draft() was the measured drafting tax
(~0.25s/stream/round at B=4 — rounds went DRAFTING-bound). draft_batch runs all B chains as ONE [B,...]
forward per chain step. That is only shippable if it is a pure wall-clock move: every row must be
BYTE-IDENTICAL to that fork's serial _draft(k) — same math, same argmax — because the fill loop treats
the two paths as interchangeable. This test pins that equivalence on CPU with a synthetic tiny head:

  1. draft_batch rows == per-fork _draft(k), across ragged context lengths + a degenerate (no-ctx) fork
  2. committed drafter state is untouched by a batched draw (scratch-tail discipline, like _draft)
  3. multi-round: extend (commit growth) between draws keeps rows identical
  4. fetch_b == [d.fetch() ...] over a mixed Hybrid (n-gram hit / miss) + plain-ngram drafter set,
     including the .matched routing flags
  5. both next_hidden carries ("prenorm" reference + "final" tunable)

NO GPU, NO model download: random weights in the real checkpoint format (safetensors + config.json).

  python research/m25_draft_batch_test.py
"""
import json
import os
import sys
import tempfile
import time

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase0"))

from eagle_draft import EagleDrafter, HybridDrafter, draft_batch, fetch_b   # noqa: E402
from ngram_draft import NgramDrafter                                        # noqa: E402

H, NH, NKV, HD, I2, DVOCAB, VOCAB = 64, 4, 2, 16, 128, 96, 160
K = 8


def make_head(tmp, seed=0):
    """A real-format EAGLE-3 checkpoint with random weights (the drafter reads config.json +
    model.safetensors; nothing about the math depends on the values being trained)."""
    json.dump({"hidden_size": H, "num_attention_heads": NH, "num_key_value_heads": NKV,
               "head_dim": HD, "rms_norm_eps": 1e-6, "rope_theta": 5e6,
               "draft_vocab_size": DVOCAB}, open(f"{tmp}/config.json", "w"))
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: (torch.randn(*s, generator=g) * 0.2).to(torch.bfloat16)
    w = {"fc.weight": r(H, 3 * H),
         "midlayer.input_layernorm.weight": (torch.ones(H) + torch.randn(H, generator=g) * 0.05).to(torch.bfloat16),
         "midlayer.hidden_norm.weight": (torch.ones(H) + torch.randn(H, generator=g) * 0.05).to(torch.bfloat16),
         "midlayer.self_attn.q_proj.weight": r(NH * HD, 2 * H),
         "midlayer.self_attn.k_proj.weight": r(NKV * HD, 2 * H),
         "midlayer.self_attn.v_proj.weight": r(NKV * HD, 2 * H),
         "midlayer.self_attn.o_proj.weight": r(H, NH * HD),
         "midlayer.post_attention_layernorm.weight": (torch.ones(H) + torch.randn(H, generator=g) * 0.05).to(torch.bfloat16),
         "midlayer.mlp.gate_proj.weight": r(I2, H),
         "midlayer.mlp.up_proj.weight": r(I2, H),
         "midlayer.mlp.down_proj.weight": r(H, I2),
         "norm.weight": (torch.ones(H) + torch.randn(H, generator=g) * 0.05).to(torch.bfloat16),
         "lm_head.weight": r(DVOCAB, H),
         "d2t": torch.randint(0, VOCAB - DVOCAB, (DVOCAB,), generator=g)}
    save_file(w, f"{tmp}/model.safetensors")


def grow(fork, n, gen, base):
    """extend() n committed positions of random (token, aux) pairs starting at absolute pos `base`."""
    toks = torch.randint(0, VOCAB, (n,), generator=gen).tolist()
    auxes = (torch.randn(n, 3, H, generator=gen) * 0.3).to(torch.bfloat16)
    fork.extend(toks, auxes, base_pos=base)
    return toks


def snap(e):
    return (e.ctx_len, e._last_pos, e._last_tok,
            None if e.kbuf is None else e.kbuf[:e.ctx_len].clone())


def check_state(before, e, tag):
    ctx, lp, lt, kb = before
    assert e.ctx_len == ctx and e._last_pos == lp and e._last_tok == lt, f"{tag}: committed state mutated"
    if kb is not None:
        assert torch.equal(e.kbuf[:ctx], kb), f"{tag}: committed KV mutated"


def test_draft_batch_identity():
    tmp = tempfile.mkdtemp()
    make_head(tmp)
    gen = torch.Generator().manual_seed(42)
    embed = (torch.randn(VOCAB, H, generator=gen) * 0.3).to(torch.bfloat16)
    for carry in ("prenorm", "final"):
        base = EagleDrafter(tmp, embed, device="cpu", max_pos=2048, next_hidden=carry)
        forks = [base.fork() for _ in range(5)]
        for f, n in zip(forks, (6, 1, 17, 9, 0)):              # ragged contexts; last fork DEGENERATE (no ctx)
            if n:
                grow(f, n, gen, 0)
        serial = [f._draft(K) for f in forks]
        states = [snap(f) for f in forks]
        batched = draft_batch(forks, K)
        for j, (s, b) in enumerate(zip(serial, batched)):
            assert s == b, f"[{carry}] fork {j}: batched != serial\n  serial ={s}\n  batched={b}"
            check_state(states[j], forks[j], f"[{carry}] fork {j}")
        # multi-round: commit growth between draws (the fill-loop shape), rows must stay identical
        for rnd in range(3):
            for j, f in enumerate(forks):
                grow(f, 1 + (j + rnd) % 4, gen, f._last_pos + 1 if f.ctx_len else 0)
            serial = [f._draft(K) for f in forks]
            batched = draft_batch(forks, K)
            assert serial == batched, f"[{carry}] round {rnd}: batched != serial"
        print(f"  [{carry}] 5 ragged forks (incl. degenerate) x 4 rounds: batched == serial byte-for-byte")


def test_fetch_b_identity():
    tmp = tempfile.mkdtemp()
    make_head(tmp, seed=7)
    gen = torch.Generator().manual_seed(99)
    embed = (torch.randn(VOCAB, H, generator=gen) * 0.3).to(torch.bfloat16)
    base = EagleDrafter(tmp, embed, device="cpu", max_pos=2048)
    pat = [5, 9, 2, 7, 4, 1]
    seqs = [pat * 60,                                          # heavy verbatim repeat -> n-gram HIT
            torch.randint(0, VOCAB, (300,), generator=gen).tolist(),   # novel -> n-gram MISS -> EAGLE
            torch.randint(0, VOCAB, (280,), generator=gen).tolist(),   # novel -> EAGLE
            pat * 55]                                          # n-gram HIT
    drafters = [HybridDrafter(NgramDrafter(ng=3), base.fork()) for _ in seqs]
    drafters.append(NgramDrafter(ng=3))                        # a plain n-gram stream in the same batch
    seqs.append(pat * 50)
    for d, s in zip(drafters, seqs):
        if hasattr(d, "eagle"):
            grow(d.eagle, 12, gen, 0)
        d.request(s, K)
    r1 = fetch_b(drafters)
    m1 = [getattr(d, "matched", None) for d in drafters]
    for d, s in zip(drafters, seqs):                           # re-arm the same pendings, run the serial ref
        d.request(s, K)
    r2 = [d.fetch() for d in drafters]
    m2 = [getattr(d, "matched", None) for d in drafters]
    assert r1 == r2, f"fetch_b != serial fetch\n  fetch_b={r1}\n  serial ={r2}"
    assert m1 == m2, f"matched routing flags diverged: {m1} vs {m2}"
    hits = sum(1 for d in drafters if getattr(d, "matched", False))
    assert 0 < hits < len(drafters), "test must exercise BOTH n-gram-hit and EAGLE-miss rows"
    # a no-context hybrid (prefill not yet extended) must take propose()'s degrade path identically
    d0 = HybridDrafter(NgramDrafter(ng=3), base.fork())
    d0.request(seqs[1], K)
    rb = fetch_b([d0])[0]
    d0.request(seqs[1], K)
    assert rb == d0.fetch(), "degenerate (no-ctx) hybrid diverged"
    print(f"  fetch_b == serial fetch over {len(drafters)} mixed streams ({hits} n-gram hits, "
          f"{len(drafters)-hits} EAGLE/plain) + matched flags + no-ctx degrade")


def bench():
    tmp = tempfile.mkdtemp()
    make_head(tmp, seed=3)
    gen = torch.Generator().manual_seed(1)
    embed = (torch.randn(VOCAB, H, generator=gen) * 0.3).to(torch.bfloat16)
    base = EagleDrafter(tmp, embed, device="cpu", max_pos=2048)
    forks = [base.fork() for _ in range(8)]
    for f in forks:
        grow(f, 64, gen, 0)
    t0 = time.time()
    for _ in range(10):
        for f in forks:
            f._draft(K)
    ts = time.time() - t0
    t0 = time.time()
    for _ in range(10):
        draft_batch(forks, K)
    tb = time.time() - t0
    print(f"  [bench, CPU-indicative] B=8 K={K}: serial {ts*100:.1f}ms/round vs batched {tb*100:.1f}ms/round "
          f"({ts/tb:.1f}x)")


if __name__ == "__main__":
    test_draft_batch_identity()
    test_fetch_b_identity()
    bench()
    print("[draft-batch] ALL PASS — batched chain forward byte-identical to serial per-stream drafting")
