"""CPU tests for correction-token hedged wait-window drafting (M25_CORRHEDGE).

WHY: on novel text the chain coordinator idles ~1 ring RTT per round waiting for the verify
reply, and on a partial accept the verifier hands back exactly one nameable token — the
correction r[n]. The drafter's ranked candidates concentrate real mass on that coordinate
(rank 2-4 of the rejected step), so while the parent frame is in flight the coordinator
pre-drafts K-token continuations from the top-mass (rejection depth j, rank r) correction
candidates and ships them as ordinary verify frames at start = anchor+j (descending start:
crop-to-start stage KV means lower-start writes would corrupt deeper branches), followed by a
RESTORE frame (exact parent copy) that rewrites the parent rows bit-identically. A hit means
the branch's already-in-flight reply IS the next round; a RESEND frame repairs the hit branch's
rows. Every commit stays verifier-greedy on a bit-identical ring prefix, so output byte-identity
with the flag OFF is structural.

Two layers, no ring needed:
  1. REAL DRAFTER MATH (torch, tiny random EAGLE head): draft_topk chain == _draft with rank-1
     == chain and 4 distinct ranks; propose_ahead identity + self-consistency laws; hedge-style
     forced prefixes deterministic; hostile fuzz never mutates the committed cache; the
     HybridDrafter fetch_hedge contract (matched -> no candidates -> no hedges).
  2. COORDINATOR LOOP MOCK (pure python) against a STATEFUL CROP-TO-START RING that computes
     every reply from its own mutated rows — a protocol bug (wrong send order, missing
     RESTORE/RESEND, bad start) produces hostile garbage and the losslessness assert fires:
     flag OFF == master mirror bit-exact; flag ON commits the identical stream on q-profile /
     pure-miss / hit-heavy / matched-streak / all-matched profiles + 40 fuzz trials; hedge-hit
     accounting equals an independent oracle; a miss costs frames only; all-matched sends zero
     hedge frames.

Run: python3 -m pytest tests/test_corrhedge.py -q   (or: python3 tests/test_corrhedge.py)
"""
import ast
import json
import os
import random
import sys
import tempfile

try:
    import pytest
except ImportError:                                   # standalone `python3 tests/test_corrhedge.py`
    pytest = None

if pytest is not None:
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
else:
    import torch
    import safetensors.torch as st

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE0 = os.path.join(os.path.dirname(HERE), "phase0")
sys.path.insert(0, PHASE0)

import importlib.util as _ilu


def _load(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eagle = _load("eagle_hedge_test", os.path.join(PHASE0, "eagle_draft.py"))


def _pipe_hedge_order():
    """Read _HEDGE_ORDER out of phase0/m25_pipe.py by ast (the module needs a GPU to import) —
    the mock loop below must exercise the SAME branch set the coordinator ships."""
    tree = ast.parse(open(os.path.join(PHASE0, "m25_pipe.py")).read())
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_HEDGE_ORDER":
            return tuple(ast.literal_eval(n.value))
    raise AssertionError("_HEDGE_ORDER not found in phase0/m25_pipe.py")


HEDGE_ORDER = _pipe_hedge_order()


def test_hedge_order_is_the_registered_branch_set():
    assert HEDGE_ORDER == ((3, 2), (2, 2), (1, 2), (1, 3), (1, 4))


# ---------------------------------------------------------------------------------------
# Layer 1: tiny fake EAGLE head (random weights, CPU) — validates the REAL drafter math.
# ---------------------------------------------------------------------------------------
def make_fake_head(tmp, H=16, NH=2, NKV=1, HD=8, I=32, DV=64):
    torch.manual_seed(1234)
    cfg = {"hidden_size": H, "num_attention_heads": NH, "num_key_value_heads": NKV,
           "head_dim": HD, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
           "draft_vocab_size": DV}
    json.dump(cfg, open(os.path.join(tmp, "config.json"), "w"))
    r = lambda *s: (torch.randn(*s) * 0.5).to(torch.bfloat16)
    w = {"fc.weight": r(H, 3 * H),
         "midlayer.input_layernorm.weight": r(H).abs() + 0.5,
         "midlayer.hidden_norm.weight": r(H).abs() + 0.5,
         "midlayer.self_attn.q_proj.weight": r(NH * HD, 2 * H),
         "midlayer.self_attn.k_proj.weight": r(NKV * HD, 2 * H),
         "midlayer.self_attn.v_proj.weight": r(NKV * HD, 2 * H),
         "midlayer.self_attn.o_proj.weight": r(H, NH * HD),
         "midlayer.post_attention_layernorm.weight": r(H).abs() + 0.5,
         "midlayer.mlp.gate_proj.weight": r(I, H),
         "midlayer.mlp.up_proj.weight": r(I, H),
         "midlayer.mlp.down_proj.weight": r(H, I),
         "norm.weight": r(H).abs() + 0.5,
         "lm_head.weight": r(DV, H),
         "d2t": torch.zeros(DV, dtype=torch.long)}
    st.save_file(w, os.path.join(tmp, "model.safetensors"))
    torch.manual_seed(99)
    embed = (torch.randn(DV, H) * 0.5).to(torch.bfloat16)
    return embed


def seeded_drafter(n_ctx=12):
    tmp = tempfile.mkdtemp(prefix="hedge_eagle_")
    embed = make_fake_head(tmp)
    d = eagle.EagleDrafter(tmp, embed.clone(), device="cpu", next_hidden="prenorm")
    torch.manual_seed(4321)
    toks = [int(t) for t in torch.randint(0, 64, (n_ctx,))]
    aux = (torch.randn(n_ctx, 3, 16) * 0.5).to(torch.bfloat16)
    d.extend(toks, aux, base_pos=0)
    return d


class _StubNgram:
    """Minimal n-gram stand-in for the HybridDrafter contract test."""

    def __init__(self):
        self.matched = False
        self._p = None

    def request(self, ids, k): self._p = (list(ids), k)

    def fetch(self):
        ids, k = self._p
        return [ids[-1]] * k if self.matched else []

    def cancel(self): self._p = None


def test_draft_topk_matches_chain():
    dp = seeded_drafter()
    for k in (4, 8, 16):
        ds, cands = dp.draft_topk(k)
        assert ds == dp._draft(k), "draft_topk chain must equal _draft"
        assert len(cands) == k
        assert all(c[0] == d for c, d in zip(cands, ds)), "rank-1 candidate must be the chain token"
        assert all(len(c) == 4 for c in cands)
        assert all(len(set(c)) == 4 for c in cands), "ranks distinct (d2t=0 here)"


def test_propose_ahead_identity_and_self_consistency():
    dp = seeded_drafter()
    assert all(dp.propose_ahead([], k) == dp._draft(k) for k in (4, 8, 16))
    for a in (1, 2, 4, 8, 16, 24):
        full = dp._draft(a + 8)
        assert dp.propose_ahead(full[:a], 8) == full[a:], f"self-consistency at a={a}"


def test_hedge_forced_prefixes_deterministic():
    dp = seeded_drafter()
    ds, cands = dp.draft_topk(8)
    for (j, r) in HEDGE_ORDER:
        c = cands[j - 1][r - 1]
        c1 = dp.propose_ahead(ds[:j - 1] + [c], 8)
        c2 = dp.propose_ahead(ds[:j - 1] + [c], 8)
        assert c1 == c2 and len(c1) == 8, (j, r)


def test_fuzz_never_mutates_committed_cache():
    dp = seeded_drafter()
    ref = dp._draft(8)
    ctx_before = dp.ctx_len
    torch.manual_seed(7)
    for _ in range(20):
        spec = [int(t) for t in torch.randint(0, 64, (int(torch.randint(1, 25, (1,))),))]
        assert len(dp.propose_ahead(spec, 8)) == 8
        dp.draft_topk(8)
    assert dp.ctx_len == ctx_before
    assert dp._draft(8) == ref


def test_hybrid_fetch_hedge_contract():
    dp = seeded_drafter()
    ng = _StubNgram()
    hy = eagle.HybridDrafter(ng, dp)
    hy.request(list(range(5)), 8)
    ng.matched = True
    d_m, c_m = hy.fetch_hedge()
    hy.request(list(range(5)), 8)
    ng.matched = False
    d_n, c_n = hy.fetch_hedge()
    ref_ds, ref_c = dp.draft_topk(8)
    assert c_m is None and d_m == [4] * 8, "matched round: (ngram draft, None) -> no hedges"
    assert c_n is not None and d_n == ref_ds and c_n == ref_c, "novel round: eagle.draft_topk"
    assert hy.propose_ahead(ref_ds[:2], 8) == dp.propose_ahead(ref_ds[:2], 8)


# ---------------------------------------------------------------------------------------
# Layer 2: coordinator-loop mock with a STATEFUL crop-to-start ring.
# ---------------------------------------------------------------------------------------
class KVRing:
    """Mock ring with REAL crop-to-start KV semantics: rows are kept; verify(token_ids, start)
    crops rows to `start`, writes the frame, and computes each reply position's greedy argmax
    FROM THE ROWS (not from the frame args) — a corrupted prefix produces hostile garbage, so
    protocol bugs surface as losslessness failures, never silent passes."""

    def __init__(self, target):
        self.target = list(target)
        self.rows = []
        self.frames = 0

    def _pred(self, plen):
        if self.rows[:plen] == self.target[:plen] and plen < len(self.target):
            return self.target[plen]
        return -(abs(hash(("ring", tuple(self.rows[max(0, plen - 8):plen]), plen))) % 100000) - 10

    def verify(self, token_ids, start):
        self.frames += 1
        assert 0 <= start <= len(self.rows), f"KV GAP: start {start} vs rows {len(self.rows)}"
        self.rows = self.rows[:start] + list(token_ids)
        return [self._pred(start + i + 1) for i in range(len(token_ids))]


class MockDrafter:
    """Deterministic drafter with ranked candidates. Per abs position p (given conditioning
    sequence cond): rank-1 correct w.p. acc(p) (hash-gated, cond-dependent = deterministic);
    when rank-1 is wrong, the TRUE target token sits at rank rankfn(p) in {2,3,4} or nowhere.
    matched_fn(p) marks n-gram-matched rounds (no candidate ranking -> no hedges)."""

    def __init__(self, target, acc, rankfn, matched_fn=None):
        self.target = list(target)
        self.acc = acc
        self.rankfn = rankfn
        self.matched_fn = matched_fn or (lambda p: False)

    def _tops(self, cond):
        p = len(cond)
        h = hash(("d", tuple(cond[-6:]), p)) & 0xFFFF
        correct = (h / 65536.0) < self.acc(p) and p < len(self.target)
        tgt = self.target[p] if p < len(self.target) else 0
        g = lambda r: 900000 + r * 10000 + (abs(hash(("g", tuple(cond[-6:]), p, r))) % 9973)
        if correct:
            return [tgt, g(2), g(3), g(4)]
        top = [g(1), g(2), g(3), g(4)]
        rk = self.rankfn(p)
        if rk is not None:
            top[rk - 1] = tgt
        return top

    def draft(self, cond, k):
        cond = list(cond)
        out = []
        for _ in range(k):
            t = self._tops(cond)[0]
            out.append(t); cond.append(t)
        return out

    def draft_topk(self, cond, k):
        cond = list(cond)
        out = []; cands = []
        for _ in range(k):
            top = self._tops(cond)
            out.append(top[0]); cands.append(top); cond.append(top[0])
        return out, cands


T_TRAV = 250.0     # ms, ring traversal (novel-text WAN class)
DRAFT_MS = 2.0     # ms per drafter chain step
REPLY_SP = 6.0     # ms spacing between consecutive replies (bottleneck stage span class)


def run_master(target, P, K, max_new, dr):
    """Master depth-1 EAGLE chain mirror on the stateful ring (bonus rule verbatim)."""
    ring = KVRing(target)
    ring.rows = list(target[:P])                       # prefill rows
    cur = target[P]
    out = [cur]; pos = P; send_pos = P
    rounds = 0; t = 0.0; done = False
    while not done:
        cond = list(target[:P]) + out
        ds = dr.draft(cond, K)
        t += K * DRAFT_MS
        r = ring.verify([cond[-1]] + ds, send_pos)
        t += T_TRAV
        rounds += 1
        n = 0
        while n < K and ds[n] == r[n]: n += 1
        if n == K:
            committed = ds + [r[K]]                    # depth-1: bonus always live
            cur = r[K]; pos += K + 1
        else:
            committed = ds[:n] + [r[n]]; cur = r[n]; pos += n + 1
        assert committed == target[P + len(out):P + len(out) + len(committed)], "LOSSLESS VIOLATION (master)"
        out += committed
        send_pos = pos
        if len(out) >= max_new: done = True
    return out, rounds, ring.frames, t, ring.rows


def run_hedged(target, P, K, max_new, dr, flag):
    """The M25_CORRHEDGE loop faithfully reduced (same branch structure as the coordinator)."""
    ring = KVRing(target)
    ring.rows = list(target[:P])
    cur = target[P]
    out = [cur]; pos = P; send_pos = P
    rounds = wasted = 0; t = 0.0; done = False
    stats = {"rounds_hedged": 0, "branches_sent": 0, "hits": 0, "by_branch": {}, "rounds_matched": 0}
    drain_n = 0
    while not done:
        wasted += drain_n; drain_n = 0                 # stale RESEND reply drained
        cond = list(target[:P]) + out
        if flag:
            if dr.matched_fn(len(cond)):
                ds, cands = dr.draft(cond, K), None
                stats["rounds_matched"] += 1
            else:
                ds, cands = dr.draft_topk(cond, K)
        else:
            ds, cands = dr.draft(cond, K), None
        t += K * DRAFT_MS
        sp0 = send_pos
        r = ring.verify([cond[-1]] + ds, sp0)          # parent frame (processed first)
        branches = []
        hedge_draft_ms = 0.0
        if flag and cands is not None:
            seen = set()
            for bj, br in HEDGE_ORDER:
                if bj > len(ds) or br > len(cands[bj - 1]): continue
                c = cands[bj - 1][br - 1]
                if c == ds[bj - 1] or (bj, c) in seen: continue
                seen.add((bj, c))
                cont = dr.draft(cond + ds[:bj - 1] + [c], K)         # propose_ahead semantics
                hedge_draft_ms += (bj - 1 + K) * DRAFT_MS
                rb = ring.verify([c] + cont, sp0 + bj)               # descending start
                branches.append((bj, br, c, cont, rb))
            if branches:
                ring.verify([cond[-1]] + ds, sp0)                    # RESTORE (reply discarded)
                stats["rounds_hedged"] += 1; stats["branches_sent"] += len(branches)
        t += T_TRAV + max(0.0, hedge_draft_ms - T_TRAV)              # hedge drafting rides the wait window
        rounds += 1
        n = 0
        while n < K and ds[n] == r[n]: n += 1
        if n == K:
            committed = ds + [r[K]]; cur = r[K]; pos += K + 1
        else:
            committed = ds[:n] + [r[n]]; cur = r[n]; pos += n + 1
        assert committed == target[P + len(out):P + len(out) + len(committed)], "LOSSLESS VIOLATION (parent)"
        out += committed
        if len(out) >= max_new: done = True
        hit = None
        if not done and n < K:
            for bi, (bj, br, c, cont, rb) in enumerate(branches):
                if bj == n + 1 and c == r[n]: hit = bi; break
        for bi, (bj, br, c, cont, rb) in enumerate(branches):
            if bi != hit:
                wasted += 1; continue
            t += (bi + 2) * REPLY_SP                                 # branch reply lands just after parent's
            stats["hits"] += 1
            hk = f"{bj},{br}"; stats["by_branch"][hk] = stats["by_branch"].get(hk, 0) + 1
            rounds += 1
            nb = 0
            while nb < K and cont[nb] == rb[nb]: nb += 1
            if nb == K:
                committed = cont + [rb[K]]; cur = rb[K]; pos += K + 1
            else:
                committed = cont[:nb] + [rb[nb]]; cur = rb[nb]; pos += nb + 1
            assert committed == target[P + len(out):P + len(out) + len(committed)], "LOSSLESS VIOLATION (branch)"
            out += committed
            if len(out) >= max_new: done = True
        if branches:
            wasted += 1                                              # RESTORE reply
        if hit is not None and not done:
            bj, br, c, cont, _ = branches[hit]
            ring.verify([c] + cont, sp0 + bj)                        # RESEND repair
            drain_n = 1
        send_pos = pos
    wasted += drain_n
    return out, rounds, wasted, ring.frames, t, ring.rows, stats


def oracle_replay(target, P, K, max_new, dr):
    """Independent hit-accounting oracle: replays outcomes straight off the deterministic
    profile + commit arithmetic — no ring, no KV, no frame machinery."""
    out_len = 1
    hedged = branches_sent = hits = 0; by = {}; rounds = 0; matched = 0
    while out_len < max_new:
        base = P + out_len
        cond = list(target[:base])
        is_matched = dr.matched_fn(base)
        if is_matched:
            ds, cands = dr.draft(cond, K), None
            matched += 1
        else:
            ds, cands = dr.draft_topk(cond, K)
        n = 0
        while n < K and base + n < len(target) and ds[n] == target[base + n]: n += 1
        blist = []
        if cands is not None:
            seen = set()
            for bj, br in HEDGE_ORDER:
                if bj > len(ds) or br > len(cands[bj - 1]): continue
                c = cands[bj - 1][br - 1]
                if c == ds[bj - 1] or (bj, c) in seen: continue
                seen.add((bj, c))
                blist.append((bj, c))
            if blist:
                hedged += 1; branches_sent += len(blist)
        rounds += 1
        out_len += (K + 1) if n == K else (n + 1)
        if out_len >= max_new: break
        hit = None
        if n < K and blist:
            corr = target[base + n]
            for bj, c in blist:
                if bj == n + 1 and c == corr: hit = (bj, c); break
        if hit and not is_matched:
            bj, c = hit
            hits += 1
            for bjo, bro in HEDGE_ORDER:
                if bjo == bj and cands[bj - 1][bro - 1] == c:
                    hk = f"{bj},{bro}"; by[hk] = by.get(hk, 0) + 1; break
            base2 = P + out_len
            cont = dr.draft(list(target[:base2]), K)
            nb = 0
            while nb < K and base2 + nb < len(target) and cont[nb] == target[base2 + nb]: nb += 1
            rounds += 1
            out_len += (K + 1) if nb == K else (nb + 1)
    return {"rounds_hedged": hedged, "branches_sent": branches_sent, "hits": hits,
            "by_branch": by, "rounds_matched": matched, "rounds": rounds}


P2, K2, MAX_NEW = 40, 8, 256
random.seed(11)
TARGET2 = list(range(1, P2 + 1)) + [random.randrange(200, 999) for _ in range(900)]


def _rank_qprofile(p):
    """Measured conditional correction-rank masses given rejection: rank2 0.248, rank3 0.114,
    rank4 0.076, miss 0.562 (from the ring's per-round rank-profile measurement)."""
    u = (hash(("r", p)) & 0xFFFF) / 65536.0
    if u < 0.248: return 2
    if u < 0.362: return 3
    if u < 0.438: return 4
    return None


PROFILES = {
    "qprofile": (lambda p: 0.566, _rank_qprofile, None),
    "puremiss": (lambda p: 0.566, lambda p: None, None),
    "hitheavy": (lambda p: 0.35, lambda p: 2, None),
    "matchedstreak": (lambda p: 0.9, _rank_qprofile, lambda p: (p // 48) % 2 == 0),
    "allmatched": (lambda p: 0.9, _rank_qprofile, lambda p: True),
}


def _runs(name):
    acc, rankfn, mfn = PROFILES[name]
    dr = MockDrafter(TARGET2, acc, rankfn, mfn)
    m = run_master(TARGET2, P2, K2, MAX_NEW, dr)
    o = run_hedged(TARGET2, P2, K2, MAX_NEW, MockDrafter(TARGET2, acc, rankfn, mfn), flag=True)
    return m, o


def test_flag_off_is_master_bit_exact():
    for name in ("qprofile", "puremiss", "hitheavy"):
        acc, rankfn, mfn = PROFILES[name]
        dr = MockDrafter(TARGET2, acc, rankfn, mfn)
        m = run_master(TARGET2, P2, K2, MAX_NEW, dr)
        h = run_hedged(TARGET2, P2, K2, MAX_NEW, dr, flag=False)
        assert m[0] == h[0] and m[1] == h[1] and m[2] == h[3] and m[4] == h[5] and h[2] == 0, name


def test_flag_on_lossless_all_profiles():
    for name in PROFILES:
        m, o = _runs(name)
        cut = min(len(m[0]), len(o[0]))
        assert m[0][:cut] == o[0][:cut], name


def test_hit_accounting_matches_oracle():
    for name in ("qprofile", "hitheavy", "matchedstreak"):
        acc, rankfn, mfn = PROFILES[name]
        o = run_hedged(TARGET2, P2, K2, MAX_NEW, MockDrafter(TARGET2, acc, rankfn, mfn), flag=True)
        orc = oracle_replay(TARGET2, P2, K2, MAX_NEW, MockDrafter(TARGET2, acc, rankfn, mfn))
        keys = ("rounds_hedged", "branches_sent", "hits", "by_branch", "rounds_matched")
        assert {k: o[6][k] for k in keys} == {k: orc[k] for k in keys}, name
        assert o[1] == orc["rounds"], name


def test_pure_miss_costs_frames_only():
    m, o = _runs("puremiss")
    stats = o[6]
    assert m[0] == o[0] and m[1] == o[1] and stats["hits"] == 0
    assert o[2] == stats["branches_sent"] + stats["rounds_hedged"], "wasted == misses + restores"
    assert o[3] == m[2] + stats["branches_sent"] + stats["rounds_hedged"]
    assert abs(m[3] / o[4] - 1.0) < 0.05, "a miss round costs ~nothing on the wall model"


def test_all_matched_sends_zero_hedge_frames():
    m, o = _runs("allmatched")
    assert m[0] == o[0] and m[1] == o[1] and o[3] == m[2] and o[2] == 0
    assert o[6]["rounds_hedged"] == 0 and o[6]["branches_sent"] == 0


def test_wall_model_ev_direction():
    mh, oh = _runs("hitheavy")
    mq, oq = _runs("qprofile")
    assert mh[3] / oh[4] >= 1.15, "hit-heavy must win clearly"
    assert mq[3] / oq[4] >= 0.97, "q-profile class must be parity-or-better"


def test_fuzz_lossless_and_accounting():
    for trial in range(40):
        random.seed(3000 + trial)
        tp = random.randrange(5, 30)
        tgt = list(range(1, tp + 1)) + [random.randrange(200, 999) for _ in range(500)]
        base = random.uniform(0.1, 0.95)
        rmass = sorted(random.uniform(0, 1) for _ in range(3))
        per = random.randrange(24, 90)

        def rf(p, rm=rmass):
            u = (hash(("rf", p)) & 0xFFFF) / 65536.0
            if u < rm[0]: return 2
            if u < rm[1]: return 3
            if u < rm[2]: return 4
            return None

        mf = (lambda p, q=per: (p // q) % 3 == 0) if random.random() < 0.5 else None
        kk = random.choice((4, 8, 16))
        mn = random.randrange(24, 160)
        m = run_master(tgt, tp, kk, mn, MockDrafter(tgt, lambda p, b=base: b, rf, mf))
        o = run_hedged(tgt, tp, kk, mn, MockDrafter(tgt, lambda p, b=base: b, rf, mf), flag=True)
        orc = oracle_replay(tgt, tp, kk, mn, MockDrafter(tgt, lambda p, b=base: b, rf, mf))
        cut = min(len(m[0]), len(o[0]))
        assert m[0][:cut] == o[0][:cut], f"trial {trial}: lossless (K={kk} base={base:.2f})"
        assert o[6]["hits"] == orc["hits"] and o[1] == orc["rounds"], f"trial {trial}: accounting"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"RESULT: PASS ({len(fns)} tests)")
