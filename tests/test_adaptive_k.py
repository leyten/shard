"""CPU tests for the adaptive round-length policy (phase0/m25_pipe.py _AdaptiveK, M25_ADAPTIVE_K).

WHY: the chain coordinator's round length K is per-FRAME, not per-connection (each verify frame
is [anchor]+ds at a `start`; stages crop KV to `start` and take any s — prefill already ships
s=512 chunks). Acceptance is streaky (survivorship: per-depth accept probability RISES inside an
accepted run), so upsizing K only right after a high-accept round harvests streaks while bust
rounds keep the base-K payload. K changes what is DRAFTED per round, never what is committed —
the ring greedy-verifies every proposal, so losslessness is structural.

These tests pin, WITHOUT CUDA (the policy class + env parse are extracted from the module source
by ast, and the coordinator decode loop is mirrored as a pure-python mock in BOTH shapes):
  A. flag OFF => bit-exact: the patched-shape loop with a disabled policy produces identical
     commits, round counts, wasted counts and frame traces to the master-shape loop across
     K x depth x acceptance-mask scenarios (incl. divergence/discard churn);
  B. flag ON => lossless: the committed stream equals the target greedy stream at EVERY commit;
  C. round economics: strictly fewer rounds on streak-rich masks; never fires on streak-poor
     masks (runs capped below the upsize threshold); bounded on a Markov survivorship mask;
  D. policy transition table, env-constant overrides, and randomized fuzz.

Run: python3 -m pytest tests/test_adaptive_k.py -q   (or: python3 tests/test_adaptive_k.py)
"""
import ast
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(os.path.dirname(HERE), "phase0", "m25_pipe.py")


def load_policy(env=None):
    """exec the real _AdaptiveK class + M25_ADAPTIVE_K parse out of the module source, under a
    controlled env (the module itself needs a GPU context to import). Returns the namespace."""
    old = {}
    env = env or {}
    for k, v in env.items():
        old[k] = os.environ.get(k)
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
    try:
        tree = ast.parse(open(PIPE).read())
        nodes = [n for n in tree.body
                 if (isinstance(n, ast.ClassDef) and n.name == "_AdaptiveK")
                 or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "M25_ADAPTIVE_K")]
        assert len(nodes) == 2, f"expected M25_ADAPTIVE_K assign + _AdaptiveK class, got {len(nodes)} nodes"
        ns = {"os": os}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), PIPE, "exec"), ns)
        return ns
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


BAD = 10 ** 7  # drafter garbage ids: never collide with target vocab


def mock_decode(target, prompt_len, K, depth, max_new, know, policy=None):
    """Mirror of coordinate_pipe's decode loop in both shapes. target: the greedy stream
    (prompt + generation), absolute token per position. know[p]: the drafter proposes target[p]
    at absolute position p, else a garbage id — acceptance then EMERGES through the accept loop
    exactly as on the engine. The mock ring answers each frame at send time with r[j] = target
    argmax after input position start+j. The drafter fetches from the LIVE frontier (the EAGLE
    persistent-context behavior); pending-k re-request on a policy move mirrors the
    d_cancel/d_request hook. policy=None => master loop verbatim; policy=_AdaptiveK => patched."""
    def propose(dp, k):
        return [target[len(dp) + j] if (len(dp) + j < len(target) and know[len(dp) + j])
                else BAD + len(dp) + j for j in range(k)]

    def verify(start, chunk):
        return [target[min(start + j + 1, len(target) - 1)] for j in range(len(chunk))]

    prompt_ids = target[:prompt_len]
    cur = target[prompt_len]                       # prefill's free token
    pos = prompt_len; out = [cur]
    inflight = []; discard = 0; send_pos = pos; dprefix = prompt_ids + [cur]
    valid = wasted = accepted = 0; done = False
    pend_k = policy.next_k() if policy is not None else K
    frames = []
    while not done:
        while len(inflight) < depth and not done:
            if policy is not None and policy.enabled and policy.next_k() != pend_k:
                pend_k = policy.next_k()           # patched hook: d_cancel + d_request at the new k
            ds = propose(dprefix, pend_k)
            frames.append((send_pos, len(ds)))
            r = verify(send_pos, [dprefix[-1]] + ds)
            inflight.append((send_pos, ds, r))
            dprefix = dprefix + ds
            send_pos += (len(ds) if policy is not None else K)
            pend_k = policy.next_k() if policy is not None else K
        sp, ds, r = inflight.pop(0)
        if discard > 0:
            discard -= 1; wasted += 1; continue
        kf = len(ds) if policy is not None else K
        n = 0
        for j in range(kf):
            if ds[j] == r[j]: n += 1
            else: break
        valid += 1; accepted += n
        if policy is not None:
            policy.observe(n, kf)                  # the patched hook site (non-discarded only)
        if n == kf:
            out.extend(ds); pos += kf; cur = ds[-1]
            if not inflight and len(r) > kf:       # full-accept bonus
                out.append(r[kf]); cur = r[kf]
                pos += 1; send_pos += 1; dprefix = dprefix + [r[kf]]
        else:
            committed = ds[:n] + [r[n]]; out.extend(committed); cur = r[n]; pos += n + 1
            discard = len(inflight); dprefix = prompt_ids + out; send_pos = pos
            pend_k = policy.next_k() if policy is not None else K
        assert out == target[prompt_len:prompt_len + len(out)], (
            f"LOSSLESS VIOLATION at pos {pos}: committed diverges from target greedy stream")
        if len(out) >= max_new:
            done = True
    return {"out": out, "rounds": valid, "wasted": wasted, "frames": frames, "accepted": accepted}


# ---- masks (acceptance scenarios over absolute positions) -------------------------------
def mask_runs(n, run_true, run_false):
    """Deterministic long-run mask: streak-rich (verbatim-copy class) when run_true is large."""
    m = []
    while len(m) < n:
        m += [True] * run_true + [False] * run_false
    return m[:n]


def mask_capped(n, cap, p=0.7, seed=1):
    """Streak-poor: i.i.d.-ish knowledge but NO run of True ever reaches `cap` — accepts stay
    below the upsize threshold so the adaptive policy must never fire."""
    rng = random.Random(seed); m = []; run = 0
    for _ in range(n):
        v = rng.random() < p and run < cap - 1
        run = run + 1 if v else 0
        m.append(v)
    return m


def mask_markov(n, p_tt=0.78, p_ft=0.32, seed=2):
    """Novel-text-class mask: 2-state Markov knowledge — persistence creates the survivorship
    signature (per-depth accept probability rising inside accepted runs)."""
    rng = random.Random(seed); m = []; s = False
    for _ in range(n):
        s = rng.random() < (p_tt if s else p_ft)
        m.append(s)
    return m


N = 6000
PL = 40
_rng = random.Random(9)
TARGET = [_rng.randrange(1000, 9000) for _ in range(N)]
SCENARIOS = [("runs", mask_runs(N, 40, 6)), ("capped", mask_capped(N, 5)),
             ("markov", mask_markov(N)), ("none", [False] * N), ("all", [True] * N)]

_ns_off = load_policy({"M25_ADAPTIVE_K": None, "M25_AK_HI": None, "M25_AK_UP": None})
_ns_on = load_policy({"M25_ADAPTIVE_K": "1", "M25_AK_HI": None, "M25_AK_UP": None})
AK_off, AK_on = _ns_off["_AdaptiveK"], _ns_on["_AdaptiveK"]


def test_extraction_and_defaults():
    assert not AK_off(8).enabled, "env unset must disable the policy"
    p = AK_on(8)
    assert p.enabled and p.hi == 16 and p.up == 6, "M25_ADAPTIVE_K=1 defaults hi=16 up=6"


def test_flag_off_bit_exact_vs_master_loop():
    """OFF-path parity: 3 K x 4 depth x 5 masks = 60 decode pairs, comparing commits, rounds,
    wasted counts AND frame traces (start, size) — the deployed flag-OFF surface is master."""
    for K in (4, 8, 16):
        for depth in (1, 2, 4, 8):
            for name, m in SCENARIOS:
                a = mock_decode(TARGET, PL, K, depth, 400, m, policy=None)
                b = mock_decode(TARGET, PL, K, depth, 400, m, policy=AK_off(K, enabled=False))
                assert (a["out"], a["rounds"], a["wasted"], a["frames"]) == \
                       (b["out"], b["rounds"], b["wasted"], b["frames"]), (K, depth, name)


def test_flag_on_lossless():
    """ON-path losslessness (the invariant is also asserted at every commit inside the loop)
    and ON == OFF on the common committed prefix."""
    for depth in (1, 4):
        for name, m in SCENARIOS:
            off = mock_decode(TARGET, PL, 8, depth, 400, m, policy=AK_off(8, enabled=False))
            on = mock_decode(TARGET, PL, 8, depth, 400, m, policy=AK_on(8, enabled=True))
            c = min(len(off["out"]), len(on["out"]))
            assert on["out"][:c] == off["out"][:c], (depth, name)


def test_streak_rich_fewer_rounds():
    """Verbatim-class masks (runs of 40 known): adaptive must use strictly fewer rounds and
    must actually have shipped K=16 frames."""
    for depth in (1, 4):
        off = mock_decode(TARGET, PL, 8, depth, 400, mask_runs(N, 40, 6), policy=AK_off(8, enabled=False))
        on = mock_decode(TARGET, PL, 8, depth, 400, mask_runs(N, 40, 6), policy=AK_on(8, enabled=True))
        assert on["rounds"] < off["rounds"], depth
        assert any(sz > 8 for _, sz in on["frames"]), depth


def test_streak_poor_never_fires():
    """Runs capped at 5 < up=6: the policy must never upsize — trajectory identical to fixed-K
    (bound: rounds within fixed+1%)."""
    for depth in (1, 4):
        off = mock_decode(TARGET, PL, 8, depth, 400, mask_capped(N, 5), policy=AK_off(8, enabled=False))
        on = mock_decode(TARGET, PL, 8, depth, 400, mask_capped(N, 5), policy=AK_on(8, enabled=True))
        assert on["rounds"] <= off["rounds"] * 1.01 + 1, depth
        assert all(sz == 8 for _, sz in on["frames"]), depth


def test_markov_mask_bounded():
    """Novel-text-class Markov survivorship mask: adaptive never loses rounds vs fixed-K."""
    m = mask_markov(N)
    off = mock_decode(TARGET, PL, 8, 1, 400, m, policy=AK_off(8, enabled=False))
    on = mock_decode(TARGET, PL, 8, 1, 400, m, policy=AK_on(8, enabled=True))
    assert on["rounds"] <= off["rounds"] * 1.01 + 1


def test_policy_transitions_env_overrides_and_fuzz():
    pol = AK_on(8, enabled=True)
    assert pol.next_k() == 8, "starts at base"
    pol.observe(8, 8); assert pol.next_k() == 16, "full accept -> hi"
    pol.observe(10, 16); assert pol.next_k() == 16, "n>=up stays hi"
    pol.observe(3, 16); assert pol.next_k() == 8, "bust -> decay to base"
    pol.observe(6, 8); assert pol.next_k() == 16, "n>=up (6) -> hi"
    pol.observe(5, 8); assert pol.next_k() == 8, "n<up -> base"
    d = AK_off(8, enabled=False)
    for n, kf in ((8, 8), (16, 16), (0, 8), (7, 8)):
        d.observe(n, kf)
        assert d.next_k() == 8, "disabled policy must always return base"
    os.environ["M25_AK_HI"] = "12"; os.environ["M25_AK_UP"] = "4"   # __init__ reads env live
    try:
        pe = AK_on(8, enabled=True)
        assert pe.hi == 12 and pe.up == 4 and pe.enabled, "env constants honored"
        pe.observe(4, 8); assert pe.next_k() == 12
    finally:
        os.environ.pop("M25_AK_HI", None); os.environ.pop("M25_AK_UP", None)
    rng = random.Random(31)
    for _ in range(60):
        mm = [rng.random() < rng.choice((0.2, 0.6, 0.9)) for _ in range(N)]
        K = rng.choice((4, 8, 16)); depth = rng.choice((1, 2, 4, 8))
        for pol in (None, AK_off(K, enabled=False), AK_on(K, enabled=True)):
            mock_decode(TARGET, PL, K, depth, rng.randrange(50, 300), mm, policy=pol)
            if pol is not None:
                assert pol.k in (pol.base, pol.hi), "policy state escaped {base, hi}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"RESULT: PASS ({len(fns)} tests)")
