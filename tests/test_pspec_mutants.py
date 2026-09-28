"""Anti-vacuity for the pspec gate: every mutant of the accept rule or the fencing must fail a test.

Each mutant is a one-site source edit of the SHIPPED module (shard/pspec/accept.py or coord.py),
compiled into a fresh module object; the site must occur exactly once, so a mutant can never
silently not apply. It then runs a battery of gate scenarios (lossless vs toy greedy, full accept,
W truncation, chunk divergence, the cpos wire check, the sender fence) over tests/scripted_pipe.py,
and must fail at least one of them on BEHAVIOUR: a NameError/TypeError/SyntaxError-style crash does
not count, because that would be a broken mutant rather than a caught one. The unmutated modules
must pass the whole battery (so a failure is the mutant's, not the harness's).
"""
import inspect
import types

import pytest

from scripted_pipe import (alpha_miss, entry_drafter, run_acked, run_pspec, tail_blocks,
                           toy_greedy, truth)
from test_pspec_coord import sender_fence_case

from shard.pspec import accept as ACC
from shard.pspec import coord as COORD

T = truth(300, seed=77)
P, K = 6, 5


def _mutant(mod, old, new):
    src = inspect.getsource(mod)
    assert src.count(old) == 1, f"mutation site must occur exactly once in {mod.__name__}: {old!r}"
    m = types.ModuleType(mod.__name__ + "_mutant")
    m.__package__, m.__file__ = mod.__package__, mod.__file__
    exec(compile(src.replace(old, new), mod.__file__, "exec"), m.__dict__)
    return m


def _coord_with(accept_mod):
    """A fresh coord module whose `accept` is the (mutated) accept module's."""
    c = _mutant(COORD, "POLICIES = (", "POLICIES = (")
    c.accept = accept_mod.accept
    return c


# ── the battery ────────────────────────────────────────────────────────────────────────────────

def _lossless(cm, fr, policy, src, miss, W=K + 1, **kw):
    s = {"blocks": tail_blocks(T, K, miss)} if src == "tail" else {"drafter": entry_drafter(T, K, miss)}
    r, pipe, _ = run_pspec(T, P, 40, W=W, policy=policy, mod=cm, frontier=fr, timeout=4, **s, **kw)
    assert r["tokens"] == toy_greedy(T, P, 40), "not lossless"
    return r, pipe


def case_lossless_matrix(cm, fr):
    for policy in COORD.POLICIES:
        for src in ("tail", "entry"):
            for miss in (lambda p, d: d == 2, lambda p, d: d >= 3, alpha_miss(0.7, 3),
                         alpha_miss(0.0)):
                _lossless(cm, fr, policy, src, miss)


def case_full_accept(cm, fr):
    for policy in COORD.POLICIES:
        for src in ("tail", "entry"):
            r, _ = _lossless(cm, fr, policy, src, None)
            assert r["cancels"] == 0 and r["max_inflight"] == K + 1, "a perfect drafter cancelled"


def case_W_truncation(cm, fr):
    for policy in COORD.POLICIES:
        for W in (2, 3):
            r, pipe = _lossless(cm, fr, policy, "tail", alpha_miss(0.8, 5), W=W, floor=K)
            assert r["max_inflight"] <= W and all(f["n"] <= W for f in pipe.log[1:])


def case_chunk_divergence_at_every_row(cm, fr):
    """A miss at depth `bad` of every block, including the last row: every chunk is a cancel."""
    for bad in range(1, K + 1):
        r, _ = _lossless(cm, fr, "chunk", "entry", lambda p, d, b=bad: d == b)
        assert r["cancels"] == r["frames"] - 1, f"a miss at row {bad} was not a cancel"


def case_cpos_on_the_wire(cm, fr):
    r, pipe = _lossless(cm, fr, "pertoken", "tail", alpha_miss(0.7, 9))
    stream = T[:P] + r["tokens"]
    for f in pipe.log[1:]:
        assert f["cpos"] <= f["pos"]
        if f["pos"] == f["cpos"]:
            assert f["toks"][0] == stream[f["pos"]]


def case_sender_fence(cm, fr):
    sender_fence_case(cm)


def case_ack_lag(cm, fr):
    """A gateway 3 tokens behind: cpos never passes its ack, re-adoption stays legal."""
    r, pipe, s, built = run_acked(T, P, 40, 3, W=K + 1, blocks=tail_blocks(T, K, alpha_miss(0.7, 4)),
                                  frontier=fr, mod=cm, timeout=4)
    assert r["tokens"] == toy_greedy(T, P, 40), "not lossless"
    assert all(f["cpos"] <= built[f["seq"]] - 1 for f in pipe.log), "cpos passed the gateway ack"
    assert all(st.committed <= s.acked - 1 for st in pipe.st)


BATTERY = [case_lossless_matrix, case_full_accept, case_W_truncation,
           case_chunk_divergence_at_every_row, case_cpos_on_the_wire, case_sender_fence,
           case_ack_lag]
_BROKEN = (NameError, AttributeError, TypeError, SyntaxError, ImportError, UnboundLocalError)


def _run(cm, fr):
    fails = []
    for case in BATTERY:
        try:
            case(cm, fr)
        except _BROKEN as e:
            raise AssertionError(f"{case.__name__}: the mutant is broken, not caught: {e!r}") from e
        except Exception as e:                               # noqa: BLE001 - a caught mutant
            fails.append((case.__name__, f"{type(e).__name__}: {str(e)[:90]}"))
    return fails


def test_the_unmutated_code_passes_the_battery():
    assert _run(COORD, ACC.Frontier) == []


# ── the mutants ────────────────────────────────────────────────────────────────────────────────
# (module, old, new): accept.py mutants run through a coord whose `accept` is the mutant's and a tail
# whose Frontier is the mutant's; coord.py mutants run against the real accept rule.
MUTANTS = {
    # the accept rule
    "accept past the first mismatch (plan_verify_round keeps counting)":
        (ACC, "        if d != r:\n            break\n", "        if d != r:\n            continue\n"),
    "a frame boundary never rejects (next frame's token not compared)":
        (ACC, "return committed, (RUN if int(nxt) == committed[-1] else REJECT)",
         "return committed, RUN"),
    "off-by-one inside a chunk (last-row mismatch missed)":
        (ACC, "    if k < n - 1:\n        return committed, REJECT",
         "    if k < n - 2:\n        return committed, REJECT"),
    "tail frontier ignores the fed token":
        (ACC, "if pos != self.cfront + 1 or fed[0] != self.mfront:", "if pos != self.cfront + 1:"),
    "tail frontier off by one":
        (ACC, "self.cfront, self.mfront = pos + k, committed[-1]",
         "self.cfront, self.mfront = pos + k + 1, committed[-1]"),
    # the frontier / cpos
    "cpos = c (off-by-one frontier watermark)":
        (COORD, "return f if self.acked is None else", "return f + 1 if self.acked is None else"),
    "cpos ignores the gateway ack":
        (COORD, "return f if self.acked is None else min(f, self.acked - 1)", "return f"),
    "block placed one position early":
        (COORD, "                    p = f + 1 + i\n", "                    p = f + i\n"),
    "correction fed one position late":
        (COORD, "            burst.append((f, m))", "            burst.append((f + 1, m))"),
    "in-flight cap W off by one":
        (COORD, "self._cpos() + self.W - 1 - f)", "self._cpos() + self.W - f)"),
    "drafts streamed past the rows the job can use":
        (COORD, "return min(self.max_new - len(self.out) - 1, ", "return min(10 ** 9, "),
    "every frame marked T":
        (COORD, 'kind = "T" if pos + len(toks) - 1 <= f else "S"', 'kind = "T"'),
    "every frame marked S":
        (COORD, 'kind = "T" if pos + len(toks) - 1 <= f else "S"', 'kind = "S"'),
    # the fence
    "stale-sep reply not fenced (judged as live)":
        (COORD, 'if rep["fen"] or sep != self.st.sep:', 'if rep["fen"]:'),
    "stale-sep frame not fenced at the sender":
        (COORD, 'if msg["sep"] != st.sep:', "if False:"),
    "no sep bump on a cancel":
        (COORD, "            self.run = 0\n            self.fence()\n", "            self.run = 0\n"),
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_every_mutant_fails_a_gate_test(name):
    mod, old, new = MUTANTS[name]
    m = _mutant(mod, old, new)
    if mod is ACC:
        fails = _run(_coord_with(m), m.Frontier)
    else:
        fails = _run(m, ACC.Frontier)
    assert fails, f"mutant survived the whole battery: {name}"
    print(f"\n{name}: caught by {[c for c, _ in fails]}; first: {fails[0][1]}")
