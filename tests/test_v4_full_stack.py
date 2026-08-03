"""EVERY LEVER IN THE COMPOSED STACK ACTUALLY FIRES — one subprocess per env flag.

WHY THIS FILE EXISTS. M2.5 once banked the conclusion "CUDA graphs don't help" off a ring where a
forgotten env meant every job had run eager: the flag was set on the launcher and never reached the
process that mattered, and nothing in the run could tell the difference between "the lever fired and
did nothing" and "the lever never fired". That is the single most expensive failure mode a stack of
seven opt-in flags has, and it is not a correctness bug — every test passes, the receipts verify, the
number is just meaningless.

So each lever here is armed in a FRESH INTERPRETER and the process is made to report the state it
actually reached. A subprocess is not fussiness: every one of these flags is read at MODULE IMPORT
(`V4_x = os.environ.get(...)` at the top of the file), so monkeypatching the env inside a running
pytest proves nothing about a stage that imported the module ten tests ago. `Stage.__repr__` is the
same surface an operator reads on the live ring, and it reports OBSERVED state — which forward is
bound to the reference class, which overrides are live, how many layers took the bank — so these
assertions and a ring's own logs are the same evidence.

The CUDA-only levers (graphs, the grouped MoE kernel) cannot fire on a CPU box. What is proven here
is the half that a CPU CAN prove and that the ring bug was actually in: the flag is READ, resolved to
the right MODE, and declines LOUDLY with a reason rather than silently staying off.

Run:  OMP_NUM_THREADS=1 python3 -m pytest tests/test_v4_full_stack.py -q
"""
import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE0 = os.path.join(ROOT, "phase0")

pytest.importorskip("torch")

# Every probe here spawns a fresh interpreter -- these flags are read at module import, so
# an in-process monkeypatch would prove nothing about a stage that imported the module first.
#
# SKIPPED ON SHARED CI RUNNERS, and the reason is resource, not logic. Each probe stands up a
# THREE-PROCESS socket ring, every stage importing torch and building its own toy model; several
# probes run back to back. On a 2-core / 7 GB hosted runner a stage gets killed and its peers log
# `forward re-dial 127.0.0.1:PORT still failing (120 tries)` until the probe exits 1 -- the dial is
# REFUSED, not slow, so no timeout bump fixes it. These are the tests that prove the shipped ring
# recipe serves what the default serves, so they must keep running: locally, and on a self-hosted
# runner with the headroom for them. Set V4_FORCE_RING_PROBES=1 to run them on CI anyway.
_CI = os.environ.get("CI") == "true" and os.environ.get("V4_FORCE_RING_PROBES", "0") in ("", "0")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_CI, reason="3-process socket rings per probe exceed a shared CI runner; "
                                   "run locally or set V4_FORCE_RING_PROBES=1"),
]


def run_probe(body, **env):
    """Execute `body` in a fresh interpreter with `env` set, return its stdout. Fails loudly."""
    e = dict(os.environ)
    e.setdefault("OMP_NUM_THREADS", "1")
    e.update({k: str(v) for k, v in env.items()})
    e["PYTHONPATH"] = PHASE0 + os.pathsep + ROOT + os.pathsep + e.get("PYTHONPATH", "")
    src = "import sys\nsys.path.insert(0, %r)\n" % PHASE0 + textwrap.dedent(body)
    p = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=900, env=e)
    assert p.returncode == 0, f"probe failed ({p.returncode})\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}"
    return p.stdout


# ── the flags are READ, and read as the right thing ───────────────────────────────────────────────

@pytest.mark.parametrize("val,mode", [
    (None, "off"), ("0", "off"), ("", "off"),
    ("1", "island"), ("island", "island"), ("on", "island"),
    ("whole", "whole"), ("2", "whole"), ("eager", "whole"),
    ("yes", "off"),          # unknown -> OFF, never a half-armed guess
])
def test_cuda_graph_env_resolves_to_the_documented_mode(val, mode):
    """V4_CUDA_GRAPH is a MODE, not a bool, after the whole-layer merge — and an unrecognised value
    must fall to `off` rather than to some default mode. Read in a fresh process because the module
    binds it at import."""
    env = {} if val is None else {"V4_CUDA_GRAPH": val}
    out = run_probe("""
        import v4_stage
        print("MODE", v4_stage._graph_mode())
    """, **env)
    assert out.strip() == f"MODE {mode}"


def test_every_lever_flag_is_read_at_import():
    """One process, all flags on: each module-level flag must come back TRUE. This is the cheap gate
    on the actual ring bug — a flag typo, a renamed env, a module that reads a different name than
    the launcher sets — and it fails on the name, not on a silent no-op three hours into a run."""
    out = run_probe("""
        import v4_stage, v4_pipe, v4_moe_grouped, v4_ref_slim, v4_dspark_fast, v4_moe_decode
        print("GRAPH", v4_stage._graph_mode())
        print("FAST_VERIFY", v4_stage.V4_FAST_VERIFY, v4_stage.V4_FAST_VERIFY_MAX)
        print("SPEC_DEPTH_ENV", __import__("os").environ["V4_SPEC_DEPTH"])
        print("PIPELINED", v4_pipe.V4_PIPELINED_SPEC)
        print("CONF_GATE", v4_pipe.V4_DSPARK_CONF_GATE)
        print("MOE_GROUPED", v4_moe_grouped.V4_MOE_GROUPED)
        print("MOE_DECODE", v4_moe_decode.V4_MOE_DECODE)
        print("REF_SLIM", v4_ref_slim.V4_REF_SLIM, v4_ref_slim.V4_REF_SLIM_NOQAT)
        print("DSPARK_FAST", v4_dspark_fast.V4_DSPARK_FAST)
    """, V4_CUDA_GRAPH="whole", V4_FAST_VERIFY=1, V4_FAST_VERIFY_MAX=12, V4_SPEC_DEPTH=24,
         V4_PIPELINED_SPEC=1, V4_DSPARK_CONF_GATE=1, V4_MOE_GROUPED=1, V4_MOE_DECODE=1,
         V4_REF_SLIM=1, V4_REF_SLIM_NOQAT=1, V4_DSPARK_FAST=1)
    got = dict(line.split(" ", 1) for line in out.strip().splitlines())
    assert got["GRAPH"] == "whole"
    assert got["FAST_VERIFY"] == "True 12"
    assert got["SPEC_DEPTH_ENV"] == "24"
    assert got["PIPELINED"] == "True"
    assert got["CONF_GATE"] == "True"
    assert got["MOE_GROUPED"] == "True"
    assert got["MOE_DECODE"] == "True"
    assert got["REF_SLIM"] == "True True"
    assert got["DSPARK_FAST"] == "True"


def test_no_lever_is_armed_by_default():
    """The other half of the same bar, and the one that keeps "default = byte-identical to base"
    honest: with a clean env every flag is off and the reference's own methods are still bound."""
    out = run_probe("""
        import v4_stage, v4_pipe, v4_moe_grouped, v4_ref_slim, v4_dspark_fast
        import v4_ref_cpu
        mod = v4_ref_cpu.load_ref()
        print("GRAPH", v4_stage._graph_mode())
        print("FLAGS", v4_stage.V4_FAST_VERIFY, v4_pipe.V4_PIPELINED_SPEC,
              v4_pipe.V4_DSPARK_CONF_GATE, v4_moe_grouped.V4_MOE_GROUPED,
              v4_ref_slim.V4_REF_SLIM, v4_ref_slim.V4_REF_SLIM_NOQAT,
              v4_dspark_fast.V4_DSPARK_FAST)
        print("REBOUND", getattr(mod.Indexer.forward, "_v4_ref_slim", False),
              getattr(mod.act_quant, "_v4_ref_slim", False),
              getattr(mod.MoE.forward, "_v4_grouped", False))
    """, **{k: "" for k in ("V4_CUDA_GRAPH", "V4_FAST_VERIFY", "V4_PIPELINED_SPEC",
                            "V4_DSPARK_CONF_GATE", "V4_MOE_GROUPED", "V4_REF_SLIM",
                            "V4_REF_SLIM_NOQAT", "V4_DSPARK_FAST")})
    assert "GRAPH off" in out
    assert "FLAGS False False False False False False False" in out
    assert "REBOUND False False False" in out


# ── the flags REACH the object that acts on them ──────────────────────────────────────────────────

STAGE_PROBE = """
    import v4_ref_cpu, v4_stage
    args = v4_ref_cpu.cpu_args()
    st = v4_stage.Stage(0, 2, args, head=True, tail=False, device="cpu")
    print("REPR", repr(st))
    print("FAST", st._fast, st._chunk_cap, st._chunk_ok(1), st._chunk_ok(4))
    print("SPEC_DEPTH", st._spec_depth, st._spec_ckpts.maxlen)
    print("GRAPHMODE", st._graph_mode, st._block_graphs is None)
    print("BANKED", st._moe_banked)
"""

# Appended to STAGE_PROBE (same indent — the two are dedented together) to report WHY the grouped
# MoE could not arm on this box, rather than only that it did not.
GROUPED_TAIL = """
    import torch, v4_moe_grouped
    mod = v4_ref_cpu.load_ref()
    moe = st.layers[0].ffn
    e = moe.experts[moe.experts_start_idx]
    print("FLAG", v4_moe_grouped.V4_MOE_GROUPED)
    print("EXPERT_DTYPE_IS_FP4", e.w1.weight.dtype == torch.float4_e2m1fn_x2)
    print("CUDA", torch.cuda.is_available())
    print("INSTALL", v4_moe_grouped.install(mod))
"""


def test_fast_verify_reaches_the_stage_and_never_claims_s_equals_1():
    """V4_FAST_VERIFY arms the chunk path AND the chunk-block class — and `_chunk_ok(1)` is False,
    which is the composition fact the pipelined recipe rests on: pipelining sends only s=1 frames, so
    an armed fast-verify has nothing to claim and the per-token levers keep the position."""
    out = run_probe(STAGE_PROBE, V4_FAST_VERIFY=1, V4_FAST_VERIFY_MAX=6)
    assert "FAST True 6 False True" in out, out
    assert "fast_verify=<=6" in out, "the repr must show the armed cap"

    off = run_probe(STAGE_PROBE, V4_FAST_VERIFY="")
    assert "FAST False 0 False False" in off
    assert "fast_verify=off" in off


def test_spec_depth_reaches_the_rollback_ring():
    """The W-deep rollback ring is what lets pipelining stream W frames before the first is judged;
    its depth must come from the env, not from a default nobody set."""
    out = run_probe(STAGE_PROBE, V4_SPEC_DEPTH=24)
    assert "SPEC_DEPTH 24 24" in out, out
    assert "spec=off/24" in out, "the repr must show the depth even before the stage is armed"


def test_graph_mode_reaches_the_stage_and_declines_loudly_on_cpu():
    """A CUDA-only lever on a CPU box must decline with a REASON on stdout, never silently stay off.
    That distinction is the whole point: `graph=off` with no line above it is indistinguishable from
    a flag that was never set, which is exactly how m25 banked a wrong conclusion."""
    for mode in ("island", "whole"):
        out = run_probe(STAGE_PROBE, V4_CUDA_GRAPH=mode)
        assert f"GRAPHMODE {mode} True" in out, out
        assert "GRAPH REFUSED" in out and "CUDA" in out, "a CPU decline must say why"
        assert "graph=off" in out, "the repr reports what was ARMED, not what was asked for"


def test_ref_slim_reaches_the_reference_and_the_repr_reports_it():
    """Item 1 and item 2 are separately gated; the repr names the ones that are actually live."""
    both = run_probe(STAGE_PROBE, V4_REF_SLIM=1, V4_REF_SLIM_NOQAT=1)
    assert "ref_slim=indexer+noqat" in both, both
    only1 = run_probe(STAGE_PROBE, V4_REF_SLIM=1, V4_REF_SLIM_NOQAT="")
    assert "ref_slim=indexer" in only1 and "noqat" not in only1.split("ref_slim=")[1]
    only2 = run_probe(STAGE_PROBE, V4_REF_SLIM="", V4_REF_SLIM_NOQAT=1)
    assert "ref_slim=noqat" in only2
    off = run_probe(STAGE_PROBE, V4_REF_SLIM="", V4_REF_SLIM_NOQAT="")
    assert "ref_slim=off" in off


def test_grouped_moe_declines_on_this_box_for_the_documented_reason():
    """The grouped MoE is the one lever a CPU box cannot arm, and it must decline for the reason the
    module documents rather than for a reason nobody checked.

    Two independent gates, both real: `install` refuses without CUDA (the kernel is tilelang), and
    `bank_layout`/`_relayout_moe` refuses experts that are not fp4 (the CPU parity model is bf16 and
    has to stay byte-identical). So on this box `V4_MOE_GROUPED=1` legitimately yields `moe=decode`
    and `BANKED 0` — and the test asserts the CAUSE, not just the outcome, because "declined because
    it is bf16 on a CPU" and "declined because the flag never arrived" look identical otherwise. The
    layout mechanism itself is proven against an fp4 stand-in in tests/test_v4_moe_grouped.py; the
    ON-A-REAL-CARD half is `moe=grouped/N` in the repr, which is what made the ring failure visible
    (the lever read as on for every job while `N` was 0)."""
    out = run_probe(STAGE_PROBE + GROUPED_TAIL, V4_MOE_GROUPED=1, V4_MOE_DECODE=1)
    assert "FLAG True" in out, "the flag must be READ even where the lever cannot arm"
    assert "EXPERT_DTYPE_IS_FP4 False" in out, "this box's decline reason is the bf16 parity model"
    assert "CUDA False" in out and "INSTALL False" in out, "no CUDA -> the kernel install declines"
    assert "BANKED 0" in out and "moe=decode" in out, out
    assert "grouped-MoE bank layout" not in out, "a decline must not announce a layout it did not do"


def test_dspark_fast_rebinds_the_tail_advance():
    """V4_DSPARK_FAST collapses the drafter's wasted intermediate forwards; prove the rebind lands on
    DSparkTail rather than being reported by a flag nobody consumed."""
    out = run_probe("""
        import v4_dspark_draft as D, v4_dspark_fast as F
        took = F.install(D)
        print("TOOK", took, getattr(D.DSparkTail.advance_and_draft, "_v4_dspark_fast", False))
        print("IDEMPOTENT", F.install(D))
    """, V4_DSPARK_FAST=1)
    assert "TOOK True True" in out, out
    assert "IDEMPOTENT False" in out, "a second install must not chain the fast path onto itself"

    off = run_probe("""
        import v4_dspark_draft as D, v4_dspark_fast as F
        print("TOOK", F.install(D), getattr(D.DSparkTail.advance_and_draft, "_v4_dspark_fast", False))
    """, V4_DSPARK_FAST="")
    assert "TOOK False False" in off


def test_pipelined_flag_reaches_the_coordinator_choice():
    """The serve loop picks the coordinator from V4_PIPELINED_SPEC or the job's own `pipelined`; both
    inputs must reach the same branch, and the default must stay serial."""
    out = run_probe("""
        import v4_pipe
        print("ENV", v4_pipe.V4_PIPELINED_SPEC)
        print("HAS", callable(v4_pipe.coordinate_dspark_pipelined))
    """, V4_PIPELINED_SPEC=1)
    assert "ENV True" in out and "HAS True" in out
    off = run_probe("import v4_pipe; print('ENV', v4_pipe.V4_PIPELINED_SPEC)", V4_PIPELINED_SPEC="")
    assert "ENV False" in off


# ── the one combination that is REFUSED rather than silently ignored ──────────────────────────────

def test_conf_gate_with_pipelining_is_refused_not_dropped():
    """The confidence gate trims the tail's OFFERED BLOCK LENGTH. The pipelined coordinator streams
    one s=1 frame per position and never sends a block, so there is nothing for the gate to trim.
    Passing both must RAISE — a gate that silently did nothing would be measured on the ring as
    "confidence gating did not help", which is the same lie as a graph that never captured."""
    out = run_probe("""
        import inspect, v4_pipe
        src = inspect.getsource(v4_pipe._coord_cli)
        assert "wants_conf" in src, "the serve loop no longer reconciles the two -- recheck"
        assert "raise ValueError" in src
        # the serial coordinator takes the knobs, the pipelined one does not even accept them
        assert "conf_gate" in inspect.signature(v4_pipe.coordinate_dspark).parameters
        assert "conf_gate" not in inspect.signature(v4_pipe.coordinate_dspark_pipelined).parameters
        print("REFUSED OK")
    """)
    assert "REFUSED OK" in out


def test_fast_verify_is_bypassed_by_every_s_equals_1_frame():
    """The composition rule for the ring recipe, asserted on the dispatch itself rather than trusted.

    Stage.forward tests `start_pos == 0 or s == 1` BEFORE `_chunk_ok(s)`, so a pipelined frame (always
    s=1) takes `_run` — where the graphs, the grouped MoE and ref-slim live — and can never take the
    chunk path. Fast-verify is therefore not additive with the per-token levers; it is the alternative
    to them, and only the serial coordinator can reach it."""
    out = run_probe("""
        import ast, inspect, textwrap, v4_stage
        tree = ast.parse(textwrap.dedent(inspect.getsource(v4_stage.Stage.forward)))
        # Find the if/elif chain that dispatches the pass, and read its arms IN ORDER.
        chain = None
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and "_chunk_ok" in ast.dump(node):
                chain = node
                break
        assert chain is not None, "Stage.forward no longer dispatches on _chunk_ok"
        arms = []
        n = chain
        while isinstance(n, ast.If):
            arms.append(ast.dump(n.test))
            n = n.orelse[0] if len(n.orelse) == 1 and isinstance(n.orelse[0], ast.If) else None
        assert "start_pos" in arms[0] and "'s'" in arms[0], arms
        assert "_chunk_ok" in arms[1], arms
        assert not any("_chunk_ok" in a for a in arms[:1]), "the chunk path moved ahead of s==1"
        print("ARMS", len(arms))
        print("ORDER OK")
    """)
    assert "ORDER OK" in out and "ARMS 2" in out


# ── the levers do not change WHAT THE RING SERVES ────────────────────────────────────────────────

TOKENS = __import__("re").compile(r"^\s*tokens \((\w+)\)\s+(\[[0-9, ]+\])\s*$", __import__("re").M)


def selftest_streams(**env):
    """Run the offline CPU ring selftest under `env`; return {path: token list} and require ALL PASS."""
    out = run_probe("""
        import v4_pipe
        v4_pipe.selftest()
    """, **env)
    assert "ALL PASS" in out, f"the selftest itself failed under {env}\n{out}"
    streams = dict(TOKENS.findall(out))
    assert streams, f"no token lines parsed\n{out}"
    return streams


# The ring launch recipe, as documented in docs/V4_FULL_STACK.md. If this list changes, the doc and
# this test change together — that is the point of pinning it in a test rather than only in prose.
RING_RECIPE = dict(V4_PIPELINED_SPEC=1, V4_CUDA_GRAPH="whole", V4_DSPARK_FAST=1,
                   V4_MOE_GROUPED=1, V4_MOE_DECODE=1, V4_REF_SLIM=1)


def test_the_ring_recipe_serves_exactly_what_the_default_serves():
    """THE BAR THIS BRANCH IS FOR. Every lever in the launch recipe on at once must emit the SAME
    tokens as the all-off default, on every path the selftest drives.

    "ALL PASS" is NOT this bar and cannot be. The selftest's own assertions compare each config's
    ring against THAT CONFIG'S reference, so a lever that moved the ring and the reference together
    passes in-config while changing what the ring actually serves. Only a cross-config comparison
    catches that, and it did: it is what found V4_REF_SLIM_NOQAT below."""
    base = selftest_streams()
    lever = selftest_streams(**RING_RECIPE)
    common = sorted(set(base) & set(lever))
    assert set(common) >= {"ring", "ref", "spec", "dspark", "pipe"}, common
    for path in common:
        assert base[path] == lever[path], (
            f"the ring recipe changed the {path} stream\n  default: {base[path]}\n  recipe:  {lever[path]}")


def test_noqat_is_the_one_lever_that_changes_the_stream_and_is_therefore_not_in_the_recipe():
    """V4_REF_SLIM_NOQAT removes a deliberate PRECISION REDUCTION (the reference quantizes KV/Q to
    fp8/fp4 and dequantizes straight back, to simulate an fp8 KV deployment). Dropping it makes the
    run strictly MORE precise than the reference — which is still a different answer, and on the toy
    config it moves tokens from step 3 on.

    It is documented APPROXIMATE in v4_ref_slim's module docstring, and the selftest passes with it
    on because the ring and its reference move together. So it must stay OUT of any run whose claim
    is "bit-identical to greedy", which is the headline claim of the pipelined arm. Asserting the
    divergence rather than the equality keeps it that way: if a future change makes NOQAT lossless
    this test fails and someone has to decide deliberately, instead of it drifting into the recipe."""
    assert "V4_REF_SLIM_NOQAT" not in RING_RECIPE, "NOQAT is not lossless — keep it out of the recipe"
    base = selftest_streams()
    noqat = selftest_streams(V4_PIPELINED_SPEC=1, V4_REF_SLIM=1, V4_REF_SLIM_NOQAT=1)
    assert base["ref"] != noqat["ref"], (
        "NOQAT no longer changes the reference stream — re-derive whether it is now lossless "
        "before letting it into the recipe")
    # and it moves every path together, which is why the in-config selftest cannot see it
    assert noqat["ref"] == noqat["ring"] == noqat["pipe"], noqat


# ── where the two biggest levers actually touch ───────────────────────────────────────────────────

def test_a_pipelined_rollback_rebuilds_state_eager_under_graphs():
    """THE COMPOSITION THAT BINDS PIPELINING TO THE GRAPHS' TIER-1 BAR. Pin it, because nothing else
    states it and it is invisible on a CPU box.

    A pipelined cancel calls `Stage._replay`, which re-feeds the accepted prefix to rebuild the window
    ring and both compressor accumulators. `_replay` sets `_replaying = True`, and `_run`'s graph gate
    excludes it — so the rollback rebuilds that state through the EAGER path, while the forward frames
    that originally wrote it went through the GRAPH.

    Therefore: if graphed and eager differ by even one bit, every cancel leaves the stage holding
    state the un-cancelled run would not have had, and the divergence is silent, behind valid
    receipts. Pipelining cancels often — the CPU selftest shows 3 cancels in 4 cycles — so this is the
    common path, not a corner.

    That makes the whole-layer graph's Tier-1 bar (`graphed == its eager twin`, torch.equal, incl.
    across bucket crossings) not a quality nicety but the CORRECTNESS PRECONDITION for running
    V4_CUDA_GRAPH with V4_PIPELINED_SPEC. Tier 2 (vs the vendored reference, tie-bounded) is NOT
    sufficient here: an approximate-but-defensible graph would still poison every rollback.

    Follow-up worth taking, deliberately NOT taken on this branch: `_replay` is s=1 per position and
    throws its outputs away, so it could use the graph instead — which would be faster AND would
    dissolve this dependency entirely. That is a behaviour change to a separately verified module and
    belongs in its own change, measured on its own."""
    out = run_probe("""
        import inspect, textwrap, v4_stage
        rep = textwrap.dedent(inspect.getsource(v4_stage.Stage._replay))
        run = textwrap.dedent(inspect.getsource(v4_stage.Stage._run))
        assert "self._replaying = True" in rep, "the replay no longer flags itself"
        assert "not self._replaying" in run, "the graph gate no longer excludes a replay"
        # and the replay drives _run per POSITION, so each call is the s=1 shape a graph would take
        assert "self._run(h[:, i:i + 1]" in rep, rep
        print("BOUND OK")
    """)
    assert "BOUND OK" in out


def test_the_drafter_moe_lever_serves_exactly_what_the_default_serves():
    """V4_MOE_MULTI on top of the whole recipe must emit the SAME tokens as the all-off default.

    The lever claims the one MoE shape no other lever claims — the DSpark drafter's block — so it
    only ever changes bytes on the DRAFTED and PIPELINED paths, which is precisely where a wrong
    draft is least visible: acceptance falls, nothing raises, and the receipts still settle. A
    cross-config comparison is the only thing that catches that (test_the_ring_recipe_serves_exactly
    _what_the_default_serves's own reasoning), so it gets one.

    It is NOT in RING_RECIPE. Bit-exact is necessary but not sufficient for a launch recipe: this
    lever has never been measured on a card, and an unmeasured lever in the documented recipe is how
    a bench reports a win it did not get. It goes in when the tail measures it."""
    base = selftest_streams()
    lever = selftest_streams(**RING_RECIPE, V4_MOE_MULTI=1)
    common = sorted(set(base) & set(lever))
    assert set(common) >= {"ring", "ref", "spec", "dspark", "pipe"}, common
    for path in common:
        assert base[path] == lever[path], (
            f"the drafter MoE lever changed the {path} stream\n  default: {base[path]}\n"
            f"  lever:   {lever[path]}")


def test_the_wide_drafter_serves_exactly_what_the_default_serves():
    """V4_DSPARK_BLOCK=7 over the whole recipe must emit the SAME tokens as the all-off default, on
    EVERY path the selftest drives — greedy, spec, serial dspark, pipelined, lazy at three depths,
    and the refill floor at three settings, all over real sockets against the same reference stream.

    This is the lever's entire correctness claim in one comparison: width moves only what is
    SPECULATED. A wider block perturbs every draft (the block attends to itself bidirectionally),
    so acceptance may move at every depth — but a draft is only ever COMMITTED when the ring's own
    reply equals it, so the served stream cannot. Like V4_MOE_MULTI above, it is deliberately NOT
    in RING_RECIPE: bit-exact is proven here, the per-depth acceptance under widening is a ring
    measurement (`accept_by_depth` at width 7+ vs the trained width), and an unmeasured lever has
    no place in the documented recipe."""
    base = selftest_streams()
    wide = selftest_streams(**RING_RECIPE, V4_DSPARK_BLOCK=7)
    common = sorted(set(base) & set(wide))
    assert set(common) >= {"ring", "ref", "spec", "dspark", "pipe"}, common
    for path in common:
        assert base[path] == wide[path], (
            f"the wide drafter changed the {path} stream\n  default: {base[path]}\n"
            f"  wide:    {wide[path]}")


def test_the_wide_drafter_lifts_the_pipelined_cap_on_a_real_ring():
    """The cap the MULTIBLOCK verdict pinned at block+1, demonstrably LIFTED on a real localhost
    socket ring: the tiny checkpoint's trained width is 3 (cap 4), and with V4_DSPARK_BLOCK=7 the
    same ring holds 8 frames in flight — at zero acceptance (random weights), so the depth comes
    from the streamed block alone, every cycle rewinds through the stages' checkpoint rings, and
    the emitted stream still equals the vendored reference's greedy decode bit for bit. Floors 1
    and 7 both serve the reference stream; the identity between them is the zero-accept floor
    identity the selftest already pins, now at the widened cap."""
    out = run_probe("""
        import json, tempfile, torch, v4_pipe, v4_ref_cpu as R
        from v4_pipe import (_write_tiny_checkpoint, _reference_tokens, _dspark_tiling,
                             _spawn_ring, coordinate_dspark_pipelined, send_msg)
        args = R.cpu_args()
        d = tempfile.mkdtemp(prefix="v4wide_")
        model = R.build_oracle(args)
        _write_tiny_checkpoint(d, args, model)
        import os
        os.environ["V4_DIR"] = d
        prompt, max_new = [168, 15, 493, 72, 22], 6
        ref = _reference_tokens(model, prompt, max_new)
        ranges = _dspark_tiling(args, 3)
        pipe, ret = _spawn_ring(d, ranges, 1, dspark=True, tag="w")
        outs = {f: coordinate_dspark_pipelined(pipe, ret, prompt, max_new, nonce=f"wide-{f}",
                                               receipts=True, layer_count=args.n_layers,
                                               timeout=120, depth=16, floor=f)
                for f in (1, 7)}
        send_msg(pipe, {"op": "stop"})
        print("REF", json.dumps(ref))
        for f, r in outs.items():
            print("RUN", json.dumps({"floor": f, "tokens": r["tokens"],
                                     "max_inflight": r["max_inflight"],
                                     "block_len": r["block_len"],
                                     "inflight_time_avg": r["inflight_time_avg"],
                                     "frames_per_token": r["frames_per_token"],
                                     "receipts_ok": r["receipts_ok"]}))
    """, V4_DSPARK_BLOCK=7)
    import json as _json
    ref = _json.loads(next(l[4:] for l in out.splitlines() if l.startswith("REF ")))
    runs = [_json.loads(line[4:]) for line in out.splitlines() if line.startswith("RUN ")]
    assert len(runs) == 2
    for r in runs:
        assert r["tokens"] == ref, f"floor={r['floor']}: the widened ring left the greedy stream"
        assert r["block_len"] == 7, "the tail did not draft at the widened width"
        assert r["max_inflight"] == 8, (
            f"width 7 must hold 8 frames in flight on the real ring (trained width 3 holds 4), "
            f"got {r['max_inflight']}")
        assert 0 < r["inflight_time_avg"] <= 8 and r["frames_per_token"] >= 1.0
        assert r["receipts_ok"] is True
