"""A LEVER'S ENV VAR AND ITS LIVE STATE CANNOT DISAGREE QUIETLY — the gate for the whole class.

tests/test_v4_full_stack.py already proves each flag is READ and REACHES the object that acts on it.
That was not enough, and the sixth instance of this bug is why: on a real 6-box ring every stage
carried `V4_MOE_GROUPED=1 V4_MOE_MULTI=1`, every stage's repr said `moe=ref`, and BOTH were true —
the levers were installed and serving, and the status function classified only the two markers it
knew, so the third (`multi`, sitting on top as the chain's head) fell through to the literal string
"ref". A night went into chasing a lever that was already on.

So this file gates the INSTRUMENT, not the lever:

  * the registry is TOTAL against the source. Every `os.environ.get("V4_...")` anywhere in the
    engine's modules is either a registered lever or an explicitly listed knob. Add a lever without
    registering it and this fails — the registry cannot silently fall behind the code, which is how
    five of the six instances survived a green suite.
  * the observation is of LIVE STATE. Break an install deliberately and the audit must say MISMATCH;
    break only the MARKER, leaving the lever installed and serving, and it must still say MISMATCH,
    because an instrument that cannot see a link is exactly what cost the night.
  * V4_LEVERS_STRICT REFUSES. A guard that cannot be shown to fire is not a guard.
  * the side table is enforced. V4_LAZY_DRAFT set on a stage is reported WRONG PROCESS instead of
    doing nothing for six hours.

Subprocesses everywhere for the same reason test_v4_full_stack uses them: every flag is read at
MODULE IMPORT, so an in-process monkeypatch proves nothing about a stage that imported the module
first.

Run:  OMP_NUM_THREADS=1 python3 -m pytest tests/test_v4_levers.py -q
"""
import glob
import os
import re
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE0 = os.path.join(ROOT, "phase0")
sys.path.insert(0, PHASE0)

pytest.importorskip("torch")
VL = pytest.importorskip("v4_levers")
VP = pytest.importorskip("v4_pipe")


def run_probe(body, expect_fail=False, **env):
    """Execute `body` in a fresh interpreter with `env` set; return stdout+stderr merged.

    Merged because the audit writes to STDERR by design (a stage's log takes both; the coordinator's
    stdout is the SHARD_JOB_* contract), so a test that only read stdout would be blind to the thing
    it is here to check."""
    e = dict(os.environ)
    e.setdefault("OMP_NUM_THREADS", "1")
    for k in list(e):
        if k.startswith("V4_"):
            del e[k]                                   # a lever leaking in from the runner is a lie
    e.update({k: str(v) for k, v in env.items()})
    e["PYTHONPATH"] = PHASE0 + os.pathsep + ROOT + os.pathsep + e.get("PYTHONPATH", "")
    src = "import sys\nsys.path.insert(0, %r)\n" % PHASE0 + textwrap.dedent(body)
    p = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=900, env=e)
    ok = (p.returncode != 0) if expect_fail else (p.returncode == 0)
    assert ok, (f"probe returncode {p.returncode} (expect_fail={expect_fail})\n"
                f"--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}")
    return p.stdout + p.stderr


# ── the registry cannot fall behind the code ──────────────────────────────────────────────────────

def test_the_registry_is_total_against_the_engine_source():
    """EVERY V4_* env name the engine reads is registered, derived from the source rather than listed.

    This is the test that makes the registry trustworthy. A new `os.environ.get("V4_NEW_LEVER")` in
    any engine module fails here until it appears in LEVERS (with a live observation) or in
    NON_LEVER_ENV (with a reason it has nothing to observe). Nobody has to remember."""
    scraped = VL.env_names_in_source()
    assert "V4_MOE_MULTI" in scraped and "V4_LAZY_DRAFT" in scraped, \
        f"the scraper stopped seeing levers: {sorted(scraped)}"
    unregistered = scraped - set(VL.LEVERS_BY_ENV) - set(VL.NON_LEVER_ENV)
    assert not unregistered, (
        f"{sorted(unregistered)} are read by the engine and are in no registry. Add each to "
        f"v4_levers.LEVERS with a check that inspects LIVE state, or to NON_LEVER_ENV with a reason.")


def test_no_registry_entry_is_a_dead_string():
    """The other direction: a registered name nothing reads is an instrument pointed at nothing —
    it would report `requested=off observed=off OK` forever and look like coverage."""
    scraped = VL.env_names_in_source()
    dead = (set(VL.LEVERS_BY_ENV) | set(VL.NON_LEVER_ENV)) - scraped
    assert not dead, f"registered but read by nothing in the engine: {sorted(dead)}"


def test_every_lever_names_the_module_that_actually_reads_it():
    """`owner` is what an operator is told to look at when a lever misbehaves, so it has to be the
    module that really parses the var — checked against the source, not against the docstring."""
    for lv in VL.LEVERS:
        src = open(os.path.join(PHASE0, lv.owner + ".py")).read()
        assert re.search(r"""environ(?:\.get)?[.(\[]+["']%s["']""" % lv.env, src), \
            f"{lv.env} is registered to {lv.owner}, which does not read it"


def test_engine_modules_accounts_for_every_v4_module_on_disk():
    """The scrape is only total if the module list is. A new phase0/v4_*.py must be classified as
    engine (scraped, and shipped to a box) or as a bench helper, deliberately."""
    on_disk = {os.path.basename(p) for p in glob.glob(os.path.join(PHASE0, "v4_*.py"))}
    bench_only = {"v4_whole_layer_bench.py", "v4_wire_bench.py"}
    unclassified = on_disk - set(VL.ENGINE_MODULES) - bench_only
    assert not unclassified, (
        f"{sorted(unclassified)} is neither in v4_levers.ENGINE_MODULES (scraped for levers, and the "
        f"list of files a stage box needs) nor a known bench helper.")


def test_every_stage_side_lever_is_routed_to_a_launched_stage():
    """The side table and the propagation list have to agree: a lever a STAGE reads must be one
    stage_launch_cmd sends. (Coordinator-side levers ride ENG_ENV too, deliberately — v4_pipe runs on
    both sides and an operator exporting one on the launcher should get it wherever it is read.)"""
    emitted = {"V4_DIR", "V4_DEV", "V4_CUDA_GRAPH"}
    for lv in VL.LEVERS:
        if lv.side in (VL.STAGE, VL.BOTH):
            assert lv.env in VP.ENG_ENV or lv.env in emitted, (
                f"{lv.env} is read inside a stage process but stage_launch_cmd never sends it — "
                f"exporting it on the launcher configures nothing.")


# ── the regression: a chain is reported whole, not just its head ──────────────────────────────────

REPR_PROBE = """
    import v4_ref_cpu, v4_stage
    args = v4_ref_cpu.cpu_args()
    st = v4_stage.Stage(0, 2, args, head=True, tail=False, device="cpu")
    print("REPR", repr(st))
    print("CHAIN", ">".join(__import__("v4_levers").moe_chain(v4_stage.ref())))
"""


def test_a_multi_install_no_longer_erases_the_levers_underneath_it():
    """THE BUG, pinned. `V4_MOE_MULTI=1` puts `multi_forward` on top of the chain; it carries neither
    of the two markers the old status function knew, so the repr printed `moe=ref` while the decode
    fast path (and, on a card, the grouped kernel) were installed and serving underneath.

    On this CPU box the grouped kernel legitimately declines (no CUDA), so the live chain is
    multi -> decode -> ref, and every link must be visible. `moe=ref` for this env is the exact
    string that cost a night, so it is asserted ABSENT rather than merely unequal."""
    out = run_probe(REPR_PROBE, V4_MOE_MULTI=1, V4_MOE_DECODE=1)
    assert "CHAIN multi>decode>ref" in out, out
    assert "moe=multi>decode>ref" in out, out
    assert "moe=ref " not in out, "the head of the chain is erasing the levers under it again"


def test_the_chain_reports_the_reference_when_nothing_is_installed():
    """The same instrument has to be able to say `ref` and mean it — a status that can only ever
    report a lever is not an observation either."""
    out = run_probe(REPR_PROBE, V4_MOE_MULTI="", V4_MOE_DECODE="0")
    assert "CHAIN ref" in out and "moe=ref " in out, out


def test_the_repr_carries_the_audit_verdict():
    """Two of these six bugs were caught only because a repr happened to show observed state. Make it
    the rule: the line an operator already reads carries the verdict, so `requested` disagreeing with
    `observed` is visible without anyone thinking to look."""
    clean = run_probe(REPR_PROBE, V4_MOE_DECODE=1)
    assert "levers=ok" in clean, clean
    # grouped requested on a box with no CUDA: a true finding, and it must reach the repr
    dirty = run_probe(REPR_PROBE, V4_MOE_GROUPED=1)
    assert "levers=!V4_MOE_GROUPED:mismatch" in dirty, dirty


# ── ADVERSARIAL: break an install, and a marker, and require the guard to fire ────────────────────

BREAK_INSTALL = """
    # The deploy that silently did not take: the flag is set, the module is imported, and its
    # install is a no-op. This is instance 1 and 4 of the bug in one line.
    import v4_moe_decode
    v4_moe_decode.install = lambda mod: False
    import v4_ref_cpu, v4_levers
    v4_ref_cpu.load_ref()
    print(v4_levers.report(side=v4_levers.STAGE))
"""

BREAK_MARKER = """
    # The instrument that cannot see a link: the lever IS installed and IS serving, but the marker
    # the audit classifies by is gone. This is instance 6 — the one the whole file is named after —
    # and it must be caught as loudly as a lever that never installed.
    import v4_ref_cpu, v4_levers
    mod = v4_ref_cpu.load_ref()
    assert mod.MoE.forward.__module__ == "v4_moe_decode", "the fast path did not install at all"
    del mod.MoE.forward._v4_decode_fast
    print(v4_levers.report(side=v4_levers.STAGE))
"""


@pytest.mark.parametrize("probe,label", [(BREAK_INSTALL, "install"), (BREAK_MARKER, "marker")])
def test_a_requested_lever_that_did_not_take_is_caught(probe, label):
    """THE GUARD, SHOWN FIRING. V4_MOE_DECODE is requested (it is on by default), the install is
    sabotaged, and the audit must report the disagreement rather than the request."""
    out = run_probe(probe, V4_MOE_DECODE=1)
    assert "V4_MOE_DECODE" in out and "MISMATCH" in out, out
    assert "requested=on" in out and "observed=off" in out, out
    assert "1 PROBLEM(S)" in out, out


@pytest.mark.parametrize("probe,label", [(BREAK_INSTALL, "install"), (BREAK_MARKER, "marker")])
def test_strict_refuses_to_serve_a_lever_that_did_not_take(probe, label):
    """And under V4_LEVERS_STRICT it is not a log line, it is a refusal — the process dies before it
    can produce a number about a configuration nobody asked for."""
    out = run_probe(probe, expect_fail=True, V4_MOE_DECODE=1, V4_LEVERS_STRICT=1)
    assert "V4_LEVERS_STRICT: refusing to serve" in out, out


def test_a_clean_process_passes_the_same_gate():
    """The control. Without the sabotage the identical probe is clean under STRICT, so the two tests
    above prove the guard fires on the fault and not on the weather."""
    out = run_probe("""
        import v4_ref_cpu, v4_levers
        v4_ref_cpu.load_ref()
        print(v4_levers.report(side=v4_levers.STAGE))
    """, V4_MOE_DECODE=1, V4_LEVERS_STRICT=1)
    assert "all " in out and "clean" in out, out


# ── the side table: which process must carry which var ────────────────────────────────────────────

def test_a_coordinator_lever_set_on_a_stage_is_reported_as_the_wrong_process():
    """INSTANCE 5, pinned. V4_LAZY_DRAFT is read by the coordinator loop and by nothing in a stage.
    It was set on six stages for hours and did exactly nothing, with no signal anywhere."""
    out = run_probe("""
        import v4_pipe, v4_levers
        print(v4_levers.report(side=v4_levers.STAGE))
    """, V4_LAZY_DRAFT=1)
    assert "V4_LAZY_DRAFT" in out and "WRONG PROCESS" in out, out
    assert "coordinator-side lever set on a stage" in out, out


def test_a_stage_lever_set_on_a_coordinator_is_reported_too():
    """Symmetric, and not hypothetical: a coordinator that exports a stage lever it never launched
    stages with has configured nothing either."""
    out = run_probe("""
        import v4_pipe, v4_levers
        print(v4_levers.report(side=v4_levers.COORDINATOR))
    """, V4_FAST_VERIFY=1)
    assert "V4_FAST_VERIFY" in out and "WRONG PROCESS" in out, out


def test_a_v4_var_nothing_reads_is_reported_unknown():
    """V4_TOPK_STABLE is written up in docs/V4_FLASH_ENGINE.md as "the real acceptance fix" and is
    read by NOTHING in phase0 on any branch — setting it configures a bench banner. A typo has the
    same shape. Either way the process must say so instead of accepting it."""
    out = run_probe("""
        import v4_pipe, v4_levers
        print(v4_levers.report(side=v4_levers.STAGE))
    """, V4_TOPK_STABLE=1)
    assert "V4_TOPK_STABLE" in out and "UNKNOWN" in out, out


# ── coordinator levers are observed from the RUN, not from the flag ───────────────────────────────

def test_the_pipelined_coordinator_records_what_it_ran_with():
    """A coordinator lever has no rebound method to inspect — `lazy` is an argument to a loop — so
    the only honest observation is what the loop used. Pin that the loop states it, and that the
    audit reads THAT rather than re-reading the env it already believed."""
    src = __import__("inspect").getsource(VP.coordinate_dspark_pipelined)
    assert 'v4_levers.note("V4_LAZY_DRAFT", lazy)' in src, \
        "the pipelined loop no longer records the lazy it ran with"
    assert src.index("lazy = V4_LAZY_DRAFT") < src.index('note("V4_LAZY_DRAFT"'), \
        "the note must record the RESOLVED value, not the env"
    out = run_probe("""
        import v4_pipe, v4_levers
        v4_levers.note("V4_LAZY_DRAFT", False)          # what a run actually did
        print(v4_levers.report(side=v4_levers.COORDINATOR))
    """, V4_LAZY_DRAFT=1)
    assert "V4_LAZY_DRAFT" in out and "MISMATCH" in out and "observed=off" in out, out


    # The end-to-end half — a real pipelined round over real sockets, and the observation coming back
    # from the loop — needs the socket ring fixture and lives next to it, in
    # tests/test_v4_pipe.py::test_the_audit_observes_the_lazy_lever_a_real_round_actually_used.
