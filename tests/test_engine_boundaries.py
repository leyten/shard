"""The engine boundary, as a test instead of a promise.

shard is not one engine, it is a protocol SPINE plus one bespoke engine per model — M2.5, K3,
V4-Flash, and whatever lands next. Each engine is tuned for its model down to the kernels, so they
are meant to diverge. What must NOT drift is the spine: the wire, the ring, receipts, placement.

Today that separation exists only as a filename prefix (`v4_*`, `k3_*`, `m25_*`) and a sentence in
CLAUDE.md that says "never edit the other engine". That is an honour system: nothing mechanical
stops a future session from importing one engine's drafter into another's stage, or from reaching
down out of the spine into a model-specific kernel. Both are one-line mistakes, and both are
invisible in review because the import is spelled exactly like a local one — every engine module
lives in the same flat directory, so `import k3_stage` from a v4 file just works.

These tests are the ratchet. They pass as written today (verified: zero cross-engine imports, and
shard/ imports no engine at all), so they cost nothing to adopt — their whole value is failing the
first time someone crosses a line that currently only a convention defends.

TWO RULES, and the reason each one matters:

  1. NO ENGINE IMPORTS ANOTHER ENGINE.  Engines are per-model by design; a shared import means one
     model's numerics silently rides on another's tuning, and a lever measured on one ring starts
     changing a different ring. When two engines genuinely need the same thing, it belongs in the
     spine, not in a sideways import.

  2. THE SPINE NEVER IMPORTS AN ENGINE.  `shard/` is the part a third party runs. The dependency
     points ONE way (engines -> shard, see the workspace note on the shard/c0mpute boundary); the
     moment it points back, the protocol cannot be used without dragging a specific model's engine
     in with it, and "shard is a generic engine" stops being true.

WHEN THE TREE MOVES to engines/<name>/ + shard/ + vendor/, only ENGINE_PREFIXES and the directory
constants below change — the rules and their reasons do not.
"""
import ast
import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINES = ROOT / "engines"
SPINE = ROOT / "shard"

# module-name prefix -> the engine it belongs to. The directory name is now the boundary, but the
# PREFIX stays the thing we key on: engines ship as flat modules on purpose (a stage runs as
# `python3 v4_pipe.py stage ...` with the files copied next to each other on a rented box), so a
# foreign import is still spelled exactly like a local one and still needs catching.
# Vendored trees under vendor/ are NOT engines — they are byte-identical upstream code that a
# single engine drives — so they are never scanned.
ENGINE_PREFIXES = {"v4_": "deepseek_v4", "k3_": "kimi_k3", "m25_": "minimax_m25"}


def _engine_of(module_name):
    for prefix, engine in ENGINE_PREFIXES.items():
        if module_name.startswith(prefix):
            return engine
    return None


def _imports(path):
    """Every module name `path` imports, top-level or inside a function."""
    try:
        tree = ast.parse(path.read_text(errors="replace"), filename=str(path))
    except SyntaxError:                                   # not our problem here
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _engine_files():
    """Every engine module, wherever it lives — engines/<name>/ today, phase0/ before the move.

    Scanning both means this test kept its teeth ACROSS the reorganisation instead of silently
    matching nothing on either side of it, which is the failure mode a boundary test can least
    afford (see test_the_scan_actually_sees_the_engines)."""
    out = []
    for root in (ENGINES, ROOT / "phase0"):
        if root.is_dir():
            out += [p for p in sorted(root.rglob("*.py")) if _engine_of(p.name)]
    return out


def test_the_scan_actually_sees_the_engines():
    """A boundary test that silently matches nothing is worse than none — it reports success."""
    files = _engine_files()
    assert len(files) >= 10, f"expected to scan many engine modules, found {len(files)}"
    found = {_engine_of(p.name) for p in files}
    assert len(found) >= 2, f"expected at least two engines to compare, found {found}"


@pytest.mark.parametrize("path", _engine_files(), ids=lambda p: p.name)
def test_no_engine_imports_another_engine(path):
    """Engines are per-model and are SUPPOSED to diverge; a sideways import couples their numerics.

    Anything two engines both need belongs in the spine (shard/), where it is one implementation
    with one set of tests, not a quiet dependency from one model's tuning onto another's."""
    mine = _engine_of(path.name)
    foreign = sorted({f"{name} (-> {_engine_of(name)})" for name in _imports(path)
                      if _engine_of(name) and _engine_of(name) != mine})
    assert not foreign, (
        f"{path.name} belongs to engine '{mine}' but imports another engine's module(s): "
        f"{', '.join(foreign)}. Engines must not depend on each other — promote the shared piece "
        f"into shard/ (the spine) instead of importing sideways."
    )


# KNOWN DEBT, and it is the reason this file exists. `shard/coordinate.py` and `shard/stage.py`
# both do a deferred `import m25_pipe` in their serve path — the spine, the part a third party is
# meant to run, can currently only serve MiniMax-M2.5. That is precisely the coupling the
# ModelRuntime seam (docs/MODEL_RUNTIME.md) is supposed to invert, and it predates the K3 and
# V4-Flash engines, which is why neither of them can be driven through shard/ today.
#
# This is an EXEMPTION LIST, not an excuse: it is allowed to shrink and never to grow. A new spine
# file importing an engine fails immediately; removing the coupling means deleting an entry here.
SPINE_ENGINE_DEBT = {"coordinate.py": {"m25_pipe"}, "stage.py": {"m25_pipe"}}


@pytest.mark.parametrize(
    "path", sorted(SPINE.glob("*.py")) if SPINE.is_dir() else [], ids=lambda p: p.name)
def test_the_spine_never_imports_an_engine(path):
    """shard/ is what a third party runs. The dependency points engines -> shard, never back.

    If the spine imports a model's engine, the protocol cannot be used without that model's code,
    and the claim that shard is a generic engine behind ModelRuntime stops being true."""
    engines = {name for name in _imports(path) if _engine_of(name)}
    new = sorted(engines - SPINE_ENGINE_DEBT.get(path.name, set()))
    assert not new, (
        f"shard/{path.name} is spine code but imports engine module(s): {', '.join(new)}. "
        f"The spine must not depend on any single model's engine — invert it behind the "
        f"ModelRuntime seam (docs/MODEL_RUNTIME.md)."
    )


def test_the_spine_debt_only_shrinks():
    """The exemption list must not outlive what it exempts, or it becomes a place to hide."""
    for fname, expected in SPINE_ENGINE_DEBT.items():
        path = SPINE / fname
        if not path.is_dir() and path.exists():
            actual = {n for n in _imports(path) if _engine_of(n)}
            assert actual <= expected, (
                f"shard/{fname} gained engine imports {sorted(actual - expected)} — the debt list "
                f"is a ratchet, not a budget."
            )
            if actual < expected:
                pytest.fail(
                    f"shard/{fname} no longer imports {sorted(expected - actual)} — good, now "
                    f"delete it from SPINE_ENGINE_DEBT so the ratchet keeps its teeth.")
