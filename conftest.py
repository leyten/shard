"""Make the engine layout importable, in one place, for both this process and its children.

WHY THIS FILE EXISTS.  The engines are flat module namespaces by deliberate choice — a stage runs
as `python3 v4_pipe.py stage ...` on a rented box with the files copied next to each other, so
`import v4_stage` has to resolve without a package install, a venv, or a repo checkout. Turning
them into real packages would buy tidier imports and cost the thing that actually ships.

So the directories are the boundary and `sys.path` is the wiring. This file does that wiring once,
at collection, for every engine dir plus the vendored reference trees:

  engines/deepseek_v4/   v4_*      DeepSeek-V4-Flash
  engines/kimi_k3/       k3_*      Kimi-K3
  engines/minimax_m25/   m25_*     MiniMax-M2.5
  phase0/                          shared tooling that is not any one engine's
  vendor/                          byte-identical upstream trees the engines drive

It also prepends the same list to PYTHONPATH. That is not belt-and-braces: several suites prove
things about flags that are read AT MODULE IMPORT, so they assert by spawning a fresh interpreter
(see tests/test_v4_full_stack.py). Those children inherit os.environ, and without this they would
import nothing. Setting it here rather than in each test is what let the move to engines/ leave
~30 `os.path.join(ROOT, "phase0")` call sites untouched — `phase0/` is still a real directory of
shared tooling, so those keep resolving; only the engines moved out from under it.
"""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
ENGINES = ROOT / "engines"

# Order matters only in that engines come first: if a name ever collides between an engine and the
# shared tooling, the engine's own copy is the one it meant.
PATHS = [str(p) for p in sorted(ENGINES.iterdir()) if p.is_dir()] if ENGINES.is_dir() else []
PATHS += [str(ROOT / "phase0"), str(ROOT / "vendor"), str(ROOT)]
PATHS = [p for p in PATHS if os.path.isdir(p)]

for _p in reversed(PATHS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_existing = os.environ.get("PYTHONPATH", "")
_want = os.pathsep.join(PATHS)
if _want not in _existing:
    os.environ["PYTHONPATH"] = _want + (os.pathsep + _existing if _existing else "")
