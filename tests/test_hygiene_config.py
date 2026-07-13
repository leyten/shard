"""Test-suite + packaging hygiene (audit M13 / wheel-deps).

Guards two config invariants in pyproject.toml:
  1. pytest is confined to tests/ (research/ + scratchpad/ scripts execute real work at
     import — model loads, subprocesses, network — and must never be collected), and the
     hardware/integration/gpu/mlx markers are declared.
  2. the wheel declares every module-level runtime import of the advertised `shard*`
     modules (a clean-env `pip install shard` must yield importable modules).
"""
import ast
import os
import sys
import tomllib

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyproject():
    with open(os.path.join(_REPO, "pyproject.toml"), "rb") as f:
        return tomllib.load(f)


def test_pytest_confined_to_tests_dir():
    ini = _pyproject().get("tool", {}).get("pytest", {}).get("ini_options", {})
    assert ini.get("testpaths") == ["tests"], (
        "pytest must only discover tests/ — a bare root `pytest` otherwise imports "
        "research/scratch scripts that do real work at import time")


def test_hardware_markers_declared():
    ini = _pyproject().get("tool", {}).get("pytest", {}).get("ini_options", {})
    declared = {m.split(":")[0].strip() for m in ini.get("markers", [])}
    assert {"hardware", "integration", "gpu", "mlx"} <= declared


def test_wheel_declares_module_level_runtime_deps():
    """Every top-level (non-lazy) third-party import in shard/*.py must be resolvable
    from [project.dependencies] — today that is cryptography, numpy, and torch."""
    deps = " ".join(_pyproject()["project"]["dependencies"])
    stdlib = sys.stdlib_module_names
    top_imports = set()
    shard_dir = os.path.join(_REPO, "shard")
    for fn in os.listdir(shard_dir):
        if not fn.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(shard_dir, fn)).read())
        for node in tree.body:                      # module body only: lazy imports exempt
            if isinstance(node, ast.Import):
                top_imports.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                top_imports.add(node.module.split(".")[0])
    third_party = {m for m in top_imports if m not in stdlib and m != "shard"}
    dist_of = {"numpy": "numpy", "torch": "torch", "cryptography": "cryptography"}
    for mod in sorted(third_party):
        assert mod in dist_of, f"shard/ imports {mod!r} at module level — map it to a dist here"
        assert dist_of[mod] in deps, (
            f"shard/ needs {mod!r} at import time but pyproject dependencies omit "
            f"{dist_of[mod]!r} — a clean-env install of the wheel breaks")
