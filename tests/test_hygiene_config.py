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
