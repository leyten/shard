"""Shared test setup: repo root + phase0 importable no matter where pytest runs from.

Collection discipline (audit M13): `pytest --collect-only` must do no network,
subprocess, GPU, or long-import work — anything expensive belongs in a fixture.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO, os.path.join(_REPO, "phase0")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
