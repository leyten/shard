"""fastverify tree-topology validation (validate_tree, wired into FastVerify.tree_decode).

_tbuild's ancestor walk (`while j != -1: j = par[j]`) had no cycle or bound check: a cyclic
`par` from the wire spins the stage FOREVER, and an out-of-range parent index-errors or wraps
(negative indexing) into a silently wrong ancestor mask. These tests pin the fix: length,
parent-range, acyclicity, and depth consistency are all validated before any walk, and
tree_decode rejects a malformed tree (ValueError) or an overflowing one (ContextOverflow)
before touching the cache.

Run: pytest tests/test_tree_validate.py -q   (CPU-only, no GPU / no model needed)
"""
import os
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase0"))

from fastverify import ContextOverflow, FastVerify, validate_tree


def test_valid_trees_pass():
    validate_tree(1, [-1], [0])
    validate_tree(6, [-1, 0, 1, 2, 0, 4], [0, 1, 2, 3, 1, 2])       # tree.py's own example
    validate_tree(3, [1, -1, 1], [1, 0, 1])                          # any node order (root not node 0)


def test_cycle_rejected():
    """THE regression: a 2-cycle used to spin the ancestor walk forever (wedged stage)."""
    with pytest.raises(ValueError, match="cycle|depth"):
        validate_tree(3, [-1, 2, 1], [0, 1, 1])
    with pytest.raises(ValueError, match="range"):
        validate_tree(2, [-1, 1], [0, 1])                            # self-parent


def test_parent_out_of_range_rejected():
    with pytest.raises(ValueError, match="range"):
        validate_tree(3, [-1, 5, 0], [0, 1, 1])                      # index past the tree
    with pytest.raises(ValueError, match="range"):
        validate_tree(3, [-1, -2, 0], [0, 1, 1])                     # negative wrap
    with pytest.raises(ValueError, match="range"):
        validate_tree(2, [-1, "0"], [0, 1])                          # non-int from the wire


def test_length_mismatch_rejected():
    with pytest.raises(ValueError, match="length"):
        validate_tree(3, [-1, 0], [0, 1, 1])
    with pytest.raises(ValueError, match="length"):
        validate_tree(3, [-1, 0, 0], [0, 1])
    with pytest.raises(ValueError, match="invalid"):
        validate_tree(0, [], [])


def test_bad_depth_rejected():
    with pytest.raises(ValueError, match="depth"):
        validate_tree(2, [-1, 0], [0, 5])                            # not parent-depth + 1
    with pytest.raises(ValueError, match="invalid"):
        validate_tree(2, [-1, 0], [0, -1])
    with pytest.raises(ValueError, match="invalid"):
        validate_tree(2, [-1, 0], [0, "1"])


def _fake_fv(maxlen=64):
    """Just enough of a FastVerify for tree_decode to reach (and stop at) its guards."""
    return SimpleNamespace(maxlen=maxlen, tM=None, tpar=None)


def test_tree_decode_rejects_before_any_walk():
    """Wire-level: a malformed msg hits ValueError in tree_decode (-> the serve loop's
    bad-msg reset), never the unbounded parent walk. Before the fix a cycle hung here
    and an out-of-range parent raised IndexError from inside _tbuild instead."""
    h = torch.zeros(1, 3, 8)
    with pytest.raises(ValueError, match="range"):
        FastVerify.tree_decode(_fake_fv(), h, 0, [-1, 5, 0], [0, 1, 1])
    with pytest.raises(ValueError, match="cycle|depth"):
        FastVerify.tree_decode(_fake_fv(), h, 0, [-1, 2, 1], [0, 1, 1])


def test_tree_decode_overflow_fails_clean():
    h = torch.zeros(1, 3, 8)
    with pytest.raises(ContextOverflow):
        FastVerify.tree_decode(_fake_fv(maxlen=64), h, 62, [-1, 0, 1], [0, 1, 2])
