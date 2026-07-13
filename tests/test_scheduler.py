"""Scheduler facade (shard/scheduler.py) — the joint plan() result.

The old pair composed incorrectly: allocate() gave the coordinator a block and laid ranges
fat-first, topology() excluded the coordinator and ordered by latency — zipping the two could put
layers [20,30) BEFORE [0,20) in the pipeline. plan() decides order + blocks in ONE solve, so the
ordered ranges always tile the model in pipeline order.

Run: python3 -m pytest tests/test_scheduler.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shard.scheduler import JoinedNode, Scheduler  # noqa: E402


def _mesh(total_layers, vrams, rtt):
    sched = Scheduler("m25", total_layers)
    for nid in vrams:
        sched.register(JoinedNode(nid, vrams[nid], dict(rtt[nid])))
    return sched


def test_plan_is_one_joint_result_tiling_in_ring_order():
    # fat-first (allocate) order != latency (topology) order: 'a' is the fattest node but sits on
    # the pool's worst edges, so a latency ordering and a fat-first block layout MUST disagree —
    # exactly the composition the old facade got wrong.
    vrams = {"a": 6.0, "b": 4.0, "c": 3.0}
    rtt = {"a": {"b": 50.0, "c": 5.0},
           "b": {"a": 50.0, "c": 5.0},
           "c": {"a": 5.0, "b": 5.0}}
    sched = _mesh(8, vrams, rtt)
    plan = sched.plan(gb_per_layer=1.0, kv_gb_per_layer=0.0, headroom_gb=0.0, boundary_gb=0.0)
    stages = sorted(plan["stages"], key=lambda s: s["index"])
    assert [s["id"] for s in stages] == plan["order"]   # blocks laid IN pipeline order
    assert plan["order"][0] == plan["head"]             # the coordinator runs on stage 0
    lo = 0
    for s in stages:                                    # ordered ranges exactly tile the model
        assert s["lo"] == lo, f"pipeline visits layers out of order at stage {s['index']}"
        assert s["hi"] > s["lo"]
        lo = s["hi"]
    assert lo == 8


def test_plan_raises_when_pool_cannot_hold_the_model():
    vrams = {"a": 2.0, "b": 2.0}
    rtt = {"a": {"b": 10.0}, "b": {"a": 10.0}}
    sched = _mesh(8, vrams, rtt)
    with pytest.raises(ValueError):
        sched.plan(gb_per_layer=1.0, headroom_gb=0.0, boundary_gb=0.0)
