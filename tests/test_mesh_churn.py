"""BUILD-PLAN M3 gate: reproduce placement_sim.out coverage within 1 point.

Two claims, both over the sim's exact event stream (tests/mesh_churn.py):
1. Fidelity. With the sim's own join key (Petals' min capacity, raw floats summed in join order)
   and its repair rule swapped into the real Cell, every row of placement_sim.out comes back
   within 1 pt (measured: within 0.05 pt, and hops and moves/node/day match too). The replay,
   the fit arithmetic and the coverage accounting are the sim's.
2. The shipped policy (design-mesh §3.3's join priority: holes, then k < k_min, then capacity,
   disk, valley fill, lane) is never worse than the sim by more than 1 pt, equals it at 40 nodes,
   and at 12-20 nodes it is far better on >= 2-holder coverage (+8..+13 pt), because it raises
   single-holder layers explicitly where the sim only raised the minimum capacity.
"""
import pytest

import mesh_churn as mc
from shard.mesh import placement as P


class SimRuleCell(P.Cell):
    """placement_sim.py's policy on the real Cell: best_start's key and its repair loop."""

    def capacity(self, exclude=None):
        c = [0.0] * self.profile.n_layers
        for hid, h in self.holdings.items():                  # join order, as the sim's dict
            if hid != exclude:
                for i in range(*h.span):
                    c[i] += 1.0 / (h.span[1] - h.span[0])
        return c

    def span_key(self, span, t, cnt, cap, disk):
        lo, hi = span
        c2 = cap[:]
        for i in range(lo, hi):
            c2[i] += 1.0 / (hi - lo)
        return (min(c2), -sum(cap[lo:hi]), lo % 4 == 0)

    def repair(self, now):
        h = self.counts()
        if min(h) != 0:
            return []
        for hid in sorted(self.holdings, key=lambda k: self.holdings[k].span[1] - self.holdings[k].span[0]):
            x = self.holdings[hid]
            if any(h[i] < 2 for i in range(*x.span)):
                continue
            best = self._choose(P.Fit(x.node, self.profile, self.policy).spans(), 1.0,
                                self.counts(exclude=hid), self.capacity(exclude=hid), set())
            if best != x.span:
                x.target = best
                return [P.Move(hid, x.span, best, 0, "hole")]
        return []


ROWS = sorted(mc.SIM_OUT)


@pytest.fixture
def sim_rules(monkeypatch):
    monkeypatch.setattr(mc, "Cell", SimRuleCell)


@pytest.mark.parametrize("nodes,policy", ROWS)
def test_replay_reproduces_placement_sim_out(sim_rules, nodes, policy):
    r = mc.averaged(policy, nodes)
    cov1, cov2 = mc.SIM_OUT[(nodes, policy)]
    assert abs(r["cov>=1"] - cov1) <= 0.01 and abs(r["cov>=2"] - cov2) <= 0.01, r


@pytest.mark.parametrize("nodes,policy", ROWS)
def test_design_policy_within_one_point_or_better(nodes, policy):
    r = mc.averaged(policy, nodes)
    cov1, cov2 = mc.SIM_OUT[(nodes, policy)]
    assert r["cov>=1"] >= cov1 - 0.01 and r["cov>=2"] >= cov2 - 0.01, r
    if nodes == 40:
        assert abs(r["cov>=1"] - cov1) <= 0.01 and abs(r["cov>=2"] - cov2) <= 0.01, r
    if nodes in (12, 20):
        assert r["cov>=2"] >= cov2 + 0.05, r


def test_kmin_repair_holds_k2_coverage_where_repair_only_lets_it_decay():
    base = mc.averaged("repair", 20)
    kmin = mc.averaged("repair", 20, kmin_repair=True)
    assert kmin["cov>=2"] >= base["cov>=2"] + 0.02 and kmin["cov>=1"] >= base["cov>=1"] - 0.005, (base, kmin)
