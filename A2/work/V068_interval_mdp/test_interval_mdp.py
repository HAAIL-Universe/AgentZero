"""Tests for V068: Interval MDP Analysis"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V067_pctl_model_checking'))

from interval_mdp import (
    ProbInterval, IntervalMDP, make_interval_mc, make_interval_mdp,
    check_feasibility, check_all_feasible,
    robust_reachability, robust_safety, robust_expected_reward,
    OptimizationDirection,
    IntervalPCTLChecker, check_interval_pctl, check_interval_pctl_state,
    verify_robust_property, batch_interval_check,
    resolve_to_mc, resolve_feasible,
    compare_point_vs_interval, interval_width_analysis, sensitivity_analysis,
    _optimal_distribution,
)
from pctl_model_check import (
    tt, ff, atom, pnot, pand, por, prob_geq, prob_leq, prob_gt, prob_lt,
    next_f, until, bounded_until, eventually, always, bounded_eventually,
)


# ============================================================
# Section 1: ProbInterval basics
# ============================================================

class TestProbInterval:
    def test_creation(self):
        iv = ProbInterval(0.2, 0.5)
        assert iv.lo == 0.2
        assert iv.hi == 0.5

    def test_contains(self):
        iv = ProbInterval(0.2, 0.5)
        assert iv.contains(0.3)
        assert iv.contains(0.2)
        assert iv.contains(0.5)
        assert not iv.contains(0.1)
        assert not iv.contains(0.6)

    def test_width(self):
        iv = ProbInterval(0.2, 0.5)
        assert abs(iv.width() - 0.3) < 1e-10

    def test_midpoint(self):
        iv = ProbInterval(0.2, 0.8)
        assert abs(iv.midpoint() - 0.5) < 1e-10

    def test_clamping(self):
        iv = ProbInterval(-0.1, 1.2)
        assert iv.lo == 0.0
        assert iv.hi == 1.0

    def test_invalid_interval(self):
        with pytest.raises(ValueError):
            ProbInterval(0.6, 0.3)

    def test_point_interval(self):
        iv = ProbInterval(0.5, 0.5)
        assert iv.width() < 1e-10
        assert iv.contains(0.5)


# ============================================================
# Section 2: IntervalMDP construction
# ============================================================

class TestIntervalMDPConstruction:
    def test_interval_mc(self):
        imdp = make_interval_mc([
            [(0.4, 0.6), (0.4, 0.6)],
            [(0.3, 0.5), (0.5, 0.7)],
        ])
        assert imdp.n_states == 2
        assert imdp.is_interval_mc()
        assert len(imdp.validate()) == 0

    def test_interval_mdp(self):
        imdp = make_interval_mdp(2, {
            0: {
                "a": [(0.4, 0.6), (0.4, 0.6)],
                "b": [(0.8, 1.0), (0.0, 0.2)],
            },
            1: {
                "a": [(0.1, 0.3), (0.7, 0.9)],
            },
        })
        assert imdp.n_states == 2
        assert not imdp.is_interval_mc()
        assert len(imdp.actions[0]) == 2
        assert len(imdp.actions[1]) == 1

    def test_with_labels(self):
        imdp = make_interval_mc(
            [[(0.5, 0.5), (0.5, 0.5)], [(1.0, 1.0), (0.0, 0.0)]],
            state_labels=["good", "bad"],
            ap_labels={0: {"safe"}, 1: {"fail"}},
        )
        assert imdp.state_labels == ["good", "bad"]
        assert "safe" in imdp.ap_labels[0]
        assert "fail" in imdp.ap_labels[1]

    def test_three_state(self):
        imdp = make_interval_mc([
            [(0.0, 0.0), (0.4, 0.6), (0.4, 0.6)],
            [(0.2, 0.4), (0.0, 0.0), (0.6, 0.8)],
            [(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)],
        ])
        assert imdp.n_states == 3
        assert len(imdp.validate()) == 0


# ============================================================
# Section 3: Feasibility checking
# ============================================================

class TestFeasibility:
    def test_feasible(self):
        imdp = make_interval_mc([
            [(0.4, 0.6), (0.4, 0.6)],
            [(0.3, 0.5), (0.5, 0.7)],
        ])
        feas, dist = check_feasibility(imdp, 0)
        assert feas
        assert dist is not None
        assert abs(sum(dist) - 1.0) < 1e-8

    def test_infeasible_sum_too_low(self):
        imdp = make_interval_mc([
            [(0.1, 0.2), (0.1, 0.2)],
            [(0.5, 0.5), (0.5, 0.5)],
        ])
        feas, dist = check_feasibility(imdp, 0)
        assert not feas

    def test_infeasible_sum_too_high(self):
        imdp = make_interval_mc([
            [(0.6, 0.8), (0.6, 0.8)],
            [(0.5, 0.5), (0.5, 0.5)],
        ])
        feas, dist = check_feasibility(imdp, 0)
        assert not feas

    def test_exact_feasible(self):
        imdp = make_interval_mc([
            [(0.3, 0.3), (0.7, 0.7)],
            [(1.0, 1.0), (0.0, 0.0)],
        ])
        feas, dist = check_feasibility(imdp, 0)
        assert feas
        assert abs(dist[0] - 0.3) < 1e-8
        assert abs(dist[1] - 0.7) < 1e-8

    def test_all_feasible(self):
        imdp = make_interval_mc([
            [(0.4, 0.6), (0.4, 0.6)],
            [(0.3, 0.5), (0.5, 0.7)],
        ])
        ok, issues = check_all_feasible(imdp)
        assert ok
        assert len(issues) == 0


# ============================================================
# Section 4: Optimal distribution
# ============================================================

class TestOptimalDistribution:
    def test_maximize(self):
        row = [ProbInterval(0.2, 0.5), ProbInterval(0.2, 0.5), ProbInterval(0.1, 0.3)]
        values = [10.0, 1.0, 5.0]
        dist = _optimal_distribution(row, values, OptimizationDirection.MAX)
        assert abs(sum(dist) - 1.0) < 1e-8
        # Should push mass toward highest value (index 0)
        assert dist[0] >= 0.2 - 1e-10

    def test_minimize(self):
        row = [ProbInterval(0.2, 0.5), ProbInterval(0.2, 0.5), ProbInterval(0.1, 0.3)]
        values = [10.0, 1.0, 5.0]
        dist = _optimal_distribution(row, values, OptimizationDirection.MIN)
        assert abs(sum(dist) - 1.0) < 1e-8
        # Should push mass toward lowest value (index 1)
        assert dist[1] >= 0.2 - 1e-10

    def test_exact_intervals(self):
        row = [ProbInterval(0.3, 0.3), ProbInterval(0.7, 0.7)]
        values = [1.0, 0.0]
        dist_max = _optimal_distribution(row, values, OptimizationDirection.MAX)
        dist_min = _optimal_distribution(row, values, OptimizationDirection.MIN)
        # No flexibility: both should be the same
        assert abs(dist_max[0] - 0.3) < 1e-8
        assert abs(dist_min[0] - 0.3) < 1e-8


# ============================================================
# Section 5: Robust reachability
# ============================================================

class TestRobustReachability:
    def test_absorbing_target(self):
        # State 1 is absorbing and target
        imdp = make_interval_mc([
            [(0.0, 0.0), (1.0, 1.0)],
            [(0.0, 0.0), (1.0, 1.0)],
        ])
        prob_min = robust_reachability(imdp, {1}, OptimizationDirection.MIN)
        prob_max = robust_reachability(imdp, {1}, OptimizationDirection.MAX)
        assert abs(prob_min[0] - 1.0) < 1e-6
        assert abs(prob_max[0] - 1.0) < 1e-6

    def test_unreachable_target(self):
        # State 0 is absorbing, target is state 1
        imdp = make_interval_mc([
            [(1.0, 1.0), (0.0, 0.0)],
            [(0.0, 0.0), (1.0, 1.0)],
        ])
        prob_min = robust_reachability(imdp, {1}, OptimizationDirection.MIN)
        assert abs(prob_min[0]) < 1e-6

    def test_interval_uncertainty(self):
        # State 0: go to target (1) with prob [0.3, 0.7], stay with [0.3, 0.7]
        imdp = make_interval_mc([
            [(0.3, 0.7), (0.3, 0.7)],
            [(0.0, 0.0), (1.0, 1.0)],
        ])
        prob_min = robust_reachability(imdp, {1}, OptimizationDirection.MIN)
        prob_max = robust_reachability(imdp, {1}, OptimizationDirection.MAX)
        # Both should be 1.0 (eventually reach 1 with positive prob)
        assert abs(prob_min[0] - 1.0) < 1e-6
        assert abs(prob_max[0] - 1.0) < 1e-6

    def test_three_state_reachability(self):
        # s0 -> s1 or s2, s1 -> s2 (absorbing target), s2 absorbing
        imdp = make_interval_mc([
            [(0.0, 0.0), (0.3, 0.7), (0.3, 0.7)],
            [(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)],
            [(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)],
        ])
        prob_min = robust_reachability(imdp, {2}, OptimizationDirection.MIN)
        assert abs(prob_min[0] - 1.0) < 1e-6
        assert abs(prob_min[1] - 1.0) < 1e-6

    def test_min_max_gap(self):
        # State 0 -> state 1 (target) with [0.2, 0.8], -> state 2 (absorbing) with [0.2, 0.8]
        imdp = make_interval_mc([
            [(0.0, 0.0), (0.2, 0.8), (0.2, 0.8)],
            [(0.0, 0.0), (1.0, 1.0), (0.0, 0.0)],
            [(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)],
        ])
        prob_min = robust_reachability(imdp, {1}, OptimizationDirection.MIN)
        prob_max = robust_reachability(imdp, {1}, OptimizationDirection.MAX)
        # Min: adversary sends mass to s2 (absorbing non-target) -> prob 0.2
        # Max: maximize mass to s1 -> prob 0.8
        assert abs(prob_min[0] - 0.2) < 1e-6
        assert abs(prob_max[0] - 0.8) < 1e-6


# ============================================================
# Section 6: Robust safety
# ============================================================

class TestRobustSafety:
    def test_always_safe(self):
        imdp = make_interval_mc([
            [(0.5, 0.5), (0.5, 0.5)],
            [(0.5, 0.5), (0.5, 0.5)],
        ])
        prob = robust_safety(imdp, {0, 1}, steps=10, direction=OptimizationDirection.MIN)
        assert abs(prob[0] - 1.0) < 1e-6

    def test_unsafe_absorbing(self):
        # s0 safe, s1 unsafe. From s0 go to s1 with [0.3, 0.7]
        imdp = make_interval_mc([
            [(0.3, 0.7), (0.3, 0.7)],
            [(0.0, 0.0), (1.0, 1.0)],
        ])
        prob_min = robust_safety(imdp, {0}, steps=5, direction=OptimizationDirection.MIN)
        prob_max = robust_safety(imdp, {0}, steps=5, direction=OptimizationDirection.MAX)
        # Min safety: adversary maximizes going to unsafe (0.7 each step)
        # Max safety: stay in s0 as much as possible (0.7 each step)
        assert prob_min[0] < prob_max[0]
        assert prob_min[0] > 0  # Some chance of staying safe

    def test_one_step_safety(self):
        imdp = make_interval_mc([
            [(0.6, 0.8), (0.2, 0.4)],
            [(0.0, 0.0), (1.0, 1.0)],
        ])
        prob_min = robust_safety(imdp, {0}, steps=1, direction=OptimizationDirection.MIN)
        # Min: adversary pushes to unsafe -> stay prob = 0.6
        assert abs(prob_min[0] - 0.6) < 1e-6


# ============================================================
# Section 7: Interval PCTL - Next
# ============================================================

class TestIntervalPCTLNext:
    def test_next_exact(self):
        imdp = make_interval_mc(
            [[(0.3, 0.3), (0.7, 0.7)], [(0.5, 0.5), (0.5, 0.5)]],
            ap_labels={0: set(), 1: {"target"}},
        )
        formula = prob_geq(0.7, next_f(atom("target")))
        result = check_interval_pctl(imdp, formula)
        assert 0 in result.sat_pessimistic  # P(X target | s0) = 0.7 >= 0.7
        assert 0 in result.sat_optimistic

    def test_next_interval(self):
        imdp = make_interval_mc(
            [[(0.2, 0.6), (0.4, 0.8)], [(0.0, 0.0), (1.0, 1.0)]],
            ap_labels={0: set(), 1: {"target"}},
        )
        # P(X target | s0) in [0.4, 0.8]
        formula = prob_geq(0.5, next_f(atom("target")))
        result = check_interval_pctl(imdp, formula)
        # Pessimistic: min prob = 0.4 < 0.5 -> NOT in pessimistic
        assert 0 not in result.sat_pessimistic
        # Optimistic: max prob = 0.8 >= 0.5 -> in optimistic
        assert 0 in result.sat_optimistic

    def test_next_all_satisfy(self):
        imdp = make_interval_mc(
            [[(0.0, 0.1), (0.9, 1.0)], [(0.0, 0.0), (1.0, 1.0)]],
            ap_labels={0: set(), 1: {"target"}},
        )
        formula = prob_geq(0.9, next_f(atom("target")))
        result = check_interval_pctl(imdp, formula)
        assert 0 in result.sat_pessimistic
        assert 1 in result.sat_pessimistic


# ============================================================
# Section 8: Interval PCTL - Until
# ============================================================

class TestIntervalPCTLUntil:
    def test_until_certain(self):
        # s0 -> s1 certainly, s1 has "target"
        imdp = make_interval_mc(
            [[(0.0, 0.0), (1.0, 1.0)], [(0.0, 0.0), (1.0, 1.0)]],
            ap_labels={0: set(), 1: {"target"}},
        )
        formula = prob_geq(1.0, until(tt(), atom("target")))
        result = check_interval_pctl(imdp, formula)
        assert 0 in result.sat_pessimistic

    def test_until_interval(self):
        # s0 -> s1 [0.3, 0.7], s0 -> s2 [0.3, 0.7]
        # s1 has "target", s2 absorbing
        imdp = make_interval_mc(
            [
                [(0.0, 0.0), (0.3, 0.7), (0.3, 0.7)],
                [(0.0, 0.0), (1.0, 1.0), (0.0, 0.0)],
                [(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)],
            ],
            ap_labels={0: set(), 1: {"target"}, 2: set()},
        )
        # P(F target | s0) in [0.3, 0.7]
        formula = prob_geq(0.5, until(tt(), atom("target")))
        result = check_interval_pctl(imdp, formula)
        assert 0 not in result.sat_pessimistic  # min=0.3 < 0.5
        assert 0 in result.sat_optimistic       # max=0.7 >= 0.5

    def test_bounded_until(self):
        imdp = make_interval_mc(
            [[(0.4, 0.6), (0.4, 0.6)], [(0.0, 0.0), (1.0, 1.0)]],
            ap_labels={0: set(), 1: {"target"}},
        )
        # Bounded: reach target within 3 steps
        formula = prob_geq(0.8, bounded_until(tt(), atom("target"), 3))
        result = check_interval_pctl(imdp, formula)
        # s1 is target: always satisfies
        assert 1 in result.sat_pessimistic


# ============================================================
# Section 9: Interval PCTL - Boolean combinations
# ============================================================

class TestIntervalPCTLBoolean:
    def test_and(self):
        imdp = make_interval_mc(
            [[(0.5, 0.5), (0.5, 0.5)], [(0.5, 0.5), (0.5, 0.5)]],
            ap_labels={0: {"a", "b"}, 1: {"a"}},
        )
        formula = pand(atom("a"), atom("b"))
        checker = IntervalPCTLChecker(imdp)
        sat = checker._check_direction(formula, pessimistic=True)
        assert 0 in sat
        assert 1 not in sat

    def test_or(self):
        imdp = make_interval_mc(
            [[(0.5, 0.5), (0.5, 0.5)], [(0.5, 0.5), (0.5, 0.5)]],
            ap_labels={0: {"a"}, 1: {"b"}},
        )
        formula = por(atom("a"), atom("b"))
        checker = IntervalPCTLChecker(imdp)
        sat = checker._check_direction(formula, pessimistic=True)
        assert 0 in sat
        assert 1 in sat

    def test_not(self):
        imdp = make_interval_mc(
            [[(0.5, 0.5), (0.5, 0.5)], [(0.5, 0.5), (0.5, 0.5)]],
            ap_labels={0: {"a"}, 1: set()},
        )
        formula = pnot(atom("a"))
        checker = IntervalPCTLChecker(imdp)
        sat = checker._check_direction(formula, pessimistic=True)
        assert 0 not in sat
        assert 1 in sat


# ============================================================
# Section 10: MDP with nondeterministic actions
# ============================================================

class TestMDPActions:
    def test_two_actions(self):
        # State 0 has two actions: 'safe' (high self-loop) and 'risky' (high target)
        imdp = make_interval_mdp(2, {
            0: {
                "safe": [(0.8, 1.0), (0.0, 0.2)],
                "risky": [(0.0, 0.2), (0.8, 1.0)],
            },
            1: {
                "stay": [(0.0, 0.0), (1.0, 1.0)],
            },
        })
        # Max reachability to s1: pick risky action -> prob 0.8-1.0
        prob_max = robust_reachability(imdp, {1}, OptimizationDirection.MAX)
        assert prob_max[0] >= 0.8 - 1e-6

        # Min reachability to s1: pick safe action -> prob 0.0-0.2
        prob_min = robust_reachability(imdp, {1}, OptimizationDirection.MIN)
        # Still reaches eventually via safe action (0-0.2 prob each step)
        # but min is via safe action with adversarial distribution

    def test_action_choice_matters(self):
        imdp = make_interval_mdp(3, {
            0: {
                "left": [(0.0, 0.0), (1.0, 1.0), (0.0, 0.0)],
                "right": [(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)],
            },
            1: {"stay": [(0.0, 0.0), (1.0, 1.0), (0.0, 0.0)]},
            2: {"stay": [(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)]},
        })
        # Max reach s1: take "left"
        prob_max = robust_reachability(imdp, {1}, OptimizationDirection.MAX)
        assert abs(prob_max[0] - 1.0) < 1e-6
        # Min reach s1: take "right" -> 0
        prob_min = robust_reachability(imdp, {1}, OptimizationDirection.MIN)
        assert abs(prob_min[0]) < 1e-6


# ============================================================
# Section 11: Expected reward
# ============================================================

class TestExpectedReward:
    def test_one_step_reward(self):
        imdp = make_interval_mc([
            [(0.0, 0.0), (1.0, 1.0)],
            [(0.0, 0.0), (1.0, 1.0)],
        ])
        rewards = [5.0, 0.0]
        vals = robust_expected_reward(imdp, rewards, {1}, OptimizationDirection.MIN)
        assert abs(vals[0] - 5.0) < 1e-6

    def test_interval_reward(self):
        # s0 -> s1 [0.3, 0.7], s0 -> s0 [0.3, 0.7]; s1 target
        imdp = make_interval_mc([
            [(0.3, 0.7), (0.3, 0.7)],
            [(0.0, 0.0), (1.0, 1.0)],
        ])
        rewards = [1.0, 0.0]
        vals_min = robust_expected_reward(imdp, rewards, {1}, OptimizationDirection.MIN)
        vals_max = robust_expected_reward(imdp, rewards, {1}, OptimizationDirection.MAX)
        # Min reward: maximize probability of reaching target quickly (min steps)
        # Max reward: minimize probability of reaching target (max steps)
        assert vals_min[0] < vals_max[0]


# ============================================================
# Section 12: Point resolution and comparison
# ============================================================

class TestPointResolution:
    def test_midpoint_resolution(self):
        imdp = make_interval_mc([
            [(0.2, 0.4), (0.6, 0.8)],
            [(0.5, 0.5), (0.5, 0.5)],
        ])
        mc = resolve_to_mc(imdp, "midpoint")
        assert abs(mc.transition[0][0] - 0.3) < 1e-6
        assert abs(mc.transition[0][1] - 0.7) < 1e-6

    def test_feasible_resolution(self):
        imdp = make_interval_mc([
            [(0.4, 0.6), (0.4, 0.6)],
            [(0.3, 0.5), (0.5, 0.7)],
        ])
        dist = resolve_feasible(imdp, 0)
        assert dist is not None
        assert abs(sum(dist) - 1.0) < 1e-8

    def test_comparison(self):
        imdp = make_interval_mc(
            [
                [(0.0, 0.0), (0.2, 0.8), (0.2, 0.8)],
                [(0.0, 0.0), (1.0, 1.0), (0.0, 0.0)],
                [(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)],
            ],
        )
        result = compare_point_vs_interval(imdp, {1})
        assert "min_reachability" in result
        assert "max_reachability" in result
        assert "gap" in result
        assert result["max_gap"] > 0  # There's uncertainty


# ============================================================
# Section 13: Width analysis
# ============================================================

class TestWidthAnalysis:
    def test_exact_mc(self):
        imdp = make_interval_mc([
            [(0.3, 0.3), (0.7, 0.7)],
            [(0.5, 0.5), (0.5, 0.5)],
        ])
        analysis = interval_width_analysis(imdp)
        assert analysis["uncertain_intervals"] == 0
        assert analysis["mean_width"] < 1e-8

    def test_uncertain_mc(self):
        imdp = make_interval_mc([
            [(0.2, 0.5), (0.5, 0.8)],
            [(0.3, 0.7), (0.3, 0.7)],
        ])
        analysis = interval_width_analysis(imdp)
        assert analysis["uncertain_intervals"] > 0
        assert analysis["max_width"] > 0


# ============================================================
# Section 14: Sensitivity analysis
# ============================================================

class TestSensitivity:
    def test_sensitivity(self):
        imdp = make_interval_mc([
            [(0.0, 0.0), (0.2, 0.8), (0.2, 0.8)],
            [(0.0, 0.0), (1.0, 1.0), (0.0, 0.0)],
            [(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)],
        ])
        sens = sensitivity_analysis(imdp, {1}, perturbation=0.05)
        # Should have entries for s0's transitions to s1 and s2
        assert len(sens) > 0


# ============================================================
# Section 15: Verify robust property
# ============================================================

class TestVerifyRobust:
    def test_robust_verified(self):
        # All transitions go to target with high prob
        imdp = make_interval_mc([
            [(0.0, 0.0), (1.0, 1.0)],
            [(0.0, 0.0), (1.0, 1.0)],
        ])
        result = verify_robust_property(imdp, {1}, min_prob=0.9)
        assert result["verdict"] == "ROBUST"

    def test_robust_violated(self):
        imdp = make_interval_mc([
            [(1.0, 1.0), (0.0, 0.0)],
            [(0.0, 0.0), (1.0, 1.0)],
        ])
        result = verify_robust_property(imdp, {1}, min_prob=0.5)
        assert 0 not in result["verified_states"]

    def test_uncertain(self):
        imdp = make_interval_mc([
            [(0.0, 0.0), (0.2, 0.8), (0.2, 0.8)],
            [(0.0, 0.0), (1.0, 1.0), (0.0, 0.0)],
            [(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)],
        ])
        result = verify_robust_property(imdp, {1}, min_prob=0.5)
        # s0: min=0.2, max=0.8 -> uncertain for threshold 0.5
        assert 0 in result["uncertain_states"]


# ============================================================
# Section 16: Batch checking
# ============================================================

class TestBatchCheck:
    def test_batch(self):
        imdp = make_interval_mc(
            [[(0.3, 0.3), (0.7, 0.7)], [(0.5, 0.5), (0.5, 0.5)]],
            ap_labels={0: {"a"}, 1: {"b"}},
        )
        formulas = [
            prob_geq(0.5, next_f(atom("b"))),
            prob_leq(0.5, next_f(atom("a"))),
        ]
        results = batch_interval_check(imdp, formulas)
        assert len(results) == 2


# ============================================================
# Section 17: Edge cases
# ============================================================

class TestEdgeCases:
    def test_single_state(self):
        imdp = make_interval_mc([[(1.0, 1.0)]])
        prob = robust_reachability(imdp, {0}, OptimizationDirection.MIN)
        assert abs(prob[0] - 1.0) < 1e-6

    def test_all_exact(self):
        # When all intervals are points, results should match V067
        imdp = make_interval_mc(
            [[(0.3, 0.3), (0.7, 0.7)], [(0.4, 0.4), (0.6, 0.6)]],
            ap_labels={0: set(), 1: {"target"}},
        )
        result = check_interval_pctl(imdp, prob_geq(0.7, next_f(atom("target"))))
        # Pessimistic should equal optimistic (no uncertainty)
        assert result.sat_pessimistic == result.sat_optimistic

    def test_check_state_api(self):
        imdp = make_interval_mc(
            [[(0.0, 0.0), (1.0, 1.0)], [(0.0, 0.0), (1.0, 1.0)]],
            ap_labels={0: set(), 1: {"target"}},
        )
        result = check_interval_pctl_state(imdp, 0, prob_geq(1.0, next_f(atom("target"))))
        assert result["definitely"] is True
        assert result["possibly"] is True
        assert result["uncertain"] is False

    def test_eventually_sugar(self):
        imdp = make_interval_mc(
            [[(0.0, 0.0), (1.0, 1.0)], [(0.0, 0.0), (1.0, 1.0)]],
            ap_labels={0: set(), 1: {"target"}},
        )
        formula = prob_geq(1.0, eventually(atom("target")))
        result = check_interval_pctl(imdp, formula)
        assert 0 in result.sat_pessimistic


# ============================================================
# Section 18: Complex scenarios
# ============================================================

class TestComplexScenarios:
    def test_gambler_ruin_interval(self):
        """Gambler's ruin with uncertain win probability."""
        # 5 states: 0 and 4 absorbing, 1-3 uncertain transitions
        n = 5
        intervals = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                intervals[i][j] = (0.0, 0.0)
        # Absorbing
        intervals[0][0] = (1.0, 1.0)
        intervals[4][4] = (1.0, 1.0)
        # Uncertain transitions
        for i in [1, 2, 3]:
            intervals[i][i-1] = (0.3, 0.6)  # lose
            intervals[i][i+1] = (0.4, 0.7)  # win

        imdp = make_interval_mc(intervals,
                                ap_labels={0: {"lose"}, 4: {"win"}})

        # Min probability of winning from state 2
        prob_min = robust_reachability(imdp, {4}, OptimizationDirection.MIN)
        prob_max = robust_reachability(imdp, {4}, OptimizationDirection.MAX)

        assert prob_min[2] > 0  # Some chance
        assert prob_max[2] < 1  # Not certain
        assert prob_min[2] < prob_max[2]  # Gap exists

    def test_protocol_with_uncertainty(self):
        """Simple communication protocol: send -> ack/nack -> retry/done."""
        # s0: send, s1: ack (done), s2: nack (retry -> s0)
        imdp = make_interval_mc([
            [(0.0, 0.0), (0.7, 0.95), (0.05, 0.3)],  # send
            [(0.0, 0.0), (1.0, 1.0), (0.0, 0.0)],    # ack (absorbing)
            [(1.0, 1.0), (0.0, 0.0), (0.0, 0.0)],    # nack -> retry
        ], ap_labels={0: {"sending"}, 1: {"success"}, 2: {"retry"}})

        # Will we eventually succeed?
        prob_min = robust_reachability(imdp, {1}, OptimizationDirection.MIN)
        assert abs(prob_min[0] - 1.0) < 1e-6  # Always eventually succeeds

        # Within 3 steps?
        formula = prob_geq(0.9, bounded_until(tt(), atom("success"), 3))
        result = check_interval_pctl(imdp, formula)
        # Pessimistic might not hit 0.9 in 3 steps
        # but optimistic should


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
