"""Tests for V069: MDP Verification."""

import sys
import os
import pytest
from math import isclose

sys.path.insert(0, os.path.dirname(__file__))

from mdp_verification import (
    MDP, Policy, RandomizedPolicy, Objective,
    make_mdp, mdp_to_mc, mdp_to_interval_mdp,
    value_iteration, q_values, evaluate_policy,
    reachability, expected_steps,
    verify_reachability_bound, verify_policy_optimality,
    verify_reward_bound, verify_safety,
    smt_verify_policy_dominance, smt_verify_bellman_optimality,
    long_run_average_reward, compare_with_interval_mdp,
    policy_iteration, analyze_mdp, verify_mdp,
    ValueResult, ReachResult, VerificationResult,
)


# -----------------------------------------------------------------------
# Fixtures: common MDPs
# -----------------------------------------------------------------------

def simple_mdp():
    """Two-state MDP with two actions in state 0."""
    # State 0: action "left" stays, action "right" goes to 1
    # State 1: absorbing (target)
    return make_mdp(
        n_states=2,
        action_transitions={
            0: {"left": [0.9, 0.1], "right": [0.3, 0.7]},
            1: {"stay": [0.0, 1.0]},
        },
        rewards={0: {"left": 1.0, "right": 2.0}, 1: {"stay": 0.0}},
    )


def grid_mdp():
    """Simple 4-state grid MDP.
    States: 0(start) 1
            2        3(goal)
    Actions: up, down, left, right (deterministic or near-deterministic)
    """
    return make_mdp(
        n_states=4,
        action_transitions={
            0: {"right": [0.0, 1.0, 0.0, 0.0], "down": [0.0, 0.0, 1.0, 0.0]},
            1: {"left": [1.0, 0.0, 0.0, 0.0], "down": [0.0, 0.0, 0.0, 1.0]},
            2: {"right": [0.0, 0.0, 0.0, 1.0], "up": [1.0, 0.0, 0.0, 0.0]},
            3: {"stay": [0.0, 0.0, 0.0, 1.0]},
        },
        rewards={
            0: {"right": -1.0, "down": -1.0},
            1: {"left": -1.0, "down": 10.0},
            2: {"right": 10.0, "up": -1.0},
            3: {"stay": 0.0},
        },
    )


def stochastic_mdp():
    """3-state MDP with stochastic transitions."""
    return make_mdp(
        n_states=3,
        action_transitions={
            0: {"risky": [0.0, 0.8, 0.2], "safe": [0.0, 0.5, 0.5]},
            1: {"a": [0.0, 0.0, 1.0]},
            2: {"stay": [0.0, 0.0, 1.0]},
        },
        rewards={0: {"risky": 0.0, "safe": 0.0}, 1: {"a": 5.0}, 2: {"stay": 0.0}},
    )


def gambler_mdp(n=5, p=0.4):
    """Gambler's ruin as MDP: states 0..n, bet 1 at each step.
    State 0 and n are absorbing. State n is the goal (win).
    """
    at = {}
    for s in range(n + 1):
        if s == 0 or s == n:
            row = [0.0] * (n + 1)
            row[s] = 1.0
            at[s] = {"stay": row}
        else:
            bet_row = [0.0] * (n + 1)
            bet_row[s + 1] = p
            bet_row[s - 1] = 1 - p
            at[s] = {"bet": bet_row}
    return make_mdp(n_states=n + 1, action_transitions=at)


# -----------------------------------------------------------------------
# 1. MDP construction and validation
# -----------------------------------------------------------------------

class TestMDPConstruction:
    def test_make_mdp_basic(self):
        mdp = simple_mdp()
        assert mdp.n_states == 2
        assert len(mdp.actions[0]) == 2
        assert len(mdp.actions[1]) == 1
        assert mdp.validate() == []

    def test_make_mdp_labels(self):
        mdp = make_mdp(2, {0: {"a": [0.5, 0.5]}, 1: {"b": [0.0, 1.0]}},
                       state_labels=["start", "end"])
        assert mdp.state_labels == ["start", "end"]

    def test_validation_catches_bad_probs(self):
        mdp = MDP(n_states=2, actions=[["a"], ["b"]],
                  transition=[[[0.5, 0.3]], [[0.0, 1.0]]])
        errors = mdp.validate()
        assert any("sum" in e for e in errors)

    def test_is_mc(self):
        mdp = make_mdp(2, {0: {"only": [0.5, 0.5]}, 1: {"only": [0.0, 1.0]}})
        assert mdp.is_mc()
        mdp2 = simple_mdp()
        assert not mdp2.is_mc()

    def test_default_rewards(self):
        mdp = make_mdp(2, {0: {"a": [0.5, 0.5]}, 1: {"b": [0.0, 1.0]}})
        assert mdp.rewards[0] == [0.0]
        assert mdp.rewards[1] == [0.0]

    def test_ap_labels(self):
        mdp = make_mdp(2, {0: {"a": [0.5, 0.5]}, 1: {"b": [0.0, 1.0]}},
                       ap_labels={1: {"goal"}})
        assert mdp.ap_labels[1] == {"goal"}


# -----------------------------------------------------------------------
# 2. MDP to MC conversion
# -----------------------------------------------------------------------

class TestMDPtoMC:
    def test_induced_mc_simple(self):
        mdp = simple_mdp()
        policy = Policy({0: 1, 1: 0})  # "right" at state 0
        mc = mdp_to_mc(mdp, policy)
        assert mc.n_states == 2
        assert isclose(mc.transition[0][1], 0.7)

    def test_induced_mc_grid(self):
        mdp = grid_mdp()
        # Policy: right at 0, down at 1, right at 2, stay at 3
        policy = Policy({0: 0, 1: 1, 2: 0, 3: 0})
        mc = mdp_to_mc(mdp, policy)
        assert mc.n_states == 4
        assert isclose(mc.transition[0][1], 1.0)
        assert isclose(mc.transition[1][3], 1.0)


# -----------------------------------------------------------------------
# 3. Value iteration
# -----------------------------------------------------------------------

class TestValueIteration:
    def test_simple_mdp_converges(self):
        mdp = simple_mdp()
        result = value_iteration(mdp, discount=0.9, objective=Objective.MAXIMIZE,
                                 terminal_states={1})
        assert result.converged
        # "left" (reward 1, stay 0.9) accumulates more than "right" (reward 2, leave 0.7)
        # V(0) = 1 + 0.9*0.9*V(0) => V(0) = 1/0.19 = 5.26
        assert result.policy.get_action(0) == 0  # "left"
        assert result.values[0] > 5.0

    def test_grid_mdp_optimal(self):
        mdp = grid_mdp()
        result = value_iteration(mdp, discount=0.9, objective=Objective.MAXIMIZE,
                                 terminal_states={3})
        assert result.converged
        # From state 0, should prefer either right->down or down->right to reach goal
        assert result.values[0] > 0

    def test_minimize_objective(self):
        mdp = simple_mdp()
        result_max = value_iteration(mdp, discount=0.9, terminal_states={1})
        result_min = value_iteration(mdp, discount=0.9, objective=Objective.MINIMIZE,
                                     terminal_states={1})
        # Min value <= max value
        assert result_min.values[0] <= result_max.values[0] + 1e-9

    def test_terminal_states_have_zero_value(self):
        mdp = simple_mdp()
        result = value_iteration(mdp, discount=0.9, terminal_states={1})
        assert result.values[1] == 0.0

    def test_discount_factor_effect(self):
        mdp = simple_mdp()
        r1 = value_iteration(mdp, discount=0.5, terminal_states={1})
        r2 = value_iteration(mdp, discount=0.99, terminal_states={1})
        # Higher discount = higher value (future matters more)
        assert r2.values[0] >= r1.values[0] - 1e-6


# -----------------------------------------------------------------------
# 4. Q-values
# -----------------------------------------------------------------------

class TestQValues:
    def test_q_values_simple(self):
        mdp = simple_mdp()
        result = value_iteration(mdp, discount=0.9, terminal_states={1})
        qv = q_values(mdp, result.values, discount=0.9, state=0)
        assert "left" in qv
        assert "right" in qv
        # Left accumulates more reward by staying (1 + 0.9*0.9*V vs 2 + 0.9*0.3*V)
        assert qv["left"] > qv["right"]

    def test_q_values_match_policy(self):
        mdp = simple_mdp()
        result = value_iteration(mdp, discount=0.9, terminal_states={1})
        qv = q_values(mdp, result.values, discount=0.9, state=0)
        # Value = max Q
        assert isclose(result.values[0], max(qv.values()), rel_tol=1e-6)


# -----------------------------------------------------------------------
# 5. Policy evaluation
# -----------------------------------------------------------------------

class TestPolicyEvaluation:
    def test_evaluate_optimal_matches_vi(self):
        mdp = simple_mdp()
        vi = value_iteration(mdp, discount=0.9, terminal_states={1})
        pe = evaluate_policy(mdp, vi.policy, discount=0.9, terminal_states={1})
        for s in range(mdp.n_states):
            assert isclose(vi.values[s], pe[s], rel_tol=1e-5)

    def test_suboptimal_policy_lower_value(self):
        mdp = simple_mdp()
        vi = value_iteration(mdp, discount=0.9, terminal_states={1})
        # Force suboptimal: "left" at state 0
        subopt = Policy({0: 0, 1: 0})
        pe = evaluate_policy(mdp, subopt, discount=0.9, terminal_states={1})
        assert pe[0] < vi.values[0] + 1e-6


# -----------------------------------------------------------------------
# 6. Reachability
# -----------------------------------------------------------------------

class TestReachability:
    def test_max_reachability_simple(self):
        mdp = simple_mdp()
        result = reachability(mdp, {1}, Objective.MAXIMIZE)
        # From state 0, we can always reach state 1 (both actions have > 0 prob)
        assert result.probabilities[0] > 0.99
        assert isclose(result.probabilities[1], 1.0)

    def test_min_reachability_simple(self):
        mdp = simple_mdp()
        result = reachability(mdp, {1}, Objective.MINIMIZE)
        # Even minimizing, both actions reach 1 with > 0 prob, so reach prob = 1
        assert result.probabilities[0] > 0.99

    def test_gambler_reachability(self):
        mdp = gambler_mdp(n=4, p=0.5)
        result = reachability(mdp, {4}, Objective.MAXIMIZE)
        # Fair game: from state 2, reach prob = 2/4 = 0.5
        assert isclose(result.probabilities[2], 0.5, rel_tol=0.05)
        assert isclose(result.probabilities[0], 0.0)
        assert isclose(result.probabilities[4], 1.0)

    def test_unreachable_stays_zero(self):
        # State 0 cannot reach state 2
        mdp = make_mdp(3, {
            0: {"a": [0.5, 0.5, 0.0]},
            1: {"a": [0.5, 0.5, 0.0]},
            2: {"a": [0.0, 0.0, 1.0]},
        })
        result = reachability(mdp, {2}, Objective.MAXIMIZE)
        assert isclose(result.probabilities[0], 0.0)
        assert isclose(result.probabilities[1], 0.0)


# -----------------------------------------------------------------------
# 7. Expected steps
# -----------------------------------------------------------------------

class TestExpectedSteps:
    def test_deterministic_steps(self):
        # Linear chain: 0 -> 1 -> 2 (absorbing)
        mdp = make_mdp(3, {
            0: {"go": [0.0, 1.0, 0.0]},
            1: {"go": [0.0, 0.0, 1.0]},
            2: {"stay": [0.0, 0.0, 1.0]},
        })
        steps, policy = expected_steps(mdp, {2})
        assert isclose(steps[0], 2.0, rel_tol=0.01)
        assert isclose(steps[1], 1.0, rel_tol=0.01)
        assert isclose(steps[2], 0.0)

    def test_unreachable_inf(self):
        mdp = make_mdp(3, {
            0: {"a": [1.0, 0.0, 0.0]},  # self-loop
            1: {"a": [0.0, 0.0, 1.0]},
            2: {"stay": [0.0, 0.0, 1.0]},
        })
        steps, _ = expected_steps(mdp, {2})
        assert steps[0] == float('inf')


# -----------------------------------------------------------------------
# 8. SMT verification: reachability bound
# -----------------------------------------------------------------------

class TestVerifyReachability:
    def test_satisfied_bound(self):
        mdp = simple_mdp()
        result = verify_reachability_bound(mdp, 0, {1}, 0.5)
        assert result.verified

    def test_violated_bound(self):
        # State 0 self-loops, cannot reach 1
        mdp = make_mdp(2, {0: {"stay": [1.0, 0.0]}, 1: {"stay": [0.0, 1.0]}})
        result = verify_reachability_bound(mdp, 0, {1}, 0.5)
        assert not result.verified

    def test_exact_bound(self):
        mdp = gambler_mdp(n=4, p=0.5)
        result = verify_reachability_bound(mdp, 2, {4}, 0.49)
        assert result.verified


# -----------------------------------------------------------------------
# 9. Policy optimality verification
# -----------------------------------------------------------------------

class TestPolicyOptimality:
    def test_optimal_policy_verified(self):
        mdp = simple_mdp()
        vi = value_iteration(mdp, discount=0.9, terminal_states={1})
        result = verify_policy_optimality(mdp, vi.policy, discount=0.9,
                                          terminal_states={1})
        assert result.verified

    def test_suboptimal_detected(self):
        mdp = simple_mdp()
        # Force suboptimal: "right" at state 0 (left is actually optimal)
        result = verify_policy_optimality(mdp, Policy({0: 1, 1: 0}),
                                          discount=0.9, terminal_states={1})
        assert not result.verified
        assert result.counterexample is not None


# -----------------------------------------------------------------------
# 10. Reward bound verification
# -----------------------------------------------------------------------

class TestRewardBound:
    def test_reward_bound_satisfied(self):
        mdp = simple_mdp()
        result = verify_reward_bound(mdp, 0, 1.0, discount=0.9,
                                     terminal_states={1})
        assert result.verified

    def test_reward_bound_violated(self):
        mdp = simple_mdp()
        result = verify_reward_bound(mdp, 0, 1000.0, discount=0.9,
                                     terminal_states={1})
        assert not result.verified


# -----------------------------------------------------------------------
# 11. Safety verification
# -----------------------------------------------------------------------

class TestSafety:
    def test_safety_in_safe_region(self):
        # All states safe, trivially satisfied
        mdp = simple_mdp()
        result = verify_safety(mdp, {0, 1}, 0, 0.99, steps=5)
        assert result.verified

    def test_safety_violated(self):
        # Only state 0 is safe, but transitions go to 1
        mdp = simple_mdp()
        result = verify_safety(mdp, {0}, 0, 0.99, steps=10)
        assert not result.verified


# -----------------------------------------------------------------------
# 12. Policy dominance (SMT)
# -----------------------------------------------------------------------

class TestPolicyDominance:
    def test_optimal_dominates_suboptimal(self):
        mdp = simple_mdp()
        vi = value_iteration(mdp, discount=0.9, terminal_states={1})
        subopt = Policy({0: 0, 1: 0})
        result = smt_verify_policy_dominance(mdp, vi.policy, subopt,
                                             discount=0.9, terminal_states={1})
        assert result.verified

    def test_suboptimal_does_not_dominate(self):
        mdp = simple_mdp()
        vi = value_iteration(mdp, discount=0.9, terminal_states={1})
        # "right" (action 1) is suboptimal
        subopt = Policy({0: 1, 1: 0})
        result = smt_verify_policy_dominance(mdp, subopt, vi.policy,
                                             discount=0.9, terminal_states={1})
        # subopt does not dominate optimal
        assert not result.verified


# -----------------------------------------------------------------------
# 13. Bellman optimality verification (SMT)
# -----------------------------------------------------------------------

class TestBellmanOptimality:
    def test_vi_satisfies_bellman(self):
        mdp = simple_mdp()
        vi = value_iteration(mdp, discount=0.9, terminal_states={1})
        result = smt_verify_bellman_optimality(mdp, vi.values, discount=0.9,
                                               terminal_states={1})
        assert result.verified

    def test_wrong_values_fail_bellman(self):
        mdp = simple_mdp()
        result = smt_verify_bellman_optimality(mdp, [999.0, 0.0], discount=0.9,
                                               terminal_states={1})
        assert not result.verified


# -----------------------------------------------------------------------
# 14. Long-run average reward
# -----------------------------------------------------------------------

class TestLongRunAverage:
    def test_ergodic_chain_average(self):
        # 2-state with positive rewards
        mdp = make_mdp(2, {
            0: {"a": [0.3, 0.7]},
            1: {"a": [0.6, 0.4]},
        }, rewards={0: {"a": 2.0}, 1: {"a": 1.0}})
        policy = Policy({0: 0, 1: 0})
        avg = long_run_average_reward(mdp, policy)
        assert avg is not None
        assert avg > 0


# -----------------------------------------------------------------------
# 15. Policy iteration
# -----------------------------------------------------------------------

class TestPolicyIteration:
    def test_matches_value_iteration(self):
        mdp = simple_mdp()
        vi = value_iteration(mdp, discount=0.9, terminal_states={1})
        pi = policy_iteration(mdp, discount=0.9, terminal_states={1})
        assert pi.converged
        for s in range(mdp.n_states):
            assert isclose(vi.values[s], pi.values[s], rel_tol=1e-4)

    def test_grid_mdp_policy_iteration(self):
        mdp = grid_mdp()
        pi = policy_iteration(mdp, discount=0.9, terminal_states={3})
        assert pi.converged
        assert pi.values[0] > 0


# -----------------------------------------------------------------------
# 16. Interval MDP comparison
# -----------------------------------------------------------------------

class TestIntervalMDPComparison:
    def test_comparison_runs(self):
        mdp = simple_mdp()
        result = compare_with_interval_mdp(mdp, {1}, epsilon=0.05)
        assert result["interval_feasible"]
        # Interval should be wider than point
        for s in range(mdp.n_states):
            assert result["interval_min_reach"][s] <= result["mdp_max_reach"][s] + 1e-6
            assert result["interval_max_reach"][s] >= result["mdp_min_reach"][s] - 1e-6

    def test_zero_epsilon_matches(self):
        mdp = simple_mdp()
        result = compare_with_interval_mdp(mdp, {1}, epsilon=0.0)
        assert result["interval_feasible"]
        for s in range(mdp.n_states):
            assert isclose(result["interval_min_reach"][s],
                           result["mdp_min_reach"][s], abs_tol=0.05)


# -----------------------------------------------------------------------
# 17. Full analysis
# -----------------------------------------------------------------------

class TestAnalyzeMDP:
    def test_full_analysis(self):
        mdp = simple_mdp()
        result = analyze_mdp(mdp, discount=0.9, terminal_states={1})
        assert result["valid"]
        assert result["n_states"] == 2
        assert result["value_iteration"]["converged"]
        assert result["policy_iteration"]["converged"]

    def test_invalid_mdp(self):
        mdp = MDP(n_states=2, actions=[["a"], ["b"]],
                  transition=[[[0.5, 0.3]], [[0.0, 1.0]]])
        result = analyze_mdp(mdp)
        assert not result["valid"]


# -----------------------------------------------------------------------
# 18. Batch verification
# -----------------------------------------------------------------------

class TestVerifyMDP:
    def test_batch_properties(self):
        mdp = simple_mdp()
        props = [
            {"type": "reachability", "start": 0, "targets": [1], "min_prob": 0.5},
            {"type": "reward_bound", "start": 0, "min_reward": 0.5},
            {"type": "safety", "safe_states": [0, 1], "start": 0,
             "min_prob": 0.99, "steps": 3},
        ]
        results = verify_mdp(mdp, props, discount=0.9, terminal_states={1})
        assert len(results) == 3
        assert results[0].verified  # reachability
        assert results[1].verified  # reward bound
        assert results[2].verified  # safety (all states safe)

    def test_unknown_property_type(self):
        mdp = simple_mdp()
        results = verify_mdp(mdp, [{"type": "nonexistent"}])
        assert len(results) == 1
        assert not results[0].verified


# -----------------------------------------------------------------------
# 19. MDP to interval MDP conversion
# -----------------------------------------------------------------------

class TestMDPtoIntervalMDP:
    def test_zero_epsilon(self):
        mdp = simple_mdp()
        imdp = mdp_to_interval_mdp(mdp, epsilon=0.0)
        assert imdp.n_states == 2
        # Point intervals
        for s in range(imdp.n_states):
            for a_idx in range(len(imdp.actions[s])):
                for t in range(imdp.n_states):
                    iv = imdp.transition[s][a_idx][t]
                    assert isclose(iv.lo, iv.hi, abs_tol=1e-9)

    def test_positive_epsilon(self):
        mdp = simple_mdp()
        imdp = mdp_to_interval_mdp(mdp, epsilon=0.1)
        # Intervals should be wider
        iv = imdp.transition[0][0][0]  # state 0, action "left", target 0
        assert iv.hi - iv.lo >= 0.19  # 2*epsilon (approximately)


# -----------------------------------------------------------------------
# 20. Stochastic MDP
# -----------------------------------------------------------------------

class TestStochasticMDP:
    def test_stochastic_reachability(self):
        mdp = stochastic_mdp()
        result = reachability(mdp, {2}, Objective.MAXIMIZE)
        assert isclose(result.probabilities[0], 1.0, abs_tol=0.01)

    def test_risky_vs_safe_policy(self):
        mdp = stochastic_mdp()
        # Risky gets more reward from state 1 (via path 0->1->2: 0 + 5 = 5)
        # Both paths reach 2 anyway
        vi = value_iteration(mdp, discount=0.9, terminal_states={2})
        assert vi.converged


# -----------------------------------------------------------------------
# 21. Edge cases
# -----------------------------------------------------------------------

class TestEdgeCases:
    def test_single_state_mdp(self):
        mdp = make_mdp(1, {0: {"stay": [1.0]}}, rewards={0: {"stay": 1.0}})
        vi = value_iteration(mdp, discount=0.5, terminal_states={0})
        assert isclose(vi.values[0], 0.0)

    def test_single_action_mdp(self):
        mdp = make_mdp(2, {
            0: {"go": [0.0, 1.0]},
            1: {"stay": [0.0, 1.0]},
        })
        assert mdp.is_mc()

    def test_self_loop_mdp(self):
        mdp = make_mdp(2, {
            0: {"loop": [1.0, 0.0], "go": [0.0, 1.0]},
            1: {"stay": [0.0, 1.0]},
        })
        result = reachability(mdp, {1}, Objective.MAXIMIZE)
        # Both actions converge to 1.0 reachability; "go" is direct
        assert isclose(result.probabilities[0], 1.0, abs_tol=0.01)
        # Either policy works since both reach with prob 1
        assert result.probabilities[0] > 0.99

    def test_policy_default_action(self):
        p = Policy({})
        assert p.get_action(42) == 0

    def test_randomized_policy(self):
        rp = RandomizedPolicy({0: [0.5, 0.5]})
        assert rp.get_distribution(0) == [0.5, 0.5]
        assert rp.get_distribution(99) == [1.0]
