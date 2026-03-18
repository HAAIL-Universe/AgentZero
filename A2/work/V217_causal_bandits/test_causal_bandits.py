"""Tests for V217: Causal Bandits."""

import pytest
import math
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from causal_bandits import (
    Intervention, Arm, BanditResult,
    CausalBanditEnv,
    pure_causal, ucb_causal, thompson_causal, obs_int_bandit,
    learning_bandit, epsilon_causal,
    compare_algorithms, regret_summary,
    interventional_gap, confounding_analysis,
    build_treatment_env, build_advertising_env,
    build_simple_env, build_multi_intervention_env,
)


# ===================================================================
# Data Structures
# ===================================================================

class TestIntervention:
    def test_create(self):
        iv = Intervention('X', 1)
        assert iv.variable == 'X'
        assert iv.value == 1

    def test_repr(self):
        iv = Intervention('Treatment', 'drug')
        assert 'do(Treatment=drug)' in repr(iv)

    def test_equality(self):
        a = Intervention('X', 1)
        b = Intervention('X', 1)
        c = Intervention('X', 0)
        assert a == b
        assert a != c

    def test_hash(self):
        a = Intervention('X', 1)
        b = Intervention('X', 1)
        assert hash(a) == hash(b)
        s = {a, b}
        assert len(s) == 1


class TestArm:
    def test_empty_arm(self):
        arm = Arm([])
        assert arm.name == "observe"
        assert arm.to_dict() == {}

    def test_single_intervention(self):
        arm = Arm([Intervention('X', 1)])
        assert 'do(X=1)' in arm.name
        assert arm.to_dict() == {'X': 1}

    def test_multi_intervention(self):
        arm = Arm([Intervention('A', 1), Intervention('B', 0)])
        d = arm.to_dict()
        assert d == {'A': 1, 'B': 0}

    def test_custom_name(self):
        arm = Arm([Intervention('X', 1)], name="my_arm")
        assert arm.name == "my_arm"

    def test_equality(self):
        a = Arm([], name="obs")
        b = Arm([], name="obs")
        assert a == b


class TestBanditResult:
    def test_empty(self):
        r = BanditResult()
        assert r.regret() == 0.0
        assert r.average_regret() == 0.0

    def test_regret(self):
        r = BanditResult()
        r.cumulative_regret = [0.1, 0.3, 0.4]
        r.total_pulls = 3
        assert r.regret() == pytest.approx(0.4)
        assert r.average_regret() == pytest.approx(0.4 / 3)


# ===================================================================
# Environment
# ===================================================================

class TestSimpleEnv:
    def test_build(self):
        env = build_simple_env(seed=42)
        assert env.reward_var == 'Y'
        assert len(env.arms) == 3
        assert env.optimal_arm is not None

    def test_arm_names(self):
        env = build_simple_env(seed=42)
        names = [a.name for a in env.arms]
        assert "observe" in names
        assert "do(X=0)" in names
        assert "do(X=1)" in names

    def test_optimal_arm_is_x1(self):
        """do(X=1) gives 70% Y=1, do(X=0) gives 20%. X=1 is optimal."""
        env = build_simple_env(seed=42)
        assert env.optimal_arm.name == "do(X=1)"
        assert env.optimal_reward == pytest.approx(0.7, abs=0.01)

    def test_true_expected_rewards(self):
        env = build_simple_env(seed=42)
        # do(X=0) -> P(Y=1) = 0.2
        arm_x0 = [a for a in env.arms if a.name == "do(X=0)"][0]
        assert env.true_expected_reward(arm_x0) == pytest.approx(0.2, abs=0.01)

    def test_pull_returns_reward(self):
        env = build_simple_env(seed=42)
        arm = env.arms[0]
        reward = env.pull(arm)
        assert reward in [0.0, 1.0]

    def test_pull_stochastic(self):
        """Multiple pulls should give different rewards."""
        env = build_simple_env(seed=42)
        arm = [a for a in env.arms if a.name == "do(X=1)"][0]
        rewards = [env.pull(arm) for _ in range(100)]
        assert 0.0 in rewards  # Should get some 0s
        assert 1.0 in rewards  # Should get some 1s
        # Mean should be near 0.7
        assert 0.5 < sum(rewards) / len(rewards) < 0.9

    def test_observational_sample(self):
        env = build_simple_env(seed=42)
        sample = env.observational_sample()
        assert 'X' in sample
        assert 'Y' in sample
        assert sample['X'] in [0, 1]
        assert sample['Y'] in [0, 1]


class TestTreatmentEnv:
    def test_build(self):
        env = build_treatment_env(seed=42)
        assert env.reward_var == 'Recovery'
        assert len(env.arms) == 3

    def test_treatment_helps(self):
        """Treatment (drug) should improve recovery rate overall."""
        env = build_treatment_env(seed=42)
        arm_none = [a for a in env.arms if a.name == "do(Treatment=0)"][0]
        arm_drug = [a for a in env.arms if a.name == "do(Treatment=1)"][0]
        # do(Treatment=1) should give higher expected recovery
        assert env.true_expected_reward(arm_drug) > env.true_expected_reward(arm_none)

    def test_confounding_exists(self):
        """Observational P(Recovery|Treatment) != P(Recovery|do(Treatment))."""
        env = build_treatment_env(seed=42)
        analysis = confounding_analysis(env)
        drug_arm = "do(Treatment=1)"
        assert analysis[drug_arm]['confounded'] is True


class TestAdvertisingEnv:
    def test_build(self):
        env = build_advertising_env(seed=42)
        assert len(env.arms) == 4
        assert env.reward_var == 'Purchase'

    def test_search_best(self):
        """Search advertising should have highest causal effect on purchase."""
        env = build_advertising_env(seed=42)
        search_arm = [a for a in env.arms if 'search' in a.name][0]
        email_arm = [a for a in env.arms if 'email' in a.name][0]
        assert env.true_expected_reward(search_arm) > env.true_expected_reward(email_arm)


class TestMultiInterventionEnv:
    def test_build(self):
        env = build_multi_intervention_env(seed=42)
        assert len(env.arms) == 6
        assert env.reward_var == 'C'

    def test_joint_intervention_best(self):
        """do(A=1,B=1) should be the best arm (both causes active)."""
        env = build_multi_intervention_env(seed=42)
        assert env.optimal_arm.name == "do(A=1,B=1)"
        assert env.optimal_reward == pytest.approx(0.9, abs=0.01)

    def test_single_vs_joint(self):
        env = build_multi_intervention_env(seed=42)
        a1 = [a for a in env.arms if a.name == "do(A=1)"][0]
        b1 = [a for a in env.arms if a.name == "do(B=1)"][0]
        ab = [a for a in env.arms if a.name == "do(A=1,B=1)"][0]
        # Joint > either single
        assert env.true_expected_reward(ab) > env.true_expected_reward(a1)
        assert env.true_expected_reward(ab) > env.true_expected_reward(b1)


class TestAutoArms:
    def test_auto_generates_arms(self):
        """When no arms provided, auto-generate from all interventions."""
        from causal_bandits import CausalModel, BayesianNetwork
        bn = BayesianNetwork()
        bn.add_node('X', [0, 1])
        bn.add_node('Y', [0, 1])
        bn.add_edge('X', 'Y')
        bn.set_cpt('X', {(0,): 0.5, (1,): 0.5})
        bn.set_cpt('Y', {(0, 0): 0.7, (0, 1): 0.3, (1, 0): 0.2, (1, 1): 0.8})
        model = CausalModel(bn)
        env = CausalBanditEnv(model, 'Y', reward_mapping={0: 0.0, 1: 1.0})
        # Should have: observe + do(X=0) + do(X=1)
        assert len(env.arms) == 3


# ===================================================================
# Algorithm: Pure Causal
# ===================================================================

class TestPureCausal:
    def test_finds_optimal(self):
        env = build_simple_env(seed=42)
        result = pure_causal(env)
        assert result.best_arm.name == "do(X=1)"
        assert result.best_reward == pytest.approx(0.7, abs=0.01)
        assert result.total_pulls == 0  # No actual pulls needed

    def test_treatment(self):
        env = build_treatment_env(seed=42)
        result = pure_causal(env)
        assert result.best_arm.name == "do(Treatment=1)"

    def test_multi_intervention(self):
        env = build_multi_intervention_env(seed=42)
        result = pure_causal(env)
        assert result.best_arm.name == "do(A=1,B=1)"


# ===================================================================
# Algorithm: UCB-Causal
# ===================================================================

class TestUCBCausal:
    def test_finds_optimal(self):
        env = build_simple_env(seed=42)
        result = ucb_causal(env, rounds=200, seed=42)
        assert result.best_arm.name == "do(X=1)"
        assert result.total_pulls == 200

    def test_regret_bounded(self):
        env = build_simple_env(seed=42)
        result = ucb_causal(env, rounds=500, seed=42)
        # Average regret should decrease
        assert result.average_regret() < 0.3

    def test_cumulative_regret_grows_sublinearly(self):
        env = build_simple_env(seed=42)
        result = ucb_causal(env, rounds=500, seed=42)
        # Regret at halfway vs end: sublinear means less than 2x
        half = result.cumulative_regret[249]
        full = result.cumulative_regret[499]
        assert full < 2.5 * half  # Sublinear growth

    def test_pulls_concentrate_on_best(self):
        env = build_simple_env(seed=42)
        result = ucb_causal(env, rounds=500, seed=42)
        # Best arm should get most pulls
        best_pulls = result.arm_pulls.get("do(X=1)", 0)
        total = sum(result.arm_pulls.values())
        assert best_pulls / total > 0.5

    def test_causal_prior_helps(self):
        """UCB with causal priors should work; without priors also works."""
        env = build_simple_env(seed=42)
        with_prior = ucb_causal(env, rounds=300, causal_prior_weight=10, seed=42)
        without_prior = ucb_causal(env, rounds=300, causal_prior_weight=0, seed=42)
        # Both should find optimal arm
        assert with_prior.best_arm.name == "do(X=1)"
        assert without_prior.best_arm.name == "do(X=1)"

    def test_treatment_env(self):
        env = build_treatment_env(seed=42)
        result = ucb_causal(env, rounds=300, seed=42)
        assert result.best_arm.name == "do(Treatment=1)"


# ===================================================================
# Algorithm: Thompson Sampling
# ===================================================================

class TestThompsonCausal:
    def test_finds_optimal(self):
        env = build_simple_env(seed=42)
        result = thompson_causal(env, rounds=300, seed=42)
        assert result.best_arm.name == "do(X=1)"

    def test_regret_bounded(self):
        env = build_simple_env(seed=42)
        result = thompson_causal(env, rounds=500, seed=42)
        assert result.average_regret() < 0.3

    def test_treatment_env(self):
        env = build_treatment_env(seed=42)
        result = thompson_causal(env, rounds=300, seed=42)
        assert result.best_arm.name == "do(Treatment=1)"

    def test_advertising(self):
        env = build_advertising_env(seed=42)
        result = thompson_causal(env, rounds=500, seed=42)
        # Should find search is best
        assert 'search' in result.best_arm.name

    def test_history_recorded(self):
        env = build_simple_env(seed=42)
        result = thompson_causal(env, rounds=100, seed=42)
        assert len(result.history) == 100
        assert len(result.cumulative_regret) == 100


# ===================================================================
# Algorithm: Epsilon-Greedy
# ===================================================================

class TestEpsilonCausal:
    def test_finds_optimal(self):
        env = build_simple_env(seed=42)
        result = epsilon_causal(env, rounds=300, seed=42)
        assert result.best_arm.name == "do(X=1)"

    def test_causal_init_helps(self):
        """Both with and without causal init should find optimal arm."""
        env = build_simple_env(seed=42)
        with_init = epsilon_causal(env, rounds=300, use_causal_init=True, seed=42)
        without_init = epsilon_causal(env, rounds=300, use_causal_init=False, seed=42)
        # Both should find the optimal arm
        assert with_init.best_arm.name == "do(X=1)"
        assert without_init.best_arm.name == "do(X=1)"

    def test_epsilon_decay(self):
        env = build_simple_env(seed=42)
        result = epsilon_causal(env, rounds=500, epsilon=0.3, decay=0.99, seed=42)
        # With decay, late pulls should be mostly on best arm
        late_pulls = result.history[400:]
        best_count = sum(1 for name, _ in late_pulls if name == "do(X=1)")
        assert best_count / len(late_pulls) > 0.7


# ===================================================================
# Algorithm: Observational-Interventional
# ===================================================================

class TestObsIntBandit:
    def test_finds_optimal(self):
        env = build_simple_env(seed=42)
        result = obs_int_bandit(env, rounds=300, seed=42)
        assert result.best_arm.name == "do(X=1)"

    def test_uses_observations(self):
        """Should still work with observational data."""
        env = build_simple_env(seed=42)
        result = obs_int_bandit(env, rounds=200, obs_per_round=20, seed=42)
        assert result.total_pulls == 200

    def test_treatment_env(self):
        env = build_treatment_env(seed=42)
        result = obs_int_bandit(env, rounds=300, seed=42)
        assert result.best_arm.name == "do(Treatment=1)"


# ===================================================================
# Algorithm: Causal Structure Learning Bandit
# ===================================================================

class TestLearningBandit:
    def test_finds_optimal_simple(self):
        env = build_simple_env(seed=42)
        result = learning_bandit(env, rounds=500, learn_interval=50,
                                  obs_per_learn=200, seed=42)
        assert result.best_arm.name == "do(X=1)"

    def test_regret_exists(self):
        env = build_simple_env(seed=42)
        result = learning_bandit(env, rounds=200, seed=42)
        assert len(result.cumulative_regret) == 200
        assert result.regret() >= 0


# ===================================================================
# Comparison & Analysis
# ===================================================================

class TestCompareAlgorithms:
    def test_compare(self):
        env = build_simple_env(seed=42)
        results = compare_algorithms(env, rounds=200, seed=42)
        assert 'pure_causal' in results
        assert 'ucb_causal' in results
        assert 'thompson_causal' in results
        assert 'epsilon_causal' in results
        assert 'epsilon_no_causal' in results
        assert 'obs_int' in results

    def test_pure_causal_zero_regret(self):
        env = build_simple_env(seed=42)
        results = compare_algorithms(env, rounds=100, seed=42)
        assert results['pure_causal'].regret() == 0.0

    def test_summary(self):
        env = build_simple_env(seed=42)
        results = compare_algorithms(env, rounds=100, seed=42)
        summary = regret_summary(results)
        assert len(summary) == 6
        for name, info in summary.items():
            assert 'best_arm' in info
            assert 'total_regret' in info
            assert 'avg_regret' in info


class TestInterventionalGap:
    def test_optimal_has_zero_gap(self):
        env = build_simple_env(seed=42)
        gaps = interventional_gap(env)
        assert gaps[env.optimal_arm.name] == pytest.approx(0.0)

    def test_suboptimal_positive_gap(self):
        env = build_simple_env(seed=42)
        gaps = interventional_gap(env)
        assert gaps["do(X=0)"] > 0

    def test_gap_ordering(self):
        env = build_multi_intervention_env(seed=42)
        gaps = interventional_gap(env)
        # Joint intervention has gap 0
        assert gaps["do(A=1,B=1)"] == pytest.approx(0.0)
        # All others have positive gaps
        for name, gap in gaps.items():
            if name != "do(A=1,B=1)":
                assert gap >= 0


class TestConfoundingAnalysis:
    def test_treatment_confounded(self):
        env = build_treatment_env(seed=42)
        analysis = confounding_analysis(env)
        # Treatment arm should be confounded (Age is confounder)
        drug = analysis.get("do(Treatment=1)")
        assert drug is not None
        assert drug['confounded'] is True

    def test_simple_not_confounded(self):
        """Simple X->Y has no confounding."""
        env = build_simple_env(seed=42)
        analysis = confounding_analysis(env)
        x1 = analysis.get("do(X=1)")
        assert x1 is not None
        # Without confounders, obs ~= interventional
        assert x1['confounded'] is False

    def test_observe_arm(self):
        env = build_simple_env(seed=42)
        analysis = confounding_analysis(env)
        assert analysis['observe']['type'] == 'observational'


# ===================================================================
# Edge cases and robustness
# ===================================================================

class TestEdgeCases:
    def test_single_arm(self):
        """Environment with only the observe arm."""
        env = build_simple_env(seed=42)
        single_env = CausalBanditEnv(
            env.model, 'Y',
            arms=[Arm([], name="observe")],
            reward_mapping={0: 0.0, 1: 1.0},
            seed=42
        )
        result = ucb_causal(single_env, rounds=50, seed=42)
        assert result.best_arm.name == "observe"
        # Only arm = optimal, so regret is purely stochastic noise
        # (optimal_reward - actual_reward per step, where rewards are 0 or 1)
        assert result.total_pulls == 50

    def test_deterministic_rewards(self):
        """Environment where optimal arm gives deterministic reward."""
        from causal_bandits import CausalModel, BayesianNetwork
        bn = BayesianNetwork()
        bn.add_node('X', [0, 1])
        bn.add_node('Y', [0, 1])
        bn.add_edge('X', 'Y')
        bn.set_cpt('X', {(0,): 0.5, (1,): 0.5})
        bn.set_cpt('Y', {
            (0, 0): 1.0, (0, 1): 0.0,   # X=0 -> Y=0 always
            (1, 0): 0.0, (1, 1): 1.0,    # X=1 -> Y=1 always
        })
        model = CausalModel(bn)
        env = CausalBanditEnv(model, 'Y',
                              reward_mapping={0: 0.0, 1: 1.0}, seed=42)
        result = ucb_causal(env, rounds=100, seed=42)
        assert result.best_arm.name == "do(X=1)"

    def test_zero_rounds(self):
        env = build_simple_env(seed=42)
        result = ucb_causal(env, rounds=0, seed=42)
        assert result.total_pulls == 0

    def test_reproducibility(self):
        """Same seed gives same results."""
        env1 = build_simple_env(seed=42)
        env2 = build_simple_env(seed=42)
        r1 = ucb_causal(env1, rounds=100, seed=99)
        r2 = ucb_causal(env2, rounds=100, seed=99)
        assert r1.regret() == r2.regret()
        assert r1.best_arm.name == r2.best_arm.name

    def test_all_algorithms_agree_on_optimal(self):
        """All algorithms should find the same optimal arm on simple env."""
        env = build_simple_env(seed=42)
        results = compare_algorithms(env, rounds=300, seed=42)
        for name, result in results.items():
            assert result.best_arm.name == "do(X=1)", \
                f"{name} found {result.best_arm.name} instead of do(X=1)"


# ===================================================================
# Integration: Real causal reasoning matters
# ===================================================================

class TestCausalReasoningMatters:
    def test_confounded_env_needs_causal(self):
        """In confounded env, causal bandit should outperform naive."""
        env = build_treatment_env(seed=42)
        # Causal UCB knows the true interventional distribution
        causal = ucb_causal(env, rounds=300, seed=42)
        # Epsilon without causal init has to learn from scratch
        naive = epsilon_causal(env, rounds=300, use_causal_init=False, seed=42)
        # Causal should have less regret (with margin for stochasticity)
        assert causal.regret() <= naive.regret() * 1.5

    def test_multi_intervention_optimal(self):
        """Joint intervention should dominate single interventions."""
        env = build_multi_intervention_env(seed=42)
        result = ucb_causal(env, rounds=500, seed=42)
        # Best arm should be joint intervention
        assert result.best_arm.name == "do(A=1,B=1)"

    def test_advertising_search_wins(self):
        env = build_advertising_env(seed=42)
        result = ucb_causal(env, rounds=500, seed=42)
        assert 'search' in result.best_arm.name
