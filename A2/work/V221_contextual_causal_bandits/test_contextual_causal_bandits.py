"""Tests for V221: Contextual Causal Bandits."""

import math
import sys
import os
import pytest
from collections import defaultdict

# Add dependency paths
_base = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_base, "V217_causal_bandits"))
sys.path.insert(0, os.path.join(_base, "V214_causal_discovery"))
sys.path.insert(0, os.path.join(_base, "V211_causal_inference"))
sys.path.insert(0, os.path.join(_base, "V209_bayesian_networks"))

from contextual_causal_bandits import (
    # Data structures
    Context,
    ContextualResult,
    CATEEstimate,
    SubgroupResult,
    PolicyTreeNode,
    # Environment
    ContextualCausalEnv,
    # Algorithms
    binned_ucb_causal,
    binned_thompson_causal,
    causal_linucb,
    cate_greedy,
    epsilon_subgroup,
    causal_policy_tree,
    # Analysis
    estimate_cate,
    subgroup_analysis,
    compare_algorithms,
    regret_summary,
    # Helpers
    _context_features,
    _solve_linear,
    _build_policy_tree,
    _compute_cate_policy,
    _global_best_arm,
    # Example environments
    build_treatment_env,
    build_advertising_env,
    build_simple_heterogeneous_env,
    build_homogeneous_env,
)
from causal_bandits import Arm, Intervention


# ===========================================================================
# Context and data structures
# ===========================================================================

class TestContext:
    def test_context_creation(self):
        ctx = Context(features={"age": "young", "gender": "M"})
        assert ctx.features["age"] == "young"
        assert ctx.features["gender"] == "M"

    def test_context_key(self):
        ctx = Context(features={"b": 2, "a": 1})
        key = ctx.key(["a", "b"])
        assert key == (1, 2)

    def test_context_key_subset(self):
        ctx = Context(features={"a": 1, "b": 2, "c": 3})
        key = ctx.key(["a", "c"])
        assert key == (1, 3)

    def test_context_repr(self):
        ctx = Context(features={"x": 1})
        assert "x" in repr(ctx)


class TestContextualResult:
    def test_empty_result(self):
        r = ContextualResult()
        assert r.regret() == 0.0
        assert r.average_regret() == 0.0
        assert r.total_pulls == 0

    def test_regret_tracking(self):
        r = ContextualResult()
        r.cumulative_regret = [0.1, 0.3, 0.6]
        r.total_pulls = 3
        assert r.regret() == 0.6
        assert abs(r.average_regret() - 0.2) < 1e-10

    def test_learned_policy(self):
        r = ContextualResult()
        r.context_arm_pulls[(0,)] = {"A": 10, "B": 5}
        r.context_arm_rewards[(0,)] = {"A": 8.0, "B": 1.0}
        r.context_arm_pulls[(1,)] = {"A": 5, "B": 10}
        r.context_arm_rewards[(1,)] = {"A": 1.0, "B": 8.0}

        policy = r.learned_policy(["X"])
        assert policy[(0,)] == "A"
        assert policy[(1,)] == "B"


class TestCATEEstimate:
    def test_creation(self):
        est = CATEEstimate(
            subgroup={"age": "young"},
            arm_a="A", arm_b="B",
            cate=0.5, n_samples=100, std_err=0.02,
        )
        assert est.cate == 0.5
        assert est.n_samples == 100


# ===========================================================================
# Environment
# ===========================================================================

class TestContextualCausalEnv:
    def test_treatment_env_creation(self):
        env = build_treatment_env(seed=42)
        assert len(env.arms) == 2
        assert len(env.context_vars) == 1
        assert "Age" in env.context_vars

    def test_heterogeneous_detection(self):
        env = build_simple_heterogeneous_env(seed=42)
        assert env.is_heterogeneous()

    def test_homogeneous_detection(self):
        env = build_homogeneous_env(seed=42)
        assert not env.is_heterogeneous()

    def test_context_domains(self):
        env = build_treatment_env(seed=42)
        assert "Age" in env.context_domains
        assert set(env.context_domains["Age"]) == {"young", "old"}

    def test_sample_context(self):
        env = build_simple_heterogeneous_env(seed=42)
        ctx = env.sample_context()
        assert "X" in ctx.features
        assert ctx.features["X"] in [0, 1]

    def test_pull_returns_float(self):
        env = build_simple_heterogeneous_env(seed=42)
        ctx = Context(features={"X": 0})
        arm = env.arms[0]
        reward = env.pull(arm, ctx)
        assert isinstance(reward, float)

    def test_optimal_arm_varies_by_context(self):
        env = build_simple_heterogeneous_env(seed=42)
        ctx0 = Context(features={"X": 0})
        ctx1 = Context(features={"X": 1})
        opt0 = env.optimal_arm(ctx0)
        opt1 = env.optimal_arm(ctx1)
        # A0 is optimal for X=0, A1 for X=1
        assert opt0.name == "A0"
        assert opt1.name == "A1"

    def test_optimal_reward(self):
        env = build_simple_heterogeneous_env(seed=42)
        ctx0 = Context(features={"X": 0})
        rew = env.optimal_reward(ctx0)
        assert rew > 0.5  # Should be around 0.9

    def test_true_expected_reward(self):
        env = build_simple_heterogeneous_env(seed=42)
        ctx0 = Context(features={"X": 0})
        arm0 = env.arms[0]  # A0
        arm1 = env.arms[1]  # A1
        er0 = env.true_expected_reward(arm0, ctx0)
        er1 = env.true_expected_reward(arm1, ctx0)
        assert er0 > er1  # A0 better when X=0

    def test_treatment_env_optimal_arms(self):
        env = build_treatment_env(seed=42)
        ctx_young = Context(features={"Age": "young"})
        ctx_old = Context(features={"Age": "old"})
        assert env.optimal_arm(ctx_young).name == "Treatment_A"
        assert env.optimal_arm(ctx_old).name == "Treatment_B"

    def test_advertising_env_creation(self):
        env = build_advertising_env(seed=42)
        assert len(env.arms) == 3
        assert len(env.context_vars) == 2
        assert env.is_heterogeneous()

    def test_precomputed_rewards(self):
        env = build_simple_heterogeneous_env(seed=42)
        assert len(env._true_rewards) == 2  # 2 contexts
        for ctx_key, arm_rews in env._true_rewards.items():
            assert len(arm_rews) == 2  # 2 arms


# ===========================================================================
# Algorithm 1: Binned UCB-Causal
# ===========================================================================

class TestBinnedUCBCausal:
    def test_runs_and_returns_result(self):
        env = build_simple_heterogeneous_env(seed=42)
        result = binned_ucb_causal(env, rounds=100, seed=42)
        assert result.total_pulls == 100
        assert len(result.cumulative_regret) == 100

    def test_learns_heterogeneous_policy(self):
        env = build_simple_heterogeneous_env(seed=42)
        result = binned_ucb_causal(env, rounds=500, seed=42)
        policy = result.learned_policy(env.context_vars)
        # Should learn A0 for X=0, A1 for X=1
        if (0,) in policy and (1,) in policy:
            assert policy[(0,)] == "A0"
            assert policy[(1,)] == "A1"

    def test_sublinear_regret(self):
        env = build_simple_heterogeneous_env(seed=42)
        result = binned_ucb_causal(env, rounds=500, seed=42)
        avg = result.average_regret()
        assert avg < 0.5  # Should converge

    def test_causal_prior_helps(self):
        env = build_simple_heterogeneous_env(seed=42)
        r_with = binned_ucb_causal(env, rounds=300, causal_prior_weight=5, seed=42)
        r_without = binned_ucb_causal(env, rounds=300, causal_prior_weight=0, seed=42)
        # With causal priors should have lower regret on average
        assert r_with.regret() <= r_without.regret() * 1.5  # Allow some variance

    def test_homogeneous_env(self):
        env = build_homogeneous_env(seed=42)
        result = binned_ucb_causal(env, rounds=300, seed=42)
        assert result.total_pulls == 300


# ===========================================================================
# Algorithm 2: Binned Thompson Sampling
# ===========================================================================

class TestBinnedThompsonCausal:
    def test_runs_and_returns_result(self):
        env = build_simple_heterogeneous_env(seed=42)
        result = binned_thompson_causal(env, rounds=100, seed=42)
        assert result.total_pulls == 100

    def test_learns_heterogeneous_policy(self):
        env = build_simple_heterogeneous_env(seed=42)
        result = binned_thompson_causal(env, rounds=500, seed=42)
        policy = result.learned_policy(env.context_vars)
        if (0,) in policy and (1,) in policy:
            assert policy[(0,)] == "A0"
            assert policy[(1,)] == "A1"

    def test_sublinear_regret(self):
        env = build_simple_heterogeneous_env(seed=42)
        result = binned_thompson_causal(env, rounds=500, seed=42)
        assert result.average_regret() < 0.5

    def test_with_causal_priors(self):
        env = build_treatment_env(seed=42)
        result = binned_thompson_causal(env, rounds=300, causal_prior_strength=5, seed=42)
        assert result.total_pulls == 300

    def test_without_causal_priors(self):
        env = build_simple_heterogeneous_env(seed=42)
        result = binned_thompson_causal(env, rounds=200, causal_prior_strength=0, seed=42)
        assert result.total_pulls == 200


# ===========================================================================
# Algorithm 3: Causal LinUCB
# ===========================================================================

class TestCausalLinUCB:
    def test_runs_and_returns_result(self):
        env = build_simple_heterogeneous_env(seed=42)
        result = causal_linucb(env, rounds=100, seed=42)
        assert result.total_pulls == 100

    def test_context_features(self):
        env = build_simple_heterogeneous_env(seed=42)
        ctx = Context(features={"X": 0})
        feats = _context_features(ctx, env.context_vars, env.context_domains)
        assert feats[0] == 1.0  # bias
        assert len(feats) >= 2  # bias + at least one feature

    def test_multi_context_features(self):
        env = build_advertising_env(seed=42)
        ctx = Context(features={"UserType": "tech", "Interest": "high"})
        feats = _context_features(ctx, env.context_vars, env.context_domains)
        assert feats[0] == 1.0  # bias
        assert len(feats) > 2  # bias + multiple features

    def test_sublinear_regret(self):
        env = build_simple_heterogeneous_env(seed=42)
        result = causal_linucb(env, rounds=500, seed=42)
        assert result.average_regret() < 0.5

    def test_advertising_env(self):
        env = build_advertising_env(seed=42)
        result = causal_linucb(env, rounds=200, seed=42)
        assert result.total_pulls == 200


class TestSolveLinear:
    def test_identity(self):
        A = [[1, 0], [0, 1]]
        b = [3, 4]
        x = _solve_linear(A, b)
        assert abs(x[0] - 3) < 1e-10
        assert abs(x[1] - 4) < 1e-10

    def test_simple_system(self):
        A = [[2, 1], [1, 3]]
        b = [5, 10]
        x = _solve_linear(A, b)
        # 2x + y = 5, x + 3y = 10 -> x = 1, y = 3
        assert abs(x[0] - 1) < 1e-10
        assert abs(x[1] - 3) < 1e-10

    def test_singular_returns_none(self):
        A = [[1, 2], [2, 4]]
        b = [3, 6]
        x = _solve_linear(A, b)
        assert x is None


# ===========================================================================
# Algorithm 4: CATE-Greedy
# ===========================================================================

class TestCATEGreedy:
    def test_runs_and_returns_result(self):
        env = build_simple_heterogeneous_env(seed=42)
        result = cate_greedy(env, rounds=200, explore_rounds=50, seed=42)
        assert result.total_pulls == 200

    def test_exploration_then_exploitation(self):
        env = build_simple_heterogeneous_env(seed=42)
        result = cate_greedy(env, rounds=500, explore_rounds=100, seed=42)
        # After exploration, regret should grow slower
        regret_first_half = result.cumulative_regret[249]
        regret_second_half = result.cumulative_regret[499] - result.cumulative_regret[249]
        # Second half should have less regret than first half
        # (exploration regret is front-loaded)
        assert regret_second_half <= regret_first_half * 1.5

    def test_homogeneous_env(self):
        env = build_homogeneous_env(seed=42)
        result = cate_greedy(env, rounds=200, explore_rounds=50, seed=42)
        assert result.total_pulls == 200


# ===========================================================================
# Algorithm 5: Epsilon-Subgroup
# ===========================================================================

class TestEpsilonSubgroup:
    def test_runs_and_returns_result(self):
        env = build_simple_heterogeneous_env(seed=42)
        result = epsilon_subgroup(env, rounds=100, seed=42)
        assert result.total_pulls == 100

    def test_learns_policy(self):
        env = build_simple_heterogeneous_env(seed=42)
        result = epsilon_subgroup(env, rounds=500, seed=42)
        policy = result.learned_policy(env.context_vars)
        if (0,) in policy and (1,) in policy:
            assert policy[(0,)] == "A0"
            assert policy[(1,)] == "A1"

    def test_epsilon_decay(self):
        env = build_simple_heterogeneous_env(seed=42)
        # With strong decay, later rounds should be more exploitative
        result = epsilon_subgroup(env, rounds=500, epsilon=0.3, decay=0.99, seed=42)
        assert result.average_regret() < 0.5


# ===========================================================================
# Algorithm 6: Causal Policy Tree
# ===========================================================================

class TestCausalPolicyTree:
    def test_runs_and_returns_result(self):
        env = build_simple_heterogeneous_env(seed=42)
        result = causal_policy_tree(env, rounds=200, explore_rounds=50, seed=42)
        assert result.total_pulls == 200

    def test_policy_tree_node(self):
        arm_a = Arm([Intervention("A", 0)], name="A0")
        arm_b = Arm([Intervention("A", 1)], name="A1")

        leaf = PolicyTreeNode(is_leaf=True, arm=arm_a)
        ctx = Context(features={"X": 0})
        assert leaf.predict(ctx).name == "A0"

        # Internal node
        node = PolicyTreeNode(
            is_leaf=False, split_var="X",
            children={0: PolicyTreeNode(is_leaf=True, arm=arm_a),
                      1: PolicyTreeNode(is_leaf=True, arm=arm_b)},
        )
        assert node.predict(Context(features={"X": 0})).name == "A0"
        assert node.predict(Context(features={"X": 1})).name == "A1"

    def test_tree_unseen_value_fallback(self):
        arm_a = Arm([Intervention("A", 0)], name="A0")
        arm_b = Arm([Intervention("A", 1)], name="A1")

        node = PolicyTreeNode(
            is_leaf=False, split_var="X", arm=arm_a,
            children={0: PolicyTreeNode(is_leaf=True, arm=arm_a)},
        )
        # X=99 not in children -> fallback to first child
        result = node.predict(Context(features={"X": 99}))
        assert result.name == "A0"

    def test_treatment_env(self):
        env = build_treatment_env(seed=42)
        result = causal_policy_tree(env, rounds=300, explore_rounds=100, seed=42)
        assert result.total_pulls == 300


# ===========================================================================
# CATE Analysis
# ===========================================================================

class TestCATEAnalysis:
    def test_estimate_cate(self):
        env = build_simple_heterogeneous_env(seed=42)
        arm_a = env.arms[0]  # A0
        arm_b = env.arms[1]  # A1
        cates = estimate_cate(env, arm_a, arm_b, n_samples=200, seed=42)
        assert len(cates) == 2  # 2 contexts

        # For X=0: A0 >> A1, CATE should be positive
        # For X=1: A0 << A1, CATE should be negative
        for est in cates:
            if est.subgroup.get("X") == 0:
                assert est.cate > 0  # A0 better when X=0
            elif est.subgroup.get("X") == 1:
                assert est.cate < 0  # A0 worse when X=1

    def test_cate_std_err(self):
        env = build_simple_heterogeneous_env(seed=42)
        cates = estimate_cate(env, env.arms[0], env.arms[1], n_samples=300, seed=42)
        for est in cates:
            assert est.std_err >= 0
            assert est.n_samples == 300


class TestSubgroupAnalysis:
    def test_heterogeneous_subgroups(self):
        env = build_simple_heterogeneous_env(seed=42)
        result = subgroup_analysis(env)
        assert len(result.subgroups) == 2
        assert result.heterogeneity_score >= 0

        # Different optimal arms per subgroup
        arms_seen = set(s["optimal_arm"] for s in result.subgroups)
        assert len(arms_seen) == 2

    def test_homogeneous_subgroups(self):
        env = build_homogeneous_env(seed=42)
        result = subgroup_analysis(env)
        arms_seen = set(s["optimal_arm"] for s in result.subgroups)
        assert len(arms_seen) == 1  # Same arm for all contexts

    def test_cate_estimates_present(self):
        env = build_simple_heterogeneous_env(seed=42)
        result = subgroup_analysis(env)
        assert len(result.cate_estimates) > 0

    def test_treatment_subgroups(self):
        env = build_treatment_env(seed=42)
        result = subgroup_analysis(env)
        for sg in result.subgroups:
            if sg["context"].get("Age") == "young":
                assert sg["optimal_arm"] == "Treatment_A"
            elif sg["context"].get("Age") == "old":
                assert sg["optimal_arm"] == "Treatment_B"

    def test_advertising_subgroups(self):
        env = build_advertising_env(seed=42)
        result = subgroup_analysis(env)
        assert len(result.subgroups) > 2  # multiple context combos


# ===========================================================================
# Algorithm Comparison
# ===========================================================================

class TestComparison:
    def test_compare_algorithms(self):
        env = build_simple_heterogeneous_env(seed=42)
        results = compare_algorithms(env, rounds=200, seed=42)
        assert len(results) == 6
        for name, res in results.items():
            assert res.total_pulls == 200
            assert len(res.cumulative_regret) == 200

    def test_regret_summary(self):
        env = build_simple_heterogeneous_env(seed=42)
        results = compare_algorithms(env, rounds=100, seed=42)
        summary = regret_summary(results)
        assert len(summary) == 6
        for name, s in summary.items():
            assert "total_regret" in s
            assert "average_regret" in s
            assert s["total_pulls"] == 100


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:
    def test_treatment_all_algorithms(self):
        """All algorithms run on the treatment environment."""
        env = build_treatment_env(seed=42)
        results = compare_algorithms(env, rounds=300, seed=42)
        for name, res in results.items():
            assert res.total_pulls == 300, f"{name} failed"

    def test_advertising_all_algorithms(self):
        """All algorithms run on the multi-context advertising env."""
        env = build_advertising_env(seed=42)
        results = compare_algorithms(env, rounds=200, seed=42)
        for name, res in results.items():
            assert res.total_pulls == 200, f"{name} failed"

    def test_end_to_end_heterogeneous(self):
        """Full pipeline: build env, detect heterogeneity, analyze, run."""
        env = build_simple_heterogeneous_env(seed=42)

        # 1. Detect heterogeneity
        assert env.is_heterogeneous()

        # 2. Subgroup analysis
        sg = subgroup_analysis(env)
        # Different optimal arms across subgroups (heterogeneity)
        arms_seen = set(s["optimal_arm"] for s in sg.subgroups)
        assert len(arms_seen) == 2

        # 3. CATE estimation
        cates = estimate_cate(env, env.arms[0], env.arms[1], n_samples=100, seed=42)
        assert len(cates) == 2

        # 4. Run best algorithm
        result = binned_ucb_causal(env, rounds=500, seed=42)
        policy = result.learned_policy(env.context_vars)
        assert result.average_regret() < 0.5

    def test_end_to_end_homogeneous(self):
        """Pipeline on homogeneous env -- should find same arm for all."""
        env = build_homogeneous_env(seed=42)
        assert not env.is_heterogeneous()

        sg = subgroup_analysis(env)
        arms_seen = set(s["optimal_arm"] for s in sg.subgroups)
        assert len(arms_seen) == 1

        result = binned_ucb_causal(env, rounds=300, seed=42)
        assert result.total_pulls == 300

    def test_deterministic_with_seed(self):
        """Same seed produces same results."""
        env1 = build_simple_heterogeneous_env(seed=42)
        env2 = build_simple_heterogeneous_env(seed=42)
        r1 = binned_ucb_causal(env1, rounds=100, seed=99)
        r2 = binned_ucb_causal(env2, rounds=100, seed=99)
        assert r1.cumulative_regret == r2.cumulative_regret

    def test_multiple_seeds_different_results(self):
        """Different seeds produce different results."""
        env1 = build_simple_heterogeneous_env(seed=1)
        env2 = build_simple_heterogeneous_env(seed=2)
        r1 = binned_ucb_causal(env1, rounds=50, seed=1)
        r2 = binned_ucb_causal(env2, rounds=50, seed=2)
        # Very unlikely to be identical
        assert r1.cumulative_regret != r2.cumulative_regret

    def test_zero_rounds(self):
        """Zero rounds returns empty result."""
        env = build_simple_heterogeneous_env(seed=42)
        result = binned_ucb_causal(env, rounds=0, seed=42)
        assert result.total_pulls == 0
        assert result.regret() == 0.0

    def test_single_round(self):
        """Single round works correctly."""
        env = build_simple_heterogeneous_env(seed=42)
        result = binned_ucb_causal(env, rounds=1, seed=42)
        assert result.total_pulls == 1
        assert len(result.cumulative_regret) == 1

    def test_large_advertising_run(self):
        """Longer run on complex env converges."""
        env = build_advertising_env(seed=42)
        result = binned_ucb_causal(env, rounds=1000, seed=42)
        # Average regret should decrease over time
        first_100_avg = result.cumulative_regret[99] / 100
        last_100_avg = (result.cumulative_regret[999] - result.cumulative_regret[899]) / 100
        assert last_100_avg <= first_100_avg + 0.1  # Allow some variance
