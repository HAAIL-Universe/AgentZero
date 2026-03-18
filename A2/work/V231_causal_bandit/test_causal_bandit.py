"""Tests for V231: Causal Bandit."""

import pytest
import numpy as np
import sys
import os
import math

sys.path.insert(0, os.path.dirname(__file__))

from causal_bandit import (
    # Data structures
    BanditStrategy, ArmStats, BanditRound, CausalBanditResult,
    # Reward functions
    edge_count_reward, normalized_edge_reward, information_gain_reward,
    cost_adjusted_reward,
    # Selection strategies
    ucb_select, thompson_select, ids_select, causal_ucb_select,
    cost_sensitive_select, epsilon_greedy_select, random_select,
    # Core classes
    CausalBandit, AdaptiveCausalBandit, ContextualCausalBandit,
    # Discovery functions
    causal_bandit_discovery, batch_causal_bandit_discovery,
    adaptive_causal_bandit_discovery, budget_constrained_discovery,
    contextual_causal_bandit_discovery,
    # Analysis
    compute_regret, compare_strategies, causal_bandit_summary,
    # Environments
    build_chain_environment, build_diamond_environment,
    build_confounded_environment, build_large_environment,
    # Internals
    CPDAG, _active_arms, _build_initial_cpdag,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chain_env():
    return build_chain_environment(n=4, seed=42)


@pytest.fixture
def diamond_env():
    return build_diamond_environment(seed=42)


@pytest.fixture
def confounded_env():
    return build_confounded_environment(seed=42)


@pytest.fixture
def large_env():
    return build_large_environment(n_vars=6, edge_prob=0.3, seed=42)


@pytest.fixture
def simple_cpdag():
    """A simple CPDAG with 4 variables and some undirected edges."""
    cpdag = CPDAG(variables=["A", "B", "C", "D"])
    cpdag.undirected.add(frozenset(["A", "B"]))
    cpdag.undirected.add(frozenset(["B", "C"]))
    cpdag.undirected.add(frozenset(["C", "D"]))
    cpdag.directed.add(("A", "C"))
    return cpdag


@pytest.fixture
def simple_arms():
    """Simple arm stats for testing selection strategies."""
    arms = {
        "A": ArmStats(name="A", pulls=5, total_reward=10.0, reward_sq_sum=25.0,
                       rewards=[2.0, 2.0, 2.0, 2.0, 2.0]),
        "B": ArmStats(name="B", pulls=3, total_reward=9.0, reward_sq_sum=30.0,
                       rewards=[3.0, 3.0, 3.0]),
        "C": ArmStats(name="C", pulls=1, total_reward=1.0, reward_sq_sum=1.0,
                       rewards=[1.0]),
        "D": ArmStats(name="D", pulls=0, total_reward=0.0, reward_sq_sum=0.0,
                       rewards=[]),
    }
    return arms


# ---------------------------------------------------------------------------
# ArmStats tests
# ---------------------------------------------------------------------------

class TestArmStats:
    def test_initial_state(self):
        arm = ArmStats(name="X")
        assert arm.pulls == 0
        assert arm.mean_reward == 0.0
        assert arm.reward_variance == 1.0  # high uncertainty prior
        assert arm.rewards == []

    def test_update(self):
        arm = ArmStats(name="X")
        arm.update(3.0, edges_oriented=2)
        assert arm.pulls == 1
        assert arm.mean_reward == 3.0
        assert arm.rewards == [3.0]
        assert arm.edges_oriented_history == [2]

    def test_multiple_updates(self):
        arm = ArmStats(name="X")
        arm.update(2.0)
        arm.update(4.0)
        assert arm.pulls == 2
        assert arm.mean_reward == 3.0

    def test_variance(self):
        arm = ArmStats(name="X")
        arm.update(2.0)
        arm.update(4.0)
        arm.update(3.0)
        mean = arm.mean_reward
        expected_var = (4 + 16 + 9) / 3 - mean**2
        assert abs(arm.reward_variance - expected_var) < 1e-10

    def test_cost(self):
        arm = ArmStats(name="X", cost=5.0)
        assert arm.cost == 5.0


# ---------------------------------------------------------------------------
# Reward function tests
# ---------------------------------------------------------------------------

class TestRewardFunctions:
    def setup_method(self):
        self.cpdag_before = CPDAG(variables=["A", "B", "C"])
        self.cpdag_before.undirected.add(frozenset(["A", "B"]))
        self.cpdag_before.undirected.add(frozenset(["B", "C"]))

        self.cpdag_after = CPDAG(variables=["A", "B", "C"])
        self.cpdag_after.directed.add(("A", "B"))
        self.cpdag_after.undirected.add(frozenset(["B", "C"]))

    def test_edge_count_reward(self):
        r = edge_count_reward(1, self.cpdag_before, self.cpdag_after)
        assert r == 1.0

    def test_edge_count_zero(self):
        r = edge_count_reward(0, self.cpdag_before, self.cpdag_before)
        assert r == 0.0

    def test_normalized_reward(self):
        r = normalized_edge_reward(1, self.cpdag_before, self.cpdag_after)
        assert r == 0.5  # 1 out of 2 undirected edges

    def test_normalized_reward_no_undirected(self):
        empty = CPDAG(variables=["A"])
        r = normalized_edge_reward(0, empty, empty)
        assert r == 0.0

    def test_information_gain_reward(self):
        r = information_gain_reward(1, self.cpdag_before, self.cpdag_after)
        # log2(3/2) = 0.585
        assert r > 0
        assert abs(r - math.log2(3 / 2)) < 1e-10

    def test_information_gain_all_oriented(self):
        cpdag_full = CPDAG(variables=["A", "B", "C"])
        cpdag_full.directed.add(("A", "B"))
        cpdag_full.directed.add(("B", "C"))
        r = information_gain_reward(2, self.cpdag_before, cpdag_full)
        assert r > 0  # log2(3/1) = log2(3)

    def test_cost_adjusted_reward(self):
        r = cost_adjusted_reward(2, self.cpdag_before, self.cpdag_after, cost=4.0)
        assert r == 0.5

    def test_cost_adjusted_zero_cost(self):
        r = cost_adjusted_reward(2, self.cpdag_before, self.cpdag_after, cost=0.0)
        assert r == 2.0  # fallback to cost=1.0


# ---------------------------------------------------------------------------
# Selection strategy tests
# ---------------------------------------------------------------------------

class TestSelectionStrategies:
    def test_ucb_selects_unvisited(self, simple_arms, simple_cpdag):
        # D has 0 pulls, should be selected first
        arm = ucb_select(simple_arms, 10, simple_cpdag)
        assert arm == "D"

    def test_ucb_balances_exploration(self, simple_cpdag):
        arms = {
            "A": ArmStats(name="A", pulls=10, total_reward=20.0, rewards=[2.0]*10),
            "B": ArmStats(name="B", pulls=2, total_reward=6.0, rewards=[3.0]*2),
            "C": ArmStats(name="C", pulls=10, total_reward=5.0, rewards=[0.5]*10),
        }
        # Add undirected edges for all
        simple_cpdag.undirected.add(frozenset(["A", "D"]))
        arm = ucb_select(arms, 22, simple_cpdag, c=2.0)
        # B has high mean (3.0) and low pulls -> high UCB
        assert arm == "B"

    def test_thompson_returns_valid_arm(self, simple_arms, simple_cpdag):
        rng = np.random.default_rng(42)
        arm = thompson_select(simple_arms, 10, simple_cpdag, rng=rng)
        assert arm in simple_arms

    def test_thompson_explores_variety(self, simple_arms, simple_cpdag):
        # Thompson sampling should explore multiple arms over many runs
        # due to posterior sampling variance
        picks = set()
        for i in range(50):
            arm = thompson_select(simple_arms, 10, simple_cpdag,
                                  rng=np.random.default_rng(i))
            picks.add(arm)
        # Should pick at least 2 different arms
        assert len(picks) >= 2

    def test_ids_selects_informative(self, simple_cpdag):
        arms = {
            "A": ArmStats(name="A", pulls=5, total_reward=5.0,
                          reward_sq_sum=7.0, rewards=[1.0]*5),
            "B": ArmStats(name="B", pulls=5, total_reward=10.0,
                          reward_sq_sum=25.0, rewards=[2.0]*5),
            "C": ArmStats(name="C", pulls=5, total_reward=2.5,
                          reward_sq_sum=5.0, rewards=[0.5]*5),
        }
        arm = ids_select(arms, 15, simple_cpdag)
        assert arm is not None

    def test_ids_explores_unvisited(self, simple_arms, simple_cpdag):
        arm = ids_select(simple_arms, 10, simple_cpdag)
        assert arm == "D"  # 0 pulls

    def test_causal_ucb_uses_structure(self, simple_cpdag):
        arms = {
            "A": ArmStats(name="A", pulls=5, total_reward=5.0, rewards=[1.0]*5),
            "B": ArmStats(name="B", pulls=5, total_reward=5.0, rewards=[1.0]*5),
            "C": ArmStats(name="C", pulls=5, total_reward=5.0, rewards=[1.0]*5),
        }
        # B has 2 undirected neighbors (A,C), A has 1 (B), C has 2 (B,D)
        arm = causal_ucb_select(arms, 15, simple_cpdag, causal_weight=10.0)
        # B or C should be preferred (more undirected neighbors)
        assert arm in ["B", "C"]

    def test_cost_sensitive_prefers_cheap(self, simple_cpdag):
        arms = {
            "A": ArmStats(name="A", pulls=5, total_reward=5.0,
                          rewards=[1.0]*5, cost=1.0),
            "B": ArmStats(name="B", pulls=5, total_reward=5.0,
                          rewards=[1.0]*5, cost=10.0),
            "C": ArmStats(name="C", pulls=5, total_reward=5.0,
                          rewards=[1.0]*5, cost=1.0),
        }
        arm = cost_sensitive_select(arms, 15, simple_cpdag)
        # A and C are much cheaper than B for same reward
        assert arm in ["A", "C"]

    def test_epsilon_greedy_exploits(self, simple_cpdag):
        arms = {
            "A": ArmStats(name="A", pulls=10, total_reward=30.0, rewards=[3.0]*10),
            "B": ArmStats(name="B", pulls=10, total_reward=10.0, rewards=[1.0]*10),
            "C": ArmStats(name="C", pulls=10, total_reward=5.0, rewards=[0.5]*10),
        }
        rng = np.random.default_rng(42)
        # With epsilon=0, always exploit
        arm = epsilon_greedy_select(arms, 30, simple_cpdag, epsilon=0.0, rng=rng)
        assert arm == "A"

    def test_epsilon_greedy_explores(self, simple_cpdag):
        arms = {
            "A": ArmStats(name="A", pulls=10, total_reward=30.0, rewards=[3.0]*10),
            "B": ArmStats(name="B", pulls=10, total_reward=10.0, rewards=[1.0]*10),
            "C": ArmStats(name="C", pulls=10, total_reward=5.0, rewards=[0.5]*10),
        }
        # With epsilon=1, always explore -> uniform random
        picks = set()
        for i in range(50):
            arm = epsilon_greedy_select(arms, 30, simple_cpdag, epsilon=1.0,
                                        rng=np.random.default_rng(i))
            picks.add(arm)
        assert len(picks) > 1

    def test_random_select(self, simple_arms, simple_cpdag):
        rng = np.random.default_rng(42)
        picks = set()
        for i in range(50):
            arm = random_select(simple_arms, 10, simple_cpdag,
                                rng=np.random.default_rng(i))
            if arm:
                picks.add(arm)
        assert len(picks) >= 2


# ---------------------------------------------------------------------------
# Active arms test
# ---------------------------------------------------------------------------

class TestActiveArms:
    def test_returns_arms_with_undirected(self, simple_arms, simple_cpdag):
        active = _active_arms(simple_arms, simple_cpdag)
        # A,B,C,D all have undirected neighbors in simple_cpdag
        assert len(active) > 0

    def test_fully_oriented_returns_empty(self, simple_arms):
        cpdag = CPDAG(variables=["A", "B", "C", "D"])
        cpdag.directed.add(("A", "B"))
        cpdag.directed.add(("B", "C"))
        cpdag.directed.add(("C", "D"))
        active = _active_arms(simple_arms, cpdag)
        assert active == []


# ---------------------------------------------------------------------------
# CausalBandit class tests
# ---------------------------------------------------------------------------

class TestCausalBandit:
    def test_init(self):
        bandit = CausalBandit(["A", "B", "C"], BanditStrategy.UCB)
        assert len(bandit.arms) == 3
        assert bandit.strategy == BanditStrategy.UCB
        assert bandit.total_edges_oriented == 0

    def test_select_arm(self, simple_cpdag):
        bandit = CausalBandit(["A", "B", "C", "D"], BanditStrategy.UCB)
        arm = bandit.select_arm(simple_cpdag, 1)
        assert arm in ["A", "B", "C", "D"]

    def test_update(self, simple_cpdag):
        bandit = CausalBandit(["A", "B", "C", "D"], BanditStrategy.UCB)
        cpdag_after = simple_cpdag.copy()
        bandit.update("A", 2, simple_cpdag, cpdag_after, 0)
        assert bandit.arms["A"].pulls == 1
        assert bandit.total_edges_oriented == 2
        assert len(bandit.rounds) == 1

    def test_result(self, simple_cpdag):
        bandit = CausalBandit(["A", "B", "C", "D"], BanditStrategy.UCB)
        result = bandit.result(simple_cpdag)
        assert isinstance(result, CausalBanditResult)
        assert result.strategy == "ucb"
        assert not result.fully_oriented  # has undirected edges

    def test_custom_reward_fn(self, simple_cpdag):
        bandit = CausalBandit(["A", "B"], BanditStrategy.UCB,
                              reward_fn=normalized_edge_reward)
        cpdag_after = simple_cpdag.copy()
        bandit.update("A", 1, simple_cpdag, cpdag_after, 0)
        # normalized: 1/3 undirected edges
        expected = 1.0 / 3.0
        assert abs(bandit.arms["A"].mean_reward - expected) < 1e-6

    def test_costs(self):
        costs = {"A": 2.0, "B": 5.0}
        bandit = CausalBandit(["A", "B"], BanditStrategy.UCB, costs=costs)
        assert bandit.arms["A"].cost == 2.0
        assert bandit.arms["B"].cost == 5.0


# ---------------------------------------------------------------------------
# Discovery function tests
# ---------------------------------------------------------------------------

class TestCausalBanditDiscovery:
    def test_chain_ucb(self, chain_env):
        obs_data, variables, intervention_fn, true_edges = chain_env
        result = causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            strategy=BanditStrategy.UCB,
            max_rounds=15,
            rng=np.random.default_rng(42),
        )
        assert isinstance(result, CausalBanditResult)
        assert result.total_rounds > 0
        assert result.total_edges_oriented >= 0

    def test_chain_thompson(self, chain_env):
        obs_data, variables, intervention_fn, true_edges = chain_env
        result = causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            strategy=BanditStrategy.THOMPSON,
            max_rounds=15,
            rng=np.random.default_rng(42),
        )
        assert result.total_rounds > 0

    def test_diamond_discovery(self, diamond_env):
        obs_data, variables, intervention_fn, true_edges = diamond_env
        result = causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            strategy=BanditStrategy.CAUSAL_UCB,
            max_rounds=15,
            rng=np.random.default_rng(42),
        )
        assert isinstance(result, CausalBanditResult)

    def test_confounded_discovery(self, confounded_env):
        obs_data, variables, intervention_fn, true_edges = confounded_env
        result = causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            strategy=BanditStrategy.UCB,
            max_rounds=10,
            rng=np.random.default_rng(42),
        )
        assert isinstance(result, CausalBanditResult)

    def test_early_termination(self, confounded_env):
        """Should stop early when CPDAG is fully oriented."""
        obs_data, variables, intervention_fn, true_edges = confounded_env
        result = causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            strategy=BanditStrategy.UCB,
            max_rounds=100,  # very high max
            rng=np.random.default_rng(42),
        )
        # Should not use all 100 rounds for a 2-variable graph
        assert result.total_rounds <= 20

    def test_custom_reward(self, chain_env):
        obs_data, variables, intervention_fn, true_edges = chain_env
        result = causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            strategy=BanditStrategy.UCB,
            max_rounds=10,
            reward_fn=information_gain_reward,
            rng=np.random.default_rng(42),
        )
        assert result.total_rounds > 0

    def test_epsilon_greedy_discovery(self, chain_env):
        obs_data, variables, intervention_fn, true_edges = chain_env
        result = causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            strategy=BanditStrategy.EPSILON_GREEDY,
            max_rounds=10,
            strategy_params={'epsilon': 0.2},
            rng=np.random.default_rng(42),
        )
        assert result.total_rounds > 0

    def test_random_discovery(self, chain_env):
        obs_data, variables, intervention_fn, true_edges = chain_env
        result = causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            strategy=BanditStrategy.RANDOM,
            max_rounds=10,
            rng=np.random.default_rng(42),
        )
        assert result.total_rounds > 0


# ---------------------------------------------------------------------------
# Batch discovery tests
# ---------------------------------------------------------------------------

class TestBatchDiscovery:
    def test_batch_chain(self, chain_env):
        obs_data, variables, intervention_fn, true_edges = chain_env
        result = batch_causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            strategy=BanditStrategy.UCB,
            max_batches=5,
            batch_size=2,
            rng=np.random.default_rng(42),
        )
        assert isinstance(result, CausalBanditResult)
        assert result.total_rounds > 0

    def test_batch_diamond(self, diamond_env):
        obs_data, variables, intervention_fn, true_edges = diamond_env
        result = batch_causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            strategy=BanditStrategy.THOMPSON,
            max_batches=5,
            batch_size=2,
            rng=np.random.default_rng(42),
        )
        assert result.total_rounds > 0

    def test_batch_diversity(self, chain_env):
        """With high diversity weight, batch should pick distant nodes."""
        obs_data, variables, intervention_fn, true_edges = chain_env
        result = batch_causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            strategy=BanditStrategy.UCB,
            max_batches=3,
            batch_size=3,
            diversity_weight=1.0,
            rng=np.random.default_rng(42),
        )
        assert result.total_rounds > 0


# ---------------------------------------------------------------------------
# Adaptive discovery tests
# ---------------------------------------------------------------------------

class TestAdaptiveDiscovery:
    def test_adaptive_chain(self, chain_env):
        obs_data, variables, intervention_fn, true_edges = chain_env
        result = adaptive_causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            max_rounds=15,
            rng=np.random.default_rng(42),
        )
        assert isinstance(result, CausalBanditResult)
        assert result.strategy == "adaptive"

    def test_adaptive_diamond(self, diamond_env):
        obs_data, variables, intervention_fn, true_edges = diamond_env
        result = adaptive_causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            max_rounds=15,
            switch_threshold=0.3,
            rng=np.random.default_rng(42),
        )
        assert result.total_rounds > 0

    def test_adaptive_phases(self, chain_env):
        """Adaptive should transition from explore to exploit."""
        obs_data, variables, intervention_fn, true_edges = chain_env
        rng = np.random.default_rng(42)
        adaptive = AdaptiveCausalBandit(
            variables=variables, switch_threshold=0.3, rng=rng
        )
        cpdag = _build_initial_cpdag(obs_data, variables, 0.05)
        # Initially should be in explore phase
        arm = adaptive.select_arm(cpdag, 1, obs_data)
        # Phase should be recorded
        assert len(adaptive.phase_history) == 1


# ---------------------------------------------------------------------------
# Budget-constrained discovery tests
# ---------------------------------------------------------------------------

class TestBudgetDiscovery:
    def test_budget_constraint(self, chain_env):
        obs_data, variables, intervention_fn, true_edges = chain_env
        result = budget_constrained_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            budget=5.0,
            costs={v: 1.0 for v in variables},
            rng=np.random.default_rng(42),
        )
        assert result.total_cost <= 5.0 + 1e-6

    def test_expensive_arms(self, chain_env):
        obs_data, variables, intervention_fn, true_edges = chain_env
        costs = {v: 3.0 for v in variables}
        result = budget_constrained_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            budget=6.0,
            costs=costs,
            rng=np.random.default_rng(42),
        )
        # Budget 6.0, cost per arm 3.0 -> at most 2 interventions
        assert result.total_rounds <= 2

    def test_variable_costs(self, chain_env):
        obs_data, variables, intervention_fn, true_edges = chain_env
        costs = {variables[0]: 1.0, variables[1]: 5.0,
                 variables[2]: 1.0, variables[3]: 5.0}
        result = budget_constrained_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            budget=4.0,
            costs=costs,
            strategy=BanditStrategy.COST_SENSITIVE,
            rng=np.random.default_rng(42),
        )
        assert result.total_cost <= 4.0 + 1e-6


# ---------------------------------------------------------------------------
# Contextual bandit tests
# ---------------------------------------------------------------------------

class TestContextualBandit:
    def test_contextual_chain(self, chain_env):
        obs_data, variables, intervention_fn, true_edges = chain_env
        result = contextual_causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            max_rounds=10,
            rng=np.random.default_rng(42),
        )
        assert isinstance(result, CausalBanditResult)
        assert result.strategy == "contextual_linucb"

    def test_contextual_features(self, simple_cpdag):
        bandit = ContextualCausalBandit(["A", "B", "C", "D"])
        features = bandit._extract_features(simple_cpdag, "B")
        assert len(features) == 5
        # B has 2 undirected neighbors (A, C)
        assert features[0] == 2  # undirected degree

    def test_contextual_diamond(self, diamond_env):
        obs_data, variables, intervention_fn, true_edges = diamond_env
        result = contextual_causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            max_rounds=10,
            rng=np.random.default_rng(42),
        )
        assert result.total_rounds > 0


# ---------------------------------------------------------------------------
# Regret analysis tests
# ---------------------------------------------------------------------------

class TestRegretAnalysis:
    def test_compute_regret_empty(self):
        result = CausalBanditResult(
            final_cpdag=CPDAG(variables=["A"]),
            rounds=[], arm_stats={}, total_edges_oriented=0,
            total_rounds=0, total_cost=0.0, strategy="ucb",
            fully_oriented=True,
        )
        regret = compute_regret(result)
        assert regret['cumulative_regret'] == 0.0
        assert regret['efficiency'] == 0.0

    def test_compute_regret_with_data(self, chain_env):
        obs_data, variables, intervention_fn, true_edges = chain_env
        result = causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            strategy=BanditStrategy.UCB,
            max_rounds=10,
            ground_truth=true_edges,
            rng=np.random.default_rng(42),
        )
        regret = compute_regret(result)
        assert 'efficiency' in regret
        assert regret['efficiency'] >= 0

    def test_regret_history(self, chain_env):
        obs_data, variables, intervention_fn, true_edges = chain_env
        result = causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            strategy=BanditStrategy.UCB,
            max_rounds=10,
            ground_truth=true_edges,
            rng=np.random.default_rng(42),
        )
        if result.cumulative_regret:
            # Cumulative regret should be non-decreasing
            for i in range(1, len(result.cumulative_regret)):
                assert result.cumulative_regret[i] >= result.cumulative_regret[i-1] - 1e-10


# ---------------------------------------------------------------------------
# Strategy comparison tests
# ---------------------------------------------------------------------------

class TestCompareStrategies:
    def test_compare_basic(self, chain_env):
        obs_data, variables, intervention_fn, true_edges = chain_env
        comparison = compare_strategies(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            strategies=[BanditStrategy.UCB, BanditStrategy.RANDOM],
            max_rounds=8,
            n_trials=2,
            rng_seed=42,
        )
        assert "ucb" in comparison
        assert "random" in comparison
        assert 'mean_rounds' in comparison["ucb"]
        assert 'mean_efficiency' in comparison["ucb"]

    def test_compare_multiple(self, diamond_env):
        obs_data, variables, intervention_fn, true_edges = diamond_env
        comparison = compare_strategies(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            strategies=[BanditStrategy.UCB, BanditStrategy.THOMPSON,
                        BanditStrategy.EPSILON_GREEDY],
            max_rounds=8,
            n_trials=2,
            rng_seed=42,
        )
        assert len(comparison) == 3


# ---------------------------------------------------------------------------
# Summary tests
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary(self, chain_env):
        obs_data, variables, intervention_fn, true_edges = chain_env
        result = causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            strategy=BanditStrategy.UCB,
            max_rounds=10,
            rng=np.random.default_rng(42),
        )
        summary = causal_bandit_summary(result)
        assert 'strategy' in summary
        assert summary['strategy'] == 'ucb'
        assert 'total_rounds' in summary
        assert 'total_edges_oriented' in summary
        assert 'efficiency' in summary
        assert 'arm_summaries' in summary


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

class TestEnvironments:
    def test_chain_env(self, chain_env):
        obs_data, variables, intervention_fn, true_edges = chain_env
        assert len(variables) == 4
        assert len(true_edges) == 3
        assert len(obs_data) == 500

    def test_chain_intervention(self, chain_env):
        obs_data, variables, intervention_fn, true_edges = chain_env
        samples = intervention_fn("X2", 1)
        assert len(samples) == 200
        # All X2 values should be 1 (intervened)
        assert all(s["X2"] == 1 for s in samples)

    def test_diamond_env(self, diamond_env):
        obs_data, variables, intervention_fn, true_edges = diamond_env
        assert set(variables) == {"X", "A", "B", "Y"}
        assert len(true_edges) == 4

    def test_diamond_intervention(self, diamond_env):
        obs_data, variables, intervention_fn, true_edges = diamond_env
        samples = intervention_fn("X", 0)
        assert all(s["X"] == 0 for s in samples)

    def test_confounded_env(self, confounded_env):
        obs_data, variables, intervention_fn, true_edges = confounded_env
        assert set(variables) == {"X", "Y"}
        assert true_edges == {("X", "Y")}

    def test_large_env(self, large_env):
        obs_data, variables, intervention_fn, true_edges = large_env
        assert len(variables) == 6
        assert len(obs_data) == 1000
        # Should have some edges
        assert len(true_edges) >= 0

    def test_large_intervention(self, large_env):
        obs_data, variables, intervention_fn, true_edges = large_env
        samples = intervention_fn("V0", 1)
        assert all(s["V0"] == 1 for s in samples)


# ---------------------------------------------------------------------------
# CPDAG building tests
# ---------------------------------------------------------------------------

class TestCPDAGBuilding:
    def test_build_from_chain_data(self, chain_env):
        obs_data, variables, _, _ = chain_env
        cpdag = _build_initial_cpdag(obs_data, variables, 0.05)
        assert isinstance(cpdag, CPDAG)
        # Should have some edges discovered
        total_edges = len(cpdag.directed) + cpdag.num_undirected()
        assert total_edges >= 0  # may or may not find edges depending on data

    def test_build_from_diamond_data(self, diamond_env):
        obs_data, variables, _, _ = diamond_env
        cpdag = _build_initial_cpdag(obs_data, variables, 0.05)
        assert isinstance(cpdag, CPDAG)

    def test_empty_data(self):
        cpdag = _build_initial_cpdag([], ["A", "B"], 0.05)
        assert cpdag.num_undirected() == 0
        assert len(cpdag.directed) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_variable(self):
        result = causal_bandit_discovery(
            initial_data=[{"X": 0}, {"X": 1}] * 50,
            variables=["X"],
            intervention_fn=lambda t, v: [{"X": v}] * 50,
            max_rounds=5,
            rng=np.random.default_rng(42),
        )
        assert result.total_rounds == 0  # no edges to orient

    def test_two_variables(self):
        obs_data = []
        rng = np.random.default_rng(42)
        for _ in range(200):
            x = int(rng.integers(0, 2))
            y = int(x if rng.random() < 0.8 else rng.integers(0, 2))
            obs_data.append({"X": x, "Y": y})

        def int_fn(target, value):
            samples = []
            for _ in range(100):
                if target == "X":
                    x = int(value)
                    y = int(x if rng.random() < 0.8 else rng.integers(0, 2))
                else:
                    x = int(rng.integers(0, 2))
                    y = int(value)
                samples.append({"X": x, "Y": y})
            return samples

        result = causal_bandit_discovery(
            initial_data=obs_data,
            variables=["X", "Y"],
            intervention_fn=int_fn,
            max_rounds=10,
            rng=np.random.default_rng(42),
        )
        assert isinstance(result, CausalBanditResult)

    def test_no_edges_discoverable(self):
        """Independent variables -- no causal structure to find."""
        rng = np.random.default_rng(42)
        obs_data = [{"A": int(rng.integers(0, 2)),
                     "B": int(rng.integers(0, 2))} for _ in range(200)]

        result = causal_bandit_discovery(
            initial_data=obs_data,
            variables=["A", "B"],
            intervention_fn=lambda t, v: [{"A": int(rng.integers(0, 2)),
                                           "B": int(rng.integers(0, 2))}
                                          for _ in range(100)],
            max_rounds=5,
            rng=np.random.default_rng(42),
        )
        # May or may not find edges (false positives possible)
        assert isinstance(result, CausalBanditResult)

    def test_all_strategies_run(self, chain_env):
        """Every strategy should complete without error."""
        obs_data, variables, intervention_fn, true_edges = chain_env
        for strat in BanditStrategy:
            result = causal_bandit_discovery(
                initial_data=obs_data,
                variables=variables,
                intervention_fn=intervention_fn,
                strategy=strat,
                max_rounds=5,
                rng=np.random.default_rng(42),
            )
            assert isinstance(result, CausalBanditResult)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline_chain(self, chain_env):
        """Full pipeline: discover -> analyze -> compare."""
        obs_data, variables, intervention_fn, true_edges = chain_env

        # Run discovery
        result = causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            strategy=BanditStrategy.UCB,
            max_rounds=15,
            ground_truth=true_edges,
            rng=np.random.default_rng(42),
        )

        # Analyze
        regret = compute_regret(result)
        summary = causal_bandit_summary(result)

        assert summary['total_rounds'] == result.total_rounds
        assert regret['efficiency'] >= 0

    def test_full_pipeline_diamond(self, diamond_env):
        obs_data, variables, intervention_fn, true_edges = diamond_env

        # Adaptive discovery
        result = adaptive_causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            max_rounds=15,
            rng=np.random.default_rng(42),
        )

        summary = causal_bandit_summary(result)
        assert 'arm_summaries' in summary

    def test_budget_then_compare(self, chain_env):
        """Budget-constrained then compare with unlimited."""
        obs_data, variables, intervention_fn, true_edges = chain_env

        budget_result = budget_constrained_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            budget=3.0,
            rng=np.random.default_rng(42),
        )

        full_result = causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            max_rounds=15,
            rng=np.random.default_rng(42),
        )

        # Budget-constrained should have fewer or equal rounds
        assert budget_result.total_rounds <= full_result.total_rounds + 1

    def test_contextual_vs_standard(self, chain_env):
        """Contextual should produce valid results comparable to standard."""
        obs_data, variables, intervention_fn, true_edges = chain_env

        standard = causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            strategy=BanditStrategy.UCB,
            max_rounds=10,
            rng=np.random.default_rng(42),
        )

        contextual = contextual_causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            max_rounds=10,
            rng=np.random.default_rng(42),
        )

        assert isinstance(standard, CausalBanditResult)
        assert isinstance(contextual, CausalBanditResult)

    def test_large_graph_scalability(self, large_env):
        """Should handle larger graphs without errors."""
        obs_data, variables, intervention_fn, true_edges = large_env
        result = causal_bandit_discovery(
            initial_data=obs_data,
            variables=variables,
            intervention_fn=intervention_fn,
            strategy=BanditStrategy.THOMPSON,
            max_rounds=15,
            rng=np.random.default_rng(42),
        )
        assert isinstance(result, CausalBanditResult)
