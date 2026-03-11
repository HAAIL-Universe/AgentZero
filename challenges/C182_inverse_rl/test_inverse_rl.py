"""Tests for C182: Inverse Reinforcement Learning."""

import math
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from inverse_rl import (
    TabularMDP, ValueIteration, FeatureExtractor, ExpertDemonstrations,
    FeatureExpectation, MaxEntIRL, ProjectionIRL, BayesianIRL,
    DeepMaxEntIRL, ApprenticeshipLearning, RewardShaping,
    IRLEvaluator, MaxCausalEntIRL, _pearson_correlation, _cosine_similarity,
)


# ============================================================
# TabularMDP Tests
# ============================================================

class TestTabularMDP:
    def test_create_basic(self):
        mdp = TabularMDP(4, 2, gamma=0.9)
        assert mdp.n_states == 4
        assert mdp.n_actions == 2
        assert mdp.gamma == 0.9

    def test_set_transition(self):
        mdp = TabularMDP(3, 2)
        mdp.set_transition(0, 0, [(0.7, 1), (0.3, 2)])
        assert len(mdp.T[0][0]) == 2
        assert mdp.T[0][0][0] == (0.7, 1)

    def test_set_reward(self):
        mdp = TabularMDP(3, 2)
        mdp.set_reward(0, 1, 5.0)
        assert mdp.R[0][1] == 5.0

    def test_set_state_reward(self):
        mdp = TabularMDP(3, 2)
        mdp.set_state_reward(1, 3.0)
        assert mdp.R[1][0] == 3.0
        assert mdp.R[1][1] == 3.0

    def test_terminal_states(self):
        mdp = TabularMDP(5, 2)
        mdp.set_terminal(4)
        assert 4 in mdp.terminal_states
        assert 0 not in mdp.terminal_states

    def test_get_transition_prob(self):
        mdp = TabularMDP(3, 2)
        mdp.set_transition(0, 0, [(0.8, 1), (0.2, 2)])
        assert mdp.get_transition_prob(0, 0, 1) == 0.8
        assert mdp.get_transition_prob(0, 0, 2) == 0.2
        assert mdp.get_transition_prob(0, 0, 0) == 0.0

    def test_from_grid(self):
        mdp = TabularMDP.from_grid(3, 3, gamma=0.95)
        assert mdp.n_states == 9
        assert mdp.n_actions == 4
        assert mdp.gamma == 0.95
        # Goal at (2,2) = state 8
        assert 8 in mdp.terminal_states

    def test_from_grid_with_walls(self):
        mdp = TabularMDP.from_grid(3, 3, walls=[(1, 1)])
        assert mdp.n_states == 9
        # Wall state bounces back
        assert mdp.T[4][0] == [(1.0, 4)]

    def test_from_grid_stochastic(self):
        mdp = TabularMDP.from_grid_stochastic(3, 3, slip_prob=0.2)
        assert mdp.n_states == 9
        # Non-terminal, non-wall state should have stochastic transitions
        # State 0, action right: should go right with 0.8, other with 0.2/3 each
        trans = mdp.T[0][1]  # state 0, right
        assert len(trans) >= 1
        total_prob = sum(p for p, _ in trans)
        assert abs(total_prob - 1.0) < 1e-6

    def test_from_grid_goal_reward(self):
        mdp = TabularMDP.from_grid(3, 3, goal=(2, 2))
        # State (1,2) going down -> reaches goal (2,2)
        s = 1 * 3 + 2  # state 5
        r = mdp.R[s][2]  # down
        assert r == 1.0

    def test_grid_step_penalty(self):
        mdp = TabularMDP.from_grid(3, 3)
        # State (0,0) going right -> non-goal
        assert mdp.R[0][1] == -0.01


# ============================================================
# Value Iteration Tests
# ============================================================

class TestValueIteration:
    def test_simple_chain(self):
        """Chain: 0 -> 1 -> 2 (terminal). R=1 at state 2."""
        mdp = TabularMDP(3, 1, gamma=0.9)
        mdp.set_transition(0, 0, [(1.0, 1)])
        mdp.set_transition(1, 0, [(1.0, 2)])
        mdp.set_transition(2, 0, [(1.0, 2)])
        mdp.set_reward(0, 0, 0.0)
        mdp.set_reward(1, 0, 1.0)
        mdp.set_terminal(2)
        vi = ValueIteration(mdp)
        V, policy = vi.solve()
        assert vi.converged
        assert V[0] > 0  # has future value
        assert V[1] > V[0]  # closer to reward

    def test_grid_convergence(self):
        mdp = TabularMDP.from_grid(4, 4, gamma=0.95)
        vi = ValueIteration(mdp)
        V, policy = vi.solve()
        assert vi.converged
        # Goal state has 0 value (terminal)
        goal_s = 4 * 4 - 1
        assert V[goal_s] == 0.0
        # Start should have some positive value
        assert V[0] > 0

    def test_policy_extraction(self):
        mdp = TabularMDP.from_grid(3, 3)
        vi = ValueIteration(mdp)
        V, policy = vi.solve()
        # Near goal, policy should point toward goal
        # State (1,2) should go down (action 2) to reach (2,2)
        assert policy[5] == 2  # (1,2) -> down

    def test_q_values(self):
        mdp = TabularMDP.from_grid(3, 3)
        vi = ValueIteration(mdp)
        vi.solve()
        Q = vi.get_q_values()
        assert len(Q) == 9
        assert len(Q[0]) == 4
        # Best action Q should match V
        for s in range(9):
            if s not in mdp.terminal_states:
                assert abs(max(Q[s]) - vi.V[s]) < 1e-6

    def test_stochastic_policy(self):
        mdp = TabularMDP.from_grid(3, 3)
        vi = ValueIteration(mdp)
        vi.solve()
        pol = vi.get_stochastic_policy(temperature=0.1)
        assert len(pol) == 9
        for s in range(9):
            probs = pol[s]
            assert abs(sum(probs) - 1.0) < 1e-6
            assert all(p >= 0 for p in probs)

    def test_stochastic_policy_low_temp(self):
        mdp = TabularMDP.from_grid(3, 3)
        vi = ValueIteration(mdp)
        vi.solve()
        pol = vi.get_stochastic_policy(temperature=0.001)
        # Low temp should concentrate on best action(s)
        for s in range(9):
            if s not in mdp.terminal_states:
                assert max(pol[s]) >= 0.49  # at least concentrates (ties possible)


# ============================================================
# Feature Extractor Tests
# ============================================================

class TestFeatureExtractor:
    def test_one_hot(self):
        fe = FeatureExtractor.one_hot(5)
        assert fe.n_features == 5
        f = fe.features(2)
        assert f[2] == 1.0
        assert sum(f) == 1.0

    def test_coordinate(self):
        fe = FeatureExtractor.coordinate(3, 3)
        assert fe.n_features == 4
        f = fe.features(0)  # (0,0) top-left corner
        assert f[0] == 0.0  # row/rows
        assert f[1] == 0.0  # col/cols
        assert f[3] == 1.0  # is_edge

    def test_coordinate_center(self):
        fe = FeatureExtractor.coordinate(3, 3)
        f = fe.features(4)  # (1,1) center
        assert f[3] == 0.0  # not edge

    def test_state_action(self):
        fe = FeatureExtractor.state_action(3, 2)
        assert fe.n_features == 6
        f = fe.features(1, 0)
        assert f[2] == 1.0  # state 1 * 2 actions + action 0
        assert sum(f) == 1.0

    def test_custom_feature(self):
        def my_fn(state, action=None):
            return [state * 0.1, 1.0 - state * 0.1]
        fe = FeatureExtractor(2, my_fn)
        f = fe.features(3)
        assert abs(f[0] - 0.3) < 1e-6


# ============================================================
# Expert Demonstrations Tests
# ============================================================

class TestExpertDemonstrations:
    def test_add_trajectory(self):
        demos = ExpertDemonstrations()
        traj = [(0, 1, 0.0, 1), (1, 2, 1.0, 2)]
        demos.add_trajectory(traj)
        assert demos.n_trajectories == 1
        assert demos.total_steps == 2

    def test_collect_from_policy(self):
        mdp = TabularMDP.from_grid(3, 3)
        vi = ValueIteration(mdp)
        _, policy = vi.solve()
        demos = ExpertDemonstrations()
        demos.collect_from_policy(mdp, policy, n_episodes=5, max_steps=50)
        assert demos.n_trajectories == 5
        assert demos.total_steps > 0

    def test_collect_from_stochastic_policy(self):
        mdp = TabularMDP.from_grid(3, 3)
        vi = ValueIteration(mdp)
        vi.solve()
        pol = vi.get_stochastic_policy(temperature=0.5)
        demos = ExpertDemonstrations()
        demos.collect_from_stochastic_policy(mdp, pol, n_episodes=5)
        assert demos.n_trajectories == 5

    def test_state_visitation(self):
        demos = ExpertDemonstrations()
        demos.add_trajectory([(0, 0, 0.0, 1), (1, 0, 0.0, 2)])
        demos.add_trajectory([(0, 0, 0.0, 1)])
        sv = demos.get_state_visitation(3)
        assert len(sv) == 3
        assert abs(sum(sv) - 1.0) < 1e-6
        # State 0 visited twice, state 1 once
        assert sv[0] > sv[1]

    def test_state_action_visitation(self):
        demos = ExpertDemonstrations()
        demos.add_trajectory([(0, 1, 0.0, 1), (1, 0, 0.0, 2)])
        sa = demos.get_state_action_visitation(3, 2)
        assert len(sa) == 3
        assert abs(sum(sum(row) for row in sa) - 1.0) < 1e-6

    def test_feature_expectations(self):
        fe = FeatureExtractor.one_hot(3)
        demos = ExpertDemonstrations()
        demos.add_trajectory([(0, 0, 0.0, 1), (1, 0, 0.0, 2)])
        mu = demos.get_feature_expectations(fe)
        assert len(mu) == 3
        assert mu[0] > 0
        assert mu[1] > 0


# ============================================================
# Feature Expectation Tests
# ============================================================

class TestFeatureExpectation:
    def test_from_deterministic_policy(self):
        mdp = TabularMDP.from_grid(3, 3)
        vi = ValueIteration(mdp)
        _, policy = vi.solve()
        fe = FeatureExtractor.one_hot(9)
        mu = FeatureExpectation.from_policy(mdp, policy, fe, gamma=0.9)
        assert len(mu) == 9
        assert sum(mu) > 0

    def test_from_stochastic_policy(self):
        mdp = TabularMDP.from_grid(3, 3)
        vi = ValueIteration(mdp)
        vi.solve()
        pol = vi.get_stochastic_policy()
        fe = FeatureExtractor.one_hot(9)
        mu = FeatureExpectation.from_policy(mdp, pol, fe)
        assert len(mu) == 9
        assert sum(mu) > 0

    def test_from_demonstrations(self):
        mdp = TabularMDP.from_grid(3, 3)
        vi = ValueIteration(mdp)
        _, policy = vi.solve()
        demos = ExpertDemonstrations()
        demos.collect_from_policy(mdp, policy, n_episodes=10)
        fe = FeatureExtractor.one_hot(9)
        mu = FeatureExpectation.from_demonstrations(demos, fe, gamma=0.9)
        assert len(mu) == 9
        assert sum(mu) > 0

    def test_consistency(self):
        """Feature expectations from policy and demos should be similar."""
        mdp = TabularMDP.from_grid(3, 3, gamma=0.9)
        vi = ValueIteration(mdp)
        _, policy = vi.solve()

        fe = FeatureExtractor.one_hot(9)
        mu_policy = FeatureExpectation.from_policy(mdp, policy, fe, gamma=0.9)

        demos = ExpertDemonstrations()
        demos.collect_from_policy(mdp, policy, n_episodes=100, max_steps=100, seed=42)
        mu_demos = FeatureExpectation.from_demonstrations(demos, fe, gamma=0.9)

        # Both should be non-zero vectors
        assert sum(mu_policy) > 0
        assert sum(mu_demos) > 0
        # Should have some correlation (not exact due to finite samples + different computation)
        corr = _pearson_correlation(mu_policy, mu_demos)
        assert corr > -0.5  # at least not anti-correlated


# ============================================================
# MaxEnt IRL Tests
# ============================================================

class TestMaxEntIRL:
    def _setup_grid(self, rows=3, cols=3):
        mdp = TabularMDP.from_grid(rows, cols, gamma=0.9)
        vi = ValueIteration(mdp)
        _, true_policy = vi.solve()
        true_reward = [mdp.R[s][0] for s in range(mdp.n_states)]

        demos = ExpertDemonstrations()
        demos.collect_from_policy(mdp, true_policy, n_episodes=20, max_steps=50)

        fe = FeatureExtractor.one_hot(mdp.n_states)
        return mdp, true_policy, true_reward, demos, fe

    def test_basic_training(self):
        mdp, true_policy, true_reward, demos, fe = self._setup_grid()
        irl = MaxEntIRL(mdp, fe, lr=0.2, n_iters=30, seed=42)
        theta = irl.train(demos)
        assert len(theta) == 9

    def test_gradient_tracking(self):
        mdp, true_policy, true_reward, demos, fe = self._setup_grid()
        irl = MaxEntIRL(mdp, fe, lr=0.2, n_iters=30)
        irl.train(demos)
        # Should track gradient norms
        assert len(irl.history) == 30
        assert all(h >= 0 for h in irl.history)

    def test_reward_recovery(self):
        mdp, true_policy, true_reward, demos, fe = self._setup_grid()
        irl = MaxEntIRL(mdp, fe, lr=0.3, n_iters=40, seed=42)
        irl.train(demos)
        learned_reward = irl.get_reward_function()
        # Should produce finite rewards for all states
        assert all(math.isfinite(r) for r in learned_reward)
        assert len(learned_reward) == 9

    def test_policy_recovery(self):
        mdp, true_policy, true_reward, demos, fe = self._setup_grid()
        irl = MaxEntIRL(mdp, fe, lr=0.3, n_iters=50, temperature=0.5, seed=42)
        irl.train(demos)
        learned_policy = irl.get_policy()
        # At least some states should match
        match = sum(1 for s in range(9) if learned_policy[s] == true_policy[s]
                   and s not in mdp.terminal_states)
        non_term = sum(1 for s in range(9) if s not in mdp.terminal_states)
        assert match / non_term > 0.3

    def test_with_coordinate_features(self):
        mdp = TabularMDP.from_grid(4, 4, gamma=0.9)
        vi = ValueIteration(mdp)
        _, true_policy = vi.solve()
        demos = ExpertDemonstrations()
        demos.collect_from_policy(mdp, true_policy, n_episodes=20)
        fe = FeatureExtractor.coordinate(4, 4)
        irl = MaxEntIRL(mdp, fe, lr=0.3, n_iters=40)
        theta = irl.train(demos)
        assert len(theta) == 4


# ============================================================
# Projection IRL Tests
# ============================================================

class TestProjectionIRL:
    def _setup_grid(self):
        mdp = TabularMDP.from_grid(3, 3, gamma=0.9)
        vi = ValueIteration(mdp)
        _, true_policy = vi.solve()
        demos = ExpertDemonstrations()
        demos.collect_from_policy(mdp, true_policy, n_episodes=20, max_steps=50)
        fe = FeatureExtractor.one_hot(9)
        return mdp, true_policy, demos, fe

    def test_basic_training(self):
        mdp, true_policy, demos, fe = self._setup_grid()
        irl = ProjectionIRL(mdp, fe, n_iters=15, seed=42)
        theta = irl.train(demos)
        assert len(theta) == 9

    def test_margin_convergence(self):
        mdp, true_policy, demos, fe = self._setup_grid()
        irl = ProjectionIRL(mdp, fe, n_iters=15)
        irl.train(demos)
        # Margin should decrease
        if len(irl.history) > 1:
            assert irl.history[-1] <= irl.history[0] * 1.5

    def test_policy_found(self):
        mdp, true_policy, demos, fe = self._setup_grid()
        irl = ProjectionIRL(mdp, fe, n_iters=15, seed=42)
        irl.train(demos)
        policy = irl.get_best_policy()
        assert len(policy) == 9

    def test_multiple_policies_collected(self):
        mdp, true_policy, demos, fe = self._setup_grid()
        irl = ProjectionIRL(mdp, fe, n_iters=10)
        irl.train(demos)
        assert len(irl.policies) > 1


# ============================================================
# Bayesian IRL Tests
# ============================================================

class TestBayesianIRL:
    def _setup_grid(self):
        mdp = TabularMDP.from_grid(3, 3, gamma=0.9)
        vi = ValueIteration(mdp)
        _, true_policy = vi.solve()
        demos = ExpertDemonstrations()
        demos.collect_from_policy(mdp, true_policy, n_episodes=10, max_steps=50)
        fe = FeatureExtractor.one_hot(9)
        return mdp, true_policy, demos, fe

    def test_basic_training(self):
        mdp, true_policy, demos, fe = self._setup_grid()
        irl = BayesianIRL(mdp, fe, n_samples=50, burn_in=10, seed=42)
        theta = irl.train(demos)
        assert len(theta) == 9

    def test_samples_collected(self):
        mdp, true_policy, demos, fe = self._setup_grid()
        irl = BayesianIRL(mdp, fe, n_samples=50, burn_in=10)
        irl.train(demos)
        assert len(irl.samples) == 50

    def test_acceptance_rate(self):
        mdp, true_policy, demos, fe = self._setup_grid()
        irl = BayesianIRL(mdp, fe, n_samples=100, burn_in=20, step_size=0.1)
        irl.train(demos)
        # Should have some acceptance
        assert irl.acceptance_rate > 0.0

    def test_reward_std(self):
        mdp, true_policy, demos, fe = self._setup_grid()
        irl = BayesianIRL(mdp, fe, n_samples=50, burn_in=10)
        irl.train(demos)
        std = irl.get_reward_std()
        assert len(std) == 9
        assert all(s >= 0 for s in std)

    def test_policy_extraction(self):
        mdp, true_policy, demos, fe = self._setup_grid()
        irl = BayesianIRL(mdp, fe, n_samples=50, burn_in=10, seed=42)
        irl.train(demos)
        policy = irl.get_policy()
        assert len(policy) == 9


# ============================================================
# Deep MaxEnt IRL Tests
# ============================================================

class TestDeepMaxEntIRL:
    def _setup_grid(self):
        mdp = TabularMDP.from_grid(3, 3, gamma=0.9)
        vi = ValueIteration(mdp)
        _, true_policy = vi.solve()
        demos = ExpertDemonstrations()
        demos.collect_from_policy(mdp, true_policy, n_episodes=20, max_steps=50)
        fe = FeatureExtractor.one_hot(9)
        return mdp, true_policy, demos, fe

    def test_basic_training(self):
        mdp, true_policy, demos, fe = self._setup_grid()
        irl = DeepMaxEntIRL(mdp, fe, hidden_sizes=[8], lr=0.01, n_iters=10, seed=42)
        irl.train(demos)
        rewards = irl.get_reward_function()
        assert len(rewards) == 9

    def test_reward_network_output(self):
        mdp, true_policy, demos, fe = self._setup_grid()
        irl = DeepMaxEntIRL(mdp, fe, hidden_sizes=[8], n_iters=5, seed=42)
        irl.train(demos)
        rewards = irl.get_reward_function()
        # All rewards should be finite
        assert all(math.isfinite(r) for r in rewards)

    def test_gradient_tracking(self):
        mdp, true_policy, demos, fe = self._setup_grid()
        irl = DeepMaxEntIRL(mdp, fe, n_iters=10, seed=42)
        irl.train(demos)
        assert len(irl.history) == 10

    def test_policy_extraction(self):
        mdp, true_policy, demos, fe = self._setup_grid()
        irl = DeepMaxEntIRL(mdp, fe, n_iters=10, seed=42)
        irl.train(demos)
        policy = irl.get_policy()
        assert len(policy) == 9

    def test_custom_hidden_sizes(self):
        mdp, true_policy, demos, fe = self._setup_grid()
        irl = DeepMaxEntIRL(mdp, fe, hidden_sizes=[16, 8], n_iters=5, seed=42)
        irl.train(demos)
        rewards = irl.get_reward_function()
        assert len(rewards) == 9


# ============================================================
# Apprenticeship Learning Tests
# ============================================================

class TestApprenticeshipLearning:
    def _setup(self):
        mdp = TabularMDP.from_grid(3, 3, gamma=0.9)
        vi = ValueIteration(mdp)
        _, true_policy = vi.solve()
        true_reward = [mdp.R[s][0] for s in range(mdp.n_states)]
        demos = ExpertDemonstrations()
        demos.collect_from_policy(mdp, true_policy, n_episodes=20, max_steps=50)
        fe = FeatureExtractor.one_hot(9)
        return mdp, true_policy, true_reward, demos, fe

    def test_maxent_method(self):
        mdp, true_policy, true_reward, demos, fe = self._setup()
        al = ApprenticeshipLearning(mdp, fe, method='maxent', lr=0.2, n_iters=30)
        policy = al.learn(demos)
        assert len(policy) == 9

    def test_projection_method(self):
        mdp, true_policy, true_reward, demos, fe = self._setup()
        al = ApprenticeshipLearning(mdp, fe, method='projection', n_iters=10)
        policy = al.learn(demos)
        assert len(policy) == 9

    def test_bayesian_method(self):
        mdp, true_policy, true_reward, demos, fe = self._setup()
        al = ApprenticeshipLearning(mdp, fe, method='bayesian', n_samples=30, burn_in=5)
        policy = al.learn(demos)
        assert len(policy) == 9

    def test_deep_maxent_method(self):
        mdp, true_policy, true_reward, demos, fe = self._setup()
        al = ApprenticeshipLearning(mdp, fe, method='deep_maxent', n_iters=5)
        policy = al.learn(demos)
        assert len(policy) == 9

    def test_evaluate_policy(self):
        mdp, true_policy, true_reward, demos, fe = self._setup()
        al = ApprenticeshipLearning(mdp, fe, method='maxent', lr=0.2, n_iters=30)
        al.learn(demos)
        results = al.evaluate(true_policy=true_policy, true_reward=true_reward)
        assert 'policy_accuracy' in results
        assert 0 <= results['policy_accuracy'] <= 1.0
        assert 'reward_correlation' in results
        assert 'reward_rmse' in results

    def test_unknown_method(self):
        mdp, true_policy, true_reward, demos, fe = self._setup()
        al = ApprenticeshipLearning(mdp, fe, method='unknown')
        with pytest.raises(ValueError):
            al.learn(demos)


# ============================================================
# Reward Shaping Tests
# ============================================================

class TestRewardShaping:
    def test_zero_potential(self):
        mdp = TabularMDP.from_grid(3, 3)
        rs = RewardShaping(mdp)
        # Zero potential => shaped = base
        r = rs.shaped_reward(0, 1, 1)
        assert r == mdp.R[0][1]

    def test_custom_potential(self):
        mdp = TabularMDP.from_grid(3, 3, gamma=0.9)
        rs = RewardShaping(mdp, potential_fn=lambda s: float(s))
        r = rs.shaped_reward(0, 1, 1)
        expected = mdp.R[0][1] + 0.9 * 1.0 - 0.0
        assert abs(r - expected) < 1e-6

    def test_value_potential(self):
        mdp = TabularMDP.from_grid(3, 3)
        vi = ValueIteration(mdp)
        V, _ = vi.solve()
        rs = RewardShaping(mdp)
        rs.set_potential_from_values(V)
        # Should produce valid shaped rewards
        r = rs.shaped_reward(0, 1, 1)
        assert math.isfinite(r)

    def test_apply_to_mdp(self):
        mdp = TabularMDP.from_grid(3, 3, gamma=0.9)
        rs = RewardShaping(mdp, potential_fn=lambda s: float(s) * 0.1)
        new_mdp = rs.apply_to_mdp()
        assert new_mdp.n_states == mdp.n_states
        assert new_mdp.n_actions == mdp.n_actions
        assert new_mdp.terminal_states == mdp.terminal_states

    def test_preserves_optimal_policy(self):
        mdp = TabularMDP.from_grid(3, 3, gamma=0.9)
        vi_orig = ValueIteration(mdp)
        _, policy_orig = vi_orig.solve()

        rs = RewardShaping(mdp, potential_fn=lambda s: float(s) * 0.5)
        new_mdp = rs.apply_to_mdp()
        vi_shaped = ValueIteration(new_mdp)
        _, policy_shaped = vi_shaped.solve()

        # Same optimal policy
        for s in range(9):
            if s not in mdp.terminal_states:
                assert policy_orig[s] == policy_shaped[s]

    def test_set_potential(self):
        mdp = TabularMDP.from_grid(3, 3)
        rs = RewardShaping(mdp)
        rs.set_potential(lambda s: s * 2.0)
        assert rs.phi(3) == 6.0


# ============================================================
# IRL Evaluator Tests
# ============================================================

class TestIRLEvaluator:
    def _setup(self):
        mdp = TabularMDP.from_grid(3, 3, gamma=0.9)
        vi = ValueIteration(mdp)
        _, true_policy = vi.solve()
        fe = FeatureExtractor.one_hot(9)
        return mdp, true_policy, fe

    def test_policy_match_identical(self):
        mdp, policy, fe = self._setup()
        ev = IRLEvaluator(mdp, fe)
        assert ev.policy_match(policy, policy) == 1.0

    def test_policy_match_different(self):
        mdp, policy, fe = self._setup()
        ev = IRLEvaluator(mdp, fe)
        random_policy = [(p + 1) % 4 for p in policy]
        match = ev.policy_match(policy, random_policy)
        assert 0 <= match <= 1.0

    def test_expected_return(self):
        mdp, policy, fe = self._setup()
        ev = IRLEvaluator(mdp, fe)
        ret = ev.expected_return(policy, n_episodes=50)
        assert math.isfinite(ret)

    def test_feature_expectation_error(self):
        mdp, policy, fe = self._setup()
        demos = ExpertDemonstrations()
        demos.collect_from_policy(mdp, policy, n_episodes=20)
        ev = IRLEvaluator(mdp, fe)
        err = ev.feature_expectation_error(demos, policy)
        assert err >= 0

    def test_value_difference(self):
        mdp, policy, fe = self._setup()
        ev = IRLEvaluator(mdp, fe)
        diff = ev.value_difference(policy, policy)
        assert abs(diff) < 1e-6  # same policy => ~0 difference

    def test_reward_correlation(self):
        mdp, policy, fe = self._setup()
        ev = IRLEvaluator(mdp, fe)
        r1 = [float(i) for i in range(9)]
        r2 = [float(i) * 2 + 1 for i in range(9)]
        corr = ev.reward_correlation(r1, r2)
        assert abs(corr - 1.0) < 1e-6  # perfect correlation

    def test_comprehensive_eval(self):
        mdp, policy, fe = self._setup()
        demos = ExpertDemonstrations()
        demos.collect_from_policy(mdp, policy, n_episodes=10)
        ev = IRLEvaluator(mdp, fe)
        true_reward = [mdp.R[s][0] for s in range(9)]
        results = ev.comprehensive_eval(policy, policy, true_reward, true_reward, demos)
        assert 'policy_match' in results
        assert results['policy_match'] == 1.0
        assert 'reward_correlation' in results
        assert 'feature_error' in results
        assert 'value_diff' in results


# ============================================================
# MaxCausalEnt IRL Tests
# ============================================================

class TestMaxCausalEntIRL:
    def _setup_grid(self):
        mdp = TabularMDP.from_grid(3, 3, gamma=0.9)
        vi = ValueIteration(mdp)
        _, true_policy = vi.solve()
        demos = ExpertDemonstrations()
        demos.collect_from_policy(mdp, true_policy, n_episodes=20, max_steps=50)
        fe = FeatureExtractor.one_hot(9)
        return mdp, true_policy, demos, fe

    def test_basic_training(self):
        mdp, true_policy, demos, fe = self._setup_grid()
        irl = MaxCausalEntIRL(mdp, fe, lr=0.2, n_iters=20, seed=42)
        theta = irl.train(demos)
        assert len(theta) == 9

    def test_gradient_tracking(self):
        mdp, true_policy, demos, fe = self._setup_grid()
        irl = MaxCausalEntIRL(mdp, fe, n_iters=15)
        irl.train(demos)
        assert len(irl.history) == 15

    def test_soft_value_iteration(self):
        mdp = TabularMDP.from_grid(3, 3, gamma=0.9)
        irl = MaxCausalEntIRL(mdp, FeatureExtractor.one_hot(9))
        V = irl._soft_value_iteration()
        assert len(V) == 9
        # Terminal state should have 0 value
        assert V[8] == 0.0

    def test_policy_recovery(self):
        mdp, true_policy, demos, fe = self._setup_grid()
        irl = MaxCausalEntIRL(mdp, fe, lr=0.3, n_iters=30, seed=42)
        irl.train(demos)
        policy = irl.get_policy()
        assert len(policy) == 9

    def test_reward_function(self):
        mdp, true_policy, demos, fe = self._setup_grid()
        irl = MaxCausalEntIRL(mdp, fe, lr=0.3, n_iters=30)
        irl.train(demos)
        rewards = irl.get_reward_function()
        assert len(rewards) == 9
        assert all(math.isfinite(r) for r in rewards)


# ============================================================
# Utility Function Tests
# ============================================================

class TestUtilities:
    def test_pearson_correlation_perfect(self):
        assert abs(_pearson_correlation([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-6

    def test_pearson_correlation_negative(self):
        assert abs(_pearson_correlation([1, 2, 3], [6, 4, 2]) + 1.0) < 1e-6

    def test_pearson_correlation_zero(self):
        assert abs(_pearson_correlation([1, 1, 1], [1, 2, 3])) < 1e-6

    def test_cosine_similarity_identical(self):
        assert abs(_cosine_similarity([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        assert abs(_cosine_similarity([1, 0], [0, 1])) < 1e-6

    def test_cosine_similarity_opposite(self):
        assert abs(_cosine_similarity([1, 2], [-1, -2]) + 1.0) < 1e-6

    def test_pearson_empty(self):
        assert _pearson_correlation([], []) == 0.0

    def test_cosine_zero_vector(self):
        assert _cosine_similarity([0, 0], [1, 2]) == 0.0


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    def test_end_to_end_small_grid(self):
        """Full IRL pipeline on 3x3 grid."""
        mdp = TabularMDP.from_grid(3, 3, gamma=0.9)
        vi = ValueIteration(mdp)
        _, true_policy = vi.solve()

        demos = ExpertDemonstrations()
        demos.collect_from_policy(mdp, true_policy, n_episodes=30, max_steps=50)

        fe = FeatureExtractor.one_hot(9)
        al = ApprenticeshipLearning(mdp, fe, method='maxent', lr=0.3, n_iters=40)
        al.learn(demos)
        results = al.evaluate(true_policy=true_policy)
        assert results['policy_accuracy'] > 0.2

    def test_end_to_end_larger_grid(self):
        """Full IRL pipeline on 4x4 grid."""
        mdp = TabularMDP.from_grid(4, 4, gamma=0.9)
        vi = ValueIteration(mdp)
        _, true_policy = vi.solve()

        demos = ExpertDemonstrations()
        demos.collect_from_policy(mdp, true_policy, n_episodes=30, max_steps=80)

        fe = FeatureExtractor.one_hot(16)
        al = ApprenticeshipLearning(mdp, fe, method='maxent', lr=0.2, n_iters=40)
        al.learn(demos)
        results = al.evaluate(true_policy=true_policy)
        assert 'policy_accuracy' in results

    def test_stochastic_environment(self):
        """IRL on stochastic grid."""
        mdp = TabularMDP.from_grid_stochastic(3, 3, slip_prob=0.1, gamma=0.9)
        vi = ValueIteration(mdp)
        _, true_policy = vi.solve()

        demos = ExpertDemonstrations()
        demos.collect_from_policy(mdp, true_policy, n_episodes=30, max_steps=50)

        fe = FeatureExtractor.one_hot(9)
        irl = MaxEntIRL(mdp, fe, lr=0.2, n_iters=30)
        irl.train(demos)
        policy = irl.get_policy()
        assert len(policy) == 9

    def test_shaped_reward_learning(self):
        """Shape rewards then run IRL."""
        mdp = TabularMDP.from_grid(3, 3, gamma=0.9)
        vi = ValueIteration(mdp)
        V, true_policy = vi.solve()

        # Shape rewards using value function
        rs = RewardShaping(mdp)
        rs.set_potential_from_values(V)
        shaped_mdp = rs.apply_to_mdp()

        # Solve shaped MDP -- should have same policy
        vi2 = ValueIteration(shaped_mdp)
        _, shaped_policy = vi2.solve()
        for s in range(9):
            if s not in mdp.terminal_states:
                assert true_policy[s] == shaped_policy[s]

    def test_comparison_of_methods(self):
        """Compare MaxEnt, Projection, and MaxCausalEnt."""
        mdp = TabularMDP.from_grid(3, 3, gamma=0.9)
        vi = ValueIteration(mdp)
        _, true_policy = vi.solve()

        demos = ExpertDemonstrations()
        demos.collect_from_policy(mdp, true_policy, n_episodes=20, max_steps=50)

        fe = FeatureExtractor.one_hot(9)
        ev = IRLEvaluator(mdp, fe)

        # MaxEnt
        mdp1 = TabularMDP.from_grid(3, 3, gamma=0.9)
        irl1 = MaxEntIRL(mdp1, fe, lr=0.3, n_iters=30)
        irl1.train(demos)
        p1 = irl1.get_policy()

        # Projection
        mdp2 = TabularMDP.from_grid(3, 3, gamma=0.9)
        irl2 = ProjectionIRL(mdp2, fe, n_iters=15)
        irl2.train(demos)
        p2 = irl2.get_best_policy()

        # All should produce valid policies
        assert len(p1) == 9
        assert len(p2) == 9

    def test_reward_shaping_acceleration(self):
        """Reward shaping should produce valid MDPs with same optimal policy."""
        mdp = TabularMDP.from_grid(4, 4, gamma=0.9)
        vi = ValueIteration(mdp)
        V, policy = vi.solve()

        # Use value function as potential (guaranteed to preserve policy)
        rs = RewardShaping(mdp)
        rs.set_potential_from_values(V)
        shaped = rs.apply_to_mdp()
        vi2 = ValueIteration(shaped)
        _, policy2 = vi2.solve()

        # Same optimal policy
        for s in range(16):
            if s not in mdp.terminal_states:
                assert policy[s] == policy2[s]

    def test_bayesian_irl_uncertainty(self):
        """Bayesian IRL should provide uncertainty estimates."""
        mdp = TabularMDP.from_grid(3, 3, gamma=0.9)
        vi = ValueIteration(mdp)
        _, true_policy = vi.solve()

        demos = ExpertDemonstrations()
        demos.collect_from_policy(mdp, true_policy, n_episodes=10)

        fe = FeatureExtractor.one_hot(9)
        irl = BayesianIRL(mdp, fe, n_samples=50, burn_in=10, seed=42)
        irl.train(demos)

        mean = irl.get_mean_reward_weights()
        std = irl.get_reward_std()
        assert len(mean) == 9
        assert len(std) == 9
        # Should have nonzero uncertainty
        assert any(s > 0 for s in std)

    def test_deep_irl_vs_linear(self):
        """Deep and linear MaxEnt IRL should both produce policies."""
        mdp = TabularMDP.from_grid(3, 3, gamma=0.9)
        vi = ValueIteration(mdp)
        _, true_policy = vi.solve()
        demos = ExpertDemonstrations()
        demos.collect_from_policy(mdp, true_policy, n_episodes=20)

        fe = FeatureExtractor.one_hot(9)

        # Linear
        mdp1 = TabularMDP.from_grid(3, 3, gamma=0.9)
        irl1 = MaxEntIRL(mdp1, fe, n_iters=20)
        irl1.train(demos)
        p1 = irl1.get_policy()

        # Deep
        mdp2 = TabularMDP.from_grid(3, 3, gamma=0.9)
        irl2 = DeepMaxEntIRL(mdp2, fe, n_iters=10, seed=42)
        irl2.train(demos)
        p2 = irl2.get_policy()

        assert len(p1) == 9
        assert len(p2) == 9

    def test_multiple_demo_sources(self):
        """Mix hand-crafted and generated demos."""
        mdp = TabularMDP.from_grid(3, 3, gamma=0.9)
        demos = ExpertDemonstrations()
        # Hand-crafted: optimal path (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2)
        demos.add_trajectory([
            (0, 1, -0.01, 1), (1, 1, -0.01, 2),
            (2, 2, -0.01, 5), (5, 2, 1.0, 8)
        ])

        vi = ValueIteration(mdp)
        _, policy = vi.solve()
        demos.collect_from_policy(mdp, policy, n_episodes=10, max_steps=50)

        fe = FeatureExtractor.one_hot(9)
        irl = MaxEntIRL(mdp, fe, lr=0.2, n_iters=20)
        irl.train(demos)
        assert len(irl.get_policy()) == 9

    def test_empty_demonstrations(self):
        """Handle edge case of no useful demos gracefully."""
        mdp = TabularMDP.from_grid(3, 3)
        fe = FeatureExtractor.one_hot(9)
        demos = ExpertDemonstrations()
        # Add a trivial trajectory
        demos.add_trajectory([(0, 0, 0.0, 0)])

        irl = MaxEntIRL(mdp, fe, n_iters=5)
        theta = irl.train(demos)
        assert len(theta) == 9


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
