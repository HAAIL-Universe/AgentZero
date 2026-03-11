"""
Tests for C181: Multi-Agent Reinforcement Learning
"""

import sys
import os
import math
import random

sys.path.insert(0, os.path.dirname(__file__))
from multi_agent_rl import (
    MultiAgentEnv, MatrixGame, GridWorldMA,
    IndependentLearners, CentralizedCritic, CommChannel,
    QMIX, SelfPlay, TeamReward, MATrainer, TournamentEvaluator,
    _argmax, _softmax, _sample_categorical, _flatten_obs,
)


# ============================================================
# Helper tests
# ============================================================

class TestHelpers:
    def test_argmax_basic(self):
        assert _argmax([1, 3, 2]) == 1

    def test_argmax_first_wins(self):
        assert _argmax([5, 5, 5]) == 0

    def test_argmax_negative(self):
        assert _argmax([-3, -1, -2]) == 1

    def test_softmax_sums_to_one(self):
        p = _softmax([1.0, 2.0, 3.0])
        assert abs(sum(p) - 1.0) < 1e-6

    def test_softmax_ordering(self):
        p = _softmax([1.0, 2.0, 3.0])
        assert p[2] > p[1] > p[0]

    def test_softmax_uniform(self):
        p = _softmax([0.0, 0.0, 0.0])
        for v in p:
            assert abs(v - 1.0 / 3) < 1e-6

    def test_sample_categorical_valid(self):
        from multi_agent_rl import _RNG
        rng = _RNG(42)
        counts = [0, 0, 0]
        for _ in range(100):
            idx = _sample_categorical([0.7, 0.2, 0.1], rng)
            assert 0 <= idx <= 2
            counts[idx] += 1
        assert counts[0] > counts[2]  # most likely should be most frequent

    def test_flatten_obs(self):
        result = _flatten_obs([[1, 2], [3, 4]])
        assert result == [1, 2, 3, 4]

    def test_flatten_obs_scalars(self):
        result = _flatten_obs([1, 2, 3])
        assert result == [1, 2, 3]


# ============================================================
# MatrixGame tests
# ============================================================

class TestMatrixGame:
    def test_prisoners_dilemma_creation(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        assert game.n_agents == 2
        assert game.action_size == 2

    def test_prisoners_dilemma_cooperate(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        game.reset()
        obs, rewards, done, info = game.step([0, 0])  # both cooperate
        assert rewards == [-1, -1]
        assert done is True

    def test_prisoners_dilemma_defect(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        game.reset()
        obs, rewards, done, info = game.step([1, 1])  # both defect
        assert rewards == [-2, -2]

    def test_prisoners_dilemma_asymmetric(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        game.reset()
        obs, rewards, done, info = game.step([0, 1])  # cooperate, defect
        assert rewards == [-3, 0]

    def test_stag_hunt(self):
        game = MatrixGame(game_def=MatrixGame.STAG_HUNT)
        game.reset()
        _, rewards, _, _ = game.step([0, 0])  # both hunt stag
        assert rewards == [4, 4]

    def test_stag_hunt_miscoordination(self):
        game = MatrixGame(game_def=MatrixGame.STAG_HUNT)
        game.reset()
        _, rewards, _, _ = game.step([0, 1])  # stag vs hare
        assert rewards == [0, 3]

    def test_chicken(self):
        game = MatrixGame(game_def=MatrixGame.CHICKEN)
        game.reset()
        _, rewards, _, _ = game.step([0, 0])  # both straight
        assert rewards == [0, 0]

    def test_chicken_crash(self):
        game = MatrixGame(game_def=MatrixGame.CHICKEN)
        game.reset()
        _, rewards, _, _ = game.step([1, 1])  # both swerve
        assert rewards == [-5, -5]

    def test_matching_pennies(self):
        game = MatrixGame(game_def=MatrixGame.MATCHING_PENNIES)
        game.reset()
        _, rewards, _, _ = game.step([0, 0])  # both heads
        assert rewards == [1, -1]

    def test_matching_pennies_zero_sum(self):
        game = MatrixGame(game_def=MatrixGame.MATCHING_PENNIES)
        game.reset()
        _, rewards, _, _ = game.step([0, 1])
        assert rewards[0] + rewards[1] == 0

    def test_coordination_game(self):
        game = MatrixGame(game_def=MatrixGame.COORDINATION)
        game.reset()
        _, rewards, _, _ = game.step([0, 0])
        assert rewards == [2, 2]

    def test_single_round_done(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        game.reset()
        _, _, done, _ = game.step([0, 0])
        assert done is True

    def test_multi_round(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA, n_rounds=3)
        obs = game.reset()
        assert len(obs) == 2
        _, _, done, _ = game.step([0, 0])
        assert done is False
        _, _, done, _ = game.step([1, 1])
        assert done is False
        _, _, done, _ = game.step([0, 1])
        assert done is True

    def test_multi_round_history(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA, n_rounds=3)
        game.reset()
        game.step([0, 1])
        assert game.history == [(0, 1)]
        game.step([1, 0])
        assert len(game.history) == 2

    def test_obs_size_single_round(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA, n_rounds=1)
        obs = game.reset()
        assert obs == [[1.0], [1.0]]

    def test_obs_size_multi_round(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA, n_rounds=3)
        obs = game.reset()
        assert len(obs[0]) == game.obs_size

    def test_custom_payoffs(self):
        payoffs = [
            [(10, 10), (0, 0)],
            [(0, 0), (5, 5)],
        ]
        game = MatrixGame(payoffs=payoffs)
        game.reset()
        _, rewards, _, _ = game.step([0, 0])
        assert rewards == [10, 10]

    def test_info_contains_actions(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        game.reset()
        _, _, _, info = game.step([0, 1])
        assert info['actions'] == (0, 1)


# ============================================================
# GridWorldMA tests
# ============================================================

class TestGridWorldMA:
    def test_cooperative_creation(self):
        env = GridWorldMA(rows=3, cols=3, n_agents=2, mode='cooperative')
        assert env.n_agents == 2
        assert env.action_size == 5
        assert env.obs_size == 9 * 2

    def test_reset_returns_obs(self):
        env = GridWorldMA(rows=3, cols=3, n_agents=2)
        obs = env.reset()
        assert len(obs) == 2
        assert len(obs[0]) == env.obs_size

    def test_obs_encoding(self):
        env = GridWorldMA(rows=3, cols=3, n_agents=2,
                          starts=[(0, 0), (2, 2)])
        obs = env.reset()
        # Agent 0 at (0,0) -> index 0 in first grid_size block
        assert obs[0][0] == 1.0
        # Agent 1 at (2,2) -> index 8 in second grid_size block
        assert obs[0][9 + 8] == 1.0

    def test_movement(self):
        env = GridWorldMA(rows=3, cols=3, n_agents=2,
                          starts=[(0, 0), (2, 2)])
        env.reset()
        # Agent 0: move right (1), Agent 1: stay (4)
        obs, _, _, info = env.step([1, 4])
        assert info['positions'][0] == (0, 1)
        assert info['positions'][1] == (2, 2)

    def test_wall_blocking(self):
        env = GridWorldMA(rows=3, cols=3, n_agents=1,
                          starts=[(1, 1)], walls={(0, 1)})
        env.reset()
        # Try to move up into wall
        obs, _, _, info = env.step([0])
        assert info['positions'][0] == (1, 1)  # didn't move

    def test_boundary_blocking(self):
        env = GridWorldMA(rows=3, cols=3, n_agents=1,
                          starts=[(0, 0)])
        env.reset()
        obs, _, _, info = env.step([0])  # up from (0,0)
        assert info['positions'][0] == (0, 0)

    def test_collision_resolution(self):
        env = GridWorldMA(rows=3, cols=3, n_agents=2,
                          starts=[(0, 0), (0, 2)])
        env.reset()
        # Both try to reach (0,1)
        obs, _, _, info = env.step([1, 3])  # right, left
        # Collision: neither moves
        assert info['positions'][0] == (0, 0)
        assert info['positions'][1] == (0, 2)

    def test_cooperative_goal(self):
        env = GridWorldMA(rows=2, cols=2, n_agents=2, mode='cooperative',
                          starts=[(0, 0), (0, 1)],
                          goals=[(1, 0), (1, 1)])
        env.reset()
        obs, rewards, done, info = env.step([2, 2])  # both move down
        assert done is True
        assert all(r > 0 for r in rewards)

    def test_cooperative_partial(self):
        env = GridWorldMA(rows=2, cols=2, n_agents=2, mode='cooperative',
                          starts=[(0, 0), (0, 1)],
                          goals=[(1, 0), (1, 1)])
        env.reset()
        obs, rewards, done, info = env.step([2, 4])  # one moves, one stays
        assert done is False

    def test_predator_prey(self):
        env = GridWorldMA(rows=3, cols=3, n_agents=2, mode='predator_prey',
                          starts=[(1, 1), (1, 2)])
        env.reset()
        # Predator (agent 1) catches prey (agent 0)
        obs, rewards, done, info = env.step([4, 3])  # prey stays, predator left
        assert done is True
        assert rewards[0] == -1.0  # prey caught
        assert rewards[1] == 1.0   # predator wins

    def test_predator_prey_chase(self):
        env = GridWorldMA(rows=3, cols=3, n_agents=2, mode='predator_prey',
                          starts=[(0, 0), (2, 2)])
        env.reset()
        obs, rewards, done, info = env.step([4, 4])  # both stay
        assert done is False
        assert rewards[0] > 0  # prey survives

    def test_competitive(self):
        env = GridWorldMA(rows=2, cols=2, n_agents=2, mode='competitive',
                          starts=[(0, 0), (0, 1)],
                          goals=[(1, 1), (1, 1)])
        env.reset()
        # Agent 1 reaches goal first (one step right+down, but let's just go right first)
        obs, rewards, done, info = env.step([2, 2])  # both down
        # Agent 1 at (1,1) = goal
        if info['positions'][1] == (1, 1) and info['positions'][0] != (1, 1):
            assert rewards[1] == 1.0
            assert done is True

    def test_max_steps(self):
        env = GridWorldMA(rows=3, cols=3, n_agents=2, max_steps=2,
                          starts=[(0, 0), (2, 2)])
        env.reset()
        env.step([4, 4])  # stay
        _, _, done, _ = env.step([4, 4])  # stay
        assert done is True

    def test_stay_action(self):
        env = GridWorldMA(rows=3, cols=3, n_agents=1,
                          starts=[(1, 1)])
        env.reset()
        _, _, _, info = env.step([4])  # stay
        assert info['positions'][0] == (1, 1)


# ============================================================
# IndependentLearners tests
# ============================================================

class TestIndependentLearners:
    def test_creation(self):
        il = IndependentLearners(n_agents=2, obs_size=1, action_size=2)
        assert il.n_agents == 2
        assert len(il.q_tables) == 2

    def test_select_actions(self):
        il = IndependentLearners(n_agents=2, obs_size=1, action_size=2, epsilon=0.0)
        actions = il.select_actions([[1.0], [1.0]])
        assert len(actions) == 2
        assert all(0 <= a < 2 for a in actions)

    def test_exploration(self):
        il = IndependentLearners(n_agents=1, obs_size=1, action_size=2, epsilon=1.0)
        actions_seen = set()
        for _ in range(50):
            a = il.select_actions([[1.0]])
            actions_seen.add(a[0])
        assert len(actions_seen) == 2  # should see both actions

    def test_update_changes_q(self):
        il = IndependentLearners(n_agents=1, obs_size=1, action_size=2, lr=0.5)
        obs = [[1.0]]
        il.update(obs, [0], [10.0], obs, False)
        q = il._get_q(0, [1.0])
        assert q[0] > 0  # Q for action 0 should increase

    def test_greedy_actions(self):
        il = IndependentLearners(n_agents=1, obs_size=1, action_size=2, lr=0.5)
        obs = [[1.0]]
        for _ in range(10):
            il.update(obs, [1], [10.0], obs, True)
        actions = il.get_greedy_actions(obs)
        assert actions[0] == 1

    def test_epsilon_decay(self):
        il = IndependentLearners(n_agents=1, obs_size=1, action_size=2,
                                  epsilon=1.0, epsilon_decay=0.9)
        initial_eps = il.epsilon
        il.update([[1.0]], [0], [0.0], [[1.0]], False)
        assert il.epsilon < initial_eps

    def test_multiple_agents_independent(self):
        il = IndependentLearners(n_agents=2, obs_size=1, action_size=2, lr=0.5)
        # Train agent 0 on action 0, agent 1 on action 1
        for _ in range(10):
            il.update([[1.0], [1.0]], [0, 1], [10.0, 10.0], [[1.0], [1.0]], True)
        assert il._get_q(0, [1.0])[0] > il._get_q(0, [1.0])[1]
        assert il._get_q(1, [1.0])[1] > il._get_q(1, [1.0])[0]

    def test_different_states(self):
        il = IndependentLearners(n_agents=1, obs_size=2, action_size=2, lr=0.5)
        il.update([[1.0, 0.0]], [0], [10.0], [[1.0, 0.0]], True)
        il.update([[0.0, 1.0]], [1], [10.0], [[0.0, 1.0]], True)
        assert il._get_q(0, [1.0, 0.0])[0] > 0
        assert il._get_q(0, [0.0, 1.0])[1] > 0


# ============================================================
# CentralizedCritic tests
# ============================================================

class TestCentralizedCritic:
    def test_creation(self):
        cc = CentralizedCritic(n_agents=2, obs_size=4, action_size=2)
        assert cc.n_agents == 2
        assert len(cc.policies) == 2
        assert cc.critic is not None

    def test_select_actions(self):
        cc = CentralizedCritic(n_agents=2, obs_size=4, action_size=2)
        obs = [[0.0] * 4, [0.0] * 4]
        actions = cc.select_actions(obs, explore=True)
        assert len(actions) == 2
        assert all(0 <= a < 2 for a in actions)

    def test_select_actions_no_explore(self):
        cc = CentralizedCritic(n_agents=2, obs_size=4, action_size=2)
        obs = [[0.0] * 4, [0.0] * 4]
        actions = cc.select_actions(obs, explore=False)
        assert len(actions) == 2

    def test_joint_input(self):
        cc = CentralizedCritic(n_agents=2, obs_size=2, action_size=2)
        joint = cc._joint_input([[1, 0], [0, 1]], [0, 1])
        # obs0 + obs1 + action0_onehot + action1_onehot
        assert len(joint) == 2 + 2 + 2 + 2

    def test_update_runs(self):
        cc = CentralizedCritic(n_agents=2, obs_size=2, action_size=2, lr=0.01)
        obs = [[1, 0], [0, 1]]
        next_obs = [[0, 1], [1, 0]]
        actions = [0, 1]
        next_actions = [1, 0]
        loss = cc.update(obs, actions, [1.0, -1.0], next_obs, next_actions, done=False)
        assert isinstance(loss, float)

    def test_update_terminal(self):
        cc = CentralizedCritic(n_agents=2, obs_size=2, action_size=2, lr=0.01)
        obs = [[1, 0], [0, 1]]
        next_obs = [[0, 0], [0, 0]]
        loss = cc.update(obs, [0, 0], [5.0, 5.0], next_obs, [0, 0], done=True)
        assert isinstance(loss, float)


# ============================================================
# CommChannel tests
# ============================================================

class TestCommChannel:
    def test_creation(self):
        ch = CommChannel(n_agents=2, obs_size=4, msg_size=3)
        assert ch.msg_size == 3
        assert len(ch.encoders) == 2

    def test_produce_messages(self):
        ch = CommChannel(n_agents=2, obs_size=4, msg_size=3)
        msgs = ch.produce_messages([[0] * 4, [0] * 4])
        assert len(msgs) == 2
        assert len(msgs[0]) == 3

    def test_aggregate_mean(self):
        ch = CommChannel(n_agents=3, obs_size=2, msg_size=2, aggregation='mean')
        messages = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        agg = ch.aggregate(messages, 0)  # mean of agents 1,2
        assert abs(agg[0] - 4.0) < 1e-6
        assert abs(agg[1] - 5.0) < 1e-6

    def test_aggregate_sum(self):
        ch = CommChannel(n_agents=2, obs_size=2, msg_size=2, aggregation='sum')
        messages = [[1.0, 2.0], [3.0, 4.0]]
        agg = ch.aggregate(messages, 0)
        assert abs(agg[0] - 3.0) < 1e-6
        assert abs(agg[1] - 4.0) < 1e-6

    def test_aggregate_max(self):
        ch = CommChannel(n_agents=3, obs_size=2, msg_size=2, aggregation='max')
        messages = [[1.0, 6.0], [3.0, 2.0], [5.0, 4.0]]
        agg = ch.aggregate(messages, 0)
        assert agg[0] == 5.0
        assert agg[1] == 4.0

    def test_augment_observations(self):
        ch = CommChannel(n_agents=2, obs_size=4, msg_size=3)
        obs = [[0] * 4, [0] * 4]
        augmented, messages = ch.augment_observations(obs)
        assert len(augmented) == 2
        assert len(augmented[0]) == 4 + 3  # obs + aggregated message

    def test_trainable_layers(self):
        ch = CommChannel(n_agents=2, obs_size=4, msg_size=3)
        layers = ch.get_trainable_layers()
        assert len(layers) > 0

    def test_single_agent_aggregate(self):
        ch = CommChannel(n_agents=1, obs_size=2, msg_size=2, aggregation='mean')
        messages = [[1.0, 2.0]]
        agg = ch.aggregate(messages, 0)
        assert agg == [0.0, 0.0]  # no others


# ============================================================
# QMIX tests
# ============================================================

class TestQMIX:
    def test_creation(self):
        qm = QMIX(n_agents=2, obs_size=4, action_size=2)
        assert len(qm.q_nets) == 2

    def test_get_q_values(self):
        qm = QMIX(n_agents=2, obs_size=4, action_size=2)
        obs = [[0] * 4, [0] * 4]
        q_vals = qm._get_q_values(obs)
        assert len(q_vals) == 2
        assert len(q_vals[0]) == 2

    def test_mix(self):
        qm = QMIX(n_agents=2, obs_size=4, action_size=2)
        state = [0] * 8
        q_tot = qm._mix([1.0, 2.0], state)
        assert isinstance(q_tot, float)

    def test_select_actions(self):
        qm = QMIX(n_agents=2, obs_size=4, action_size=2)
        obs = [[0] * 4, [0] * 4]
        actions = qm.select_actions(obs)
        assert len(actions) == 2

    def test_greedy_actions(self):
        qm = QMIX(n_agents=2, obs_size=4, action_size=2, epsilon=0.0)
        obs = [[0] * 4, [0] * 4]
        actions = qm.get_greedy_actions(obs)
        assert len(actions) == 2

    def test_store_and_update(self):
        qm = QMIX(n_agents=2, obs_size=2, action_size=2)
        obs = [[1, 0], [0, 1]]
        next_obs = [[0, 1], [1, 0]]
        for _ in range(40):
            qm.store(obs, [0, 1], [1.0, 1.0], next_obs, False)
        loss = qm.update(batch_size=16)
        assert isinstance(loss, float)

    def test_epsilon_decay(self):
        qm = QMIX(n_agents=2, obs_size=2, action_size=2, epsilon=1.0)
        obs = [[1, 0], [0, 1]]
        for _ in range(40):
            qm.store(obs, [0, 0], [0, 0], obs, False)
        initial_eps = qm.epsilon
        qm.update(batch_size=16)
        assert qm.epsilon < initial_eps

    def test_buffer_capacity(self):
        qm = QMIX(n_agents=2, obs_size=2, action_size=2)
        qm.buffer_capacity = 10
        obs = [[1, 0], [0, 1]]
        for i in range(20):
            qm.store(obs, [0, 0], [float(i), 0], obs, False)
        assert len(qm.buffer) == 10


# ============================================================
# SelfPlay tests
# ============================================================

class TestSelfPlay:
    def test_creation(self):
        sp = SelfPlay(obs_size=1, action_size=2, population_size=3)
        assert len(sp.population) == 3
        assert len(sp.elo_ratings) == 3

    def test_select_action(self):
        sp = SelfPlay(obs_size=1, action_size=2)
        action = sp.select_action(0, [1.0])
        assert 0 <= action < 2

    def test_select_opponents_population(self):
        sp = SelfPlay(obs_size=1, action_size=2, population_size=5, mode='population')
        i, j = sp.select_opponents()
        assert 0 <= i < 5
        assert 0 <= j < 5

    def test_select_opponents_latest(self):
        sp = SelfPlay(obs_size=1, action_size=2, population_size=5, mode='latest')
        i, j = sp.select_opponents()
        assert i == 4 and j == 4

    def test_select_opponents_latest_vs_historical(self):
        sp = SelfPlay(obs_size=1, action_size=2, population_size=5, mode='latest_vs_historical')
        i, j = sp.select_opponents()
        assert i == 4
        assert 0 <= j < 5

    def test_elo_update_win(self):
        sp = SelfPlay(obs_size=1, action_size=2)
        initial_0 = sp.elo_ratings[0]
        initial_1 = sp.elo_ratings[1]
        sp.update_elo(0, 1)
        assert sp.elo_ratings[0] > initial_0
        assert sp.elo_ratings[1] < initial_1

    def test_elo_update_draw(self):
        sp = SelfPlay(obs_size=1, action_size=2)
        initial_0 = sp.elo_ratings[0]
        sp.update_elo(0, 1, draw=True)
        # Equal rating draw should be ~no change
        assert abs(sp.elo_ratings[0] - initial_0) < 0.1

    def test_play_game(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        sp = SelfPlay(obs_size=1, action_size=2, population_size=3)
        rewards, hist_i, hist_j = sp.play_game(game, 0, 1)
        assert len(rewards) == 2
        assert len(hist_i[0]) > 0  # has observations
        assert len(hist_i[1]) > 0  # has actions

    def test_train_step(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        sp = SelfPlay(obs_size=1, action_size=2, population_size=3)
        rewards = sp.train_step(game)
        assert len(rewards) == 2
        assert sp.generation == 1

    def test_multiple_train_steps(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        sp = SelfPlay(obs_size=1, action_size=2, population_size=3)
        for _ in range(10):
            sp.train_step(game)
        assert sp.generation == 10

    def test_game_counts_track(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        sp = SelfPlay(obs_size=1, action_size=2, population_size=2, mode='latest')
        sp.train_step(game)
        assert sum(sp.game_counts) >= 2


# ============================================================
# TeamReward tests
# ============================================================

class TestTeamReward:
    def test_shared_mode(self):
        tr = TeamReward(mode='shared')
        shaped = tr.shape([2.0, 4.0])
        assert shaped == [3.0, 3.0]

    def test_shared_three_agents(self):
        tr = TeamReward(mode='shared')
        shaped = tr.shape([3.0, 6.0, 9.0])
        assert abs(shaped[0] - 6.0) < 1e-6
        assert shaped[0] == shaped[1] == shaped[2]

    def test_difference_mode(self):
        tr = TeamReward(mode='difference')
        # Team reward = (10 + 5) / 2 = 7.5
        # Without agent 0 = 4.0, Without agent 1 = 8.0
        shaped = tr.shape([10.0, 5.0], counterfactual_rewards=[4.0, 8.0])
        assert abs(shaped[0] - (7.5 - 4.0)) < 1e-6  # agent 0's contribution
        assert abs(shaped[1] - (7.5 - 8.0)) < 1e-6  # agent 1's contribution

    def test_difference_no_counterfactual(self):
        tr = TeamReward(mode='difference')
        shaped = tr.shape([1.0, 2.0])
        assert shaped == [1.0, 2.0]  # fallback to individual

    def test_shaped_mode(self):
        tr = TeamReward(mode='shaped', shaping_factor=0.5)
        shaped = tr.shape([2.0, 4.0])
        team = 3.0
        assert abs(shaped[0] - (0.5 * 3.0 + 0.5 * 2.0)) < 1e-6
        assert abs(shaped[1] - (0.5 * 3.0 + 0.5 * 4.0)) < 1e-6

    def test_shaped_pure_team(self):
        tr = TeamReward(mode='shaped', shaping_factor=1.0)
        shaped = tr.shape([2.0, 4.0])
        assert shaped == [3.0, 3.0]

    def test_shaped_pure_individual(self):
        tr = TeamReward(mode='shaped', shaping_factor=0.0)
        shaped = tr.shape([2.0, 4.0])
        assert shaped == [2.0, 4.0]


# ============================================================
# MATrainer tests
# ============================================================

class TestMATrainer:
    def test_iql_training(self):
        env = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        agents = IndependentLearners(n_agents=2, obs_size=1, action_size=2)
        trainer = MATrainer(env, agents)
        results = trainer.train(n_episodes=5)
        assert len(results) == 5

    def test_iql_episode_tracking(self):
        env = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        agents = IndependentLearners(n_agents=2, obs_size=1, action_size=2)
        trainer = MATrainer(env, agents)
        trainer.train(n_episodes=10)
        assert len(trainer.episode_rewards) == 10
        assert len(trainer.episode_lengths) == 10

    def test_iql_evaluate(self):
        env = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        agents = IndependentLearners(n_agents=2, obs_size=1, action_size=2)
        trainer = MATrainer(env, agents)
        result = trainer.evaluate(n_episodes=5)
        assert 'avg_rewards' in result
        assert 'avg_team_reward' in result

    def test_ctde_training(self):
        game = MatrixGame(game_def=MatrixGame.COORDINATION, n_rounds=1)
        agents = CentralizedCritic(n_agents=2, obs_size=1, action_size=2, lr=0.01)
        trainer = MATrainer(game, agents)
        results = trainer.train(n_episodes=5)
        assert len(results) == 5

    def test_qmix_training(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        agents = QMIX(n_agents=2, obs_size=1, action_size=2)
        trainer = MATrainer(game, agents)
        results = trainer.train(n_episodes=5)
        assert len(results) == 5

    def test_reward_shaping(self):
        env = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        agents = IndependentLearners(n_agents=2, obs_size=1, action_size=2)
        shaper = TeamReward(mode='shared')
        trainer = MATrainer(env, agents, reward_shaper=shaper)
        results = trainer.train(n_episodes=5)
        assert len(results) == 5

    def test_metrics(self):
        env = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        agents = IndependentLearners(n_agents=2, obs_size=1, action_size=2)
        trainer = MATrainer(env, agents)
        trainer.train(n_episodes=5)
        metrics = trainer.get_metrics()
        assert metrics['episodes'] == 5

    def test_grid_world_training(self):
        env = GridWorldMA(rows=3, cols=3, n_agents=2, mode='cooperative',
                          starts=[(0, 0), (0, 2)],
                          goals=[(2, 2), (2, 0)],
                          max_steps=20)
        agents = IndependentLearners(n_agents=2, obs_size=env.obs_size,
                                      action_size=env.action_size)
        trainer = MATrainer(env, agents)
        results = trainer.train(n_episodes=3)
        assert len(results) == 3

    def test_per_agent_rewards(self):
        env = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        agents = IndependentLearners(n_agents=2, obs_size=1, action_size=2)
        trainer = MATrainer(env, agents)
        trainer.train(n_episodes=5)
        assert len(trainer.per_agent_rewards) == 2
        assert len(trainer.per_agent_rewards[0]) == 5


# ============================================================
# TournamentEvaluator tests
# ============================================================

class TestTournamentEvaluator:
    def test_creation(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        te = TournamentEvaluator(game)
        assert te.results == {}

    def test_run_match(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        te = TournamentEvaluator(game)

        def always_cooperate(obs):
            return 0
        def always_defect(obs):
            return 1

        wa, wb, d, ra, rb = te.run_match(always_cooperate, always_defect, n_games=5)
        assert wa + wb + d == 5
        # Defector always wins against cooperator
        assert rb > ra

    def test_round_robin(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        te = TournamentEvaluator(game)

        def coop(obs): return 0
        def defect(obs): return 1

        results = te.round_robin([coop, defect], names=['Cooperator', 'Defector'], n_games=5)
        assert 'standings' in results
        assert 'elo' in results
        assert 'Cooperator' in results['standings']
        assert 'Defector' in results['standings']

    def test_round_robin_three_agents(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        te = TournamentEvaluator(game)

        def coop(obs): return 0
        def defect(obs): return 1
        def random_play(obs): return 0 if random.random() < 0.5 else 1

        results = te.round_robin([coop, defect, random_play],
                                 names=['Coop', 'Defect', 'Random'], n_games=10)
        assert len(results['standings']) == 3
        assert len(results['elo']) == 3

    def test_elo_defector_wins_pd(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        te = TournamentEvaluator(game)

        def coop(obs): return 0
        def defect(obs): return 1

        results = te.round_robin([coop, defect], names=['Coop', 'Defect'], n_games=20)
        # In PD, defector should dominate
        assert results['elo']['Defect'] > results['elo']['Coop']

    def test_rankings(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        te = TournamentEvaluator(game)

        def coop(obs): return 0
        def defect(obs): return 1

        te.round_robin([coop, defect], names=['Coop', 'Defect'], n_games=10)
        rankings = te.get_rankings()
        assert len(rankings) == 2
        assert rankings[0][1] >= rankings[1][1]  # sorted by Elo

    def test_empty_rankings(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        te = TournamentEvaluator(game)
        assert te.get_rankings() == []

    def test_default_names(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        te = TournamentEvaluator(game)

        def a(obs): return 0
        def b(obs): return 1

        results = te.round_robin([a, b], n_games=3)
        assert 'Agent_0' in results['standings']
        assert 'Agent_1' in results['standings']

    def test_game_counts(self):
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        te = TournamentEvaluator(game)

        def a(obs): return 0
        def b(obs): return 1

        results = te.round_robin([a, b], n_games=5)
        assert results['standings']['Agent_0']['games'] == 5
        assert results['standings']['Agent_1']['games'] == 5


# ============================================================
# Integration tests
# ============================================================

class TestIntegration:
    def test_iql_on_prisoners_dilemma(self):
        """IQL should learn to defect in Prisoner's Dilemma."""
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)
        agents = IndependentLearners(n_agents=2, obs_size=1, action_size=2,
                                      lr=0.3, epsilon=0.5, epsilon_decay=0.99)
        trainer = MATrainer(game, agents)
        trainer.train(n_episodes=200)
        # After training, agents should learn to defect (Nash equilibrium)
        actions = agents.get_greedy_actions([[1.0], [1.0]])
        # At least one should learn to defect
        assert 1 in actions

    def test_iql_on_coordination(self):
        """IQL should learn to coordinate."""
        game = MatrixGame(game_def=MatrixGame.COORDINATION)
        agents = IndependentLearners(n_agents=2, obs_size=1, action_size=2,
                                      lr=0.3, epsilon=0.5, epsilon_decay=0.99)
        trainer = MATrainer(game, agents)
        trainer.train(n_episodes=200)
        actions = agents.get_greedy_actions([[1.0], [1.0]])
        # Should coordinate on same action
        assert actions[0] == actions[1]

    def test_selfplay_trains(self):
        game = MatrixGame(game_def=MatrixGame.MATCHING_PENNIES)
        sp = SelfPlay(obs_size=1, action_size=2, population_size=3, lr=0.01)
        for _ in range(20):
            sp.train_step(game)
        # Elo should have diverged from initial 1000
        assert any(abs(e - 1000) > 0.01 for e in sp.elo_ratings)

    def test_comm_channel_with_iql(self):
        """CommChannel augments observations for IQL."""
        env = MatrixGame(game_def=MatrixGame.COORDINATION)
        ch = CommChannel(n_agents=2, obs_size=1, msg_size=2)

        obs = env.reset()
        augmented, msgs = ch.augment_observations(obs)
        assert len(augmented[0]) == 1 + 2  # obs + message

    def test_qmix_on_matrix_game(self):
        game = MatrixGame(game_def=MatrixGame.COORDINATION)
        agents = QMIX(n_agents=2, obs_size=1, action_size=2,
                       epsilon=0.5, epsilon_decay=0.99)
        trainer = MATrainer(game, agents)
        results = trainer.train(n_episodes=50)
        assert len(results) == 50

    def test_predator_prey_training(self):
        env = GridWorldMA(rows=3, cols=3, n_agents=2, mode='predator_prey',
                          starts=[(0, 0), (2, 2)], max_steps=10)
        agents = IndependentLearners(n_agents=2, obs_size=env.obs_size,
                                      action_size=env.action_size, epsilon=0.5)
        trainer = MATrainer(env, agents)
        results = trainer.train(n_episodes=10)
        assert len(results) == 10

    def test_tournament_after_training(self):
        """Train two populations, then tournament."""
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA)

        # Train IQL agents
        il = IndependentLearners(n_agents=2, obs_size=1, action_size=2, lr=0.3)
        for _ in range(100):
            obs = game.reset()
            actions = il.select_actions(obs)
            next_obs, rewards, done, _ = game.step(actions)
            il.update(obs, actions, rewards, next_obs, done)

        def iql_agent(obs):
            q = il._get_q(0, obs)
            return _argmax(q)

        def always_defect(obs):
            return 1

        te = TournamentEvaluator(game)
        results = te.round_robin([iql_agent, always_defect],
                                 names=['IQL', 'AlwaysDefect'], n_games=20)
        assert 'IQL' in results['standings']

    def test_shared_reward_training(self):
        env = GridWorldMA(rows=3, cols=3, n_agents=2, mode='cooperative',
                          starts=[(0, 0), (0, 2)],
                          goals=[(2, 2), (2, 0)],
                          max_steps=15)
        agents = IndependentLearners(n_agents=2, obs_size=env.obs_size,
                                      action_size=env.action_size)
        shaper = TeamReward(mode='shared')
        trainer = MATrainer(env, agents, reward_shaper=shaper)
        results = trainer.train(n_episodes=5)
        assert len(results) == 5

    def test_competitive_grid(self):
        env = GridWorldMA(rows=3, cols=3, n_agents=2, mode='competitive',
                          starts=[(0, 0), (0, 2)], max_steps=15)
        agents = IndependentLearners(n_agents=2, obs_size=env.obs_size,
                                      action_size=env.action_size)
        trainer = MATrainer(env, agents)
        results = trainer.train(n_episodes=5)
        assert len(results) == 5

    def test_ctde_on_grid(self):
        env = GridWorldMA(rows=3, cols=3, n_agents=2, mode='cooperative',
                          starts=[(0, 0), (0, 2)],
                          goals=[(2, 2), (2, 0)],
                          max_steps=10)
        agents = CentralizedCritic(n_agents=2, obs_size=env.obs_size,
                                    action_size=env.action_size,
                                    hidden_size=16, lr=0.01)
        trainer = MATrainer(env, agents)
        results = trainer.train(n_episodes=3)
        assert len(results) == 3

    def test_multiround_pd_iql(self):
        """IQL on iterated Prisoner's Dilemma."""
        game = MatrixGame(game_def=MatrixGame.PRISONERS_DILEMMA, n_rounds=5)
        agents = IndependentLearners(n_agents=2, obs_size=game.obs_size,
                                      action_size=2, lr=0.1)
        trainer = MATrainer(game, agents)
        results = trainer.train(n_episodes=20)
        assert len(results) == 20

    def test_full_pipeline(self):
        """Full pipeline: create env, train, evaluate, tournament."""
        env = MatrixGame(game_def=MatrixGame.COORDINATION)

        # 1. Train IQL
        agents = IndependentLearners(n_agents=2, obs_size=1, action_size=2,
                                      lr=0.3, epsilon=0.5, epsilon_decay=0.99)
        trainer = MATrainer(env, agents)
        trainer.train(n_episodes=100)

        # 2. Evaluate
        eval_result = trainer.evaluate(n_episodes=10)
        assert 'avg_team_reward' in eval_result

        # 3. Tournament vs fixed strategies
        def trained_agent(obs):
            q = agents._get_q(0, obs)
            return _argmax(q)
        def action_a(obs): return 0
        def action_b(obs): return 1

        te = TournamentEvaluator(env)
        results = te.round_robin([trained_agent, action_a, action_b],
                                 names=['Trained', 'AlwaysA', 'AlwaysB'],
                                 n_games=10)
        rankings = te.get_rankings()
        assert len(rankings) == 3


# ============================================================
# Run all tests
# ============================================================

def run_tests():
    test_classes = [
        TestHelpers, TestMatrixGame, TestGridWorldMA,
        TestIndependentLearners, TestCentralizedCritic, TestCommChannel,
        TestQMIX, TestSelfPlay, TestTeamReward, TestMATrainer,
        TestTournamentEvaluator, TestIntegration,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        for method_name in sorted(methods):
            total += 1
            try:
                getattr(instance, method_name)()
                passed += 1
            except Exception as e:
                failed += 1
                errors.append(f"  FAIL {cls.__name__}.{method_name}: {e}")

    print(f"Tests: {passed}/{total} passed, {failed} failed")
    for err in errors:
        print(err)

    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
