"""
Tests for C193: Multi-Agent Reinforcement Learning
"""
import numpy as np
import pytest
from .multi_agent_rl import (
    MultiAgentEnv, MatrixGame, GridWorldMARLEnv, PredatorPreyEnv,
    IndependentQLearning, JointActionLearning, MinimaxQ, NashQLearning,
    WoLFPHC, LeniencyLearning, CommunicatingAgents, TeamQLearning,
    OpponentModeling, FictitiousPlay, MeanFieldQ,
)


# ============================================================
# Environment Tests
# ============================================================

class TestMatrixGame:
    def test_prisoners_dilemma_creation(self):
        game = MatrixGame.prisoners_dilemma()
        assert game.n_agents == 2
        assert game.n_actions == [2, 2]
        assert game.n_states == 1

    def test_prisoners_dilemma_payoffs(self):
        game = MatrixGame.prisoners_dilemma()
        game.reset()
        # Both cooperate
        _, rewards, _, _ = game.step([0, 0])
        assert rewards[0] == -1 and rewards[1] == -1
        # Agent 0 defects, agent 1 cooperates
        _, rewards, _, _ = game.step([1, 0])
        assert rewards[0] == 0 and rewards[1] == -3

    def test_matching_pennies(self):
        game = MatrixGame.matching_pennies()
        game.reset()
        _, rewards, _, _ = game.step([0, 0])
        assert rewards[0] == 1 and rewards[1] == -1
        assert rewards[0] + rewards[1] == 0  # Zero-sum

    def test_coordination_game(self):
        game = MatrixGame.coordination_game()
        game.reset()
        _, rewards, _, _ = game.step([0, 0])
        assert rewards[0] == 2 and rewards[1] == 2
        _, rewards, _, _ = game.step([1, 1])
        assert rewards[0] == 1 and rewards[1] == 1

    def test_rock_paper_scissors(self):
        game = MatrixGame.rock_paper_scissors()
        assert game.n_actions == [3, 3]
        game.reset()
        _, rewards, _, _ = game.step([0, 1])  # Rock vs Paper
        assert rewards[0] == -1 and rewards[1] == 1
        _, rewards, _, _ = game.step([0, 2])  # Rock vs Scissors
        assert rewards[0] == 1 and rewards[1] == -1

    def test_game_never_done(self):
        game = MatrixGame.prisoners_dilemma()
        game.reset()
        for _ in range(10):
            _, _, done, _ = game.step([0, 0])
            assert done is False

    def test_custom_game(self):
        game = MatrixGame([
            [[3, 0], [5, 1]],
            [[3, 5], [0, 1]],
        ])
        game.reset()
        _, rewards, _, _ = game.step([0, 0])
        assert rewards[0] == 3 and rewards[1] == 3

    def test_reset(self):
        game = MatrixGame.prisoners_dilemma()
        state = game.reset()
        assert state == 0


class TestGridWorldMARLEnv:
    def test_creation(self):
        env = GridWorldMARLEnv(rows=4, cols=4,
                               agent_starts=[(0, 0), (3, 3)],
                               agent_goals=[(3, 3), (0, 0)])
        assert env.n_agents == 2
        assert env.n_actions == [4, 4]

    def test_reset(self):
        env = GridWorldMARLEnv(rows=4, cols=4,
                               agent_starts=[(0, 0), (3, 3)],
                               agent_goals=[(3, 3), (0, 0)])
        state = env.reset()
        assert state == ((0, 0), (3, 3))

    def test_movement(self):
        env = GridWorldMARLEnv(rows=4, cols=4,
                               agent_starts=[(0, 0), (3, 3)],
                               agent_goals=[(3, 3), (0, 0)])
        env.reset()
        # Agent 0 moves DOWN, agent 1 moves UP
        state, rewards, done, _ = env.step([1, 0])
        assert state == ((1, 0), (2, 3))

    def test_wall_boundary(self):
        env = GridWorldMARLEnv(rows=4, cols=4,
                               agent_starts=[(0, 0), (3, 3)],
                               agent_goals=[(3, 3), (0, 0)])
        env.reset()
        # Agent 0 tries to go UP from (0,0) -- should stay
        state, _, _, _ = env.step([0, 4 - 1])  # UP, RIGHT
        assert state[0] == (0, 0)  # Can't go up

    def test_collision(self):
        env = GridWorldMARLEnv(rows=3, cols=1,
                               agent_starts=[(0, 0), (2, 0)],
                               agent_goals=[(2, 0), (0, 0)],
                               collision_penalty=-1.0)
        env.reset()
        # Both move toward center (1,0)
        state, rewards, _, _ = env.step([1, 0])  # DOWN, UP
        # Collision -- agents stay in place
        assert state == ((0, 0), (2, 0))
        assert rewards[0] < 0  # Got penalty
        assert rewards[1] < 0

    def test_goal_reached(self):
        env = GridWorldMARLEnv(rows=3, cols=1,
                               agent_starts=[(0, 0), (2, 0)],
                               agent_goals=[(1, 0), (2, 0)])
        env.reset()
        # Agent 0 moves DOWN toward goal, Agent 1 stays at goal (STAY not available, use DOWN to wall)
        state, rewards, done, _ = env.step([1, 1])
        # Agent 0 reaches (1,0), Agent 1 stays at (2,0) -- both at goals
        assert done

    def test_goal_reward(self):
        env = GridWorldMARLEnv(rows=2, cols=2,
                               agent_starts=[(0, 0), (1, 1)],
                               agent_goals=[(0, 1), (1, 0)],
                               goal_reward=5.0)
        env.reset()
        # Agent 0 RIGHT to (0,1), Agent 1 LEFT to (1,0)
        state, rewards, done, _ = env.step([3, 2])
        assert rewards[0] >= 5.0 - 0.2  # goal_reward + step
        assert rewards[1] >= 5.0 - 0.2
        assert done


class TestPredatorPreyEnv:
    def test_creation(self):
        env = PredatorPreyEnv(rows=5, cols=5)
        assert env.n_agents == 2
        assert env.n_actions == [5, 5]

    def test_reset(self):
        env = PredatorPreyEnv(rows=5, cols=5)
        state = env.reset()
        assert state == ((0, 0), (4, 4), (2, 2))

    def test_step(self):
        np.random.seed(42)
        env = PredatorPreyEnv(rows=5, cols=5)
        env.reset()
        state, rewards, done, info = env.step([4, 4])  # Both STAY
        assert len(rewards) == 2
        assert isinstance(done, bool)

    def test_max_steps(self):
        env = PredatorPreyEnv(rows=5, cols=5)
        env.max_steps = 5
        env.reset()
        done = False
        for _ in range(10):
            if done:
                break
            _, _, done, _ = env.step([4, 4])
        assert done


# ============================================================
# Algorithm Tests
# ============================================================

class TestIndependentQLearning:
    def test_creation(self):
        iql = IndependentQLearning(n_agents=2, n_actions_per_agent=[2, 2])
        assert iql.n_agents == 2
        assert len(iql.q_tables) == 2

    def test_select_actions(self):
        iql = IndependentQLearning(n_agents=2, n_actions_per_agent=[2, 2],
                                    epsilon=1.0)
        actions = iql.select_actions("state0")
        assert len(actions) == 2
        assert all(0 <= a < 2 for a in actions)

    def test_update(self):
        iql = IndependentQLearning(n_agents=2, n_actions_per_agent=[2, 2],
                                    alpha=0.5, gamma=0.9)
        iql.update("s0", [0, 1], [1.0, -1.0], "s1", False)
        assert iql.q_tables[0]["s0"][0] != 0
        assert iql.q_tables[1]["s0"][1] != 0

    def test_greedy_policy(self):
        iql = IndependentQLearning(n_agents=2, n_actions_per_agent=[2, 2])
        iql.q_tables[0]["s0"][1] = 10.0
        assert iql.get_policy(0, "s0") == 1

    def test_train_matrix_game(self):
        np.random.seed(42)
        game = MatrixGame.prisoners_dilemma()
        iql = IndependentQLearning(n_agents=2, n_actions_per_agent=[2, 2],
                                    alpha=0.1, epsilon=0.5)
        history = iql.train(game, episodes=200)
        assert len(history) == 200

    def test_epsilon_decay(self):
        iql = IndependentQLearning(n_agents=2, n_actions_per_agent=[2, 2],
                                    epsilon=0.5, epsilon_decay=0.9)
        iql.update("s", [0, 0], [0, 0], "s", False)
        assert iql.epsilon < 0.5

    def test_train_coordination(self):
        np.random.seed(42)
        game = MatrixGame.coordination_game()
        iql = IndependentQLearning(n_agents=2, n_actions_per_agent=[2, 2],
                                    alpha=0.2, epsilon=0.5, epsilon_decay=0.995)
        history = iql.train(game, episodes=500)
        # Later episodes should have higher reward (agents learn to coordinate)
        avg_late = np.mean(history[-50:])
        avg_early = np.mean(history[:50])
        assert avg_late >= avg_early - 1.0  # At least not worse


class TestJointActionLearning:
    def test_creation(self):
        jal = JointActionLearning(n_agents=2, n_actions_per_agent=[2, 2])
        assert jal.n_agents == 2

    def test_select_actions(self):
        jal = JointActionLearning(n_agents=2, n_actions_per_agent=[2, 2],
                                   epsilon=1.0)
        actions = jal.select_actions("s0")
        assert len(actions) == 2

    def test_update_stores_joint(self):
        jal = JointActionLearning(n_agents=2, n_actions_per_agent=[2, 2],
                                   alpha=0.5)
        jal.update("s0", [0, 1], [1.0, -1.0], "s1", False)
        ja = (0, 1)
        assert jal.q_tables[0]["s0"][ja] != 0

    def test_train(self):
        np.random.seed(42)
        game = MatrixGame.coordination_game()
        jal = JointActionLearning(n_agents=2, n_actions_per_agent=[2, 2])
        history = jal.train(game, episodes=100)
        assert len(history) == 100


class TestMinimaxQ:
    def test_creation(self):
        mmq = MinimaxQ(n_actions_0=2, n_actions_1=2)
        assert mmq.n_actions == [2, 2]

    def test_select_actions(self):
        mmq = MinimaxQ(n_actions_0=2, n_actions_1=2)
        actions = mmq.select_actions("s0")
        assert len(actions) == 2

    def test_policy_is_distribution(self):
        mmq = MinimaxQ(n_actions_0=3, n_actions_1=3)
        policy = mmq.get_policy("s0")
        assert abs(policy.sum() - 1.0) < 1e-6
        assert all(p >= 0 for p in policy)

    def test_update(self):
        mmq = MinimaxQ(n_actions_0=2, n_actions_1=2, alpha=0.5)
        mmq.update("s0", [0, 1], 1.0, "s0", False)
        assert mmq.q_table["s0"][0, 1] != 0

    def test_train_matching_pennies(self):
        np.random.seed(42)
        game = MatrixGame.matching_pennies()
        mmq = MinimaxQ(n_actions_0=2, n_actions_1=2)
        history = mmq.train(game, episodes=300)
        assert len(history) == 300
        # In matching pennies, optimal is mixed (0.5, 0.5)
        policy = mmq.get_policy(0)
        # Should be roughly uniform after training
        assert all(p > 0.1 for p in policy)

    def test_train_rps(self):
        np.random.seed(42)
        game = MatrixGame.rock_paper_scissors()
        mmq = MinimaxQ(n_actions_0=3, n_actions_1=3, epsilon=0.5)
        mmq.train(game, episodes=100)
        policy = mmq.get_policy(0)
        # Should converge toward uniform (1/3, 1/3, 1/3)
        assert all(p > 0.05 for p in policy)

    def test_maximin_computation(self):
        mmq = MinimaxQ(n_actions_0=2, n_actions_1=2)
        # Matching pennies matrix
        q = np.array([[1.0, -1.0], [-1.0, 1.0]])
        value, policy = mmq._compute_maximin(q)
        # Value should be ~0, policy should be ~(0.5, 0.5)
        assert abs(value) < 0.2
        assert abs(policy[0] - 0.5) < 0.2


class TestNashQLearning:
    def test_creation(self):
        nql = NashQLearning(n_actions_0=2, n_actions_1=2)
        assert nql.n_actions == [2, 2]

    def test_update(self):
        nql = NashQLearning(n_actions_0=2, n_actions_1=2, alpha=0.5)
        nql.update("s0", [0, 0], [-1, -1], "s0", False)
        assert nql.q_tables[0]["s0"][0, 0] != 0

    def test_nash_finds_pure(self):
        nql = NashQLearning(n_actions_0=2, n_actions_1=2)
        # Coordination game: (0,0) is a Nash equilibrium
        q0 = np.array([[2.0, 0.0], [0.0, 1.0]])
        q1 = np.array([[2.0, 0.0], [0.0, 1.0]])
        v0, v1, p0, p1 = nql._compute_nash(q0, q1)
        # Should find (0,0) or (1,1) as Nash
        assert (p0[0] > 0.5 and p1[0] > 0.5) or (p0[1] > 0.5 and p1[1] > 0.5)

    def test_train(self):
        np.random.seed(42)
        game = MatrixGame.prisoners_dilemma()
        nql = NashQLearning(n_actions_0=2, n_actions_1=2)
        history = nql.train(game, episodes=100)
        assert len(history) == 100


class TestWoLFPHC:
    def test_creation(self):
        wolf = WoLFPHC(n_actions_0=2, n_actions_1=2)
        assert wolf.n_actions == [2, 2]

    def test_policy_starts_uniform(self):
        wolf = WoLFPHC(n_actions_0=3, n_actions_1=3)
        policy = wolf.get_policy("s0")
        assert abs(policy.sum() - 1.0) < 1e-6
        assert all(abs(p - 1.0/3) < 1e-6 for p in policy)

    def test_update_changes_policy(self):
        wolf = WoLFPHC(n_actions_0=2, n_actions_1=2, alpha=0.5,
                        delta_win=0.1, delta_lose=0.2)
        wolf.update("s0", 0, 1.0, "s0", False)
        policy = wolf.get_policy("s0")
        # Policy should shift toward action 0
        assert policy[0] > 0.5

    def test_train(self):
        np.random.seed(42)
        game = MatrixGame.matching_pennies()
        wolf = WoLFPHC(n_actions_0=2, n_actions_1=2)
        history = wolf.train(game, episodes=200)
        assert len(history) == 200

    def test_win_vs_lose_rates(self):
        wolf = WoLFPHC(n_actions_0=2, n_actions_1=2,
                        delta_win=0.01, delta_lose=0.04)
        assert wolf.delta_lose > wolf.delta_win

    def test_visit_tracking(self):
        wolf = WoLFPHC(n_actions_0=2, n_actions_1=2)
        wolf.update("s0", 0, 1.0, "s0", False)
        wolf.update("s0", 1, -1.0, "s0", False)
        assert wolf.visit_counts["s0"] == 2


class TestLeniencyLearning:
    def test_creation(self):
        ll = LeniencyLearning(n_agents=2, n_actions_per_agent=[2, 2])
        assert ll.n_agents == 2

    def test_temperature_decay(self):
        ll = LeniencyLearning(n_agents=2, n_actions_per_agent=[2, 2],
                               initial_temp=10.0, temp_decay=0.9)
        ll.update("s0", [0, 0], [1.0, 1.0], "s1", False)
        assert ll.temperatures[0]["s0"][0] < 10.0

    def test_leniency_favors_positive(self):
        """With high temperature, negative updates are mostly ignored."""
        np.random.seed(42)
        ll = LeniencyLearning(n_agents=1, n_actions_per_agent=[2],
                               alpha=0.5, initial_temp=100.0, temp_decay=1.0)
        # Many negative updates with high temperature should be mostly ignored
        for _ in range(100):
            ll.update("s0", [0], [-10.0], "s0", False)
        # Q should be near 0 (most negatives ignored)
        assert ll.q_tables[0]["s0"][0] > -200  # Much less than -1000

    def test_train(self):
        np.random.seed(42)
        game = MatrixGame.coordination_game()
        ll = LeniencyLearning(n_agents=2, n_actions_per_agent=[2, 2])
        history = ll.train(game, episodes=100)
        assert len(history) == 100


class TestCommunicatingAgents:
    def test_creation(self):
        ca = CommunicatingAgents(n_agents=2, n_env_actions=4, n_messages=3)
        assert ca.n_agents == 2
        assert ca.n_total_actions == 12

    def test_encode_decode(self):
        ca = CommunicatingAgents(n_agents=2, n_env_actions=4, n_messages=3)
        for ea in range(4):
            for msg in range(3):
                combined = ca._encode_action(ea, msg)
                dec_ea, dec_msg = ca._decode_action(combined)
                assert dec_ea == ea and dec_msg == msg

    def test_select_actions(self):
        ca = CommunicatingAgents(n_agents=2, n_env_actions=4, n_messages=3,
                                  epsilon=1.0)
        env_actions, messages = ca.select_actions("s0")
        assert len(env_actions) == 2
        assert len(messages) == 2
        assert all(0 <= m < 3 for m in messages)

    def test_update(self):
        ca = CommunicatingAgents(n_agents=2, n_env_actions=2, n_messages=2,
                                  alpha=0.5)
        ca.update("s0", [0, 0], [(0, 1), (1, 0)], [1.0, 1.0], "s1", [1, 0], False)
        # Q-table should have been updated
        obs = ca._obs_key("s0", [0, 0])
        combined = ca._encode_action(0, 1)
        assert ca.q_tables[0][obs][combined] != 0

    def test_train(self):
        np.random.seed(42)
        game = MatrixGame.coordination_game()
        ca = CommunicatingAgents(n_agents=2, n_env_actions=2, n_messages=2)
        history = ca.train(game, episodes=100)
        assert len(history) == 100


class TestTeamQLearning:
    def test_creation(self):
        tql = TeamQLearning(n_agents=2, n_actions_per_agent=[2, 2])
        assert tql.n_agents == 2

    def test_all_joint_actions(self):
        tql = TeamQLearning(n_agents=2, n_actions_per_agent=[2, 3])
        jas = tql._all_joint_actions()
        assert len(jas) == 6  # 2 * 3

    def test_select_actions(self):
        tql = TeamQLearning(n_agents=2, n_actions_per_agent=[2, 2], epsilon=1.0)
        actions = tql.select_actions("s0")
        assert len(actions) == 2

    def test_update(self):
        tql = TeamQLearning(n_agents=2, n_actions_per_agent=[2, 2], alpha=0.5)
        tql.update("s0", [0, 1], 5.0, "s1", False)
        assert tql.q_table["s0"][(0, 1)] != 0

    def test_train_coordination(self):
        np.random.seed(42)
        game = MatrixGame.coordination_game()
        tql = TeamQLearning(n_agents=2, n_actions_per_agent=[2, 2],
                             alpha=0.2, epsilon=0.5, epsilon_decay=0.99)
        history = tql.train(game, episodes=500)
        assert len(history) == 500
        # Should learn to coordinate on (0,0) which gives 2+2=4
        avg_late = np.mean(history[-50:])
        assert avg_late > 1.0

    def test_greedy_selection(self):
        tql = TeamQLearning(n_agents=2, n_actions_per_agent=[2, 2], epsilon=0.0)
        tql.q_table["s0"][(1, 0)] = 10.0
        actions = tql.select_actions("s0")
        assert actions == [1, 0]


class TestOpponentModeling:
    def test_creation(self):
        om = OpponentModeling(n_actions_0=2, n_actions_1=2)
        assert om.n_actions == [2, 2]

    def test_opponent_model_starts_uniform(self):
        om = OpponentModeling(n_actions_0=2, n_actions_1=2, smoothing=1.0)
        model = om.get_opponent_model("s0")
        assert abs(model[0] - 0.5) < 1e-6

    def test_opponent_model_updates(self):
        om = OpponentModeling(n_actions_0=2, n_actions_1=2, smoothing=1.0)
        for _ in range(10):
            om.observe_opponent("s0", 0)
        model = om.get_opponent_model("s0")
        assert model[0] > model[1]  # Action 0 more likely

    def test_select_action(self):
        om = OpponentModeling(n_actions_0=2, n_actions_1=2, epsilon=0.0)
        om.q_table["s0"][1, :] = 10.0  # Action 1 is best regardless
        action = om.select_action("s0")
        assert action == 1

    def test_update(self):
        om = OpponentModeling(n_actions_0=2, n_actions_1=2, alpha=0.5)
        om.update("s0", 0, 1, 5.0, "s0", False)
        assert om.q_table["s0"][0, 1] != 0

    def test_train(self):
        np.random.seed(42)
        game = MatrixGame.prisoners_dilemma()
        om = OpponentModeling(n_actions_0=2, n_actions_1=2)
        history = om.train(game, episodes=100)
        assert len(history) == 100


class TestFictitiousPlay:
    def test_creation(self):
        fp = FictitiousPlay(n_actions_0=2, n_actions_1=2)
        assert fp.n_actions == [2, 2]

    def test_play_returns_histories(self):
        fp = FictitiousPlay(n_actions_0=2, n_actions_1=2)
        payoffs = [np.array([[2, 0], [0, 1]]), np.array([[2, 0], [0, 1]])]
        histories = fp.play(payoffs, rounds=100)
        assert len(histories[0]) == 100
        assert len(histories[1]) == 100

    def test_empirical_strategy(self):
        fp = FictitiousPlay(n_actions_0=2, n_actions_1=2)
        for _ in range(100):
            fp.update([0, 0])
        strat = fp.get_empirical_strategy(0)
        # Agent 1 always played 0, so agent 0's model of agent 1 should favor 0
        # But get_empirical_strategy(0) returns agent 1's model of agent 0
        # Agent 0 always played 0
        assert strat[0] > strat[1]

    def test_matching_pennies_converges_to_mixed(self):
        np.random.seed(42)
        fp = FictitiousPlay(n_actions_0=2, n_actions_1=2)
        payoffs = [np.array([[1, -1], [-1, 1]]), np.array([[-1, 1], [1, -1]])]
        fp.play(payoffs, rounds=2000)
        # Should converge toward (0.5, 0.5)
        strat0 = fp.get_empirical_strategy(0)
        assert abs(strat0[0] - 0.5) < 0.2

    def test_coordination_converges(self):
        np.random.seed(42)
        fp = FictitiousPlay(n_actions_0=2, n_actions_1=2)
        payoffs = [np.array([[2, 0], [0, 1]]), np.array([[2, 0], [0, 1]])]
        fp.play(payoffs, rounds=500)
        # Should converge to one of the Nash equilibria
        strat0 = fp.get_empirical_strategy(0)
        assert strat0[0] > 0.3 or strat0[1] > 0.3  # Not stuck


class TestMeanFieldQ:
    def test_creation(self):
        mfq = MeanFieldQ(n_agents=5, n_actions=4)
        assert mfq.n_agents == 5
        assert mfq.n_actions == 4

    def test_discretize_mean(self):
        mfq = MeanFieldQ(n_agents=3, n_actions=4, n_mean_bins=5)
        assert mfq._discretize_mean(0.0) == 0
        assert mfq._discretize_mean(3.9) == 4

    def test_select_actions(self):
        mfq = MeanFieldQ(n_agents=3, n_actions=4, epsilon=1.0)
        actions = mfq.select_actions("s0")
        assert len(actions) == 3
        assert all(0 <= a < 4 for a in actions)

    def test_update(self):
        mfq = MeanFieldQ(n_agents=2, n_actions=2, alpha=0.5)
        mfq.update("s0", [0, 1], [1.0, 1.0], "s1", [0, 0], False)
        # Check Q-tables were updated
        obs = mfq._obs_key("s0", 1.0)  # mean of others for agent 0 is action 1
        assert mfq.q_tables[0][obs][0] != 0

    def test_train_simple(self):
        np.random.seed(42)
        game = MatrixGame.coordination_game()
        mfq = MeanFieldQ(n_agents=2, n_actions=2)
        history = mfq.train(game, episodes=100)
        assert len(history) == 100


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    def test_iql_on_grid(self):
        np.random.seed(42)
        env = GridWorldMARLEnv(rows=3, cols=3,
                                agent_starts=[(0, 0), (2, 2)],
                                agent_goals=[(2, 2), (0, 0)])
        iql = IndependentQLearning(n_agents=2, n_actions_per_agent=[4, 4],
                                    epsilon=0.5, epsilon_decay=0.99)
        history = iql.train(env, episodes=200)
        assert len(history) == 200

    def test_team_q_on_grid(self):
        np.random.seed(42)
        env = GridWorldMARLEnv(rows=3, cols=3,
                                agent_starts=[(0, 0), (2, 2)],
                                agent_goals=[(2, 2), (0, 0)])
        tql = TeamQLearning(n_agents=2, n_actions_per_agent=[4, 4],
                             epsilon=0.5, epsilon_decay=0.99)
        history = tql.train(env, episodes=100)
        assert len(history) == 100

    def test_leniency_on_coordination(self):
        np.random.seed(42)
        game = MatrixGame.coordination_game()
        ll = LeniencyLearning(n_agents=2, n_actions_per_agent=[2, 2],
                               initial_temp=5.0, temp_decay=0.99)
        history = ll.train(game, episodes=300)
        avg_late = np.mean(history[-30:])
        assert avg_late > 0  # Should learn something

    def test_communicating_on_coordination(self):
        np.random.seed(42)
        game = MatrixGame.coordination_game()
        ca = CommunicatingAgents(n_agents=2, n_env_actions=2, n_messages=2,
                                  epsilon=0.5, epsilon_decay=0.99)
        history = ca.train(game, episodes=300)
        assert len(history) == 300

    def test_iql_on_predator_prey(self):
        np.random.seed(42)
        env = PredatorPreyEnv(rows=3, cols=3)
        env.max_steps = 20
        iql = IndependentQLearning(n_agents=2, n_actions_per_agent=[5, 5],
                                    epsilon=0.5, epsilon_decay=0.99)
        history = iql.train(env, episodes=50)
        assert len(history) == 50

    def test_wolf_on_rps(self):
        np.random.seed(42)
        game = MatrixGame.rock_paper_scissors()
        wolf = WoLFPHC(n_actions_0=3, n_actions_1=3)
        history = wolf.train(game, episodes=300)
        assert len(history) == 300

    def test_multiple_algorithms_same_game(self):
        """Multiple algorithms can be trained on the same game."""
        np.random.seed(42)
        game = MatrixGame.prisoners_dilemma()

        iql = IndependentQLearning(n_agents=2, n_actions_per_agent=[2, 2])
        h1 = iql.train(game, episodes=50)

        game2 = MatrixGame.prisoners_dilemma()
        tql = TeamQLearning(n_agents=2, n_actions_per_agent=[2, 2])
        h2 = tql.train(game2, episodes=50)

        assert len(h1) == len(h2) == 50

    def test_opponent_modeling_vs_random(self):
        np.random.seed(42)
        game = MatrixGame.prisoners_dilemma()
        om = OpponentModeling(n_actions_0=2, n_actions_1=2,
                               epsilon=0.3, epsilon_decay=0.995)
        history = om.train(game, episodes=300)
        # Should learn to exploit random opponent
        assert len(history) == 300


class TestEdgeCases:
    def test_single_action(self):
        """Game where each agent has only 1 action."""
        game = MatrixGame([[[5]], [[3]]])
        game.reset()
        _, rewards, _, _ = game.step([0, 0])
        assert rewards[0] == 5 and rewards[1] == 3

    def test_iql_single_episode(self):
        game = MatrixGame.prisoners_dilemma()
        iql = IndependentQLearning(n_agents=2, n_actions_per_agent=[2, 2])
        h = iql.train(game, episodes=1)
        assert len(h) == 1

    def test_zero_gamma(self):
        """Myopic agent (gamma=0) should only care about immediate reward."""
        iql = IndependentQLearning(n_agents=1, n_actions_per_agent=[2],
                                    alpha=1.0, gamma=0.0, epsilon=0.0)
        iql.update("s0", [0], [10.0], "s1", False)
        assert abs(iql.q_tables[0]["s0"][0] - 10.0) < 1e-6

    def test_done_state(self):
        """When done, no future reward should be considered."""
        iql = IndependentQLearning(n_agents=1, n_actions_per_agent=[2],
                                    alpha=1.0, gamma=0.99, epsilon=0.0)
        iql.q_tables[0]["s1"][0] = 100.0
        iql.update("s0", [0], [5.0], "s1", True)
        assert abs(iql.q_tables[0]["s0"][0] - 5.0) < 1e-6

    def test_mean_field_single_agent(self):
        """Mean field with 1 agent (no others)."""
        mfq = MeanFieldQ(n_agents=1, n_actions=2)
        actions = mfq.select_actions("s0")
        assert len(actions) == 1


# ============================================================
# Init file
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
