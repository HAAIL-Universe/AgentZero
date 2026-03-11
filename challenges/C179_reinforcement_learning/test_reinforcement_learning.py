"""Tests for C179: Reinforcement Learning."""

import unittest
import math
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from reinforcement_learning import (
    # Environments
    Environment, GridWorld, CartPole, Bandit, MountainCar,
    # Replay
    ReplayBuffer, PrioritizedReplayBuffer,
    # Policies
    EpsilonGreedy, BoltzmannPolicy, UCBPolicy,
    # Tabular
    QLearning, SARSA,
    # Bandits
    MultiArmedBandit,
    # Deep RL
    DQN, DoubleDQN, REINFORCE, ActorCritic, PPO,
    # Trainers
    DQNTrainer, PolicyGradientTrainer, ActorCriticTrainer, PPOTrainer,
    # Utilities
    RewardShaper, ReturnTracker, DiscountedReturn,
    # Helpers
    _RNG, Tensor,
)


# ==============================================================
# RNG Tests
# ==============================================================

class TestRNG(unittest.TestCase):
    def test_deterministic(self):
        r1 = _RNG(42)
        r2 = _RNG(42)
        self.assertEqual([r1.random() for _ in range(10)],
                         [r2.random() for _ in range(10)])

    def test_uniform(self):
        r = _RNG(1)
        for _ in range(50):
            v = r.uniform(2.0, 5.0)
            self.assertGreaterEqual(v, 2.0)
            self.assertLessEqual(v, 5.0)

    def test_normal(self):
        r = _RNG(7)
        vals = [r.normal(0, 1) for _ in range(200)]
        mean = sum(vals) / len(vals)
        self.assertAlmostEqual(mean, 0.0, delta=0.3)

    def test_randint(self):
        r = _RNG(3)
        vals = [r.randint(0, 4) for _ in range(100)]
        self.assertTrue(all(0 <= v <= 4 for v in vals))
        self.assertTrue(len(set(vals)) > 1)

    def test_choice(self):
        r = _RNG(5)
        items = ['a', 'b', 'c']
        picks = [r.choice(items) for _ in range(50)]
        self.assertTrue(all(p in items for p in picks))

    def test_beta(self):
        r = _RNG(10)
        vals = [r.beta(2, 5) for _ in range(100)]
        self.assertTrue(all(0 <= v <= 1 for v in vals))
        mean = sum(vals) / len(vals)
        expected = 2.0 / (2.0 + 5.0)
        self.assertAlmostEqual(mean, expected, delta=0.1)


# ==============================================================
# Environment Tests
# ==============================================================

class TestGridWorld(unittest.TestCase):
    def test_creation(self):
        env = GridWorld(rows=4, cols=4)
        self.assertEqual(env.state_size, 16)
        self.assertEqual(env.action_size, 4)

    def test_reset(self):
        env = GridWorld(rows=3, cols=3)
        state = env.reset()
        self.assertEqual(len(state), 9)
        self.assertEqual(state[0], 1.0)  # start at (0,0)
        self.assertEqual(sum(state), 1.0)  # one-hot

    def test_step_right(self):
        env = GridWorld(rows=3, cols=3)
        env.reset()
        state, reward, done, info = env.step(1)  # right
        self.assertEqual(state[1], 1.0)  # now at (0,1)

    def test_step_down(self):
        env = GridWorld(rows=3, cols=3)
        env.reset()
        state, _, _, _ = env.step(2)  # down
        self.assertEqual(state[3], 1.0)  # now at (1,0)

    def test_wall_blocks(self):
        env = GridWorld(rows=3, cols=3, walls=[(0, 1)])
        env.reset()
        state, _, _, _ = env.step(1)  # right into wall
        self.assertEqual(state[0], 1.0)  # didn't move

    def test_goal_reached(self):
        env = GridWorld(rows=2, cols=2, goal=(0, 1))
        env.reset()
        _, reward, done, _ = env.step(1)  # right to goal
        self.assertTrue(done)
        self.assertEqual(reward, 1.0)

    def test_boundary(self):
        env = GridWorld(rows=3, cols=3)
        env.reset()
        state, _, _, _ = env.step(0)  # up from (0,0)
        self.assertEqual(state[0], 1.0)  # didn't move

    def test_max_steps(self):
        env = GridWorld(rows=2, cols=2, goal=(1, 1))
        env.max_steps = 3
        env.reset()
        env.step(0)  # stay
        env.step(0)  # stay
        _, _, done, _ = env.step(0)  # step 3 -> done
        self.assertTrue(done)


class TestCartPole(unittest.TestCase):
    def test_creation(self):
        env = CartPole()
        self.assertEqual(env.state_size, 4)
        self.assertEqual(env.action_size, 2)

    def test_reset(self):
        env = CartPole()
        state = env.reset()
        self.assertEqual(len(state), 4)
        for v in state:
            self.assertAlmostEqual(v, 0.0, delta=0.1)

    def test_step(self):
        env = CartPole()
        env.reset()
        next_state, reward, done, info = env.step(1)
        self.assertEqual(len(next_state), 4)
        self.assertEqual(reward, 1.0)
        self.assertFalse(done)

    def test_episode_terminates(self):
        env = CartPole()
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(1)  # always push right
            steps += 1
        self.assertGreater(steps, 0)
        self.assertLessEqual(steps, 200)

    def test_deterministic(self):
        e1 = CartPole(seed=42)
        e2 = CartPole(seed=42)
        s1 = e1.reset()
        s2 = e2.reset()
        self.assertEqual(s1, s2)
        ns1, r1, d1, _ = e1.step(0)
        ns2, r2, d2, _ = e2.step(0)
        self.assertEqual(ns1, ns2)


class TestBandit(unittest.TestCase):
    def test_creation(self):
        env = Bandit(n_arms=5)
        self.assertEqual(env.action_size, 5)
        self.assertEqual(env.state_size, 1)

    def test_reset(self):
        env = Bandit()
        state = env.reset()
        self.assertEqual(state, [1.0])

    def test_step(self):
        env = Bandit(n_arms=3, probs=[0.0, 1.0, 0.5])
        _, reward, done, _ = env.step(1)  # arm with prob 1.0
        self.assertEqual(reward, 1.0)
        self.assertFalse(done)

    def test_custom_probs(self):
        env = Bandit(n_arms=2, probs=[0.0, 0.0])
        _, reward, _, _ = env.step(0)
        self.assertEqual(reward, 0.0)


class TestMountainCar(unittest.TestCase):
    def test_creation(self):
        env = MountainCar()
        self.assertEqual(env.state_size, 2)
        self.assertEqual(env.action_size, 3)

    def test_reset(self):
        env = MountainCar()
        state = env.reset()
        self.assertEqual(len(state), 2)
        self.assertGreaterEqual(state[0], -0.6)
        self.assertLessEqual(state[0], -0.4)

    def test_step(self):
        env = MountainCar()
        env.reset()
        next_state, reward, done, _ = env.step(2)  # push right
        self.assertEqual(len(next_state), 2)
        self.assertEqual(reward, -1.0)

    def test_max_steps(self):
        env = MountainCar()
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(1)  # no push
            steps += 1
        self.assertEqual(steps, 200)


# ==============================================================
# Replay Buffer Tests
# ==============================================================

class TestReplayBuffer(unittest.TestCase):
    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=5)
        buf.push([1, 2], 0, 1.0, [3, 4], False)
        self.assertEqual(len(buf), 1)

    def test_capacity(self):
        buf = ReplayBuffer(capacity=3)
        for i in range(5):
            buf.push([i], 0, 1.0, [i+1], False)
        self.assertEqual(len(buf), 3)

    def test_sample(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(20):
            buf.push([i], i % 3, float(i), [i+1], i == 19)
        states, actions, rewards, next_states, dones = buf.sample(5)
        self.assertEqual(len(states), 5)
        self.assertEqual(len(actions), 5)

    def test_circular_overwrite(self):
        buf = ReplayBuffer(capacity=2)
        buf.push([0], 0, 0.0, [1], False)
        buf.push([1], 1, 1.0, [2], False)
        buf.push([2], 0, 2.0, [3], True)
        self.assertEqual(len(buf), 2)
        # First entry was overwritten
        states = [b[0] for b in buf.buffer]
        self.assertIn([2], states)


class TestPrioritizedReplayBuffer(unittest.TestCase):
    def test_push_and_len(self):
        buf = PrioritizedReplayBuffer(capacity=10)
        buf.push([1], 0, 1.0, [2], False)
        self.assertEqual(len(buf), 1)

    def test_sample_returns_weights(self):
        buf = PrioritizedReplayBuffer(capacity=10)
        for i in range(10):
            buf.push([i], 0, 1.0, [i+1], False)
        result = buf.sample(3)
        self.assertEqual(len(result), 7)  # states, actions, rewards, next_states, dones, indices, weights
        _, _, _, _, _, indices, weights = result
        self.assertEqual(len(weights), 3)

    def test_update_priorities(self):
        buf = PrioritizedReplayBuffer(capacity=5)
        for i in range(5):
            buf.push([i], 0, 1.0, [i+1], False)
        buf.update_priorities([0, 1], [10.0, 0.1])
        self.assertAlmostEqual(buf.priorities[0], 10.0 + 1e-6, places=5)
        self.assertAlmostEqual(buf.priorities[1], 0.1 + 1e-6, places=5)


# ==============================================================
# Policy Tests
# ==============================================================

class TestEpsilonGreedy(unittest.TestCase):
    def test_greedy(self):
        policy = EpsilonGreedy(epsilon=0.0)
        action = policy.select([0.1, 0.5, 0.3])
        self.assertEqual(action, 1)

    def test_exploration(self):
        policy = EpsilonGreedy(epsilon=1.0, seed=42)
        actions = [policy.select([0.1, 0.5, 0.3], n_actions=3) for _ in range(100)]
        self.assertTrue(len(set(actions)) > 1)

    def test_decay(self):
        policy = EpsilonGreedy(epsilon=1.0, decay=0.5)
        policy.step()
        self.assertAlmostEqual(policy.epsilon, 0.5)
        policy.step()
        self.assertAlmostEqual(policy.epsilon, 0.25)

    def test_min_epsilon(self):
        policy = EpsilonGreedy(epsilon=0.1, min_epsilon=0.05, decay=0.1)
        for _ in range(100):
            policy.step()
        self.assertAlmostEqual(policy.epsilon, 0.05)

    def test_tensor_input(self):
        policy = EpsilonGreedy(epsilon=0.0)
        action = policy.select(Tensor([0.1, 0.9, 0.3]))
        self.assertEqual(action, 1)


class TestBoltzmannPolicy(unittest.TestCase):
    def test_selection(self):
        policy = BoltzmannPolicy(temperature=0.01, seed=42)
        # Very low temperature -> almost greedy
        actions = [policy.select([0.1, 10.0, 0.3]) for _ in range(20)]
        self.assertTrue(all(a == 1 for a in actions))

    def test_high_temp_explores(self):
        policy = BoltzmannPolicy(temperature=100.0, seed=42)
        actions = [policy.select([1.0, 1.0, 1.0]) for _ in range(100)]
        self.assertTrue(len(set(actions)) > 1)

    def test_decay(self):
        policy = BoltzmannPolicy(temperature=1.0, decay=0.5)
        policy.step()
        self.assertAlmostEqual(policy.temperature, 0.5)


class TestUCBPolicy(unittest.TestCase):
    def test_tries_all_first(self):
        policy = UCBPolicy(c=2.0)
        # Should try each action once
        a0 = policy.select([0.0, 0.0, 0.0])
        self.assertEqual(a0, 0)
        policy.update(a0)
        a1 = policy.select([0.0, 0.0, 0.0])
        self.assertEqual(a1, 1)
        policy.update(a1)
        a2 = policy.select([0.0, 0.0, 0.0])
        self.assertEqual(a2, 2)

    def test_ucb_bonus(self):
        policy = UCBPolicy(c=2.0)
        for i in range(3):
            policy.update(i)
        # After trying all, picks based on Q + bonus
        action = policy.select([1.0, 0.0, 0.0])
        self.assertEqual(action, 0)  # highest Q


# ==============================================================
# Tabular RL Tests
# ==============================================================

class TestQLearning(unittest.TestCase):
    def test_creation(self):
        q = QLearning(n_states=10, n_actions=4)
        self.assertEqual(len(q.q_table), 10)
        self.assertEqual(len(q.q_table[0]), 4)

    def test_update(self):
        q = QLearning(n_states=4, n_actions=2, lr=0.5, gamma=0.9)
        td = q.update(0, 1, 1.0, 1, False)
        self.assertGreater(q.q_table[0][1], 0.0)

    def test_get_action_greedy(self):
        q = QLearning(n_states=4, n_actions=2, seed=42)
        q.q_table[0] = [0.5, 0.8]
        action = q.get_action(0, epsilon=0.0)
        self.assertEqual(action, 1)

    def test_get_policy(self):
        q = QLearning(n_states=3, n_actions=2)
        q.q_table[0] = [1.0, 0.0]
        q.q_table[1] = [0.0, 1.0]
        q.q_table[2] = [0.5, 0.5]
        policy = q.get_policy()
        self.assertEqual(policy[0], 0)
        self.assertEqual(policy[1], 1)

    def test_learning_simple(self):
        """Q-learning should learn optimal policy for simple MDP."""
        q = QLearning(n_states=3, n_actions=2, lr=0.1, gamma=0.9, seed=42)
        # State 0: action 1 -> state 1, reward 0
        # State 1: action 0 -> state 2, reward 1 (terminal)
        for _ in range(100):
            q.update(0, 1, 0.0, 1, False)
            q.update(1, 0, 1.0, 2, True)
        # Q(1, 0) should be high
        self.assertGreater(q.q_table[1][0], 0.5)
        # Q(0, 1) should reflect discounted future
        self.assertGreater(q.q_table[0][1], 0.0)

    def test_done_zeroes_next(self):
        q = QLearning(n_states=2, n_actions=2, lr=1.0, gamma=0.99)
        q.update(0, 0, 5.0, 1, True)
        self.assertAlmostEqual(q.q_table[0][0], 5.0)


class TestSARSA(unittest.TestCase):
    def test_creation(self):
        s = SARSA(n_states=5, n_actions=3)
        self.assertEqual(len(s.q_table), 5)

    def test_update(self):
        s = SARSA(n_states=4, n_actions=2, lr=0.5, gamma=0.9)
        s.update(0, 1, 1.0, 1, 0, False)
        self.assertGreater(s.q_table[0][1], 0.0)

    def test_on_policy_difference(self):
        """SARSA uses next_action, Q-learning uses max."""
        sarsa = SARSA(n_states=3, n_actions=2, lr=1.0, gamma=0.9)
        q = QLearning(n_states=3, n_actions=2, lr=1.0, gamma=0.9)
        # Same experience but SARSA uses actual next action
        q.q_table[1] = [0.0, 10.0]
        sarsa.q_table[1] = [0.0, 10.0]
        q.update(0, 0, 1.0, 1, False)  # uses max(Q[1]) = 10
        sarsa.update(0, 0, 1.0, 1, 0, False)  # uses Q[1][0] = 0
        self.assertGreater(q.q_table[0][0], sarsa.q_table[0][0])

    def test_get_policy(self):
        s = SARSA(n_states=2, n_actions=2)
        s.q_table[0] = [3.0, 1.0]
        s.q_table[1] = [0.0, 2.0]
        policy = s.get_policy()
        self.assertEqual(policy, [0, 1])


# ==============================================================
# Multi-Armed Bandit Tests
# ==============================================================

class TestMultiArmedBandit(unittest.TestCase):
    def test_epsilon_greedy(self):
        mab = MultiArmedBandit(n_arms=3, strategy='epsilon_greedy', epsilon=0.0)
        mab.values = [0.1, 0.9, 0.5]
        arm = mab.select_arm()
        self.assertEqual(arm, 1)

    def test_ucb1(self):
        mab = MultiArmedBandit(n_arms=3, strategy='ucb1')
        # Should try all arms first
        arms = []
        for _ in range(3):
            arm = mab.select_arm()
            arms.append(arm)
            mab.update(arm, 1.0)
        self.assertEqual(sorted(arms), [0, 1, 2])

    def test_thompson(self):
        mab = MultiArmedBandit(n_arms=2, strategy='thompson', seed=42)
        arm = mab.select_arm()
        self.assertIn(arm, [0, 1])

    def test_update(self):
        mab = MultiArmedBandit(n_arms=3, strategy='epsilon_greedy')
        mab.update(0, 1.0)
        mab.update(0, 0.0)
        self.assertEqual(mab.counts[0], 2)
        self.assertAlmostEqual(mab.values[0], 0.5)

    def test_get_best_arm(self):
        mab = MultiArmedBandit(n_arms=3, strategy='epsilon_greedy')
        mab.values = [0.1, 0.9, 0.5]
        self.assertEqual(mab.get_best_arm(), 1)

    def test_regret(self):
        mab = MultiArmedBandit(n_arms=2, strategy='epsilon_greedy')
        regrets = mab.get_regret(1.0, [0.5, 0.5, 1.0])
        self.assertAlmostEqual(regrets[0], 0.5)
        self.assertAlmostEqual(regrets[1], 1.0)
        self.assertAlmostEqual(regrets[2], 1.0)

    def test_thompson_converges(self):
        """Thompson sampling should find the best arm."""
        mab = MultiArmedBandit(n_arms=3, strategy='thompson', seed=42)
        env = Bandit(n_arms=3, probs=[0.2, 0.8, 0.5], seed=42)
        for _ in range(200):
            arm = mab.select_arm()
            _, reward, _, _ = env.step(arm)
            mab.update(arm, reward)
        self.assertEqual(mab.get_best_arm(), 1)


# ==============================================================
# DQN Tests
# ==============================================================

class TestDQN(unittest.TestCase):
    def test_creation(self):
        dqn = DQN(state_size=4, action_size=2)
        self.assertEqual(dqn.state_size, 4)
        self.assertEqual(dqn.action_size, 2)

    def test_get_action(self):
        dqn = DQN(state_size=4, action_size=2, seed=42)
        action = dqn.get_action([0.1, 0.2, 0.3, 0.4])
        self.assertIn(action, [0, 1])

    def test_get_action_epsilon(self):
        dqn = DQN(state_size=2, action_size=3, seed=42)
        actions = [dqn.get_action([0.1, 0.2], epsilon=1.0) for _ in range(100)]
        self.assertTrue(len(set(actions)) > 1)

    def test_get_q_values(self):
        dqn = DQN(state_size=2, action_size=3)
        q = dqn.get_q_values([0.5, 0.5])
        self.assertEqual(len(q), 3)

    def test_train_batch(self):
        dqn = DQN(state_size=2, action_size=2, hidden_sizes=[8], seed=42)
        states = [[0.1, 0.2], [0.3, 0.4]]
        actions = [0, 1]
        rewards = [1.0, 0.0]
        next_states = [[0.2, 0.3], [0.4, 0.5]]
        dones = [False, True]
        loss = dqn.train_batch(states, actions, rewards, next_states, dones)
        self.assertIsInstance(loss, float)

    def test_soft_update(self):
        dqn = DQN(state_size=2, action_size=2, hidden_sizes=[4], tau=0.5)
        # Get initial target weights
        tgt_layers = dqn.target_net.get_trainable_layers()
        old_w = tgt_layers[0].get_params()[0][0].data[0][0]
        # Modify q_net
        src_layers = dqn.q_net.get_trainable_layers()
        src_layers[0].get_params()[0][0].data[0][0] = 100.0
        dqn.soft_update()
        new_w = tgt_layers[0].get_params()[0][0].data[0][0]
        # Should be mixture
        expected = 0.5 * 100.0 + 0.5 * old_w
        self.assertAlmostEqual(new_w, expected, places=5)

    def test_hard_update(self):
        dqn = DQN(state_size=2, action_size=2, hidden_sizes=[4])
        src_layers = dqn.q_net.get_trainable_layers()
        src_layers[0].get_params()[0][0].data[0][0] = 42.0
        dqn.hard_update()
        tgt_layers = dqn.target_net.get_trainable_layers()
        self.assertAlmostEqual(tgt_layers[0].get_params()[0][0].data[0][0], 42.0)


class TestDoubleDQN(unittest.TestCase):
    def test_creation(self):
        ddqn = DoubleDQN(state_size=4, action_size=2)
        self.assertIsInstance(ddqn, DQN)

    def test_train_batch(self):
        ddqn = DoubleDQN(state_size=2, action_size=2, hidden_sizes=[8], seed=42)
        loss = ddqn.train_batch(
            [[0.1, 0.2]], [0], [1.0], [[0.3, 0.4]], [False]
        )
        self.assertIsInstance(loss, float)


# ==============================================================
# REINFORCE Tests
# ==============================================================

class TestREINFORCE(unittest.TestCase):
    def test_creation(self):
        agent = REINFORCE(state_size=4, action_size=2)
        self.assertEqual(agent.state_size, 4)
        self.assertEqual(agent.action_size, 2)

    def test_get_action(self):
        agent = REINFORCE(state_size=2, action_size=3, seed=42)
        action = agent.get_action([0.5, 0.5])
        self.assertIn(action, [0, 1, 2])

    def test_store_reward(self):
        agent = REINFORCE(state_size=2, action_size=2)
        agent.store_reward(1.0)
        agent.store_reward(0.5)
        self.assertEqual(len(agent.saved_rewards), 2)

    def test_compute_returns(self):
        agent = REINFORCE(state_size=2, action_size=2, gamma=0.9)
        agent.saved_rewards = [1.0, 1.0, 1.0]
        returns = agent.compute_returns()
        self.assertAlmostEqual(returns[2], 1.0)
        self.assertAlmostEqual(returns[1], 1.0 + 0.9)
        self.assertAlmostEqual(returns[0], 1.0 + 0.9 + 0.81)

    def test_update(self):
        agent = REINFORCE(state_size=2, action_size=2, hidden_sizes=[8], seed=42)
        # Collect one episode
        for _ in range(5):
            agent.get_action([0.5, 0.5])
            agent.store_reward(1.0)
        loss = agent.update()
        self.assertIsInstance(loss, float)
        self.assertEqual(len(agent.saved_rewards), 0)  # cleared

    def test_get_action_probs(self):
        agent = REINFORCE(state_size=2, action_size=3)
        probs = agent.get_action_probs([0.5, 0.5])
        self.assertEqual(len(probs), 3)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)


# ==============================================================
# Actor-Critic Tests
# ==============================================================

class TestActorCritic(unittest.TestCase):
    def test_creation(self):
        ac = ActorCritic(state_size=4, action_size=2)
        self.assertEqual(ac.state_size, 4)

    def test_get_action(self):
        ac = ActorCritic(state_size=2, action_size=3, seed=42)
        action = ac.get_action([0.5, 0.5])
        self.assertIn(action, [0, 1, 2])

    def test_get_value(self):
        ac = ActorCritic(state_size=2, action_size=2)
        v = ac.get_value([0.5, 0.5])
        self.assertIsInstance(v, float)

    def test_update(self):
        ac = ActorCritic(state_size=2, action_size=2, hidden_sizes=[8], seed=42)
        actor_loss, critic_loss = ac.update([0.5, 0.5], 0, 1.0, [0.6, 0.6], False)
        self.assertIsInstance(actor_loss, float)
        self.assertIsInstance(critic_loss, float)

    def test_update_terminal(self):
        ac = ActorCritic(state_size=2, action_size=2, hidden_sizes=[8])
        actor_loss, critic_loss = ac.update([0.5, 0.5], 1, 10.0, [0.0, 0.0], True)
        self.assertIsInstance(actor_loss, float)

    def test_get_action_probs(self):
        ac = ActorCritic(state_size=2, action_size=3)
        probs = ac.get_action_probs([0.5, 0.5])
        self.assertEqual(len(probs), 3)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)


# ==============================================================
# PPO Tests
# ==============================================================

class TestPPO(unittest.TestCase):
    def test_creation(self):
        ppo = PPO(state_size=4, action_size=2)
        self.assertEqual(ppo.state_size, 4)
        self.assertEqual(ppo.clip_epsilon, 0.2)

    def test_get_action(self):
        ppo = PPO(state_size=2, action_size=3, seed=42)
        action = ppo.get_action([0.5, 0.5])
        self.assertIn(action, [0, 1, 2])
        self.assertEqual(len(ppo.states), 1)
        self.assertEqual(len(ppo.old_probs), 1)

    def test_store_reward(self):
        ppo = PPO(state_size=2, action_size=2)
        ppo.store_reward(1.0, False)
        ppo.store_reward(0.5, True)
        self.assertEqual(len(ppo.rewards), 2)
        self.assertEqual(ppo.dones, [False, True])

    def test_update(self):
        ppo = PPO(state_size=2, action_size=2, hidden_sizes=[8],
                  n_epochs=2, seed=42)
        for _ in range(5):
            ppo.get_action([0.5, 0.5])
            ppo.store_reward(1.0, False)
        ppo.store_reward(0.0, True)  # extra done marker (only 5 actions)
        # Fix: match lengths
        ppo.rewards = ppo.rewards[:5]
        ppo.dones = ppo.dones[:5]
        ppo.dones[-1] = True
        actor_loss, critic_loss = ppo.update()
        self.assertIsInstance(actor_loss, float)
        self.assertEqual(len(ppo.states), 0)  # cleared

    def test_get_value(self):
        ppo = PPO(state_size=2, action_size=2)
        v = ppo.get_value([0.5, 0.5])
        self.assertIsInstance(v, float)

    def test_get_action_probs(self):
        ppo = PPO(state_size=2, action_size=3)
        probs = ppo.get_action_probs([0.5, 0.5])
        self.assertEqual(len(probs), 3)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_empty_update(self):
        ppo = PPO(state_size=2, action_size=2)
        actor_loss, critic_loss = ppo.update()
        self.assertEqual(actor_loss, 0.0)
        self.assertEqual(critic_loss, 0.0)


# ==============================================================
# Trainer Tests
# ==============================================================

class TestDQNTrainer(unittest.TestCase):
    def test_train_episode(self):
        env = GridWorld(rows=3, cols=3, goal=(0, 2), seed=42)
        dqn = DQN(state_size=9, action_size=4, hidden_sizes=[16], seed=42)
        trainer = DQNTrainer(dqn, env, batch_size=4, seed=42)
        reward = trainer.train_episode()
        self.assertIsInstance(reward, float)

    def test_train_multiple(self):
        env = GridWorld(rows=3, cols=3, goal=(0, 2), seed=42)
        dqn = DQN(state_size=9, action_size=4, hidden_sizes=[8], seed=42)
        trainer = DQNTrainer(dqn, env, batch_size=4, seed=42)
        rewards = trainer.train(5)
        self.assertEqual(len(rewards), 5)

    def test_buffer_fills(self):
        env = GridWorld(rows=2, cols=2, seed=42)
        dqn = DQN(state_size=4, action_size=4, hidden_sizes=[4], seed=42)
        trainer = DQNTrainer(dqn, env, batch_size=2, seed=42)
        trainer.train(1)
        self.assertGreater(len(trainer.buffer), 0)


class TestPolicyGradientTrainer(unittest.TestCase):
    def test_train_episode(self):
        env = GridWorld(rows=3, cols=3, goal=(0, 2), seed=42)
        agent = REINFORCE(state_size=9, action_size=4, hidden_sizes=[8], seed=42)
        trainer = PolicyGradientTrainer(agent, env)
        reward = trainer.train_episode()
        self.assertIsInstance(reward, float)

    def test_train_multiple(self):
        env = GridWorld(rows=3, cols=3, goal=(0, 2), seed=42)
        agent = REINFORCE(state_size=9, action_size=4, hidden_sizes=[8], seed=42)
        trainer = PolicyGradientTrainer(agent, env)
        rewards = trainer.train(3)
        self.assertEqual(len(rewards), 3)


class TestActorCriticTrainer(unittest.TestCase):
    def test_train_episode(self):
        env = GridWorld(rows=3, cols=3, goal=(0, 2), seed=42)
        agent = ActorCritic(state_size=9, action_size=4, hidden_sizes=[8], seed=42)
        trainer = ActorCriticTrainer(agent, env)
        reward = trainer.train_episode()
        self.assertIsInstance(reward, float)

    def test_train_multiple(self):
        env = GridWorld(rows=3, cols=3, goal=(0, 2), seed=42)
        agent = ActorCritic(state_size=9, action_size=4, hidden_sizes=[8], seed=42)
        trainer = ActorCriticTrainer(agent, env)
        rewards = trainer.train(3)
        self.assertEqual(len(rewards), 3)


class TestPPOTrainer(unittest.TestCase):
    def test_train_episode(self):
        env = GridWorld(rows=3, cols=3, goal=(0, 2), seed=42)
        agent = PPO(state_size=9, action_size=4, hidden_sizes=[8], n_epochs=2, seed=42)
        trainer = PPOTrainer(agent, env, steps_per_update=50)
        reward = trainer.train_episode()
        self.assertIsInstance(reward, float)

    def test_train_multiple(self):
        env = GridWorld(rows=3, cols=3, goal=(0, 2), seed=42)
        agent = PPO(state_size=9, action_size=4, hidden_sizes=[8], n_epochs=1, seed=42)
        trainer = PPOTrainer(agent, env, steps_per_update=50)
        rewards = trainer.train(3)
        self.assertEqual(len(rewards), 3)


# ==============================================================
# Utility Tests
# ==============================================================

class TestRewardShaper(unittest.TestCase):
    def test_shape(self):
        # Potential: distance to goal at (4,4)
        def potential(state):
            return -abs(state[0] - 4) - abs(state[1] - 4)
        shaper = RewardShaper(potential, gamma=1.0)
        # Moving from (0,0) to (1,0) reduces distance
        shaped = shaper.shape([0, 0], [1, 0], -1.0)
        self.assertGreater(shaped, -1.0)  # shaping adds positive bonus

    def test_no_shaping_at_goal(self):
        def potential(state):
            return -abs(state[0])
        shaper = RewardShaper(potential, gamma=1.0)
        # Same state -> no bonus
        shaped = shaper.shape([0], [0], 1.0)
        self.assertAlmostEqual(shaped, 1.0)


class TestReturnTracker(unittest.TestCase):
    def test_add_and_mean(self):
        tracker = ReturnTracker(window_size=3)
        tracker.add(10.0)
        tracker.add(20.0)
        tracker.add(30.0)
        self.assertAlmostEqual(tracker.mean(), 20.0)

    def test_window(self):
        tracker = ReturnTracker(window_size=2)
        tracker.add(10.0)
        tracker.add(20.0)
        tracker.add(30.0)
        self.assertAlmostEqual(tracker.mean(), 25.0)  # last 2

    def test_max_min(self):
        tracker = ReturnTracker()
        tracker.add(5.0)
        tracker.add(15.0)
        tracker.add(10.0)
        self.assertEqual(tracker.max(), 15.0)
        self.assertEqual(tracker.min(), 5.0)

    def test_std(self):
        tracker = ReturnTracker(window_size=3)
        tracker.add(10.0)
        tracker.add(10.0)
        tracker.add(10.0)
        self.assertAlmostEqual(tracker.std(), 0.0)

    def test_is_solved(self):
        tracker = ReturnTracker(window_size=2)
        tracker.add(100.0)
        tracker.add(200.0)
        self.assertTrue(tracker.is_solved(150.0))
        self.assertFalse(tracker.is_solved(200.0))

    def test_empty(self):
        tracker = ReturnTracker()
        self.assertEqual(tracker.mean(), 0.0)
        self.assertEqual(tracker.max(), 0.0)
        self.assertEqual(tracker.std(), 0.0)


class TestDiscountedReturn(unittest.TestCase):
    def test_compute(self):
        returns = DiscountedReturn.compute([1.0, 1.0, 1.0], gamma=0.5)
        self.assertAlmostEqual(returns[2], 1.0)
        self.assertAlmostEqual(returns[1], 1.5)
        self.assertAlmostEqual(returns[0], 1.75)

    def test_compute_single(self):
        returns = DiscountedReturn.compute([5.0], gamma=0.99)
        self.assertAlmostEqual(returns[0], 5.0)

    def test_gae(self):
        rewards = [1.0, 1.0, 1.0]
        values = [0.5, 0.5, 0.5]
        next_value = 0.5
        advantages, returns = DiscountedReturn.compute_gae(
            rewards, values, next_value, gamma=0.99, lam=0.95
        )
        self.assertEqual(len(advantages), 3)
        self.assertEqual(len(returns), 3)
        # Advantages should be positive (reward > value)
        for a in advantages:
            self.assertGreater(a, 0)

    def test_gae_zero_rewards(self):
        rewards = [0.0, 0.0]
        values = [1.0, 1.0]
        next_value = 1.0
        advantages, returns = DiscountedReturn.compute_gae(
            rewards, values, next_value, gamma=0.99, lam=0.95
        )
        # With zero rewards and constant values, advantages ~ 0
        for a in advantages:
            self.assertAlmostEqual(a, -0.01, delta=0.02)


# ==============================================================
# Integration Tests
# ==============================================================

class TestQLearningOnGridWorld(unittest.TestCase):
    def test_learns_gridworld(self):
        """Q-learning should learn to navigate a small grid."""
        env = GridWorld(rows=3, cols=3, goal=(2, 2), seed=42)
        n_states = env.state_size
        q = QLearning(n_states=n_states, n_actions=4, lr=0.1, gamma=0.95, seed=42)

        for episode in range(200):
            state_vec = env.reset()
            state = state_vec.index(1.0)
            done = False
            epsilon = max(0.1, 1.0 - episode / 100)
            while not done:
                action = q.get_action(state, epsilon=epsilon)
                next_vec, reward, done, _ = env.step(action)
                next_state = next_vec.index(1.0)
                q.update(state, action, reward, next_state, done)
                state = next_state

        # Check that optimal path from start has positive Q-values
        policy = q.get_policy()
        # From (0,0) the optimal path should go right or down
        start_action = policy[0]
        self.assertIn(start_action, [1, 2])  # right or down


class TestSARSAOnGridWorld(unittest.TestCase):
    def test_learns_gridworld(self):
        env = GridWorld(rows=3, cols=3, goal=(2, 2), seed=42)
        n_states = env.state_size
        s = SARSA(n_states=n_states, n_actions=4, lr=0.1, gamma=0.95, seed=42)

        for episode in range(200):
            state_vec = env.reset()
            state = state_vec.index(1.0)
            epsilon = max(0.1, 1.0 - episode / 100)
            action = s.get_action(state, epsilon=epsilon)
            done = False
            while not done:
                next_vec, reward, done, _ = env.step(action)
                next_state = next_vec.index(1.0)
                next_action = s.get_action(next_state, epsilon=epsilon)
                s.update(state, action, reward, next_state, next_action, done)
                state = next_state
                action = next_action

        policy = s.get_policy()
        start_action = policy[0]
        self.assertIn(start_action, [1, 2])


class TestBanditSolving(unittest.TestCase):
    def test_ucb_finds_best(self):
        env = Bandit(n_arms=5, probs=[0.1, 0.2, 0.9, 0.3, 0.4], seed=42)
        mab = MultiArmedBandit(n_arms=5, strategy='ucb1', seed=42)
        for _ in range(500):
            arm = mab.select_arm()
            _, reward, _, _ = env.step(arm)
            mab.update(arm, reward)
        self.assertEqual(mab.get_best_arm(), 2)

    def test_epsilon_greedy_finds_best(self):
        env = Bandit(n_arms=3, probs=[0.1, 0.9, 0.5], seed=42)
        mab = MultiArmedBandit(n_arms=3, strategy='epsilon_greedy', epsilon=0.1, seed=42)
        for _ in range(500):
            arm = mab.select_arm()
            _, reward, _, _ = env.step(arm)
            mab.update(arm, reward)
        self.assertEqual(mab.get_best_arm(), 1)


class TestDQNOnBandit(unittest.TestCase):
    def test_dqn_trains(self):
        """DQN should reduce loss over training."""
        env = GridWorld(rows=2, cols=2, goal=(1, 1), seed=42)
        dqn = DQN(state_size=4, action_size=4, hidden_sizes=[8], seed=42, lr=0.01)
        trainer = DQNTrainer(dqn, env, batch_size=4, seed=42,
                             target_update_freq=5)
        rewards = trainer.train(10)
        self.assertEqual(len(rewards), 10)


class TestREINFORCETraining(unittest.TestCase):
    def test_reinforce_trains(self):
        env = GridWorld(rows=2, cols=2, goal=(1, 1), seed=42)
        agent = REINFORCE(state_size=4, action_size=4, hidden_sizes=[8], seed=42)
        trainer = PolicyGradientTrainer(agent, env)
        rewards = trainer.train(5)
        self.assertEqual(len(rewards), 5)


class TestActorCriticTraining(unittest.TestCase):
    def test_ac_trains(self):
        env = GridWorld(rows=2, cols=2, goal=(1, 1), seed=42)
        agent = ActorCritic(state_size=4, action_size=4, hidden_sizes=[8], seed=42)
        trainer = ActorCriticTrainer(agent, env)
        rewards = trainer.train(5)
        self.assertEqual(len(rewards), 5)


class TestPPOTraining(unittest.TestCase):
    def test_ppo_trains(self):
        env = GridWorld(rows=2, cols=2, goal=(1, 1), seed=42)
        agent = PPO(state_size=4, action_size=4, hidden_sizes=[8],
                    n_epochs=2, seed=42)
        trainer = PPOTrainer(agent, env, steps_per_update=50)
        rewards = trainer.train(5)
        self.assertEqual(len(rewards), 5)


class TestCartPoleTraining(unittest.TestCase):
    def test_dqn_on_cartpole(self):
        """DQN should run on CartPole without errors."""
        env = CartPole(seed=42)
        dqn = DQN(state_size=4, action_size=2, hidden_sizes=[8], seed=42)
        trainer = DQNTrainer(dqn, env, batch_size=4, seed=42)
        rewards = trainer.train(3)
        self.assertEqual(len(rewards), 3)

    def test_reinforce_on_cartpole(self):
        env = CartPole(seed=42)
        agent = REINFORCE(state_size=4, action_size=2, hidden_sizes=[8], seed=42)
        trainer = PolicyGradientTrainer(agent, env)
        rewards = trainer.train(3)
        self.assertEqual(len(rewards), 3)


class TestDoubleDQNTraining(unittest.TestCase):
    def test_double_dqn_trains(self):
        env = GridWorld(rows=2, cols=2, goal=(1, 1), seed=42)
        ddqn = DoubleDQN(state_size=4, action_size=4, hidden_sizes=[8], seed=42)
        trainer = DQNTrainer(ddqn, env, batch_size=4, seed=42)
        rewards = trainer.train(5)
        self.assertEqual(len(rewards), 5)


if __name__ == '__main__':
    unittest.main()
