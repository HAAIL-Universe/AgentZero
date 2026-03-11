"""
C191: Reinforcement Learning
==============================
From-scratch RL algorithms using only NumPy.

Components:
- Environment: Base environment interface + GridWorld, CliffWalking, FrozenLake
- EpsilonGreedy: Epsilon-greedy exploration strategy
- UCB: Upper Confidence Bound exploration
- MultiArmedBandit: N-armed bandit with multiple strategies
- QLearning: Off-policy TD control
- SARSA: On-policy TD control
- ExpectedSARSA: Expected SARSA (averages over next actions)
- NStepSARSA: N-step bootstrapping SARSA
- MonteCarloControl: First-visit MC with exploring starts
- DynaQ: Model-based RL (Q-learning + planning with learned model)
- PolicyGradient: REINFORCE with baseline
- ActorCritic: One-step actor-critic (TD(0))
- TDLambda: TD(lambda) with eligibility traces
- DoubleQLearning: Reduces maximization bias with two Q-tables
- PrioritizedSweeping: Priority queue model-based planning
"""

import numpy as np
from collections import defaultdict, deque
import heapq


# ============================================================
# Environments
# ============================================================
class Environment:
    """Base environment interface."""

    def __init__(self):
        self.n_states = 0
        self.n_actions = 0
        self.state = None

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        """Returns (next_state, reward, done, info)."""
        raise NotImplementedError

    def get_actions(self, state=None):
        """Return list of valid actions."""
        return list(range(self.n_actions))


class GridWorld(Environment):
    """Simple grid world: agent navigates to goal, walls block movement."""

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

    def __init__(self, rows=5, cols=5, start=(0, 0), goal=None, walls=None,
                 step_reward=-1.0, goal_reward=10.0, wall_penalty=-1.0):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal if goal is not None else (rows - 1, cols - 1)
        self.walls = set(walls) if walls else set()
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.wall_penalty = wall_penalty
        self.n_states = rows * cols
        self.n_actions = 4
        self.state = None

    def _to_idx(self, pos):
        return pos[0] * self.cols + pos[1]

    def _to_pos(self, idx):
        return (idx // self.cols, idx % self.cols)

    def reset(self):
        self.state = self._to_idx(self.start)
        return self.state

    def step(self, action):
        r, c = self._to_pos(self.state)
        if action == self.UP:
            nr, nc = r - 1, c
        elif action == self.DOWN:
            nr, nc = r + 1, c
        elif action == self.LEFT:
            nr, nc = r, c - 1
        elif action == self.RIGHT:
            nr, nc = r, c + 1
        else:
            nr, nc = r, c

        # Boundary check
        if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
            nr, nc = r, c

        # Wall check
        if (nr, nc) in self.walls:
            nr, nc = r, c

        self.state = self._to_idx((nr, nc))
        goal_idx = self._to_idx(self.goal)
        done = (self.state == goal_idx)
        reward = self.goal_reward if done else self.step_reward
        return self.state, reward, done, {}


class CliffWalking(Environment):
    """Cliff Walking: 4x12 grid, cliff along bottom row, big penalty for falling."""

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

    def __init__(self):
        super().__init__()
        self.rows = 4
        self.cols = 12
        self.n_states = self.rows * self.cols
        self.n_actions = 4
        self.start = (3, 0)
        self.goal = (3, 11)
        self.cliff = set((3, c) for c in range(1, 11))
        self.state = None

    def _to_idx(self, pos):
        return pos[0] * self.cols + pos[1]

    def _to_pos(self, idx):
        return (idx // self.cols, idx % self.cols)

    def reset(self):
        self.state = self._to_idx(self.start)
        return self.state

    def step(self, action):
        r, c = self._to_pos(self.state)
        if action == self.UP:
            nr, nc = r - 1, c
        elif action == self.DOWN:
            nr, nc = r + 1, c
        elif action == self.LEFT:
            nr, nc = r, c - 1
        elif action == self.RIGHT:
            nr, nc = r, c + 1
        else:
            nr, nc = r, c

        # Boundary
        nr = max(0, min(self.rows - 1, nr))
        nc = max(0, min(self.cols - 1, nc))

        if (nr, nc) in self.cliff:
            # Fell off cliff -- back to start with big penalty
            self.state = self._to_idx(self.start)
            return self.state, -100.0, False, {'cliff': True}

        self.state = self._to_idx((nr, nc))
        done = (nr, nc) == self.goal
        return self.state, -1.0, done, {}


class FrozenLake(Environment):
    """Frozen Lake: slippery grid with holes. Stochastic transitions."""

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

    def __init__(self, map_str=None, slip_prob=0.1):
        super().__init__()
        if map_str is None:
            map_str = [
                "SFFF",
                "FHFH",
                "FFFH",
                "HFFG"
            ]
        self.map = [list(row) for row in map_str]
        self.rows = len(self.map)
        self.cols = len(self.map[0])
        self.n_states = self.rows * self.cols
        self.n_actions = 4
        self.slip_prob = slip_prob
        self._rng = np.random.RandomState(42)

        # Find start and goal
        self.start_pos = None
        self.goal_pos = None
        self.holes = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if self.map[r][c] == 'S':
                    self.start_pos = (r, c)
                elif self.map[r][c] == 'G':
                    self.goal_pos = (r, c)
                elif self.map[r][c] == 'H':
                    self.holes.add((r, c))
        self.state = None

    def seed(self, s):
        self._rng = np.random.RandomState(s)

    def _to_idx(self, pos):
        return pos[0] * self.cols + pos[1]

    def _to_pos(self, idx):
        return (idx // self.cols, idx % self.cols)

    def reset(self):
        self.state = self._to_idx(self.start_pos)
        return self.state

    def _move(self, pos, action):
        r, c = pos
        if action == self.UP:
            nr, nc = r - 1, c
        elif action == self.DOWN:
            nr, nc = r + 1, c
        elif action == self.LEFT:
            nr, nc = r, c - 1
        elif action == self.RIGHT:
            nr, nc = r, c + 1
        else:
            nr, nc = r, c
        nr = max(0, min(self.rows - 1, nr))
        nc = max(0, min(self.cols - 1, nc))
        return (nr, nc)

    def step(self, action):
        pos = self._to_pos(self.state)
        # Slippery: sometimes take random action
        if self._rng.random() < self.slip_prob:
            action = self._rng.randint(0, self.n_actions)

        new_pos = self._move(pos, action)
        self.state = self._to_idx(new_pos)

        if new_pos in self.holes:
            return self.state, -1.0, True, {'hole': True}
        if new_pos == self.goal_pos:
            return self.state, 1.0, True, {'goal': True}
        return self.state, -0.01, False, {}


# ============================================================
# Exploration Strategies
# ============================================================
class EpsilonGreedy:
    """Epsilon-greedy action selection."""

    def __init__(self, epsilon=0.1, decay=1.0, min_epsilon=0.01, seed=None):
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self._rng = np.random.RandomState(seed)

    def select(self, q_values):
        """Select action given Q-values array."""
        if self._rng.random() < self.epsilon:
            return self._rng.randint(len(q_values))
        # Break ties randomly
        max_q = np.max(q_values)
        best = np.where(q_values == max_q)[0]
        return self._rng.choice(best)

    def decay_step(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

    def reset(self):
        self.epsilon = self.initial_epsilon


class UCB:
    """Upper Confidence Bound action selection."""

    def __init__(self, c=2.0, seed=None):
        self.c = c
        self._rng = np.random.RandomState(seed)
        self.counts = None
        self.total_count = 0

    def init(self, n_actions):
        self.counts = np.zeros(n_actions)
        self.total_count = 0

    def select(self, q_values):
        n_actions = len(q_values)
        if self.counts is None:
            self.init(n_actions)
        # Play each arm at least once
        unplayed = np.where(self.counts == 0)[0]
        if len(unplayed) > 0:
            return self._rng.choice(unplayed)
        ucb_values = q_values + self.c * np.sqrt(
            np.log(self.total_count) / self.counts
        )
        max_v = np.max(ucb_values)
        best = np.where(ucb_values == max_v)[0]
        return self._rng.choice(best)

    def update(self, action):
        self.counts[action] += 1
        self.total_count += 1


# ============================================================
# Multi-Armed Bandit
# ============================================================
class MultiArmedBandit:
    """N-armed bandit testbed."""

    def __init__(self, n_arms=10, seed=None):
        self.n_arms = n_arms
        self._rng = np.random.RandomState(seed)
        self.true_values = self._rng.randn(n_arms)
        self.q_estimates = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms)
        self.total_reward = 0.0
        self.steps = 0

    def pull(self, arm):
        """Pull arm, get reward (true value + noise)."""
        reward = self.true_values[arm] + self._rng.randn()
        self.action_counts[arm] += 1
        self.steps += 1
        # Incremental mean update
        self.q_estimates[arm] += (
            (reward - self.q_estimates[arm]) / self.action_counts[arm]
        )
        self.total_reward += reward
        return reward

    def optimal_arm(self):
        return int(np.argmax(self.true_values))

    def run(self, n_steps, strategy):
        """Run bandit for n_steps with given strategy. Returns rewards array."""
        rewards = np.zeros(n_steps)
        optimal_actions = np.zeros(n_steps)
        opt = self.optimal_arm()
        for t in range(n_steps):
            action = strategy.select(self.q_estimates)
            reward = self.pull(action)
            rewards[t] = reward
            optimal_actions[t] = 1.0 if action == opt else 0.0
            if hasattr(strategy, 'update'):
                strategy.update(action)
            if hasattr(strategy, 'decay_step'):
                strategy.decay_step()
        return rewards, optimal_actions

    def reset(self):
        self.q_estimates = np.zeros(self.n_arms)
        self.action_counts = np.zeros(self.n_arms)
        self.total_reward = 0.0
        self.steps = 0


# ============================================================
# Q-Learning (Off-policy TD Control)
# ============================================================
class QLearning:
    """Tabular Q-Learning."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=0.1, epsilon_decay=1.0, min_epsilon=0.01, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))
        self.strategy = EpsilonGreedy(epsilon, epsilon_decay, min_epsilon, seed)
        self._rng = np.random.RandomState(seed)

    def select_action(self, state):
        return self.strategy.select(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    def train(self, env, n_episodes=1000, max_steps=200):
        """Train on environment. Returns episode rewards."""
        episode_rewards = []
        for ep in range(n_episodes):
            state = env.reset()
            total_reward = 0.0
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                self.update(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                if done:
                    break
            self.strategy.decay_step()
            episode_rewards.append(total_reward)
        return episode_rewards

    def get_policy(self):
        """Return greedy policy as array of best actions per state."""
        return np.argmax(self.Q, axis=1)

    def get_value(self):
        """Return value function (max Q per state)."""
        return np.max(self.Q, axis=1)


# ============================================================
# SARSA (On-policy TD Control)
# ============================================================
class SARSA:
    """Tabular SARSA."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=0.1, epsilon_decay=1.0, min_epsilon=0.01, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))
        self.strategy = EpsilonGreedy(epsilon, epsilon_decay, min_epsilon, seed)

    def select_action(self, state):
        return self.strategy.select(self.Q[state])

    def update(self, state, action, reward, next_state, next_action, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state, next_action]
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    def train(self, env, n_episodes=1000, max_steps=200):
        episode_rewards = []
        for ep in range(n_episodes):
            state = env.reset()
            action = self.select_action(state)
            total_reward = 0.0
            for step in range(max_steps):
                next_state, reward, done, info = env.step(action)
                next_action = self.select_action(next_state)
                self.update(state, action, reward, next_state, next_action, done)
                total_reward += reward
                state = next_state
                action = next_action
                if done:
                    break
            self.strategy.decay_step()
            episode_rewards.append(total_reward)
        return episode_rewards

    def get_policy(self):
        return np.argmax(self.Q, axis=1)


# ============================================================
# Expected SARSA
# ============================================================
class ExpectedSARSA:
    """Expected SARSA: uses expected value under policy instead of sample."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=0.1, epsilon_decay=1.0, min_epsilon=0.01, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))
        self.strategy = EpsilonGreedy(epsilon, epsilon_decay, min_epsilon, seed)

    def select_action(self, state):
        return self.strategy.select(self.Q[state])

    def _expected_q(self, state):
        """Compute expected Q under epsilon-greedy policy."""
        eps = self.strategy.epsilon
        q = self.Q[state]
        best_a = np.argmax(q)
        # Each action has eps/n_actions probability; best has extra (1-eps)
        probs = np.full(self.n_actions, eps / self.n_actions)
        probs[best_a] += 1.0 - eps
        return np.dot(probs, q)

    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * self._expected_q(next_state)
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    def train(self, env, n_episodes=1000, max_steps=200):
        episode_rewards = []
        for ep in range(n_episodes):
            state = env.reset()
            total_reward = 0.0
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                self.update(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                if done:
                    break
            self.strategy.decay_step()
            episode_rewards.append(total_reward)
        return episode_rewards

    def get_policy(self):
        return np.argmax(self.Q, axis=1)


# ============================================================
# N-Step SARSA
# ============================================================
class NStepSARSA:
    """N-step SARSA with bootstrapping."""

    def __init__(self, n_states, n_actions, n=3, alpha=0.1, gamma=0.99,
                 epsilon=0.1, epsilon_decay=1.0, min_epsilon=0.01, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))
        self.strategy = EpsilonGreedy(epsilon, epsilon_decay, min_epsilon, seed)

    def select_action(self, state):
        return self.strategy.select(self.Q[state])

    def train(self, env, n_episodes=1000, max_steps=200):
        episode_rewards = []
        for ep in range(n_episodes):
            state = env.reset()
            action = self.select_action(state)
            states = [state]
            actions = [action]
            rewards = [0.0]  # R_0 unused, index from 1
            T = max_steps + n_episodes  # large sentinel (not inf)
            t = 0
            total_reward = 0.0

            while True:
                if t < T:
                    next_state, reward, done, _ = env.step(actions[t])
                    rewards.append(reward)
                    states.append(next_state)
                    total_reward += reward
                    if done:
                        T = t + 1
                    else:
                        next_action = self.select_action(next_state)
                        actions.append(next_action)

                tau = t - self.n + 1
                if tau >= 0:
                    G = 0.0
                    for i in range(tau + 1, min(tau + self.n, T) + 1):
                        G += (self.gamma ** (i - tau - 1)) * rewards[i]
                    if tau + self.n < T:
                        s_n = states[tau + self.n]
                        a_n = actions[tau + self.n]
                        G += (self.gamma ** self.n) * self.Q[s_n, a_n]
                    s_tau = states[tau]
                    a_tau = actions[tau]
                    self.Q[s_tau, a_tau] += self.alpha * (G - self.Q[s_tau, a_tau])

                if tau == T - 1:
                    break
                t += 1
                if t >= max_steps:
                    break

            self.strategy.decay_step()
            episode_rewards.append(total_reward)
        return episode_rewards

    def get_policy(self):
        return np.argmax(self.Q, axis=1)


# ============================================================
# Monte Carlo Control (First-Visit)
# ============================================================
class MonteCarloControl:
    """First-visit Monte Carlo control with epsilon-greedy."""

    def __init__(self, n_states, n_actions, gamma=0.99,
                 epsilon=0.1, epsilon_decay=1.0, min_epsilon=0.01, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(int)
        self.strategy = EpsilonGreedy(epsilon, epsilon_decay, min_epsilon, seed)
        self._rng = np.random.RandomState(seed)

    def select_action(self, state):
        return self.strategy.select(self.Q[state])

    def train(self, env, n_episodes=1000, max_steps=200):
        episode_rewards = []
        for ep in range(n_episodes):
            # Generate episode
            episode = []
            state = env.reset()
            total_reward = 0.0
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                episode.append((state, action, reward))
                total_reward += reward
                state = next_state
                if done:
                    break

            # First-visit MC update
            G = 0.0
            visited = set()
            for t in range(len(episode) - 1, -1, -1):
                s, a, r = episode[t]
                G = self.gamma * G + r
                if (s, a) not in visited:
                    visited.add((s, a))
                    self.returns_sum[(s, a)] += G
                    self.returns_count[(s, a)] += 1
                    self.Q[s, a] = (
                        self.returns_sum[(s, a)] / self.returns_count[(s, a)]
                    )

            self.strategy.decay_step()
            episode_rewards.append(total_reward)
        return episode_rewards

    def get_policy(self):
        return np.argmax(self.Q, axis=1)


# ============================================================
# Dyna-Q (Model-Based)
# ============================================================
class DynaQ:
    """Dyna-Q: Q-learning + learned model + planning steps."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=0.1, n_planning=5, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.n_planning = n_planning
        self.Q = np.zeros((n_states, n_actions))
        self.strategy = EpsilonGreedy(epsilon, seed=seed)
        self._rng = np.random.RandomState(seed)
        # Model: (s, a) -> (r, s')
        self.model = {}
        self.observed_sa = []

    def select_action(self, state):
        return self.strategy.select(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        # Direct RL update
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

        # Update model
        self.model[(state, action)] = (reward, next_state, done)
        if (state, action) not in self.observed_sa:
            self.observed_sa.append((state, action))

        # Planning: simulate n_planning steps from model
        for _ in range(self.n_planning):
            if not self.observed_sa:
                break
            idx = self._rng.randint(len(self.observed_sa))
            s, a = self.observed_sa[idx]
            r, ns, d = self.model[(s, a)]
            if d:
                t = r
            else:
                t = r + self.gamma * np.max(self.Q[ns])
            self.Q[s, a] += self.alpha * (t - self.Q[s, a])

    def train(self, env, n_episodes=1000, max_steps=200):
        episode_rewards = []
        for ep in range(n_episodes):
            state = env.reset()
            total_reward = 0.0
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                if done:
                    break
            episode_rewards.append(total_reward)
        return episode_rewards

    def get_policy(self):
        return np.argmax(self.Q, axis=1)


# ============================================================
# REINFORCE (Policy Gradient)
# ============================================================
class PolicyGradient:
    """REINFORCE with baseline (tabular softmax policy)."""

    def __init__(self, n_states, n_actions, alpha=0.01, gamma=0.99,
                 baseline_alpha=0.01, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.baseline_alpha = baseline_alpha
        # Preferences (logits) for softmax policy
        self.theta = np.zeros((n_states, n_actions))
        # State-value baseline
        self.V = np.zeros(n_states)
        self._rng = np.random.RandomState(seed)

    def _softmax(self, x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    def get_probs(self, state):
        return self._softmax(self.theta[state])

    def select_action(self, state):
        probs = self.get_probs(state)
        return self._rng.choice(self.n_actions, p=probs)

    def train(self, env, n_episodes=1000, max_steps=200):
        episode_rewards = []
        for ep in range(n_episodes):
            # Generate episode
            states, actions, rewards = [], [], []
            state = env.reset()
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state
                if done:
                    break

            total_reward = sum(rewards)
            episode_rewards.append(total_reward)

            # Compute returns
            T = len(rewards)
            G = np.zeros(T)
            g = 0.0
            for t in range(T - 1, -1, -1):
                g = rewards[t] + self.gamma * g
                G[t] = g

            # Update policy and baseline
            for t in range(T):
                s = states[t]
                a = actions[t]
                advantage = G[t] - self.V[s]
                # Update baseline
                self.V[s] += self.baseline_alpha * (G[t] - self.V[s])
                # Policy gradient: increase log-prob of action by advantage
                probs = self.get_probs(s)
                grad = -probs.copy()
                grad[a] += 1.0  # grad of log pi(a|s) w.r.t. theta
                self.theta[s] += self.alpha * (self.gamma ** t) * advantage * grad

        return episode_rewards

    def get_policy(self):
        """Return greedy policy."""
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            policy[s] = np.argmax(self.get_probs(s))
        return policy


# ============================================================
# Actor-Critic (One-step TD)
# ============================================================
class ActorCritic:
    """One-step Actor-Critic with tabular softmax policy."""

    def __init__(self, n_states, n_actions, actor_alpha=0.01, critic_alpha=0.1,
                 gamma=0.99, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.actor_alpha = actor_alpha
        self.critic_alpha = critic_alpha
        self.gamma = gamma
        self.theta = np.zeros((n_states, n_actions))
        self.V = np.zeros(n_states)
        self._rng = np.random.RandomState(seed)

    def _softmax(self, x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    def get_probs(self, state):
        return self._softmax(self.theta[state])

    def select_action(self, state):
        probs = self.get_probs(state)
        return self._rng.choice(self.n_actions, p=probs)

    def train(self, env, n_episodes=1000, max_steps=200):
        episode_rewards = []
        for ep in range(n_episodes):
            state = env.reset()
            total_reward = 0.0
            I = 1.0
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                # TD error
                if done:
                    td_error = reward - self.V[state]
                else:
                    td_error = reward + self.gamma * self.V[next_state] - self.V[state]

                # Critic update
                self.V[state] += self.critic_alpha * td_error

                # Actor update
                probs = self.get_probs(state)
                grad = -probs.copy()
                grad[action] += 1.0
                self.theta[state] += self.actor_alpha * I * td_error * grad

                I *= self.gamma
                state = next_state
                if done:
                    break
            episode_rewards.append(total_reward)
        return episode_rewards

    def get_policy(self):
        policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            policy[s] = np.argmax(self.get_probs(s))
        return policy


# ============================================================
# TD(lambda) with Eligibility Traces
# ============================================================
class TDLambda:
    """TD(lambda) with eligibility traces for state-value estimation + Q control."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 lam=0.8, epsilon=0.1, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.Q = np.zeros((n_states, n_actions))
        self.strategy = EpsilonGreedy(epsilon, seed=seed)

    def select_action(self, state):
        return self.strategy.select(self.Q[state])

    def train(self, env, n_episodes=1000, max_steps=200):
        episode_rewards = []
        for ep in range(n_episodes):
            state = env.reset()
            action = self.select_action(state)
            E = np.zeros((self.n_states, self.n_actions))  # eligibility traces
            total_reward = 0.0

            for step in range(max_steps):
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                next_action = self.select_action(next_state)

                if done:
                    td_error = reward - self.Q[state, action]
                else:
                    td_error = (reward + self.gamma * self.Q[next_state, next_action]
                                - self.Q[state, action])

                E[state, action] += 1.0  # accumulating traces

                # Update all Q values
                self.Q += self.alpha * td_error * E
                E *= self.gamma * self.lam

                state = next_state
                action = next_action
                if done:
                    break

            self.strategy.decay_step()
            episode_rewards.append(total_reward)
        return episode_rewards

    def get_policy(self):
        return np.argmax(self.Q, axis=1)


# ============================================================
# Double Q-Learning
# ============================================================
class DoubleQLearning:
    """Double Q-Learning: reduces maximization bias."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=0.1, epsilon_decay=1.0, min_epsilon=0.01, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.Q1 = np.zeros((n_states, n_actions))
        self.Q2 = np.zeros((n_states, n_actions))
        self.strategy = EpsilonGreedy(epsilon, epsilon_decay, min_epsilon, seed)
        self._rng = np.random.RandomState(seed)

    def select_action(self, state):
        # Use sum of Q1 + Q2 for behavior
        return self.strategy.select(self.Q1[state] + self.Q2[state])

    def update(self, state, action, reward, next_state, done):
        if self._rng.random() < 0.5:
            # Update Q1 using Q2's evaluation
            if done:
                target = reward
            else:
                best_a = np.argmax(self.Q1[next_state])
                target = reward + self.gamma * self.Q2[next_state, best_a]
            self.Q1[state, action] += self.alpha * (target - self.Q1[state, action])
        else:
            # Update Q2 using Q1's evaluation
            if done:
                target = reward
            else:
                best_a = np.argmax(self.Q2[next_state])
                target = reward + self.gamma * self.Q1[next_state, best_a]
            self.Q2[state, action] += self.alpha * (target - self.Q2[state, action])

    def train(self, env, n_episodes=1000, max_steps=200):
        episode_rewards = []
        for ep in range(n_episodes):
            state = env.reset()
            total_reward = 0.0
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                if done:
                    break
            self.strategy.decay_step()
            episode_rewards.append(total_reward)
        return episode_rewards

    def get_policy(self):
        return np.argmax(self.Q1 + self.Q2, axis=1)

    def get_value(self):
        return np.max(self.Q1 + self.Q2, axis=1) / 2.0


# ============================================================
# Prioritized Sweeping
# ============================================================
class PrioritizedSweeping:
    """Prioritized Sweeping: model-based planning with priority queue."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=0.1, n_planning=5, priority_threshold=0.01, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.n_planning = n_planning
        self.priority_threshold = priority_threshold
        self.Q = np.zeros((n_states, n_actions))
        self.strategy = EpsilonGreedy(epsilon, seed=seed)
        self._rng = np.random.RandomState(seed)
        # Model
        self.model = {}
        # Predecessors: which (s,a) lead to s'?
        self.predecessors = defaultdict(set)
        # Priority queue (max-heap via negative priorities)
        self.pq = []
        self.pq_set = {}  # track current priorities

    def select_action(self, state):
        return self.strategy.select(self.Q[state])

    def _add_to_pq(self, state, action, priority):
        if priority >= self.priority_threshold:
            # Use negative for max-heap behavior
            heapq.heappush(self.pq, (-priority, state, action))
            self.pq_set[(state, action)] = priority

    def update(self, state, action, reward, next_state, done):
        # Update model and predecessors
        self.model[(state, action)] = (reward, next_state, done)
        self.predecessors[next_state].add((state, action))

        # Compute priority
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        priority = abs(target - self.Q[state, action])
        self._add_to_pq(state, action, priority)

        # Planning loop
        for _ in range(self.n_planning):
            if not self.pq:
                break
            neg_p, s, a = heapq.heappop(self.pq)
            self.pq_set.pop((s, a), None)

            r, ns, d = self.model[(s, a)]
            if d:
                t = r
            else:
                t = r + self.gamma * np.max(self.Q[ns])
            self.Q[s, a] += self.alpha * (t - self.Q[s, a])

            # Update predecessors
            for ps, pa in self.predecessors[s]:
                if (ps, pa) in self.model:
                    pr, pns, pd = self.model[(ps, pa)]
                    if pd:
                        pt = pr
                    else:
                        pt = pr + self.gamma * np.max(self.Q[pns])
                    pp = abs(pt - self.Q[ps, pa])
                    self._add_to_pq(ps, pa, pp)

    def train(self, env, n_episodes=1000, max_steps=200):
        episode_rewards = []
        for ep in range(n_episodes):
            state = env.reset()
            total_reward = 0.0
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                if done:
                    break
            episode_rewards.append(total_reward)
        return episode_rewards

    def get_policy(self):
        return np.argmax(self.Q, axis=1)
