"""
C179: Reinforcement Learning
Composing C140 Neural Network for function approximation.

Components:
- Environment: Base class + CartPole, GridWorld, Bandit environments
- ReplayBuffer: Experience replay with uniform and prioritized sampling
- Policies: EpsilonGreedy, Boltzmann, UCB
- DQN: Deep Q-Network with target network and experience replay
- REINFORCE: Policy gradient with baseline
- ActorCritic: Advantage actor-critic (A2C)
- PPO: Proximal Policy Optimization (clipped objective)
- QLearning: Tabular Q-learning
- SARSA: Tabular on-policy TD control
- MultiArmedBandit: UCB1, Thompson sampling, epsilon-greedy
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))
from neural_network import (
    Tensor, Dense, Activation, Sequential,
    MSELoss, SGD, Adam,
    relu, sigmoid, tanh_act, softmax, softmax_batch,
    xavier_init, he_init,
    train_step,
)


# ============================================================
# Tensor helpers
# ============================================================

def _get_row(t, i):
    if len(t.shape) == 1:
        return Tensor([t.data[i]])
    return Tensor(t.data[i][:])

def _set_row(t, i, row):
    if len(t.shape) == 1:
        t.data[i] = row.data[0] if isinstance(row, Tensor) else row
    else:
        t.data[i] = row.data[:] if isinstance(row, Tensor) else row[:]

def _stack_rows(rows):
    if not rows:
        return Tensor([])
    data = []
    for r in rows:
        if isinstance(r, Tensor):
            if len(r.shape) == 1:
                data.append(r.data[:])
            else:
                data.extend(r.data)
        elif isinstance(r, list):
            data.append(r[:])
    return Tensor(data)

def _batch_size(t):
    return t.shape[0]

def _ones_like(t):
    return Tensor.ones(t.shape)

def _zeros_like(t):
    return Tensor.zeros(t.shape)

def _flat_values(t):
    if len(t.shape) == 1:
        return t.data[:]
    vals = []
    for row in t.data:
        vals.extend(row)
    return vals

def _scale_tensor(t, s):
    if len(t.shape) == 1:
        return Tensor([v * s for v in t.data])
    return Tensor([[v * s for v in row] for row in t.data])

def _add_tensors(a, b):
    if len(a.shape) == 1:
        return Tensor([x + y for x, y in zip(a.data, b.data)])
    return Tensor([[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a.data, b.data)])

def _sub_tensors(a, b):
    if len(a.shape) == 1:
        return Tensor([x - y for x, y in zip(a.data, b.data)])
    return Tensor([[x - y for x, y in zip(ra, rb)] for ra, rb in zip(a.data, b.data)])


# ============================================================
# RNG
# ============================================================

class _RNG:
    def __init__(self, seed=42):
        self._rng = random.Random(seed)

    def random(self):
        return self._rng.random()

    def uniform(self, lo, hi):
        return lo + (hi - lo) * self._rng.random()

    def normal(self, mean=0.0, std=1.0):
        u1 = max(self._rng.random(), 1e-10)
        u2 = self._rng.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mean + std * z

    def randint(self, lo, hi):
        return self._rng.randint(lo, hi)

    def choice(self, seq):
        return seq[self._rng.randint(0, len(seq) - 1)]

    def shuffle(self, lst):
        self._rng.shuffle(lst)

    def beta(self, a, b):
        """Beta distribution via gamma sampling."""
        x = self._gamma(a)
        y = self._gamma(b)
        return x / (x + y) if (x + y) > 0 else 0.5

    def _gamma(self, shape_param):
        """Gamma(shape, 1) via Marsaglia-Tsang method."""
        if shape_param < 1:
            return self._gamma(shape_param + 1) * (self.random() ** (1.0 / shape_param))
        d = shape_param - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            x = self.normal()
            v = (1.0 + c * x) ** 3
            if v > 0:
                u = self.random()
                if u < 1.0 - 0.0331 * (x * x) * (x * x):
                    return d * v
                if math.log(max(u, 1e-10)) < 0.5 * x * x + d * (1.0 - v + math.log(max(v, 1e-10))):
                    return d * v


# ============================================================
# Environments
# ============================================================

class Environment:
    """Base environment interface."""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        """Returns (next_state, reward, done, info)."""
        raise NotImplementedError


class GridWorld(Environment):
    """Simple grid world environment.

    Agent navigates a grid to reach goal. Walls block movement.
    Actions: 0=up, 1=right, 2=down, 3=left
    """

    def __init__(self, rows=5, cols=5, start=(0, 0), goal=None, walls=None, seed=42):
        super().__init__(state_size=rows * cols, action_size=4)
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal or (rows - 1, cols - 1)
        self.walls = set(walls) if walls else set()
        self.pos = start
        self.rng = _RNG(seed)
        self.max_steps = rows * cols * 2
        self.steps = 0

    def reset(self):
        self.pos = self.start
        self.steps = 0
        return self._state()

    def _state(self):
        """One-hot encoding of position."""
        s = [0.0] * (self.rows * self.cols)
        s[self.pos[0] * self.cols + self.pos[1]] = 1.0
        return s

    def step(self, action):
        self.steps += 1
        r, c = self.pos
        dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
        nr, nc = r + dr, c + dc

        if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self.walls:
            self.pos = (nr, nc)

        done = self.pos == self.goal or self.steps >= self.max_steps
        reward = 1.0 if self.pos == self.goal else -0.01
        return self._state(), reward, done, {}


class CartPole(Environment):
    """Simplified CartPole environment.

    State: [x, x_dot, theta, theta_dot]
    Actions: 0=push left, 1=push right
    """

    def __init__(self, seed=42):
        super().__init__(state_size=4, action_size=2)
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.total_mass = self.cart_mass + self.pole_mass
        self.pole_length = 0.5
        self.force_mag = 10.0
        self.dt = 0.02
        self.theta_threshold = 12 * math.pi / 180
        self.x_threshold = 2.4
        self.rng = _RNG(seed)
        self.state = None
        self.steps = 0
        self.max_steps = 200

    def reset(self):
        self.state = [
            self.rng.uniform(-0.05, 0.05),
            self.rng.uniform(-0.05, 0.05),
            self.rng.uniform(-0.05, 0.05),
            self.rng.uniform(-0.05, 0.05),
        ]
        self.steps = 0
        return self.state[:]

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        temp = (force + self.pole_mass * self.pole_length * theta_dot ** 2 * sin_theta) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.pole_length * (4.0 / 3.0 - self.pole_mass * cos_theta ** 2 / self.total_mass)
        )
        x_acc = temp - self.pole_mass * self.pole_length * theta_acc * cos_theta / self.total_mass

        x += self.dt * x_dot
        x_dot += self.dt * x_acc
        theta += self.dt * theta_dot
        theta_dot += self.dt * theta_acc

        self.state = [x, x_dot, theta, theta_dot]
        self.steps += 1

        done = (
            abs(x) > self.x_threshold
            or abs(theta) > self.theta_threshold
            or self.steps >= self.max_steps
        )
        reward = 0.0 if done and self.steps < self.max_steps else 1.0
        return self.state[:], reward, done, {}


class Bandit(Environment):
    """Multi-armed bandit environment.

    Each arm has a fixed reward probability.
    State is always [1.0] (stateless).
    """

    def __init__(self, n_arms=10, probs=None, seed=42):
        super().__init__(state_size=1, action_size=n_arms)
        self.rng = _RNG(seed)
        if probs:
            self.probs = probs[:]
        else:
            self.probs = [self.rng.random() for _ in range(n_arms)]

    def reset(self):
        return [1.0]

    def step(self, action):
        reward = 1.0 if self.rng.random() < self.probs[action] else 0.0
        return [1.0], reward, False, {}


class MountainCar(Environment):
    """Simplified Mountain Car environment.

    State: [position, velocity]
    Actions: 0=push left, 1=no push, 2=push right
    Goal: reach position >= 0.5
    """

    def __init__(self, seed=42):
        super().__init__(state_size=2, action_size=3)
        self.min_pos = -1.2
        self.max_pos = 0.6
        self.max_vel = 0.07
        self.goal_pos = 0.5
        self.rng = _RNG(seed)
        self.state = None
        self.steps = 0
        self.max_steps = 200

    def reset(self):
        self.state = [self.rng.uniform(-0.6, -0.4), 0.0]
        self.steps = 0
        return self.state[:]

    def step(self, action):
        pos, vel = self.state
        vel += (action - 1) * 0.001 + math.cos(3 * pos) * (-0.0025)
        vel = max(-self.max_vel, min(self.max_vel, vel))
        pos += vel
        pos = max(self.min_pos, min(self.max_pos, pos))
        if pos == self.min_pos and vel < 0:
            vel = 0.0

        self.state = [pos, vel]
        self.steps += 1
        done = pos >= self.goal_pos or self.steps >= self.max_steps
        reward = -1.0  # penalty each step
        if pos >= self.goal_pos:
            reward = 0.0
        return self.state[:], reward, done, {}


# ============================================================
# Experience Replay
# ============================================================

class ReplayBuffer:
    """Experience replay buffer with uniform sampling."""

    def __init__(self, capacity=10000, seed=42):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.rng = _RNG(seed)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        indices = [self.rng.randint(0, len(self.buffer) - 1) for _ in range(batch_size)]
        batch = [self.buffer[i] for i in indices]
        states = [b[0] for b in batch]
        actions = [b[1] for b in batch]
        rewards = [b[2] for b in batch]
        next_states = [b[3] for b in batch]
        dones = [b[4] for b in batch]
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized experience replay with proportional priorities."""

    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, seed=42):
        self.capacity = capacity
        self.alpha = alpha  # prioritization exponent
        self.beta = beta    # importance sampling exponent
        self.buffer = []
        self.priorities = []
        self.pos = 0
        self.rng = _RNG(seed)
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(self.max_priority)
        else:
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        n = len(self.buffer)
        probs = [p ** self.alpha for p in self.priorities]
        total = sum(probs)
        probs = [p / total for p in probs]

        # Sample proportional to priorities
        indices = []
        for _ in range(batch_size):
            r = self.rng.random()
            cumsum = 0.0
            for i, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    indices.append(i)
                    break
            else:
                indices.append(n - 1)

        # Importance sampling weights
        weights = []
        max_w = 0.0
        for i in indices:
            w = (n * probs[i]) ** (-self.beta)
            weights.append(w)
            max_w = max(max_w, w)
        weights = [w / max_w for w in weights]  # normalize

        batch = [self.buffer[i] for i in indices]
        states = [b[0] for b in batch]
        actions = [b[1] for b in batch]
        rewards = [b[2] for b in batch]
        next_states = [b[3] for b in batch]
        dones = [b[4] for b in batch]
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        for i, td in zip(indices, td_errors):
            self.priorities[i] = abs(td) + 1e-6
            self.max_priority = max(self.max_priority, self.priorities[i])

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Policies
# ============================================================

class EpsilonGreedy:
    """Epsilon-greedy action selection with decay."""

    def __init__(self, epsilon=1.0, min_epsilon=0.01, decay=0.995, seed=42):
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.rng = _RNG(seed)

    def select(self, q_values, n_actions=None):
        if self.rng.random() < self.epsilon:
            n = n_actions or len(q_values)
            return self.rng.randint(0, n - 1)
        if isinstance(q_values, Tensor):
            q_values = _flat_values(q_values)
        return max(range(len(q_values)), key=lambda i: q_values[i])

    def step(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)


class BoltzmannPolicy:
    """Boltzmann (softmax) action selection with temperature."""

    def __init__(self, temperature=1.0, min_temp=0.1, decay=0.995, seed=42):
        self.temperature = temperature
        self.min_temp = min_temp
        self.decay = decay
        self.rng = _RNG(seed)

    def select(self, q_values):
        if isinstance(q_values, Tensor):
            q_values = _flat_values(q_values)
        scaled = [q / self.temperature for q in q_values]
        probs = softmax(scaled)
        r = self.rng.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                return i
        return len(probs) - 1

    def step(self):
        self.temperature = max(self.min_temp, self.temperature * self.decay)


class UCBPolicy:
    """Upper Confidence Bound action selection."""

    def __init__(self, c=2.0):
        self.c = c
        self.counts = {}
        self.total = 0

    def select(self, q_values):
        if isinstance(q_values, Tensor):
            q_values = _flat_values(q_values)
        n_actions = len(q_values)
        # Try each action at least once
        for a in range(n_actions):
            if self.counts.get(a, 0) == 0:
                return a
        ucb_values = []
        for a in range(n_actions):
            bonus = self.c * math.sqrt(math.log(self.total) / self.counts[a])
            ucb_values.append(q_values[a] + bonus)
        return max(range(n_actions), key=lambda i: ucb_values[i])

    def update(self, action):
        self.counts[action] = self.counts.get(action, 0) + 1
        self.total += 1


# ============================================================
# Tabular Methods
# ============================================================

class QLearning:
    """Tabular Q-learning (off-policy TD control)."""

    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, seed=42):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.q_table = [[0.0] * n_actions for _ in range(n_states)]
        self.rng = _RNG(seed)

    def get_action(self, state, epsilon=0.1):
        if self.rng.random() < epsilon:
            return self.rng.randint(0, self.n_actions - 1)
        q_vals = self.q_table[state]
        return max(range(self.n_actions), key=lambda a: q_vals[a])

    def update(self, state, action, reward, next_state, done):
        best_next = max(self.q_table[next_state]) if not done else 0.0
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error
        return td_error

    def get_q_values(self, state):
        return self.q_table[state][:]

    def get_policy(self):
        """Returns greedy policy as list of best actions per state."""
        return [max(range(self.n_actions), key=lambda a: self.q_table[s][a])
                for s in range(self.n_states)]


class SARSA:
    """Tabular SARSA (on-policy TD control)."""

    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, seed=42):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.q_table = [[0.0] * n_actions for _ in range(n_states)]
        self.rng = _RNG(seed)

    def get_action(self, state, epsilon=0.1):
        if self.rng.random() < epsilon:
            return self.rng.randint(0, self.n_actions - 1)
        q_vals = self.q_table[state]
        return max(range(self.n_actions), key=lambda a: q_vals[a])

    def update(self, state, action, reward, next_state, next_action, done):
        next_q = self.q_table[next_state][next_action] if not done else 0.0
        td_target = reward + self.gamma * next_q
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error
        return td_error

    def get_q_values(self, state):
        return self.q_table[state][:]

    def get_policy(self):
        return [max(range(self.n_actions), key=lambda a: self.q_table[s][a])
                for s in range(self.n_states)]


# ============================================================
# Multi-Armed Bandit Solvers
# ============================================================

class MultiArmedBandit:
    """Multi-armed bandit solver with multiple strategies."""

    def __init__(self, n_arms, strategy='ucb1', seed=42, **kwargs):
        self.n_arms = n_arms
        self.strategy = strategy
        self.rng = _RNG(seed)
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.total_pulls = 0

        # Strategy-specific params
        self.epsilon = kwargs.get('epsilon', 0.1)
        self.c = kwargs.get('c', 2.0)
        # Thompson sampling
        self.alpha_params = [1.0] * n_arms
        self.beta_params = [1.0] * n_arms

    def select_arm(self):
        if self.strategy == 'epsilon_greedy':
            return self._epsilon_greedy()
        elif self.strategy == 'ucb1':
            return self._ucb1()
        elif self.strategy == 'thompson':
            return self._thompson()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _epsilon_greedy(self):
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.n_arms - 1)
        return max(range(self.n_arms), key=lambda a: self.values[a])

    def _ucb1(self):
        for a in range(self.n_arms):
            if self.counts[a] == 0:
                return a
        ucb_vals = []
        for a in range(self.n_arms):
            bonus = self.c * math.sqrt(math.log(self.total_pulls) / self.counts[a])
            ucb_vals.append(self.values[a] + bonus)
        return max(range(self.n_arms), key=lambda a: ucb_vals[a])

    def _thompson(self):
        samples = [self.rng.beta(self.alpha_params[a], self.beta_params[a])
                   for a in range(self.n_arms)]
        return max(range(self.n_arms), key=lambda a: samples[a])

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.total_pulls += 1
        # Incremental mean update
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        # Thompson sampling beta update
        if reward > 0:
            self.alpha_params[arm] += 1.0
        else:
            self.beta_params[arm] += 1.0

    def get_best_arm(self):
        return max(range(self.n_arms), key=lambda a: self.values[a])

    def get_regret(self, optimal_value, rewards):
        """Calculate cumulative regret."""
        cumulative = 0.0
        regrets = []
        for r in rewards:
            cumulative += optimal_value - r
            regrets.append(cumulative)
        return regrets


# ============================================================
# Deep Q-Network (DQN)
# ============================================================

class DQN:
    """Deep Q-Network with target network and experience replay.

    Composes C140 Sequential for Q-value approximation.
    """

    def __init__(self, state_size, action_size, hidden_sizes=None,
                 lr=0.001, gamma=0.99, tau=0.01, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau  # soft update rate
        self.rng = _RNG(seed)

        hidden_sizes = hidden_sizes or [64, 64]

        # Build Q-network
        self.q_net = self._build_network(state_size, action_size, hidden_sizes)
        # Build target network (copy)
        self.target_net = self._build_network(state_size, action_size, hidden_sizes)
        self._copy_weights(self.q_net, self.target_net)

        self.optimizer = Adam(lr=lr)
        self.loss_fn = MSELoss()

    def _build_network(self, input_size, output_size, hidden_sizes):
        net = Sequential()
        prev = input_size
        for h in hidden_sizes:
            net.add(Dense(prev, h, init='he', rng=random.Random(self.rng.randint(0, 99999))))
            net.add(Activation('relu'))
            prev = h
        net.add(Dense(prev, output_size, init='xavier', rng=random.Random(self.rng.randint(0, 99999))))
        return net

    def _copy_weights(self, source, target):
        """Hard copy weights from source to target."""
        src_layers = source.get_trainable_layers()
        tgt_layers = target.get_trainable_layers()
        for sl, tl in zip(src_layers, tgt_layers):
            src_params = sl.get_params()
            tgt_params = tl.get_params()
            for (sv, _, _), (tv, _, _) in zip(src_params, tgt_params):
                if len(sv.shape) == 1:
                    tv.data = sv.data[:]
                else:
                    tv.data = [row[:] for row in sv.data]

    def soft_update(self):
        """Polyak averaging: target = tau * q + (1-tau) * target."""
        src_layers = self.q_net.get_trainable_layers()
        tgt_layers = self.target_net.get_trainable_layers()
        for sl, tl in zip(src_layers, tgt_layers):
            src_params = sl.get_params()
            tgt_params = tl.get_params()
            for (sv, _, _), (tv, _, _) in zip(src_params, tgt_params):
                if len(sv.shape) == 1:
                    tv.data = [self.tau * s + (1 - self.tau) * t
                               for s, t in zip(sv.data, tv.data)]
                else:
                    tv.data = [[self.tau * s + (1 - self.tau) * t
                                for s, t in zip(sr, tr)]
                               for sr, tr in zip(sv.data, tv.data)]

    def hard_update(self):
        """Copy Q-network weights to target network."""
        self._copy_weights(self.q_net, self.target_net)

    def get_action(self, state, epsilon=0.0):
        """Select action using epsilon-greedy on Q-values."""
        if self.rng.random() < epsilon:
            return self.rng.randint(0, self.action_size - 1)
        q_values = self.q_net.forward(Tensor([state]))
        q_flat = _flat_values(q_values)
        return max(range(self.action_size), key=lambda a: q_flat[a])

    def get_q_values(self, state):
        """Get Q-values for a state."""
        q_values = self.q_net.forward(Tensor([state]))
        return _flat_values(q_values)

    def train_batch(self, states, actions, rewards, next_states, dones):
        """Train on a batch of experiences. Returns mean loss."""
        batch_size = len(states)

        # Compute target Q-values
        targets = []
        for i in range(batch_size):
            if dones[i]:
                targets.append(rewards[i])
            else:
                next_q = self.target_net.forward(Tensor([next_states[i]]))
                next_q_flat = _flat_values(next_q)
                max_next_q = max(next_q_flat)
                targets.append(rewards[i] + self.gamma * max_next_q)

        # Compute current Q-values and build target tensor
        total_loss = 0.0
        self.q_net.train()
        for i in range(batch_size):
            state_t = Tensor([states[i]])
            q_values = self.q_net.forward(state_t)
            q_flat = _flat_values(q_values)

            # Build target: same as q_values except for the taken action
            target_flat = q_flat[:]
            target_flat[actions[i]] = targets[i]

            q_tensor = Tensor([q_flat])
            target_tensor = Tensor([target_flat])

            loss = self.loss_fn.forward(q_tensor, target_tensor)
            total_loss += loss

            grad = self.loss_fn.backward(q_tensor, target_tensor)
            self.q_net.backward(grad)
            self.optimizer.step(self.q_net.get_trainable_layers())

        return total_loss / batch_size


class DoubleDQN(DQN):
    """Double DQN: uses online network for action selection,
    target network for value estimation."""

    def train_batch(self, states, actions, rewards, next_states, dones):
        batch_size = len(states)

        targets = []
        for i in range(batch_size):
            if dones[i]:
                targets.append(rewards[i])
            else:
                # Select action with online network
                online_q = self.q_net.forward(Tensor([next_states[i]]))
                online_flat = _flat_values(online_q)
                best_action = max(range(self.action_size), key=lambda a: online_flat[a])
                # Evaluate with target network
                target_q = self.target_net.forward(Tensor([next_states[i]]))
                target_flat = _flat_values(target_q)
                targets.append(rewards[i] + self.gamma * target_flat[best_action])

        total_loss = 0.0
        self.q_net.train()
        for i in range(batch_size):
            state_t = Tensor([states[i]])
            q_values = self.q_net.forward(state_t)
            q_flat = _flat_values(q_values)

            target_flat = q_flat[:]
            target_flat[actions[i]] = targets[i]

            q_tensor = Tensor([q_flat])
            target_tensor = Tensor([target_flat])

            loss = self.loss_fn.forward(q_tensor, target_tensor)
            total_loss += loss

            grad = self.loss_fn.backward(q_tensor, target_tensor)
            self.q_net.backward(grad)
            self.optimizer.step(self.q_net.get_trainable_layers())

        return total_loss / batch_size


# ============================================================
# REINFORCE (Policy Gradient)
# ============================================================

class REINFORCE:
    """REINFORCE policy gradient with optional baseline.

    The policy network outputs action probabilities via softmax.
    """

    def __init__(self, state_size, action_size, hidden_sizes=None,
                 lr=0.001, gamma=0.99, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.rng = _RNG(seed)

        hidden_sizes = hidden_sizes or [64]

        # Policy network (outputs logits)
        self.policy_net = Sequential()
        prev = state_size
        for h in hidden_sizes:
            self.policy_net.add(Dense(prev, h, init='he', rng=random.Random(self.rng.randint(0, 99999))))
            self.policy_net.add(Activation('relu'))
            prev = h
        self.policy_net.add(Dense(prev, action_size, init='xavier', rng=random.Random(self.rng.randint(0, 99999))))
        self.policy_net.add(Activation('softmax'))

        self.optimizer = Adam(lr=lr)

        # Episode storage
        self.saved_log_probs = []
        self.saved_rewards = []
        self.saved_states = []
        self.saved_actions = []

        # Baseline (running average of returns)
        self.baseline = 0.0
        self.baseline_count = 0

    def get_action(self, state):
        """Sample action from policy."""
        state_t = Tensor([state])
        probs = self.policy_net.forward(state_t)
        probs_flat = _flat_values(probs)

        # Sample from categorical distribution
        r = self.rng.random()
        cumsum = 0.0
        action = len(probs_flat) - 1
        for i, p in enumerate(probs_flat):
            cumsum += p
            if r <= cumsum:
                action = i
                break

        log_prob = math.log(max(probs_flat[action], 1e-10))
        self.saved_log_probs.append(log_prob)
        self.saved_states.append(state)
        self.saved_actions.append(action)
        return action

    def store_reward(self, reward):
        self.saved_rewards.append(reward)

    def compute_returns(self):
        """Compute discounted returns."""
        returns = []
        G = 0.0
        for r in reversed(self.saved_rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    def update(self):
        """Update policy using collected episode data. Returns mean loss."""
        if not self.saved_rewards:
            return 0.0

        returns = self.compute_returns()

        # Update baseline
        mean_return = sum(returns) / len(returns)
        self.baseline_count += 1
        self.baseline += (mean_return - self.baseline) / self.baseline_count

        # Normalize returns
        std_return = math.sqrt(sum((r - mean_return) ** 2 for r in returns) / max(len(returns), 1))
        if std_return > 1e-8:
            returns = [(r - self.baseline) / std_return for r in returns]
        else:
            returns = [r - self.baseline for r in returns]

        # Policy gradient update
        total_loss = 0.0
        self.policy_net.train()
        for state, action, G in zip(self.saved_states, self.saved_actions, returns):
            state_t = Tensor([state])
            probs = self.policy_net.forward(state_t)
            probs_flat = _flat_values(probs)

            # Gradient of log(pi(a|s)) * G
            # For softmax output: grad = -G * (one_hot(a) - probs)
            grad_flat = [0.0] * self.action_size
            for i in range(self.action_size):
                if i == action:
                    grad_flat[i] = -G * (1.0 - probs_flat[i])
                else:
                    grad_flat[i] = -G * (-probs_flat[i])

            grad = Tensor([grad_flat])
            self.policy_net.backward(grad)
            self.optimizer.step(self.policy_net.get_trainable_layers())
            total_loss += abs(G)

        loss = total_loss / max(len(self.saved_states), 1)
        self._clear_episode()
        return loss

    def _clear_episode(self):
        self.saved_log_probs = []
        self.saved_rewards = []
        self.saved_states = []
        self.saved_actions = []

    def get_action_probs(self, state):
        """Get action probabilities for a state."""
        state_t = Tensor([state])
        probs = self.policy_net.forward(state_t)
        return _flat_values(probs)


# ============================================================
# Actor-Critic (A2C)
# ============================================================

class ActorCritic:
    """Advantage Actor-Critic (A2C).

    Actor: policy network (action probabilities)
    Critic: value network (state value V(s))
    """

    def __init__(self, state_size, action_size, hidden_sizes=None,
                 actor_lr=0.001, critic_lr=0.005, gamma=0.99, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.rng = _RNG(seed)

        hidden_sizes = hidden_sizes or [64]

        # Actor (policy) network
        self.actor = Sequential()
        prev = state_size
        for h in hidden_sizes:
            self.actor.add(Dense(prev, h, init='he', rng=random.Random(self.rng.randint(0, 99999))))
            self.actor.add(Activation('relu'))
            prev = h
        self.actor.add(Dense(prev, action_size, init='xavier', rng=random.Random(self.rng.randint(0, 99999))))
        self.actor.add(Activation('softmax'))

        # Critic (value) network
        self.critic = Sequential()
        prev = state_size
        for h in hidden_sizes:
            self.critic.add(Dense(prev, h, init='he', rng=random.Random(self.rng.randint(0, 99999))))
            self.critic.add(Activation('relu'))
            prev = h
        self.critic.add(Dense(prev, 1, init='xavier', rng=random.Random(self.rng.randint(0, 99999))))

        self.actor_optimizer = Adam(lr=actor_lr)
        self.critic_optimizer = Adam(lr=critic_lr)
        self.value_loss_fn = MSELoss()

    def get_action(self, state):
        """Sample action from actor policy."""
        state_t = Tensor([state])
        probs = self.actor.forward(state_t)
        probs_flat = _flat_values(probs)

        r = self.rng.random()
        cumsum = 0.0
        action = len(probs_flat) - 1
        for i, p in enumerate(probs_flat):
            cumsum += p
            if r <= cumsum:
                action = i
                break
        return action

    def get_value(self, state):
        """Get state value from critic."""
        state_t = Tensor([state])
        v = self.critic.forward(state_t)
        return _flat_values(v)[0]

    def update(self, state, action, reward, next_state, done):
        """Single-step TD actor-critic update. Returns (actor_loss, critic_loss)."""
        # Compute TD error (advantage)
        v = self.get_value(state)
        v_next = 0.0 if done else self.get_value(next_state)
        td_target = reward + self.gamma * v_next
        advantage = td_target - v

        # Update critic
        self.critic.train()
        state_t = Tensor([state])
        v_pred = self.critic.forward(state_t)
        v_target = Tensor([[td_target]])
        critic_loss = self.value_loss_fn.forward(v_pred, v_target)
        critic_grad = self.value_loss_fn.backward(v_pred, v_target)
        self.critic.backward(critic_grad)
        self.critic_optimizer.step(self.critic.get_trainable_layers())

        # Update actor
        self.actor.train()
        probs = self.actor.forward(state_t)
        probs_flat = _flat_values(probs)

        # Policy gradient with advantage
        grad_flat = [0.0] * self.action_size
        for i in range(self.action_size):
            if i == action:
                grad_flat[i] = -advantage * (1.0 - probs_flat[i])
            else:
                grad_flat[i] = -advantage * (-probs_flat[i])

        actor_grad = Tensor([grad_flat])
        self.actor.backward(actor_grad)
        self.actor_optimizer.step(self.actor.get_trainable_layers())

        return abs(advantage), critic_loss

    def get_action_probs(self, state):
        """Get action probabilities from actor."""
        state_t = Tensor([state])
        probs = self.actor.forward(state_t)
        return _flat_values(probs)


# ============================================================
# Proximal Policy Optimization (PPO)
# ============================================================

class PPO:
    """Proximal Policy Optimization with clipped objective.

    Uses actor-critic architecture with clipped surrogate objective.
    """

    def __init__(self, state_size, action_size, hidden_sizes=None,
                 actor_lr=0.0003, critic_lr=0.001, gamma=0.99,
                 clip_epsilon=0.2, n_epochs=4, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.rng = _RNG(seed)

        hidden_sizes = hidden_sizes or [64]

        # Actor network
        self.actor = Sequential()
        prev = state_size
        for h in hidden_sizes:
            self.actor.add(Dense(prev, h, init='he', rng=random.Random(self.rng.randint(0, 99999))))
            self.actor.add(Activation('relu'))
            prev = h
        self.actor.add(Dense(prev, action_size, init='xavier', rng=random.Random(self.rng.randint(0, 99999))))
        self.actor.add(Activation('softmax'))

        # Critic network
        self.critic = Sequential()
        prev = state_size
        for h in hidden_sizes:
            self.critic.add(Dense(prev, h, init='he', rng=random.Random(self.rng.randint(0, 99999))))
            self.critic.add(Activation('relu'))
            prev = h
        self.critic.add(Dense(prev, 1, init='xavier', rng=random.Random(self.rng.randint(0, 99999))))

        self.actor_optimizer = Adam(lr=actor_lr)
        self.critic_optimizer = Adam(lr=critic_lr)
        self.value_loss_fn = MSELoss()

        # Trajectory storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.old_probs = []

    def get_action(self, state):
        """Sample action and store old probability."""
        state_t = Tensor([state])
        probs = self.actor.forward(state_t)
        probs_flat = _flat_values(probs)

        r = self.rng.random()
        cumsum = 0.0
        action = len(probs_flat) - 1
        for i, p in enumerate(probs_flat):
            cumsum += p
            if r <= cumsum:
                action = i
                break

        self.states.append(state)
        self.actions.append(action)
        self.old_probs.append(probs_flat[action])
        return action

    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_returns_and_advantages(self):
        """Compute discounted returns and GAE advantages."""
        returns = []
        G = 0.0
        for i in range(len(self.rewards) - 1, -1, -1):
            if self.dones[i]:
                G = 0.0
            G = self.rewards[i] + self.gamma * G
            returns.insert(0, G)

        # Compute advantages
        advantages = []
        for i, (state, ret) in enumerate(zip(self.states, returns)):
            v = self.get_value(state)
            advantages.append(ret - v)

        # Normalize advantages
        if len(advantages) > 1:
            mean_adv = sum(advantages) / len(advantages)
            std_adv = math.sqrt(sum((a - mean_adv) ** 2 for a in advantages) / len(advantages))
            if std_adv > 1e-8:
                advantages = [(a - mean_adv) / std_adv for a in advantages]

        return returns, advantages

    def get_value(self, state):
        state_t = Tensor([state])
        v = self.critic.forward(state_t)
        return _flat_values(v)[0]

    def update(self):
        """Update policy and value networks. Returns (actor_loss, critic_loss)."""
        if not self.rewards:
            return 0.0, 0.0

        returns, advantages = self.compute_returns_and_advantages()
        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for epoch in range(self.n_epochs):
            for i in range(len(self.states)):
                state = self.states[i]
                action = self.actions[i]
                old_prob = self.old_probs[i]
                advantage = advantages[i]
                ret = returns[i]

                # Actor update with clipped objective
                self.actor.train()
                state_t = Tensor([state])
                probs = self.actor.forward(state_t)
                probs_flat = _flat_values(probs)
                new_prob = max(probs_flat[action], 1e-10)

                ratio = new_prob / max(old_prob, 1e-10)
                clipped_ratio = max(1.0 - self.clip_epsilon,
                                    min(1.0 + self.clip_epsilon, ratio))
                surrogate = min(ratio * advantage, clipped_ratio * advantage)

                # Policy gradient
                grad_flat = [0.0] * self.action_size
                if ratio * advantage <= clipped_ratio * advantage:
                    # Use unclipped gradient
                    for j in range(self.action_size):
                        if j == action:
                            grad_flat[j] = -advantage * (1.0 - probs_flat[j]) / max(old_prob, 1e-10)
                        else:
                            grad_flat[j] = -advantage * (-probs_flat[j]) / max(old_prob, 1e-10)
                # else: clipped -> zero gradient (no update)

                actor_grad = Tensor([grad_flat])
                self.actor.backward(actor_grad)
                self.actor_optimizer.step(self.actor.get_trainable_layers())
                total_actor_loss += abs(surrogate)

                # Critic update
                self.critic.train()
                v_pred = self.critic.forward(state_t)
                v_target = Tensor([[ret]])
                c_loss = self.value_loss_fn.forward(v_pred, v_target)
                c_grad = self.value_loss_fn.backward(v_pred, v_target)
                self.critic.backward(c_grad)
                self.critic_optimizer.step(self.critic.get_trainable_layers())
                total_critic_loss += c_loss

        n = max(len(self.states) * self.n_epochs, 1)
        result = (total_actor_loss / n, total_critic_loss / n)
        self._clear_trajectory()
        return result

    def _clear_trajectory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.old_probs = []

    def get_action_probs(self, state):
        state_t = Tensor([state])
        probs = self.actor.forward(state_t)
        return _flat_values(probs)


# ============================================================
# Training Utilities
# ============================================================

class DQNTrainer:
    """Training loop for DQN agents."""

    def __init__(self, agent, env, buffer=None, policy=None,
                 batch_size=32, target_update_freq=10, seed=42):
        self.agent = agent
        self.env = env
        self.buffer = buffer or ReplayBuffer(capacity=10000, seed=seed)
        self.policy = policy or EpsilonGreedy(epsilon=1.0, min_epsilon=0.01, decay=0.995, seed=seed)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.episode_rewards = []
        self.losses = []

    def train_episode(self):
        """Run one episode, return total reward."""
        state = self.env.reset()
        total_reward = 0.0
        done = False
        episode_loss = 0.0
        steps = 0

        while not done:
            q_values = self.agent.get_q_values(state)
            action = self.policy.select(q_values, self.env.action_size)
            next_state, reward, done, _ = self.env.step(action)

            self.buffer.push(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            steps += 1

            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                loss = self.agent.train_batch(*batch[:5])
                episode_loss += loss
                self.agent.soft_update()

        self.policy.step()
        self.episode_rewards.append(total_reward)
        if steps > 0:
            self.losses.append(episode_loss / steps)
        return total_reward

    def train(self, n_episodes):
        """Train for n episodes, return reward history."""
        for ep in range(n_episodes):
            self.train_episode()
            if (ep + 1) % self.target_update_freq == 0:
                self.agent.hard_update()
        return self.episode_rewards


class PolicyGradientTrainer:
    """Training loop for REINFORCE agents."""

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.episode_rewards = []
        self.losses = []

    def train_episode(self):
        """Run one episode, update policy, return total reward."""
        state = self.env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = self.agent.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.agent.store_reward(reward)
            total_reward += reward
            state = next_state

        loss = self.agent.update()
        self.episode_rewards.append(total_reward)
        self.losses.append(loss)
        return total_reward

    def train(self, n_episodes):
        for _ in range(n_episodes):
            self.train_episode()
        return self.episode_rewards


class ActorCriticTrainer:
    """Training loop for Actor-Critic agents."""

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.episode_rewards = []

    def train_episode(self):
        state = self.env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = self.agent.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

        self.episode_rewards.append(total_reward)
        return total_reward

    def train(self, n_episodes):
        for _ in range(n_episodes):
            self.train_episode()
        return self.episode_rewards


class PPOTrainer:
    """Training loop for PPO agents."""

    def __init__(self, agent, env, steps_per_update=128):
        self.agent = agent
        self.env = env
        self.steps_per_update = steps_per_update
        self.episode_rewards = []

    def train_episode(self):
        """Collect trajectory and update."""
        state = self.env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            action = self.agent.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.agent.store_reward(reward, done)
            total_reward += reward
            state = next_state
            steps += 1

            if steps % self.steps_per_update == 0 and not done:
                self.agent.update()

        self.agent.update()
        self.episode_rewards.append(total_reward)
        return total_reward

    def train(self, n_episodes):
        for _ in range(n_episodes):
            self.train_episode()
        return self.episode_rewards


# ============================================================
# Reward Shaping & Utilities
# ============================================================

class RewardShaper:
    """Potential-based reward shaping."""

    def __init__(self, potential_fn, gamma=0.99):
        self.potential = potential_fn
        self.gamma = gamma

    def shape(self, state, next_state, reward):
        """Apply potential-based shaping: r' = r + gamma*phi(s') - phi(s)."""
        phi_s = self.potential(state)
        phi_s_next = self.potential(next_state)
        return reward + self.gamma * phi_s_next - phi_s


class ReturnTracker:
    """Track and compute statistics on episode returns."""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.returns = []

    def add(self, episode_return):
        self.returns.append(episode_return)

    def mean(self, last_n=None):
        if not self.returns:
            return 0.0
        window = self.returns[-(last_n or self.window_size):]
        return sum(window) / len(window)

    def max(self):
        return max(self.returns) if self.returns else 0.0

    def min(self):
        return min(self.returns) if self.returns else 0.0

    def std(self, last_n=None):
        if not self.returns:
            return 0.0
        window = self.returns[-(last_n or self.window_size):]
        m = sum(window) / len(window)
        return math.sqrt(sum((r - m) ** 2 for r in window) / len(window))

    def is_solved(self, threshold):
        """Check if mean return exceeds threshold."""
        return self.mean() >= threshold


class DiscountedReturn:
    """Utility for computing discounted returns."""

    @staticmethod
    def compute(rewards, gamma=0.99):
        """Compute discounted return for a sequence of rewards."""
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return returns

    @staticmethod
    def compute_gae(rewards, values, next_value, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = []
        gae = 0.0
        values_ext = values + [next_value]
        for t in range(len(rewards) - 1, -1, -1):
            delta = rewards[t] + gamma * values_ext[t + 1] - values_ext[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
        returns = [a + v for a, v in zip(advantages, values)]
        return advantages, returns
