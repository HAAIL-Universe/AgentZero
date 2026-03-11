"""
C146: Reinforcement Learning
New domain -- agents learning from environment rewards.

Components:
- Environments: GridWorld, CartPole, MultiArmedBandit, CliffWalking
- Tabular Methods: Q-Learning, SARSA, Expected SARSA, Double Q-Learning
- Bandits: Epsilon-Greedy, UCB, Thompson Sampling
- Policy Gradient: REINFORCE with baseline
- Deep Q-Network: DQN with experience replay + target network (composes C140)
- Utilities: epsilon schedules, replay buffers, training loops
"""

import math
import random
from collections import defaultdict, deque

# ============================================================
# Environments
# ============================================================

class Environment:
    """Base environment interface."""
    def reset(self):
        raise NotImplementedError
    def step(self, action):
        """Returns (next_state, reward, done, info)."""
        raise NotImplementedError
    def action_space(self):
        raise NotImplementedError
    def state_space(self):
        raise NotImplementedError


class GridWorld(Environment):
    """Grid world with walls, goal, and optional traps."""

    def __init__(self, rows=5, cols=5, start=(0, 0), goal=None, walls=None, traps=None, step_reward=-0.1):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal or (rows - 1, cols - 1)
        self.walls = set(walls or [])
        self.traps = set(traps or [])
        self.step_reward = step_reward
        self.state = None
        # Actions: 0=up, 1=right, 2=down, 3=left
        self._deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        dr, dc = self._deltas[action]
        nr, nc = self.state[0] + dr, self.state[1] + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self.walls:
            self.state = (nr, nc)
        reward = self.step_reward
        done = False
        info = {}
        if self.state == self.goal:
            reward = 1.0
            done = True
            info["reason"] = "goal"
        elif self.state in self.traps:
            reward = -1.0
            done = True
            info["reason"] = "trap"
        return self.state, reward, done, info

    def action_space(self):
        return list(range(4))

    def state_space(self):
        states = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.walls:
                    states.append((r, c))
        return states


class CliffWalking(Environment):
    """Classic cliff walking: agent must navigate along cliff edge."""

    def __init__(self, rows=4, cols=12):
        self.rows = rows
        self.cols = cols
        self.start = (rows - 1, 0)
        self.goal = (rows - 1, cols - 1)
        self.cliff = {(rows - 1, c) for c in range(1, cols - 1)}
        self.state = None
        self._deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        dr, dc = self._deltas[action]
        nr, nc = self.state[0] + dr, self.state[1] + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols:
            self.state = (nr, nc)
        if self.state in self.cliff:
            return self.start, -100.0, False, {"reason": "cliff"}
        if self.state == self.goal:
            return self.state, -1.0, True, {"reason": "goal"}
        return self.state, -1.0, False, {}

    def action_space(self):
        return list(range(4))

    def state_space(self):
        states = []
        for r in range(self.rows):
            for c in range(self.cols):
                states.append((r, c))
        return states


class MultiArmedBandit(Environment):
    """K-armed bandit with Gaussian reward distributions."""

    def __init__(self, k=10, means=None, stds=None, seed=None):
        self.k = k
        rng = random.Random(seed)
        self.means = means or [rng.gauss(0, 1) for _ in range(k)]
        self.stds = stds or [1.0] * k
        self.state = 0  # stateless, single state
        self._rng = rng

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        reward = self._rng.gauss(self.means[action], self.stds[action])
        return self.state, reward, False, {}

    def action_space(self):
        return list(range(self.k))

    def state_space(self):
        return [0]

    def optimal_action(self):
        return max(range(self.k), key=lambda a: self.means[a])


class CartPole(Environment):
    """Simplified CartPole (continuous state, discrete actions)."""

    def __init__(self, gravity=9.8, cart_mass=1.0, pole_mass=0.1, pole_length=0.5,
                 force_mag=10.0, dt=0.02, max_steps=200, seed=None):
        self.gravity = gravity
        self.cart_mass = cart_mass
        self.pole_mass = pole_mass
        self.pole_length = pole_length
        self.force_mag = force_mag
        self.dt = dt
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self.state = None
        self._step_count = 0

    def reset(self):
        self.state = tuple(self._rng.uniform(-0.05, 0.05) for _ in range(4))
        self._step_count = 0
        return self.state

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        total_mass = self.cart_mass + self.pole_mass
        temp = (force + self.pole_mass * self.pole_length * theta_dot ** 2 * sin_t) / total_mass
        theta_acc = (self.gravity * sin_t - cos_t * temp) / (
            self.pole_length * (4.0 / 3.0 - self.pole_mass * cos_t ** 2 / total_mass))
        x_acc = temp - self.pole_mass * self.pole_length * theta_acc * cos_t / total_mass

        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * x_acc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * theta_acc
        self.state = (x, x_dot, theta, theta_dot)
        self._step_count += 1

        done = (abs(x) > 2.4 or abs(theta) > 0.2095 or self._step_count >= self.max_steps)
        reward = 1.0 if not done or self._step_count >= self.max_steps else 0.0
        info = {}
        if done:
            info["steps"] = self._step_count
            if self._step_count >= self.max_steps:
                info["reason"] = "max_steps"
            elif abs(x) > 2.4:
                info["reason"] = "x_out"
            else:
                info["reason"] = "theta_out"
        return self.state, reward, done, info

    def action_space(self):
        return [0, 1]  # left, right

    def state_space(self):
        return None  # continuous

    def discretize(self, state, bins=10):
        """Discretize continuous state for tabular methods."""
        bounds = [(-2.4, 2.4), (-3.0, 3.0), (-0.2095, 0.2095), (-3.0, 3.0)]
        discrete = []
        for val, (lo, hi) in zip(state, bounds):
            val = max(lo, min(hi, val))
            idx = int((val - lo) / (hi - lo) * bins)
            idx = min(idx, bins - 1)
            discrete.append(idx)
        return tuple(discrete)


class FrozenLake(Environment):
    """Frozen lake: slippery surface, holes to avoid."""

    def __init__(self, size=4, holes=None, slippery=True, seed=None):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.holes = set(holes or [(1, 1), (1, 3), (2, 3), (3, 0)])
        self.slippery = slippery
        self._rng = random.Random(seed)
        self._deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        self.state = None

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        if self.slippery and self._rng.random() < 0.333:
            action = self._rng.choice(self.action_space())
        dr, dc = self._deltas[action]
        nr, nc = self.state[0] + dr, self.state[1] + dc
        if 0 <= nr < self.size and 0 <= nc < self.size:
            self.state = (nr, nc)
        if self.state in self.holes:
            return self.state, 0.0, True, {"reason": "hole"}
        if self.state == self.goal:
            return self.state, 1.0, True, {"reason": "goal"}
        return self.state, 0.0, False, {}

    def action_space(self):
        return list(range(4))

    def state_space(self):
        states = []
        for r in range(self.size):
            for c in range(self.size):
                states.append((r, c))
        return states


# ============================================================
# Epsilon Schedules
# ============================================================

class EpsilonSchedule:
    """Base epsilon schedule."""
    def get(self, step):
        raise NotImplementedError


class ConstantEpsilon(EpsilonSchedule):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
    def get(self, step):
        return self.epsilon


class LinearDecayEpsilon(EpsilonSchedule):
    def __init__(self, start=1.0, end=0.01, decay_steps=1000):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
    def get(self, step):
        frac = min(step / self.decay_steps, 1.0)
        return self.start + frac * (self.end - self.start)


class ExponentialDecayEpsilon(EpsilonSchedule):
    def __init__(self, start=1.0, end=0.01, decay_rate=0.995):
        self.start = start
        self.end = end
        self.decay_rate = decay_rate
    def get(self, step):
        return max(self.end, self.start * (self.decay_rate ** step))


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, rng=None):
        rng = rng or random
        return rng.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Proportional prioritized replay."""

    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        max_prio = max(self.priorities) if self.priorities else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, rng=None):
        rng = rng or random
        n = len(self.buffer)
        probs = [p ** self.alpha for p in self.priorities]
        total = sum(probs)
        probs = [p / total for p in probs]

        indices = []
        for _ in range(min(batch_size, n)):
            r = rng.random()
            cumulative = 0
            for i, p in enumerate(probs):
                cumulative += p
                if r <= cumulative:
                    indices.append(i)
                    break
            else:
                indices.append(n - 1)

        self.beta = min(1.0, self.beta + self.beta_increment)
        max_w = (n * min(probs)) ** (-self.beta) if min(probs) > 0 else 1.0
        samples = []
        weights = []
        for i in indices:
            samples.append(self.buffer[i])
            w = (n * probs[i]) ** (-self.beta) / max_w if probs[i] > 0 else 1.0
            weights.append(w)

        return samples, weights, indices

    def update_priorities(self, indices, priorities):
        for i, p in zip(indices, priorities):
            if i < len(self.priorities):
                self.priorities[i] = abs(p) + 1e-6

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Tabular Methods
# ============================================================

class QLearning:
    """Tabular Q-Learning (off-policy TD control)."""

    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon_schedule=None):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_schedule = epsilon_schedule or ConstantEpsilon(0.1)
        self.q = defaultdict(lambda: {a: 0.0 for a in self.actions})
        self.step_count = 0

    def select_action(self, state, rng=None):
        rng = rng or random
        eps = self.epsilon_schedule.get(self.step_count)
        if rng.random() < eps:
            return rng.choice(self.actions)
        return max(self.actions, key=lambda a: self.q[state][a])

    def update(self, state, action, reward, next_state, done):
        best_next = max(self.q[next_state][a] for a in self.actions) if not done else 0.0
        td_error = reward + self.gamma * best_next - self.q[state][action]
        self.q[state][action] += self.alpha * td_error
        self.step_count += 1
        return td_error

    def greedy_action(self, state):
        return max(self.actions, key=lambda a: self.q[state][a])

    def get_value(self, state):
        return max(self.q[state][a] for a in self.actions)

    def get_policy(self):
        return {s: self.greedy_action(s) for s in self.q}


class SARSA:
    """Tabular SARSA (on-policy TD control)."""

    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon_schedule=None):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_schedule = epsilon_schedule or ConstantEpsilon(0.1)
        self.q = defaultdict(lambda: {a: 0.0 for a in self.actions})
        self.step_count = 0

    def select_action(self, state, rng=None):
        rng = rng or random
        eps = self.epsilon_schedule.get(self.step_count)
        if rng.random() < eps:
            return rng.choice(self.actions)
        return max(self.actions, key=lambda a: self.q[state][a])

    def update(self, state, action, reward, next_state, next_action, done):
        next_q = self.q[next_state][next_action] if not done else 0.0
        td_error = reward + self.gamma * next_q - self.q[state][action]
        self.q[state][action] += self.alpha * td_error
        self.step_count += 1
        return td_error

    def greedy_action(self, state):
        return max(self.actions, key=lambda a: self.q[state][a])

    def get_policy(self):
        return {s: self.greedy_action(s) for s in self.q}


class ExpectedSARSA:
    """Expected SARSA -- uses expected Q over policy instead of sampled next action."""

    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon_schedule=None):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_schedule = epsilon_schedule or ConstantEpsilon(0.1)
        self.q = defaultdict(lambda: {a: 0.0 for a in self.actions})
        self.step_count = 0

    def select_action(self, state, rng=None):
        rng = rng or random
        eps = self.epsilon_schedule.get(self.step_count)
        if rng.random() < eps:
            return rng.choice(self.actions)
        return max(self.actions, key=lambda a: self.q[state][a])

    def update(self, state, action, reward, next_state, done):
        if done:
            expected_q = 0.0
        else:
            eps = self.epsilon_schedule.get(self.step_count)
            n_actions = len(self.actions)
            best_a = max(self.actions, key=lambda a: self.q[next_state][a])
            expected_q = 0.0
            for a in self.actions:
                if a == best_a:
                    prob = (1 - eps) + eps / n_actions
                else:
                    prob = eps / n_actions
                expected_q += prob * self.q[next_state][a]
        td_error = reward + self.gamma * expected_q - self.q[state][action]
        self.q[state][action] += self.alpha * td_error
        self.step_count += 1
        return td_error

    def greedy_action(self, state):
        return max(self.actions, key=lambda a: self.q[state][a])


class DoubleQLearning:
    """Double Q-Learning -- reduces overestimation bias."""

    def __init__(self, actions, alpha=0.1, gamma=0.99, epsilon_schedule=None):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_schedule = epsilon_schedule or ConstantEpsilon(0.1)
        self.q1 = defaultdict(lambda: {a: 0.0 for a in self.actions})
        self.q2 = defaultdict(lambda: {a: 0.0 for a in self.actions})
        self.step_count = 0

    def select_action(self, state, rng=None):
        rng = rng or random
        eps = self.epsilon_schedule.get(self.step_count)
        if rng.random() < eps:
            return rng.choice(self.actions)
        combined = {a: self.q1[state][a] + self.q2[state][a] for a in self.actions}
        return max(self.actions, key=lambda a: combined[a])

    def update(self, state, action, reward, next_state, done, rng=None):
        rng = rng or random
        self.step_count += 1
        if rng.random() < 0.5:
            best_a = max(self.actions, key=lambda a: self.q1[next_state][a])
            target = (reward + self.gamma * self.q2[next_state][best_a]) if not done else reward
            td_error = target - self.q1[state][action]
            self.q1[state][action] += self.alpha * td_error
        else:
            best_a = max(self.actions, key=lambda a: self.q2[next_state][a])
            target = (reward + self.gamma * self.q1[next_state][best_a]) if not done else reward
            td_error = target - self.q2[state][action]
            self.q2[state][action] += self.alpha * td_error
        return td_error

    def greedy_action(self, state):
        combined = {a: self.q1[state][a] + self.q2[state][a] for a in self.actions}
        return max(self.actions, key=lambda a: combined[a])


# ============================================================
# N-Step Methods
# ============================================================

class NStepQLearning:
    """N-step Q-Learning for faster propagation."""

    def __init__(self, actions, n=3, alpha=0.1, gamma=0.99, epsilon_schedule=None):
        self.actions = actions
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_schedule = epsilon_schedule or ConstantEpsilon(0.1)
        self.q = defaultdict(lambda: {a: 0.0 for a in self.actions})
        self.step_count = 0
        self._buffer = []

    def select_action(self, state, rng=None):
        rng = rng or random
        eps = self.epsilon_schedule.get(self.step_count)
        if rng.random() < eps:
            return rng.choice(self.actions)
        return max(self.actions, key=lambda a: self.q[state][a])

    def store(self, state, action, reward, next_state, done):
        self._buffer.append((state, action, reward, next_state, done))
        self.step_count += 1

    def update(self):
        """Process buffer and update Q-values."""
        updates = 0
        while len(self._buffer) >= self.n or (self._buffer and self._buffer[-1][4]):
            if len(self._buffer) < self.n and not self._buffer[-1][4]:
                break
            n = min(self.n, len(self._buffer))
            g = 0.0
            for i in range(n - 1, -1, -1):
                s, a, r, ns, d = self._buffer[i]
                if i == n - 1 and not d:
                    g = r + self.gamma * max(self.q[ns][a2] for a2 in self.actions)
                elif d:
                    g = r
                else:
                    g = r + self.gamma * g
            s0, a0 = self._buffer[0][0], self._buffer[0][1]
            td_error = g - self.q[s0][a0]
            self.q[s0][a0] += self.alpha * td_error
            self._buffer.pop(0)
            updates += 1
        return updates

    def flush(self):
        """Flush remaining buffer at episode end."""
        updates = 0
        while self._buffer:
            n = len(self._buffer)
            g = 0.0
            for i in range(n - 1, -1, -1):
                s, a, r, ns, d = self._buffer[i]
                if d:
                    g = r
                else:
                    g = r + self.gamma * g
            s0, a0 = self._buffer[0][0], self._buffer[0][1]
            self.q[s0][a0] += self.alpha * (g - self.q[s0][a0])
            self._buffer.pop(0)
            updates += 1
        return updates


# ============================================================
# Bandit Strategies
# ============================================================

class EpsilonGreedyBandit:
    """Epsilon-greedy multi-armed bandit."""

    def __init__(self, k, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.q_estimates = [0.0] * k
        self.counts = [0] * k

    def select_action(self, rng=None):
        rng = rng or random
        if rng.random() < self.epsilon:
            return rng.randint(0, self.k - 1)
        return max(range(self.k), key=lambda a: self.q_estimates[a])

    def update(self, action, reward):
        self.counts[action] += 1
        n = self.counts[action]
        self.q_estimates[action] += (reward - self.q_estimates[action]) / n


class UCBBandit:
    """Upper Confidence Bound bandit strategy."""

    def __init__(self, k, c=2.0):
        self.k = k
        self.c = c
        self.q_estimates = [0.0] * k
        self.counts = [0] * k
        self.total_count = 0

    def select_action(self, rng=None):
        for a in range(self.k):
            if self.counts[a] == 0:
                return a
        ucb_values = []
        for a in range(self.k):
            bonus = self.c * math.sqrt(math.log(self.total_count) / self.counts[a])
            ucb_values.append(self.q_estimates[a] + bonus)
        return max(range(self.k), key=lambda a: ucb_values[a])

    def update(self, action, reward):
        self.counts[action] += 1
        self.total_count += 1
        n = self.counts[action]
        self.q_estimates[action] += (reward - self.q_estimates[action]) / n


class ThompsonSamplingBandit:
    """Thompson Sampling bandit (Beta distribution for Bernoulli rewards)."""

    def __init__(self, k):
        self.k = k
        self.alpha = [1.0] * k  # successes + 1
        self.beta = [1.0] * k   # failures + 1

    def select_action(self, rng=None):
        rng = rng or random
        samples = [rng.betavariate(self.alpha[a], self.beta[a]) for a in range(self.k)]
        return max(range(self.k), key=lambda a: samples[a])

    def update(self, action, reward):
        """Reward should be 0 or 1 for Bernoulli bandits."""
        if reward > 0:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1


# ============================================================
# Policy Gradient: REINFORCE
# ============================================================

class SoftmaxPolicy:
    """Softmax policy over state-action preferences (tabular)."""

    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.preferences = defaultdict(lambda: {a: 0.0 for a in self.actions})

    def action_probs(self, state):
        h = self.preferences[state]
        max_h = max(h[a] for a in self.actions)
        exp_h = {a: math.exp(h[a] - max_h) for a in self.actions}
        total = sum(exp_h.values())
        return {a: exp_h[a] / total for a in self.actions}

    def select_action(self, state, rng=None):
        rng = rng or random
        probs = self.action_probs(state)
        r = rng.random()
        cumulative = 0
        for a in self.actions:
            cumulative += probs[a]
            if r <= cumulative:
                return a
        return self.actions[-1]


class REINFORCE:
    """REINFORCE (Monte Carlo Policy Gradient) with optional baseline."""

    def __init__(self, states, actions, alpha=0.01, gamma=0.99, use_baseline=True):
        self.policy = SoftmaxPolicy(states, actions)
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.baseline = defaultdict(float)  # state value baseline
        self.baseline_count = defaultdict(int)

    def select_action(self, state, rng=None):
        return self.policy.select_action(state, rng)

    def update_episode(self, episode):
        """Update from a complete episode: list of (state, action, reward)."""
        T = len(episode)
        returns = [0.0] * T
        G = 0.0
        for t in range(T - 1, -1, -1):
            G = episode[t][2] + self.gamma * G
            returns[t] = G

        for t in range(T):
            state, action, _ = episode[t]
            G_t = returns[t]

            if self.use_baseline:
                self.baseline_count[state] += 1
                self.baseline[state] += (G_t - self.baseline[state]) / self.baseline_count[state]
                advantage = G_t - self.baseline[state]
            else:
                advantage = G_t

            probs = self.policy.action_probs(state)
            for a in self.actions:
                if a == action:
                    self.policy.preferences[state][a] += self.alpha * advantage * (1 - probs[a])
                else:
                    self.policy.preferences[state][a] -= self.alpha * advantage * probs[a]

    def get_policy(self):
        result = {}
        for s in self.policy.preferences:
            probs = self.policy.action_probs(s)
            result[s] = max(self.actions, key=lambda a: probs[a])
        return result


# ============================================================
# Monte Carlo Methods
# ============================================================

class MonteCarloControl:
    """First-visit Monte Carlo control with epsilon-greedy."""

    def __init__(self, actions, gamma=0.99, epsilon_schedule=None):
        self.actions = actions
        self.gamma = gamma
        self.epsilon_schedule = epsilon_schedule or ConstantEpsilon(0.1)
        self.q = defaultdict(lambda: {a: 0.0 for a in self.actions})
        self.returns_count = defaultdict(lambda: {a: 0 for a in self.actions})
        self.episode_count = 0

    def select_action(self, state, rng=None):
        rng = rng or random
        eps = self.epsilon_schedule.get(self.episode_count)
        if rng.random() < eps:
            return rng.choice(self.actions)
        return max(self.actions, key=lambda a: self.q[state][a])

    def update_episode(self, episode):
        """Update from complete episode: list of (state, action, reward)."""
        G = 0.0
        visited = set()
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = reward + self.gamma * G
            if (state, action) not in visited:
                visited.add((state, action))
                self.returns_count[state][action] += 1
                n = self.returns_count[state][action]
                self.q[state][action] += (G - self.q[state][action]) / n
        self.episode_count += 1

    def greedy_action(self, state):
        return max(self.actions, key=lambda a: self.q[state][a])


# ============================================================
# Value Iteration & Policy Iteration (Model-Based)
# ============================================================

class ValueIteration:
    """Value iteration for known MDPs."""

    def __init__(self, states, actions, transition_fn, reward_fn, gamma=0.99, theta=1e-6):
        self.states = states
        self.actions = actions
        self.transition_fn = transition_fn  # (s, a) -> [(prob, next_state)]
        self.reward_fn = reward_fn          # (s, a, next_state) -> reward
        self.gamma = gamma
        self.theta = theta
        self.v = {s: 0.0 for s in states}
        self.policy = {s: actions[0] for s in states}

    def solve(self, max_iterations=1000):
        for iteration in range(max_iterations):
            delta = 0
            for s in self.states:
                old_v = self.v[s]
                action_values = []
                for a in self.actions:
                    transitions = self.transition_fn(s, a)
                    val = sum(p * (self.reward_fn(s, a, ns) + self.gamma * self.v[ns])
                              for p, ns in transitions)
                    action_values.append(val)
                self.v[s] = max(action_values)
                delta = max(delta, abs(old_v - self.v[s]))
            if delta < self.theta:
                break

        for s in self.states:
            best_a = self.actions[0]
            best_val = float('-inf')
            for a in self.actions:
                transitions = self.transition_fn(s, a)
                val = sum(p * (self.reward_fn(s, a, ns) + self.gamma * self.v[ns])
                          for p, ns in transitions)
                if val > best_val:
                    best_val = val
                    best_a = a
            self.policy[s] = best_a
        return self.v, self.policy


class PolicyIteration:
    """Policy iteration for known MDPs."""

    def __init__(self, states, actions, transition_fn, reward_fn, gamma=0.99, theta=1e-6):
        self.states = states
        self.actions = actions
        self.transition_fn = transition_fn
        self.reward_fn = reward_fn
        self.gamma = gamma
        self.theta = theta
        self.v = {s: 0.0 for s in states}
        self.policy = {s: actions[0] for s in states}

    def _evaluate_policy(self, max_iterations=1000):
        for _ in range(max_iterations):
            delta = 0
            for s in self.states:
                old_v = self.v[s]
                a = self.policy[s]
                transitions = self.transition_fn(s, a)
                self.v[s] = sum(p * (self.reward_fn(s, a, ns) + self.gamma * self.v[ns])
                                for p, ns in transitions)
                delta = max(delta, abs(old_v - self.v[s]))
            if delta < self.theta:
                break

    def _improve_policy(self):
        stable = True
        for s in self.states:
            old_action = self.policy[s]
            best_a = self.actions[0]
            best_val = float('-inf')
            for a in self.actions:
                transitions = self.transition_fn(s, a)
                val = sum(p * (self.reward_fn(s, a, ns) + self.gamma * self.v[ns])
                          for p, ns in transitions)
                if val > best_val:
                    best_val = val
                    best_a = a
            self.policy[s] = best_a
            if old_action != best_a:
                stable = False
        return stable

    def solve(self, max_iterations=100):
        for _ in range(max_iterations):
            self._evaluate_policy()
            if self._improve_policy():
                break
        return self.v, self.policy


# ============================================================
# Deep Q-Network (using C140 neural net primitives)
# ============================================================

class SimpleNeuralNet:
    """Minimal neural net for DQN (no C140 dependency -- self-contained)."""

    def __init__(self, layer_sizes, lr=0.001, seed=None):
        self.layers = []
        rng = random.Random(seed)
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            scale = math.sqrt(2.0 / fan_in)
            w = [[rng.gauss(0, scale) for _ in range(fan_out)] for _ in range(fan_in)]
            b = [0.0] * fan_out
            self.layers.append((w, b))
        self.lr = lr

    def forward(self, x):
        """Forward pass, returns (output, activations_for_backprop)."""
        activations = [list(x)]
        pre_activations = []
        h = list(x)
        for i, (w, b) in enumerate(self.layers):
            z = [sum(h[j] * w[j][k] for j in range(len(h))) + b[k] for k in range(len(b))]
            pre_activations.append(z)
            if i < len(self.layers) - 1:
                h = [max(0, zi) for zi in z]  # ReLU
            else:
                h = z  # linear output for Q-values
            activations.append(list(h))
        return h, (activations, pre_activations)

    def predict(self, x):
        return self.forward(x)[0]

    def backward(self, cache, target_output):
        """Backprop with MSE loss. Returns loss."""
        activations, pre_activations = cache
        output = activations[-1]
        n_out = len(output)
        loss = sum((output[k] - target_output[k]) ** 2 for k in range(n_out)) / n_out
        grad = [2 * (output[k] - target_output[k]) / n_out for k in range(n_out)]

        for i in range(len(self.layers) - 1, -1, -1):
            w, b = self.layers[i]
            h = activations[i]

            if i < len(self.layers) - 1:
                grad = [grad[k] * (1 if pre_activations[i][k] > 0 else 0) for k in range(len(grad))]

            dw = [[h[j] * grad[k] for k in range(len(grad))] for j in range(len(h))]
            db = list(grad)
            grad = [sum(w[j][k] * grad[k] for k in range(len(grad))) for j in range(len(h))]

            for j in range(len(h)):
                for k in range(len(b)):
                    w[j][k] -= self.lr * dw[j][k]
            for k in range(len(b)):
                b[k] -= self.lr * db[k]

        return loss

    def copy_from(self, other):
        """Copy weights from another network."""
        for i in range(len(self.layers)):
            w, b = self.layers[i]
            ow, ob = other.layers[i]
            for j in range(len(w)):
                for k in range(len(w[j])):
                    w[j][k] = ow[j][k]
            for k in range(len(b)):
                b[k] = ob[k]


class DQN:
    """Deep Q-Network with experience replay and target network."""

    def __init__(self, state_dim, n_actions, hidden_sizes=None, lr=0.001,
                 gamma=0.99, epsilon_schedule=None, buffer_size=10000,
                 batch_size=32, target_update_freq=100, seed=None):
        hidden_sizes = hidden_sizes or [64, 64]
        layer_sizes = [state_dim] + hidden_sizes + [n_actions]
        self.q_net = SimpleNeuralNet(layer_sizes, lr=lr, seed=seed)
        self.target_net = SimpleNeuralNet(layer_sizes, lr=lr, seed=(seed + 1 if seed else None))
        self.target_net.copy_from(self.q_net)
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_schedule = epsilon_schedule or LinearDecayEpsilon(1.0, 0.01, 1000)
        self.buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0
        self._rng = random.Random(seed)

    def select_action(self, state):
        eps = self.epsilon_schedule.get(self.step_count)
        if self._rng.random() < eps:
            return self._rng.randint(0, self.n_actions - 1)
        q_values = self.q_net.predict(state)
        return max(range(self.n_actions), key=lambda a: q_values[a])

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """Sample batch and do one gradient step."""
        if len(self.buffer) < self.batch_size:
            return 0.0
        batch = self.buffer.sample(self.batch_size, self._rng)
        total_loss = 0.0
        for state, action, reward, next_state, done in batch:
            q_values, cache = self.q_net.forward(state)
            target_q = list(q_values)
            if done:
                target_q[action] = reward
            else:
                next_q = self.target_net.predict(next_state)
                target_q[action] = reward + self.gamma * max(next_q)
            loss = self.q_net.backward(cache, target_q)
            total_loss += loss
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.copy_from(self.q_net)
        return total_loss / len(batch)


# ============================================================
# TD(lambda) with Eligibility Traces
# ============================================================

class TDLambda:
    """TD(lambda) with eligibility traces."""

    def __init__(self, actions, alpha=0.1, gamma=0.99, lam=0.9, epsilon_schedule=None):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.epsilon_schedule = epsilon_schedule or ConstantEpsilon(0.1)
        self.q = defaultdict(lambda: {a: 0.0 for a in self.actions})
        self.step_count = 0

    def select_action(self, state, rng=None):
        rng = rng or random
        eps = self.epsilon_schedule.get(self.step_count)
        if rng.random() < eps:
            return rng.choice(self.actions)
        return max(self.actions, key=lambda a: self.q[state][a])

    def run_episode(self, env, max_steps=1000, rng=None):
        """Run one episode with eligibility traces."""
        rng = rng or random
        state = env.reset()
        action = self.select_action(state, rng)
        traces = defaultdict(lambda: defaultdict(float))
        total_reward = 0
        steps = 0

        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            self.step_count += 1

            next_action = self.select_action(next_state, rng)
            best_next = max(self.q[next_state][a] for a in self.actions) if not done else 0.0
            td_error = reward + self.gamma * best_next - self.q[state][action]

            traces[state][action] += 1

            for s in list(traces.keys()):
                for a in list(traces[s].keys()):
                    self.q[s][a] += self.alpha * td_error * traces[s][a]
                    traces[s][a] *= self.gamma * self.lam
                    if traces[s][a] < 1e-10:
                        del traces[s][a]
                if not traces[s]:
                    del traces[s]

            if done:
                break
            state = next_state
            action = next_action

        return total_reward, steps


# ============================================================
# Training Utilities
# ============================================================

def train_tabular(agent, env, episodes=1000, max_steps=500, rng=None):
    """Train a tabular agent (Q-Learning, ExpectedSARSA, DoubleQ) on an environment."""
    rng = rng or random
    rewards = []
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state, rng)
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            if isinstance(agent, SARSA):
                next_action = agent.select_action(next_state, rng)
                agent.update(state, action, reward, next_state, next_action, done)
            elif isinstance(agent, DoubleQLearning):
                agent.update(state, action, reward, next_state, done, rng)
            else:
                agent.update(state, action, reward, next_state, done)

            if done:
                break
            state = next_state
        rewards.append(total_reward)
    return rewards


def train_reinforce(agent, env, episodes=1000, max_steps=500, rng=None):
    """Train REINFORCE agent on an environment."""
    rng = rng or random
    rewards = []
    for ep in range(episodes):
        state = env.reset()
        episode = []
        total_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state, rng)
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))
            total_reward += reward
            if done:
                break
            state = next_state
        agent.update_episode(episode)
        rewards.append(total_reward)
    return rewards


def train_dqn(agent, env, episodes=100, max_steps=200, discretize_fn=None):
    """Train DQN agent. discretize_fn converts state to feature vector."""
    rewards = []
    for ep in range(episodes):
        raw_state = env.reset()
        state = discretize_fn(raw_state) if discretize_fn else raw_state
        total_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            raw_next, reward, done, info = env.step(action)
            next_state = discretize_fn(raw_next) if discretize_fn else raw_next
            agent.store(state, action, reward, next_state, done)
            agent.train_step()
            total_reward += reward
            if done:
                break
            state = next_state
        rewards.append(total_reward)
    return rewards


def train_monte_carlo(agent, env, episodes=1000, max_steps=500, rng=None):
    """Train Monte Carlo control agent."""
    rng = rng or random
    rewards = []
    for ep in range(episodes):
        state = env.reset()
        episode = []
        total_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state, rng)
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))
            total_reward += reward
            if done:
                break
            state = next_state
        agent.update_episode(episode)
        rewards.append(total_reward)
    return rewards


def evaluate_policy(agent, env, episodes=100, max_steps=500):
    """Evaluate a trained agent's greedy policy."""
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps):
            if hasattr(agent, 'greedy_action'):
                action = agent.greedy_action(state)
            elif hasattr(agent, 'get_policy'):
                policy = agent.get_policy()
                action = policy.get(state, agent.actions[0] if hasattr(agent, 'actions') else 0)
            else:
                action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
            state = next_state
        rewards.append(total_reward)
    return sum(rewards) / len(rewards) if rewards else 0.0
