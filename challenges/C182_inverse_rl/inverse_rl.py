"""
C182: Inverse Reinforcement Learning
Composing C179 Reinforcement Learning.

Inverse RL recovers reward functions from expert demonstrations.
Instead of maximizing a given reward, IRL asks: "What reward function
makes the observed behavior optimal?"

Components:
- TabularMDP: Explicit MDP with transition/reward matrices for IRL solvers
- FeatureExtractor: State/action feature mappings (one-hot, coordinate, custom)
- ExpertDemonstrations: Collect/store/manage demonstration trajectories
- ValueIteration: Solve MDPs exactly for policy/value given reward
- FeatureExpectation: Compute expected feature counts under a policy
- MaxEntIRL: Maximum entropy IRL (Ziebart 2008) -- gradient-based reward learning
- ProjectionIRL: Abbeel & Ng projection algorithm -- feature matching via SVM-like steps
- BayesianIRL: Posterior over reward functions via MCMC sampling
- DeepMaxEntIRL: Neural network reward function with MaxEnt objective
- ApprenticeshipLearning: Full pipeline from demos to learned policy
- RewardShaping: Potential-based reward shaping (preserves optimal policy)
- IRLEvaluator: Metrics for comparing learned vs true reward/policy
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C179_reinforcement_learning'))
from reinforcement_learning import (
    _RNG, Environment, GridWorld, QLearning, SARSA,
    Tensor, Dense, Activation, Sequential,
    MSELoss, SGD, Adam,
    relu, sigmoid, softmax, softmax_batch,
    xavier_init, he_init,
    _get_row, _set_row, _stack_rows, _flat_values,
    _scale_tensor, _add_tensors, _sub_tensors,
)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============================================================
# TabularMDP
# ============================================================

class TabularMDP:
    """Explicit tabular MDP with transition and reward matrices.

    T[s][a] = list of (prob, next_state) pairs
    R[s][a] = immediate reward
    """

    def __init__(self, n_states, n_actions, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        # T[s][a] = [(prob, next_state), ...]
        self.T = [[[] for _ in range(n_actions)] for _ in range(n_states)]
        # R[s][a] = reward (can be overridden by IRL)
        self.R = [[0.0] * n_actions for _ in range(n_states)]
        self.terminal_states = set()

    def set_transition(self, s, a, transitions):
        """Set transitions: list of (prob, next_state)."""
        self.T[s][a] = transitions

    def set_reward(self, s, a, r):
        self.R[s][a] = r

    def set_state_reward(self, s, r):
        """Set same reward for all actions in state s."""
        for a in range(self.n_actions):
            self.R[s][a] = r

    def set_terminal(self, s):
        self.terminal_states.add(s)

    def get_transition_prob(self, s, a, s_next):
        """Get P(s'|s,a)."""
        for prob, ns in self.T[s][a]:
            if ns == s_next:
                return prob
        return 0.0

    @classmethod
    def from_grid(cls, rows, cols, goal=None, walls=None, gamma=0.99):
        """Create a GridWorld MDP.

        Actions: 0=up, 1=right, 2=down, 3=left
        Goal gives +1 reward, steps give -0.01.
        """
        walls = set(walls) if walls else set()
        goal = goal or (rows - 1, cols - 1)
        n_states = rows * cols
        n_actions = 4
        mdp = cls(n_states, n_actions, gamma)

        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left

        for r in range(rows):
            for c in range(cols):
                s = r * cols + c
                if (r, c) in walls:
                    # Wall state: all actions stay in place
                    for a in range(n_actions):
                        mdp.set_transition(s, a, [(1.0, s)])
                        mdp.set_reward(s, a, 0.0)
                    continue

                if (r, c) == goal:
                    mdp.set_terminal(s)
                    for a in range(n_actions):
                        mdp.set_transition(s, a, [(1.0, s)])
                        mdp.set_reward(s, a, 0.0)
                    continue

                for a in range(n_actions):
                    dr, dc = deltas[a]
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in walls:
                        ns = nr * cols + nc
                    else:
                        ns = s  # bounce back
                    mdp.set_transition(s, a, [(1.0, ns)])
                    if (nr, nc) == goal and 0 <= nr < rows and 0 <= nc < cols:
                        mdp.set_reward(s, a, 1.0)
                    else:
                        mdp.set_reward(s, a, -0.01)

        return mdp

    @classmethod
    def from_grid_stochastic(cls, rows, cols, goal=None, walls=None,
                              gamma=0.99, slip_prob=0.1):
        """Create stochastic GridWorld -- intended action with (1-slip),
        uniform random other action with slip_prob."""
        walls = set(walls) if walls else set()
        goal = goal or (rows - 1, cols - 1)
        n_states = rows * cols
        n_actions = 4
        mdp = cls(n_states, n_actions, gamma)

        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        def next_state(r, c, a):
            dr, dc = deltas[a]
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in walls:
                return nr * cols + nc
            return r * cols + c

        for r in range(rows):
            for c in range(cols):
                s = r * cols + c
                if (r, c) in walls or (r, c) == goal:
                    for a in range(n_actions):
                        mdp.set_transition(s, a, [(1.0, s)])
                        mdp.set_reward(s, a, 0.0 if (r, c) == goal else 0.0)
                    if (r, c) == goal:
                        mdp.set_terminal(s)
                    continue

                for a in range(n_actions):
                    trans = {}
                    # Intended action
                    ns = next_state(r, c, a)
                    trans[ns] = trans.get(ns, 0.0) + (1.0 - slip_prob)
                    # Slip to other actions
                    for a2 in range(n_actions):
                        if a2 != a:
                            ns2 = next_state(r, c, a2)
                            trans[ns2] = trans.get(ns2, 0.0) + slip_prob / (n_actions - 1)
                    # Convert to list of (prob, state) pairs
                    mdp.set_transition(s, a, [(p, ns_) for ns_, p in trans.items()])

                    # Reward based on where we might end up
                    goal_s = goal[0] * cols + goal[1]
                    mdp.set_reward(s, a, 1.0 if goal_s in trans else -0.01)

        return mdp


# ============================================================
# Value Iteration
# ============================================================

class ValueIteration:
    """Exact MDP solver using value iteration."""

    def __init__(self, mdp, tol=1e-8, max_iters=1000):
        self.mdp = mdp
        self.tol = tol
        self.max_iters = max_iters
        self.V = [0.0] * mdp.n_states
        self.policy = [0] * mdp.n_states
        self.converged = False
        self.iterations = 0

    def solve(self):
        """Run value iteration. Returns (V, policy)."""
        mdp = self.mdp
        V = self.V
        for it in range(self.max_iters):
            delta = 0.0
            for s in range(mdp.n_states):
                if s in mdp.terminal_states:
                    continue
                old_v = V[s]
                best_v = float('-inf')
                for a in range(mdp.n_actions):
                    q = mdp.R[s][a]
                    for prob, ns in mdp.T[s][a]:
                        q += mdp.gamma * prob * V[ns]
                    if q > best_v:
                        best_v = q
                if best_v > float('-inf'):
                    V[s] = best_v
                delta = max(delta, abs(V[s] - old_v))
            if delta < self.tol:
                self.converged = True
                self.iterations = it + 1
                break
        else:
            self.iterations = self.max_iters

        # Extract greedy policy
        for s in range(mdp.n_states):
            if s in mdp.terminal_states:
                continue
            best_a = 0
            best_q = float('-inf')
            for a in range(mdp.n_actions):
                q = mdp.R[s][a]
                for prob, ns in mdp.T[s][a]:
                    q += mdp.gamma * prob * V[ns]
                if q > best_q:
                    best_q = q
                    best_a = a
            self.policy[s] = best_a

        return self.V[:], self.policy[:]

    def get_q_values(self):
        """Return Q[s][a] matrix."""
        mdp = self.mdp
        Q = [[0.0] * mdp.n_actions for _ in range(mdp.n_states)]
        for s in range(mdp.n_states):
            for a in range(mdp.n_actions):
                q = mdp.R[s][a]
                for prob, ns in mdp.T[s][a]:
                    q += mdp.gamma * prob * self.V[ns]
                Q[s][a] = q
        return Q

    def get_stochastic_policy(self, temperature=1.0):
        """Boltzmann (soft) policy from Q-values."""
        Q = self.get_q_values()
        policy = []
        for s in range(self.mdp.n_states):
            q_s = Q[s]
            max_q = max(q_s)
            exps = [math.exp((q - max_q) / max(temperature, 1e-10)) for q in q_s]
            total = sum(exps)
            if total < 1e-15:
                policy.append([1.0 / self.mdp.n_actions] * self.mdp.n_actions)
            else:
                policy.append([e / total for e in exps])
        return policy


# ============================================================
# Feature Extraction
# ============================================================

class FeatureExtractor:
    """Maps states (and optionally actions) to feature vectors."""

    def __init__(self, n_features, feature_fn=None):
        self.n_features = n_features
        self._fn = feature_fn

    def features(self, state, action=None):
        """Return feature vector for (state, action)."""
        if self._fn:
            return self._fn(state, action)
        raise NotImplementedError("No feature function provided")

    @classmethod
    def one_hot(cls, n_states):
        """One-hot state features."""
        def fn(state, action=None):
            f = [0.0] * n_states
            f[state] = 1.0
            return f
        return cls(n_states, fn)

    @classmethod
    def coordinate(cls, rows, cols):
        """Coordinate features: [row/rows, col/cols, dist_to_center, is_edge]."""
        n_features = 4
        def fn(state, action=None):
            r = state // cols
            c = state % cols
            cr, cc = (rows - 1) / 2.0, (cols - 1) / 2.0
            dist = math.sqrt((r - cr) ** 2 + (c - cc) ** 2) / math.sqrt(cr ** 2 + cc ** 2)
            is_edge = 1.0 if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) else 0.0
            return [r / max(rows - 1, 1), c / max(cols - 1, 1), dist, is_edge]
        return cls(n_features, fn)

    @classmethod
    def state_action(cls, n_states, n_actions):
        """One-hot (state, action) features."""
        n_features = n_states * n_actions
        def fn(state, action=None):
            f = [0.0] * n_features
            if action is not None:
                f[state * n_actions + action] = 1.0
            return f
        return cls(n_features, fn)


# ============================================================
# Expert Demonstrations
# ============================================================

class ExpertDemonstrations:
    """Collect and manage expert demonstration trajectories."""

    def __init__(self):
        self.trajectories = []  # list of [(s, a, r, s'), ...]

    def add_trajectory(self, traj):
        """Add a trajectory: list of (state, action, reward, next_state) tuples."""
        self.trajectories.append(traj)

    def collect_from_policy(self, mdp, policy, n_episodes=10, max_steps=100, seed=42):
        """Collect demonstrations by running a deterministic policy on an MDP."""
        rng = _RNG(seed)
        for _ in range(n_episodes):
            traj = []
            s = rng.randint(0, mdp.n_states - 1)
            # Skip terminal/wall starts
            attempts = 0
            while s in mdp.terminal_states and attempts < 50:
                s = rng.randint(0, mdp.n_states - 1)
                attempts += 1

            for step in range(max_steps):
                if s in mdp.terminal_states:
                    break
                a = policy[s]
                # Sample next state from transition
                transitions = mdp.T[s][a]
                r_val = rng.random()
                cumulative = 0.0
                ns = s
                for prob, next_s in transitions:
                    cumulative += prob
                    if r_val <= cumulative:
                        ns = next_s
                        break
                reward = mdp.R[s][a]
                traj.append((s, a, reward, ns))
                s = ns
            if traj:
                self.trajectories.append(traj)

    def collect_from_stochastic_policy(self, mdp, policy_probs, n_episodes=10,
                                        max_steps=100, seed=42):
        """Collect from stochastic policy. policy_probs[s] = [p(a0), p(a1), ...]."""
        rng = _RNG(seed)
        for _ in range(n_episodes):
            traj = []
            s = rng.randint(0, mdp.n_states - 1)
            attempts = 0
            while s in mdp.terminal_states and attempts < 50:
                s = rng.randint(0, mdp.n_states - 1)
                attempts += 1

            for step in range(max_steps):
                if s in mdp.terminal_states:
                    break
                probs = policy_probs[s]
                r_val = rng.random()
                cumulative = 0.0
                a = 0
                for i, p in enumerate(probs):
                    cumulative += p
                    if r_val <= cumulative:
                        a = i
                        break

                transitions = mdp.T[s][a]
                r_val2 = rng.random()
                cumulative = 0.0
                ns = s
                for prob, next_s in transitions:
                    cumulative += prob
                    if r_val2 <= cumulative:
                        ns = next_s
                        break
                reward = mdp.R[s][a]
                traj.append((s, a, reward, ns))
                s = ns
            if traj:
                self.trajectories.append(traj)

    def get_state_visitation(self, n_states):
        """Compute empirical state visitation frequencies."""
        counts = [0.0] * n_states
        total = 0
        for traj in self.trajectories:
            for s, a, r, ns in traj:
                counts[s] += 1.0
                total += 1
        if total > 0:
            counts = [c / total for c in counts]
        return counts

    def get_state_action_visitation(self, n_states, n_actions):
        """Compute empirical (state, action) visitation frequencies."""
        counts = [[0.0] * n_actions for _ in range(n_states)]
        total = 0
        for traj in self.trajectories:
            for s, a, r, ns in traj:
                counts[s][a] += 1.0
                total += 1
        if total > 0:
            for s in range(n_states):
                for a in range(n_actions):
                    counts[s][a] /= total
        return counts

    def get_feature_expectations(self, feature_extractor):
        """Compute average feature expectations from demonstrations."""
        n_f = feature_extractor.n_features
        mu = [0.0] * n_f
        total = 0
        for traj in self.trajectories:
            for s, a, r, ns in traj:
                feats = feature_extractor.features(s, a)
                for i in range(n_f):
                    mu[i] += feats[i]
                total += 1
        if total > 0:
            mu = [m / total for m in mu]
        return mu

    @property
    def n_trajectories(self):
        return len(self.trajectories)

    @property
    def total_steps(self):
        return sum(len(t) for t in self.trajectories)


# ============================================================
# Feature Expectation Computation
# ============================================================

class FeatureExpectation:
    """Compute expected feature counts under a policy in an MDP."""

    @staticmethod
    def from_policy(mdp, policy, feature_extractor, gamma=None, max_iters=200, tol=1e-6):
        """Compute feature expectations using state visitation frequencies.

        Uses the forward algorithm:
        D_0 = initial state distribution (uniform over non-terminal)
        D_{t+1}[s'] = sum_{s,a} D_t[s] * pi(a|s) * T(s'|s,a)
        mu = sum_t gamma^t * sum_s D_t[s] * sum_a pi(a|s) * f(s,a)
        """
        gamma = gamma if gamma is not None else mdp.gamma
        n_s = mdp.n_states
        n_a = mdp.n_actions
        n_f = feature_extractor.n_features

        # Convert policy to stochastic if deterministic
        if isinstance(policy, list) and isinstance(policy[0], (int, float)) and not isinstance(policy[0], list):
            # Deterministic policy
            pi = [[0.0] * n_a for _ in range(n_s)]
            for s in range(n_s):
                a = int(policy[s])
                if 0 <= a < n_a:
                    pi[s][a] = 1.0
        else:
            pi = policy

        # Initial state distribution: uniform over non-terminal
        non_terminal = [s for s in range(n_s) if s not in mdp.terminal_states]
        D = [0.0] * n_s
        if non_terminal:
            p = 1.0 / len(non_terminal)
            for s in non_terminal:
                D[s] = p

        mu = [0.0] * n_f
        for t in range(max_iters):
            discount = gamma ** t
            if discount < tol:
                break

            # Accumulate features
            for s in range(n_s):
                if D[s] < 1e-15:
                    continue
                for a in range(n_a):
                    weight = D[s] * pi[s][a] * discount
                    if weight < 1e-15:
                        continue
                    feats = feature_extractor.features(s, a)
                    for i in range(n_f):
                        mu[i] += weight * feats[i]

            # Next state distribution
            D_next = [0.0] * n_s
            for s in range(n_s):
                if D[s] < 1e-15:
                    continue
                for a in range(n_a):
                    prob_a = pi[s][a]
                    if prob_a < 1e-15:
                        continue
                    for prob_t, ns in mdp.T[s][a]:
                        D_next[ns] += D[s] * prob_a * prob_t
            D = D_next

        return mu

    @staticmethod
    def from_demonstrations(demos, feature_extractor, gamma=0.99):
        """Compute discounted feature expectations from demonstration trajectories."""
        n_f = feature_extractor.n_features
        mu = [0.0] * n_f
        n_traj = len(demos.trajectories)
        if n_traj == 0:
            return mu
        for traj in demos.trajectories:
            for t, (s, a, r, ns) in enumerate(traj):
                discount = gamma ** t
                feats = feature_extractor.features(s, a)
                for i in range(n_f):
                    mu[i] += discount * feats[i]
        mu = [m / n_traj for m in mu]
        return mu


# ============================================================
# Maximum Entropy IRL (Ziebart 2008)
# ============================================================

class MaxEntIRL:
    """Maximum Entropy Inverse Reinforcement Learning.

    Learns reward weights theta such that R(s) = theta . f(s).
    Maximizes likelihood of demonstrations under maximum entropy model.
    """

    def __init__(self, mdp, feature_extractor, lr=0.1, n_iters=50,
                 temperature=1.0, seed=42):
        self.mdp = mdp
        self.fe = feature_extractor
        self.lr = lr
        self.n_iters = n_iters
        self.temperature = temperature
        self.rng = _RNG(seed)
        self.theta = [self.rng.uniform(-0.1, 0.1) for _ in range(feature_extractor.n_features)]
        self.history = []  # gradient norms per iteration

    def _set_rewards_from_theta(self):
        """Update MDP rewards: R(s,a) = theta . f(s,a)."""
        for s in range(self.mdp.n_states):
            for a in range(self.mdp.n_actions):
                feats = self.fe.features(s, a)
                r = sum(t * f for t, f in zip(self.theta, feats))
                self.mdp.R[s][a] = r

    def _compute_state_visitation(self, policy_probs, max_iters=200, tol=1e-6):
        """Forward pass: compute expected state visitation frequencies under policy."""
        mdp = self.mdp
        n_s = mdp.n_states

        # Uniform initial distribution over non-terminal states
        non_terminal = [s for s in range(n_s) if s not in mdp.terminal_states]
        D = [0.0] * n_s
        if non_terminal:
            p = 1.0 / len(non_terminal)
            for s in non_terminal:
                D[s] = p

        svf = [0.0] * n_s
        for t in range(max_iters):
            discount = mdp.gamma ** t
            if discount < tol:
                break
            for s in range(n_s):
                svf[s] += D[s] * discount

            D_next = [0.0] * n_s
            for s in range(n_s):
                if D[s] < 1e-15:
                    continue
                for a in range(mdp.n_actions):
                    p_a = policy_probs[s][a]
                    if p_a < 1e-15:
                        continue
                    for prob, ns in mdp.T[s][a]:
                        D_next[ns] += D[s] * p_a * prob
            D = D_next

        return svf

    def train(self, demonstrations):
        """Train MaxEnt IRL on expert demonstrations.

        Returns learned theta (reward weights).
        """
        # Expert feature expectations
        mu_expert = demonstrations.get_feature_expectations(self.fe)

        for iteration in range(self.n_iters):
            # Set rewards from current theta
            self._set_rewards_from_theta()

            # Solve MDP under current reward
            vi = ValueIteration(self.mdp)
            vi.solve()

            # Get soft policy
            policy_probs = vi.get_stochastic_policy(temperature=self.temperature)

            # Compute expected SVF under learned policy
            svf = self._compute_state_visitation(policy_probs)

            # Learner feature expectations
            mu_learner = [0.0] * self.fe.n_features
            for s in range(self.mdp.n_states):
                if svf[s] < 1e-15:
                    continue
                for a in range(self.mdp.n_actions):
                    weight = svf[s] * policy_probs[s][a]
                    if weight < 1e-15:
                        continue
                    feats = self.fe.features(s, a)
                    for i in range(self.fe.n_features):
                        mu_learner[i] += weight * feats[i]

            # Gradient = mu_expert - mu_learner
            grad = [mu_expert[i] - mu_learner[i] for i in range(self.fe.n_features)]
            grad_norm = math.sqrt(sum(g * g for g in grad))
            self.history.append(grad_norm)

            # Update theta
            for i in range(len(self.theta)):
                self.theta[i] += self.lr * grad[i]

        return self.theta[:]

    def get_reward_function(self):
        """Return learned reward as list: R[s] = theta . f(s)."""
        rewards = []
        for s in range(self.mdp.n_states):
            feats = self.fe.features(s)
            r = sum(t * f for t, f in zip(self.theta, feats))
            rewards.append(r)
        return rewards

    def get_policy(self):
        """Solve MDP with learned rewards and return policy."""
        self._set_rewards_from_theta()
        vi = ValueIteration(self.mdp)
        vi.solve()
        return vi.policy[:]


# ============================================================
# Projection IRL (Abbeel & Ng 2004)
# ============================================================

class ProjectionIRL:
    """Projection-based IRL (Abbeel & Ng apprenticeship learning).

    Iteratively finds reward weights by projecting expert feature expectations
    onto the set of achievable feature expectations.
    """

    def __init__(self, mdp, feature_extractor, n_iters=20, tol=0.01, seed=42):
        self.mdp = mdp
        self.fe = feature_extractor
        self.n_iters = n_iters
        self.tol = tol
        self.rng = _RNG(seed)
        self.theta = None
        self.history = []  # margins per iteration
        self.policies = []  # policies found at each iteration

    def train(self, demonstrations):
        """Train Projection IRL.

        Returns learned theta (reward weights).
        """
        n_f = self.fe.n_features
        gamma = self.mdp.gamma

        # Expert feature expectations
        mu_expert = FeatureExpectation.from_demonstrations(demonstrations, self.fe, gamma)

        # Initialize with random policy
        policy = [self.rng.randint(0, self.mdp.n_actions - 1) for _ in range(self.mdp.n_states)]
        mu_pi = FeatureExpectation.from_policy(self.mdp, policy, self.fe, gamma)
        self.policies.append(policy[:])

        mu_bar = mu_pi[:]  # running best approximation

        for iteration in range(self.n_iters):
            # theta = mu_expert - mu_bar (direction of maximum violation)
            theta = [mu_expert[i] - mu_bar[i] for i in range(n_f)]
            norm = math.sqrt(sum(t * t for t in theta))

            if norm < self.tol:
                self.history.append(norm)
                break

            # Normalize
            if norm > 1e-10:
                theta = [t / norm for t in theta]

            self.history.append(norm)

            # Set rewards from theta: R(s,a) = theta . f(s,a)
            for s in range(self.mdp.n_states):
                for a in range(self.mdp.n_actions):
                    feats = self.fe.features(s, a)
                    r = sum(t * f for t, f in zip(theta, feats))
                    self.mdp.R[s][a] = r

            # Solve MDP
            vi = ValueIteration(self.mdp)
            vi.solve()
            policy = vi.policy[:]
            self.policies.append(policy[:])

            # Compute feature expectations of new policy
            mu_pi = FeatureExpectation.from_policy(self.mdp, policy, self.fe, gamma)

            # Project: mu_bar = mu_bar + alpha * (mu_pi - mu_bar)
            # where alpha = (mu_expert - mu_bar) . (mu_pi - mu_bar) / ||mu_pi - mu_bar||^2
            diff = [mu_pi[i] - mu_bar[i] for i in range(n_f)]
            diff_norm_sq = sum(d * d for d in diff)
            if diff_norm_sq < 1e-15:
                break
            expert_diff = [mu_expert[i] - mu_bar[i] for i in range(n_f)]
            alpha = sum(e * d for e, d in zip(expert_diff, diff)) / diff_norm_sq
            alpha = max(0.0, min(1.0, alpha))

            mu_bar = [mu_bar[i] + alpha * diff[i] for i in range(n_f)]

        self.theta = theta if 'theta' in dir() else [0.0] * n_f
        return self.theta[:]

    def get_best_policy(self):
        """Return the last (best) policy found."""
        if self.policies:
            return self.policies[-1]
        return [0] * self.mdp.n_states


# ============================================================
# Bayesian IRL
# ============================================================

class BayesianIRL:
    """Bayesian Inverse Reinforcement Learning via MCMC.

    Places a prior over reward weights and uses MCMC to sample from
    the posterior P(theta | demonstrations).
    """

    def __init__(self, mdp, feature_extractor, n_samples=200, burn_in=50,
                 step_size=0.2, beta=1.0, prior_std=1.0, seed=42):
        self.mdp = mdp
        self.fe = feature_extractor
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.step_size = step_size
        self.beta = beta  # confidence in expert optimality
        self.prior_std = prior_std
        self.rng = _RNG(seed)
        self.samples = []
        self.acceptance_rate = 0.0

    def _log_prior(self, theta):
        """Gaussian prior on theta."""
        return -0.5 * sum(t * t / (self.prior_std ** 2) for t in theta)

    def _log_likelihood(self, theta, demonstrations):
        """Log likelihood of demonstrations under MaxEnt model with reward theta."""
        # Set rewards
        for s in range(self.mdp.n_states):
            for a in range(self.mdp.n_actions):
                feats = self.fe.features(s, a)
                r = sum(t * f for t, f in zip(theta, feats))
                self.mdp.R[s][a] = r

        # Solve for Q-values
        vi = ValueIteration(self.mdp)
        vi.solve()
        Q = vi.get_q_values()

        # Log-likelihood: sum over demos of log P(a|s)
        # P(a|s) = exp(beta * Q(s,a)) / sum_a' exp(beta * Q(s,a'))
        log_lik = 0.0
        for traj in demonstrations.trajectories:
            for s, a, r, ns in traj:
                q_s = Q[s]
                max_q = max(q_s)
                log_sum = max_q + math.log(sum(math.exp(self.beta * (q - max_q))
                                               for q in q_s) + 1e-15)
                log_lik += self.beta * q_s[a] - log_sum
        return log_lik

    def train(self, demonstrations):
        """Run MCMC to sample from posterior over reward weights."""
        n_f = self.fe.n_features
        theta = [self.rng.uniform(-0.5, 0.5) for _ in range(n_f)]
        log_post = self._log_prior(theta) + self._log_likelihood(theta, demonstrations)

        accepted = 0
        total = 0

        for i in range(self.n_samples + self.burn_in):
            # Propose new theta
            theta_new = [t + self.rng.normal(0, self.step_size) for t in theta]
            log_post_new = self._log_prior(theta_new) + self._log_likelihood(theta_new, demonstrations)

            # Metropolis-Hastings acceptance
            log_alpha = log_post_new - log_post
            if math.log(max(self.rng.random(), 1e-15)) < log_alpha:
                theta = theta_new
                log_post = log_post_new
                accepted += 1
            total += 1

            if i >= self.burn_in:
                self.samples.append(theta[:])

        self.acceptance_rate = accepted / max(total, 1)
        return self.get_mean_reward_weights()

    def get_mean_reward_weights(self):
        """Return posterior mean of theta."""
        if not self.samples:
            return [0.0] * self.fe.n_features
        n_f = len(self.samples[0])
        mean = [0.0] * n_f
        for sample in self.samples:
            for i in range(n_f):
                mean[i] += sample[i]
        mean = [m / len(self.samples) for m in mean]
        return mean

    def get_reward_std(self):
        """Return posterior std of theta."""
        if not self.samples:
            return [0.0] * self.fe.n_features
        mean = self.get_mean_reward_weights()
        n_f = len(mean)
        var = [0.0] * n_f
        for sample in self.samples:
            for i in range(n_f):
                var[i] += (sample[i] - mean[i]) ** 2
        return [math.sqrt(v / len(self.samples)) for v in var]

    def get_policy(self):
        """Get policy from posterior mean reward."""
        theta = self.get_mean_reward_weights()
        for s in range(self.mdp.n_states):
            for a in range(self.mdp.n_actions):
                feats = self.fe.features(s, a)
                r = sum(t * f for t, f in zip(theta, feats))
                self.mdp.R[s][a] = r
        vi = ValueIteration(self.mdp)
        vi.solve()
        return vi.policy[:]


# ============================================================
# Deep MaxEnt IRL (Neural Network Reward)
# ============================================================

class DeepMaxEntIRL:
    """MaxEnt IRL with a neural network reward function.

    Instead of linear R(s) = theta.f(s), uses R(s) = net(f(s)).
    """

    def __init__(self, mdp, feature_extractor, hidden_sizes=None,
                 lr=0.01, n_iters=30, temperature=1.0, seed=42):
        self.mdp = mdp
        self.fe = feature_extractor
        self.lr = lr
        self.n_iters = n_iters
        self.temperature = temperature
        self.rng = _RNG(seed)
        random.seed(seed)

        # Build reward network
        hidden_sizes = hidden_sizes or [16]
        layers = []
        in_size = feature_extractor.n_features
        for h in hidden_sizes:
            layers.append(Dense(in_size, h, init='xavier'))
            layers.append(Activation('relu'))
            in_size = h
        layers.append(Dense(in_size, 1, init='xavier'))
        self.reward_net = Sequential(layers)
        self.history = []

    def _get_reward(self, state):
        """Compute reward for a single state."""
        feats = self.fe.features(state)
        inp = Tensor([feats])
        out = self.reward_net.forward(inp)
        return out.data[0][0] if len(out.shape) > 1 else out.data[0]

    def _set_rewards(self):
        """Set MDP rewards from neural network."""
        for s in range(self.mdp.n_states):
            r = self._get_reward(s)
            for a in range(self.mdp.n_actions):
                self.mdp.R[s][a] = r

    def _compute_svf(self, policy_probs, max_iters=200, tol=1e-6):
        """Compute state visitation frequencies."""
        mdp = self.mdp
        n_s = mdp.n_states
        non_terminal = [s for s in range(n_s) if s not in mdp.terminal_states]
        D = [0.0] * n_s
        if non_terminal:
            p = 1.0 / len(non_terminal)
            for s in non_terminal:
                D[s] = p

        svf = [0.0] * n_s
        for t in range(max_iters):
            discount = mdp.gamma ** t
            if discount < tol:
                break
            for s in range(n_s):
                svf[s] += D[s] * discount
            D_next = [0.0] * n_s
            for s in range(n_s):
                if D[s] < 1e-15:
                    continue
                for a in range(mdp.n_actions):
                    p_a = policy_probs[s][a]
                    if p_a < 1e-15:
                        continue
                    for prob, ns in mdp.T[s][a]:
                        D_next[ns] += D[s] * p_a * prob
            D = D_next
        return svf

    def train(self, demonstrations):
        """Train Deep MaxEnt IRL."""
        # Expert state visitation
        expert_svf = demonstrations.get_state_visitation(self.mdp.n_states)

        for iteration in range(self.n_iters):
            self._set_rewards()

            vi = ValueIteration(self.mdp)
            vi.solve()
            policy_probs = vi.get_stochastic_policy(temperature=self.temperature)

            learner_svf = self._compute_svf(policy_probs)

            # Gradient for reward network: expert_svf - learner_svf
            # Manual gradient update through the network
            grad_reward = [expert_svf[s] - learner_svf[s] for s in range(self.mdp.n_states)]
            grad_norm = math.sqrt(sum(g * g for g in grad_reward))
            self.history.append(grad_norm)

            # Update network weights via simple gradient step
            # For each state, propagate the gradient signal
            trainable = self.reward_net.get_trainable_layers()

            # Accumulate weight gradients
            weight_grads = {}
            for layer in trainable:
                w = layer.weights
                b = layer.bias
                w_rows = len(w.data) if len(w.shape) > 1 else 1
                w_cols = len(w.data[0]) if len(w.shape) > 1 else len(w.data)
                b_len = len(b.data[0]) if len(b.shape) > 1 else len(b.data)
                weight_grads[id(layer)] = {
                    'W': [[0.0] * w_cols for _ in range(w_rows)],
                    'b': [0.0] * b_len
                }

            for s in range(self.mdp.n_states):
                if abs(grad_reward[s]) < 1e-15:
                    continue
                feats = self.fe.features(s)
                inp = Tensor([feats])
                out = self.reward_net.forward(inp)

                # Use loss gradient = -grad_reward[s] (we want to maximize)
                grad_out = Tensor([[-grad_reward[s]]])
                self.reward_net.backward(grad_out)

                # Accumulate
                for layer in trainable:
                    gw = layer.grad_weights
                    gb = layer.grad_bias
                    if gw is not None:
                        gw_data = gw.data
                        wg = weight_grads[id(layer)]['W']
                        if len(gw.shape) > 1:
                            for i in range(len(gw_data)):
                                for j in range(len(gw_data[i])):
                                    wg[i][j] += gw_data[i][j]
                        else:
                            for j in range(len(gw_data)):
                                wg[0][j] += gw_data[j]
                    if gb is not None:
                        gb_data = gb.data
                        bg = weight_grads[id(layer)]['b']
                        if len(gb.shape) > 1:
                            for j in range(len(gb_data[0])):
                                bg[j] += gb_data[0][j]
                        else:
                            for j in range(len(gb_data)):
                                bg[j] += gb_data[j]

            # Apply accumulated gradients
            for layer in trainable:
                g = weight_grads[id(layer)]
                w = layer.weights
                b = layer.bias
                if len(w.shape) > 1:
                    for i in range(len(w.data)):
                        for j in range(len(w.data[i])):
                            w.data[i][j] -= self.lr * g['W'][i][j]
                else:
                    for j in range(len(w.data)):
                        w.data[j] -= self.lr * g['W'][0][j]
                if len(b.shape) > 1:
                    for j in range(len(b.data[0])):
                        b.data[0][j] -= self.lr * g['b'][j]
                else:
                    for j in range(len(b.data)):
                        b.data[j] -= self.lr * g['b'][j]

    def get_reward_function(self):
        """Return learned rewards as list."""
        return [self._get_reward(s) for s in range(self.mdp.n_states)]

    def get_policy(self):
        """Get policy from learned reward."""
        self._set_rewards()
        vi = ValueIteration(self.mdp)
        vi.solve()
        return vi.policy[:]


# ============================================================
# Apprenticeship Learning (Full Pipeline)
# ============================================================

class ApprenticeshipLearning:
    """Full apprenticeship learning pipeline.

    Given an MDP and expert demonstrations:
    1. Extract features
    2. Run IRL to learn reward
    3. Solve MDP with learned reward
    4. Evaluate learned policy
    """

    def __init__(self, mdp, feature_extractor, method='maxent', seed=42, **kwargs):
        self.mdp = mdp
        self.fe = feature_extractor
        self.method = method
        self.seed = seed
        self.kwargs = kwargs
        self.irl_model = None
        self.learned_theta = None
        self.learned_policy = None
        self.learned_reward = None

    def learn(self, demonstrations):
        """Run the full pipeline."""
        if self.method == 'maxent':
            self.irl_model = MaxEntIRL(
                self.mdp, self.fe,
                lr=self.kwargs.get('lr', 0.1),
                n_iters=self.kwargs.get('n_iters', 50),
                temperature=self.kwargs.get('temperature', 1.0),
                seed=self.seed
            )
            self.learned_theta = self.irl_model.train(demonstrations)
            self.learned_policy = self.irl_model.get_policy()
            self.learned_reward = self.irl_model.get_reward_function()

        elif self.method == 'projection':
            self.irl_model = ProjectionIRL(
                self.mdp, self.fe,
                n_iters=self.kwargs.get('n_iters', 20),
                tol=self.kwargs.get('tol', 0.01),
                seed=self.seed
            )
            self.learned_theta = self.irl_model.train(demonstrations)
            self.learned_policy = self.irl_model.get_best_policy()
            # Compute reward from theta
            self.learned_reward = []
            for s in range(self.mdp.n_states):
                feats = self.fe.features(s)
                r = sum(t * f for t, f in zip(self.learned_theta, feats))
                self.learned_reward.append(r)

        elif self.method == 'bayesian':
            self.irl_model = BayesianIRL(
                self.mdp, self.fe,
                n_samples=self.kwargs.get('n_samples', 200),
                burn_in=self.kwargs.get('burn_in', 50),
                step_size=self.kwargs.get('step_size', 0.2),
                beta=self.kwargs.get('beta', 1.0),
                seed=self.seed
            )
            self.learned_theta = self.irl_model.train(demonstrations)
            self.learned_policy = self.irl_model.get_policy()
            self.learned_reward = []
            for s in range(self.mdp.n_states):
                feats = self.fe.features(s)
                r = sum(t * f for t, f in zip(self.learned_theta, feats))
                self.learned_reward.append(r)

        elif self.method == 'deep_maxent':
            self.irl_model = DeepMaxEntIRL(
                self.mdp, self.fe,
                hidden_sizes=self.kwargs.get('hidden_sizes', [16]),
                lr=self.kwargs.get('lr', 0.01),
                n_iters=self.kwargs.get('n_iters', 30),
                temperature=self.kwargs.get('temperature', 1.0),
                seed=self.seed
            )
            self.irl_model.train(demonstrations)
            self.learned_policy = self.irl_model.get_policy()
            self.learned_reward = self.irl_model.get_reward_function()

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self.learned_policy

    def evaluate(self, true_policy=None, true_reward=None):
        """Evaluate learned policy/reward against ground truth."""
        results = {}

        if true_policy is not None and self.learned_policy is not None:
            match = sum(1 for s in range(self.mdp.n_states)
                       if self.learned_policy[s] == true_policy[s]
                       and s not in self.mdp.terminal_states)
            non_term = sum(1 for s in range(self.mdp.n_states)
                          if s not in self.mdp.terminal_states)
            results['policy_accuracy'] = match / max(non_term, 1)

        if true_reward is not None and self.learned_reward is not None:
            # Correlation between true and learned reward
            results['reward_correlation'] = _pearson_correlation(
                true_reward, self.learned_reward)
            # RMSE
            mse = sum((t - l) ** 2 for t, l in zip(true_reward, self.learned_reward)) / len(true_reward)
            results['reward_rmse'] = math.sqrt(mse)

        return results


# ============================================================
# Reward Shaping
# ============================================================

class RewardShaping:
    """Potential-based reward shaping (Ng et al. 1999).

    F(s, s') = gamma * phi(s') - phi(s)
    where phi is the potential function.

    Preserves optimal policy while potentially accelerating learning.
    """

    def __init__(self, mdp, potential_fn=None):
        self.mdp = mdp
        self.phi = potential_fn or (lambda s: 0.0)

    def set_potential(self, potential_fn):
        self.phi = potential_fn

    def set_potential_from_values(self, values):
        """Use value function as potential."""
        self.phi = lambda s: values[s]

    def shaped_reward(self, s, a, s_next):
        """Compute shaped reward: R(s,a) + gamma * phi(s') - phi(s)."""
        base_reward = self.mdp.R[s][a]
        shaping = self.mdp.gamma * self.phi(s_next) - self.phi(s)
        return base_reward + shaping

    def apply_to_mdp(self):
        """Create a new MDP with shaped rewards."""
        new_mdp = TabularMDP(self.mdp.n_states, self.mdp.n_actions, self.mdp.gamma)
        new_mdp.terminal_states = set(self.mdp.terminal_states)

        for s in range(self.mdp.n_states):
            for a in range(self.mdp.n_actions):
                new_mdp.T[s][a] = self.mdp.T[s][a][:]
                # Average shaped reward over transitions
                shaped_r = 0.0
                for prob, ns in self.mdp.T[s][a]:
                    shaped_r += prob * self.shaped_reward(s, a, ns)
                new_mdp.R[s][a] = shaped_r

        return new_mdp


# ============================================================
# IRL Evaluator
# ============================================================

class IRLEvaluator:
    """Comprehensive evaluation of IRL results."""

    def __init__(self, mdp, feature_extractor):
        self.mdp = mdp
        self.fe = feature_extractor

    def policy_match(self, policy_a, policy_b):
        """Fraction of states where two policies agree (excluding terminal)."""
        match = 0
        total = 0
        for s in range(self.mdp.n_states):
            if s in self.mdp.terminal_states:
                continue
            total += 1
            if policy_a[s] == policy_b[s]:
                match += 1
        return match / max(total, 1)

    def expected_return(self, policy, n_episodes=100, max_steps=100, seed=42):
        """Estimate expected return of a policy via rollouts."""
        rng = _RNG(seed)
        total_return = 0.0
        for ep in range(n_episodes):
            # Random non-terminal start
            non_term = [s for s in range(self.mdp.n_states)
                       if s not in self.mdp.terminal_states]
            if not non_term:
                break
            s = non_term[rng.randint(0, len(non_term) - 1)]
            episode_return = 0.0
            discount = 1.0
            for step in range(max_steps):
                if s in self.mdp.terminal_states:
                    break
                a = policy[s]
                r = self.mdp.R[s][a]
                episode_return += discount * r
                discount *= self.mdp.gamma

                # Sample transition
                transitions = self.mdp.T[s][a]
                r_val = rng.random()
                cumulative = 0.0
                ns = s
                for prob, next_s in transitions:
                    cumulative += prob
                    if r_val <= cumulative:
                        ns = next_s
                        break
                s = ns
            total_return += episode_return
        return total_return / max(n_episodes, 1)

    def feature_expectation_error(self, expert_demos, learned_policy, gamma=None):
        """L2 error between expert and learned feature expectations."""
        gamma = gamma if gamma is not None else self.mdp.gamma
        mu_expert = FeatureExpectation.from_demonstrations(expert_demos, self.fe, gamma)
        mu_learned = FeatureExpectation.from_policy(self.mdp, learned_policy, self.fe, gamma)
        error = math.sqrt(sum((e - l) ** 2 for e, l in zip(mu_expert, mu_learned)))
        return error

    def reward_correlation(self, true_reward, learned_reward):
        """Pearson correlation between true and learned reward."""
        return _pearson_correlation(true_reward, learned_reward)

    def value_difference(self, true_policy, learned_policy, n_episodes=100, seed=42):
        """Difference in expected return between true and learned policy."""
        v_true = self.expected_return(true_policy, n_episodes=n_episodes, seed=seed)
        v_learned = self.expected_return(learned_policy, n_episodes=n_episodes, seed=seed)
        return v_true - v_learned

    def comprehensive_eval(self, true_policy, learned_policy, true_reward=None,
                           learned_reward=None, demos=None):
        """Full evaluation report."""
        results = {
            'policy_match': self.policy_match(true_policy, learned_policy),
        }
        if true_reward is not None and learned_reward is not None:
            results['reward_correlation'] = _pearson_correlation(true_reward, learned_reward)
        if demos is not None:
            results['feature_error'] = self.feature_expectation_error(demos, learned_policy)
        results['value_diff'] = self.value_difference(true_policy, learned_policy)
        return results


# ============================================================
# MaxCausalEnt IRL (Soft Value Iteration variant)
# ============================================================

class MaxCausalEntIRL:
    """Maximum Causal Entropy IRL with soft value iteration.

    Uses soft Bellman equations for exact gradient computation.
    More principled than the basic MaxEnt variant.
    """

    def __init__(self, mdp, feature_extractor, lr=0.1, n_iters=50, seed=42):
        self.mdp = mdp
        self.fe = feature_extractor
        self.lr = lr
        self.n_iters = n_iters
        self.rng = _RNG(seed)
        self.theta = [self.rng.uniform(-0.1, 0.1) for _ in range(feature_extractor.n_features)]
        self.history = []

    def _soft_value_iteration(self, tol=1e-8, max_iters=500):
        """Soft value iteration: V(s) = logsumexp_a [R(s,a) + gamma * sum_s' T(s'|s,a) V(s')]."""
        mdp = self.mdp
        V = [0.0] * mdp.n_states
        for it in range(max_iters):
            delta = 0.0
            for s in range(mdp.n_states):
                if s in mdp.terminal_states:
                    continue
                q_vals = []
                for a in range(mdp.n_actions):
                    q = mdp.R[s][a]
                    for prob, ns in mdp.T[s][a]:
                        q += mdp.gamma * prob * V[ns]
                    q_vals.append(q)
                # logsumexp
                max_q = max(q_vals)
                new_v = max_q + math.log(sum(math.exp(q - max_q) for q in q_vals))
                delta = max(delta, abs(new_v - V[s]))
                V[s] = new_v
            if delta < tol:
                break
        return V

    def _get_soft_policy(self, V):
        """Get soft (Boltzmann) policy from soft values."""
        mdp = self.mdp
        policy = []
        for s in range(mdp.n_states):
            q_vals = []
            for a in range(mdp.n_actions):
                q = mdp.R[s][a]
                for prob, ns in mdp.T[s][a]:
                    q += mdp.gamma * prob * V[ns]
                q_vals.append(q)
            max_q = max(q_vals)
            exps = [math.exp(q - max_q) for q in q_vals]
            total = sum(exps)
            if total < 1e-15:
                policy.append([1.0 / mdp.n_actions] * mdp.n_actions)
            else:
                policy.append([e / total for e in exps])
        return policy

    def _compute_svf(self, policy, max_t=200, tol=1e-6):
        mdp = self.mdp
        n_s = mdp.n_states
        non_term = [s for s in range(n_s) if s not in mdp.terminal_states]
        D = [0.0] * n_s
        if non_term:
            p = 1.0 / len(non_term)
            for s in non_term:
                D[s] = p
        svf = [0.0] * n_s
        for t in range(max_t):
            discount = mdp.gamma ** t
            if discount < tol:
                break
            for s in range(n_s):
                svf[s] += D[s] * discount
            D_next = [0.0] * n_s
            for s in range(n_s):
                if D[s] < 1e-15:
                    continue
                for a in range(mdp.n_actions):
                    p_a = policy[s][a]
                    if p_a < 1e-15:
                        continue
                    for prob, ns in mdp.T[s][a]:
                        D_next[ns] += D[s] * p_a * prob
            D = D_next
        return svf

    def train(self, demonstrations):
        """Train MaxCausalEnt IRL."""
        mu_expert = demonstrations.get_feature_expectations(self.fe)

        for iteration in range(self.n_iters):
            # Set rewards from theta
            for s in range(self.mdp.n_states):
                for a in range(self.mdp.n_actions):
                    feats = self.fe.features(s, a)
                    r = sum(t * f for t, f in zip(self.theta, feats))
                    self.mdp.R[s][a] = r

            # Soft value iteration
            V = self._soft_value_iteration()
            policy = self._get_soft_policy(V)
            svf = self._compute_svf(policy)

            # Learner feature expectations
            mu_learner = [0.0] * self.fe.n_features
            for s in range(self.mdp.n_states):
                if svf[s] < 1e-15:
                    continue
                for a in range(self.mdp.n_actions):
                    weight = svf[s] * policy[s][a]
                    if weight < 1e-15:
                        continue
                    feats = self.fe.features(s, a)
                    for i in range(self.fe.n_features):
                        mu_learner[i] += weight * feats[i]

            grad = [mu_expert[i] - mu_learner[i] for i in range(self.fe.n_features)]
            grad_norm = math.sqrt(sum(g * g for g in grad))
            self.history.append(grad_norm)

            for i in range(len(self.theta)):
                self.theta[i] += self.lr * grad[i]

        return self.theta[:]

    def get_policy(self):
        """Get policy from learned reward (deterministic)."""
        for s in range(self.mdp.n_states):
            for a in range(self.mdp.n_actions):
                feats = self.fe.features(s, a)
                r = sum(t * f for t, f in zip(self.theta, feats))
                self.mdp.R[s][a] = r
        vi = ValueIteration(self.mdp)
        vi.solve()
        return vi.policy[:]

    def get_reward_function(self):
        rewards = []
        for s in range(self.mdp.n_states):
            feats = self.fe.features(s)
            r = sum(t * f for t, f in zip(self.theta, feats))
            rewards.append(r)
        return rewards


# ============================================================
# Utility functions
# ============================================================

def _pearson_correlation(x, y):
    """Pearson correlation coefficient."""
    n = len(x)
    if n == 0:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if std_x < 1e-15 or std_y < 1e-15:
        return 0.0
    return cov / (std_x * std_y)


def _cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-15 or norm_b < 1e-15:
        return 0.0
    return dot / (norm_a * norm_b)
