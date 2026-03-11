"""
C193: Multi-Agent Reinforcement Learning
==========================================
From-scratch multi-agent RL algorithms using only NumPy.

Components:
- MultiAgentEnv: Base multi-agent environment interface
- MatrixGame: Normal-form (matrix) game environment
- GridWorldMARLEnv: Grid world with multiple agents
- IndependentQLearning: Each agent learns independently with own Q-table
- JointActionLearning: Agents track joint action Q-values
- MinimaxQ: Minimax-Q for two-player zero-sum games
- NashQLearning: Nash equilibrium Q-learning for general-sum games
- WoLFPHC: Win-or-Learn-Fast Policy Hill Climbing
- LeniencyLearning: Lenient learners for cooperative games
- CommunicatingAgents: Agents share messages to coordinate
- TeamQLearning: Cooperative team Q-learning with shared reward
- OpponentModeling: Agent models opponent policy to improve own learning
"""

import numpy as np
from collections import defaultdict


# ============================================================
# Multi-Agent Environments
# ============================================================

class MultiAgentEnv:
    """Base multi-agent environment interface."""

    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.n_actions = []  # per-agent action counts
        self.n_states = 0

    def reset(self):
        """Returns initial state (or list of observations)."""
        raise NotImplementedError

    def step(self, actions):
        """Takes list of actions (one per agent).
        Returns (next_state, rewards, done, info).
        rewards is a list of floats, one per agent."""
        raise NotImplementedError


class MatrixGame(MultiAgentEnv):
    """Normal-form (matrix) game. Single state, agents choose simultaneously.

    payoff_matrices: list of 2D arrays, one per agent.
    payoff_matrices[i][a0][a1] = reward for agent i when agents play (a0, a1).
    """

    def __init__(self, payoff_matrices):
        n_agents = len(payoff_matrices)
        super().__init__(n_agents)
        self.payoff_matrices = [np.array(m, dtype=float) for m in payoff_matrices]
        self.n_actions = [m.shape[0] if i == 0 else m.shape[1]
                          for i, m in enumerate(self.payoff_matrices)]
        self.n_states = 1
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, actions):
        rewards = []
        for i in range(self.n_agents):
            rewards.append(float(self.payoff_matrices[i][actions[0], actions[1]]))
        return self.state, rewards, False, {}

    @staticmethod
    def prisoners_dilemma():
        """Classic Prisoner's Dilemma. Actions: 0=Cooperate, 1=Defect."""
        return MatrixGame([
            [[-1, -3], [0, -2]],   # Agent 0
            [[-1, 0], [-3, -2]],   # Agent 1
        ])

    @staticmethod
    def matching_pennies():
        """Zero-sum matching pennies. Actions: 0=Heads, 1=Tails."""
        return MatrixGame([
            [[1, -1], [-1, 1]],
            [[-1, 1], [1, -1]],
        ])

    @staticmethod
    def coordination_game():
        """Both agents get reward for matching actions."""
        return MatrixGame([
            [[2, 0], [0, 1]],
            [[2, 0], [0, 1]],
        ])

    @staticmethod
    def rock_paper_scissors():
        """Zero-sum RPS. 0=Rock, 1=Paper, 2=Scissors."""
        return MatrixGame([
            [[0, -1, 1], [1, 0, -1], [-1, 1, 0]],
            [[0, 1, -1], [-1, 0, 1], [1, -1, 0]],
        ])


class GridWorldMARLEnv(MultiAgentEnv):
    """Grid world with multiple agents navigating to individual goals."""

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

    def __init__(self, rows=5, cols=5, agent_starts=None, agent_goals=None,
                 step_reward=-0.1, goal_reward=5.0, collision_penalty=-1.0):
        n_agents = len(agent_starts) if agent_starts else 2
        super().__init__(n_agents)
        self.rows = rows
        self.cols = cols
        self.agent_starts = agent_starts or [(0, 0), (rows - 1, cols - 1)]
        self.agent_goals = agent_goals or [(rows - 1, cols - 1), (0, 0)]
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.collision_penalty = collision_penalty
        self.n_actions = [4] * n_agents
        self.n_states = (rows * cols) ** n_agents
        self.positions = list(self.agent_starts)
        self.done_agents = [False] * n_agents

    def _pos_to_idx(self, pos):
        return pos[0] * self.cols + pos[1]

    def _state(self):
        """Joint state as tuple of positions."""
        return tuple(self.positions)

    def reset(self):
        self.positions = list(self.agent_starts)
        self.done_agents = [False] * self.n_agents
        return self._state()

    def step(self, actions):
        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        new_positions = []
        for i, action in enumerate(actions):
            if self.done_agents[i]:
                new_positions.append(self.positions[i])
                continue
            dr, dc = deltas[action]
            nr = max(0, min(self.rows - 1, self.positions[i][0] + dr))
            nc = max(0, min(self.cols - 1, self.positions[i][1] + dc))
            new_positions.append((nr, nc))

        # Check collisions
        rewards = [self.step_reward] * self.n_agents
        collision = False
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if not self.done_agents[i] and not self.done_agents[j]:
                    if new_positions[i] == new_positions[j]:
                        collision = True
                        rewards[i] += self.collision_penalty
                        rewards[j] += self.collision_penalty

        if collision:
            # On collision, agents stay in place
            for i in range(self.n_agents):
                if not self.done_agents[i]:
                    new_positions[i] = self.positions[i]

        self.positions = new_positions

        # Check goals
        for i in range(self.n_agents):
            if not self.done_agents[i] and self.positions[i] == self.agent_goals[i]:
                rewards[i] += self.goal_reward
                self.done_agents[i] = True

        done = all(self.done_agents)
        return self._state(), rewards, done, {}


class PredatorPreyEnv(MultiAgentEnv):
    """Two predators try to catch one prey (random policy).
    Cooperative: both predators share positive reward on capture."""

    UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4

    def __init__(self, rows=5, cols=5):
        super().__init__(n_agents=2)  # 2 predators
        self.rows = rows
        self.cols = cols
        self.n_actions = [5, 5]
        self.n_states = (rows * cols) ** 3
        self.predators = [(0, 0), (rows - 1, cols - 1)]
        self.prey = (rows // 2, cols // 2)
        self.max_steps = 50
        self.step_count = 0

    def _state(self):
        return (self.predators[0], self.predators[1], self.prey)

    def reset(self):
        self.predators = [(0, 0), (self.rows - 1, self.cols - 1)]
        self.prey = (self.rows // 2, self.cols // 2)
        self.step_count = 0
        return self._state()

    def _move(self, pos, action):
        deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
        dr, dc = deltas[action]
        nr = max(0, min(self.rows - 1, pos[0] + dr))
        nc = max(0, min(self.cols - 1, pos[1] + dc))
        return (nr, nc)

    def step(self, actions):
        self.step_count += 1
        # Move predators
        for i in range(2):
            self.predators[i] = self._move(self.predators[i], actions[i])

        # Move prey randomly
        prey_action = np.random.randint(5)
        self.prey = self._move(self.prey, prey_action)

        # Check capture (either predator on prey)
        captured = any(p == self.prey for p in self.predators)
        done = captured or self.step_count >= self.max_steps

        if captured:
            rewards = [10.0, 10.0]
        else:
            rewards = [-0.1, -0.1]

        return self._state(), rewards, done, {"captured": captured}


# ============================================================
# Independent Q-Learning
# ============================================================

class IndependentQLearning:
    """Each agent has its own Q-table and learns independently.
    Other agents are part of the environment from each agent's perspective."""

    def __init__(self, n_agents, n_actions_per_agent, alpha=0.1, gamma=0.99,
                 epsilon=0.3, epsilon_decay=0.999, epsilon_min=0.01):
        self.n_agents = n_agents
        self.n_actions = n_actions_per_agent  # list
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Q-table per agent: state -> action -> value
        self.q_tables = [defaultdict(lambda: np.zeros(na))
                         for na in self.n_actions]

    def select_actions(self, state):
        actions = []
        for i in range(self.n_agents):
            s = str(state)
            if np.random.random() < self.epsilon:
                actions.append(np.random.randint(self.n_actions[i]))
            else:
                actions.append(int(np.argmax(self.q_tables[i][s])))
        return actions

    def update(self, state, actions, rewards, next_state, done):
        s = str(state)
        ns = str(next_state)
        for i in range(self.n_agents):
            a = actions[i]
            r = rewards[i]
            best_next = 0.0 if done else np.max(self.q_tables[i][ns])
            td_target = r + self.gamma * best_next
            td_error = td_target - self.q_tables[i][s][a]
            self.q_tables[i][s][a] += self.alpha * td_error
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self, agent_idx, state):
        """Return greedy action for agent at state."""
        s = str(state)
        return int(np.argmax(self.q_tables[agent_idx][s]))

    def train(self, env, episodes=500):
        """Train on a MultiAgentEnv. Returns list of total reward per episode."""
        history = []
        for _ in range(episodes):
            state = env.reset()
            total_rewards = [0.0] * self.n_agents
            for _step in range(200):
                actions = self.select_actions(state)
                next_state, rewards, done, _ = env.step(actions)
                self.update(state, actions, rewards, next_state, done)
                total_rewards = [t + r for t, r in zip(total_rewards, rewards)]
                state = next_state
                if done:
                    break
            history.append(sum(total_rewards))
        return history


# ============================================================
# Joint Action Learning
# ============================================================

class JointActionLearning:
    """Each agent maintains Q-values over joint actions and models opponents."""

    def __init__(self, n_agents, n_actions_per_agent, alpha=0.1, gamma=0.99,
                 epsilon=0.3, epsilon_decay=0.999, epsilon_min=0.01):
        self.n_agents = n_agents
        self.n_actions = n_actions_per_agent
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Q[i][state][joint_action] = value
        self.q_tables = [defaultdict(lambda: defaultdict(float))
                         for _ in range(n_agents)]
        # Opponent action counts: opponent_counts[i][state] = np.array of counts for agent i's actions
        # We need to capture na in the closure properly
        def _make_count_dict(na):
            return defaultdict(lambda: np.zeros(na))
        self.opponent_counts = [_make_count_dict(na) for na in self.n_actions]

    def _joint_action_key(self, actions):
        return tuple(actions)

    def select_actions(self, state):
        s = str(state)
        actions = []
        for i in range(self.n_agents):
            if np.random.random() < self.epsilon:
                actions.append(np.random.randint(self.n_actions[i]))
            else:
                # Pick action that maximizes expected value against opponent model
                best_a = 0
                best_val = -float('inf')
                for a in range(self.n_actions[i]):
                    val = self._expected_value(i, s, a)
                    if val > best_val:
                        best_val = val
                        best_a = a
                actions.append(best_a)
        return actions

    def _expected_value(self, agent_idx, state, action):
        """Expected value of action given opponent model."""
        if self.n_agents != 2:
            # Simplified: just use Q value with uniform opponent
            total = 0.0
            n_opp = self.n_actions[1 - agent_idx]
            for opp_a in range(n_opp):
                ja = [0, 0]
                ja[agent_idx] = action
                ja[1 - agent_idx] = opp_a
                total += self.q_tables[agent_idx][state][self._joint_action_key(ja)]
            return total / max(1, n_opp)

        opp_idx = 1 - agent_idx
        opp_counts = self.opponent_counts[opp_idx][state]
        total_opp = opp_counts.sum()
        if total_opp == 0:
            # Uniform
            n_opp = self.n_actions[opp_idx]
            total = 0.0
            for opp_a in range(n_opp):
                ja = [0, 0]
                ja[agent_idx] = action
                ja[opp_idx] = opp_a
                total += self.q_tables[agent_idx][state][self._joint_action_key(ja)]
            return total / n_opp

        total = 0.0
        for opp_a in range(self.n_actions[opp_idx]):
            prob = opp_counts[opp_a] / total_opp
            ja = [0, 0]
            ja[agent_idx] = action
            ja[opp_idx] = opp_a
            total += prob * self.q_tables[agent_idx][state][self._joint_action_key(ja)]
        return total

    def update(self, state, actions, rewards, next_state, done):
        s = str(state)
        ns = str(next_state)
        ja = self._joint_action_key(actions)

        # Update opponent models
        for i in range(self.n_agents):
            self.opponent_counts[i][s][actions[i]] += 1

        # Update Q-values
        for i in range(self.n_agents):
            if done:
                best_next = 0.0
            else:
                # Best expected value at next state
                best_next = max(
                    self._expected_value(i, ns, a)
                    for a in range(self.n_actions[i])
                )
            td_target = rewards[i] + self.gamma * best_next
            old_q = self.q_tables[i][s][ja]
            self.q_tables[i][s][ja] = old_q + self.alpha * (td_target - old_q)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, episodes=500):
        history = []
        for _ in range(episodes):
            state = env.reset()
            total_rewards = [0.0] * self.n_agents
            for _step in range(200):
                actions = self.select_actions(state)
                next_state, rewards, done, _ = env.step(actions)
                self.update(state, actions, rewards, next_state, done)
                total_rewards = [t + r for t, r in zip(total_rewards, rewards)]
                state = next_state
                if done:
                    break
            history.append(sum(total_rewards))
        return history


# ============================================================
# Minimax-Q Learning (Two-player zero-sum)
# ============================================================

class MinimaxQ:
    """Minimax-Q for two-player zero-sum games.
    Agent 0 is the maximizer, agent 1 is the minimizer.
    Uses linear programming approximation via iterative best response."""

    def __init__(self, n_actions_0, n_actions_1, alpha=0.2, gamma=0.99,
                 epsilon=0.3, epsilon_decay=0.999, epsilon_min=0.01):
        self.n_actions = [n_actions_0, n_actions_1]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Q[state][(a0, a1)] = value (from agent 0's perspective)
        self.q_table = defaultdict(lambda: np.zeros((n_actions_0, n_actions_1)))
        # Mixed strategy for agent 0
        self.policies = defaultdict(lambda: np.ones(n_actions_0) / n_actions_0)
        # Value function
        self.values = defaultdict(float)

    def _compute_maximin(self, q_matrix):
        """Compute maximin strategy via iterative fictitious play.
        Returns (value, mixed_strategy) for the row player (maximizer)."""
        n_rows, n_cols = q_matrix.shape
        if n_rows == 0 or n_cols == 0:
            return 0.0, np.ones(max(n_rows, 1)) / max(n_rows, 1)

        # Fictitious play to approximate maximin
        row_counts = np.ones(n_rows)  # Start uniform
        col_counts = np.ones(n_cols)
        for _ in range(50):
            # Column player best-responds to row player's mixed strategy
            row_strat = row_counts / row_counts.sum()
            col_payoffs = row_strat @ q_matrix  # column player minimizes
            best_col = np.argmin(col_payoffs)
            col_counts[best_col] += 1

            # Row player best-responds to column player's mixed strategy
            col_strat = col_counts / col_counts.sum()
            row_payoffs = q_matrix @ col_strat  # row player maximizes
            best_row = np.argmax(row_payoffs)
            row_counts[best_row] += 1

        policy = row_counts / row_counts.sum()
        col_strat = col_counts / col_counts.sum()
        value = float(policy @ q_matrix @ col_strat)
        return value, policy

    def select_actions(self, state):
        s = str(state)
        actions = [0, 0]
        # Agent 0: uses mixed strategy
        if np.random.random() < self.epsilon:
            actions[0] = np.random.randint(self.n_actions[0])
        else:
            policy = self.policies[s]
            actions[0] = np.random.choice(self.n_actions[0], p=policy)
        # Agent 1: uniform random (opponent)
        actions[1] = np.random.randint(self.n_actions[1])
        return actions

    def update(self, state, actions, reward_0, next_state, done):
        """Update from agent 0's perspective. reward_0 is agent 0's reward."""
        s = str(state)
        ns = str(next_state)
        a0, a1 = actions

        # TD update on Q(s, a0, a1)
        if done:
            target = reward_0
        else:
            target = reward_0 + self.gamma * self.values[ns]

        self.q_table[s][a0, a1] += self.alpha * (target - self.q_table[s][a0, a1])

        # Recompute maximin policy and value
        value, policy = self._compute_maximin(self.q_table[s])
        self.policies[s] = policy
        self.values[s] = value

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, episodes=500):
        """Train on a 2-player zero-sum MatrixGame or similar env."""
        history = []
        for _ in range(episodes):
            state = env.reset()
            total_reward = 0.0
            for _step in range(200):
                actions = self.select_actions(state)
                next_state, rewards, done, _ = env.step(actions)
                self.update(state, actions, rewards[0], next_state, done)
                total_reward += rewards[0]
                state = next_state
                if done:
                    break
            history.append(total_reward)
        return history

    def get_policy(self, state):
        """Return mixed strategy for agent 0 at state."""
        return self.policies[str(state)]


# ============================================================
# Nash Q-Learning (General-sum games)
# ============================================================

class NashQLearning:
    """Nash Q-Learning for two-player general-sum games.
    Computes Nash equilibrium at each state to derive update target."""

    def __init__(self, n_actions_0, n_actions_1, alpha=0.1, gamma=0.99,
                 epsilon=0.3, epsilon_decay=0.999, epsilon_min=0.01):
        self.n_actions = [n_actions_0, n_actions_1]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Separate Q-tables for each agent
        self.q_tables = [
            defaultdict(lambda: np.zeros((n_actions_0, n_actions_1))),
            defaultdict(lambda: np.zeros((n_actions_0, n_actions_1))),
        ]
        self.nash_values = [defaultdict(float), defaultdict(float)]

    def _compute_nash(self, q0, q1):
        """Approximate Nash equilibrium via support enumeration for small games.
        Returns (value0, value1, policy0, policy1)."""
        n0, n1 = q0.shape

        # Try pure strategy Nash first
        best_pure = None
        best_sum = -float('inf')
        for a0 in range(n0):
            for a1 in range(n1):
                # Check if (a0, a1) is a Nash equilibrium
                is_nash = True
                for alt_a0 in range(n0):
                    if q0[alt_a0, a1] > q0[a0, a1] + 1e-10:
                        is_nash = False
                        break
                if is_nash:
                    for alt_a1 in range(n1):
                        if q1[a0, alt_a1] > q1[a0, a1] + 1e-10:
                            is_nash = False
                            break
                if is_nash:
                    total = q0[a0, a1] + q1[a0, a1]
                    if total > best_sum:
                        best_sum = total
                        best_pure = (a0, a1)

        if best_pure is not None:
            a0, a1 = best_pure
            p0 = np.zeros(n0)
            p0[a0] = 1.0
            p1 = np.zeros(n1)
            p1[a1] = 1.0
            return q0[a0, a1], q1[a0, a1], p0, p1

        # Fallback: use maximin for each player independently
        # (approximation -- true Nash requires solving a complementarity problem)
        _, p0 = self._maximin(q0)
        _, p1_t = self._maximin(q1.T)
        p1 = p1_t  # transpose context
        v0 = float(p0 @ q0 @ p1)
        v1 = float(p0 @ q1 @ p1)
        return v0, v1, p0, p1

    def _maximin(self, matrix):
        """Maximin via fictitious play."""
        n_rows, n_cols = matrix.shape
        row_counts = np.ones(n_rows)
        col_counts = np.ones(n_cols)
        for _ in range(100):
            row_strat = row_counts / row_counts.sum()
            col_payoffs = row_strat @ matrix
            col_counts[np.argmin(col_payoffs)] += 1
            col_strat = col_counts / col_counts.sum()
            row_payoffs = matrix @ col_strat
            row_counts[np.argmax(row_payoffs)] += 1
        policy = row_counts / row_counts.sum()
        col_strat = col_counts / col_counts.sum()
        value = float(policy @ matrix @ col_strat)
        return value, policy

    def select_actions(self, state):
        actions = []
        for i in range(2):
            if np.random.random() < self.epsilon:
                actions.append(np.random.randint(self.n_actions[i]))
            else:
                actions.append(np.random.randint(self.n_actions[i]))
        return actions

    def update(self, state, actions, rewards, next_state, done):
        s = str(state)
        ns = str(next_state)
        a0, a1 = actions

        for i in range(2):
            self.q_tables[i][s][a0, a1] += self.alpha * (
                rewards[i] + (0 if done else self.gamma * self.nash_values[i][ns])
                - self.q_tables[i][s][a0, a1]
            )

        # Recompute Nash values at current state
        v0, v1, _, _ = self._compute_nash(self.q_tables[0][s], self.q_tables[1][s])
        self.nash_values[0][s] = v0
        self.nash_values[1][s] = v1

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, episodes=500):
        history = []
        for _ in range(episodes):
            state = env.reset()
            total = [0.0, 0.0]
            for _step in range(200):
                actions = self.select_actions(state)
                next_state, rewards, done, _ = env.step(actions)
                self.update(state, actions, rewards, next_state, done)
                total = [t + r for t, r in zip(total, rewards)]
                state = next_state
                if done:
                    break
            history.append(sum(total))
        return history


# ============================================================
# WoLF-PHC (Win or Learn Fast - Policy Hill Climbing)
# ============================================================

class WoLFPHC:
    """Win or Learn Fast - Policy Hill Climbing.
    Adapts learning rate based on whether agent is winning or losing.
    Winning (above average) -> slow learning rate (cautious).
    Losing (below average) -> fast learning rate (explore more)."""

    def __init__(self, n_actions_0, n_actions_1, alpha=0.1, gamma=0.99,
                 delta_win=0.01, delta_lose=0.04):
        self.n_actions = [n_actions_0, n_actions_1]
        self.alpha = alpha
        self.gamma = gamma
        self.delta_win = delta_win
        self.delta_lose = delta_lose
        # Q-table for agent 0
        self.q_table = defaultdict(lambda: np.zeros(n_actions_0))
        # Current policy: state -> probability distribution
        self.policy = defaultdict(lambda: np.ones(n_actions_0) / n_actions_0)
        # Average policy for comparison
        self.avg_policy = defaultdict(lambda: np.ones(n_actions_0) / n_actions_0)
        # State visit counts
        self.visit_counts = defaultdict(int)

    def select_action(self, state):
        s = str(state)
        return np.random.choice(self.n_actions[0], p=self.policy[s])

    def update(self, state, action, reward, next_state, done):
        s = str(state)
        ns = str(next_state)
        self.visit_counts[s] += 1

        # Standard Q-learning update
        best_next = 0.0 if done else np.max(self.q_table[ns])
        target = reward + self.gamma * best_next
        self.q_table[s][action] += self.alpha * (target - self.q_table[s][action])

        # Update average policy
        c = self.visit_counts[s]
        self.avg_policy[s] = self.avg_policy[s] + (1.0 / c) * (self.policy[s] - self.avg_policy[s])

        # Determine if winning or losing
        current_val = np.dot(self.policy[s], self.q_table[s])
        avg_val = np.dot(self.avg_policy[s], self.q_table[s])
        delta = self.delta_win if current_val > avg_val else self.delta_lose

        # Hill climb: move probability toward best action
        best_action = int(np.argmax(self.q_table[s]))
        for a in range(self.n_actions[0]):
            if a == best_action:
                self.policy[s][a] = min(1.0, self.policy[s][a] +
                                        delta * (self.n_actions[0] - 1) / self.n_actions[0])
            else:
                # Decrease, but not below a small minimum to keep exploring
                self.policy[s][a] = max(0.0, self.policy[s][a] -
                                        delta / self.n_actions[0])

        # Renormalize
        total = self.policy[s].sum()
        if total > 0:
            self.policy[s] /= total

    def train(self, env, episodes=500):
        """Train agent 0 against random agent 1."""
        history = []
        for _ in range(episodes):
            state = env.reset()
            total_reward = 0.0
            for _step in range(200):
                a0 = self.select_action(state)
                a1 = np.random.randint(self.n_actions[1])
                next_state, rewards, done, _ = env.step([a0, a1])
                self.update(state, a0, rewards[0], next_state, done)
                total_reward += rewards[0]
                state = next_state
                if done:
                    break
            history.append(total_reward)
        return history

    def get_policy(self, state):
        return self.policy[str(state)]


# ============================================================
# Leniency Learning
# ============================================================

class LeniencyLearning:
    """Lenient learners for cooperative games.
    Each agent tracks 'temperature' per state-action pair.
    High temperature -> more lenient (ignore low rewards, likely from partner mistakes).
    Temperature decays over time, making agent less lenient as it learns."""

    def __init__(self, n_agents, n_actions_per_agent, alpha=0.1, gamma=0.99,
                 epsilon=0.3, epsilon_decay=0.999, epsilon_min=0.01,
                 initial_temp=10.0, temp_decay=0.995):
        self.n_agents = n_agents
        self.n_actions = n_actions_per_agent
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.initial_temp = initial_temp
        self.temp_decay = temp_decay
        self.q_tables = [defaultdict(lambda: np.zeros(na))
                         for na in self.n_actions]
        # Temperature per agent per state-action
        self.temperatures = [defaultdict(lambda: defaultdict(lambda: initial_temp))
                             for _ in range(n_agents)]

    def select_actions(self, state):
        s = str(state)
        actions = []
        for i in range(self.n_agents):
            if np.random.random() < self.epsilon:
                actions.append(np.random.randint(self.n_actions[i]))
            else:
                actions.append(int(np.argmax(self.q_tables[i][s])))
        return actions

    def update(self, state, actions, rewards, next_state, done):
        s = str(state)
        ns = str(next_state)
        for i in range(self.n_agents):
            a = actions[i]
            r = rewards[i]
            temp = self.temperatures[i][s][a]

            best_next = 0.0 if done else np.max(self.q_tables[i][ns])
            td_target = r + self.gamma * best_next
            td_error = td_target - self.q_tables[i][s][a]

            # Leniency: only update if td_error is positive OR temperature is low
            # With probability 1-exp(-temp), ignore negative updates
            if td_error >= 0:
                self.q_tables[i][s][a] += self.alpha * td_error
            else:
                leniency_prob = 1.0 - np.exp(-temp)
                if np.random.random() > leniency_prob:
                    # Not lenient -- apply negative update
                    self.q_tables[i][s][a] += self.alpha * td_error

            # Decay temperature
            self.temperatures[i][s][a] = temp * self.temp_decay

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, episodes=500):
        history = []
        for _ in range(episodes):
            state = env.reset()
            total_rewards = [0.0] * self.n_agents
            for _step in range(200):
                actions = self.select_actions(state)
                next_state, rewards, done, _ = env.step(actions)
                self.update(state, actions, rewards, next_state, done)
                total_rewards = [t + r for t, r in zip(total_rewards, rewards)]
                state = next_state
                if done:
                    break
            history.append(sum(total_rewards))
        return history


# ============================================================
# Communicating Agents
# ============================================================

class CommunicatingAgents:
    """Agents can send discrete messages to coordinate.
    Each agent has an extended action: (env_action, message).
    Messages from last round are part of the observation."""

    def __init__(self, n_agents, n_env_actions, n_messages=4,
                 alpha=0.1, gamma=0.99, epsilon=0.3,
                 epsilon_decay=0.999, epsilon_min=0.01):
        self.n_agents = n_agents
        self.n_env_actions = n_env_actions
        self.n_messages = n_messages
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Q-table: (state, received_messages) -> (env_action, message) -> value
        # Total actions per agent: n_env_actions * n_messages
        self.n_total_actions = n_env_actions * n_messages
        self.q_tables = [defaultdict(lambda: np.zeros(n_env_actions * n_messages))
                         for _ in range(n_agents)]
        self.last_messages = [0] * n_agents

    def _obs_key(self, state, messages):
        return str((state, tuple(messages)))

    def _decode_action(self, combined):
        env_action = combined // self.n_messages
        message = combined % self.n_messages
        return env_action, message

    def _encode_action(self, env_action, message):
        return env_action * self.n_messages + message

    def select_actions(self, state, received_messages=None):
        if received_messages is None:
            received_messages = self.last_messages
        obs = self._obs_key(state, received_messages)
        env_actions = []
        new_messages = []
        for i in range(self.n_agents):
            if np.random.random() < self.epsilon:
                combined = np.random.randint(self.n_total_actions)
            else:
                combined = int(np.argmax(self.q_tables[i][obs]))
            ea, msg = self._decode_action(combined)
            # Clamp env_action to valid range
            ea = ea % self.n_env_actions
            env_actions.append(ea)
            new_messages.append(msg)
        return env_actions, new_messages

    def update(self, state, messages_received, actions_and_msgs, rewards,
               next_state, next_messages, done):
        obs = self._obs_key(state, messages_received)
        next_obs = self._obs_key(next_state, next_messages)
        for i in range(self.n_agents):
            combined = self._encode_action(actions_and_msgs[i][0],
                                           actions_and_msgs[i][1])
            best_next = 0.0 if done else np.max(self.q_tables[i][next_obs])
            target = rewards[i] + self.gamma * best_next
            self.q_tables[i][obs][combined] += self.alpha * (
                target - self.q_tables[i][obs][combined])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, episodes=500):
        history = []
        for _ in range(episodes):
            state = env.reset()
            messages = [0] * self.n_agents
            total_rewards = [0.0] * self.n_agents
            for _step in range(200):
                env_actions, new_messages = self.select_actions(state, messages)
                next_state, rewards, done, _ = env.step(env_actions)
                actions_and_msgs = list(zip(env_actions, new_messages))
                self.update(state, messages, actions_and_msgs, rewards,
                            next_state, new_messages, done)
                total_rewards = [t + r for t, r in zip(total_rewards, rewards)]
                state = next_state
                messages = new_messages
                if done:
                    break
            history.append(sum(total_rewards))
        return history


# ============================================================
# Team Q-Learning
# ============================================================

class TeamQLearning:
    """Cooperative team Q-learning. All agents share a single Q-table
    over joint state-action space and optimize shared team reward."""

    def __init__(self, n_agents, n_actions_per_agent, alpha=0.1, gamma=0.99,
                 epsilon=0.3, epsilon_decay=0.999, epsilon_min=0.01):
        self.n_agents = n_agents
        self.n_actions = n_actions_per_agent
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Single shared Q-table: state -> joint_action -> value
        self.q_table = defaultdict(lambda: defaultdict(float))

    def _joint_key(self, actions):
        return tuple(actions)

    def _all_joint_actions(self):
        """Generate all possible joint actions."""
        if self.n_agents == 1:
            return [(a,) for a in range(self.n_actions[0])]
        # For 2 agents
        result = []
        from itertools import product
        ranges = [range(na) for na in self.n_actions]
        for combo in product(*ranges):
            result.append(combo)
        return result

    def select_actions(self, state):
        s = str(state)
        if np.random.random() < self.epsilon:
            return [np.random.randint(na) for na in self.n_actions]
        # Pick joint action that maximizes team Q
        best_ja = None
        best_q = -float('inf')
        for ja in self._all_joint_actions():
            q = self.q_table[s][ja]
            if q > best_q:
                best_q = q
                best_ja = ja
        return list(best_ja)

    def update(self, state, actions, team_reward, next_state, done):
        s = str(state)
        ns = str(next_state)
        ja = self._joint_key(actions)

        if done:
            best_next = 0.0
        else:
            best_next = max(
                (self.q_table[ns][nj] for nj in self._all_joint_actions()),
                default=0.0
            )
        target = team_reward + self.gamma * best_next
        self.q_table[s][ja] += self.alpha * (target - self.q_table[s][ja])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, episodes=500):
        history = []
        for _ in range(episodes):
            state = env.reset()
            total_reward = 0.0
            for _step in range(200):
                actions = self.select_actions(state)
                next_state, rewards, done, _ = env.step(actions)
                team_r = sum(rewards)
                self.update(state, actions, team_r, next_state, done)
                total_reward += team_r
                state = next_state
                if done:
                    break
            history.append(total_reward)
        return history


# ============================================================
# Opponent Modeling
# ============================================================

class OpponentModeling:
    """Agent models opponent's policy from observations, then best-responds.
    Uses frequency-based opponent model with Bayesian smoothing."""

    def __init__(self, n_actions_0, n_actions_1, alpha=0.1, gamma=0.99,
                 epsilon=0.3, epsilon_decay=0.999, epsilon_min=0.01,
                 smoothing=1.0):
        self.n_actions = [n_actions_0, n_actions_1]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.smoothing = smoothing
        # Q-table for agent 0: Q(s, a0, a1)
        self.q_table = defaultdict(lambda: np.zeros((n_actions_0, n_actions_1)))
        # Opponent model: P(a1 | s)
        self.opponent_counts = defaultdict(lambda: np.ones(n_actions_1) * smoothing)

    def get_opponent_model(self, state):
        """Return estimated opponent policy at state."""
        s = str(state)
        counts = self.opponent_counts[s]
        return counts / counts.sum()

    def select_action(self, state):
        s = str(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions[0])
        # Best response to opponent model
        opp_policy = self.get_opponent_model(state)
        expected_q = self.q_table[s] @ opp_policy  # shape: (n_actions_0,)
        return int(np.argmax(expected_q))

    def observe_opponent(self, state, opponent_action):
        """Update opponent model with observed action."""
        s = str(state)
        self.opponent_counts[s][opponent_action] += 1

    def update(self, state, action_0, action_1, reward, next_state, done):
        s = str(state)
        ns = str(next_state)

        # Update opponent model
        self.observe_opponent(state, action_1)

        # Q-learning update
        if done:
            target = reward
        else:
            opp_policy = self.get_opponent_model(next_state)
            expected_next = np.max(self.q_table[ns] @ opp_policy)
            target = reward + self.gamma * expected_next

        self.q_table[s][action_0, action_1] += self.alpha * (
            target - self.q_table[s][action_0, action_1])

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, episodes=500):
        """Train against env. Agent 0 learns, agent 1 acts randomly."""
        history = []
        for _ in range(episodes):
            state = env.reset()
            total_reward = 0.0
            for _step in range(200):
                a0 = self.select_action(state)
                a1 = np.random.randint(self.n_actions[1])
                next_state, rewards, done, _ = env.step([a0, a1])
                self.update(state, a0, a1, rewards[0], next_state, done)
                total_reward += rewards[0]
                state = next_state
                if done:
                    break
            history.append(total_reward)
        return history


# ============================================================
# Fictitious Play
# ============================================================

class FictitiousPlay:
    """Classical fictitious play for repeated matrix games.
    Each player best-responds to the empirical frequency of opponent actions."""

    def __init__(self, n_actions_0, n_actions_1):
        self.n_actions = [n_actions_0, n_actions_1]
        # Empirical counts of opponent actions
        self.counts = [
            np.ones(n_actions_1),  # Agent 0's model of agent 1
            np.ones(n_actions_0),  # Agent 1's model of agent 0
        ]
        self.action_history = [[], []]

    def select_actions(self, payoff_matrices):
        """Select actions based on best response to empirical opponent distribution."""
        actions = []
        for i in range(2):
            opp_idx = 1 - i
            opp_dist = self.counts[i] / self.counts[i].sum()
            # Expected payoff for each action
            expected = payoff_matrices[i] @ opp_dist if i == 0 else \
                       payoff_matrices[i].T @ opp_dist
            # Best response (break ties randomly)
            best = np.where(expected == expected.max())[0]
            action = np.random.choice(best)
            actions.append(int(action))
        return actions

    def update(self, actions):
        """Update empirical counts."""
        self.counts[0][actions[1]] += 1  # Agent 0 observes agent 1
        self.counts[1][actions[0]] += 1  # Agent 1 observes agent 0
        self.action_history[0].append(actions[0])
        self.action_history[1].append(actions[1])

    def get_empirical_strategy(self, agent_idx):
        """Return empirical mixed strategy for an agent."""
        opp_idx = 1 - agent_idx
        return self.counts[opp_idx] / self.counts[opp_idx].sum()

    def play(self, payoff_matrices, rounds=1000):
        """Play repeated game and return action histories."""
        for _ in range(rounds):
            actions = self.select_actions(payoff_matrices)
            self.update(actions)
        return self.action_history


# ============================================================
# Mean Field Q-Learning
# ============================================================

class MeanFieldQ:
    """Mean Field Q-Learning for many-agent settings.
    Each agent's Q-function depends on its own action and the mean action
    of its neighbors, rather than all individual actions."""

    def __init__(self, n_agents, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=0.3, epsilon_decay=0.999, epsilon_min=0.01,
                 n_mean_bins=5):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_mean_bins = n_mean_bins
        # Q[agent][state, mean_bin][action]
        self.q_tables = [defaultdict(lambda: np.zeros(n_actions))
                         for _ in range(n_agents)]

    def _discretize_mean(self, mean_action):
        """Discretize mean action into bins."""
        bin_idx = int(mean_action * self.n_mean_bins / self.n_actions)
        return min(bin_idx, self.n_mean_bins - 1)

    def _obs_key(self, state, mean_action):
        return str((state, self._discretize_mean(mean_action)))

    def select_actions(self, state, prev_actions=None):
        actions = []
        for i in range(self.n_agents):
            # Compute mean action of others
            if prev_actions is not None:
                others = [prev_actions[j] for j in range(self.n_agents) if j != i]
                mean_a = np.mean(others) if others else 0.0
            else:
                mean_a = 0.0
            obs = self._obs_key(state, mean_a)
            if np.random.random() < self.epsilon:
                actions.append(np.random.randint(self.n_actions))
            else:
                actions.append(int(np.argmax(self.q_tables[i][obs])))
        return actions

    def update(self, state, actions, rewards, next_state, next_actions, done):
        for i in range(self.n_agents):
            others = [actions[j] for j in range(self.n_agents) if j != i]
            mean_a = np.mean(others) if others else 0.0
            obs = self._obs_key(state, mean_a)

            next_others = [next_actions[j] for j in range(self.n_agents) if j != i]
            next_mean = np.mean(next_others) if next_others else 0.0
            next_obs = self._obs_key(next_state, next_mean)

            best_next = 0.0 if done else np.max(self.q_tables[i][next_obs])
            target = rewards[i] + self.gamma * best_next
            self.q_tables[i][obs][actions[i]] += self.alpha * (
                target - self.q_tables[i][obs][actions[i]])

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, episodes=500):
        history = []
        for _ in range(episodes):
            state = env.reset()
            prev_actions = [0] * self.n_agents
            total_rewards = [0.0] * self.n_agents
            for _step in range(200):
                actions = self.select_actions(state, prev_actions)
                next_state, rewards, done, _ = env.step(actions)
                next_actions = self.select_actions(next_state, actions)
                self.update(state, actions, rewards, next_state, next_actions, done)
                total_rewards = [t + r for t, r in zip(total_rewards, rewards)]
                state = next_state
                prev_actions = actions
                if done:
                    break
            history.append(sum(total_rewards))
        return history
