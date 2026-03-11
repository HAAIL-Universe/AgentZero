"""
C181: Multi-Agent Reinforcement Learning
Composes C179 (RL) + C180 (Model-Based RL)

Components:
- MultiAgentEnv: base class for multi-agent environments
- MatrixGame: normal-form games (Prisoner's Dilemma, Stag Hunt, Chicken, Matching Pennies)
- GridWorldMA: multi-agent grid world (predator-prey, cooperative navigation)
- IndependentLearners: independent Q-learning (IQL) -- each agent learns separately
- CentralizedCritic: centralized training, decentralized execution (CTDE)
- CommChannel: differentiable communication channel between agents
- QMIX: monotonic value decomposition for cooperative MARL
- SelfPlay: self-play training for competitive two-player games
- TeamReward: reward shaping for cooperation (shared, difference, shaped)
- MATrainer: multi-agent training loop
- TournamentEvaluator: round-robin evaluation of agent populations
"""

import sys
import os
import math
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C179_reinforcement_learning'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C180_model_based_rl'))

from neural_network import (
    Tensor, Dense, Activation, Sequential, MSELoss, Adam,
)
from reinforcement_learning import (
    Environment, GridWorld, ReplayBuffer, EpsilonGreedy,
    QLearning, DQN, _RNG,
)
from model_based_rl import DataBuffer, _to_tensor_2d, _to_flat, _one_hot_action, _clip


# ============================================================
# Helpers
# ============================================================

def _argmax(lst):
    best_i, best_v = 0, lst[0]
    for i in range(1, len(lst)):
        if lst[i] > best_v:
            best_i, best_v = i, lst[i]
    return best_i


def _softmax(logits):
    mx = max(logits)
    exps = [math.exp(x - mx) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _sample_categorical(probs, rng):
    r = rng.random()
    cumsum = 0.0
    for i, p in enumerate(probs):
        cumsum += p
        if r < cumsum:
            return i
    return len(probs) - 1


def _flatten_obs(observations):
    """Flatten a list of observations into a single vector."""
    flat = []
    for obs in observations:
        if isinstance(obs, list):
            flat.extend(obs)
        else:
            flat.append(obs)
    return flat


# ============================================================
# MultiAgentEnv -- base class
# ============================================================

class MultiAgentEnv:
    """Base class for multi-agent environments.

    Each step takes a list of actions (one per agent) and returns:
    - observations: list of obs per agent
    - rewards: list of rewards per agent
    - done: bool
    - info: dict
    """

    def __init__(self, n_agents, obs_size, action_size):
        self.n_agents = n_agents
        self.obs_size = obs_size
        self.action_size = action_size

    def reset(self):
        """Returns list of observations, one per agent."""
        raise NotImplementedError

    def step(self, actions):
        """Takes list of actions. Returns (observations, rewards, done, info)."""
        raise NotImplementedError


# ============================================================
# MatrixGame -- normal-form games
# ============================================================

class MatrixGame(MultiAgentEnv):
    """Normal-form (matrix) game for 2 players.

    Payoff matrices define rewards for each action combination.
    Single-step game: each step is one round, always done after.
    Obs is just [1.0] (stateless) unless repeated.
    """

    # Classic games
    PRISONERS_DILEMMA = {
        'payoffs': [
            [(-1, -1), (-3, 0)],   # player 0: cooperate
            [(0, -3), (-2, -2)],    # player 0: defect
        ],
        'actions': ['cooperate', 'defect'],
    }

    STAG_HUNT = {
        'payoffs': [
            [(4, 4), (0, 3)],   # stag, stag vs stag, hare
            [(3, 0), (3, 3)],   # hare, stag vs hare, hare
        ],
        'actions': ['stag', 'hare'],
    }

    CHICKEN = {
        'payoffs': [
            [(0, 0), (-1, 1)],   # straight, straight vs straight, swerve
            [(1, -1), (-5, -5)],  # swerve, straight vs swerve, swerve
        ],
        'actions': ['straight', 'swerve'],
    }

    MATCHING_PENNIES = {
        'payoffs': [
            [(1, -1), (-1, 1)],
            [(-1, 1), (1, -1)],
        ],
        'actions': ['heads', 'tails'],
    }

    COORDINATION = {
        'payoffs': [
            [(2, 2), (0, 0)],
            [(0, 0), (1, 1)],
        ],
        'actions': ['A', 'B'],
    }

    def __init__(self, game_def=None, payoffs=None, n_rounds=1, seed=42):
        if game_def is not None:
            payoffs = game_def['payoffs']
        if payoffs is None:
            payoffs = self.PRISONERS_DILEMMA['payoffs']

        n_actions = len(payoffs)
        # obs_size: round number one-hot + last actions one-hot per agent
        obs_size = 1 if n_rounds == 1 else n_rounds + n_actions * 2
        super().__init__(n_agents=2, obs_size=obs_size, action_size=n_actions)

        self.payoffs = payoffs
        self.n_rounds = n_rounds
        self.round = 0
        self.history = []
        self.rng = _RNG(seed)

    def reset(self):
        self.round = 0
        self.history = []
        return self._get_obs()

    def _get_obs(self):
        if self.n_rounds == 1:
            return [[1.0], [1.0]]
        obs = []
        for agent_id in range(2):
            o = [0.0] * self.obs_size
            if self.round < self.n_rounds:
                o[self.round] = 1.0
            if self.history:
                last_a0, last_a1 = self.history[-1]
                o[self.n_rounds + last_a0] = 1.0
                o[self.n_rounds + self.action_size + last_a1] = 1.0
            obs.append(o)
        return obs

    def step(self, actions):
        a0, a1 = actions[0], actions[1]
        r0, r1 = self.payoffs[a0][a1]
        self.history.append((a0, a1))
        self.round += 1
        done = self.round >= self.n_rounds
        return self._get_obs(), [r0, r1], done, {'actions': (a0, a1)}


# ============================================================
# GridWorldMA -- multi-agent grid world
# ============================================================

class GridWorldMA(MultiAgentEnv):
    """Multi-agent grid world.

    Modes:
    - 'cooperative': agents get shared reward for all reaching goals
    - 'predator_prey': predators try to catch prey
    - 'competitive': agents race to reach a single goal

    Actions: 0=up, 1=right, 2=down, 3=left, 4=stay
    Obs per agent: own position one-hot + other agents' positions one-hot
    """

    def __init__(self, rows=5, cols=5, n_agents=2, mode='cooperative',
                 starts=None, goals=None, walls=None, max_steps=50, seed=42):
        grid_size = rows * cols
        obs_size = grid_size * n_agents  # each agent sees all positions
        super().__init__(n_agents=n_agents, obs_size=obs_size, action_size=5)

        self.rows = rows
        self.cols = cols
        self.mode = mode
        self.walls = set(walls) if walls else set()
        self.max_steps = max_steps
        self.rng = _RNG(seed)

        if starts:
            self.default_starts = list(starts)
        else:
            self.default_starts = [(0, 0)] + [(rows - 1, cols - 1)] * (n_agents - 1)

        if goals:
            self.goals = list(goals)
        else:
            if mode == 'predator_prey':
                self.goals = [None] * n_agents  # no fixed goals
            elif mode == 'competitive':
                self.goals = [(rows - 1, cols - 1)] * n_agents  # same goal
            else:
                # cooperative: each agent has own goal
                self.goals = [(rows - 1, cols - 1), (0, cols - 1)]
                while len(self.goals) < n_agents:
                    self.goals.append((rows - 1, 0))

        self.positions = list(self.default_starts)
        self.steps = 0

    def reset(self):
        self.positions = list(self.default_starts)
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        obs = []
        grid_size = self.rows * self.cols
        for i in range(self.n_agents):
            o = [0.0] * self.obs_size
            for j in range(self.n_agents):
                r, c = self.positions[j]
                idx = j * grid_size + r * self.cols + c
                o[idx] = 1.0
            obs.append(o)
        return obs

    def _move(self, pos, action):
        r, c = pos
        dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)][action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self.walls:
            return (nr, nc)
        return pos

    def step(self, actions):
        self.steps += 1

        # Move all agents
        new_positions = []
        for i in range(self.n_agents):
            new_positions.append(self._move(self.positions[i], actions[i]))

        # Collision resolution: if two agents try to occupy same cell, neither moves
        # (except predator_prey where overlap = capture)
        if self.mode != 'predator_prey':
            for i in range(self.n_agents):
                for j in range(i + 1, self.n_agents):
                    if new_positions[i] == new_positions[j]:
                        new_positions[i] = self.positions[i]
                        new_positions[j] = self.positions[j]

        self.positions = new_positions

        # Compute rewards based on mode
        if self.mode == 'cooperative':
            rewards, done = self._cooperative_rewards()
        elif self.mode == 'predator_prey':
            rewards, done = self._predator_prey_rewards()
        elif self.mode == 'competitive':
            rewards, done = self._competitive_rewards()
        else:
            rewards = [0.0] * self.n_agents
            done = False

        if self.steps >= self.max_steps:
            done = True

        info = {'positions': list(self.positions), 'steps': self.steps}
        return self._get_obs(), rewards, done, info

    def _cooperative_rewards(self):
        at_goal = sum(1 for i in range(self.n_agents)
                      if self.goals[i] and self.positions[i] == self.goals[i])
        if at_goal == self.n_agents:
            return [1.0] * self.n_agents, True
        step_penalty = -0.01
        return [step_penalty + 0.1 * at_goal / self.n_agents] * self.n_agents, False

    def _predator_prey_rewards(self):
        # Agent 0 is prey, rest are predators
        prey_pos = self.positions[0]
        caught = any(self.positions[i] == prey_pos for i in range(1, self.n_agents))
        if caught:
            rewards = [-1.0]  # prey caught
            rewards.extend([1.0] * (self.n_agents - 1))  # predators win
            return rewards, True
        # Step rewards: predators get closer = better
        rewards = [0.01]  # prey survives
        for i in range(1, self.n_agents):
            dist = abs(self.positions[i][0] - prey_pos[0]) + abs(self.positions[i][1] - prey_pos[1])
            rewards.append(-0.01 * dist)
        return rewards, False

    def _competitive_rewards(self):
        goal = self.goals[0]  # single shared goal
        for i in range(self.n_agents):
            if self.positions[i] == goal:
                rewards = [-0.5] * self.n_agents
                rewards[i] = 1.0
                return rewards, True
        return [-0.01] * self.n_agents, False


# ============================================================
# IndependentLearners -- IQL
# ============================================================

class IndependentLearners:
    """Independent Q-Learning: each agent has its own Q-table.

    Simple but surprisingly effective baseline. Each agent treats
    other agents as part of the environment (non-stationary from
    each agent's perspective).
    """

    def __init__(self, n_agents, obs_size, action_size, lr=0.1, gamma=0.99,
                 epsilon=0.3, epsilon_decay=0.995, epsilon_min=0.01, seed=42):
        self.n_agents = n_agents
        self.obs_size = obs_size
        self.action_size = action_size
        self.gamma = gamma
        self.rng = _RNG(seed)

        # Each agent gets its own Q-table (state-hash -> action values)
        self.q_tables = [{} for _ in range(n_agents)]
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def _obs_key(self, obs):
        return tuple(round(x, 4) for x in obs)

    def _get_q(self, agent_id, obs):
        key = self._obs_key(obs)
        if key not in self.q_tables[agent_id]:
            self.q_tables[agent_id][key] = [0.0] * self.action_size
        return self.q_tables[agent_id][key]

    def select_actions(self, observations):
        actions = []
        for i in range(self.n_agents):
            if self.rng.random() < self.epsilon:
                actions.append(int(self.rng.random() * self.action_size))
            else:
                q = self._get_q(i, observations[i])
                actions.append(_argmax(q))
        return actions

    def update(self, observations, actions, rewards, next_observations, done):
        for i in range(self.n_agents):
            q = self._get_q(i, observations[i])
            if done:
                target = rewards[i]
            else:
                next_q = self._get_q(i, next_observations[i])
                target = rewards[i] + self.gamma * max(next_q)
            q[actions[i]] += self.lr * (target - q[actions[i]])

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_greedy_actions(self, observations):
        actions = []
        for i in range(self.n_agents):
            q = self._get_q(i, observations[i])
            actions.append(_argmax(q))
        return actions


# ============================================================
# CentralizedCritic -- CTDE
# ============================================================

class CentralizedCritic:
    """Centralized Training with Decentralized Execution (CTDE).

    Each agent has its own policy network (decentralized execution).
    A centralized critic sees all agents' observations and actions
    to provide better value estimates during training.
    """

    def __init__(self, n_agents, obs_size, action_size, hidden_size=32,
                 lr=0.01, gamma=0.99, seed=42):
        self.n_agents = n_agents
        self.obs_size = obs_size
        self.action_size = action_size
        self.gamma = gamma
        self.rng = _RNG(seed)

        # Per-agent policy networks (obs -> action logits)
        self.policies = []
        for _ in range(n_agents):
            policy = Sequential([
                Dense(obs_size, hidden_size, init='xavier'),
                Activation('relu'),
                Dense(hidden_size, action_size, init='xavier'),
            ])
            self.policies.append(policy)

        # Centralized critic (all obs + all actions -> value)
        joint_input = obs_size * n_agents + action_size * n_agents
        self.critic = Sequential([
            Dense(joint_input, hidden_size, init='xavier'),
            Activation('relu'),
            Dense(hidden_size, 1, init='xavier'),
        ])

        self.policy_optimizers = [Adam(lr=lr) for _ in range(n_agents)]
        self.critic_optimizer = Adam(lr=lr)
        self.loss_fn = MSELoss()

    def select_actions(self, observations, explore=True):
        actions = []
        for i in range(self.n_agents):
            obs_t = _to_tensor_2d(observations[i])
            logits = self.policies[i].forward(obs_t)
            logits_flat = _to_flat(logits)
            probs = _softmax(logits_flat)
            if explore:
                action = _sample_categorical(probs, self.rng)
            else:
                action = _argmax(probs)
            actions.append(action)
        return actions

    def _joint_input(self, observations, actions):
        joint = []
        for obs in observations:
            joint.extend(obs if isinstance(obs, list) else [obs])
        for i, a in enumerate(actions):
            oh = [0.0] * self.action_size
            oh[a] = 1.0
            joint.extend(oh)
        return joint

    def update(self, observations, actions, rewards, next_observations, next_actions, done):
        # Update critic
        joint_in = self._joint_input(observations, actions)
        joint_t = _to_tensor_2d(joint_in)
        value = self.critic.forward(joint_t)

        if done:
            target_val = sum(rewards) / self.n_agents
        else:
            next_joint = self._joint_input(next_observations, next_actions)
            next_joint_t = _to_tensor_2d(next_joint)
            next_value = self.critic.forward(next_joint_t)
            next_v = _to_flat(next_value)
            if isinstance(next_v, list):
                next_v = next_v[0]
            target_val = sum(rewards) / self.n_agents + self.gamma * next_v

        target_t = Tensor([[target_val]])
        critic_loss = self.loss_fn.forward(value, target_t)
        critic_grad = self.loss_fn.backward(value, target_t)
        self.critic.backward(critic_grad)
        self.critic_optimizer.step(self.critic.get_trainable_layers())
        # Update each agent's policy using advantage
        v = _to_flat(value)
        if isinstance(v, list):
            v = v[0]

        for i in range(self.n_agents):
            advantage = rewards[i] - v
            obs_t = _to_tensor_2d(observations[i])
            logits = self.policies[i].forward(obs_t)
            logits_flat = _to_flat(logits)
            probs = _softmax(logits_flat)

            # Policy gradient: grad = -advantage * d(log pi) / d(logits)
            grad = [0.0] * self.action_size
            for a in range(self.action_size):
                if a == actions[i]:
                    grad[a] = -advantage * (1.0 - probs[a])
                else:
                    grad[a] = advantage * probs[a]

            grad_t = _to_tensor_2d(grad)
            self.policies[i].backward(grad_t)
            self.policy_optimizers[i].step(self.policies[i].get_trainable_layers())
            self.policies[i]
        return float(critic_loss) if isinstance(critic_loss, (int, float)) else _to_flat(critic_loss)[0] if isinstance(_to_flat(critic_loss), list) else float(_to_flat(critic_loss))


# ============================================================
# CommChannel -- differentiable communication
# ============================================================

class CommChannel:
    """Differentiable communication channel between agents.

    Each agent produces a message vector. Messages are aggregated
    (mean or attention) and appended to each agent's observation
    before action selection.

    This enables emergent communication protocols.
    """

    def __init__(self, n_agents, obs_size, msg_size=8, aggregation='mean', seed=42):
        self.n_agents = n_agents
        self.obs_size = obs_size
        self.msg_size = msg_size
        self.aggregation = aggregation
        self.rng = _RNG(seed)

        # Message encoder per agent: obs -> message
        self.encoders = []
        for _ in range(n_agents):
            enc = Sequential([
                Dense(obs_size, msg_size, init='xavier'),
                Activation('tanh'),
            ])
            self.encoders.append(enc)

        self.last_messages = None

    def produce_messages(self, observations):
        messages = []
        for i in range(self.n_agents):
            obs_t = _to_tensor_2d(observations[i])
            msg = self.encoders[i].forward(obs_t)
            msg_flat = _to_flat(msg)
            if not isinstance(msg_flat, list):
                msg_flat = [msg_flat]
            messages.append(msg_flat)
        self.last_messages = messages
        return messages

    def aggregate(self, messages, agent_id):
        """Aggregate messages from other agents for agent_id."""
        others = [m for j, m in enumerate(messages) if j != agent_id]
        if not others:
            return [0.0] * self.msg_size

        if self.aggregation == 'mean':
            agg = [0.0] * self.msg_size
            for m in others:
                for k in range(self.msg_size):
                    agg[k] += m[k]
            for k in range(self.msg_size):
                agg[k] /= len(others)
            return agg
        elif self.aggregation == 'sum':
            agg = [0.0] * self.msg_size
            for m in others:
                for k in range(self.msg_size):
                    agg[k] += m[k]
            return agg
        elif self.aggregation == 'max':
            agg = [float('-inf')] * self.msg_size
            for m in others:
                for k in range(self.msg_size):
                    agg[k] = max(agg[k], m[k])
            return agg
        else:
            return others[0] if others else [0.0] * self.msg_size

    def augment_observations(self, observations):
        """Produce messages and augment each agent's observation."""
        messages = self.produce_messages(observations)
        augmented = []
        for i in range(self.n_agents):
            agg = self.aggregate(messages, i)
            augmented.append(observations[i] + agg)
        return augmented, messages

    def get_trainable_layers(self):
        layers = []
        for enc in self.encoders:
            layers.extend(enc.get_trainable_layers())
        return layers


# ============================================================
# QMIX -- value decomposition
# ============================================================

class QMIX:
    """QMIX: Monotonic Value Function Decomposition.

    Each agent has a local Q-network. A mixing network combines
    individual Q-values into a joint Q_tot with monotonicity constraint
    (weights are non-negative via abs).

    Q_tot = mix(Q_1, Q_2, ..., Q_n | state)
    dQ_tot/dQ_i >= 0 (monotonicity)
    """

    def __init__(self, n_agents, obs_size, action_size, state_size=None,
                 hidden_size=32, mixing_hidden=32, lr=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05, seed=42):
        self.n_agents = n_agents
        self.obs_size = obs_size
        self.action_size = action_size
        self.state_size = state_size or obs_size * n_agents
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = _RNG(seed)

        # Per-agent Q-networks
        self.q_nets = []
        for _ in range(n_agents):
            net = Sequential([
                Dense(obs_size, hidden_size, init='xavier'),
                Activation('relu'),
                Dense(hidden_size, action_size, init='xavier'),
            ])
            self.q_nets.append(net)

        # Mixing network: takes individual Q-values + global state
        # Hypernetwork: state -> mixing weights (kept non-negative)
        self.hyper_w1 = Dense(self.state_size, n_agents * mixing_hidden, init='xavier')
        self.hyper_b1 = Dense(self.state_size, mixing_hidden, init='xavier')
        self.hyper_w2 = Dense(self.state_size, mixing_hidden, init='xavier')
        self.hyper_b2 = Dense(self.state_size, 1, init='xavier')
        self.mixing_hidden = mixing_hidden

        self.optimizer = Adam(lr=lr)
        self.loss_fn = MSELoss()

        # Experience buffer
        self.buffer = []
        self.buffer_capacity = 5000
        self.buffer_pos = 0

    def _get_q_values(self, observations):
        """Get Q-values for each agent."""
        q_values = []
        for i in range(self.n_agents):
            obs_t = _to_tensor_2d(observations[i])
            q = self.q_nets[i].forward(obs_t)
            q_flat = _to_flat(q)
            if not isinstance(q_flat, list):
                q_flat = [q_flat]
            q_values.append(q_flat)
        return q_values

    def _mix(self, agent_qs, state):
        """Mix individual Q-values using hypernetwork."""
        state_t = _to_tensor_2d(state)

        # First layer weights (non-negative via abs)
        w1_flat = _to_flat(self.hyper_w1.forward(state_t))
        if not isinstance(w1_flat, list):
            w1_flat = [w1_flat]
        w1 = [abs(x) for x in w1_flat]

        b1_flat = _to_flat(self.hyper_b1.forward(state_t))
        if not isinstance(b1_flat, list):
            b1_flat = [b1_flat]

        # agent_qs: [q1, q2, ...] -> matmul with w1 reshaped as (n_agents, mixing_hidden)
        hidden = [0.0] * self.mixing_hidden
        for j in range(self.mixing_hidden):
            val = b1_flat[j]
            for i in range(self.n_agents):
                val += agent_qs[i] * w1[i * self.mixing_hidden + j]
            hidden[j] = max(0.0, val)  # ReLU

        # Second layer (non-negative via abs)
        w2_flat = _to_flat(self.hyper_w2.forward(state_t))
        if not isinstance(w2_flat, list):
            w2_flat = [w2_flat]
        w2 = [abs(x) for x in w2_flat]

        b2_flat = _to_flat(self.hyper_b2.forward(state_t))
        if not isinstance(b2_flat, list):
            b2_flat = [b2_flat]

        q_tot = b2_flat[0]
        for j in range(self.mixing_hidden):
            q_tot += hidden[j] * w2[j]

        return q_tot

    def select_actions(self, observations):
        q_values = self._get_q_values(observations)
        actions = []
        for i in range(self.n_agents):
            if self.rng.random() < self.epsilon:
                actions.append(int(self.rng.random() * self.action_size))
            else:
                actions.append(_argmax(q_values[i]))
        return actions

    def get_greedy_actions(self, observations):
        q_values = self._get_q_values(observations)
        return [_argmax(q) for q in q_values]

    def store(self, observations, actions, rewards, next_observations, done, state=None, next_state=None):
        if state is None:
            state = _flatten_obs(observations)
        if next_state is None:
            next_state = _flatten_obs(next_observations)

        entry = (observations, actions, rewards, next_observations, done, state, next_state)
        if len(self.buffer) < self.buffer_capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self.buffer_pos] = entry
        self.buffer_pos = (self.buffer_pos + 1) % self.buffer_capacity

    def update(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return 0.0

        # Sample batch
        indices = [int(self.rng.random() * len(self.buffer)) for _ in range(batch_size)]

        total_loss = 0.0
        for idx in indices:
            obs, acts, rews, next_obs, done, state, next_state = self.buffer[idx]

            # Current Q-values for chosen actions
            q_values = self._get_q_values(obs)
            chosen_qs = [q_values[i][acts[i]] for i in range(self.n_agents)]
            q_tot = self._mix(chosen_qs, state)

            # Target Q-values (max over actions)
            next_q_values = self._get_q_values(next_obs)
            max_next_qs = [max(nq) for nq in next_q_values]

            if done:
                target = sum(rews) / self.n_agents
            else:
                target = sum(rews) / self.n_agents + self.gamma * self._mix(max_next_qs, next_state)

            # Loss
            loss = (q_tot - target) ** 2
            total_loss += loss

            # Simplified gradient update: nudge Q-values toward target
            error = q_tot - target
            lr = 0.001
            for i in range(self.n_agents):
                obs_t = _to_tensor_2d(obs[i])
                q = self.q_nets[i].forward(obs_t)
                q_flat = _to_flat(q)
                if not isinstance(q_flat, list):
                    q_flat = [q_flat]

                grad = [0.0] * self.action_size
                grad[acts[i]] = error * lr
                grad_t = _to_tensor_2d(grad)
                self.q_nets[i].backward(grad_t)
                self.optimizer.step(self.q_nets[i].get_trainable_layers())
                self.q_nets[i]
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return total_loss / batch_size


# ============================================================
# SelfPlay -- competitive training
# ============================================================

class SelfPlay:
    """Self-play training for competitive two-player games.

    Maintains a population of agents. Each training step:
    1. Sample two agents (possibly the same)
    2. Play a game
    3. Update both based on outcomes

    Supports:
    - Latest vs latest (simple self-play)
    - Latest vs historical (fictitious self-play)
    - Population-based (sample from pool)
    """

    def __init__(self, obs_size, action_size, population_size=5,
                 hidden_size=32, lr=0.01, mode='population', seed=42):
        self.obs_size = obs_size
        self.action_size = action_size
        self.population_size = population_size
        self.mode = mode
        self.rng = _RNG(seed)

        # Population of policy networks
        self.population = []
        for _ in range(population_size):
            net = Sequential([
                Dense(obs_size, hidden_size, init='xavier'),
                Activation('relu'),
                Dense(hidden_size, action_size, init='xavier'),
            ])
            self.population.append(net)

        self.optimizers = [Adam(lr=lr) for _ in range(population_size)]
        self.elo_ratings = [1000.0] * population_size
        self.win_counts = [0] * population_size
        self.game_counts = [0] * population_size
        self.generation = 0

    def select_action(self, agent_idx, observation, explore=True):
        obs_t = _to_tensor_2d(observation)
        logits = self.population[agent_idx].forward(obs_t)
        logits_flat = _to_flat(logits)
        if not isinstance(logits_flat, list):
            logits_flat = [logits_flat]
        probs = _softmax(logits_flat)
        if explore:
            return _sample_categorical(probs, self.rng)
        return _argmax(probs)

    def select_opponents(self):
        """Select two agent indices to play against each other."""
        if self.mode == 'latest':
            return self.population_size - 1, self.population_size - 1
        elif self.mode == 'latest_vs_historical':
            latest = self.population_size - 1
            opponent = int(self.rng.random() * self.population_size)
            return latest, opponent
        else:  # population
            i = int(self.rng.random() * self.population_size)
            j = int(self.rng.random() * self.population_size)
            return i, j

    def update_agent(self, agent_idx, observations, actions, reward):
        """REINFORCE-style update for one agent."""
        net = self.population[agent_idx]
        opt = self.optimizers[agent_idx]

        for obs, action in zip(observations, actions):
            obs_t = _to_tensor_2d(obs)
            logits = net.forward(obs_t)
            logits_flat = _to_flat(logits)
            if not isinstance(logits_flat, list):
                logits_flat = [logits_flat]
            probs = _softmax(logits_flat)

            grad = [0.0] * self.action_size
            for a in range(self.action_size):
                if a == action:
                    grad[a] = -reward * (1.0 - probs[a])
                else:
                    grad[a] = reward * probs[a]

            grad_t = _to_tensor_2d(grad)
            net.backward(grad_t)
            opt.step(net.get_trainable_layers())
            net
    def update_elo(self, winner_idx, loser_idx, draw=False):
        """Update Elo ratings after a game."""
        K = 32.0
        expected_w = 1.0 / (1.0 + 10 ** ((self.elo_ratings[loser_idx] - self.elo_ratings[winner_idx]) / 400.0))
        expected_l = 1.0 - expected_w

        if draw:
            self.elo_ratings[winner_idx] += K * (0.5 - expected_w)
            self.elo_ratings[loser_idx] += K * (0.5 - expected_l)
        else:
            self.elo_ratings[winner_idx] += K * (1.0 - expected_w)
            self.elo_ratings[loser_idx] += K * (0.0 - expected_l)

        self.game_counts[winner_idx] += 1
        self.game_counts[loser_idx] += 1
        if not draw:
            self.win_counts[winner_idx] += 1

    def play_game(self, env, agent_i, agent_j):
        """Play one game between two agents. Returns (reward_i, reward_j, history)."""
        obs = env.reset()
        total_rewards = [0.0, 0.0]
        history_i, history_j = [], []
        actions_i, actions_j = [], []

        done = False
        while not done:
            a_i = self.select_action(agent_i, obs[0])
            a_j = self.select_action(agent_j, obs[1])

            history_i.append(obs[0])
            history_j.append(obs[1])
            actions_i.append(a_i)
            actions_j.append(a_j)

            obs, rewards, done, info = env.step([a_i, a_j])
            total_rewards[0] += rewards[0]
            total_rewards[1] += rewards[1]

        return total_rewards, (history_i, actions_i), (history_j, actions_j)

    def train_step(self, env):
        """One training step: select opponents, play, update."""
        agent_i, agent_j = self.select_opponents()
        rewards, hist_i, hist_j = self.play_game(env, agent_i, agent_j)

        # Update both agents
        self.update_agent(agent_i, hist_i[0], hist_i[1], rewards[0])
        if agent_i != agent_j:
            self.update_agent(agent_j, hist_j[0], hist_j[1], rewards[1])

        # Update Elo
        if rewards[0] > rewards[1]:
            self.update_elo(agent_i, agent_j)
        elif rewards[1] > rewards[0]:
            self.update_elo(agent_j, agent_i)
        else:
            self.update_elo(agent_i, agent_j, draw=True)

        self.generation += 1
        return rewards


# ============================================================
# TeamReward -- reward shaping for cooperation
# ============================================================

class TeamReward:
    """Reward shaping for cooperative multi-agent settings.

    Modes:
    - 'shared': all agents get average team reward
    - 'difference': counterfactual difference reward (agent's marginal contribution)
    - 'shaped': combination of individual + shared with shaping factor
    """

    def __init__(self, mode='shared', shaping_factor=0.5):
        self.mode = mode
        self.shaping_factor = shaping_factor

    def shape(self, individual_rewards, counterfactual_rewards=None):
        """Shape rewards.

        Args:
            individual_rewards: list of rewards per agent
            counterfactual_rewards: list of team rewards without each agent (for 'difference' mode)

        Returns:
            shaped rewards per agent
        """
        n = len(individual_rewards)
        team_reward = sum(individual_rewards) / n

        if self.mode == 'shared':
            return [team_reward] * n

        elif self.mode == 'difference':
            if counterfactual_rewards is None:
                return individual_rewards
            shaped = []
            for i in range(n):
                # Agent i's contribution = team reward - team reward without agent i
                diff = team_reward - counterfactual_rewards[i]
                shaped.append(diff)
            return shaped

        elif self.mode == 'shaped':
            alpha = self.shaping_factor
            shaped = []
            for i in range(n):
                r = alpha * team_reward + (1 - alpha) * individual_rewards[i]
                shaped.append(r)
            return shaped

        return individual_rewards


# ============================================================
# MATrainer -- multi-agent training loop
# ============================================================

class MATrainer:
    """Multi-agent training loop.

    Supports training with:
    - IndependentLearners (IQL)
    - CentralizedCritic (CTDE)
    - QMIX

    Tracks per-agent and team metrics.
    """

    def __init__(self, env, agents, reward_shaper=None, seed=42):
        self.env = env
        self.agents = agents
        self.reward_shaper = reward_shaper
        self.rng = _RNG(seed)

        self.episode_rewards = []
        self.episode_lengths = []
        self.per_agent_rewards = [[] for _ in range(env.n_agents)]

    def train_episode(self):
        """Run one episode and update agents."""
        obs = self.env.reset()
        total_rewards = [0.0] * self.env.n_agents
        steps = 0
        done = False

        while not done:
            if isinstance(self.agents, IndependentLearners):
                actions = self.agents.select_actions(obs)
            elif isinstance(self.agents, CentralizedCritic):
                actions = self.agents.select_actions(obs, explore=True)
            elif isinstance(self.agents, QMIX):
                actions = self.agents.select_actions(obs)
            else:
                actions = [0] * self.env.n_agents

            next_obs, rewards, done, info = self.env.step(actions)

            # Shape rewards if shaper provided
            if self.reward_shaper:
                rewards = self.reward_shaper.shape(rewards)

            # Update agents
            if isinstance(self.agents, IndependentLearners):
                self.agents.update(obs, actions, rewards, next_obs, done)
            elif isinstance(self.agents, CentralizedCritic):
                next_actions = self.agents.select_actions(next_obs, explore=False)
                self.agents.update(obs, actions, rewards, next_obs, next_actions, done)
            elif isinstance(self.agents, QMIX):
                self.agents.store(obs, actions, rewards, next_obs, done)
                self.agents.update(batch_size=16)

            for i in range(self.env.n_agents):
                total_rewards[i] += rewards[i]

            obs = next_obs
            steps += 1

        self.episode_rewards.append(sum(total_rewards) / self.env.n_agents)
        self.episode_lengths.append(steps)
        for i in range(self.env.n_agents):
            self.per_agent_rewards[i].append(total_rewards[i])

        return total_rewards, steps

    def train(self, n_episodes=100):
        """Train for n episodes."""
        results = []
        for ep in range(n_episodes):
            rewards, steps = self.train_episode()
            results.append({'episode': ep, 'rewards': rewards, 'steps': steps})
        return results

    def evaluate(self, n_episodes=10):
        """Evaluate without exploration."""
        total = [0.0] * self.env.n_agents
        total_steps = 0

        for _ in range(n_episodes):
            obs = self.env.reset()
            done = False
            steps = 0

            while not done:
                if isinstance(self.agents, IndependentLearners):
                    actions = self.agents.get_greedy_actions(obs)
                elif isinstance(self.agents, CentralizedCritic):
                    actions = self.agents.select_actions(obs, explore=False)
                elif isinstance(self.agents, QMIX):
                    actions = self.agents.get_greedy_actions(obs)
                else:
                    actions = [0] * self.env.n_agents

                obs, rewards, done, info = self.env.step(actions)
                for i in range(self.env.n_agents):
                    total[i] += rewards[i]
                steps += 1
            total_steps += steps

        avg = [t / n_episodes for t in total]
        return {
            'avg_rewards': avg,
            'avg_team_reward': sum(avg) / self.env.n_agents,
            'avg_steps': total_steps / n_episodes,
        }

    def get_metrics(self):
        return {
            'episodes': len(self.episode_rewards),
            'avg_team_reward': sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards)) if self.episode_rewards else 0,
            'avg_length': sum(self.episode_lengths[-10:]) / min(10, len(self.episode_lengths)) if self.episode_lengths else 0,
        }


# ============================================================
# TournamentEvaluator -- round-robin evaluation
# ============================================================

class TournamentEvaluator:
    """Round-robin tournament for evaluating agent populations.

    Plays each pair of agents multiple times and tracks:
    - Win/loss/draw record
    - Elo ratings
    - Nash equilibrium approximation (for matrix games)
    """

    def __init__(self, env, seed=42):
        self.env = env
        self.rng = _RNG(seed)
        self.results = {}

    def run_match(self, agent_a_fn, agent_b_fn, n_games=10):
        """Run n_games between two agent action functions.

        agent_fn: callable(observation) -> action
        Returns: (wins_a, wins_b, draws, avg_reward_a, avg_reward_b)
        """
        wins_a, wins_b, draws = 0, 0, 0
        total_r_a, total_r_b = 0.0, 0.0

        for _ in range(n_games):
            obs = self.env.reset()
            done = False
            ep_r_a, ep_r_b = 0.0, 0.0

            while not done:
                a_a = agent_a_fn(obs[0])
                a_b = agent_b_fn(obs[1])
                obs, rewards, done, info = self.env.step([a_a, a_b])
                ep_r_a += rewards[0]
                ep_r_b += rewards[1]

            total_r_a += ep_r_a
            total_r_b += ep_r_b
            if ep_r_a > ep_r_b:
                wins_a += 1
            elif ep_r_b > ep_r_a:
                wins_b += 1
            else:
                draws += 1

        return wins_a, wins_b, draws, total_r_a / n_games, total_r_b / n_games

    def round_robin(self, agent_fns, names=None, n_games=10):
        """Run all-vs-all tournament.

        agent_fns: list of callable(observation) -> action
        Returns: standings dict
        """
        n = len(agent_fns)
        if names is None:
            names = [f'Agent_{i}' for i in range(n)]

        standings = {name: {'wins': 0, 'losses': 0, 'draws': 0, 'total_reward': 0.0, 'games': 0}
                     for name in names}
        elo = {name: 1000.0 for name in names}

        for i in range(n):
            for j in range(i + 1, n):
                wa, wb, d, ra, rb = self.run_match(agent_fns[i], agent_fns[j], n_games)

                standings[names[i]]['wins'] += wa
                standings[names[i]]['losses'] += wb
                standings[names[i]]['draws'] += d
                standings[names[i]]['total_reward'] += ra * n_games
                standings[names[i]]['games'] += n_games

                standings[names[j]]['wins'] += wb
                standings[names[j]]['losses'] += wa
                standings[names[j]]['draws'] += d
                standings[names[j]]['total_reward'] += rb * n_games
                standings[names[j]]['games'] += n_games

                # Elo update
                K = 32.0
                exp_i = 1.0 / (1.0 + 10 ** ((elo[names[j]] - elo[names[i]]) / 400.0))
                score_i = (wa + d * 0.5) / n_games
                elo[names[i]] += K * (score_i - exp_i)
                elo[names[j]] += K * ((1.0 - score_i) - (1.0 - exp_i))

        self.results = {'standings': standings, 'elo': elo}
        return self.results

    def get_rankings(self):
        """Get agents sorted by Elo rating."""
        if not self.results:
            return []
        elo = self.results['elo']
        return sorted(elo.items(), key=lambda x: x[1], reverse=True)
