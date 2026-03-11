"""
C180: Model-Based Reinforcement Learning
Composes C179 (Reinforcement Learning) + C140 (Neural Network)

Components:
- WorldModel: learns environment dynamics (state, action) -> (next_state, reward)
- RewardModel: standalone reward predictor
- ModelEnsemble: ensemble of world models for uncertainty estimation
- DynaTabular: classic Dyna-Q with tabular model
- DynaDeep: deep Dyna with neural world model + DQN
- MPC: Model Predictive Control (random shooting + CEM)
- MBPO: Model-Based Policy Optimization (short rollouts from real states)
- LatentWorldModel: learns compressed latent dynamics
- DreamerAgent: actor-critic in latent imagination
- ModelBasedTrainer: unified training loop
- DataBuffer: transition storage for model training
- PlanningMetrics: model accuracy and planning quality tracking
"""

import sys
import os
import math
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C179_reinforcement_learning'))

from neural_network import (
    Tensor, Dense, Activation, Sequential, MSELoss, SGD, Adam,
    relu, sigmoid, tanh_act, softmax, softmax_batch,
    xavier_init, he_init, train_step, save_weights, load_weights,
)
from reinforcement_learning import (
    Environment, GridWorld, CartPole, Bandit, MountainCar,
    ReplayBuffer, EpsilonGreedy, QLearning, DQN, DoubleDQN,
    DQNTrainer, DiscountedReturn, ReturnTracker, _RNG,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor_2d(data):
    """Convert list of lists or flat list to 2D Tensor."""
    if isinstance(data, Tensor):
        if len(data.shape) == 1:
            return Tensor([data.data])
        return data
    if isinstance(data, list):
        if len(data) == 0:
            return Tensor([[]])
        if isinstance(data[0], (int, float)):
            return Tensor([data])
        return Tensor(data)
    return Tensor([[data]])


def _to_flat(tensor_or_list):
    """Extract flat list from Tensor or list."""
    if isinstance(tensor_or_list, Tensor):
        if len(tensor_or_list.shape) == 2:
            return tensor_or_list.data[0] if tensor_or_list.shape[0] == 1 else tensor_or_list.data
        return tensor_or_list.data
    return tensor_or_list


def _one_hot_action(action, n_actions):
    """One-hot encode an action."""
    oh = [0.0] * n_actions
    oh[action] = 1.0
    return oh


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# DataBuffer -- stores transitions for model training
# ---------------------------------------------------------------------------

class DataBuffer:
    """Stores (state, action, reward, next_state, done) transitions."""

    def __init__(self, capacity=50000, seed=42):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.rng = _RNG(seed)

    def push(self, state, action, reward, next_state, done):
        entry = (list(state), action, reward, list(next_state), done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self.pos] = entry
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        indices = []
        for _ in range(batch_size):
            indices.append(self.rng.randint(0, len(self.buffer) - 1))
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            s, a, r, ns, d = self.buffer[i]
            states.append(list(s))
            actions.append(a)
            rewards.append(r)
            next_states.append(list(ns))
            dones.append(d)
        return states, actions, rewards, next_states, dones

    def sample_all(self):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for s, a, r, ns, d in self.buffer:
            states.append(list(s))
            actions.append(a)
            rewards.append(r)
            next_states.append(list(ns))
            dones.append(d)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# WorldModel -- neural network dynamics model
# ---------------------------------------------------------------------------

class WorldModel:
    """Learns (state, action) -> (next_state, reward).

    Input: concatenation of state and one-hot action.
    Output: predicted next_state (state_size dims) + reward (1 dim).
    """

    def __init__(self, state_size, action_size, hidden_sizes=None,
                 lr=0.001, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.input_size = state_size + action_size
        self.output_size = state_size + 1  # next_state + reward
        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        self.rng = random.Random(seed)

        self.net = Sequential()
        prev = self.input_size
        for h in hidden_sizes:
            self.net.add(Dense(prev, h, init='he', rng=self.rng))
            self.net.add(Activation('relu'))
            prev = h
        self.net.add(Dense(prev, self.output_size, init='xavier', rng=self.rng))

        self.optimizer = Adam(lr=lr)
        self.loss_fn = MSELoss()
        self.train_losses = []

    def predict(self, state, action):
        """Predict (next_state, reward) from (state, action)."""
        inp = list(state) + _one_hot_action(action, self.action_size)
        self.net.eval()
        out = self.net.forward(Tensor([inp]))
        out_flat = out.data[0] if len(out.shape) == 2 else out.data
        next_state = out_flat[:self.state_size]
        reward = out_flat[self.state_size]
        return next_state, reward

    def predict_batch(self, states, actions):
        """Batch prediction."""
        inputs = []
        for s, a in zip(states, actions):
            inputs.append(list(s) + _one_hot_action(a, self.action_size))
        self.net.eval()
        out = self.net.forward(Tensor(inputs))
        pred_states = []
        pred_rewards = []
        for i in range(len(states)):
            row = out.data[i] if len(out.shape) == 2 else out.data
            pred_states.append(row[:self.state_size])
            pred_rewards.append(row[self.state_size])
        return pred_states, pred_rewards

    def train_batch(self, states, actions, next_states, rewards):
        """Train on a batch of transitions. Returns loss."""
        inputs = []
        targets = []
        for s, a, ns, r in zip(states, actions, next_states, rewards):
            inputs.append(list(s) + _one_hot_action(a, self.action_size))
            targets.append(list(ns) + [r])

        x = Tensor(inputs)
        y = Tensor(targets)

        self.net.train()
        pred = self.net.forward(x)
        loss = self.loss_fn.forward(pred, y)
        grad = self.loss_fn.backward(pred, y)
        self.net.backward(grad)
        layers = self.net.get_trainable_layers()
        self.optimizer.step(layers)
        self.train_losses.append(loss)
        return loss

    def get_accuracy(self, states, actions, next_states, rewards):
        """Compute prediction MSE on a batch."""
        pred_states, pred_rewards = self.predict_batch(states, actions)
        state_mse = 0.0
        reward_mse = 0.0
        n = len(states)
        for i in range(n):
            for j in range(self.state_size):
                state_mse += (pred_states[i][j] - next_states[i][j]) ** 2
            reward_mse += (pred_rewards[i] - rewards[i]) ** 2
        state_mse /= max(n * self.state_size, 1)
        reward_mse /= max(n, 1)
        return {'state_mse': state_mse, 'reward_mse': reward_mse}


# ---------------------------------------------------------------------------
# RewardModel -- standalone reward predictor
# ---------------------------------------------------------------------------

class RewardModel:
    """Learns (state, action) -> reward."""

    def __init__(self, state_size, action_size, hidden_sizes=None,
                 lr=0.001, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.input_size = state_size + action_size
        if hidden_sizes is None:
            hidden_sizes = [32, 32]
        rng = random.Random(seed)

        self.net = Sequential()
        prev = self.input_size
        for h in hidden_sizes:
            self.net.add(Dense(prev, h, init='he', rng=rng))
            self.net.add(Activation('relu'))
            prev = h
        self.net.add(Dense(prev, 1, init='xavier', rng=rng))

        self.optimizer = Adam(lr=lr)
        self.loss_fn = MSELoss()

    def predict(self, state, action):
        inp = list(state) + _one_hot_action(action, self.action_size)
        self.net.eval()
        out = self.net.forward(Tensor([inp]))
        val = out.data[0][0] if len(out.shape) == 2 else out.data[0]
        return val

    def train_batch(self, states, actions, rewards):
        inputs = []
        targets = []
        for s, a, r in zip(states, actions, rewards):
            inputs.append(list(s) + _one_hot_action(a, self.action_size))
            targets.append([r])
        x = Tensor(inputs)
        y = Tensor(targets)
        self.net.train()
        pred = self.net.forward(x)
        loss = self.loss_fn.forward(pred, y)
        grad = self.loss_fn.backward(pred, y)
        self.net.backward(grad)
        self.optimizer.step(self.net.get_trainable_layers())
        return loss


# ---------------------------------------------------------------------------
# ModelEnsemble -- ensemble of world models for uncertainty
# ---------------------------------------------------------------------------

class ModelEnsemble:
    """Ensemble of WorldModels for epistemic uncertainty estimation."""

    def __init__(self, state_size, action_size, n_models=5,
                 hidden_sizes=None, lr=0.001, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.n_models = n_models
        self.models = []
        for i in range(n_models):
            m = WorldModel(state_size, action_size, hidden_sizes=hidden_sizes,
                           lr=lr, seed=seed + i)
            self.models.append(m)
        self.rng = _RNG(seed)

    def predict(self, state, action):
        """Predict using random model from ensemble."""
        idx = self.rng.randint(0, self.n_models - 1)
        return self.models[idx].predict(state, action)

    def predict_mean(self, state, action):
        """Predict using ensemble mean."""
        all_ns = []
        all_r = []
        for m in self.models:
            ns, r = m.predict(state, action)
            all_ns.append(ns)
            all_r.append(r)
        mean_ns = [0.0] * self.state_size
        mean_r = 0.0
        for ns, r in zip(all_ns, all_r):
            for j in range(self.state_size):
                mean_ns[j] += ns[j]
            mean_r += r
        mean_ns = [x / self.n_models for x in mean_ns]
        mean_r /= self.n_models
        return mean_ns, mean_r

    def get_uncertainty(self, state, action):
        """Compute disagreement (variance) across ensemble predictions."""
        all_ns = []
        all_r = []
        for m in self.models:
            ns, r = m.predict(state, action)
            all_ns.append(ns)
            all_r.append(r)
        # Variance of next_state predictions
        mean_ns = [0.0] * self.state_size
        for ns in all_ns:
            for j in range(self.state_size):
                mean_ns[j] += ns[j]
        mean_ns = [x / self.n_models for x in mean_ns]
        var = 0.0
        for ns in all_ns:
            for j in range(self.state_size):
                var += (ns[j] - mean_ns[j]) ** 2
        var /= max(self.n_models * self.state_size, 1)
        # Reward variance
        mean_r = sum(all_r) / self.n_models
        r_var = sum((r - mean_r) ** 2 for r in all_r) / self.n_models
        return {'state_variance': var, 'reward_variance': r_var,
                'total_uncertainty': var + r_var}

    def train_batch(self, states, actions, next_states, rewards):
        """Train all models. Each sees the full batch (could bootstrap)."""
        losses = []
        for m in self.models:
            loss = m.train_batch(states, actions, next_states, rewards)
            losses.append(loss)
        return sum(losses) / len(losses)

    def train_bootstrapped(self, states, actions, next_states, rewards):
        """Train each model on a bootstrap sample of the batch."""
        n = len(states)
        losses = []
        for m in self.models:
            indices = [self.rng.randint(0, n - 1) for _ in range(n)]
            bs = [states[i] for i in indices]
            ba = [actions[i] for i in indices]
            bns = [next_states[i] for i in indices]
            br = [rewards[i] for i in indices]
            loss = m.train_batch(bs, ba, bns, br)
            losses.append(loss)
        return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# DynaTabular -- classic Dyna-Q
# ---------------------------------------------------------------------------

class DynaTabular:
    """Dyna-Q: tabular Q-learning + tabular model + planning.

    After each real step, does k planning steps using the learned model.
    """

    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99,
                 planning_steps=5, seed=42):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.planning_steps = planning_steps
        self.rng = _RNG(seed)

        # Q-table
        self.q = [[0.0] * n_actions for _ in range(n_states)]

        # Tabular model: model[s][a] = (next_s, reward, done)
        self.model = {}
        # Track visited (state, action) pairs for planning
        self.visited = []

    def get_action(self, state, epsilon=0.1):
        if self.rng.random() < epsilon:
            return self.rng.randint(0, self.n_actions - 1)
        q_vals = self.q[state]
        best = max(q_vals)
        ties = [a for a in range(self.n_actions) if q_vals[a] == best]
        return ties[self.rng.randint(0, len(ties) - 1)]

    def update(self, state, action, reward, next_state, done):
        """Update Q from real experience and learn model."""
        # Q-learning update
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.q[next_state])
        self.q[state][action] += self.lr * (target - self.q[state][action])

        # Update model
        if state not in self.model:
            self.model[state] = {}
        self.model[state][action] = (next_state, reward, done)
        if (state, action) not in self.visited:
            self.visited.append((state, action))

        # Planning
        self._plan()

        return target - self.q[state][action]  # approximate td_error

    def _plan(self):
        """Simulate transitions using learned model."""
        if not self.visited:
            return
        for _ in range(self.planning_steps):
            idx = self.rng.randint(0, len(self.visited) - 1)
            s, a = self.visited[idx]
            ns, r, d = self.model[s][a]
            if d:
                target = r
            else:
                target = r + self.gamma * max(self.q[ns])
            self.q[s][a] += self.lr * (target - self.q[s][a])

    def get_q_values(self, state):
        return list(self.q[state])

    def get_policy(self):
        policy = []
        for s in range(self.n_states):
            best = max(self.q[s])
            for a in range(self.n_actions):
                if self.q[s][a] == best:
                    policy.append(a)
                    break
        return policy


# ---------------------------------------------------------------------------
# DynaDeep -- deep Dyna with neural world model
# ---------------------------------------------------------------------------

class DynaDeep:
    """Deep Dyna: DQN agent + neural WorldModel + simulated planning.

    After each real step, trains the world model, then does k simulated
    rollouts from recently visited states.
    """

    def __init__(self, state_size, action_size, hidden_sizes=None,
                 lr=0.001, model_lr=0.001, gamma=0.99, tau=0.01,
                 planning_steps=5, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.planning_steps = planning_steps
        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        self.dqn = DQN(state_size, action_size, hidden_sizes=hidden_sizes,
                       lr=lr, gamma=gamma, tau=tau, seed=seed)
        self.world_model = WorldModel(state_size, action_size,
                                      hidden_sizes=hidden_sizes,
                                      lr=model_lr, seed=seed + 100)
        self.buffer = DataBuffer(capacity=10000, seed=seed)
        self.rng = _RNG(seed)
        self.recent_states = []
        self.max_recent = 1000

    def get_action(self, state, epsilon=0.0):
        return self.dqn.get_action(state, epsilon)

    def get_q_values(self, state):
        return self.dqn.get_q_values(state)

    def update(self, state, action, reward, next_state, done, batch_size=32):
        """Update from real experience, train model, plan."""
        self.buffer.push(state, action, reward, next_state, done)

        # Store recent state for planning
        self.recent_states.append(list(state))
        if len(self.recent_states) > self.max_recent:
            self.recent_states.pop(0)

        model_loss = None
        dqn_loss = None

        if len(self.buffer) >= batch_size:
            # Train world model
            s, a, r, ns, d = self.buffer.sample(batch_size)
            model_loss = self.world_model.train_batch(s, a, ns, r)

            # Train DQN on real data
            dqn_loss = self.dqn.train_batch(s, a, r, ns, d)
            self.dqn.soft_update()

            # Planning: simulate from recent states
            self._plan(batch_size)

        return {'model_loss': model_loss, 'dqn_loss': dqn_loss}

    def _plan(self, batch_size):
        """Generate simulated experience and train DQN on it."""
        if not self.recent_states:
            return
        sim_s, sim_a, sim_r, sim_ns, sim_d = [], [], [], [], []
        for _ in range(self.planning_steps):
            idx = self.rng.randint(0, len(self.recent_states) - 1)
            s = self.recent_states[idx]
            a = self.rng.randint(0, self.action_size - 1)
            ns, r = self.world_model.predict(s, a)
            sim_s.append(s)
            sim_a.append(a)
            sim_r.append(r)
            sim_ns.append(ns)
            sim_d.append(False)  # assume not done in imagination

        if sim_s:
            self.dqn.train_batch(sim_s, sim_a, sim_r, sim_ns, sim_d)


# ---------------------------------------------------------------------------
# MPC -- Model Predictive Control
# ---------------------------------------------------------------------------

class MPC:
    """Model Predictive Control using random shooting or CEM.

    Plans over the world model to find the best action sequence.
    """

    def __init__(self, world_model, action_size, horizon=10,
                 n_candidates=100, n_elite=10, cem_iterations=3,
                 gamma=0.99, method='random_shooting', seed=42):
        self.world_model = world_model
        self.action_size = action_size
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.n_elite = n_elite
        self.cem_iterations = cem_iterations
        self.gamma = gamma
        self.method = method
        self.rng = _RNG(seed)

    def get_action(self, state):
        """Plan and return best first action."""
        if self.method == 'cem':
            return self._cem_plan(state)
        return self._random_shooting(state)

    def _random_shooting(self, state):
        """Evaluate random action sequences, pick best."""
        best_return = float('-inf')
        best_action = 0
        for _ in range(self.n_candidates):
            actions = [self.rng.randint(0, self.action_size - 1)
                       for _ in range(self.horizon)]
            ret = self._simulate(state, actions)
            if ret > best_return:
                best_return = ret
                best_action = actions[0]
        return best_action

    def _cem_plan(self, state):
        """Cross-Entropy Method: iteratively refine action distribution."""
        # Initialize uniform distribution over actions
        # For discrete actions: maintain probability for each action at each step
        probs = [[1.0 / self.action_size] * self.action_size
                 for _ in range(self.horizon)]

        for _ in range(self.cem_iterations):
            # Sample candidates
            candidates = []
            returns = []
            for _ in range(self.n_candidates):
                actions = []
                for t in range(self.horizon):
                    r = self.rng.random()
                    cum = 0.0
                    chosen = self.action_size - 1
                    for a in range(self.action_size):
                        cum += probs[t][a]
                        if r < cum:
                            chosen = a
                            break
                    actions.append(chosen)
                ret = self._simulate(state, actions)
                candidates.append(actions)
                returns.append(ret)

            # Select elite
            indexed = sorted(enumerate(returns), key=lambda x: -x[1])
            elite_indices = [indexed[i][0] for i in range(min(self.n_elite, len(indexed)))]
            elite = [candidates[i] for i in elite_indices]

            # Update distribution from elite
            for t in range(self.horizon):
                counts = [0.0] * self.action_size
                for seq in elite:
                    counts[seq[t]] += 1.0
                total = sum(counts)
                if total > 0:
                    probs[t] = [c / total for c in counts]

        # Sample from final distribution
        r = self.rng.random()
        cum = 0.0
        best_action = self.action_size - 1
        for a in range(self.action_size):
            cum += probs[0][a]
            if r < cum:
                best_action = a
                break
        return best_action

    def _simulate(self, start_state, actions):
        """Simulate action sequence through world model, return total discounted reward."""
        state = list(start_state)
        total = 0.0
        discount = 1.0
        for a in actions:
            ns, r = self.world_model.predict(state, a)
            total += discount * r
            discount *= self.gamma
            state = ns
        return total


# ---------------------------------------------------------------------------
# MBPO -- Model-Based Policy Optimization
# ---------------------------------------------------------------------------

class MBPO:
    """Model-Based Policy Optimization.

    Uses short model rollouts from real states to augment the replay buffer.
    Trains a DQN on the augmented buffer.
    """

    def __init__(self, state_size, action_size, hidden_sizes=None,
                 lr=0.001, model_lr=0.001, gamma=0.99, tau=0.01,
                 rollout_length=1, rollout_batch=10,
                 model_train_freq=250, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.rollout_length = rollout_length
        self.rollout_batch = rollout_batch
        self.model_train_freq = model_train_freq
        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        self.ensemble = ModelEnsemble(state_size, action_size, n_models=3,
                                      hidden_sizes=hidden_sizes,
                                      lr=model_lr, seed=seed)
        self.dqn = DQN(state_size, action_size, hidden_sizes=hidden_sizes,
                       lr=lr, gamma=gamma, tau=tau, seed=seed)

        self.real_buffer = DataBuffer(capacity=10000, seed=seed)
        self.model_buffer = DataBuffer(capacity=50000, seed=seed + 1)
        self.rng = _RNG(seed)
        self.step_count = 0

    def get_action(self, state, epsilon=0.0):
        return self.dqn.get_action(state, epsilon)

    def update(self, state, action, reward, next_state, done, batch_size=32):
        """Store real experience, periodically train model, generate rollouts."""
        self.real_buffer.push(state, action, reward, next_state, done)
        self.step_count += 1

        # Periodically train ensemble model
        if self.step_count % self.model_train_freq == 0 and len(self.real_buffer) >= batch_size:
            s, a, r, ns, d = self.real_buffer.sample(min(batch_size * 4, len(self.real_buffer)))
            self.ensemble.train_bootstrapped(s, a, ns, r)

        # Generate model rollouts from real states
        if len(self.real_buffer) >= batch_size:
            self._generate_rollouts()

        # Train DQN on mix of real + model data
        dqn_loss = None
        total_data = len(self.real_buffer) + len(self.model_buffer)
        if total_data >= batch_size:
            # Sample from both buffers
            real_ratio = 0.5
            real_bs = max(1, int(batch_size * real_ratio))
            model_bs = batch_size - real_bs

            s, a, r, ns, d = [], [], [], [], []
            if len(self.real_buffer) >= real_bs:
                rs, ra, rr, rns, rd = self.real_buffer.sample(real_bs)
                s.extend(rs); a.extend(ra); r.extend(rr)
                ns.extend(rns); d.extend(rd)
            if len(self.model_buffer) >= model_bs and model_bs > 0:
                ms, ma, mr, mns, md = self.model_buffer.sample(model_bs)
                s.extend(ms); a.extend(ma); r.extend(mr)
                ns.extend(mns); d.extend(md)

            if s:
                dqn_loss = self.dqn.train_batch(s, a, r, ns, d)
                self.dqn.soft_update()

        return dqn_loss

    def _generate_rollouts(self):
        """Generate short rollouts from real states using ensemble."""
        states, _, _, _, _ = self.real_buffer.sample(self.rollout_batch)
        for start_state in states:
            state = list(start_state)
            for _ in range(self.rollout_length):
                action = self.dqn.get_action(state, epsilon=0.1)
                ns, r = self.ensemble.predict(state, action)
                self.model_buffer.push(state, action, r, ns, False)
                state = ns


# ---------------------------------------------------------------------------
# LatentWorldModel -- learns dynamics in a compressed latent space
# ---------------------------------------------------------------------------

class LatentWorldModel:
    """World model with encoder/decoder for latent dynamics.

    encoder: state -> latent
    dynamics: (latent, action) -> next_latent
    decoder: latent -> state
    reward_head: (latent, action) -> reward
    """

    def __init__(self, state_size, action_size, latent_size=32,
                 hidden_sizes=None, lr=0.001, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.latent_size = latent_size
        if hidden_sizes is None:
            hidden_sizes = [64]
        rng = random.Random(seed)

        # Encoder: state -> latent
        self.encoder = Sequential()
        prev = state_size
        for h in hidden_sizes:
            self.encoder.add(Dense(prev, h, init='he', rng=rng))
            self.encoder.add(Activation('relu'))
            prev = h
        self.encoder.add(Dense(prev, latent_size, init='xavier', rng=rng))
        self.encoder.add(Activation('tanh'))

        # Dynamics: (latent, action) -> next_latent
        self.dynamics = Sequential()
        prev = latent_size + action_size
        for h in hidden_sizes:
            self.dynamics.add(Dense(prev, h, init='he', rng=rng))
            self.dynamics.add(Activation('relu'))
            prev = h
        self.dynamics.add(Dense(prev, latent_size, init='xavier', rng=rng))
        self.dynamics.add(Activation('tanh'))

        # Decoder: latent -> state
        self.decoder = Sequential()
        prev = latent_size
        for h in hidden_sizes:
            self.decoder.add(Dense(prev, h, init='he', rng=rng))
            self.decoder.add(Activation('relu'))
            prev = h
        self.decoder.add(Dense(prev, state_size, init='xavier', rng=rng))

        # Reward head: (latent, action) -> reward
        self.reward_head = Sequential()
        prev = latent_size + action_size
        for h in hidden_sizes[:1]:
            self.reward_head.add(Dense(prev, h, init='he', rng=rng))
            self.reward_head.add(Activation('relu'))
            prev = h
        self.reward_head.add(Dense(prev, 1, init='xavier', rng=rng))

        self.enc_opt = Adam(lr=lr)
        self.dyn_opt = Adam(lr=lr)
        self.dec_opt = Adam(lr=lr)
        self.rew_opt = Adam(lr=lr)
        self.loss_fn = MSELoss()

    def encode(self, state):
        """Encode state to latent."""
        self.encoder.eval()
        inp = _to_tensor_2d(state)
        out = self.encoder.forward(inp)
        return out.data[0] if len(out.shape) == 2 else out.data

    def decode(self, latent):
        """Decode latent to state."""
        self.decoder.eval()
        inp = _to_tensor_2d(latent)
        out = self.decoder.forward(inp)
        return out.data[0] if len(out.shape) == 2 else out.data

    def predict_latent(self, latent, action):
        """Predict next latent from (latent, action)."""
        self.dynamics.eval()
        inp = list(latent) + _one_hot_action(action, self.action_size)
        out = self.dynamics.forward(Tensor([inp]))
        return out.data[0] if len(out.shape) == 2 else out.data

    def predict_reward(self, latent, action):
        """Predict reward from (latent, action)."""
        self.reward_head.eval()
        inp = list(latent) + _one_hot_action(action, self.action_size)
        out = self.reward_head.forward(Tensor([inp]))
        return out.data[0][0] if len(out.shape) == 2 else out.data[0]

    def predict(self, state, action):
        """Full predict: state -> encode -> dynamics -> decode -> next_state, reward."""
        latent = self.encode(state)
        next_latent = self.predict_latent(latent, action)
        next_state = self.decode(next_latent)
        reward = self.predict_reward(latent, action)
        return next_state, reward

    def train_step(self, states, actions, next_states, rewards):
        """Train all components on a batch."""
        n = len(states)

        # Train encoder + decoder (reconstruction)
        self.encoder.train()
        self.decoder.train()
        enc_in = Tensor(states)
        latents = self.encoder.forward(enc_in)
        recon = self.decoder.forward(latents)
        recon_loss = self.loss_fn.forward(recon, Tensor(states))
        recon_grad = self.loss_fn.backward(recon, Tensor(states))
        dec_grad = self.decoder.backward(recon_grad)
        self.dec_opt.step(self.decoder.get_trainable_layers())
        self.encoder.backward(dec_grad)
        self.enc_opt.step(self.encoder.get_trainable_layers())

        # Train dynamics
        self.encoder.eval()
        self.dynamics.train()
        latents_data = self.encoder.forward(Tensor(states))
        next_latents_target = self.encoder.forward(Tensor(next_states))

        dyn_inputs = []
        for i in range(n):
            lat = latents_data.data[i] if len(latents_data.shape) == 2 else latents_data.data
            dyn_inputs.append(list(lat) + _one_hot_action(actions[i], self.action_size))
        dyn_out = self.dynamics.forward(Tensor(dyn_inputs))
        dyn_loss = self.loss_fn.forward(dyn_out, next_latents_target)
        dyn_grad = self.loss_fn.backward(dyn_out, next_latents_target)
        self.dynamics.backward(dyn_grad)
        self.dyn_opt.step(self.dynamics.get_trainable_layers())

        # Train reward head
        self.reward_head.train()
        rew_inputs = []
        rew_targets = []
        for i in range(n):
            lat = latents_data.data[i] if len(latents_data.shape) == 2 else latents_data.data
            rew_inputs.append(list(lat) + _one_hot_action(actions[i], self.action_size))
            rew_targets.append([rewards[i]])
        rew_out = self.reward_head.forward(Tensor(rew_inputs))
        rew_loss = self.loss_fn.forward(rew_out, Tensor(rew_targets))
        rew_grad = self.loss_fn.backward(rew_out, Tensor(rew_targets))
        self.reward_head.backward(rew_grad)
        self.rew_opt.step(self.reward_head.get_trainable_layers())

        return {'recon_loss': recon_loss, 'dynamics_loss': dyn_loss,
                'reward_loss': rew_loss}


# ---------------------------------------------------------------------------
# DreamerAgent -- actor-critic in latent imagination
# ---------------------------------------------------------------------------

class DreamerAgent:
    """Dreamer-style agent: learns latent world model, then trains
    actor-critic purely in imagination (latent rollouts).
    """

    def __init__(self, state_size, action_size, latent_size=32,
                 hidden_sizes=None, lr=0.001, gamma=0.99,
                 imagination_horizon=5, seed=42):
        self.state_size = state_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.gamma = gamma
        self.imagination_horizon = imagination_horizon
        if hidden_sizes is None:
            hidden_sizes = [64]
        rng = random.Random(seed)

        self.world_model = LatentWorldModel(
            state_size, action_size, latent_size=latent_size,
            hidden_sizes=hidden_sizes, lr=lr, seed=seed)

        # Actor: latent -> action probs
        self.actor = Sequential()
        prev = latent_size
        for h in hidden_sizes:
            self.actor.add(Dense(prev, h, init='he', rng=rng))
            self.actor.add(Activation('relu'))
            prev = h
        self.actor.add(Dense(prev, action_size, init='xavier', rng=rng))
        self.actor.add(Activation('softmax'))

        # Critic: latent -> value
        self.critic = Sequential()
        prev = latent_size
        for h in hidden_sizes:
            self.critic.add(Dense(prev, h, init='he', rng=rng))
            self.critic.add(Activation('relu'))
            prev = h
        self.critic.add(Dense(prev, 1, init='xavier', rng=rng))

        self.actor_opt = Adam(lr=lr)
        self.critic_opt = Adam(lr=lr * 2)
        self.loss_fn = MSELoss()
        self.rng = _RNG(seed)
        self.buffer = DataBuffer(capacity=10000, seed=seed)

    def get_action(self, state):
        """Sample action from actor policy."""
        latent = self.world_model.encode(state)
        self.actor.eval()
        probs_t = self.actor.forward(Tensor([latent]))
        probs = probs_t.data[0] if len(probs_t.shape) == 2 else probs_t.data
        # Sample from distribution
        r = self.rng.random()
        cum = 0.0
        for a in range(self.action_size):
            cum += max(probs[a], 1e-8)
            if r < cum:
                return a
        return self.action_size - 1

    def get_action_probs(self, state):
        latent = self.world_model.encode(state)
        self.actor.eval()
        probs_t = self.actor.forward(Tensor([latent]))
        probs = probs_t.data[0] if len(probs_t.shape) == 2 else probs_t.data
        return probs

    def get_value(self, state):
        latent = self.world_model.encode(state)
        self.critic.eval()
        val_t = self.critic.forward(Tensor([latent]))
        return val_t.data[0][0] if len(val_t.shape) == 2 else val_t.data[0]

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_world_model(self, batch_size=32):
        """Train the latent world model on real data."""
        if len(self.buffer) < batch_size:
            return None
        s, a, r, ns, d = self.buffer.sample(batch_size)
        return self.world_model.train_step(s, a, ns, r)

    def train_actor_critic(self, batch_size=16):
        """Train actor-critic in imagination (latent rollouts)."""
        if len(self.buffer) < batch_size:
            return None

        # Sample starting states
        states, _, _, _, _ = self.buffer.sample(batch_size)
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        n_updates = 0

        for start_state in states:
            latent = self.world_model.encode(start_state)
            # Imagine a trajectory
            latents = [latent]
            actions = []
            rewards = []

            for _ in range(self.imagination_horizon):
                self.actor.eval()
                probs_t = self.actor.forward(Tensor([latent]))
                probs = probs_t.data[0] if len(probs_t.shape) == 2 else probs_t.data
                # Sample action
                r = self.rng.random()
                cum = 0.0
                action = self.action_size - 1
                for a in range(self.action_size):
                    cum += max(probs[a], 1e-8)
                    if r < cum:
                        action = a
                        break
                actions.append(action)
                # Predict next latent and reward
                next_latent = self.world_model.predict_latent(latent, action)
                reward = self.world_model.predict_reward(latent, action)
                rewards.append(reward)
                latent = next_latent
                latents.append(latent)

            # Compute returns (critic bootstrap at the end)
            self.critic.eval()
            last_val_t = self.critic.forward(Tensor([latents[-1]]))
            last_val = last_val_t.data[0][0] if len(last_val_t.shape) == 2 else last_val_t.data[0]
            returns = [0.0] * len(rewards)
            G = last_val
            for t in range(len(rewards) - 1, -1, -1):
                G = rewards[t] + self.gamma * G
                returns[t] = G

            # Update critic
            for t in range(len(latents) - 1):
                self.critic.train()
                val_t = self.critic.forward(Tensor([latents[t]]))
                target = Tensor([[returns[t]]])
                c_loss = self.loss_fn.forward(val_t, target)
                c_grad = self.loss_fn.backward(val_t, target)
                self.critic.backward(c_grad)
                self.critic_opt.step(self.critic.get_trainable_layers())
                total_critic_loss += c_loss

            # Update actor
            for t in range(len(latents) - 1):
                self.critic.eval()
                val_t = self.critic.forward(Tensor([latents[t]]))
                val = val_t.data[0][0] if len(val_t.shape) == 2 else val_t.data[0]
                advantage = returns[t] - val

                self.actor.train()
                probs_t = self.actor.forward(Tensor([latents[t]]))
                probs = probs_t.data[0] if len(probs_t.shape) == 2 else probs_t.data
                # Policy gradient
                grad = [0.0] * self.action_size
                for a in range(self.action_size):
                    if a == actions[t]:
                        grad[a] = -(1.0 - probs[a]) * advantage
                    else:
                        grad[a] = probs[a] * advantage
                scale = 1.0 / max(abs(advantage), 1.0)
                grad = [g * scale for g in grad]
                self.actor.backward(Tensor([grad]))
                self.actor_opt.step(self.actor.get_trainable_layers())
                total_actor_loss += abs(advantage)

            n_updates += len(latents) - 1

        if n_updates > 0:
            total_actor_loss /= n_updates
            total_critic_loss /= n_updates
        return {'actor_loss': total_actor_loss, 'critic_loss': total_critic_loss}


# ---------------------------------------------------------------------------
# ModelBasedTrainer -- unified training loop
# ---------------------------------------------------------------------------

class ModelBasedTrainer:
    """Generic trainer for model-based RL agents."""

    def __init__(self, agent, env, agent_type='dyna_tabular'):
        self.agent = agent
        self.env = env
        self.agent_type = agent_type

    def train_episode(self, epsilon=0.1, max_steps=200):
        """Run one episode, return total reward."""
        state = self.env.reset()
        total_reward = 0.0
        steps = 0

        while steps < max_steps:
            if self.agent_type == 'dyna_tabular':
                # GridWorld: state is one-hot, need index
                s_idx = state.index(1.0) if 1.0 in state else 0
                action = self.agent.get_action(s_idx, epsilon)
            elif self.agent_type == 'dreamer':
                action = self.agent.get_action(state)
            elif self.agent_type == 'mpc':
                action = self.agent.get_action(state)
            else:
                action = self.agent.get_action(state, epsilon)

            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            if self.agent_type == 'dyna_tabular':
                ns_idx = next_state.index(1.0) if 1.0 in next_state else 0
                self.agent.update(s_idx, action, reward, ns_idx, done)
            elif self.agent_type == 'dreamer':
                self.agent.store_transition(state, action, reward, next_state, done)
            elif self.agent_type == 'mpc':
                pass  # MPC is planning-only, no update needed per step
            else:
                self.agent.update(state, action, reward, next_state, done)

            state = next_state
            steps += 1
            if done:
                break

        # Dreamer: train after episode
        if self.agent_type == 'dreamer':
            self.agent.train_world_model()
            self.agent.train_actor_critic()

        return total_reward

    def train(self, n_episodes, epsilon_start=1.0, epsilon_end=0.01,
              epsilon_decay=0.995, max_steps=200):
        """Train for n episodes, return reward history."""
        rewards = []
        epsilon = epsilon_start
        for ep in range(n_episodes):
            r = self.train_episode(epsilon=epsilon, max_steps=max_steps)
            rewards.append(r)
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
        return rewards


# ---------------------------------------------------------------------------
# PlanningMetrics -- model accuracy and planning quality
# ---------------------------------------------------------------------------

class PlanningMetrics:
    """Track model prediction accuracy and planning quality."""

    def __init__(self):
        self.state_errors = []
        self.reward_errors = []
        self.planning_rewards = []
        self.real_rewards = []

    def record_prediction(self, pred_state, true_state, pred_reward, true_reward):
        s_err = sum((p - t) ** 2 for p, t in zip(pred_state, true_state))
        s_err /= max(len(pred_state), 1)
        self.state_errors.append(s_err)
        self.reward_errors.append((pred_reward - true_reward) ** 2)

    def record_episode(self, planned_reward, actual_reward):
        self.planning_rewards.append(planned_reward)
        self.real_rewards.append(actual_reward)

    def get_metrics(self, last_n=100):
        metrics = {}
        if self.state_errors:
            errs = self.state_errors[-last_n:]
            metrics['mean_state_mse'] = sum(errs) / len(errs)
        if self.reward_errors:
            errs = self.reward_errors[-last_n:]
            metrics['mean_reward_mse'] = sum(errs) / len(errs)
        if self.planning_rewards and self.real_rewards:
            n = min(last_n, len(self.planning_rewards), len(self.real_rewards))
            pr = self.planning_rewards[-n:]
            rr = self.real_rewards[-n:]
            metrics['planning_mean'] = sum(pr) / len(pr)
            metrics['real_mean'] = sum(rr) / len(rr)
            if metrics['real_mean'] != 0:
                metrics['planning_accuracy'] = 1.0 - abs(
                    metrics['planning_mean'] - metrics['real_mean']
                ) / abs(metrics['real_mean'])
        return metrics

    def reset(self):
        self.state_errors = []
        self.reward_errors = []
        self.planning_rewards = []
        self.real_rewards = []
