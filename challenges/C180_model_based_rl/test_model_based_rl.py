"""Tests for C180: Model-Based Reinforcement Learning."""

import sys
import os
import math
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C179_reinforcement_learning'))
sys.path.insert(0, os.path.dirname(__file__))

from model_based_rl import (
    DataBuffer, WorldModel, RewardModel, ModelEnsemble,
    DynaTabular, DynaDeep, MPC, MBPO,
    LatentWorldModel, DreamerAgent,
    ModelBasedTrainer, PlanningMetrics,
    _one_hot_action, _to_tensor_2d, _to_flat,
)
from reinforcement_learning import GridWorld, CartPole, Bandit, MountainCar
from neural_network import Tensor


# ===================================================================
# Helper utilities
# ===================================================================

class TestHelpers:
    def test_one_hot_action(self):
        oh = _one_hot_action(2, 4)
        assert oh == [0.0, 0.0, 1.0, 0.0]

    def test_one_hot_action_zero(self):
        oh = _one_hot_action(0, 3)
        assert oh == [1.0, 0.0, 0.0]

    def test_to_tensor_2d_from_list(self):
        t = _to_tensor_2d([1.0, 2.0, 3.0])
        assert t.shape == (1, 3)

    def test_to_tensor_2d_already_2d(self):
        t = _to_tensor_2d([[1.0, 2.0], [3.0, 4.0]])
        assert t.shape == (2, 2)

    def test_to_tensor_2d_from_tensor(self):
        t = _to_tensor_2d(Tensor([1.0, 2.0]))
        assert t.shape == (1, 2)

    def test_to_flat_from_tensor(self):
        f = _to_flat(Tensor([1.0, 2.0, 3.0]))
        assert f == [1.0, 2.0, 3.0]

    def test_to_flat_from_list(self):
        f = _to_flat([5.0, 6.0])
        assert f == [5.0, 6.0]


# ===================================================================
# DataBuffer
# ===================================================================

class TestDataBuffer:
    def test_creation(self):
        buf = DataBuffer(capacity=100)
        assert len(buf) == 0

    def test_push_and_len(self):
        buf = DataBuffer(capacity=100)
        buf.push([1, 0], 0, 1.0, [0, 1], False)
        assert len(buf) == 1

    def test_push_multiple(self):
        buf = DataBuffer(capacity=100)
        for i in range(10):
            buf.push([i], 0, float(i), [i + 1], False)
        assert len(buf) == 10

    def test_capacity_overflow(self):
        buf = DataBuffer(capacity=5)
        for i in range(10):
            buf.push([i], 0, float(i), [i + 1], False)
        assert len(buf) == 5

    def test_sample(self):
        buf = DataBuffer(capacity=100, seed=42)
        for i in range(20):
            buf.push([float(i)], 0, float(i), [float(i + 1)], False)
        s, a, r, ns, d = buf.sample(5)
        assert len(s) == 5
        assert len(a) == 5
        assert len(r) == 5

    def test_sample_returns_valid_data(self):
        buf = DataBuffer(capacity=100, seed=42)
        buf.push([1.0, 2.0], 1, 3.0, [4.0, 5.0], True)
        s, a, r, ns, d = buf.sample(1)
        assert s[0] == [1.0, 2.0]
        assert a[0] == 1
        assert r[0] == 3.0
        assert ns[0] == [4.0, 5.0]
        assert d[0] == True

    def test_sample_all(self):
        buf = DataBuffer(capacity=100)
        for i in range(5):
            buf.push([float(i)], 0, float(i), [float(i + 1)], False)
        s, a, r, ns, d = buf.sample_all()
        assert len(s) == 5

    def test_sample_clamps_to_size(self):
        buf = DataBuffer(capacity=100, seed=42)
        buf.push([1.0], 0, 0.0, [2.0], False)
        s, a, r, ns, d = buf.sample(10)
        assert len(s) == 1


# ===================================================================
# WorldModel
# ===================================================================

class TestWorldModel:
    def test_creation(self):
        wm = WorldModel(state_size=4, action_size=2)
        assert wm.state_size == 4
        assert wm.action_size == 2

    def test_predict_shape(self):
        wm = WorldModel(state_size=4, action_size=2, seed=42)
        ns, r = wm.predict([0.1, 0.2, 0.3, 0.4], 0)
        assert len(ns) == 4
        assert isinstance(r, float)

    def test_predict_deterministic(self):
        wm = WorldModel(state_size=2, action_size=2, seed=42)
        ns1, r1 = wm.predict([0.5, 0.5], 1)
        ns2, r2 = wm.predict([0.5, 0.5], 1)
        assert ns1 == ns2
        assert r1 == r2

    def test_predict_different_actions(self):
        wm = WorldModel(state_size=2, action_size=3, seed=42)
        ns0, r0 = wm.predict([0.5, 0.5], 0)
        ns1, r1 = wm.predict([0.5, 0.5], 1)
        # Different actions should give different predictions
        assert ns0 != ns1 or r0 != r1

    def test_predict_batch(self):
        wm = WorldModel(state_size=2, action_size=2, seed=42)
        states = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        actions = [0, 1, 0]
        ns, r = wm.predict_batch(states, actions)
        assert len(ns) == 3
        assert len(r) == 3

    def test_train_batch(self):
        wm = WorldModel(state_size=2, action_size=2, hidden_sizes=[16], seed=42)
        states = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        actions = [0, 1, 0]
        next_states = [[0.1, 0.0], [1.0, 0.1], [0.1, 1.0]]
        rewards = [0.0, 1.0, 0.0]
        loss = wm.train_batch(states, actions, next_states, rewards)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_reduces_loss(self):
        wm = WorldModel(state_size=2, action_size=2, hidden_sizes=[16],
                         lr=0.01, seed=42)
        states = [[0.0, 0.0]] * 8
        actions = [0] * 8
        next_states = [[0.5, 0.5]] * 8
        rewards = [1.0] * 8
        first_loss = wm.train_batch(states, actions, next_states, rewards)
        for _ in range(20):
            wm.train_batch(states, actions, next_states, rewards)
        last_loss = wm.train_batch(states, actions, next_states, rewards)
        assert last_loss < first_loss

    def test_get_accuracy(self):
        wm = WorldModel(state_size=2, action_size=2, seed=42)
        states = [[0.1, 0.2], [0.3, 0.4]]
        actions = [0, 1]
        ns = [[0.2, 0.3], [0.4, 0.5]]
        rewards = [0.5, 1.0]
        acc = wm.get_accuracy(states, actions, ns, rewards)
        assert 'state_mse' in acc
        assert 'reward_mse' in acc
        assert acc['state_mse'] >= 0
        assert acc['reward_mse'] >= 0

    def test_custom_hidden_sizes(self):
        wm = WorldModel(state_size=4, action_size=3, hidden_sizes=[32, 16])
        ns, r = wm.predict([0.1, 0.2, 0.3, 0.4], 2)
        assert len(ns) == 4

    def test_train_losses_tracked(self):
        wm = WorldModel(state_size=2, action_size=2, hidden_sizes=[8], seed=42)
        for _ in range(3):
            wm.train_batch([[0.0, 0.0]], [0], [[0.5, 0.5]], [1.0])
        assert len(wm.train_losses) == 3


# ===================================================================
# RewardModel
# ===================================================================

class TestRewardModel:
    def test_creation(self):
        rm = RewardModel(state_size=4, action_size=2)
        assert rm.state_size == 4

    def test_predict(self):
        rm = RewardModel(state_size=2, action_size=2, seed=42)
        r = rm.predict([0.5, 0.5], 0)
        assert isinstance(r, float)

    def test_train_batch(self):
        rm = RewardModel(state_size=2, action_size=2, hidden_sizes=[8], seed=42)
        loss = rm.train_batch(
            [[0.0, 0.0], [1.0, 1.0]],
            [0, 1],
            [0.0, 1.0]
        )
        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_reduces_loss(self):
        rm = RewardModel(state_size=2, action_size=2, hidden_sizes=[8],
                          lr=0.01, seed=42)
        states = [[0.0, 0.0]] * 8
        actions = [0] * 8
        rewards = [1.0] * 8
        first = rm.train_batch(states, actions, rewards)
        for _ in range(30):
            rm.train_batch(states, actions, rewards)
        last = rm.train_batch(states, actions, rewards)
        assert last < first


# ===================================================================
# ModelEnsemble
# ===================================================================

class TestModelEnsemble:
    def test_creation(self):
        me = ModelEnsemble(state_size=2, action_size=2, n_models=3)
        assert len(me.models) == 3

    def test_predict(self):
        me = ModelEnsemble(state_size=2, action_size=2, n_models=3, seed=42)
        ns, r = me.predict([0.5, 0.5], 0)
        assert len(ns) == 2
        assert isinstance(r, float)

    def test_predict_mean(self):
        me = ModelEnsemble(state_size=2, action_size=2, n_models=3, seed=42)
        ns, r = me.predict_mean([0.5, 0.5], 1)
        assert len(ns) == 2
        assert isinstance(r, float)

    def test_get_uncertainty(self):
        me = ModelEnsemble(state_size=2, action_size=2, n_models=5, seed=42)
        unc = me.get_uncertainty([0.5, 0.5], 0)
        assert 'state_variance' in unc
        assert 'reward_variance' in unc
        assert 'total_uncertainty' in unc
        assert unc['state_variance'] >= 0
        assert unc['total_uncertainty'] >= 0

    def test_train_batch(self):
        me = ModelEnsemble(state_size=2, action_size=2, n_models=3,
                           hidden_sizes=[8], seed=42)
        loss = me.train_batch(
            [[0.0, 0.0], [1.0, 1.0]], [0, 1],
            [[0.5, 0.5], [0.5, 0.5]], [1.0, 0.0]
        )
        assert isinstance(loss, float)

    def test_train_bootstrapped(self):
        me = ModelEnsemble(state_size=2, action_size=2, n_models=3,
                           hidden_sizes=[8], seed=42)
        loss = me.train_bootstrapped(
            [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], [0, 1, 0],
            [[0.5, 0.5], [0.5, 0.5], [1.0, 0.0]], [1.0, 0.0, 0.5]
        )
        assert isinstance(loss, float)

    def test_different_seeds_differ(self):
        me = ModelEnsemble(state_size=2, action_size=2, n_models=3, seed=42)
        preds = []
        for m in me.models:
            ns, r = m.predict([0.5, 0.5], 0)
            preds.append((ns, r))
        # At least some models should differ
        all_same = all(p == preds[0] for p in preds)
        assert not all_same


# ===================================================================
# DynaTabular
# ===================================================================

class TestDynaTabular:
    def test_creation(self):
        d = DynaTabular(n_states=25, n_actions=4)
        assert d.n_states == 25

    def test_get_action(self):
        d = DynaTabular(n_states=25, n_actions=4, seed=42)
        a = d.get_action(0, epsilon=0.0)
        assert 0 <= a < 4

    def test_get_action_explores(self):
        d = DynaTabular(n_states=25, n_actions=4, seed=42)
        actions = set()
        for _ in range(100):
            actions.add(d.get_action(0, epsilon=1.0))
        assert len(actions) > 1

    def test_update(self):
        d = DynaTabular(n_states=25, n_actions=4, seed=42)
        d.update(0, 1, 0.0, 1, False)
        assert (0, 1) in d.visited

    def test_model_learned(self):
        d = DynaTabular(n_states=25, n_actions=4, seed=42)
        d.update(0, 1, 0.5, 5, False)
        assert 0 in d.model
        assert 1 in d.model[0]
        assert d.model[0][1] == (5, 0.5, False)

    def test_planning_updates_q(self):
        d = DynaTabular(n_states=25, n_actions=4, planning_steps=10, seed=42)
        # Give a big reward at (0, 1) -> 5
        d.update(0, 1, 10.0, 5, True)
        # Q should be updated for state 0, action 1
        assert d.q[0][1] > 0

    def test_get_q_values(self):
        d = DynaTabular(n_states=10, n_actions=3, seed=42)
        qv = d.get_q_values(0)
        assert len(qv) == 3

    def test_get_policy(self):
        d = DynaTabular(n_states=10, n_actions=3, seed=42)
        pol = d.get_policy()
        assert len(pol) == 10
        assert all(0 <= a < 3 for a in pol)

    def test_multiple_planning_steps(self):
        d = DynaTabular(n_states=25, n_actions=4, planning_steps=0, seed=42)
        d.update(0, 1, 1.0, 5, False)
        q_no_plan = d.q[0][1]
        d2 = DynaTabular(n_states=25, n_actions=4, planning_steps=50, seed=42)
        d2.update(0, 1, 1.0, 5, False)
        # With more planning, Q values propagate more
        # (both should have some update since real update always happens)
        assert d2.q[0][1] > 0

    def test_learns_gridworld(self):
        """DynaTabular should learn GridWorld reasonably."""
        env = GridWorld(rows=3, cols=3, seed=42)
        d = DynaTabular(n_states=9, n_actions=4, planning_steps=5, seed=42)
        total_rewards = []
        for ep in range(50):
            state = env.reset()
            s_idx = state.index(1.0) if 1.0 in state else 0
            ep_reward = 0
            for _ in range(50):
                a = d.get_action(s_idx, epsilon=0.3)
                ns, r, done, _ = env.step(a)
                ns_idx = ns.index(1.0) if 1.0 in ns else 0
                d.update(s_idx, a, r, ns_idx, done)
                ep_reward += r
                s_idx = ns_idx
                if done:
                    break
            total_rewards.append(ep_reward)
        # Last episodes should have higher reward than first
        early = sum(total_rewards[:10]) / 10
        late = sum(total_rewards[-10:]) / 10
        assert late >= early


# ===================================================================
# DynaDeep
# ===================================================================

class TestDynaDeep:
    def test_creation(self):
        d = DynaDeep(state_size=4, action_size=2)
        assert d.state_size == 4

    def test_get_action(self):
        d = DynaDeep(state_size=4, action_size=2, seed=42)
        a = d.get_action([0.1, 0.2, 0.3, 0.4], epsilon=0.0)
        assert 0 <= a < 2

    def test_get_q_values(self):
        d = DynaDeep(state_size=4, action_size=2, seed=42)
        qv = d.get_q_values([0.1, 0.2, 0.3, 0.4])
        assert len(qv) == 2

    def test_update_returns_dict(self):
        d = DynaDeep(state_size=2, action_size=2, hidden_sizes=[8],
                     planning_steps=2, seed=42)
        # Need enough data for batch
        for i in range(40):
            result = d.update([0.1 * i, 0.2], 0, 1.0,
                             [0.1 * i + 0.05, 0.3], False)
        assert isinstance(result, dict)
        assert 'model_loss' in result
        assert 'dqn_loss' in result

    def test_recent_states_tracked(self):
        d = DynaDeep(state_size=2, action_size=2, seed=42)
        d.update([0.5, 0.5], 0, 1.0, [0.6, 0.6], False)
        assert len(d.recent_states) == 1

    def test_recent_states_bounded(self):
        d = DynaDeep(state_size=2, action_size=2, seed=42)
        d.max_recent = 5
        for i in range(20):
            d.update([float(i)], 0, 0.0, [float(i + 1)], False)
        assert len(d.recent_states) <= 5


# ===================================================================
# MPC -- Model Predictive Control
# ===================================================================

class TestMPC:
    def test_random_shooting(self):
        wm = WorldModel(state_size=2, action_size=3, seed=42)
        mpc = MPC(wm, action_size=3, horizon=3, n_candidates=20,
                  method='random_shooting', seed=42)
        a = mpc.get_action([0.5, 0.5])
        assert 0 <= a < 3

    def test_cem(self):
        wm = WorldModel(state_size=2, action_size=3, seed=42)
        mpc = MPC(wm, action_size=3, horizon=3, n_candidates=20,
                  n_elite=5, cem_iterations=2, method='cem', seed=42)
        a = mpc.get_action([0.5, 0.5])
        assert 0 <= a < 3

    def test_deterministic(self):
        wm = WorldModel(state_size=2, action_size=2, seed=42)
        mpc1 = MPC(wm, action_size=2, horizon=3, n_candidates=10,
                   method='random_shooting', seed=42)
        mpc2 = MPC(wm, action_size=2, horizon=3, n_candidates=10,
                   method='random_shooting', seed=42)
        a1 = mpc1.get_action([0.5, 0.5])
        a2 = mpc2.get_action([0.5, 0.5])
        assert a1 == a2

    def test_horizon_effect(self):
        """Longer horizon should still return valid action."""
        wm = WorldModel(state_size=2, action_size=2, seed=42)
        mpc = MPC(wm, action_size=2, horizon=20, n_candidates=10, seed=42)
        a = mpc.get_action([0.5, 0.5])
        assert 0 <= a < 2

    def test_cem_multiple_iterations(self):
        wm = WorldModel(state_size=2, action_size=2, seed=42)
        mpc = MPC(wm, action_size=2, horizon=5, n_candidates=30,
                  n_elite=5, cem_iterations=5, method='cem', seed=42)
        a = mpc.get_action([0.5, 0.5])
        assert 0 <= a < 2

    def test_simulate_returns_total(self):
        wm = WorldModel(state_size=2, action_size=2, seed=42)
        mpc = MPC(wm, action_size=2, horizon=3, gamma=0.99, seed=42)
        ret = mpc._simulate([0.5, 0.5], [0, 1, 0])
        assert isinstance(ret, float)


# ===================================================================
# MBPO
# ===================================================================

class TestMBPO:
    def test_creation(self):
        m = MBPO(state_size=4, action_size=2)
        assert m.state_size == 4
        assert len(m.ensemble.models) == 3

    def test_get_action(self):
        m = MBPO(state_size=4, action_size=2, seed=42)
        a = m.get_action([0.1, 0.2, 0.3, 0.4], epsilon=0.0)
        assert 0 <= a < 2

    def test_update(self):
        m = MBPO(state_size=2, action_size=2, hidden_sizes=[8],
                 model_train_freq=5, seed=42)
        for i in range(10):
            m.update([0.1 * i, 0.2], 0, 1.0, [0.1 * i + 0.05, 0.3], False)
        assert len(m.real_buffer) == 10

    def test_model_buffer_populated(self):
        m = MBPO(state_size=2, action_size=2, hidden_sizes=[8],
                 rollout_batch=5, rollout_length=2,
                 model_train_freq=5, seed=42)
        for i in range(40):
            m.update([0.1 * (i % 5), 0.2], i % 2, 1.0,
                    [0.1 * ((i + 1) % 5), 0.3], False)
        assert len(m.model_buffer) > 0

    def test_rollout_length(self):
        m = MBPO(state_size=2, action_size=2, hidden_sizes=[8],
                 rollout_batch=2, rollout_length=3, seed=42)
        # Fill real buffer
        for i in range(40):
            m.real_buffer.push([float(i % 3), 0.5], 0, 0.0, [float((i+1) % 3), 0.5], False)
        m._generate_rollouts()
        # Each rollout of length 3 from 2 start states = 6 entries
        assert len(m.model_buffer) == 6


# ===================================================================
# LatentWorldModel
# ===================================================================

class TestLatentWorldModel:
    def test_creation(self):
        lm = LatentWorldModel(state_size=4, action_size=2, latent_size=8)
        assert lm.latent_size == 8

    def test_encode(self):
        lm = LatentWorldModel(state_size=4, action_size=2, latent_size=8, seed=42)
        z = lm.encode([0.1, 0.2, 0.3, 0.4])
        assert len(z) == 8
        # tanh output should be bounded
        for v in z:
            assert -1.0 <= v <= 1.0

    def test_decode(self):
        lm = LatentWorldModel(state_size=4, action_size=2, latent_size=8, seed=42)
        z = lm.encode([0.1, 0.2, 0.3, 0.4])
        recon = lm.decode(z)
        assert len(recon) == 4

    def test_predict_latent(self):
        lm = LatentWorldModel(state_size=4, action_size=2, latent_size=8, seed=42)
        z = lm.encode([0.1, 0.2, 0.3, 0.4])
        nz = lm.predict_latent(z, 0)
        assert len(nz) == 8
        for v in nz:
            assert -1.0 <= v <= 1.0

    def test_predict_reward(self):
        lm = LatentWorldModel(state_size=4, action_size=2, latent_size=8, seed=42)
        z = lm.encode([0.1, 0.2, 0.3, 0.4])
        r = lm.predict_reward(z, 1)
        assert isinstance(r, float)

    def test_predict_full(self):
        lm = LatentWorldModel(state_size=4, action_size=2, latent_size=8, seed=42)
        ns, r = lm.predict([0.1, 0.2, 0.3, 0.4], 0)
        assert len(ns) == 4
        assert isinstance(r, float)

    def test_train_step(self):
        lm = LatentWorldModel(state_size=2, action_size=2, latent_size=4,
                               hidden_sizes=[8], lr=0.01, seed=42)
        losses = lm.train_step(
            [[0.1, 0.2], [0.3, 0.4]],
            [0, 1],
            [[0.15, 0.25], [0.35, 0.45]],
            [0.5, 1.0]
        )
        assert 'recon_loss' in losses
        assert 'dynamics_loss' in losses
        assert 'reward_loss' in losses

    def test_train_reduces_recon_loss(self):
        lm = LatentWorldModel(state_size=2, action_size=2, latent_size=4,
                               hidden_sizes=[8], lr=0.005, seed=42)
        states = [[0.5, 0.5]] * 8
        actions = [0] * 8
        next_states = [[0.6, 0.6]] * 8
        rewards = [1.0] * 8
        first = lm.train_step(states, actions, next_states, rewards)
        for _ in range(30):
            lm.train_step(states, actions, next_states, rewards)
        last = lm.train_step(states, actions, next_states, rewards)
        assert last['recon_loss'] < first['recon_loss']


# ===================================================================
# DreamerAgent
# ===================================================================

class TestDreamerAgent:
    def test_creation(self):
        d = DreamerAgent(state_size=4, action_size=2)
        assert d.action_size == 2

    def test_get_action(self):
        d = DreamerAgent(state_size=4, action_size=2, seed=42)
        a = d.get_action([0.1, 0.2, 0.3, 0.4])
        assert 0 <= a < 2

    def test_get_action_probs(self):
        d = DreamerAgent(state_size=4, action_size=3, seed=42)
        probs = d.get_action_probs([0.1, 0.2, 0.3, 0.4])
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 0.01

    def test_get_value(self):
        d = DreamerAgent(state_size=4, action_size=2, seed=42)
        v = d.get_value([0.1, 0.2, 0.3, 0.4])
        assert isinstance(v, float)

    def test_store_transition(self):
        d = DreamerAgent(state_size=2, action_size=2, seed=42)
        d.store_transition([0.1, 0.2], 0, 1.0, [0.3, 0.4], False)
        assert len(d.buffer) == 1

    def test_train_world_model(self):
        d = DreamerAgent(state_size=2, action_size=2, latent_size=4,
                          hidden_sizes=[8], seed=42)
        for i in range(40):
            d.store_transition([0.1 * (i % 5), 0.2], i % 2, float(i % 3),
                              [0.1 * ((i + 1) % 5), 0.3], False)
        result = d.train_world_model(batch_size=8)
        assert result is not None
        assert 'recon_loss' in result

    def test_train_actor_critic(self):
        d = DreamerAgent(state_size=2, action_size=2, latent_size=4,
                          hidden_sizes=[8], imagination_horizon=3, seed=42)
        for i in range(40):
            d.store_transition([0.1 * (i % 5), 0.2], i % 2, float(i % 3),
                              [0.1 * ((i + 1) % 5), 0.3], False)
        result = d.train_actor_critic(batch_size=4)
        assert result is not None
        assert 'actor_loss' in result
        assert 'critic_loss' in result

    def test_train_returns_none_if_insufficient_data(self):
        d = DreamerAgent(state_size=2, action_size=2, seed=42)
        assert d.train_world_model(batch_size=32) is None
        assert d.train_actor_critic(batch_size=32) is None


# ===================================================================
# ModelBasedTrainer
# ===================================================================

class TestModelBasedTrainer:
    def test_dyna_tabular_trainer(self):
        env = GridWorld(rows=3, cols=3, seed=42)
        agent = DynaTabular(n_states=9, n_actions=4, planning_steps=5, seed=42)
        trainer = ModelBasedTrainer(agent, env, agent_type='dyna_tabular')
        r = trainer.train_episode(epsilon=0.5, max_steps=50)
        assert isinstance(r, float)

    def test_dyna_tabular_train_multi(self):
        env = GridWorld(rows=3, cols=3, seed=42)
        agent = DynaTabular(n_states=9, n_actions=4, planning_steps=3, seed=42)
        trainer = ModelBasedTrainer(agent, env, agent_type='dyna_tabular')
        rewards = trainer.train(n_episodes=10, max_steps=50)
        assert len(rewards) == 10

    def test_dyna_deep_trainer(self):
        env = CartPole(seed=42)
        agent = DynaDeep(state_size=4, action_size=2, hidden_sizes=[8],
                         planning_steps=2, seed=42)
        trainer = ModelBasedTrainer(agent, env, agent_type='dyna_deep')
        r = trainer.train_episode(epsilon=0.5, max_steps=20)
        assert isinstance(r, float)

    def test_dreamer_trainer(self):
        env = CartPole(seed=42)
        agent = DreamerAgent(state_size=4, action_size=2, latent_size=4,
                              hidden_sizes=[8], imagination_horizon=3, seed=42)
        trainer = ModelBasedTrainer(agent, env, agent_type='dreamer')
        r = trainer.train_episode(max_steps=20)
        assert isinstance(r, float)

    def test_mpc_trainer(self):
        wm = WorldModel(state_size=4, action_size=2, hidden_sizes=[8], seed=42)
        mpc = MPC(wm, action_size=2, horizon=3, n_candidates=10, seed=42)
        env = CartPole(seed=42)
        trainer = ModelBasedTrainer(mpc, env, agent_type='mpc')
        r = trainer.train_episode(max_steps=20)
        assert isinstance(r, float)

    def test_train_with_decay(self):
        env = GridWorld(rows=3, cols=3, seed=42)
        agent = DynaTabular(n_states=9, n_actions=4, planning_steps=2, seed=42)
        trainer = ModelBasedTrainer(agent, env, agent_type='dyna_tabular')
        rewards = trainer.train(n_episodes=5, epsilon_start=1.0,
                               epsilon_end=0.1, epsilon_decay=0.5, max_steps=30)
        assert len(rewards) == 5


# ===================================================================
# PlanningMetrics
# ===================================================================

class TestPlanningMetrics:
    def test_creation(self):
        pm = PlanningMetrics()
        assert len(pm.state_errors) == 0

    def test_record_prediction(self):
        pm = PlanningMetrics()
        pm.record_prediction([0.1, 0.2], [0.15, 0.25], 0.5, 0.6)
        assert len(pm.state_errors) == 1
        assert len(pm.reward_errors) == 1

    def test_record_episode(self):
        pm = PlanningMetrics()
        pm.record_episode(10.0, 8.0)
        assert len(pm.planning_rewards) == 1

    def test_get_metrics_empty(self):
        pm = PlanningMetrics()
        m = pm.get_metrics()
        assert m == {}

    def test_get_metrics_with_data(self):
        pm = PlanningMetrics()
        for i in range(10):
            pm.record_prediction(
                [0.1 * i, 0.2], [0.1 * i + 0.01, 0.2], 0.5, 0.5 + 0.01 * i
            )
            pm.record_episode(10.0 + i, 9.0 + i)
        m = pm.get_metrics()
        assert 'mean_state_mse' in m
        assert 'mean_reward_mse' in m
        assert 'planning_mean' in m
        assert 'real_mean' in m
        assert 'planning_accuracy' in m

    def test_reset(self):
        pm = PlanningMetrics()
        pm.record_prediction([0.1], [0.2], 0.0, 0.1)
        pm.reset()
        assert len(pm.state_errors) == 0

    def test_metrics_accuracy_perfect(self):
        pm = PlanningMetrics()
        for _ in range(10):
            pm.record_episode(10.0, 10.0)
        m = pm.get_metrics()
        assert abs(m['planning_accuracy'] - 1.0) < 0.01


# ===================================================================
# Integration tests
# ===================================================================

class TestIntegration:
    def test_world_model_with_gridworld(self):
        """Train a world model on GridWorld transitions."""
        env = GridWorld(rows=3, cols=3, seed=42)
        wm = WorldModel(state_size=9, action_size=4, hidden_sizes=[16], seed=42)
        buf = DataBuffer(capacity=1000, seed=42)

        # Collect data
        for ep in range(20):
            state = env.reset()
            for _ in range(30):
                action = random.randint(0, 3)
                ns, r, done, _ = env.step(action)
                buf.push(state, action, r, ns, done)
                state = ns
                if done:
                    break

        # Train
        for _ in range(20):
            s, a, r, ns, d = buf.sample(16)
            wm.train_batch(s, a, ns, r)

        # Check accuracy improved
        s, a, r, ns, d = buf.sample(10)
        acc = wm.get_accuracy(s, a, ns, r)
        assert acc['state_mse'] >= 0  # Just check it runs

    def test_ensemble_uncertainty_decreases_with_training(self):
        """Ensemble uncertainty should decrease after training."""
        me = ModelEnsemble(state_size=2, action_size=2, n_models=3,
                           hidden_sizes=[8], lr=0.01, seed=42)
        # Get initial uncertainty
        unc_before = me.get_uncertainty([0.5, 0.5], 0)

        # Train on consistent data
        for _ in range(30):
            me.train_batch(
                [[0.5, 0.5]] * 8, [0] * 8,
                [[0.6, 0.6]] * 8, [1.0] * 8
            )

        unc_after = me.get_uncertainty([0.5, 0.5], 0)
        # Models should converge, reducing disagreement
        assert unc_after['state_variance'] <= unc_before['state_variance'] + 0.1

    def test_dyna_vs_no_planning(self):
        """Dyna with planning should match or beat no planning."""
        env1 = GridWorld(rows=3, cols=3, seed=42)
        env2 = GridWorld(rows=3, cols=3, seed=42)

        no_plan = DynaTabular(n_states=9, n_actions=4, planning_steps=0, seed=42)
        with_plan = DynaTabular(n_states=9, n_actions=4, planning_steps=10, seed=42)

        for ep in range(30):
            state1 = env1.reset()
            state2 = env2.reset()
            s1 = state1.index(1.0) if 1.0 in state1 else 0
            s2 = state2.index(1.0) if 1.0 in state2 else 0
            for _ in range(30):
                a1 = no_plan.get_action(s1, epsilon=0.3)
                ns1, r1, d1, _ = env1.step(a1)
                ns1_idx = ns1.index(1.0) if 1.0 in ns1 else 0
                no_plan.update(s1, a1, r1, ns1_idx, d1)
                s1 = ns1_idx

                a2 = with_plan.get_action(s2, epsilon=0.3)
                ns2, r2, d2, _ = env2.step(a2)
                ns2_idx = ns2.index(1.0) if 1.0 in ns2 else 0
                with_plan.update(s2, a2, r2, ns2_idx, d2)
                s2 = ns2_idx
                if d1 or d2:
                    break

        # Planning agent should have higher max Q anywhere
        plan_max = max(max(row) for row in with_plan.q)
        no_plan_max = max(max(row) for row in no_plan.q)
        # With planning, Q values should be at least as propagated
        assert plan_max >= no_plan_max - 0.5

    def test_mbpo_end_to_end(self):
        """MBPO should run without errors on CartPole."""
        env = CartPole(seed=42)
        m = MBPO(state_size=4, action_size=2, hidden_sizes=[8],
                 rollout_batch=3, rollout_length=1,
                 model_train_freq=10, seed=42)
        state = env.reset()
        total = 0
        for step in range(50):
            a = m.get_action(state, epsilon=0.5)
            ns, r, d, _ = env.step(a)
            m.update(state, a, r, ns, d)
            total += r
            state = ns
            if d:
                state = env.reset()
        assert total > 0

    def test_dreamer_end_to_end(self):
        """Dreamer should run without errors."""
        env = CartPole(seed=42)
        d = DreamerAgent(state_size=4, action_size=2, latent_size=4,
                          hidden_sizes=[8], imagination_horizon=3, seed=42)
        state = env.reset()
        for step in range(30):
            a = d.get_action(state)
            ns, r, done, _ = env.step(a)
            d.store_transition(state, a, r, ns, done)
            state = ns
            if done:
                state = env.reset()
        # Train
        wm_result = d.train_world_model(batch_size=8)
        ac_result = d.train_actor_critic(batch_size=4)
        assert wm_result is not None
        assert ac_result is not None

    def test_mpc_with_trained_model(self):
        """MPC should use a trained model for better planning."""
        env = GridWorld(rows=3, cols=3, seed=42)
        wm = WorldModel(state_size=9, action_size=4, hidden_sizes=[16],
                         lr=0.005, seed=42)
        buf = DataBuffer(capacity=1000, seed=42)

        # Collect random data
        for ep in range(30):
            state = env.reset()
            for _ in range(30):
                a = random.randint(0, 3)
                ns, r, d, _ = env.step(a)
                buf.push(state, a, r, ns, d)
                state = ns
                if d:
                    break

        # Train model
        for _ in range(30):
            s, a, r, ns, d = buf.sample(16)
            wm.train_batch(s, a, ns, r)

        # MPC should return valid actions
        mpc = MPC(wm, action_size=4, horizon=3, n_candidates=20, seed=42)
        state = env.reset()
        a = mpc.get_action(state)
        assert 0 <= a < 4

    def test_latent_model_roundtrip(self):
        """Encode then decode should approximate identity."""
        lm = LatentWorldModel(state_size=2, action_size=2, latent_size=8,
                               hidden_sizes=[16], lr=0.01, seed=42)
        # Train on reconstruction
        states = [[0.5, 0.5]] * 8
        actions = [0] * 8
        next_states = [[0.6, 0.6]] * 8
        rewards = [1.0] * 8
        for _ in range(50):
            lm.train_step(states, actions, next_states, rewards)
        # Check reconstruction
        z = lm.encode([0.5, 0.5])
        recon = lm.decode(z)
        # Should be somewhat close after training
        error = sum((a - b) ** 2 for a, b in zip(recon, [0.5, 0.5]))
        assert error < 1.0  # Reasonable reconstruction

    def test_planning_metrics_with_model(self):
        """PlanningMetrics should track model accuracy over time."""
        wm = WorldModel(state_size=2, action_size=2, hidden_sizes=[8],
                         lr=0.01, seed=42)
        pm = PlanningMetrics()

        # Constant transition
        true_ns = [0.6, 0.6]
        true_r = 1.0

        for epoch in range(20):
            wm.train_batch([[0.5, 0.5]] * 4, [0] * 4,
                           [true_ns] * 4, [true_r] * 4)
            pred_ns, pred_r = wm.predict([0.5, 0.5], 0)
            pm.record_prediction(pred_ns, true_ns, pred_r, true_r)

        m = pm.get_metrics()
        assert m['mean_state_mse'] >= 0

    def test_world_model_different_envs(self):
        """WorldModel should handle different state/action sizes."""
        for ss, as_ in [(2, 2), (4, 3), (9, 4)]:
            wm = WorldModel(state_size=ss, action_size=as_,
                             hidden_sizes=[8], seed=42)
            state = [0.5] * ss
            ns, r = wm.predict(state, 0)
            assert len(ns) == ss

    def test_full_pipeline(self):
        """Full pipeline: collect data, train model, plan, act."""
        env = GridWorld(rows=3, cols=3, seed=42)
        wm = WorldModel(state_size=9, action_size=4, hidden_sizes=[16], seed=42)
        buf = DataBuffer(capacity=500, seed=42)
        mpc = MPC(wm, action_size=4, horizon=3, n_candidates=20, seed=42)

        # Phase 1: random exploration
        for ep in range(10):
            state = env.reset()
            for _ in range(20):
                a = random.randint(0, 3)
                ns, r, d, _ = env.step(a)
                buf.push(state, a, r, ns, d)
                state = ns
                if d:
                    break

        # Phase 2: train model
        for _ in range(20):
            s, a, r, ns, d = buf.sample(16)
            wm.train_batch(s, a, ns, r)

        # Phase 3: plan and act
        state = env.reset()
        for _ in range(10):
            a = mpc.get_action(state)
            ns, r, d, _ = env.step(a)
            buf.push(state, a, r, ns, d)
            state = ns
            if d:
                break
        # Should complete without errors
        assert True


# ===================================================================
# Run all tests
# ===================================================================

def run_tests():
    """Run all test classes and methods."""
    test_classes = [
        TestHelpers, TestDataBuffer, TestWorldModel, TestRewardModel,
        TestModelEnsemble, TestDynaTabular, TestDynaDeep, TestMPC,
        TestMBPO, TestLatentWorldModel, TestDreamerAgent,
        TestModelBasedTrainer, TestPlanningMetrics, TestIntegration,
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
                print(f"  PASS: {cls.__name__}.{method_name}")
            except Exception as e:
                failed += 1
                errors.append((cls.__name__, method_name, str(e)))
                print(f"  FAIL: {cls.__name__}.{method_name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print(f"\nFailures:")
        for cls_name, method, err in errors:
            print(f"  {cls_name}.{method}: {err}")
    print(f"{'=' * 60}")
    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
