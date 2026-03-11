"""
Tests for C129: Neural Network Framework

Covers: Tensor, Parameter, Init, Module, Linear, Conv1D, RNN, LSTM, Embedding,
BatchNorm, Dropout, Activations, Loss functions, Optimizers, LR Schedulers,
Sequential, Trainer, gradient checking, end-to-end training.
"""

import sys
import os
import math
import random
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C128_automatic_differentiation'))

from neural_network import (
    Tensor, Parameter, Init, Module, Linear, Conv1D, RNNCell, LSTMCell,
    Embedding, BatchNorm1D, Dropout, ReLU, Sigmoid, Tanh, LeakyReLU, ELU,
    GELU, Softmax, MSELoss, L1Loss, CrossEntropyLoss, BinaryCrossEntropyLoss,
    HuberLoss, SGD, Adam, RMSProp, AdaGrad, StepLR, ExponentialLR,
    CosineAnnealingLR, Sequential, Trainer, TrainHistory, num_grad,
    accuracy, _prod, _randn
)
from autodiff import Var, ReverseAD, var_sigmoid, var_tanh


# ===== Tensor =====

class TestTensor:
    def test_scalar(self):
        t = Tensor(3.0)
        assert t.shape == (1,)
        assert t.item() == 3.0

    def test_1d(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert t.size == 3
        assert t.ndim == 1
        assert len(t) == 3

    def test_2d(self):
        t = Tensor([[1, 2], [3, 4], [5, 6]])
        assert t.shape == (3, 2)
        assert t.size == 6

    def test_3d(self):
        t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert t.shape == (2, 2, 2)
        assert t.size == 8

    def test_to_list_1d(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert t.to_list() == [1.0, 2.0, 3.0]

    def test_to_list_2d(self):
        t = Tensor([[1, 2], [3, 4]])
        assert t.to_list() == [[1.0, 2.0], [3.0, 4.0]]

    def test_to_list_3d(self):
        t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = t.to_list()
        assert result[0][0] == [1.0, 2.0]
        assert result[1][1] == [7.0, 8.0]

    def test_zeros(self):
        t = Tensor.zeros((2, 3))
        assert t.shape == (2, 3)
        assert all(v == 0.0 for v in t.data)

    def test_ones(self):
        t = Tensor.ones((3,))
        assert t.shape == (3,)
        assert all(v == 1.0 for v in t.data)

    def test_randn(self):
        random.seed(42)
        t = Tensor.randn((100,))
        assert t.shape == (100,)
        mean = sum(t.data) / len(t.data)
        assert abs(mean) < 0.5  # rough check

    def test_reshape(self):
        t = Tensor([1, 2, 3, 4, 5, 6])
        t2 = t.reshape((2, 3))
        assert t2.shape == (2, 3)
        assert t2.size == 6

    def test_to_vars(self):
        t = Tensor([1.0, 2.0, 3.0])
        t.to_vars()
        assert all(isinstance(v, Var) for v in t.data)
        assert t.requires_grad

    def test_from_tensor(self):
        t1 = Tensor([1.0, 2.0])
        t2 = Tensor(t1)
        assert t2.shape == t1.shape
        assert t2.data == t1.data

    def test_item_error_non_scalar(self):
        t = Tensor([1.0, 2.0])
        with pytest.raises(ValueError):
            t.item()

    def test_repr(self):
        t = Tensor([1.0, 2.0])
        assert "Tensor" in repr(t)

    def test_zeros_with_grad(self):
        t = Tensor.zeros((3,), requires_grad=True)
        assert all(isinstance(v, Var) for v in t.data)

    def test_randn_with_grad(self):
        random.seed(42)
        t = Tensor.randn((5,), requires_grad=True)
        assert all(isinstance(v, Var) for v in t.data)


# ===== Parameter =====

class TestParameter:
    def test_create(self):
        t = Tensor([1.0, 2.0, 3.0])
        p = Parameter(t)
        assert p.shape == (3,)
        assert all(isinstance(v, Var) for v in p.data)

    def test_values(self):
        t = Tensor([1.0, 2.0, 3.0])
        p = Parameter(t)
        vals = p.values()
        assert vals == [1.0, 2.0, 3.0]

    def test_update(self):
        t = Tensor([1.0, 2.0])
        p = Parameter(t)
        p.update([5.0, 6.0])
        assert p.values() == [5.0, 6.0]

    def test_zero_grad(self):
        t = Tensor([1.0, 2.0])
        p = Parameter(t)
        p.data[0].grad = 5.0
        p.zero_grad()
        assert p.data[0].grad == 0.0

    def test_collect_grads(self):
        t = Tensor([1.0, 2.0])
        p = Parameter(t)
        p.data[0].grad = 3.0
        p.data[1].grad = 4.0
        p.collect_grads()
        assert p.grads() == [3.0, 4.0]

    def test_repr(self):
        p = Parameter(Tensor([1.0]))
        assert "Parameter" in repr(p)


# ===== Init =====

class TestInit:
    def test_xavier_uniform(self):
        random.seed(42)
        p = Init.xavier_uniform((4, 3))
        assert p.shape == (4, 3)
        vals = p.values()
        limit = math.sqrt(6.0 / (3 + 4))
        assert all(-limit - 0.01 <= v <= limit + 0.01 for v in vals)

    def test_xavier_normal(self):
        random.seed(42)
        p = Init.xavier_normal((10, 10))
        assert p.shape == (10, 10)

    def test_he_uniform(self):
        random.seed(42)
        p = Init.he_uniform((5, 3))
        assert p.shape == (5, 3)
        vals = p.values()
        limit = math.sqrt(6.0 / 3)
        assert all(-limit - 0.01 <= v <= limit + 0.01 for v in vals)

    def test_he_normal(self):
        random.seed(42)
        p = Init.he_normal((5, 3))
        assert p.shape == (5, 3)

    def test_zeros_init(self):
        p = Init.zeros((3, 2))
        assert all(v == 0.0 for v in p.values())

    def test_ones_init(self):
        p = Init.ones((2, 2))
        assert all(v == 1.0 for v in p.values())

    def test_constant(self):
        p = Init.constant((3,), 0.5)
        assert all(abs(v - 0.5) < 1e-10 for v in p.values())

    def test_uniform(self):
        random.seed(42)
        p = Init.uniform((10,), -1.0, 1.0)
        vals = p.values()
        assert all(-1.0 <= v <= 1.0 for v in vals)


# ===== Module =====

class TestModule:
    def test_base_module(self):
        m = Module()
        assert m.training
        assert m.parameters() == []

    def test_train_eval(self):
        m = Module()
        m.eval()
        assert not m.training
        m.train()
        assert m.training

    def test_register_parameter(self):
        m = Module()
        p = Parameter(Tensor([1.0]))
        m.register_parameter('w', p)
        assert m.parameters() == [p]

    def test_named_parameters(self):
        m = Module()
        p = Parameter(Tensor([1.0]))
        m.register_parameter('w', p)
        named = m.named_parameters()
        assert named[0] == ('w', p)

    def test_register_module(self):
        parent = Module()
        child = Module()
        p = Parameter(Tensor([1.0]))
        child.register_parameter('w', p)
        parent.register_module('child', child)
        assert len(parent.parameters()) == 1

    def test_num_parameters(self):
        m = Module()
        m.register_parameter('w', Parameter(Tensor([1.0, 2.0, 3.0])))
        assert m.num_parameters() == 3

    def test_state_dict(self):
        m = Module()
        m.register_parameter('w', Parameter(Tensor([1.0, 2.0])))
        state = m.state_dict()
        assert 'w' in state
        assert state['w'] == [1.0, 2.0]

    def test_load_state_dict(self):
        m = Module()
        m.register_parameter('w', Parameter(Tensor([0.0, 0.0])))
        m.load_state_dict({'w': [3.0, 4.0]})
        assert m._parameters['w'].values() == [3.0, 4.0]

    def test_zero_grad(self):
        m = Module()
        p = Parameter(Tensor([1.0]))
        m.register_parameter('w', p)
        p.data[0].grad = 5.0
        m.zero_grad()
        assert p.data[0].grad == 0.0

    def test_nested_train_eval(self):
        parent = Module()
        child = Module()
        parent.register_module('child', child)
        parent.eval()
        assert not child.training
        parent.train()
        assert child.training


# ===== Linear =====

class TestLinear:
    def test_basic(self):
        random.seed(42)
        layer = Linear(3, 2)
        x = [Var(1.0), Var(2.0), Var(3.0)]
        out = layer(x)
        assert len(out) == 2
        assert all(isinstance(v, Var) for v in out)

    def test_no_bias(self):
        random.seed(42)
        layer = Linear(2, 2, bias=False)
        x = [Var(1.0), Var(1.0)]
        out = layer(x)
        assert len(out) == 2

    def test_gradient_flow(self):
        random.seed(42)
        layer = Linear(2, 1)
        x = [Var(1.0), Var(2.0)]
        out = layer(x)
        out[0].backward()
        # Weight grads should be x values
        w = layer._parameters['weight']
        assert w.data[0].grad != 0.0  # has gradient

    def test_parameter_count(self):
        layer = Linear(3, 2)
        # 3*2 weights + 2 biases = 8
        assert layer.num_parameters() == 8

    def test_parameter_count_no_bias(self):
        layer = Linear(3, 2, bias=False)
        assert layer.num_parameters() == 6

    def test_he_init(self):
        random.seed(42)
        layer = Linear(10, 5, init='he')
        assert layer.num_parameters() == 55  # 50 + 5


# ===== Conv1D =====

class TestConv1D:
    def test_basic(self):
        random.seed(42)
        conv = Conv1D(1, 1, kernel_size=3)
        x = [[Var(1.0), Var(2.0), Var(3.0), Var(4.0), Var(5.0)]]
        out = conv(x)
        assert len(out) == 1
        assert len(out[0]) == 3  # (5 - 3) / 1 + 1

    def test_padding(self):
        random.seed(42)
        conv = Conv1D(1, 1, kernel_size=3, padding=1)
        x = [[Var(1.0), Var(2.0), Var(3.0)]]
        out = conv(x)
        assert len(out[0]) == 3  # same size with padding=1

    def test_stride(self):
        random.seed(42)
        conv = Conv1D(1, 1, kernel_size=2, stride=2)
        x = [[Var(1.0), Var(2.0), Var(3.0), Var(4.0)]]
        out = conv(x)
        assert len(out[0]) == 2  # (4 - 2) / 2 + 1

    def test_multi_channel(self):
        random.seed(42)
        conv = Conv1D(2, 3, kernel_size=2)
        x = [[Var(1.0), Var(2.0), Var(3.0)], [Var(4.0), Var(5.0), Var(6.0)]]
        out = conv(x)
        assert len(out) == 3  # 3 output channels
        assert len(out[0]) == 2


# ===== RNNCell =====

class TestRNNCell:
    def test_basic(self):
        random.seed(42)
        rnn = RNNCell(3, 4)
        x = [Var(1.0), Var(0.5), Var(-1.0)]
        h = rnn(x)
        assert len(h) == 4
        assert all(isinstance(v, Var) for v in h)

    def test_with_hidden(self):
        random.seed(42)
        rnn = RNNCell(2, 3)
        x = [Var(1.0), Var(0.0)]
        h0 = [Var(0.1), Var(0.2), Var(0.3)]
        h1 = rnn(x, h0)
        assert len(h1) == 3

    def test_sequential_steps(self):
        random.seed(42)
        rnn = RNNCell(2, 3)
        h = None
        for _ in range(3):
            x = [Var(1.0), Var(0.0)]
            h = rnn(x, h)
        assert len(h) == 3

    def test_gradient_flow(self):
        random.seed(42)
        rnn = RNNCell(2, 2)
        x = [Var(1.0), Var(0.5)]
        h = rnn(x)
        loss = h[0] + h[1]
        loss.backward()
        assert rnn._parameters['W_ih'].data[0].grad != 0.0


# ===== LSTMCell =====

class TestLSTMCell:
    def test_basic(self):
        random.seed(42)
        lstm = LSTMCell(3, 4)
        x = [Var(1.0), Var(0.5), Var(-1.0)]
        h, c = lstm(x)
        assert len(h) == 4
        assert len(c) == 4

    def test_with_state(self):
        random.seed(42)
        lstm = LSTMCell(2, 3)
        x = [Var(1.0), Var(0.0)]
        h0 = [Var(0.0)] * 3
        c0 = [Var(0.0)] * 3
        h, c = lstm(x, (h0, c0))
        assert len(h) == 3

    def test_sequential(self):
        random.seed(42)
        lstm = LSTMCell(2, 3)
        state = None
        for _ in range(5):
            x = [Var(1.0), Var(0.0)]
            h, c = lstm(x, state)
            state = (h, c)
        assert len(h) == 3

    def test_forget_gate_bias(self):
        """Forget gate bias should be initialized to 1."""
        lstm = LSTMCell(2, 3)
        hs = 3
        # Forget gate bias is at indices hs to 2*hs
        for i in range(hs):
            v = lstm._parameters['bias'].data[hs + i]
            assert abs(v.val - 1.0) < 1e-10


# ===== Embedding =====

class TestEmbedding:
    def test_basic(self):
        random.seed(42)
        emb = Embedding(10, 4)
        result = emb([0, 3, 7])
        assert len(result) == 3
        assert len(result[0]) == 4

    def test_same_index_same_embedding(self):
        random.seed(42)
        emb = Embedding(5, 3)
        r1 = emb([2])
        r2 = emb([2])
        for a, b in zip(r1[0], r2[0]):
            assert a.val == b.val

    def test_gradient_flow(self):
        random.seed(42)
        emb = Embedding(5, 2)
        result = emb([1])
        loss = result[0][0] + result[0][1]
        loss.backward()
        # Embedding params for index 1 should have grad
        assert emb._parameters['weight'].data[2].grad != 0.0


# ===== BatchNorm1D =====

class TestBatchNorm1D:
    def test_training(self):
        random.seed(42)
        bn = BatchNorm1D(3)
        batch = [[Var(1.0), Var(2.0), Var(3.0)],
                 [Var(4.0), Var(5.0), Var(6.0)]]
        out = bn(batch)
        assert len(out) == 2
        assert len(out[0]) == 3

    def test_normalized_output(self):
        bn = BatchNorm1D(2)
        batch = [[Var(0.0), Var(10.0)],
                 [Var(2.0), Var(20.0)]]
        out = bn(batch)
        # After normalization, values should be roughly centered
        v0 = out[0][0].val
        v1 = out[1][0].val
        # Should have opposite signs (centered)
        assert v0 * v1 < 0

    def test_eval_mode(self):
        bn = BatchNorm1D(2)
        # Train first
        batch = [[Var(1.0), Var(2.0)], [Var(3.0), Var(4.0)]]
        bn(batch)
        # Eval
        bn.eval()
        out = bn([[Var(2.0), Var(3.0)]])
        assert len(out) == 1

    def test_running_stats_update(self):
        bn = BatchNorm1D(2, momentum=0.1)
        batch = [[Var(0.0), Var(0.0)], [Var(10.0), Var(10.0)]]
        bn(batch)
        # Running mean should be updated
        assert bn.running_mean[0] != 0.0


# ===== Dropout =====

class TestDropout:
    def test_training(self):
        random.seed(42)
        d = Dropout(0.5)
        x = [Var(1.0)] * 100
        out = d(x)
        zeros = sum(1 for v in out if v.val == 0.0)
        assert zeros > 20  # Should drop ~50%

    def test_eval(self):
        d = Dropout(0.5)
        d.eval()
        x = [Var(1.0)] * 10
        out = d(x)
        assert all(v.val == 1.0 for v in out)

    def test_zero_dropout(self):
        d = Dropout(0.0)
        x = [Var(1.0)] * 10
        out = d(x)
        assert all(v.val == 1.0 for v in out)


# ===== Activations =====

class TestActivations:
    def test_relu(self):
        relu = ReLU()
        out = relu([Var(-1.0), Var(0.0), Var(2.0)])
        assert out[0].val == 0.0
        assert out[1].val == 0.0
        assert out[2].val == 2.0

    def test_sigmoid(self):
        sig = Sigmoid()
        out = sig([Var(0.0)])
        assert abs(out[0].val - 0.5) < 1e-10

    def test_tanh(self):
        tanh = Tanh()
        out = tanh([Var(0.0)])
        assert abs(out[0].val) < 1e-10

    def test_leaky_relu(self):
        lr = LeakyReLU(0.1)
        out = lr([Var(-2.0), Var(3.0)])
        assert abs(out[0].val - (-0.2)) < 1e-10
        assert out[1].val == 3.0

    def test_elu(self):
        elu = ELU(1.0)
        out = elu([Var(-1.0), Var(1.0)])
        assert out[0].val < 0  # alpha * (exp(-1) - 1) < 0
        assert out[1].val == 1.0

    def test_gelu(self):
        gelu = GELU()
        out = gelu([Var(0.0), Var(1.0)])
        assert abs(out[0].val) < 0.01  # GELU(0) ~ 0

    def test_softmax(self):
        sm = Softmax()
        out = sm([Var(1.0), Var(2.0), Var(3.0)])
        total = sum(v.val for v in out)
        assert abs(total - 1.0) < 1e-10
        # Should be monotonically increasing
        assert out[0].val < out[1].val < out[2].val

    def test_softmax_numerical_stability(self):
        sm = Softmax()
        out = sm([Var(1000.0), Var(1001.0)])
        assert all(not math.isnan(v.val) and not math.isinf(v.val) for v in out)


# ===== Loss Functions =====

class TestMSELoss:
    def test_zero_loss(self):
        loss_fn = MSELoss()
        pred = [Var(1.0), Var(2.0)]
        target = [1.0, 2.0]
        loss = loss_fn(pred, target)
        assert abs(loss.val) < 1e-10

    def test_nonzero_loss(self):
        loss_fn = MSELoss()
        pred = [Var(1.0), Var(2.0)]
        target = [2.0, 4.0]
        loss = loss_fn(pred, target)
        expected = ((1 - 2)**2 + (2 - 4)**2) / 2  # 2.5
        assert abs(loss.val - expected) < 1e-10

    def test_gradient(self):
        loss_fn = MSELoss()
        x = Var(3.0)
        loss = loss_fn([x], [1.0])
        loss.backward()
        # d/dx (x-1)^2 / 1 = 2*(3-1) = 4
        assert abs(x.grad - 4.0) < 1e-6


class TestL1Loss:
    def test_zero_loss(self):
        loss_fn = L1Loss()
        loss = loss_fn([Var(1.0)], [1.0])
        assert abs(loss.val) < 1e-10

    def test_nonzero(self):
        loss_fn = L1Loss()
        loss = loss_fn([Var(3.0), Var(1.0)], [1.0, 4.0])
        # (|3-1| + |1-4|) / 2 = (2+3)/2 = 2.5
        assert abs(loss.val - 2.5) < 1e-10


class TestCrossEntropyLoss:
    def test_basic(self):
        ce = CrossEntropyLoss()
        logits = [Var(2.0), Var(1.0), Var(0.1)]
        loss = ce(logits, 0)  # target is class 0
        assert loss.val > 0

    def test_correct_class_low_loss(self):
        ce = CrossEntropyLoss()
        # Strong prediction for class 0
        logits = [Var(10.0), Var(0.0), Var(0.0)]
        loss = ce(logits, 0)
        assert loss.val < 0.1

    def test_wrong_class_high_loss(self):
        ce = CrossEntropyLoss()
        logits = [Var(10.0), Var(0.0)]
        loss = ce(logits, 1)  # target is class 1
        assert loss.val > 5.0

    def test_gradient(self):
        ce = CrossEntropyLoss()
        logits = [Var(1.0), Var(2.0)]
        loss = ce(logits, 0)
        loss.backward()
        assert logits[0].grad != 0.0


class TestBinaryCrossEntropyLoss:
    def test_basic(self):
        bce = BinaryCrossEntropyLoss()
        pred = [var_sigmoid(Var(2.0))]
        loss = bce(pred, [1.0])
        assert loss.val > 0

    def test_perfect_pred(self):
        bce = BinaryCrossEntropyLoss()
        pred = [Var(0.999)]
        loss = bce(pred, [1.0])
        assert loss.val < 0.01


class TestHuberLoss:
    def test_small_error(self):
        huber = HuberLoss(delta=1.0)
        loss = huber([Var(0.5)], [0.0])
        # |0.5| < 1.0, so 0.5 * 0.5^2 = 0.125
        assert abs(loss.val - 0.125) < 1e-10

    def test_large_error(self):
        huber = HuberLoss(delta=1.0)
        loss = huber([Var(3.0)], [0.0])
        # |3| > 1, so 1 * (3 - 0.5) = 2.5
        assert abs(loss.val - 2.5) < 1e-10


# ===== Optimizers =====

class TestSGD:
    def test_basic_step(self):
        random.seed(42)
        layer = Linear(2, 1)
        opt = SGD(layer.parameters(), lr=0.1)
        x = [Var(1.0), Var(1.0)]
        out = layer(x)
        out[0].backward()
        before = [p.values() for p in layer.parameters()]
        opt.step()
        after = [p.values() for p in layer.parameters()]
        # Values should change
        assert before != after

    def test_momentum(self):
        p = Parameter(Tensor([5.0]))
        opt = SGD([p], lr=0.1, momentum=0.9)
        for _ in range(3):
            p.data[0] = Var(p.values()[0])
            loss = p.data[0] * p.data[0]
            loss.backward()
            opt.step()
        assert abs(p.values()[0]) < 5.0

    def test_weight_decay(self):
        p = Parameter(Tensor([5.0]))
        opt = SGD([p], lr=0.1, weight_decay=0.01)
        p.data[0] = Var(5.0)
        p.data[0].grad = 0.0  # no gradient, only decay
        opt.step()
        # Should decrease due to weight decay
        assert p.values()[0] < 5.0


class TestAdam:
    def test_basic(self):
        random.seed(42)
        p = Parameter(Tensor([5.0]))
        opt = Adam([p], lr=0.1)
        for _ in range(10):
            p.data[0] = Var(p.values()[0])
            loss = p.data[0] * p.data[0]
            loss.backward()
            opt.step()
        assert abs(p.values()[0]) < 5.0

    def test_convergence(self):
        """Adam should minimize x^2 toward 0."""
        random.seed(42)
        p = Parameter(Tensor([3.0]))
        opt = Adam([p], lr=0.1)
        for _ in range(50):
            p.data[0] = Var(p.values()[0])
            loss = p.data[0] * p.data[0]
            loss.backward()
            opt.step()
        assert abs(p.values()[0]) < 0.5


class TestRMSProp:
    def test_basic(self):
        p = Parameter(Tensor([5.0]))
        opt = RMSProp([p], lr=0.01)
        for _ in range(10):
            p.data[0] = Var(p.values()[0])
            loss = p.data[0] * p.data[0]
            loss.backward()
            opt.step()
        assert abs(p.values()[0]) < 5.0


class TestAdaGrad:
    def test_basic(self):
        p = Parameter(Tensor([5.0]))
        opt = AdaGrad([p], lr=0.5)
        for _ in range(10):
            p.data[0] = Var(p.values()[0])
            loss = p.data[0] * p.data[0]
            loss.backward()
            opt.step()
        assert abs(p.values()[0]) < 5.0


# ===== LR Schedulers =====

class TestSchedulers:
    def test_step_lr(self):
        p = Parameter(Tensor([1.0]))
        opt = SGD([p], lr=0.1)
        sched = StepLR(opt, step_size=2, gamma=0.5)
        assert opt.lr == 0.1
        sched.step()
        sched.step()
        # After 2 steps, lr should be halved
        assert abs(opt.lr - 0.05) < 1e-10

    def test_exponential_lr(self):
        p = Parameter(Tensor([1.0]))
        opt = SGD([p], lr=1.0)
        sched = ExponentialLR(opt, gamma=0.9)
        sched.step()
        assert abs(opt.lr - 0.9) < 1e-10
        sched.step()
        assert abs(opt.lr - 0.81) < 1e-10

    def test_cosine_annealing(self):
        p = Parameter(Tensor([1.0]))
        opt = SGD([p], lr=1.0)
        sched = CosineAnnealingLR(opt, T_max=10, eta_min=0.0)
        sched.step()
        assert opt.lr < 1.0 and opt.lr > 0.0
        for _ in range(9):
            sched.step()
        # At T_max, should be at eta_min
        assert abs(opt.lr - 0.0) < 0.01


# ===== Sequential =====

class TestSequential:
    def test_basic(self):
        random.seed(42)
        model = Sequential(
            Linear(2, 4),
            ReLU(),
            Linear(4, 1)
        )
        x = [Var(1.0), Var(2.0)]
        out = model(x)
        assert len(out) == 1

    def test_parameters(self):
        random.seed(42)
        model = Sequential(
            Linear(2, 3),
            Linear(3, 1)
        )
        # 2*3+3 + 3*1+1 = 13
        assert model.num_parameters() == 13

    def test_add(self):
        model = Sequential()
        model.add(Linear(2, 3))
        model.add(ReLU())
        model.add(Linear(3, 1))
        assert model.num_parameters() == 13

    def test_gradient_flow(self):
        random.seed(42)
        model = Sequential(
            Linear(2, 3),
            Tanh(),
            Linear(3, 1)
        )
        x = [Var(1.0), Var(0.5)]
        out = model(x)
        out[0].backward()
        # All parameters should have gradients
        for p in model.parameters():
            has_grad = any(v.grad != 0.0 for v in p.data if isinstance(v, Var))
            assert has_grad


# ===== Trainer =====

class TestTrainer:
    def test_train_step(self):
        random.seed(42)
        model = Sequential(Linear(2, 1))
        opt = SGD(model.parameters(), lr=0.01)
        loss_fn = MSELoss()
        trainer = Trainer(model, opt, loss_fn)

        x = [Var(1.0), Var(2.0)]
        y = [3.0]
        loss = trainer.train_step(x, y)
        assert isinstance(loss, float)

    def test_fit_reduces_loss(self):
        random.seed(42)
        model = Sequential(Linear(2, 1))
        opt = SGD(model.parameters(), lr=0.01)
        loss_fn = MSELoss()
        trainer = Trainer(model, opt, loss_fn)

        data = [([Var(1.0), Var(0.0)], [1.0]),
                ([Var(0.0), Var(1.0)], [0.0])]

        history = trainer.fit(data, epochs=20)
        assert history.train_losses[-1] < history.train_losses[0]

    def test_evaluate(self):
        random.seed(42)
        model = Sequential(Linear(2, 1))
        opt = SGD(model.parameters(), lr=0.01)
        loss_fn = MSELoss()
        trainer = Trainer(model, opt, loss_fn)

        data = [([Var(1.0), Var(0.0)], [1.0])]
        val_loss = trainer.evaluate(data)
        assert isinstance(val_loss, float)

    def test_with_validation(self):
        random.seed(42)
        model = Sequential(Linear(2, 1))
        opt = SGD(model.parameters(), lr=0.01)
        loss_fn = MSELoss()
        trainer = Trainer(model, opt, loss_fn)

        train_data = [([Var(1.0), Var(0.0)], [1.0]),
                      ([Var(0.0), Var(1.0)], [0.0])]
        val_data = [([Var(0.5), Var(0.5)], [0.5])]

        history = trainer.fit(train_data, epochs=5, val_data=val_data)
        assert len(history.val_losses) == 5

    def test_callback(self):
        random.seed(42)
        model = Sequential(Linear(2, 1))
        opt = SGD(model.parameters(), lr=0.01)
        loss_fn = MSELoss()
        trainer = Trainer(model, opt, loss_fn)
        events = []
        trainer.add_callback(lambda event, **kw: events.append(event))

        data = [([Var(1.0), Var(0.0)], [1.0])]
        trainer.fit(data, epochs=2)
        assert 'train_start' in events
        assert 'epoch_start' in events
        assert 'epoch_end' in events
        assert 'train_end' in events

    def test_with_scheduler(self):
        random.seed(42)
        model = Sequential(Linear(2, 1))
        opt = SGD(model.parameters(), lr=0.1)
        sched = StepLR(opt, step_size=5, gamma=0.5)
        loss_fn = MSELoss()
        trainer = Trainer(model, opt, loss_fn, scheduler=sched)

        data = [([Var(1.0), Var(0.0)], [1.0])]
        trainer.fit(data, epochs=10)
        assert len(trainer.history.lrs) == 10

    def test_grad_clip(self):
        random.seed(42)
        model = Sequential(Linear(2, 1))
        opt = SGD(model.parameters(), lr=0.01)
        loss_fn = MSELoss()
        trainer = Trainer(model, opt, loss_fn, grad_clip=1.0)

        x = [Var(100.0), Var(100.0)]
        y = [0.0]
        loss = trainer.train_step(x, y)
        assert isinstance(loss, float)

    def test_l2_regularization(self):
        random.seed(42)
        model = Sequential(Linear(2, 1))
        opt = SGD(model.parameters(), lr=0.01)
        loss_fn = MSELoss()
        trainer = Trainer(model, opt, loss_fn, l2_reg=0.01)

        data = [([Var(1.0), Var(0.0)], [1.0])]
        trainer.fit(data, epochs=5)
        assert trainer.history.epochs == 5

    def test_batching(self):
        random.seed(42)
        model = Sequential(Linear(2, 1))
        opt = SGD(model.parameters(), lr=0.01)
        loss_fn = MSELoss()
        trainer = Trainer(model, opt, loss_fn)

        data = [([Var(float(i)), Var(float(i))], [float(i)])
                for i in range(10)]
        history = trainer.fit(data, epochs=3, batch_size=3)
        assert history.epochs == 3


# ===== End-to-end Training =====

class TestEndToEnd:
    def test_xor(self):
        """Train a network to learn XOR."""
        random.seed(42)
        model = Sequential(
            Linear(2, 8),
            Tanh(),
            Linear(8, 1),
            Sigmoid()
        )
        opt = Adam(model.parameters(), lr=0.05)
        loss_fn = MSELoss()
        trainer = Trainer(model, opt, loss_fn)

        data = [
            ([Var(0.0), Var(0.0)], [0.0]),
            ([Var(0.0), Var(1.0)], [1.0]),
            ([Var(1.0), Var(0.0)], [1.0]),
            ([Var(1.0), Var(1.0)], [0.0]),
        ]

        trainer.fit(data, epochs=200)

        # Test predictions
        model.eval()
        correct = 0
        for x, y in data:
            out = model(x)
            pred = round(out[0].val)
            if pred == y[0]:
                correct += 1
        assert correct >= 3  # at least 3/4 correct

    def test_classification(self):
        """Train a classifier on separable data."""
        random.seed(42)
        model = Sequential(
            Linear(2, 4),
            ReLU(),
            Linear(4, 2)
        )
        opt = Adam(model.parameters(), lr=0.05)
        loss_fn = CrossEntropyLoss()
        trainer = Trainer(model, opt, loss_fn)

        # Linearly separable data
        data = [
            ([Var(1.0), Var(1.0)], 0),
            ([Var(1.5), Var(1.2)], 0),
            ([Var(-1.0), Var(-1.0)], 1),
            ([Var(-1.5), Var(-0.8)], 1),
        ]

        for _ in range(100):
            for x, y in data:
                trainer.model.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()

        # Check accuracy
        acc = accuracy(model, data)
        assert acc >= 0.75

    def test_regression(self):
        """Train a regression model on y = 2x + 1."""
        random.seed(42)
        model = Linear(1, 1)
        opt = SGD(model.parameters(), lr=0.01)
        loss_fn = MSELoss()

        for _ in range(200):
            x_val = random.uniform(-2, 2)
            y_val = 2 * x_val + 1

            model.zero_grad()
            out = model([Var(x_val)])
            loss = loss_fn(out, [y_val])
            loss.backward()
            opt.step()

        # Test
        model.eval()
        test_x = 1.5
        pred = model([Var(test_x)])[0].val
        expected = 2 * test_x + 1
        assert abs(pred - expected) < 1.0  # reasonable approximation

    def test_state_dict_save_load(self):
        """Save and load model state."""
        random.seed(42)
        model1 = Sequential(Linear(2, 3), Linear(3, 1))
        state = model1.state_dict()

        model2 = Sequential(Linear(2, 3), Linear(3, 1))
        model2.load_state_dict(state)

        x = [Var(1.0), Var(2.0)]
        out1 = model1(x)
        out2 = model2(x)
        assert abs(out1[0].val - out2[0].val) < 1e-10

    def test_rnn_sequence(self):
        """RNN processes a sequence."""
        random.seed(42)
        rnn = RNNCell(2, 3)
        h = None
        sequence = [[Var(1.0), Var(0.0)],
                    [Var(0.0), Var(1.0)],
                    [Var(1.0), Var(1.0)]]
        for x in sequence:
            h = rnn(x, h)
        # Final hidden state should be all tanh values
        assert all(-1.0 <= v.val <= 1.0 for v in h)

    def test_lstm_gradient_flow(self):
        """LSTM should allow gradient flow through time."""
        random.seed(42)
        lstm = LSTMCell(2, 3)
        state = None
        for _ in range(5):
            x = [Var(1.0), Var(0.5)]
            h, c = lstm(x, state)
            state = (h, c)
        loss = h[0] + h[1] + h[2]
        loss.backward()
        # Should have gradients
        assert lstm._parameters['weight'].data[0].grad != 0.0

    def test_conv_gradient(self):
        """Conv1D gradient flow."""
        random.seed(42)
        conv = Conv1D(1, 1, kernel_size=2)
        x = [[Var(1.0), Var(2.0), Var(3.0)]]
        out = conv(x)
        loss = out[0][0] + out[0][1]
        loss.backward()
        assert conv._parameters['weight'].data[0].grad != 0.0

    def test_deep_network(self):
        """Train a deeper network."""
        random.seed(42)
        model = Sequential(
            Linear(2, 8),
            ReLU(),
            Linear(8, 4),
            ReLU(),
            Linear(4, 1)
        )
        opt = Adam(model.parameters(), lr=0.01)
        loss_fn = MSELoss()
        trainer = Trainer(model, opt, loss_fn)

        data = [([Var(float(i)), Var(float(i))], [float(i * 2)])
                for i in range(5)]
        history = trainer.fit(data, epochs=50)
        assert history.train_losses[-1] < history.train_losses[0]


# ===== Numerical Gradient Check =====

class TestGradientCheck:
    def test_linear_grads(self):
        """Check analytical vs numerical gradients for Linear."""
        random.seed(42)
        layer = Linear(2, 1)
        x_vals = [1.0, 2.0]

        def compute_loss():
            x = [Var(v) for v in x_vals]
            out = layer(x)
            return out[0].val

        # Analytical
        x = [Var(v) for v in x_vals]
        out = layer(x)
        out[0].backward()
        analytical = layer._parameters['weight'].data[0].grad

        # Numerical
        ng = num_grad(compute_loss, layer.parameters(), eps=1e-5)
        numerical = ng[0][0]

        assert abs(analytical - numerical) < 1e-4

    def test_mse_grads(self):
        """Check MSE loss gradient."""
        x = Var(3.0)
        loss_fn = MSELoss()
        loss = loss_fn([x], [1.0])
        loss.backward()
        # d/dx (x-1)^2 = 2(x-1) = 4
        assert abs(x.grad - 4.0) < 1e-6


# ===== Utility =====

class TestUtility:
    def test_prod(self):
        assert _prod((2, 3, 4)) == 24
        assert _prod((5,)) == 5
        assert _prod(()) == 1

    def test_randn(self):
        random.seed(42)
        vals = [_randn() for _ in range(1000)]
        mean = sum(vals) / len(vals)
        assert abs(mean) < 0.2

    def test_accuracy(self):
        random.seed(42)
        model = Sequential(Linear(2, 2))
        # Manually set weights so class 0 wins for (1,0) and class 1 for (0,1)
        w = model._modules['0']._parameters['weight']
        w.update([1.0, 0.0, 0.0, 1.0])
        b = model._modules['0']._parameters['bias']
        b.update([0.0, 0.0])

        data = [
            ([Var(1.0), Var(0.0)], 0),
            ([Var(0.0), Var(1.0)], 1),
        ]
        acc = accuracy(model, data)
        assert acc == 1.0

    def test_train_history(self):
        h = TrainHistory()
        h.train_losses.append(1.0)
        h.train_losses.append(0.5)
        h.epochs = 2
        assert len(h.train_losses) == 2

    def test_num_grad(self):
        p = Parameter(Tensor([3.0]))

        def loss_fn():
            return p.values()[0] ** 2

        grads = num_grad(loss_fn, [p])
        # d/dx x^2 at x=3 = 6
        assert abs(grads[0][0] - 6.0) < 1e-3


# ===== Additional Edge Cases =====

class TestEdgeCases:
    def test_empty_sequential(self):
        model = Sequential()
        assert model.num_parameters() == 0

    def test_single_neuron(self):
        random.seed(42)
        model = Linear(1, 1)
        x = [Var(1.0)]
        out = model(x)
        assert len(out) == 1

    def test_large_batch(self):
        random.seed(42)
        model = Sequential(Linear(2, 1))
        opt = SGD(model.parameters(), lr=0.01)
        loss_fn = MSELoss()
        trainer = Trainer(model, opt, loss_fn)

        data = [([Var(float(i % 3)), Var(float(i % 5))], [float(i % 7)])
                for i in range(20)]
        history = trainer.fit(data, epochs=3, batch_size=5)
        assert history.epochs == 3

    def test_dropout_scaling(self):
        """Dropout should scale activations in training mode."""
        random.seed(42)
        d = Dropout(0.5)
        x = [Var(2.0)] * 1000
        out = d(x)
        non_zero = [v.val for v in out if v.val != 0.0]
        if non_zero:
            avg = sum(non_zero) / len(non_zero)
            assert abs(avg - 4.0) < 0.5  # scaled by 1/(1-0.5) = 2

    def test_embedding_different_indices(self):
        random.seed(42)
        emb = Embedding(10, 3)
        r = emb([0, 1, 2])
        # Different indices should give different embeddings
        assert r[0][0].val != r[1][0].val or r[0][1].val != r[1][1].val

    def test_multiple_forward_passes(self):
        """Multiple forward passes should work (new Var nodes each time)."""
        random.seed(42)
        model = Linear(2, 1)
        for _ in range(5):
            x = [Var(1.0), Var(2.0)]
            out = model(x)
            assert len(out) == 1

    def test_l1_regularization(self):
        random.seed(42)
        model = Sequential(Linear(2, 1))
        opt = SGD(model.parameters(), lr=0.01)
        loss_fn = MSELoss()
        trainer = Trainer(model, opt, loss_fn, l1_reg=0.01)

        data = [([Var(1.0), Var(0.0)], [1.0])]
        trainer.fit(data, epochs=5)
        assert trainer.history.epochs == 5

    def test_all_activations_gradient(self):
        """All activation functions should allow gradient flow."""
        activations = [ReLU(), Sigmoid(), Tanh(), LeakyReLU(), ELU(), GELU()]
        for act in activations:
            x = [Var(0.5), Var(-0.5)]
            out = act(x)
            loss = out[0] + out[1]
            loss.backward()
            # At least one input should have a gradient
            has_grad = x[0].grad != 0.0 or x[1].grad != 0.0
            assert has_grad, f"{act.__class__.__name__} has no gradient"

    def test_softmax_gradient(self):
        """Softmax gradient: use single output (not sum, which is always 1)."""
        sm = Softmax()
        x = [Var(0.5), Var(-0.5)]
        out = sm(x)
        out[0].backward()  # d/dx softmax[0]
        has_grad = x[0].grad != 0.0 or x[1].grad != 0.0
        assert has_grad, "Softmax has no gradient"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
