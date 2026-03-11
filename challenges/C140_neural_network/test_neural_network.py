"""
Tests for C140: Neural Network
"""

import math
import random
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from neural_network import (
    Tensor, xavier_init, he_init, lecun_init,
    relu, relu_deriv, sigmoid, sigmoid_deriv, tanh_act, tanh_deriv,
    leaky_relu, leaky_relu_deriv, softmax, softmax_batch,
    MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss,
    Dense, Activation, Dropout, BatchNorm,
    SGD, Adam, RMSProp,
    StepLR, ExponentialLR, CosineAnnealingLR,
    Sequential, train_step, fit, evaluate, predict_classes, accuracy,
    one_hot, normalize, train_test_split,
    make_xor_data, make_spiral_data, make_regression_data, make_circles_data,
    save_weights, load_weights, build_model,
)


# ============================================================
# Tensor tests
# ============================================================

class TestTensor:
    def test_create_1d(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert t.data == [1.0, 2.0, 3.0]

    def test_create_2d(self):
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert t.shape == (2, 2)

    def test_zeros(self):
        t = Tensor.zeros((2, 3))
        assert t.shape == (2, 3)
        assert all(t.data[i][j] == 0.0 for i in range(2) for j in range(3))

    def test_ones(self):
        t = Tensor.ones(5)
        assert t.shape == (5,)
        assert all(x == 1.0 for x in t.data)

    def test_random_normal(self):
        rng = random.Random(42)
        t = Tensor.random_normal((3, 4), rng=rng)
        assert t.shape == (3, 4)

    def test_random_uniform(self):
        rng = random.Random(42)
        t = Tensor.random_uniform((2, 3), low=-1.0, high=1.0, rng=rng)
        assert t.shape == (2, 3)
        for row in t.data:
            for v in row:
                assert -1.0 <= v <= 1.0

    def test_add_1d(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        c = a + b
        assert c.data == [5.0, 7.0, 9.0]

    def test_add_2d(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        c = a + b
        assert c.data == [[6.0, 8.0], [10.0, 12.0]]

    def test_add_broadcast(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([10.0, 20.0])
        c = a + b
        assert c.data == [[11.0, 22.0], [13.0, 24.0]]

    def test_add_scalar(self):
        a = Tensor([1.0, 2.0])
        c = a + 5.0
        assert c.data == [6.0, 7.0]

    def test_sub(self):
        a = Tensor([5.0, 3.0])
        b = Tensor([1.0, 2.0])
        c = a - b
        assert c.data == [4.0, 1.0]

    def test_mul_scalar(self):
        a = Tensor([1.0, 2.0, 3.0])
        c = a * 2.0
        assert c.data == [2.0, 4.0, 6.0]

    def test_mul_element(self):
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])
        c = a * b
        assert c.data == [3.0, 8.0]

    def test_neg(self):
        a = Tensor([1.0, -2.0])
        b = -a
        assert b.data == [-1.0, 2.0]

    def test_div(self):
        a = Tensor([4.0, 6.0])
        b = a / 2.0
        assert b.data == [2.0, 3.0]

    def test_dot_1d(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        assert a.dot(b) == 32.0

    def test_dot_mat_vec(self):
        m = Tensor([[1.0, 2.0], [3.0, 4.0]])
        v = Tensor([1.0, 1.0])
        r = m.dot(v)
        assert r.data == [3.0, 7.0]

    def test_dot_mat_mat(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        c = a.dot(b)
        assert c.data == [[19.0, 22.0], [43.0, 50.0]]

    def test_transpose(self):
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        t = a.T()
        assert t.shape == (3, 2)
        assert t.data[0] == [1.0, 4.0]
        assert t.data[2] == [3.0, 6.0]

    def test_sum(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert a.sum() == 10.0
        s0 = a.sum(axis=0)
        assert s0.data == [4.0, 6.0]
        s1 = a.sum(axis=1)
        assert s1.data == [3.0, 7.0]

    def test_mean(self):
        a = Tensor([[2.0, 4.0], [6.0, 8.0]])
        assert a.mean() == 5.0
        m0 = a.mean(axis=0)
        assert m0.data == [4.0, 6.0]

    def test_flatten(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        f = a.flatten()
        assert f.shape == (4,)
        assert f.data == [1.0, 2.0, 3.0, 4.0]

    def test_reshape(self):
        a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        b = a.reshape((2, 3))
        assert b.shape == (2, 3)
        assert b.data == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    def test_apply(self):
        a = Tensor([1.0, -2.0, 3.0])
        b = a.apply(abs)
        assert b.data == [1.0, 2.0, 3.0]

    def test_argmax_1d(self):
        a = Tensor([1.0, 5.0, 3.0])
        assert a.argmax() == 1

    def test_argmax_2d(self):
        a = Tensor([[1.0, 5.0], [3.0, 2.0]])
        assert a.argmax(axis=1) == [1, 0]

    def test_clip(self):
        a = Tensor([-5.0, 0.5, 10.0])
        b = a.clip(0.0, 1.0)
        assert b.data == [0.0, 0.5, 1.0]

    def test_copy(self):
        a = Tensor([1.0, 2.0])
        b = a.copy()
        b.data[0] = 99.0
        assert a.data[0] == 1.0  # original unchanged

    def test_len(self):
        a = Tensor([1.0, 2.0, 3.0])
        assert len(a) == 3

    def test_getitem_setitem(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        row = a[0]
        assert row.data == [1.0, 2.0]
        a[1] = [5.0, 6.0]
        assert a.data[1] == [5.0, 6.0]

    def test_max_val(self):
        a = Tensor([[1.0, 5.0], [3.0, 2.0]])
        assert a.max_val() == 5.0

    def test_zeros_1d(self):
        t = Tensor.zeros(3)
        assert t.shape == (3,)
        assert t.data == [0.0, 0.0, 0.0]

    def test_empty(self):
        t = Tensor([])
        assert t.shape == (0,)

    def test_radd(self):
        t = Tensor([1.0, 2.0])
        r = 3.0 + t
        assert r.data == [4.0, 5.0]

    def test_rmul(self):
        t = Tensor([1.0, 2.0])
        r = 3.0 * t
        assert r.data == [3.0, 6.0]

    def test_sub_broadcast(self):
        a = Tensor([[5.0, 10.0], [3.0, 6.0]])
        b = Tensor([1.0, 2.0])
        c = a - b
        assert c.data == [[4.0, 8.0], [2.0, 4.0]]


# ============================================================
# Initialization tests
# ============================================================

class TestInitialization:
    def test_xavier(self):
        rng = random.Random(42)
        w = xavier_init(100, 50, rng=rng)
        assert w.shape == (100, 50)
        flat = [v for row in w.data for v in row]
        mean = sum(flat) / len(flat)
        assert abs(mean) < 0.1  # approximately zero mean

    def test_he(self):
        rng = random.Random(42)
        w = he_init(100, 50, rng=rng)
        assert w.shape == (100, 50)

    def test_lecun(self):
        rng = random.Random(42)
        w = lecun_init(100, 50, rng=rng)
        assert w.shape == (100, 50)


# ============================================================
# Activation tests
# ============================================================

class TestActivations:
    def test_relu(self):
        assert relu(5.0) == 5.0
        assert relu(-3.0) == 0.0
        assert relu(0.0) == 0.0

    def test_relu_deriv(self):
        assert relu_deriv(5.0) == 1.0
        assert relu_deriv(-3.0) == 0.0

    def test_sigmoid(self):
        assert abs(sigmoid(0.0) - 0.5) < 1e-10
        assert sigmoid(100) > 0.99
        assert sigmoid(-100) < 0.01

    def test_sigmoid_deriv(self):
        assert abs(sigmoid_deriv(0.0) - 0.25) < 1e-10

    def test_tanh(self):
        assert abs(tanh_act(0.0)) < 1e-10
        assert tanh_act(100) > 0.99
        assert tanh_act(-100) < -0.99

    def test_tanh_deriv(self):
        assert abs(tanh_deriv(0.0) - 1.0) < 1e-10

    def test_leaky_relu(self):
        assert leaky_relu(5.0) == 5.0
        assert abs(leaky_relu(-5.0) - (-0.05)) < 1e-10

    def test_leaky_relu_deriv(self):
        assert leaky_relu_deriv(5.0) == 1.0
        assert leaky_relu_deriv(-5.0) == 0.01

    def test_softmax(self):
        probs = softmax([1.0, 2.0, 3.0])
        assert abs(sum(probs) - 1.0) < 1e-10
        assert probs[2] > probs[1] > probs[0]

    def test_softmax_numerical_stability(self):
        probs = softmax([1000.0, 1001.0, 1002.0])
        assert abs(sum(probs) - 1.0) < 1e-10

    def test_softmax_batch(self):
        t = Tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        out = softmax_batch(t)
        assert out.shape == (2, 3)
        assert abs(sum(out.data[0]) - 1.0) < 1e-10
        assert abs(sum(out.data[1]) - 1.0) < 1e-10

    def test_activation_layer_relu(self):
        act = Activation('relu')
        x = Tensor([-1.0, 0.0, 1.0, 2.0])
        out = act.forward(x)
        assert out.data == [0.0, 0.0, 1.0, 2.0]

    def test_activation_layer_sigmoid(self):
        act = Activation('sigmoid')
        x = Tensor([0.0])
        out = act.forward(x)
        assert abs(out.data[0] - 0.5) < 1e-10

    def test_activation_layer_tanh(self):
        act = Activation('tanh')
        x = Tensor([0.0])
        out = act.forward(x)
        assert abs(out.data[0]) < 1e-10

    def test_activation_backward_relu(self):
        act = Activation('relu')
        x = Tensor([-1.0, 2.0, -3.0, 4.0])
        act.forward(x)
        grad = Tensor([1.0, 1.0, 1.0, 1.0])
        grad_in = act.backward(grad)
        assert grad_in.data == [0.0, 1.0, 0.0, 1.0]

    def test_activation_backward_sigmoid(self):
        act = Activation('sigmoid')
        x = Tensor([0.0])
        act.forward(x)
        grad = Tensor([1.0])
        grad_in = act.backward(grad)
        assert abs(grad_in.data[0] - 0.25) < 1e-10

    def test_activation_linear(self):
        act = Activation('linear')
        x = Tensor([1.0, 2.0])
        out = act.forward(x)
        assert out.data == [1.0, 2.0]
        grad_in = act.backward(Tensor([1.0, 1.0]))
        assert grad_in.data == [1.0, 1.0]


# ============================================================
# Loss function tests
# ============================================================

class TestLossFunctions:
    def test_mse_1d(self):
        loss = MSELoss()
        pred = Tensor([1.0, 2.0, 3.0])
        target = Tensor([1.0, 2.0, 3.0])
        assert loss.forward(pred, target) == 0.0

    def test_mse_nonzero(self):
        loss = MSELoss()
        pred = Tensor([1.0, 2.0])
        target = Tensor([3.0, 4.0])
        # ((1-3)^2 + (2-4)^2) / 2 = 4
        assert loss.forward(pred, target) == 4.0

    def test_mse_backward(self):
        loss = MSELoss()
        pred = Tensor([1.0, 2.0])
        target = Tensor([3.0, 4.0])
        grad = loss.backward(pred, target)
        # 2*(1-3)/2 = -2, 2*(2-4)/2 = -2
        assert abs(grad.data[0] - (-2.0)) < 1e-10
        assert abs(grad.data[1] - (-2.0)) < 1e-10

    def test_mse_batch(self):
        loss = MSELoss()
        pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
        target = Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert loss.forward(pred, target) == 0.0

    def test_cross_entropy_single(self):
        loss = CrossEntropyLoss()
        logits = Tensor([10.0, 0.0, 0.0])
        val = loss.forward(logits, 0)
        assert val < 0.001  # very confident correct prediction

    def test_cross_entropy_wrong(self):
        loss = CrossEntropyLoss()
        logits = Tensor([0.0, 10.0, 0.0])
        val = loss.forward(logits, 0)
        assert val > 5.0  # wrong prediction = high loss

    def test_cross_entropy_batch(self):
        loss = CrossEntropyLoss()
        logits = Tensor([[10.0, 0.0], [0.0, 10.0]])
        targets = [0, 1]
        val = loss.forward(logits, targets)
        assert val < 0.001

    def test_cross_entropy_backward_single(self):
        loss = CrossEntropyLoss()
        logits = Tensor([1.0, 2.0, 3.0])
        grad = loss.backward(logits, 1)
        probs = softmax([1.0, 2.0, 3.0])
        assert abs(grad.data[0] - probs[0]) < 1e-10
        assert abs(grad.data[1] - (probs[1] - 1.0)) < 1e-10

    def test_cross_entropy_backward_batch(self):
        loss = CrossEntropyLoss()
        logits = Tensor([[2.0, 1.0], [1.0, 2.0]])
        targets = [0, 1]
        grad = loss.backward(logits, targets)
        assert grad.shape == (2, 2)

    def test_binary_cross_entropy(self):
        loss = BinaryCrossEntropyLoss()
        pred = Tensor([0.9, 0.1])
        target = Tensor([1.0, 0.0])
        val = loss.forward(pred, target)
        assert val < 0.2  # good prediction

    def test_binary_cross_entropy_bad(self):
        loss = BinaryCrossEntropyLoss()
        pred = Tensor([0.1, 0.9])
        target = Tensor([1.0, 0.0])
        val = loss.forward(pred, target)
        assert val > 1.0  # bad prediction

    def test_binary_cross_entropy_backward(self):
        loss = BinaryCrossEntropyLoss()
        pred = Tensor([0.5, 0.5])
        target = Tensor([1.0, 0.0])
        grad = loss.backward(pred, target)
        assert len(grad.data) == 2

    def test_cross_entropy_one_hot(self):
        loss = CrossEntropyLoss()
        logits = Tensor([10.0, 0.0, 0.0])
        targets = Tensor([1.0, 0.0, 0.0])
        val = loss.forward(logits, targets)
        assert val < 0.001

    def test_mse_backward_batch(self):
        loss = MSELoss()
        pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
        target = Tensor([[0.0, 0.0], [0.0, 0.0]])
        grad = loss.backward(pred, target)
        assert grad.shape == (2, 2)

    def test_bce_batch(self):
        loss = BinaryCrossEntropyLoss()
        pred = Tensor([[0.9, 0.1], [0.1, 0.9]])
        target = Tensor([[1.0, 0.0], [0.0, 1.0]])
        val = loss.forward(pred, target)
        assert val < 0.2

    def test_bce_backward_batch(self):
        loss = BinaryCrossEntropyLoss()
        pred = Tensor([[0.9, 0.1], [0.1, 0.9]])
        target = Tensor([[1.0, 0.0], [0.0, 1.0]])
        grad = loss.backward(pred, target)
        assert grad.shape == (2, 2)


# ============================================================
# Dense layer tests
# ============================================================

class TestDense:
    def test_create(self):
        rng = random.Random(42)
        d = Dense(3, 2, rng=rng)
        assert d.weights.shape == (3, 2)
        assert d.bias.shape == (2,)

    def test_forward_1d(self):
        d = Dense(2, 3, init='zeros')
        d.bias = Tensor([1.0, 2.0, 3.0])
        x = Tensor([1.0, 1.0])
        out = d.forward(x)
        assert out.data == [1.0, 2.0, 3.0]  # weights zero, just bias

    def test_forward_batch(self):
        d = Dense(2, 2, init='zeros')
        d.bias = Tensor([1.0, 2.0])
        x = Tensor([[1.0, 0.0], [0.0, 1.0]])
        out = d.forward(x)
        assert out.shape == (2, 2)
        assert out.data[0] == [1.0, 2.0]
        assert out.data[1] == [1.0, 2.0]

    def test_backward_1d(self):
        d = Dense(2, 2, init='zeros')
        d.weights = Tensor([[1.0, 0.0], [0.0, 1.0]])
        d.bias = Tensor([0.0, 0.0])
        x = Tensor([3.0, 4.0])
        d.forward(x)
        grad_out = Tensor([1.0, 1.0])
        grad_in = d.backward(grad_out)
        assert grad_in.data == [1.0, 1.0]

    def test_backward_batch(self):
        d = Dense(2, 2, init='zeros')
        d.weights = Tensor([[1.0, 0.0], [0.0, 1.0]])
        d.bias = Tensor([0.0, 0.0])
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        d.forward(x)
        grad_out = Tensor([[1.0, 0.0], [0.0, 1.0]])
        grad_in = d.backward(grad_out)
        assert grad_in.shape == (2, 2)

    def test_grad_weights_shape(self):
        d = Dense(3, 2, init='zeros')
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        d.forward(x)
        grad = Tensor([[1.0, 1.0], [1.0, 1.0]])
        d.backward(grad)
        assert d.grad_weights.shape == (3, 2)
        assert d.grad_bias.shape == (2,)

    def test_no_bias(self):
        d = Dense(2, 2, init='zeros', bias=False)
        assert d.bias is None
        x = Tensor([1.0, 2.0])
        out = d.forward(x)
        assert out.data == [0.0, 0.0]

    def test_he_init(self):
        rng = random.Random(42)
        d = Dense(100, 50, init='he', rng=rng)
        assert d.weights.shape == (100, 50)

    def test_forward_identity_weights(self):
        d = Dense(2, 2, init='zeros')
        d.weights = Tensor([[1.0, 0.0], [0.0, 1.0]])
        d.bias = Tensor([0.0, 0.0])
        x = Tensor([5.0, 7.0])
        out = d.forward(x)
        assert abs(out.data[0] - 5.0) < 1e-10
        assert abs(out.data[1] - 7.0) < 1e-10


# ============================================================
# Dropout tests
# ============================================================

class TestDropout:
    def test_training_mode(self):
        rng = random.Random(42)
        d = Dropout(rate=0.5, rng=rng)
        d.training = True
        x = Tensor([1.0] * 100)
        out = d.forward(x)
        # Some should be zero, some should be scaled
        zeros = sum(1 for v in out.data if v == 0.0)
        assert zeros > 10  # some dropped
        assert zeros < 90  # not all dropped

    def test_eval_mode(self):
        d = Dropout(rate=0.5)
        d.training = False
        x = Tensor([1.0, 2.0, 3.0])
        out = d.forward(x)
        assert out.data == [1.0, 2.0, 3.0]  # no dropout in eval

    def test_backward(self):
        rng = random.Random(42)
        d = Dropout(rate=0.5, rng=rng)
        d.training = True
        x = Tensor([1.0, 1.0, 1.0, 1.0])
        d.forward(x)
        grad = Tensor([1.0, 1.0, 1.0, 1.0])
        grad_in = d.backward(grad)
        # Same mask applied
        assert len(grad_in.data) == 4

    def test_zero_rate(self):
        d = Dropout(rate=0.0)
        x = Tensor([1.0, 2.0])
        out = d.forward(x)
        assert out.data == [1.0, 2.0]


# ============================================================
# BatchNorm tests
# ============================================================

class TestBatchNorm:
    def test_create(self):
        bn = BatchNorm(5)
        assert bn.num_features == 5
        assert bn.gamma.shape == (5,)
        assert bn.beta.shape == (5,)

    def test_forward_training(self):
        bn = BatchNorm(2)
        x = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out = bn.forward(x)
        assert out.shape == (3, 2)
        # Should be approximately normalized
        col0 = [out.data[i][0] for i in range(3)]
        assert abs(sum(col0) / 3) < 1e-5  # near-zero mean

    def test_forward_eval(self):
        bn = BatchNorm(2)
        bn.training = False
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        out = bn.forward(x)
        assert out.shape == (2, 2)

    def test_backward(self):
        bn = BatchNorm(2)
        x = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        bn.forward(x)
        grad = Tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        grad_in = bn.backward(grad)
        assert grad_in.shape == (3, 2)

    def test_running_stats_update(self):
        bn = BatchNorm(2, momentum=0.1)
        x = Tensor([[0.0, 0.0], [10.0, 10.0]])
        bn.forward(x)
        # Running mean should have moved from 0
        assert bn.running_mean.data[0] != 0.0

    def test_get_params(self):
        bn = BatchNorm(3)
        params = bn.get_params()
        assert len(params) == 2


# ============================================================
# Optimizer tests
# ============================================================

class TestSGD:
    def test_basic_update(self):
        d = Dense(2, 1, init='zeros')
        d.weights = Tensor([[1.0], [1.0]])
        d.bias = Tensor([0.0])
        x = Tensor([[1.0, 1.0]])
        d.forward(x)
        d.backward(Tensor([[1.0]]))

        opt = SGD(lr=0.1)
        opt.step([d])
        # Weights should have decreased
        assert d.weights.data[0][0] < 1.0

    def test_momentum(self):
        d = Dense(2, 1, init='zeros')
        d.weights = Tensor([[1.0], [1.0]])
        d.bias = Tensor([0.0])
        opt = SGD(lr=0.1, momentum=0.9)

        # Two steps
        x = Tensor([[1.0, 1.0]])
        d.forward(x)
        d.backward(Tensor([[1.0]]))
        opt.step([d])
        w1 = d.weights.data[0][0]

        d.forward(x)
        d.backward(Tensor([[1.0]]))
        opt.step([d])
        w2 = d.weights.data[0][0]

        # With momentum, second step should be larger
        assert w2 < w1

    def test_weight_decay(self):
        d = Dense(2, 1, init='zeros')
        d.weights = Tensor([[5.0], [5.0]])
        d.bias = Tensor([0.0])
        x = Tensor([[0.0, 0.0]])
        d.forward(x)
        d.backward(Tensor([[0.0]]))

        opt = SGD(lr=0.1, weight_decay=0.1)
        opt.step([d])
        # Weights should decrease due to weight decay even with zero grad
        assert d.weights.data[0][0] < 5.0


class TestAdam:
    def test_basic_update(self):
        d = Dense(2, 1, init='zeros')
        d.weights = Tensor([[1.0], [1.0]])
        d.bias = Tensor([0.0])
        x = Tensor([[1.0, 1.0]])
        d.forward(x)
        d.backward(Tensor([[1.0]]))

        opt = Adam(lr=0.01)
        opt.step([d])
        assert d.weights.data[0][0] < 1.0

    def test_multiple_steps(self):
        d = Dense(1, 1, init='zeros')
        d.weights = Tensor([[5.0]])
        d.bias = Tensor([0.0])
        opt = Adam(lr=0.1)

        for _ in range(10):
            x = Tensor([[1.0]])
            d.forward(x)
            d.backward(Tensor([[1.0]]))
            opt.step([d])

        # Should have moved significantly
        assert d.weights.data[0][0] < 5.0


class TestRMSProp:
    def test_basic_update(self):
        d = Dense(2, 1, init='zeros')
        d.weights = Tensor([[1.0], [1.0]])
        d.bias = Tensor([0.0])
        x = Tensor([[1.0, 1.0]])
        d.forward(x)
        d.backward(Tensor([[1.0]]))

        opt = RMSProp(lr=0.01)
        opt.step([d])
        assert d.weights.data[0][0] < 1.0


# ============================================================
# LR Scheduler tests
# ============================================================

class TestSchedulers:
    def test_step_lr(self):
        opt = SGD(lr=0.1)
        sched = StepLR(opt, step_size=2, gamma=0.5)
        sched.step()  # epoch 1: no change
        assert opt.lr == 0.1
        sched.step()  # epoch 2: decay
        assert abs(opt.lr - 0.05) < 1e-10

    def test_exponential_lr(self):
        opt = SGD(lr=0.1)
        sched = ExponentialLR(opt, gamma=0.9)
        sched.step()
        assert abs(opt.lr - 0.09) < 1e-10

    def test_cosine_annealing(self):
        opt = SGD(lr=0.1)
        sched = CosineAnnealingLR(opt, T_max=10)
        for _ in range(5):
            sched.step()
        assert opt.lr < 0.1  # should have decreased
        for _ in range(5):
            sched.step()
        assert abs(opt.lr - 0.0) < 0.01  # near minimum


# ============================================================
# Sequential model tests
# ============================================================

class TestSequential:
    def test_create(self):
        model = Sequential()
        model.add(Dense(2, 3))
        model.add(Activation('relu'))
        model.add(Dense(3, 1))
        assert len(model.layers) == 3

    def test_forward(self):
        model = Sequential()
        d = Dense(2, 2, init='zeros')
        d.weights = Tensor([[1.0, 0.0], [0.0, 1.0]])
        d.bias = Tensor([0.0, 0.0])
        model.add(d)
        x = Tensor([3.0, 5.0])
        out = model.forward(x)
        assert abs(out.data[0] - 3.0) < 1e-10
        assert abs(out.data[1] - 5.0) < 1e-10

    def test_forward_backward(self):
        model = Sequential()
        d = Dense(2, 2, init='zeros')
        d.weights = Tensor([[1.0, 0.0], [0.0, 1.0]])
        d.bias = Tensor([0.0, 0.0])
        model.add(d)
        model.add(Activation('relu'))

        x = Tensor([1.0, -1.0])
        out = model.forward(x)
        assert out.data == [1.0, 0.0]

        grad = Tensor([1.0, 1.0])
        model.backward(grad)

    def test_predict(self):
        model = Sequential()
        d = Dense(2, 2, init='zeros')
        d.bias = Tensor([1.0, 2.0])
        model.add(d)
        model.add(Dropout(0.5))  # should be disabled in predict
        out = model.predict(Tensor([0.0, 0.0]))
        assert out.data == [1.0, 2.0]

    def test_count_params(self):
        model = Sequential()
        model.add(Dense(10, 5))  # 10*5 + 5 = 55
        model.add(Activation('relu'))
        model.add(Dense(5, 2))   # 5*2 + 2 = 12
        assert model.count_params() == 67

    def test_summary(self):
        model = Sequential()
        model.add(Dense(10, 5))
        model.add(Activation('relu'))
        s = model.summary()
        assert "Dense" in s
        assert "Total params" in s

    def test_train_eval_modes(self):
        model = Sequential()
        model.add(Dense(2, 2))
        model.add(Dropout(0.5))
        model.train()
        assert model.layers[1].training is True
        model.eval()
        assert model.layers[1].training is False

    def test_get_trainable_layers(self):
        model = Sequential()
        model.add(Dense(2, 2))
        model.add(Activation('relu'))
        model.add(Dense(2, 1))
        tl = model.get_trainable_layers()
        assert len(tl) == 2  # two Dense layers


# ============================================================
# Training tests
# ============================================================

class TestTraining:
    def test_train_step(self):
        rng = random.Random(42)
        model = Sequential()
        model.add(Dense(2, 1, rng=rng))
        loss_fn = MSELoss()
        opt = SGD(lr=0.01)

        x = Tensor([[1.0, 2.0]])
        y = Tensor([[3.0]])
        loss = train_step(model, x, y, loss_fn, opt)
        assert isinstance(loss, float)

    def test_fit_regression(self):
        rng = random.Random(42)
        X, Y = make_regression_data(n=50, noise=0.01, seed=42)
        model = Sequential()
        model.add(Dense(1, 4, rng=rng))
        model.add(Activation('relu'))
        model.add(Dense(4, 1, rng=rng))
        loss_fn = MSELoss()
        opt = Adam(lr=0.01)

        history = fit(model, X, Y, loss_fn, opt, epochs=100, batch_size=10)
        assert history['loss'][-1] < history['loss'][0]  # loss decreased

    def test_fit_with_validation(self):
        rng = random.Random(42)
        X, Y = make_regression_data(n=100, seed=42)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_ratio=0.2)
        model = Sequential()
        model.add(Dense(1, 4, rng=rng))
        model.add(Activation('relu'))
        model.add(Dense(4, 1, rng=rng))
        loss_fn = MSELoss()
        opt = Adam(lr=0.01)

        history = fit(model, X_train, Y_train, loss_fn, opt, epochs=50,
                     validation_data=(X_test, Y_test))
        assert len(history['val_loss']) == 50

    def test_evaluate(self):
        rng = random.Random(42)
        model = Sequential()
        model.add(Dense(2, 1, rng=rng))
        loss_fn = MSELoss()
        X = Tensor([[1.0, 2.0], [3.0, 4.0]])
        Y = Tensor([[3.0], [7.0]])
        loss = evaluate(model, X, Y, loss_fn)
        assert isinstance(loss, float)


# ============================================================
# XOR learning test
# ============================================================

class TestXOR:
    def test_xor_learning(self):
        """Network should learn XOR function."""
        rng = random.Random(42)
        X, Y_labels = make_xor_data(n=200, seed=42)
        Y = one_hot(Y_labels, 2)

        model = Sequential()
        model.add(Dense(2, 8, init='he', rng=rng))
        model.add(Activation('relu'))
        model.add(Dense(8, 2, init='xavier', rng=rng))

        loss_fn = CrossEntropyLoss()
        opt = Adam(lr=0.01)

        history = fit(model, X, Y_labels, loss_fn, opt, epochs=200, batch_size=32)
        # Loss should decrease
        assert history['loss'][-1] < history['loss'][0]

    def test_xor_accuracy(self):
        """After training, XOR accuracy should be high."""
        rng = random.Random(123)
        # Deterministic XOR data
        X = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        Y = [0, 1, 1, 0]

        model = Sequential()
        model.add(Dense(2, 16, init='he', rng=rng))
        model.add(Activation('relu'))
        model.add(Dense(16, 2, init='xavier', rng=rng))

        loss_fn = CrossEntropyLoss()
        opt = Adam(lr=0.01)
        fit(model, X, Y, loss_fn, opt, epochs=500, shuffle=False)

        acc = accuracy(model, X, Y)
        assert acc >= 0.75  # should learn XOR


# ============================================================
# Classification tests
# ============================================================

class TestClassification:
    def test_predict_classes(self):
        model = Sequential()
        d = Dense(2, 3, init='zeros')
        d.bias = Tensor([0.0, 10.0, 0.0])
        model.add(d)
        preds = predict_classes(model, Tensor([[0.0, 0.0], [0.0, 0.0]]))
        assert preds == [1, 1]

    def test_accuracy_perfect(self):
        model = Sequential()
        d = Dense(2, 3, init='zeros')
        d.bias = Tensor([0.0, 0.0, 10.0])
        model.add(d)
        acc = accuracy(model, Tensor([[0.0, 0.0]]), [2])
        assert acc == 1.0

    def test_accuracy_wrong(self):
        model = Sequential()
        d = Dense(2, 3, init='zeros')
        d.bias = Tensor([10.0, 0.0, 0.0])
        model.add(d)
        acc = accuracy(model, Tensor([[0.0, 0.0]]), [2])
        assert acc == 0.0


# ============================================================
# Data utility tests
# ============================================================

class TestDataUtils:
    def test_one_hot(self):
        oh = one_hot([0, 1, 2], 3)
        assert oh.data == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def test_normalize(self):
        X = Tensor([[0.0, 100.0], [10.0, 200.0]])
        X_norm, means, stds = normalize(X)
        # Each column should have ~zero mean
        col0_mean = (X_norm.data[0][0] + X_norm.data[1][0]) / 2
        assert abs(col0_mean) < 1e-5

    def test_normalize_1d(self):
        X = Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        X_norm, mean, std = normalize(X)
        assert abs(sum(X_norm.data) / len(X_norm.data)) < 1e-5

    def test_train_test_split(self):
        X = Tensor([[i, i] for i in range(10)])
        Y = list(range(10))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_ratio=0.3)
        assert len(X_train) == 7
        assert len(X_test) == 3
        assert len(Y_train) == 7
        assert len(Y_test) == 3

    def test_train_test_split_tensor_Y(self):
        X = Tensor([[i, i] for i in range(10)])
        Y = Tensor([[i] for i in range(10)])
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_ratio=0.2)
        assert len(X_train) == 8
        assert len(Y_train) == 8


# ============================================================
# Data generator tests
# ============================================================

class TestDataGenerators:
    def test_xor_data(self):
        X, Y = make_xor_data(n=100, seed=42)
        assert X.shape == (100, 2)
        assert len(Y) == 100
        assert all(y in (0, 1) for y in Y)

    def test_spiral_data(self):
        X, Y = make_spiral_data(n_per_class=20, n_classes=3)
        assert X.shape == (60, 2)
        assert len(Y) == 60
        assert set(Y) == {0, 1, 2}

    def test_regression_data(self):
        X, Y = make_regression_data(n=50)
        assert X.shape == (50, 1)
        assert Y.shape == (50, 1)

    def test_circles_data(self):
        X, Y = make_circles_data(n=100)
        assert X.shape == (100, 2)
        assert len(Y) == 100
        assert set(Y) == {0, 1}


# ============================================================
# Save/Load weights tests
# ============================================================

class TestSaveLoad:
    def test_save_load(self):
        rng = random.Random(42)
        model = Sequential()
        model.add(Dense(2, 3, rng=rng))
        model.add(Activation('relu'))
        model.add(Dense(3, 1, rng=rng))

        weights = save_weights(model)
        assert len(weights) > 0

        # Create new model with same architecture
        rng2 = random.Random(99)  # different init
        model2 = Sequential()
        model2.add(Dense(2, 3, rng=rng2))
        model2.add(Activation('relu'))
        model2.add(Dense(3, 1, rng=rng2))

        load_weights(model2, weights)

        # Should produce same output
        x = Tensor([1.0, 2.0])
        out1 = model.forward(x)
        out2 = model2.forward(x)
        assert abs(out1.data[0] - out2.data[0]) < 1e-10

    def test_save_keys(self):
        model = Sequential()
        model.add(Dense(2, 3))
        model.add(Dense(3, 1))
        weights = save_weights(model)
        assert "layer_0_param_0" in weights  # Dense weights
        assert "layer_0_param_1" in weights  # Dense bias


# ============================================================
# Build model helper tests
# ============================================================

class TestBuildModel:
    def test_basic(self):
        model = build_model([2, 4, 1])
        # Should have Dense(2,4) + ReLU + Dense(4,1)
        assert len(model.layers) == 3

    def test_custom_activations(self):
        model = build_model([2, 4, 3], activations=['tanh', 'softmax'])
        x = Tensor([1.0, 2.0])
        out = model.forward(x)
        assert len(out.data) == 3

    def test_with_dropout(self):
        model = build_model([2, 4, 3, 1], dropout=0.3)
        # Dense + ReLU + Dropout + Dense + ReLU + Dropout + Dense
        assert any(isinstance(l, Dropout) for l in model.layers)

    def test_deep_model(self):
        model = build_model([10, 8, 6, 4, 2])
        assert model.count_params() > 0


# ============================================================
# Integration tests
# ============================================================

class TestIntegration:
    def test_regression_end_to_end(self):
        """Full regression pipeline."""
        rng = random.Random(42)
        X, Y = make_regression_data(n=100, noise=0.05, seed=42)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
        X_train_n, mean, std = normalize(X_train)

        model = build_model([1, 8, 1], activations=['relu', 'linear'], rng=rng)
        loss_fn = MSELoss()
        opt = Adam(lr=0.01)

        history = fit(model, X_train_n, Y_train, loss_fn, opt, epochs=100, batch_size=16)
        assert history['loss'][-1] < history['loss'][0]

    def test_binary_classification_end_to_end(self):
        """Full binary classification pipeline."""
        rng = random.Random(42)
        X, Y_labels = make_circles_data(n=200, seed=42)
        Y = one_hot(Y_labels, 2)
        X_n, _, _ = normalize(X)

        model = Sequential()
        model.add(Dense(2, 16, init='he', rng=rng))
        model.add(Activation('relu'))
        model.add(Dense(16, 2, init='xavier', rng=rng))

        loss_fn = CrossEntropyLoss()
        opt = Adam(lr=0.01)

        history = fit(model, X_n, Y_labels, loss_fn, opt, epochs=100, batch_size=32)
        assert history['loss'][-1] < history['loss'][0]

    def test_multi_class_classification(self):
        """Three-class spiral classification."""
        rng = random.Random(42)
        X, Y = make_spiral_data(n_per_class=30, n_classes=3, seed=42)
        X_n, _, _ = normalize(X)

        model = Sequential()
        model.add(Dense(2, 32, init='he', rng=rng))
        model.add(Activation('relu'))
        model.add(Dense(32, 16, init='he', rng=rng))
        model.add(Activation('relu'))
        model.add(Dense(16, 3, init='xavier', rng=rng))

        loss_fn = CrossEntropyLoss()
        opt = Adam(lr=0.01)

        history = fit(model, X_n, Y, loss_fn, opt, epochs=200, batch_size=16)
        assert history['loss'][-1] < history['loss'][0]

    def test_with_scheduler(self):
        """Training with learning rate scheduler."""
        rng = random.Random(42)
        X, Y = make_regression_data(n=50, seed=42)
        model = build_model([1, 4, 1], rng=rng)
        loss_fn = MSELoss()
        opt = Adam(lr=0.01)
        sched = StepLR(opt, step_size=25, gamma=0.5)

        history = fit(model, X, Y, loss_fn, opt, epochs=50, scheduler=sched)
        assert opt.lr < 0.01  # LR should have decayed

    def test_with_batch_norm(self):
        """Training with batch normalization."""
        rng = random.Random(42)
        X, Y = make_regression_data(n=50, seed=42)

        model = Sequential()
        model.add(Dense(1, 8, rng=rng))
        model.add(BatchNorm(8))
        model.add(Activation('relu'))
        model.add(Dense(8, 1, rng=rng))

        loss_fn = MSELoss()
        opt = Adam(lr=0.01)

        history = fit(model, X, Y, loss_fn, opt, epochs=50, batch_size=16)
        assert history['loss'][-1] < history['loss'][0]

    def test_gradient_clipping(self):
        """SGD with gradient clipping."""
        rng = random.Random(42)
        model = Sequential()
        model.add(Dense(2, 4, rng=rng))
        model.add(Activation('relu'))
        model.add(Dense(4, 1, rng=rng))

        loss_fn = MSELoss()
        opt = SGD(lr=0.01, clip_norm=1.0)

        X = Tensor([[100.0, 200.0]])
        Y = Tensor([[1.0]])
        # Should not explode with clipping
        loss = train_step(model, X, Y, loss_fn, opt)
        assert math.isfinite(loss)

    def test_rmsprop_training(self):
        """Training with RMSProp."""
        rng = random.Random(42)
        X, Y = make_regression_data(n=50, seed=42)
        model = build_model([1, 4, 1], rng=rng)
        loss_fn = MSELoss()
        opt = RMSProp(lr=0.01)

        history = fit(model, X, Y, loss_fn, opt, epochs=50)
        assert history['loss'][-1] < history['loss'][0]

    def test_sgd_momentum_training(self):
        """Training with SGD + momentum."""
        rng = random.Random(42)
        X, Y = make_regression_data(n=50, seed=42)
        model = build_model([1, 4, 1], rng=rng)
        loss_fn = MSELoss()
        opt = SGD(lr=0.01, momentum=0.9)

        history = fit(model, X, Y, loss_fn, opt, epochs=100)
        assert history['loss'][-1] < history['loss'][0]

    def test_cosine_annealing_schedule(self):
        rng = random.Random(42)
        X, Y = make_regression_data(n=50, seed=42)
        model = build_model([1, 4, 1], rng=rng)
        loss_fn = MSELoss()
        opt = Adam(lr=0.01)
        sched = CosineAnnealingLR(opt, T_max=50)

        history = fit(model, X, Y, loss_fn, opt, epochs=50, scheduler=sched)
        # LR should have decreased
        assert opt.lr < 0.01

    def test_exponential_lr_schedule(self):
        rng = random.Random(42)
        X, Y = make_regression_data(n=50, seed=42)
        model = build_model([1, 4, 1], rng=rng)
        loss_fn = MSELoss()
        opt = Adam(lr=0.01)
        sched = ExponentialLR(opt, gamma=0.95)

        fit(model, X, Y, loss_fn, opt, epochs=20, scheduler=sched)
        expected_lr = 0.01 * (0.95 ** 20)
        assert abs(opt.lr - expected_lr) < 1e-10

    def test_weight_decay_adam(self):
        """Adam with weight decay (L2 regularization)."""
        rng = random.Random(42)
        X, Y = make_regression_data(n=50, seed=42)
        model = build_model([1, 4, 1], rng=rng)
        loss_fn = MSELoss()
        opt = Adam(lr=0.01, weight_decay=0.01)

        history = fit(model, X, Y, loss_fn, opt, epochs=50)
        assert history['loss'][-1] < history['loss'][0]


# ============================================================
# Edge case tests
# ============================================================

class TestEdgeCases:
    def test_single_sample(self):
        rng = random.Random(42)
        model = build_model([2, 3, 1], rng=rng)
        x = Tensor([1.0, 2.0])
        out = model.forward(x)
        assert len(out.data) == 1

    def test_single_neuron(self):
        model = Sequential()
        model.add(Dense(1, 1, init='zeros'))
        x = Tensor([5.0])
        out = model.forward(x)
        assert len(out.data) == 1

    def test_large_batch(self):
        rng = random.Random(42)
        model = build_model([2, 4, 1], rng=rng)
        X = Tensor([[float(i), float(i)] for i in range(100)])
        out = model.forward(X)
        assert out.shape[0] == 100

    def test_zero_input(self):
        rng = random.Random(42)
        model = build_model([3, 2], rng=rng)
        x = Tensor([0.0, 0.0, 0.0])
        out = model.forward(x)
        assert len(out.data) == 2

    def test_negative_inputs(self):
        rng = random.Random(42)
        model = Sequential()
        model.add(Dense(2, 4, rng=rng))
        model.add(Activation('relu'))
        model.add(Dense(4, 1, rng=rng))
        x = Tensor([-5.0, -10.0])
        out = model.forward(x)
        assert math.isfinite(out.data[0])

    def test_leaky_relu_activation(self):
        act = Activation('leaky_relu', alpha=0.1)
        x = Tensor([-2.0, 0.0, 3.0])
        out = act.forward(x)
        assert abs(out.data[0] - (-0.2)) < 1e-10
        assert out.data[1] == 0.0
        assert out.data[2] == 3.0

    def test_leaky_relu_backward(self):
        act = Activation('leaky_relu', alpha=0.1)
        x = Tensor([-2.0, 3.0])
        act.forward(x)
        grad = Tensor([1.0, 1.0])
        grad_in = act.backward(grad)
        assert abs(grad_in.data[0] - 0.1) < 1e-10
        assert grad_in.data[1] == 1.0


# ============================================================
# Gradient checking (numerical vs analytical)
# ============================================================

class TestGradientCheck:
    def _numerical_grad(self, model, x, y, loss_fn, layer_idx, param_idx, is_bias, i, j=None, eps=1e-5):
        """Compute numerical gradient for a single parameter."""
        layers = model.get_trainable_layers()
        layer = layers[layer_idx]
        params = layer.get_params()
        tensor = params[param_idx][0]

        if is_bias or len(tensor.shape) == 1:
            orig = tensor.data[i]
            tensor.data[i] = orig + eps
            out_plus = model.forward(x)
            loss_plus = loss_fn.forward(out_plus, y)
            tensor.data[i] = orig - eps
            out_minus = model.forward(x)
            loss_minus = loss_fn.forward(out_minus, y)
            tensor.data[i] = orig
        else:
            orig = tensor.data[i][j]
            tensor.data[i][j] = orig + eps
            out_plus = model.forward(x)
            loss_plus = loss_fn.forward(out_plus, y)
            tensor.data[i][j] = orig - eps
            out_minus = model.forward(x)
            loss_minus = loss_fn.forward(out_minus, y)
            tensor.data[i][j] = orig

        return (loss_plus - loss_minus) / (2 * eps)

    def test_dense_gradient_check(self):
        """Check analytical gradients match numerical for a simple network."""
        rng = random.Random(42)
        model = Sequential()
        model.add(Dense(2, 3, rng=rng))
        model.add(Dense(3, 1, rng=rng))

        loss_fn = MSELoss()
        x = Tensor([[1.0, 2.0]])
        y = Tensor([[3.0]])

        # Forward + backward
        out = model.forward(x)
        loss = loss_fn.forward(out, y)
        grad = loss_fn.backward(out, y)
        model.backward(grad)

        # Check first layer weight gradient
        tl = model.get_trainable_layers()
        analytical = tl[0].grad_weights.data[0][0]
        numerical = self._numerical_grad(model, x, y, loss_fn, 0, 0, False, 0, 0)
        assert abs(analytical - numerical) < 1e-4, f"analytical={analytical}, numerical={numerical}"

    def test_relu_gradient_check(self):
        """Gradient check with ReLU activation."""
        rng = random.Random(42)
        model = Sequential()
        model.add(Dense(2, 4, rng=rng))
        model.add(Activation('relu'))
        model.add(Dense(4, 1, rng=rng))

        loss_fn = MSELoss()
        x = Tensor([[1.5, -0.5]])
        y = Tensor([[2.0]])

        out = model.forward(x)
        grad = loss_fn.backward(out, y)
        model.backward(grad)

        tl = model.get_trainable_layers()
        # Check output layer bias gradient
        analytical = tl[1].grad_bias.data[0]
        numerical = self._numerical_grad(model, x, y, loss_fn, 1, 1, True, 0)
        assert abs(analytical - numerical) < 1e-4


# ============================================================
# Complex architecture tests
# ============================================================

class TestComplexArchitectures:
    def test_deep_network(self):
        """5-layer deep network."""
        rng = random.Random(42)
        model = build_model([2, 8, 8, 8, 4, 2], rng=rng)
        x = Tensor([[1.0, 2.0]])
        out = model.forward(x)
        assert out.shape == (1, 2)

    def test_wide_network(self):
        """Wide network with many neurons."""
        rng = random.Random(42)
        model = build_model([2, 64, 1], rng=rng)
        x = Tensor([[1.0, 2.0]])
        out = model.forward(x)
        assert out.shape == (1, 1)

    def test_different_activations(self):
        """Network with mixed activations."""
        rng = random.Random(42)
        model = Sequential()
        model.add(Dense(2, 4, rng=rng))
        model.add(Activation('relu'))
        model.add(Dense(4, 4, rng=rng))
        model.add(Activation('tanh'))
        model.add(Dense(4, 1, rng=rng))
        model.add(Activation('sigmoid'))

        x = Tensor([[1.0, 2.0]])
        out = model.forward(x)
        assert 0.0 <= out.data[0][0] <= 1.0  # sigmoid output

    def test_batchnorm_dropout_dense(self):
        """Full architecture: Dense + BN + Activation + Dropout."""
        rng = random.Random(42)
        model = Sequential()
        model.add(Dense(2, 8, rng=rng))
        model.add(BatchNorm(8))
        model.add(Activation('relu'))
        model.add(Dropout(0.3, rng=rng))
        model.add(Dense(8, 1, rng=rng))

        X = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out = model.forward(X)
        assert out.shape == (3, 1)

    def test_autoencoder(self):
        """Autoencoder (encode then decode)."""
        rng = random.Random(42)
        model = Sequential()
        # Encoder
        model.add(Dense(4, 2, rng=rng))
        model.add(Activation('relu'))
        # Decoder
        model.add(Dense(2, 4, rng=rng))
        model.add(Activation('sigmoid'))

        X = Tensor([[0.1, 0.9, 0.1, 0.9],
                    [0.9, 0.1, 0.9, 0.1]])
        loss_fn = MSELoss()
        opt = Adam(lr=0.01)

        # Train to reconstruct
        history = fit(model, X, X, loss_fn, opt, epochs=200, shuffle=False)
        assert history['loss'][-1] < history['loss'][0]


# ============================================================
# Numerical stability tests
# ============================================================

class TestNumericalStability:
    def test_softmax_large_values(self):
        probs = softmax([1000.0, 1000.0, 1000.0])
        for p in probs:
            assert math.isfinite(p)
        assert abs(sum(probs) - 1.0) < 1e-10

    def test_sigmoid_extreme(self):
        assert math.isfinite(sigmoid(500.0))
        assert math.isfinite(sigmoid(-500.0))
        assert sigmoid(500.0) > 0.99
        assert sigmoid(-500.0) < 0.01

    def test_ce_loss_zero_prob(self):
        """Cross entropy should handle near-zero probabilities."""
        loss = CrossEntropyLoss()
        logits = Tensor([-100.0, -100.0, 100.0])
        val = loss.forward(logits, 0)  # target is the low-prob class
        assert math.isfinite(val)

    def test_bce_loss_clipping(self):
        """BCE should handle 0 and 1 predictions."""
        loss = BinaryCrossEntropyLoss()
        pred = Tensor([0.0, 1.0])
        target = Tensor([0.0, 1.0])
        val = loss.forward(pred, target)
        assert math.isfinite(val)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
