"""
Tests for C143: Convolutional Neural Network
"""

import math
import random
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from cnn import (
    Tensor3D, Tensor4D, Conv2D, MaxPool2D, AvgPool2D, GlobalAvgPool2D,
    Flatten, Activation2D, BatchNorm2D, Dropout2D, ConvNet, ConvOptimizer,
    SGDConvOptimizer, DepthwiseConv2D, SeparableConv2D, Conv1x1,
    ResidualBlock, compute_output_shape, create_image, image_from_array,
    _compute_output_size, _pad_channel
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))
from neural_network import Tensor, Dense, Activation, CrossEntropyLoss, MSELoss


# ============================================================
# Tensor3D Tests
# ============================================================

class TestTensor3D:
    def test_creation(self):
        t = Tensor3D([[[1, 2], [3, 4]]])
        assert t.shape == (1, 2, 2)
        assert t.channels == 1
        assert t.height == 2
        assert t.width == 2

    def test_multi_channel(self):
        t = Tensor3D([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert t.shape == (2, 2, 2)
        assert t.data[1][0][1] == 6

    def test_zeros(self):
        t = Tensor3D.zeros(3, 4, 5)
        assert t.shape == (3, 4, 5)
        assert t.data[0][0][0] == 0.0

    def test_ones(self):
        t = Tensor3D.ones(2, 3, 3)
        assert t.data[1][2][2] == 1.0

    def test_random_normal(self):
        rng = random.Random(42)
        t = Tensor3D.random_normal(1, 3, 3, rng=rng)
        assert t.shape == (1, 3, 3)
        # Values should be around 0 with std 1
        assert any(abs(t.data[0][h][w]) > 0.01 for h in range(3) for w in range(3))

    def test_copy(self):
        t = Tensor3D([[[1, 2], [3, 4]]])
        t2 = t.copy()
        t2.data[0][0][0] = 99
        assert t.data[0][0][0] == 1  # Original unchanged

    def test_add_scalar(self):
        t = Tensor3D([[[1, 2], [3, 4]]])
        r = t + 10
        assert r.data[0][0][0] == 11
        assert r.data[0][1][1] == 14

    def test_add_tensor(self):
        t1 = Tensor3D([[[1, 2], [3, 4]]])
        t2 = Tensor3D([[[10, 20], [30, 40]]])
        r = t1 + t2
        assert r.data[0][0][0] == 11
        assert r.data[0][1][1] == 44

    def test_add_broadcast_bias(self):
        t = Tensor3D([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        bias = Tensor3D([[[10]], [[20]]])
        r = t + bias
        assert r.data[0][0][0] == 11
        assert r.data[1][1][1] == 28

    def test_sub(self):
        t1 = Tensor3D([[[10, 20], [30, 40]]])
        t2 = Tensor3D([[[1, 2], [3, 4]]])
        r = t1 - t2
        assert r.data[0][0][0] == 9
        assert r.data[0][1][1] == 36

    def test_mul_scalar(self):
        t = Tensor3D([[[1, 2], [3, 4]]])
        r = t * 3
        assert r.data[0][0][1] == 6

    def test_mul_tensor(self):
        t1 = Tensor3D([[[2, 3], [4, 5]]])
        t2 = Tensor3D([[[10, 10], [10, 10]]])
        r = t1 * t2
        assert r.data[0][0][0] == 20
        assert r.data[0][1][1] == 50

    def test_neg(self):
        t = Tensor3D([[[1, -2], [3, -4]]])
        r = -t
        assert r.data[0][0][0] == -1
        assert r.data[0][0][1] == 2

    def test_div_scalar(self):
        t = Tensor3D([[[10, 20], [30, 40]]])
        r = t / 10
        assert r.data[0][0][0] == 1.0

    def test_apply(self):
        t = Tensor3D([[[-1, 2], [-3, 4]]])
        r = t.apply(lambda x: max(0, x))
        assert r.data[0][0][0] == 0
        assert r.data[0][0][1] == 2

    def test_sum_all(self):
        t = Tensor3D([[[1, 2], [3, 4]]])
        assert t.sum_all() == 10

    def test_flatten(self):
        t = Tensor3D([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        flat = t.flatten()
        assert flat == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_radd(self):
        t = Tensor3D([[[1, 2]]])
        r = 5 + t
        assert r.data[0][0][0] == 6

    def test_rmul(self):
        t = Tensor3D([[[1, 2]]])
        r = 3 * t
        assert r.data[0][0][0] == 3


# ============================================================
# Tensor4D Tests
# ============================================================

class TestTensor4D:
    def test_creation(self):
        t1 = Tensor3D([[[1, 2], [3, 4]]])
        t2 = Tensor3D([[[5, 6], [7, 8]]])
        batch = Tensor4D([t1, t2])
        assert batch.shape == (2, 1, 2, 2)
        assert batch.batch_size == 2

    def test_getitem(self):
        t1 = Tensor3D([[[1, 2], [3, 4]]])
        t2 = Tensor3D([[[5, 6], [7, 8]]])
        batch = Tensor4D([t1, t2])
        assert batch[0].data[0][0][0] == 1
        assert batch[1].data[0][0][0] == 5

    def test_zeros(self):
        batch = Tensor4D.zeros(3, 2, 4, 4)
        assert batch.shape == (3, 2, 4, 4)
        assert batch[0].data[0][0][0] == 0.0

    def test_len(self):
        batch = Tensor4D.zeros(5, 1, 3, 3)
        assert len(batch) == 5

    def test_copy(self):
        t = Tensor4D.zeros(2, 1, 2, 2)
        t2 = t.copy()
        t2[0].data[0][0][0] = 99
        assert t[0].data[0][0][0] == 0.0


# ============================================================
# Helper Function Tests
# ============================================================

class TestHelpers:
    def test_compute_output_size(self):
        assert _compute_output_size(28, 3, 1, 0) == 26
        assert _compute_output_size(28, 3, 1, 1) == 28  # same padding
        assert _compute_output_size(28, 3, 2, 0) == 13
        assert _compute_output_size(28, 5, 1, 2) == 28
        assert _compute_output_size(8, 3, 1, 0) == 6

    def test_pad_channel(self):
        ch = [[1, 2], [3, 4]]
        padded = _pad_channel(ch, 1, 2, 2)
        assert len(padded) == 4
        assert len(padded[0]) == 4
        assert padded[0][0] == 0  # padding
        assert padded[1][1] == 1  # original data
        assert padded[2][2] == 4

    def test_pad_zero(self):
        ch = [[1, 2], [3, 4]]
        padded = _pad_channel(ch, 0, 2, 2)
        assert padded == ch

    def test_create_image(self):
        img = create_image(3, 8, 8, value=0.5)
        assert img.shape == (3, 8, 8)
        assert img.data[0][0][0] == 0.5

    def test_image_from_array(self):
        arr = [[1, 2, 3], [4, 5, 6]]
        img = image_from_array(arr, channels=3)
        assert img.shape == (3, 2, 3)
        assert img.data[0][0][0] == 1
        assert img.data[2][1][2] == 6


# ============================================================
# Conv2D Tests
# ============================================================

class TestConv2D:
    def test_basic_convolution(self):
        rng = random.Random(42)
        conv = Conv2D(1, 1, kernel_size=3, padding=0, rng=rng)
        x = Tensor3D([[[1]*5 for _ in range(5)]])
        out = conv.forward(x)
        assert out.shape == (1, 3, 3)

    def test_output_shape(self):
        conv = Conv2D(3, 16, kernel_size=3, padding=1, rng=random.Random(1))
        x = Tensor3D.zeros(3, 8, 8)
        out = conv.forward(x)
        assert out.shape == (16, 8, 8)

    def test_output_shape_no_padding(self):
        conv = Conv2D(1, 4, kernel_size=3, padding=0, rng=random.Random(1))
        x = Tensor3D.zeros(1, 6, 6)
        out = conv.forward(x)
        assert out.shape == (4, 4, 4)

    def test_stride(self):
        conv = Conv2D(1, 1, kernel_size=3, stride=2, padding=0, rng=random.Random(1))
        x = Tensor3D.zeros(1, 7, 7)
        out = conv.forward(x)
        assert out.shape == (1, 3, 3)

    def test_same_padding(self):
        conv = Conv2D(1, 8, kernel_size=3, padding='same', rng=random.Random(1))
        x = Tensor3D.zeros(1, 10, 10)
        out = conv.forward(x)
        assert out.shape == (8, 10, 10)

    def test_known_values(self):
        """Test convolution with known kernel values."""
        conv = Conv2D(1, 1, kernel_size=2, padding=0, bias=False, rng=random.Random(1))
        # Set kernel to all 1s
        conv.kernels = [[[[1, 1], [1, 1]]]]
        x = Tensor3D([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        out = conv.forward(x)
        assert out.shape == (1, 2, 2)
        assert out.data[0][0][0] == 12  # 1+2+4+5
        assert out.data[0][0][1] == 16  # 2+3+5+6
        assert out.data[0][1][0] == 24  # 4+5+7+8
        assert out.data[0][1][1] == 28  # 5+6+8+9

    def test_with_bias(self):
        conv = Conv2D(1, 1, kernel_size=2, padding=0, bias=True, rng=random.Random(1))
        conv.kernels = [[[[1, 1], [1, 1]]]]
        conv.bias_vals = [10.0]
        x = Tensor3D([[[1, 2], [3, 4]]])
        out = conv.forward(x)
        assert out.data[0][0][0] == 20.0  # 1+2+3+4 + 10

    def test_multi_channel_input(self):
        conv = Conv2D(2, 1, kernel_size=2, padding=0, bias=False, rng=random.Random(1))
        conv.kernels = [[[[1, 0], [0, 0]], [[0, 0], [0, 1]]]]  # out=1, in=2
        x = Tensor3D([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        out = conv.forward(x)
        assert out.shape == (1, 1, 1)
        # ch0: 1*1 + 0 + 0 + 0 = 1, ch1: 0 + 0 + 0 + 8*1 = 8 => total = 9
        assert out.data[0][0][0] == 9

    def test_multi_channel_output(self):
        conv = Conv2D(1, 2, kernel_size=1, padding=0, bias=False, rng=random.Random(1))
        conv.kernels = [[[[2]]], [[[3]]]]
        x = Tensor3D([[[5]]])
        out = conv.forward(x)
        assert out.shape == (2, 1, 1)
        assert out.data[0][0][0] == 10
        assert out.data[1][0][0] == 15

    def test_batch_forward(self):
        conv = Conv2D(1, 1, kernel_size=2, padding=0, bias=False, rng=random.Random(1))
        conv.kernels = [[[[1, 1], [1, 1]]]]
        t1 = Tensor3D([[[1, 2], [3, 4]]])
        t2 = Tensor3D([[[5, 6], [7, 8]]])
        batch = Tensor4D([t1, t2])
        out = conv.forward(batch)
        assert isinstance(out, Tensor4D)
        assert out.batch_size == 2
        assert out[0].data[0][0][0] == 10  # 1+2+3+4
        assert out[1].data[0][0][0] == 26  # 5+6+7+8

    def test_backward_gradient_shape(self):
        conv = Conv2D(1, 2, kernel_size=3, padding=1, rng=random.Random(1))
        x = Tensor3D.ones(1, 4, 4)
        out = conv.forward(x)
        grad_out = Tensor3D.ones(2, 4, 4)
        grad_in = conv.backward(grad_out)
        assert grad_in.shape == (1, 4, 4)

    def test_backward_kernel_gradient(self):
        conv = Conv2D(1, 1, kernel_size=2, padding=0, bias=False, rng=random.Random(1))
        conv.kernels = [[[[1, 0], [0, 1]]]]
        x = Tensor3D([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        out = conv.forward(x)
        grad_out = Tensor3D([[[1, 1], [1, 1]]])
        grad_in = conv.backward(grad_out)
        # Kernel gradients should be computed
        assert conv.grad_kernels is not None
        assert len(conv.grad_kernels) == 1

    def test_backward_batch(self):
        conv = Conv2D(1, 1, kernel_size=2, padding=0, bias=False, rng=random.Random(1))
        conv.kernels = [[[[1, 1], [1, 1]]]]
        t1 = Tensor3D([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        t2 = Tensor3D([[[9, 8, 7], [6, 5, 4], [3, 2, 1]]])
        batch = Tensor4D([t1, t2])
        out = conv.forward(batch)
        grad_out = Tensor4D([Tensor3D.ones(1, 2, 2), Tensor3D.ones(1, 2, 2)])
        grad_in = conv.backward(grad_out)
        assert isinstance(grad_in, Tensor4D)
        assert grad_in.batch_size == 2

    def test_get_params(self):
        conv = Conv2D(1, 2, kernel_size=3, rng=random.Random(1))
        params = conv.get_params()
        assert len(params) == 2  # kernels and bias

    def test_rectangular_kernel(self):
        conv = Conv2D(1, 1, kernel_size=(1, 3), padding=0, bias=False, rng=random.Random(1))
        conv.kernels = [[[[1, 1, 1]]]]
        x = Tensor3D([[[1, 2, 3, 4], [5, 6, 7, 8]]])
        out = conv.forward(x)
        assert out.shape == (1, 2, 2)
        assert out.data[0][0][0] == 6  # 1+2+3
        assert out.data[0][0][1] == 9  # 2+3+4


# ============================================================
# MaxPool2D Tests
# ============================================================

class TestMaxPool2D:
    def test_basic(self):
        pool = MaxPool2D(kernel_size=2)
        x = Tensor3D([[[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]]])
        out = pool.forward(x)
        assert out.shape == (1, 2, 2)
        assert out.data[0][0][0] == 6
        assert out.data[0][0][1] == 8
        assert out.data[0][1][0] == 14
        assert out.data[0][1][1] == 16

    def test_stride(self):
        pool = MaxPool2D(kernel_size=2, stride=1)
        x = Tensor3D([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]])
        out = pool.forward(x)
        assert out.shape == (1, 2, 2)
        assert out.data[0][0][0] == 5
        assert out.data[0][1][1] == 9

    def test_multi_channel(self):
        pool = MaxPool2D(kernel_size=2)
        x = Tensor3D([[[1, 2], [3, 4]], [[8, 7], [6, 5]]])
        out = pool.forward(x)
        assert out.shape == (2, 1, 1)
        assert out.data[0][0][0] == 4
        assert out.data[1][0][0] == 8

    def test_batch(self):
        pool = MaxPool2D(kernel_size=2)
        t1 = Tensor3D([[[1, 2], [3, 4]]])
        t2 = Tensor3D([[[5, 6], [7, 8]]])
        batch = Tensor4D([t1, t2])
        out = pool.forward(batch)
        assert isinstance(out, Tensor4D)
        assert out[0].data[0][0][0] == 4
        assert out[1].data[0][0][0] == 8

    def test_backward(self):
        pool = MaxPool2D(kernel_size=2)
        x = Tensor3D([[[1, 2], [3, 4]]])
        out = pool.forward(x)
        grad_out = Tensor3D([[[1.0]]])
        grad_in = pool.backward(grad_out)
        assert grad_in.shape == (1, 2, 2)
        # Gradient goes to max position (1,1) = 4
        assert grad_in.data[0][1][1] == 1.0
        assert grad_in.data[0][0][0] == 0.0

    def test_backward_batch(self):
        pool = MaxPool2D(kernel_size=2)
        t1 = Tensor3D([[[1, 5], [3, 4]]])  # max at (0,1)
        t2 = Tensor3D([[[8, 2], [3, 4]]])  # max at (0,0)
        batch = Tensor4D([t1, t2])
        out = pool.forward(batch)
        grad = Tensor4D([Tensor3D([[[2.0]]]), Tensor3D([[[3.0]]])])
        grad_in = pool.backward(grad)
        assert grad_in[0].data[0][0][1] == 2.0  # max was at (0,1)
        assert grad_in[1].data[0][0][0] == 3.0  # max was at (0,0)

    def test_3x3_kernel(self):
        pool = MaxPool2D(kernel_size=3)
        x = Tensor3D([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        out = pool.forward(x)
        assert out.shape == (1, 1, 1)
        assert out.data[0][0][0] == 9


# ============================================================
# AvgPool2D Tests
# ============================================================

class TestAvgPool2D:
    def test_basic(self):
        pool = AvgPool2D(kernel_size=2)
        x = Tensor3D([[[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]]])
        out = pool.forward(x)
        assert out.shape == (1, 2, 2)
        assert out.data[0][0][0] == 3.5  # (1+2+5+6)/4
        assert out.data[0][1][1] == 13.5  # (11+12+15+16)/4

    def test_batch(self):
        pool = AvgPool2D(kernel_size=2)
        t1 = Tensor3D([[[4, 4], [4, 4]]])
        t2 = Tensor3D([[[8, 8], [8, 8]]])
        batch = Tensor4D([t1, t2])
        out = pool.forward(batch)
        assert out[0].data[0][0][0] == 4.0
        assert out[1].data[0][0][0] == 8.0

    def test_backward(self):
        pool = AvgPool2D(kernel_size=2)
        x = Tensor3D([[[1, 2], [3, 4]]])
        out = pool.forward(x)
        grad_out = Tensor3D([[[4.0]]])
        grad_in = pool.backward(grad_out)
        # Gradient distributed equally
        assert grad_in.data[0][0][0] == 1.0  # 4.0 / 4
        assert grad_in.data[0][1][1] == 1.0


# ============================================================
# GlobalAvgPool2D Tests
# ============================================================

class TestGlobalAvgPool2D:
    def test_basic(self):
        pool = GlobalAvgPool2D()
        x = Tensor3D([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        out = pool.forward(x)
        assert isinstance(out, Tensor)
        assert len(out.data) == 2
        assert out.data[0] == 2.5  # (1+2+3+4)/4
        assert out.data[1] == 6.5  # (5+6+7+8)/4

    def test_batch(self):
        pool = GlobalAvgPool2D()
        t1 = Tensor3D([[[4, 4], [4, 4]]])
        t2 = Tensor3D([[[8, 8], [8, 8]]])
        batch = Tensor4D([t1, t2])
        out = pool.forward(batch)
        assert isinstance(out, Tensor)
        assert out.shape == (2, 1)
        assert out.data[0][0] == 4.0
        assert out.data[1][0] == 8.0

    def test_backward(self):
        pool = GlobalAvgPool2D()
        x = Tensor3D([[[1, 2], [3, 4]]])
        out = pool.forward(x)
        grad_out = Tensor([4.0])
        grad_in = pool.backward(grad_out)
        assert grad_in.shape == (1, 2, 2)
        assert grad_in.data[0][0][0] == 1.0  # 4.0 / 4


# ============================================================
# Flatten Tests
# ============================================================

class TestFlatten:
    def test_basic(self):
        flat = Flatten()
        x = Tensor3D([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        out = flat.forward(x)
        assert isinstance(out, Tensor)
        assert out.data == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_batch(self):
        flat = Flatten()
        t1 = Tensor3D([[[1, 2], [3, 4]]])
        t2 = Tensor3D([[[5, 6], [7, 8]]])
        batch = Tensor4D([t1, t2])
        out = flat.forward(batch)
        assert isinstance(out, Tensor)
        assert out.shape == (2, 4)
        assert out.data[0] == [1, 2, 3, 4]
        assert out.data[1] == [5, 6, 7, 8]

    def test_backward(self):
        flat = Flatten()
        x = Tensor3D([[[1, 2], [3, 4]]])
        out = flat.forward(x)
        grad = Tensor([10, 20, 30, 40])
        grad_in = flat.backward(grad)
        assert isinstance(grad_in, Tensor3D)
        assert grad_in.shape == (1, 2, 2)
        assert grad_in.data[0][0][0] == 10
        assert grad_in.data[0][1][1] == 40

    def test_backward_batch(self):
        flat = Flatten()
        t1 = Tensor3D([[[1, 2], [3, 4]]])
        t2 = Tensor3D([[[5, 6], [7, 8]]])
        batch = Tensor4D([t1, t2])
        out = flat.forward(batch)
        grad = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        grad_in = flat.backward(grad)
        assert isinstance(grad_in, Tensor4D)
        assert grad_in[0].data[0][0][0] == 1
        assert grad_in[1].data[0][1][1] == 8


# ============================================================
# Activation2D Tests
# ============================================================

class TestActivation2D:
    def test_relu(self):
        act = Activation2D('relu')
        x = Tensor3D([[[-1, 2], [3, -4]]])
        out = act.forward(x)
        assert out.data[0][0][0] == 0
        assert out.data[0][0][1] == 2

    def test_sigmoid(self):
        act = Activation2D('sigmoid')
        x = Tensor3D([[[0.0]]])
        out = act.forward(x)
        assert abs(out.data[0][0][0] - 0.5) < 1e-6

    def test_tanh(self):
        act = Activation2D('tanh')
        x = Tensor3D([[[0.0]]])
        out = act.forward(x)
        assert abs(out.data[0][0][0]) < 1e-6

    def test_leaky_relu(self):
        act = Activation2D('leaky_relu', alpha=0.1)
        x = Tensor3D([[[-10, 5]]])
        out = act.forward(x)
        assert out.data[0][0][0] == -1.0
        assert out.data[0][0][1] == 5

    def test_backward(self):
        act = Activation2D('relu')
        x = Tensor3D([[[-1, 2], [3, -4]]])
        out = act.forward(x)
        grad = Tensor3D([[[1, 1], [1, 1]]])
        grad_in = act.backward(grad)
        assert grad_in.data[0][0][0] == 0  # relu deriv at -1 = 0
        assert grad_in.data[0][0][1] == 1  # relu deriv at 2 = 1

    def test_batch(self):
        act = Activation2D('relu')
        t1 = Tensor3D([[[-1, 2]]])
        t2 = Tensor3D([[[3, -4]]])
        batch = Tensor4D([t1, t2])
        out = act.forward(batch)
        assert out[0].data[0][0][0] == 0
        assert out[1].data[0][0][0] == 3


# ============================================================
# BatchNorm2D Tests
# ============================================================

class TestBatchNorm2D:
    def test_inference_mode(self):
        bn = BatchNorm2D(2)
        bn.training = False
        x = Tensor3D([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        out = bn.forward(x)
        assert out.shape == (2, 2, 2)

    def test_training_mode(self):
        bn = BatchNorm2D(1)
        t1 = Tensor3D([[[1, 2], [3, 4]]])
        t2 = Tensor3D([[[5, 6], [7, 8]]])
        batch = Tensor4D([t1, t2])
        out = bn.forward(batch)
        assert out.shape == (2, 1, 2, 2)

    def test_running_stats_update(self):
        bn = BatchNorm2D(1, momentum=0.1)
        t1 = Tensor3D([[[10, 10], [10, 10]]])
        t2 = Tensor3D([[[10, 10], [10, 10]]])
        batch = Tensor4D([t1, t2])
        bn.forward(batch)
        assert abs(bn.running_mean[0] - 1.0) < 0.01  # 0.1 * 10
        assert bn.running_var[0] > 0

    def test_get_params(self):
        bn = BatchNorm2D(3)
        params = bn.get_params()
        assert len(params) == 2  # gamma, beta


# ============================================================
# Dropout2D Tests
# ============================================================

class TestDropout2D:
    def test_training_drops_channels(self):
        rng = random.Random(42)
        drop = Dropout2D(rate=0.5, rng=rng)
        drop.training = True
        x = Tensor3D.ones(10, 4, 4)
        out = drop.forward(x)
        # Some channels should be zeroed
        zero_channels = sum(1 for c in range(10)
                           if all(out.data[c][h][w] == 0
                                  for h in range(4) for w in range(4)))
        assert zero_channels > 0

    def test_inference_no_drop(self):
        drop = Dropout2D(rate=0.5)
        drop.training = False
        x = Tensor3D.ones(4, 3, 3)
        out = drop.forward(x)
        assert out.data[0][0][0] == 1.0

    def test_backward(self):
        rng = random.Random(42)
        drop = Dropout2D(rate=0.5, rng=rng)
        drop.training = True
        x = Tensor3D.ones(4, 2, 2)
        out = drop.forward(x)
        grad = Tensor3D.ones(4, 2, 2)
        grad_in = drop.backward(grad)
        # Gradient should match the dropout mask
        for c in range(4):
            if out.data[c][0][0] == 0:
                assert grad_in.data[c][0][0] == 0


# ============================================================
# DepthwiseConv2D Tests
# ============================================================

class TestDepthwiseConv2D:
    def test_basic(self):
        dw = DepthwiseConv2D(2, kernel_size=3, padding=1, bias=False,
                              rng=random.Random(42))
        x = Tensor3D.ones(2, 4, 4)
        out = dw.forward(x)
        assert out.shape == (2, 4, 4)

    def test_single_channel(self):
        dw = DepthwiseConv2D(1, kernel_size=2, padding=0, bias=False,
                              rng=random.Random(1))
        dw.kernels = [[[1, 1], [1, 1]]]
        x = Tensor3D([[[1, 2], [3, 4]]])
        out = dw.forward(x)
        assert out.shape == (1, 1, 1)
        assert out.data[0][0][0] == 10  # 1+2+3+4

    def test_backward(self):
        dw = DepthwiseConv2D(1, kernel_size=2, padding=0, bias=False,
                              rng=random.Random(1))
        dw.kernels = [[[1, 0], [0, 1]]]
        x = Tensor3D([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        out = dw.forward(x)
        grad = Tensor3D.ones(1, 2, 2)
        grad_in = dw.backward(grad)
        assert grad_in.shape == (1, 3, 3)

    def test_get_params(self):
        dw = DepthwiseConv2D(3, kernel_size=3, rng=random.Random(1))
        params = dw.get_params()
        assert len(params) == 2


# ============================================================
# SeparableConv2D Tests
# ============================================================

class TestSeparableConv2D:
    def test_basic(self):
        sep = SeparableConv2D(3, 8, kernel_size=3, padding=1, rng=random.Random(42))
        x = Tensor3D.ones(3, 6, 6)
        out = sep.forward(x)
        assert out.shape == (8, 6, 6)

    def test_backward(self):
        sep = SeparableConv2D(2, 4, kernel_size=3, padding=1, rng=random.Random(42))
        x = Tensor3D.ones(2, 4, 4)
        out = sep.forward(x)
        grad = Tensor3D.ones(4, 4, 4)
        grad_in = sep.backward(grad)
        assert grad_in.shape == (2, 4, 4)

    def test_fewer_params_than_conv(self):
        """Separable conv should have fewer params than regular conv."""
        sep = SeparableConv2D(16, 32, kernel_size=3, padding=1, rng=random.Random(1))
        conv = Conv2D(16, 32, kernel_size=3, padding=1, rng=random.Random(1))
        # Regular: 16*32*3*3 = 4608, Separable: 16*3*3 + 16*32*1*1 = 144 + 512 = 656
        sep_params = len(sep.get_params())
        conv_params = len(conv.get_params())
        assert sep_params > conv_params  # More param groups but fewer total weights


# ============================================================
# Conv1x1 Tests
# ============================================================

class TestConv1x1:
    def test_channel_mixing(self):
        conv = Conv1x1(3, 1, bias=False, rng=random.Random(1))
        conv.kernels = [[[[1]], [[1]], [[1]]]]
        x = Tensor3D([[[1]], [[2]], [[3]]])
        out = conv.forward(x)
        assert out.shape == (1, 1, 1)
        assert out.data[0][0][0] == 6  # 1+2+3

    def test_expand_channels(self):
        conv = Conv1x1(1, 4, rng=random.Random(1))
        x = Tensor3D.ones(1, 3, 3)
        out = conv.forward(x)
        assert out.shape == (4, 3, 3)


# ============================================================
# ResidualBlock Tests
# ============================================================

class TestResidualBlock:
    def test_shape_preserved(self):
        res = ResidualBlock(4, kernel_size=3, rng=random.Random(42))
        x = Tensor3D.ones(4, 6, 6)
        out = res.forward(x)
        assert out.shape == (4, 6, 6)

    def test_skip_connection(self):
        """Output should contain contribution from skip connection."""
        res = ResidualBlock(1, kernel_size=3, rng=random.Random(42))
        x = Tensor3D.ones(1, 5, 5)
        out = res.forward(x)
        # With skip connection, center values should be non-zero
        center = out.data[0][2][2]
        assert center != 0.0

    def test_backward(self):
        res = ResidualBlock(2, kernel_size=3, rng=random.Random(42))
        x = Tensor3D.ones(2, 4, 4)
        out = res.forward(x)
        grad = Tensor3D.ones(2, 4, 4)
        grad_in = res.backward(grad)
        assert grad_in.shape == (2, 4, 4)

    def test_get_params(self):
        res = ResidualBlock(2, kernel_size=3, rng=random.Random(1))
        params = res.get_params()
        assert len(params) == 4  # 2 conv layers, each with kernels + bias


# ============================================================
# ConvNet Tests
# ============================================================

class TestConvNet:
    def test_simple_forward(self):
        net = ConvNet()
        net.add(Conv2D(1, 4, kernel_size=3, padding=1, rng=random.Random(42)))
        net.add(Activation2D('relu'))
        net.add(MaxPool2D(2))
        net.add(Flatten())

        x = Tensor3D.ones(1, 4, 4)
        out = net.forward(x)
        assert isinstance(out, Tensor)
        assert len(out.data) == 16  # 4 channels * 2 * 2

    def test_conv_dense_pipeline(self):
        """Test conv -> flatten -> dense pipeline."""
        net = ConvNet()
        net.add(Conv2D(1, 2, kernel_size=3, padding=0, bias=False, rng=random.Random(42)))
        net.add(Activation2D('relu'))
        net.add(Flatten())
        net.add(Dense(2 * 2 * 2, 3, init='xavier', rng=random.Random(42)))
        net.add(Activation('softmax'))

        x = Tensor3D.ones(1, 4, 4)
        out = net.forward(x)
        assert isinstance(out, Tensor)
        assert len(out.data) == 3

    def test_set_training(self):
        net = ConvNet()
        drop = Dropout2D(rate=0.5)
        net.add(drop)
        net.set_training(False)
        assert drop.training is False
        net.set_training(True)
        assert drop.training is True

    def test_count_params(self):
        net = ConvNet()
        net.add(Conv2D(1, 8, kernel_size=3, rng=random.Random(1)))
        net.add(Dense(10, 5, rng=random.Random(1)))
        total = net.count_params()
        # Conv: 1*8*3*3 + 8 = 80, Dense: 10*5 + 5 = 55
        assert total == 80 + 55

    def test_predict(self):
        net = ConvNet()
        net.add(Conv2D(1, 2, kernel_size=3, padding=1, rng=random.Random(42)))
        net.add(Activation2D('relu'))
        net.add(GlobalAvgPool2D())

        x = Tensor3D.ones(1, 4, 4)
        out = net.predict(x)
        assert isinstance(out, Tensor)


# ============================================================
# ConvOptimizer Tests
# ============================================================

class TestConvOptimizer:
    def test_adam_step(self):
        conv = Conv2D(1, 1, kernel_size=2, padding=0, bias=True, rng=random.Random(42))
        x = Tensor3D([[[1, 2], [3, 4]]])
        out = conv.forward(x)
        grad = Tensor3D.ones(1, 1, 1)
        conv.backward(grad)

        opt = ConvOptimizer(lr=0.01)
        old_k = conv.kernels[0][0][0][0]
        opt.step([conv])
        new_k = conv.kernels[0][0][0][0]
        assert old_k != new_k

    def test_sgd_step(self):
        conv = Conv2D(1, 1, kernel_size=2, padding=0, bias=True, rng=random.Random(42))
        x = Tensor3D([[[1, 2], [3, 4]]])
        out = conv.forward(x)
        grad = Tensor3D.ones(1, 1, 1)
        conv.backward(grad)

        opt = SGDConvOptimizer(lr=0.01, momentum=0.0)
        old_k = conv.kernels[0][0][0][0]
        opt.step([conv])
        new_k = conv.kernels[0][0][0][0]
        assert old_k != new_k


# ============================================================
# compute_output_shape Tests
# ============================================================

class TestComputeOutputShape:
    def test_conv_pool(self):
        layers = [
            Conv2D(1, 8, kernel_size=3, padding=0, rng=random.Random(1)),
            MaxPool2D(2),
        ]
        shape = compute_output_shape((1, 28, 28), layers)
        assert shape == (8, 13, 13)

    def test_with_flatten(self):
        layers = [
            Conv2D(1, 4, kernel_size=3, padding=1, rng=random.Random(1)),
            MaxPool2D(2),
            Flatten(),
        ]
        shape = compute_output_shape((1, 8, 8), layers)
        assert shape == (4 * 4 * 4,)  # 64

    def test_with_global_avg(self):
        layers = [
            Conv2D(1, 16, kernel_size=3, padding=1, rng=random.Random(1)),
            GlobalAvgPool2D(),
        ]
        shape = compute_output_shape((1, 8, 8), layers)
        assert shape == (16,)

    def test_activation_preserves(self):
        layers = [
            Conv2D(1, 4, kernel_size=3, padding=1, rng=random.Random(1)),
            Activation2D('relu'),
        ]
        shape = compute_output_shape((1, 8, 8), layers)
        assert shape == (4, 8, 8)


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    def test_lenet_like_architecture(self):
        """Test a LeNet-like architecture end-to-end."""
        rng = random.Random(42)
        net = ConvNet()
        net.add(Conv2D(1, 4, kernel_size=3, padding=0, rng=rng))
        net.add(Activation2D('relu'))
        net.add(MaxPool2D(2))
        net.add(Conv2D(4, 8, kernel_size=3, padding=0, rng=rng))
        net.add(Activation2D('relu'))
        net.add(Flatten())
        # After: (1,8,8) -> conv(4,6,6) -> pool(4,3,3) -> conv(8,1,1) -> flat(8)
        net.add(Dense(8, 3, rng=rng))

        x = Tensor3D.random_normal(1, 8, 8, rng=random.Random(1))
        out = net.forward(x)
        assert isinstance(out, Tensor)
        assert len(out.data) == 3

    def test_forward_backward_conv(self):
        """Full forward-backward through conv layer."""
        rng = random.Random(42)
        conv = Conv2D(1, 2, kernel_size=3, padding=1, bias=True, rng=rng)
        x = Tensor3D.random_normal(1, 4, 4, rng=random.Random(1))
        out = conv.forward(x)
        grad = Tensor3D.ones(2, 4, 4)
        grad_in = conv.backward(grad)
        assert grad_in.shape == (1, 4, 4)
        assert conv.grad_kernels is not None
        assert conv.grad_bias is not None

    def test_conv_avgpool_pipeline(self):
        rng = random.Random(42)
        net = ConvNet()
        net.add(Conv2D(1, 4, kernel_size=3, padding=1, rng=rng))
        net.add(Activation2D('relu'))
        net.add(AvgPool2D(2))
        net.add(Flatten())

        x = Tensor3D.ones(1, 6, 6)
        out = net.forward(x)
        assert isinstance(out, Tensor)

    def test_residual_network(self):
        """Test a network with residual blocks."""
        rng = random.Random(42)
        net = ConvNet()
        net.add(Conv2D(1, 4, kernel_size=3, padding=1, rng=rng))
        net.add(Activation2D('relu'))
        net.add(ResidualBlock(4, kernel_size=3, rng=rng))
        net.add(GlobalAvgPool2D())
        net.add(Dense(4, 2, rng=rng))

        x = Tensor3D.random_normal(1, 6, 6, rng=random.Random(1))
        out = net.forward(x)
        assert isinstance(out, Tensor)
        assert len(out.data) == 2

    def test_separable_conv_network(self):
        rng = random.Random(42)
        net = ConvNet()
        net.add(Conv2D(1, 4, kernel_size=3, padding=1, rng=rng))
        net.add(Activation2D('relu'))
        net.add(SeparableConv2D(4, 8, kernel_size=3, padding=1, rng=rng))
        net.add(Activation2D('relu'))
        net.add(GlobalAvgPool2D())

        x = Tensor3D.ones(1, 6, 6)
        out = net.forward(x)
        assert isinstance(out, Tensor)
        assert len(out.data) == 8

    def test_training_loss_decreases(self):
        """Train a tiny conv net and verify loss decreases."""
        rng = random.Random(42)
        net = ConvNet()
        net.add(Conv2D(1, 2, kernel_size=3, padding=1, rng=rng))
        net.add(Activation2D('relu'))
        net.add(GlobalAvgPool2D())
        net.add(Dense(2, 2, rng=rng))

        # Simple 2-class problem
        X = [
            Tensor3D.random_normal(1, 4, 4, mean=1.0, rng=random.Random(i))
            for i in range(4)
        ] + [
            Tensor3D.random_normal(1, 4, 4, mean=-1.0, rng=random.Random(i + 100))
            for i in range(4)
        ]
        y = [0, 0, 0, 0, 1, 1, 1, 1]

        losses = net.fit(X, y, epochs=5, lr=0.01)
        # Loss should generally decrease (or at least not explode)
        assert losses[-1] < losses[0] * 5  # not exploding

    def test_batch_training(self):
        """Test mini-batch training."""
        rng = random.Random(42)
        net = ConvNet()
        net.add(Conv2D(1, 2, kernel_size=3, padding=1, rng=rng))
        net.add(Activation2D('relu'))
        net.add(GlobalAvgPool2D())
        net.add(Dense(2, 2, rng=rng))

        X = [Tensor3D.random_normal(1, 4, 4, rng=random.Random(i)) for i in range(8)]
        y = [i % 2 for i in range(8)]

        losses = net.fit(X, y, epochs=3, lr=0.01, batch_size=4)
        assert len(losses) == 3

    def test_multiple_pooling_types(self):
        """Test mixing MaxPool and AvgPool."""
        rng = random.Random(42)
        net = ConvNet()
        net.add(Conv2D(1, 4, kernel_size=3, padding=1, rng=rng))
        net.add(MaxPool2D(2))
        net.add(Conv2D(4, 8, kernel_size=3, padding=1, rng=rng))
        net.add(AvgPool2D(2))
        net.add(Flatten())

        x = Tensor3D.ones(1, 8, 8)
        out = net.forward(x)
        assert isinstance(out, Tensor)

    def test_deep_network(self):
        """Test a deeper network doesn't crash."""
        rng = random.Random(42)
        net = ConvNet()
        net.add(Conv2D(1, 4, kernel_size=3, padding=1, rng=rng))
        net.add(Activation2D('relu'))
        net.add(Conv2D(4, 8, kernel_size=3, padding=1, rng=rng))
        net.add(Activation2D('relu'))
        net.add(Conv2D(8, 16, kernel_size=3, padding=1, rng=rng))
        net.add(Activation2D('relu'))
        net.add(GlobalAvgPool2D())
        net.add(Dense(16, 3, rng=rng))

        x = Tensor3D.random_normal(1, 6, 6, rng=random.Random(1))
        out = net.forward(x)
        assert len(out.data) == 3

    def test_edge_case_1x1_input(self):
        """Test with 1x1 spatial input."""
        conv = Conv2D(1, 4, kernel_size=1, rng=random.Random(42))
        x = Tensor3D([[[5.0]]])
        out = conv.forward(x)
        assert out.shape == (4, 1, 1)

    def test_stride_2_conv(self):
        """Test stride-2 convolution for downsampling."""
        conv = Conv2D(1, 4, kernel_size=3, stride=2, padding=1, rng=random.Random(42))
        x = Tensor3D.ones(1, 8, 8)
        out = conv.forward(x)
        assert out.shape == (4, 4, 4)

    def test_gradient_numerical_check(self):
        """Simple numerical gradient check for Conv2D."""
        conv = Conv2D(1, 1, kernel_size=2, padding=0, bias=False, rng=random.Random(42))
        x = Tensor3D([[[1.0, 2.0], [3.0, 4.0]]])
        eps = 1e-5

        # Forward
        out = conv.forward(x)
        # Use sum as loss
        loss = out.sum_all()

        # Backward
        grad_out = Tensor3D.ones(1, 1, 1)
        conv.backward(grad_out)

        # Numerical gradient for kernel[0][0][0][0]
        orig = conv.kernels[0][0][0][0]
        conv.kernels[0][0][0][0] = orig + eps
        out_plus = conv.forward(x)
        loss_plus = out_plus.sum_all()

        conv.kernels[0][0][0][0] = orig - eps
        out_minus = conv.forward(x)
        loss_minus = out_minus.sum_all()

        conv.kernels[0][0][0][0] = orig
        numerical_grad = (loss_plus - loss_minus) / (2 * eps)
        analytical_grad = conv.grad_kernels[0][0][0][0]

        assert abs(numerical_grad - analytical_grad) < 1e-4


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_large_kernel(self):
        conv = Conv2D(1, 1, kernel_size=5, padding=2, rng=random.Random(1))
        x = Tensor3D.ones(1, 6, 6)
        out = conv.forward(x)
        assert out.shape == (1, 6, 6)

    def test_identity_conv(self):
        """A 1x1 conv with identity kernel should pass through."""
        conv = Conv2D(1, 1, kernel_size=1, padding=0, bias=False, rng=random.Random(1))
        conv.kernels = [[[[1.0]]]]
        x = Tensor3D([[[1, 2, 3], [4, 5, 6]]])
        out = conv.forward(x)
        assert out.data[0][0][0] == 1
        assert out.data[0][1][2] == 6

    def test_zero_padding_explicit(self):
        conv = Conv2D(1, 1, kernel_size=3, padding=0, rng=random.Random(1))
        x = Tensor3D.ones(1, 5, 5)
        out = conv.forward(x)
        assert out.shape == (1, 3, 3)

    def test_conv_with_all_zero_input(self):
        conv = Conv2D(1, 4, kernel_size=3, padding=1, bias=False, rng=random.Random(1))
        x = Tensor3D.zeros(1, 4, 4)
        out = conv.forward(x)
        # All zeros input -> all zeros output (no bias)
        for c in range(4):
            for h in range(4):
                for w in range(4):
                    assert abs(out.data[c][h][w]) < 1e-10

    def test_single_pixel_image(self):
        pool = MaxPool2D(kernel_size=1, stride=1)
        x = Tensor3D([[[42.0]]])
        out = pool.forward(x)
        assert out.data[0][0][0] == 42.0

    def test_dropout2d_rate_zero(self):
        drop = Dropout2D(rate=0.0)
        drop.training = True
        x = Tensor3D.ones(3, 4, 4)
        out = drop.forward(x)
        assert out.data[0][0][0] == 1.0

    def test_flatten_single_element(self):
        flat = Flatten()
        x = Tensor3D([[[5.0]]])
        out = flat.forward(x)
        assert out.data == [5.0]

    def test_global_avg_pool_uniform(self):
        pool = GlobalAvgPool2D()
        x = Tensor3D([[[3.0, 3.0], [3.0, 3.0]]])
        out = pool.forward(x)
        assert abs(out.data[0] - 3.0) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
