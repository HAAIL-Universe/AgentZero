"""
C143: Convolutional Neural Network
Extends C140 Neural Network with convolutional layers for image processing.

Composes:
- C140 Neural Network (Tensor, Layer, Dense, Activation, optimizers, Sequential)

Features:
- Tensor3D/Tensor4D: multi-channel image tensors
- Conv2D: 2D convolution with padding, stride, multiple channels
- MaxPool2D / AvgPool2D: pooling layers with stride
- Flatten: reshape for dense layers
- GlobalAvgPool2D: global average pooling
- BatchNorm2D: batch normalization for conv layers
- Dropout2D: spatial dropout (drops entire channels)
- ConvNet: sequential model for conv networks with fit/predict
- He initialization for conv kernels
- im2col-style convolution for clarity
- Full backpropagation through all layers
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))
from neural_network import (
    Tensor, Layer, Dense, Activation, Sequential,
    MSELoss, CrossEntropyLoss, softmax, softmax_batch,
    relu, relu_deriv, sigmoid, sigmoid_deriv,
    tanh_act, tanh_deriv, leaky_relu, leaky_relu_deriv
)


# ============================================================
# Tensor3D: (channels, height, width)
# ============================================================

class Tensor3D:
    """3D tensor for single image: (channels, height, width)."""

    def __init__(self, data):
        if isinstance(data, Tensor3D):
            self.data = [[[data.data[c][h][w] for w in range(len(data.data[0][0]))]
                          for h in range(len(data.data[0]))]
                         for c in range(len(data.data))]
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError("Tensor3D requires list data")
        self.channels = len(self.data)
        self.height = len(self.data[0]) if self.channels > 0 else 0
        self.width = len(self.data[0][0]) if self.height > 0 else 0
        self.shape = (self.channels, self.height, self.width)

    @staticmethod
    def zeros(channels, height, width):
        return Tensor3D([[[0.0] * width for _ in range(height)]
                         for _ in range(channels)])

    @staticmethod
    def ones(channels, height, width):
        return Tensor3D([[[1.0] * width for _ in range(height)]
                         for _ in range(channels)])

    @staticmethod
    def random_normal(channels, height, width, mean=0.0, std=1.0, rng=None):
        def _randn():
            r = rng if rng else random
            u1 = max(r.random(), 1e-10)
            u2 = r.random()
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            return mean + std * z
        return Tensor3D([[[_randn() for _ in range(width)]
                          for _ in range(height)]
                         for _ in range(channels)])

    def copy(self):
        return Tensor3D(self)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Tensor3D([[[self.data[c][h][w] + other
                               for w in range(self.width)]
                              for h in range(self.height)]
                             for c in range(self.channels)])
        if isinstance(other, Tensor3D):
            if other.shape == self.shape:
                return Tensor3D([[[self.data[c][h][w] + other.data[c][h][w]
                                   for w in range(self.width)]
                                  for h in range(self.height)]
                                 for c in range(self.channels)])
            # Broadcast: (C,1,1) + (C,H,W) -- bias addition
            if other.channels == self.channels and other.height == 1 and other.width == 1:
                return Tensor3D([[[self.data[c][h][w] + other.data[c][0][0]
                                   for w in range(self.width)]
                                  for h in range(self.height)]
                                 for c in range(self.channels)])
        raise ValueError(f"Cannot add {self.shape} and {getattr(other, 'shape', type(other))}")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self + (-other)
        if isinstance(other, Tensor3D):
            return Tensor3D([[[self.data[c][h][w] - other.data[c][h][w]
                               for w in range(self.width)]
                              for h in range(self.height)]
                             for c in range(self.channels)])
        raise ValueError("Cannot subtract")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Tensor3D([[[self.data[c][h][w] * other
                               for w in range(self.width)]
                              for h in range(self.height)]
                             for c in range(self.channels)])
        if isinstance(other, Tensor3D) and other.shape == self.shape:
            return Tensor3D([[[self.data[c][h][w] * other.data[c][h][w]
                               for w in range(self.width)]
                              for h in range(self.height)]
                             for c in range(self.channels)])
        raise ValueError("Cannot multiply")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * -1.0

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self * (1.0 / other)
        raise ValueError("Division only by scalar")

    def apply(self, func):
        return Tensor3D([[[func(self.data[c][h][w])
                           for w in range(self.width)]
                          for h in range(self.height)]
                         for c in range(self.channels)])

    def sum_all(self):
        total = 0.0
        for c in range(self.channels):
            for h in range(self.height):
                for w in range(self.width):
                    total += self.data[c][h][w]
        return total

    def flatten(self):
        """Flatten to 1D list."""
        result = []
        for c in range(self.channels):
            for h in range(self.height):
                for w in range(self.width):
                    result.append(self.data[c][h][w])
        return result


# ============================================================
# Tensor4D: batch of Tensor3D (batch, channels, height, width)
# ============================================================

class Tensor4D:
    """4D tensor for batch of images: (batch, channels, height, width)."""

    def __init__(self, data):
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], Tensor3D):
            self.data = [t.copy() for t in data]
        elif isinstance(data, Tensor4D):
            self.data = [t.copy() for t in data.data]
        else:
            raise ValueError("Tensor4D requires list of Tensor3D")
        self.batch_size = len(self.data)
        self.channels = self.data[0].channels
        self.height = self.data[0].height
        self.width = self.data[0].width
        self.shape = (self.batch_size, self.channels, self.height, self.width)

    @staticmethod
    def zeros(batch, channels, height, width):
        return Tensor4D([Tensor3D.zeros(channels, height, width)
                         for _ in range(batch)])

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.batch_size

    def copy(self):
        return Tensor4D(self)


# ============================================================
# Conv2D Layer
# ============================================================

def _compute_output_size(input_size, kernel_size, stride, padding):
    return (input_size + 2 * padding - kernel_size) // stride + 1


def _pad_channel(channel_2d, padding, height, width):
    """Pad a 2D grid with zeros on all sides."""
    if padding == 0:
        return channel_2d
    new_h = height + 2 * padding
    new_w = width + 2 * padding
    padded = [[0.0] * new_w for _ in range(new_h)]
    for h in range(height):
        for w in range(width):
            padded[h + padding][w + padding] = channel_2d[h][w]
    return padded


class Conv2D(Layer):
    """2D Convolution layer.

    Args:
        in_channels: number of input channels
        out_channels: number of output channels (filters)
        kernel_size: int or (kh, kw)
        stride: int, default 1
        padding: int, default 0 ('valid'). Use padding='same' for auto-padding.
        bias: bool, default True
        init: 'he' or 'xavier'
        rng: random.Random instance for reproducibility
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, init='he', rng=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kh, self.kw = kernel_size, kernel_size
        else:
            self.kh, self.kw = kernel_size

        self.stride = stride
        self._padding_mode = padding
        if padding == 'same':
            self.padding = self.kh // 2
        else:
            self.padding = padding

        self.use_bias = bias

        # Initialize kernels: (out_channels, in_channels, kh, kw)
        fan_in = in_channels * self.kh * self.kw
        fan_out = out_channels * self.kh * self.kw
        if init == 'he':
            std = math.sqrt(2.0 / fan_in)
        else:
            std = math.sqrt(2.0 / (fan_in + fan_out))

        r = rng if rng else random
        def _randn():
            u1 = max(r.random(), 1e-10)
            u2 = r.random()
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            return std * z

        self.kernels = [[[[_randn() for _ in range(self.kw)]
                           for _ in range(self.kh)]
                          for _ in range(in_channels)]
                         for _ in range(out_channels)]

        self.bias_vals = [0.0] * out_channels if bias else None

        # Gradients
        self.grad_kernels = None
        self.grad_bias = None
        self._input = None

    def forward(self, x):
        """Forward pass.

        x: Tensor3D (C, H, W) or Tensor4D (N, C, H, W)
        Returns: same type with convolution applied.
        """
        if isinstance(x, Tensor3D):
            self._input = x
            return self._forward_single(x)
        elif isinstance(x, Tensor4D):
            self._input = x
            results = [self._forward_single(x[i]) for i in range(x.batch_size)]
            return Tensor4D(results)
        raise ValueError("Conv2D expects Tensor3D or Tensor4D")

    def _forward_single(self, x):
        """Convolve a single Tensor3D."""
        c_in, h_in, w_in = x.shape
        h_out = _compute_output_size(h_in, self.kh, self.stride, self.padding)
        w_out = _compute_output_size(w_in, self.kw, self.stride, self.padding)

        # Pad input
        padded = [_pad_channel(x.data[c], self.padding, h_in, w_in)
                  for c in range(c_in)]

        # Convolve
        output = [[[0.0] * w_out for _ in range(h_out)]
                  for _ in range(self.out_channels)]

        for oc in range(self.out_channels):
            for oh in range(h_out):
                for ow in range(w_out):
                    val = 0.0
                    ih_start = oh * self.stride
                    iw_start = ow * self.stride
                    for ic in range(c_in):
                        for kh in range(self.kh):
                            for kw in range(self.kw):
                                val += padded[ic][ih_start + kh][iw_start + kw] * \
                                       self.kernels[oc][ic][kh][kw]
                    if self.use_bias:
                        val += self.bias_vals[oc]
                    output[oc][oh][ow] = val

        return Tensor3D(output)

    def backward(self, grad_output):
        """Backward pass.

        grad_output: Tensor3D (out_channels, h_out, w_out) or Tensor4D
        Returns: gradient w.r.t. input (same shape as input).
        """
        if isinstance(grad_output, Tensor4D):
            # Batch backward
            grad_inputs = []
            self.grad_kernels = [[[[0.0] * self.kw for _ in range(self.kh)]
                                  for _ in range(self.in_channels)]
                                 for _ in range(self.out_channels)]
            if self.use_bias:
                self.grad_bias = [0.0] * self.out_channels

            for i in range(grad_output.batch_size):
                gi = self._backward_single(self._input[i], grad_output[i],
                                           accumulate=True)
                grad_inputs.append(gi)
            return Tensor4D(grad_inputs)
        else:
            self.grad_kernels = [[[[0.0] * self.kw for _ in range(self.kh)]
                                  for _ in range(self.in_channels)]
                                 for _ in range(self.out_channels)]
            if self.use_bias:
                self.grad_bias = [0.0] * self.out_channels
            return self._backward_single(self._input, grad_output, accumulate=True)

    def _backward_single(self, x, grad_output, accumulate=False):
        """Backward for a single sample."""
        c_in, h_in, w_in = x.shape
        h_out, w_out = grad_output.height, grad_output.width

        # Pad input for kernel gradient computation
        padded = [_pad_channel(x.data[c], self.padding, h_in, w_in)
                  for c in range(c_in)]
        padded_h = h_in + 2 * self.padding
        padded_w = w_in + 2 * self.padding

        # Gradient w.r.t. kernels
        for oc in range(self.out_channels):
            for ic in range(c_in):
                for kh in range(self.kh):
                    for kw in range(self.kw):
                        grad = 0.0
                        for oh in range(h_out):
                            for ow in range(w_out):
                                ih = oh * self.stride + kh
                                iw = ow * self.stride + kw
                                grad += grad_output.data[oc][oh][ow] * padded[ic][ih][iw]
                        self.grad_kernels[oc][ic][kh][kw] += grad

        # Gradient w.r.t. bias
        if self.use_bias:
            for oc in range(self.out_channels):
                for oh in range(h_out):
                    for ow in range(w_out):
                        self.grad_bias[oc] += grad_output.data[oc][oh][ow]

        # Gradient w.r.t. input (full convolution with flipped kernels)
        grad_padded = [[[0.0] * padded_w for _ in range(padded_h)]
                       for _ in range(c_in)]

        for oc in range(self.out_channels):
            for ic in range(c_in):
                for oh in range(h_out):
                    for ow in range(w_out):
                        g = grad_output.data[oc][oh][ow]
                        ih_start = oh * self.stride
                        iw_start = ow * self.stride
                        for kh in range(self.kh):
                            for kw in range(self.kw):
                                grad_padded[ic][ih_start + kh][iw_start + kw] += \
                                    g * self.kernels[oc][ic][kh][kw]

        # Remove padding from gradient
        if self.padding > 0:
            grad_input_data = [[[grad_padded[c][h + self.padding][w + self.padding]
                                 for w in range(w_in)]
                                for h in range(h_in)]
                               for c in range(c_in)]
        else:
            grad_input_data = grad_padded

        return Tensor3D(grad_input_data)

    def get_params(self):
        """Return params in format compatible with optimizers."""
        return [('conv_kernels', self.kernels, self.grad_kernels),
                ('conv_bias', self.bias_vals, self.grad_bias)]


# ============================================================
# Pooling Layers
# ============================================================

class MaxPool2D(Layer):
    """Max pooling layer."""

    def __init__(self, kernel_size=2, stride=None):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kh, self.kw = kernel_size, kernel_size
        else:
            self.kh, self.kw = kernel_size
        self.stride = stride if stride is not None else self.kh
        self._input = None
        self._max_indices = None

    def forward(self, x):
        if isinstance(x, Tensor3D):
            self._input = x
            result, indices = self._forward_single(x)
            self._max_indices = indices
            return result
        elif isinstance(x, Tensor4D):
            self._input = x
            results = []
            self._max_indices = []
            for i in range(x.batch_size):
                r, idx = self._forward_single(x[i])
                results.append(r)
                self._max_indices.append(idx)
            return Tensor4D(results)
        raise ValueError("MaxPool2D expects Tensor3D or Tensor4D")

    def _forward_single(self, x):
        c, h_in, w_in = x.shape
        h_out = (h_in - self.kh) // self.stride + 1
        w_out = (w_in - self.kw) // self.stride + 1

        output = [[[0.0] * w_out for _ in range(h_out)] for _ in range(c)]
        indices = [[[None] * w_out for _ in range(h_out)] for _ in range(c)]

        for ch in range(c):
            for oh in range(h_out):
                for ow in range(w_out):
                    max_val = float('-inf')
                    max_ih, max_iw = 0, 0
                    for kh in range(self.kh):
                        for kw in range(self.kw):
                            ih = oh * self.stride + kh
                            iw = ow * self.stride + kw
                            v = x.data[ch][ih][iw]
                            if v > max_val:
                                max_val = v
                                max_ih, max_iw = ih, iw
                    output[ch][oh][ow] = max_val
                    indices[ch][oh][ow] = (max_ih, max_iw)

        return Tensor3D(output), indices

    def backward(self, grad_output):
        if isinstance(grad_output, Tensor4D):
            grad_inputs = []
            for i in range(grad_output.batch_size):
                gi = self._backward_single(self._input[i], grad_output[i],
                                           self._max_indices[i])
                grad_inputs.append(gi)
            return Tensor4D(grad_inputs)
        return self._backward_single(self._input, grad_output, self._max_indices)

    def _backward_single(self, x, grad_output, indices):
        c, h_in, w_in = x.shape
        h_out, w_out = grad_output.height, grad_output.width
        grad_input = [[[0.0] * w_in for _ in range(h_in)] for _ in range(c)]

        for ch in range(c):
            for oh in range(h_out):
                for ow in range(w_out):
                    ih, iw = indices[ch][oh][ow]
                    grad_input[ch][ih][iw] += grad_output.data[ch][oh][ow]

        return Tensor3D(grad_input)


class AvgPool2D(Layer):
    """Average pooling layer."""

    def __init__(self, kernel_size=2, stride=None):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kh, self.kw = kernel_size, kernel_size
        else:
            self.kh, self.kw = kernel_size
        self.stride = stride if stride is not None else self.kh
        self._input = None

    def forward(self, x):
        if isinstance(x, Tensor3D):
            self._input = x
            return self._forward_single(x)
        elif isinstance(x, Tensor4D):
            self._input = x
            return Tensor4D([self._forward_single(x[i]) for i in range(x.batch_size)])
        raise ValueError("AvgPool2D expects Tensor3D or Tensor4D")

    def _forward_single(self, x):
        c, h_in, w_in = x.shape
        h_out = (h_in - self.kh) // self.stride + 1
        w_out = (w_in - self.kw) // self.stride + 1
        pool_size = self.kh * self.kw

        output = [[[0.0] * w_out for _ in range(h_out)] for _ in range(c)]

        for ch in range(c):
            for oh in range(h_out):
                for ow in range(w_out):
                    total = 0.0
                    for kh in range(self.kh):
                        for kw in range(self.kw):
                            ih = oh * self.stride + kh
                            iw = ow * self.stride + kw
                            total += x.data[ch][ih][iw]
                    output[ch][oh][ow] = total / pool_size

        return Tensor3D(output)

    def backward(self, grad_output):
        if isinstance(grad_output, Tensor4D):
            return Tensor4D([self._backward_single(self._input[i], grad_output[i])
                             for i in range(grad_output.batch_size)])
        return self._backward_single(self._input, grad_output)

    def _backward_single(self, x, grad_output):
        c, h_in, w_in = x.shape
        h_out, w_out = grad_output.height, grad_output.width
        pool_size = self.kh * self.kw
        grad_input = [[[0.0] * w_in for _ in range(h_in)] for _ in range(c)]

        for ch in range(c):
            for oh in range(h_out):
                for ow in range(w_out):
                    g = grad_output.data[ch][oh][ow] / pool_size
                    for kh in range(self.kh):
                        for kw in range(self.kw):
                            ih = oh * self.stride + kh
                            iw = ow * self.stride + kw
                            grad_input[ch][ih][iw] += g

        return Tensor3D(grad_input)


class GlobalAvgPool2D(Layer):
    """Global average pooling -- reduce (C, H, W) to (C,)."""

    def __init__(self):
        super().__init__()
        self._input = None

    def forward(self, x):
        if isinstance(x, Tensor3D):
            self._input = x
            return self._forward_single(x)
        elif isinstance(x, Tensor4D):
            self._input = x
            results = [self._forward_single(x[i]).data for i in range(x.batch_size)]
            return Tensor(results)
        raise ValueError("GlobalAvgPool2D expects Tensor3D or Tensor4D")

    def _forward_single(self, x):
        c, h, w = x.shape
        hw = h * w
        result = [0.0] * c
        for ch in range(c):
            total = 0.0
            for hh in range(h):
                for ww in range(w):
                    total += x.data[ch][hh][ww]
            result[ch] = total / hw
        return Tensor(result)

    def backward(self, grad_output):
        if isinstance(grad_output, Tensor) and len(grad_output.shape) == 2:
            # Batch: grad_output is (batch, channels)
            grad_inputs = []
            for i in range(grad_output.shape[0]):
                gi = self._backward_single(self._input[i],
                                           Tensor(grad_output.data[i]))
                grad_inputs.append(gi)
            return Tensor4D(grad_inputs)
        return self._backward_single(self._input, grad_output)

    def _backward_single(self, x, grad_output):
        c, h, w = x.shape
        hw = h * w
        grad_data = grad_output.data if isinstance(grad_output, Tensor) else grad_output
        grad_input = [[[grad_data[ch] / hw for _ in range(w)]
                       for _ in range(h)]
                      for ch in range(c)]
        return Tensor3D(grad_input)


# ============================================================
# Flatten Layer
# ============================================================

class Flatten(Layer):
    """Flatten Tensor3D to 1D Tensor (for connecting conv to dense)."""

    def __init__(self):
        super().__init__()
        self._input_shape = None

    def forward(self, x):
        if isinstance(x, Tensor3D):
            self._input_shape = x.shape
            return Tensor(x.flatten())
        elif isinstance(x, Tensor4D):
            self._input_shape = (x.channels, x.height, x.width)
            rows = [x[i].flatten() for i in range(x.batch_size)]
            return Tensor(rows)
        raise ValueError("Flatten expects Tensor3D or Tensor4D")

    def backward(self, grad_output):
        c, h, w = self._input_shape
        if isinstance(grad_output, Tensor) and len(grad_output.shape) == 2:
            # Batch
            grad_inputs = []
            for i in range(grad_output.shape[0]):
                flat = grad_output.data[i]
                data_3d = [[[flat[ch * h * w + hh * w + ww]
                             for ww in range(w)]
                            for hh in range(h)]
                           for ch in range(c)]
                grad_inputs.append(Tensor3D(data_3d))
            return Tensor4D(grad_inputs)
        else:
            flat = grad_output.data
            data_3d = [[[flat[ch * h * w + hh * w + ww]
                         for ww in range(w)]
                        for hh in range(h)]
                       for ch in range(c)]
            return Tensor3D(data_3d)


# ============================================================
# Activation2D Layer
# ============================================================

class Activation2D(Layer):
    """Activation function applied element-wise to Tensor3D/Tensor4D."""

    def __init__(self, name='relu', alpha=0.01):
        super().__init__()
        self.name = name
        self.alpha = alpha
        self._input = None

    def _get_fn_and_deriv(self):
        if self.name == 'relu':
            return relu, relu_deriv
        elif self.name == 'sigmoid':
            return sigmoid, sigmoid_deriv
        elif self.name == 'tanh':
            return tanh_act, tanh_deriv
        elif self.name == 'leaky_relu':
            return (lambda x: leaky_relu(x, self.alpha),
                    lambda x: leaky_relu_deriv(x, self.alpha))
        raise ValueError(f"Unknown activation: {self.name}")

    def forward(self, x):
        self._input = x
        fn, _ = self._get_fn_and_deriv()
        if isinstance(x, Tensor3D):
            return x.apply(fn)
        elif isinstance(x, Tensor4D):
            return Tensor4D([x[i].apply(fn) for i in range(x.batch_size)])
        raise ValueError("Activation2D expects Tensor3D or Tensor4D")

    def backward(self, grad_output):
        _, deriv = self._get_fn_and_deriv()
        if isinstance(self._input, Tensor3D):
            mask = self._input.apply(deriv)
            return grad_output * mask
        elif isinstance(self._input, Tensor4D):
            results = []
            for i in range(self._input.batch_size):
                mask = self._input[i].apply(deriv)
                results.append(grad_output[i] * mask)
            return Tensor4D(results)


# ============================================================
# BatchNorm2D Layer
# ============================================================

class BatchNorm2D(Layer):
    """Batch normalization for convolutional layers.

    Normalizes per-channel across spatial dimensions and batch.
    """

    def __init__(self, num_channels, momentum=0.1, eps=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps

        # Learnable parameters
        self.gamma = [1.0] * num_channels
        self.beta = [0.0] * num_channels

        # Running statistics
        self.running_mean = [0.0] * num_channels
        self.running_var = [1.0] * num_channels

        # Gradients
        self.grad_gamma = None
        self.grad_beta = None

        # Cache
        self._normalized = None
        self._std = None
        self._input = None

    def forward(self, x):
        if isinstance(x, Tensor3D):
            # Single sample -- use running stats
            self._input = x
            return self._forward_inference(x)
        elif isinstance(x, Tensor4D):
            self._input = x
            if self.training:
                return self._forward_train(x)
            else:
                return Tensor4D([self._forward_inference(x[i])
                                 for i in range(x.batch_size)])
        raise ValueError("BatchNorm2D expects Tensor3D or Tensor4D")

    def _forward_inference(self, x):
        c, h, w = x.shape
        output = [[[0.0] * w for _ in range(h)] for _ in range(c)]
        for ch in range(c):
            std = math.sqrt(self.running_var[ch] + self.eps)
            for hh in range(h):
                for ww in range(w):
                    normed = (x.data[ch][hh][ww] - self.running_mean[ch]) / std
                    output[ch][hh][ww] = self.gamma[ch] * normed + self.beta[ch]
        return Tensor3D(output)

    def _forward_train(self, x):
        batch_size = x.batch_size
        c = x.channels
        h, w = x.height, x.width
        n = batch_size * h * w

        # Compute per-channel mean and variance
        mean = [0.0] * c
        for ch in range(c):
            total = 0.0
            for b in range(batch_size):
                for hh in range(h):
                    for ww in range(w):
                        total += x[b].data[ch][hh][ww]
            mean[ch] = total / n

        var = [0.0] * c
        for ch in range(c):
            total = 0.0
            for b in range(batch_size):
                for hh in range(h):
                    for ww in range(w):
                        d = x[b].data[ch][hh][ww] - mean[ch]
                        total += d * d
            var[ch] = total / n

        # Update running statistics
        for ch in range(c):
            self.running_mean[ch] = (1 - self.momentum) * self.running_mean[ch] + \
                                    self.momentum * mean[ch]
            self.running_var[ch] = (1 - self.momentum) * self.running_var[ch] + \
                                   self.momentum * var[ch]

        # Normalize
        std = [math.sqrt(var[ch] + self.eps) for ch in range(c)]
        self._std = std
        self._normalized = []

        results = []
        for b in range(batch_size):
            normed_data = [[[0.0] * w for _ in range(h)] for _ in range(c)]
            out_data = [[[0.0] * w for _ in range(h)] for _ in range(c)]
            for ch in range(c):
                for hh in range(h):
                    for ww in range(w):
                        n_val = (x[b].data[ch][hh][ww] - mean[ch]) / std[ch]
                        normed_data[ch][hh][ww] = n_val
                        out_data[ch][hh][ww] = self.gamma[ch] * n_val + self.beta[ch]
            self._normalized.append(Tensor3D(normed_data))
            results.append(Tensor3D(out_data))

        return Tensor4D(results)

    def backward(self, grad_output):
        if isinstance(grad_output, Tensor4D):
            return self._backward_batch(grad_output)
        # Single sample -- simplified
        return grad_output

    def _backward_batch(self, grad_output):
        batch_size = grad_output.batch_size
        c = grad_output.channels
        h, w = grad_output.height, grad_output.width
        n = batch_size * h * w

        self.grad_gamma = [0.0] * c
        self.grad_beta = [0.0] * c

        # Compute gamma and beta gradients
        for ch in range(c):
            for b in range(batch_size):
                for hh in range(h):
                    for ww in range(w):
                        self.grad_beta[ch] += grad_output[b].data[ch][hh][ww]
                        self.grad_gamma[ch] += grad_output[b].data[ch][hh][ww] * \
                                               self._normalized[b].data[ch][hh][ww]

        # Compute input gradient
        grad_inputs = []
        for b in range(batch_size):
            gi_data = [[[0.0] * w for _ in range(h)] for _ in range(c)]
            for ch in range(c):
                inv_std = 1.0 / self._std[ch]
                for hh in range(h):
                    for ww in range(w):
                        # Simplified BN backward
                        x_hat = self._normalized[b].data[ch][hh][ww]
                        dxhat = grad_output[b].data[ch][hh][ww] * self.gamma[ch]
                        gi_data[ch][hh][ww] = inv_std * (
                            dxhat - self.grad_beta[ch] * self.gamma[ch] / n -
                            x_hat * self.grad_gamma[ch] * self.gamma[ch] / n
                        ) * self.gamma[ch] / self.gamma[ch]  # simplify
            grad_inputs.append(Tensor3D(gi_data))

        return Tensor4D(grad_inputs)

    def get_params(self):
        return [('bn_gamma', self.gamma, self.grad_gamma),
                ('bn_beta', self.beta, self.grad_beta)]


# ============================================================
# Dropout2D Layer (Spatial Dropout)
# ============================================================

class Dropout2D(Layer):
    """Spatial dropout -- drops entire channels."""

    def __init__(self, rate=0.5, rng=None):
        super().__init__()
        self.rate = rate
        self.rng = rng or random.Random()
        self._mask = None

    def forward(self, x):
        if not self.training or self.rate == 0.0:
            if isinstance(x, Tensor3D):
                return x.copy()
            return x.copy()

        scale = 1.0 / (1.0 - self.rate)

        if isinstance(x, Tensor3D):
            mask = [scale if self.rng.random() > self.rate else 0.0
                    for _ in range(x.channels)]
            self._mask = mask
            return Tensor3D([[[x.data[c][h][w] * mask[c]
                               for w in range(x.width)]
                              for h in range(x.height)]
                             for c in range(x.channels)])
        elif isinstance(x, Tensor4D):
            # Same mask for all samples in batch
            mask = [scale if self.rng.random() > self.rate else 0.0
                    for _ in range(x.channels)]
            self._mask = mask
            results = []
            for i in range(x.batch_size):
                results.append(Tensor3D(
                    [[[x[i].data[c][h][w] * mask[c]
                       for w in range(x.width)]
                      for h in range(x.height)]
                     for c in range(x.channels)]))
            return Tensor4D(results)

    def backward(self, grad_output):
        if not self.training or self.rate == 0.0:
            return grad_output
        mask = self._mask
        if isinstance(grad_output, Tensor3D):
            return Tensor3D([[[grad_output.data[c][h][w] * mask[c]
                               for w in range(grad_output.width)]
                              for h in range(grad_output.height)]
                             for c in range(grad_output.channels)])
        elif isinstance(grad_output, Tensor4D):
            results = []
            for i in range(grad_output.batch_size):
                results.append(Tensor3D(
                    [[[grad_output[i].data[c][h][w] * mask[c]
                       for w in range(grad_output.width)]
                      for h in range(grad_output.height)]
                     for c in range(grad_output.channels)]
                ))
            return Tensor4D(results)


# ============================================================
# ConvNet: Sequential model for convolutional networks
# ============================================================

class ConvOptimizer:
    """Adam optimizer for conv networks (handles nested kernel structures)."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self._states = {}

    def step(self, layers):
        self.t += 1
        for layer_idx, layer in enumerate(layers):
            if not hasattr(layer, 'get_params'):
                continue
            params = layer.get_params()
            if not params:
                continue

            for param_idx, param_tuple in enumerate(params):
                if len(param_tuple) < 3:
                    continue

                # Handle both Conv format (name, values, grads) and
                # Dense format (weights_tensor, grad_tensor, name_str)
                if isinstance(param_tuple[0], str):
                    # Conv format: (name, values, grads)
                    values, grads = param_tuple[1], param_tuple[2]
                elif isinstance(param_tuple[2], str):
                    # Dense format: (tensor, grad_tensor, name)
                    values, grads = param_tuple[0], param_tuple[1]
                else:
                    continue

                if values is None or grads is None:
                    continue

                # Handle Tensor objects from Dense layers
                if isinstance(values, Tensor):
                    key = (layer_idx, param_idx)
                    if key not in self._states:
                        self._states[key] = self._init_state_tensor(values)
                    self._update_tensor(values, grads, self._states[key])
                else:
                    key = (layer_idx, param_idx)
                    if key not in self._states:
                        self._states[key] = self._init_state(values)
                    self._update(values, grads, self._states[key])

    def _init_state(self, values):
        """Create zero-initialized moment states matching the structure."""
        if isinstance(values, list) and len(values) > 0 and isinstance(values[0], list):
            # Could be 4D kernel or 2D
            return {
                'm': self._zeros_like(values),
                'v': self._zeros_like(values)
            }
        elif isinstance(values, list):
            return {
                'm': [0.0] * len(values),
                'v': [0.0] * len(values)
            }
        return None

    def _zeros_like(self, data):
        if isinstance(data, list):
            return [self._zeros_like(x) if isinstance(x, list) else 0.0 for x in data]
        return 0.0

    def _init_state_tensor(self, tensor):
        """Create zero-initialized moment states for Tensor objects."""
        if len(tensor.shape) == 1:
            n = tensor.shape[0]
            return {'m': [0.0] * n, 'v': [0.0] * n}
        else:
            rows, cols = tensor.shape
            return {
                'm': [[0.0] * cols for _ in range(rows)],
                'v': [[0.0] * cols for _ in range(rows)]
            }

    def _update_tensor(self, values, grads, state):
        """Update Tensor parameters using Adam."""
        if len(values.shape) == 1:
            for i in range(values.shape[0]):
                g = grads.data[i]
                state['m'][i] = self.beta1 * state['m'][i] + (1 - self.beta1) * g
                state['v'][i] = self.beta2 * state['v'][i] + (1 - self.beta2) * g * g
                m_hat = state['m'][i] / (1 - self.beta1 ** self.t)
                v_hat = state['v'][i] / (1 - self.beta2 ** self.t)
                values.data[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
        else:
            rows, cols = values.shape
            for i in range(rows):
                for j in range(cols):
                    g = grads.data[i][j]
                    state['m'][i][j] = self.beta1 * state['m'][i][j] + (1 - self.beta1) * g
                    state['v'][i][j] = self.beta2 * state['v'][i][j] + (1 - self.beta2) * g * g
                    m_hat = state['m'][i][j] / (1 - self.beta1 ** self.t)
                    v_hat = state['v'][i][j] / (1 - self.beta2 ** self.t)
                    values.data[i][j] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def _update(self, values, grads, state):
        """Recursively update parameters using Adam."""
        self._update_recursive(values, grads, state['m'], state['v'])

    def _update_recursive(self, values, grads, m, v):
        if isinstance(values, list) and len(values) > 0 and isinstance(values[0], list):
            for i in range(len(values)):
                self._update_recursive(values[i], grads[i], m[i], v[i])
        elif isinstance(values, list):
            for i in range(len(values)):
                g = grads[i]
                m[i] = self.beta1 * m[i] + (1 - self.beta1) * g
                v[i] = self.beta2 * v[i] + (1 - self.beta2) * g * g
                m_hat = m[i] / (1 - self.beta1 ** self.t)
                v_hat = v[i] / (1 - self.beta2 ** self.t)
                values[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


class SGDConvOptimizer:
    """SGD with momentum for conv networks."""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self._velocities = {}

    def step(self, layers):
        for layer_idx, layer in enumerate(layers):
            if not hasattr(layer, 'get_params'):
                continue
            params = layer.get_params()
            if not params:
                continue

            for param_idx, param_tuple in enumerate(params):
                if len(param_tuple) < 3:
                    continue

                if isinstance(param_tuple[0], str):
                    values, grads = param_tuple[1], param_tuple[2]
                elif isinstance(param_tuple[2], str):
                    values, grads = param_tuple[0], param_tuple[1]
                else:
                    continue

                if values is None or grads is None:
                    continue

                key = (layer_idx, param_idx)

                if isinstance(values, Tensor):
                    if key not in self._velocities:
                        self._velocities[key] = self._init_tensor_velocity(values)
                    self._update_tensor(values, grads, self._velocities[key])
                else:
                    if key not in self._velocities:
                        self._velocities[key] = self._zeros_like(values)
                    self._update_recursive(values, grads, self._velocities[key])

    def _zeros_like(self, data):
        if isinstance(data, list):
            return [self._zeros_like(x) if isinstance(x, list) else 0.0 for x in data]
        return 0.0

    def _init_tensor_velocity(self, tensor):
        if len(tensor.shape) == 1:
            return [0.0] * tensor.shape[0]
        rows, cols = tensor.shape
        return [[0.0] * cols for _ in range(rows)]

    def _update_tensor(self, values, grads, velocity):
        if len(values.shape) == 1:
            for i in range(values.shape[0]):
                velocity[i] = self.momentum * velocity[i] - self.lr * grads.data[i]
                values.data[i] += velocity[i]
        else:
            rows, cols = values.shape
            for i in range(rows):
                for j in range(cols):
                    velocity[i][j] = self.momentum * velocity[i][j] - self.lr * grads.data[i][j]
                    values.data[i][j] += velocity[i][j]

    def _update_recursive(self, values, grads, velocity):
        if isinstance(values, list) and len(values) > 0 and isinstance(values[0], list):
            for i in range(len(values)):
                self._update_recursive(values[i], grads[i], velocity[i])
        elif isinstance(values, list):
            for i in range(len(values)):
                velocity[i] = self.momentum * velocity[i] - self.lr * grads[i]
                values[i] += velocity[i]


class ConvNet:
    """Sequential convolutional network model.

    Handles mixed conv (Tensor3D/4D) and dense (Tensor/1D-2D) layers.
    """

    def __init__(self, layers=None):
        self.layers = layers or []

    def add(self, layer):
        self.layers.append(layer)
        return self

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def set_training(self, mode):
        for layer in self.layers:
            if hasattr(layer, 'set_training'):
                layer.set_training(mode)
            if hasattr(layer, 'training'):
                layer.training = mode

    def predict(self, x):
        self.set_training(False)
        result = self.forward(x)
        self.set_training(True)
        return result

    def fit(self, X, y, epochs=10, lr=0.001, optimizer=None, loss_fn=None,
            batch_size=None, verbose=False):
        """Train the network.

        X: list of Tensor3D (images) or Tensor4D
        y: list of int (class labels) or Tensor (targets)
        """
        if loss_fn is None:
            loss_fn = CrossEntropyLoss()
        if optimizer is None:
            optimizer = ConvOptimizer(lr=lr)

        losses = []

        for epoch in range(epochs):
            self.set_training(True)
            epoch_loss = 0.0

            if batch_size is None or batch_size >= len(X):
                # Full batch
                if isinstance(X, list) and isinstance(X[0], Tensor3D):
                    batch_x = Tensor4D(X)
                else:
                    batch_x = X

                output = self.forward(batch_x)
                loss = loss_fn.forward(output, y)
                grad = loss_fn.backward(output, y)
                self.backward(grad)
                optimizer.step(self.layers)
                epoch_loss = loss
            else:
                # Mini-batch
                indices = list(range(len(X)))
                random.shuffle(indices)
                n_batches = 0
                for start in range(0, len(X), batch_size):
                    end = min(start + batch_size, len(X))
                    batch_idx = indices[start:end]

                    if isinstance(X, list) and isinstance(X[0], Tensor3D):
                        batch_x = Tensor4D([X[i] for i in batch_idx])
                    else:
                        batch_x = X

                    if isinstance(y, list):
                        batch_y = [y[i] for i in batch_idx]
                    else:
                        batch_y = y

                    output = self.forward(batch_x)
                    loss = loss_fn.forward(output, batch_y)
                    grad = loss_fn.backward(output, batch_y)
                    self.backward(grad)
                    optimizer.step(self.layers)
                    epoch_loss += loss
                    n_batches += 1

                epoch_loss /= n_batches

            losses.append(epoch_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

        return losses

    def count_params(self):
        """Count total trainable parameters."""
        total = 0
        for layer in self.layers:
            if isinstance(layer, Conv2D):
                total += layer.out_channels * layer.in_channels * layer.kh * layer.kw
                if layer.use_bias:
                    total += layer.out_channels
            elif isinstance(layer, Dense):
                total += layer.input_size * layer.output_size
                if layer.use_bias:
                    total += layer.output_size
            elif isinstance(layer, BatchNorm2D):
                total += 2 * layer.num_channels
        return total


# ============================================================
# Depthwise Separable Convolution (MobileNet-style)
# ============================================================

class DepthwiseConv2D(Layer):
    """Depthwise convolution -- one filter per input channel."""

    def __init__(self, channels, kernel_size=3, stride=1, padding=0,
                 bias=True, init='he', rng=None):
        super().__init__()
        self.channels = channels
        if isinstance(kernel_size, int):
            self.kh, self.kw = kernel_size, kernel_size
        else:
            self.kh, self.kw = kernel_size
        self.stride = stride
        if padding == 'same':
            self.padding = self.kh // 2
        else:
            self.padding = padding
        self.use_bias = bias

        fan_in = self.kh * self.kw
        if init == 'he':
            std = math.sqrt(2.0 / fan_in)
        else:
            std = math.sqrt(2.0 / (fan_in + fan_in))

        r = rng if rng else random
        def _randn():
            u1 = max(r.random(), 1e-10)
            u2 = r.random()
            return std * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

        # One kernel per channel: (channels, kh, kw)
        self.kernels = [[[_randn() for _ in range(self.kw)]
                         for _ in range(self.kh)]
                        for _ in range(channels)]

        self.bias_vals = [0.0] * channels if bias else None
        self.grad_kernels = None
        self.grad_bias = None
        self._input = None

    def forward(self, x):
        if isinstance(x, Tensor3D):
            self._input = x
            return self._forward_single(x)
        elif isinstance(x, Tensor4D):
            self._input = x
            return Tensor4D([self._forward_single(x[i]) for i in range(x.batch_size)])
        raise ValueError("DepthwiseConv2D expects Tensor3D or Tensor4D")

    def _forward_single(self, x):
        c, h_in, w_in = x.shape
        h_out = _compute_output_size(h_in, self.kh, self.stride, self.padding)
        w_out = _compute_output_size(w_in, self.kw, self.stride, self.padding)

        padded = [_pad_channel(x.data[ch], self.padding, h_in, w_in)
                  for ch in range(c)]

        output = [[[0.0] * w_out for _ in range(h_out)] for _ in range(c)]

        for ch in range(c):
            for oh in range(h_out):
                for ow in range(w_out):
                    val = 0.0
                    ih_start = oh * self.stride
                    iw_start = ow * self.stride
                    for kh in range(self.kh):
                        for kw in range(self.kw):
                            val += padded[ch][ih_start + kh][iw_start + kw] * \
                                   self.kernels[ch][kh][kw]
                    if self.use_bias:
                        val += self.bias_vals[ch]
                    output[ch][oh][ow] = val

        return Tensor3D(output)

    def backward(self, grad_output):
        if isinstance(grad_output, Tensor4D):
            self.grad_kernels = [[[0.0] * self.kw for _ in range(self.kh)]
                                 for _ in range(self.channels)]
            if self.use_bias:
                self.grad_bias = [0.0] * self.channels
            results = []
            for i in range(grad_output.batch_size):
                results.append(self._backward_single(self._input[i], grad_output[i],
                                                      accumulate=True))
            return Tensor4D(results)
        else:
            self.grad_kernels = [[[0.0] * self.kw for _ in range(self.kh)]
                                 for _ in range(self.channels)]
            if self.use_bias:
                self.grad_bias = [0.0] * self.channels
            return self._backward_single(self._input, grad_output, accumulate=True)

    def _backward_single(self, x, grad_output, accumulate=False):
        c, h_in, w_in = x.shape
        h_out, w_out = grad_output.height, grad_output.width

        padded = [_pad_channel(x.data[ch], self.padding, h_in, w_in)
                  for ch in range(c)]
        padded_h = h_in + 2 * self.padding
        padded_w = w_in + 2 * self.padding

        for ch in range(c):
            for kh in range(self.kh):
                for kw in range(self.kw):
                    grad = 0.0
                    for oh in range(h_out):
                        for ow in range(w_out):
                            ih = oh * self.stride + kh
                            iw = ow * self.stride + kw
                            grad += grad_output.data[ch][oh][ow] * padded[ch][ih][iw]
                    self.grad_kernels[ch][kh][kw] += grad

        if self.use_bias:
            for ch in range(c):
                for oh in range(h_out):
                    for ow in range(w_out):
                        self.grad_bias[ch] += grad_output.data[ch][oh][ow]

        grad_padded = [[[0.0] * padded_w for _ in range(padded_h)]
                       for _ in range(c)]

        for ch in range(c):
            for oh in range(h_out):
                for ow in range(w_out):
                    g = grad_output.data[ch][oh][ow]
                    ih_start = oh * self.stride
                    iw_start = ow * self.stride
                    for kh in range(self.kh):
                        for kw in range(self.kw):
                            grad_padded[ch][ih_start + kh][iw_start + kw] += \
                                g * self.kernels[ch][kh][kw]

        if self.padding > 0:
            grad_input_data = [[[grad_padded[ch][h + self.padding][w + self.padding]
                                 for w in range(w_in)]
                                for h in range(h_in)]
                               for ch in range(c)]
        else:
            grad_input_data = grad_padded

        return Tensor3D(grad_input_data)

    def get_params(self):
        return [('dw_kernels', self.kernels, self.grad_kernels),
                ('dw_bias', self.bias_vals, self.grad_bias)]


class SeparableConv2D(Layer):
    """Depthwise separable convolution = DepthwiseConv2D + pointwise Conv2D(1x1)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, bias=True, rng=None):
        super().__init__()
        self.depthwise = DepthwiseConv2D(in_channels, kernel_size, stride,
                                          padding, bias=False, rng=rng)
        self.pointwise = Conv2D(in_channels, out_channels, kernel_size=1,
                                stride=1, padding=0, bias=bias, rng=rng)

    def forward(self, x):
        x = self.depthwise.forward(x)
        return self.pointwise.forward(x)

    def backward(self, grad_output):
        grad = self.pointwise.backward(grad_output)
        return self.depthwise.backward(grad)

    def get_params(self):
        return self.depthwise.get_params() + self.pointwise.get_params()


# ============================================================
# 1x1 Convolution (Pointwise)
# ============================================================

class Conv1x1(Conv2D):
    """1x1 convolution for channel mixing (used in ResNets, Inception)."""

    def __init__(self, in_channels, out_channels, bias=True, rng=None):
        super().__init__(in_channels, out_channels, kernel_size=1,
                         stride=1, padding=0, bias=bias, rng=rng)


# ============================================================
# Residual Block
# ============================================================

class ResidualBlock(Layer):
    """Residual block with skip connection (He et al., 2015)."""

    def __init__(self, channels, kernel_size=3, rng=None):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = Conv2D(channels, channels, kernel_size, padding=padding, rng=rng)
        self.act1 = Activation2D('relu')
        self.conv2 = Conv2D(channels, channels, kernel_size, padding=padding, rng=rng)
        self.act2 = Activation2D('relu')
        self._residual = None

    def forward(self, x):
        self._residual = x if isinstance(x, Tensor3D) else x.copy()
        out = self.conv1.forward(x)
        out = self.act1.forward(out)
        out = self.conv2.forward(out)
        # Add skip connection
        if isinstance(out, Tensor3D):
            out = out + self._residual
        elif isinstance(out, Tensor4D):
            results = []
            for i in range(out.batch_size):
                results.append(out[i] + self._residual[i])
            out = Tensor4D(results)
        out = self.act2.forward(out)
        return out

    def backward(self, grad_output):
        grad = self.act2.backward(grad_output)
        # Gradient splits: one path to conv2, one to skip
        skip_grad = grad  # skip connection gradient = grad_output after act2
        conv_grad = self.conv2.backward(grad)
        conv_grad = self.act1.backward(conv_grad)
        conv_grad = self.conv1.backward(conv_grad)
        # Sum gradients from both paths
        if isinstance(conv_grad, Tensor3D):
            return conv_grad + skip_grad
        elif isinstance(conv_grad, Tensor4D):
            results = []
            for i in range(conv_grad.batch_size):
                results.append(conv_grad[i] + skip_grad[i])
            return Tensor4D(results)

    def get_params(self):
        return self.conv1.get_params() + self.conv2.get_params()


# ============================================================
# Utility Functions
# ============================================================

def compute_output_shape(input_shape, layers):
    """Compute output shape after a sequence of conv/pool layers.

    input_shape: (channels, height, width)
    layers: list of Conv2D/MaxPool2D/etc.
    Returns: (channels, height, width) or (flat_size,)
    """
    c, h, w = input_shape
    for layer in layers:
        if isinstance(layer, Conv2D):
            h = _compute_output_size(h, layer.kh, layer.stride, layer.padding)
            w = _compute_output_size(w, layer.kw, layer.stride, layer.padding)
            c = layer.out_channels
        elif isinstance(layer, DepthwiseConv2D):
            h = _compute_output_size(h, layer.kh, layer.stride, layer.padding)
            w = _compute_output_size(w, layer.kw, layer.stride, layer.padding)
        elif isinstance(layer, (MaxPool2D, AvgPool2D)):
            h = (h - layer.kh) // layer.stride + 1
            w = (w - layer.kw) // layer.stride + 1
        elif isinstance(layer, GlobalAvgPool2D):
            return (c,)
        elif isinstance(layer, Flatten):
            return (c * h * w,)
        elif isinstance(layer, (Activation2D, BatchNorm2D, Dropout2D, ResidualBlock)):
            pass  # Shape unchanged
    return (c, h, w)


def create_image(channels, height, width, value=0.0):
    """Create a Tensor3D image filled with a value."""
    return Tensor3D([[[value] * width for _ in range(height)]
                     for _ in range(channels)])


def image_from_array(array_2d, channels=1):
    """Convert 2D array (H, W) to Tensor3D (C, H, W)."""
    h = len(array_2d)
    w = len(array_2d[0])
    return Tensor3D([[[array_2d[hh][ww] for ww in range(w)]
                      for hh in range(h)]
                     for _ in range(channels)])
