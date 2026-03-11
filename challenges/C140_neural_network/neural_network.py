"""
C140: Neural Network
Feedforward neural network with backpropagation.

Composes:
- C128 Automatic Differentiation (Var, reverse-mode AD for backprop)
- C138 Optimization (line search utilities)

Features:
- Tensor: lightweight multi-dimensional array
- Layers: Dense, Dropout, BatchNorm
- Activations: ReLU, Sigmoid, Tanh, Softmax, LeakyReLU
- Loss functions: MSE, CrossEntropy, BinaryCrossEntropy
- Optimizers: SGD (with momentum), Adam, RMSProp
- Sequential model with fit/predict/evaluate API
- Xavier/He weight initialization
- Mini-batch training with shuffling
- Learning rate scheduling
- Gradient clipping
- L2 regularization (weight decay)
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C128_automatic_differentiation'))
from autodiff import Var, var_exp, var_log, var_tanh, var_sigmoid, var_relu


# ============================================================
# Tensor: lightweight multi-dimensional array
# ============================================================

class Tensor:
    """Simple multi-dimensional array for neural network computations."""

    def __init__(self, data):
        if isinstance(data, Tensor):
            self.data = [row[:] for row in data.data] if isinstance(data.data[0], list) else data.data[:]
            self.shape = data.shape
        elif isinstance(data, list):
            if len(data) == 0:
                self.data = []
                self.shape = (0,)
            elif isinstance(data[0], list):
                # 2D
                self.data = [row[:] for row in data]
                self.shape = (len(data), len(data[0]))
            else:
                # 1D
                self.data = data[:]
                self.shape = (len(data),)
        else:
            raise ValueError("Tensor requires list data")

    @staticmethod
    def zeros(shape):
        if isinstance(shape, int):
            return Tensor([0.0] * shape)
        if len(shape) == 1:
            return Tensor([0.0] * shape[0])
        return Tensor([[0.0] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def ones(shape):
        if isinstance(shape, int):
            return Tensor([1.0] * shape)
        if len(shape) == 1:
            return Tensor([1.0] * shape[0])
        return Tensor([[1.0] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def random_normal(shape, mean=0.0, std=1.0, rng=None):
        """Random tensor from normal distribution (Box-Muller)."""
        def _randn():
            if rng:
                u1 = rng.random()
                u2 = rng.random()
            else:
                u1 = random.random()
                u2 = random.random()
            # Avoid log(0)
            u1 = max(u1, 1e-10)
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            return mean + std * z

        if isinstance(shape, int):
            return Tensor([_randn() for _ in range(shape)])
        if len(shape) == 1:
            return Tensor([_randn() for _ in range(shape[0])])
        return Tensor([[_randn() for _ in range(shape[1])] for _ in range(shape[0])])

    @staticmethod
    def random_uniform(shape, low=0.0, high=1.0, rng=None):
        def _rand():
            r = rng.random() if rng else random.random()
            return low + (high - low) * r

        if isinstance(shape, int):
            return Tensor([_rand() for _ in range(shape)])
        if len(shape) == 1:
            return Tensor([_rand() for _ in range(shape[0])])
        return Tensor([[_rand() for _ in range(shape[1])] for _ in range(shape[0])])

    def __repr__(self):
        return f"Tensor({self.data})"

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        if len(self.shape) == 2:
            row = self.data[idx]
            if isinstance(row, list):
                return Tensor(row)
            return row
        return self.data[idx]

    def __setitem__(self, idx, value):
        if isinstance(value, Tensor):
            if len(self.shape) == 2:
                self.data[idx] = value.data[:]
            else:
                self.data[idx] = value.data if isinstance(value.data, list) else value
        else:
            if len(self.shape) == 2 and isinstance(value, list):
                self.data[idx] = value[:]
            else:
                self.data[idx] = value

    def tolist(self):
        return self.data

    def flatten(self):
        if len(self.shape) == 1:
            return Tensor(self.data[:])
        result = []
        for row in self.data:
            result.extend(row)
        return Tensor(result)

    def reshape(self, shape):
        flat = self.flatten().data
        if len(shape) == 1:
            return Tensor(flat)
        rows, cols = shape
        return Tensor([flat[i * cols:(i + 1) * cols] for i in range(rows)])

    def T(self):
        """Transpose (2D only)."""
        if len(self.shape) == 1:
            # Column vector
            return Tensor([[x] for x in self.data])
        rows, cols = self.shape
        return Tensor([[self.data[r][c] for r in range(rows)] for c in range(cols)])

    def dot(self, other):
        """Matrix multiply or dot product."""
        if len(self.shape) == 1 and len(other.shape) == 1:
            return sum(a * b for a, b in zip(self.data, other.data))
        if len(self.shape) == 2 and len(other.shape) == 1:
            # Matrix-vector
            return Tensor([sum(self.data[i][j] * other.data[j]
                              for j in range(self.shape[1]))
                          for i in range(self.shape[0])])
        if len(self.shape) == 2 and len(other.shape) == 2:
            # Matrix-matrix
            rows_a, cols_a = self.shape
            rows_b, cols_b = other.shape
            return Tensor([[sum(self.data[i][k] * other.data[k][j]
                               for k in range(cols_a))
                           for j in range(cols_b)]
                          for i in range(rows_a)])
        raise ValueError(f"Cannot dot shapes {self.shape} and {other.shape}")

    def __add__(self, other):
        if isinstance(other, (int, float)):
            if len(self.shape) == 1:
                return Tensor([x + other for x in self.data])
            return Tensor([[x + other for x in row] for row in self.data])
        if len(self.shape) == 1 and len(other.shape) == 1:
            return Tensor([a + b for a, b in zip(self.data, other.data)])
        if len(self.shape) == 2 and len(other.shape) == 1:
            # Broadcast: add vector to each row
            return Tensor([[self.data[i][j] + other.data[j]
                           for j in range(self.shape[1])]
                          for i in range(self.shape[0])])
        if len(self.shape) == 2 and len(other.shape) == 2:
            return Tensor([[self.data[i][j] + other.data[i][j]
                           for j in range(self.shape[1])]
                          for i in range(self.shape[0])])
        raise ValueError(f"Cannot add shapes {self.shape} and {other.shape}")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            if len(self.shape) == 1:
                return Tensor([x - other for x in self.data])
            return Tensor([[x - other for x in row] for row in self.data])
        if isinstance(other, Tensor):
            if len(self.shape) == 1 and len(other.shape) == 1:
                return Tensor([a - b for a, b in zip(self.data, other.data)])
            if len(self.shape) == 2 and len(other.shape) == 2:
                return Tensor([[self.data[i][j] - other.data[i][j]
                               for j in range(self.shape[1])]
                              for i in range(self.shape[0])])
            if len(self.shape) == 2 and len(other.shape) == 1:
                return Tensor([[self.data[i][j] - other.data[j]
                               for j in range(self.shape[1])]
                              for i in range(self.shape[0])])
        raise ValueError(f"Cannot subtract")

    def __mul__(self, other):
        """Element-wise multiplication or scalar."""
        if isinstance(other, (int, float)):
            if len(self.shape) == 1:
                return Tensor([x * other for x in self.data])
            return Tensor([[x * other for x in row] for row in self.data])
        if isinstance(other, Tensor):
            if self.shape == other.shape:
                if len(self.shape) == 1:
                    return Tensor([a * b for a, b in zip(self.data, other.data)])
                return Tensor([[self.data[i][j] * other.data[i][j]
                               for j in range(self.shape[1])]
                              for i in range(self.shape[0])])
        raise ValueError(f"Cannot multiply shapes {self.shape} and {getattr(other, 'shape', type(other))}")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * -1.0

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self * (1.0 / other)
        raise ValueError("Division only by scalar")

    def sum(self, axis=None):
        if axis is None:
            if len(self.shape) == 1:
                return sum(self.data)
            return sum(sum(row) for row in self.data)
        if axis == 0:
            # Sum along rows -> 1D
            cols = self.shape[1]
            return Tensor([sum(self.data[i][j] for i in range(self.shape[0]))
                          for j in range(cols)])
        if axis == 1:
            # Sum along columns -> 1D
            return Tensor([sum(row) for row in self.data])
        raise ValueError(f"Invalid axis {axis}")

    def mean(self, axis=None):
        if axis is None:
            s = self.sum()
            n = self.shape[0] if len(self.shape) == 1 else self.shape[0] * self.shape[1]
            return s / n
        if axis == 0:
            s = self.sum(axis=0)
            return s / self.shape[0]
        if axis == 1:
            s = self.sum(axis=1)
            return s / self.shape[1]

    def apply(self, func):
        """Apply function element-wise."""
        if len(self.shape) == 1:
            return Tensor([func(x) for x in self.data])
        return Tensor([[func(x) for x in row] for row in self.data])

    def max_val(self):
        if len(self.shape) == 1:
            return max(self.data)
        return max(max(row) for row in self.data)

    def argmax(self, axis=None):
        if axis is None and len(self.shape) == 1:
            return max(range(len(self.data)), key=lambda i: self.data[i])
        if axis == 1:
            return [max(range(len(row)), key=lambda j: row[j]) for row in self.data]
        raise ValueError(f"argmax axis={axis} not supported")

    def clip(self, low, high):
        def _clip(x):
            return max(low, min(high, x))
        return self.apply(_clip)

    def copy(self):
        return Tensor(self)


# ============================================================
# Weight Initialization
# ============================================================

def xavier_init(fan_in, fan_out, rng=None):
    """Xavier/Glorot initialization."""
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return Tensor.random_normal((fan_in, fan_out), std=std, rng=rng)


def he_init(fan_in, fan_out, rng=None):
    """He/Kaiming initialization for ReLU networks."""
    std = math.sqrt(2.0 / fan_in)
    return Tensor.random_normal((fan_in, fan_out), std=std, rng=rng)


def lecun_init(fan_in, fan_out, rng=None):
    """LeCun initialization."""
    std = math.sqrt(1.0 / fan_in)
    return Tensor.random_normal((fan_in, fan_out), std=std, rng=rng)


# ============================================================
# Activation Functions (operate on lists of floats)
# ============================================================

def relu(x):
    return max(0.0, x)

def relu_deriv(x):
    return 1.0 if x > 0 else 0.0

def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1.0 - s)

def tanh_act(x):
    return math.tanh(x)

def tanh_deriv(x):
    t = math.tanh(x)
    return 1.0 - t * t

def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x

def leaky_relu_deriv(x, alpha=0.01):
    return 1.0 if x > 0 else alpha

def softmax(values):
    """Numerically stable softmax on a list of floats."""
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    s = sum(exps)
    return [e / s for e in exps]

def softmax_batch(tensor):
    """Softmax on each row of a 2D tensor."""
    if len(tensor.shape) == 1:
        return Tensor(softmax(tensor.data))
    return Tensor([softmax(row) for row in tensor.data])


# ============================================================
# Loss Functions
# ============================================================

class MSELoss:
    """Mean Squared Error loss."""

    def forward(self, predicted, target):
        """
        predicted, target: Tensor (batch_size, output_size) or (output_size,)
        Returns scalar loss.
        """
        if len(predicted.shape) == 1:
            n = len(predicted.data)
            return sum((p - t) ** 2 for p, t in zip(predicted.data, target.data)) / n
        # Batch
        batch_size = predicted.shape[0]
        total = 0.0
        for i in range(batch_size):
            for j in range(predicted.shape[1]):
                d = predicted.data[i][j] - target.data[i][j]
                total += d * d
        return total / (batch_size * predicted.shape[1])

    def backward(self, predicted, target):
        """Returns gradient of loss w.r.t. predicted."""
        if len(predicted.shape) == 1:
            n = len(predicted.data)
            return Tensor([2.0 * (p - t) / n for p, t in zip(predicted.data, target.data)])
        batch_size = predicted.shape[0]
        cols = predicted.shape[1]
        scale = 2.0 / (batch_size * cols)
        return Tensor([[scale * (predicted.data[i][j] - target.data[i][j])
                        for j in range(cols)]
                       for i in range(batch_size)])


class CrossEntropyLoss:
    """Cross-entropy loss for multi-class classification (with softmax)."""

    def forward(self, logits, targets):
        """
        logits: Tensor (batch_size, num_classes) -- raw scores
        targets: list of int (class indices) or Tensor of one-hot vectors
        Returns scalar loss.
        """
        if len(logits.shape) == 1:
            # Single sample
            probs = softmax(logits.data)
            if isinstance(targets, int):
                return -math.log(max(probs[targets], 1e-15))
            # One-hot
            return -sum(t * math.log(max(p, 1e-15))
                       for p, t in zip(probs, targets.data if isinstance(targets, Tensor) else targets))

        batch_size = logits.shape[0]
        total = 0.0
        for i in range(batch_size):
            probs = softmax(logits.data[i])
            if isinstance(targets, list) and isinstance(targets[0], int):
                total -= math.log(max(probs[targets[i]], 1e-15))
            elif isinstance(targets, Tensor) and len(targets.shape) == 2:
                for j in range(logits.shape[1]):
                    total -= targets.data[i][j] * math.log(max(probs[j], 1e-15))
            elif isinstance(targets, list):
                total -= math.log(max(probs[targets[i]], 1e-15))
        return total / batch_size

    def backward(self, logits, targets):
        """Gradient of CE loss w.r.t. logits (softmax - one_hot)."""
        if len(logits.shape) == 1:
            probs = softmax(logits.data)
            if isinstance(targets, int):
                grad = probs[:]
                grad[targets] -= 1.0
                return Tensor(grad)
            t = targets.data if isinstance(targets, Tensor) else targets
            return Tensor([p - tt for p, tt in zip(probs, t)])

        batch_size = logits.shape[0]
        cols = logits.shape[1]
        grads = []
        for i in range(batch_size):
            probs = softmax(logits.data[i])
            if isinstance(targets, list) and isinstance(targets[0], int):
                row = probs[:]
                row[targets[i]] -= 1.0
                grads.append([g / batch_size for g in row])
            elif isinstance(targets, Tensor) and len(targets.shape) == 2:
                grads.append([(probs[j] - targets.data[i][j]) / batch_size for j in range(cols)])
            else:
                row = probs[:]
                row[targets[i]] -= 1.0
                grads.append([g / batch_size for g in row])
        return Tensor(grads)


class BinaryCrossEntropyLoss:
    """Binary cross-entropy loss."""

    def forward(self, predicted, target):
        """predicted: after sigmoid, target: 0 or 1 values."""
        eps = 1e-15
        if len(predicted.shape) == 1:
            n = len(predicted.data)
            total = 0.0
            for p, t in zip(predicted.data, target.data):
                p_c = max(min(p, 1.0 - eps), eps)
                total -= t * math.log(p_c) + (1.0 - t) * math.log(1.0 - p_c)
            return total / n

        batch_size = predicted.shape[0]
        total = 0.0
        count = 0
        for i in range(batch_size):
            for j in range(predicted.shape[1]):
                p = max(min(predicted.data[i][j], 1.0 - eps), eps)
                t = target.data[i][j]
                total -= t * math.log(p) + (1.0 - t) * math.log(1.0 - p)
                count += 1
        return total / count

    def backward(self, predicted, target):
        eps = 1e-15
        if len(predicted.shape) == 1:
            n = len(predicted.data)
            return Tensor([((max(min(p, 1.0 - eps), eps) - t) /
                           (max(min(p, 1.0 - eps), eps) * (1.0 - max(min(p, 1.0 - eps), eps)))) / n
                          for p, t in zip(predicted.data, target.data)])
        batch_size = predicted.shape[0]
        cols = predicted.shape[1]
        count = batch_size * cols
        grads = []
        for i in range(batch_size):
            row = []
            for j in range(cols):
                p = max(min(predicted.data[i][j], 1.0 - eps), eps)
                t = target.data[i][j]
                row.append(((p - t) / (p * (1.0 - p))) / count)
            grads.append(row)
        return Tensor(grads)


# ============================================================
# Layers
# ============================================================

class Layer:
    """Base class for network layers."""
    def __init__(self):
        self.params = []  # list of (weights_tensor, bias_tensor)
        self.training = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def get_params(self):
        return self.params

    def set_training(self, mode):
        self.training = mode


class Dense(Layer):
    """Fully connected layer."""

    def __init__(self, input_size, output_size, init='xavier', rng=None, bias=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = bias

        if init == 'xavier':
            self.weights = xavier_init(input_size, output_size, rng=rng)
        elif init == 'he':
            self.weights = he_init(input_size, output_size, rng=rng)
        elif init == 'lecun':
            self.weights = lecun_init(input_size, output_size, rng=rng)
        elif init == 'zeros':
            self.weights = Tensor.zeros((input_size, output_size))
        else:
            self.weights = xavier_init(input_size, output_size, rng=rng)

        self.bias = Tensor.zeros(output_size) if bias else None

        # Cache for backward
        self._input = None
        self._pre_activation = None

        # Gradients
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, x):
        """x: Tensor (batch_size, input_size) or (input_size,)"""
        self._input = x
        if len(x.shape) == 1:
            # Single sample
            out = self.weights.T().dot(Tensor(x.data))
            if self.use_bias:
                out = out + self.bias
            self._pre_activation = out
            return out
        # Batch: x @ weights + bias
        out = x.dot(self.weights)
        if self.use_bias:
            out = out + self.bias  # broadcast
        self._pre_activation = out
        return out

    def backward(self, grad_output):
        """
        grad_output: Tensor same shape as forward output
        Returns: grad_input (same shape as forward input)
        """
        x = self._input
        if len(x.shape) == 1:
            # Single sample: grad_output is (output_size,)
            # grad_weights[i][j] = x[i] * grad_output[j]
            self.grad_weights = Tensor([[x.data[i] * grad_output.data[j]
                                         for j in range(self.output_size)]
                                        for i in range(self.input_size)])
            if self.use_bias:
                self.grad_bias = Tensor(grad_output.data[:])
            # grad_input[i] = sum_j(weights[i][j] * grad_output[j])
            grad_input = self.weights.dot(Tensor(grad_output.data))
            return grad_input

        # Batch: grad_output is (batch_size, output_size)
        batch_size = x.shape[0]
        # grad_weights = x^T @ grad_output
        self.grad_weights = x.T().dot(grad_output)
        if self.use_bias:
            self.grad_bias = grad_output.sum(axis=0)
        # grad_input = grad_output @ weights^T
        grad_input = grad_output.dot(self.weights.T())
        return grad_input

    def get_params(self):
        if self.use_bias:
            return [(self.weights, self.grad_weights, 'weight'),
                    (self.bias, self.grad_bias, 'bias')]
        return [(self.weights, self.grad_weights, 'weight')]


class Activation(Layer):
    """Activation layer."""

    def __init__(self, name='relu', alpha=0.01):
        super().__init__()
        self.name = name
        self.alpha = alpha  # for leaky_relu
        self._input = None

    def forward(self, x):
        self._input = x
        if self.name == 'relu':
            return x.apply(relu)
        elif self.name == 'sigmoid':
            return x.apply(sigmoid)
        elif self.name == 'tanh':
            return x.apply(tanh_act)
        elif self.name == 'leaky_relu':
            return x.apply(lambda v: leaky_relu(v, self.alpha))
        elif self.name == 'softmax':
            return softmax_batch(x)
        elif self.name == 'linear' or self.name == 'none':
            return x.copy()
        raise ValueError(f"Unknown activation: {self.name}")

    def backward(self, grad_output):
        x = self._input
        if self.name == 'relu':
            mask = x.apply(relu_deriv)
            return grad_output * mask
        elif self.name == 'sigmoid':
            deriv = x.apply(sigmoid_deriv)
            return grad_output * deriv
        elif self.name == 'tanh':
            deriv = x.apply(tanh_deriv)
            return grad_output * deriv
        elif self.name == 'leaky_relu':
            mask = x.apply(lambda v: leaky_relu_deriv(v, self.alpha))
            return grad_output * mask
        elif self.name == 'softmax':
            # For softmax + CE loss, gradient is handled by CE loss backward
            # This returns grad_output unchanged (identity for softmax layer
            # when combined with cross-entropy)
            return grad_output
        elif self.name == 'linear' or self.name == 'none':
            return grad_output
        raise ValueError(f"Unknown activation: {self.name}")


class Dropout(Layer):
    """Dropout regularization layer."""

    def __init__(self, rate=0.5, rng=None):
        super().__init__()
        self.rate = rate
        self.rng = rng or random.Random()
        self._mask = None

    def forward(self, x):
        if not self.training or self.rate == 0.0:
            return x.copy()

        # Create mask
        scale = 1.0 / (1.0 - self.rate)
        if len(x.shape) == 1:
            self._mask = Tensor([scale if self.rng.random() > self.rate else 0.0
                                for _ in range(x.shape[0])])
        else:
            self._mask = Tensor([[scale if self.rng.random() > self.rate else 0.0
                                 for _ in range(x.shape[1])]
                                for _ in range(x.shape[0])])
        return x * self._mask

    def backward(self, grad_output):
        if not self.training or self.rate == 0.0:
            return grad_output
        return grad_output * self._mask


class BatchNorm(Layer):
    """Batch normalization layer (simplified, 1D)."""

    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # Learnable parameters
        self.gamma = Tensor([1.0] * num_features)
        self.beta = Tensor([0.0] * num_features)

        # Running statistics for inference
        self.running_mean = Tensor([0.0] * num_features)
        self.running_var = Tensor([1.0] * num_features)

        # Cache
        self._x_norm = None
        self._std_inv = None
        self._x_centered = None
        self._input = None
        self.grad_gamma = None
        self.grad_beta = None

    def forward(self, x):
        self._input = x
        if len(x.shape) == 1:
            # Single sample -- use running stats
            x_norm_data = [(x.data[j] - self.running_mean.data[j]) /
                          math.sqrt(self.running_var.data[j] + self.eps)
                          for j in range(self.num_features)]
            out = [self.gamma.data[j] * x_norm_data[j] + self.beta.data[j]
                  for j in range(self.num_features)]
            return Tensor(out)

        batch_size = x.shape[0]
        n = self.num_features

        if self.training:
            # Compute batch statistics
            mean = x.mean(axis=0)
            x_centered = x - mean
            var_data = [0.0] * n
            for i in range(batch_size):
                for j in range(n):
                    var_data[j] += x_centered.data[i][j] ** 2
            var_data = [v / batch_size for v in var_data]
            var = Tensor(var_data)

            std_inv = Tensor([1.0 / math.sqrt(v + self.eps) for v in var_data])

            x_norm = Tensor([[x_centered.data[i][j] * std_inv.data[j]
                             for j in range(n)]
                            for i in range(batch_size)])

            # Update running stats
            for j in range(n):
                self.running_mean.data[j] = (1 - self.momentum) * self.running_mean.data[j] + self.momentum * mean.data[j]
                self.running_var.data[j] = (1 - self.momentum) * self.running_var.data[j] + self.momentum * var_data[j]

            self._x_norm = x_norm
            self._std_inv = std_inv
            self._x_centered = x_centered
        else:
            x_norm = Tensor([[(x.data[i][j] - self.running_mean.data[j]) /
                              math.sqrt(self.running_var.data[j] + self.eps)
                             for j in range(n)]
                            for i in range(batch_size)])
            self._x_norm = x_norm

        # Scale and shift
        out = Tensor([[self.gamma.data[j] * x_norm.data[i][j] + self.beta.data[j]
                       for j in range(n)]
                      for i in range(batch_size)])
        return out

    def backward(self, grad_output):
        if len(grad_output.shape) == 1:
            # Single sample -- approximate
            self.grad_gamma = Tensor([grad_output.data[j] * self._x_norm.data[j]
                                     if isinstance(self._x_norm.data[j], (int, float))
                                     else grad_output.data[j]
                                     for j in range(self.num_features)])
            self.grad_beta = Tensor(grad_output.data[:])
            return grad_output

        batch_size = grad_output.shape[0]
        n = self.num_features

        # grad_gamma = sum(grad_output * x_norm, axis=0)
        self.grad_gamma = Tensor([sum(grad_output.data[i][j] * self._x_norm.data[i][j]
                                      for i in range(batch_size))
                                  for j in range(n)])
        self.grad_beta = grad_output.sum(axis=0)

        # grad_input (full batch norm backward)
        dx_norm = Tensor([[grad_output.data[i][j] * self.gamma.data[j]
                          for j in range(n)]
                         for i in range(batch_size)])

        dvar = Tensor([sum(dx_norm.data[i][j] * self._x_centered.data[i][j] *
                          (-0.5) * (self._std_inv.data[j] ** 3)
                          for i in range(batch_size))
                      for j in range(n)])

        dmean = Tensor([sum(-dx_norm.data[i][j] * self._std_inv.data[j]
                           for i in range(batch_size))
                       for j in range(n)])

        grad_input = Tensor([[dx_norm.data[i][j] * self._std_inv.data[j] +
                              dvar.data[j] * 2.0 * self._x_centered.data[i][j] / batch_size +
                              dmean.data[j] / batch_size
                             for j in range(n)]
                            for i in range(batch_size)])
        return grad_input

    def get_params(self):
        return [(self.gamma, self.grad_gamma, 'gamma'),
                (self.beta, self.grad_beta, 'beta')]


# ============================================================
# Optimizers
# ============================================================

class SGD:
    """Stochastic Gradient Descent with optional momentum."""

    def __init__(self, lr=0.01, momentum=0.0, weight_decay=0.0, clip_norm=None):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.velocities = {}

    def step(self, layers):
        """Update parameters of all layers."""
        for layer_idx, layer in enumerate(layers):
            for param_idx, param_tuple in enumerate(layer.get_params()):
                tensor, grad, name = param_tuple
                if grad is None:
                    continue

                key = (layer_idx, param_idx)

                # Gradient clipping
                if self.clip_norm is not None:
                    grad = self._clip_grad(grad)

                # Weight decay (L2 regularization)
                if self.weight_decay > 0 and name == 'weight':
                    if len(grad.shape) == 1:
                        grad = Tensor([g + self.weight_decay * w
                                      for g, w in zip(grad.data, tensor.data)])
                    else:
                        grad = Tensor([[g + self.weight_decay * tensor.data[i][j]
                                       for j, g in enumerate(row)]
                                      for i, row in enumerate(grad.data)])

                if self.momentum > 0:
                    if key not in self.velocities:
                        self.velocities[key] = Tensor.zeros(tensor.shape)
                    v = self.velocities[key]
                    # v = momentum * v + grad
                    if len(tensor.shape) == 1:
                        for i in range(len(v.data)):
                            v.data[i] = self.momentum * v.data[i] + grad.data[i]
                            tensor.data[i] -= self.lr * v.data[i]
                    else:
                        for i in range(tensor.shape[0]):
                            for j in range(tensor.shape[1]):
                                v.data[i][j] = self.momentum * v.data[i][j] + grad.data[i][j]
                                tensor.data[i][j] -= self.lr * v.data[i][j]
                else:
                    if len(tensor.shape) == 1:
                        for i in range(len(tensor.data)):
                            tensor.data[i] -= self.lr * grad.data[i]
                    else:
                        for i in range(tensor.shape[0]):
                            for j in range(tensor.shape[1]):
                                tensor.data[i][j] -= self.lr * grad.data[i][j]

    def _clip_grad(self, grad):
        if len(grad.shape) == 1:
            norm = math.sqrt(sum(g * g for g in grad.data))
        else:
            norm = math.sqrt(sum(g * g for row in grad.data for g in row))
        if norm > self.clip_norm:
            scale = self.clip_norm / norm
            if len(grad.shape) == 1:
                return Tensor([g * scale for g in grad.data])
            return Tensor([[g * scale for g in row] for row in grad.data])
        return grad


class Adam:
    """Adam optimizer."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0.0, clip_norm=None):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step

    def step(self, layers):
        self.t += 1
        for layer_idx, layer in enumerate(layers):
            for param_idx, param_tuple in enumerate(layer.get_params()):
                tensor, grad, name = param_tuple
                if grad is None:
                    continue

                key = (layer_idx, param_idx)

                # Gradient clipping
                if self.clip_norm is not None:
                    grad = SGD._clip_grad(None, grad)  # reuse static-like method
                    # Actually let's inline it
                    if len(grad.shape) == 1:
                        norm = math.sqrt(sum(g * g for g in grad.data))
                    else:
                        norm = math.sqrt(sum(g * g for row in grad.data for g in row))
                    if norm > self.clip_norm:
                        scale = self.clip_norm / norm
                        if len(grad.shape) == 1:
                            grad = Tensor([g * scale for g in grad.data])
                        else:
                            grad = Tensor([[g * scale for g in row] for row in grad.data])

                # Weight decay
                if self.weight_decay > 0 and name == 'weight':
                    if len(grad.shape) == 1:
                        grad = Tensor([g + self.weight_decay * w
                                      for g, w in zip(grad.data, tensor.data)])
                    else:
                        grad = Tensor([[g + self.weight_decay * tensor.data[i][j]
                                       for j, g in enumerate(row)]
                                      for i, row in enumerate(grad.data)])

                if key not in self.m:
                    self.m[key] = Tensor.zeros(tensor.shape)
                    self.v[key] = Tensor.zeros(tensor.shape)

                m = self.m[key]
                v = self.v[key]

                if len(tensor.shape) == 1:
                    for i in range(len(tensor.data)):
                        g = grad.data[i]
                        m.data[i] = self.beta1 * m.data[i] + (1 - self.beta1) * g
                        v.data[i] = self.beta2 * v.data[i] + (1 - self.beta2) * g * g
                        m_hat = m.data[i] / (1 - self.beta1 ** self.t)
                        v_hat = v.data[i] / (1 - self.beta2 ** self.t)
                        tensor.data[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
                else:
                    for i in range(tensor.shape[0]):
                        for j in range(tensor.shape[1]):
                            g = grad.data[i][j]
                            m.data[i][j] = self.beta1 * m.data[i][j] + (1 - self.beta1) * g
                            v.data[i][j] = self.beta2 * v.data[i][j] + (1 - self.beta2) * g * g
                            m_hat = m.data[i][j] / (1 - self.beta1 ** self.t)
                            v_hat = v.data[i][j] / (1 - self.beta2 ** self.t)
                            tensor.data[i][j] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


class RMSProp:
    """RMSProp optimizer."""

    def __init__(self, lr=0.001, alpha=0.99, eps=1e-8, weight_decay=0.0, clip_norm=None):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.v = {}

    def step(self, layers):
        for layer_idx, layer in enumerate(layers):
            for param_idx, param_tuple in enumerate(layer.get_params()):
                tensor, grad, name = param_tuple
                if grad is None:
                    continue

                key = (layer_idx, param_idx)

                # Weight decay
                if self.weight_decay > 0 and name == 'weight':
                    if len(grad.shape) == 1:
                        grad = Tensor([g + self.weight_decay * w
                                      for g, w in zip(grad.data, tensor.data)])
                    else:
                        grad = Tensor([[g + self.weight_decay * tensor.data[i][j]
                                       for j, g in enumerate(row)]
                                      for i, row in enumerate(grad.data)])

                if key not in self.v:
                    self.v[key] = Tensor.zeros(tensor.shape)

                v = self.v[key]

                if len(tensor.shape) == 1:
                    for i in range(len(tensor.data)):
                        g = grad.data[i]
                        v.data[i] = self.alpha * v.data[i] + (1 - self.alpha) * g * g
                        tensor.data[i] -= self.lr * g / (math.sqrt(v.data[i]) + self.eps)
                else:
                    for i in range(tensor.shape[0]):
                        for j in range(tensor.shape[1]):
                            g = grad.data[i][j]
                            v.data[i][j] = self.alpha * v.data[i][j] + (1 - self.alpha) * g * g
                            tensor.data[i][j] -= self.lr * g / (math.sqrt(v.data[i][j]) + self.eps)


# ============================================================
# Learning Rate Schedulers
# ============================================================

class StepLR:
    """Step learning rate decay."""
    def __init__(self, optimizer, step_size=10, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0
        self.initial_lr = optimizer.lr

    def step(self):
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma

    def get_lr(self):
        return self.optimizer.lr


class ExponentialLR:
    """Exponential learning rate decay."""
    def __init__(self, optimizer, gamma=0.95):
        self.optimizer = optimizer
        self.gamma = gamma
        self.epoch = 0

    def step(self):
        self.epoch += 1
        self.optimizer.lr *= self.gamma

    def get_lr(self):
        return self.optimizer.lr


class CosineAnnealingLR:
    """Cosine annealing learning rate."""
    def __init__(self, optimizer, T_max, eta_min=0.0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.initial_lr = optimizer.lr
        self.epoch = 0

    def step(self):
        self.epoch += 1
        self.optimizer.lr = self.eta_min + (self.initial_lr - self.eta_min) * \
            (1 + math.cos(math.pi * self.epoch / self.T_max)) / 2

    def get_lr(self):
        return self.optimizer.lr


# ============================================================
# Sequential Model
# ============================================================

class Sequential:
    """Sequential neural network model."""

    def __init__(self, layers=None):
        self.layers = layers or []

    def add(self, layer):
        self.layers.append(layer)
        return self

    def forward(self, x):
        """Forward pass through all layers."""
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad):
        """Backward pass through all layers (reverse order)."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict(self, x):
        """Forward pass in eval mode."""
        self.eval()
        result = self.forward(x)
        self.train()
        return result

    def train(self):
        for layer in self.layers:
            layer.set_training(True)

    def eval(self):
        for layer in self.layers:
            layer.set_training(False)

    def get_trainable_layers(self):
        """Return layers that have parameters."""
        return [l for l in self.layers if l.get_params()]

    def count_params(self):
        """Count total trainable parameters."""
        total = 0
        for layer in self.layers:
            for param_tuple in layer.get_params():
                tensor = param_tuple[0]
                if len(tensor.shape) == 1:
                    total += tensor.shape[0]
                else:
                    total += tensor.shape[0] * tensor.shape[1]
        return total

    def summary(self):
        """Return model summary as string."""
        lines = ["Layer | Output Shape | Params"]
        lines.append("-" * 40)
        total = 0
        for i, layer in enumerate(self.layers):
            name = layer.__class__.__name__
            params = 0
            for pt in layer.get_params():
                t = pt[0]
                if len(t.shape) == 1:
                    params += t.shape[0]
                else:
                    params += t.shape[0] * t.shape[1]
            total += params
            if isinstance(layer, Dense):
                lines.append(f"{name}({layer.input_size},{layer.output_size}) | ({layer.output_size},) | {params}")
            elif isinstance(layer, Activation):
                lines.append(f"{name}({layer.name}) | - | 0")
            elif isinstance(layer, Dropout):
                lines.append(f"{name}({layer.rate}) | - | 0")
            elif isinstance(layer, BatchNorm):
                lines.append(f"{name}({layer.num_features}) | ({layer.num_features},) | {params}")
            else:
                lines.append(f"{name} | - | {params}")
        lines.append(f"Total params: {total}")
        return "\n".join(lines)


# ============================================================
# Training Utilities
# ============================================================

def train_step(model, x_batch, y_batch, loss_fn, optimizer):
    """Single training step: forward, loss, backward, update."""
    # Forward
    output = model.forward(x_batch)
    # Loss
    loss = loss_fn.forward(output, y_batch)
    # Backward
    grad = loss_fn.backward(output, y_batch)
    model.backward(grad)
    # Update
    optimizer.step(model.get_trainable_layers())
    return loss


def fit(model, X, Y, loss_fn, optimizer, epochs=100, batch_size=None,
        shuffle=True, verbose=False, scheduler=None, validation_data=None):
    """
    Train the model.
    X: Tensor (num_samples, features)
    Y: Tensor (num_samples, outputs) or list of int
    Returns dict with 'loss_history' and optionally 'val_loss_history'.
    """
    history = {'loss': [], 'val_loss': []}

    if isinstance(Y, list) and isinstance(Y[0], int):
        targets_are_indices = True
        num_samples = X.shape[0] if len(X.shape) == 2 else 1
    else:
        targets_are_indices = False
        num_samples = X.shape[0] if len(X.shape) == 2 else 1

    if batch_size is None:
        batch_size = num_samples

    rng = random.Random(42)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Create indices
        indices = list(range(num_samples))
        if shuffle:
            rng.shuffle(indices)

        # Mini-batch training
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_idx = indices[start:end]
            bs = len(batch_idx)

            if len(X.shape) == 2:
                x_batch = Tensor([X.data[i] for i in batch_idx])
            else:
                x_batch = X

            if targets_are_indices:
                y_batch = [Y[i] for i in batch_idx]
            elif isinstance(Y, Tensor):
                if len(Y.shape) == 2:
                    y_batch = Tensor([Y.data[i] for i in batch_idx])
                else:
                    y_batch = Y
            else:
                y_batch = Y

            loss = train_step(model, x_batch, y_batch, loss_fn, optimizer)
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        history['loss'].append(avg_loss)

        # Validation
        if validation_data is not None:
            model.eval()
            val_x, val_y = validation_data
            val_output = model.forward(val_x)
            val_loss = loss_fn.forward(val_output, val_y)
            history['val_loss'].append(val_loss)
            model.train()

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            msg = f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.6f}"
            if validation_data is not None:
                msg += f" - val_loss: {history['val_loss'][-1]:.6f}"
            if scheduler is not None:
                msg += f" - lr: {scheduler.get_lr():.6f}"
            print(msg)

    return history


def evaluate(model, X, Y, loss_fn):
    """Evaluate model on data."""
    model.eval()
    output = model.forward(X)
    loss = loss_fn.forward(output, Y)
    model.train()
    return loss


def predict_classes(model, X):
    """Predict class labels (argmax of output)."""
    model.eval()
    output = model.forward(X)
    model.train()
    if len(output.shape) == 1:
        return output.argmax()
    return output.argmax(axis=1)


def accuracy(model, X, Y):
    """Compute classification accuracy."""
    preds = predict_classes(model, X)
    if isinstance(Y, list):
        targets = Y
    elif isinstance(Y, Tensor) and len(Y.shape) == 2:
        targets = Y.argmax(axis=1)
    else:
        targets = Y.data if isinstance(Y, Tensor) else Y

    if isinstance(preds, int):
        return 1.0 if preds == targets[0] else 0.0

    correct = sum(1 for p, t in zip(preds, targets) if p == t)
    return correct / len(preds)


# ============================================================
# Data Utilities
# ============================================================

def one_hot(labels, num_classes):
    """Convert class labels to one-hot encoding."""
    result = []
    for label in labels:
        row = [0.0] * num_classes
        row[label] = 1.0
        result.append(row)
    return Tensor(result)


def normalize(X, axis=0):
    """Normalize features to zero mean and unit variance."""
    if len(X.shape) == 1:
        mean_val = sum(X.data) / len(X.data)
        var_val = sum((x - mean_val) ** 2 for x in X.data) / len(X.data)
        std_val = math.sqrt(var_val + 1e-8)
        return Tensor([(x - mean_val) / std_val for x in X.data]), mean_val, std_val

    rows, cols = X.shape
    means = [0.0] * cols
    for i in range(rows):
        for j in range(cols):
            means[j] += X.data[i][j]
    means = [m / rows for m in means]

    stds = [0.0] * cols
    for i in range(rows):
        for j in range(cols):
            stds[j] += (X.data[i][j] - means[j]) ** 2
    stds = [math.sqrt(s / rows + 1e-8) for s in stds]

    normalized = [[(X.data[i][j] - means[j]) / stds[j] for j in range(cols)]
                  for i in range(rows)]
    return Tensor(normalized), means, stds


def train_test_split(X, Y, test_ratio=0.2, seed=42):
    """Split data into training and test sets."""
    rng = random.Random(seed)
    n = X.shape[0] if len(X.shape) == 2 else len(X.data)
    indices = list(range(n))
    rng.shuffle(indices)
    split = int(n * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]

    if len(X.shape) == 2:
        X_train = Tensor([X.data[i] for i in train_idx])
        X_test = Tensor([X.data[i] for i in test_idx])
    else:
        X_train = Tensor([X.data[i] for i in train_idx])
        X_test = Tensor([X.data[i] for i in test_idx])

    if isinstance(Y, list):
        Y_train = [Y[i] for i in train_idx]
        Y_test = [Y[i] for i in test_idx]
    elif isinstance(Y, Tensor):
        if len(Y.shape) == 2:
            Y_train = Tensor([Y.data[i] for i in train_idx])
            Y_test = Tensor([Y.data[i] for i in test_idx])
        else:
            Y_train = Tensor([Y.data[i] for i in train_idx])
            Y_test = Tensor([Y.data[i] for i in test_idx])
    else:
        Y_train = Y
        Y_test = Y

    return X_train, X_test, Y_train, Y_test


# ============================================================
# Data Generators (for testing)
# ============================================================

def make_xor_data(n=100, seed=42):
    """Generate XOR classification data."""
    rng = random.Random(seed)
    X = []
    Y = []
    for _ in range(n):
        x1 = rng.choice([0, 1])
        x2 = rng.choice([0, 1])
        X.append([float(x1), float(x2)])
        Y.append(x1 ^ x2)
    return Tensor(X), Y


def make_spiral_data(n_per_class=50, n_classes=3, seed=42):
    """Generate spiral classification data."""
    rng = random.Random(seed)
    X = []
    Y = []
    for c in range(n_classes):
        for i in range(n_per_class):
            r = i / n_per_class
            t = c * 4.0 + (i / n_per_class) * 4.0 + rng.gauss(0, 0.2)
            X.append([r * math.sin(t), r * math.cos(t)])
            Y.append(c)
    return Tensor(X), Y


def make_regression_data(n=100, noise=0.1, seed=42):
    """Generate simple regression data: y = 2*x + 1 + noise."""
    rng = random.Random(seed)
    X = []
    Y = []
    for _ in range(n):
        x = rng.uniform(-2, 2)
        y = 2.0 * x + 1.0 + rng.gauss(0, noise)
        X.append([x])
        Y.append([y])
    return Tensor(X), Tensor(Y)


def make_circles_data(n=200, noise=0.05, factor=0.5, seed=42):
    """Generate concentric circles classification data."""
    rng = random.Random(seed)
    X = []
    Y = []
    for i in range(n):
        if i < n // 2:
            angle = rng.uniform(0, 2 * math.pi)
            r = 1.0 + rng.gauss(0, noise)
            X.append([r * math.cos(angle), r * math.sin(angle)])
            Y.append(0)
        else:
            angle = rng.uniform(0, 2 * math.pi)
            r = factor + rng.gauss(0, noise)
            X.append([r * math.cos(angle), r * math.sin(angle)])
            Y.append(1)
    return Tensor(X), Y


# ============================================================
# Model Save/Load (dict-based)
# ============================================================

def save_weights(model):
    """Save model weights to a dict."""
    weights = {}
    for i, layer in enumerate(model.layers):
        for j, param_tuple in enumerate(layer.get_params()):
            tensor = param_tuple[0]
            weights[f"layer_{i}_param_{j}"] = tensor.tolist()
    return weights


def load_weights(model, weights):
    """Load weights from a dict into model."""
    for i, layer in enumerate(model.layers):
        params = layer.get_params()
        for j, param_tuple in enumerate(params):
            tensor = param_tuple[0]
            key = f"layer_{i}_param_{j}"
            if key in weights:
                data = weights[key]
                if isinstance(data[0], list):
                    for r in range(len(data)):
                        for c in range(len(data[r])):
                            tensor.data[r][c] = data[r][c]
                else:
                    for k in range(len(data)):
                        tensor.data[k] = data[k]


# ============================================================
# Convenience: build model from spec
# ============================================================

def build_model(layer_sizes, activations=None, init='xavier', dropout=None, rng=None):
    """
    Build a Sequential model from layer sizes.
    layer_sizes: [input, hidden1, hidden2, ..., output]
    activations: list of activation names (len = len(layer_sizes) - 1)
    """
    model = Sequential()
    if activations is None:
        activations = ['relu'] * (len(layer_sizes) - 2) + ['linear']

    for i in range(len(layer_sizes) - 1):
        model.add(Dense(layer_sizes[i], layer_sizes[i + 1], init=init, rng=rng))
        if i < len(activations):
            act = activations[i]
            if act != 'linear' and act != 'none':
                model.add(Activation(act))
        if dropout is not None and i < len(layer_sizes) - 2:
            model.add(Dropout(dropout, rng=rng))

    return model
