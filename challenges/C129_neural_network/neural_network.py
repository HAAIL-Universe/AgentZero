"""
C129: Neural Network Framework

A complete neural network library built on C128's automatic differentiation:
1. Tensor -- multi-dimensional array with AD support
2. Parameter -- trainable parameter with gradient tracking
3. Module -- base class for all neural network layers
4. Layers -- Linear, Conv1D, RNN, LSTM, BatchNorm, Dropout, Embedding
5. Activations -- ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, GELU, ELU
6. Loss functions -- MSE, CrossEntropy, BinaryCrossEntropy, Huber, L1
7. Optimizers -- SGD (momentum), Adam, RMSProp, AdaGrad
8. Trainer -- training loop with batching, validation, callbacks
9. Sequential/Functional -- model composition patterns
10. Regularization -- L1/L2, weight decay, gradient clipping
11. Initialization -- Xavier, He, uniform, normal
12. LR Schedulers -- StepLR, ExponentialLR, CosineAnnealing

Composes: C128 (AutomaticDifferentiation) Var, ReverseAD, var_* functions.
No numpy dependency -- all math from scratch.
"""

import sys
import os
import math
import random
from typing import Optional, Callable, List, Tuple, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C128_automatic_differentiation'))
from autodiff import Var, ReverseAD, var_exp, var_log, var_sqrt, var_tanh, var_sigmoid, var_relu


# ---------------------------------------------------------------------------
# Tensor: Multi-dimensional array with AD support
# ---------------------------------------------------------------------------

class Tensor:
    """Multi-dimensional array that can hold floats or Var nodes.

    Stored as a flat list with shape metadata. Supports basic operations.
    """
    __slots__ = ('data', 'shape', 'requires_grad')

    def __init__(self, data, shape=None, requires_grad=False):
        if isinstance(data, Tensor):
            self.data = list(data.data)
            self.shape = data.shape
            self.requires_grad = requires_grad or data.requires_grad
            return

        if isinstance(data, (int, float)):
            self.data = [float(data)]
            self.shape = (1,)
            self.requires_grad = requires_grad
            return

        # Nested list -> flatten
        if isinstance(data, list) and data and isinstance(data[0], list):
            if data[0] and isinstance(data[0][0], list):
                # 3D
                d0 = len(data)
                d1 = len(data[0])
                d2 = len(data[0][0])
                flat = []
                for i in range(d0):
                    for j in range(d1):
                        for k in range(d2):
                            flat.append(data[i][j][k])
                self.data = flat
                self.shape = (d0, d1, d2)
            else:
                # 2D
                rows = len(data)
                cols = len(data[0])
                flat = []
                for row in data:
                    flat.extend(row)
                self.data = flat
                self.shape = (rows, cols)
        elif isinstance(data, list):
            self.data = list(data)
            self.shape = shape if shape else (len(data),)
        else:
            self.data = [data]
            self.shape = (1,)

        self.requires_grad = requires_grad

    @property
    def size(self):
        result = 1
        for d in self.shape:
            result *= d
        return result

    @property
    def ndim(self):
        return len(self.shape)

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return f"Tensor(shape={self.shape})"

    def item(self):
        """Get scalar value."""
        if self.size == 1:
            v = self.data[0]
            return v.val if isinstance(v, Var) else float(v)
        raise ValueError("item() only for scalar tensors")

    def to_list(self):
        """Convert to nested list."""
        vals = [v.val if isinstance(v, Var) else float(v) for v in self.data]
        if len(self.shape) == 1:
            return vals
        if len(self.shape) == 2:
            rows, cols = self.shape
            return [vals[i * cols:(i + 1) * cols] for i in range(rows)]
        if len(self.shape) == 3:
            d0, d1, d2 = self.shape
            result = []
            for i in range(d0):
                plane = []
                for j in range(d1):
                    start = (i * d1 + j) * d2
                    plane.append(vals[start:start + d2])
                result.append(plane)
            return result
        return vals

    def to_vars(self):
        """Convert data to Var nodes for gradient tracking."""
        self.data = [Var(v.val if isinstance(v, Var) else float(v)) for v in self.data]
        self.requires_grad = True
        return self

    def get(self, *indices):
        """Get element by indices."""
        idx = 0
        stride = 1
        for i in range(len(self.shape) - 1, -1, -1):
            if i < len(indices):
                idx += indices[i] * stride
            stride *= self.shape[i]
        return self.data[idx]

    def set(self, indices, value):
        """Set element by indices."""
        idx = 0
        stride = 1
        for i in range(len(self.shape) - 1, -1, -1):
            if i < len(indices):
                idx += indices[i] * stride
            stride *= self.shape[i]
        self.data[idx] = value

    def reshape(self, new_shape):
        """Reshape tensor."""
        new_size = 1
        for d in new_shape:
            new_size *= d
        assert new_size == self.size, f"Cannot reshape {self.shape} to {new_shape}"
        return Tensor(list(self.data), shape=new_shape, requires_grad=self.requires_grad)

    @staticmethod
    def zeros(shape, requires_grad=False):
        size = 1
        for d in shape:
            size *= d
        data = [Var(0.0) if requires_grad else 0.0 for _ in range(size)]
        return Tensor(data, shape=shape, requires_grad=requires_grad)

    @staticmethod
    def ones(shape, requires_grad=False):
        size = 1
        for d in shape:
            size *= d
        data = [Var(1.0) if requires_grad else 1.0 for _ in range(size)]
        return Tensor(data, shape=shape, requires_grad=requires_grad)

    @staticmethod
    def randn(shape, requires_grad=False):
        """Random normal (Box-Muller)."""
        size = 1
        for d in shape:
            size *= d
        data = []
        for _ in range(size):
            u1 = random.random()
            u2 = random.random()
            while u1 == 0:
                u1 = random.random()
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            data.append(Var(z) if requires_grad else z)
        return Tensor(data, shape=shape, requires_grad=requires_grad)


# ---------------------------------------------------------------------------
# Parameter: Trainable weight
# ---------------------------------------------------------------------------

class Parameter:
    """A trainable parameter wrapping a Tensor of Var nodes."""

    def __init__(self, tensor: Tensor):
        self.tensor = tensor
        if not tensor.requires_grad:
            tensor.to_vars()
        self._grads = None

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def data(self):
        return self.tensor.data

    def zero_grad(self):
        """Reset gradients."""
        self._grads = None
        for v in self.tensor.data:
            if isinstance(v, Var):
                v.grad = 0.0

    def collect_grads(self):
        """Collect gradients from Var nodes."""
        self._grads = [v.grad if isinstance(v, Var) else 0.0 for v in self.tensor.data]

    def update(self, new_values: list):
        """Replace Var nodes with new values (for optimizer step)."""
        for i, val in enumerate(new_values):
            self.tensor.data[i] = Var(float(val))

    def values(self) -> list:
        """Get current float values."""
        return [v.val if isinstance(v, Var) else float(v) for v in self.tensor.data]

    def grads(self) -> list:
        """Get collected gradients."""
        if self._grads is not None:
            return self._grads
        return [v.grad if isinstance(v, Var) else 0.0 for v in self.tensor.data]

    def __repr__(self):
        return f"Parameter(shape={self.shape})"


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class Init:
    """Weight initialization strategies."""

    @staticmethod
    def xavier_uniform(shape: tuple) -> Parameter:
        """Xavier/Glorot uniform initialization."""
        fan_in = shape[-1] if len(shape) > 1 else shape[0]
        fan_out = shape[0]
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        data = [Var(random.uniform(-limit, limit)) for _ in range(_prod(shape))]
        return Parameter(Tensor(data, shape=shape, requires_grad=True))

    @staticmethod
    def xavier_normal(shape: tuple) -> Parameter:
        fan_in = shape[-1] if len(shape) > 1 else shape[0]
        fan_out = shape[0]
        std = math.sqrt(2.0 / (fan_in + fan_out))
        data = [Var(_randn() * std) for _ in range(_prod(shape))]
        return Parameter(Tensor(data, shape=shape, requires_grad=True))

    @staticmethod
    def he_uniform(shape: tuple) -> Parameter:
        """He/Kaiming uniform initialization."""
        fan_in = shape[-1] if len(shape) > 1 else shape[0]
        limit = math.sqrt(6.0 / fan_in)
        data = [Var(random.uniform(-limit, limit)) for _ in range(_prod(shape))]
        return Parameter(Tensor(data, shape=shape, requires_grad=True))

    @staticmethod
    def he_normal(shape: tuple) -> Parameter:
        fan_in = shape[-1] if len(shape) > 1 else shape[0]
        std = math.sqrt(2.0 / fan_in)
        data = [Var(_randn() * std) for _ in range(_prod(shape))]
        return Parameter(Tensor(data, shape=shape, requires_grad=True))

    @staticmethod
    def zeros(shape: tuple) -> Parameter:
        data = [Var(0.0) for _ in range(_prod(shape))]
        return Parameter(Tensor(data, shape=shape, requires_grad=True))

    @staticmethod
    def ones(shape: tuple) -> Parameter:
        data = [Var(1.0) for _ in range(_prod(shape))]
        return Parameter(Tensor(data, shape=shape, requires_grad=True))

    @staticmethod
    def constant(shape: tuple, value: float) -> Parameter:
        data = [Var(value) for _ in range(_prod(shape))]
        return Parameter(Tensor(data, shape=shape, requires_grad=True))

    @staticmethod
    def uniform(shape: tuple, low: float = -0.1, high: float = 0.1) -> Parameter:
        data = [Var(random.uniform(low, high)) for _ in range(_prod(shape))]
        return Parameter(Tensor(data, shape=shape, requires_grad=True))


def _prod(shape):
    r = 1
    for d in shape:
        r *= d
    return r


def _randn():
    u1 = random.random()
    u2 = random.random()
    while u1 == 0:
        u1 = random.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


# ---------------------------------------------------------------------------
# Module: Base class for layers
# ---------------------------------------------------------------------------

class Module:
    """Base class for all neural network modules."""

    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self._training = True

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self) -> List[Parameter]:
        """Collect all parameters recursively."""
        params = list(self._parameters.values())
        for mod in self._modules.values():
            params.extend(mod.parameters())
        return params

    def named_parameters(self) -> List[Tuple[str, Parameter]]:
        result = [(k, v) for k, v in self._parameters.items()]
        for name, mod in self._modules.items():
            for k, v in mod.named_parameters():
                result.append((f"{name}.{k}", v))
        return result

    def register_parameter(self, name: str, param: Parameter):
        self._parameters[name] = param

    def register_module(self, name: str, module: 'Module'):
        self._modules[name] = module

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self):
        self._training = True
        for mod in self._modules.values():
            mod.train()

    def eval(self):
        self._training = False
        for mod in self._modules.values():
            mod.eval()

    @property
    def training(self):
        return self._training

    def num_parameters(self) -> int:
        return sum(_prod(p.shape) for p in self.parameters())

    def state_dict(self) -> Dict[str, list]:
        """Export parameter values."""
        return {name: param.values() for name, param in self.named_parameters()}

    def load_state_dict(self, state: Dict[str, list]):
        """Load parameter values."""
        named = dict(self.named_parameters())
        for name, values in state.items():
            if name in named:
                named[name].update(values)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class Linear(Module):
    """Fully connected layer: y = xW^T + b."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 init: str = 'xavier'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if init == 'xavier':
            self.register_parameter('weight', Init.xavier_uniform((out_features, in_features)))
        elif init == 'he':
            self.register_parameter('weight', Init.he_uniform((out_features, in_features)))
        else:
            self.register_parameter('weight', Init.uniform((out_features, in_features)))

        if bias:
            self.register_parameter('bias', Init.zeros((out_features,)))
            self.has_bias = True
        else:
            self.has_bias = False

    def forward(self, x: list) -> list:
        """x: list of Var (length in_features) -> list of Var (length out_features)."""
        w = self._parameters['weight']
        out = []
        for i in range(self.out_features):
            s = w.data[i * self.in_features] * x[0]
            for j in range(1, self.in_features):
                s = s + w.data[i * self.in_features + j] * x[j]
            if self.has_bias:
                s = s + self._parameters['bias'].data[i]
            out.append(s)
        return out


class Conv1D(Module):
    """1D convolution layer."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Weight: (out_channels, in_channels, kernel_size)
        shape = (out_channels, in_channels * kernel_size)
        self.register_parameter('weight', Init.he_uniform(shape))
        self.register_parameter('bias', Init.zeros((out_channels,)))

    def forward(self, x: list) -> list:
        """x: list of lists (in_channels x seq_len) -> list of lists (out_channels x out_len).

        Each channel is a list of Var.
        """
        seq_len = len(x[0]) if isinstance(x[0], list) else len(x) // self.in_channels
        # Pad
        padded = []
        for c in range(self.in_channels):
            ch = x[c] if isinstance(x[0], list) else x[c * seq_len:(c + 1) * seq_len]
            padded_ch = [Var(0.0)] * self.padding + list(ch) + [Var(0.0)] * self.padding
            padded.append(padded_ch)

        padded_len = seq_len + 2 * self.padding
        out_len = (padded_len - self.kernel_size) // self.stride + 1
        w = self._parameters['weight']
        b = self._parameters['bias']

        output = []
        for oc in range(self.out_channels):
            ch_out = []
            for pos in range(out_len):
                start = pos * self.stride
                s = b.data[oc]
                for ic in range(self.in_channels):
                    for k in range(self.kernel_size):
                        w_idx = oc * (self.in_channels * self.kernel_size) + ic * self.kernel_size + k
                        s = s + w.data[w_idx] * padded[ic][start + k]
                ch_out.append(s)
            output.append(ch_out)
        return output


class RNNCell(Module):
    """Simple RNN cell: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.register_parameter('W_ih', Init.xavier_uniform((hidden_size, input_size)))
        self.register_parameter('W_hh', Init.xavier_uniform((hidden_size, hidden_size)))
        self.register_parameter('bias', Init.zeros((hidden_size,)))

    def forward(self, x: list, h: list = None) -> list:
        """x: list[Var] (input_size), h: list[Var] (hidden_size) -> h_new."""
        if h is None:
            h = [Var(0.0) for _ in range(self.hidden_size)]
        w_ih = self._parameters['W_ih']
        w_hh = self._parameters['W_hh']
        bias = self._parameters['bias']

        h_new = []
        for i in range(self.hidden_size):
            s = bias.data[i]
            for j in range(self.input_size):
                s = s + w_ih.data[i * self.input_size + j] * x[j]
            for j in range(self.hidden_size):
                s = s + w_hh.data[i * self.hidden_size + j] * h[j]
            h_new.append(var_tanh(s))
        return h_new


class LSTMCell(Module):
    """LSTM cell with forget, input, output gates."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Combined weights for all 4 gates: (4*hidden, input+hidden)
        gate_size = 4 * hidden_size
        combined_input = input_size + hidden_size
        self.register_parameter('weight', Init.xavier_uniform((gate_size, combined_input)))
        self.register_parameter('bias', Init.zeros((gate_size,)))
        # Initialize forget gate bias to 1
        for i in range(hidden_size):
            self._parameters['bias'].tensor.data[hidden_size + i] = Var(1.0)

    def forward(self, x: list, state: tuple = None) -> tuple:
        """x: list[Var], state: (h, c) -> (h_new, c_new)."""
        hs = self.hidden_size
        if state is None:
            h = [Var(0.0) for _ in range(hs)]
            c = [Var(0.0) for _ in range(hs)]
        else:
            h, c = state

        combined = list(x) + list(h)
        w = self._parameters['weight']
        b = self._parameters['bias']
        combined_size = self.input_size + hs

        # Compute all gates at once
        gates = []
        for i in range(4 * hs):
            s = b.data[i]
            for j in range(combined_size):
                s = s + w.data[i * combined_size + j] * combined[j]
            gates.append(s)

        # Split into i, f, g, o
        i_gate = [var_sigmoid(gates[k]) for k in range(hs)]
        f_gate = [var_sigmoid(gates[hs + k]) for k in range(hs)]
        g_gate = [var_tanh(gates[2 * hs + k]) for k in range(hs)]
        o_gate = [var_sigmoid(gates[3 * hs + k]) for k in range(hs)]

        # Cell state and hidden state
        c_new = [f_gate[k] * c[k] + i_gate[k] * g_gate[k] for k in range(hs)]
        h_new = [o_gate[k] * var_tanh(c_new[k]) for k in range(hs)]

        return h_new, c_new


class Embedding(Module):
    """Lookup table for discrete tokens."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.register_parameter('weight', Init.uniform(
            (num_embeddings, embedding_dim), -0.1, 0.1))

    def forward(self, indices: list) -> list:
        """indices: list of int -> list of list[Var] (each embedding_dim)."""
        w = self._parameters['weight']
        result = []
        for idx in indices:
            start = idx * self.embedding_dim
            result.append([w.data[start + j] for j in range(self.embedding_dim)])
        return result


class BatchNorm1D(Module):
    """Batch normalization for 1D inputs."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.register_parameter('gamma', Init.ones((num_features,)))
        self.register_parameter('beta', Init.zeros((num_features,)))
        self.running_mean = [0.0] * num_features
        self.running_var = [1.0] * num_features

    def forward(self, batch: list) -> list:
        """batch: list of list[Var] (batch_size x num_features)."""
        n = len(batch)
        gamma = self._parameters['gamma']
        beta = self._parameters['beta']

        if self._training and n > 1:
            # Compute batch statistics
            means = []
            variances = []
            for f in range(self.num_features):
                vals = [batch[i][f] for i in range(n)]
                mean = vals[0]
                for v in vals[1:]:
                    mean = mean + v
                mean = mean / n

                var = Var(0.0)
                for v in vals:
                    diff = v - mean
                    var = var + diff * diff
                var = var / n

                means.append(mean)
                variances.append(var)

                # Update running stats
                m_val = mean.val if isinstance(mean, Var) else float(mean)
                v_val = var.val if isinstance(var, Var) else float(var)
                self.running_mean[f] = (1 - self.momentum) * self.running_mean[f] + self.momentum * m_val
                self.running_var[f] = (1 - self.momentum) * self.running_var[f] + self.momentum * v_val

            result = []
            for i in range(n):
                row = []
                for f in range(self.num_features):
                    normed = (batch[i][f] - means[f]) / var_sqrt(variances[f] + self.eps)
                    row.append(gamma.data[f] * normed + beta.data[f])
                result.append(row)
            return result
        else:
            # Use running stats in eval mode
            result = []
            for i in range(n):
                row = []
                for f in range(self.num_features):
                    normed = (batch[i][f] - self.running_mean[f]) / math.sqrt(self.running_var[f] + self.eps)
                    row.append(gamma.data[f] * normed + beta.data[f])
                result.append(row)
            return result


class Dropout(Module):
    """Dropout regularization."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: list) -> list:
        if not self._training or self.p == 0:
            return x
        scale = 1.0 / (1.0 - self.p)
        return [v * scale if random.random() > self.p else Var(0.0) for v in x]


# ---------------------------------------------------------------------------
# Activation Functions (as Modules)
# ---------------------------------------------------------------------------

class ReLU(Module):
    def forward(self, x: list) -> list:
        return [var_relu(v) for v in x]

class Sigmoid(Module):
    def forward(self, x: list) -> list:
        return [var_sigmoid(v) for v in x]

class Tanh(Module):
    def forward(self, x: list) -> list:
        return [var_tanh(v) for v in x]

class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: list) -> list:
        result = []
        for v in x:
            val = v.val if isinstance(v, Var) else float(v)
            if val >= 0:
                result.append(v)
            else:
                result.append(v * self.negative_slope)
        return result

class ELU(Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: list) -> list:
        result = []
        for v in x:
            val = v.val if isinstance(v, Var) else float(v)
            if val >= 0:
                result.append(v)
            else:
                result.append(self.alpha * (var_exp(v) - 1.0))
        return result

class GELU(Module):
    """Gaussian Error Linear Unit (approximate)."""
    def forward(self, x: list) -> list:
        # GELU(x) ~ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        coeff = math.sqrt(2.0 / math.pi)
        result = []
        for v in x:
            inner = coeff * (v + 0.044715 * v * v * v)
            result.append(v * 0.5 * (1.0 + var_tanh(inner)))
        return result

class Softmax(Module):
    def forward(self, x: list) -> list:
        max_val = max(v.val if isinstance(v, Var) else float(v) for v in x)
        exps = [var_exp(v - max_val) for v in x]
        total = exps[0]
        for e in exps[1:]:
            total = total + e
        return [e / total for e in exps]


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class Loss(Module):
    """Base class for loss functions."""
    pass

class MSELoss(Loss):
    def forward(self, predicted: list, target: list) -> Var:
        n = len(predicted)
        loss = Var(0.0)
        for p, t in zip(predicted, target):
            t_val = t if isinstance(t, Var) else Var(float(t)) if isinstance(t, (int, float)) else t
            diff = p - t_val
            loss = loss + diff * diff
        return loss / n

class L1Loss(Loss):
    def forward(self, predicted: list, target: list) -> Var:
        n = len(predicted)
        loss = Var(0.0)
        for p, t in zip(predicted, target):
            t_val = float(t) if not isinstance(t, Var) else t.val
            diff = p - t_val
            val = diff.val if isinstance(diff, Var) else float(diff)
            if val >= 0:
                loss = loss + diff
            else:
                loss = loss - diff
        return loss / n

class CrossEntropyLoss(Loss):
    """Cross-entropy with built-in softmax."""
    def forward(self, logits: list, target_index: int) -> Var:
        # Softmax
        max_val = max(v.val if isinstance(v, Var) else float(v) for v in logits)
        exps = [var_exp(v - max_val) for v in logits]
        total = exps[0]
        for e in exps[1:]:
            total = total + e
        log_sum = var_log(total)
        # -log(softmax[target])
        return -(logits[target_index] - max_val - log_sum)

class BinaryCrossEntropyLoss(Loss):
    def forward(self, predicted: list, target: list) -> Var:
        n = len(predicted)
        loss = Var(0.0)
        eps = 1e-10
        for p, t in zip(predicted, target):
            t_val = float(t) if not isinstance(t, Var) else t.val
            loss = loss - (t_val * var_log(p + eps) + (1 - t_val) * var_log(1.0 - p + eps))
        return loss / n

class HuberLoss(Loss):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, predicted: list, target: list) -> Var:
        n = len(predicted)
        loss = Var(0.0)
        for p, t in zip(predicted, target):
            t_val = float(t) if not isinstance(t, Var) else t.val
            diff = p - t_val
            val = diff.val if isinstance(diff, Var) else float(diff)
            if abs(val) <= self.delta:
                loss = loss + 0.5 * diff * diff
            else:
                if val > 0:
                    loss = loss + self.delta * (diff - 0.5 * self.delta)
                else:
                    loss = loss + self.delta * (Var(0.0) - diff - 0.5 * self.delta)
        return loss / n


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

class Optimizer:
    """Base optimizer."""
    def __init__(self, parameters: List[Parameter], lr: float = 0.01):
        self.parameters = parameters
        self.lr = lr
        self.step_count = 0

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

    def step(self):
        raise NotImplementedError

    def collect_grads(self):
        for p in self.parameters:
            p.collect_grads()


class SGD(Optimizer):
    """SGD with optional momentum and weight decay."""

    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [None] * len(parameters)

    def step(self):
        self.collect_grads()
        self.step_count += 1
        for i, p in enumerate(self.parameters):
            grads = p.grads()
            vals = p.values()
            if self.velocities[i] is None:
                self.velocities[i] = [0.0] * len(vals)

            new_vals = []
            for j in range(len(vals)):
                g = grads[j] + self.weight_decay * vals[j]
                v = self.momentum * self.velocities[i][j] + g
                self.velocities[i][j] = v
                new_vals.append(vals[j] - self.lr * v)
            p.update(new_vals)


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [None] * len(parameters)
        self.v = [None] * len(parameters)

    def step(self):
        self.collect_grads()
        self.step_count += 1
        for i, p in enumerate(self.parameters):
            grads = p.grads()
            vals = p.values()
            n = len(vals)

            if self.m[i] is None:
                self.m[i] = [0.0] * n
                self.v[i] = [0.0] * n

            new_vals = []
            for j in range(n):
                g = grads[j] + self.weight_decay * vals[j]
                self.m[i][j] = self.beta1 * self.m[i][j] + (1 - self.beta1) * g
                self.v[i][j] = self.beta2 * self.v[i][j] + (1 - self.beta2) * g * g

                m_hat = self.m[i][j] / (1 - self.beta1 ** self.step_count)
                v_hat = self.v[i][j] / (1 - self.beta2 ** self.step_count)

                new_vals.append(vals[j] - self.lr * m_hat / (math.sqrt(v_hat) + self.eps))
            p.update(new_vals)


class RMSProp(Optimizer):
    """RMSProp optimizer."""

    def __init__(self, parameters, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.v = [None] * len(parameters)

    def step(self):
        self.collect_grads()
        self.step_count += 1
        for i, p in enumerate(self.parameters):
            grads = p.grads()
            vals = p.values()
            n = len(vals)

            if self.v[i] is None:
                self.v[i] = [0.0] * n

            new_vals = []
            for j in range(n):
                g = grads[j] + self.weight_decay * vals[j]
                self.v[i][j] = self.alpha * self.v[i][j] + (1 - self.alpha) * g * g
                new_vals.append(vals[j] - self.lr * g / (math.sqrt(self.v[i][j]) + self.eps))
            p.update(new_vals)


class AdaGrad(Optimizer):
    """AdaGrad optimizer."""

    def __init__(self, parameters, lr=0.01, eps=1e-8):
        super().__init__(parameters, lr)
        self.eps = eps
        self.v = [None] * len(parameters)

    def step(self):
        self.collect_grads()
        self.step_count += 1
        for i, p in enumerate(self.parameters):
            grads = p.grads()
            vals = p.values()
            n = len(vals)

            if self.v[i] is None:
                self.v[i] = [0.0] * n

            new_vals = []
            for j in range(n):
                g = grads[j]
                self.v[i][j] += g * g
                new_vals.append(vals[j] - self.lr * g / (math.sqrt(self.v[i][j]) + self.eps))
            p.update(new_vals)


# ---------------------------------------------------------------------------
# Learning Rate Schedulers
# ---------------------------------------------------------------------------

class LRScheduler:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.step_count = 0

    def step(self):
        self.step_count += 1
        self.optimizer.lr = self.get_lr()

    def get_lr(self) -> float:
        raise NotImplementedError


class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self):
        return self.base_lr * (self.gamma ** (self.step_count // self.step_size))


class ExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma: float = 0.95):
        super().__init__(optimizer)
        self.gamma = gamma

    def get_lr(self):
        return self.base_lr * (self.gamma ** self.step_count)


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self):
        return self.eta_min + (self.base_lr - self.eta_min) * \
            (1 + math.cos(math.pi * self.step_count / self.T_max)) / 2


# ---------------------------------------------------------------------------
# Sequential Model
# ---------------------------------------------------------------------------

class Sequential(Module):
    """Sequential container -- layers applied in order."""

    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            self.register_module(str(i), layer)
        self._layers = list(layers)

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x

    def add(self, layer: Module):
        idx = len(self._layers)
        self.register_module(str(idx), layer)
        self._layers.append(layer)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class TrainHistory:
    """Training history."""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    lrs: List[float] = field(default_factory=list)
    epochs: int = 0


class Trainer:
    """Training loop with batching, validation, and callbacks."""

    def __init__(self, model: Module, optimizer: Optimizer, loss_fn: Loss,
                 scheduler: LRScheduler = None, grad_clip: float = 0.0,
                 l1_reg: float = 0.0, l2_reg: float = 0.0):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.history = TrainHistory()
        self.callbacks = []

    def add_callback(self, callback: Callable):
        self.callbacks.append(callback)

    def _fire_callback(self, event: str, **kwargs):
        for cb in self.callbacks:
            cb(event, **kwargs)

    def _compute_loss(self, outputs, targets) -> Var:
        """Compute loss with optional regularization."""
        loss = self.loss_fn(outputs, targets)

        if self.l1_reg > 0 or self.l2_reg > 0:
            for p in self.model.parameters():
                for v in p.data:
                    if isinstance(v, Var):
                        if self.l2_reg > 0:
                            loss = loss + self.l2_reg * v * v
                        if self.l1_reg > 0:
                            val = v.val
                            if val > 0:
                                loss = loss + self.l1_reg * v
                            elif val < 0:
                                loss = loss - self.l1_reg * v
        return loss

    def _clip_grads(self):
        """Gradient clipping by global norm."""
        if self.grad_clip <= 0:
            return
        total_norm_sq = 0.0
        for p in self.model.parameters():
            for v in p.data:
                if isinstance(v, Var):
                    total_norm_sq += v.grad * v.grad
        total_norm = math.sqrt(total_norm_sq)
        if total_norm > self.grad_clip:
            scale = self.grad_clip / total_norm
            for p in self.model.parameters():
                for v in p.data:
                    if isinstance(v, Var):
                        v.grad *= scale

    def train_step(self, x, y) -> float:
        """Single training step. Returns loss value."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward
        output = self.model(x)
        loss = self._compute_loss(output, y)

        # Backward
        loss.backward()

        # Clip gradients
        self._clip_grads()

        # Optimizer step
        self.optimizer.step()

        return loss.val

    def train_epoch(self, data: list, batch_size: int = None) -> float:
        """Train one epoch. data: list of (x, y) pairs. Returns avg loss."""
        if batch_size is None:
            batch_size = len(data)

        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_loss = 0.0
            for x, y in batch:
                batch_loss += self.train_step(x, y)
            total_loss += batch_loss / len(batch)
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0.0

    def evaluate(self, data: list) -> float:
        """Evaluate on validation data. Returns avg loss."""
        self.model.eval()
        total_loss = 0.0
        for x, y in data:
            output = self.model(x)
            loss = self.loss_fn(output, y)
            total_loss += loss.val if isinstance(loss, Var) else float(loss)
        self.model.train()
        return total_loss / len(data) if data else 0.0

    def fit(self, train_data: list, epochs: int, val_data: list = None,
            batch_size: int = None, verbose: bool = False) -> TrainHistory:
        """Full training loop."""
        self._fire_callback('train_start', epochs=epochs)

        for epoch in range(epochs):
            self._fire_callback('epoch_start', epoch=epoch)

            # Shuffle
            shuffled = list(train_data)
            random.shuffle(shuffled)

            train_loss = self.train_epoch(shuffled, batch_size)
            self.history.train_losses.append(train_loss)
            self.history.lrs.append(self.optimizer.lr)

            val_loss = None
            if val_data:
                val_loss = self.evaluate(val_data)
                self.history.val_losses.append(val_loss)

            if self.scheduler:
                self.scheduler.step()

            self.history.epochs += 1
            self._fire_callback('epoch_end', epoch=epoch,
                                train_loss=train_loss, val_loss=val_loss)

        self._fire_callback('train_end', history=self.history)
        return self.history


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def num_grad(f, params: List[Parameter], eps: float = 1e-5) -> List[list]:
    """Numerical gradient for gradient checking."""
    grads = []
    for p in params:
        p_grads = []
        vals = p.values()
        for i in range(len(vals)):
            # f(x + eps)
            vals_plus = list(vals)
            vals_plus[i] += eps
            p.update(vals_plus)
            loss_plus = f()

            # f(x - eps)
            vals_minus = list(vals)
            vals_minus[i] -= eps
            p.update(vals_minus)
            loss_minus = f()

            p_grads.append((loss_plus - loss_minus) / (2 * eps))

            # Restore
            p.update(vals)
        grads.append(p_grads)
    return grads


def accuracy(model: Module, data: list) -> float:
    """Classification accuracy. data: list of (x, target_index) pairs."""
    model.eval()
    correct = 0
    for x, target in data:
        output = model(x)
        vals = [v.val if isinstance(v, Var) else float(v) for v in output]
        pred = vals.index(max(vals))
        correct += (pred == target)
    model.train()
    return correct / len(data) if data else 0.0
