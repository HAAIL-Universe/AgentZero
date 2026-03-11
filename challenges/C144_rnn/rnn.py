"""
C144: Recurrent Neural Network
Sequence modeling with RNN, LSTM, GRU cells and bidirectional wrappers.

Extends:
- C140 Neural Network (Tensor, Dense, activations, optimizers, Sequential)

Features:
- RNNCell: vanilla recurrent cell (tanh activation)
- LSTMCell: Long Short-Term Memory (forget/input/output gates)
- GRUCell: Gated Recurrent Unit (reset/update gates)
- RNN layer: unrolled sequence processing with RNNCell
- LSTM layer: unrolled sequence processing with LSTMCell
- GRU layer: unrolled sequence processing with GRUCell
- Bidirectional wrapper: forward + backward pass concatenation
- TimeDistributed wrapper: apply layer to each timestep
- Embedding layer: integer-to-vector lookup
- Sequence utilities: pad_sequences, one_hot_encode_sequence
- Many-to-one and many-to-many modes
- Gradient clipping for recurrent gradients
- Sequence-to-sequence model building
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))
from neural_network import (
    Tensor, Dense, Layer, Activation, Dropout, BatchNorm,
    Sequential, SGD, Adam, RMSProp, MSELoss, CrossEntropyLoss,
    xavier_init, he_init, softmax, softmax_batch,
    relu, relu_deriv, sigmoid, sigmoid_deriv, tanh_act, tanh_deriv,
    fit, evaluate, predict_classes, accuracy, one_hot
)


# ============================================================
# Utility functions
# ============================================================

def _mat_mul(A, B):
    """Multiply 2D lists A (m x n) and B (n x p) -> (m x p)."""
    m, n = len(A), len(A[0])
    p = len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)] for i in range(m)]


def _mat_vec(A, v):
    """Multiply 2D list A (m x n) by 1D list v (n) -> 1D list (m)."""
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def _vec_outer(u, v):
    """Outer product of two 1D lists -> 2D list."""
    return [[u[i] * v[j] for j in range(len(v))] for i in range(len(u))]


def _vec_add(a, b):
    """Element-wise add two 1D lists."""
    return [a[i] + b[i] for i in range(len(a))]


def _vec_mul(a, b):
    """Element-wise multiply two 1D lists."""
    return [a[i] * b[i] for i in range(len(a))]


def _vec_sub(a, b):
    """Element-wise subtract."""
    return [a[i] - b[i] for i in range(len(a))]


def _vec_scale(a, s):
    """Scale a vector by scalar."""
    return [x * s for x in a]


def _sigmoid_vec(v):
    """Apply sigmoid element-wise to a list."""
    return [1.0 / (1.0 + math.exp(-max(-500, min(500, x)))) for x in v]


def _tanh_vec(v):
    """Apply tanh element-wise to a list."""
    return [math.tanh(x) for x in v]


def _relu_vec(v):
    """Apply ReLU element-wise to a list."""
    return [max(0.0, x) for x in v]


def _zeros(n):
    """Create zero vector of length n."""
    return [0.0] * n


def _zeros_2d(rows, cols):
    """Create zero matrix."""
    return [[0.0] * cols for _ in range(rows)]


def _randn(rng=None):
    """Sample from standard normal."""
    r = rng if rng else random
    u1 = max(r.random(), 1e-10)
    u2 = r.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _init_weight(rows, cols, fan_in, fan_out, rng=None):
    """Xavier-initialized weight matrix."""
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return [[_randn(rng) * std for _ in range(cols)] for _ in range(rows)]


def _clip_vec(v, max_norm):
    """Clip a vector by max norm."""
    norm = math.sqrt(sum(x * x for x in v))
    if norm > max_norm:
        scale = max_norm / norm
        return [x * scale for x in v]
    return v


def _clip_mat(m, max_norm):
    """Clip a matrix by max Frobenius norm."""
    norm = math.sqrt(sum(x * x for row in m for x in row))
    if norm > max_norm:
        scale = max_norm / norm
        return [[x * scale for x in row] for row in m]
    return m


def _transpose(m):
    """Transpose a 2D list."""
    if not m:
        return m
    rows, cols = len(m), len(m[0])
    return [[m[i][j] for i in range(rows)] for j in range(cols)]


# ============================================================
# RNN Cell
# ============================================================

class RNNCell:
    """Vanilla RNN cell: h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)."""

    def __init__(self, input_size, hidden_size, rng=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = _init_weight(hidden_size, input_size, input_size, hidden_size, rng)
        self.W_hh = _init_weight(hidden_size, hidden_size, hidden_size, hidden_size, rng)
        self.b_h = _zeros(hidden_size)

    def forward(self, x, h_prev):
        """Forward pass. Returns h_new and cache for backprop."""
        # h = tanh(W_ih @ x + W_hh @ h_prev + b)
        ih = _mat_vec(self.W_ih, x)
        hh = _mat_vec(self.W_hh, h_prev)
        pre_act = _vec_add(_vec_add(ih, hh), self.b_h)
        h_new = _tanh_vec(pre_act)
        cache = (x, h_prev, pre_act, h_new)
        return h_new, cache

    def backward(self, dh, cache):
        """Backward pass. Returns dx, dh_prev, and gradients."""
        x, h_prev, pre_act, h_new = cache
        # d_pre = dh * (1 - tanh^2)
        d_pre = [dh[i] * (1.0 - h_new[i] ** 2) for i in range(self.hidden_size)]

        dW_ih = _vec_outer(d_pre, x)
        dW_hh = _vec_outer(d_pre, h_prev)
        db_h = d_pre[:]

        dx = _mat_vec(_transpose(self.W_ih), d_pre)
        dh_prev = _mat_vec(_transpose(self.W_hh), d_pre)

        return dx, dh_prev, {'dW_ih': dW_ih, 'dW_hh': dW_hh, 'db_h': db_h}

    def params(self):
        return {'W_ih': self.W_ih, 'W_hh': self.W_hh, 'b_h': self.b_h}

    def param_count(self):
        return (self.input_size * self.hidden_size +
                self.hidden_size * self.hidden_size +
                self.hidden_size)


# ============================================================
# LSTM Cell
# ============================================================

class LSTMCell:
    """LSTM cell with forget, input, output gates and cell state.

    f_t = sigmoid(W_if @ x + W_hf @ h + b_f)   -- forget gate
    i_t = sigmoid(W_ii @ x + W_hi @ h + b_i)   -- input gate
    g_t = tanh(W_ig @ x + W_hg @ h + b_g)      -- candidate
    o_t = sigmoid(W_io @ x + W_ho @ h + b_o)    -- output gate
    c_t = f_t * c_{t-1} + i_t * g_t
    h_t = o_t * tanh(c_t)
    """

    def __init__(self, input_size, hidden_size, rng=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        hs = hidden_size

        # Gates: forget, input, candidate, output
        self.W_if = _init_weight(hs, input_size, input_size, hs, rng)
        self.W_hf = _init_weight(hs, hs, hs, hs, rng)
        self.b_f = [1.0] * hs  # Bias forget gate to 1 (helps learning)

        self.W_ii = _init_weight(hs, input_size, input_size, hs, rng)
        self.W_hi = _init_weight(hs, hs, hs, hs, rng)
        self.b_i = _zeros(hs)

        self.W_ig = _init_weight(hs, input_size, input_size, hs, rng)
        self.W_hg = _init_weight(hs, hs, hs, hs, rng)
        self.b_g = _zeros(hs)

        self.W_io = _init_weight(hs, input_size, input_size, hs, rng)
        self.W_ho = _init_weight(hs, hs, hs, hs, rng)
        self.b_o = _zeros(hs)

    def forward(self, x, h_prev, c_prev):
        """Forward pass. Returns (h_new, c_new, cache)."""
        hs = self.hidden_size

        # Forget gate
        f_pre = _vec_add(_vec_add(_mat_vec(self.W_if, x), _mat_vec(self.W_hf, h_prev)), self.b_f)
        f = _sigmoid_vec(f_pre)

        # Input gate
        i_pre = _vec_add(_vec_add(_mat_vec(self.W_ii, x), _mat_vec(self.W_hi, h_prev)), self.b_i)
        i = _sigmoid_vec(i_pre)

        # Candidate
        g_pre = _vec_add(_vec_add(_mat_vec(self.W_ig, x), _mat_vec(self.W_hg, h_prev)), self.b_g)
        g = _tanh_vec(g_pre)

        # Output gate
        o_pre = _vec_add(_vec_add(_mat_vec(self.W_io, x), _mat_vec(self.W_ho, h_prev)), self.b_o)
        o = _sigmoid_vec(o_pre)

        # Cell state and hidden state
        c_new = _vec_add(_vec_mul(f, c_prev), _vec_mul(i, g))
        tanh_c = _tanh_vec(c_new)
        h_new = _vec_mul(o, tanh_c)

        cache = (x, h_prev, c_prev, f, i, g, o, c_new, tanh_c, f_pre, i_pre, g_pre, o_pre)
        return h_new, c_new, cache

    def backward(self, dh, dc, cache):
        """Backward pass through LSTM cell."""
        x, h_prev, c_prev, f, i, g, o, c_new, tanh_c, f_pre, i_pre, g_pre, o_pre = cache
        hs = self.hidden_size

        # dh into output gate
        do_act = _vec_mul(dh, tanh_c)
        d_tanh_c = _vec_mul(dh, o)

        # dc from both paths
        dc_total = _vec_add(dc, [d_tanh_c[j] * (1.0 - tanh_c[j] ** 2) for j in range(hs)])

        # Gate gradients
        df = _vec_mul(dc_total, c_prev)
        di = _vec_mul(dc_total, g)
        dg = _vec_mul(dc_total, i)
        dc_prev = _vec_mul(dc_total, f)

        # Through sigmoid/tanh
        df_pre = [df[j] * f[j] * (1.0 - f[j]) for j in range(hs)]
        di_pre = [di[j] * i[j] * (1.0 - i[j]) for j in range(hs)]
        dg_pre = [dg[j] * (1.0 - g[j] ** 2) for j in range(hs)]
        do_pre = [do_act[j] * o[j] * (1.0 - o[j]) for j in range(hs)]

        # Parameter gradients
        grads = {}
        for gate_name, d_pre in [('f', df_pre), ('i', di_pre), ('g', dg_pre), ('o', do_pre)]:
            W_i_key = f'dW_i{gate_name}'
            W_h_key = f'dW_h{gate_name}'
            b_key = f'db_{gate_name}'
            grads[W_i_key] = _vec_outer(d_pre, x)
            grads[W_h_key] = _vec_outer(d_pre, h_prev)
            grads[b_key] = d_pre[:]

        # Input and hidden gradients
        dx = _zeros(len(x))
        dh_prev = _zeros(hs)
        for gate_name, d_pre in [('f', df_pre), ('i', di_pre), ('g', dg_pre), ('o', do_pre)]:
            W_i = getattr(self, f'W_i{gate_name}')
            W_h = getattr(self, f'W_h{gate_name}')
            dx = _vec_add(dx, _mat_vec(_transpose(W_i), d_pre))
            dh_prev = _vec_add(dh_prev, _mat_vec(_transpose(W_h), d_pre))

        return dx, dh_prev, dc_prev, grads

    def params(self):
        result = {}
        for gate in ['f', 'i', 'g', 'o']:
            result[f'W_i{gate}'] = getattr(self, f'W_i{gate}')
            result[f'W_h{gate}'] = getattr(self, f'W_h{gate}')
            result[f'b_{gate}'] = getattr(self, f'b_{gate}')
        return result

    def param_count(self):
        return 4 * (self.input_size * self.hidden_size +
                     self.hidden_size * self.hidden_size +
                     self.hidden_size)


# ============================================================
# GRU Cell
# ============================================================

class GRUCell:
    """GRU cell with reset and update gates.

    z_t = sigmoid(W_iz @ x + W_hz @ h + b_z)   -- update gate
    r_t = sigmoid(W_ir @ x + W_hr @ h + b_r)   -- reset gate
    n_t = tanh(W_in @ x + r_t * (W_hn @ h) + b_n)  -- candidate
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    """

    def __init__(self, input_size, hidden_size, rng=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        hs = hidden_size

        # Update gate
        self.W_iz = _init_weight(hs, input_size, input_size, hs, rng)
        self.W_hz = _init_weight(hs, hs, hs, hs, rng)
        self.b_z = _zeros(hs)

        # Reset gate
        self.W_ir = _init_weight(hs, input_size, input_size, hs, rng)
        self.W_hr = _init_weight(hs, hs, hs, hs, rng)
        self.b_r = _zeros(hs)

        # Candidate
        self.W_in = _init_weight(hs, input_size, input_size, hs, rng)
        self.W_hn = _init_weight(hs, hs, hs, hs, rng)
        self.b_n = _zeros(hs)

    def forward(self, x, h_prev):
        """Forward pass. Returns (h_new, cache)."""
        hs = self.hidden_size

        # Update gate
        z_pre = _vec_add(_vec_add(_mat_vec(self.W_iz, x), _mat_vec(self.W_hz, h_prev)), self.b_z)
        z = _sigmoid_vec(z_pre)

        # Reset gate
        r_pre = _vec_add(_vec_add(_mat_vec(self.W_ir, x), _mat_vec(self.W_hr, h_prev)), self.b_r)
        r = _sigmoid_vec(r_pre)

        # Candidate
        r_h = _vec_mul(r, h_prev)
        n_pre = _vec_add(_vec_add(_mat_vec(self.W_in, x), _mat_vec(self.W_hn, r_h)), self.b_n)
        n = _tanh_vec(n_pre)

        # Output: h = (1 - z) * n + z * h_prev
        h_new = [((1.0 - z[j]) * n[j] + z[j] * h_prev[j]) for j in range(hs)]

        cache = (x, h_prev, z, r, n, r_h, z_pre, r_pre, n_pre)
        return h_new, cache

    def backward(self, dh, cache):
        """Backward pass through GRU cell."""
        x, h_prev, z, r, n, r_h, z_pre, r_pre, n_pre = cache
        hs = self.hidden_size

        # h = (1-z)*n + z*h_prev
        dz = [dh[j] * (h_prev[j] - n[j]) for j in range(hs)]
        dn = [dh[j] * (1.0 - z[j]) for j in range(hs)]
        dh_prev_direct = [dh[j] * z[j] for j in range(hs)]

        # Through tanh
        dn_pre = [dn[j] * (1.0 - n[j] ** 2) for j in range(hs)]

        # Candidate gradients
        dW_in = _vec_outer(dn_pre, x)
        dW_hn = _vec_outer(dn_pre, r_h)
        db_n = dn_pre[:]

        # Through r_h = r * h_prev
        dr_h = _mat_vec(_transpose(self.W_hn), dn_pre)
        dr = _vec_mul(dr_h, h_prev)
        dh_prev_r = _vec_mul(dr_h, r)

        # Through sigmoid for z and r
        dz_pre = [dz[j] * z[j] * (1.0 - z[j]) for j in range(hs)]
        dr_pre = [dr[j] * r[j] * (1.0 - r[j]) for j in range(hs)]

        # Update gate gradients
        dW_iz = _vec_outer(dz_pre, x)
        dW_hz = _vec_outer(dz_pre, h_prev)
        db_z = dz_pre[:]

        # Reset gate gradients
        dW_ir = _vec_outer(dr_pre, x)
        dW_hr = _vec_outer(dr_pre, h_prev)
        db_r = dr_pre[:]

        # Input gradient
        dx = _zeros(len(x))
        dx = _vec_add(dx, _mat_vec(_transpose(self.W_in), dn_pre))
        dx = _vec_add(dx, _mat_vec(_transpose(self.W_iz), dz_pre))
        dx = _vec_add(dx, _mat_vec(_transpose(self.W_ir), dr_pre))

        # Hidden gradient
        dh_prev = dh_prev_direct[:]
        dh_prev = _vec_add(dh_prev, dh_prev_r)
        dh_prev = _vec_add(dh_prev, _mat_vec(_transpose(self.W_hz), dz_pre))
        dh_prev = _vec_add(dh_prev, _mat_vec(_transpose(self.W_hr), dr_pre))

        grads = {
            'dW_iz': dW_iz, 'dW_hz': dW_hz, 'db_z': db_z,
            'dW_ir': dW_ir, 'dW_hr': dW_hr, 'db_r': db_r,
            'dW_in': dW_in, 'dW_hn': dW_hn, 'db_n': db_n,
        }
        return dx, dh_prev, grads

    def params(self):
        result = {}
        for gate in ['z', 'r', 'n']:
            result[f'W_i{gate}'] = getattr(self, f'W_i{gate}')
            result[f'W_h{gate}'] = getattr(self, f'W_h{gate}')
            result[f'b_{gate}'] = getattr(self, f'b_{gate}')
        return result

    def param_count(self):
        return 3 * (self.input_size * self.hidden_size +
                     self.hidden_size * self.hidden_size +
                     self.hidden_size)


# ============================================================
# RNN Layer (unrolled sequence processing)
# ============================================================

class RNN:
    """Unrolled RNN layer using RNNCell.

    Input: list of sequences, each sequence is list of vectors (timesteps x input_size)
    Output: depends on return_sequences:
      - True: list of all hidden states (timesteps x hidden_size)
      - False: last hidden state only (hidden_size)
    """

    def __init__(self, input_size, hidden_size, return_sequences=False,
                 clip_grad=None, rng=None):
        self.cell = RNNCell(input_size, hidden_size, rng)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.clip_grad = clip_grad
        self.caches = []
        self.training = True

    def forward(self, sequences):
        """Process a batch of sequences.

        Args:
            sequences: list of sequences, each is list of vectors (list of floats)
        Returns:
            If return_sequences: list of [h_1, h_2, ..., h_T] per sequence
            Else: list of h_T per sequence
        """
        batch_outputs = []
        self.caches = []

        for seq in sequences:
            h = _zeros(self.hidden_size)
            seq_caches = []
            hidden_states = []

            for t, x_t in enumerate(seq):
                h, cache = self.cell.forward(x_t, h)
                seq_caches.append(cache)
                hidden_states.append(h[:])

            self.caches.append(seq_caches)
            if self.return_sequences:
                batch_outputs.append(hidden_states)
            else:
                batch_outputs.append(h[:])

        return batch_outputs

    def backward(self, d_outputs):
        """Backward pass through time (BPTT).

        Args:
            d_outputs: gradients w.r.t. outputs
                If return_sequences: list of [dh_1, ..., dh_T] per sequence
                Else: list of dh_T per sequence
        Returns:
            d_inputs: gradients w.r.t. inputs (list of sequences of dx)
        """
        # Accumulate gradients
        all_grads = {}
        d_inputs = []

        for b, seq_caches in enumerate(self.caches):
            T = len(seq_caches)
            seq_dx = [None] * T

            if self.return_sequences:
                dh_list = d_outputs[b]
            else:
                dh_list = [_zeros(self.hidden_size)] * T
                dh_list[T - 1] = d_outputs[b]

            dh_next = _zeros(self.hidden_size)

            for t in reversed(range(T)):
                dh = _vec_add(dh_list[t], dh_next)
                if self.clip_grad:
                    dh = _clip_vec(dh, self.clip_grad)
                dx, dh_next, grads = self.cell.backward(dh, seq_caches[t])
                seq_dx[t] = dx

                for k, v in grads.items():
                    if k not in all_grads:
                        all_grads[k] = v if isinstance(v, list) and isinstance(v[0], list) else v[:]
                    else:
                        if isinstance(v[0], list):
                            for i in range(len(v)):
                                for j in range(len(v[0])):
                                    all_grads[k][i][j] += v[i][j]
                        else:
                            for i in range(len(v)):
                                all_grads[k][i] += v[i]

            d_inputs.append(seq_dx)

        return d_inputs, all_grads

    def update(self, grads, lr):
        """Apply gradient update with learning rate."""
        cell = self.cell
        for param_name in ['W_ih', 'W_hh']:
            grad_key = f'd{param_name}'
            if grad_key in grads:
                w = getattr(cell, param_name)
                g = grads[grad_key]
                if self.clip_grad:
                    g = _clip_mat(g, self.clip_grad)
                for i in range(len(w)):
                    for j in range(len(w[0])):
                        w[i][j] -= lr * g[i][j]
        if 'db_h' in grads:
            g = grads['db_h']
            if self.clip_grad:
                g = _clip_vec(g, self.clip_grad)
            for i in range(len(cell.b_h)):
                cell.b_h[i] -= lr * g[i]

    def param_count(self):
        return self.cell.param_count()


# ============================================================
# LSTM Layer
# ============================================================

class LSTMLayer:
    """Unrolled LSTM layer."""

    def __init__(self, input_size, hidden_size, return_sequences=False,
                 clip_grad=None, rng=None):
        self.cell = LSTMCell(input_size, hidden_size, rng)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.clip_grad = clip_grad
        self.caches = []
        self.training = True

    def forward(self, sequences):
        """Process batch of sequences through LSTM."""
        batch_outputs = []
        self.caches = []

        for seq in sequences:
            h = _zeros(self.hidden_size)
            c = _zeros(self.hidden_size)
            seq_caches = []
            hidden_states = []

            for x_t in seq:
                h, c, cache = self.cell.forward(x_t, h, c)
                seq_caches.append(cache)
                hidden_states.append(h[:])

            self.caches.append(seq_caches)
            if self.return_sequences:
                batch_outputs.append(hidden_states)
            else:
                batch_outputs.append(h[:])

        return batch_outputs

    def backward(self, d_outputs):
        """BPTT through LSTM."""
        all_grads = {}
        d_inputs = []

        for b, seq_caches in enumerate(self.caches):
            T = len(seq_caches)
            seq_dx = [None] * T

            if self.return_sequences:
                dh_list = d_outputs[b]
            else:
                dh_list = [_zeros(self.hidden_size)] * T
                dh_list[T - 1] = d_outputs[b]

            dh_next = _zeros(self.hidden_size)
            dc_next = _zeros(self.hidden_size)

            for t in reversed(range(T)):
                dh = _vec_add(dh_list[t], dh_next)
                if self.clip_grad:
                    dh = _clip_vec(dh, self.clip_grad)
                dx, dh_next, dc_next, grads = self.cell.backward(dh, dc_next, seq_caches[t])
                seq_dx[t] = dx

                for k, v in grads.items():
                    if k not in all_grads:
                        all_grads[k] = [row[:] for row in v] if isinstance(v[0], list) else v[:]
                    else:
                        if isinstance(v[0], list):
                            for i in range(len(v)):
                                for j in range(len(v[0])):
                                    all_grads[k][i][j] += v[i][j]
                        else:
                            for i in range(len(v)):
                                all_grads[k][i] += v[i]

            d_inputs.append(seq_dx)

        return d_inputs, all_grads

    def update(self, grads, lr):
        """Apply gradient update."""
        cell = self.cell
        for gate in ['f', 'i', 'g', 'o']:
            for wtype in ['W_i', 'W_h']:
                param = f'{wtype}{gate}'
                grad_key = f'd{param}'
                if grad_key in grads:
                    w = getattr(cell, param)
                    g = grads[grad_key]
                    if self.clip_grad:
                        g = _clip_mat(g, self.clip_grad)
                    for i in range(len(w)):
                        for j in range(len(w[0])):
                            w[i][j] -= lr * g[i][j]
            bias_key = f'db_{gate}'
            if bias_key in grads:
                b = getattr(cell, f'b_{gate}')
                g = grads[bias_key]
                if self.clip_grad:
                    g = _clip_vec(g, self.clip_grad)
                for i in range(len(b)):
                    b[i] -= lr * g[i]

    def param_count(self):
        return self.cell.param_count()


# ============================================================
# GRU Layer
# ============================================================

class GRULayer:
    """Unrolled GRU layer."""

    def __init__(self, input_size, hidden_size, return_sequences=False,
                 clip_grad=None, rng=None):
        self.cell = GRUCell(input_size, hidden_size, rng)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.clip_grad = clip_grad
        self.caches = []
        self.training = True

    def forward(self, sequences):
        """Process batch of sequences through GRU."""
        batch_outputs = []
        self.caches = []

        for seq in sequences:
            h = _zeros(self.hidden_size)
            seq_caches = []
            hidden_states = []

            for x_t in seq:
                h, cache = self.cell.forward(x_t, h)
                seq_caches.append(cache)
                hidden_states.append(h[:])

            self.caches.append(seq_caches)
            if self.return_sequences:
                batch_outputs.append(hidden_states)
            else:
                batch_outputs.append(h[:])

        return batch_outputs

    def backward(self, d_outputs):
        """BPTT through GRU."""
        all_grads = {}
        d_inputs = []

        for b, seq_caches in enumerate(self.caches):
            T = len(seq_caches)
            seq_dx = [None] * T

            if self.return_sequences:
                dh_list = d_outputs[b]
            else:
                dh_list = [_zeros(self.hidden_size)] * T
                dh_list[T - 1] = d_outputs[b]

            dh_next = _zeros(self.hidden_size)

            for t in reversed(range(T)):
                dh = _vec_add(dh_list[t], dh_next)
                if self.clip_grad:
                    dh = _clip_vec(dh, self.clip_grad)
                dx, dh_next, grads = self.cell.backward(dh, seq_caches[t])
                seq_dx[t] = dx

                for k, v in grads.items():
                    if k not in all_grads:
                        all_grads[k] = [row[:] for row in v] if isinstance(v[0], list) else v[:]
                    else:
                        if isinstance(v[0], list):
                            for i in range(len(v)):
                                for j in range(len(v[0])):
                                    all_grads[k][i][j] += v[i][j]
                        else:
                            for i in range(len(v)):
                                all_grads[k][i] += v[i]

            d_inputs.append(seq_dx)

        return d_inputs, all_grads

    def update(self, grads, lr):
        """Apply gradient update."""
        cell = self.cell
        for gate in ['z', 'r', 'n']:
            for wtype in ['W_i', 'W_h']:
                param = f'{wtype}{gate}'
                grad_key = f'd{param}'
                if grad_key in grads:
                    w = getattr(cell, param)
                    g = grads[grad_key]
                    if self.clip_grad:
                        g = _clip_mat(g, self.clip_grad)
                    for i in range(len(w)):
                        for j in range(len(w[0])):
                            w[i][j] -= lr * g[i][j]
            bias_key = f'db_{gate}'
            if bias_key in grads:
                b = getattr(cell, f'b_{gate}')
                g = grads[bias_key]
                if self.clip_grad:
                    g = _clip_vec(g, self.clip_grad)
                for i in range(len(b)):
                    b[i] -= lr * g[i]

    def param_count(self):
        return self.cell.param_count()


# ============================================================
# Bidirectional Wrapper
# ============================================================

class Bidirectional:
    """Bidirectional wrapper that runs forward and backward RNN and concatenates outputs.

    Output size is 2 * hidden_size.
    """

    def __init__(self, layer_cls, input_size, hidden_size, return_sequences=False,
                 clip_grad=None, rng=None):
        self.forward_layer = layer_cls(input_size, hidden_size,
                                       return_sequences=return_sequences,
                                       clip_grad=clip_grad, rng=rng)
        self.backward_layer = layer_cls(input_size, hidden_size,
                                        return_sequences=return_sequences,
                                        clip_grad=clip_grad, rng=rng)
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.training = True

    def forward(self, sequences):
        """Process sequences in both directions and concatenate."""
        # Reverse sequences for backward layer
        rev_sequences = [list(reversed(seq)) for seq in sequences]

        fwd_out = self.forward_layer.forward(sequences)
        bwd_out = self.backward_layer.forward(rev_sequences)

        batch_outputs = []
        for b in range(len(sequences)):
            if self.return_sequences:
                T = len(sequences[b])
                # Reverse backward outputs to align with forward
                bwd_reversed = list(reversed(bwd_out[b]))
                concat = [fwd_out[b][t] + bwd_reversed[t] for t in range(T)]
                batch_outputs.append(concat)
            else:
                batch_outputs.append(fwd_out[b] + bwd_out[b])

        return batch_outputs

    def backward(self, d_outputs):
        """Backward through bidirectional layer."""
        hs = self.hidden_size
        batch_size = len(d_outputs)

        # Split gradients for forward and backward
        fwd_d = []
        bwd_d = []

        for b in range(batch_size):
            if self.return_sequences:
                T = len(d_outputs[b])
                fwd_dh = [d_outputs[b][t][:hs] for t in range(T)]
                bwd_dh = [d_outputs[b][t][hs:] for t in range(T)]
                # Reverse backward gradients
                bwd_dh = list(reversed(bwd_dh))
                fwd_d.append(fwd_dh)
                bwd_d.append(bwd_dh)
            else:
                fwd_d.append(d_outputs[b][:hs])
                bwd_d.append(d_outputs[b][hs:])

        fwd_dx, fwd_grads = self.forward_layer.backward(fwd_d)
        bwd_dx, bwd_grads = self.backward_layer.backward(bwd_d)

        # Combine input gradients (reverse backward dx)
        d_inputs = []
        for b in range(batch_size):
            T = len(fwd_dx[b])
            bwd_dx_rev = list(reversed(bwd_dx[b]))
            seq_dx = [_vec_add(fwd_dx[b][t], bwd_dx_rev[t]) for t in range(T)]
            d_inputs.append(seq_dx)

        return d_inputs, (fwd_grads, bwd_grads)

    def update(self, grads_pair, lr):
        """Update both forward and backward layers."""
        fwd_grads, bwd_grads = grads_pair
        self.forward_layer.update(fwd_grads, lr)
        self.backward_layer.update(bwd_grads, lr)

    def param_count(self):
        return self.forward_layer.param_count() + self.backward_layer.param_count()


# ============================================================
# Embedding Layer
# ============================================================

class Embedding:
    """Embedding layer: maps integer indices to dense vectors.

    Lookup table of shape (vocab_size, embed_dim).
    """

    def __init__(self, vocab_size, embed_dim, rng=None):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        std = 1.0 / math.sqrt(embed_dim)
        self.weight = [[_randn(rng) * std for _ in range(embed_dim)]
                       for _ in range(vocab_size)]
        self.last_inputs = None
        self.training = True

    def forward(self, sequences):
        """Look up embeddings for integer sequences.

        Args:
            sequences: list of sequences, each is list of integers
        Returns:
            list of sequences of embedding vectors
        """
        self.last_inputs = sequences
        batch_outputs = []
        for seq in sequences:
            embedded = [self.weight[idx][:] for idx in seq]
            batch_outputs.append(embedded)
        return batch_outputs

    def backward(self, d_outputs):
        """Accumulate gradients for embedding weights."""
        grad_weight = _zeros_2d(self.vocab_size, self.embed_dim)
        for b, seq in enumerate(self.last_inputs):
            for t, idx in enumerate(seq):
                for j in range(self.embed_dim):
                    grad_weight[idx][j] += d_outputs[b][t][j]
        return grad_weight

    def update(self, grad_weight, lr):
        """Update embedding weights."""
        for i in range(self.vocab_size):
            for j in range(self.embed_dim):
                self.weight[i][j] -= lr * grad_weight[i][j]

    def param_count(self):
        return self.vocab_size * self.embed_dim


# ============================================================
# TimeDistributed Wrapper
# ============================================================

class TimeDistributed:
    """Apply a layer independently to each timestep.

    Wraps a Dense-like layer to process sequence outputs.
    """

    def __init__(self, in_features, out_features, rng=None):
        self.in_features = in_features
        self.out_features = out_features
        std = math.sqrt(2.0 / (in_features + out_features))
        self.W = _init_weight(out_features, in_features, in_features, out_features, rng)
        self.b = _zeros(out_features)
        self.last_inputs = None
        self.training = True

    def forward(self, sequences):
        """Apply linear transform to each timestep.

        Args:
            sequences: list of sequences of vectors
        Returns:
            list of sequences of transformed vectors
        """
        self.last_inputs = sequences
        batch_outputs = []
        for seq in sequences:
            transformed = []
            for x_t in seq:
                y = _vec_add(_mat_vec(self.W, x_t), self.b)
                transformed.append(y)
            batch_outputs.append(transformed)
        return batch_outputs

    def backward(self, d_outputs):
        """Backward through time-distributed linear layer."""
        dW = _zeros_2d(self.out_features, self.in_features)
        db = _zeros(self.out_features)
        d_inputs = []

        for b in range(len(d_outputs)):
            seq_dx = []
            for t in range(len(d_outputs[b])):
                dy = d_outputs[b][t]
                x = self.last_inputs[b][t]

                # Accumulate weight gradients
                outer = _vec_outer(dy, x)
                for i in range(self.out_features):
                    for j in range(self.in_features):
                        dW[i][j] += outer[i][j]
                    db[i] += dy[i]

                # Input gradient
                dx = _mat_vec(_transpose(self.W), dy)
                seq_dx.append(dx)
            d_inputs.append(seq_dx)

        return d_inputs, {'dW': dW, 'db': db}

    def update(self, grads, lr):
        if 'dW' in grads:
            g = grads['dW']
            for i in range(self.out_features):
                for j in range(self.in_features):
                    self.W[i][j] -= lr * g[i][j]
        if 'db' in grads:
            g = grads['db']
            for i in range(self.out_features):
                self.b[i] -= lr * g[i]

    def param_count(self):
        return self.out_features * self.in_features + self.out_features


# ============================================================
# Sequence Utilities
# ============================================================

def pad_sequences(sequences, max_len=None, pad_value=0.0):
    """Pad sequences to uniform length.

    Args:
        sequences: list of variable-length sequences (lists of vectors or ints)
        max_len: target length (default: longest sequence)
        pad_value: padding value
    Returns:
        list of padded sequences, all same length
    """
    if max_len is None:
        max_len = max(len(s) for s in sequences)

    padded = []
    for seq in sequences:
        if len(seq) >= max_len:
            padded.append(seq[:max_len])
        else:
            # Determine element type
            if seq and isinstance(seq[0], list):
                dim = len(seq[0])
                pad_elem = [pad_value] * dim
                padded.append(seq + [pad_elem[:] for _ in range(max_len - len(seq))])
            else:
                padded.append(seq + [pad_value] * (max_len - len(seq)))

    return padded


def one_hot_encode_sequence(int_seq, vocab_size):
    """Convert integer sequence to one-hot vectors.

    Args:
        int_seq: list of integers
        vocab_size: size of vocabulary
    Returns:
        list of one-hot vectors
    """
    result = []
    for idx in int_seq:
        vec = _zeros(vocab_size)
        if 0 <= idx < vocab_size:
            vec[idx] = 1.0
        result.append(vec)
    return result


def create_sequence_pairs(data, window_size):
    """Create input/target pairs from sequential data using sliding window.

    Args:
        data: list of values
        window_size: number of past values to use as input
    Returns:
        (inputs, targets) where inputs are windows and targets are next values
    """
    inputs = []
    targets = []
    for i in range(len(data) - window_size):
        inputs.append(data[i:i + window_size])
        targets.append(data[i + window_size])
    return inputs, targets


def make_char_data(text, seq_len=10):
    """Convert text to character-level sequences for language modeling.

    Returns:
        sequences: list of integer sequences (input)
        targets: list of integer sequences (shifted by 1)
        char_to_idx: mapping dict
        idx_to_char: reverse mapping dict
    """
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}

    sequences = []
    targets = []
    for i in range(0, len(text) - seq_len):
        seq = [char_to_idx[c] for c in text[i:i + seq_len]]
        tgt = [char_to_idx[c] for c in text[i + 1:i + seq_len + 1]]
        sequences.append(seq)
        targets.append(tgt)

    return sequences, targets, char_to_idx, idx_to_char


# ============================================================
# Sequence Model (high-level API)
# ============================================================

class SequenceModel:
    """High-level sequence model combining embedding + recurrent + output layers.

    Supports:
    - Many-to-one (classification/regression from sequences)
    - Many-to-many (sequence labeling, language modeling)
    - Stacked recurrent layers
    """

    def __init__(self):
        self.layers = []
        self.training = True

    def add(self, layer):
        """Add a layer to the model."""
        self.layers.append(layer)
        return self

    def forward(self, inputs):
        """Forward pass through all layers."""
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict(self, inputs):
        """Forward pass in eval mode."""
        old_training = []
        for layer in self.layers:
            old_training.append(getattr(layer, 'training', True))
            layer.training = False
        result = self.forward(inputs)
        for i, layer in enumerate(self.layers):
            layer.training = old_training[i]
        return result

    def param_count(self):
        """Total parameter count."""
        return sum(l.param_count() for l in self.layers if hasattr(l, 'param_count'))


# ============================================================
# Loss functions for sequences
# ============================================================

class SequenceCrossEntropyLoss:
    """Cross-entropy loss for sequence outputs (e.g., language modeling).

    Expects logits per timestep and integer targets.
    """

    def __init__(self):
        self.last_probs = None
        self.last_targets = None

    def forward(self, logits_batch, targets_batch):
        """Compute average cross-entropy loss over all timesteps.

        Args:
            logits_batch: list of sequences of logit vectors
            targets_batch: list of sequences of integer targets
        Returns:
            scalar loss
        """
        total_loss = 0.0
        count = 0
        self.last_probs = []
        self.last_targets = targets_batch

        for b in range(len(logits_batch)):
            seq_probs = []
            for t in range(len(logits_batch[b])):
                logits = logits_batch[b][t]
                target = targets_batch[b][t]

                # Softmax
                max_l = max(logits)
                exps = [math.exp(l - max_l) for l in logits]
                s = sum(exps)
                probs = [e / s for e in exps]
                seq_probs.append(probs)

                # Cross-entropy
                p = max(probs[target], 1e-12)
                total_loss -= math.log(p)
                count += 1
            self.last_probs.append(seq_probs)

        return total_loss / max(count, 1)

    def backward(self):
        """Compute gradients w.r.t. logits."""
        d_logits = []
        for b in range(len(self.last_probs)):
            seq_d = []
            for t in range(len(self.last_probs[b])):
                probs = self.last_probs[b][t]
                target = self.last_targets[b][t]
                d = probs[:]
                d[target] -= 1.0
                # Average over total count
                total = sum(len(self.last_probs[bb]) for bb in range(len(self.last_probs)))
                d = _vec_scale(d, 1.0 / max(total, 1))
                seq_d.append(d)
            d_logits.append(seq_d)
        return d_logits


class SequenceMSELoss:
    """MSE loss for sequence regression (e.g., time series prediction)."""

    def __init__(self):
        self.last_preds = None
        self.last_targets = None

    def forward(self, preds, targets):
        """Compute MSE loss.

        Args:
            preds: list of prediction vectors (batch of hidden states)
            targets: list of target vectors
        Returns:
            scalar loss
        """
        self.last_preds = preds
        self.last_targets = targets
        total = 0.0
        count = 0
        for b in range(len(preds)):
            if isinstance(preds[b][0], list):
                # Many-to-many
                for t in range(len(preds[b])):
                    for j in range(len(preds[b][t])):
                        diff = preds[b][t][j] - targets[b][t][j]
                        total += diff * diff
                        count += 1
            else:
                # Many-to-one
                n_out = min(len(preds[b]), len(targets[b]))
                for j in range(n_out):
                    diff = preds[b][j] - targets[b][j]
                    total += diff * diff
                    count += 1
        return total / max(count, 1)

    def backward(self):
        """Compute gradients."""
        d_preds = []
        total_count = 0
        for b in range(len(self.last_preds)):
            if isinstance(self.last_preds[b][0], list):
                for t in range(len(self.last_preds[b])):
                    total_count += len(self.last_preds[b][t])
            else:
                total_count += min(len(self.last_preds[b]), len(self.last_targets[b]))

        for b in range(len(self.last_preds)):
            if isinstance(self.last_preds[b][0], list):
                seq_d = []
                for t in range(len(self.last_preds[b])):
                    d = [(2.0 * (self.last_preds[b][t][j] - self.last_targets[b][t][j])) / total_count
                         for j in range(len(self.last_preds[b][t]))]
                    seq_d.append(d)
                d_preds.append(seq_d)
            else:
                n_out = min(len(self.last_preds[b]), len(self.last_targets[b]))
                d = [(2.0 * (self.last_preds[b][j] - self.last_targets[b][j])) / total_count
                     for j in range(n_out)]
                # Pad with zeros for remaining hidden dims
                d.extend([0.0] * (len(self.last_preds[b]) - n_out))
                d_preds.append(d)

        return d_preds


# ============================================================
# Training utilities
# ============================================================

def train_sequence_model(model, X, Y, loss_fn, lr=0.01, epochs=100,
                         batch_size=None, clip_grad=None, verbose=False,
                         shuffle=True, seed=42):
    """Train a sequence model.

    Args:
        model: SequenceModel instance
        X: list of input sequences
        Y: list of targets
        loss_fn: loss function with forward/backward
        lr: learning rate
        epochs: number of training epochs
        batch_size: mini-batch size (None = full batch)
        clip_grad: gradient clipping value
        verbose: print loss each epoch
        shuffle: shuffle data each epoch
        seed: random seed
    Returns:
        list of loss values per epoch
    """
    rng = random.Random(seed)
    n = len(X)
    if batch_size is None:
        batch_size = n

    losses = []
    for epoch in range(epochs):
        # Shuffle
        if shuffle:
            indices = list(range(n))
            rng.shuffle(indices)
            X_shuffled = [X[i] for i in indices]
            Y_shuffled = [Y[i] for i in indices]
        else:
            X_shuffled = X
            Y_shuffled = Y

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x_batch = X_shuffled[start:end]
            y_batch = Y_shuffled[start:end]

            # Forward
            output = model.forward(x_batch)
            loss = loss_fn.forward(output, y_batch)
            epoch_loss += loss

            # Backward
            d_output = loss_fn.backward()

            # Backward through layers in reverse
            d = d_output
            layer_grads = []
            for layer in reversed(model.layers):
                if hasattr(layer, 'backward'):
                    result = layer.backward(d)
                    if isinstance(result, tuple) and len(result) == 2:
                        d, grads = result
                        layer_grads.append((layer, grads))
                    else:
                        # Embedding returns just grad_weight
                        layer_grads.append((layer, result))
                        d = None

            # Update
            for layer, grads in layer_grads:
                if hasattr(layer, 'update'):
                    layer.update(grads, lr)

            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch}: loss={avg_loss:.6f}")

    return losses


# ============================================================
# Data generators for testing
# ============================================================

def make_sine_data(n=100, seq_len=10, seed=42):
    """Generate sine wave prediction data.

    Returns (X, Y) where X are windows of sine values and Y are next values.
    """
    rng = random.Random(seed)
    # Generate sine wave with some noise
    data = [math.sin(i * 0.1) + rng.gauss(0, 0.01) for i in range(n + seq_len)]
    X = []
    Y = []
    for i in range(n):
        # Input: seq_len timesteps, each a 1D vector
        seq = [[data[i + t]] for t in range(seq_len)]
        X.append(seq)
        Y.append([data[i + seq_len]])
    return X, Y


def make_echo_data(n=100, seq_len=5, delay=2, vocab_size=4, seed=42):
    """Generate echo task data (classify the delay-th element).

    Returns (X, Y) where X are one-hot sequences and Y are class labels.
    """
    rng = random.Random(seed)
    X = []
    Y = []
    for _ in range(n):
        seq = [rng.randint(0, vocab_size - 1) for _ in range(seq_len)]
        x = one_hot_encode_sequence(seq, vocab_size)
        y_class = seq[delay] if delay < len(seq) else 0
        # One-hot target
        y = _zeros(vocab_size)
        y[y_class] = 1.0
        X.append(x)
        Y.append(y)
    return X, Y


def make_sequence_copy_data(n=50, seq_len=4, vocab_size=4, seed=42):
    """Generate sequence copy data (output = input shifted).

    Returns (X, Y) as integer sequences for many-to-many.
    """
    rng = random.Random(seed)
    X = []
    Y = []
    for _ in range(n):
        seq = [rng.randint(0, vocab_size - 1) for _ in range(seq_len)]
        X.append(seq)
        Y.append(seq[:])  # Copy task
    return X, Y


def make_addition_data(n=100, max_val=10, seed=42):
    """Generate addition task: sequence [a, b] -> sum.

    Returns (X, Y) for many-to-one regression.
    """
    rng = random.Random(seed)
    X = []
    Y = []
    for _ in range(n):
        a = rng.uniform(0, max_val)
        b = rng.uniform(0, max_val)
        X.append([[a / max_val], [b / max_val]])
        Y.append([(a + b) / (2 * max_val)])
    return X, Y


def generate_text(model, embedding, start_indices, length, vocab_size, temperature=1.0):
    """Generate text using a trained language model.

    Args:
        model: SequenceModel with recurrent + output layers
        embedding: Embedding layer
        start_indices: list of starting token indices
        length: number of tokens to generate
        vocab_size: vocabulary size
        temperature: sampling temperature (lower = more deterministic)
    Returns:
        list of generated token indices
    """
    generated = list(start_indices)

    for _ in range(length):
        # Use current sequence
        seq = [generated[-len(start_indices):]]  # Last context_len tokens
        embedded = embedding.forward(seq)
        output = model.forward(embedded)

        # Get logits from last timestep
        if isinstance(output[0], list) and isinstance(output[0][0], list):
            logits = output[0][-1]  # Many-to-many: last timestep
        else:
            logits = output[0]  # Many-to-one

        # Temperature scaling and softmax
        scaled = [l / max(temperature, 1e-8) for l in logits]
        max_l = max(scaled)
        exps = [math.exp(l - max_l) for l in scaled]
        s = sum(exps)
        probs = [e / s for e in exps]

        # Sample
        r = random.random()
        cumsum = 0.0
        chosen = len(probs) - 1
        for i, p in enumerate(probs):
            cumsum += p
            if r < cumsum:
                chosen = i
                break

        generated.append(chosen)

    return generated
