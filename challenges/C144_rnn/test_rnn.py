"""Tests for C144: Recurrent Neural Network."""
import math
import random
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from rnn import (
    # Cells
    RNNCell, LSTMCell, GRUCell,
    # Layers
    RNN, LSTMLayer, GRULayer,
    # Wrappers
    Bidirectional, TimeDistributed, Embedding,
    # Model
    SequenceModel,
    # Loss
    SequenceCrossEntropyLoss, SequenceMSELoss,
    # Training
    train_sequence_model,
    # Utils
    pad_sequences, one_hot_encode_sequence, create_sequence_pairs,
    make_char_data, make_sine_data, make_echo_data,
    make_sequence_copy_data, make_addition_data, generate_text,
    # Internal
    _mat_mul, _mat_vec, _vec_outer, _vec_add, _vec_mul, _vec_sub,
    _vec_scale, _sigmoid_vec, _tanh_vec, _relu_vec, _zeros, _zeros_2d,
    _clip_vec, _clip_mat, _transpose, _init_weight,
)


# ============================================================
# Utility function tests
# ============================================================

class TestUtilities:
    def test_mat_mul(self):
        A = [[1, 2], [3, 4]]
        B = [[5, 6], [7, 8]]
        result = _mat_mul(A, B)
        assert result == [[19, 22], [43, 50]]

    def test_mat_vec(self):
        A = [[1, 2], [3, 4]]
        v = [5, 6]
        result = _mat_vec(A, v)
        assert result == [17, 39]

    def test_vec_outer(self):
        u = [1, 2]
        v = [3, 4, 5]
        result = _vec_outer(u, v)
        assert result == [[3, 4, 5], [6, 8, 10]]

    def test_vec_add(self):
        assert _vec_add([1, 2], [3, 4]) == [4, 6]

    def test_vec_mul(self):
        assert _vec_mul([1, 2], [3, 4]) == [3, 8]

    def test_vec_sub(self):
        assert _vec_sub([5, 3], [2, 1]) == [3, 2]

    def test_vec_scale(self):
        assert _vec_scale([1, 2, 3], 2.0) == [2.0, 4.0, 6.0]

    def test_sigmoid_vec(self):
        result = _sigmoid_vec([0.0])
        assert abs(result[0] - 0.5) < 1e-10

    def test_sigmoid_extremes(self):
        result = _sigmoid_vec([-1000, 1000])
        assert result[0] < 0.01
        assert result[1] > 0.99

    def test_tanh_vec(self):
        result = _tanh_vec([0.0])
        assert abs(result[0]) < 1e-10

    def test_relu_vec(self):
        assert _relu_vec([-1, 0, 1, 2]) == [0.0, 0.0, 1, 2]

    def test_zeros(self):
        z = _zeros(5)
        assert z == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_zeros_2d(self):
        z = _zeros_2d(2, 3)
        assert len(z) == 2
        assert all(len(row) == 3 for row in z)

    def test_clip_vec(self):
        v = [3.0, 4.0]  # norm = 5
        clipped = _clip_vec(v, 2.5)
        norm = math.sqrt(sum(x * x for x in clipped))
        assert abs(norm - 2.5) < 1e-10

    def test_clip_vec_no_clip(self):
        v = [1.0, 1.0]
        clipped = _clip_vec(v, 10.0)
        assert clipped == v

    def test_clip_mat(self):
        m = [[3.0, 4.0]]  # Frobenius norm = 5
        clipped = _clip_mat(m, 2.5)
        norm = math.sqrt(sum(x * x for row in clipped for x in row))
        assert abs(norm - 2.5) < 1e-10

    def test_transpose(self):
        m = [[1, 2, 3], [4, 5, 6]]
        t = _transpose(m)
        assert t == [[1, 4], [2, 5], [3, 6]]

    def test_init_weight_shape(self):
        w = _init_weight(3, 4, 4, 3)
        assert len(w) == 3
        assert all(len(row) == 4 for row in w)


# ============================================================
# RNNCell tests
# ============================================================

class TestRNNCell:
    def test_creation(self):
        cell = RNNCell(4, 8, rng=random.Random(42))
        assert cell.input_size == 4
        assert cell.hidden_size == 8

    def test_forward_shape(self):
        cell = RNNCell(3, 5, rng=random.Random(42))
        x = [1.0, 0.0, -1.0]
        h = _zeros(5)
        h_new, cache = cell.forward(x, h)
        assert len(h_new) == 5

    def test_forward_nonzero(self):
        cell = RNNCell(2, 3, rng=random.Random(42))
        x = [1.0, -1.0]
        h = _zeros(3)
        h_new, _ = cell.forward(x, h)
        assert any(abs(v) > 0 for v in h_new)

    def test_forward_depends_on_hidden(self):
        cell = RNNCell(2, 3, rng=random.Random(42))
        x = [1.0, 0.0]
        h1, _ = cell.forward(x, _zeros(3))
        h2, _ = cell.forward(x, [1.0, 1.0, 1.0])
        assert h1 != h2

    def test_backward_shapes(self):
        cell = RNNCell(3, 4, rng=random.Random(42))
        x = [1.0, 0.5, -0.5]
        h = _zeros(4)
        h_new, cache = cell.forward(x, h)
        dh = [0.1] * 4
        dx, dh_prev, grads = cell.backward(dh, cache)
        assert len(dx) == 3
        assert len(dh_prev) == 4
        assert 'dW_ih' in grads
        assert 'dW_hh' in grads
        assert 'db_h' in grads

    def test_tanh_bounds(self):
        cell = RNNCell(2, 3, rng=random.Random(42))
        x = [100.0, -100.0]
        h = [50.0, -50.0, 50.0]
        h_new, _ = cell.forward(x, h)
        for v in h_new:
            assert -1.0 <= v <= 1.0

    def test_param_count(self):
        cell = RNNCell(4, 8)
        # W_ih: 8x4=32, W_hh: 8x8=64, b_h: 8 = 104
        assert cell.param_count() == 104

    def test_params_dict(self):
        cell = RNNCell(3, 5, rng=random.Random(42))
        p = cell.params()
        assert 'W_ih' in p and 'W_hh' in p and 'b_h' in p


# ============================================================
# LSTMCell tests
# ============================================================

class TestLSTMCell:
    def test_creation(self):
        cell = LSTMCell(4, 8, rng=random.Random(42))
        assert cell.input_size == 4
        assert cell.hidden_size == 8

    def test_forget_gate_bias(self):
        cell = LSTMCell(3, 5)
        # Forget gate bias initialized to 1
        assert all(b == 1.0 for b in cell.b_f)

    def test_forward_shape(self):
        cell = LSTMCell(3, 5, rng=random.Random(42))
        x = [1.0, 0.0, -1.0]
        h = _zeros(5)
        c = _zeros(5)
        h_new, c_new, cache = cell.forward(x, h, c)
        assert len(h_new) == 5
        assert len(c_new) == 5

    def test_cell_state_persists(self):
        cell = LSTMCell(2, 4, rng=random.Random(42))
        x = [1.0, -1.0]
        h = _zeros(4)
        c = _zeros(4)
        h1, c1, _ = cell.forward(x, h, c)
        h2, c2, _ = cell.forward(x, h1, c1)
        # Cell state should change
        assert c1 != c2

    def test_backward_shapes(self):
        cell = LSTMCell(3, 4, rng=random.Random(42))
        x = [1.0, 0.5, -0.5]
        h = _zeros(4)
        c = _zeros(4)
        h_new, c_new, cache = cell.forward(x, h, c)
        dh = [0.1] * 4
        dc = [0.1] * 4
        dx, dh_prev, dc_prev, grads = cell.backward(dh, dc, cache)
        assert len(dx) == 3
        assert len(dh_prev) == 4
        assert len(dc_prev) == 4

    def test_backward_gradient_keys(self):
        cell = LSTMCell(2, 3, rng=random.Random(42))
        x = [1.0, -1.0]
        h_new, c_new, cache = cell.forward(x, _zeros(3), _zeros(3))
        _, _, _, grads = cell.backward([0.1] * 3, [0.1] * 3, cache)
        for gate in ['f', 'i', 'g', 'o']:
            assert f'dW_i{gate}' in grads
            assert f'dW_h{gate}' in grads
            assert f'db_{gate}' in grads

    def test_param_count(self):
        cell = LSTMCell(4, 8)
        # 4 gates * (4*8 + 8*8 + 8) = 4 * 104 = 416
        assert cell.param_count() == 416

    def test_params_dict(self):
        cell = LSTMCell(3, 5, rng=random.Random(42))
        p = cell.params()
        assert len(p) == 12  # 4 gates * 3 params each


# ============================================================
# GRUCell tests
# ============================================================

class TestGRUCell:
    def test_creation(self):
        cell = GRUCell(4, 8, rng=random.Random(42))
        assert cell.input_size == 4
        assert cell.hidden_size == 8

    def test_forward_shape(self):
        cell = GRUCell(3, 5, rng=random.Random(42))
        x = [1.0, 0.0, -1.0]
        h = _zeros(5)
        h_new, cache = cell.forward(x, h)
        assert len(h_new) == 5

    def test_forward_changes_state(self):
        cell = GRUCell(2, 4, rng=random.Random(42))
        x = [1.0, -1.0]
        h1, _ = cell.forward(x, _zeros(4))
        h2, _ = cell.forward(x, h1)
        assert h1 != h2

    def test_backward_shapes(self):
        cell = GRUCell(3, 4, rng=random.Random(42))
        x = [1.0, 0.5, -0.5]
        h_new, cache = cell.forward(x, _zeros(4))
        dh = [0.1] * 4
        dx, dh_prev, grads = cell.backward(dh, cache)
        assert len(dx) == 3
        assert len(dh_prev) == 4

    def test_backward_gradient_keys(self):
        cell = GRUCell(2, 3, rng=random.Random(42))
        h_new, cache = cell.forward([1.0, -1.0], _zeros(3))
        _, _, grads = cell.backward([0.1] * 3, cache)
        for gate in ['z', 'r', 'n']:
            assert f'dW_i{gate}' in grads
            assert f'dW_h{gate}' in grads
            assert f'db_{gate}' in grads

    def test_param_count(self):
        cell = GRUCell(4, 8)
        # 3 gates * (4*8 + 8*8 + 8) = 3 * 104 = 312
        assert cell.param_count() == 312

    def test_params_dict(self):
        cell = GRUCell(3, 5, rng=random.Random(42))
        p = cell.params()
        assert len(p) == 9  # 3 gates * 3 params each


# ============================================================
# RNN Layer tests
# ============================================================

class TestRNNLayer:
    def test_forward_many_to_one(self):
        layer = RNN(3, 5, return_sequences=False, rng=random.Random(42))
        seqs = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        out = layer.forward(seqs)
        assert len(out) == 1
        assert len(out[0]) == 5

    def test_forward_many_to_many(self):
        layer = RNN(3, 5, return_sequences=True, rng=random.Random(42))
        seqs = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        out = layer.forward(seqs)
        assert len(out) == 1
        assert len(out[0]) == 3  # 3 timesteps
        assert len(out[0][0]) == 5  # hidden_size

    def test_batch_processing(self):
        layer = RNN(2, 4, return_sequences=False, rng=random.Random(42))
        seqs = [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        ]
        out = layer.forward(seqs)
        assert len(out) == 2
        assert len(out[0]) == 4
        assert len(out[1]) == 4

    def test_backward_many_to_one(self):
        layer = RNN(3, 4, return_sequences=False, rng=random.Random(42))
        seqs = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        out = layer.forward(seqs)
        d_out = [[0.1] * 4]
        d_inputs, grads = layer.backward(d_out)
        assert len(d_inputs) == 1
        assert len(d_inputs[0]) == 2  # 2 timesteps
        assert len(d_inputs[0][0]) == 3  # input_size

    def test_backward_many_to_many(self):
        layer = RNN(3, 4, return_sequences=True, rng=random.Random(42))
        seqs = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        out = layer.forward(seqs)
        d_out = [[[0.1] * 4, [0.1] * 4]]
        d_inputs, grads = layer.backward(d_out)
        assert len(d_inputs) == 1
        assert len(d_inputs[0]) == 2

    def test_gradient_clipping(self):
        layer = RNN(2, 3, clip_grad=1.0, rng=random.Random(42))
        seqs = [[[100.0, 100.0], [100.0, 100.0]]]
        layer.forward(seqs)
        d_out = [[10.0, 10.0, 10.0]]
        _, grads = layer.backward(d_out)
        # Verify gradients were clipped during backward
        # (clip happens inside backward, so just check it runs)
        assert grads is not None

    def test_update(self):
        layer = RNN(2, 3, rng=random.Random(42))
        seqs = [[[1.0, 0.0], [0.0, 1.0]]]
        layer.forward(seqs)
        d_out = [[0.1, 0.1, 0.1]]
        _, grads = layer.backward(d_out)
        w_before = [row[:] for row in layer.cell.W_ih]
        layer.update(grads, 0.01)
        # Weights should change
        assert layer.cell.W_ih != w_before

    def test_param_count(self):
        layer = RNN(4, 8)
        assert layer.param_count() == 104


# ============================================================
# LSTM Layer tests
# ============================================================

class TestLSTMLayer:
    def test_forward_many_to_one(self):
        layer = LSTMLayer(3, 5, return_sequences=False, rng=random.Random(42))
        seqs = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        out = layer.forward(seqs)
        assert len(out) == 1
        assert len(out[0]) == 5

    def test_forward_many_to_many(self):
        layer = LSTMLayer(3, 5, return_sequences=True, rng=random.Random(42))
        seqs = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        out = layer.forward(seqs)
        assert len(out) == 1
        assert len(out[0]) == 3
        assert len(out[0][0]) == 5

    def test_batch_processing(self):
        layer = LSTMLayer(2, 4, return_sequences=False, rng=random.Random(42))
        seqs = [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5]],
        ]
        out = layer.forward(seqs)
        assert len(out) == 2

    def test_backward(self):
        layer = LSTMLayer(3, 4, return_sequences=False, rng=random.Random(42))
        seqs = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        layer.forward(seqs)
        d_out = [[0.1] * 4]
        d_inputs, grads = layer.backward(d_out)
        assert len(d_inputs[0]) == 2
        assert 'dW_if' in grads  # forget gate weight grad

    def test_backward_many_to_many(self):
        layer = LSTMLayer(2, 3, return_sequences=True, rng=random.Random(42))
        seqs = [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]
        layer.forward(seqs)
        d_out = [[[0.1] * 3] * 3]
        d_inputs, grads = layer.backward(d_out)
        assert len(d_inputs[0]) == 3

    def test_update(self):
        layer = LSTMLayer(2, 3, rng=random.Random(42))
        seqs = [[[1.0, 0.0], [0.0, 1.0]]]
        layer.forward(seqs)
        d_out = [[0.1, 0.1, 0.1]]
        _, grads = layer.backward(d_out)
        bf_before = layer.cell.b_f[:]
        layer.update(grads, 0.01)
        assert layer.cell.b_f != bf_before

    def test_gradient_clipping(self):
        layer = LSTMLayer(2, 3, clip_grad=0.5, rng=random.Random(42))
        seqs = [[[10.0, 10.0], [10.0, 10.0]]]
        layer.forward(seqs)
        d_out = [[1.0, 1.0, 1.0]]
        _, grads = layer.backward(d_out)
        layer.update(grads, 0.01)  # Should not crash

    def test_param_count(self):
        layer = LSTMLayer(4, 8)
        assert layer.param_count() == 416

    def test_long_sequence(self):
        layer = LSTMLayer(2, 4, return_sequences=False, rng=random.Random(42))
        seq = [[random.gauss(0, 1), random.gauss(0, 1)] for _ in range(20)]
        out = layer.forward([seq])
        assert len(out[0]) == 4


# ============================================================
# GRU Layer tests
# ============================================================

class TestGRULayer:
    def test_forward_many_to_one(self):
        layer = GRULayer(3, 5, return_sequences=False, rng=random.Random(42))
        seqs = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        out = layer.forward(seqs)
        assert len(out) == 1
        assert len(out[0]) == 5

    def test_forward_many_to_many(self):
        layer = GRULayer(3, 5, return_sequences=True, rng=random.Random(42))
        seqs = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        out = layer.forward(seqs)
        assert len(out[0]) == 2
        assert len(out[0][0]) == 5

    def test_backward(self):
        layer = GRULayer(3, 4, return_sequences=False, rng=random.Random(42))
        seqs = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        layer.forward(seqs)
        d_out = [[0.1] * 4]
        d_inputs, grads = layer.backward(d_out)
        assert len(d_inputs[0]) == 2
        assert 'dW_iz' in grads  # update gate

    def test_update(self):
        layer = GRULayer(2, 3, rng=random.Random(42))
        seqs = [[[1.0, 0.0], [0.0, 1.0]]]
        layer.forward(seqs)
        d_out = [[0.1, 0.1, 0.1]]
        _, grads = layer.backward(d_out)
        w_before = [row[:] for row in layer.cell.W_iz]
        layer.update(grads, 0.01)
        assert layer.cell.W_iz != w_before

    def test_param_count(self):
        layer = GRULayer(4, 8)
        assert layer.param_count() == 312

    def test_batch(self):
        layer = GRULayer(2, 3, rng=random.Random(42))
        seqs = [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [0.5, 0.5]],
        ]
        out = layer.forward(seqs)
        assert len(out) == 2


# ============================================================
# Bidirectional tests
# ============================================================

class TestBidirectional:
    def test_forward_many_to_one(self):
        layer = Bidirectional(RNN, 3, 5, return_sequences=False, rng=random.Random(42))
        seqs = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        out = layer.forward(seqs)
        assert len(out) == 1
        assert len(out[0]) == 10  # 2 * hidden_size

    def test_forward_many_to_many(self):
        layer = Bidirectional(RNN, 3, 5, return_sequences=True, rng=random.Random(42))
        seqs = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        out = layer.forward(seqs)
        assert len(out[0]) == 2  # 2 timesteps
        assert len(out[0][0]) == 10  # 2 * hidden_size

    def test_bidirectional_lstm(self):
        layer = Bidirectional(LSTMLayer, 3, 4, return_sequences=False, rng=random.Random(42))
        seqs = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        out = layer.forward(seqs)
        assert len(out[0]) == 8  # 2 * 4

    def test_bidirectional_gru(self):
        layer = Bidirectional(GRULayer, 3, 4, return_sequences=False, rng=random.Random(42))
        seqs = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        out = layer.forward(seqs)
        assert len(out[0]) == 8

    def test_backward(self):
        layer = Bidirectional(RNN, 2, 3, return_sequences=False, rng=random.Random(42))
        seqs = [[[1.0, 0.0], [0.0, 1.0]]]
        out = layer.forward(seqs)
        d_out = [[0.1] * 6]
        d_inputs, grads_pair = layer.backward(d_out)
        assert len(d_inputs[0]) == 2
        assert len(d_inputs[0][0]) == 2

    def test_backward_many_to_many(self):
        layer = Bidirectional(RNN, 2, 3, return_sequences=True, rng=random.Random(42))
        seqs = [[[1.0, 0.0], [0.0, 1.0]]]
        out = layer.forward(seqs)
        d_out = [[[0.1] * 6, [0.1] * 6]]
        d_inputs, grads_pair = layer.backward(d_out)
        assert len(d_inputs[0]) == 2

    def test_update(self):
        layer = Bidirectional(RNN, 2, 3, rng=random.Random(42))
        seqs = [[[1.0, 0.0], [0.0, 1.0]]]
        layer.forward(seqs)
        d_out = [[0.1] * 6]
        _, grads_pair = layer.backward(d_out)
        layer.update(grads_pair, 0.01)  # Should not crash

    def test_param_count(self):
        layer = Bidirectional(RNN, 4, 8)
        assert layer.param_count() == 208  # 2 * 104


# ============================================================
# Embedding tests
# ============================================================

class TestEmbedding:
    def test_creation(self):
        emb = Embedding(10, 5, rng=random.Random(42))
        assert emb.vocab_size == 10
        assert emb.embed_dim == 5

    def test_forward(self):
        emb = Embedding(10, 4, rng=random.Random(42))
        seqs = [[0, 3, 7]]
        out = emb.forward(seqs)
        assert len(out) == 1
        assert len(out[0]) == 3  # 3 tokens
        assert len(out[0][0]) == 4  # embed_dim

    def test_same_index_same_embedding(self):
        emb = Embedding(10, 4, rng=random.Random(42))
        seqs = [[3, 3, 3]]
        out = emb.forward(seqs)
        assert out[0][0] == out[0][1] == out[0][2]

    def test_different_indices_different_embeddings(self):
        emb = Embedding(10, 4, rng=random.Random(42))
        seqs = [[0, 1]]
        out = emb.forward(seqs)
        assert out[0][0] != out[0][1]

    def test_backward(self):
        emb = Embedding(5, 3, rng=random.Random(42))
        seqs = [[0, 2, 4]]
        emb.forward(seqs)
        d_out = [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]
        grad = emb.backward(d_out)
        assert len(grad) == 5
        assert grad[0] == [0.1, 0.2, 0.3]  # Index 0
        assert grad[2] == [0.4, 0.5, 0.6]  # Index 2
        assert grad[1] == [0.0, 0.0, 0.0]  # Index 1 not used

    def test_backward_accumulates(self):
        emb = Embedding(5, 2, rng=random.Random(42))
        seqs = [[1, 1]]  # Same index twice
        emb.forward(seqs)
        d_out = [[[1.0, 2.0], [3.0, 4.0]]]
        grad = emb.backward(d_out)
        assert grad[1] == [4.0, 6.0]  # Accumulated

    def test_update(self):
        emb = Embedding(5, 3, rng=random.Random(42))
        w_before = [row[:] for row in emb.weight]
        seqs = [[0, 1]]
        emb.forward(seqs)
        d_out = [[[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]]
        grad = emb.backward(d_out)
        emb.update(grad, 0.1)
        assert emb.weight[0] != w_before[0]

    def test_param_count(self):
        emb = Embedding(100, 32)
        assert emb.param_count() == 3200

    def test_batch(self):
        emb = Embedding(10, 4, rng=random.Random(42))
        seqs = [[0, 1, 2], [3, 4, 5]]
        out = emb.forward(seqs)
        assert len(out) == 2


# ============================================================
# TimeDistributed tests
# ============================================================

class TestTimeDistributed:
    def test_forward(self):
        td = TimeDistributed(4, 3, rng=random.Random(42))
        seqs = [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]
        out = td.forward(seqs)
        assert len(out) == 1
        assert len(out[0]) == 2  # 2 timesteps
        assert len(out[0][0]) == 3  # out_features

    def test_backward(self):
        td = TimeDistributed(3, 2, rng=random.Random(42))
        seqs = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        td.forward(seqs)
        d_out = [[[0.1, 0.2], [0.3, 0.4]]]
        d_inputs, grads = td.backward(d_out)
        assert len(d_inputs[0]) == 2
        assert len(d_inputs[0][0]) == 3
        assert 'dW' in grads
        assert 'db' in grads

    def test_update(self):
        td = TimeDistributed(3, 2, rng=random.Random(42))
        w_before = [row[:] for row in td.W]
        seqs = [[[1.0, 0.0, 0.0]]]
        td.forward(seqs)
        d_out = [[[0.1, 0.2]]]
        _, grads = td.backward(d_out)
        td.update(grads, 0.01)
        assert td.W != w_before

    def test_param_count(self):
        td = TimeDistributed(10, 5)
        assert td.param_count() == 55  # 10*5 + 5


# ============================================================
# Sequence utilities tests
# ============================================================

class TestSequenceUtils:
    def test_pad_sequences_vectors(self):
        seqs = [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0]],
        ]
        padded = pad_sequences(seqs)
        assert len(padded[0]) == 2
        assert len(padded[1]) == 2
        assert padded[1][1] == [0.0, 0.0]

    def test_pad_sequences_integers(self):
        seqs = [[1, 2, 3], [4, 5]]
        padded = pad_sequences(seqs)
        assert padded == [[1, 2, 3], [4, 5, 0.0]]

    def test_pad_sequences_max_len(self):
        seqs = [[1, 2, 3, 4], [5, 6]]
        padded = pad_sequences(seqs, max_len=3)
        assert padded == [[1, 2, 3], [5, 6, 0.0]]

    def test_pad_sequences_custom_value(self):
        seqs = [[1, 2], [3]]
        padded = pad_sequences(seqs, pad_value=-1)
        assert padded[1] == [3, -1]

    def test_one_hot_encode_sequence(self):
        result = one_hot_encode_sequence([0, 2, 1], 3)
        assert result[0] == [1.0, 0.0, 0.0]
        assert result[1] == [0.0, 0.0, 1.0]
        assert result[2] == [0.0, 1.0, 0.0]

    def test_one_hot_out_of_range(self):
        result = one_hot_encode_sequence([5], 3)
        assert result[0] == [0.0, 0.0, 0.0]

    def test_create_sequence_pairs(self):
        data = [1, 2, 3, 4, 5]
        inputs, targets = create_sequence_pairs(data, 3)
        assert inputs == [[1, 2, 3], [2, 3, 4]]
        assert targets == [4, 5]

    def test_make_char_data(self):
        seqs, targets, c2i, i2c = make_char_data("abcabc", seq_len=3)
        assert len(seqs) > 0
        assert len(seqs[0]) == 3
        assert len(c2i) == 3  # a, b, c
        assert len(targets) == len(seqs)

    def test_make_char_data_targets_shifted(self):
        seqs, targets, c2i, i2c = make_char_data("abcde", seq_len=2)
        # First sequence is "ab", target is "bc"
        assert seqs[0] == [c2i['a'], c2i['b']]
        assert targets[0] == [c2i['b'], c2i['c']]


# ============================================================
# Data generator tests
# ============================================================

class TestDataGenerators:
    def test_make_sine_data(self):
        X, Y = make_sine_data(n=50, seq_len=5)
        assert len(X) == 50
        assert len(X[0]) == 5  # seq_len timesteps
        assert len(X[0][0]) == 1  # 1D values
        assert len(Y[0]) == 1

    def test_make_echo_data(self):
        X, Y = make_echo_data(n=30, seq_len=5, delay=2, vocab_size=4)
        assert len(X) == 30
        assert len(X[0]) == 5
        assert len(X[0][0]) == 4  # one-hot
        assert len(Y[0]) == 4  # one-hot target

    def test_make_sequence_copy_data(self):
        X, Y = make_sequence_copy_data(n=20, seq_len=4, vocab_size=3)
        assert len(X) == 20
        assert len(X[0]) == 4
        # Copy task: output == input
        for x, y in zip(X, Y):
            assert x == y

    def test_make_addition_data(self):
        X, Y = make_addition_data(n=50, max_val=10)
        assert len(X) == 50
        assert len(X[0]) == 2  # Two numbers
        assert len(X[0][0]) == 1  # 1D
        assert len(Y[0]) == 1

    def test_addition_data_normalized(self):
        X, Y = make_addition_data(n=50, max_val=10)
        for x in X:
            assert 0 <= x[0][0] <= 1.0
            assert 0 <= x[1][0] <= 1.0
        for y in Y:
            assert 0 <= y[0] <= 1.0


# ============================================================
# SequenceModel tests
# ============================================================

class TestSequenceModel:
    def test_creation(self):
        model = SequenceModel()
        assert model.layers == []

    def test_add_layers(self):
        model = SequenceModel()
        model.add(RNN(3, 5, rng=random.Random(42)))
        assert len(model.layers) == 1

    def test_forward(self):
        model = SequenceModel()
        model.add(RNN(3, 5, return_sequences=False, rng=random.Random(42)))
        seqs = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        out = model.forward(seqs)
        assert len(out) == 1
        assert len(out[0]) == 5

    def test_predict(self):
        model = SequenceModel()
        model.add(RNN(3, 5, rng=random.Random(42)))
        seqs = [[[1.0, 0.0, 0.0]]]
        out = model.predict(seqs)
        assert len(out) == 1

    def test_param_count(self):
        model = SequenceModel()
        model.add(RNN(3, 5, rng=random.Random(42)))
        model.add(TimeDistributed(5, 2, rng=random.Random(42)))
        total = model.param_count()
        assert total > 0

    def test_stacked_rnn(self):
        model = SequenceModel()
        model.add(RNN(3, 5, return_sequences=True, rng=random.Random(42)))
        model.add(RNN(5, 4, return_sequences=False, rng=random.Random(42)))
        seqs = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        out = model.forward(seqs)
        assert len(out[0]) == 4

    def test_lstm_model(self):
        model = SequenceModel()
        model.add(LSTMLayer(3, 5, return_sequences=False, rng=random.Random(42)))
        seqs = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        out = model.forward(seqs)
        assert len(out[0]) == 5

    def test_gru_model(self):
        model = SequenceModel()
        model.add(GRULayer(3, 5, return_sequences=False, rng=random.Random(42)))
        seqs = [[[1.0, 0.0, 0.0]]]
        out = model.forward(seqs)
        assert len(out[0]) == 5


# ============================================================
# Loss function tests
# ============================================================

class TestSequenceCrossEntropyLoss:
    def test_forward(self):
        loss_fn = SequenceCrossEntropyLoss()
        logits = [[[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]]]
        targets = [[0, 1]]
        loss = loss_fn.forward(logits, targets)
        assert loss > 0

    def test_perfect_prediction_low_loss(self):
        loss_fn = SequenceCrossEntropyLoss()
        logits = [[[10.0, -10.0], [-10.0, 10.0]]]
        targets = [[0, 1]]
        loss = loss_fn.forward(logits, targets)
        assert loss < 0.01

    def test_backward_shape(self):
        loss_fn = SequenceCrossEntropyLoss()
        logits = [[[2.0, 1.0, 0.0]]]
        targets = [[1]]
        loss_fn.forward(logits, targets)
        d = loss_fn.backward()
        assert len(d) == 1
        assert len(d[0]) == 1
        assert len(d[0][0]) == 3

    def test_backward_gradient_sums_near_zero(self):
        # For a single example, softmax grads sum to ~0
        loss_fn = SequenceCrossEntropyLoss()
        logits = [[[1.0, 2.0, 3.0]]]
        targets = [[1]]
        loss_fn.forward(logits, targets)
        d = loss_fn.backward()
        grad_sum = sum(d[0][0])
        assert abs(grad_sum) < 1e-10


class TestSequenceMSELoss:
    def test_forward_many_to_one(self):
        loss_fn = SequenceMSELoss()
        preds = [[1.0, 0.0]]
        targets = [[0.0, 1.0]]
        loss = loss_fn.forward(preds, targets)
        assert abs(loss - 1.0) < 1e-10  # (1^2 + 1^2) / 2

    def test_forward_zero_loss(self):
        loss_fn = SequenceMSELoss()
        preds = [[1.0, 2.0]]
        targets = [[1.0, 2.0]]
        loss = loss_fn.forward(preds, targets)
        assert abs(loss) < 1e-10

    def test_backward_shape(self):
        loss_fn = SequenceMSELoss()
        preds = [[1.0, 0.0]]
        targets = [[0.0, 1.0]]
        loss_fn.forward(preds, targets)
        d = loss_fn.backward()
        assert len(d) == 1
        assert len(d[0]) == 2


# ============================================================
# Training tests
# ============================================================

class TestTraining:
    def test_train_sine_rnn(self):
        """Train RNN on sine wave prediction."""
        X, Y = make_sine_data(n=50, seq_len=5, seed=42)
        model = SequenceModel()
        model.add(RNN(1, 8, return_sequences=False, rng=random.Random(42)))
        loss_fn = SequenceMSELoss()
        losses = train_sequence_model(model, X, Y, loss_fn, lr=0.01,
                                       epochs=30, seed=42)
        assert losses[-1] < losses[0]  # Loss decreased

    def test_train_sine_lstm(self):
        """Train LSTM on sine wave prediction."""
        X, Y = make_sine_data(n=50, seq_len=5, seed=42)
        model = SequenceModel()
        model.add(LSTMLayer(1, 8, return_sequences=False, rng=random.Random(42)))
        loss_fn = SequenceMSELoss()
        losses = train_sequence_model(model, X, Y, loss_fn, lr=0.01,
                                       epochs=30, seed=42)
        assert losses[-1] < losses[0]

    def test_train_sine_gru(self):
        """Train GRU on sine wave prediction."""
        X, Y = make_sine_data(n=50, seq_len=5, seed=42)
        model = SequenceModel()
        model.add(GRULayer(1, 8, return_sequences=False, rng=random.Random(42)))
        loss_fn = SequenceMSELoss()
        losses = train_sequence_model(model, X, Y, loss_fn, lr=0.01,
                                       epochs=30, seed=42)
        assert losses[-1] < losses[0]

    def test_train_addition_rnn(self):
        """Train RNN on addition task."""
        X, Y = make_addition_data(n=80, max_val=5, seed=42)
        model = SequenceModel()
        model.add(RNN(1, 10, return_sequences=False, rng=random.Random(42)))
        loss_fn = SequenceMSELoss()
        losses = train_sequence_model(model, X, Y, loss_fn, lr=0.005,
                                       epochs=50, seed=42)
        assert losses[-1] < losses[0]

    def test_train_with_batch_size(self):
        """Train with mini-batches."""
        X, Y = make_sine_data(n=40, seq_len=5, seed=42)
        model = SequenceModel()
        model.add(RNN(1, 6, return_sequences=False, rng=random.Random(42)))
        loss_fn = SequenceMSELoss()
        losses = train_sequence_model(model, X, Y, loss_fn, lr=0.01,
                                       epochs=20, batch_size=10, seed=42)
        assert len(losses) == 20

    def test_train_with_gradient_clipping(self):
        """Train with gradient clipping."""
        X, Y = make_sine_data(n=30, seq_len=5, seed=42)
        model = SequenceModel()
        model.add(RNN(1, 6, return_sequences=False, clip_grad=1.0,
                       rng=random.Random(42)))
        loss_fn = SequenceMSELoss()
        losses = train_sequence_model(model, X, Y, loss_fn, lr=0.01,
                                       epochs=20, seed=42)
        # Should not have NaN
        assert all(math.isfinite(l) for l in losses)

    def test_train_no_shuffle(self):
        X, Y = make_sine_data(n=20, seq_len=3, seed=42)
        model = SequenceModel()
        model.add(RNN(1, 4, rng=random.Random(42)))
        loss_fn = SequenceMSELoss()
        losses = train_sequence_model(model, X, Y, loss_fn, lr=0.01,
                                       epochs=10, shuffle=False, seed=42)
        assert len(losses) == 10


# ============================================================
# Integration tests
# ============================================================

class TestIntegration:
    def test_embedding_to_rnn(self):
        """Embedding -> RNN pipeline."""
        emb = Embedding(10, 4, rng=random.Random(42))
        rnn = RNN(4, 6, return_sequences=False, rng=random.Random(42))
        seqs = [[0, 3, 7, 1]]
        embedded = emb.forward(seqs)
        out = rnn.forward(embedded)
        assert len(out[0]) == 6

    def test_embedding_to_lstm(self):
        emb = Embedding(10, 4, rng=random.Random(42))
        lstm = LSTMLayer(4, 6, return_sequences=False, rng=random.Random(42))
        seqs = [[0, 3, 7]]
        embedded = emb.forward(seqs)
        out = lstm.forward(embedded)
        assert len(out[0]) == 6

    def test_stacked_lstm(self):
        """Stacked LSTM layers."""
        model = SequenceModel()
        model.add(LSTMLayer(2, 4, return_sequences=True, rng=random.Random(42)))
        model.add(LSTMLayer(4, 3, return_sequences=False, rng=random.Random(42)))
        seqs = [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]
        out = model.forward(seqs)
        assert len(out[0]) == 3

    def test_bidirectional_lstm_pipeline(self):
        bidir = Bidirectional(LSTMLayer, 3, 4, return_sequences=False, rng=random.Random(42))
        seqs = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        out = bidir.forward(seqs)
        assert len(out[0]) == 8

    def test_rnn_to_time_distributed(self):
        """RNN -> TimeDistributed for sequence labeling."""
        model = SequenceModel()
        model.add(RNN(3, 5, return_sequences=True, rng=random.Random(42)))
        model.add(TimeDistributed(5, 2, rng=random.Random(42)))
        seqs = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        out = model.forward(seqs)
        assert len(out[0]) == 2  # 2 timesteps
        assert len(out[0][0]) == 2  # 2 output features

    def test_full_pipeline_echo_task(self):
        """Full pipeline: data -> embed -> RNN -> train."""
        vocab = 4
        X, Y = make_echo_data(n=40, seq_len=5, delay=2, vocab_size=vocab, seed=42)
        model = SequenceModel()
        model.add(RNN(vocab, 8, return_sequences=False, rng=random.Random(42)))
        loss_fn = SequenceMSELoss()
        losses = train_sequence_model(model, X, Y, loss_fn, lr=0.01,
                                       epochs=30, seed=42)
        assert losses[-1] < losses[0]

    def test_sequence_model_deterministic(self):
        """Same seed produces same results."""
        def run():
            X, Y = make_sine_data(n=20, seq_len=3, seed=42)
            model = SequenceModel()
            model.add(RNN(1, 4, rng=random.Random(42)))
            loss_fn = SequenceMSELoss()
            return train_sequence_model(model, X, Y, loss_fn, lr=0.01,
                                         epochs=5, seed=42)
        losses1 = run()
        losses2 = run()
        assert losses1 == losses2

    def test_gru_vs_lstm_both_learn(self):
        """Both GRU and LSTM can learn the same task."""
        X, Y = make_sine_data(n=40, seq_len=5, seed=42)

        gru_model = SequenceModel()
        gru_model.add(GRULayer(1, 8, rng=random.Random(42)))
        gru_losses = train_sequence_model(gru_model, X, Y, SequenceMSELoss(),
                                           lr=0.01, epochs=20, seed=42)

        lstm_model = SequenceModel()
        lstm_model.add(LSTMLayer(1, 8, rng=random.Random(42)))
        lstm_losses = train_sequence_model(lstm_model, X, Y, SequenceMSELoss(),
                                            lr=0.01, epochs=20, seed=42)

        assert gru_losses[-1] < gru_losses[0]
        assert lstm_losses[-1] < lstm_losses[0]

    def test_bidirectional_learns(self):
        """Bidirectional RNN can learn."""
        X, Y = make_sine_data(n=40, seq_len=5, seed=42)
        bidir = Bidirectional(RNN, 1, 4, return_sequences=False, rng=random.Random(42))
        model = SequenceModel()
        model.add(bidir)
        loss_fn = SequenceMSELoss()
        losses = train_sequence_model(model, X, Y, loss_fn, lr=0.005,
                                       epochs=20, seed=42)
        assert losses[-1] < losses[0]

    def test_embedding_rnn_language_model(self):
        """Embedding -> LSTM -> TimeDistributed for language modeling."""
        emb = Embedding(5, 4, rng=random.Random(42))
        model = SequenceModel()
        model.add(LSTMLayer(4, 8, return_sequences=True, rng=random.Random(42)))
        model.add(TimeDistributed(8, 5, rng=random.Random(42)))

        # Simple integer sequences
        int_seqs = [[0, 1, 2, 3], [1, 2, 3, 4]]
        embedded = emb.forward(int_seqs)
        out = model.forward(embedded)
        assert len(out) == 2
        assert len(out[0]) == 4  # 4 timesteps
        assert len(out[0][0]) == 5  # vocab_size output

    def test_text_generation(self):
        """Test generate_text function."""
        emb = Embedding(5, 4, rng=random.Random(42))
        model = SequenceModel()
        model.add(RNN(4, 8, return_sequences=True, rng=random.Random(42)))
        model.add(TimeDistributed(8, 5, rng=random.Random(42)))

        generated = generate_text(model, emb, [0, 1], length=5, vocab_size=5)
        assert len(generated) == 7  # 2 start + 5 generated
        assert all(0 <= idx < 5 for idx in generated)


# ============================================================
# Edge case tests
# ============================================================

class TestEdgeCases:
    def test_single_timestep(self):
        layer = RNN(3, 5, rng=random.Random(42))
        out = layer.forward([[[1.0, 0.0, 0.0]]])
        assert len(out[0]) == 5

    def test_single_timestep_lstm(self):
        layer = LSTMLayer(3, 5, rng=random.Random(42))
        out = layer.forward([[[1.0, 0.0, 0.0]]])
        assert len(out[0]) == 5

    def test_single_timestep_gru(self):
        layer = GRULayer(3, 5, rng=random.Random(42))
        out = layer.forward([[[1.0, 0.0, 0.0]]])
        assert len(out[0]) == 5

    def test_large_hidden_size(self):
        layer = RNN(2, 32, rng=random.Random(42))
        out = layer.forward([[[1.0, 0.0], [0.0, 1.0]]])
        assert len(out[0]) == 32

    def test_zero_input(self):
        layer = RNN(3, 5, rng=random.Random(42))
        out = layer.forward([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        # Should still produce output (bias drives it)
        assert len(out[0]) == 5

    def test_pad_empty_sequence(self):
        seqs = [[], [1, 2]]
        padded = pad_sequences(seqs)
        assert len(padded[0]) == 2

    def test_one_hot_single(self):
        result = one_hot_encode_sequence([0], 1)
        assert result == [[1.0]]

    def test_create_sequence_pairs_short(self):
        inputs, targets = create_sequence_pairs([1, 2], 1)
        assert inputs == [[1]]
        assert targets == [2]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
