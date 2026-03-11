"""Tests for C142: Transformer."""

import math
import random
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from transformer import (
    Tensor, tensor_slice_rows, tensor_slice_cols, tensor_concat_cols,
    tensor_concat_rows, tensor_outer,
    LayerNorm, Embedding, PositionalEncoding,
    scaled_dot_product_attention, scaled_dot_product_attention_backward,
    causal_mask, padding_mask,
    MultiHeadAttention, FeedForward,
    EncoderBlock, DecoderBlock,
    TransformerEncoder, TransformerDecoder,
    Transformer, TransformerClassifier,
    TransformerAdam, train_transformer, train_classifier,
    CrossEntropyLoss, softmax
)


# ============================================================
# Tensor utility tests
# ============================================================

class TestTensorUtils:
    def test_slice_rows(self):
        t = Tensor([[1, 2], [3, 4], [5, 6]])
        s = tensor_slice_rows(t, 0, 2)
        assert s.shape == (2, 2)
        assert s.data[0] == [1, 2]
        assert s.data[1] == [3, 4]

    def test_slice_cols(self):
        t = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        s = tensor_slice_cols(t, 1, 3)
        assert s.shape == (2, 2)
        assert s.data[0] == [2, 3]
        assert s.data[1] == [6, 7]

    def test_concat_cols(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        c = tensor_concat_cols([a, b])
        assert c.shape == (2, 4)
        assert c.data[0] == [1, 2, 5, 6]

    def test_concat_rows(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6]])
        c = tensor_concat_rows([a, b])
        assert c.shape == (3, 2)
        assert c.data[2] == [5, 6]

    def test_outer_product(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0])
        c = tensor_outer(a, b)
        assert c.shape == (3, 2)
        assert c.data[0] == [4.0, 5.0]
        assert c.data[1] == [8.0, 10.0]
        assert c.data[2] == [12.0, 15.0]

    def test_slice_roundtrip(self):
        """Slice then concat should recover original."""
        t = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        a = tensor_slice_cols(t, 0, 2)
        b = tensor_slice_cols(t, 2, 4)
        c = tensor_concat_cols([a, b])
        assert c.data == t.data


# ============================================================
# Layer Normalization tests
# ============================================================

class TestLayerNorm:
    def test_1d_normalization(self):
        ln = LayerNorm(4)
        x = Tensor([2.0, 4.0, 6.0, 8.0])
        out = ln.forward(x)
        # After norm: mean should be ~0, std ~1 (before gamma/beta)
        vals = out.data
        mean_out = sum(vals) / len(vals)
        assert abs(mean_out) < 0.1  # gamma=1, beta=0 so close to normalized

    def test_2d_normalization(self):
        ln = LayerNorm(3)
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = ln.forward(x)
        assert out.shape == (2, 3)
        # Each row should be normalized
        for i in range(2):
            row = out.data[i]
            mean_r = sum(row) / 3
            assert abs(mean_r) < 0.1

    def test_backward_shapes(self):
        ln = LayerNorm(4)
        x = Tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        out = ln.forward(x)
        grad = Tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        grad_input = ln.backward(grad)
        assert grad_input.shape == (2, 4)
        assert ln.grad_gamma is not None
        assert ln.grad_beta is not None

    def test_get_params(self):
        ln = LayerNorm(4)
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        ln.forward(x)
        ln.backward(Tensor([1.0, 0.0, 0.0, 0.0]))
        params = ln.get_params()
        assert len(params) == 2
        assert params[0][2] == 'gamma'
        assert params[1][2] == 'beta'

    def test_scale_and_shift(self):
        """Custom gamma and beta should affect output."""
        ln = LayerNorm(3)
        ln.gamma = Tensor([2.0, 2.0, 2.0])
        ln.beta = Tensor([1.0, 1.0, 1.0])
        x = Tensor([1.0, 2.0, 3.0])
        out = ln.forward(x)
        # Output should be 2 * normalized + 1
        mean_out = sum(out.data) / 3
        assert abs(mean_out - 1.0) < 0.1  # beta shifts mean to ~1

    def test_identity_input(self):
        """Constant input should produce zeros (normalized) + beta."""
        ln = LayerNorm(4, eps=1e-6)
        x = Tensor([5.0, 5.0, 5.0, 5.0])
        out = ln.forward(x)
        # All same -> variance ~0, normalized -> 0, output -> beta (0)
        for v in out.data:
            assert abs(v) < 0.01


# ============================================================
# Embedding tests
# ============================================================

class TestEmbedding:
    def test_basic_lookup(self):
        rng = random.Random(42)
        emb = Embedding(10, 4, rng=rng)
        out = emb.forward([0, 3, 7])
        assert out.shape == (3, 4)
        # Row 0 should match embedding weight[0]
        assert out.data[0] == emb.weight.data[0]
        assert out.data[1] == emb.weight.data[3]

    def test_backward_accumulates(self):
        emb = Embedding(5, 3)
        out = emb.forward([1, 1, 2])  # token 1 appears twice
        grad = Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        emb.backward(grad)
        # Token 1 gradient should be sum of first two grad rows
        assert emb.grad_weight.data[1] == [1.0, 1.0, 0.0]
        assert emb.grad_weight.data[2] == [0.0, 0.0, 1.0]

    def test_get_params(self):
        emb = Embedding(5, 3)
        emb.forward([0])
        emb.backward(Tensor([[1.0, 0.0, 0.0]]))
        params = emb.get_params()
        assert len(params) == 1
        assert params[0][2] == 'embedding'

    def test_different_vocab_sizes(self):
        for vs in [2, 10, 50]:
            emb = Embedding(vs, 8)
            assert emb.weight.shape == (vs, 8)


# ============================================================
# Positional Encoding tests
# ============================================================

class TestPositionalEncoding:
    def test_shape(self):
        pe = PositionalEncoding(8, max_len=100)
        out = pe.forward(5)
        assert out.shape == (5, 8)

    def test_first_position_zero(self):
        pe = PositionalEncoding(4)
        out = pe.forward(1)
        # pos=0: sin(0)=0, cos(0)=1
        assert abs(out.data[0][0] - 0.0) < 1e-10  # sin(0)
        assert abs(out.data[0][1] - 1.0) < 1e-10  # cos(0)

    def test_different_positions(self):
        pe = PositionalEncoding(4)
        out = pe.forward(3)
        # Different positions should have different encodings
        assert out.data[0] != out.data[1]
        assert out.data[1] != out.data[2]

    def test_caching(self):
        pe = PositionalEncoding(4)
        out1 = pe.forward(3)
        out2 = pe.forward(3)
        # Should be equal values but different objects
        assert out1.data == out2.data
        assert out1 is not out2

    def test_values_bounded(self):
        pe = PositionalEncoding(16)
        out = pe.forward(10)
        for row in out.data:
            for v in row:
                assert -1.0 <= v <= 1.0

    def test_sin_cos_alternation(self):
        """Even dims are sin, odd dims are cos."""
        pe = PositionalEncoding(6)
        out = pe.forward(5)
        # pos=1, dim=0: sin(1/1) = sin(1)
        assert abs(out.data[1][0] - math.sin(1.0)) < 1e-10
        assert abs(out.data[1][1] - math.cos(1.0)) < 1e-10


# ============================================================
# Scaled Dot-Product Attention tests
# ============================================================

class TestScaledDotProductAttention:
    def test_basic_attention(self):
        Q = Tensor([[1.0, 0.0], [0.0, 1.0]])
        K = Tensor([[1.0, 0.0], [0.0, 1.0]])
        V = Tensor([[1.0, 2.0], [3.0, 4.0]])
        output, attn = scaled_dot_product_attention(Q, K, V)
        assert output.shape == (2, 2)
        assert attn.shape == (2, 2)
        # Q[0] aligns with K[0], so attn[0] should weight V[0] more
        assert attn.data[0][0] > attn.data[0][1]

    def test_attention_weights_sum_to_one(self):
        Q = Tensor([[1.0, 2.0], [3.0, 4.0]])
        K = Tensor([[0.5, 1.0], [1.5, 2.0], [0.1, 0.2]])
        V = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        _, attn = scaled_dot_product_attention(Q, K, V)
        assert attn.shape == (2, 3)
        for row in attn.data:
            assert abs(sum(row) - 1.0) < 1e-10

    def test_causal_mask_effect(self):
        Q = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        K = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        V = Tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        mask = causal_mask(3)
        _, attn = scaled_dot_product_attention(Q, K, V, mask=mask)
        # First token can only attend to itself
        assert abs(attn.data[0][0] - 1.0) < 1e-5
        assert abs(attn.data[0][1]) < 1e-5
        assert abs(attn.data[0][2]) < 1e-5

    def test_output_is_weighted_sum(self):
        """With uniform attention, output should be mean of values."""
        Q = Tensor([[0.0, 0.0]])
        K = Tensor([[0.0, 0.0], [0.0, 0.0]])
        V = Tensor([[2.0, 4.0], [6.0, 8.0]])
        output, attn = scaled_dot_product_attention(Q, K, V)
        # Uniform attention -> mean
        assert abs(output.data[0][0] - 4.0) < 1e-5
        assert abs(output.data[0][1] - 6.0) < 1e-5

    def test_single_query_key(self):
        Q = Tensor([[1.0, 0.0]])
        K = Tensor([[1.0, 0.0]])
        V = Tensor([[5.0, 10.0]])
        output, attn = scaled_dot_product_attention(Q, K, V)
        assert abs(attn.data[0][0] - 1.0) < 1e-10
        assert abs(output.data[0][0] - 5.0) < 1e-5

    def test_backward_shapes(self):
        Q = Tensor([[1.0, 0.0], [0.0, 1.0]])
        K = Tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        V = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        output, attn = scaled_dot_product_attention(Q, K, V)
        grad_out = Tensor([[1.0, 0.0], [0.0, 1.0]])
        gQ, gK, gV = scaled_dot_product_attention_backward(grad_out, Q, K, V, attn)
        assert gQ.shape == (2, 2)
        assert gK.shape == (3, 2)
        assert gV.shape == (3, 2)


# ============================================================
# Causal Mask tests
# ============================================================

class TestCausalMask:
    def test_shape(self):
        m = causal_mask(4)
        assert m.shape == (4, 4)

    def test_lower_triangular(self):
        m = causal_mask(3)
        assert m.data[0][0] == 0.0
        assert m.data[0][1] == -1e9
        assert m.data[1][0] == 0.0
        assert m.data[1][1] == 0.0
        assert m.data[1][2] == -1e9
        assert m.data[2][2] == 0.0

    def test_single_token(self):
        m = causal_mask(1)
        assert m.data == [[0.0]]


# ============================================================
# Padding Mask tests
# ============================================================

class TestPaddingMask:
    def test_basic(self):
        m = padding_mask([3, 2], 4)
        assert m.shape == (2, 4)
        assert m.data[0] == [0.0, 0.0, 0.0, -1e9]
        assert m.data[1] == [0.0, 0.0, -1e9, -1e9]

    def test_full_length(self):
        m = padding_mask([5], 5)
        assert m.data[0] == [0.0, 0.0, 0.0, 0.0, 0.0]


# ============================================================
# Multi-Head Attention tests
# ============================================================

class TestMultiHeadAttention:
    def test_basic_forward(self):
        rng = random.Random(42)
        mha = MultiHeadAttention(8, 2, rng=rng)
        x = Tensor([[1.0] * 8, [2.0] * 8, [3.0] * 8])
        out = mha.forward(x, x, x)
        assert out.shape == (3, 8)

    def test_self_attention(self):
        rng = random.Random(42)
        mha = MultiHeadAttention(4, 2, rng=rng)
        x = Tensor.random_normal((4, 4), rng=rng)
        out = mha.forward(x, x, x)
        assert out.shape == (4, 4)

    def test_cross_attention(self):
        rng = random.Random(42)
        mha = MultiHeadAttention(4, 2, rng=rng)
        q = Tensor.random_normal((3, 4), rng=rng)
        kv = Tensor.random_normal((5, 4), rng=rng)
        out = mha.forward(q, kv, kv)
        assert out.shape == (3, 4)

    def test_with_mask(self):
        rng = random.Random(42)
        mha = MultiHeadAttention(4, 2, rng=rng)
        x = Tensor.random_normal((3, 4), rng=rng)
        mask = causal_mask(3)
        out = mha.forward(x, x, x, mask=mask)
        assert out.shape == (3, 4)

    def test_backward(self):
        rng = random.Random(42)
        mha = MultiHeadAttention(4, 2, rng=rng)
        x = Tensor.random_normal((3, 4), rng=rng)
        out = mha.forward(x, x, x)
        grad = Tensor.ones((3, 4))
        gQ, gK, gV = mha.backward(grad)
        assert gQ.shape == (3, 4)
        assert gK.shape == (3, 4)

    def test_attention_weights(self):
        rng = random.Random(42)
        mha = MultiHeadAttention(4, 2, rng=rng)
        x = Tensor.random_normal((3, 4), rng=rng)
        mha.forward(x, x, x)
        weights = mha.get_attention_weights()
        assert len(weights) == 2  # 2 heads
        for w in weights:
            assert w.shape == (3, 3)
            # Each row sums to 1
            for row in w.data:
                assert abs(sum(row) - 1.0) < 1e-10

    def test_get_params(self):
        mha = MultiHeadAttention(4, 2)
        params = mha.get_params()
        # W_Q, W_K, W_V, W_O each has 1 weight (no bias)
        assert len(params) == 4

    def test_num_heads_1(self):
        rng = random.Random(42)
        mha = MultiHeadAttention(4, 1, rng=rng)
        x = Tensor.random_normal((3, 4), rng=rng)
        out = mha.forward(x, x, x)
        assert out.shape == (3, 4)

    def test_many_heads(self):
        rng = random.Random(42)
        mha = MultiHeadAttention(8, 4, rng=rng)
        x = Tensor.random_normal((2, 8), rng=rng)
        out = mha.forward(x, x, x)
        assert out.shape == (2, 8)


# ============================================================
# Feed-Forward Network tests
# ============================================================

class TestFeedForward:
    def test_basic(self):
        rng = random.Random(42)
        ff = FeedForward(4, 8, rng=rng)
        x = Tensor.random_normal((3, 4), rng=rng)
        out = ff.forward(x)
        assert out.shape == (3, 4)

    def test_backward(self):
        rng = random.Random(42)
        ff = FeedForward(4, 8, rng=rng)
        x = Tensor.random_normal((3, 4), rng=rng)
        out = ff.forward(x)
        grad = Tensor.ones((3, 4))
        grad_in = ff.backward(grad)
        assert grad_in.shape == (3, 4)

    def test_get_params(self):
        ff = FeedForward(4, 8)
        params = ff.get_params()
        # Two Dense layers: 2 weights + 2 biases = 4
        assert len(params) == 4

    def test_d_ff_expansion(self):
        """FFN should expand to d_ff then compress back."""
        ff = FeedForward(4, 16)
        assert ff.linear1.output_size == 16
        assert ff.linear2.output_size == 4


# ============================================================
# Encoder Block tests
# ============================================================

class TestEncoderBlock:
    def test_basic(self):
        rng = random.Random(42)
        block = EncoderBlock(8, 2, 16, rng=rng)
        x = Tensor.random_normal((4, 8), rng=rng)
        out = block.forward(x)
        assert out.shape == (4, 8)

    def test_with_mask(self):
        rng = random.Random(42)
        block = EncoderBlock(4, 2, 8, rng=rng)
        x = Tensor.random_normal((3, 4), rng=rng)
        mask = causal_mask(3)
        out = block.forward(x, mask=mask)
        assert out.shape == (3, 4)

    def test_backward(self):
        rng = random.Random(42)
        block = EncoderBlock(4, 2, 8, rng=rng)
        x = Tensor.random_normal((3, 4), rng=rng)
        out = block.forward(x)
        grad = Tensor.ones((3, 4))
        grad_in = block.backward(grad)
        assert grad_in.shape == (3, 4)

    def test_get_params(self):
        block = EncoderBlock(4, 2, 8)
        params = block.get_params()
        # MHA: 4 params + norm1: 2 + FFN: 4 + norm2: 2 = 12
        assert len(params) == 12

    def test_residual_connection(self):
        """Output should differ from input (transformation applied)."""
        rng = random.Random(42)
        block = EncoderBlock(4, 2, 8, rng=rng)
        x = Tensor.ones((2, 4))
        out = block.forward(x)
        # Should be different from input due to attention + FFN
        differs = False
        for i in range(2):
            for j in range(4):
                if abs(out.data[i][j] - x.data[i][j]) > 1e-6:
                    differs = True
        assert differs


# ============================================================
# Decoder Block tests
# ============================================================

class TestDecoderBlock:
    def test_basic(self):
        rng = random.Random(42)
        block = DecoderBlock(8, 2, 16, rng=rng)
        x = Tensor.random_normal((3, 8), rng=rng)
        enc_out = Tensor.random_normal((5, 8), rng=rng)
        out = block.forward(x, enc_out)
        assert out.shape == (3, 8)

    def test_with_masks(self):
        rng = random.Random(42)
        block = DecoderBlock(4, 2, 8, rng=rng)
        x = Tensor.random_normal((3, 4), rng=rng)
        enc_out = Tensor.random_normal((4, 4), rng=rng)
        tgt_mask = causal_mask(3)
        out = block.forward(x, enc_out, tgt_mask=tgt_mask)
        assert out.shape == (3, 4)

    def test_backward(self):
        rng = random.Random(42)
        block = DecoderBlock(4, 2, 8, rng=rng)
        x = Tensor.random_normal((3, 4), rng=rng)
        enc_out = Tensor.random_normal((4, 4), rng=rng)
        out = block.forward(x, enc_out)
        grad = Tensor.ones((3, 4))
        grad_input, grad_encoder = block.backward(grad)
        assert grad_input.shape == (3, 4)
        assert grad_encoder.shape == (4, 4)

    def test_get_params(self):
        block = DecoderBlock(4, 2, 8)
        params = block.get_params()
        # self_attn: 4 + norm1: 2 + cross_attn: 4 + norm2: 2 + FFN: 4 + norm3: 2 = 18
        assert len(params) == 18


# ============================================================
# Transformer Encoder tests
# ============================================================

class TestTransformerEncoder:
    def test_single_layer(self):
        rng = random.Random(42)
        enc = TransformerEncoder(1, 4, 2, 8, rng=rng)
        x = Tensor.random_normal((3, 4), rng=rng)
        out = enc.forward(x)
        assert out.shape == (3, 4)

    def test_multi_layer(self):
        rng = random.Random(42)
        enc = TransformerEncoder(3, 4, 2, 8, rng=rng)
        x = Tensor.random_normal((3, 4), rng=rng)
        out = enc.forward(x)
        assert out.shape == (3, 4)

    def test_backward(self):
        rng = random.Random(42)
        enc = TransformerEncoder(2, 4, 2, 8, rng=rng)
        x = Tensor.random_normal((3, 4), rng=rng)
        out = enc.forward(x)
        grad = Tensor.ones((3, 4))
        grad_in = enc.backward(grad)
        assert grad_in.shape == (3, 4)


# ============================================================
# Transformer Decoder tests
# ============================================================

class TestTransformerDecoder:
    def test_single_layer(self):
        rng = random.Random(42)
        dec = TransformerDecoder(1, 4, 2, 8, rng=rng)
        x = Tensor.random_normal((3, 4), rng=rng)
        enc_out = Tensor.random_normal((5, 4), rng=rng)
        out = dec.forward(x, enc_out)
        assert out.shape == (3, 4)

    def test_multi_layer(self):
        rng = random.Random(42)
        dec = TransformerDecoder(2, 4, 2, 8, rng=rng)
        x = Tensor.random_normal((3, 4), rng=rng)
        enc_out = Tensor.random_normal((5, 4), rng=rng)
        out = dec.forward(x, enc_out)
        assert out.shape == (3, 4)

    def test_backward(self):
        rng = random.Random(42)
        dec = TransformerDecoder(2, 4, 2, 8, rng=rng)
        x = Tensor.random_normal((3, 4), rng=rng)
        enc_out = Tensor.random_normal((5, 4), rng=rng)
        out = dec.forward(x, enc_out)
        grad = Tensor.ones((3, 4))
        grad_in, grad_enc = dec.backward(grad)
        assert grad_in.shape == (3, 4)
        assert grad_enc.shape == (5, 4)


# ============================================================
# Full Transformer tests
# ============================================================

class TestTransformer:
    def test_basic_forward(self):
        rng = random.Random(42)
        model = Transformer(10, 10, d_model=8, num_heads=2, d_ff=16,
                            num_encoder_layers=1, num_decoder_layers=1, rng=rng)
        logits = model.forward([1, 2, 3], [4, 5])
        assert logits.shape == (2, 10)  # (tgt_len, tgt_vocab)

    def test_backward(self):
        rng = random.Random(42)
        model = Transformer(10, 10, d_model=8, num_heads=2, d_ff=16,
                            num_encoder_layers=1, num_decoder_layers=1, rng=rng)
        logits = model.forward([1, 2, 3], [4, 5])
        loss_fn = CrossEntropyLoss()
        loss = loss_fn.forward(logits, [6, 7])
        grad = loss_fn.backward(logits, [6, 7])
        model.backward(grad)  # Should not error

    def test_greedy_decode(self):
        rng = random.Random(42)
        model = Transformer(10, 10, d_model=8, num_heads=2, d_ff=16,
                            num_encoder_layers=1, num_decoder_layers=1, rng=rng)
        tokens = model.greedy_decode([1, 2, 3], max_len=5, start_token=1, end_token=9)
        assert isinstance(tokens, list)
        assert tokens[0] == 1  # starts with start_token
        assert len(tokens) <= 7  # max_len + start token

    def test_get_params(self):
        model = Transformer(10, 10, d_model=4, num_heads=2, d_ff=8,
                            num_encoder_layers=1, num_decoder_layers=1)
        params = model.get_params()
        assert len(params) > 0
        # Should have: 2 embeddings + encoder params + decoder params + output proj
        # Each param is (tensor, grad, name)
        for p in params:
            assert len(p) == 3

    def test_different_src_tgt_vocab(self):
        rng = random.Random(42)
        model = Transformer(20, 15, d_model=8, num_heads=2, d_ff=16,
                            num_encoder_layers=1, num_decoder_layers=1, rng=rng)
        logits = model.forward([1, 5, 10], [0, 3])
        assert logits.shape == (2, 15)

    def test_auto_causal_mask(self):
        """Forward should auto-generate causal mask for target."""
        rng = random.Random(42)
        model = Transformer(10, 10, d_model=4, num_heads=2, d_ff=8,
                            num_encoder_layers=1, num_decoder_layers=1, rng=rng)
        logits = model.forward([1, 2], [3, 4, 5])
        assert logits.shape == (3, 10)

    def test_multiple_encoder_decoder_layers(self):
        rng = random.Random(42)
        model = Transformer(10, 10, d_model=8, num_heads=2, d_ff=16,
                            num_encoder_layers=3, num_decoder_layers=3, rng=rng)
        logits = model.forward([1, 2, 3], [4, 5])
        assert logits.shape == (2, 10)


# ============================================================
# Transformer Classifier tests
# ============================================================

class TestTransformerClassifier:
    def test_basic_forward(self):
        rng = random.Random(42)
        model = TransformerClassifier(20, 3, d_model=8, num_heads=2,
                                      d_ff=16, num_layers=1, rng=rng)
        logits = model.forward([1, 2, 3, 4])
        assert logits.shape == (3,)

    def test_backward(self):
        rng = random.Random(42)
        model = TransformerClassifier(20, 3, d_model=8, num_heads=2,
                                      d_ff=16, num_layers=1, rng=rng)
        logits = model.forward([1, 2, 3])
        loss_fn = CrossEntropyLoss()
        loss = loss_fn.forward(logits, 1)
        grad = loss_fn.backward(logits, 1)
        model.backward(grad)  # Should not error

    def test_get_params(self):
        model = TransformerClassifier(10, 3, d_model=4, num_heads=2,
                                      d_ff=8, num_layers=1)
        params = model.get_params()
        assert len(params) > 0

    def test_multi_layer(self):
        rng = random.Random(42)
        model = TransformerClassifier(10, 5, d_model=8, num_heads=2,
                                      d_ff=16, num_layers=3, rng=rng)
        logits = model.forward([1, 2, 3])
        assert logits.shape == (5,)


# ============================================================
# Adam Optimizer tests
# ============================================================

class TestTransformerAdam:
    def test_basic_step(self):
        rng = random.Random(42)
        model = TransformerClassifier(10, 3, d_model=4, num_heads=2,
                                      d_ff=8, num_layers=1, rng=rng)
        opt = TransformerAdam(model.get_params, lr=0.01)

        logits = model.forward([1, 2, 3])
        loss_fn = CrossEntropyLoss()
        grad = loss_fn.backward(logits, 1)
        model.backward(grad)

        # Save a param value
        old_val = model.classifier.weights.data[0][0]
        opt.step()
        new_val = model.classifier.weights.data[0][0]
        assert old_val != new_val  # Should have changed

    def test_warmup(self):
        rng = random.Random(42)
        model = TransformerClassifier(10, 3, d_model=4, num_heads=2,
                                      d_ff=8, num_layers=1, rng=rng)
        opt = TransformerAdam(model.get_params, lr=0.01, warmup_steps=5)

        for i in range(3):
            logits = model.forward([1, 2])
            loss_fn = CrossEntropyLoss()
            grad = loss_fn.backward(logits, 0)
            model.backward(grad)
            opt.step()
        # Should have done 3 warmup steps (lr scaled)
        assert opt.t == 3

    def test_weight_decay(self):
        rng = random.Random(42)
        model = TransformerClassifier(10, 3, d_model=4, num_heads=2,
                                      d_ff=8, num_layers=1, rng=rng)
        opt = TransformerAdam(model.get_params, lr=0.01, weight_decay=0.01)

        logits = model.forward([1, 2])
        loss_fn = CrossEntropyLoss()
        grad = loss_fn.backward(logits, 0)
        model.backward(grad)
        opt.step()
        # Should work without error
        assert opt.t == 1


# ============================================================
# Training tests
# ============================================================

class TestTraining:
    def test_train_transformer_loss_decreases(self):
        rng = random.Random(42)
        model = Transformer(5, 5, d_model=8, num_heads=2, d_ff=16,
                            num_encoder_layers=1, num_decoder_layers=1, rng=rng)
        loss_fn = CrossEntropyLoss()
        opt = TransformerAdam(model.get_params, lr=0.005)

        # Simple copy task: src=[1,2,3], tgt=[1,2], labels=[2,3]
        data = [([1, 2, 3], [1, 2], [2, 3])]
        history = train_transformer(model, data, loss_fn, opt, epochs=20)
        assert history[-1] < history[0]  # Loss decreased

    def test_train_classifier_loss_decreases(self):
        rng = random.Random(42)
        model = TransformerClassifier(10, 3, d_model=8, num_heads=2,
                                      d_ff=16, num_layers=1, rng=rng)
        loss_fn = CrossEntropyLoss()
        opt = TransformerAdam(model.get_params, lr=0.005)

        data = [([1, 2, 3], 0), ([4, 5, 6], 1), ([7, 8, 9], 2)]
        history = train_classifier(model, data, loss_fn, opt, epochs=20)
        assert history[-1] < history[0]

    def test_overfit_single_example(self):
        """Model should be able to overfit a single example."""
        rng = random.Random(42)
        model = Transformer(5, 5, d_model=8, num_heads=2, d_ff=16,
                            num_encoder_layers=1, num_decoder_layers=1, rng=rng)
        loss_fn = CrossEntropyLoss()
        opt = TransformerAdam(model.get_params, lr=0.01)

        data = [([1, 2], [1], [2])]
        history = train_transformer(model, data, loss_fn, opt, epochs=50)
        # Should get loss quite low
        assert history[-1] < 1.0

    def test_classifier_overfit(self):
        rng = random.Random(42)
        model = TransformerClassifier(10, 2, d_model=8, num_heads=2,
                                      d_ff=16, num_layers=1, rng=rng)
        loss_fn = CrossEntropyLoss()
        opt = TransformerAdam(model.get_params, lr=0.01)

        data = [([1, 2, 3], 0), ([7, 8, 9], 1)]
        history = train_classifier(model, data, loss_fn, opt, epochs=50)
        assert history[-1] < 0.5


# ============================================================
# Integration tests
# ============================================================

class TestIntegration:
    def test_copy_task(self):
        """Transformer should learn to copy short sequences."""
        rng = random.Random(42)
        model = Transformer(6, 6, d_model=8, num_heads=2, d_ff=16,
                            num_encoder_layers=1, num_decoder_layers=1, rng=rng)
        loss_fn = CrossEntropyLoss()
        opt = TransformerAdam(model.get_params, lr=0.01)

        # Copy task: input [1,2,3] -> output [1,2,3]
        data = [
            ([1, 2, 3], [0, 1, 2], [1, 2, 3]),  # shifted target with start token 0
        ]
        history = train_transformer(model, data, loss_fn, opt, epochs=80)
        assert history[-1] < history[0]

    def test_encoder_only_classification(self):
        """Classifier should distinguish token patterns."""
        rng = random.Random(42)
        model = TransformerClassifier(10, 2, d_model=8, num_heads=2,
                                      d_ff=16, num_layers=2, rng=rng)
        loss_fn = CrossEntropyLoss()
        opt = TransformerAdam(model.get_params, lr=0.005)

        data = [
            ([1, 1, 1], 0),
            ([2, 2, 2], 1),
        ]
        history = train_classifier(model, data, loss_fn, opt, epochs=60)
        assert history[-1] < 0.5

    def test_greedy_decode_after_training(self):
        """After training, greedy decode should produce sensible output."""
        rng = random.Random(42)
        model = Transformer(5, 5, d_model=8, num_heads=2, d_ff=16,
                            num_encoder_layers=1, num_decoder_layers=1, rng=rng)
        loss_fn = CrossEntropyLoss()
        opt = TransformerAdam(model.get_params, lr=0.01)

        # Train on: [1,2] -> [3]
        data = [([1, 2], [0], [3])]
        train_transformer(model, data, loss_fn, opt, epochs=100)

        tokens = model.greedy_decode([1, 2], max_len=3, start_token=0, end_token=4)
        assert isinstance(tokens, list)
        assert tokens[0] == 0

    def test_full_pipeline(self):
        """End-to-end: build model, train, evaluate."""
        rng = random.Random(42)
        model = Transformer(8, 8, d_model=8, num_heads=2, d_ff=16,
                            num_encoder_layers=1, num_decoder_layers=1, rng=rng)
        loss_fn = CrossEntropyLoss()
        opt = TransformerAdam(model.get_params, lr=0.01)

        # Training data
        data = [
            ([1, 2, 3], [0, 1], [1, 2]),
            ([4, 5, 6], [0, 4], [4, 5]),
        ]
        h = train_transformer(model, data, loss_fn, opt, epochs=30)
        assert len(h) == 30
        assert h[-1] < h[0]

    def test_attention_pattern_causal(self):
        """With causal mask, future positions should get zero attention."""
        rng = random.Random(42)
        mha = MultiHeadAttention(4, 1, rng=rng)
        x = Tensor.random_normal((4, 4), rng=rng)
        mask = causal_mask(4)
        mha.forward(x, x, x, mask=mask)
        weights = mha.get_attention_weights()[0]
        # Upper triangle should be ~0
        for i in range(4):
            for j in range(i + 1, 4):
                assert weights.data[i][j] < 1e-5

    def test_position_encoding_makes_difference(self):
        """Same tokens at different positions should produce different outputs."""
        rng = random.Random(42)
        model = TransformerClassifier(5, 3, d_model=8, num_heads=2,
                                      d_ff=16, num_layers=1, rng=rng)
        # Same token repeated
        out1 = model.forward([1, 2, 3])
        out2 = model.forward([3, 2, 1])
        # Outputs should differ due to position encoding
        differs = any(abs(out1.data[i] - out2.data[i]) > 1e-6 for i in range(3))
        assert differs


# ============================================================
# Edge case tests
# ============================================================

class TestEdgeCases:
    def test_single_token_input(self):
        rng = random.Random(42)
        model = Transformer(10, 10, d_model=4, num_heads=2, d_ff=8,
                            num_encoder_layers=1, num_decoder_layers=1, rng=rng)
        logits = model.forward([1], [2])
        assert logits.shape == (1, 10)

    def test_single_head(self):
        rng = random.Random(42)
        mha = MultiHeadAttention(4, 1, rng=rng)
        x = Tensor.random_normal((3, 4), rng=rng)
        out = mha.forward(x, x, x)
        assert out.shape == (3, 4)

    def test_large_d_model(self):
        rng = random.Random(42)
        mha = MultiHeadAttention(16, 4, rng=rng)
        x = Tensor.random_normal((2, 16), rng=rng)
        out = mha.forward(x, x, x)
        assert out.shape == (2, 16)

    def test_layer_norm_1d(self):
        ln = LayerNorm(3)
        x = Tensor([1.0, 2.0, 3.0])
        out = ln.forward(x)
        grad = Tensor([1.0, 0.0, 0.0])
        grad_in = ln.backward(grad)
        assert len(grad_in.data) == 3

    def test_embedding_all_same_token(self):
        emb = Embedding(5, 3)
        out = emb.forward([2, 2, 2])
        assert out.data[0] == out.data[1] == out.data[2]

    def test_causal_mask_single(self):
        m = causal_mask(1)
        assert m.data == [[0.0]]

    def test_greedy_decode_max_len(self):
        rng = random.Random(42)
        model = Transformer(5, 5, d_model=4, num_heads=2, d_ff=8,
                            num_encoder_layers=1, num_decoder_layers=1, rng=rng)
        tokens = model.greedy_decode([1, 2], max_len=3, start_token=0, end_token=99)
        # Should stop after max_len iterations
        assert len(tokens) <= 4  # start + max_len

    def test_empty_padding_mask(self):
        m = padding_mask([3], 3)
        assert m.data == [[0.0, 0.0, 0.0]]

    def test_softmax_stability(self):
        """Softmax should handle large values without overflow."""
        vals = [1000.0, 999.0, 998.0]
        result = softmax(vals)
        assert abs(sum(result) - 1.0) < 1e-10
        assert result[0] > result[1] > result[2]


# ============================================================
# Gradient checking tests
# ============================================================

class TestGradientChecking:
    def test_layernorm_gradient_finite_diff(self):
        """Numerical gradient check for LayerNorm."""
        ln = LayerNorm(3)
        x = Tensor([1.0, 3.0, 5.0])
        out = ln.forward(x)
        # Upstream gradient
        grad_out = Tensor([1.0, 0.0, 0.0])
        grad_in = ln.backward(grad_out)

        # Numerical gradient
        eps = 1e-5
        for i in range(3):
            x_plus = Tensor(x.data[:])
            x_plus.data[i] += eps
            ln_p = LayerNorm(3)
            ln_p.gamma = Tensor(ln.gamma.data[:])
            ln_p.beta = Tensor(ln.beta.data[:])
            out_p = ln_p.forward(x_plus)

            x_minus = Tensor(x.data[:])
            x_minus.data[i] -= eps
            ln_m = LayerNorm(3)
            ln_m.gamma = Tensor(ln.gamma.data[:])
            ln_m.beta = Tensor(ln.beta.data[:])
            out_m = ln_m.forward(x_minus)

            numerical = (sum(out_p.data[j] * grad_out.data[j] for j in range(3)) -
                         sum(out_m.data[j] * grad_out.data[j] for j in range(3))) / (2 * eps)
            assert abs(grad_in.data[i] - numerical) < 1e-3, \
                f"Gradient mismatch at {i}: analytical={grad_in.data[i]}, numerical={numerical}"

    def test_attention_gradient_flow(self):
        """Verify gradients flow through attention mechanism."""
        rng = random.Random(42)
        mha = MultiHeadAttention(4, 2, rng=rng)
        x = Tensor.random_normal((2, 4), rng=rng)
        out = mha.forward(x, x, x)
        grad = Tensor.ones((2, 4))
        gQ, gK, gV = mha.backward(grad)
        # Gradients should be nonzero
        has_nonzero = False
        for i in range(2):
            for j in range(4):
                if abs(gQ.data[i][j]) > 1e-10:
                    has_nonzero = True
        assert has_nonzero

    def test_full_model_gradient_flow(self):
        """Verify gradients flow through entire transformer."""
        rng = random.Random(42)
        model = Transformer(5, 5, d_model=4, num_heads=2, d_ff=8,
                            num_encoder_layers=1, num_decoder_layers=1, rng=rng)
        logits = model.forward([1, 2], [0, 1])
        loss_fn = CrossEntropyLoss()
        grad = loss_fn.backward(logits, [2, 3])
        model.backward(grad)

        # Check embedding gradients exist
        assert model.src_embedding.grad_weight is not None
        assert model.tgt_embedding.grad_weight is not None
        # Check some gradient is nonzero
        has_nonzero = False
        for row in model.src_embedding.grad_weight.data:
            for v in row:
                if abs(v) > 1e-15:
                    has_nonzero = True
        assert has_nonzero


# ============================================================
# Numerical stability tests
# ============================================================

class TestNumericalStability:
    def test_layernorm_small_variance(self):
        """LayerNorm should handle near-constant inputs."""
        ln = LayerNorm(4, eps=1e-6)
        x = Tensor([1.0, 1.0 + 1e-8, 1.0 - 1e-8, 1.0])
        out = ln.forward(x)
        # Should not produce NaN or Inf
        for v in out.data:
            assert math.isfinite(v)

    def test_attention_large_values(self):
        """Attention should handle large key values via scaling."""
        Q = Tensor([[100.0, 0.0]])
        K = Tensor([[100.0, 0.0], [0.0, 100.0]])
        V = Tensor([[1.0, 0.0], [0.0, 1.0]])
        output, attn = scaled_dot_product_attention(Q, K, V)
        # Should still produce valid output
        assert all(math.isfinite(v) for v in output.data[0])
        assert abs(sum(attn.data[0]) - 1.0) < 1e-5

    def test_training_stability(self):
        """Training should not produce NaN or Inf."""
        rng = random.Random(42)
        model = Transformer(5, 5, d_model=8, num_heads=2, d_ff=16,
                            num_encoder_layers=1, num_decoder_layers=1, rng=rng)
        loss_fn = CrossEntropyLoss()
        opt = TransformerAdam(model.get_params, lr=0.001)

        data = [([1, 2], [0, 1], [1, 2])]
        history = train_transformer(model, data, loss_fn, opt, epochs=10)
        for loss in history:
            assert math.isfinite(loss)


# ============================================================
# Architecture tests
# ============================================================

class TestArchitecture:
    def test_d_model_divisible_by_heads(self):
        """Should raise if d_model not divisible by num_heads."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(7, 3)

    def test_encoder_decoder_composition(self):
        """Encoder output should feed into decoder."""
        rng = random.Random(42)
        enc = TransformerEncoder(1, 4, 2, 8, rng=rng)
        dec = TransformerDecoder(1, 4, 2, 8, rng=rng)

        src = Tensor.random_normal((3, 4), rng=rng)
        enc_out = enc.forward(src)

        tgt = Tensor.random_normal((2, 4), rng=rng)
        dec_out = dec.forward(tgt, enc_out)
        assert dec_out.shape == (2, 4)

    def test_parameter_count(self):
        """Verify parameter shapes are consistent."""
        model = Transformer(10, 10, d_model=8, num_heads=2, d_ff=16,
                            num_encoder_layers=1, num_decoder_layers=1)
        params = model.get_params()
        # Count total parameters
        total = 0
        for tensor, _, name in params:
            if len(tensor.shape) == 1:
                total += tensor.shape[0]
            else:
                total += tensor.shape[0] * tensor.shape[1]
        assert total > 0

    def test_embedding_scale(self):
        """Embeddings should be scaled by sqrt(d_model)."""
        rng = random.Random(42)
        model = Transformer(10, 10, d_model=16, num_heads=2, d_ff=32,
                            num_encoder_layers=1, num_decoder_layers=1, rng=rng)
        assert model.scale == math.sqrt(16)

    def test_positional_encoding_different_lengths(self):
        pe = PositionalEncoding(8)
        out3 = pe.forward(3)
        out5 = pe.forward(5)
        # First 3 rows should match
        for i in range(3):
            for j in range(8):
                assert abs(out3.data[i][j] - out5.data[i][j]) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
