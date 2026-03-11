"""
Tests for C200: Transformer (Milestone Challenge #200)
From-scratch Transformer architecture -- NumPy only.
"""

import numpy as np
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from transformer import (
    softmax, gelu, gelu_deriv, relu, layer_norm, rms_norm, dropout,
    TokenEmbedding, SinusoidalPositionalEncoding, LearnedPositionalEncoding,
    RotaryPositionalEncoding,
    ScaledDotProductAttention, MultiHeadAttention, GroupedQueryAttention,
    FeedForward, GatedFeedForward,
    TransformerEncoderBlock, TransformerDecoderBlock, PreNormEncoderBlock,
    KVCache,
    TransformerEncoder, TransformerDecoder, Transformer,
    GPTModel, BERTModel,
    BPETokenizer,
    AdamOptimizer, LRScheduler,
    cross_entropy_loss, compute_grad_cross_entropy, clip_grad_norm,
    greedy_decode, top_k_sample, top_p_sample, gpt_generate,
    BeamSearchDecoder,
    AttentionVisualizer,
    create_causal_mask, create_padding_mask, count_parameters, perplexity,
)


# ============================================================
# Core Operations
# ============================================================

class TestSoftmax:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        s = softmax(x)
        assert abs(s.sum() - 1.0) < 1e-6
        assert s[2] > s[1] > s[0]

    def test_numerical_stability(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        s = softmax(x)
        assert abs(s.sum() - 1.0) < 1e-6
        assert not np.any(np.isnan(s))

    def test_batch(self):
        x = np.random.randn(4, 10)
        s = softmax(x, axis=-1)
        assert s.shape == (4, 10)
        np.testing.assert_allclose(s.sum(axis=-1), np.ones(4), atol=1e-6)

    def test_uniform_input(self):
        x = np.ones(5)
        s = softmax(x)
        np.testing.assert_allclose(s, np.full(5, 0.2), atol=1e-6)


class TestGELU:
    def test_zero(self):
        assert abs(gelu(np.array(0.0))) < 1e-6

    def test_positive(self):
        assert gelu(np.array(2.0)) > 0

    def test_negative_small(self):
        # GELU is slightly negative for small negative inputs
        val = gelu(np.array(-0.5))
        assert val < 0

    def test_large_positive(self):
        val = gelu(np.array(5.0))
        assert abs(val - 5.0) < 0.01  # Approximately identity for large +

    def test_derivative(self):
        x = np.array(1.0)
        # Numerical derivative
        eps = 1e-5
        num_deriv = (gelu(np.array(x + eps)) - gelu(np.array(x - eps))) / (2 * eps)
        ana_deriv = gelu_deriv(np.array(x))
        assert abs(float(num_deriv) - float(ana_deriv)) < 1e-4


class TestLayerNorm:
    def test_basic(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        gamma = np.ones(4)
        beta = np.zeros(4)
        out = layer_norm(x, gamma, beta)
        assert abs(np.mean(out)) < 1e-5
        assert abs(np.var(out) - 1.0) < 0.1

    def test_with_scale_shift(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        gamma = np.array([2.0, 2.0, 2.0, 2.0])
        beta = np.array([1.0, 1.0, 1.0, 1.0])
        out = layer_norm(x, gamma, beta)
        assert abs(np.mean(out) - 1.0) < 0.1

    def test_batch(self):
        x = np.random.randn(3, 8)
        gamma = np.ones(8)
        beta = np.zeros(8)
        out = layer_norm(x, gamma, beta)
        means = np.mean(out, axis=-1)
        np.testing.assert_allclose(means, np.zeros(3), atol=1e-5)


class TestRMSNorm:
    def test_basic(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        gamma = np.ones(4)
        out = rms_norm(x, gamma)
        rms = np.sqrt(np.mean(out ** 2))
        assert abs(rms - 1.0) < 0.1

    def test_scale(self):
        x = np.array([[1.0, 1.0, 1.0, 1.0]])
        gamma = np.array([2.0, 2.0, 2.0, 2.0])
        out = rms_norm(x, gamma)
        assert abs(out[0, 0] - 2.0) < 0.1


class TestDropout:
    def test_no_dropout_eval(self):
        x = np.ones((3, 4))
        out = dropout(x, 0.5, training=False)
        np.testing.assert_array_equal(out, x)

    def test_zero_rate(self):
        x = np.ones((3, 4))
        out = dropout(x, 0.0, training=True)
        np.testing.assert_array_equal(out, x)

    def test_dropout_scales(self):
        rng = np.random.default_rng(42)
        x = np.ones((1000,))
        out = dropout(x, 0.5, training=True, rng=rng)
        # Mean should be approximately 1.0 due to scaling
        assert abs(np.mean(out) - 1.0) < 0.15


# ============================================================
# Embeddings
# ============================================================

class TestTokenEmbedding:
    def test_shape(self):
        emb = TokenEmbedding(100, 32)
        ids = np.array([[1, 2, 3], [4, 5, 6]])
        out = emb.forward(ids)
        assert out.shape == (2, 3, 32)

    def test_same_token_same_embedding(self):
        emb = TokenEmbedding(100, 16)
        ids = np.array([[5, 5, 5]])
        out = emb.forward(ids)
        np.testing.assert_array_equal(out[0, 0], out[0, 1])
        np.testing.assert_array_equal(out[0, 0], out[0, 2])

    def test_parameters(self):
        emb = TokenEmbedding(100, 32)
        params = emb.parameters()
        assert len(params) == 1
        assert params[0].shape == (100, 32)


class TestSinusoidalPE:
    def test_shape(self):
        pe = SinusoidalPositionalEncoding(100, 32)
        out = pe.forward(10)
        assert out.shape == (10, 32)

    def test_different_positions(self):
        pe = SinusoidalPositionalEncoding(100, 32)
        out = pe.forward(5)
        assert not np.allclose(out[0], out[1])

    def test_deterministic(self):
        pe = SinusoidalPositionalEncoding(100, 32)
        out1 = pe.forward(10)
        out2 = pe.forward(10)
        np.testing.assert_array_equal(out1, out2)

    def test_offset(self):
        pe = SinusoidalPositionalEncoding(100, 32)
        full = pe.forward(10)
        partial = pe.forward(5, offset=5)
        np.testing.assert_array_equal(full[5:10], partial)


class TestLearnedPE:
    def test_shape(self):
        pe = LearnedPositionalEncoding(100, 32)
        out = pe.forward(10)
        assert out.shape == (10, 32)

    def test_parameters(self):
        pe = LearnedPositionalEncoding(100, 32)
        params = pe.parameters()
        assert len(params) == 1
        assert params[0].shape == (100, 32)


class TestRoPE:
    def test_shape_preserved(self):
        rope = RotaryPositionalEncoding(32)
        x = np.random.randn(2, 10, 32)
        out = rope.apply(x)
        assert out.shape == x.shape

    def test_different_positions(self):
        rope = RotaryPositionalEncoding(16)
        x = np.ones((1, 5, 16))
        out = rope.apply(x)
        assert not np.allclose(out[0, 0], out[0, 1])

    def test_offset(self):
        rope = RotaryPositionalEncoding(16)
        x = np.ones((1, 1, 16))
        out0 = rope.apply(x, offset=0)
        out5 = rope.apply(x, offset=5)
        assert not np.allclose(out0, out5)


# ============================================================
# Attention
# ============================================================

class TestScaledDotProductAttention:
    def test_basic(self):
        attn = ScaledDotProductAttention()
        Q = np.random.randn(2, 4, 8)
        K = np.random.randn(2, 6, 8)
        V = np.random.randn(2, 6, 8)
        out, weights = attn.forward(Q, K, V)
        assert out.shape == (2, 4, 8)
        assert weights.shape == (2, 4, 6)

    def test_weights_sum_to_one(self):
        attn = ScaledDotProductAttention()
        Q = np.random.randn(1, 3, 4)
        K = np.random.randn(1, 5, 4)
        V = np.random.randn(1, 5, 4)
        _, weights = attn.forward(Q, K, V)
        np.testing.assert_allclose(weights.sum(axis=-1), np.ones((1, 3)), atol=1e-6)

    def test_masking(self):
        attn = ScaledDotProductAttention()
        Q = np.ones((1, 3, 4))
        K = np.ones((1, 3, 4))
        V = np.eye(3).reshape(1, 3, 3)
        mask = np.array([[[False, True, True],
                          [False, False, True],
                          [False, False, False]]])
        _, weights = attn.forward(Q, K, V, mask)
        # Position 0 should only attend to position 0
        assert float(weights[0, 0, 0]) > 0.99

    def test_identical_qk_uniform(self):
        attn = ScaledDotProductAttention()
        x = np.ones((1, 4, 8))
        _, weights = attn.forward(x, x, x)
        # All identical -> uniform attention
        np.testing.assert_allclose(weights[0, 0], np.full(4, 0.25), atol=1e-5)


class TestMultiHeadAttention:
    def test_shape(self):
        mha = MultiHeadAttention(d_model=32, n_heads=4)
        x = np.random.randn(2, 10, 32)
        out, weights = mha.forward(x, x, x)
        assert out.shape == (2, 10, 32)
        assert weights.shape == (2, 4, 10, 10)

    def test_cross_attention(self):
        mha = MultiHeadAttention(d_model=32, n_heads=4)
        q = np.random.randn(2, 5, 32)
        kv = np.random.randn(2, 8, 32)
        out, weights = mha.forward(q, kv, kv)
        assert out.shape == (2, 5, 32)
        assert weights.shape == (2, 4, 5, 8)

    def test_parameters_count(self):
        mha = MultiHeadAttention(d_model=64, n_heads=8)
        params = mha.parameters()
        assert len(params) == 8  # W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o

    def test_causal_masking(self):
        mha = MultiHeadAttention(d_model=16, n_heads=2)
        x = np.random.randn(1, 4, 16)
        mask = create_causal_mask(4)[np.newaxis, np.newaxis, :, :]
        out, weights = mha.forward(x, x, x, mask)
        # Upper triangle of attention should be ~0
        for h in range(2):
            for i in range(4):
                for j in range(i + 1, 4):
                    assert weights[0, h, i, j] < 1e-4


class TestGroupedQueryAttention:
    def test_shape(self):
        gqa = GroupedQueryAttention(d_model=32, n_heads=4, n_kv_heads=2)
        x = np.random.randn(2, 8, 32)
        out, weights = gqa.forward(x, x, x)
        assert out.shape == (2, 8, 32)
        assert weights.shape == (2, 4, 8, 8)

    def test_single_kv_head(self):
        """n_kv_heads=1 is multi-query attention."""
        gqa = GroupedQueryAttention(d_model=16, n_heads=4, n_kv_heads=1)
        x = np.random.randn(1, 5, 16)
        out, weights = gqa.forward(x, x, x)
        assert out.shape == (1, 5, 16)

    def test_parameters(self):
        gqa = GroupedQueryAttention(d_model=32, n_heads=4, n_kv_heads=2)
        params = gqa.parameters()
        assert len(params) == 8


# ============================================================
# Feed-Forward
# ============================================================

class TestFeedForward:
    def test_shape(self):
        ff = FeedForward(32, 64)
        x = np.random.randn(2, 10, 32)
        out = ff.forward(x)
        assert out.shape == (2, 10, 32)

    def test_relu_activation(self):
        ff = FeedForward(16, 32, activation='relu')
        x = np.random.randn(1, 5, 16)
        out = ff.forward(x)
        assert out.shape == (1, 5, 16)

    def test_parameters(self):
        ff = FeedForward(32, 64)
        params = ff.parameters()
        assert len(params) == 4


class TestGatedFeedForward:
    def test_shape(self):
        gff = GatedFeedForward(32, 64)
        x = np.random.randn(2, 10, 32)
        out = gff.forward(x)
        assert out.shape == (2, 10, 32)

    def test_parameters(self):
        gff = GatedFeedForward(32, 64)
        assert len(gff.parameters()) == 3


# ============================================================
# Transformer Blocks
# ============================================================

class TestEncoderBlock:
    def test_shape(self):
        block = TransformerEncoderBlock(d_model=32, n_heads=4, d_ff=64)
        x = np.random.randn(2, 10, 32)
        out, weights = block.forward(x, training=False)
        assert out.shape == (2, 10, 32)
        assert weights.shape == (2, 4, 10, 10)

    def test_with_mask(self):
        block = TransformerEncoderBlock(d_model=16, n_heads=2, d_ff=32)
        x = np.random.randn(1, 5, 16)
        mask = np.zeros((1, 1, 5, 5), dtype=bool)
        mask[0, 0, :, 4] = True  # Mask out position 4
        out, _ = block.forward(x, mask, training=False)
        assert out.shape == (1, 5, 16)


class TestDecoderBlock:
    def test_shape(self):
        block = TransformerDecoderBlock(d_model=32, n_heads=4, d_ff=64)
        x = np.random.randn(2, 8, 32)
        enc = np.random.randn(2, 12, 32)
        out, sw, cw = block.forward(x, enc, training=False)
        assert out.shape == (2, 8, 32)
        assert sw.shape == (2, 4, 8, 8)
        assert cw.shape == (2, 4, 8, 12)


class TestPreNormBlock:
    def test_shape(self):
        block = PreNormEncoderBlock(d_model=32, n_heads=4, d_ff=64)
        x = np.random.randn(2, 10, 32)
        out, weights = block.forward(x, training=False)
        assert out.shape == (2, 10, 32)


# ============================================================
# KV Cache
# ============================================================

class TestKVCache:
    def test_initial_empty(self):
        cache = KVCache(n_layers=2, n_heads=4, d_k=8)
        assert cache.seq_len(0) == 0

    def test_update_and_get(self):
        cache = KVCache(n_layers=2, n_heads=4, d_k=8)
        k = np.random.randn(1, 4, 1, 8)
        v = np.random.randn(1, 4, 1, 8)
        k_all, v_all = cache.update(0, k, v)
        assert k_all.shape == (1, 4, 1, 8)
        assert cache.seq_len(0) == 1

    def test_accumulation(self):
        cache = KVCache(n_layers=1, n_heads=2, d_k=4)
        for i in range(5):
            k = np.random.randn(1, 2, 1, 4)
            v = np.random.randn(1, 2, 1, 4)
            cache.update(0, k, v)
        assert cache.seq_len(0) == 5

    def test_clear(self):
        cache = KVCache(n_layers=2, n_heads=4, d_k=8)
        cache.update(0, np.zeros((1, 4, 1, 8)), np.zeros((1, 4, 1, 8)))
        cache.clear()
        assert cache.seq_len(0) == 0


# ============================================================
# Full Models
# ============================================================

class TestTransformerEncoder:
    def test_shape(self):
        enc = TransformerEncoder(n_layers=2, d_model=32, n_heads=4, d_ff=64, vocab_size=100)
        ids = np.array([[1, 2, 3, 4, 5]])
        out, weights = enc.forward(ids, training=False)
        assert out.shape == (1, 5, 32)
        assert len(weights) == 2

    def test_parameters(self):
        enc = TransformerEncoder(n_layers=2, d_model=32, n_heads=4, d_ff=64, vocab_size=50)
        params = enc.parameters()
        assert len(params) > 0
        total = sum(p.size for p in params)
        assert total > 0


class TestTransformerDecoder:
    def test_shape(self):
        dec = TransformerDecoder(n_layers=2, d_model=32, n_heads=4, d_ff=64, vocab_size=100)
        tgt = np.array([[1, 2, 3]])
        enc_out = np.random.randn(1, 5, 32)
        out, sw, cw = dec.forward(tgt, enc_out, training=False)
        assert out.shape == (1, 3, 32)
        assert len(sw) == 2
        assert len(cw) == 2


class TestTransformer:
    def test_forward(self):
        model = Transformer(n_layers=2, d_model=32, n_heads=4, d_ff=64,
                           src_vocab_size=50, tgt_vocab_size=50)
        src = np.array([[1, 2, 3, 4]])
        tgt = np.array([[1, 2, 3]])
        logits, _, _, _ = model.forward(src, tgt, training=False)
        assert logits.shape == (1, 3, 50)

    def test_different_vocab_sizes(self):
        model = Transformer(n_layers=1, d_model=16, n_heads=2, d_ff=32,
                           src_vocab_size=30, tgt_vocab_size=40)
        src = np.array([[1, 2]])
        tgt = np.array([[1, 2, 3]])
        logits, _, _, _ = model.forward(src, tgt, training=False)
        assert logits.shape == (1, 3, 40)

    def test_batch(self):
        model = Transformer(n_layers=1, d_model=16, n_heads=2, d_ff=32,
                           src_vocab_size=20, tgt_vocab_size=20)
        src = np.array([[1, 2, 3], [4, 5, 6]])
        tgt = np.array([[1, 2], [3, 4]])
        logits, _, _, _ = model.forward(src, tgt, training=False)
        assert logits.shape == (2, 2, 20)

    def test_parameter_count(self):
        model = Transformer(n_layers=2, d_model=32, n_heads=4, d_ff=64,
                           src_vocab_size=50, tgt_vocab_size=50)
        n = count_parameters(model)
        assert n > 10000  # Non-trivial model


class TestGPTModel:
    def test_forward(self):
        model = GPTModel(n_layers=2, d_model=32, n_heads=4, d_ff=64, vocab_size=50)
        ids = np.array([[1, 2, 3, 4, 5]])
        logits, weights = model.forward(ids, training=False)
        assert logits.shape == (1, 5, 50)
        assert len(weights) == 2

    def test_causal(self):
        """Output at position i should not depend on positions > i."""
        model = GPTModel(n_layers=1, d_model=16, n_heads=2, d_ff=32, vocab_size=20)
        ids = np.array([[1, 2, 3, 4, 5]])
        logits1, _ = model.forward(ids, training=False)
        # Change token at position 4
        ids2 = np.array([[1, 2, 3, 4, 10]])
        logits2, _ = model.forward(ids2, training=False)
        # Positions 0-3 should be identical
        np.testing.assert_allclose(logits1[0, :4], logits2[0, :4], atol=1e-6)

    def test_post_norm(self):
        model = GPTModel(n_layers=2, d_model=32, n_heads=4, d_ff=64,
                        vocab_size=50, pre_norm=False)
        ids = np.array([[1, 2, 3]])
        logits, _ = model.forward(ids, training=False)
        assert logits.shape == (1, 3, 50)


class TestBERTModel:
    def test_forward(self):
        model = BERTModel(n_layers=2, d_model=32, n_heads=4, d_ff=64, vocab_size=50)
        ids = np.array([[1, 2, 3, 4, 5]])
        mlm_logits, cls_repr, weights = model.forward(ids, training=False)
        assert mlm_logits.shape == (1, 5, 50)
        assert cls_repr.shape == (1, 32)
        assert len(weights) == 2

    def test_cls_representation(self):
        model = BERTModel(n_layers=1, d_model=16, n_heads=2, d_ff=32, vocab_size=20)
        ids = np.array([[1, 2, 3]])
        _, cls_repr, _ = model.forward(ids, training=False)
        # CLS repr should be bounded by tanh
        assert np.all(np.abs(cls_repr) <= 1.0 + 1e-6)

    def test_bidirectional(self):
        """BERT is bidirectional -- changing any token affects all outputs."""
        model = BERTModel(n_layers=1, d_model=16, n_heads=2, d_ff=32, vocab_size=20)
        ids1 = np.array([[1, 2, 3, 4, 5]])
        ids2 = np.array([[1, 2, 10, 4, 5]])  # Changed middle token
        mlm1, _, _ = model.forward(ids1, training=False)
        mlm2, _, _ = model.forward(ids2, training=False)
        # All positions should differ (bidirectional)
        assert not np.allclose(mlm1[0, 0], mlm2[0, 0], atol=1e-3)


# ============================================================
# Tokenizer
# ============================================================

class TestBPETokenizer:
    def test_init(self):
        tok = BPETokenizer()
        assert tok.get_vocab_size() > 0

    def test_encode_decode_roundtrip(self):
        tok = BPETokenizer()
        text = "hello world"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text

    def test_special_tokens(self):
        tok = BPETokenizer()
        assert tok.special_tokens['<pad>'] == 0
        assert tok.special_tokens['<bos>'] == 2
        assert tok.special_tokens['<eos>'] == 3

    def test_train(self):
        tok = BPETokenizer()
        texts = ["hello world", "hello there", "world hello", "hello hello"]
        initial_size = tok.get_vocab_size()
        tok.train(texts, target_vocab_size=initial_size + 5)
        assert tok.get_vocab_size() > initial_size

    def test_trained_encodes_shorter(self):
        tok = BPETokenizer()
        text = "hello hello hello"
        ids_before = tok.encode(text)
        tok.train(["hello"] * 100, target_vocab_size=tok.get_vocab_size() + 10)
        ids_after = tok.encode(text)
        assert len(ids_after) <= len(ids_before)

    def test_unknown_chars(self):
        tok = BPETokenizer()
        ids = tok.encode("abc")
        assert all(isinstance(i, int) for i in ids)


# ============================================================
# Training Components
# ============================================================

class TestAdamOptimizer:
    def test_step(self):
        params = [np.array([1.0, 2.0, 3.0])]
        opt = AdamOptimizer(params, lr=0.1)
        grads = [np.array([0.1, 0.2, 0.3])]
        opt.step(grads)
        # Parameters should have changed
        assert not np.allclose(params[0], [1.0, 2.0, 3.0])

    def test_convergence(self):
        """Minimize x^2."""
        params = [np.array([5.0])]
        opt = AdamOptimizer(params, lr=0.5)
        for _ in range(100):
            grad = 2 * params[0]
            opt.step([grad])
        assert abs(params[0][0]) < 0.1

    def test_weight_decay(self):
        params = [np.array([1.0, 2.0])]
        opt = AdamOptimizer(params, lr=0.01, weight_decay=0.1)
        grads = [np.zeros(2)]
        initial = params[0].copy()
        opt.step(grads)
        # Weight decay should push towards zero even with zero gradients
        assert np.all(np.abs(params[0]) < np.abs(initial))


class TestLRScheduler:
    def test_warmup(self):
        params = [np.array([1.0])]
        opt = AdamOptimizer(params, lr=1.0)
        sched = LRScheduler(opt, warmup_steps=10, total_steps=100)
        # During warmup, LR should increase
        lrs = []
        for _ in range(10):
            opt.step([np.array([0.01])])
            lr = sched.step()
            lrs.append(lr)
        assert lrs[-1] > lrs[0]

    def test_cosine_decay(self):
        params = [np.array([1.0])]
        opt = AdamOptimizer(params, lr=1.0)
        sched = LRScheduler(opt, warmup_steps=0, total_steps=100, schedule='cosine')
        lrs = []
        for _ in range(100):
            opt.step([np.array([0.01])])
            lr = sched.step()
            lrs.append(lr)
        assert lrs[-1] < lrs[0]

    def test_linear_decay(self):
        params = [np.array([1.0])]
        opt = AdamOptimizer(params, lr=1.0)
        sched = LRScheduler(opt, warmup_steps=0, total_steps=100, schedule='linear')
        opt.step([np.array([0.01])])
        lr = sched.step()
        assert lr < 1.0


class TestCrossEntropyLoss:
    def test_perfect_prediction(self):
        logits = np.zeros((1, 3, 10))
        logits[0, 0, 5] = 100  # Confident prediction
        logits[0, 1, 3] = 100
        logits[0, 2, 7] = 100
        targets = np.array([[5, 3, 7]])
        loss = cross_entropy_loss(logits, targets)
        assert loss < 0.01

    def test_random_prediction(self):
        logits = np.zeros((1, 3, 10))  # Uniform
        targets = np.array([[5, 3, 7]])
        loss = cross_entropy_loss(logits, targets)
        assert abs(loss - math.log(10)) < 0.1

    def test_gradient_shape(self):
        logits = np.random.randn(2, 5, 20)
        targets = np.random.randint(0, 20, (2, 5))
        grad = compute_grad_cross_entropy(logits, targets)
        assert grad.shape == logits.shape

    def test_gradient_numerical(self):
        """Numerical gradient check."""
        logits = np.random.randn(1, 2, 5)
        targets = np.array([[1, 3]])
        grad = compute_grad_cross_entropy(logits, targets)

        eps = 1e-5
        for i in range(5):
            logits_plus = logits.copy()
            logits_plus[0, 0, i] += eps
            logits_minus = logits.copy()
            logits_minus[0, 0, i] -= eps
            num_grad = (cross_entropy_loss(logits_plus, targets) -
                       cross_entropy_loss(logits_minus, targets)) / (2 * eps)
            assert abs(num_grad - grad[0, 0, i]) < 1e-4


class TestGradClip:
    def test_no_clip_needed(self):
        grads = [np.array([0.1, 0.2])]
        clipped, norm = clip_grad_norm(grads, max_norm=10.0)
        np.testing.assert_allclose(clipped[0], grads[0])

    def test_clip(self):
        grads = [np.array([3.0, 4.0])]  # norm = 5
        clipped, norm = clip_grad_norm(grads, max_norm=1.0)
        assert abs(norm - 5.0) < 1e-6
        clipped_norm = math.sqrt(sum(np.sum(g ** 2) for g in clipped))
        assert abs(clipped_norm - 1.0) < 1e-4


# ============================================================
# Inference / Decoding
# ============================================================

class TestGreedyDecode:
    def test_produces_output(self):
        model = Transformer(n_layers=1, d_model=16, n_heads=2, d_ff=32,
                           src_vocab_size=20, tgt_vocab_size=20)
        src = np.array([[1, 2, 3]])
        output = greedy_decode(model, src, max_len=5)
        assert output.shape[0] == 1
        assert output.shape[1] >= 2  # At least BOS + one token
        assert output[0, 0] == 2  # Starts with BOS


class TestTopKSample:
    def test_returns_valid_index(self):
        logits = np.random.randn(100)
        rng = np.random.default_rng(42)
        idx = top_k_sample(logits, k=10, rng=rng)
        assert 0 <= idx < 100

    def test_top_k_constraint(self):
        logits = np.zeros(100)
        logits[50] = 100.0  # Dominant
        rng = np.random.default_rng(42)
        # With temperature very low, should always pick 50
        idx = top_k_sample(logits, k=5, temperature=0.01, rng=rng)
        assert idx == 50

    def test_temperature(self):
        logits = np.array([0.0, 0.0, 10.0])
        rng = np.random.default_rng(42)
        # Very low temperature -> deterministic
        idx = top_k_sample(logits, k=3, temperature=0.001, rng=rng)
        assert idx == 2


class TestTopPSample:
    def test_returns_valid_index(self):
        logits = np.random.randn(50)
        rng = np.random.default_rng(42)
        idx = top_p_sample(logits, p=0.9, rng=rng)
        assert 0 <= idx < 50

    def test_dominant_token(self):
        logits = np.zeros(10)
        logits[7] = 100.0
        rng = np.random.default_rng(42)
        idx = top_p_sample(logits, p=0.5, temperature=0.01, rng=rng)
        assert idx == 7


class TestGPTGenerate:
    def test_length(self):
        model = GPTModel(n_layers=1, d_model=16, n_heads=2, d_ff=32, vocab_size=20, max_len=64)
        rng = np.random.default_rng(42)
        output = gpt_generate(model, [1, 2, 3], max_new_tokens=5, rng=rng)
        assert len(output) == 8  # 3 prompt + 5 generated

    def test_top_k(self):
        model = GPTModel(n_layers=1, d_model=16, n_heads=2, d_ff=32, vocab_size=20, max_len=64)
        rng = np.random.default_rng(42)
        output = gpt_generate(model, [1], max_new_tokens=3, top_k=5, rng=rng)
        assert len(output) == 4

    def test_top_p(self):
        model = GPTModel(n_layers=1, d_model=16, n_heads=2, d_ff=32, vocab_size=20, max_len=64)
        rng = np.random.default_rng(42)
        output = gpt_generate(model, [1], max_new_tokens=3, top_p=0.9, rng=rng)
        assert len(output) == 4

    def test_deterministic_temp_zero(self):
        model = GPTModel(n_layers=1, d_model=16, n_heads=2, d_ff=32, vocab_size=20, max_len=64)
        out1 = gpt_generate(model, [1, 2], max_new_tokens=5, temperature=0)
        out2 = gpt_generate(model, [1, 2], max_new_tokens=5, temperature=0)
        assert out1 == out2


class TestBeamSearch:
    def test_returns_results(self):
        model = Transformer(n_layers=1, d_model=16, n_heads=2, d_ff=32,
                           src_vocab_size=20, tgt_vocab_size=20)
        decoder = BeamSearchDecoder(model, beam_width=3, max_len=5)
        src = np.array([[1, 2, 3]])
        results = decoder.decode(src)
        assert len(results) > 0
        assert all(r[0][0] == 2 for r in results)  # Start with BOS

    def test_beam_width(self):
        model = Transformer(n_layers=1, d_model=16, n_heads=2, d_ff=32,
                           src_vocab_size=20, tgt_vocab_size=20)
        decoder = BeamSearchDecoder(model, beam_width=4, max_len=5)
        src = np.array([[1, 2]])
        results = decoder.decode(src)
        assert len(results) <= 4

    def test_scores_ordered(self):
        model = Transformer(n_layers=1, d_model=16, n_heads=2, d_ff=32,
                           src_vocab_size=20, tgt_vocab_size=20)
        decoder = BeamSearchDecoder(model, beam_width=3, max_len=8)
        src = np.array([[1, 2, 3]])
        results = decoder.decode(src)
        if len(results) >= 2:
            # First result should have highest score
            lp = lambda s: ((5 + len(s)) / 6) ** 0.6
            score0 = results[0][1] / lp(results[0][0])
            score1 = results[1][1] / lp(results[1][0])
            assert score0 >= score1 - 1e-6


# ============================================================
# Attention Visualizer
# ============================================================

class TestAttentionVisualizer:
    def test_entropy(self):
        # Uniform attention -> max entropy
        w = np.full((1, 4, 8, 8), 1.0 / 8)
        entropy = AttentionVisualizer.get_attention_entropy(w)
        expected = -8 * (1 / 8) * math.log(1 / 8)
        np.testing.assert_allclose(entropy, np.full((1, 4, 8), expected), atol=1e-5)

    def test_focused_low_entropy(self):
        w = np.zeros((1, 1, 3, 3))
        w[0, 0, 0, 0] = 1.0
        w[0, 0, 1, 1] = 1.0
        w[0, 0, 2, 2] = 1.0
        entropy = AttentionVisualizer.get_attention_entropy(w)
        assert np.all(entropy < 1e-6)

    def test_head_importance(self):
        weights = [np.random.randn(2, 4, 8, 8)]
        weights[0] = softmax(weights[0], axis=-1)
        importances = AttentionVisualizer.get_head_importance(weights)
        assert len(importances) == 4
        assert all(len(t) == 3 for t in importances)

    def test_attention_rollout(self):
        w1 = softmax(np.random.randn(1, 2, 5, 5), axis=-1)
        w2 = softmax(np.random.randn(1, 2, 5, 5), axis=-1)
        rollout = AttentionVisualizer.attention_rollout([w1, w2])
        assert rollout.shape == (1, 5, 5)
        # Rows should sum to ~1
        row_sums = rollout.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones((1, 5)), atol=0.1)


# ============================================================
# Utility Functions
# ============================================================

class TestCausalMask:
    def test_shape(self):
        mask = create_causal_mask(5)
        assert mask.shape == (5, 5)

    def test_lower_triangle_false(self):
        mask = create_causal_mask(4)
        for i in range(4):
            for j in range(i + 1):
                assert not mask[i, j]

    def test_upper_triangle_true(self):
        mask = create_causal_mask(4)
        for i in range(4):
            for j in range(i + 1, 4):
                assert mask[i, j]


class TestPaddingMask:
    def test_basic(self):
        ids = np.array([[1, 2, 3, 0, 0]])
        mask = create_padding_mask(ids, pad_id=0)
        assert mask.shape == (1, 1, 1, 5)
        assert not mask[0, 0, 0, 0]
        assert mask[0, 0, 0, 3]
        assert mask[0, 0, 0, 4]


class TestCountParameters:
    def test_gpt(self):
        model = GPTModel(n_layers=2, d_model=32, n_heads=4, d_ff=64, vocab_size=50)
        n = count_parameters(model)
        assert n > 5000

    def test_transformer(self):
        model = Transformer(n_layers=1, d_model=16, n_heads=2, d_ff=32,
                           src_vocab_size=20, tgt_vocab_size=20)
        n = count_parameters(model)
        assert n > 1000


class TestPerplexity:
    def test_zero_loss(self):
        assert abs(perplexity(0.0) - 1.0) < 1e-6

    def test_known_value(self):
        assert abs(perplexity(1.0) - math.e) < 0.01

    def test_high_loss_clipped(self):
        p = perplexity(200)  # Should not overflow
        assert p < float('inf')


# ============================================================
# Integration Tests
# ============================================================

class TestEndToEnd:
    def test_encoder_decoder_training_step(self):
        """Simulate one training step of encoder-decoder Transformer."""
        model = Transformer(n_layers=1, d_model=16, n_heads=2, d_ff=32,
                           src_vocab_size=20, tgt_vocab_size=20)
        src = np.array([[1, 2, 3, 4]])
        tgt = np.array([[2, 3, 4, 5]])
        logits, _, _, _ = model.forward(src, tgt, training=True)
        loss = cross_entropy_loss(logits, tgt)
        assert loss > 0
        grad = compute_grad_cross_entropy(logits, tgt)
        assert grad.shape == logits.shape

    def test_gpt_training_step(self):
        """Simulate one training step of GPT model."""
        model = GPTModel(n_layers=1, d_model=16, n_heads=2, d_ff=32, vocab_size=20)
        ids = np.array([[1, 2, 3, 4, 5]])
        logits, _ = model.forward(ids, training=True)
        targets = np.array([[2, 3, 4, 5, 6]])
        loss = cross_entropy_loss(logits, targets)
        assert loss > 0

    def test_bert_mlm_step(self):
        """Simulate one MLM training step."""
        model = BERTModel(n_layers=1, d_model=16, n_heads=2, d_ff=32, vocab_size=20)
        # Mask token at position 2
        ids = np.array([[1, 2, 4, 4, 5]])  # 4 = mask token
        mlm_logits, cls_repr, _ = model.forward(ids, training=True)
        # Loss only on masked position
        targets = np.array([[0, 0, 3, 0, 0]])  # True token at pos 2 was 3
        loss = cross_entropy_loss(mlm_logits, targets)
        assert loss > 0

    def test_tokenizer_model_pipeline(self):
        """Full pipeline: tokenize -> model -> decode."""
        tok = BPETokenizer()
        model = GPTModel(n_layers=1, d_model=16, n_heads=2, d_ff=32,
                        vocab_size=tok.get_vocab_size(), max_len=64)
        text = "hello"
        ids = tok.encode(text)
        generated = gpt_generate(model, ids, max_new_tokens=3, temperature=0)
        decoded = tok.decode(generated)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_model_determinism(self):
        """Same input -> same output in eval mode."""
        model = GPTModel(n_layers=2, d_model=32, n_heads=4, d_ff=64, vocab_size=50)
        ids = np.array([[1, 2, 3, 4]])
        logits1, _ = model.forward(ids, training=False)
        logits2, _ = model.forward(ids, training=False)
        np.testing.assert_array_equal(logits1, logits2)

    def test_multi_sequence_batch(self):
        """Batch processing multiple sequences."""
        model = Transformer(n_layers=1, d_model=16, n_heads=2, d_ff=32,
                           src_vocab_size=20, tgt_vocab_size=20)
        src = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        tgt = np.array([[2, 3], [5, 6], [8, 9]])
        logits, _, _, _ = model.forward(src, tgt, training=False)
        assert logits.shape == (3, 2, 20)


class TestScaling:
    def test_deeper_model(self):
        model = GPTModel(n_layers=4, d_model=64, n_heads=8, d_ff=128, vocab_size=100)
        ids = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        logits, _ = model.forward(ids, training=False)
        assert logits.shape == (1, 8, 100)
        assert not np.any(np.isnan(logits))

    def test_wider_model(self):
        model = GPTModel(n_layers=1, d_model=128, n_heads=8, d_ff=512, vocab_size=50)
        ids = np.array([[1, 2, 3]])
        logits, _ = model.forward(ids, training=False)
        assert logits.shape == (1, 3, 50)
        assert not np.any(np.isnan(logits))

    def test_longer_sequence(self):
        model = GPTModel(n_layers=1, d_model=32, n_heads=4, d_ff=64,
                        vocab_size=50, max_len=256)
        ids = np.random.randint(1, 50, (1, 100))
        logits, _ = model.forward(ids, training=False)
        assert logits.shape == (1, 100, 50)
        assert not np.any(np.isnan(logits))


class TestArchitecturalVariants:
    def test_gpt_vs_bert(self):
        """GPT is causal, BERT is bidirectional -- same size, different behavior."""
        gpt = GPTModel(n_layers=1, d_model=16, n_heads=2, d_ff=32, vocab_size=20)
        bert = BERTModel(n_layers=1, d_model=16, n_heads=2, d_ff=32, vocab_size=20)
        assert count_parameters(gpt) != count_parameters(bert)  # Different architectures

    def test_pre_norm_vs_post_norm(self):
        """Both should produce valid outputs."""
        pre = GPTModel(n_layers=2, d_model=16, n_heads=2, d_ff=32,
                      vocab_size=20, pre_norm=True)
        post = GPTModel(n_layers=2, d_model=16, n_heads=2, d_ff=32,
                       vocab_size=20, pre_norm=False)
        ids = np.array([[1, 2, 3]])
        logits_pre, _ = pre.forward(ids, training=False)
        logits_post, _ = post.forward(ids, training=False)
        assert logits_pre.shape == logits_post.shape
        # They should produce different outputs (different architectures)
        assert not np.allclose(logits_pre, logits_post)

    def test_gqa_vs_mha(self):
        """GQA should work with fewer KV parameters."""
        mha = MultiHeadAttention(d_model=32, n_heads=4)
        gqa = GroupedQueryAttention(d_model=32, n_heads=4, n_kv_heads=2)
        x = np.random.randn(1, 5, 32)
        out_mha, _ = mha.forward(x, x, x)
        out_gqa, _ = gqa.forward(x, x, x)
        assert out_mha.shape == out_gqa.shape
        # GQA has fewer KV params
        mha_kv_params = sum(p.size for p in [mha.W_k, mha.W_v])
        gqa_kv_params = sum(p.size for p in [gqa.W_k, gqa.W_v])
        assert gqa_kv_params < mha_kv_params


class TestNumericalStability:
    def test_large_logits(self):
        logits = np.random.randn(1, 5, 100) * 100
        targets = np.random.randint(0, 100, (1, 5))
        loss = cross_entropy_loss(logits, targets)
        assert not np.isnan(loss)
        assert not np.isinf(loss)

    def test_softmax_large_values(self):
        x = np.array([1000.0, 1000.1, 999.9])
        s = softmax(x)
        assert not np.any(np.isnan(s))
        assert abs(s.sum() - 1.0) < 1e-6

    def test_layer_norm_near_zero_var(self):
        x = np.ones((1, 4)) * 5.0
        gamma = np.ones(4)
        beta = np.zeros(4)
        out = layer_norm(x, gamma, beta)
        assert not np.any(np.isnan(out))

    def test_gelu_extreme_values(self):
        x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        out = gelu(x)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))
