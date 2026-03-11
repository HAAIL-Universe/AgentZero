"""
C200: Transformer (Milestone Challenge #200)
From-scratch implementation of the Transformer architecture.

Built from NumPy only -- no PyTorch, no TensorFlow, no frameworks.

Components:
- Tensor operations: matmul, softmax, layer norm, GELU
- Embeddings: token + positional (sinusoidal & learned)
- Multi-Head Attention: scaled dot-product, causal masking, KV cache
- Feed-Forward Network: 2-layer with GELU activation
- Transformer Block: attention + FFN + residual + layer norm
- Encoder: stack of transformer blocks
- Decoder: stack of transformer blocks with cross-attention
- Full Transformer: encoder-decoder architecture
- GPT-style Decoder-only model
- BERT-style Encoder-only model
- Tokenizer: byte-pair encoding (BPE)
- Training: Adam optimizer, learning rate warmup, gradient clipping
- Inference: greedy decoding, top-k sampling, temperature scaling
- Beam search decoding

This is the 200th challenge. The architecture that changed everything, built from first principles.
"""

import numpy as np
import math
from collections import Counter, defaultdict


# ============================================================
# Core Operations
# ============================================================

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def gelu(x):
    """Gaussian Error Linear Unit activation."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def gelu_deriv(x):
    """GELU derivative for backprop."""
    c = math.sqrt(2.0 / math.pi)
    inner = c * (x + 0.044715 * x ** 3)
    tanh_val = np.tanh(inner)
    sech2 = 1.0 - tanh_val ** 2
    inner_deriv = c * (1.0 + 3.0 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * inner_deriv


def relu(x):
    """ReLU activation."""
    return np.maximum(0, x)


def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


def rms_norm(x, gamma, eps=1e-6):
    """Root Mean Square Layer Normalization."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return gamma * (x / rms)


def dropout(x, rate, training=True, rng=None):
    """Dropout regularization."""
    if not training or rate == 0.0:
        return x
    if rng is None:
        rng = np.random.default_rng()
    mask = rng.random(x.shape) > rate
    return x * mask / (1.0 - rate)


# ============================================================
# Embeddings
# ============================================================

class TokenEmbedding:
    """Learnable token embedding lookup table."""

    def __init__(self, vocab_size, d_model, rng=None):
        self.vocab_size = vocab_size
        self.d_model = d_model
        rng = rng or np.random.default_rng(42)
        self.weight = rng.normal(0, 0.02, (vocab_size, d_model))

    def forward(self, token_ids):
        """token_ids: (batch, seq_len) -> (batch, seq_len, d_model)"""
        return self.weight[token_ids]

    def parameters(self):
        return [self.weight]


class SinusoidalPositionalEncoding:
    """Fixed sinusoidal positional encoding from 'Attention Is All You Need'."""

    def __init__(self, max_len, d_model):
        self.max_len = max_len
        self.d_model = d_model
        self.encoding = self._build_encoding(max_len, d_model)

    def _build_encoding(self, max_len, d_model):
        pe = np.zeros((max_len, d_model))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[:d_model // 2])
        return pe

    def forward(self, seq_len, offset=0):
        """Returns positional encoding for seq_len positions."""
        return self.encoding[offset:offset + seq_len]


class LearnedPositionalEncoding:
    """Learned positional embedding (GPT-style)."""

    def __init__(self, max_len, d_model, rng=None):
        self.max_len = max_len
        self.d_model = d_model
        rng = rng or np.random.default_rng(42)
        self.weight = rng.normal(0, 0.02, (max_len, d_model))

    def forward(self, seq_len, offset=0):
        return self.weight[offset:offset + seq_len]

    def parameters(self):
        return [self.weight]


class RotaryPositionalEncoding:
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, d_model, max_len=4096, base=10000.0):
        self.d_model = d_model
        self.max_len = max_len
        freqs = 1.0 / (base ** (np.arange(0, d_model, 2).astype(float) / d_model))
        t = np.arange(max_len)
        angles = np.outer(t, freqs)
        self.cos_cache = np.cos(angles)
        self.sin_cache = np.sin(angles)

    def apply(self, x, offset=0):
        """Apply rotary encoding to x: (..., seq_len, d_model)."""
        seq_len = x.shape[-2]
        half = x.shape[-1] // 2
        cos = self.cos_cache[offset:offset + seq_len, :half]
        sin = self.sin_cache[offset:offset + seq_len, :half]
        x1 = x[..., :half]
        x2 = x[..., half:]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return np.concatenate([out1, out2], axis=-1)


# ============================================================
# Attention Mechanisms
# ============================================================

class ScaledDotProductAttention:
    """Scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V"""

    def forward(self, Q, K, V, mask=None):
        """
        Q: (..., seq_q, d_k)
        K: (..., seq_k, d_k)
        V: (..., seq_k, d_v)
        mask: broadcastable to (..., seq_q, seq_k), True = mask out
        Returns: (..., seq_q, d_v), attention_weights
        """
        d_k = Q.shape[-1]
        scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = np.where(mask, -1e9, scores)
        weights = softmax(scores, axis=-1)
        output = np.matmul(weights, V)
        return output, weights


class MultiHeadAttention:
    """Multi-head attention with linear projections."""

    def __init__(self, d_model, n_heads, rng=None):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        rng = rng or np.random.default_rng(42)
        scale = math.sqrt(2.0 / (d_model + self.d_k))
        self.W_q = rng.normal(0, scale, (d_model, d_model))
        self.b_q = np.zeros(d_model)
        self.W_k = rng.normal(0, scale, (d_model, d_model))
        self.b_k = np.zeros(d_model)
        self.W_v = rng.normal(0, scale, (d_model, d_model))
        self.b_v = np.zeros(d_model)
        self.W_o = rng.normal(0, scale, (d_model, d_model))
        self.b_o = np.zeros(d_model)
        self.attention = ScaledDotProductAttention()

    def _split_heads(self, x):
        """(batch, seq, d_model) -> (batch, n_heads, seq, d_k)"""
        batch, seq, _ = x.shape
        x = x.reshape(batch, seq, self.n_heads, self.d_k)
        return np.transpose(x, (0, 2, 1, 3))

    def _merge_heads(self, x):
        """(batch, n_heads, seq, d_k) -> (batch, seq, d_model)"""
        batch, _, seq, _ = x.shape
        x = np.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch, seq, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        query, key, value: (batch, seq, d_model)
        mask: (batch, 1, seq_q, seq_k) or broadcastable
        Returns: (batch, seq_q, d_model), attention_weights
        """
        Q = query @ self.W_q + self.b_q
        K = key @ self.W_k + self.b_k
        V = value @ self.W_v + self.b_v

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        attn_out, attn_weights = self.attention.forward(Q, K, V, mask)
        output = self._merge_heads(attn_out)
        output = output @ self.W_o + self.b_o
        return output, attn_weights

    def parameters(self):
        return [self.W_q, self.b_q, self.W_k, self.b_k,
                self.W_v, self.b_v, self.W_o, self.b_o]


class GroupedQueryAttention:
    """Grouped-Query Attention (GQA) -- fewer KV heads than Q heads."""

    def __init__(self, d_model, n_heads, n_kv_heads, rng=None):
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.group_size = n_heads // n_kv_heads
        rng = rng or np.random.default_rng(42)
        scale = math.sqrt(2.0 / (d_model + self.d_k))
        self.W_q = rng.normal(0, scale, (d_model, d_model))
        self.b_q = np.zeros(d_model)
        kv_dim = n_kv_heads * self.d_k
        self.W_k = rng.normal(0, scale, (d_model, kv_dim))
        self.b_k = np.zeros(kv_dim)
        self.W_v = rng.normal(0, scale, (d_model, kv_dim))
        self.b_v = np.zeros(kv_dim)
        self.W_o = rng.normal(0, scale, (d_model, d_model))
        self.b_o = np.zeros(d_model)
        self.attention = ScaledDotProductAttention()

    def forward(self, query, key, value, mask=None):
        batch, seq_q, _ = query.shape
        seq_k = key.shape[1]

        Q = (query @ self.W_q + self.b_q).reshape(batch, seq_q, self.n_heads, self.d_k)
        Q = np.transpose(Q, (0, 2, 1, 3))

        K = (key @ self.W_k + self.b_k).reshape(batch, seq_k, self.n_kv_heads, self.d_k)
        K = np.transpose(K, (0, 2, 1, 3))
        V = (value @ self.W_v + self.b_v).reshape(batch, seq_k, self.n_kv_heads, self.d_k)
        V = np.transpose(V, (0, 2, 1, 3))

        # Repeat KV heads for each group
        K = np.repeat(K, self.group_size, axis=1)
        V = np.repeat(V, self.group_size, axis=1)

        attn_out, attn_weights = self.attention.forward(Q, K, V, mask)
        output = np.transpose(attn_out, (0, 2, 1, 3)).reshape(batch, seq_q, self.d_model)
        output = output @ self.W_o + self.b_o
        return output, attn_weights

    def parameters(self):
        return [self.W_q, self.b_q, self.W_k, self.b_k,
                self.W_v, self.b_v, self.W_o, self.b_o]


# ============================================================
# Feed-Forward Network
# ============================================================

class FeedForward:
    """Position-wise feed-forward network."""

    def __init__(self, d_model, d_ff, activation='gelu', rng=None):
        rng = rng or np.random.default_rng(42)
        scale1 = math.sqrt(2.0 / (d_model + d_ff))
        scale2 = math.sqrt(2.0 / (d_ff + d_model))
        self.W1 = rng.normal(0, scale1, (d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.normal(0, scale2, (d_ff, d_model))
        self.b2 = np.zeros(d_model)
        self.activation = activation

    def forward(self, x):
        h = x @ self.W1 + self.b1
        if self.activation == 'gelu':
            h = gelu(h)
        elif self.activation == 'relu':
            h = relu(h)
        return h @ self.W2 + self.b2

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]


class GatedFeedForward:
    """SwiGLU-style gated feed-forward (LLaMA-style)."""

    def __init__(self, d_model, d_ff, rng=None):
        rng = rng or np.random.default_rng(42)
        scale = math.sqrt(2.0 / (d_model + d_ff))
        self.W_gate = rng.normal(0, scale, (d_model, d_ff))
        self.W_up = rng.normal(0, scale, (d_model, d_ff))
        self.W_down = rng.normal(0, scale, (d_ff, d_model))

    def forward(self, x):
        gate = x @ self.W_gate
        gate = gate * (1.0 / (1.0 + np.exp(-gate)))  # SiLU/Swish
        up = x @ self.W_up
        return (gate * up) @ self.W_down

    def parameters(self):
        return [self.W_gate, self.W_up, self.W_down]


# ============================================================
# Transformer Blocks
# ============================================================

class TransformerEncoderBlock:
    """Single encoder block: self-attention + FFN with residual + layer norm."""

    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1, activation='gelu', rng=None):
        rng = rng or np.random.default_rng(42)
        self.attention = MultiHeadAttention(d_model, n_heads, rng=rng)
        self.ffn = FeedForward(d_model, d_ff, activation=activation, rng=rng)
        self.norm1_gamma = np.ones(d_model)
        self.norm1_beta = np.zeros(d_model)
        self.norm2_gamma = np.ones(d_model)
        self.norm2_beta = np.zeros(d_model)
        self.dropout_rate = dropout_rate

    def forward(self, x, mask=None, training=True):
        # Self-attention with residual + layer norm
        attn_out, attn_weights = self.attention.forward(x, x, x, mask)
        attn_out = dropout(attn_out, self.dropout_rate, training)
        x = layer_norm(x + attn_out, self.norm1_gamma, self.norm1_beta)

        # FFN with residual + layer norm
        ffn_out = self.ffn.forward(x)
        ffn_out = dropout(ffn_out, self.dropout_rate, training)
        x = layer_norm(x + ffn_out, self.norm2_gamma, self.norm2_beta)

        return x, attn_weights

    def parameters(self):
        return (self.attention.parameters() + self.ffn.parameters() +
                [self.norm1_gamma, self.norm1_beta, self.norm2_gamma, self.norm2_beta])


class TransformerDecoderBlock:
    """Single decoder block: masked self-attention + cross-attention + FFN."""

    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1, activation='gelu', rng=None):
        rng = rng or np.random.default_rng(42)
        self.self_attention = MultiHeadAttention(d_model, n_heads, rng=rng)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, rng=rng)
        self.ffn = FeedForward(d_model, d_ff, activation=activation, rng=rng)
        self.norm1_gamma = np.ones(d_model)
        self.norm1_beta = np.zeros(d_model)
        self.norm2_gamma = np.ones(d_model)
        self.norm2_beta = np.zeros(d_model)
        self.norm3_gamma = np.ones(d_model)
        self.norm3_beta = np.zeros(d_model)
        self.dropout_rate = dropout_rate

    def forward(self, x, encoder_output, self_mask=None, cross_mask=None, training=True):
        # Masked self-attention
        attn_out, self_attn_w = self.self_attention.forward(x, x, x, self_mask)
        attn_out = dropout(attn_out, self.dropout_rate, training)
        x = layer_norm(x + attn_out, self.norm1_gamma, self.norm1_beta)

        # Cross-attention
        cross_out, cross_attn_w = self.cross_attention.forward(x, encoder_output, encoder_output, cross_mask)
        cross_out = dropout(cross_out, self.dropout_rate, training)
        x = layer_norm(x + cross_out, self.norm2_gamma, self.norm2_beta)

        # FFN
        ffn_out = self.ffn.forward(x)
        ffn_out = dropout(ffn_out, self.dropout_rate, training)
        x = layer_norm(x + ffn_out, self.norm3_gamma, self.norm3_beta)

        return x, self_attn_w, cross_attn_w

    def parameters(self):
        return (self.self_attention.parameters() + self.cross_attention.parameters() +
                self.ffn.parameters() +
                [self.norm1_gamma, self.norm1_beta, self.norm2_gamma, self.norm2_beta,
                 self.norm3_gamma, self.norm3_beta])


class PreNormEncoderBlock:
    """Pre-norm variant (GPT-2 style): norm before attention/FFN."""

    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1, activation='gelu', rng=None):
        rng = rng or np.random.default_rng(42)
        self.attention = MultiHeadAttention(d_model, n_heads, rng=rng)
        self.ffn = FeedForward(d_model, d_ff, activation=activation, rng=rng)
        self.norm1_gamma = np.ones(d_model)
        self.norm1_beta = np.zeros(d_model)
        self.norm2_gamma = np.ones(d_model)
        self.norm2_beta = np.zeros(d_model)
        self.dropout_rate = dropout_rate

    def forward(self, x, mask=None, training=True):
        normed = layer_norm(x, self.norm1_gamma, self.norm1_beta)
        attn_out, attn_weights = self.attention.forward(normed, normed, normed, mask)
        attn_out = dropout(attn_out, self.dropout_rate, training)
        x = x + attn_out

        normed = layer_norm(x, self.norm2_gamma, self.norm2_beta)
        ffn_out = self.ffn.forward(normed)
        ffn_out = dropout(ffn_out, self.dropout_rate, training)
        x = x + ffn_out

        return x, attn_weights

    def parameters(self):
        return (self.attention.parameters() + self.ffn.parameters() +
                [self.norm1_gamma, self.norm1_beta, self.norm2_gamma, self.norm2_beta])


# ============================================================
# KV Cache for Efficient Inference
# ============================================================

class KVCache:
    """Key-Value cache for autoregressive generation."""

    def __init__(self, n_layers, n_heads, d_k):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_k = d_k
        self.keys = [None] * n_layers
        self.values = [None] * n_layers

    def update(self, layer_idx, new_k, new_v):
        """Append new K, V to cache for given layer."""
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = new_k
            self.values[layer_idx] = new_v
        else:
            self.keys[layer_idx] = np.concatenate([self.keys[layer_idx], new_k], axis=2)
            self.values[layer_idx] = np.concatenate([self.values[layer_idx], new_v], axis=2)
        return self.keys[layer_idx], self.values[layer_idx]

    def get(self, layer_idx):
        return self.keys[layer_idx], self.values[layer_idx]

    def seq_len(self, layer_idx=0):
        if self.keys[layer_idx] is None:
            return 0
        return self.keys[layer_idx].shape[2]

    def clear(self):
        self.keys = [None] * self.n_layers
        self.values = [None] * self.n_layers


# ============================================================
# Full Transformer Models
# ============================================================

class TransformerEncoder:
    """Stack of encoder blocks."""

    def __init__(self, n_layers, d_model, n_heads, d_ff, vocab_size,
                 max_len=512, dropout_rate=0.1, rng=None):
        rng = rng or np.random.default_rng(42)
        self.token_emb = TokenEmbedding(vocab_size, d_model, rng=rng)
        self.pos_enc = SinusoidalPositionalEncoding(max_len, d_model)
        self.layers = [
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout_rate, rng=rng)
            for _ in range(n_layers)
        ]
        self.d_model = d_model
        self.dropout_rate = dropout_rate

    def forward(self, token_ids, mask=None, training=True):
        """token_ids: (batch, seq_len) -> (batch, seq_len, d_model)"""
        seq_len = token_ids.shape[1]
        x = self.token_emb.forward(token_ids) * math.sqrt(self.d_model)
        x = x + self.pos_enc.forward(seq_len)
        x = dropout(x, self.dropout_rate, training)

        all_weights = []
        for layer in self.layers:
            x, w = layer.forward(x, mask, training)
            all_weights.append(w)

        return x, all_weights

    def parameters(self):
        params = self.token_emb.parameters()
        for layer in self.layers:
            params += layer.parameters()
        return params


class TransformerDecoder:
    """Stack of decoder blocks."""

    def __init__(self, n_layers, d_model, n_heads, d_ff, vocab_size,
                 max_len=512, dropout_rate=0.1, rng=None):
        rng = rng or np.random.default_rng(42)
        self.token_emb = TokenEmbedding(vocab_size, d_model, rng=rng)
        self.pos_enc = SinusoidalPositionalEncoding(max_len, d_model)
        self.layers = [
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout_rate, rng=rng)
            for _ in range(n_layers)
        ]
        self.d_model = d_model
        self.dropout_rate = dropout_rate

    def forward(self, token_ids, encoder_output, self_mask=None, cross_mask=None, training=True):
        seq_len = token_ids.shape[1]
        x = self.token_emb.forward(token_ids) * math.sqrt(self.d_model)
        x = x + self.pos_enc.forward(seq_len)
        x = dropout(x, self.dropout_rate, training)

        all_self_w = []
        all_cross_w = []
        for layer in self.layers:
            x, sw, cw = layer.forward(x, encoder_output, self_mask, cross_mask, training)
            all_self_w.append(sw)
            all_cross_w.append(cw)

        return x, all_self_w, all_cross_w

    def parameters(self):
        params = self.token_emb.parameters()
        for layer in self.layers:
            params += layer.parameters()
        return params


class Transformer:
    """Full encoder-decoder Transformer (original 'Attention Is All You Need')."""

    def __init__(self, n_layers=2, d_model=64, n_heads=4, d_ff=128,
                 src_vocab_size=100, tgt_vocab_size=100, max_len=512,
                 dropout_rate=0.1, rng=None):
        rng = rng or np.random.default_rng(42)
        self.encoder = TransformerEncoder(
            n_layers, d_model, n_heads, d_ff, src_vocab_size, max_len, dropout_rate, rng=rng)
        self.decoder = TransformerDecoder(
            n_layers, d_model, n_heads, d_ff, tgt_vocab_size, max_len, dropout_rate, rng=rng)
        self.output_proj = rng.normal(0, 0.02, (d_model, tgt_vocab_size))
        self.output_bias = np.zeros(tgt_vocab_size)
        self.d_model = d_model

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None, cross_mask=None, training=True):
        enc_out, enc_weights = self.encoder.forward(src_ids, src_mask, training)
        dec_out, dec_self_w, dec_cross_w = self.decoder.forward(
            tgt_ids, enc_out, tgt_mask, cross_mask, training)
        logits = dec_out @ self.output_proj + self.output_bias
        return logits, enc_weights, dec_self_w, dec_cross_w

    def parameters(self):
        return (self.encoder.parameters() + self.decoder.parameters() +
                [self.output_proj, self.output_bias])


class GPTModel:
    """Decoder-only Transformer (GPT-style)."""

    def __init__(self, n_layers=2, d_model=64, n_heads=4, d_ff=128,
                 vocab_size=100, max_len=512, dropout_rate=0.1,
                 pre_norm=True, rng=None):
        rng = rng or np.random.default_rng(42)
        self.token_emb = TokenEmbedding(vocab_size, d_model, rng=rng)
        self.pos_enc = LearnedPositionalEncoding(max_len, d_model, rng=rng)
        BlockClass = PreNormEncoderBlock if pre_norm else TransformerEncoderBlock
        self.layers = [
            BlockClass(d_model, n_heads, d_ff, dropout_rate, rng=rng)
            for _ in range(n_layers)
        ]
        self.final_norm_gamma = np.ones(d_model)
        self.final_norm_beta = np.zeros(d_model)
        self.output_proj = rng.normal(0, 0.02, (d_model, vocab_size))
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.dropout_rate = dropout_rate
        self.pre_norm = pre_norm

    def forward(self, token_ids, training=True):
        batch, seq_len = token_ids.shape
        # Causal mask
        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        causal_mask = causal_mask[np.newaxis, np.newaxis, :, :]

        x = self.token_emb.forward(token_ids) + self.pos_enc.forward(seq_len)
        x = dropout(x, self.dropout_rate, training)

        all_weights = []
        for layer in self.layers:
            x, w = layer.forward(x, causal_mask, training)
            all_weights.append(w)

        if self.pre_norm:
            x = layer_norm(x, self.final_norm_gamma, self.final_norm_beta)

        logits = x @ self.output_proj
        return logits, all_weights

    def parameters(self):
        params = self.token_emb.parameters() + self.pos_enc.parameters()
        for layer in self.layers:
            params += layer.parameters()
        params += [self.final_norm_gamma, self.final_norm_beta, self.output_proj]
        return params


class BERTModel:
    """Encoder-only Transformer (BERT-style) with MLM head."""

    def __init__(self, n_layers=2, d_model=64, n_heads=4, d_ff=128,
                 vocab_size=100, max_len=512, dropout_rate=0.1, rng=None):
        rng = rng or np.random.default_rng(42)
        self.encoder = TransformerEncoder(
            n_layers, d_model, n_heads, d_ff, vocab_size, max_len, dropout_rate, rng=rng)
        self.mlm_proj = rng.normal(0, 0.02, (d_model, vocab_size))
        self.mlm_bias = np.zeros(vocab_size)
        self.cls_proj = rng.normal(0, 0.02, (d_model, d_model))
        self.cls_bias = np.zeros(d_model)
        self.d_model = d_model
        self.vocab_size = vocab_size

    def forward(self, token_ids, mask=None, training=True):
        enc_out, weights = self.encoder.forward(token_ids, mask, training)
        # MLM logits
        mlm_logits = enc_out @ self.mlm_proj + self.mlm_bias
        # CLS token representation
        cls_repr = np.tanh(enc_out[:, 0, :] @ self.cls_proj + self.cls_bias)
        return mlm_logits, cls_repr, weights

    def parameters(self):
        return (self.encoder.parameters() +
                [self.mlm_proj, self.mlm_bias, self.cls_proj, self.cls_bias])


# ============================================================
# Tokenizer (Byte-Pair Encoding)
# ============================================================

class BPETokenizer:
    """Byte-Pair Encoding tokenizer."""

    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = {}
        self.inverse_vocab = {}
        self.special_tokens = {
            '<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3, '<mask>': 4
        }
        self._init_vocab()

    def _init_vocab(self):
        self.vocab = dict(self.special_tokens)
        idx = len(self.special_tokens)
        for i in range(256):
            self.vocab[chr(i) if 32 <= i < 127 else f'<byte_{i}>'] = idx
            idx += 1
        self.vocab['</w>'] = idx
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def train(self, texts, target_vocab_size=None):
        """Learn BPE merges from training texts."""
        target = target_vocab_size or self.vocab_size
        # Tokenize to characters
        word_freqs = Counter()
        for text in texts:
            words = text.split()
            for word in words:
                tokens = tuple(word) + ('</w>',)
                word_freqs[tokens] += 1

        if '</w>' not in self.vocab:
            self.vocab['</w>'] = len(self.vocab)
            self.inverse_vocab[self.vocab['</w>']] = '</w>'

        while len(self.vocab) < target:
            # Count all adjacent pairs
            pairs = Counter()
            for word, freq in word_freqs.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += freq

            if not pairs:
                break

            best_pair = pairs.most_common(1)[0][0]
            self.merges.append(best_pair)

            merged = best_pair[0] + best_pair[1]
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)
                self.inverse_vocab[self.vocab[merged]] = merged

            # Apply merge
            new_word_freqs = Counter()
            for word, freq in word_freqs.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_freqs[tuple(new_word)] = freq
            word_freqs = new_word_freqs

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        """Encode text to token IDs."""
        tokens = []
        words = text.split()
        for wi, word in enumerate(words):
            chars = list(word) + ['</w>']
            # Apply merges
            for left, right in self.merges:
                merged = left + right
                new_chars = []
                i = 0
                while i < len(chars):
                    if i < len(chars) - 1 and chars[i] == left and chars[i + 1] == right:
                        new_chars.append(merged)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                chars = new_chars
            for ch in chars:
                tokens.append(self.vocab.get(ch, self.special_tokens['<unk>']))
        return tokens

    def decode(self, token_ids):
        """Decode token IDs back to text."""
        tokens = [self.inverse_vocab.get(tid, '<unk>') for tid in token_ids
                  if tid not in self.special_tokens.values()]
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()

    def get_vocab_size(self):
        return len(self.vocab)


# ============================================================
# Training Components
# ============================================================

class AdamOptimizer:
    """Adam optimizer with optional weight decay and warmup."""

    def __init__(self, parameters, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.m = [np.zeros_like(p) for p in parameters]
        self.v = [np.zeros_like(p) for p in parameters]

    def step(self, gradients):
        """Update parameters with gradients."""
        self.step_count += 1
        for i, (param, grad) in enumerate(zip(self.parameters, gradients)):
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

            m_hat = self.m[i] / (1 - self.beta1 ** self.step_count)
            v_hat = self.v[i] / (1 - self.beta2 ** self.step_count)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        pass  # Gradients are passed explicitly


class LRScheduler:
    """Learning rate scheduler with warmup and cosine decay."""

    def __init__(self, optimizer, warmup_steps=100, total_steps=1000,
                 min_lr=1e-6, schedule='cosine'):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.lr
        self.schedule = schedule

    def step(self):
        step = self.optimizer.step_count
        if step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (step / max(1, self.warmup_steps))
        elif self.schedule == 'cosine':
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        elif self.schedule == 'linear':
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            lr = self.base_lr * (1 - progress) + self.min_lr * progress
        else:
            lr = self.base_lr

        self.optimizer.lr = lr
        return lr


def cross_entropy_loss(logits, targets):
    """Cross-entropy loss for classification.
    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len) integer labels
    """
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    # Stable log-softmax
    log_probs = logits_flat - np.log(np.sum(np.exp(logits_flat - np.max(logits_flat, axis=-1, keepdims=True)), axis=-1, keepdims=True)) - np.max(logits_flat, axis=-1, keepdims=True)
    loss = -log_probs[np.arange(len(targets_flat)), targets_flat]
    return np.mean(loss)


def compute_grad_cross_entropy(logits, targets):
    """Gradient of cross-entropy loss w.r.t. logits."""
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    probs = softmax(logits_flat, axis=-1)
    probs[np.arange(len(targets_flat)), targets_flat] -= 1
    probs /= len(targets_flat)
    return probs.reshape(batch, seq_len, vocab_size)


def clip_grad_norm(gradients, max_norm):
    """Clip gradient norm."""
    total_norm = 0.0
    for g in gradients:
        total_norm += np.sum(g ** 2)
    total_norm = math.sqrt(total_norm)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        gradients = [g * scale for g in gradients]
    return gradients, total_norm


# ============================================================
# Inference / Decoding
# ============================================================

def greedy_decode(model, src_ids, max_len, bos_id=2, eos_id=3):
    """Greedy decoding for encoder-decoder Transformer."""
    enc_out, _ = model.encoder.forward(src_ids, training=False)
    batch = src_ids.shape[0]
    tgt_ids = np.full((batch, 1), bos_id, dtype=int)

    for _ in range(max_len):
        seq_len = tgt_ids.shape[1]
        causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        causal_mask = causal_mask[np.newaxis, np.newaxis, :, :]

        dec_out, _, _ = model.decoder.forward(tgt_ids, enc_out, causal_mask, training=False)
        logits = dec_out[:, -1:, :] @ model.output_proj + model.output_bias
        next_token = np.argmax(logits[:, 0, :], axis=-1, keepdims=True)
        tgt_ids = np.concatenate([tgt_ids, next_token], axis=1)

        if np.all(next_token == eos_id):
            break

    return tgt_ids


def top_k_sample(logits, k=10, temperature=1.0, rng=None):
    """Top-k sampling from logits."""
    rng = rng or np.random.default_rng()
    logits = logits / max(temperature, 1e-8)
    # Get top-k
    top_indices = np.argpartition(logits, -k)[-k:]
    top_logits = logits[top_indices]
    probs = softmax(top_logits)
    chosen_idx = rng.choice(len(top_indices), p=probs)
    return top_indices[chosen_idx]


def top_p_sample(logits, p=0.9, temperature=1.0, rng=None):
    """Nucleus (top-p) sampling."""
    rng = rng or np.random.default_rng()
    logits = logits / max(temperature, 1e-8)
    probs = softmax(logits)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumsum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumsum, p) + 1
    top_indices = sorted_indices[:cutoff]
    top_probs = probs[top_indices]
    top_probs = top_probs / top_probs.sum()
    chosen_idx = rng.choice(len(top_indices), p=top_probs)
    return top_indices[chosen_idx]


def gpt_generate(model, prompt_ids, max_new_tokens, temperature=1.0,
                 top_k=None, top_p=None, rng=None):
    """Generate tokens from a GPT-style model."""
    rng = rng or np.random.default_rng()
    token_ids = np.array(prompt_ids).reshape(1, -1)

    for _ in range(max_new_tokens):
        # Truncate to max_len if needed
        if token_ids.shape[1] > model.max_len:
            token_ids = token_ids[:, -model.max_len:]

        logits, _ = model.forward(token_ids, training=False)
        next_logits = logits[0, -1, :]

        if top_k is not None:
            next_token = top_k_sample(next_logits, k=top_k, temperature=temperature, rng=rng)
        elif top_p is not None:
            next_token = top_p_sample(next_logits, p=top_p, temperature=temperature, rng=rng)
        else:
            if temperature > 0:
                probs = softmax(next_logits / temperature)
                next_token = rng.choice(len(probs), p=probs)
            else:
                next_token = np.argmax(next_logits)

        token_ids = np.concatenate([token_ids, [[next_token]]], axis=1)

    return token_ids[0].tolist()


class BeamSearchDecoder:
    """Beam search decoding for encoder-decoder Transformer."""

    def __init__(self, model, beam_width=4, max_len=50, length_penalty=0.6):
        self.model = model
        self.beam_width = beam_width
        self.max_len = max_len
        self.length_penalty = length_penalty

    def decode(self, src_ids, bos_id=2, eos_id=3):
        """Run beam search. Returns list of (sequence, score) tuples."""
        enc_out, _ = self.model.encoder.forward(src_ids, training=False)

        # Each beam: (token_ids_list, cumulative_log_prob)
        beams = [([bos_id], 0.0)]
        completed = []

        for step in range(self.max_len):
            candidates = []
            for seq, score in beams:
                if seq[-1] == eos_id:
                    completed.append((seq, score))
                    continue

                tgt = np.array([seq]).astype(int)
                seq_len = tgt.shape[1]
                causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
                causal_mask = causal_mask[np.newaxis, np.newaxis, :, :]

                dec_out, _, _ = self.model.decoder.forward(tgt, enc_out, causal_mask, training=False)
                logits = dec_out[0, -1, :] @ self.model.output_proj + self.model.output_bias
                log_probs = logits - np.log(np.sum(np.exp(logits - np.max(logits)))) - np.max(logits)

                top_k = min(self.beam_width * 2, len(log_probs))
                top_indices = np.argpartition(log_probs, -top_k)[-top_k:]

                for idx in top_indices:
                    new_seq = seq + [int(idx)]
                    new_score = score + log_probs[idx]
                    candidates.append((new_seq, new_score))

            if not candidates:
                break

            # Length-normalized scoring
            def beam_score(item):
                seq, s = item
                lp = ((5 + len(seq)) / 6) ** self.length_penalty
                return s / lp

            candidates.sort(key=beam_score, reverse=True)
            beams = candidates[:self.beam_width]

        completed.extend(beams)

        def final_score(item):
            seq, s = item
            lp = ((5 + len(seq)) / 6) ** self.length_penalty
            return s / lp

        completed.sort(key=final_score, reverse=True)
        return completed[:self.beam_width]


# ============================================================
# Attention Visualization
# ============================================================

class AttentionVisualizer:
    """Extract and analyze attention patterns."""

    @staticmethod
    def get_attention_entropy(weights):
        """Compute entropy of attention weights (higher = more spread)."""
        eps = 1e-10
        entropy = -np.sum(weights * np.log(weights + eps), axis=-1)
        return entropy

    @staticmethod
    def get_head_importance(all_layer_weights):
        """Rank heads by attention entropy variance (diverse heads = more important)."""
        importances = []
        for layer_idx, weights in enumerate(all_layer_weights):
            n_heads = weights.shape[1]
            for head_idx in range(n_heads):
                head_w = weights[:, head_idx, :, :]
                entropy = -np.sum(head_w * np.log(head_w + 1e-10), axis=-1)
                importance = np.std(entropy)
                importances.append((layer_idx, head_idx, float(importance)))
        importances.sort(key=lambda x: x[2], reverse=True)
        return importances

    @staticmethod
    def attention_rollout(all_layer_weights):
        """Attention rollout: multiply attention across layers for global view."""
        result = None
        for weights in all_layer_weights:
            # Average across heads
            avg = np.mean(weights, axis=1)  # (batch, seq, seq)
            # Add identity (residual connection)
            eye = np.eye(avg.shape[-1])
            avg = 0.5 * avg + 0.5 * eye
            # Renormalize
            avg = avg / avg.sum(axis=-1, keepdims=True)
            if result is None:
                result = avg
            else:
                result = np.matmul(result, avg)
        return result


# ============================================================
# Utilities
# ============================================================

def create_causal_mask(seq_len):
    """Create causal attention mask. True = masked (cannot attend)."""
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)


def create_padding_mask(token_ids, pad_id=0):
    """Create padding mask. True = masked (padding position)."""
    return (token_ids == pad_id)[:, np.newaxis, np.newaxis, :]


def count_parameters(model):
    """Count total number of parameters in model."""
    total = 0
    for p in model.parameters():
        total += p.size
    return total


def perplexity(loss):
    """Compute perplexity from cross-entropy loss."""
    return math.exp(min(loss, 100))  # Clip to avoid overflow
