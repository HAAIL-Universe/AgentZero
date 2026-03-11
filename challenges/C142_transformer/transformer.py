"""
C142: Transformer
Attention-based sequence model (Vaswani et al. "Attention Is All You Need").

Extends:
- C140 Neural Network (Tensor, Dense, Activation, optimizers, training)

Features:
- Scaled dot-product attention
- Multi-head attention (split/concat heads)
- Sinusoidal positional encoding
- Layer normalization
- Position-wise feed-forward network
- Encoder block (self-attention + FFN + residual + LayerNorm)
- Decoder block (self-attention + cross-attention + FFN + residual + LayerNorm)
- Full Transformer (encoder stack + decoder stack)
- Embedding layer with learned weights
- Causal masking for autoregressive decoding
- Greedy decoding
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))
from neural_network import (
    Tensor, Dense, Layer, Activation, softmax, softmax_batch,
    xavier_init, MSELoss, CrossEntropyLoss
)


# ============================================================
# Tensor extensions needed for transformer ops
# ============================================================

def tensor_slice_rows(t, start, end):
    """Slice rows [start:end] from a 2D tensor."""
    if len(t.shape) == 2:
        return Tensor([row[:] for row in t.data[start:end]])
    # 1D: slice elements
    return Tensor(t.data[start:end])


def tensor_slice_cols(t, start, end):
    """Slice columns [start:end] from a 2D tensor."""
    return Tensor([row[start:end] for row in t.data])


def tensor_concat_cols(tensors):
    """Concatenate list of 2D tensors along columns (axis=1)."""
    rows = tensors[0].shape[0]
    result = []
    for i in range(rows):
        row = []
        for t in tensors:
            row.extend(t.data[i])
        result.append(row)
    return Tensor(result)


def tensor_concat_rows(tensors):
    """Concatenate list of 2D tensors along rows (axis=0)."""
    result = []
    for t in tensors:
        if len(t.shape) == 2:
            result.extend([row[:] for row in t.data])
        else:
            result.append(t.data[:])
    return Tensor(result)


def tensor_outer(a, b):
    """Outer product of two 1D tensors -> 2D tensor."""
    return Tensor([[a.data[i] * b.data[j] for j in range(len(b.data))]
                   for i in range(len(a.data))])


# ============================================================
# Layer Normalization
# ============================================================

class LayerNorm(Layer):
    """Layer normalization (Ba et al. 2016).

    Normalizes across features (last dimension) for each sample.
    """

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # Learnable scale (gamma) and shift (beta)
        self.gamma = Tensor([1.0] * d_model)
        self.beta = Tensor([0.0] * d_model)
        # Gradients
        self.grad_gamma = None
        self.grad_beta = None
        # Cache
        self._input = None
        self._normalized = None
        self._std_inv = None
        self._mean = None

    def forward(self, x):
        """x: Tensor (seq_len, d_model) or (batch, seq_len, d_model) as list of 2D."""
        self._input = x
        if len(x.shape) == 1:
            # Single vector
            mu = sum(x.data) / len(x.data)
            var = sum((v - mu) ** 2 for v in x.data) / len(x.data)
            std_inv = 1.0 / math.sqrt(var + self.eps)
            self._mean = mu
            self._std_inv = std_inv
            normed = [(v - mu) * std_inv for v in x.data]
            self._normalized = Tensor(normed)
            return Tensor([self.gamma.data[i] * normed[i] + self.beta.data[i]
                           for i in range(self.d_model)])

        # 2D: (seq_len, d_model) -- normalize each row
        seq_len = x.shape[0]
        means = []
        std_invs = []
        normed_data = []
        for i in range(seq_len):
            row = x.data[i]
            mu = sum(row) / self.d_model
            var = sum((v - mu) ** 2 for v in row) / self.d_model
            si = 1.0 / math.sqrt(var + self.eps)
            means.append(mu)
            std_invs.append(si)
            normed_row = [(v - mu) * si for v in row]
            normed_data.append(normed_row)

        self._mean = means
        self._std_inv = std_invs
        self._normalized = Tensor(normed_data)

        out = []
        for i in range(seq_len):
            out.append([self.gamma.data[j] * normed_data[i][j] + self.beta.data[j]
                        for j in range(self.d_model)])
        return Tensor(out)

    def backward(self, grad_output):
        """Backward pass for layer norm."""
        if len(grad_output.shape) == 1:
            # Single vector
            self.grad_gamma = Tensor([grad_output.data[i] * self._normalized.data[i]
                                      for i in range(self.d_model)])
            self.grad_beta = Tensor(grad_output.data[:])
            # grad_input
            d = self.d_model
            dx_hat = [grad_output.data[i] * self.gamma.data[i] for i in range(d)]
            sum_dx = sum(dx_hat)
            sum_dx_x = sum(dx_hat[i] * self._normalized.data[i] for i in range(d))
            grad_input = []
            for i in range(d):
                gi = (dx_hat[i] - sum_dx / d - self._normalized.data[i] * sum_dx_x / d) * self._std_inv
                grad_input.append(gi)
            return Tensor(grad_input)

        # 2D
        seq_len = grad_output.shape[0]
        d = self.d_model

        # Accumulate gamma/beta gradients
        grad_gamma = [0.0] * d
        grad_beta = [0.0] * d
        grad_input_data = []

        for i in range(seq_len):
            dx_hat = [grad_output.data[i][j] * self.gamma.data[j] for j in range(d)]
            sum_dx = sum(dx_hat)
            sum_dx_x = sum(dx_hat[j] * self._normalized.data[i][j] for j in range(d))
            row = []
            for j in range(d):
                gi = (dx_hat[j] - sum_dx / d - self._normalized.data[i][j] * sum_dx_x / d) * self._std_inv[i]
                row.append(gi)
                grad_gamma[j] += grad_output.data[i][j] * self._normalized.data[i][j]
                grad_beta[j] += grad_output.data[i][j]
            grad_input_data.append(row)

        self.grad_gamma = Tensor(grad_gamma)
        self.grad_beta = Tensor(grad_beta)
        return Tensor(grad_input_data)

    def get_params(self):
        return [(self.gamma, self.grad_gamma, 'gamma'),
                (self.beta, self.grad_beta, 'beta')]


# ============================================================
# Embedding Layer
# ============================================================

class Embedding(Layer):
    """Learned embedding lookup table."""

    def __init__(self, vocab_size, d_model, rng=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Initialize embedding weights
        std = 1.0 / math.sqrt(d_model)
        self.weight = Tensor.random_normal((vocab_size, d_model), std=std, rng=rng)
        self.grad_weight = None
        self._input_indices = None

    def forward(self, indices):
        """
        indices: list of int (token IDs) for a single sequence
        Returns: Tensor (seq_len, d_model)
        """
        self._input_indices = indices
        return Tensor([self.weight.data[idx][:] for idx in indices])

    def backward(self, grad_output):
        """Accumulate gradients into embedding rows."""
        # Initialize grad to zero
        self.grad_weight = Tensor.zeros((self.vocab_size, self.d_model))
        for i, idx in enumerate(self._input_indices):
            for j in range(self.d_model):
                self.grad_weight.data[idx][j] += grad_output.data[i][j]
        return None  # No upstream gradient for embeddings

    def get_params(self):
        return [(self.weight, self.grad_weight, 'embedding')]


# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding:
    """Sinusoidal positional encoding (Vaswani et al.)."""

    def __init__(self, d_model, max_len=512):
        self.d_model = d_model
        self.max_len = max_len
        self._cache = {}

    def forward(self, seq_len):
        """Returns Tensor (seq_len, d_model) of positional encodings."""
        if seq_len in self._cache:
            # Return a copy
            cached = self._cache[seq_len]
            return Tensor([row[:] for row in cached.data])

        pe = []
        for pos in range(seq_len):
            row = [0.0] * self.d_model
            for i in range(0, self.d_model, 2):
                denom = math.pow(10000.0, i / self.d_model)
                row[i] = math.sin(pos / denom)
                if i + 1 < self.d_model:
                    row[i + 1] = math.cos(pos / denom)
            pe.append(row)

        result = Tensor(pe)
        self._cache[seq_len] = Tensor([row[:] for row in pe])
        return result


# ============================================================
# Scaled Dot-Product Attention
# ============================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute attention: softmax(Q @ K^T / sqrt(d_k)) @ V

    Q: (seq_len_q, d_k)
    K: (seq_len_k, d_k)
    V: (seq_len_k, d_v)
    mask: optional (seq_len_q, seq_len_k) -- positions to mask get -1e9

    Returns: (output, attention_weights)
      output: (seq_len_q, d_v)
      attention_weights: (seq_len_q, seq_len_k)
    """
    d_k = Q.shape[1] if len(Q.shape) == 2 else len(Q.data)
    scale = math.sqrt(d_k)

    # Q @ K^T -> (seq_len_q, seq_len_k)
    scores = Q.dot(K.T())

    # Scale
    seq_q = scores.shape[0]
    seq_k = scores.shape[1]
    scaled = [[scores.data[i][j] / scale for j in range(seq_k)] for i in range(seq_q)]

    # Apply mask
    if mask is not None:
        for i in range(seq_q):
            for j in range(seq_k):
                if mask.data[i][j] < -0.5:  # mask value is -1e9 or similar
                    scaled[i][j] += mask.data[i][j]

    # Softmax per row
    attn_weights = []
    for i in range(seq_q):
        attn_weights.append(softmax(scaled[i]))

    attn = Tensor(attn_weights)

    # attn @ V -> (seq_len_q, d_v)
    output = attn.dot(V)

    return output, attn


def scaled_dot_product_attention_backward(grad_output, Q, K, V, attn_weights, mask=None):
    """
    Backward pass for scaled dot-product attention.

    grad_output: (seq_len_q, d_v)
    Returns: (grad_Q, grad_K, grad_V)
    """
    d_k = Q.shape[1]
    scale = math.sqrt(d_k)
    seq_q = Q.shape[0]
    seq_k = K.shape[0]
    d_v = V.shape[1]

    # grad_V = attn^T @ grad_output
    grad_V = attn_weights.T().dot(grad_output)

    # grad_attn = grad_output @ V^T  (seq_q, seq_k)
    grad_attn = grad_output.dot(V.T())

    # Softmax backward: for each row
    grad_scores = []
    for i in range(seq_q):
        a = attn_weights.data[i]
        ga = grad_attn.data[i]
        # softmax grad: ds_j = a_j * (ga_j - sum_k(a_k * ga_k))
        dot_ag = sum(a[k] * ga[k] for k in range(seq_k))
        row = [(a[j] * (ga[j] - dot_ag)) / scale for j in range(seq_k)]
        grad_scores.append(row)

    grad_scores_t = Tensor(grad_scores)

    # grad_Q = grad_scores @ K
    grad_Q = grad_scores_t.dot(K)

    # grad_K = grad_scores^T @ Q
    grad_K = grad_scores_t.T().dot(Q)

    return grad_Q, grad_K, grad_V


# ============================================================
# Causal Mask
# ============================================================

def causal_mask(seq_len):
    """Create causal (look-ahead) mask. Future positions get -1e9."""
    mask = []
    for i in range(seq_len):
        row = [0.0] * seq_len
        for j in range(i + 1, seq_len):
            row[j] = -1e9
        mask.append(row)
    return Tensor(mask)


def padding_mask(lengths, max_len):
    """Create padding mask from sequence lengths. Padded positions get -1e9."""
    batch = len(lengths)
    masks = []
    for b in range(batch):
        row = [0.0 if j < lengths[b] else -1e9 for j in range(max_len)]
        masks.append(row)
    return Tensor(masks)


# ============================================================
# Multi-Head Attention
# ============================================================

class MultiHeadAttention(Layer):
    """Multi-head attention mechanism."""

    def __init__(self, d_model, num_heads, rng=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_Q = Dense(d_model, d_model, init='xavier', rng=rng, bias=False)
        self.W_K = Dense(d_model, d_model, init='xavier', rng=rng, bias=False)
        self.W_V = Dense(d_model, d_model, init='xavier', rng=rng, bias=False)
        self.W_O = Dense(d_model, d_model, init='xavier', rng=rng, bias=False)

        # Cache
        self._Q_proj = None
        self._K_proj = None
        self._V_proj = None
        self._attn_weights = []  # per head
        self._Q_heads = []
        self._K_heads = []
        self._V_heads = []

    def _split_heads(self, x):
        """Split (seq_len, d_model) -> list of (seq_len, d_k) per head."""
        seq_len = x.shape[0]
        heads = []
        for h in range(self.num_heads):
            start = h * self.d_k
            end = start + self.d_k
            head = tensor_slice_cols(x, start, end)
            heads.append(head)
        return heads

    def _concat_heads(self, heads):
        """Concat list of (seq_len, d_k) -> (seq_len, d_model)."""
        return tensor_concat_cols(heads)

    def forward(self, query, key, value, mask=None):
        """
        query: (seq_len_q, d_model)
        key: (seq_len_k, d_model)
        value: (seq_len_k, d_model)
        mask: optional (seq_len_q, seq_len_k)

        Returns: (seq_len_q, d_model)
        """
        # Project
        self._Q_proj = self.W_Q.forward(query)
        self._K_proj = self.W_K.forward(key)
        self._V_proj = self.W_V.forward(value)

        # Split into heads
        self._Q_heads = self._split_heads(self._Q_proj)
        self._K_heads = self._split_heads(self._K_proj)
        self._V_heads = self._split_heads(self._V_proj)

        # Attention per head
        head_outputs = []
        self._attn_weights = []
        for h in range(self.num_heads):
            out_h, attn_h = scaled_dot_product_attention(
                self._Q_heads[h], self._K_heads[h], self._V_heads[h], mask=mask
            )
            head_outputs.append(out_h)
            self._attn_weights.append(attn_h)

        # Concat heads
        concat = self._concat_heads(head_outputs)

        # Output projection
        output = self.W_O.forward(concat)
        return output

    def backward(self, grad_output):
        """Backward pass through multi-head attention."""
        # Through output projection
        grad_concat = self.W_O.backward(grad_output)

        # Split grad into heads
        grad_heads = self._split_heads(grad_concat)

        # Attention backward per head
        grad_Q_heads = []
        grad_K_heads = []
        grad_V_heads = []
        for h in range(self.num_heads):
            gQ, gK, gV = scaled_dot_product_attention_backward(
                grad_heads[h],
                self._Q_heads[h], self._K_heads[h], self._V_heads[h],
                self._attn_weights[h]
            )
            grad_Q_heads.append(gQ)
            grad_K_heads.append(gK)
            grad_V_heads.append(gV)

        # Concat head gradients
        grad_Q_proj = self._concat_heads(grad_Q_heads)
        grad_K_proj = self._concat_heads(grad_K_heads)
        grad_V_proj = self._concat_heads(grad_V_heads)

        # Through projections
        grad_query = self.W_Q.backward(grad_Q_proj)
        grad_key = self.W_K.backward(grad_K_proj)
        grad_value = self.W_V.backward(grad_V_proj)

        return grad_query, grad_key, grad_value

    def get_params(self):
        params = []
        params.extend(self.W_Q.get_params())
        params.extend(self.W_K.get_params())
        params.extend(self.W_V.get_params())
        params.extend(self.W_O.get_params())
        return params

    def get_attention_weights(self):
        """Return attention weights from last forward pass."""
        return self._attn_weights


# ============================================================
# Position-wise Feed-Forward Network
# ============================================================

class FeedForward(Layer):
    """Position-wise feed-forward network: FFN(x) = max(0, xW1+b1)W2+b2."""

    def __init__(self, d_model, d_ff, rng=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = Dense(d_model, d_ff, init='xavier', rng=rng)
        self.activation = Activation('relu')
        self.linear2 = Dense(d_ff, d_model, init='xavier', rng=rng)

    def forward(self, x):
        """x: (seq_len, d_model) -> (seq_len, d_model)"""
        h = self.linear1.forward(x)
        h = self.activation.forward(h)
        return self.linear2.forward(h)

    def backward(self, grad_output):
        grad = self.linear2.backward(grad_output)
        grad = self.activation.backward(grad)
        return self.linear1.backward(grad)

    def get_params(self):
        params = []
        params.extend(self.linear1.get_params())
        params.extend(self.linear2.get_params())
        return params


# ============================================================
# Encoder Block
# ============================================================

class EncoderBlock(Layer):
    """Single transformer encoder block.

    self-attention -> add & norm -> FFN -> add & norm
    """

    def __init__(self, d_model, num_heads, d_ff, rng=None):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, rng=rng)
        self.norm1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, rng=rng)
        self.norm2 = LayerNorm(d_model)
        self._input = None
        self._attn_out = None
        self._norm1_out = None

    def forward(self, x, mask=None):
        """x: (seq_len, d_model) -> (seq_len, d_model)"""
        self._input = x

        # Self-attention + residual + norm
        attn_out = self.self_attn.forward(x, x, x, mask=mask)
        self._attn_out = attn_out
        residual1 = x + attn_out
        self._norm1_out = self.norm1.forward(residual1)

        # FFN + residual + norm
        ffn_out = self.ffn.forward(self._norm1_out)
        residual2 = self._norm1_out + ffn_out
        out = self.norm2.forward(residual2)
        return out

    def backward(self, grad_output):
        # Through norm2
        grad_res2 = self.norm2.backward(grad_output)

        # Through FFN + residual
        grad_ffn = self.ffn.backward(grad_res2)
        grad_norm1_out = grad_res2 + grad_ffn  # residual connection

        # Through norm1
        grad_res1 = self.norm1.backward(grad_norm1_out)

        # Through self-attention + residual
        grad_q, grad_k, grad_v = self.self_attn.backward(grad_res1)
        # All three point to same input x
        grad_input = grad_res1 + grad_q + grad_k + grad_v  # residual + attn grads
        return grad_input

    def get_params(self):
        params = []
        params.extend(self.self_attn.get_params())
        params.extend(self.norm1.get_params())
        params.extend(self.ffn.get_params())
        params.extend(self.norm2.get_params())
        return params


# ============================================================
# Decoder Block
# ============================================================

class DecoderBlock(Layer):
    """Single transformer decoder block.

    masked self-attention -> add & norm -> cross-attention -> add & norm -> FFN -> add & norm
    """

    def __init__(self, d_model, num_heads, d_ff, rng=None):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, rng=rng)
        self.norm1 = LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, rng=rng)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, rng=rng)
        self.norm3 = LayerNorm(d_model)
        # Cache
        self._input = None
        self._encoder_output = None
        self._norm1_out = None
        self._norm2_out = None

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        x: (seq_len_tgt, d_model)
        encoder_output: (seq_len_src, d_model)
        src_mask: mask for cross-attention
        tgt_mask: causal mask for self-attention
        """
        self._input = x
        self._encoder_output = encoder_output

        # Masked self-attention
        attn1 = self.self_attn.forward(x, x, x, mask=tgt_mask)
        residual1 = x + attn1
        self._norm1_out = self.norm1.forward(residual1)

        # Cross-attention (query from decoder, key/value from encoder)
        attn2 = self.cross_attn.forward(self._norm1_out, encoder_output, encoder_output, mask=src_mask)
        residual2 = self._norm1_out + attn2
        self._norm2_out = self.norm2.forward(residual2)

        # FFN
        ffn_out = self.ffn.forward(self._norm2_out)
        residual3 = self._norm2_out + ffn_out
        out = self.norm3.forward(residual3)
        return out

    def backward(self, grad_output):
        # Through norm3
        grad_res3 = self.norm3.backward(grad_output)

        # Through FFN + residual
        grad_ffn = self.ffn.backward(grad_res3)
        grad_norm2_out = grad_res3 + grad_ffn

        # Through norm2
        grad_res2 = self.norm2.backward(grad_norm2_out)

        # Through cross-attention + residual
        grad_cq, grad_ck, grad_cv = self.cross_attn.backward(grad_res2)
        grad_norm1_out = grad_res2 + grad_cq
        grad_encoder = grad_ck + grad_cv

        # Through norm1
        grad_res1 = self.norm1.backward(grad_norm1_out)

        # Through self-attention + residual
        grad_sq, grad_sk, grad_sv = self.self_attn.backward(grad_res1)
        grad_input = grad_res1 + grad_sq + grad_sk + grad_sv

        return grad_input, grad_encoder

    def get_params(self):
        params = []
        params.extend(self.self_attn.get_params())
        params.extend(self.norm1.get_params())
        params.extend(self.cross_attn.get_params())
        params.extend(self.norm2.get_params())
        params.extend(self.ffn.get_params())
        params.extend(self.norm3.get_params())
        return params


# ============================================================
# Transformer Encoder
# ============================================================

class TransformerEncoder:
    """Stack of encoder blocks."""

    def __init__(self, num_layers, d_model, num_heads, d_ff, rng=None):
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(EncoderBlock(d_model, num_heads, d_ff, rng=rng))

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer.forward(x, mask=mask)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params


# ============================================================
# Transformer Decoder
# ============================================================

class TransformerDecoder:
    """Stack of decoder blocks."""

    def __init__(self, num_layers, d_model, num_heads, d_ff, rng=None):
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(DecoderBlock(d_model, num_heads, d_ff, rng=rng))

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer.forward(x, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
        return x

    def backward(self, grad):
        total_encoder_grad = None
        for layer in reversed(self.layers):
            grad, enc_grad = layer.backward(grad)
            if total_encoder_grad is None:
                total_encoder_grad = enc_grad
            else:
                total_encoder_grad = total_encoder_grad + enc_grad
        return grad, total_encoder_grad

    def get_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params


# ============================================================
# Full Transformer Model
# ============================================================

class Transformer:
    """Full encoder-decoder transformer.

    Architecture:
        src -> embedding + PE -> encoder -> encoder_output
        tgt -> embedding + PE -> decoder(encoder_output) -> linear -> logits
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64, num_heads=4,
                 d_ff=128, num_encoder_layers=2, num_decoder_layers=2,
                 max_len=128, rng=None):
        self.d_model = d_model
        self.src_vocab = src_vocab_size
        self.tgt_vocab = tgt_vocab_size

        # Embeddings
        self.src_embedding = Embedding(src_vocab_size, d_model, rng=rng)
        self.tgt_embedding = Embedding(tgt_vocab_size, d_model, rng=rng)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Encoder and decoder
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ff, rng=rng)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ff, rng=rng)

        # Output projection
        self.output_proj = Dense(d_model, tgt_vocab_size, init='xavier', rng=rng)

        # Scale factor for embeddings
        self.scale = math.sqrt(d_model)

        # Cache
        self._encoder_output = None

    def forward(self, src_tokens, tgt_tokens, src_mask=None, tgt_mask=None):
        """
        src_tokens: list of int (source token IDs)
        tgt_tokens: list of int (target token IDs)
        Returns: logits Tensor (tgt_len, tgt_vocab_size)
        """
        # Source embedding + positional encoding
        src_emb = self.src_embedding.forward(src_tokens)
        src_emb = src_emb * self.scale
        src_pe = self.pos_encoding.forward(len(src_tokens))
        src_input = src_emb + src_pe

        # Encode
        self._encoder_output = self.encoder.forward(src_input, mask=src_mask)

        # Target embedding + positional encoding
        tgt_emb = self.tgt_embedding.forward(tgt_tokens)
        tgt_emb = tgt_emb * self.scale
        tgt_pe = self.pos_encoding.forward(len(tgt_tokens))
        tgt_input = tgt_emb + tgt_pe

        # Auto-generate causal mask for target if not provided
        if tgt_mask is None:
            tgt_mask = causal_mask(len(tgt_tokens))

        # Decode
        dec_output = self.decoder.forward(tgt_input, self._encoder_output,
                                          src_mask=src_mask, tgt_mask=tgt_mask)

        # Project to vocabulary
        logits = self.output_proj.forward(dec_output)
        return logits

    def backward(self, grad_logits):
        """Backward pass through the full transformer."""
        # Through output projection
        grad_dec = self.output_proj.backward(grad_logits)

        # Through decoder
        grad_tgt_input, grad_encoder_output = self.decoder.backward(grad_dec)

        # Through encoder
        grad_src_input = self.encoder.backward(grad_encoder_output)

        # Through embeddings (scale gradients)
        grad_tgt_emb = grad_tgt_input * self.scale
        self.tgt_embedding.backward(grad_tgt_emb)

        grad_src_emb = grad_src_input * self.scale
        self.src_embedding.backward(grad_src_emb)

    def get_params(self):
        params = []
        params.extend(self.src_embedding.get_params())
        params.extend(self.tgt_embedding.get_params())
        params.extend(self.encoder.get_params())
        params.extend(self.decoder.get_params())
        params.extend(self.output_proj.get_params())
        return params

    def greedy_decode(self, src_tokens, max_len=50, start_token=1, end_token=2):
        """Greedy autoregressive decoding."""
        # Encode source
        src_emb = self.src_embedding.forward(src_tokens)
        src_emb = src_emb * self.scale
        src_pe = self.pos_encoding.forward(len(src_tokens))
        src_input = src_emb + src_pe
        encoder_output = self.encoder.forward(src_input)

        # Decode step by step
        output_tokens = [start_token]
        for _ in range(max_len):
            tgt_emb = self.tgt_embedding.forward(output_tokens)
            tgt_emb = tgt_emb * self.scale
            tgt_pe = self.pos_encoding.forward(len(output_tokens))
            tgt_input = tgt_emb + tgt_pe
            tgt_mask = causal_mask(len(output_tokens))

            dec_out = self.decoder.forward(tgt_input, encoder_output, tgt_mask=tgt_mask)

            # Get logits for last position
            last_hidden = Tensor(dec_out.data[-1]) if len(dec_out.shape) == 2 else dec_out
            logits = self.output_proj.forward(last_hidden)
            # Pick argmax
            if len(logits.shape) == 1:
                next_token = max(range(len(logits.data)), key=lambda i: logits.data[i])
            else:
                next_token = max(range(logits.shape[1]), key=lambda i: logits.data[0][i])

            output_tokens.append(next_token)
            if next_token == end_token:
                break

        return output_tokens


# ============================================================
# Encoder-Only Model (for classification, etc.)
# ============================================================

class TransformerClassifier:
    """Encoder-only transformer for classification tasks.

    Uses [CLS] token (first position) representation for classification.
    """

    def __init__(self, vocab_size, num_classes, d_model=64, num_heads=4,
                 d_ff=128, num_layers=2, max_len=128, rng=None):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        self.embedding = Embedding(vocab_size, d_model, rng=rng)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, rng=rng)
        self.classifier = Dense(d_model, num_classes, init='xavier', rng=rng)
        self.scale = math.sqrt(d_model)

        self._encoder_output = None

    def forward(self, tokens, mask=None):
        """
        tokens: list of int
        Returns: logits Tensor (num_classes,)
        """
        emb = self.embedding.forward(tokens)
        emb = emb * self.scale
        pe = self.pos_encoding.forward(len(tokens))
        x = emb + pe

        self._encoder_output = self.encoder.forward(x, mask=mask)

        # Use first token [CLS] representation
        cls_repr = Tensor(self._encoder_output.data[0])
        logits = self.classifier.forward(cls_repr)
        return logits

    def backward(self, grad_logits):
        grad_cls = self.classifier.backward(grad_logits)

        # Expand gradient back to full sequence (only CLS has gradient)
        seq_len = self._encoder_output.shape[0]
        grad_enc_out = Tensor.zeros((seq_len, self.d_model))
        grad_enc_out.data[0] = grad_cls.data[:]

        grad_input = self.encoder.backward(grad_enc_out)
        grad_emb = grad_input * self.scale
        self.embedding.backward(grad_emb)

    def get_params(self):
        params = []
        params.extend(self.embedding.get_params())
        params.extend(self.encoder.get_params())
        params.extend(self.classifier.get_params())
        return params


# ============================================================
# Simple Optimizer for Transformer (Adam)
# ============================================================

class TransformerAdam:
    """Adam optimizer for transformer models."""

    def __init__(self, params_fn, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0.0, warmup_steps=0):
        self.params_fn = params_fn
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.t = 0
        self.m = {}  # first moments
        self.v = {}  # second moments

    def step(self):
        self.t += 1
        params = self.params_fn()

        # Learning rate with warmup
        if self.warmup_steps > 0 and self.t <= self.warmup_steps:
            lr = self.lr * (self.t / self.warmup_steps)
        else:
            lr = self.lr

        for idx, param_tuple in enumerate(params):
            tensor, grad, name = param_tuple
            if grad is None:
                continue

            key = idx
            if key not in self.m:
                if len(tensor.shape) == 1:
                    self.m[key] = [0.0] * tensor.shape[0]
                    self.v[key] = [0.0] * tensor.shape[0]
                else:
                    self.m[key] = [[0.0] * tensor.shape[1] for _ in range(tensor.shape[0])]
                    self.v[key] = [[0.0] * tensor.shape[1] for _ in range(tensor.shape[0])]

            bc1 = 1.0 - self.beta1 ** self.t
            bc2 = 1.0 - self.beta2 ** self.t

            if len(tensor.shape) == 1:
                for i in range(tensor.shape[0]):
                    g = grad.data[i]
                    if self.weight_decay > 0:
                        g += self.weight_decay * tensor.data[i]
                    self.m[key][i] = self.beta1 * self.m[key][i] + (1 - self.beta1) * g
                    self.v[key][i] = self.beta2 * self.v[key][i] + (1 - self.beta2) * g * g
                    m_hat = self.m[key][i] / bc1
                    v_hat = self.v[key][i] / bc2
                    tensor.data[i] -= lr * m_hat / (math.sqrt(v_hat) + self.eps)
            else:
                for i in range(tensor.shape[0]):
                    for j in range(tensor.shape[1]):
                        g = grad.data[i][j]
                        if self.weight_decay > 0:
                            g += self.weight_decay * tensor.data[i][j]
                        self.m[key][i][j] = self.beta1 * self.m[key][i][j] + (1 - self.beta1) * g
                        self.v[key][i][j] = self.beta2 * self.v[key][i][j] + (1 - self.beta2) * g * g
                        m_hat = self.m[key][i][j] / bc1
                        v_hat = self.v[key][i][j] / bc2
                        tensor.data[i][j] -= lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Reset all gradients to None (they get recreated in backward)."""
        pass  # Gradients are recreated each backward pass


# ============================================================
# Training Utilities
# ============================================================

def train_transformer(model, data, loss_fn, optimizer, epochs=10):
    """
    Train a transformer model.

    data: list of (src_tokens, tgt_tokens, target_labels) tuples
      - target_labels: list of int (class indices for each position)
    """
    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for src, tgt, labels in data:
            # Forward
            logits = model.forward(src, tgt)

            # Compute loss (per position cross-entropy)
            loss = loss_fn.forward(logits, labels)
            total_loss += loss

            # Backward
            grad = loss_fn.backward(logits, labels)
            model.backward(grad)

            # Update
            optimizer.step()

        avg_loss = total_loss / len(data)
        history.append(avg_loss)

    return history


def train_classifier(model, data, loss_fn, optimizer, epochs=10):
    """
    Train a transformer classifier.

    data: list of (tokens, label) tuples
    """
    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        for tokens, label in data:
            logits = model.forward(tokens)
            loss = loss_fn.forward(logits, label)
            total_loss += loss

            grad = loss_fn.backward(logits, label)
            model.backward(grad)
            optimizer.step()

        avg_loss = total_loss / len(data)
        history.append(avg_loss)

    return history
