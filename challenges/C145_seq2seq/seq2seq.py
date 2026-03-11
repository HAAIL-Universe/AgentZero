"""
C145: Sequence-to-Sequence with Attention
Encoder-decoder architecture with Bahdanau and Luong attention, beam search.

Extends:
- C144 RNN (RNNCell, LSTMCell, GRUCell, Embedding, utilities)

Features:
- Encoder: bidirectional LSTM/GRU producing encoder outputs + final state
- Attention: Bahdanau (additive) and Luong (dot, general, concat) mechanisms
- Decoder: attention-augmented recurrent decoder with context vector
- Seq2Seq: full encoder-decoder model with teacher forcing
- Beam search: k-best decoding with length normalization
- Vocabulary: token <-> index mapping with special tokens (PAD, SOS, EOS, UNK)
- Data utilities: tokenization, batching, padding
- Training: cross-entropy loss over sequences, gradient clipping
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C144_rnn'))
from rnn import (
    LSTMCell, GRUCell, RNNCell,
    _mat_mul, _mat_vec, _vec_outer, _vec_add, _vec_mul, _vec_sub,
    _vec_scale, _sigmoid_vec, _tanh_vec, _zeros, _zeros_2d,
    _randn, _init_weight, _clip_vec, _clip_mat, _transpose,
)


# ============================================================
# Additional utilities
# ============================================================

def _softmax(v):
    """Softmax over a 1D list."""
    m = max(v)
    exps = [math.exp(x - m) for x in v]
    s = sum(exps)
    return [e / s for e in exps]


def _dot(a, b):
    """Dot product of two vectors."""
    return sum(a[i] * b[i] for i in range(len(a)))


def _concat(a, b):
    """Concatenate two vectors."""
    return a + b


def _log(x):
    """Safe log."""
    return math.log(max(x, 1e-12))


class Embedding:
    """Simple embedding: maps integer index to dense vector."""

    def __init__(self, vocab_size, embed_dim, rng=None):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        std = 1.0 / math.sqrt(embed_dim)
        self.weights = [[_randn(rng) * std for _ in range(embed_dim)]
                        for _ in range(vocab_size)]

    def forward(self, idx):
        """Look up embedding for a single integer index."""
        return self.weights[idx][:]


# ============================================================
# Vocabulary
# ============================================================

class Vocabulary:
    """Token <-> index mapping with special tokens."""

    PAD = '<PAD>'
    SOS = '<SOS>'
    EOS = '<EOS>'
    UNK = '<UNK>'

    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        self.size = 0
        # Add special tokens
        for tok in [self.PAD, self.SOS, self.EOS, self.UNK]:
            self._add(tok)
        self.pad_idx = self.token2idx[self.PAD]
        self.sos_idx = self.token2idx[self.SOS]
        self.eos_idx = self.token2idx[self.EOS]
        self.unk_idx = self.token2idx[self.UNK]

    def _add(self, token):
        if token not in self.token2idx:
            idx = self.size
            self.token2idx[token] = idx
            self.idx2token[idx] = token
            self.size += 1
        return self.token2idx[token]

    def build(self, sequences):
        """Build vocabulary from a list of token sequences."""
        for seq in sequences:
            for token in seq:
                self._add(token)
        return self

    def encode(self, tokens, add_sos=False, add_eos=False):
        """Convert tokens to indices."""
        indices = []
        if add_sos:
            indices.append(self.sos_idx)
        for t in tokens:
            indices.append(self.token2idx.get(t, self.unk_idx))
        if add_eos:
            indices.append(self.eos_idx)
        return indices

    def decode(self, indices, strip_special=True):
        """Convert indices to tokens."""
        tokens = []
        for idx in indices:
            tok = self.idx2token.get(idx, self.UNK)
            if strip_special and tok in (self.PAD, self.SOS, self.EOS):
                continue
            tokens.append(tok)
        return tokens


def pad_sequences(sequences, max_len=None, pad_value=0):
    """Pad sequences to uniform length."""
    if max_len is None:
        max_len = max(len(s) for s in sequences)
    result = []
    for seq in sequences:
        padded = seq[:max_len] + [pad_value] * max(0, max_len - len(seq))
        result.append(padded)
    return result


def tokenize(text):
    """Simple whitespace tokenizer."""
    return text.strip().split()


# ============================================================
# Encoder
# ============================================================

class Encoder:
    """Recurrent encoder producing encoder outputs and final state.

    Supports LSTM and GRU cells, with optional bidirectional mode.
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, cell_type='lstm',
                 bidirectional=False, num_layers=1, rng=None):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        self.embedding = Embedding(vocab_size, embed_dim, rng=rng)

        # Build cells for each layer
        self.fwd_cells = []
        self.bwd_cells = []
        for layer in range(num_layers):
            inp_size = embed_dim if layer == 0 else (hidden_size * 2 if bidirectional else hidden_size)
            CellClass = LSTMCell if cell_type == 'lstm' else GRUCell
            self.fwd_cells.append(CellClass(inp_size, hidden_size, rng=rng))
            if bidirectional:
                self.bwd_cells.append(CellClass(inp_size, hidden_size, rng=rng))

    def forward(self, input_indices):
        """Encode a sequence of token indices.

        Args:
            input_indices: list of integer token indices (length T)

        Returns:
            encoder_outputs: list of T vectors, each of size hidden_size (*2 if bidirectional)
            final_state: (h, c) for LSTM or h for GRU -- final hidden state
            cache: dict for backpropagation
        """
        T = len(input_indices)
        hs = self.hidden_size
        is_lstm = (self.cell_type == 'lstm')

        # Embed
        embedded = [self.embedding.forward(idx) for idx in input_indices]

        cache = {'input_indices': input_indices, 'embedded': embedded,
                 'layer_caches': [], 'layer_outputs': []}

        layer_input = embedded

        final_h = None
        final_c = None

        for layer in range(self.num_layers):
            fwd_cell = self.fwd_cells[layer]

            # Forward pass
            fwd_h = _zeros(hs)
            fwd_c = _zeros(hs) if is_lstm else None
            fwd_outputs = []
            fwd_caches = []

            for t in range(T):
                x = layer_input[t]
                if is_lstm:
                    fwd_h, fwd_c, c = fwd_cell.forward(x, fwd_h, fwd_c)
                    fwd_caches.append(c)
                else:
                    fwd_h, c = fwd_cell.forward(x, fwd_h)
                    fwd_caches.append(c)
                fwd_outputs.append(fwd_h[:])

            bwd_outputs = None
            bwd_caches = []
            bwd_h = None
            bwd_c = None

            if self.bidirectional:
                bwd_cell = self.bwd_cells[layer]
                bwd_h = _zeros(hs)
                bwd_c = _zeros(hs) if is_lstm else None
                bwd_outputs = [None] * T

                for t in range(T - 1, -1, -1):
                    x = layer_input[t]
                    if is_lstm:
                        bwd_h, bwd_c, c = bwd_cell.forward(x, bwd_h, bwd_c)
                        bwd_caches.append(c)
                    else:
                        bwd_h, c = bwd_cell.forward(x, bwd_h)
                        bwd_caches.append(c)
                    bwd_outputs[t] = bwd_h[:]

                # Concatenate forward and backward
                layer_output = [_concat(fwd_outputs[t], bwd_outputs[t]) for t in range(T)]
            else:
                layer_output = fwd_outputs

            cache['layer_caches'].append({
                'fwd_caches': fwd_caches,
                'bwd_caches': bwd_caches,
                'fwd_outputs': fwd_outputs,
                'bwd_outputs': bwd_outputs,
            })
            cache['layer_outputs'].append(layer_output)

            layer_input = layer_output
            final_h = fwd_h
            final_c = fwd_c

        encoder_outputs = layer_input  # Final layer outputs

        if is_lstm:
            final_state = (final_h, final_c)
        else:
            final_state = final_h

        cache['final_state'] = final_state
        return encoder_outputs, final_state, cache

    def backward(self, d_encoder_outputs, d_final_state, cache):
        """Backward pass through encoder."""
        T = len(cache['input_indices'])
        hs = self.hidden_size
        is_lstm = (self.cell_type == 'lstm')

        grads = {}

        # Work backward through layers
        d_layer_output = d_encoder_outputs

        for layer in range(self.num_layers - 1, -1, -1):
            lc = cache['layer_caches'][layer]
            fwd_cell = self.fwd_cells[layer]

            if self.bidirectional:
                # Split gradients for forward and backward
                d_fwd = [d_layer_output[t][:hs] for t in range(T)]
                d_bwd = [d_layer_output[t][hs:] for t in range(T)]
            else:
                d_fwd = d_layer_output

            # Forward cell backward
            d_h = _zeros(hs)
            d_c = _zeros(hs) if is_lstm else None

            # Add final state gradient for the top layer
            if layer == self.num_layers - 1 and d_final_state is not None:
                if is_lstm:
                    d_h = _vec_add(d_h, d_final_state[0])
                    d_c = _vec_add(d_c, d_final_state[1])
                else:
                    d_h = _vec_add(d_h, d_final_state)

            fwd_grads_list = []
            d_layer_input = [None] * T

            for t in range(T - 1, -1, -1):
                d_h = _vec_add(d_h, d_fwd[t])
                fc = lc['fwd_caches'][t]

                if is_lstm:
                    dx, d_h, d_c, g = fwd_cell.backward(d_h, d_c, fc)
                else:
                    dx, d_h, g = fwd_cell.backward(d_h, fc)

                fwd_grads_list.append(g)
                d_layer_input[t] = dx

            # Backward cell backward (if bidirectional)
            if self.bidirectional:
                bwd_cell = self.bwd_cells[layer]
                d_h_bwd = _zeros(hs)
                d_c_bwd = _zeros(hs) if is_lstm else None

                # bwd_caches were appended in reverse time order
                bwd_caches_time_order = list(reversed(lc['bwd_caches']))

                for t in range(T):
                    d_h_bwd = _vec_add(d_h_bwd, d_bwd[t])
                    bc = bwd_caches_time_order[t]

                    if is_lstm:
                        dx, d_h_bwd, d_c_bwd, g = bwd_cell.backward(d_h_bwd, d_c_bwd, bc)
                    else:
                        dx, d_h_bwd, g = bwd_cell.backward(d_h_bwd, bc)

                    d_layer_input[t] = _vec_add(d_layer_input[t], dx)
                    fwd_grads_list.append(g)

            # Store accumulated gradients
            grads[f'layer_{layer}'] = fwd_grads_list
            d_layer_output = d_layer_input

        # Embedding gradients
        d_embed = {}
        for t in range(T):
            idx = cache['input_indices'][t]
            if idx not in d_embed:
                d_embed[idx] = _zeros(self.embed_dim)
            d_embed[idx] = _vec_add(d_embed[idx], d_layer_output[t])

        grads['d_embed'] = d_embed
        return grads

    def params(self):
        """Return all parameters."""
        p = {'embedding': self.embedding.weights}
        for layer in range(self.num_layers):
            p[f'fwd_cell_{layer}'] = self.fwd_cells[layer].params()
            if self.bidirectional:
                p[f'bwd_cell_{layer}'] = self.bwd_cells[layer].params()
        return p


# ============================================================
# Attention Mechanisms
# ============================================================

class BahdanauAttention:
    """Additive attention (Bahdanau et al., 2015).

    score(h_t, h_s) = v^T * tanh(W_a @ h_t + U_a @ h_s)

    Where h_t is the decoder hidden state and h_s are encoder outputs.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim, rng=None):
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim

        self.W_a = _init_weight(attention_dim, decoder_dim, decoder_dim, attention_dim, rng)
        self.U_a = _init_weight(attention_dim, encoder_dim, encoder_dim, attention_dim, rng)
        self.v = [_randn(rng) * math.sqrt(2.0 / attention_dim) for _ in range(attention_dim)]

    def forward(self, decoder_hidden, encoder_outputs):
        """Compute attention weights and context vector.

        Args:
            decoder_hidden: decoder hidden state [decoder_dim]
            encoder_outputs: list of T encoder output vectors [encoder_dim]

        Returns:
            context: weighted sum of encoder outputs [encoder_dim]
            weights: attention weights [T]
            cache: for backprop
        """
        T = len(encoder_outputs)

        # W_a @ decoder_hidden  [attention_dim]
        Wh = _mat_vec(self.W_a, decoder_hidden)

        # For each encoder output, compute score
        scores = []
        tanh_values = []
        Ue_list = []
        for t in range(T):
            Ue = _mat_vec(self.U_a, encoder_outputs[t])
            Ue_list.append(Ue)
            pre_tanh = _vec_add(Wh, Ue)
            tanh_val = _tanh_vec(pre_tanh)
            tanh_values.append(tanh_val)
            score = _dot(self.v, tanh_val)
            scores.append(score)

        weights = _softmax(scores)

        # Context = weighted sum of encoder outputs
        context = _zeros(self.encoder_dim)
        for t in range(T):
            context = _vec_add(context, _vec_scale(encoder_outputs[t], weights[t]))

        cache = {
            'decoder_hidden': decoder_hidden,
            'encoder_outputs': encoder_outputs,
            'Wh': Wh,
            'Ue_list': Ue_list,
            'tanh_values': tanh_values,
            'scores': scores,
            'weights': weights,
        }
        return context, weights, cache

    def backward(self, d_context, d_weights_ext, cache):
        """Backward pass through attention.

        Args:
            d_context: gradient of context vector [encoder_dim]
            d_weights_ext: external gradient on weights (can be None)
            cache: from forward pass

        Returns:
            d_decoder_hidden, d_encoder_outputs, grads
        """
        T = len(cache['encoder_outputs'])
        weights = cache['weights']
        encoder_outputs = cache['encoder_outputs']
        decoder_hidden = cache['decoder_hidden']
        tanh_values = cache['tanh_values']

        # d_context -> d_weights and d_encoder_outputs
        d_weights = [0.0] * T
        d_encoder_outputs = [_zeros(self.encoder_dim) for _ in range(T)]

        for t in range(T):
            # From context = sum(w_t * e_t)
            d_weights[t] += _dot(d_context, encoder_outputs[t])
            d_encoder_outputs[t] = _vec_add(d_encoder_outputs[t],
                                             _vec_scale(d_context, weights[t]))

        if d_weights_ext is not None:
            d_weights = _vec_add(d_weights, d_weights_ext)

        # Softmax backward: d_scores from d_weights
        # d_score_i = sum_j (weights_j * (delta_ij - weights_i) * d_weights_j)
        d_scores = [0.0] * T
        for i in range(T):
            for j in range(T):
                if i == j:
                    d_scores[i] += weights[i] * (1.0 - weights[i]) * d_weights[j]
                else:
                    d_scores[i] += -weights[j] * weights[i] * d_weights[j]

        # d_scores -> d_v, d_tanh_values
        d_v = _zeros(self.attention_dim)
        d_decoder_hidden = _zeros(self.decoder_dim)
        dW_a = _zeros_2d(self.attention_dim, self.decoder_dim)
        dU_a = _zeros_2d(self.attention_dim, self.encoder_dim)

        for t in range(T):
            # score_t = dot(v, tanh_val_t)
            d_tanh = _vec_scale(self.v, d_scores[t])
            d_v = _vec_add(d_v, _vec_scale(tanh_values[t], d_scores[t]))

            # Through tanh: d_pre = d_tanh * (1 - tanh^2)
            d_pre = [d_tanh[k] * (1.0 - tanh_values[t][k] ** 2) for k in range(self.attention_dim)]

            # d_pre -> dW_a, dU_a, d_decoder_hidden, d_encoder_output
            # pre = W_a @ decoder_hidden + U_a @ encoder_output
            outer_dh = _vec_outer(d_pre, decoder_hidden)
            outer_eo = _vec_outer(d_pre, encoder_outputs[t])
            dW_a = [[dW_a[r][c] + outer_dh[r][c] for c in range(self.decoder_dim)]
                     for r in range(self.attention_dim)]
            dU_a = [[dU_a[r][c] + outer_eo[r][c] for c in range(self.encoder_dim)]
                     for r in range(self.attention_dim)]

            d_decoder_hidden = _vec_add(d_decoder_hidden, _mat_vec(_transpose(self.W_a), d_pre))
            d_encoder_outputs[t] = _vec_add(d_encoder_outputs[t],
                                             _mat_vec(_transpose(self.U_a), d_pre))

        grads = {'dW_a': dW_a, 'dU_a': dU_a, 'dv': d_v}
        return d_decoder_hidden, d_encoder_outputs, grads

    def params(self):
        return {'W_a': self.W_a, 'U_a': self.U_a, 'v': self.v}


class LuongAttention:
    """Multiplicative attention (Luong et al., 2015).

    Supports three scoring functions:
    - 'dot': score = h_t . h_s
    - 'general': score = h_t . W @ h_s
    - 'concat': score = v . tanh(W @ [h_t; h_s])
    """

    def __init__(self, encoder_dim, decoder_dim, method='dot', rng=None):
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.method = method

        if method == 'general':
            self.W = _init_weight(decoder_dim, encoder_dim, encoder_dim, decoder_dim, rng)
        elif method == 'concat':
            concat_dim = encoder_dim + decoder_dim
            self.W = _init_weight(decoder_dim, concat_dim, concat_dim, decoder_dim, rng)
            self.v = [_randn(rng) * math.sqrt(2.0 / decoder_dim) for _ in range(decoder_dim)]

    def forward(self, decoder_hidden, encoder_outputs):
        """Compute attention weights and context vector."""
        T = len(encoder_outputs)

        scores = []
        score_caches = []

        for t in range(T):
            if self.method == 'dot':
                # Dimensions must match for dot product
                score = _dot(decoder_hidden, encoder_outputs[t])
                score_caches.append(None)
            elif self.method == 'general':
                We = _mat_vec(self.W, encoder_outputs[t])
                score = _dot(decoder_hidden, We)
                score_caches.append(We)
            elif self.method == 'concat':
                cat = _concat(decoder_hidden, encoder_outputs[t])
                Wc = _mat_vec(self.W, cat)
                tanh_val = _tanh_vec(Wc)
                score = _dot(self.v, tanh_val)
                score_caches.append((cat, Wc, tanh_val))
            scores.append(score)

        weights = _softmax(scores)

        context = _zeros(self.encoder_dim)
        for t in range(T):
            context = _vec_add(context, _vec_scale(encoder_outputs[t], weights[t]))

        cache = {
            'decoder_hidden': decoder_hidden,
            'encoder_outputs': encoder_outputs,
            'scores': scores,
            'weights': weights,
            'score_caches': score_caches,
        }
        return context, weights, cache

    def backward(self, d_context, d_weights_ext, cache):
        """Backward pass through Luong attention."""
        T = len(cache['encoder_outputs'])
        weights = cache['weights']
        encoder_outputs = cache['encoder_outputs']
        decoder_hidden = cache['decoder_hidden']

        # d_context -> d_weights, d_encoder_outputs
        d_weights = [0.0] * T
        d_encoder_outputs = [_zeros(self.encoder_dim) for _ in range(T)]

        for t in range(T):
            d_weights[t] += _dot(d_context, encoder_outputs[t])
            d_encoder_outputs[t] = _vec_add(d_encoder_outputs[t],
                                             _vec_scale(d_context, weights[t]))

        if d_weights_ext is not None:
            d_weights = _vec_add(d_weights, d_weights_ext)

        # Softmax backward
        d_scores = [0.0] * T
        for i in range(T):
            for j in range(T):
                if i == j:
                    d_scores[i] += weights[i] * (1.0 - weights[i]) * d_weights[j]
                else:
                    d_scores[i] += -weights[j] * weights[i] * d_weights[j]

        # Score backward
        d_decoder_hidden = _zeros(self.decoder_dim)
        grads = {}

        if self.method == 'dot':
            for t in range(T):
                d_decoder_hidden = _vec_add(d_decoder_hidden,
                                             _vec_scale(encoder_outputs[t], d_scores[t]))
                d_encoder_outputs[t] = _vec_add(d_encoder_outputs[t],
                                                 _vec_scale(decoder_hidden, d_scores[t]))

        elif self.method == 'general':
            dW = _zeros_2d(self.decoder_dim, self.encoder_dim)
            for t in range(T):
                We = cache['score_caches'][t]
                # score = dot(dh, W @ e)
                d_decoder_hidden = _vec_add(d_decoder_hidden,
                                             _vec_scale(We, d_scores[t]))
                d_We = _vec_scale(decoder_hidden, d_scores[t])
                d_encoder_outputs[t] = _vec_add(d_encoder_outputs[t],
                                                 _mat_vec(_transpose(self.W), d_We))
                outer = _vec_outer(d_We, encoder_outputs[t])
                dW = [[dW[r][c] + outer[r][c] for c in range(self.encoder_dim)]
                       for r in range(self.decoder_dim)]
            grads['dW'] = dW

        elif self.method == 'concat':
            dW = _zeros_2d(self.decoder_dim, self.encoder_dim + self.decoder_dim)
            dv = _zeros(self.decoder_dim)
            for t in range(T):
                cat, Wc, tanh_val = cache['score_caches'][t]
                # score = dot(v, tanh(W @ cat))
                d_tanh = _vec_scale(self.v, d_scores[t])
                dv = _vec_add(dv, _vec_scale(tanh_val, d_scores[t]))
                d_pre = [d_tanh[k] * (1.0 - tanh_val[k] ** 2) for k in range(self.decoder_dim)]
                outer = _vec_outer(d_pre, cat)
                dW = [[dW[r][c] + outer[r][c] for c in range(self.encoder_dim + self.decoder_dim)]
                       for r in range(self.decoder_dim)]
                d_cat = _mat_vec(_transpose(self.W), d_pre)
                d_decoder_hidden = _vec_add(d_decoder_hidden, d_cat[:self.decoder_dim])
                d_encoder_outputs[t] = _vec_add(d_encoder_outputs[t], d_cat[self.decoder_dim:])
            grads['dW'] = dW
            grads['dv'] = dv

        return d_decoder_hidden, d_encoder_outputs, grads

    def params(self):
        p = {}
        if self.method in ('general', 'concat'):
            p['W'] = self.W
        if self.method == 'concat':
            p['v'] = self.v
        return p


# ============================================================
# Decoder
# ============================================================

class Decoder:
    """Attention-augmented recurrent decoder.

    At each step:
    1. Embed the input token
    2. RNN step with [embedding; previous_context] as input
    3. Compute attention over encoder outputs
    4. Concatenate RNN output with context, project to output
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, encoder_dim,
                 attention_type='bahdanau', attention_dim=None,
                 cell_type='lstm', rng=None):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.encoder_dim = encoder_dim
        self.cell_type = cell_type

        self.embedding = Embedding(vocab_size, embed_dim, rng=rng)

        # RNN input is [embed_dim + encoder_dim] (feeding context)
        rnn_input_size = embed_dim + encoder_dim
        CellClass = LSTMCell if cell_type == 'lstm' else GRUCell
        self.cell = CellClass(rnn_input_size, hidden_size, rng=rng)

        # Attention
        if attention_dim is None:
            attention_dim = hidden_size
        if attention_type == 'bahdanau':
            self.attention = BahdanauAttention(encoder_dim, hidden_size, attention_dim, rng=rng)
        elif attention_type.startswith('luong'):
            method = 'dot'
            if '_' in attention_type:
                method = attention_type.split('_', 1)[1]
            self.attention = LuongAttention(encoder_dim, hidden_size, method=method, rng=rng)
        self.attention_type = attention_type

        # Output projection: [hidden_size + encoder_dim] -> vocab_size
        out_input_size = hidden_size + encoder_dim
        self.W_out = _init_weight(vocab_size, out_input_size, out_input_size, vocab_size, rng)
        self.b_out = _zeros(vocab_size)

    def forward_step(self, input_idx, h_prev, c_prev, prev_context, encoder_outputs):
        """Single decoder step.

        Returns:
            output_logits: [vocab_size]
            h_new, c_new: new hidden state (c_new is None for GRU)
            context: new context vector
            cache: for backprop
        """
        is_lstm = (self.cell_type == 'lstm')

        # Embed
        embed = self.embedding.forward(input_idx)

        # RNN input = [embed; prev_context]
        rnn_input = _concat(embed, prev_context)

        # RNN step
        if is_lstm:
            h_new, c_new, cell_cache = self.cell.forward(rnn_input, h_prev, c_prev)
        else:
            h_new, cell_cache = self.cell.forward(rnn_input, h_prev)
            c_new = None

        # Attention
        context, attn_weights, attn_cache = self.attention.forward(h_new, encoder_outputs)

        # Output projection
        concat_out = _concat(h_new, context)
        logits = _vec_add(_mat_vec(self.W_out, concat_out), self.b_out)

        cache = {
            'input_idx': input_idx,
            'embed': embed,
            'rnn_input': rnn_input,
            'cell_cache': cell_cache,
            'attn_cache': attn_cache,
            'concat_out': concat_out,
            'h_new': h_new,
            'context': context,
            'attn_weights': attn_weights,
            'h_prev': h_prev,
            'c_prev': c_prev,
            'prev_context': prev_context,
        }
        return logits, h_new, c_new, context, cache

    def backward_step(self, d_logits, d_h_next, d_c_next, d_context_next, cache):
        """Backward through one decoder step.

        Returns:
            d_h_prev, d_c_prev, d_prev_context, d_encoder_outputs, grads
        """
        is_lstm = (self.cell_type == 'lstm')
        concat_out = cache['concat_out']

        # Output projection backward
        d_concat_out = _mat_vec(_transpose(self.W_out), d_logits)
        dW_out = _vec_outer(d_logits, concat_out)
        db_out = d_logits[:]

        # Split d_concat_out -> d_h_attn, d_context
        d_h_from_out = d_concat_out[:self.hidden_size]
        d_context = d_concat_out[self.hidden_size:]

        # Add context gradient from next step
        if d_context_next is not None:
            d_context = _vec_add(d_context, d_context_next)

        # Attention backward
        d_h_from_attn, d_encoder_outputs, attn_grads = self.attention.backward(
            d_context, None, cache['attn_cache'])

        # Total d_h_new
        d_h = _vec_add(d_h_from_out, d_h_from_attn)
        if d_h_next is not None:
            d_h = _vec_add(d_h, d_h_next)

        # RNN cell backward
        if is_lstm:
            d_c = d_c_next if d_c_next is not None else _zeros(self.hidden_size)
            d_rnn_input, d_h_prev, d_c_prev, cell_grads = self.cell.backward(d_h, d_c, cache['cell_cache'])
        else:
            d_rnn_input, d_h_prev, cell_grads = self.cell.backward(d_h, cache['cell_cache'])
            d_c_prev = None

        # Split d_rnn_input -> d_embed, d_prev_context
        d_embed = d_rnn_input[:self.embed_dim]
        d_prev_context = d_rnn_input[self.embed_dim:]

        grads = {
            'dW_out': dW_out,
            'db_out': db_out,
            'cell_grads': cell_grads,
            'attn_grads': attn_grads,
            'd_embed': (cache['input_idx'], d_embed),
        }
        return d_h_prev, d_c_prev, d_prev_context, d_encoder_outputs, grads

    def params(self):
        p = {
            'embedding': self.embedding.weights,
            'cell': self.cell.params(),
            'attention': self.attention.params(),
            'W_out': self.W_out,
            'b_out': self.b_out,
        }
        return p


# ============================================================
# Seq2Seq Model
# ============================================================

class Seq2Seq:
    """Full encoder-decoder sequence-to-sequence model.

    Supports:
    - Teacher forcing (with configurable ratio)
    - Greedy decoding
    - Beam search decoding
    - Bridge network (encoder final state -> decoder initial state)
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, hidden_size,
                 encoder_layers=1, bidirectional=False,
                 attention_type='bahdanau', attention_dim=None,
                 cell_type='lstm', rng=None):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.bidirectional = bidirectional

        encoder_dim = hidden_size * 2 if bidirectional else hidden_size

        self.encoder = Encoder(
            src_vocab_size, embed_dim, hidden_size,
            cell_type=cell_type, bidirectional=bidirectional,
            num_layers=encoder_layers, rng=rng
        )

        self.decoder = Decoder(
            tgt_vocab_size, embed_dim, hidden_size, encoder_dim,
            attention_type=attention_type, attention_dim=attention_dim,
            cell_type=cell_type, rng=rng
        )

        # Bridge: project encoder final state to decoder initial state
        if bidirectional:
            self.bridge_h = _init_weight(hidden_size, hidden_size * 2, hidden_size * 2, hidden_size, rng)
            if cell_type == 'lstm':
                self.bridge_c = _init_weight(hidden_size, hidden_size * 2, hidden_size * 2, hidden_size, rng)
        else:
            self.bridge_h = None
            self.bridge_c = None

    def _bridge_state(self, encoder_final_state):
        """Project encoder final state for decoder initialization."""
        is_lstm = (self.cell_type == 'lstm')

        if self.bidirectional and self.bridge_h is not None:
            if is_lstm:
                h, c = encoder_final_state
                # For bidirectional, we need to handle the states differently
                # Just use forward final state and project
                dec_h = _mat_vec(self.bridge_h, _concat(h, _zeros(self.hidden_size)))
                dec_c = _mat_vec(self.bridge_c, _concat(c, _zeros(self.hidden_size)))
                return dec_h, dec_c
            else:
                h = encoder_final_state
                dec_h = _mat_vec(self.bridge_h, _concat(h, _zeros(self.hidden_size)))
                return dec_h
        else:
            return encoder_final_state

    def forward(self, src_indices, tgt_indices, teacher_forcing_ratio=1.0):
        """Forward pass with teacher forcing.

        Args:
            src_indices: source token indices [src_len]
            tgt_indices: target token indices [tgt_len] (includes SOS, EOS)
            teacher_forcing_ratio: probability of using ground truth as next input

        Returns:
            all_logits: list of tgt_len-1 logit vectors [vocab_size]
            cache: for backprop
        """
        is_lstm = (self.cell_type == 'lstm')
        encoder_dim = self.hidden_size * 2 if self.bidirectional else self.hidden_size

        # Encode
        encoder_outputs, encoder_final, enc_cache = self.encoder.forward(src_indices)

        # Bridge
        dec_state = self._bridge_state(encoder_final)

        if is_lstm:
            h, c = dec_state
        else:
            h = dec_state
            c = None

        # Decode
        context = _zeros(encoder_dim)
        all_logits = []
        all_caches = []
        input_idx = tgt_indices[0]  # SOS

        for t in range(len(tgt_indices) - 1):
            logits, h, c, context, step_cache = self.decoder.forward_step(
                input_idx, h, c, context, encoder_outputs)
            all_logits.append(logits)
            all_caches.append(step_cache)

            # Teacher forcing
            if random.random() < teacher_forcing_ratio:
                input_idx = tgt_indices[t + 1]
            else:
                input_idx = max(range(len(logits)), key=lambda i: logits[i])

        cache = {
            'enc_cache': enc_cache,
            'dec_caches': all_caches,
            'encoder_outputs': encoder_outputs,
            'encoder_final': encoder_final,
            'src_indices': src_indices,
            'tgt_indices': tgt_indices,
        }
        return all_logits, cache

    def backward(self, d_logits_list, cache):
        """Backward pass through the full model.

        Args:
            d_logits_list: list of gradients on logits (one per decoder step)
            cache: from forward pass

        Returns:
            grads: dict of all parameter gradients
        """
        is_lstm = (self.cell_type == 'lstm')
        T_dec = len(d_logits_list)
        encoder_dim = self.hidden_size * 2 if self.bidirectional else self.hidden_size

        d_h = _zeros(self.hidden_size)
        d_c = _zeros(self.hidden_size) if is_lstm else None
        d_context = _zeros(encoder_dim)
        d_encoder_outputs_total = [_zeros(encoder_dim) for _ in range(len(cache['encoder_outputs']))]

        dec_grads_list = []

        for t in range(T_dec - 1, -1, -1):
            d_h, d_c, d_context, d_enc_out, step_grads = self.decoder.backward_step(
                d_logits_list[t], d_h if t < T_dec - 1 else None,
                d_c if (t < T_dec - 1 and is_lstm) else None,
                d_context if t < T_dec - 1 else None,
                cache['dec_caches'][t])
            dec_grads_list.append(step_grads)

            for i in range(len(d_enc_out)):
                d_encoder_outputs_total[i] = _vec_add(d_encoder_outputs_total[i], d_enc_out[i])

        # Bridge backward (simplified -- just pass through for non-bidirectional)
        if is_lstm:
            d_final_state = (d_h, d_c)
        else:
            d_final_state = d_h

        # Encoder backward
        enc_grads = self.encoder.backward(d_encoder_outputs_total, d_final_state, cache['enc_cache'])

        return {'encoder': enc_grads, 'decoder': dec_grads_list}

    def greedy_decode(self, src_indices, sos_idx, eos_idx, max_len=50):
        """Greedy decoding (always pick highest probability token)."""
        is_lstm = (self.cell_type == 'lstm')
        encoder_dim = self.hidden_size * 2 if self.bidirectional else self.hidden_size

        encoder_outputs, encoder_final, _ = self.encoder.forward(src_indices)
        dec_state = self._bridge_state(encoder_final)

        if is_lstm:
            h, c = dec_state
        else:
            h = dec_state
            c = None

        context = _zeros(encoder_dim)
        output_indices = []
        all_weights = []
        input_idx = sos_idx

        for _ in range(max_len):
            logits, h, c, context, step_cache = self.decoder.forward_step(
                input_idx, h, c, context, encoder_outputs)
            probs = _softmax(logits)
            best_idx = max(range(len(probs)), key=lambda i: probs[i])
            output_indices.append(best_idx)
            all_weights.append(step_cache['attn_weights'])

            if best_idx == eos_idx:
                break
            input_idx = best_idx

        return output_indices, all_weights

    def beam_search(self, src_indices, sos_idx, eos_idx, beam_width=3,
                    max_len=50, length_penalty=0.6):
        """Beam search decoding with length normalization.

        Args:
            src_indices: source token indices
            sos_idx: start-of-sequence token index
            eos_idx: end-of-sequence token index
            beam_width: number of beams to maintain
            max_len: maximum output length
            length_penalty: alpha for length normalization (score / len^alpha)

        Returns:
            best_sequence: list of token indices
            best_score: normalized log probability
        """
        is_lstm = (self.cell_type == 'lstm')
        encoder_dim = self.hidden_size * 2 if self.bidirectional else self.hidden_size

        encoder_outputs, encoder_final, _ = self.encoder.forward(src_indices)
        dec_state = self._bridge_state(encoder_final)

        if is_lstm:
            h0, c0 = dec_state
        else:
            h0 = dec_state
            c0 = None

        context0 = _zeros(encoder_dim)

        # Each beam: (score, tokens, h, c, context)
        beams = [(0.0, [sos_idx], h0, c0, context0)]
        completed = []

        for step in range(max_len):
            candidates = []
            for score, tokens, h, c, context in beams:
                if tokens[-1] == eos_idx:
                    completed.append((score, tokens))
                    continue

                input_idx = tokens[-1]
                logits, h_new, c_new, context_new, _ = self.decoder.forward_step(
                    input_idx, h, c, context, encoder_outputs)
                log_probs = [_log(p) for p in _softmax(logits)]

                # Get top-k candidates
                top_k = sorted(range(len(log_probs)), key=lambda i: log_probs[i], reverse=True)[:beam_width]

                for idx in top_k:
                    new_score = score + log_probs[idx]
                    new_tokens = tokens + [idx]
                    candidates.append((new_score, new_tokens, h_new, c_new, context_new))

            if not candidates:
                break

            # Keep top beam_width candidates (by raw score)
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]

        # Add any remaining beams
        for score, tokens, h, c, context in beams:
            completed.append((score, tokens))

        if not completed:
            return [sos_idx], 0.0

        # Length-normalized scoring
        def normalized_score(item):
            score, tokens = item
            length = len(tokens)
            return score / (length ** length_penalty) if length > 0 else score

        best = max(completed, key=normalized_score)
        return best[1], normalized_score(best)


# ============================================================
# Loss Functions
# ============================================================

class Seq2SeqLoss:
    """Cross-entropy loss over sequence of logits, ignoring padding."""

    def __init__(self, pad_idx=0):
        self.pad_idx = pad_idx

    def forward(self, all_logits, target_indices):
        """Compute loss and gradients.

        Args:
            all_logits: list of T logit vectors [vocab_size]
            target_indices: list of T target token indices

        Returns:
            loss: average cross-entropy loss (excluding padding)
            d_logits: list of T gradient vectors
        """
        total_loss = 0.0
        count = 0
        d_logits = []

        for t in range(len(all_logits)):
            target = target_indices[t]
            if target == self.pad_idx:
                d_logits.append(_zeros(len(all_logits[t])))
                continue

            logits = all_logits[t]
            probs = _softmax(logits)

            # Cross-entropy loss
            loss_t = -_log(probs[target])
            total_loss += loss_t
            count += 1

            # Gradient: probs - one_hot(target)
            d = probs[:]
            d[target] -= 1.0
            d_logits.append(d)

        avg_loss = total_loss / max(count, 1)
        # Scale gradients
        if count > 0:
            scale = 1.0 / count
            d_logits = [_vec_scale(d, scale) for d in d_logits]

        return avg_loss, d_logits


# ============================================================
# Training
# ============================================================

def train_seq2seq(model, train_pairs, src_vocab, tgt_vocab,
                  epochs=10, lr=0.01, clip_norm=5.0,
                  teacher_forcing_ratio=1.0, tf_decay=0.0,
                  print_every=None):
    """Train a Seq2Seq model.

    Args:
        model: Seq2Seq model instance
        train_pairs: list of (src_tokens, tgt_tokens) pairs
        src_vocab: source Vocabulary
        tgt_vocab: target Vocabulary
        epochs: number of training epochs
        lr: learning rate
        clip_norm: gradient clipping norm
        teacher_forcing_ratio: initial teacher forcing ratio
        tf_decay: decay for teacher forcing per epoch
        print_every: print loss every N samples (None = silent)

    Returns:
        losses: list of average loss per epoch
    """
    loss_fn = Seq2SeqLoss(pad_idx=tgt_vocab.pad_idx)
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        tf_ratio = max(0.0, teacher_forcing_ratio - epoch * tf_decay)

        # Shuffle training data
        pairs = list(train_pairs)
        random.shuffle(pairs)

        for pair_idx, (src_tokens, tgt_tokens) in enumerate(pairs):
            # Encode
            src_ids = src_vocab.encode(src_tokens)
            tgt_ids = tgt_vocab.encode(tgt_tokens, add_sos=True, add_eos=True)

            # Forward
            all_logits, cache = model.forward(src_ids, tgt_ids, teacher_forcing_ratio=tf_ratio)

            # Loss (target is tgt_ids[1:] -- shifted by 1)
            loss, d_logits = loss_fn.forward(all_logits, tgt_ids[1:])
            total_loss += loss

            # Backward
            grads = model.backward(d_logits, cache)

            # Simple SGD update with gradient clipping
            _apply_gradients(model, grads, lr, clip_norm)

            if print_every and (pair_idx + 1) % print_every == 0:
                avg = total_loss / (pair_idx + 1)
                print(f"  Epoch {epoch+1}, sample {pair_idx+1}: loss={avg:.4f}")

        avg_loss = total_loss / max(len(pairs), 1)
        epoch_losses.append(avg_loss)

    return epoch_losses


def _apply_gradients(model, grads, lr, clip_norm):
    """Apply gradients with SGD and clipping."""
    # Decoder gradients
    for step_grads in grads['decoder']:
        # Output projection
        dW_out = step_grads['dW_out']
        db_out = step_grads['db_out']
        dW_out = _clip_mat(dW_out, clip_norm)
        db_out = _clip_vec(db_out, clip_norm)

        for i in range(len(model.decoder.W_out)):
            for j in range(len(model.decoder.W_out[0])):
                model.decoder.W_out[i][j] -= lr * dW_out[i][j]
        for i in range(len(model.decoder.b_out)):
            model.decoder.b_out[i] -= lr * db_out[i]

        # Cell gradients
        _apply_cell_grads(model.decoder.cell, step_grads['cell_grads'], lr, clip_norm)

        # Attention gradients
        _apply_attention_grads(model.decoder.attention, step_grads['attn_grads'], lr, clip_norm)

        # Embedding gradient
        idx, d_embed = step_grads['d_embed']
        d_embed = _clip_vec(d_embed, clip_norm)
        for j in range(len(d_embed)):
            model.decoder.embedding.weights[idx][j] -= lr * d_embed[j]

    # Encoder embedding gradients
    if 'd_embed' in grads['encoder']:
        for idx, d_vec in grads['encoder']['d_embed'].items():
            d_vec = _clip_vec(d_vec, clip_norm)
            for j in range(len(d_vec)):
                model.encoder.embedding.weights[idx][j] -= lr * d_vec[j]


def _apply_cell_grads(cell, grads, lr, clip_norm):
    """Apply gradients to an RNN/LSTM/GRU cell."""
    if isinstance(cell, LSTMCell):
        for gate in ['f', 'i', 'g', 'o']:
            dW_i = grads.get(f'dW_i{gate}')
            dW_h = grads.get(f'dW_h{gate}')
            db = grads.get(f'db_{gate}')
            if dW_i is not None:
                dW_i = _clip_mat(dW_i, clip_norm)
                W_i = getattr(cell, f'W_i{gate}')
                for r in range(len(W_i)):
                    for c in range(len(W_i[0])):
                        W_i[r][c] -= lr * dW_i[r][c]
            if dW_h is not None:
                dW_h = _clip_mat(dW_h, clip_norm)
                W_h = getattr(cell, f'W_h{gate}')
                for r in range(len(W_h)):
                    for c in range(len(W_h[0])):
                        W_h[r][c] -= lr * dW_h[r][c]
            if db is not None:
                db = _clip_vec(db, clip_norm)
                b = getattr(cell, f'b_{gate}')
                for i in range(len(b)):
                    b[i] -= lr * db[i]
    elif isinstance(cell, GRUCell):
        for key in ['dW_ir', 'dW_hr', 'db_r', 'dW_iz', 'dW_hz', 'db_z', 'dW_in', 'dW_hn', 'db_n']:
            d = grads.get(key)
            if d is None:
                continue
            param_key = key[1:]  # Remove 'd' prefix
            param = getattr(cell, param_key, None)
            if param is None:
                continue
            if isinstance(d, list) and isinstance(d[0], list):
                d = _clip_mat(d, clip_norm)
                for r in range(len(param)):
                    for c in range(len(param[0])):
                        param[r][c] -= lr * d[r][c]
            else:
                d = _clip_vec(d, clip_norm)
                for i in range(len(param)):
                    param[i] -= lr * d[i]
    else:  # RNNCell
        for key in ['dW_ih', 'dW_hh', 'db_h']:
            d = grads.get(key)
            if d is None:
                continue
            param_key = key[1:]
            param = getattr(cell, param_key, None)
            if param is None:
                continue
            if isinstance(d, list) and isinstance(d[0], list):
                d = _clip_mat(d, clip_norm)
                for r in range(len(param)):
                    for c in range(len(param[0])):
                        param[r][c] -= lr * d[r][c]
            else:
                d = _clip_vec(d, clip_norm)
                for i in range(len(param)):
                    param[i] -= lr * d[i]


def _apply_attention_grads(attention, grads, lr, clip_norm):
    """Apply gradients to attention mechanism."""
    if isinstance(attention, BahdanauAttention):
        if 'dW_a' in grads:
            dW = _clip_mat(grads['dW_a'], clip_norm)
            for r in range(len(attention.W_a)):
                for c in range(len(attention.W_a[0])):
                    attention.W_a[r][c] -= lr * dW[r][c]
        if 'dU_a' in grads:
            dU = _clip_mat(grads['dU_a'], clip_norm)
            for r in range(len(attention.U_a)):
                for c in range(len(attention.U_a[0])):
                    attention.U_a[r][c] -= lr * dU[r][c]
        if 'dv' in grads:
            dv = _clip_vec(grads['dv'], clip_norm)
            for i in range(len(attention.v)):
                attention.v[i] -= lr * dv[i]
    elif isinstance(attention, LuongAttention):
        if 'dW' in grads:
            dW = _clip_mat(grads['dW'], clip_norm)
            for r in range(len(attention.W)):
                for c in range(len(attention.W[0])):
                    attention.W[r][c] -= lr * dW[r][c]
        if 'dv' in grads:
            dv = _clip_vec(grads['dv'], clip_norm)
            for i in range(len(attention.v)):
                attention.v[i] -= lr * dv[i]


# ============================================================
# Evaluation Metrics
# ============================================================

def bleu_score(reference, hypothesis, max_n=4):
    """Compute BLEU score (simplified, single reference).

    Args:
        reference: list of tokens
        hypothesis: list of tokens
        max_n: maximum n-gram order

    Returns:
        BLEU score (0.0 to 1.0)
    """
    if len(hypothesis) == 0:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1.0 - len(reference) / len(hypothesis))) if len(hypothesis) > 0 else 0.0

    # N-gram precisions with add-1 smoothing for n > 1
    log_prec_sum = 0.0
    n_count = 0

    for n in range(1, max_n + 1):
        if len(hypothesis) < n or len(reference) < n:
            break

        # Count n-grams in hypothesis
        hyp_ngrams = {}
        for i in range(len(hypothesis) - n + 1):
            ng = tuple(hypothesis[i:i + n])
            hyp_ngrams[ng] = hyp_ngrams.get(ng, 0) + 1

        # Count n-grams in reference
        ref_ngrams = {}
        for i in range(len(reference) - n + 1):
            ng = tuple(reference[i:i + n])
            ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1

        # Clipped counts
        clipped = 0
        total = 0
        for ng, count in hyp_ngrams.items():
            clipped += min(count, ref_ngrams.get(ng, 0))
            total += count

        if total == 0:
            break

        # Add-1 smoothing for n > 1 to avoid zero precision killing the score
        if n > 1:
            precision = (clipped + 1) / (total + 1)
        else:
            precision = clipped / total
            if precision == 0:
                return 0.0

        log_prec_sum += _log(precision)
        n_count += 1

    if n_count == 0:
        return 0.0

    return bp * math.exp(log_prec_sum / n_count)


# ============================================================
# Data Generators
# ============================================================

def generate_copy_data(num_samples, min_len=2, max_len=5, vocab_size=10, rng=None):
    """Generate copy task data: input -> same output."""
    r = rng if rng else random
    pairs = []
    for _ in range(num_samples):
        length = r.randint(min_len, max_len)
        seq = [str(r.randint(0, vocab_size - 1)) for _ in range(length)]
        pairs.append((seq, seq[:]))
    return pairs


def generate_reverse_data(num_samples, min_len=2, max_len=5, vocab_size=10, rng=None):
    """Generate reverse task data: input -> reversed output."""
    r = rng if rng else random
    pairs = []
    for _ in range(num_samples):
        length = r.randint(min_len, max_len)
        seq = [str(r.randint(0, vocab_size - 1)) for _ in range(length)]
        pairs.append((seq, seq[::-1]))
    return pairs


def generate_sort_data(num_samples, min_len=2, max_len=5, vocab_size=10, rng=None):
    """Generate sort task data: input -> sorted output."""
    r = rng if rng else random
    pairs = []
    for _ in range(num_samples):
        length = r.randint(min_len, max_len)
        seq = [str(r.randint(0, vocab_size - 1)) for _ in range(length)]
        pairs.append((seq, sorted(seq)))
    return pairs


def generate_addition_data(num_samples, max_val=50, rng=None):
    """Generate simple addition data: 'a + b' -> 'c'."""
    r = rng if rng else random
    pairs = []
    for _ in range(num_samples):
        a = r.randint(0, max_val)
        b = r.randint(0, max_val)
        src = list(str(a) + '+' + str(b))
        tgt = list(str(a + b))
        pairs.append((src, tgt))
    return pairs
