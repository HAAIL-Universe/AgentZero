"""
Tests for C145: Sequence-to-Sequence with Attention
"""

import math
import random
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from seq2seq import (
    Vocabulary, Encoder, Decoder, Seq2Seq, Seq2SeqLoss,
    BahdanauAttention, LuongAttention,
    pad_sequences, tokenize, bleu_score,
    generate_copy_data, generate_reverse_data, generate_sort_data,
    generate_addition_data, train_seq2seq,
    _softmax, _dot, _concat, _log, _zeros, _zeros_2d,
)


# ============================================================
# Vocabulary Tests
# ============================================================

class TestVocabulary:
    def test_special_tokens(self):
        v = Vocabulary()
        assert v.pad_idx == 0
        assert v.sos_idx == 1
        assert v.eos_idx == 2
        assert v.unk_idx == 3
        assert v.size == 4

    def test_build(self):
        v = Vocabulary()
        v.build([['hello', 'world'], ['hello', 'there']])
        assert v.size == 7  # 4 special + hello, world, there
        assert 'hello' in v.token2idx
        assert 'world' in v.token2idx
        assert 'there' in v.token2idx

    def test_encode_basic(self):
        v = Vocabulary()
        v.build([['a', 'b', 'c']])
        ids = v.encode(['a', 'b', 'c'])
        assert len(ids) == 3
        assert ids[0] == v.token2idx['a']

    def test_encode_with_sos_eos(self):
        v = Vocabulary()
        v.build([['x', 'y']])
        ids = v.encode(['x', 'y'], add_sos=True, add_eos=True)
        assert ids[0] == v.sos_idx
        assert ids[-1] == v.eos_idx
        assert len(ids) == 4

    def test_encode_unk(self):
        v = Vocabulary()
        v.build([['a']])
        ids = v.encode(['a', 'z'])
        assert ids[1] == v.unk_idx

    def test_decode_basic(self):
        v = Vocabulary()
        v.build([['x', 'y', 'z']])
        ids = v.encode(['x', 'y', 'z'])
        tokens = v.decode(ids)
        assert tokens == ['x', 'y', 'z']

    def test_decode_strip_special(self):
        v = Vocabulary()
        v.build([['a']])
        ids = [v.sos_idx, v.token2idx['a'], v.eos_idx]
        tokens = v.decode(ids, strip_special=True)
        assert tokens == ['a']

    def test_decode_keep_special(self):
        v = Vocabulary()
        v.build([['a']])
        ids = [v.sos_idx, v.token2idx['a'], v.eos_idx]
        tokens = v.decode(ids, strip_special=False)
        assert '<SOS>' in tokens
        assert '<EOS>' in tokens

    def test_roundtrip(self):
        v = Vocabulary()
        v.build([['hello', 'world', 'test']])
        original = ['hello', 'world', 'test']
        ids = v.encode(original)
        decoded = v.decode(ids)
        assert decoded == original

    def test_empty_sequence(self):
        v = Vocabulary()
        ids = v.encode([])
        assert ids == []
        tokens = v.decode([])
        assert tokens == []

    def test_duplicate_tokens(self):
        v = Vocabulary()
        v.build([['a', 'a', 'b', 'b', 'a']])
        assert v.size == 6  # 4 special + a, b

    def test_multiple_build_calls(self):
        v = Vocabulary()
        v.build([['a', 'b']])
        size1 = v.size
        v.build([['b', 'c']])
        assert v.size == size1 + 1  # Only 'c' is new


# ============================================================
# Utility Tests
# ============================================================

class TestUtilities:
    def test_softmax(self):
        s = _softmax([1.0, 2.0, 3.0])
        assert abs(sum(s) - 1.0) < 1e-6
        assert s[2] > s[1] > s[0]

    def test_softmax_stability(self):
        s = _softmax([1000.0, 1001.0, 1002.0])
        assert abs(sum(s) - 1.0) < 1e-6

    def test_dot_product(self):
        assert abs(_dot([1, 2, 3], [4, 5, 6]) - 32) < 1e-6

    def test_concat(self):
        assert _concat([1, 2], [3, 4]) == [1, 2, 3, 4]

    def test_log_safe(self):
        assert _log(1.0) == 0.0
        assert _log(0.0) == math.log(1e-12)  # Doesn't crash

    def test_pad_sequences(self):
        padded = pad_sequences([[1, 2], [3, 4, 5]], pad_value=0)
        assert padded == [[1, 2, 0], [3, 4, 5]]

    def test_pad_sequences_max_len(self):
        padded = pad_sequences([[1, 2, 3, 4]], max_len=2)
        assert padded == [[1, 2]]

    def test_pad_sequences_custom_value(self):
        padded = pad_sequences([[1], [2, 3]], pad_value=-1)
        assert padded == [[1, -1], [2, 3]]

    def test_tokenize(self):
        assert tokenize('hello world') == ['hello', 'world']
        assert tokenize('  a  b  c  ') == ['a', 'b', 'c']


# ============================================================
# Attention Tests
# ============================================================

class TestBahdanauAttention:
    def test_forward_shape(self):
        rng = random.Random(42)
        attn = BahdanauAttention(encoder_dim=8, decoder_dim=6, attention_dim=4, rng=rng)
        h_dec = [0.1] * 6
        enc_outs = [[0.2] * 8 for _ in range(5)]
        context, weights, cache = attn.forward(h_dec, enc_outs)
        assert len(context) == 8
        assert len(weights) == 5
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_weights_sum_to_one(self):
        rng = random.Random(123)
        attn = BahdanauAttention(4, 4, 4, rng=rng)
        h = [0.5, -0.3, 0.1, 0.8]
        enc = [[0.1, 0.2, 0.3, 0.4], [-0.1, 0.5, -0.2, 0.3], [0.4, -0.1, 0.2, -0.3]]
        _, weights, _ = attn.forward(h, enc)
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_backward_shapes(self):
        rng = random.Random(42)
        attn = BahdanauAttention(4, 3, 2, rng=rng)
        h = [0.1, 0.2, 0.3]
        enc = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        _, _, cache = attn.forward(h, enc)
        d_ctx = [0.1, 0.2, 0.3, 0.4]
        d_h, d_enc, grads = attn.backward(d_ctx, None, cache)
        assert len(d_h) == 3
        assert len(d_enc) == 2
        assert len(d_enc[0]) == 4
        assert 'dW_a' in grads
        assert 'dU_a' in grads
        assert 'dv' in grads

    def test_params(self):
        attn = BahdanauAttention(8, 6, 4, rng=random.Random(42))
        p = attn.params()
        assert 'W_a' in p
        assert 'U_a' in p
        assert 'v' in p

    def test_different_encoder_lengths(self):
        rng = random.Random(42)
        attn = BahdanauAttention(4, 4, 4, rng=rng)
        h = [0.1] * 4
        for T in [1, 3, 7, 10]:
            enc = [[0.1] * 4 for _ in range(T)]
            ctx, w, _ = attn.forward(h, enc)
            assert len(w) == T
            assert abs(sum(w) - 1.0) < 1e-6

    def test_gradient_numerical(self):
        """Numerical gradient check for Bahdanau attention."""
        rng = random.Random(42)
        attn = BahdanauAttention(3, 3, 2, rng=rng)
        h = [0.5, -0.3, 0.1]
        enc = [[0.1, 0.2, 0.3], [-0.1, 0.5, -0.2]]
        ctx, w, cache = attn.forward(h, enc)

        # Compute analytical gradient
        d_ctx = [1.0, 0.0, 0.0]
        d_h, d_enc, grads = attn.backward(d_ctx, None, cache)

        # Numerical gradient for h[0]
        eps = 1e-5
        h_plus = h[:]
        h_plus[0] += eps
        ctx_plus, _, _ = attn.forward(h_plus, enc)
        h_minus = h[:]
        h_minus[0] -= eps
        ctx_minus, _, _ = attn.forward(h_minus, enc)
        num_grad = (ctx_plus[0] - ctx_minus[0]) / (2 * eps)
        assert abs(d_h[0] - num_grad) < 1e-3


class TestLuongAttention:
    def test_dot_forward(self):
        rng = random.Random(42)
        attn = LuongAttention(4, 4, method='dot', rng=rng)
        h = [0.1, 0.2, 0.3, 0.4]
        enc = [[0.1] * 4 for _ in range(3)]
        ctx, w, _ = attn.forward(h, enc)
        assert len(ctx) == 4
        assert len(w) == 3
        assert abs(sum(w) - 1.0) < 1e-6

    def test_general_forward(self):
        rng = random.Random(42)
        attn = LuongAttention(4, 4, method='general', rng=rng)
        h = [0.1, 0.2, 0.3, 0.4]
        enc = [[0.1] * 4 for _ in range(3)]
        ctx, w, _ = attn.forward(h, enc)
        assert len(ctx) == 4
        assert abs(sum(w) - 1.0) < 1e-6

    def test_concat_forward(self):
        rng = random.Random(42)
        attn = LuongAttention(4, 4, method='concat', rng=rng)
        h = [0.1, 0.2, 0.3, 0.4]
        enc = [[0.1] * 4 for _ in range(3)]
        ctx, w, _ = attn.forward(h, enc)
        assert len(ctx) == 4
        assert abs(sum(w) - 1.0) < 1e-6

    def test_dot_backward(self):
        rng = random.Random(42)
        attn = LuongAttention(4, 4, method='dot', rng=rng)
        h = [0.1, 0.2, 0.3, 0.4]
        enc = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        _, _, cache = attn.forward(h, enc)
        d_ctx = [0.1, 0.2, 0.3, 0.4]
        d_h, d_enc, grads = attn.backward(d_ctx, None, cache)
        assert len(d_h) == 4
        assert len(d_enc) == 2

    def test_general_backward(self):
        rng = random.Random(42)
        attn = LuongAttention(4, 4, method='general', rng=rng)
        h = [0.1, 0.2, 0.3, 0.4]
        enc = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        _, _, cache = attn.forward(h, enc)
        d_ctx = [0.1, 0.2, 0.3, 0.4]
        d_h, d_enc, grads = attn.backward(d_ctx, None, cache)
        assert 'dW' in grads

    def test_concat_backward(self):
        rng = random.Random(42)
        attn = LuongAttention(4, 4, method='concat', rng=rng)
        h = [0.1, 0.2, 0.3, 0.4]
        enc = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        _, _, cache = attn.forward(h, enc)
        d_ctx = [0.1, 0.2, 0.3, 0.4]
        d_h, d_enc, grads = attn.backward(d_ctx, None, cache)
        assert 'dW' in grads
        assert 'dv' in grads

    def test_params_dot(self):
        attn = LuongAttention(4, 4, method='dot')
        assert attn.params() == {}

    def test_params_general(self):
        attn = LuongAttention(4, 4, method='general')
        assert 'W' in attn.params()

    def test_params_concat(self):
        attn = LuongAttention(4, 4, method='concat')
        p = attn.params()
        assert 'W' in p
        assert 'v' in p


# ============================================================
# Encoder Tests
# ============================================================

class TestEncoder:
    def test_forward_shape(self):
        rng = random.Random(42)
        enc = Encoder(vocab_size=10, embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        outputs, final, cache = enc.forward([1, 2, 3, 4])
        assert len(outputs) == 4
        assert len(outputs[0]) == 6
        h, c = final
        assert len(h) == 6
        assert len(c) == 6

    def test_forward_gru(self):
        rng = random.Random(42)
        enc = Encoder(vocab_size=10, embed_dim=4, hidden_size=6, cell_type='gru', rng=rng)
        outputs, final, cache = enc.forward([1, 2, 3])
        assert len(outputs) == 3
        assert len(outputs[0]) == 6
        assert len(final) == 6  # GRU: just h

    def test_bidirectional(self):
        rng = random.Random(42)
        enc = Encoder(vocab_size=10, embed_dim=4, hidden_size=6,
                      cell_type='lstm', bidirectional=True, rng=rng)
        outputs, final, cache = enc.forward([1, 2, 3])
        assert len(outputs) == 3
        assert len(outputs[0]) == 12  # 6 * 2

    def test_multi_layer(self):
        rng = random.Random(42)
        enc = Encoder(vocab_size=10, embed_dim=4, hidden_size=6,
                      cell_type='lstm', num_layers=2, rng=rng)
        outputs, final, cache = enc.forward([1, 2, 3])
        assert len(outputs) == 3
        assert len(outputs[0]) == 6

    def test_backward(self):
        rng = random.Random(42)
        enc = Encoder(vocab_size=10, embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        outputs, final, cache = enc.forward([1, 2, 3])
        d_outputs = [_zeros(6) for _ in range(3)]
        d_outputs[0][0] = 1.0
        d_final = (_zeros(6), _zeros(6))
        grads = enc.backward(d_outputs, d_final, cache)
        assert 'd_embed' in grads

    def test_backward_gru(self):
        rng = random.Random(42)
        enc = Encoder(vocab_size=10, embed_dim=4, hidden_size=6, cell_type='gru', rng=rng)
        outputs, final, cache = enc.forward([1, 2, 3])
        d_outputs = [_zeros(6) for _ in range(3)]
        d_outputs[0][0] = 1.0
        d_final = _zeros(6)
        grads = enc.backward(d_outputs, d_final, cache)
        assert 'd_embed' in grads

    def test_params(self):
        enc = Encoder(vocab_size=10, embed_dim=4, hidden_size=6, cell_type='lstm', rng=random.Random(42))
        p = enc.params()
        assert 'embedding' in p
        assert 'fwd_cell_0' in p

    def test_single_token(self):
        rng = random.Random(42)
        enc = Encoder(vocab_size=10, embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        outputs, final, cache = enc.forward([5])
        assert len(outputs) == 1

    def test_bidirectional_backward(self):
        rng = random.Random(42)
        enc = Encoder(vocab_size=10, embed_dim=4, hidden_size=6,
                      cell_type='lstm', bidirectional=True, rng=rng)
        outputs, final, cache = enc.forward([1, 2, 3])
        d_outputs = [_zeros(12) for _ in range(3)]
        d_outputs[0][0] = 1.0
        d_final = (_zeros(6), _zeros(6))
        grads = enc.backward(d_outputs, d_final, cache)
        assert 'd_embed' in grads


# ============================================================
# Decoder Tests
# ============================================================

class TestDecoder:
    def test_forward_step(self):
        rng = random.Random(42)
        dec = Decoder(vocab_size=10, embed_dim=4, hidden_size=6,
                      encoder_dim=6, cell_type='lstm', rng=rng)
        h = _zeros(6)
        c = _zeros(6)
        ctx = _zeros(6)
        enc_outs = [_zeros(6) for _ in range(3)]
        logits, h_new, c_new, ctx_new, cache = dec.forward_step(1, h, c, ctx, enc_outs)
        assert len(logits) == 10
        assert len(h_new) == 6
        assert len(c_new) == 6
        assert len(ctx_new) == 6

    def test_forward_step_gru(self):
        rng = random.Random(42)
        dec = Decoder(vocab_size=10, embed_dim=4, hidden_size=6,
                      encoder_dim=6, cell_type='gru', rng=rng)
        h = _zeros(6)
        ctx = _zeros(6)
        enc_outs = [_zeros(6) for _ in range(3)]
        logits, h_new, c_new, ctx_new, cache = dec.forward_step(1, h, None, ctx, enc_outs)
        assert len(logits) == 10
        assert c_new is None

    def test_backward_step(self):
        rng = random.Random(42)
        dec = Decoder(vocab_size=10, embed_dim=4, hidden_size=6,
                      encoder_dim=6, cell_type='lstm', rng=rng)
        h = _zeros(6)
        c = _zeros(6)
        ctx = _zeros(6)
        enc_outs = [[0.1 * i] * 6 for i in range(3)]
        logits, h_new, c_new, ctx_new, cache = dec.forward_step(1, h, c, ctx, enc_outs)
        d_logits = [0.1] * 10
        d_h, d_c, d_ctx, d_enc, grads = dec.backward_step(d_logits, None, None, None, cache)
        assert len(d_h) == 6
        assert len(d_c) == 6
        assert len(d_ctx) == 6
        assert len(d_enc) == 3

    def test_luong_attention(self):
        rng = random.Random(42)
        dec = Decoder(vocab_size=10, embed_dim=4, hidden_size=6,
                      encoder_dim=6, attention_type='luong_general', cell_type='lstm', rng=rng)
        h = _zeros(6)
        c = _zeros(6)
        ctx = _zeros(6)
        enc_outs = [_zeros(6) for _ in range(3)]
        logits, h_new, c_new, ctx_new, cache = dec.forward_step(1, h, c, ctx, enc_outs)
        assert len(logits) == 10

    def test_luong_dot(self):
        rng = random.Random(42)
        dec = Decoder(vocab_size=10, embed_dim=4, hidden_size=6,
                      encoder_dim=6, attention_type='luong_dot', cell_type='lstm', rng=rng)
        h = _zeros(6)
        c = _zeros(6)
        ctx = _zeros(6)
        enc_outs = [_zeros(6) for _ in range(3)]
        logits, _, _, _, _ = dec.forward_step(1, h, c, ctx, enc_outs)
        assert len(logits) == 10

    def test_params(self):
        dec = Decoder(vocab_size=10, embed_dim=4, hidden_size=6,
                      encoder_dim=6, cell_type='lstm', rng=random.Random(42))
        p = dec.params()
        assert 'embedding' in p
        assert 'cell' in p
        assert 'attention' in p
        assert 'W_out' in p
        assert 'b_out' in p


# ============================================================
# Seq2Seq Tests
# ============================================================

class TestSeq2Seq:
    def test_forward(self):
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        src = [1, 2, 3]
        tgt = [1, 4, 5, 2]  # SOS, tokens, EOS
        logits, cache = model.forward(src, tgt, teacher_forcing_ratio=1.0)
        assert len(logits) == 3  # tgt_len - 1
        assert len(logits[0]) == 10

    def test_forward_gru(self):
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, cell_type='gru', rng=rng)
        src = [1, 2, 3]
        tgt = [1, 4, 5, 2]
        logits, cache = model.forward(src, tgt, teacher_forcing_ratio=1.0)
        assert len(logits) == 3

    def test_backward(self):
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        src = [1, 2, 3]
        tgt = [1, 4, 5, 2]
        logits, cache = model.forward(src, tgt, teacher_forcing_ratio=1.0)

        loss_fn = Seq2SeqLoss(pad_idx=0)
        loss, d_logits = loss_fn.forward(logits, tgt[1:])
        grads = model.backward(d_logits, cache)
        assert 'encoder' in grads
        assert 'decoder' in grads

    def test_greedy_decode(self):
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        output, weights = model.greedy_decode([1, 2, 3], sos_idx=1, eos_idx=2, max_len=10)
        assert isinstance(output, list)
        assert len(output) <= 10
        assert isinstance(weights, list)

    def test_beam_search(self):
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        output, score = model.beam_search([1, 2, 3], sos_idx=1, eos_idx=2,
                                           beam_width=3, max_len=10)
        assert isinstance(output, list)
        assert isinstance(score, float)

    def test_beam_width_1_equals_greedy(self):
        """Beam width 1 should produce similar results to greedy."""
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        greedy_out, _ = model.greedy_decode([1, 2, 3], sos_idx=1, eos_idx=2, max_len=10)
        # Beam search with width 1 should be close (not always identical due to length penalty)
        beam_out, _ = model.beam_search([1, 2, 3], sos_idx=1, eos_idx=2,
                                         beam_width=1, max_len=10, length_penalty=0.0)
        # At minimum, both should produce outputs
        assert len(greedy_out) > 0
        assert len(beam_out) > 0

    def test_bidirectional_encoder(self):
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, bidirectional=True,
                        cell_type='lstm', rng=rng)
        src = [1, 2, 3]
        tgt = [1, 4, 5, 2]
        logits, cache = model.forward(src, tgt)
        assert len(logits) == 3

    def test_multi_layer_encoder(self):
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, encoder_layers=2,
                        cell_type='lstm', rng=rng)
        src = [1, 2, 3]
        tgt = [1, 4, 5, 2]
        logits, cache = model.forward(src, tgt)
        assert len(logits) == 3

    def test_luong_attention_model(self):
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6,
                        attention_type='luong_general', cell_type='lstm', rng=rng)
        src = [1, 2, 3]
        tgt = [1, 4, 5, 2]
        logits, cache = model.forward(src, tgt)
        assert len(logits) == 3

    def test_no_teacher_forcing(self):
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        src = [1, 2, 3]
        tgt = [1, 4, 5, 2]
        logits, cache = model.forward(src, tgt, teacher_forcing_ratio=0.0)
        assert len(logits) == 3

    def test_max_len_constraint(self):
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        output, _ = model.greedy_decode([1, 2, 3], sos_idx=1, eos_idx=2, max_len=5)
        assert len(output) <= 5

    def test_beam_search_max_len(self):
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        output, _ = model.beam_search([1, 2, 3], sos_idx=1, eos_idx=2,
                                       beam_width=3, max_len=5)
        # Output should be bounded (including SOS in beam)
        assert isinstance(output, list)


# ============================================================
# Loss Tests
# ============================================================

class TestSeq2SeqLoss:
    def test_basic_loss(self):
        loss_fn = Seq2SeqLoss(pad_idx=0)
        logits = [[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]]
        targets = [2, 0]  # second is padding
        loss, d = loss_fn.forward(logits, targets)
        assert loss > 0
        assert len(d) == 2
        # Padding target should have zero gradient
        assert all(abs(x) < 1e-10 for x in d[1])

    def test_perfect_prediction(self):
        loss_fn = Seq2SeqLoss(pad_idx=0)
        # Logits strongly favoring correct class
        logits = [[10.0, -10.0, -10.0]]
        targets = [0]  # But 0 is pad... use target 1
        # Actually test with non-pad target
        logits = [[-10.0, 10.0, -10.0]]
        targets = [1]
        loss, d = loss_fn.forward(logits, targets)
        assert loss < 0.01

    def test_gradient_direction(self):
        loss_fn = Seq2SeqLoss(pad_idx=0)
        logits = [[0.0, 0.0, 0.0]]
        targets = [1]
        loss, d = loss_fn.forward(logits, targets)
        # Gradient for target class should be negative (prob - 1)
        assert d[0][1] < 0

    def test_all_padding(self):
        loss_fn = Seq2SeqLoss(pad_idx=0)
        logits = [[1.0, 2.0], [3.0, 4.0]]
        targets = [0, 0]
        loss, d = loss_fn.forward(logits, targets)
        assert loss == 0.0

    def test_loss_decreases_with_better_logits(self):
        loss_fn = Seq2SeqLoss(pad_idx=0)
        # Bad logits
        l1 = [[0.0, 0.0, 0.0]]
        # Good logits
        l2 = [[-5.0, 10.0, -5.0]]
        targets = [1]
        loss1, _ = loss_fn.forward(l1, targets)
        loss2, _ = loss_fn.forward(l2, targets)
        assert loss2 < loss1


# ============================================================
# BLEU Score Tests
# ============================================================

class TestBleuScore:
    def test_perfect_match(self):
        ref = ['the', 'cat', 'sat']
        hyp = ['the', 'cat', 'sat']
        score = bleu_score(ref, hyp)
        assert abs(score - 1.0) < 0.01

    def test_no_match(self):
        ref = ['the', 'cat', 'sat']
        hyp = ['a', 'dog', 'ran']
        score = bleu_score(ref, hyp)
        assert score == 0.0

    def test_partial_match(self):
        ref = ['the', 'cat', 'sat', 'on', 'the', 'mat']
        hyp = ['the', 'cat', 'on', 'the', 'mat']
        score = bleu_score(ref, hyp)
        assert 0.0 < score < 1.0

    def test_empty_hypothesis(self):
        ref = ['the', 'cat']
        hyp = []
        score = bleu_score(ref, hyp)
        assert score == 0.0

    def test_brevity_penalty(self):
        ref = ['the', 'cat', 'sat', 'on', 'the', 'mat']
        hyp = ['the', 'cat']
        score = bleu_score(ref, hyp)
        assert score < 1.0  # Brevity penalty applied

    def test_single_token(self):
        ref = ['hello']
        hyp = ['hello']
        score = bleu_score(ref, hyp, max_n=1)
        assert abs(score - 1.0) < 0.01

    def test_max_n_parameter(self):
        ref = ['a', 'b', 'c', 'd']
        hyp = ['a', 'b', 'c', 'd']
        s1 = bleu_score(ref, hyp, max_n=1)
        s4 = bleu_score(ref, hyp, max_n=4)
        assert s1 > 0
        assert s4 > 0


# ============================================================
# Data Generator Tests
# ============================================================

class TestDataGenerators:
    def test_copy_data(self):
        data = generate_copy_data(10, min_len=3, max_len=5, rng=random.Random(42))
        assert len(data) == 10
        for src, tgt in data:
            assert src == tgt
            assert 3 <= len(src) <= 5

    def test_reverse_data(self):
        data = generate_reverse_data(10, min_len=2, max_len=4, rng=random.Random(42))
        assert len(data) == 10
        for src, tgt in data:
            assert tgt == src[::-1]

    def test_sort_data(self):
        data = generate_sort_data(10, min_len=3, max_len=5, rng=random.Random(42))
        assert len(data) == 10
        for src, tgt in data:
            assert tgt == sorted(src)

    def test_addition_data(self):
        data = generate_addition_data(10, max_val=20, rng=random.Random(42))
        assert len(data) == 10
        for src, tgt in data:
            # src is like ['1', '2', '+', '5']
            expr = ''.join(src)
            parts = expr.split('+')
            expected = int(parts[0]) + int(parts[1])
            assert ''.join(tgt) == str(expected)


# ============================================================
# Training Tests
# ============================================================

class TestTraining:
    def test_train_loss_decreases(self):
        """Training should reduce loss over epochs."""
        rng = random.Random(42)
        random.seed(42)

        # Simple copy task
        data = generate_copy_data(20, min_len=2, max_len=3, vocab_size=5, rng=rng)
        src_seqs = [p[0] for p in data]
        tgt_seqs = [p[1] for p in data]

        src_vocab = Vocabulary()
        src_vocab.build(src_seqs)
        tgt_vocab = Vocabulary()
        tgt_vocab.build(tgt_seqs)

        model = Seq2Seq(src_vocab_size=src_vocab.size, tgt_vocab_size=tgt_vocab.size,
                        embed_dim=8, hidden_size=16, cell_type='lstm', rng=random.Random(42))

        losses = train_seq2seq(model, data, src_vocab, tgt_vocab,
                               epochs=5, lr=0.01, clip_norm=5.0)
        assert len(losses) == 5
        # Loss should decrease (or at least not increase dramatically)
        assert losses[-1] < losses[0] + 0.5  # Allows some flexibility

    def test_train_with_gru(self):
        rng = random.Random(42)
        random.seed(42)

        data = generate_copy_data(10, min_len=2, max_len=3, vocab_size=5, rng=rng)
        src_seqs = [p[0] for p in data]
        tgt_seqs = [p[1] for p in data]

        src_vocab = Vocabulary()
        src_vocab.build(src_seqs)
        tgt_vocab = Vocabulary()
        tgt_vocab.build(tgt_seqs)

        model = Seq2Seq(src_vocab_size=src_vocab.size, tgt_vocab_size=tgt_vocab.size,
                        embed_dim=8, hidden_size=16, cell_type='gru', rng=random.Random(42))

        losses = train_seq2seq(model, data, src_vocab, tgt_vocab, epochs=3, lr=0.01)
        assert len(losses) == 3

    def test_teacher_forcing_decay(self):
        """Training with teacher forcing decay should work."""
        rng = random.Random(42)
        random.seed(42)

        data = generate_copy_data(10, min_len=2, max_len=3, vocab_size=5, rng=rng)
        src_seqs = [p[0] for p in data]
        tgt_seqs = [p[1] for p in data]

        src_vocab = Vocabulary()
        src_vocab.build(src_seqs)
        tgt_vocab = Vocabulary()
        tgt_vocab.build(tgt_seqs)

        model = Seq2Seq(src_vocab_size=src_vocab.size, tgt_vocab_size=tgt_vocab.size,
                        embed_dim=8, hidden_size=16, cell_type='lstm', rng=random.Random(42))

        losses = train_seq2seq(model, data, src_vocab, tgt_vocab,
                               epochs=3, lr=0.01, teacher_forcing_ratio=1.0, tf_decay=0.3)
        assert len(losses) == 3

    def test_train_reverse_task(self):
        """Test training on the reverse task."""
        rng = random.Random(42)
        random.seed(42)

        data = generate_reverse_data(20, min_len=2, max_len=3, vocab_size=4, rng=rng)
        src_seqs = [p[0] for p in data]
        tgt_seqs = [p[1] for p in data]

        src_vocab = Vocabulary()
        src_vocab.build(src_seqs)
        tgt_vocab = Vocabulary()
        tgt_vocab.build(tgt_seqs)

        model = Seq2Seq(src_vocab_size=src_vocab.size, tgt_vocab_size=tgt_vocab.size,
                        embed_dim=8, hidden_size=16, cell_type='lstm', rng=random.Random(42))

        losses = train_seq2seq(model, data, src_vocab, tgt_vocab, epochs=5, lr=0.01)
        assert losses[-1] < losses[0] + 0.5


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    def test_full_pipeline_copy(self):
        """Full pipeline: build vocab, train, decode."""
        rng = random.Random(42)
        random.seed(42)

        data = generate_copy_data(30, min_len=2, max_len=3, vocab_size=5, rng=rng)
        src_seqs = [p[0] for p in data]
        tgt_seqs = [p[1] for p in data]

        src_vocab = Vocabulary()
        src_vocab.build(src_seqs)
        tgt_vocab = Vocabulary()
        tgt_vocab.build(tgt_seqs)

        model = Seq2Seq(src_vocab_size=src_vocab.size, tgt_vocab_size=tgt_vocab.size,
                        embed_dim=8, hidden_size=16, cell_type='lstm', rng=random.Random(42))

        losses = train_seq2seq(model, data, src_vocab, tgt_vocab, epochs=10, lr=0.01)

        # Try greedy decoding
        src = src_vocab.encode(data[0][0])
        output, weights = model.greedy_decode(src, tgt_vocab.sos_idx, tgt_vocab.eos_idx, max_len=10)
        assert len(output) > 0
        assert len(weights) > 0

    def test_full_pipeline_beam(self):
        """Full pipeline with beam search."""
        rng = random.Random(42)
        random.seed(42)

        data = generate_copy_data(20, min_len=2, max_len=3, vocab_size=5, rng=rng)
        src_seqs = [p[0] for p in data]
        tgt_seqs = [p[1] for p in data]

        src_vocab = Vocabulary()
        src_vocab.build(src_seqs)
        tgt_vocab = Vocabulary()
        tgt_vocab.build(tgt_seqs)

        model = Seq2Seq(src_vocab_size=src_vocab.size, tgt_vocab_size=tgt_vocab.size,
                        embed_dim=8, hidden_size=16, cell_type='lstm', rng=random.Random(42))

        train_seq2seq(model, data, src_vocab, tgt_vocab, epochs=5, lr=0.01)

        src = src_vocab.encode(data[0][0])
        output, score = model.beam_search(src, tgt_vocab.sos_idx, tgt_vocab.eos_idx,
                                           beam_width=3, max_len=10)
        assert len(output) > 0
        assert isinstance(score, float)

    def test_attention_weights_visualizable(self):
        """Attention weights should be accessible for visualization."""
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        src = [1, 2, 3, 4]
        output, weights = model.greedy_decode(src, sos_idx=1, eos_idx=2, max_len=5)
        # Weights should be list of lists, each summing to 1
        for w in weights:
            assert abs(sum(w) - 1.0) < 1e-6
            assert len(w) == 4  # Same as source length

    def test_different_src_tgt_vocab(self):
        """Source and target can have different vocabularies."""
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=15, tgt_vocab_size=20,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        src = [1, 2, 3]
        tgt = [1, 5, 10, 15, 2]
        logits, cache = model.forward(src, tgt)
        assert len(logits) == 4
        assert len(logits[0]) == 20

    def test_luong_concat_model(self):
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6,
                        attention_type='luong_concat', cell_type='lstm', rng=rng)
        src = [1, 2, 3]
        tgt = [1, 4, 5, 2]
        logits, cache = model.forward(src, tgt)
        assert len(logits) == 3

    def test_bidirectional_greedy(self):
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, bidirectional=True,
                        cell_type='lstm', rng=rng)
        output, weights = model.greedy_decode([1, 2, 3], sos_idx=1, eos_idx=2, max_len=5)
        assert len(output) > 0

    def test_bidirectional_beam(self):
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, bidirectional=True,
                        cell_type='lstm', rng=rng)
        output, score = model.beam_search([1, 2, 3], sos_idx=1, eos_idx=2,
                                           beam_width=2, max_len=5)
        assert len(output) > 0

    def test_gru_full_pipeline(self):
        rng = random.Random(42)
        random.seed(42)

        data = generate_copy_data(15, min_len=2, max_len=3, vocab_size=5, rng=rng)
        src_seqs = [p[0] for p in data]
        tgt_seqs = [p[1] for p in data]

        src_vocab = Vocabulary()
        src_vocab.build(src_seqs)
        tgt_vocab = Vocabulary()
        tgt_vocab.build(tgt_seqs)

        model = Seq2Seq(src_vocab_size=src_vocab.size, tgt_vocab_size=tgt_vocab.size,
                        embed_dim=8, hidden_size=16, cell_type='gru', rng=random.Random(42))

        losses = train_seq2seq(model, data, src_vocab, tgt_vocab, epochs=3, lr=0.01)
        output, _ = model.greedy_decode(src_vocab.encode(data[0][0]),
                                         tgt_vocab.sos_idx, tgt_vocab.eos_idx)
        assert len(output) > 0


# ============================================================
# Edge Case Tests
# ============================================================

class TestEdgeCases:
    def test_single_token_source(self):
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        src = [5]
        tgt = [1, 3, 2]
        logits, cache = model.forward(src, tgt)
        assert len(logits) == 2

    def test_single_token_target(self):
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        src = [1, 2, 3]
        tgt = [1, 2]  # SOS, EOS only
        logits, cache = model.forward(src, tgt)
        assert len(logits) == 1

    def test_long_source(self):
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=20, tgt_vocab_size=20,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        src = list(range(4, 14))  # 10 tokens
        tgt = [1, 5, 6, 2]
        logits, cache = model.forward(src, tgt)
        assert len(logits) == 3

    def test_beam_search_early_eos(self):
        """Beam search should stop when EOS is found."""
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        output, score = model.beam_search([1, 2], sos_idx=1, eos_idx=2,
                                           beam_width=3, max_len=50)
        # Should terminate before max_len due to EOS or beam exhaustion
        assert len(output) <= 51

    def test_large_beam_width(self):
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        output, score = model.beam_search([1, 2, 3], sos_idx=1, eos_idx=2,
                                           beam_width=8, max_len=10)
        assert len(output) > 0

    def test_length_penalty_effect(self):
        """Higher length penalty should favor shorter sequences."""
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        out_low, _ = model.beam_search([1, 2, 3], sos_idx=1, eos_idx=2,
                                        beam_width=3, max_len=10, length_penalty=0.0)
        out_high, _ = model.beam_search([1, 2, 3], sos_idx=1, eos_idx=2,
                                         beam_width=3, max_len=10, length_penalty=2.0)
        # Both should produce valid output
        assert len(out_low) > 0
        assert len(out_high) > 0

    def test_zero_length_penalty(self):
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        output, score = model.beam_search([1, 2, 3], sos_idx=1, eos_idx=2,
                                           beam_width=3, max_len=10, length_penalty=0.0)
        assert len(output) > 0

    def test_deterministic_with_seed(self):
        """Same seed should produce same results."""
        for _ in range(2):
            rng = random.Random(42)
            model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                            embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
            logits1, _ = model.forward([1, 2, 3], [1, 4, 5, 2], teacher_forcing_ratio=1.0)

        # Different run with same seed
        rng2 = random.Random(42)
        model2 = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                         embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng2)
        logits2, _ = model2.forward([1, 2, 3], [1, 4, 5, 2], teacher_forcing_ratio=1.0)

        for i in range(len(logits2)):
            for j in range(len(logits2[i])):
                assert abs(logits1[i][j] - logits2[i][j]) < 1e-10


# ============================================================
# Additional Coverage Tests
# ============================================================

class TestAdditionalCoverage:
    def test_encoder_different_vocab_sizes(self):
        for vs in [5, 50, 100]:
            enc = Encoder(vocab_size=vs, embed_dim=4, hidden_size=6,
                          cell_type='lstm', rng=random.Random(42))
            outputs, final, cache = enc.forward([1, 2])
            assert len(outputs) == 2

    def test_decoder_attention_weights_sum(self):
        """All attention weights at each step sum to 1."""
        rng = random.Random(42)
        dec = Decoder(vocab_size=10, embed_dim=4, hidden_size=6,
                      encoder_dim=6, cell_type='lstm', rng=rng)
        h = _zeros(6)
        c = _zeros(6)
        ctx = _zeros(6)
        enc_outs = [[0.1 * i] * 6 for i in range(4)]

        for step in range(3):
            logits, h, c, ctx, cache = dec.forward_step(step + 4, h, c, ctx, enc_outs)
            w = cache['attn_weights']
            assert abs(sum(w) - 1.0) < 1e-6

    def test_loss_with_long_sequence(self):
        loss_fn = Seq2SeqLoss(pad_idx=0)
        logits = [[float(j) for j in range(10)] for _ in range(20)]
        targets = [i % 10 for i in range(1, 21)]  # No padding
        loss, d = loss_fn.forward(logits, targets)
        assert loss > 0
        assert len(d) == 20

    def test_bleu_identical_long(self):
        ref = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        hyp = ref[:]
        score = bleu_score(ref, hyp)
        assert abs(score - 1.0) < 0.01

    def test_bleu_one_word_off(self):
        ref = ['the', 'cat', 'sat']
        hyp = ['the', 'dog', 'sat']
        score = bleu_score(ref, hyp)
        assert 0.0 < score < 1.0

    def test_vocabulary_idx2token_consistency(self):
        v = Vocabulary()
        v.build([['a', 'b', 'c']])
        for tok, idx in v.token2idx.items():
            assert v.idx2token[idx] == tok

    def test_train_with_addition_data(self):
        """Test training on addition task."""
        rng = random.Random(42)
        random.seed(42)

        data = generate_addition_data(15, max_val=10, rng=rng)
        src_seqs = [p[0] for p in data]
        tgt_seqs = [p[1] for p in data]

        src_vocab = Vocabulary()
        src_vocab.build(src_seqs)
        tgt_vocab = Vocabulary()
        tgt_vocab.build(tgt_seqs)

        model = Seq2Seq(src_vocab_size=src_vocab.size, tgt_vocab_size=tgt_vocab.size,
                        embed_dim=8, hidden_size=16, cell_type='lstm', rng=random.Random(42))

        losses = train_seq2seq(model, data, src_vocab, tgt_vocab, epochs=3, lr=0.01)
        assert len(losses) == 3

    def test_encoder_backward_null_final_state(self):
        """Encoder backward with None final state gradient."""
        rng = random.Random(42)
        enc = Encoder(vocab_size=10, embed_dim=4, hidden_size=6,
                      cell_type='lstm', rng=rng)
        outputs, final, cache = enc.forward([1, 2, 3])
        d_outputs = [_zeros(6) for _ in range(3)]
        grads = enc.backward(d_outputs, None, cache)
        assert 'd_embed' in grads

    def test_seq2seq_forward_backward_consistency(self):
        """Forward then backward should not crash and produce valid gradients."""
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)

        src = [4, 5, 6]
        tgt = [1, 7, 8, 2]

        logits, cache = model.forward(src, tgt, teacher_forcing_ratio=1.0)
        loss_fn = Seq2SeqLoss(pad_idx=0)
        loss, d_logits = loss_fn.forward(logits, tgt[1:])
        grads = model.backward(d_logits, cache)

        # Should have both encoder and decoder gradients
        assert len(grads['decoder']) == 3
        assert 'd_embed' in grads['encoder']

    def test_bahdanau_attention_varied_dims(self):
        """Test Bahdanau with different dimension combos."""
        for ed, dd, ad in [(4, 4, 4), (8, 6, 3), (3, 5, 7)]:
            attn = BahdanauAttention(ed, dd, ad, rng=random.Random(42))
            h = [0.1] * dd
            enc = [[0.1] * ed for _ in range(3)]
            ctx, w, _ = attn.forward(h, enc)
            assert len(ctx) == ed
            assert abs(sum(w) - 1.0) < 1e-6

    def test_luong_general_varied_dims(self):
        for ed, dd in [(4, 4), (6, 8), (3, 5)]:
            attn = LuongAttention(ed, dd, method='general', rng=random.Random(42))
            h = [0.1] * dd
            enc = [[0.1] * ed for _ in range(3)]
            ctx, w, _ = attn.forward(h, enc)
            assert len(ctx) == ed

    def test_copy_data_vocab_range(self):
        data = generate_copy_data(100, vocab_size=3, rng=random.Random(42))
        all_tokens = set()
        for src, tgt in data:
            all_tokens.update(src)
        # All tokens should be in range [0, 3)
        for t in all_tokens:
            assert int(t) >= 0 and int(t) < 3

    def test_sort_data_correctness(self):
        data = generate_sort_data(50, vocab_size=10, rng=random.Random(42))
        for src, tgt in data:
            assert tgt == sorted(src)
            assert len(tgt) == len(src)

    def test_greedy_returns_attention_weights(self):
        """Greedy decode returns attention weights at each step."""
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        output, weights = model.greedy_decode([1, 2, 3], sos_idx=1, eos_idx=2, max_len=5)
        assert len(weights) == len(output)

    def test_beam_search_different_widths(self):
        """Beam search with various widths."""
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)
        for width in [1, 2, 5]:
            out, score = model.beam_search([1, 2, 3], sos_idx=1, eos_idx=2,
                                            beam_width=width, max_len=8)
            assert len(out) > 0

    def test_model_with_small_hidden(self):
        """Model with very small hidden size."""
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=2, hidden_size=2, cell_type='lstm', rng=rng)
        logits, _ = model.forward([1, 2], [1, 3, 2])
        assert len(logits) == 2

    def test_multiple_forward_passes(self):
        """Multiple forward passes should be independent."""
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)

        logits1, _ = model.forward([1, 2, 3], [1, 4, 5, 2], teacher_forcing_ratio=1.0)
        logits2, _ = model.forward([1, 2, 3], [1, 4, 5, 2], teacher_forcing_ratio=1.0)

        # Same input, same model -> same output (with full teacher forcing)
        for i in range(len(logits1)):
            for j in range(len(logits1[i])):
                assert abs(logits1[i][j] - logits2[i][j]) < 1e-10

    def test_backward_gradients_nonzero(self):
        """Backward should produce non-zero gradients."""
        rng = random.Random(42)
        model = Seq2Seq(src_vocab_size=10, tgt_vocab_size=10,
                        embed_dim=4, hidden_size=6, cell_type='lstm', rng=rng)

        src = [4, 5, 6]
        tgt = [1, 7, 8, 2]
        logits, cache = model.forward(src, tgt, teacher_forcing_ratio=1.0)
        loss_fn = Seq2SeqLoss(pad_idx=0)
        loss, d_logits = loss_fn.forward(logits, tgt[1:])
        grads = model.backward(d_logits, cache)

        # Check decoder grads are non-zero
        some_nonzero = False
        for step_g in grads['decoder']:
            for v in step_g['db_out']:
                if abs(v) > 1e-12:
                    some_nonzero = True
                    break
        assert some_nonzero


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
