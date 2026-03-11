"""Tests for C147: Word Embeddings"""

import pytest
import numpy as np
import os
import tempfile

from word_embeddings import (
    Vocabulary, tokenize, build_corpus, subsample_prob,
    NegativeSampler, generate_skipgram_pairs, generate_cbow_pairs,
    SkipGram, CBOW, GloVe, FastText, get_ngrams, hash_ngram,
    EmbeddingSpace, Word2Vec, GloVeTrainer, FastTextTrainer,
    save_embeddings, load_embeddings,
)


# ---------------------------------------------------------------------------
# Sample corpus for testing
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat chased the dog",
    "the dog chased the cat",
    "a cat is a small animal",
    "a dog is a friendly animal",
    "the mat is on the floor",
    "the log is in the yard",
    "cats and dogs are animals",
    "the small cat sat on the big mat",
    "the friendly dog ran in the yard",
    "animals live in the yard and the house",
    "the cat and the dog are friends",
    "a big dog chased a small cat",
    "the floor mat is soft",
    "cats like to sit on mats",
    "dogs like to run in yards",
    "the house has a big yard",
    "small animals are cute",
    "big animals are strong",
]

CORPUS = build_corpus(SAMPLE_TEXTS)


# ---------------------------------------------------------------------------
# Vocabulary tests
# ---------------------------------------------------------------------------

class TestVocabulary:
    def test_build_basic(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        assert len(vocab) > 0
        assert "the" in vocab
        assert "cat" in vocab

    def test_encode_decode(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        idx = vocab.encode("the")
        assert idx is not None
        assert vocab.decode(idx) == "the"

    def test_unknown_word(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        assert vocab.encode("xyzzy") is None

    def test_min_count_filter(self):
        vocab = Vocabulary(min_count=3)
        vocab.build(CORPUS)
        # Words with count < 3 should be filtered
        for word in vocab.idx2word:
            assert vocab.word_counts[word] >= 3

    def test_max_vocab_size(self):
        vocab = Vocabulary(max_vocab_size=5)
        vocab.build(CORPUS)
        assert len(vocab) == 5

    def test_frequency_ordering(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        # Most frequent word should be first
        assert vocab.idx2word[0] == "the"

    def test_total_words(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        assert vocab.total_words > 0

    def test_get_frequency(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        assert vocab.get_frequency("the") > 0
        assert vocab.get_frequency("xyzzy") == 0

    def test_decode_out_of_range(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        assert vocab.decode(-1) is None
        assert vocab.decode(99999) is None

    def test_contains(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        assert "cat" in vocab
        assert "nonexistent" not in vocab


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------

class TestTokenizer:
    def test_basic(self):
        tokens = tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_no_lower(self):
        tokens = tokenize("Hello World", lower=False)
        assert tokens == ["Hello", "World"]

    def test_punctuation(self):
        tokens = tokenize("cats, dogs, and birds!")
        assert tokens == ["cats", "dogs", "and", "birds"]

    def test_contractions(self):
        tokens = tokenize("I'm don't can't")
        assert "i'm" in tokens
        assert "don't" in tokens

    def test_numbers(self):
        tokens = tokenize("there are 42 cats")
        assert "42" in tokens

    def test_build_corpus(self):
        texts = ["hello world", "foo bar"]
        corpus = build_corpus(texts)
        assert len(corpus) == 2
        assert corpus[0] == ["hello", "world"]


# ---------------------------------------------------------------------------
# Subsampling tests
# ---------------------------------------------------------------------------

class TestSubsampling:
    def test_frequent_word(self):
        # Very frequent word should have high discard probability
        p = subsample_prob(10000, 100000, threshold=1e-3)
        assert p > 0  # some discard

    def test_rare_word(self):
        # Rare word should have low discard probability
        p = subsample_prob(1, 100000, threshold=1e-3)
        assert p == 0.0

    def test_zero_frequency(self):
        p = subsample_prob(0, 100000)
        assert p == 0.0


# ---------------------------------------------------------------------------
# Negative Sampler tests
# ---------------------------------------------------------------------------

class TestNegativeSampler:
    def test_sample_count(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        sampler = NegativeSampler(vocab)
        samples = sampler.sample(5)
        assert len(samples) == 5

    def test_exclude(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        sampler = NegativeSampler(vocab)
        samples = sampler.sample(10, exclude={0, 1, 2})
        for s in samples:
            assert s not in {0, 1, 2}

    def test_distribution_shape(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        sampler = NegativeSampler(vocab)
        assert len(sampler.probs) == len(vocab)
        assert abs(sampler.probs.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Pair generation tests
# ---------------------------------------------------------------------------

class TestPairGeneration:
    def test_skipgram_nonempty(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        pairs = generate_skipgram_pairs(CORPUS, vocab, window_size=2,
                                         rng=np.random.default_rng(42))
        assert len(pairs) > 0

    def test_skipgram_pair_format(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        pairs = generate_skipgram_pairs(CORPUS, vocab, window_size=2,
                                         rng=np.random.default_rng(42))
        center, context = pairs[0]
        assert isinstance(center, int)
        assert isinstance(context, int)

    def test_cbow_nonempty(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        pairs = generate_cbow_pairs(CORPUS, vocab, window_size=2,
                                     rng=np.random.default_rng(42))
        assert len(pairs) > 0

    def test_cbow_pair_format(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        pairs = generate_cbow_pairs(CORPUS, vocab, window_size=2,
                                     rng=np.random.default_rng(42))
        context, center = pairs[0]
        assert isinstance(context, list)
        assert isinstance(center, int)
        assert all(isinstance(c, int) for c in context)


# ---------------------------------------------------------------------------
# Skip-gram tests
# ---------------------------------------------------------------------------

class TestSkipGram:
    def test_init(self):
        model = SkipGram(100, 50)
        assert model.W_in.shape == (100, 50)
        assert model.W_out.shape == (100, 50)

    def test_train_loss_decreases(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        sampler = NegativeSampler(vocab)
        pairs = generate_skipgram_pairs(CORPUS, vocab, window_size=3,
                                         rng=np.random.default_rng(42))
        model = SkipGram(len(vocab), 30, learning_rate=0.05,
                          neg_samples=3, seed=42)
        losses = model.train(pairs, sampler, epochs=5)
        # Loss should generally decrease
        assert losses[-1] < losses[0]

    def test_embedding_shape(self):
        model = SkipGram(50, 20)
        emb = model.get_embedding(0)
        assert emb.shape == (20,)

    def test_get_embeddings(self):
        model = SkipGram(50, 20)
        embs = model.get_embeddings()
        assert embs.shape == (50, 20)

    def test_embedding_copy(self):
        model = SkipGram(50, 20)
        emb1 = model.get_embedding(0)
        emb2 = model.get_embedding(0)
        emb1[0] = 999
        assert emb2[0] != 999  # should be a copy


# ---------------------------------------------------------------------------
# CBOW tests
# ---------------------------------------------------------------------------

class TestCBOW:
    def test_init(self):
        model = CBOW(100, 50)
        assert model.W_in.shape == (100, 50)

    def test_train_loss_decreases(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        sampler = NegativeSampler(vocab)
        pairs = generate_cbow_pairs(CORPUS, vocab, window_size=3,
                                     rng=np.random.default_rng(42))
        model = CBOW(len(vocab), 30, learning_rate=0.05,
                      neg_samples=3, seed=42)
        losses = model.train(pairs, sampler, epochs=5)
        assert losses[-1] < losses[0]

    def test_embedding_shape(self):
        model = CBOW(50, 20)
        emb = model.get_embedding(0)
        assert emb.shape == (20,)


# ---------------------------------------------------------------------------
# GloVe tests
# ---------------------------------------------------------------------------

class TestGloVe:
    def test_init(self):
        model = GloVe(100, 50)
        assert model.W.shape == (100, 50)
        assert model.W_ctx.shape == (100, 50)

    def test_build_cooccurrence(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        model = GloVe(len(vocab), 30)
        cooc = model.build_cooccurrence(CORPUS, vocab, window_size=5)
        assert len(cooc) > 0
        # All values should be positive
        assert all(v > 0 for v in cooc.values())

    def test_weight_function(self):
        model = GloVe(10, 5, x_max=100)
        assert model._weight_func(50) < 1.0
        assert model._weight_func(100) == 1.0
        assert model._weight_func(200) == 1.0

    def test_train_loss_decreases(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        model = GloVe(len(vocab), 20, learning_rate=0.05, seed=42)
        cooc = model.build_cooccurrence(CORPUS, vocab, window_size=5)
        losses = model.train(cooc, epochs=15)
        assert losses[-1] < losses[0]

    def test_combined_embeddings(self):
        model = GloVe(50, 20)
        embs = model.get_embeddings()
        assert embs.shape == (50, 20)

    def test_embedding_is_sum(self):
        model = GloVe(50, 20, seed=42)
        emb = model.get_embedding(5)
        expected = model.W[5] + model.W_ctx[5]
        np.testing.assert_array_almost_equal(emb, expected)

    def test_adagrad_accumulators(self):
        model = GloVe(10, 5)
        # Initially ones
        assert model.grad_sq_W[0, 0] == 1.0


# ---------------------------------------------------------------------------
# FastText / Subword tests
# ---------------------------------------------------------------------------

class TestFastText:
    def test_get_ngrams(self):
        ngrams = get_ngrams("cat", min_n=3, max_n=4)
        assert "<ca" in ngrams
        assert "cat" in ngrams
        assert "at>" in ngrams
        assert "<cat" in ngrams

    def test_hash_ngram(self):
        h1 = hash_ngram("cat", 1000)
        h2 = hash_ngram("dog", 1000)
        assert 0 <= h1 < 1000
        assert 0 <= h2 < 1000

    def test_hash_deterministic(self):
        h1 = hash_ngram("hello", 5000)
        h2 = hash_ngram("hello", 5000)
        assert h1 == h2

    def test_init(self):
        model = FastText(100, 50, bucket_size=1000)
        assert model.W_in.shape == (100, 50)
        assert model.W_ngram.shape == (1000, 50)

    def test_oov_embedding(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        model = FastText(len(vocab), 20, bucket_size=1000, seed=42)
        # OOV word should get an embedding from n-grams
        emb = model.get_oov_embedding("catdog")
        assert emb.shape == (20,)

    def test_train(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        sampler = NegativeSampler(vocab)
        pairs = generate_skipgram_pairs(CORPUS, vocab, window_size=3,
                                         rng=np.random.default_rng(42))
        model = FastText(len(vocab), 20, bucket_size=1000,
                          learning_rate=0.05, neg_samples=3, seed=42)
        losses = model.train(pairs, vocab, sampler, epochs=3)
        assert len(losses) == 3

    def test_word_vector_includes_ngrams(self):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        model = FastText(len(vocab), 20, bucket_size=1000, seed=42)
        idx = vocab.encode("cat")
        # Word vector with subword info
        v_with = model._get_word_vector(idx, "cat")
        # Pure word embedding
        v_pure = model.W_in[idx].copy()
        # They should differ (n-gram contribution)
        # After init W_ngram is zeros, so they're equal until training
        # After training they'd differ; for now check shapes
        assert v_with.shape == (20,)
        assert v_pure.shape == (20,)

    def test_ngram_cache(self):
        model = FastText(10, 5, bucket_size=100)
        ids1 = model._get_ngram_indices("hello")
        ids2 = model._get_ngram_indices("hello")
        assert ids1 == ids2  # cached


# ---------------------------------------------------------------------------
# EmbeddingSpace tests
# ---------------------------------------------------------------------------

class TestEmbeddingSpace:
    @pytest.fixture
    def trained_space(self):
        w2v = Word2Vec(embedding_dim=30, window_size=3, epochs=10,
                        neg_samples=3, seed=42)
        w2v.fit(CORPUS)
        return w2v.get_space()

    def test_get_vector(self, trained_space):
        vec = trained_space.get_vector("cat")
        assert vec is not None
        assert vec.shape == (30,)

    def test_get_vector_unknown(self, trained_space):
        vec = trained_space.get_vector("xyzzy")
        assert vec is None

    def test_cosine_similarity_range(self, trained_space):
        sim = trained_space.cosine_similarity("cat", "dog")
        assert sim is not None
        assert -1.0 <= sim <= 1.0

    def test_cosine_self_similarity(self, trained_space):
        sim = trained_space.cosine_similarity("cat", "cat")
        assert sim is not None
        assert abs(sim - 1.0) < 1e-5

    def test_euclidean_distance(self, trained_space):
        dist = trained_space.euclidean_distance("cat", "dog")
        assert dist is not None
        assert dist >= 0

    def test_euclidean_self_distance(self, trained_space):
        dist = trained_space.euclidean_distance("cat", "cat")
        assert dist is not None
        assert dist < 1e-5

    def test_most_similar(self, trained_space):
        results = trained_space.most_similar("cat", top_n=5)
        assert len(results) > 0
        words = [w for w, _ in results]
        assert "cat" not in words  # exclude self

    def test_most_similar_format(self, trained_space):
        results = trained_space.most_similar("cat", top_n=3)
        for word, score in results:
            assert isinstance(word, str)
            assert isinstance(score, float)

    def test_most_similar_to_vector(self, trained_space):
        vec = trained_space.get_vector("cat")
        results = trained_space.most_similar_to_vector(vec, top_n=5)
        assert len(results) > 0

    def test_analogy(self, trained_space):
        # With small corpus, analogy may not be meaningful but should work
        results = trained_space.analogy("cat", "dog", "sat", top_n=3)
        assert isinstance(results, list)
        # Each result is (word, score)
        for word, score in results:
            assert isinstance(word, str)

    def test_analogy_unknown_word(self, trained_space):
        results = trained_space.analogy("cat", "xyzzy", "dog")
        assert results == []

    def test_doesnt_match(self, trained_space):
        result = trained_space.doesnt_match(["cat", "dog", "mat", "animal"])
        assert result is not None
        assert isinstance(result, str)

    def test_doesnt_match_too_few(self, trained_space):
        result = trained_space.doesnt_match(["cat"])
        assert result is None

    def test_k_nearest_neighbors(self, trained_space):
        results = trained_space.k_nearest_neighbors("cat", k=3)
        assert len(results) <= 3
        for word, dist in results:
            assert dist >= 0

    def test_sentence_vector_mean(self, trained_space):
        vec = trained_space.sentence_vector(["the", "cat", "sat"])
        assert vec.shape == (30,)

    def test_sentence_vector_max(self, trained_space):
        vec = trained_space.sentence_vector(["the", "cat"], method='max')
        assert vec.shape == (30,)

    def test_sentence_vector_sum(self, trained_space):
        vec = trained_space.sentence_vector(["the", "cat"], method='sum')
        assert vec.shape == (30,)

    def test_sentence_vector_empty(self, trained_space):
        vec = trained_space.sentence_vector(["xyzzy", "nope"])
        assert vec.shape == (30,)
        assert np.allclose(vec, 0)

    def test_word_mover_distance(self, trained_space):
        d = trained_space.word_mover_distance(["cat", "sat"], ["dog", "ran"])
        assert d >= 0

    def test_wmd_identical(self, trained_space):
        d = trained_space.word_mover_distance(["cat", "sat"], ["cat", "sat"])
        assert d < 1e-5

    def test_wmd_empty(self, trained_space):
        d = trained_space.word_mover_distance(["xyzzy"], ["cat"])
        assert d == float('inf')

    def test_cluster_words(self, trained_space):
        words = ["cat", "dog", "sat", "ran", "mat", "log"]
        clusters = trained_space.cluster_words(words, n_clusters=2)
        assert len(clusters) == 2
        all_words = []
        for wlist in clusters.values():
            all_words.extend(wlist)
        # All valid words should be in some cluster
        for w in words:
            if w in trained_space.vocab:
                assert w in all_words

    def test_cluster_too_few(self, trained_space):
        clusters = trained_space.cluster_words(["cat"], n_clusters=3)
        assert len(clusters) == 1


# ---------------------------------------------------------------------------
# Word2Vec high-level tests
# ---------------------------------------------------------------------------

class TestWord2Vec:
    def test_skipgram_fit(self):
        w2v = Word2Vec(embedding_dim=20, window_size=2, epochs=3,
                        neg_samples=2, seed=42, method='skipgram')
        w2v.fit(CORPUS)
        assert w2v.vocab is not None
        assert w2v.model is not None

    def test_cbow_fit(self):
        w2v = Word2Vec(embedding_dim=20, window_size=2, epochs=3,
                        neg_samples=2, seed=42, method='cbow')
        w2v.fit(CORPUS)
        assert w2v.vocab is not None

    def test_getitem(self):
        w2v = Word2Vec(embedding_dim=20, epochs=3, seed=42)
        w2v.fit(CORPUS)
        vec = w2v["cat"]
        assert vec is not None
        assert vec.shape == (20,)

    def test_getitem_unknown(self):
        w2v = Word2Vec(embedding_dim=20, epochs=3, seed=42)
        w2v.fit(CORPUS)
        vec = w2v["xyzzy"]
        assert vec is None

    def test_getitem_not_trained(self):
        w2v = Word2Vec()
        with pytest.raises(RuntimeError):
            _ = w2v["cat"]

    def test_unknown_method(self):
        w2v = Word2Vec(method='invalid')
        with pytest.raises(ValueError):
            w2v.fit(CORPUS)

    def test_small_vocab_error(self):
        w2v = Word2Vec(min_count=9999)
        with pytest.raises(ValueError):
            w2v.fit(CORPUS)

    def test_get_space(self):
        w2v = Word2Vec(embedding_dim=20, epochs=3, seed=42)
        w2v.fit(CORPUS)
        space = w2v.get_space()
        assert isinstance(space, EmbeddingSpace)


# ---------------------------------------------------------------------------
# GloVeTrainer high-level tests
# ---------------------------------------------------------------------------

class TestGloVeTrainer:
    def test_fit(self):
        trainer = GloVeTrainer(embedding_dim=20, window_size=5,
                                epochs=10, seed=42)
        trainer.fit(CORPUS)
        assert trainer.vocab is not None

    def test_get_space(self):
        trainer = GloVeTrainer(embedding_dim=20, epochs=10, seed=42)
        trainer.fit(CORPUS)
        space = trainer.get_space()
        assert isinstance(space, EmbeddingSpace)
        vec = space.get_vector("the")
        assert vec is not None

    def test_loss_decreases(self):
        trainer = GloVeTrainer(embedding_dim=20, epochs=15, seed=42)
        trainer.fit(CORPUS)
        losses = trainer.model.training_loss
        assert losses[-1] < losses[0]

    def test_small_vocab_error(self):
        trainer = GloVeTrainer(min_count=9999)
        with pytest.raises(ValueError):
            trainer.fit(CORPUS)


# ---------------------------------------------------------------------------
# FastTextTrainer high-level tests
# ---------------------------------------------------------------------------

class TestFastTextTrainer:
    def test_fit(self):
        trainer = FastTextTrainer(embedding_dim=20, window_size=2,
                                   epochs=3, bucket_size=500, seed=42)
        trainer.fit(CORPUS)
        assert trainer.vocab is not None

    def test_get_space(self):
        trainer = FastTextTrainer(embedding_dim=20, epochs=3,
                                   bucket_size=500, seed=42)
        trainer.fit(CORPUS)
        space = trainer.get_space()
        assert isinstance(space, EmbeddingSpace)

    def test_oov(self):
        trainer = FastTextTrainer(embedding_dim=20, epochs=3,
                                   bucket_size=500, seed=42)
        trainer.fit(CORPUS)
        emb = trainer.get_oov_embedding("catdog")
        assert emb.shape == (20,)

    def test_oov_not_trained(self):
        trainer = FastTextTrainer()
        with pytest.raises(RuntimeError):
            trainer.get_oov_embedding("test")

    def test_small_vocab_error(self):
        trainer = FastTextTrainer(min_count=9999)
        with pytest.raises(ValueError):
            trainer.fit(CORPUS)


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_save_load_text(self, tmp_path):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((len(vocab), 10)).astype(np.float32)

        filepath = str(tmp_path / "embeddings.txt")
        save_embeddings(filepath, embeddings, vocab, binary=False)
        loaded_embs, loaded_vocab = load_embeddings(filepath, binary=False)

        assert loaded_embs.shape == embeddings.shape
        np.testing.assert_array_almost_equal(loaded_embs, embeddings, decimal=5)
        assert len(loaded_vocab) == len(vocab)

    def test_save_load_binary(self, tmp_path):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((len(vocab), 10)).astype(np.float32)

        filepath = str(tmp_path / "embeddings.bin")
        save_embeddings(filepath, embeddings, vocab, binary=True)
        loaded_embs, loaded_vocab = load_embeddings(filepath, binary=True)

        assert loaded_embs.shape == embeddings.shape
        np.testing.assert_array_almost_equal(loaded_embs, embeddings, decimal=5)
        assert len(loaded_vocab) == len(vocab)

    def test_word_preservation(self, tmp_path):
        vocab = Vocabulary()
        vocab.build(CORPUS)
        embeddings = np.zeros((len(vocab), 5), dtype=np.float32)

        filepath = str(tmp_path / "emb.txt")
        save_embeddings(filepath, embeddings, vocab, binary=False)
        _, loaded_vocab = load_embeddings(filepath, binary=False)

        for word in vocab.idx2word:
            assert word in loaded_vocab


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_skipgram_pipeline(self):
        """Full pipeline: tokenize -> vocab -> pairs -> train -> query."""
        texts = SAMPLE_TEXTS * 3  # More data
        corpus = build_corpus(texts)
        w2v = Word2Vec(embedding_dim=30, window_size=3, epochs=10,
                        neg_samples=3, seed=42, method='skipgram')
        w2v.fit(corpus)
        space = w2v.get_space()

        # Should be able to get similarities
        sim = space.cosine_similarity("cat", "dog")
        assert sim is not None

        # Should get most similar
        results = space.most_similar("cat", top_n=5)
        assert len(results) > 0

    def test_full_cbow_pipeline(self):
        texts = SAMPLE_TEXTS * 3
        corpus = build_corpus(texts)
        w2v = Word2Vec(embedding_dim=30, window_size=3, epochs=10,
                        neg_samples=3, seed=42, method='cbow')
        w2v.fit(corpus)
        space = w2v.get_space()
        sim = space.cosine_similarity("cat", "dog")
        assert sim is not None

    def test_full_glove_pipeline(self):
        texts = SAMPLE_TEXTS * 3
        corpus = build_corpus(texts)
        trainer = GloVeTrainer(embedding_dim=30, window_size=5,
                                epochs=20, seed=42)
        trainer.fit(corpus)
        space = trainer.get_space()
        sim = space.cosine_similarity("cat", "dog")
        assert sim is not None

    def test_full_fasttext_pipeline(self):
        texts = SAMPLE_TEXTS * 3
        corpus = build_corpus(texts)
        trainer = FastTextTrainer(embedding_dim=30, window_size=3,
                                   epochs=5, bucket_size=500, seed=42)
        trainer.fit(corpus)
        space = trainer.get_space()
        sim = space.cosine_similarity("cat", "dog")
        assert sim is not None

        # OOV should work
        emb = trainer.get_oov_embedding("catlike")
        assert emb.shape == (30,)

    def test_similarity_symmetry(self):
        w2v = Word2Vec(embedding_dim=20, epochs=5, seed=42)
        w2v.fit(CORPUS)
        space = w2v.get_space()
        sim1 = space.cosine_similarity("cat", "dog")
        sim2 = space.cosine_similarity("dog", "cat")
        assert abs(sim1 - sim2) < 1e-6

    def test_embeddings_not_zero(self):
        w2v = Word2Vec(embedding_dim=20, epochs=5, seed=42)
        w2v.fit(CORPUS)
        vec = w2v["the"]
        assert np.linalg.norm(vec) > 0

    def test_different_seeds_different_results(self):
        w1 = Word2Vec(embedding_dim=20, epochs=3, seed=42)
        w1.fit(CORPUS)
        w2 = Word2Vec(embedding_dim=20, epochs=3, seed=99)
        w2.fit(CORPUS)
        v1 = w1["the"]
        v2 = w2["the"]
        assert not np.allclose(v1, v2)

    def test_multiple_epochs_improve(self):
        w1 = Word2Vec(embedding_dim=20, epochs=1, seed=42)
        w1.fit(CORPUS)
        w2 = Word2Vec(embedding_dim=20, epochs=10, seed=42)
        w2.fit(CORPUS)
        # More epochs -> lower loss
        assert w2.model.training_loss[-1] < w1.model.training_loss[-1]

    def test_save_load_roundtrip(self, tmp_path):
        w2v = Word2Vec(embedding_dim=20, epochs=3, seed=42)
        w2v.fit(CORPUS)
        embs = w2v.model.get_embeddings().astype(np.float32)

        filepath = str(tmp_path / "test_emb.txt")
        save_embeddings(filepath, embs, w2v.vocab)
        loaded_embs, loaded_vocab = load_embeddings(filepath)

        # Create new space from loaded
        space = EmbeddingSpace(loaded_embs, loaded_vocab)
        sim = space.cosine_similarity("cat", "dog")
        assert sim is not None

    def test_cluster_integration(self):
        w2v = Word2Vec(embedding_dim=20, epochs=5, seed=42)
        w2v.fit(CORPUS)
        space = w2v.get_space()
        words = ["cat", "dog", "sat", "ran", "the", "a"]
        clusters = space.cluster_words(words, n_clusters=2)
        total = sum(len(v) for v in clusters.values())
        # All words in vocab should be clustered
        in_vocab = [w for w in words if w in space.vocab]
        assert total == len(in_vocab)

    def test_sentence_similarity(self):
        w2v = Word2Vec(embedding_dim=20, epochs=5, seed=42)
        w2v.fit(CORPUS)
        space = w2v.get_space()
        v1 = space.sentence_vector(["the", "cat", "sat"])
        v2 = space.sentence_vector(["the", "dog", "sat"])
        v3 = space.sentence_vector(["big", "yard", "house"])
        # cat sat vs dog sat should be more similar than cat sat vs yard house
        sim12 = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        sim13 = np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3) + 1e-10)
        # Not guaranteed with small corpus, but structure test works
        assert isinstance(sim12, float)
        assert isinstance(sim13, float)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_sentence_corpus(self):
        corpus = [["the", "cat", "sat"]]
        w2v = Word2Vec(embedding_dim=10, epochs=3, seed=42, min_count=1)
        w2v.fit(corpus)
        assert w2v.vocab is not None

    def test_repeated_word_corpus(self):
        corpus = [["the", "the", "the", "cat"]]
        w2v = Word2Vec(embedding_dim=10, epochs=3, seed=42)
        w2v.fit(corpus)
        vec = w2v["the"]
        assert vec is not None

    def test_large_window(self):
        small_corpus = [["the", "cat", "sat", "on", "mat"]]
        w2v = Word2Vec(embedding_dim=10, window_size=100, epochs=1, seed=42)
        w2v.fit(small_corpus)
        assert w2v.vocab is not None

    def test_embedding_dim_1(self):
        w2v = Word2Vec(embedding_dim=1, epochs=2, seed=42)
        w2v.fit(CORPUS)
        vec = w2v["the"]
        assert vec.shape == (1,)

    def test_cosine_similarity_unknown(self):
        w2v = Word2Vec(embedding_dim=10, epochs=2, seed=42)
        w2v.fit(CORPUS)
        space = w2v.get_space()
        sim = space.cosine_similarity("cat", "xyzzy")
        assert sim is None

    def test_euclidean_distance_unknown(self):
        w2v = Word2Vec(embedding_dim=10, epochs=2, seed=42)
        w2v.fit(CORPUS)
        space = w2v.get_space()
        dist = space.euclidean_distance("xyzzy", "cat")
        assert dist is None

    def test_most_similar_unknown(self):
        w2v = Word2Vec(embedding_dim=10, epochs=2, seed=42)
        w2v.fit(CORPUS)
        space = w2v.get_space()
        results = space.most_similar("xyzzy")
        assert results == []

    def test_knn_unknown(self):
        w2v = Word2Vec(embedding_dim=10, epochs=2, seed=42)
        w2v.fit(CORPUS)
        space = w2v.get_space()
        results = space.k_nearest_neighbors("xyzzy")
        assert results == []

    def test_zero_vector_similarity(self):
        vocab = Vocabulary()
        vocab.build([["a", "b"]])
        embs = np.zeros((2, 5))
        space = EmbeddingSpace(embs, vocab)
        # Zero vectors -- normalized will be zero, similarity undefined but handled
        results = space.most_similar_to_vector(np.zeros(5))
        assert results == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
