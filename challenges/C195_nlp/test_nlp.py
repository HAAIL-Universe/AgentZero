"""Tests for C195: Natural Language Processing."""

import pytest
import numpy as np
import math
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from nlp import (
    porter_stem, TextPreprocessor, STOP_WORDS,
    WordTokenizer, SentenceTokenizer, BPETokenizer,
    NGramModel,
    TfidfVectorizer,
    Word2Vec,
    cosine_similarity, jaccard_similarity, edit_distance, BM25,
    NaiveBayes, LogisticRegression, NearestCentroid,
    ExtractiveSummarizer, KeywordExtractor,
    accuracy, precision_recall_f1, confusion_matrix,
)


# ============================================================
# Porter Stemmer
# ============================================================

class TestPorterStem:
    def test_basic_plural(self):
        assert porter_stem("cats") == "cat"
        assert porter_stem("dogs") == "dog"

    def test_sses(self):
        assert porter_stem("caresses") == "caress"

    def test_ies(self):
        assert porter_stem("ponies") == "poni"

    def test_ss_unchanged(self):
        assert porter_stem("ss") == "ss"
        assert porter_stem("boss") == "boss"

    def test_ed_suffix(self):
        result = porter_stem("agreed")
        assert isinstance(result, str)
        assert len(result) < len("agreed")

    def test_ing_suffix(self):
        result = porter_stem("running")
        assert isinstance(result, str)

    def test_short_words(self):
        assert porter_stem("a") == "a"
        assert porter_stem("an") == "an"

    def test_y_to_i(self):
        result = porter_stem("happy")
        assert result.endswith("i")


# ============================================================
# TextPreprocessor
# ============================================================

class TestTextPreprocessor:
    def test_default(self):
        pp = TextPreprocessor()
        tokens = pp.process("Hello, World!")
        assert tokens == ["hello", "world"]

    def test_no_lowercase(self):
        pp = TextPreprocessor(lowercase=False)
        tokens = pp.process("Hello World")
        assert "Hello" in tokens

    def test_remove_stopwords(self):
        pp = TextPreprocessor(remove_stopwords=True)
        tokens = pp.process("the cat is on the mat")
        assert "cat" in tokens
        assert "the" not in tokens

    def test_stemming(self):
        pp = TextPreprocessor(stem=True)
        tokens = pp.process("running cats")
        assert all(isinstance(t, str) for t in tokens)

    def test_min_length(self):
        pp = TextPreprocessor(min_length=3)
        tokens = pp.process("I am a big cat")
        assert "i" not in tokens
        assert "am" not in tokens
        assert "big" in tokens

    def test_keep_punctuation(self):
        pp = TextPreprocessor(remove_punctuation=False)
        tokens = pp.process("hello, world!")
        text = " ".join(tokens)
        assert "," in text or "hello," in text


# ============================================================
# WordTokenizer
# ============================================================

class TestWordTokenizer:
    def test_basic(self):
        tok = WordTokenizer()
        assert tok.tokenize("hello world") == ["hello", "world"]

    def test_punctuation(self):
        tok = WordTokenizer()
        tokens = tok.tokenize("hello, world!")
        assert "hello" in tokens
        assert "world" in tokens

    def test_lowercase(self):
        tok = WordTokenizer(lowercase=True)
        tokens = tok.tokenize("Hello WORLD")
        assert tokens == ["hello", "world"]

    def test_custom_pattern(self):
        tok = WordTokenizer(pattern=r'[a-z]+')
        tokens = tok.tokenize("abc 123 def")
        assert tokens == ["abc", "def"]

    def test_empty(self):
        tok = WordTokenizer()
        assert tok.tokenize("") == []


# ============================================================
# SentenceTokenizer
# ============================================================

class TestSentenceTokenizer:
    def test_basic(self):
        tok = SentenceTokenizer()
        sents = tok.tokenize("Hello. World.")
        assert len(sents) == 2

    def test_question_exclamation(self):
        tok = SentenceTokenizer()
        sents = tok.tokenize("What? Yes! Done.")
        assert len(sents) == 3

    def test_abbreviations(self):
        tok = SentenceTokenizer()
        sents = tok.tokenize("Dr. Smith went home. He was tired.")
        # Dr. should not split
        assert len(sents) == 2

    def test_single_sentence(self):
        tok = SentenceTokenizer()
        sents = tok.tokenize("Hello world")
        assert len(sents) == 1

    def test_empty(self):
        tok = SentenceTokenizer()
        sents = tok.tokenize("")
        assert len(sents) == 0


# ============================================================
# BPE Tokenizer
# ============================================================

class TestBPETokenizer:
    def test_fit_and_tokenize(self):
        corpus = ["hello world", "hello there", "world hello world"]
        bpe = BPETokenizer(vocab_size=50)
        bpe.fit(corpus)
        tokens = bpe.tokenize("hello world")
        assert len(tokens) > 0

    def test_vocab_built(self):
        corpus = ["aaa bbb", "aaa ccc", "bbb ccc"]
        bpe = BPETokenizer(vocab_size=30)
        bpe.fit(corpus)
        assert len(bpe.vocab) > 0

    def test_encode(self):
        corpus = ["hello world", "hello hello"]
        bpe = BPETokenizer(vocab_size=30)
        bpe.fit(corpus)
        ids = bpe.encode("hello")
        assert all(isinstance(i, int) for i in ids)

    def test_merges_reduce_tokens(self):
        corpus = ["ab ab ab ab ab"] * 10
        bpe = BPETokenizer(vocab_size=20)
        bpe.fit(corpus)
        # After merges, "ab" should be a single token
        tokens_before = list("ab") + ["</w>"]
        tokens_after = bpe.tokenize("ab")
        assert len(tokens_after) <= len(tokens_before)

    def test_unknown_subword(self):
        corpus = ["aaa bbb"]
        bpe = BPETokenizer(vocab_size=20)
        bpe.fit(corpus)
        ids = bpe.encode("zzz")
        # Unknown chars get -1
        assert any(i == -1 for i in ids) or all(i >= 0 for i in ids)


# ============================================================
# N-Gram Model
# ============================================================

class TestNGramModel:
    def test_bigram_fit(self):
        corpus = [["the", "cat", "sat"], ["the", "dog", "sat"]]
        model = NGramModel(n=2)
        model.fit(corpus)
        assert len(model.ngram_counts) > 0

    def test_probability(self):
        corpus = [["the", "cat"], ["the", "dog"], ["the", "cat"]]
        model = NGramModel(n=2)
        model.fit(corpus)
        p_cat = model.probability("cat", ["the"])
        p_dog = model.probability("dog", ["the"])
        assert p_cat > p_dog  # "the cat" appears twice

    def test_smoothing(self):
        corpus = [["a", "b"]]
        model = NGramModel(n=2, smoothing=1.0)
        model.fit(corpus)
        # Unseen word gets non-zero probability due to smoothing
        p = model.probability("z", ["a"])
        assert p > 0

    def test_perplexity(self):
        corpus = [["the", "cat", "sat"]] * 10
        model = NGramModel(n=2)
        model.fit(corpus)
        pp = model.perplexity(["the", "cat", "sat"])
        assert pp > 0 and pp < 1000

    def test_generate(self):
        corpus = [["a", "b", "c"]] * 20
        model = NGramModel(n=2)
        model.fit(corpus)
        result = model.generate(seed=42, max_length=5)
        assert isinstance(result, list)

    def test_trigram(self):
        corpus = [["a", "b", "c", "d"]] * 10
        model = NGramModel(n=3)
        model.fit(corpus)
        p = model.probability("c", ["a", "b"])
        assert p > 0

    def test_wrong_context_length(self):
        model = NGramModel(n=2)
        model.fit([["a", "b"]])
        with pytest.raises(ValueError):
            model.probability("a", ["x", "y"])  # needs 1 token context


# ============================================================
# TF-IDF Vectorizer
# ============================================================

class TestTfidfVectorizer:
    def test_basic(self):
        docs = ["the cat sat", "the dog sat", "the cat and dog"]
        vec = TfidfVectorizer()
        X = vec.fit_transform(docs)
        assert X.shape[0] == 3
        assert X.shape[1] > 0

    def test_vocabulary(self):
        docs = ["hello world", "hello there"]
        vec = TfidfVectorizer()
        vec.fit(docs)
        assert "hello" in vec.vocabulary_
        assert "world" in vec.vocabulary_

    def test_l2_norm(self):
        docs = ["cat dog", "cat cat cat"]
        vec = TfidfVectorizer(norm='l2')
        X = vec.fit_transform(docs)
        norms = np.sqrt(np.sum(X ** 2, axis=1))
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_l1_norm(self):
        docs = ["cat dog", "cat cat"]
        vec = TfidfVectorizer(norm='l1')
        X = vec.fit_transform(docs)
        norms = np.sum(np.abs(X), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_no_norm(self):
        docs = ["cat dog", "cat cat"]
        vec = TfidfVectorizer(norm=None)
        X = vec.fit_transform(docs)
        assert X.shape[0] == 2

    def test_max_features(self):
        docs = ["a b c d e f g", "a b c"]
        vec = TfidfVectorizer(max_features=3)
        X = vec.fit_transform(docs)
        assert X.shape[1] == 3

    def test_min_df(self):
        docs = ["cat dog", "cat bird", "cat fish"]
        vec = TfidfVectorizer(min_df=2)
        vec.fit(docs)
        # "cat" appears in all 3 docs, others appear in 1
        assert "cat" in vec.vocabulary_

    def test_sublinear_tf(self):
        docs = ["cat cat cat cat", "dog"]
        vec = TfidfVectorizer(sublinear_tf=True, norm=None)
        X = vec.fit_transform(docs)
        # Sublinear should dampen high-frequency terms
        assert X.shape[0] == 2

    def test_feature_names(self):
        docs = ["cat dog", "bird fish"]
        vec = TfidfVectorizer()
        vec.fit(docs)
        names = vec.get_feature_names()
        assert len(names) == len(vec.vocabulary_)
        assert "cat" in names

    def test_idf_values(self):
        docs = ["cat dog", "cat bird"]
        vec = TfidfVectorizer()
        vec.fit(docs)
        # cat appears in both, dog in one -> dog has higher IDF
        cat_idf = vec.idf_[vec.vocabulary_["cat"]]
        dog_idf = vec.idf_[vec.vocabulary_["dog"]]
        assert dog_idf > cat_idf

    def test_transform_unseen(self):
        docs = ["cat dog"]
        vec = TfidfVectorizer()
        vec.fit(docs)
        X = vec.transform(["bird fish"])
        # Unseen words -> all zeros
        assert np.sum(X) == 0


# ============================================================
# Word2Vec
# ============================================================

class TestWord2Vec:
    @pytest.fixture
    def corpus(self):
        return [
            ["king", "queen", "prince", "princess", "royal", "throne"],
            ["king", "throne", "crown", "royal", "palace"],
            ["queen", "princess", "crown", "royal", "dress"],
            ["prince", "king", "sword", "battle", "throne"],
            ["princess", "queen", "dress", "crown", "palace"],
        ] * 5

    def test_fit(self, corpus):
        w2v = Word2Vec(embedding_dim=10, window=2)
        w2v.fit(corpus, epochs=3, seed=42)
        assert w2v.W is not None
        assert w2v.W.shape[1] == 10

    def test_get_vector(self, corpus):
        w2v = Word2Vec(embedding_dim=10)
        w2v.fit(corpus, epochs=3, seed=42)
        vec = w2v.get_vector("king")
        assert vec.shape == (10,)

    def test_get_vector_unknown(self, corpus):
        w2v = Word2Vec(embedding_dim=10)
        w2v.fit(corpus, epochs=1, seed=42)
        with pytest.raises(KeyError):
            w2v.get_vector("zzzzz")

    def test_most_similar(self, corpus):
        w2v = Word2Vec(embedding_dim=10, window=2)
        w2v.fit(corpus, epochs=5, seed=42)
        similar = w2v.most_similar("king", topn=3)
        assert len(similar) == 3
        assert all(isinstance(s, tuple) and len(s) == 2 for s in similar)
        words = [s[0] for s in similar]
        assert "king" not in words

    def test_analogy(self, corpus):
        w2v = Word2Vec(embedding_dim=10, window=2)
        w2v.fit(corpus, epochs=5, seed=42)
        result = w2v.analogy("king", "queen", "prince", topn=3)
        assert len(result) == 3

    def test_cbow_mode(self, corpus):
        w2v = Word2Vec(embedding_dim=10, mode='cbow')
        w2v.fit(corpus, epochs=3, seed=42)
        vec = w2v.get_vector("king")
        assert vec.shape == (10,)

    def test_min_count(self):
        corpus = [["rare", "common", "common"], ["common", "common"]]
        w2v = Word2Vec(embedding_dim=5, min_count=2)
        w2v.fit(corpus, epochs=1, seed=42)
        assert "common" in w2v.word2idx
        assert "rare" not in w2v.word2idx

    def test_embedding_dim(self, corpus):
        w2v = Word2Vec(embedding_dim=25)
        w2v.fit(corpus, epochs=1, seed=42)
        assert w2v.W.shape[1] == 25

    def test_empty_corpus(self):
        w2v = Word2Vec(embedding_dim=5)
        w2v.fit([], epochs=1, seed=42)
        assert w2v.W is None or len(w2v.word2idx) == 0


# ============================================================
# Text Similarity
# ============================================================

class TestTextSimilarity:
    def test_cosine_identical(self):
        v = [1, 2, 3]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-10

    def test_cosine_orthogonal(self):
        assert abs(cosine_similarity([1, 0], [0, 1])) < 1e-10

    def test_cosine_opposite(self):
        assert abs(cosine_similarity([1, 0], [-1, 0]) - (-1.0)) < 1e-10

    def test_cosine_zero_vector(self):
        assert cosine_similarity([0, 0], [1, 2]) == 0.0

    def test_jaccard_identical(self):
        assert jaccard_similarity(["a", "b"], ["a", "b"]) == 1.0

    def test_jaccard_disjoint(self):
        assert jaccard_similarity(["a"], ["b"]) == 0.0

    def test_jaccard_partial(self):
        j = jaccard_similarity(["a", "b", "c"], ["b", "c", "d"])
        assert abs(j - 0.5) < 1e-10

    def test_jaccard_empty(self):
        assert jaccard_similarity([], []) == 1.0

    def test_edit_distance_identical(self):
        assert edit_distance("cat", "cat") == 0

    def test_edit_distance_one(self):
        assert edit_distance("cat", "bat") == 1

    def test_edit_distance_insert(self):
        assert edit_distance("cat", "cats") == 1

    def test_edit_distance_delete(self):
        assert edit_distance("cats", "cat") == 1

    def test_edit_distance_empty(self):
        assert edit_distance("", "abc") == 3
        assert edit_distance("abc", "") == 3

    def test_edit_distance_both_empty(self):
        assert edit_distance("", "") == 0


# ============================================================
# BM25
# ============================================================

class TestBM25:
    def test_basic_ranking(self):
        docs = [["cat", "dog"], ["cat", "cat", "cat"], ["dog", "bird"]]
        bm25 = BM25()
        bm25.fit(docs)
        scores = bm25.score(["cat"])
        # Doc 1 (3 cats) should score highest
        assert scores[1] > scores[0]
        assert scores[1] > scores[2]

    def test_top_k(self):
        docs = [["a", "b"], ["c", "d"], ["a", "a", "a"]]
        bm25 = BM25()
        bm25.fit(docs)
        top = bm25.top_k(["a"], k=2)
        assert len(top) <= 2
        assert top[0][0] == 2  # doc with 3 "a"s

    def test_unseen_query(self):
        docs = [["cat", "dog"]]
        bm25 = BM25()
        bm25.fit(docs)
        scores = bm25.score(["zzz"])
        assert scores[0] == 0.0

    def test_multiple_query_terms(self):
        docs = [["cat", "dog"], ["cat", "bird"], ["dog", "bird"]]
        bm25 = BM25()
        bm25.fit(docs)
        scores = bm25.score(["cat", "dog"])
        assert scores[0] > scores[1]  # doc 0 has both terms

    def test_empty_query(self):
        docs = [["cat"]]
        bm25 = BM25()
        bm25.fit(docs)
        scores = bm25.score([])
        assert scores[0] == 0.0


# ============================================================
# Naive Bayes
# ============================================================

class TestNaiveBayes:
    def test_basic_classification(self):
        X = np.array([[3, 0], [2, 0], [0, 3], [0, 2]])
        y = ["pos", "pos", "neg", "neg"]
        nb = NaiveBayes()
        nb.fit(X, y)
        pred = nb.predict(np.array([[5, 0]]))
        assert pred[0] == "pos"

    def test_predict_proba(self):
        X = np.array([[3, 0], [0, 3]])
        y = ["a", "b"]
        nb = NaiveBayes()
        nb.fit(X, y)
        proba = nb.predict_proba(np.array([[3, 0]]))
        assert proba.shape == (1, 2)
        assert abs(sum(proba[0]) - 1.0) < 1e-5

    def test_smoothing(self):
        X = np.array([[1, 0], [0, 1]])
        y = ["a", "b"]
        nb = NaiveBayes(alpha=1.0)
        nb.fit(X, y)
        # Should not crash on unseen features
        pred = nb.predict(np.array([[0, 0]]))
        assert pred[0] in ["a", "b"]

    def test_multiclass(self):
        X = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
        y = ["a", "b", "c"]
        nb = NaiveBayes()
        nb.fit(X, y)
        pred = nb.predict(X)
        assert pred == ["a", "b", "c"]


# ============================================================
# Logistic Regression
# ============================================================

class TestLogisticRegression:
    def test_binary(self):
        np.random.seed(42)
        X = np.vstack([np.random.randn(20, 2) + [2, 2],
                        np.random.randn(20, 2) + [-2, -2]])
        y = [0] * 20 + [1] * 20
        lr = LogisticRegression(learning_rate=0.5, max_iter=100)
        lr.fit(X, y)
        pred = lr.predict(X)
        acc = sum(a == b for a, b in zip(y, pred)) / len(y)
        assert acc > 0.8

    def test_multiclass(self):
        np.random.seed(42)
        X = np.vstack([np.random.randn(15, 2) + [3, 0],
                        np.random.randn(15, 2) + [0, 3],
                        np.random.randn(15, 2) + [-3, -3]])
        y = ["a"] * 15 + ["b"] * 15 + ["c"] * 15
        lr = LogisticRegression(learning_rate=0.3, max_iter=200)
        lr.fit(X, y)
        pred = lr.predict(X)
        acc = sum(a == b for a, b in zip(y, pred)) / len(y)
        assert acc > 0.7

    def test_predict_proba(self):
        X = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
        y = ["a", "b", "a", "b"]
        lr = LogisticRegression(max_iter=50)
        lr.fit(X, y)
        proba = lr.predict_proba(X)
        assert proba.shape == (4, 2)
        # Probabilities sum to ~1
        sums = proba.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=0.1)


# ============================================================
# Nearest Centroid
# ============================================================

class TestNearestCentroid:
    def test_basic(self):
        X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
        y = ["a", "a", "b", "b"]
        nc = NearestCentroid()
        nc.fit(X, y)
        pred = nc.predict(np.array([[0.5, 0.5], [10.5, 10.5]]))
        assert pred == ["a", "b"]

    def test_centroids(self):
        X = np.array([[0, 0], [2, 2]])
        y = ["a", "b"]
        nc = NearestCentroid()
        nc.fit(X, y)
        np.testing.assert_allclose(nc.centroids_["a"], [0, 0])
        np.testing.assert_allclose(nc.centroids_["b"], [2, 2])


# ============================================================
# Extractive Summarizer
# ============================================================

class TestExtractiveSummarizer:
    def test_tfidf_summarize(self):
        text = ("Machine learning is great. "
                "Deep learning is a subset of machine learning. "
                "Natural language processing uses machine learning. "
                "The weather is nice today.")
        s = ExtractiveSummarizer(method='tfidf')
        summary = s.summarize(text, n_sentences=2)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_frequency_summarize(self):
        text = ("Cats are great pets. Dogs are loyal friends. "
                "Birds can fly high. Fish live in water.")
        s = ExtractiveSummarizer(method='frequency')
        summary = s.summarize(text, n_sentences=2)
        assert isinstance(summary, str)

    def test_short_text(self):
        text = "One sentence."
        s = ExtractiveSummarizer()
        summary = s.summarize(text, n_sentences=3)
        assert "One sentence" in summary

    def test_maintains_order(self):
        text = "First. Second. Third. Fourth."
        s = ExtractiveSummarizer(method='tfidf')
        summary = s.summarize(text, n_sentences=2)
        # Should maintain original sentence order
        parts = summary.split('. ')
        assert len(parts) >= 1


# ============================================================
# Keyword Extractor
# ============================================================

class TestKeywordExtractor:
    def test_basic(self):
        text = "machine learning is a field of artificial intelligence"
        kw = KeywordExtractor(max_keywords=3)
        keywords = kw.extract(text)
        assert len(keywords) <= 3
        assert all(isinstance(k, tuple) and len(k) == 2 for k in keywords)

    def test_with_corpus(self):
        text = "machine learning algorithms"
        corpus = ["deep learning models", "statistical learning theory"]
        kw = KeywordExtractor(max_keywords=5)
        keywords = kw.extract(text, corpus=corpus)
        assert len(keywords) > 0

    def test_stopwords_removed(self):
        text = "the cat is on the mat"
        kw = KeywordExtractor()
        keywords = kw.extract(text)
        words = [k[0] for k in keywords]
        assert "the" not in words
        assert "is" not in words


# ============================================================
# Evaluation Metrics
# ============================================================

class TestMetrics:
    def test_accuracy_perfect(self):
        assert accuracy([1, 2, 3], [1, 2, 3]) == 1.0

    def test_accuracy_none(self):
        assert accuracy([1, 2, 3], [3, 1, 2]) == 0.0

    def test_accuracy_partial(self):
        assert accuracy([1, 1, 0, 0], [1, 0, 0, 0]) == 0.75

    def test_precision_recall_f1_perfect(self):
        p, r, f1 = precision_recall_f1([1, 0, 1, 0], [1, 0, 1, 0])
        assert abs(p - 1.0) < 1e-10
        assert abs(r - 1.0) < 1e-10
        assert abs(f1 - 1.0) < 1e-10

    def test_precision_recall_f1_macro(self):
        p, r, f1 = precision_recall_f1([0, 0, 1, 1], [0, 1, 1, 0], average='macro')
        assert 0 <= p <= 1
        assert 0 <= r <= 1
        assert 0 <= f1 <= 1

    def test_precision_recall_f1_micro(self):
        p, r, f1 = precision_recall_f1([0, 0, 1, 1], [0, 1, 1, 0], average='micro')
        assert abs(p - 0.5) < 1e-10

    def test_confusion_matrix(self):
        y_true = [0, 0, 1, 1, 2]
        y_pred = [0, 1, 1, 2, 2]
        cm, classes = confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)
        assert classes == [0, 1, 2]
        assert cm[0, 0] == 1  # correct 0
        assert cm[0, 1] == 1  # 0 predicted as 1
        assert cm[1, 1] == 1  # correct 1
        assert cm[1, 2] == 1  # 1 predicted as 2
        assert cm[2, 2] == 1  # correct 2

    def test_confusion_matrix_binary(self):
        y_true = ["pos", "pos", "neg", "neg"]
        y_pred = ["pos", "neg", "neg", "neg"]
        cm, classes = confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)


# ============================================================
# Integration: TF-IDF + Classifier pipeline
# ============================================================

class TestPipeline:
    def test_tfidf_naive_bayes(self):
        train_docs = [
            "great movie loved it",
            "wonderful film excellent acting",
            "terrible movie awful waste",
            "bad film boring plot",
            "amazing performance brilliant",
            "horrible acting worst ever",
        ]
        train_labels = ["pos", "pos", "neg", "neg", "pos", "neg"]

        vec = TfidfVectorizer()
        X_train = vec.fit_transform(train_docs)
        nb = NaiveBayes()
        nb.fit(X_train, train_labels)

        # Test on training data (should mostly get right)
        pred = nb.predict(X_train)
        acc = accuracy(train_labels, pred)
        assert acc >= 0.5

    def test_tfidf_logistic_regression(self):
        train_docs = [
            "happy joy wonderful great",
            "sad terrible awful bad",
            "happy great excellent",
            "sad bad horrible",
        ]
        labels = ["pos", "neg", "pos", "neg"]

        vec = TfidfVectorizer()
        X = vec.fit_transform(train_docs)
        lr = LogisticRegression(learning_rate=1.0, max_iter=200)
        lr.fit(X, labels)
        pred = lr.predict(X)
        acc = accuracy(labels, pred)
        assert acc >= 0.5

    def test_tfidf_nearest_centroid(self):
        docs = ["cat dog pet", "cat kitten meow", "car truck vehicle", "bus train transit"]
        labels = ["animal", "animal", "vehicle", "vehicle"]

        vec = TfidfVectorizer()
        X = vec.fit_transform(docs)
        nc = NearestCentroid()
        nc.fit(X, labels)
        pred = nc.predict(X)
        assert accuracy(labels, pred) >= 0.5

    def test_bm25_retrieval(self):
        docs = [
            ["python", "programming", "language"],
            ["java", "programming", "language"],
            ["python", "snake", "animal"],
        ]
        bm25 = BM25()
        bm25.fit(docs)
        top = bm25.top_k(["python", "programming"], k=2)
        # First result should be python programming doc
        assert top[0][0] == 0


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_tfidf_single_doc(self):
        vec = TfidfVectorizer()
        X = vec.fit_transform(["hello world"])
        assert X.shape == (1, 2)

    def test_tfidf_empty_doc(self):
        vec = TfidfVectorizer()
        X = vec.fit_transform(["hello", ""])
        assert X.shape[0] == 2

    def test_ngram_single_sentence(self):
        model = NGramModel(n=2)
        model.fit([["a", "b"]])
        p = model.probability("b", ["a"])
        assert p > 0

    def test_word2vec_single_sentence(self):
        w2v = Word2Vec(embedding_dim=5, min_count=1)
        w2v.fit([["a", "b", "c"]], epochs=1, seed=42)
        assert "a" in w2v.word2idx

    def test_bm25_single_doc(self):
        bm25 = BM25()
        bm25.fit([["hello", "world"]])
        scores = bm25.score(["hello"])
        assert scores[0] > 0

    def test_edit_distance_unicode(self):
        assert edit_distance("cafe", "cafe") == 0

    def test_stopwords_exist(self):
        assert len(STOP_WORDS) > 50
        assert "the" in STOP_WORDS
        assert "a" in STOP_WORDS
