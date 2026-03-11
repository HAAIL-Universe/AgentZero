"""Tests for C148: Text Classification"""
import pytest
import sys, os, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from text_classification import (
    TextPreprocessor, BagOfWords, TfIdf,
    NaiveBayesClassifier, LogisticRegression, MLPClassifier,
    RNNClassifier, EmbeddingClassifier,
    ClassificationMetrics, TextClassificationPipeline,
    train_test_split, make_sentiment_data, make_topic_data, cross_validate
)


# ===========================================================================
# TextPreprocessor
# ===========================================================================

class TestTextPreprocessor:
    def test_fit_builds_vocab(self):
        pp = TextPreprocessor()
        pp.fit(["hello world", "hello there"])
        assert pp.fitted
        assert pp.vocab_size >= 4  # PAD, UNK, hello, world/there
        assert '<PAD>' in pp.word2idx
        assert '<UNK>' in pp.word2idx

    def test_encode_basic(self):
        pp = TextPreprocessor()
        pp.fit(["the cat sat"])
        encoded = pp.encode("the cat")
        assert len(encoded) == 2
        assert all(isinstance(i, int) for i in encoded)

    def test_encode_unknown_word(self):
        pp = TextPreprocessor()
        pp.fit(["hello world"])
        encoded = pp.encode("hello xyz")
        assert encoded[1] == pp.word2idx['<UNK>']

    def test_max_seq_len(self):
        pp = TextPreprocessor(max_seq_len=3)
        pp.fit(["a b c d e f"])
        encoded = pp.encode("a b c d e f")
        assert len(encoded) == 3

    def test_encode_batch_padding(self):
        pp = TextPreprocessor()
        pp.fit(["short", "a longer sentence here"])
        batch = pp.encode_batch(["short", "a longer sentence here"])
        assert len(batch[0]) == len(batch[1])  # Same length due to padding

    def test_decode(self):
        pp = TextPreprocessor()
        pp.fit(["hello world"])
        encoded = pp.encode("hello world")
        decoded = pp.decode(encoded)
        assert decoded == ['hello', 'world']

    def test_min_freq(self):
        pp = TextPreprocessor(min_freq=2)
        pp.fit(["hello world", "hello there", "hello again"])
        # "hello" appears 3x, others 1x
        assert 'hello' in pp.word2idx
        # world/there/again appear once, should be filtered
        assert 'world' not in pp.word2idx

    def test_max_vocab_size(self):
        pp = TextPreprocessor(max_vocab_size=2)
        pp.fit(["a b c d e"])
        # 2 real words + PAD + UNK = 4
        assert pp.vocab_size == 4

    def test_lower_case(self):
        pp = TextPreprocessor(lower=True)
        pp.fit(["Hello WORLD"])
        assert 'hello' in pp.word2idx
        assert 'Hello' not in pp.word2idx

    def test_no_lower_case(self):
        pp = TextPreprocessor(lower=False)
        pp.fit(["Hello WORLD"])
        assert 'Hello' in pp.word2idx


# ===========================================================================
# BagOfWords
# ===========================================================================

class TestBagOfWords:
    def test_fit_transform(self):
        bow = BagOfWords()
        X = bow.fit_transform(["hello world", "hello there"])
        assert X.shape[0] == 2
        assert X.shape[1] == 3  # hello, world, there

    def test_binary_mode(self):
        bow = BagOfWords(binary=True)
        X = bow.fit_transform(["hello hello hello"])
        assert X[0, bow.vocabulary['hello']] == 1.0  # Binary, not count

    def test_max_features(self):
        bow = BagOfWords(max_features=2)
        X = bow.fit_transform(["a b c d e"])
        assert X.shape[1] == 2

    def test_transform_unseen_words(self):
        bow = BagOfWords()
        bow.fit(["hello world"])
        X = bow.transform(["hello xyz"])
        assert X[0, bow.vocabulary['hello']] == 1.0
        # xyz not in vocab, ignored

    def test_counts_correct(self):
        bow = BagOfWords()
        X = bow.fit_transform(["cat cat dog"])
        assert X[0, bow.vocabulary['cat']] == 2.0
        assert X[0, bow.vocabulary['dog']] == 1.0

    def test_multiple_documents(self):
        bow = BagOfWords()
        X = bow.fit_transform(["a b", "c d", "a c"])
        assert X.shape == (3, 4)


# ===========================================================================
# TfIdf
# ===========================================================================

class TestTfIdf:
    def test_fit_transform(self):
        tfidf = TfIdf()
        X = tfidf.fit_transform(["hello world", "hello there", "world here"])
        assert X.shape[0] == 3
        assert X.shape[1] == 4  # hello, world, there, here

    def test_l2_normalized(self):
        tfidf = TfIdf()
        X = tfidf.fit_transform(["hello world", "hello there"])
        for i in range(X.shape[0]):
            norm = np.linalg.norm(X[i])
            if norm > 0:
                assert abs(norm - 1.0) < 1e-6

    def test_idf_computed(self):
        tfidf = TfIdf()
        tfidf.fit(["cat dog", "cat bird", "cat fish"])
        # "cat" appears in all 3 docs, should have lower IDF
        # "dog" appears in 1 doc, should have higher IDF
        assert tfidf.idf['cat'] < tfidf.idf['dog']

    def test_sublinear_tf(self):
        tfidf = TfIdf(sublinear_tf=True)
        X = tfidf.fit_transform(["cat cat cat cat dog"])
        # With sublinear TF, repeated words get dampened
        assert X.shape == (1, 2)

    def test_max_features(self):
        tfidf = TfIdf(max_features=2)
        X = tfidf.fit_transform(["a b c d e"])
        assert X.shape[1] == 2

    def test_transform_new_docs(self):
        tfidf = TfIdf()
        tfidf.fit(["hello world", "foo bar"])
        X = tfidf.transform(["hello foo"])
        assert X.shape == (1, 4)


# ===========================================================================
# NaiveBayesClassifier
# ===========================================================================

class TestNaiveBayes:
    def test_binary_classification(self):
        bow = BagOfWords()
        texts = ["good great excellent", "bad terrible awful",
                 "good wonderful nice", "horrible bad poor"]
        labels = ['pos', 'neg', 'pos', 'neg']
        X = bow.fit_transform(texts)
        nb = NaiveBayesClassifier()
        nb.fit(X, labels)
        assert nb.fitted
        # Should predict training data correctly
        preds = nb.predict(X)
        assert preds == labels

    def test_predict_proba(self):
        bow = BagOfWords()
        texts = ["good great", "bad terrible", "good nice", "bad awful"]
        labels = ['pos', 'neg', 'pos', 'neg']
        X = bow.fit_transform(texts)
        nb = NaiveBayesClassifier()
        nb.fit(X, labels)
        probs = nb.predict_proba(X)
        assert len(probs) == 4
        for p in probs:
            assert abs(sum(p.values()) - 1.0) < 1e-6

    def test_smoothing(self):
        bow = BagOfWords()
        texts = ["cat", "dog"]
        labels = ['a', 'b']
        X = bow.fit_transform(texts)
        nb = NaiveBayesClassifier(alpha=1.0)
        nb.fit(X, labels)
        # Should not crash on unseen features
        X_new = bow.transform(["cat dog"])
        preds = nb.predict(X_new)
        assert preds[0] in ['a', 'b']

    def test_multiclass(self):
        bow = BagOfWords()
        texts = ["sport game team", "computer code data", "cell biology research",
                 "match score win", "software program digital", "experiment lab study"]
        labels = ['sports', 'tech', 'science', 'sports', 'tech', 'science']
        X = bow.fit_transform(texts)
        nb = NaiveBayesClassifier()
        nb.fit(X, labels)
        preds = nb.predict(X)
        # Should get most training samples right
        correct = sum(1 for t, p in zip(labels, preds) if t == p)
        assert correct >= 4

    def test_sentiment_data(self):
        texts, labels = make_sentiment_data(n_per_class=30, seed=42)
        bow = BagOfWords()
        X = bow.fit_transform(texts)
        nb = NaiveBayesClassifier()
        nb.fit(X, labels)
        preds = nb.predict(X)
        acc = sum(1 for t, p in zip(labels, preds) if t == p) / len(labels)
        assert acc > 0.8


# ===========================================================================
# LogisticRegression
# ===========================================================================

class TestLogisticRegression:
    def test_binary_classification(self):
        bow = BagOfWords()
        texts = ["good great wonderful", "bad terrible awful",
                 "excellent amazing love", "horrible poor worst"]
        labels = ['pos', 'neg', 'pos', 'neg']
        X = bow.fit_transform(texts)
        lr = LogisticRegression(lr=0.1, epochs=200, seed=42)
        lr.fit(X, labels)
        assert lr.fitted
        preds = lr.predict(X)
        assert preds == labels

    def test_multiclass(self):
        bow = BagOfWords()
        texts, labels = make_topic_data(n_per_class=20, seed=42)
        X = bow.fit_transform(texts)
        lr = LogisticRegression(lr=0.1, epochs=200, seed=42)
        lr.fit(X, labels)
        preds = lr.predict(X)
        acc = sum(1 for t, p in zip(labels, preds) if t == p) / len(labels)
        assert acc > 0.7

    def test_predict_proba(self):
        bow = BagOfWords()
        texts = ["good nice", "bad poor"]
        labels = ['pos', 'neg']
        X = bow.fit_transform(texts)
        lr = LogisticRegression(lr=0.1, epochs=100, seed=42)
        lr.fit(X, labels)
        probs = lr.predict_proba(X)
        assert len(probs) == 2
        for p in probs:
            total = sum(p.values())
            assert abs(total - 1.0) < 1e-5

    def test_loss_decreases(self):
        bow = BagOfWords()
        texts, labels = make_sentiment_data(n_per_class=20, seed=42)
        X = bow.fit_transform(texts)
        lr = LogisticRegression(lr=0.05, epochs=50, seed=42)
        lr.fit(X, labels)
        assert lr.loss_history[-1] < lr.loss_history[0]

    def test_regularization(self):
        bow = BagOfWords()
        texts, labels = make_sentiment_data(n_per_class=20, seed=42)
        X = bow.fit_transform(texts)
        lr = LogisticRegression(lr=0.05, epochs=100, regularization=0.1, seed=42)
        lr.fit(X, labels)
        # Should still learn
        preds = lr.predict(X)
        acc = sum(1 for t, p in zip(labels, preds) if t == p) / len(labels)
        assert acc > 0.6


# ===========================================================================
# MLPClassifier
# ===========================================================================

class TestMLPClassifier:
    def test_binary_classification(self):
        bow = BagOfWords()
        texts = ["good great wonderful amazing", "bad terrible awful horrible",
                 "excellent love beautiful perfect", "worst poor boring disappointing"]
        labels = ['pos', 'neg', 'pos', 'neg']
        X = bow.fit_transform(texts)
        mlp = MLPClassifier(hidden_sizes=(16,), lr=0.01, epochs=100, seed=42)
        mlp.fit(X, labels)
        assert mlp.fitted
        preds = mlp.predict(X)
        correct = sum(1 for t, p in zip(labels, preds) if t == p)
        assert correct >= 3

    def test_multiclass(self):
        bow = BagOfWords()
        texts, labels = make_topic_data(n_per_class=15, seed=42)
        X = bow.fit_transform(texts)
        mlp = MLPClassifier(hidden_sizes=(32, 16), lr=0.01, epochs=100,
                            batch_size=16, seed=42)
        mlp.fit(X, labels)
        preds = mlp.predict(X)
        acc = sum(1 for t, p in zip(labels, preds) if t == p) / len(labels)
        assert acc > 0.5

    def test_predict_proba(self):
        bow = BagOfWords()
        texts = ["good nice", "bad poor", "good love"]
        labels = ['pos', 'neg', 'pos']
        X = bow.fit_transform(texts)
        mlp = MLPClassifier(hidden_sizes=(8,), lr=0.01, epochs=50, seed=42)
        mlp.fit(X, labels)
        probs = mlp.predict_proba(X)
        assert len(probs) == 3
        for p in probs:
            total = sum(p.values())
            assert abs(total - 1.0) < 1e-5

    def test_loss_history(self):
        bow = BagOfWords()
        texts, labels = make_sentiment_data(n_per_class=20, seed=42)
        X = bow.fit_transform(texts)
        mlp = MLPClassifier(hidden_sizes=(16,), lr=0.01, epochs=30, seed=42)
        mlp.fit(X, labels)
        assert len(mlp.loss_history) == 30

    def test_dropout(self):
        bow = BagOfWords()
        texts, labels = make_sentiment_data(n_per_class=20, seed=42)
        X = bow.fit_transform(texts)
        mlp = MLPClassifier(hidden_sizes=(16,), lr=0.01, epochs=30,
                            dropout=0.2, seed=42)
        mlp.fit(X, labels)
        assert mlp.fitted


# ===========================================================================
# RNNClassifier
# ===========================================================================

class TestRNNClassifier:
    def test_lstm_classifier(self):
        pp = TextPreprocessor(max_seq_len=10)
        texts = ["good great wonderful", "bad terrible awful",
                 "excellent amazing love", "horrible poor worst",
                 "good nice beautiful", "bad boring ugly"]
        labels = ['pos', 'neg', 'pos', 'neg', 'pos', 'neg']
        pp.fit(texts)
        X = pp.encode_batch(texts)

        rnn = RNNClassifier(vocab_size=pp.vocab_size, embed_dim=8,
                            hidden_size=8, cell_type='lstm',
                            lr=0.05, epochs=30, batch_size=6, seed=42)
        rnn.fit(X, labels)
        assert rnn.fitted
        assert len(rnn.loss_history) == 30

    def test_gru_classifier(self):
        pp = TextPreprocessor(max_seq_len=10)
        texts = ["good great", "bad terrible", "excellent amazing", "horrible poor"]
        labels = ['pos', 'neg', 'pos', 'neg']
        pp.fit(texts)
        X = pp.encode_batch(texts)

        rnn = RNNClassifier(vocab_size=pp.vocab_size, embed_dim=8,
                            hidden_size=8, cell_type='gru',
                            lr=0.05, epochs=20, batch_size=4, seed=42)
        rnn.fit(X, labels)
        assert rnn.fitted

    def test_predict(self):
        pp = TextPreprocessor(max_seq_len=8)
        texts = ["good great wonderful", "bad terrible awful",
                 "excellent amazing", "horrible poor"]
        labels = ['pos', 'neg', 'pos', 'neg']
        pp.fit(texts)
        X = pp.encode_batch(texts)

        rnn = RNNClassifier(vocab_size=pp.vocab_size, embed_dim=8,
                            hidden_size=8, cell_type='lstm',
                            lr=0.05, epochs=40, batch_size=4, seed=42)
        rnn.fit(X, labels)
        preds = rnn.predict(X)
        assert len(preds) == 4
        assert all(p in ['pos', 'neg'] for p in preds)

    def test_predict_proba(self):
        pp = TextPreprocessor(max_seq_len=8)
        texts = ["good great", "bad terrible"]
        labels = ['pos', 'neg']
        pp.fit(texts)
        X = pp.encode_batch(texts)

        rnn = RNNClassifier(vocab_size=pp.vocab_size, embed_dim=8,
                            hidden_size=8, cell_type='lstm',
                            lr=0.05, epochs=20, batch_size=2, seed=42)
        rnn.fit(X, labels)
        probs = rnn.predict_proba(X)
        assert len(probs) == 2
        for p in probs:
            assert abs(sum(p.values()) - 1.0) < 1e-5

    def test_loss_decreases(self):
        pp = TextPreprocessor(max_seq_len=8)
        texts = ["good great wonderful", "bad terrible awful",
                 "excellent amazing love", "horrible poor worst"]
        labels = ['pos', 'neg', 'pos', 'neg']
        pp.fit(texts)
        X = pp.encode_batch(texts)

        rnn = RNNClassifier(vocab_size=pp.vocab_size, embed_dim=8,
                            hidden_size=8, cell_type='lstm',
                            lr=0.05, epochs=50, batch_size=4, seed=42)
        rnn.fit(X, labels)
        assert rnn.loss_history[-1] < rnn.loss_history[0]


# ===========================================================================
# EmbeddingClassifier
# ===========================================================================

class TestEmbeddingClassifier:
    def _make_space(self):
        """Create a small embedding space for testing."""
        # Build a simple corpus and train embeddings
        texts = ["good great excellent wonderful love",
                 "bad terrible awful horrible hate",
                 "neutral normal average okay fine"]
        corpus = []
        for t in texts:
            corpus.append(t.split())
        # Repeat to get more training data
        corpus = corpus * 5

        from word_embeddings import Word2Vec, Vocabulary
        w2v = Word2Vec(embedding_dim=16, window_size=2, epochs=10,
                       min_count=1, method='skipgram')
        w2v.fit(corpus)
        return w2v.get_space()

    def test_logistic_classifier(self):
        space = self._make_space()
        texts = ["good great wonderful", "bad terrible awful",
                 "excellent love wonderful", "horrible hate terrible"]
        labels = ['pos', 'neg', 'pos', 'neg']

        ec = EmbeddingClassifier(space, pooling='mean', classifier='logistic',
                                  lr=0.1, epochs=200, seed=42)
        ec.fit(texts, labels)
        assert ec.fitted
        preds = ec.predict(texts)
        assert len(preds) == 4

    def test_nb_classifier(self):
        space = self._make_space()
        texts = ["good great", "bad terrible", "excellent love", "horrible hate"]
        labels = ['pos', 'neg', 'pos', 'neg']

        ec = EmbeddingClassifier(space, pooling='mean', classifier='nb', seed=42)
        ec.fit(texts, labels)
        preds = ec.predict(texts)
        assert len(preds) == 4

    def test_predict_proba(self):
        space = self._make_space()
        texts = ["good great", "bad terrible"]
        labels = ['pos', 'neg']

        ec = EmbeddingClassifier(space, pooling='mean', classifier='logistic',
                                  lr=0.1, epochs=100, seed=42)
        ec.fit(texts, labels)
        probs = ec.predict_proba(texts)
        assert len(probs) == 2
        for p in probs:
            assert abs(sum(p.values()) - 1.0) < 1e-5

    def test_max_pooling(self):
        space = self._make_space()
        texts = ["good great", "bad terrible"]
        labels = ['pos', 'neg']

        ec = EmbeddingClassifier(space, pooling='max', classifier='logistic',
                                  lr=0.1, epochs=100, seed=42)
        ec.fit(texts, labels)
        assert ec.fitted


# ===========================================================================
# ClassificationMetrics
# ===========================================================================

class TestMetrics:
    def test_accuracy_perfect(self):
        assert ClassificationMetrics.accuracy(['a', 'b', 'a'], ['a', 'b', 'a']) == 1.0

    def test_accuracy_half(self):
        assert ClassificationMetrics.accuracy(['a', 'b', 'a', 'b'],
                                               ['a', 'a', 'a', 'a']) == 0.5

    def test_confusion_matrix(self):
        y_true = ['a', 'a', 'b', 'b']
        y_pred = ['a', 'b', 'a', 'b']
        cm = ClassificationMetrics.confusion_matrix(y_true, y_pred)
        assert cm['a']['a'] == 1
        assert cm['a']['b'] == 1
        assert cm['b']['a'] == 1
        assert cm['b']['b'] == 1

    def test_precision_per_class(self):
        y_true = ['a', 'a', 'b', 'b']
        y_pred = ['a', 'a', 'a', 'b']
        p = ClassificationMetrics.precision(y_true, y_pred, label='a')
        assert abs(p - 2/3) < 1e-6  # TP=2, FP=1

    def test_recall_per_class(self):
        y_true = ['a', 'a', 'b', 'b']
        y_pred = ['a', 'a', 'a', 'b']
        r = ClassificationMetrics.recall(y_true, y_pred, label='a')
        assert r == 1.0  # TP=2, FN=0

    def test_f1_score(self):
        y_true = ['a', 'a', 'b', 'b']
        y_pred = ['a', 'a', 'a', 'b']
        f1 = ClassificationMetrics.f1_score(y_true, y_pred, label='a')
        p = 2/3
        r = 1.0
        expected = 2 * p * r / (p + r)
        assert abs(f1 - expected) < 1e-6

    def test_macro_precision(self):
        y_true = ['a', 'a', 'b', 'b']
        y_pred = ['a', 'b', 'a', 'b']
        p = ClassificationMetrics.precision(y_true, y_pred, average='macro')
        assert abs(p - 0.5) < 1e-6  # Each class: 1/(1+1) = 0.5

    def test_micro_precision(self):
        y_true = ['a', 'a', 'b', 'b']
        y_pred = ['a', 'b', 'a', 'b']
        p = ClassificationMetrics.precision(y_true, y_pred, average='micro')
        assert abs(p - 0.5) < 1e-6

    def test_classification_report(self):
        y_true = ['a', 'a', 'b', 'b', 'c', 'c']
        y_pred = ['a', 'a', 'b', 'c', 'c', 'c']
        report = ClassificationMetrics.classification_report(y_true, y_pred)
        assert 'a' in report
        assert 'b' in report
        assert 'c' in report
        assert 'macro' in report
        assert 'accuracy' in report
        assert report['a']['precision'] == 1.0
        assert report['a']['recall'] == 1.0

    def test_multiclass_f1(self):
        y_true = ['a', 'b', 'c', 'a', 'b', 'c']
        y_pred = ['a', 'b', 'c', 'a', 'b', 'c']
        f1 = ClassificationMetrics.f1_score(y_true, y_pred, average='macro')
        assert f1 == 1.0

    def test_zero_division(self):
        # Class 'c' never predicted
        y_true = ['a', 'b', 'c']
        y_pred = ['a', 'b', 'a']
        p = ClassificationMetrics.precision(y_true, y_pred, label='c')
        assert p == 0.0
        r = ClassificationMetrics.recall(y_true, y_pred, label='c')
        assert r == 0.0

    def test_empty_input(self):
        acc = ClassificationMetrics.accuracy([], [])
        assert acc == 0.0


# ===========================================================================
# TextClassificationPipeline
# ===========================================================================

class TestPipeline:
    def test_bow_nb_pipeline(self):
        pipe = TextClassificationPipeline(
            vectorizer=BagOfWords(),
            classifier=NaiveBayesClassifier()
        )
        texts, labels = make_sentiment_data(n_per_class=30, seed=42)
        pipe.fit(texts, labels)
        preds = pipe.predict(texts)
        acc = ClassificationMetrics.accuracy(labels, preds)
        assert acc > 0.8

    def test_tfidf_lr_pipeline(self):
        pipe = TextClassificationPipeline(
            vectorizer=TfIdf(),
            classifier=LogisticRegression(lr=0.5, epochs=200, seed=42)
        )
        texts, labels = make_sentiment_data(n_per_class=30, seed=42)
        pipe.fit(texts, labels)
        preds = pipe.predict(texts)
        acc = ClassificationMetrics.accuracy(labels, preds)
        assert acc > 0.7

    def test_bow_mlp_pipeline(self):
        pipe = TextClassificationPipeline(
            vectorizer=BagOfWords(),
            classifier=MLPClassifier(hidden_sizes=(16,), lr=0.01,
                                     epochs=50, seed=42)
        )
        texts, labels = make_sentiment_data(n_per_class=20, seed=42)
        pipe.fit(texts, labels)
        preds = pipe.predict(texts)
        acc = ClassificationMetrics.accuracy(labels, preds)
        assert acc > 0.5

    def test_evaluate_returns_report(self):
        pipe = TextClassificationPipeline(
            vectorizer=BagOfWords(),
            classifier=NaiveBayesClassifier()
        )
        texts, labels = make_sentiment_data(n_per_class=20, seed=42)
        pipe.fit(texts, labels)
        report = pipe.evaluate(texts, labels)
        assert 'positive' in report
        assert 'negative' in report
        assert 'macro' in report
        assert 'accuracy' in report

    def test_predict_proba(self):
        pipe = TextClassificationPipeline(
            vectorizer=BagOfWords(),
            classifier=NaiveBayesClassifier()
        )
        texts, labels = make_sentiment_data(n_per_class=20, seed=42)
        pipe.fit(texts, labels)
        probs = pipe.predict_proba(texts[:3])
        assert len(probs) == 3
        for p in probs:
            assert abs(sum(p.values()) - 1.0) < 1e-6

    def test_topic_classification(self):
        pipe = TextClassificationPipeline(
            vectorizer=TfIdf(),
            classifier=NaiveBayesClassifier()
        )
        texts, labels = make_topic_data(n_per_class=20, seed=42)
        pipe.fit(texts, labels)
        preds = pipe.predict(texts)
        acc = ClassificationMetrics.accuracy(labels, preds)
        assert acc > 0.7

    def test_default_pipeline(self):
        pipe = TextClassificationPipeline()
        texts, labels = make_sentiment_data(n_per_class=20, seed=42)
        pipe.fit(texts, labels)
        preds = pipe.predict(texts)
        assert len(preds) == len(labels)


# ===========================================================================
# Data Utilities
# ===========================================================================

class TestDataUtils:
    def test_train_test_split(self):
        texts = [f"text {i}" for i in range(100)]
        labels = ['a'] * 50 + ['b'] * 50
        t_train, t_test, l_train, l_test = train_test_split(texts, labels, test_ratio=0.2)
        assert len(t_train) == 80
        assert len(t_test) == 20
        assert len(l_train) == 80
        assert len(l_test) == 20

    def test_train_test_split_deterministic(self):
        texts = [f"text {i}" for i in range(50)]
        labels = ['a'] * 25 + ['b'] * 25
        r1 = train_test_split(texts, labels, seed=42)
        r2 = train_test_split(texts, labels, seed=42)
        assert r1[0] == r2[0]

    def test_make_sentiment_data(self):
        texts, labels = make_sentiment_data(n_per_class=10, seed=42)
        assert len(texts) == 20
        assert labels.count('positive') == 10
        assert labels.count('negative') == 10

    def test_make_topic_data(self):
        texts, labels = make_topic_data(n_per_class=10, seed=42)
        assert len(texts) == 30
        assert labels.count('sports') == 10
        assert labels.count('tech') == 10
        assert labels.count('science') == 10


# ===========================================================================
# Cross-Validation
# ===========================================================================

class TestCrossValidation:
    def test_basic_cv(self):
        texts, labels = make_sentiment_data(n_per_class=25, seed=42)
        result = cross_validate(
            lambda: TextClassificationPipeline(BagOfWords(), NaiveBayesClassifier()),
            texts, labels, n_folds=5, seed=42
        )
        assert 'accuracy_mean' in result
        assert 'accuracy_std' in result
        assert len(result['fold_accuracies']) == 5
        assert result['accuracy_mean'] > 0.5

    def test_cv_fold_count(self):
        texts, labels = make_sentiment_data(n_per_class=30, seed=42)
        result = cross_validate(
            lambda: TextClassificationPipeline(BagOfWords(), NaiveBayesClassifier()),
            texts, labels, n_folds=3, seed=42
        )
        assert len(result['fold_accuracies']) == 3

    def test_cv_deterministic(self):
        texts, labels = make_sentiment_data(n_per_class=20, seed=42)
        r1 = cross_validate(
            lambda: TextClassificationPipeline(BagOfWords(), NaiveBayesClassifier()),
            texts, labels, n_folds=3, seed=42
        )
        r2 = cross_validate(
            lambda: TextClassificationPipeline(BagOfWords(), NaiveBayesClassifier()),
            texts, labels, n_folds=3, seed=42
        )
        assert r1['fold_accuracies'] == r2['fold_accuracies']


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestIntegration:
    def test_full_sentiment_pipeline(self):
        """End-to-end sentiment analysis."""
        texts, labels = make_sentiment_data(n_per_class=40, seed=42)
        train_t, test_t, train_l, test_l = train_test_split(texts, labels, test_ratio=0.25)

        pipe = TextClassificationPipeline(
            vectorizer=TfIdf(),
            classifier=NaiveBayesClassifier()
        )
        pipe.fit(train_t, train_l)
        report = pipe.evaluate(test_t, test_l)
        assert report['accuracy'] > 0.6

    def test_full_topic_pipeline(self):
        """End-to-end topic classification."""
        texts, labels = make_topic_data(n_per_class=25, seed=42)
        train_t, test_t, train_l, test_l = train_test_split(texts, labels, test_ratio=0.2)

        pipe = TextClassificationPipeline(
            vectorizer=BagOfWords(),
            classifier=NaiveBayesClassifier()
        )
        pipe.fit(train_t, train_l)
        report = pipe.evaluate(test_t, test_l)
        assert report['accuracy'] > 0.5

    def test_compare_classifiers(self):
        """Compare multiple classifiers on same data."""
        texts, labels = make_sentiment_data(n_per_class=30, seed=42)
        bow = BagOfWords()
        X = bow.fit_transform(texts)

        nb = NaiveBayesClassifier()
        nb.fit(X, labels)
        nb_acc = ClassificationMetrics.accuracy(labels, nb.predict(X))

        lr = LogisticRegression(lr=0.1, epochs=200, seed=42)
        lr.fit(X, labels)
        lr_acc = ClassificationMetrics.accuracy(labels, lr.predict(X))

        # Both should do reasonably well on training data
        assert nb_acc > 0.7
        assert lr_acc > 0.7

    def test_rnn_sentiment(self):
        """RNN on sentiment data."""
        texts, labels = make_sentiment_data(n_per_class=15, seed=42)
        pp = TextPreprocessor(max_seq_len=12)
        pp.fit(texts)
        X = pp.encode_batch(texts)

        rnn = RNNClassifier(vocab_size=pp.vocab_size, embed_dim=8,
                            hidden_size=8, cell_type='lstm',
                            lr=0.05, epochs=30, batch_size=10, seed=42)
        rnn.fit(X, labels)
        preds = rnn.predict(X)
        assert len(preds) == len(labels)

    def test_pipeline_with_preprocessor(self):
        """Pipeline using preprocessor for vectorizer."""
        pp = TextPreprocessor(lower=True)
        pipe = TextClassificationPipeline(
            vectorizer=BagOfWords(preprocessor=pp),
            classifier=NaiveBayesClassifier()
        )
        texts, labels = make_sentiment_data(n_per_class=20, seed=42)
        pipe.fit(texts, labels)
        preds = pipe.predict(texts)
        assert len(preds) == len(labels)

    def test_tfidf_topic_pipeline(self):
        """TF-IDF + LR on topic data."""
        texts, labels = make_topic_data(n_per_class=20, seed=42)
        pipe = TextClassificationPipeline(
            vectorizer=TfIdf(),
            classifier=LogisticRegression(lr=0.5, epochs=300, seed=42)
        )
        pipe.fit(texts, labels)
        preds = pipe.predict(texts)
        acc = ClassificationMetrics.accuracy(labels, preds)
        assert acc > 0.6

    def test_multiclass_metrics(self):
        """Full metrics on multiclass problem."""
        texts, labels = make_topic_data(n_per_class=20, seed=42)
        pipe = TextClassificationPipeline(
            vectorizer=BagOfWords(),
            classifier=NaiveBayesClassifier()
        )
        pipe.fit(texts, labels)
        report = pipe.evaluate(texts, labels)

        # Check structure
        for topic in ['sports', 'tech', 'science']:
            assert topic in report
            assert 'precision' in report[topic]
            assert 'recall' in report[topic]
            assert 'f1' in report[topic]
            assert 'support' in report[topic]

    def test_binary_bow_features(self):
        """Binary bag of words features."""
        pipe = TextClassificationPipeline(
            vectorizer=BagOfWords(binary=True),
            classifier=NaiveBayesClassifier()
        )
        texts, labels = make_sentiment_data(n_per_class=20, seed=42)
        pipe.fit(texts, labels)
        preds = pipe.predict(texts)
        acc = ClassificationMetrics.accuracy(labels, preds)
        assert acc > 0.6

    def test_pipeline_generalization(self):
        """Test on unseen data (not just training accuracy)."""
        texts, labels = make_sentiment_data(n_per_class=50, seed=42)
        train_t, test_t, train_l, test_l = train_test_split(
            texts, labels, test_ratio=0.3, seed=42
        )

        pipe = TextClassificationPipeline(
            vectorizer=BagOfWords(),
            classifier=NaiveBayesClassifier()
        )
        pipe.fit(train_t, train_l)
        test_preds = pipe.predict(test_t)
        test_acc = ClassificationMetrics.accuracy(test_l, test_preds)
        # Should generalize reasonably on synthetic data
        assert test_acc > 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
