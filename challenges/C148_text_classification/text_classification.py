"""
C148: Text Classification
Composing C147 (Word Embeddings) + C144 (RNN) + C140 (Neural Network)

Components:
- TextPreprocessor: tokenization, vocabulary, encoding
- BagOfWords / TF-IDF vectorizers
- NaiveBayesClassifier: multinomial NB baseline
- LogisticRegression: binary/multiclass via C140 Dense
- MLPClassifier: multi-layer perceptron via C140 Sequential
- RNNClassifier: LSTM/GRU sequence classifier via C144
- EmbeddingClassifier: C147 sentence vectors + classifier
- TextClassificationPipeline: end-to-end pipeline
- Evaluation: accuracy, precision, recall, F1, confusion matrix
"""

import sys, os, math, re, random
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C147_word_embeddings'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C144_rnn'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C140_neural_network'))

import numpy as np

from word_embeddings import tokenize, build_corpus, Vocabulary, Word2Vec, EmbeddingSpace
from rnn import (RNN, LSTMLayer, GRULayer, Embedding, SequenceModel,
                 SequenceCrossEntropyLoss, train_sequence_model, TimeDistributed,
                 pad_sequences)
from neural_network import (Tensor, Dense, Activation, Sequential, CrossEntropyLoss,
                            BinaryCrossEntropyLoss, MSELoss, SGD, Adam, Dropout,
                            train_step, one_hot, accuracy)


# ---------------------------------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------------------------------

class TextPreprocessor:
    """Tokenize, build vocab, encode texts to integer sequences."""

    def __init__(self, max_vocab_size=None, min_freq=1, lower=True,
                 max_seq_len=None, pad_token='<PAD>', unk_token='<UNK>'):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        self.fitted = False

    def fit(self, texts):
        """Build vocabulary from list of texts."""
        self.word_counts = Counter()
        for text in texts:
            tokens = tokenize(text, lower=self.lower)
            self.word_counts.update(tokens)

        # Filter by min_freq
        filtered = {w: c for w, c in self.word_counts.items() if c >= self.min_freq}

        # Sort by frequency descending
        sorted_words = sorted(filtered.keys(), key=lambda w: (-filtered[w], w))

        # Limit vocab size
        if self.max_vocab_size is not None:
            sorted_words = sorted_words[:self.max_vocab_size]

        # Build mappings with special tokens
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        for i, w in enumerate(sorted_words):
            self.word2idx[w] = i + 2
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.fitted = True
        return self

    @property
    def vocab_size(self):
        return len(self.word2idx)

    def encode(self, text):
        """Encode text to integer sequence."""
        tokens = tokenize(text, lower=self.lower)
        indices = [self.word2idx.get(t, self.word2idx[self.unk_token]) for t in tokens]
        if self.max_seq_len is not None:
            indices = indices[:self.max_seq_len]
        return indices

    def encode_batch(self, texts, pad=True):
        """Encode multiple texts, optionally padding to uniform length."""
        encoded = [self.encode(t) for t in texts]
        if pad:
            max_len = self.max_seq_len or max(len(s) for s in encoded)
            padded = []
            for seq in encoded:
                if len(seq) < max_len:
                    seq = seq + [0] * (max_len - len(seq))
                padded.append(seq)
            return padded
        return encoded

    def decode(self, indices):
        """Decode integer sequence back to tokens."""
        return [self.idx2word.get(i, self.unk_token) for i in indices
                if i != self.word2idx[self.pad_token]]

    def tokenize(self, text):
        """Tokenize text."""
        return tokenize(text, lower=self.lower)


# ---------------------------------------------------------------------------
# Vectorizers
# ---------------------------------------------------------------------------

class BagOfWords:
    """Bag-of-words vectorizer. Produces sparse-like count vectors."""

    def __init__(self, preprocessor=None, max_features=None, binary=False):
        self.preprocessor = preprocessor
        self.max_features = max_features
        self.binary = binary
        self.vocabulary = {}
        self.fitted = False

    def fit(self, texts):
        """Build vocabulary from texts."""
        word_counts = Counter()
        for text in texts:
            if self.preprocessor:
                tokens = self.preprocessor.tokenize(text)
            else:
                tokens = tokenize(text)
            word_counts.update(tokens)

        sorted_words = sorted(word_counts.keys(), key=lambda w: (-word_counts[w], w))
        if self.max_features:
            sorted_words = sorted_words[:self.max_features]

        self.vocabulary = {w: i for i, w in enumerate(sorted_words)}
        self.fitted = True
        return self

    def transform(self, texts):
        """Transform texts to count vectors."""
        result = []
        for text in texts:
            if self.preprocessor:
                tokens = self.preprocessor.tokenize(text)
            else:
                tokens = tokenize(text)
            vec = np.zeros(len(self.vocabulary))
            for t in tokens:
                if t in self.vocabulary:
                    if self.binary:
                        vec[self.vocabulary[t]] = 1.0
                    else:
                        vec[self.vocabulary[t]] += 1.0
            result.append(vec)
        return np.array(result)

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)


class TfIdf:
    """TF-IDF vectorizer."""

    def __init__(self, preprocessor=None, max_features=None, sublinear_tf=False):
        self.preprocessor = preprocessor
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.vocabulary = {}
        self.idf = {}
        self.fitted = False

    def fit(self, texts):
        """Compute IDF from corpus."""
        doc_freq = Counter()
        word_counts = Counter()
        n_docs = len(texts)

        for text in texts:
            if self.preprocessor:
                tokens = self.preprocessor.tokenize(text)
            else:
                tokens = tokenize(text)
            word_counts.update(tokens)
            unique_tokens = set(tokens)
            doc_freq.update(unique_tokens)

        sorted_words = sorted(word_counts.keys(), key=lambda w: (-word_counts[w], w))
        if self.max_features:
            sorted_words = sorted_words[:self.max_features]

        self.vocabulary = {w: i for i, w in enumerate(sorted_words)}

        # IDF: log(N / (1 + df)) + 1
        self.idf = {}
        for w in self.vocabulary:
            df = doc_freq.get(w, 0)
            self.idf[w] = math.log(n_docs / (1 + df)) + 1.0

        self.fitted = True
        return self

    def transform(self, texts):
        """Transform texts to TF-IDF vectors."""
        result = []
        for text in texts:
            if self.preprocessor:
                tokens = self.preprocessor.tokenize(text)
            else:
                tokens = tokenize(text)

            # Compute TF
            tf_counts = Counter(tokens)
            vec = np.zeros(len(self.vocabulary))
            for t, count in tf_counts.items():
                if t in self.vocabulary:
                    tf = count
                    if self.sublinear_tf:
                        tf = 1.0 + math.log(count) if count > 0 else 0.0
                    vec[self.vocabulary[t]] = tf * self.idf.get(t, 1.0)

            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            result.append(vec)
        return np.array(result)

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

class NaiveBayesClassifier:
    """Multinomial Naive Bayes for text classification."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing
        self.class_log_prior = {}
        self.feature_log_prob = {}
        self.classes = []
        self.fitted = False

    def fit(self, X, y):
        """
        Train NB classifier.
        X: numpy array (n_samples, n_features) -- count vectors
        y: list of class labels
        """
        self.classes = sorted(set(y))
        n_samples = len(y)
        n_features = X.shape[1]

        for c in self.classes:
            # Class prior
            class_mask = np.array([1.0 if yi == c else 0.0 for yi in y])
            n_c = class_mask.sum()
            self.class_log_prior[c] = math.log(n_c / n_samples)

            # Feature likelihoods with Laplace smoothing
            class_X = X[class_mask.astype(bool)]
            feature_counts = class_X.sum(axis=0) + self.alpha
            total_count = feature_counts.sum()
            self.feature_log_prob[c] = np.log(feature_counts / total_count)

        self.fitted = True
        return self

    def predict(self, X):
        """Predict class labels."""
        predictions = []
        for i in range(X.shape[0]):
            scores = {}
            for c in self.classes:
                score = self.class_log_prior[c]
                score += (X[i] * self.feature_log_prob[c]).sum()
                scores[c] = score
            predictions.append(max(scores, key=scores.get))
        return predictions

    def predict_proba(self, X):
        """Predict class probabilities."""
        all_probs = []
        for i in range(X.shape[0]):
            log_scores = []
            for c in self.classes:
                score = self.class_log_prior[c]
                score += (X[i] * self.feature_log_prob[c]).sum()
                log_scores.append(score)
            # Softmax over log scores
            max_score = max(log_scores)
            exp_scores = [math.exp(s - max_score) for s in log_scores]
            total = sum(exp_scores)
            probs = {c: e / total for c, e in zip(self.classes, exp_scores)}
            all_probs.append(probs)
        return all_probs


class LogisticRegression:
    """Logistic regression using gradient descent. Supports binary and multiclass."""

    def __init__(self, lr=0.01, epochs=100, regularization=0.0, seed=42):
        self.lr = lr
        self.epochs = epochs
        self.regularization = regularization
        self.seed = seed
        self.weights = None
        self.bias = None
        self.classes = []
        self.fitted = False
        self.loss_history = []

    def _softmax(self, z):
        """Numerically stable softmax."""
        shifted = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        """Train logistic regression."""
        rng = np.random.RandomState(self.seed)
        self.classes = sorted(set(y))
        n_samples, n_features = X.shape
        n_classes = len(self.classes)
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        y_idx = np.array([class_to_idx[yi] for yi in y])

        if n_classes == 2:
            # Binary logistic regression
            self.weights = rng.randn(n_features) * 0.01
            self.bias = 0.0
            self.loss_history = []

            for epoch in range(self.epochs):
                z = X @ self.weights + self.bias
                pred = self._sigmoid(z)

                # Binary cross-entropy loss
                eps = 1e-15
                loss = -np.mean(y_idx * np.log(pred + eps) + (1 - y_idx) * np.log(1 - pred + eps))
                loss += 0.5 * self.regularization * np.sum(self.weights ** 2)
                self.loss_history.append(loss)

                # Gradients
                error = pred - y_idx
                grad_w = (X.T @ error) / n_samples + self.regularization * self.weights
                grad_b = error.mean()

                self.weights -= self.lr * grad_w
                self.bias -= self.lr * grad_b
        else:
            # Multiclass via softmax
            self.weights = rng.randn(n_features, n_classes) * 0.01
            self.bias = np.zeros(n_classes)
            self.loss_history = []

            # One-hot encode targets
            y_onehot = np.zeros((n_samples, n_classes))
            for i, yi in enumerate(y_idx):
                y_onehot[i, yi] = 1.0

            for epoch in range(self.epochs):
                z = X @ self.weights + self.bias
                pred = self._softmax(z)

                # Cross-entropy loss
                eps = 1e-15
                loss = -np.mean(np.sum(y_onehot * np.log(pred + eps), axis=1))
                loss += 0.5 * self.regularization * np.sum(self.weights ** 2)
                self.loss_history.append(loss)

                # Gradients
                error = pred - y_onehot
                grad_w = (X.T @ error) / n_samples + self.regularization * self.weights
                grad_b = error.mean(axis=0)

                self.weights -= self.lr * grad_w
                self.bias -= self.lr * grad_b

        self.fitted = True
        return self

    def predict(self, X):
        """Predict class labels."""
        if len(self.classes) == 2:
            z = X @ self.weights + self.bias
            pred = self._sigmoid(z)
            idx = (pred >= 0.5).astype(int)
            return [self.classes[i] for i in idx]
        else:
            z = X @ self.weights + self.bias
            pred = self._softmax(z)
            idx = pred.argmax(axis=1)
            return [self.classes[i] for i in idx]

    def predict_proba(self, X):
        """Predict class probabilities."""
        if len(self.classes) == 2:
            z = X @ self.weights + self.bias
            p = self._sigmoid(z)
            return [{self.classes[0]: 1 - pi, self.classes[1]: pi} for pi in p]
        else:
            z = X @ self.weights + self.bias
            probs = self._softmax(z)
            return [{c: probs[i, j] for j, c in enumerate(self.classes)}
                    for i in range(probs.shape[0])]


class MLPClassifier:
    """Multi-layer perceptron classifier using C140 Sequential model."""

    def __init__(self, hidden_sizes=(64, 32), activation='relu',
                 lr=0.01, epochs=100, batch_size=32, dropout=0.0, seed=42):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.seed = seed
        self.model = None
        self.classes = []
        self.fitted = False
        self.loss_history = []

    def fit(self, X, y):
        """Train MLP classifier."""
        self.classes = sorted(set(y))
        n_classes = len(self.classes)
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        n_features = X.shape[1]

        # Build model
        self.model = Sequential()
        prev_size = n_features
        rng_counter = self.seed
        for hs in self.hidden_sizes:
            rng_obj = random.Random(rng_counter)
            self.model.add(Dense(prev_size, hs, init='he', rng=rng_obj))
            self.model.add(Activation(self.activation))
            if self.dropout > 0:
                self.model.add(Dropout(self.dropout, rng=random.Random(rng_counter + 100)))
            prev_size = hs
            rng_counter += 1

        self.model.add(Dense(prev_size, n_classes, init='xavier', rng=random.Random(rng_counter)))

        # Prepare data as Tensors
        y_idx = [class_to_idx[yi] for yi in y]
        optimizer = Adam(lr=self.lr)
        loss_fn = CrossEntropyLoss()

        self.loss_history = []
        rng = np.random.RandomState(self.seed)

        for epoch in range(self.epochs):
            # Shuffle
            indices = rng.permutation(len(y_idx))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(y_idx), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                x_batch = Tensor(X[batch_idx].tolist())
                y_batch = [y_idx[i] for i in batch_idx]

                self.model.train()
                output = self.model.forward(x_batch)
                loss = loss_fn.forward(output, y_batch)
                grad = loss_fn.backward(output, y_batch)
                self.model.backward(grad)
                optimizer.step(self.model.get_trainable_layers())

                epoch_loss += loss
                n_batches += 1

            self.loss_history.append(epoch_loss / max(n_batches, 1))

        self.fitted = True
        return self

    def predict(self, X):
        """Predict class labels."""
        self.model.eval()
        x_tensor = Tensor(X.tolist())
        output = self.model.forward(x_tensor)
        # Get argmax per sample
        predictions = []
        if len(output.shape) == 1:
            # Single sample
            vals = output.data
            idx = vals.index(max(vals))
            predictions.append(self.classes[idx])
        else:
            for i in range(output.shape[0]):
                row = output.data[i]
                idx = row.index(max(row))
                predictions.append(self.classes[idx])
        return predictions

    def predict_proba(self, X):
        """Predict class probabilities via softmax."""
        self.model.eval()
        x_tensor = Tensor(X.tolist())
        output = self.model.forward(x_tensor)

        def softmax_row(row):
            max_val = max(row)
            exp_vals = [math.exp(v - max_val) for v in row]
            total = sum(exp_vals)
            return [e / total for e in exp_vals]

        results = []
        if len(output.shape) == 1:
            probs = softmax_row(output.data)
            results.append({c: p for c, p in zip(self.classes, probs)})
        else:
            for i in range(output.shape[0]):
                probs = softmax_row(output.data[i])
                results.append({c: p for c, p in zip(self.classes, probs)})
        return results


class RNNClassifier:
    """RNN-based text classifier using C144 LSTM/GRU."""

    def __init__(self, vocab_size, embed_dim=32, hidden_size=32,
                 cell_type='lstm', lr=0.01, epochs=20, batch_size=16,
                 clip_grad=5.0, seed=42):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.clip_grad = clip_grad
        self.seed = seed
        self.embedding = None
        self.model = None
        self.classes = []
        self.fitted = False
        self.loss_history = []

    def _build_model(self, n_classes):
        """Build the RNN model."""
        rng_obj = random.Random(self.seed)
        self.embedding = Embedding(self.vocab_size, self.embed_dim, rng=rng_obj)

        layer_cls = LSTMLayer if self.cell_type == 'lstm' else GRULayer
        rng_obj2 = random.Random(self.seed + 1)
        self.model = SequenceModel()
        self.model.add(layer_cls(self.embed_dim, self.hidden_size,
                                 return_sequences=False, clip_grad=self.clip_grad,
                                 rng=rng_obj2))
        # Output projection
        rng_obj3 = random.Random(self.seed + 2)
        self.output_layer = TimeDistributed(self.hidden_size, n_classes, rng=rng_obj3)

    def fit(self, X_sequences, y):
        """
        Train RNN classifier.
        X_sequences: list of integer sequences (already encoded)
        y: list of class labels
        """
        self.classes = sorted(set(y))
        n_classes = len(self.classes)
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        y_idx = [class_to_idx[yi] for yi in y]

        self._build_model(n_classes)

        rng = np.random.RandomState(self.seed)
        self.loss_history = []

        for epoch in range(self.epochs):
            indices = rng.permutation(len(y_idx))
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(y_idx), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]

                # Get batch sequences and targets
                batch_seqs = [X_sequences[i] for i in batch_idx]
                batch_y = [y_idx[i] for i in batch_idx]

                # Embed
                embedded = self.embedding.forward(batch_seqs)

                # Forward through RNN (returns last hidden state per sequence)
                hidden_states = self.model.forward(embedded)
                # hidden_states is list of vectors (one per sequence)

                # Project to class logits
                # TimeDistributed expects sequences, wrap each hidden state
                wrapped = [[h] for h in hidden_states]
                logits_wrapped = self.output_layer.forward(wrapped)
                logits = [lw[0] for lw in logits_wrapped]  # unwrap

                # Compute cross-entropy loss
                loss = 0.0
                d_logits = []
                for j, (logit, target) in enumerate(zip(logits, batch_y)):
                    # Softmax
                    max_l = max(logit)
                    exp_l = [math.exp(l - max_l) for l in logit]
                    total = sum(exp_l)
                    probs = [e / total for e in exp_l]

                    # Cross-entropy
                    loss -= math.log(max(probs[target], 1e-15))

                    # Gradient
                    grad = list(probs)
                    grad[target] -= 1.0
                    d_logits.append(grad)

                loss /= len(batch_y)
                epoch_loss += loss

                # Scale gradients
                d_logits = [[g / len(batch_y) for g in gl] for gl in d_logits]

                # Backward through output layer
                d_logits_wrapped = [[dl] for dl in d_logits]
                d_hidden, out_grads = self.output_layer.backward(d_logits_wrapped)
                d_hidden_unwrapped = [dh[0] for dh in d_hidden]

                # Backward through RNN
                d_inputs, rnn_grads = self.model.layers[0].backward(d_hidden_unwrapped)

                # Backward through embedding
                embedded_flat = []
                for seq in embedded:
                    for vec in seq:
                        embedded_flat.append(vec)
                emb_grads = self.embedding.backward(
                    [d_inputs[i] if i < len(d_inputs) else [[0.0]*self.embed_dim]
                     for i in range(len(batch_seqs))]
                )

                # Update parameters
                self.output_layer.update(out_grads, self.lr)
                self.model.layers[0].update(rnn_grads, self.lr)
                self.embedding.update(emb_grads, self.lr)

                n_batches += 1

            self.loss_history.append(epoch_loss / max(n_batches, 1))

        self.fitted = True
        return self

    def predict(self, X_sequences):
        """Predict class labels for integer sequences."""
        embedded = self.embedding.forward(X_sequences)
        # Set training mode off
        for layer in self.model.layers:
            layer.training = False
        hidden_states = self.model.forward(embedded)
        wrapped = [[h] for h in hidden_states]
        logits_wrapped = self.output_layer.forward(wrapped)
        logits = [lw[0] for lw in logits_wrapped]

        predictions = []
        for logit in logits:
            idx = logit.index(max(logit))
            predictions.append(self.classes[idx])
        return predictions

    def predict_proba(self, X_sequences):
        """Predict class probabilities."""
        embedded = self.embedding.forward(X_sequences)
        for layer in self.model.layers:
            layer.training = False
        hidden_states = self.model.forward(embedded)
        wrapped = [[h] for h in hidden_states]
        logits_wrapped = self.output_layer.forward(wrapped)
        logits = [lw[0] for lw in logits_wrapped]

        results = []
        for logit in logits:
            max_l = max(logit)
            exp_l = [math.exp(l - max_l) for l in logit]
            total = sum(exp_l)
            probs = {c: e / total for c, e in zip(self.classes, exp_l)}
            results.append(probs)
        return results


class EmbeddingClassifier:
    """Classifier using C147 sentence embeddings + a simple classifier head."""

    def __init__(self, embedding_space, pooling='mean', classifier='logistic',
                 lr=0.01, epochs=100, seed=42):
        """
        embedding_space: EmbeddingSpace from C147
        pooling: 'mean', 'max', 'sum'
        classifier: 'logistic' or 'nb' or 'mlp'
        """
        self.space = embedding_space
        self.pooling = pooling
        self.classifier_type = classifier
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.classifier = None
        self.fitted = False

    def _vectorize(self, texts):
        """Convert texts to sentence vectors using embedding space."""
        vectors = []
        for text in texts:
            tokens = tokenize(text)
            vec = self.space.sentence_vector(tokens, method=self.pooling)
            if vec is None:
                # No known tokens -- zero vector
                vec = np.zeros(self.space.embeddings.shape[1])
            vectors.append(vec)
        return np.array(vectors)

    def fit(self, texts, y):
        """Train on texts using sentence embeddings."""
        X = self._vectorize(texts)

        if self.classifier_type == 'logistic':
            self.classifier = LogisticRegression(lr=self.lr, epochs=self.epochs,
                                                  seed=self.seed)
        elif self.classifier_type == 'nb':
            # NB needs non-negative features -- shift
            self._shift = X.min()
            X = X - self._shift
            self.classifier = NaiveBayesClassifier()
        elif self.classifier_type == 'mlp':
            self.classifier = MLPClassifier(hidden_sizes=(64,), lr=self.lr,
                                             epochs=self.epochs, seed=self.seed)
        else:
            raise ValueError(f"Unknown classifier: {self.classifier_type}")

        self.classifier.fit(X, y)
        self.fitted = True
        return self

    def predict(self, texts):
        X = self._vectorize(texts)
        if self.classifier_type == 'nb':
            X = X - self._shift
        return self.classifier.predict(X)

    def predict_proba(self, texts):
        X = self._vectorize(texts)
        if self.classifier_type == 'nb':
            X = X - self._shift
        return self.classifier.predict_proba(X)


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

class ClassificationMetrics:
    """Compute classification evaluation metrics."""

    @staticmethod
    def accuracy(y_true, y_pred):
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_true) if y_true else 0.0

    @staticmethod
    def confusion_matrix(y_true, y_pred, labels=None):
        """Return confusion matrix as dict of dicts: {true_label: {pred_label: count}}."""
        if labels is None:
            labels = sorted(set(y_true) | set(y_pred))
        matrix = {t: {p: 0 for p in labels} for t in labels}
        for t, p in zip(y_true, y_pred):
            matrix[t][p] += 1
        return matrix

    @staticmethod
    def precision(y_true, y_pred, label=None, average='macro'):
        """Compute precision. For multiclass, use average='macro' or 'micro'."""
        labels = sorted(set(y_true) | set(y_pred))

        if label is not None:
            tp = sum(1 for t, p in zip(y_true, y_pred) if p == label and t == label)
            fp = sum(1 for t, p in zip(y_true, y_pred) if p == label and t != label)
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if average == 'micro':
            tp_total = sum(1 for t, p in zip(y_true, y_pred) if t == p)
            return tp_total / len(y_pred) if y_pred else 0.0

        # Macro
        precisions = []
        for c in labels:
            tp = sum(1 for t, p in zip(y_true, y_pred) if p == c and t == c)
            fp = sum(1 for t, p in zip(y_true, y_pred) if p == c and t != c)
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        return sum(precisions) / len(precisions) if precisions else 0.0

    @staticmethod
    def recall(y_true, y_pred, label=None, average='macro'):
        """Compute recall."""
        labels = sorted(set(y_true) | set(y_pred))

        if label is not None:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if average == 'micro':
            tp_total = sum(1 for t, p in zip(y_true, y_pred) if t == p)
            return tp_total / len(y_true) if y_true else 0.0

        recalls = []
        for c in labels:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        return sum(recalls) / len(recalls) if recalls else 0.0

    @staticmethod
    def f1_score(y_true, y_pred, label=None, average='macro'):
        """Compute F1 score."""
        p = ClassificationMetrics.precision(y_true, y_pred, label=label, average=average)
        r = ClassificationMetrics.recall(y_true, y_pred, label=label, average=average)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @staticmethod
    def classification_report(y_true, y_pred, labels=None):
        """Generate classification report as dict."""
        if labels is None:
            labels = sorted(set(y_true) | set(y_pred))

        report = {}
        for c in labels:
            p = ClassificationMetrics.precision(y_true, y_pred, label=c)
            r = ClassificationMetrics.recall(y_true, y_pred, label=c)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            support = sum(1 for t in y_true if t == c)
            report[c] = {'precision': p, 'recall': r, 'f1': f1, 'support': support}

        # Add averages
        report['macro'] = {
            'precision': ClassificationMetrics.precision(y_true, y_pred, average='macro'),
            'recall': ClassificationMetrics.recall(y_true, y_pred, average='macro'),
            'f1': ClassificationMetrics.f1_score(y_true, y_pred, average='macro'),
            'support': len(y_true)
        }
        report['accuracy'] = ClassificationMetrics.accuracy(y_true, y_pred)
        return report


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TextClassificationPipeline:
    """End-to-end text classification pipeline."""

    def __init__(self, vectorizer=None, classifier=None):
        """
        vectorizer: BagOfWords, TfIdf, or None (uses BagOfWords by default)
        classifier: any classifier with fit/predict interface
        """
        self.vectorizer = vectorizer or BagOfWords()
        self.classifier = classifier or NaiveBayesClassifier()

    def fit(self, texts, y):
        """Fit vectorizer and classifier on training texts."""
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, y)
        return self

    def predict(self, texts):
        """Predict labels for new texts."""
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)

    def predict_proba(self, texts):
        """Predict probabilities for new texts."""
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def evaluate(self, texts, y_true):
        """Evaluate pipeline and return metrics."""
        y_pred = self.predict(texts)
        return ClassificationMetrics.classification_report(y_true, y_pred)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def train_test_split(texts, labels, test_ratio=0.2, seed=42):
    """Split texts and labels into train/test sets."""
    rng = np.random.RandomState(seed)
    n = len(texts)
    indices = rng.permutation(n)
    split = int(n * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return (
        [texts[i] for i in train_idx],
        [texts[i] for i in test_idx],
        [labels[i] for i in train_idx],
        [labels[i] for i in test_idx]
    )


def make_sentiment_data(n_per_class=50, seed=42):
    """Generate synthetic sentiment data for testing."""
    rng = np.random.RandomState(seed)

    positive_words = ['good', 'great', 'excellent', 'wonderful', 'fantastic',
                      'amazing', 'love', 'best', 'happy', 'beautiful',
                      'perfect', 'enjoyed', 'brilliant', 'superb', 'nice']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst',
                      'hate', 'boring', 'poor', 'disappointing', 'ugly',
                      'waste', 'annoying', 'stupid', 'dreadful', 'mediocre']
    neutral_words = ['the', 'a', 'is', 'it', 'was', 'this', 'that', 'movie',
                     'film', 'book', 'story', 'product', 'thing', 'really', 'very']

    texts = []
    labels = []

    for _ in range(n_per_class):
        # Positive
        n_words = rng.randint(5, 15)
        n_pos = rng.randint(2, min(5, n_words))
        words = list(rng.choice(positive_words, n_pos))
        words += list(rng.choice(neutral_words, n_words - n_pos))
        rng.shuffle(words)
        texts.append(' '.join(words))
        labels.append('positive')

    for _ in range(n_per_class):
        # Negative
        n_words = rng.randint(5, 15)
        n_neg = rng.randint(2, min(5, n_words))
        words = list(rng.choice(negative_words, n_neg))
        words += list(rng.choice(neutral_words, n_words - n_neg))
        rng.shuffle(words)
        texts.append(' '.join(words))
        labels.append('negative')

    return texts, labels


def make_topic_data(n_per_class=30, seed=42):
    """Generate synthetic topic classification data."""
    rng = np.random.RandomState(seed)

    topic_words = {
        'sports': ['game', 'team', 'player', 'score', 'win', 'championship',
                   'coach', 'season', 'match', 'goal', 'tournament', 'athlete'],
        'tech': ['computer', 'software', 'algorithm', 'data', 'network', 'code',
                 'program', 'digital', 'system', 'device', 'processor', 'memory'],
        'science': ['experiment', 'hypothesis', 'research', 'theory', 'molecule',
                    'cell', 'energy', 'physics', 'biology', 'chemistry', 'lab', 'study']
    }
    filler = ['the', 'a', 'is', 'was', 'in', 'of', 'and', 'to', 'for', 'with']

    texts = []
    labels = []

    for topic, words in topic_words.items():
        for _ in range(n_per_class):
            n_words = rng.randint(6, 12)
            n_topic = rng.randint(2, min(5, n_words))
            sent = list(rng.choice(words, n_topic))
            sent += list(rng.choice(filler, n_words - n_topic))
            rng.shuffle(sent)
            texts.append(' '.join(sent))
            labels.append(topic)

    return texts, labels


def cross_validate(pipeline_factory, texts, labels, n_folds=5, seed=42):
    """K-fold cross-validation for a pipeline factory function.

    pipeline_factory: callable that returns a fresh TextClassificationPipeline
    Returns: dict with mean/std of metrics across folds
    """
    rng = np.random.RandomState(seed)
    n = len(texts)
    indices = rng.permutation(n)
    fold_size = n // n_folds

    fold_metrics = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        pipeline = pipeline_factory()
        pipeline.fit(train_texts, train_labels)
        report = pipeline.evaluate(test_texts, test_labels)
        fold_metrics.append(report)

    # Aggregate
    accuracies = [fm['accuracy'] for fm in fold_metrics]
    result = {
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'fold_accuracies': accuracies,
        'fold_reports': fold_metrics
    }
    return result
