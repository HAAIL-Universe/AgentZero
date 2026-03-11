"""
C147: Word Embeddings -- Word2Vec (Skip-gram, CBOW) + GloVe + FastText

Composes with C144 (RNN) concepts. Implements from scratch using NumPy only.

Components:
- Vocabulary builder with frequency-based filtering
- Skip-gram with negative sampling (SGNS)
- CBOW with negative sampling
- GloVe (Global Vectors for Word Representation)
- FastText (subword embeddings)
- Analogy solver (king - man + woman = queen)
- Similarity search (cosine, euclidean)
- Embedding arithmetic and visualization helpers
"""

import numpy as np
from collections import Counter
import re
import math
import struct
import json


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    """Builds and manages word-to-index mapping with frequency filtering."""

    def __init__(self, min_count=1, max_vocab_size=None):
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.word2idx = {}
        self.idx2word = []
        self.word_counts = Counter()
        self.total_words = 0

    def build(self, corpus):
        """Build vocabulary from tokenized corpus (list of list of tokens)."""
        self.word_counts = Counter()
        for sentence in corpus:
            self.word_counts.update(sentence)

        # Filter by min_count
        filtered = {w: c for w, c in self.word_counts.items() if c >= self.min_count}

        # Sort by frequency (descending), then alphabetically for ties
        sorted_words = sorted(filtered.keys(), key=lambda w: (-filtered[w], w))

        # Limit vocab size
        if self.max_vocab_size is not None:
            sorted_words = sorted_words[:self.max_vocab_size]

        self.idx2word = sorted_words
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        self.total_words = sum(filtered[w] for w in self.idx2word)
        return self

    def __len__(self):
        return len(self.idx2word)

    def __contains__(self, word):
        return word in self.word2idx

    def encode(self, word):
        """Word to index."""
        return self.word2idx.get(word, None)

    def decode(self, idx):
        """Index to word."""
        if 0 <= idx < len(self.idx2word):
            return self.idx2word[idx]
        return None

    def get_frequency(self, word):
        """Get word frequency count."""
        return self.word_counts.get(word, 0)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(text, lower=True):
    """Simple whitespace + punctuation tokenizer."""
    if lower:
        text = text.lower()
    # Split on non-alphanumeric, keep words
    tokens = re.findall(r"[a-zA-Z0-9]+(?:'[a-zA-Z]+)?", text)
    return tokens


def build_corpus(texts, lower=True):
    """Tokenize a list of texts into a corpus (list of token lists)."""
    return [tokenize(t, lower=lower) for t in texts]


# ---------------------------------------------------------------------------
# Subsampling (frequent word downsampling)
# ---------------------------------------------------------------------------

def subsample_prob(word_freq, total, threshold=1e-5):
    """Probability of keeping a word (Mikolov subsampling formula)."""
    f = word_freq / total
    if f == 0:
        return 0.0
    p = 1.0 - math.sqrt(threshold / f)
    return max(0.0, p)


# ---------------------------------------------------------------------------
# Negative Sampling Distribution
# ---------------------------------------------------------------------------

class NegativeSampler:
    """Unigram distribution raised to 3/4 power for negative sampling."""

    def __init__(self, vocab, power=0.75):
        self.vocab_size = len(vocab)
        counts = np.array([vocab.word_counts.get(vocab.idx2word[i], 1)
                           for i in range(self.vocab_size)], dtype=np.float64)
        counts = np.power(counts, power)
        self.probs = counts / counts.sum()
        self._table = None
        self._table_size = 0

    def sample(self, n, exclude=None, rng=None):
        """Sample n negative indices."""
        if rng is None:
            rng = np.random.default_rng()
        # Check if enough candidates exist
        available = self.vocab_size - (len(exclude) if exclude else 0)
        if available <= 0:
            return []
        n = min(n, available)
        samples = []
        max_attempts = n * 20
        attempts = 0
        while len(samples) < n and attempts < max_attempts:
            batch = rng.choice(self.vocab_size, size=n * 2, p=self.probs)
            for idx in batch:
                if exclude is not None and idx in exclude:
                    continue
                samples.append(int(idx))
                if len(samples) >= n:
                    break
            attempts += 1
        return samples[:n]


# ---------------------------------------------------------------------------
# Training Data Generators
# ---------------------------------------------------------------------------

def generate_skipgram_pairs(corpus, vocab, window_size=5, rng=None):
    """Generate (center, context) pairs for skip-gram."""
    if rng is None:
        rng = np.random.default_rng()
    pairs = []
    for sentence in corpus:
        indices = [vocab.encode(w) for w in sentence if w in vocab]
        for i, center in enumerate(indices):
            # Dynamic window: sample window from [1, window_size]
            actual_window = rng.integers(1, window_size + 1)
            start = max(0, i - actual_window)
            end = min(len(indices), i + actual_window + 1)
            for j in range(start, end):
                if j != i:
                    pairs.append((center, indices[j]))
    return pairs


def generate_cbow_pairs(corpus, vocab, window_size=5, rng=None):
    """Generate (context_list, center) pairs for CBOW."""
    if rng is None:
        rng = np.random.default_rng()
    pairs = []
    for sentence in corpus:
        indices = [vocab.encode(w) for w in sentence if w in vocab]
        for i, center in enumerate(indices):
            actual_window = rng.integers(1, window_size + 1)
            start = max(0, i - actual_window)
            end = min(len(indices), i + actual_window + 1)
            context = [indices[j] for j in range(start, end) if j != i]
            if context:
                pairs.append((context, center))
    return pairs


# ---------------------------------------------------------------------------
# Skip-gram with Negative Sampling (SGNS)
# ---------------------------------------------------------------------------

class SkipGram:
    """Skip-gram model with negative sampling."""

    def __init__(self, vocab_size, embedding_dim, learning_rate=0.025,
                 neg_samples=5, seed=42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate
        self.neg_samples = neg_samples
        self.rng = np.random.default_rng(seed)

        # Initialize embeddings
        scale = 0.5 / embedding_dim
        self.W_in = self.rng.uniform(-scale, scale, (vocab_size, embedding_dim))
        self.W_out = np.zeros((vocab_size, embedding_dim))

        self.training_loss = []

    def _sigmoid(self, x):
        x = np.clip(x, -10, 10)
        return 1.0 / (1.0 + np.exp(-x))

    def train_pair(self, center, context, neg_sampler):
        """Train on a single (center, context) pair with negative sampling."""
        # Positive sample
        v_c = self.W_in[center]  # center embedding
        u_o = self.W_out[context]  # context embedding

        score = np.dot(v_c, u_o)
        sig = self._sigmoid(score)
        grad_pos = (sig - 1.0) * v_c  # gradient for context
        grad_center = (sig - 1.0) * u_o  # gradient for center

        loss = -np.log(self._sigmoid(score) + 1e-10)

        # Update context (positive)
        self.W_out[context] -= self.lr * grad_pos

        # Negative samples
        neg_indices = neg_sampler.sample(self.neg_samples,
                                         exclude={center, context},
                                         rng=self.rng)
        for neg_idx in neg_indices:
            u_neg = self.W_out[neg_idx]
            score_neg = np.dot(v_c, u_neg)
            sig_neg = self._sigmoid(score_neg)

            grad_neg = sig_neg * v_c
            grad_center += sig_neg * u_neg

            loss -= np.log(1.0 - self._sigmoid(score_neg) + 1e-10)

            self.W_out[neg_idx] -= self.lr * grad_neg

        # Update center
        self.W_in[center] -= self.lr * grad_center

        return loss

    def train(self, pairs, neg_sampler, epochs=1, lr_decay=True):
        """Train on all pairs."""
        initial_lr = self.lr
        total_pairs = len(pairs) * epochs
        processed = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            # Shuffle pairs
            order = self.rng.permutation(len(pairs))
            for idx in order:
                center, context = pairs[idx]
                if lr_decay:
                    self.lr = max(initial_lr * 0.0001,
                                  initial_lr * (1 - processed / total_pairs))
                loss = self.train_pair(center, context, neg_sampler)
                epoch_loss += loss
                processed += 1

            avg_loss = epoch_loss / len(pairs) if pairs else 0
            self.training_loss.append(avg_loss)

        self.lr = initial_lr
        return self.training_loss

    def get_embedding(self, idx):
        """Get word embedding vector."""
        return self.W_in[idx].copy()

    def get_embeddings(self):
        """Get all embeddings (input embeddings)."""
        return self.W_in.copy()


# ---------------------------------------------------------------------------
# CBOW with Negative Sampling
# ---------------------------------------------------------------------------

class CBOW:
    """Continuous Bag of Words model with negative sampling."""

    def __init__(self, vocab_size, embedding_dim, learning_rate=0.025,
                 neg_samples=5, seed=42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate
        self.neg_samples = neg_samples
        self.rng = np.random.default_rng(seed)

        scale = 0.5 / embedding_dim
        self.W_in = self.rng.uniform(-scale, scale, (vocab_size, embedding_dim))
        self.W_out = np.zeros((vocab_size, embedding_dim))

        self.training_loss = []

    def _sigmoid(self, x):
        x = np.clip(x, -10, 10)
        return 1.0 / (1.0 + np.exp(-x))

    def train_pair(self, context_indices, center, neg_sampler):
        """Train on a (context_list, center) pair."""
        # Average context embeddings
        v_ctx = np.mean(self.W_in[context_indices], axis=0)

        u_o = self.W_out[center]
        score = np.dot(v_ctx, u_o)
        sig = self._sigmoid(score)

        grad_out = (sig - 1.0) * v_ctx
        grad_ctx = (sig - 1.0) * u_o

        loss = -np.log(self._sigmoid(score) + 1e-10)

        self.W_out[center] -= self.lr * grad_out

        # Negative samples
        neg_indices = neg_sampler.sample(self.neg_samples,
                                         exclude=set(context_indices) | {center},
                                         rng=self.rng)
        for neg_idx in neg_indices:
            u_neg = self.W_out[neg_idx]
            score_neg = np.dot(v_ctx, u_neg)
            sig_neg = self._sigmoid(score_neg)

            self.W_out[neg_idx] -= self.lr * sig_neg * v_ctx
            grad_ctx += sig_neg * u_neg

            loss -= np.log(1.0 - self._sigmoid(score_neg) + 1e-10)

        # Distribute gradient to all context words equally
        grad_per_ctx = grad_ctx / len(context_indices)
        for ci in context_indices:
            self.W_in[ci] -= self.lr * grad_per_ctx

        return loss

    def train(self, pairs, neg_sampler, epochs=1, lr_decay=True):
        """Train on all pairs."""
        initial_lr = self.lr
        total_pairs = len(pairs) * epochs
        processed = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            order = self.rng.permutation(len(pairs))
            for idx in order:
                context, center = pairs[idx]
                if lr_decay:
                    self.lr = max(initial_lr * 0.0001,
                                  initial_lr * (1 - processed / total_pairs))
                loss = self.train_pair(context, center, neg_sampler)
                epoch_loss += loss
                processed += 1

            avg_loss = epoch_loss / len(pairs) if pairs else 0
            self.training_loss.append(avg_loss)

        self.lr = initial_lr
        return self.training_loss

    def get_embedding(self, idx):
        return self.W_in[idx].copy()

    def get_embeddings(self):
        return self.W_in.copy()


# ---------------------------------------------------------------------------
# GloVe (Global Vectors)
# ---------------------------------------------------------------------------

class GloVe:
    """GloVe: Global Vectors for Word Representation.

    Factorizes the log co-occurrence matrix using weighted least squares.
    """

    def __init__(self, vocab_size, embedding_dim, learning_rate=0.05,
                 x_max=100, alpha=0.75, seed=42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate
        self.x_max = x_max
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

        # Two sets of embeddings + biases (word and context)
        scale = 0.5 / embedding_dim
        self.W = self.rng.uniform(-scale, scale, (vocab_size, embedding_dim))
        self.W_ctx = self.rng.uniform(-scale, scale, (vocab_size, embedding_dim))
        self.b_w = np.zeros(vocab_size)
        self.b_ctx = np.zeros(vocab_size)

        # AdaGrad accumulators
        self.grad_sq_W = np.ones((vocab_size, embedding_dim))
        self.grad_sq_W_ctx = np.ones((vocab_size, embedding_dim))
        self.grad_sq_bw = np.ones(vocab_size)
        self.grad_sq_bctx = np.ones(vocab_size)

        self.training_loss = []

    def build_cooccurrence(self, corpus, vocab, window_size=10):
        """Build co-occurrence matrix from corpus."""
        cooccurrence = {}
        for sentence in corpus:
            indices = [vocab.encode(w) for w in sentence if w in vocab]
            for i, wi in enumerate(indices):
                start = max(0, i - window_size)
                end = min(len(indices), i + window_size + 1)
                for j in range(start, end):
                    if j != i:
                        wj = indices[j]
                        dist = abs(i - j)
                        weight = 1.0 / dist  # distance-weighted
                        key = (wi, wj)
                        cooccurrence[key] = cooccurrence.get(key, 0.0) + weight
        return cooccurrence

    def _weight_func(self, x):
        """Weighting function f(x) = (x/x_max)^alpha if x < x_max, else 1."""
        if x < self.x_max:
            return (x / self.x_max) ** self.alpha
        return 1.0

    def train(self, cooccurrence, epochs=25):
        """Train GloVe on co-occurrence data."""
        # Convert to list for shuffling
        entries = [(i, j, v) for (i, j), v in cooccurrence.items() if v > 0]

        for epoch in range(epochs):
            epoch_loss = 0.0
            order = self.rng.permutation(len(entries))

            for eidx in order:
                i, j, x_ij = entries[eidx]
                log_x = np.log(x_ij)
                weight = self._weight_func(x_ij)

                # Compute cost
                diff = np.dot(self.W[i], self.W_ctx[j]) + self.b_w[i] + self.b_ctx[j] - log_x
                fdiff = weight * diff
                loss = 0.5 * fdiff * diff
                epoch_loss += loss

                # Gradients
                grad_w = fdiff * self.W_ctx[j]
                grad_ctx = fdiff * self.W[i]
                grad_bw = fdiff
                grad_bctx = fdiff

                # AdaGrad updates
                self.grad_sq_W[i] += grad_w ** 2
                self.grad_sq_W_ctx[j] += grad_ctx ** 2
                self.grad_sq_bw[i] += grad_bw ** 2
                self.grad_sq_bctx[j] += grad_bctx ** 2

                self.W[i] -= self.lr * grad_w / np.sqrt(self.grad_sq_W[i])
                self.W_ctx[j] -= self.lr * grad_ctx / np.sqrt(self.grad_sq_W_ctx[j])
                self.b_w[i] -= self.lr * grad_bw / np.sqrt(self.grad_sq_bw[i])
                self.b_ctx[j] -= self.lr * grad_bctx / np.sqrt(self.grad_sq_bctx[j])

            avg_loss = epoch_loss / len(entries) if entries else 0
            self.training_loss.append(avg_loss)

        return self.training_loss

    def get_embeddings(self):
        """Get combined embeddings (W + W_ctx, as in original paper)."""
        return self.W + self.W_ctx

    def get_embedding(self, idx):
        return (self.W[idx] + self.W_ctx[idx]).copy()


# ---------------------------------------------------------------------------
# FastText (Subword Embeddings)
# ---------------------------------------------------------------------------

def get_ngrams(word, min_n=3, max_n=6):
    """Extract character n-grams with boundary markers."""
    padded = "<" + word + ">"
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(padded) - n + 1):
            ngrams.append(padded[i:i + n])
    return ngrams


def hash_ngram(ngram, bucket_size=2000000):
    """FNV-1a hash for n-gram to bucket index."""
    h = 2166136261
    for ch in ngram:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h % bucket_size


class FastText:
    """FastText model: word embeddings enriched with subword information."""

    def __init__(self, vocab_size, embedding_dim, bucket_size=50000,
                 min_n=3, max_n=6, learning_rate=0.025, neg_samples=5, seed=42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.bucket_size = bucket_size
        self.min_n = min_n
        self.max_n = max_n
        self.lr = learning_rate
        self.neg_samples = neg_samples
        self.rng = np.random.default_rng(seed)

        scale = 0.5 / embedding_dim
        # Word embeddings
        self.W_in = self.rng.uniform(-scale, scale, (vocab_size, embedding_dim))
        # Subword (n-gram bucket) embeddings
        self.W_ngram = np.zeros((bucket_size, embedding_dim))
        # Output embeddings
        self.W_out = np.zeros((vocab_size, embedding_dim))

        self.training_loss = []
        self._ngram_cache = {}

    def _get_ngram_indices(self, word):
        """Get bucket indices for all n-grams of a word."""
        if word in self._ngram_cache:
            return self._ngram_cache[word]
        ngrams = get_ngrams(word, self.min_n, self.max_n)
        indices = [hash_ngram(ng, self.bucket_size) for ng in ngrams]
        self._ngram_cache[word] = indices
        return indices

    def _get_word_vector(self, word_idx, word_str):
        """Get word vector = word embedding + avg of n-gram embeddings."""
        v = self.W_in[word_idx].copy()
        ngram_ids = self._get_ngram_indices(word_str)
        if ngram_ids:
            v += np.mean(self.W_ngram[ngram_ids], axis=0)
        return v

    def _sigmoid(self, x):
        x = np.clip(x, -10, 10)
        return 1.0 / (1.0 + np.exp(-x))

    def train_pair(self, center_idx, center_word, context_idx, neg_sampler):
        """Train on a single skip-gram pair with subword information."""
        v_c = self._get_word_vector(center_idx, center_word)
        u_o = self.W_out[context_idx]

        score = np.dot(v_c, u_o)
        sig = self._sigmoid(score)
        grad_center = (sig - 1.0) * u_o

        loss = -np.log(self._sigmoid(score) + 1e-10)

        self.W_out[context_idx] -= self.lr * (sig - 1.0) * v_c

        neg_indices = neg_sampler.sample(self.neg_samples,
                                         exclude={center_idx, context_idx},
                                         rng=self.rng)
        for neg_idx in neg_indices:
            u_neg = self.W_out[neg_idx]
            score_neg = np.dot(v_c, u_neg)
            sig_neg = self._sigmoid(score_neg)

            self.W_out[neg_idx] -= self.lr * sig_neg * v_c
            grad_center += sig_neg * u_neg

            loss -= np.log(1.0 - self._sigmoid(score_neg) + 1e-10)

        # Update word embedding
        self.W_in[center_idx] -= self.lr * grad_center

        # Update n-gram embeddings
        ngram_ids = self._get_ngram_indices(center_word)
        if ngram_ids:
            grad_per_ng = grad_center / len(ngram_ids)
            for ng_id in ngram_ids:
                self.W_ngram[ng_id] -= self.lr * grad_per_ng

        return loss

    def train(self, pairs, vocab, neg_sampler, epochs=1, lr_decay=True):
        """Train FastText. pairs = list of (center_idx, context_idx)."""
        initial_lr = self.lr
        total = len(pairs) * epochs
        processed = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            order = self.rng.permutation(len(pairs))
            for idx in order:
                center_idx, context_idx = pairs[idx]
                center_word = vocab.decode(center_idx)
                if lr_decay:
                    self.lr = max(initial_lr * 0.0001,
                                  initial_lr * (1 - processed / total))
                loss = self.train_pair(center_idx, center_word, context_idx,
                                       neg_sampler)
                epoch_loss += loss
                processed += 1

            avg_loss = epoch_loss / len(pairs) if pairs else 0
            self.training_loss.append(avg_loss)

        self.lr = initial_lr
        return self.training_loss

    def get_embedding(self, idx, word=None):
        """Get word embedding. If word provided, includes subword info."""
        if word is not None:
            return self._get_word_vector(idx, word)
        return self.W_in[idx].copy()

    def get_oov_embedding(self, word):
        """Get embedding for out-of-vocabulary word using subword n-grams."""
        ngram_ids = self._get_ngram_indices(word)
        if not ngram_ids:
            return np.zeros(self.embedding_dim)
        return np.mean(self.W_ngram[ngram_ids], axis=0)

    def get_embeddings(self):
        return self.W_in.copy()


# ---------------------------------------------------------------------------
# Embedding Operations
# ---------------------------------------------------------------------------

class EmbeddingSpace:
    """Operations on trained embeddings: similarity, analogy, clustering."""

    def __init__(self, embeddings, vocab):
        """
        Args:
            embeddings: numpy array (vocab_size, dim)
            vocab: Vocabulary object
        """
        self.embeddings = embeddings.copy()
        self.vocab = vocab
        # Precompute normalized embeddings
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self.normalized = self.embeddings / norms

    def get_vector(self, word):
        """Get embedding vector for a word."""
        idx = self.vocab.encode(word)
        if idx is None:
            return None
        return self.embeddings[idx].copy()

    def cosine_similarity(self, word1, word2):
        """Cosine similarity between two words."""
        idx1 = self.vocab.encode(word1)
        idx2 = self.vocab.encode(word2)
        if idx1 is None or idx2 is None:
            return None
        return float(np.dot(self.normalized[idx1], self.normalized[idx2]))

    def euclidean_distance(self, word1, word2):
        """Euclidean distance between two words."""
        v1 = self.get_vector(word1)
        v2 = self.get_vector(word2)
        if v1 is None or v2 is None:
            return None
        return float(np.linalg.norm(v1 - v2))

    def most_similar(self, word, top_n=10, exclude_self=True):
        """Find most similar words by cosine similarity."""
        idx = self.vocab.encode(word)
        if idx is None:
            return []
        vec = self.normalized[idx]
        scores = self.normalized @ vec
        # Sort descending
        ranked = np.argsort(-scores)
        results = []
        for r in ranked:
            if exclude_self and r == idx:
                continue
            results.append((self.vocab.decode(int(r)), float(scores[r])))
            if len(results) >= top_n:
                break
        return results

    def most_similar_to_vector(self, vec, top_n=10, exclude=None):
        """Find words most similar to a given vector."""
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return []
        nvec = vec / norm
        scores = self.normalized @ nvec
        ranked = np.argsort(-scores)
        exclude_set = set()
        if exclude:
            for w in exclude:
                eidx = self.vocab.encode(w)
                if eidx is not None:
                    exclude_set.add(eidx)
        results = []
        for r in ranked:
            if int(r) in exclude_set:
                continue
            results.append((self.vocab.decode(int(r)), float(scores[r])))
            if len(results) >= top_n:
                break
        return results

    def analogy(self, a, b, c, top_n=5):
        """Solve analogy: a is to b as c is to ?

        Computes b - a + c and finds nearest words.
        Example: king - man + woman = queen
        """
        va = self.get_vector(a)
        vb = self.get_vector(b)
        vc = self.get_vector(c)
        if va is None or vb is None or vc is None:
            return []
        target = vb - va + vc
        return self.most_similar_to_vector(target, top_n=top_n,
                                            exclude=[a, b, c])

    def doesnt_match(self, words):
        """Find the word that doesn't match the others (odd one out)."""
        vecs = []
        valid_words = []
        for w in words:
            v = self.get_vector(w)
            if v is not None:
                vecs.append(v)
                valid_words.append(w)
        if len(vecs) < 2:
            return None
        vecs = np.array(vecs)
        mean = np.mean(vecs, axis=0)
        mean_norm = np.linalg.norm(mean)
        if mean_norm < 1e-10:
            return valid_words[0]
        mean_normalized = mean / mean_norm
        sims = []
        for i, v in enumerate(vecs):
            n = np.linalg.norm(v)
            if n < 1e-10:
                sims.append(-1.0)
            else:
                sims.append(float(np.dot(v / n, mean_normalized)))
        min_idx = int(np.argmin(sims))
        return valid_words[min_idx]

    def k_nearest_neighbors(self, word, k=5):
        """KNN by euclidean distance."""
        idx = self.vocab.encode(word)
        if idx is None:
            return []
        vec = self.embeddings[idx]
        dists = np.linalg.norm(self.embeddings - vec, axis=1)
        ranked = np.argsort(dists)
        results = []
        for r in ranked:
            if r == idx:
                continue
            results.append((self.vocab.decode(int(r)), float(dists[r])))
            if len(results) >= k:
                break
        return results

    def cluster_words(self, words, n_clusters=3, max_iter=50):
        """Simple k-means clustering of words."""
        vecs = []
        valid_words = []
        for w in words:
            v = self.get_vector(w)
            if v is not None:
                vecs.append(v)
                valid_words.append(w)
        if len(vecs) < n_clusters:
            return {i: [w] for i, w in enumerate(valid_words)}

        vecs = np.array(vecs)
        rng = np.random.default_rng(42)

        # K-means++
        centroids = [vecs[rng.integers(len(vecs))]]
        for _ in range(1, n_clusters):
            dists = np.array([min(np.linalg.norm(v - c) ** 2 for c in centroids)
                              for v in vecs])
            probs = dists / (dists.sum() + 1e-10)
            centroids.append(vecs[rng.choice(len(vecs), p=probs)])
        centroids = np.array(centroids)

        for _ in range(max_iter):
            # Assign
            assignments = np.array([np.argmin(np.linalg.norm(centroids - v, axis=1))
                                     for v in vecs])
            # Update
            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                mask = assignments == k
                if mask.any():
                    new_centroids[k] = vecs[mask].mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        result = {}
        for k in range(n_clusters):
            mask = assignments == k
            result[k] = [valid_words[i] for i in range(len(valid_words)) if mask[i]]
        return result

    def word_mover_distance(self, tokens1, tokens2):
        """Simplified Word Mover's Distance (greedy approximation).

        For each word in tokens1, finds closest word in tokens2 and averages distances.
        """
        vecs1 = [(w, self.get_vector(w)) for w in tokens1 if self.get_vector(w) is not None]
        vecs2 = [(w, self.get_vector(w)) for w in tokens2 if self.get_vector(w) is not None]

        if not vecs1 or not vecs2:
            return float('inf')

        # Forward: for each word in s1, find closest in s2
        fwd = 0.0
        for _, v1 in vecs1:
            dists = [np.linalg.norm(v1 - v2) for _, v2 in vecs2]
            fwd += min(dists)

        # Backward
        bwd = 0.0
        for _, v2 in vecs2:
            dists = [np.linalg.norm(v1 - v2) for _, v1 in vecs1]
            bwd += min(dists)

        return (fwd / len(vecs1) + bwd / len(vecs2)) / 2.0

    def sentence_vector(self, tokens, method='mean'):
        """Compute sentence embedding from token list."""
        vecs = [self.get_vector(w) for w in tokens if self.get_vector(w) is not None]
        if not vecs:
            return np.zeros(self.embeddings.shape[1])
        vecs = np.array(vecs)
        if method == 'mean':
            return np.mean(vecs, axis=0)
        elif method == 'max':
            return np.max(vecs, axis=0)
        elif method == 'sum':
            return np.sum(vecs, axis=0)
        return np.mean(vecs, axis=0)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_embeddings(filepath, embeddings, vocab, binary=False):
    """Save embeddings in word2vec text or binary format."""
    n, dim = embeddings.shape
    if binary:
        with open(filepath, 'wb') as f:
            header = f"{n} {dim}\n".encode('utf-8')
            f.write(header)
            for i in range(n):
                word = vocab.idx2word[i]
                f.write(word.encode('utf-8') + b' ')
                f.write(embeddings[i].astype(np.float32).tobytes())
                f.write(b'\n')
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{n} {dim}\n")
            for i in range(n):
                word = vocab.idx2word[i]
                vec_str = ' '.join(f"{v:.6f}" for v in embeddings[i])
                f.write(f"{word} {vec_str}\n")


def load_embeddings(filepath, binary=False):
    """Load embeddings from word2vec text or binary format."""
    vocab = Vocabulary()
    if binary:
        with open(filepath, 'rb') as f:
            header = f.readline().decode('utf-8').strip()
            n, dim = map(int, header.split())
            embeddings = np.zeros((n, dim), dtype=np.float32)
            for i in range(n):
                word_bytes = b''
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    word_bytes += ch
                word = word_bytes.decode('utf-8')
                vec = np.frombuffer(f.read(4 * dim), dtype=np.float32)
                embeddings[i] = vec
                f.read(1)  # newline
                vocab.idx2word.append(word)
                vocab.word2idx[word] = i
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            n, dim = map(int, header.split())
            embeddings = np.zeros((n, dim), dtype=np.float32)
            for i in range(n):
                parts = f.readline().strip().split()
                word = parts[0]
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                embeddings[i] = vec
                vocab.idx2word.append(word)
                vocab.word2idx[word] = i

    return embeddings, vocab


# ---------------------------------------------------------------------------
# High-level training pipeline
# ---------------------------------------------------------------------------

class Word2Vec:
    """High-level interface for Word2Vec training (skip-gram or CBOW)."""

    def __init__(self, embedding_dim=50, window_size=5, min_count=1,
                 neg_samples=5, learning_rate=0.025, method='skipgram',
                 epochs=5, seed=42):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.neg_samples = neg_samples
        self.lr = learning_rate
        self.method = method
        self.epochs = epochs
        self.seed = seed

        self.vocab = None
        self.model = None
        self.embedding_space = None

    def fit(self, corpus):
        """Train on tokenized corpus (list of list of tokens)."""
        rng = np.random.default_rng(self.seed)

        # Build vocabulary
        self.vocab = Vocabulary(min_count=self.min_count)
        self.vocab.build(corpus)

        if len(self.vocab) < 2:
            raise ValueError("Vocabulary too small (need at least 2 words)")

        neg_sampler = NegativeSampler(self.vocab)

        if self.method == 'skipgram':
            pairs = generate_skipgram_pairs(corpus, self.vocab,
                                             self.window_size, rng)
            self.model = SkipGram(len(self.vocab), self.embedding_dim,
                                   self.lr, self.neg_samples, self.seed)
            self.model.train(pairs, neg_sampler, self.epochs)
        elif self.method == 'cbow':
            pairs = generate_cbow_pairs(corpus, self.vocab,
                                         self.window_size, rng)
            self.model = CBOW(len(self.vocab), self.embedding_dim,
                               self.lr, self.neg_samples, self.seed)
            self.model.train(pairs, neg_sampler, self.epochs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        embeddings = self.model.get_embeddings()
        self.embedding_space = EmbeddingSpace(embeddings, self.vocab)
        return self

    def get_space(self):
        """Get the embedding space for similarity queries."""
        return self.embedding_space

    def __getitem__(self, word):
        """Get embedding vector for a word."""
        if self.embedding_space is None:
            raise RuntimeError("Model not trained yet")
        return self.embedding_space.get_vector(word)


class GloVeTrainer:
    """High-level interface for GloVe training."""

    def __init__(self, embedding_dim=50, window_size=10, min_count=1,
                 learning_rate=0.05, x_max=100, alpha=0.75,
                 epochs=25, seed=42):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.lr = learning_rate
        self.x_max = x_max
        self.alpha = alpha
        self.epochs = epochs
        self.seed = seed

        self.vocab = None
        self.model = None
        self.embedding_space = None

    def fit(self, corpus):
        """Train on tokenized corpus."""
        self.vocab = Vocabulary(min_count=self.min_count)
        self.vocab.build(corpus)

        if len(self.vocab) < 2:
            raise ValueError("Vocabulary too small")

        self.model = GloVe(len(self.vocab), self.embedding_dim,
                            self.lr, self.x_max, self.alpha, self.seed)
        cooc = self.model.build_cooccurrence(corpus, self.vocab, self.window_size)
        self.model.train(cooc, self.epochs)

        embeddings = self.model.get_embeddings()
        self.embedding_space = EmbeddingSpace(embeddings, self.vocab)
        return self

    def get_space(self):
        return self.embedding_space


class FastTextTrainer:
    """High-level interface for FastText training."""

    def __init__(self, embedding_dim=50, window_size=5, min_count=1,
                 min_n=3, max_n=6, bucket_size=50000,
                 neg_samples=5, learning_rate=0.025,
                 epochs=5, seed=42):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.min_n = min_n
        self.max_n = max_n
        self.bucket_size = bucket_size
        self.neg_samples = neg_samples
        self.lr = learning_rate
        self.epochs = epochs
        self.seed = seed

        self.vocab = None
        self.model = None
        self.embedding_space = None

    def fit(self, corpus):
        """Train on tokenized corpus."""
        rng = np.random.default_rng(self.seed)

        self.vocab = Vocabulary(min_count=self.min_count)
        self.vocab.build(corpus)

        if len(self.vocab) < 2:
            raise ValueError("Vocabulary too small")

        self.model = FastText(len(self.vocab), self.embedding_dim,
                               self.bucket_size, self.min_n, self.max_n,
                               self.lr, self.neg_samples, self.seed)

        neg_sampler = NegativeSampler(self.vocab)
        pairs = generate_skipgram_pairs(corpus, self.vocab,
                                         self.window_size, rng)
        self.model.train(pairs, self.vocab, neg_sampler, self.epochs)

        embeddings = self.model.get_embeddings()
        self.embedding_space = EmbeddingSpace(embeddings, self.vocab)
        return self

    def get_space(self):
        return self.embedding_space

    def get_oov_embedding(self, word):
        """Get embedding for out-of-vocabulary word."""
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        return self.model.get_oov_embedding(word)
