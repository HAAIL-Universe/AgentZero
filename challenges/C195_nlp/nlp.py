"""C195: Natural Language Processing -- built from scratch with NumPy.

Components:
- Tokenizer (word, sentence, regex-based, subword BPE)
- TF-IDF (term frequency-inverse document frequency vectorizer)
- Word2Vec (skip-gram with negative sampling, CBOW)
- TextClassifier (Naive Bayes, logistic regression, nearest centroid)
- NGramModel (n-gram language model with smoothing)
- TextSimilarity (cosine, Jaccard, edit distance, BM25)
- TextPreprocessor (stopwords, stemming, lemma rules)

All use NumPy only. No NLTK, spaCy, or scikit-learn.
"""

import numpy as np
import re
import math
from collections import Counter, defaultdict


# ============================================================
# Text Preprocessing
# ============================================================

STOP_WORDS = frozenset([
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'am', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'need',
    'dare', 'ought', 'used', 'it', 'its', 'this', 'that', 'these', 'those',
    'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his',
    'she', 'her', 'they', 'them', 'their', 'what', 'which', 'who', 'whom',
    'not', 'no', 'nor', 'so', 'if', 'then', 'than', 'too', 'very', 'just',
    'about', 'above', 'after', 'again', 'all', 'also', 'any', 'because',
    'before', 'between', 'both', 'each', 'few', 'how', 'into', 'more',
    'most', 'other', 'out', 'over', 'own', 'same', 'some', 'such', 'up',
    'down', 'only', 'now', 'here', 'there', 'when', 'where', 'why',
])


def porter_stem(word):
    """Simplified Porter stemmer (handles common English suffixes)."""
    if len(word) <= 2:
        return word
    w = word.lower()

    # Step 1a: plurals
    if w.endswith('sses'):
        w = w[:-2]
    elif w.endswith('ies'):
        w = w[:-2]
    elif w.endswith('ss'):
        pass
    elif w.endswith('s'):
        w = w[:-1]

    # Step 1b: -ed, -ing
    if w.endswith('eed'):
        if len(w) > 4:
            w = w[:-1]
    elif w.endswith('ed'):
        stem = w[:-2]
        if any(c in 'aeiou' for c in stem):
            w = stem
            w = _step1b_cleanup(w)
    elif w.endswith('ing'):
        stem = w[:-3]
        if any(c in 'aeiou' for c in stem):
            w = stem
            w = _step1b_cleanup(w)

    # Step 1c: y -> i
    if w.endswith('y') and len(w) > 2 and w[-2] not in 'aeiou':
        w = w[:-1] + 'i'

    # Step 2: common suffixes
    step2_map = {
        'ational': 'ate', 'tional': 'tion', 'enci': 'ence', 'anci': 'ance',
        'izer': 'ize', 'isation': 'ize', 'ization': 'ize', 'ation': 'ate',
        'ator': 'ate', 'alism': 'al', 'iveness': 'ive', 'fulness': 'ful',
        'ousli': 'ous', 'ousne': 'ous', 'ousness': 'ous',
        'aliti': 'al', 'iviti': 'ive', 'biliti': 'ble',
        'alli': 'al', 'entli': 'ent', 'eli': 'e', 'ousli': 'ous',
        'lessli': 'less',
    }
    for suffix, replacement in sorted(step2_map.items(), key=lambda x: -len(x[0])):
        if w.endswith(suffix) and len(w) - len(suffix) > 1:
            w = w[:-len(suffix)] + replacement
            break

    # Step 3
    step3_map = {
        'icate': 'ic', 'ative': '', 'alize': 'al', 'iciti': 'ic',
        'ical': 'ic', 'ful': '', 'ness': '',
    }
    for suffix, replacement in sorted(step3_map.items(), key=lambda x: -len(x[0])):
        if w.endswith(suffix) and len(w) - len(suffix) > 1:
            w = w[:-len(suffix)] + replacement
            break

    return w


def _step1b_cleanup(w):
    """Cleanup after step 1b of Porter stemmer."""
    if w.endswith(('at', 'bl', 'iz')):
        w += 'e'
    elif len(w) >= 2 and w[-1] == w[-2] and w[-1] not in 'lsz':
        w = w[:-1]
    elif len(w) <= 3 and w[-1] not in 'aeiou' and len(w) >= 2 and w[-2] in 'aeiou':
        w += 'e'
    return w


class TextPreprocessor:
    """Text preprocessing pipeline."""

    def __init__(self, lowercase=True, remove_punctuation=True,
                 remove_stopwords=False, stem=False, min_length=1):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.min_length = min_length

    def process(self, text):
        """Process text and return list of tokens."""
        if self.lowercase:
            text = text.lower()
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in STOP_WORDS]
        if self.stem:
            tokens = [porter_stem(t) for t in tokens]
        if self.min_length > 1:
            tokens = [t for t in tokens if len(t) >= self.min_length]
        return tokens


# ============================================================
# Tokenizer
# ============================================================

class WordTokenizer:
    """Simple word tokenizer with configurable pattern."""

    def __init__(self, pattern=r'\w+', lowercase=False):
        self.pattern = re.compile(pattern)
        self.lowercase = lowercase

    def tokenize(self, text):
        if self.lowercase:
            text = text.lower()
        return self.pattern.findall(text)


class SentenceTokenizer:
    """Rule-based sentence tokenizer."""

    ABBREVS = frozenset(['mr', 'mrs', 'dr', 'prof', 'sr', 'jr', 'vs', 'etc',
                          'inc', 'ltd', 'co', 'corp', 'dept', 'univ', 'gen',
                          'gov', 'sgt', 'cpl', 'pvt', 'capt', 'lt', 'col',
                          'fig', 'vol', 'no', 'al', 'approx', 'est'])

    def tokenize(self, text):
        """Split text into sentences."""
        sentences = []
        current = []
        tokens = re.split(r'(\s+)', text)
        for token in tokens:
            current.append(token)
            if re.search(r'[.!?]+$', token.strip()):
                word = re.sub(r'[.!?]+$', '', token.strip()).lower()
                if word in self.ABBREVS:
                    continue
                sentences.append(''.join(current).strip())
                current = []
        if current:
            remaining = ''.join(current).strip()
            if remaining:
                sentences.append(remaining)
        return sentences


class BPETokenizer:
    """Byte Pair Encoding subword tokenizer."""

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = []  # list of (a, b) pairs
        self.vocab = {}   # token -> id

    def _get_word_freqs(self, corpus):
        """Get word frequencies with character-level split."""
        freqs = Counter()
        for text in corpus:
            words = re.findall(r'\w+|[^\w\s]', text.lower())
            for w in words:
                chars = tuple(list(w) + ['</w>'])
                freqs[chars] += 1
        return freqs

    def _get_pair_freqs(self, word_freqs):
        """Count frequency of adjacent pairs."""
        pairs = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def _merge_pair(self, pair, word_freqs):
        """Merge the most frequent pair in all words."""
        new_freqs = {}
        a, b = pair
        merged = a + b
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_freqs[tuple(new_word)] = freq
        return new_freqs

    def fit(self, corpus):
        """Learn BPE merges from corpus."""
        word_freqs = self._get_word_freqs(corpus)

        # Initial vocab: all characters
        vocab = set()
        for word in word_freqs:
            for ch in word:
                vocab.add(ch)

        self.merges = []
        while len(vocab) < self.vocab_size:
            pair_freqs = self._get_pair_freqs(word_freqs)
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)
            if pair_freqs[best_pair] < 2:
                break
            word_freqs = self._merge_pair(best_pair, word_freqs)
            merged = best_pair[0] + best_pair[1]
            vocab.add(merged)
            self.merges.append(best_pair)

        self.vocab = {token: i for i, token in enumerate(sorted(vocab))}
        return self

    def tokenize(self, text):
        """Tokenize text using learned BPE merges."""
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        result = []
        for w in words:
            chars = list(w) + ['</w>']
            for a, b in self.merges:
                i = 0
                new_chars = []
                merged = a + b
                while i < len(chars):
                    if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                        new_chars.append(merged)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                chars = new_chars
            result.extend(chars)
        return result

    def encode(self, text):
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        return [self.vocab.get(t, -1) for t in tokens]


# ============================================================
# N-Gram Language Model
# ============================================================

class NGramModel:
    """N-gram language model with Laplace smoothing."""

    def __init__(self, n=2, smoothing=1.0):
        self.n = n
        self.smoothing = smoothing
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = set()

    def fit(self, corpus):
        """Train on list of tokenized sentences (list of list of str)."""
        for tokens in corpus:
            padded = ['<s>'] * (self.n - 1) + list(tokens) + ['</s>']
            for t in tokens:
                self.vocab.add(t)
            self.vocab.add('</s>')
            for i in range(len(padded) - self.n + 1):
                ngram = tuple(padded[i:i + self.n])
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
        return self

    def probability(self, word, context):
        """P(word | context) with Laplace smoothing."""
        context = tuple(context)
        if len(context) != self.n - 1:
            raise ValueError(f"Context must be {self.n - 1} tokens")
        ngram = context + (word,)
        count = self.ngram_counts[ngram]
        context_count = self.context_counts[context]
        V = len(self.vocab)
        return (count + self.smoothing) / (context_count + self.smoothing * V)

    def perplexity(self, tokens):
        """Compute perplexity of a token sequence."""
        padded = ['<s>'] * (self.n - 1) + list(tokens) + ['</s>']
        log_prob = 0.0
        count = 0
        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i - self.n + 1:i])
            word = padded[i]
            p = self.probability(word, context)
            if p > 0:
                log_prob += math.log2(p)
            else:
                log_prob += -100  # penalty
            count += 1
        if count == 0:
            return float('inf')
        return 2.0 ** (-log_prob / count)

    def generate(self, context=None, max_length=20, seed=None):
        """Generate text from the model."""
        rng = np.random.RandomState(seed)
        if context is None:
            context = ['<s>'] * (self.n - 1)
        else:
            context = list(context)
            while len(context) < self.n - 1:
                context = ['<s>'] + context

        result = []
        ctx = list(context[-(self.n - 1):])
        for _ in range(max_length):
            words = list(self.vocab)
            probs = [self.probability(w, ctx) for w in words]
            total = sum(probs)
            if total == 0:
                break
            probs = [p / total for p in probs]
            idx = rng.choice(len(words), p=probs)
            word = words[idx]
            if word == '</s>':
                break
            result.append(word)
            ctx = ctx[1:] + [word]
        return result


# ============================================================
# TF-IDF Vectorizer
# ============================================================

class TfidfVectorizer:
    """TF-IDF vectorizer with configurable options."""

    def __init__(self, max_features=None, min_df=1, max_df=1.0,
                 sublinear_tf=False, norm='l2', preprocessor=None):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.preprocessor = preprocessor or TextPreprocessor()
        self.vocabulary_ = {}
        self.idf_ = None

    def _tokenize(self, doc):
        return self.preprocessor.process(doc)

    def fit(self, documents):
        """Learn vocabulary and IDF from documents."""
        n_docs = len(documents)
        doc_freq = Counter()
        word_set = set()

        tokenized = []
        for doc in documents:
            tokens = self._tokenize(doc)
            tokenized.append(tokens)
            unique = set(tokens)
            for w in unique:
                doc_freq[w] += 1
            word_set.update(tokens)

        # Filter by document frequency
        min_df = self.min_df if isinstance(self.min_df, int) else int(self.min_df * n_docs)
        max_df = self.max_df if isinstance(self.max_df, float) else self.max_df / n_docs
        max_df_count = int(max_df * n_docs) if isinstance(max_df, float) else max_df

        filtered = {w for w in word_set
                     if doc_freq[w] >= min_df and doc_freq[w] <= max_df_count}

        # Sort and optionally limit
        words = sorted(filtered)
        if self.max_features:
            # Keep top by document frequency
            words = sorted(words, key=lambda w: -doc_freq[w])[:self.max_features]
            words = sorted(words)

        self.vocabulary_ = {w: i for i, w in enumerate(words)}

        # IDF: log((1 + n) / (1 + df)) + 1 (sklearn-style smooth IDF)
        self.idf_ = np.zeros(len(words))
        for w, idx in self.vocabulary_.items():
            self.idf_[idx] = math.log((1 + n_docs) / (1 + doc_freq[w])) + 1
        return self

    def transform(self, documents):
        """Transform documents to TF-IDF matrix."""
        n_docs = len(documents)
        n_features = len(self.vocabulary_)
        X = np.zeros((n_docs, n_features))

        for i, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            tf = Counter(tokens)
            for w, count in tf.items():
                if w in self.vocabulary_:
                    j = self.vocabulary_[w]
                    if self.sublinear_tf:
                        X[i, j] = (1 + math.log(count)) * self.idf_[j]
                    else:
                        X[i, j] = count * self.idf_[j]

        if self.norm == 'l2':
            norms = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
            norms[norms == 0] = 1
            X = X / norms
        elif self.norm == 'l1':
            norms = np.sum(np.abs(X), axis=1, keepdims=True)
            norms[norms == 0] = 1
            X = X / norms

        return X

    def fit_transform(self, documents):
        """Fit and transform in one step."""
        self.fit(documents)
        return self.transform(documents)

    def get_feature_names(self):
        """Return feature names (vocabulary in order)."""
        names = [''] * len(self.vocabulary_)
        for w, i in self.vocabulary_.items():
            names[i] = w
        return names


# ============================================================
# Word2Vec (Skip-gram with negative sampling)
# ============================================================

class Word2Vec:
    """Word2Vec with skip-gram and CBOW modes + negative sampling."""

    def __init__(self, embedding_dim=50, window=2, min_count=1,
                 negative_samples=5, learning_rate=0.025, mode='skipgram'):
        self.embedding_dim = embedding_dim
        self.window = window
        self.min_count = min_count
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        self.mode = mode  # 'skipgram' or 'cbow'
        self.word2idx = {}
        self.idx2word = {}
        self.W = None  # input embeddings
        self.W_out = None  # output embeddings
        self._neg_table = None

    def _build_vocab(self, tokenized_corpus):
        """Build vocabulary from tokenized corpus."""
        counts = Counter()
        for tokens in tokenized_corpus:
            counts.update(tokens)

        vocab = {w: c for w, c in counts.items() if c >= self.min_count}
        words = sorted(vocab.keys())
        self.word2idx = {w: i for i, w in enumerate(words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self._word_counts = np.array([vocab[self.idx2word[i]] for i in range(len(words))], dtype=np.float64)
        self._build_neg_table()
        return vocab

    def _build_neg_table(self):
        """Build negative sampling table (unigram^0.75)."""
        freqs = self._word_counts ** 0.75
        freqs /= freqs.sum()
        self._neg_probs = freqs

    def _sigmoid(self, x):
        x = np.clip(x, -10, 10)
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, tokenized_corpus, epochs=5, seed=42):
        """Train Word2Vec on tokenized corpus (list of list of str)."""
        rng = np.random.RandomState(seed)
        self._build_vocab(tokenized_corpus)
        V = len(self.word2idx)
        if V == 0:
            return self

        self.W = (rng.randn(V, self.embedding_dim) * 0.1).astype(np.float64)
        self.W_out = np.zeros((V, self.embedding_dim), dtype=np.float64)

        lr = self.learning_rate
        for epoch in range(epochs):
            for tokens in tokenized_corpus:
                indices = [self.word2idx[t] for t in tokens if t in self.word2idx]
                if len(indices) < 2:
                    continue
                for pos, center_idx in enumerate(indices):
                    w_start = max(0, pos - self.window)
                    w_end = min(len(indices), pos + self.window + 1)
                    context_indices = [indices[j] for j in range(w_start, w_end) if j != pos]

                    if not context_indices:
                        continue

                    if self.mode == 'skipgram':
                        self._train_skipgram(center_idx, context_indices, lr, rng)
                    else:
                        self._train_cbow(center_idx, context_indices, lr, rng)

            # Decay learning rate
            lr = self.learning_rate * (1 - (epoch + 1) / epochs)
            lr = max(lr, self.learning_rate * 0.01)

        return self

    def _train_skipgram(self, center_idx, context_indices, lr, rng):
        """Train one skip-gram step."""
        for ctx_idx in context_indices:
            # Positive sample
            h = self.W[center_idx]
            score = np.dot(h, self.W_out[ctx_idx])
            grad = (self._sigmoid(score) - 1.0) * lr
            grad_out = grad * h
            grad_h = grad * self.W_out[ctx_idx]
            self.W_out[ctx_idx] -= grad_out
            total_grad_h = grad_h

            # Negative samples
            neg_indices = rng.choice(len(self.word2idx), size=self.negative_samples,
                                      p=self._neg_probs)
            for neg_idx in neg_indices:
                if neg_idx == ctx_idx:
                    continue
                score = np.dot(h, self.W_out[neg_idx])
                grad = self._sigmoid(score) * lr
                grad_out = grad * h
                total_grad_h += grad * self.W_out[neg_idx]
                self.W_out[neg_idx] -= grad_out

            self.W[center_idx] -= total_grad_h

    def _train_cbow(self, center_idx, context_indices, lr, rng):
        """Train one CBOW step."""
        # Average context vectors
        h = np.mean(self.W[context_indices], axis=0)

        # Positive sample
        score = np.dot(h, self.W_out[center_idx])
        grad = (self._sigmoid(score) - 1.0) * lr
        grad_out = grad * h
        grad_h = grad * self.W_out[center_idx]
        self.W_out[center_idx] -= grad_out
        total_grad_h = grad_h

        # Negative samples
        neg_indices = rng.choice(len(self.word2idx), size=self.negative_samples,
                                  p=self._neg_probs)
        for neg_idx in neg_indices:
            if neg_idx == center_idx:
                continue
            score = np.dot(h, self.W_out[neg_idx])
            grad = self._sigmoid(score) * lr
            grad_out = grad * h
            total_grad_h += grad * self.W_out[neg_idx]
            self.W_out[neg_idx] -= grad_out

        # Distribute gradient to context words
        update = total_grad_h / len(context_indices)
        for ctx_idx in context_indices:
            self.W[ctx_idx] -= update

    def get_vector(self, word):
        """Get embedding vector for a word."""
        if word not in self.word2idx:
            raise KeyError(f"Word '{word}' not in vocabulary")
        return self.W[self.word2idx[word]].copy()

    def most_similar(self, word, topn=5):
        """Find most similar words by cosine similarity."""
        if word not in self.word2idx:
            raise KeyError(f"Word '{word}' not in vocabulary")
        vec = self.W[self.word2idx[word]]
        norms = np.sqrt(np.sum(self.W ** 2, axis=1))
        norms[norms == 0] = 1
        normed = self.W / norms[:, None]
        vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
        sims = normed @ vec_norm
        # Exclude the word itself
        sims[self.word2idx[word]] = -2
        top_indices = np.argsort(sims)[::-1][:topn]
        return [(self.idx2word[i], float(sims[i])) for i in top_indices]

    def analogy(self, a, b, c, topn=5):
        """a is to b as c is to ? (vector arithmetic)."""
        va = self.get_vector(a)
        vb = self.get_vector(b)
        vc = self.get_vector(c)
        target = vb - va + vc
        norms = np.sqrt(np.sum(self.W ** 2, axis=1))
        norms[norms == 0] = 1
        normed = self.W / norms[:, None]
        target_norm = target / (np.linalg.norm(target) + 1e-10)
        sims = normed @ target_norm
        exclude = {self.word2idx.get(w, -1) for w in [a, b, c]}
        for idx in exclude:
            if 0 <= idx < len(sims):
                sims[idx] = -2
        top_indices = np.argsort(sims)[::-1][:topn]
        return [(self.idx2word[i], float(sims[i])) for i in top_indices]


# ============================================================
# Text Similarity
# ============================================================

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def jaccard_similarity(tokens_a, tokens_b):
    """Jaccard similarity between two token sets."""
    a = set(tokens_a)
    b = set(tokens_b)
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def edit_distance(s1, s2):
    """Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


class BM25:
    """BM25 (Okapi BM25) ranking function."""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_len = []
        self.avgdl = 0
        self.doc_freqs = Counter()
        self.idf_ = {}
        self.doc_term_freqs = []
        self.n_docs = 0

    def fit(self, tokenized_docs):
        """Fit BM25 on tokenized documents."""
        self.n_docs = len(tokenized_docs)
        self.doc_len = []
        self.doc_term_freqs = []
        self.doc_freqs = Counter()

        for tokens in tokenized_docs:
            self.doc_len.append(len(tokens))
            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            for w in set(tokens):
                self.doc_freqs[w] += 1

        self.avgdl = sum(self.doc_len) / max(self.n_docs, 1)

        # IDF with floor at 0
        for w, df in self.doc_freqs.items():
            self.idf_[w] = max(0, math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1))

        return self

    def score(self, query_tokens):
        """Score all documents for a query. Returns array of scores."""
        scores = np.zeros(self.n_docs)
        for i in range(self.n_docs):
            tf = self.doc_term_freqs[i]
            dl = self.doc_len[i]
            for q in query_tokens:
                if q not in self.idf_:
                    continue
                f = tf.get(q, 0)
                idf = self.idf_[q]
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-10))
                scores[i] += idf * numerator / denominator
        return scores

    def top_k(self, query_tokens, k=5):
        """Return top-k document indices for a query."""
        scores = self.score(query_tokens)
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


# ============================================================
# Text Classifiers
# ============================================================

class NaiveBayes:
    """Multinomial Naive Bayes text classifier."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior_ = {}
        self.feature_log_prob_ = {}
        self.classes_ = []
        self.vocabulary_ = {}

    def fit(self, X, y):
        """Fit on TF-IDF or count matrix X with labels y."""
        X = np.asarray(X)
        classes = sorted(set(y))
        self.classes_ = classes
        n_samples, n_features = X.shape

        for c in classes:
            mask = np.array([1 if yi == c else 0 for yi in y], dtype=bool)
            n_c = mask.sum()
            self.class_log_prior_[c] = math.log(n_c / n_samples)

            # Sum features for this class + smoothing
            feature_count = X[mask].sum(axis=0) + self.alpha
            total = feature_count.sum()
            self.feature_log_prob_[c] = np.log(feature_count / total)

        return self

    def predict(self, X):
        """Predict class labels."""
        X = np.asarray(X)
        predictions = []
        for x in X:
            scores = {}
            for c in self.classes_:
                score = self.class_log_prior_[c] + np.sum(x * self.feature_log_prob_[c])
                scores[c] = score
            predictions.append(max(scores, key=scores.get))
        return predictions

    def predict_proba(self, X):
        """Predict class probabilities."""
        X = np.asarray(X)
        proba = []
        for x in X:
            scores = []
            for c in self.classes_:
                score = self.class_log_prior_[c] + np.sum(x * self.feature_log_prob_[c])
                scores.append(score)
            # Log-sum-exp for numerical stability
            max_s = max(scores)
            exp_scores = [math.exp(s - max_s) for s in scores]
            total = sum(exp_scores)
            proba.append([s / total for s in exp_scores])
        return np.array(proba)


class LogisticRegression:
    """Logistic regression classifier (one-vs-rest for multiclass)."""

    def __init__(self, learning_rate=0.1, max_iter=100, reg=0.01):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.reg = reg
        self.weights_ = {}
        self.bias_ = {}
        self.classes_ = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        """Fit on feature matrix X with labels y."""
        X = np.asarray(X, dtype=np.float64)
        self.classes_ = sorted(set(y))
        n_samples, n_features = X.shape

        for c in self.classes_:
            binary_y = np.array([1.0 if yi == c else 0.0 for yi in y])
            w = np.zeros(n_features)
            b = 0.0

            for _ in range(self.max_iter):
                z = X @ w + b
                pred = self._sigmoid(z)
                error = pred - binary_y

                grad_w = (X.T @ error) / n_samples + self.reg * w
                grad_b = error.mean()

                w -= self.learning_rate * grad_w
                b -= self.learning_rate * grad_b

            self.weights_[c] = w
            self.bias_[c] = b

        return self

    def predict(self, X):
        """Predict class labels."""
        X = np.asarray(X, dtype=np.float64)
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for i, c in enumerate(self.classes_):
            scores[:, i] = X @ self.weights_[c] + self.bias_[c]
        indices = np.argmax(scores, axis=1)
        return [self.classes_[i] for i in indices]

    def predict_proba(self, X):
        """Predict probabilities using softmax over OVR scores."""
        X = np.asarray(X, dtype=np.float64)
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for i, c in enumerate(self.classes_):
            scores[:, i] = self._sigmoid(X @ self.weights_[c] + self.bias_[c])
        totals = scores.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1
        return scores / totals


class NearestCentroid:
    """Nearest centroid (Rocchio) classifier."""

    def __init__(self):
        self.centroids_ = {}
        self.classes_ = []

    def fit(self, X, y):
        """Fit by computing class centroids."""
        X = np.asarray(X, dtype=np.float64)
        self.classes_ = sorted(set(y))
        for c in self.classes_:
            mask = np.array([yi == c for yi in y])
            self.centroids_[c] = X[mask].mean(axis=0)
        return self

    def predict(self, X):
        """Predict by nearest centroid."""
        X = np.asarray(X, dtype=np.float64)
        centroids = np.array([self.centroids_[c] for c in self.classes_])
        predictions = []
        for x in X:
            dists = np.sqrt(np.sum((centroids - x) ** 2, axis=1))
            predictions.append(self.classes_[np.argmin(dists)])
        return predictions


# ============================================================
# Document summarization (extractive)
# ============================================================

class ExtractiveSummarizer:
    """Extractive summarization using sentence scoring."""

    def __init__(self, method='tfidf'):
        self.method = method

    def summarize(self, text, n_sentences=3):
        """Extract top-n sentences from text."""
        sent_tok = SentenceTokenizer()
        sentences = sent_tok.tokenize(text)
        if len(sentences) <= n_sentences:
            return ' '.join(sentences)

        if self.method == 'tfidf':
            return self._tfidf_summarize(sentences, n_sentences)
        elif self.method == 'frequency':
            return self._frequency_summarize(sentences, n_sentences)
        else:
            return ' '.join(sentences[:n_sentences])

    def _tfidf_summarize(self, sentences, n):
        """Score sentences by TF-IDF sum."""
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(sentences)
        scores = np.sum(X, axis=1)
        top_indices = np.argsort(scores)[::-1][:n]
        # Maintain original order
        top_indices = sorted(top_indices)
        return ' '.join(sentences[i] for i in top_indices)

    def _frequency_summarize(self, sentences, n):
        """Score sentences by word frequency."""
        prep = TextPreprocessor(remove_stopwords=True)
        all_tokens = []
        sent_tokens = []
        for s in sentences:
            tokens = prep.process(s)
            sent_tokens.append(tokens)
            all_tokens.extend(tokens)

        if not all_tokens:
            return ' '.join(sentences[:n])

        freq = Counter(all_tokens)
        max_freq = max(freq.values())
        norm_freq = {w: c / max_freq for w, c in freq.items()}

        scores = []
        for tokens in sent_tokens:
            score = sum(norm_freq.get(t, 0) for t in tokens) / max(len(tokens), 1)
            scores.append(score)

        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        top_indices = sorted(top_indices)
        return ' '.join(sentences[i] for i in top_indices)


# ============================================================
# Keyword Extraction
# ============================================================

class KeywordExtractor:
    """TF-IDF based keyword extraction."""

    def __init__(self, max_keywords=10):
        self.max_keywords = max_keywords

    def extract(self, text, corpus=None):
        """Extract keywords from text. If corpus given, use IDF from it."""
        prep = TextPreprocessor(remove_stopwords=True, min_length=2)
        tokens = prep.process(text)
        tf = Counter(tokens)

        if corpus:
            n_docs = len(corpus) + 1
            doc_freq = Counter()
            for doc in corpus:
                unique = set(prep.process(doc))
                for w in unique:
                    doc_freq[w] += 1
            for w in set(tokens):
                doc_freq[w] += 1

            scores = {}
            for w, count in tf.items():
                idf = math.log((1 + n_docs) / (1 + doc_freq.get(w, 0))) + 1
                scores[w] = count * idf
        else:
            scores = dict(tf)

        sorted_kw = sorted(scores.items(), key=lambda x: -x[1])
        return [(w, s) for w, s in sorted_kw[:self.max_keywords]]


# ============================================================
# Evaluation metrics
# ============================================================

def accuracy(y_true, y_pred):
    """Classification accuracy."""
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true) if y_true else 0.0


def precision_recall_f1(y_true, y_pred, average='macro'):
    """Precision, recall, F1 score."""
    classes = sorted(set(y_true) | set(y_pred))
    precisions = []
    recalls = []
    f1s = []

    for c in classes:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    if average == 'macro':
        return np.mean(precisions), np.mean(recalls), np.mean(f1s)
    elif average == 'micro':
        tp_total = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        p = tp_total / len(y_pred) if y_pred else 0.0
        r = tp_total / len(y_true) if y_true else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1
    else:
        return precisions, recalls, f1s


def confusion_matrix(y_true, y_pred):
    """Compute confusion matrix."""
    classes = sorted(set(y_true) | set(y_pred))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    matrix = np.zeros((n, n), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[class_to_idx[true], class_to_idx[pred]] += 1
    return matrix, classes
