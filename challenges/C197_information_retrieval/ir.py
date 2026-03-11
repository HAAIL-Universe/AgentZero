"""
C197: Information Retrieval
Inverted index, TF-IDF, BM25, boolean retrieval, ranked retrieval,
query expansion, phrase search, proximity search, faceted search.
All from scratch using only Python stdlib + math.
"""

import math
import re
from collections import defaultdict, Counter


# ============================================================
# Tokenization & Text Processing
# ============================================================

# Common English stop words
STOP_WORDS = frozenset([
    'a', 'an', 'the', 'and', 'or', 'not', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'shall', 'should', 'may', 'might', 'can', 'could',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'out', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 'just', 'because', 'but', 'if', 'while', 'about',
    'up', 'down', 'it', 'its', 'he', 'she', 'they', 'them', 'his',
    'her', 'their', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
    'we', 'us', 'our', 'you', 'your', 'what', 'which', 'who', 'whom',
])


def tokenize(text):
    """Split text into lowercase tokens (alphanumeric sequences)."""
    return re.findall(r'[a-z0-9]+', text.lower())


def stem(word):
    """Simple suffix-stripping stemmer (Porter-lite)."""
    if len(word) <= 3:
        return word
    # Step 1: plurals and past tenses
    if word.endswith('ies') and len(word) > 4:
        word = word[:-3] + 'i'
    elif word.endswith('sses'):
        word = word[:-2]
    elif word.endswith('ss'):
        pass
    elif word.endswith('s') and not word.endswith('us') and not word.endswith('ss'):
        word = word[:-1]

    if word.endswith('eed'):
        if len(word) > 4:
            word = word[:-1]
    elif word.endswith('ed') and len(word) > 4:
        word = word[:-2]
        if word.endswith('at') or word.endswith('bl') or word.endswith('iz'):
            word = word + 'e'
    elif word.endswith('ing') and len(word) > 5:
        word = word[:-3]
        if word.endswith('at') or word.endswith('bl') or word.endswith('iz'):
            word = word + 'e'

    # Step 2: y -> i
    if word.endswith('y') and len(word) > 3:
        word = word[:-1] + 'i'

    # Step 3: common suffixes
    suffix_map = {
        'ational': 'ate', 'tional': 'tion', 'enci': 'ence',
        'anci': 'ance', 'izer': 'ize', 'ation': 'ate',
        'ator': 'ate', 'alism': 'al', 'iveness': 'ive',
        'fulness': 'ful', 'ousness': 'ous', 'aliti': 'al',
        'iviti': 'ive', 'biliti': 'ble',
    }
    for suffix, replacement in suffix_map.items():
        if word.endswith(suffix) and len(word) - len(suffix) > 2:
            word = word[:-len(suffix)] + replacement
            break

    return word


def analyze(text, use_stemming=True, remove_stopwords=True):
    """Full text analysis pipeline: tokenize, optionally remove stop words and stem."""
    tokens = tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    if use_stemming:
        tokens = [stem(t) for t in tokens]
    return tokens


# ============================================================
# Document
# ============================================================

class Document:
    """A document with an ID, text content, and optional metadata/fields."""

    def __init__(self, doc_id, text, fields=None):
        self.doc_id = doc_id
        self.text = text
        self.fields = fields or {}  # name -> value (for faceted search)

    def __repr__(self):
        return f"Document({self.doc_id!r})"


# ============================================================
# Inverted Index
# ============================================================

class Posting:
    """A posting: doc_id + list of positions."""
    __slots__ = ('doc_id', 'positions', 'term_freq')

    def __init__(self, doc_id, positions):
        self.doc_id = doc_id
        self.positions = positions  # list of int positions
        self.term_freq = len(positions)

    def __repr__(self):
        return f"Posting({self.doc_id}, tf={self.term_freq})"


class InvertedIndex:
    """Positional inverted index with TF-IDF and BM25 scoring."""

    def __init__(self, use_stemming=True, remove_stopwords=True):
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        # term -> list of Posting (sorted by doc_id)
        self.index = defaultdict(list)
        # doc_id -> Document
        self.documents = {}
        # doc_id -> token count (for length normalization)
        self.doc_lengths = {}
        # doc_id -> analyzed tokens (for term vectors)
        self.doc_tokens = {}
        # total docs
        self.num_docs = 0
        # average doc length
        self.avg_dl = 0.0
        # field indexes for faceted search
        self.field_index = defaultdict(lambda: defaultdict(set))  # field -> value -> {doc_ids}

    def _analyze(self, text):
        return analyze(text, self.use_stemming, self.remove_stopwords)

    def add_document(self, doc):
        """Add a document to the index."""
        self.documents[doc.doc_id] = doc
        tokens = self._analyze(doc.text)
        self.doc_lengths[doc.doc_id] = len(tokens)
        self.doc_tokens[doc.doc_id] = tokens
        self.num_docs += 1
        # Update average doc length
        self.avg_dl = sum(self.doc_lengths.values()) / self.num_docs

        # Build position list per term
        term_positions = defaultdict(list)
        for pos, token in enumerate(tokens):
            term_positions[token].append(pos)

        # Add postings
        for term, positions in term_positions.items():
            self.index[term].append(Posting(doc.doc_id, positions))

        # Index fields for faceted search
        for field_name, field_value in doc.fields.items():
            if isinstance(field_value, list):
                for v in field_value:
                    self.field_index[field_name][v].add(doc.doc_id)
            else:
                self.field_index[field_name][field_value].add(doc.doc_id)

    def add_documents(self, docs):
        """Add multiple documents."""
        for doc in docs:
            self.add_document(doc)

    def get_postings(self, term):
        """Get postings list for a term (after analysis)."""
        analyzed = self._analyze(term)
        if not analyzed:
            return []
        return self.index.get(analyzed[0], [])

    def doc_freq(self, term):
        """Document frequency for a term."""
        return len(self.get_postings(term))

    def term_freq(self, term, doc_id):
        """Term frequency in a specific document."""
        for posting in self.get_postings(term):
            if posting.doc_id == doc_id:
                return posting.term_freq
        return 0

    # --------------------------------------------------------
    # TF-IDF Scoring
    # --------------------------------------------------------

    def idf(self, term):
        """Inverse document frequency: log(N / df)."""
        df = self.doc_freq(term)
        if df == 0:
            return 0.0
        return math.log(self.num_docs / df)

    def tfidf_score(self, query, doc_id):
        """TF-IDF score for a query against a document."""
        terms = self._analyze(query)
        score = 0.0
        for term in terms:
            tf = self.term_freq(term, doc_id)
            if tf > 0:
                # Log-normalized TF * IDF
                score += (1 + math.log(tf)) * self.idf(term)
        return score

    def tfidf_search(self, query, top_k=10):
        """Search using TF-IDF scoring. Returns [(doc_id, score), ...]."""
        terms = self._analyze(query)
        if not terms:
            return []
        # Collect candidate docs
        candidates = set()
        for term in terms:
            for posting in self.index.get(term, []):
                candidates.add(posting.doc_id)

        results = []
        for doc_id in candidates:
            score = self.tfidf_score(query, doc_id)
            if score > 0:
                results.append((doc_id, score))

        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:top_k]

    # --------------------------------------------------------
    # BM25 Scoring
    # --------------------------------------------------------

    def bm25_score(self, query, doc_id, k1=1.2, b=0.75):
        """BM25 score for a query against a document."""
        terms = self._analyze(query)
        score = 0.0
        dl = self.doc_lengths.get(doc_id, 0)
        for term in terms:
            tf = self.term_freq(term, doc_id)
            if tf == 0:
                continue
            df = self.doc_freq(term)
            # BM25 IDF (with smoothing)
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)
            # BM25 TF normalization
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self.avg_dl))
            score += idf * tf_norm
        return score

    def bm25_search(self, query, top_k=10, k1=1.2, b=0.75):
        """Search using BM25 scoring. Returns [(doc_id, score), ...]."""
        terms = self._analyze(query)
        if not terms:
            return []
        candidates = set()
        for term in terms:
            for posting in self.index.get(term, []):
                candidates.add(posting.doc_id)

        results = []
        for doc_id in candidates:
            score = self.bm25_score(query, doc_id, k1, b)
            if score > 0:
                results.append((doc_id, score))

        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:top_k]

    # --------------------------------------------------------
    # Boolean Retrieval
    # --------------------------------------------------------

    def _term_doc_set(self, term):
        """Get set of doc_ids containing a term."""
        analyzed = self._analyze(term)
        if not analyzed:
            return set()
        postings = self.index.get(analyzed[0], [])
        return {p.doc_id for p in postings}

    def boolean_search(self, query):
        """
        Boolean search supporting AND, OR, NOT.
        Query syntax: "term1 AND term2", "term1 OR term2", "NOT term1"
        Parentheses not supported -- use left-to-right evaluation.
        Returns set of doc_ids.
        """
        tokens = query.split()
        if not tokens:
            return set()

        result = None
        op = 'OR'  # default operator
        negate = False

        i = 0
        while i < len(tokens):
            token = tokens[i]
            upper = token.upper()
            if upper == 'AND':
                op = 'AND'
                i += 1
                continue
            elif upper == 'OR':
                op = 'OR'
                i += 1
                continue
            elif upper == 'NOT':
                negate = True
                i += 1
                continue

            term_set = self._term_doc_set(token)
            if negate:
                all_docs = set(self.documents.keys())
                term_set = all_docs - term_set
                negate = False

            if result is None:
                result = term_set
            elif op == 'AND':
                result = result & term_set
            elif op == 'OR':
                result = result | term_set
            op = 'AND'  # default between consecutive terms is AND
            i += 1

        return result or set()

    # --------------------------------------------------------
    # Phrase Search
    # --------------------------------------------------------

    def phrase_search(self, phrase):
        """
        Search for an exact phrase (terms must appear consecutively).
        Returns set of doc_ids where the phrase appears.
        """
        terms = self._analyze(phrase)
        if not terms:
            return set()
        if len(terms) == 1:
            return self._term_doc_set(terms[0])

        # Get postings for first term
        first_postings = self.index.get(terms[0], [])
        if not first_postings:
            return set()

        # Build doc_id -> positions map for each term
        term_postings = []
        for term in terms:
            postings = self.index.get(term, [])
            doc_pos = {p.doc_id: p.positions for p in postings}
            term_postings.append(doc_pos)

        result = set()
        # For each candidate doc (appears in all terms)
        candidate_docs = set(term_postings[0].keys())
        for tp in term_postings[1:]:
            candidate_docs &= set(tp.keys())

        for doc_id in candidate_docs:
            # Check if terms appear consecutively
            positions_0 = term_postings[0][doc_id]
            for start_pos in positions_0:
                match = True
                for offset in range(1, len(terms)):
                    if (start_pos + offset) not in term_postings[offset].get(doc_id, []):
                        match = False
                        break
                if match:
                    result.add(doc_id)
                    break

        return result

    # --------------------------------------------------------
    # Proximity Search
    # --------------------------------------------------------

    def proximity_search(self, term1, term2, max_distance):
        """
        Find documents where term1 and term2 appear within max_distance positions.
        Returns set of doc_ids.
        """
        analyzed1 = self._analyze(term1)
        analyzed2 = self._analyze(term2)
        if not analyzed1 or not analyzed2:
            return set()
        t1, t2 = analyzed1[0], analyzed2[0]

        postings1 = {p.doc_id: p.positions for p in self.index.get(t1, [])}
        postings2 = {p.doc_id: p.positions for p in self.index.get(t2, [])}

        result = set()
        common_docs = set(postings1.keys()) & set(postings2.keys())
        for doc_id in common_docs:
            pos1 = postings1[doc_id]
            pos2 = postings2[doc_id]
            # Check if any pair is within distance
            for p1 in pos1:
                for p2 in pos2:
                    if abs(p1 - p2) <= max_distance:
                        result.add(doc_id)
                        break
                else:
                    continue
                break

        return result

    # --------------------------------------------------------
    # Faceted Search
    # --------------------------------------------------------

    def faceted_search(self, query, facets):
        """
        Search with facet filters.
        query: text query (BM25 scored)
        facets: dict of field_name -> value (or list of values for OR within facet)
        Returns [(doc_id, score), ...].
        """
        # Get text search results
        if query:
            results = self.bm25_search(query, top_k=len(self.documents))
            result_docs = {doc_id for doc_id, _ in results}
        else:
            result_docs = set(self.documents.keys())

        # Apply facet filters (AND between facets)
        for field_name, values in facets.items():
            if not isinstance(values, list):
                values = [values]
            # OR within same facet
            facet_docs = set()
            for v in values:
                facet_docs |= self.field_index.get(field_name, {}).get(v, set())
            result_docs &= facet_docs

        # Re-score filtered results
        if query:
            scored = []
            for doc_id in result_docs:
                score = self.bm25_score(query, doc_id)
                if score > 0:
                    scored.append((doc_id, score))
            scored.sort(key=lambda x: (-x[1], x[0]))
            return scored
        else:
            return [(doc_id, 0.0) for doc_id in sorted(result_docs)]

    # --------------------------------------------------------
    # Term Vectors & Cosine Similarity
    # --------------------------------------------------------

    def term_vector(self, doc_id):
        """Get TF-IDF term vector for a document."""
        tokens = self.doc_tokens.get(doc_id, [])
        tf_counts = Counter(tokens)
        vector = {}
        for term, tf in tf_counts.items():
            df = len(self.index.get(term, []))
            if df > 0:
                idf = math.log(self.num_docs / df)
                vector[term] = (1 + math.log(tf)) * idf
        return vector

    def cosine_similarity(self, vec1, vec2):
        """Cosine similarity between two sparse vectors (dicts)."""
        common = set(vec1.keys()) & set(vec2.keys())
        if not common:
            return 0.0
        dot = sum(vec1[k] * vec2[k] for k in common)
        norm1 = math.sqrt(sum(v * v for v in vec1.values()))
        norm2 = math.sqrt(sum(v * v for v in vec2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def similar_documents(self, doc_id, top_k=5):
        """Find documents most similar to the given one (cosine similarity)."""
        vec = self.term_vector(doc_id)
        if not vec:
            return []
        results = []
        for other_id in self.documents:
            if other_id == doc_id:
                continue
            other_vec = self.term_vector(other_id)
            sim = self.cosine_similarity(vec, other_vec)
            if sim > 0:
                results.append((other_id, sim))
        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:top_k]


# ============================================================
# Query Expansion
# ============================================================

class QueryExpander:
    """Expand queries using pseudo-relevance feedback (Rocchio-like)."""

    def __init__(self, index, top_docs=3, top_terms=5, alpha=1.0, beta=0.75):
        self.index = index
        self.top_docs = top_docs
        self.top_terms = top_terms
        self.alpha = alpha
        self.beta = beta

    def expand(self, query):
        """
        Expand query using pseudo-relevance feedback.
        1. Run initial query
        2. Take top docs as pseudo-relevant
        3. Extract top terms from those docs
        4. Add to original query
        Returns expanded query string.
        """
        # Initial retrieval
        results = self.index.bm25_search(query, top_k=self.top_docs)
        if not results:
            return query

        # Collect term scores from top docs
        original_terms = set(self.index._analyze(query))
        term_scores = defaultdict(float)

        for doc_id, _ in results:
            vec = self.index.term_vector(doc_id)
            for term, score in vec.items():
                if term not in original_terms:
                    term_scores[term] += score

        # Pick top expansion terms
        expansion = sorted(term_scores.items(), key=lambda x: -x[1])[:self.top_terms]
        expansion_terms = [t for t, _ in expansion]

        # Build expanded query
        expanded = query + ' ' + ' '.join(expansion_terms)
        return expanded.strip()


# ============================================================
# Relevance Feedback
# ============================================================

class RelevanceFeedback:
    """Rocchio algorithm for relevance feedback."""

    def __init__(self, index, alpha=1.0, beta=0.75, gamma=0.15):
        self.index = index
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def reweight(self, query, relevant_ids, non_relevant_ids=None):
        """
        Compute modified query vector using Rocchio formula:
        q' = alpha * q + beta * avg(relevant) - gamma * avg(non_relevant)
        Returns a term -> weight dict (modified query vector).
        """
        if non_relevant_ids is None:
            non_relevant_ids = []

        # Original query vector
        query_terms = self.index._analyze(query)
        q_vec = defaultdict(float)
        for term in query_terms:
            df = len(self.index.index.get(term, []))
            if df > 0:
                idf = math.log(self.index.num_docs / df)
                q_vec[term] = idf

        # Average relevant vector
        rel_vec = defaultdict(float)
        if relevant_ids:
            for doc_id in relevant_ids:
                vec = self.index.term_vector(doc_id)
                for term, weight in vec.items():
                    rel_vec[term] += weight / len(relevant_ids)

        # Average non-relevant vector
        nrel_vec = defaultdict(float)
        if non_relevant_ids:
            for doc_id in non_relevant_ids:
                vec = self.index.term_vector(doc_id)
                for term, weight in vec.items():
                    nrel_vec[term] += weight / len(non_relevant_ids)

        # Rocchio combination
        modified = defaultdict(float)
        all_terms = set(q_vec.keys()) | set(rel_vec.keys()) | set(nrel_vec.keys())
        for term in all_terms:
            w = (self.alpha * q_vec.get(term, 0) +
                 self.beta * rel_vec.get(term, 0) -
                 self.gamma * nrel_vec.get(term, 0))
            if w > 0:
                modified[term] = w

        return dict(modified)

    def search_with_feedback(self, query_vector, top_k=10):
        """Search using a modified query vector (from reweight)."""
        scores = {}
        for doc_id in self.index.documents:
            doc_vec = self.index.term_vector(doc_id)
            score = self.index.cosine_similarity(query_vector, doc_vec)
            if score > 0:
                scores[doc_id] = score

        results = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return results[:top_k]


# ============================================================
# Evaluation Metrics
# ============================================================

class Evaluator:
    """IR evaluation metrics."""

    @staticmethod
    def precision(retrieved, relevant):
        """Precision = |retrieved & relevant| / |retrieved|."""
        if not retrieved:
            return 0.0
        return len(set(retrieved) & set(relevant)) / len(retrieved)

    @staticmethod
    def recall(retrieved, relevant):
        """Recall = |retrieved & relevant| / |relevant|."""
        if not relevant:
            return 0.0
        return len(set(retrieved) & set(relevant)) / len(relevant)

    @staticmethod
    def f1(retrieved, relevant):
        """F1 = 2 * P * R / (P + R)."""
        p = Evaluator.precision(retrieved, relevant)
        r = Evaluator.recall(retrieved, relevant)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @staticmethod
    def precision_at_k(retrieved, relevant, k):
        """P@k."""
        return Evaluator.precision(retrieved[:k], relevant)

    @staticmethod
    def average_precision(retrieved, relevant):
        """Average Precision (AP) for a single query."""
        if not relevant:
            return 0.0
        relevant_set = set(relevant)
        hits = 0
        sum_prec = 0.0
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                hits += 1
                sum_prec += hits / (i + 1)
        return sum_prec / len(relevant_set)

    @staticmethod
    def mean_average_precision(results_list, relevant_list):
        """MAP over multiple queries."""
        if not results_list:
            return 0.0
        return sum(
            Evaluator.average_precision(r, rel)
            for r, rel in zip(results_list, relevant_list)
        ) / len(results_list)

    @staticmethod
    def dcg(retrieved, relevant, k=None):
        """Discounted Cumulative Gain."""
        if k is None:
            k = len(retrieved)
        relevant_set = set(relevant)
        dcg_val = 0.0
        for i in range(min(k, len(retrieved))):
            rel = 1.0 if retrieved[i] in relevant_set else 0.0
            dcg_val += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
        return dcg_val

    @staticmethod
    def ndcg(retrieved, relevant, k=None):
        """Normalized DCG."""
        if not relevant:
            return 0.0
        actual = Evaluator.dcg(retrieved, relevant, k)
        # Ideal: all relevant docs first
        ideal = Evaluator.dcg(relevant, relevant, k)
        if ideal == 0:
            return 0.0
        return actual / ideal

    @staticmethod
    def reciprocal_rank(retrieved, relevant):
        """Reciprocal Rank (1/rank of first relevant result)."""
        relevant_set = set(relevant)
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def mean_reciprocal_rank(results_list, relevant_list):
        """MRR over multiple queries."""
        if not results_list:
            return 0.0
        return sum(
            Evaluator.reciprocal_rank(r, rel)
            for r, rel in zip(results_list, relevant_list)
        ) / len(results_list)


# ============================================================
# Wildcard Search
# ============================================================

class WildcardSearcher:
    """Wildcard query support using permuterm index."""

    def __init__(self, index):
        self.index = index
        self.permuterm = defaultdict(set)  # rotated_term -> original_terms
        self._build_permuterm()

    def _build_permuterm(self):
        """Build permuterm index from all terms in the inverted index."""
        for term in self.index.index:
            augmented = term + '$'
            for i in range(len(augmented)):
                rotation = augmented[i:] + augmented[:i]
                self.permuterm[rotation].add(term)

    def search(self, pattern):
        """
        Search with wildcard pattern. Supports * as wildcard.
        Returns set of doc_ids matching the pattern.
        """
        # Convert pattern to permuterm lookup
        analyzed = pattern.lower()
        if '*' not in analyzed:
            return self.index._term_doc_set(analyzed)

        # Handle pattern with single *
        parts = analyzed.split('*')
        if len(parts) == 2:
            prefix, suffix = parts
            # Rotate: suffix$prefix
            lookup = suffix + '$' + prefix
            matching_terms = set()
            for key, terms in self.permuterm.items():
                if key.startswith(lookup):
                    matching_terms |= terms
        else:
            # Multiple wildcards: use regex fallback
            regex = re.compile('^' + analyzed.replace('*', '.*') + '$')
            matching_terms = {t for t in self.index.index if regex.match(t)}

        result = set()
        for term in matching_terms:
            for posting in self.index.index.get(term, []):
                result.add(posting.doc_id)
        return result


# ============================================================
# Spell Correction
# ============================================================

class SpellCorrector:
    """Simple spell correction using edit distance on index vocabulary."""

    def __init__(self, index):
        self.index = index
        self.vocab = set(index.index.keys())

    @staticmethod
    def edit_distance(s1, s2):
        """Levenshtein edit distance."""
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if s1[i-1] == s2[j-1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j-1])
                prev = temp
        return dp[n]

    def correct(self, term, max_dist=2):
        """Find closest term in vocabulary."""
        term = term.lower()
        if term in self.vocab:
            return term
        best = None
        best_dist = max_dist + 1
        for vocab_term in self.vocab:
            d = self.edit_distance(term, vocab_term)
            if d < best_dist:
                best_dist = d
                best = vocab_term
            elif d == best_dist and best is not None:
                # Prefer higher document frequency
                if len(self.index.index.get(vocab_term, [])) > len(self.index.index.get(best, [])):
                    best = vocab_term
        return best

    def correct_query(self, query, max_dist=2):
        """Correct all terms in a query."""
        tokens = tokenize(query)
        corrected = []
        for token in tokens:
            if token.upper() in ('AND', 'OR', 'NOT'):
                corrected.append(token)
            else:
                c = self.correct(token, max_dist)
                corrected.append(c if c else token)
        return ' '.join(corrected)


# ============================================================
# Snippet/Highlight Generation
# ============================================================

class SnippetGenerator:
    """Generate search result snippets with highlighted terms."""

    def __init__(self, index, snippet_length=100):
        self.index = index
        self.snippet_length = snippet_length

    def generate(self, doc_id, query, highlight_start='**', highlight_end='**'):
        """Generate a snippet for a document, highlighting query terms."""
        doc = self.index.documents.get(doc_id)
        if not doc:
            return ''
        text = doc.text
        query_terms = set(self.index._analyze(query))

        # Find best window
        tokens = tokenize(text)
        if not tokens:
            return ''

        # Score each position by number of query terms nearby
        best_start = 0
        best_score = 0
        window = min(self.snippet_length, len(tokens))

        analyzed_tokens = [stem(t) if self.index.use_stemming else t for t in tokens]

        for i in range(len(tokens) - window + 1):
            score = sum(1 for t in analyzed_tokens[i:i+window] if t in query_terms)
            if score > best_score:
                best_score = score
                best_start = i

        # Build snippet with highlights
        snippet_tokens = tokens[best_start:best_start+window]
        snippet_analyzed = analyzed_tokens[best_start:best_start+window]
        highlighted = []
        for raw, analyzed in zip(snippet_tokens, snippet_analyzed):
            if analyzed in query_terms:
                highlighted.append(f"{highlight_start}{raw}{highlight_end}")
            else:
                highlighted.append(raw)

        snippet = ' '.join(highlighted)
        if best_start > 0:
            snippet = '...' + snippet
        if best_start + window < len(tokens):
            snippet = snippet + '...'
        return snippet


# ============================================================
# Zone Scoring
# ============================================================

class ZoneIndex:
    """
    Zone-weighted scoring: different parts of a document
    (title, body, abstract) get different weights.
    """

    def __init__(self, zone_weights=None, use_stemming=True, remove_stopwords=True):
        self.zone_weights = zone_weights or {'title': 0.4, 'body': 0.6}
        self.zone_indexes = {}
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        for zone in self.zone_weights:
            self.zone_indexes[zone] = InvertedIndex(use_stemming, remove_stopwords)
        self.documents = {}

    def add_document(self, doc_id, zones):
        """
        Add a document with zones.
        zones: dict of zone_name -> text
        """
        self.documents[doc_id] = zones
        for zone_name, text in zones.items():
            if zone_name in self.zone_indexes:
                doc = Document(doc_id, text)
                self.zone_indexes[zone_name].add_document(doc)

    def search(self, query, top_k=10):
        """Search across zones with weighted scoring."""
        scores = defaultdict(float)
        for zone_name, idx in self.zone_indexes.items():
            weight = self.zone_weights.get(zone_name, 0)
            results = idx.bm25_search(query, top_k=len(idx.documents))
            for doc_id, score in results:
                scores[doc_id] += weight * score

        results = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return results[:top_k]
