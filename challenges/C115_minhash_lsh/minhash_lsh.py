"""
C115: MinHash / Locality-Sensitive Hashing (LSH)

A suite of probabilistic similarity search algorithms:

1. MinHash -- Fast Jaccard similarity estimation via min-wise hashing
2. WeightedMinHash -- MinHash for weighted sets (consistent weighted sampling)
3. LSH -- Locality-Sensitive Hashing index for approximate nearest neighbor search
4. LSHForest -- LSH Forest for top-k similarity queries with adaptive probing
5. SimHash -- Cosine similarity via random hyperplane hashing (for text/vectors)
6. MinHashLSHEnsemble -- Containment search (is A a subset of B?)

Domain: Probabilistic similarity search, approximate nearest neighbors.
"""

import hashlib
import struct
import math
import random
from collections import defaultdict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def _sha1_hash(item):
    """Hash an item to a 32-bit integer using SHA1."""
    if isinstance(item, str):
        item = item.encode('utf-8')
    elif isinstance(item, int):
        item = str(item).encode('utf-8')
    elif not isinstance(item, bytes):
        item = str(item).encode('utf-8')
    return struct.unpack('<I', hashlib.sha1(item).digest()[:4])[0]


def _generate_hash_functions(num_perm, seed=1):
    """Generate coefficients for universal hash functions: h(x) = (a*x + b) mod p."""
    rng = random.Random(seed)
    a_vals = [rng.randint(1, _MERSENNE_PRIME - 1) for _ in range(num_perm)]
    b_vals = [rng.randint(0, _MERSENNE_PRIME - 1) for _ in range(num_perm)]
    return a_vals, b_vals


# ---------------------------------------------------------------------------
# 1. MinHash
# ---------------------------------------------------------------------------
class MinHash:
    """MinHash signature for Jaccard similarity estimation.

    Uses universal hashing: h_i(x) = (a_i * x + b_i) mod p mod MAX_HASH.
    The signature is the vector of minimum hash values across all elements.
    """

    def __init__(self, num_perm=128, seed=1):
        self.num_perm = num_perm
        self.seed = seed
        self._a, self._b = _generate_hash_functions(num_perm, seed)
        self.hashvalues = [_MAX_HASH] * num_perm
        self._count = 0

    def update(self, item):
        """Add an item to the set."""
        hv = _sha1_hash(item)
        for i in range(self.num_perm):
            val = ((self._a[i] * hv + self._b[i]) % _MERSENNE_PRIME) & _MAX_HASH
            if val < self.hashvalues[i]:
                self.hashvalues[i] = val
        self._count += 1

    def update_batch(self, items):
        """Add multiple items."""
        for item in items:
            self.update(item)

    def jaccard(self, other):
        """Estimate Jaccard similarity with another MinHash."""
        if self.num_perm != other.num_perm:
            raise ValueError("num_perm mismatch")
        if self.seed != other.seed:
            raise ValueError("seed mismatch")
        matches = sum(1 for a, b in zip(self.hashvalues, other.hashvalues) if a == b)
        return matches / self.num_perm

    def merge(self, other):
        """Merge another MinHash into this one (union)."""
        if self.num_perm != other.num_perm:
            raise ValueError("num_perm mismatch")
        if self.seed != other.seed:
            raise ValueError("seed mismatch")
        for i in range(self.num_perm):
            if other.hashvalues[i] < self.hashvalues[i]:
                self.hashvalues[i] = other.hashvalues[i]

    def copy(self):
        """Return a copy of this MinHash."""
        m = MinHash(self.num_perm, self.seed)
        m.hashvalues = list(self.hashvalues)
        m._count = self._count
        return m

    def is_empty(self):
        """Check if no items have been added."""
        return all(v == _MAX_HASH for v in self.hashvalues)

    def count(self):
        """Return the number of items added."""
        return self._count

    def __eq__(self, other):
        return (isinstance(other, MinHash) and
                self.num_perm == other.num_perm and
                self.hashvalues == other.hashvalues)


# ---------------------------------------------------------------------------
# 2. WeightedMinHash
# ---------------------------------------------------------------------------
class WeightedMinHash:
    """MinHash for weighted sets using Consistent Weighted Sampling (CWS).

    Estimates generalized Jaccard similarity:
    J(A, B) = sum(min(a_i, b_i)) / sum(max(a_i, b_i))

    Uses the Ioffe method: for each dimension and each permutation,
    generate (r, ln_c, beta) and compute k = floor(ln(v)/r + beta), t = exp(r*(k - beta)),
    then a = ln_c - r*t. The signature is argmin over dimensions for each perm.
    """

    def __init__(self, dim, num_perm=128, seed=1):
        self.dim = dim
        self.num_perm = num_perm
        self.seed = seed
        # Pre-generate random parameters
        rng = random.Random(seed)
        self._r = [[rng.gammavariate(2, 1) for _ in range(dim)] for _ in range(num_perm)]
        self._ln_c = [[rng.gammavariate(2, 1) for _ in range(dim)] for _ in range(num_perm)]
        self._beta = [[rng.uniform(0, 1) for _ in range(dim)] for _ in range(num_perm)]
        self.hashvalues = None  # (k_vals, dim_indices)

    def update(self, weights):
        """Set the weight vector. Weights must be non-negative."""
        if len(weights) != self.dim:
            raise ValueError(f"Expected {self.dim} weights, got {len(weights)}")
        k_vals = []
        idx_vals = []
        for i in range(self.num_perm):
            min_a = float('inf')
            min_k = 0
            min_dim = 0
            for d in range(self.dim):
                w = weights[d]
                if w <= 0:
                    continue
                ln_w = math.log(w)
                r = self._r[i][d]
                beta = self._beta[i][d]
                ln_c = self._ln_c[i][d]
                k = math.floor(ln_w / r + beta)
                t = math.exp(r * (k - beta))
                a = ln_c - r * t
                if a < min_a:
                    min_a = a
                    min_k = k
                    min_dim = d
            k_vals.append(min_k)
            idx_vals.append(min_dim)
        self.hashvalues = (k_vals, idx_vals)

    def jaccard(self, other):
        """Estimate generalized Jaccard similarity."""
        if self.hashvalues is None or other.hashvalues is None:
            raise ValueError("Must call update() first")
        if self.num_perm != other.num_perm:
            raise ValueError("num_perm mismatch")
        matches = sum(
            1 for i in range(self.num_perm)
            if self.hashvalues[0][i] == other.hashvalues[0][i]
            and self.hashvalues[1][i] == other.hashvalues[1][i]
        )
        return matches / self.num_perm


# ---------------------------------------------------------------------------
# 3. LSH (Locality-Sensitive Hashing Index)
# ---------------------------------------------------------------------------
class LSH:
    """LSH index for approximate nearest neighbor search using banded MinHash.

    Divides the MinHash signature into `bands` bands of `rows` rows each.
    Two items are candidates if they match in at least one band.

    Probability of being a candidate = 1 - (1 - s^r)^b
    where s = Jaccard similarity, r = rows per band, b = number of bands.
    """

    def __init__(self, threshold=0.5, num_perm=128):
        self.num_perm = num_perm
        self.threshold = threshold
        # Compute optimal b and r
        self._bands, self._rows = self._optimal_params(threshold, num_perm)
        # Hash tables: one per band
        self._tables = [defaultdict(set) for _ in range(self._bands)]
        self._signatures = {}  # key -> MinHash

    @staticmethod
    def _optimal_params(threshold, num_perm):
        """Find b, r that minimize the false positive/negative rate at threshold."""
        best_b, best_r = 1, num_perm
        min_error = float('inf')
        for b in range(1, num_perm + 1):
            r = num_perm // b
            if r == 0:
                continue
            # The "S-curve" crossing point is (1/b)^(1/r)
            # We want this close to threshold
            try:
                crossing = (1.0 / b) ** (1.0 / r)
            except (ZeroDivisionError, OverflowError):
                continue
            error = abs(crossing - threshold)
            if error < min_error:
                min_error = error
                best_b = b
                best_r = r
        return best_b, best_r

    def insert(self, key, minhash):
        """Insert a MinHash signature with a key."""
        if minhash.num_perm != self.num_perm:
            raise ValueError("num_perm mismatch")
        if key in self._signatures:
            raise ValueError(f"Key '{key}' already exists")
        self._signatures[key] = minhash
        for i in range(self._bands):
            start = i * self._rows
            end = start + self._rows
            band_hash = tuple(minhash.hashvalues[start:end])
            self._tables[i][band_hash].add(key)

    def query(self, minhash):
        """Find candidate keys similar to the given MinHash."""
        if minhash.num_perm != self.num_perm:
            raise ValueError("num_perm mismatch")
        candidates = set()
        for i in range(self._bands):
            start = i * self._rows
            end = start + self._rows
            band_hash = tuple(minhash.hashvalues[start:end])
            if band_hash in self._tables[i]:
                candidates.update(self._tables[i][band_hash])
        return candidates

    def query_ranked(self, minhash, top_k=None):
        """Find candidates ranked by estimated Jaccard similarity."""
        candidates = self.query(minhash)
        scored = []
        for key in candidates:
            sim = minhash.jaccard(self._signatures[key])
            scored.append((key, sim))
        scored.sort(key=lambda x: -x[1])
        if top_k is not None:
            scored = scored[:top_k]
        return scored

    def remove(self, key):
        """Remove a key from the index."""
        if key not in self._signatures:
            raise KeyError(f"Key '{key}' not found")
        minhash = self._signatures.pop(key)
        for i in range(self._bands):
            start = i * self._rows
            end = start + self._rows
            band_hash = tuple(minhash.hashvalues[start:end])
            if band_hash in self._tables[i]:
                self._tables[i][band_hash].discard(key)
                if not self._tables[i][band_hash]:
                    del self._tables[i][band_hash]

    def __contains__(self, key):
        return key in self._signatures

    def __len__(self):
        return len(self._signatures)

    def get_bands(self):
        """Return (bands, rows) configuration."""
        return self._bands, self._rows


# ---------------------------------------------------------------------------
# 4. LSHForest
# ---------------------------------------------------------------------------
class LSHForest:
    """LSH Forest for top-k approximate nearest neighbor queries.

    Uses multiple prefix trees (sorted hash arrays) for adaptive probing.
    Each tree stores MinHash prefixes; queries expand the prefix length
    until enough candidates are found.
    """

    def __init__(self, num_perm=128, num_trees=16):
        self.num_perm = num_perm
        self.num_trees = num_trees
        # Each tree is a sorted list of (prefix_hashes, key)
        self._trees = [[] for _ in range(num_trees)]
        self._signatures = {}
        self._sorted = [False] * num_trees
        # Hash function permutations for each tree
        self._perms = self._generate_permutations(num_trees, num_perm)

    def _generate_permutations(self, num_trees, num_perm):
        """Generate index permutations for each tree."""
        rng = random.Random(42)
        perms = []
        base = list(range(num_perm))
        for _ in range(num_trees):
            p = list(base)
            rng.shuffle(p)
            perms.append(p)
        return perms

    def _get_prefix(self, minhash, tree_idx):
        """Get the permuted hash values for a tree."""
        perm = self._perms[tree_idx]
        return tuple(minhash.hashvalues[perm[i]] for i in range(self.num_perm))

    def add(self, key, minhash):
        """Add a MinHash with a key."""
        if minhash.num_perm != self.num_perm:
            raise ValueError("num_perm mismatch")
        self._signatures[key] = minhash
        for t in range(self.num_trees):
            prefix = self._get_prefix(minhash, t)
            self._trees[t].append((prefix, key))
            self._sorted[t] = False

    def _ensure_sorted(self):
        """Sort all trees."""
        for t in range(self.num_trees):
            if not self._sorted[t]:
                self._trees[t].sort()
                self._sorted[t] = True

    def query(self, minhash, top_k=10):
        """Find top-k most similar items using adaptive prefix probing."""
        if minhash.num_perm != self.num_perm:
            raise ValueError("num_perm mismatch")
        self._ensure_sorted()
        candidates = set()
        # Start with longer prefixes, shorten until we have enough candidates
        for prefix_len in range(self.num_perm, 0, -max(1, self.num_perm // 8)):
            for t in range(self.num_trees):
                prefix = self._get_prefix(minhash, t)[:prefix_len]
                # Binary search for matching prefixes
                tree = self._trees[t]
                lo, hi = 0, len(tree)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if tree[mid][0][:prefix_len] < prefix:
                        lo = mid + 1
                    else:
                        hi = mid
                # Collect matches
                while lo < len(tree) and tree[lo][0][:prefix_len] == prefix:
                    candidates.add(tree[lo][1])
                    lo += 1
            if len(candidates) >= top_k * 2:
                break
        # Rank by actual similarity
        scored = []
        for key in candidates:
            sim = minhash.jaccard(self._signatures[key])
            scored.append((key, sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def __len__(self):
        return len(self._signatures)

    def __contains__(self, key):
        return key in self._signatures


# ---------------------------------------------------------------------------
# 5. SimHash
# ---------------------------------------------------------------------------
class SimHash:
    """SimHash for cosine similarity estimation via random hyperplane hashing.

    Projects feature vectors onto random hyperplanes and takes the sign.
    Hamming distance of SimHash signatures approximates cosine distance.
    """

    def __init__(self, num_bits=64, seed=1):
        self.num_bits = num_bits
        self.seed = seed
        self._planes = None  # Lazily initialized
        self.value = 0  # The bit signature as an integer

    def _ensure_planes(self, dim):
        """Generate random hyperplanes for the given dimensionality."""
        if self._planes is not None and len(self._planes[0]) == dim:
            return
        rng = random.Random(self.seed)
        self._planes = []
        for _ in range(self.num_bits):
            plane = [rng.gauss(0, 1) for _ in range(dim)]
            self._planes.append(plane)

    def hash_vector(self, vector):
        """Compute SimHash of a numeric vector."""
        dim = len(vector)
        self._ensure_planes(dim)
        bits = 0
        for i, plane in enumerate(self._planes):
            dot = sum(v * p for v, p in zip(vector, plane))
            if dot >= 0:
                bits |= (1 << i)
        self.value = bits
        return self

    def hash_tokens(self, tokens, weights=None):
        """Compute SimHash from tokens (strings) with optional weights.

        Each token is hashed to num_bits bits, then bits are accumulated
        with weights. Final sign determines the SimHash.
        """
        v = [0.0] * self.num_bits
        for idx, token in enumerate(tokens):
            w = weights[idx] if weights else 1.0
            h = _sha1_hash(token)
            for i in range(self.num_bits):
                if h & (1 << (i % 32)):
                    v[i] += w
                else:
                    v[i] -= w
        bits = 0
        for i in range(self.num_bits):
            if v[i] >= 0:
                bits |= (1 << i)
        self.value = bits
        self._planes = None  # Not using planes for token mode
        return self

    @staticmethod
    def hamming_distance(a, b):
        """Compute Hamming distance between two SimHash values."""
        if isinstance(a, SimHash):
            a = a.value
        if isinstance(b, SimHash):
            b = b.value
        xor = a ^ b
        dist = 0
        while xor:
            dist += 1
            xor &= xor - 1
        return dist

    @staticmethod
    def cosine_similarity(a, b, num_bits=64):
        """Estimate cosine similarity from SimHash Hamming distance."""
        dist = SimHash.hamming_distance(a, b)
        if isinstance(a, SimHash):
            num_bits = a.num_bits
        return math.cos(math.pi * dist / num_bits)

    def __eq__(self, other):
        return isinstance(other, SimHash) and self.value == other.value

    def __hash__(self):
        return hash(self.value)


# ---------------------------------------------------------------------------
# 6. MinHashLSHEnsemble -- Containment Search
# ---------------------------------------------------------------------------
class MinHashLSHEnsemble:
    """LSH Ensemble for containment (subset) queries.

    Given a query set Q and a threshold t, find all indexed sets S where
    |Q intersect S| / |Q| >= t (i.e., at least t fraction of Q is in S).

    Partitions indexed sets by size, uses different LSH parameters per partition.
    """

    def __init__(self, threshold=0.5, num_perm=128, num_partitions=8):
        self.threshold = threshold
        self.num_perm = num_perm
        self.num_partitions = num_partitions
        self._entries = []  # (key, minhash, size)
        self._indexed = False
        self._partitions = []  # [(min_size, max_size, LSH)]

    def index(self, entries):
        """Index a list of (key, minhash, size) tuples.

        Must be called once with all entries before querying.
        """
        self._entries = list(entries)
        if not self._entries:
            self._indexed = True
            return

        sizes = [e[2] for e in self._entries]
        min_size = min(sizes)
        max_size = max(sizes)

        if min_size == max_size:
            # All same size -- single partition
            self._partitions = [(min_size, max_size, self._build_lsh(self._entries, min_size))]
        else:
            # Create partitions by size quantiles
            sorted_sizes = sorted(set(sizes))
            n = len(sorted_sizes)
            num_parts = min(self.num_partitions, n)
            boundaries = []
            for i in range(num_parts):
                idx = i * n // num_parts
                boundaries.append(sorted_sizes[idx])
            boundaries.append(max_size + 1)

            self._partitions = []
            for i in range(len(boundaries) - 1):
                lo, hi = boundaries[i], boundaries[i + 1]
                partition_entries = [e for e in self._entries if lo <= e[2] < hi]
                if partition_entries:
                    avg_size = sum(e[2] for e in partition_entries) / len(partition_entries)
                    lsh = self._build_lsh(partition_entries, avg_size)
                    self._partitions.append((lo, hi - 1, lsh))

        self._indexed = True

    def _build_lsh(self, entries, representative_size):
        """Build an LSH index for a size partition."""
        # For containment, the effective Jaccard threshold depends on set sizes.
        # containment(Q, S) >= t implies jaccard(Q, S) >= t * |Q| / (|Q| + |S| - t*|Q|)
        # We use a very low Jaccard threshold to avoid missing containment candidates
        lsh = LSH(threshold=max(0.05, self.threshold * 0.2), num_perm=self.num_perm)
        for key, minhash, size in entries:
            lsh.insert(key, minhash)
        return lsh

    def query(self, minhash, query_size):
        """Find all indexed sets that contain at least `threshold` fraction of query."""
        if not self._indexed:
            raise RuntimeError("Must call index() first")
        results = set()
        for lo, hi, lsh in self._partitions:
            candidates = lsh.query(minhash)
            for key in candidates:
                stored_mh = lsh._signatures[key]
                # Estimate containment: |Q & S| / |Q|
                # jaccard = |Q & S| / |Q u S|, containment = jaccard * |Q u S| / |Q|
                # |Q u S| = |Q| + |S| - |Q & S|
                # Approximate: containment ~ jaccard * (|Q| + |S|) / (|Q| + |S| * (1 - jaccard))
                j = minhash.jaccard(stored_mh)
                # Direct containment estimation from MinHash intersection count
                # containment = j * (query_size + stored_size) / (query_size + stored_size - j * (query_size + stored_size))
                # Simplified: if sets overlap at Jaccard j, containment ~ j * (1 + |S|/|Q|) approximately
                # More accurate: containment(Q in S) = |Q & S| / |Q|
                # |Q & S| = j * |Q u S| = j * (|Q| + |S| - |Q & S|)
                # |Q & S| = j * |Q| + j * |S| - j * |Q & S|
                # |Q & S| * (1 + j) = j * (|Q| + |S|)
                # |Q & S| = j * (|Q| + |S|) / (1 + j)
                # containment = j * (|Q| + |S|) / ((1 + j) * |Q|)
                # Find stored size
                stored_entry = None
                for e in self._entries:
                    if e[0] == key:
                        stored_entry = e
                        break
                if stored_entry is None:
                    continue
                stored_size = stored_entry[2]
                if j > 0:
                    intersection = j * (query_size + stored_size) / (1 + j)
                    containment = intersection / query_size if query_size > 0 else 0
                else:
                    containment = 0
                if containment >= self.threshold:
                    results.add(key)
        return results
