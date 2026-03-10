"""
C080: Bloom Filter -- Probabilistic Data Structures

Implements:
- Standard Bloom filter (add, query, FPR estimation)
- Counting Bloom filter (supports deletion)
- Scalable Bloom filter (auto-grows to maintain target FPR)
- Partitioned Bloom filter (cache-friendly)
- Cuckoo filter (fingerprint-based, supports deletion)
- Set operations (union, intersection, jaccard similarity)
- Serialization/deserialization
"""

import math
import struct
import hashlib
from typing import Any, Optional, List, Tuple


# --- Hash functions ---

def _fnv1a_32(data: bytes) -> int:
    """FNV-1a 32-bit hash."""
    h = 0x811c9dc5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _murmur3_32(data: bytes, seed: int = 0) -> int:
    """MurmurHash3 32-bit."""
    h = seed & 0xFFFFFFFF
    length = len(data)
    nblocks = length // 4

    for i in range(nblocks):
        k = struct.unpack_from('<I', data, i * 4)[0]
        k = (k * 0xcc9e2d51) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * 0x1b873593) & 0xFFFFFFFF
        h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        h = (h * 5 + 0xe6546b64) & 0xFFFFFFFF

    tail = data[nblocks * 4:]
    k = 0
    if len(tail) >= 3:
        k ^= tail[2] << 16
    if len(tail) >= 2:
        k ^= tail[1] << 8
    if len(tail) >= 1:
        k ^= tail[0]
        k = (k * 0xcc9e2d51) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * 0x1b873593) & 0xFFFFFFFF
        h ^= k

    h ^= length
    h ^= (h >> 16)
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h ^= (h >> 13)
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h ^= (h >> 16)
    return h


def _to_bytes(item: Any) -> bytes:
    """Convert an item to bytes for hashing."""
    if isinstance(item, bytes):
        return item
    if isinstance(item, str):
        return item.encode('utf-8')
    return str(item).encode('utf-8')


def _double_hash(item: Any, k: int, m: int) -> List[int]:
    """Generate k hash positions using double hashing (Kirsch-Mitzenmacher)."""
    data = _to_bytes(item)
    h1 = _murmur3_32(data, 0)
    h2 = _fnv1a_32(data)
    if h2 == 0:
        h2 = 1
    return [(h1 + i * h2) % m for i in range(k)]


# --- Bit Array ---

class BitArray:
    """Compact bit array using bytearray."""

    __slots__ = ('_bits', '_size')

    def __init__(self, size: int):
        self._size = size
        self._bits = bytearray((size + 7) // 8)

    def set(self, i: int):
        self._bits[i >> 3] |= (1 << (i & 7))

    def get(self, i: int) -> bool:
        return bool(self._bits[i >> 3] & (1 << (i & 7)))

    def clear(self, i: int):
        self._bits[i >> 3] &= ~(1 << (i & 7))

    def count_ones(self) -> int:
        return sum(bin(b).count('1') for b in self._bits)

    def __or__(self, other: 'BitArray') -> 'BitArray':
        if self._size != other._size:
            raise ValueError("BitArray sizes must match")
        result = BitArray(self._size)
        for i in range(len(self._bits)):
            result._bits[i] = self._bits[i] | other._bits[i]
        return result

    def __and__(self, other: 'BitArray') -> 'BitArray':
        if self._size != other._size:
            raise ValueError("BitArray sizes must match")
        result = BitArray(self._size)
        for i in range(len(self._bits)):
            result._bits[i] = self._bits[i] & other._bits[i]
        return result

    def to_bytes(self) -> bytes:
        return bytes(self._bits)

    @classmethod
    def from_bytes(cls, data: bytes, size: int) -> 'BitArray':
        ba = cls(size)
        ba._bits = bytearray(data)
        return ba


# --- Standard Bloom Filter ---

class BloomFilter:
    """Standard Bloom filter with configurable capacity and false positive rate."""

    def __init__(self, capacity: int = 1000, fpr: float = 0.01, m: int = None, k: int = None):
        """
        Args:
            capacity: expected number of elements
            fpr: target false positive rate
            m: override bit array size
            k: override number of hash functions
        """
        self._capacity = capacity
        self._target_fpr = fpr
        if m is not None:
            self._m = m
        else:
            self._m = self._optimal_m(capacity, fpr)
        if k is not None:
            self._k = k
        else:
            self._k = self._optimal_k(self._m, capacity)
        self._bits = BitArray(self._m)
        self._count = 0

    @staticmethod
    def _optimal_m(n: int, p: float) -> int:
        """Optimal bit array size: m = -(n * ln(p)) / (ln(2)^2)"""
        if p <= 0 or p >= 1:
            raise ValueError("FPR must be between 0 and 1")
        m = -n * math.log(p) / (math.log(2) ** 2)
        return max(1, int(math.ceil(m)))

    @staticmethod
    def _optimal_k(m: int, n: int) -> int:
        """Optimal number of hash functions: k = (m/n) * ln(2)"""
        if n == 0:
            return 1
        k = (m / n) * math.log(2)
        return max(1, int(round(k)))

    def add(self, item: Any) -> bool:
        """Add an item. Returns True if item was not previously present (probable)."""
        positions = _double_hash(item, self._k, self._m)
        was_present = all(self._bits.get(p) for p in positions)
        for p in positions:
            self._bits.set(p)
        if not was_present:
            self._count += 1
        return not was_present

    def __contains__(self, item: Any) -> bool:
        """Check if item might be in the set. No false negatives."""
        positions = _double_hash(item, self._k, self._m)
        return all(self._bits.get(p) for p in positions)

    def __len__(self) -> int:
        return self._count

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def bit_size(self) -> int:
        return self._m

    @property
    def num_hashes(self) -> int:
        return self._k

    @property
    def fill_ratio(self) -> float:
        """Proportion of bits set to 1."""
        return self._bits.count_ones() / self._m

    def estimated_fpr(self) -> float:
        """Current estimated false positive rate based on fill ratio."""
        x = self._bits.count_ones() / self._m
        if x >= 1.0:
            return 1.0
        return x ** self._k

    def estimated_count(self) -> float:
        """Estimate number of items using bit density (Swamidass-Baldi)."""
        x = self._bits.count_ones()
        if x == 0:
            return 0.0
        if x >= self._m:
            return float('inf')
        return -(self._m / self._k) * math.log(1 - x / self._m)

    def clear(self):
        """Reset the filter."""
        self._bits = BitArray(self._m)
        self._count = 0

    def union(self, other: 'BloomFilter') -> 'BloomFilter':
        """Return a new filter that is the union of self and other."""
        if self._m != other._m or self._k != other._k:
            raise ValueError("Filters must have same m and k for union")
        result = BloomFilter(self._capacity, self._target_fpr, m=self._m, k=self._k)
        result._bits = self._bits | other._bits
        result._count = int(result.estimated_count())
        return result

    def intersection(self, other: 'BloomFilter') -> 'BloomFilter':
        """Return a new filter that is the intersection of self and other."""
        if self._m != other._m or self._k != other._k:
            raise ValueError("Filters must have same m and k for intersection")
        result = BloomFilter(self._capacity, self._target_fpr, m=self._m, k=self._k)
        result._bits = self._bits & other._bits
        result._count = int(result.estimated_count())
        return result

    def jaccard_similarity(self, other: 'BloomFilter') -> float:
        """Estimate Jaccard similarity between two sets."""
        if self._m != other._m or self._k != other._k:
            raise ValueError("Filters must have same m and k")
        intersection_bits = (self._bits & other._bits).count_ones()
        union_bits = (self._bits | other._bits).count_ones()
        if union_bits == 0:
            return 1.0
        return intersection_bits / union_bits

    def serialize(self) -> bytes:
        """Serialize to bytes."""
        header = struct.pack('<IIIQ', self._m, self._k, self._capacity, self._count)
        fpr_bytes = struct.pack('<d', self._target_fpr)
        return header + fpr_bytes + self._bits.to_bytes()

    @classmethod
    def deserialize(cls, data: bytes) -> 'BloomFilter':
        """Deserialize from bytes."""
        m, k, cap, count = struct.unpack_from('<IIIQ', data, 0)
        fpr = struct.unpack_from('<d', data, 20)[0]
        bf = cls(cap, fpr, m=m, k=k)
        bf._bits = BitArray.from_bytes(data[28:], m)
        bf._count = count
        return bf

    def copy(self) -> 'BloomFilter':
        """Return a deep copy."""
        data = self.serialize()
        return BloomFilter.deserialize(data)


# --- Counting Bloom Filter ---

class CountingBloomFilter:
    """Bloom filter with 4-bit counters, supporting deletion."""

    def __init__(self, capacity: int = 1000, fpr: float = 0.01):
        self._capacity = capacity
        self._target_fpr = fpr
        self._m = BloomFilter._optimal_m(capacity, fpr)
        self._k = BloomFilter._optimal_k(self._m, capacity)
        # 4-bit counters (max 15)
        self._counters = bytearray(self._m)
        self._count = 0

    def _get_positions(self, item: Any) -> List[int]:
        return _double_hash(item, self._k, self._m)

    def add(self, item: Any) -> bool:
        """Add an item. Returns True if not previously present."""
        positions = self._get_positions(item)
        was_present = all(self._counters[p] > 0 for p in positions)
        for p in positions:
            if self._counters[p] < 255:
                self._counters[p] += 1
        if not was_present:
            self._count += 1
        return not was_present

    def remove(self, item: Any) -> bool:
        """Remove an item. Returns True if it was (probably) present."""
        positions = self._get_positions(item)
        if not all(self._counters[p] > 0 for p in positions):
            return False
        for p in positions:
            self._counters[p] -= 1
        self._count = max(0, self._count - 1)
        return True

    def __contains__(self, item: Any) -> bool:
        positions = self._get_positions(item)
        return all(self._counters[p] > 0 for p in positions)

    def __len__(self) -> int:
        return self._count

    @property
    def bit_size(self) -> int:
        return self._m

    @property
    def num_hashes(self) -> int:
        return self._k

    def count_of(self, item: Any) -> int:
        """Return minimum counter value across positions (estimate of add count)."""
        positions = self._get_positions(item)
        return min(self._counters[p] for p in positions)


# --- Partitioned Bloom Filter ---

class PartitionedBloomFilter:
    """Bloom filter with k partitions of m/k bits each (better cache locality)."""

    def __init__(self, capacity: int = 1000, fpr: float = 0.01):
        self._capacity = capacity
        self._target_fpr = fpr
        total_m = BloomFilter._optimal_m(capacity, fpr)
        self._k = BloomFilter._optimal_k(total_m, capacity)
        self._partition_size = max(1, total_m // self._k)
        self._m = self._partition_size * self._k
        self._partitions = [BitArray(self._partition_size) for _ in range(self._k)]
        self._count = 0

    def _get_positions(self, item: Any) -> List[int]:
        """One hash position per partition."""
        data = _to_bytes(item)
        h1 = _murmur3_32(data, 0)
        h2 = _fnv1a_32(data)
        if h2 == 0:
            h2 = 1
        return [(h1 + i * h2) % self._partition_size for i in range(self._k)]

    def add(self, item: Any) -> bool:
        positions = self._get_positions(item)
        was_present = all(self._partitions[i].get(positions[i]) for i in range(self._k))
        for i in range(self._k):
            self._partitions[i].set(positions[i])
        if not was_present:
            self._count += 1
        return not was_present

    def __contains__(self, item: Any) -> bool:
        positions = self._get_positions(item)
        return all(self._partitions[i].get(positions[i]) for i in range(self._k))

    def __len__(self) -> int:
        return self._count

    @property
    def bit_size(self) -> int:
        return self._m

    @property
    def num_hashes(self) -> int:
        return self._k

    def fill_ratio(self) -> float:
        total = sum(p.count_ones() for p in self._partitions)
        return total / self._m


# --- Scalable Bloom Filter ---

class ScalableBloomFilter:
    """Bloom filter that grows dynamically to maintain target FPR."""

    def __init__(self, initial_capacity: int = 100, fpr: float = 0.01, growth_factor: int = 2, fpr_tightening: float = 0.5):
        """
        Args:
            initial_capacity: initial filter capacity
            fpr: overall target false positive rate
            growth_factor: capacity multiplier for new filters
            fpr_tightening: FPR ratio between successive filters (r < 1)
        """
        self._initial_capacity = initial_capacity
        self._target_fpr = fpr
        self._growth_factor = growth_factor
        self._fpr_tightening = fpr_tightening
        # First filter gets fpr * (1 - r) to ensure geometric sum converges
        first_fpr = fpr * (1 - fpr_tightening)
        self._filters: List[BloomFilter] = [BloomFilter(initial_capacity, first_fpr)]
        self._count = 0

    def add(self, item: Any) -> bool:
        """Add item. Creates new filter if current is at capacity."""
        # Check if already present
        if item in self:
            return False
        # Check if current filter needs expansion
        current = self._filters[-1]
        if current._count >= current._capacity:
            # Create a new filter with tightened FPR
            new_cap = current._capacity * self._growth_factor
            level = len(self._filters)
            new_fpr = self._target_fpr * (1 - self._fpr_tightening) * (self._fpr_tightening ** level)
            new_fpr = max(new_fpr, 1e-12)  # floor to avoid zero
            self._filters.append(BloomFilter(new_cap, new_fpr))
            current = self._filters[-1]
        current.add(item)
        self._count += 1
        return True

    def __contains__(self, item: Any) -> bool:
        return any(item in f for f in self._filters)

    def __len__(self) -> int:
        return self._count

    @property
    def num_filters(self) -> int:
        return len(self._filters)

    @property
    def total_bits(self) -> int:
        return sum(f.bit_size for f in self._filters)

    def estimated_fpr(self) -> float:
        """Overall FPR is 1 - product(1 - fpr_i)."""
        product = 1.0
        for f in self._filters:
            product *= (1 - f.estimated_fpr())
        return 1 - product


# --- Cuckoo Filter ---

class CuckooFilter:
    """Cuckoo filter -- supports add, query, and delete with fingerprints."""

    MAX_KICKS = 500

    def __init__(self, capacity: int = 1000, bucket_size: int = 4, fingerprint_bits: int = 8):
        """
        Args:
            capacity: number of items (rounded up to number of buckets * bucket_size)
            bucket_size: entries per bucket
            fingerprint_bits: bits per fingerprint (8-32)
        """
        self._bucket_size = bucket_size
        self._fp_bits = fingerprint_bits
        self._fp_mask = (1 << fingerprint_bits) - 1
        self._num_buckets = max(1, capacity // bucket_size)
        # Round to power of 2 for fast modulo
        self._num_buckets = 1 << (self._num_buckets - 1).bit_length()
        self._buckets: List[List[int]] = [[] for _ in range(self._num_buckets)]
        self._count = 0

    def _fingerprint(self, item: Any) -> int:
        """Generate a non-zero fingerprint."""
        data = _to_bytes(item)
        h = _murmur3_32(data, 0x12345678)
        fp = h & self._fp_mask
        return fp if fp != 0 else 1

    def _index(self, item: Any) -> int:
        data = _to_bytes(item)
        return _murmur3_32(data, 0) % self._num_buckets

    def _alt_index(self, i: int, fp: int) -> int:
        """Alternate index using XOR with hash of fingerprint."""
        return (i ^ _murmur3_32(struct.pack('<I', fp), 0x9e3779b9)) % self._num_buckets

    def add(self, item: Any) -> bool:
        """Add item. Returns True on success, False if filter is full."""
        fp = self._fingerprint(item)
        i1 = self._index(item)
        i2 = self._alt_index(i1, fp)

        if len(self._buckets[i1]) < self._bucket_size:
            self._buckets[i1].append(fp)
            self._count += 1
            return True
        if len(self._buckets[i2]) < self._bucket_size:
            self._buckets[i2].append(fp)
            self._count += 1
            return True

        # Must evict
        import random
        i = random.choice([i1, i2])
        for _ in range(self.MAX_KICKS):
            # Pick random entry to evict
            j = random.randrange(len(self._buckets[i]))
            fp, self._buckets[i][j] = self._buckets[i][j], fp
            i = self._alt_index(i, fp)
            if len(self._buckets[i]) < self._bucket_size:
                self._buckets[i].append(fp)
                self._count += 1
                return True

        return False  # Filter is too full

    def __contains__(self, item: Any) -> bool:
        fp = self._fingerprint(item)
        i1 = self._index(item)
        i2 = self._alt_index(i1, fp)
        return fp in self._buckets[i1] or fp in self._buckets[i2]

    def remove(self, item: Any) -> bool:
        """Remove an item. Returns True if found and removed."""
        fp = self._fingerprint(item)
        i1 = self._index(item)
        i2 = self._alt_index(i1, fp)
        if fp in self._buckets[i1]:
            self._buckets[i1].remove(fp)
            self._count -= 1
            return True
        if fp in self._buckets[i2]:
            self._buckets[i2].remove(fp)
            self._count -= 1
            return True
        return False

    def __len__(self) -> int:
        return self._count

    @property
    def capacity(self) -> int:
        return self._num_buckets * self._bucket_size

    @property
    def load_factor(self) -> float:
        return self._count / self.capacity if self.capacity > 0 else 0.0

    @property
    def num_buckets(self) -> int:
        return self._num_buckets


# --- HyperLogLog (Cardinality Estimation) ---

class HyperLogLog:
    """HyperLogLog cardinality estimator."""

    def __init__(self, precision: int = 14):
        """
        Args:
            precision: number of register index bits (4-16). More = more accurate, more memory.
        """
        if precision < 4 or precision > 16:
            raise ValueError("Precision must be between 4 and 16")
        self._p = precision
        self._m = 1 << precision
        self._registers = bytearray(self._m)
        self._alpha = self._compute_alpha(self._m)

    @staticmethod
    def _compute_alpha(m: int) -> float:
        if m == 16:
            return 0.673
        if m == 32:
            return 0.697
        if m == 64:
            return 0.709
        return 0.7213 / (1 + 1.079 / m)

    def add(self, item: Any):
        """Add an item."""
        data = _to_bytes(item)
        # Use a good 32-bit hash
        h = _murmur3_32(data, 0x42)
        # First p bits = register index
        idx = h >> (32 - self._p)
        # Remaining bits: count leading zeros + 1
        w = h & ((1 << (32 - self._p)) - 1)
        # Count leading zeros of the remaining bits
        rho = 1
        for bit in range(31 - self._p, -1, -1):
            if w & (1 << bit):
                break
            rho += 1
        self._registers[idx] = max(self._registers[idx], rho)

    def count(self) -> float:
        """Estimate cardinality."""
        # Raw estimate
        indicator = sum(2.0 ** (-r) for r in self._registers)
        estimate = self._alpha * self._m * self._m / indicator

        # Small range correction
        if estimate <= 2.5 * self._m:
            zeros = self._registers.count(0)
            if zeros > 0:
                estimate = self._m * math.log(self._m / zeros)

        # Large range correction (not needed for 32-bit hash in practice)
        return estimate

    def merge(self, other: 'HyperLogLog') -> 'HyperLogLog':
        """Merge two HLL sketches (union)."""
        if self._p != other._p:
            raise ValueError("Precision must match for merge")
        result = HyperLogLog(self._p)
        for i in range(self._m):
            result._registers[i] = max(self._registers[i], other._registers[i])
        return result

    def __len__(self) -> int:
        return int(self.count())

    @property
    def precision(self) -> int:
        return self._p

    def relative_error(self) -> float:
        """Theoretical relative error: 1.04 / sqrt(m)."""
        return 1.04 / math.sqrt(self._m)


# --- Count-Min Sketch ---

class CountMinSketch:
    """Count-Min Sketch for frequency estimation."""

    def __init__(self, width: int = 1000, depth: int = 7):
        """
        Args:
            width: number of counters per row
            depth: number of hash functions (rows)
        """
        self._width = width
        self._depth = depth
        self._table = [[0] * width for _ in range(depth)]
        self._total = 0

    @classmethod
    def from_error(cls, epsilon: float, delta: float) -> 'CountMinSketch':
        """Create sketch with error bounds.

        Args:
            epsilon: additive error factor (count <= true + epsilon * N)
            delta: probability of exceeding error bound
        """
        width = int(math.ceil(math.e / epsilon))
        depth = int(math.ceil(math.log(1 / delta)))
        return cls(width, max(1, depth))

    def _hashes(self, item: Any) -> List[int]:
        data = _to_bytes(item)
        h1 = _murmur3_32(data, 0)
        h2 = _fnv1a_32(data)
        if h2 == 0:
            h2 = 1
        return [(h1 + i * h2) % self._width for i in range(self._depth)]

    def add(self, item: Any, count: int = 1):
        """Add item with given count."""
        for i, h in enumerate(self._hashes(item)):
            self._table[i][h] += count
        self._total += count

    def query(self, item: Any) -> int:
        """Estimate frequency of item (always >= true count)."""
        return min(self._table[i][h] for i, h in enumerate(self._hashes(item)))

    def __getitem__(self, item: Any) -> int:
        return self.query(item)

    @property
    def total(self) -> int:
        return self._total

    @property
    def width(self) -> int:
        return self._width

    @property
    def depth(self) -> int:
        return self._depth

    def merge(self, other: 'CountMinSketch') -> 'CountMinSketch':
        """Merge two sketches."""
        if self._width != other._width or self._depth != other._depth:
            raise ValueError("Dimensions must match")
        result = CountMinSketch(self._width, self._depth)
        for i in range(self._depth):
            for j in range(self._width):
                result._table[i][j] = self._table[i][j] + other._table[i][j]
        result._total = self._total + other._total
        return result

    def heavy_hitters(self, items: List[Any], threshold: float) -> List[Tuple[Any, int]]:
        """Find items with estimated frequency above threshold * total."""
        cutoff = threshold * self._total
        result = []
        for item in items:
            est = self.query(item)
            if est >= cutoff:
                result.append((item, est))
        result.sort(key=lambda x: -x[1])
        return result


# --- Top-K (Space-Saving) ---

class TopK:
    """Space-Saving algorithm for tracking top-k frequent items."""

    def __init__(self, k: int = 10):
        self._k = k
        self._counts: dict = {}  # item -> count
        self._min_item = None
        self._min_count = 0

    def add(self, item: Any, count: int = 1):
        """Add item."""
        if item in self._counts:
            self._counts[item] += count
        elif len(self._counts) < self._k:
            self._counts[item] = count
        else:
            # Replace minimum
            self._update_min()
            old_count = self._min_count
            del self._counts[self._min_item]
            self._counts[item] = old_count + count
        self._update_min()

    def _update_min(self):
        if not self._counts:
            self._min_item = None
            self._min_count = 0
            return
        self._min_item = min(self._counts, key=self._counts.get)
        self._min_count = self._counts[self._min_item]

    def query(self, item: Any) -> int:
        """Get estimated count (0 if not tracked)."""
        return self._counts.get(item, 0)

    def top(self) -> List[Tuple[Any, int]]:
        """Get top-k items sorted by count descending."""
        return sorted(self._counts.items(), key=lambda x: -x[1])

    def __contains__(self, item: Any) -> bool:
        return item in self._counts

    def __len__(self) -> int:
        return len(self._counts)

    @property
    def k(self) -> int:
        return self._k

    @property
    def min_count(self) -> int:
        return self._min_count if self._counts else 0
