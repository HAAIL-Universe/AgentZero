"""Tests for C080: Bloom Filter -- Probabilistic Data Structures."""

import pytest
import math
import struct
from bloom import (
    BloomFilter, CountingBloomFilter, PartitionedBloomFilter,
    ScalableBloomFilter, CuckooFilter, HyperLogLog, CountMinSketch,
    TopK, BitArray, _double_hash, _to_bytes, _murmur3_32, _fnv1a_32
)


# ============================================================
# Hash Functions
# ============================================================

class TestHashFunctions:
    def test_murmur3_deterministic(self):
        assert _murmur3_32(b"hello") == _murmur3_32(b"hello")

    def test_murmur3_different_inputs(self):
        assert _murmur3_32(b"hello") != _murmur3_32(b"world")

    def test_murmur3_seed(self):
        assert _murmur3_32(b"hello", 0) != _murmur3_32(b"hello", 42)

    def test_murmur3_empty(self):
        h = _murmur3_32(b"")
        assert isinstance(h, int)

    def test_fnv1a_deterministic(self):
        assert _fnv1a_32(b"test") == _fnv1a_32(b"test")

    def test_fnv1a_different(self):
        assert _fnv1a_32(b"abc") != _fnv1a_32(b"xyz")

    def test_to_bytes_str(self):
        assert _to_bytes("hello") == b"hello"

    def test_to_bytes_bytes(self):
        assert _to_bytes(b"raw") == b"raw"

    def test_to_bytes_int(self):
        assert _to_bytes(42) == b"42"

    def test_double_hash_count(self):
        positions = _double_hash("item", 7, 100)
        assert len(positions) == 7

    def test_double_hash_in_range(self):
        positions = _double_hash("item", 5, 50)
        assert all(0 <= p < 50 for p in positions)

    def test_double_hash_deterministic(self):
        a = _double_hash("item", 5, 100)
        b = _double_hash("item", 5, 100)
        assert a == b


# ============================================================
# BitArray
# ============================================================

class TestBitArray:
    def test_init_all_zero(self):
        ba = BitArray(100)
        assert ba.count_ones() == 0

    def test_set_get(self):
        ba = BitArray(100)
        ba.set(42)
        assert ba.get(42)
        assert not ba.get(41)

    def test_clear(self):
        ba = BitArray(100)
        ba.set(10)
        ba.clear(10)
        assert not ba.get(10)

    def test_count_ones(self):
        ba = BitArray(100)
        for i in range(0, 100, 10):
            ba.set(i)
        assert ba.count_ones() == 10

    def test_or(self):
        a = BitArray(64)
        b = BitArray(64)
        a.set(0)
        a.set(10)
        b.set(10)
        b.set(20)
        c = a | b
        assert c.get(0) and c.get(10) and c.get(20)
        assert c.count_ones() == 3

    def test_and(self):
        a = BitArray(64)
        b = BitArray(64)
        a.set(5)
        a.set(10)
        b.set(10)
        b.set(20)
        c = a & b
        assert c.get(10)
        assert not c.get(5)
        assert not c.get(20)
        assert c.count_ones() == 1

    def test_or_size_mismatch(self):
        a = BitArray(32)
        b = BitArray(64)
        with pytest.raises(ValueError):
            a | b

    def test_serialization(self):
        ba = BitArray(100)
        ba.set(7)
        ba.set(99)
        data = ba.to_bytes()
        restored = BitArray.from_bytes(data, 100)
        assert restored.get(7)
        assert restored.get(99)
        assert not restored.get(50)

    def test_boundary_bits(self):
        ba = BitArray(8)
        ba.set(0)
        ba.set(7)
        assert ba.get(0) and ba.get(7)
        assert not ba.get(3)


# ============================================================
# Standard Bloom Filter
# ============================================================

class TestBloomFilter:
    def test_add_and_query(self):
        bf = BloomFilter(100, 0.01)
        bf.add("hello")
        assert "hello" in bf
        assert "world" not in bf  # might false positive but unlikely with low fill

    def test_no_false_negatives(self):
        bf = BloomFilter(1000, 0.01)
        items = [f"item_{i}" for i in range(500)]
        for item in items:
            bf.add(item)
        for item in items:
            assert item in bf

    def test_len(self):
        bf = BloomFilter(100, 0.01)
        assert len(bf) == 0
        bf.add("a")
        assert len(bf) == 1
        bf.add("b")
        assert len(bf) == 2

    def test_duplicate_add(self):
        bf = BloomFilter(100, 0.01)
        assert bf.add("x") is True  # new
        assert bf.add("x") is False  # duplicate

    def test_capacity_property(self):
        bf = BloomFilter(500, 0.05)
        assert bf.capacity == 500

    def test_optimal_m(self):
        m = BloomFilter._optimal_m(1000, 0.01)
        assert m > 0
        # Should be approximately 9585
        assert 9000 < m < 10000

    def test_optimal_k(self):
        m = BloomFilter._optimal_m(1000, 0.01)
        k = BloomFilter._optimal_k(m, 1000)
        assert k >= 1
        # Should be approximately 7
        assert 5 <= k <= 9

    def test_fpr_invalid(self):
        with pytest.raises(ValueError):
            BloomFilter._optimal_m(100, 0.0)
        with pytest.raises(ValueError):
            BloomFilter._optimal_m(100, 1.0)

    def test_fill_ratio(self):
        bf = BloomFilter(100, 0.01)
        assert bf.fill_ratio == 0.0
        bf.add("test")
        assert bf.fill_ratio > 0.0

    def test_estimated_fpr_empty(self):
        bf = BloomFilter(100, 0.01)
        assert bf.estimated_fpr() == 0.0

    def test_estimated_fpr_grows(self):
        bf = BloomFilter(100, 0.01)
        for i in range(50):
            bf.add(f"item_{i}")
        fpr1 = bf.estimated_fpr()
        for i in range(50, 100):
            bf.add(f"item_{i}")
        fpr2 = bf.estimated_fpr()
        assert fpr2 >= fpr1

    def test_estimated_count_accuracy(self):
        bf = BloomFilter(10000, 0.01)
        for i in range(1000):
            bf.add(f"item_{i}")
        est = bf.estimated_count()
        # Should be within 10% of 1000
        assert 800 < est < 1200

    def test_estimated_count_empty(self):
        bf = BloomFilter(100, 0.01)
        assert bf.estimated_count() == 0.0

    def test_clear(self):
        bf = BloomFilter(100, 0.01)
        bf.add("a")
        bf.add("b")
        bf.clear()
        assert len(bf) == 0
        assert "a" not in bf

    def test_false_positive_rate_empirical(self):
        bf = BloomFilter(1000, 0.05)
        for i in range(1000):
            bf.add(f"in_{i}")
        fps = sum(1 for i in range(10000) if f"out_{i}" in bf)
        # FPR should be roughly 5% -- allow generous margin
        assert fps < 1500  # < 15% (generous for 10k tests)

    def test_bit_size_and_hashes(self):
        bf = BloomFilter(100, 0.01)
        assert bf.bit_size > 0
        assert bf.num_hashes > 0

    def test_custom_m_k(self):
        bf = BloomFilter(100, 0.01, m=500, k=3)
        assert bf.bit_size == 500
        assert bf.num_hashes == 3

    def test_many_items(self):
        bf = BloomFilter(10000, 0.001)
        for i in range(5000):
            bf.add(i)
        for i in range(5000):
            assert i in bf

    def test_various_types(self):
        bf = BloomFilter(100, 0.01)
        bf.add("string")
        bf.add(42)
        bf.add(3.14)
        bf.add(b"bytes")
        assert "string" in bf
        assert 42 in bf
        assert 3.14 in bf
        assert b"bytes" in bf


# ============================================================
# Bloom Filter Set Operations
# ============================================================

class TestBloomSetOps:
    def test_union(self):
        bf1 = BloomFilter(100, 0.01)
        bf2 = BloomFilter(100, 0.01)
        bf1.add("a")
        bf1.add("b")
        bf2.add("b")
        bf2.add("c")
        u = bf1.union(bf2)
        assert "a" in u
        assert "b" in u
        assert "c" in u

    def test_intersection(self):
        bf1 = BloomFilter(100, 0.01)
        bf2 = BloomFilter(100, 0.01)
        bf1.add("a")
        bf1.add("b")
        bf2.add("b")
        bf2.add("c")
        inter = bf1.intersection(bf2)
        assert "b" in inter
        # "a" and "c" should probably not be in intersection
        # but could be false positives, so we just test "b" is in

    def test_union_mismatched(self):
        bf1 = BloomFilter(100, 0.01, m=100, k=5)
        bf2 = BloomFilter(100, 0.01, m=200, k=5)
        with pytest.raises(ValueError):
            bf1.union(bf2)

    def test_jaccard_identical(self):
        bf1 = BloomFilter(100, 0.01)
        bf1.add("x")
        bf1.add("y")
        bf2 = bf1.copy()
        assert bf1.jaccard_similarity(bf2) == pytest.approx(1.0)

    def test_jaccard_disjoint(self):
        bf1 = BloomFilter(1000, 0.001)
        bf2 = BloomFilter(1000, 0.001)
        for i in range(100):
            bf1.add(f"a_{i}")
            bf2.add(f"b_{i}")
        j = bf1.jaccard_similarity(bf2)
        assert j < 0.3  # mostly disjoint

    def test_jaccard_empty(self):
        bf1 = BloomFilter(100, 0.01)
        bf2 = BloomFilter(100, 0.01)
        assert bf1.jaccard_similarity(bf2) == 1.0  # both empty


# ============================================================
# Serialization
# ============================================================

class TestBloomSerialization:
    def test_round_trip(self):
        bf = BloomFilter(500, 0.01)
        for i in range(200):
            bf.add(f"item_{i}")
        data = bf.serialize()
        restored = BloomFilter.deserialize(data)
        assert restored.bit_size == bf.bit_size
        assert restored.num_hashes == bf.num_hashes
        assert len(restored) == len(bf)
        for i in range(200):
            assert f"item_{i}" in restored

    def test_copy(self):
        bf = BloomFilter(100, 0.01)
        bf.add("hello")
        cp = bf.copy()
        assert "hello" in cp
        cp.add("world")
        assert "world" not in bf  # original unchanged


# ============================================================
# Counting Bloom Filter
# ============================================================

class TestCountingBloomFilter:
    def test_add_query(self):
        cbf = CountingBloomFilter(100, 0.01)
        cbf.add("hello")
        assert "hello" in cbf

    def test_remove(self):
        cbf = CountingBloomFilter(100, 0.01)
        cbf.add("hello")
        assert "hello" in cbf
        assert cbf.remove("hello") is True
        assert "hello" not in cbf

    def test_remove_nonexistent(self):
        cbf = CountingBloomFilter(100, 0.01)
        assert cbf.remove("nothing") is False

    def test_len(self):
        cbf = CountingBloomFilter(100, 0.01)
        cbf.add("a")
        cbf.add("b")
        assert len(cbf) == 2
        cbf.remove("a")
        assert len(cbf) == 1

    def test_multiple_adds(self):
        cbf = CountingBloomFilter(100, 0.01)
        cbf.add("x")
        cbf.add("x")
        cbf.add("x")
        assert cbf.count_of("x") >= 3
        cbf.remove("x")
        assert "x" in cbf  # still present after one remove

    def test_no_false_negatives(self):
        cbf = CountingBloomFilter(1000, 0.01)
        items = [f"item_{i}" for i in range(500)]
        for item in items:
            cbf.add(item)
        for item in items:
            assert item in cbf

    def test_properties(self):
        cbf = CountingBloomFilter(100, 0.01)
        assert cbf.bit_size > 0
        assert cbf.num_hashes > 0

    def test_count_of_nonexistent(self):
        cbf = CountingBloomFilter(100, 0.01)
        assert cbf.count_of("nothing") == 0


# ============================================================
# Partitioned Bloom Filter
# ============================================================

class TestPartitionedBloomFilter:
    def test_add_query(self):
        pbf = PartitionedBloomFilter(100, 0.01)
        pbf.add("hello")
        assert "hello" in pbf

    def test_no_false_negatives(self):
        pbf = PartitionedBloomFilter(1000, 0.01)
        items = [f"item_{i}" for i in range(500)]
        for item in items:
            pbf.add(item)
        for item in items:
            assert item in pbf

    def test_len(self):
        pbf = PartitionedBloomFilter(100, 0.01)
        pbf.add("a")
        pbf.add("b")
        assert len(pbf) == 2

    def test_fill_ratio(self):
        pbf = PartitionedBloomFilter(100, 0.01)
        assert pbf.fill_ratio() == 0.0
        pbf.add("test")
        assert pbf.fill_ratio() > 0.0

    def test_properties(self):
        pbf = PartitionedBloomFilter(100, 0.01)
        assert pbf.bit_size > 0
        assert pbf.num_hashes > 0


# ============================================================
# Scalable Bloom Filter
# ============================================================

class TestScalableBloomFilter:
    def test_add_query(self):
        sbf = ScalableBloomFilter(100, 0.01)
        sbf.add("hello")
        assert "hello" in sbf

    def test_no_false_negatives(self):
        sbf = ScalableBloomFilter(50, 0.01)
        items = [f"item_{i}" for i in range(500)]
        for item in items:
            sbf.add(item)
        for item in items:
            assert item in sbf

    def test_growth(self):
        sbf = ScalableBloomFilter(10, 0.01, growth_factor=2)
        for i in range(100):
            sbf.add(f"item_{i}")
        assert sbf.num_filters > 1

    def test_len(self):
        sbf = ScalableBloomFilter(100, 0.01)
        for i in range(50):
            sbf.add(f"item_{i}")
        assert len(sbf) == 50

    def test_duplicate_not_counted(self):
        sbf = ScalableBloomFilter(100, 0.01)
        sbf.add("x")
        sbf.add("x")
        assert len(sbf) == 1

    def test_total_bits_grows(self):
        sbf = ScalableBloomFilter(10, 0.01)
        initial = sbf.total_bits
        for i in range(100):
            sbf.add(f"item_{i}")
        assert sbf.total_bits > initial

    def test_estimated_fpr(self):
        sbf = ScalableBloomFilter(100, 0.05)
        for i in range(50):
            sbf.add(f"item_{i}")
        fpr = sbf.estimated_fpr()
        assert 0 <= fpr <= 1


# ============================================================
# Cuckoo Filter
# ============================================================

class TestCuckooFilter:
    def test_add_query(self):
        cf = CuckooFilter(100)
        assert cf.add("hello") is True
        assert "hello" in cf

    def test_remove(self):
        cf = CuckooFilter(100)
        cf.add("hello")
        assert cf.remove("hello") is True
        assert "hello" not in cf

    def test_remove_nonexistent(self):
        cf = CuckooFilter(100)
        assert cf.remove("nothing") is False

    def test_no_false_negatives(self):
        cf = CuckooFilter(1000, bucket_size=4)
        items = [f"item_{i}" for i in range(200)]
        for item in items:
            cf.add(item)
        for item in items:
            assert item in cf

    def test_len(self):
        cf = CuckooFilter(100)
        cf.add("a")
        cf.add("b")
        assert len(cf) == 2
        cf.remove("a")
        assert len(cf) == 1

    def test_capacity(self):
        cf = CuckooFilter(100, bucket_size=4)
        assert cf.capacity >= 100

    def test_load_factor(self):
        cf = CuckooFilter(100)
        assert cf.load_factor == 0.0
        cf.add("x")
        assert cf.load_factor > 0.0

    def test_num_buckets_power_of_2(self):
        cf = CuckooFilter(100)
        assert (cf.num_buckets & (cf.num_buckets - 1)) == 0

    def test_many_items(self):
        cf = CuckooFilter(500, bucket_size=4, fingerprint_bits=16)
        added = 0
        for i in range(300):
            if cf.add(f"item_{i}"):
                added += 1
        assert added >= 250  # should fit most

    def test_delete_then_add(self):
        cf = CuckooFilter(100)
        cf.add("temp")
        cf.remove("temp")
        cf.add("permanent")
        assert "permanent" in cf
        assert "temp" not in cf


# ============================================================
# HyperLogLog
# ============================================================

class TestHyperLogLog:
    def test_empty(self):
        hll = HyperLogLog(10)
        assert hll.count() == 0.0

    def test_single_item(self):
        hll = HyperLogLog(14)
        hll.add("hello")
        assert hll.count() >= 0.5  # at least something

    def test_cardinality_estimate(self):
        hll = HyperLogLog(14)
        for i in range(10000):
            hll.add(f"item_{i}")
        est = hll.count()
        # Should be within 5% for p=14
        assert 8000 < est < 12000

    def test_duplicates_ignored(self):
        hll = HyperLogLog(14)
        for _ in range(1000):
            hll.add("same")
        assert hll.count() < 5  # should be ~1

    def test_merge(self):
        hll1 = HyperLogLog(10)
        hll2 = HyperLogLog(10)
        for i in range(500):
            hll1.add(f"a_{i}")
        for i in range(500):
            hll2.add(f"b_{i}")
        merged = hll1.merge(hll2)
        est = merged.count()
        assert 700 < est < 1300  # ~1000

    def test_merge_overlap(self):
        hll1 = HyperLogLog(12)
        hll2 = HyperLogLog(12)
        for i in range(500):
            hll1.add(f"item_{i}")
            hll2.add(f"item_{i}")
        merged = hll1.merge(hll2)
        est = merged.count()
        assert 350 < est < 700  # ~500 (same items)

    def test_merge_precision_mismatch(self):
        hll1 = HyperLogLog(10)
        hll2 = HyperLogLog(12)
        with pytest.raises(ValueError):
            hll1.merge(hll2)

    def test_precision_bounds(self):
        with pytest.raises(ValueError):
            HyperLogLog(3)
        with pytest.raises(ValueError):
            HyperLogLog(17)

    def test_precision_property(self):
        hll = HyperLogLog(10)
        assert hll.precision == 10

    def test_relative_error(self):
        hll = HyperLogLog(14)
        err = hll.relative_error()
        assert 0 < err < 0.1  # should be ~0.008

    def test_len(self):
        hll = HyperLogLog(14)
        for i in range(100):
            hll.add(f"x_{i}")
        assert len(hll) > 50  # approximate

    def test_various_types(self):
        hll = HyperLogLog(10)
        hll.add("string")
        hll.add(42)
        hll.add(b"bytes")
        assert hll.count() >= 1


# ============================================================
# Count-Min Sketch
# ============================================================

class TestCountMinSketch:
    def test_add_query(self):
        cms = CountMinSketch(1000, 5)
        cms.add("hello", 5)
        assert cms.query("hello") >= 5

    def test_getitem(self):
        cms = CountMinSketch(1000, 5)
        cms.add("x", 3)
        assert cms["x"] >= 3

    def test_never_undercount(self):
        cms = CountMinSketch(1000, 7)
        counts = {"a": 10, "b": 20, "c": 5}
        for item, count in counts.items():
            cms.add(item, count)
        for item, count in counts.items():
            assert cms.query(item) >= count

    def test_total(self):
        cms = CountMinSketch(100, 3)
        cms.add("a", 5)
        cms.add("b", 3)
        assert cms.total == 8

    def test_from_error(self):
        cms = CountMinSketch.from_error(0.001, 0.01)
        assert cms.width > 0
        assert cms.depth > 0

    def test_merge(self):
        cms1 = CountMinSketch(100, 5)
        cms2 = CountMinSketch(100, 5)
        cms1.add("a", 5)
        cms2.add("a", 3)
        merged = cms1.merge(cms2)
        assert merged.query("a") >= 8

    def test_merge_mismatch(self):
        cms1 = CountMinSketch(100, 5)
        cms2 = CountMinSketch(200, 5)
        with pytest.raises(ValueError):
            cms1.merge(cms2)

    def test_heavy_hitters(self):
        cms = CountMinSketch(1000, 7)
        items = []
        for i in range(100):
            items.append(f"item_{i}")
            cms.add(f"item_{i}", 1)
        # Add a heavy hitter
        cms.add("heavy", 50)
        items.append("heavy")
        hh = cms.heavy_hitters(items, 0.2)
        # "heavy" should be in heavy hitters
        found_items = [item for item, _ in hh]
        assert "heavy" in found_items

    def test_width_depth_properties(self):
        cms = CountMinSketch(500, 4)
        assert cms.width == 500
        assert cms.depth == 4

    def test_frequency_estimation_accuracy(self):
        cms = CountMinSketch(2000, 7)
        expected = {}
        for i in range(1000):
            key = f"item_{i % 100}"
            cms.add(key)
            expected[key] = expected.get(key, 0) + 1
        # Estimates should be close to true counts
        total_error = 0
        for key, true_count in expected.items():
            est = cms.query(key)
            assert est >= true_count  # never undercount
            total_error += est - true_count
        avg_error = total_error / len(expected)
        assert avg_error < 5  # reasonable for this size


# ============================================================
# TopK (Space-Saving)
# ============================================================

class TestTopK:
    def test_basic(self):
        tk = TopK(3)
        tk.add("a", 10)
        tk.add("b", 5)
        tk.add("c", 3)
        top = tk.top()
        assert top[0] == ("a", 10)
        assert len(top) == 3

    def test_eviction(self):
        tk = TopK(2)
        tk.add("a", 10)
        tk.add("b", 5)
        tk.add("c", 20)
        top = tk.top()
        assert len(top) == 2
        items = {item for item, _ in top}
        assert "c" in items

    def test_query(self):
        tk = TopK(5)
        tk.add("x", 10)
        assert tk.query("x") == 10
        assert tk.query("y") == 0

    def test_contains(self):
        tk = TopK(5)
        tk.add("x")
        assert "x" in tk
        assert "y" not in tk

    def test_len(self):
        tk = TopK(5)
        tk.add("a")
        tk.add("b")
        assert len(tk) == 2

    def test_k_property(self):
        tk = TopK(7)
        assert tk.k == 7

    def test_min_count(self):
        tk = TopK(3)
        assert tk.min_count == 0
        tk.add("a", 10)
        tk.add("b", 5)
        tk.add("c", 3)
        assert tk.min_count == 3

    def test_incremental(self):
        tk = TopK(5)
        for _ in range(100):
            tk.add("hot")
        for i in range(50):
            tk.add(f"cold_{i}")
        assert tk.query("hot") >= 50  # should be dominant

    def test_streaming(self):
        """Simulate a stream and verify top items are tracked."""
        tk = TopK(5)
        # Item frequencies: a=100, b=50, c=30, rest=1 each
        for _ in range(100):
            tk.add("a")
        for _ in range(50):
            tk.add("b")
        for _ in range(30):
            tk.add("c")
        for i in range(20):
            tk.add(f"rare_{i}")
        top = tk.top()
        top_items = [item for item, _ in top]
        assert "a" in top_items
        assert "b" in top_items


# ============================================================
# Integration / Cross-Structure Tests
# ============================================================

class TestIntegration:
    def test_bloom_vs_counting_consistency(self):
        """Standard and counting filters should agree on membership."""
        bf = BloomFilter(100, 0.01)
        cbf = CountingBloomFilter(100, 0.01)
        items = [f"item_{i}" for i in range(50)]
        for item in items:
            bf.add(item)
            cbf.add(item)
        for item in items:
            assert item in bf
            assert item in cbf

    def test_bloom_union_subset(self):
        """Union should contain all items from both filters."""
        bf1 = BloomFilter(200, 0.01)
        bf2 = BloomFilter(200, 0.01)
        for i in range(50):
            bf1.add(f"set1_{i}")
        for i in range(50):
            bf2.add(f"set2_{i}")
        u = bf1.union(bf2)
        for i in range(50):
            assert f"set1_{i}" in u
            assert f"set2_{i}" in u

    def test_hll_vs_actual(self):
        """HyperLogLog estimate should be in reasonable range."""
        hll = HyperLogLog(14)
        n = 5000
        for i in range(n):
            hll.add(i)
        est = hll.count()
        assert abs(est - n) / n < 0.05  # within 5%

    def test_cms_vs_actual(self):
        """Count-Min Sketch should never undercount."""
        cms = CountMinSketch(2000, 5)
        actual = {}
        import random
        random.seed(42)
        for _ in range(5000):
            item = random.choice(["a", "b", "c", "d", "e", "f", "g"])
            cms.add(item)
            actual[item] = actual.get(item, 0) + 1
        for item, count in actual.items():
            assert cms.query(item) >= count

    def test_scalable_fpr_controlled(self):
        """Scalable filter should maintain reasonable FPR."""
        sbf = ScalableBloomFilter(100, 0.05)
        for i in range(1000):
            sbf.add(f"in_{i}")
        fps = sum(1 for i in range(5000) if f"out_{i}" in sbf)
        fpr = fps / 5000
        assert fpr < 0.15  # generous bound

    def test_cuckoo_add_remove_cycle(self):
        """Add items, remove half, verify remaining."""
        cf = CuckooFilter(200, bucket_size=4)
        for i in range(100):
            cf.add(f"item_{i}")
        for i in range(50):
            cf.remove(f"item_{i}")
        for i in range(50, 100):
            assert f"item_{i}" in cf

    def test_all_structures_empty_query(self):
        """Empty structures should (probably) not contain items."""
        bf = BloomFilter(100, 0.01)
        cbf = CountingBloomFilter(100, 0.01)
        pbf = PartitionedBloomFilter(100, 0.01)
        sbf = ScalableBloomFilter(100, 0.01)
        cf = CuckooFilter(100)
        assert "nothing" not in bf
        assert "nothing" not in cbf
        assert "nothing" not in pbf
        assert "nothing" not in sbf
        assert "nothing" not in cf

    def test_large_scale_bloom(self):
        """Test Bloom filter at scale."""
        bf = BloomFilter(100000, 0.001)
        for i in range(50000):
            bf.add(i)
        for i in range(50000):
            assert i in bf
        # Sample FPR
        fps = sum(1 for i in range(50000, 55000) if i in bf)
        assert fps < 25  # < 0.5% (target 0.1%)

    def test_topk_with_cms_validation(self):
        """TopK and CMS should identify same heavy hitters."""
        tk = TopK(5)
        cms = CountMinSketch(1000, 5)
        for _ in range(100):
            tk.add("dominant")
            cms.add("dominant")
        for i in range(50):
            tk.add(f"noise_{i}")
            cms.add(f"noise_{i}")
        top = tk.top()
        assert top[0][0] == "dominant"
        assert cms.query("dominant") >= 100


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_bloom_single_capacity(self):
        bf = BloomFilter(1, 0.5)
        bf.add("only")
        assert "only" in bf

    def test_bloom_very_low_fpr(self):
        bf = BloomFilter(100, 0.0001)
        assert bf.bit_size > 1000  # needs many bits

    def test_empty_string(self):
        bf = BloomFilter(100, 0.01)
        bf.add("")
        assert "" in bf

    def test_long_string(self):
        bf = BloomFilter(100, 0.01)
        long_s = "x" * 10000
        bf.add(long_s)
        assert long_s in bf

    def test_hll_small_precision(self):
        hll = HyperLogLog(4)
        for i in range(100):
            hll.add(i)
        est = hll.count()
        assert est > 10  # rough, low precision

    def test_cms_single_depth(self):
        cms = CountMinSketch(100, 1)
        cms.add("x", 5)
        assert cms.query("x") >= 5

    def test_cuckoo_fingerprint_bits(self):
        cf = CuckooFilter(50, fingerprint_bits=16)
        cf.add("test")
        assert "test" in cf

    def test_topk_single(self):
        tk = TopK(1)
        tk.add("a", 10)
        tk.add("b", 20)
        top = tk.top()
        assert len(top) == 1
        assert top[0][1] >= 20

    def test_bloom_intersection_empty(self):
        bf1 = BloomFilter(100, 0.01)
        bf2 = BloomFilter(100, 0.01)
        inter = bf1.intersection(bf2)
        assert inter.fill_ratio == 0.0

    def test_counting_overflow_protection(self):
        """Counters should not overflow past 255."""
        cbf = CountingBloomFilter(10, 0.5)
        for _ in range(300):
            cbf.add("overflow")
        assert cbf.count_of("overflow") == 255


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
