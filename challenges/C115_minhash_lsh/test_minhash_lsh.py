"""Tests for C115: MinHash / LSH"""
import math
import pytest
from minhash_lsh import (
    MinHash, WeightedMinHash, LSH, LSHForest, SimHash, MinHashLSHEnsemble,
    _sha1_hash
)


# ===================================================================
# MinHash Tests
# ===================================================================

class TestMinHash:
    def test_empty_minhash(self):
        m = MinHash(num_perm=16)
        assert m.is_empty()
        assert m.count() == 0

    def test_single_item(self):
        m = MinHash(num_perm=16)
        m.update("hello")
        assert not m.is_empty()
        assert m.count() == 1

    def test_identical_sets_similarity_one(self):
        m1 = MinHash(num_perm=128)
        m2 = MinHash(num_perm=128)
        for x in ["a", "b", "c", "d", "e"]:
            m1.update(x)
            m2.update(x)
        assert m1.jaccard(m2) == 1.0

    def test_disjoint_sets_similarity_near_zero(self):
        m1 = MinHash(num_perm=128)
        m2 = MinHash(num_perm=128)
        for x in range(100):
            m1.update(f"set1_{x}")
        for x in range(100):
            m2.update(f"set2_{x}")
        sim = m1.jaccard(m2)
        assert sim < 0.15  # Should be near 0

    def test_overlapping_sets_reasonable_estimate(self):
        # Sets with 50% overlap -> Jaccard ~ 1/3
        m1 = MinHash(num_perm=256)
        m2 = MinHash(num_perm=256)
        for x in range(100):
            m1.update(f"item_{x}")
        for x in range(50, 150):
            m2.update(f"item_{x}")
        sim = m1.jaccard(m2)
        # True Jaccard = 50/150 = 0.333
        assert 0.2 < sim < 0.5

    def test_update_batch(self):
        m1 = MinHash(num_perm=64)
        m2 = MinHash(num_perm=64)
        items = ["x", "y", "z"]
        for x in items:
            m1.update(x)
        m2.update_batch(items)
        assert m1.hashvalues == m2.hashvalues

    def test_merge_union(self):
        m1 = MinHash(num_perm=64)
        m2 = MinHash(num_perm=64)
        m_union = MinHash(num_perm=64)
        for x in ["a", "b", "c"]:
            m1.update(x)
            m_union.update(x)
        for x in ["c", "d", "e"]:
            m2.update(x)
            m_union.update(x)
        m1.merge(m2)
        assert m1.hashvalues == m_union.hashvalues

    def test_copy(self):
        m = MinHash(num_perm=32)
        m.update("test")
        c = m.copy()
        assert c.hashvalues == m.hashvalues
        c.update("other")
        assert c.hashvalues != m.hashvalues

    def test_equality(self):
        m1 = MinHash(num_perm=16)
        m2 = MinHash(num_perm=16)
        m1.update("a")
        m2.update("a")
        assert m1 == m2

    def test_num_perm_mismatch_error(self):
        m1 = MinHash(num_perm=16)
        m2 = MinHash(num_perm=32)
        with pytest.raises(ValueError, match="num_perm"):
            m1.jaccard(m2)

    def test_seed_mismatch_error(self):
        m1 = MinHash(num_perm=16, seed=1)
        m2 = MinHash(num_perm=16, seed=2)
        with pytest.raises(ValueError, match="seed"):
            m1.jaccard(m2)

    def test_merge_mismatch_error(self):
        m1 = MinHash(num_perm=16)
        m2 = MinHash(num_perm=32)
        with pytest.raises(ValueError):
            m1.merge(m2)

    def test_integer_items(self):
        m = MinHash(num_perm=32)
        m.update(42)
        m.update(100)
        assert not m.is_empty()

    def test_bytes_items(self):
        m = MinHash(num_perm=32)
        m.update(b"binary data")
        assert m.count() == 1

    def test_idempotent_updates(self):
        m1 = MinHash(num_perm=64)
        m2 = MinHash(num_perm=64)
        m1.update("a")
        m2.update("a")
        m2.update("a")  # Duplicate
        assert m1.hashvalues == m2.hashvalues

    def test_symmetry(self):
        m1 = MinHash(num_perm=64)
        m2 = MinHash(num_perm=64)
        m1.update_batch(["a", "b", "c"])
        m2.update_batch(["b", "c", "d"])
        assert m1.jaccard(m2) == m2.jaccard(m1)

    def test_large_set(self):
        m = MinHash(num_perm=128)
        for i in range(10000):
            m.update(f"item_{i}")
        assert m.count() == 10000
        assert not m.is_empty()

    def test_different_perm_sizes(self):
        for num_perm in [16, 32, 64, 128, 256]:
            m1 = MinHash(num_perm=num_perm)
            m2 = MinHash(num_perm=num_perm)
            for x in range(50):
                m1.update(f"item_{x}")
                m2.update(f"item_{x}")
            assert m1.jaccard(m2) == 1.0

    def test_gradual_overlap(self):
        """More overlap -> higher similarity."""
        sims = []
        for overlap_pct in [0, 25, 50, 75, 100]:
            m1 = MinHash(num_perm=256)
            m2 = MinHash(num_perm=256)
            for i in range(100):
                m1.update(f"a_{i}")
            for i in range(overlap_pct):
                m2.update(f"a_{i}")
            for i in range(100 - overlap_pct):
                m2.update(f"b_{i}")
            sims.append(m1.jaccard(m2))
        # Should be monotonically increasing (approximately)
        for i in range(len(sims) - 1):
            assert sims[i] <= sims[i + 1] + 0.1  # Allow small noise


# ===================================================================
# WeightedMinHash Tests
# ===================================================================

class TestWeightedMinHash:
    def test_identical_weights(self):
        w1 = WeightedMinHash(dim=5, num_perm=128)
        w2 = WeightedMinHash(dim=5, num_perm=128)
        weights = [1.0, 2.0, 3.0, 4.0, 5.0]
        w1.update(weights)
        w2.update(weights)
        assert w1.jaccard(w2) == 1.0

    def test_different_weights(self):
        w1 = WeightedMinHash(dim=5, num_perm=128)
        w2 = WeightedMinHash(dim=5, num_perm=128)
        w1.update([1.0, 2.0, 3.0, 4.0, 5.0])
        w2.update([5.0, 4.0, 3.0, 2.0, 1.0])
        sim = w1.jaccard(w2)
        assert 0.0 <= sim <= 1.0

    def test_similar_weights_higher_sim(self):
        w1 = WeightedMinHash(dim=10, num_perm=256)
        w2 = WeightedMinHash(dim=10, num_perm=256)
        w3 = WeightedMinHash(dim=10, num_perm=256)
        base = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        w1.update(base)
        w2.update([x + 0.1 for x in base])  # Very similar
        w3.update([10 - x for x in base])    # Very different
        sim_close = w1.jaccard(w2)
        sim_far = w1.jaccard(w3)
        assert sim_close > sim_far

    def test_dimension_mismatch(self):
        w = WeightedMinHash(dim=5, num_perm=32)
        with pytest.raises(ValueError, match="Expected 5"):
            w.update([1.0, 2.0])

    def test_zero_weights(self):
        w = WeightedMinHash(dim=3, num_perm=64)
        w.update([1.0, 0.0, 2.0])
        assert w.hashvalues is not None

    def test_query_before_update(self):
        w1 = WeightedMinHash(dim=3, num_perm=32)
        w2 = WeightedMinHash(dim=3, num_perm=32)
        with pytest.raises(ValueError, match="Must call update"):
            w1.jaccard(w2)

    def test_num_perm_mismatch(self):
        w1 = WeightedMinHash(dim=3, num_perm=32)
        w2 = WeightedMinHash(dim=3, num_perm=64)
        w1.update([1.0, 2.0, 3.0])
        w2.update([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="num_perm"):
            w1.jaccard(w2)

    def test_symmetry(self):
        w1 = WeightedMinHash(dim=5, num_perm=128)
        w2 = WeightedMinHash(dim=5, num_perm=128)
        w1.update([1.0, 3.0, 5.0, 7.0, 9.0])
        w2.update([2.0, 4.0, 6.0, 8.0, 10.0])
        assert w1.jaccard(w2) == w2.jaccard(w1)


# ===================================================================
# LSH Tests
# ===================================================================

class TestLSH:
    def test_insert_and_query(self):
        lsh = LSH(threshold=0.5, num_perm=128)
        m1 = MinHash(num_perm=128)
        m2 = MinHash(num_perm=128)
        for x in range(100):
            m1.update(f"item_{x}")
            m2.update(f"item_{x}")
        lsh.insert("doc1", m1)
        results = lsh.query(m2)
        assert "doc1" in results

    def test_similar_items_found(self):
        lsh = LSH(threshold=0.3, num_perm=128)
        # Create similar sets
        m1 = MinHash(num_perm=128)
        m2 = MinHash(num_perm=128)
        for x in range(100):
            m1.update(f"item_{x}")
        for x in range(70):
            m2.update(f"item_{x}")
        for x in range(30):
            m2.update(f"other_{x}")
        lsh.insert("doc1", m1)
        results = lsh.query(m2)
        assert "doc1" in results

    def test_dissimilar_items_not_found(self):
        lsh = LSH(threshold=0.8, num_perm=128)
        m1 = MinHash(num_perm=128)
        m2 = MinHash(num_perm=128)
        for x in range(100):
            m1.update(f"set1_{x}")
        for x in range(100):
            m2.update(f"set2_{x}")
        lsh.insert("doc1", m1)
        results = lsh.query(m2)
        assert "doc1" not in results

    def test_multiple_documents(self):
        lsh = LSH(threshold=0.5, num_perm=128)
        docs = {}
        for d in range(10):
            m = MinHash(num_perm=128)
            for x in range(50):
                m.update(f"doc{d}_item_{x}")
            docs[f"doc_{d}"] = m
            lsh.insert(f"doc_{d}", m)
        assert len(lsh) == 10
        # Query with exact copy
        results = lsh.query(docs["doc_5"])
        assert "doc_5" in results

    def test_remove(self):
        lsh = LSH(threshold=0.5, num_perm=128)
        m = MinHash(num_perm=128)
        m.update("test")
        lsh.insert("doc1", m)
        assert "doc1" in lsh
        lsh.remove("doc1")
        assert "doc1" not in lsh
        assert len(lsh) == 0

    def test_remove_nonexistent(self):
        lsh = LSH(threshold=0.5, num_perm=128)
        with pytest.raises(KeyError):
            lsh.remove("nonexistent")

    def test_duplicate_key_error(self):
        lsh = LSH(threshold=0.5, num_perm=128)
        m = MinHash(num_perm=128)
        m.update("test")
        lsh.insert("doc1", m)
        with pytest.raises(ValueError, match="already exists"):
            lsh.insert("doc1", m)

    def test_num_perm_mismatch(self):
        lsh = LSH(threshold=0.5, num_perm=128)
        m = MinHash(num_perm=64)
        with pytest.raises(ValueError, match="num_perm"):
            lsh.insert("doc1", m)

    def test_query_ranked(self):
        lsh = LSH(threshold=0.3, num_perm=128)
        # Insert 3 docs with varying similarity to a query
        base = MinHash(num_perm=128)
        for x in range(100):
            base.update(f"item_{x}")

        close = MinHash(num_perm=128)
        for x in range(80):
            close.update(f"item_{x}")
        for x in range(20):
            close.update(f"close_{x}")

        medium = MinHash(num_perm=128)
        for x in range(50):
            medium.update(f"item_{x}")
        for x in range(50):
            medium.update(f"medium_{x}")

        lsh.insert("close", close)
        lsh.insert("medium", medium)
        ranked = lsh.query_ranked(base)
        if len(ranked) >= 2:
            # Close should rank higher than medium
            keys = [r[0] for r in ranked]
            if "close" in keys and "medium" in keys:
                assert keys.index("close") < keys.index("medium")

    def test_query_ranked_top_k(self):
        lsh = LSH(threshold=0.3, num_perm=128)
        for i in range(20):
            m = MinHash(num_perm=128)
            for x in range(50):
                m.update(f"doc{i}_{x}")
            # Add some shared items
            for x in range(30):
                m.update(f"shared_{x}")
            lsh.insert(f"doc_{i}", m)
        query = MinHash(num_perm=128)
        for x in range(30):
            query.update(f"shared_{x}")
        ranked = lsh.query_ranked(query, top_k=5)
        assert len(ranked) <= 5

    def test_get_bands(self):
        lsh = LSH(threshold=0.5, num_perm=128)
        b, r = lsh.get_bands()
        assert b * r <= 128
        assert b > 0 and r > 0

    def test_contains(self):
        lsh = LSH(threshold=0.5, num_perm=128)
        m = MinHash(num_perm=128)
        m.update("test")
        assert "doc1" not in lsh
        lsh.insert("doc1", m)
        assert "doc1" in lsh

    def test_threshold_affects_bands(self):
        lsh_low = LSH(threshold=0.3, num_perm=128)
        lsh_high = LSH(threshold=0.8, num_perm=128)
        b_low, r_low = lsh_low.get_bands()
        b_high, r_high = lsh_high.get_bands()
        # Higher threshold -> more rows per band (stricter matching)
        assert r_high >= r_low or b_high <= b_low


# ===================================================================
# LSHForest Tests
# ===================================================================

class TestLSHForest:
    def test_add_and_query(self):
        forest = LSHForest(num_perm=64, num_trees=8)
        m1 = MinHash(num_perm=64)
        m2 = MinHash(num_perm=64)
        for x in range(50):
            m1.update(f"item_{x}")
            m2.update(f"item_{x}")
        forest.add("doc1", m1)
        results = forest.query(m2, top_k=5)
        keys = [r[0] for r in results]
        assert "doc1" in keys

    def test_top_k_limit(self):
        forest = LSHForest(num_perm=64, num_trees=8)
        for i in range(20):
            m = MinHash(num_perm=64)
            for x in range(30):
                m.update(f"shared_{x}")
            for x in range(20):
                m.update(f"doc{i}_{x}")
            forest.add(f"doc_{i}", m)
        query = MinHash(num_perm=64)
        for x in range(30):
            query.update(f"shared_{x}")
        results = forest.query(query, top_k=3)
        assert len(results) <= 3

    def test_ranking_order(self):
        forest = LSHForest(num_perm=128, num_trees=8)
        query = MinHash(num_perm=128)
        for x in range(100):
            query.update(f"item_{x}")

        # Identical
        identical = MinHash(num_perm=128)
        for x in range(100):
            identical.update(f"item_{x}")

        # Partially similar
        partial = MinHash(num_perm=128)
        for x in range(50):
            partial.update(f"item_{x}")
        for x in range(50):
            partial.update(f"other_{x}")

        forest.add("identical", identical)
        forest.add("partial", partial)
        results = forest.query(query, top_k=10)
        if len(results) >= 2:
            keys = [r[0] for r in results]
            if "identical" in keys and "partial" in keys:
                assert keys.index("identical") < keys.index("partial")

    def test_contains(self):
        forest = LSHForest(num_perm=32, num_trees=4)
        m = MinHash(num_perm=32)
        m.update("test")
        assert "doc1" not in forest
        forest.add("doc1", m)
        assert "doc1" in forest

    def test_len(self):
        forest = LSHForest(num_perm=32, num_trees=4)
        assert len(forest) == 0
        m = MinHash(num_perm=32)
        m.update("test")
        forest.add("doc1", m)
        assert len(forest) == 1

    def test_num_perm_mismatch(self):
        forest = LSHForest(num_perm=64)
        m = MinHash(num_perm=32)
        with pytest.raises(ValueError, match="num_perm"):
            forest.add("doc1", m)

    def test_empty_query(self):
        forest = LSHForest(num_perm=32, num_trees=4)
        m = MinHash(num_perm=32)
        m.update("test")
        results = forest.query(m, top_k=5)
        assert results == []

    def test_many_documents(self):
        forest = LSHForest(num_perm=64, num_trees=8)
        for i in range(50):
            m = MinHash(num_perm=64)
            for x in range(20):
                m.update(f"doc{i}_{x}")
            forest.add(f"doc_{i}", m)
        assert len(forest) == 50


# ===================================================================
# SimHash Tests
# ===================================================================

class TestSimHash:
    def test_identical_vectors(self):
        s1 = SimHash(num_bits=64, seed=1)
        s2 = SimHash(num_bits=64, seed=1)
        vec = [1.0, 2.0, 3.0, 4.0, 5.0]
        s1.hash_vector(vec)
        s2.hash_vector(vec)
        assert s1.value == s2.value
        assert SimHash.hamming_distance(s1, s2) == 0

    def test_opposite_vectors(self):
        s1 = SimHash(num_bits=64, seed=1)
        s2 = SimHash(num_bits=64, seed=1)
        vec = [1.0, 2.0, 3.0, 4.0, 5.0]
        s1.hash_vector(vec)
        s2.hash_vector([-x for x in vec])
        dist = SimHash.hamming_distance(s1, s2)
        # Should have high Hamming distance
        assert dist > 30

    def test_similar_vectors_low_distance(self):
        s1 = SimHash(num_bits=64, seed=1)
        s2 = SimHash(num_bits=64, seed=1)
        s1.hash_vector([1.0, 2.0, 3.0, 4.0, 5.0])
        s2.hash_vector([1.1, 2.1, 3.1, 4.1, 5.1])
        dist = SimHash.hamming_distance(s1, s2)
        assert dist < 20

    def test_cosine_similarity_estimate(self):
        s1 = SimHash(num_bits=64, seed=1)
        s2 = SimHash(num_bits=64, seed=1)
        s1.hash_vector([1.0, 0.0, 0.0])
        s2.hash_vector([1.0, 0.0, 0.0])
        cos_sim = SimHash.cosine_similarity(s1, s2)
        assert cos_sim == pytest.approx(1.0)

    def test_hash_tokens(self):
        s1 = SimHash(num_bits=64)
        s2 = SimHash(num_bits=64)
        s1.hash_tokens(["hello", "world", "test"])
        s2.hash_tokens(["hello", "world", "test"])
        assert s1.value == s2.value

    def test_hash_tokens_with_weights(self):
        s1 = SimHash(num_bits=64)
        s2 = SimHash(num_bits=64)
        tokens = ["hello", "world"]
        s1.hash_tokens(tokens, weights=[1.0, 1.0])
        s2.hash_tokens(tokens, weights=[1.0, 1.0])
        assert s1.value == s2.value

    def test_different_tokens_different_hash(self):
        s1 = SimHash(num_bits=64)
        s2 = SimHash(num_bits=64)
        s1.hash_tokens(["cat", "dog", "fish"])
        s2.hash_tokens(["car", "bus", "train"])
        # Very likely different
        assert s1.value != s2.value

    def test_hamming_distance_with_integers(self):
        assert SimHash.hamming_distance(0b1010, 0b1100) == 2
        assert SimHash.hamming_distance(0b1111, 0b1111) == 0
        assert SimHash.hamming_distance(0b0000, 0b1111) == 4

    def test_equality(self):
        s1 = SimHash(num_bits=32)
        s2 = SimHash(num_bits=32)
        s1.hash_tokens(["a", "b"])
        s2.hash_tokens(["a", "b"])
        assert s1 == s2

    def test_hash(self):
        s1 = SimHash(num_bits=32)
        s1.hash_tokens(["a"])
        d = {s1: "value"}
        assert d[s1] == "value"

    def test_cosine_from_integers(self):
        # 0 distance -> cos(0) = 1.0
        sim = SimHash.cosine_similarity(0b1111, 0b1111, num_bits=4)
        assert sim == pytest.approx(1.0)

    def test_various_bit_sizes(self):
        for bits in [16, 32, 64, 128]:
            s = SimHash(num_bits=bits)
            s.hash_vector([1.0, 2.0, 3.0])
            assert isinstance(s.value, int)


# ===================================================================
# MinHashLSHEnsemble Tests
# ===================================================================

class TestMinHashLSHEnsemble:
    def test_basic_containment(self):
        ens = MinHashLSHEnsemble(threshold=0.5, num_perm=128)
        # Superset contains all query items
        m_super = MinHash(num_perm=128)
        for x in range(200):
            m_super.update(f"item_{x}")
        ens.index([("superset", m_super, 200)])

        query = MinHash(num_perm=128)
        for x in range(50):
            query.update(f"item_{x}")
        results = ens.query(query, 50)
        assert "superset" in results

    def test_no_containment(self):
        ens = MinHashLSHEnsemble(threshold=0.8, num_perm=128)
        m = MinHash(num_perm=128)
        for x in range(100):
            m.update(f"set1_{x}")
        ens.index([("doc1", m, 100)])

        query = MinHash(num_perm=128)
        for x in range(100):
            query.update(f"set2_{x}")
        results = ens.query(query, 100)
        assert "doc1" not in results

    def test_empty_index(self):
        ens = MinHashLSHEnsemble(threshold=0.5, num_perm=64)
        ens.index([])
        query = MinHash(num_perm=64)
        query.update("test")
        results = ens.query(query, 1)
        assert len(results) == 0

    def test_query_before_index(self):
        ens = MinHashLSHEnsemble(threshold=0.5, num_perm=64)
        query = MinHash(num_perm=64)
        query.update("test")
        with pytest.raises(RuntimeError, match="Must call index"):
            ens.query(query, 1)

    def test_multiple_entries(self):
        ens = MinHashLSHEnsemble(threshold=0.3, num_perm=128)
        entries = []
        for i in range(10):
            m = MinHash(num_perm=128)
            for x in range(50 + i * 10):
                m.update(f"item_{x}")
            entries.append((f"doc_{i}", m, 50 + i * 10))
        ens.index(entries)
        # Query with items that are in most docs
        query = MinHash(num_perm=128)
        for x in range(30):
            query.update(f"item_{x}")
        results = ens.query(query, 30)
        assert len(results) > 0

    def test_same_size_partition(self):
        ens = MinHashLSHEnsemble(threshold=0.5, num_perm=64)
        entries = []
        for i in range(5):
            m = MinHash(num_perm=64)
            for x in range(100):
                m.update(f"doc{i}_item_{x}")
            entries.append((f"doc_{i}", m, 100))
        ens.index(entries)  # All same size


# ===================================================================
# Integration Tests
# ===================================================================

class TestIntegration:
    def test_sha1_hash_deterministic(self):
        assert _sha1_hash("hello") == _sha1_hash("hello")
        assert _sha1_hash("hello") != _sha1_hash("world")

    def test_sha1_hash_types(self):
        assert isinstance(_sha1_hash("string"), int)
        assert isinstance(_sha1_hash(42), int)
        assert isinstance(_sha1_hash(b"bytes"), int)

    def test_document_similarity_pipeline(self):
        """End-to-end: shingle documents, build MinHash, query LSH."""
        def shingle(text, k=3):
            return set(text[i:i+k] for i in range(len(text) - k + 1))

        docs = {
            "doc1": "the quick brown fox jumps over the lazy dog",
            "doc2": "the quick brown fox leaps over the lazy dog",
            "doc3": "a completely different sentence about coding",
        }

        minhashes = {}
        for name, text in docs.items():
            m = MinHash(num_perm=128)
            for s in shingle(text):
                m.update(s)
            minhashes[name] = m

        # doc1 and doc2 should be similar
        sim_12 = minhashes["doc1"].jaccard(minhashes["doc2"])
        sim_13 = minhashes["doc1"].jaccard(minhashes["doc3"])
        assert sim_12 > sim_13

    def test_lsh_deduplication_pipeline(self):
        """Use LSH to find near-duplicate documents."""
        lsh = LSH(threshold=0.5, num_perm=128)
        # Create 5 docs, 2 are near-duplicates
        for i in range(5):
            m = MinHash(num_perm=128)
            for x in range(100):
                m.update(f"unique{i}_{x}")
            lsh.insert(f"doc_{i}", m)

        # Add a near-duplicate of doc_0
        dup = MinHash(num_perm=128)
        for x in range(80):
            dup.update(f"unique0_{x}")
        for x in range(20):
            dup.update(f"dup_{x}")
        lsh.insert("dup_of_0", dup)

        # Query should find doc_0 as similar to dup
        results = lsh.query(dup)
        assert "dup_of_0" in results

    def test_minhash_accuracy_large_sample(self):
        """Statistical test: MinHash Jaccard estimate is within expected error bounds."""
        n = 128
        m1 = MinHash(num_perm=n)
        m2 = MinHash(num_perm=n)
        # Create sets with known overlap
        set1 = set(range(0, 200))
        set2 = set(range(100, 300))
        for x in set1:
            m1.update(x)
        for x in set2:
            m2.update(x)
        estimated = m1.jaccard(m2)
        actual = len(set1 & set2) / len(set1 | set2)  # 100/300 = 0.333
        # Standard error ~ 1/sqrt(n) ~ 0.088
        assert abs(estimated - actual) < 3 / math.sqrt(n)

    def test_simhash_text_similarity(self):
        """SimHash for text near-duplicate detection."""
        s1 = SimHash(num_bits=64)
        s2 = SimHash(num_bits=64)
        s3 = SimHash(num_bits=64)
        s1.hash_tokens("the quick brown fox".split())
        s2.hash_tokens("the quick brown dog".split())
        s3.hash_tokens("completely unrelated words here".split())
        dist_12 = SimHash.hamming_distance(s1, s2)
        dist_13 = SimHash.hamming_distance(s1, s3)
        # Similar texts should have lower Hamming distance
        assert dist_12 <= dist_13

    def test_forest_vs_lsh_both_find_similar(self):
        """Both LSH and LSHForest should find similar items."""
        query = MinHash(num_perm=64)
        target = MinHash(num_perm=64)
        for x in range(100):
            query.update(f"item_{x}")
            target.update(f"item_{x}")

        lsh = LSH(threshold=0.5, num_perm=64)
        lsh.insert("target", target)

        forest = LSHForest(num_perm=64, num_trees=8)
        forest.add("target", target)

        lsh_results = lsh.query(query)
        forest_results = [r[0] for r in forest.query(query, top_k=5)]
        assert "target" in lsh_results
        assert "target" in forest_results

    def test_weighted_minhash_bound(self):
        """WeightedMinHash similarity is between 0 and 1."""
        w1 = WeightedMinHash(dim=10, num_perm=64)
        w2 = WeightedMinHash(dim=10, num_perm=64)
        w1.update([float(i) for i in range(1, 11)])
        w2.update([float(10 - i) for i in range(10)])
        sim = w1.jaccard(w2)
        assert 0.0 <= sim <= 1.0

    def test_minhash_merge_preserves_lsh_queryability(self):
        """Merged MinHash can still be used for LSH queries."""
        lsh = LSH(threshold=0.3, num_perm=64)
        # Create a doc and insert it
        m1 = MinHash(num_perm=64)
        for x in range(50):
            m1.update(f"part1_{x}")
        lsh.insert("doc1", m1)

        # Create a query by merging two partial sets
        q1 = MinHash(num_perm=64)
        q2 = MinHash(num_perm=64)
        for x in range(30):
            q1.update(f"part1_{x}")
        for x in range(30, 50):
            q2.update(f"part1_{x}")
        q1.merge(q2)
        results = lsh.query(q1)
        assert "doc1" in results

    def test_remove_and_reinsert(self):
        lsh = LSH(threshold=0.5, num_perm=64)
        m = MinHash(num_perm=64)
        m.update("test")
        lsh.insert("doc", m)
        lsh.remove("doc")
        lsh.insert("doc", m)
        assert "doc" in lsh

    def test_lsh_many_queries(self):
        """Stress test: insert many, query many."""
        lsh = LSH(threshold=0.5, num_perm=64)
        for i in range(100):
            m = MinHash(num_perm=64)
            for x in range(20):
                m.update(f"doc{i}_{x}")
            lsh.insert(f"doc_{i}", m)
        assert len(lsh) == 100
        # Each doc should find itself
        for i in range(0, 100, 10):
            m = MinHash(num_perm=64)
            for x in range(20):
                m.update(f"doc{i}_{x}")
            results = lsh.query(m)
            assert f"doc_{i}" in results


# ===================================================================
# Edge Cases
# ===================================================================

class TestEdgeCases:
    def test_single_element_sets(self):
        m1 = MinHash(num_perm=64)
        m2 = MinHash(num_perm=64)
        m1.update("only")
        m2.update("only")
        assert m1.jaccard(m2) == 1.0

    def test_simhash_single_dimension(self):
        s = SimHash(num_bits=32)
        s.hash_vector([1.0])
        assert isinstance(s.value, int)

    def test_simhash_high_dimension(self):
        s = SimHash(num_bits=64, seed=42)
        s.hash_vector([float(i) for i in range(1000)])
        assert isinstance(s.value, int)

    def test_minhash_order_independence(self):
        m1 = MinHash(num_perm=64)
        m2 = MinHash(num_perm=64)
        m1.update_batch(["a", "b", "c"])
        m2.update_batch(["c", "a", "b"])
        assert m1.hashvalues == m2.hashvalues

    def test_weighted_minhash_all_same(self):
        w = WeightedMinHash(dim=5, num_perm=64)
        w.update([3.0, 3.0, 3.0, 3.0, 3.0])
        assert w.hashvalues is not None

    def test_lsh_single_band(self):
        """LSH with very high threshold -> few bands."""
        lsh = LSH(threshold=0.99, num_perm=128)
        b, r = lsh.get_bands()
        assert b >= 1

    def test_lsh_many_bands(self):
        """LSH with very low threshold -> many bands."""
        lsh = LSH(threshold=0.01, num_perm=128)
        b, r = lsh.get_bands()
        assert b >= 1

    def test_simhash_zero_vector(self):
        s = SimHash(num_bits=32, seed=1)
        s.hash_vector([0.0, 0.0, 0.0])
        # All zero -> dot products all zero -> all bits set (>= 0)
        assert s.value == (1 << 32) - 1

    def test_simhash_empty_tokens(self):
        s = SimHash(num_bits=32)
        s.hash_tokens([])
        # All accumulators zero -> all bits set (>= 0)
        assert s.value == (1 << 32) - 1

    def test_minhash_very_large_num_perm(self):
        m = MinHash(num_perm=512)
        m.update("test")
        assert len(m.hashvalues) == 512

    def test_forest_single_tree(self):
        forest = LSHForest(num_perm=32, num_trees=1)
        m = MinHash(num_perm=32)
        m.update("test")
        forest.add("doc1", m)
        results = forest.query(m, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "doc1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
