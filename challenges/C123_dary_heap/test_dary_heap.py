"""Tests for C123: D-ary Heap."""
import pytest
from dary_heap import (
    DaryHeap, MaxDaryHeap, DaryHeapMap, MergeableDaryHeap,
    DaryHeapPQ, MedianDaryHeap,
    heap_sort, k_smallest, k_largest, merge_sorted, nsmallest, nlargest,
)


# ============================================================
# DaryHeap (min-heap)
# ============================================================

class TestDaryHeap:
    def test_basic_push_pop(self):
        h = DaryHeap(d=2)
        h.push(5)
        h.push(3)
        h.push(7)
        assert h.pop() == 3
        assert h.pop() == 5
        assert h.pop() == 7

    def test_default_d_is_4(self):
        h = DaryHeap()
        assert h.d == 4

    def test_various_d_values(self):
        for d in [2, 3, 4, 5, 8, 16]:
            h = DaryHeap(d=d)
            items = [10, 4, 7, 1, 9, 3, 6, 2, 8, 5]
            for x in items:
                h.push(x)
            result = [h.pop() for _ in range(len(h))]
            assert result == sorted(items), f"Failed for d={d}"

    def test_d_must_be_at_least_2(self):
        with pytest.raises(ValueError):
            DaryHeap(d=1)

    def test_build_from_items(self):
        h = DaryHeap(d=3, items=[5, 1, 8, 3, 2])
        assert h.pop() == 1
        assert h.pop() == 2
        assert h.pop() == 3

    def test_peek(self):
        h = DaryHeap(d=4, items=[5, 1, 3])
        assert h.peek() == 1
        assert len(h) == 3  # peek doesn't remove

    def test_peek_empty(self):
        with pytest.raises(IndexError):
            DaryHeap().peek()

    def test_pop_empty(self):
        with pytest.raises(IndexError):
            DaryHeap().pop()

    def test_pushpop_smaller(self):
        h = DaryHeap(d=2, items=[3, 5, 7])
        result = h.pushpop(1)
        assert result == 1
        assert h.peek() == 3

    def test_pushpop_larger(self):
        h = DaryHeap(d=2, items=[3, 5, 7])
        result = h.pushpop(4)
        assert result == 3
        assert h.peek() == 4

    def test_pushpop_empty(self):
        h = DaryHeap()
        assert h.pushpop(5) == 5

    def test_replace(self):
        h = DaryHeap(d=2, items=[1, 5, 7])
        old = h.replace(3)
        assert old == 1
        assert h.peek() == 3

    def test_replace_empty(self):
        with pytest.raises(IndexError):
            DaryHeap().replace(5)

    def test_len(self):
        h = DaryHeap()
        assert len(h) == 0
        h.push(1)
        assert len(h) == 1
        h.push(2)
        assert len(h) == 2
        h.pop()
        assert len(h) == 1

    def test_bool(self):
        h = DaryHeap()
        assert not h
        h.push(1)
        assert h

    def test_iter_sorted(self):
        h = DaryHeap(d=3, items=[5, 1, 8, 3, 2])
        assert list(h) == [1, 2, 3, 5, 8]
        assert len(h) == 5  # iteration doesn't modify original

    def test_repr(self):
        h = DaryHeap(d=3, items=[1, 2])
        assert "d=3" in repr(h)
        assert "size=2" in repr(h)

    def test_single_element(self):
        h = DaryHeap(d=2)
        h.push(42)
        assert h.pop() == 42
        assert len(h) == 0

    def test_duplicates(self):
        h = DaryHeap(d=2, items=[3, 1, 3, 1, 2])
        result = list(h)
        assert result == [1, 1, 2, 3, 3]

    def test_negative_values(self):
        h = DaryHeap(d=4, items=[-5, 3, -1, 7, -3])
        assert h.pop() == -5
        assert h.pop() == -3
        assert h.pop() == -1

    def test_large_d(self):
        h = DaryHeap(d=100, items=list(range(50, 0, -1)))
        result = list(h)
        assert result == list(range(1, 51))

    def test_stress_random_order(self):
        import random
        random.seed(123)
        items = random.sample(range(1000), 200)
        h = DaryHeap(d=5, items=items)
        result = list(h)
        assert result == sorted(items)


# ============================================================
# MaxDaryHeap
# ============================================================

class TestMaxDaryHeap:
    def test_basic(self):
        h = MaxDaryHeap(d=2, items=[5, 1, 8, 3])
        assert h.pop() == 8
        assert h.pop() == 5
        assert h.pop() == 3
        assert h.pop() == 1

    def test_various_d(self):
        for d in [2, 3, 4, 8]:
            h = MaxDaryHeap(d=d, items=[10, 4, 7, 1, 9])
            result = [h.pop() for _ in range(len(h))]
            assert result == sorted([10, 4, 7, 1, 9], reverse=True)

    def test_push_pop(self):
        h = MaxDaryHeap(d=3)
        h.push(3)
        h.push(7)
        h.push(1)
        assert h.pop() == 7

    def test_peek(self):
        h = MaxDaryHeap(d=4, items=[1, 9, 3])
        assert h.peek() == 9

    def test_pushpop_larger(self):
        h = MaxDaryHeap(d=2, items=[3, 5, 7])
        result = h.pushpop(10)
        assert result == 10
        assert h.peek() == 7

    def test_pushpop_smaller(self):
        h = MaxDaryHeap(d=2, items=[3, 5, 7])
        result = h.pushpop(4)
        assert result == 7
        assert h.peek() == 5

    def test_replace(self):
        h = MaxDaryHeap(d=2, items=[3, 5, 7])
        old = h.replace(10)
        assert old == 7
        assert h.peek() == 10

    def test_iter(self):
        h = MaxDaryHeap(d=3, items=[5, 1, 8, 3])
        assert list(h) == [8, 5, 3, 1]

    def test_d_validation(self):
        with pytest.raises(ValueError):
            MaxDaryHeap(d=0)

    def test_empty_errors(self):
        h = MaxDaryHeap()
        with pytest.raises(IndexError):
            h.pop()
        with pytest.raises(IndexError):
            h.peek()
        with pytest.raises(IndexError):
            h.replace(1)

    def test_duplicates(self):
        h = MaxDaryHeap(d=2, items=[3, 3, 1, 1])
        assert list(h) == [3, 3, 1, 1]


# ============================================================
# DaryHeapMap (indexed heap with decrease-key)
# ============================================================

class TestDaryHeapMap:
    def test_push_pop(self):
        h = DaryHeapMap(d=4)
        h.push("a", 5)
        h.push("b", 3)
        h.push("c", 7)
        key, pri = h.pop()
        assert key == "b"
        assert pri == 3

    def test_decrease_key(self):
        h = DaryHeapMap(d=2)
        h.push("a", 10)
        h.push("b", 5)
        h.push("c", 8)
        h.decrease_key("a", 1)
        key, pri = h.pop()
        assert key == "a"
        assert pri == 1

    def test_decrease_key_cannot_increase(self):
        h = DaryHeapMap(d=2)
        h.push("a", 5)
        with pytest.raises(ValueError):
            h.decrease_key("a", 10)

    def test_decrease_key_not_found(self):
        h = DaryHeapMap(d=2)
        with pytest.raises(KeyError):
            h.decrease_key("x", 1)

    def test_update_key_increase(self):
        h = DaryHeapMap(d=2)
        h.push("a", 1)
        h.push("b", 5)
        h.push("c", 3)
        h.update_key("a", 10)
        key, _ = h.pop()
        assert key == "c"

    def test_update_key_decrease(self):
        h = DaryHeapMap(d=2)
        h.push("a", 10)
        h.push("b", 5)
        h.update_key("a", 1)
        key, _ = h.pop()
        assert key == "a"

    def test_remove(self):
        h = DaryHeapMap(d=4)
        h.push("a", 1)
        h.push("b", 2)
        h.push("c", 3)
        h.remove("a")
        assert len(h) == 2
        key, pri = h.pop()
        assert key == "b"
        assert pri == 2

    def test_remove_last(self):
        h = DaryHeapMap(d=2)
        h.push("a", 1)
        h.remove("a")
        assert len(h) == 0

    def test_remove_not_found(self):
        h = DaryHeapMap(d=2)
        with pytest.raises(KeyError):
            h.remove("x")

    def test_contains(self):
        h = DaryHeapMap(d=2)
        h.push("a", 1)
        assert "a" in h
        assert "b" not in h
        h.pop()
        assert "a" not in h

    def test_get_priority(self):
        h = DaryHeapMap(d=2)
        h.push("a", 5)
        assert h.get_priority("a") == 5
        h.decrease_key("a", 2)
        assert h.get_priority("a") == 2

    def test_get_priority_not_found(self):
        h = DaryHeapMap(d=2)
        with pytest.raises(KeyError):
            h.get_priority("x")

    def test_duplicate_key_push(self):
        h = DaryHeapMap(d=2)
        h.push("a", 1)
        with pytest.raises(KeyError):
            h.push("a", 2)

    def test_peek(self):
        h = DaryHeapMap(d=4)
        h.push("x", 3)
        h.push("y", 1)
        key, pri = h.peek()
        assert key == "y"
        assert pri == 1
        assert len(h) == 2

    def test_peek_empty(self):
        with pytest.raises(IndexError):
            DaryHeapMap().peek()

    def test_pop_empty(self):
        with pytest.raises(IndexError):
            DaryHeapMap().pop()

    def test_dijkstra_simulation(self):
        """Simulate Dijkstra's shortest path with decrease-key."""
        h = DaryHeapMap(d=4)
        # Graph: A->B(1), A->C(4), B->C(2), B->D(5), C->D(1)
        h.push("A", 0)
        h.push("B", float('inf'))
        h.push("C", float('inf'))
        h.push("D", float('inf'))

        # Process A (dist=0)
        node, dist = h.pop()
        assert node == "A"
        h.decrease_key("B", 1)  # A->B cost 1
        h.decrease_key("C", 4)  # A->C cost 4

        # Process B (dist=1)
        node, dist = h.pop()
        assert node == "B"
        assert dist == 1
        h.decrease_key("C", 3)  # B->C cost 1+2=3

        # Process C (dist=3)
        node, dist = h.pop()
        assert node == "C"
        assert dist == 3

        # Process D
        h.decrease_key("D", 4)  # C->D cost 3+1=4
        node, dist = h.pop()
        assert node == "D"
        assert dist == 4

    def test_many_operations(self):
        h = DaryHeapMap(d=3)
        for i in range(50):
            h.push(f"k{i}", 100 - i)
        assert len(h) == 50
        h.decrease_key("k0", 0)
        key, pri = h.pop()
        assert key == "k0"
        assert pri == 0

    def test_remove_middle_element(self):
        """Remove an element that's neither root nor last."""
        h = DaryHeapMap(d=2)
        for i in range(10):
            h.push(f"k{i}", i)
        h.remove("k5")
        assert len(h) == 9
        assert "k5" not in h
        # All other elements still extractable in order
        result = []
        while h:
            k, p = h.pop()
            result.append(p)
        expected = [i for i in range(10) if i != 5]
        assert result == expected


# ============================================================
# MergeableDaryHeap
# ============================================================

class TestMergeableDaryHeap:
    def test_basic(self):
        h = MergeableDaryHeap(d=4, items=[5, 1, 3])
        assert h.pop() == 1

    def test_merge(self):
        h1 = MergeableDaryHeap(d=3, items=[5, 1, 3])
        h2 = MergeableDaryHeap(d=3, items=[4, 2, 6])
        h1.merge(h2)
        assert list(h1) == [1, 2, 3, 4, 5, 6]

    def test_merge_empty(self):
        h1 = MergeableDaryHeap(d=2, items=[3, 1])
        h2 = MergeableDaryHeap(d=2)
        h1.merge(h2)
        assert list(h1) == [1, 3]

    def test_merge_into_empty(self):
        h1 = MergeableDaryHeap(d=2)
        h2 = MergeableDaryHeap(d=2, items=[3, 1])
        h1.merge(h2)
        assert list(h1) == [1, 3]

    def test_merge_list(self):
        h = MergeableDaryHeap(d=2, items=[5, 3])
        h.merge([1, 7])
        assert list(h) == [1, 3, 5, 7]

    def test_merge_many(self):
        h = MergeableDaryHeap(d=4, items=[10])
        h.merge_many([5, 7], [1, 3], [2, 8])
        assert list(h) == [1, 2, 3, 5, 7, 8, 10]

    def test_split(self):
        h = MergeableDaryHeap(d=2, items=[1, 2, 3, 4, 5, 6])
        left, right = h.split()
        combined = sorted(list(left) + list(right))
        assert combined == [1, 2, 3, 4, 5, 6]

    def test_split_single(self):
        h = MergeableDaryHeap(d=2, items=[42])
        left, right = h.split()
        assert len(left) + len(right) == 1

    def test_push_pop(self):
        h = MergeableDaryHeap(d=3)
        h.push(5)
        h.push(3)
        assert h.pop() == 3
        assert h.pop() == 5

    def test_peek(self):
        h = MergeableDaryHeap(d=2, items=[3, 1, 5])
        assert h.peek() == 1

    def test_empty_errors(self):
        h = MergeableDaryHeap()
        with pytest.raises(IndexError):
            h.pop()
        with pytest.raises(IndexError):
            h.peek()


# ============================================================
# DaryHeapPQ
# ============================================================

class TestDaryHeapPQ:
    def test_enqueue_dequeue(self):
        pq = DaryHeapPQ(d=4)
        pq.enqueue("task1", 5)
        pq.enqueue("task2", 1)
        pq.enqueue("task3", 3)
        name, pri = pq.dequeue()
        assert name == "task2"
        assert pri == 1

    def test_peek(self):
        pq = DaryHeapPQ(d=2)
        pq.enqueue("a", 3)
        pq.enqueue("b", 1)
        name, pri = pq.peek()
        assert name == "b"
        assert len(pq) == 2

    def test_update_priority(self):
        pq = DaryHeapPQ(d=4)
        pq.enqueue("a", 10)
        pq.enqueue("b", 5)
        pq.update_priority("a", 1)
        name, _ = pq.dequeue()
        assert name == "a"

    def test_remove(self):
        pq = DaryHeapPQ(d=2)
        pq.enqueue("a", 1)
        pq.enqueue("b", 2)
        pq.enqueue("c", 3)
        pq.remove("a")
        name, _ = pq.dequeue()
        assert name == "b"

    def test_contains(self):
        pq = DaryHeapPQ(d=4)
        pq.enqueue("x", 1)
        assert "x" in pq
        assert "y" not in pq

    def test_empty_errors(self):
        pq = DaryHeapPQ()
        with pytest.raises(IndexError):
            pq.dequeue()
        with pytest.raises(IndexError):
            pq.peek()

    def test_len_bool(self):
        pq = DaryHeapPQ()
        assert len(pq) == 0
        assert not pq
        pq.enqueue("a", 1)
        assert len(pq) == 1
        assert pq

    def test_repr(self):
        pq = DaryHeapPQ(d=3)
        assert "d=3" in repr(pq)


# ============================================================
# MedianDaryHeap
# ============================================================

class TestMedianDaryHeap:
    def test_single(self):
        m = MedianDaryHeap(d=2)
        m.push(5)
        assert m.median() == 5

    def test_two_elements(self):
        m = MedianDaryHeap(d=2)
        m.push(1)
        m.push(3)
        assert m.median() == 2.0

    def test_three_elements(self):
        m = MedianDaryHeap(d=4)
        m.push(1)
        m.push(5)
        m.push(3)
        assert m.median() == 3

    def test_sequential(self):
        m = MedianDaryHeap(d=3)
        m.push(1)
        assert m.median() == 1
        m.push(2)
        assert m.median() == 1.5
        m.push(3)
        assert m.median() == 2
        m.push(4)
        assert m.median() == 2.5
        m.push(5)
        assert m.median() == 3

    def test_pop_median(self):
        m = MedianDaryHeap(d=2)
        m.push(1)
        m.push(3)
        m.push(5)
        val = m.pop_median()
        assert val == 3
        assert len(m) == 2

    def test_pop_median_rebalance(self):
        m = MedianDaryHeap(d=4)
        for x in [5, 1, 3, 7, 2]:
            m.push(x)
        # Median should be 3
        assert m.median() == 3
        m.pop_median()
        # After removing 3, remaining: 1,2,5,7 -> median = 3.5
        assert m.median() == 3.5

    def test_empty_errors(self):
        m = MedianDaryHeap()
        with pytest.raises(IndexError):
            m.median()
        with pytest.raises(IndexError):
            m.pop_median()

    def test_len_bool(self):
        m = MedianDaryHeap()
        assert len(m) == 0
        assert not m
        m.push(1)
        assert len(m) == 1
        assert m

    def test_all_same(self):
        m = MedianDaryHeap(d=2)
        for _ in range(5):
            m.push(7)
        assert m.median() == 7

    def test_descending_insert(self):
        m = MedianDaryHeap(d=4)
        for x in [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
            m.push(x)
        assert m.median() == 5.5

    def test_d_validation(self):
        with pytest.raises(ValueError):
            MedianDaryHeap(d=1)


# ============================================================
# Utility Functions
# ============================================================

class TestHeapSort:
    def test_basic(self):
        assert heap_sort([5, 3, 1, 4, 2]) == [1, 2, 3, 4, 5]

    def test_reverse(self):
        assert heap_sort([5, 3, 1, 4, 2], reverse=True) == [5, 4, 3, 2, 1]

    def test_empty(self):
        assert heap_sort([]) == []

    def test_single(self):
        assert heap_sort([42]) == [42]

    def test_various_d(self):
        items = [7, 2, 5, 1, 9, 3]
        for d in [2, 3, 4, 8]:
            assert heap_sort(items, d=d) == sorted(items)

    def test_duplicates(self):
        assert heap_sort([3, 1, 3, 1, 2]) == [1, 1, 2, 3, 3]

    def test_already_sorted(self):
        assert heap_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]

    def test_reverse_sorted(self):
        assert heap_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


class TestKSmallest:
    def test_basic(self):
        assert k_smallest([5, 3, 1, 4, 2], 3) == [1, 2, 3]

    def test_k_larger_than_list(self):
        assert k_smallest([3, 1, 2], 5) == [1, 2, 3]

    def test_k_zero(self):
        assert k_smallest([1, 2, 3], 0) == []

    def test_empty(self):
        assert k_smallest([], 3) == []


class TestKLargest:
    def test_basic(self):
        assert k_largest([5, 3, 1, 4, 2], 3) == [5, 4, 3]

    def test_k_larger_than_list(self):
        assert k_largest([3, 1, 2], 5) == [3, 2, 1]

    def test_k_zero(self):
        assert k_largest([1, 2, 3], 0) == []


class TestMergeSorted:
    def test_basic(self):
        result = merge_sorted([1, 4, 7], [2, 5, 8], [3, 6, 9])
        assert result == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_empty_lists(self):
        result = merge_sorted([], [1, 2], [])
        assert result == [1, 2]

    def test_single_list(self):
        result = merge_sorted([1, 2, 3])
        assert result == [1, 2, 3]

    def test_all_empty(self):
        result = merge_sorted([], [], [])
        assert result == []

    def test_no_args(self):
        result = merge_sorted()
        assert result == []

    def test_duplicates_across(self):
        result = merge_sorted([1, 3, 5], [1, 3, 5])
        assert result == [1, 1, 3, 3, 5, 5]


class TestNSmallest:
    def test_basic(self):
        assert nsmallest([5, 3, 1, 4, 2], 3) == [1, 2, 3]

    def test_n_zero(self):
        assert nsmallest([1, 2, 3], 0) == []

    def test_n_larger(self):
        result = nsmallest([3, 1, 2], 5)
        assert result == [1, 2, 3]

    def test_large_list_small_n(self):
        items = list(range(100, 0, -1))
        result = nsmallest(items, 5)
        assert result == [1, 2, 3, 4, 5]


class TestNLargest:
    def test_basic(self):
        assert nlargest([5, 3, 1, 4, 2], 3) == [5, 4, 3]

    def test_n_zero(self):
        assert nlargest([1, 2, 3], 0) == []

    def test_n_larger(self):
        result = nlargest([3, 1, 2], 5)
        assert result == [3, 2, 1]

    def test_large_list_small_n(self):
        items = list(range(1, 101))
        result = nlargest(items, 5)
        assert result == [100, 99, 98, 97, 96]


# ============================================================
# Structural / Property-based tests
# ============================================================

class TestHeapProperty:
    def test_min_heap_property_after_operations(self):
        """Verify heap property holds after mixed push/pop."""
        h = DaryHeap(d=3)
        import random
        random.seed(42)
        for _ in range(100):
            h.push(random.randint(0, 1000))
        for _ in range(50):
            h.pop()
        for _ in range(50):
            h.push(random.randint(0, 1000))
        # Verify heap property
        data = h._data
        d = h.d
        for i in range(len(data)):
            for c in range(i * d + 1, min(i * d + d + 1, len(data))):
                assert data[i] <= data[c], f"Heap property violated at {i}->{c}"

    def test_max_heap_property(self):
        h = MaxDaryHeap(d=4)
        import random
        random.seed(99)
        for _ in range(100):
            h.push(random.randint(0, 1000))
        data = h._data
        d = h.d
        for i in range(len(data)):
            for c in range(i * d + 1, min(i * d + d + 1, len(data))):
                assert data[i] >= data[c], f"Max heap property violated at {i}->{c}"

    def test_index_map_consistency(self):
        """DaryHeapMap index map matches actual positions."""
        h = DaryHeapMap(d=3)
        import random
        random.seed(77)
        keys = [f"k{i}" for i in range(50)]
        for k in keys:
            h.push(k, random.randint(0, 100))
        # Verify index map
        for key, idx in h._index.items():
            assert h._data[idx][1] == key, f"Index mismatch for {key}"
        # Remove some
        for k in keys[:20]:
            h.remove(k)
        for key, idx in h._index.items():
            assert h._data[idx][1] == key, f"Index mismatch after remove for {key}"

    def test_d2_matches_binary_heap(self):
        """d=2 should behave identically to a binary heap."""
        import random
        random.seed(55)
        items = random.sample(range(500), 100)
        h = DaryHeap(d=2, items=items)
        result = list(h)
        assert result == sorted(items)

    def test_pushpop_maintains_property(self):
        h = DaryHeap(d=4, items=[2, 5, 8, 3, 7])
        h.pushpop(4)
        data = h._data
        d = h.d
        for i in range(len(data)):
            for c in range(i * d + 1, min(i * d + d + 1, len(data))):
                assert data[i] <= data[c]


# ============================================================
# Edge cases and integration
# ============================================================

class TestEdgeCases:
    def test_all_same_values(self):
        h = DaryHeap(d=3, items=[7, 7, 7, 7, 7])
        assert list(h) == [7, 7, 7, 7, 7]

    def test_two_elements_heap(self):
        h = DaryHeap(d=8, items=[2, 1])
        assert h.pop() == 1
        assert h.pop() == 2

    def test_large_d_small_heap(self):
        h = DaryHeap(d=1000, items=[3, 1, 2])
        assert list(h) == [1, 2, 3]

    def test_merge_then_extract_all(self):
        h1 = MergeableDaryHeap(d=4, items=[1, 5, 9])
        h2 = MergeableDaryHeap(d=4, items=[2, 6, 10])
        h3 = MergeableDaryHeap(d=4, items=[3, 7, 11])
        h1.merge_many(h2, h3)
        assert list(h1) == [1, 2, 3, 5, 6, 7, 9, 10, 11]

    def test_median_large_sequence(self):
        m = MedianDaryHeap(d=4)
        for x in range(1, 101):
            m.push(x)
        assert m.median() == 50.5

    def test_heap_sort_stability_with_tuples(self):
        items = [(3, 'c'), (1, 'a'), (2, 'b'), (1, 'x')]
        result = heap_sort(items)
        assert result == [(1, 'a'), (1, 'x'), (2, 'b'), (3, 'c')]

    def test_pq_full_workflow(self):
        pq = DaryHeapPQ(d=4)
        pq.enqueue("low", 10)
        pq.enqueue("high", 1)
        pq.enqueue("mid", 5)
        pq.update_priority("low", 0)
        name, _ = pq.dequeue()
        assert name == "low"
        pq.remove("mid")
        name, _ = pq.dequeue()
        assert name == "high"
        assert len(pq) == 0

    def test_heapmap_full_cycle(self):
        h = DaryHeapMap(d=4)
        h.push("a", 10)
        h.push("b", 20)
        h.push("c", 30)
        h.update_key("c", 5)
        h.decrease_key("b", 1)
        k1, p1 = h.pop()
        assert k1 == "b" and p1 == 1
        k2, p2 = h.pop()
        assert k2 == "c" and p2 == 5
        k3, p3 = h.pop()
        assert k3 == "a" and p3 == 10
