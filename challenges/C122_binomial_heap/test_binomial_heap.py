"""Tests for C122: Binomial Heap"""

import pytest
from binomial_heap import (
    BinomialNode, BinomialHeap, MaxBinomialHeap, BinomialHeapMap,
    MergeableBinomialHeap, BinomialHeapPQ, LazyBinomialHeap,
    heap_sort, k_smallest, merge_sorted_streams,
    _link, _merge_root_lists, _union, _reverse_children,
)


# === BinomialNode Tests ===

class TestBinomialNode:
    def test_create_node(self):
        n = BinomialNode(5)
        assert n.key == 5
        assert n.value == 5
        assert n.degree == 0
        assert n.parent is None
        assert n.child is None
        assert n.sibling is None

    def test_create_node_with_value(self):
        n = BinomialNode(3, "hello")
        assert n.key == 3
        assert n.value == "hello"

    def test_repr(self):
        n = BinomialNode(7, "x")
        assert "7" in repr(n)
        assert "x" in repr(n)


# === Internal Function Tests ===

class TestInternalFunctions:
    def test_link_two_nodes(self):
        a = BinomialNode(3)
        b = BinomialNode(5)
        result = _link(b, a)  # b becomes child of a
        assert result is a
        assert a.child is b
        assert b.parent is a
        assert a.degree == 1

    def test_merge_root_lists_empty(self):
        assert _merge_root_lists(None, None) is None
        n = BinomialNode(1)
        assert _merge_root_lists(n, None) is n
        assert _merge_root_lists(None, n) is n

    def test_merge_root_lists_sorted(self):
        a = BinomialNode(1)
        a.degree = 0
        b = BinomialNode(2)
        b.degree = 1
        c = BinomialNode(3)
        c.degree = 2

        a.sibling = c  # list1: degree 0, 2
        b_head = b      # list2: degree 1

        result = _merge_root_lists(a, b_head)
        # Should be sorted by degree: 0, 1, 2
        assert result.degree == 0
        assert result.sibling.degree == 1
        assert result.sibling.sibling.degree == 2

    def test_reverse_children(self):
        a = BinomialNode(1)
        b = BinomialNode(2)
        c = BinomialNode(3)
        a.sibling = b
        b.sibling = c
        a.parent = BinomialNode(0)
        b.parent = BinomialNode(0)
        c.parent = BinomialNode(0)

        result = _reverse_children(a)
        assert result is c
        assert c.sibling is b
        assert b.sibling is a
        assert a.sibling is None
        # Parents cleared
        assert a.parent is None
        assert b.parent is None
        assert c.parent is None

    def test_reverse_children_none(self):
        assert _reverse_children(None) is None

    def test_union_empty(self):
        assert _union(None, None) is None
        n = BinomialNode(1)
        assert _union(n, None) is n
        assert _union(None, n) is n


# === BinomialHeap Tests ===

class TestBinomialHeap:
    def test_empty_heap(self):
        h = BinomialHeap()
        assert len(h) == 0
        assert h.is_empty()
        assert not h

    def test_insert_one(self):
        h = BinomialHeap()
        h.insert(5)
        assert len(h) == 1
        assert not h.is_empty()
        assert h

    def test_find_min(self):
        h = BinomialHeap()
        h.insert(5)
        h.insert(3)
        h.insert(7)
        assert h.find_min() == (3, 3)

    def test_peek_alias(self):
        h = BinomialHeap()
        h.insert(10)
        assert h.peek() == h.find_min()

    def test_find_min_empty(self):
        h = BinomialHeap()
        with pytest.raises(IndexError):
            h.find_min()

    def test_extract_min_single(self):
        h = BinomialHeap()
        h.insert(42)
        assert h.extract_min() == (42, 42)
        assert h.is_empty()

    def test_extract_min_multiple(self):
        h = BinomialHeap()
        for v in [5, 3, 7, 1, 4]:
            h.insert(v)
        assert h.extract_min() == (1, 1)
        assert h.extract_min() == (3, 3)
        assert h.extract_min() == (4, 4)
        assert h.extract_min() == (5, 5)
        assert h.extract_min() == (7, 7)
        assert h.is_empty()

    def test_extract_min_empty(self):
        h = BinomialHeap()
        with pytest.raises(IndexError):
            h.extract_min()

    def test_insert_and_extract_sorted(self):
        h = BinomialHeap()
        import random
        values = list(range(20))
        random.seed(42)
        random.shuffle(values)
        for v in values:
            h.insert(v)
        result = [h.extract_min()[0] for _ in range(20)]
        assert result == list(range(20))

    def test_insert_with_value(self):
        h = BinomialHeap()
        h.insert(1, "one")
        h.insert(2, "two")
        assert h.extract_min() == (1, "one")
        assert h.extract_min() == (2, "two")

    def test_decrease_key(self):
        h = BinomialHeap()
        n1 = h.insert(5)
        n2 = h.insert(3)
        n3 = h.insert(7)
        h.decrease_key(n3, 1)
        assert h.find_min() == (1, 7)

    def test_decrease_key_invalid(self):
        h = BinomialHeap()
        n = h.insert(5)
        with pytest.raises(ValueError):
            h.decrease_key(n, 10)

    def test_delete(self):
        h = BinomialHeap()
        n1 = h.insert(5)
        n2 = h.insert(3)
        n3 = h.insert(7)
        h.delete(n2)
        assert len(h) == 2
        assert h.extract_min() == (5, 5)

    def test_delete_min_node(self):
        h = BinomialHeap()
        n1 = h.insert(1)
        n2 = h.insert(2)
        n3 = h.insert(3)
        h.delete(n1)
        assert h.find_min() == (2, 2)

    def test_merge(self):
        h1 = BinomialHeap()
        h2 = BinomialHeap()
        for v in [3, 1, 5]:
            h1.insert(v)
        for v in [4, 2, 6]:
            h2.insert(v)
        h1.merge(h2)
        assert len(h1) == 6
        assert len(h2) == 0
        result = [h1.extract_min()[0] for _ in range(6)]
        assert result == [1, 2, 3, 4, 5, 6]

    def test_merge_empty(self):
        h1 = BinomialHeap()
        h2 = BinomialHeap()
        h1.insert(1)
        h1.merge(h2)
        assert len(h1) == 1

    def test_merge_into_empty(self):
        h1 = BinomialHeap()
        h2 = BinomialHeap()
        h2.insert(1)
        h1.merge(h2)
        assert len(h1) == 1
        assert h1.extract_min() == (1, 1)

    def test_merge_type_error(self):
        h = BinomialHeap()
        with pytest.raises(TypeError):
            h.merge("not a heap")

    def test_to_sorted_list(self):
        h = BinomialHeap()
        for v in [5, 3, 1, 4, 2]:
            h.insert(v)
        assert h.to_sorted_list() == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
        assert h.is_empty()

    def test_iter(self):
        h = BinomialHeap()
        for v in [3, 1, 2]:
            h.insert(v)
        items = list(h)
        assert sorted(items) == [(1, 1), (2, 2), (3, 3)]

    def test_contains(self):
        h = BinomialHeap()
        h.insert(3)
        h.insert(5)
        assert 3 in h
        assert 5 in h
        assert 7 not in h

    def test_large_heap(self):
        h = BinomialHeap()
        n = 1000
        import random
        random.seed(123)
        values = list(range(n))
        random.shuffle(values)
        for v in values:
            h.insert(v)
        result = [h.extract_min()[0] for _ in range(n)]
        assert result == list(range(n))

    def test_duplicate_keys(self):
        h = BinomialHeap()
        h.insert(3, "a")
        h.insert(3, "b")
        h.insert(3, "c")
        assert len(h) == 3
        results = [h.extract_min() for _ in range(3)]
        assert all(k == 3 for k, v in results)

    def test_negative_keys(self):
        h = BinomialHeap()
        h.insert(-5)
        h.insert(-10)
        h.insert(0)
        assert h.extract_min() == (-10, -10)

    def test_float_keys(self):
        h = BinomialHeap()
        h.insert(3.14)
        h.insert(2.71)
        h.insert(1.41)
        assert h.extract_min() == (1.41, 1.41)

    def test_decrease_key_to_equal(self):
        h = BinomialHeap()
        n = h.insert(5)
        h.decrease_key(n, 5)  # should be fine
        assert h.find_min() == (5, 5)

    def test_multiple_merges(self):
        heaps = []
        for i in range(5):
            h = BinomialHeap()
            for j in range(3):
                h.insert(i * 3 + j)
            heaps.append(h)
        main = heaps[0]
        for h in heaps[1:]:
            main.merge(h)
        assert len(main) == 15
        result = [main.extract_min()[0] for _ in range(15)]
        assert result == list(range(15))

    def test_interleaved_insert_extract(self):
        h = BinomialHeap()
        h.insert(5)
        h.insert(3)
        assert h.extract_min() == (3, 3)
        h.insert(1)
        h.insert(4)
        assert h.extract_min() == (1, 1)
        assert h.extract_min() == (4, 4)
        assert h.extract_min() == (5, 5)

    def test_structural_correctness_power_of_two(self):
        """After inserting 2^k items, should have single B_k tree."""
        h = BinomialHeap()
        for i in range(8):  # 2^3 = 8
            h.insert(i)
        # Root list should have a single tree of degree 3
        root = h._head
        assert root is not None
        assert root.sibling is None
        assert root.degree == 3


# === MaxBinomialHeap Tests ===

class TestMaxBinomialHeap:
    def test_empty(self):
        h = MaxBinomialHeap()
        assert len(h) == 0
        assert h.is_empty()

    def test_insert_and_find_max(self):
        h = MaxBinomialHeap()
        h.insert(3)
        h.insert(7)
        h.insert(5)
        assert h.find_max() == (7, 7)

    def test_peek(self):
        h = MaxBinomialHeap()
        h.insert(10)
        assert h.peek() == h.find_max()

    def test_extract_max(self):
        h = MaxBinomialHeap()
        for v in [3, 7, 1, 5, 9]:
            h.insert(v)
        assert h.extract_max() == (9, 9)
        assert h.extract_max() == (7, 7)
        assert h.extract_max() == (5, 5)

    def test_extract_max_empty(self):
        h = MaxBinomialHeap()
        with pytest.raises(IndexError):
            h.extract_max()

    def test_find_max_empty(self):
        h = MaxBinomialHeap()
        with pytest.raises(IndexError):
            h.find_max()

    def test_increase_key(self):
        h = MaxBinomialHeap()
        n1 = h.insert(5)
        n2 = h.insert(3)
        h.increase_key(n2, 10)
        assert h.find_max() == (10, 3)

    def test_increase_key_invalid(self):
        h = MaxBinomialHeap()
        n = h.insert(5)
        with pytest.raises(ValueError):
            h.increase_key(n, 2)

    def test_delete(self):
        h = MaxBinomialHeap()
        n1 = h.insert(5)
        n2 = h.insert(3)
        n3 = h.insert(7)
        h.delete(n3)
        assert h.find_max() == (5, 5)

    def test_merge(self):
        h1 = MaxBinomialHeap()
        h2 = MaxBinomialHeap()
        h1.insert(3)
        h1.insert(7)
        h2.insert(5)
        h2.insert(9)
        h1.merge(h2)
        assert len(h1) == 4
        assert h1.extract_max() == (9, 9)

    def test_merge_type_error(self):
        h = MaxBinomialHeap()
        with pytest.raises(TypeError):
            h.merge("nope")

    def test_to_sorted_list(self):
        h = MaxBinomialHeap()
        for v in [3, 1, 5, 2, 4]:
            h.insert(v)
        result = h.to_sorted_list()
        assert result == [(5, 5), (4, 4), (3, 3), (2, 2), (1, 1)]

    def test_iter(self):
        h = MaxBinomialHeap()
        for v in [1, 2, 3]:
            h.insert(v)
        items = sorted(list(h))
        assert items == [(1, 1), (2, 2), (3, 3)]

    def test_large_max_heap(self):
        h = MaxBinomialHeap()
        import random
        random.seed(99)
        values = list(range(100))
        random.shuffle(values)
        for v in values:
            h.insert(v)
        result = [h.extract_max()[0] for _ in range(100)]
        assert result == list(range(99, -1, -1))


# === BinomialHeapMap Tests ===

class TestBinomialHeapMap:
    def test_empty(self):
        m = BinomialHeapMap()
        assert len(m) == 0
        assert m.is_empty()

    def test_insert_and_find_min(self):
        m = BinomialHeapMap()
        m.insert("a", 5)
        m.insert("b", 3)
        m.insert("c", 7)
        assert m.find_min() == (3, "b")

    def test_peek(self):
        m = BinomialHeapMap()
        m.insert("x", 10)
        assert m.peek() == m.find_min()

    def test_duplicate_name_error(self):
        m = BinomialHeapMap()
        m.insert("a", 5)
        with pytest.raises(KeyError):
            m.insert("a", 3)

    def test_extract_min(self):
        m = BinomialHeapMap()
        m.insert("a", 5)
        m.insert("b", 3)
        m.insert("c", 7)
        assert m.extract_min() == (3, "b")
        assert "b" not in m
        assert len(m) == 2

    def test_decrease_key(self):
        m = BinomialHeapMap()
        m.insert("a", 5)
        m.insert("b", 3)
        m.decrease_key("a", 1)
        assert m.find_min() == (1, "a")

    def test_decrease_key_not_found(self):
        m = BinomialHeapMap()
        with pytest.raises(KeyError):
            m.decrease_key("x", 1)

    def test_delete(self):
        m = BinomialHeapMap()
        m.insert("a", 5)
        m.insert("b", 3)
        m.insert("c", 7)
        m.delete("b")
        assert len(m) == 2
        assert "b" not in m
        assert m.find_min() == (5, "a")

    def test_delete_not_found(self):
        m = BinomialHeapMap()
        with pytest.raises(KeyError):
            m.delete("x")

    def test_contains(self):
        m = BinomialHeapMap()
        m.insert("a", 5)
        assert "a" in m
        assert "b" not in m

    def test_get_priority(self):
        m = BinomialHeapMap()
        m.insert("a", 5)
        assert m.get_priority("a") == 5

    def test_get_priority_not_found(self):
        m = BinomialHeapMap()
        with pytest.raises(KeyError):
            m.get_priority("x")

    def test_iter(self):
        m = BinomialHeapMap()
        m.insert("a", 1)
        m.insert("b", 2)
        items = list(m)
        assert len(items) == 2

    def test_dijkstra_simulation(self):
        """Simulate Dijkstra's algorithm with decrease-key."""
        m = BinomialHeapMap()
        m.insert("A", 0)
        m.insert("B", 10)
        m.insert("C", 20)
        m.insert("D", 30)

        # Relax edges
        m.decrease_key("B", 5)
        m.decrease_key("C", 8)

        result = []
        while m:
            result.append(m.extract_min())
        assert result[0] == (0, "A")
        assert result[1] == (5, "B")
        assert result[2] == (8, "C")
        assert result[3] == (30, "D")


# === MergeableBinomialHeap Tests ===

class TestMergeableBinomialHeap:
    def test_basic(self):
        h = MergeableBinomialHeap("heap1")
        h.insert(5)
        h.insert(3)
        assert h.find_min() == (3, 3)
        assert h.peek() == (3, 3)

    def test_merge_tracking(self):
        h1 = MergeableBinomialHeap("alpha")
        h2 = MergeableBinomialHeap("beta")
        h1.insert(3)
        h2.insert(1)
        h1.merge(h2)
        assert h1.merge_history == ["beta"]
        assert h1.find_min() == (1, 1)

    def test_merge_chain(self):
        h1 = MergeableBinomialHeap("A")
        h2 = MergeableBinomialHeap("B")
        h3 = MergeableBinomialHeap("C")
        h1.insert(3)
        h2.insert(2)
        h3.insert(1)
        h2.merge(h3)
        h1.merge(h2)
        assert "B" in h1.merge_history
        assert "C" in h1.merge_history

    def test_merge_type_error(self):
        h = MergeableBinomialHeap()
        with pytest.raises(TypeError):
            h.merge("nope")

    def test_extract_min(self):
        h = MergeableBinomialHeap()
        h.insert(5)
        h.insert(3)
        assert h.extract_min() == (3, 3)

    def test_to_sorted_list(self):
        h = MergeableBinomialHeap()
        for v in [3, 1, 2]:
            h.insert(v)
        assert h.to_sorted_list() == [(1, 1), (2, 2), (3, 3)]

    def test_iter(self):
        h = MergeableBinomialHeap()
        h.insert(1)
        h.insert(2)
        items = sorted(list(h))
        assert items == [(1, 1), (2, 2)]

    def test_bool_and_len(self):
        h = MergeableBinomialHeap()
        assert not h
        h.insert(1)
        assert h
        assert len(h) == 1


# === BinomialHeapPQ Tests ===

class TestBinomialHeapPQ:
    def test_empty(self):
        pq = BinomialHeapPQ()
        assert len(pq) == 0
        assert pq.is_empty()
        assert not pq

    def test_push_pop(self):
        pq = BinomialHeapPQ()
        pq.push("task1", 3)
        pq.push("task2", 1)
        pq.push("task3", 2)
        assert pq.pop() == "task2"
        assert pq.pop() == "task3"
        assert pq.pop() == "task1"

    def test_peek(self):
        pq = BinomialHeapPQ()
        pq.push("a", 5)
        pq.push("b", 2)
        assert pq.peek() == "b"
        assert len(pq) == 2  # peek doesn't remove

    def test_pop_empty(self):
        pq = BinomialHeapPQ()
        with pytest.raises(IndexError):
            pq.pop()

    def test_peek_empty(self):
        pq = BinomialHeapPQ()
        with pytest.raises(IndexError):
            pq.peek()

    def test_fifo_tiebreaking(self):
        """Same priority: FIFO order."""
        pq = BinomialHeapPQ()
        pq.push("first", 1)
        pq.push("second", 1)
        pq.push("third", 1)
        assert pq.pop() == "first"
        assert pq.pop() == "second"
        assert pq.pop() == "third"

    def test_push_pop_combined(self):
        pq = BinomialHeapPQ()
        result = pq.push_pop("a", 5)
        assert result == "a"  # only item

    def test_push_pop_returns_min(self):
        pq = BinomialHeapPQ()
        pq.push("existing", 3)
        result = pq.push_pop("new", 5)
        assert result == "existing"

    def test_iter_priority_order(self):
        pq = BinomialHeapPQ()
        pq.push("c", 3)
        pq.push("a", 1)
        pq.push("b", 2)
        result = list(pq)
        assert result == ["a", "b", "c"]

    def test_many_items(self):
        pq = BinomialHeapPQ()
        import random
        random.seed(77)
        items = [(f"task_{i}", random.randint(0, 100)) for i in range(50)]
        for name, pri in items:
            pq.push(name, pri)
        result = list(pq)
        assert len(result) == 50


# === LazyBinomialHeap Tests ===

class TestLazyBinomialHeap:
    def test_empty(self):
        h = LazyBinomialHeap()
        assert len(h) == 0
        assert h.is_empty()
        assert not h

    def test_insert_o1(self):
        h = LazyBinomialHeap()
        h.insert(5)
        h.insert(3)
        assert len(h) == 2
        assert h.find_min() == (3, 3)

    def test_peek(self):
        h = LazyBinomialHeap()
        h.insert(10)
        assert h.peek() == h.find_min()

    def test_find_min_empty(self):
        h = LazyBinomialHeap()
        with pytest.raises(IndexError):
            h.find_min()

    def test_extract_min(self):
        h = LazyBinomialHeap()
        for v in [5, 3, 7, 1, 4]:
            h.insert(v)
        assert h.extract_min() == (1, 1)
        assert h.extract_min() == (3, 3)
        assert h.extract_min() == (4, 4)

    def test_extract_min_empty(self):
        h = LazyBinomialHeap()
        with pytest.raises(IndexError):
            h.extract_min()

    def test_extract_all_sorted(self):
        h = LazyBinomialHeap()
        import random
        random.seed(55)
        values = list(range(50))
        random.shuffle(values)
        for v in values:
            h.insert(v)
        result = [h.extract_min()[0] for _ in range(50)]
        assert result == list(range(50))

    def test_merge(self):
        h1 = LazyBinomialHeap()
        h2 = LazyBinomialHeap()
        h1.insert(3)
        h1.insert(5)
        h2.insert(1)
        h2.insert(4)
        h1.merge(h2)
        assert len(h1) == 4
        assert len(h2) == 0
        assert h1.find_min() == (1, 1)

    def test_merge_type_error(self):
        h = LazyBinomialHeap()
        with pytest.raises(TypeError):
            h.merge("bad")

    def test_decrease_key(self):
        h = LazyBinomialHeap()
        n1 = h.insert(5)
        n2 = h.insert(3)
        n3 = h.insert(7)
        h.decrease_key(n3, 1)
        assert h.find_min() == (1, 7)

    def test_decrease_key_invalid(self):
        h = LazyBinomialHeap()
        n = h.insert(5)
        with pytest.raises(ValueError):
            h.decrease_key(n, 10)

    def test_delete(self):
        h = LazyBinomialHeap()
        n1 = h.insert(5)
        n2 = h.insert(3)
        n3 = h.insert(7)
        h.delete(n2)
        assert len(h) == 2
        assert h.extract_min() == (5, 5)

    def test_to_sorted_list(self):
        h = LazyBinomialHeap()
        for v in [3, 1, 2]:
            h.insert(v)
        assert h.to_sorted_list() == [(1, 1), (2, 2), (3, 3)]

    def test_iter(self):
        h = LazyBinomialHeap()
        for v in [3, 1, 2]:
            h.insert(v)
        items = sorted(list(h))
        assert items == [(1, 1), (2, 2), (3, 3)]

    def test_cached_min_updates_on_insert(self):
        h = LazyBinomialHeap()
        h.insert(5)
        assert h.find_min() == (5, 5)
        h.insert(2)
        assert h.find_min() == (2, 2)
        h.insert(8)
        assert h.find_min() == (2, 2)

    def test_large_lazy(self):
        h = LazyBinomialHeap()
        import random
        random.seed(33)
        values = list(range(200))
        random.shuffle(values)
        for v in values:
            h.insert(v)
        result = [h.extract_min()[0] for _ in range(200)]
        assert result == list(range(200))

    def test_interleaved_ops(self):
        h = LazyBinomialHeap()
        h.insert(10)
        h.insert(5)
        assert h.extract_min() == (5, 5)
        h.insert(3)
        h.insert(8)
        assert h.extract_min() == (3, 3)
        h.insert(1)
        assert h.extract_min() == (1, 1)
        assert h.extract_min() == (8, 8)
        assert h.extract_min() == (10, 10)
        assert h.is_empty()

    def test_merge_empty_heaps(self):
        h1 = LazyBinomialHeap()
        h2 = LazyBinomialHeap()
        h1.merge(h2)
        assert h1.is_empty()

    def test_merge_into_empty(self):
        h1 = LazyBinomialHeap()
        h2 = LazyBinomialHeap()
        h2.insert(5)
        h1.merge(h2)
        assert len(h1) == 1
        assert h1.find_min() == (5, 5)

    def test_with_values(self):
        h = LazyBinomialHeap()
        h.insert(3, "three")
        h.insert(1, "one")
        h.insert(2, "two")
        assert h.extract_min() == (1, "one")
        assert h.extract_min() == (2, "two")
        assert h.extract_min() == (3, "three")


# === Utility Function Tests ===

class TestHeapSort:
    def test_basic(self):
        result = heap_sort([5, 3, 1, 4, 2])
        assert result == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

    def test_empty(self):
        assert heap_sort([]) == []

    def test_single(self):
        assert heap_sort([42]) == [(42, 42)]

    def test_with_tuples(self):
        result = heap_sort([(3, "c"), (1, "a"), (2, "b")])
        assert result == [(1, "a"), (2, "b"), (3, "c")]

    def test_duplicates(self):
        result = heap_sort([3, 1, 3, 1])
        keys = [k for k, v in result]
        assert keys == [1, 1, 3, 3]

    def test_already_sorted(self):
        result = heap_sort([1, 2, 3, 4, 5])
        assert result == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

    def test_reverse_sorted(self):
        result = heap_sort([5, 4, 3, 2, 1])
        assert result == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]


class TestKSmallest:
    def test_basic(self):
        result = k_smallest([5, 3, 1, 4, 2], 3)
        assert result == [(1, 1), (2, 2), (3, 3)]

    def test_k_zero(self):
        assert k_smallest([1, 2, 3], 0) == []

    def test_k_negative(self):
        assert k_smallest([1, 2, 3], -1) == []

    def test_k_larger_than_list(self):
        result = k_smallest([3, 1, 2], 10)
        assert result == [(1, 1), (2, 2), (3, 3)]

    def test_with_tuples(self):
        result = k_smallest([(5, "e"), (1, "a"), (3, "c")], 2)
        assert result == [(1, "a"), (3, "c")]

    def test_empty(self):
        assert k_smallest([], 5) == []


class TestMergeSortedStreams:
    def test_basic(self):
        result = list(merge_sorted_streams([1, 3, 5], [2, 4, 6]))
        assert result == [1, 2, 3, 4, 5, 6]

    def test_three_streams(self):
        result = list(merge_sorted_streams([1, 4], [2, 5], [3, 6]))
        assert result == [1, 2, 3, 4, 5, 6]

    def test_empty_streams(self):
        result = list(merge_sorted_streams([], [1, 2], []))
        assert result == [1, 2]

    def test_all_empty(self):
        result = list(merge_sorted_streams([], [], []))
        assert result == []

    def test_single_stream(self):
        result = list(merge_sorted_streams([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_unequal_lengths(self):
        result = list(merge_sorted_streams([1], [2, 3, 4, 5]))
        assert result == [1, 2, 3, 4, 5]

    def test_duplicates_across_streams(self):
        result = list(merge_sorted_streams([1, 3, 5], [1, 3, 5]))
        assert result == [1, 1, 3, 3, 5, 5]


# === Edge Cases & Stress Tests ===

class TestEdgeCases:
    def test_single_element_operations(self):
        h = BinomialHeap()
        n = h.insert(1)
        assert h.find_min() == (1, 1)
        h.decrease_key(n, 0)
        assert h.find_min() == (0, 1)
        assert h.extract_min() == (0, 1)
        assert h.is_empty()

    def test_all_same_keys(self):
        h = BinomialHeap()
        nodes = [h.insert(5, f"v{i}") for i in range(8)]
        assert len(h) == 8
        for _ in range(8):
            k, v = h.extract_min()
            assert k == 5

    def test_string_keys(self):
        h = BinomialHeap()
        h.insert("banana")
        h.insert("apple")
        h.insert("cherry")
        assert h.extract_min() == ("apple", "apple")
        assert h.extract_min() == ("banana", "banana")

    def test_tuple_keys(self):
        h = BinomialHeap()
        h.insert((1, 2))
        h.insert((1, 1))
        h.insert((2, 0))
        assert h.extract_min() == ((1, 1), (1, 1))

    def test_decrease_key_chain(self):
        """Decrease key multiple times on different nodes."""
        h = BinomialHeap()
        nodes = [h.insert(i * 10) for i in range(5)]
        h.decrease_key(nodes[4], 1)  # 40 -> 1
        h.decrease_key(nodes[3], 2)  # 30 -> 2
        assert h.extract_min()[0] == 0
        assert h.extract_min()[0] == 1
        assert h.extract_min()[0] == 2

    def test_delete_all(self):
        h = BinomialHeap()
        nodes = [h.insert(i) for i in range(10)]
        for n in nodes:
            h.delete(n)
        assert h.is_empty()

    def test_merge_self_protection(self):
        """Merging with an empty heap doesn't break anything."""
        h1 = BinomialHeap()
        h2 = BinomialHeap()
        for v in [1, 2, 3]:
            h1.insert(v)
        h1.merge(h2)
        assert len(h1) == 3
        assert h1.extract_min() == (1, 1)

    def test_binomial_structure_16_elements(self):
        """16 = 2^4, single tree of degree 4."""
        h = BinomialHeap()
        for i in range(16):
            h.insert(i)
        root = h._head
        assert root.degree == 4
        assert root.sibling is None

    def test_binomial_structure_7_elements(self):
        """7 = 111 in binary => trees of degree 2, 1, 0."""
        h = BinomialHeap()
        for i in range(7):
            h.insert(i)
        degrees = []
        curr = h._head
        while curr:
            degrees.append(curr.degree)
            curr = curr.sibling
        assert sorted(degrees) == [0, 1, 2]

    def test_binomial_structure_5_elements(self):
        """5 = 101 in binary => trees of degree 2 and 0."""
        h = BinomialHeap()
        for i in range(5):
            h.insert(i)
        degrees = []
        curr = h._head
        while curr:
            degrees.append(curr.degree)
            curr = curr.sibling
        assert sorted(degrees) == [0, 2]

    def test_heap_sort_stability(self):
        """Verify heap_sort handles mixed types of input."""
        result = heap_sort([10, 1, 5, 2, 8, 3])
        keys = [k for k, v in result]
        assert keys == [1, 2, 3, 5, 8, 10]

    def test_lazy_consolidation_correctness(self):
        """Lazy heap should produce same results as eager heap."""
        import random
        random.seed(42)
        values = [random.randint(0, 1000) for _ in range(100)]

        h1 = BinomialHeap()
        h2 = LazyBinomialHeap()
        for v in values:
            h1.insert(v)
            h2.insert(v)

        r1 = h1.to_sorted_list()
        r2 = h2.to_sorted_list()
        assert r1 == r2
