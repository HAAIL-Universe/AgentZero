"""Tests for C120: Fibonacci Heap."""
import pytest
from fibonacci_heap import (
    FibonacciHeap, FibNode, MaxFibonacciHeap, FibonacciHeapMap,
    MergeableFibonacciHeap, FibonacciHeapPQ, dijkstra, prim_mst,
)


# ============================================================
# FibonacciHeap (min-heap) tests
# ============================================================

class TestFibonacciHeapBasic:
    def test_empty_heap(self):
        h = FibonacciHeap()
        assert len(h) == 0
        assert h.is_empty()
        assert not h

    def test_single_insert(self):
        h = FibonacciHeap()
        node = h.insert(42)
        assert len(h) == 1
        assert h.find_min().key == 42
        assert h.peek() == (42, 42)

    def test_insert_with_value(self):
        h = FibonacciHeap()
        h.insert(5, "hello")
        assert h.peek() == (5, "hello")

    def test_multiple_inserts(self):
        h = FibonacciHeap()
        for k in [10, 3, 7, 1, 5]:
            h.insert(k)
        assert len(h) == 5
        assert h.find_min().key == 1

    def test_extract_min_single(self):
        h = FibonacciHeap()
        h.insert(42)
        node = h.extract_min()
        assert node.key == 42
        assert len(h) == 0

    def test_extract_min_order(self):
        h = FibonacciHeap()
        for k in [10, 3, 7, 1, 5]:
            h.insert(k)
        result = []
        while h:
            result.append(h.extract_min().key)
        assert result == [1, 3, 5, 7, 10]

    def test_extract_min_empty_raises(self):
        h = FibonacciHeap()
        with pytest.raises(IndexError):
            h.extract_min()

    def test_peek_empty_raises(self):
        h = FibonacciHeap()
        with pytest.raises(IndexError):
            h.peek()

    def test_to_sorted_list(self):
        h = FibonacciHeap()
        for k in [8, 2, 6, 4]:
            h.insert(k, f"v{k}")
        result = h.to_sorted_list()
        assert result == [(2, "v2"), (4, "v4"), (6, "v6"), (8, "v8")]
        assert len(h) == 0


class TestFibonacciHeapDecreaseKey:
    def test_decrease_key_basic(self):
        h = FibonacciHeap()
        nodes = [h.insert(k) for k in [10, 20, 30]]
        h.decrease_key(nodes[2], 5)
        assert h.find_min().key == 5

    def test_decrease_key_no_parent(self):
        h = FibonacciHeap()
        node = h.insert(10)
        h.decrease_key(node, 3)
        assert h.find_min().key == 3

    def test_decrease_key_invalid_raises(self):
        h = FibonacciHeap()
        node = h.insert(10)
        with pytest.raises(ValueError):
            h.decrease_key(node, 15)

    def test_decrease_key_cascading(self):
        h = FibonacciHeap()
        # Build a tree structure by inserting and extracting
        nodes = {}
        for k in range(20):
            nodes[k] = h.insert(k)
        h.extract_min()  # triggers consolidation

        # Now decrease keys to trigger cascading cuts
        h.decrease_key(nodes[15], -1)
        assert h.find_min().key == -1

    def test_decrease_key_preserves_heap(self):
        h = FibonacciHeap()
        nodes = [h.insert(k) for k in range(50)]
        h.extract_min()  # consolidate

        # Decrease several keys
        h.decrease_key(nodes[30], -5)
        h.decrease_key(nodes[40], -10)
        h.decrease_key(nodes[20], -3)

        result = []
        while h:
            result.append(h.extract_min().key)
        assert result == sorted(result)


class TestFibonacciHeapDelete:
    def test_delete_min(self):
        h = FibonacciHeap()
        nodes = [h.insert(k) for k in [5, 3, 8]]
        h.delete(nodes[1])  # delete 3
        assert h.find_min().key == 5

    def test_delete_non_min(self):
        h = FibonacciHeap()
        nodes = [h.insert(k) for k in [5, 3, 8]]
        h.delete(nodes[2])  # delete 8
        assert len(h) == 2
        assert h.extract_min().key == 3
        assert h.extract_min().key == 5

    def test_delete_all(self):
        h = FibonacciHeap()
        nodes = [h.insert(k) for k in [1, 2, 3]]
        for n in nodes:
            h.delete(n)
        assert h.is_empty()


class TestFibonacciHeapMerge:
    def test_merge_two_heaps(self):
        h1 = FibonacciHeap()
        h2 = FibonacciHeap()
        h1.insert(5)
        h1.insert(3)
        h2.insert(1)
        h2.insert(7)
        h1.merge(h2)
        assert len(h1) == 4
        assert len(h2) == 0
        assert h1.find_min().key == 1

    def test_merge_into_empty(self):
        h1 = FibonacciHeap()
        h2 = FibonacciHeap()
        h2.insert(10)
        h1.merge(h2)
        assert len(h1) == 1
        assert h1.find_min().key == 10

    def test_merge_empty_into_nonempty(self):
        h1 = FibonacciHeap()
        h2 = FibonacciHeap()
        h1.insert(10)
        h1.merge(h2)
        assert len(h1) == 1

    def test_merge_type_error(self):
        h = FibonacciHeap()
        with pytest.raises(TypeError):
            h.merge([1, 2, 3])

    def test_merge_preserves_order(self):
        h1 = FibonacciHeap()
        h2 = FibonacciHeap()
        for k in [10, 20, 30]:
            h1.insert(k)
        for k in [5, 15, 25]:
            h2.insert(k)
        h1.merge(h2)
        result = h1.to_sorted_list()
        assert [k for k, v in result] == [5, 10, 15, 20, 25, 30]


class TestFibonacciHeapIteration:
    def test_iterate_empty(self):
        h = FibonacciHeap()
        assert list(h) == []

    def test_iterate_all_elements(self):
        h = FibonacciHeap()
        expected = {1, 2, 3, 4, 5}
        for k in expected:
            h.insert(k)
        keys = {k for k, v in h}
        assert keys == expected

    def test_iterate_after_extract(self):
        h = FibonacciHeap()
        for k in range(10):
            h.insert(k)
        h.extract_min()  # triggers consolidation, creates tree structure
        keys = {k for k, v in h}
        assert keys == set(range(1, 10))


class TestFibonacciHeapStress:
    def test_large_heap_sorted(self):
        h = FibonacciHeap()
        import random
        data = list(range(200))
        random.seed(42)
        random.shuffle(data)
        for k in data:
            h.insert(k)
        result = []
        while h:
            result.append(h.extract_min().key)
        assert result == list(range(200))

    def test_interleaved_insert_extract(self):
        h = FibonacciHeap()
        h.insert(10)
        h.insert(5)
        assert h.extract_min().key == 5
        h.insert(3)
        h.insert(8)
        assert h.extract_min().key == 3
        assert h.extract_min().key == 8
        assert h.extract_min().key == 10

    def test_decrease_key_stress(self):
        h = FibonacciHeap()
        nodes = [h.insert(k * 10) for k in range(50)]
        h.extract_min()  # consolidate
        # Decrease keys in reverse
        for i in range(49, 0, -1):
            h.decrease_key(nodes[i], -(50 - i))
        result = []
        while h:
            result.append(h.extract_min().key)
        assert result == sorted(result)

    def test_duplicate_keys(self):
        h = FibonacciHeap()
        for _ in range(5):
            h.insert(1)
            h.insert(2)
        result = []
        while h:
            result.append(h.extract_min().key)
        assert result == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

    def test_negative_keys(self):
        h = FibonacciHeap()
        for k in [-5, -3, -10, -1]:
            h.insert(k)
        assert h.extract_min().key == -10

    def test_float_keys(self):
        h = FibonacciHeap()
        for k in [1.5, 0.5, 2.5, 1.0]:
            h.insert(k)
        result = []
        while h:
            result.append(h.extract_min().key)
        assert result == [0.5, 1.0, 1.5, 2.5]


# ============================================================
# MaxFibonacciHeap tests
# ============================================================

class TestMaxFibonacciHeap:
    def test_basic(self):
        h = MaxFibonacciHeap()
        h.insert(5, "a")
        h.insert(10, "b")
        h.insert(3, "c")
        assert h.peek() == (10, "b")

    def test_extract_max(self):
        h = MaxFibonacciHeap()
        for k in [5, 10, 3, 8]:
            h.insert(k)
        result = []
        while h:
            result.append(h.extract_max()[0])
        assert result == [10, 8, 5, 3]

    def test_increase_key(self):
        h = MaxFibonacciHeap()
        n1 = h.insert(5)
        n2 = h.insert(10)
        h.increase_key(n1, 15)
        assert h.peek()[0] == 15

    def test_increase_key_invalid(self):
        h = MaxFibonacciHeap()
        n = h.insert(10)
        with pytest.raises(ValueError):
            h.increase_key(n, 5)

    def test_delete(self):
        h = MaxFibonacciHeap()
        nodes = [h.insert(k) for k in [5, 10, 3]]
        h.delete(nodes[1])  # delete 10
        assert h.peek()[0] == 5

    def test_merge(self):
        h1 = MaxFibonacciHeap()
        h2 = MaxFibonacciHeap()
        h1.insert(5)
        h2.insert(10)
        h1.merge(h2)
        assert h1.peek()[0] == 10
        assert len(h2) == 0

    def test_merge_type_error(self):
        h = MaxFibonacciHeap()
        with pytest.raises(TypeError):
            h.merge(FibonacciHeap())

    def test_empty(self):
        h = MaxFibonacciHeap()
        assert h.is_empty()
        assert not h
        with pytest.raises(IndexError):
            h.peek()


# ============================================================
# FibonacciHeapMap tests
# ============================================================

class TestFibonacciHeapMap:
    def test_basic_operations(self):
        m = FibonacciHeapMap()
        m.insert("a", 5)
        m.insert("b", 3)
        m.insert("c", 8)
        assert m.peek() == ("b", 3)
        assert len(m) == 3

    def test_pop(self):
        m = FibonacciHeapMap()
        m.insert("x", 10)
        m.insert("y", 5)
        assert m.pop() == ("y", 5)
        assert m.pop() == ("x", 10)

    def test_contains(self):
        m = FibonacciHeapMap()
        m.insert("a", 1)
        assert "a" in m
        assert "b" not in m

    def test_get_priority(self):
        m = FibonacciHeapMap()
        m.insert("a", 5)
        assert m.get_priority("a") == 5

    def test_get_priority_missing(self):
        m = FibonacciHeapMap()
        with pytest.raises(KeyError):
            m.get_priority("missing")

    def test_update_decrease(self):
        m = FibonacciHeapMap()
        m.insert("a", 10)
        m.insert("b", 5)
        m.update("a", 2)
        assert m.peek() == ("a", 2)

    def test_update_increase(self):
        m = FibonacciHeapMap()
        m.insert("a", 1)
        m.insert("b", 5)
        m.update("a", 10)
        assert m.peek() == ("b", 5)

    def test_update_missing(self):
        m = FibonacciHeapMap()
        with pytest.raises(KeyError):
            m.update("missing", 5)

    def test_delete(self):
        m = FibonacciHeapMap()
        m.insert("a", 5)
        m.insert("b", 3)
        m.delete("b")
        assert len(m) == 1
        assert m.peek() == ("a", 5)

    def test_delete_missing(self):
        m = FibonacciHeapMap()
        with pytest.raises(KeyError):
            m.delete("missing")

    def test_insert_duplicate_updates(self):
        m = FibonacciHeapMap()
        m.insert("a", 10)
        m.insert("a", 3)  # should update, not duplicate
        assert len(m) == 1
        assert m.get_priority("a") == 3

    def test_empty(self):
        m = FibonacciHeapMap()
        assert m.is_empty()
        with pytest.raises(IndexError):
            m.peek()


# ============================================================
# MergeableFibonacciHeap tests
# ============================================================

class TestMergeableFibonacciHeap:
    def test_create_and_insert(self):
        m = MergeableFibonacciHeap()
        m.create_heap("h1")
        m.insert("h1", 5)
        m.insert("h1", 3)
        assert m.find_min("h1").key == 3

    def test_merge_heaps(self):
        m = MergeableFibonacciHeap()
        m.create_heap("h1")
        m.create_heap("h2")
        m.insert("h1", 10)
        m.insert("h2", 5)
        m.merge_heaps("h1", "h2")
        assert m.find_min("h1").key == 5
        assert "h2" not in m

    def test_merge_heaps_new_name(self):
        m = MergeableFibonacciHeap()
        m.create_heap("a")
        m.create_heap("b")
        m.insert("a", 10)
        m.insert("b", 5)
        m.merge_heaps("a", "b", new_name="c")
        assert "c" in m
        assert "a" not in m

    def test_create_duplicate_raises(self):
        m = MergeableFibonacciHeap()
        m.create_heap("h1")
        with pytest.raises(ValueError):
            m.create_heap("h1")

    def test_heap_names(self):
        m = MergeableFibonacciHeap()
        m.create_heap("a")
        m.create_heap("b")
        assert set(m.heap_names()) == {"a", "b"}

    def test_extract_min(self):
        m = MergeableFibonacciHeap()
        m.create_heap("h")
        m.insert("h", 5)
        m.insert("h", 3)
        node = m.extract_min("h")
        assert node.key == 3


# ============================================================
# FibonacciHeapPQ tests
# ============================================================

class TestFibonacciHeapPQ:
    def test_push_pop(self):
        pq = FibonacciHeapPQ()
        pq.push(5, "a")
        pq.push(3, "b")
        pq.push(7, "c")
        assert pq.pop() == (3, "b")
        assert pq.pop() == (5, "a")
        assert pq.pop() == (7, "c")

    def test_peek(self):
        pq = FibonacciHeapPQ()
        pq.push(5, "a")
        assert pq.peek() == (5, "a")
        assert len(pq) == 1

    def test_peek_empty(self):
        pq = FibonacciHeapPQ()
        with pytest.raises(IndexError):
            pq.peek()

    def test_update(self):
        pq = FibonacciHeapPQ()
        pq.push(10, "a")
        pq.push(5, "b")
        pq.update("a", 2)
        assert pq.peek() == (2, "a")

    def test_update_missing(self):
        pq = FibonacciHeapPQ()
        with pytest.raises(KeyError):
            pq.update("missing", 5)

    def test_delete(self):
        pq = FibonacciHeapPQ()
        pq.push(5, "a")
        pq.push(3, "b")
        pq.delete("b")
        assert pq.peek() == (5, "a")

    def test_contains(self):
        pq = FibonacciHeapPQ()
        pq.push(5, "a")
        assert "a" in pq
        assert "b" not in pq

    def test_get_priority(self):
        pq = FibonacciHeapPQ()
        pq.push(5, "a")
        assert pq.get_priority("a") == 5

    def test_get_priority_missing(self):
        pq = FibonacciHeapPQ()
        with pytest.raises(KeyError):
            pq.get_priority("missing")

    def test_init_with_items(self):
        pq = FibonacciHeapPQ(items=[(5, "a"), (3, "b"), (8, "c")])
        assert pq.pop() == (3, "b")

    def test_push_duplicate_updates(self):
        pq = FibonacciHeapPQ()
        pq.push(10, "a")
        pq.push(3, "a")  # update
        assert len(pq) == 1
        assert pq.peek() == (3, "a")

    def test_max_heap_mode(self):
        pq = FibonacciHeapPQ(max_heap=True)
        pq.push(5, "a")
        pq.push(10, "b")
        pq.push(3, "c")
        assert pq.pop() == (10, "b")
        assert pq.pop() == (5, "a")
        assert pq.pop() == (3, "c")

    def test_max_heap_peek(self):
        pq = FibonacciHeapPQ(max_heap=True)
        pq.push(5, "a")
        pq.push(10, "b")
        assert pq.peek() == (10, "b")

    def test_max_heap_update(self):
        pq = FibonacciHeapPQ(max_heap=True)
        pq.push(5, "a")
        pq.push(10, "b")
        pq.update("a", 15)
        assert pq.peek() == (15, "a")

    def test_empty_pq(self):
        pq = FibonacciHeapPQ()
        assert pq.is_empty()
        assert not pq


# ============================================================
# Dijkstra tests
# ============================================================

class TestDijkstra:
    def test_simple_graph(self):
        graph = {
            'A': [('B', 1), ('C', 4)],
            'B': [('C', 2), ('D', 5)],
            'C': [('D', 1)],
            'D': [],
        }
        dist, pred = dijkstra(graph, 'A')
        assert dist['A'] == 0
        assert dist['B'] == 1
        assert dist['C'] == 3
        assert dist['D'] == 4

    def test_single_node(self):
        graph = {'A': []}
        dist, pred = dijkstra(graph, 'A')
        assert dist['A'] == 0

    def test_disconnected(self):
        graph = {'A': [('B', 1)], 'B': [], 'C': []}
        dist, pred = dijkstra(graph, 'A')
        assert dist['A'] == 0
        assert dist['B'] == 1
        assert dist['C'] == float('inf')

    def test_diamond(self):
        graph = {
            'S': [('A', 1), ('B', 5)],
            'A': [('B', 2)],
            'B': [],
        }
        dist, _ = dijkstra(graph, 'S')
        assert dist['B'] == 3

    def test_longer_path(self):
        graph = {
            0: [(1, 10), (2, 3)],
            1: [(3, 1)],
            2: [(1, 4), (3, 8)],
            3: [],
        }
        dist, pred = dijkstra(graph, 0)
        assert dist[3] == 8  # 0->2->1->3 = 3+4+1
        assert pred[3] == 1

    def test_empty_graph(self):
        dist, pred = dijkstra({}, 'A')
        assert dist == {'A': 0}


# ============================================================
# Prim's MST tests
# ============================================================

class TestPrimMST:
    def test_simple_graph(self):
        graph = {
            'A': [('B', 1), ('C', 4)],
            'B': [('A', 1), ('C', 2)],
            'C': [('A', 4), ('B', 2)],
        }
        edges = prim_mst(graph)
        total = sum(w for _, _, w in edges)
        assert total == 3  # edges AB(1) + BC(2)
        assert len(edges) == 2

    def test_single_node(self):
        graph = {'A': []}
        edges = prim_mst(graph)
        assert edges == []

    def test_empty_graph(self):
        assert prim_mst({}) == []

    def test_four_nodes(self):
        graph = {
            'A': [('B', 1), ('C', 3), ('D', 4)],
            'B': [('A', 1), ('C', 2), ('D', 5)],
            'C': [('A', 3), ('B', 2), ('D', 1)],
            'D': [('A', 4), ('B', 5), ('C', 1)],
        }
        edges = prim_mst(graph)
        total = sum(w for _, _, w in edges)
        assert total == 4  # AB(1) + BC(2) + CD(1)
        assert len(edges) == 3


# ============================================================
# Edge cases and structural tests
# ============================================================

class TestStructural:
    def test_consolidation_creates_tree(self):
        """After extract_min, heap should have tree structure."""
        h = FibonacciHeap()
        for k in range(16):
            h.insert(k)
        h.extract_min()
        # Should have consolidated into binomial-like trees
        # Verify by checking that further extracts are ordered
        prev = -1
        while h:
            key = h.extract_min().key
            assert key > prev
            prev = key

    def test_cascading_cut_chain(self):
        """Test cascading cuts with deep tree."""
        h = FibonacciHeap()
        nodes = [h.insert(k) for k in range(100)]
        # Extract to build structure
        for _ in range(10):
            h.extract_min()
        # Multiple decrease keys to trigger cascading cuts
        for i in range(99, 50, -1):
            if nodes[i].parent is not None:
                h.decrease_key(nodes[i], -(100 - i))
        result = []
        while h:
            result.append(h.extract_min().key)
        assert result == sorted(result)

    def test_node_slots(self):
        """FibNode uses __slots__ for memory efficiency."""
        node = FibNode(1)
        with pytest.raises(AttributeError):
            node.extra = "fail"

    def test_bool_semantics(self):
        h = FibonacciHeap()
        assert not h
        h.insert(1)
        assert h

    def test_equal_keys_stability(self):
        """Equal keys should all be extractable."""
        h = FibonacciHeap()
        for i in range(20):
            h.insert(0, f"item_{i}")
        values = set()
        while h:
            node = h.extract_min()
            values.add(node.value)
        assert len(values) == 20

    def test_decrease_to_same_key(self):
        """Decreasing to same key should be a no-op."""
        h = FibonacciHeap()
        node = h.insert(5)
        h.decrease_key(node, 5)  # same key, no change
        assert h.find_min().key == 5

    def test_large_merge(self):
        h1 = FibonacciHeap()
        h2 = FibonacciHeap()
        for i in range(100):
            h1.insert(i * 2)
            h2.insert(i * 2 + 1)
        h1.merge(h2)
        assert len(h1) == 200
        result = []
        while h1:
            result.append(h1.extract_min().key)
        assert result == list(range(200))


# ============================================================
# Additional coverage
# ============================================================

class TestAdditionalCoverage:
    def test_insert_extract_insert(self):
        h = FibonacciHeap()
        h.insert(5)
        h.extract_min()
        h.insert(3)
        assert h.find_min().key == 3

    def test_decrease_key_new_min(self):
        h = FibonacciHeap()
        nodes = [h.insert(k) for k in [10, 20, 30, 40, 50]]
        h.extract_min()  # remove 10, consolidate
        h.decrease_key(nodes[4], 1)
        assert h.find_min().key == 1

    def test_delete_then_insert(self):
        h = FibonacciHeap()
        nodes = [h.insert(k) for k in [5, 3, 8]]
        h.delete(nodes[0])
        h.insert(1)
        assert h.find_min().key == 1
        assert len(h) == 3

    def test_max_heap_large(self):
        h = MaxFibonacciHeap()
        for k in range(50):
            h.insert(k)
        result = []
        while h:
            result.append(h.extract_max()[0])
        assert result == list(range(49, -1, -1))

    def test_pq_delete_then_pop(self):
        pq = FibonacciHeapPQ()
        pq.push(5, "a")
        pq.push(3, "b")
        pq.push(7, "c")
        pq.delete("b")
        assert pq.pop() == (5, "a")

    def test_heap_map_full_workflow(self):
        m = FibonacciHeapMap()
        m.insert("task1", 5)
        m.insert("task2", 3)
        m.insert("task3", 8)
        assert m.peek() == ("task2", 3)
        m.update("task3", 1)
        assert m.peek() == ("task3", 1)
        m.delete("task3")
        assert m.peek() == ("task2", 3)
        m.pop()
        assert m.peek() == ("task1", 5)

    def test_dijkstra_with_update(self):
        """Graph where shortest path requires relaxation."""
        graph = {
            'S': [('A', 10), ('B', 1)],
            'A': [('T', 1)],
            'B': [('A', 1)],
            'T': [],
        }
        dist, _ = dijkstra(graph, 'S')
        assert dist['A'] == 2  # S->B->A
        assert dist['T'] == 3  # S->B->A->T

    def test_prim_complete_graph(self):
        """Complete graph K4."""
        graph = {
            0: [(1, 3), (2, 1), (3, 6)],
            1: [(0, 3), (2, 5), (3, 2)],
            2: [(0, 1), (1, 5), (3, 4)],
            3: [(0, 6), (1, 2), (2, 4)],
        }
        edges = prim_mst(graph)
        total = sum(w for _, _, w in edges)
        assert total == 6  # edges: 0-2(1), 0-1(3), 1-3(2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
