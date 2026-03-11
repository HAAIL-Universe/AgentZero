"""Tests for C121: Pairing Heap"""
import pytest
from pairing_heap import (
    PairingHeap, MaxPairingHeap, PairingHeapMap,
    MergeablePairingHeap, PairingHeapPQ, LazyPairingHeap,
    PairingNode, dijkstra, prim_mst, k_smallest, heap_sort
)


# ===== PairingNode =====

class TestPairingNode:
    def test_create_node(self):
        n = PairingNode(5, "hello")
        assert n.key == 5
        assert n.value == "hello"
        assert n.child is None
        assert n.sibling is None
        assert n.parent is None

    def test_default_value(self):
        n = PairingNode(10)
        assert n.value == 10

    def test_repr(self):
        n = PairingNode(3, "x")
        assert "3" in repr(n)
        assert "x" in repr(n)


# ===== PairingHeap (min) =====

class TestPairingHeap:
    def test_empty(self):
        h = PairingHeap()
        assert len(h) == 0
        assert h.is_empty()
        assert not h

    def test_insert_one(self):
        h = PairingHeap()
        h.insert(5)
        assert len(h) == 1
        assert h.find_min() == 5
        assert not h.is_empty()

    def test_insert_multiple(self):
        h = PairingHeap()
        h.insert(5)
        h.insert(3)
        h.insert(7)
        assert h.find_min() == 3
        assert len(h) == 3

    def test_peek(self):
        h = PairingHeap()
        h.insert(5, "five")
        assert h.peek() == (5, "five")

    def test_delete_min(self):
        h = PairingHeap()
        h.insert(5)
        h.insert(3)
        h.insert(7)
        assert h.delete_min() == (3, 3)
        assert h.find_min() == 5
        assert len(h) == 2

    def test_extract_min(self):
        h = PairingHeap()
        h.insert(10)
        assert h.extract_min() == (10, 10)
        assert h.is_empty()

    def test_sorted_extraction(self):
        h = PairingHeap()
        import random
        values = list(range(50))
        random.shuffle(values)
        for v in values:
            h.insert(v)
        result = []
        while h:
            result.append(h.delete_min()[0])
        assert result == sorted(values)

    def test_decrease_key(self):
        h = PairingHeap()
        n1 = h.insert(10)
        h.insert(5)
        h.insert(7)
        h.decrease_key(n1, 2)
        assert h.find_min() == 2

    def test_decrease_key_to_same(self):
        h = PairingHeap()
        n1 = h.insert(5)
        h.decrease_key(n1, 5)  # same key is fine
        assert h.find_min() == 5

    def test_decrease_key_root(self):
        h = PairingHeap()
        n1 = h.insert(5)
        h.insert(10)
        h.decrease_key(n1, 3)
        assert h.find_min() == 3

    def test_decrease_key_raises(self):
        h = PairingHeap()
        n1 = h.insert(5)
        with pytest.raises(ValueError):
            h.decrease_key(n1, 10)

    def test_delete_arbitrary(self):
        h = PairingHeap()
        h.insert(1)
        n2 = h.insert(5)
        h.insert(3)
        h.delete(n2)
        assert len(h) == 2
        result = []
        while h:
            result.append(h.delete_min()[0])
        assert result == [1, 3]

    def test_delete_root(self):
        h = PairingHeap()
        n1 = h.insert(1)
        h.insert(5)
        h.delete(n1)
        assert h.find_min() == 5
        assert len(h) == 1

    def test_delete_node_with_children(self):
        h = PairingHeap()
        h.insert(1)
        n = h.insert(3)
        h.insert(2)
        h.insert(4)
        h.insert(5)
        h.delete(n)
        result = []
        while h:
            result.append(h.delete_min()[0])
        assert 3 not in result
        assert sorted(result) == result

    def test_merge(self):
        h1 = PairingHeap()
        h2 = PairingHeap()
        h1.insert(5)
        h1.insert(3)
        h2.insert(4)
        h2.insert(1)
        h1.merge(h2)
        assert len(h1) == 4
        assert h1.find_min() == 1
        assert len(h2) == 0

    def test_merge_empty(self):
        h1 = PairingHeap()
        h2 = PairingHeap()
        h1.insert(5)
        h1.merge(h2)
        assert len(h1) == 1
        assert h1.find_min() == 5

    def test_merge_into_empty(self):
        h1 = PairingHeap()
        h2 = PairingHeap()
        h2.insert(5)
        h1.merge(h2)
        assert len(h1) == 1
        assert h1.find_min() == 5

    def test_merge_type_check(self):
        h = PairingHeap()
        with pytest.raises(TypeError):
            h.merge("not a heap")

    def test_iter_sorted(self):
        h = PairingHeap()
        for v in [5, 1, 3, 2, 4]:
            h.insert(v)
        result = list(h)
        assert [k for k, v in result] == [1, 2, 3, 4, 5]
        # Original heap should be unchanged
        assert len(h) == 5

    def test_to_sorted_list(self):
        h = PairingHeap()
        for v in [3, 1, 2]:
            h.insert(v)
        assert [k for k, v in h.to_sorted_list()] == [1, 2, 3]

    def test_clear(self):
        h = PairingHeap()
        for v in range(10):
            h.insert(v)
        h.clear()
        assert len(h) == 0
        assert h.is_empty()

    def test_find_min_empty_raises(self):
        h = PairingHeap()
        with pytest.raises(IndexError):
            h.find_min()

    def test_peek_empty_raises(self):
        h = PairingHeap()
        with pytest.raises(IndexError):
            h.peek()

    def test_delete_min_empty_raises(self):
        h = PairingHeap()
        with pytest.raises(IndexError):
            h.delete_min()

    def test_duplicate_keys(self):
        h = PairingHeap()
        h.insert(3, "a")
        h.insert(3, "b")
        h.insert(3, "c")
        assert h.find_min() == 3
        results = set()
        while h:
            k, v = h.delete_min()
            assert k == 3
            results.add(v)
        assert results == {"a", "b", "c"}

    def test_negative_keys(self):
        h = PairingHeap()
        h.insert(-5)
        h.insert(-10)
        h.insert(0)
        assert h.find_min() == -10

    def test_float_keys(self):
        h = PairingHeap()
        h.insert(1.5)
        h.insert(0.5)
        h.insert(2.5)
        assert h.find_min() == 0.5

    def test_large_heap(self):
        h = PairingHeap()
        import random
        values = list(range(1000))
        random.shuffle(values)
        for v in values:
            h.insert(v)
        result = []
        while h:
            result.append(h.delete_min()[0])
        assert result == list(range(1000))

    def test_insert_returns_node(self):
        h = PairingHeap()
        n = h.insert(5, "data")
        assert isinstance(n, PairingNode)
        assert n.key == 5
        assert n.value == "data"

    def test_size_property(self):
        h = PairingHeap()
        assert h.size == 0
        h.insert(1)
        assert h.size == 1

    def test_bool(self):
        h = PairingHeap()
        assert not h
        h.insert(1)
        assert h


# ===== MaxPairingHeap =====

class TestMaxPairingHeap:
    def test_empty(self):
        h = MaxPairingHeap()
        assert h.is_empty()

    def test_insert_and_find_max(self):
        h = MaxPairingHeap()
        h.insert(5)
        h.insert(10)
        h.insert(3)
        assert h.find_max() == 10

    def test_peek(self):
        h = MaxPairingHeap()
        h.insert(5, "five")
        assert h.peek() == (5, "five")

    def test_delete_max(self):
        h = MaxPairingHeap()
        h.insert(5)
        h.insert(10)
        h.insert(3)
        assert h.delete_max() == (10, 10)
        assert h.find_max() == 5

    def test_extract_max(self):
        h = MaxPairingHeap()
        h.insert(7)
        assert h.extract_max() == (7, 7)

    def test_sorted_extraction(self):
        h = MaxPairingHeap()
        for v in [5, 1, 3, 2, 4]:
            h.insert(v)
        result = []
        while h:
            result.append(h.delete_max()[0])
        assert result == [5, 4, 3, 2, 1]

    def test_increase_key(self):
        h = MaxPairingHeap()
        n = h.insert(5)
        h.insert(10)
        h.increase_key(n, 15)
        assert h.find_max() == 15

    def test_increase_key_root(self):
        h = MaxPairingHeap()
        n = h.insert(10)
        h.insert(5)
        h.increase_key(n, 20)
        assert h.find_max() == 20

    def test_increase_key_raises(self):
        h = MaxPairingHeap()
        n = h.insert(10)
        with pytest.raises(ValueError):
            h.increase_key(n, 5)

    def test_delete_arbitrary(self):
        h = MaxPairingHeap()
        h.insert(10)
        n = h.insert(5)
        h.insert(8)
        h.delete(n)
        assert len(h) == 2
        result = []
        while h:
            result.append(h.delete_max()[0])
        assert result == [10, 8]

    def test_delete_root(self):
        h = MaxPairingHeap()
        n = h.insert(10)
        h.insert(5)
        h.delete(n)
        assert h.find_max() == 5

    def test_merge(self):
        h1 = MaxPairingHeap()
        h2 = MaxPairingHeap()
        h1.insert(5)
        h2.insert(10)
        h1.merge(h2)
        assert h1.find_max() == 10
        assert len(h1) == 2
        assert len(h2) == 0

    def test_merge_type_check(self):
        h = MaxPairingHeap()
        with pytest.raises(TypeError):
            h.merge(PairingHeap())

    def test_clear(self):
        h = MaxPairingHeap()
        h.insert(5)
        h.clear()
        assert h.is_empty()

    def test_find_max_empty_raises(self):
        with pytest.raises(IndexError):
            MaxPairingHeap().find_max()

    def test_peek_empty_raises(self):
        with pytest.raises(IndexError):
            MaxPairingHeap().peek()

    def test_delete_max_empty_raises(self):
        with pytest.raises(IndexError):
            MaxPairingHeap().delete_max()

    def test_large_max_heap(self):
        h = MaxPairingHeap()
        import random
        values = list(range(500))
        random.shuffle(values)
        for v in values:
            h.insert(v)
        result = []
        while h:
            result.append(h.delete_max()[0])
        assert result == list(range(499, -1, -1))


# ===== PairingHeapMap =====

class TestPairingHeapMap:
    def test_empty(self):
        m = PairingHeapMap()
        assert m.is_empty()
        assert len(m) == 0

    def test_insert_and_peek(self):
        m = PairingHeapMap()
        m.insert("task_a", 5)
        assert m.peek() == (5, "task_a")

    def test_insert_multiple(self):
        m = PairingHeapMap()
        m.insert("a", 5)
        m.insert("b", 3)
        m.insert("c", 7)
        assert m.peek() == (3, "b")

    def test_extract_min(self):
        m = PairingHeapMap()
        m.insert("a", 5)
        m.insert("b", 3)
        assert m.extract_min() == (3, "b")
        assert len(m) == 1

    def test_decrease_key(self):
        m = PairingHeapMap()
        m.insert("a", 5)
        m.insert("b", 3)
        m.decrease_key("a", 1)
        assert m.peek() == (1, "a")

    def test_get_priority(self):
        m = PairingHeapMap()
        m.insert("x", 10)
        assert m.get_priority("x") == 10

    def test_delete(self):
        m = PairingHeapMap()
        m.insert("a", 5)
        m.insert("b", 3)
        m.delete("b")
        assert len(m) == 1
        assert m.peek() == (5, "a")

    def test_update_existing(self):
        m = PairingHeapMap()
        m.insert("a", 5)
        m.update("a", 3)  # lower, should update
        assert m.get_priority("a") == 3

    def test_update_no_decrease(self):
        m = PairingHeapMap()
        m.insert("a", 3)
        m.update("a", 5)  # higher, should not update
        assert m.get_priority("a") == 3

    def test_update_new(self):
        m = PairingHeapMap()
        m.update("a", 5)  # new entry
        assert "a" in m

    def test_contains(self):
        m = PairingHeapMap()
        m.insert("a", 5)
        assert "a" in m
        assert "b" not in m

    def test_insert_duplicate_raises(self):
        m = PairingHeapMap()
        m.insert("a", 5)
        with pytest.raises(KeyError):
            m.insert("a", 3)

    def test_decrease_key_missing_raises(self):
        m = PairingHeapMap()
        with pytest.raises(KeyError):
            m.decrease_key("x", 5)

    def test_get_priority_missing_raises(self):
        m = PairingHeapMap()
        with pytest.raises(KeyError):
            m.get_priority("x")

    def test_delete_missing_raises(self):
        m = PairingHeapMap()
        with pytest.raises(KeyError):
            m.delete("x")

    def test_clear(self):
        m = PairingHeapMap()
        m.insert("a", 5)
        m.insert("b", 3)
        m.clear()
        assert m.is_empty()

    def test_full_workflow(self):
        m = PairingHeapMap()
        m.insert("server_a", 100)
        m.insert("server_b", 50)
        m.insert("server_c", 75)
        m.decrease_key("server_a", 25)
        pri, name = m.extract_min()
        assert name == "server_a"
        assert pri == 25
        m.delete("server_c")
        assert len(m) == 1


# ===== MergeablePairingHeap =====

class TestMergeablePairingHeap:
    def test_basic(self):
        h = MergeablePairingHeap()
        h.insert(5)
        h.insert(3)
        assert h.find_min() == 3

    def test_peek(self):
        h = MergeablePairingHeap()
        h.insert(5, "x")
        assert h.peek() == (5, "x")

    def test_extract_min(self):
        h = MergeablePairingHeap()
        h.insert(5)
        h.insert(3)
        assert h.extract_min() == (3, 3)

    def test_merge(self):
        h1 = MergeablePairingHeap()
        h2 = MergeablePairingHeap()
        h1.insert(5)
        h2.insert(3)
        h1.merge(h2)
        assert h1.find_min() == 3
        assert len(h1) == 2

    def test_merge_type_check(self):
        h = MergeablePairingHeap()
        with pytest.raises(TypeError):
            h.merge("nope")

    def test_merge_many(self):
        heaps = []
        for i in range(5):
            h = MergeablePairingHeap()
            h.insert(i * 10)
            heaps.append(h)
        main = MergeablePairingHeap()
        main.insert(100)
        main.merge_many(*heaps)
        assert main.find_min() == 0
        assert len(main) == 6

    def test_split_min(self):
        h = MergeablePairingHeap()
        h.insert(5)
        h.insert(3)
        h.insert(7)
        split = h.split_min()
        assert split.find_min() == 3
        assert len(split) == 1
        assert h.find_min() == 5
        assert len(h) == 2

    def test_split_min_empty_raises(self):
        h = MergeablePairingHeap()
        with pytest.raises(IndexError):
            h.split_min()

    def test_decrease_key(self):
        h = MergeablePairingHeap()
        n = h.insert(10)
        h.insert(5)
        h.decrease_key(n, 1)
        assert h.find_min() == 1

    def test_to_sorted_list(self):
        h = MergeablePairingHeap()
        for v in [5, 1, 3]:
            h.insert(v)
        assert [k for k, v in h.to_sorted_list()] == [1, 3, 5]

    def test_clear(self):
        h = MergeablePairingHeap()
        h.insert(5)
        h.clear()
        assert h.is_empty()


# ===== PairingHeapPQ =====

class TestPairingHeapPQ:
    def test_empty(self):
        pq = PairingHeapPQ()
        assert pq.is_empty()
        assert len(pq) == 0

    def test_add_task(self):
        pq = PairingHeapPQ()
        pq.add_task("task1", priority=5)
        assert not pq.is_empty()
        assert len(pq) == 1

    def test_get_next(self):
        pq = PairingHeapPQ()
        pq.add_task("task1", priority=5, data={"info": "test"})
        pri, tid, data = pq.get_next()
        assert pri == 5
        assert tid == "task1"
        assert data == {"info": "test"}

    def test_pop_task(self):
        pq = PairingHeapPQ()
        pq.add_task("a", 5)
        pq.add_task("b", 3)
        pq.add_task("c", 7)
        pri, tid, data = pq.pop_task()
        assert pri == 3
        assert tid == "b"
        assert len(pq) == 2

    def test_cancel_task(self):
        pq = PairingHeapPQ()
        pq.add_task("a", 5)
        pq.add_task("b", 3)
        pq.cancel_task("b")
        assert len(pq) == 1
        pri, tid, _ = pq.pop_task()
        assert tid == "a"

    def test_update_priority(self):
        pq = PairingHeapPQ()
        pq.add_task("a", 5)
        pq.add_task("b", 3)
        pq.update_priority("a", 1)
        pri, tid, _ = pq.pop_task()
        assert tid == "a"
        assert pri == 1

    def test_update_priority_increase(self):
        pq = PairingHeapPQ()
        pq.add_task("a", 1)
        pq.add_task("b", 5)
        pq.update_priority("a", 10)
        pri, tid, _ = pq.pop_task()
        assert tid == "b"

    def test_has_task(self):
        pq = PairingHeapPQ()
        pq.add_task("a", 5)
        assert pq.has_task("a")
        assert not pq.has_task("b")

    def test_get_priority(self):
        pq = PairingHeapPQ()
        pq.add_task("a", 5)
        assert pq.get_priority("a") == 5

    def test_batch_add(self):
        pq = PairingHeapPQ()
        pq.batch_add([("a", 5), ("b", 3), ("c", 7, "data")])
        assert len(pq) == 3
        pri, tid, _ = pq.pop_task()
        assert tid == "b"

    def test_drain(self):
        pq = PairingHeapPQ()
        pq.add_task("a", 5)
        pq.add_task("b", 3)
        pq.add_task("c", 1)
        result = list(pq.drain())
        assert [tid for _, tid, _ in result] == ["c", "b", "a"]
        assert pq.is_empty()

    def test_add_duplicate_raises(self):
        pq = PairingHeapPQ()
        pq.add_task("a", 5)
        with pytest.raises(KeyError):
            pq.add_task("a", 3)

    def test_cancel_missing_raises(self):
        pq = PairingHeapPQ()
        with pytest.raises(KeyError):
            pq.cancel_task("x")

    def test_update_missing_raises(self):
        pq = PairingHeapPQ()
        with pytest.raises(KeyError):
            pq.update_priority("x", 5)

    def test_get_priority_missing_raises(self):
        pq = PairingHeapPQ()
        with pytest.raises(KeyError):
            pq.get_priority("x")

    def test_get_next_empty_raises(self):
        pq = PairingHeapPQ()
        with pytest.raises(IndexError):
            pq.get_next()

    def test_pop_empty_raises(self):
        pq = PairingHeapPQ()
        with pytest.raises(IndexError):
            pq.pop_task()

    def test_clear(self):
        pq = PairingHeapPQ()
        pq.add_task("a", 5)
        pq.clear()
        assert pq.is_empty()

    def test_stable_ordering(self):
        """Tasks with same priority should come out in insertion order."""
        pq = PairingHeapPQ()
        pq.add_task("first", 5)
        pq.add_task("second", 5)
        pq.add_task("third", 5)
        result = list(pq.drain())
        assert [tid for _, tid, _ in result] == ["first", "second", "third"]

    def test_bool(self):
        pq = PairingHeapPQ()
        assert not pq
        pq.add_task("a", 1)
        assert pq

    def test_scheduling_scenario(self):
        """Simulate a task scheduler scenario."""
        pq = PairingHeapPQ()
        pq.add_task("build", 3, {"cmd": "make"})
        pq.add_task("test", 5, {"cmd": "pytest"})
        pq.add_task("deploy", 10, {"cmd": "deploy.sh"})
        pq.add_task("lint", 2, {"cmd": "flake8"})
        # Urgent hotfix
        pq.update_priority("deploy", 1)
        pri, tid, data = pq.pop_task()
        assert tid == "deploy"
        assert data["cmd"] == "deploy.sh"


# ===== LazyPairingHeap =====

class TestLazyPairingHeap:
    def test_empty(self):
        h = LazyPairingHeap()
        assert h.is_empty()

    def test_insert_and_find_min(self):
        h = LazyPairingHeap()
        h.insert(5)
        h.insert(3)
        h.insert(7)
        assert h.find_min() == 3

    def test_peek(self):
        h = LazyPairingHeap()
        h.insert(5, "x")
        assert h.peek() == (5, "x")

    def test_delete_min(self):
        h = LazyPairingHeap()
        h.insert(5)
        h.insert(3)
        assert h.delete_min() == (3, 3)
        assert len(h) == 1

    def test_extract_min(self):
        h = LazyPairingHeap()
        h.insert(10)
        assert h.extract_min() == (10, 10)

    def test_sorted_extraction(self):
        h = LazyPairingHeap()
        import random
        values = list(range(100))
        random.shuffle(values)
        for v in values:
            h.insert(v)
        result = []
        while h:
            result.append(h.delete_min()[0])
        assert result == list(range(100))

    def test_merge(self):
        h1 = LazyPairingHeap()
        h2 = LazyPairingHeap()
        h1.insert(5)
        h2.insert(3)
        h1.merge(h2)
        assert h1.find_min() == 3
        assert len(h1) == 2
        assert len(h2) == 0

    def test_merge_type_check(self):
        h = LazyPairingHeap()
        with pytest.raises(TypeError):
            h.merge(PairingHeap())

    def test_clear(self):
        h = LazyPairingHeap()
        h.insert(5)
        h.clear()
        assert h.is_empty()

    def test_find_min_empty_raises(self):
        with pytest.raises(IndexError):
            LazyPairingHeap().find_min()

    def test_peek_empty_raises(self):
        with pytest.raises(IndexError):
            LazyPairingHeap().peek()

    def test_delete_min_empty_raises(self):
        with pytest.raises(IndexError):
            LazyPairingHeap().delete_min()

    def test_lazy_consolidation(self):
        """Test that inserts are buffered and consolidated on demand."""
        h = LazyPairingHeap()
        for i in range(20):
            h.insert(i)
        assert len(h) == 20
        # First find_min triggers consolidation
        assert h.find_min() == 0

    def test_interleaved_ops(self):
        """Insert and delete interleaved."""
        h = LazyPairingHeap()
        h.insert(10)
        h.insert(5)
        assert h.delete_min() == (5, 5)
        h.insert(3)
        h.insert(7)
        assert h.delete_min() == (3, 3)
        assert h.delete_min() == (7, 7)
        assert h.delete_min() == (10, 10)
        assert h.is_empty()


# ===== Utility Functions =====

class TestDijkstra:
    def test_simple_graph(self):
        graph = {
            'A': [('B', 1), ('C', 4)],
            'B': [('C', 2), ('D', 5)],
            'C': [('D', 1)],
            'D': []
        }
        dist, pred = dijkstra(graph, 'A')
        assert dist['A'] == 0
        assert dist['B'] == 1
        assert dist['C'] == 3
        assert dist['D'] == 4

    def test_single_node(self):
        graph = {'A': []}
        dist, pred = dijkstra(graph, 'A')
        assert dist == {'A': 0}
        assert pred == {'A': None}

    def test_disconnected(self):
        graph = {
            'A': [('B', 1)],
            'B': [],
            'C': []
        }
        dist, pred = dijkstra(graph, 'A')
        assert 'C' not in dist

    def test_shortest_path_reconstruction(self):
        graph = {
            'A': [('B', 1), ('C', 4)],
            'B': [('C', 2)],
            'C': []
        }
        dist, pred = dijkstra(graph, 'A')
        # Reconstruct path to C
        path = []
        node = 'C'
        while node is not None:
            path.append(node)
            node = pred[node]
        path.reverse()
        assert path == ['A', 'B', 'C']

    def test_diamond_graph(self):
        graph = {
            'S': [('A', 1), ('B', 10)],
            'A': [('B', 2)],
            'B': [('T', 1)],
            'T': []
        }
        dist, _ = dijkstra(graph, 'S')
        assert dist['T'] == 4  # S->A->B->T


class TestPrimMST:
    def test_simple(self):
        graph = {
            'A': [('B', 1), ('C', 3)],
            'B': [('A', 1), ('C', 2)],
            'C': [('A', 3), ('B', 2)]
        }
        mst = prim_mst(graph)
        total = sum(w for _, _, w in mst)
        assert total == 3  # AB(1) + BC(2)
        assert len(mst) == 2

    def test_empty(self):
        assert prim_mst({}) == []

    def test_single_node(self):
        mst = prim_mst({'A': []})
        assert mst == []

    def test_four_nodes(self):
        graph = {
            'A': [('B', 1), ('C', 4), ('D', 3)],
            'B': [('A', 1), ('C', 2)],
            'C': [('A', 4), ('B', 2), ('D', 5)],
            'D': [('A', 3), ('C', 5)]
        }
        mst = prim_mst(graph)
        total = sum(w for _, _, w in mst)
        assert total == 6  # AB(1) + BC(2) + AD(3)
        assert len(mst) == 3


class TestKSmallest:
    def test_basic(self):
        result = k_smallest([5, 3, 1, 4, 2], 3)
        assert [k for k, v in result] == [1, 2, 3]

    def test_k_zero(self):
        assert k_smallest([1, 2, 3], 0) == []

    def test_k_larger_than_list(self):
        result = k_smallest([3, 1, 2], 10)
        assert [k for k, v in result] == [1, 2, 3]

    def test_tuples(self):
        result = k_smallest([(3, "c"), (1, "a"), (2, "b")], 2)
        assert result == [(1, "a"), (2, "b")]

    def test_empty(self):
        assert k_smallest([], 5) == []


class TestHeapSort:
    def test_basic(self):
        result = heap_sort([5, 3, 1, 4, 2])
        assert [k for k, v in result] == [1, 2, 3, 4, 5]

    def test_already_sorted(self):
        result = heap_sort([1, 2, 3])
        assert [k for k, v in result] == [1, 2, 3]

    def test_reverse_sorted(self):
        result = heap_sort([3, 2, 1])
        assert [k for k, v in result] == [1, 2, 3]

    def test_empty(self):
        assert heap_sort([]) == []

    def test_single(self):
        result = heap_sort([42])
        assert [k for k, v in result] == [42]

    def test_duplicates(self):
        result = heap_sort([3, 1, 3, 1, 2])
        assert [k for k, v in result] == [1, 1, 2, 3, 3]

    def test_tuples(self):
        result = heap_sort([(5, "e"), (1, "a"), (3, "c")])
        assert result == [(1, "a"), (3, "c"), (5, "e")]


# ===== Cross-variant tests =====

class TestCrossVariant:
    def test_all_heaps_same_result(self):
        """All min-heap variants should produce the same sorted output."""
        import random
        values = list(range(30))
        random.shuffle(values)

        # PairingHeap
        h1 = PairingHeap()
        for v in values:
            h1.insert(v)
        r1 = []
        while h1:
            r1.append(h1.delete_min()[0])

        # LazyPairingHeap
        h2 = LazyPairingHeap()
        for v in values:
            h2.insert(v)
        r2 = []
        while h2:
            r2.append(h2.delete_min()[0])

        # MergeablePairingHeap
        h3 = MergeablePairingHeap()
        for v in values:
            h3.insert(v)
        r3 = []
        while h3:
            r3.append(h3.delete_min()[0])

        assert r1 == r2 == r3 == sorted(values)

    def test_merge_across_sizes(self):
        """Merge heaps of different sizes."""
        h1 = PairingHeap()
        h2 = PairingHeap()
        for i in range(100):
            h1.insert(i)
        for i in range(100, 105):
            h2.insert(i)
        h1.merge(h2)
        assert len(h1) == 105
        assert h1.find_min() == 0

    def test_decrease_key_chain(self):
        """Multiple decrease-key operations on different nodes."""
        h = PairingHeap()
        nodes = [h.insert(i * 10) for i in range(10)]
        # Decrease all to negative values
        for i, n in enumerate(nodes):
            h.decrease_key(n, -i)
        assert h.find_min() == -9

    def test_delete_all_nodes(self):
        """Delete all nodes one by one (not via delete_min)."""
        h = PairingHeap()
        nodes = [h.insert(i) for i in range(10)]
        import random
        random.shuffle(nodes)
        for n in nodes:
            h.delete(n)
        assert h.is_empty()

    def test_stress_insert_delete(self):
        """Stress test with random insert/delete operations."""
        import random
        h = PairingHeap()
        inserted = []
        for _ in range(200):
            if random.random() < 0.6 or not inserted:
                v = random.randint(0, 1000)
                n = h.insert(v)
                inserted.append((v, n))
            else:
                h.delete_min()
                inserted.clear()  # lose references, that's ok
        # Just verify we can drain without error
        while h:
            h.delete_min()
        assert h.is_empty()
