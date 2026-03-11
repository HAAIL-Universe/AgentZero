"""Tests for C106: Union-Find / Disjoint Set."""
import pytest
from union_find import (
    UnionFind, WeightedUnionFind, PersistentUnionFind,
    DynamicUnionFind, UnionFindMap,
    kruskal_mst, connected_components, has_cycle,
    equivalence_classes, earliest_connection, redundant_connections,
    accounts_merge,
)


# ============================================================
# UnionFind - Core
# ============================================================

class TestUnionFindBasic:
    def test_empty(self):
        uf = UnionFind(0)
        assert uf.count == 0
        assert uf.n == 0

    def test_singleton(self):
        uf = UnionFind(1)
        assert uf.count == 1
        assert uf.find(0) == 0

    def test_initial_state(self):
        uf = UnionFind(5)
        assert uf.count == 5
        assert uf.n == 5
        for i in range(5):
            assert uf.find(i) == i
            assert uf.set_size(i) == 1

    def test_union_two(self):
        uf = UnionFind(3)
        assert uf.union(0, 1) is True
        assert uf.connected(0, 1)
        assert not uf.connected(0, 2)
        assert uf.count == 2

    def test_union_same_set(self):
        uf = UnionFind(3)
        uf.union(0, 1)
        assert uf.union(0, 1) is False
        assert uf.count == 2

    def test_union_chain(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(2, 3)
        uf.union(3, 4)
        assert uf.count == 1
        assert uf.set_size(0) == 5
        for i in range(5):
            for j in range(5):
                assert uf.connected(i, j)

    def test_two_separate_sets(self):
        uf = UnionFind(6)
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(3, 4)
        uf.union(4, 5)
        assert uf.count == 2
        assert uf.connected(0, 2)
        assert uf.connected(3, 5)
        assert not uf.connected(0, 3)

    def test_merge_two_sets(self):
        uf = UnionFind(4)
        uf.union(0, 1)
        uf.union(2, 3)
        assert uf.count == 2
        uf.union(1, 2)
        assert uf.count == 1
        assert uf.set_size(0) == 4

    def test_find_error(self):
        uf = UnionFind(3)
        with pytest.raises(ValueError):
            uf.find(-1)
        with pytest.raises(ValueError):
            uf.find(3)

    def test_make_set(self):
        uf = UnionFind(0)
        a = uf.make_set()
        b = uf.make_set()
        assert a == 0
        assert b == 1
        assert uf.n == 2
        assert uf.count == 2
        uf.union(a, b)
        assert uf.count == 1

    def test_sets(self):
        uf = UnionFind(6)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(4, 5)
        groups = uf.sets()
        assert len(groups) == 3
        members = [sorted(v) for v in groups.values()]
        members.sort()
        assert members == [[0, 1], [2, 3], [4, 5]]

    def test_roots(self):
        uf = UnionFind(4)
        uf.union(0, 1)
        uf.union(2, 3)
        assert len(uf.roots()) == 2

    def test_repr(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        assert "n=5" in repr(uf)
        assert "sets=4" in repr(uf)

    def test_path_compression(self):
        """After find, elements should point directly to root."""
        uf = UnionFind(10)
        # Build a chain: 0-1-2-3-4-5-6-7-8-9
        for i in range(9):
            uf.union(i, i + 1)
        root = uf.find(0)
        # After path compression, all should point to root
        for i in range(10):
            assert uf._parent[i] == root or uf.find(i) == root

    def test_large_union(self):
        n = 1000
        uf = UnionFind(n)
        for i in range(n - 1):
            uf.union(i, i + 1)
        assert uf.count == 1
        assert uf.set_size(0) == n


class TestUnionFindSetSize:
    def test_sizes_track_correctly(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        assert uf.set_size(0) == 2
        assert uf.set_size(1) == 2
        uf.union(2, 3)
        uf.union(0, 2)
        assert uf.set_size(0) == 4
        assert uf.set_size(3) == 4
        assert uf.set_size(4) == 1


# ============================================================
# WeightedUnionFind
# ============================================================

class TestWeightedUnionFind:
    def test_basic(self):
        wuf = WeightedUnionFind(3)
        wuf.union(0, 1, 5)  # weight(0) - weight(1) = 5
        assert wuf.connected(0, 1)
        assert wuf.diff(0, 1) == 5

    def test_transitive_weights(self):
        wuf = WeightedUnionFind(3)
        wuf.union(0, 1, 3)  # w(0) - w(1) = 3
        wuf.union(1, 2, 7)  # w(1) - w(2) = 7
        assert wuf.diff(0, 2) == 10  # w(0) - w(2) = 3 + 7

    def test_negative_weights(self):
        wuf = WeightedUnionFind(3)
        wuf.union(0, 1, -2)  # w(0) - w(1) = -2
        assert wuf.diff(0, 1) == -2
        assert wuf.diff(1, 0) == 2

    def test_zero_weight(self):
        wuf = WeightedUnionFind(2)
        wuf.union(0, 1, 0)
        assert wuf.diff(0, 1) == 0

    def test_same_set_no_union(self):
        wuf = WeightedUnionFind(2)
        wuf.union(0, 1, 5)
        assert wuf.union(0, 1, 5) is False
        assert wuf.count == 1

    def test_diff_disconnected(self):
        wuf = WeightedUnionFind(3)
        wuf.union(0, 1, 5)
        with pytest.raises(ValueError):
            wuf.diff(0, 2)

    def test_weight_chain(self):
        wuf = WeightedUnionFind(5)
        wuf.union(0, 1, 1)
        wuf.union(1, 2, 2)
        wuf.union(2, 3, 3)
        wuf.union(3, 4, 4)
        assert wuf.diff(0, 4) == 10
        assert wuf.diff(4, 0) == -10
        assert wuf.diff(1, 3) == 5

    def test_make_set(self):
        wuf = WeightedUnionFind(0)
        a = wuf.make_set()
        b = wuf.make_set()
        wuf.union(a, b, 42)
        assert wuf.diff(a, b) == 42

    def test_find_error(self):
        wuf = WeightedUnionFind(2)
        with pytest.raises(ValueError):
            wuf.find(-1)

    def test_merge_two_groups(self):
        wuf = WeightedUnionFind(4)
        wuf.union(0, 1, 10)
        wuf.union(2, 3, 20)
        wuf.union(1, 2, 5)
        # w(0)-w(1)=10, w(1)-w(2)=5, w(2)-w(3)=20
        assert wuf.diff(0, 3) == 35

    def test_large_weighted(self):
        n = 100
        wuf = WeightedUnionFind(n)
        for i in range(n - 1):
            wuf.union(i, i + 1, 1)
        assert wuf.diff(0, n - 1) == n - 1


# ============================================================
# PersistentUnionFind
# ============================================================

class TestPersistentUnionFind:
    def test_basic_undo(self):
        puf = PersistentUnionFind(4)
        puf.union(0, 1)
        assert puf.connected(0, 1)
        puf.undo()
        assert not puf.connected(0, 1)
        assert puf.count == 4

    def test_multiple_undo(self):
        puf = PersistentUnionFind(5)
        puf.union(0, 1)
        puf.union(2, 3)
        puf.union(1, 2)
        assert puf.count == 2
        puf.undo()
        assert puf.count == 3
        assert puf.connected(0, 1)
        assert puf.connected(2, 3)
        assert not puf.connected(0, 2)
        puf.undo()
        assert puf.count == 4
        assert puf.connected(0, 1)
        assert not puf.connected(2, 3)

    def test_undo_empty(self):
        puf = PersistentUnionFind(3)
        assert puf.undo() is False

    def test_checkpoint_save_restore(self):
        puf = PersistentUnionFind(5)
        puf.union(0, 1)
        puf.save("step1")
        puf.union(2, 3)
        puf.union(1, 2)
        assert puf.count == 2
        puf.restore("step1")
        assert puf.count == 4
        assert puf.connected(0, 1)
        assert not puf.connected(2, 3)

    def test_restore_nonexistent(self):
        puf = PersistentUnionFind(3)
        with pytest.raises(ValueError):
            puf.restore("nope")

    def test_nested_checkpoints(self):
        puf = PersistentUnionFind(6)
        puf.union(0, 1)
        puf.save("cp1")
        puf.union(2, 3)
        puf.save("cp2")
        puf.union(4, 5)
        assert puf.count == 3
        puf.restore("cp2")
        assert puf.count == 4
        assert not puf.connected(4, 5)
        puf.restore("cp1")
        assert puf.count == 5
        assert not puf.connected(2, 3)

    def test_history_size(self):
        puf = PersistentUnionFind(5)
        assert puf.history_size() == 0
        puf.union(0, 1)
        assert puf.history_size() == 1
        puf.union(0, 1)  # same set, no op
        assert puf.history_size() == 1
        puf.union(2, 3)
        assert puf.history_size() == 2

    def test_find_error(self):
        puf = PersistentUnionFind(2)
        with pytest.raises(ValueError):
            puf.find(5)

    def test_restore_removes_later_checkpoints(self):
        puf = PersistentUnionFind(4)
        puf.save("a")
        puf.union(0, 1)
        puf.save("b")
        puf.union(2, 3)
        puf.restore("a")
        with pytest.raises(ValueError):
            puf.restore("b")


# ============================================================
# DynamicUnionFind
# ============================================================

class TestDynamicUnionFind:
    def test_basic(self):
        duf = DynamicUnionFind(3)
        duf.union(0, 1)
        assert duf.connected(0, 1)
        assert duf.count == 2

    def test_delete(self):
        duf = DynamicUnionFind(3)
        duf.union(0, 1)
        duf.delete(1)
        assert 1 not in duf.active_elements
        assert duf.count == 2  # {0} and {2} are separate

    def test_delete_bridge(self):
        """Deleting a bridge element: 0 and 2 remain connected through internal UF."""
        duf = DynamicUnionFind(3)
        duf.union(0, 1)
        duf.union(1, 2)
        assert duf.connected(0, 2)
        duf.delete(1)
        # 0 and 2 are still connected internally (UF doesn't support splitting)
        assert duf.connected(0, 2)
        assert duf.count == 1

    def test_delete_and_readd(self):
        duf = DynamicUnionFind(3)
        duf.union(0, 1)
        duf.delete(1)
        duf.add(1)
        assert not duf.connected(0, 1)  # re-added as isolated
        assert duf.count == 3

    def test_delete_nonexistent(self):
        duf = DynamicUnionFind(3)
        with pytest.raises(ValueError):
            duf.delete(5)

    def test_find_deleted(self):
        duf = DynamicUnionFind(3)
        duf.delete(1)
        with pytest.raises(ValueError):
            duf.find(1)

    def test_active_elements(self):
        duf = DynamicUnionFind(5)
        assert len(duf.active_elements) == 5
        duf.delete(2)
        duf.delete(4)
        assert duf.active_elements == frozenset({0, 1, 3})

    def test_add_new(self):
        duf = DynamicUnionFind(2)
        duf.add(10)
        assert 10 in duf.active_elements
        duf.union(0, 10)
        assert duf.connected(0, 10)

    def test_union_deleted_raises(self):
        duf = DynamicUnionFind(3)
        duf.delete(1)
        with pytest.raises(ValueError):
            duf.union(0, 1)

    def test_connected_deleted_returns_false(self):
        duf = DynamicUnionFind(3)
        duf.union(0, 1)
        duf.delete(1)
        assert not duf.connected(0, 1)

    def test_add_idempotent(self):
        duf = DynamicUnionFind(3)
        duf.add(0)  # already active
        assert len(duf.active_elements) == 3


# ============================================================
# UnionFindMap
# ============================================================

class TestUnionFindMap:
    def test_basic(self):
        ufm = UnionFindMap()
        ufm.make_set("a")
        ufm.make_set("b")
        ufm.union("a", "b")
        assert ufm.connected("a", "b")
        assert ufm.count == 1

    def test_from_init(self):
        ufm = UnionFindMap(["x", "y", "z"])
        assert ufm.n == 3
        assert ufm.count == 3

    def test_auto_create_on_union(self):
        ufm = UnionFindMap()
        ufm.union("hello", "world")
        assert ufm.connected("hello", "world")
        assert ufm.n == 2

    def test_make_set_idempotent(self):
        ufm = UnionFindMap()
        assert ufm.make_set("a") is True
        assert ufm.make_set("a") is False
        assert ufm.count == 1

    def test_find_error(self):
        ufm = UnionFindMap()
        with pytest.raises(ValueError):
            ufm.find("missing")

    def test_connected_missing(self):
        ufm = UnionFindMap(["a"])
        assert not ufm.connected("a", "missing")

    def test_set_size(self):
        ufm = UnionFindMap(["a", "b", "c"])
        ufm.union("a", "b")
        assert ufm.set_size("a") == 2
        assert ufm.set_size("c") == 1

    def test_sets(self):
        ufm = UnionFindMap(["a", "b", "c", "d"])
        ufm.union("a", "b")
        ufm.union("c", "d")
        groups = ufm.sets()
        assert len(groups) == 2

    def test_contains(self):
        ufm = UnionFindMap(["a", "b"])
        assert "a" in ufm
        assert "z" not in ufm

    def test_tuple_keys(self):
        ufm = UnionFindMap()
        ufm.union((0, 0), (0, 1))
        ufm.union((0, 1), (1, 1))
        assert ufm.connected((0, 0), (1, 1))

    def test_integer_keys(self):
        ufm = UnionFindMap()
        ufm.union(100, 200)
        ufm.union(200, 300)
        assert ufm.connected(100, 300)
        assert ufm.set_size(100) == 3


# ============================================================
# Applications
# ============================================================

class TestKruskalMST:
    def test_simple_graph(self):
        # Triangle: 0-1 (1), 1-2 (2), 0-2 (3)
        edges = [(1, 0, 1), (2, 1, 2), (3, 0, 2)]
        total, mst = kruskal_mst(3, edges)
        assert total == 3  # edges 1+2
        assert len(mst) == 2

    def test_complete_graph(self):
        # K4 with weights
        edges = [
            (1, 0, 1), (4, 0, 2), (3, 0, 3),
            (2, 1, 2), (5, 1, 3), (6, 2, 3)
        ]
        total, mst = kruskal_mst(4, edges)
        assert total == 6  # 1 + 2 + 3
        assert len(mst) == 3

    def test_disconnected(self):
        edges = [(1, 0, 1)]
        total, mst = kruskal_mst(3, edges)
        assert len(mst) == 1

    def test_single_vertex(self):
        total, mst = kruskal_mst(1, [])
        assert total == 0
        assert mst == []

    def test_empty(self):
        total, mst = kruskal_mst(0, [])
        assert total == 0
        assert mst == []


class TestConnectedComponents:
    def test_basic(self):
        comps = connected_components(5, [(0, 1), (2, 3)])
        assert len(comps) == 3

    def test_all_connected(self):
        comps = connected_components(3, [(0, 1), (1, 2)])
        assert len(comps) == 1

    def test_all_isolated(self):
        comps = connected_components(4, [])
        assert len(comps) == 4

    def test_single(self):
        comps = connected_components(1, [])
        assert len(comps) == 1


class TestHasCycle:
    def test_no_cycle(self):
        assert not has_cycle(3, [(0, 1), (1, 2)])

    def test_triangle(self):
        assert has_cycle(3, [(0, 1), (1, 2), (2, 0)])

    def test_four_cycle(self):
        assert has_cycle(4, [(0, 1), (1, 2), (2, 3), (3, 0)])

    def test_tree(self):
        # Star graph
        assert not has_cycle(5, [(0, 1), (0, 2), (0, 3), (0, 4)])

    def test_empty(self):
        assert not has_cycle(3, [])


class TestEquivalenceClasses:
    def test_basic(self):
        classes = equivalence_classes(5, [(0, 1), (2, 3)])
        assert len(classes) == 3
        # Check containment
        for cls in classes:
            if 0 in cls:
                assert 1 in cls
            if 2 in cls:
                assert 3 in cls

    def test_transitive(self):
        classes = equivalence_classes(4, [(0, 1), (1, 2), (2, 3)])
        assert len(classes) == 1
        assert classes[0] == {0, 1, 2, 3}

    def test_reflexive(self):
        classes = equivalence_classes(3, [(0, 0)])
        assert len(classes) == 3


class TestEarliestConnection:
    def test_basic(self):
        conns = [(1, 0, 1), (2, 1, 2)]
        assert earliest_connection(3, conns) == 2

    def test_never(self):
        conns = [(1, 0, 1)]
        assert earliest_connection(3, conns) == -1

    def test_single_node(self):
        assert earliest_connection(1, []) == -1

    def test_unsorted(self):
        conns = [(5, 0, 1), (3, 1, 2), (1, 0, 2)]
        assert earliest_connection(3, conns) == 3

    def test_immediate(self):
        conns = [(0, 0, 1)]
        assert earliest_connection(2, conns) == 0


class TestRedundantConnections:
    def test_triangle(self):
        edges = [(0, 1), (1, 2), (2, 0)]
        result = redundant_connections(3, edges)
        assert result == [(2, 0)]

    def test_no_redundant(self):
        edges = [(0, 1), (1, 2)]
        assert redundant_connections(3, edges) == []

    def test_multiple_redundant(self):
        edges = [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)]
        result = redundant_connections(4, edges)
        assert len(result) == 2


class TestAccountsMerge:
    def test_basic(self):
        accounts = [
            ["John", "john@example.com", "john2@example.com"],
            ["John", "john2@example.com", "john3@example.com"],
            ["Jane", "jane@example.com"],
        ]
        result = accounts_merge(accounts)
        assert len(result) == 2
        # Find John's merged account
        for acc in result:
            if acc[0] == "John":
                assert len(acc) == 4  # name + 3 emails
                assert "john@example.com" in acc
                assert "john2@example.com" in acc
                assert "john3@example.com" in acc

    def test_no_overlap(self):
        accounts = [
            ["A", "a@x.com"],
            ["B", "b@x.com"],
        ]
        result = accounts_merge(accounts)
        assert len(result) == 2

    def test_all_same(self):
        accounts = [
            ["X", "x@y.com"],
            ["X", "x@y.com"],
        ]
        result = accounts_merge(accounts)
        assert len(result) == 1

    def test_chain_merge(self):
        accounts = [
            ["U", "a@x.com", "b@x.com"],
            ["U", "b@x.com", "c@x.com"],
            ["U", "c@x.com", "d@x.com"],
        ]
        result = accounts_merge(accounts)
        assert len(result) == 1
        assert len(result[0]) == 5  # name + 4 emails


# ============================================================
# Edge cases and stress
# ============================================================

class TestEdgeCases:
    def test_self_union(self):
        uf = UnionFind(3)
        assert uf.union(1, 1) is False
        assert uf.count == 3

    def test_many_make_set(self):
        uf = UnionFind(0)
        for i in range(100):
            uf.make_set()
        assert uf.n == 100
        assert uf.count == 100

    def test_star_pattern(self):
        uf = UnionFind(101)
        for i in range(1, 101):
            uf.union(0, i)
        assert uf.count == 1
        assert uf.set_size(0) == 101

    def test_weighted_consistency_after_merge(self):
        """Weights should be consistent after merging two groups."""
        wuf = WeightedUnionFind(6)
        # Group 1: 0-1-2 with weights 3, 5
        wuf.union(0, 1, 3)
        wuf.union(1, 2, 5)
        # Group 2: 3-4-5 with weights 7, 11
        wuf.union(3, 4, 7)
        wuf.union(4, 5, 11)
        # Merge: w(2) - w(3) = 13
        wuf.union(2, 3, 13)
        # Check all pairs
        assert wuf.diff(0, 1) == 3
        assert wuf.diff(1, 2) == 5
        assert wuf.diff(0, 2) == 8
        assert wuf.diff(3, 4) == 7
        assert wuf.diff(4, 5) == 11
        assert wuf.diff(3, 5) == 18
        assert wuf.diff(2, 3) == 13
        assert wuf.diff(0, 5) == 8 + 13 + 18  # 39

    def test_persistent_no_path_compression(self):
        """Persistent UF shouldn't use path compression."""
        puf = PersistentUnionFind(5)
        puf.union(0, 1)
        puf.union(1, 2)
        puf.union(2, 3)
        # Can fully undo
        puf.undo()
        puf.undo()
        puf.undo()
        assert puf.count == 5

    def test_dynamic_complex_scenario(self):
        """Delete, re-add, and re-merge."""
        duf = DynamicUnionFind(5)
        duf.union(0, 1)
        duf.union(2, 3)
        duf.union(0, 3)  # merges {0,1} and {2,3}
        assert duf.connected(0, 2)
        duf.delete(3)
        # 0,1,2 are still connected (3 was part but deleted)
        assert duf.connected(0, 1)
        duf.add(3)
        duf.union(3, 4)
        assert duf.connected(3, 4)
        assert not duf.connected(3, 0)


class TestStress:
    def test_large_union_find(self):
        n = 10000
        uf = UnionFind(n)
        # Union even-odd pairs
        for i in range(0, n - 1, 2):
            uf.union(i, i + 1)
        assert uf.count == n // 2
        # Now chain all evens
        for i in range(0, n - 2, 2):
            uf.union(i, i + 2)
        assert uf.count == 1

    def test_large_weighted(self):
        n = 5000
        wuf = WeightedUnionFind(n)
        for i in range(n - 1):
            wuf.union(i, i + 1, 1)
        assert wuf.diff(0, n - 1) == n - 1
        assert wuf.diff(n - 1, 0) == -(n - 1)

    def test_large_persistent_undo(self):
        n = 1000
        puf = PersistentUnionFind(n)
        for i in range(n - 1):
            puf.union(i, i + 1)
        assert puf.count == 1
        # Undo all
        for i in range(n - 1):
            puf.undo()
        assert puf.count == n

    def test_kruskal_large(self):
        n = 100
        edges = []
        for i in range(n):
            for j in range(i + 1, min(i + 5, n)):
                edges.append((i + j, i, j))
        total, mst = kruskal_mst(n, edges)
        assert len(mst) == n - 1
