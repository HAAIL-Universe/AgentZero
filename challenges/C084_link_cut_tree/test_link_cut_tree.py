"""Tests for C084: Link-Cut Tree."""

import pytest
import random
from link_cut_tree import (
    LinkCutTree, PathAggregateTree, WeightedLinkCutTree,
    LinkCutForest, SplayNode, _splay, _access, _make_root,
    _is_root, _push_down, _pull_up
)


# ============================================================
# SplayNode basics
# ============================================================

class TestSplayNode:
    def test_create_default(self):
        n = SplayNode(0)
        assert n.id == 0
        assert n.value == 0
        assert n.left is None
        assert n.right is None
        assert n.parent is None
        assert n.rev is False
        assert n.size == 1

    def test_create_with_value(self):
        n = SplayNode(5, 42)
        assert n.id == 5
        assert n.value == 42
        assert n.agg_sum == 42
        assert n.agg_min == 42
        assert n.agg_max == 42

    def test_is_root_isolated(self):
        n = SplayNode(0)
        assert _is_root(n)

    def test_is_root_with_parent(self):
        p = SplayNode(0)
        c = SplayNode(1)
        p.left = c
        c.parent = p
        assert not _is_root(c)
        assert _is_root(p)

    def test_is_root_path_parent(self):
        """Path-parent: parent exists but doesn't have us as child."""
        p = SplayNode(0)
        c = SplayNode(1)
        c.parent = p  # path-parent link
        assert _is_root(c)  # still root of its splay tree

    def test_push_down_reversal(self):
        p = SplayNode(0)
        l = SplayNode(1)
        r = SplayNode(2)
        p.left = l
        p.right = r
        l.parent = p
        r.parent = p
        p.rev = True
        _push_down(p)
        assert p.left is r
        assert p.right is l
        assert p.rev is False

    def test_pull_up_aggregates(self):
        p = SplayNode(0, 10)
        l = SplayNode(1, 3)
        r = SplayNode(2, 7)
        p.left = l
        p.right = r
        _pull_up(p)
        assert p.size == 3
        assert p.agg_sum == 20
        assert p.agg_min == 3
        assert p.agg_max == 10


# ============================================================
# LinkCutTree -- basic operations
# ============================================================

class TestLinkCutTree:
    def test_create_empty(self):
        t = LinkCutTree()
        assert len(t.nodes) == 0

    def test_create_n_nodes(self):
        t = LinkCutTree(5)
        assert len(t.nodes) == 5

    def test_create_with_values(self):
        t = LinkCutTree(3, [10, 20, 30])
        assert t.get_value(0) == 10
        assert t.get_value(1) == 20
        assert t.get_value(2) == 30

    def test_add_node(self):
        t = LinkCutTree()
        t.add_node(42, 100)
        assert t.get_value(42) == 100

    def test_add_duplicate_node(self):
        t = LinkCutTree(3)
        with pytest.raises(ValueError, match="already exists"):
            t.add_node(1)

    def test_get_nonexistent(self):
        t = LinkCutTree(3)
        with pytest.raises(ValueError, match="does not exist"):
            t.get_value(99)

    def test_set_value(self):
        t = LinkCutTree(3, [1, 2, 3])
        t.set_value(1, 50)
        assert t.get_value(1) == 50

    # -- link --

    def test_link_two_nodes(self):
        t = LinkCutTree(2)
        t.link(0, 1)
        assert t.connected(0, 1)

    def test_link_chain(self):
        t = LinkCutTree(4)
        t.link(0, 1)
        t.link(1, 2)
        t.link(2, 3)
        assert t.connected(0, 3)

    def test_link_already_connected(self):
        t = LinkCutTree(3)
        t.link(0, 1)
        t.link(1, 2)
        with pytest.raises(ValueError, match="already connected"):
            t.link(0, 2)

    def test_link_star(self):
        t = LinkCutTree(5)
        for i in range(1, 5):
            t.link(i, 0)
        for i in range(1, 5):
            assert t.connected(i, 0)

    # -- cut --

    def test_cut_edge(self):
        t = LinkCutTree(3)
        t.link(0, 1)
        t.link(1, 2)
        t.cut(1, 2)
        assert t.connected(0, 1)
        assert not t.connected(0, 2)

    def test_cut_nonexistent_edge(self):
        t = LinkCutTree(3)
        t.link(0, 1)
        t.link(1, 2)
        with pytest.raises(ValueError, match="No direct edge"):
            t.cut(0, 2)

    def test_cut_and_relink(self):
        t = LinkCutTree(4)
        t.link(0, 1)
        t.link(1, 2)
        t.link(2, 3)
        t.cut(1, 2)
        t.link(1, 3)  # Relink 1 to tree containing 3
        assert t.connected(0, 3)

    # -- connected --

    def test_connected_self(self):
        t = LinkCutTree(3)
        assert t.connected(0, 0)

    def test_not_connected(self):
        t = LinkCutTree(3)
        assert not t.connected(0, 1)

    def test_connected_after_operations(self):
        t = LinkCutTree(5)
        t.link(0, 1)
        t.link(2, 3)
        assert t.connected(0, 1)
        assert t.connected(2, 3)
        assert not t.connected(0, 2)
        t.link(1, 2)
        assert t.connected(0, 3)

    # -- find_root --

    def test_find_root_isolated(self):
        t = LinkCutTree(3)
        assert t.find_root(0) == 0

    def test_find_root_chain(self):
        t = LinkCutTree(4)
        t.link(0, 1)
        t.link(1, 2)
        t.link(2, 3)
        # After linking, root depends on evert state
        # find_root returns the root of the represented tree
        root = t.find_root(0)
        # All should have same root
        assert t.find_root(1) == root
        assert t.find_root(2) == root
        assert t.find_root(3) == root

    def test_find_root_after_evert(self):
        t = LinkCutTree(4)
        t.link(0, 1)
        t.link(1, 2)
        t.link(2, 3)
        t.evert(0)
        assert t.find_root(3) == 0

    # -- evert --

    def test_evert_makes_root(self):
        t = LinkCutTree(3)
        t.link(0, 1)
        t.link(1, 2)
        t.evert(2)
        assert t.find_root(0) == 2
        assert t.find_root(1) == 2

    def test_evert_preserves_connectivity(self):
        t = LinkCutTree(5)
        t.link(0, 1)
        t.link(1, 2)
        t.link(2, 3)
        t.link(3, 4)
        t.evert(4)
        for i in range(5):
            assert t.connected(0, i)

    # -- lca --

    def test_lca_same_node(self):
        t = LinkCutTree(3)
        assert t.lca(1, 1) == 1

    def test_lca_parent_child(self):
        t = LinkCutTree(3)
        t.link(0, 1)
        t.link(1, 2)
        t.evert(2)  # root at 2
        assert t.lca(0, 1) == 1

    def test_lca_siblings(self):
        t = LinkCutTree(4)
        t.link(1, 0)
        t.link(2, 0)
        t.link(3, 0)
        t.evert(0)
        assert t.lca(1, 2) == 0

    def test_lca_not_connected(self):
        t = LinkCutTree(3)
        t.link(0, 1)
        assert t.lca(0, 2) is None

    def test_lca_deep_tree(self):
        t = LinkCutTree(7)
        # Build tree:
        #        0
        #       / \
        #      1   2
        #     / \
        #    3   4
        #   /
        #  5
        t.link(1, 0)
        t.link(2, 0)
        t.link(3, 1)
        t.link(4, 1)
        t.link(5, 3)
        t.link(6, 3)
        t.evert(0)
        assert t.lca(5, 4) == 1
        assert t.lca(5, 6) == 3
        assert t.lca(5, 2) == 0

    # -- path_aggregate --

    def test_path_aggregate_single(self):
        t = LinkCutTree(1, [42])
        s, mn, mx = t.path_aggregate(0, 0)
        assert s == 42
        assert mn == 42
        assert mx == 42

    def test_path_aggregate_chain(self):
        t = LinkCutTree(4, [1, 2, 3, 4])
        t.link(0, 1)
        t.link(1, 2)
        t.link(2, 3)
        s, mn, mx = t.path_aggregate(0, 3)
        assert s == 10
        assert mn == 1
        assert mx == 4

    def test_path_aggregate_partial(self):
        t = LinkCutTree(4, [10, 20, 30, 40])
        t.link(0, 1)
        t.link(1, 2)
        t.link(2, 3)
        s, mn, mx = t.path_aggregate(1, 2)
        assert s == 50
        assert mn == 20
        assert mx == 30

    def test_path_aggregate_after_value_change(self):
        t = LinkCutTree(3, [1, 2, 3])
        t.link(0, 1)
        t.link(1, 2)
        t.set_value(1, 100)
        s, mn, mx = t.path_aggregate(0, 2)
        assert s == 104
        assert mn == 1
        assert mx == 100

    # -- path_length --

    def test_path_length_same(self):
        t = LinkCutTree(3)
        assert t.path_length(0, 0) == 0

    def test_path_length_adjacent(self):
        t = LinkCutTree(2)
        t.link(0, 1)
        assert t.path_length(0, 1) == 1

    def test_path_length_chain(self):
        t = LinkCutTree(5)
        for i in range(4):
            t.link(i, i + 1)
        assert t.path_length(0, 4) == 4
        assert t.path_length(1, 3) == 2

    # -- subtree operations --

    def test_subtree_size_isolated(self):
        t = LinkCutTree(3)
        assert t.subtree_size(0) == 1

    def test_subtree_size_tree(self):
        t = LinkCutTree(5)
        t.link(1, 0)
        t.link(2, 0)
        t.link(3, 1)
        t.link(4, 1)
        t.evert(0)
        assert t.subtree_size(0) == 5
        assert t.subtree_size(1) == 3  # 1, 3, 4

    def test_subtree_sum(self):
        t = LinkCutTree(4, [10, 20, 30, 40])
        t.link(1, 0)
        t.link(2, 0)
        t.link(3, 1)
        t.evert(0)
        assert t.subtree_sum(0) == 100
        assert t.subtree_sum(1) == 60  # 20 + 40


# ============================================================
# PathAggregateTree
# ============================================================

class TestPathAggregateTree:
    def test_path_sum(self):
        t = PathAggregateTree(4, [5, 10, 15, 20])
        t.link(0, 1)
        t.link(1, 2)
        t.link(2, 3)
        assert t.path_sum(0, 3) == 50

    def test_path_min(self):
        t = PathAggregateTree(4, [5, 10, 15, 20])
        t.link(0, 1)
        t.link(1, 2)
        t.link(2, 3)
        assert t.path_min(0, 3) == 5

    def test_path_max(self):
        t = PathAggregateTree(4, [5, 10, 15, 20])
        t.link(0, 1)
        t.link(1, 2)
        t.link(2, 3)
        assert t.path_max(0, 3) == 20

    def test_path_update(self):
        t = PathAggregateTree(4, [1, 2, 3, 4])
        t.link(0, 1)
        t.link(1, 2)
        t.link(2, 3)
        t.path_update(1, 2, 10)
        assert t.path_sum(1, 2) == 25  # 12 + 13
        assert t.get_value(0) == 1  # unchanged
        assert t.get_value(3) == 4  # unchanged

    def test_path_update_single(self):
        t = PathAggregateTree(3, [5, 10, 15])
        t.link(0, 1)
        t.link(1, 2)
        t.path_update(1, 1, 100)
        assert t.get_value(1) == 110

    def test_path_aggregate_star(self):
        # Star topology: center = 0
        t = PathAggregateTree(5, [100, 1, 2, 3, 4])
        for i in range(1, 5):
            t.link(i, 0)
        assert t.path_sum(1, 2) == 103  # 1 + 100 + 2
        assert t.path_min(1, 3) == 1
        assert t.path_max(3, 4) == 100


# ============================================================
# WeightedLinkCutTree
# ============================================================

class TestWeightedLinkCutTree:
    def test_create(self):
        t = WeightedLinkCutTree(3)
        assert len(t.nodes) == 3

    def test_link_with_weight(self):
        t = WeightedLinkCutTree(3)
        t.link(0, 1, 5)
        t.link(1, 2, 10)
        assert t.connected(0, 2)

    def test_path_weight(self):
        t = WeightedLinkCutTree(4)
        t.link(0, 1, 3)
        t.link(1, 2, 7)
        t.link(2, 3, 2)
        assert t.path_weight(0, 3) == 12

    def test_path_weight_partial(self):
        t = WeightedLinkCutTree(4)
        t.link(0, 1, 3)
        t.link(1, 2, 7)
        t.link(2, 3, 2)
        assert t.path_weight(1, 3) == 9

    def test_path_weight_same_node(self):
        t = WeightedLinkCutTree(2)
        t.link(0, 1, 5)
        assert t.path_weight(0, 0) == 0

    def test_min_edge(self):
        t = WeightedLinkCutTree(4)
        t.link(0, 1, 3)
        t.link(1, 2, 7)
        t.link(2, 3, 1)
        assert t.min_edge(0, 3) == 1

    def test_max_edge(self):
        t = WeightedLinkCutTree(4)
        t.link(0, 1, 3)
        t.link(1, 2, 7)
        t.link(2, 3, 1)
        assert t.max_edge(0, 3) == 7

    def test_min_max_same_node(self):
        t = WeightedLinkCutTree(1)
        assert t.min_edge(0, 0) == 0
        assert t.max_edge(0, 0) == 0

    def test_cut_weighted(self):
        t = WeightedLinkCutTree(3)
        t.link(0, 1, 5)
        t.link(1, 2, 10)
        t.cut(1, 2)
        assert not t.connected(0, 2)
        assert t.connected(0, 1)
        assert t.path_weight(0, 1) == 5

    def test_add_node(self):
        t = WeightedLinkCutTree(2)
        t.add_node(10)
        t.link(10, 0, 42)
        assert t.connected(10, 0)
        assert t.path_weight(10, 0) == 42

    def test_add_duplicate(self):
        t = WeightedLinkCutTree(2)
        with pytest.raises(ValueError):
            t.add_node(0)

    def test_find_root(self):
        t = WeightedLinkCutTree(3)
        t.link(0, 1, 1)
        t.link(1, 2, 2)
        root = t.find_root(0)
        assert t.find_root(1) == root
        assert t.find_root(2) == root

    def test_already_connected(self):
        t = WeightedLinkCutTree(3)
        t.link(0, 1, 1)
        t.link(1, 2, 2)
        with pytest.raises(ValueError, match="already connected"):
            t.link(0, 2, 3)


# ============================================================
# LinkCutForest
# ============================================================

class TestLinkCutForest:
    def test_create(self):
        f = LinkCutForest(5)
        assert f.num_components == 5

    def test_link_reduces_components(self):
        f = LinkCutForest(5)
        f.link(0, 1)
        assert f.num_components == 4
        f.link(1, 2)
        assert f.num_components == 3

    def test_link_already_connected(self):
        f = LinkCutForest(3)
        f.link(0, 1)
        f.link(1, 2)
        result = f.link(0, 2)
        assert result is False
        assert f.num_components == 1

    def test_cut_increases_components(self):
        f = LinkCutForest(3)
        f.link(0, 1)
        f.link(1, 2)
        f.cut(1, 2)
        assert f.num_components == 2

    def test_add_node(self):
        f = LinkCutForest(2)
        f.add_node(10, 42)
        assert f.num_components == 3
        assert f.get_value(10) == 42

    def test_connected(self):
        f = LinkCutForest(4)
        f.link(0, 1)
        f.link(2, 3)
        assert f.connected(0, 1)
        assert f.connected(2, 3)
        assert not f.connected(0, 2)

    def test_get_components(self):
        f = LinkCutForest(5)
        f.link(0, 1)
        f.link(2, 3)
        components = f.get_components()
        assert len(components) == 3  # {0,1}, {2,3}, {4}
        sizes = sorted(len(c) for c in components)
        assert sizes == [1, 2, 2]

    def test_evert(self):
        f = LinkCutForest(3)
        f.link(0, 1)
        f.link(1, 2)
        f.evert(2)
        assert f.find_root(0) == 2

    def test_path_aggregate(self):
        f = LinkCutForest(3, [10, 20, 30])
        f.link(0, 1)
        f.link(1, 2)
        s, mn, mx = f.path_aggregate(0, 2)
        assert s == 60

    def test_path_length(self):
        f = LinkCutForest(4)
        f.link(0, 1)
        f.link(1, 2)
        f.link(2, 3)
        assert f.path_length(0, 3) == 3

    def test_subtree_size(self):
        f = LinkCutForest(4)
        f.link(1, 0)
        f.link(2, 0)
        f.link(3, 1)
        f.evert(0)
        assert f.subtree_size(0) == 4

    def test_subtree_sum(self):
        f = LinkCutForest(3, [5, 10, 15])
        f.link(1, 0)
        f.link(2, 0)
        f.evert(0)
        assert f.subtree_sum(0) == 30


# ============================================================
# Stress tests and complex scenarios
# ============================================================

class TestStress:
    def test_long_chain(self):
        """Build a chain of 100 nodes, test connectivity and path queries."""
        n = 100
        t = LinkCutTree(n, list(range(n)))
        for i in range(n - 1):
            t.link(i, i + 1)
        assert t.connected(0, 99)
        assert t.path_length(0, 99) == 99
        s, mn, mx = t.path_aggregate(0, 99)
        assert s == sum(range(100))
        assert mn == 0
        assert mx == 99

    def test_star_topology(self):
        """Star with 50 leaves."""
        n = 51
        t = LinkCutTree(n, [1] * n)
        for i in range(1, n):
            t.link(i, 0)
        for i in range(1, n):
            assert t.connected(i, 0)
            assert t.path_length(i, 0) == 1
        # Path between two leaves goes through center
        assert t.path_length(1, 2) == 2

    def test_binary_tree(self):
        """Build a complete binary tree."""
        n = 31  # 5 levels
        t = LinkCutTree(n)
        for i in range(1, n):
            parent = (i - 1) // 2
            t.link(i, parent)
        # All connected
        for i in range(n):
            assert t.connected(0, i)
        # Leaves to root
        t.evert(0)
        assert t.path_length(0, 30) == 4  # depth 4

    def test_dynamic_forest(self):
        """Link and cut repeatedly, maintaining connectivity."""
        n = 20
        t = LinkCutTree(n)
        edges = []
        # Build initial tree
        for i in range(1, n):
            t.link(i, i - 1)
            edges.append((i, i - 1))
        assert t.connected(0, n - 1)

        # Cut some edges and verify
        t.cut(5, 4)
        assert not t.connected(0, 5)
        assert t.connected(0, 4)
        assert t.connected(5, 19)

        t.cut(10, 9)
        assert not t.connected(5, 10)
        assert t.connected(10, 19)

        # Relink
        t.link(5, 3)
        assert t.connected(0, 9)

    def test_random_operations(self):
        """Random link/cut/query operations."""
        random.seed(42)
        n = 30
        t = LinkCutTree(n, [random.randint(1, 100) for _ in range(n)])
        edges = set()

        for _ in range(100):
            op = random.choice(['link', 'cut', 'connected', 'find_root', 'value'])
            if op == 'link' and len(edges) < n - 1:
                # Try to link two disconnected nodes
                u, v = random.sample(range(n), 2)
                if not t.connected(u, v):
                    t.link(u, v)
                    edges.add((min(u, v), max(u, v)))
            elif op == 'cut' and edges:
                u, v = random.choice(list(edges))
                t.cut(u, v)
                edges.discard((u, v))
            elif op == 'connected':
                u, v = random.sample(range(n), 2)
                t.connected(u, v)  # just verify no crash
            elif op == 'find_root':
                u = random.randint(0, n - 1)
                t.find_root(u)
            elif op == 'value':
                u = random.randint(0, n - 1)
                t.set_value(u, random.randint(1, 100))
                t.get_value(u)

    def test_evert_chain(self):
        """Evert repeatedly on a chain."""
        n = 20
        t = LinkCutTree(n)
        for i in range(n - 1):
            t.link(i, i + 1)
        # Evert each node as root
        for i in range(n):
            t.evert(i)
            assert t.find_root(0) == i
            assert t.connected(0, n - 1)

    def test_repeated_access_same_node(self):
        """Repeated access shouldn't break anything."""
        t = LinkCutTree(5, [1, 2, 3, 4, 5])
        for i in range(4):
            t.link(i, i + 1)
        for _ in range(50):
            t.find_root(0)
            t.path_aggregate(0, 4)
            t.connected(0, 4)

    def test_link_cut_cycle(self):
        """Repeatedly link and cut the same edge."""
        t = LinkCutTree(2, [10, 20])
        for _ in range(20):
            t.link(0, 1)
            assert t.connected(0, 1)
            s, _, _ = t.path_aggregate(0, 1)
            assert s == 30
            t.cut(0, 1)
            assert not t.connected(0, 1)

    def test_caterpillar_tree(self):
        """Caterpillar: chain with leaves at each internal node."""
        spine = 10
        t = LinkCutTree(spine * 2)
        # Build spine
        for i in range(spine - 1):
            t.link(i, i + 1)
        # Attach leaves
        for i in range(spine):
            t.link(spine + i, i)
        assert t.connected(0, spine * 2 - 1)
        t.evert(0)
        # Leaf to leaf across spine
        assert t.path_length(spine, spine * 2 - 1) == spine + 1


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_single_node(self):
        t = LinkCutTree(1, [42])
        assert t.find_root(0) == 0
        assert t.get_value(0) == 42
        assert t.path_length(0, 0) == 0
        assert t.subtree_size(0) == 1
        assert t.subtree_sum(0) == 42

    def test_two_nodes(self):
        t = LinkCutTree(2, [3, 7])
        t.link(0, 1)
        assert t.connected(0, 1)
        s, mn, mx = t.path_aggregate(0, 1)
        assert s == 10
        assert mn == 3
        assert mx == 7

    def test_negative_values(self):
        t = LinkCutTree(3, [-5, -10, -3])
        t.link(0, 1)
        t.link(1, 2)
        s, mn, mx = t.path_aggregate(0, 2)
        assert s == -18
        assert mn == -10
        assert mx == -3

    def test_zero_values(self):
        t = LinkCutTree(3, [0, 0, 0])
        t.link(0, 1)
        t.link(1, 2)
        s, mn, mx = t.path_aggregate(0, 2)
        assert s == 0

    def test_large_values(self):
        t = LinkCutTree(3, [10**9, 10**9, 10**9])
        t.link(0, 1)
        t.link(1, 2)
        s, _, _ = t.path_aggregate(0, 2)
        assert s == 3 * 10**9

    def test_string_node_ids(self):
        """Support non-integer node IDs."""
        t = LinkCutTree()
        t.add_node("a", 1)
        t.add_node("b", 2)
        t.add_node("c", 3)
        t.link("a", "b")
        t.link("b", "c")
        assert t.connected("a", "c")
        s, _, _ = t.path_aggregate("a", "c")
        assert s == 6

    def test_lca_after_evert(self):
        """LCA should work correctly after re-rooting."""
        t = LinkCutTree(5)
        t.link(1, 0)
        t.link(2, 0)
        t.link(3, 1)
        t.link(4, 2)
        t.evert(0)
        assert t.lca(3, 4) == 0
        t.evert(1)
        assert t.lca(3, 4) == 1  # 1 is now root, path 3->1->0->2->4


# ============================================================
# Correctness verification against naive implementation
# ============================================================

class TestCorrectness:
    """Verify link-cut tree against naive adjacency list implementation."""

    def _naive_connected(self, adj, u, v, n):
        """BFS connectivity check."""
        visited = set()
        queue = [u]
        visited.add(u)
        while queue:
            curr = queue.pop(0)
            if curr == v:
                return True
            for nb in adj.get(curr, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return False

    def _naive_path(self, adj, u, v):
        """BFS to find path, return list of nodes."""
        parent = {u: None}
        queue = [u]
        while queue:
            curr = queue.pop(0)
            if curr == v:
                path = []
                while curr is not None:
                    path.append(curr)
                    curr = parent[curr]
                return path[::-1]
            for nb in adj.get(curr, []):
                if nb not in parent:
                    parent[nb] = curr
                    queue.append(nb)
        return None

    def test_random_vs_naive(self):
        """Random operations: compare LCT results against naive graph."""
        random.seed(123)
        n = 25
        values = [random.randint(1, 50) for _ in range(n)]
        t = LinkCutTree(n, values)
        adj = {i: [] for i in range(n)}

        for _ in range(200):
            op = random.choice(['link', 'cut', 'connected', 'path_sum'])
            if op == 'link':
                u, v = random.sample(range(n), 2)
                if not self._naive_connected(adj, u, v, n):
                    t.link(u, v)
                    adj[u].append(v)
                    adj[v].append(u)
            elif op == 'cut':
                # Find an edge to cut
                all_edges = []
                for a in adj:
                    for b in adj[a]:
                        if a < b:
                            all_edges.append((a, b))
                if all_edges:
                    u, v = random.choice(all_edges)
                    t.cut(u, v)
                    adj[u].remove(v)
                    adj[v].remove(u)
            elif op == 'connected':
                u, v = random.sample(range(n), 2)
                lct_result = t.connected(u, v)
                naive_result = self._naive_connected(adj, u, v, n)
                assert lct_result == naive_result, f"connected({u},{v}): LCT={lct_result}, naive={naive_result}"
            elif op == 'path_sum':
                u, v = random.sample(range(n), 2)
                if self._naive_connected(adj, u, v, n):
                    path = self._naive_path(adj, u, v)
                    naive_sum = sum(values[node] for node in path)
                    lct_sum, _, _ = t.path_aggregate(u, v)
                    assert lct_sum == naive_sum, f"path_sum({u},{v}): LCT={lct_sum}, naive={naive_sum}"


# ============================================================
# Weighted correctness
# ============================================================

class TestWeightedCorrectness:
    def test_path_weight_triangle_minus_one(self):
        """Build 0-1-2, weights 3 and 7. Verify path weight."""
        t = WeightedLinkCutTree(3)
        t.link(0, 1, 3)
        t.link(1, 2, 7)
        assert t.path_weight(0, 2) == 10
        assert t.path_weight(0, 1) == 3
        assert t.path_weight(1, 2) == 7

    def test_cut_and_relink_weighted(self):
        t = WeightedLinkCutTree(4)
        t.link(0, 1, 5)
        t.link(1, 2, 10)
        t.link(2, 3, 15)
        assert t.path_weight(0, 3) == 30
        t.cut(1, 2)
        t.link(1, 3, 20)
        # Now path 0->1->3: 5 + 20 = 25
        assert t.path_weight(0, 3) == 25

    def test_weighted_star(self):
        t = WeightedLinkCutTree(5)
        for i in range(1, 5):
            t.link(i, 0, i * 10)
        assert t.path_weight(1, 2) == 30  # 10 + 20
        assert t.path_weight(3, 4) == 70  # 30 + 40


# ============================================================
# Forest component management
# ============================================================

class TestForestComponents:
    def test_single_component(self):
        f = LinkCutForest(5)
        for i in range(4):
            f.link(i, i + 1)
        assert f.num_components == 1
        comps = f.get_components()
        assert len(comps) == 1
        assert comps[0] == {0, 1, 2, 3, 4}

    def test_multiple_components(self):
        f = LinkCutForest(6)
        f.link(0, 1)
        f.link(2, 3)
        f.link(4, 5)
        assert f.num_components == 3
        comps = f.get_components()
        assert len(comps) == 3

    def test_merge_components(self):
        f = LinkCutForest(4)
        f.link(0, 1)
        f.link(2, 3)
        assert f.num_components == 2
        f.link(1, 2)
        assert f.num_components == 1

    def test_split_components(self):
        f = LinkCutForest(4)
        f.link(0, 1)
        f.link(1, 2)
        f.link(2, 3)
        assert f.num_components == 1
        f.cut(1, 2)
        assert f.num_components == 2


# ============================================================
# Performance sanity (shouldn't timeout)
# ============================================================

class TestPerformance:
    def test_chain_1000_nodes(self):
        """Chain of 1000 nodes, various queries."""
        n = 1000
        t = LinkCutTree(n)
        for i in range(n - 1):
            t.link(i, i + 1)
        assert t.connected(0, n - 1)
        assert t.path_length(0, n - 1) == n - 1
        t.evert(n // 2)
        assert t.find_root(0) == n // 2

    def test_star_500_nodes(self):
        """Star with 500 nodes."""
        n = 500
        t = LinkCutTree(n)
        for i in range(1, n):
            t.link(i, 0)
        for i in range(1, n, 50):
            assert t.connected(i, 0)
            assert t.path_length(i, 0) == 1

    def test_random_forest_operations(self):
        """500 random operations on 100 nodes."""
        random.seed(99)
        n = 100
        t = LinkCutTree(n)
        edges = set()
        for _ in range(500):
            op = random.choice(['link', 'cut', 'find_root', 'evert'])
            if op == 'link' and len(edges) < n - 1:
                u, v = random.sample(range(n), 2)
                if not t.connected(u, v):
                    t.link(u, v)
                    edges.add((min(u, v), max(u, v)))
            elif op == 'cut' and edges:
                u, v = random.choice(list(edges))
                t.cut(u, v)
                edges.discard((u, v))
            elif op == 'find_root':
                t.find_root(random.randint(0, n - 1))
            elif op == 'evert':
                t.evert(random.randint(0, n - 1))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
