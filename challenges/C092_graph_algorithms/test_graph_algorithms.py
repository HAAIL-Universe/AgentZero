"""Tests for C092: Graph Algorithms."""

import pytest
import math
from graph_algorithms import (
    Graph, UnionFind,
    dijkstra, reconstruct_path, bellman_ford, floyd_warshall, floyd_warshall_path, astar,
    kruskal, prim,
    edmonds_karp, min_cut,
    topological_sort, tarjan_scc,
    has_cycle, find_cycle,
    is_bipartite, connected_components,
    bfs, dfs,
)


# ===================================================================
# Graph data structure
# ===================================================================

class TestGraph:
    def test_empty_graph(self):
        g = Graph()
        assert g.node_count() == 0
        assert g.edge_count() == 0

    def test_add_node(self):
        g = Graph()
        g.add_node("A")
        assert g.has_node("A")
        assert g.node_count() == 1

    def test_add_edge_undirected(self):
        g = Graph()
        g.add_edge("A", "B", 5)
        assert g.has_edge("A", "B")
        assert g.has_edge("B", "A")
        assert g.edge_count() == 1

    def test_add_edge_directed(self):
        g = Graph(directed=True)
        g.add_edge("A", "B", 5)
        assert g.has_edge("A", "B")
        assert not g.has_edge("B", "A")

    def test_neighbors(self):
        g = Graph()
        g.add_edge(1, 2, 3)
        g.add_edge(1, 3, 4)
        nbrs = g.neighbors(1)
        assert len(nbrs) == 2
        assert (2, 3) in nbrs
        assert (3, 4) in nbrs

    def test_edges_undirected(self):
        g = Graph()
        g.add_edge(1, 2, 1)
        g.add_edge(2, 3, 2)
        edges = g.edges()
        assert len(edges) == 2

    def test_edges_directed(self):
        g = Graph(directed=True)
        g.add_edge(1, 2, 1)
        g.add_edge(2, 1, 2)
        edges = g.edges()
        assert len(edges) == 2

    def test_degree_undirected(self):
        g = Graph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        assert g.degree(1) == 2
        assert g.degree(2) == 1

    def test_out_degree(self):
        g = Graph(directed=True)
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        assert g.out_degree(1) == 2
        assert g.out_degree(2) == 0

    def test_in_degree(self):
        g = Graph(directed=True)
        g.add_edge(1, 2)
        g.add_edge(3, 2)
        assert g.in_degree(2) == 2
        assert g.in_degree(1) == 0

    def test_reverse(self):
        g = Graph(directed=True)
        g.add_edge(1, 2, 5)
        r = g.reverse()
        assert r.has_edge(2, 1)
        assert not r.has_edge(1, 2)

    def test_reverse_undirected_raises(self):
        g = Graph()
        with pytest.raises(ValueError):
            g.reverse()

    def test_subgraph(self):
        g = Graph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 4)
        sg = g.subgraph({1, 2, 3})
        assert sg.has_node(1)
        assert sg.has_node(3)
        assert not sg.has_node(4)

    def test_default_weight(self):
        g = Graph()
        g.add_edge("A", "B")
        assert g.neighbors("A")[0][1] == 1.0

    def test_isolated_node(self):
        g = Graph()
        g.add_node("X")
        assert g.has_node("X")
        assert g.neighbors("X") == []

    def test_self_loop(self):
        g = Graph(directed=True)
        g.add_edge(1, 1, 3)
        assert g.has_edge(1, 1)

    def test_multiple_edges(self):
        g = Graph(directed=True)
        g.add_edge(1, 2, 3)
        g.add_edge(1, 2, 5)
        assert len(g.neighbors(1)) == 2


# ===================================================================
# Dijkstra
# ===================================================================

class TestDijkstra:
    def _simple_graph(self):
        g = Graph()
        g.add_edge("A", "B", 1)
        g.add_edge("B", "C", 2)
        g.add_edge("A", "C", 4)
        return g

    def test_simple(self):
        g = self._simple_graph()
        dist, pred = dijkstra(g, "A")
        assert dist["A"] == 0
        assert dist["B"] == 1
        assert dist["C"] == 3  # A->B->C

    def test_path_reconstruction(self):
        g = self._simple_graph()
        dist, pred = dijkstra(g, "A")
        path = reconstruct_path(pred, "C")
        assert path == ["A", "B", "C"]

    def test_unreachable(self):
        g = Graph(directed=True)
        g.add_edge(1, 2)
        g.add_node(3)
        dist, pred = dijkstra(g, 1)
        assert 3 not in dist

    def test_single_node(self):
        g = Graph()
        g.add_node(1)
        dist, pred = dijkstra(g, 1)
        assert dist[1] == 0

    def test_target_early_stop(self):
        g = Graph()
        g.add_edge(1, 2, 1)
        g.add_edge(2, 3, 1)
        g.add_edge(3, 4, 1)
        dist, pred = dijkstra(g, 1, target=2)
        assert dist[2] == 1

    def test_negative_weight_raises(self):
        g = Graph()
        g.add_edge(1, 2, -1)
        with pytest.raises(ValueError, match="negative"):
            dijkstra(g, 1)

    def test_directed_shortest_path(self):
        g = Graph(directed=True)
        g.add_edge("A", "B", 1)
        g.add_edge("A", "C", 10)
        g.add_edge("B", "C", 1)
        dist, _ = dijkstra(g, "A")
        assert dist["C"] == 2

    def test_large_graph(self):
        g = Graph()
        for i in range(100):
            g.add_edge(i, i + 1, 1)
        dist, _ = dijkstra(g, 0)
        assert dist[100] == 100

    def test_reconstruct_unreachable(self):
        assert reconstruct_path({}, "X") is None


# ===================================================================
# Bellman-Ford
# ===================================================================

class TestBellmanFord:
    def test_positive_weights(self):
        g = Graph(directed=True)
        g.add_edge("A", "B", 1)
        g.add_edge("B", "C", 2)
        g.add_edge("A", "C", 5)
        dist, pred = bellman_ford(g, "A")
        assert dist["C"] == 3

    def test_negative_weights(self):
        g = Graph(directed=True)
        g.add_edge("A", "B", 1)
        g.add_edge("B", "C", -3)
        g.add_edge("A", "C", 5)
        dist, _ = bellman_ford(g, "A")
        assert dist["C"] == -2

    def test_negative_cycle(self):
        g = Graph(directed=True)
        g.add_edge("A", "B", 1)
        g.add_edge("B", "C", -3)
        g.add_edge("C", "A", 1)
        with pytest.raises(ValueError, match="negative"):
            bellman_ford(g, "A")

    def test_single_node(self):
        g = Graph(directed=True)
        g.add_node(1)
        dist, _ = bellman_ford(g, 1)
        assert dist[1] == 0

    def test_matches_dijkstra(self):
        g = Graph(directed=True)
        g.add_edge(1, 2, 3)
        g.add_edge(1, 3, 7)
        g.add_edge(2, 3, 2)
        g.add_edge(2, 4, 5)
        g.add_edge(3, 4, 1)
        d_dij, _ = dijkstra(g, 1)
        d_bf, _ = bellman_ford(g, 1)
        for node in [1, 2, 3, 4]:
            assert abs(d_dij[node] - d_bf[node]) < 1e-9


# ===================================================================
# Floyd-Warshall
# ===================================================================

class TestFloydWarshall:
    def test_simple(self):
        g = Graph(directed=True)
        g.add_edge(1, 2, 3)
        g.add_edge(2, 3, 1)
        g.add_edge(1, 3, 10)
        dist, nxt = floyd_warshall(g)
        assert dist[1][3] == 4
        assert dist[1][2] == 3

    def test_path_reconstruction(self):
        g = Graph(directed=True)
        g.add_edge(1, 2, 1)
        g.add_edge(2, 3, 1)
        g.add_edge(1, 3, 10)
        _, nxt = floyd_warshall(g)
        path = floyd_warshall_path(nxt, 1, 3)
        assert path == [1, 2, 3]

    def test_no_path(self):
        g = Graph(directed=True)
        g.add_edge(1, 2, 1)
        g.add_node(3)
        dist, nxt = floyd_warshall(g)
        assert dist[1][3] == float('inf')
        assert floyd_warshall_path(nxt, 1, 3) is None

    def test_self_distance(self):
        g = Graph(directed=True)
        g.add_edge(1, 2, 5)
        dist, _ = floyd_warshall(g)
        assert dist[1][1] == 0
        assert dist[2][2] == 0

    def test_negative_weights(self):
        g = Graph(directed=True)
        g.add_edge(1, 2, -2)
        g.add_edge(2, 3, 3)
        dist, _ = floyd_warshall(g)
        assert dist[1][3] == 1


# ===================================================================
# A*
# ===================================================================

class TestAStar:
    def test_grid_manhattan(self):
        g = Graph()
        # 3x3 grid
        for r in range(3):
            for c in range(3):
                if c + 1 < 3:
                    g.add_edge((r, c), (r, c + 1), 1)
                if r + 1 < 3:
                    g.add_edge((r, c), (r + 1, c), 1)

        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        dist, path = astar(g, (0, 0), (2, 2), manhattan)
        assert dist == 4
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)

    def test_unreachable(self):
        g = Graph()
        g.add_node((0, 0))
        g.add_node((1, 1))
        dist, path = astar(g, (0, 0), (1, 1), lambda a, b: 0)
        assert dist == float('inf')
        assert path is None

    def test_same_as_dijkstra(self):
        g = Graph()
        g.add_edge("A", "B", 2)
        g.add_edge("B", "C", 3)
        g.add_edge("A", "C", 6)
        dist_a, _ = astar(g, "A", "C", lambda a, b: 0)
        dist_d, _ = dijkstra(g, "A", "C")
        assert dist_a == dist_d["C"]

    def test_direct_path(self):
        g = Graph()
        g.add_edge(0, 1, 5)
        dist, path = astar(g, 0, 1, lambda a, b: 0)
        assert dist == 5
        assert path == [0, 1]


# ===================================================================
# Union-Find
# ===================================================================

class TestUnionFind:
    def test_basic(self):
        uf = UnionFind()
        uf.make_set(1)
        uf.make_set(2)
        assert uf.find(1) != uf.find(2)
        uf.union(1, 2)
        assert uf.find(1) == uf.find(2)

    def test_multiple_unions(self):
        uf = UnionFind()
        for i in range(5):
            uf.make_set(i)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(0, 3)
        assert uf.find(0) == uf.find(3)
        assert uf.find(1) == uf.find(2)

    def test_union_same_set(self):
        uf = UnionFind()
        uf.make_set(1)
        uf.make_set(2)
        uf.union(1, 2)
        assert uf.union(1, 2) is False


# ===================================================================
# Kruskal MST
# ===================================================================

class TestKruskal:
    def test_simple_mst(self):
        g = Graph()
        g.add_edge("A", "B", 1)
        g.add_edge("B", "C", 2)
        g.add_edge("A", "C", 3)
        mst, total = kruskal(g)
        assert total == 3  # A-B(1) + B-C(2)
        assert len(mst) == 2

    def test_single_node(self):
        g = Graph()
        g.add_node(1)
        mst, total = kruskal(g)
        assert mst == []
        assert total == 0

    def test_disconnected(self):
        g = Graph()
        g.add_edge(1, 2, 1)
        g.add_edge(3, 4, 2)
        mst, total = kruskal(g)
        assert total == 3
        assert len(mst) == 2

    def test_larger_graph(self):
        g = Graph()
        g.add_edge(0, 1, 4)
        g.add_edge(0, 7, 8)
        g.add_edge(1, 2, 8)
        g.add_edge(1, 7, 11)
        g.add_edge(2, 3, 7)
        g.add_edge(2, 5, 4)
        g.add_edge(2, 8, 2)
        g.add_edge(3, 4, 9)
        g.add_edge(3, 5, 14)
        g.add_edge(4, 5, 10)
        g.add_edge(5, 6, 2)
        g.add_edge(6, 7, 1)
        g.add_edge(6, 8, 6)
        g.add_edge(7, 8, 7)
        mst, total = kruskal(g)
        assert total == 37
        assert len(mst) == 8


# ===================================================================
# Prim MST
# ===================================================================

class TestPrim:
    def test_simple_mst(self):
        g = Graph()
        g.add_edge("A", "B", 1)
        g.add_edge("B", "C", 2)
        g.add_edge("A", "C", 3)
        mst, total = prim(g)
        assert total == 3
        assert len(mst) == 2

    def test_empty(self):
        g = Graph()
        mst, total = prim(g)
        assert mst == []
        assert total == 0

    def test_matches_kruskal(self):
        g = Graph()
        g.add_edge(0, 1, 4)
        g.add_edge(0, 7, 8)
        g.add_edge(1, 2, 8)
        g.add_edge(1, 7, 11)
        g.add_edge(2, 3, 7)
        g.add_edge(2, 5, 4)
        g.add_edge(2, 8, 2)
        g.add_edge(3, 4, 9)
        g.add_edge(3, 5, 14)
        g.add_edge(4, 5, 10)
        g.add_edge(5, 6, 2)
        g.add_edge(6, 7, 1)
        g.add_edge(6, 8, 6)
        g.add_edge(7, 8, 7)
        _, total_k = kruskal(g)
        _, total_p = prim(g)
        assert total_k == total_p

    def test_with_start(self):
        g = Graph()
        g.add_edge(1, 2, 3)
        g.add_edge(2, 3, 1)
        mst, total = prim(g, start=1)
        assert total == 4


# ===================================================================
# Edmonds-Karp Max Flow
# ===================================================================

class TestEdmondsKarp:
    def test_simple_flow(self):
        g = Graph(directed=True)
        g.add_edge("S", "A", 10)
        g.add_edge("S", "B", 5)
        g.add_edge("A", "B", 15)
        g.add_edge("A", "T", 10)
        g.add_edge("B", "T", 10)
        flow_val, _ = edmonds_karp(g, "S", "T")
        assert flow_val == 15

    def test_bottleneck(self):
        g = Graph(directed=True)
        g.add_edge("S", "A", 100)
        g.add_edge("A", "T", 1)
        flow_val, _ = edmonds_karp(g, "S", "T")
        assert flow_val == 1

    def test_parallel_paths(self):
        g = Graph(directed=True)
        g.add_edge("S", "A", 5)
        g.add_edge("S", "B", 5)
        g.add_edge("A", "T", 5)
        g.add_edge("B", "T", 5)
        flow_val, _ = edmonds_karp(g, "S", "T")
        assert flow_val == 10

    def test_no_path(self):
        g = Graph(directed=True)
        g.add_edge("S", "A", 5)
        g.add_node("T")
        flow_val, _ = edmonds_karp(g, "S", "T")
        assert flow_val == 0

    def test_classic_example(self):
        # Classic max-flow example
        g = Graph(directed=True)
        g.add_edge(0, 1, 16)
        g.add_edge(0, 2, 13)
        g.add_edge(1, 2, 10)
        g.add_edge(1, 3, 12)
        g.add_edge(2, 1, 4)
        g.add_edge(2, 4, 14)
        g.add_edge(3, 2, 9)
        g.add_edge(3, 5, 20)
        g.add_edge(4, 3, 7)
        g.add_edge(4, 5, 4)
        flow_val, _ = edmonds_karp(g, 0, 5)
        assert flow_val == 23


# ===================================================================
# Min Cut
# ===================================================================

class TestMinCut:
    def test_simple_cut(self):
        g = Graph(directed=True)
        g.add_edge("S", "A", 3)
        g.add_edge("S", "B", 2)
        g.add_edge("A", "T", 2)
        g.add_edge("B", "T", 3)
        flow_val, s_set, t_set = min_cut(g, "S", "T")
        assert "S" in s_set
        assert "T" in t_set
        assert flow_val == 4

    def test_min_cut_equals_max_flow(self):
        g = Graph(directed=True)
        g.add_edge(0, 1, 16)
        g.add_edge(0, 2, 13)
        g.add_edge(1, 3, 12)
        g.add_edge(2, 1, 4)
        g.add_edge(2, 4, 14)
        g.add_edge(3, 2, 9)
        g.add_edge(3, 5, 20)
        g.add_edge(4, 3, 7)
        g.add_edge(4, 5, 4)
        flow_val, s_set, t_set = min_cut(g, 0, 5)
        max_flow, _ = edmonds_karp(g, 0, 5)
        assert flow_val == max_flow


# ===================================================================
# Topological Sort
# ===================================================================

class TestTopologicalSort:
    def test_simple_dag(self):
        g = Graph(directed=True)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("A", "C")
        order = topological_sort(g)
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")

    def test_cycle_raises(self):
        g = Graph(directed=True)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")
        with pytest.raises(ValueError, match="cycle"):
            topological_sort(g)

    def test_undirected_raises(self):
        g = Graph()
        g.add_edge(1, 2)
        with pytest.raises(ValueError, match="directed"):
            topological_sort(g)

    def test_disconnected(self):
        g = Graph(directed=True)
        g.add_edge(1, 2)
        g.add_edge(3, 4)
        order = topological_sort(g)
        assert order.index(1) < order.index(2)
        assert order.index(3) < order.index(4)

    def test_single_node(self):
        g = Graph(directed=True)
        g.add_node(1)
        assert topological_sort(g) == [1]

    def test_diamond_dag(self):
        g = Graph(directed=True)
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")
        order = topological_sort(g)
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")


# ===================================================================
# Tarjan SCC
# ===================================================================

class TestTarjanSCC:
    def test_simple_scc(self):
        g = Graph(directed=True)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 1)
        sccs = tarjan_scc(g)
        assert len(sccs) == 1
        assert set(sccs[0]) == {1, 2, 3}

    def test_dag_all_singletons(self):
        g = Graph(directed=True)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        sccs = tarjan_scc(g)
        assert len(sccs) == 3

    def test_two_sccs(self):
        g = Graph(directed=True)
        g.add_edge(1, 2)
        g.add_edge(2, 1)
        g.add_edge(3, 4)
        g.add_edge(4, 3)
        g.add_edge(2, 3)
        sccs = tarjan_scc(g)
        scc_sets = [set(s) for s in sccs]
        assert {1, 2} in scc_sets
        assert {3, 4} in scc_sets

    def test_undirected_raises(self):
        g = Graph()
        g.add_edge(1, 2)
        with pytest.raises(ValueError, match="directed"):
            tarjan_scc(g)

    def test_single_node(self):
        g = Graph(directed=True)
        g.add_node(1)
        sccs = tarjan_scc(g)
        assert len(sccs) == 1
        assert sccs[0] == [1]

    def test_complex_sccs(self):
        g = Graph(directed=True)
        # SCC1: {0,1,2}
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        # SCC2: {3,4}
        g.add_edge(3, 4)
        g.add_edge(4, 3)
        # Cross-SCC edges
        g.add_edge(2, 3)
        g.add_edge(4, 5)
        # Singleton: {5}
        g.add_node(5)
        sccs = tarjan_scc(g)
        scc_sets = [set(s) for s in sccs]
        assert {0, 1, 2} in scc_sets
        assert {3, 4} in scc_sets
        assert {5} in scc_sets


# ===================================================================
# Cycle Detection
# ===================================================================

class TestCycleDetection:
    def test_directed_cycle(self):
        g = Graph(directed=True)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 1)
        assert has_cycle(g) is True

    def test_directed_no_cycle(self):
        g = Graph(directed=True)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        assert has_cycle(g) is False

    def test_undirected_cycle(self):
        g = Graph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 1)
        assert has_cycle(g) is True

    def test_undirected_tree(self):
        g = Graph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(2, 4)
        assert has_cycle(g) is False

    def test_find_cycle_directed(self):
        g = Graph(directed=True)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 1)
        cycle = find_cycle(g)
        assert cycle is not None
        assert len(cycle) >= 2

    def test_find_cycle_none(self):
        g = Graph(directed=True)
        g.add_edge(1, 2)
        assert find_cycle(g) is None

    def test_self_loop(self):
        g = Graph(directed=True)
        g.add_edge(1, 1)
        assert has_cycle(g) is True

    def test_empty(self):
        g = Graph()
        assert has_cycle(g) is False


# ===================================================================
# Bipartite
# ===================================================================

class TestBipartite:
    def test_bipartite(self):
        g = Graph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 4)
        is_bip, partition = is_bipartite(g)
        assert is_bip is True
        a, b = partition
        assert (1 in a and 2 in b) or (1 in b and 2 in a)

    def test_not_bipartite(self):
        g = Graph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 1)  # odd cycle
        is_bip, partition = is_bipartite(g)
        assert is_bip is False
        assert partition is None

    def test_empty(self):
        g = Graph()
        is_bip, _ = is_bipartite(g)
        assert is_bip is True

    def test_disconnected_bipartite(self):
        g = Graph()
        g.add_edge(1, 2)
        g.add_edge(3, 4)
        is_bip, _ = is_bipartite(g)
        assert is_bip is True

    def test_complete_bipartite(self):
        g = Graph()
        for i in range(3):
            for j in range(3, 6):
                g.add_edge(i, j)
        is_bip, (a, b) = is_bipartite(g)
        assert is_bip is True
        assert len(a) == 3
        assert len(b) == 3


# ===================================================================
# Connected Components
# ===================================================================

class TestConnectedComponents:
    def test_single_component(self):
        g = Graph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        comps = connected_components(g)
        assert len(comps) == 1
        assert set(comps[0]) == {1, 2, 3}

    def test_multiple_components(self):
        g = Graph()
        g.add_edge(1, 2)
        g.add_edge(3, 4)
        g.add_node(5)
        comps = connected_components(g)
        assert len(comps) == 3

    def test_directed_weakly_connected(self):
        g = Graph(directed=True)
        g.add_edge(1, 2)
        g.add_edge(3, 2)
        comps = connected_components(g)
        assert len(comps) == 1

    def test_empty(self):
        g = Graph()
        assert connected_components(g) == []


# ===================================================================
# BFS / DFS
# ===================================================================

class TestBFS:
    def test_order(self):
        g = Graph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 4)
        order, dists = bfs(g, 1)
        assert order[0] == 1
        assert dists[4] == 2

    def test_distances(self):
        g = Graph()
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")
        _, dists = bfs(g, "A")
        assert dists["A"] == 0
        assert dists["D"] == 3

    def test_disconnected(self):
        g = Graph()
        g.add_edge(1, 2)
        g.add_node(3)
        order, dists = bfs(g, 1)
        assert 3 not in order


class TestDFS:
    def test_order(self):
        g = Graph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 4)
        order = dfs(g, 1)
        assert order[0] == 1
        assert set(order) == {1, 2, 3, 4}

    def test_single_node(self):
        g = Graph()
        g.add_node(1)
        assert dfs(g, 1) == [1]


# ===================================================================
# Integration / Edge cases
# ===================================================================

class TestIntegration:
    def test_dijkstra_on_grid(self):
        """5x5 grid graph, shortest path corner to corner."""
        g = Graph()
        for r in range(5):
            for c in range(5):
                if c + 1 < 5:
                    g.add_edge((r, c), (r, c + 1), 1)
                if r + 1 < 5:
                    g.add_edge((r, c), (r + 1, c), 1)
        dist, pred = dijkstra(g, (0, 0), (4, 4))
        assert dist[(4, 4)] == 8

    def test_flow_bipartite_matching(self):
        """Max bipartite matching via max flow."""
        g = Graph(directed=True)
        # Left: L1, L2, L3; Right: R1, R2, R3
        g.add_edge("S", "L1", 1)
        g.add_edge("S", "L2", 1)
        g.add_edge("S", "L3", 1)
        g.add_edge("L1", "R1", 1)
        g.add_edge("L1", "R2", 1)
        g.add_edge("L2", "R2", 1)
        g.add_edge("L3", "R3", 1)
        g.add_edge("R1", "T", 1)
        g.add_edge("R2", "T", 1)
        g.add_edge("R3", "T", 1)
        flow_val, _ = edmonds_karp(g, "S", "T")
        assert flow_val == 3  # Perfect matching

    def test_scc_condensation(self):
        """Condensed DAG from SCCs should be acyclic."""
        g = Graph(directed=True)
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 2)
        sccs = tarjan_scc(g)
        # Build condensation graph
        node_to_scc = {}
        for i, scc in enumerate(sccs):
            for n in scc:
                node_to_scc[n] = i
        cg = Graph(directed=True)
        for i in range(len(sccs)):
            cg.add_node(i)
        for u in g._adj:
            for v, _ in g._adj[u]:
                su, sv = node_to_scc[u], node_to_scc[v]
                if su != sv:
                    cg.add_edge(su, sv)
        assert has_cycle(cg) is False

    def test_negative_edges_bellman_ford_path(self):
        g = Graph(directed=True)
        g.add_edge("A", "B", 4)
        g.add_edge("A", "C", 2)
        g.add_edge("C", "B", -5)
        dist, pred = bellman_ford(g, "A")
        assert dist["B"] == -3  # A->C->B
        path = reconstruct_path(pred, "B")
        assert path == ["A", "C", "B"]

    def test_mst_spanning_property(self):
        """MST should connect all nodes."""
        g = Graph()
        for i in range(10):
            g.add_edge(i, (i + 1) % 10, i + 1)
        mst, _ = kruskal(g)
        # MST should have n-1 edges
        assert len(mst) == 9

    def test_topo_sort_respects_all_edges(self):
        g = Graph(directed=True)
        edges = [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1)]
        for u, v in edges:
            g.add_edge(u, v)
        order = topological_sort(g)
        for u, v in edges:
            assert order.index(u) < order.index(v)

    def test_flow_conservation(self):
        """Flow conservation: flow in = flow out at non-source/sink."""
        g = Graph(directed=True)
        g.add_edge("S", "A", 10)
        g.add_edge("S", "B", 8)
        g.add_edge("A", "B", 5)
        g.add_edge("A", "C", 7)
        g.add_edge("B", "C", 6)
        g.add_edge("C", "T", 15)
        flow_val, flow_dict = edmonds_karp(g, "S", "T")
        # Check conservation at A and B
        for node in ["A", "B", "C"]:
            flow_in = sum(flow_dict.get(u, {}).get(node, 0) for u in ["S", "A", "B", "C", "T"] if flow_dict.get(u, {}).get(node, 0) > 0)
            flow_out = sum(flow_dict.get(node, {}).get(v, 0) for v in ["S", "A", "B", "C", "T"] if flow_dict.get(node, {}).get(v, 0) > 0)
            assert abs(flow_in - flow_out) < 1e-9

    def test_strongly_connected_reverse(self):
        """SCC of reversed graph should have same components."""
        g = Graph(directed=True)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        sccs_orig = [set(s) for s in tarjan_scc(g)]
        sccs_rev = [set(s) for s in tarjan_scc(g.reverse())]
        assert sorted(sccs_orig, key=lambda s: min(s)) == sorted(sccs_rev, key=lambda s: min(s))

    def test_astar_finds_optimal(self):
        """A* with admissible heuristic finds optimal path."""
        g = Graph()
        g.add_edge("A", "B", 1)
        g.add_edge("B", "D", 3)
        g.add_edge("A", "C", 2)
        g.add_edge("C", "D", 1)
        dist, path = astar(g, "A", "D", lambda a, b: 0)
        assert dist == 3
        assert path == ["A", "C", "D"]

    def test_floyd_warshall_all_pairs(self):
        """All-pairs matches individual Dijkstra runs."""
        g = Graph(directed=True)
        g.add_edge(0, 1, 2)
        g.add_edge(1, 2, 3)
        g.add_edge(0, 2, 7)
        g.add_edge(2, 0, 1)
        fw_dist, _ = floyd_warshall(g)
        for src in [0, 1, 2]:
            d_dist, _ = dijkstra(g, src)
            for tgt in [0, 1, 2]:
                if tgt in d_dist:
                    assert abs(fw_dist[src][tgt] - d_dist[tgt]) < 1e-9

    def test_kruskal_prim_same_weight_different_graph(self):
        """Both MST algorithms produce same total weight on random-ish graph."""
        g = Graph()
        edges = [(0,1,7),(0,3,5),(1,2,8),(1,3,9),(1,4,7),(2,4,5),
                 (3,4,15),(3,5,6),(4,5,8),(4,6,9),(5,6,11)]
        for u, v, w in edges:
            g.add_edge(u, v, w)
        _, tk = kruskal(g)
        _, tp = prim(g)
        assert tk == tp

    def test_dijkstra_equal_weight_paths(self):
        """Multiple shortest paths of equal length."""
        g = Graph()
        g.add_edge(0, 1, 1)
        g.add_edge(0, 2, 1)
        g.add_edge(1, 3, 1)
        g.add_edge(2, 3, 1)
        dist, _ = dijkstra(g, 0, 3)
        assert dist[3] == 2

    def test_bellman_ford_long_chain(self):
        g = Graph(directed=True)
        for i in range(50):
            g.add_edge(i, i + 1, 1)
        dist, _ = bellman_ford(g, 0)
        assert dist[50] == 50

    def test_flow_single_edge(self):
        g = Graph(directed=True)
        g.add_edge("S", "T", 42)
        flow_val, _ = edmonds_karp(g, "S", "T")
        assert flow_val == 42

    def test_cycle_detection_large_dag(self):
        """Large DAG should have no cycle."""
        g = Graph(directed=True)
        for i in range(100):
            for j in range(i + 1, min(i + 3, 100)):
                g.add_edge(i, j)
        assert has_cycle(g) is False

    def test_bipartite_star(self):
        """Star graph is bipartite."""
        g = Graph()
        for i in range(1, 6):
            g.add_edge(0, i)
        is_bip, (a, b) = is_bipartite(g)
        assert is_bip is True

    def test_connected_components_chain(self):
        g = Graph()
        for i in range(9):
            g.add_edge(i, i + 1)
        comps = connected_components(g)
        assert len(comps) == 1
        assert len(comps[0]) == 10

    def test_bfs_directed(self):
        g = Graph(directed=True)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 1)
        order, dists = bfs(g, 1)
        assert set(order) == {1, 2, 3}
        assert dists[3] == 2

    def test_dfs_directed(self):
        g = Graph(directed=True)
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 4)
        order = dfs(g, 1)
        assert order[0] == 1
        # DFS should visit 2 before backtracking to 3 (sorted neighbors)
        assert order.index(2) < order.index(3)

    def test_min_cut_partition_valid(self):
        """S and T must be in different partitions."""
        g = Graph(directed=True)
        g.add_edge(0, 1, 10)
        g.add_edge(0, 2, 10)
        g.add_edge(1, 3, 4)
        g.add_edge(2, 3, 8)
        g.add_edge(1, 2, 2)
        _, s_set, t_set = min_cut(g, 0, 3)
        assert 0 in s_set
        assert 3 in t_set
        assert s_set & t_set == set()
        assert s_set | t_set == {0, 1, 2, 3}

    def test_topo_sort_linear(self):
        g = Graph(directed=True)
        for i in range(10):
            g.add_edge(i, i + 1)
        order = topological_sort(g)
        assert order == list(range(11))

    def test_scc_fully_connected(self):
        """Fully connected directed graph = 1 SCC."""
        g = Graph(directed=True)
        for i in range(5):
            for j in range(5):
                if i != j:
                    g.add_edge(i, j)
        sccs = tarjan_scc(g)
        assert len(sccs) == 1
        assert set(sccs[0]) == {0, 1, 2, 3, 4}

    def test_graph_string_nodes(self):
        """Graph works with string node names."""
        g = Graph()
        g.add_edge("alice", "bob", 2)
        g.add_edge("bob", "charlie", 3)
        dist, _ = dijkstra(g, "alice")
        assert dist["charlie"] == 5

    def test_dijkstra_weighted_directed(self):
        """Weighted directed graph with asymmetric edges."""
        g = Graph(directed=True)
        g.add_edge("A", "B", 1)
        g.add_edge("B", "A", 10)
        dist_a, _ = dijkstra(g, "A")
        dist_b, _ = dijkstra(g, "B")
        assert dist_a["B"] == 1
        assert dist_b["A"] == 10

    def test_flow_diamond(self):
        g = Graph(directed=True)
        g.add_edge("S", "A", 10)
        g.add_edge("S", "B", 10)
        g.add_edge("A", "T", 10)
        g.add_edge("B", "T", 10)
        g.add_edge("A", "B", 1)
        flow_val, _ = edmonds_karp(g, "S", "T")
        assert flow_val == 20

    def test_find_cycle_undirected(self):
        g = Graph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(3, 1)
        cycle = find_cycle(g)
        assert cycle is not None

    def test_find_cycle_undirected_none(self):
        g = Graph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        assert find_cycle(g) is None

    def test_union_find_path_compression(self):
        uf = UnionFind()
        for i in range(10):
            uf.make_set(i)
        # Chain: 0-1-2-...-9
        for i in range(9):
            uf.union(i, i + 1)
        # All should have same root
        root = uf.find(0)
        for i in range(10):
            assert uf.find(i) == root

    def test_bellman_ford_unreachable(self):
        g = Graph(directed=True)
        g.add_edge(1, 2, 5)
        g.add_node(3)
        dist, _ = bellman_ford(g, 1)
        assert dist[3] == float('inf')

    def test_graph_node_count_with_isolated(self):
        g = Graph()
        g.add_node("X")
        g.add_node("Y")
        g.add_edge("A", "B")
        assert g.node_count() == 4
        assert g.edge_count() == 1
