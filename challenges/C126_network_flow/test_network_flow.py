"""Tests for C126: Network Flow Algorithms."""

import pytest
import math
from network_flow import (
    FlowNetwork, Edge, EdmondsKarp, Dinic, PushRelabel,
    MinCostFlow, HopcroftKarp,
    edge_disjoint_paths, node_disjoint_paths,
    circulation_with_demands, assignment_problem, project_selection,
)


# ═══════════════════════════════════════════════════════════════════
# FlowNetwork basics
# ═══════════════════════════════════════════════════════════════════

class TestFlowNetwork:

    def test_add_edge(self):
        net = FlowNetwork()
        e = net.add_edge('a', 'b', 10)
        assert e.src == 'a'
        assert e.dst == 'b'
        assert e.cap == 10
        assert e.flow == 0
        assert e.residual == 10
        assert e.rev.src == 'b'
        assert e.rev.dst == 'a'
        assert e.rev.cap == 0

    def test_nodes_tracked(self):
        net = FlowNetwork()
        net.add_edge('a', 'b', 5)
        net.add_edge('b', 'c', 3)
        assert net.nodes == {'a', 'b', 'c'}

    def test_get_edges(self):
        net = FlowNetwork()
        net.add_edge('a', 'b', 5)
        net.add_edge('b', 'c', 3)
        edges = net.get_edges()
        assert len(edges) == 2
        caps = {(e.src, e.dst): e.cap for e in edges}
        assert caps == {('a', 'b'): 5, ('b', 'c'): 3}

    def test_reset_flow(self):
        net = FlowNetwork()
        e = net.add_edge('a', 'b', 10)
        e.flow = 5
        e.rev.flow = -5
        net.reset_flow()
        assert e.flow == 0
        assert e.rev.flow == 0

    def test_add_undirected_edge(self):
        net = FlowNetwork()
        e = net.add_undirected_edge('a', 'b', 7)
        assert e.cap == 7
        assert e.rev.cap == 7  # Both directions have capacity

    def test_edge_repr(self):
        e = Edge('x', 'y', 10)
        e.flow = 3
        assert 'x' in repr(e)
        assert 'y' in repr(e)

    def test_copy(self):
        net = FlowNetwork()
        net.add_edge('a', 'b', 10)
        net.add_edge('b', 'c', 5)
        copy = net.copy()
        assert len(copy.get_edges()) == 2
        # Modify original, copy unaffected
        net.get_edges()[0].flow = 5
        assert all(e.flow == 0 for e in copy.get_edges())


# ═══════════════════════════════════════════════════════════════════
# Edmonds-Karp
# ═══════════════════════════════════════════════════════════════════

class TestEdmondsKarp:

    def _simple_network(self):
        """s->a(10), s->b(10), a->b(2), a->t(6), b->t(10)"""
        net = FlowNetwork()
        net.add_edge('s', 'a', 10)
        net.add_edge('s', 'b', 10)
        net.add_edge('a', 'b', 2)
        net.add_edge('a', 't', 6)
        net.add_edge('b', 't', 10)
        return net

    def test_simple_max_flow(self):
        net = self._simple_network()
        solver = EdmondsKarp(net)
        assert solver.max_flow('s', 't') == 16

    def test_single_edge(self):
        net = FlowNetwork()
        net.add_edge('s', 't', 5)
        assert EdmondsKarp(net).max_flow('s', 't') == 5

    def test_two_paths(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 3)
        net.add_edge('s', 'b', 4)
        net.add_edge('a', 't', 3)
        net.add_edge('b', 't', 4)
        assert EdmondsKarp(net).max_flow('s', 't') == 7

    def test_bottleneck(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 100)
        net.add_edge('a', 'b', 1)
        net.add_edge('b', 't', 100)
        assert EdmondsKarp(net).max_flow('s', 't') == 1

    def test_no_path(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 10)
        net.add_edge('b', 't', 10)
        assert EdmondsKarp(net).max_flow('s', 't') == 0

    def test_parallel_edges(self):
        net = FlowNetwork()
        net.add_edge('s', 't', 3)
        net.add_edge('s', 't', 4)
        assert EdmondsKarp(net).max_flow('s', 't') == 7

    def test_diamond(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 3)
        net.add_edge('s', 'b', 2)
        net.add_edge('a', 't', 2)
        net.add_edge('b', 't', 3)
        net.add_edge('a', 'b', 1)
        assert EdmondsKarp(net).max_flow('s', 't') == 5

    def test_chain(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 5)
        net.add_edge('a', 'b', 3)
        net.add_edge('b', 'c', 7)
        net.add_edge('c', 't', 4)
        assert EdmondsKarp(net).max_flow('s', 't') == 3

    def test_complex_graph(self):
        """Classic max-flow example with crossing paths."""
        net = FlowNetwork()
        net.add_edge('s', 'a', 10)
        net.add_edge('s', 'c', 10)
        net.add_edge('a', 'b', 4)
        net.add_edge('a', 'c', 2)
        net.add_edge('a', 'd', 8)
        net.add_edge('b', 't', 10)
        net.add_edge('c', 'd', 9)
        net.add_edge('d', 'b', 6)
        net.add_edge('d', 't', 10)
        assert EdmondsKarp(net).max_flow('s', 't') == 19

    def test_min_cut_simple(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 3)
        net.add_edge('s', 'b', 2)
        net.add_edge('a', 't', 2)
        net.add_edge('b', 't', 3)
        solver = EdmondsKarp(net)
        flow = solver.max_flow('s', 't')
        S, T, cut_edges, cut_value = solver.min_cut('s', 't')
        assert 's' in S
        assert 't' in T
        assert cut_value == flow

    def test_min_cut_bottleneck(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 100)
        net.add_edge('a', 'b', 1)
        net.add_edge('b', 't', 100)
        solver = EdmondsKarp(net)
        solver.max_flow('s', 't')
        S, T, cut_edges, cut_value = solver.min_cut('s', 't')
        assert cut_value == 1

    def test_zero_capacity(self):
        net = FlowNetwork()
        net.add_edge('s', 't', 0)
        assert EdmondsKarp(net).max_flow('s', 't') == 0


# ═══════════════════════════════════════════════════════════════════
# Dinic's Algorithm
# ═══════════════════════════════════════════════════════════════════

class TestDinic:

    def test_simple(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 10)
        net.add_edge('s', 'b', 10)
        net.add_edge('a', 'b', 2)
        net.add_edge('a', 't', 6)
        net.add_edge('b', 't', 10)
        assert Dinic(net).max_flow('s', 't') == 16

    def test_single_edge(self):
        net = FlowNetwork()
        net.add_edge('s', 't', 5)
        assert Dinic(net).max_flow('s', 't') == 5

    def test_two_paths(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 3)
        net.add_edge('s', 'b', 4)
        net.add_edge('a', 't', 3)
        net.add_edge('b', 't', 4)
        assert Dinic(net).max_flow('s', 't') == 7

    def test_bottleneck(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 100)
        net.add_edge('a', 'b', 1)
        net.add_edge('b', 't', 100)
        assert Dinic(net).max_flow('s', 't') == 1

    def test_no_path(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 10)
        net.add_edge('b', 't', 10)
        assert Dinic(net).max_flow('s', 't') == 0

    def test_diamond(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 3)
        net.add_edge('s', 'b', 2)
        net.add_edge('a', 't', 2)
        net.add_edge('b', 't', 3)
        net.add_edge('a', 'b', 1)
        assert Dinic(net).max_flow('s', 't') == 5

    def test_complex_graph(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 10)
        net.add_edge('s', 'c', 10)
        net.add_edge('a', 'b', 4)
        net.add_edge('a', 'c', 2)
        net.add_edge('a', 'd', 8)
        net.add_edge('b', 't', 10)
        net.add_edge('c', 'd', 9)
        net.add_edge('d', 'b', 6)
        net.add_edge('d', 't', 10)
        assert Dinic(net).max_flow('s', 't') == 19

    def test_min_cut(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 3)
        net.add_edge('s', 'b', 2)
        net.add_edge('a', 't', 2)
        net.add_edge('b', 't', 3)
        solver = Dinic(net)
        flow = solver.max_flow('s', 't')
        S, T, cut_edges, cut_value = solver.min_cut('s', 't')
        assert cut_value == flow

    def test_parallel_edges(self):
        net = FlowNetwork()
        net.add_edge('s', 't', 3)
        net.add_edge('s', 't', 4)
        assert Dinic(net).max_flow('s', 't') == 7

    def test_chain(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 5)
        net.add_edge('a', 'b', 3)
        net.add_edge('b', 'c', 7)
        net.add_edge('c', 't', 4)
        assert Dinic(net).max_flow('s', 't') == 3

    def test_large_capacity(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 1000000)
        net.add_edge('a', 't', 1000000)
        assert Dinic(net).max_flow('s', 't') == 1000000


# ═══════════════════════════════════════════════════════════════════
# Push-Relabel
# ═══════════════════════════════════════════════════════════════════

class TestPushRelabel:

    def test_simple(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 10)
        net.add_edge('s', 'b', 10)
        net.add_edge('a', 'b', 2)
        net.add_edge('a', 't', 6)
        net.add_edge('b', 't', 10)
        assert PushRelabel(net).max_flow('s', 't') == 16

    def test_single_edge(self):
        net = FlowNetwork()
        net.add_edge('s', 't', 5)
        assert PushRelabel(net).max_flow('s', 't') == 5

    def test_two_paths(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 3)
        net.add_edge('s', 'b', 4)
        net.add_edge('a', 't', 3)
        net.add_edge('b', 't', 4)
        assert PushRelabel(net).max_flow('s', 't') == 7

    def test_bottleneck(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 100)
        net.add_edge('a', 'b', 1)
        net.add_edge('b', 't', 100)
        assert PushRelabel(net).max_flow('s', 't') == 1

    def test_no_path(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 10)
        net.add_edge('b', 't', 10)
        assert PushRelabel(net).max_flow('s', 't') == 0

    def test_diamond(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 3)
        net.add_edge('s', 'b', 2)
        net.add_edge('a', 't', 2)
        net.add_edge('b', 't', 3)
        net.add_edge('a', 'b', 1)
        assert PushRelabel(net).max_flow('s', 't') == 5

    def test_complex_graph(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 10)
        net.add_edge('s', 'c', 10)
        net.add_edge('a', 'b', 4)
        net.add_edge('a', 'c', 2)
        net.add_edge('a', 'd', 8)
        net.add_edge('b', 't', 10)
        net.add_edge('c', 'd', 9)
        net.add_edge('d', 'b', 6)
        net.add_edge('d', 't', 10)
        assert PushRelabel(net).max_flow('s', 't') == 19

    def test_min_cut(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 3)
        net.add_edge('s', 'b', 2)
        net.add_edge('a', 't', 2)
        net.add_edge('b', 't', 3)
        solver = PushRelabel(net)
        flow = solver.max_flow('s', 't')
        S, T, cut_edges, cut_value = solver.min_cut('s', 't')
        assert cut_value == flow

    def test_chain(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 5)
        net.add_edge('a', 'b', 3)
        net.add_edge('b', 'c', 7)
        net.add_edge('c', 't', 4)
        assert PushRelabel(net).max_flow('s', 't') == 3


# ═══════════════════════════════════════════════════════════════════
# All three algorithms agree
# ═══════════════════════════════════════════════════════════════════

class TestAlgorithmsAgree:

    def _make_network(self):
        """Complex network to test all algorithms give same answer."""
        edges = [
            ('s', 'a', 16), ('s', 'c', 13),
            ('a', 'b', 12), ('a', 'c', 4),
            ('b', 't', 20), ('c', 'a', 10),
            ('c', 'd', 14), ('d', 'b', 7),
            ('d', 't', 4),
        ]
        return edges

    def test_all_agree_complex(self):
        edges = self._make_network()
        results = []
        for Solver in [EdmondsKarp, Dinic, PushRelabel]:
            net = FlowNetwork()
            for s, d, c in edges:
                net.add_edge(s, d, c)
            results.append(Solver(net).max_flow('s', 't'))
        assert results[0] == results[1] == results[2] == 23

    def test_all_agree_grid(self):
        """3x3 grid network."""
        results = []
        for Solver in [EdmondsKarp, Dinic, PushRelabel]:
            net = FlowNetwork()
            # 3x3 grid: (0,0) to (2,2)
            for r in range(3):
                for c in range(3):
                    if c + 1 < 3:
                        net.add_edge(f"{r},{c}", f"{r},{c+1}", r + c + 1)
                    if r + 1 < 3:
                        net.add_edge(f"{r},{c}", f"{r+1},{c}", r + c + 2)
            results.append(Solver(net).max_flow("0,0", "2,2"))
        assert results[0] == results[1] == results[2]

    def test_all_agree_bipartite(self):
        """Bipartite graph as flow network."""
        results = []
        for Solver in [EdmondsKarp, Dinic, PushRelabel]:
            net = FlowNetwork()
            net.add_edge('s', 'L1', 1)
            net.add_edge('s', 'L2', 1)
            net.add_edge('s', 'L3', 1)
            net.add_edge('L1', 'R1', 1)
            net.add_edge('L1', 'R2', 1)
            net.add_edge('L2', 'R2', 1)
            net.add_edge('L2', 'R3', 1)
            net.add_edge('L3', 'R3', 1)
            net.add_edge('R1', 't', 1)
            net.add_edge('R2', 't', 1)
            net.add_edge('R3', 't', 1)
            results.append(Solver(net).max_flow('s', 't'))
        assert results[0] == results[1] == results[2] == 3


# ═══════════════════════════════════════════════════════════════════
# Min-Cost Max-Flow
# ═══════════════════════════════════════════════════════════════════

class TestMinCostFlow:

    def test_simple(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 2, cost=1)
        net.add_edge('s', 'b', 2, cost=3)
        net.add_edge('a', 't', 2, cost=2)
        net.add_edge('b', 't', 2, cost=1)
        solver = MinCostFlow(net)
        flow, cost = solver.min_cost_max_flow('s', 't')
        assert flow == 4
        # Cheapest: 2 units via s->a->t (cost 3 each = 6) + 2 via s->b->t (cost 4 each = 8) = 14
        assert cost == 14

    def test_single_path(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 5, cost=2)
        net.add_edge('a', 't', 3, cost=1)
        solver = MinCostFlow(net)
        flow, cost = solver.min_cost_max_flow('s', 't')
        assert flow == 3
        assert cost == 3 * 3  # 3 units * (2+1) cost

    def test_prefer_cheap_path(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 5, cost=1)
        net.add_edge('s', 'b', 5, cost=10)
        net.add_edge('a', 't', 3, cost=0)
        net.add_edge('b', 't', 5, cost=0)
        solver = MinCostFlow(net)
        flow, cost = solver.min_cost_max_flow('s', 't')
        assert flow == 8
        # 3 units via cheap path (cost 1 each) + 5 via expensive (cost 10 each)
        assert cost == 3 * 1 + 5 * 10

    def test_no_path(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 5, cost=1)
        net.add_edge('b', 't', 5, cost=1)
        solver = MinCostFlow(net)
        flow, cost = solver.min_cost_max_flow('s', 't')
        assert flow == 0
        assert cost == 0

    def test_target_flow(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 10, cost=1)
        net.add_edge('a', 't', 10, cost=2)
        solver = MinCostFlow(net)
        result = solver.min_cost_flow('s', 't', 5)
        assert result is not None
        flow, cost = result
        assert flow == 5
        assert cost == 5 * 3

    def test_target_flow_impossible(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 3, cost=1)
        net.add_edge('a', 't', 3, cost=1)
        solver = MinCostFlow(net)
        result = solver.min_cost_flow('s', 't', 5)
        assert result is None

    def test_multiple_paths_different_costs(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 2, cost=1)
        net.add_edge('s', 'b', 2, cost=5)
        net.add_edge('a', 'c', 2, cost=1)
        net.add_edge('b', 'c', 2, cost=1)
        net.add_edge('c', 't', 4, cost=0)
        solver = MinCostFlow(net)
        flow, cost = solver.min_cost_max_flow('s', 't')
        assert flow == 4
        assert cost == 2 * 2 + 2 * 6  # cheap path: 2*(1+1), expensive: 2*(5+1)


# ═══════════════════════════════════════════════════════════════════
# Hopcroft-Karp Bipartite Matching
# ═══════════════════════════════════════════════════════════════════

class TestHopcroftKarp:

    def test_perfect_matching(self):
        hk = HopcroftKarp(
            [1, 2, 3], ['a', 'b', 'c'],
            [(1, 'a'), (2, 'b'), (3, 'c')]
        )
        m = hk.maximum_matching()
        assert len(m) == 3
        assert set(m.values()) == {'a', 'b', 'c'}

    def test_no_matching(self):
        hk = HopcroftKarp([1, 2], ['a', 'b'], [])
        m = hk.maximum_matching()
        assert len(m) == 0

    def test_partial_matching(self):
        hk = HopcroftKarp(
            [1, 2, 3], ['a', 'b'],
            [(1, 'a'), (2, 'a'), (3, 'b')]
        )
        m = hk.maximum_matching()
        assert len(m) == 2

    def test_augmenting_path(self):
        """Needs augmenting path to find optimal matching."""
        hk = HopcroftKarp(
            [1, 2], ['a', 'b'],
            [(1, 'a'), (1, 'b'), (2, 'a')]
        )
        m = hk.maximum_matching()
        assert len(m) == 2
        # Must be: 1->b, 2->a (augmenting from greedy 1->a)
        assert m[1] == 'b'
        assert m[2] == 'a'

    def test_complete_bipartite(self):
        left = [1, 2, 3]
        right = ['a', 'b', 'c']
        edges = [(l, r) for l in left for r in right]
        hk = HopcroftKarp(left, right, edges)
        m = hk.maximum_matching()
        assert len(m) == 3

    def test_unbalanced(self):
        hk = HopcroftKarp(
            [1, 2, 3, 4], ['a', 'b'],
            [(1, 'a'), (2, 'a'), (3, 'b'), (4, 'b')]
        )
        m = hk.maximum_matching()
        assert len(m) == 2

    def test_star_graph(self):
        """One right node connected to all left nodes."""
        hk = HopcroftKarp(
            [1, 2, 3], ['a'],
            [(1, 'a'), (2, 'a'), (3, 'a')]
        )
        m = hk.maximum_matching()
        assert len(m) == 1

    def test_multiple_components(self):
        hk = HopcroftKarp(
            [1, 2, 3, 4], ['a', 'b', 'c', 'd'],
            [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]
        )
        m = hk.maximum_matching()
        assert len(m) == 4

    def test_complex_matching(self):
        """Requires multiple BFS phases."""
        hk = HopcroftKarp(
            [1, 2, 3, 4], ['a', 'b', 'c', 'd'],
            [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'c'),
             (3, 'b'), (3, 'd'), (4, 'c'), (4, 'd')]
        )
        m = hk.maximum_matching()
        assert len(m) == 4

    def test_vertex_cover(self):
        hk = HopcroftKarp(
            [1, 2, 3], ['a', 'b', 'c'],
            [(1, 'a'), (2, 'b'), (3, 'c')]
        )
        cover = hk.minimum_vertex_cover()
        assert len(cover) == 3  # Perfect matching -> cover = matching size

    def test_vertex_cover_partial(self):
        hk = HopcroftKarp(
            [1, 2], ['a', 'b'],
            [(1, 'a'), (1, 'b'), (2, 'a')]
        )
        cover = hk.minimum_vertex_cover()
        # Matching size = 2, cover size = 2
        assert len(cover) == 2
        # Every edge must be covered
        for u, v in [(1, 'a'), (1, 'b'), (2, 'a')]:
            assert u in cover or v in cover

    def test_independent_set(self):
        hk = HopcroftKarp(
            [1, 2, 3], ['a', 'b', 'c'],
            [(1, 'a'), (2, 'b'), (3, 'c')]
        )
        indep = hk.maximum_independent_set()
        # Perfect matching: MIS = total - matching = 6 - 3 = 3
        assert len(indep) == 3
        # No edge should connect two nodes in the independent set
        for u, v in [(1, 'a'), (2, 'b'), (3, 'c')]:
            assert not (u in indep and v in indep)

    def test_independent_set_larger(self):
        hk = HopcroftKarp(
            [1, 2, 3], ['a'],
            [(1, 'a'), (2, 'a'), (3, 'a')]
        )
        indep = hk.maximum_independent_set()
        # Only 1 edge in matching, cover = 1, MIS = 4 - 1 = 3
        assert len(indep) == 3


# ═══════════════════════════════════════════════════════════════════
# Edge-Disjoint Paths
# ═══════════════════════════════════════════════════════════════════

class TestEdgeDisjointPaths:

    def test_simple(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 1)
        net.add_edge('s', 'b', 1)
        net.add_edge('a', 't', 1)
        net.add_edge('b', 't', 1)
        num, paths = edge_disjoint_paths(net, 's', 't')
        assert num == 2
        assert len(paths) == 2

    def test_single_path(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 1)
        net.add_edge('a', 't', 1)
        num, paths = edge_disjoint_paths(net, 's', 't')
        assert num == 1
        assert paths[0] == ['s', 'a', 't']

    def test_no_path(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 1)
        net.add_edge('b', 't', 1)
        num, paths = edge_disjoint_paths(net, 's', 't')
        assert num == 0

    def test_bottleneck(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 1)
        net.add_edge('s', 'b', 1)
        net.add_edge('a', 'c', 1)
        net.add_edge('b', 'c', 1)
        net.add_edge('c', 't', 1)
        num, paths = edge_disjoint_paths(net, 's', 't')
        assert num == 1  # Bottleneck at c->t


# ═══════════════════════════════════════════════════════════════════
# Node-Disjoint Paths
# ═══════════════════════════════════════════════════════════════════

class TestNodeDisjointPaths:

    def test_simple(self):
        adj = {'s': ['a', 'b'], 'a': ['t'], 'b': ['t'], 't': []}
        assert node_disjoint_paths(adj, 's', 't') == 2

    def test_bottleneck(self):
        adj = {'s': ['a', 'b'], 'a': ['c'], 'b': ['c'], 'c': ['t'], 't': []}
        assert node_disjoint_paths(adj, 's', 't') == 1

    def test_no_path(self):
        adj = {'s': ['a'], 'a': [], 'b': ['t'], 't': []}
        assert node_disjoint_paths(adj, 's', 't') == 0


# ═══════════════════════════════════════════════════════════════════
# Circulation with Demands
# ═══════════════════════════════════════════════════════════════════

class TestCirculation:

    def test_feasible(self):
        # Simple cycle with lower bounds
        edges = [
            ('a', 'b', 1, 5, 0),  # (src, dst, lower, upper, cost)
            ('b', 'c', 1, 5, 0),
            ('c', 'a', 1, 5, 0),
        ]
        feasible, flows = circulation_with_demands(['a', 'b', 'c'], edges)
        assert feasible
        for key, val in flows.items():
            assert 1 <= val <= 5

    def test_infeasible(self):
        # Lower bound > upper bound effectively
        edges = [
            ('a', 'b', 5, 3, 0),  # lower > upper -- impossible
        ]
        feasible, flows = circulation_with_demands(['a', 'b'], edges)
        assert not feasible

    def test_exact_demands(self):
        edges = [
            ('a', 'b', 3, 3, 0),  # Exactly 3 must flow
            ('b', 'a', 3, 3, 0),
        ]
        feasible, flows = circulation_with_demands(['a', 'b'], edges)
        assert feasible
        assert flows[('a', 'b')] == 3
        assert flows[('b', 'a')] == 3


# ═══════════════════════════════════════════════════════════════════
# Assignment Problem
# ═══════════════════════════════════════════════════════════════════

class TestAssignment:

    def test_simple(self):
        costs = [
            [3, 1, 2],
            [2, 3, 1],
            [1, 2, 3],
        ]
        total_cost, assignment = assignment_problem(costs)
        assert total_cost == 3  # Each worker gets cost-1 job
        assert len(assignment) == 3

    def test_identity(self):
        costs = [
            [1, 100],
            [100, 1],
        ]
        total_cost, assignment = assignment_problem(costs)
        assert total_cost == 2
        assert assignment[0] == 0
        assert assignment[1] == 1

    def test_all_same(self):
        costs = [
            [5, 5],
            [5, 5],
        ]
        total_cost, assignment = assignment_problem(costs)
        assert total_cost == 10

    def test_larger(self):
        costs = [
            [10, 5, 13, 4],
            [3, 7, 15, 6],
            [5, 8, 12, 9],
            [7, 3, 10, 2],
        ]
        total_cost, assignment = assignment_problem(costs)
        # Verify each worker assigned to unique job
        assert len(set(assignment.values())) == 4


# ═══════════════════════════════════════════════════════════════════
# Project Selection
# ═══════════════════════════════════════════════════════════════════

class TestProjectSelection:

    def test_independent_projects(self):
        profits = {'A': 10, 'B': 5}
        costs = {'A': 3, 'B': 8}
        deps = []
        max_profit, selected = project_selection(profits, costs, deps)
        assert 'A' in selected  # A has net profit 7
        assert 'B' not in selected  # B has net cost 3
        assert max_profit == 7

    def test_with_dependency(self):
        profits = {'A': 10}
        costs = {'A': 0, 'B': 0}
        deps = [('A', 'B')]  # A requires B
        max_profit, selected = project_selection(profits, costs, deps)
        assert 'A' in selected
        assert 'B' in selected
        assert max_profit == 10

    def test_dependency_makes_project_unprofitable(self):
        profits = {'A': 5}
        costs = {'A': 0, 'B': 0}
        deps = [('A', 'B')]
        # Both A and B are profitable together
        costs_neg = {'A': 0, 'B': 10}
        max_profit, selected = project_selection(
            profits, costs_neg, deps
        )
        # A gives 5, B costs 10 -- not worth it
        assert max_profit == 0
        assert 'A' not in selected

    def test_all_profitable(self):
        profits = {'A': 10, 'B': 20, 'C': 5}
        costs = {'A': 0, 'B': 0, 'C': 0}
        deps = []
        max_profit, selected = project_selection(profits, costs, deps)
        assert max_profit == 35
        assert selected == {'A', 'B', 'C'}

    def test_none_profitable(self):
        profits = {}
        costs = {'A': 5, 'B': 10}
        deps = []
        max_profit, selected = project_selection(profits, costs, deps)
        assert max_profit == 0
        assert len(selected) == 0


# ═══════════════════════════════════════════════════════════════════
# Stress / Edge cases
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_self_loop(self):
        """Self-loop should not affect max flow."""
        net = FlowNetwork()
        net.add_edge('s', 'a', 5)
        net.add_edge('a', 'a', 100)  # self-loop
        net.add_edge('a', 't', 3)
        assert EdmondsKarp(net).max_flow('s', 't') == 3

    def test_source_equals_sink(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 5)
        assert EdmondsKarp(net).max_flow('s', 's') == 0

    def test_disconnected_graph(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 5)
        net.add_edge('a', 'b', 3)
        net.add_edge('c', 'd', 10)
        net.add_edge('d', 't', 8)
        assert Dinic(net).max_flow('s', 't') == 0

    def test_multiple_sources_merged(self):
        """Simulate multiple sources via super-source."""
        net = FlowNetwork()
        net.add_edge('super_s', 's1', math.inf)
        net.add_edge('super_s', 's2', math.inf)
        net.add_edge('s1', 't', 5)
        net.add_edge('s2', 't', 3)
        assert EdmondsKarp(net).max_flow('super_s', 't') == 8

    def test_integer_capacities(self):
        net = FlowNetwork()
        net.add_edge('s', 'a', 7)
        net.add_edge('a', 'b', 11)
        net.add_edge('s', 'b', 13)
        net.add_edge('b', 't', 17)
        flow_ek = EdmondsKarp(FlowNetwork())
        # Build fresh for each
        for Solver in [EdmondsKarp, Dinic, PushRelabel]:
            net = FlowNetwork()
            net.add_edge('s', 'a', 7)
            net.add_edge('a', 'b', 11)
            net.add_edge('s', 'b', 13)
            net.add_edge('b', 't', 17)
            assert Solver(net).max_flow('s', 't') == 17

    def test_flow_conservation(self):
        """After max flow, flow in = flow out for intermediate nodes."""
        net = FlowNetwork()
        net.add_edge('s', 'a', 10)
        net.add_edge('s', 'b', 10)
        net.add_edge('a', 'b', 2)
        net.add_edge('a', 't', 6)
        net.add_edge('b', 't', 10)
        solver = EdmondsKarp(net)
        solver.max_flow('s', 't')
        # Check flow conservation at 'a' and 'b'
        for node in ['a', 'b']:
            flow_in = sum(e.flow for n in net.adj for e in net.adj[n]
                         if e.dst == node and e.cap > 0)
            flow_out = sum(e.flow for e in net.adj[node] if e.cap > 0)
            assert flow_in == flow_out

    def test_undirected_flow(self):
        net = FlowNetwork()
        net.add_undirected_edge('s', 'a', 5)
        net.add_undirected_edge('a', 't', 3)
        assert EdmondsKarp(net).max_flow('s', 't') == 3

    def test_max_flow_min_cut_theorem(self):
        """Max flow equals min cut for all algorithms."""
        for Solver in [EdmondsKarp, Dinic, PushRelabel]:
            net = FlowNetwork()
            net.add_edge('s', 'a', 10)
            net.add_edge('s', 'c', 10)
            net.add_edge('a', 'b', 4)
            net.add_edge('a', 'c', 2)
            net.add_edge('a', 'd', 8)
            net.add_edge('b', 't', 10)
            net.add_edge('c', 'd', 9)
            net.add_edge('d', 'b', 6)
            net.add_edge('d', 't', 10)
            solver = Solver(net)
            flow = solver.max_flow('s', 't')
            _, _, _, cut_value = solver.min_cut('s', 't')
            assert flow == cut_value


# ═══════════════════════════════════════════════════════════════════
# Larger integration tests
# ═══════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_layered_graph(self):
        """3-layer graph with 3 nodes per layer."""
        for Solver in [EdmondsKarp, Dinic, PushRelabel]:
            net = FlowNetwork()
            # Layer 0 (source) -> Layer 1
            for i in range(3):
                net.add_edge('s', f'L1_{i}', 5)
            # Layer 1 -> Layer 2
            for i in range(3):
                for j in range(3):
                    net.add_edge(f'L1_{i}', f'L2_{j}', 2)
            # Layer 2 -> sink
            for j in range(3):
                net.add_edge(f'L2_{j}', 't', 5)
            flow = Solver(net).max_flow('s', 't')
            assert flow == 15  # 3 * 5 in, 3 * 5 out, 9 * 2 = 18 middle capacity

    def test_matching_via_flow(self):
        """Bipartite matching via max-flow should agree with Hopcroft-Karp."""
        left = [1, 2, 3]
        right = ['a', 'b', 'c']
        edges = [(1, 'a'), (1, 'b'), (2, 'b'), (2, 'c'), (3, 'c')]

        # Hopcroft-Karp
        hk = HopcroftKarp(left, right, edges)
        hk_matching = hk.maximum_matching()

        # Flow-based
        net = FlowNetwork()
        for l in left:
            net.add_edge('s', l, 1)
        for l, r in edges:
            net.add_edge(l, r, 1)
        for r in right:
            net.add_edge(r, 't', 1)
        flow = EdmondsKarp(net).max_flow('s', 't')

        assert flow == len(hk_matching)

    def test_min_cost_assignment(self):
        """Assignment via min-cost flow should find optimal."""
        costs = [
            [9, 2, 7],
            [6, 4, 3],
            [5, 8, 1],
        ]
        total, assignment = assignment_problem(costs)
        # Optimal: 0->1(2), 1->0(6), 2->2(1) = 9
        # Or: 0->1(2), 1->2(3), 2->0(5) = 10
        # Minimum is 2+3+5=10? Let's check: 0->1=2, 1->2=3, 2->0=5 = 10
        # Or 0->2=7, 1->1=4, 2->0=5 = 16
        # Or 0->0=9, 1->2=3, 2->1=8 = 20
        # Or 0->1=2, 1->0=6, 2->2=1 = 9
        assert total == 9

    def test_path_graph(self):
        """Long path with varying capacities."""
        net = FlowNetwork()
        caps = [10, 7, 12, 3, 8, 5]
        nodes = ['s'] + [f'n{i}' for i in range(len(caps) - 1)] + ['t']
        for i, cap in enumerate(caps):
            net.add_edge(nodes[i], nodes[i + 1], cap)
        for Solver in [EdmondsKarp, Dinic, PushRelabel]:
            net2 = FlowNetwork()
            for i, cap in enumerate(caps):
                net2.add_edge(nodes[i], nodes[i + 1], cap)
            assert Solver(net2).max_flow('s', 't') == min(caps)

    def test_hopcroft_karp_job_scheduling(self):
        """Workers with time slots -- bipartite matching."""
        workers = ['Alice', 'Bob', 'Charlie', 'Diana']
        slots = ['Mon', 'Tue', 'Wed', 'Thu']
        availability = [
            ('Alice', 'Mon'), ('Alice', 'Tue'),
            ('Bob', 'Tue'), ('Bob', 'Wed'),
            ('Charlie', 'Wed'), ('Charlie', 'Thu'),
            ('Diana', 'Mon'), ('Diana', 'Thu'),
        ]
        hk = HopcroftKarp(workers, slots, availability)
        m = hk.maximum_matching()
        assert len(m) == 4  # Everyone gets a slot
