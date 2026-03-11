"""Tests for C099: Graph Coloring"""
import pytest
from graph_coloring import (
    Graph, greedy_color, dsatur_color, chromatic_number, is_k_colorable,
    is_valid_coloring, num_colors_used, color_classes, coloring_stats,
    order_natural, order_largest_first, order_smallest_last,
    complete_graph, cycle_graph, complete_bipartite, petersen_graph,
    wheel_graph, crown_graph, grid_graph,
    is_bipartite, is_chordal, is_complete, is_cycle,
    edge_chromatic_number_bounds, edge_color_greedy,
    fractional_chromatic_bound, interval_graph_color,
    register_allocate, welsh_powell_color, connected_components,
    chromatic_polynomial_tree, chromatic_polynomial_cycle,
    chromatic_polynomial_complete, map_color_greedy,
    max_clique_greedy, _max_independent_set_greedy,
)


# === Graph Construction ===

class TestGraphBasics:
    def test_empty_graph(self):
        g = Graph(0)
        assert g.n == 0
        assert g.edge_count() == 0

    def test_add_vertex(self):
        g = Graph(0)
        v = g.add_vertex()
        assert v == 0
        assert g.n == 1

    def test_add_edge(self):
        g = Graph(3)
        g.add_edge(0, 1)
        assert g.has_edge(0, 1)
        assert g.has_edge(1, 0)
        assert not g.has_edge(0, 2)

    def test_self_loop_rejected(self):
        g = Graph(3)
        with pytest.raises(ValueError):
            g.add_edge(1, 1)

    def test_out_of_range(self):
        g = Graph(3)
        with pytest.raises(ValueError):
            g.add_edge(0, 5)

    def test_degree(self):
        g = Graph(4)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        assert g.degree(0) == 3
        assert g.degree(1) == 1

    def test_max_degree(self):
        g = complete_graph(5)
        assert g.max_degree() == 4

    def test_edge_count(self):
        g = complete_graph(4)
        assert g.edge_count() == 6

    def test_complement(self):
        g = Graph(4)
        g.add_edge(0, 1)
        gc = g.complement()
        assert not gc.has_edge(0, 1)
        assert gc.has_edge(0, 2)
        assert gc.has_edge(0, 3)
        assert gc.has_edge(1, 2)
        assert gc.edge_count() == 5

    def test_subgraph(self):
        g = complete_graph(5)
        sg, mapping = g.subgraph([1, 2, 3])
        assert sg.n == 3
        assert sg.edge_count() == 3  # K3

    def test_copy(self):
        g = Graph(3)
        g.add_edge(0, 1)
        g2 = g.copy()
        g2.add_edge(1, 2)
        assert not g.has_edge(1, 2)
        assert g2.has_edge(1, 2)

    def test_neighbors(self):
        g = Graph(3)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        assert g.neighbors(0) == {1, 2}


# === Graph Constructors ===

class TestGraphConstructors:
    def test_complete_graph(self):
        g = complete_graph(5)
        assert g.n == 5
        assert g.edge_count() == 10

    def test_cycle_graph(self):
        g = cycle_graph(6)
        assert g.n == 6
        assert g.edge_count() == 6
        assert all(g.degree(v) == 2 for v in range(6))

    def test_cycle_too_small(self):
        with pytest.raises(ValueError):
            cycle_graph(2)

    def test_complete_bipartite(self):
        g = complete_bipartite(3, 4)
        assert g.n == 7
        assert g.edge_count() == 12

    def test_petersen(self):
        g = petersen_graph()
        assert g.n == 10
        assert g.edge_count() == 15
        assert all(g.degree(v) == 3 for v in range(10))

    def test_wheel(self):
        g = wheel_graph(5)
        assert g.n == 6
        assert g.degree(0) == 5  # Hub

    def test_wheel_too_small(self):
        with pytest.raises(ValueError):
            wheel_graph(2)

    def test_crown(self):
        g = crown_graph(4)
        assert g.n == 8
        assert all(g.degree(v) == 3 for v in range(8))

    def test_crown_too_small(self):
        with pytest.raises(ValueError):
            crown_graph(1)

    def test_grid(self):
        g = grid_graph(3, 3)
        assert g.n == 9
        assert g.edge_count() == 12


# === Vertex Ordering ===

class TestOrdering:
    def test_natural(self):
        g = Graph(5)
        assert order_natural(g) == [0, 1, 2, 3, 4]

    def test_largest_first(self):
        g = Graph(4)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        order = order_largest_first(g)
        assert order[0] == 0  # Highest degree

    def test_smallest_last(self):
        g = Graph(4)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        order = order_smallest_last(g)
        assert len(order) == 4
        assert set(order) == {0, 1, 2, 3}


# === Greedy Coloring ===

class TestGreedyColor:
    def test_empty(self):
        g = Graph(0)
        c = greedy_color(g)
        assert c == {}

    def test_single_vertex(self):
        g = Graph(1)
        c = greedy_color(g)
        assert c == {0: 0}

    def test_edge(self):
        g = Graph(2)
        g.add_edge(0, 1)
        c = greedy_color(g)
        assert c[0] != c[1]

    def test_triangle(self):
        g = complete_graph(3)
        c = greedy_color(g)
        assert num_colors_used(c) == 3
        assert is_valid_coloring(g, c)

    def test_bipartite(self):
        g = complete_bipartite(3, 3)
        c = greedy_color(g, order_smallest_last(g))
        assert is_valid_coloring(g, c)
        assert num_colors_used(c) <= 3  # May not be optimal

    def test_path(self):
        g = Graph(5)
        for i in range(4):
            g.add_edge(i, i + 1)
        c = greedy_color(g)
        assert is_valid_coloring(g, c)
        assert num_colors_used(c) == 2

    def test_custom_order(self):
        g = complete_graph(4)
        c = greedy_color(g, [3, 2, 1, 0])
        assert is_valid_coloring(g, c)
        assert num_colors_used(c) == 4

    def test_independent_set(self):
        g = Graph(5)  # No edges
        c = greedy_color(g)
        assert num_colors_used(c) == 1
        assert all(c[v] == 0 for v in range(5))

    def test_star(self):
        g = Graph(6)
        for i in range(1, 6):
            g.add_edge(0, i)
        c = greedy_color(g)
        assert is_valid_coloring(g, c)
        assert num_colors_used(c) == 2


# === DSatur ===

class TestDSatur:
    def test_empty(self):
        assert dsatur_color(Graph(0)) == {}

    def test_complete(self):
        g = complete_graph(5)
        c = dsatur_color(g)
        assert num_colors_used(c) == 5
        assert is_valid_coloring(g, c)

    def test_cycle_even(self):
        g = cycle_graph(6)
        c = dsatur_color(g)
        assert is_valid_coloring(g, c)
        assert num_colors_used(c) == 2

    def test_cycle_odd(self):
        g = cycle_graph(5)
        c = dsatur_color(g)
        assert is_valid_coloring(g, c)
        assert num_colors_used(c) == 3

    def test_petersen(self):
        g = petersen_graph()
        c = dsatur_color(g)
        assert is_valid_coloring(g, c)
        assert num_colors_used(c) <= 4  # DSatur often gets 3

    def test_bipartite_optimal(self):
        g = complete_bipartite(4, 4)
        c = dsatur_color(g)
        assert is_valid_coloring(g, c)
        assert num_colors_used(c) == 2

    def test_wheel_even(self):
        g = wheel_graph(4)
        c = dsatur_color(g)
        assert is_valid_coloring(g, c)

    def test_wheel_odd(self):
        g = wheel_graph(5)
        c = dsatur_color(g)
        assert is_valid_coloring(g, c)

    def test_grid(self):
        g = grid_graph(4, 4)
        c = dsatur_color(g)
        assert is_valid_coloring(g, c)
        assert num_colors_used(c) == 2

    def test_crown(self):
        g = crown_graph(5)
        c = dsatur_color(g)
        assert is_valid_coloring(g, c)


# === Chromatic Number ===

class TestChromaticNumber:
    def test_empty(self):
        chi, c = chromatic_number(Graph(0))
        assert chi == 0

    def test_single(self):
        chi, c = chromatic_number(Graph(1))
        assert chi == 1

    def test_independent(self):
        chi, c = chromatic_number(Graph(5))
        assert chi == 1
        assert is_valid_coloring(Graph(5), c)

    def test_edge(self):
        g = Graph(2)
        g.add_edge(0, 1)
        chi, c = chromatic_number(g)
        assert chi == 2

    def test_triangle(self):
        chi, c = chromatic_number(complete_graph(3))
        assert chi == 3
        assert is_valid_coloring(complete_graph(3), c)

    def test_k4(self):
        chi, c = chromatic_number(complete_graph(4))
        assert chi == 4

    def test_k5(self):
        chi, c = chromatic_number(complete_graph(5))
        assert chi == 5

    def test_cycle_4(self):
        chi, c = chromatic_number(cycle_graph(4))
        assert chi == 2

    def test_cycle_5(self):
        chi, c = chromatic_number(cycle_graph(5))
        assert chi == 3

    def test_cycle_6(self):
        chi, c = chromatic_number(cycle_graph(6))
        assert chi == 2

    def test_petersen(self):
        g = petersen_graph()
        chi, c = chromatic_number(g, timeout=5)
        assert chi == 3
        assert is_valid_coloring(g, c)

    def test_bipartite(self):
        g = complete_bipartite(3, 3)
        chi, c = chromatic_number(g)
        assert chi == 2

    def test_wheel_4(self):
        g = wheel_graph(4)
        chi, c = chromatic_number(g)
        assert chi == 3  # W4: hub + C4 (even), 3-colorable

    def test_wheel_5(self):
        g = wheel_graph(5)
        chi, c = chromatic_number(g)
        # W5 hub adjacent to all, rim is C5 (odd), so chi=4
        assert chi == 4

    def test_wheel_6(self):
        g = wheel_graph(6)
        chi, c = chromatic_number(g)
        # W6 hub adjacent to all, rim is C6 (even), so chi=3
        assert chi == 3


# === k-Colorability ===

class TestKColorable:
    def test_trivial(self):
        g = Graph(3)
        assert is_k_colorable(g, 1)

    def test_edge_1_color(self):
        g = Graph(2)
        g.add_edge(0, 1)
        assert not is_k_colorable(g, 1)

    def test_edge_2_colors(self):
        g = Graph(2)
        g.add_edge(0, 1)
        assert is_k_colorable(g, 2)

    def test_triangle_2_colors(self):
        g = complete_graph(3)
        assert not is_k_colorable(g, 2)

    def test_triangle_3_colors(self):
        assert is_k_colorable(complete_graph(3), 3)

    def test_enough_colors(self):
        g = complete_graph(5)
        assert is_k_colorable(g, 10)

    def test_zero_colors(self):
        assert not is_k_colorable(Graph(1), 0)
        assert is_k_colorable(Graph(0), 0)


# === Validation and Analysis ===

class TestAnalysis:
    def test_valid_coloring(self):
        g = Graph(3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        assert is_valid_coloring(g, {0: 0, 1: 1, 2: 0})

    def test_invalid_coloring(self):
        g = Graph(3)
        g.add_edge(0, 1)
        assert not is_valid_coloring(g, {0: 0, 1: 0, 2: 0})

    def test_incomplete_coloring(self):
        g = Graph(3)
        assert not is_valid_coloring(g, {0: 0, 1: 1})

    def test_num_colors(self):
        assert num_colors_used({0: 0, 1: 1, 2: 0, 3: 2}) == 3
        assert num_colors_used({}) == 0

    def test_color_classes(self):
        cc = color_classes({0: 0, 1: 1, 2: 0, 3: 1})
        assert cc[0] == {0, 2}
        assert cc[1] == {1, 3}

    def test_coloring_stats(self):
        g = Graph(4)
        g.add_edge(0, 1)
        g.add_edge(2, 3)
        c = {0: 0, 1: 1, 2: 0, 3: 1}
        stats = coloring_stats(g, c)
        assert stats['num_colors'] == 2
        assert stats['valid']
        assert stats['balance'] == 1.0

    def test_coloring_stats_unbalanced(self):
        g = Graph(4)
        g.add_edge(0, 1)
        c = {0: 0, 1: 1, 2: 0, 3: 0}
        stats = coloring_stats(g, c)
        assert stats['balance'] == pytest.approx(1 / 3)


# === Special Graph Recognition ===

class TestRecognition:
    def test_bipartite_true(self):
        g = complete_bipartite(3, 3)
        bip, parts = is_bipartite(g)
        assert bip
        assert len(parts[0]) + len(parts[1]) == 6

    def test_bipartite_false(self):
        g = complete_graph(3)
        bip, _ = is_bipartite(g)
        assert not bip

    def test_bipartite_empty(self):
        bip, parts = is_bipartite(Graph(0))
        assert bip

    def test_bipartite_cycle_even(self):
        bip, _ = is_bipartite(cycle_graph(4))
        assert bip

    def test_bipartite_cycle_odd(self):
        bip, _ = is_bipartite(cycle_graph(5))
        assert not bip

    def test_bipartite_disconnected(self):
        g = Graph(4)
        g.add_edge(0, 1)
        g.add_edge(2, 3)
        bip, _ = is_bipartite(g)
        assert bip

    def test_chordal_complete(self):
        assert is_chordal(complete_graph(5))

    def test_chordal_tree(self):
        g = Graph(5)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 3)
        g.add_edge(1, 4)
        assert is_chordal(g)

    def test_chordal_cycle4(self):
        assert not is_chordal(cycle_graph(4))

    def test_chordal_small(self):
        assert is_chordal(Graph(2))

    def test_complete_true(self):
        assert is_complete(complete_graph(4))

    def test_complete_false(self):
        assert not is_complete(cycle_graph(4))

    def test_complete_single(self):
        assert is_complete(Graph(1))

    def test_is_cycle_true(self):
        assert is_cycle(cycle_graph(5))

    def test_is_cycle_false(self):
        assert not is_cycle(complete_graph(4))

    def test_is_cycle_small(self):
        assert not is_cycle(Graph(2))

    def test_is_cycle_disconnected(self):
        g = Graph(6)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        g.add_edge(3, 4)
        g.add_edge(4, 5)
        g.add_edge(5, 3)
        assert not is_cycle(g)


# === Edge Coloring ===

class TestEdgeColoring:
    def test_bounds_bipartite(self):
        g = complete_bipartite(3, 3)
        lo, hi = edge_chromatic_number_bounds(g)
        assert lo == 3
        assert hi == 3  # Bipartite: exact

    def test_bounds_triangle(self):
        g = complete_graph(3)
        lo, hi = edge_chromatic_number_bounds(g)
        assert lo == 2
        assert hi == 3

    def test_bounds_petersen(self):
        g = petersen_graph()
        lo, hi = edge_chromatic_number_bounds(g)
        assert lo == 3
        assert hi == 4

    def test_greedy_valid(self):
        g = complete_graph(5)
        ec = edge_color_greedy(g)
        assert len(ec) == 10
        # Check no vertex uses same color twice
        for v in range(5):
            colors = set()
            for u in g.adj[v]:
                key = (min(v, u), max(v, u))
                c = ec[key]
                assert c not in colors
                colors.add(c)

    def test_greedy_bipartite(self):
        g = complete_bipartite(2, 3)
        ec = edge_color_greedy(g)
        assert len(ec) == 6

    def test_greedy_path(self):
        g = Graph(4)
        for i in range(3):
            g.add_edge(i, i + 1)
        ec = edge_color_greedy(g)
        assert len(ec) == 3
        num = len(set(ec.values()))
        assert num == 2  # Path needs 2 edge colors


# === Fractional Chromatic ===

class TestFractionalChromatic:
    def test_complete(self):
        g = complete_graph(5)
        lo, hi = fractional_chromatic_bound(g)
        assert lo >= 5  # n/alpha = 5/1 = 5
        assert hi == 5

    def test_empty_graph(self):
        lo, hi = fractional_chromatic_bound(Graph(0))
        assert lo == 0
        assert hi == 0

    def test_bipartite(self):
        g = complete_bipartite(3, 3)
        lo, hi = fractional_chromatic_bound(g)
        assert lo >= 1
        assert hi == 2


# === Interval Graph Coloring ===

class TestIntervalGraph:
    def test_empty(self):
        c, n = interval_graph_color([])
        assert n == 0

    def test_non_overlapping(self):
        c, n = interval_graph_color([(0, 1), (2, 3), (4, 5)])
        assert n == 1

    def test_all_overlapping(self):
        c, n = interval_graph_color([(0, 10), (1, 10), (2, 10)])
        assert n == 3

    def test_partial_overlap(self):
        c, n = interval_graph_color([(0, 3), (1, 4), (5, 8)])
        assert n == 2

    def test_nested(self):
        c, n = interval_graph_color([(0, 10), (2, 5), (7, 9)])
        assert n == 2

    def test_color_reuse(self):
        c, n = interval_graph_color([(0, 1), (2, 3), (0, 1)])
        assert n == 2  # Two overlap at [0,1]


# === Register Allocation ===

class TestRegisterAllocation:
    def test_empty(self):
        alloc, n = register_allocate({})
        assert n == 0

    def test_no_conflict(self):
        alloc, n = register_allocate({'a': (0, 1), 'b': (2, 3)})
        assert n == 1

    def test_conflict(self):
        alloc, n = register_allocate({'a': (0, 5), 'b': (2, 7), 'c': (6, 10)})
        assert n == 2
        assert alloc['a'] != alloc['b']

    def test_all_conflict(self):
        alloc, n = register_allocate({'a': (0, 10), 'b': (0, 10), 'c': (0, 10)})
        assert n == 3
        assert len(set(alloc.values())) == 3


# === Welsh-Powell ===

class TestWelshPowell:
    def test_complete(self):
        g = complete_graph(4)
        c = welsh_powell_color(g)
        assert is_valid_coloring(g, c)
        assert num_colors_used(c) == 4

    def test_cycle(self):
        g = cycle_graph(6)
        c = welsh_powell_color(g)
        assert is_valid_coloring(g, c)
        assert num_colors_used(c) == 2

    def test_bipartite(self):
        g = complete_bipartite(3, 3)
        c = welsh_powell_color(g)
        assert is_valid_coloring(g, c)
        assert num_colors_used(c) == 2


# === Connected Components ===

class TestComponents:
    def test_connected(self):
        g = complete_graph(4)
        comps = connected_components(g)
        assert len(comps) == 1

    def test_disconnected(self):
        g = Graph(6)
        g.add_edge(0, 1)
        g.add_edge(2, 3)
        g.add_edge(4, 5)
        comps = connected_components(g)
        assert len(comps) == 3

    def test_isolated(self):
        g = Graph(3)
        comps = connected_components(g)
        assert len(comps) == 3

    def test_mixed(self):
        g = Graph(5)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        comps = connected_components(g)
        assert len(comps) == 3  # {0,1,2}, {3}, {4}


# === Chromatic Polynomials ===

class TestChromaticPolynomials:
    def test_tree_2(self):
        P = chromatic_polynomial_tree(2)
        assert P(2) == 2  # 2 * 1^1
        assert P(3) == 6

    def test_tree_3(self):
        P = chromatic_polynomial_tree(3)
        assert P(2) == 2  # 2 * 1^2
        assert P(3) == 12  # 3 * 2^2

    def test_tree_1(self):
        P = chromatic_polynomial_tree(1)
        assert P(5) == 5

    def test_cycle_3(self):
        P = chromatic_polynomial_cycle(3)
        assert P(2) == 0  # Odd cycle, not 2-colorable
        assert P(3) == 6  # (3-1)^3 + (-1)^3 * (3-1) = 8 - 2 = 6

    def test_cycle_4(self):
        P = chromatic_polynomial_cycle(4)
        assert P(2) == 2  # (2-1)^4 + (2-1) = 2
        assert P(3) == 18  # (3-1)^4 + (3-1) = 16+2 = 18

    def test_complete_3(self):
        P = chromatic_polynomial_complete(3)
        assert P(3) == 6  # 3*2*1
        assert P(4) == 24  # 4*3*2
        assert P(2) == 0  # 2*1*0

    def test_complete_1(self):
        P = chromatic_polynomial_complete(1)
        assert P(5) == 5


# === Map Coloring ===

class TestMapColor:
    def test_planar(self):
        g = grid_graph(3, 3)
        c = map_color_greedy(g)
        assert is_valid_coloring(g, c)
        assert num_colors_used(c) <= 4

    def test_cycle(self):
        g = cycle_graph(5)
        c = map_color_greedy(g)
        assert is_valid_coloring(g, c)


# === Clique and Independence ===

class TestCliqueIndependence:
    def test_clique_complete(self):
        assert max_clique_greedy(complete_graph(5)) == 5

    def test_clique_empty(self):
        assert max_clique_greedy(Graph(0)) == 0

    def test_clique_path(self):
        g = Graph(5)
        for i in range(4):
            g.add_edge(i, i + 1)
        assert max_clique_greedy(g) == 2

    def test_independent_empty(self):
        assert _max_independent_set_greedy(Graph(0)) == 0

    def test_independent_complete(self):
        assert _max_independent_set_greedy(complete_graph(5)) == 1

    def test_independent_edgeless(self):
        assert _max_independent_set_greedy(Graph(5)) == 5


# === Integration Tests ===

class TestIntegration:
    def test_greedy_vs_dsatur(self):
        """DSatur should be at least as good as natural-order greedy."""
        g = petersen_graph()
        c_greedy = greedy_color(g)
        c_dsatur = dsatur_color(g)
        assert num_colors_used(c_dsatur) <= num_colors_used(c_greedy)
        assert is_valid_coloring(g, c_dsatur)

    def test_chromatic_matches_dsatur_on_bipartite(self):
        g = complete_bipartite(3, 4)
        chi, c = chromatic_number(g)
        c_ds = dsatur_color(g)
        assert chi == num_colors_used(c_ds) == 2

    def test_complement_coloring(self):
        """Complement of K5 is edgeless, chi=1."""
        g = complete_graph(5).complement()
        chi, c = chromatic_number(g)
        assert chi == 1

    def test_crown_chromatic(self):
        g = crown_graph(3)
        chi, c = chromatic_number(g)
        assert chi == 2  # Crown graphs are bipartite

    def test_grid_chromatic(self):
        g = grid_graph(3, 4)
        chi, c = chromatic_number(g)
        assert chi == 2  # Grids are bipartite

    def test_wheel_analysis(self):
        g = wheel_graph(6)
        chi, c = chromatic_number(g)
        stats = coloring_stats(g, c)
        assert stats['valid']
        assert stats['num_colors'] == chi

    def test_chordal_optimal_greedy(self):
        """For chordal graphs, greedy with PEO gives optimal coloring."""
        # K4 minus one edge is chordal
        g = Graph(4)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(0, 3)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        assert is_chordal(g)
        chi, c = chromatic_number(g)
        assert chi == 3

    def test_edge_color_within_bounds(self):
        g = petersen_graph()
        lo, hi = edge_chromatic_number_bounds(g)
        ec = edge_color_greedy(g)
        n_colors = len(set(ec.values()))
        assert n_colors >= lo

    def test_all_algorithms_agree_triangle(self):
        g = complete_graph(3)
        assert num_colors_used(greedy_color(g)) == 3
        assert num_colors_used(dsatur_color(g)) == 3
        assert num_colors_used(welsh_powell_color(g)) == 3
        chi, _ = chromatic_number(g)
        assert chi == 3

    def test_polynomial_matches_exact(self):
        """Chromatic polynomial at k=chi should be > 0, at k=chi-1 should be 0."""
        P = chromatic_polynomial_complete(4)
        assert P(4) > 0
        assert P(3) == 0

    def test_large_sparse(self):
        """Test on a larger sparse graph."""
        g = Graph(20)
        for i in range(19):
            g.add_edge(i, i + 1)
        chi, c = chromatic_number(g)
        assert chi == 2
        assert is_valid_coloring(g, c)

    def test_disconnected_chromatic(self):
        """Chromatic number of disconnected graph = max over components."""
        g = Graph(7)
        # Triangle (chi=3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)
        # Edge (chi=2)
        g.add_edge(3, 4)
        # Two isolated
        chi, c = chromatic_number(g)
        assert chi == 3
        assert is_valid_coloring(g, c)
