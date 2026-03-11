"""Tests for C093: Network Analysis."""

import pytest
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C092_graph_algorithms'))

from network_analysis import (
    Graph, pagerank, degree_centrality, closeness_centrality,
    betweenness_centrality, eigenvector_centrality, hits,
    clustering_coefficient, average_clustering, transitivity,
    k_core_decomposition, k_core, label_propagation, communities_from_labels,
    louvain, modularity, connected_components, strongly_connected_components,
    density, eccentricity, diameter, radius, center,
    bridges, articulation_points, katz_centrality, degree_assortativity,
    rich_club_coefficient, neighborhood_overlap, common_neighbors,
    adamic_adar, reciprocity,
)


# ===========================================================================
# Helper
# ===========================================================================

def approx(a, b, tol=0.05):
    """Check approximate equality."""
    return abs(a - b) < tol


def make_triangle():
    """A-B-C triangle."""
    g = Graph()
    g.add_edge('A', 'B')
    g.add_edge('B', 'C')
    g.add_edge('A', 'C')
    return g


def make_path(n):
    """Path graph 0-1-2-..-(n-1)."""
    g = Graph()
    for i in range(n - 1):
        g.add_edge(i, i + 1)
    return g


def make_star(n):
    """Star graph: center 0 connected to 1..n-1."""
    g = Graph()
    for i in range(1, n):
        g.add_edge(0, i)
    return g


def make_complete(n):
    """Complete graph on n nodes."""
    g = Graph()
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(i, j)
    return g


def make_karate_like():
    """Small social network (simplified Zachary-like)."""
    g = Graph()
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 3),
        (2, 3),
        (4, 5), (4, 6),
        (5, 6), (5, 7),
        (6, 7),
        (7, 8), (7, 9),
        (8, 9),
    ]
    for u, v in edges:
        g.add_edge(u, v)
    return g


def make_two_cliques():
    """Two cliques connected by a bridge."""
    g = Graph()
    # Clique 1: 0,1,2,3
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(i, j)
    # Clique 2: 4,5,6,7
    for i in range(4, 8):
        for j in range(i + 1, 8):
            g.add_edge(i, j)
    # Bridge
    g.add_edge(3, 4)
    return g


# ===========================================================================
# PageRank
# ===========================================================================

class TestPageRank:
    def test_empty_graph(self):
        g = Graph(directed=True)
        assert pagerank(g) == {}

    def test_single_node(self):
        g = Graph(directed=True)
        g.add_node('A')
        pr = pagerank(g)
        assert approx(pr['A'], 1.0)

    def test_two_nodes_mutual(self):
        g = Graph(directed=True)
        g.add_edge('A', 'B')
        g.add_edge('B', 'A')
        pr = pagerank(g)
        assert approx(pr['A'], 0.5)
        assert approx(pr['B'], 0.5)

    def test_chain(self):
        g = Graph(directed=True)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        pr = pagerank(g)
        # C should have highest rank (sink)
        assert pr['C'] > pr['B'] > pr['A']

    def test_star_directed(self):
        g = Graph(directed=True)
        for i in range(1, 5):
            g.add_edge(0, i)
        pr = pagerank(g)
        # All leaves are sinks, get rank from center
        assert all(approx(pr[i], pr[1]) for i in range(2, 5))

    def test_sums_to_one(self):
        g = Graph(directed=True)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        g.add_edge('C', 'A')
        g.add_edge('A', 'C')
        pr = pagerank(g)
        assert approx(sum(pr.values()), 1.0, tol=0.01)

    def test_damping_factor(self):
        g = Graph(directed=True)
        g.add_edge('A', 'B')
        g.add_edge('B', 'A')
        pr1 = pagerank(g, damping=0.5)
        pr2 = pagerank(g, damping=0.99)
        # Both should give equal ranks for symmetric graph
        assert approx(pr1['A'], 0.5)
        assert approx(pr2['A'], 0.5)

    def test_dangling_node(self):
        g = Graph(directed=True)
        g.add_edge('A', 'B')
        g.add_edge('A', 'C')
        # B and C are dangling (no outgoing edges)
        pr = pagerank(g)
        assert approx(pr['B'], pr['C'], tol=0.01)
        assert sum(pr.values()) > 0.99

    def test_cycle(self):
        g = Graph(directed=True)
        for i in range(5):
            g.add_edge(i, (i + 1) % 5)
        pr = pagerank(g)
        # Symmetric cycle -> equal ranks
        for i in range(5):
            assert approx(pr[i], 0.2, tol=0.01)


# ===========================================================================
# Degree Centrality
# ===========================================================================

class TestDegreeCentrality:
    def test_single_node(self):
        g = Graph()
        g.add_node(0)
        dc = degree_centrality(g)
        assert dc[0] == 0.0

    def test_complete_graph(self):
        g = make_complete(5)
        dc = degree_centrality(g)
        for v in range(5):
            assert approx(dc[v], 1.0)

    def test_star(self):
        g = make_star(5)
        dc = degree_centrality(g)
        assert approx(dc[0], 1.0)
        for i in range(1, 5):
            assert approx(dc[i], 1.0 / 4)

    def test_path(self):
        g = make_path(4)  # 0-1-2-3
        dc = degree_centrality(g)
        assert approx(dc[0], 1.0 / 3)
        assert approx(dc[1], 2.0 / 3)
        assert approx(dc[2], 2.0 / 3)
        assert approx(dc[3], 1.0 / 3)

    def test_directed(self):
        g = Graph(directed=True)
        g.add_edge('A', 'B')
        g.add_edge('A', 'C')
        g.add_edge('B', 'C')
        in_c, out_c = degree_centrality(g)
        assert approx(out_c['A'], 1.0)
        assert approx(in_c['C'], 1.0)
        assert approx(in_c['A'], 0.0)
        assert approx(out_c['C'], 0.0)


# ===========================================================================
# Closeness Centrality
# ===========================================================================

class TestClosenessCentrality:
    def test_single_node(self):
        g = Graph()
        g.add_node(0)
        cc = closeness_centrality(g)
        assert cc[0] == 0.0

    def test_complete_graph(self):
        g = make_complete(4)
        cc = closeness_centrality(g)
        for v in range(4):
            assert approx(cc[v], 1.0)

    def test_star_center(self):
        g = make_star(5)
        cc = closeness_centrality(g)
        assert cc[0] > cc[1]  # center is more central

    def test_path(self):
        g = make_path(5)  # 0-1-2-3-4
        cc = closeness_centrality(g)
        # Middle node (2) should be most central
        assert cc[2] > cc[0]
        assert cc[2] > cc[4]

    def test_disconnected(self):
        g = Graph()
        g.add_edge(0, 1)
        g.add_node(2)  # isolated
        cc = closeness_centrality(g)
        assert cc[2] == 0.0  # isolated node
        assert cc[0] > 0.0


# ===========================================================================
# Betweenness Centrality
# ===========================================================================

class TestBetweennessCentrality:
    def test_single_node(self):
        g = Graph()
        g.add_node(0)
        bc = betweenness_centrality(g)
        assert bc[0] == 0.0

    def test_path(self):
        g = make_path(5)  # 0-1-2-3-4
        bc = betweenness_centrality(g, normalized=False)
        # Node 2 has highest betweenness (on 6 shortest paths)
        assert bc[2] > bc[1]
        assert bc[2] > bc[3]

    def test_star(self):
        g = make_star(5)
        bc = betweenness_centrality(g, normalized=False)
        # Center is on all shortest paths between leaves
        assert bc[0] > 0
        for i in range(1, 5):
            assert bc[i] == 0.0

    def test_triangle(self):
        g = make_triangle()
        bc = betweenness_centrality(g, normalized=False)
        # No node is on any shortest path (direct edges exist)
        for v in ['A', 'B', 'C']:
            assert bc[v] == 0.0

    def test_bridge_node(self):
        g = make_two_cliques()
        bc = betweenness_centrality(g, normalized=False)
        # Nodes 3 and 4 bridge the cliques
        assert bc[3] > bc[0]
        assert bc[4] > bc[5]

    def test_normalized(self):
        g = make_path(4)
        bc = betweenness_centrality(g, normalized=True)
        for v in range(4):
            assert bc[v] <= 1.0


# ===========================================================================
# Eigenvector Centrality
# ===========================================================================

class TestEigenvectorCentrality:
    def test_empty(self):
        g = Graph()
        assert eigenvector_centrality(g) == {}

    def test_complete(self):
        g = make_complete(4)
        ec = eigenvector_centrality(g)
        # All nodes equal in complete graph
        vals = list(ec.values())
        for v in vals:
            assert approx(v, vals[0], tol=0.01)

    def test_star(self):
        g = make_star(5)
        ec = eigenvector_centrality(g)
        # Center has same eigenvector centrality as each leaf in star
        # (principal eigenvector has equal entries when degree pattern is symmetric)
        assert ec[0] >= ec[1]
        # But all leaves should be equal
        leaf_vals = [ec[i] for i in range(1, 5)]
        for v in leaf_vals:
            assert approx(v, leaf_vals[0], tol=0.01)

    def test_path(self):
        g = make_path(5)
        ec = eigenvector_centrality(g)
        # Middle nodes should have higher centrality
        assert ec[2] > ec[0]
        assert ec[2] > ec[4]


# ===========================================================================
# HITS
# ===========================================================================

class TestHITS:
    def test_empty(self):
        g = Graph(directed=True)
        h, a = hits(g)
        assert h == {}
        assert a == {}

    def test_simple_chain(self):
        g = Graph(directed=True)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        h, a = hits(g)
        # A is hub (points to things), C is authority (pointed to)
        assert h['A'] > h['C']  # A is best hub
        assert a['C'] > a['A']  # C is best authority

    def test_mutual_links(self):
        g = Graph(directed=True)
        g.add_edge('A', 'B')
        g.add_edge('B', 'A')
        h, a = hits(g)
        assert approx(h['A'], h['B'], tol=0.01)
        assert approx(a['A'], a['B'], tol=0.01)

    def test_star_hub(self):
        g = Graph(directed=True)
        for i in range(1, 5):
            g.add_edge(0, i)
        h, a = hits(g)
        # Node 0 is the hub
        assert h[0] > h[1]
        # Leaves are authorities
        assert a[1] > a[0]


# ===========================================================================
# Clustering Coefficient
# ===========================================================================

class TestClusteringCoefficient:
    def test_triangle(self):
        g = make_triangle()
        cc = clustering_coefficient(g)
        for v in ['A', 'B', 'C']:
            assert approx(cc[v], 1.0)

    def test_star(self):
        g = make_star(5)
        cc = clustering_coefficient(g)
        # Center: neighbors not connected to each other
        assert cc[0] == 0.0
        # Leaves: only one neighbor
        for i in range(1, 5):
            assert cc[i] == 0.0

    def test_single_node(self):
        g = Graph()
        g.add_node(0)
        cc = clustering_coefficient(g, node=0)
        assert cc == 0.0

    def test_complete(self):
        g = make_complete(5)
        cc = clustering_coefficient(g)
        for v in range(5):
            assert approx(cc[v], 1.0)

    def test_path(self):
        g = make_path(4)
        cc = clustering_coefficient(g)
        assert cc[0] == 0.0
        assert cc[1] == 0.0
        assert cc[3] == 0.0

    def test_average(self):
        g = make_triangle()
        assert approx(average_clustering(g), 1.0)

    def test_transitivity_complete(self):
        g = make_complete(4)
        assert approx(transitivity(g), 1.0)

    def test_transitivity_star(self):
        g = make_star(5)
        assert transitivity(g) == 0.0


# ===========================================================================
# K-Core Decomposition
# ===========================================================================

class TestKCore:
    def test_single_node(self):
        g = Graph()
        g.add_node(0)
        cores = k_core_decomposition(g)
        assert cores[0] == 0

    def test_triangle(self):
        g = make_triangle()
        cores = k_core_decomposition(g)
        for v in ['A', 'B', 'C']:
            assert cores[v] == 2

    def test_path(self):
        g = make_path(5)
        cores = k_core_decomposition(g)
        assert cores[0] == 1
        assert cores[4] == 1
        assert cores[2] == 1  # path is 1-core

    def test_complete(self):
        g = make_complete(5)
        cores = k_core_decomposition(g)
        for v in range(5):
            assert cores[v] == 4  # K5 is 4-core

    def test_k_core_subgraph(self):
        g = make_two_cliques()
        sub = k_core(g, 3)
        # Each clique is a 3-core (K4)
        assert sub.node_count() == 8

    def test_k_core_high(self):
        g = make_two_cliques()
        sub = k_core(g, 4)
        # No 4-core in two K4s connected by bridge
        assert sub.node_count() == 0


# ===========================================================================
# Label Propagation
# ===========================================================================

class TestLabelPropagation:
    def test_single_node(self):
        g = Graph()
        g.add_node(0)
        labels = label_propagation(g, seed=42)
        assert 0 in labels

    def test_two_cliques(self):
        g = make_two_cliques()
        labels = label_propagation(g, seed=42)
        comms = communities_from_labels(labels)
        # Should find 2 communities
        assert len(comms) >= 2

    def test_complete_graph(self):
        g = make_complete(5)
        labels = label_propagation(g, seed=42)
        comms = communities_from_labels(labels)
        # Complete graph: all nodes in same community
        assert len(comms) == 1

    def test_disconnected(self):
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(2, 3)
        labels = label_propagation(g, seed=42)
        comms = communities_from_labels(labels)
        assert len(comms) >= 2


# ===========================================================================
# Louvain Community Detection
# ===========================================================================

class TestLouvain:
    def test_empty(self):
        g = Graph()
        assert louvain(g) == []

    def test_single_node(self):
        g = Graph()
        g.add_node(0)
        comms = louvain(g, seed=42)
        assert len(comms) == 1
        assert 0 in comms[0]

    def test_two_cliques(self):
        g = make_two_cliques()
        comms = louvain(g, seed=42)
        # Should detect 2 communities
        assert len(comms) >= 2

    def test_complete(self):
        g = make_complete(5)
        comms = louvain(g, seed=42)
        # Complete graph: one community
        assert len(comms) == 1

    def test_covers_all_nodes(self):
        g = make_karate_like()
        comms = louvain(g, seed=42)
        all_nodes = set()
        for c in comms:
            all_nodes |= c
        assert all_nodes == g.nodes()


# ===========================================================================
# Modularity
# ===========================================================================

class TestModularity:
    def test_single_community(self):
        g = make_complete(5)
        q = modularity(g, [set(range(5))])
        # All edges internal, but also expected = 1 community
        assert q >= 0.0

    def test_two_cliques_good_partition(self):
        g = make_two_cliques()
        q = modularity(g, [{0, 1, 2, 3}, {4, 5, 6, 7}])
        assert q > 0.3  # Good partition should have high modularity

    def test_two_cliques_bad_partition(self):
        g = make_two_cliques()
        q_good = modularity(g, [{0, 1, 2, 3}, {4, 5, 6, 7}])
        q_bad = modularity(g, [{0, 1, 4, 5}, {2, 3, 6, 7}])
        assert q_good > q_bad

    def test_all_singletons(self):
        g = make_complete(4)
        q = modularity(g, [{0}, {1}, {2}, {3}])
        assert q < 0  # Each node alone -> negative modularity


# ===========================================================================
# Connected Components
# ===========================================================================

class TestConnectedComponents:
    def test_single_component(self):
        g = make_complete(4)
        comps = connected_components(g)
        assert len(comps) == 1

    def test_two_components(self):
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(2, 3)
        comps = connected_components(g)
        assert len(comps) == 2

    def test_isolated_nodes(self):
        g = Graph()
        g.add_node(0)
        g.add_node(1)
        comps = connected_components(g)
        assert len(comps) == 2


# ===========================================================================
# Strongly Connected Components
# ===========================================================================

class TestSCC:
    def test_cycle(self):
        g = Graph(directed=True)
        for i in range(4):
            g.add_edge(i, (i + 1) % 4)
        sccs = strongly_connected_components(g)
        assert len(sccs) == 1
        assert len(sccs[0]) == 4

    def test_dag(self):
        g = Graph(directed=True)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        sccs = strongly_connected_components(g)
        assert len(sccs) == 3

    def test_two_sccs(self):
        g = Graph(directed=True)
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        g.add_edge(2, 3)
        g.add_edge(3, 2)
        g.add_edge(1, 2)
        sccs = strongly_connected_components(g)
        assert len(sccs) == 2


# ===========================================================================
# Density
# ===========================================================================

class TestDensity:
    def test_empty(self):
        g = Graph()
        g.add_node(0)
        assert density(g) == 0.0

    def test_complete(self):
        g = make_complete(5)
        assert approx(density(g), 1.0)

    def test_sparse(self):
        g = make_path(5)
        # 4 edges out of 10 possible
        assert approx(density(g), 4 / 10)

    def test_directed(self):
        g = Graph(directed=True)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        # 2 edges out of 6 possible
        assert approx(density(g), 2 / 6)


# ===========================================================================
# Diameter, Radius, Center, Eccentricity
# ===========================================================================

class TestDistanceMetrics:
    def test_path_diameter(self):
        g = make_path(5)
        assert diameter(g) == 4

    def test_complete_diameter(self):
        g = make_complete(5)
        assert diameter(g) == 1

    def test_star_diameter(self):
        g = make_star(5)
        assert diameter(g) == 2

    def test_path_radius(self):
        g = make_path(5)
        assert radius(g) == 2

    def test_path_center(self):
        g = make_path(5)
        c = center(g)
        assert 2 in c

    def test_complete_center(self):
        g = make_complete(4)
        c = center(g)
        assert len(c) == 4  # all nodes are center

    def test_eccentricity_single(self):
        g = Graph()
        g.add_node(0)
        assert eccentricity(g, 0) == 0

    def test_eccentricity_path_endpoint(self):
        g = make_path(5)
        assert eccentricity(g, 0) == 4

    def test_eccentricity_path_middle(self):
        g = make_path(5)
        assert eccentricity(g, 2) == 2

    def test_eccentricity_all(self):
        g = make_path(3)
        ecc = eccentricity(g)
        assert ecc[0] == 2
        assert ecc[1] == 1
        assert ecc[2] == 2


# ===========================================================================
# Bridges and Articulation Points
# ===========================================================================

class TestBridgesAndCutVertices:
    def test_no_bridges_in_triangle(self):
        g = make_triangle()
        assert len(bridges(g)) == 0

    def test_bridge_in_path(self):
        g = make_path(3)  # 0-1-2
        b = bridges(g)
        assert len(b) == 2  # all edges are bridges

    def test_bridge_between_cliques(self):
        g = make_two_cliques()
        b = bridges(g)
        assert len(b) == 1
        bridge = b[0]
        assert (3 in bridge and 4 in bridge)

    def test_no_articulation_in_complete(self):
        g = make_complete(4)
        assert len(articulation_points(g)) == 0

    def test_articulation_in_path(self):
        g = make_path(5)  # 0-1-2-3-4
        ap = articulation_points(g)
        assert 1 in ap
        assert 2 in ap
        assert 3 in ap

    def test_articulation_between_cliques(self):
        g = make_two_cliques()
        ap = articulation_points(g)
        assert 3 in ap
        assert 4 in ap

    def test_star_center_is_articulation(self):
        g = make_star(5)
        ap = articulation_points(g)
        assert 0 in ap


# ===========================================================================
# Katz Centrality
# ===========================================================================

class TestKatzCentrality:
    def test_empty(self):
        g = Graph()
        assert katz_centrality(g) == {}

    def test_complete(self):
        g = make_complete(4)
        kc = katz_centrality(g)
        vals = list(kc.values())
        for v in vals:
            assert approx(v, vals[0], tol=0.01)

    def test_star(self):
        g = make_star(5)
        kc = katz_centrality(g)
        assert kc[0] > kc[1]

    def test_directed_chain(self):
        g = Graph(directed=True)
        g.add_edge('A', 'B')
        g.add_edge('B', 'C')
        kc = katz_centrality(g)
        # C has most paths leading to it
        assert kc['C'] > kc['A']


# ===========================================================================
# Degree Assortativity
# ===========================================================================

class TestAssortativity:
    def test_complete_graph(self):
        g = make_complete(5)
        # All same degree -> undefined but returns 0 (var = 0)
        r = degree_assortativity(g)
        assert r == 0.0

    def test_star(self):
        g = make_star(5)
        # Star is disassortative (high-degree center connects to low-degree leaves)
        r = degree_assortativity(g)
        assert r < 0

    def test_path(self):
        g = make_path(5)
        r = degree_assortativity(g)
        # Path: endpoints (deg 1) connect to deg-2 nodes -> disassortative
        assert r < 0

    def test_empty(self):
        g = Graph()
        assert degree_assortativity(g) == 0.0


# ===========================================================================
# Rich Club Coefficient
# ===========================================================================

class TestRichClub:
    def test_complete(self):
        g = make_complete(5)
        rc = rich_club_coefficient(g, k=0)
        assert approx(rc, 1.0)

    def test_star(self):
        g = make_star(5)
        # k=0: all nodes qualify, edges among them
        rc = rich_club_coefficient(g, k=0)
        assert rc > 0

    def test_high_k(self):
        g = make_path(4)
        # Max degree is 2, so k=3 -> no nodes qualify
        rc = rich_club_coefficient(g, k=3)
        assert rc == 0.0

    def test_all_k(self):
        g = make_karate_like()
        rc = rich_club_coefficient(g)
        assert isinstance(rc, dict)
        assert 0 in rc


# ===========================================================================
# Neighborhood Overlap and Link Prediction
# ===========================================================================

class TestLinkPrediction:
    def test_common_neighbors(self):
        g = make_triangle()
        assert common_neighbors(g, 'A', 'B') == 1  # C is common

    def test_no_common(self):
        g = make_path(3)
        assert common_neighbors(g, 0, 2) == 1  # node 1 is common

    def test_overlap_complete(self):
        g = make_complete(4)
        # A and B share 2 common neighbors out of 2 each (excluding each other)
        overlap = neighborhood_overlap(g, 0, 1)
        assert approx(overlap, 1.0)

    def test_overlap_disjoint(self):
        g = Graph()
        g.add_edge(0, 1)
        g.add_edge(2, 3)
        g.add_edge(1, 2)
        overlap = neighborhood_overlap(g, 0, 3)
        # Neighbors of 0: {1}, Neighbors of 3: {2}. No overlap.
        assert overlap == 0.0

    def test_adamic_adar(self):
        g = make_complete(4)
        aa = adamic_adar(g, 0, 1)
        # Common neighbors: 2 and 3, each with degree 3
        expected = 2.0 / math.log(3)
        assert approx(aa, expected, tol=0.01)

    def test_adamic_adar_no_common(self):
        g = Graph()
        g.add_edge(0, 1)
        g.add_node(2)
        assert adamic_adar(g, 0, 2) == 0.0


# ===========================================================================
# Reciprocity
# ===========================================================================

class TestReciprocity:
    def test_undirected(self):
        g = Graph()
        g.add_edge(0, 1)
        assert reciprocity(g) == 1.0

    def test_fully_reciprocated(self):
        g = Graph(directed=True)
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        assert approx(reciprocity(g), 1.0)

    def test_no_reciprocity(self):
        g = Graph(directed=True)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        assert reciprocity(g) == 0.0

    def test_partial(self):
        g = Graph(directed=True)
        g.add_edge(0, 1)
        g.add_edge(1, 0)
        g.add_edge(0, 2)
        # 2 out of 3 edges are reciprocated
        assert approx(reciprocity(g), 2 / 3)

    def test_empty(self):
        g = Graph(directed=True)
        g.add_node(0)
        assert reciprocity(g) == 0.0


# ===========================================================================
# Integration: Full Network Analysis Pipeline
# ===========================================================================

class TestIntegration:
    def test_karate_analysis(self):
        """Full analysis pipeline on karate-like network."""
        g = make_karate_like()

        # Centrality measures
        pr = pagerank(Graph(directed=True))  # separate directed
        dc = degree_centrality(g)
        cc = closeness_centrality(g)
        bc = betweenness_centrality(g)
        ec = eigenvector_centrality(g)

        # All return dicts covering all nodes
        nodes = g.nodes()
        assert all(n in dc for n in nodes)
        assert all(n in cc for n in nodes)
        assert all(n in bc for n in nodes)
        assert all(n in ec for n in nodes)

    def test_community_pipeline(self):
        """Community detection + modularity evaluation."""
        g = make_two_cliques()
        comms = louvain(g, seed=42)
        q = modularity(g, comms)
        assert q > 0  # Should be positive modularity

    def test_structural_analysis(self):
        """Bridges, articulation points, cores, clustering."""
        g = make_two_cliques()
        b = bridges(g)
        ap = articulation_points(g)
        cores = k_core_decomposition(g)
        cc = clustering_coefficient(g)
        d = density(g)

        assert len(b) == 1
        assert len(ap) == 2
        assert all(v in cores for v in g.nodes())
        assert all(v in cc for v in g.nodes())
        assert 0 < d < 1

    def test_distance_pipeline(self):
        """Eccentricity, diameter, radius, center."""
        g = make_path(7)
        ecc = eccentricity(g)
        d = diameter(g)
        r = radius(g)
        c = center(g)
        assert d == 6
        assert r == 3
        assert 3 in c

    def test_link_prediction_pipeline(self):
        """Link prediction metrics."""
        g = make_karate_like()
        for u in list(g.nodes())[:3]:
            for v in list(g.nodes())[3:6]:
                cn = common_neighbors(g, u, v)
                ov = neighborhood_overlap(g, u, v)
                aa = adamic_adar(g, u, v)
                assert cn >= 0
                assert 0 <= ov <= 1
                assert aa >= 0
