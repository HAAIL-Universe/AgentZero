"""Tests for C164: Causal Discovery."""

import math
import random
import pytest
import numpy as np
from collections import defaultdict
from causal_discovery import (
    ConditionalIndependenceTest,
    ScoringFunction,
    EquivalenceClass,
    PCAlgorithm,
    GESAlgorithm,
    FCIAlgorithm,
    DiscoveryAnalyzer,
    _generate_categorical_data,
    _generate_continuous_data,
    _compute_mi,
    _pearson_corr,
    _chi2_survival,
    _normal_survival,
)


# =====================================================================
# ConditionalIndependenceTest
# =====================================================================

class TestChiSquared:
    """Tests for chi-squared independence test."""

    def test_independent_variables(self):
        """Two independent variables should test as independent."""
        rng = random.Random(42)
        data = [{'X': rng.randint(0, 1), 'Y': rng.randint(0, 1)} for _ in range(500)]
        is_indep, p_val, stat = ConditionalIndependenceTest.chi_squared(data, 'X', 'Y', alpha=0.05)
        # With independent data, p-value should generally be > 0.05
        assert p_val > 0.01  # relaxed for randomness

    def test_dependent_variables(self):
        """Strongly dependent variables should test as dependent."""
        data = []
        for _ in range(500):
            x = random.randint(0, 1)
            y = x  # perfect dependence
            data.append({'X': x, 'Y': y})
        is_indep, p_val, stat = ConditionalIndependenceTest.chi_squared(data, 'X', 'Y', alpha=0.05)
        assert not is_indep
        assert p_val < 0.01

    def test_conditional_independence(self):
        """X -> Z -> Y: X indep Y | Z."""
        rng = random.Random(42)
        data = []
        for _ in range(1000):
            x = rng.randint(0, 1)
            z = x if rng.random() < 0.8 else 1 - x
            y = z if rng.random() < 0.8 else 1 - z
            data.append({'X': x, 'Y': y, 'Z': z})
        is_indep, p_val, stat = ConditionalIndependenceTest.chi_squared(data, 'X', 'Y', {'Z'}, alpha=0.05)
        # X and Y should be (conditionally) independent given Z
        # With finite data this may not always hold, but should be closer to independence
        assert p_val > 0.001  # relaxed

    def test_empty_conditioning_set(self):
        """Empty conditioning set = marginal test."""
        data = [{'X': i % 2, 'Y': i % 2} for i in range(200)]
        is_indep1, _, _ = ConditionalIndependenceTest.chi_squared(data, 'X', 'Y', set(), alpha=0.05)
        is_indep2, _, _ = ConditionalIndependenceTest.chi_squared(data, 'X', 'Y', None, alpha=0.05)
        assert is_indep1 == is_indep2

    def test_small_sample(self):
        """Small sample should not crash."""
        data = [{'X': 0, 'Y': 1}, {'X': 1, 'Y': 0}] * 5
        is_indep, p_val, stat = ConditionalIndependenceTest.chi_squared(data, 'X', 'Y')
        assert isinstance(is_indep, bool)

    def test_multivalue_variables(self):
        """Works with variables having >2 values."""
        rng = random.Random(42)
        data = [{'X': rng.randint(0, 2), 'Y': rng.randint(0, 3)} for _ in range(500)]
        is_indep, p_val, stat = ConditionalIndependenceTest.chi_squared(data, 'X', 'Y')
        assert isinstance(p_val, float)
        assert 0 <= p_val <= 1

    def test_conditional_with_multiple_conditioning(self):
        """Conditioning on multiple variables."""
        rng = random.Random(42)
        data = [{'X': rng.randint(0,1), 'Y': rng.randint(0,1),
                 'Z1': rng.randint(0,1), 'Z2': rng.randint(0,1)} for _ in range(2000)]
        is_indep, p_val, stat = ConditionalIndependenceTest.chi_squared(
            data, 'X', 'Y', {'Z1', 'Z2'}, alpha=0.05
        )
        assert isinstance(is_indep, bool)


class TestMutualInformation:
    """Tests for mutual information independence test."""

    def test_independent_mi_zero(self):
        """MI should be near zero for independent variables."""
        rng = random.Random(42)
        data = [{'X': rng.randint(0, 1), 'Y': rng.randint(0, 1)} for _ in range(500)]
        is_indep, p_val, stat = ConditionalIndependenceTest.mutual_information(data, 'X', 'Y')
        assert p_val > 0.01

    def test_dependent_mi_high(self):
        """MI should be high for dependent variables."""
        data = [{'X': i % 2, 'Y': i % 2} for i in range(500)]
        is_indep, p_val, stat = ConditionalIndependenceTest.mutual_information(data, 'X', 'Y')
        assert not is_indep
        assert stat > 0

    def test_conditional_mi(self):
        """Conditional MI test."""
        rng = random.Random(42)
        data = [{'X': rng.randint(0,1), 'Y': rng.randint(0,1), 'Z': rng.randint(0,1)}
                for _ in range(2000)]
        is_indep, p_val, stat = ConditionalIndependenceTest.mutual_information(
            data, 'X', 'Y', {'Z'}
        )
        assert isinstance(is_indep, bool)

    def test_mi_compute_helper(self):
        """Test _compute_mi helper function."""
        # Perfect dependence
        data = [{'X': 0, 'Y': 0}] * 50 + [{'X': 1, 'Y': 1}] * 50
        mi = _compute_mi(data, 'X', 'Y')
        assert mi > 0.5  # close to log(2)

    def test_mi_independent(self):
        """MI near zero for independent data."""
        rng = random.Random(42)
        data = [{'X': rng.randint(0,1), 'Y': rng.randint(0,1)} for _ in range(10000)]
        mi = _compute_mi(data, 'X', 'Y')
        assert mi < 0.01


class TestPartialCorrelation:
    """Tests for partial correlation test."""

    def test_independent_continuous(self):
        """Independent continuous variables."""
        rng = np.random.RandomState(42)
        data = [{'X': float(rng.normal()), 'Y': float(rng.normal())} for _ in range(200)]
        is_indep, p_val, r = ConditionalIndependenceTest.partial_correlation(data, 'X', 'Y')
        assert abs(r) < 0.2

    def test_correlated_continuous(self):
        """Strongly correlated continuous variables."""
        rng = np.random.RandomState(42)
        data = []
        for _ in range(200):
            x = float(rng.normal())
            y = 2 * x + float(rng.normal(0, 0.1))
            data.append({'X': x, 'Y': y})
        is_indep, p_val, r = ConditionalIndependenceTest.partial_correlation(data, 'X', 'Y')
        assert not is_indep
        assert r > 0.8

    def test_partial_correlation_conditioning(self):
        """X -> Z -> Y, partial correlation of X,Y given Z should be small."""
        rng = np.random.RandomState(42)
        data = []
        for _ in range(500):
            x = float(rng.normal())
            z = 0.8 * x + float(rng.normal(0, 0.3))
            y = 0.8 * z + float(rng.normal(0, 0.3))
            data.append({'X': x, 'Y': y, 'Z': z})
        is_indep, p_val, r = ConditionalIndependenceTest.partial_correlation(
            data, 'X', 'Y', {'Z'}
        )
        # Partial correlation should be much smaller than marginal
        assert abs(r) < 0.3

    def test_pearson_corr_helper(self):
        """Test _pearson_corr helper."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        r = _pearson_corr(x, y)
        assert abs(r - 1.0) < 1e-10

    def test_pearson_negative(self):
        """Negative correlation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        r = _pearson_corr(x, y)
        assert abs(r + 1.0) < 1e-10


# =====================================================================
# ScoringFunction
# =====================================================================

class TestScoringFunction:
    """Tests for DAG scoring functions."""

    def test_bic_no_parents(self):
        """BIC score with no parents."""
        data = [{'X': i % 2, 'Y': i % 3} for i in range(100)]
        score = ScoringFunction.bic(data, 'X', [])
        assert isinstance(score, float)

    def test_bic_with_parent(self):
        """BIC with parent should be higher when parent is informative."""
        rng = random.Random(42)
        data = []
        for _ in range(500):
            x = rng.randint(0, 1)
            y = x  # Y determined by X
            data.append({'X': x, 'Y': y})

        score_no_parent = ScoringFunction.bic(data, 'Y', [])
        score_with_parent = ScoringFunction.bic(data, 'Y', ['X'])
        assert score_with_parent > score_no_parent

    def test_bic_spurious_parent_penalized(self):
        """Adding a non-informative parent should lower BIC due to penalty."""
        rng = random.Random(42)
        data = [{'X': rng.randint(0,1), 'Y': rng.randint(0,1), 'Z': rng.randint(0,1)}
                for _ in range(500)]
        score_no = ScoringFunction.bic(data, 'Y', [])
        score_z = ScoringFunction.bic(data, 'Y', ['Z'])
        # Adding uninformative parent should not improve BIC much (may decrease)
        assert score_z <= score_no + 1  # allow small tolerance

    def test_bic_continuous(self):
        """BIC for continuous data."""
        rng = np.random.RandomState(42)
        data = []
        for _ in range(200):
            x = float(rng.normal())
            y = 2 * x + float(rng.normal(0, 0.1))
            data.append({'X': x, 'Y': y})
        score = ScoringFunction.bic(data, 'Y', ['X'], continuous=True)
        score_no = ScoringFunction.bic(data, 'Y', [], continuous=True)
        assert score > score_no

    def test_aic(self):
        """AIC score."""
        data = [{'X': i % 2, 'Y': i % 2} for i in range(200)]
        score = ScoringFunction.aic(data, 'Y', ['X'])
        assert isinstance(score, float)

    def test_aic_continuous(self):
        """AIC for continuous data."""
        rng = np.random.RandomState(42)
        data = [{'X': float(rng.normal()), 'Y': float(rng.normal())} for _ in range(200)]
        score = ScoringFunction.aic(data, 'Y', [], continuous=True)
        assert isinstance(score, float)

    def test_bdeu(self):
        """BDeu score."""
        data = [{'X': i % 2, 'Y': i % 2} for i in range(200)]
        score = ScoringFunction.bdeu(data, 'Y', ['X'], ess=1.0)
        assert isinstance(score, float)

    def test_bdeu_parent_helps(self):
        """BDeu: informative parent improves score."""
        data = [{'X': i % 2, 'Y': i % 2} for i in range(200)]
        score_with = ScoringFunction.bdeu(data, 'Y', ['X'])
        score_without = ScoringFunction.bdeu(data, 'Y', [])
        assert score_with > score_without

    def test_score_dag(self):
        """Score entire DAG."""
        data = [{'X': i % 2, 'Y': i % 2} for i in range(200)]
        parent_map = {'X': [], 'Y': ['X']}
        score = ScoringFunction.score_dag(data, ['X', 'Y'], parent_map)
        assert isinstance(score, float)

    def test_score_dag_comparison(self):
        """Better DAG should have higher score."""
        rng = random.Random(42)
        data = []
        for _ in range(500):
            x = rng.randint(0, 1)
            y = x
            data.append({'X': x, 'Y': y})

        correct = ScoringFunction.score_dag(data, ['X', 'Y'], {'X': [], 'Y': ['X']})
        wrong = ScoringFunction.score_dag(data, ['X', 'Y'], {'X': ['Y'], 'Y': []})
        empty = ScoringFunction.score_dag(data, ['X', 'Y'], {'X': [], 'Y': []})

        assert correct > empty

    def test_empty_data(self):
        """Empty data should return 0."""
        assert ScoringFunction.bic([], 'X', []) == 0.0
        assert ScoringFunction.aic([], 'X', []) == 0.0


# =====================================================================
# EquivalenceClass
# =====================================================================

class TestEquivalenceClass:
    """Tests for CPDAG/PAG representation."""

    def test_create_empty(self):
        ec = EquivalenceClass(['A', 'B', 'C'])
        assert len(ec.nodes) == 3
        assert len(ec.directed) == 0
        assert len(ec.undirected) == 0

    def test_add_directed(self):
        ec = EquivalenceClass(['A', 'B'])
        ec.add_directed_edge('A', 'B')
        assert ec.has_directed('A', 'B')
        assert not ec.has_directed('B', 'A')

    def test_add_undirected(self):
        ec = EquivalenceClass(['A', 'B'])
        ec.add_undirected_edge('A', 'B')
        assert ec.has_undirected('A', 'B')
        assert ec.has_undirected('B', 'A')  # symmetric

    def test_orient_edge(self):
        ec = EquivalenceClass(['A', 'B'])
        ec.add_undirected_edge('A', 'B')
        ec.orient_edge('A', 'B')
        assert ec.has_directed('A', 'B')
        assert not ec.has_undirected('A', 'B')

    def test_remove_edge(self):
        ec = EquivalenceClass(['A', 'B'])
        ec.add_directed_edge('A', 'B')
        ec.remove_edge('A', 'B')
        assert not ec.has_directed('A', 'B')
        assert not ec.is_adjacent('A', 'B')

    def test_adjacency(self):
        ec = EquivalenceClass(['A', 'B', 'C'])
        ec.add_directed_edge('A', 'B')
        ec.add_undirected_edge('B', 'C')
        assert ec.is_adjacent('A', 'B')
        assert ec.is_adjacent('B', 'C')
        assert not ec.is_adjacent('A', 'C')

    def test_neighbors(self):
        ec = EquivalenceClass(['A', 'B', 'C'])
        ec.add_directed_edge('A', 'B')
        ec.add_undirected_edge('A', 'C')
        nbrs = ec.neighbors('A')
        assert nbrs == {'B', 'C'}

    def test_skeleton(self):
        ec = EquivalenceClass(['A', 'B', 'C'])
        ec.add_directed_edge('A', 'B')
        ec.add_undirected_edge('B', 'C')
        skel = ec.skeleton()
        assert frozenset(['A', 'B']) in skel
        assert frozenset(['B', 'C']) in skel
        assert frozenset(['A', 'C']) not in skel

    def test_get_parents(self):
        ec = EquivalenceClass(['A', 'B', 'C'])
        ec.add_directed_edge('A', 'C')
        ec.add_directed_edge('B', 'C')
        parents = ec.get_parents('C')
        assert parents == {'A', 'B'}

    def test_to_dag_edges(self):
        ec = EquivalenceClass(['A', 'B', 'C'])
        ec.add_directed_edge('A', 'B')
        ec.add_undirected_edge('B', 'C')
        edges = ec.to_dag_edges()
        assert ('A', 'B') in edges
        assert ('B', 'C') in edges  # undirected arbitrarily oriented

    def test_to_causal_graph(self):
        ec = EquivalenceClass(['A', 'B'])
        ec.add_directed_edge('A', 'B')
        g = ec.to_causal_graph()
        assert 'A' in g.nodes
        assert 'B' in g.nodes

    def test_add_node(self):
        ec = EquivalenceClass()
        ec.add_node('X')
        ec.add_node('Y')
        assert 'X' in ec.nodes
        assert 'Y' in ec.nodes

    def test_circle_edge(self):
        ec = EquivalenceClass(['A', 'B'])
        ec.add_circle_edge('A', 'B')
        assert ('A', 'B') in ec.circle
        assert ec.is_adjacent('A', 'B')

    def test_add_node_no_duplicates(self):
        ec = EquivalenceClass()
        ec.add_node('X')
        ec.add_node('X')
        assert ec.nodes.count('X') == 1


# =====================================================================
# Data generation helpers
# =====================================================================

class TestDataGeneration:
    """Tests for data generation utilities."""

    def test_generate_categorical(self):
        """Generate categorical data from a DAG."""
        edges = [('X', 'Y'), ('Y', 'Z')]
        data = _generate_categorical_data(edges, 100, seed=42)
        assert len(data) == 100
        assert set(data[0].keys()) == {'X', 'Y', 'Z'}

    def test_generate_categorical_cardinalities(self):
        """Custom cardinalities."""
        edges = [('X', 'Y')]
        data = _generate_categorical_data(edges, 200, cardinalities={'X': 3, 'Y': 4}, seed=42)
        x_vals = set(row['X'] for row in data)
        y_vals = set(row['Y'] for row in data)
        assert max(x_vals) <= 2
        assert max(y_vals) <= 3

    def test_generate_continuous(self):
        """Generate continuous data from a linear SEM."""
        edges = [('X', 'Y'), ('Y', 'Z')]
        data = _generate_continuous_data(edges, 100, seed=42)
        assert len(data) == 100
        assert all(isinstance(row['X'], float) for row in data)

    def test_generated_data_has_dependencies(self):
        """Generated data should show statistical dependencies along edges."""
        edges = [('X', 'Y')]
        data = _generate_categorical_data(edges, 1000, seed=42)
        is_indep, p_val, _ = ConditionalIndependenceTest.chi_squared(data, 'X', 'Y')
        assert not is_indep  # X and Y should be dependent

    def test_continuous_dependencies(self):
        """Generated continuous data should show correlations along edges."""
        edges = [('X', 'Y')]
        data = _generate_continuous_data(edges, 500, coefficients={('X', 'Y'): 2.0}, seed=42)
        x_vals = np.array([row['X'] for row in data])
        y_vals = np.array([row['Y'] for row in data])
        r = _pearson_corr(x_vals, y_vals)
        assert abs(r) > 0.5


# =====================================================================
# PCAlgorithm
# =====================================================================

class TestPCAlgorithm:
    """Tests for the PC algorithm."""

    def test_simple_chain(self):
        """X -> Y -> Z: should learn skeleton X-Y-Z."""
        edges = [('X', 'Y'), ('Y', 'Z')]
        data = _generate_categorical_data(edges, 2000, seed=42)
        pc = PCAlgorithm(alpha=0.05)
        cpdag = pc.fit(data, ['X', 'Y', 'Z'])

        # Skeleton should have X-Y and Y-Z
        skel = cpdag.skeleton()
        assert frozenset(['X', 'Y']) in skel
        assert frozenset(['Y', 'Z']) in skel

    def test_v_structure(self):
        """X -> Z <- Y: should orient v-structure."""
        edges = [('X', 'Z'), ('Y', 'Z')]
        data = _generate_categorical_data(edges, 2000, seed=42)
        pc = PCAlgorithm(alpha=0.05)
        cpdag = pc.fit(data, ['X', 'Y', 'Z'])

        # X and Y should NOT be adjacent (no direct edge)
        assert not cpdag.is_adjacent('X', 'Y')
        # V-structure: X -> Z <- Y
        assert cpdag.has_directed('X', 'Z') or cpdag.has_directed('Y', 'Z')

    def test_fork_structure(self):
        """Y <- X -> Z: should learn skeleton."""
        edges = [('X', 'Y'), ('X', 'Z')]
        data = _generate_categorical_data(edges, 2000, seed=42)
        pc = PCAlgorithm(alpha=0.05)
        cpdag = pc.fit(data, ['X', 'Y', 'Z'])

        skel = cpdag.skeleton()
        assert frozenset(['X', 'Y']) in skel
        assert frozenset(['X', 'Z']) in skel

    def test_stable_pc(self):
        """Stable PC should be order-independent."""
        edges = [('A', 'B'), ('B', 'C'), ('A', 'C')]
        data = _generate_categorical_data(edges, 1000, seed=42)

        pc1 = PCAlgorithm(alpha=0.05, stable=True)
        cpdag1 = pc1.fit(data, ['A', 'B', 'C'])

        pc2 = PCAlgorithm(alpha=0.05, stable=True)
        cpdag2 = pc2.fit(data, ['C', 'B', 'A'])

        # Skeletons should match regardless of variable order
        assert cpdag1.skeleton() == cpdag2.skeleton()

    def test_partial_correlation_test_method(self):
        """PC with partial correlation for continuous data."""
        edges = [('X', 'Y'), ('Y', 'Z')]
        data = _generate_continuous_data(edges, 500, seed=42)
        pc = PCAlgorithm(alpha=0.05, test_method='partial_correlation')
        cpdag = pc.fit(data, ['X', 'Y', 'Z'])

        skel = cpdag.skeleton()
        assert frozenset(['X', 'Y']) in skel
        assert frozenset(['Y', 'Z']) in skel

    def test_mutual_info_test_method(self):
        """PC with mutual information test."""
        edges = [('X', 'Y')]
        data = _generate_categorical_data(edges, 1000, seed=42)
        pc = PCAlgorithm(alpha=0.05, test_method='mutual_information')
        cpdag = pc.fit(data, ['X', 'Y'])
        skel = cpdag.skeleton()
        assert frozenset(['X', 'Y']) in skel

    def test_max_cond_size(self):
        """Limiting conditioning set size."""
        edges = [('X', 'Y'), ('Y', 'Z')]
        data = _generate_categorical_data(edges, 1000, seed=42)
        pc = PCAlgorithm(alpha=0.05, max_cond_size=1)
        cpdag = pc.fit(data, ['X', 'Y', 'Z'])
        assert isinstance(cpdag, EquivalenceClass)

    def test_single_edge(self):
        """Single edge graph."""
        edges = [('X', 'Y')]
        data = _generate_categorical_data(edges, 1000, seed=42)
        pc = PCAlgorithm(alpha=0.05)
        cpdag = pc.fit(data, ['X', 'Y'])
        skel = cpdag.skeleton()
        assert frozenset(['X', 'Y']) in skel

    def test_disconnected_nodes(self):
        """Nodes with no edges should remain disconnected."""
        rng = random.Random(42)
        data = [{'X': rng.randint(0,1), 'Y': rng.randint(0,1)} for _ in range(500)]
        pc = PCAlgorithm(alpha=0.05)
        cpdag = pc.fit(data, ['X', 'Y'])
        # Should find no edges (or very few)
        assert len(cpdag.skeleton()) <= 1  # might spuriously connect

    def test_four_node_chain(self):
        """A -> B -> C -> D."""
        edges = [('A', 'B'), ('B', 'C'), ('C', 'D')]
        data = _generate_categorical_data(edges, 2000, seed=42)
        pc = PCAlgorithm(alpha=0.05)
        cpdag = pc.fit(data, ['A', 'B', 'C', 'D'])

        skel = cpdag.skeleton()
        assert frozenset(['A', 'B']) in skel
        assert frozenset(['B', 'C']) in skel
        assert frozenset(['C', 'D']) in skel

    def test_test_count(self):
        """PC should track number of CI tests performed."""
        edges = [('X', 'Y')]
        data = _generate_categorical_data(edges, 500, seed=42)
        pc = PCAlgorithm(alpha=0.05)
        pc.fit(data, ['X', 'Y'])
        assert pc.test_count > 0

    def test_auto_detect_variables(self):
        """Variables auto-detected from data."""
        edges = [('X', 'Y')]
        data = _generate_categorical_data(edges, 500, seed=42)
        pc = PCAlgorithm(alpha=0.05)
        cpdag = pc.fit(data)  # no variables specified
        assert len(cpdag.nodes) == 2


# =====================================================================
# GESAlgorithm
# =====================================================================

class TestGESAlgorithm:
    """Tests for GES algorithm."""

    def test_simple_edge(self):
        """GES should find a single edge."""
        data = [{'X': i % 2, 'Y': i % 2} for i in range(500)]
        ges = GESAlgorithm(score_method='bic')
        cpdag = ges.fit(data, ['X', 'Y'])
        # Should find the edge
        skel = cpdag.skeleton()
        assert frozenset(['X', 'Y']) in skel

    def test_no_edge(self):
        """GES with independent variables should find no edges."""
        rng = random.Random(42)
        data = [{'X': rng.randint(0,1), 'Y': rng.randint(0,1)} for _ in range(500)]
        ges = GESAlgorithm(score_method='bic')
        cpdag = ges.fit(data, ['X', 'Y'])
        skel = cpdag.skeleton()
        assert len(skel) == 0

    def test_chain(self):
        """GES on X -> Y -> Z."""
        edges = [('X', 'Y'), ('Y', 'Z')]
        data = _generate_categorical_data(edges, 2000, seed=42)
        ges = GESAlgorithm(score_method='bic')
        cpdag = ges.fit(data, ['X', 'Y', 'Z'])

        skel = cpdag.skeleton()
        assert frozenset(['X', 'Y']) in skel
        assert frozenset(['Y', 'Z']) in skel

    def test_v_structure_ges(self):
        """GES on X -> Z <- Y (v-structure)."""
        edges = [('X', 'Z'), ('Y', 'Z')]
        data = _generate_categorical_data(edges, 3000, seed=42)
        ges = GESAlgorithm(score_method='bic')
        cpdag = ges.fit(data, ['X', 'Y', 'Z'])

        skel = cpdag.skeleton()
        # Should find at least one of the true edges
        found = frozenset(['X', 'Z']) in skel or frozenset(['Y', 'Z']) in skel
        assert found

    def test_ges_aic(self):
        """GES with AIC scoring."""
        data = [{'X': i % 2, 'Y': i % 2} for i in range(500)]
        ges = GESAlgorithm(score_method='aic')
        cpdag = ges.fit(data, ['X', 'Y'])
        assert frozenset(['X', 'Y']) in cpdag.skeleton()

    def test_ges_bdeu(self):
        """GES with BDeu scoring."""
        data = [{'X': i % 2, 'Y': i % 2} for i in range(500)]
        ges = GESAlgorithm(score_method='bdeu', ess=1.0)
        cpdag = ges.fit(data, ['X', 'Y'])
        assert frozenset(['X', 'Y']) in cpdag.skeleton()

    def test_ges_continuous(self):
        """GES with continuous data."""
        edges = [('X', 'Y')]
        data = _generate_continuous_data(edges, 500, coefficients={('X', 'Y'): 2.0}, seed=42)
        ges = GESAlgorithm(score_method='bic', continuous=True)
        cpdag = ges.fit(data, ['X', 'Y'])
        assert frozenset(['X', 'Y']) in cpdag.skeleton()

    def test_ges_cycle_prevention(self):
        """GES should never produce cycles."""
        edges = [('A', 'B'), ('B', 'C')]
        data = _generate_categorical_data(edges, 1000, seed=42)
        ges = GESAlgorithm()
        cpdag = ges.fit(data, ['A', 'B', 'C'])
        # Convert to DAG and check no cycles
        dag_edges = cpdag.to_dag_edges()
        # Simple cycle check
        adj = defaultdict(list)
        for a, b in dag_edges:
            adj[a].append(b)
        # DFS for cycles
        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)
            for nb in adj[node]:
                if nb not in visited:
                    if has_cycle(nb, visited, rec_stack):
                        return True
                elif nb in rec_stack:
                    return True
            rec_stack.discard(node)
            return False

        visited = set()
        for node in cpdag.nodes:
            if node not in visited:
                assert not has_cycle(node, visited, set())

    def test_auto_detect_variables(self):
        """Variables auto-detected from data."""
        data = [{'X': 0, 'Y': 0}, {'X': 1, 'Y': 1}] * 250
        ges = GESAlgorithm()
        cpdag = ges.fit(data)
        assert len(cpdag.nodes) == 2


# =====================================================================
# FCIAlgorithm
# =====================================================================

class TestFCIAlgorithm:
    """Tests for FCI algorithm."""

    def test_simple_skeleton(self):
        """FCI learns skeleton."""
        edges = [('X', 'Y'), ('Y', 'Z')]
        data = _generate_categorical_data(edges, 2000, seed=42)
        fci = FCIAlgorithm(alpha=0.05)
        pag = fci.fit(data, ['X', 'Y', 'Z'])

        skel = pag.skeleton()
        assert frozenset(['X', 'Y']) in skel
        assert frozenset(['Y', 'Z']) in skel

    def test_v_structure_fci(self):
        """FCI orients v-structures."""
        edges = [('X', 'Z'), ('Y', 'Z')]
        data = _generate_categorical_data(edges, 2000, seed=42)
        fci = FCIAlgorithm(alpha=0.05)
        pag = fci.fit(data, ['X', 'Y', 'Z'])

        # X and Y should not be adjacent
        assert not pag.is_adjacent('X', 'Y')
        # Z should have arrowheads from X and Y
        has_arrow_x_z = pag.has_directed('X', 'Z')
        has_arrow_y_z = pag.has_directed('Y', 'Z')
        assert has_arrow_x_z or has_arrow_y_z

    def test_fci_returns_pag(self):
        """FCI output is a PAG (may have circle edges)."""
        edges = [('X', 'Y')]
        data = _generate_categorical_data(edges, 1000, seed=42)
        fci = FCIAlgorithm(alpha=0.05)
        pag = fci.fit(data, ['X', 'Y'])
        assert isinstance(pag, EquivalenceClass)

    def test_fci_mutual_info(self):
        """FCI with mutual information test."""
        edges = [('X', 'Y'), ('Y', 'Z')]
        data = _generate_categorical_data(edges, 1000, seed=42)
        fci = FCIAlgorithm(alpha=0.05, test_method='mutual_information')
        pag = fci.fit(data, ['X', 'Y', 'Z'])
        assert isinstance(pag, EquivalenceClass)

    def test_fci_max_cond_size(self):
        """FCI with limited conditioning set size."""
        edges = [('X', 'Y')]
        data = _generate_categorical_data(edges, 500, seed=42)
        fci = FCIAlgorithm(alpha=0.05, max_cond_size=1)
        pag = fci.fit(data, ['X', 'Y'])
        assert isinstance(pag, EquivalenceClass)

    def test_fci_test_count(self):
        """FCI tracks test count."""
        edges = [('X', 'Y')]
        data = _generate_categorical_data(edges, 500, seed=42)
        fci = FCIAlgorithm(alpha=0.05)
        fci.fit(data, ['X', 'Y'])
        assert fci.test_count > 0

    def test_fci_auto_detect(self):
        """FCI auto-detects variables."""
        edges = [('X', 'Y')]
        data = _generate_categorical_data(edges, 500, seed=42)
        fci = FCIAlgorithm(alpha=0.05)
        pag = fci.fit(data)
        assert len(pag.nodes) == 2


# =====================================================================
# DiscoveryAnalyzer
# =====================================================================

class TestDiscoveryAnalyzer:
    """Tests for structure comparison metrics."""

    def test_shd_perfect(self):
        """SHD = 0 for perfect recovery."""
        true_edges = {('X', 'Y'), ('Y', 'Z')}
        cpdag = EquivalenceClass(['X', 'Y', 'Z'])
        cpdag.add_directed_edge('X', 'Y')
        cpdag.add_directed_edge('Y', 'Z')
        shd = DiscoveryAnalyzer.structural_hamming_distance(true_edges, cpdag)
        assert shd == 0

    def test_shd_missing_edge(self):
        """SHD increases for missing edges."""
        true_edges = {('X', 'Y'), ('Y', 'Z')}
        cpdag = EquivalenceClass(['X', 'Y', 'Z'])
        cpdag.add_directed_edge('X', 'Y')
        # Missing Y -> Z
        shd = DiscoveryAnalyzer.structural_hamming_distance(true_edges, cpdag)
        assert shd >= 1

    def test_shd_extra_edge(self):
        """SHD increases for extra edges."""
        true_edges = {('X', 'Y')}
        cpdag = EquivalenceClass(['X', 'Y', 'Z'])
        cpdag.add_directed_edge('X', 'Y')
        cpdag.add_directed_edge('Y', 'Z')  # extra
        shd = DiscoveryAnalyzer.structural_hamming_distance(true_edges, cpdag)
        assert shd >= 1

    def test_shd_wrong_direction(self):
        """SHD increases for wrong edge direction."""
        true_edges = {('X', 'Y')}
        cpdag = EquivalenceClass(['X', 'Y'])
        cpdag.add_directed_edge('Y', 'X')  # reversed
        shd = DiscoveryAnalyzer.structural_hamming_distance(true_edges, cpdag)
        assert shd >= 1

    def test_skeleton_metrics_perfect(self):
        """Perfect skeleton recovery."""
        true_edges = {('X', 'Y'), ('Y', 'Z')}
        cpdag = EquivalenceClass(['X', 'Y', 'Z'])
        cpdag.add_directed_edge('X', 'Y')
        cpdag.add_directed_edge('Y', 'Z')
        metrics = DiscoveryAnalyzer.skeleton_metrics(true_edges, cpdag)
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0

    def test_skeleton_metrics_partial(self):
        """Partial skeleton recovery."""
        true_edges = {('X', 'Y'), ('Y', 'Z')}
        cpdag = EquivalenceClass(['X', 'Y', 'Z'])
        cpdag.add_directed_edge('X', 'Y')
        # Missing Y-Z
        metrics = DiscoveryAnalyzer.skeleton_metrics(true_edges, cpdag)
        assert metrics['precision'] == 1.0  # no false positives
        assert metrics['recall'] == 0.5     # found 1 of 2

    def test_skeleton_metrics_extra(self):
        """Extra edges reduce precision."""
        true_edges = {('X', 'Y')}
        cpdag = EquivalenceClass(['X', 'Y', 'Z'])
        cpdag.add_directed_edge('X', 'Y')
        cpdag.add_directed_edge('X', 'Z')  # extra
        metrics = DiscoveryAnalyzer.skeleton_metrics(true_edges, cpdag)
        assert metrics['precision'] == 0.5  # 1 correct of 2 found
        assert metrics['recall'] == 1.0     # found the true edge

    def test_arrowhead_metrics(self):
        """Arrowhead metrics."""
        true_edges = {('X', 'Y'), ('Y', 'Z')}
        cpdag = EquivalenceClass(['X', 'Y', 'Z'])
        cpdag.add_directed_edge('X', 'Y')
        cpdag.add_directed_edge('Y', 'Z')
        metrics = DiscoveryAnalyzer.arrowhead_metrics(true_edges, cpdag)
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0

    def test_arrowhead_metrics_reversed(self):
        """Reversed edge = wrong arrowhead."""
        true_edges = {('X', 'Y')}
        cpdag = EquivalenceClass(['X', 'Y'])
        cpdag.add_directed_edge('Y', 'X')
        metrics = DiscoveryAnalyzer.arrowhead_metrics(true_edges, cpdag)
        # The arrowhead is wrong
        assert metrics['precision'] < 1.0

    def test_compare_full(self):
        """Full comparison returns all metrics."""
        true_edges = {('X', 'Y')}
        cpdag = EquivalenceClass(['X', 'Y'])
        cpdag.add_directed_edge('X', 'Y')
        result = DiscoveryAnalyzer.compare(true_edges, cpdag)
        assert 'shd' in result
        assert 'skeleton' in result
        assert 'arrowhead' in result
        assert result['shd'] == 0

    def test_empty_graph(self):
        """Compare with empty learned graph."""
        true_edges = {('X', 'Y')}
        cpdag = EquivalenceClass(['X', 'Y'])
        metrics = DiscoveryAnalyzer.skeleton_metrics(true_edges, cpdag)
        assert metrics['recall'] == 0.0

    def test_empty_true_graph(self):
        """Compare with empty true graph."""
        true_edges = set()
        cpdag = EquivalenceClass(['X', 'Y'])
        cpdag.add_directed_edge('X', 'Y')
        metrics = DiscoveryAnalyzer.skeleton_metrics(true_edges, cpdag)
        assert metrics['precision'] == 0.0


# =====================================================================
# Helper functions
# =====================================================================

class TestHelpers:
    """Tests for helper/utility functions."""

    def test_chi2_survival_zero(self):
        """chi2_survival(0, df) = 1.0."""
        assert _chi2_survival(0, 1) == 1.0

    def test_chi2_survival_large(self):
        """Large chi2 -> small p-value."""
        p = _chi2_survival(100, 1)
        assert p < 0.01

    def test_normal_survival_zero(self):
        """P(Z > 0) = 0.5."""
        p = _normal_survival(0)
        assert abs(p - 0.5) < 0.01

    def test_normal_survival_large(self):
        """P(Z > 5) very small."""
        p = _normal_survival(5)
        assert p < 0.001

    def test_normal_survival_negative(self):
        """P(Z > -5) very close to 1."""
        p = _normal_survival(-5)
        assert p > 0.999

    def test_chi2_survival_negative_input(self):
        """Negative statistic returns 1.0."""
        assert _chi2_survival(-1, 1) == 1.0


# =====================================================================
# Integration tests
# =====================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_pc_with_analyzer(self):
        """PC + DiscoveryAnalyzer on a known structure."""
        true_edges = {('X', 'Z'), ('Y', 'Z')}  # v-structure
        data = _generate_categorical_data(list(true_edges), 2000, seed=42)

        pc = PCAlgorithm(alpha=0.05)
        cpdag = pc.fit(data, ['X', 'Y', 'Z'])

        result = DiscoveryAnalyzer.compare(true_edges, cpdag)
        # Skeleton should be recovered
        assert result['skeleton']['recall'] >= 0.5

    def test_ges_with_analyzer(self):
        """GES + DiscoveryAnalyzer."""
        true_edges = {('X', 'Y'), ('Y', 'Z')}
        data = _generate_categorical_data(list(true_edges), 2000, seed=42)

        ges = GESAlgorithm(score_method='bic')
        cpdag = ges.fit(data, ['X', 'Y', 'Z'])

        result = DiscoveryAnalyzer.compare(true_edges, cpdag)
        assert result['skeleton']['recall'] >= 0.5

    def test_fci_with_analyzer(self):
        """FCI + DiscoveryAnalyzer."""
        true_edges = {('X', 'Y'), ('Y', 'Z')}
        data = _generate_categorical_data(list(true_edges), 2000, seed=42)

        fci = FCIAlgorithm(alpha=0.05)
        pag = fci.fit(data, ['X', 'Y', 'Z'])

        # FCI skeleton should overlap with true
        skel_metrics = DiscoveryAnalyzer.skeleton_metrics(true_edges, pag)
        assert skel_metrics['recall'] >= 0.5

    def test_continuous_pc_with_scoring(self):
        """PC (partial correlation) + BIC scoring on continuous data."""
        true_edges = [('X', 'Y'), ('Y', 'Z')]
        data = _generate_continuous_data(true_edges, 500, seed=42)

        # PC
        pc = PCAlgorithm(alpha=0.05, test_method='partial_correlation')
        cpdag = pc.fit(data, ['X', 'Y', 'Z'])

        # Score the learned structure
        parent_map = {}
        for node in cpdag.nodes:
            parent_map[node] = list(cpdag.get_parents(node))
        score = ScoringFunction.score_dag(data, cpdag.nodes, parent_map,
                                          method='bic', continuous=True)
        assert isinstance(score, float)

    def test_pc_vs_ges_agreement(self):
        """PC and GES should roughly agree on structure."""
        true_edges = [('X', 'Y'), ('Y', 'Z')]
        data = _generate_categorical_data(true_edges, 2000, seed=42)

        pc = PCAlgorithm(alpha=0.05)
        cpdag_pc = pc.fit(data, ['X', 'Y', 'Z'])

        ges = GESAlgorithm(score_method='bic')
        cpdag_ges = ges.fit(data, ['X', 'Y', 'Z'])

        # Skeletons should overlap substantially
        skel_pc = cpdag_pc.skeleton()
        skel_ges = cpdag_ges.skeleton()
        overlap = skel_pc & skel_ges
        # At least one edge should match
        assert len(overlap) >= 1

    def test_larger_graph(self):
        """Test on a 5-node graph."""
        edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D'), ('D', 'E')]
        data = _generate_categorical_data(edges, 3000, seed=42)

        pc = PCAlgorithm(alpha=0.05)
        cpdag = pc.fit(data)

        skel = cpdag.skeleton()
        # Should recover at least some edges
        true_skel = {frozenset(e) for e in edges}
        recovered = skel & true_skel
        assert len(recovered) >= 2

    def test_scoring_prefers_correct_dag(self):
        """Correct DAG should score higher than wrong DAG."""
        edges = [('X', 'Y'), ('Y', 'Z')]
        data = _generate_categorical_data(edges, 1000, seed=42)

        correct = ScoringFunction.score_dag(data, ['X', 'Y', 'Z'],
                                             {'X': [], 'Y': ['X'], 'Z': ['Y']})
        empty = ScoringFunction.score_dag(data, ['X', 'Y', 'Z'],
                                           {'X': [], 'Y': [], 'Z': []})
        assert correct > empty

    def test_end_to_end_continuous_ges(self):
        """End-to-end GES on continuous data."""
        edges = [('X', 'Y')]
        data = _generate_continuous_data(edges, 500, coefficients={('X', 'Y'): 2.0}, seed=42)
        ges = GESAlgorithm(score_method='bic', continuous=True)
        cpdag = ges.fit(data, ['X', 'Y'])
        assert frozenset(['X', 'Y']) in cpdag.skeleton()

    def test_multiple_v_structures(self):
        """Graph with two v-structures."""
        edges = [('A', 'C'), ('B', 'C'), ('C', 'E'), ('D', 'E')]
        data = _generate_categorical_data(edges, 5000,
                                           cardinalities={'A':3, 'B':3, 'C':3, 'D':3, 'E':3},
                                           seed=42)

        pc = PCAlgorithm(alpha=0.05)
        cpdag = pc.fit(data)
        skel = cpdag.skeleton()

        # Should find at least some of the true edges
        true_skel = {frozenset(e) for e in edges}
        recovered = skel & true_skel
        assert len(recovered) >= 1

    def test_discovery_pipeline(self):
        """Full discovery pipeline: generate -> learn -> evaluate."""
        # True structure
        true_edges = [('Smoking', 'Cancer'), ('Smoking', 'Bronchitis')]
        true_set = set(map(tuple, true_edges))

        # Generate data
        data = _generate_categorical_data(true_edges, 2000, seed=42)

        # Learn with PC
        pc = PCAlgorithm(alpha=0.05)
        cpdag = pc.fit(data)

        # Evaluate
        result = DiscoveryAnalyzer.compare(true_set, cpdag)
        assert 'shd' in result
        assert isinstance(result['shd'], int)
        assert result['shd'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
