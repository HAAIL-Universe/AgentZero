"""Tests for C160: Probabilistic Graphical Models."""

import pytest
import math
import random
import numpy as np
from pgm import (
    Factor, BayesianNetwork, MarkovNetwork, FactorGraph,
    JunctionTree, StructureLearning, DynamicBayesNet, PGMUtils
)


# ===== Factor Tests =====

class TestFactor:
    def test_create_factor(self):
        f = Factor(['A', 'B'], {'A': 2, 'B': 2}, [0.1, 0.9, 0.4, 0.6])
        assert f.variables == ['A', 'B']
        assert f.cardinalities == {'A': 2, 'B': 2}

    def test_get_set_value(self):
        f = Factor(['A'], {'A': 3})
        f.set_value({'A': 0}, 0.2)
        f.set_value({'A': 1}, 0.3)
        f.set_value({'A': 2}, 0.5)
        assert f.get_value({'A': 0}) == pytest.approx(0.2)
        assert f.get_value({'A': 1}) == pytest.approx(0.3)
        assert f.get_value({'A': 2}) == pytest.approx(0.5)

    def test_multiply_factors(self):
        f1 = Factor(['A'], {'A': 2}, [0.6, 0.4])
        f2 = Factor(['A', 'B'], {'A': 2, 'B': 2}, [0.2, 0.8, 0.5, 0.5])
        result = f1.multiply(f2)
        assert set(result.variables) == {'A', 'B'}
        assert result.get_value({'A': 0, 'B': 0}) == pytest.approx(0.6 * 0.2)
        assert result.get_value({'A': 0, 'B': 1}) == pytest.approx(0.6 * 0.8)
        assert result.get_value({'A': 1, 'B': 0}) == pytest.approx(0.4 * 0.5)

    def test_multiply_disjoint(self):
        f1 = Factor(['A'], {'A': 2}, [0.3, 0.7])
        f2 = Factor(['B'], {'B': 2}, [0.4, 0.6])
        result = f1.multiply(f2)
        assert result.get_value({'A': 0, 'B': 0}) == pytest.approx(0.12)
        assert result.get_value({'A': 1, 'B': 1}) == pytest.approx(0.42)

    def test_marginalize(self):
        f = Factor(['A', 'B'], {'A': 2, 'B': 2}, [0.1, 0.2, 0.3, 0.4])
        m = f.marginalize('B')
        assert m.variables == ['A']
        assert m.get_value({'A': 0}) == pytest.approx(0.3)
        assert m.get_value({'A': 1}) == pytest.approx(0.7)

    def test_marginalize_all(self):
        f = Factor(['A'], {'A': 3}, [0.2, 0.3, 0.5])
        m = f.marginalize('A')
        assert m.variables == []
        assert float(m.values) == pytest.approx(1.0)

    def test_reduce(self):
        f = Factor(['A', 'B'], {'A': 2, 'B': 2}, [0.1, 0.9, 0.4, 0.6])
        r = f.reduce({'A': 0})
        assert r.variables == ['B']
        assert r.get_value({'B': 0}) == pytest.approx(0.1)
        assert r.get_value({'B': 1}) == pytest.approx(0.9)

    def test_reduce_all(self):
        f = Factor(['A', 'B'], {'A': 2, 'B': 2}, [0.1, 0.9, 0.4, 0.6])
        r = f.reduce({'A': 0, 'B': 1})
        assert r.variables == []
        assert float(r.values) == pytest.approx(0.9)

    def test_reduce_no_match(self):
        f = Factor(['A'], {'A': 2}, [0.3, 0.7])
        r = f.reduce({'B': 0})
        assert r.variables == ['A']

    def test_normalize(self):
        f = Factor(['A'], {'A': 3}, [2.0, 3.0, 5.0])
        f.normalize()
        assert f.get_value({'A': 0}) == pytest.approx(0.2)
        assert f.get_value({'A': 1}) == pytest.approx(0.3)
        assert f.get_value({'A': 2}) == pytest.approx(0.5)

    def test_factor_repr(self):
        f = Factor(['X', 'Y'], {'X': 2, 'Y': 3})
        assert 'X' in repr(f) and 'Y' in repr(f)

    def test_three_variable_factor(self):
        f = Factor(['A', 'B', 'C'], {'A': 2, 'B': 2, 'C': 2})
        f.set_value({'A': 0, 'B': 0, 'C': 0}, 0.5)
        f.set_value({'A': 1, 'B': 1, 'C': 1}, 0.8)
        assert f.get_value({'A': 0, 'B': 0, 'C': 0}) == pytest.approx(0.5)
        assert f.get_value({'A': 1, 'B': 1, 'C': 1}) == pytest.approx(0.8)

    def test_marginalize_three_to_two(self):
        f = Factor(['A', 'B', 'C'], {'A': 2, 'B': 2, 'C': 2},
                   [0.1, 0.2, 0.3, 0.4, 0.05, 0.15, 0.35, 0.45])
        m = f.marginalize('C')
        assert set(m.variables) == {'A', 'B'}
        # A=0,B=0: 0.1+0.2=0.3
        assert m.get_value({'A': 0, 'B': 0}) == pytest.approx(0.3)

    def test_multiply_then_marginalize(self):
        f1 = Factor(['A'], {'A': 2}, [0.6, 0.4])
        f2 = Factor(['A', 'B'], {'A': 2, 'B': 2}, [0.2, 0.8, 0.5, 0.5])
        prod = f1.multiply(f2)
        m = prod.marginalize('A')
        # B=0: 0.6*0.2 + 0.4*0.5 = 0.32
        # B=1: 0.6*0.8 + 0.4*0.5 = 0.68
        assert m.get_value({'B': 0}) == pytest.approx(0.32)
        assert m.get_value({'B': 1}) == pytest.approx(0.68)


# ===== BayesianNetwork Tests =====

def make_simple_bn():
    """Create A -> B -> C network."""
    bn = BayesianNetwork()
    bn.add_node('A', 2)
    bn.add_node('B', 2)
    bn.add_node('C', 2)
    bn.add_edge('A', 'B')
    bn.add_edge('B', 'C')

    # P(A)
    cpd_a = Factor(['A'], {'A': 2}, [0.6, 0.4])
    # P(B|A)
    cpd_b = Factor(['A', 'B'], {'A': 2, 'B': 2}, [0.2, 0.8, 0.75, 0.25])
    # P(C|B)
    cpd_c = Factor(['B', 'C'], {'B': 2, 'C': 2}, [0.1, 0.9, 0.8, 0.2])

    bn.set_cpd('A', cpd_a)
    bn.set_cpd('B', cpd_b)
    bn.set_cpd('C', cpd_c)
    return bn


def make_collider_bn():
    """Create A -> C <- B network (collider/v-structure)."""
    bn = BayesianNetwork()
    bn.add_node('A', 2)
    bn.add_node('B', 2)
    bn.add_node('C', 2)
    bn.add_edge('A', 'C')
    bn.add_edge('B', 'C')

    cpd_a = Factor(['A'], {'A': 2}, [0.5, 0.5])
    cpd_b = Factor(['B'], {'B': 2}, [0.5, 0.5])
    cpd_c = Factor(['A', 'B', 'C'], {'A': 2, 'B': 2, 'C': 2},
                   [0.9, 0.1, 0.5, 0.5, 0.5, 0.5, 0.1, 0.9])

    bn.set_cpd('A', cpd_a)
    bn.set_cpd('B', cpd_b)
    bn.set_cpd('C', cpd_c)
    return bn


class TestBayesianNetwork:
    def test_add_nodes_edges(self):
        bn = make_simple_bn()
        assert len(bn.nodes) == 3
        assert len(bn.edges) == 2

    def test_parents_children(self):
        bn = make_simple_bn()
        assert bn.get_parents('B') == ['A']
        assert bn.get_children('A') == ['B']
        assert bn.get_parents('A') == []
        assert bn.get_children('C') == []

    def test_topological_sort(self):
        bn = make_simple_bn()
        order = bn.topological_sort()
        assert order.index('A') < order.index('B')
        assert order.index('B') < order.index('C')

    def test_variable_elimination_prior(self):
        bn = make_simple_bn()
        result = bn.variable_elimination(['A'])
        assert result.get_value({'A': 0}) == pytest.approx(0.6, abs=0.01)
        assert result.get_value({'A': 1}) == pytest.approx(0.4, abs=0.01)

    def test_variable_elimination_marginal(self):
        bn = make_simple_bn()
        result = bn.variable_elimination(['B'])
        # P(B=0) = P(B=0|A=0)*P(A=0) + P(B=0|A=1)*P(A=1)
        #        = 0.2*0.6 + 0.75*0.4 = 0.12 + 0.30 = 0.42
        assert result.get_value({'B': 0}) == pytest.approx(0.42, abs=0.01)

    def test_variable_elimination_with_evidence(self):
        bn = make_simple_bn()
        result = bn.variable_elimination(['C'], evidence={'A': 0})
        # P(C|A=0): P(B=0|A=0)=0.2, P(B=1|A=0)=0.8
        # P(C=0|A=0) = P(C=0|B=0)*P(B=0|A=0) + P(C=0|B=1)*P(B=1|A=0)
        #            = 0.1*0.2 + 0.8*0.8 = 0.02 + 0.64 = 0.66
        assert result.get_value({'C': 0}) == pytest.approx(0.66, abs=0.01)

    def test_variable_elimination_joint_query(self):
        bn = make_simple_bn()
        result = bn.variable_elimination(['A', 'B'])
        # P(A=0,B=0) = 0.6*0.2 = 0.12
        assert result.get_value({'A': 0, 'B': 0}) == pytest.approx(0.12, abs=0.01)

    def test_markov_blanket(self):
        bn = make_simple_bn()
        mb = bn.markov_blanket('B')
        assert 'A' in mb  # parent
        assert 'C' in mb  # child

    def test_markov_blanket_collider(self):
        bn = make_collider_bn()
        mb = bn.markov_blanket('A')
        assert 'C' in mb   # child
        assert 'B' in mb   # co-parent

    def test_d_separation_chain(self):
        bn = make_simple_bn()
        # A -> B -> C: A and C are d-separated given B
        assert bn.is_d_separated('A', 'C', ['B'])
        # A and C are NOT d-separated given nothing
        assert not bn.is_d_separated('A', 'C', [])

    def test_d_separation_collider(self):
        bn = make_collider_bn()
        # A -> C <- B: A and B are d-separated given nothing (collider blocks)
        assert bn.is_d_separated('A', 'B', [])
        # A and B are NOT d-separated given C (explaining away)
        assert not bn.is_d_separated('A', 'B', ['C'])

    def test_sample(self):
        bn = make_simple_bn()
        np.random.seed(42)
        samples = bn.sample(100)
        assert len(samples) == 100
        for s in samples:
            assert set(s.keys()) == {'A', 'B', 'C'}
            assert s['A'] in [0, 1]

    def test_sample_with_evidence(self):
        bn = make_simple_bn()
        np.random.seed(42)
        samples = bn.sample(50, evidence={'A': 0})
        assert all(s['A'] == 0 for s in samples)

    def test_to_markov_network(self):
        bn = make_collider_bn()
        mn = bn.to_markov_network()
        assert len(mn.nodes) == 3
        # Moralization should marry A and B
        assert 'B' in mn.get_neighbors('A')

    def test_log_likelihood(self):
        bn = make_simple_bn()
        data = [{'A': 0, 'B': 0, 'C': 0}, {'A': 0, 'B': 1, 'C': 1}]
        ll = bn.log_likelihood(data)
        assert ll < 0  # log-likelihood should be negative
        assert math.isfinite(ll)

    def test_four_node_network(self):
        """D -> A -> B -> C, D -> B."""
        bn = BayesianNetwork()
        for n in ['A', 'B', 'C', 'D']:
            bn.add_node(n, 2)
        bn.add_edge('D', 'A')
        bn.add_edge('A', 'B')
        bn.add_edge('B', 'C')
        bn.add_edge('D', 'B')

        cpd_d = Factor(['D'], {'D': 2}, [0.5, 0.5])
        cpd_a = Factor(['D', 'A'], {'D': 2, 'A': 2}, [0.3, 0.7, 0.8, 0.2])
        cpd_b = Factor(['A', 'D', 'B'], {'A': 2, 'D': 2, 'B': 2},
                       [0.1, 0.9, 0.4, 0.6, 0.7, 0.3, 0.9, 0.1])
        cpd_c = Factor(['B', 'C'], {'B': 2, 'C': 2}, [0.2, 0.8, 0.6, 0.4])

        bn.set_cpd('D', cpd_d)
        bn.set_cpd('A', cpd_a)
        bn.set_cpd('B', cpd_b)
        bn.set_cpd('C', cpd_c)

        result = bn.variable_elimination(['C'], evidence={'D': 0})
        probs = [result.get_value({'C': 0}), result.get_value({'C': 1})]
        assert sum(probs) == pytest.approx(1.0, abs=0.01)

    def test_single_node_network(self):
        bn = BayesianNetwork()
        bn.add_node('X', 3)
        cpd = Factor(['X'], {'X': 3}, [0.2, 0.3, 0.5])
        bn.set_cpd('X', cpd)
        result = bn.variable_elimination(['X'])
        assert result.get_value({'X': 2}) == pytest.approx(0.5, abs=0.01)


# ===== MarkovNetwork Tests =====

class TestMarkovNetwork:
    def test_create(self):
        mn = MarkovNetwork()
        mn.add_node('A', 2)
        mn.add_node('B', 2)
        mn.add_edge('A', 'B')
        assert len(mn.nodes) == 2
        assert 'B' in mn.get_neighbors('A')

    def test_partition_function(self):
        mn = MarkovNetwork()
        mn.add_node('A', 2)
        mn.add_node('B', 2)
        mn.add_edge('A', 'B')
        f = Factor(['A', 'B'], {'A': 2, 'B': 2}, [2.0, 1.0, 1.0, 3.0])
        mn.add_factor(f)
        Z = mn.partition_function()
        assert Z == pytest.approx(7.0)

    def test_query(self):
        mn = MarkovNetwork()
        mn.add_node('A', 2)
        mn.add_node('B', 2)
        mn.add_edge('A', 'B')
        f = Factor(['A', 'B'], {'A': 2, 'B': 2}, [2.0, 1.0, 1.0, 3.0])
        mn.add_factor(f)
        result = mn.query(['A'])
        # P(A=0) = (2+1)/7 = 3/7
        assert result.get_value({'A': 0}) == pytest.approx(3.0/7, abs=0.01)

    def test_query_with_evidence(self):
        mn = MarkovNetwork()
        mn.add_node('A', 2)
        mn.add_node('B', 2)
        mn.add_edge('A', 'B')
        f = Factor(['A', 'B'], {'A': 2, 'B': 2}, [2.0, 1.0, 1.0, 3.0])
        mn.add_factor(f)
        result = mn.query(['A'], evidence={'B': 0})
        # P(A=0|B=0) = 2/(2+1) = 2/3
        assert result.get_value({'A': 0}) == pytest.approx(2.0/3, abs=0.01)

    def test_three_node_markov(self):
        mn = MarkovNetwork()
        for n in ['A', 'B', 'C']:
            mn.add_node(n, 2)
        mn.add_edge('A', 'B')
        mn.add_edge('B', 'C')
        f1 = Factor(['A', 'B'], {'A': 2, 'B': 2}, [2.0, 1.0, 1.0, 3.0])
        f2 = Factor(['B', 'C'], {'B': 2, 'C': 2}, [1.0, 2.0, 3.0, 1.0])
        mn.add_factor(f1)
        mn.add_factor(f2)
        result = mn.query(['A'])
        probs = [result.get_value({'A': 0}), result.get_value({'A': 1})]
        assert sum(probs) == pytest.approx(1.0, abs=0.01)

    def test_gibbs_sample(self):
        random.seed(42)
        np.random.seed(42)
        mn = MarkovNetwork()
        mn.add_node('A', 2)
        mn.add_node('B', 2)
        mn.add_edge('A', 'B')
        f = Factor(['A', 'B'], {'A': 2, 'B': 2}, [5.0, 1.0, 1.0, 5.0])
        mn.add_factor(f)
        samples = mn.gibbs_sample(500, burn_in=100)
        assert len(samples) == 500
        # Strong diagonal: A=B should be more common
        agree = sum(1 for s in samples if s['A'] == s['B'])
        assert agree > 250  # Should be ~83%

    def test_gibbs_with_evidence(self):
        random.seed(42)
        mn = MarkovNetwork()
        mn.add_node('A', 2)
        mn.add_node('B', 2)
        mn.add_edge('A', 'B')
        f = Factor(['A', 'B'], {'A': 2, 'B': 2}, [2.0, 1.0, 1.0, 3.0])
        mn.add_factor(f)
        samples = mn.gibbs_sample(100, burn_in=50, evidence={'A': 0})
        assert all(s['A'] == 0 for s in samples)

    def test_partition_with_evidence(self):
        mn = MarkovNetwork()
        mn.add_node('A', 2)
        mn.add_node('B', 2)
        mn.add_edge('A', 'B')
        f = Factor(['A', 'B'], {'A': 2, 'B': 2}, [2.0, 1.0, 1.0, 3.0])
        mn.add_factor(f)
        Z = mn.partition_function(evidence={'A': 0})
        assert Z == pytest.approx(3.0)  # 2+1


# ===== FactorGraph Tests =====

class TestFactorGraph:
    def test_create(self):
        fg = FactorGraph()
        fg.add_variable('A', 2)
        fg.add_variable('B', 2)
        f = Factor(['A', 'B'], {'A': 2, 'B': 2}, [0.3, 0.7, 0.6, 0.4])
        fg.add_factor(f, 'f0')
        assert len(fg.variables) == 2
        assert len(fg.factors) == 1

    def test_belief_propagation_single_factor(self):
        fg = FactorGraph()
        fg.add_variable('A', 2)
        f = Factor(['A'], {'A': 2}, [0.3, 0.7])
        fg.add_factor(f, 'f0')
        beliefs = fg.belief_propagation()
        assert beliefs['A'][0] == pytest.approx(0.3, abs=0.01)
        assert beliefs['A'][1] == pytest.approx(0.7, abs=0.01)

    def test_belief_propagation_chain(self):
        fg = FactorGraph()
        fg.add_variable('A', 2)
        fg.add_variable('B', 2)
        f1 = Factor(['A'], {'A': 2}, [0.6, 0.4])
        f2 = Factor(['A', 'B'], {'A': 2, 'B': 2}, [0.2, 0.8, 0.75, 0.25])
        fg.add_factor(f1, 'f1')
        fg.add_factor(f2, 'f2')
        beliefs = fg.belief_propagation()
        # P(B=0) = 0.6*0.2 + 0.4*0.75 = 0.42
        assert beliefs['B'][0] == pytest.approx(0.42, abs=0.05)

    def test_belief_propagation_triangle(self):
        fg = FactorGraph()
        for v in ['A', 'B', 'C']:
            fg.add_variable(v, 2)
        f1 = Factor(['A', 'B'], {'A': 2, 'B': 2}, [3.0, 1.0, 1.0, 3.0])
        f2 = Factor(['B', 'C'], {'B': 2, 'C': 2}, [3.0, 1.0, 1.0, 3.0])
        f3 = Factor(['A', 'C'], {'A': 2, 'C': 2}, [3.0, 1.0, 1.0, 3.0])
        fg.add_factor(f1, 'f1')
        fg.add_factor(f2, 'f2')
        fg.add_factor(f3, 'f3')
        beliefs = fg.belief_propagation(max_iter=100)
        # Symmetric -- all beliefs should be approximately [0.5, 0.5]
        for v in ['A', 'B', 'C']:
            assert abs(beliefs[v][0] - 0.5) < 0.1

    def test_max_product(self):
        fg = FactorGraph()
        fg.add_variable('A', 2)
        fg.add_variable('B', 2)
        f1 = Factor(['A'], {'A': 2}, [0.3, 0.7])
        f2 = Factor(['A', 'B'], {'A': 2, 'B': 2}, [0.9, 0.1, 0.1, 0.9])
        fg.add_factor(f1, 'f1')
        fg.add_factor(f2, 'f2')
        assignment, beliefs = fg.max_product()
        # Most likely: A=1 (prior 0.7), B=1 (agrees with A=1)
        assert assignment['A'] == 1
        assert assignment['B'] == 1

    def test_max_product_three_vars(self):
        fg = FactorGraph()
        for v in ['X', 'Y', 'Z']:
            fg.add_variable(v, 2)
        f1 = Factor(['X'], {'X': 2}, [0.1, 0.9])
        f2 = Factor(['X', 'Y'], {'X': 2, 'Y': 2}, [0.8, 0.2, 0.2, 0.8])
        f3 = Factor(['Y', 'Z'], {'Y': 2, 'Z': 2}, [0.8, 0.2, 0.2, 0.8])
        fg.add_factor(f1, 'fx')
        fg.add_factor(f2, 'fxy')
        fg.add_factor(f3, 'fyz')
        assignment, _ = fg.max_product()
        # X=1 (strong prior), Y=1 (agrees), Z=1 (agrees)
        assert assignment['X'] == 1
        assert assignment['Y'] == 1
        assert assignment['Z'] == 1

    def test_bp_convergence(self):
        fg = FactorGraph()
        fg.add_variable('A', 2)
        f = Factor(['A'], {'A': 2}, [0.4, 0.6])
        fg.add_factor(f, 'f0')
        beliefs = fg.belief_propagation(max_iter=5)
        assert sum(beliefs['A']) == pytest.approx(1.0, abs=0.01)

    def test_bp_four_variables(self):
        fg = FactorGraph()
        for v in ['A', 'B', 'C', 'D']:
            fg.add_variable(v, 2)
        f1 = Factor(['A', 'B'], {'A': 2, 'B': 2}, [2, 1, 1, 2])
        f2 = Factor(['B', 'C'], {'B': 2, 'C': 2}, [2, 1, 1, 2])
        f3 = Factor(['C', 'D'], {'C': 2, 'D': 2}, [2, 1, 1, 2])
        fg.add_factor(f1)
        fg.add_factor(f2)
        fg.add_factor(f3)
        beliefs = fg.belief_propagation()
        for v in ['A', 'B', 'C', 'D']:
            assert sum(beliefs[v]) == pytest.approx(1.0, abs=0.01)


# ===== JunctionTree Tests =====

class TestJunctionTree:
    def test_build_from_chain(self):
        bn = make_simple_bn()
        jt = JunctionTree(bn)
        assert len(jt.cliques) >= 1

    def test_query_prior(self):
        bn = make_simple_bn()
        jt = JunctionTree(bn)
        result = jt.query(['A'])
        assert result.get_value({'A': 0}) == pytest.approx(0.6, abs=0.02)

    def test_query_marginal(self):
        bn = make_simple_bn()
        jt = JunctionTree(bn)
        result = jt.query(['B'])
        assert result.get_value({'B': 0}) == pytest.approx(0.42, abs=0.02)

    def test_query_with_evidence(self):
        bn = make_simple_bn()
        jt = JunctionTree(bn)
        result = jt.query(['C'], evidence={'A': 0})
        expected = 0.1 * 0.2 + 0.8 * 0.8  # 0.66
        assert result.get_value({'C': 0}) == pytest.approx(expected, abs=0.02)

    def test_query_collider(self):
        bn = make_collider_bn()
        jt = JunctionTree(bn)
        result = jt.query(['A'])
        assert result.get_value({'A': 0}) == pytest.approx(0.5, abs=0.02)

    def test_single_node_jt(self):
        bn = BayesianNetwork()
        bn.add_node('X', 3)
        cpd = Factor(['X'], {'X': 3}, [0.2, 0.3, 0.5])
        bn.set_cpd('X', cpd)
        jt = JunctionTree(bn)
        result = jt.query(['X'])
        assert result.get_value({'X': 2}) == pytest.approx(0.5, abs=0.02)

    def test_jt_agrees_with_ve(self):
        """Junction tree should give same results as variable elimination."""
        bn = make_simple_bn()
        jt = JunctionTree(bn)
        ve_result = bn.variable_elimination(['C'], evidence={'A': 1})
        jt_result = jt.query(['C'], evidence={'A': 1})
        for c_val in [0, 1]:
            assert (jt_result.get_value({'C': c_val}) ==
                    pytest.approx(ve_result.get_value({'C': c_val}), abs=0.02))


# ===== StructureLearning Tests =====

def generate_bn_data(bn, n=500):
    """Generate data from a BN."""
    np.random.seed(42)
    return bn.sample(n)


class TestStructureLearning:
    def test_k2_learns_chain(self):
        bn = make_simple_bn()
        np.random.seed(42)
        data = bn.sample(500)
        sl = StructureLearning(data, bn.cardinalities)
        learned = sl.k2_search(ordering=['A', 'B', 'C'])
        # Should learn A->B and B->C
        edges = set(learned.edges)
        assert ('A', 'B') in edges
        assert ('B', 'C') in edges

    def test_k2_no_false_edges(self):
        """K2 should not add spurious edges for independent variables."""
        np.random.seed(42)
        bn = BayesianNetwork()
        bn.add_node('A', 2)
        bn.add_node('B', 2)
        cpd_a = Factor(['A'], {'A': 2}, [0.5, 0.5])
        cpd_b = Factor(['B'], {'B': 2}, [0.5, 0.5])
        bn.set_cpd('A', cpd_a)
        bn.set_cpd('B', cpd_b)
        data = bn.sample(500)
        sl = StructureLearning(data, bn.cardinalities)
        learned = sl.k2_search(ordering=['A', 'B'])
        assert len(learned.edges) == 0

    def test_bic_score(self):
        bn = make_simple_bn()
        np.random.seed(42)
        data = bn.sample(300)
        sl = StructureLearning(data, bn.cardinalities)
        # Score with correct parent should be better than no parent
        score_with = sl.bic_score('B', ['A'])
        score_without = sl.bic_score('B', [])
        assert score_with > score_without

    def test_hill_climb(self):
        bn = make_simple_bn()
        np.random.seed(42)
        data = bn.sample(500)
        sl = StructureLearning(data, bn.cardinalities)
        learned = sl.hill_climb(max_parents=2)
        # Should learn some structure
        assert len(learned.nodes) == 3
        # At minimum, should find the A-B relationship
        has_ab = any((a, b) in learned.edges or (b, a) in learned.edges
                     for a, b in [('A', 'B')])
        assert has_ab

    def test_estimate_cpd(self):
        bn = make_simple_bn()
        np.random.seed(42)
        data = bn.sample(1000)
        sl = StructureLearning(data, bn.cardinalities)
        cpd = sl._estimate_cpd('A', [])
        # Should be close to [0.6, 0.4]
        assert cpd.get_value({'A': 0}) == pytest.approx(0.6, abs=0.1)

    def test_hill_climb_cycle_prevention(self):
        """Hill climbing should never create cycles."""
        np.random.seed(42)
        bn = make_simple_bn()
        data = bn.sample(300)
        sl = StructureLearning(data, bn.cardinalities)
        learned = sl.hill_climb()
        # Verify DAG property
        order = learned.topological_sort()
        assert len(order) == len(learned.nodes)


# ===== DynamicBayesNet Tests =====

class TestDynamicBayesNet:
    def test_create_dbn(self):
        dbn = DynamicBayesNet()
        dbn.add_node('X', 2)
        dbn.add_node('Y', 2)
        dbn.add_intra_edge('X', 'Y')
        dbn.add_inter_edge('X', 'X')
        assert len(dbn.nodes) == 2
        assert len(dbn.intra_edges) == 1
        assert len(dbn.inter_edges) == 1

    def test_unroll(self):
        dbn = DynamicBayesNet()
        dbn.add_node('X', 2)
        dbn.add_inter_edge('X', 'X')

        # Initial CPD
        cpd_init = Factor(['X'], {'X': 2}, [0.5, 0.5])
        dbn.set_initial_cpd('X', cpd_init)

        # Transition CPD: P(X_t | X_{t-1})
        cpd_trans = Factor(['X_prev', 'X'], {'X_prev': 2, 'X': 2},
                           [0.7, 0.3, 0.3, 0.7])
        dbn.set_transition_cpd('X', cpd_trans)

        bn = dbn.unroll(3)
        assert 'X_t0' in bn.nodes
        assert 'X_t1' in bn.nodes
        assert 'X_t2' in bn.nodes

    def test_unroll_inference(self):
        dbn = DynamicBayesNet()
        dbn.add_node('X', 2)
        dbn.add_inter_edge('X', 'X')

        cpd_init = Factor(['X'], {'X': 2}, [0.5, 0.5])
        dbn.set_initial_cpd('X', cpd_init)

        cpd_trans = Factor(['X_prev', 'X'], {'X_prev': 2, 'X': 2},
                           [0.9, 0.1, 0.1, 0.9])
        dbn.set_transition_cpd('X', cpd_trans)

        bn = dbn.unroll(3)
        result = bn.variable_elimination(['X_t2'], evidence={'X_t0': 0})
        # Strong self-transition: X_t2 should still be close to 0
        assert result.get_value({'X_t2': 0}) > 0.5

    def test_dbn_with_observation(self):
        dbn = DynamicBayesNet()
        dbn.add_node('H', 2)  # hidden
        dbn.add_node('O', 2)  # observed
        dbn.add_intra_edge('H', 'O')
        dbn.add_inter_edge('H', 'H')

        cpd_h_init = Factor(['H'], {'H': 2}, [0.6, 0.4])
        cpd_o = Factor(['H', 'O'], {'H': 2, 'O': 2}, [0.9, 0.1, 0.2, 0.8])
        cpd_h_trans = Factor(['H_prev', 'H'], {'H_prev': 2, 'H': 2},
                             [0.7, 0.3, 0.3, 0.7])

        dbn.set_initial_cpd('H', cpd_h_init)
        dbn.set_initial_cpd('O', cpd_o)
        dbn.set_transition_cpd('H', cpd_h_trans)
        dbn.set_transition_cpd('O', cpd_o)

        bn = dbn.unroll(2)
        result = bn.variable_elimination(['H_t1'], evidence={'O_t0': 0, 'O_t1': 0})
        # Observing O=0 twice suggests H=0
        assert result.get_value({'H_t1': 0}) > result.get_value({'H_t1': 1})

    def test_filter(self):
        dbn = DynamicBayesNet()
        dbn.add_node('H', 2)
        dbn.add_node('O', 2)
        dbn.add_intra_edge('H', 'O')
        dbn.add_inter_edge('H', 'H')

        cpd_h_init = Factor(['H'], {'H': 2}, [0.5, 0.5])
        cpd_o = Factor(['H', 'O'], {'H': 2, 'O': 2}, [0.9, 0.1, 0.1, 0.9])
        cpd_h_trans = Factor(['H_prev', 'H'], {'H_prev': 2, 'H': 2},
                             [0.8, 0.2, 0.2, 0.8])

        dbn.set_initial_cpd('H', cpd_h_init)
        dbn.set_initial_cpd('O', cpd_o)
        dbn.set_transition_cpd('H', cpd_h_trans)
        dbn.set_transition_cpd('O', cpd_o)

        observations = [{'O': 0}, {'O': 0}, {'O': 1}]
        results = dbn.filter(observations)
        assert len(results) == 3

    def test_unroll_two_nodes(self):
        dbn = DynamicBayesNet()
        dbn.add_node('A', 2)
        dbn.add_node('B', 2)
        dbn.add_intra_edge('A', 'B')
        dbn.add_inter_edge('A', 'A')
        dbn.add_inter_edge('B', 'B')

        cpd_a = Factor(['A'], {'A': 2}, [0.5, 0.5])
        cpd_b = Factor(['A', 'B'], {'A': 2, 'B': 2}, [0.8, 0.2, 0.3, 0.7])
        cpd_a_trans = Factor(['A_prev', 'A'], {'A_prev': 2, 'A': 2},
                             [0.9, 0.1, 0.1, 0.9])
        cpd_b_trans = Factor(['A', 'B_prev', 'B'], {'A': 2, 'B_prev': 2, 'B': 2},
                             [0.7, 0.3, 0.4, 0.6, 0.2, 0.8, 0.5, 0.5])

        dbn.set_initial_cpd('A', cpd_a)
        dbn.set_initial_cpd('B', cpd_b)
        dbn.set_transition_cpd('A', cpd_a_trans)
        dbn.set_transition_cpd('B', cpd_b_trans)

        bn = dbn.unroll(2)
        assert len(bn.nodes) == 4  # A_t0, B_t0, A_t1, B_t1


# ===== PGMUtils Tests =====

class TestPGMUtils:
    def test_mutual_information_independent(self):
        """MI of independent vars should be near 0."""
        np.random.seed(42)
        data = [{'X': random.randint(0, 1), 'Y': random.randint(0, 1)}
                for _ in range(1000)]
        mi = PGMUtils.mutual_information(data, 'X', 'Y', {'X': 2, 'Y': 2})
        assert mi < 0.1

    def test_mutual_information_dependent(self):
        """MI of identical vars should be high."""
        data = [{'X': v, 'Y': v} for v in [0, 1] * 500]
        mi = PGMUtils.mutual_information(data, 'X', 'Y', {'X': 2, 'Y': 2})
        assert mi > 0.5

    def test_conditional_mutual_information(self):
        random.seed(42)
        data = []
        for _ in range(1000):
            z = random.randint(0, 1)
            x = z if random.random() < 0.8 else 1 - z
            y = z if random.random() < 0.8 else 1 - z
            data.append({'X': x, 'Y': y, 'Z': z})
        cmi = PGMUtils.conditional_mutual_information(
            data, 'X', 'Y', 'Z', {'X': 2, 'Y': 2, 'Z': 2})
        # X and Y are conditionally less dependent given Z
        mi = PGMUtils.mutual_information(data, 'X', 'Y', {'X': 2, 'Y': 2})
        assert cmi < mi

    def test_independence_test_independent(self):
        random.seed(42)
        data = [{'X': random.randint(0, 1), 'Y': random.randint(0, 1)}
                for _ in range(500)]
        independent, chi2 = PGMUtils.independence_test(
            data, 'X', 'Y', set(), {'X': 2, 'Y': 2})
        assert independent

    def test_independence_test_dependent(self):
        data = [{'X': v, 'Y': v} for v in [0, 1] * 250]
        independent, chi2 = PGMUtils.independence_test(
            data, 'X', 'Y', set(), {'X': 2, 'Y': 2})
        assert not independent
        assert chi2 > 0

    def test_conditional_independence_test(self):
        random.seed(42)
        data = []
        for _ in range(500):
            z = random.randint(0, 1)
            x = z
            y = z
            data.append({'X': x, 'Y': y, 'Z': z})
        independent, _ = PGMUtils.independence_test(
            data, 'X', 'Y', {'Z'}, {'X': 2, 'Y': 2, 'Z': 2})
        # X and Y are conditionally independent given Z
        assert independent

    def test_kl_divergence(self):
        p = [0.5, 0.5]
        q = [0.5, 0.5]
        assert PGMUtils.kl_divergence(p, q) == pytest.approx(0.0)

    def test_kl_divergence_nonzero(self):
        p = [0.9, 0.1]
        q = [0.5, 0.5]
        kl = PGMUtils.kl_divergence(p, q)
        assert kl > 0

    def test_entropy(self):
        # Uniform distribution has max entropy
        h_uniform = PGMUtils.entropy([0.5, 0.5])
        h_peaked = PGMUtils.entropy([0.9, 0.1])
        assert h_uniform > h_peaked

    def test_entropy_deterministic(self):
        h = PGMUtils.entropy([1.0, 0.0])
        assert h == pytest.approx(0.0)

    def test_pc_algorithm(self):
        """PC should find some structure in dependent data."""
        random.seed(42)
        data = []
        for _ in range(500):
            a = random.randint(0, 1)
            b = a if random.random() < 0.9 else 1 - a
            c = b if random.random() < 0.9 else 1 - b
            data.append({'A': a, 'B': b, 'C': c})
        result = PGMUtils.pc_algorithm(
            data, {'A': 2, 'B': 2, 'C': 2}, alpha=0.05)
        # Should find edges between A-B and B-C
        adj = result['adjacency']
        assert 'B' in adj.get('A', set()) or 'A' in adj.get('B', set())

    def test_pc_independent_vars(self):
        """PC should find no edges for independent variables."""
        random.seed(42)
        data = [{'A': random.randint(0, 1), 'B': random.randint(0, 1)}
                for _ in range(500)]
        result = PGMUtils.pc_algorithm(
            data, {'A': 2, 'B': 2}, alpha=0.05)
        # Should have no edges
        adj = result['adjacency']
        assert len(adj.get('A', set())) == 0

    def test_mutual_information_empty_data(self):
        mi = PGMUtils.mutual_information([], 'X', 'Y', {'X': 2, 'Y': 2})
        assert mi == 0.0


# ===== Integration Tests =====

class TestIntegration:
    def test_bn_to_fg_bp(self):
        """Convert BN to factor graph and run BP."""
        bn = make_simple_bn()
        fg = FactorGraph()
        for node in bn.nodes:
            fg.add_variable(node, bn.cardinalities[node])
        for node in bn.nodes:
            fg.add_factor(bn.cpds[node], f"cpd_{node}")

        beliefs = fg.belief_propagation()
        # P(B=0) should be ~0.42
        assert beliefs['B'][0] == pytest.approx(0.42, abs=0.05)

    def test_ve_vs_jt_consistency(self):
        """VE and JT should give consistent results."""
        bn = make_collider_bn()
        ve = bn.variable_elimination(['A'], evidence={'C': 0})
        jt = JunctionTree(bn)
        jt_result = jt.query(['A'], evidence={'C': 0})
        assert (ve.get_value({'A': 0}) ==
                pytest.approx(jt_result.get_value({'A': 0}), abs=0.02))

    def test_learn_and_infer(self):
        """Learn structure from data, then infer."""
        bn = make_simple_bn()
        np.random.seed(42)
        data = bn.sample(500)
        sl = StructureLearning(data, bn.cardinalities)
        learned = sl.k2_search(ordering=['A', 'B', 'C'])
        result = learned.variable_elimination(['A'])
        assert result.get_value({'A': 0}) == pytest.approx(0.6, abs=0.15)

    def test_moralize_then_query(self):
        """Moralize BN and query via Markov Network."""
        bn = make_simple_bn()
        mn = bn.to_markov_network()
        result = mn.query(['B'])
        assert result.get_value({'B': 0}) == pytest.approx(0.42, abs=0.05)

    def test_dbn_filter_consistency(self):
        """DBN filtering should track observations."""
        dbn = DynamicBayesNet()
        dbn.add_node('H', 2)
        dbn.add_node('O', 2)
        dbn.add_intra_edge('H', 'O')
        dbn.add_inter_edge('H', 'H')

        cpd_h_init = Factor(['H'], {'H': 2}, [0.5, 0.5])
        cpd_o = Factor(['H', 'O'], {'H': 2, 'O': 2}, [0.9, 0.1, 0.1, 0.9])
        cpd_h_trans = Factor(['H_prev', 'H'], {'H_prev': 2, 'H': 2},
                             [0.8, 0.2, 0.2, 0.8])

        dbn.set_initial_cpd('H', cpd_h_init)
        dbn.set_initial_cpd('O', cpd_o)
        dbn.set_transition_cpd('H', cpd_h_trans)
        dbn.set_transition_cpd('O', cpd_o)

        # All observations are 0 -- should shift belief toward H=0
        obs = [{'O': 0}] * 3
        results = dbn.filter(obs)
        # After seeing O=0 three times, P(H=0) should be high
        assert results[-1] is not None

    def test_full_pipeline(self):
        """Generate data -> learn structure -> infer -> verify."""
        np.random.seed(123)
        # Create ground truth
        bn = BayesianNetwork()
        bn.add_node('Rain', 2)
        bn.add_node('Sprinkler', 2)
        bn.add_node('Wet', 2)
        bn.add_edge('Rain', 'Sprinkler')
        bn.add_edge('Rain', 'Wet')
        bn.add_edge('Sprinkler', 'Wet')

        cpd_rain = Factor(['Rain'], {'Rain': 2}, [0.8, 0.2])
        cpd_sprinkler = Factor(['Rain', 'Sprinkler'], {'Rain': 2, 'Sprinkler': 2},
                               [0.6, 0.4, 0.99, 0.01])
        cpd_wet = Factor(['Rain', 'Sprinkler', 'Wet'],
                         {'Rain': 2, 'Sprinkler': 2, 'Wet': 2},
                         [1.0, 0.0, 0.1, 0.9, 0.1, 0.9, 0.01, 0.99])

        bn.set_cpd('Rain', cpd_rain)
        bn.set_cpd('Sprinkler', cpd_sprinkler)
        bn.set_cpd('Wet', cpd_wet)

        # Generate data
        data = bn.sample(1000)
        assert len(data) == 1000

        # Learn structure
        sl = StructureLearning(data, bn.cardinalities)
        learned = sl.k2_search(ordering=['Rain', 'Sprinkler', 'Wet'])

        # Infer
        result = learned.variable_elimination(['Rain'], evidence={'Wet': 1})
        probs = [result.get_value({'Rain': 0}), result.get_value({'Rain': 1})]
        assert sum(probs) == pytest.approx(1.0, abs=0.01)

    def test_factor_chain_operations(self):
        """Chain of factor operations: multiply, reduce, marginalize."""
        f1 = Factor(['A', 'B'], {'A': 2, 'B': 2}, [0.3, 0.7, 0.6, 0.4])
        f2 = Factor(['B', 'C'], {'B': 2, 'C': 2}, [0.1, 0.9, 0.8, 0.2])
        product = f1.multiply(f2)
        reduced = product.reduce({'A': 0})
        marginalized = reduced.marginalize('B')
        marginalized.normalize()
        probs = [marginalized.get_value({'C': c}) for c in range(2)]
        assert sum(probs) == pytest.approx(1.0, abs=0.01)

    def test_large_network(self):
        """Test with a larger network (5 nodes in a chain)."""
        bn = BayesianNetwork()
        nodes = ['A', 'B', 'C', 'D', 'E']
        for n in nodes:
            bn.add_node(n, 2)
        for i in range(len(nodes) - 1):
            bn.add_edge(nodes[i], nodes[i + 1])

        # Simple CPDs
        cpd_a = Factor(['A'], {'A': 2}, [0.5, 0.5])
        bn.set_cpd('A', cpd_a)
        for i in range(1, len(nodes)):
            parent = nodes[i - 1]
            child = nodes[i]
            cpd = Factor([parent, child], {parent: 2, child: 2},
                         [0.8, 0.2, 0.2, 0.8])
            bn.set_cpd(child, cpd)

        result = bn.variable_elimination(['E'], evidence={'A': 0})
        # E should lean toward 0 through chain
        assert result.get_value({'E': 0}) > result.get_value({'E': 1})

    def test_multiple_evidence(self):
        """Test with multiple evidence variables."""
        bn = make_collider_bn()
        result = bn.variable_elimination(['C'], evidence={'A': 0, 'B': 1})
        probs = [result.get_value({'C': c}) for c in range(2)]
        assert sum(probs) == pytest.approx(1.0, abs=0.01)

    def test_markov_network_multiple_factors(self):
        """MN with multiple overlapping factors."""
        mn = MarkovNetwork()
        for n in ['A', 'B', 'C']:
            mn.add_node(n, 2)
        mn.add_edge('A', 'B')
        mn.add_edge('B', 'C')
        mn.add_edge('A', 'C')

        f1 = Factor(['A', 'B'], {'A': 2, 'B': 2}, [3, 1, 1, 3])
        f2 = Factor(['B', 'C'], {'B': 2, 'C': 2}, [3, 1, 1, 3])
        f3 = Factor(['A', 'C'], {'A': 2, 'C': 2}, [3, 1, 1, 3])
        mn.add_factor(f1)
        mn.add_factor(f2)
        mn.add_factor(f3)

        result = mn.query(['A'])
        assert result.get_value({'A': 0}) == pytest.approx(0.5, abs=0.01)

    def test_factor_normalize_zeros(self):
        """Normalizing all-zero factor should not crash."""
        f = Factor(['A'], {'A': 2}, [0.0, 0.0])
        f.normalize()
        assert f.get_value({'A': 0}) == 0.0


# ===== Edge Case Tests =====

class TestEdgeCases:
    def test_factor_single_value(self):
        f = Factor([], {})
        assert f.values.shape == ()

    def test_bn_no_edges(self):
        bn = BayesianNetwork()
        bn.add_node('A', 2)
        bn.add_node('B', 2)
        cpd_a = Factor(['A'], {'A': 2}, [0.3, 0.7])
        cpd_b = Factor(['B'], {'B': 2}, [0.6, 0.4])
        bn.set_cpd('A', cpd_a)
        bn.set_cpd('B', cpd_b)
        result = bn.variable_elimination(['A', 'B'])
        assert result.get_value({'A': 0, 'B': 0}) == pytest.approx(0.18, abs=0.01)

    def test_ternary_variables(self):
        bn = BayesianNetwork()
        bn.add_node('X', 3)
        cpd = Factor(['X'], {'X': 3}, [0.2, 0.3, 0.5])
        bn.set_cpd('X', cpd)
        result = bn.variable_elimination(['X'])
        assert result.get_value({'X': 0}) == pytest.approx(0.2, abs=0.01)
        assert result.get_value({'X': 2}) == pytest.approx(0.5, abs=0.01)

    def test_dsep_self(self):
        bn = make_simple_bn()
        # Node is d-separated from itself given itself (trivially)
        # But in our implementation, x != y check doesn't exist,
        # so just test non-trivial cases
        assert bn.is_d_separated('A', 'C', ['B'])

    def test_empty_markov_blanket(self):
        bn = BayesianNetwork()
        bn.add_node('X', 2)
        cpd = Factor(['X'], {'X': 2}, [0.5, 0.5])
        bn.set_cpd('X', cpd)
        mb = bn.markov_blanket('X')
        assert len(mb) == 0

    def test_factor_graph_single_variable(self):
        fg = FactorGraph()
        fg.add_variable('X', 3)
        f = Factor(['X'], {'X': 3}, [0.2, 0.3, 0.5])
        fg.add_factor(f)
        beliefs = fg.belief_propagation()
        assert beliefs['X'][2] == pytest.approx(0.5, abs=0.01)

    def test_duplicate_node_add(self):
        bn = BayesianNetwork()
        bn.add_node('A', 2)
        bn.add_node('A', 3)  # should not duplicate
        assert bn.nodes.count('A') == 1

    def test_mn_duplicate_node(self):
        mn = MarkovNetwork()
        mn.add_node('A', 2)
        mn.add_node('A', 2)
        assert mn.nodes.count('A') == 1

    def test_fg_duplicate_variable(self):
        fg = FactorGraph()
        fg.add_variable('X', 2)
        fg.add_variable('X', 2)
        assert fg.variables.count('X') == 1

    def test_dbn_duplicate_node(self):
        dbn = DynamicBayesNet()
        dbn.add_node('H', 2)
        dbn.add_node('H', 2)
        assert dbn.nodes.count('H') == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
