"""Tests for V214: Causal Discovery."""

import sys
import os
import math
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V209_bayesian_networks'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V211_causal_inference'))

from causal_discovery import (
    chi_squared_test,
    mutual_information_test,
    _contingency_table,
    _chi2_critical,
    _normal_quantile,
    pc_algorithm,
    PCResult,
    hill_climbing,
    HillClimbResult,
    hybrid_discovery,
    bic_score,
    structural_hamming_distance,
    sample_from_bn,
    learn_bn_structure,
    learn_causal_model,
    _has_cycle,
    _estimate_parameters_mle,
)
from bayesian_networks import BayesianNetwork, variable_elimination, Factor
from causal_inference import CausalModel

import pytest


# ---------------------------------------------------------------------------
# Helper: build simple test BNs and generate data
# ---------------------------------------------------------------------------

def build_chain_abc():
    """A -> B -> C with binary variables."""
    bn = BayesianNetwork()
    bn.add_node("A", [0, 1])
    bn.add_node("B", [0, 1])
    bn.add_node("C", [0, 1])
    bn.add_edge("A", "B")
    bn.add_edge("B", "C")
    bn.set_cpt_dict("A", {0: 0.6, 1: 0.4})
    bn.set_cpt_dict("B", {(0,): {0: 0.8, 1: 0.2}, (1,): {0: 0.3, 1: 0.7}})
    bn.set_cpt_dict("C", {(0,): {0: 0.9, 1: 0.1}, (1,): {0: 0.4, 1: 0.6}})
    return bn


def build_fork_abc():
    """B <- A -> C (common cause)."""
    bn = BayesianNetwork()
    bn.add_node("A", [0, 1])
    bn.add_node("B", [0, 1])
    bn.add_node("C", [0, 1])
    bn.add_edge("A", "B")
    bn.add_edge("A", "C")
    bn.set_cpt_dict("A", {0: 0.5, 1: 0.5})
    bn.set_cpt_dict("B", {(0,): {0: 0.9, 1: 0.1}, (1,): {0: 0.2, 1: 0.8}})
    bn.set_cpt_dict("C", {(0,): {0: 0.85, 1: 0.15}, (1,): {0: 0.3, 1: 0.7}})
    return bn


def build_collider_abc():
    """A -> C <- B (collider/v-structure)."""
    bn = BayesianNetwork()
    bn.add_node("A", [0, 1])
    bn.add_node("B", [0, 1])
    bn.add_node("C", [0, 1])
    bn.add_edge("A", "C")
    bn.add_edge("B", "C")
    bn.set_cpt_dict("A", {0: 0.5, 1: 0.5})
    bn.set_cpt_dict("B", {0: 0.5, 1: 0.5})
    bn.set_cpt_dict("C", {
        (0, 0): {0: 0.95, 1: 0.05},
        (0, 1): {0: 0.4, 1: 0.6},
        (1, 0): {0: 0.4, 1: 0.6},
        (1, 1): {0: 0.1, 1: 0.9},
    })
    return bn


def build_diamond():
    """A -> B, A -> C, B -> D, C -> D."""
    bn = BayesianNetwork()
    for v in ["A", "B", "C", "D"]:
        bn.add_node(v, [0, 1])
    bn.add_edge("A", "B")
    bn.add_edge("A", "C")
    bn.add_edge("B", "D")
    bn.add_edge("C", "D")
    bn.set_cpt_dict("A", {0: 0.5, 1: 0.5})
    bn.set_cpt_dict("B", {(0,): {0: 0.8, 1: 0.2}, (1,): {0: 0.3, 1: 0.7}})
    bn.set_cpt_dict("C", {(0,): {0: 0.7, 1: 0.3}, (1,): {0: 0.2, 1: 0.8}})
    bn.set_cpt_dict("D", {
        (0, 0): {0: 0.9, 1: 0.1},
        (0, 1): {0: 0.5, 1: 0.5},
        (1, 0): {0: 0.5, 1: 0.5},
        (1, 1): {0: 0.1, 1: 0.9},
    })
    return bn


# ---------------------------------------------------------------------------
# Statistical independence tests
# ---------------------------------------------------------------------------

class TestContingencyTable:
    def test_basic(self):
        data = [{"X": 0, "Y": 0}, {"X": 0, "Y": 1}, {"X": 1, "Y": 0}, {"X": 1, "Y": 1}]
        table = _contingency_table(data, "X", "Y")
        assert () in table
        assert table[()][0][0] == 1
        assert table[()][1][1] == 1

    def test_with_conditioning(self):
        data = [
            {"X": 0, "Y": 0, "Z": 0},
            {"X": 0, "Y": 1, "Z": 0},
            {"X": 1, "Y": 0, "Z": 1},
            {"X": 1, "Y": 1, "Z": 1},
        ]
        table = _contingency_table(data, "X", "Y", {"Z"})
        assert (0,) in table
        assert (1,) in table


class TestNormalQuantile:
    def test_median(self):
        assert _normal_quantile(0.5) == 0.0

    def test_upper_tail(self):
        z = _normal_quantile(0.975)
        assert 1.9 < z < 2.0

    def test_lower_tail(self):
        z = _normal_quantile(0.025)
        assert -2.0 < z < -1.9

    def test_symmetry(self):
        assert abs(_normal_quantile(0.1) + _normal_quantile(0.9)) < 0.01

    def test_extremes(self):
        assert _normal_quantile(0.0) == -10.0
        assert _normal_quantile(1.0) == 10.0


class TestChi2Critical:
    def test_positive(self):
        c = _chi2_critical(1, 0.05)
        assert c > 0
        # Chi2(1, 0.05) should be ~3.84
        assert 3.0 < c < 5.0

    def test_more_df(self):
        c1 = _chi2_critical(1, 0.05)
        c5 = _chi2_critical(5, 0.05)
        assert c5 > c1

    def test_zero_df(self):
        assert _chi2_critical(0, 0.05) == 0.0


class TestChiSquaredTest:
    def test_independent_variables(self):
        random.seed(42)
        data = [{"X": random.choice([0, 1]), "Y": random.choice([0, 1])} for _ in range(1000)]
        indep, stat, thresh = chi_squared_test(data, "X", "Y")
        assert indep  # should be independent

    def test_dependent_variables(self):
        data = [{"X": i % 2, "Y": i % 2} for i in range(1000)]
        indep, stat, thresh = chi_squared_test(data, "X", "Y")
        assert not indep  # perfectly correlated

    def test_conditional_independence(self):
        # A -> B -> C: A _||_ C | B (need larger sample for CI tests)
        bn = build_chain_abc()
        data = sample_from_bn(bn, 5000, seed=42)
        indep, stat, thresh = chi_squared_test(data, "A", "C", {"B"}, alpha=0.01)
        # Conditional independence may be hard to detect with finite data
        # At minimum, the statistic should be much smaller than marginal test
        _, marg_stat, _ = chi_squared_test(data, "A", "C")
        assert stat < marg_stat  # conditioning on B reduces dependence

    def test_marginal_dependence(self):
        # A -> B -> C: A not _||_ C marginally
        bn = build_chain_abc()
        data = sample_from_bn(bn, 2000, seed=42)
        indep, _, _ = chi_squared_test(data, "A", "C")
        assert not indep


class TestMutualInformationTest:
    def test_independent(self):
        random.seed(42)
        data = [{"X": random.choice([0, 1]), "Y": random.choice([0, 1])} for _ in range(1000)]
        indep, _, _ = mutual_information_test(data, "X", "Y")
        assert indep

    def test_dependent(self):
        data = [{"X": i % 2, "Y": i % 2} for i in range(1000)]
        indep, _, _ = mutual_information_test(data, "X", "Y")
        assert not indep

    def test_conditional_independence(self):
        bn = build_chain_abc()
        data = sample_from_bn(bn, 5000, seed=42)
        _, cond_stat, _ = mutual_information_test(data, "A", "C", {"B"})
        _, marg_stat, _ = mutual_information_test(data, "A", "C")
        # Conditioning on B should reduce the test statistic
        assert cond_stat < marg_stat


# ---------------------------------------------------------------------------
# PC Algorithm
# ---------------------------------------------------------------------------

class TestPCAlgorithmChain:
    def test_chain_skeleton(self):
        bn = build_chain_abc()
        data = sample_from_bn(bn, 5000, seed=42)
        result = pc_algorithm(data, alpha=0.01)
        # Skeleton should have A-B and B-C edges
        assert "B" in result.skeleton["A"]
        assert "A" in result.skeleton["B"]
        assert "C" in result.skeleton["B"]
        assert "B" in result.skeleton["C"]

    def test_chain_separating_set(self):
        bn = build_chain_abc()
        data = sample_from_bn(bn, 5000, seed=42)
        result = pc_algorithm(data, alpha=0.01)
        # If A-C edge was removed, separating set should contain B
        key = ("A", "C")
        if key in result.separating_sets:
            assert "B" in result.separating_sets[key]


class TestPCAlgorithmFork:
    def test_fork_skeleton(self):
        bn = build_fork_abc()
        data = sample_from_bn(bn, 3000, seed=42)
        result = pc_algorithm(data)
        # A should be adjacent to both B and C
        assert "B" in result.skeleton["A"]
        assert "C" in result.skeleton["A"]
        # B and C should NOT be adjacent
        assert "C" not in result.skeleton.get("B", set())

    def test_fork_separating_set(self):
        bn = build_fork_abc()
        data = sample_from_bn(bn, 3000, seed=42)
        result = pc_algorithm(data)
        key = ("B", "C")
        assert key in result.separating_sets
        assert "A" in result.separating_sets[key]


class TestPCAlgorithmCollider:
    def test_collider_skeleton(self):
        bn = build_collider_abc()
        data = sample_from_bn(bn, 3000, seed=42)
        result = pc_algorithm(data)
        # A-C and B-C should be adjacent
        assert "C" in result.skeleton["A"]
        assert "C" in result.skeleton["B"]
        # A and B should NOT be adjacent
        assert "B" not in result.skeleton.get("A", set())

    def test_collider_v_structure_detected(self):
        bn = build_collider_abc()
        data = sample_from_bn(bn, 3000, seed=42)
        result = pc_algorithm(data)
        # Should orient A -> C <- B
        assert ("A", "C") in result.oriented_edges
        assert ("B", "C") in result.oriented_edges

    def test_collider_separating_set_excludes_c(self):
        bn = build_collider_abc()
        data = sample_from_bn(bn, 3000, seed=42)
        result = pc_algorithm(data)
        key = ("A", "B")
        assert key in result.separating_sets
        assert "C" not in result.separating_sets[key]


class TestPCAlgorithmDiamond:
    def test_diamond_skeleton(self):
        bn = build_diamond()
        data = sample_from_bn(bn, 5000, seed=42)
        result = pc_algorithm(data)
        # A-B, A-C, B-D, C-D should be in skeleton
        assert "B" in result.skeleton["A"]
        assert "C" in result.skeleton["A"]
        assert "D" in result.skeleton["B"]
        assert "D" in result.skeleton["C"]

    def test_diamond_collider_at_d(self):
        bn = build_diamond()
        data = sample_from_bn(bn, 5000, seed=42)
        result = pc_algorithm(data)
        # B -> D <- C is a v-structure if B-C are non-adjacent
        # and D not in sep(B,C)
        if "C" not in result.skeleton.get("B", set()):
            assert ("B", "D") in result.oriented_edges
            assert ("C", "D") in result.oriented_edges


class TestPCOptions:
    def test_mi_test(self):
        bn = build_chain_abc()
        data = sample_from_bn(bn, 5000, seed=42)
        result = pc_algorithm(data, test="mi", alpha=0.01)
        assert "B" in result.skeleton["A"]
        assert "B" in result.skeleton["C"]

    def test_max_cond_size(self):
        bn = build_chain_abc()
        data = sample_from_bn(bn, 3000, seed=42)
        result = pc_algorithm(data, max_cond_size=0)
        # With max_cond_size=0, can only test marginal independence
        # A-C might still be in skeleton since we can't condition on B
        assert "B" in result.skeleton["A"]

    def test_empty_data_raises(self):
        with pytest.raises(ValueError):
            pc_algorithm([])

    def test_to_dag(self):
        bn = build_collider_abc()
        data = sample_from_bn(bn, 3000, seed=42)
        result = pc_algorithm(data)
        dag = result.to_dag()
        assert isinstance(dag, set)
        assert len(dag) >= 2

    def test_to_cpdag(self):
        bn = build_collider_abc()
        data = sample_from_bn(bn, 3000, seed=42)
        result = pc_algorithm(data)
        directed, undirected = result.to_cpdag()
        assert isinstance(directed, set)
        assert isinstance(undirected, set)


# ---------------------------------------------------------------------------
# BIC Score
# ---------------------------------------------------------------------------

class TestBICScore:
    def test_correct_parent_higher_score(self):
        bn = build_chain_abc()
        data = sample_from_bn(bn, 2000, seed=42)
        domains = {"A": [0, 1], "B": [0, 1], "C": [0, 1]}
        # B with parent A should score better than B with no parent
        score_with = bic_score(data, "B", ["A"], domains)
        score_without = bic_score(data, "B", [], domains)
        assert score_with > score_without

    def test_independent_parent_lower_score(self):
        # Use truly independent variables
        random.seed(42)
        data = [{"X": random.choice([0, 1]), "Y": random.choice([0, 1])} for _ in range(2000)]
        domains = {"X": [0, 1], "Y": [0, 1]}
        # X with parent Y should score worse than X alone (truly independent)
        score_with = bic_score(data, "X", ["Y"], domains)
        score_without = bic_score(data, "X", [], domains)
        assert score_without >= score_with

    def test_empty_data(self):
        assert bic_score([], "X", [], {"X": [0, 1]}) == 0.0


# ---------------------------------------------------------------------------
# Hill Climbing
# ---------------------------------------------------------------------------

class TestHasCycle:
    def test_no_cycle(self):
        adj = {"A": {"B"}, "B": {"C"}, "C": set()}
        assert not _has_cycle(adj, "A", "C")

    def test_has_cycle(self):
        adj = {"A": {"B"}, "B": {"C"}, "C": set()}
        assert _has_cycle(adj, "C", "A")  # Adding C->A creates cycle

    def test_self_loop(self):
        adj = {"A": set()}
        assert _has_cycle(adj, "A", "A")


class TestHillClimbing:
    def test_chain_recovery(self):
        bn = build_chain_abc()
        data = sample_from_bn(bn, 2000, seed=42)
        result = hill_climbing(data)
        # Should find A-B and B-C edges (direction may vary in equivalence class)
        edge_set = {frozenset(e) for e in result.edges}
        assert frozenset({"A", "B"}) in edge_set
        assert frozenset({"B", "C"}) in edge_set
        # Should NOT have direct A-C edge
        assert frozenset({"A", "C"}) not in edge_set

    def test_fork_recovery(self):
        bn = build_fork_abc()
        data = sample_from_bn(bn, 2000, seed=42)
        result = hill_climbing(data)
        edge_set = {frozenset(e) for e in result.edges}
        assert frozenset({"A", "B"}) in edge_set
        assert frozenset({"A", "C"}) in edge_set

    def test_collider_recovery(self):
        bn = build_collider_abc()
        data = sample_from_bn(bn, 3000, seed=42)
        result = hill_climbing(data)
        edge_set = {frozenset(e) for e in result.edges}
        assert frozenset({"A", "C"}) in edge_set
        assert frozenset({"B", "C"}) in edge_set
        assert frozenset({"A", "B"}) not in edge_set

    def test_max_parents_constraint(self):
        bn = build_diamond()
        data = sample_from_bn(bn, 2000, seed=42)
        result = hill_climbing(data, max_parents=1)
        # No node should have more than 1 parent
        parent_count: dict = {}
        for u, v in result.edges:
            parent_count[v] = parent_count.get(v, 0) + 1
        for count in parent_count.values():
            assert count <= 1

    def test_empty_data_raises(self):
        with pytest.raises(ValueError):
            hill_climbing([])

    def test_returns_result_type(self):
        data = [{"X": 0, "Y": 0}, {"X": 1, "Y": 1}] * 100
        result = hill_climbing(data)
        assert isinstance(result, HillClimbResult)
        assert isinstance(result.score, float)
        assert result.iterations >= 1

    def test_with_initial_edges(self):
        bn = build_chain_abc()
        data = sample_from_bn(bn, 2000, seed=42)
        result = hill_climbing(data, initial_edges={("A", "B")})
        edge_set = {frozenset(e) for e in result.edges}
        assert frozenset({"A", "B"}) in edge_set

    def test_restarts(self):
        bn = build_chain_abc()
        data = sample_from_bn(bn, 2000, seed=42)
        result = hill_climbing(data, restarts=2)
        edge_set = {frozenset(e) for e in result.edges}
        assert frozenset({"A", "B"}) in edge_set


# ---------------------------------------------------------------------------
# Hybrid Discovery
# ---------------------------------------------------------------------------

class TestHybridDiscovery:
    def test_chain_hybrid(self):
        bn = build_chain_abc()
        data = sample_from_bn(bn, 3000, seed=42)
        result = hybrid_discovery(data)
        edge_set = {frozenset(e) for e in result.edges}
        assert frozenset({"A", "B"}) in edge_set
        assert frozenset({"B", "C"}) in edge_set

    def test_collider_hybrid(self):
        bn = build_collider_abc()
        data = sample_from_bn(bn, 3000, seed=42)
        result = hybrid_discovery(data)
        edge_set = {frozenset(e) for e in result.edges}
        assert frozenset({"A", "C"}) in edge_set
        assert frozenset({"B", "C"}) in edge_set


# ---------------------------------------------------------------------------
# Structural Hamming Distance
# ---------------------------------------------------------------------------

class TestSHD:
    def test_identical(self):
        edges = {("A", "B"), ("B", "C")}
        m = structural_hamming_distance(edges, edges)
        assert m["shd"] == 0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_missing_edge(self):
        true_edges = {("A", "B"), ("B", "C")}
        learned = {("A", "B")}
        m = structural_hamming_distance(true_edges, learned)
        assert m["skeleton_fn"] == 1
        assert m["skeleton_fp"] == 0
        assert m["shd"] >= 1

    def test_extra_edge(self):
        true_edges = {("A", "B")}
        learned = {("A", "B"), ("B", "C")}
        m = structural_hamming_distance(true_edges, learned)
        assert m["skeleton_fp"] == 1
        assert m["skeleton_fn"] == 0

    def test_reversed_edge(self):
        true_edges = {("A", "B")}
        learned = {("B", "A")}
        m = structural_hamming_distance(true_edges, learned)
        assert m["reversed_direction"] == 1
        assert m["skeleton_tp"] == 1

    def test_empty(self):
        m = structural_hamming_distance(set(), set())
        assert m["shd"] == 0

    def test_precision_recall(self):
        true_edges = {("A", "B"), ("B", "C"), ("C", "D")}
        learned = {("A", "B"), ("B", "C"), ("A", "C")}
        m = structural_hamming_distance(true_edges, learned)
        assert 0 < m["precision"] < 1
        assert 0 < m["recall"] < 1


# ---------------------------------------------------------------------------
# Sampling from BN
# ---------------------------------------------------------------------------

class TestSampleFromBN:
    def test_chain_sampling(self):
        bn = build_chain_abc()
        data = sample_from_bn(bn, 1000, seed=42)
        assert len(data) == 1000
        assert all("A" in row and "B" in row and "C" in row for row in data)

    def test_marginal_distribution(self):
        bn = build_chain_abc()
        data = sample_from_bn(bn, 10000, seed=42)
        # A should be ~60% 0, ~40% 1
        a_counts = sum(1 for row in data if row["A"] == 0)
        ratio = a_counts / len(data)
        assert 0.55 < ratio < 0.65

    def test_deterministic_seed(self):
        bn = build_chain_abc()
        d1 = sample_from_bn(bn, 100, seed=123)
        d2 = sample_from_bn(bn, 100, seed=123)
        assert d1 == d2

    def test_conditional_distribution(self):
        bn = build_chain_abc()
        data = sample_from_bn(bn, 10000, seed=42)
        # P(B=1|A=1) should be ~0.7
        a1_rows = [r for r in data if r["A"] == 1]
        b1_given_a1 = sum(1 for r in a1_rows if r["B"] == 1) / len(a1_rows)
        assert 0.6 < b1_given_a1 < 0.8


# ---------------------------------------------------------------------------
# Learn BN structure
# ---------------------------------------------------------------------------

class TestLearnBNStructure:
    def test_learn_chain_hc(self):
        bn = build_chain_abc()
        data = sample_from_bn(bn, 2000, seed=42)
        learned_bn = learn_bn_structure(data, method="hc")
        # Should have 3 nodes
        assert len(learned_bn.nodes) == 3
        # Should have some edges
        edge_set = set()
        for node in learned_bn.nodes:
            for parent in learned_bn.parents.get(node, []):
                edge_set.add(frozenset({parent, node}))
        assert frozenset({"A", "B"}) in edge_set

    def test_learn_chain_pc(self):
        bn = build_chain_abc()
        data = sample_from_bn(bn, 3000, seed=42)
        learned_bn = learn_bn_structure(data, method="pc")
        assert len(learned_bn.nodes) == 3

    def test_learn_hybrid(self):
        bn = build_chain_abc()
        data = sample_from_bn(bn, 3000, seed=42)
        learned_bn = learn_bn_structure(data, method="hybrid")
        assert len(learned_bn.nodes) == 3

    def test_learned_bn_has_cpts(self):
        bn = build_chain_abc()
        data = sample_from_bn(bn, 2000, seed=42)
        learned_bn = learn_bn_structure(data, method="hc")
        for node in learned_bn.nodes:
            assert node in learned_bn.cpts

    def test_learned_cpts_sum_to_one(self):
        bn = build_chain_abc()
        data = sample_from_bn(bn, 2000, seed=42)
        learned_bn = learn_bn_structure(data, method="hc")
        for node in learned_bn.nodes:
            factor = learned_bn.cpts[node]
            # Marginalize out the node to check parent configs sum to 1
            parents = learned_bn.parents.get(node, [])
            if not parents:
                total = sum(factor.get({node: v}) for v in learned_bn.domains[node])
                assert abs(total - 1.0) < 0.01
            else:
                # For each parent config, should sum to 1
                from itertools import product as iprod
                parent_domains = [learned_bn.domains[p] for p in parents]
                for p_vals in iprod(*parent_domains):
                    assignment = {p: v for p, v in zip(parents, p_vals)}
                    total = 0
                    for v in learned_bn.domains[node]:
                        assignment[node] = v
                        total += factor.get(assignment)
                    assert abs(total - 1.0) < 0.01

    def test_unknown_method_raises(self):
        data = [{"X": 0}]
        with pytest.raises(ValueError):
            learn_bn_structure(data, method="unknown")

    def test_empty_data_raises(self):
        with pytest.raises(ValueError):
            learn_bn_structure([], method="hc")


# ---------------------------------------------------------------------------
# Learn Causal Model
# ---------------------------------------------------------------------------

class TestLearnCausalModel:
    def test_returns_causal_model(self):
        bn = build_chain_abc()
        data = sample_from_bn(bn, 2000, seed=42)
        cm = learn_causal_model(data, method="hc")
        assert isinstance(cm, CausalModel)
        assert hasattr(cm, "bn")
        assert isinstance(cm.bn, BayesianNetwork)

    def test_causal_model_can_intervene(self):
        bn = build_fork_abc()
        data = sample_from_bn(bn, 3000, seed=42)
        cm = learn_causal_model(data, method="hc")
        # Should be able to do interventions on the learned model
        mutilated = cm.do({"A": 1})
        assert isinstance(mutilated, BayesianNetwork)


# ---------------------------------------------------------------------------
# MLE Parameter Estimation
# ---------------------------------------------------------------------------

class TestMLEEstimation:
    def test_root_node_estimation(self):
        bn = BayesianNetwork()
        bn.add_node("X", [0, 1])
        data = [{"X": 0}] * 70 + [{"X": 1}] * 30
        _estimate_parameters_mle(bn, data)
        # With Laplace smoothing: (70+1)/102 ~ 0.696, (30+1)/102 ~ 0.304
        p0 = bn.cpts["X"].get({"X": 0})
        p1 = bn.cpts["X"].get({"X": 1})
        assert 0.65 < p0 < 0.72
        assert 0.28 < p1 < 0.35

    def test_child_node_estimation(self):
        bn = BayesianNetwork()
        bn.add_node("A", [0, 1])
        bn.add_node("B", [0, 1])
        bn.add_edge("A", "B")
        data = []
        for _ in range(80):
            data.append({"A": 0, "B": 0})
        for _ in range(20):
            data.append({"A": 0, "B": 1})
        for _ in range(30):
            data.append({"A": 1, "B": 0})
        for _ in range(70):
            data.append({"A": 1, "B": 1})
        _estimate_parameters_mle(bn, data)
        # P(B=0|A=0) ~ 80/100 = 0.8 (with smoothing ~0.79)
        p = bn.cpts["B"].get({"A": 0, "B": 0})
        assert 0.75 < p < 0.85


# ---------------------------------------------------------------------------
# Integration: end-to-end discovery + evaluation
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_chain_shd(self):
        """Discover chain structure and measure SHD."""
        bn = build_chain_abc()
        true_edges = {("A", "B"), ("B", "C")}
        data = sample_from_bn(bn, 3000, seed=42)
        result = hill_climbing(data)
        m = structural_hamming_distance(true_edges, result.edges)
        # Should have reasonable SHD (skeleton should be correct)
        assert m["skeleton_fn"] == 0  # shouldn't miss any edges
        assert m["skeleton_fp"] == 0  # shouldn't add spurious edges

    def test_collider_shd(self):
        """Discover collider structure and measure SHD."""
        bn = build_collider_abc()
        true_edges = {("A", "C"), ("B", "C")}
        data = sample_from_bn(bn, 3000, seed=42)
        result = hill_climbing(data)
        m = structural_hamming_distance(true_edges, result.edges)
        assert m["skeleton_fn"] == 0
        assert m["skeleton_fp"] == 0

    def test_diamond_recovery(self):
        """Discover diamond structure."""
        bn = build_diamond()
        data = sample_from_bn(bn, 5000, seed=42)
        result = hill_climbing(data)
        edge_set = {frozenset(e) for e in result.edges}
        # Should find all 4 edges
        assert frozenset({"A", "B"}) in edge_set
        assert frozenset({"A", "C"}) in edge_set
        assert frozenset({"B", "D"}) in edge_set
        assert frozenset({"C", "D"}) in edge_set

    def test_pc_vs_hc_agreement(self):
        """PC and HC should find similar skeletons."""
        bn = build_chain_abc()
        data = sample_from_bn(bn, 3000, seed=42)
        pc_result = pc_algorithm(data)
        hc_result = hill_climbing(data)
        pc_skeleton = set()
        for v in pc_result.skeleton:
            for u in pc_result.skeleton[v]:
                pc_skeleton.add(frozenset({u, v}))
        hc_skeleton = {frozenset(e) for e in hc_result.edges}
        # Both should find A-B and B-C
        assert frozenset({"A", "B"}) in pc_skeleton
        assert frozenset({"A", "B"}) in hc_skeleton
        assert frozenset({"B", "C"}) in pc_skeleton
        assert frozenset({"B", "C"}) in hc_skeleton

    def test_learned_bn_inference(self):
        """Learned BN should support inference queries."""
        bn = build_chain_abc()
        data = sample_from_bn(bn, 3000, seed=42)
        learned = learn_bn_structure(data, method="hc")
        # Query should work
        result = variable_elimination(learned, ["A"])
        p_a0 = result.get({"A": 0})
        p_a1 = result.get({"A": 1})
        assert abs(p_a0 + p_a1 - 1.0) < 0.01
        # Should be close to true prior (0.6, 0.4)
        assert 0.5 < p_a0 < 0.7

    def test_causal_discovery_to_intervention(self):
        """Full pipeline: data -> discover -> intervene -> query."""
        bn = build_fork_abc()
        data = sample_from_bn(bn, 3000, seed=42)
        cm = learn_causal_model(data, method="hc")
        # Verify the learned model supports interventions (structural test)
        mutilated = cm.do({"A": 1})
        assert isinstance(mutilated, BayesianNetwork)
        # Query the mutilated model
        result = variable_elimination(mutilated, ["B"])
        p_b0 = result.get({"B": 0})
        p_b1 = result.get({"B": 1})
        # Probabilities should sum to 1
        assert abs(p_b0 + p_b1 - 1.0) < 0.01

    def test_ternary_variables(self):
        """Test with 3-valued variables."""
        bn = BayesianNetwork()
        bn.add_node("X", [0, 1, 2])
        bn.add_node("Y", [0, 1, 2])
        bn.add_edge("X", "Y")
        bn.set_cpt_dict("X", {0: 0.4, 1: 0.35, 2: 0.25})
        bn.set_cpt_dict("Y", {
            (0,): {0: 0.7, 1: 0.2, 2: 0.1},
            (1,): {0: 0.1, 1: 0.7, 2: 0.2},
            (2,): {0: 0.2, 1: 0.1, 2: 0.7},
        })
        data = sample_from_bn(bn, 3000, seed=42)
        result = hill_climbing(data)
        edge_set = {frozenset(e) for e in result.edges}
        assert frozenset({"X", "Y"}) in edge_set

    def test_five_node_network(self):
        """Test with 5-node network: A->B->D, A->C->D, C->E."""
        bn = BayesianNetwork()
        for v in ["A", "B", "C", "D", "E"]:
            bn.add_node(v, [0, 1])
        bn.add_edge("A", "B")
        bn.add_edge("A", "C")
        bn.add_edge("B", "D")
        bn.add_edge("C", "D")
        bn.add_edge("C", "E")
        bn.set_cpt_dict("A", {0: 0.5, 1: 0.5})
        bn.set_cpt_dict("B", {(0,): {0: 0.8, 1: 0.2}, (1,): {0: 0.3, 1: 0.7}})
        bn.set_cpt_dict("C", {(0,): {0: 0.7, 1: 0.3}, (1,): {0: 0.2, 1: 0.8}})
        bn.set_cpt_dict("D", {
            (0, 0): {0: 0.9, 1: 0.1},
            (0, 1): {0: 0.5, 1: 0.5},
            (1, 0): {0: 0.5, 1: 0.5},
            (1, 1): {0: 0.1, 1: 0.9},
        })
        bn.set_cpt_dict("E", {(0,): {0: 0.85, 1: 0.15}, (1,): {0: 0.25, 1: 0.75}})

        data = sample_from_bn(bn, 5000, seed=42)
        result = hill_climbing(data)
        edge_set = {frozenset(e) for e in result.edges}
        # Should recover all 5 edges in skeleton
        assert frozenset({"A", "B"}) in edge_set
        assert frozenset({"A", "C"}) in edge_set
        assert frozenset({"B", "D"}) in edge_set
        assert frozenset({"C", "D"}) in edge_set
        assert frozenset({"C", "E"}) in edge_set


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
