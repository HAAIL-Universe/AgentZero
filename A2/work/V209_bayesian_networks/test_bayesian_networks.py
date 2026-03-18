"""Tests for V209: Bayesian Network Inference."""

import pytest
import sys
import os
from math import log, isclose

sys.path.insert(0, os.path.dirname(__file__))
from bayesian_networks import (
    Factor, BayesianNetwork, variable_elimination, map_inference,
    JunctionTree, conditional_independence, build_chain, build_naive_bayes,
    most_probable_explanation, mutual_information, sensitivity_analysis,
)

ABS_TOL = 1e-6


# ===== Factor Tests =====

class TestFactor:
    def test_create_factor(self):
        f = Factor(["A"], {"A": [0, 1]}, {(0,): 0.3, (1,): 0.7})
        assert f.get({"A": 0}) == pytest.approx(0.3)
        assert f.get({"A": 1}) == pytest.approx(0.7)

    def test_multiply_factors(self):
        f1 = Factor(["A"], {"A": [0, 1]}, {(0,): 0.4, (1,): 0.6})
        f2 = Factor(["A", "B"], {"A": [0, 1], "B": [0, 1]},
                     {(0, 0): 0.2, (0, 1): 0.8, (1, 0): 0.5, (1, 1): 0.5})
        result = f1.multiply(f2)
        assert "A" in result.variables and "B" in result.variables
        assert result.get({"A": 0, "B": 0}) == pytest.approx(0.08)
        assert result.get({"A": 0, "B": 1}) == pytest.approx(0.32)
        assert result.get({"A": 1, "B": 0}) == pytest.approx(0.30)
        assert result.get({"A": 1, "B": 1}) == pytest.approx(0.30)

    def test_marginalize(self):
        f = Factor(["A", "B"], {"A": [0, 1], "B": [0, 1]},
                    {(0, 0): 0.1, (0, 1): 0.2, (1, 0): 0.3, (1, 1): 0.4})
        m = f.marginalize("B")
        assert m.variables == ["A"]
        assert m.get({"A": 0}) == pytest.approx(0.3)
        assert m.get({"A": 1}) == pytest.approx(0.7)

    def test_reduce(self):
        f = Factor(["A", "B"], {"A": [0, 1], "B": [0, 1]},
                    {(0, 0): 0.1, (0, 1): 0.2, (1, 0): 0.3, (1, 1): 0.4})
        r = f.reduce("B", 1)
        assert r.variables == ["A"]
        assert r.get({"A": 0}) == pytest.approx(0.2)
        assert r.get({"A": 1}) == pytest.approx(0.4)

    def test_normalize(self):
        f = Factor(["A"], {"A": [0, 1]}, {(0,): 3.0, (1,): 7.0})
        n = f.normalize()
        assert n.get({"A": 0}) == pytest.approx(0.3)
        assert n.get({"A": 1}) == pytest.approx(0.7)

    def test_max_assignment(self):
        f = Factor(["A", "B"], {"A": [0, 1], "B": [0, 1]},
                    {(0, 0): 0.1, (0, 1): 0.2, (1, 0): 0.05, (1, 1): 0.65})
        assign, prob = f.max_assignment()
        assert assign == {"A": 1, "B": 1}
        assert prob == pytest.approx(0.65)

    def test_entropy(self):
        f = Factor(["A"], {"A": [0, 1]}, {(0,): 0.5, (1,): 0.5})
        h = f.entropy()
        assert h == pytest.approx(log(2))

    def test_entropy_deterministic(self):
        f = Factor(["A"], {"A": [0, 1]}, {(0,): 1.0, (1,): 0.0})
        assert f.entropy() == pytest.approx(0.0)

    def test_kl_divergence_same(self):
        f = Factor(["A"], {"A": [0, 1]}, {(0,): 0.5, (1,): 0.5})
        assert f.kl_divergence(f) == pytest.approx(0.0)

    def test_kl_divergence_different(self):
        p = Factor(["A"], {"A": [0, 1]}, {(0,): 0.3, (1,): 0.7})
        q = Factor(["A"], {"A": [0, 1]}, {(0,): 0.5, (1,): 0.5})
        kl = p.kl_divergence(q)
        expected = 0.3 * log(0.3 / 0.5) + 0.7 * log(0.7 / 0.5)
        assert kl == pytest.approx(expected)

    def test_reduce_nonexistent_variable(self):
        f = Factor(["A"], {"A": [0, 1]}, {(0,): 0.5, (1,): 0.5})
        r = f.reduce("B", 0)
        assert r.variables == ["A"]

    def test_marginalize_nonexistent_variable(self):
        f = Factor(["A"], {"A": [0, 1]}, {(0,): 0.5, (1,): 0.5})
        m = f.marginalize("B")
        assert m.variables == ["A"]

    def test_empty_factor(self):
        f = Factor([], {})
        assert f._all_assignments() == [()]

    def test_multiply_disjoint_variables(self):
        f1 = Factor(["A"], {"A": [0, 1]}, {(0,): 0.4, (1,): 0.6})
        f2 = Factor(["B"], {"B": [0, 1]}, {(0,): 0.3, (1,): 0.7})
        result = f1.multiply(f2)
        assert set(result.variables) == {"A", "B"}
        assert result.get({"A": 0, "B": 0}) == pytest.approx(0.12)
        assert result.get({"A": 1, "B": 1}) == pytest.approx(0.42)


# ===== Bayesian Network Construction =====

def make_alarm_network():
    """Classic alarm network: Burglary -> Alarm <- Earthquake, Alarm -> JohnCalls, Alarm -> MaryCalls."""
    bn = BayesianNetwork()
    bn.add_node("B", [True, False])
    bn.add_node("E", [True, False])
    bn.add_node("A", [True, False])
    bn.add_node("J", [True, False])
    bn.add_node("M", [True, False])

    bn.add_edge("B", "A")
    bn.add_edge("E", "A")
    bn.add_edge("A", "J")
    bn.add_edge("A", "M")

    bn.set_cpt_dict("B", {True: 0.001, False: 0.999})
    bn.set_cpt_dict("E", {True: 0.002, False: 0.998})
    bn.set_cpt_dict("A", {
        (True, True): {True: 0.95, False: 0.05},
        (True, False): {True: 0.94, False: 0.06},
        (False, True): {True: 0.29, False: 0.71},
        (False, False): {True: 0.001, False: 0.999},
    })
    bn.set_cpt_dict("J", {
        True: {True: 0.90, False: 0.10},
        False: {True: 0.05, False: 0.95},
    })
    bn.set_cpt_dict("M", {
        True: {True: 0.70, False: 0.30},
        False: {True: 0.01, False: 0.99},
    })
    return bn


def make_simple_chain():
    """A -> B -> C with binary variables."""
    bn = BayesianNetwork()
    bn.add_node("A", [0, 1])
    bn.add_node("B", [0, 1])
    bn.add_node("C", [0, 1])
    bn.add_edge("A", "B")
    bn.add_edge("B", "C")
    bn.set_cpt_dict("A", {0: 0.6, 1: 0.4})
    bn.set_cpt_dict("B", {
        0: {0: 0.7, 1: 0.3},
        1: {0: 0.2, 1: 0.8},
    })
    bn.set_cpt_dict("C", {
        0: {0: 0.9, 1: 0.1},
        1: {0: 0.4, 1: 0.6},
    })
    return bn


class TestBayesianNetwork:
    def test_add_nodes_and_edges(self):
        bn = make_alarm_network()
        assert len(bn.nodes) == 5
        assert "B" in bn.parents["A"]
        assert "E" in bn.parents["A"]
        assert "A" in bn.children["B"]

    def test_duplicate_node_error(self):
        bn = BayesianNetwork()
        bn.add_node("A", [0, 1])
        with pytest.raises(ValueError):
            bn.add_node("A", [0, 1])

    def test_edge_missing_node_error(self):
        bn = BayesianNetwork()
        bn.add_node("A", [0, 1])
        with pytest.raises(ValueError):
            bn.add_edge("A", "B")

    def test_topological_sort(self):
        bn = make_alarm_network()
        order = bn.topological_sort()
        idx = {n: i for i, n in enumerate(order)}
        assert idx["B"] < idx["A"]
        assert idx["E"] < idx["A"]
        assert idx["A"] < idx["J"]
        assert idx["A"] < idx["M"]

    def test_ancestors(self):
        bn = make_alarm_network()
        assert bn.ancestors("J") == {"A", "B", "E"}
        assert bn.ancestors("B") == set()

    def test_descendants(self):
        bn = make_alarm_network()
        assert bn.descendants("B") == {"A", "J", "M"}
        assert bn.descendants("J") == set()

    def test_markov_blanket(self):
        bn = make_alarm_network()
        mb = bn.markov_blanket("A")
        assert mb == {"B", "E", "J", "M"}

    def test_markov_blanket_root(self):
        bn = make_alarm_network()
        mb = bn.markov_blanket("B")
        # B's children: A, co-parents of A: E
        assert mb == {"A", "E"}


# ===== Variable Elimination =====

class TestVariableElimination:
    def test_prior_marginal(self):
        bn = make_simple_chain()
        result = variable_elimination(bn, ["A"])
        assert result.get({"A": 0}) == pytest.approx(0.6)
        assert result.get({"A": 1}) == pytest.approx(0.4)

    def test_chain_marginal_C(self):
        bn = make_simple_chain()
        # P(C=0) = P(C=0|B=0)P(B=0) + P(C=0|B=1)P(B=1)
        # P(B=0) = 0.6*0.7 + 0.4*0.2 = 0.50
        # P(B=1) = 0.6*0.3 + 0.4*0.8 = 0.50
        # P(C=0) = 0.9*0.50 + 0.4*0.50 = 0.65
        result = variable_elimination(bn, ["C"])
        assert result.get({"C": 0}) == pytest.approx(0.65)
        assert result.get({"C": 1}) == pytest.approx(0.35)

    def test_posterior_with_evidence(self):
        bn = make_simple_chain()
        # P(A|C=1)
        result = variable_elimination(bn, ["A"], {"C": 1})
        # By Bayes: P(A=0|C=1) = P(C=1|A=0)*P(A=0) / P(C=1)
        # P(C=1|A=0) = P(C=1|B=0)P(B=0|A=0) + P(C=1|B=1)P(B=1|A=0)
        #            = 0.1*0.7 + 0.6*0.3 = 0.25
        # P(C=1|A=1) = 0.1*0.2 + 0.6*0.8 = 0.50
        # P(C=1) = 0.25*0.6 + 0.50*0.4 = 0.35
        # P(A=0|C=1) = 0.25*0.6 / 0.35 = 3/7
        assert result.get({"A": 0}) == pytest.approx(3 / 7)
        assert result.get({"A": 1}) == pytest.approx(4 / 7)

    def test_alarm_prior(self):
        bn = make_alarm_network()
        result = variable_elimination(bn, ["B"])
        assert result.get({"B": True}) == pytest.approx(0.001)
        assert result.get({"B": False}) == pytest.approx(0.999)

    def test_alarm_posterior_john_calls(self):
        bn = make_alarm_network()
        # P(B|J=True)
        result = variable_elimination(bn, ["B"], {"J": True})
        # P(B=True|J=True) should increase from prior 0.001
        assert result.get({"B": True}) > 0.001

    def test_alarm_posterior_john_and_mary(self):
        bn = make_alarm_network()
        # P(B|J=True, M=True) -- both neighbors called
        result = variable_elimination(bn, ["B"], {"J": True, "M": True})
        p_burglary = result.get({"B": True})
        # Should be much higher than just John calling
        result2 = variable_elimination(bn, ["B"], {"J": True})
        assert p_burglary > result2.get({"B": True})

    def test_joint_query(self):
        bn = make_simple_chain()
        result = variable_elimination(bn, ["A", "B"])
        # P(A=0, B=0) = P(B=0|A=0)P(A=0) = 0.7 * 0.6 = 0.42
        assert result.get({"A": 0, "B": 0}) == pytest.approx(0.42)

    def test_explaining_away(self):
        """Classic explaining-away: observing Alarm makes B and E dependent."""
        bn = make_alarm_network()
        # Without alarm observation, B and E are independent
        p_b_no_ev = variable_elimination(bn, ["B"])
        p_b_alarm = variable_elimination(bn, ["B"], {"A": True})
        # With alarm and earthquake, burglary becomes less likely (explained away)
        p_b_alarm_eq = variable_elimination(bn, ["B"], {"A": True, "E": True})
        assert p_b_alarm_eq.get({"B": True}) < p_b_alarm.get({"B": True})

    def test_custom_elimination_order(self):
        bn = make_simple_chain()
        result1 = variable_elimination(bn, ["C"], elimination_order=["A", "B"])
        result2 = variable_elimination(bn, ["C"], elimination_order=["B", "A"])
        assert result1.get({"C": 0}) == pytest.approx(result2.get({"C": 0}))

    def test_all_evidence_single_var(self):
        bn = make_simple_chain()
        # Everything observed except query
        result = variable_elimination(bn, ["C"], {"A": 0, "B": 1})
        assert result.get({"C": 0}) == pytest.approx(0.4)
        assert result.get({"C": 1}) == pytest.approx(0.6)


# ===== MAP Inference =====

class TestMAPInference:
    def test_map_no_evidence(self):
        bn = make_simple_chain()
        assignment, prob = map_inference(bn)
        # Most probable: A=0 (0.6), B=0 (0.7|A=0), C=0 (0.9|B=0)
        assert assignment["A"] == 0
        assert assignment["B"] == 0
        assert assignment["C"] == 0

    def test_map_with_evidence(self):
        bn = make_simple_chain()
        assignment, prob = map_inference(bn, {"C": 1})
        # Given C=1, most probable A,B?
        # A=1,B=1 gives P(C=1|B=1)=0.6 vs A=0,B=0 gives P(C=1|B=0)=0.1
        assert assignment["B"] == 1

    def test_map_specific_vars(self):
        bn = make_simple_chain()
        assignment, prob = map_inference(bn, map_vars=["B"])
        # P(B=0) = 0.50, P(B=1) = 0.50 -- equal marginals
        assert assignment["B"] in [0, 1]

    def test_map_alarm(self):
        bn = make_alarm_network()
        assignment, prob = map_inference(bn, {"J": True, "M": True},
                                          map_vars=["B", "E", "A"])
        # Most probable: Alarm=True, but B and E priors very low
        # So MAP should have A=True (both called) but B=False, E=False
        # Actually it depends on exact numbers
        assert assignment["A"] == True


# ===== D-Separation =====

class TestDSeparation:
    def test_chain_blocked(self):
        bn = make_simple_chain()
        # A -- B -- C: A and C are d-separated given B
        assert bn.is_d_separated({"A"}, {"C"}, {"B"}) == True

    def test_chain_unblocked(self):
        bn = make_simple_chain()
        # A and C are NOT d-separated with no evidence
        assert bn.is_d_separated({"A"}, {"C"}, set()) == False

    def test_fork_blocked(self):
        bn = BayesianNetwork()
        bn.add_node("A", [0, 1])
        bn.add_node("B", [0, 1])
        bn.add_node("C", [0, 1])
        bn.add_edge("B", "A")
        bn.add_edge("B", "C")
        # A <- B -> C: A and C are d-separated given B
        assert bn.is_d_separated({"A"}, {"C"}, {"B"}) == True

    def test_fork_unblocked(self):
        bn = BayesianNetwork()
        bn.add_node("A", [0, 1])
        bn.add_node("B", [0, 1])
        bn.add_node("C", [0, 1])
        bn.add_edge("B", "A")
        bn.add_edge("B", "C")
        assert bn.is_d_separated({"A"}, {"C"}, set()) == False

    def test_collider_blocked(self):
        bn = make_alarm_network()
        # B -> A <- E: B and E are d-separated without evidence on A
        assert bn.is_d_separated({"B"}, {"E"}, set()) == True

    def test_collider_unblocked(self):
        bn = make_alarm_network()
        # B -> A <- E: B and E are NOT d-separated given A (v-structure)
        assert bn.is_d_separated({"B"}, {"E"}, {"A"}) == False

    def test_collider_descendant_unblocks(self):
        bn = make_alarm_network()
        # Observing J (descendant of A) also unblocks B-E
        assert bn.is_d_separated({"B"}, {"E"}, {"J"}) == False

    def test_conditional_independence_helper(self):
        bn = make_simple_chain()
        assert conditional_independence(bn, "A", "C", {"B"}) == True
        assert conditional_independence(bn, "A", "C") == False


# ===== Junction Tree =====

class TestJunctionTree:
    def test_build_junction_tree(self):
        bn = make_alarm_network()
        jt = JunctionTree(bn)
        assert len(jt.cliques) > 0
        # Every node should be in at least one clique
        all_vars_in_cliques = set()
        for c in jt.cliques:
            all_vars_in_cliques |= c
        for node in bn.nodes:
            assert node in all_vars_in_cliques

    def test_junction_tree_query(self):
        bn = make_simple_chain()
        jt = JunctionTree(bn)
        jt.calibrate()
        result = jt.query(["A"])
        assert result.get({"A": 0}) == pytest.approx(0.6, abs=0.01)
        assert result.get({"A": 1}) == pytest.approx(0.4, abs=0.01)

    def test_junction_tree_with_evidence(self):
        bn = make_simple_chain()
        jt = JunctionTree(bn)
        jt.calibrate({"C": 1})
        result = jt.query(["A"])
        # Should match VE result
        ve_result = variable_elimination(bn, ["A"], {"C": 1})
        assert result.get({"A": 0}) == pytest.approx(
            ve_result.get({"A": 0}), abs=0.01)

    def test_junction_tree_alarm(self):
        bn = make_alarm_network()
        jt = JunctionTree(bn)
        jt.calibrate({"J": True, "M": True})
        result = jt.query(["B"])
        ve_result = variable_elimination(bn, ["B"], {"J": True, "M": True})
        assert result.get({"B": True}) == pytest.approx(
            ve_result.get({"B": True}), abs=0.01)

    def test_junction_tree_single_node(self):
        bn = BayesianNetwork()
        bn.add_node("A", [0, 1])
        bn.set_cpt_dict("A", {0: 0.3, 1: 0.7})
        jt = JunctionTree(bn)
        jt.calibrate()
        result = jt.query(["A"])
        assert result.get({"A": 0}) == pytest.approx(0.3)


# ===== Builder Helpers =====

class TestBuilders:
    def test_build_chain(self):
        bn = build_chain(
            ["A", "B", "C"],
            [[0, 1], [0, 1], [0, 1]],
            [
                {0: 0.5, 1: 0.5},
                {0: {0: 0.8, 1: 0.2}, 1: {0: 0.3, 1: 0.7}},
                {0: {0: 0.9, 1: 0.1}, 1: {0: 0.4, 1: 0.6}},
            ]
        )
        assert len(bn.nodes) == 3
        result = variable_elimination(bn, ["A"])
        assert result.get({"A": 0}) == pytest.approx(0.5)

    def test_build_naive_bayes(self):
        bn = build_naive_bayes(
            "C", ["spam", "ham"],
            {"spam": 0.3, "ham": 0.7},
            ["W1", "W2"],
            [["yes", "no"], ["yes", "no"]],
            [
                {"spam": {"yes": 0.8, "no": 0.2}, "ham": {"yes": 0.1, "no": 0.9}},
                {"spam": {"yes": 0.6, "no": 0.4}, "ham": {"yes": 0.2, "no": 0.8}},
            ]
        )
        # P(spam | W1=yes, W2=yes) should be high
        result = variable_elimination(bn, ["C"], {"W1": "yes", "W2": "yes"})
        assert result.get({"C": "spam"}) > result.get({"C": "ham"})

    def test_naive_bayes_classification(self):
        bn = build_naive_bayes(
            "C", ["spam", "ham"],
            {"spam": 0.3, "ham": 0.7},
            ["W1", "W2"],
            [["yes", "no"], ["yes", "no"]],
            [
                {"spam": {"yes": 0.8, "no": 0.2}, "ham": {"yes": 0.1, "no": 0.9}},
                {"spam": {"yes": 0.6, "no": 0.4}, "ham": {"yes": 0.2, "no": 0.8}},
            ]
        )
        # P(spam | W1=no, W2=no) should be low
        result = variable_elimination(bn, ["C"], {"W1": "no", "W2": "no"})
        assert result.get({"C": "ham"}) > result.get({"C": "spam"})


# ===== Diagnostic Queries =====

class TestDiagnostics:
    def test_mpe(self):
        bn = make_simple_chain()
        assignment, prob = most_probable_explanation(bn, {"C": 0})
        # Given C=0, most likely: A=0, B=0 (strong chain)
        assert assignment["B"] == 0

    def test_mutual_information_independent(self):
        """Independent variables should have ~0 MI."""
        bn = make_alarm_network()
        # B and E are independent (no evidence)
        mi = mutual_information(bn, "B", "E")
        assert mi == pytest.approx(0.0, abs=1e-4)

    def test_mutual_information_dependent(self):
        """Dependent variables should have positive MI."""
        bn = make_simple_chain()
        mi = mutual_information(bn, "A", "B")
        assert mi > 0.0

    def test_mutual_information_with_evidence(self):
        bn = make_alarm_network()
        # B and E become dependent given A (v-structure)
        mi = mutual_information(bn, "B", "E", {"A": True})
        assert mi > 0.0

    def test_sensitivity_analysis(self):
        bn = make_alarm_network()
        sens = sensitivity_analysis(bn, "B", True)
        # Alarm should have highest sensitivity for burglary
        assert "A" in sens
        assert "J" in sens
        assert "M" in sens
        # Direct child of B (alarm) should be very sensitive
        assert sens["A"] > 0.0


# ===== Advanced: Multi-valued domains =====

class TestMultiValued:
    def test_three_valued(self):
        bn = BayesianNetwork()
        bn.add_node("Weather", ["sunny", "cloudy", "rainy"])
        bn.add_node("Mood", ["happy", "sad"])
        bn.add_edge("Weather", "Mood")
        bn.set_cpt_dict("Weather", {"sunny": 0.6, "cloudy": 0.3, "rainy": 0.1})
        bn.set_cpt_dict("Mood", {
            "sunny": {"happy": 0.9, "sad": 0.1},
            "cloudy": {"happy": 0.5, "sad": 0.5},
            "rainy": {"happy": 0.2, "sad": 0.8},
        })
        result = variable_elimination(bn, ["Weather"], {"Mood": "happy"})
        # Sunny should be most likely given happy mood
        assert result.get({"Weather": "sunny"}) > result.get({"Weather": "rainy"})

    def test_four_node_diamond(self):
        """Diamond: A -> B, A -> C, B -> D, C -> D."""
        bn = BayesianNetwork()
        for n in ["A", "B", "C", "D"]:
            bn.add_node(n, [0, 1])
        bn.add_edge("A", "B")
        bn.add_edge("A", "C")
        bn.add_edge("B", "D")
        bn.add_edge("C", "D")
        bn.set_cpt_dict("A", {0: 0.5, 1: 0.5})
        bn.set_cpt_dict("B", {0: {0: 0.8, 1: 0.2}, 1: {0: 0.3, 1: 0.7}})
        bn.set_cpt_dict("C", {0: {0: 0.7, 1: 0.3}, 1: {0: 0.4, 1: 0.6}})
        bn.set_cpt_dict("D", {
            (0, 0): {0: 0.9, 1: 0.1},
            (0, 1): {0: 0.6, 1: 0.4},
            (1, 0): {0: 0.5, 1: 0.5},
            (1, 1): {0: 0.1, 1: 0.9},
        })
        result = variable_elimination(bn, ["D"])
        assert result.get({"D": 0}) + result.get({"D": 1}) == pytest.approx(1.0)

    def test_diamond_d_separation(self):
        bn = BayesianNetwork()
        for n in ["A", "B", "C", "D"]:
            bn.add_node(n, [0, 1])
        bn.add_edge("A", "B")
        bn.add_edge("A", "C")
        bn.add_edge("B", "D")
        bn.add_edge("C", "D")
        # B and C are d-separated given A (fork structure)
        assert bn.is_d_separated({"B"}, {"C"}, {"A"}) == True
        # B and C are NOT d-separated given D (collider)
        assert bn.is_d_separated({"B"}, {"C"}, {"D"}) == False


# ===== Sampling =====

class TestSampling:
    def test_sample_prior(self):
        bn = make_simple_chain()
        samples = bn.sample(n_samples=100)
        assert len(samples) == 100
        for s in samples:
            assert "A" in s and "B" in s and "C" in s

    def test_sample_with_evidence(self):
        bn = make_simple_chain()
        samples = bn.sample(evidence={"A": 0}, n_samples=50)
        for s in samples:
            assert s["A"] == 0

    def test_sample_distribution_approximate(self):
        """Samples should roughly match true distribution."""
        bn = make_simple_chain()
        samples = bn.sample(n_samples=1000)
        count_a0 = sum(1 for s in samples if s["A"] == 0)
        # P(A=0) = 0.6, so count should be roughly 600
        assert 500 < count_a0 < 700


# ===== Edge Cases =====

class TestEdgeCases:
    def test_single_node_network(self):
        bn = BayesianNetwork()
        bn.add_node("X", [0, 1, 2])
        bn.set_cpt_dict("X", {0: 0.2, 1: 0.5, 2: 0.3})
        result = variable_elimination(bn, ["X"])
        assert result.get({"X": 0}) == pytest.approx(0.2)
        assert result.get({"X": 1}) == pytest.approx(0.5)

    def test_deterministic_cpt(self):
        bn = BayesianNetwork()
        bn.add_node("A", [0, 1])
        bn.add_node("B", [0, 1])
        bn.add_edge("A", "B")
        bn.set_cpt_dict("A", {0: 1.0, 1: 0.0})
        bn.set_cpt_dict("B", {0: {0: 1.0, 1: 0.0}, 1: {0: 0.0, 1: 1.0}})
        result = variable_elimination(bn, ["B"])
        assert result.get({"B": 0}) == pytest.approx(1.0)
        assert result.get({"B": 1}) == pytest.approx(0.0)

    def test_evidence_on_query_var(self):
        bn = make_simple_chain()
        result = variable_elimination(bn, ["A"], {"A": 0})
        assert result.get({"A": 0}) == pytest.approx(1.0)

    def test_all_vars_observed(self):
        bn = make_simple_chain()
        result = variable_elimination(bn, ["A"], {"A": 0, "B": 0, "C": 0})
        assert result.get({"A": 0}) == pytest.approx(1.0)

    def test_large_domain(self):
        """Test with larger domain size."""
        bn = BayesianNetwork()
        domain = list(range(10))
        bn.add_node("X", domain)
        cpt = {i: 0.1 for i in domain}
        bn.set_cpt_dict("X", cpt)
        result = variable_elimination(bn, ["X"])
        for i in domain:
            assert result.get({"X": i}) == pytest.approx(0.1)

    def test_normalize_zero_table(self):
        """Normalizing a zero table should not crash."""
        f = Factor(["A"], {"A": [0, 1]}, {(0,): 0.0, (1,): 0.0})
        n = f.normalize()
        # Should return unchanged (total is 0)
        assert n.get({"A": 0}) == 0.0


# ===== Consistency Checks =====

class TestConsistency:
    def test_ve_matches_junction_tree(self):
        """VE and JT should give same answers."""
        bn = make_alarm_network()
        ve = variable_elimination(bn, ["A"], {"J": True})
        jt = JunctionTree(bn)
        jt.calibrate({"J": True})
        jt_result = jt.query(["A"])
        assert ve.get({"A": True}) == pytest.approx(
            jt_result.get({"A": True}), abs=0.01)

    def test_marginals_sum_to_one(self):
        bn = make_alarm_network()
        for node in bn.nodes:
            result = variable_elimination(bn, [node])
            total = sum(result.get({node: v}) for v in bn.domains[node])
            assert total == pytest.approx(1.0)

    def test_conditional_consistent(self):
        """P(A|B) * P(B) should equal P(A,B) marginalized."""
        bn = make_simple_chain()
        joint = variable_elimination(bn, ["A", "B"])
        marginal_b = variable_elimination(bn, ["B"])
        cond_a_given_b0 = variable_elimination(bn, ["A"], {"B": 0})
        # P(A=0, B=0) = P(A=0|B=0) * P(B=0)
        expected = cond_a_given_b0.get({"A": 0}) * marginal_b.get({"B": 0})
        actual = joint.get({"A": 0, "B": 0})
        assert actual == pytest.approx(expected, abs=1e-6)

    def test_chain_rule(self):
        """P(A,B,C) = P(A)P(B|A)P(C|B) should be consistent."""
        bn = make_simple_chain()
        # P(A=0, B=0, C=0) = P(A=0) * P(B=0|A=0) * P(C=0|B=0)
        expected = 0.6 * 0.7 * 0.9  # = 0.378
        joint = variable_elimination(bn, ["A", "B", "C"])
        actual = joint.get({"A": 0, "B": 0, "C": 0})
        assert actual == pytest.approx(expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
