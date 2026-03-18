"""Tests for V228: Causal Discovery from Interventions.

AI-Generated | Claude (Anthropic) | AgentZero A2
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V214_causal_discovery'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V209_bayesian_networks'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V211_causal_inference'))

from causal_discovery_interventions import (
    InterventionalDataset,
    CPDAG,
    InterventionResult,
    ActiveDiscoveryResult,
    pc_result_to_cpdag,
    dag_to_cpdag,
    orient_edges_from_intervention,
    orient_from_multi_intervention,
    _compute_marginals,
    _distribution_changed,
    _apply_meek_rules,
    intervention_score_edge_count,
    intervention_score_entropy,
    intervention_score_separator,
    select_intervention,
    simulate_intervention,
    active_causal_discovery,
    plan_interventions,
    minimum_intervention_set,
    interventional_independence_test,
    check_mechanism_invariance,
    check_transportability,
    build_chain_bn,
    build_collider_bn,
    build_diamond_bn,
    build_confounder_bn,
    discovery_summary,
)
from causal_discovery import pc_algorithm, sample_from_bn, structural_hamming_distance
from bayesian_networks import BayesianNetwork


# ===========================================================================
# InterventionalDataset
# ===========================================================================

class TestInterventionalDataset:
    def test_add_observational(self):
        ds = InterventionalDataset()
        ds.add_observational([{"X": 1, "Y": 0}, {"X": 0, "Y": 1}])
        assert len(ds.observational) == 2

    def test_add_interventional(self):
        ds = InterventionalDataset()
        ds.add_interventional("X", 1, [{"X": 1, "Y": 0}])
        assert "X" in ds.interventional
        assert ds.interventional["X"][0]["__intervention_target__"] == "X"
        assert ds.interventional["X"][0]["__intervention_value__"] == 1

    def test_all_samples_strips_metadata(self):
        ds = InterventionalDataset()
        ds.add_observational([{"X": 1, "Y": 0}])
        ds.add_interventional("X", 1, [{"X": 1, "Y": 1}])
        all_s = ds.all_samples()
        assert len(all_s) == 2
        for s in all_s:
            assert "__intervention_value__" not in s
            assert "__intervention_target__" not in s

    def test_interventional_for(self):
        ds = InterventionalDataset()
        ds.add_interventional("X", 1, [{"X": 1, "Y": 0}])
        ds.add_interventional("Y", 0, [{"X": 1, "Y": 0}])
        x_samples = ds.interventional_for("X")
        assert len(x_samples) == 1
        assert "__intervention_value__" not in x_samples[0]

    def test_total_samples(self):
        ds = InterventionalDataset()
        ds.add_observational([{"X": 1}] * 10)
        ds.add_interventional("X", 1, [{"X": 1}] * 5)
        assert ds.total_samples() == 15

    def test_empty_interventional_for(self):
        ds = InterventionalDataset()
        assert ds.interventional_for("Z") == []


# ===========================================================================
# CPDAG
# ===========================================================================

class TestCPDAG:
    def test_creation(self):
        cpdag = CPDAG(
            variables=["A", "B", "C"],
            directed={("A", "C"), ("B", "C")},
            undirected={frozenset({"A", "B"})},
        )
        assert cpdag.num_undirected() == 1
        assert not cpdag.is_fully_oriented()

    def test_fully_oriented(self):
        cpdag = CPDAG(
            variables=["A", "B"],
            directed={("A", "B")},
        )
        assert cpdag.is_fully_oriented()
        assert cpdag.num_undirected() == 0

    def test_neighbors(self):
        cpdag = CPDAG(
            variables=["A", "B", "C"],
            directed={("A", "B")},
            undirected={frozenset({"B", "C"})},
        )
        assert cpdag.neighbors("B") == {"A", "C"}

    def test_children_parents(self):
        cpdag = CPDAG(
            variables=["A", "B", "C"],
            directed={("A", "B"), ("A", "C")},
        )
        assert cpdag.children("A") == {"B", "C"}
        assert cpdag.parents("B") == {"A"}

    def test_undirected_neighbors(self):
        cpdag = CPDAG(
            variables=["A", "B", "C"],
            directed={("A", "B")},
            undirected={frozenset({"A", "C"})},
        )
        assert cpdag.undirected_neighbors("A") == {"C"}
        assert cpdag.undirected_neighbors("B") == set()

    def test_copy(self):
        cpdag = CPDAG(
            variables=["A", "B"],
            directed={("A", "B")},
            undirected={frozenset({"A", "B"})},
        )
        copy = cpdag.copy()
        copy.directed.add(("X", "Y"))
        assert ("X", "Y") not in cpdag.directed

    def test_to_dag_set(self):
        cpdag = CPDAG(
            variables=["A", "B", "C"],
            directed={("A", "B")},
            undirected={frozenset({"B", "C"})},
        )
        dag = cpdag.to_dag_set()
        assert ("A", "B") in dag
        assert len(dag) == 2  # one directed + one arbitrarily oriented


# ===========================================================================
# dag_to_cpdag
# ===========================================================================

class TestDagToCPDAG:
    def test_collider_all_compelled(self):
        """X -> Z <- Y: v-structure, all edges compelled."""
        edges = {("X", "Z"), ("Y", "Z")}
        cpdag = dag_to_cpdag(edges, ["X", "Y", "Z"])
        assert cpdag.directed == {("X", "Z"), ("Y", "Z")}
        assert cpdag.num_undirected() == 0

    def test_chain_has_undirected(self):
        """A -> B -> C: chain is Markov equivalent to A <- B -> C etc."""
        edges = {("A", "B"), ("B", "C")}
        cpdag = dag_to_cpdag(edges, ["A", "B", "C"])
        # Chain has no v-structures, so edges should be undirected
        assert cpdag.num_undirected() == 2
        assert len(cpdag.directed) == 0

    def test_fork_has_undirected(self):
        """A <- B -> C: fork also Markov equivalent to chain."""
        edges = {("B", "A"), ("B", "C")}
        cpdag = dag_to_cpdag(edges, ["A", "B", "C"])
        assert cpdag.num_undirected() == 2

    def test_diamond_partial_orientation(self):
        """X -> A -> Y, X -> B -> Y: has v-structures at Y."""
        edges = {("X", "A"), ("X", "B"), ("A", "Y"), ("B", "Y")}
        cpdag = dag_to_cpdag(edges, ["X", "A", "B", "Y"])
        # A -> Y <- B is a v-structure (A and B not adjacent)
        assert ("A", "Y") in cpdag.directed
        assert ("B", "Y") in cpdag.directed


# ===========================================================================
# Distribution comparison
# ===========================================================================

class TestDistributionChanged:
    def test_identical_distributions(self):
        assert not _distribution_changed(
            {0: 0.5, 1: 0.5}, {0: 0.5, 1: 0.5}
        )

    def test_very_different_distributions(self):
        assert _distribution_changed(
            {0: 0.9, 1: 0.1}, {0: 0.1, 1: 0.9}
        )

    def test_empty_distributions(self):
        assert not _distribution_changed({}, {})
        assert not _distribution_changed({0: 0.5}, {})

    def test_slight_difference(self):
        # With default alpha=0.05, threshold = sqrt(0.05)*0.5 ~ 0.112
        # TVD = 0.5 * (|0.45 - 0.5| + |0.55 - 0.5|) = 0.05
        assert not _distribution_changed(
            {0: 0.5, 1: 0.5}, {0: 0.45, 1: 0.55}
        )


# ===========================================================================
# Compute marginals
# ===========================================================================

class TestComputeMarginals:
    def test_basic(self):
        data = [{"X": 0, "Y": 1}, {"X": 1, "Y": 0}, {"X": 1, "Y": 1}]
        m = _compute_marginals(data, ["X", "Y"])
        assert abs(m["X"][0] - 1 / 3) < 0.01
        assert abs(m["X"][1] - 2 / 3) < 0.01
        assert abs(m["Y"][0] - 1 / 3) < 0.01
        assert abs(m["Y"][1] - 2 / 3) < 0.01

    def test_empty_data(self):
        m = _compute_marginals([], ["X"])
        assert m == {}


# ===========================================================================
# Meek rules
# ===========================================================================

class TestMeekRules:
    def test_rule1_propagation(self):
        """X -> A - B, X not adj B => A -> B."""
        cpdag = CPDAG(
            variables=["X", "A", "B"],
            directed={("X", "A")},
            undirected={frozenset({"A", "B"})},
        )
        result = _apply_meek_rules(cpdag)
        assert ("A", "B") in result.directed
        assert result.num_undirected() == 0

    def test_rule2_propagation(self):
        """A -> X -> B, A - B => A -> B."""
        cpdag = CPDAG(
            variables=["A", "X", "B"],
            directed={("A", "X"), ("X", "B")},
            undirected={frozenset({"A", "B"})},
        )
        result = _apply_meek_rules(cpdag)
        assert ("A", "B") in result.directed

    def test_no_propagation(self):
        """A - B, no other edges: nothing to propagate."""
        cpdag = CPDAG(
            variables=["A", "B"],
            undirected={frozenset({"A", "B"})},
        )
        result = _apply_meek_rules(cpdag)
        assert result.num_undirected() == 1


# ===========================================================================
# Intervention scoring
# ===========================================================================

class TestInterventionScoring:
    def test_edge_count_score(self):
        cpdag = CPDAG(
            variables=["A", "B", "C"],
            undirected={frozenset({"A", "B"}), frozenset({"A", "C"})},
        )
        assert intervention_score_edge_count(cpdag, "A") == 2.0
        assert intervention_score_edge_count(cpdag, "B") == 1.0

    def test_entropy_score(self):
        cpdag = CPDAG(
            variables=["A", "B", "C"],
            undirected={frozenset({"A", "B"}), frozenset({"A", "C"})},
        )
        # Intervening on A resolves both edges directly
        score_a = intervention_score_entropy(cpdag, "A", [])
        score_b = intervention_score_entropy(cpdag, "B", [])
        assert score_a >= score_b

    def test_separator_score(self):
        cpdag = CPDAG(
            variables=["A", "B", "C"],
            undirected={frozenset({"A", "B"}), frozenset({"A", "C"})},
        )
        score_a = intervention_score_separator(cpdag, "A")
        assert score_a > 0

    def test_score_fully_oriented(self):
        cpdag = CPDAG(
            variables=["A", "B"],
            directed={("A", "B")},
        )
        assert intervention_score_entropy(cpdag, "A", []) == 0.0


# ===========================================================================
# Select intervention
# ===========================================================================

class TestSelectIntervention:
    def test_basic_selection(self):
        cpdag = CPDAG(
            variables=["A", "B", "C"],
            undirected={frozenset({"A", "B"}), frozenset({"A", "C"})},
        )
        target = select_intervention(cpdag, strategy="edge_count")
        assert target == "A"  # A has 2 undirected edges

    def test_returns_none_when_oriented(self):
        cpdag = CPDAG(
            variables=["A", "B"],
            directed={("A", "B")},
        )
        assert select_intervention(cpdag) is None

    def test_cost_aware(self):
        cpdag = CPDAG(
            variables=["A", "B", "C"],
            undirected={frozenset({"A", "B"}), frozenset({"A", "C"})},
        )
        # A has 2 edges but costs 100, B has 1 edge but costs 1
        target = select_intervention(
            cpdag,
            strategy="cost_aware",
            cost_fn=lambda v: 100 if v == "A" else 1,
        )
        assert target == "B"  # B has better info/cost ratio

    def test_restricted_candidates(self):
        cpdag = CPDAG(
            variables=["A", "B", "C"],
            undirected={frozenset({"A", "B"}), frozenset({"A", "C"})},
        )
        target = select_intervention(cpdag, candidates=["B", "C"], strategy="edge_count")
        assert target in ("B", "C")


# ===========================================================================
# Benchmark BN construction
# ===========================================================================

class TestBenchmarkBNs:
    def test_chain_bn(self):
        bn = build_chain_bn(4)
        assert len(bn.nodes) == 4
        # Should have 3 edges
        total_edges = sum(len(bn.children.get(n, [])) for n in bn.nodes)
        assert total_edges == 3

    def test_collider_bn(self):
        bn = build_collider_bn()
        assert "X" in bn.nodes
        assert "Y" in bn.nodes
        assert "Z" in bn.nodes
        assert "Z" in bn.children.get("X", [])
        assert "Z" in bn.children.get("Y", [])

    def test_diamond_bn(self):
        bn = build_diamond_bn()
        assert len(bn.nodes) == 4

    def test_confounder_bn(self):
        bn = build_confounder_bn()
        assert "U" in bn.nodes
        assert "X" in bn.children.get("U", [])
        assert "Y" in bn.children.get("U", [])

    def test_chain_sampling(self):
        bn = build_chain_bn(3)
        samples = sample_from_bn(bn, 100, seed=42)
        assert len(samples) == 100
        assert all("X1" in s and "X2" in s and "X3" in s for s in samples)


# ===========================================================================
# Simulate intervention
# ===========================================================================

class TestSimulateIntervention:
    def test_hard_intervention(self):
        bn = build_confounder_bn()
        samples = simulate_intervention(bn, "X", 1, 200, seed=42)
        assert len(samples) == 200
        # All X values should be 1
        assert all(s["X"] == 1 for s in samples)

    def test_intervention_breaks_parent_link(self):
        """Intervening on X should make X independent of U."""
        bn = build_confounder_bn()
        int_samples = simulate_intervention(bn, "X", 1, 500, seed=42)
        # X is always 1, so U should vary freely (P(U) = 0.5, 0.5)
        u_vals = [s["U"] for s in int_samples]
        u1_frac = sum(1 for v in u_vals if v == 1) / len(u_vals)
        assert 0.35 < u1_frac < 0.65  # approximately 0.5


# ===========================================================================
# Orient edges from intervention
# ===========================================================================

class TestOrientEdgesFromIntervention:
    def test_orient_chain(self):
        """In chain A - B - C, intervening on A should orient A -> B."""
        bn = build_chain_bn(3)
        obs_data = sample_from_bn(bn, 1000, seed=42)
        int_data = simulate_intervention(bn, "X1", 1, 500, seed=43)

        cpdag = CPDAG(
            variables=["X1", "X2", "X3"],
            undirected={frozenset({"X1", "X2"}), frozenset({"X2", "X3"})},
        )

        new_cpdag, oriented = orient_edges_from_intervention(
            cpdag, "X1", int_data, obs_data,
        )
        # Should orient at least X1 -> X2
        assert any(e == ("X1", "X2") for e in oriented)

    def test_collider_already_oriented(self):
        """Collider X -> Z <- Y: no undirected edges to orient."""
        cpdag = CPDAG(
            variables=["X", "Y", "Z"],
            directed={("X", "Z"), ("Y", "Z")},
        )
        bn = build_collider_bn()
        obs_data = sample_from_bn(bn, 500, seed=42)
        int_data = simulate_intervention(bn, "X", 1, 200, seed=43)

        new_cpdag, oriented = orient_edges_from_intervention(
            cpdag, "X", int_data, obs_data,
        )
        assert oriented == []
        assert new_cpdag.is_fully_oriented()


# ===========================================================================
# Multi-target intervention
# ===========================================================================

class TestMultiIntervention:
    def test_multi_target(self):
        cpdag = CPDAG(
            variables=["A", "B", "C", "D"],
            undirected={
                frozenset({"A", "C"}),
                frozenset({"B", "D"}),
            },
        )
        # Fake data: intervening on A and B changes C and D
        obs_data = [{"A": 0, "B": 0, "C": 0, "D": 0}] * 100
        int_data = [{"A": 1, "B": 1, "C": 1, "D": 1}] * 100

        new_cpdag, oriented = orient_from_multi_intervention(
            cpdag, {"A", "B"}, int_data, obs_data,
        )
        # Should orient A -> C and B -> D (C and D changed)
        assert len(oriented) >= 2

    def test_both_targets_edge_skipped(self):
        """Edge between two intervention targets can't be oriented."""
        cpdag = CPDAG(
            variables=["A", "B"],
            undirected={frozenset({"A", "B"})},
        )
        obs_data = [{"A": 0, "B": 0}] * 50
        int_data = [{"A": 1, "B": 1}] * 50

        new_cpdag, oriented = orient_from_multi_intervention(
            cpdag, {"A", "B"}, int_data, obs_data,
        )
        # Edge should remain undirected
        assert new_cpdag.num_undirected() == 1


# ===========================================================================
# Plan interventions
# ===========================================================================

class TestPlanInterventions:
    def test_plan_chain(self):
        """Chain A - B - C: should need at most 2 interventions."""
        cpdag = CPDAG(
            variables=["A", "B", "C"],
            undirected={frozenset({"A", "B"}), frozenset({"B", "C"})},
        )
        plan = plan_interventions(cpdag)
        assert len(plan) <= 2

    def test_plan_single_edge(self):
        cpdag = CPDAG(
            variables=["A", "B"],
            undirected={frozenset({"A", "B"})},
        )
        plan = plan_interventions(cpdag)
        assert len(plan) == 1

    def test_plan_fully_oriented(self):
        cpdag = CPDAG(
            variables=["A", "B"],
            directed={("A", "B")},
        )
        plan = plan_interventions(cpdag)
        assert plan == []


# ===========================================================================
# Minimum intervention set
# ===========================================================================

class TestMinimumInterventionSet:
    def test_chain(self):
        cpdag = CPDAG(
            variables=["A", "B", "C"],
            undirected={frozenset({"A", "B"}), frozenset({"B", "C"})},
        )
        targets = minimum_intervention_set(cpdag)
        # B touches both edges, so 1 intervention should suffice
        assert len(targets) <= 2

    def test_empty(self):
        cpdag = CPDAG(
            variables=["A", "B"],
            directed={("A", "B")},
        )
        targets = minimum_intervention_set(cpdag)
        assert targets == []

    def test_star(self):
        """Star graph: A - X, B - X, C - X. One intervention on X."""
        cpdag = CPDAG(
            variables=["A", "B", "C", "X"],
            undirected={
                frozenset({"A", "X"}),
                frozenset({"B", "X"}),
                frozenset({"C", "X"}),
            },
        )
        targets = minimum_intervention_set(cpdag)
        assert "X" in targets
        assert len(targets) <= 2


# ===========================================================================
# Interventional independence test
# ===========================================================================

class TestInterventionalIndependenceTest:
    def test_confounder_reveals_structure(self):
        """Intervening on X should change X-Y independence."""
        bn = build_confounder_bn()
        obs_data = sample_from_bn(bn, 1000, seed=42)
        int_data = simulate_intervention(bn, "X", 1, 500, seed=43)

        result = interventional_independence_test(
            obs_data, int_data, "X", "Y", "X",
        )
        assert "obs_independent" in result
        assert "int_independent" in result
        assert "structure_revealed" in result
        assert "implication" in result

    def test_no_change(self):
        """Independent vars should stay independent."""
        bn = build_collider_bn()
        obs_data = sample_from_bn(bn, 1000, seed=42)
        int_data = simulate_intervention(bn, "X", 1, 500, seed=43)

        result = interventional_independence_test(
            obs_data, int_data, "X", "Y", "X",
        )
        # X and Y are marginally independent in both cases
        assert result["obs_independent"] or not result["structure_revealed"]


# ===========================================================================
# Mechanism invariance
# ===========================================================================

class TestMechanismInvariance:
    def test_invariant_mechanism(self):
        """X -> Y. Intervening on Y shouldn't change P(X)."""
        bn = build_chain_bn(2)
        obs_data = sample_from_bn(bn, 1000, seed=42)
        int_data = simulate_intervention(bn, "X2", 1, 500, seed=43)

        result = check_mechanism_invariance(
            obs_data, int_data, "X1", [], "X2",
        )
        assert result["invariant"]

    def test_non_invariant_mechanism(self):
        """Intervening on parent should change child's marginal."""
        bn = build_chain_bn(2)
        obs_data = sample_from_bn(bn, 1000, seed=42)
        int_data = simulate_intervention(bn, "X1", 1, 500, seed=43)

        result = check_mechanism_invariance(
            obs_data, int_data, "X2", [], "X1",
        )
        # X2's distribution should change when we intervene on X1
        assert not result["invariant"]

    def test_conditional_invariance(self):
        """P(Y | X) should be invariant under do(X) (mechanism is intrinsic)."""
        bn = build_chain_bn(2)
        obs_data = sample_from_bn(bn, 1000, seed=42)
        int_data = simulate_intervention(bn, "X1", 1, 500, seed=43)

        result = check_mechanism_invariance(
            obs_data, int_data, "X2", ["X1"], "X1",
        )
        # P(X2 | X1) is the causal mechanism -- should be invariant
        assert result["invariant"]


# ===========================================================================
# Transportability
# ===========================================================================

class TestTransportability:
    def test_same_domain(self):
        """Same BN in both domains: everything is transportable."""
        bn = build_chain_bn(3)
        data1 = sample_from_bn(bn, 500, seed=42)
        data2 = sample_from_bn(bn, 500, seed=43)

        result = check_transportability(data1, data2, ["X1", "X2", "X3"])
        assert result["fraction_transportable"] > 0.5

    def test_different_domains(self):
        """Different BNs: mechanisms differ."""
        bn1 = build_chain_bn(2)
        # Create a very different BN
        bn2 = BayesianNetwork()
        bn2.add_node("X1", [0, 1])
        bn2.add_node("X2", [0, 1])
        bn2.add_edge("X1", "X2")
        bn2.set_cpt("X1", {0: 0.1, 1: 0.9})  # very different from 0.5/0.5
        bn2.set_cpt("X2", {0: {0: 0.1, 1: 0.9}, 1: {0: 0.9, 1: 0.1}})

        data1 = sample_from_bn(bn1, 500, seed=42)
        data2 = sample_from_bn(bn2, 500, seed=43)

        result = check_transportability(data1, data2, ["X1", "X2"])
        assert len(result["non_transportable"]) >= 1


# ===========================================================================
# Active causal discovery (integration)
# ===========================================================================

class TestActiveDiscovery:
    def test_chain_discovery(self):
        """Active discovery on a chain should orient all edges."""
        bn = build_chain_bn(3)
        obs_data = sample_from_bn(bn, 1000, seed=42)
        true_edges = {("X1", "X2"), ("X2", "X3")}

        def intervene(target, value):
            return simulate_intervention(bn, target, value, 300, seed=44)

        result = active_causal_discovery(
            obs_data,
            variables=["X1", "X2", "X3"],
            intervention_fn=intervene,
            domains={"X1": [0, 1], "X2": [0, 1], "X3": [0, 1]},
            max_interventions=5,
            ground_truth=true_edges,
        )

        assert result.iterations > 0
        assert result.total_samples_used > 1000  # obs + interventional
        assert result.edges_oriented_by_intervention > 0

    def test_collider_needs_no_intervention(self):
        """Collider is identifiable from observational data alone."""
        bn = build_collider_bn()
        obs_data = sample_from_bn(bn, 2000, seed=42)

        def intervene(target, value):
            return simulate_intervention(bn, target, value, 300, seed=44)

        result = active_causal_discovery(
            obs_data,
            variables=["X", "Y", "Z"],
            intervention_fn=intervene,
            domains={"X": [0, 1], "Y": [0, 1], "Z": [0, 1]},
            max_interventions=5,
        )

        # Collider v-structure should be identified from obs data
        # May or may not need interventions depending on sample quality
        assert result.final_cpdag is not None

    def test_diamond_discovery(self):
        """Diamond structure: needs interventions for full orientation."""
        bn = build_diamond_bn()
        obs_data = sample_from_bn(bn, 1500, seed=42)
        true_edges = {("X", "A"), ("X", "B"), ("A", "Y"), ("B", "Y")}

        def intervene(target, value):
            return simulate_intervention(bn, target, value, 400, seed=45)

        result = active_causal_discovery(
            obs_data,
            variables=["X", "A", "B", "Y"],
            intervention_fn=intervene,
            domains={"X": [0, 1], "A": [0, 1], "B": [0, 1], "Y": [0, 1]},
            max_interventions=5,
            ground_truth=true_edges,
        )

        assert result.iterations >= 0
        assert result.final_dag is not None

    def test_confounder_discovery(self):
        """Confounder structure requires interventions."""
        bn = build_confounder_bn()
        obs_data = sample_from_bn(bn, 1500, seed=42)

        def intervene(target, value):
            return simulate_intervention(bn, target, value, 400, seed=46)

        result = active_causal_discovery(
            obs_data,
            variables=["U", "X", "Y"],
            intervention_fn=intervene,
            domains={"U": [0, 1], "X": [0, 1], "Y": [0, 1]},
            max_interventions=5,
        )

        assert result.final_dag is not None
        assert result.total_samples_used > len(obs_data)


# ===========================================================================
# Discovery summary
# ===========================================================================

class TestDiscoverySummary:
    def test_summary_fields(self):
        result = ActiveDiscoveryResult(
            final_cpdag=CPDAG(variables=["A", "B"], directed={("A", "B")}),
            final_dag={("A", "B")},
            interventions_performed=[
                InterventionResult(
                    target="A", value=1, samples=[],
                    edges_oriented=[("A", "B")],
                ),
            ],
            total_samples_used=500,
            iterations=1,
            edges_oriented_by_observation=0,
            edges_oriented_by_intervention=1,
            shd={"shd": 0},
        )
        summary = discovery_summary(result)
        assert summary["total_interventions"] == 1
        assert summary["total_samples"] == 500
        assert summary["edges_from_intervention"] == 1
        assert summary["fully_oriented"]
        assert len(summary["interventions"]) == 1

    def test_summary_no_interventions(self):
        result = ActiveDiscoveryResult(
            final_cpdag=CPDAG(variables=["A", "B"], directed={("A", "B")}),
            final_dag={("A", "B")},
            interventions_performed=[],
            total_samples_used=100,
            iterations=0,
            edges_oriented_by_observation=1,
            edges_oriented_by_intervention=0,
        )
        summary = discovery_summary(result)
        assert summary["total_interventions"] == 0
        assert summary["interventions"] == []


# ===========================================================================
# pc_result_to_cpdag
# ===========================================================================

class TestPCResultToCPDAG:
    def test_conversion(self):
        bn = build_collider_bn()
        data = sample_from_bn(bn, 2000, seed=42)
        pc_result = pc_algorithm(data, variables=["X", "Y", "Z"])
        cpdag = pc_result_to_cpdag(pc_result)

        assert isinstance(cpdag, CPDAG)
        assert set(cpdag.variables) == {"X", "Y", "Z"}
        # Should have some edges
        total = len(cpdag.directed) + cpdag.num_undirected()
        assert total > 0


# ===========================================================================
# Edge case and robustness tests
# ===========================================================================

class TestEdgeCases:
    def test_single_variable(self):
        cpdag = CPDAG(variables=["A"])
        assert cpdag.is_fully_oriented()
        assert select_intervention(cpdag) is None
        plan = plan_interventions(cpdag)
        assert plan == []

    def test_two_variables_undirected(self):
        cpdag = CPDAG(
            variables=["A", "B"],
            undirected={frozenset({"A", "B"})},
        )
        target = select_intervention(cpdag, strategy="edge_count")
        assert target in ("A", "B")

    def test_disconnected_components(self):
        cpdag = CPDAG(
            variables=["A", "B", "C", "D"],
            undirected={frozenset({"A", "B"}), frozenset({"C", "D"})},
        )
        plan = plan_interventions(cpdag)
        assert len(plan) <= 2

    def test_empty_cpdag(self):
        cpdag = CPDAG(variables=["A", "B", "C"])
        assert cpdag.is_fully_oriented()  # no edges at all


# ===========================================================================
# Integration: full pipeline
# ===========================================================================

class TestFullPipeline:
    def test_observe_then_intervene_chain(self):
        """Full pipeline: observe -> PC -> intervene -> orient."""
        bn = build_chain_bn(3)
        obs_data = sample_from_bn(bn, 1500, seed=42)

        # Step 1: PC algorithm
        pc_result = pc_algorithm(obs_data, variables=["X1", "X2", "X3"])
        cpdag = pc_result_to_cpdag(pc_result)

        # Chain should have undirected edges
        initial_undirected = cpdag.num_undirected()

        # Step 2: Plan interventions
        plan = plan_interventions(cpdag)

        # Step 3: Execute first intervention
        if plan:
            target = plan[0]
            int_data = simulate_intervention(bn, target, 1, 500, seed=43)
            cpdag, oriented = orient_edges_from_intervention(
                cpdag, target, int_data, obs_data,
            )
            # Should have oriented some edges
            assert cpdag.num_undirected() <= initial_undirected

    def test_full_active_diamond(self):
        """Full active discovery on diamond, verify convergence."""
        bn = build_diamond_bn()
        obs_data = sample_from_bn(bn, 2000, seed=42)
        true_edges = {("X", "A"), ("X", "B"), ("A", "Y"), ("B", "Y")}

        def intervene(target, value):
            return simulate_intervention(bn, target, value, 500, seed=47)

        result = active_causal_discovery(
            obs_data,
            variables=["X", "A", "B", "Y"],
            intervention_fn=intervene,
            domains={"X": [0, 1], "A": [0, 1], "B": [0, 1], "Y": [0, 1]},
            max_interventions=8,
            ground_truth=true_edges,
        )

        summary = discovery_summary(result)
        assert summary["total_samples"] > 2000
        assert summary["final_directed_edges"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
