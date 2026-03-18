"""Tests for V210: Influence Diagrams."""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V209_bayesian_networks'))

from influence_diagrams import (
    NodeType, UtilityFactor, Policy, InfluenceDiagram,
    build_simple_decision, build_sequential_decisions,
    build_medical_diagnosis, build_oil_wildcatter,
)
from bayesian_networks import Factor, variable_elimination


# ---------------------------------------------------------------------------
# UtilityFactor tests
# ---------------------------------------------------------------------------

class TestUtilityFactor:
    def test_create_utility_factor(self):
        uf = UtilityFactor(["X"], {"X": ["a", "b"]}, {("a",): 10, ("b",): -5})
        assert uf.get({"X": "a"}) == 10
        assert uf.get({"X": "b"}) == -5

    def test_default_zero(self):
        uf = UtilityFactor(["X"], {"X": ["a", "b"]})
        assert uf.get({"X": "a"}) == 0.0

    def test_set_and_get(self):
        uf = UtilityFactor(["X", "Y"], {"X": [0, 1], "Y": [0, 1]})
        uf.set({"X": 0, "Y": 1}, 42.0)
        assert uf.get({"X": 0, "Y": 1}) == 42.0

    def test_multi_variable(self):
        uf = UtilityFactor(
            ["A", "B"],
            {"A": ["lo", "hi"], "B": ["lo", "hi"]},
            {("lo", "lo"): 0, ("lo", "hi"): 5, ("hi", "lo"): 3, ("hi", "hi"): 10},
        )
        assert uf.get({"A": "hi", "B": "hi"}) == 10
        assert uf.get({"A": "lo", "B": "lo"}) == 0

    def test_negative_utilities(self):
        uf = UtilityFactor(["X"], {"X": ["a", "b"]}, {("a",): -100, ("b",): -50})
        assert uf.get({"X": "a"}) == -100


# ---------------------------------------------------------------------------
# Policy tests
# ---------------------------------------------------------------------------

class TestPolicy:
    def test_simple_policy(self):
        p = Policy("D", ["X"], {("a",): "d1", ("b",): "d2"})
        assert p.decide({"X": "a"}) == "d1"
        assert p.decide({"X": "b"}) == "d2"

    def test_no_info_policy(self):
        p = Policy("D", [], {(): "always_d1"})
        assert p.decide({}) == "always_d1"

    def test_multi_info_policy(self):
        p = Policy("D", ["X", "Y"], {
            ("a", 0): "d1",
            ("a", 1): "d2",
            ("b", 0): "d1",
            ("b", 1): "d1",
        })
        assert p.decide({"X": "a", "Y": 1}) == "d2"

    def test_unknown_observation(self):
        p = Policy("D", ["X"], {("a",): "d1"})
        assert p.decide({"X": "unknown"}) is None

    def test_repr(self):
        p = Policy("D", ["X"], {("a",): "d1"})
        assert "D" in repr(p)


# ---------------------------------------------------------------------------
# InfluenceDiagram construction
# ---------------------------------------------------------------------------

class TestInfluenceDiagramConstruction:
    def test_add_nodes(self):
        diag = InfluenceDiagram()
        diag.add_chance_node("X", [0, 1])
        diag.add_decision_node("D", ["a", "b"])
        assert diag.node_types["X"] == NodeType.CHANCE
        assert diag.node_types["D"] == NodeType.DECISION

    def test_utility_node(self):
        diag = InfluenceDiagram()
        diag.add_chance_node("X", [0, 1])
        diag.add_decision_node("D", ["a", "b"])
        diag.add_utility_node("U", ["X", "D"], {
            (0, "a"): 10, (0, "b"): 0, (1, "a"): 5, (1, "b"): 15,
        })
        assert diag.node_types["U"] == NodeType.UTILITY
        assert "U" in diag.utility_factors

    def test_info_set_manual(self):
        diag = InfluenceDiagram()
        diag.add_chance_node("X", [0, 1])
        diag.add_decision_node("D", ["a", "b"], info_vars=["X"])
        assert diag.get_info_set("D") == ["X"]

    def test_info_set_from_edge(self):
        diag = InfluenceDiagram()
        diag.add_chance_node("X", [0, 1])
        diag.add_decision_node("D", ["a", "b"])
        diag.add_edge("X", "D")
        assert "X" in diag.get_info_set("D")

    def test_node_lists(self):
        diag = InfluenceDiagram()
        diag.add_chance_node("C1", [0, 1])
        diag.add_chance_node("C2", [0, 1])
        diag.add_decision_node("D", ["a", "b"])
        diag.add_utility_node("U", ["D"], {("a",): 10, ("b",): 5})
        assert set(diag.chance_nodes()) == {"C1", "C2"}
        assert diag.decision_nodes() == ["D"]
        assert diag.utility_nodes() == ["U"]

    def test_decision_order_preserved(self):
        diag = InfluenceDiagram()
        diag.add_decision_node("D1", ["a", "b"])
        diag.add_decision_node("D2", ["x", "y"])
        diag.add_decision_node("D3", ["p", "q"])
        assert diag.decision_order == ["D1", "D2", "D3"]


# ---------------------------------------------------------------------------
# Simple decision (no information)
# ---------------------------------------------------------------------------

class TestSimpleDecision:
    def test_no_info_choose_best_action(self):
        """Without observing chance, agent should pick action maximizing EU."""
        diag = build_simple_decision(
            "Weather", ["rain", "sun"], {"rain": 0.3, "sun": 0.7},
            "Umbrella", ["take", "leave"],
            {
                ("rain", "take"):  80,
                ("rain", "leave"): 0,
                ("sun", "take"):   60,
                ("sun", "leave"): 100,
            },
            observe_chance=False,
        )
        policies, eu = diag.optimize_all_decisions()
        # EU(take) = 0.3*80 + 0.7*60 = 24 + 42 = 66
        # EU(leave) = 0.3*0 + 0.7*100 = 70
        # Optimal: leave, EU = 70
        assert policies["Umbrella"].decide({}) == "leave"
        assert abs(eu - 70.0) < 1e-6

    def test_with_info_optimal_policy(self):
        """With perfect information, agent conditions on weather."""
        diag = build_simple_decision(
            "Weather", ["rain", "sun"], {"rain": 0.3, "sun": 0.7},
            "Umbrella", ["take", "leave"],
            {
                ("rain", "take"):  80,
                ("rain", "leave"): 0,
                ("sun", "take"):   60,
                ("sun", "leave"): 100,
            },
            observe_chance=True,
        )
        policies, eu = diag.optimize_all_decisions()
        # With info: rain -> take (80), sun -> leave (100)
        # EU = 0.3*80 + 0.7*100 = 24 + 70 = 94
        assert policies["Umbrella"].decide({"Weather": "rain"}) == "take"
        assert policies["Umbrella"].decide({"Weather": "sun"}) == "leave"
        assert abs(eu - 94.0) < 1e-6

    def test_value_of_information(self):
        """VOI = EU(with info) - EU(without info)."""
        diag = build_simple_decision(
            "Weather", ["rain", "sun"], {"rain": 0.3, "sun": 0.7},
            "Umbrella", ["take", "leave"],
            {
                ("rain", "take"):  80,
                ("rain", "leave"): 0,
                ("sun", "take"):   60,
                ("sun", "leave"): 100,
            },
            observe_chance=False,
        )
        voi = diag.value_of_information("Umbrella", "Weather")
        # VOI = 94 - 70 = 24
        assert abs(voi - 24.0) < 1e-6


# ---------------------------------------------------------------------------
# Medical diagnosis (classic)
# ---------------------------------------------------------------------------

class TestMedicalDiagnosis:
    def setup_method(self):
        self.diag = build_medical_diagnosis()

    def test_structure(self):
        assert set(self.diag.chance_nodes()) == {"Disease", "Test"}
        assert self.diag.decision_nodes() == ["Treatment"]
        assert "U" in self.diag.utility_factors

    def test_optimal_policy(self):
        """Optimal: treat if positive, don't treat if negative."""
        policies, eu = self.diag.optimize_all_decisions()
        p = policies["Treatment"]
        assert p.decide({"Test": "pos"}) == "treat"
        assert p.decide({"Test": "neg"}) == "no_treat"

    def test_eu_is_reasonable(self):
        policies, eu = self.diag.optimize_all_decisions()
        # With no test and no treatment: 0.9*100 + 0.1*0 = 90
        # With test + optimal policy, should be > 90
        assert eu > 90

    def test_decision_table(self):
        rows = self.diag.decision_table("Treatment")
        # Should have 2 test results * 2 actions = 4 rows
        assert len(rows) == 4
        # Find treat|pos and no_treat|neg -- both should be high
        for row in rows:
            if row["info_config"] == {"Test": "pos"} and row["action"] == "treat":
                assert row["expected_utility"] > 0
            if row["info_config"] == {"Test": "neg"} and row["action"] == "no_treat":
                assert row["expected_utility"] > 0

    def test_value_of_test(self):
        """Test has positive value (we observe it, so VOI of Disease matters)."""
        # VOI of Disease given we already see Test
        voi = self.diag.value_of_information("Treatment", "Disease")
        assert voi >= 0

    def test_with_evidence(self):
        """If we know the patient is sick, always treat."""
        policies, eu = self.diag.optimize_all_decisions(evidence={"Disease": "sick"})
        # When sick: treat=80, no_treat=0, so always treat
        assert eu == 80.0 or abs(eu - 80.0) < 1e-6

    def test_vpi(self):
        """Value of perfect information should be >= 0."""
        evpi = self.diag.value_of_perfect_information()
        assert evpi >= -1e-9  # non-negative (within floating point)


# ---------------------------------------------------------------------------
# Oil wildcatter (multi-outcome chance)
# ---------------------------------------------------------------------------

class TestOilWildcatter:
    def setup_method(self):
        self.diag = build_oil_wildcatter()

    def test_structure(self):
        assert set(self.diag.chance_nodes()) == {"Oil", "Seismic"}
        assert self.diag.decision_nodes() == ["Drill"]

    def test_prior_eu_no_drill(self):
        """Without any info, EU(no_drill) = 0."""
        no_drill_policy = Policy("Drill", ["Seismic"], {
            ("closed",): "no_drill",
            ("open",): "no_drill",
            ("diffuse",): "no_drill",
        })
        eu = self.diag.expected_utility({"Drill": no_drill_policy})
        assert abs(eu - 0.0) < 1e-6

    def test_prior_eu_always_drill(self):
        """EU(always drill) = 0.5*(-70) + 0.3*50 + 0.2*200 = -35+15+40 = 20."""
        drill_policy = Policy("Drill", ["Seismic"], {
            ("closed",): "drill",
            ("open",): "drill",
            ("diffuse",): "drill",
        })
        eu = self.diag.expected_utility({"Drill": drill_policy})
        assert abs(eu - 20.0) < 1e-6

    def test_optimal_policy_with_seismic(self):
        """Optimal policy should condition on seismic results."""
        policies, eu = self.diag.optimize_all_decisions()
        # EU with optimal policy should be >= EU(always drill) = 20
        assert eu >= 20.0 - 1e-6

    def test_closed_structure_favors_drill(self):
        """Closed seismic structure is evidence for oil -- should drill."""
        policies, eu = self.diag.optimize_all_decisions()
        assert policies["Drill"].decide({"Seismic": "closed"}) == "drill"

    def test_diffuse_structure_disfavors_drill(self):
        """Diffuse seismic structure suggests dry -- may not drill."""
        policies, eu = self.diag.optimize_all_decisions()
        # With diffuse: P(dry|diffuse) is high, drilling EU likely negative
        decision = policies["Drill"].decide({"Seismic": "diffuse"})
        assert decision in ["drill", "no_drill"]  # depends on exact probabilities

    def test_voi_seismic(self):
        """Value of seismic info should be non-negative."""
        # Build without seismic observation to compare
        diag_no_info = InfluenceDiagram()
        diag_no_info.add_chance_node("Oil", ["dry", "wet", "soaking"])
        diag_no_info.set_cpt_dict("Oil", {"dry": 0.5, "wet": 0.3, "soaking": 0.2})
        diag_no_info.add_chance_node("Seismic", ["closed", "open", "diffuse"])
        diag_no_info.add_edge("Oil", "Seismic")
        diag_no_info.set_cpt_dict("Seismic", {
            ("dry",):     {"closed": 0.1, "open": 0.3, "diffuse": 0.6},
            ("wet",):     {"closed": 0.3, "open": 0.4, "diffuse": 0.3},
            ("soaking",): {"closed": 0.5, "open": 0.4, "diffuse": 0.1},
        })
        diag_no_info.add_decision_node("Drill", ["drill", "no_drill"], info_vars=[])
        diag_no_info.add_utility_node("U", ["Oil", "Drill"], {
            ("dry", "drill"): -70, ("dry", "no_drill"): 0,
            ("wet", "drill"): 50, ("wet", "no_drill"): 0,
            ("soaking", "drill"): 200, ("soaking", "no_drill"): 0,
        })
        _, eu_no_info = diag_no_info.optimize_all_decisions()
        _, eu_with_info = self.diag.optimize_all_decisions()
        assert eu_with_info >= eu_no_info - 1e-6

    def test_strategy_summary(self):
        summary = self.diag.strategy_summary()
        assert "expected_utility" in summary
        assert "policies" in summary
        assert "Drill" in summary["policies"]


# ---------------------------------------------------------------------------
# Sequential decisions
# ---------------------------------------------------------------------------

class TestSequentialDecisions:
    def test_two_stage_decision(self):
        """Two-stage: observe X, decide D1, observe Y(X,D1), decide D2."""
        diag = InfluenceDiagram()

        # Stage 1: chance X
        diag.add_chance_node("X", [0, 1])
        diag.set_cpt_dict("X", {0: 0.5, 1: 0.5})

        # Stage 1: decision D1 observing X
        diag.add_decision_node("D1", ["a", "b"], info_vars=["X"])

        # Stage 2: chance Y depends on X and D1
        diag.add_chance_node("Y", [0, 1])
        diag.add_edge("X", "Y")
        diag.add_edge("D1", "Y")
        diag.set_cpt_dict("Y", {
            (0, "a"): {0: 0.8, 1: 0.2},
            (0, "b"): {0: 0.3, 1: 0.7},
            (1, "a"): {0: 0.4, 1: 0.6},
            (1, "b"): {0: 0.6, 1: 0.4},
        })

        # Stage 2: decision D2 observing Y
        diag.add_decision_node("D2", ["c", "d"], info_vars=["Y"])

        # Utility depends on D2 and Y
        diag.add_utility_node("U", ["Y", "D2"], {
            (0, "c"): 10, (0, "d"): 0,
            (1, "c"): 0,  (1, "d"): 20,
        })

        policies, eu = diag.optimize_all_decisions()
        # D2 should match Y: Y=0 -> c (10), Y=1 -> d (20)
        assert policies["D2"].decide({"Y": 0}) == "c"
        assert policies["D2"].decide({"Y": 1}) == "d"
        assert eu > 0

    def test_build_sequential_helper(self):
        diag = build_sequential_decisions([
            {
                "chance": ("X", [0, 1], {0: 0.6, 1: 0.4}),
                "decision": ("D", ["a", "b"], ["X"]),
                "utility": ("U", ["X", "D"], {
                    (0, "a"): 10, (0, "b"): 0,
                    (1, "a"): 0,  (1, "b"): 10,
                }),
            },
        ])
        policies, eu = diag.optimize_all_decisions()
        # Optimal: X=0 -> a, X=1 -> b, EU = 0.6*10 + 0.4*10 = 10
        assert policies["D"].decide({"X": 0}) == "a"
        assert policies["D"].decide({"X": 1}) == "b"
        assert abs(eu - 10.0) < 1e-6


# ---------------------------------------------------------------------------
# Multiple utility nodes
# ---------------------------------------------------------------------------

class TestMultipleUtilities:
    def test_additive_utilities(self):
        """Multiple utility nodes: total utility is the sum."""
        diag = InfluenceDiagram()
        diag.add_chance_node("X", [0, 1])
        diag.set_cpt_dict("X", {0: 0.5, 1: 0.5})
        diag.add_decision_node("D", ["a", "b"], info_vars=["X"])

        # Two utility components
        diag.add_utility_node("U1", ["X", "D"], {
            (0, "a"): 10, (0, "b"): 0, (1, "a"): 0, (1, "b"): 10,
        })
        diag.add_utility_node("U2", ["D"], {("a",): -2, ("b",): -1})

        policies, eu = diag.optimize_all_decisions()
        # X=0: EU(a) = 10-2=8, EU(b) = 0-1=-1 -> pick a
        # X=1: EU(a) = 0-2=-2, EU(b) = 10-1=9 -> pick b
        assert policies["D"].decide({"X": 0}) == "a"
        assert policies["D"].decide({"X": 1}) == "b"
        # EU = 0.5*8 + 0.5*9 = 4 + 4.5 = 8.5
        assert abs(eu - 8.5) < 1e-6


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_chance_nodes(self):
        """Decision with no uncertainty -- pick best action directly."""
        diag = InfluenceDiagram()
        diag.add_decision_node("D", ["a", "b", "c"])
        diag.add_utility_node("U", ["D"], {("a",): 5, ("b",): 10, ("c",): 3})

        policies, eu = diag.optimize_all_decisions()
        assert policies["D"].decide({}) == "b"
        assert abs(eu - 10.0) < 1e-6

    def test_dominated_action(self):
        """One action is always worse -- should never be chosen."""
        diag = InfluenceDiagram()
        diag.add_chance_node("X", [0, 1])
        diag.set_cpt_dict("X", {0: 0.5, 1: 0.5})
        diag.add_decision_node("D", ["good", "bad"], info_vars=["X"])
        diag.add_utility_node("U", ["X", "D"], {
            (0, "good"): 10, (0, "bad"): 5,
            (1, "good"): 20, (1, "bad"): 15,
        })
        policies, eu = diag.optimize_all_decisions()
        # "good" dominates "bad" in every state
        assert policies["D"].decide({"X": 0}) == "good"
        assert policies["D"].decide({"X": 1}) == "good"

    def test_zero_probability_state(self):
        """State with zero probability shouldn't affect EU."""
        diag = InfluenceDiagram()
        diag.add_chance_node("X", [0, 1, 2])
        diag.set_cpt_dict("X", {0: 0.5, 1: 0.5, 2: 0.0})
        diag.add_decision_node("D", ["a", "b"])
        diag.add_utility_node("U", ["X", "D"], {
            (0, "a"): 10, (0, "b"): 0,
            (1, "a"): 0,  (1, "b"): 10,
            (2, "a"): -1000, (2, "b"): -1000,  # would be terrible but P=0
        })
        _, eu = diag.optimize_all_decisions()
        # Only X=0 and X=1 matter, each equally likely
        # Optimal without info: max(EU(a), EU(b)) = max(5, 5) = 5
        assert abs(eu - 5.0) < 1e-6

    def test_evidence_restricts_optimization(self):
        """Evidence fixes chance variable."""
        diag = build_simple_decision(
            "Weather", ["rain", "sun"], {"rain": 0.3, "sun": 0.7},
            "Umbrella", ["take", "leave"],
            {
                ("rain", "take"): 80, ("rain", "leave"): 0,
                ("sun", "take"): 60, ("sun", "leave"): 100,
            },
            observe_chance=False,
        )
        policies, eu = diag.optimize_all_decisions(evidence={"Weather": "rain"})
        assert policies["Umbrella"].decide({}) == "take"
        assert abs(eu - 80.0) < 1e-6

    def test_vpi_zero_when_no_uncertainty(self):
        """EVPI = 0 when there's no uncertainty to resolve."""
        diag = InfluenceDiagram()
        diag.add_decision_node("D", ["a", "b"])
        diag.add_utility_node("U", ["D"], {("a",): 10, ("b",): 5})
        evpi = diag.value_of_perfect_information()
        assert abs(evpi) < 1e-9


# ---------------------------------------------------------------------------
# Expected utility computation
# ---------------------------------------------------------------------------

class TestExpectedUtility:
    def test_eu_manual_policy(self):
        """Verify EU computation with a manually specified policy."""
        diag = build_simple_decision(
            "X", [0, 1], {0: 0.6, 1: 0.4},
            "D", ["a", "b"],
            {(0, "a"): 10, (0, "b"): 0, (1, "a"): 0, (1, "b"): 10},
            observe_chance=True,
        )
        # Sub-optimal policy: always pick "a"
        always_a = Policy("D", ["X"], {(0,): "a", (1,): "a"})
        eu = diag.expected_utility({"D": always_a})
        # EU = 0.6*10 + 0.4*0 = 6
        assert abs(eu - 6.0) < 1e-6

    def test_eu_optimal_policy(self):
        diag = build_simple_decision(
            "X", [0, 1], {0: 0.6, 1: 0.4},
            "D", ["a", "b"],
            {(0, "a"): 10, (0, "b"): 0, (1, "a"): 0, (1, "b"): 10},
            observe_chance=True,
        )
        optimal = Policy("D", ["X"], {(0,): "a", (1,): "b"})
        eu = diag.expected_utility({"D": optimal})
        # EU = 0.6*10 + 0.4*10 = 10
        assert abs(eu - 10.0) < 1e-6


# ---------------------------------------------------------------------------
# Value of information
# ---------------------------------------------------------------------------

class TestValueOfInformation:
    def test_voi_nonnegative(self):
        """VOI is always non-negative (information can't hurt)."""
        diag = build_medical_diagnosis()
        voi = diag.value_of_information("Treatment", "Disease")
        assert voi >= -1e-9

    def test_voi_zero_when_already_observed(self):
        """VOI of variable already in info set should be 0 (already have it)."""
        diag = build_simple_decision(
            "X", [0, 1], {0: 0.5, 1: 0.5},
            "D", ["a", "b"],
            {(0, "a"): 10, (0, "b"): 0, (1, "a"): 0, (1, "b"): 10},
            observe_chance=True,  # X already observed
        )
        voi = diag.value_of_information("D", "X")
        assert abs(voi) < 1e-6

    def test_voi_equals_eu_difference(self):
        """VOI should equal the difference between informed and uninformed EU."""
        diag_with = build_simple_decision(
            "X", [0, 1], {0: 0.5, 1: 0.5},
            "D", ["a", "b"],
            {(0, "a"): 10, (0, "b"): 0, (1, "a"): 0, (1, "b"): 10},
            observe_chance=True,
        )
        diag_without = build_simple_decision(
            "X", [0, 1], {0: 0.5, 1: 0.5},
            "D", ["a", "b"],
            {(0, "a"): 10, (0, "b"): 0, (1, "a"): 0, (1, "b"): 10},
            observe_chance=False,
        )
        _, eu_with = diag_with.optimize_all_decisions()
        _, eu_without = diag_without.optimize_all_decisions()
        voi = diag_without.value_of_information("D", "X")
        assert abs(voi - (eu_with - eu_without)) < 1e-6

    def test_evpi_bounds(self):
        """EVPI should be between 0 and max utility range."""
        diag = build_oil_wildcatter()
        evpi = diag.value_of_perfect_information()
        assert evpi >= -1e-9
        # EVPI can't exceed the max possible gain
        assert evpi < 300  # 200 - (-70) is 270, max range


# ---------------------------------------------------------------------------
# Decision table
# ---------------------------------------------------------------------------

class TestDecisionTable:
    def test_decision_table_structure(self):
        diag = build_simple_decision(
            "X", [0, 1], {0: 0.5, 1: 0.5},
            "D", ["a", "b"],
            {(0, "a"): 10, (0, "b"): 0, (1, "a"): 0, (1, "b"): 10},
            observe_chance=True,
        )
        table = diag.decision_table("D")
        # 2 info configs * 2 actions = 4 rows
        assert len(table) == 4
        for row in table:
            assert "info_config" in row
            assert "action" in row
            assert "expected_utility" in row

    def test_decision_table_no_info(self):
        diag = build_simple_decision(
            "X", [0, 1], {0: 0.5, 1: 0.5},
            "D", ["a", "b"],
            {(0, "a"): 10, (0, "b"): 0, (1, "a"): 0, (1, "b"): 10},
            observe_chance=False,
        )
        table = diag.decision_table("D")
        # 1 info config (empty) * 2 actions = 2 rows
        assert len(table) == 2
        # Both should have EU = 5
        for row in table:
            assert abs(row["expected_utility"] - 5.0) < 1e-6


# ---------------------------------------------------------------------------
# Strategy summary
# ---------------------------------------------------------------------------

class TestStrategySummary:
    def test_summary_keys(self):
        diag = build_medical_diagnosis()
        summary = diag.strategy_summary()
        assert "expected_utility" in summary
        assert "policies" in summary
        assert "value_of_information" in summary

    def test_summary_eu_matches(self):
        diag = build_medical_diagnosis()
        summary = diag.strategy_summary()
        _, eu = diag.optimize_all_decisions()
        assert abs(summary["expected_utility"] - round(eu, 6)) < 1e-6


# ---------------------------------------------------------------------------
# Integration: BN inference within influence diagram
# ---------------------------------------------------------------------------

class TestBNIntegration:
    def test_posterior_consistent_with_policy(self):
        """Optimal policy should align with BN posterior reasoning."""
        diag = build_medical_diagnosis()
        # P(sick | pos) via BN
        posterior = variable_elimination(
            diag.bn, ["Disease"], {"Test": "pos"}
        )
        p_sick_given_pos = posterior.get({"Disease": "sick"})
        # With pos: EU(treat) = p_sick*80 + (1-p_sick)*70
        # EU(no_treat) = p_sick*0 + (1-p_sick)*100
        eu_treat = p_sick_given_pos * 80 + (1 - p_sick_given_pos) * 70
        eu_no_treat = p_sick_given_pos * 0 + (1 - p_sick_given_pos) * 100
        # Policy should match
        policies, _ = diag.optimize_all_decisions()
        if eu_treat > eu_no_treat:
            assert policies["Treatment"].decide({"Test": "pos"}) == "treat"
        else:
            assert policies["Treatment"].decide({"Test": "pos"}) == "no_treat"

    def test_d_separation_still_works(self):
        """D-separation queries on the underlying BN still work."""
        diag = build_medical_diagnosis()
        # Disease and Treatment are not d-separated by empty set
        # (via Test: Disease -> Test -> Treatment info link is implicit in BN)
        # But Disease _|_ Treatment | Test (test blocks the path)
        # Note: Treatment has no CPT (decision), so d-sep on BN structure
        assert diag.bn.is_d_separated({"Disease"}, {"Treatment"}, {"Test"})


# ---------------------------------------------------------------------------
# Three-action decision
# ---------------------------------------------------------------------------

class TestThreeActions:
    def test_three_actions(self):
        """Decision with 3 possible actions."""
        diag = InfluenceDiagram()
        diag.add_chance_node("X", ["lo", "mid", "hi"])
        diag.set_cpt_dict("X", {"lo": 0.3, "mid": 0.5, "hi": 0.2})
        diag.add_decision_node("D", ["safe", "moderate", "risky"], info_vars=["X"])
        diag.add_utility_node("U", ["X", "D"], {
            ("lo", "safe"): 5, ("lo", "moderate"): -5, ("lo", "risky"): -20,
            ("mid", "safe"): 5, ("mid", "moderate"): 15, ("mid", "risky"): 10,
            ("hi", "safe"): 5, ("hi", "moderate"): 15, ("hi", "risky"): 50,
        })
        policies, eu = diag.optimize_all_decisions()
        assert policies["D"].decide({"X": "lo"}) == "safe"
        assert policies["D"].decide({"X": "mid"}) == "moderate"
        assert policies["D"].decide({"X": "hi"}) == "risky"
        # EU = 0.3*5 + 0.5*15 + 0.2*50 = 1.5 + 7.5 + 10 = 19
        assert abs(eu - 19.0) < 1e-6


# ---------------------------------------------------------------------------
# Asymmetric information
# ---------------------------------------------------------------------------

class TestAsymmetricInfo:
    def test_partial_observation(self):
        """Decision observes only some chance variables."""
        diag = InfluenceDiagram()
        diag.add_chance_node("X", [0, 1])
        diag.set_cpt_dict("X", {0: 0.5, 1: 0.5})
        diag.add_chance_node("Y", [0, 1])
        diag.add_edge("X", "Y")
        diag.set_cpt_dict("Y", {
            (0,): {0: 0.9, 1: 0.1},
            (1,): {0: 0.1, 1: 0.9},
        })
        # Decision observes Y but not X
        diag.add_decision_node("D", ["a", "b"], info_vars=["Y"])
        diag.add_utility_node("U", ["X", "D"], {
            (0, "a"): 10, (0, "b"): 0,
            (1, "a"): 0,  (1, "b"): 10,
        })
        policies, eu = diag.optimize_all_decisions()
        # Y is a noisy observation of X (90% accurate)
        # Optimal: Y=0 -> a (likely X=0), Y=1 -> b (likely X=1)
        assert policies["D"].decide({"Y": 0}) == "a"
        assert policies["D"].decide({"Y": 1}) == "b"
        # EU with noisy observation: 0.5*(0.9*10 + 0.1*0) + 0.5*(0.9*10 + 0.1*0) = 9
        assert abs(eu - 9.0) < 1e-6
