"""Tests for V085: Quantitative Language Inclusion."""

import sys
import os
import math
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V083_weighted_automata'))

from weighted_automata import (
    WFA, BooleanSemiring, TropicalSemiring, MaxPlusSemiring,
    ProbabilitySemiring, CountingSemiring, ViterbiSemiring,
    MinMaxSemiring, LogSemiring, make_semiring,
    wfa_from_word, wfa_from_symbol, wfa_epsilon, wfa_empty,
    wfa_union, wfa_concat, wfa_star, wfa_intersect,
    wfa_run_weight, wfa_trim,
)
from quantitative_inclusion import (
    InclusionVerdict, InclusionWitness, InclusionResult,
    EquivalenceResult, BisimulationResult, DistanceResult,
    RefinementResult, ComprehensiveResult,
    bounded_inclusion_check, bounded_equivalence_check,
    build_product_wfa, product_inclusion_check,
    weighted_bisimulation, simulation_inclusion_check,
    smt_tropical_inclusion, quantitative_distance,
    check_refinement, language_quotient, approximate_inclusion,
    comprehensive_check,
    check_inclusion, check_equivalence, check_strict_inclusion,
    inclusion_summary, compare_inclusions,
    _numeric_leq, _weight_diff, _get_alphabet,
)


# ============================================================
# Helpers
# ============================================================

def make_tropical_wfa(states, initial, final, transitions):
    """Helper to build a tropical WFA.

    states: list of state ids
    initial: dict {state: weight}
    final: dict {state: weight}
    transitions: list of (src, label, dst, weight)
    """
    sr = TropicalSemiring()
    wfa = WFA(sr)
    for s in states:
        iw = initial.get(s, sr.zero())
        fw = final.get(s, sr.zero())
        wfa.add_state(s, initial=iw, final=fw)
    for src, label, dst, w in transitions:
        wfa.add_transition(src, label, dst, w)
    return wfa


def make_probability_wfa(states, initial, final, transitions):
    """Helper to build a probability WFA."""
    sr = ProbabilitySemiring()
    wfa = WFA(sr)
    for s in states:
        iw = initial.get(s, sr.zero())
        fw = final.get(s, sr.zero())
        wfa.add_state(s, initial=iw, final=fw)
    for src, label, dst, w in transitions:
        wfa.add_transition(src, label, dst, w)
    return wfa


def make_counting_wfa(states, initial, final, transitions):
    """Helper to build a counting WFA."""
    sr = CountingSemiring()
    wfa = WFA(sr)
    for s in states:
        iw = initial.get(s, sr.zero())
        fw = final.get(s, sr.zero())
        wfa.add_state(s, initial=iw, final=fw)
    for src, label, dst, w in transitions:
        wfa.add_transition(src, label, dst, w)
    return wfa


# ============================================================
# Section 1: Numeric Ordering
# ============================================================

class TestNumericOrdering:
    def test_tropical_leq(self):
        sr = TropicalSemiring()
        assert _numeric_leq(sr, 3.0, 5.0)
        assert _numeric_leq(sr, 5.0, 5.0)
        assert not _numeric_leq(sr, 7.0, 5.0)

    def test_tropical_inf(self):
        sr = TropicalSemiring()
        assert _numeric_leq(sr, 3.0, float('inf'))
        assert _numeric_leq(sr, float('inf'), float('inf'))
        assert not _numeric_leq(sr, float('inf'), 3.0)

    def test_maxplus_leq(self):
        sr = MaxPlusSemiring()
        assert _numeric_leq(sr, 3.0, 5.0)
        assert _numeric_leq(sr, float('-inf'), 5.0)
        assert not _numeric_leq(sr, 5.0, 3.0)

    def test_probability_leq(self):
        sr = ProbabilitySemiring()
        assert _numeric_leq(sr, 0.3, 0.5)
        assert _numeric_leq(sr, 0.0, 0.1)
        assert not _numeric_leq(sr, 0.9, 0.1)

    def test_boolean_leq(self):
        sr = BooleanSemiring()
        assert _numeric_leq(sr, False, True)
        assert _numeric_leq(sr, False, False)
        assert _numeric_leq(sr, True, True)
        assert not _numeric_leq(sr, True, False)

    def test_weight_diff(self):
        sr = TropicalSemiring()
        assert _weight_diff(sr, 5.0, 3.0) == 2.0
        assert _weight_diff(sr, 3.0, 5.0) == -2.0
        assert _weight_diff(sr, float('inf'), float('inf')) == 0.0


# ============================================================
# Section 2: Bounded Inclusion (Tropical)
# ============================================================

class TestBoundedInclusionTropical:
    def test_identical_wfas(self):
        """Identical WFAs are included."""
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        result = bounded_inclusion_check(a, b, max_length=3)
        assert result.included
        assert len(result.witnesses) == 0

    def test_smaller_weights_included(self):
        """A with smaller weights is included in B with larger weights."""
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 5.0)]
        )
        result = bounded_inclusion_check(a, b, max_length=3)
        assert result.included

    def test_larger_weight_not_included(self):
        """A with larger weight NOT included in B with smaller weight."""
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 10.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        result = bounded_inclusion_check(a, b, max_length=3)
        assert not result.included
        assert len(result.witnesses) > 0
        w = result.witnesses[0]
        assert w.word == 'a'
        assert w.weight_a == 10.0
        assert w.weight_b == 1.0

    def test_different_structure(self):
        """WFAs with different structures."""
        # A: a -> cost 2, b -> cost 3
        a = make_tropical_wfa(
            [0, 1, 2], {0: 0.0}, {1: 0.0, 2: 0.0},
            [(0, 'a', 1, 2.0), (0, 'b', 2, 3.0)]
        )
        # B: a -> cost 5, b -> cost 5
        b = make_tropical_wfa(
            [0, 1, 2], {0: 0.0}, {1: 0.0, 2: 0.0},
            [(0, 'a', 1, 5.0), (0, 'b', 2, 5.0)]
        )
        result = bounded_inclusion_check(a, b, max_length=3)
        assert result.included

    def test_empty_language(self):
        """WFA with no final states: weight is inf (tropical zero) for all words.
        inf is NOT <= finite, so not included. This is correct for tropical."""
        sr = TropicalSemiring()
        a = WFA(sr)
        a.add_state(0, initial=0.0)
        # No final states -> all words have weight inf

        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        result = bounded_inclusion_check(a, b, max_length=3)
        # inf > 1.0, so not included. But empty string: a has inf, b has inf -> ok.
        # "a": a has inf, b has 1.0 -> inf > 1.0 -> not included
        assert not result.included

    def test_loop_wfa(self):
        """WFA with loops."""
        # A: a* with cost 1 per 'a'
        a = make_tropical_wfa(
            [0], {0: 0.0}, {0: 0.0},
            [(0, 'a', 0, 1.0)]
        )
        # B: a* with cost 2 per 'a'
        b = make_tropical_wfa(
            [0], {0: 0.0}, {0: 0.0},
            [(0, 'a', 0, 2.0)]
        )
        result = bounded_inclusion_check(a, b, max_length=4)
        assert result.included  # A's cost <= B's cost for all strings


# ============================================================
# Section 3: Bounded Inclusion (Probability)
# ============================================================

class TestBoundedInclusionProbability:
    def test_prob_included(self):
        """Lower probability included in higher."""
        a = make_probability_wfa(
            [0, 1], {0: 1.0}, {1: 1.0},
            [(0, 'a', 1, 0.3)]
        )
        b = make_probability_wfa(
            [0, 1], {0: 1.0}, {1: 1.0},
            [(0, 'a', 1, 0.7)]
        )
        result = bounded_inclusion_check(a, b, max_length=3)
        assert result.included

    def test_prob_not_included(self):
        """Higher probability not included in lower."""
        a = make_probability_wfa(
            [0, 1], {0: 1.0}, {1: 1.0},
            [(0, 'a', 1, 0.9)]
        )
        b = make_probability_wfa(
            [0, 1], {0: 1.0}, {1: 1.0},
            [(0, 'a', 1, 0.1)]
        )
        result = bounded_inclusion_check(a, b, max_length=3)
        assert not result.included

    def test_prob_equal(self):
        """Equal probabilities are included both ways."""
        a = make_probability_wfa(
            [0, 1], {0: 1.0}, {1: 1.0},
            [(0, 'a', 1, 0.5)]
        )
        b = make_probability_wfa(
            [0, 1], {0: 1.0}, {1: 1.0},
            [(0, 'a', 1, 0.5)]
        )
        result = bounded_inclusion_check(a, b, max_length=3)
        assert result.included


# ============================================================
# Section 4: Bounded Equivalence
# ============================================================

class TestBoundedEquivalence:
    def test_equivalent_wfas(self):
        """Identical WFAs are equivalent."""
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 3.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 3.0)]
        )
        result = bounded_equivalence_check(a, b, max_length=3)
        assert result.equivalent

    def test_not_equivalent(self):
        """Different weights are not equivalent."""
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 2.0)]
        )
        result = bounded_equivalence_check(a, b, max_length=3)
        assert not result.equivalent
        assert len(result.witnesses) > 0

    def test_empty_wfas_equivalent(self):
        """Two empty WFAs are equivalent."""
        sr = TropicalSemiring()
        a = WFA(sr)
        a.add_state(0, initial=0.0)
        b = WFA(sr)
        b.add_state(0, initial=0.0)
        result = bounded_equivalence_check(a, b, max_length=3)
        assert result.equivalent

    def test_structurally_different_but_equivalent(self):
        """Different structure, same weights."""
        # A: 0 -a/2-> 1 -b/3-> 2 (total ab = 5)
        a = make_tropical_wfa(
            [0, 1, 2], {0: 0.0}, {2: 0.0},
            [(0, 'a', 1, 2.0), (1, 'b', 2, 3.0)]
        )
        # B: 0 -a/1-> 1 -b/4-> 2 (total ab = 5)
        b = make_tropical_wfa(
            [0, 1, 2], {0: 0.0}, {2: 0.0},
            [(0, 'a', 1, 1.0), (1, 'b', 2, 4.0)]
        )
        result = bounded_equivalence_check(a, b, max_length=3)
        assert result.equivalent


# ============================================================
# Section 5: Weighted Bisimulation
# ============================================================

class TestWeightedBisimulation:
    def test_identical_bisimilar(self):
        """Identical WFAs are bisimilar."""
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        result = weighted_bisimulation(a, b)
        assert result.bisimilar

    def test_different_weight_not_bisimilar(self):
        """Different weights break bisimulation."""
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 2.0)]
        )
        result = weighted_bisimulation(a, b)
        assert not result.bisimilar
        assert result.distinguishing_word is not None

    def test_different_final_weight(self):
        """Different final weights break bisimulation."""
        sr = TropicalSemiring()
        a = WFA(sr)
        a.add_state(0, initial=0.0, final=1.0)
        b = WFA(sr)
        b.add_state(0, initial=0.0, final=2.0)

        result = weighted_bisimulation(a, b)
        assert not result.bisimilar

    def test_partition_info(self):
        """Bisimulation produces partition information."""
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        result = weighted_bisimulation(a, b)
        assert result.partition is not None
        assert len(result.partition) > 0


# ============================================================
# Section 6: Simulation-Based Inclusion
# ============================================================

class TestSimulationInclusion:
    def test_simulation_found(self):
        """Simulation establishes inclusion."""
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 5.0)]
        )
        result = simulation_inclusion_check(a, b)
        assert result.included

    def test_simulation_not_found_fallback(self):
        """When simulation fails, falls back to bounded check."""
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 10.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        result = simulation_inclusion_check(a, b)
        assert not result.included


# ============================================================
# Section 7: SMT Tropical Inclusion
# ============================================================

class TestSMTTropicalInclusion:
    def test_tropical_included(self):
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 5.0)]
        )
        result = smt_tropical_inclusion(a, b, max_length=3)
        assert result.included

    def test_tropical_not_included(self):
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 10.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        result = smt_tropical_inclusion(a, b, max_length=3)
        assert not result.included

    def test_non_tropical_fallback(self):
        """Non-tropical semiring falls back to bounded check."""
        a = make_probability_wfa(
            [0, 1], {0: 1.0}, {1: 1.0},
            [(0, 'a', 1, 0.3)]
        )
        b = make_probability_wfa(
            [0, 1], {0: 1.0}, {1: 1.0},
            [(0, 'a', 1, 0.7)]
        )
        result = smt_tropical_inclusion(a, b, max_length=3)
        assert result.included


# ============================================================
# Section 8: Quantitative Distance
# ============================================================

class TestQuantitativeDistance:
    def test_identical_zero_distance(self):
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 3.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 3.0)]
        )
        result = quantitative_distance(a, b, max_length=3)
        assert result.distance == 0.0

    def test_different_weights_positive_distance(self):
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 4.0)]
        )
        result = quantitative_distance(a, b, max_length=3)
        assert result.distance == 3.0
        assert result.worst_word == 'a'

    def test_distance_with_loops(self):
        """Distance grows with loop iterations."""
        a = make_tropical_wfa(
            [0], {0: 0.0}, {0: 0.0},
            [(0, 'a', 0, 1.0)]
        )
        b = make_tropical_wfa(
            [0], {0: 0.0}, {0: 0.0},
            [(0, 'a', 0, 3.0)]
        )
        result = quantitative_distance(a, b, max_length=4)
        # "aaaa" gives distance 4*2 = 8
        assert result.distance > 0


# ============================================================
# Section 9: Refinement Checking
# ============================================================

class TestRefinement:
    def test_mutual_refinement_equivalent(self):
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 5.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 5.0)]
        )
        result = check_refinement(a, b, max_length=3)
        assert result.refines
        assert result.direction == "equivalent"

    def test_a_refines_b(self):
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 5.0)]
        )
        result = check_refinement(a, b, max_length=3)
        assert result.refines
        assert result.direction == "A refines B"

    def test_b_refines_a(self):
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 10.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 5.0)]
        )
        result = check_refinement(a, b, max_length=3)
        assert not result.refines
        assert result.direction == "B refines A"

    def test_incomparable(self):
        """Neither refines the other."""
        # A: a -> 1, b -> 10
        a = make_tropical_wfa(
            [0, 1, 2], {0: 0.0}, {1: 0.0, 2: 0.0},
            [(0, 'a', 1, 1.0), (0, 'b', 2, 10.0)]
        )
        # B: a -> 5, b -> 3
        b = make_tropical_wfa(
            [0, 1, 2], {0: 0.0}, {1: 0.0, 2: 0.0},
            [(0, 'a', 1, 5.0), (0, 'b', 2, 3.0)]
        )
        result = check_refinement(a, b, max_length=3)
        assert not result.refines
        assert result.direction == "incomparable"


# ============================================================
# Section 10: Language Quotient
# ============================================================

class TestLanguageQuotient:
    def test_quotient_equal_wfas(self):
        a = make_probability_wfa(
            [0, 1], {0: 1.0}, {1: 1.0},
            [(0, 'a', 1, 0.5)]
        )
        b = make_probability_wfa(
            [0, 1], {0: 1.0}, {1: 1.0},
            [(0, 'a', 1, 0.5)]
        )
        result = language_quotient(a, b, max_length=2)
        if result["n_words"] > 0:
            assert result["min_ratio"] == pytest.approx(1.0)
            assert result["max_ratio"] == pytest.approx(1.0)

    def test_quotient_half(self):
        a = make_probability_wfa(
            [0, 1], {0: 1.0}, {1: 1.0},
            [(0, 'a', 1, 0.3)]
        )
        b = make_probability_wfa(
            [0, 1], {0: 1.0}, {1: 1.0},
            [(0, 'a', 1, 0.6)]
        )
        result = language_quotient(a, b, max_length=2)
        if result["n_words"] > 0:
            assert result["max_ratio"] <= 1.0 + 1e-9
            assert result["all_leq_1"]


# ============================================================
# Section 11: Approximate Inclusion
# ============================================================

class TestApproximateInclusion:
    def test_exact_inclusion_is_approximate(self):
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 5.0)]
        )
        result = approximate_inclusion(a, b, epsilon=0.1, max_length=3)
        assert result.included

    def test_approximate_with_tolerance(self):
        """Slight violation within epsilon is accepted."""
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 5.05)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 5.0)]
        )
        result = approximate_inclusion(a, b, epsilon=0.1, max_length=3)
        assert result.included

    def test_approximate_beyond_tolerance(self):
        """Large violation beyond epsilon is rejected."""
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 10.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 5.0)]
        )
        result = approximate_inclusion(a, b, epsilon=0.1, max_length=3)
        assert not result.included


# ============================================================
# Section 12: Comprehensive Check
# ============================================================

class TestComprehensiveCheck:
    def test_comprehensive_included(self):
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 5.0)]
        )
        result = comprehensive_check(a, b, max_length=3)
        assert result.verdict in (InclusionVerdict.INCLUDED, InclusionVerdict.EQUIVALENT)
        assert result.bounded_result is not None

    def test_comprehensive_equivalent(self):
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 3.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 3.0)]
        )
        result = comprehensive_check(a, b, max_length=3)
        assert result.verdict == InclusionVerdict.EQUIVALENT
        assert result.bisimulation_result is not None
        assert result.bisimulation_result.bisimilar

    def test_comprehensive_not_included(self):
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 10.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        result = comprehensive_check(a, b, max_length=3)
        assert result.verdict == InclusionVerdict.NOT_INCLUDED

    def test_comprehensive_with_distance(self):
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 2.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 5.0)]
        )
        result = comprehensive_check(a, b, max_length=3, run_distance=True)
        assert result.distance is not None
        assert result.distance.distance == 3.0


# ============================================================
# Section 13: Convenience APIs
# ============================================================

class TestConvenienceAPIs:
    def test_check_inclusion_api(self):
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 2.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 5.0)]
        )
        result = check_inclusion(a, b)
        assert result.included

    def test_check_equivalence_api(self):
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 3.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 3.0)]
        )
        result = check_equivalence(a, b)
        assert result.equivalent

    def test_check_strict_inclusion(self):
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 5.0)]
        )
        result = check_strict_inclusion(a, b)
        assert result.included
        assert result.stats.get("is_strict") == True

    def test_strict_inclusion_equivalent(self):
        """Equivalent WFAs are not strictly included."""
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 3.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 3.0)]
        )
        result = check_strict_inclusion(a, b)
        assert result.verdict == InclusionVerdict.EQUIVALENT

    def test_inclusion_summary(self):
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 10.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        result = bounded_inclusion_check(a, b, max_length=3)
        summary = inclusion_summary(result)
        assert "NOT_INCLUDED" in summary
        assert "Witnesses" in summary


# ============================================================
# Section 14: Product Construction
# ============================================================

class TestProductConstruction:
    def test_product_basic(self):
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 2.0)]
        )
        product, state_map = build_product_wfa(a, b)
        assert len(product.states) > 0
        assert len(state_map) > 0

    def test_product_state_mapping(self):
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 2.0)]
        )
        _, state_map = build_product_wfa(a, b)
        assert (0, 0) in state_map  # initial pair
        assert (1, 1) in state_map  # final pair

    def test_product_inclusion(self):
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 5.0)]
        )
        result = product_inclusion_check(a, b, max_length=3)
        assert result.included


# ============================================================
# Section 15: Counting Semiring
# ============================================================

class TestCountingSemiring:
    def test_counting_inclusion(self):
        """More runs = higher count = not included."""
        # A: a has 2 accepting runs (via two paths)
        a = make_counting_wfa(
            [0, 1, 2, 3], {0: 1}, {2: 1, 3: 1},
            [(0, 'a', 1, 1), (1, 'a', 2, 1), (0, 'a', 3, 1)]
        )
        # B: a has 1 accepting run
        b = make_counting_wfa(
            [0, 1], {0: 1}, {1: 1},
            [(0, 'a', 1, 1)]
        )
        # A("a") = 1 (only the direct path 0->3), B("a") = 1
        # A("aa") = 1 (path 0->1->2), B("aa") = 0
        result = check_inclusion(a, b, max_length=3)
        assert not result.included

    def test_counting_equivalence(self):
        a = make_counting_wfa(
            [0, 1], {0: 1}, {1: 1},
            [(0, 'a', 1, 1)]
        )
        b = make_counting_wfa(
            [0, 1], {0: 1}, {1: 1},
            [(0, 'a', 1, 1)]
        )
        result = check_equivalence(a, b, max_length=3)
        assert result.equivalent


# ============================================================
# Section 16: Multi-WFA Comparison
# ============================================================

class TestCompareInclusions:
    def test_compare_three_wfas(self):
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 5.0)]
        )
        c = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 10.0)]
        )
        result = compare_inclusions(a, b, c, max_length=3)
        assert result["A_leq_B"]
        assert result["A_leq_C"]
        assert result["B_leq_C"]
        assert result["tighter"] == "B"  # B is tighter than C

    def test_compare_equivalent(self):
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 5.0)]
        )
        c = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 5.0)]
        )
        result = compare_inclusions(a, b, c, max_length=3)
        assert result["tighter"] == "equivalent"


# ============================================================
# Section 17: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_single_state_wfa(self):
        """WFA with one state that is both initial and final."""
        sr = TropicalSemiring()
        a = WFA(sr)
        a.add_state(0, initial=0.0, final=0.0)

        b = WFA(sr)
        b.add_state(0, initial=0.0, final=0.0)

        result = check_inclusion(a, b, max_length=2)
        assert result.included

    def test_no_accepting_paths(self):
        """WFA with no final states: weight is inf for all words.
        inf > finite, so not included in tropical."""
        sr = TropicalSemiring()
        a = WFA(sr)
        a.add_state(0, initial=0.0)
        a.add_state(1)
        a.add_transition(0, 'a', 1, 1.0)

        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 5.0)]
        )
        result = check_inclusion(a, b, max_length=3)
        # A("a") = inf (no final state), B("a") = 5.0. inf > 5.0, not included.
        assert not result.included

    def test_self_loop_wfa(self):
        """WFA with self-loop."""
        a = make_tropical_wfa(
            [0], {0: 0.0}, {0: 0.0},
            [(0, 'a', 0, 2.0)]
        )
        b = make_tropical_wfa(
            [0], {0: 0.0}, {0: 0.0},
            [(0, 'a', 0, 3.0)]
        )
        result = check_inclusion(a, b, max_length=5)
        assert result.included

    def test_disjoint_alphabets(self):
        """WFAs over different alphabets."""
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 1.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'b', 1, 1.0)]
        )
        # "a" has weight 1.0 in A, inf in B
        result = check_inclusion(a, b, max_length=2)
        assert not result.included

    def test_multiple_paths_same_word(self):
        """WFA with multiple paths for the same word (nondeterministic)."""
        # A: "a" via two paths, weights 3 and 5. Tropical min = 3
        a = make_tropical_wfa(
            [0, 1, 2], {0: 0.0}, {1: 0.0, 2: 0.0},
            [(0, 'a', 1, 3.0), (0, 'a', 2, 5.0)]
        )
        # B: "a" via one path, weight 4
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 4.0)]
        )
        # A("a") = min(3, 5) = 3, B("a") = 4. 3 <= 4, so included.
        result = check_inclusion(a, b, max_length=3)
        assert result.included

    def test_empty_alphabet(self):
        """WFAs with no transitions (only empty string)."""
        sr = TropicalSemiring()
        a = WFA(sr)
        a.add_state(0, initial=0.0, final=2.0)
        b = WFA(sr)
        b.add_state(0, initial=0.0, final=5.0)
        result = check_inclusion(a, b, max_length=3)
        assert result.included


# ============================================================
# Section 18: MaxPlus Semiring
# ============================================================

class TestMaxPlusSemiring:
    def test_maxplus_inclusion(self):
        sr = MaxPlusSemiring()
        a = WFA(sr)
        a.add_state(0, initial=0.0, final=0.0)
        a.add_state(1, final=0.0)
        a.add_transition(0, 'a', 1, 3.0)

        b = WFA(sr)
        b.add_state(0, initial=0.0, final=0.0)
        b.add_state(1, final=0.0)
        b.add_transition(0, 'a', 1, 5.0)

        result = check_inclusion(a, b, max_length=3)
        assert result.included

    def test_maxplus_not_included(self):
        sr = MaxPlusSemiring()
        a = WFA(sr)
        a.add_state(0, initial=0.0, final=0.0)
        a.add_state(1, final=0.0)
        a.add_transition(0, 'a', 1, 8.0)

        b = WFA(sr)
        b.add_state(0, initial=0.0, final=0.0)
        b.add_state(1, final=0.0)
        b.add_transition(0, 'a', 1, 2.0)

        result = check_inclusion(a, b, max_length=3)
        assert not result.included


# ============================================================
# Section 19: Viterbi Semiring
# ============================================================

class TestViterbiSemiring:
    def test_viterbi_inclusion(self):
        sr = ViterbiSemiring()
        a = WFA(sr)
        a.add_state(0, initial=1.0, final=1.0)
        a.add_state(1, final=1.0)
        a.add_transition(0, 'a', 1, 0.3)

        b = WFA(sr)
        b.add_state(0, initial=1.0, final=1.0)
        b.add_state(1, final=1.0)
        b.add_transition(0, 'a', 1, 0.7)

        result = check_inclusion(a, b, max_length=3)
        assert result.included

    def test_viterbi_equivalence(self):
        sr = ViterbiSemiring()
        a = WFA(sr)
        a.add_state(0, initial=1.0, final=1.0)
        a.add_state(1, final=1.0)
        a.add_transition(0, 'a', 1, 0.5)

        b = WFA(sr)
        b.add_state(0, initial=1.0, final=1.0)
        b.add_state(1, final=1.0)
        b.add_transition(0, 'a', 1, 0.5)

        result = check_equivalence(a, b, max_length=3)
        assert result.equivalent


# ============================================================
# Section 20: Complex Structures
# ============================================================

class TestComplexStructures:
    def test_chain_wfa(self):
        """Chain: 0 -a-> 1 -b-> 2 -c-> 3."""
        a = make_tropical_wfa(
            [0, 1, 2, 3], {0: 0.0}, {3: 0.0},
            [(0, 'a', 1, 1.0), (1, 'b', 2, 2.0), (2, 'c', 3, 3.0)]
        )
        b = make_tropical_wfa(
            [0, 1, 2, 3], {0: 0.0}, {3: 0.0},
            [(0, 'a', 1, 2.0), (1, 'b', 2, 2.0), (2, 'c', 3, 2.0)]
        )
        # A("abc") = 6, B("abc") = 6, equal
        result = check_equivalence(a, b, max_length=4)
        assert result.equivalent

    def test_diamond_wfa(self):
        """Diamond: two paths from 0 to 3."""
        a = make_tropical_wfa(
            [0, 1, 2, 3], {0: 0.0}, {3: 0.0},
            [(0, 'a', 1, 1.0), (0, 'a', 2, 3.0),
             (1, 'b', 3, 2.0), (2, 'b', 3, 1.0)]
        )
        # Path 0-1-3: cost 3, Path 0-2-3: cost 4. Min = 3.
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 0, 1.0), (0, 'b', 1, 2.0)]
        )
        # B("ab") = 3. A("ab") = min(3, 4) = 3. Equal for "ab".
        result = check_inclusion(a, b, max_length=3)
        # Check it runs without error
        assert result.verdict in (InclusionVerdict.INCLUDED, InclusionVerdict.NOT_INCLUDED,
                                  InclusionVerdict.EQUIVALENT)

    def test_branching_wfa(self):
        """WFA with multiple branches."""
        a = make_tropical_wfa(
            [0, 1, 2], {0: 0.0}, {1: 0.0, 2: 0.0},
            [(0, 'a', 1, 2.0), (0, 'b', 2, 4.0)]
        )
        b = make_tropical_wfa(
            [0, 1, 2], {0: 0.0}, {1: 0.0, 2: 0.0},
            [(0, 'a', 1, 3.0), (0, 'b', 2, 5.0)]
        )
        result = check_inclusion(a, b, max_length=3)
        assert result.included


# ============================================================
# Section 21: Alphabet Utilities
# ============================================================

class TestAlphabetUtilities:
    def test_get_alphabet(self):
        a = make_tropical_wfa(
            [0, 1, 2], {0: 0.0}, {1: 0.0, 2: 0.0},
            [(0, 'a', 1, 1.0), (0, 'b', 2, 2.0), (1, 'c', 2, 3.0)]
        )
        alpha = _get_alphabet(a)
        assert set(alpha) == {'a', 'b', 'c'}

    def test_get_alphabet_empty(self):
        sr = TropicalSemiring()
        a = WFA(sr)
        a.add_state(0, initial=0.0, final=0.0)
        alpha = _get_alphabet(a)
        assert alpha == []


# ============================================================
# Section 22: Boolean Semiring
# ============================================================

class TestBooleanSemiring:
    def test_boolean_inclusion(self):
        """NFA language inclusion via boolean semiring."""
        sr = BooleanSemiring()
        # A accepts "a"
        a = WFA(sr)
        a.add_state(0, initial=True, final=False)
        a.add_state(1, final=True)
        a.add_transition(0, 'a', 1, True)

        # B accepts "a" and "b"
        b = WFA(sr)
        b.add_state(0, initial=True, final=False)
        b.add_state(1, final=True)
        b.add_transition(0, 'a', 1, True)
        b.add_transition(0, 'b', 1, True)

        result = check_inclusion(a, b, max_length=3)
        assert result.included

    def test_boolean_not_included(self):
        """A accepts "b" which B doesn't."""
        sr = BooleanSemiring()
        a = WFA(sr)
        a.add_state(0, initial=True, final=False)
        a.add_state(1, final=True)
        a.add_transition(0, 'a', 1, True)
        a.add_transition(0, 'b', 1, True)

        b = WFA(sr)
        b.add_state(0, initial=True, final=False)
        b.add_state(1, final=True)
        b.add_transition(0, 'a', 1, True)

        result = check_inclusion(a, b, max_length=3)
        assert not result.included


# ============================================================
# Section 23: WFA Composition Checks
# ============================================================

class TestCompositionChecks:
    def test_union_preserves_inclusion(self):
        """If A <= B, then A <= union(A, B) in tropical (min)."""
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 3.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 5.0)]
        )
        u = wfa_union(a, b)
        # union in tropical: min of weights
        # u("a") = min(3, 5) = 3, a("a") = 3. Equal.
        result = check_inclusion(a, u, max_length=3)
        assert result.included

    def test_concat_weight_addition(self):
        """Concatenation adds weights in tropical."""
        a = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'a', 1, 2.0)]
        )
        b = make_tropical_wfa(
            [0, 1], {0: 0.0}, {1: 0.0},
            [(0, 'b', 1, 3.0)]
        )
        c = wfa_concat(a, b)
        # c("ab") = 2 + 3 = 5
        w = wfa_run_weight(c, "ab")
        assert w == 5.0


# ============================================================
# Section 24: MinMax Semiring
# ============================================================

class TestMinMaxSemiring:
    def test_minmax_inclusion(self):
        sr = MinMaxSemiring()
        a = WFA(sr)
        a.add_state(0, initial=sr.one(), final=sr.one())
        a.add_state(1, final=sr.one())
        a.add_transition(0, 'a', 1, 3.0)

        b = WFA(sr)
        b.add_state(0, initial=sr.one(), final=sr.one())
        b.add_state(1, final=sr.one())
        b.add_transition(0, 'a', 1, 5.0)

        result = check_inclusion(a, b, max_length=3)
        assert result.included

    def test_minmax_not_included(self):
        sr = MinMaxSemiring()
        a = WFA(sr)
        a.add_state(0, initial=sr.one(), final=sr.one())
        a.add_state(1, final=sr.one())
        a.add_transition(0, 'a', 1, 8.0)

        b = WFA(sr)
        b.add_state(0, initial=sr.one(), final=sr.one())
        b.add_state(1, final=sr.one())
        b.add_transition(0, 'a', 1, 2.0)

        result = check_inclusion(a, b, max_length=3)
        assert not result.included


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
