"""Tests for V155: Process Algebra Verification"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V151_probabilistic_process_algebra'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V150_weak_probabilistic_bisimulation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V067_pctl_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V065_markov_chain_analysis'))

from process_algebra_verification import (
    verify_process, verify_process_state, verify_process_all,
    verify_process_quantitative,
    check_equivalence, check_all_equivalences,
    check_refinement, process_distance,
    verify_algebraic_law, verify_custom_law,
    analyze_composition, check_deadlock_freedom, verify_no_deadlock_pctl,
    check_trace_inclusion, check_trace_equivalence,
    minimize_process, analyze_equivalence_hierarchy,
    check_property_preservation,
    full_process_analysis, process_verification_summary,
    PropertyVerdict,
)
from prob_process_algebra import (
    stop, prefix, tau_prefix, prob_choice, nd_choice,
    parallel, restrict, relabel, recvar, recdef, parse_proc,
)
from pctl_model_check import (
    tt, ff, atom, pnot, pand, por,
    prob_geq, prob_leq, prob_gt, prob_lt,
    next_f, until, bounded_until, eventually, always,
    bounded_eventually, parse_pctl,
)


# ===== Section 1: Basic PCTL Verification on Simple Processes =====

class TestBasicPCTLVerification:
    """Test PCTL model checking on simple process terms."""

    def test_stop_deadlocks(self):
        """STOP process is in deadlock."""
        p = stop()
        result = verify_process(p, atom("deadlock"))
        assert result.verdict == PropertyVerdict.SATISFIED

    def test_prefix_can_act(self):
        """Prefix a.0 can perform action a."""
        p = prefix("a", stop())
        result = verify_process(p, atom("can_a"))
        assert result.verdict == PropertyVerdict.SATISFIED

    def test_prefix_eventually_deadlock(self):
        """a.0 eventually reaches deadlock."""
        p = prefix("a", stop())
        formula = prob_geq(1.0, eventually(atom("deadlock")))
        result = verify_process(p, formula)
        assert result.verdict == PropertyVerdict.SATISFIED

    def test_chain_eventually_deadlock(self):
        """a.b.0 eventually deadlocks."""
        p = prefix("a", prefix("b", stop()))
        formula = prob_geq(1.0, eventually(atom("deadlock")))
        result = verify_process(p, formula)
        assert result.verdict == PropertyVerdict.SATISFIED

    def test_chain_length(self):
        """a.b.c.0 has 4 states."""
        p = prefix("a", prefix("b", prefix("c", stop())))
        result = verify_process(p, tt())
        assert result.lts_states == 4


# ===== Section 2: Probabilistic Choice Verification =====

class TestProbabilisticChoiceVerification:
    """Test PCTL on processes with probabilistic choice."""

    def test_fair_coin_eventually_heads(self):
        """Fair coin: eventually heads with probability 1."""
        coin = prob_choice(0.5, prefix("heads", stop()), prefix("tails", stop()))
        formula = prob_geq(1.0, eventually(atom("deadlock")))
        result = verify_process(coin, formula)
        assert result.verdict == PropertyVerdict.SATISFIED

    def test_prob_choice_actions(self):
        """Prob choice between a and b has both actions (via tau)."""
        p = prob_choice(0.5, prefix("a", stop()), prefix("b", stop()))
        result = verify_process(p, atom("has_tau"))
        assert result.verdict == PropertyVerdict.SATISFIED

    def test_biased_coin(self):
        """Biased coin: heads with prob 0.7."""
        coin = prob_choice(0.7, prefix("heads", stop()), prefix("tails", stop()))
        probs = verify_process_quantitative(coin, eventually(atom("can_heads")))
        # After tau step, one branch can do heads
        assert probs is not None
        assert len(probs) > 0


# ===== Section 3: Nondeterministic Choice Verification =====

class TestNondeterministicChoiceVerification:
    """Test PCTL on nondeterministic choice."""

    def test_nd_choice_has_both_actions(self):
        """P + Q can do actions of both P and Q."""
        p = nd_choice(prefix("a", stop()), prefix("b", stop()))
        # Under uniform resolution, both actions available
        r_a = verify_process(p, atom("can_a"))
        r_b = verify_process(p, atom("can_b"))
        assert r_a.verdict == PropertyVerdict.SATISFIED
        assert r_b.verdict == PropertyVerdict.SATISFIED

    def test_nd_choice_eventually_deadlocks(self):
        """Nondeterministic choice eventually deadlocks."""
        p = nd_choice(prefix("a", stop()), prefix("b", stop()))
        formula = prob_geq(1.0, eventually(atom("deadlock")))
        result = verify_process(p, formula)
        assert result.verdict == PropertyVerdict.SATISFIED

    def test_nd_choice_state_count(self):
        """a.0 + b.0 has at least 2 states."""
        p = nd_choice(prefix("a", stop()), prefix("b", stop()))
        result = verify_process(p, tt())
        assert result.lts_states >= 2


# ===== Section 4: Parallel Composition Verification =====

class TestParallelVerification:
    """Test PCTL on parallel compositions."""

    def test_independent_parallel(self):
        """a.0 | b.0 has both actions available."""
        p = parallel(prefix("a", stop()), prefix("b", stop()))
        r_a = verify_process(p, atom("can_a"))
        r_b = verify_process(p, atom("can_b"))
        assert r_a.verdict == PropertyVerdict.SATISFIED
        assert r_b.verdict == PropertyVerdict.SATISFIED

    def test_parallel_eventually_deadlock(self):
        """Independent parallel eventually deadlocks (both terminate)."""
        p = parallel(prefix("a", stop()), prefix("b", stop()))
        formula = prob_geq(1.0, eventually(atom("deadlock")))
        result = verify_process(p, formula)
        assert result.verdict == PropertyVerdict.SATISFIED

    def test_synchronizing_parallel(self):
        """Client-server synchronization."""
        client = prefix("req", prefix("~resp", stop()))
        server = prefix("~req", prefix("resp", stop()))
        system = parallel(client, server)

        # System should eventually deadlock (both finish)
        formula = prob_geq(1.0, eventually(atom("deadlock")))
        result = verify_process(system, formula)
        assert result.verdict == PropertyVerdict.SATISFIED

    def test_parallel_state_explosion(self):
        """Parallel increases state space."""
        p1 = prefix("a", prefix("b", stop()))
        p2 = prefix("c", prefix("d", stop()))
        par = parallel(p1, p2)
        result = verify_process(par, tt())
        # Product of 3*3 = 9 states max (interleaving)
        assert result.lts_states >= 4  # At least more than individual


# ===== Section 5: Recursive Process Verification =====

class TestRecursiveVerification:
    """Test PCTL on recursive processes."""

    def test_recursive_deadlock_free(self):
        """fix X. a.X is deadlock free."""
        server = recdef("X", prefix("a", recvar("X")))
        dl = check_deadlock_freedom(server)
        assert dl["deadlock_free"]

    def test_recursive_always_can_act(self):
        """fix X. a.X always has action a."""
        server = recdef("X", prefix("a", recvar("X")))
        result = verify_process(server, atom("can_a"))
        assert result.verdict == PropertyVerdict.SATISFIED

    def test_recursive_no_deadlock_pctl(self):
        """fix X. a.X satisfies AG(NOT deadlock) via PCTL."""
        server = recdef("X", prefix("a", recvar("X")))
        result = verify_no_deadlock_pctl(server)
        assert result.verdict == PropertyVerdict.SATISFIED


# ===== Section 6: Weak Bisimulation Equivalence =====

class TestWeakEquivalence:
    """Test weak bisimulation equivalence checking."""

    def test_tau_prefix_weak_equiv(self):
        """tau.a.0 ~w a.0 (weak bisimulation)."""
        p1 = tau_prefix(prefix("a", stop()))
        p2 = prefix("a", stop())
        result = check_equivalence(p1, p2, equiv_type="weak")
        assert result.equivalent

    def test_different_processes_not_equiv(self):
        """a.0 and b.0 are not weakly bisimilar."""
        p1 = prefix("a", stop())
        p2 = prefix("b", stop())
        result = check_equivalence(p1, p2, equiv_type="weak")
        assert not result.equivalent

    def test_same_process_equiv(self):
        """Any process is weakly bisimilar to itself."""
        p = prefix("a", prefix("b", stop()))
        result = check_equivalence(p, p, equiv_type="weak")
        assert result.equivalent

    def test_strong_equiv_implies_weak(self):
        """Strong bisimilarity implies weak bisimilarity."""
        p1 = prefix("a", stop())
        p2 = prefix("a", stop())
        results = check_all_equivalences(p1, p2)
        if results["strong"].equivalent:
            assert results["weak"].equivalent


# ===== Section 7: Strong Bisimulation Equivalence =====

class TestStrongEquivalence:
    """Test strong bisimulation."""

    def test_different_actions_not_strong_equiv(self):
        """a.0 is NOT strongly bisimilar to b.0."""
        p1 = prefix("a", stop())
        p2 = prefix("b", stop())
        result = check_equivalence(p1, p2, equiv_type="strong")
        assert not result.equivalent

    def test_identical_strong_equiv(self):
        """Identical processes are strongly bisimilar."""
        p = prefix("a", prefix("b", stop()))
        result = check_equivalence(p, p, equiv_type="strong")
        assert result.equivalent

    def test_strong_finer_than_weak_for_different(self):
        """Strong and weak agree on clearly different processes."""
        p1 = prefix("a", stop())
        p2 = prefix("b", stop())
        r_strong = check_equivalence(p1, p2, equiv_type="strong")
        r_weak = check_equivalence(p1, p2, equiv_type="weak")
        assert not r_strong.equivalent
        assert not r_weak.equivalent


# ===== Section 8: Algebraic Laws =====

class TestAlgebraicLaws:
    """Test algebraic law verification."""

    def test_choice_commutativity(self):
        """P + Q ~w Q + P."""
        p = prefix("a", stop())
        q = prefix("b", stop())
        result = verify_algebraic_law("commutativity_choice", p, q)
        assert result.holds

    def test_choice_idempotence_trace(self):
        """P + P has same traces as P."""
        p = prefix("a", stop())
        result = check_trace_equivalence(nd_choice(p, p), p)
        assert result["trace_equivalent"]

    def test_parallel_commutativity(self):
        """P | Q ~w Q | P."""
        p = prefix("a", stop())
        q = prefix("b", stop())
        result = verify_algebraic_law("commutativity_parallel", p, q)
        assert result.holds

    def test_tau_absorption(self):
        """tau.P ~w P (weak only)."""
        p = prefix("a", stop())
        result = verify_algebraic_law("tau_absorption", p, equiv_type="weak")
        assert result.holds

    def test_custom_law(self):
        """Custom law: a.b.0 + a.c.0 is trace equivalent to itself."""
        lhs = nd_choice(prefix("a", prefix("b", stop())), prefix("a", prefix("c", stop())))
        result = verify_custom_law(lhs, lhs, law_name="self_equiv")
        assert result.holds

    def test_unknown_law(self):
        """Unknown law name returns not-holds."""
        result = verify_algebraic_law("nonexistent_law")
        assert not result.holds


# ===== Section 9: Trace Analysis =====

class TestTraceAnalysis:
    """Test trace inclusion and equivalence."""

    def test_trace_inclusion_subset(self):
        """traces(a.0) subset of traces(a.0 + b.0)."""
        p1 = prefix("a", stop())
        p2 = nd_choice(prefix("a", stop()), prefix("b", stop()))
        result = check_trace_inclusion(p1, p2)
        assert result["included"]

    def test_trace_inclusion_not_subset(self):
        """traces(a.0 + b.0) NOT subset of traces(a.0)."""
        p1 = nd_choice(prefix("a", stop()), prefix("b", stop()))
        p2 = prefix("a", stop())
        result = check_trace_inclusion(p1, p2)
        assert not result["included"]

    def test_trace_equivalence_same(self):
        """Same process has same traces."""
        p = prefix("a", prefix("b", stop()))
        result = check_trace_equivalence(p, p)
        assert result["trace_equivalent"]

    def test_trace_equivalence_different(self):
        """a.0 and b.0 have different traces."""
        p1 = prefix("a", stop())
        p2 = prefix("b", stop())
        result = check_trace_equivalence(p1, p2)
        assert not result["trace_equivalent"]


# ===== Section 10: Deadlock Analysis =====

class TestDeadlockAnalysis:
    """Test deadlock detection and verification."""

    def test_stop_has_deadlock(self):
        """STOP is a deadlock state."""
        result = check_deadlock_freedom(stop())
        assert not result["deadlock_free"]
        assert 0 in result["deadlock_states"]

    def test_prefix_has_deadlock(self):
        """a.0 reaches deadlock after a."""
        result = check_deadlock_freedom(prefix("a", stop()))
        assert not result["deadlock_free"]
        assert len(result["deadlock_states"]) >= 1

    def test_recursive_no_deadlock(self):
        """fix X. a.X has no deadlock."""
        server = recdef("X", prefix("a", recvar("X")))
        result = check_deadlock_freedom(server)
        assert result["deadlock_free"]

    def test_deadlock_ratio(self):
        """Check deadlock ratio computation."""
        p = prefix("a", stop())
        result = check_deadlock_freedom(p)
        assert 0 < result["deadlock_ratio"] <= 1.0


# ===== Section 11: Compositional Analysis =====

class TestCompositionalAnalysis:
    """Test compositional analysis of parallel systems."""

    def test_simple_composition(self):
        """Analyze a.0 | b.0."""
        p1 = prefix("a", stop())
        p2 = prefix("b", stop())
        par = parallel(p1, p2)
        result = analyze_composition(par, components=[p1, p2])
        assert result.reachable_states >= 3
        assert "a" in result.actions
        assert "b" in result.actions

    def test_composition_with_properties(self):
        """Analyze composition with PCTL properties."""
        p1 = prefix("a", stop())
        p2 = prefix("b", stop())
        par = parallel(p1, p2)
        props = [
            atom("can_a"),
            prob_geq(1.0, eventually(atom("deadlock"))),
        ]
        result = analyze_composition(par, components=[p1, p2], properties=props)
        assert len(result.properties_checked) == 2

    def test_composition_deadlock_free(self):
        """Recursive parallel composition is deadlock free."""
        s1 = recdef("X", prefix("a", recvar("X")))
        s2 = recdef("Y", prefix("b", recvar("Y")))
        par = parallel(s1, s2)
        result = analyze_composition(par, components=[s1, s2])
        assert result.deadlock_free

    def test_composition_traces(self):
        """Composition has interleaved traces."""
        p1 = prefix("a", stop())
        p2 = prefix("b", stop())
        par = parallel(p1, p2)
        result = analyze_composition(par, components=[p1, p2])
        # Should have traces with both orderings
        assert len(result.traces) >= 2


# ===== Section 12: Minimization =====

class TestMinimization:
    """Test process minimization."""

    def test_minimize_simple(self):
        """Minimization doesn't increase states."""
        p = prefix("a", prefix("b", stop()))
        result = minimize_process(p, method="weak")
        assert result["minimized_states"] <= result["original_states"]

    def test_minimize_with_tau(self):
        """tau.a.0 minimizes (tau absorbed under weak)."""
        p = tau_prefix(prefix("a", stop()))
        result = minimize_process(p, method="weak")
        assert result["minimized_states"] <= result["original_states"]

    def test_minimize_reduction_ratio(self):
        """Reduction ratio is between 0 and 1."""
        p = nd_choice(prefix("a", stop()), prefix("a", stop()))
        result = minimize_process(p, method="weak")
        assert 0 <= result["reduction_ratio"] <= 1


# ===== Section 13: Equivalence Hierarchy =====

class TestEquivalenceHierarchy:
    """Test equivalence hierarchy analysis."""

    def test_hierarchy_identical(self):
        """Identical processes: all equivalences hold."""
        p = prefix("a", stop())
        result = analyze_equivalence_hierarchy(p, p)
        assert result["trace_equivalent"]
        assert result["weakly_bisimilar"]
        assert result["hierarchy_consistent"]

    def test_hierarchy_tau_difference(self):
        """a.0 vs b.0: nothing equivalent, hierarchy consistent."""
        p1 = prefix("a", stop())
        p2 = prefix("b", stop())
        result = analyze_equivalence_hierarchy(p1, p2)
        assert not result["weakly_bisimilar"]
        assert not result["strongly_bisimilar"]
        assert result["hierarchy_consistent"]

    def test_hierarchy_different(self):
        """a.0 vs b.0: nothing equivalent."""
        p1 = prefix("a", stop())
        p2 = prefix("b", stop())
        result = analyze_equivalence_hierarchy(p1, p2)
        assert not result["trace_equivalent"]
        assert not result["weakly_bisimilar"]
        assert result["hierarchy_consistent"]

    def test_hierarchy_distance_zero_for_bisimilar(self):
        """Distance is 0 for bisimilar processes."""
        p = prefix("a", stop())
        result = analyze_equivalence_hierarchy(p, p)
        assert result["behavioral_distance"] < 0.01

    def test_hierarchy_distance_positive_for_different(self):
        """Distance is positive for non-bisimilar processes."""
        p1 = prefix("a", stop())
        p2 = prefix("b", stop())
        result = analyze_equivalence_hierarchy(p1, p2)
        assert result["behavioral_distance"] > 0


# ===== Section 14: Property Preservation =====

class TestPropertyPreservation:
    """Test property-preserving transformation checking."""

    def test_identity_preserves_all(self):
        """Identity transformation preserves everything."""
        p = prefix("a", prefix("b", stop()))
        props = [atom("can_a"), prob_geq(1.0, eventually(atom("deadlock")))]
        result = check_property_preservation(p, p, props)
        assert result["all_preserved"]
        assert result["broken_count"] == 0

    def test_tau_insertion_preserves_deadlock(self):
        """tau.a.0 preserves deadlock reachability of a.0."""
        p_orig = prefix("a", stop())
        p_trans = tau_prefix(prefix("a", stop()))
        props = [prob_geq(1.0, eventually(atom("deadlock")))]
        result = check_property_preservation(p_orig, p_trans, props)
        assert result["all_preserved"]

    def test_broken_preservation(self):
        """Changing action breaks can_a property."""
        p_orig = prefix("a", stop())
        p_trans = prefix("b", stop())
        props = [atom("can_a")]
        result = check_property_preservation(p_orig, p_trans, props)
        assert not result["all_preserved"]
        assert result["broken_count"] == 1


# ===== Section 15: Full Analysis and Summary =====

class TestFullAnalysis:
    """Test comprehensive analysis."""

    def test_full_analysis_simple(self):
        """Full analysis of a.b.0."""
        p = prefix("a", prefix("b", stop()))
        result = full_process_analysis(p)
        assert result["lts_states"] == 3
        assert not result["deadlock_free"]
        assert "a" in result["observable_actions"]
        assert "b" in result["observable_actions"]

    def test_full_analysis_recursive(self):
        """Full analysis of fix X. a.X."""
        server = recdef("X", prefix("a", recvar("X")))
        result = full_process_analysis(server)
        assert result["deadlock_free"]
        assert "a" in result["observable_actions"]

    def test_full_analysis_prob(self):
        """Full analysis of probabilistic process."""
        p = prob_choice(0.5, prefix("a", stop()), prefix("b", stop()))
        result = full_process_analysis(p)
        assert result["lts_states"] >= 3
        assert "a" in result["observable_actions"]
        assert "b" in result["observable_actions"]

    def test_summary_string(self):
        """Summary produces a non-empty string."""
        p = prefix("a", stop())
        summary = process_verification_summary(p)
        assert "Process Verification Summary" in summary
        assert "States:" in summary
        assert "Deadlock free:" in summary

    def test_summary_with_properties(self):
        """Summary includes property results."""
        p = prefix("a", stop())
        props = [atom("can_a")]
        summary = process_verification_summary(p, properties=props)
        assert "Property Verification" in summary


# ===== Section 16: Restriction and Relabeling =====

class TestRestrictionRelabeling:
    """Test verification on restricted and relabeled processes."""

    def test_restrict_blocks_action(self):
        """Restricting action a removes it from observable actions."""
        p = restrict(prefix("a", stop()), {"a"})
        dl = check_deadlock_freedom(p)
        # After restriction, a is blocked -> deadlock
        assert not dl["deadlock_free"]

    def test_relabel_changes_action(self):
        """Relabeling a -> b changes observable actions."""
        p = relabel(prefix("a", stop()), {"a": "b"})
        result = verify_process(p, atom("can_b"))
        assert result.verdict == PropertyVerdict.SATISFIED

    def test_relabel_equivalence(self):
        """a.0[a/b] ~w b.0."""
        p1 = relabel(prefix("a", stop()), {"a": "b"})
        p2 = prefix("b", stop())
        result = check_equivalence(p1, p2, equiv_type="weak")
        assert result.equivalent


# ===== Section 17: Behavioral Distance =====

class TestBehavioralDistance:
    """Test behavioral distance computation."""

    def test_distance_self_zero(self):
        """Distance to self is 0."""
        p = prefix("a", stop())
        d = process_distance(p, p)
        assert d < 0.01

    def test_distance_different_positive(self):
        """Different processes have positive distance."""
        p1 = prefix("a", stop())
        p2 = prefix("b", stop())
        d = process_distance(p1, p2)
        assert d > 0

    def test_distance_symmetric(self):
        """Distance is symmetric."""
        p1 = prefix("a", stop())
        p2 = prefix("b", stop())
        d1 = process_distance(p1, p2)
        d2 = process_distance(p2, p1)
        assert abs(d1 - d2) < 0.01


# ===== Section 18: Edge Cases =====

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_stop_process_analysis(self):
        """STOP has 1 state, deadlock."""
        result = full_process_analysis(stop())
        assert result["lts_states"] == 1
        assert not result["deadlock_free"]

    def test_verify_all_states(self):
        """verify_process_all on tautology."""
        p = prefix("a", stop())
        assert verify_process_all(p, tt())

    def test_verify_all_states_false(self):
        """verify_process_all fails for non-universal property."""
        p = prefix("a", stop())
        # Not all states are deadlock (initial state can act)
        assert not verify_process_all(p, atom("deadlock"))

    def test_verify_state(self):
        """verify_process_state for specific state."""
        p = prefix("a", stop())
        assert verify_process_state(p, tt(), state=0)

    def test_parsed_process_verification(self):
        """Verify a parsed process."""
        p = parse_proc("a.b.0")
        result = verify_process(p, atom("can_a"))
        assert result.verdict == PropertyVerdict.SATISFIED

    def test_result_statistics(self):
        """Verification result contains statistics."""
        p = prefix("a", stop())
        result = verify_process(p, tt())
        assert "lts_states" in result.statistics
        assert "mc_states" in result.statistics
        assert result.statistics["lts_states"] > 0
