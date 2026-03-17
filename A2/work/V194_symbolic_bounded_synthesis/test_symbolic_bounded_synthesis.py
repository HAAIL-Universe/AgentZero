"""
Tests for V194: Symbolic Bounded Synthesis
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V190_bounded_synthesis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V023_ltl_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V021_bdd_model_checking'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V186_reactive_synthesis'))

from ltl_model_checker import (
    Atom, Not, And, Or, Implies,
    Next, Finally, Globally, Until, Release,
    LTLTrue, LTLFalse
)
from bounded_synthesis import (
    SynthVerdict, Controller, ucw_from_ltl,
    verify_annotation, verify_controller, controller_statistics
)

from symbolic_bounded_synthesis import (
    symbolic_bounded_synthesize,
    symbolic_fixpoint_synthesize,
    synthesize_safety,
    synthesize_liveness,
    synthesize_response,
    synthesize_assume_guarantee,
    synthesize_stability,
    find_minimum_controller,
    compare_with_smt,
    compare_with_game,
    synthesis_statistics,
    verify_synthesis,
    symbolic_synthesis_summary,
    _bits_needed,
    _encode_int,
    _encode_range,
    _make_layout,
    _build_ucw_trans_bdd,
    _check_trivial,
    _collect_atoms,
    _solve_annotation,
    _has_strict_cycle,
    _check_controller_bdd_fast,
    BDD,
)


# ===================================================================
# Utility Tests
# ===================================================================

class TestBitsNeeded:
    def test_zero(self):
        assert _bits_needed(0) == 1

    def test_one(self):
        assert _bits_needed(1) == 1

    def test_two(self):
        assert _bits_needed(2) == 1

    def test_three(self):
        assert _bits_needed(3) == 2

    def test_four(self):
        assert _bits_needed(4) == 2

    def test_five(self):
        assert _bits_needed(5) == 3

    def test_eight(self):
        assert _bits_needed(8) == 3

    def test_nine(self):
        assert _bits_needed(9) == 4

    def test_sixteen(self):
        assert _bits_needed(16) == 4

    def test_seventeen(self):
        assert _bits_needed(17) == 5


class TestEncodeInt:
    def test_zero(self):
        bdd = BDD()
        bdd.named_var('b0')
        bdd.named_var('b1')
        result = _encode_int(bdd, [0, 1], 0)
        # Should encode ~b0 & ~b1
        assert result is not None
        assert bdd.sat_count(result, 2) == 1

    def test_one(self):
        bdd = BDD()
        bdd.named_var('b0')
        bdd.named_var('b1')
        result = _encode_int(bdd, [0, 1], 1)
        assert bdd.sat_count(result, 2) == 1

    def test_two(self):
        bdd = BDD()
        bdd.named_var('b0')
        bdd.named_var('b1')
        result = _encode_int(bdd, [0, 1], 2)
        assert bdd.sat_count(result, 2) == 1

    def test_three(self):
        bdd = BDD()
        bdd.named_var('b0')
        bdd.named_var('b1')
        result = _encode_int(bdd, [0, 1], 3)
        assert bdd.sat_count(result, 2) == 1

    def test_different_values_distinct(self):
        bdd = BDD()
        bdd.named_var('b0')
        bdd.named_var('b1')
        results = [_encode_int(bdd, [0, 1], v) for v in range(4)]
        # All should be distinct BDDs
        ids = [r._id for r in results]
        assert len(set(ids)) == 4


class TestEncodeRange:
    def test_full_range(self):
        bdd = BDD()
        bdd.named_var('b0')
        bdd.named_var('b1')
        result = _encode_range(bdd, [0, 1], 4)
        assert result._id == bdd.TRUE._id

    def test_partial_range(self):
        bdd = BDD()
        bdd.named_var('b0')
        bdd.named_var('b1')
        result = _encode_range(bdd, [0, 1], 3)
        assert bdd.sat_count(result, 2) == 3

    def test_single_value(self):
        bdd = BDD()
        bdd.named_var('b0')
        bdd.named_var('b1')
        result = _encode_range(bdd, [0, 1], 1)
        assert bdd.sat_count(result, 2) == 1


class TestCollectAtoms:
    def test_atom(self):
        assert _collect_atoms(Atom('a')) == {'a'}

    def test_not(self):
        assert _collect_atoms(Not(Atom('a'))) == {'a'}

    def test_and(self):
        assert _collect_atoms(And(Atom('a'), Atom('b'))) == {'a', 'b'}

    def test_globally(self):
        assert _collect_atoms(Globally(Atom('x'))) == {'x'}

    def test_complex(self):
        spec = Globally(Implies(Atom('r'), Finally(Atom('g'))))
        assert _collect_atoms(spec) == {'r', 'g'}

    def test_true(self):
        assert _collect_atoms(LTLTrue()) == set()


# ===================================================================
# Layout Tests
# ===================================================================

class TestLayout:
    def test_simple_layout(self):
        bdd = BDD()
        ucw = ucw_from_ltl(Globally(Atom('g')))
        layout = _make_layout(bdd, ucw, {'r'}, {'g'}, 2)
        assert len(layout.env_bits) == 1
        assert len(layout.sys_bits) == 1
        assert layout.env_names == ['r']
        assert layout.sys_names == ['g']
        assert layout.n_ctrl_states == 2

    def test_empty_env(self):
        bdd = BDD()
        ucw = ucw_from_ltl(Globally(Atom('g')))
        layout = _make_layout(bdd, ucw, set(), {'g'}, 1)
        assert len(layout.env_bits) == 0
        assert len(layout.sys_bits) == 1

    def test_multiple_vars(self):
        bdd = BDD()
        ucw = ucw_from_ltl(Globally(Atom('a')))
        layout = _make_layout(bdd, ucw, {'r1', 'r2'}, {'g1', 'g2'}, 3)
        assert len(layout.env_bits) == 2
        assert len(layout.sys_bits) == 2
        assert layout.n_ctrl_states == 3


# ===================================================================
# Annotation Solver Tests
# ===================================================================

class TestAnnotationSolver:
    def test_no_constraints(self):
        reachable = {(0, 0)}
        constraints = []
        ann = _solve_annotation(reachable, constraints, 5)
        assert ann is not None

    def test_weak_constraint(self):
        reachable = {(0, 0), (1, 0)}
        constraints = [((0, 0), (1, 0), False)]
        ann = _solve_annotation(reachable, constraints, 5)
        assert ann is not None
        assert ann.values[(1, 0)] <= ann.values[(0, 0)]

    def test_strict_constraint(self):
        reachable = {(0, 0), (1, 0)}
        constraints = [((0, 0), (1, 0), True)]
        ann = _solve_annotation(reachable, constraints, 5)
        assert ann is not None
        assert ann.values[(1, 0)] < ann.values[(0, 0)]

    def test_strict_self_loop_fails(self):
        reachable = {(0, 0)}
        constraints = [((0, 0), (0, 0), True)]
        ann = _solve_annotation(reachable, constraints, 5)
        assert ann is None

    def test_strict_cycle_fails(self):
        reachable = {(0, 0), (1, 0)}
        constraints = [
            ((0, 0), (1, 0), True),
            ((1, 0), (0, 0), False),  # weak back
        ]
        # Strict 0->1, weak 1->0: lambda(1) < lambda(0), lambda(0) <= lambda(1)
        # Impossible
        ann = _solve_annotation(reachable, constraints, 5)
        assert ann is None

    def test_chain_of_strict(self):
        reachable = {(0, 0), (1, 0), (2, 0)}
        constraints = [
            ((0, 0), (1, 0), True),
            ((1, 0), (2, 0), True),
        ]
        ann = _solve_annotation(reachable, constraints, 5)
        assert ann is not None
        assert ann.values[(2, 0)] < ann.values[(1, 0)] < ann.values[(0, 0)]

    def test_bound_too_small(self):
        reachable = {(0, 0), (1, 0), (2, 0)}
        constraints = [
            ((0, 0), (1, 0), True),
            ((1, 0), (2, 0), True),
        ]
        ann = _solve_annotation(reachable, constraints, 1)
        # Need at least bound 2 for chain of 2 strict edges
        assert ann is None


class TestStrictCycleDetection:
    def test_no_cycle(self):
        reachable = {(0, 0), (1, 0)}
        constraints = [((0, 0), (1, 0), True)]
        assert not _has_strict_cycle(reachable, constraints)

    def test_strict_self_loop(self):
        reachable = {(0, 0)}
        constraints = [((0, 0), (0, 0), True)]
        assert _has_strict_cycle(reachable, constraints)

    def test_strict_cycle(self):
        reachable = {(0, 0), (1, 0)}
        constraints = [
            ((0, 0), (1, 0), True),
            ((1, 0), (0, 0), True),
        ]
        assert _has_strict_cycle(reachable, constraints)

    def test_weak_cycle_ok(self):
        reachable = {(0, 0), (1, 0)}
        constraints = [
            ((0, 0), (1, 0), False),
            ((1, 0), (0, 0), False),
        ]
        assert not _has_strict_cycle(reachable, constraints)

    def test_mixed_cycle_with_strict(self):
        reachable = {(0, 0), (1, 0), (2, 0)}
        constraints = [
            ((0, 0), (1, 0), False),
            ((1, 0), (2, 0), True),
            ((2, 0), (0, 0), False),
        ]
        assert _has_strict_cycle(reachable, constraints)


# ===================================================================
# Trivial Spec Tests
# ===================================================================

class TestTrivialSpecs:
    def test_true_realizable(self):
        result = symbolic_bounded_synthesize(LTLTrue(), set(), set())
        assert result.verdict == SynthVerdict.REALIZABLE

    def test_false_unrealizable(self):
        result = symbolic_bounded_synthesize(LTLFalse(), set(), set())
        assert result.verdict == SynthVerdict.UNREALIZABLE

    def test_true_with_env(self):
        result = symbolic_bounded_synthesize(LTLTrue(), {'r'}, {'g'})
        assert result.verdict == SynthVerdict.REALIZABLE
        assert result.controller is not None

    def test_false_with_vars(self):
        result = symbolic_bounded_synthesize(LTLFalse(), {'r'}, {'g'})
        assert result.verdict == SynthVerdict.UNREALIZABLE


# ===================================================================
# Safety Synthesis Tests
# ===================================================================

class TestSafetySynthesis:
    def test_always_output(self):
        """G(g) -- system must always output g."""
        spec = Globally(Atom('g'))
        result = symbolic_bounded_synthesize(spec, set(), {'g'}, max_states=2)
        assert result.verdict == SynthVerdict.REALIZABLE
        assert result.controller is not None
        # Verify controller always outputs g
        ctrl = result.controller
        trace = ctrl.simulate([frozenset()] * 5)
        for state, inp, out, ns in trace:
            assert 'g' in out

    def test_always_output_with_env(self):
        """G(g) with env variable r."""
        spec = Globally(Atom('g'))
        result = symbolic_bounded_synthesize(spec, {'r'}, {'g'}, max_states=2)
        assert result.verdict == SynthVerdict.REALIZABLE
        # Must output g regardless of env
        ctrl = result.controller
        for inp in [frozenset(), frozenset({'r'})]:
            trace = ctrl.simulate([inp] * 3)
            for state, i, out, ns in trace:
                assert 'g' in out

    def test_never_bad(self):
        """G(!bad) -- safety spec via convenience function."""
        result = synthesize_safety(Atom('bad'), set(), {'bad'}, max_states=2)
        assert result.verdict == SynthVerdict.REALIZABLE
        ctrl = result.controller
        trace = ctrl.simulate([frozenset()] * 5)
        for state, inp, out, ns in trace:
            assert 'bad' not in out

    def test_mutual_exclusion(self):
        """G(!(g1 & g2)) -- never both outputs simultaneously."""
        bad = And(Atom('g1'), Atom('g2'))
        spec = Globally(Not(bad))
        result = symbolic_bounded_synthesize(spec, set(), {'g1', 'g2'}, max_states=2)
        assert result.verdict == SynthVerdict.REALIZABLE
        ctrl = result.controller
        trace = ctrl.simulate([frozenset()] * 5)
        for state, inp, out, ns in trace:
            assert not ('g1' in out and 'g2' in out)


# ===================================================================
# Liveness Synthesis Tests
# ===================================================================

class TestLivenessSynthesis:
    def test_always_eventually_output(self):
        """G(F(g)) -- must output g infinitely often."""
        spec = Globally(Finally(Atom('g')))
        result = symbolic_bounded_synthesize(spec, set(), {'g'}, max_states=3)
        assert result.verdict == SynthVerdict.REALIZABLE

    def test_liveness_convenience(self):
        """Convenience synthesize_liveness function."""
        result = synthesize_liveness(Atom('g'), set(), {'g'}, max_states=3)
        assert result.verdict == SynthVerdict.REALIZABLE


# ===================================================================
# Response Synthesis Tests
# ===================================================================

class TestResponseSynthesis:
    def test_request_response(self):
        """G(r -> F(g)) -- every request eventually gets a grant."""
        spec = Globally(Implies(Atom('r'), Finally(Atom('g'))))
        result = symbolic_bounded_synthesize(spec, {'r'}, {'g'}, max_states=3)
        assert result.verdict == SynthVerdict.REALIZABLE

    def test_response_convenience(self):
        """Convenience synthesize_response function."""
        result = synthesize_response(Atom('r'), Atom('g'), {'r'}, {'g'}, max_states=3)
        assert result.verdict == SynthVerdict.REALIZABLE


# ===================================================================
# Controller Verification Tests
# ===================================================================

class TestControllerVerification:
    def test_verify_safety_controller(self):
        spec = Globally(Atom('g'))
        result = symbolic_bounded_synthesize(spec, set(), {'g'}, max_states=2)
        assert result.verdict == SynthVerdict.REALIZABLE

        verification = verify_synthesis(result, spec, set(), {'g'})
        assert verification['valid']

    def test_verify_annotation(self):
        spec = Globally(Atom('g'))
        result = symbolic_bounded_synthesize(spec, set(), {'g'}, max_states=2)
        assert result.verdict == SynthVerdict.REALIZABLE

        if result.annotation:
            ucw = ucw_from_ltl(spec)
            valid, violations = verify_annotation(result.controller, ucw, result.annotation)
            assert valid

    def test_controller_simulation(self):
        spec = Globally(Atom('g'))
        result = symbolic_bounded_synthesize(spec, set(), {'g'}, max_states=2)
        ctrl = result.controller
        trace = ctrl.simulate([frozenset()] * 10)
        assert len(trace) == 10
        for state, inp, out, ns in trace:
            assert 'g' in out


# ===================================================================
# Symbolic Fixpoint Tests
# ===================================================================

class TestSymbolicFixpoint:
    def test_always_output(self):
        spec = Globally(Atom('g'))
        result = symbolic_fixpoint_synthesize(spec, set(), {'g'}, max_states=2)
        assert result.verdict == SynthVerdict.REALIZABLE
        assert result.method == "symbolic_fixpoint"

    def test_safety(self):
        spec = Globally(Not(Atom('bad')))
        result = symbolic_fixpoint_synthesize(spec, set(), {'bad'}, max_states=2)
        assert result.verdict == SynthVerdict.REALIZABLE

    def test_trivial_true(self):
        result = symbolic_fixpoint_synthesize(LTLTrue(), set(), set())
        assert result.verdict == SynthVerdict.REALIZABLE

    def test_trivial_false(self):
        result = symbolic_fixpoint_synthesize(LTLFalse(), set(), set())
        assert result.verdict == SynthVerdict.UNREALIZABLE


# ===================================================================
# UCW and BDD Construction Tests
# ===================================================================

class TestUCWConstruction:
    def test_ucw_from_safety(self):
        spec = Globally(Atom('g'))
        ucw = ucw_from_ltl(spec)
        assert len(ucw.states) > 0
        assert len(ucw.initial) > 0

    def test_ucw_from_liveness(self):
        spec = Globally(Finally(Atom('g')))
        ucw = ucw_from_ltl(spec)
        assert len(ucw.states) > 0

    def test_ucw_transitions_exist(self):
        spec = Globally(Atom('g'))
        ucw = ucw_from_ltl(spec)
        total_trans = sum(len(v) for v in ucw.transitions.values())
        assert total_trans > 0


class TestBDDConstruction:
    def test_ucw_trans_bdd(self):
        bdd = BDD()
        ucw = ucw_from_ltl(Globally(Atom('g')))
        layout = _make_layout(bdd, ucw, set(), {'g'}, 1)
        trans_bdd, ucw_next_bits, state_list, state_to_idx = \
            _build_ucw_trans_bdd(bdd, layout, ucw)
        assert trans_bdd._id != bdd.FALSE._id

    def test_layout_variables(self):
        bdd = BDD()
        ucw = ucw_from_ltl(Globally(Atom('g')))
        layout = _make_layout(bdd, ucw, {'r'}, {'g'}, 2)
        assert layout.total_vars > 0
        assert len(layout.ucw_bits) > 0
        assert len(layout.ctrl_bits) > 0


# ===================================================================
# Assume-Guarantee Synthesis Tests
# ===================================================================

class TestAssumeGuarantee:
    def test_simple_ag(self):
        """If env always provides r, system always provides g."""
        assumptions = Globally(Atom('r'))
        guarantees = Globally(Atom('g'))
        result = synthesize_assume_guarantee(assumptions, guarantees,
                                              {'r'}, {'g'}, max_states=3)
        assert result.verdict == SynthVerdict.REALIZABLE

    def test_ag_response(self):
        """Under fairness, respond to requests."""
        assumptions = Globally(Finally(Atom('r')))
        guarantees = Globally(Implies(Atom('r'), Finally(Atom('g'))))
        result = synthesize_assume_guarantee(assumptions, guarantees,
                                              {'r'}, {'g'}, max_states=3)
        assert result.verdict == SynthVerdict.REALIZABLE


# ===================================================================
# Stability Synthesis Tests
# ===================================================================

class TestStabilitySynthesis:
    def test_eventually_always(self):
        """F(G(g)) -- eventually stabilize to always outputting g."""
        result = synthesize_stability(Atom('g'), set(), {'g'}, max_states=2)
        assert result.verdict == SynthVerdict.REALIZABLE


# ===================================================================
# Minimum Controller Tests
# ===================================================================

class TestMinimumController:
    def test_find_minimum_safety(self):
        spec = Globally(Atom('g'))
        result = find_minimum_controller(spec, set(), {'g'}, max_states=4)
        assert result.verdict == SynthVerdict.REALIZABLE
        assert result.n_states == 1  # Memoryless suffices

    def test_find_minimum_never_bad(self):
        spec = Globally(Not(Atom('b')))
        result = find_minimum_controller(spec, set(), {'b'}, max_states=4)
        assert result.verdict == SynthVerdict.REALIZABLE
        assert result.n_states == 1


# ===================================================================
# Statistics and Summary Tests
# ===================================================================

class TestStatistics:
    def test_synthesis_statistics(self):
        spec = Globally(Atom('g'))
        result = symbolic_bounded_synthesize(spec, set(), {'g'}, max_states=2)
        stats = synthesis_statistics(result)
        assert 'verdict' in stats
        assert 'method' in stats
        assert 'controller_states' in stats
        assert stats['verdict'] == 'realizable'

    def test_summary(self):
        spec = Globally(Atom('g'))
        result = symbolic_bounded_synthesize(spec, set(), {'g'}, max_states=2)
        summary = symbolic_synthesis_summary(result)
        assert 'Symbolic Bounded Synthesis' in summary
        assert 'realizable' in summary.lower()

    def test_statistics_unrealizable(self):
        result = symbolic_bounded_synthesize(LTLFalse(), set(), set())
        stats = synthesis_statistics(result)
        assert stats['verdict'] == 'unrealizable'


# ===================================================================
# Comparison Tests
# ===================================================================

class TestComparison:
    def test_compare_with_smt_safety(self):
        spec = Globally(Atom('g'))
        comp = compare_with_smt(spec, set(), {'g'}, max_states=2)
        assert comp['agreement']
        assert comp['bdd_result'].verdict == SynthVerdict.REALIZABLE
        assert comp['smt_result'].verdict == SynthVerdict.REALIZABLE

    def test_compare_with_smt_unrealizable(self):
        comp = compare_with_smt(LTLFalse(), set(), set())
        assert comp['agreement']

    def test_compare_with_game_safety(self):
        spec = Globally(Atom('g'))
        comp = compare_with_game(spec, set(), {'g'}, max_states=2)
        assert comp['agreement']

    def test_compare_timing_included(self):
        spec = Globally(Atom('g'))
        comp = compare_with_smt(spec, set(), {'g'}, max_states=2)
        assert 'bdd_time' in comp
        assert 'smt_time' in comp
        assert comp['bdd_time'] >= 0
        assert comp['smt_time'] >= 0


# ===================================================================
# Edge Cases
# ===================================================================

class TestEdgeCases:
    def test_empty_env_and_sys(self):
        result = symbolic_bounded_synthesize(LTLTrue(), set(), set())
        assert result.verdict == SynthVerdict.REALIZABLE

    def test_multiple_env_vars(self):
        spec = Globally(Atom('g'))
        result = symbolic_bounded_synthesize(spec, {'r1', 'r2'}, {'g'}, max_states=2)
        assert result.verdict == SynthVerdict.REALIZABLE

    def test_multiple_sys_vars(self):
        spec = Globally(And(Atom('g1'), Atom('g2')))
        result = symbolic_bounded_synthesize(spec, set(), {'g1', 'g2'}, max_states=2)
        assert result.verdict == SynthVerdict.REALIZABLE

    def test_next_operator(self):
        """X(g) -- output g at next step."""
        spec = Next(Atom('g'))
        result = symbolic_bounded_synthesize(spec, set(), {'g'}, max_states=2)
        # Should be realizable (just output g at step 1)
        assert result.verdict == SynthVerdict.REALIZABLE

    def test_controller_step(self):
        spec = Globally(Atom('g'))
        result = symbolic_bounded_synthesize(spec, {'r'}, {'g'}, max_states=2)
        ctrl = result.controller
        state, output = ctrl.step(0, frozenset())
        assert 'g' in output
        state, output = ctrl.step(0, frozenset({'r'}))
        assert 'g' in output


# ===================================================================
# Method Attribute Tests
# ===================================================================

class TestMethodAttribute:
    def test_bounded_method(self):
        spec = Globally(Atom('g'))
        result = symbolic_bounded_synthesize(spec, set(), {'g'}, max_states=2)
        assert result.method == "symbolic_bounded"

    def test_fixpoint_method(self):
        spec = Globally(Atom('g'))
        result = symbolic_fixpoint_synthesize(spec, set(), {'g'}, max_states=2)
        assert result.method == "symbolic_fixpoint"


# ===================================================================
# Complex Specs
# ===================================================================

class TestComplexSpecs:
    def test_or_spec(self):
        """G(g1 | g2) -- always output at least one."""
        spec = Globally(Or(Atom('g1'), Atom('g2')))
        result = symbolic_bounded_synthesize(spec, set(), {'g1', 'g2'}, max_states=2)
        assert result.verdict == SynthVerdict.REALIZABLE

    def test_implies_spec(self):
        """G(r -> g) -- respond immediately."""
        spec = Globally(Implies(Atom('r'), Atom('g')))
        result = symbolic_bounded_synthesize(spec, {'r'}, {'g'}, max_states=2)
        assert result.verdict == SynthVerdict.REALIZABLE
        ctrl = result.controller
        # When r is present, g must be present
        state, output = ctrl.step(0, frozenset({'r'}))
        assert 'g' in output

    def test_until_spec(self):
        """g U done -- keep outputting g until done."""
        spec = Until(Atom('g'), Atom('done'))
        result = symbolic_bounded_synthesize(spec, set(), {'g', 'done'}, max_states=2)
        assert result.verdict == SynthVerdict.REALIZABLE

    def test_release_spec(self):
        """g R done -- done must hold until g occurs (and at that point too)."""
        spec = Release(Atom('g'), Atom('done'))
        result = symbolic_bounded_synthesize(spec, set(), {'g', 'done'}, max_states=2)
        assert result.verdict == SynthVerdict.REALIZABLE


# ===================================================================
# Annotation Verification Tests
# ===================================================================

class TestAnnotationVerificationIntegration:
    def test_safety_annotation_valid(self):
        spec = Globally(Atom('g'))
        result = symbolic_bounded_synthesize(spec, set(), {'g'}, max_states=2)
        assert result.verdict == SynthVerdict.REALIZABLE
        if result.annotation:
            ucw = ucw_from_ltl(spec)
            valid, _ = verify_annotation(result.controller, ucw, result.annotation)
            assert valid

    def test_liveness_annotation_valid(self):
        spec = Globally(Finally(Atom('g')))
        result = symbolic_bounded_synthesize(spec, set(), {'g'}, max_states=3)
        if result.verdict == SynthVerdict.REALIZABLE and result.annotation:
            ucw = ucw_from_ltl(spec)
            valid, _ = verify_annotation(result.controller, ucw, result.annotation)
            assert valid


# ===================================================================
# Verify Synthesis Tests
# ===================================================================

class TestVerifySynthesis:
    def test_verify_realizable(self):
        spec = Globally(Atom('g'))
        result = symbolic_bounded_synthesize(spec, set(), {'g'}, max_states=2)
        v = verify_synthesis(result, spec, set(), {'g'})
        assert v['valid']

    def test_verify_unrealizable(self):
        result = symbolic_bounded_synthesize(LTLFalse(), set(), set())
        v = verify_synthesis(result, LTLFalse(), set(), set())
        assert v['valid']  # No controller, but that's correct for unrealizable

    def test_verify_with_env(self):
        spec = Globally(Implies(Atom('r'), Atom('g')))
        result = symbolic_bounded_synthesize(spec, {'r'}, {'g'}, max_states=2)
        v = verify_synthesis(result, spec, {'r'}, {'g'})
        assert v['valid']


# ===================================================================
# Controller Properties Tests
# ===================================================================

class TestControllerProperties:
    def test_deterministic(self):
        """Controller must be deterministic."""
        spec = Globally(Atom('g'))
        result = symbolic_bounded_synthesize(spec, {'r'}, {'g'}, max_states=2)
        ctrl = result.controller
        # Same input -> same output
        s1, o1 = ctrl.step(0, frozenset({'r'}))
        s2, o2 = ctrl.step(0, frozenset({'r'}))
        assert s1 == s2
        assert o1 == o2

    def test_complete_for_all_inputs(self):
        """Controller should handle all env inputs."""
        spec = Globally(Atom('g'))
        result = symbolic_bounded_synthesize(spec, {'r'}, {'g'}, max_states=2)
        ctrl = result.controller
        for e_val in [frozenset(), frozenset({'r'})]:
            step = ctrl.step(0, e_val)
            assert step is not None

    def test_controller_statistics(self):
        spec = Globally(Atom('g'))
        result = symbolic_bounded_synthesize(spec, set(), {'g'}, max_states=2)
        stats = controller_statistics(result.controller)
        assert stats['n_states'] >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
