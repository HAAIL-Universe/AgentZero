"""Tests for V190: Bounded Synthesis (SMT-based)."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V023_ltl_model_checking'))

from bounded_synthesis import (
    UCW, Annotation, Controller, SynthResult, SynthVerdict,
    ucw_from_ltl, ucw_from_nba, ucw_statistics,
    encode_bounded, extract_controller,
    bounded_synthesize, synthesize_safety, synthesize_liveness,
    synthesize_response, synthesize_assume_guarantee,
    find_minimum_controller, synthesize_with_constraints,
    verify_annotation, verify_controller,
    compare_with_game, synthesis_summary,
    controller_to_dict, controller_statistics,
    _all_valuations, _label_matches, _quick_check,
    _trivial_controller, _eval_propositional,
)
from ltl_model_checker import (
    Atom, Not as LTLNot, And as LTLAnd, Or as LTLOr,
    Implies as LTLImplies, Next, Finally, Globally, Until, Release,
    LTL, LTLOp, Label, NBA, LTLTrue, LTLFalse,
)


# ===================================================================
# 1. Data structures
# ===================================================================

class TestDataStructures:
    def test_ucw_creation(self):
        ucw = UCW(
            states={0, 1}, initial={0},
            transitions={0: [(Label(frozenset(), frozenset()), 1)]},
            rejecting={1}, ap={'a'}
        )
        assert len(ucw.states) == 2
        assert ucw.rejecting == {1}

    def test_annotation_creation(self):
        ann = Annotation(values={(0, 0): 3, (1, 0): 2}, max_bound=5)
        assert ann.values[(0, 0)] == 3
        assert ann.max_bound == 5

    def test_controller_creation(self):
        ctrl = Controller(
            n_states=1, initial=0,
            transitions={(0, frozenset()): (0, frozenset({'b'}))},
            env_vars={'a'}, sys_vars={'b'}
        )
        assert ctrl.n_states == 1
        assert ctrl.initial == 0

    def test_controller_step(self):
        ctrl = Controller(
            n_states=2, initial=0,
            transitions={
                (0, frozenset()): (1, frozenset({'b'})),
                (0, frozenset({'a'})): (0, frozenset()),
                (1, frozenset()): (0, frozenset()),
                (1, frozenset({'a'})): (1, frozenset({'b'})),
            },
            env_vars={'a'}, sys_vars={'b'}
        )
        r = ctrl.step(0, frozenset())
        assert r == (1, frozenset({'b'}))
        r = ctrl.step(0, frozenset({'a'}))
        assert r == (0, frozenset())

    def test_controller_simulate(self):
        ctrl = Controller(
            n_states=1, initial=0,
            transitions={(0, frozenset()): (0, frozenset({'b'}))},
            env_vars=set(), sys_vars={'b'}
        )
        trace = ctrl.simulate([frozenset(), frozenset(), frozenset()])
        assert len(trace) == 3
        for state, inp, out, ns in trace:
            assert out == frozenset({'b'})

    def test_synth_result(self):
        r = SynthResult(verdict=SynthVerdict.REALIZABLE, n_states=2, bound=1)
        assert r.verdict == SynthVerdict.REALIZABLE
        assert r.n_states == 2

    def test_synth_verdict_values(self):
        assert SynthVerdict.REALIZABLE.value == "realizable"
        assert SynthVerdict.UNREALIZABLE.value == "unrealizable"
        assert SynthVerdict.UNKNOWN.value == "unknown"


# ===================================================================
# 2. Helpers
# ===================================================================

class TestHelpers:
    def test_all_valuations_empty(self):
        vals = _all_valuations(set())
        assert vals == [frozenset()]

    def test_all_valuations_one(self):
        vals = _all_valuations({'a'})
        assert len(vals) == 2
        assert frozenset() in vals
        assert frozenset({'a'}) in vals

    def test_all_valuations_two(self):
        vals = _all_valuations({'a', 'b'})
        assert len(vals) == 4

    def test_all_valuations_three(self):
        vals = _all_valuations({'a', 'b', 'c'})
        assert len(vals) == 8

    def test_label_matches_empty(self):
        label = Label(frozenset(), frozenset())
        assert _label_matches(label, frozenset())
        assert _label_matches(label, frozenset({'a'}))

    def test_label_matches_pos(self):
        label = Label(frozenset({'a'}), frozenset())
        assert _label_matches(label, frozenset({'a'}))
        assert not _label_matches(label, frozenset())

    def test_label_matches_neg(self):
        label = Label(frozenset(), frozenset({'b'}))
        assert _label_matches(label, frozenset())
        assert not _label_matches(label, frozenset({'b'}))

    def test_label_matches_both(self):
        label = Label(frozenset({'a'}), frozenset({'b'}))
        assert _label_matches(label, frozenset({'a'}))
        assert not _label_matches(label, frozenset({'a', 'b'}))
        assert not _label_matches(label, frozenset())

    def test_eval_propositional_atom(self):
        assert _eval_propositional(Atom('a'), frozenset({'a'}))
        assert not _eval_propositional(Atom('a'), frozenset())

    def test_eval_propositional_not(self):
        assert _eval_propositional(LTLNot(Atom('a')), frozenset())
        assert not _eval_propositional(LTLNot(Atom('a')), frozenset({'a'}))

    def test_eval_propositional_and(self):
        f = LTLAnd(Atom('a'), Atom('b'))
        assert _eval_propositional(f, frozenset({'a', 'b'}))
        assert not _eval_propositional(f, frozenset({'a'}))

    def test_eval_propositional_or(self):
        f = LTLOr(Atom('a'), Atom('b'))
        assert _eval_propositional(f, frozenset({'a'}))
        assert _eval_propositional(f, frozenset({'b'}))
        assert not _eval_propositional(f, frozenset())

    def test_eval_propositional_implies(self):
        f = LTLImplies(Atom('a'), Atom('b'))
        assert _eval_propositional(f, frozenset({'a', 'b'}))
        assert _eval_propositional(f, frozenset())
        assert not _eval_propositional(f, frozenset({'a'}))

    def test_eval_propositional_true_false(self):
        assert _eval_propositional(LTLTrue(), frozenset())
        assert not _eval_propositional(LTLFalse(), frozenset())

    def test_trivial_controller(self):
        ctrl = _trivial_controller({'a'}, {'b'})
        assert ctrl.n_states == 1
        assert len(ctrl.transitions) == 2  # 2 env valuations


# ===================================================================
# 3. UCW construction
# ===================================================================

class TestUCW:
    def test_ucw_from_ltl_true(self):
        ucw = ucw_from_ltl(LTLTrue())
        # not(true) = false, NBA of false has no accepting runs
        # UCW should have states (possibly empty or trivial)
        assert isinstance(ucw, UCW)

    def test_ucw_from_ltl_safety(self):
        # G(a) -- always a
        spec = Globally(Atom('a'))
        ucw = ucw_from_ltl(spec)
        assert isinstance(ucw, UCW)
        assert len(ucw.states) >= 1

    def test_ucw_from_ltl_liveness(self):
        # GF(a) -- a infinitely often
        spec = Globally(Finally(Atom('a')))
        ucw = ucw_from_ltl(spec)
        assert isinstance(ucw, UCW)
        assert len(ucw.ap) >= 1

    def test_ucw_statistics(self):
        spec = Globally(Atom('a'))
        ucw = ucw_from_ltl(spec)
        stats = ucw_statistics(ucw)
        assert 'states' in stats
        assert 'rejecting' in stats
        assert 'transitions' in stats
        assert stats['states'] >= 1

    def test_ucw_from_nba(self):
        nba = NBA(
            states={0, 1}, initial={0},
            transitions={0: [(Label(frozenset({'a'}), frozenset()), 1)],
                         1: [(Label(frozenset(), frozenset()), 0)]},
            accepting={1}, ap={'a'}
        )
        ucw = ucw_from_nba(nba)
        assert ucw.rejecting == {1}
        assert ucw.states == {0, 1}

    def test_ucw_preserves_transitions(self):
        spec = Globally(LTLNot(Atom('a')))
        ucw = ucw_from_ltl(spec)
        total_trans = sum(len(ts) for ts in ucw.transitions.values())
        assert total_trans >= 1


# ===================================================================
# 4. Quick checks
# ===================================================================

class TestQuickCheck:
    def test_true_spec(self):
        r = _quick_check(LTLTrue(), {'a'}, {'b'})
        assert r is not None
        assert r.verdict == SynthVerdict.REALIZABLE

    def test_false_spec(self):
        r = _quick_check(LTLFalse(), {'a'}, {'b'})
        assert r is not None
        assert r.verdict == SynthVerdict.UNREALIZABLE

    def test_g_true(self):
        r = _quick_check(Globally(LTLTrue()), {'a'}, {'b'})
        assert r is not None
        assert r.verdict == SynthVerdict.REALIZABLE

    def test_g_false(self):
        r = _quick_check(Globally(LTLFalse()), {'a'}, {'b'})
        assert r is not None
        assert r.verdict == SynthVerdict.UNREALIZABLE

    def test_nontrivial_returns_none(self):
        r = _quick_check(Globally(Atom('a')), {'a'}, {'b'})
        assert r is None


# ===================================================================
# 5. SMT encoding
# ===================================================================

class TestEncoding:
    def test_encode_creates_solver(self):
        spec = Globally(Atom('b'))
        ucw = ucw_from_ltl(spec)
        solver, var_info, n_vars = encode_bounded(ucw, {'a'}, {'b'}, k=1, b=1)
        assert solver is not None
        assert 'sel' in var_info
        assert 'out' in var_info
        assert 'lam' in var_info
        assert n_vars > 0

    def test_encode_var_counts(self):
        spec = Globally(Atom('b'))
        ucw = ucw_from_ltl(spec)
        _, var_info, n_vars = encode_bounded(ucw, {'a'}, {'b'}, k=2, b=2)
        k = 2
        n_env_vals = 2  # {}, {a}
        n_sys_vars = 1  # b
        n_ucw = len(ucw.states)
        # sel: k * n_env_vals * k (one bool per (c, e_idx, c_next))
        expected_sel = k * n_env_vals * k
        expected_out = k * n_env_vals * n_sys_vars
        expected_lam = n_ucw * k
        assert len(var_info['sel']) == expected_sel
        assert len(var_info['out']) == expected_out
        assert len(var_info['lam']) == expected_lam

    def test_encode_respects_k(self):
        spec = Globally(Atom('b'))
        ucw = ucw_from_ltl(spec)
        _, vi1, _ = encode_bounded(ucw, {'a'}, {'b'}, k=1, b=0)
        _, vi2, _ = encode_bounded(ucw, {'a'}, {'b'}, k=3, b=0)
        assert vi1['k'] == 1
        assert vi2['k'] == 3

    def test_encode_respects_b(self):
        spec = Globally(Atom('b'))
        ucw = ucw_from_ltl(spec)
        _, vi, _ = encode_bounded(ucw, {'a'}, {'b'}, k=1, b=5)
        assert vi['b'] == 5


# ===================================================================
# 6. Core synthesis -- safety specs
# ===================================================================

class TestSafetySynthesis:
    def test_sys_controls_output_safety(self):
        """G(b) where sys controls b -- trivially realizable: always set b."""
        spec = Globally(Atom('b'))
        r = bounded_synthesize(spec, set(), {'b'}, max_states=2)
        assert r.verdict == SynthVerdict.REALIZABLE
        assert r.controller is not None
        # Controller should always output b
        for (c, e), (nc, out) in r.controller.transitions.items():
            assert 'b' in out

    def test_env_controls_safety_unrealizable(self):
        """G(a) where env controls a -- unrealizable (env can set a=false)."""
        spec = Globally(Atom('a'))
        r = bounded_synthesize(spec, {'a'}, set(), max_states=3, max_bound=3)
        # Should be UNKNOWN (can't find controller) since env controls a
        assert r.verdict != SynthVerdict.REALIZABLE

    def test_safety_negation(self):
        """G(!b) where sys controls b -- realizable: never set b."""
        spec = Globally(LTLNot(Atom('b')))
        r = bounded_synthesize(spec, set(), {'b'}, max_states=2)
        assert r.verdict == SynthVerdict.REALIZABLE
        for (c, e), (nc, out) in r.controller.transitions.items():
            assert 'b' not in out

    def test_synthesize_safety_api(self):
        """G(!b) via synthesize_safety API."""
        r = synthesize_safety(Atom('b'), set(), {'b'}, max_states=2)
        assert r.verdict == SynthVerdict.REALIZABLE

    def test_safety_with_env_input(self):
        """G(a -> b): if env sets a, sys must set b."""
        spec = Globally(LTLImplies(Atom('a'), Atom('b')))
        r = bounded_synthesize(spec, {'a'}, {'b'}, max_states=2)
        assert r.verdict == SynthVerdict.REALIZABLE
        # When a is true, b must be true
        ctrl = r.controller
        for (c, e), (nc, out) in ctrl.transitions.items():
            if 'a' in e:
                assert 'b' in out

    def test_safety_conjunction(self):
        """G(b1 & b2): sys must set both outputs."""
        spec = Globally(LTLAnd(Atom('b1'), Atom('b2')))
        r = bounded_synthesize(spec, set(), {'b1', 'b2'}, max_states=2)
        assert r.verdict == SynthVerdict.REALIZABLE
        for (c, e), (nc, out) in r.controller.transitions.items():
            assert 'b1' in out and 'b2' in out


# ===================================================================
# 7. Core synthesis -- liveness specs
# ===================================================================

class TestLivenessSynthesis:
    def test_gf_sys_output(self):
        """GF(b) where sys controls b -- realizable: always set b."""
        spec = Globally(Finally(Atom('b')))
        r = bounded_synthesize(spec, set(), {'b'}, max_states=2)
        assert r.verdict == SynthVerdict.REALIZABLE

    def test_synthesize_liveness_api(self):
        """GF(b) via synthesize_liveness API."""
        r = synthesize_liveness(Atom('b'), set(), {'b'}, max_states=2)
        assert r.verdict == SynthVerdict.REALIZABLE

    def test_gf_env_unrealizable(self):
        """GF(a) where env controls a -- unrealizable."""
        spec = Globally(Finally(Atom('a')))
        r = bounded_synthesize(spec, {'a'}, set(), max_states=3, max_bound=3)
        assert r.verdict != SynthVerdict.REALIZABLE


# ===================================================================
# 8. Core synthesis -- response specs
# ===================================================================

class TestResponseSynthesis:
    def test_response_sys_only(self):
        """G(b1 -> F(b2)): sys controls both -- realizable."""
        spec = Globally(LTLImplies(Atom('b1'), Finally(Atom('b2'))))
        r = bounded_synthesize(spec, set(), {'b1', 'b2'}, max_states=2)
        assert r.verdict == SynthVerdict.REALIZABLE

    def test_synthesize_response_api(self):
        """G(a -> F(b)) via synthesize_response API."""
        r = synthesize_response(Atom('a'), Atom('b'), {'a'}, {'b'}, max_states=3)
        # Sys can respond to env's a by setting b
        assert r.verdict == SynthVerdict.REALIZABLE


# ===================================================================
# 9. Trivial specs
# ===================================================================

class TestTrivialSpecs:
    def test_true_realizable(self):
        r = bounded_synthesize(LTLTrue(), {'a'}, {'b'})
        assert r.verdict == SynthVerdict.REALIZABLE

    def test_false_unrealizable(self):
        r = bounded_synthesize(LTLFalse(), {'a'}, {'b'})
        assert r.verdict == SynthVerdict.UNREALIZABLE

    def test_g_true_realizable(self):
        r = bounded_synthesize(Globally(LTLTrue()), {'a'}, {'b'})
        assert r.verdict == SynthVerdict.REALIZABLE

    def test_g_false_unrealizable(self):
        r = bounded_synthesize(Globally(LTLFalse()), {'a'}, {'b'})
        assert r.verdict == SynthVerdict.UNREALIZABLE

    def test_atom_realizable(self):
        """b (just at initial step) where sys controls b."""
        spec = Atom('b')
        r = bounded_synthesize(spec, set(), {'b'}, max_states=2)
        assert r.verdict == SynthVerdict.REALIZABLE


# ===================================================================
# 10. Minimum controller search
# ===================================================================

class TestMinimumController:
    def test_minimum_safety(self):
        """G(b) needs 1 state."""
        spec = Globally(Atom('b'))
        r = find_minimum_controller(spec, set(), {'b'}, max_states=4)
        assert r.verdict == SynthVerdict.REALIZABLE
        assert r.n_states == 1

    def test_minimum_returns_smallest(self):
        """find_minimum_controller returns first k that works."""
        spec = Globally(LTLNot(Atom('b')))
        r = find_minimum_controller(spec, set(), {'b'}, max_states=4)
        assert r.verdict == SynthVerdict.REALIZABLE
        assert r.n_states == 1

    def test_minimum_trivial(self):
        """Trivial spec needs 1 state."""
        r = find_minimum_controller(LTLTrue(), {'a'}, {'b'})
        assert r.verdict == SynthVerdict.REALIZABLE
        assert r.n_states == 1


# ===================================================================
# 11. Annotation verification
# ===================================================================

class TestAnnotation:
    def test_verify_valid_annotation(self):
        """Synthesized annotation should be valid."""
        spec = Globally(Atom('b'))
        r = bounded_synthesize(spec, set(), {'b'}, max_states=2)
        if r.verdict == SynthVerdict.REALIZABLE and r.annotation:
            ucw = ucw_from_ltl(spec)
            valid, violations = verify_annotation(r.controller, ucw, r.annotation)
            assert valid, f"Annotation violations: {violations}"

    def test_annotation_has_values(self):
        spec = Globally(Atom('b'))
        r = bounded_synthesize(spec, set(), {'b'}, max_states=2)
        if r.annotation:
            assert len(r.annotation.values) > 0

    def test_verify_trivial_annotation(self):
        """All-zero annotation with no rejecting states is valid."""
        ucw = UCW(states={0}, initial={0},
                  transitions={0: [(Label(frozenset(), frozenset()), 0)]},
                  rejecting=set(), ap=set())
        ctrl = Controller(
            n_states=1, initial=0,
            transitions={(0, frozenset()): (0, frozenset())},
            env_vars=set(), sys_vars=set()
        )
        ann = Annotation(values={(0, 0): 0}, max_bound=0)
        valid, violations = verify_annotation(ctrl, ucw, ann)
        assert valid


# ===================================================================
# 12. Controller verification
# ===================================================================

class TestControllerVerification:
    def test_verify_safety_controller(self):
        """Synthesized safety controller should pass verification."""
        spec = Globally(Atom('b'))
        r = bounded_synthesize(spec, set(), {'b'}, max_states=2)
        if r.verdict == SynthVerdict.REALIZABLE:
            valid, violations = verify_controller(r.controller, spec, set(), {'b'})
            assert valid, f"Violations: {violations}"

    def test_verify_bad_controller(self):
        """Controller that violates safety should fail verification."""
        spec = Globally(Atom('b'))
        # Controller that never sets b
        bad_ctrl = Controller(
            n_states=1, initial=0,
            transitions={(0, frozenset()): (0, frozenset())},
            env_vars=set(), sys_vars={'b'}
        )
        valid, violations = verify_controller(bad_ctrl, spec, set(), {'b'})
        assert not valid

    def test_verify_implication_controller(self):
        """G(a -> b) controller should pass."""
        spec = Globally(LTLImplies(Atom('a'), Atom('b')))
        r = bounded_synthesize(spec, {'a'}, {'b'}, max_states=2)
        if r.verdict == SynthVerdict.REALIZABLE:
            valid, violations = verify_controller(r.controller, spec, {'a'}, {'b'})
            assert valid


# ===================================================================
# 13. Synthesis with constraints
# ===================================================================

class TestConstrainedSynthesis:
    def test_constrained_output(self):
        """Force controller to always output b."""
        spec = Globally(Atom('b'))

        def force_b(solver, var_info):
            # Force all outputs to include b
            for key, var in var_info['out'].items():
                c, e_idx, v = key
                if v == 'b':
                    solver.add(var)

        r = synthesize_with_constraints(spec, set(), {'b'}, force_b, max_states=2)
        assert r.verdict == SynthVerdict.REALIZABLE

    def test_constrained_initial_output(self):
        """Constrain initial state output."""
        spec = Globally(LTLNot(Atom('b')))

        def no_b(solver, var_info):
            for key, var in var_info['out'].items():
                c, e_idx, v = key
                if v == 'b':
                    solver.add(solver.Not(var))

        r = synthesize_with_constraints(spec, set(), {'b'}, no_b, max_states=2)
        assert r.verdict == SynthVerdict.REALIZABLE


# ===================================================================
# 14. Assume-guarantee synthesis
# ===================================================================

class TestAssumeGuarantee:
    def test_assume_guarantee_trivial(self):
        """false -> G(b) is trivially realizable (vacuous assumption)."""
        r = synthesize_assume_guarantee(
            LTLFalse(), Globally(Atom('b')),
            {'a'}, {'b'}, max_states=2
        )
        assert r.verdict == SynthVerdict.REALIZABLE

    def test_assume_guarantee_safety(self):
        """G(a) -> G(b): if env always provides a, sys always provides b."""
        r = synthesize_assume_guarantee(
            Globally(Atom('a')), Globally(Atom('b')),
            {'a'}, {'b'}, max_states=2
        )
        assert r.verdict == SynthVerdict.REALIZABLE


# ===================================================================
# 15. Controller utilities
# ===================================================================

class TestControllerUtils:
    def test_controller_to_dict(self):
        ctrl = Controller(
            n_states=1, initial=0,
            transitions={(0, frozenset()): (0, frozenset({'b'}))},
            env_vars=set(), sys_vars={'b'}
        )
        d = controller_to_dict(ctrl)
        assert d['n_states'] == 1
        assert d['initial'] == 0
        assert 'transitions' in d

    def test_controller_statistics(self):
        ctrl = Controller(
            n_states=2, initial=0,
            transitions={
                (0, frozenset()): (1, frozenset({'b'})),
                (1, frozenset()): (0, frozenset()),
            },
            env_vars=set(), sys_vars={'b'}
        )
        stats = controller_statistics(ctrl)
        assert stats['n_states'] == 2
        assert stats['transitions'] == 2
        assert stats['states_used'] == 2

    def test_controller_simulate_length(self):
        ctrl = Controller(
            n_states=1, initial=0,
            transitions={(0, frozenset()): (0, frozenset({'b'}))},
            env_vars=set(), sys_vars={'b'}
        )
        trace = ctrl.simulate([frozenset()] * 10)
        assert len(trace) == 10

    def test_controller_simulate_max_steps(self):
        ctrl = Controller(
            n_states=1, initial=0,
            transitions={(0, frozenset()): (0, frozenset())},
            env_vars=set(), sys_vars=set()
        )
        trace = ctrl.simulate([frozenset()] * 100, max_steps=5)
        assert len(trace) == 5


# ===================================================================
# 16. Synthesis summary
# ===================================================================

class TestSummary:
    def test_summary_realizable(self):
        r = SynthResult(
            verdict=SynthVerdict.REALIZABLE, n_states=2, bound=1,
            ucw_states=3, method="bounded_synthesis"
        )
        s = synthesis_summary(r)
        assert "realizable" in s
        assert "2" in s

    def test_summary_unknown(self):
        r = SynthResult(
            verdict=SynthVerdict.UNKNOWN, ucw_states=5,
            method="bounded_synthesis"
        )
        s = synthesis_summary(r)
        assert "unknown" in s

    def test_summary_unrealizable(self):
        r = SynthResult(verdict=SynthVerdict.UNREALIZABLE, method="quick_check")
        s = synthesis_summary(r)
        assert "unrealizable" in s


# ===================================================================
# 17. Comparison with game-based approach
# ===================================================================

class TestComparison:
    def test_compare_safety(self):
        """Both methods should agree on simple safety spec."""
        spec = Globally(Atom('b'))
        result = compare_with_game(spec, set(), {'b'}, max_states=2)
        assert 'smt_verdict' in result
        assert 'game_verdict' in result

    def test_compare_trivial(self):
        result = compare_with_game(LTLTrue(), {'a'}, {'b'}, max_states=2)
        assert result['smt_verdict'] == 'realizable'


# ===================================================================
# 18. Edge cases
# ===================================================================

class TestEdgeCases:
    def test_no_env_vars(self):
        """No env vars -- sys has full control."""
        spec = Globally(Atom('b'))
        r = bounded_synthesize(spec, set(), {'b'}, max_states=2)
        assert r.verdict == SynthVerdict.REALIZABLE

    def test_no_sys_vars(self):
        """No sys vars -- sys can't influence anything."""
        spec = Globally(Atom('a'))
        r = bounded_synthesize(spec, {'a'}, set(), max_states=2, max_bound=2)
        # Unrealizable since sys has no control
        assert r.verdict != SynthVerdict.REALIZABLE

    def test_single_state_controller(self):
        """Simple spec should be solvable with 1 state."""
        spec = Globally(Atom('b'))
        r = bounded_synthesize(spec, set(), {'b'}, max_states=1)
        assert r.verdict == SynthVerdict.REALIZABLE
        assert r.n_states == 1

    def test_controller_step_missing(self):
        """Step on undefined transition returns None."""
        ctrl = Controller(
            n_states=1, initial=0,
            transitions={(0, frozenset()): (0, frozenset())},
            env_vars=set(), sys_vars=set()
        )
        assert ctrl.step(0, frozenset({'x'})) is None

    def test_empty_simulation(self):
        ctrl = Controller(
            n_states=1, initial=0,
            transitions={(0, frozenset()): (0, frozenset())},
            env_vars=set(), sys_vars=set()
        )
        trace = ctrl.simulate([])
        assert trace == []


# ===================================================================
# 19. Multi-variable specs
# ===================================================================

class TestMultiVariable:
    def test_two_sys_vars(self):
        """G(b1 & !b2): set b1, clear b2."""
        spec = Globally(LTLAnd(Atom('b1'), LTLNot(Atom('b2'))))
        r = bounded_synthesize(spec, set(), {'b1', 'b2'}, max_states=2)
        assert r.verdict == SynthVerdict.REALIZABLE
        for (c, e), (nc, out) in r.controller.transitions.items():
            assert 'b1' in out
            assert 'b2' not in out

    def test_env_sys_interaction(self):
        """G(a -> b): sys copies env input."""
        spec = Globally(LTLImplies(Atom('a'), Atom('b')))
        r = bounded_synthesize(spec, {'a'}, {'b'}, max_states=2)
        assert r.verdict == SynthVerdict.REALIZABLE
        ctrl = r.controller
        for (c, e), (nc, out) in ctrl.transitions.items():
            if 'a' in e:
                assert 'b' in out

    def test_or_spec(self):
        """G(b1 | b2): at least one output must be true."""
        spec = Globally(LTLOr(Atom('b1'), Atom('b2')))
        r = bounded_synthesize(spec, set(), {'b1', 'b2'}, max_states=2)
        assert r.verdict == SynthVerdict.REALIZABLE
        for (c, e), (nc, out) in r.controller.transitions.items():
            assert 'b1' in out or 'b2' in out


# ===================================================================
# 20. UCW structure tests
# ===================================================================

class TestUCWStructure:
    def test_ucw_has_initial(self):
        spec = Globally(Atom('a'))
        ucw = ucw_from_ltl(spec)
        assert len(ucw.initial) >= 1

    def test_ucw_ap_matches_spec(self):
        spec = Globally(LTLAnd(Atom('a'), Atom('b')))
        ucw = ucw_from_ltl(spec)
        assert 'a' in ucw.ap
        assert 'b' in ucw.ap

    def test_ucw_rejecting_subset_of_states(self):
        spec = Globally(Finally(Atom('a')))
        ucw = ucw_from_ltl(spec)
        assert ucw.rejecting.issubset(ucw.states)

    def test_ucw_transitions_well_formed(self):
        spec = Globally(Atom('b'))
        ucw = ucw_from_ltl(spec)
        for q, trans in ucw.transitions.items():
            assert q in ucw.states
            for label, q_next in trans:
                assert q_next in ucw.states
                assert isinstance(label, Label)


# ===================================================================
# 21. Integration: synthesis + annotation + verification
# ===================================================================

class TestIntegration:
    def test_full_pipeline_safety(self):
        """Full pipeline: synthesize, verify annotation, verify controller."""
        spec = Globally(Atom('b'))
        r = bounded_synthesize(spec, set(), {'b'}, max_states=2)
        assert r.verdict == SynthVerdict.REALIZABLE

        # Verify annotation
        ucw = ucw_from_ltl(spec)
        valid, _ = verify_annotation(r.controller, ucw, r.annotation)
        assert valid

        # Verify controller
        valid, _ = verify_controller(r.controller, spec, set(), {'b'})
        assert valid

    def test_full_pipeline_implication(self):
        """Full pipeline for G(a -> b)."""
        spec = Globally(LTLImplies(Atom('a'), Atom('b')))
        r = bounded_synthesize(spec, {'a'}, {'b'}, max_states=2)
        assert r.verdict == SynthVerdict.REALIZABLE

        ucw = ucw_from_ltl(spec)
        valid, _ = verify_annotation(r.controller, ucw, r.annotation)
        assert valid

        valid, _ = verify_controller(r.controller, spec, {'a'}, {'b'})
        assert valid

    def test_synthesize_and_simulate(self):
        """Synthesize, then simulate to confirm behavior."""
        spec = Globally(Atom('b'))
        r = bounded_synthesize(spec, {'a'}, {'b'}, max_states=2)
        assert r.verdict == SynthVerdict.REALIZABLE

        inputs = [frozenset(), frozenset({'a'}), frozenset(), frozenset({'a'})]
        trace = r.controller.simulate(inputs)
        assert len(trace) == 4
        for state, inp, out, ns in trace:
            assert 'b' in out

    def test_minimum_then_verify(self):
        """Find minimum controller, then verify it."""
        spec = Globally(LTLNot(Atom('b')))
        r = find_minimum_controller(spec, set(), {'b'}, max_states=4)
        assert r.verdict == SynthVerdict.REALIZABLE
        assert r.n_states == 1

        valid, _ = verify_controller(r.controller, spec, set(), {'b'})
        assert valid


# ===================================================================
# 22. Result details
# ===================================================================

class TestResultDetails:
    def test_result_has_method(self):
        r = bounded_synthesize(LTLTrue(), set(), {'b'})
        assert r.method != ""

    def test_result_ucw_states(self):
        spec = Globally(Atom('b'))
        r = bounded_synthesize(spec, set(), {'b'}, max_states=2)
        # Either realizable with ucw_states or quick-checked
        assert isinstance(r.ucw_states, int)

    def test_result_smt_vars(self):
        spec = Globally(Atom('b'))
        r = bounded_synthesize(spec, set(), {'b'}, max_states=2)
        if r.verdict == SynthVerdict.REALIZABLE and r.method == "bounded_synthesis":
            assert r.smt_vars > 0

    def test_result_bound(self):
        spec = Globally(Atom('b'))
        r = bounded_synthesize(spec, set(), {'b'}, max_states=2)
        if r.verdict == SynthVerdict.REALIZABLE:
            assert r.bound >= 0
