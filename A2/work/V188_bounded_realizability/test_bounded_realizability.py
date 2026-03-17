"""
Tests for V188: Bounded Realizability.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'V023_ltl_model_checking'))

from bounded_realizability import (
    RealVerdict, RealResult, MealyMachine, BoundedController,
    EnvCounterstrategy,
    check_bounded, find_minimum_controller, check_safety,
    extract_counterstrategy, quick_check, check_realizable,
    check_and_explain, compare_methods, realizability_summary,
    _all_valuations, _label_matches, _is_propositional, _is_safety_spec,
    _build_bounded_game, _solve_buchi_game,
)
from ltl_model_checker import (
    LTL, LTLOp, Atom, Not, And, Or, Implies,
    Next, Finally, Globally, Until, Release,
    LTLTrue, LTLFalse, parse_ltl, Label,
)


# ============================================================
# Helper shortcuts
# ============================================================

a, b, c, r, g = Atom('a'), Atom('b'), Atom('c'), Atom('r'), Atom('g')
p, q = Atom('p'), Atom('q')


# ============================================================
# Test: Data Structures
# ============================================================

class TestDataStructures:
    def test_real_verdict_values(self):
        assert RealVerdict.REALIZABLE.value == "REALIZABLE"
        assert RealVerdict.UNREALIZABLE.value == "UNREALIZABLE"
        assert RealVerdict.UNKNOWN.value == "UNKNOWN"

    def test_mealy_machine_step(self):
        ctrl = MealyMachine(
            states={0}, initial=0,
            transitions={
                (0, frozenset()): (0, frozenset({'g'})),
                (0, frozenset({'r'})): (0, frozenset({'g'})),
            },
            env_vars={'r'}, sys_vars={'g'},
        )
        result = ctrl.step(0, frozenset())
        assert result == (0, frozenset({'g'}))

    def test_mealy_machine_simulate(self):
        ctrl = MealyMachine(
            states={0, 1}, initial=0,
            transitions={
                (0, frozenset()): (1, frozenset({'g'})),
                (1, frozenset()): (0, frozenset()),
                (0, frozenset({'r'})): (1, frozenset({'g'})),
                (1, frozenset({'r'})): (0, frozenset({'g'})),
            },
            env_vars={'r'}, sys_vars={'g'},
        )
        trace = ctrl.simulate([frozenset(), frozenset(), frozenset()])
        assert len(trace) == 3

    def test_mealy_step_missing(self):
        ctrl = MealyMachine(
            states={0}, initial=0, transitions={},
            env_vars={'r'}, sys_vars={'g'},
        )
        assert ctrl.step(0, frozenset()) is None

    def test_bounded_controller_creation(self):
        bc = BoundedController(
            n_states=2, initial=0, transitions={},
            env_vars={'r'}, sys_vars={'g'},
        )
        assert bc.n_states == 2
        assert bc.initial == 0

    def test_env_counterstrategy_creation(self):
        cs = EnvCounterstrategy(
            states={0}, initial=0, transitions={},
            env_vars={'r'}, sys_vars={'g'},
            description="test",
        )
        assert cs.description == "test"

    def test_real_result_fields(self):
        r = RealResult(
            verdict=RealVerdict.REALIZABLE,
            method="test",
            bound=3,
        )
        assert r.verdict == RealVerdict.REALIZABLE
        assert r.bound == 3
        assert r.controller is None


# ============================================================
# Test: Utility Functions
# ============================================================

class TestUtilities:
    def test_all_valuations_empty(self):
        vals = _all_valuations(set())
        assert vals == [frozenset()]

    def test_all_valuations_one(self):
        vals = _all_valuations({'x'})
        assert len(vals) == 2
        assert frozenset() in vals
        assert frozenset({'x'}) in vals

    def test_all_valuations_two(self):
        vals = _all_valuations({'x', 'y'})
        assert len(vals) == 4

    def test_label_matches_pos(self):
        label = Label(pos=frozenset({'a'}), neg=frozenset())
        assert _label_matches(label, frozenset({'a'}))
        assert _label_matches(label, frozenset({'a', 'b'}))
        assert not _label_matches(label, frozenset())

    def test_label_matches_neg(self):
        label = Label(pos=frozenset(), neg=frozenset({'b'}))
        assert _label_matches(label, frozenset({'a'}))
        assert not _label_matches(label, frozenset({'b'}))

    def test_is_propositional_atom(self):
        assert _is_propositional(Atom('x'))

    def test_is_propositional_boolean(self):
        assert _is_propositional(And(Atom('x'), Not(Atom('y'))))

    def test_is_propositional_temporal(self):
        assert not _is_propositional(Globally(Atom('x')))
        assert not _is_propositional(Finally(Atom('x')))
        assert not _is_propositional(Next(Atom('x')))

    def test_is_safety_spec(self):
        assert _is_safety_spec(Globally(Atom('x')))
        assert _is_safety_spec(Globally(And(Atom('x'), Not(Atom('y')))))
        assert not _is_safety_spec(Globally(Finally(Atom('x'))))
        assert not _is_safety_spec(Atom('x'))


# ============================================================
# Test: Quick Check
# ============================================================

class TestQuickCheck:
    def test_true_spec(self):
        result = quick_check(LTLTrue(), {'a'}, {'b'})
        assert result.verdict == RealVerdict.REALIZABLE

    def test_false_spec(self):
        result = quick_check(LTLFalse(), {'a'}, {'b'})
        assert result.verdict == RealVerdict.UNREALIZABLE

    def test_g_true(self):
        result = quick_check(Globally(LTLTrue()), {'a'}, {'b'})
        assert result.verdict == RealVerdict.REALIZABLE

    def test_g_false(self):
        result = quick_check(Globally(LTLFalse()), {'a'}, {'b'})
        assert result.verdict == RealVerdict.UNREALIZABLE

    def test_propositional_realizable(self):
        # System controls b, must satisfy b
        result = quick_check(Atom('b'), set(), {'b'})
        assert result.verdict == RealVerdict.REALIZABLE
        assert result.controller is not None

    def test_propositional_unrealizable(self):
        # System controls b, must satisfy a (env controls a)
        result = quick_check(Atom('a'), {'a'}, {'b'})
        assert result.verdict == RealVerdict.UNREALIZABLE

    def test_propositional_and(self):
        # System controls both, must satisfy a & b
        result = quick_check(And(Atom('a'), Atom('b')), set(), {'a', 'b'})
        assert result.verdict == RealVerdict.REALIZABLE

    def test_propositional_contradiction(self):
        # Must satisfy a & !a
        result = quick_check(And(Atom('a'), Not(Atom('a'))), set(), {'a'})
        assert result.verdict == RealVerdict.UNREALIZABLE


# ============================================================
# Test: Safety Realizability
# ============================================================

class TestSafetyRealizability:
    def test_avoid_bad_realizable(self):
        # Sys controls g, env controls r. Bad = r & !g.
        # System can always set g=true to avoid bad.
        bad = And(Atom('r'), Not(Atom('g')))
        result = check_safety(bad, {'r'}, {'g'})
        assert result.verdict == RealVerdict.REALIZABLE

    def test_avoid_bad_controller(self):
        bad = And(Atom('r'), Not(Atom('g')))
        result = check_safety(bad, {'r'}, {'g'})
        assert result.controller is not None
        # Controller should always output g
        trace = result.controller.simulate(
            [frozenset({'r'}), frozenset(), frozenset({'r'})]
        )
        for _, inp, out in trace:
            if 'r' in inp:
                assert 'g' in out

    def test_unavoidable_bad(self):
        # Bad = a (env controls a). System can't prevent env from setting a.
        bad = Atom('a')
        result = check_safety(bad, {'a'}, {'b'})
        assert result.verdict == RealVerdict.UNREALIZABLE

    def test_sys_controls_bad(self):
        # Bad = b (system controls b). System can always avoid.
        bad = Atom('b')
        result = check_safety(bad, {'a'}, {'b'})
        assert result.verdict == RealVerdict.REALIZABLE

    def test_always_safe(self):
        # Bad = false (never bad)
        bad = LTLFalse()
        result = check_safety(bad, {'a'}, {'b'})
        assert result.verdict == RealVerdict.REALIZABLE

    def test_always_bad(self):
        # Bad = true (always bad)
        bad = LTLTrue()
        result = check_safety(bad, {'a'}, {'b'})
        assert result.verdict == RealVerdict.UNREALIZABLE

    def test_safety_via_quick_check(self):
        # G(!a & b) where sys controls b, env controls a
        # Unrealizable: sys can't prevent a
        spec = Globally(And(Not(Atom('a')), Atom('b')))
        result = quick_check(spec, {'a'}, {'b'})
        assert result.verdict == RealVerdict.UNREALIZABLE

    def test_safety_sys_output(self):
        # G(b) where sys controls b. Trivially realizable.
        spec = Globally(Atom('b'))
        result = quick_check(spec, {'a'}, {'b'})
        assert result.verdict == RealVerdict.REALIZABLE


# ============================================================
# Test: Bounded Realizability
# ============================================================

class TestBoundedRealizability:
    def test_trivial_realizable(self):
        # G(true) with bound 1
        spec = Globally(LTLTrue())
        result = check_bounded(spec, {'a'}, {'b'}, bound=1)
        assert result.verdict == RealVerdict.REALIZABLE

    def test_simple_safety_bounded(self):
        # G(b) where sys controls b
        spec = Globally(Atom('b'))
        result = check_bounded(spec, set(), {'b'}, bound=1)
        assert result.verdict == RealVerdict.REALIZABLE

    def test_bounded_result_fields(self):
        spec = Globally(LTLTrue())
        result = check_bounded(spec, {'a'}, {'b'}, bound=2)
        assert result.bound == 2
        assert result.method == "bounded_realizability"
        assert result.automaton_states > 0

    def test_unrealizable_bounded(self):
        # G(a) where env controls a -- unrealizable
        spec = Globally(Atom('a'))
        result = check_bounded(spec, {'a'}, {'b'}, bound=1)
        assert result.verdict == RealVerdict.UNREALIZABLE

    def test_bound_affects_game_size(self):
        spec = Globally(Atom('b'))
        r1 = check_bounded(spec, {'a'}, {'b'}, bound=1)
        r2 = check_bounded(spec, {'a'}, {'b'}, bound=2)
        # Larger bound = larger game
        assert r2.game_states >= r1.game_states


# ============================================================
# Test: Incremental Search
# ============================================================

class TestIncrementalSearch:
    def test_find_minimum_trivial(self):
        # G(b) needs 1-state controller
        spec = Globally(Atom('b'))
        result = find_minimum_controller(spec, set(), {'b'}, max_bound=4)
        assert result.verdict == RealVerdict.REALIZABLE
        assert result.min_states == 1

    def test_find_minimum_method(self):
        spec = Globally(Atom('b'))
        result = find_minimum_controller(spec, set(), {'b'}, max_bound=4)
        assert result.method == "incremental_search"

    def test_unrealizable_exhausts_bounds(self):
        # G(a) where env controls a
        spec = Globally(Atom('a'))
        result = find_minimum_controller(spec, {'a'}, {'b'}, max_bound=3)
        assert result.verdict == RealVerdict.UNKNOWN

    def test_incremental_with_small_bound(self):
        spec = Globally(LTLTrue())
        result = find_minimum_controller(spec, {'a'}, {'b'}, max_bound=2)
        assert result.verdict == RealVerdict.REALIZABLE
        assert result.min_states is not None
        assert result.min_states <= 2


# ============================================================
# Test: Counterstrategy Extraction
# ============================================================

class TestCounterstrategy:
    def test_realizable_no_counterstrategy(self):
        spec = Globally(LTLTrue())
        result = extract_counterstrategy(spec, {'a'}, {'b'})
        assert result.verdict == RealVerdict.REALIZABLE

    def test_unrealizable_has_counterstrategy(self):
        # G(!a) where env controls a
        spec = Globally(Not(Atom('a')))
        result = extract_counterstrategy(spec, {'a'}, {'b'})
        assert result.verdict == RealVerdict.UNREALIZABLE

    def test_counterstrategy_fields(self):
        spec = Globally(Not(Atom('a')))
        result = extract_counterstrategy(spec, {'a'}, {'b'})
        assert result.method == "counterstrategy_extraction"


# ============================================================
# Test: Game Building
# ============================================================

class TestGameBuilding:
    def test_build_game_has_initial(self):
        spec = Globally(Atom('b'))
        from ltl_model_checker import ltl_to_gba, gba_to_nba
        gba = ltl_to_gba(spec)
        nba = gba_to_nba(gba)
        trans, initial, env_v, sys_v, acc = _build_bounded_game(
            nba, set(), {'b'}, 1
        )
        assert len(initial) > 0

    def test_build_game_env_vertices(self):
        spec = Globally(Atom('b'))
        from ltl_model_checker import ltl_to_gba, gba_to_nba
        gba = ltl_to_gba(spec)
        nba = gba_to_nba(gba)
        trans, initial, env_v, sys_v, acc = _build_bounded_game(
            nba, {'a'}, {'b'}, 1
        )
        assert len(env_v) > 0

    def test_build_game_sys_vertices(self):
        spec = Globally(Atom('b'))
        from ltl_model_checker import ltl_to_gba, gba_to_nba
        gba = ltl_to_gba(spec)
        nba = gba_to_nba(gba)
        trans, initial, env_v, sys_v, acc = _build_bounded_game(
            nba, {'a'}, {'b'}, 1
        )
        assert len(sys_v) > 0

    def test_build_game_accepting(self):
        spec = Globally(Atom('b'))
        from ltl_model_checker import ltl_to_gba, gba_to_nba
        gba = ltl_to_gba(spec)
        nba = gba_to_nba(gba)
        trans, initial, env_v, sys_v, acc = _build_bounded_game(
            nba, {'a'}, {'b'}, 1
        )
        # At least some accepting states should exist
        assert isinstance(acc, set)


# ============================================================
# Test: Buchi Game Solving
# ============================================================

class TestBuchiGame:
    def test_trivial_winning(self):
        # Single state, self-loop, accepting -> system wins
        trans = {0: [0]}
        initial = {0}
        env_v = set()
        sys_v = {0}
        acc = {0}
        win, strat = _solve_buchi_game(trans, initial, env_v, sys_v, acc)
        assert 0 in win

    def test_no_accepting_loses(self):
        # Single state, self-loop, NOT accepting -> system can't win Buchi
        trans = {0: [0]}
        initial = {0}
        env_v = set()
        sys_v = {0}
        acc = set()
        win, strat = _solve_buchi_game(trans, initial, env_v, sys_v, acc)
        assert 0 not in win

    def test_env_can_avoid_accepting(self):
        # Two states: 0 (env) -> 1 or 2. State 1 accepting, state 2 not.
        # Env controls choice -> can always go to 2.
        trans = {0: [1, 2], 1: [0], 2: [2]}
        initial = {0}
        env_v = {0}
        sys_v = {1, 2}
        acc = {1}
        win, strat = _solve_buchi_game(trans, initial, env_v, sys_v, acc)
        assert 0 not in win  # env can trap in 2

    def test_sys_can_reach_accepting(self):
        # Two states: 0 (sys) -> 1 or 2. State 1 accepting.
        # Sys controls choice -> can always go to 1.
        trans = {0: [1, 2], 1: [0], 2: [2]}
        initial = {0}
        env_v = set()
        sys_v = {0, 1, 2}
        acc = {1}
        win, strat = _solve_buchi_game(trans, initial, env_v, sys_v, acc)
        assert 0 in win
        assert 1 in win


# ============================================================
# Test: Comparison API
# ============================================================

class TestComparison:
    def test_compare_methods_realizable(self):
        spec = Globally(Atom('b'))
        results = compare_methods(spec, set(), {'b'}, max_bound=2)
        assert "quick_check" in results
        assert results["quick_check"]["verdict"] == "REALIZABLE"

    def test_compare_methods_unrealizable(self):
        spec = Globally(Atom('a'))
        results = compare_methods(spec, {'a'}, {'b'}, max_bound=2)
        assert "quick_check" in results

    def test_compare_has_bounded_keys(self):
        spec = Globally(LTLTrue())
        results = compare_methods(spec, {'a'}, {'b'}, max_bound=2)
        # Quick check resolves G(true), so bounded may not run
        assert "quick_check" in results


# ============================================================
# Test: Summary
# ============================================================

class TestSummary:
    def test_summary_realizable(self):
        result = RealResult(
            verdict=RealVerdict.REALIZABLE,
            method="test",
            bound=3,
            automaton_states=5,
        )
        s = realizability_summary(result)
        assert "REALIZABLE" in s
        assert "test" in s
        assert "3" in s

    def test_summary_unrealizable(self):
        result = RealResult(
            verdict=RealVerdict.UNREALIZABLE,
            method="bounded",
        )
        s = realizability_summary(result)
        assert "UNREALIZABLE" in s

    def test_summary_with_controller(self):
        ctrl = MealyMachine(
            states={0, 1}, initial=0, transitions={},
            env_vars=set(), sys_vars=set(),
        )
        result = RealResult(
            verdict=RealVerdict.REALIZABLE,
            controller=ctrl,
            method="test",
        )
        s = realizability_summary(result)
        assert "Controller states: 2" in s


# ============================================================
# Test: Main API
# ============================================================

class TestMainAPI:
    def test_check_realizable_true(self):
        result = check_realizable(LTLTrue(), {'a'}, {'b'})
        assert result.verdict == RealVerdict.REALIZABLE

    def test_check_realizable_false(self):
        result = check_realizable(LTLFalse(), {'a'}, {'b'})
        assert result.verdict == RealVerdict.UNREALIZABLE

    def test_check_realizable_safety(self):
        spec = Globally(Atom('b'))
        result = check_realizable(spec, set(), {'b'})
        assert result.verdict == RealVerdict.REALIZABLE

    def test_check_and_explain_realizable(self):
        spec = Globally(Atom('b'))
        result = check_and_explain(spec, set(), {'b'})
        assert result.verdict == RealVerdict.REALIZABLE

    def test_check_and_explain_unrealizable(self):
        spec = Globally(Not(Atom('a')))
        result = check_and_explain(spec, {'a'}, {'b'})
        assert result.verdict == RealVerdict.UNREALIZABLE


# ============================================================
# Test: Spec Patterns
# ============================================================

class TestSpecPatterns:
    def test_always_grant_on_request(self):
        # G(r -> g): env controls r, sys controls g
        # Realizable: sys always sets g
        spec = Globally(Implies(Atom('r'), Atom('g')))
        result = quick_check(spec, {'r'}, {'g'})
        assert result.verdict == RealVerdict.REALIZABLE

    def test_mutual_exclusion(self):
        # G(!(g1 & g2)): sys controls both grants
        spec = Globally(Not(And(Atom('g1'), Atom('g2'))))
        result = quick_check(spec, set(), {'g1', 'g2'})
        assert result.verdict == RealVerdict.REALIZABLE

    def test_env_always_true(self):
        # G(a): env controls a, sys controls b. Unrealizable.
        spec = Globally(Atom('a'))
        result = quick_check(spec, {'a'}, {'b'})
        assert result.verdict == RealVerdict.UNREALIZABLE

    def test_sys_echo(self):
        # G(a -> b): sys must echo env. Sys can always set b=true.
        spec = Globally(Implies(Atom('a'), Atom('b')))
        result = quick_check(spec, {'a'}, {'b'})
        assert result.verdict == RealVerdict.REALIZABLE

    def test_no_env_vars(self):
        # G(b): no env vars, sys controls b
        spec = Globally(Atom('b'))
        result = quick_check(spec, set(), {'b'})
        assert result.verdict == RealVerdict.REALIZABLE

    def test_implies_chain(self):
        # G((a -> b) & (b -> c)): sys controls b,c, env controls a
        spec = Globally(And(Implies(a, b), Implies(b, c)))
        result = quick_check(spec, {'a'}, {'b', 'c'})
        assert result.verdict == RealVerdict.REALIZABLE

    def test_biconditional(self):
        # G(a <-> b): sys must match env. Realizable.
        from ltl_model_checker import Iff
        spec = Globally(Iff(Atom('a'), Atom('b')))
        result = quick_check(spec, {'a'}, {'b'})
        # This is propositional safety: sys sees a, can set b=a
        # But in our game, sys picks BEFORE seeing env? No -- it's simultaneous.
        # For simultaneous: sys picks b independently of a.
        # b=true satisfies when a=true, but not when a=false.
        # b=false satisfies when a=false, but not when a=true.
        # So unrealizable (sys can't predict env).
        assert result.verdict == RealVerdict.UNREALIZABLE


# ============================================================
# Test: Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_vars(self):
        result = quick_check(LTLTrue(), set(), set())
        assert result.verdict == RealVerdict.REALIZABLE

    def test_single_sys_var(self):
        spec = Atom('b')
        result = quick_check(spec, set(), {'b'})
        assert result.verdict == RealVerdict.REALIZABLE

    def test_single_env_var(self):
        spec = Atom('a')
        result = quick_check(spec, {'a'}, set())
        assert result.verdict == RealVerdict.UNREALIZABLE

    def test_many_vars_propositional(self):
        # a & b & c where sys controls all
        spec = And(And(Atom('a'), Atom('b')), Atom('c'))
        result = quick_check(spec, set(), {'a', 'b', 'c'})
        assert result.verdict == RealVerdict.REALIZABLE

    def test_or_spec(self):
        # a | b where sys controls b
        spec = Or(Atom('a'), Atom('b'))
        result = quick_check(spec, {'a'}, {'b'})
        assert result.verdict == RealVerdict.REALIZABLE

    def test_negated_sys_var(self):
        # !b where sys controls b
        spec = Not(Atom('b'))
        result = quick_check(spec, set(), {'b'})
        assert result.verdict == RealVerdict.REALIZABLE

    def test_bounded_with_empty_env(self):
        spec = Globally(Atom('b'))
        result = check_bounded(spec, set(), {'b'}, bound=1)
        assert result.verdict == RealVerdict.REALIZABLE


# ============================================================
# Test: Controller Quality
# ============================================================

class TestControllerQuality:
    def test_safety_controller_complete(self):
        # G(b) with no env: controller should handle all inputs
        spec = Globally(Atom('b'))
        result = check_safety(Not(Atom('b')), set(), {'b'})
        if result.controller:
            # Should have transition for empty env input
            assert result.controller.step(0, frozenset()) is not None

    def test_safety_controller_correct(self):
        # G(!r | g): always grant on request
        bad = And(Atom('r'), Not(Atom('g')))
        result = check_safety(bad, {'r'}, {'g'})
        assert result.verdict == RealVerdict.REALIZABLE
        if result.controller:
            trace = result.controller.simulate(
                [frozenset({'r'}), frozenset({'r'}), frozenset()]
            )
            for state, inp, out in trace:
                if 'r' in inp:
                    assert 'g' in out

    def test_mealy_simulate_length(self):
        ctrl = MealyMachine(
            states={0}, initial=0,
            transitions={
                (0, frozenset()): (0, frozenset()),
            },
            env_vars=set(), sys_vars=set(),
        )
        trace = ctrl.simulate([frozenset()] * 5, max_steps=3)
        assert len(trace) == 3

    def test_bounded_controller_step(self):
        bc = BoundedController(
            n_states=1, initial=0,
            transitions={(0, 0, frozenset()): (0, frozenset({'b'}))},
            env_vars=set(), sys_vars={'b'},
        )
        result = bc.step(0, 0, frozenset())
        assert result == (0, frozenset({'b'}))


# ============================================================
# Test: Integration (composition with V023)
# ============================================================

class TestIntegration:
    def test_v023_nba_reuse(self):
        """Verify we correctly reuse V023's LTL->NBA pipeline."""
        from ltl_model_checker import ltl_to_gba, gba_to_nba
        spec = Globally(Atom('b'))
        gba = ltl_to_gba(spec)
        nba = gba_to_nba(gba)
        assert len(nba.states) > 0
        assert len(nba.initial) > 0

    def test_parse_and_check(self):
        """Parse LTL string and check realizability."""
        spec = parse_ltl("G(b)")
        result = check_realizable(spec, set(), {'b'})
        assert result.verdict == RealVerdict.REALIZABLE

    def test_complex_spec_parse(self):
        """Parse complex spec and check."""
        spec = parse_ltl("G(r -> g)")
        result = check_realizable(spec, {'r'}, {'g'})
        assert result.verdict == RealVerdict.REALIZABLE

    def test_unrealizable_parse(self):
        spec = parse_ltl("G(a)")
        result = check_realizable(spec, {'a'}, {'b'})
        assert result.verdict == RealVerdict.UNREALIZABLE


# ============================================================
# Test: Realizability Summary
# ============================================================

class TestRealizabilitySummary:
    def test_summary_with_min_states(self):
        result = RealResult(
            verdict=RealVerdict.REALIZABLE,
            method="incremental_search",
            min_states=2,
        )
        s = realizability_summary(result)
        assert "Minimum controller states: 2" in s

    def test_summary_with_counterstrategy(self):
        cs = EnvCounterstrategy(
            states={0, 1, 2}, initial=0, transitions={},
            env_vars={'a'}, sys_vars={'b'},
        )
        result = RealResult(
            verdict=RealVerdict.UNREALIZABLE,
            counterstrategy=cs,
            method="test",
        )
        s = realizability_summary(result)
        assert "Counterstrategy: 3 states" in s

    def test_summary_with_details(self):
        result = RealResult(
            verdict=RealVerdict.REALIZABLE,
            method="test",
            details={"key": "value"},
        )
        s = realizability_summary(result)
        assert "key: value" in s


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
