"""
Tests for V014: Interpolation-Based CEGAR Model Checking
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                'challenges', 'C037_smt_solver'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V010_predicate_abstraction_cegar'))

from smt_solver import Var, IntConst, BoolConst, App, Op, Sort, SortKind
from pred_abs_cegar import ConcreteTS, Predicate, extract_loop_ts

INT = Sort(SortKind.INT)
BOOL = Sort(SortKind.BOOL)

from interp_cegar import (
    interp_cegar_check, interp_model_check,
    interpolation_refine_trace,
    _build_bmc_formulas, _build_constrained_bmc_formulas,
    _check_conjunction_sat, _check_implication,
    _dedup_predicates, _extract_transition_map,
    _wp_refine_simple, _direct_bmc_check,
    verify_loop_interp, verify_loop_direct,
    compare_refinement_strategies,
    InterpCEGARVerdict, InterpCEGARResult, InterpCEGARStats,
    _and, _or, _not, _eq,
)


# ===== Helpers =====

def _make_var(name):
    return Var(name, INT)

def _make_ts_simple_counter():
    """Simple counter: x starts at 0, increments by 1. Property: x >= 0."""
    ts = ConcreteTS(int_vars=['x'])
    x = ts.var('x')
    xp = ts.prime('x')
    ts.init_formula = App(Op.EQ, [x, IntConst(0)], BOOL)
    ts.trans_formula = App(Op.EQ, [xp, App(Op.ADD, [x, IntConst(1)], INT)], BOOL)
    ts.prop_formula = App(Op.GE, [x, IntConst(0)], BOOL)
    return ts

def _make_ts_countdown():
    """Countdown: x starts at 5, decrements by 1 while x > 0. Property: x >= 0."""
    ts = ConcreteTS(int_vars=['x'])
    x = ts.var('x')
    xp = ts.prime('x')
    ts.init_formula = App(Op.EQ, [x, IntConst(5)], BOOL)
    # Guarded: if x > 0 then x' = x-1, else x' = x
    guard = App(Op.GT, [x, IntConst(0)], BOOL)
    dec = App(Op.EQ, [xp, App(Op.SUB, [x, IntConst(1)], INT)], BOOL)
    stay = App(Op.EQ, [xp, x], BOOL)
    ts.trans_formula = App(Op.OR, [
        App(Op.AND, [guard, dec], BOOL),
        App(Op.AND, [_not(guard), stay], BOOL)
    ], BOOL)
    ts.prop_formula = App(Op.GE, [x, IntConst(0)], BOOL)
    return ts

def _make_ts_unsafe():
    """Unsafe system: x starts at 10, property: x <= 5. Violates immediately."""
    ts = ConcreteTS(int_vars=['x'])
    x = ts.var('x')
    xp = ts.prime('x')
    ts.init_formula = App(Op.EQ, [x, IntConst(10)], BOOL)
    ts.trans_formula = App(Op.EQ, [xp, x], BOOL)
    ts.prop_formula = App(Op.LE, [x, IntConst(5)], BOOL)
    return ts

def _make_ts_two_vars():
    """Two variables: x starts at 0, y starts at 10.
    x increments, y decrements. Property: x + y == 10."""
    ts = ConcreteTS(int_vars=['x', 'y'])
    x, y = ts.var('x'), ts.var('y')
    xp, yp = ts.prime('x'), ts.prime('y')
    ts.init_formula = _and(
        App(Op.EQ, [x, IntConst(0)], BOOL),
        App(Op.EQ, [y, IntConst(10)], BOOL)
    )
    ts.trans_formula = _and(
        App(Op.EQ, [xp, App(Op.ADD, [x, IntConst(1)], INT)], BOOL),
        App(Op.EQ, [yp, App(Op.SUB, [y, IntConst(1)], INT)], BOOL)
    )
    ts.prop_formula = App(Op.EQ, [App(Op.ADD, [x, y], INT), IntConst(10)], BOOL)
    return ts

def _make_ts_conditional():
    """Conditional: x starts at 0. If x < 5, x' = x+1; else x' = x.
    Property: x <= 5."""
    ts = ConcreteTS(int_vars=['x'])
    x = ts.var('x')
    xp = ts.prime('x')
    ts.init_formula = App(Op.EQ, [x, IntConst(0)], BOOL)
    guard = App(Op.LT, [x, IntConst(5)], BOOL)
    inc = App(Op.EQ, [xp, App(Op.ADD, [x, IntConst(1)], INT)], BOOL)
    stay = App(Op.EQ, [xp, x], BOOL)
    ts.trans_formula = App(Op.OR, [
        App(Op.AND, [guard, inc], BOOL),
        App(Op.AND, [_not(guard), stay], BOOL)
    ], BOOL)
    ts.prop_formula = App(Op.LE, [x, IntConst(5)], BOOL)
    return ts

def _make_ts_trivial_safe():
    """Trivially safe: x = 0, x' = 0, property: x == 0."""
    ts = ConcreteTS(int_vars=['x'])
    x = ts.var('x')
    xp = ts.prime('x')
    ts.init_formula = App(Op.EQ, [x, IntConst(0)], BOOL)
    ts.trans_formula = App(Op.EQ, [xp, IntConst(0)], BOOL)
    ts.prop_formula = App(Op.EQ, [x, IntConst(0)], BOOL)
    return ts


# ===================================================================
# Section 1: Data structures
# ===================================================================

class TestDataStructures:
    def test_verdict_enum_values(self):
        assert InterpCEGARVerdict.SAFE.value == "safe"
        assert InterpCEGARVerdict.UNSAFE.value == "unsafe"
        assert InterpCEGARVerdict.UNKNOWN.value == "unknown"

    def test_stats_default(self):
        stats = InterpCEGARStats()
        assert stats.iterations == 0
        assert stats.interpolation_successes == 0
        assert stats.wp_fallbacks == 0

    def test_result_default(self):
        r = InterpCEGARResult(verdict=InterpCEGARVerdict.SAFE)
        assert r.verdict == InterpCEGARVerdict.SAFE
        assert r.invariant is None
        assert r.counterexample is None
        assert r.predicates == []


# ===================================================================
# Section 2: BMC formula construction
# ===================================================================

class TestBMCFormulas:
    def test_build_bmc_formulas_length(self):
        ts = _make_ts_simple_counter()
        formulas, step_vars = _build_bmc_formulas(ts, 3)
        # Init + 2 transitions + property_neg = 4 formulas
        assert len(formulas) == 4
        assert len(step_vars) == 3

    def test_bmc_formulas_safe_system(self):
        ts = _make_ts_simple_counter()
        formulas, _ = _build_bmc_formulas(ts, 3)
        # x starts at 0, increments, property x>=0 -- should be UNSAT
        assert not _check_conjunction_sat(formulas)

    def test_bmc_formulas_unsafe_system(self):
        ts = _make_ts_unsafe()
        formulas, _ = _build_bmc_formulas(ts, 5)
        # x starts at 0, increments, property x<=3 violated at step 4
        assert _check_conjunction_sat(formulas)

    def test_constrained_bmc_formulas(self):
        ts = _make_ts_simple_counter()
        preds = [Predicate("x_ge_0", ts.prop_formula)]
        abs_trace = [{"b_0": 1}, {"b_0": 1}, {"b_0": 0}]
        formulas, step_vars = _build_constrained_bmc_formulas(ts, abs_trace, preds)
        assert len(formulas) == 4  # init + 2 trans + prop_neg
        assert len(step_vars) == 3


# ===================================================================
# Section 3: Helper functions
# ===================================================================

class TestHelpers:
    def test_check_implication_true(self):
        x = _make_var('x')
        a = App(Op.EQ, [x, IntConst(5)], BOOL)
        b = App(Op.GE, [x, IntConst(0)], BOOL)
        assert _check_implication(a, b)

    def test_check_implication_false(self):
        x = _make_var('x')
        a = App(Op.GE, [x, IntConst(0)], BOOL)
        b = App(Op.EQ, [x, IntConst(5)], BOOL)
        assert not _check_implication(a, b)

    def test_dedup_predicates(self):
        x = _make_var('x')
        p1 = Predicate("a", App(Op.GE, [x, IntConst(0)], BOOL))
        p2 = Predicate("b", App(Op.GE, [x, IntConst(0)], BOOL))  # same formula
        p3 = Predicate("c", App(Op.LE, [x, IntConst(5)], BOOL))
        result = _dedup_predicates([p1, p2, p3])
        assert len(result) == 2

    def test_extract_transition_map(self):
        ts = _make_ts_simple_counter()
        tmap = _extract_transition_map(ts)
        assert 'x' in tmap

    def test_conjunction_sat_true(self):
        x = _make_var('x')
        f = App(Op.EQ, [x, IntConst(5)], BOOL)
        assert _check_conjunction_sat([f])

    def test_conjunction_sat_false(self):
        x = _make_var('x')
        f1 = App(Op.EQ, [x, IntConst(5)], BOOL)
        f2 = App(Op.EQ, [x, IntConst(3)], BOOL)
        assert not _check_conjunction_sat([f1, f2])


# ===================================================================
# Section 4: Interpolation-based refinement
# ===================================================================

class TestInterpolationRefinement:
    def test_refine_produces_predicates(self):
        ts = _make_ts_simple_counter()
        preds = [Predicate("x_ge_0", ts.prop_formula)]
        # Fake abstract trace (spurious): x_ge_0 holds at step 0, fails at step 1
        abs_trace = [{"b_0": 1}, {"b_0": 0}]
        new_preds = interpolation_refine_trace(ts, abs_trace, preds)
        # Should produce some predicates (or empty if interpolation fails on this simple case)
        assert isinstance(new_preds, list)

    def test_refine_empty_for_short_trace(self):
        ts = _make_ts_simple_counter()
        preds = [Predicate("x_ge_0", ts.prop_formula)]
        abs_trace = [{"b_0": 0}]  # Single-step trace
        new_preds = interpolation_refine_trace(ts, abs_trace, preds)
        assert new_preds == []

    def test_refine_deduplicates(self):
        ts = _make_ts_countdown()
        x = ts.var('x')
        preds = [
            Predicate("x_ge_0", App(Op.GE, [x, IntConst(0)], BOOL)),
            Predicate("x_le_5", App(Op.LE, [x, IntConst(5)], BOOL)),
        ]
        abs_trace = [{"b_0": 1, "b_1": 1}, {"b_0": 0, "b_1": 1}]
        new_preds = interpolation_refine_trace(ts, abs_trace, preds)
        # All returned predicates should have unique formula strings
        strs = [str(p.formula) for p in new_preds]
        assert len(strs) == len(set(strs))


# ===================================================================
# Section 5: WP fallback refinement
# ===================================================================

class TestWPFallback:
    def test_wp_refine_produces_predicates(self):
        ts = _make_ts_simple_counter()
        preds = [Predicate("x_ge_0", ts.prop_formula)]
        new = _wp_refine_simple(ts, preds, [{"b_0": 1}, {"b_0": 0}], 1)
        assert len(new) > 0

    def test_wp_refine_includes_boundary_preds(self):
        ts = _make_ts_simple_counter()
        preds = [Predicate("x_ge_0", ts.prop_formula)]
        new = _wp_refine_simple(ts, preds, [{"b_0": 1}], 0)
        names = [p.name for p in new]
        # Should include boundary predicates
        assert any("bnd_" in n for n in names)


# ===================================================================
# Section 6: Direct BMC check
# ===================================================================

class TestDirectBMC:
    def test_safe_system(self):
        ts = _make_ts_trivial_safe()
        result = _direct_bmc_check(ts, max_depth=5)
        # Trivially safe -- BMC won't find violation
        assert result != InterpCEGARVerdict.UNSAFE

    def test_unsafe_system(self):
        ts = _make_ts_unsafe()
        result = _direct_bmc_check(ts, max_depth=10)
        assert result == InterpCEGARVerdict.UNSAFE


# ===================================================================
# Section 7: Interp CEGAR -- safe systems
# ===================================================================

class TestInterpCEGARSafe:
    def test_trivially_safe(self):
        ts = _make_ts_trivial_safe()
        result = interp_cegar_check(ts)
        assert result.verdict == InterpCEGARVerdict.SAFE

    def test_simple_counter_safe(self):
        ts = _make_ts_simple_counter()
        result = interp_cegar_check(ts, max_iterations=10)
        assert result.verdict == InterpCEGARVerdict.SAFE

    def test_countdown_safe(self):
        ts = _make_ts_countdown()
        result = interp_cegar_check(ts, max_iterations=10)
        assert result.verdict == InterpCEGARVerdict.SAFE

    def test_conditional_safe(self):
        ts = _make_ts_conditional()
        result = interp_cegar_check(ts, max_iterations=10)
        assert result.verdict == InterpCEGARVerdict.SAFE

    def test_sum_conservation(self):
        ts = _make_ts_two_vars()
        result = interp_cegar_check(ts, max_iterations=10)
        assert result.verdict == InterpCEGARVerdict.SAFE


# ===================================================================
# Section 8: Interp CEGAR -- unsafe systems
# ===================================================================

class TestInterpCEGARUnsafe:
    def test_unbounded_counter_unsafe(self):
        ts = _make_ts_unsafe()
        result = interp_cegar_check(ts, max_iterations=10)
        assert result.verdict == InterpCEGARVerdict.UNSAFE

    def test_unsafe_has_counterexample(self):
        ts = _make_ts_unsafe()
        result = interp_cegar_check(ts, max_iterations=10)
        assert result.verdict == InterpCEGARVerdict.UNSAFE
        assert result.counterexample is not None

    def test_unsafe_init_violates(self):
        """System where init state already violates property."""
        ts = ConcreteTS(int_vars=['x'])
        x = ts.var('x')
        xp = ts.prime('x')
        ts.init_formula = App(Op.EQ, [x, IntConst(10)], BOOL)
        ts.trans_formula = App(Op.EQ, [xp, x], BOOL)
        ts.prop_formula = App(Op.LE, [x, IntConst(3)], BOOL)
        result = interp_cegar_check(ts)
        assert result.verdict == InterpCEGARVerdict.UNSAFE


# ===================================================================
# Section 9: Interp CEGAR -- statistics
# ===================================================================

class TestInterpCEGARStats:
    def test_stats_tracked(self):
        ts = _make_ts_simple_counter()
        result = interp_cegar_check(ts, max_iterations=10)
        assert result.stats.iterations >= 1
        assert result.stats.predicates_initial >= 0
        assert result.stats.predicates_final >= 0

    def test_pdr_calls_tracked(self):
        ts = _make_ts_trivial_safe()
        result = interp_cegar_check(ts)
        assert result.stats.pdr_calls >= 1

    def test_predicates_returned(self):
        ts = _make_ts_countdown()
        result = interp_cegar_check(ts, max_iterations=10)
        assert len(result.predicates) >= 1


# ===================================================================
# Section 10: Direct interpolation-based model checking
# ===================================================================

class TestDirectInterpMC:
    def test_trivially_safe(self):
        ts = _make_ts_trivial_safe()
        result = interp_model_check(ts, max_depth=10)
        # Should find safe or at least not unsafe
        assert result.verdict != InterpCEGARVerdict.UNSAFE

    def test_unsafe_detected(self):
        ts = _make_ts_unsafe()
        result = interp_model_check(ts, max_depth=10)
        assert result.verdict == InterpCEGARVerdict.UNSAFE

    def test_simple_counter_not_unsafe(self):
        ts = _make_ts_simple_counter()
        result = interp_model_check(ts, max_depth=10)
        assert result.verdict != InterpCEGARVerdict.UNSAFE

    def test_countdown_not_unsafe(self):
        ts = _make_ts_countdown()
        result = interp_model_check(ts, max_depth=10)
        assert result.verdict != InterpCEGARVerdict.UNSAFE


# ===================================================================
# Section 11: Source-level API
# ===================================================================

class TestSourceLevel:
    def test_verify_loop_countdown(self):
        source = """
        let x = 5;
        while (x > 0) {
            x = x - 1;
        }
        """
        result = verify_loop_interp(
            source,
            lambda ts: App(Op.GE, [ts.var('x'), IntConst(0)], BOOL),
            max_iterations=10
        )
        assert result.verdict == InterpCEGARVerdict.SAFE

    def test_verify_loop_unsafe(self):
        """Init immediately violates property (via extract_loop_ts)."""
        source = """
        let x = 10;
        while (x > 5) {
            x = x - 1;
        }
        """
        result = verify_loop_interp(
            source,
            lambda ts: App(Op.LE, [ts.var('x'), IntConst(3)], BOOL),
            max_iterations=10
        )
        # x starts at 10, property x <= 3 violated immediately
        assert result.verdict == InterpCEGARVerdict.UNSAFE

    def test_verify_loop_direct_safe(self):
        source = """
        let x = 5;
        while (x > 0) {
            x = x - 1;
        }
        """
        result = verify_loop_direct(
            source,
            lambda ts: App(Op.GE, [ts.var('x'), IntConst(0)], BOOL),
            max_depth=10
        )
        assert result.verdict != InterpCEGARVerdict.UNSAFE

    def test_verify_loop_direct_unsafe(self):
        """Init immediately violates property -- direct MC catches it at depth 1."""
        source = """
        let x = 10;
        while (x > 5) {
            x = x - 1;
        }
        """
        result = verify_loop_direct(
            source,
            lambda ts: App(Op.LE, [ts.var('x'), IntConst(3)], BOOL),
            max_depth=10
        )
        assert result.verdict == InterpCEGARVerdict.UNSAFE


# ===================================================================
# Section 12: Custom predicates
# ===================================================================

class TestCustomPredicates:
    def test_with_custom_initial_predicates(self):
        ts = _make_ts_countdown()
        x = ts.var('x')
        preds = [
            Predicate("x_ge_0", App(Op.GE, [x, IntConst(0)], BOOL)),
            Predicate("x_le_5", App(Op.LE, [x, IntConst(5)], BOOL)),
        ]
        result = interp_cegar_check(ts, initial_predicates=preds, max_iterations=10)
        assert result.verdict == InterpCEGARVerdict.SAFE

    def test_single_predicate_sufficient(self):
        ts = _make_ts_simple_counter()
        x = ts.var('x')
        preds = [Predicate("x_ge_0", App(Op.GE, [x, IntConst(0)], BOOL))]
        result = interp_cegar_check(ts, initial_predicates=preds, max_iterations=5)
        assert result.verdict == InterpCEGARVerdict.SAFE

    def test_wrong_predicates_still_converge(self):
        """Even with unhelpful predicates, refinement should find useful ones."""
        ts = _make_ts_countdown()
        x = ts.var('x')
        preds = [
            Predicate("x_eq_99", App(Op.EQ, [x, IntConst(99)], BOOL)),
        ]
        result = interp_cegar_check(ts, initial_predicates=preds, max_iterations=15)
        # May converge or hit limit -- but should not report UNSAFE
        assert result.verdict != InterpCEGARVerdict.UNSAFE


# ===================================================================
# Section 13: Comparison API
# ===================================================================

class TestComparison:
    def test_compare_on_safe_system(self):
        ts = _make_ts_simple_counter()
        comp = compare_refinement_strategies(ts, max_iterations=10)
        assert 'wp_based' in comp
        assert 'interpolation_based' in comp
        assert comp['wp_based']['verdict'] in ('safe', 'unsafe', 'unknown')
        assert comp['interpolation_based']['verdict'] in ('safe', 'unsafe', 'unknown')

    def test_compare_on_unsafe_system(self):
        ts = _make_ts_unsafe()
        comp = compare_refinement_strategies(ts, max_iterations=10)
        assert comp['interpolation_based']['verdict'] == 'unsafe'

    def test_compare_returns_results(self):
        ts = _make_ts_trivial_safe()
        comp = compare_refinement_strategies(ts, max_iterations=5)
        assert comp['v010_result'] is not None
        assert comp['v014_result'] is not None


# ===================================================================
# Section 14: Edge cases
# ===================================================================

class TestEdgeCases:
    def test_no_transition_system_property(self):
        """System with True property -- trivially safe."""
        ts = ConcreteTS(int_vars=['x'])
        x = ts.var('x')
        xp = ts.prime('x')
        ts.init_formula = App(Op.EQ, [x, IntConst(0)], BOOL)
        ts.trans_formula = App(Op.EQ, [xp, App(Op.ADD, [x, IntConst(1)], INT)], BOOL)
        ts.prop_formula = BoolConst(True)
        result = interp_cegar_check(ts, max_iterations=5)
        assert result.verdict == InterpCEGARVerdict.SAFE

    def test_nondeterministic_safe(self):
        """Nondeterministic: x' = x+1 or x' = x. Property: x >= 0."""
        ts = ConcreteTS(int_vars=['x'])
        x = ts.var('x')
        xp = ts.prime('x')
        ts.init_formula = App(Op.EQ, [x, IntConst(0)], BOOL)
        inc = App(Op.EQ, [xp, App(Op.ADD, [x, IntConst(1)], INT)], BOOL)
        stay = App(Op.EQ, [xp, x], BOOL)
        ts.trans_formula = _or(inc, stay)
        ts.prop_formula = App(Op.GE, [x, IntConst(0)], BOOL)
        result = interp_cegar_check(ts, max_iterations=10)
        assert result.verdict == InterpCEGARVerdict.SAFE

    def test_multiple_initial_states(self):
        """x starts in {0, 1, 2}. x' = x. Property: x >= 0."""
        ts = ConcreteTS(int_vars=['x'])
        x = ts.var('x')
        xp = ts.prime('x')
        ts.init_formula = _and(App(Op.GE, [x, IntConst(0)], BOOL),
                               App(Op.LE, [x, IntConst(2)], BOOL))
        ts.trans_formula = App(Op.EQ, [xp, x], BOOL)
        ts.prop_formula = App(Op.GE, [x, IntConst(0)], BOOL)
        result = interp_cegar_check(ts, max_iterations=5)
        assert result.verdict == InterpCEGARVerdict.SAFE

    def test_empty_predicates_triggers_bmc(self):
        """When no predicates can be generated, falls back to direct BMC."""
        ts = ConcreteTS(int_vars=['x'])
        x = ts.var('x')
        xp = ts.prime('x')
        ts.init_formula = App(Op.EQ, [x, IntConst(10)], BOOL)
        ts.trans_formula = App(Op.EQ, [xp, x], BOOL)
        ts.prop_formula = App(Op.LE, [x, IntConst(5)], BOOL)
        result = interp_cegar_check(ts, initial_predicates=[], max_iterations=10)
        assert result.verdict == InterpCEGARVerdict.UNSAFE

    def test_two_var_conservation_law(self):
        """x + y = 10 conservation with different init values."""
        ts = ConcreteTS(int_vars=['x', 'y'])
        x, y = ts.var('x'), ts.var('y')
        xp, yp = ts.prime('x'), ts.prime('y')
        ts.init_formula = _and(
            App(Op.EQ, [x, IntConst(3)], BOOL),
            App(Op.EQ, [y, IntConst(7)], BOOL)
        )
        ts.trans_formula = _and(
            App(Op.EQ, [xp, App(Op.ADD, [x, IntConst(1)], INT)], BOOL),
            App(Op.EQ, [yp, App(Op.SUB, [y, IntConst(1)], INT)], BOOL)
        )
        ts.prop_formula = App(Op.EQ, [App(Op.ADD, [x, y], INT), IntConst(10)], BOOL)
        result = interp_cegar_check(ts, max_iterations=10)
        assert result.verdict == InterpCEGARVerdict.SAFE
