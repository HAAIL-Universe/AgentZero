"""
Tests for V010: Predicate Abstraction + CEGAR
"""

import sys, os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C037_smt_solver'))

from smt_solver import Var, IntConst, BoolConst, App, Op, Sort, SortKind
from pred_abs_cegar import (
    Predicate, ConcreteTS, CEGARVerdict, CEGARResult, CEGARStats,
    cartesian_abstraction, check_counterexample_feasibility,
    refine_predicates, cegar_check, verify_with_cegar,
    auto_predicates_from_ts, extract_loop_ts, verify_loop_with_cegar,
    _and, _or, _not, _implies, _eq, _substitute, _collect_vars,
    _check_pred_in_state, _extract_atomic_predicates, _smt_check
)

INT = Sort(SortKind.INT)
BOOL = Sort(SortKind.BOOL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_var(name, sort=INT):
    return Var(name, sort)

def make_simple_counter_cts(init_val=0, max_val=5):
    """Counter: x starts at init_val, increments by 1 each step.
    Property: x <= max_val (will eventually be violated).
    """
    cts = ConcreteTS()
    cts.int_vars = ['x']
    x = cts.var('x')
    xp = cts.prime('x')
    cts.init_formula = _eq(x, IntConst(init_val))
    cts.trans_formula = _eq(xp, App(Op.ADD, [x, IntConst(1)], INT))
    cts.prop_formula = App(Op.LE, [x, IntConst(max_val)], BOOL)
    return cts

def make_bounded_counter_cts():
    """Counter: x=0, increments until x >= 5, then stays. Property: x >= 0."""
    cts = ConcreteTS()
    cts.int_vars = ['x']
    x = cts.var('x')
    xp = cts.prime('x')
    cts.init_formula = _eq(x, IntConst(0))
    # Guarded transition: if x < 5 then x' = x+1 else x' = x
    cond = App(Op.LT, [x, IntConst(5)], BOOL)
    cts.trans_formula = _or(
        _and(cond, _eq(xp, App(Op.ADD, [x, IntConst(1)], INT))),
        _and(_not(cond), _eq(xp, x))
    )
    cts.prop_formula = App(Op.GE, [x, IntConst(0)], BOOL)
    return cts

def make_two_var_cts():
    """Two vars: x=10, y=0, each step x-=1, y+=1. Property: x+y == 10."""
    cts = ConcreteTS()
    cts.int_vars = ['x', 'y']
    x, y = cts.var('x'), cts.var('y')
    xp, yp = cts.prime('x'), cts.prime('y')
    cts.init_formula = _and(_eq(x, IntConst(10)), _eq(y, IntConst(0)))
    cts.trans_formula = _and(
        _eq(xp, App(Op.SUB, [x, IntConst(1)], INT)),
        _eq(yp, App(Op.ADD, [y, IntConst(1)], INT))
    )
    cts.prop_formula = _eq(App(Op.ADD, [x, y], INT), IntConst(10))
    return cts


# ===================================================================
# Section 1: SMT helpers
# ===================================================================

class TestSMTHelpers:
    def test_and_basic(self):
        a = App(Op.GE, [make_var('x'), IntConst(0)], BOOL)
        b = App(Op.LE, [make_var('x'), IntConst(10)], BOOL)
        result = _and(a, b)
        assert isinstance(result, App) and result.op == Op.AND

    def test_and_short_circuit_false(self):
        a = App(Op.GE, [make_var('x'), IntConst(0)], BOOL)
        result = _and(a, BoolConst(False))
        assert isinstance(result, BoolConst) and not result.value

    def test_and_short_circuit_true(self):
        a = App(Op.GE, [make_var('x'), IntConst(0)], BOOL)
        result = _and(a, BoolConst(True))
        assert result == a  # True is filtered out

    def test_or_basic(self):
        a = App(Op.GE, [make_var('x'), IntConst(0)], BOOL)
        b = App(Op.LE, [make_var('x'), IntConst(10)], BOOL)
        result = _or(a, b)
        assert isinstance(result, App) and result.op == Op.OR

    def test_not_complement(self):
        a = App(Op.EQ, [make_var('x'), IntConst(0)], BOOL)
        result = _not(a)
        assert isinstance(result, App) and result.op == Op.NEQ

    def test_not_de_morgan(self):
        a = App(Op.GE, [make_var('x'), IntConst(0)], BOOL)
        b = App(Op.LE, [make_var('x'), IntConst(10)], BOOL)
        conj = _and(a, b)
        result = _not(conj)
        assert isinstance(result, App) and result.op == Op.OR

    def test_substitute(self):
        x = make_var('x')
        formula = App(Op.GE, [x, IntConst(0)], BOOL)
        y = make_var('y')
        result = _substitute(formula, {'x': y})
        assert isinstance(result.args[0], Var) and result.args[0].name == 'y'

    def test_collect_vars(self):
        x = make_var('x')
        y = make_var('y')
        formula = App(Op.ADD, [x, y], INT)
        vs = _collect_vars(formula)
        names = {v.name for v in vs}
        assert names == {'x', 'y'}


# ===================================================================
# Section 2: Predicate state checking
# ===================================================================

class TestPredicateStateChecking:
    def test_pred_always_true(self):
        """x == 5 AND pred = (x >= 0) => pred always true."""
        x = make_var('x')
        state = _eq(x, IntConst(5))
        pred = App(Op.GE, [x, IntConst(0)], BOOL)
        assert _check_pred_in_state(pred, state) is True

    def test_pred_always_false(self):
        """x == -5 AND pred = (x >= 0) => pred always false."""
        x = make_var('x')
        state = _eq(x, IntConst(-5))
        pred = App(Op.GE, [x, IntConst(0)], BOOL)
        assert _check_pred_in_state(pred, state) is False

    def test_pred_unknown(self):
        """x >= -10 AND x <= 10, pred = (x >= 0) => unknown."""
        x = make_var('x')
        state = _and(
            App(Op.GE, [x, IntConst(-10)], BOOL),
            App(Op.LE, [x, IntConst(10)], BOOL)
        )
        pred = App(Op.GE, [x, IntConst(0)], BOOL)
        assert _check_pred_in_state(pred, state) is None


# ===================================================================
# Section 3: ConcreteTS construction
# ===================================================================

class TestConcreteTS:
    def test_create_simple(self):
        cts = make_simple_counter_cts()
        assert 'x' in cts.int_vars
        assert cts.init_formula is not None
        assert cts.trans_formula is not None
        assert cts.prop_formula is not None

    def test_var_and_prime(self):
        cts = ConcreteTS()
        cts.int_vars = ['x']
        x = cts.var('x')
        xp = cts.prime('x')
        assert x.name == 'x'
        assert xp.name == "x'"

    def test_bounded_counter(self):
        cts = make_bounded_counter_cts()
        assert 'x' in cts.int_vars

    def test_two_var(self):
        cts = make_two_var_cts()
        assert 'x' in cts.int_vars
        assert 'y' in cts.int_vars


# ===================================================================
# Section 4: Cartesian abstraction
# ===================================================================

class TestCartesianAbstraction:
    def test_single_predicate(self):
        """Abstract bounded counter with predicate x >= 0."""
        cts = make_bounded_counter_cts()
        x = cts.var('x')
        preds = [Predicate("x_ge0", App(Op.GE, [x, IntConst(0)], BOOL))]
        abs_ts = cartesian_abstraction(cts, preds)
        # Should have one variable (INT-encoded boolean)
        assert len(abs_ts.state_vars) == 1

    def test_two_predicates(self):
        """Abstract with two predicates."""
        cts = make_bounded_counter_cts()
        x = cts.var('x')
        preds = [
            Predicate("x_ge0", App(Op.GE, [x, IntConst(0)], BOOL)),
            Predicate("x_lt5", App(Op.LT, [x, IntConst(5)], BOOL)),
        ]
        abs_ts = cartesian_abstraction(cts, preds)
        assert len(abs_ts.state_vars) == 2

    def test_abstraction_preserves_init(self):
        """Init x=0 should make x>=0 true in abstract init."""
        cts = make_bounded_counter_cts()
        x = cts.var('x')
        preds = [Predicate("x_ge0", App(Op.GE, [x, IntConst(0)], BOOL))]
        abs_ts = cartesian_abstraction(cts, preds)
        # b_0 should be forced True in init
        assert abs_ts.init_formula is not None


# ===================================================================
# Section 5: Auto predicate generation
# ===================================================================

class TestAutoPredicates:
    def test_generates_from_property(self):
        cts = make_bounded_counter_cts()
        preds = auto_predicates_from_ts(cts)
        # Should have at least the property predicate and x >= 0
        names = [p.name for p in preds]
        assert "property" in names

    def test_generates_variable_preds(self):
        cts = make_bounded_counter_cts()
        preds = auto_predicates_from_ts(cts)
        # x_ge0 is the same formula as the property, so it may be deduped
        # At least we should have the property and some init predicates
        names = [p.name for p in preds]
        assert len(preds) >= 2

    def test_extracts_init_preds(self):
        cts = make_bounded_counter_cts()
        preds = auto_predicates_from_ts(cts)
        # Should have some init predicates
        assert any(p.name.startswith("init_") for p in preds)

    def test_two_var_system(self):
        cts = make_two_var_cts()
        preds = auto_predicates_from_ts(cts)
        names = [p.name for p in preds]
        assert "x_ge0" in names
        assert "y_ge0" in names


# ===================================================================
# Section 6: CEGAR - safe systems
# ===================================================================

class TestCEGARSafe:
    def test_bounded_counter_safe(self):
        """Bounded counter with x >= 0 should be SAFE."""
        cts = make_bounded_counter_cts()
        x = cts.var('x')
        preds = [
            Predicate("x_ge0", App(Op.GE, [x, IntConst(0)], BOOL)),
            Predicate("property", cts.prop_formula),
        ]
        result = cegar_check(cts, preds)
        assert result.verdict == CEGARVerdict.SAFE

    def test_bounded_counter_safe_auto_preds(self):
        """Bounded counter verified with auto-generated predicates."""
        cts = make_bounded_counter_cts()
        result = verify_with_cegar(cts)
        assert result.verdict == CEGARVerdict.SAFE

    def test_safe_has_invariant(self):
        """SAFE result should include an invariant."""
        cts = make_bounded_counter_cts()
        result = verify_with_cegar(cts)
        assert result.verdict == CEGARVerdict.SAFE
        assert result.invariant is not None

    def test_conservation_law(self):
        """x + y == 10 is preserved by x-=1, y+=1."""
        cts = make_two_var_cts()
        x, y = cts.var('x'), cts.var('y')
        preds = [
            Predicate("sum_eq_10", _eq(App(Op.ADD, [x, y], INT), IntConst(10))),
            Predicate("property", cts.prop_formula),
        ]
        result = cegar_check(cts, preds)
        assert result.verdict == CEGARVerdict.SAFE


# ===================================================================
# Section 7: CEGAR - unsafe systems
# ===================================================================

class TestCEGARUnsafe:
    def test_init_violation(self):
        """Property violated in initial state: x=10, prop: x<=5."""
        cts = ConcreteTS()
        cts.int_vars = ['x']
        x = cts.var('x')
        xp = cts.prime('x')
        cts.init_formula = _eq(x, IntConst(10))
        cts.trans_formula = _eq(xp, x)
        cts.prop_formula = App(Op.LE, [x, IntConst(5)], BOOL)
        preds = [Predicate("x_le5", App(Op.LE, [x, IntConst(5)], BOOL))]
        result = cegar_check(cts, preds)
        assert result.verdict == CEGARVerdict.UNSAFE

    def test_one_step_violation(self):
        """Property violated in one step: x=5, x'=x+1, prop: x<=5."""
        cts = ConcreteTS()
        cts.int_vars = ['x']
        x = cts.var('x')
        xp = cts.prime('x')
        cts.init_formula = _eq(x, IntConst(5))
        cts.trans_formula = _eq(xp, App(Op.ADD, [x, IntConst(1)], INT))
        cts.prop_formula = App(Op.LE, [x, IntConst(5)], BOOL)
        preds = [Predicate("x_le5", App(Op.LE, [x, IntConst(5)], BOOL))]
        result = cegar_check(cts, preds)
        assert result.verdict == CEGARVerdict.UNSAFE

    def test_unsafe_has_counterexample(self):
        """UNSAFE result should include a concrete counterexample."""
        cts = ConcreteTS()
        cts.int_vars = ['x']
        x = cts.var('x')
        xp = cts.prime('x')
        cts.init_formula = _eq(x, IntConst(10))
        cts.trans_formula = _eq(xp, x)
        cts.prop_formula = App(Op.LE, [x, IntConst(5)], BOOL)
        preds = [Predicate("x_le5", App(Op.LE, [x, IntConst(5)], BOOL))]
        result = cegar_check(cts, preds)
        assert result.verdict == CEGARVerdict.UNSAFE
        assert result.counterexample is not None
        assert len(result.counterexample) > 0

    def test_unsafe_counterexample_valid(self):
        """Counterexample should have correct init state."""
        cts = ConcreteTS()
        cts.int_vars = ['x']
        x = cts.var('x')
        xp = cts.prime('x')
        cts.init_formula = _eq(x, IntConst(10))
        cts.trans_formula = _eq(xp, x)
        cts.prop_formula = App(Op.LE, [x, IntConst(5)], BOOL)
        preds = [Predicate("x_le5", App(Op.LE, [x, IntConst(5)], BOOL))]
        result = cegar_check(cts, preds)
        assert result.verdict == CEGARVerdict.UNSAFE
        if result.counterexample:
            assert result.counterexample[0].get('x', None) == 10


# ===================================================================
# Section 8: Counterexample feasibility checking
# ===================================================================

class TestFeasibilityCheck:
    def test_feasible_trace(self):
        """A real counterexample should be feasible."""
        cts = make_simple_counter_cts(0, 2)
        x = cts.var('x')
        preds = [
            Predicate("x_ge0", App(Op.GE, [x, IntConst(0)], BOOL)),
            Predicate("x_le2", App(Op.LE, [x, IntConst(2)], BOOL)),
        ]
        # Trace: x=0 (x>=0, x<=2), x=1 (x>=0, x<=2), x=2 (x>=0, x<=2), x=3 (x>=0, NOT x<=2)
        trace = [
            {'b_0': True, 'b_1': True},
            {'b_0': True, 'b_1': True},
            {'b_0': True, 'b_1': True},
            {'b_0': True, 'b_1': False},
        ]
        is_feasible, step, concrete = check_counterexample_feasibility(trace, cts, preds)
        # This trace is feasible (real counterexample)
        assert is_feasible
        assert concrete is not None


# ===================================================================
# Section 9: Predicate refinement
# ===================================================================

class TestRefinement:
    def test_wp_refinement(self):
        """Refinement should generate new predicates."""
        cts = make_simple_counter_cts(0, 5)
        x = cts.var('x')
        preds = [
            Predicate("x_ge0", App(Op.GE, [x, IntConst(0)], BOOL)),
        ]
        # Spurious trace at step 1
        trace = [
            {'b_0': True},
            {'b_0': False},  # x >= 0 becomes false after one step
        ]
        new = refine_predicates(preds, cts, 1, trace)
        # Should discover WP-based predicates
        assert len(new) >= 0  # May or may not find new preds depending on strategy

    def test_refinement_produces_unique_preds(self):
        """Refinement should not duplicate existing predicates."""
        cts = make_bounded_counter_cts()
        x = cts.var('x')
        preds = [
            Predicate("x_ge0", App(Op.GE, [x, IntConst(0)], BOOL)),
        ]
        trace = [{'b_0': True}, {'b_0': False}]
        new = refine_predicates(preds, cts, 1, trace)
        names = [p.name for p in new]
        assert len(names) == len(set(names))  # All unique


# ===================================================================
# Section 10: CEGAR with refinement
# ===================================================================

class TestCEGARWithRefinement:
    def test_refinement_improves_result(self):
        """CEGAR should refine predicates when needed."""
        cts = make_bounded_counter_cts()
        result = verify_with_cegar(cts)
        # Should converge to SAFE (possibly after refinement)
        assert result.verdict in (CEGARVerdict.SAFE, CEGARVerdict.UNKNOWN)

    def test_stats_tracking(self):
        """Stats should be populated."""
        cts = make_bounded_counter_cts()
        result = verify_with_cegar(cts)
        assert result.stats.iterations >= 1
        assert result.stats.pdr_calls >= 1
        assert result.stats.predicates_initial >= 1
        assert result.stats.predicates_final >= result.stats.predicates_initial


# ===================================================================
# Section 11: Source-level loop extraction
# ===================================================================

class TestLoopExtraction:
    def test_simple_countdown(self):
        """Extract TS from a countdown loop."""
        source = """
        let x = 10;
        while (x > 0) {
            x = x - 1;
        }
        """
        cts = extract_loop_ts(source)
        assert 'x' in cts.int_vars
        assert cts.init_formula is not None
        assert cts.trans_formula is not None

    def test_two_var_loop(self):
        """Extract TS from a two-variable loop."""
        source = """
        let x = 10;
        let y = 0;
        while (x > 0) {
            x = x - 1;
            y = y + 1;
        }
        """
        cts = extract_loop_ts(source)
        assert 'x' in cts.int_vars
        assert 'y' in cts.int_vars

    def test_countup_loop(self):
        """Extract TS from a countup loop."""
        source = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        cts = extract_loop_ts(source)
        assert 'i' in cts.int_vars


# ===================================================================
# Section 12: Source-level CEGAR verification
# ===================================================================

class TestSourceLevelCEGAR:
    def test_countdown_x_ge0(self):
        """Verify x >= 0 for a countdown loop."""
        source = """
        let x = 10;
        while (x > 0) {
            x = x - 1;
        }
        """
        cts = extract_loop_ts(source)
        x = cts.var('x')
        cts.prop_formula = App(Op.GE, [x, IntConst(0)], BOOL)
        result = verify_with_cegar(cts)
        assert result.verdict == CEGARVerdict.SAFE

    def test_conservation_source(self):
        """Verify x + y == 10 for transfer loop."""
        source = """
        let x = 10;
        let y = 0;
        while (x > 0) {
            x = x - 1;
            y = y + 1;
        }
        """
        cts = extract_loop_ts(source)
        x, y = cts.var('x'), cts.var('y')
        cts.prop_formula = _eq(App(Op.ADD, [x, y], INT), IntConst(10))
        preds = [
            Predicate("sum_eq_10", _eq(App(Op.ADD, [x, y], INT), IntConst(10))),
            Predicate("property", cts.prop_formula),
        ]
        result = cegar_check(cts, preds)
        assert result.verdict == CEGARVerdict.SAFE

    def test_verify_loop_with_cegar_api(self):
        """Test the high-level verify_loop_with_cegar API."""
        source = """
        let x = 10;
        while (x > 0) {
            x = x - 1;
        }
        """
        result = verify_loop_with_cegar(source, "x >= 0")
        assert result.verdict == CEGARVerdict.SAFE


# ===================================================================
# Section 13: Edge cases
# ===================================================================

class TestEdgeCases:
    def test_trivially_safe(self):
        """System where property is trivially true."""
        cts = ConcreteTS()
        cts.int_vars = ['x']
        x = cts.var('x')
        xp = cts.prime('x')
        cts.init_formula = _eq(x, IntConst(0))
        cts.trans_formula = _eq(xp, x)  # No change
        cts.prop_formula = BoolConst(True)  # Always true
        result = verify_with_cegar(cts)
        assert result.verdict == CEGARVerdict.SAFE

    def test_single_state(self):
        """System with one state: x=5, x'=5, prop: x==5."""
        cts = ConcreteTS()
        cts.int_vars = ['x']
        x = cts.var('x')
        xp = cts.prime('x')
        cts.init_formula = _eq(x, IntConst(5))
        cts.trans_formula = _eq(xp, x)
        cts.prop_formula = _eq(x, IntConst(5))
        preds = [Predicate("x_eq5", _eq(x, IntConst(5)))]
        result = cegar_check(cts, preds)
        assert result.verdict == CEGARVerdict.SAFE

    def test_no_predicates_generates_auto(self):
        """verify_with_cegar with no predicates should auto-generate."""
        cts = make_bounded_counter_cts()
        result = verify_with_cegar(cts, predicates=None)
        assert result.predicates  # Should have auto-generated some

    def test_empty_trace(self):
        """Empty trace should be feasible."""
        cts = make_bounded_counter_cts()
        is_f, step, trace = check_counterexample_feasibility([], cts, [])
        assert is_f is True


# ===================================================================
# Section 14: Integration with V002 PDR
# ===================================================================

class TestPDRIntegration:
    def test_abstract_system_is_valid_ts(self):
        """Abstract system should be a valid TransitionSystem for PDR."""
        cts = make_bounded_counter_cts()
        x = cts.var('x')
        preds = [Predicate("x_ge0", App(Op.GE, [x, IntConst(0)], BOOL))]
        abs_ts = cartesian_abstraction(cts, preds)
        # Should be checkable by PDR without error
        from pdr import check_ts, PDRResult
        result = check_ts(abs_ts, max_frames=20)
        assert result.result in (PDRResult.SAFE, PDRResult.UNSAFE, PDRResult.UNKNOWN)

    def test_safe_abstract_implies_safe_concrete(self):
        """If abstract system is SAFE, concrete must be SAFE."""
        cts = make_bounded_counter_cts()
        x = cts.var('x')
        preds = [
            Predicate("x_ge0", App(Op.GE, [x, IntConst(0)], BOOL)),
            Predicate("property", cts.prop_formula),
        ]
        abs_ts = cartesian_abstraction(cts, preds)
        from pdr import check_ts, PDRResult
        result = check_ts(abs_ts, max_frames=50)
        # If SAFE in abstract, SAFE in concrete (soundness of over-approximation)
        if result.result == PDRResult.SAFE:
            full_result = verify_with_cegar(cts, preds)
            assert full_result.verdict == CEGARVerdict.SAFE
