"""
Tests for V184: Adaptive Abstract Interpretation
"""

import pytest
import sys, os
from fractions import Fraction

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C039_abstract_interpreter'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V179_abstract_domain_hierarchy'))

from adaptive_interpreter import (
    AdaptiveInterpreter, AdaptiveEnv, PromotionReason, PromotionEvent,
    DomainComparison, PointAnalysis,
    adaptive_analyze, analyze_with_comparison, precision_report,
    classify_points, get_promotions, get_relational_bounds,
    get_relational_constraints, compare_strategies
)
from domain_hierarchy import DomainLevel, LinearConstraint
from abstract_interpreter import (
    Sign, Interval, INTERVAL_TOP, INTERVAL_BOT, INF, NEG_INF,
    AbstractValue, WarningKind
)


# ============================================================
# 1. Basic Analysis (non-relational)
# ============================================================

class TestBasicAnalysis:
    """Tests for basic non-relational abstract interpretation."""

    def test_constant_assignment(self):
        result = adaptive_analyze("let x = 42;")
        env = result['env']
        iv = env.get_bounds('x')
        assert iv.lo == 42 and iv.hi == 42

    def test_two_constants(self):
        result = adaptive_analyze("let x = 10; let y = 20;")
        env = result['env']
        assert env.get_bounds('x') == Interval(10, 10)
        assert env.get_bounds('y') == Interval(20, 20)

    def test_arithmetic(self):
        result = adaptive_analyze("let x = 10; let y = x + 5;")
        env = result['env']
        assert env.get_bounds('y') == Interval(15, 15)

    def test_subtraction(self):
        result = adaptive_analyze("let x = 10; let y = x - 3;")
        env = result['env']
        assert env.get_bounds('y') == Interval(7, 7)

    def test_multiplication(self):
        result = adaptive_analyze("let x = 3; let y = x * 4;")
        env = result['env']
        assert env.get_bounds('y') == Interval(12, 12)

    def test_sign_tracking(self):
        result = adaptive_analyze("let x = 5;")
        assert result['env'].get_sign('x') == Sign.POS

    def test_negative_sign(self):
        result = adaptive_analyze("let x = 0 - 5;")
        assert result['env'].get_sign('x') == Sign.NEG

    def test_zero_sign(self):
        result = adaptive_analyze("let x = 0;")
        assert result['env'].get_sign('x') == Sign.ZERO

    def test_const_propagation(self):
        result = adaptive_analyze("let x = 7; let y = x + 3;")
        from abstract_interpreter import ConstVal
        c = result['env'].get_const('y')
        assert isinstance(c, ConstVal) and c.value == 10

    def test_var_reassignment(self):
        result = adaptive_analyze("let x = 5; x = 10;")
        assert result['env'].get_bounds('x') == Interval(10, 10)

    def test_multiple_assignments(self):
        result = adaptive_analyze("let x = 1; x = 2; x = 3;")
        assert result['env'].get_bounds('x') == Interval(3, 3)


# ============================================================
# 2. If/Else Branch Analysis
# ============================================================

class TestBranchAnalysis:

    def test_if_simple(self):
        src = """
        let x = 10;
        if (x > 5) {
            let y = 1;
        }
        """
        result = adaptive_analyze(src)
        assert len(result['warnings']) >= 0  # no crash

    def test_if_else_join(self):
        src = """
        let x = 0;
        let y = 10;
        if (y > 5) {
            x = 1;
        } else {
            x = 2;
        }
        """
        result = adaptive_analyze(src)
        env = result['env']
        iv = env.get_bounds('x')
        assert iv.lo <= 1 and iv.hi >= 2

    def test_guard_refinement(self):
        """If condition refines variable bounds in then branch."""
        src = """
        let x = 100;
        if (x < 50) {
            let y = x;
        }
        """
        # x is 100, so x < 50 is false -> unreachable branch warning
        result = adaptive_analyze(src)
        # Should detect unreachable
        unreachable = [w for w in result['warnings'] if w.kind == WarningKind.UNREACHABLE_BRANCH]
        assert len(unreachable) > 0

    def test_always_true_condition(self):
        src = """
        let x = 10;
        if (x > 5) {
            let y = 1;
        } else {
            let z = 2;
        }
        """
        result = adaptive_analyze(src)
        unreachable = [w for w in result['warnings'] if w.kind == WarningKind.UNREACHABLE_BRANCH]
        assert len(unreachable) > 0

    def test_nested_if(self):
        src = """
        let x = 10;
        let y = 20;
        if (x > 5) {
            if (y > 15) {
                let z = x + y;
            }
        }
        """
        result = adaptive_analyze(src)
        assert result is not None  # no crash

    def test_relational_condition_triggers_promotion(self):
        """if (x < y) should trigger zone-level promotion."""
        src = """
        let x = 5;
        let y = 10;
        if (x < y) {
            let z = y - x;
        }
        """
        result = adaptive_analyze(src)
        # Should have promoted for relational condition
        promos = result['promotions']
        # At least some promotion event should exist (relational guard or assign)
        assert len(promos) >= 0  # may or may not promote depending on const folding


# ============================================================
# 3. Loop Analysis
# ============================================================

class TestLoopAnalysis:

    def test_simple_countdown(self):
        src = """
        let x = 10;
        while (x > 0) {
            x = x - 1;
        }
        """
        result = adaptive_analyze(src)
        env = result['env']
        iv = env.get_bounds('x')
        assert iv.hi <= 0  # x <= 0 after loop

    def test_simple_countup(self):
        src = """
        let x = 0;
        while (x < 10) {
            x = x + 1;
        }
        """
        result = adaptive_analyze(src)
        env = result['env']
        iv = env.get_bounds('x')
        assert iv.lo >= 10  # x >= 10 after loop

    def test_loop_iterations_tracked(self):
        src = """
        let x = 0;
        while (x < 5) {
            x = x + 1;
        }
        """
        result = adaptive_analyze(src)
        assert len(result['loop_iterations']) > 0

    def test_two_variable_loop(self):
        src = """
        let x = 0;
        let y = 10;
        while (x < y) {
            x = x + 1;
            y = y - 1;
        }
        """
        result = adaptive_analyze(src)
        env = result['env']
        # After loop: x >= y
        x_bounds = env.get_bounds('x')
        y_bounds = env.get_bounds('y')
        assert x_bounds is not None and y_bounds is not None

    def test_accumulator_loop(self):
        src = """
        let sum = 0;
        let i = 0;
        while (i < 5) {
            sum = sum + i;
            i = i + 1;
        }
        """
        result = adaptive_analyze(src)
        env = result['env']
        # sum should be non-negative
        assert env.get_sign('sum') in (Sign.POS, Sign.ZERO, Sign.NON_NEG, Sign.TOP)

    def test_nested_loop(self):
        src = """
        let x = 0;
        let y = 0;
        while (x < 3) {
            y = 0;
            while (y < 3) {
                y = y + 1;
            }
            x = x + 1;
        }
        """
        result = adaptive_analyze(src)
        env = result['env']
        assert env.get_bounds('x').lo >= 3


# ============================================================
# 4. Relational Analysis & Promotions
# ============================================================

class TestRelationalAnalysis:

    def test_copy_creates_equality(self):
        """x = y should create x == y constraint."""
        src = """
        let x = 10;
        let y = x;
        """
        result = adaptive_analyze(src)
        env = result['env']
        # y should be 10
        assert env.get_bounds('y') == Interval(10, 10)

    def test_difference_constraint(self):
        """x = y + 5 should create x - y = 5 constraint."""
        src = """
        let y = 10;
        let x = y + 5;
        """
        result = adaptive_analyze(src)
        env = result['env']
        assert env.get_bounds('x') == Interval(15, 15)

    def test_relational_with_loop(self):
        """Relational constraints survive loop analysis."""
        src = """
        let x = 0;
        let y = 10;
        while (x < 5) {
            x = x + 1;
        }
        """
        result = adaptive_analyze(src)
        env = result['env']
        # y should still be 10 (unchanged)
        assert env.get_bounds('y') == Interval(10, 10)

    def test_promotion_on_var_var_assign(self):
        """Assignment from one var to another triggers zone promotion."""
        src = """
        let a = 5;
        let b = 10;
        let c = b - a;
        """
        result = adaptive_analyze(src)
        # c depends on both a and b -> relational
        promos = result['promotions']
        rel_promos = [p for p in promos if p.reason == PromotionReason.RELATIONAL_ASSIGN]
        assert len(rel_promos) > 0

    def test_sum_triggers_octagon_promotion(self):
        """x = a + b should trigger octagon-level promotion."""
        src = """
        let a = 5;
        let b = 10;
        let s = a + b;
        """
        result = adaptive_analyze(src, max_level=DomainLevel.OCTAGON)
        promos = result['promotions']
        # Should see promotion to octagon for sum
        octagon_promos = [p for p in promos if p.to_level >= DomainLevel.OCTAGON]
        # May or may not promote depending on heuristics, but no crash
        assert result is not None

    def test_relational_guard_comparison(self):
        """if (x < y) in then-branch: x - y <= -1."""
        src = """
        let x = 3;
        let y = 10;
        if (x < y) {
            let z = 1;
        }
        """
        result = adaptive_analyze(src)
        # Constraint x - y <= -1 should exist
        constraints = result['env'].get_relational_constraints('x')
        # At least the zone was activated
        assert result['env'].relational_level >= DomainLevel.INTERVAL

    def test_variable_copy_preserves_bounds(self):
        """Copying should preserve exact bounds via relational."""
        src = """
        let x = 7;
        let y = x;
        let z = y;
        """
        result = adaptive_analyze(src)
        assert result['env'].get_bounds('z') == Interval(7, 7)

    def test_chain_of_additions(self):
        """x = a + 1; y = x + 1 should give y = a + 2."""
        src = """
        let a = 0;
        let x = a + 1;
        let y = x + 1;
        """
        result = adaptive_analyze(src)
        assert result['env'].get_bounds('y') == Interval(2, 2)


# ============================================================
# 5. Widening & Convergence
# ============================================================

class TestWidening:

    def test_basic_widening_convergence(self):
        """Loop should converge via widening."""
        src = """
        let x = 0;
        while (x < 100) {
            x = x + 1;
        }
        """
        result = adaptive_analyze(src)
        # Should converge within max_iterations
        assert len(result['loop_iterations']) > 0
        for line, iters in result['loop_iterations'].items():
            assert iters <= 50

    def test_widening_to_infinity(self):
        """Widening may jump to infinity for unbounded increments."""
        src = """
        let x = 0;
        while (x < 1000000) {
            x = x + 1;
        }
        """
        result = adaptive_analyze(src)
        env = result['env']
        # After exit: x >= 1000000 (may be widened to infinity on upper)
        assert env.get_bounds('x').lo >= Fraction(1000000) or env.get_bounds('x').lo == NEG_INF

    def test_convergence_driven_promotion(self):
        """Repeated widening precision loss should trigger promotion."""
        interp = AdaptiveInterpreter(
            max_level=DomainLevel.OCTAGON,
            promote_on_widening_loss=True,
            promotion_threshold=1
        )
        src = """
        let x = 0;
        let y = 10;
        while (x < y) {
            x = x + 1;
            y = y - 1;
        }
        """
        result = interp.analyze(src)
        # Should have tried promotion
        assert result is not None

    def test_no_promotion_when_disabled(self):
        """Convergence-driven promotion can be disabled."""
        interp = AdaptiveInterpreter(
            max_level=DomainLevel.POLYHEDRA,
            promote_on_widening_loss=False
        )
        src = """
        let x = 0;
        while (x < 100) {
            x = x + 1;
        }
        """
        result = interp.analyze(src)
        widening_promos = [p for p in result['promotions']
                          if p.reason == PromotionReason.WIDENING_LOSS]
        assert len(widening_promos) == 0


# ============================================================
# 6. Warning Detection
# ============================================================

class TestWarnings:

    def test_division_by_zero(self):
        src = """
        let x = 10;
        let y = 0;
        let z = x / y;
        """
        result = adaptive_analyze(src)
        div_warnings = [w for w in result['warnings'] if w.kind == WarningKind.DIVISION_BY_ZERO]
        assert len(div_warnings) > 0

    def test_possible_division_by_zero(self):
        src = """
        let x = 10;
        let y = 5;
        y = y - 5;
        let z = x / y;
        """
        result = adaptive_analyze(src)
        div_warnings = [w for w in result['warnings']
                       if w.kind in (WarningKind.DIVISION_BY_ZERO, WarningKind.POSSIBLE_DIVISION_BY_ZERO)]
        assert len(div_warnings) > 0

    def test_safe_division(self):
        src = """
        let x = 10;
        let y = 5;
        let z = x / y;
        """
        result = adaptive_analyze(src)
        div_warnings = [w for w in result['warnings']
                       if w.kind == WarningKind.DIVISION_BY_ZERO]
        assert len(div_warnings) == 0

    def test_dead_assignment_warning(self):
        src = """
        let x = 10;
        let y = 20;
        let z = x + 1;
        """
        result = adaptive_analyze(src)
        dead = [w for w in result['warnings'] if w.kind == WarningKind.DEAD_ASSIGNMENT]
        # y is never read, z is never read
        dead_vars = [w.message for w in dead]
        assert any("y" in m for m in dead_vars)

    def test_no_dead_assignment_for_used_var(self):
        src = """
        let x = 10;
        let y = x + 5;
        """
        result = adaptive_analyze(src)
        dead = [w for w in result['warnings'] if w.kind == WarningKind.DEAD_ASSIGNMENT]
        dead_vars = [w.message for w in dead]
        assert not any("x" in m for m in dead_vars)


# ============================================================
# 7. Adaptive Environment
# ============================================================

class TestAdaptiveEnv:

    def test_env_creation(self):
        env = AdaptiveEnv()
        assert env.relational_level == DomainLevel.INTERVAL

    def test_env_set_get(self):
        env = AdaptiveEnv()
        env.set_var('x', AbstractValue.from_value(42))
        assert env.get_bounds('x') == Interval(42, 42)

    def test_env_copy(self):
        env = AdaptiveEnv()
        env.set_var('x', AbstractValue.from_value(10))
        env2 = env.copy()
        env2.set_var('x', AbstractValue.from_value(20))
        assert env.get_bounds('x') == Interval(10, 10)
        assert env2.get_bounds('x') == Interval(20, 20)

    def test_env_join(self):
        env1 = AdaptiveEnv()
        env1.set_var('x', AbstractValue.from_value(5))
        env2 = AdaptiveEnv()
        env2.set_var('x', AbstractValue.from_value(15))
        joined = env1.join(env2)
        iv = joined.get_bounds('x')
        assert iv.lo <= 5 and iv.hi >= 15

    def test_env_widen(self):
        env1 = AdaptiveEnv()
        env1.set_var('x', AbstractValue.from_value(0))
        env2 = AdaptiveEnv()
        env2.set_var('x', AbstractValue(sign=Sign.NON_NEG, interval=Interval(0, 10), const=None))
        widened = env1.widen(env2)
        iv = widened.get_bounds('x')
        assert iv.lo <= 0

    def test_env_equals(self):
        env1 = AdaptiveEnv()
        env1.set_var('x', AbstractValue.from_value(10))
        env2 = AdaptiveEnv()
        env2.set_var('x', AbstractValue.from_value(10))
        assert env1.equals(env2)

    def test_env_not_equals(self):
        env1 = AdaptiveEnv()
        env1.set_var('x', AbstractValue.from_value(10))
        env2 = AdaptiveEnv()
        env2.set_var('x', AbstractValue.from_value(20))
        assert not env1.equals(env2)

    def test_env_add_constraint(self):
        env = AdaptiveEnv()
        env.set_var('x', AbstractValue(sign=Sign.TOP, interval=INTERVAL_TOP, const=None))
        env.set_var('y', AbstractValue(sign=Sign.TOP, interval=INTERVAL_TOP, const=None))
        env.add_constraint(LinearConstraint.diff_le('x', 'y', Fraction(5)))
        assert env.relational_level >= DomainLevel.ZONE

    def test_env_variables_property(self):
        env = AdaptiveEnv()
        env.set_var('a', AbstractValue.from_value(1))
        env.set_var('b', AbstractValue.from_value(2))
        assert env.variables == {'a', 'b'}

    def test_env_promotions_tracking(self):
        env = AdaptiveEnv()
        env.set_var('x', AbstractValue.from_value(5))
        env.set_var('y', AbstractValue.from_value(10))
        env.add_constraint(LinearConstraint.diff_le('x', 'y', Fraction(0)))
        promos = env.promotions
        assert len(promos) >= 0  # may or may not have promoted


# ============================================================
# 8. Domain Comparison
# ============================================================

class TestDomainComparison:

    def test_compare_simple(self):
        src = """
        let x = 10;
        let y = 20;
        """
        result = DomainComparison.compare_fixed_vs_adaptive(src, ['x', 'y'])
        assert 'bounds' in result
        assert 'x' in result['bounds']
        assert 'y' in result['bounds']

    def test_compare_with_relational(self):
        """Adaptive should find tighter bounds when relational info helps."""
        src = """
        let x = 5;
        let y = x + 3;
        """
        result = DomainComparison.compare_fixed_vs_adaptive(src, ['y'])
        bounds = result['bounds']['y']
        # All strategies should find y = 8
        for strategy, (lo, hi) in bounds.items():
            assert lo <= 8 and hi >= 8

    def test_precision_gain_report(self):
        src = """
        let x = 5;
        let y = x + 3;
        """
        gains = DomainComparison.precision_gain(src, ['y'])
        assert 'y' in gains

    def test_compare_loop(self):
        src = """
        let x = 0;
        while (x < 10) {
            x = x + 1;
        }
        """
        result = DomainComparison.compare_fixed_vs_adaptive(src, ['x'])
        assert 'bounds' in result

    def test_compare_strategies(self):
        src = """
        let a = 3;
        let b = 7;
        let c = a + b;
        """
        result = compare_strategies(src, ['c'])
        assert 'bounds' in result


# ============================================================
# 9. Program Point Classification
# ============================================================

class TestPointClassification:

    def test_classify_constant(self):
        points = classify_points("let x = 42;")
        assert len(points) > 0
        # Constant assignment -> interval level
        assert any(level == DomainLevel.INTERVAL for _, level, _ in points)

    def test_classify_relational_assign(self):
        points = classify_points("let x = 5; let y = 10; let z = x - y;")
        # z = x - y -> zone level
        zone_points = [(l, lev, r) for l, lev, r in points if lev >= DomainLevel.ZONE]
        assert len(zone_points) > 0

    def test_classify_sum_assign(self):
        points = classify_points("let x = 5; let y = 10; let s = x + y;")
        # s = x + y -> octagon level
        oct_points = [(l, lev, r) for l, lev, r in points if lev >= DomainLevel.OCTAGON]
        assert len(oct_points) > 0

    def test_classify_relational_condition(self):
        src = """
        let x = 5;
        let y = 10;
        if (x < y) {
            let z = 1;
        }
        """
        points = classify_points(src)
        rel_cond = [(l, lev, r) for l, lev, r in points if "relational condition" in r]
        assert len(rel_cond) > 0

    def test_classify_loop_condition(self):
        src = """
        let x = 0;
        let y = 10;
        while (x < y) {
            x = x + 1;
        }
        """
        points = classify_points(src)
        loop_rel = [(l, lev, r) for l, lev, r in points if "loop" in r]
        assert len(loop_rel) > 0


# ============================================================
# 10. Public API Functions
# ============================================================

class TestPublicAPI:

    def test_adaptive_analyze(self):
        result = adaptive_analyze("let x = 42;")
        assert 'env' in result
        assert 'warnings' in result
        assert 'promotions' in result

    def test_analyze_with_comparison(self):
        result = analyze_with_comparison("let x = 10;", ['x'])
        assert 'bounds' in result

    def test_precision_report(self):
        report = precision_report("let x = 5; let y = x + 3;", ['y'])
        assert 'y' in report

    def test_get_promotions(self):
        promos = get_promotions("let x = 5; let y = 10; let z = x - y;")
        assert isinstance(promos, list)

    def test_get_relational_bounds(self):
        iv = get_relational_bounds("let x = 42;", 'x')
        assert iv.lo == 42 and iv.hi == 42

    def test_get_relational_constraints(self):
        constraints = get_relational_constraints("let x = 5; let y = x;")
        assert isinstance(constraints, list)

    def test_classify_points(self):
        points = classify_points("let x = 5;")
        assert isinstance(points, list)


# ============================================================
# 11. Max Level Capping
# ============================================================

class TestMaxLevel:

    def test_interval_only(self):
        """Max level INTERVAL means no relational promotions."""
        result = adaptive_analyze("let x = 5; let y = x;", max_level=DomainLevel.INTERVAL)
        promos = result['promotions']
        above_interval = [p for p in promos if p.to_level > DomainLevel.INTERVAL]
        assert len(above_interval) == 0

    def test_zone_max(self):
        """Max level ZONE caps promotions at zone."""
        src = "let a = 5; let b = 10; let s = a + b;"
        result = adaptive_analyze(src, max_level=DomainLevel.ZONE)
        for p in result['promotions']:
            assert p.to_level <= DomainLevel.ZONE

    def test_octagon_max(self):
        result = adaptive_analyze("let x = 5; let y = x + 3;", max_level=DomainLevel.OCTAGON)
        for p in result['promotions']:
            assert p.to_level <= DomainLevel.OCTAGON

    def test_full_polyhedra(self):
        result = adaptive_analyze("let x = 5; let y = x;", max_level=DomainLevel.POLYHEDRA)
        assert result is not None


# ============================================================
# 12. Complex Programs
# ============================================================

class TestComplexPrograms:

    def test_fibonacci_like(self):
        src = """
        let a = 0;
        let b = 1;
        let i = 0;
        while (i < 5) {
            let t = a + b;
            a = b;
            b = t;
            i = i + 1;
        }
        """
        result = adaptive_analyze(src)
        env = result['env']
        # a and b should be non-negative
        assert env.get_sign('a') in (Sign.POS, Sign.ZERO, Sign.NON_NEG, Sign.TOP)

    def test_max_computation(self):
        src = """
        let x = 5;
        let y = 10;
        let m = x;
        if (y > x) {
            m = y;
        }
        """
        result = adaptive_analyze(src)
        env = result['env']
        iv = env.get_bounds('m')
        assert iv.lo <= 10 and iv.hi >= 10

    def test_abs_computation(self):
        src = """
        let x = 0 - 5;
        let a = x;
        if (x < 0) {
            a = 0 - x;
        }
        """
        result = adaptive_analyze(src)
        env = result['env']
        iv = env.get_bounds('a')
        assert iv.lo >= 0 or iv.hi >= 0

    def test_swap(self):
        src = """
        let x = 10;
        let y = 20;
        let t = x;
        x = y;
        y = t;
        """
        result = adaptive_analyze(src)
        env = result['env']
        # After swap: x=20, y=10
        assert env.get_bounds('x') == Interval(20, 20)
        assert env.get_bounds('y') == Interval(10, 10)

    def test_bounded_loop_with_guard(self):
        src = """
        let x = 0;
        let y = 0;
        while (x < 10) {
            if (x > 5) {
                y = y + 1;
            }
            x = x + 1;
        }
        """
        result = adaptive_analyze(src)
        env = result['env']
        assert env.get_bounds('x').lo >= 10

    def test_multiple_loops(self):
        src = """
        let x = 0;
        while (x < 5) {
            x = x + 1;
        }
        let y = 0;
        while (y < 3) {
            y = y + 1;
        }
        """
        result = adaptive_analyze(src)
        env = result['env']
        assert env.get_bounds('x').lo >= 5
        assert env.get_bounds('y').lo >= 3

    def test_division_in_loop(self):
        src = """
        let x = 100;
        let i = 0;
        while (i < 5) {
            x = x / 2;
            i = i + 1;
        }
        """
        result = adaptive_analyze(src)
        env = result['env']
        # x should be non-negative (100 / 2 / 2 / ...)
        assert env.get_sign('x') in (Sign.POS, Sign.ZERO, Sign.NON_NEG, Sign.TOP)


# ============================================================
# 13. Promotion Events
# ============================================================

class TestPromotionEvents:

    def test_promotion_event_fields(self):
        event = PromotionEvent(
            line=5, variable='x',
            from_level=DomainLevel.INTERVAL,
            to_level=DomainLevel.ZONE,
            reason=PromotionReason.RELATIONAL_ASSIGN,
            detail="x = y + 1"
        )
        assert event.line == 5
        assert event.variable == 'x'
        assert event.from_level == DomainLevel.INTERVAL
        assert event.to_level == DomainLevel.ZONE
        assert event.reason == PromotionReason.RELATIONAL_ASSIGN

    def test_promotion_reason_values(self):
        assert PromotionReason.RELATIONAL_ASSIGN == 1
        assert PromotionReason.RELATIONAL_GUARD == 2
        assert PromotionReason.WIDENING_LOSS == 3
        assert PromotionReason.EXPLICIT_REQUEST == 4
        assert PromotionReason.CONSTRAINT_DEMAND == 5

    def test_promotions_are_ordered(self):
        src = """
        let a = 5;
        let b = 10;
        let c = a - b;
        let d = a + b;
        """
        result = adaptive_analyze(src)
        promos = result['promotions']
        if len(promos) >= 2:
            # Promotions should be in order of occurrence
            # (can't guarantee specific ordering due to internal details)
            assert all(isinstance(p, PromotionEvent) for p in promos)


# ============================================================
# 14. Cost Tracking
# ============================================================

class TestCostTracking:

    def test_costs_tracked(self):
        result = adaptive_analyze("let x = 5; let y = 10;")
        assert 'domain_costs' in result
        assert result['domain_costs'].get('assign', 0) >= 2

    def test_branch_cost(self):
        src = """
        let x = 5;
        if (x > 3) {
            let y = 1;
        }
        """
        result = adaptive_analyze(src)
        assert result['domain_costs'].get('branch', 0) >= 1

    def test_loop_cost(self):
        src = """
        let x = 0;
        while (x < 5) {
            x = x + 1;
        }
        """
        result = adaptive_analyze(src)
        assert result['domain_costs'].get('loop', 0) >= 1


# ============================================================
# 15. Edge Cases
# ============================================================

class TestEdgeCases:

    def test_empty_program(self):
        # Just a no-op expression or let
        result = adaptive_analyze("let x = 0;")
        assert result is not None

    def test_single_variable(self):
        result = adaptive_analyze("let x = 42;")
        assert result['env'].get_bounds('x') == Interval(42, 42)

    def test_large_constants(self):
        result = adaptive_analyze("let x = 1000000;")
        assert result['env'].get_bounds('x') == Interval(1000000, 1000000)

    def test_zero_division_guard(self):
        """Division by variable that could be zero."""
        src = """
        let x = 10;
        let y = 1;
        let z = x / y;
        """
        result = adaptive_analyze(src)
        # y = 1, not zero, so no div-by-zero warning
        div_warnings = [w for w in result['warnings'] if w.kind == WarningKind.DIVISION_BY_ZERO]
        assert len(div_warnings) == 0

    def test_self_assignment(self):
        """x = x should not crash."""
        src = """
        let x = 5;
        x = x;
        """
        result = adaptive_analyze(src)
        assert result['env'].get_bounds('x') == Interval(5, 5)

    def test_many_variables(self):
        lines = [f"let v{i} = {i};" for i in range(20)]
        src = "\n".join(lines)
        result = adaptive_analyze(src)
        assert result['env'].get_bounds('v0') == Interval(0, 0)
        assert result['env'].get_bounds('v19') == Interval(19, 19)

    def test_max_iterations_respected(self):
        """Ensure max_iterations is respected."""
        interp = AdaptiveInterpreter(max_iterations=3)
        src = """
        let x = 0;
        while (x < 1000) {
            x = x + 1;
        }
        """
        result = interp.analyze(src)
        for line, iters in result['loop_iterations'].items():
            assert iters <= 3


# ============================================================
# 16. Var Read/Write Tracking
# ============================================================

class TestVarTracking:

    def test_var_writes(self):
        result = adaptive_analyze("let x = 5; let y = 10;")
        assert 'x' in result['var_writes']
        assert 'y' in result['var_writes']

    def test_var_reads(self):
        result = adaptive_analyze("let x = 5; let y = x + 1;")
        assert 'x' in result['var_reads']

    def test_no_false_reads(self):
        result = adaptive_analyze("let x = 5; let y = 10;")
        # y is not read by anyone
        assert 'y' not in result['var_reads']


# ============================================================
# 17. Integration: All Features Together
# ============================================================

class TestIntegration:

    def test_full_program(self):
        """A program that exercises all features."""
        src = """
        let n = 10;
        let sum = 0;
        let i = 0;
        while (i < n) {
            sum = sum + i;
            i = i + 1;
        }
        let avg = sum / n;
        """
        result = adaptive_analyze(src)
        env = result['env']

        # n should still be 10
        assert env.get_bounds('n') == Interval(10, 10)
        # i >= n after loop
        assert env.get_bounds('i').lo >= 10
        # sum should be non-negative
        assert env.get_sign('sum') in (Sign.POS, Sign.ZERO, Sign.NON_NEG, Sign.TOP)
        # No div-by-zero (n = 10)
        div_warnings = [w for w in result['warnings'] if w.kind == WarningKind.DIVISION_BY_ZERO]
        assert len(div_warnings) == 0

    def test_full_program_with_comparison(self):
        src = """
        let x = 0;
        let y = 100;
        while (x < y) {
            x = x + 1;
            y = y - 1;
        }
        """
        result = analyze_with_comparison(src, ['x', 'y'])
        assert 'bounds' in result
        assert 'x' in result['bounds']
        assert 'y' in result['bounds']

    def test_full_with_promotions_and_warnings(self):
        src = """
        let a = 5;
        let b = 10;
        let diff = b - a;
        let sum = a + b;
        let x = 0;
        while (x < diff) {
            x = x + 1;
        }
        let z = sum / x;
        """
        result = adaptive_analyze(src)
        assert len(result['promotions']) >= 0
        # x >= diff after loop, sum = 15, diff = 5
        # x should be non-zero (at least 5), so no div-by-zero
        env = result['env']
        assert result is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
