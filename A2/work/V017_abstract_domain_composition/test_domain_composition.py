"""
Tests for V017: Abstract Domain Composition (Reduced Product)
"""
import pytest
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHALLENGES = os.path.normpath(os.path.join(_HERE, '..', '..', '..', 'challenges'))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_CHALLENGES, 'C039_abstract_interpreter'))
sys.path.insert(0, os.path.join(_CHALLENGES, 'C010_stack_vm'))

from domain_composition import (
    # Core reduction
    reduce_value, reduce_env, reduce_extended,
    sign_from_interval, interval_from_sign, sign_from_const, interval_from_const,
    const_from_interval, sign_meet,
    # Parity
    Parity, parity_from_value, parity_join, parity_meet,
    parity_add, parity_sub, parity_mul, parity_neg,
    interval_refine_by_parity,
    # Extended types
    ExtendedValue, ExtendedEnv, ExtendedInterpreter,
    # Composed interpreter (3-domain)
    ComposedInterpreter,
    # High-level API
    composed_analyze, get_variable_info, get_precision_gains,
    compare_analyses, composed_reduce, analyze_with_comparison,
    ComparisonResult,
)
from abstract_interpreter import (
    Sign, Interval, AbstractValue, AbstractEnv,
    ConstBot, ConstTop, ConstVal,
    INTERVAL_BOT, INTERVAL_TOP, CONST_BOT, CONST_TOP,
    INF, NEG_INF,
    analyze as c039_analyze,
)


# ============================================================
# Section 1: Sign-Interval reduction
# ============================================================

class TestSignIntervalReduction:
    def test_positive_interval_gives_pos_sign(self):
        assert sign_from_interval(Interval(1, 10)) == Sign.POS

    def test_negative_interval_gives_neg_sign(self):
        assert sign_from_interval(Interval(-5, -1)) == Sign.NEG

    def test_zero_interval_gives_zero_sign(self):
        assert sign_from_interval(Interval(0, 0)) == Sign.ZERO

    def test_non_neg_interval(self):
        assert sign_from_interval(Interval(0, 10)) == Sign.NON_NEG

    def test_non_pos_interval(self):
        assert sign_from_interval(Interval(-5, 0)) == Sign.NON_POS

    def test_mixed_interval_gives_top(self):
        assert sign_from_interval(Interval(-5, 5)) == Sign.TOP

    def test_pos_sign_gives_positive_interval(self):
        iv = interval_from_sign(Sign.POS)
        assert iv.lo == 1
        assert iv.hi == INF

    def test_neg_sign_gives_negative_interval(self):
        iv = interval_from_sign(Sign.NEG)
        assert iv.lo == NEG_INF
        assert iv.hi == -1

    def test_zero_sign_gives_zero_interval(self):
        iv = interval_from_sign(Sign.ZERO)
        assert iv.lo == 0 and iv.hi == 0

    def test_bot_interval_gives_bot_sign(self):
        assert sign_from_interval(INTERVAL_BOT) == Sign.BOT


# ============================================================
# Section 2: Constant reduction
# ============================================================

class TestConstantReduction:
    def test_const_to_sign(self):
        assert sign_from_const(ConstVal(5)) == Sign.POS
        assert sign_from_const(ConstVal(-3)) == Sign.NEG
        assert sign_from_const(ConstVal(0)) == Sign.ZERO

    def test_const_to_interval(self):
        iv = interval_from_const(ConstVal(7))
        assert iv.lo == 7 and iv.hi == 7

    def test_singleton_interval_to_const(self):
        c = const_from_interval(Interval(5, 5))
        assert isinstance(c, ConstVal) and c.value == 5

    def test_non_singleton_interval_top(self):
        c = const_from_interval(Interval(3, 7))
        assert isinstance(c, ConstTop)

    def test_bot_const_to_bot_sign(self):
        assert sign_from_const(CONST_BOT) == Sign.BOT

    def test_top_const_to_top_sign(self):
        assert sign_from_const(CONST_TOP) == Sign.TOP


# ============================================================
# Section 3: Sign meet
# ============================================================

class TestSignMeet:
    def test_meet_same(self):
        assert sign_meet(Sign.POS, Sign.POS) == Sign.POS

    def test_meet_with_top(self):
        assert sign_meet(Sign.POS, Sign.TOP) == Sign.POS
        assert sign_meet(Sign.TOP, Sign.NEG) == Sign.NEG

    def test_meet_non_neg_and_non_pos(self):
        assert sign_meet(Sign.NON_NEG, Sign.NON_POS) == Sign.ZERO

    def test_meet_pos_and_neg(self):
        assert sign_meet(Sign.POS, Sign.NEG) == Sign.BOT

    def test_meet_non_neg_and_pos(self):
        assert sign_meet(Sign.NON_NEG, Sign.POS) == Sign.POS

    def test_meet_with_bot(self):
        assert sign_meet(Sign.BOT, Sign.POS) == Sign.BOT


# ============================================================
# Section 4: 3-domain reduce_value
# ============================================================

class TestReduceValue:
    def test_constant_tightens_all(self):
        """Constant 5 should give POS sign and [5,5] interval."""
        av = AbstractValue(Sign.TOP, INTERVAL_TOP, ConstVal(5))
        rv = reduce_value(av)
        assert rv.sign == Sign.POS
        assert rv.interval.lo == 5 and rv.interval.hi == 5

    def test_interval_tightens_sign(self):
        """Interval [1,10] should give POS sign."""
        av = AbstractValue(Sign.TOP, Interval(1, 10), CONST_TOP)
        rv = reduce_value(av)
        assert rv.sign == Sign.POS

    def test_sign_tightens_interval(self):
        """NEG sign should restrict interval to [-inf, -1]."""
        av = AbstractValue(Sign.NEG, INTERVAL_TOP, CONST_TOP)
        rv = reduce_value(av)
        assert rv.interval.hi == -1

    def test_singleton_becomes_constant(self):
        """Interval [7,7] should produce constant 7."""
        av = AbstractValue(Sign.TOP, Interval(7, 7), CONST_TOP)
        rv = reduce_value(av)
        assert isinstance(rv.const, ConstVal) and rv.const.value == 7
        assert rv.sign == Sign.POS

    def test_bot_propagation(self):
        """Any BOT domain should make all BOT."""
        av = AbstractValue(Sign.BOT, Interval(1, 10), ConstVal(5))
        rv = reduce_value(av)
        assert rv.sign == Sign.BOT
        assert rv.interval.is_bot()
        assert isinstance(rv.const, ConstBot)

    def test_conflicting_info_goes_bot(self):
        """POS sign + negative interval -> BOT."""
        av = AbstractValue(Sign.POS, Interval(-5, -1), CONST_TOP)
        rv = reduce_value(av)
        assert rv.sign == Sign.BOT


# ============================================================
# Section 5: Parity domain
# ============================================================

class TestParityDomain:
    def test_parity_from_value(self):
        assert parity_from_value(0) == Parity.EVEN
        assert parity_from_value(1) == Parity.ODD
        assert parity_from_value(4) == Parity.EVEN
        assert parity_from_value(-3) == Parity.ODD

    def test_parity_add(self):
        assert parity_add(Parity.EVEN, Parity.EVEN) == Parity.EVEN
        assert parity_add(Parity.ODD, Parity.ODD) == Parity.EVEN
        assert parity_add(Parity.EVEN, Parity.ODD) == Parity.ODD
        assert parity_add(Parity.ODD, Parity.EVEN) == Parity.ODD

    def test_parity_mul(self):
        assert parity_mul(Parity.EVEN, Parity.ODD) == Parity.EVEN
        assert parity_mul(Parity.ODD, Parity.ODD) == Parity.ODD
        assert parity_mul(Parity.EVEN, Parity.EVEN) == Parity.EVEN

    def test_parity_join(self):
        assert parity_join(Parity.EVEN, Parity.ODD) == Parity.TOP
        assert parity_join(Parity.EVEN, Parity.EVEN) == Parity.EVEN
        assert parity_join(Parity.BOT, Parity.ODD) == Parity.ODD

    def test_parity_meet(self):
        assert parity_meet(Parity.EVEN, Parity.ODD) == Parity.BOT
        assert parity_meet(Parity.EVEN, Parity.TOP) == Parity.EVEN
        assert parity_meet(Parity.TOP, Parity.ODD) == Parity.ODD


# ============================================================
# Section 6: Parity-interval interaction
# ============================================================

class TestParityIntervalReduction:
    def test_even_tightens_odd_bounds(self):
        """[1,5] with EVEN -> [2,4]."""
        iv = interval_refine_by_parity(Interval(1, 5), Parity.EVEN)
        assert iv.lo == 2 and iv.hi == 4

    def test_odd_tightens_even_bounds(self):
        """[0,4] with ODD -> [1,3]."""
        iv = interval_refine_by_parity(Interval(0, 4), Parity.ODD)
        assert iv.lo == 1 and iv.hi == 3

    def test_parity_refine_no_change(self):
        """[2,6] with EVEN -> [2,6] (already aligned)."""
        iv = interval_refine_by_parity(Interval(2, 6), Parity.EVEN)
        assert iv.lo == 2 and iv.hi == 6

    def test_parity_refine_to_bot(self):
        """[3,3] with EVEN -> BOT (3 is odd)."""
        iv = interval_refine_by_parity(Interval(3, 3), Parity.EVEN)
        assert iv.is_bot()

    def test_parity_top_no_change(self):
        iv = interval_refine_by_parity(Interval(1, 5), Parity.TOP)
        assert iv.lo == 1 and iv.hi == 5


# ============================================================
# Section 7: Extended reduction (4-domain)
# ============================================================

class TestExtendedReduction:
    def test_full_reduction_from_constant(self):
        ev = ExtendedValue(Sign.TOP, INTERVAL_TOP, ConstVal(6), Parity.TOP)
        rv = reduce_extended(ev)
        assert rv.sign == Sign.POS
        assert rv.interval.lo == 6 and rv.interval.hi == 6
        assert rv.parity == Parity.EVEN

    def test_parity_tightens_interval(self):
        """[0,3] + ODD -> [1,3] + constant discovery possible."""
        ev = ExtendedValue(Sign.NON_NEG, Interval(0, 3), CONST_TOP, Parity.ODD)
        rv = reduce_extended(ev)
        assert rv.interval.lo == 1 and rv.interval.hi == 3

    def test_interval_singleton_discovers_parity(self):
        ev = ExtendedValue(Sign.TOP, Interval(4, 4), CONST_TOP, Parity.TOP)
        rv = reduce_extended(ev)
        assert rv.parity == Parity.EVEN
        assert isinstance(rv.const, ConstVal) and rv.const.value == 4

    def test_conflicting_parity_goes_bot(self):
        """Interval [5,5] (odd) + EVEN parity -> BOT."""
        ev = ExtendedValue(Sign.POS, Interval(5, 5), ConstVal(5), Parity.EVEN)
        rv = reduce_extended(ev)
        assert rv.sign == Sign.BOT


# ============================================================
# Section 8: ExtendedEnv operations
# ============================================================

class TestExtendedEnv:
    def test_set_and_get(self):
        env = ExtendedEnv()
        env.set_from_value('x', 5)
        assert env.get_sign('x') == Sign.POS
        assert env.get_interval('x') == Interval(5, 5)
        assert env.get_parity('x') == Parity.ODD

    def test_join(self):
        e1 = ExtendedEnv()
        e1.set_from_value('x', 3)
        e2 = ExtendedEnv()
        e2.set_from_value('x', 7)
        joined = e1.join(e2)
        assert joined.get_interval('x') == Interval(3, 7)
        assert joined.get_sign('x') == Sign.POS
        assert joined.get_parity('x') == Parity.ODD  # 3 and 7 both odd

    def test_join_different_parity(self):
        e1 = ExtendedEnv()
        e1.set_from_value('x', 2)
        e2 = ExtendedEnv()
        e2.set_from_value('x', 3)
        joined = e1.join(e2)
        assert joined.get_parity('x') == Parity.TOP  # even + odd = top

    def test_reduce_env(self):
        env = ExtendedEnv()
        env.set('x', sign=Sign.TOP, interval=Interval(5, 5), const=CONST_TOP, parity=Parity.TOP)
        env.reduce()
        assert env.get_sign('x') == Sign.POS
        assert isinstance(env.get_const('x'), ConstVal)
        assert env.get_parity('x') == Parity.ODD

    def test_copy_independence(self):
        env = ExtendedEnv()
        env.set_from_value('x', 5)
        c = env.copy()
        c.set_from_value('x', 10)
        assert env.get_interval('x') == Interval(5, 5)
        assert c.get_interval('x') == Interval(10, 10)


# ============================================================
# Section 9: Composed 3-domain interpreter
# ============================================================

class TestComposedInterpreter:
    def test_constant_propagation_tightens(self):
        src = "let x = 5;"
        interp = ComposedInterpreter()
        result = interp.analyze(src)
        env = result['env']
        # C039 baseline: x has const=5, sign=POS, interval=[5,5]
        # Composed: should be same or better (const gives sign+interval)
        assert env.get_sign('x') == Sign.POS
        assert env.get_interval('x') == Interval(5, 5)

    def test_reduction_count(self):
        src = "let x = 5; let y = x + 3;"
        interp = ComposedInterpreter()
        result = interp.analyze(src)
        assert interp.reduction_count > 0

    def test_conditional_with_reduction(self):
        src = """
        let x = 10;
        let y = 0;
        if (x > 5) {
            y = 1;
        } else {
            y = 2;
        }
        """
        interp = ComposedInterpreter()
        result = interp.analyze(src)
        env = result['env']
        # x = 10 > 5 is always true, so y = 1
        assert env.get_const('y') == ConstVal(1) or env.get_interval('y') == Interval(1, 1)


# ============================================================
# Section 10: Extended 4-domain interpreter
# ============================================================

class TestExtendedInterpreter:
    def test_simple_assignment(self):
        src = "let x = 6;"
        result = composed_analyze(src)
        env = result['env']
        assert env.get_sign('x') == Sign.POS
        assert env.get_interval('x') == Interval(6, 6)
        assert isinstance(env.get_const('x'), ConstVal) and env.get_const('x').value == 6
        assert env.get_parity('x') == Parity.EVEN

    def test_arithmetic(self):
        src = "let x = 3; let y = 4; let z = x + y;"
        result = composed_analyze(src)
        env = result['env']
        assert isinstance(env.get_const('z'), ConstVal) and env.get_const('z').value == 7
        assert env.get_parity('z') == Parity.ODD

    def test_multiplication_parity(self):
        src = "let x = 3; let y = 5; let z = x * y;"
        result = composed_analyze(src)
        env = result['env']
        assert env.get_parity('z') == Parity.ODD  # odd * odd
        assert isinstance(env.get_const('z'), ConstVal) and env.get_const('z').value == 15

    def test_conditional_join(self):
        src = """
        let x = 10;
        let y = 0;
        if (x > 0) {
            y = 2;
        } else {
            y = 4;
        }
        """
        result = composed_analyze(src)
        env = result['env']
        # x=10>0 always true, y=2
        assert env.get_parity('y') == Parity.EVEN

    def test_while_loop(self):
        src = """
        let i = 10;
        while (i > 0) {
            i = i - 1;
        }
        """
        result = composed_analyze(src)
        env = result['env']
        # After loop: i <= 0 (from condition false)
        assert env.get_interval('i').hi <= 0

    def test_negation_preserves_parity(self):
        src = "let x = 3; let y = -x;"
        result = composed_analyze(src)
        env = result['env']
        assert env.get_parity('y') == Parity.ODD
        assert env.get_sign('y') == Sign.NEG


# ============================================================
# Section 11: Comparison with C039 baseline
# ============================================================

class TestComparisonAPI:
    def test_compare_basic(self):
        src = "let x = 5;"
        cr = compare_analyses(src)
        assert isinstance(cr, ComparisonResult)
        assert cr.reductions > 0

    def test_parity_is_always_a_gain(self):
        src = "let x = 4;"
        cr = compare_analyses(src)
        parity_gains = [g for g in cr.precision_gains if g[1] == 'parity']
        assert len(parity_gains) > 0

    def test_precision_gains_detected(self):
        # C039 runs domains independently; V017 composes them.
        # For simple constants, both should agree on sign/interval/const.
        # Parity is always a new gain.
        src = "let x = 6; let y = x + 2;"
        cr = compare_analyses(src)
        assert cr.has_gains  # at least parity gains

    def test_compare_with_conditional(self):
        src = """
        let x = 10;
        let y = 0;
        if (x > 5) {
            y = x - 3;
        } else {
            y = x + 3;
        }
        """
        cr = compare_analyses(src)
        assert isinstance(cr, ComparisonResult)


# ============================================================
# Section 12: High-level API
# ============================================================

class TestHighLevelAPI:
    def test_composed_analyze(self):
        result = composed_analyze("let x = 42;")
        assert 'env' in result
        assert 'warnings' in result
        assert 'reductions' in result

    def test_get_variable_info(self):
        info = get_variable_info("let x = 7;", 'x')
        assert isinstance(info, ExtendedValue)
        assert info.sign == Sign.POS
        assert info.parity == Parity.ODD

    def test_get_precision_gains(self):
        gains = get_precision_gains("let x = 8;")
        assert isinstance(gains, list)

    def test_composed_reduce_function(self):
        rv = composed_reduce(sign=Sign.TOP, interval=Interval(3, 3))
        assert rv.sign == Sign.POS
        assert isinstance(rv.const, ConstVal) and rv.const.value == 3
        assert rv.parity == Parity.ODD

    def test_analyze_with_comparison(self):
        cr = analyze_with_comparison("let x = 10; let y = x * 2;")
        assert isinstance(cr, ComparisonResult)


# ============================================================
# Section 13: Real programs with precision gains
# ============================================================

class TestRealPrograms:
    def test_accumulator_parity(self):
        """Accumulate even numbers -> result parity is even."""
        src = """
        let sum = 0;
        let x = 2;
        let y = 4;
        let z = sum + x + y;
        """
        result = composed_analyze(src)
        env = result['env']
        assert env.get_parity('z') == Parity.EVEN

    def test_sign_from_comparison(self):
        """After if (x > 0), x is positive."""
        src = """
        let x = 5;
        let y = 0;
        if (x > 0) {
            y = x;
        }
        """
        result = composed_analyze(src)
        env = result['env']
        # y should be positive (x=5>0, so y=x=5)
        assert env.get_sign('y') in (Sign.POS, Sign.NON_NEG)

    def test_multi_variable_tracking(self):
        src = """
        let a = 3;
        let b = 4;
        let c = a + b;
        let d = a * b;
        """
        result = composed_analyze(src)
        env = result['env']
        assert isinstance(env.get_const('c'), ConstVal) and env.get_const('c').value == 7
        assert env.get_parity('c') == Parity.ODD
        assert isinstance(env.get_const('d'), ConstVal) and env.get_const('d').value == 12
        assert env.get_parity('d') == Parity.EVEN

    def test_subtraction_tracking(self):
        src = "let x = 10; let y = 3; let z = x - y;"
        result = composed_analyze(src)
        env = result['env']
        assert isinstance(env.get_const('z'), ConstVal) and env.get_const('z').value == 7
        assert env.get_parity('z') == Parity.ODD


# ============================================================
# Section 14: Edge cases
# ============================================================

class TestEdgeCases:
    def test_zero_value(self):
        info = get_variable_info("let x = 0;", 'x')
        assert info.sign == Sign.ZERO
        assert info.parity == Parity.EVEN
        assert isinstance(info.const, ConstVal) and info.const.value == 0

    def test_negative_value(self):
        info = get_variable_info("let x = -4;", 'x')
        assert info.sign == Sign.NEG
        assert info.parity == Parity.EVEN

    def test_large_value(self):
        info = get_variable_info("let x = 1000000;", 'x')
        assert info.sign == Sign.POS
        assert info.parity == Parity.EVEN

    def test_empty_program(self):
        result = composed_analyze("let x = 1;")
        assert result['env'] is not None

    def test_reductions_performed(self):
        result = composed_analyze("let x = 5; let y = x + 3;")
        assert result['reductions'] > 0


# ============================================================
# Section 15: Parity-driven precision
# ============================================================

class TestParityDrivenPrecision:
    def test_even_plus_even_is_even(self):
        src = "let a = 2; let b = 4; let c = a + b;"
        info = get_variable_info(src, 'c')
        assert info.parity == Parity.EVEN

    def test_odd_plus_odd_is_even(self):
        src = "let a = 3; let b = 5; let c = a + b;"
        info = get_variable_info(src, 'c')
        assert info.parity == Parity.EVEN

    def test_even_times_anything_is_even(self):
        src = "let a = 2; let b = 7; let c = a * b;"
        info = get_variable_info(src, 'c')
        assert info.parity == Parity.EVEN

    def test_parity_comparison_can_resolve_neq(self):
        """If two values have different parity, they can't be equal."""
        interp = ExtendedInterpreter()
        # 3 (odd) vs 4 (even) -> definitely not equal
        src = """
        let x = 3;
        let y = 4;
        let r = (x != y);
        """
        result = interp.analyze(src)
        env = result['env']
        r = env.get_extended('r')
        # Should know result is 1 (true) since parities differ
        assert isinstance(r.const, ConstVal) and r.const.value == 1

    def test_parity_preserves_through_negation(self):
        src = "let x = 5; let y = -x;"
        info = get_variable_info(src, 'y')
        assert info.parity == Parity.ODD
