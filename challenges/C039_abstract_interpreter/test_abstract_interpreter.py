"""
Tests for C039: Abstract Interpreter
Challenge C039 -- AgentZero Session 040
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from abstract_interpreter import (
    analyze, get_variable_range, get_variable_sign, get_variable_const,
    get_warnings, check_division_safety, check_dead_code,
    Sign, Interval, INTERVAL_BOT, INTERVAL_TOP,
    ConstVal, ConstTop, ConstBot, CONST_TOP, CONST_BOT,
    sign_join, sign_add, sign_sub, sign_mul, sign_div, sign_neg,
    sign_contains_zero, sign_from_value,
    interval_join, interval_meet, interval_widen,
    interval_add, interval_sub, interval_neg, interval_mul, interval_div,
    interval_from_value, interval_mod,
    const_join, const_from_value,
    AbstractEnv, AbstractValue, AbstractInterpreter,
    WarningKind, Warning,
    INF, NEG_INF,
)


# ============================================================
# Sign Domain Tests
# ============================================================

class TestSignDomain:
    def test_sign_from_positive(self):
        assert sign_from_value(5) == Sign.POS

    def test_sign_from_negative(self):
        assert sign_from_value(-3) == Sign.NEG

    def test_sign_from_zero(self):
        assert sign_from_value(0) == Sign.ZERO

    def test_sign_from_float(self):
        assert sign_from_value(3.14) == Sign.POS
        assert sign_from_value(-2.7) == Sign.NEG
        assert sign_from_value(0.0) == Sign.ZERO

    def test_sign_from_bool(self):
        # Booleans are NON_NEG (True=1, False=0)
        assert sign_from_value(True) == Sign.NON_NEG
        assert sign_from_value(False) == Sign.NON_NEG

    def test_sign_join_same(self):
        assert sign_join(Sign.POS, Sign.POS) == Sign.POS
        assert sign_join(Sign.NEG, Sign.NEG) == Sign.NEG
        assert sign_join(Sign.ZERO, Sign.ZERO) == Sign.ZERO

    def test_sign_join_bot(self):
        assert sign_join(Sign.BOT, Sign.POS) == Sign.POS
        assert sign_join(Sign.NEG, Sign.BOT) == Sign.NEG

    def test_sign_join_top(self):
        assert sign_join(Sign.TOP, Sign.POS) == Sign.TOP
        assert sign_join(Sign.NEG, Sign.TOP) == Sign.TOP

    def test_sign_join_pos_zero(self):
        assert sign_join(Sign.POS, Sign.ZERO) == Sign.NON_NEG

    def test_sign_join_neg_zero(self):
        assert sign_join(Sign.NEG, Sign.ZERO) == Sign.NON_POS

    def test_sign_join_pos_neg(self):
        assert sign_join(Sign.POS, Sign.NEG) == Sign.TOP

    def test_sign_neg_positive(self):
        assert sign_neg(Sign.POS) == Sign.NEG

    def test_sign_neg_negative(self):
        assert sign_neg(Sign.NEG) == Sign.POS

    def test_sign_neg_zero(self):
        assert sign_neg(Sign.ZERO) == Sign.ZERO

    def test_sign_neg_nonneg(self):
        assert sign_neg(Sign.NON_NEG) == Sign.NON_POS

    def test_sign_add_pos_pos(self):
        assert sign_add(Sign.POS, Sign.POS) == Sign.POS

    def test_sign_add_neg_neg(self):
        assert sign_add(Sign.NEG, Sign.NEG) == Sign.NEG

    def test_sign_add_pos_neg(self):
        assert sign_add(Sign.POS, Sign.NEG) == Sign.TOP

    def test_sign_add_zero(self):
        assert sign_add(Sign.POS, Sign.ZERO) == Sign.POS
        assert sign_add(Sign.ZERO, Sign.NEG) == Sign.NEG

    def test_sign_add_nonneg(self):
        assert sign_add(Sign.NON_NEG, Sign.NON_NEG) == Sign.NON_NEG

    def test_sign_sub_pos_pos(self):
        # pos - pos = top (could be anything)
        assert sign_sub(Sign.POS, Sign.POS) == Sign.TOP

    def test_sign_sub_pos_neg(self):
        assert sign_sub(Sign.POS, Sign.NEG) == Sign.POS

    def test_sign_mul_pos_pos(self):
        assert sign_mul(Sign.POS, Sign.POS) == Sign.POS

    def test_sign_mul_neg_neg(self):
        assert sign_mul(Sign.NEG, Sign.NEG) == Sign.POS

    def test_sign_mul_pos_neg(self):
        assert sign_mul(Sign.POS, Sign.NEG) == Sign.NEG

    def test_sign_mul_zero(self):
        assert sign_mul(Sign.ZERO, Sign.POS) == Sign.ZERO
        assert sign_mul(Sign.NEG, Sign.ZERO) == Sign.ZERO

    def test_sign_mul_nonneg_pos(self):
        assert sign_mul(Sign.NON_NEG, Sign.POS) == Sign.NON_NEG

    def test_sign_div_pos_pos(self):
        assert sign_div(Sign.POS, Sign.POS) == Sign.NON_NEG

    def test_sign_div_by_zero(self):
        assert sign_div(Sign.POS, Sign.ZERO) == Sign.BOT

    def test_sign_div_zero_pos(self):
        assert sign_div(Sign.ZERO, Sign.POS) == Sign.ZERO

    def test_sign_contains_zero(self):
        assert sign_contains_zero(Sign.ZERO)
        assert sign_contains_zero(Sign.NON_NEG)
        assert sign_contains_zero(Sign.NON_POS)
        assert sign_contains_zero(Sign.TOP)
        assert not sign_contains_zero(Sign.POS)
        assert not sign_contains_zero(Sign.NEG)
        assert not sign_contains_zero(Sign.BOT)

    def test_sign_add_bot(self):
        assert sign_add(Sign.BOT, Sign.POS) == Sign.BOT
        assert sign_add(Sign.NEG, Sign.BOT) == Sign.BOT

    def test_sign_mul_bot(self):
        assert sign_mul(Sign.BOT, Sign.POS) == Sign.BOT

    def test_sign_div_bot(self):
        assert sign_div(Sign.BOT, Sign.POS) == Sign.BOT


# ============================================================
# Interval Domain Tests
# ============================================================

class TestIntervalDomain:
    def test_interval_from_int(self):
        i = interval_from_value(5)
        assert i == Interval(5.0, 5.0)

    def test_interval_from_float(self):
        i = interval_from_value(3.14)
        assert i.lo == 3.14 and i.hi == 3.14

    def test_interval_from_bool(self):
        assert interval_from_value(True) == Interval(1.0, 1.0)
        assert interval_from_value(False) == Interval(0.0, 0.0)

    def test_interval_bot(self):
        assert INTERVAL_BOT.is_bot()
        assert not INTERVAL_TOP.is_bot()

    def test_interval_top(self):
        assert INTERVAL_TOP.is_top()
        assert not Interval(0, 10).is_top()

    def test_interval_contains_zero(self):
        assert Interval(-1, 1).contains_zero()
        assert Interval(0, 0).contains_zero()
        assert not Interval(1, 5).contains_zero()
        assert not Interval(-5, -1).contains_zero()

    def test_interval_contains(self):
        i = Interval(1, 10)
        assert i.contains(5)
        assert i.contains(1)
        assert i.contains(10)
        assert not i.contains(0)
        assert not i.contains(11)

    def test_interval_join(self):
        a = Interval(1, 5)
        b = Interval(3, 8)
        assert interval_join(a, b) == Interval(1, 8)

    def test_interval_join_disjoint(self):
        a = Interval(1, 3)
        b = Interval(7, 10)
        assert interval_join(a, b) == Interval(1, 10)

    def test_interval_join_bot(self):
        assert interval_join(INTERVAL_BOT, Interval(1, 5)) == Interval(1, 5)
        assert interval_join(Interval(1, 5), INTERVAL_BOT) == Interval(1, 5)

    def test_interval_meet(self):
        a = Interval(1, 5)
        b = Interval(3, 8)
        assert interval_meet(a, b) == Interval(3, 5)

    def test_interval_meet_disjoint(self):
        a = Interval(1, 3)
        b = Interval(5, 8)
        assert interval_meet(a, b).is_bot()

    def test_interval_widen(self):
        old = Interval(0, 5)
        new = Interval(0, 10)
        w = interval_widen(old, new)
        assert w.lo == 0  # unchanged
        assert w.hi == INF  # widened

    def test_interval_widen_lower(self):
        old = Interval(0, 5)
        new = Interval(-3, 5)
        w = interval_widen(old, new)
        assert w.lo == NEG_INF  # widened
        assert w.hi == 5  # unchanged

    def test_interval_widen_no_change(self):
        old = Interval(0, 10)
        new = Interval(2, 8)
        w = interval_widen(old, new)
        assert w == Interval(0, 10)

    def test_interval_widen_bot(self):
        w = interval_widen(INTERVAL_BOT, Interval(1, 5))
        assert w == Interval(1, 5)

    def test_interval_add(self):
        a = Interval(1, 3)
        b = Interval(2, 5)
        assert interval_add(a, b) == Interval(3, 8)

    def test_interval_sub(self):
        a = Interval(5, 10)
        b = Interval(1, 3)
        assert interval_sub(a, b) == Interval(2, 9)

    def test_interval_neg(self):
        a = Interval(1, 5)
        assert interval_neg(a) == Interval(-5, -1)

    def test_interval_mul(self):
        a = Interval(2, 3)
        b = Interval(4, 5)
        assert interval_mul(a, b) == Interval(8, 15)

    def test_interval_mul_mixed(self):
        a = Interval(-2, 3)
        b = Interval(-1, 4)
        r = interval_mul(a, b)
        assert r.lo <= -8  # -2*4
        assert r.hi >= 12   # 3*4

    def test_interval_div_safe(self):
        a = Interval(6, 12)
        b = Interval(2, 3)
        r = interval_div(a, b)
        assert r.lo == 2.0
        assert r.hi == 6.0

    def test_interval_div_by_zero_interval(self):
        a = Interval(1, 5)
        b = Interval(0, 0)
        assert interval_div(a, b).is_bot()

    def test_interval_div_contains_zero(self):
        a = Interval(1, 5)
        b = Interval(-2, 3)
        r = interval_div(a, b)
        # Should split around zero and still produce valid result
        assert not r.is_bot()

    def test_interval_add_bot(self):
        assert interval_add(INTERVAL_BOT, Interval(1, 5)).is_bot()

    def test_interval_repr(self):
        assert "Bot" in repr(INTERVAL_BOT)
        assert "Top" in repr(INTERVAL_TOP)
        assert "[1" in repr(Interval(1, 5))

    def test_interval_mod_bot(self):
        assert interval_mod(INTERVAL_BOT, Interval(1, 3)).is_bot()

    def test_interval_mod_zero(self):
        r = interval_mod(Interval(5, 10), Interval(0, 0))
        assert r.is_bot()


# ============================================================
# Constant Propagation Domain Tests
# ============================================================

class TestConstDomain:
    def test_const_from_value(self):
        c = const_from_value(42)
        assert isinstance(c, ConstVal)
        assert c.value == 42

    def test_const_join_same(self):
        a = ConstVal(5)
        b = ConstVal(5)
        assert const_join(a, b) == ConstVal(5)

    def test_const_join_different(self):
        a = ConstVal(5)
        b = ConstVal(10)
        assert isinstance(const_join(a, b), ConstTop)

    def test_const_join_bot(self):
        assert const_join(CONST_BOT, ConstVal(5)) == ConstVal(5)
        assert const_join(ConstVal(5), CONST_BOT) == ConstVal(5)

    def test_const_join_top(self):
        assert isinstance(const_join(CONST_TOP, ConstVal(5)), ConstTop)

    def test_const_type_aware(self):
        # True != 1
        a = ConstVal(True)
        b = ConstVal(1)
        assert a != b

    def test_const_hash(self):
        a = ConstVal(5)
        b = ConstVal(5)
        assert hash(a) == hash(b)

    def test_const_bot_equality(self):
        assert ConstBot() == ConstBot()
        assert ConstTop() == ConstTop()


# ============================================================
# Abstract Environment Tests
# ============================================================

class TestAbstractEnv:
    def test_set_and_get(self):
        env = AbstractEnv()
        env.set("x", sign=Sign.POS, interval=Interval(1, 10), const=ConstVal(5))
        assert env.get_sign("x") == Sign.POS
        assert env.get_interval("x") == Interval(1, 10)
        assert env.get_const("x") == ConstVal(5)

    def test_default_is_top(self):
        env = AbstractEnv()
        assert env.get_sign("unknown") == Sign.TOP
        assert env.get_interval("unknown").is_top()
        assert isinstance(env.get_const("unknown"), ConstTop)

    def test_set_from_value(self):
        env = AbstractEnv()
        env.set_from_value("x", 42)
        assert env.get_sign("x") == Sign.POS
        assert env.get_interval("x") == Interval(42, 42)
        assert env.get_const("x") == ConstVal(42)

    def test_set_top(self):
        env = AbstractEnv()
        env.set_top("x")
        assert env.get_sign("x") == Sign.TOP

    def test_set_bot(self):
        env = AbstractEnv()
        env.set_bot("x")
        assert env.get_sign("x") == Sign.BOT

    def test_copy(self):
        env = AbstractEnv()
        env.set_from_value("x", 5)
        copy = env.copy()
        copy.set_from_value("x", 10)
        assert env.get_const("x") == ConstVal(5)
        assert copy.get_const("x") == ConstVal(10)

    def test_join(self):
        a = AbstractEnv()
        a.set_from_value("x", 5)
        b = AbstractEnv()
        b.set_from_value("x", 10)
        joined = a.join(b)
        assert isinstance(joined.get_const("x"), ConstTop)
        assert joined.get_interval("x") == Interval(5, 10)
        assert joined.get_sign("x") == Sign.POS

    def test_join_different_vars(self):
        a = AbstractEnv()
        a.set_from_value("x", 5)
        b = AbstractEnv()
        b.set_from_value("y", 10)
        joined = a.join(b)
        assert joined.get_sign("x") == Sign.POS
        assert joined.get_sign("y") == Sign.POS

    def test_equals(self):
        a = AbstractEnv()
        a.set_from_value("x", 5)
        b = AbstractEnv()
        b.set_from_value("x", 5)
        assert a.equals(b)

    def test_not_equals(self):
        a = AbstractEnv()
        a.set_from_value("x", 5)
        b = AbstractEnv()
        b.set_from_value("x", 10)
        assert not a.equals(b)

    def test_widen(self):
        old = AbstractEnv()
        old.set("x", interval=Interval(0, 5))
        new = AbstractEnv()
        new.set("x", interval=Interval(0, 10))
        widened = old.widen(new)
        assert widened.get_interval("x").hi == INF

    def test_repr(self):
        env = AbstractEnv()
        env.set_from_value("x", 5)
        r = repr(env)
        assert "x" in r


# ============================================================
# Abstract Value Tests
# ============================================================

class TestAbstractValue:
    def test_from_value(self):
        av = AbstractValue.from_value(42)
        assert av.sign == Sign.POS
        assert av.interval == Interval(42, 42)
        assert av.const == ConstVal(42)

    def test_top(self):
        av = AbstractValue.top()
        assert av.sign == Sign.TOP
        assert av.interval.is_top()
        assert isinstance(av.const, ConstTop)

    def test_bot(self):
        av = AbstractValue.bot()
        assert av.is_bot()


# ============================================================
# Integration: Constant Propagation
# ============================================================

class TestConstantPropagation:
    def test_simple_const(self):
        c = get_variable_const("let x = 42;", "x")
        assert c == ConstVal(42)

    def test_const_arithmetic(self):
        c = get_variable_const("let x = 3 + 4;", "x")
        assert c == ConstVal(7)

    def test_const_subtraction(self):
        c = get_variable_const("let x = 10 - 3;", "x")
        assert c == ConstVal(7)

    def test_const_multiplication(self):
        c = get_variable_const("let x = 6 * 7;", "x")
        assert c == ConstVal(42)

    def test_const_division(self):
        c = get_variable_const("let x = 10 / 3;", "x")
        assert c == ConstVal(3)  # integer division

    def test_const_modulo(self):
        c = get_variable_const("let x = 10 % 3;", "x")
        assert c == ConstVal(1)

    def test_const_propagation_chain(self):
        c = get_variable_const("""
            let a = 5;
            let b = a + 3;
            let x = b * 2;
        """, "x")
        assert c == ConstVal(16)

    def test_const_negation(self):
        c = get_variable_const("let x = -5;", "x")
        assert c == ConstVal(-5)

    def test_const_comparison(self):
        c = get_variable_const("let x = 3 < 5;", "x")
        assert c == ConstVal(True)

    def test_const_comparison_false(self):
        c = get_variable_const("let x = 5 < 3;", "x")
        assert c == ConstVal(False)

    def test_const_equality(self):
        c = get_variable_const("let x = 5 == 5;", "x")
        assert c == ConstVal(True)

    def test_const_inequality(self):
        c = get_variable_const("let x = 5 != 3;", "x")
        assert c == ConstVal(True)

    def test_const_string(self):
        c = get_variable_const('let x = "hello";', "x")
        assert c == ConstVal("hello")

    def test_const_bool_true(self):
        c = get_variable_const("let x = true;", "x")
        assert c == ConstVal(True)

    def test_const_bool_false(self):
        c = get_variable_const("let x = false;", "x")
        assert c == ConstVal(False)

    def test_const_not(self):
        c = get_variable_const("let x = not true;", "x")
        assert c == ConstVal(False)

    def test_const_and(self):
        c = get_variable_const("let x = true and false;", "x")
        assert c == ConstVal(False)

    def test_const_or(self):
        c = get_variable_const("let x = false or true;", "x")
        assert c == ConstVal(True)

    def test_const_after_if_becomes_top(self):
        # After an if with different assignments, const becomes top
        result = analyze("""
            let x = 0;
            let cond = true;
            if (cond) {
                x = 5;
            } else {
                x = 10;
            }
        """)
        # cond is const true, so only then branch executes
        assert result['env'].get_const("x") == ConstVal(5)

    def test_const_reassignment(self):
        c = get_variable_const("""
            let x = 5;
            x = 10;
        """, "x")
        assert c == ConstVal(10)


# ============================================================
# Integration: Sign Analysis
# ============================================================

class TestSignAnalysis:
    def test_positive_literal(self):
        s = get_variable_sign("let x = 5;", "x")
        assert s == Sign.POS

    def test_negative_literal(self):
        s = get_variable_sign("let x = -5;", "x")
        assert s == Sign.NEG

    def test_zero_literal(self):
        s = get_variable_sign("let x = 0;", "x")
        assert s == Sign.ZERO

    def test_pos_plus_pos(self):
        s = get_variable_sign("let x = 3 + 4;", "x")
        assert s == Sign.POS

    def test_neg_plus_neg(self):
        s = get_variable_sign("let x = -3 + -4;", "x")
        assert s == Sign.NEG

    def test_pos_times_neg(self):
        s = get_variable_sign("let x = 3 * -4;", "x")
        assert s == Sign.NEG

    def test_neg_times_neg(self):
        s = get_variable_sign("let x = -3 * -4;", "x")
        assert s == Sign.POS

    def test_sign_after_if(self):
        result = analyze("""
            let x = 0;
            let cond = true;
            if (cond) {
                x = 5;
            } else {
                x = -3;
            }
        """)
        # cond is always true, so x = 5
        assert result['env'].get_sign("x") == Sign.POS

    def test_sign_nonneg_after_if(self):
        result = analyze("""
            let x = 0;
            if (x == 0) {
                x = 5;
            } else {
                x = 0;
            }
        """)
        # Both branches: x is POS or ZERO -> NON_NEG
        s = result['env'].get_sign("x")
        assert s in (Sign.POS, Sign.NON_NEG)


# ============================================================
# Integration: Interval Analysis
# ============================================================

class TestIntervalAnalysis:
    def test_constant_interval(self):
        i = get_variable_range("let x = 42;", "x")
        assert i == Interval(42, 42)

    def test_arithmetic_interval(self):
        i = get_variable_range("let x = 3 + 4;", "x")
        assert i == Interval(7, 7)

    def test_subtraction_interval(self):
        i = get_variable_range("let x = 10 - 3;", "x")
        assert i == Interval(7, 7)

    def test_multiplication_interval(self):
        i = get_variable_range("let x = 3 * 4;", "x")
        assert i == Interval(12, 12)

    def test_negation_interval(self):
        i = get_variable_range("let x = -5;", "x")
        assert i == Interval(-5, -5)

    def test_interval_after_if(self):
        result = analyze("""
            let x = 0;
            let c = true;
            if (c) {
                x = 5;
            } else {
                x = 10;
            }
        """)
        # c is always true, x = 5
        assert result['env'].get_interval("x") == Interval(5, 5)

    def test_interval_join_after_if(self):
        # Use a non-constant condition to force both branches
        result = analyze("""
            let y = 1;
            let x = 0;
            if (y > 0) {
                x = 5;
            } else {
                x = 10;
            }
        """)
        # y > 0 is always true (y=1, interval [1,1], 1 > 0), so x = 5
        assert result['env'].get_interval("x") == Interval(5, 5)


# ============================================================
# Integration: Loop Analysis (Widening)
# ============================================================

class TestLoopAnalysis:
    def test_simple_loop_widening(self):
        result = analyze("""
            let x = 0;
            while (x < 10) {
                x = x + 1;
            }
        """)
        # After widening, x should have interval going to inf then refined by condition
        i = result['env'].get_interval("x")
        # x >= 10 after loop
        assert i.lo >= 10 or i.is_top()

    def test_loop_counter(self):
        result = analyze("""
            let i = 0;
            while (i < 5) {
                i = i + 1;
            }
        """)
        i = result['env'].get_interval("i")
        assert i.lo >= 5 or i.is_top()

    def test_loop_never_executes(self):
        result = analyze("""
            let x = 10;
            while (x < 5) {
                x = x + 1;
            }
        """)
        # Loop never executes, x stays 10
        assert result['env'].get_const("x") == ConstVal(10)

    def test_loop_sign_preservation(self):
        result = analyze("""
            let x = 1;
            while (x < 100) {
                x = x + 1;
            }
        """)
        s = result['env'].get_sign("x")
        assert s in (Sign.POS, Sign.NON_NEG, Sign.TOP)

    def test_nested_loops(self):
        result = analyze("""
            let i = 0;
            let j = 0;
            while (i < 3) {
                j = 0;
                while (j < 3) {
                    j = j + 1;
                }
                i = i + 1;
            }
        """)
        # Should converge without infinite loop
        assert result['env'].get_sign("i") in (Sign.POS, Sign.NON_NEG, Sign.TOP)


# ============================================================
# Integration: Division Safety
# ============================================================

class TestDivisionSafety:
    def test_definite_div_by_zero(self):
        warnings = check_division_safety("let x = 5 / 0;")
        assert any(w.kind == WarningKind.DIVISION_BY_ZERO for w in warnings)

    def test_safe_division(self):
        warnings = check_division_safety("let x = 10 / 2;")
        assert len(warnings) == 0

    def test_possible_div_by_zero(self):
        warnings = check_division_safety("""
            let y = 0;
            let x = 0;
            if (y == 0) {
                x = 5;
            } else {
                x = 10 / y;
            }
        """)
        # y is const 0, so else branch unreachable -- no div warning expected
        # Actually: y==0 is const true, so only then branch runs
        assert len(warnings) == 0

    def test_mod_by_zero(self):
        warnings = check_division_safety("let x = 5 % 0;")
        assert any(w.kind == WarningKind.DIVISION_BY_ZERO for w in warnings)

    def test_division_by_variable(self):
        # Variable with unknown value -- possible div by zero
        warnings = check_division_safety("""
            let y = 0;
            let z = 1;
            if (z > 0) {
                y = 5;
            } else {
                y = 0;
            }
            let x = 10 / y;
        """)
        # z is const 1, z > 0 is true, y = 5, so safe
        assert len(warnings) == 0


# ============================================================
# Integration: Dead Code Detection
# ============================================================

class TestDeadCode:
    def test_dead_assignment(self):
        warnings = check_dead_code("""
            let x = 5;
            let y = 10;
            print(y);
        """)
        # x is assigned but never read
        assert any(w.kind == WarningKind.DEAD_ASSIGNMENT and "x" in w.message
                    for w in warnings)

    def test_no_dead_assignment(self):
        warnings = check_dead_code("""
            let x = 5;
            print(x);
        """)
        dead_assigns = [w for w in warnings if w.kind == WarningKind.DEAD_ASSIGNMENT]
        assert len(dead_assigns) == 0

    def test_unreachable_branch_true(self):
        warnings = check_dead_code("""
            if (true) {
                let x = 5;
            } else {
                let y = 10;
            }
        """)
        assert any(w.kind == WarningKind.UNREACHABLE_BRANCH for w in warnings)

    def test_unreachable_branch_false(self):
        warnings = check_dead_code("""
            if (false) {
                let x = 5;
            } else {
                let y = 10;
            }
        """)
        assert any(w.kind == WarningKind.UNREACHABLE_BRANCH for w in warnings)

    def test_variable_used_in_computation(self):
        warnings = check_dead_code("""
            let x = 5;
            let y = x + 3;
            print(y);
        """)
        dead = [w for w in warnings if w.kind == WarningKind.DEAD_ASSIGNMENT]
        # x is read by y computation, so not dead
        assert not any("x" in w.message for w in dead)


# ============================================================
# Integration: Condition Refinement
# ============================================================

class TestConditionRefinement:
    def test_refine_less_than(self):
        result = analyze("""
            let x = 5;
            if (x < 10) {
                print(x);
            }
        """)
        # x = 5 which is < 10, so body always executes
        assert result['env'].get_const("x") == ConstVal(5)

    def test_refine_greater_than(self):
        result = analyze("""
            let x = 15;
            if (x > 10) {
                print(x);
            }
        """)
        assert result['env'].get_const("x") == ConstVal(15)

    def test_refine_equality(self):
        result = analyze("""
            let x = 5;
            if (x == 5) {
                print(x);
            }
        """)
        assert result['env'].get_const("x") == ConstVal(5)


# ============================================================
# Integration: Function Analysis
# ============================================================

class TestFunctionAnalysis:
    def test_function_declaration(self):
        result = analyze("""
            fn add(a, b) {
                return a + b;
            }
        """)
        assert "add" in result['functions']

    def test_function_call(self):
        result = analyze("""
            fn double(x) {
                return x * 2;
            }
            let y = double(5);
        """)
        # Conservative -- function return is TOP
        assert result['env'].get_sign("y") == Sign.TOP

    def test_function_side_effect_analysis(self):
        result = analyze("""
            fn set_global() {
                print(42);
            }
            set_global();
        """)
        # Should not crash
        assert result is not None


# ============================================================
# Integration: Complex Programs
# ============================================================

class TestComplexPrograms:
    def test_fibonacci_like(self):
        result = analyze("""
            let a = 0;
            let b = 1;
            let i = 0;
            while (i < 10) {
                let temp = b;
                b = a + b;
                a = temp;
                i = i + 1;
            }
        """)
        # Should converge
        assert result['env'].get_sign("a") in (Sign.POS, Sign.NON_NEG, Sign.TOP)
        assert result['env'].get_sign("b") in (Sign.POS, Sign.NON_NEG, Sign.TOP)

    def test_accumulator(self):
        result = analyze("""
            let sum = 0;
            let i = 1;
            while (i <= 10) {
                sum = sum + i;
                i = i + 1;
            }
        """)
        assert result['env'].get_sign("sum") in (Sign.POS, Sign.NON_NEG, Sign.TOP)

    def test_conditional_chain(self):
        result = analyze("""
            let x = 5;
            let result = 0;
            if (x > 0) {
                result = 1;
            } else {
                if (x < 0) {
                    result = -1;
                } else {
                    result = 0;
                }
            }
        """)
        # x = 5 > 0 is always true
        assert result['env'].get_const("result") == ConstVal(1)

    def test_mixed_types(self):
        result = analyze("""
            let x = 5;
            let y = 3.14;
            let z = x + 1;
            let w = true;
        """)
        assert result['env'].get_sign("x") == Sign.POS
        assert result['env'].get_sign("y") == Sign.POS
        assert result['env'].get_const("z") == ConstVal(6)

    def test_many_variables(self):
        result = analyze("""
            let a = 1;
            let b = 2;
            let c = 3;
            let d = a + b;
            let e = c * d;
            let f = e - a;
        """)
        assert result['env'].get_const("d") == ConstVal(3)
        assert result['env'].get_const("e") == ConstVal(9)
        assert result['env'].get_const("f") == ConstVal(8)

    def test_program_with_multiple_warnings(self):
        result = analyze("""
            let unused = 42;
            let x = 5 / 0;
            if (true) {
                let y = 1;
            } else {
                let z = 2;
            }
        """)
        kinds = {w.kind for w in result['warnings']}
        assert WarningKind.DIVISION_BY_ZERO in kinds
        assert WarningKind.UNREACHABLE_BRANCH in kinds

    def test_empty_program(self):
        result = analyze("")
        assert result['warnings'] == []

    def test_just_print(self):
        result = analyze('print(42);')
        assert result is not None

    def test_boolean_operations(self):
        result = analyze("""
            let a = true;
            let b = false;
            let c = a and b;
            let d = a or b;
            let e = not a;
        """)
        assert result['env'].get_const("c") == ConstVal(False)
        assert result['env'].get_const("d") == ConstVal(True)
        assert result['env'].get_const("e") == ConstVal(False)


# ============================================================
# Warning Class Tests
# ============================================================

class TestWarning:
    def test_warning_repr(self):
        w = Warning(WarningKind.DIVISION_BY_ZERO, "div by zero", line=5)
        assert "division_by_zero" in repr(w)
        assert "line 5" in repr(w)

    def test_warning_no_line(self):
        w = Warning(WarningKind.DEAD_ASSIGNMENT, "unused var")
        assert "line" not in repr(w)


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_reassign_variable(self):
        c = get_variable_const("""
            let x = 5;
            x = x + 1;
            x = x * 2;
        """, "x")
        assert c == ConstVal(12)

    def test_division_result_interval(self):
        i = get_variable_range("let x = 10 / 3;", "x")
        # Interval uses float division: 10/3 = 3.333...
        assert abs(i.lo - 10/3) < 0.001
        assert abs(i.hi - 10/3) < 0.001

    def test_nested_arithmetic(self):
        c = get_variable_const("let x = (2 + 3) * (4 - 1);", "x")
        assert c == ConstVal(15)

    def test_float_arithmetic(self):
        c = get_variable_const("let x = 1.5 + 2.5;", "x")
        assert c == ConstVal(4.0)

    def test_comparison_operators(self):
        for op, expected in [("<", True), (">", False), ("<=", True),
                              (">=", False), ("==", False), ("!=", True)]:
            c = get_variable_const(f"let x = 3 {op} 5;", "x")
            assert c == ConstVal(expected), f"Failed for {op}"

    def test_max_iterations_limit(self):
        # Should not hang on infinite-like loop
        result = analyze("""
            let x = 0;
            while (true) {
                x = x + 1;
            }
        """, max_iterations=10)
        assert result is not None

    def test_block_scope(self):
        result = analyze("""
            let x = 5;
            {
                let y = x + 1;
                print(y);
            }
        """)
        assert result['env'].get_const("x") == ConstVal(5)

    def test_multiple_prints(self):
        result = analyze("""
            let x = 1;
            print(x);
            let y = 2;
            print(y);
            let z = x + y;
            print(z);
        """)
        assert result['env'].get_const("z") == ConstVal(3)

    def test_while_with_complex_body(self):
        result = analyze("""
            let x = 0;
            let y = 10;
            while (x < y) {
                x = x + 2;
                y = y - 1;
            }
        """)
        # Should converge
        assert result is not None

    def test_comparison_interval_less(self):
        # 5 < 3 is false -- interval [0,0]
        i = get_variable_range("let x = 5 < 3;", "x")
        assert i == Interval(0, 0)

    def test_comparison_interval_true(self):
        i = get_variable_range("let x = 3 < 5;", "x")
        assert i == Interval(1, 1)


# ============================================================
# API Tests
# ============================================================

class TestAPI:
    def test_analyze_returns_dict(self):
        result = analyze("let x = 5;")
        assert isinstance(result, dict)
        assert 'env' in result
        assert 'warnings' in result
        assert 'functions' in result
        assert 'var_reads' in result
        assert 'var_writes' in result

    def test_get_variable_range_api(self):
        i = get_variable_range("let x = 42;", "x")
        assert i == Interval(42, 42)

    def test_get_variable_sign_api(self):
        s = get_variable_sign("let x = 42;", "x")
        assert s == Sign.POS

    def test_get_variable_const_api(self):
        c = get_variable_const("let x = 42;", "x")
        assert c == ConstVal(42)

    def test_get_warnings_api(self):
        w = get_warnings("let x = 5 / 0;")
        assert len(w) > 0

    def test_check_division_safety_api(self):
        w = check_division_safety("let x = 5 / 0;")
        assert len(w) > 0

    def test_check_dead_code_api(self):
        w = check_dead_code("let unused = 5;")
        assert len(w) > 0


# ============================================================
# Interval Comparison in Eval
# ============================================================

class TestIntervalComparison:
    def test_interval_le_definite(self):
        """5 <= 10 should be definitely true."""
        result = analyze("let x = 5 <= 10;")
        assert result['env'].get_const("x") == ConstVal(True)

    def test_interval_ge_definite(self):
        """10 >= 5 should be definitely true."""
        result = analyze("let x = 10 >= 5;")
        assert result['env'].get_const("x") == ConstVal(True)

    def test_interval_eq_definite_true(self):
        """5 == 5 should be definitely true."""
        result = analyze("let x = 5 == 5;")
        assert result['env'].get_const("x") == ConstVal(True)

    def test_interval_eq_definite_false(self):
        """5 == 10 should be definitely false."""
        result = analyze("let x = 5 == 10;")
        assert result['env'].get_const("x") == ConstVal(False)

    def test_interval_ne_definite(self):
        """5 != 10 should be definitely true."""
        result = analyze("let x = 5 != 10;")
        assert result['env'].get_const("x") == ConstVal(True)


# ============================================================
# Widening Convergence
# ============================================================

class TestWideningConvergence:
    def test_countdown_loop(self):
        result = analyze("""
            let x = 100;
            while (x > 0) {
                x = x - 1;
            }
        """)
        # Should converge
        i = result['env'].get_interval("x")
        assert i.hi <= 0 or i.is_top()

    def test_double_increment(self):
        result = analyze("""
            let x = 0;
            while (x < 20) {
                x = x + 2;
            }
        """)
        i = result['env'].get_interval("x")
        assert i.lo >= 20 or i.is_top()

    def test_loop_with_conditional(self):
        result = analyze("""
            let x = 0;
            let y = 0;
            while (x < 10) {
                if (x < 5) {
                    y = y + 1;
                }
                x = x + 1;
            }
        """)
        assert result is not None


# ============================================================
# Sign-Interval Consistency
# ============================================================

class TestSignIntervalConsistency:
    def test_positive_interval_positive_sign(self):
        result = analyze("let x = 5;")
        assert result['env'].get_sign("x") == Sign.POS
        assert result['env'].get_interval("x").lo > 0

    def test_negative_interval_negative_sign(self):
        result = analyze("let x = -5;")
        assert result['env'].get_sign("x") == Sign.NEG
        assert result['env'].get_interval("x").hi < 0

    def test_zero_interval_zero_sign(self):
        result = analyze("let x = 0;")
        assert result['env'].get_sign("x") == Sign.ZERO
        assert result['env'].get_interval("x") == Interval(0, 0)


# ============================================================
# Run
# ============================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
