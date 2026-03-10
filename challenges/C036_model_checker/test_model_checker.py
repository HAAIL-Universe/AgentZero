"""
Tests for C036: Bounded Model Checker
Composes C035 (SAT Solver) + C010 (Stack VM parser)
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from model_checker import (
    check, check_property, verify_safe, find_bug,
    ModelChecker, CheckResult, VerifyResult, Counterexample,
    parse, AssertStmt, AssumeStmt, ModelCheckError,
    Encoder, BitVec, SymbolicState
)
from sat_solver import Solver, SolverResult


# ============================================================
# Section 1: Parser extensions (assert/assume)
# ============================================================

class TestParserExtensions:
    def test_parse_assert(self):
        prog = parse("assert(x > 0);")
        assert len(prog.stmts) == 1
        assert isinstance(prog.stmts[0], AssertStmt)

    def test_parse_assert_with_message(self):
        prog = parse('assert(x > 0, "x must be positive");')
        stmt = prog.stmts[0]
        assert isinstance(stmt, AssertStmt)
        assert stmt.message == "x must be positive"

    def test_parse_assume(self):
        prog = parse("assume(x > 0);")
        assert len(prog.stmts) == 1
        assert isinstance(prog.stmts[0], AssumeStmt)

    def test_parse_mixed(self):
        prog = parse("""
            let x = 5;
            assume(x > 0);
            let y = x + 1;
            assert(y > 1);
        """)
        assert len(prog.stmts) == 4

    def test_parse_assert_complex_expr(self):
        prog = parse("assert(x + y == z * 2);")
        stmt = prog.stmts[0]
        assert isinstance(stmt, AssertStmt)

    def test_parse_assert_line_number(self):
        prog = parse("let x = 1;\nassert(x > 0);")
        assert prog.stmts[1].line == 2


# ============================================================
# Section 2: Encoder basics (bit-vector operations)
# ============================================================

class TestEncoder:
    def test_const_bitvec(self):
        solver = Solver()
        enc = Encoder(solver, bit_width=8)
        bv = enc.const_bitvec(5)
        assert bv.width == 8
        result = solver.solve()
        assert result == SolverResult.SAT
        model = solver.model()
        val = enc.extract_value(bv, model)
        assert val == 5

    def test_const_bitvec_negative(self):
        solver = Solver()
        enc = Encoder(solver, bit_width=8)
        bv = enc.const_bitvec(-3)
        result = solver.solve()
        assert result == SolverResult.SAT
        model = solver.model()
        val = enc.extract_value(bv, model)
        assert val == -3

    def test_const_bitvec_zero(self):
        solver = Solver()
        enc = Encoder(solver, bit_width=8)
        bv = enc.const_bitvec(0)
        result = solver.solve()
        model = solver.model()
        assert enc.extract_value(bv, model) == 0

    def test_add(self):
        solver = Solver()
        enc = Encoder(solver, bit_width=8)
        a = enc.const_bitvec(3)
        b = enc.const_bitvec(4)
        c = enc.bv_add(a, b)
        result = solver.solve()
        assert result == SolverResult.SAT
        model = solver.model()
        assert enc.extract_value(c, model) == 7

    def test_sub(self):
        solver = Solver()
        enc = Encoder(solver, bit_width=8)
        a = enc.const_bitvec(10)
        b = enc.const_bitvec(3)
        c = enc.bv_sub(a, b)
        result = solver.solve()
        model = solver.model()
        assert enc.extract_value(c, model) == 7

    def test_mul(self):
        solver = Solver()
        enc = Encoder(solver, bit_width=8)
        a = enc.const_bitvec(3)
        b = enc.const_bitvec(5)
        c = enc.bv_mul(a, b)
        result = solver.solve()
        model = solver.model()
        assert enc.extract_value(c, model) == 15

    def test_neg(self):
        solver = Solver()
        enc = Encoder(solver, bit_width=8)
        a = enc.const_bitvec(5)
        b = enc.bv_neg(a)
        result = solver.solve()
        model = solver.model()
        assert enc.extract_value(b, model) == -5

    def test_eq_true(self):
        solver = Solver()
        enc = Encoder(solver, bit_width=8)
        a = enc.const_bitvec(7)
        b = enc.const_bitvec(7)
        eq = enc.bv_eq(a, b)
        result = solver.solve()
        model = solver.model()
        assert enc.extract_bool(eq, model) == True

    def test_eq_false(self):
        solver = Solver()
        enc = Encoder(solver, bit_width=8)
        a = enc.const_bitvec(7)
        b = enc.const_bitvec(8)
        eq = enc.bv_eq(a, b)
        # Force eq to be true -- should be UNSAT
        solver.add_clause([eq])
        result = solver.solve()
        assert result == SolverResult.UNSAT

    def test_slt(self):
        solver = Solver()
        enc = Encoder(solver, bit_width=8)
        a = enc.const_bitvec(3)
        b = enc.const_bitvec(5)
        lt = enc.bv_slt(a, b)
        result = solver.solve()
        model = solver.model()
        assert enc.extract_bool(lt, model) == True

    def test_slt_negative(self):
        solver = Solver()
        enc = Encoder(solver, bit_width=8)
        a = enc.const_bitvec(-3)
        b = enc.const_bitvec(5)
        lt = enc.bv_slt(a, b)
        result = solver.solve()
        model = solver.model()
        assert enc.extract_bool(lt, model) == True

    def test_slt_false(self):
        solver = Solver()
        enc = Encoder(solver, bit_width=8)
        a = enc.const_bitvec(5)
        b = enc.const_bitvec(3)
        lt = enc.bv_slt(a, b)
        result = solver.solve()
        model = solver.model()
        assert enc.extract_bool(lt, model) == False

    def test_ite_bitvec(self):
        solver = Solver()
        enc = Encoder(solver, bit_width=8)
        cond = enc.const_bool(True)
        a = enc.const_bitvec(10)
        b = enc.const_bitvec(20)
        r = enc.bv_ite(cond, a, b)
        result = solver.solve()
        model = solver.model()
        assert enc.extract_value(r, model) == 10

    def test_ite_bitvec_false(self):
        solver = Solver()
        enc = Encoder(solver, bit_width=8)
        cond = enc.const_bool(False)
        a = enc.const_bitvec(10)
        b = enc.const_bitvec(20)
        r = enc.bv_ite(cond, a, b)
        result = solver.solve()
        model = solver.model()
        assert enc.extract_value(r, model) == 20

    def test_bool_and(self):
        solver = Solver()
        enc = Encoder(solver, bit_width=8)
        a = enc.const_bool(True)
        b = enc.const_bool(False)
        r = enc.bool_and(a, b)
        result = solver.solve()
        model = solver.model()
        assert enc.extract_bool(r, model) == False

    def test_bool_or(self):
        solver = Solver()
        enc = Encoder(solver, bit_width=8)
        a = enc.const_bool(False)
        b = enc.const_bool(True)
        r = enc.bool_or(a, b)
        result = solver.solve()
        model = solver.model()
        assert enc.extract_bool(r, model) == True

    def test_bool_not(self):
        solver = Solver()
        enc = Encoder(solver, bit_width=8)
        a = enc.const_bool(True)
        r = enc.bool_not(a)
        result = solver.solve()
        model = solver.model()
        assert enc.extract_bool(r, model) == False

    def test_free_variable(self):
        """A free variable can take any value."""
        solver = Solver()
        enc = Encoder(solver, bit_width=8)
        x = enc.new_bitvec()
        # Constrain x == 42
        c42 = enc.const_bitvec(42)
        eq = enc.bv_eq(x, c42)
        solver.add_clause([eq])
        result = solver.solve()
        assert result == SolverResult.SAT
        model = solver.model()
        assert enc.extract_value(x, model) == 42

    def test_overflow_wraps(self):
        """8-bit addition wraps: 127 + 1 = -128."""
        solver = Solver()
        enc = Encoder(solver, bit_width=8)
        a = enc.const_bitvec(127)
        b = enc.const_bitvec(1)
        c = enc.bv_add(a, b)
        result = solver.solve()
        model = solver.model()
        assert enc.extract_value(c, model) == -128


# ============================================================
# Section 3: Simple safe programs
# ============================================================

class TestSafePrograms:
    def test_simple_assert_true(self):
        result = check("let x = 5; assert(x == 5);")
        assert result.verdict == VerifyResult.SAFE

    def test_addition_safe(self):
        result = check("let x = 3; let y = 4; assert(x + y == 7);")
        assert result.verdict == VerifyResult.SAFE

    def test_subtraction_safe(self):
        result = check("let x = 10; let y = 3; assert(x - y == 7);")
        assert result.verdict == VerifyResult.SAFE

    def test_comparison_safe(self):
        result = check("let x = 5; assert(x > 3);")
        assert result.verdict == VerifyResult.SAFE

    def test_multiple_asserts_all_safe(self):
        result = check("""
            let x = 10;
            assert(x > 0);
            assert(x < 20);
            assert(x == 10);
        """)
        assert result.verdict == VerifyResult.SAFE
        assert result.assertions_checked == 3

    def test_no_assertions(self):
        result = check("let x = 5;")
        assert result.verdict == VerifyResult.SAFE
        assert result.assertions_checked == 0

    def test_boolean_true(self):
        result = check("assert(true);")
        assert result.verdict == VerifyResult.SAFE

    def test_negation_safe(self):
        result = check("let x = 5; assert(0 - x == 0 - 5);")
        assert result.verdict == VerifyResult.SAFE

    def test_multiplication_safe(self):
        result = check("let x = 3; let y = 4; assert(x * y == 12);", bit_width=6)
        assert result.verdict == VerifyResult.SAFE

    def test_le_safe(self):
        result = check("let x = 5; assert(x <= 5);")
        assert result.verdict == VerifyResult.SAFE

    def test_ge_safe(self):
        result = check("let x = 5; assert(x >= 5);")
        assert result.verdict == VerifyResult.SAFE

    def test_ne_safe(self):
        result = check("let x = 5; assert(x != 4);")
        assert result.verdict == VerifyResult.SAFE

    def test_verify_safe_convenience(self):
        assert verify_safe("let x = 1; assert(x == 1);") == True

    def test_find_bug_returns_none_when_safe(self):
        assert find_bug("let x = 1; assert(x == 1);") is None


# ============================================================
# Section 4: Unsafe programs (assertion violations)
# ============================================================

class TestUnsafePrograms:
    def test_simple_violation(self):
        result = check("let x = 5; assert(x == 6);")
        assert result.verdict == VerifyResult.UNSAFE
        assert len(result.counterexamples) == 1

    def test_violation_counterexample(self):
        result = check("let x = 5; assert(x > 10);")
        assert result.verdict == VerifyResult.UNSAFE
        ce = result.counterexamples[0]
        assert ce.variables['x'] == 5

    def test_boolean_false(self):
        result = check("assert(false);")
        assert result.verdict == VerifyResult.UNSAFE

    def test_addition_overflow_violation(self):
        """8-bit: 100 + 100 = -56 (overflow), so < 0."""
        result = check("let x = 100; let y = 100; assert(x + y > 0);", bit_width=8)
        assert result.verdict == VerifyResult.UNSAFE

    def test_multiple_asserts_one_fails(self):
        result = check("""
            let x = 5;
            assert(x > 0);
            assert(x > 10);
        """)
        assert result.verdict == VerifyResult.UNSAFE
        assert result.assertions_checked == 2

    def test_find_bug_returns_counterexample(self):
        ce = find_bug("let x = 5; assert(x == 0);")
        assert ce is not None
        assert ce.variables['x'] == 5

    def test_assert_with_message(self):
        result = check('let x = 0; assert(x > 0, "x must be positive");')
        assert result.verdict == VerifyResult.UNSAFE
        assert result.counterexamples[0].assertion_msg == "x must be positive"


# ============================================================
# Section 5: Symbolic inputs (free variables)
# ============================================================

class TestSymbolicInputs:
    def test_free_var_can_violate(self):
        """Free variable x: assert(x > 0) should be UNSAFE since x can be <= 0."""
        result = check("assert(x > 0);")
        assert result.verdict == VerifyResult.UNSAFE
        ce = result.counterexamples[0]
        assert ce.variables['x'] <= 0

    def test_free_var_with_assume(self):
        """With assume(x > 0), assert(x > 0) should be SAFE."""
        result = check("assume(x > 0); assert(x > 0);")
        assert result.verdict == VerifyResult.SAFE

    def test_free_var_constrained(self):
        """assume(x == 5); assert(x + 1 == 6) should be SAFE."""
        result = check("assume(x == 5); assert(x + 1 == 6);")
        assert result.verdict == VerifyResult.SAFE

    def test_two_free_vars(self):
        """x + y == y + x should always hold (commutativity)."""
        result = check("assert(x + y == y + x);")
        assert result.verdict == VerifyResult.SAFE

    def test_free_var_range(self):
        """assume(x >= 0 and x < 10); assert(x < 10) should be SAFE."""
        result = check("""
            assume(x >= 0);
            assume(x < 10);
            assert(x < 10);
        """)
        assert result.verdict == VerifyResult.SAFE

    def test_check_property_convenience(self):
        result = check_property("let x = 5;", "x == 5")
        assert result.verdict == VerifyResult.SAFE

    def test_check_property_violation(self):
        result = check_property("let x = 5;", "x == 6")
        assert result.verdict == VerifyResult.UNSAFE

    def test_free_var_arithmetic(self):
        """For any x: x + 0 == x."""
        result = check("assert(x + 0 == x);")
        assert result.verdict == VerifyResult.SAFE

    def test_free_var_double(self):
        """For any x: x + x == x * 2 (4-bit for tractability)."""
        result = check("assert(x + x == x * 2);", bit_width=4)
        assert result.verdict == VerifyResult.SAFE


# ============================================================
# Section 6: If/else control flow
# ============================================================

class TestIfElse:
    def test_if_then_safe(self):
        result = check("""
            let x = 5;
            let y = 0;
            if (x > 3) {
                y = 1;
            }
            assert(y == 1);
        """)
        assert result.verdict == VerifyResult.SAFE

    def test_if_else_safe(self):
        result = check("""
            let x = 5;
            let y = 0;
            if (x > 10) {
                y = 1;
            } else {
                y = 2;
            }
            assert(y == 2);
        """)
        assert result.verdict == VerifyResult.SAFE

    def test_if_symbolic_safe(self):
        """If x > 0 then y = 1 else y = 2. assert(y == 1 or y == 2)."""
        result = check("""
            let y = 0;
            if (x > 0) {
                y = 1;
            } else {
                y = 2;
            }
            assert(y == 1 or y == 2);
        """)
        assert result.verdict == VerifyResult.SAFE

    def test_if_symbolic_unsafe(self):
        """If x > 0 then y = 1 else y = 2. assert(y == 1) is UNSAFE."""
        result = check("""
            let y = 0;
            if (x > 0) {
                y = 1;
            } else {
                y = 2;
            }
            assert(y == 1);
        """)
        assert result.verdict == VerifyResult.UNSAFE

    def test_nested_if(self):
        result = check("""
            let y = 0;
            if (x > 0) {
                if (x > 5) {
                    y = 2;
                } else {
                    y = 1;
                }
            } else {
                y = 0;
            }
            assert(y >= 0);
        """)
        assert result.verdict == VerifyResult.SAFE

    def test_if_assert_in_then_branch(self):
        result = check("""
            let x = 5;
            if (x > 0) {
                assert(x > 0);
            }
        """)
        assert result.verdict == VerifyResult.SAFE

    def test_if_assert_in_else_branch(self):
        result = check("""
            let x = 0;
            if (x > 0) {
                let y = 1;
            } else {
                assert(x <= 0);
            }
        """)
        assert result.verdict == VerifyResult.SAFE


# ============================================================
# Section 7: While loops (bounded unrolling)
# ============================================================

class TestWhileLoops:
    def test_simple_loop_safe(self):
        result = check("""
            let x = 0;
            let i = 0;
            while (i < 3) {
                x = x + 1;
                i = i + 1;
            }
            assert(x == 3);
        """, loop_bound=5)
        assert result.verdict == VerifyResult.SAFE

    def test_loop_accumulator(self):
        result = check("""
            let sum = 0;
            let i = 1;
            while (i <= 5) {
                sum = sum + i;
                i = i + 1;
            }
            assert(sum == 15);
        """, loop_bound=10)
        assert result.verdict == VerifyResult.SAFE

    def test_loop_assert_inside(self):
        """Assert inside loop body: i should always be >= 0."""
        result = check("""
            let i = 0;
            while (i < 5) {
                assert(i >= 0);
                i = i + 1;
            }
        """, loop_bound=10)
        assert result.verdict == VerifyResult.SAFE

    def test_loop_overflow_unsafe(self):
        """8-bit counter will overflow past 127."""
        result = check("""
            let x = 100;
            let i = 0;
            while (i < 5) {
                x = x + 10;
                i = i + 1;
            }
            assert(x > 0);
        """, bit_width=8, loop_bound=10)
        # x = 150 overflows in 8-bit signed -> negative
        assert result.verdict == VerifyResult.UNSAFE


# ============================================================
# Section 8: Assume + assert patterns
# ============================================================

class TestAssumeAssert:
    def test_assume_constrains_input(self):
        """4-bit signed: -8 to 7, so x<=3 ensures x*2 <= 6 (no overflow)."""
        result = check("""
            assume(x >= 1);
            assume(x <= 3);
            let y = x * 2;
            assert(y >= 2);
        """, bit_width=4)
        assert result.verdict == VerifyResult.SAFE

    def test_assume_constrains_multiple(self):
        result = check("""
            assume(x >= 0);
            assume(y >= 0);
            assume(x + y < 50);
            assert(x + y < 50);
        """)
        assert result.verdict == VerifyResult.SAFE

    def test_assume_makes_reachable_path_safe(self):
        """x > 0 and x < 100 prevents overflow, so x+1 > 1."""
        result = check("""
            assume(x > 0);
            assume(x < 100);
            let y = x + 1;
            assert(y > 1);
        """)
        assert result.verdict == VerifyResult.SAFE

    def test_assume_before_assert_in_branch(self):
        result = check("""
            assume(x >= 0);
            if (x > 5) {
                assert(x > 5);
            }
        """)
        assert result.verdict == VerifyResult.SAFE


# ============================================================
# Section 9: Bit-width sensitivity
# ============================================================

class TestBitWidth:
    def test_4bit_range(self):
        """4-bit signed: -8 to 7."""
        result = check("let x = 7; assert(x == 7);", bit_width=4)
        assert result.verdict == VerifyResult.SAFE

    def test_4bit_overflow(self):
        """4-bit: 7 + 1 = -8 (overflow)."""
        result = check("let x = 7; let y = x + 1; assert(y > 0);", bit_width=4)
        assert result.verdict == VerifyResult.UNSAFE

    def test_different_widths_same_result(self):
        """Small values should behave the same at any width."""
        for w in [4, 6, 8]:
            result = check("let x = 3; assert(x + 2 == 5);", bit_width=w)
            assert result.verdict == VerifyResult.SAFE

    def test_16bit(self):
        """16-bit supports larger values."""
        result = check("let x = 1000; assert(x == 1000);", bit_width=16)
        assert result.verdict == VerifyResult.SAFE


# ============================================================
# Section 10: Mathematical properties
# ============================================================

class TestMathProperties:
    """Use 4-bit width for symbolic math to keep SAT encoding tractable."""

    def test_commutativity_add(self):
        result = check("assert(x + y == y + x);", bit_width=4)
        assert result.verdict == VerifyResult.SAFE

    def test_commutativity_mul(self):
        result = check("assert(x * y == y * x);", bit_width=4)
        assert result.verdict == VerifyResult.SAFE

    def test_identity_add(self):
        result = check("assert(x + 0 == x);", bit_width=4)
        assert result.verdict == VerifyResult.SAFE

    def test_identity_mul(self):
        result = check("assert(x * 1 == x);", bit_width=4)
        assert result.verdict == VerifyResult.SAFE

    def test_zero_mul(self):
        result = check("assert(x * 0 == 0);", bit_width=4)
        assert result.verdict == VerifyResult.SAFE

    def test_double_negation(self):
        result = check("assert(0 - (0 - x) == x);", bit_width=4)
        assert result.verdict == VerifyResult.SAFE

    def test_distributivity(self):
        """x * (y + z) == x * y + x * z (4-bit for tractability)"""
        result = check("""
            let a = x * (y + z);
            let b = x * y + x * z;
            assert(a == b);
        """, bit_width=4)
        assert result.verdict == VerifyResult.SAFE

    def test_associativity_add(self):
        result = check("assert((x + y) + z == x + (y + z));", bit_width=4)
        assert result.verdict == VerifyResult.SAFE


# ============================================================
# Section 11: Logical operators
# ============================================================

class TestLogical:
    def test_and_both_true(self):
        result = check("let x = 1; let y = 1; assert(x and y);")
        assert result.verdict == VerifyResult.SAFE

    def test_and_one_false(self):
        result = check("let x = 1; let y = 0; assert(x and y);")
        assert result.verdict == VerifyResult.UNSAFE

    def test_or_one_true(self):
        result = check("let x = 0; let y = 1; assert(x or y);")
        assert result.verdict == VerifyResult.SAFE

    def test_or_both_false(self):
        result = check("let x = 0; let y = 0; assert(x or y);")
        assert result.verdict == VerifyResult.UNSAFE

    def test_not_false_is_true(self):
        result = check("let x = 0; assert(not x);")
        assert result.verdict == VerifyResult.SAFE

    def test_not_true_is_false(self):
        result = check("let x = 1; assert(not x);")
        assert result.verdict == VerifyResult.UNSAFE


# ============================================================
# Section 12: Complex programs
# ============================================================

class TestComplexPrograms:
    def test_abs_value(self):
        """Absolute value should be >= 0 (excluding MIN_INT edge case)."""
        result = check("""
            assume(x > 0 - 127);
            let r = 0;
            if (x >= 0) {
                r = x;
            } else {
                r = 0 - x;
            }
            assert(r >= 0);
        """)
        assert result.verdict == VerifyResult.SAFE

    def test_max_of_two(self):
        """max(x, y) >= x and max(x, y) >= y."""
        result = check("""
            let m = 0;
            if (x > y) {
                m = x;
            } else {
                m = y;
            }
            assert(m >= x);
        """)
        assert result.verdict == VerifyResult.SAFE

    def test_max_ge_both(self):
        result = check("""
            let m = 0;
            if (x > y) {
                m = x;
            } else {
                m = y;
            }
            assert(m >= y);
        """)
        assert result.verdict == VerifyResult.SAFE

    def test_swap(self):
        """Swap two variables and check."""
        result = check("""
            assume(x == 3);
            assume(y == 7);
            let t = x;
            x = y;
            y = t;
            assert(x == 7);
        """)
        assert result.verdict == VerifyResult.SAFE

    def test_swap_y(self):
        result = check("""
            assume(x == 3);
            assume(y == 7);
            let t = x;
            x = y;
            y = t;
            assert(y == 3);
        """)
        assert result.verdict == VerifyResult.SAFE

    def test_clamp(self):
        """Clamping x to [0, 10] keeps it in range."""
        result = check("""
            let r = x;
            if (r < 0) {
                r = 0;
            }
            if (r > 10) {
                r = 10;
            }
            assert(r >= 0);
        """)
        assert result.verdict == VerifyResult.SAFE

    def test_clamp_upper(self):
        result = check("""
            let r = x;
            if (r < 0) {
                r = 0;
            }
            if (r > 10) {
                r = 10;
            }
            assert(r <= 10);
        """)
        assert result.verdict == VerifyResult.SAFE

    def test_fibonacci_first_few(self):
        """Compute fib(4) = 3 via loop (smaller for tractability)."""
        result = check("""
            let a = 0;
            let b = 1;
            let i = 0;
            while (i < 4) {
                let t = b;
                b = a + b;
                a = t;
                i = i + 1;
            }
            assert(a == 3);
        """, loop_bound=6)
        assert result.verdict == VerifyResult.SAFE


# ============================================================
# Section 13: Edge cases
# ============================================================

class TestEdgeCases:
    def test_empty_program(self):
        result = check("")
        assert result.verdict == VerifyResult.SAFE

    def test_only_lets(self):
        result = check("let x = 1; let y = 2;")
        assert result.verdict == VerifyResult.SAFE

    def test_reassignment(self):
        result = check("""
            let x = 5;
            x = 10;
            assert(x == 10);
        """)
        assert result.verdict == VerifyResult.SAFE

    def test_chained_assignment(self):
        result = check("""
            let x = 1;
            let y = x;
            let z = y;
            assert(z == 1);
        """)
        assert result.verdict == VerifyResult.SAFE

    def test_print_ignored(self):
        result = check("""
            let x = 5;
            print(x);
            assert(x == 5);
        """)
        assert result.verdict == VerifyResult.SAFE

    def test_bool_literal_true(self):
        result = check("let x = true; assert(x);")
        assert result.verdict == VerifyResult.SAFE

    def test_bool_literal_false(self):
        result = check("let x = false; assert(x);")
        assert result.verdict == VerifyResult.UNSAFE

    def test_multiple_assertions_first_fails(self):
        """When first assertion fails, it should still check others."""
        result = check("""
            let x = 5;
            assert(x == 0);
            assert(x == 5);
        """)
        assert result.verdict == VerifyResult.UNSAFE
        assert result.assertions_checked == 2

    def test_negative_literal(self):
        result = check("let x = 0 - 3; assert(x == 0 - 3);")
        assert result.verdict == VerifyResult.SAFE


# ============================================================
# Section 14: Counterexample quality
# ============================================================

class TestCounterexamples:
    def test_counterexample_has_variables(self):
        result = check("let x = 5; let y = 3; assert(x + y == 0);")
        assert result.verdict == VerifyResult.UNSAFE
        ce = result.counterexamples[0]
        assert 'x' in ce.variables
        assert 'y' in ce.variables

    def test_counterexample_values_consistent(self):
        result = check("let x = 5; assert(x > 10);")
        ce = result.counterexamples[0]
        assert ce.variables['x'] == 5

    def test_free_var_counterexample(self):
        result = check("assume(x >= 0); assume(x <= 5); assert(x == 3);")
        assert result.verdict == VerifyResult.UNSAFE
        ce = result.counterexamples[0]
        x = ce.variables['x']
        assert 0 <= x <= 5
        assert x != 3

    def test_counterexample_line_info(self):
        result = check("let x = 0;\nassert(x > 0);")
        assert result.verdict == VerifyResult.UNSAFE
        assert result.counterexamples[0].assertion_line == 2


# ============================================================
# Section 15: Loop invariants via bounded checking
# ============================================================

class TestLoopInvariants:
    def test_counter_stays_nonneg(self):
        result = check("""
            let i = 0;
            while (i < 5) {
                assert(i >= 0);
                i = i + 1;
            }
        """, loop_bound=8)
        assert result.verdict == VerifyResult.SAFE

    def test_sum_increases(self):
        result = check("""
            let sum = 0;
            let i = 0;
            while (i < 3) {
                let old = sum;
                sum = sum + 1;
                assert(sum > old);
                i = i + 1;
            }
        """, loop_bound=5)
        assert result.verdict == VerifyResult.SAFE

    def test_product_nonneg(self):
        """Product of non-negative values stays non-negative (4-bit for mul)."""
        result = check("""
            let p = 1;
            let i = 1;
            while (i <= 2) {
                p = p * i;
                assert(p > 0);
                i = i + 1;
            }
        """, bit_width=4, loop_bound=4)
        assert result.verdict == VerifyResult.SAFE


# ============================================================
# Section 16: Result metadata
# ============================================================

class TestMetadata:
    def test_result_has_bound(self):
        result = check("assert(true);", loop_bound=5)
        assert result.bound == 5

    def test_result_has_bit_width(self):
        result = check("assert(true);", bit_width=4)
        assert result.bit_width == 4

    def test_assertion_count(self):
        result = check("""
            assert(true);
            assert(true);
            assert(true);
        """)
        assert result.assertions_checked == 3


# ============================================================
# Section 17: Sequential assert dependencies
# ============================================================

class TestSequentialAsserts:
    def test_earlier_assert_constrains_later(self):
        """Earlier assertions act as assumes for later ones."""
        result = check("""
            let x = 5;
            assert(x > 0);
            let y = x + 1;
            assert(y > 1);
        """)
        assert result.verdict == VerifyResult.SAFE

    def test_conditional_then_assert(self):
        result = check("""
            let x = 0;
            if (true) {
                x = 42;
            }
            assert(x == 42);
        """)
        assert result.verdict == VerifyResult.SAFE


# ============================================================
# Section 18: Modular arithmetic
# ============================================================

class TestBitwiseOps:
    def test_add_sub_inverse(self):
        """x + y - y == x for any x, y."""
        result = check("assert(x + y - y == x);", bit_width=4)
        assert result.verdict == VerifyResult.SAFE

    def test_sub_self_is_zero(self):
        result = check("assert(x - x == 0);", bit_width=4)
        assert result.verdict == VerifyResult.SAFE


# ============================================================
# Section 19: Composition verification (SAT solver properties)
# ============================================================

class TestComposition:
    def test_encoder_and_solver_compose(self):
        """Verify the encoder correctly uses the solver API."""
        solver = Solver()
        enc = Encoder(solver, bit_width=4)
        x = enc.new_bitvec()
        # x > 5 (unsigned)
        five = enc.const_bitvec(5)
        gt = enc.bv_sgt(x, five)
        solver.add_clause([gt])
        result = solver.solve()
        assert result == SolverResult.SAT
        model = solver.model()
        val = enc.extract_value(x, model)
        assert val > 5 or val < 0  # In 4-bit signed, 6,7 > 5; or negative wraps

    def test_multiple_constraints(self):
        solver = Solver()
        enc = Encoder(solver, bit_width=8)
        x = enc.new_bitvec()
        # x >= 3
        three = enc.const_bitvec(3)
        ge = enc.bv_sge(x, three)
        solver.add_clause([ge])
        # x <= 5
        five = enc.const_bitvec(5)
        le = enc.bv_sle(x, five)
        solver.add_clause([le])
        result = solver.solve()
        assert result == SolverResult.SAT
        model = solver.model()
        val = enc.extract_value(x, model)
        assert 3 <= val <= 5


# ============================================================
# Section 20: Stress / larger programs
# ============================================================

class TestStress:
    def test_many_variables(self):
        """Program with many variables."""
        lines = []
        for i in range(10):
            lines.append(f"let v{i} = {i};")
        lines.append("assert(v0 + v9 == 9);")
        result = check("\n".join(lines))
        assert result.verdict == VerifyResult.SAFE

    def test_deep_nesting(self):
        result = check("""
            let x = 10;
            if (x > 5) {
                if (x > 7) {
                    if (x > 9) {
                        assert(x > 9);
                    }
                }
            }
        """)
        assert result.verdict == VerifyResult.SAFE

    def test_chain_of_computations(self):
        result = check("""
            let a = 1;
            let b = a + 1;
            let c = b + 1;
            let d = c + 1;
            let e = d + 1;
            assert(e == 5);
        """)
        assert result.verdict == VerifyResult.SAFE


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
