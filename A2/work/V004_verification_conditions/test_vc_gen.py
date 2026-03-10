"""
Tests for V004: Verification Condition Generation
"""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from vc_gen import (
    verify_function, verify_program, verify_hoare_triple,
    VCStatus, VCResult, VerificationResult,
    SVar, SInt, SBool, SBinOp, SUnaryOp, SImplies, SAnd, SOr, SNot,
    s_and, s_or, s_not, s_implies, substitute, ast_to_sexpr,
    WPCalculus, check_vc, extract_fn_spec, lower_to_smt,
    SMTSolver,
)

# ============================================================
# Section 1: Symbolic Expression Construction
# ============================================================

class TestSymbolicExpressions:
    def test_svar(self):
        x = SVar('x')
        assert x.name == 'x'
        assert repr(x) == 'x'

    def test_sint(self):
        n = SInt(42)
        assert n.value == 42

    def test_sbool(self):
        t = SBool(True)
        f = SBool(False)
        assert t.value is True
        assert f.value is False

    def test_sbinop(self):
        expr = SBinOp('+', SVar('x'), SInt(1))
        assert expr.op == '+'

    def test_sunaryop(self):
        expr = SUnaryOp('-', SVar('x'))
        assert expr.op == '-'

    def test_s_and_simplification(self):
        # True AND x = x
        assert s_and(SBool(True), SVar('x')) == SVar('x')
        # False AND x = False
        assert s_and(SBool(False), SVar('x')) == SBool(False)
        # Empty = True
        assert s_and() == SBool(True)

    def test_s_or_simplification(self):
        assert s_or(SBool(True), SVar('x')) == SBool(True)
        assert s_or(SBool(False), SVar('x')) == SVar('x')
        assert s_or() == SBool(False)

    def test_s_not_simplification(self):
        assert s_not(SBool(True)) == SBool(False)
        assert s_not(SBool(False)) == SBool(True)
        assert s_not(SNot(SVar('x'))) == SVar('x')  # double negation

    def test_s_implies_simplification(self):
        assert s_implies(SBool(True), SVar('x')) == SVar('x')
        assert s_implies(SBool(False), SVar('x')) == SBool(True)
        assert s_implies(SVar('x'), SBool(True)) == SBool(True)


# ============================================================
# Section 2: Substitution
# ============================================================

class TestSubstitution:
    def test_substitute_var(self):
        expr = SVar('x')
        result = substitute(expr, 'x', SInt(5))
        assert result == SInt(5)

    def test_substitute_other_var(self):
        expr = SVar('y')
        result = substitute(expr, 'x', SInt(5))
        assert result == SVar('y')

    def test_substitute_in_binop(self):
        expr = SBinOp('+', SVar('x'), SInt(1))
        result = substitute(expr, 'x', SInt(5))
        assert result == SBinOp('+', SInt(5), SInt(1))

    def test_substitute_in_implies(self):
        expr = SImplies(SVar('x'), SVar('y'))
        result = substitute(expr, 'x', SBool(True))
        assert result == SImplies(SBool(True), SVar('y'))

    def test_substitute_in_and(self):
        expr = SAnd((SVar('x'), SVar('y')))
        result = substitute(expr, 'x', SBool(True))
        assert result == SAnd((SBool(True), SVar('y')))

    def test_substitute_in_not(self):
        expr = SNot(SVar('x'))
        result = substitute(expr, 'x', SBool(False))
        assert result == SNot(SBool(False))

    def test_nested_substitution(self):
        # (x + 1) > y => substitute x with (a + b)
        inner = SBinOp('+', SVar('x'), SInt(1))
        expr = SBinOp('>', inner, SVar('y'))
        replacement = SBinOp('+', SVar('a'), SVar('b'))
        result = substitute(expr, 'x', replacement)
        expected = SBinOp('>', SBinOp('+', replacement, SInt(1)), SVar('y'))
        assert result == expected

    def test_substitute_in_ite(self):
        from vc_gen import SIte
        expr = SIte(SVar('c'), SVar('x'), SVar('y'))
        result = substitute(expr, 'x', SInt(10))
        assert result == SIte(SVar('c'), SInt(10), SVar('y'))


# ============================================================
# Section 3: AST to SExpr conversion
# ============================================================

class TestASTConversion:
    def test_int_lit(self):
        from stack_vm import IntLit
        result = ast_to_sexpr(IntLit(42))
        assert result == SInt(42)

    def test_bool_lit(self):
        from stack_vm import BoolLit
        result = ast_to_sexpr(BoolLit(True))
        assert result == SBool(True)

    def test_var(self):
        from stack_vm import Var
        result = ast_to_sexpr(Var('x'))
        assert result == SVar('x')

    def test_binop(self):
        from stack_vm import BinOp, Var, IntLit
        node = BinOp('+', Var('x'), IntLit(1))
        result = ast_to_sexpr(node)
        assert result == SBinOp('+', SVar('x'), SInt(1))

    def test_unaryop(self):
        from stack_vm import UnaryOp, Var
        node = UnaryOp('-', Var('x'))
        result = ast_to_sexpr(node)
        assert result == SUnaryOp('-', SVar('x'))


# ============================================================
# Section 4: SMT lowering
# ============================================================

class TestSMTLowering:
    def test_lower_var(self):
        s = SMTSolver()
        term = lower_to_smt(s, SVar('x'))
        assert term is not None

    def test_lower_int(self):
        s = SMTSolver()
        term = lower_to_smt(s, SInt(42))
        assert term is not None

    def test_lower_bool(self):
        s = SMTSolver()
        term = lower_to_smt(s, SBool(True))
        assert term is not None

    def test_lower_binop_arithmetic(self):
        s = SMTSolver()
        expr = SBinOp('+', SVar('x'), SInt(1))
        term = lower_to_smt(s, expr)
        assert term is not None

    def test_lower_comparison(self):
        s = SMTSolver()
        expr = SBinOp('>', SVar('x'), SInt(0))
        term = lower_to_smt(s, expr)
        # Check it works in solver
        s.add(term)
        from smt_solver import SMTResult
        assert s.check() == SMTResult.SAT

    def test_lower_implies(self):
        s = SMTSolver()
        expr = SImplies(SBinOp('>', SVar('x'), SInt(0)),
                        SBinOp('>=', SVar('x'), SInt(1)))
        term = lower_to_smt(s, expr)
        # This is valid (x > 0 => x >= 1 for integers)
        s.add(s.Not(term))
        from smt_solver import SMTResult
        assert s.check() == SMTResult.UNSAT

    def test_lower_and_or(self):
        s = SMTSolver()
        expr = s_and(SBinOp('>', SVar('x'), SInt(0)),
                     SBinOp('<', SVar('x'), SInt(10)))
        term = lower_to_smt(s, expr)
        s.add(term)
        from smt_solver import SMTResult
        assert s.check() == SMTResult.SAT


# ============================================================
# Section 5: VC checking
# ============================================================

class TestVCChecking:
    def test_valid_vc(self):
        # x > 0 => x >= 1 (valid for integers)
        formula = SImplies(SBinOp('>', SVar('x'), SInt(0)),
                          SBinOp('>=', SVar('x'), SInt(1)))
        result = check_vc("test", formula)
        assert result.status == VCStatus.VALID

    def test_invalid_vc(self):
        # x > 0 => x > 5 (not valid)
        formula = SImplies(SBinOp('>', SVar('x'), SInt(0)),
                          SBinOp('>', SVar('x'), SInt(5)))
        result = check_vc("test", formula)
        assert result.status == VCStatus.INVALID
        assert result.counterexample is not None

    def test_tautology(self):
        # x == x (always true)
        formula = SBinOp('==', SVar('x'), SVar('x'))
        result = check_vc("test", formula)
        assert result.status == VCStatus.VALID

    def test_contradiction(self):
        # x > 0 AND x < 0 (never true => not valid as a VC)
        formula = s_and(SBinOp('>', SVar('x'), SInt(0)),
                       SBinOp('<', SVar('x'), SInt(0)))
        result = check_vc("test", formula)
        assert result.status == VCStatus.INVALID


# ============================================================
# Section 6: Simple function verification
# ============================================================

class TestSimpleFunctionVerification:
    def test_identity_function(self):
        """fn id(x) { requires(x > 0); ensures(result > 0); return x; }"""
        source = """
        fn id(x) {
            requires(x > 0);
            ensures(result > 0);
            return x;
        }
        """
        result = verify_function(source, 'id')
        assert result.verified is True
        assert result.total_vcs >= 1

    def test_increment_function(self):
        """fn inc(x) { requires(x >= 0); ensures(result > 0); return x + 1; }"""
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        """
        result = verify_function(source, 'inc')
        assert result.verified is True

    def test_failing_spec(self):
        """fn bad(x) { requires(x > 0); ensures(result > 10); return x; }"""
        source = """
        fn bad(x) {
            requires(x > 0);
            ensures(result > 10);
            return x;
        }
        """
        result = verify_function(source, 'bad')
        assert result.verified is False
        assert result.invalid_vcs >= 1

    def test_no_precondition(self):
        """fn abs_wrong(x) { ensures(result >= 0); return x; }  -- should fail"""
        source = """
        fn abs_wrong(x) {
            ensures(result >= 0);
            return x;
        }
        """
        result = verify_function(source, 'abs_wrong')
        assert result.verified is False

    def test_addition(self):
        """fn add(a, b) { requires(a > 0); requires(b > 0); ensures(result > 1); return a + b; }"""
        source = """
        fn add(a, b) {
            requires(a > 0);
            requires(b > 0);
            ensures(result > 1);
            return a + b;
        }
        """
        result = verify_function(source, 'add')
        assert result.verified is True


# ============================================================
# Section 7: Assignment and sequencing
# ============================================================

class TestAssignmentVerification:
    def test_let_then_return(self):
        """fn f(x) { requires(x > 0); ensures(result > 1); let y = x + 1; return y; }"""
        source = """
        fn f(x) {
            requires(x > 0);
            ensures(result > 1);
            let y = x + 1;
            return y;
        }
        """
        result = verify_function(source, 'f')
        assert result.verified is True

    def test_multiple_assignments(self):
        source = """
        fn f(x) {
            requires(x > 0);
            ensures(result > 2);
            let y = x + 1;
            let z = y + 1;
            return z;
        }
        """
        result = verify_function(source, 'f')
        assert result.verified is True

    def test_assignment_overwrite(self):
        source = """
        fn f(x) {
            requires(x > 0);
            ensures(result == 5);
            let y = x;
            y = 5;
            return y;
        }
        """
        result = verify_function(source, 'f')
        assert result.verified is True

    def test_swap_logic(self):
        """Verify a swap-like pattern."""
        source = """
        fn swap_sum(a, b) {
            requires(a > 0);
            requires(b > 0);
            ensures(result == a + b);
            let t = a;
            let s = t + b;
            return s;
        }
        """
        result = verify_function(source, 'swap_sum')
        assert result.verified is True


# ============================================================
# Section 8: Conditional verification
# ============================================================

class TestConditionalVerification:
    def test_if_then_else(self):
        """Absolute value function."""
        source = """
        fn abs(x) {
            requires(x != 0);
            ensures(result > 0);
            let r = 0;
            if (x > 0) {
                r = x;
            } else {
                r = 0 - x;
            }
            return r;
        }
        """
        result = verify_function(source, 'abs')
        assert result.verified is True

    def test_max_function(self):
        source = """
        fn max(a, b) {
            ensures(result >= a);
            ensures(result >= b);
            let r = 0;
            if (a >= b) {
                r = a;
            } else {
                r = b;
            }
            return r;
        }
        """
        result = verify_function(source, 'max')
        assert result.verified is True

    def test_if_no_else(self):
        source = """
        fn clamp_pos(x) {
            requires(x >= 0);
            ensures(result >= 0);
            let r = x;
            if (x < 0) {
                r = 0;
            }
            return r;
        }
        """
        result = verify_function(source, 'clamp_pos')
        assert result.verified is True

    def test_nested_if(self):
        source = """
        fn classify(x) {
            requires(x >= 0);
            requires(x <= 100);
            ensures(result >= 1);
            ensures(result <= 3);
            let r = 0;
            if (x < 30) {
                r = 1;
            } else {
                if (x < 70) {
                    r = 2;
                } else {
                    r = 3;
                }
            }
            return r;
        }
        """
        result = verify_function(source, 'classify')
        assert result.verified is True


# ============================================================
# Section 9: Loop verification with invariants
# ============================================================

class TestLoopVerification:
    def test_simple_countdown(self):
        source = """
        fn countdown(n) {
            requires(n >= 0);
            ensures(result == 0);
            let i = n;
            while (i > 0) {
                invariant(i >= 0);
                i = i - 1;
            }
            return i;
        }
        """
        result = verify_function(source, 'countdown')
        assert result.verified is True

    def test_accumulator(self):
        """Sum from 1 to n. Invariant: s == n*(n+1)/2 - i*(i+1)/2 ... simplified."""
        source = """
        fn sum_to(n) {
            requires(n >= 0);
            requires(n <= 10);
            ensures(result >= 0);
            let s = 0;
            let i = 0;
            while (i < n) {
                invariant(s >= 0);
                invariant(i >= 0);
                invariant(i <= n);
                s = s + i + 1;
                i = i + 1;
            }
            return s;
        }
        """
        result = verify_function(source, 'sum_to')
        assert result.verified is True

    def test_loop_invariant_failure(self):
        """Invariant that doesn't hold."""
        source = """
        fn bad_loop(n) {
            requires(n >= 0);
            ensures(result >= 0);
            let i = 0;
            while (i < n) {
                invariant(i < 5);
                i = i + 1;
            }
            return i;
        }
        """
        result = verify_function(source, 'bad_loop')
        # Invariant i < 5 is not preserved when n >= 5
        assert result.verified is False

    def test_loop_missing_invariant(self):
        """Loop without invariant should raise error."""
        source = """
        fn f(n) {
            requires(n >= 0);
            ensures(result >= 0);
            let i = 0;
            while (i < n) {
                i = i + 1;
            }
            return i;
        }
        """
        result = verify_function(source, 'f')
        assert not result.verified
        assert len(result.errors) > 0


# ============================================================
# Section 10: Hoare triple verification
# ============================================================

class TestHoareTriples:
    def test_assignment_triple(self):
        """{true} x = 5 {x == 5}"""
        result = verify_hoare_triple("true", "let x = 5;", "x == 5")
        assert result.verified is True

    def test_sequential_triple(self):
        """{x > 0} y = x + 1 {y > 1}"""
        result = verify_hoare_triple(
            "x > 0",
            "let y = x + 1;",
            "y > 1",
            var_types={'x': 'int'}
        )
        assert result.verified is True

    def test_conditional_triple(self):
        """{true} if (x > 0) r = 1 else r = 0 {r >= 0}"""
        result = verify_hoare_triple(
            "true",
            "let r = 0; if (x > 0) { r = 1; } else { r = 0; }",
            "r >= 0",
            var_types={'x': 'int'}
        )
        assert result.verified is True

    def test_failing_triple(self):
        """{x > 0} y = x - 2 {y > 0}  -- fails for x == 1"""
        result = verify_hoare_triple(
            "x > 0",
            "let y = x - 2;",
            "y > 0",
            var_types={'x': 'int'}
        )
        assert result.verified is False
        # Counterexample should have x = 1
        invalid_vcs = [vc for vc in result.vcs if vc.status == VCStatus.INVALID]
        assert len(invalid_vcs) >= 1


# ============================================================
# Section 11: Multiple VCs and complex programs
# ============================================================

class TestComplexPrograms:
    def test_function_with_assertion(self):
        source = """
        fn safe_div(a, b) {
            requires(b != 0);
            ensures(result * b == a);
            assert(b != 0);
            return a / b;
        }
        """
        # Note: integer division verification is tricky; just check it runs
        result = verify_function(source, 'safe_div')
        assert result.total_vcs >= 1

    def test_multiple_functions(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        fn dec(x) {
            requires(x > 1);
            ensures(result > 0);
            return x - 1;
        }
        """
        result = verify_program(source)
        assert result.verified is True
        assert result.total_vcs >= 2

    def test_no_annotations(self):
        """Functions without annotations should pass (nothing to verify)."""
        source = """
        fn f(x) {
            return x + 1;
        }
        """
        result = verify_function(source, 'f')
        assert result.verified is True
        assert result.total_vcs == 0


# ============================================================
# Section 12: Edge cases
# ============================================================

class TestEdgeCases:
    def test_parse_error(self):
        result = verify_function("fn {{{", 'f')
        assert result.verified is False
        assert len(result.errors) > 0

    def test_function_not_found(self):
        source = "fn f(x) { return x; }"
        result = verify_function(source, 'nonexistent')
        assert result.verified is False
        assert len(result.errors) > 0

    def test_trivial_true_postcondition(self):
        source = """
        fn f(x) {
            requires(x > 0);
            ensures(true);
            return x;
        }
        """
        result = verify_function(source, 'f')
        assert result.verified is True

    def test_constant_function(self):
        source = """
        fn zero() {
            ensures(result == 0);
            return 0;
        }
        """
        result = verify_function(source, 'zero')
        assert result.verified is True

    def test_print_stmt_ignored(self):
        source = """
        fn f(x) {
            requires(x > 0);
            ensures(result > 0);
            print(x);
            return x;
        }
        """
        result = verify_function(source, 'f')
        assert result.verified is True


# ============================================================
# Section 13: Counterexample quality
# ============================================================

class TestCounterexamples:
    def test_counterexample_has_values(self):
        source = """
        fn f(x) {
            requires(x > 0);
            ensures(result > 100);
            return x;
        }
        """
        result = verify_function(source, 'f')
        assert result.verified is False
        invalid = [vc for vc in result.vcs if vc.status == VCStatus.INVALID]
        assert len(invalid) >= 1
        # Counterexample should show x in [1, 100]
        ce = invalid[0].counterexample
        assert ce is not None
        assert 'x' in ce
        assert ce['x'] <= 100

    def test_counterexample_two_vars(self):
        source = """
        fn f(a, b) {
            requires(a > 0);
            requires(b > 0);
            ensures(result > 10);
            return a + b;
        }
        """
        result = verify_function(source, 'f')
        assert result.verified is False
        invalid = [vc for vc in result.vcs if vc.status == VCStatus.INVALID]
        ce = invalid[0].counterexample
        assert ce is not None


# ============================================================
# Section 14: Verify_program (top-level code)
# ============================================================

class TestProgramVerification:
    def test_top_level_with_assertion(self):
        source = """
        let x = 5;
        assert(x > 0);
        """
        result = verify_program(source)
        # The assertion VC should be valid since x=5 > 0
        assert result.total_vcs >= 1

    def test_program_mixed(self):
        source = """
        fn inc(x) {
            requires(x >= 0);
            ensures(result > 0);
            return x + 1;
        }
        let y = 10;
        """
        result = verify_program(source)
        assert result.verified is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
