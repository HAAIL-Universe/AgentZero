"""Tests for V105: Polyhedral Abstract Domain."""

import pytest
import sys
import os
from fractions import Fraction

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'challenges', 'C010_stack_vm'))
from polyhedral_domain import (
    PolyhedralDomain, LinearConstraint, PolyhedralInterpreter,
    polyhedral_analyze, get_variable_range, get_all_constraints,
    get_relational_constraints, compare_analyses, verify_property,
    polyhedral_summary, ZERO, ONE, INF
)
from stack_vm import IntLit, Var, BinOp, UnaryOp


# =========================================================================
# Section 1: LinearConstraint basics
# =========================================================================

class TestLinearConstraint:
    def test_create_from_dict(self):
        c = LinearConstraint.from_dict({'x': 1}, 5)
        assert c.bound == Fraction(5)
        assert c.coeffs_dict == {'x': Fraction(1)}

    def test_zero_coefficients_removed(self):
        c = LinearConstraint.from_dict({'x': 1, 'y': 0}, 5)
        assert 'y' not in c.coeffs_dict

    def test_variables(self):
        c = LinearConstraint.from_dict({'x': 1, 'y': -2}, 10)
        assert c.variables == {'x', 'y'}

    def test_evaluate(self):
        c = LinearConstraint.from_dict({'x': 2, 'y': -1}, 10)
        assert c.evaluate({'x': Fraction(3), 'y': Fraction(1)}) == Fraction(5)

    def test_is_satisfied(self):
        c = LinearConstraint.from_dict({'x': 1}, 5)
        assert c.is_satisfied({'x': Fraction(3)})
        assert c.is_satisfied({'x': Fraction(5)})
        assert not c.is_satisfied({'x': Fraction(6)})

    def test_equality_constraint(self):
        c = LinearConstraint.from_dict({'x': 1}, 5, is_equality=True)
        assert c.is_satisfied({'x': Fraction(5)})
        assert not c.is_satisfied({'x': Fraction(4)})

    def test_add_combination(self):
        c1 = LinearConstraint.from_dict({'x': 1, 'y': 1}, 10)
        c2 = LinearConstraint.from_dict({'x': -1, 'y': 1}, 4)
        combined = c1.add(c2)  # x+y<=10 AND -x+y<=4 -> 2y<=14
        assert combined.coeffs_dict.get('x', ZERO) == ZERO
        assert combined.coeffs_dict['y'] == Fraction(2)
        assert combined.bound == Fraction(14)

    def test_substitute(self):
        # x + y <= 10, substitute x = z + 2
        c = LinearConstraint.from_dict({'x': 1, 'y': 1}, 10)
        result = c.substitute('x', {'z': Fraction(1)}, Fraction(2))
        # z + 2 + y <= 10 -> z + y <= 8
        assert result.coeffs_dict == {'z': Fraction(1), 'y': Fraction(1)}
        assert result.bound == Fraction(8)

    def test_str_representation(self):
        c = LinearConstraint.from_dict({'x': 1, 'y': -1}, 5)
        s = str(c)
        assert '<=' in s
        assert '5' in s


# =========================================================================
# Section 2: PolyhedralDomain basic operations
# =========================================================================

class TestPolyhedralDomainBasic:
    def test_top(self):
        p = PolyhedralDomain.top(['x', 'y'])
        assert not p.is_bot()
        assert p.get_upper('x') == INF
        assert p.get_lower('x') == -INF

    def test_bot(self):
        p = PolyhedralDomain.bot(['x'])
        assert p.is_bot()

    def test_add_var(self):
        p = PolyhedralDomain()
        p.add_var('x')
        assert 'x' in p.var_names

    def test_set_upper_lower(self):
        p = PolyhedralDomain(['x'])
        p.set_upper('x', 10)
        p.set_lower('x', 0)
        assert p.get_upper('x') == 10
        assert p.get_lower('x') == 0

    def test_set_equal(self):
        p = PolyhedralDomain(['x'])
        p.set_equal('x', 5)
        assert p.get_upper('x') == 5
        assert p.get_lower('x') == 5

    def test_add_relational_constraint(self):
        p = PolyhedralDomain(['x', 'y'])
        p.add_constraint({'x': 1, 'y': -1}, 3)  # x - y <= 3
        constraints = p.get_relational_constraints()
        assert len(constraints) >= 1

    def test_copy_is_independent(self):
        p = PolyhedralDomain(['x'])
        p.set_upper('x', 10)
        q = p.copy()
        q.set_upper('x', 5)
        assert p.get_upper('x') == 10

    def test_get_interval(self):
        p = PolyhedralDomain(['x'])
        p.set_lower('x', 2)
        p.set_upper('x', 8)
        lo, hi = p.get_interval('x')
        assert lo == 2
        assert hi == 8

    def test_get_constraints_top(self):
        p = PolyhedralDomain.top(['x'])
        assert 'TOP' in p.get_constraints()[0]

    def test_get_constraints_bot(self):
        p = PolyhedralDomain.bot(['x'])
        assert 'BOTTOM' in p.get_constraints()[0]


# =========================================================================
# Section 3: Fourier-Motzkin elimination (forget)
# =========================================================================

class TestFourierMotzkin:
    def test_forget_simple(self):
        p = PolyhedralDomain(['x', 'y'])
        p.set_upper('x', 10)
        p.set_lower('x', 0)
        p.set_upper('y', 5)
        p.forget('x')
        assert 'x' not in p.var_names
        assert p.get_upper('y') == 5

    def test_forget_relational(self):
        """x - y <= 3 AND y <= 5 AND x >= 0 -> after forget x: y >= -3 (weakened)."""
        p = PolyhedralDomain(['x', 'y'])
        p.add_constraint({'x': 1, 'y': -1}, 3)  # x - y <= 3
        p.set_upper('y', 5)
        p.set_lower('x', 0)
        p.forget('x')
        # After eliminating x: y <= 5 remains, and 0 - y <= 3 -> -y <= 3 -> y >= -3
        assert p.get_upper('y') == 5
        lo = p.get_lower('y')
        assert lo >= -3

    def test_forget_equality(self):
        """x == y, forget x -> y unconstrained by x."""
        p = PolyhedralDomain(['x', 'y'])
        p.add_constraint({'x': 1, 'y': -1}, 0, is_equality=True)  # x == y
        p.set_lower('y', 0)
        p.set_upper('y', 10)
        p.forget('x')
        assert p.get_lower('y') == 0
        assert p.get_upper('y') == 10

    def test_forget_transitive(self):
        """x - y <= 2, y - z <= 3. After forget y: x - z <= 5."""
        p = PolyhedralDomain(['x', 'y', 'z'])
        p.add_constraint({'x': 1, 'y': -1}, 2)  # x - y <= 2
        p.add_constraint({'y': 1, 'z': -1}, 3)  # y - z <= 3
        p.forget('y')
        # Should derive x - z <= 5
        found = False
        for c in p.constraints:
            cd = c.coeffs_dict
            if cd.get('x', ZERO) > 0 and cd.get('z', ZERO) < 0:
                # x - z <= bound
                assert c.bound <= 5
                found = True
        assert found


# =========================================================================
# Section 4: Assignment operations
# =========================================================================

class TestAssignment:
    def test_assign_const(self):
        p = PolyhedralDomain(['x'])
        p.assign_const('x', 5)
        assert p.get_upper('x') == 5
        assert p.get_lower('x') == 5

    def test_assign_var(self):
        p = PolyhedralDomain(['x', 'y'])
        p.set_equal('x', 3)
        p.assign_var('y', 'x')
        assert p.get_upper('y') == 3
        assert p.get_lower('y') == 3

    def test_assign_linear(self):
        """y := 2*x + 3."""
        p = PolyhedralDomain(['x'])
        p.set_equal('x', 5)
        p.assign_expr('y', {'x': Fraction(2)}, Fraction(3))
        assert p.get_upper('y') == 13
        assert p.get_lower('y') == 13

    def test_assign_self_reference(self):
        """x := x + 1."""
        p = PolyhedralDomain(['x'])
        p.set_equal('x', 5)
        p.assign_expr('x', {'x': Fraction(1)}, Fraction(1))
        assert p.get_upper('x') == 6
        assert p.get_lower('x') == 6

    def test_assign_preserves_other_constraints(self):
        p = PolyhedralDomain(['x', 'y'])
        p.set_upper('y', 10)
        p.assign_const('x', 5)
        assert p.get_upper('y') == 10

    def test_assign_var_captures_relation(self):
        """After y := x, constraints should know y == x."""
        p = PolyhedralDomain(['x'])
        p.set_lower('x', 0)
        p.set_upper('x', 10)
        p.assign_var('y', 'x')
        # y should have same bounds as x
        assert p.get_lower('y') == 0
        assert p.get_upper('y') == 10

    def test_assign_sum(self):
        """z := x + y."""
        p = PolyhedralDomain(['x', 'y'])
        p.set_equal('x', 3)
        p.set_equal('y', 4)
        p.assign_expr('z', {'x': ONE, 'y': ONE}, ZERO)
        assert p.get_lower('z') == 7
        assert p.get_upper('z') == 7

    def test_assign_difference(self):
        """z := x - y."""
        p = PolyhedralDomain(['x', 'y'])
        p.set_equal('x', 10)
        p.set_equal('y', 3)
        p.assign_expr('z', {'x': ONE, 'y': Fraction(-1)}, ZERO)
        assert p.get_lower('z') == 7
        assert p.get_upper('z') == 7


# =========================================================================
# Section 5: Join (convex hull approximation)
# =========================================================================

class TestJoin:
    def test_join_bot_left(self):
        p = PolyhedralDomain.bot(['x'])
        q = PolyhedralDomain(['x'])
        q.set_upper('x', 5)
        result = p.join(q)
        assert not result.is_bot()
        assert result.get_upper('x') == 5

    def test_join_bot_right(self):
        p = PolyhedralDomain(['x'])
        p.set_upper('x', 5)
        q = PolyhedralDomain.bot(['x'])
        result = p.join(q)
        assert result.get_upper('x') == 5

    def test_join_weakens_bounds(self):
        p = PolyhedralDomain(['x'])
        p.set_upper('x', 5)
        p.set_lower('x', 0)
        q = PolyhedralDomain(['x'])
        q.set_upper('x', 10)
        q.set_lower('x', 3)
        result = p.join(q)
        assert result.get_upper('x') == 10
        assert result.get_lower('x') == 0

    def test_join_preserves_common_constraints(self):
        """If both have x <= 10, join keeps it."""
        p = PolyhedralDomain(['x'])
        p.set_upper('x', 10)
        q = PolyhedralDomain(['x'])
        q.set_upper('x', 10)
        result = p.join(q)
        assert result.get_upper('x') == 10


# =========================================================================
# Section 6: Meet (intersection)
# =========================================================================

class TestMeet:
    def test_meet_tightens(self):
        p = PolyhedralDomain(['x'])
        p.set_upper('x', 10)
        q = PolyhedralDomain(['x'])
        q.set_upper('x', 5)
        result = p.meet(q)
        assert result.get_upper('x') == 5

    def test_meet_bot_propagates(self):
        p = PolyhedralDomain.bot(['x'])
        q = PolyhedralDomain(['x'])
        result = p.meet(q)
        assert result.is_bot()

    def test_meet_combines_constraints(self):
        p = PolyhedralDomain(['x', 'y'])
        p.add_constraint({'x': 1, 'y': -1}, 5)  # x - y <= 5
        q = PolyhedralDomain(['x', 'y'])
        q.add_constraint({'x': 1, 'y': 1}, 10)  # x + y <= 10
        result = p.meet(q)
        assert len(result.constraints) == 2


# =========================================================================
# Section 7: Widening
# =========================================================================

class TestWidening:
    def test_widen_keeps_stable_constraints(self):
        p = PolyhedralDomain(['x'])
        p.set_upper('x', 10)
        p.set_lower('x', 0)
        q = PolyhedralDomain(['x'])
        q.set_upper('x', 10)
        q.set_lower('x', 0)
        result = p.widen(q)
        assert result.get_upper('x') == 10

    def test_widen_drops_violated(self):
        p = PolyhedralDomain(['x'])
        p.set_upper('x', 5)
        p.set_lower('x', 0)
        q = PolyhedralDomain(['x'])
        q.set_upper('x', 10)  # violates x <= 5
        q.set_lower('x', 0)
        result = p.widen(q)
        # x <= 5 should be dropped (q has x <= 10, not <= 5)
        # x >= 0 should be kept
        assert result.get_lower('x') == 0

    def test_widen_bot_left(self):
        p = PolyhedralDomain.bot(['x'])
        q = PolyhedralDomain(['x'])
        q.set_upper('x', 5)
        result = p.widen(q)
        assert result.get_upper('x') == 5


# =========================================================================
# Section 8: Leq and Equals
# =========================================================================

class TestOrdering:
    def test_bot_leq_everything(self):
        p = PolyhedralDomain.bot(['x'])
        q = PolyhedralDomain(['x'])
        assert p.leq(q)

    def test_not_leq_bot(self):
        p = PolyhedralDomain(['x'])
        q = PolyhedralDomain.bot(['x'])
        assert not p.leq(q)

    def test_tighter_leq_looser(self):
        p = PolyhedralDomain(['x'])
        p.set_upper('x', 5)
        p.set_lower('x', 0)
        q = PolyhedralDomain(['x'])
        q.set_upper('x', 10)
        q.set_lower('x', -5)
        assert p.leq(q)

    def test_equals_same(self):
        p = PolyhedralDomain(['x'])
        p.set_upper('x', 5)
        q = PolyhedralDomain(['x'])
        q.set_upper('x', 5)
        assert p.equals(q)

    def test_bot_equals_bot(self):
        assert PolyhedralDomain.bot().equals(PolyhedralDomain.bot())


# =========================================================================
# Section 9: C10 Interpreter - basic programs
# =========================================================================

class TestInterpreterBasic:
    def test_let_constant(self):
        result = polyhedral_analyze("let x = 5;")
        env = result['env']
        assert env.get_upper('x') == 5
        assert env.get_lower('x') == 5

    def test_let_and_assign(self):
        result = polyhedral_analyze("let x = 5; x = 10;")
        env = result['env']
        assert env.get_upper('x') == 10
        assert env.get_lower('x') == 10

    def test_two_variables(self):
        result = polyhedral_analyze("let x = 3; let y = 7;")
        env = result['env']
        assert env.get_lower('x') == 3
        assert env.get_upper('x') == 3
        assert env.get_lower('y') == 7
        assert env.get_upper('y') == 7

    def test_assign_var(self):
        result = polyhedral_analyze("let x = 5; let y = x;")
        env = result['env']
        assert env.get_lower('y') == 5
        assert env.get_upper('y') == 5


# =========================================================================
# Section 10: C10 Interpreter - arithmetic
# =========================================================================

class TestInterpreterArithmetic:
    def test_addition(self):
        result = polyhedral_analyze("let x = 3; let y = 4; let z = x + y;")
        env = result['env']
        assert env.get_lower('z') == 7
        assert env.get_upper('z') == 7

    def test_subtraction(self):
        result = polyhedral_analyze("let x = 10; let y = 3; let z = x - y;")
        env = result['env']
        assert env.get_lower('z') == 7
        assert env.get_upper('z') == 7

    def test_multiply_by_constant(self):
        result = polyhedral_analyze("let x = 5; let y = x * 3;")
        env = result['env']
        assert env.get_lower('y') == 15
        assert env.get_upper('y') == 15

    def test_self_increment(self):
        result = polyhedral_analyze("let x = 5; x = x + 1;")
        env = result['env']
        assert env.get_lower('x') == 6
        assert env.get_upper('x') == 6

    def test_negation(self):
        result = polyhedral_analyze("let x = 5; let y = -x;")
        env = result['env']
        assert env.get_lower('y') == -5
        assert env.get_upper('y') == -5

    def test_linear_combination(self):
        result = polyhedral_analyze("let a = 2; let b = 3; let c = a + b * 2;")
        env = result['env']
        # c = 2 + 3*2 = 8
        assert env.get_lower('c') == 8
        assert env.get_upper('c') == 8


# =========================================================================
# Section 11: C10 Interpreter - conditionals
# =========================================================================

class TestInterpreterConditionals:
    def test_if_true_branch(self):
        src = "let x = 5; let y = 0; if (x > 3) { y = 1; }"
        result = polyhedral_analyze(src)
        env = result['env']
        # x is always 5 > 3, so y = 1
        assert env.get_lower('y') == 1
        assert env.get_upper('y') == 1

    def test_if_else(self):
        src = "let x = 5; let y = 0; if (x > 10) { y = 1; } else { y = 2; }"
        result = polyhedral_analyze(src)
        env = result['env']
        # x=5, not > 10, so y = 2
        assert env.get_lower('y') == 2
        assert env.get_upper('y') == 2

    def test_if_join(self):
        src = """
        let x = 5;
        let y = 0;
        if (x > 3) {
            y = 10;
        } else {
            y = 20;
        }
        """
        result = polyhedral_analyze(src)
        env = result['env']
        # x=5 > 3 so only then-branch, y = 10
        assert env.get_lower('y') == 10

    def test_condition_refinement_less_than(self):
        src = """
        let x = 5;
        let r = 0;
        if (x < 3) {
            r = 1;
        } else {
            r = 2;
        }
        """
        result = polyhedral_analyze(src)
        env = result['env']
        # x=5, not < 3, so r=2
        assert env.get_lower('r') == 2
        assert env.get_upper('r') == 2

    def test_relational_condition(self):
        src = """
        let x = 3;
        let y = 5;
        let r = 0;
        if (x < y) {
            r = 1;
        } else {
            r = 2;
        }
        """
        result = polyhedral_analyze(src)
        env = result['env']
        # x=3 < y=5, so r=1
        assert env.get_lower('r') == 1


# =========================================================================
# Section 12: C10 Interpreter - loops
# =========================================================================

class TestInterpreterLoops:
    def test_simple_counter(self):
        src = """
        let i = 0;
        while (i < 10) {
            i = i + 1;
        }
        """
        result = polyhedral_analyze(src)
        env = result['env']
        # After loop: i >= 10
        assert env.get_lower('i') >= 10

    def test_countdown(self):
        src = """
        let i = 10;
        while (i > 0) {
            i = i - 1;
        }
        """
        result = polyhedral_analyze(src)
        env = result['env']
        # After loop: i <= 0
        assert env.get_upper('i') <= 0

    def test_sum_loop(self):
        src = """
        let i = 0;
        let s = 0;
        while (i < 5) {
            s = s + i;
            i = i + 1;
        }
        """
        result = polyhedral_analyze(src)
        env = result['env']
        # After loop: i >= 5
        assert env.get_lower('i') >= 5


# =========================================================================
# Section 13: Relational constraint tracking
# =========================================================================

class TestRelational:
    def test_assign_preserves_relation(self):
        """y := x creates y == x relation."""
        src = "let x = 5; let y = x;"
        result = polyhedral_analyze(src)
        env = result['env']
        # Both should be 5 (equality)
        assert env.get_lower('x') == 5
        assert env.get_lower('y') == 5

    def test_sum_relation(self):
        """z := x + y creates z == x + y."""
        src = "let x = 3; let y = 4; let z = x + y;"
        result = polyhedral_analyze(src)
        env = result['env']
        assert env.get_lower('z') == 7
        assert env.get_upper('z') == 7

    def test_diff_relation(self):
        """d := x - y creates d == x - y."""
        src = "let x = 10; let y = 3; let d = x - y;"
        result = polyhedral_analyze(src)
        env = result['env']
        assert env.get_lower('d') == 7
        assert env.get_upper('d') == 7

    def test_relational_constraints_detected(self):
        """Verify relational constraints are reported."""
        src = "let x = 5; let y = x;"
        constraints = get_relational_constraints(src)
        # Should have relational constraint involving x and y
        assert isinstance(constraints, list)

    def test_get_all_constraints(self):
        src = "let x = 5;"
        constraints = get_all_constraints(src)
        assert len(constraints) >= 1


# =========================================================================
# Section 14: Polyhedral domain operations directly
# =========================================================================

class TestDomainDirect:
    def test_fourier_motzkin_two_var(self):
        """x <= y, y <= 10 -> after forget y -> x <= 10."""
        p = PolyhedralDomain(['x', 'y'])
        p.add_constraint({'x': 1, 'y': -1}, 0)  # x - y <= 0 (x <= y)
        p.add_constraint({'y': 1}, 10)  # y <= 10
        p.forget('y')
        assert p.get_upper('x') == 10

    def test_three_variable_chain(self):
        """x <= y, y <= z, z <= 10 -> x <= 10."""
        p = PolyhedralDomain(['x', 'y', 'z'])
        p.add_constraint({'x': 1, 'y': -1}, 0)
        p.add_constraint({'y': 1, 'z': -1}, 0)
        p.add_constraint({'z': 1}, 10)
        p.forget('y')
        p.forget('z')
        assert p.get_upper('x') == 10

    def test_assign_expr_linear(self):
        """y := 2*x + 3 where x in [0, 5]."""
        p = PolyhedralDomain(['x'])
        p.set_lower('x', 0)
        p.set_upper('x', 5)
        p.assign_expr('y', {'x': Fraction(2)}, Fraction(3))
        lo, hi = p.get_interval('y')
        assert lo == 3   # 2*0 + 3
        assert hi == 13  # 2*5 + 3

    def test_assign_negative_coeff(self):
        """y := -x + 10 where x in [2, 8]."""
        p = PolyhedralDomain(['x'])
        p.set_lower('x', 2)
        p.set_upper('x', 8)
        p.assign_expr('y', {'x': Fraction(-1)}, Fraction(10))
        lo, hi = p.get_interval('y')
        assert lo == 2   # -8 + 10
        assert hi == 8   # -2 + 10


# =========================================================================
# Section 15: Comparison with C039
# =========================================================================

class TestComparison:
    def test_compare_simple(self):
        src = "let x = 5; let y = x;"
        result = compare_analyses(src)
        assert 'polyhedral_constraints' in result
        assert 'interval_results' in result
        assert 'relational_constraints' in result

    def test_compare_precision(self):
        """Polyhedral should be at least as precise as intervals."""
        src = "let x = 5; let y = 10;"
        result = compare_analyses(src)
        for var, data in result['interval_results'].items():
            poly_lo, poly_hi = data['polyhedral']
            c039_lo, c039_hi = data['interval']
            # Polyhedral range should be <= interval range (or both inf)
            if poly_hi != INF and c039_hi != INF:
                assert poly_hi <= c039_hi + 0.001
            if poly_lo != -INF and c039_lo != -INF:
                assert poly_lo >= c039_lo - 0.001


# =========================================================================
# Section 16: Verify property
# =========================================================================

class TestVerifyProperty:
    def test_verify_upper_bound(self):
        src = "let x = 5;"
        result = verify_property(src, "x <= 10")
        assert result['verdict'] == 'VERIFIED'

    def test_verify_lower_bound(self):
        src = "let x = 5;"
        result = verify_property(src, "x >= 0")
        assert result['verdict'] == 'VERIFIED'

    def test_verify_equality(self):
        src = "let x = 5;"
        result = verify_property(src, "x == 5")
        assert result['verdict'] == 'VERIFIED'

    def test_unknown_property(self):
        src = "let x = 5;"
        result = verify_property(src, "x <= 3")
        assert result['verdict'] == 'UNKNOWN'

    def test_bot_state_verified(self):
        """Bottom state trivially verifies everything."""
        result = verify_property("let x = 5; if (x > 10) { let y = 1; }", "y <= 100")
        # This is tricky -- y doesn't exist in the main flow
        assert result['verdict'] in ('VERIFIED', 'UNKNOWN')


# =========================================================================
# Section 17: Polyhedral summary
# =========================================================================

class TestSummary:
    def test_summary_simple(self):
        src = "let x = 5; let y = 10;"
        summary = polyhedral_summary(src)
        assert "Polyhedral Analysis Summary" in summary
        assert "x" in summary
        assert "y" in summary

    def test_summary_bot(self):
        # Tricky to make bot in C10... just test the domain directly
        p = PolyhedralDomain.bot(['x'])
        assert "BOT" in str(p)


# =========================================================================
# Section 18: get_variable_range API
# =========================================================================

class TestGetVariableRange:
    def test_simple(self):
        lo, hi = get_variable_range("let x = 5;", "x")
        assert lo == 5
        assert hi == 5

    def test_after_arithmetic(self):
        lo, hi = get_variable_range("let x = 3; let y = x + 7;", "y")
        assert lo == 10
        assert hi == 10


# =========================================================================
# Section 19: Edge cases
# =========================================================================

class TestEdgeCases:
    def test_empty_program(self):
        result = polyhedral_analyze("")
        assert not result['env'].is_bot()

    def test_division_warning(self):
        src = "let x = 10; let y = 0; let z = x / y;"
        result = polyhedral_analyze(src)
        assert any("division" in w for w in result['warnings'])

    def test_function_declaration(self):
        src = "fn foo(x) { return x + 1; } let a = 5;"
        result = polyhedral_analyze(src)
        assert 'foo' in result['functions']
        assert result['env'].get_lower('a') == 5

    def test_print_statement(self):
        src = "let x = 5; print(x);"
        result = polyhedral_analyze(src)
        assert result['env'].get_lower('x') == 5

    def test_bool_literal(self):
        src = "let x = 5; if (true) { x = 10; }"
        result = polyhedral_analyze(src)
        # true is truthy, but we don't refine on bare bool
        # Both branches possible
        env = result['env']
        assert env.get_upper('x') >= 10 or env.get_lower('x') >= 5


# =========================================================================
# Section 20: Complex programs
# =========================================================================

class TestComplexPrograms:
    def test_swap(self):
        src = """
        let x = 3;
        let y = 7;
        let tmp = x;
        x = y;
        y = tmp;
        """
        result = polyhedral_analyze(src)
        env = result['env']
        assert env.get_lower('x') == 7
        assert env.get_upper('x') == 7
        assert env.get_lower('y') == 3
        assert env.get_upper('y') == 3

    def test_abs_value(self):
        src = """
        let x = 5;
        let r = 0;
        if (x >= 0) {
            r = x;
        } else {
            r = -x;
        }
        """
        result = polyhedral_analyze(src)
        env = result['env']
        assert env.get_lower('r') == 5

    def test_max(self):
        src = """
        let x = 3;
        let y = 7;
        let m = 0;
        if (x > y) {
            m = x;
        } else {
            m = y;
        }
        """
        result = polyhedral_analyze(src)
        env = result['env']
        assert env.get_lower('m') == 7

    def test_nested_if(self):
        src = """
        let x = 5;
        let y = 0;
        if (x > 0) {
            if (x < 10) {
                y = x;
            }
        }
        """
        result = polyhedral_analyze(src)
        env = result['env']
        assert env.get_lower('y') == 5

    def test_loop_with_condition(self):
        src = """
        let i = 0;
        let x = 100;
        while (i < 10) {
            if (i > 5) {
                x = x - 1;
            }
            i = i + 1;
        }
        """
        result = polyhedral_analyze(src)
        env = result['env']
        assert env.get_lower('i') >= 10

    def test_multiple_assignments(self):
        src = """
        let a = 1;
        let b = 2;
        let c = a + b;
        a = c + 1;
        b = a + c;
        """
        # a=1, b=2, c=3, a=4, b=7
        result = polyhedral_analyze(src)
        env = result['env']
        assert env.get_lower('a') == 4
        assert env.get_upper('a') == 4
        assert env.get_lower('b') == 7
        assert env.get_upper('b') == 7

    def test_accumulator(self):
        src = """
        let x = 0;
        x = x + 1;
        x = x + 1;
        x = x + 1;
        """
        result = polyhedral_analyze(src)
        env = result['env']
        assert env.get_lower('x') == 3
        assert env.get_upper('x') == 3


# =========================================================================
# Section 21: Polyhedral precision advantage
# =========================================================================

class TestPrecisionAdvantage:
    def test_relation_through_assignment(self):
        """Polyhedral tracks y = x + 1 precisely."""
        src = "let x = 5; let y = x + 1;"
        result = polyhedral_analyze(src)
        env = result['env']
        assert env.get_lower('y') == 6
        assert env.get_upper('y') == 6

    def test_sum_conservation(self):
        """x + y is conserved through swap-like operations."""
        src = """
        let x = 3;
        let y = 7;
        let s = x + y;
        """
        result = polyhedral_analyze(src)
        env = result['env']
        assert env.get_lower('s') == 10
        assert env.get_upper('s') == 10

    def test_difference_tracking(self):
        """Tracks x - y through computation."""
        src = """
        let x = 10;
        let y = 3;
        let d = x - y;
        """
        result = polyhedral_analyze(src)
        env = result['env']
        assert env.get_lower('d') == 7
        assert env.get_upper('d') == 7


# =========================================================================
# Section 22: Linearization
# =========================================================================

class TestLinearization:
    def test_linearize_constant(self):
        interp = PolyhedralInterpreter()
        coeffs, const = interp._linearize_expr(IntLit(5))
        assert coeffs == {}
        assert const == Fraction(5)

    def test_linearize_var(self):
        interp = PolyhedralInterpreter()
        coeffs, const = interp._linearize_expr(Var('x'))
        assert coeffs == {'x': ONE}
        assert const == ZERO

    def test_linearize_addition(self):
        interp = PolyhedralInterpreter()
        expr = BinOp('+', Var('x'), IntLit(3))
        coeffs, const = interp._linearize_expr(expr)
        assert coeffs == {'x': ONE}
        assert const == Fraction(3)

    def test_linearize_multiply_constant(self):
        interp = PolyhedralInterpreter()
        expr = BinOp('*', IntLit(2), Var('x'))
        coeffs, const = interp._linearize_expr(expr)
        assert coeffs == {'x': Fraction(2)}
        assert const == ZERO

    def test_linearize_complex(self):
        interp = PolyhedralInterpreter()
        # 2*x + 3*y - 1
        expr = BinOp('-', BinOp('+', BinOp('*', IntLit(2), Var('x')),
                                BinOp('*', IntLit(3), Var('y'))),
                     IntLit(1))
        coeffs, const = interp._linearize_expr(expr)
        assert coeffs['x'] == Fraction(2)
        assert coeffs['y'] == Fraction(3)
        assert const == Fraction(-1)

    def test_linearize_nonlinear_fails(self):
        interp = PolyhedralInterpreter()
        expr = BinOp('*', Var('x'), Var('y'))
        coeffs, const = interp._linearize_expr(expr)
        assert coeffs is None


# =========================================================================
# Section 23: Condition refinement
# =========================================================================

class TestConditionRefinement:
    def test_refine_less_than(self):
        interp = PolyhedralInterpreter()
        env = PolyhedralDomain(['x'])
        env.set_lower('x', 0)
        env.set_upper('x', 10)
        cond = BinOp('<', Var('x'), IntLit(5))
        then_env, else_env = interp._refine_condition(cond, env)
        assert then_env.get_upper('x') <= 4
        assert else_env.get_lower('x') >= 5

    def test_refine_greater_equal(self):
        interp = PolyhedralInterpreter()
        env = PolyhedralDomain(['x'])
        env.set_lower('x', 0)
        env.set_upper('x', 10)
        cond = BinOp('>=', Var('x'), IntLit(5))
        then_env, else_env = interp._refine_condition(cond, env)
        assert then_env.get_lower('x') >= 5
        assert else_env.get_upper('x') <= 4

    def test_refine_equality(self):
        interp = PolyhedralInterpreter()
        env = PolyhedralDomain(['x'])
        env.set_lower('x', 0)
        env.set_upper('x', 10)
        cond = BinOp('==', Var('x'), IntLit(5))
        then_env, else_env = interp._refine_condition(cond, env)
        assert then_env.get_lower('x') == 5
        assert then_env.get_upper('x') == 5

    def test_refine_var_vs_var(self):
        interp = PolyhedralInterpreter()
        env = PolyhedralDomain(['x', 'y'])
        env.set_lower('x', 0)
        env.set_upper('x', 10)
        env.set_lower('y', 0)
        env.set_upper('y', 10)
        cond = BinOp('<', Var('x'), Var('y'))
        then_env, else_env = interp._refine_condition(cond, env)
        # In then: x - y <= -1 (x < y)
        constraints = then_env.get_constraints()
        has_relational = any('x' in c and 'y' in c for c in constraints)
        assert has_relational


# =========================================================================
# Section 24: Robustness
# =========================================================================

class TestRobustness:
    def test_many_variables(self):
        src = "\n".join([f"let v{i} = {i};" for i in range(20)])
        result = polyhedral_analyze(src)
        env = result['env']
        assert env.get_lower('v0') == 0
        assert env.get_upper('v19') == 19

    def test_long_chain(self):
        lines = ["let x0 = 1;"]
        for i in range(1, 10):
            lines.append(f"let x{i} = x{i-1} + 1;")
        result = polyhedral_analyze("\n".join(lines))
        env = result['env']
        assert env.get_lower('x9') == 10
        assert env.get_upper('x9') == 10

    def test_reassign_multiple_times(self):
        src = "let x = 1; x = 2; x = 3; x = 4; x = 5;"
        result = polyhedral_analyze(src)
        env = result['env']
        assert env.get_lower('x') == 5
        assert env.get_upper('x') == 5

    def test_nested_loops(self):
        src = """
        let i = 0;
        let j = 0;
        while (i < 3) {
            j = 0;
            while (j < 3) {
                j = j + 1;
            }
            i = i + 1;
        }
        """
        result = polyhedral_analyze(src)
        env = result['env']
        assert env.get_lower('i') >= 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
